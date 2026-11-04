"""Conservative trigger-run payload retention tests.

Default snapshots keep only a payload hash and allow-listed metadata; the
original payload is stored encrypted only when the trigger opts in, and can
be read back only by the owner through an audited include_payload request.
"""

from __future__ import annotations

import hashlib
import json
import time
from unittest.mock import MagicMock, patch

import pytest

from xagent.web.models.trigger import TriggerAudit, TriggerRun
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.services.trigger_providers import sign_webhook_payload

from .conftest import (
    _admin_headers,
    _direct_db_session,
    _register_second_user,
    client,
)

pytestmark = pytest.mark.usefixtures("_test_db")


@pytest.fixture(autouse=True)
def mock_bg_scheduler():
    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ) as mocked:
        yield mocked


def _create_agent(headers: dict[str, str]) -> int:
    resp = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": "Payload Agent",
            "description": "test",
            "instructions": "You are a payload retention test agent.",
            "execution_mode": "balanced",
        },
    )
    assert resp.status_code == 200, resp.text
    return int(resp.json()["id"])


def _create_webhook_trigger(
    headers: dict[str, str],
    agent_id: int,
    *,
    store_full_payload: bool = False,
) -> dict:
    config = {"store_full_payload": True} if store_full_payload else {}
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={"type": "webhook", "name": "Payload webhook", "config": config},
    )
    assert created.status_code == 200, created.text
    return created.json()


def _fire_webhook(trigger: dict, payload: dict, event_id: str = "evt-1") -> int:
    raw_body = json.dumps(payload).encode("utf-8")
    timestamp = str(int(time.time()))
    fired = client.post(
        f"/api/triggers/callback/webhook/{trigger['callback_id']}",
        headers={
            "x-xagent-signature": sign_webhook_payload(
                trigger["webhook_secret"], timestamp, raw_body
            ),
            "x-xagent-timestamp": timestamp,
            "x-xagent-event-id": event_id,
        },
        content=raw_body,
    )
    assert fired.status_code == 200, fired.text
    run_ids = fired.json()["trigger_run_ids"]
    assert len(run_ids) == 1
    return int(run_ids[0])


SENSITIVE_PAYLOAD = {
    "from": "boss@company.com",
    "subject": "confidential merger",
    "snippet": "do not leak",
    "body": "full text",
    "headers": {"X-Secret": "token"},
}


def test_default_snapshot_is_hash_plus_allow_listed_metadata() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    trigger = _create_webhook_trigger(headers, agent_id)
    run_id = _fire_webhook(trigger, SENSITIVE_PAYLOAD)

    db = _direct_db_session()
    try:
        run = db.query(TriggerRun).filter(TriggerRun.id == run_id).one()
        snapshot = run.payload_snapshot
        assert set(snapshot) == {"payload_sha256", "metadata"}
        expected_hash = hashlib.sha256(
            json.dumps(
                SENSITIVE_PAYLOAD, ensure_ascii=False, sort_keys=True, default=str
            ).encode("utf-8")
        ).hexdigest()
        assert snapshot["payload_sha256"] == expected_hash
        assert set(snapshot["metadata"]) == {
            "source_event_id",
            "event_type",
            "resource_id",
            "received_at",
        }
        assert snapshot["metadata"]["source_event_id"] == "evt-1"
        assert snapshot["metadata"]["event_type"] == "webhook"
        serialized = str(snapshot)
        for sensitive in (
            "boss@company.com",
            "confidential",
            "do not leak",
            "X-Secret",
        ):
            assert sensitive not in serialized
    finally:
        db.close()


def test_opt_in_full_payload_is_stored_encrypted_only() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    trigger = _create_webhook_trigger(headers, agent_id, store_full_payload=True)
    run_id = _fire_webhook(trigger, SENSITIVE_PAYLOAD)

    db = _direct_db_session()
    try:
        run = db.query(TriggerRun).filter(TriggerRun.id == run_id).one()
        snapshot = run.payload_snapshot
        assert set(snapshot) == {"payload_sha256", "metadata", "encrypted_payload"}
        assert "boss@company.com" not in str(snapshot)
        assert "confidential" not in str(snapshot)
    finally:
        db.close()


def test_owner_include_payload_read_returns_decrypted_payload_and_audits() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    trigger = _create_webhook_trigger(headers, agent_id, store_full_payload=True)
    run_id = _fire_webhook(trigger, SENSITIVE_PAYLOAD)

    url = f"/api/agents/{agent_id}/triggers/{trigger['id']}/runs/{run_id}"

    plain = client.get(url, headers=headers)
    assert plain.status_code == 200, plain.text
    assert plain.json()["payload"] is None
    assert plain.json()["payload_stored"] is True
    assert "encrypted_payload" not in (plain.json()["payload_snapshot"] or {})

    included = client.get(url, headers=headers, params={"include_payload": "true"})
    assert included.status_code == 200, included.text
    assert included.json()["payload"] == SENSITIVE_PAYLOAD

    db = _direct_db_session()
    try:
        audits = (
            db.query(TriggerAudit).filter(TriggerAudit.outcome == "payload_read").all()
        )
        assert len(audits) == 1
        assert audits[0].trigger_id == trigger["id"]
        assert audits[0].detail["trigger_run_id"] == run_id
        assert audits[0].detail["success"] is True
    finally:
        db.close()


def test_failed_payload_read_leaves_an_audit_trail() -> None:
    """A stored payload that can no longer be decrypted (e.g. after key
    rotation) still records a PAYLOAD_READ audit row with a failure outcome,
    and the error keeps propagating to the caller."""
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    trigger = _create_webhook_trigger(headers, agent_id, store_full_payload=True)
    run_id = _fire_webhook(trigger, SENSITIVE_PAYLOAD)

    db = _direct_db_session()
    try:
        run = db.query(TriggerRun).filter(TriggerRun.id == run_id).one()
        snapshot = dict(run.payload_snapshot)
        snapshot["encrypted_payload"] = "not-decryptable-after-key-rotation"
        run.payload_snapshot = snapshot
        db.add(run)
        db.commit()
    finally:
        db.close()

    url = f"/api/agents/{agent_id}/triggers/{trigger['id']}/runs/{run_id}"
    response = client.get(url, headers=headers, params={"include_payload": "true"})
    assert response.status_code in (400, 500)

    db = _direct_db_session()
    try:
        audits = (
            db.query(TriggerAudit).filter(TriggerAudit.outcome == "payload_read").all()
        )
        assert len(audits) == 1
        assert audits[0].trigger_id == trigger["id"]
        assert audits[0].detail["trigger_run_id"] == run_id
        assert audits[0].detail["success"] is False
        assert audits[0].detail["error"]
    finally:
        db.close()


def test_include_payload_without_stored_payload_is_a_validation_error() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    trigger = _create_webhook_trigger(headers, agent_id)
    run_id = _fire_webhook(trigger, SENSITIVE_PAYLOAD)

    url = f"/api/agents/{agent_id}/triggers/{trigger['id']}/runs/{run_id}"
    response = client.get(url, headers=headers, params={"include_payload": "true"})
    assert response.status_code == 400
    assert "not enabled" in response.json()["detail"].lower()

    db = _direct_db_session()
    try:
        assert (
            db.query(TriggerAudit)
            .filter(TriggerAudit.outcome == "payload_read")
            .count()
            == 0
        )
    finally:
        db.close()


def test_include_payload_read_is_owner_only() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    trigger = _create_webhook_trigger(headers, agent_id, store_full_payload=True)
    run_id = _fire_webhook(trigger, SENSITIVE_PAYLOAD)

    other_headers = _register_second_user()
    url = f"/api/agents/{agent_id}/triggers/{trigger['id']}/runs/{run_id}"
    response = client.get(
        url, headers=other_headers, params={"include_payload": "true"}
    )
    assert response.status_code == 404


def test_gmail_full_payload_opt_in_uses_same_retention_path() -> None:
    """Gmail runs share the retention behavior once configs opt in."""
    headers = _admin_headers()
    agent_id = _create_agent(headers)

    db = _direct_db_session()
    try:
        from xagent.web.models.user import User

        admin = db.query(User).filter(User.username == "admin").one()
        account = UserOAuth(
            user_id=int(admin.id),
            provider="gmail",
            access_token="token",
            email="owner@gmail.example",
        )
        db.add(account)
        db.commit()
        db.refresh(account)
        account_id = int(account.id)
    finally:
        db.close()

    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "gmail",
            "name": "Gmail payload",
            "config": {
                "watch_label": "INBOX",
                "oauth_account_id": account_id,
                "store_full_payload": True,
            },
        },
    )
    assert created.status_code == 200, created.text
