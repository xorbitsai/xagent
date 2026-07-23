from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from xagent.core.tools.adapters.vibe.config import (
    MCPUnavailableSummary,
    RequiredMCPUnavailableError,
)
from xagent.core.utils.encryption import decrypt_value
from xagent.web.models.agent import Agent, AgentOrigin
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.task import Task, TaskConnectorRuntimeContext, TaskStatus
from xagent.web.models.trigger import (
    AgentTrigger,
    TriggerAudit,
    TriggerProvisioningStatus,
    TriggerRun,
    TriggerRunStatus,
)
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.services.agent_team_scope import (
    AgentTeamScope,
    set_agent_team_scope_hook,
)
from xagent.web.services.connector_runtime import (
    ConnectorRuntimeValues,
    load_connector_runtime_view,
    set_connector_runtime_resolver_for_testing,
)
from xagent.web.services.task_orchestrator import (
    TurnStarted,
)
from xagent.web.services.task_orchestrator import _schedule_bg as _real_schedule_bg
from xagent.web.services.task_orchestrator import (
    finish_turn,
)
from xagent.web.services.trigger_providers import sign_webhook_payload
from xagent.web.services.triggers import (
    _compute_next_run_at,
    _start_prepared_trigger_run_id,
    dispatch_pending_trigger_runs,
    scan_due_scheduled_triggers,
)

from .conftest import (
    _admin_headers,
    _direct_db_session,
    _register_second_user,
    app_for_tests,
    client,
)

pytestmark = pytest.mark.usefixtures("_test_db")


@pytest.fixture(autouse=True)
def reset_connector_runtime_resolver():
    set_connector_runtime_resolver_for_testing(None)
    yield
    set_connector_runtime_resolver_for_testing(None)


@pytest.fixture(autouse=True)
def mock_bg_scheduler():
    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ) as mocked:
        yield mocked


def _create_agent(headers: dict[str, str], name: str = "Trigger Agent") -> int:
    resp = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": name,
            "description": "test",
            "instructions": "You are a trigger test agent.",
            "execution_mode": "balanced",
        },
    )
    assert resp.status_code == 200, resp.text
    return int(resp.json()["id"])


def _connect_gmail_account(
    username: str = "admin",
    *,
    email: str = "owner@gmail.example",
    provider: str = "gmail",
) -> int:
    db = _direct_db_session()
    try:
        user = db.query(User).filter(User.username == username).one()
        account = UserOAuth(
            user_id=int(user.id),
            provider=provider,
            access_token="access-token",
            email=email,
        )
        db.add(account)
        db.commit()
        db.refresh(account)
        return int(account.id)
    finally:
        db.close()


def _install_runtime_mcp_connector(
    agent_id: int,
    *,
    context_required: bool = True,
    secret_required: bool = False,
    connector_user_id: int | None = None,
) -> int:
    db = _direct_db_session()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).one()
        server_name = f"ShiftCare Trigger {agent_id}"
        runtime_input_schema = {
            "context": {
                "account_id": {
                    "type": "string",
                    "required": context_required,
                }
            }
        }
        runtime_bindings = [
            {
                "source": {"input_type": "context", "key": "account_id"},
                "target": {"target_type": "mcp_meta", "key": "account_id"},
            }
        ]
        if secret_required:
            runtime_input_schema["secrets"] = {
                "authorization": {"type": "string", "required": True}
            }
            runtime_bindings.append(
                {
                    "source": {"input_type": "secrets", "key": "authorization"},
                    "target": {
                        "target_type": "transport_headers",
                        "key": "Authorization",
                    },
                }
            )
        server = MCPServer(
            name=server_name,
            description="ShiftCare trigger MCP",
            managed="external",
            transport="streamable_http",
            url="https://mcp.shiftcare.test",
            runtime_input_schema=runtime_input_schema,
            runtime_bindings=runtime_bindings,
            allow_delegated_authorization=secret_required,
        )
        db.add(server)
        db.flush()
        db.add(
            UserMCPServer(
                user_id=(
                    int(connector_user_id)
                    if connector_user_id is not None
                    else int(agent.user_id)
                ),
                mcpserver_id=int(server.id),
                is_owner=True,
                can_edit=True,
                is_active=True,
            )
        )
        agent.tool_categories = [f"mcp:{server_name}"]
        db.commit()
        return int(server.id)
    finally:
        db.close()


def test_webhook_trigger_crud_returns_secret_once() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)

    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "webhook",
            "name": "Inbound webhook",
            "prompt_template": "payload={{payload}}",
            "config": {"source": "crm"},
        },
    )
    assert created.status_code == 200, created.text
    body = created.json()
    assert body["type"] == "webhook"
    assert body["callback_id"]
    assert body["webhook_secret"]

    db = _direct_db_session()
    try:
        trigger = db.query(AgentTrigger).filter(AgentTrigger.id == body["id"]).one()
        # New rows store only the encrypted HMAC secret; no bcrypt hash.
        assert trigger.secret_hash is None
        assert trigger.secret_encrypted
        assert trigger.secret_encrypted != body["webhook_secret"]
        assert decrypt_value(str(trigger.secret_encrypted)) == body["webhook_secret"]
        assert trigger.provider == "webhook"
        assert trigger.callback_id == body["callback_id"]
    finally:
        db.close()

    listed = client.get(f"/api/agents/{agent_id}/triggers", headers=headers)
    assert listed.status_code == 200, listed.text
    assert len(listed.json()) == 1
    assert listed.json()[0]["webhook_secret"] is None

    patched = client.patch(
        f"/api/agents/{agent_id}/triggers/{body['id']}",
        headers=headers,
        json={"name": "Renamed webhook", "rotate_secret": True},
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["name"] == "Renamed webhook"
    assert patched.json()["webhook_secret"]


def test_trigger_routes_reject_workforce_manager_agent() -> None:
    # Workforce-generated manager agents are private implementation details
    # and must not be addressable through trigger management, matching the
    # exclusion applied by the share/widget/api-key channels.
    headers = _admin_headers()
    agent_id = _create_agent(headers, name="Workforce Manager Agent")

    # Create a trigger while the agent is still a normal one, so the
    # trigger-scoped routes below traverse ``get_owned_trigger``'s internal
    # ``get_owned_agent`` guard rather than short-circuiting on a missing
    # trigger row.
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "webhook",
            "name": "Inbound webhook",
            "prompt_template": "payload={{payload}}",
            "config": {"source": "crm"},
        },
    )
    assert created.status_code == 200, created.text
    trigger_id = int(created.json()["id"])

    db = _direct_db_session()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).one()
        agent.origin = AgentOrigin.WORKFORCE_GENERATED_MANAGER.value
        db.commit()
    finally:
        db.close()

    # Routes resolving the agent directly through ``get_owned_agent``.
    listed = client.get(f"/api/agents/{agent_id}/triggers", headers=headers)
    assert listed.status_code == 404

    recreated = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "webhook",
            "name": "Another webhook",
            "prompt_template": "payload={{payload}}",
            "config": {"source": "crm"},
        },
    )
    assert recreated.status_code == 404

    # Trigger-scoped routes gate through ``get_owned_trigger``, which relies
    # on its internal ``get_owned_agent`` call for the same exclusion.
    updated = client.patch(
        f"/api/agents/{agent_id}/triggers/{trigger_id}",
        headers=headers,
        json={"name": "Renamed"},
    )
    assert updated.status_code == 404

    runs = client.get(
        f"/api/agents/{agent_id}/triggers/{trigger_id}/runs", headers=headers
    )
    assert runs.status_code == 404

    deleted = client.delete(
        f"/api/agents/{agent_id}/triggers/{trigger_id}", headers=headers
    )
    assert deleted.status_code == 404


def test_trigger_config_validation_dispatches_through_provider() -> None:
    """CRUD config validation must go through TriggerProvider.validate_config
    for provider-backed types, not the module-level schema parser."""
    headers = _admin_headers()
    agent_id = _create_agent(headers)

    from xagent.web.services.trigger_providers import get_trigger_provider

    provider = get_trigger_provider("webhook")
    seen_configs: list[dict] = []
    original_validate = type(provider).validate_config

    def recording_validate(self, config):
        seen_configs.append(dict(config))
        return original_validate(self, config)

    with patch.object(type(provider), "validate_config", recording_validate):
        created = client.post(
            f"/api/agents/{agent_id}/triggers",
            headers=headers,
            json={
                "type": "webhook",
                "name": "Provider-validated webhook",
                "prompt_template": "payload={{payload}}",
                "config": {"source": "crm"},
            },
        )
    assert created.status_code == 200, created.text
    assert seen_configs == [{"source": "crm"}]

    # Provider validation errors surface as the same 400 config error.
    invalid = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "webhook",
            "name": "Bad webhook config",
            "prompt_template": "payload={{payload}}",
            "config": {"event_types": "not-a-list"},
        },
    )
    assert invalid.status_code == 400, invalid.text
    assert "webhook trigger config invalid" in invalid.json()["detail"]


def test_trigger_test_run_creates_hidden_agent_task(mock_bg_scheduler) -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "webhook",
            "name": "Test webhook",
            "prompt_template": "Handle {{payload}}",
        },
    )
    trigger_id = created.json()["id"]

    fired = client.post(
        f"/api/agents/{agent_id}/triggers/{trigger_id}/test",
        headers=headers,
        json={"payload": {"subject": "hello"}, "source_event_id": "test-event"},
    )
    assert fired.status_code == 200, fired.text
    run_body = fired.json()["trigger_run"]
    assert run_body["status"] == TriggerRunStatus.RUNNING.value
    assert run_body["task_id"]
    assert fired.json()["duplicate"] is False
    assert mock_bg_scheduler.call_count == 1

    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == run_body["task_id"]).one()
        assert task.agent_id == agent_id
        assert task.source == "trigger"
        assert task.is_visible is False
        assert task.status == TaskStatus.RUNNING
        assert "hello" in (task.description or "")
    finally:
        db.close()


def test_trigger_test_run_mcp_setup_failure_marks_run_failed() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={"type": "webhook", "name": "Required MCP failure"},
    )
    assert created.status_code == 200, created.text
    trigger_id = int(created.json()["id"])

    private_detail = "connector-token-must-not-leak"
    setup_error = RequiredMCPUnavailableError(
        [
            MCPUnavailableSummary.from_values(
                private_detail,
                "oauth_token_required",
            )
        ]
    )
    setup_calls = 0

    async def fail_required_mcp_setup(_manager, *_args, **_kwargs):
        nonlocal setup_calls
        setup_calls += 1
        raise setup_error

    final_state: tuple[str, str | None, str, str | None, datetime | None] | None = None
    with (
        patch(
            "xagent.web.services.task_orchestrator._schedule_bg",
            new=_real_schedule_bg,
        ),
        patch(
            "xagent.web.api.chat.AgentServiceManager.get_agent_for_task",
            new=fail_required_mcp_setup,
        ),
        TestClient(app_for_tests, raise_server_exceptions=False) as live_client,
    ):
        fired = live_client.post(
            f"/api/agents/{agent_id}/triggers/{trigger_id}/test",
            headers=headers,
            json={"payload": {"subject": "exercise MCP setup"}},
        )
        assert fired.status_code == 200, fired.text
        run_id = int(fired.json()["trigger_run"]["id"])

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            db = _direct_db_session()
            try:
                run = db.query(TriggerRun).filter(TriggerRun.id == run_id).one()
                task = db.query(Task).filter(Task.id == run.task_id).one()
                final_state = (
                    str(run.status),
                    run.error_message,
                    task.status.value,
                    task.error_message,
                    run.finished_at,
                )
            finally:
                db.close()
            if final_state[0] == TriggerRunStatus.FAILED.value:
                break
            time.sleep(0.01)
        else:
            pytest.fail(f"trigger run did not reach FAILED: {final_state}")

    safe_error = "Required MCP servers are unavailable."
    assert final_state is not None
    run_status, run_error, task_status, task_error, run_finished_at = final_state
    assert setup_calls == 1
    assert run_status == TriggerRunStatus.FAILED.value
    assert run_error == safe_error
    assert task_status == TaskStatus.FAILED.value
    assert task_error == safe_error
    assert run_finished_at is not None
    assert private_detail not in str(final_state)


def _signed_webhook_headers(
    secret: str, raw_body: bytes, *, event_id: str | None = None
) -> dict[str, str]:
    timestamp = str(int(time.time()))
    headers = {
        "x-xagent-signature": sign_webhook_payload(secret, timestamp, raw_body),
        "x-xagent-timestamp": timestamp,
    }
    if event_id:
        headers["x-xagent-event-id"] = event_id
    return headers


def test_public_webhook_validates_signature_and_deduplicates(
    mock_bg_scheduler,
) -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={"type": "webhook", "name": "Public webhook"},
    )
    body = created.json()
    url = f"/api/triggers/callback/webhook/{body['callback_id']}"
    raw_body = b'{"subject": "hello"}'

    unsigned = client.post(url, content=raw_body)
    assert unsigned.status_code == 401

    forged_headers = _signed_webhook_headers("wrong-secret", raw_body, event_id="evt-1")
    forged = client.post(url, headers=forged_headers, content=raw_body)
    assert forged.status_code == 401

    db = _direct_db_session()
    try:
        rejected_audits = (
            db.query(TriggerAudit)
            .filter(TriggerAudit.outcome == "rejected_signature")
            .all()
        )
        assert len(rejected_audits) >= 1
        assert rejected_audits[-1].trigger_id == body["id"]
        assert rejected_audits[-1].provider == "webhook"
    finally:
        db.close()

    event_headers = _signed_webhook_headers(
        body["webhook_secret"], raw_body, event_id="evt-1"
    )
    first = client.post(url, headers=event_headers, content=raw_body)
    assert first.status_code == 200, first.text
    assert first.json()["outcome"] == "accepted"
    assert len(first.json()["trigger_run_ids"]) == 1
    assert first.json()["duplicates"] == 0

    second = client.post(url, headers=event_headers, content=raw_body)
    assert second.status_code == 200, second.text
    assert second.json()["duplicates"] == 1
    assert second.json()["trigger_run_ids"] == []
    assert mock_bg_scheduler.call_count == 1

    db = _direct_db_session()
    try:
        assert db.query(TriggerRun).count() == 1
        assert db.query(Task).filter(Task.source == "trigger").count() == 1
    finally:
        db.close()


def test_public_webhook_rejects_stale_timestamp(mock_bg_scheduler) -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={"type": "webhook", "name": "Replay webhook"},
    )
    body = created.json()
    url = f"/api/triggers/callback/webhook/{body['callback_id']}"
    raw_body = b'{"subject": "hello"}'

    stale_timestamp = str(int(time.time()) - 3600)
    stale = client.post(
        url,
        headers={
            "x-xagent-signature": sign_webhook_payload(
                body["webhook_secret"], stale_timestamp, raw_body
            ),
            "x-xagent-timestamp": stale_timestamp,
        },
        content=raw_body,
    )
    assert stale.status_code == 401
    db = _direct_db_session()
    try:
        assert db.query(TriggerRun).count() == 0
    finally:
        db.close()


def test_legacy_webhook_route_verifies_bcrypt_secret(mock_bg_scheduler) -> None:
    """Pre-pipeline webhooks keep working on the deprecated token route."""
    import bcrypt

    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={"type": "webhook", "name": "Legacy webhook"},
    )
    trigger_id = created.json()["id"]

    # Rewrite the row into its pre-migration shape: webhook token plus
    # bcrypt secret hash, none of the unified pipeline identity fields.
    db = _direct_db_session()
    try:
        trigger = db.query(AgentTrigger).filter(AgentTrigger.id == trigger_id).one()
        trigger.webhook_token = "legacy-token-1"
        trigger.secret_hash = bcrypt.hashpw(
            b"legacy-secret", bcrypt.gensalt(rounds=4)
        ).decode("utf-8")
        trigger.callback_id = None
        trigger.secret_encrypted = None
        trigger.provider = None
        db.commit()
    finally:
        db.close()

    url = "/api/triggers/webhook/legacy-token-1"
    payload = {"subject": "hello"}

    unknown = client.post("/api/triggers/webhook/unknown-token", json=payload)
    assert unknown.status_code == 404

    missing = client.post(url, json=payload)
    assert missing.status_code == 401

    wrong = client.post(
        url, json=payload, headers={"x-xagent-trigger-secret": "wrong-secret"}
    )
    assert wrong.status_code == 401

    accepted = client.post(
        url, json=payload, headers={"x-xagent-trigger-secret": "legacy-secret"}
    )
    assert accepted.status_code == 200, accepted.text
    assert accepted.headers.get("deprecation") == "true"
    assert accepted.json()["trigger_run_id"] > 0

    db = _direct_db_session()
    try:
        assert db.query(TriggerRun).count() == 1
        audits = db.query(TriggerAudit).order_by(TriggerAudit.id.asc()).all()
        outcomes = [str(a.outcome) for a in audits if a.trigger_id == trigger_id]
        assert "rejected_signature" in outcomes
        assert "accepted" in outcomes
    finally:
        db.close()


def test_public_callback_unknown_provider_and_callback_are_controlled(
    mock_bg_scheduler,
) -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={"type": "webhook", "name": "Known webhook"},
    )
    body = created.json()
    raw_body = b"{}"

    unknown_provider = client.post(
        f"/api/triggers/callback/ghost/{body['callback_id']}",
        content=raw_body,
    )
    assert unknown_provider.status_code == 404
    assert unknown_provider.json()["outcome"] == "unknown_provider"

    unknown_callback = client.post(
        "/api/triggers/callback/webhook/does-not-exist",
        content=raw_body,
    )
    assert unknown_callback.status_code == 404
    assert unknown_callback.json()["outcome"] == "unknown_callback"


def test_public_callback_disabled_trigger_is_rejected_after_verification(
    mock_bg_scheduler,
) -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={"type": "webhook", "name": "Disabled webhook", "enabled": False},
    )
    body = created.json()
    raw_body = b'{"subject": "hello"}'

    fired = client.post(
        f"/api/triggers/callback/webhook/{body['callback_id']}",
        headers=_signed_webhook_headers(body["webhook_secret"], raw_body),
        content=raw_body,
    )
    assert fired.status_code == 409
    assert fired.json()["outcome"] == "rejected_disabled"

    db = _direct_db_session()
    try:
        assert db.query(TriggerRun).count() == 0
        disabled_audits = (
            db.query(TriggerAudit)
            .filter(TriggerAudit.outcome == "rejected_disabled")
            .all()
        )
        assert len(disabled_audits) == 1
        assert disabled_audits[0].trigger_id == body["id"]
    finally:
        db.close()


def test_public_callback_filters_events_against_allow_list(
    mock_bg_scheduler,
) -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "webhook",
            "name": "Filtered webhook",
            "config": {"event_types": ["order.created"]},
        },
    )
    body = created.json()
    url = f"/api/triggers/callback/webhook/{body['callback_id']}"

    ignored_body = b'{"event_type": "order.deleted", "id": "evt-a"}'
    ignored = client.post(
        url,
        headers=_signed_webhook_headers(body["webhook_secret"], ignored_body),
        content=ignored_body,
    )
    assert ignored.status_code == 200, ignored.text
    assert ignored.json()["trigger_run_ids"] == []

    matched_body = b'{"event_type": "order.created", "id": "evt-b"}'
    matched = client.post(
        url,
        headers=_signed_webhook_headers(body["webhook_secret"], matched_body),
        content=matched_body,
    )
    assert matched.status_code == 200, matched.text
    assert len(matched.json()["trigger_run_ids"]) == 1

    db = _direct_db_session()
    try:
        assert db.query(TriggerRun).count() == 1
    finally:
        db.close()


def test_public_webhook_invalid_utf8_body_is_a_controlled_parse_failure(
    mock_bg_scheduler,
) -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={"type": "webhook", "name": "Invalid UTF-8 webhook"},
    )
    body = created.json()
    url = f"/api/triggers/callback/webhook/{body['callback_id']}"
    raw_body = b'\xff{"subject":"hello"}'

    fired = client.post(
        url,
        headers=_signed_webhook_headers(body["webhook_secret"], raw_body),
        content=raw_body,
    )
    assert fired.status_code == 400
    assert fired.json()["outcome"] == "execution_failure"

    db = _direct_db_session()
    try:
        assert db.query(TriggerRun).count() == 0
    finally:
        db.close()
    assert mock_bg_scheduler.call_count == 0


def test_trigger_name_validation_rejects_empty_and_oversized() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)

    empty = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={"type": "webhook", "name": "   "},
    )
    assert empty.status_code == 400

    oversized = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={"type": "webhook", "name": "x" * 201},
    )
    assert oversized.status_code == 422

    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={"type": "webhook", "name": "Valid"},
    )
    assert created.status_code == 200, created.text

    patched = client.patch(
        f"/api/agents/{agent_id}/triggers/{created.json()['id']}",
        headers=headers,
        json={"name": " "},
    )
    assert patched.status_code == 400


def test_gmail_trigger_crud_persists_filters() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    account_id = _connect_gmail_account(email="Owner@Gmail.Example")

    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "gmail",
            "name": "Support inbox",
            "config": {
                "watch_label": "INBOX",
                "sender_filter": "boss@company.com",
                "subject_keyword": "urgent",
                "oauth_account_id": account_id,
            },
            "prompt_template": "Handle Gmail message {{payload}}",
        },
    )
    assert created.status_code == 200, created.text
    body = created.json()
    assert body["type"] == "gmail"
    assert body["webhook_token"] is None
    assert body["webhook_secret"] is None
    assert body["config"] == {
        "watch_label": "INBOX",
        "sender_filter": "boss@company.com",
        "subject_keyword": "urgent",
        "oauth_account_id": account_id,
    }

    db = _direct_db_session()
    try:
        trigger = db.query(AgentTrigger).filter(AgentTrigger.id == body["id"]).one()
        assert trigger.provider == "gmail"
        assert trigger.resource_id == "owner@gmail.example"
    finally:
        db.close()

    patched = client.patch(
        f"/api/agents/{agent_id}/triggers/{body['id']}",
        headers=headers,
        json={
            "config": {
                "watch_label": "CATEGORY_PRIMARY",
                "sender_filter": "",
                "subject_keyword": "invoice",
                "oauth_account_id": account_id,
            }
        },
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["config"] == {
        "watch_label": "CATEGORY_PRIMARY",
        "sender_filter": "",
        "subject_keyword": "invoice",
        "oauth_account_id": account_id,
    }


def test_gmail_trigger_requires_oauth_account() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)

    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "gmail",
            "name": "No account",
            "config": {"watch_label": "INBOX"},
        },
    )
    assert created.status_code == 400
    assert "oauth_account_id" in created.json()["detail"]


def test_gmail_trigger_rejects_foreign_oauth_account() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    from .conftest import _register_second_user

    _register_second_user()
    foreign_account_id = _connect_gmail_account("bob", email="bob@gmail.example")

    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "gmail",
            "name": "Foreign account",
            "config": {"watch_label": "INBOX", "oauth_account_id": foreign_account_id},
        },
    )
    assert created.status_code == 400
    assert "not found" in created.json()["detail"].lower()


def test_gmail_trigger_rejects_non_gmail_oauth_account() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    drive_account_id = _connect_gmail_account(
        email="owner@gmail.example", provider="google-drive"
    )

    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "gmail",
            "name": "Wrong provider",
            "config": {"watch_label": "INBOX", "oauth_account_id": drive_account_id},
        },
    )
    assert created.status_code == 400
    assert "not a gmail account" in created.json()["detail"].lower()


def test_enabled_gmail_trigger_create_provisions_bound_mailbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []

    def fake_provision_gmail_trigger(db, trigger: AgentTrigger) -> str:
        calls.append((int(trigger.id), int(trigger.config["oauth_account_id"])))
        setattr(trigger, "provisioning_status", TriggerProvisioningStatus.ACTIVE.value)
        setattr(trigger, "provisioning_error", None)
        db.add(trigger)
        db.commit()
        return TriggerProvisioningStatus.ACTIVE.value

    monkeypatch.setattr(
        "xagent.web.services.trigger_providers.gmail.provision_gmail_trigger",
        fake_provision_gmail_trigger,
        raising=False,
    )
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    account_id = _connect_gmail_account()

    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "gmail",
            "name": "Support inbox",
            "config": {"watch_label": "INBOX", "oauth_account_id": account_id},
        },
    )

    assert created.status_code == 200, created.text
    assert created.json()["provisioning_status"] == "active"
    assert created.json()["provisioning_error"] is None
    db = _direct_db_session()
    try:
        trigger = (
            db.query(AgentTrigger).filter(AgentTrigger.id == created.json()["id"]).one()
        )
        assert calls == [(int(trigger.id), account_id)]
    finally:
        db.close()


def test_enabling_existing_gmail_trigger_provisions_bound_mailbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def fake_provision_gmail_trigger(db, trigger: AgentTrigger) -> str:
        calls.append(int(trigger.id))
        setattr(trigger, "provisioning_status", TriggerProvisioningStatus.ACTIVE.value)
        setattr(trigger, "provisioning_error", None)
        db.add(trigger)
        db.commit()
        return TriggerProvisioningStatus.ACTIVE.value

    monkeypatch.setattr(
        "xagent.web.services.trigger_providers.gmail.provision_gmail_trigger",
        fake_provision_gmail_trigger,
        raising=False,
    )
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    account_id = _connect_gmail_account()
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "gmail",
            "enabled": False,
            "name": "Support inbox",
            "config": {"watch_label": "INBOX", "oauth_account_id": account_id},
        },
    )
    assert created.status_code == 200, created.text
    assert calls == []

    patched = client.patch(
        f"/api/agents/{agent_id}/triggers/{created.json()['id']}",
        headers=headers,
        json={"enabled": True},
    )

    assert patched.status_code == 200, patched.text
    assert patched.json()["provisioning_status"] == "active"
    assert calls == [created.json()["id"]]


def test_listing_triggers_reflects_background_provisioning_convergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Status reported by the API self-resolves once the watch state converges,
    without requiring another user-initiated create/update."""
    from xagent.web.models.gmail_watch import GmailWatchState

    def fake_provision_gmail_trigger(db, trigger: AgentTrigger) -> str:
        setattr(trigger, "provisioning_status", TriggerProvisioningStatus.PENDING.value)
        setattr(trigger, "provisioning_error", None)
        db.add(trigger)
        db.commit()
        return TriggerProvisioningStatus.PENDING.value

    monkeypatch.setattr(
        "xagent.web.services.trigger_providers.gmail.provision_gmail_trigger",
        fake_provision_gmail_trigger,
        raising=False,
    )
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    account_id = _connect_gmail_account()

    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "gmail",
            "name": "Support inbox",
            "config": {"watch_label": "INBOX", "oauth_account_id": account_id},
        },
    )
    assert created.status_code == 200, created.text
    assert created.json()["provisioning_status"] == "pending"

    # Simulate the background thread converging the mailbox to active.
    db = _direct_db_session()
    try:
        user = db.query(User).filter(User.username == "admin").one()
        db.add(
            GmailWatchState(
                user_id=int(user.id),
                oauth_account_id=account_id,
                email="owner@gmail.example",
                history_id="hist-1",
                topic_name="projects/demo/topics/xagent-gmail-abc",
                status=TriggerProvisioningStatus.ACTIVE.value,
            )
        )
        db.commit()
    finally:
        db.close()

    listed = client.get(f"/api/agents/{agent_id}/triggers", headers=headers)
    assert listed.status_code == 200, listed.text
    assert listed.json()[0]["provisioning_status"] == "active"
    assert listed.json()[0]["provisioning_error"] is None

    db = _direct_db_session()
    try:
        trigger = (
            db.query(AgentTrigger).filter(AgentTrigger.id == created.json()["id"]).one()
        )
        assert trigger.provisioning_status == TriggerProvisioningStatus.ACTIVE.value
    finally:
        db.close()


def test_gmail_trigger_update_releases_previous_mailbox_and_provisions_new_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provisioned: list[int] = []
    released: list[int] = []

    def fake_provision_gmail_trigger(db, trigger: AgentTrigger) -> str:
        provisioned.append(int(trigger.config["oauth_account_id"]))
        setattr(trigger, "provisioning_status", TriggerProvisioningStatus.ACTIVE.value)
        setattr(trigger, "provisioning_error", None)
        db.add(trigger)
        db.commit()
        return TriggerProvisioningStatus.ACTIVE.value

    def fake_release_gmail_mailbox_if_unused(db, oauth_account_id: int) -> bool:
        released.append(oauth_account_id)
        return True

    monkeypatch.setattr(
        "xagent.web.services.trigger_providers.gmail.provision_gmail_trigger",
        fake_provision_gmail_trigger,
        raising=False,
    )
    monkeypatch.setattr(
        "xagent.web.services.trigger_providers.gmail.release_gmail_mailbox_if_unused",
        fake_release_gmail_mailbox_if_unused,
        raising=False,
    )
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    first_account_id = _connect_gmail_account(email="first@gmail.example")
    second_account_id = _connect_gmail_account(email="second@gmail.example")
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "gmail",
            "name": "Support inbox",
            "config": {"watch_label": "INBOX", "oauth_account_id": first_account_id},
        },
    )
    assert created.status_code == 200, created.text

    patched = client.patch(
        f"/api/agents/{agent_id}/triggers/{created.json()['id']}",
        headers=headers,
        json={
            "config": {
                "watch_label": "INBOX",
                "oauth_account_id": second_account_id,
            }
        },
    )

    assert patched.status_code == 200, patched.text
    assert provisioned == [first_account_id, second_account_id]
    assert released == [first_account_id]


def test_gmail_trigger_delete_releases_mailbox_after_row_is_deleted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provisioned: list[int] = []
    released: list[int] = []

    def fake_provision_gmail_trigger(db, trigger: AgentTrigger) -> str:
        provisioned.append(int(trigger.config["oauth_account_id"]))
        setattr(trigger, "provisioning_status", TriggerProvisioningStatus.ACTIVE.value)
        setattr(trigger, "provisioning_error", None)
        db.add(trigger)
        db.commit()
        return TriggerProvisioningStatus.ACTIVE.value

    def fake_release_gmail_mailbox_if_unused(db, oauth_account_id: int) -> bool:
        assert db.query(AgentTrigger).count() == 0
        released.append(oauth_account_id)
        return True

    monkeypatch.setattr(
        "xagent.web.services.trigger_providers.gmail.provision_gmail_trigger",
        fake_provision_gmail_trigger,
        raising=False,
    )
    monkeypatch.setattr(
        "xagent.web.services.trigger_providers.gmail.release_gmail_mailbox_if_unused",
        fake_release_gmail_mailbox_if_unused,
        raising=False,
    )
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    account_id = _connect_gmail_account()
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "gmail",
            "name": "Support inbox",
            "config": {"watch_label": "INBOX", "oauth_account_id": account_id},
        },
    )
    assert created.status_code == 200, created.text

    deleted = client.delete(
        f"/api/agents/{agent_id}/triggers/{created.json()['id']}",
        headers=headers,
    )

    assert deleted.status_code == 200, deleted.text
    assert provisioned == [account_id]
    assert released == [account_id]


def test_trigger_crud_dispatches_through_provider_protocol() -> None:
    """CRUD reaches provisioning only via TriggerProvider.register/unregister.

    A recording provider swapped into the registry observes every CRUD
    provisioning dispatch, proving the paths hold no provider-specific
    branches: create registers, a binding change unregisters the previous
    config and registers the new one, delete unregisters.
    """
    from xagent.web.services.trigger_providers.registry import (
        get_trigger_provider,
        register_trigger_provider,
    )
    from xagent.web.services.trigger_providers.schemas import RegistrationResult

    real = get_trigger_provider("gmail")
    calls: list[tuple[str, int]] = []

    class RecordingProvider:
        name = "gmail"
        ack_policy = real.ack_policy

        def validate_config(self, config):
            return real.validate_config(config)

        async def register(self, db, trigger, config) -> RegistrationResult:
            calls.append(("register", int(config["oauth_account_id"])))
            setattr(
                trigger,
                "provisioning_status",
                TriggerProvisioningStatus.ACTIVE.value,
            )
            db.add(trigger)
            db.commit()
            return RegistrationResult(status=TriggerProvisioningStatus.ACTIVE)

        async def unregister(self, db, trigger, config) -> None:
            calls.append(("unregister", int(config["oauth_account_id"])))

    register_trigger_provider(RecordingProvider(), replace=True)
    try:
        headers = _admin_headers()
        agent_id = _create_agent(headers)
        first_account_id = _connect_gmail_account(email="first@gmail.example")
        second_account_id = _connect_gmail_account(email="second@gmail.example")

        created = client.post(
            f"/api/agents/{agent_id}/triggers",
            headers=headers,
            json={
                "type": "gmail",
                "name": "Support inbox",
                "config": {
                    "watch_label": "INBOX",
                    "oauth_account_id": first_account_id,
                },
            },
        )
        assert created.status_code == 200, created.text
        assert calls == [("register", first_account_id)]

        patched = client.patch(
            f"/api/agents/{agent_id}/triggers/{created.json()['id']}",
            headers=headers,
            json={
                "config": {
                    "watch_label": "INBOX",
                    "oauth_account_id": second_account_id,
                }
            },
        )
        assert patched.status_code == 200, patched.text
        assert calls[1:] == [
            ("unregister", first_account_id),
            ("register", second_account_id),
        ]

        deleted = client.delete(
            f"/api/agents/{agent_id}/triggers/{created.json()['id']}",
            headers=headers,
        )
        assert deleted.status_code == 200, deleted.text
        assert calls[3:] == [("unregister", second_account_id)]
    finally:
        register_trigger_provider(real, replace=True)


def test_gmail_trigger_requires_watch_label() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)

    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "gmail",
            "name": "Missing label",
            "config": {"sender_filter": "boss@company.com"},
        },
    )
    assert created.status_code == 400
    assert "watch_label" in created.json()["detail"]


def test_gmail_trigger_test_run_creates_hidden_agent_task(mock_bg_scheduler) -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    account_id = _connect_gmail_account()
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "gmail",
            "name": "Gmail support",
            "config": {"watch_label": "INBOX", "oauth_account_id": account_id},
            "prompt_template": "Triage this email: {{payload}}",
        },
    )
    assert created.status_code == 200, created.text
    trigger_id = created.json()["id"]

    fired = client.post(
        f"/api/agents/{agent_id}/triggers/{trigger_id}/test",
        headers=headers,
        json={
            "payload": {
                "from": "boss@company.com",
                "subject": "urgent invoice",
                "snippet": "please review",
            },
            "source_event_id": "gmail-msg-1",
        },
    )
    assert fired.status_code == 200, fired.text
    run_body = fired.json()["trigger_run"]
    assert run_body["status"] == TriggerRunStatus.RUNNING.value
    assert run_body["task_id"]
    assert fired.json()["duplicate"] is False
    assert mock_bg_scheduler.call_count == 1

    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == run_body["task_id"]).one()
        assert task.agent_id == agent_id
        assert task.source == "trigger"
        assert task.is_visible is False
        assert task.status == TaskStatus.RUNNING
        assert "urgent invoice" in (task.description or "")
    finally:
        db.close()


def test_scheduled_next_run_skips_stale_intervals_without_iteration() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    stale_due_at = now - timedelta(days=3650)

    next_run_at = _compute_next_run_at(
        {"interval_seconds": 1},
        from_time=now,
        previous_due_at=stale_due_at,
        include_explicit=False,
    )

    assert next_run_at == now + timedelta(seconds=1)


def test_scheduled_scan_fires_due_trigger_and_advances_next_run(
    mock_bg_scheduler,
) -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "scheduled",
            "name": "Every minute",
            "config": {"interval_seconds": 60},
        },
    )
    assert created.status_code == 200, created.text
    trigger_id = created.json()["id"]

    due_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    db = _direct_db_session()
    try:
        trigger = db.query(AgentTrigger).filter(AgentTrigger.id == trigger_id).one()
        trigger.next_run_at = due_at
        db.add(trigger)
        db.commit()

        runs = scan_due_scheduled_triggers(db, now=datetime.now(timezone.utc))
        assert len(runs) == 1
        db.refresh(trigger)
        assert trigger.next_run_at is not None
        next_run_at = trigger.next_run_at
        if next_run_at.tzinfo is None:
            next_run_at = next_run_at.replace(tzinfo=timezone.utc)
        assert next_run_at > due_at
        run = db.query(TriggerRun).filter(TriggerRun.id == runs[0].id).one()
        assert run.status == TriggerRunStatus.PENDING.value
        assert run.task_id is not None
        task = db.query(Task).filter(Task.id == run.task_id).one()
        assert task.agent_id == agent_id
        assert task.source == "trigger"
        assert task.is_visible is False
        assert task.status == TaskStatus.PENDING

        assert mock_bg_scheduler.call_count == 0
        assert asyncio.run(dispatch_pending_trigger_runs(db)) == 1
        db.refresh(run)
        db.refresh(task)
        assert run.status == TriggerRunStatus.RUNNING.value
        assert task.status == TaskStatus.RUNNING
    finally:
        db.close()

    assert mock_bg_scheduler.call_count == 1


def test_trigger_dispatcher_loop_scans_due_scheduled_trigger(mock_bg_scheduler) -> None:
    """End-to-end: the in-process dispatcher loop itself scans a due scheduled
    trigger (no Celery) and creates a PENDING run on its first tick."""
    from xagent.web import app as app_module

    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "scheduled",
            "name": "Loop scan",
            "config": {"interval_seconds": 60},
        },
    )
    assert created.status_code == 200, created.text
    trigger_id = created.json()["id"]

    due_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    db = _direct_db_session()
    try:
        trigger = db.query(AgentTrigger).filter(AgentTrigger.id == trigger_id).one()
        trigger.next_run_at = due_at
        db.add(trigger)
        db.commit()
    finally:
        db.close()

    async def fake_dispatch(_db, *, limit):
        # Stop the loop right after the (real) scan tick so the agent runner
        # never actually spins; we only assert the scan wired up correctly.
        raise asyncio.CancelledError

    with (
        patch("xagent.web.app.get_gmail_watch_enabled", return_value=False),
        patch(
            "xagent.web.services.triggers.dispatch_pending_trigger_runs",
            new=fake_dispatch,
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        asyncio.run(
            app_module._run_trigger_dispatcher(poll_interval_seconds=60, batch_size=25)
        )

    db = _direct_db_session()
    try:
        runs = db.query(TriggerRun).filter(TriggerRun.trigger_id == trigger_id).all()
        assert len(runs) == 1
        assert runs[0].status == TriggerRunStatus.PENDING.value
        assert runs[0].task_id is not None
    finally:
        db.close()


class _FirstNoneQuery:
    """Query wrapper whose ``.first()`` returns None, delegating everything else."""

    def __init__(self, query):
        self._query = query

    def filter(self, *args, **kwargs):
        return _FirstNoneQuery(self._query.filter(*args, **kwargs))

    def first(self):
        return None

    def __getattr__(self, name):
        return getattr(self._query, name)


class _PrecheckMissSession:
    """Delegate to a real session but force the first ``TriggerRun`` lookup to
    miss, simulating a scanner whose idempotency pre-check ran before a
    concurrent insert committed. The following insert then collides on the
    unique key, driving the IntegrityError recovery branch."""

    def __init__(self, db):
        self._db = db
        self._missed = False

    def query(self, *args, **kwargs):
        query = self._db.query(*args, **kwargs)
        if not self._missed and args and args[0] is TriggerRun:
            self._missed = True
            return _FirstNoneQuery(query)
        return query

    def __getattr__(self, name):
        return getattr(self._db, name)


def test_get_or_create_trigger_run_recovers_from_racing_insert(
    mock_bg_scheduler,
) -> None:
    """Dedup safety the in-process scan relies on: if a concurrent scan commits
    the run between this call's pre-check and its own insert, the insert hits
    the unique idempotency key and recovers the existing run rather than
    creating a duplicate or raising."""
    from xagent.web.services import triggers as triggers_mod

    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "scheduled",
            "name": "Racing",
            "config": {"interval_seconds": 60},
        },
    )
    assert created.status_code == 200, created.text
    trigger_id = created.json()["id"]

    event_payload = {"trigger_id": trigger_id, "scheduled_at": "t"}
    source_event_id = f"scheduled:{trigger_id}:once"

    db = _direct_db_session()
    try:
        trigger = db.query(AgentTrigger).filter(AgentTrigger.id == trigger_id).one()

        first_run, first_created = triggers_mod._get_or_create_trigger_run(
            db,
            trigger=trigger,
            event_payload=event_payload,
            source_event_id=source_event_id,
            background_job_id=None,
            test=False,
        )
        assert first_created is True

        # Second scan's pre-check misses; its insert collides on the unique key.
        racing_session = _PrecheckMissSession(db)
        second_run, second_created = triggers_mod._get_or_create_trigger_run(
            racing_session,
            trigger=trigger,
            event_payload=event_payload,
            source_event_id=source_event_id,
            background_job_id=None,
            test=False,
        )
        # The forced pre-check miss means created=False could only come from the
        # IntegrityError recovery branch, not an early pre-check return.
        assert racing_session._missed is True
        assert second_created is False
        assert second_run.id == first_run.id

        rows = db.query(TriggerRun).filter(TriggerRun.trigger_id == trigger_id).all()
        assert len(rows) == 1
    finally:
        db.close()


def test_trigger_config_rejects_persisted_runtime_secrets() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)

    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "scheduled",
            "name": "Runtime secret schedule",
            "config": {
                "interval_seconds": 60,
                "connector_runtime_context": [
                    {
                        "connector_ref": {
                            "connector_type": "mcp",
                            "connector_id": 1,
                        },
                        "secrets": {"authorization": "Bearer delegated"},
                    }
                ],
            },
        },
    )

    assert created.status_code == 400, created.text
    assert (
        created.json()["detail"] == "Runtime secret is not allowed for this entrypoint."
    )


def test_trigger_config_update_rejects_persisted_runtime_secrets() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "scheduled",
            "name": "Runtime context schedule",
            "config": {"interval_seconds": 60},
        },
    )
    assert created.status_code == 200, created.text

    patched = client.patch(
        f"/api/agents/{agent_id}/triggers/{created.json()['id']}",
        headers=headers,
        json={
            "config": {
                "interval_seconds": 60,
                "connector_runtime_context": [
                    {
                        "connector_ref": {
                            "connector_type": "mcp",
                            "connector_id": 1,
                        },
                        "auth_selector": {"resource_owner_key": "xagent:user:1"},
                    }
                ],
            }
        },
    )

    assert patched.status_code == 400, patched.text
    assert (
        patched.json()["detail"] == "Runtime secret is not allowed for this entrypoint."
    )


def test_scheduled_scan_persists_connector_runtime_context() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    server_id = _install_runtime_mcp_connector(agent_id)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "scheduled",
            "name": "Tenant scoped schedule",
            "config": {
                "interval_seconds": 60,
                "connector_runtime_context": [
                    {
                        "connector_ref": {
                            "connector_type": "mcp",
                            "connector_id": server_id,
                        },
                        "context": {"account_id": "6185"},
                    }
                ],
            },
        },
    )
    assert created.status_code == 200, created.text
    trigger_id = created.json()["id"]

    db = _direct_db_session()
    try:
        trigger = db.query(AgentTrigger).filter(AgentTrigger.id == trigger_id).one()
        trigger.next_run_at = datetime.now(timezone.utc) - timedelta(seconds=5)
        db.add(trigger)
        db.commit()

        runs = scan_due_scheduled_triggers(db, now=datetime.now(timezone.utc))
        assert len(runs) == 1
        run = db.query(TriggerRun).filter(TriggerRun.id == runs[0].id).one()
        assert run.status == TriggerRunStatus.PENDING.value
        assert run.task_id is not None
        task = db.query(Task).filter(Task.id == run.task_id).one()
        assert task.connector_runtime_selected_refs == [
            {"connector_type": "mcp", "connector_id": server_id}
        ]
        context_row = (
            db.query(TaskConnectorRuntimeContext)
            .filter(TaskConnectorRuntimeContext.task_id == run.task_id)
            .one()
        )
        assert context_row.connector_type == "mcp"
        assert context_row.connector_id == server_id
        assert context_row.context == {"account_id": "6185"}
    finally:
        db.close()


def test_trigger_runtime_visibility_uses_trigger_task_owner() -> None:
    admin_headers = _admin_headers()
    teammate_headers = _register_second_user("trigger-teammate")
    db = _direct_db_session()
    try:
        admin_user_id = int(db.query(User).filter(User.username == "admin").one().id)
        trigger_owner_id = int(
            db.query(User).filter(User.username == "trigger-teammate").one().id
        )
    finally:
        db.close()

    set_agent_team_scope_hook(
        lambda _db, user_id: (
            AgentTeamScope(team_id=100, is_team_admin=False)
            if user_id in {admin_user_id, trigger_owner_id}
            else None
        )
    )
    try:
        agent_id = _create_agent(admin_headers)
        promoted = client.post(
            f"/api/agents/{agent_id}/promote-team",
            headers=admin_headers,
            json={"visibility": "team"},
        )
        assert promoted.status_code == 200, promoted.text
        server_id = _install_runtime_mcp_connector(
            agent_id,
            connector_user_id=trigger_owner_id,
        )
        created = client.post(
            f"/api/agents/{agent_id}/triggers",
            headers=teammate_headers,
            json={
                "type": "scheduled",
                "name": "Task owner connector visibility",
                "config": {
                    "interval_seconds": 60,
                    "connector_runtime_context": [
                        {
                            "connector_ref": {
                                "connector_type": "mcp",
                                "connector_id": server_id,
                            },
                            "context": {"account_id": "owner-account"},
                        }
                    ],
                },
            },
        )
        assert created.status_code == 200, created.text

        db = _direct_db_session()
        try:
            trigger = (
                db.query(AgentTrigger)
                .filter(AgentTrigger.id == created.json()["id"])
                .one()
            )
            assert int(trigger.user_id) == trigger_owner_id
            trigger.next_run_at = datetime.now(timezone.utc) - timedelta(seconds=5)
            db.add(trigger)
            db.commit()

            runs = scan_due_scheduled_triggers(db, now=datetime.now(timezone.utc))
            assert len(runs) == 1
            run = db.query(TriggerRun).filter(TriggerRun.id == runs[0].id).one()
            assert run.status == TriggerRunStatus.PENDING.value
            task = db.query(Task).filter(Task.id == run.task_id).one()
            assert int(task.user_id) == trigger_owner_id
            assert task.connector_runtime_selected_refs == [
                {"connector_type": "mcp", "connector_id": server_id}
            ]
        finally:
            db.close()
    finally:
        set_agent_team_scope_hook(None)


def test_scheduled_scan_fails_fast_when_required_runtime_secret_has_no_source() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    _install_runtime_mcp_connector(
        agent_id,
        context_required=False,
        secret_required=True,
    )
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "scheduled",
            "name": "Missing delegated token",
            "config": {"interval_seconds": 60},
        },
    )
    assert created.status_code == 200, created.text
    trigger_id = created.json()["id"]

    db = _direct_db_session()
    try:
        trigger = db.query(AgentTrigger).filter(AgentTrigger.id == trigger_id).one()
        trigger.next_run_at = datetime.now(timezone.utc) - timedelta(seconds=5)
        db.add(trigger)
        db.commit()

        runs = scan_due_scheduled_triggers(db, now=datetime.now(timezone.utc))
        assert len(runs) == 1
        run = db.query(TriggerRun).filter(TriggerRun.id == runs[0].id).one()
        assert run.status == TriggerRunStatus.FAILED.value
        assert run.task_id is None
        assert "scheduled_secret_unavailable" in str(run.error_message)
    finally:
        db.close()


def test_external_scoped_resolver_does_not_defer_scheduled_secret() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    _install_runtime_mcp_connector(
        agent_id,
        context_required=False,
        secret_required=True,
    )
    resolver_calls = 0

    def resolver(request):
        nonlocal resolver_calls
        resolver_calls += 1
        return request.values

    set_connector_runtime_resolver_for_testing(resolver, task_sources={"external"})
    try:
        created = client.post(
            f"/api/agents/{agent_id}/triggers",
            headers=headers,
            json={
                "type": "scheduled",
                "name": "External-only resolver",
                "config": {"interval_seconds": 60},
            },
        )
        assert created.status_code == 200, created.text
        trigger_id = created.json()["id"]

        db = _direct_db_session()
        try:
            trigger = db.query(AgentTrigger).filter(AgentTrigger.id == trigger_id).one()
            trigger.next_run_at = datetime.now(timezone.utc) - timedelta(seconds=5)
            db.add(trigger)
            db.commit()

            runs = scan_due_scheduled_triggers(db, now=datetime.now(timezone.utc))
            assert len(runs) == 1
            run = db.query(TriggerRun).filter(TriggerRun.id == runs[0].id).one()
            assert run.status == TriggerRunStatus.FAILED.value
            assert run.task_id is None
            assert "scheduled_secret_unavailable" in str(run.error_message)
        finally:
            db.close()
    finally:
        set_connector_runtime_resolver_for_testing(None)

    assert resolver_calls == 0


def test_scheduled_scan_allows_resolver_to_supply_required_runtime_secret() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    server_id = _install_runtime_mcp_connector(
        agent_id,
        context_required=False,
        secret_required=True,
    )

    resolver_requests = []

    def resolver(request):
        resolver_requests.append(request)
        return ConnectorRuntimeValues(
            context={},
            secrets={"authorization": "Bearer fresh"},
            auth_selector={},
        )

    set_connector_runtime_resolver_for_testing(resolver, task_sources={"trigger"})
    try:
        created = client.post(
            f"/api/agents/{agent_id}/triggers",
            headers=headers,
            json={
                "type": "scheduled",
                "name": "Resolver delegated token",
                "config": {"interval_seconds": 60},
            },
        )
        assert created.status_code == 200, created.text
        trigger_id = created.json()["id"]

        db = _direct_db_session()
        try:
            trigger = db.query(AgentTrigger).filter(AgentTrigger.id == trigger_id).one()
            trigger.next_run_at = datetime.now(timezone.utc) - timedelta(seconds=5)
            db.add(trigger)
            db.commit()

            runs = scan_due_scheduled_triggers(db, now=datetime.now(timezone.utc))
            assert len(runs) == 1
            run = db.query(TriggerRun).filter(TriggerRun.id == runs[0].id).one()
            assert run.status == TriggerRunStatus.PENDING.value
            assert run.task_id is not None
            task = db.query(Task).filter(Task.id == run.task_id).one()
            task_owner_user_id = int(task.user_id)
            assert task.connector_runtime_selected_refs == [
                {"connector_type": "mcp", "connector_id": server_id}
            ]
            view = load_connector_runtime_view(
                db=db,
                task_id=int(task.id),
                turn_id="scheduled-turn",
                user_id=None,
            )
        finally:
            db.close()
    finally:
        set_connector_runtime_resolver_for_testing(None)

    assert view[f"mcp:{server_id}"]["secrets"] == {"authorization": "Bearer fresh"}
    assert len(resolver_requests) == 1
    assert resolver_requests[0].task_source == "trigger"
    assert resolver_requests[0].user_id == task_owner_user_id


def test_scheduled_runtime_view_reports_scheduled_secret_when_resolver_omits_secret() -> (
    None
):
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    server_id = _install_runtime_mcp_connector(
        agent_id,
        context_required=False,
        secret_required=True,
    )

    def resolver(_request):
        return None

    set_connector_runtime_resolver_for_testing(resolver)
    try:
        created = client.post(
            f"/api/agents/{agent_id}/triggers",
            headers=headers,
            json={
                "type": "scheduled",
                "name": "Resolver missing delegated token",
                "config": {"interval_seconds": 60},
            },
        )
        assert created.status_code == 200, created.text
        trigger_id = created.json()["id"]

        db = _direct_db_session()
        try:
            trigger = db.query(AgentTrigger).filter(AgentTrigger.id == trigger_id).one()
            trigger.next_run_at = datetime.now(timezone.utc) - timedelta(seconds=5)
            db.add(trigger)
            db.commit()

            runs = scan_due_scheduled_triggers(db, now=datetime.now(timezone.utc))
            assert len(runs) == 1
            run = db.query(TriggerRun).filter(TriggerRun.id == runs[0].id).one()
            assert run.status == TriggerRunStatus.PENDING.value
            assert run.task_id is not None
            task = db.query(Task).filter(Task.id == run.task_id).one()

            with pytest.raises(Exception) as exc_info:
                load_connector_runtime_view(
                    db=db,
                    task_id=int(task.id),
                    turn_id="scheduled-turn",
                    user_id=int(task.user_id),
                )
        finally:
            db.close()
    finally:
        set_connector_runtime_resolver_for_testing(None)

    assert getattr(exc_info.value, "code", None) == "scheduled_secret_unavailable"
    assert getattr(exc_info.value, "details", {}).get("reason") == "not_provided"
    assert getattr(exc_info.value, "details", {}).get("connector_ref") == {
        "connector_type": "mcp",
        "connector_id": server_id,
    }


def test_dispatch_claims_pending_trigger_run_once_under_concurrency() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "scheduled",
            "name": "Concurrent scheduled",
            "config": {"interval_seconds": 60},
        },
    )
    assert created.status_code == 200, created.text

    db = _direct_db_session()
    try:
        trigger = (
            db.query(AgentTrigger).filter(AgentTrigger.id == created.json()["id"]).one()
        )
        trigger.next_run_at = datetime.now(timezone.utc) - timedelta(seconds=5)
        db.add(trigger)
        db.commit()
        runs = scan_due_scheduled_triggers(db, now=datetime.now(timezone.utc))
        assert len(runs) == 1
        run_id = int(runs[0].id)
    finally:
        db.close()

    begin_calls = 0

    async def fake_begin_turn(**kwargs):
        nonlocal begin_calls
        begin_calls += 1
        await asyncio.sleep(0.05)

        async def done() -> None:
            return None

        return TurnStarted(
            task_id=int(kwargs["task_id"]),
            status=TaskStatus.RUNNING,
            updated_at=None,
            before_message_id=None,
            task_source="trigger",
            background_task=asyncio.create_task(done()),
        )

    async def start_twice() -> list[bool]:
        first, second = await asyncio.gather(
            _start_prepared_trigger_run_id(run_id),
            _start_prepared_trigger_run_id(run_id),
        )
        return [first, second]

    with patch(
        "xagent.web.services.triggers.TaskTurnOrchestrator.begin_turn",
        new=fake_begin_turn,
    ):
        results = asyncio.run(start_twice())

    assert results.count(True) == 1
    assert begin_calls == 1


def test_scheduled_scan_disables_one_shot_trigger(mock_bg_scheduler) -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    due_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "scheduled",
            "name": "One shot",
            "config": {"next_run_at": due_at.isoformat()},
        },
    )
    assert created.status_code == 200, created.text
    trigger_id = created.json()["id"]

    db = _direct_db_session()
    try:
        runs = scan_due_scheduled_triggers(db, now=datetime.now(timezone.utc))
        assert len(runs) == 1

        trigger = db.query(AgentTrigger).filter(AgentTrigger.id == trigger_id).one()
        assert trigger.enabled is False
        assert trigger.next_run_at is None
        run = db.query(TriggerRun).filter(TriggerRun.id == runs[0].id).one()
        assert run.status == TriggerRunStatus.PENDING.value
    finally:
        db.close()

    assert mock_bg_scheduler.call_count == 0


def test_finish_turn_syncs_trigger_run_status() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={"type": "webhook", "name": "Completion webhook"},
    )
    trigger_id = created.json()["id"]

    fired = client.post(
        f"/api/agents/{agent_id}/triggers/{trigger_id}/test",
        headers=headers,
        json={"payload": {"subject": "done"}},
    )
    run_body = fired.json()["trigger_run"]

    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == run_body["task_id"]).one()
        task.status = TaskStatus.COMPLETED
        db.add(
            TaskChatMessage(
                task_id=int(task.id),
                user_id=int(task.user_id),
                role="assistant",
                content="done",
                message_type="assistant_message",
            )
        )
        db.add(task)
        db.commit()

        finish_turn(db, int(task.id))

        run = db.query(TriggerRun).filter(TriggerRun.id == run_body["id"]).one()
        assert run.status == TriggerRunStatus.COMPLETED.value
        assert run.finished_at is not None
        assert run.error_message is None
    finally:
        db.close()
