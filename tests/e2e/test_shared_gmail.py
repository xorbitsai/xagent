"""Gmail push -> signature verification -> mail fetch -> shared execution.

Connected OAuth/watch state is fixture data. Only Google certificate and Gmail
API responses are replaced; callback verification and ingestion remain real.
"""

import base64
import json
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jose import jwt

pytestmark = pytest.mark.e2e


def install_gmail_network_boundary(root):
    from xagent.web.services import gmail_triggers
    from xagent.web.services.trigger_providers import gmail

    def certificates(*args, **kwargs):
        return SimpleNamespace(
            status=200,
            data=json.dumps(
                {"e2e": (root / "gmail-public-key.pem").read_text()}
            ).encode(),
        )

    gmail._GOOGLE_AUTH_REQUEST = certificates
    service = Mock()
    service.users.return_value.history.return_value.list.return_value.execute.return_value = {
        "history": [{"messagesAdded": [{"message": {"id": "happy-mail"}}]}],
    }
    service.users.return_value.messages.return_value.get.return_value.execute.return_value = {
        "id": "happy-mail",
        "threadId": "happy-thread",
        "labelIds": ["INBOX"],
        "snippet": "e2e:delegate Gmail happy path",
        "payload": {
            "headers": [
                {"name": "From", "value": "sender@example.com"},
                {"name": "Subject", "value": "Gmail happy path"},
            ]
        },
    }
    gmail_triggers.build_gmail_service = lambda db, oauth: service


@pytest.mark.parametrize("owner", ["agent", "workforce"])
def test_gmail_push_runs_to_completion(shared_app, owner):
    from xagent.web.models.database import get_session_local
    from xagent.web.models.gmail_watch import GmailWatchState
    from xagent.web.models.trigger import AgentTrigger, TriggerRun
    from xagent.web.models.user_oauth import UserOAuth
    from xagent.web.models.workforce import WorkforceRun

    app = shared_app
    agent_id, _ = app.create_agent()
    ownership = {"agent_id": agent_id}
    if owner == "workforce":
        child_id, _ = app.create_agent("Gmail helper")
        for item in [agent_id, child_id]:
            response = app.client.post(
                f"/api/agents/{item}/publish", headers=app.headers
            )
            assert response.status_code == 200, response.text
        response = app.client.post(
            "/api/workforces",
            headers=app.headers,
            json={
                "name": "Gmail workforce",
                "manager_agent_id": agent_id,
                "workers": [
                    {
                        "source_type": "existing",
                        "agent_id": child_id,
                        "alias": "helper",
                        "assignment_instructions": "Handle the email",
                        "enabled": True,
                    }
                ],
            },
        )
        assert response.status_code == 200, response.text
        workforce_id = response.json()["id"]
        response = app.client.post(
            f"/api/workforces/{workforce_id}/publish", headers=app.headers
        )
        assert response.status_code == 200, response.text
        ownership = {"workforce_id": workforce_id}
    audience = "https://gmail.example.test/api/triggers/callback/gmail/happy-mailbox"
    with get_session_local()() as db:
        oauth = UserOAuth(
            user_id=app.user_id,
            provider="gmail",
            access_token="test-token",
            email="happy@example.com",
            provider_user_id="happy-mailbox",
        )
        db.add(oauth)
        db.flush()
        trigger = AgentTrigger(
            user_id=app.user_id,
            **ownership,
            type="gmail",
            provider="gmail",
            name="Gmail happy path",
            enabled=True,
            resource_id="happy@example.com",
            config={"watch_label": "INBOX", "oauth_account_id": oauth.id},
            prompt_template="Handle {{payload}}",
            provisioning_status="active",
        )
        db.add(trigger)
        db.add(
            GmailWatchState(
                user_id=app.user_id,
                oauth_account_id=oauth.id,
                email="happy@example.com",
                history_id="100",
                topic_name="projects/test/topics/gmail",
                callback_id="happy-mailbox",
                push_audience=audience,
                status="active",
                watch_expiration=datetime.now(timezone.utc) + timedelta(days=6),
            )
        )
        db.commit()
        trigger_id = trigger.id
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    (app.root / "gmail-public-key.pem").write_bytes(
        key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    app.close()
    app.processes.clear()
    app.pipes.clear()
    app.environment.update(
        {
            "XAGENT_GMAIL_WATCH_ENABLED": "true",
            "XAGENT_GMAIL_PUBSUB_PUSH_SERVICE_ACCOUNT": "push@example.com",
        }
    )
    app.start("web")
    app.start("worker")
    now = int(time.time())
    token = jwt.encode(
        {
            "iss": "https://accounts.google.com",
            "aud": audience,
            "sub": "happy-push",
            "email": "push@example.com",
            "email_verified": True,
            "iat": now,
            "exp": now + 300,
        },
        private_pem,
        algorithm="RS256",
        headers={"kid": "e2e"},
    )
    data = base64.b64encode(
        json.dumps({"emailAddress": "happy@example.com", "historyId": "222"}).encode()
    ).decode()
    response = app.client.post(
        "/api/triggers/callback/gmail/happy-mailbox",
        headers={"Authorization": f"Bearer {token}"},
        json={"message": {"data": data, "messageId": "happy-push"}},
    )
    assert response.status_code == 200, response.text
    assert response.json()["outcome"] == "accepted", response.text
    assert len(response.json()["trigger_run_ids"]) == 1
    deadline = time.monotonic() + 30
    task_id = None
    while time.monotonic() < deadline:
        with get_session_local()() as db:
            run = db.query(TriggerRun).filter_by(trigger_id=trigger_id).one()
            task_id = run.task_id
        if task_id:
            break
        time.sleep(0.03)
    assert task_id, app.diagnostics()
    assert "Shared E2E answer" in app.wait_task(task_id)["output"]
    with get_session_local()() as db:
        assert (
            db.query(TriggerRun).filter_by(trigger_id=trigger_id).one().status
            == "completed"
        )
        assert db.query(GmailWatchState).one().history_id == "222"
        if owner == "workforce":
            assert (
                db.query(WorkforceRun).filter_by(task_id=task_id).one().status
                == "completed"
            )
    calls = [
        json.loads(line)
        for path in app.root.glob("model-*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert {call["role"] for call in calls} == {"worker"}
    assert any("Gmail happy path" in json.dumps(call["messages"]) for call in calls)
    if owner == "workforce":
        assert any(
            any(m.get("tool_call_id") == "e2e-delegation" for m in call["messages"])
            for call in calls
        )
