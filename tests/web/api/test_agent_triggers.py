from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError

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
    _coerce_utc,
    _compute_next_run_at,
    _finish_trigger_run_after_task,
    _PreparedTriggerStart,
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
            if final_state[4] is not None:
                assert final_state[0] in {
                    TriggerRunStatus.COMPLETED.value,
                    TriggerRunStatus.FAILED.value,
                }, f"finished_at set on a non-terminal run: {final_state}"
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


def _fire_test_run(headers: dict[str, str]) -> _PreparedTriggerStart:
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={"type": "webhook", "name": "Finalizer invariant"},
    )
    assert created.status_code == 200, created.text
    trigger_id = int(created.json()["id"])
    fired = client.post(
        f"/api/agents/{agent_id}/triggers/{trigger_id}/test",
        headers=headers,
        json={"payload": {"subject": "finalize"}},
    )
    assert fired.status_code == 200, fired.text
    run_body = fired.json()["trigger_run"]
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == run_body["task_id"]).one()
        owner_user_id = int(task.user_id)
    finally:
        db.close()
    return _PreparedTriggerStart(
        run_id=int(run_body["id"]),
        trigger_id=trigger_id,
        task_id=int(run_body["task_id"]),
        task_owner_user_id=owner_user_id,
        prompt="finalize",
        trigger_type="webhook",
        test=True,
    )


def _set_task_status(
    task_id: int, status: TaskStatus, error_message: str | None = None
) -> None:
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        task.status = status
        task.error_message = error_message
        db.add(task)
        db.commit()
    finally:
        db.close()


@pytest.mark.parametrize(
    "non_terminal",
    [
        TaskStatus.PENDING,
        TaskStatus.RUNNING,
        TaskStatus.PAUSED,
        TaskStatus.WAITING_FOR_USER,
    ],
)
def test_finish_trigger_run_after_task_leaves_non_terminal_run_alone(
    non_terminal: TaskStatus,
) -> None:
    headers = _admin_headers()
    start = _fire_test_run(headers)
    _set_task_status(start.task_id, non_terminal)

    _finish_trigger_run_after_task(start)

    db = _direct_db_session()
    try:
        run = db.query(TriggerRun).filter(TriggerRun.id == start.run_id).one()
        assert run.status == TriggerRunStatus.RUNNING.value
        assert run.error_message is None
        assert run.finished_at is None
    finally:
        db.close()


@pytest.mark.parametrize(
    ("terminal", "expected_run_status", "expected_error"),
    [
        (TaskStatus.COMPLETED, TriggerRunStatus.COMPLETED.value, None),
        (TaskStatus.FAILED, TriggerRunStatus.FAILED.value, "boom"),
    ],
)
def test_finish_trigger_run_after_task_finalizes_terminal_run(
    terminal: TaskStatus,
    expected_run_status: str,
    expected_error: str | None,
) -> None:
    headers = _admin_headers()
    start = _fire_test_run(headers)
    _set_task_status(start.task_id, terminal, error_message=expected_error)

    _finish_trigger_run_after_task(start)

    db = _direct_db_session()
    try:
        run = db.query(TriggerRun).filter(TriggerRun.id == start.run_id).one()
        assert run.status == expected_run_status
        assert run.error_message == expected_error
        assert run.finished_at is not None
    finally:
        db.close()


def test_trigger_run_preparation_wraps_raising_team_hook_without_leaking_message() -> (
    None
):
    """``_attach_task_to_trigger_run`` -> ``prepare_create_connector_runtime`` ->
    ``_load_visible_runtime_connectors`` invokes the team-visibility hook, and
    ``prepare_trigger_run``'s ``except Exception`` handler stores whatever
    exception reaches it verbatim (``error_message = f"{type(exc).__name__}:
    {exc}"``). The typed wrap at the hook call is what keeps a raising hook's
    raw message out of that stored text: the hook's ``RuntimeError`` becomes a
    ``ConnectorRuntimeError`` first, so the stored message is the typed,
    public-safe one.
    """
    from xagent.web.models.trigger import AgentTrigger
    from xagent.web.services import connector_team_scope
    from xagent.web.services.triggers import (
        TriggerRunPreparationError,
        prepare_trigger_run,
    )

    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={"type": "webhook", "name": "Raising team hook webhook"},
    )
    assert created.status_code == 200, created.text
    trigger_id = int(created.json()["id"])

    db = _direct_db_session()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).one()
        agent.team_id = 101
        db.commit()

        trigger = db.query(AgentTrigger).filter(AgentTrigger.id == trigger_id).one()

        def _raising_hook(_db, *, team_id: int) -> dict[str, set[int]]:
            raise RuntimeError(
                "Bearer planted-hook-secret-must-not-leak: password authentication "
                "failed for 'svc'"
            )

        connector_team_scope.set_connector_team_hooks(team_visibility=_raising_hook)
        try:
            with pytest.raises(TriggerRunPreparationError) as excinfo:
                prepare_trigger_run(
                    db,
                    trigger=trigger,
                    event_payload={"subject": "hello"},
                    source_event_id="evt-raising-hook-1",
                )
        finally:
            connector_team_scope.set_connector_team_hooks()

        assert "planted-hook-secret-must-not-leak" not in str(excinfo.value)

        run = db.query(TriggerRun).one()
        assert str(run.status) == TriggerRunStatus.FAILED.value
        assert run.task_id is None
        assert run.error_message is not None
        assert "planted-hook-secret-must-not-leak" not in run.error_message
        assert "Connector team scope is unavailable." in run.error_message
    finally:
        db.close()


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


@pytest.mark.parametrize(
    "oauth_account_id",
    [None, True, 1.5, "not-an-account-id", 0, "0"],
)
def test_gmail_trigger_rejects_invalid_oauth_account_id(
    oauth_account_id: object,
) -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)

    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "gmail",
            "name": "Invalid account",
            "config": {
                "watch_label": "INBOX",
                "oauth_account_id": oauth_account_id,
            },
        },
    )

    assert created.status_code == 400
    assert created.json()["detail"] == (
        "gmail trigger config invalid: gmail.oauth_account_id: "
        "oauth_account_id must be a positive integer"
    )


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
    monkeypatch.setenv("XAGENT_GMAIL_WATCH_ENABLED", "true")
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


def test_listing_reports_disabled_and_expired_gmail_watch_through_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end flag-off derivation through the real endpoints: creating a
    Gmail trigger reports failed/disabled (no provisioning thread, no watch
    state row), and a leftover expired-active watch state surfaces as failed
    on the next listing instead of reading healthy (#1231)."""
    from xagent.web.models.gmail_watch import GmailWatchState

    monkeypatch.setenv("XAGENT_GMAIL_WATCH_ENABLED", "false")
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
    assert created.json()["provisioning_status"] == "failed"
    assert "disabled" in str(created.json()["provisioning_error"])

    # Simulate a watch left over from when the flag was on, already expired.
    db = _direct_db_session()
    try:
        user = db.query(User).filter(User.username == "admin").one()
        assert db.query(GmailWatchState).count() == 0
        db.add(
            GmailWatchState(
                user_id=int(user.id),
                oauth_account_id=account_id,
                email="owner@gmail.example",
                history_id="hist-1",
                topic_name="projects/demo/topics/xagent-gmail-abc",
                status=TriggerProvisioningStatus.ACTIVE.value,
                watch_expiration=datetime.now(timezone.utc) - timedelta(minutes=1),
            )
        )
        db.commit()
    finally:
        db.close()

    listed = client.get(f"/api/agents/{agent_id}/triggers", headers=headers)
    assert listed.status_code == 200, listed.text
    assert listed.json()[0]["provisioning_status"] == "failed"
    assert "expired" in str(listed.json()[0]["provisioning_error"])


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


@pytest.mark.parametrize("operation", ["disable", "delete", "rebind"])
def test_gmail_legacy_trigger_releases_previous_mailbox(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
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
    )
    monkeypatch.setattr(
        "xagent.web.services.trigger_providers.gmail.release_gmail_mailbox_if_unused",
        fake_release_gmail_mailbox_if_unused,
    )
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    first_email = "legacy-first@gmail.example"
    first_account_id = _connect_gmail_account(email=first_email)
    second_account_id = _connect_gmail_account(email="legacy-second@gmail.example")
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "gmail",
            "name": "Legacy inbox",
            "config": {
                "watch_label": "INBOX",
                "oauth_account_id": first_account_id,
            },
        },
    )
    assert created.status_code == 200, created.text
    trigger_id = int(created.json()["id"])

    db = _direct_db_session()
    try:
        trigger = db.get(AgentTrigger, trigger_id)
        assert trigger is not None
        trigger.config = {"watch_label": "INBOX"}
        trigger.resource_id = first_email
        db.commit()
    finally:
        db.close()

    if operation == "delete":
        response = client.delete(
            f"/api/agents/{agent_id}/triggers/{trigger_id}",
            headers=headers,
        )
    else:
        updates = (
            {"enabled": False}
            if operation == "disable"
            else {
                "config": {
                    "watch_label": "INBOX",
                    "oauth_account_id": second_account_id,
                }
            }
        )
        response = client.patch(
            f"/api/agents/{agent_id}/triggers/{trigger_id}",
            headers=headers,
            json=updates,
        )

    assert response.status_code == 200, response.text
    assert released == [first_account_id]
    expected_provisioned = (
        [first_account_id, second_account_id]
        if operation == "rebind"
        else [first_account_id]
    )
    assert provisioned == expected_provisioned


def test_gmail_trigger_update_still_provisions_new_binding_when_previous_unregister_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # PR #1051 review, F8: the new binding is committed BEFORE the previous
    # one is torn down, with no rollback. An unguarded raise from that
    # teardown (e.g. a DB error in release_gmail_mailbox_if_unused's own
    # unprotected lookups) used to propagate out of the whole update,
    # skipping registration of the NEW binding entirely — leaving the
    # trigger pointing at its new config with no working watch on either
    # account. The fix must still register the new binding and return a
    # clean response instead of a 500.
    provisioned: list[int] = []

    def fake_provision_gmail_trigger(db, trigger: AgentTrigger) -> str:
        provisioned.append(int(trigger.config["oauth_account_id"]))
        setattr(trigger, "provisioning_status", TriggerProvisioningStatus.ACTIVE.value)
        setattr(trigger, "provisioning_error", None)
        db.add(trigger)
        db.commit()
        return TriggerProvisioningStatus.ACTIVE.value

    def fake_release_gmail_mailbox_if_unused(db, oauth_account_id: int) -> bool:
        raise RuntimeError("simulated DB failure releasing the previous mailbox")

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
    first_account_id = _connect_gmail_account(email="first-f8@gmail.example")
    second_account_id = _connect_gmail_account(email="second-f8@gmail.example")
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
    # The new binding was still registered despite the teardown failure.
    assert provisioned == [first_account_id, second_account_id]
    assert patched.json()["provisioning_status"] == "active"
    # Self-review follow-up: a clean "active" status would otherwise
    # silently erase all trace of the teardown failure — the residual leak
    # (the old mailbox's watch was never released) must stay visible in
    # provisioning_error even though the trigger itself is genuinely usable.
    assert "releasing the previous binding failed" in (
        patched.json()["provisioning_error"] or ""
    )


def test_gmail_trigger_update_rolls_back_session_before_persisting_teardown_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Self-review follow-up on F8: the except block must call db.rollback()
    # before reusing the session for its own commit. Without it, a genuine
    # DB-level failure (as opposed to the plain RuntimeError this test
    # injects, which never touches the session) would leave the session's
    # transaction unusable on Postgres, and the except block's own commit
    # would itself raise — defeating the fix's entire purpose. This test
    # verifies the code actually calls rollback(), independent of whether
    # the injected failure happens to need it.
    from sqlalchemy.orm import Session as SASession

    rollback_calls: list[int] = []
    original_rollback = SASession.rollback

    def spying_rollback(self, *args, **kwargs):
        rollback_calls.append(1)
        return original_rollback(self, *args, **kwargs)

    monkeypatch.setattr(SASession, "rollback", spying_rollback)

    def fake_provision_gmail_trigger(db, trigger: AgentTrigger) -> str:
        setattr(trigger, "provisioning_status", TriggerProvisioningStatus.ACTIVE.value)
        setattr(trigger, "provisioning_error", None)
        db.add(trigger)
        db.commit()
        return TriggerProvisioningStatus.ACTIVE.value

    def fake_release_gmail_mailbox_if_unused(db, oauth_account_id: int) -> bool:
        raise RuntimeError("simulated DB failure releasing the previous mailbox")

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
    first_account_id = _connect_gmail_account(email="first-rollback@gmail.example")
    second_account_id = _connect_gmail_account(email="second-rollback@gmail.example")
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
    rollback_calls.clear()

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
    assert len(rollback_calls) >= 1


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


def test_gmail_trigger_delete_still_succeeds_when_unregister_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Self-review follow-up: _delete_trigger calls _unregister_trigger_binding
    # AFTER the trigger row is already deleted and committed, with no guard —
    # the same pattern _apply_trigger_updates was fixed to protect against.
    # An unguarded raise here would report a failed delete to the client even
    # though the trigger was actually removed.
    def fake_provision_gmail_trigger(db, trigger: AgentTrigger) -> str:
        setattr(trigger, "provisioning_status", TriggerProvisioningStatus.ACTIVE.value)
        setattr(trigger, "provisioning_error", None)
        db.add(trigger)
        db.commit()
        return TriggerProvisioningStatus.ACTIVE.value

    def fake_release_gmail_mailbox_if_unused(db, oauth_account_id: int) -> bool:
        raise RuntimeError("simulated DB failure releasing the mailbox")

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
    account_id = _connect_gmail_account(email="delete-teardown-fails@gmail.example")
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
    trigger_id = created.json()["id"]

    deleted = client.delete(
        f"/api/agents/{agent_id}/triggers/{trigger_id}",
        headers=headers,
    )

    assert deleted.status_code == 200, deleted.text
    listed = client.get(f"/api/agents/{agent_id}/triggers", headers=headers)
    assert listed.status_code == 200, listed.text
    assert all(trigger["id"] != trigger_id for trigger in listed.json())


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

        async def unregister(
            self, db, trigger, config, *, resource_id: str | None = None
        ) -> None:
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


def test_scheduled_weekly_picks_earliest_selected_weekday() -> None:
    # 2026-01-01 is a Thursday (weekday()==3).
    base = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
    # Mon=0, Wed=2, Fri=4 selected; the next one after Thursday 09:00 is
    # Friday 08:00 (same day, earlier time) -> still counts as after base? No:
    # candidate must be strictly after base, so Friday 08:00 on 2026-01-02 at
    # 08:00 is BEFORE 09:00 on the same day only if same date; here it's the
    # next calendar day so it is after base regardless of time-of-day.
    next_run_at = _compute_next_run_at(
        {"recurrence": "weekly", "weekdays": [0, 2, 4], "time_of_day": "08:00"},
        from_time=base,
        previous_due_at=base,
        include_explicit=False,
    )
    assert next_run_at == datetime(2026, 1, 2, 8, 0, tzinfo=timezone.utc)


def test_scheduled_weekly_skips_same_day_earlier_time() -> None:
    # 2026-01-05 is a Monday. A weekly schedule for Monday at 08:00, when the
    # last fire was also a Monday at 08:00, must advance a full week, not
    # repeat the same day.
    due_at = datetime(2026, 1, 5, 8, 0, tzinfo=timezone.utc)
    next_run_at = _compute_next_run_at(
        {"recurrence": "weekly", "weekdays": [0], "time_of_day": "08:00"},
        from_time=due_at,
        previous_due_at=due_at,
        include_explicit=False,
    )
    assert next_run_at == datetime(2026, 1, 12, 8, 0, tzinfo=timezone.utc)


def test_scheduled_weekly_honors_start_at_on_first_computation() -> None:
    base = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)  # Thursday
    next_run_at = _compute_next_run_at(
        {
            "recurrence": "weekly",
            "weekdays": [3],  # Thursday
            "time_of_day": "09:00",
            "start_at": "2026-01-15T00:00:00+00:00",
        },
        from_time=base,
    )
    # Without start_at this would be 2026-01-01 09:00 (same-day Thursday);
    # the explicit start pushes it to the next Thursday on/after 2026-01-15.
    assert next_run_at == datetime(2026, 1, 15, 9, 0, tzinfo=timezone.utc)


def test_scheduled_monthly_clamps_short_months() -> None:
    # day_of_month=31 in February clamps to the 28th (2026 is not a leap year).
    base = datetime(2026, 1, 31, 10, 0, tzinfo=timezone.utc)
    next_run_at = _compute_next_run_at(
        {"recurrence": "monthly", "day_of_month": 31, "time_of_day": "09:00"},
        from_time=base,
        previous_due_at=base,
        include_explicit=False,
    )
    assert next_run_at == datetime(2026, 2, 28, 9, 0, tzinfo=timezone.utc)


def test_scheduled_monthly_advances_a_full_month() -> None:
    due_at = datetime(2026, 3, 15, 9, 0, tzinfo=timezone.utc)
    next_run_at = _compute_next_run_at(
        {"recurrence": "monthly", "day_of_month": 15, "time_of_day": "09:00"},
        from_time=due_at,
        previous_due_at=due_at,
        include_explicit=False,
    )
    assert next_run_at == datetime(2026, 4, 15, 9, 0, tzinfo=timezone.utc)


def test_scheduled_weekly_skips_missed_occurrences_after_downtime() -> None:
    # Last fire three weeks ago (server downtime): the next run must land
    # after `now` in one computation — no catch-up burst of stale Mondays.
    now = datetime(2026, 1, 28, 12, 0, tzinfo=timezone.utc)  # Wednesday
    stale_due = datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)  # Monday, 3w ago
    next_run_at = _compute_next_run_at(
        {"recurrence": "weekly", "weekdays": [0], "time_of_day": "09:00"},
        from_time=now,
        previous_due_at=stale_due,
        include_explicit=False,
    )
    assert next_run_at == datetime(2026, 2, 2, 9, 0, tzinfo=timezone.utc)


def test_scheduled_daily_respects_schedule_timezone() -> None:
    # Daily at 09:00 in Asia/Shanghai (UTC+8) is 01:00 UTC — daily is routed
    # through the same tz-aware occurrence math as weekly/monthly, not the
    # flat interval mechanism (which never consulted timezone/time_of_day at
    # all, so a "daily at 09:00" schedule silently ran on UTC wall-clock
    # time and drifted across DST).
    base = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    next_run_at = _compute_next_run_at(
        {"recurrence": "daily", "time_of_day": "09:00", "timezone": "Asia/Shanghai"},
        from_time=base,
        previous_due_at=base,
    )
    assert next_run_at == datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)


def test_scheduled_daily_skips_missed_occurrences_after_downtime() -> None:
    # Mirrors the weekly/monthly downtime behavior: a daily schedule that
    # missed several days must land on the next occurrence after `now`, not
    # fire a catch-up burst of stale days.
    now = datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc)
    stale_due = datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)
    next_run_at = _compute_next_run_at(
        {"recurrence": "daily", "time_of_day": "09:00"},
        from_time=now,
        previous_due_at=stale_due,
        include_explicit=False,
    )
    assert next_run_at == datetime(2026, 1, 11, 9, 0, tzinfo=timezone.utc)


def test_scheduled_dst_spring_forward_gap_shifts_forward_instead_of_drifting() -> None:
    # 2026-03-08 is when America/New_York springs forward (2:00 AM -> 3:00
    # AM); a daily schedule at 2:30 AM names a local time that never occurs
    # that day. Rather than silently resolving to whatever UTC instant
    # fold=0 happens to pick, it must land on the first valid instant past
    # the gap (3:30 AM EDT == 07:30 UTC — verified directly against
    # zoneinfo's real 2026 transition).
    base = datetime(2026, 3, 8, 6, 0, tzinfo=timezone.utc)  # ~1:00 AM EST
    next_run_at = _compute_next_run_at(
        {"recurrence": "daily", "time_of_day": "02:30", "timezone": "America/New_York"},
        from_time=base,
        previous_due_at=base,
    )
    assert next_run_at == datetime(2026, 3, 8, 7, 30, tzinfo=timezone.utc)

    # An ordinary (non-gap) time on the same transition day is unaffected.
    ordinary = _compute_next_run_at(
        {"recurrence": "daily", "time_of_day": "09:00", "timezone": "America/New_York"},
        from_time=base,
        previous_due_at=base,
    )
    assert ordinary == datetime(2026, 3, 8, 13, 0, tzinfo=timezone.utc)


def test_scheduled_dst_fall_back_ambiguous_time_pins_the_first_occurrence() -> None:
    # 2026-11-01 is when America/New_York falls back (2:00 AM -> 1:00 AM);
    # 1:30 AM occurs twice that day. _localize's docstring documents that
    # the ambiguity-correction branch never fires here (round-tripping a
    # `fold=0` instant always equals itself), so Python's default fold=0 —
    # the first, still-daylight-saving occurrence — silently wins. This
    # pins that choice against zoneinfo's real 2026 transition so a future
    # change to the resolution policy is a deliberate, visible diff here.
    base = datetime(2026, 11, 1, 5, 0, tzinfo=timezone.utc)  # ~1:00 AM EDT
    next_run_at = _compute_next_run_at(
        {"recurrence": "daily", "time_of_day": "01:30", "timezone": "America/New_York"},
        from_time=base,
        previous_due_at=base,
    )
    assert next_run_at == datetime(2026, 11, 1, 5, 30, tzinfo=timezone.utc)


def test_scheduled_start_at_is_a_date_combined_via_the_configured_zone() -> None:
    # PR #1051 review: start_at must carry only a calendar DATE, never a
    # baked-in clock time — the actual first-occurrence clock time is always
    # time_of_day localized in `timezone`. A client (the frontend included)
    # has no reliable way to express "9am in Asia/Shanghai" as a bare ISO
    # instant without knowing which zone the RECEIVING process will read it
    # back in; sending just the date sidesteps that entirely.
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    next_run_at = _compute_next_run_at(
        {
            "recurrence": "daily",
            "time_of_day": "09:00",
            "timezone": "Asia/Shanghai",
            "start_at": "2026-08-01",
        },
        from_time=now,
    )
    # 09:00 Asia/Shanghai (UTC+8) on 2026-08-01 is 01:00 UTC.
    assert next_run_at == datetime(2026, 8, 1, 1, 0, tzinfo=timezone.utc)


def test_scheduled_start_at_ignores_a_legacy_time_component() -> None:
    # Backward compatibility: a full ISO datetime (the pre-fix shape, or a
    # direct API client sending one anyway) still works — only its DATE
    # portion is read; a non-midnight time component is simply discarded
    # rather than corrupting the computed anchor.
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    next_run_at = _compute_next_run_at(
        {
            "recurrence": "daily",
            "time_of_day": "09:00",
            "timezone": "Asia/Shanghai",
            "start_at": "2026-08-01T17:00:00+00:00",  # any time; only the date matters
        },
        from_time=base,
    )
    assert next_run_at == datetime(2026, 8, 1, 1, 0, tzinfo=timezone.utc)


def test_scheduled_explicit_anchor_is_honored_verbatim_past_or_future() -> None:
    # An explicit next_run_at is authoritative either way: a future anchor
    # is a genuine start time, and a past one means the trigger is already
    # due — the same semantics as enabling a cron job whose scheduled time
    # already passed. scan_due_scheduled_triggers (not this function) is
    # what decides whether "due" means "fire now". This applies to the flat
    # interval mechanism (hourly/custom, or no recurrence at all — legacy
    # configs); daily/weekly/monthly are tz-aware and always clamp to the
    # future instead (see test_scheduled_daily_never_rearms_to_the_past).
    now = datetime(2026, 1, 1, 15, 30, tzinfo=timezone.utc)
    config = {
        "interval_seconds": 86400,
        "next_run_at": "2026-01-01T09:00:00+00:00",
    }
    past = _compute_next_run_at(config, from_time=now)
    assert past == datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)

    future = _compute_next_run_at(
        {**config, "next_run_at": "2026-01-03T09:00:00+00:00"}, from_time=now
    )
    assert future == datetime(2026, 1, 3, 9, 0, tzinfo=timezone.utc)


def test_scheduled_explicit_anchor_realigns_to_interval_when_past_explicit_disallowed() -> (
    None
):
    # PR #1051 review (round 4, F1): allow_past_explicit=True (the default)
    # is only correct for a trigger's first-ever computation (create, or
    # re-enabling one with no prior armed schedule). _apply_trigger_updates
    # passes allow_past_explicit=False when recomputing because the
    # schedule was intentionally edited on an ALREADY-armed trigger, since
    # the resubmitted config still carries the untouched creation-time
    # anchor. Clamping straight to `now` (the prior behavior) armed
    # next_run_at to exactly `now`, which the next scan tick treats as
    # already due — firing an unwanted extra execution on every no-op
    # schedule resend. The fix realigns to the next interval-aligned
    # instant from the stale anchor instead.
    now = datetime(2026, 1, 1, 15, 30, tzinfo=timezone.utc)
    config = {"interval_seconds": 120, "next_run_at": "2020-01-01T00:00:00+00:00"}

    verbatim = _compute_next_run_at(config, from_time=now)
    assert verbatim == datetime(2020, 1, 1, tzinfo=timezone.utc)

    clamped = _compute_next_run_at(config, from_time=now, allow_past_explicit=False)
    assert clamped > now
    assert clamped == datetime(2026, 1, 1, 15, 32, tzinfo=timezone.utc)
    # Interval-aligned from the stale anchor, not an arbitrary future instant.
    assert (
        clamped - datetime(2020, 1, 1, tzinfo=timezone.utc)
    ).total_seconds() % 120 == 0

    # A future explicit anchor is unaffected either way — nothing to clamp.
    future_config = {**config, "next_run_at": "2026-02-01T00:00:00+00:00"}
    assert _compute_next_run_at(
        future_config, from_time=now, allow_past_explicit=False
    ) == datetime(2026, 2, 1, tzinfo=timezone.utc)


def test_scheduled_explicit_anchor_clamps_to_now_when_no_interval_and_past_explicit_disallowed() -> (
    None
):
    # A genuine one-shot (bare next_run_at, no interval_seconds — see
    # test_scheduled_scan_disables_one_shot_trigger) has no interval to
    # align to, so it keeps the original "catch up once" clamp-to-now
    # behavior when its stale anchor is resubmitted on an edit.
    now = datetime(2026, 1, 1, 15, 30, tzinfo=timezone.utc)
    config = {"next_run_at": "2020-01-01T00:00:00+00:00"}

    clamped = _compute_next_run_at(config, from_time=now, allow_past_explicit=False)
    assert clamped == now


def test_scheduled_weekly_respects_schedule_timezone() -> None:
    # Monday 09:00 in Asia/Shanghai (UTC+8) is Monday 01:00 UTC.
    base = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)  # Thursday
    next_run_at = _compute_next_run_at(
        {
            "recurrence": "weekly",
            "weekdays": [0],
            "time_of_day": "09:00",
            "timezone": "Asia/Shanghai",
        },
        from_time=base,
    )
    assert next_run_at == datetime(2026, 1, 5, 1, 0, tzinfo=timezone.utc)


def test_scheduled_weekly_timezone_decides_the_weekday() -> None:
    # Monday 20:00 in Honolulu (UTC-10) is Tuesday 06:00 UTC — the weekday
    # must be evaluated in the schedule's timezone, not UTC.
    base = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    next_run_at = _compute_next_run_at(
        {
            "recurrence": "weekly",
            "weekdays": [0],
            "time_of_day": "20:00",
            "timezone": "Pacific/Honolulu",
        },
        from_time=base,
    )
    # Next local Monday is 2026-01-05; 20:00 local = 06:00 UTC on the 6th.
    assert next_run_at == datetime(2026, 1, 6, 6, 0, tzinfo=timezone.utc)


def test_scheduled_rejects_unknown_timezone() -> None:
    from pydantic import ValidationError

    from xagent.web.services.trigger_providers.schemas import parse_trigger_config
    from xagent.web.services.triggers import TriggerServiceError

    with pytest.raises(ValidationError):
        parse_trigger_config(
            "scheduled",
            {
                "recurrence": "weekly",
                "weekdays": [0],
                "time_of_day": "09:00",
                "timezone": "Not/AZone",
            },
        )

    with pytest.raises(TriggerServiceError):
        _compute_next_run_at(
            {
                "recurrence": "weekly",
                "weekdays": [0],
                "time_of_day": "09:00",
                "timezone": "Not/AZone",
            },
            from_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )


def test_scheduled_config_validation_accepts_weekly_and_monthly() -> None:
    from xagent.web.services.trigger_providers.schemas import parse_trigger_config

    weekly = parse_trigger_config(
        "scheduled",
        {"recurrence": "weekly", "weekdays": [0, 2, 4], "time_of_day": "09:00"},
    )
    assert weekly.weekdays == [0, 2, 4]

    monthly = parse_trigger_config(
        "scheduled",
        {"recurrence": "monthly", "day_of_month": 31, "time_of_day": "09:00"},
    )
    assert monthly.day_of_month == 31


def test_scheduled_config_validation_rejects_incomplete_weekly_and_monthly() -> None:
    from pydantic import ValidationError

    from xagent.web.services.trigger_providers.schemas import parse_trigger_config

    with pytest.raises(ValidationError):
        parse_trigger_config("scheduled", {"recurrence": "weekly"})

    with pytest.raises(ValidationError):
        parse_trigger_config("scheduled", {"recurrence": "monthly"})

    with pytest.raises(ValidationError):
        parse_trigger_config("scheduled", {"recurrence": "weekly", "weekdays": [0, 7]})


def test_scheduled_config_validation_requires_time_of_day_for_calendar_recurrences() -> (
    None
):
    from pydantic import ValidationError

    from xagent.web.services.trigger_providers.schemas import parse_trigger_config

    with pytest.raises(ValidationError):
        parse_trigger_config("scheduled", {"recurrence": "daily"})
    with pytest.raises(ValidationError):
        parse_trigger_config("scheduled", {"recurrence": "weekly", "weekdays": [0]})
    with pytest.raises(ValidationError):
        parse_trigger_config("scheduled", {"recurrence": "monthly", "day_of_month": 1})

    # Present, it's accepted (daily has no other required field) — and
    # canonicalized to zero-padded "HH:MM".
    daily = parse_trigger_config(
        "scheduled", {"recurrence": "daily", "time_of_day": "09:00:00"}
    )
    assert daily.time_of_day == "09:00"


def test_scheduled_config_validation_rejects_blank_time_of_day_for_calendar_recurrences() -> (
    None
):
    # PR #1051 review, F6: the field validator used to canonicalize a blank
    # string to "00:00" BEFORE the model validator's required-check ran
    # (field validators run first), so time_of_day: "" silently satisfied
    # the "requires time_of_day" check as midnight instead of being rejected
    # as missing, for daily/weekly/monthly.
    from pydantic import ValidationError

    from xagent.web.services.trigger_providers.schemas import parse_trigger_config

    with pytest.raises(ValidationError):
        parse_trigger_config("scheduled", {"recurrence": "daily", "time_of_day": ""})
    with pytest.raises(ValidationError):
        parse_trigger_config(
            "scheduled",
            {"recurrence": "weekly", "weekdays": [0], "time_of_day": "   "},
        )


def test_scheduled_config_validation_rejects_interval_seconds_for_calendar_recurrences() -> (
    None
):
    from pydantic import ValidationError

    from xagent.web.services.trigger_providers.schemas import parse_trigger_config

    # A contradictory config — a calendar recurrence carrying the flat
    # mechanism's interval_seconds too — is rejected rather than silently
    # ignoring one of the two.
    with pytest.raises(ValidationError):
        parse_trigger_config(
            "scheduled",
            {"recurrence": "daily", "time_of_day": "09:00", "interval_seconds": 3600},
        )


def test_scheduled_config_validation_rejects_malformed_time_of_day() -> None:
    from pydantic import ValidationError

    from xagent.web.services.trigger_providers.schemas import parse_trigger_config

    with pytest.raises(ValidationError):
        parse_trigger_config(
            "scheduled", {"recurrence": "daily", "time_of_day": "not-a-time"}
        )


def test_scheduled_config_validation_rejects_malformed_start_at() -> None:
    from pydantic import ValidationError

    from xagent.web.services.trigger_providers.schemas import parse_trigger_config

    # A dedicated field validator, not just an incidental failure surfaced
    # later at recompute time as an unstructured TriggerServiceError — a
    # malformed start_at is now a clean, attributable 422 at write time.
    with pytest.raises(ValidationError):
        parse_trigger_config(
            "scheduled",
            {"recurrence": "daily", "time_of_day": "09:00", "start_at": "not-a-date"},
        )
    # A bare date and a full ISO datetime are both valid.
    parse_trigger_config(
        "scheduled",
        {"recurrence": "daily", "time_of_day": "09:00", "start_at": "2026-08-01"},
    )
    parse_trigger_config(
        "scheduled",
        {
            "recurrence": "daily",
            "time_of_day": "09:00",
            "start_at": "2026-08-01T00:00:00+00:00",
        },
    )


def test_scheduled_config_validation_rejects_calendar_only_fields_on_flat_recurrences() -> (
    None
):
    from pydantic import ValidationError

    from xagent.web.services.trigger_providers.schemas import parse_trigger_config

    # hourly/custom (and a bare next_run_at with no recurrence — the
    # deliberate one-shot shape) never read time_of_day/weekdays/
    # day_of_month/start_at/timezone; a client setting them got a 2xx and
    # silently nothing happened. Reject outright instead, symmetric with how
    # a calendar recurrence already rejects a stray interval_seconds.
    base_hourly = {"recurrence": "hourly", "interval_seconds": 3600}
    for stray_field, stray_value in (
        ("time_of_day", "09:00"),
        ("weekdays", [0]),
        ("day_of_month", 1),
        ("start_at", "2026-08-01"),
        ("timezone", "Asia/Shanghai"),
    ):
        with pytest.raises(ValidationError):
            parse_trigger_config("scheduled", {**base_hourly, stray_field: stray_value})
    # Without any of them, hourly is still valid.
    parse_trigger_config("scheduled", base_hourly)


def test_scheduled_config_validation_treats_blank_time_of_day_as_absent_for_flat_recurrences() -> (
    None
):
    # PR #1051 review, F6 residual asymmetry: omitting time_of_day entirely
    # for hourly/custom was already accepted (it's simply unused), but an
    # EXPLICIT time_of_day: "" used to be rejected as "time_of_day is not
    # used by hourly schedule" — "" is not None, so it tripped the stray-
    # field check above even though it means the same thing as omitting the
    # field. Blank must behave identically to absent here.
    from xagent.web.services.trigger_providers.schemas import parse_trigger_config

    hourly = parse_trigger_config(
        "scheduled",
        {"recurrence": "hourly", "interval_seconds": 3600, "time_of_day": ""},
    )
    assert hourly.time_of_day == ""

    custom = parse_trigger_config(
        "scheduled",
        {"recurrence": "custom", "interval_seconds": 120, "time_of_day": "   "},
    )
    assert custom.time_of_day == "   "

    # A genuinely non-blank time_of_day on a flat recurrence is still
    # rejected — only blank is tolerated as "effectively absent".
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        parse_trigger_config(
            "scheduled",
            {"recurrence": "hourly", "interval_seconds": 3600, "time_of_day": "09:00"},
        )


def test_scheduled_config_validation_caps_interval_seconds() -> None:
    # PR #1051 review, N1: an unbounded interval_seconds (e.g. 10**18)
    # overflows the alignment arithmetic in _compute_next_run_at with a bare
    # OverflowError. Reject absurd values at write time instead of only
    # coping with the fallout at scan time.
    from pydantic import ValidationError

    from xagent.web.services.trigger_providers.schemas import (
        _MAX_INTERVAL_SECONDS,
        parse_trigger_config,
    )

    with pytest.raises(ValidationError):
        parse_trigger_config(
            "scheduled",
            {"recurrence": "custom", "interval_seconds": _MAX_INTERVAL_SECONDS + 1},
        )
    with pytest.raises(ValidationError):
        parse_trigger_config("scheduled", {"interval_seconds": 10**18})
    # At the cap is still valid.
    parse_trigger_config(
        "scheduled",
        {"recurrence": "custom", "interval_seconds": _MAX_INTERVAL_SECONDS},
    )


def test_scheduled_config_validation_requires_interval_seconds_for_custom() -> None:
    from pydantic import ValidationError

    from xagent.web.services.trigger_providers.schemas import parse_trigger_config

    # "custom" (unlike a bare, recurrence-less next_run_at — the one-shot
    # shape) explicitly implies a user-picked interval; omitting it is a
    # caller error, not a valid degraded one-shot.
    with pytest.raises(ValidationError):
        parse_trigger_config(
            "scheduled",
            {"recurrence": "custom", "next_run_at": "2026-08-01T09:00:00+00:00"},
        )
    parse_trigger_config(
        "scheduled",
        {
            "recurrence": "custom",
            "interval_seconds": 120,
            "next_run_at": "2026-08-01T09:00:00+00:00",
        },
    )
    # The one-shot shape itself (no recurrence at all) remains valid.
    parse_trigger_config("scheduled", {"next_run_at": "2026-08-01T09:00:00+00:00"})


def test_scheduled_config_validation_rejects_stray_next_run_at_on_calendar_recurrences() -> (
    None
):
    from pydantic import ValidationError

    from xagent.web.services.trigger_providers.schemas import parse_trigger_config

    with pytest.raises(ValidationError):
        parse_trigger_config(
            "scheduled",
            {
                "recurrence": "daily",
                "time_of_day": "09:00",
                "next_run_at": "2026-08-01T09:00:00+00:00",
            },
        )


def test_scheduled_config_validation_rejects_unknown_fields() -> None:
    # PR #1051 review, F4: unrecognized fields used to be silently ignored
    # (pydantic's default), rather than rejected like a typo or a stale
    # client field would deserve.
    from pydantic import ValidationError

    from xagent.web.services.trigger_providers.schemas import parse_trigger_config

    with pytest.raises(ValidationError):
        parse_trigger_config(
            "scheduled",
            {"interval_seconds": 60, "not_a_real_field": "oops"},
        )


def test_scheduled_config_update_tolerates_legacy_timezone_on_non_calendar_recurrence(
    mock_bg_scheduler,
) -> None:
    # PR #1051 review, N7: a pre-existing stored config combining a
    # non-calendar recurrence (hourly/custom/a bare next_run_at) with a
    # `timezone` field was allowed before this PR's schema validation
    # existed. Round-tripping it through a direct API client (GET, then
    # PATCH back the same config alongside an unrelated field) must not now
    # 422 — the new dialog UI is unaffected since it never sends `timezone`
    # for non-calendar recurrences (buildConfig). This leniency is narrowly
    # scoped to a trigger whose OWN previously stored config already had
    # this exact legacy shape; a fresh config making the same mistake is
    # still rejected (see the second half of this test).
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "scheduled",
            "name": "Legacy hourly",
            "config": {"recurrence": "hourly", "interval_seconds": 3600},
        },
    )
    assert created.status_code == 200, created.text
    trigger_id = created.json()["id"]

    # Mutate the stored config directly (bypassing API validation) to add
    # the legacy `timezone` field, simulating a row that predates this PR's
    # schema tightening.
    db = _direct_db_session()
    try:
        trigger = db.query(AgentTrigger).filter(AgentTrigger.id == trigger_id).one()
        trigger.config = {
            "recurrence": "hourly",
            "interval_seconds": 3600,
            "timezone": "Asia/Shanghai",
        }
        db.add(trigger)
        db.commit()
    finally:
        db.close()

    # Resend the SAME legacy config unchanged, alongside an unrelated field
    # (a real Save always resends the complete config) — must not 422.
    patched = client.patch(
        f"/api/agents/{agent_id}/triggers/{trigger_id}",
        headers=headers,
        json={
            "config": {
                "recurrence": "hourly",
                "interval_seconds": 3600,
                "timezone": "Asia/Shanghai",
            },
            "prompt_template": "Say hi",
        },
    )
    assert patched.status_code == 200, patched.text

    # A BRAND NEW trigger making the same mistake (no prior legacy config to
    # justify leniency) is still rejected strictly.
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "scheduled",
            "name": "Fresh hourly",
            "config": {
                "recurrence": "hourly",
                "interval_seconds": 3600,
                "timezone": "Asia/Shanghai",
            },
        },
    )
    assert created.status_code == 400, created.text

    # Changing BOTH the recurrence AND the timezone value in the same PATCH
    # must still 422 — the leniency checks the loose non-calendar-recurrence
    # SHAPE on both sides, which this still satisfies (hourly -> custom is
    # still non-calendar), but the underlying timezone VALUE also changed
    # (Asia/Shanghai -> Pacific/Auckland). A client could otherwise keep
    # "resending a legacy shape" indefinitely while freely swapping in a
    # brand-new, never-before-stored timezone (PR #1051 review, N7 follow-up).
    repatched = client.patch(
        f"/api/agents/{agent_id}/triggers/{trigger_id}",
        headers=headers,
        json={
            "config": {
                "recurrence": "custom",
                "interval_seconds": 3600,
                "timezone": "Pacific/Auckland",
            },
        },
    )
    assert repatched.status_code == 400, repatched.text

    # Resending the exact same legacy pair unchanged (the actual case this
    # leniency exists for) must still be tolerated after the above.
    still_tolerated = client.patch(
        f"/api/agents/{agent_id}/triggers/{trigger_id}",
        headers=headers,
        json={
            "config": {
                "recurrence": "hourly",
                "interval_seconds": 3600,
                "timezone": "Asia/Shanghai",
            },
        },
    )
    assert still_tolerated.status_code == 200, still_tolerated.text


def test_normalize_weekdays_coerces_a_bare_scalar_to_a_single_day() -> None:
    from xagent.web.services.trigger_providers.schemas import normalize_weekdays

    # A bare scalar (e.g. weekdays=3, or the stringified "3") means one day,
    # not a sequence to iterate — must not be split character-by-character
    # (str) or rejected as non-iterable (int/bool).
    assert normalize_weekdays(3) == {3}
    assert normalize_weekdays("3") == {3}


def test_normalize_time_of_day_rejects_non_string_non_none_values() -> None:
    from datetime import time as time_cls

    from xagent.web.services.trigger_providers.schemas import normalize_time_of_day

    # None (truly absent) and blank both legitimately default to midnight.
    assert normalize_time_of_day(None) == time_cls(0, 0)
    assert normalize_time_of_day("") == time_cls(0, 0)
    assert normalize_time_of_day("   ") == time_cls(0, 0)

    # 0/False/[] are type errors, not "absent" — must not silently become
    # midnight (a config bug that stores the wrong type should surface, not
    # be masked as a valid "00:00" schedule).
    for bad_value in (0, False, [], {}):
        with pytest.raises(ValueError):
            normalize_time_of_day(bad_value)


def test_scheduled_config_validation_canonicalizes_time_of_day() -> None:
    from xagent.web.services.trigger_providers.schemas import parse_trigger_config

    parsed = parse_trigger_config(
        "scheduled", {"recurrence": "daily", "time_of_day": "9:5"}
    )
    assert parsed.time_of_day == "09:05"


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


def test_scheduled_unrelated_update_does_not_reset_advanced_next_run(
    mock_bg_scheduler,
) -> None:
    # Regression: PATCHing a field that has nothing to do with the schedule
    # (name/prompt/secret) must not re-derive next_run_at from the trigger's
    # stored config. The config's own "next_run_at" string is never advanced
    # by a scan (only the next_run_at column is) — recomputing from it on
    # every unrelated edit would re-arm an already-progressed schedule back
    # to its original, long-stale anchor.
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "scheduled",
            "name": "Every minute",
            "config": {
                "interval_seconds": 60,
                "next_run_at": "2020-01-01T00:00:00+00:00",
            },
        },
    )
    assert created.status_code == 200, created.text
    trigger_id = created.json()["id"]

    db = _direct_db_session()
    try:
        trigger = db.query(AgentTrigger).filter(AgentTrigger.id == trigger_id).one()
        # The explicit past anchor is honored verbatim at create time.
        assert _coerce_utc(trigger.next_run_at) == datetime(
            2020, 1, 1, tzinfo=timezone.utc
        )

        runs = scan_due_scheduled_triggers(db, now=datetime.now(timezone.utc))
        assert len(runs) == 1
        db.refresh(trigger)
        advanced_next_run_at = _coerce_utc(trigger.next_run_at)
        assert advanced_next_run_at is not None
        assert advanced_next_run_at > datetime(2020, 1, 1, tzinfo=timezone.utc)
    finally:
        db.close()

    patched = client.patch(
        f"/api/agents/{agent_id}/triggers/{trigger_id}",
        headers=headers,
        json={"name": "Every minute (renamed)"},
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["name"] == "Every minute (renamed)"
    # The stored config's stale anchor is untouched, but the schedule's
    # advanced next_run_at must survive the unrelated rename.
    assert _coerce_utc(datetime.fromisoformat(patched.json()["next_run_at"])) == (
        advanced_next_run_at
    )


def test_scheduled_full_form_save_with_unchanged_config_does_not_reset_next_run_at(
    mock_bg_scheduler,
) -> None:
    # The real editor's Save always resends the COMPLETE config, including
    # the original creation-time anchor, whether or not the user touched the
    # schedule (PR #1051 review: gating the "don't rewind" guard on whether
    # `config` appeared in the request body at all — rather than on whether
    # it actually CHANGED — made that guard unreachable from this exact
    # flow, since a real Save's payload always contains a `config` key).
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    original_config = {
        "interval_seconds": 60,
        "next_run_at": "2020-01-01T00:00:00+00:00",
    }
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "scheduled",
            "name": "Every minute",
            "config": original_config,
        },
    )
    assert created.status_code == 200, created.text
    trigger_id = created.json()["id"]

    db = _direct_db_session()
    try:
        scan_due_scheduled_triggers(db, now=datetime.now(timezone.utc))
        db.refresh(db.query(AgentTrigger).filter(AgentTrigger.id == trigger_id).one())
        trigger = db.query(AgentTrigger).filter(AgentTrigger.id == trigger_id).one()
        advanced_next_run_at = _coerce_utc(trigger.next_run_at)
        assert advanced_next_run_at is not None
        assert advanced_next_run_at > datetime(2020, 1, 1, tzinfo=timezone.utc)
    finally:
        db.close()

    # Resend the SAME config unchanged, exactly like a full-form Save whose
    # user only touched the prompt template — must not re-arm next_run_at
    # back to the stale 2020 anchor still sitting in the config.
    patched = client.patch(
        f"/api/agents/{agent_id}/triggers/{trigger_id}",
        headers=headers,
        json={"config": dict(original_config), "prompt_template": "Say hi"},
    )
    assert patched.status_code == 200, patched.text
    assert _coerce_utc(datetime.fromisoformat(patched.json()["next_run_at"])) == (
        advanced_next_run_at
    )

    # A genuinely changed config (the user actually edits the schedule) must
    # still recompute.
    before_repatch = datetime.now(timezone.utc)
    repatched = client.patch(
        f"/api/agents/{agent_id}/triggers/{trigger_id}",
        headers=headers,
        json={"config": {**original_config, "interval_seconds": 120}},
    )
    assert repatched.status_code == 200, repatched.text
    recomputed = _coerce_utc(datetime.fromisoformat(repatched.json()["next_run_at"]))
    # PR #1051 review: asserting only `!= advanced_next_run_at` is satisfied
    # by the bug this test exists to catch — the un-clamped recompute
    # returns the config's stale 2020 anchor verbatim, which trivially
    # differs from `advanced_next_run_at` too. Assert it's actually no
    # earlier than the request itself, which fails without the
    # allow_past_explicit=False clamp in _apply_trigger_updates.
    assert recomputed is not None
    assert recomputed >= before_repatch


def test_scheduled_start_at_backfill_alone_does_not_reset_next_run_at(
    mock_bg_scheduler,
) -> None:
    # PR #1051 review, N8: the editor's F5 fix defaults a reconstructed
    # calendar trigger's startDate to today when no start_at is stored at
    # all. Resaving with literally zero other user changes then resends
    # start_at="<today>" where the stored config previously had none — a
    # genuine diff by _schedule_signature's ordinary comparison, which would
    # otherwise force a recompute from today and can move next_run_at
    # EARLIER than its current, already-computed value purely as a side
    # effect of what looks like a no-op Save. Only a start_at that's a REAL
    # pre-existing value, or a date other than today, is a genuine edit.
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "scheduled",
            "name": "Every day",
            "config": {
                "recurrence": "weekly",
                "time_of_day": "09:00",
                "weekdays": [0, 1, 2, 3, 4, 5, 6],
            },
        },
    )
    assert created.status_code == 200, created.text
    trigger_id = created.json()["id"]
    original_next_run_at = _coerce_utc(
        datetime.fromisoformat(created.json()["next_run_at"])
    )
    assert original_next_run_at is not None

    today = datetime.now(timezone.utc).date().isoformat()
    patched = client.patch(
        f"/api/agents/{agent_id}/triggers/{trigger_id}",
        headers=headers,
        json={
            "config": {
                "recurrence": "weekly",
                "time_of_day": "09:00",
                "weekdays": [0, 1, 2, 3, 4, 5, 6],
                "start_at": today,
            },
            "prompt_template": "Say hi",
        },
    )
    assert patched.status_code == 200, patched.text
    assert (
        _coerce_utc(datetime.fromisoformat(patched.json()["next_run_at"]))
        == original_next_run_at
    )

    # A genuine schedule edit (a start_at other than "today") still forces a
    # recompute.
    two_days_out = (datetime.now(timezone.utc) + timedelta(days=2)).date().isoformat()
    repatched = client.patch(
        f"/api/agents/{agent_id}/triggers/{trigger_id}",
        headers=headers,
        json={
            "config": {
                "recurrence": "weekly",
                "time_of_day": "09:00",
                "weekdays": [0, 1, 2, 3, 4, 5, 6],
                "start_at": two_days_out,
            }
        },
    )
    assert repatched.status_code == 200, repatched.text
    recomputed = _coerce_utc(datetime.fromisoformat(repatched.json()["next_run_at"]))
    assert recomputed is not None
    assert recomputed > original_next_run_at


def test_scheduled_start_at_blank_string_backfill_alone_does_not_reset_next_run_at(
    mock_bg_scheduler,
) -> None:
    # PR #1051 review, N8 follow-up: _is_benign_start_at_backfill's "was the
    # old start_at truly absent" check used a bare `is not None`, which misses
    # a literal stored start_at="" (present key, blank value) — treated
    # identically to a wholly-missing key by _compute_next_run_at and
    # _schedule_signature themselves (both `.strip()` it away), but not by
    # this heuristic's own check. A trigger stuck with start_at="" (e.g. an
    # older client that always sent the key) must get the same no-op-Save
    # protection as one with no start_at key at all.
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "scheduled",
            "name": "Every day",
            "config": {
                "recurrence": "weekly",
                "time_of_day": "09:00",
                "weekdays": [0, 1, 2, 3, 4, 5, 6],
            },
        },
    )
    assert created.status_code == 200, created.text
    trigger_id = created.json()["id"]
    original_next_run_at = _coerce_utc(
        datetime.fromisoformat(created.json()["next_run_at"])
    )
    assert original_next_run_at is not None

    # Directly store a literal blank start_at — simulating a client that
    # always resends the key, unlike the dialog (which omits it entirely).
    db = _direct_db_session()
    try:
        trigger = db.query(AgentTrigger).filter(AgentTrigger.id == trigger_id).one()
        trigger.config = {
            "recurrence": "weekly",
            "time_of_day": "09:00",
            "weekdays": [0, 1, 2, 3, 4, 5, 6],
            "start_at": "",
        }
        db.add(trigger)
        db.commit()
    finally:
        db.close()

    today = datetime.now(timezone.utc).date().isoformat()
    patched = client.patch(
        f"/api/agents/{agent_id}/triggers/{trigger_id}",
        headers=headers,
        json={
            "config": {
                "recurrence": "weekly",
                "time_of_day": "09:00",
                "weekdays": [0, 1, 2, 3, 4, 5, 6],
                "start_at": today,
            },
            "prompt_template": "Say hi",
        },
    )
    assert patched.status_code == 200, patched.text
    assert (
        _coerce_utc(datetime.fromisoformat(patched.json()["next_run_at"]))
        == original_next_run_at
    )


def test_scheduled_re_enable_clamps_a_stale_anchor_instead_of_rewinding(
    mock_bg_scheduler,
) -> None:
    # PR #1051 review: the re-enable branch used to call _compute_next_run_at
    # with the default allow_past_explicit=True — same bug as a schedule
    # edit, reached via disable -> enable instead. Re-enabling isn't "like a
    # fresh creation": the stored config's anchor may be from long before
    # the trigger was ever disabled.
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "scheduled",
            "name": "Every minute",
            "enabled": False,
            "config": {
                "interval_seconds": 60,
                "next_run_at": "2020-01-01T00:00:00+00:00",
            },
        },
    )
    assert created.status_code == 200, created.text
    trigger_id = created.json()["id"]
    assert created.json()["next_run_at"] is None

    before_enable = datetime.now(timezone.utc)
    enabled = client.patch(
        f"/api/agents/{agent_id}/triggers/{trigger_id}",
        headers=headers,
        json={"enabled": True},
    )
    assert enabled.status_code == 200, enabled.text
    next_run_at = _coerce_utc(datetime.fromisoformat(enabled.json()["next_run_at"]))
    assert next_run_at is not None
    assert next_run_at >= before_enable


def test_scheduled_signature_ignores_unpadded_time_of_day_and_weekday_order(
    mock_bg_scheduler,
) -> None:
    # PR #1051 review: _validate_config persists the caller-provided config
    # verbatim (never rewrites stored JSON), so a stored "9:5" or an
    # out-of-order weekdays list is compared RAW by _schedule_signature — an
    # equivalent, differently-formatted resend must still count as "the
    # schedule didn't change".
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "scheduled",
            "name": "Weekly",
            "config": {
                "recurrence": "weekly",
                "time_of_day": "9:5",
                "weekdays": [2, 0],
            },
        },
    )
    assert created.status_code == 200, created.text
    trigger_id = created.json()["id"]
    original_next_run_at = created.json()["next_run_at"]
    assert original_next_run_at is not None

    patched = client.patch(
        f"/api/agents/{agent_id}/triggers/{trigger_id}",
        headers=headers,
        json={
            "config": {
                "recurrence": "weekly",
                "time_of_day": "09:05",  # zero-padded, same time
                "weekdays": [0, 2],  # sorted, same set
            },
        },
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["next_run_at"] == original_next_run_at


def test_scheduled_signature_ignores_start_at_date_vs_datetime_form(
    mock_bg_scheduler,
) -> None:
    # PR #1051 review, F7: _schedule_signature normalized time_of_day and
    # weekdays but left start_at raw, so a bare "YYYY-MM-DD" and an
    # equivalent midnight-instant ISO string for the SAME date registered as
    # a schedule "change" — _compute_next_run_at only ever reads the
    # calendar DATE of start_at, so they mean the same schedule.
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "scheduled",
            "name": "Daily with start_at",
            "config": {
                "recurrence": "daily",
                "time_of_day": "09:00",
                "start_at": "2026-01-01",
            },
        },
    )
    assert created.status_code == 200, created.text
    trigger_id = created.json()["id"]
    original_next_run_at = created.json()["next_run_at"]
    assert original_next_run_at is not None

    patched = client.patch(
        f"/api/agents/{agent_id}/triggers/{trigger_id}",
        headers=headers,
        json={
            "config": {
                "recurrence": "daily",
                "time_of_day": "09:00",
                "start_at": "2026-01-01T00:00:00",  # same calendar date, full ISO
            },
        },
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["next_run_at"] == original_next_run_at


def test_scheduled_signature_ignores_interval_seconds_string_vs_int_form(
    mock_bg_scheduler,
) -> None:
    # PR #1051 review, F7: interval_seconds may arrive as an int or a
    # numeral string; comparing them raw registered "3600" vs 3600 as a
    # schedule "change" even though int(value) is identical either way.
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "scheduled",
            "name": "Hourly",
            "config": {"recurrence": "hourly", "interval_seconds": 3600},
        },
    )
    assert created.status_code == 200, created.text
    trigger_id = created.json()["id"]
    original_next_run_at = created.json()["next_run_at"]
    assert original_next_run_at is not None

    patched = client.patch(
        f"/api/agents/{agent_id}/triggers/{trigger_id}",
        headers=headers,
        json={"config": {"recurrence": "hourly", "interval_seconds": "3600"}},
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["next_run_at"] == original_next_run_at


def test_scheduled_signature_ignores_blank_time_of_day_vs_absent_for_flat_recurrences(
    mock_bg_scheduler,
) -> None:
    # PR #1051 review, N follow-up: _schedule_signature only normalized
    # time_of_day `if value is not None`, so a resent time_of_day="" (accepted
    # for hourly/custom since the F6 fix — previously rejected) canonicalized
    # to "00:00" while a stored config with NO time_of_day key at all stayed
    # None — a spurious signature mismatch, and an unwanted recompute, for a
    # direct API client resending "" where the stored config never had the
    # key in the first place.
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "scheduled",
            "name": "Custom interval",
            "config": {"recurrence": "custom", "interval_seconds": 120},
        },
    )
    assert created.status_code == 200, created.text
    trigger_id = created.json()["id"]
    original_next_run_at = created.json()["next_run_at"]
    assert original_next_run_at is not None

    patched = client.patch(
        f"/api/agents/{agent_id}/triggers/{trigger_id}",
        headers=headers,
        json={
            "config": {
                "recurrence": "custom",
                "interval_seconds": 120,
                "time_of_day": "",
            },
        },
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["next_run_at"] == original_next_run_at


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


def test_scheduled_scan_disables_trigger_with_unrecomputable_config_instead_of_wedging(
    mock_bg_scheduler,
) -> None:
    # PR #1051 review, F3: scan_due_scheduled_triggers queries due triggers
    # ordered by next_run_at ASC. An unguarded _compute_next_run_at raise for
    # one trigger (e.g. config drift after this PR's schema tightening) used
    # to propagate out of the whole scan call, leaving that trigger's
    # next_run_at unadvanced — permanently first in line — and blocking every
    # trigger ordered after it on every subsequent tick. The fix disables the
    # unrecomputable trigger instead and keeps scanning the rest of the batch.
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    poisoned = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "scheduled",
            "name": "Poisoned weekly",
            "config": {
                "recurrence": "weekly",
                "time_of_day": "09:00",
                "weekdays": [0],
            },
        },
    )
    assert poisoned.status_code == 200, poisoned.text
    poisoned_id = poisoned.json()["id"]

    healthy = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "scheduled",
            "name": "Every minute",
            "config": {"interval_seconds": 60},
        },
    )
    assert healthy.status_code == 200, healthy.text
    healthy_id = healthy.json()["id"]

    now = datetime.now(timezone.utc)
    db = _direct_db_session()
    try:
        poisoned_trigger = (
            db.query(AgentTrigger).filter(AgentTrigger.id == poisoned_id).one()
        )
        # Corrupt the stored config directly (bypassing API validation, like
        # config drift or a manual DB edit would) so recompute raises:
        # weekdays is now empty, which normalize_weekdays rejects.
        poisoned_trigger.config = {
            "recurrence": "weekly",
            "time_of_day": "09:00",
            "weekdays": [],
        }
        # Ordered first: an earlier next_run_at than the healthy trigger.
        poisoned_trigger.next_run_at = now - timedelta(minutes=10)
        db.add(poisoned_trigger)

        healthy_trigger = (
            db.query(AgentTrigger).filter(AgentTrigger.id == healthy_id).one()
        )
        healthy_trigger.next_run_at = now - timedelta(seconds=5)
        db.add(healthy_trigger)
        db.commit()

        runs = scan_due_scheduled_triggers(db, now=now)
        assert len(runs) == 2

        db.refresh(poisoned_trigger)
        assert poisoned_trigger.enabled is False
        assert poisoned_trigger.next_run_at is None

        db.refresh(healthy_trigger)
        assert healthy_trigger.enabled is True
        assert healthy_trigger.next_run_at is not None
        healthy_next_run_at = healthy_trigger.next_run_at
        if healthy_next_run_at.tzinfo is None:
            healthy_next_run_at = healthy_next_run_at.replace(tzinfo=timezone.utc)
        assert healthy_next_run_at > now
    finally:
        db.close()


def test_scheduled_scan_disables_trigger_on_overflow_and_surfaces_a_reason(
    mock_bg_scheduler,
) -> None:
    # PR #1051 review, N1a/N6: an absurd interval_seconds (config drift, or
    # data predating this PR's new upper-bound validator) overflows the
    # alignment arithmetic in _compute_next_run_at with a bare OverflowError,
    # not the service's own TriggerServiceError — the scan loop's recompute
    # guard must catch that too (broadened to `except Exception`) or it
    # wedges the whole batch, same as F3/N1's TriggerServiceError case. N6:
    # the disabled trigger must also surface a reason via provisioning_error
    # (the same field the Gmail provider uses), not just a server-side log.
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "scheduled",
            "name": "Overflowing interval",
            "config": {"interval_seconds": 60},
        },
    )
    assert created.status_code == 200, created.text
    trigger_id = created.json()["id"]

    now = datetime.now(timezone.utc)
    db = _direct_db_session()
    try:
        trigger = db.query(AgentTrigger).filter(AgentTrigger.id == trigger_id).one()
        # Bypass API validation (like a manual DB edit or pre-upper-bound
        # data would) so the recompute genuinely overflows.
        trigger.config = {"interval_seconds": 10**18}
        trigger.next_run_at = now - timedelta(seconds=5)
        db.add(trigger)
        db.commit()

        runs = scan_due_scheduled_triggers(db, now=now)
        assert len(runs) == 1

        db.refresh(trigger)
        assert trigger.enabled is False
        assert trigger.next_run_at is None
        assert trigger.provisioning_status == TriggerProvisioningStatus.FAILED.value
        assert trigger.provisioning_error
    finally:
        db.close()


def test_reenable_trigger_with_pathological_stored_config_returns_clean_error(
    mock_bg_scheduler,
) -> None:
    # Third review round: a bare `{"enabled": true}` PATCH (no "config" key)
    # skips _validate_config entirely (and with it, this PR's interval_seconds
    # upper-bound check), so _apply_trigger_updates's re-enable branch calls
    # _compute_next_run_at directly on the STALE, unvalidated stored config.
    # If that config predates the cap (legacy data) and is pathological, the
    # alignment arithmetic overflows with a bare OverflowError that used to
    # propagate uncaught through the API route's generic exception handler as
    # an ugly 500. This is a synchronous user-facing request (unlike the scan
    # loop, which silently disables instead), so the fix wraps the call and
    # raises a TriggerServiceError instead — which _handle_service_error maps
    # to a clean 4xx with an actionable message.
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "scheduled",
            "name": "Legacy pathological interval",
            "config": {"interval_seconds": 60},
        },
    )
    assert created.status_code == 200, created.text
    trigger_id = created.json()["id"]

    db = _direct_db_session()
    try:
        trigger = db.query(AgentTrigger).filter(AgentTrigger.id == trigger_id).one()
        # Bypass API validation (like legacy data predating the upper-bound
        # cap would) and disable the trigger, so the stored config is both
        # unvalidated and stale by the time the re-enable PATCH arrives.
        trigger.config = {"interval_seconds": 10**18}
        trigger.enabled = False
        db.add(trigger)
        db.commit()
    finally:
        db.close()

    resp = client.patch(
        f"/api/agents/{agent_id}/triggers/{trigger_id}",
        headers=headers,
        json={"enabled": True},
    )
    assert resp.status_code in (400, 422), resp.text
    assert resp.status_code != 500
    assert resp.json()["detail"]

    db = _direct_db_session()
    try:
        trigger = db.query(AgentTrigger).filter(AgentTrigger.id == trigger_id).one()
        # The failed re-enable must not have left the trigger in some
        # half-updated state: still disabled, config untouched.
        assert trigger.enabled is False
        assert trigger.config == {"interval_seconds": 10**18}
    finally:
        db.close()


def test_scheduled_scan_continues_batch_after_unexpected_prepare_failure(
    mock_bg_scheduler,
) -> None:
    # PR #1051 review, N1b: prepare_trigger_run's own _get_or_create_trigger_run
    # can re-raise a bare IntegrityError (an insert race whose post-rollback
    # lookup still misses the real duplicate row) instead of the service's
    # own TriggerRunPreparationError. The scan loop's call-site guard used to
    # only catch TriggerRunPreparationError, so this propagated out of
    # scan_due_scheduled_triggers entirely, aborting the whole batch and
    # wedging every trigger ordered after it — the same class of bug as
    # N1a, at the other call site. Now it's caught, the session is rolled
    # back, and the batch continues to the next due trigger untouched (left
    # for the next scan tick to retry, since this is treated as transient,
    # not a poisoned config).
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    poisoned = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "scheduled",
            "name": "Poisoned",
            "config": {"interval_seconds": 60},
        },
    )
    assert poisoned.status_code == 200, poisoned.text
    poisoned_id = poisoned.json()["id"]

    healthy = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "scheduled",
            "name": "Healthy",
            "config": {"interval_seconds": 60},
        },
    )
    assert healthy.status_code == 200, healthy.text
    healthy_id = healthy.json()["id"]

    now = datetime.now(timezone.utc)
    poisoned_due_at = now - timedelta(minutes=10)
    db = _direct_db_session()
    try:
        poisoned_trigger = (
            db.query(AgentTrigger).filter(AgentTrigger.id == poisoned_id).one()
        )
        # Ordered first: an earlier next_run_at than the healthy trigger.
        poisoned_trigger.next_run_at = poisoned_due_at
        db.add(poisoned_trigger)

        healthy_trigger = (
            db.query(AgentTrigger).filter(AgentTrigger.id == healthy_id).one()
        )
        healthy_trigger.next_run_at = now - timedelta(seconds=5)
        db.add(healthy_trigger)
        db.commit()

        from xagent.web.services import triggers as triggers_mod

        real_prepare_trigger_run = triggers_mod.prepare_trigger_run

        def flaky_prepare_trigger_run(db_arg, *, trigger, **kwargs):
            if trigger.id == poisoned_id:
                raise IntegrityError("insert", {}, Exception("duplicate key"))
            return real_prepare_trigger_run(db_arg, trigger=trigger, **kwargs)

        with patch.object(
            triggers_mod, "prepare_trigger_run", side_effect=flaky_prepare_trigger_run
        ):
            runs = scan_due_scheduled_triggers(db, now=now)

        # Only the healthy trigger produced a run; the poisoned one's
        # unexpected failure was swallowed instead of aborting the batch.
        assert len(runs) == 1

        db.refresh(poisoned_trigger)
        assert poisoned_trigger.enabled is True
        assert _coerce_utc(poisoned_trigger.next_run_at) == poisoned_due_at

        db.refresh(healthy_trigger)
        assert healthy_trigger.enabled is True
        assert healthy_trigger.next_run_at is not None
    finally:
        db.close()


def test_scheduled_scan_continues_batch_after_encryption_key_failure(
    mock_bg_scheduler,
) -> None:
    # Third review round: prepare_trigger_run -> _get_or_create_trigger_run ->
    # _payload_snapshot -> encrypt_value -> _get_encryption_key raises a bare
    # `ValueError` (not a TriggerServiceError) when ENCRYPTION_KEY is unset in
    # a non-development environment and the trigger's config has
    # `store_full_payload: true`. The scan loop's call-site guard used to only
    # catch `TriggerServiceError`, so this different-but-related ValueError
    # escaped scan_due_scheduled_triggers entirely, aborting the whole batch
    # and wedging every trigger ordered after it — the same class of bug as
    # the sibling IntegrityError-escape regression
    # (test_scheduled_scan_continues_batch_after_unexpected_prepare_failure),
    # just via a different exception type. Now it's caught, rolled back, and
    # the batch continues to the next due trigger.
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    poisoned = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "scheduled",
            "name": "Poisoned by encryption failure",
            "config": {"interval_seconds": 60, "store_full_payload": True},
        },
    )
    assert poisoned.status_code == 200, poisoned.text
    poisoned_id = poisoned.json()["id"]

    healthy = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "scheduled",
            "name": "Healthy",
            "config": {"interval_seconds": 60},
        },
    )
    assert healthy.status_code == 200, healthy.text
    healthy_id = healthy.json()["id"]

    now = datetime.now(timezone.utc)
    poisoned_due_at = now - timedelta(minutes=10)
    db = _direct_db_session()
    try:
        poisoned_trigger = (
            db.query(AgentTrigger).filter(AgentTrigger.id == poisoned_id).one()
        )
        # Ordered first: an earlier next_run_at than the healthy trigger.
        poisoned_trigger.next_run_at = poisoned_due_at
        db.add(poisoned_trigger)

        healthy_trigger = (
            db.query(AgentTrigger).filter(AgentTrigger.id == healthy_id).one()
        )
        healthy_trigger.next_run_at = now - timedelta(seconds=5)
        db.add(healthy_trigger)
        db.commit()

        from xagent.web.services import triggers as triggers_mod

        def failing_encrypt_value(value: str) -> str:
            raise ValueError(
                "ENCRYPTION_KEY environment variable is not set in "
                "non-development environment"
            )

        with patch.object(
            triggers_mod, "encrypt_value", side_effect=failing_encrypt_value
        ):
            runs = scan_due_scheduled_triggers(db, now=now)

        # Only the healthy trigger produced a run; the poisoned one's
        # encryption-key failure was swallowed instead of aborting the batch.
        assert len(runs) == 1

        db.refresh(poisoned_trigger)
        assert poisoned_trigger.enabled is True
        assert _coerce_utc(poisoned_trigger.next_run_at) == poisoned_due_at

        db.refresh(healthy_trigger)
        assert healthy_trigger.enabled is True
        assert healthy_trigger.next_run_at is not None
    finally:
        db.close()


def test_scheduled_scan_surfaces_repeated_prepare_failures_without_disabling(
    mock_bg_scheduler,
) -> None:
    # PR #1051 review, N follow-up: a single prepare_trigger_run failure is
    # expected and silently retried (see
    # test_scheduled_scan_continues_batch_after_unexpected_prepare_failure) —
    # but a trigger that fails EVERY scan tick for a while is no longer just
    # an unlucky race, and used to be retried forever with zero user-visible
    # signal, unlike the sibling recompute-failure guard (which already sets
    # provisioning_status/provisioning_error and disables the trigger).
    # After _PREPARE_FAILURE_SURFACE_THRESHOLD consecutive failures for the
    # SAME trigger, this guard now surfaces the same fields too — WITHOUT
    # disabling the trigger (unlike the recompute guard): this failure mode
    # is commonly a transient infrastructure issue, so it should stay
    # visible-but-still-retried rather than silently killed. The counter
    # itself is a column on the trigger row (not an in-process dict — see
    # test_scheduled_scan_consecutive_prepare_failures_counter_is_shared_across_processes),
    # so a fresh trigger created in this test starts with no prior state to
    # reset.
    from xagent.web.services import triggers as triggers_mod

    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "scheduled",
            "name": "Always poisoned",
            "config": {"interval_seconds": 60},
        },
    )
    assert created.status_code == 200, created.text
    trigger_id = created.json()["id"]

    real_prepare_trigger_run = triggers_mod.prepare_trigger_run

    def always_fails(db_arg, *, trigger, **kwargs):
        if trigger.id == trigger_id:
            raise IntegrityError("insert", {}, Exception("duplicate key"))
        return real_prepare_trigger_run(db_arg, trigger=trigger, **kwargs)

    db = _direct_db_session()
    try:
        threshold = triggers_mod._PREPARE_FAILURE_SURFACE_THRESHOLD
        with patch.object(
            triggers_mod, "prepare_trigger_run", side_effect=always_fails
        ):
            for tick in range(threshold):
                trigger = (
                    db.query(AgentTrigger).filter(AgentTrigger.id == trigger_id).one()
                )
                trigger.next_run_at = datetime.now(timezone.utc) - timedelta(seconds=5)
                db.add(trigger)
                db.commit()

                runs = scan_due_scheduled_triggers(db, now=datetime.now(timezone.utc))
                assert runs == []

                db.refresh(trigger)
                # Never disabled, and next_run_at is left untouched by this
                # guard every tick, so the next scan keeps retrying it.
                assert trigger.enabled is True

                is_last_tick = tick == threshold - 1
                if is_last_tick:
                    assert (
                        trigger.provisioning_status
                        == TriggerProvisioningStatus.FAILED.value
                    )
                    assert trigger.provisioning_error
                else:
                    assert (
                        trigger.provisioning_status
                        != TriggerProvisioningStatus.FAILED.value
                    )

        # Recovering (prepare succeeds again) clears the stale failure
        # signal instead of leaving a permanently "failed" badge on a
        # trigger that's actually working again.
        trigger = db.query(AgentTrigger).filter(AgentTrigger.id == trigger_id).one()
        trigger.next_run_at = datetime.now(timezone.utc) - timedelta(seconds=5)
        db.add(trigger)
        db.commit()
        runs = scan_due_scheduled_triggers(db, now=datetime.now(timezone.utc))
        assert len(runs) == 1
        db.refresh(trigger)
        assert trigger.provisioning_status != TriggerProvisioningStatus.FAILED.value
        assert trigger.consecutive_prepare_failures is None
    finally:
        db.close()


def test_scheduled_scan_trigger_run_preparation_error_does_not_clear_failure_badge(
    mock_bg_scheduler,
) -> None:
    # Third review round: the `except TriggerRunPreparationError` branch
    # intentionally does NOT `continue` (a TriggerRun record does exist even
    # though its task failed to attach, so next_run_at still needs
    # recomputing and the run still needs to be returned) — but it used to
    # fall through into the "recovered, clear the counter/badge" cleanup
    # block, which naively assumed "didn't hit the incrementing except clause
    # this tick" meant "succeeded cleanly". A TriggerRunPreparationError on
    # THIS tick means the trigger is still failing (a different failure mode,
    # not a recovery), so it must not clear a previously-surfaced failed
    # badge. Only a genuinely clean prepare_trigger_run call (no exception at
    # all) should count as recovered — see the sibling
    # ...surfaces_repeated_prepare_failures_without_disabling test for that
    # genuine-recovery case.
    from xagent.web.services import triggers as triggers_mod
    from xagent.web.services.triggers import TriggerRunPreparationError

    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "scheduled",
            "name": "Flaky then differently flaky",
            "config": {"interval_seconds": 60},
        },
    )
    assert created.status_code == 200, created.text
    trigger_id = created.json()["id"]

    def always_integrity_error(db_arg, *, trigger, **kwargs):
        raise IntegrityError("insert", {}, Exception("duplicate key"))

    db = _direct_db_session()
    try:
        threshold = triggers_mod._PREPARE_FAILURE_SURFACE_THRESHOLD
        with patch.object(
            triggers_mod, "prepare_trigger_run", side_effect=always_integrity_error
        ):
            for _tick in range(threshold):
                trigger = (
                    db.query(AgentTrigger).filter(AgentTrigger.id == trigger_id).one()
                )
                trigger.next_run_at = datetime.now(timezone.utc) - timedelta(seconds=5)
                db.add(trigger)
                db.commit()
                runs = scan_due_scheduled_triggers(db, now=datetime.now(timezone.utc))
                assert runs == []

        trigger = db.query(AgentTrigger).filter(AgentTrigger.id == trigger_id).one()
        assert trigger.provisioning_status == TriggerProvisioningStatus.FAILED.value
        failed_provisioning_error = trigger.provisioning_error
        assert failed_provisioning_error

        # 6th tick: a DIFFERENT failure mode (TriggerRunPreparationError, not
        # the IntegrityError-type failure the counter above tracks). This
        # must NOT clear the badge set above, even though it doesn't go
        # through the incrementing except clause either.
        def raises_run_preparation_error(db_arg, *, trigger, **kwargs):
            failed_run = TriggerRun(
                trigger_id=int(trigger.id),
                status=TriggerRunStatus.FAILED.value,
                error_message="task attach failed",
                idempotency_key=f"test-prep-error-{trigger.id}-{time.time()}",
                payload_snapshot={},
            )
            db_arg.add(failed_run)
            db_arg.commit()
            db_arg.refresh(failed_run)
            raise TriggerRunPreparationError("task attach failed", run=failed_run)

        trigger.next_run_at = datetime.now(timezone.utc) - timedelta(seconds=5)
        db.add(trigger)
        db.commit()
        with patch.object(
            triggers_mod,
            "prepare_trigger_run",
            side_effect=raises_run_preparation_error,
        ):
            runs = scan_due_scheduled_triggers(db, now=datetime.now(timezone.utc))
        # The TriggerRunPreparationError branch doesn't `continue`: the run
        # is still returned and next_run_at still advances.
        assert len(runs) == 1

        db.refresh(trigger)
        assert trigger.enabled is True
        # The badge from the earlier IntegrityError-type failures must still
        # be showing — a different failure mode this tick is not a recovery.
        assert trigger.provisioning_status == TriggerProvisioningStatus.FAILED.value
        assert trigger.provisioning_error == failed_provisioning_error
    finally:
        db.close()


def test_scheduled_scan_consecutive_prepare_failures_counter_is_shared_across_processes(
    mock_bg_scheduler,
) -> None:
    # Third review round: _consecutive_prepare_failures used to be an
    # in-process dict, but scan_due_scheduled_triggers runs from at least two
    # genuinely separate OS processes concurrently in this deployment
    # (backend's in-process asyncio dispatcher, and a separate Celery
    # beat/worker scan). Simulate that with two independent DB sessions
    # standing in for two processes: an increment from "process B" must be
    # visible to "process A", proving the counter is now a persisted column
    # on the trigger row (read/written via atomic SQL), not disjoint
    # per-process state.
    from xagent.web.services import triggers as triggers_mod

    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "scheduled",
            "name": "Cross-process counter",
            "config": {"interval_seconds": 60},
        },
    )
    assert created.status_code == 200, created.text
    trigger_id = created.json()["id"]

    process_a_db = _direct_db_session()
    process_b_db = _direct_db_session()
    try:
        trigger_as_seen_by_a = (
            process_a_db.query(AgentTrigger).filter(AgentTrigger.id == trigger_id).one()
        )
        trigger_as_seen_by_b = (
            process_b_db.query(AgentTrigger).filter(AgentTrigger.id == trigger_id).one()
        )

        count_after_a = triggers_mod._increment_consecutive_prepare_failures(
            process_a_db, trigger_as_seen_by_a
        )
        assert count_after_a == 1

        count_after_b = triggers_mod._increment_consecutive_prepare_failures(
            process_b_db, trigger_as_seen_by_b
        )
        # Process B's increment builds on process A's, not on its own
        # independent zero-initialized counter — proving the state is
        # shared, authoritative, and read/written atomically at the DB
        # layer rather than split across two disjoint in-memory dicts.
        assert count_after_b == 2

        # A "recovery" observed by process A must see and clear the value
        # process B most recently wrote, even though process A never
        # incremented past 1 itself.
        triggers_mod._clear_consecutive_prepare_failures_if_recovered(
            process_a_db, trigger_as_seen_by_a
        )
        process_b_db.refresh(trigger_as_seen_by_b)
        assert trigger_as_seen_by_b.consecutive_prepare_failures is None
    finally:
        process_a_db.close()
        process_b_db.close()


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
