import asyncio
import inspect
import json
from datetime import UTC, datetime, timedelta
from threading import Event, get_ident
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from tests.web.pool_contention_shared import (
    CONTENTION_POOL_TIMEOUT,
    GUARD_TIMEOUT,
    LOOP_LIVENESS_TICKS,
    gated_pool_checkout,
    wait_for_ticks,
)
from xagent.core.agent.checkpoint import (
    CheckpointAccessRefusedError,
    CheckpointCorruptError,
    CheckpointReadError,
    CheckpointUnavailableError,
)
from xagent.web.api import a2a as a2a_api
from xagent.web.models.agent import Agent
from xagent.web.models.agent_api_key import AgentApiKey
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.task_interaction import TaskInteractionRequest
from xagent.web.services.a2a_protocol import (
    A2A_MAX_MESSAGE_TEXT_LENGTH,
    A2AApiError,
    A2ATaskSnapshot,
)
from xagent.web.services.task_command_transport import (
    COMMAND_FAILED,
    MAX_COMMAND_DEFERS,
    MAX_COMMAND_FAILURES,
    ClaimedTaskCommand,
    TaskCommandKind,
    TaskCommandRejected,
)
from xagent.web.services.task_execution_controller import TaskControlState
from xagent.web.services.task_lease_service import TaskLease, current_task_lease
from xagent.web.services.task_orchestrator import (
    TaskTurnError,
    TaskTurnOrchestrator,
    TaskTurnPayload,
    TurnKind,
)

from .conftest import _admin_headers, _direct_db_session, client

pytestmark = pytest.mark.usefixtures("_test_db")


def _bearer(full_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {full_key}",
        "A2A-Version": "1.0",
    }


def _create_agent(headers: dict[str, str], name: str = "A2A Test Agent") -> int:
    response = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": name,
            "description": "A2A test agent",
            "instructions": "You are an A2A test agent.",
            "execution_mode": "balanced",
            "suggested_prompts": ["Summarize this"],
        },
    )
    assert response.status_code == 200, response.text
    return int(response.json()["id"])


def _publish_agent(headers: dict[str, str], agent_id: int) -> None:
    response = client.post(f"/api/agents/{agent_id}/publish", headers=headers)
    assert response.status_code == 200, response.text


def _create_published_agent_with_key() -> tuple[int, str]:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    _publish_agent(headers, agent_id)
    key_response = client.post(f"/api/agents/{agent_id}/api-key", headers=headers)
    assert key_response.status_code == 200, key_response.text
    return agent_id, key_response.json()["full_key"]


def _seed_active_interaction_row(
    db: Session, *, task_id: int, run_id: str, idempotency_key: str
) -> int:
    """One legal active TaskInteractionRequest row, for the legacy resume
    close/compensation tests below. Not the interaction staging primitive's
    fixture builder (tests/web/services/task_interaction_schema_shared.py)
    -- this file has no other reason to depend on that directory, so this
    stays a small, local, single-purpose row builder instead."""
    anchor = TraceEvent(
        task_id=task_id,
        event_id=f"anchor-{idempotency_key}",
        event_type="agent_execution_checkpoint",
        timestamp=datetime.now(UTC),
        data={},
    )
    db.add(anchor)
    db.flush()
    row = TaskInteractionRequest(
        task_id=task_id,
        run_id=run_id,
        kind="clarification",
        protocol_version=1,
        status="active",
        active_slot=1,
        origin="a2a",
        request_payload={"prompt": "example"},
        request_idempotency_key=idempotency_key,
        resume_trace_event_id=int(anchor.id),
        resume_event_id="resume-event-1",
        resume_execution_id="resume-execution-1",
        resume_locator_format="trace_event_pk_v1",
        resume_checkpoint_type="agent_execution_checkpoint",
        resume_run_partition=run_id,
        expires_at=datetime.now(UTC) + timedelta(minutes=15),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return int(row.id)


def test_agent_card_exposes_published_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XAGENT_PUBLIC_API_BASE_URL", raising=False)
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    _publish_agent(headers, agent_id)

    response = client.get(f"/api/a2a/agents/{agent_id}/.well-known/agent-card.json")

    assert response.status_code == 200, response.text
    assert response.headers["a2a-version"] == "1.0"
    assert response.headers["content-type"].startswith("application/a2a+json")
    body = response.json()
    assert body["name"] == "A2A Test Agent"
    assert body["defaultInputModes"] == ["text/plain", "application/json"]
    assert body["defaultOutputModes"] == ["text/plain"]
    assert body["supportedInterfaces"] == [
        {
            "url": f"http://testserver/api/a2a/agents/{agent_id}",
            "protocolBinding": "HTTP+JSON",
            "protocolVersion": "1.0",
        }
    ]
    assert body["securitySchemes"]["xagentAgentApiKey"] == {
        "httpAuthSecurityScheme": {
            "scheme": "Bearer",
            "description": "Xagent agent API key",
        }
    }
    assert body["securityRequirements"] == [{"schemes": {"xagentAgentApiKey": {}}}]
    assert body["skills"][0]["examples"] == ["Summarize this"]


def test_agent_card_uses_s2s_api_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", " https://sg.cloud.xagent.co/ ")
    monkeypatch.setenv(
        "XAGENT_S2S_API_BASE_URL", " https://sg-origin.cloud.xagent.co/ "
    )
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    _publish_agent(headers, agent_id)

    response = client.get(f"/api/a2a/agents/{agent_id}/.well-known/agent-card.json")

    assert response.status_code == 200, response.text
    assert response.json()["supportedInterfaces"][0]["url"] == (
        f"https://sg-origin.cloud.xagent.co/api/a2a/agents/{agent_id}"
    )


def test_agent_card_hides_draft_agent() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)

    response = client.get(f"/api/a2a/agents/{agent_id}/.well-known/agent-card.json")

    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == 404
    assert error["status"] == "NOT_FOUND"
    assert error["details"][0]["reason"] == "AGENT_NOT_FOUND"


def test_agent_card_does_not_expose_private_instructions() -> None:
    headers = _admin_headers()
    response = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": "Private Prompt Agent",
            "instructions": "secret system prompt",
            "execution_mode": "balanced",
        },
    )
    assert response.status_code == 200, response.text
    agent_id = int(response.json()["id"])
    _publish_agent(headers, agent_id)

    card = client.get(f"/api/a2a/agents/{agent_id}/.well-known/agent-card.json").json()

    assert card["description"] == "Private Prompt Agent"
    assert "secret system prompt" not in json.dumps(card)


def test_message_send_creates_hidden_a2a_task() -> None:
    agent_id, full_key = _create_published_agent_with_key()

    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ) as schedule_bg:
        response = client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=_bearer(full_key),
            json={
                "message": {
                    "messageId": "msg-1",
                    "contextId": "ctx-1",
                    "role": "ROLE_USER",
                    "parts": [{"text": "hello from a2a"}],
                },
                "configuration": {"returnImmediately": True},
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    task = body["task"]
    assert task["contextId"] == "ctx-1"
    assert task["status"]["state"] == "TASK_STATE_WORKING"

    db = _direct_db_session()
    try:
        row = db.query(Task).filter(Task.id == int(task["id"])).one()
        assert row.agent_id == agent_id
        assert row.source == "a2a"
        assert row.is_visible is False
        assert row.input == "hello from a2a"
        assert row.agent_config == {"a2a_context_id": "ctx-1"}
        assert row.status == TaskStatus.RUNNING

        key = db.query(AgentApiKey).filter(AgentApiKey.agent_id == agent_id).one()
        assert key.usage_month == datetime.now(UTC).strftime("%Y-%m")
        assert key.usage_month_calls == 1
    finally:
        db.close()

    assert schedule_bg.call_count == 1
    assert schedule_bg.call_args.kwargs["task_id"] == int(task["id"])


def test_message_send_rejects_key_bound_to_different_agent() -> None:
    agent_id, full_key = _create_published_agent_with_key()
    headers = _admin_headers()
    other_agent_id = _create_agent(headers, name="Other A2A Agent")
    _publish_agent(headers, other_agent_id)

    response = client.post(
        f"/api/a2a/agents/{other_agent_id}/message:send",
        headers=_bearer(full_key),
        json={
            "message": {
                "messageId": "msg-1",
                "role": "ROLE_USER",
                "parts": [{"text": "wrong target"}],
            },
            "configuration": {"returnImmediately": True},
        },
    )

    assert other_agent_id != agent_id
    assert response.status_code == 404
    assert response.json()["error"]["details"][0]["reason"] == "AGENT_NOT_FOUND"


def test_message_send_rejects_draft_agent_key() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    key_response = client.post(f"/api/agents/{agent_id}/api-key", headers=headers)
    assert key_response.status_code == 200, key_response.text

    response = client.post(
        f"/api/a2a/agents/{agent_id}/message:send",
        headers=_bearer(key_response.json()["full_key"]),
        json={
            "message": {
                "messageId": "msg-1",
                "role": "ROLE_USER",
                "parts": [{"text": "draft should not run"}],
            },
            "configuration": {"returnImmediately": True},
        },
    )

    assert response.status_code == 404
    assert response.json()["error"]["details"][0]["reason"] == "AGENT_NOT_FOUND"


def test_message_send_requires_supported_a2a_version() -> None:
    agent_id, full_key = _create_published_agent_with_key()
    payload = {
        "message": {
            "messageId": "msg-version",
            "role": "ROLE_USER",
            "parts": [{"text": "hello"}],
        },
        "configuration": {"returnImmediately": True},
    }

    response = client.post(
        f"/api/a2a/agents/{agent_id}/message:send",
        headers={"Authorization": f"Bearer {full_key}"},
        json=payload,
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["message"] == "A2A-Version header or query parameter is required."
    assert error["details"][0]["reason"] == "VERSION_NOT_SUPPORTED"
    assert error["details"][0]["metadata"]["supportedVersions"] == "1.0"

    incompatible = client.post(
        f"/api/a2a/agents/{agent_id}/message:send",
        headers={
            "Authorization": f"Bearer {full_key}",
            "A2A-Version": "2.0",
        },
        json=payload,
    )

    assert incompatible.status_code == 400
    incompatible_error = incompatible.json()["error"]
    assert incompatible_error["details"][0]["reason"] == "VERSION_NOT_SUPPORTED"
    assert incompatible_error["details"][0]["metadata"]["supportedVersions"] == ("1.0")


def test_message_send_rejects_oversized_content() -> None:
    agent_id, full_key = _create_published_agent_with_key()

    response = client.post(
        f"/api/a2a/agents/{agent_id}/message:send",
        headers=_bearer(full_key),
        json={
            "message": {
                "messageId": "msg-too-large",
                "role": "ROLE_USER",
                "parts": [{"text": "x" * (A2A_MAX_MESSAGE_TEXT_LENGTH + 1)}],
            },
            "configuration": {"returnImmediately": True},
        },
    )

    assert response.status_code == 413
    error = response.json()["error"]
    assert error["status"] == "RESOURCE_EXHAUSTED"
    assert error["details"][0]["metadata"]["maxLength"] == str(
        A2A_MAX_MESSAGE_TEXT_LENGTH
    )


def test_message_send_requires_message_id() -> None:
    agent_id, full_key = _create_published_agent_with_key()

    response = client.post(
        f"/api/a2a/agents/{agent_id}/message:send",
        headers=_bearer(full_key),
        json={
            "message": {
                "role": "ROLE_USER",
                "parts": [{"text": "missing id"}],
            },
            "configuration": {"returnImmediately": True},
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["details"][0]["reason"] == "INVALID_ARGUMENT"
    assert error["details"][0]["metadata"]["field"] == "message.messageId"


def test_a2a_fastapi_validation_uses_protocol_error_shape() -> None:
    agent_id, full_key = _create_published_agent_with_key()

    response = client.get(
        f"/api/a2a/agents/{agent_id}/tasks",
        headers=_bearer(full_key),
        params={"pageSize": 0},
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["status"] == "INVALID_ARGUMENT"
    assert error["details"][0]["reason"] == "INVALID_ARGUMENT"


def test_a2a_auth_error_includes_bearer_challenge() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    _publish_agent(headers, agent_id)

    response = client.get(
        f"/api/a2a/agents/{agent_id}/tasks",
        headers={"A2A-Version": "1.0"},
    )

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["error"]["status"] == "UNAUTHENTICATED"


def test_message_send_blocks_by_default_until_task_finishes() -> None:
    agent_id, full_key = _create_published_agent_with_key()

    async def _complete_turn(**kwargs: object) -> object:
        db = _direct_db_session()
        try:
            row = db.query(Task).filter(Task.id == int(kwargs["task_id"])).one()
            row.status = TaskStatus.COMPLETED
            row.output = "blocking response"
            db.commit()
        finally:
            db.close()
        return object()

    with patch(
        "xagent.web.api.a2a.TaskTurnOrchestrator.schedule_claimed_create_turn",
        new=_complete_turn,
    ):
        response = client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=_bearer(full_key),
            json={
                "message": {
                    "messageId": "msg-blocking",
                    "role": "ROLE_USER",
                    "parts": [{"text": "wait for me"}],
                }
            },
        )

    assert response.status_code == 200, response.text
    task = response.json()["task"]
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert task["artifacts"][0]["parts"] == [{"text": "blocking response"}]


def test_message_send_returns_working_task_when_wait_deadline_expires(
    monkeypatch,
) -> None:
    agent_id, full_key = _create_published_agent_with_key()
    monkeypatch.setattr(a2a_api, "A2A_BLOCKING_WAIT_TIMEOUT_SECONDS", 0.0)

    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ):
        response = client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=_bearer(full_key),
            json={
                "message": {
                    "messageId": "msg-wait-timeout",
                    "role": "ROLE_USER",
                    "parts": [{"text": "keep working"}],
                }
            },
        )

    assert response.status_code == 200, response.text
    assert response.json()["task"]["status"]["state"] == "TASK_STATE_WORKING"


def test_failed_create_scheduling_does_not_leave_pending_a2a_task() -> None:
    agent_id, full_key = _create_published_agent_with_key()

    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        side_effect=TaskTurnError("busy"),
    ):
        response = client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=_bearer(full_key),
            json={
                "message": {
                    "messageId": "msg-failed-create",
                    "role": "ROLE_USER",
                    "parts": [{"text": "fail before scheduling"}],
                },
                "configuration": {"returnImmediately": True},
            },
        )

    assert response.status_code == 400, response.text
    db = _direct_db_session()
    try:
        tasks = db.query(Task).filter(Task.source == "a2a").all()
        assert len(tasks) == 1
        assert tasks[0].status == TaskStatus.FAILED
        assert tasks[0].runner_id is None
        assert tasks[0].lease_expires_at is None
    finally:
        db.close()


def test_unexpected_a2a_error_uses_internal_protocol_envelope() -> None:
    agent_id, full_key = _create_published_agent_with_key()

    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        side_effect=RuntimeError("sensitive implementation detail"),
    ):
        response = client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=_bearer(full_key),
            json={
                "message": {
                    "messageId": "msg-error",
                    "role": "ROLE_USER",
                    "parts": [{"text": "fail"}],
                },
                "configuration": {"returnImmediately": True},
            },
        )

    assert response.status_code == 500
    error = response.json()["error"]
    assert error["status"] == "INTERNAL"
    assert error["details"][0]["reason"] == "INTERNAL"
    assert "sensitive implementation detail" not in response.text
    db = _direct_db_session()
    try:
        tasks = db.query(Task).filter(Task.source == "a2a").all()
        assert len(tasks) == 1
        assert tasks[0].status == TaskStatus.FAILED
        assert tasks[0].runner_id is None
        assert tasks[0].lease_expires_at is None
    finally:
        db.close()


def test_get_and_list_tasks_use_rest_binding_shapes() -> None:
    agent_id, full_key = _create_published_agent_with_key()
    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ):
        created = client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=_bearer(full_key),
            json={
                "message": {
                    "messageId": "msg-list",
                    "contextId": "ctx-list",
                    "role": "ROLE_USER",
                    "parts": [{"text": "list me"}],
                },
                "configuration": {"returnImmediately": True},
            },
        )
    task_id = created.json()["task"]["id"]

    db = _direct_db_session()
    try:
        row = db.query(Task).filter(Task.id == int(task_id)).one()
        row.status = TaskStatus.COMPLETED
        row.output = "listed output"
        db.commit()
    finally:
        db.close()

    fetched = client.get(
        f"/api/a2a/agents/{agent_id}/tasks/{task_id}",
        headers=_bearer(full_key),
    )
    listed = client.get(
        f"/api/a2a/agents/{agent_id}/tasks",
        headers=_bearer(full_key),
        params={"contextId": "ctx-list"},
    )
    listed_with_artifacts = client.get(
        f"/api/a2a/agents/{agent_id}/tasks",
        headers=_bearer(full_key),
        params={"includeArtifacts": "true"},
    )

    assert fetched.status_code == 200, fetched.text
    assert fetched.json()["id"] == task_id
    assert "task" not in fetched.json()
    assert fetched.json()["artifacts"][0]["parts"] == [{"text": "listed output"}]
    assert listed.json()["totalSize"] == 1
    assert "artifacts" not in listed.json()["tasks"][0]
    assert listed_with_artifacts.json()["tasks"][0]["artifacts"][0]["parts"] == [
        {"text": "listed output"}
    ]


def test_follow_up_infers_context_for_input_required_task() -> None:
    agent_id, full_key = _create_published_agent_with_key()
    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ):
        created = client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=_bearer(full_key),
            json={
                "message": {
                    "messageId": "msg-initial",
                    "contextId": "ctx-follow-up",
                    "role": "ROLE_USER",
                    "parts": [{"text": "initial"}],
                },
                "configuration": {"returnImmediately": True},
            },
        )
    task_id = created.json()["task"]["id"]
    db = _direct_db_session()
    try:
        row = db.query(Task).filter(Task.id == int(task_id)).one()
        row.status = TaskStatus.WAITING_FOR_USER
        row.control_state = TaskControlState.WAITING_FOR_USER.value
        row.runner_id = None
        row.last_heartbeat_at = None
        row.lease_expires_at = None
        db.commit()
    finally:
        db.close()

    observed_lease: dict[str, object] = {}

    async def post_user_message(*_args: object, **_kwargs: object) -> bool:
        lease = current_task_lease()
        assert lease is not None
        assert lease.run_id is not None
        observed_lease["lease"] = lease
        lease_db = _direct_db_session()
        try:
            leased = lease_db.query(Task).filter(Task.id == int(task_id)).one()
            assert leased.status == TaskStatus.RUNNING
            assert leased.runner_id == lease.runner_id
            assert leased.run_id == lease.run_id
            assert leased.lease_expires_at is not None
        finally:
            lease_db.close()
        return True

    agent_service = MagicMock()
    agent_service.post_user_message = AsyncMock(side_effect=post_user_message)
    agent_manager = MagicMock()
    agent_manager.get_agent_for_task = AsyncMock(return_value=agent_service)
    begin_turn = AsyncMock()
    with (
        patch(
            "xagent.web.api.chat.get_agent_manager",
            return_value=agent_manager,
        ),
        patch("xagent.web.api.a2a._schedule_waiting_a2a_resume") as schedule_resume,
        patch(
            "xagent.web.api.a2a.TaskTurnOrchestrator.begin_turn",
            new=begin_turn,
        ),
    ):
        response = client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=_bearer(full_key),
            json={
                "message": {
                    "messageId": "msg-follow-up",
                    "taskId": task_id,
                    "role": "ROLE_USER",
                    "parts": [{"text": "follow up"}],
                },
                "configuration": {"returnImmediately": True},
            },
        )

    assert response.status_code == 200, response.text
    assert response.json()["task"]["contextId"] == "ctx-follow-up"
    assert response.json()["task"]["status"]["state"] == "TASK_STATE_WORKING"
    agent_service.post_user_message.assert_awaited_once_with(
        task_id,
        execution_message="follow up",
        display_message="follow up",
        turn_id=f"a2a:{task_id}:msg-follow-up",
        request_interrupt=False,
        reason="A2A input-required response",
    )
    begin_turn.assert_not_awaited()
    schedule_resume.assert_called_once()
    scheduled_lease = schedule_resume.call_args.kwargs["task_lease"]
    assert scheduled_lease == observed_lease["lease"]
    db = _direct_db_session()
    try:
        resumed = db.query(Task).filter(Task.id == int(task_id)).one()
        assert resumed.status == TaskStatus.RUNNING
        assert resumed.control_state == TaskControlState.RUNNING.value
        assert resumed.runner_id == scheduled_lease.runner_id
        assert resumed.run_id == scheduled_lease.run_id
        assert resumed.lease_expires_at is not None
        assert resumed.input == "follow up"
    finally:
        db.close()


def test_checkpoint_resume_schedule_failure_exactly_restores_waiting_task() -> None:
    agent_id, full_key = _create_published_agent_with_key()
    db = _direct_db_session()
    try:
        owner_id = int(db.query(Agent).filter(Agent.id == agent_id).one().user_id)
        task = Task(
            user_id=owner_id,
            title="waiting",
            status=TaskStatus.WAITING_FOR_USER,
            control_state=TaskControlState.WAITING_FOR_USER.value,
            run_id="run-a",
            agent_id=agent_id,
            source="a2a",
            is_visible=False,
            agent_config={"a2a_context_id": "ctx-schedule-failure"},
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = int(task.id)
    finally:
        db.close()

    agent_service = MagicMock()
    agent_service.post_user_message = AsyncMock(return_value=True)
    agent_manager = MagicMock()
    agent_manager.get_agent_for_task = AsyncMock(return_value=agent_service)
    scheduled_lease: dict[str, object] = {}

    def fail_schedule(*, task_lease: TaskLease, **_kwargs: object) -> None:
        scheduled_lease["lease"] = task_lease
        schedule_db = _direct_db_session()
        try:
            durable = schedule_db.query(Task).filter(Task.id == task_id).one()
            assert durable.status == TaskStatus.RUNNING
            assert durable.runner_id == task_lease.runner_id
            assert durable.run_id == task_lease.run_id
            assert durable.lease_expires_at is not None
        finally:
            schedule_db.close()
        raise RuntimeError("scheduler unavailable")

    with (
        patch(
            "xagent.web.api.chat.get_agent_manager",
            return_value=agent_manager,
        ),
        patch(
            "xagent.web.api.a2a._schedule_waiting_a2a_resume",
            side_effect=fail_schedule,
        ),
    ):
        response = client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=_bearer(full_key),
            json={
                "message": {
                    "messageId": "msg-schedule-failure",
                    "taskId": task_id,
                    "role": "ROLE_USER",
                    "parts": [{"text": "retry safely"}],
                },
                "configuration": {"returnImmediately": True},
            },
        )

    assert response.status_code == 500
    assert scheduled_lease["lease"] is not None
    db = _direct_db_session()
    try:
        restored = db.query(Task).filter(Task.id == task_id).one()
        assert restored.status == TaskStatus.WAITING_FOR_USER
        assert restored.control_state == TaskControlState.WAITING_FOR_USER.value
        assert restored.run_id == "run-a"
        assert restored.runner_id is None
        assert restored.lease_expires_at is None
    finally:
        db.close()


def test_update_a2a_resume_input_rolls_back_the_interaction_close_with_the_fence() -> (
    None
):
    """The fence UPDATE and the legacy resume interaction close are one
    atomic write: a fence miss (ownership changed under this lease) must
    roll back both together, not close the row while rejecting the input.

    What this actually pins is that the close must never commit
    independently of the host transaction -- reordering the two statements
    within that same transaction changes nothing observable, because a
    rollback undoes every statement issued since the last commit regardless
    of program order. Turning this red needs two changes at once: the close
    must move ahead of the fence's early return (unreachable there today,
    since a fence miss returns before the close ever runs) and it must
    commit independently of this function's session -- either change alone
    leaves this cell green, verified directly.
    """
    db = _direct_db_session()
    try:
        agent_id, _full_key = _create_published_agent_with_key()
        owner_id = int(db.query(Agent).filter(Agent.id == agent_id).one().user_id)
        task = Task(
            user_id=owner_id,
            title="atomicity",
            status=TaskStatus.RUNNING,
            runner_id="current-runner",
            run_id="run-atomicity",
            agent_id=agent_id,
            source="a2a",
            is_visible=False,
            interaction_protocol_version=1,
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = int(task.id)
        row_id = _seed_active_interaction_row(
            db, task_id=task_id, run_id="run-atomicity", idempotency_key="atomicity-q1"
        )
    finally:
        db.close()

    # A different runner_id than the row's current one: the fence's WHERE
    # clause requires an exact match, so this lease has already lost the
    # race by the time the write is attempted.
    stale_lease = TaskLease(
        task_id=task_id, runner_id="a-different-runner", run_id="run-atomicity"
    )
    updated = a2a_api._update_a2a_resume_input_sync(
        stale_lease, "attempted text", row_id
    )

    assert updated is False
    db = _direct_db_session()
    try:
        row = (
            db.query(TaskInteractionRequest)
            .filter(TaskInteractionRequest.id == row_id)
            .one()
        )
        assert row.status == "active"
        refreshed = db.query(Task).filter(Task.id == task_id).one()
        assert refreshed.interaction_protocol_version == 1
        assert refreshed.input is None
    finally:
        db.close()


def test_message_send_closes_the_legacy_resume_interaction_row_on_successful_injection() -> (
    None
):
    """The success path this whole change exists for, driven through the
    real HTTP message:send call rather than the sync helper directly: once
    the fence UPDATE lands, the run's active interaction row is retired
    (``terminated`` / ``answered_via_legacy_resume``) and the task's
    protocol marker is cleared back to NULL in the same commit."""
    agent_id, full_key = _create_published_agent_with_key()
    db = _direct_db_session()
    try:
        owner_id = int(db.query(Agent).filter(Agent.id == agent_id).one().user_id)
        task = Task(
            user_id=owner_id,
            title="legacy resume close success",
            status=TaskStatus.PAUSED,
            control_state=TaskControlState.PAUSED.value,
            run_id="run-close-success",
            agent_id=agent_id,
            source="a2a",
            is_visible=False,
            agent_config={"a2a_context_id": "ctx-close-success"},
            interaction_protocol_version=1,
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = int(task.id)
        row_id = _seed_active_interaction_row(
            db,
            task_id=task_id,
            run_id="run-close-success",
            idempotency_key="close-success-q1",
        )
    finally:
        db.close()

    agent_service = MagicMock()
    agent_service.post_user_message = AsyncMock(return_value=True)
    agent_manager = MagicMock()
    agent_manager.get_agent_for_task = AsyncMock(return_value=agent_service)
    begin_turn = AsyncMock()
    with (
        patch(
            "xagent.web.api.chat.get_agent_manager",
            return_value=agent_manager,
        ),
        patch("xagent.web.api.a2a._schedule_waiting_a2a_resume"),
        patch(
            "xagent.web.api.a2a.TaskTurnOrchestrator.begin_turn",
            new=begin_turn,
        ),
    ):
        response = client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=_bearer(full_key),
            json={
                "message": {
                    "messageId": "msg-close-success",
                    "taskId": task_id,
                    "role": "ROLE_USER",
                    "parts": [{"text": "answered via legacy resume"}],
                },
                "configuration": {"returnImmediately": True},
            },
        )

    assert response.status_code == 200, response.text
    agent_service.post_user_message.assert_awaited_once()
    db = _direct_db_session()
    try:
        row = (
            db.query(TaskInteractionRequest)
            .filter(TaskInteractionRequest.id == row_id)
            .one()
        )
        assert row.status == "terminated"
        assert row.terminal_reason == "answered_via_legacy_resume"
        refreshed = db.query(Task).filter(Task.id == task_id).one()
        assert refreshed.interaction_protocol_version is None
    finally:
        db.close()


# A fabricated id, not the seeded row's -- test_message_send_reads_the_
# interaction_row_before_injecting hands this to the close instead of the
# real row id, so a site that re-read the row at close time would hand the
# close the real id and fail there instead.
_OBSERVED_INTERACTION_ID = 4321


def test_message_send_reads_the_interaction_row_before_injecting() -> None:
    """The close is keyed on the row observed *before* the injection,
    and only the ordering makes that true -- see task_interaction_close's
    module docstring. Moving the read after the injection leaves the whole
    change doing nothing while the row-level assertions in the test above
    stay green. The observed value is a fabricated id, not the seeded row's,
    so a site that re-read the row at close time would hand the close the
    real id and fail here."""

    agent_id, full_key = _create_published_agent_with_key()
    db = _direct_db_session()
    try:
        owner_id = int(db.query(Agent).filter(Agent.id == agent_id).one().user_id)
        task = Task(
            user_id=owner_id,
            title="legacy resume close ordering",
            status=TaskStatus.PAUSED,
            control_state=TaskControlState.PAUSED.value,
            run_id="run-close-order",
            agent_id=agent_id,
            source="a2a",
            is_visible=False,
            agent_config={"a2a_context_id": "ctx-close-order"},
            interaction_protocol_version=1,
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = int(task.id)
        # Kept real and distinct from _OBSERVED_INTERACTION_ID: a site that
        # re-read the row at close time (instead of using the id observed
        # before injection) would hand the close this real id and fail the
        # assertion below.
        _seed_active_interaction_row(
            db,
            task_id=task_id,
            run_id="run-close-order",
            idempotency_key="close-order-q1",
        )
    finally:
        db.close()

    order: list[str] = []

    def record_read(_task_id: int) -> int:
        order.append("read")
        return _OBSERVED_INTERACTION_ID

    async def record_injection(*_args: object, **_kwargs: object) -> bool:
        order.append("inject")
        return True

    agent_service = MagicMock()
    agent_service.post_user_message = AsyncMock(side_effect=record_injection)
    agent_manager = MagicMock()
    agent_manager.get_agent_for_task = AsyncMock(return_value=agent_service)
    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=agent_manager),
        patch("xagent.web.api.a2a._schedule_waiting_a2a_resume"),
        patch("xagent.web.api.a2a.TaskTurnOrchestrator.begin_turn", new=AsyncMock()),
        patch(
            "xagent.web.api.a2a.active_interaction_id_sync",
            side_effect=record_read,
        ),
        patch(
            "xagent.web.api.a2a.close_legacy_resume_interaction", return_value=1
        ) as close_mock,
    ):
        response = client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=_bearer(full_key),
            json={
                "message": {
                    "messageId": "msg-close-order",
                    "taskId": task_id,
                    "role": "ROLE_USER",
                    "parts": [{"text": "answered via legacy resume"}],
                },
                "configuration": {"returnImmediately": True},
            },
        )

    assert response.status_code == 200, response.text
    assert order == ["read", "inject"]
    close_mock.assert_called_once()
    assert close_mock.call_args.kwargs["task_id"] == task_id
    assert close_mock.call_args.kwargs["run_id"] == "run-close-order"
    assert close_mock.call_args.kwargs["interaction_id"] == _OBSERVED_INTERACTION_ID


@pytest.mark.asyncio
async def test_a2a_handover_restores_input_required_on_unreadable_checkpoint() -> None:
    """The A2A handover carries the pre-claim status into the resume.

    The prelease claims the task out of WAITING_FOR_USER and commits RUNNING
    before handing the lease to ``execute_resume_background``, which therefore
    never runs the acquisition that captures a prior status. The status travels
    only as the ``preacquired_prior_status`` kwarg, so drive the real path
    across the handover: a checkpoint the resume cannot read must land the row
    back on WAITING_FOR_USER under its original run, not on a terminal FAILED.
    """
    from xagent.web.api import websocket as websocket_api

    agent_id, _full_key = _create_published_agent_with_key()
    db = _direct_db_session()
    try:
        owner_id = int(db.query(Agent).filter(Agent.id == agent_id).one().user_id)
        task = Task(
            user_id=owner_id,
            title="waiting on handover",
            status=TaskStatus.WAITING_FOR_USER,
            control_state=TaskControlState.WAITING_FOR_USER.value,
            run_id="run-handover",
            agent_id=agent_id,
            source="a2a",
            is_visible=False,
            agent_config={"a2a_context_id": "ctx-handover"},
            error_message="stale a2a failure",
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = int(task.id)
        snapshot = A2ATaskSnapshot.from_task(task)
    finally:
        db.close()

    agent_service = MagicMock()
    agent_service.post_user_message = AsyncMock(return_value=True)
    agent_service.resume_execution_by_id = AsyncMock(
        side_effect=CheckpointUnavailableError("checkpoint store unavailable")
    )
    agent_manager = MagicMock()
    agent_manager.get_agent_for_task = AsyncMock(return_value=agent_service)

    with patch(
        "xagent.web.api.chat.get_agent_manager",
        return_value=agent_manager,
    ):
        assert await a2a_api._resume_input_required_a2a_task(
            agent_id=agent_id,
            task_owner_user_id=owner_id,
            task=snapshot,
            text="follow up after handover",
            message_id="msg-handover",
        )
        # Ownership transferred synchronously: the scheduled resume has not run
        # yet, so its registration is still the one this handover created.
        resume_task = websocket_api.background_task_manager.resume_tasks[task_id]
        # The coordinator is evidence only for the run it was created to
        # resume: a command for this run sees it, a command for any other run
        # must not be able to treat it as an idempotent success.
        assert (
            websocket_api.background_task_manager.resume_admission_state(
                task_id, expected_run_id="run-handover"
            )
            is websocket_api.ResumeReservationOutcome.COORDINATOR_RUNNING
        )
        assert (
            websocket_api.background_task_manager.resume_admission_state(
                task_id, expected_run_id="some-other-run"
            )
            is websocket_api.ResumeReservationOutcome.RESERVATION_HELD
        )
        await asyncio.wait_for(resume_task, timeout=30)

    agent_service.resume_execution_by_id.assert_awaited_once()
    db = _direct_db_session()
    try:
        restored = db.query(Task).filter(Task.id == task_id).one()
        assert restored.status == TaskStatus.WAITING_FOR_USER
        assert restored.control_state == TaskControlState.WAITING_FOR_USER.value
        assert restored.run_id == "run-handover"
        assert restored.runner_id is None
        assert restored.lease_expires_at is None
        assert restored.error_message is None
    finally:
        db.close()


def test_recovered_paused_checkpoint_resumes_without_transcript_fallback() -> None:
    agent_id, full_key = _create_published_agent_with_key()
    db = _direct_db_session()
    try:
        owner_id = int(db.query(Agent).filter(Agent.id == agent_id).one().user_id)
        task = Task(
            user_id=owner_id,
            title="recovered checkpoint",
            status=TaskStatus.PAUSED,
            control_state=TaskControlState.PAUSED.value,
            run_id="run-recovered",
            agent_id=agent_id,
            source="a2a",
            is_visible=False,
            agent_config={"a2a_context_id": "ctx-recovered"},
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = int(task.id)
    finally:
        db.close()

    agent_service = MagicMock()
    agent_service.post_user_message = AsyncMock(return_value=True)
    agent_manager = MagicMock()
    agent_manager.get_agent_for_task = AsyncMock(return_value=agent_service)
    begin_turn = AsyncMock()
    with (
        patch(
            "xagent.web.api.chat.get_agent_manager",
            return_value=agent_manager,
        ),
        patch("xagent.web.api.a2a._schedule_waiting_a2a_resume") as schedule_resume,
        patch(
            "xagent.web.api.a2a.TaskTurnOrchestrator.begin_turn",
            new=begin_turn,
        ),
    ):
        response = client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=_bearer(full_key),
            json={
                "message": {
                    "messageId": "msg-recovered",
                    "taskId": task_id,
                    "role": "ROLE_USER",
                    "parts": [{"text": "resume recovered run"}],
                },
                "configuration": {"returnImmediately": True},
            },
        )

    assert response.status_code == 200, response.text
    agent_service.post_user_message.assert_awaited_once()
    schedule_resume.assert_called_once()
    begin_turn.assert_not_awaited()
    db = _direct_db_session()
    try:
        resumed = db.query(Task).filter(Task.id == task_id).one()
        assert resumed.status == TaskStatus.RUNNING
        assert resumed.run_id == "run-recovered"
        assert resumed.runner_id is not None
        assert resumed.lease_expires_at is not None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_untagged_checkpoint_is_not_resumed_without_an_exact_run() -> None:
    agent_id, _full_key = _create_published_agent_with_key()
    db = _direct_db_session()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).one()
        task = Task(
            user_id=int(agent.user_id),
            title="legacy waiting",
            status=TaskStatus.WAITING_FOR_USER,
            control_state=TaskControlState.WAITING_FOR_USER.value,
            run_id=None,
            last_checkpoint_event_id="legacy-checkpoint",
            agent_id=agent_id,
            source="a2a",
            is_visible=False,
            agent_config={"a2a_context_id": "ctx-legacy"},
        )
        db.add(task)
        db.flush()
        db.add(
            TraceEvent(
                task_id=int(task.id),
                event_id="legacy-checkpoint",
                event_type="system_update_general",
                timestamp=datetime.now(UTC),
                data={
                    "checkpoint_type": "xagent.agent.checkpoint.v1",
                    "snapshot": {"type": "checkpoint"},
                },
            )
        )
        db.commit()
        db.refresh(task)
        task_id = int(task.id)

        async def post_user_message(*_args: object, **_kwargs: object) -> bool:
            lease = current_task_lease()
            assert lease is not None
            assert lease.task_id == task_id
            assert lease.run_id is not None
            verify_db = _direct_db_session()
            try:
                leased = verify_db.query(Task).filter(Task.id == task_id).one()
                assert leased.run_id == lease.run_id
                assert leased.last_checkpoint_event_id is None
            finally:
                verify_db.close()
            return False

        agent_service = MagicMock()
        agent_service.post_user_message = AsyncMock(side_effect=post_user_message)
        agent_manager = MagicMock()
        agent_manager.get_agent_for_task = AsyncMock(return_value=agent_service)
        with patch(
            "xagent.web.api.chat.get_agent_manager",
            return_value=agent_manager,
        ):
            with pytest.raises(A2AApiError) as exc_info:
                await a2a_api._resume_input_required_a2a_task(
                    agent_id=agent_id,
                    task_owner_user_id=int(agent.user_id),
                    task=A2ATaskSnapshot.from_task(task),
                    text="legacy follow up",
                    message_id="msg-legacy",
                )

        assert exc_info.value.status_code == 400
        assert exc_info.value.payload["details"][0]["reason"] == "UNSUPPORTED_OPERATION"
        db.refresh(task)
        assert task.status == TaskStatus.WAITING_FOR_USER
        assert task.control_state == TaskControlState.WAITING_FOR_USER.value
        assert task.runner_id is None
        assert task.lease_expires_at is None
        assert task.run_id is not None
    finally:
        db.close()


def test_checkpoint_resume_rejects_duplicate_request_while_exact_lease_is_live() -> (
    None
):
    agent_id, full_key = _create_published_agent_with_key()
    db = _direct_db_session()
    try:
        owner_id = int(db.query(Agent).filter(Agent.id == agent_id).one().user_id)
        task = Task(
            user_id=owner_id,
            title="resuming",
            status=TaskStatus.RUNNING,
            control_state=TaskControlState.RUNNING.value,
            run_id="run-a",
            runner_id="runner-a",
            lease_expires_at=datetime.now(UTC) + timedelta(minutes=1),
            agent_id=agent_id,
            source="a2a",
            is_visible=False,
            agent_config={"a2a_context_id": "ctx-duplicate"},
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = int(task.id)
    finally:
        db.close()

    begin_turn = AsyncMock(side_effect=TaskTurnError("busy"))
    with (
        patch(
            "xagent.web.api.chat.get_agent_manager",
        ) as get_agent_manager,
        patch(
            "xagent.web.api.a2a.TaskTurnOrchestrator.begin_turn",
            new=begin_turn,
        ),
    ):
        response = client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=_bearer(full_key),
            json={
                "message": {
                    "messageId": "msg-duplicate",
                    "taskId": task_id,
                    "role": "ROLE_USER",
                    "parts": [{"text": "do not inject twice"}],
                },
                "configuration": {"returnImmediately": True},
            },
        )

    assert response.status_code == 400
    get_agent_manager.assert_not_called()
    begin_turn.assert_awaited_once()


def test_failed_follow_up_restores_input_required_status() -> None:
    agent_id, full_key = _create_published_agent_with_key()
    db = _direct_db_session()
    try:
        owner_id = int(db.query(Agent).filter(Agent.id == agent_id).one().user_id)
        task = Task(
            user_id=owner_id,
            title="waiting",
            status=TaskStatus.WAITING_FOR_USER,
            agent_id=agent_id,
            source="a2a",
            is_visible=False,
            agent_config={"a2a_context_id": "ctx-recover"},
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = int(task.id)
    finally:
        db.close()

    agent_service = MagicMock()
    agent_service.post_user_message = AsyncMock(return_value=False)
    agent_manager = MagicMock()
    agent_manager.get_agent_for_task = AsyncMock(return_value=agent_service)
    with (
        patch(
            "xagent.web.api.chat.get_agent_manager",
            return_value=agent_manager,
        ),
        patch(
            "xagent.web.api.a2a.TaskTurnOrchestrator.begin_turn",
            side_effect=TaskTurnError("busy"),
        ),
    ):
        response = client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=_bearer(full_key),
            json={
                "message": {
                    "messageId": "msg-recover",
                    "taskId": task_id,
                    "role": "ROLE_USER",
                    "parts": [{"text": "retry"}],
                },
                "configuration": {"returnImmediately": True},
            },
        )

    assert response.status_code == 400, response.text
    db = _direct_db_session()
    try:
        recovered = db.query(Task).filter(Task.id == task_id).one()
        assert recovered.status == TaskStatus.WAITING_FOR_USER
    finally:
        db.close()


def test_failed_follow_up_leaves_a_still_active_question_and_marker_untouched() -> None:
    """Injection never happened here (post_user_message returned False), so
    the prelease restore's marker-clear compensation must not fire: the
    active interaction row and the protocol marker are both untouched.

    Deleting the NOT EXISTS guard from clear_interaction_marker_if_unpaired
    would turn this red -- the marker would be zeroed out from under a
    question that is still active.
    """
    agent_id, full_key = _create_published_agent_with_key()
    db = _direct_db_session()
    try:
        owner_id = int(db.query(Agent).filter(Agent.id == agent_id).one().user_id)
        task = Task(
            user_id=owner_id,
            title="waiting with an open question",
            status=TaskStatus.WAITING_FOR_USER,
            run_id="run-not-posted",
            agent_id=agent_id,
            source="a2a",
            is_visible=False,
            agent_config={"a2a_context_id": "ctx-not-posted"},
            interaction_protocol_version=1,
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = int(task.id)
        row_id = _seed_active_interaction_row(
            db,
            task_id=task_id,
            run_id="run-not-posted",
            idempotency_key="not-posted-q1",
        )
    finally:
        db.close()

    agent_service = MagicMock()
    agent_service.post_user_message = AsyncMock(return_value=False)
    agent_manager = MagicMock()
    agent_manager.get_agent_for_task = AsyncMock(return_value=agent_service)
    with patch(
        "xagent.web.api.chat.get_agent_manager",
        return_value=agent_manager,
    ):
        response = client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=_bearer(full_key),
            json={
                "message": {
                    "messageId": "msg-not-posted",
                    "taskId": task_id,
                    "role": "ROLE_USER",
                    "parts": [{"text": "no checkpoint available"}],
                },
                "configuration": {"returnImmediately": True},
            },
        )

    assert response.status_code == 400, response.text
    db = _direct_db_session()
    try:
        recovered = db.query(Task).filter(Task.id == task_id).one()
        assert recovered.status == TaskStatus.WAITING_FOR_USER
        assert recovered.interaction_protocol_version == 1
        row = (
            db.query(TaskInteractionRequest)
            .filter(TaskInteractionRequest.id == row_id)
            .one()
        )
        assert row.status == "active"
    finally:
        db.close()


def test_prelease_restore_from_a_cancelled_acquisition_leaves_marker_untouched() -> (
    None
):
    """``_restore_a2a_resume_prelease_sync`` is also the cleanup callback
    ``acquire_task_lease_cancellation_safe`` invokes directly when the
    acquisition committed a prelease but the caller was then cancelled --
    no ``post_user_message`` outcome participates in that path at all.
    Called here exactly as that callback calls it: with a live active
    interaction row still in place, the marker must survive.

    Deleting the NOT EXISTS guard would turn this red the same way it does
    for the ``if not posted`` path above -- both routes converge on the
    same shared clear_interaction_marker_if_unpaired call.
    """
    agent_id, _full_key = _create_published_agent_with_key()
    db = _direct_db_session()
    try:
        owner_id = int(db.query(Agent).filter(Agent.id == agent_id).one().user_id)
        task = Task(
            user_id=owner_id,
            title="prelease acquired then cancelled",
            status=TaskStatus.RUNNING,
            runner_id="cancelled-acquire-runner",
            run_id="run-cancelled-acquire",
            lease_expires_at=datetime.now(UTC) + timedelta(minutes=1),
            agent_id=agent_id,
            source="a2a",
            is_visible=False,
            interaction_protocol_version=1,
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = int(task.id)
        row_id = _seed_active_interaction_row(
            db,
            task_id=task_id,
            run_id="run-cancelled-acquire",
            idempotency_key="cancelled-acquire-q1",
        )
    finally:
        db.close()

    acquired_lease = TaskLease(
        task_id=task_id,
        runner_id="cancelled-acquire-runner",
        run_id="run-cancelled-acquire",
    )
    restored = a2a_api._restore_a2a_resume_prelease_sync(
        acquired_lease, status=TaskStatus.WAITING_FOR_USER
    )

    assert restored is True
    db = _direct_db_session()
    try:
        recovered = db.query(Task).filter(Task.id == task_id).one()
        assert recovered.status == TaskStatus.WAITING_FOR_USER
        assert recovered.interaction_protocol_version == 1
        row = (
            db.query(TaskInteractionRequest)
            .filter(TaskInteractionRequest.id == row_id)
            .one()
        )
        assert row.status == "active"
    finally:
        db.close()


def test_checkpoint_resume_exception_restores_input_required_status() -> None:
    agent_id, full_key = _create_published_agent_with_key()
    db = _direct_db_session()
    try:
        owner_id = int(db.query(Agent).filter(Agent.id == agent_id).one().user_id)
        task = Task(
            user_id=owner_id,
            title="waiting",
            status=TaskStatus.WAITING_FOR_USER,
            agent_id=agent_id,
            source="a2a",
            is_visible=False,
            agent_config={"a2a_context_id": "ctx-resume-error"},
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = int(task.id)
    finally:
        db.close()

    agent_service = MagicMock()
    agent_service.post_user_message = AsyncMock(
        side_effect=RuntimeError("checkpoint callback failed")
    )
    agent_manager = MagicMock()
    agent_manager.get_agent_for_task = AsyncMock(return_value=agent_service)
    with patch(
        "xagent.web.api.chat.get_agent_manager",
        return_value=agent_manager,
    ):
        response = client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=_bearer(full_key),
            json={
                "message": {
                    "messageId": "msg-resume-error",
                    "taskId": task_id,
                    "role": "ROLE_USER",
                    "parts": [{"text": "retry safely"}],
                },
                "configuration": {"returnImmediately": True},
            },
        )

    assert response.status_code == 500
    agent_service.post_user_message.assert_awaited_once_with(
        str(task_id),
        execution_message="retry safely",
        display_message="retry safely",
        turn_id=f"a2a:{task_id}:msg-resume-error",
        request_interrupt=False,
        reason="A2A input-required response",
    )
    db = _direct_db_session()
    try:
        recovered = db.query(Task).filter(Task.id == task_id).one()
        assert recovered.status == TaskStatus.WAITING_FOR_USER
    finally:
        db.close()


def _resume_error_task(agent_id: int, *, context_id: str) -> int:
    db = _direct_db_session()
    try:
        owner_id = int(db.query(Agent).filter(Agent.id == agent_id).one().user_id)
        task = Task(
            user_id=owner_id,
            title="waiting",
            status=TaskStatus.WAITING_FOR_USER,
            agent_id=agent_id,
            source="a2a",
            is_visible=False,
            agent_config={"a2a_context_id": context_id},
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        return int(task.id)
    finally:
        db.close()


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (CheckpointUnavailableError("checkpoint query failed"), 503),
        (CheckpointCorruptError("all matching rows undecodable"), 400),
        (
            CheckpointAccessRefusedError("active lease is not bound to this reader"),
            400,
        ),
    ],
)
def test_checkpoint_read_error_maps_to_distinct_status_and_restores_waiting(
    error: Exception,
    expected_status: int,
) -> None:
    """Unlike a generic exception (500, above), a checkpoint read failure
    gets a status distinguishing retryable (unavailable) from terminal
    (corrupt, refused) -- and, like the generic case, restores the prior
    input-required status since ownership was never transferred."""
    agent_id, full_key = _create_published_agent_with_key()
    task_id = _resume_error_task(agent_id, context_id=f"ctx-{type(error).__name__}")

    agent_service = MagicMock()
    agent_service.post_user_message = AsyncMock(side_effect=error)
    agent_manager = MagicMock()
    agent_manager.get_agent_for_task = AsyncMock(return_value=agent_service)
    with patch(
        "xagent.web.api.chat.get_agent_manager",
        return_value=agent_manager,
    ):
        response = client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=_bearer(full_key),
            json={
                "message": {
                    "messageId": f"msg-{type(error).__name__}",
                    "taskId": task_id,
                    "role": "ROLE_USER",
                    "parts": [{"text": "retry safely"}],
                },
                "configuration": {"returnImmediately": True},
            },
        )

    assert response.status_code == expected_status, response.text
    db = _direct_db_session()
    try:
        recovered = db.query(Task).filter(Task.id == task_id).one()
        assert recovered.status == TaskStatus.WAITING_FOR_USER
    finally:
        db.close()


def test_checkpoint_access_refused_reuses_existing_running_task_message() -> None:
    """Refused reuses the same 400 message as the pre-existing 'another run
    is already active' rejection -- it is the same fact from the reader's
    point of view, discovered later in the resume attempt."""
    agent_id, full_key = _create_published_agent_with_key()
    task_id = _resume_error_task(agent_id, context_id="ctx-refused-message")

    agent_service = MagicMock()
    agent_service.post_user_message = AsyncMock(
        side_effect=CheckpointAccessRefusedError("active lease is not bound")
    )
    agent_manager = MagicMock()
    agent_manager.get_agent_for_task = AsyncMock(return_value=agent_service)
    with patch(
        "xagent.web.api.chat.get_agent_manager",
        return_value=agent_manager,
    ):
        response = client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=_bearer(full_key),
            json={
                "message": {
                    "messageId": "msg-refused-message",
                    "taskId": task_id,
                    "role": "ROLE_USER",
                    "parts": [{"text": "retry safely"}],
                },
                "configuration": {"returnImmediately": True},
            },
        )

    assert response.status_code == 400, response.text
    assert "currently running" in response.json()["error"]["message"]


@pytest.mark.parametrize(
    ("reason", "unexpected_phrase"),
    [
        ("lease_mismatch", "currently running"),
        ("superseded_legacy", "currently running"),
    ],
)
def test_checkpoint_access_refused_reason_gets_a_distinct_message(
    reason: str,
    unexpected_phrase: str,
) -> None:
    """Only the ``active_run`` reason reuses the pre-existing 'currently
    running' message; the other two refusal reasons are distinct facts
    (a stray lease, a superseded legacy partition) and must not be reported
    with a message that claims a run is in progress."""
    agent_id, full_key = _create_published_agent_with_key()
    task_id = _resume_error_task(agent_id, context_id=f"ctx-refused-{reason}")

    agent_service = MagicMock()
    agent_service.post_user_message = AsyncMock(
        side_effect=CheckpointAccessRefusedError(f"refused for {reason}", reason=reason)
    )
    agent_manager = MagicMock()
    agent_manager.get_agent_for_task = AsyncMock(return_value=agent_service)
    with patch(
        "xagent.web.api.chat.get_agent_manager",
        return_value=agent_manager,
    ):
        response = client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=_bearer(full_key),
            json={
                "message": {
                    "messageId": f"msg-refused-{reason}",
                    "taskId": task_id,
                    "role": "ROLE_USER",
                    "parts": [{"text": "retry safely"}],
                },
                "configuration": {"returnImmediately": True},
            },
        )

    assert response.status_code == 400, response.text
    message = response.json()["error"]["message"]
    assert unexpected_phrase not in message
    db = _direct_db_session()
    try:
        recovered = db.query(Task).filter(Task.id == task_id).one()
        assert recovered.status == TaskStatus.WAITING_FOR_USER
    finally:
        db.close()


def test_checkpoint_read_error_unknown_subclass_is_treated_as_retryable() -> None:
    """A ``CheckpointReadError`` subclass this dispatch does not recognize
    must default to the retryable (unavailable) branch, not silently fall
    into the terminal refused/corrupt handling -- conservative in the face
    of an unrecognized failure mode, matching the unavailable status code
    and the waiting-status restoration."""

    class _UnknownCheckpointReadError(CheckpointReadError):
        pass

    agent_id, full_key = _create_published_agent_with_key()
    task_id = _resume_error_task(agent_id, context_id="ctx-unknown-checkpoint-error")

    agent_service = MagicMock()
    agent_service.post_user_message = AsyncMock(
        side_effect=_UnknownCheckpointReadError("unrecognized checkpoint failure")
    )
    agent_manager = MagicMock()
    agent_manager.get_agent_for_task = AsyncMock(return_value=agent_service)
    with patch(
        "xagent.web.api.chat.get_agent_manager",
        return_value=agent_manager,
    ):
        response = client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=_bearer(full_key),
            json={
                "message": {
                    "messageId": "msg-unknown-checkpoint-error",
                    "taskId": task_id,
                    "role": "ROLE_USER",
                    "parts": [{"text": "retry safely"}],
                },
                "configuration": {"returnImmediately": True},
            },
        )

    assert response.status_code == 503, response.text
    db = _direct_db_session()
    try:
        recovered = db.query(Task).filter(Task.id == task_id).one()
        assert recovered.status == TaskStatus.WAITING_FOR_USER
    finally:
        db.close()


def test_list_tasks_uses_database_filters_and_pagination() -> None:
    agent_id, full_key = _create_published_agent_with_key()
    base_time = datetime.now(UTC) - timedelta(minutes=10)
    db = _direct_db_session()
    try:
        owner_id = int(db.query(Agent).filter(Agent.id == agent_id).one().user_id)
        task_specs = [
            (TaskStatus.PENDING, "ctx-a", {}, 0),
            (TaskStatus.RUNNING, "ctx-b", {}, 1),
            (TaskStatus.FAILED, "ctx-a", {"a2a_state": "TASK_STATE_CANCELED"}, 2),
            (TaskStatus.FAILED, "ctx-b", {}, 3),
            (TaskStatus.COMPLETED, "ctx-a", {}, 4),
        ]
        tasks: list[Task] = []
        for status, context_id, extra_config, minute in task_specs:
            task = Task(
                user_id=owner_id,
                title=f"task-{minute}",
                status=status,
                updated_at=base_time + timedelta(minutes=minute),
                agent_id=agent_id,
                source="a2a",
                is_visible=False,
                agent_config={"a2a_context_id": context_id, **extra_config},
            )
            db.add(task)
            tasks.append(task)
        db.commit()
        task_ids = [int(task.id) for task in tasks]
    finally:
        db.close()

    first = client.get(
        f"/api/a2a/agents/{agent_id}/tasks",
        headers=_bearer(full_key),
        params={"pageSize": 2},
    )
    second = client.get(
        f"/api/a2a/agents/{agent_id}/tasks",
        headers=_bearer(full_key),
        params={"pageSize": 2, "pageToken": first.json()["nextPageToken"]},
    )
    canceled = client.get(
        f"/api/a2a/agents/{agent_id}/tasks",
        headers=_bearer(full_key),
        params={"status": "TASK_STATE_CANCELED"},
    )
    failed = client.get(
        f"/api/a2a/agents/{agent_id}/tasks",
        headers=_bearer(full_key),
        params={"status": "TASK_STATE_FAILED"},
    )
    context = client.get(
        f"/api/a2a/agents/{agent_id}/tasks",
        headers=_bearer(full_key),
        params={"contextId": "ctx-a"},
    )
    recent = client.get(
        f"/api/a2a/agents/{agent_id}/tasks",
        headers=_bearer(full_key),
        params={"statusTimestampAfter": (base_time + timedelta(minutes=2)).isoformat()},
    )

    assert first.status_code == 200, first.text
    assert first.json()["totalSize"] == 5
    assert [item["id"] for item in first.json()["tasks"]] == [
        str(task_ids[4]),
        str(task_ids[3]),
    ]
    assert first.json()["nextPageToken"] == "2"
    assert [item["id"] for item in second.json()["tasks"]] == [
        str(task_ids[2]),
        str(task_ids[1]),
    ]
    assert canceled.json()["totalSize"] == 1
    assert canceled.json()["tasks"][0]["id"] == str(task_ids[2])
    assert failed.json()["totalSize"] == 1
    assert failed.json()["tasks"][0]["id"] == str(task_ids[3])
    assert context.json()["totalSize"] == 3
    assert recent.json()["totalSize"] == 2


def test_list_tasks_projects_claimed_waiting_resume_as_working() -> None:
    agent_id, full_key = _create_published_agent_with_key()
    db = _direct_db_session()
    try:
        owner_id = int(db.query(Agent).filter(Agent.id == agent_id).one().user_id)
        waiting = Task(
            user_id=owner_id,
            title="waiting",
            status=TaskStatus.WAITING_FOR_USER,
            control_state=TaskControlState.WAITING_FOR_USER.value,
            agent_id=agent_id,
            source="a2a",
            is_visible=False,
        )
        resume_requested = Task(
            user_id=owner_id,
            title="resume requested",
            status=TaskStatus.WAITING_FOR_USER,
            control_state=TaskControlState.RESUME_REQUESTED.value,
            agent_id=agent_id,
            source="a2a",
            is_visible=False,
        )
        db.add_all([waiting, resume_requested])
        db.commit()
        waiting_id = int(waiting.id)
        resume_requested_id = int(resume_requested.id)
    finally:
        db.close()

    working = client.get(
        f"/api/a2a/agents/{agent_id}/tasks",
        headers=_bearer(full_key),
        params={"status": "TASK_STATE_WORKING"},
    )
    input_required = client.get(
        f"/api/a2a/agents/{agent_id}/tasks",
        headers=_bearer(full_key),
        params={"status": "TASK_STATE_INPUT_REQUIRED"},
    )

    assert working.status_code == 200, working.text
    assert [int(task["id"]) for task in working.json()["tasks"]] == [
        resume_requested_id
    ]
    assert input_required.status_code == 200, input_required.text
    assert [int(task["id"]) for task in input_required.json()["tasks"]] == [waiting_id]


@pytest.mark.asyncio
async def test_stream_artifact_updates_are_incremental_and_finalize(
    monkeypatch,
) -> None:
    task = Task(
        id=101,
        user_id=1,
        title="stream",
        status=TaskStatus.RUNNING,
        output="part",
        agent_id=7,
        source="a2a",
        agent_config={"a2a_context_id": "ctx-artifact"},
    )
    running = Task(
        id=101,
        user_id=1,
        title="stream",
        status=TaskStatus.RUNNING,
        output="partial",
        agent_id=7,
        source="a2a",
        agent_config={"a2a_context_id": "ctx-artifact"},
    )
    completed = Task(
        id=101,
        user_id=1,
        title="stream",
        status=TaskStatus.COMPLETED,
        output="partial",
        agent_id=7,
        source="a2a",
        agent_config={"a2a_context_id": "ctx-artifact"},
    )
    fresh_tasks = iter([running, completed])

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(a2a_api.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(
        a2a_api,
        "_fetch_fresh_a2a_task",
        lambda _agent_id, _task_id: next(fresh_tasks),
    )

    response = a2a_api._task_stream_response(
        7,
        A2ATaskSnapshot.from_task(task),
    )
    events = [
        json.loads(chunk.removeprefix("data: "))
        async for chunk in response.body_iterator
    ]
    artifact_updates = [
        event["artifactUpdate"] for event in events if "artifactUpdate" in event
    ]

    assert artifact_updates[0]["artifact"]["parts"] == [{"text": "ial"}]
    assert artifact_updates[0]["append"] is True
    assert artifact_updates[0]["lastChunk"] is False
    assert artifact_updates[1]["artifact"]["parts"] == [{"text": "partial"}]
    assert artifact_updates[1]["append"] is False
    assert artifact_updates[1]["lastChunk"] is True


@pytest.mark.asyncio
async def test_a2a_poll_pool_wait_does_not_block_event_loop(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_engine(
        f"sqlite:///{tmp_path / 'a2a-poll-pool.db'}",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=1.0,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as db:
        db.add(
            Task(
                id=101,
                user_id=1,
                title="poll",
                status=TaskStatus.RUNNING,
                agent_id=7,
                source="a2a",
                is_visible=False,
                agent_config={"a2a_context_id": "ctx-poll"},
            )
        )
        db.commit()

    held_connection = engine.connect()
    monkeypatch.setattr(a2a_api, "get_session_local", lambda: SessionLocal)
    ticker_stop = asyncio.Event()
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while not ticker_stop.is_set():
            ticks += 1
            await asyncio.sleep(0.01)

    fetch_task = asyncio.create_task(a2a_api._fetch_fresh_a2a_task_isolated(7, 101))
    ticker_task = asyncio.create_task(ticker())
    try:
        await asyncio.sleep(0.08)
        assert ticks >= 3, "A2A QueuePool checkout blocked the event loop"
        assert not fetch_task.done()
    finally:
        held_connection.close()

    try:
        snapshot = await asyncio.wait_for(fetch_task, timeout=1.0)
        assert snapshot is not None
        assert snapshot.id == 101
    finally:
        ticker_stop.set()
        await ticker_task
        engine.dispose()


@pytest.mark.asyncio
async def test_a2a_task_page_pool_wait_runs_in_worker_without_blocking_loop(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page_loader = getattr(a2a_api, "_load_a2a_task_page_isolated", None)
    sync_loader = getattr(a2a_api, "_load_a2a_task_page_sync", None)
    assert page_loader is not None
    assert sync_loader is not None

    # The page loader must wait for the slot, never give up on it.
    engine = create_engine(
        f"sqlite:///{tmp_path / 'a2a-page-pool.db'}",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=CONTENTION_POOL_TIMEOUT,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as db:
        db.add(
            Task(
                id=101,
                user_id=1,
                title="page",
                status=TaskStatus.RUNNING,
                agent_id=7,
                source="a2a",
                is_visible=False,
                agent_config={"a2a_context_id": "ctx-page"},
            )
        )
        db.commit()

    held_connection = engine.connect()
    monkeypatch.setattr(a2a_api, "get_session_local", lambda: SessionLocal)
    worker_started = Event()
    worker_thread_ids: list[int] = []

    def observed_loader(**kwargs: object) -> object:
        worker_thread_ids.append(get_ident())
        worker_started.set()
        return sync_loader(**kwargs)

    monkeypatch.setattr(a2a_api, "_load_a2a_task_page_sync", observed_loader)
    ticker_stop = asyncio.Event()
    ticks = 0
    loop_thread_id = get_ident()

    async def ticker() -> None:
        nonlocal ticks
        while not ticker_stop.is_set():
            ticks += 1
            await asyncio.sleep(0.01)

    with gated_pool_checkout(engine) as gate:
        load_task = asyncio.create_task(
            page_loader(
                agent_id=7,
                context_id=None,
                status=None,
                status_timestamp_after=None,
                offset=0,
                page_size=50,
            )
        )
        ticker_task = asyncio.create_task(ticker())
        try:
            # `worker_started` only proves the loader was entered, not that it has
            # reached the pool; the gate is what establishes contention.
            assert await asyncio.to_thread(worker_started.wait, GUARD_TIMEOUT)
            await gate.wait_until_contending()
            observed = await wait_for_ticks(lambda: ticks)
            assert observed >= LOOP_LIVENESS_TICKS, (
                "A2A list QueuePool checkout blocked the event loop"
            )
            assert not load_task.done()
        finally:
            held_connection.close()
            gate.let_through()

    try:
        page = await asyncio.wait_for(load_task, timeout=GUARD_TIMEOUT)
        assert [task.id for task in page.tasks] == [101]
        assert page.total_size == 1
        assert worker_thread_ids
        assert worker_thread_ids[0] != loop_thread_id
    finally:
        ticker_stop.set()
        await ticker_task
        engine.dispose()


@pytest.mark.asyncio
async def test_subscribe_closes_loader_session_before_returning_stream(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert "db" not in inspect.signature(a2a_api.subscribe_task).parameters

    engine = create_engine(
        f"sqlite:///{tmp_path / 'a2a-subscribe-session.db'}",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=1.0,
    )
    Base.metadata.create_all(bind=engine)
    session_closed = Event()

    class TrackingSession(Session):
        def close(self) -> None:
            try:
                super().close()
            finally:
                session_closed.set()

    SessionLocal = sessionmaker(bind=engine, class_=TrackingSession)
    with SessionLocal() as db:
        db.add(
            Task(
                id=101,
                user_id=1,
                title="subscribe",
                status=TaskStatus.PAUSED,
                agent_id=7,
                source="a2a",
                is_visible=False,
                agent_config={"a2a_context_id": "ctx-subscribe"},
            )
        )
        db.commit()
    session_closed.clear()
    monkeypatch.setattr(a2a_api, "get_session_local", lambda: SessionLocal)
    agent = a2a_api.AgentPrincipalSnapshot(
        id=7,
        user_id=1,
        execution_mode="balanced",
        status="published",
        origin="user",
    )
    key = a2a_api.RuntimeApiKeySnapshot(key_prefix="ABCDEF")

    response = await a2a_api.subscribe_task(
        agent_id=7,
        task_id=101,
        authed=(agent, key),
    )

    assert session_closed.is_set()
    assert engine.pool.checkedout() == 0
    await response.body_iterator.aclose()
    engine.dispose()


@pytest.mark.asyncio
async def test_start_a2a_turn_cancellation_drains_atomic_create_into_scheduling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_id, _full_key = _create_published_agent_with_key()
    db = _direct_db_session()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).one()
        owner_user_id = int(agent.user_id)
        execution_mode = str(agent.execution_mode)
    finally:
        db.close()

    original_prepare = a2a_api._prepare_a2a_turn_sync
    preparation_committed = Event()
    allow_preparation_return = Event()
    prepared_task_ids: list[int] = []

    def delayed_prepare(**kwargs: object) -> a2a_api._A2ATurnPreparation:
        preparation = original_prepare(**kwargs)
        prepared_task_ids.append(preparation.task.id)
        preparation_committed.set()
        assert allow_preparation_return.wait(timeout=GUARD_TIMEOUT)
        return preparation

    begin_turn = AsyncMock()
    scheduled_claims: list[object] = []

    async def schedule_claimed_create_turn(**kwargs: object) -> MagicMock:
        scheduled_claims.append(kwargs["claimed"])

        async def noop() -> None:
            return None

        return MagicMock(background_task=asyncio.create_task(noop()))

    monkeypatch.setattr(a2a_api, "_prepare_a2a_turn_sync", delayed_prepare)
    monkeypatch.setattr(a2a_api.TaskTurnOrchestrator, "begin_turn", begin_turn)
    monkeypatch.setattr(
        a2a_api.TaskTurnOrchestrator,
        "schedule_claimed_create_turn",
        schedule_claimed_create_turn,
    )

    turn = asyncio.create_task(
        a2a_api._start_a2a_turn(
            agent_id=agent_id,
            task_owner_user_id=owner_user_id,
            agent_execution_mode=execution_mode,
            text="cancelled during preparation",
            message_id="msg-cancel-prepare",
            context_id="ctx-cancel-prepare",
            task_id=None,
        )
    )
    assert await asyncio.to_thread(preparation_committed.wait, GUARD_TIMEOUT)

    turn.cancel()
    await asyncio.sleep(0)
    allow_preparation_return.set()

    with pytest.raises(asyncio.CancelledError):
        await turn

    begin_turn.assert_not_awaited()
    assert len(scheduled_claims) == 1
    assert prepared_task_ids
    db = _direct_db_session()
    try:
        task = db.get(Task, prepared_task_ids[0])
        assert task is not None
        assert task.status == TaskStatus.RUNNING
        assert task.runner_id is not None
        assert task.run_id is not None
        assert task.lease_expires_at is not None
    finally:
        db.close()


def test_a2a_send_has_no_request_session_during_runtime_await() -> None:
    agent_id, full_key = _create_published_agent_with_key()

    async def start_turn(**kwargs: object) -> A2ATaskSnapshot:
        return A2ATaskSnapshot(
            id=101,
            user_id=int(kwargs["task_owner_user_id"]),
            agent_id=int(kwargs["agent_id"]),
            source="a2a",
            status=TaskStatus.RUNNING,
            control_state=TaskControlState.RUNNING.value,
            run_id="run-a",
            agent_config={"a2a_context_id": "ctx-release"},
            output=None,
            error_message=None,
            updated_at=datetime.now(UTC),
        )

    assert "db" not in inspect.signature(a2a_api.send_message).parameters

    with (
        patch(
            "xagent.web.api.a2a._start_a2a_turn",
            new=AsyncMock(side_effect=start_turn),
        ),
        patch("xagent.web.api.a2a.record_key_usage"),
    ):
        response = client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=_bearer(full_key),
            json={
                "message": {
                    "messageId": "msg-release",
                    "role": "ROLE_USER",
                    "parts": [{"text": "hello"}],
                },
                "configuration": {"returnImmediately": True},
            },
        )

    assert response.status_code == 200, response.text


def test_cancel_is_idempotent_and_subscribe_rejects_terminal_task() -> None:
    agent_id, full_key = _create_published_agent_with_key()
    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ):
        created = client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=_bearer(full_key),
            json={
                "message": {
                    "messageId": "msg-cancel",
                    "role": "ROLE_USER",
                    "parts": [{"text": "cancel me"}],
                },
                "configuration": {"returnImmediately": True},
            },
        )
    task_id = created.json()["task"]["id"]

    first = client.post(
        f"/api/a2a/agents/{agent_id}/tasks/{task_id}:cancel",
        headers=_bearer(full_key),
    )
    second = client.post(
        f"/api/a2a/agents/{agent_id}/tasks/{task_id}:cancel",
        headers=_bearer(full_key),
    )
    subscribed = client.post(
        f"/api/a2a/agents/{agent_id}/tasks/{task_id}:subscribe",
        headers=_bearer(full_key),
    )

    assert first.status_code == 200, first.text
    assert first.json()["status"]["state"] == "TASK_STATE_CANCELED"
    assert second.status_code == 200, second.text
    assert second.json()["status"]["state"] == "TASK_STATE_CANCELED"
    assert subscribed.status_code == 400
    assert subscribed.json()["error"]["details"][0]["reason"] == "UNSUPPORTED_OPERATION"


def test_cancel_retries_a_previous_terminal_transport_failure() -> None:
    agent_id, full_key = _create_published_agent_with_key()
    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ):
        created = client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=_bearer(full_key),
            json={
                "message": {
                    "messageId": "msg-retry-cancel",
                    "role": "ROLE_USER",
                    "parts": [{"text": "cancel me after a retry"}],
                },
                "configuration": {"returnImmediately": True},
            },
        )
    assert created.status_code == 200, created.text
    task_id = int(created.json()["task"]["id"])

    db = _direct_db_session()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).one()
        task = db.query(Task).filter(Task.id == task_id).one()
        target_state_version = int(task.state_version or 0)
        db.add(
            TaskExecutionCommand(
                task_id=task_id,
                actor_user_id=int(agent.user_id),
                command_id=f"cancel:{task_id}:{target_state_version}",
                kind="cancel",
                payload={
                    "agent_id": agent_id,
                    "target_state_version": target_state_version,
                },
                target_run_id=str(task.run_id) if task.run_id is not None else None,
                target_runner_id=None,
                status=COMMAND_FAILED,
                attempt_count=1,
                failure_count=MAX_COMMAND_FAILURES,
                defer_count=MAX_COMMAND_DEFERS,
                error="temporary transport failure",
                completed_at=datetime.now(UTC),
            )
        )
        db.commit()
    finally:
        db.close()

    response = client.post(
        f"/api/a2a/agents/{agent_id}/tasks/{task_id}:cancel",
        headers=_bearer(full_key),
    )

    assert response.status_code == 200, response.text
    assert response.json()["status"]["state"] == "TASK_STATE_CANCELED"
    db = _direct_db_session()
    try:
        command = (
            db.query(TaskExecutionCommand)
            .filter(TaskExecutionCommand.task_id == task_id)
            .one()
        )
        assert command.status == "completed"
        assert command.failure_count == 0
        assert command.defer_count == 0
    finally:
        db.close()


def test_prepare_cancel_command_persists_exact_state_version_target() -> None:
    agent_id, _full_key = _create_published_agent_with_key()
    db = _direct_db_session()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).one()
        task = Task(
            user_id=int(agent.user_id),
            title="cancel target",
            status=TaskStatus.RUNNING,
            control_state=TaskControlState.RUNNING.value,
            run_id="run-target",
            state_version=7,
            agent_id=agent_id,
            source="a2a",
            is_visible=False,
            agent_config={"a2a_context_id": "ctx-cancel-target"},
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = int(task.id)
        owner_user_id = int(agent.user_id)
    finally:
        db.close()

    prepared = a2a_api._prepare_a2a_cancel_command_sync(
        task_id=task_id,
        agent_id=agent_id,
        task_owner_user_id=owner_user_id,
    )

    db = _direct_db_session()
    try:
        command = db.get(TaskExecutionCommand, prepared.command_db_id)
        assert command is not None
        assert command.command_id == f"cancel:{task_id}:7"
        assert command.target_run_id == "run-target"
        assert command.payload == {
            "agent_id": agent_id,
            "target_state_version": 7,
        }
    finally:
        db.close()


@pytest.mark.parametrize(
    ("current_state_version", "expected_state_version"),
    [(8, 8), (8, 7)],
)
def test_cancel_accepts_only_exact_canceled_terminal_replay(
    current_state_version: int,
    expected_state_version: int,
) -> None:
    agent_id, _full_key = _create_published_agent_with_key()
    db = _direct_db_session()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).one()
        task = Task(
            user_id=int(agent.user_id),
            title="completed cancel target",
            status=TaskStatus.FAILED,
            control_state=TaskControlState.FAILED.value,
            run_id="run-canceled",
            state_version=current_state_version,
            agent_id=agent_id,
            source="a2a",
            is_visible=False,
            agent_config={
                "a2a_context_id": "ctx-cancel-completed",
                "a2a_state": "TASK_STATE_CANCELED",
            },
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = int(task.id)
    finally:
        db.close()

    loaded = a2a_api._load_cancelable_a2a_task_sync(
        task_id=task_id,
        agent_id=agent_id,
        expected_run_id="run-canceled",
        expected_state_version=expected_state_version,
    )
    finalized = a2a_api._finalize_a2a_cancel_sync(
        task_id=task_id,
        agent_id=agent_id,
        expected_run_id="run-canceled",
        expected_state_version=expected_state_version,
        local_cancel_requested=False,
    )

    assert loaded.status == TaskStatus.FAILED
    assert finalized.status == TaskStatus.FAILED
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        assert task.state_version == current_state_version
    finally:
        db.close()


@pytest.mark.parametrize(
    (
        "current_run_id",
        "current_state_version",
        "status",
        "control_state",
    ),
    [
        ("run-replaced", 8, TaskStatus.FAILED, TaskControlState.FAILED.value),
        ("run-canceled", 9, TaskStatus.FAILED, TaskControlState.FAILED.value),
        ("run-canceled", 8, TaskStatus.RUNNING, TaskControlState.RUNNING.value),
    ],
)
@pytest.mark.parametrize(
    "operation",
    [
        a2a_api._load_cancelable_a2a_task_sync,
        a2a_api._finalize_a2a_cancel_sync,
    ],
)
def test_cancel_rejects_stale_or_nonterminal_canceled_marker(
    current_run_id: str,
    current_state_version: int,
    status: TaskStatus,
    control_state: str,
    operation,
) -> None:
    agent_id, _full_key = _create_published_agent_with_key()
    db = _direct_db_session()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).one()
        task = Task(
            user_id=int(agent.user_id),
            title="stale cancel marker",
            status=status,
            control_state=control_state,
            run_id=current_run_id,
            state_version=current_state_version,
            agent_id=agent_id,
            source="a2a",
            is_visible=False,
            agent_config={
                "a2a_context_id": "ctx-stale-cancel-marker",
                "a2a_state": "TASK_STATE_CANCELED",
            },
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = int(task.id)
    finally:
        db.close()

    kwargs = {
        "task_id": task_id,
        "agent_id": agent_id,
        "expected_run_id": "run-canceled",
        "expected_state_version": 7,
    }
    if operation is a2a_api._finalize_a2a_cancel_sync:
        kwargs["local_cancel_requested"] = False
    with pytest.raises(a2a_api.StaleTaskRunError):
        operation(**kwargs)


@pytest.mark.parametrize("initial_run_id", ["run-old", None])
@pytest.mark.asyncio
async def test_cancel_rejects_run_replaced_during_local_cancel_await(
    initial_run_id: str | None,
) -> None:
    from xagent.web.api import websocket as websocket_api

    agent_id, _full_key = _create_published_agent_with_key()
    db = _direct_db_session()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).one()
        task = Task(
            user_id=int(agent.user_id),
            title="cancel replacement race",
            status=(
                TaskStatus.RUNNING if initial_run_id is not None else TaskStatus.PENDING
            ),
            control_state=(
                TaskControlState.RUNNING.value
                if initial_run_id is not None
                else TaskControlState.IDLE.value
            ),
            run_id=initial_run_id,
            state_version=4,
            agent_id=agent_id,
            source="a2a",
            is_visible=False,
            agent_config={"a2a_context_id": "ctx-cancel-replacement"},
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = int(task.id)
    finally:
        db.close()

    async def replace_run_during_cancel(_task_id: int) -> MagicMock:
        concurrent_db = _direct_db_session()
        try:
            concurrent_task = concurrent_db.query(Task).filter(Task.id == task_id).one()
            concurrent_task.status = TaskStatus.RUNNING
            concurrent_task.control_state = TaskControlState.RUNNING.value
            concurrent_task.run_id = "run-new"
            concurrent_task.state_version = 5
            concurrent_db.commit()
        finally:
            concurrent_db.close()
        return MagicMock(requested=True)

    command = ClaimedTaskCommand(
        id=101,
        task_id=task_id,
        actor_user_id=None,
        command_id=f"cancel:{task_id}:4",
        kind=TaskCommandKind.CANCEL,
        payload={"agent_id": agent_id, "target_state_version": 4},
        target_run_id=initial_run_id,
        attempt_count=1,
    )
    with patch(
        "xagent.web.api.websocket.background_task_manager.cancel_task",
        new=AsyncMock(side_effect=replace_run_during_cancel),
    ):
        with pytest.raises(TaskCommandRejected) as exc_info:
            await websocket_api._execute_durable_task_command(command)

    assert exc_info.value.reason == "stale_run"
    db = _direct_db_session()
    try:
        current = db.query(Task).filter(Task.id == task_id).one()
        assert current.status == TaskStatus.RUNNING
        assert current.run_id == "run-new"
        assert current.state_version == 5
        assert current.agent_config.get("a2a_state") is None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_cancel_accepts_exact_same_run_local_settlement() -> None:
    from xagent.web.api import websocket as websocket_api
    from xagent.web.services.task_lease_service import get_runner_id

    agent_id, _full_key = _create_published_agent_with_key()
    db = _direct_db_session()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).one()
        task = Task(
            user_id=int(agent.user_id),
            title="cancel local settlement",
            status=TaskStatus.RUNNING,
            control_state=TaskControlState.RUNNING.value,
            run_id="run-local-settlement",
            state_version=4,
            runner_id=get_runner_id(),
            lease_expires_at=datetime.now(UTC) + timedelta(minutes=1),
            last_heartbeat_at=datetime.now(UTC),
            agent_id=agent_id,
            source="a2a",
            is_visible=False,
            agent_config={"a2a_context_id": "ctx-local-settlement"},
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = int(task.id)
    finally:
        db.close()

    async def settle_cancelled_local_run(_task_id: int) -> MagicMock:
        settle_db = _direct_db_session()
        try:
            settled = settle_db.query(Task).filter(Task.id == task_id).one()
            assert settled.run_id == "run-local-settlement"
            assert settled.state_version == 4
            settled.status = TaskStatus.FAILED
            settled.control_state = TaskControlState.FAILED.value
            settled.state_version = 5
            settled.runner_id = None
            settled.lease_expires_at = None
            settled.last_heartbeat_at = datetime.now(UTC)
            settled.output = None
            settled.error_message = "task execution cancelled"
            settle_db.commit()
        finally:
            settle_db.close()
        return MagicMock(requested=True)

    command = ClaimedTaskCommand(
        id=103,
        task_id=task_id,
        actor_user_id=None,
        command_id=f"cancel:{task_id}:4",
        kind=TaskCommandKind.CANCEL,
        payload={"agent_id": agent_id, "target_state_version": 4},
        target_run_id="run-local-settlement",
        attempt_count=1,
    )
    with patch(
        "xagent.web.api.websocket.background_task_manager.cancel_task",
        new=AsyncMock(side_effect=settle_cancelled_local_run),
    ):
        result = await websocket_api._execute_durable_task_command(command)

    assert result is not None
    db = _direct_db_session()
    try:
        canceled = db.query(Task).filter(Task.id == task_id).one()
        assert canceled.status == TaskStatus.FAILED
        assert canceled.control_state == TaskControlState.FAILED.value
        assert canceled.run_id == "run-local-settlement"
        assert canceled.state_version == 5
        assert canceled.runner_id is None
        assert canceled.lease_expires_at is None
        assert canceled.last_heartbeat_at is None
        assert canceled.agent_config["a2a_state"] == "TASK_STATE_CANCELED"
        assert canceled.error_message == "Task canceled by A2A client."
    finally:
        db.close()


@pytest.mark.parametrize(
    ("local_cancel_requested", "runner_id", "lease_expires_at"),
    [
        (False, None, None),
        (
            True,
            "still-settling-runner",
            datetime.now(UTC) + timedelta(minutes=1),
        ),
    ],
)
def test_cancel_rejects_unattributed_or_incomplete_failed_settlement(
    local_cancel_requested: bool,
    runner_id: str | None,
    lease_expires_at: datetime | None,
) -> None:
    agent_id, _full_key = _create_published_agent_with_key()
    db = _direct_db_session()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).one()
        task = Task(
            user_id=int(agent.user_id),
            title="unattributed cancel settlement",
            status=TaskStatus.FAILED,
            control_state=TaskControlState.FAILED.value,
            run_id="run-unattributed-settlement",
            state_version=5,
            runner_id=runner_id,
            lease_expires_at=lease_expires_at,
            last_heartbeat_at=datetime.now(UTC),
            agent_id=agent_id,
            source="a2a",
            is_visible=False,
            agent_config={"a2a_context_id": "ctx-unattributed-settlement"},
            error_message="task execution failed",
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = int(task.id)
    finally:
        db.close()

    with pytest.raises(a2a_api.StaleTaskRunError):
        a2a_api._finalize_a2a_cancel_sync(
            task_id=task_id,
            agent_id=agent_id,
            expected_run_id="run-unattributed-settlement",
            expected_state_version=4,
            local_cancel_requested=local_cancel_requested,
        )

    db = _direct_db_session()
    try:
        unchanged = db.query(Task).filter(Task.id == task_id).one()
        assert unchanged.state_version == 5
        assert unchanged.agent_config.get("a2a_state") is None
        assert unchanged.error_message == "task execution failed"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_direct_cancel_atomically_clears_execution_lease() -> None:
    from xagent.web.services.task_lease_service import get_runner_id

    agent_id, _full_key = _create_published_agent_with_key()
    db = _direct_db_session()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).one()
        task = Task(
            user_id=int(agent.user_id),
            title="direct cancel lease cleanup",
            status=TaskStatus.RUNNING,
            control_state=TaskControlState.RUNNING.value,
            run_id="run-direct-cancel",
            state_version=6,
            runner_id=get_runner_id(),
            lease_attempt_id="attempt-direct-cancel",
            lease_expires_at=datetime.now(UTC) + timedelta(minutes=1),
            last_heartbeat_at=datetime.now(UTC),
            agent_id=agent_id,
            source="a2a",
            is_visible=False,
            agent_config={"a2a_context_id": "ctx-direct-cancel"},
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = int(task.id)
    finally:
        db.close()

    with patch(
        "xagent.web.api.websocket.background_task_manager.cancel_task",
        new=AsyncMock(return_value=MagicMock(requested=False)),
    ):
        async with a2a_api.task_execution_controller.command(task_id):
            await a2a_api._cancel_task_unserialized(
                task_id=task_id,
                agent_id=agent_id,
                expected_run_id="run-direct-cancel",
                expected_state_version=6,
            )

    db = _direct_db_session()
    try:
        canceled = db.query(Task).filter(Task.id == task_id).one()
        assert canceled.status == TaskStatus.FAILED
        assert canceled.control_state == TaskControlState.FAILED.value
        assert canceled.run_id == "run-direct-cancel"
        assert canceled.state_version == 7
        assert canceled.runner_id is None
        assert canceled.lease_attempt_id is None
        assert canceled.lease_expires_at is None
        assert canceled.last_heartbeat_at is None
        assert canceled.agent_config["a2a_state"] == "TASK_STATE_CANCELED"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_cancel_holds_local_command_gate_until_final_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.api import websocket as websocket_api

    agent_id, _full_key = _create_published_agent_with_key()
    db = _direct_db_session()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).one()
        task = Task(
            user_id=int(agent.user_id),
            title="cancel local gate",
            status=TaskStatus.RUNNING,
            control_state=TaskControlState.RUNNING.value,
            run_id="run-gated",
            state_version=3,
            agent_id=agent_id,
            source="a2a",
            is_visible=False,
            agent_config={"a2a_context_id": "ctx-cancel-gate"},
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = int(task.id)
        owner_user_id = int(agent.user_id)
    finally:
        db.close()

    cancel_entered = asyncio.Event()
    allow_cancel = asyncio.Event()
    begin_entered = asyncio.Event()

    async def blocking_cancel(_task_id: int) -> MagicMock:
        cancel_entered.set()
        await allow_cancel.wait()
        return MagicMock(requested=True)

    async def observe_begin(**_kwargs: object) -> MagicMock:
        begin_entered.set()
        return MagicMock()

    monkeypatch.setattr(
        websocket_api.background_task_manager,
        "cancel_task",
        blocking_cancel,
    )
    monkeypatch.setattr(
        TaskTurnOrchestrator,
        "_begin_turn_unserialized",
        observe_begin,
    )
    command = ClaimedTaskCommand(
        id=102,
        task_id=task_id,
        actor_user_id=owner_user_id,
        command_id=f"cancel:{task_id}:3",
        kind=TaskCommandKind.CANCEL,
        payload={"agent_id": agent_id, "target_state_version": 3},
        target_run_id="run-gated",
        attempt_count=1,
    )

    cancel_command = asyncio.create_task(
        websocket_api._execute_durable_task_command(command)
    )
    await cancel_entered.wait()
    begin_turn = asyncio.create_task(
        TaskTurnOrchestrator.begin_turn(
            task_id=task_id,
            task_owner_user_id=owner_user_id,
            actor_user_id=owner_user_id,
            payload=TaskTurnPayload("new local turn"),
            kind=TurnKind.APPEND,
        )
    )
    await asyncio.sleep(0.05)
    entered_while_canceling = begin_entered.is_set()
    allow_cancel.set()
    await cancel_command
    await begin_turn

    assert not entered_while_canceling
    assert begin_entered.is_set()


def test_cancel_maps_stale_run_rejection_to_conflict() -> None:
    agent_id, full_key = _create_published_agent_with_key()
    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ):
        created = client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=_bearer(full_key),
            json={
                "message": {
                    "messageId": "msg-stale-cancel",
                    "role": "ROLE_USER",
                    "parts": [{"text": "rotate before cancel"}],
                },
                "configuration": {"returnImmediately": True},
            },
        )
    assert created.status_code == 200, created.text
    task_id = int(created.json()["task"]["id"])
    real_dispatch = a2a_api.dispatch_one_task_command

    async def rotate_then_dispatch(executor, *, command_db_id=None):
        db = _direct_db_session()
        try:
            task = db.query(Task).filter(Task.id == task_id).one()
            task.run_id = "rotated-before-cancel"
            task.runner_id = None
            task.lease_expires_at = None
            db.commit()
        finally:
            db.close()
        return await real_dispatch(executor, command_db_id=command_db_id)

    with patch.object(
        a2a_api,
        "dispatch_one_task_command",
        new=rotate_then_dispatch,
    ):
        response = client.post(
            f"/api/a2a/agents/{agent_id}/tasks/{task_id}:cancel",
            headers=_bearer(full_key),
        )

    assert response.status_code == 409, response.text
    assert response.json()["error"]["details"][0]["reason"] == "INVALID_REQUEST"
    assert "run changed" in response.json()["error"]["message"].lower()


@pytest.mark.asyncio
async def test_cancel_does_not_overwrite_a_concurrent_completion() -> None:
    agent_id, _full_key = _create_published_agent_with_key()
    db = _direct_db_session()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).one()
        task = Task(
            user_id=int(agent.user_id),
            title="cancel completion race",
            status=TaskStatus.RUNNING,
            agent_id=agent_id,
            source="a2a",
            is_visible=False,
            agent_config={"a2a_context_id": "ctx-cancel-race"},
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = int(task.id)
        expected_run_id = str(task.run_id) if task.run_id is not None else None
        expected_state_version = int(task.state_version or 0)

        async def complete_during_cancel(_task_id: int) -> MagicMock:
            concurrent_db = _direct_db_session()
            try:
                concurrent_task = (
                    concurrent_db.query(Task).filter(Task.id == task_id).one()
                )
                concurrent_task.status = TaskStatus.COMPLETED
                concurrent_task.output = "completed concurrently"
                concurrent_db.commit()
            finally:
                concurrent_db.close()
            return MagicMock(requested=True)

        with patch(
            "xagent.web.api.websocket.background_task_manager.cancel_task",
            new=AsyncMock(side_effect=complete_during_cancel),
        ):
            with pytest.raises(a2a_api.StaleTaskRunError):
                async with a2a_api.task_execution_controller.command(task_id):
                    await a2a_api._cancel_task_unserialized(
                        task_id=task_id,
                        agent_id=agent_id,
                        expected_run_id=expected_run_id,
                        expected_state_version=expected_state_version,
                    )

        db.expire_all()
        completed = db.query(Task).filter(Task.id == task_id).one()
        assert completed.status == TaskStatus.COMPLETED
        assert completed.output == "completed concurrently"
        assert completed.error_message is None
    finally:
        db.close()


def test_subscribe_stream_starts_with_wrapped_task_snapshot() -> None:
    agent_id, full_key = _create_published_agent_with_key()
    db = _direct_db_session()
    try:
        owner_id = int(db.query(Agent).filter(Agent.id == agent_id).one().user_id)
        task = Task(
            user_id=owner_id,
            title="paused",
            status=TaskStatus.PAUSED,
            agent_id=agent_id,
            source="a2a",
            is_visible=False,
            agent_config={"a2a_context_id": "ctx-stream"},
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = int(task.id)
    finally:
        db.close()

    response = client.post(
        f"/api/a2a/agents/{agent_id}/tasks/{task_id}:subscribe",
        headers=_bearer(full_key),
    )

    assert response.status_code == 200, response.text
    data_lines = [
        line.removeprefix("data: ")
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert len(data_lines) == 1
    event = json.loads(data_lines[0])
    assert event["task"]["id"] == str(task_id)
    assert event["task"]["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"


def test_subscribe_stream_ends_at_server_lifetime_limit(monkeypatch) -> None:
    agent_id, full_key = _create_published_agent_with_key()
    db = _direct_db_session()
    try:
        owner_id = int(db.query(Agent).filter(Agent.id == agent_id).one().user_id)
        task = Task(
            user_id=owner_id,
            title="running",
            status=TaskStatus.RUNNING,
            agent_id=agent_id,
            source="a2a",
            is_visible=False,
            agent_config={"a2a_context_id": "ctx-stream-limit"},
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = int(task.id)
    finally:
        db.close()
    monkeypatch.setattr(a2a_api, "A2A_STREAM_MAX_DURATION_SECONDS", 0.0)

    response = client.post(
        f"/api/a2a/agents/{agent_id}/tasks/{task_id}:subscribe",
        headers=_bearer(full_key),
    )

    assert response.status_code == 200, response.text
    data_lines = [
        line for line in response.text.splitlines() if line.startswith("data: ")
    ]
    assert len(data_lines) == 1
    event = json.loads(data_lines[0].removeprefix("data: "))
    assert event["task"]["status"]["state"] == "TASK_STATE_WORKING"


@pytest.mark.asyncio
async def test_a2a_resume_syncs_connector_runtime_turn_after_reservation() -> None:
    """An A2A input-required resume must sync the connector runtime turn
    binding for the turn whose message it just injected, once (and only
    once) it is the sole admitted owner of the cached agent - not on a
    cache-hit fetch that predates admission. Guards the A2A-vs-websocket-
    resume race: without this sync, an A2A resume that inherits a cached
    agent left mid-turn by a losing websocket resume would execute against
    that losing turn's tool_config/ephemeral-secret binding instead of its
    own."""

    from xagent.web.api import websocket as websocket_api

    real_manager = websocket_api.BackgroundTaskManager()
    lease = TaskLease(task_id=5454, runner_id="runner-z", run_id="run-a2a-sync")
    resume_gate = asyncio.Event()
    mgr = MagicMock()

    async def execute_resume_background(**_kwargs) -> None:
        await resume_gate.wait()

    with (
        patch.object(websocket_api, "background_task_manager", real_manager),
        patch.object(
            websocket_api,
            "execute_resume_background",
            side_effect=execute_resume_background,
        ),
        patch("xagent.web.api.chat.get_agent_manager", return_value=mgr),
    ):
        await a2a_api._schedule_waiting_a2a_resume(
            task_id=5454,
            agent_service=MagicMock(),
            task_owner_user_id=1,
            task_lease=lease,
            heartbeat_stop=asyncio.Event(),
            heartbeat_task=asyncio.ensure_future(asyncio.sleep(0)),
            resumable_status=TaskStatus.WAITING_FOR_USER,
            connector_runtime_turn_id="a2a:5454:the-real-turn",
        )
        try:
            mgr.sync_connector_runtime_turn.assert_called_once_with(
                5454, "a2a:5454:the-real-turn"
            )
        finally:
            resume_gate.set()


def test_subscribe_projects_claimed_waiting_resume_as_working(monkeypatch) -> None:
    agent_id, full_key = _create_published_agent_with_key()
    db = _direct_db_session()
    try:
        owner_id = int(db.query(Agent).filter(Agent.id == agent_id).one().user_id)
        task = Task(
            user_id=owner_id,
            title="resume requested",
            status=TaskStatus.WAITING_FOR_USER,
            control_state=TaskControlState.RESUME_REQUESTED.value,
            agent_id=agent_id,
            source="a2a",
            is_visible=False,
            agent_config={"a2a_context_id": "ctx-resume-requested"},
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = int(task.id)
        assert not a2a_api._task_stream_ended(task)
    finally:
        db.close()
    monkeypatch.setattr(a2a_api, "A2A_STREAM_MAX_DURATION_SECONDS", 0.0)

    response = client.post(
        f"/api/a2a/agents/{agent_id}/tasks/{task_id}:subscribe",
        headers=_bearer(full_key),
    )

    assert response.status_code == 200, response.text
    data_lines = [
        line for line in response.text.splitlines() if line.startswith("data: ")
    ]
    assert len(data_lines) == 1
    event = json.loads(data_lines[0].removeprefix("data: "))
    assert event["task"]["status"]["state"] == "TASK_STATE_WORKING"
