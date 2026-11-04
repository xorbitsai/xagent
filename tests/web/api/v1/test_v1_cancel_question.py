"""Integration tests for ``POST /v1/chat/tasks/{task_id}/cancel-question``.

Covers both waiting-task shapes the read surface supports: a native
``task_interaction_requests`` row and the legacy transcript-only question.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from xagent.core.agent.runner import UserMessageInjectionOutcome
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.task_execution_event import TaskExecutionEvent
from xagent.web.models.task_interaction import TaskInteractionRequest
from xagent.web.services import task_event_bridge
from xagent.web.services.task_command_transport import (
    COMMAND_PENDING,
    COMMAND_PROCESSING,
    TaskCommandKind,
)
from xagent.web.services.task_coordinator_service import (
    acquire_task_lease_no_commit,
    release_task_lease_no_commit,
)
from xagent.web.services.task_execution_controller import TaskControlState
from xagent.web.services.task_lease_service import TaskLease
from xagent.web.services.task_orchestrator import finish_turn
from xagent.web.services.task_question_cancel import (
    TaskQuestionCancelNotWaiting,
    _cancel_waiting_task,
    _pending_resume_command_exists,
)
from xagent.web.services.task_start import SdkTaskScope

from ..conftest import (
    _admin_headers,
    _direct_db_session,
    _register_second_user,
    client,
)
from .test_workforces import _create_active_workforce, _create_workforce_key

pytestmark = pytest.mark.usefixtures("_test_db")


_agent_name_counter = 0


def _bearer(full_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {full_key}"}


@pytest.fixture(autouse=True)
def mock_start_task():
    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ):
        yield


def _create_agent_with_key(
    headers: dict[str, str] | None = None,
) -> tuple[int, str]:
    global _agent_name_counter
    _agent_name_counter += 1
    headers = headers or _admin_headers()
    agent_resp = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": f"v1 cancel-question test agent {_agent_name_counter}",
            "description": "test",
            "instructions": "you are a test agent",
            "execution_mode": "balanced",
        },
    )
    assert agent_resp.status_code == 200, agent_resp.text
    agent_id = agent_resp.json()["id"]
    key_resp = client.post(f"/api/agents/{agent_id}/api-key", headers=headers)
    assert key_resp.status_code == 200, key_resp.text
    return agent_id, key_resp.json()["full_key"]


def _create_task(full_key: str, agent_id: int, content: str = "hello") -> int:
    response = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={"agent_id": agent_id, "message": {"role": "user", "content": content}},
    )
    assert response.status_code == 202, response.text
    return response.json()["task_id"]


def _force_waiting(task_id: int, *, run_id: str | None = "run-original") -> None:
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        task.status = TaskStatus.WAITING_FOR_USER
        task.control_state = TaskControlState.WAITING_FOR_USER.value
        task.run_id = run_id
        task.runner_id = None
        task.lease_expires_at = None
        task.lease_attempt_id = None
        task.last_heartbeat_at = None
        db.commit()
    finally:
        db.close()


def _insert_question_message(
    task_id: int,
    *,
    content: str = "Continue?",
    interactions: list[dict] | None = None,
    message_type: str = "question",
    turn_id: str | None = None,
) -> int:
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        row = TaskChatMessage(
            task_id=task_id,
            user_id=int(task.user_id),
            role="assistant",
            content=content,
            message_type=message_type,
            interactions=interactions,
            turn_id=turn_id or f"question-{task_id}",
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return int(row.id)
    finally:
        db.close()


def _seed_earlier_turn(
    task_id: int, *, user_text: str = "first question", answer: str = "first answer"
) -> None:
    now = datetime.now(timezone.utc) - timedelta(minutes=5)
    db = _direct_db_session()
    try:
        db.add(
            TraceEvent(
                task_id=task_id,
                event_id=f"user-{task_id}",
                event_type="user_message",
                timestamp=now,
                data={"message": user_text},
            )
        )
        db.add(
            TraceEvent(
                task_id=task_id,
                event_id=f"answer-{task_id}",
                event_type="ai_message",
                timestamp=now + timedelta(seconds=1),
                data={"content": answer},
            )
        )
        db.commit()
    finally:
        db.close()


def _seed_active_interaction_row(
    task_id: int, *, run_id: str, idempotency_key: str = "cancel-key"
) -> int:
    db = _direct_db_session()
    try:
        anchor = TraceEvent(
            task_id=task_id,
            event_id=f"anchor-{idempotency_key}",
            event_type="agent_execution_checkpoint",
            timestamp=datetime.now(timezone.utc),
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
            origin="sdk",
            request_payload={
                "message": "Continue?",
                "interactions": [
                    {
                        "type": "text_input",
                        "field": "response",
                        "label": "Your answer",
                    }
                ],
            },
            request_idempotency_key=idempotency_key,
            resume_trace_event_id=int(anchor.id),
            resume_event_id="resume-event-1",
            resume_execution_id="resume-execution-1",
            resume_locator_format="trace_event_pk_v1",
            resume_checkpoint_type="agent_execution_checkpoint",
            resume_run_partition=run_id,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
        )
        db.add(row)
        task = db.query(Task).filter(Task.id == task_id).one()
        task.interaction_protocol_version = 1
        db.commit()
        db.refresh(row)
        return int(row.id)
    finally:
        db.close()


def _cancel(task_id: int, full_key: str):
    return client.post(
        f"/v1/chat/tasks/{task_id}/cancel-question", headers=_bearer(full_key)
    )


def _append_with_key(
    task_id: int, agent_id: int, full_key: str, content: str = "new turn"
):
    return client.post(
        f"/v1/chat/tasks/{task_id}/messages",
        headers=_bearer(full_key),
        json={"agent_id": agent_id, "message": {"role": "user", "content": content}},
    )


def _steps(task_id: int, full_key: str) -> list[dict]:
    response = client.get(f"/v1/chat/tasks/{task_id}/steps", headers=_bearer(full_key))
    assert response.status_code == 200, response.text
    return response.json()["steps"]


def _message_contents(steps: list[dict], role: str) -> list[str]:
    return [
        step["data"]["content"]
        for step in steps
        if step["type"] == "message" and step["data"]["role"] == role
    ]


def _task_row(task_id: int) -> Task:
    db = _direct_db_session()
    try:
        return db.query(Task).filter(Task.id == task_id).one()
    finally:
        db.close()


# ===== shared execution ownership =====


def _hold_ownership(task_id: int) -> TaskLease:
    """Leave the first coordinator holding the task, as while it drains."""
    lease = TaskLease(task_id, "worker-a", "run-original", "attempt-a")
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        task.runner_id = lease.runner_id
        task.lease_attempt_id = lease.attempt_id
        task.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)
        db.commit()
    finally:
        db.close()
    return lease


def test_shared_cancel_keeps_ownership_until_the_coordinator_drains(monkeypatch):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setattr(task_event_bridge, "_bridge", Mock())
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _seed_earlier_turn(task_id)
    _force_waiting(task_id)
    _insert_question_message(task_id)
    lease = _hold_ownership(task_id)

    response = _cancel(task_id, full_key)

    assert response.status_code == 202, response.text
    task = _task_row(task_id)
    assert task.status == TaskStatus.COMPLETED
    assert (task.runner_id, task.lease_attempt_id) == ("worker-a", "attempt-a")
    assert task.lease_expires_at is not None

    db = _direct_db_session()
    try:
        # Another worker cannot take the task while the first still owns it.
        assert acquire_task_lease_no_commit(db, task_id, runner_id="worker-b") is None
        db.rollback()
        # The owning coordinator's drain releases it, and only then can
        # another worker take over.
        with db.begin():
            assert release_task_lease_no_commit(db, lease) is True
    finally:
        db.close()
    task = _task_row(task_id)
    assert (task.runner_id, task.lease_attempt_id, task.lease_expires_at) == (
        None,
        None,
        None,
    )
    db = _direct_db_session()
    try:
        with db.begin():
            assert (
                acquire_task_lease_no_commit(db, task_id, runner_id="worker-b")
                is not None
            )
    finally:
        db.close()


def test_shared_finalization_after_cancel_does_not_resurrect_the_question(
    monkeypatch,
):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setattr(task_event_bridge, "_bridge", Mock())
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _seed_earlier_turn(task_id)
    _force_waiting(task_id)
    _insert_question_message(task_id, content="Where are you flying to?")
    lease = _hold_ownership(task_id)
    assert _cancel(task_id, full_key).status_code == 202

    # The still-matching owner reaches outer settlement after the cancel.
    db = _direct_db_session()
    try:
        assert finish_turn(db, task_id, task_lease=lease) is True
    finally:
        db.close()

    task = _task_row(task_id)
    assert task.status == TaskStatus.COMPLETED
    assert task.output is None
    read = client.get(f"/v1/chat/tasks/{task_id}", headers=_bearer(full_key))
    assert read.status_code == 200, read.text
    assert read.json()["output"] is None


def test_local_cancel_releases_ownership(monkeypatch):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "false")
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _seed_earlier_turn(task_id)
    _force_waiting(task_id)
    _insert_question_message(task_id)
    _hold_ownership(task_id)

    response = _cancel(task_id, full_key)

    assert response.status_code == 202, response.text
    task = _task_row(task_id)
    assert (task.runner_id, task.lease_attempt_id, task.lease_expires_at) == (
        None,
        None,
        None,
    )


# ===== legacy transcript path =====


def test_owner_cancels_legacy_question_then_appends_on_same_task():
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _seed_earlier_turn(task_id)
    _force_waiting(task_id)
    question_id = _insert_question_message(
        task_id,
        content="Where are you flying to?",
        interactions=[{"type": "text_input", "field": "city", "label": "City"}],
    )

    before = _steps(task_id, full_key)
    assert "Where are you flying to?" not in _message_contents(before, "assistant")

    response = _cancel(task_id, full_key)
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["task_id"] == task_id
    assert body["status"] == "completed"
    assert body["run_id"] == "run-original"
    assert isinstance(body["state_version"], int)
    assert body["state_version"] > 0

    task = _task_row(task_id)
    assert task.status == TaskStatus.COMPLETED
    assert task.control_state == TaskControlState.COMPLETED.value
    assert task.run_id == "run-original"

    db = _direct_db_session()
    try:
        question = (
            db.query(TaskChatMessage).filter(TaskChatMessage.id == question_id).one()
        )
        assert question.message_type == "question_superseded"
        assert question.content == "Where are you flying to?"
        assert (
            db.query(TaskInteractionRequest)
            .filter(TaskInteractionRequest.task_id == task_id)
            .count()
            == 0
        )
    finally:
        db.close()

    after = _steps(task_id, full_key)
    assistant = _message_contents(after, "assistant")
    assert assistant.count("first answer") == 1
    assert assistant.count("Where are you flying to?") == 1
    assert _message_contents(before, "assistant").count("first answer") == 1

    # The same task now accepts a fresh turn instead of 409.
    append = _append_with_key(task_id, agent_id, full_key, content="next turn")
    assert append.status_code == 202, append.text
    assert _task_row(task_id).run_id != "run-original"

    db = _direct_db_session()
    try:
        accepted = (
            db.query(TaskChatMessage)
            .filter(
                TaskChatMessage.task_id == task_id,
                TaskChatMessage.role == "user",
                TaskChatMessage.content == "next turn",
            )
            .one()
        )
        assert accepted.message_type == "user_message"
    finally:
        db.close()

    # The real background turn writes its user_message trace when it starts;
    # this test's scheduler is mocked, so stage that first trace explicitly.
    db = _direct_db_session()
    try:
        db.add(
            TraceEvent(
                task_id=task_id,
                event_id=f"user-next-{task_id}",
                event_type="user_message",
                timestamp=datetime.now(timezone.utc),
                data={"message": "next turn"},
            )
        )
        db.commit()
    finally:
        db.close()

    final = _steps(task_id, full_key)
    assert _message_contents(final, "user").count("next turn") == 1
    assert _message_contents(final, "assistant").count("Where are you flying to?") == 1


def test_repeated_cancel_is_idempotent_and_side_effect_free():
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _force_waiting(task_id)
    _insert_question_message(task_id)

    first = _cancel(task_id, full_key)
    assert first.status_code == 202, first.text

    db = _direct_db_session()
    try:
        trace_count = db.query(TraceEvent).filter(TraceEvent.task_id == task_id).count()
    finally:
        db.close()
    version = _task_row(task_id).state_version

    second = _cancel(task_id, full_key)
    assert second.status_code == 202, second.text
    assert second.json() == first.json()
    assert _task_row(task_id).state_version == version

    db = _direct_db_session()
    try:
        assert (
            db.query(TraceEvent).filter(TraceEvent.task_id == task_id).count()
            == trace_count
        )
        question = (
            db.query(TaskChatMessage)
            .filter(
                TaskChatMessage.task_id == task_id,
                TaskChatMessage.role == "assistant",
                TaskChatMessage.message_type == "question_superseded",
            )
            .one()
        )
        assert question.content == "Continue?"
    finally:
        db.close()


# ===== native interaction-row path =====


def test_owner_cancels_native_question_terminates_row_without_deleting_it():
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _force_waiting(task_id, run_id="run-native")
    _insert_question_message(task_id, content="Continue?")
    interaction_id = _seed_active_interaction_row(task_id, run_id="run-native")

    response = _cancel(task_id, full_key)
    assert response.status_code == 202, response.text
    assert response.json()["run_id"] == "run-native"
    assert response.json()["status"] == "completed"

    db = _direct_db_session()
    try:
        row = (
            db.query(TaskInteractionRequest)
            .filter(TaskInteractionRequest.id == interaction_id)
            .one()
        )
        assert row.status == "terminated"
        assert row.terminal_reason == "run_superseded"
        assert row.active_slot is None
        assert row.terminated_at is not None
        assert row.response_payload is None
        task = db.query(Task).filter(Task.id == task_id).one()
        assert task.interaction_protocol_version is None
    finally:
        db.close()

    assert (
        _message_contents(_steps(task_id, full_key), "assistant").count("Continue?")
        == 1
    )


# ===== ownership =====


def test_other_agents_task_is_404_and_untouched():
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _force_waiting(task_id)
    _insert_question_message(task_id)
    _other_agent_id, other_key = _create_agent_with_key()

    response = _cancel(task_id, other_key)
    assert response.status_code == 404, response.text
    assert response.json()["error"]["code"] == "task_not_found"

    task = _task_row(task_id)
    assert task.status == TaskStatus.WAITING_FOR_USER
    assert task.control_state == TaskControlState.WAITING_FOR_USER.value
    assert task.agent_config is None
    db = _direct_db_session()
    try:
        question = (
            db.query(TaskChatMessage)
            .filter(
                TaskChatMessage.task_id == task_id,
                TaskChatMessage.role == "assistant",
                TaskChatMessage.message_type == "question",
            )
            .one()
        )
        assert question.message_type == "question"
    finally:
        db.close()


def test_missing_task_is_404():
    _agent_id, full_key = _create_agent_with_key()
    response = _cancel(987654321, full_key)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "task_not_found"


# ===== status matrix =====


@pytest.mark.parametrize(
    ("status", "expected_code"),
    [
        (TaskStatus.PENDING, "task_not_waiting"),
        (TaskStatus.RUNNING, "task_busy"),
        (TaskStatus.PAUSED, "task_not_waiting"),
        (TaskStatus.COMPLETED, "task_not_waiting"),
        (TaskStatus.FAILED, "task_not_waiting"),
    ],
)
def test_non_waiting_statuses_are_409_and_untouched(status, expected_code):
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)

    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        task.status = status
        task.control_state = status.value
        task.run_id = "run-existing"
        task.state_version = 7
        db.commit()
        version = int(task.state_version)
    finally:
        db.close()

    response = _cancel(task_id, full_key)
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == expected_code

    task = _task_row(task_id)
    assert task.status == status
    assert task.state_version == version
    assert task.run_id == "run-existing"
    assert task.agent_config is None


# ===== interaction with reply =====


def _reply_body(agent_id: int, content: str = "yes") -> dict:
    return {"agent_id": agent_id, "message": {"role": "user", "content": content}}


@pytest.mark.parametrize("first", ["reply", "cancel"])
def test_reply_cancel_race_has_exactly_one_winner_in_both_orders(first):
    """Drive both interleavings of the same race: whichever claim commits
    first wins, and the loser gets one 409 -- never two winners."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _force_waiting(task_id)
    _insert_question_message(task_id)

    post_user_message = AsyncMock(return_value=UserMessageInjectionOutcome.POSTED_FRESH)
    agent_service = MagicMock()
    agent_service.post_user_message = post_user_message
    agent_manager = MagicMock()
    agent_manager.get_agent_for_task = AsyncMock(return_value=agent_service)

    def _reply():
        with (
            patch(
                "xagent.web.services.agent_service_manager.get_agent_manager",
                return_value=agent_manager,
            ),
            patch(
                "xagent.web.services.task_resume._schedule_waiting_reply_resume",
                new=AsyncMock(),
            ),
        ):
            return client.post(
                f"/v1/chat/tasks/{task_id}/reply",
                headers=_bearer(full_key),
                json=_reply_body(agent_id),
            )

    if first == "reply":
        winner, loser = _reply(), _cancel(task_id, full_key)
        assert winner.status_code == 202, winner.text
        assert loser.status_code == 409, loser.text
        assert loser.json()["error"]["code"] == "task_busy"
        assert _task_row(task_id).status == TaskStatus.RUNNING
    else:
        winner, loser = _cancel(task_id, full_key), _reply()
        assert winner.status_code == 202, winner.text
        assert loser.status_code == 409, loser.text
        assert loser.json()["error"]["code"] == "no_pending_interaction"
        task = _task_row(task_id)
        assert task.status == TaskStatus.COMPLETED
        db = _direct_db_session()
        try:
            assert (
                db.query(TaskChatMessage)
                .filter(
                    TaskChatMessage.task_id == task_id,
                    TaskChatMessage.role == "assistant",
                    TaskChatMessage.message_type == "question",
                )
                .count()
                == 0
            )
        finally:
            db.close()

    assert (
        sum(1 for response in (winner, loser) if 200 <= response.status_code < 300) == 1
    )
    assert sum(1 for response in (winner, loser) if response.status_code == 409) == 1


def test_reply_prelease_claimed_control_state_blocks_cancel():
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _force_waiting(task_id)
    _insert_question_message(task_id)
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        task.control_state = TaskControlState.RESUME_REQUESTED.value
        task.state_version += 1
        db.commit()
        version = int(task.state_version)
    finally:
        db.close()

    response = _cancel(task_id, full_key)
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "task_busy"
    task = _task_row(task_id)
    assert task.status == TaskStatus.WAITING_FOR_USER
    assert task.control_state == TaskControlState.RESUME_REQUESTED.value
    assert task.state_version == version


# ===== fences and idempotence internals =====


@pytest.mark.parametrize("fence", ["run_id", "state_version"])
def test_stale_claim_is_rejected_by_the_read_fence(fence):
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _force_waiting(task_id, run_id="run-old")
    db = _direct_db_session()
    try:
        db.expire_on_commit = False
        stale = db.query(Task).filter(Task.id == task_id).one()
        if fence == "run_id":
            values = {Task.run_id: "run-new"}
        else:
            values = {Task.state_version: int(stale.state_version or 0) + 1}
        db.query(Task).filter(Task.id == task_id).update(
            values, synchronize_session=False
        )
        db.commit()

        with pytest.raises(TaskQuestionCancelNotWaiting):
            _cancel_waiting_task(
                db,
                stale,
                SdkTaskScope(agent_id=agent_id, workforce_id=None),
            )
    finally:
        db.close()

    task = _task_row(task_id)
    assert task.status == TaskStatus.WAITING_FOR_USER
    assert task.control_state == TaskControlState.WAITING_FOR_USER.value
    assert task.run_id == ("run-new" if fence == "run_id" else "run-old")


def test_pending_resume_command_exists_ignores_settled_and_other_kinds():
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    db = _direct_db_session()
    try:
        assert bool(db.query(_pending_resume_command_exists(task_id)).scalar()) is False

        resume = TaskExecutionCommand(
            task_id=task_id,
            command_id="cmd-resume",
            kind=TaskCommandKind.RESUME_INPUT.value,
            payload={},
            status=COMMAND_PENDING,
        )
        db.add(resume)
        db.commit()
        assert bool(db.query(_pending_resume_command_exists(task_id)).scalar()) is True

        resume.status = COMMAND_PROCESSING
        db.commit()
        assert bool(db.query(_pending_resume_command_exists(task_id)).scalar()) is True

        resume.status = "completed"
        db.add(
            TaskExecutionCommand(
                task_id=task_id,
                command_id="cmd-message",
                kind=TaskCommandKind.MESSAGE.value,
                payload={},
                status=COMMAND_PENDING,
            )
        )
        db.commit()
        assert bool(db.query(_pending_resume_command_exists(task_id)).scalar()) is False
    finally:
        db.close()


# ===== authorization matrix =====


def test_other_users_agent_key_is_404_and_untouched():
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _force_waiting(task_id)
    _insert_question_message(task_id)
    other_headers = _register_second_user()
    _other_agent_id, other_key = _create_agent_with_key(other_headers)

    response = _cancel(task_id, other_key)
    assert response.status_code == 404, response.text
    assert response.json()["error"]["code"] == "task_not_found"
    task = _task_row(task_id)
    assert task.status == TaskStatus.WAITING_FOR_USER
    assert task.agent_config is None


def test_non_sdk_task_is_404_and_untouched():
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _force_waiting(task_id)
    _insert_question_message(task_id)
    db = _direct_db_session()
    try:
        db.query(Task).filter(Task.id == task_id).update(
            {Task.source: "web"}, synchronize_session=False
        )
        db.commit()
    finally:
        db.close()

    response = _cancel(task_id, full_key)
    assert response.status_code == 404, response.text
    assert response.json()["error"]["code"] == "task_not_found"
    task = _task_row(task_id)
    assert task.status == TaskStatus.WAITING_FOR_USER
    assert task.source == "web"


def test_workforce_key_cancels_its_own_run_question():
    headers = _admin_headers()
    workforce_id = _create_active_workforce(headers, name="Cancel WF")
    full_key = _create_workforce_key(headers, workforce_id)
    run = client.post(
        f"/v1/workforces/{workforce_id}/runs",
        headers=_bearer(full_key),
        json={"message": {"role": "user", "content": "go"}},
    )
    assert run.status_code == 202, run.text
    task_id = int(run.json()["task_id"])
    run_id = _task_row(task_id).run_id
    assert run_id
    _force_waiting(task_id, run_id=run_id)
    _insert_question_message(task_id)

    response = _cancel(task_id, full_key)
    assert response.status_code == 202, response.text
    assert response.json()["run_id"] == run_id
    assert _task_row(task_id).status == TaskStatus.COMPLETED


# ===== question selection =====


def test_native_unreadable_question_does_not_fall_back_to_a_superseded_row():
    """T3 means the native row owns the slot but cannot be read. The cancel
    must leave the empty answer slot empty, not materialize a previous
    round's already-superseded transcript question."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _force_waiting(task_id, run_id="run-native")
    _insert_question_message(
        task_id,
        content="previous round question",
        message_type="question_superseded",
        turn_id="old-question",
    )
    _seed_active_interaction_row(task_id, run_id="run-native")
    db = _direct_db_session()
    try:
        row = (
            db.query(TaskInteractionRequest)
            .filter(TaskInteractionRequest.task_id == task_id)
            .one()
        )
        row.request_payload = {"unexpected": True}
        db.commit()
    finally:
        db.close()

    response = _cancel(task_id, full_key)
    assert response.status_code == 202, response.text
    assert "previous round question" not in _message_contents(
        _steps(task_id, full_key), "assistant"
    )
    db = _direct_db_session()
    try:
        traces = (
            db.query(TraceEvent)
            .filter(
                TraceEvent.task_id == task_id,
                TraceEvent.event_type == "agent_message",
            )
            .all()
        )
        assert (
            sum(
                1
                for trace in traces
                if isinstance(trace.data, dict)
                and trace.data.get("message") == "previous round question"
            )
            == 0
        )
    finally:
        db.close()


def test_waiting_task_steps_show_the_live_question_exactly_once():
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _force_waiting(task_id)
    _insert_question_message(task_id, content="Live question?")
    db = _direct_db_session()
    try:
        db.add(
            TraceEvent(
                task_id=task_id,
                event_id=f"live-question-{task_id}",
                event_type="agent_message",
                timestamp=datetime.now(timezone.utc),
                data={
                    "message": "Live question?",
                    "message_type": "question",
                    "expect_response": True,
                    "visible": True,
                },
            )
        )
        db.commit()
    finally:
        db.close()

    assert (
        _message_contents(_steps(task_id, full_key), "assistant").count(
            "Live question?"
        )
        == 1
    )
    assert _cancel(task_id, full_key).status_code == 202
    assert (
        _message_contents(_steps(task_id, full_key), "assistant").count(
            "Live question?"
        )
        == 1
    )


def test_repeated_identical_question_in_a_later_turn_is_materialized_again():
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _force_waiting(task_id, run_id="run-1")
    _insert_question_message(task_id, content="Continue?", turn_id="q1")
    assert _cancel(task_id, full_key).status_code == 202

    append = _append_with_key(task_id, agent_id, full_key, content="next turn")
    assert append.status_code == 202, append.text
    run_id = _task_row(task_id).run_id
    assert run_id
    _force_waiting(task_id, run_id=run_id)
    _insert_question_message(task_id, content="Continue?", turn_id="q2")

    db = _direct_db_session()
    try:
        db.add(
            TraceEvent(
                task_id=task_id,
                event_id=f"user-next-repeat-{task_id}",
                event_type="user_message",
                timestamp=datetime.now(timezone.utc),
                data={"message": "next turn"},
            )
        )
        db.commit()
    finally:
        db.close()

    assert _cancel(task_id, full_key).status_code == 202

    db = _direct_db_session()
    try:
        traces = (
            db.query(TraceEvent)
            .filter(
                TraceEvent.task_id == task_id,
                TraceEvent.event_type == "agent_message",
            )
            .all()
        )
        assert (
            sum(
                1
                for trace in traces
                if isinstance(trace.data, dict)
                and trace.data.get("message") == "Continue?"
            )
            == 2
        )
    finally:
        db.close()
    assert (
        _message_contents(_steps(task_id, full_key), "assistant").count("Continue?")
        == 2
    )


def test_cancel_skips_materializing_a_question_when_traces_are_expired():
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _force_waiting(task_id)
    _insert_question_message(task_id, content="Expired question?")
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        task.traces_expired_at = datetime.now(timezone.utc)
        db.commit()
    finally:
        db.close()

    response = _cancel(task_id, full_key)
    assert response.status_code == 202, response.text

    db = _direct_db_session()
    try:
        traces = (
            db.query(TraceEvent)
            .filter(
                TraceEvent.task_id == task_id,
                TraceEvent.event_type == "agent_message",
            )
            .all()
        )
        assert (
            sum(
                1
                for trace in traces
                if isinstance(trace.data, dict)
                and trace.data.get("message") == "Expired question?"
            )
            == 0
        )
    finally:
        db.close()
    steps_response = client.get(
        f"/v1/chat/tasks/{task_id}/steps", headers=_bearer(full_key)
    )
    assert steps_response.status_code == 200, steps_response.text
    assert steps_response.json()["steps_expired"] is True
    assert _message_contents(steps_response.json()["steps"], "assistant") == []


# ===== v2 settlement =====


def test_v2_cancel_settles_completed_without_a_cancelled_notice():
    from xagent.web.services.task_event_context_service import load_task_event_context

    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        task.conversation_storage_version = 2
        db.commit()
    finally:
        db.close()
    _force_waiting(task_id)
    _insert_question_message(task_id)

    response = _cancel(task_id, full_key)
    assert response.status_code == 202, response.text

    db = _direct_db_session()
    try:
        settled = (
            db.query(TaskExecutionEvent)
            .filter(
                TaskExecutionEvent.task_id == task_id,
                TaskExecutionEvent.kind == "execution_settled",
            )
            .one()
        )
        result = settled.payload.get("data", settled.payload)["result"]
        assert result.get("status") != "cancelled"
        context = load_task_event_context(db, task_id)
        assert all(
            "Previous execution was cancelled." not in str(message.get("content", ""))
            for message in context.messages
        )
    finally:
        db.close()
