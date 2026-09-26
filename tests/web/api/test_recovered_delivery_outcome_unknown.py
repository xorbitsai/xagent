"""A recovered delivery whose run changed is settled as outcome unknown.

An earlier attempt of a durable MESSAGE command accepted the turn as a new
run (``begin_turn`` mints one) and crashed before settling it. The retry must
not run the turn again, must not report it as not accepted, and must leave the
task where recovery put it: the command settles as accepted with an unknown
outcome and the delivery row advances to ``dispatched`` ("do not resend").
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import event
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from tests.web.api.test_durable_message_resume_contention import (
    _live_control_environment,
    _live_task,
    _message_command,
    _user,
    live_task_lease,
)
from xagent.web.api import websocket as websocket_api
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.services import task_command_execution as command_execution_service
from xagent.web.services import task_execution as task_execution_service
from xagent.web.services import task_orchestrator
from xagent.web.services.chat_history_service import (
    DELIVERY_COMPLETED,
    DELIVERY_DISPATCHED,
    DELIVERY_FAILED,
    DELIVERY_OUTCOME_UNKNOWN,
    DELIVERY_PENDING,
)
from xagent.web.services.client_error_messages import ClientErrorCode
from xagent.web.services.task_command_execution import execute_durable_task_command
from xagent.web.services.task_command_transport import (
    COMMAND_COMPLETED,
    ClaimedTaskCommand,
    TaskCommandDeferred,
    TaskCommandKind,
    TaskCommandRejected,
    enqueue_task_command,
    get_runner_id,
)
from xagent.web.services.task_events import discard_command_reply
from xagent.web.services.task_execution import ResumeReservationOutcome
from xagent.web.services.task_orchestrator import (
    TaskTurnOrchestrator,
    TaskTurnPayload,
    TurnKind,
)

# Re-exported fixture from the contention suite.
live_task_lease = live_task_lease

TURN_ID = "same-turn"
MESSAGE = "apply this form response once"


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'recovered_outcome_unknown.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


class _RecordingReply:
    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []

    async def __call__(self, message: dict[str, Any]) -> None:
        self.frames.append(message)


def _settled_task(
    db,
    owner_id: int,
    *,
    status: TaskStatus,
    run_id: str = "newly-started-run",
) -> Task:
    task = _live_task(db, owner_id)
    task.status = status
    task.run_id = run_id
    task.runner_id = None
    task.lease_expires_at = None
    task.control_state = status.value.lower()
    db.commit()
    db.refresh(task)
    return task


def _pending_row(db, task: Task, owner_id: int, *, status: str = DELIVERY_PENDING):
    db.add(
        TaskChatMessage(
            task_id=int(task.id),
            user_id=owner_id,
            role="user",
            message_type="user_message",
            content=MESSAGE,
            turn_id=TURN_ID,
            delivery_status=status,
        )
    )
    db.commit()


def _user_rows(db, task_id: int) -> list[TaskChatMessage]:
    db.expire_all()
    return (
        db.query(TaskChatMessage)
        .filter(
            TaskChatMessage.task_id == task_id,
            TaskChatMessage.role == "user",
        )
        .all()
    )


def _row_status(db, task_id: int) -> str:
    rows = [row for row in _user_rows(db, task_id) if row.turn_id == TURN_ID]
    assert len(rows) == 1
    return str(rows[0].delivery_status)


def _recovered_command(task: Task, owner) -> ClaimedTaskCommand:
    # The command targeted "live-run"; the task now runs a different run.
    return _message_command(task, owner, TURN_ID, attempt_count=2)


def _outcome_unknown_result(task: Task) -> dict[str, Any]:
    return {
        "task_id": int(task.id),
        "command_id": TURN_ID,
        "kind": TaskCommandKind.MESSAGE.value,
        "delivery_outcome": DELIVERY_OUTCOME_UNKNOWN,
    }


def _assert_outcome_unknown_frames(reply: _RecordingReply) -> None:
    rejected = [f for f in reply.frames if f.get("type") == "message_rejected"]
    assert len(rejected) == 1
    assert rejected[0]["rejection_outcome"] == "outcome_unknown"
    assert rejected[0]["error_code"] == ClientErrorCode.MESSAGE_OUTCOME_UNKNOWN.value
    errors = [f for f in reply.frames if f.get("type") == "error"]
    assert len(errors) == 1
    assert errors[0]["error_code"] == ClientErrorCode.MESSAGE_OUTCOME_UNKNOWN.value
    assert errors[0]["turn_id"] == TURN_ID


@pytest.fixture()
def recording_reply() -> Iterator[_RecordingReply]:
    reply = _RecordingReply()
    with patch.object(command_execution_service, "command_reply", return_value=reply):
        yield reply


@pytest.fixture()
def begin_turn_spy() -> Iterator[AsyncMock]:
    spy = AsyncMock(wraps=TaskTurnOrchestrator.begin_turn)
    with patch.object(TaskTurnOrchestrator, "begin_turn", spy):
        yield spy


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [TaskStatus.PAUSED, TaskStatus.COMPLETED, TaskStatus.FAILED],
)
async def test_recovered_delivery_on_changed_run_settles_outcome_unknown(
    db_session,
    recording_reply: _RecordingReply,
    begin_turn_spy: AsyncMock,
    status: TaskStatus,
) -> None:
    owner = _user(db_session, f"changed-run-{status.value}")
    task = _settled_task(db_session, int(owner.id), status=status)
    task_id = int(task.id)
    _pending_row(db_session, task, int(owner.id))

    publish = AsyncMock()
    with (
        _live_control_environment(outcome=ResumeReservationOutcome.RESERVED) as (
            agent,
            background_manager,
        ),
        patch.object(command_execution_service, "publish_task_event", publish),
    ):
        result = await execute_durable_task_command(_recovered_command(task, owner))

    assert result == _outcome_unknown_result(task)
    begin_turn_spy.assert_not_awaited()
    agent.post_user_message.assert_not_awaited()
    background_manager.try_reserve_resume.assert_not_called()
    assert _row_status(db_session, task_id) == DELIVERY_DISPATCHED
    assert len(_user_rows(db_session, task_id)) == 1
    stored = db_session.get(Task, task_id)
    assert stored.status == status
    assert stored.run_id == "newly-started-run"
    _assert_outcome_unknown_frames(recording_reply)
    # A reachable origin gets the answer personally; no task-wide notice.
    publish.assert_not_awaited()


@pytest.mark.asyncio
async def test_running_task_with_dead_lease_and_changed_run_is_not_redriven(
    db_session,
    recording_reply: _RecordingReply,
) -> None:
    owner = _user(db_session, "dead-lease-owner")
    task = _live_task(db_session, int(owner.id))
    task.run_id = "newly-started-run"
    task.runner_id = "dead-runner"
    task.lease_expires_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    db_session.commit()
    task_id = int(task.id)
    _pending_row(db_session, task, int(owner.id))

    with _live_control_environment(outcome=ResumeReservationOutcome.RESERVED) as (
        agent,
        background_manager,
    ):
        result = await execute_durable_task_command(_recovered_command(task, owner))
        task_execution_service.execute_resume_background.assert_not_called()

    assert result == _outcome_unknown_result(task)
    agent.post_user_message.assert_not_awaited()
    background_manager.try_reserve_resume.assert_not_called()
    assert _row_status(db_session, task_id) == DELIVERY_DISPATCHED
    stored = db_session.get(Task, task_id)
    assert stored.status == TaskStatus.RUNNING
    assert stored.run_id == "newly-started-run"
    _assert_outcome_unknown_frames(recording_reply)


@pytest.mark.asyncio
@pytest.mark.parametrize("current_run_id", ["live-run", "rotated-run"])
async def test_recovered_delivery_with_a_reconciling_runtime_still_injects_live(
    live_task_lease,
    db_session,
    current_run_id: str,
) -> None:
    """The command's own run, or a run live in this process, reconciles the
    turn id against its checkpoint, so the recovered claim is redriven live."""

    owner = _user(db_session, f"reconciling-owner-{current_run_id}")
    task = _live_task(db_session, int(owner.id))
    task.run_id = current_run_id
    task.runner_id = get_runner_id()
    task.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=1)
    db_session.commit()
    live_task_lease(db_session, task)
    task_id = int(task.id)
    _pending_row(db_session, task, int(owner.id))

    background_manager = task_execution_service.BackgroundTaskManager()
    with _live_control_environment(background_manager=background_manager) as (
        agent,
        _,
    ):
        result = await execute_durable_task_command(_recovered_command(task, owner))

    assert result is not None and "delivery_outcome" not in result
    agent.post_user_message.assert_awaited_once()
    assert _row_status(db_session, task_id) == DELIVERY_DISPATCHED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raced_status", "expected"),
    [
        (DELIVERY_COMPLETED, "completed"),
        (DELIVERY_FAILED, "rejected"),
        (DELIVERY_OUTCOME_UNKNOWN, "outcome_unknown"),
    ],
)
async def test_another_writer_settling_first_is_answered_from_the_row(
    db_session,
    recording_reply: _RecordingReply,
    raced_status: str,
    expected: str,
) -> None:
    owner = _user(db_session, f"raced-{raced_status}")
    task = _settled_task(db_session, int(owner.id), status=TaskStatus.PAUSED)
    task_id = int(task.id)
    _pending_row(db_session, task, int(owner.id))
    real_mark = command_execution_service.mark_user_message_delivery_sync

    def mark_after_race(task_id_arg: int, turn_id: str, status: str):
        if status == DELIVERY_DISPATCHED:
            real_mark(task_id_arg, turn_id, raced_status)
        return real_mark(task_id_arg, turn_id, status)

    with (
        _live_control_environment(outcome=ResumeReservationOutcome.RESERVED),
        patch.object(
            command_execution_service,
            "mark_user_message_delivery_sync",
            side_effect=mark_after_race,
        ),
    ):
        if expected == "rejected":
            with pytest.raises(TaskCommandRejected):
                await execute_durable_task_command(_recovered_command(task, owner))
            result = None
        else:
            result = await execute_durable_task_command(_recovered_command(task, owner))

    assert _row_status(db_session, task_id) == raced_status
    if expected == "completed":
        assert result == {
            "task_id": task_id,
            "command_id": TURN_ID,
            "kind": TaskCommandKind.MESSAGE.value,
        }
    elif expected == "outcome_unknown":
        assert result == _outcome_unknown_result(task)
        _assert_outcome_unknown_frames(recording_reply)


@pytest.mark.asyncio
async def test_settlement_write_failure_never_persists_failed(
    db_session,
    recording_reply: _RecordingReply,
) -> None:
    owner = _user(db_session, "mark-fails-owner")
    task = _settled_task(db_session, int(owner.id), status=TaskStatus.PAUSED)
    task_id = int(task.id)
    _pending_row(db_session, task, int(owner.id))
    real_mark = command_execution_service.mark_user_message_delivery_sync
    targets: list[str] = []

    def failing_dispatch(task_id_arg: int, turn_id: str, status: str):
        targets.append(status)
        if status == DELIVERY_DISPATCHED:
            raise OperationalError("UPDATE", {}, Exception("database went away"))
        return real_mark(task_id_arg, turn_id, status)

    with (
        _live_control_environment(outcome=ResumeReservationOutcome.RESERVED),
        patch.object(
            command_execution_service,
            "mark_user_message_delivery_sync",
            side_effect=failing_dispatch,
        ),
        pytest.raises(OperationalError),
    ):
        await execute_durable_task_command(_recovered_command(task, owner))

    assert DELIVERY_FAILED not in targets
    assert _row_status(db_session, task_id) == DELIVERY_OUTCOME_UNKNOWN


@pytest.mark.asyncio
async def test_lost_commit_ack_then_retry_settles_outcome_unknown(
    db_session,
    recording_reply: _RecordingReply,
) -> None:
    owner = _user(db_session, "lost-ack-owner")
    task = _settled_task(
        db_session,
        int(owner.id),
        status=TaskStatus.PAUSED,
        run_id="live-run",
    )
    task_id = int(task.id)

    def lose_ack_after_commit(session: Session) -> None:
        if session.info.pop("lose_commit_ack", False):
            raise OperationalError("COMMIT", {}, Exception("connection reset"))

    def flag_turn_insert(session: Session, _flush_context: Any) -> None:
        if any(
            isinstance(obj, TaskChatMessage) and obj.turn_id == TURN_ID
            for obj in session.new
        ):
            session.info["lose_commit_ack"] = True

    event.listen(Session, "after_flush", flag_turn_insert)
    event.listen(Session, "after_commit", lose_ack_after_commit)
    try:
        with (
            _live_control_environment(outcome=ResumeReservationOutcome.RESERVED),
            patch.object(
                task_orchestrator,
                "_reconcile_claimed_turn_after_commit_ack_failure",
                return_value=False,
            ),
            pytest.raises(TaskCommandDeferred),
        ):
            await execute_durable_task_command(
                _message_command(task, owner, TURN_ID, attempt_count=1)
            )
    finally:
        event.remove(Session, "after_flush", flag_turn_insert)
        event.remove(Session, "after_commit", lose_ack_after_commit)

    # The first attempt really committed its turn under a new run.
    db_session.expire_all()
    accepted = db_session.get(Task, task_id)
    assert accepted.status == TaskStatus.RUNNING
    accepted_run_id = accepted.run_id
    assert accepted_run_id not in {None, "live-run"}
    assert _row_status(db_session, task_id) == DELIVERY_PENDING

    recording_reply.frames.clear()
    begin_turn = AsyncMock()
    with (
        _live_control_environment(outcome=ResumeReservationOutcome.RESERVED) as (
            agent,
            _,
        ),
        patch.object(TaskTurnOrchestrator, "begin_turn", begin_turn),
    ):
        result = await execute_durable_task_command(
            _message_command(task, owner, TURN_ID, attempt_count=2)
        )

    assert result == _outcome_unknown_result(task)
    begin_turn.assert_not_awaited()
    agent.post_user_message.assert_not_awaited()
    assert _row_status(db_session, task_id) == DELIVERY_DISPATCHED
    assert len(_user_rows(db_session, task_id)) == 1
    stored = db_session.get(Task, task_id)
    assert stored.status == TaskStatus.RUNNING
    assert stored.run_id == accepted_run_id
    _assert_outcome_unknown_frames(recording_reply)


def test_begin_turn_claim_with_existing_turn_row_rolls_back(db_session) -> None:
    owner = _user(db_session, "claim-rollback-owner")
    task = _settled_task(
        db_session,
        int(owner.id),
        status=TaskStatus.PAUSED,
        run_id="paused-run",
    )
    task_id = int(task.id)
    state_version = int(task.state_version or 0)
    _pending_row(db_session, task, int(owner.id))

    with pytest.raises(task_orchestrator.TaskTurnAlreadyAccepted) as exc_info:
        task_orchestrator._begin_turn_atomic_sync(
            task_id,
            int(owner.id),
            payload=TaskTurnPayload(transcript_message=MESSAGE, turn_id=TURN_ID),
            kind=TurnKind.APPEND,
        )

    assert exc_info.value.reason == "turn_already_accepted"
    db_session.expire_all()
    stored = db_session.get(Task, task_id)
    assert stored.status == TaskStatus.PAUSED
    assert stored.run_id == "paused-run"
    assert int(stored.state_version or 0) == state_version
    assert stored.runner_id is None
    assert len(_user_rows(db_session, task_id)) == 1
    assert _row_status(db_session, task_id) == DELIVERY_PENDING


@pytest.mark.asyncio
async def test_begin_turn_collision_without_target_run_settles_outcome_unknown(
    db_session,
    recording_reply: _RecordingReply,
) -> None:
    """A command with no target run cannot prove a run change up front."""

    owner = _user(db_session, "no-target-owner")
    task = _settled_task(db_session, int(owner.id), status=TaskStatus.PAUSED)
    task_id = int(task.id)
    _pending_row(db_session, task, int(owner.id))
    command = _message_command(task, owner, TURN_ID, attempt_count=2)
    command = replace(command, target_run_id=None)

    with _live_control_environment(outcome=ResumeReservationOutcome.RESERVED):
        result = await execute_durable_task_command(command)

    assert result == _outcome_unknown_result(task)
    assert _row_status(db_session, task_id) == DELIVERY_DISPATCHED
    stored = db_session.get(Task, task_id)
    assert stored.status == TaskStatus.PAUSED
    assert stored.run_id == "newly-started-run"
    _assert_outcome_unknown_frames(recording_reply)


@pytest.mark.asyncio
async def test_unexpected_error_on_recovered_claim_records_outcome_unknown(
    db_session,
    recording_reply: _RecordingReply,
) -> None:
    owner = _user(db_session, "unexpected-owner")
    task = _live_task(db_session, int(owner.id))
    task_id = int(task.id)
    _pending_row(db_session, task, int(owner.id))

    class Boom(Exception):
        pass

    with (
        _live_control_environment(outcome=ResumeReservationOutcome.RESERVED),
        patch(
            "xagent.web.services.agent_service_manager.get_agent_manager",
            side_effect=Boom("agent manager exploded"),
        ),
        pytest.raises(Boom),
    ):
        await execute_durable_task_command(_recovered_command(task, owner))

    assert _row_status(db_session, task_id) == DELIVERY_OUTCOME_UNKNOWN


@pytest.mark.asyncio
async def test_outcome_unknown_notice_is_published_when_the_reply_is_discarded(
    db_session,
) -> None:
    owner = _user(db_session, "discard-owner")
    task = _settled_task(db_session, int(owner.id), status=TaskStatus.PAUSED)
    task_id = int(task.id)
    _pending_row(db_session, task, int(owner.id))

    publish = AsyncMock()
    with (
        _live_control_environment(outcome=ResumeReservationOutcome.RESERVED),
        patch.object(
            command_execution_service,
            "command_reply",
            return_value=discard_command_reply,
        ),
        patch.object(command_execution_service, "publish_task_event", publish),
    ):
        result = await execute_durable_task_command(_recovered_command(task, owner))

    assert result == _outcome_unknown_result(task)
    published = [
        call.args[0]
        for call in publish.await_args_list
        if call.args[0].get("error_code")
        == ClientErrorCode.MESSAGE_OUTCOME_UNKNOWN.value
    ]
    assert len(published) == 1
    notice = published[0]
    assert notice["type"] == "error"
    assert notice["task_id"] == task_id
    assert notice["client_message_id"] == TURN_ID
    assert notice["turn_id"] == TURN_ID
    assert notice["message"]
    assert isinstance(notice["timestamp"], float)
    assert publish.await_args_list[-1].args[1] == task_id


def _enqueue_settled_unknown_command(
    db,
    task: Task,
    owner,
    payload: dict[str, Any],
    *,
    result: dict[str, Any] | None,
) -> None:
    enqueued = enqueue_task_command(
        db,
        task_id=int(task.id),
        actor_user_id=int(owner.id),
        command_id=TURN_ID,
        kind=TaskCommandKind.MESSAGE,
        payload=payload,
    )
    stored = db.get(TaskExecutionCommand, enqueued.command_id)
    stored.status = COMMAND_COMPLETED
    stored.result = result
    db.commit()


@pytest.mark.parametrize("with_files", [False, True])
@pytest.mark.parametrize(
    ("command_result", "resend_message", "expected_status", "expected_matches"),
    [
        (
            {"delivery_outcome": DELIVERY_OUTCOME_UNKNOWN},
            MESSAGE,
            DELIVERY_OUTCOME_UNKNOWN,
            True,
        ),
        (
            {"delivery_outcome": DELIVERY_OUTCOME_UNKNOWN},
            "a different message",
            DELIVERY_OUTCOME_UNKNOWN,
            False,
        ),
        ({}, MESSAGE, COMMAND_COMPLETED, True),
    ],
)
def test_same_id_resend_after_unknown_settlement_is_answered_outcome_unknown(
    db_session,
    with_files: bool,
    command_result: dict[str, Any],
    resend_message: str,
    expected_status: str,
    expected_matches: bool,
) -> None:
    owner = _user(
        db_session, f"resend-{with_files}-{expected_status}-{expected_matches}"
    )
    task = _settled_task(db_session, int(owner.id), status=TaskStatus.PAUSED)
    files = (
        [{"file_id": "file-1", "name": "a.txt", "size": 1, "type": "text/plain"}]
        if with_files
        else []
    )
    display = command_execution_service._display_message_for_user(MESSAGE, bool(files))
    db_session.add(
        TaskChatMessage(
            task_id=int(task.id),
            user_id=int(owner.id),
            role="user",
            message_type="user_message",
            content=display,
            turn_id=TURN_ID,
            delivery_status=DELIVERY_DISPATCHED,
            attachments=files or None,
        )
    )
    db_session.commit()

    def payload(message: str) -> dict[str, Any]:
        return {
            "type": "chat_message",
            "message": message,
            "client_message_id": TURN_ID,
            "files": files,
        }

    _enqueue_settled_unknown_command(
        db_session,
        task,
        owner,
        payload(MESSAGE),
        result={
            "task_id": int(task.id),
            "command_id": TURN_ID,
            "kind": TaskCommandKind.MESSAGE.value,
            **command_result,
        },
    )

    enqueued = websocket_api._enqueue_websocket_task_command_sync(
        task_id=int(task.id),
        actor_user_id=int(owner.id),
        actor_is_admin=False,
        command_id=TURN_ID,
        kind=TaskCommandKind.MESSAGE,
        payload=payload(resend_message),
        allow_missing_task=True,
    )

    assert enqueued is not None
    assert enqueued.created is False
    if expected_status == DELIVERY_OUTCOME_UNKNOWN:
        assert enqueued.status == DELIVERY_OUTCOME_UNKNOWN
        assert enqueued.payload_matches is expected_matches
    else:
        # An ordinary dispatched row keeps its accepted answer.
        assert enqueued.status in {DELIVERY_COMPLETED, COMMAND_COMPLETED}
        assert enqueued.payload_matches is True
