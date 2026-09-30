"""A recovered delivery never resumes a run that has ended.

A durable MESSAGE command claimed its delivery row on the task's own run and
crashed before settling it. The run then ended: lease recovery found it not
recoverable and settled the task FAILED, or its runner finished it COMPLETED.
The retry finds the pending row as a recovered claim on the command's own
run, which is normally redriven through a resume so the run replays the turn
id against its checkpoint. An ended run must not be resumed: the retry
settles the command as accepted with an unknown outcome, advances the row out
of ``pending`` without running the turn, and leaves the task in its terminal
status with its control state, run, diagnostic and result untouched.

The routing snapshot can be stale, so the refusal is also enforced by the
RESUME_REQUESTED transition and by the resume lease claim, each as part of
its own conditional UPDATE.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.exc import OperationalError

from tests.web.api.test_durable_message_resume_contention import (
    _live_control_environment,
    _live_task,
    _message_command,
    _user,
    live_task_lease,
)
from tests.web.api.test_recovered_delivery_outcome_unknown import (
    MESSAGE,
    TURN_ID,
    _assert_outcome_unknown_frames,
    _enqueue_settled_unknown_command,
    _outcome_unknown_result,
    _pending_row,
    _RecordingReply,
    _row_status,
    _settled_task,
    _user_rows,
)
from xagent.web.api import websocket as websocket_api
from xagent.web.models.database import (
    Base,
    get_db,
    get_engine,
    get_session_local,
    init_db,
)
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.services import task_command_execution as command_execution_service
from xagent.web.services import task_execution as task_execution_service
from xagent.web.services import task_execution_controller as controller_module
from xagent.web.services.chat_history_service import (
    DELIVERY_DISPATCHED,
    DELIVERY_OUTCOME_UNKNOWN,
    DELIVERY_PENDING,
)
from xagent.web.services.client_error_messages import ClientErrorCode
from xagent.web.services.task_command_execution import execute_durable_task_command
from xagent.web.services.task_command_transport import (
    COMMAND_COMPLETED,
    COMMAND_PENDING,
    COMMAND_PROCESSING,
    TaskCommandDeferred,
    TaskCommandKind,
    dispatch_one_task_command,
    enqueue_task_command,
    get_runner_id,
)
from xagent.web.services.task_execution import ResumeReservationOutcome
from xagent.web.services.task_execution_controller import (
    StaleTaskRunError,
    TaskControlState,
    TaskStatusRefusedError,
    apply_task_control_transition,
)
from xagent.web.services.task_lease_service import (
    TaskLease,
    get_expired_task_lease_candidates,
    recover_expired_task_lease_no_commit,
    release_task_lease_no_commit,
)
from xagent.web.services.task_orchestrator import TaskTurnOrchestrator, TurnKind

# Re-exported fixture from the contention suite.
live_task_lease = live_task_lease

RECOVERY_ERROR = "not recoverable: checkpoint missing"
FINAL_OUTPUT = "the run's final answer"
ENDED_STATUSES = pytest.mark.parametrize(
    "ended_status", [TaskStatus.FAILED, TaskStatus.COMPLETED]
)


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'recovered_failed_run.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


@pytest.fixture()
def recording_reply() -> Iterator[_RecordingReply]:
    reply = _RecordingReply()
    with patch.object(command_execution_service, "command_reply", return_value=reply):
        yield reply


def _expired_running_task(db, owner_id: int) -> Task:
    """A RUNNING task on the command's own run whose lease has expired."""

    task = _live_task(db, owner_id)
    task.runner_id = "dead-runner"
    task.lease_attempt_id = "dead-attempt"
    task.lease_expires_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    task.control_state = TaskControlState.RUNNING.value
    db.commit()
    db.refresh(task)
    return task


def _recover_to_failed(task_id: int) -> None:
    """Run lease recovery's NOT_RECOVERABLE settlement in its own session."""

    with get_session_local()() as db:
        now = datetime.now(timezone.utc)
        candidate = next(
            candidate
            for candidate in get_expired_task_lease_candidates(db, cutoff=now, limit=10)
            if candidate.task_id == task_id
        )
        assert recover_expired_task_lease_no_commit(
            db,
            candidate,
            status=TaskStatus.FAILED,
            recovered_at=now,
            error_message=RECOVERY_ERROR,
        )
        db.commit()


def _complete_run(task_id: int) -> None:
    """The run's own runner finishes it COMPLETED, in its own session."""

    with get_session_local()() as db:
        db.query(Task).filter(Task.id == task_id).update(
            {Task.output: FINAL_OUTPUT}, synchronize_session=False
        )
        assert release_task_lease_no_commit(
            db,
            TaskLease(
                task_id=task_id,
                runner_id="dead-runner",
                run_id="live-run",
                attempt_id="dead-attempt",
            ),
            status=TaskStatus.COMPLETED,
        )
        db.commit()


def _end_run(task_id: int, ended_status: TaskStatus) -> None:
    if ended_status == TaskStatus.FAILED:
        _recover_to_failed(task_id)
    else:
        assert ended_status == TaskStatus.COMPLETED
        _complete_run(task_id)


def _ended_task(db, owner_id: int, ended_status: TaskStatus) -> Task:
    task = _expired_running_task(db, owner_id)
    _end_run(int(task.id), ended_status)
    db.expire_all()
    task = db.get(Task, int(task.id))
    assert task.status == ended_status
    return task


def _assert_still_ended(
    db, task_id: int, ended_status: TaskStatus, *, state_version: int
) -> None:
    db.expire_all()
    stored = db.get(Task, task_id)
    assert stored.status == ended_status
    assert stored.control_state == ended_status.value.lower()
    if ended_status == TaskStatus.FAILED:
        assert stored.error_message == RECOVERY_ERROR
    else:
        assert stored.output == FINAL_OUTPUT
        assert stored.error_message is None
    assert stored.run_id == "live-run"
    assert stored.runner_id is None
    assert int(stored.state_version or 0) == state_version


@pytest.mark.asyncio
@ENDED_STATUSES
async def test_recovered_delivery_on_ended_run_settles_outcome_unknown(
    db_session,
    recording_reply: _RecordingReply,
    ended_status: TaskStatus,
) -> None:
    owner = _user(db_session, f"ended-run-owner-{ended_status.value}")
    task = _ended_task(db_session, int(owner.id), ended_status)
    task_id = int(task.id)
    state_version = int(task.state_version or 0)
    _pending_row(db_session, task, int(owner.id))

    publish = AsyncMock()
    with (
        _live_control_environment(outcome=ResumeReservationOutcome.RESERVED) as (
            agent,
            background_manager,
        ),
        patch.object(task_execution_service, "publish_task_event", publish),
    ):
        result = await execute_durable_task_command(
            _message_command(task, owner, TURN_ID, attempt_count=2)
        )
        task_execution_service.execute_resume_background.assert_not_called()

    assert result == _outcome_unknown_result(task)
    agent.post_user_message.assert_not_awaited()
    background_manager.try_reserve_resume.assert_not_called()
    assert _row_status(db_session, task_id) == DELIVERY_DISPATCHED
    assert len(_user_rows(db_session, task_id)) == 1
    _assert_still_ended(db_session, task_id, ended_status, state_version=state_version)
    _assert_outcome_unknown_frames(recording_reply)
    publish.assert_not_awaited()


@pytest.mark.asyncio
@ENDED_STATUSES
@pytest.mark.parametrize("fence", ["pre_check", "conditional_update"])
async def test_run_ended_after_routing_snapshot_is_caught_by_the_transition(
    db_session,
    recording_reply: _RecordingReply,
    fence: str,
    ended_status: TaskStatus,
) -> None:
    """The run ends after the handler routed on a RUNNING row.

    ``pre_check`` lands it before the transition loads the row;
    ``conditional_update`` lands it after the load, so only the UPDATE's own
    status predicate can refuse it.
    """

    owner = _user(db_session, f"transition-race-{fence}-{ended_status.value}")
    task = _expired_running_task(db_session, int(owner.id))
    task_id = int(task.id)
    _pending_row(db_session, task, int(owner.id))

    if fence == "pre_check":
        real_sync = controller_module.transition_task_control_state_sync

        def recover_then_transition(*args: Any, **kwargs: Any):
            _end_run(task_id, ended_status)
            return real_sync(*args, **kwargs)

        race = patch.object(
            controller_module,
            "transition_task_control_state_sync",
            side_effect=recover_then_transition,
        )
    else:
        real_apply = controller_module.apply_task_control_transition

        def recover_then_apply(task_row: Task, *args: Any, **kwargs: Any):
            assert task_row.status == TaskStatus.RUNNING
            _end_run(task_id, ended_status)
            return real_apply(task_row, *args, **kwargs)

        race = patch.object(
            controller_module,
            "apply_task_control_transition",
            side_effect=recover_then_apply,
        )

    with (
        _live_control_environment(outcome=ResumeReservationOutcome.RESERVED) as (
            agent,
            background_manager,
        ),
        race as raced,
    ):
        result = await execute_durable_task_command(
            _message_command(task, owner, TURN_ID, attempt_count=2)
        )
        task_execution_service.execute_resume_background.assert_not_called()

    assert raced.call_count == 1
    assert raced.call_args.kwargs["refuse_terminal_status"] is True
    assert result == _outcome_unknown_result(task)
    background_manager.release_resume_reservation.assert_called_with(task_id)
    background_manager.register_reserved_resume.assert_not_called()
    assert _row_status(db_session, task_id) == DELIVERY_DISPATCHED
    db_session.expire_all()
    ended_version = int(db_session.get(Task, task_id).state_version or 0)
    _assert_still_ended(db_session, task_id, ended_status, state_version=ended_version)
    _assert_outcome_unknown_frames(recording_reply)
    agent.post_user_message.assert_not_awaited()


@pytest.mark.asyncio
@ENDED_STATUSES
async def test_run_ended_after_the_transition_is_caught_by_the_lease_claim(
    db_session,
    recording_reply: _RecordingReply,
    ended_status: TaskStatus,
) -> None:
    """The run ends after RESUME_REQUESTED, before the resume claims it."""

    owner = _user(db_session, f"claim-race-owner-{ended_status.value}")
    task = _expired_running_task(db_session, int(owner.id))
    task_id = int(task.id)
    _pending_row(db_session, task, int(owner.id))
    real_sync = controller_module.transition_task_control_state_sync

    def transition_then_recover(*args: Any, **kwargs: Any):
        snapshot = real_sync(*args, **kwargs)
        _end_run(task_id, ended_status)
        return snapshot

    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.post_user_message = AsyncMock()
    background_manager = task_execution_service.BackgroundTaskManager()
    resume_spy = AsyncMock(wraps=task_execution_service.execute_resume_background)
    with (
        patch(
            "xagent.web.services.agent_service_manager.get_agent_manager",
            return_value=MagicMock(get_agent_for_task=AsyncMock(return_value=agent)),
        ),
        patch.object(
            task_execution_service, "background_task_manager", background_manager
        ),
        patch.object(task_execution_service, "execute_resume_background", resume_spy),
        patch.object(task_execution_service, "publish_task_event", AsyncMock()),
        patch.object(
            controller_module,
            "transition_task_control_state_sync",
            side_effect=transition_then_recover,
        ),
    ):
        try:
            first = await execute_durable_task_command(
                _message_command(task, owner, TURN_ID, attempt_count=2)
            )
        except TaskCommandDeferred:
            # The handoff returned before the resume claimed; the row was
            # still pending when the command read it.
            first = None
        assert resume_spy.await_count == 1
        assert resume_spy.await_args.kwargs["refuse_terminal_status"] is True
        assert resume_spy.await_args.kwargs["delivery_claimed_fresh"] is False
        for _ in range(200):
            if task_id not in background_manager.running_tasks:
                break
            await asyncio.sleep(0.01)
        assert task_id not in background_manager.running_tasks

        assert _row_status(db_session, task_id) == DELIVERY_OUTCOME_UNKNOWN
        recording_reply.frames.clear()
        result = (
            first
            if first is not None
            else await execute_durable_task_command(
                _message_command(task, owner, TURN_ID, attempt_count=3)
            )
        )

    assert result == _outcome_unknown_result(task)
    agent.post_user_message.assert_not_awaited()
    assert _row_status(db_session, task_id) == DELIVERY_OUTCOME_UNKNOWN
    db_session.expire_all()
    ended_version = int(db_session.get(Task, task_id).state_version or 0)
    _assert_still_ended(db_session, task_id, ended_status, state_version=ended_version)
    if first is None:
        _assert_outcome_unknown_frames(recording_reply)


@pytest.mark.asyncio
@ENDED_STATUSES
async def test_fresh_message_on_ended_task_still_appends_a_new_run(
    db_session,
    ended_status: TaskStatus,
) -> None:
    owner = _user(db_session, f"fresh-append-owner-{ended_status.value}")
    task = _ended_task(db_session, int(owner.id), ended_status)
    task_id = int(task.id)

    begin_turn = AsyncMock(wraps=TaskTurnOrchestrator.begin_turn)
    with (
        _live_control_environment(outcome=ResumeReservationOutcome.RESERVED) as (
            agent,
            _,
        ),
        patch.object(TaskTurnOrchestrator, "begin_turn", begin_turn),
    ):
        result = await execute_durable_task_command(
            _message_command(task, owner, "fresh-turn", attempt_count=1)
        )
        task_execution_service.execute_resume_background.assert_not_called()

    assert result == {
        "task_id": task_id,
        "command_id": "fresh-turn",
        "kind": TaskCommandKind.MESSAGE.value,
    }
    begin_turn.assert_awaited_once()
    assert begin_turn.await_args.kwargs["kind"] == TurnKind.APPEND
    agent.post_user_message.assert_not_awaited()
    db_session.expire_all()
    stored = db_session.get(Task, task_id)
    assert stored.status == TaskStatus.RUNNING
    assert stored.run_id not in {None, "live-run"}


@pytest.mark.asyncio
@ENDED_STATUSES
async def test_same_id_resend_after_ended_run_refusal_reports_outcome_unknown(
    db_session,
    recording_reply: _RecordingReply,
    ended_status: TaskStatus,
) -> None:
    owner = _user(db_session, f"ended-resend-owner-{ended_status.value}")
    task = _ended_task(db_session, int(owner.id), ended_status)
    task_id = int(task.id)
    _pending_row(db_session, task, int(owner.id))
    command = _message_command(task, owner, TURN_ID, attempt_count=2)

    with _live_control_environment(outcome=ResumeReservationOutcome.RESERVED):
        result = await execute_durable_task_command(command)
    assert result == _outcome_unknown_result(task)
    _enqueue_settled_unknown_command(
        db_session, task, owner, dict(command.payload), result=result
    )

    enqueued = websocket_api._enqueue_websocket_task_command_sync(
        task_id=task_id,
        actor_user_id=int(owner.id),
        actor_is_admin=False,
        command_id=TURN_ID,
        kind=TaskCommandKind.MESSAGE,
        payload=dict(command.payload),
        allow_missing_task=True,
    )

    assert enqueued is not None
    assert enqueued.created is False
    assert enqueued.status == DELIVERY_OUTCOME_UNKNOWN
    assert enqueued.payload_matches is True
    assert result["kind"] == TaskCommandKind.MESSAGE.value
    assert ClientErrorCode.MESSAGE_OUTCOME_UNKNOWN.value in {
        frame.get("error_code") for frame in recording_reply.frames
    }


@ENDED_STATUSES
def test_refused_status_transition_leaves_the_row_untouched(
    db_session, ended_status: TaskStatus
) -> None:
    owner = _user(db_session, f"transition-unit-owner-{ended_status.value}")
    task = _ended_task(db_session, int(owner.id), ended_status)
    task_id = int(task.id)
    state_version = int(task.state_version or 0)

    with pytest.raises(TaskStatusRefusedError) as exc_info:
        apply_task_control_transition(
            task,
            TaskControlState.RESUME_REQUESTED,
            expected_run_id="live-run",
            refuse_terminal_status=True,
        )
    assert ended_status.value in str(exc_info.value)
    db_session.rollback()
    _assert_still_ended(db_session, task_id, ended_status, state_version=state_version)

    # A stale run is still reported as a stale run, not as a status refusal.
    task = db_session.get(Task, task_id)
    task.status = TaskStatus.PAUSED
    task.control_state = TaskControlState.PAUSED.value
    db_session.commit()
    with pytest.raises(StaleTaskRunError) as stale_info:
        apply_task_control_transition(
            task,
            TaskControlState.RESUME_REQUESTED,
            expected_run_id="another-run",
            refuse_terminal_status=True,
        )
    assert not isinstance(stale_info.value, TaskStatusRefusedError)


@ENDED_STATUSES
@pytest.mark.parametrize("refuse", [False, True])
def test_resume_lease_claim_status_fence_is_opt_in(
    db_session, refuse: bool, ended_status: TaskStatus
) -> None:
    owner = _user(db_session, f"claim-unit-owner-{refuse}-{ended_status.value}")
    task = _ended_task(db_session, int(owner.id), ended_status)
    task_id = int(task.id)
    state_version = int(task.state_version or 0)
    refused: list[bool] = []

    lease = task_execution_service._acquire_resume_task_lease(
        task_id,
        int(owner.id),
        "live-run",
        refuse_terminal_status=refuse,
        run_not_resumable_out=refused,
    )

    if refuse:
        assert lease is None
        assert refused == [True]
        _assert_still_ended(
            db_session, task_id, ended_status, state_version=state_version
        )
    else:
        # Every other resume caller keeps the claim it had.
        assert lease is not None
        assert refused == []
        db_session.expire_all()
        assert db_session.get(Task, task_id).status == TaskStatus.RUNNING


def test_resume_lease_claim_refused_by_a_live_owner_is_not_a_status_refusal(
    db_session,
) -> None:
    owner = _user(db_session, "claim-live-owner")
    task = _live_task(db_session, int(owner.id))
    task.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)
    db_session.commit()
    refused: list[bool] = []

    lease = task_execution_service._acquire_resume_task_lease(
        int(task.id),
        int(owner.id),
        "live-run",
        refuse_terminal_status=True,
        run_not_resumable_out=refused,
    )

    assert lease is None
    assert refused == []


@pytest.mark.asyncio
async def test_recovered_delivery_on_completed_run_is_never_resumed(
    db_session,
    recording_reply: _RecordingReply,
) -> None:
    """Regression for the probe that reproduced the COMPLETED hole.

    Before the fix the retry deferred as "waiting for runtime injection",
    handed the turn to ``execute_resume_background`` under the finished run,
    left the task ``completed / resume_requested``, and the resume claim then
    flipped COMPLETED back to RUNNING.
    """

    owner = _user(db_session, "completed-probe-owner")
    task = _settled_task(
        db_session, int(owner.id), status=TaskStatus.COMPLETED, run_id="live-run"
    )
    task.output = FINAL_OUTPUT
    db_session.commit()
    task_id = int(task.id)
    state_version = int(task.state_version or 0)
    _pending_row(db_session, task, int(owner.id))

    with _live_control_environment(outcome=ResumeReservationOutcome.RESERVED):
        result = await execute_durable_task_command(
            _message_command(task, owner, TURN_ID, attempt_count=2)
        )
        task_execution_service.execute_resume_background.assert_not_called()

    assert result == _outcome_unknown_result(task)
    assert _row_status(db_session, task_id) == DELIVERY_DISPATCHED
    _assert_outcome_unknown_frames(recording_reply)
    db_session.expire_all()
    stored = db_session.get(Task, task_id)
    assert stored.status == TaskStatus.COMPLETED
    assert stored.control_state == TaskControlState.COMPLETED.value
    assert stored.output == FINAL_OUTPUT
    assert stored.run_id == "live-run"
    assert int(stored.state_version or 0) == state_version

    # Even a resume handed this run anyway cannot claim it.
    refused: list[bool] = []
    assert (
        task_execution_service._acquire_resume_task_lease(
            task_id,
            int(owner.id),
            "live-run",
            refuse_terminal_status=True,
            run_not_resumable_out=refused,
        )
        is None
    )
    assert refused == [True]
    db_session.expire_all()
    assert db_session.get(Task, task_id).status == TaskStatus.COMPLETED


def _update_task(task_id: int, **values: Any) -> None:
    with get_session_local()() as db:
        db.query(Task).filter(Task.id == task_id).update(
            values, synchronize_session=False
        )
        db.commit()


def _end_run_directly(task_id: int, ended_status: TaskStatus) -> None:
    """End the run whatever lease holds it (the run's own terminal write)."""

    values: dict[Any, Any] = {
        Task.status: ended_status,
        Task.control_state: ended_status.value.lower(),
        Task.runner_id: None,
        Task.lease_expires_at: None,
        Task.lease_attempt_id: None,
        Task.state_version: Task.state_version + 1,
    }
    if ended_status == TaskStatus.FAILED:
        values[Task.error_message] = RECOVERY_ERROR
    else:
        values[Task.output] = FINAL_OUTPUT
    with get_session_local()() as db:
        db.query(Task).filter(Task.id == task_id).update(
            values, synchronize_session=False
        )
        db.commit()


def _real_resume_environment(agent: MagicMock, background_manager: Any):
    """Run the real resume coordinator; only the agent is a stand-in."""

    return (
        patch(
            "xagent.web.services.agent_service_manager.get_agent_manager",
            return_value=MagicMock(get_agent_for_task=AsyncMock(return_value=agent)),
        ),
        patch.object(
            task_execution_service, "background_task_manager", background_manager
        ),
    )


async def _wait_for_resume_to_finish(background_manager: Any, task_id: int) -> None:
    for _ in range(300):
        if task_id not in background_manager.running_tasks:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("the resume coordinator never finished")


@pytest.mark.asyncio
@ENDED_STATUSES
async def test_posted_recovered_claim_whose_run_ends_before_the_claim_is_announced(
    live_task_lease,
    db_session,
    ended_status: TaskStatus,
) -> None:
    """The live injection was accepted, then the run ended before resuming.

    The ended run is still not resumed. The sender was already told the turn
    was accepted and no resume will answer it, so the task's audience gets a
    task-wide outcome-unknown notice; the row stays ``dispatched``.
    """

    owner = _user(db_session, f"posted-owner-{ended_status.value}")
    task = _live_task(db_session, int(owner.id))
    task.runner_id = get_runner_id()
    task.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=1)
    db_session.commit()
    live_task_lease(db_session, task)
    task_id = int(task.id)
    _pending_row(db_session, task, int(owner.id))
    real_sync = controller_module.transition_task_control_state_sync

    def transition_then_end(*args: Any, **kwargs: Any):
        snapshot = real_sync(*args, **kwargs)
        _end_run_directly(task_id, ended_status)
        return snapshot

    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.post_user_message = AsyncMock(
        return_value=task_execution_service.UserMessageInjectionOutcome.POSTED_FRESH
    )
    background_manager = task_execution_service.BackgroundTaskManager()
    resume_spy = AsyncMock(wraps=task_execution_service.execute_resume_background)
    publish = AsyncMock()
    agent_patch, manager_patch = _real_resume_environment(agent, background_manager)
    with (
        agent_patch,
        manager_patch,
        patch.object(task_execution_service, "execute_resume_background", resume_spy),
        patch.object(task_execution_service, "publish_task_event", publish),
        patch.object(
            controller_module,
            "transition_task_control_state_sync",
            side_effect=transition_then_end,
        ),
    ):
        result = await execute_durable_task_command(
            _message_command(task, owner, TURN_ID, attempt_count=2)
        )
        assert resume_spy.await_count == 1
        assert resume_spy.await_args.kwargs["refuse_terminal_status"] is True
        assert resume_spy.await_args.kwargs["delivery_already_dispatched"] is True
        await _wait_for_resume_to_finish(background_manager, task_id)

    # The live injection was accepted and answered as such.
    assert result is not None and "delivery_outcome" not in result
    agent.post_user_message.assert_awaited_once()
    assert _row_status(db_session, task_id) == DELIVERY_DISPATCHED
    db_session.expire_all()
    stored = db_session.get(Task, task_id)
    assert stored.status == ended_status
    assert stored.control_state == ended_status.value.lower()
    assert stored.run_id == "live-run"
    assert stored.runner_id is None
    notices = [
        call.args[0]
        for call in publish.await_args_list
        if call.args[0].get("error_code")
        == ClientErrorCode.MESSAGE_OUTCOME_UNKNOWN.value
    ]
    assert len(notices) == 1
    notice = notices[0]
    assert notice["type"] == "error"
    assert notice["task_id"] == task_id
    assert notice["turn_id"] == TURN_ID
    assert notice["client_message_id"] == TURN_ID
    assert MESSAGE not in str(notice)
    assert not any(
        call.args[0].get("error_code") == ClientErrorCode.TASK_BUSY.value
        for call in publish.await_args_list
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["run_rotated", "foreign_live_owner"])
async def test_recovered_claim_refused_for_a_live_run_records_outcome_unknown(
    db_session,
    recording_reply: _RecordingReply,
    change: str,
) -> None:
    """The claim is refused although the run did not end.

    ``run_rotated``: the task moved to another run (non-terminal) between the
    transition and the claim -- the classifier's run-changed arm.
    ``foreign_live_owner``: a live foreign runner holds the same run. An
    ordinary resume records that as a failed delivery; a recovered claim may
    already have been applied, so it records outcome unknown instead.
    """

    owner = _user(db_session, f"live-refusal-{change}")
    task = _expired_running_task(db_session, int(owner.id))
    task_id = int(task.id)
    _pending_row(db_session, task, int(owner.id))
    real_sync = controller_module.transition_task_control_state_sync

    def transition_then_change(*args: Any, **kwargs: Any):
        snapshot = real_sync(*args, **kwargs)
        if change == "run_rotated":
            _update_task(
                task_id,
                status=TaskStatus.PAUSED,
                control_state=TaskControlState.PAUSED.value,
                run_id="rotated-run",
                runner_id=None,
                lease_expires_at=None,
                lease_attempt_id=None,
            )
        else:
            _update_task(
                task_id,
                runner_id="foreign-runner",
                lease_attempt_id="foreign-attempt",
                lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            )
        return snapshot

    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.post_user_message = AsyncMock()
    background_manager = task_execution_service.BackgroundTaskManager()
    publish = AsyncMock()
    agent_patch, manager_patch = _real_resume_environment(agent, background_manager)
    with (
        agent_patch,
        manager_patch,
        patch.object(task_execution_service, "publish_task_event", publish),
        patch.object(
            controller_module,
            "transition_task_control_state_sync",
            side_effect=transition_then_change,
        ),
    ):
        try:
            first = await execute_durable_task_command(
                _message_command(task, owner, TURN_ID, attempt_count=2)
            )
        except TaskCommandDeferred:
            first = None
        await _wait_for_resume_to_finish(background_manager, task_id)
        assert _row_status(db_session, task_id) == DELIVERY_OUTCOME_UNKNOWN
        recording_reply.frames.clear()
        result = (
            first
            if first is not None
            else await execute_durable_task_command(
                _message_command(task, owner, TURN_ID, attempt_count=3)
            )
        )

    assert result == _outcome_unknown_result(task)
    agent.post_user_message.assert_not_awaited()
    assert not any(
        call.args[0].get("error_code") == ClientErrorCode.TASK_BUSY.value
        for call in publish.await_args_list
    )
    db_session.expire_all()
    stored = db_session.get(Task, task_id)
    if change == "run_rotated":
        assert stored.status == TaskStatus.PAUSED
        assert stored.run_id == "rotated-run"
    else:
        assert stored.status == TaskStatus.RUNNING
        assert stored.runner_id == "foreign-runner"
        assert stored.run_id == "live-run"


def test_resume_lease_claim_classifies_a_rotated_run_as_not_resumable(
    db_session,
) -> None:
    owner = _user(db_session, "claim-rotated-owner")
    task = _settled_task(
        db_session, int(owner.id), status=TaskStatus.PAUSED, run_id="rotated-run"
    )
    not_resumable: list[bool] = []

    lease = task_execution_service._acquire_resume_task_lease(
        int(task.id),
        int(owner.id),
        "live-run",
        refuse_terminal_status=True,
        run_not_resumable_out=not_resumable,
    )

    assert lease is None
    assert not_resumable == [True]
    db_session.expire_all()
    stored = db_session.get(Task, int(task.id))
    assert stored.status == TaskStatus.PAUSED
    assert stored.run_id == "rotated-run"


@pytest.mark.asyncio
async def test_run_rotated_at_the_transition_settles_a_recovered_claim_unknown(
    db_session,
    recording_reply: _RecordingReply,
) -> None:
    """A stale run at RESUME_REQUESTED never fails a recovered claim."""

    owner = _user(db_session, "transition-rotated-owner")
    task = _expired_running_task(db_session, int(owner.id))
    task_id = int(task.id)
    _pending_row(db_session, task, int(owner.id))
    real_sync = controller_module.transition_task_control_state_sync

    def rotate_then_transition(*args: Any, **kwargs: Any):
        _update_task(task_id, run_id="rotated-run")
        return real_sync(*args, **kwargs)

    with (
        _live_control_environment(outcome=ResumeReservationOutcome.RESERVED) as (
            agent,
            background_manager,
        ),
        patch.object(
            controller_module,
            "transition_task_control_state_sync",
            side_effect=rotate_then_transition,
        ),
    ):
        result = await execute_durable_task_command(
            _message_command(task, owner, TURN_ID, attempt_count=2)
        )
        task_execution_service.execute_resume_background.assert_not_called()

    assert result == _outcome_unknown_result(task)
    background_manager.release_resume_reservation.assert_called_with(task_id)
    assert _row_status(db_session, task_id) == DELIVERY_DISPATCHED
    _assert_outcome_unknown_frames(recording_reply)
    db_session.expire_all()
    stored = db_session.get(Task, task_id)
    assert stored.run_id == "rotated-run"
    assert stored.status == TaskStatus.RUNNING
    agent.post_user_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [TaskStatus.RUNNING, TaskStatus.WAITING_FOR_USER])
async def test_fresh_live_message_is_fenced_and_still_handed_off(
    db_session,
    status: TaskStatus,
) -> None:
    """A fresh message on a live run is fenced too, and hands off as before.

    The fences only refuse an ended run; this one is not, so the message is
    handed to the resume as usual, marked as a claim this attempt made.
    """

    owner = _user(db_session, f"fresh-live-owner-{status.value}")
    task = _expired_running_task(db_session, int(owner.id))
    task.status = status
    task.control_state = status.value.lower()
    db_session.commit()
    real_sync = controller_module.transition_task_control_state_sync
    transition_spy = MagicMock(side_effect=real_sync)

    with (
        _live_control_environment(outcome=ResumeReservationOutcome.RESERVED),
        patch.object(
            controller_module, "transition_task_control_state_sync", transition_spy
        ),
    ):
        with pytest.raises(TaskCommandDeferred):
            # A deferred delivery waits for the (mocked) resume to inject it.
            await execute_durable_task_command(
                _message_command(task, owner, "fresh-live-turn", attempt_count=1)
            )
        resume = task_execution_service.execute_resume_background
        assert resume.call_count == 1
        assert resume.call_args.kwargs["refuse_terminal_status"] is True
        assert resume.call_args.kwargs["delivery_claimed_fresh"] is True

    assert transition_spy.call_count == 1
    assert transition_spy.call_args.kwargs["refuse_terminal_status"] is True
    db_session.expire_all()
    assert db_session.get(Task, int(task.id)).status == status


def _processing_command_row(db, task: Task, owner, command) -> None:
    enqueued = enqueue_task_command(
        db,
        task_id=int(task.id),
        actor_user_id=int(owner.id),
        command_id=command.command_id,
        kind=TaskCommandKind.MESSAGE,
        payload=dict(command.payload),
    )
    stored = db.get(TaskExecutionCommand, enqueued.command_id)
    stored.status = COMMAND_PROCESSING
    # The attempt that is executing it.
    stored.attempt_count = command.attempt_count
    db.commit()


def _stored_command_result(db, task_id: int, command_id: str) -> Any:
    db.expire_all()
    row = (
        db.query(TaskExecutionCommand)
        .filter(
            TaskExecutionCommand.task_id == task_id,
            TaskExecutionCommand.command_id == command_id,
        )
        .one()
    )
    return row.result


@pytest.mark.asyncio
async def test_retry_after_an_unrecorded_unknown_settlement_still_answers_unknown(
    db_session,
    recording_reply: _RecordingReply,
) -> None:
    """The attempt that settled the turn crashed before storing its result.

    The row is ``dispatched`` and the in-memory marker is gone; the durable
    record on the command row keeps the retry from reading it as accepted.
    """

    owner = _user(db_session, "unrecorded-unknown-owner")
    task = _ended_task(db_session, int(owner.id), TaskStatus.COMPLETED)
    task_id = int(task.id)
    _pending_row(db_session, task, int(owner.id))
    command = _message_command(task, owner, TURN_ID, attempt_count=2)
    _processing_command_row(db_session, task, owner, command)

    with _live_control_environment(outcome=ResumeReservationOutcome.RESERVED):
        first = await execute_durable_task_command(command)
    assert first == _outcome_unknown_result(task)
    assert _row_status(db_session, task_id) == DELIVERY_DISPATCHED
    # Recorded before the row write, so a crash after it loses nothing.
    assert _stored_command_result(db_session, task_id, TURN_ID) == first

    # The transport never stored the result; a later attempt runs again.
    recording_reply.frames.clear()
    with _live_control_environment(outcome=ResumeReservationOutcome.RESERVED):
        retried = await execute_durable_task_command(
            _message_command(task, owner, TURN_ID, attempt_count=3)
        )

    assert retried == _outcome_unknown_result(task)
    _assert_outcome_unknown_frames(recording_reply)
    assert _row_status(db_session, task_id) == DELIVERY_DISPATCHED


@pytest.mark.asyncio
async def test_retry_of_an_ordinary_dispatched_turn_is_still_accepted(
    db_session,
    recording_reply: _RecordingReply,
) -> None:
    owner = _user(db_session, "ordinary-dispatched-owner")
    task = _ended_task(db_session, int(owner.id), TaskStatus.COMPLETED)
    task_id = int(task.id)
    _pending_row(db_session, task, int(owner.id))
    command = _message_command(task, owner, TURN_ID, attempt_count=3)
    _processing_command_row(db_session, task, owner, command)
    with get_session_local()() as db:
        db.query(command_execution_service.TaskChatMessage).filter(
            command_execution_service.TaskChatMessage.task_id == task_id
        ).update({"delivery_status": DELIVERY_DISPATCHED}, synchronize_session=False)
        db.commit()

    with _live_control_environment(outcome=ResumeReservationOutcome.RESERVED):
        result = await execute_durable_task_command(command)

    assert result == {
        "task_id": task_id,
        "command_id": TURN_ID,
        "kind": TaskCommandKind.MESSAGE.value,
    }
    assert not any(
        frame.get("error_code") == ClientErrorCode.MESSAGE_OUTCOME_UNKNOWN.value
        for frame in recording_reply.frames
    )


@pytest.mark.asyncio
async def test_refused_resume_whose_unknown_record_fails_does_not_report_a_failure(
    db_session,
) -> None:
    """A DB error recording the refusal is logged; the run is not failed."""

    owner = _user(db_session, "record-fails-owner")
    task = _expired_running_task(db_session, int(owner.id))
    task_id = int(task.id)
    _pending_row(db_session, task, int(owner.id))
    real_sync = controller_module.transition_task_control_state_sync

    def transition_then_complete(*args: Any, **kwargs: Any):
        snapshot = real_sync(*args, **kwargs)
        _end_run_directly(task_id, TaskStatus.COMPLETED)
        return snapshot

    real_mark = task_execution_service.mark_user_message_delivery_sync
    marks: list[str] = []

    def failing_mark(task_id_arg: int, turn_id: str, status: str):
        # Only the first write fails, so a fall-through into the generic
        # failure handler would get its own delivery write through and then
        # announce a task failure.
        marks.append(status)
        if len(marks) == 1:
            raise OperationalError("UPDATE", {}, Exception("database went away"))
        return real_mark(task_id_arg, turn_id, status)

    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.post_user_message = AsyncMock()
    background_manager = task_execution_service.BackgroundTaskManager()
    publish = AsyncMock()
    agent_patch, manager_patch = _real_resume_environment(agent, background_manager)
    with (
        agent_patch,
        manager_patch,
        patch.object(task_execution_service, "publish_task_event", publish),
        patch.object(
            task_execution_service,
            "mark_user_message_delivery_sync",
            side_effect=failing_mark,
        ),
        patch.object(
            controller_module,
            "transition_task_control_state_sync",
            side_effect=transition_then_complete,
        ),
    ):
        with pytest.raises(TaskCommandDeferred):
            await execute_durable_task_command(
                _message_command(task, owner, TURN_ID, attempt_count=2)
            )
        await _wait_for_resume_to_finish(background_manager, task_id)

    assert marks == [DELIVERY_OUTCOME_UNKNOWN]
    publish.assert_not_awaited()
    agent.post_user_message.assert_not_awaited()
    # Still pending: the retried command settles it through the early refusal.
    assert _row_status(db_session, task_id) == DELIVERY_PENDING
    db_session.expire_all()
    stored = db_session.get(Task, task_id)
    assert stored.status == TaskStatus.COMPLETED
    assert stored.output == FINAL_OUTPUT


@pytest.mark.asyncio
async def test_unknown_record_is_written_before_the_row_write(
    db_session,
    recording_reply: _RecordingReply,
) -> None:
    owner = _user(db_session, "record-order-owner")
    task = _ended_task(db_session, int(owner.id), TaskStatus.COMPLETED)
    task_id = int(task.id)
    _pending_row(db_session, task, int(owner.id))
    command = _message_command(task, owner, TURN_ID, attempt_count=2)
    _processing_command_row(db_session, task, owner, command)
    real_mark = command_execution_service.mark_user_message_delivery_sync
    seen_at_row_write: list[Any] = []

    def mark_checking_record(task_id_arg: int, turn_id: str, status: str):
        if status == DELIVERY_DISPATCHED:
            with get_session_local()() as db:
                seen_at_row_write.append(
                    db.query(TaskExecutionCommand.result)
                    .filter(TaskExecutionCommand.task_id == task_id_arg)
                    .scalar()
                )
        return real_mark(task_id_arg, turn_id, status)

    with (
        _live_control_environment(outcome=ResumeReservationOutcome.RESERVED),
        patch.object(
            command_execution_service,
            "mark_user_message_delivery_sync",
            side_effect=mark_checking_record,
        ),
    ):
        result = await execute_durable_task_command(command)

    assert result == _outcome_unknown_result(task)
    assert seen_at_row_write == [_outcome_unknown_result(task)]
    assert _row_status(db_session, task_id) == DELIVERY_DISPATCHED


def test_unknown_record_from_a_reclaimed_attempt_is_not_written(db_session) -> None:
    owner = _user(db_session, "stale-record-owner")
    task = _ended_task(db_session, int(owner.id), TaskStatus.COMPLETED)
    task_id = int(task.id)
    # The command was reclaimed: attempt 3 now owns it.
    _processing_command_row(
        db_session,
        task,
        owner,
        _message_command(task, owner, TURN_ID, attempt_count=3),
    )

    command_execution_service._record_command_outcome_unknown_sync(
        task_id, TURN_ID, attempt_count=2
    )
    assert _stored_command_result(db_session, task_id, TURN_ID) is None

    command_execution_service._record_command_outcome_unknown_sync(
        task_id, TURN_ID, attempt_count=3
    )
    assert _stored_command_result(db_session, task_id, TURN_ID) == (
        _outcome_unknown_result(task)
    )


@pytest.mark.asyncio
async def test_lost_row_write_ack_keeps_the_unknown_record_across_the_retry(
    db_session,
    recording_reply: _RecordingReply,
) -> None:
    """The ``dispatched`` write commits but its acknowledgement is lost.

    The attempt fails; ``dispatched`` cannot become ``outcome_unknown``, so
    the command row's record is the only trace. The transport's failure
    handling must keep it, and the retry answers outcome unknown.
    """

    owner = _user(db_session, "lost-ack-record-owner")
    task = _ended_task(db_session, int(owner.id), TaskStatus.COMPLETED)
    task_id = int(task.id)
    _pending_row(db_session, task, int(owner.id))
    enqueued = enqueue_task_command(
        db_session,
        task_id=task_id,
        actor_user_id=int(owner.id),
        command_id=TURN_ID,
        kind=TaskCommandKind.MESSAGE,
        payload=dict(_message_command(task, owner, TURN_ID).payload),
    )
    stored = db_session.get(TaskExecutionCommand, enqueued.command_id)
    # An earlier attempt claimed the row and crashed; the next claim is #2.
    stored.attempt_count = 1
    db_session.commit()
    real_mark = command_execution_service.mark_user_message_delivery_sync

    def lose_ack(task_id_arg: int, turn_id: str, status: str):
        transition = real_mark(task_id_arg, turn_id, status)
        if status == DELIVERY_DISPATCHED:
            raise OperationalError("COMMIT", {}, Exception("connection reset"))
        return transition

    with (
        _live_control_environment(outcome=ResumeReservationOutcome.RESERVED),
        patch.object(
            command_execution_service,
            "mark_user_message_delivery_sync",
            side_effect=lose_ack,
        ),
    ):
        assert await dispatch_one_task_command(
            execute_durable_task_command, command_db_id=enqueued.command_id
        )

    db_session.expire_all()
    stored = db_session.get(TaskExecutionCommand, enqueued.command_id)
    assert stored.status == COMMAND_PENDING
    assert stored.attempt_count == 2
    assert stored.result == _outcome_unknown_result(task)
    assert _row_status(db_session, task_id) == DELIVERY_DISPATCHED

    stored.claim_expires_at = None
    stored.retry_available_at = None
    db_session.commit()
    recording_reply.frames.clear()
    with _live_control_environment(outcome=ResumeReservationOutcome.RESERVED):
        assert await dispatch_one_task_command(
            execute_durable_task_command, command_db_id=enqueued.command_id
        )

    db_session.expire_all()
    stored = db_session.get(TaskExecutionCommand, enqueued.command_id)
    assert stored.status == COMMAND_COMPLETED
    assert stored.result == _outcome_unknown_result(task)
    _assert_outcome_unknown_frames(recording_reply)
    db_session.expire_all()
    assert db_session.get(Task, task_id).status == TaskStatus.COMPLETED


@pytest.mark.asyncio
async def test_settlement_by_a_reclaimed_attempt_leaves_the_row_to_the_owner(
    db_session,
    recording_reply: _RecordingReply,
) -> None:
    """Attempt 2 lost its claim while its handler ran; attempt 3 owns it.

    Attempt 2 may not write the record, so it must not advance the row
    either: a ``dispatched`` row without the record reads as accepted.
    """

    owner = _user(db_session, "reclaimed-settle-owner")
    task = _ended_task(db_session, int(owner.id), TaskStatus.COMPLETED)
    task_id = int(task.id)
    _pending_row(db_session, task, int(owner.id))
    _processing_command_row(
        db_session,
        task,
        owner,
        _message_command(task, owner, TURN_ID, attempt_count=3),
    )

    with (
        _live_control_environment(outcome=ResumeReservationOutcome.RESERVED),
        pytest.raises(TaskCommandDeferred),
    ):
        # Answered from the still-pending row; the transport's deferral of
        # this stale attempt is fenced out.
        await execute_durable_task_command(
            _message_command(task, owner, TURN_ID, attempt_count=2)
        )

    assert _row_status(db_session, task_id) == DELIVERY_PENDING
    assert _stored_command_result(db_session, task_id, TURN_ID) is None
    assert not any(
        frame.get("error_code") == ClientErrorCode.MESSAGE_OUTCOME_UNKNOWN.value
        for frame in recording_reply.frames
    )

    with _live_control_environment(outcome=ResumeReservationOutcome.RESERVED):
        result = await execute_durable_task_command(
            _message_command(task, owner, TURN_ID, attempt_count=3)
        )

    assert result == _outcome_unknown_result(task)
    assert _row_status(db_session, task_id) == DELIVERY_DISPATCHED
    assert _stored_command_result(db_session, task_id, TURN_ID) == result
    _assert_outcome_unknown_frames(recording_reply)
