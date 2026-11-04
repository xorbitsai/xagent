"""A fresh message whose run ends before its handoff opens a new turn.

A fresh (not retried) MESSAGE routed on a RUNNING or WAITING_FOR_USER row
takes the live path: it claims its delivery row, tries to inject into the
live run, requests a resume and hands off to a resume that claims the lease.
The run can end FAILED (lease recovery) or COMPLETED (its own runner) at any
point in that window. The RESUME_REQUESTED transition and the resume lease
claim refuse an ended run for every message, so it is never flipped back to
RUNNING.

A refused message that was never written into the run is not settled as
unknown: its row is withdrawn and the message is accepted as a new turn,
exactly as if the snapshot had already shown the ended run -- in the same
handler when the transition refuses it, on the durable retry when the lease
claim does. A message the live run already accepted before it ended keeps the
at-most-once answer: never resumed, never resent, outcome unknown.

Every durable test runs against a real processing command row, and lease
recovery ends a run the way production does, including the orphaned-row
reconciliation it runs in the same transaction.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.exc import OperationalError

from tests.shared.async_waits import DB_PROGRESS_TIMEOUT
from tests.web.api.test_durable_message_resume_contention import (
    _live_control_environment,
    _live_task,
    _message_command,
    _user,
    live_task_lease,
)
from tests.web.api.test_recovered_delivery_failed_run import (
    ENDED_STATUSES,
    _complete_run,
    _end_run_directly,
    _expired_running_task,
    _real_resume_environment,
    _update_task,
)
from tests.web.api.test_recovered_delivery_outcome_unknown import (
    MESSAGE,
    TURN_ID,
    _assert_outcome_unknown_frames,
    _outcome_unknown_result,
    _pending_row,
    _RecordingReply,
    _row_status,
    _settled_task,
    _user_rows,
)
from xagent.web.models.chat_message import TaskChatMessage
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
    DELIVERY_FAILED,
    DELIVERY_OUTCOME_UNKNOWN,
    DELIVERY_PENDING,
    mark_user_message_delivery_sync,
    withdraw_pending_user_message_delivery_sync,
)
from xagent.web.services.client_error_messages import ClientErrorCode
from xagent.web.services.task_command_execution import execute_durable_task_command
from xagent.web.services.task_command_transport import (
    COMMAND_PROCESSING,
    TaskCommandDeferred,
    TaskCommandKind,
    enqueue_task_command,
    get_runner_id,
)
from xagent.web.services.task_execution import ResumeReservationOutcome
from xagent.web.services.task_execution_controller import TaskControlState
from xagent.web.services.task_lease_recovery import (
    recover_task_lease_candidate_no_commit,
)
from xagent.web.services.task_lease_service import get_expired_task_lease_candidates
from xagent.web.services.task_orchestrator import TaskTurnOrchestrator, TurnKind

# Re-exported fixture from the contention suite.
live_task_lease = live_task_lease


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'fresh_message_ended_run.db'}")
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


@pytest.fixture()
def begin_turn_spy() -> Iterator[AsyncMock]:
    spy = AsyncMock(wraps=TaskTurnOrchestrator.begin_turn)
    with patch.object(TaskTurnOrchestrator, "begin_turn", spy):
        yield spy


def _recover_to_failed(task_id: int) -> None:
    """Lease recovery's production settlement, in its own transaction.

    ``recover_task_lease_candidate_no_commit`` also reconciles the task's
    orphaned ``pending`` rows in the same transaction; a processing command
    (or a resume request in flight) keeps this message's row out of it.
    """

    with get_session_local()() as db:
        now = datetime.now(timezone.utc)
        candidate = next(
            candidate
            for candidate in get_expired_task_lease_candidates(db, cutoff=now, limit=10)
            if candidate.task_id == task_id
        )
        assert (
            recover_task_lease_candidate_no_commit(db, candidate, recovered_at=now)
            == TaskStatus.FAILED
        )
        db.commit()


def _end_run(task_id: int, ended_status: TaskStatus) -> None:
    if ended_status == TaskStatus.FAILED:
        _recover_to_failed(task_id)
    else:
        assert ended_status == TaskStatus.COMPLETED
        _complete_run(task_id)


def _set_command_attempt(db, task: Task, owner, attempt: int) -> None:
    """The durable command row, processing under ``attempt``."""

    stored = (
        db.query(TaskExecutionCommand)
        .filter(
            TaskExecutionCommand.task_id == int(task.id),
            TaskExecutionCommand.command_id == TURN_ID,
        )
        .first()
    )
    if stored is None:
        enqueued = enqueue_task_command(
            db,
            task_id=int(task.id),
            actor_user_id=int(owner.id),
            command_id=TURN_ID,
            kind=TaskCommandKind.MESSAGE,
            payload=dict(_message_command(task, owner, TURN_ID).payload),
        )
        stored = db.get(TaskExecutionCommand, enqueued.command_id)
    stored.status = COMMAND_PROCESSING
    stored.attempt_count = attempt
    db.commit()


async def _run_command(db, task: Task, owner, *, attempt: int = 1):
    """Run one attempt of the durable MESSAGE against its real command row.

    Every earlier attempt is taken to have deferred, as the tests arrange.
    """

    _set_command_attempt(db, task, owner, attempt)
    return await execute_durable_task_command(
        replace(
            _message_command(task, owner, TURN_ID, attempt_count=attempt),
            defer_count=attempt - 1,
        )
    )


def _stored_command_result(db, task_id: int) -> Any:
    db.expire_all()
    return (
        db.query(TaskExecutionCommand.result)
        .filter(
            TaskExecutionCommand.task_id == task_id,
            TaskExecutionCommand.command_id == TURN_ID,
        )
        .scalar()
    )


def _accepted_result(task_id: int) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "command_id": TURN_ID,
        "kind": TaskCommandKind.MESSAGE.value,
    }


def _assert_started_a_new_turn(db, task_id: int, begin_turn_spy: AsyncMock) -> None:
    """The message ran as a new turn on a new run; the ended run is gone."""

    begin_turn_spy.assert_awaited()
    assert begin_turn_spy.await_args.kwargs["kind"] == TurnKind.APPEND
    rows = [row for row in _user_rows(db, task_id) if row.turn_id == TURN_ID]
    assert len(rows) == 1
    assert rows[0].content == MESSAGE
    db.expire_all()
    stored = db.get(Task, task_id)
    assert stored.status == TaskStatus.RUNNING
    assert stored.run_id not in {None, "live-run"}
    # A new turn resets the ended run's diagnostic, as any APPEND does.
    assert stored.error_message is None


def _assert_run_not_resumed(db, task_id: int, ended_status: TaskStatus) -> None:
    db.expire_all()
    stored = db.get(Task, task_id)
    assert stored.status == ended_status
    assert stored.control_state == ended_status.value.lower()
    assert stored.run_id == "live-run"
    assert stored.runner_id is None
    if ended_status == TaskStatus.FAILED:
        # The ending write's diagnostic stays; nothing resumed the run.
        assert stored.error_message


def _no_outcome_unknown(reply: _RecordingReply) -> bool:
    return not any(
        frame.get("error_code") == ClientErrorCode.MESSAGE_OUTCOME_UNKNOWN.value
        for frame in reply.frames
    )


def _transition_race(task_id: int, ended_status: TaskStatus, fence: str):
    """End the run just before the RESUME_REQUESTED transition writes.

    ``pre_check`` lands it before the transition loads the row;
    ``conditional_update`` after the load, so only the UPDATE's own status
    predicate can refuse it.
    """

    if fence == "pre_check":
        real_sync = controller_module.transition_task_control_state_sync

        def end_then_transition(*args: Any, **kwargs: Any):
            _end_run(task_id, ended_status)
            return real_sync(*args, **kwargs)

        return patch.object(
            controller_module,
            "transition_task_control_state_sync",
            side_effect=end_then_transition,
        )
    real_apply = controller_module.apply_task_control_transition

    def end_then_apply(task_row: Task, *args: Any, **kwargs: Any):
        assert task_row.status == TaskStatus.RUNNING
        _end_run(task_id, ended_status)
        return real_apply(task_row, *args, **kwargs)

    return patch.object(
        controller_module,
        "apply_task_control_transition",
        side_effect=end_then_apply,
    )


def _end_after_transition(task_id: int, end: Any):
    real_sync = controller_module.transition_task_control_state_sync

    def transition_then_end(*args: Any, **kwargs: Any):
        snapshot = real_sync(*args, **kwargs)
        end()
        return snapshot

    return patch.object(
        controller_module,
        "transition_task_control_state_sync",
        side_effect=transition_then_end,
    )


async def _wait_for_resume_to_finish(background_manager: Any, task_id: int) -> None:
    """Wait for the handed-off resume, including one not yet promoted."""

    for _ in range(int(DB_PROGRESS_TIMEOUT / 0.01)):
        coordinator = background_manager.resume_tasks.get(task_id)
        if task_id not in background_manager.running_tasks and (
            coordinator is None or coordinator.done()
        ):
            return
        await asyncio.sleep(0.01)
    raise AssertionError("the resume coordinator never finished")


def _deferred_resume_agent() -> MagicMock:
    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.post_user_message = AsyncMock()
    return agent


def _hold_previous_run(background_manager: Any, task_id: int) -> asyncio.Event:
    """Stand in for the ended run's coroutine; the resume waits for it.

    The handed-off resume awaits the task's previous coroutine before it
    claims the lease, so holding it keeps the claim after the command's own
    post-handoff read of the row.
    """

    release = asyncio.Event()
    background_manager.running_tasks[task_id] = asyncio.create_task(release.wait())
    return release


def _live_owned_task(db_session, owner, live_task_lease) -> Task:
    task = _live_task(db_session, int(owner.id))
    task.runner_id = get_runner_id()
    task.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=1)
    db_session.commit()
    live_task_lease(db_session, task)
    return task


# --- the transition refuses: the handler appends the message itself -------


@pytest.mark.asyncio
@ENDED_STATUSES
@pytest.mark.parametrize("fence", ["pre_check", "conditional_update"])
async def test_run_ending_before_the_transition_appends_the_message(
    db_session,
    recording_reply: _RecordingReply,
    begin_turn_spy: AsyncMock,
    fence: str,
    ended_status: TaskStatus,
) -> None:
    owner = _user(db_session, f"fresh-transition-{fence}-{ended_status.value}")
    task = _expired_running_task(db_session, int(owner.id))
    task_id = int(task.id)

    with (
        _live_control_environment(outcome=ResumeReservationOutcome.RESERVED) as (
            agent,
            background_manager,
        ),
        _transition_race(task_id, ended_status, fence) as raced,
    ):
        result = await _run_command(db_session, task, owner)
        task_execution_service.execute_resume_background.assert_not_called()

    assert raced.call_count == 1
    assert raced.call_args.kwargs["refuse_terminal_status"] is True
    assert result == _accepted_result(task_id)
    # Released once, by the refusal, and never again by the cleanup arm.
    background_manager.release_resume_reservation.assert_called_once_with(task_id)
    background_manager.register_reserved_resume.assert_not_called()
    agent.post_user_message.assert_not_awaited()
    _assert_started_a_new_turn(db_session, task_id, begin_turn_spy)
    assert _no_outcome_unknown(recording_reply)


@pytest.mark.asyncio
async def test_withdrawn_message_retries_when_another_turn_started_meanwhile(
    db_session,
    recording_reply: _RecordingReply,
    begin_turn_spy: AsyncMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The re-read after the withdrawal no longer shows the ended run."""

    caplog.set_level(logging.INFO, logger=command_execution_service.__name__)
    owner = _user(db_session, "fresh-transition-moved-on")
    task = _expired_running_task(db_session, int(owner.id))
    task_id = int(task.id)
    real_withdraw = withdraw_pending_user_message_delivery_sync

    def withdraw_then_start_another_turn(task_id_arg: int, turn_id: str) -> bool:
        withdrawn = real_withdraw(task_id_arg, turn_id)
        _update_task(
            task_id,
            status=TaskStatus.RUNNING,
            control_state=TaskControlState.RUNNING.value,
            run_id="another-run",
            runner_id="another-runner",
            lease_attempt_id="another-attempt",
            lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            error_message=None,
        )
        return withdrawn

    with (
        _live_control_environment(outcome=ResumeReservationOutcome.RESERVED),
        _transition_race(task_id, TaskStatus.FAILED, "pre_check"),
        patch.object(
            command_execution_service,
            "withdraw_pending_user_message_delivery_sync",
            side_effect=withdraw_then_start_another_turn,
        ),
    ):
        with pytest.raises(TaskCommandDeferred) as deferred:
            await _run_command(db_session, task, owner)

    # Nothing of the message remains, so its retry may deliver it afresh.
    assert deferred.value.resend_safe is True
    begin_turn_spy.assert_not_awaited()
    assert _user_rows(db_session, task_id) == []
    db_session.expire_all()
    assert db_session.get(Task, task_id).run_id == "another-run"
    assert _no_outcome_unknown(recording_reply)
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


@pytest.mark.asyncio
async def test_withdrawn_message_retries_while_the_ended_run_is_still_unwinding(
    db_session,
    recording_reply: _RecordingReply,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``begin_turn`` refuses ``bg_inflight``: the ended run's coroutine lives."""

    caplog.set_level(logging.INFO, logger=command_execution_service.__name__)
    owner = _user(db_session, "fresh-transition-bg-inflight")
    task = _expired_running_task(db_session, int(owner.id))
    task_id = int(task.id)
    release = asyncio.Event()
    ended_run = asyncio.create_task(release.wait())
    begin_turn = AsyncMock(wraps=TaskTurnOrchestrator.begin_turn)
    try:
        with (
            _live_control_environment(outcome=ResumeReservationOutcome.RESERVED) as (
                _,
                background_manager,
            ),
            _transition_race(task_id, TaskStatus.FAILED, "pre_check"),
            patch.object(TaskTurnOrchestrator, "begin_turn", begin_turn),
        ):
            background_manager.running_tasks.get.return_value = ended_run
            with pytest.raises(TaskCommandDeferred) as deferred:
                await _run_command(db_session, task, owner)
    finally:
        release.set()
        await ended_run

    assert deferred.value.resend_safe is True
    begin_turn.assert_awaited_once()
    assert _user_rows(db_session, task_id) == []
    _assert_run_not_resumed(db_session, task_id, TaskStatus.FAILED)
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]

    # Once it unwound, the retry appends the message.
    begin_turn = AsyncMock(wraps=TaskTurnOrchestrator.begin_turn)
    with (
        _live_control_environment(outcome=ResumeReservationOutcome.RESERVED),
        patch.object(TaskTurnOrchestrator, "begin_turn", begin_turn),
    ):
        result = await _run_command(db_session, task, owner, attempt=2)
    assert result == _accepted_result(task_id)
    _assert_started_a_new_turn(db_session, task_id, begin_turn)
    assert _no_outcome_unknown(recording_reply)


@pytest.mark.asyncio
async def test_row_settled_before_the_handler_withdraws_it_is_unknown(
    db_session,
    recording_reply: _RecordingReply,
    begin_turn_spy: AsyncMock,
) -> None:
    """Another writer advanced the row: withdrawal is refused, never forced."""

    owner = _user(db_session, "fresh-transition-row-settled")
    task = _expired_running_task(db_session, int(owner.id))
    task_id = int(task.id)
    real_sync = controller_module.transition_task_control_state_sync

    def settle_row_end_run_then_transition(*args: Any, **kwargs: Any):
        mark_user_message_delivery_sync(task_id, TURN_ID, DELIVERY_DISPATCHED)
        _end_run(task_id, TaskStatus.FAILED)
        return real_sync(*args, **kwargs)

    with (
        _live_control_environment(outcome=ResumeReservationOutcome.RESERVED),
        patch.object(
            controller_module,
            "transition_task_control_state_sync",
            side_effect=settle_row_end_run_then_transition,
        ),
    ):
        result = await _run_command(db_session, task, owner)

    assert result == _outcome_unknown_result(task)
    begin_turn_spy.assert_not_awaited()
    assert _row_status(db_session, task_id) == DELIVERY_DISPATCHED
    _assert_run_not_resumed(db_session, task_id, TaskStatus.FAILED)
    _assert_outcome_unknown_frames(recording_reply)


@pytest.mark.asyncio
@pytest.mark.parametrize("delete", ["not_committed", "committed"])
async def test_handler_withdrawal_that_raises_settles_by_the_row_left_behind(
    db_session,
    recording_reply: _RecordingReply,
    begin_turn_spy: AsyncMock,
    delete: str,
) -> None:
    """A failed withdrawal leaves the row unknown; its read-back decides.

    ``not_committed``: the row is still there, so the message is settled as
    outcome unknown, as the resume path does. ``committed``: the delete
    landed and only its acknowledgement was lost; the row is gone, so the
    command defers and its retry appends the message.
    """

    owner = _user(db_session, f"fresh-transition-withdraw-{delete}")
    task = _expired_running_task(db_session, int(owner.id))
    task_id = int(task.id)
    real_withdraw = withdraw_pending_user_message_delivery_sync

    def failing_withdraw(task_id_arg: int, turn_id: str) -> bool:
        if delete == "committed":
            assert real_withdraw(task_id_arg, turn_id)
        raise OperationalError("DELETE", {}, Exception("connection reset"))

    with (
        _live_control_environment(outcome=ResumeReservationOutcome.RESERVED),
        _transition_race(task_id, TaskStatus.FAILED, "pre_check"),
        patch.object(
            command_execution_service,
            "withdraw_pending_user_message_delivery_sync",
            side_effect=failing_withdraw,
        ),
    ):
        if delete == "not_committed":
            result = await _run_command(db_session, task, owner)
        else:
            with pytest.raises(TaskCommandDeferred) as deferred:
                await _run_command(db_session, task, owner)

    begin_turn_spy.assert_not_awaited()
    _assert_run_not_resumed(db_session, task_id, TaskStatus.FAILED)
    if delete == "not_committed":
        assert result == _outcome_unknown_result(task)
        assert _row_status(db_session, task_id) == DELIVERY_DISPATCHED
        _assert_outcome_unknown_frames(recording_reply)
        return

    assert deferred.value.resend_safe is True
    assert _user_rows(db_session, task_id) == []
    # The unknown record written before the row read-back is dropped again.
    assert _stored_command_result(db_session, task_id) is None
    assert _no_outcome_unknown(recording_reply)
    begin_turn = AsyncMock(wraps=TaskTurnOrchestrator.begin_turn)
    with (
        _live_control_environment(outcome=ResumeReservationOutcome.RESERVED),
        patch.object(TaskTurnOrchestrator, "begin_turn", begin_turn),
    ):
        result = await _run_command(db_session, task, owner, attempt=2)
    assert result == _accepted_result(task_id)
    _assert_started_a_new_turn(db_session, task_id, begin_turn)


@pytest.mark.asyncio
@ENDED_STATUSES
async def test_waiting_run_ending_under_the_version_fence_still_appends(
    db_session,
    recording_reply: _RecordingReply,
    begin_turn_spy: AsyncMock,
    ended_status: TaskStatus,
) -> None:
    """A row routed as not RUNNING is also fenced on its state version.

    The run's ending write bumps that version, but the terminal status is
    checked first, so the refusal is the status one: the message is appended,
    not deferred as a moved row.
    """

    owner = _user(db_session, f"fresh-waiting-ended-{ended_status.value}")
    task = _expired_running_task(db_session, int(owner.id))
    task.status = TaskStatus.WAITING_FOR_USER
    task.control_state = TaskControlState.WAITING_FOR_USER.value
    db_session.commit()
    task_id = int(task.id)
    real_sync = controller_module.transition_task_control_state_sync

    def end_then_transition(*args: Any, **kwargs: Any):
        _end_run_directly(task_id, ended_status)
        return real_sync(*args, **kwargs)

    with (
        _live_control_environment(outcome=ResumeReservationOutcome.RESERVED),
        patch.object(
            controller_module,
            "transition_task_control_state_sync",
            side_effect=end_then_transition,
        ) as raced,
    ):
        result = await _run_command(db_session, task, owner)
        task_execution_service.execute_resume_background.assert_not_called()

    assert raced.call_args.kwargs["expected_state_version"] is not None
    assert raced.call_args.kwargs["refuse_terminal_status"] is True
    assert result == _accepted_result(task_id)
    _assert_started_a_new_turn(db_session, task_id, begin_turn_spy)
    assert _no_outcome_unknown(recording_reply)


# --- the lease claim refuses: the resume withdraws, the retry appends -----


@pytest.mark.asyncio
@ENDED_STATUSES
@pytest.mark.parametrize("read", ["before_the_resume", "after_the_withdrawal"])
async def test_run_ending_before_the_lease_claim_appends_on_retry(
    db_session,
    recording_reply: _RecordingReply,
    ended_status: TaskStatus,
    read: str,
) -> None:
    """The resume's claim is refused after the handler handed the turn off.

    The message was never injected, so the resume withdraws its row and the
    command's retry, finding no row, accepts it as a new turn. ``read`` pins
    whether the command read the row before the resume withdrew it (still
    pending: the resume is held behind the ended run's coroutine) or after
    (absent, answered by the handoff marker); both defer.
    """

    owner = _user(db_session, f"fresh-claim-{read}-{ended_status.value}")
    task = _expired_running_task(db_session, int(owner.id))
    task_id = int(task.id)
    agent = _deferred_resume_agent()
    background_manager = task_execution_service.BackgroundTaskManager()
    resume_spy = AsyncMock(wraps=task_execution_service.execute_resume_background)
    real_status = command_execution_service._load_command_message_delivery_status
    status_reads: list[str | None] = []

    def recording_status(task_id_arg: int, turn_id: str) -> str | None:
        status = real_status(task_id_arg, turn_id)
        # The first read precedes the handler; only the post-handoff one waits.
        deadline = time.monotonic() + 5
        while (
            read == "after_the_withdrawal"
            and status_reads
            and status is not None
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
            status = real_status(task_id_arg, turn_id)
        status_reads.append(status)
        return status

    release = (
        _hold_previous_run(background_manager, task_id)
        if read == "before_the_resume"
        else None
    )
    agent_patch, manager_patch = _real_resume_environment(agent, background_manager)
    publish = AsyncMock()
    with (
        agent_patch,
        manager_patch,
        patch.object(task_execution_service, "execute_resume_background", resume_spy),
        patch.object(task_execution_service, "publish_task_event", publish),
        _end_after_transition(task_id, lambda: _end_run(task_id, ended_status)),
        patch.object(
            command_execution_service,
            "_load_command_message_delivery_status",
            side_effect=recording_status,
        ),
    ):
        with pytest.raises(TaskCommandDeferred) as deferred:
            await _run_command(db_session, task, owner)
        assert resume_spy.await_count == 1
        assert resume_spy.await_args.kwargs["refuse_terminal_status"] is True
        assert resume_spy.await_args.kwargs["delivery_claimed_fresh"] is True
        if release is not None:
            release.set()
        await _wait_for_resume_to_finish(background_manager, task_id)

    if read == "before_the_resume":
        assert status_reads[-1] == DELIVERY_PENDING
        assert "runtime injection" in str(deferred.value)
    else:
        assert status_reads[-1] is None
        assert "new turn" in str(deferred.value)
    agent.post_user_message.assert_not_awaited()
    # Withdrawn, and the ended run was not resumed.
    assert _user_rows(db_session, task_id) == []
    _assert_run_not_resumed(db_session, task_id, ended_status)
    assert not any(
        call.args[0].get("error_code") == ClientErrorCode.TASK_BUSY.value
        for call in publish.await_args_list
    )

    begin_turn = AsyncMock(wraps=TaskTurnOrchestrator.begin_turn)
    with (
        _live_control_environment(outcome=ResumeReservationOutcome.RESERVED) as (
            retry_agent,
            _,
        ),
        patch.object(TaskTurnOrchestrator, "begin_turn", begin_turn),
    ):
        result = await _run_command(db_session, task, owner, attempt=2)
        task_execution_service.execute_resume_background.assert_not_called()

    assert result == _accepted_result(task_id)
    retry_agent.post_user_message.assert_not_awaited()
    _assert_started_a_new_turn(db_session, task_id, begin_turn)
    assert _no_outcome_unknown(recording_reply)


@pytest.mark.asyncio
@pytest.mark.parametrize("withdrawal", ["refused", "raises"])
async def test_resume_that_cannot_withdraw_the_row_settles_it_unknown(
    db_session,
    recording_reply: _RecordingReply,
    withdrawal: str,
) -> None:
    """The resume's withdrawal finds the row not its own, or fails.

    Neither proves the message undelivered any more, so the resume records
    the row as outcome unknown and the retry answers from it.
    """

    owner = _user(db_session, f"fresh-claim-withdraw-{withdrawal}")
    task = _expired_running_task(db_session, int(owner.id))
    task_id = int(task.id)
    agent = _deferred_resume_agent()
    background_manager = task_execution_service.BackgroundTaskManager()

    def withdraw(task_id_arg: int, turn_id: str) -> bool:
        if withdrawal == "raises":
            raise OperationalError("DELETE", {}, Exception("connection reset"))
        return False

    release = _hold_previous_run(background_manager, task_id)
    agent_patch, manager_patch = _real_resume_environment(agent, background_manager)
    with (
        agent_patch,
        manager_patch,
        patch.object(task_execution_service, "publish_task_event", AsyncMock()),
        _end_after_transition(task_id, lambda: _end_run(task_id, TaskStatus.FAILED)),
        patch.object(
            task_execution_service,
            "withdraw_pending_user_message_delivery_sync",
            side_effect=withdraw,
        ),
    ):
        with pytest.raises(TaskCommandDeferred):
            await _run_command(db_session, task, owner)
        release.set()
        await _wait_for_resume_to_finish(background_manager, task_id)

    assert _row_status(db_session, task_id) == DELIVERY_OUTCOME_UNKNOWN
    _assert_run_not_resumed(db_session, task_id, TaskStatus.FAILED)
    recording_reply.frames.clear()
    with _live_control_environment(outcome=ResumeReservationOutcome.RESERVED):
        result = await _run_command(db_session, task, owner, attempt=2)
    assert result == _outcome_unknown_result(task)
    _assert_outcome_unknown_frames(recording_reply)


@pytest.mark.asyncio
async def test_resume_withdrawal_that_commits_but_raises_records_nothing(
    db_session,
    recording_reply: _RecordingReply,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The withdrawal's DELETE commits; only its acknowledgement is lost.

    The raise does not prove the row still pending: the outcome-unknown write
    that follows finds no row and records nothing, and says so rather than
    claiming a record. The message was never injected, so the retry finds no
    row and appends it as a new turn.
    """

    owner = _user(db_session, "fresh-claim-withdraw-lost-ack")
    task = _expired_running_task(db_session, int(owner.id))
    task_id = int(task.id)
    agent = _deferred_resume_agent()
    background_manager = task_execution_service.BackgroundTaskManager()
    real_withdraw = withdraw_pending_user_message_delivery_sync

    def withdraw_then_lose_the_ack(task_id_arg: int, turn_id: str) -> bool:
        assert real_withdraw(task_id_arg, turn_id)
        raise OperationalError("DELETE", {}, Exception("connection reset"))

    release = _hold_previous_run(background_manager, task_id)
    agent_patch, manager_patch = _real_resume_environment(agent, background_manager)
    with (
        caplog.at_level(logging.WARNING, logger=task_execution_service.logger.name),
        agent_patch,
        manager_patch,
        patch.object(task_execution_service, "publish_task_event", AsyncMock()),
        _end_after_transition(task_id, lambda: _end_run(task_id, TaskStatus.FAILED)),
        patch.object(
            task_execution_service,
            "withdraw_pending_user_message_delivery_sync",
            side_effect=withdraw_then_lose_the_ack,
        ),
    ):
        with pytest.raises(TaskCommandDeferred):
            await _run_command(db_session, task, owner)
        release.set()
        await _wait_for_resume_to_finish(background_manager, task_id)

    messages = [record.getMessage() for record in caplog.records]
    assert any("no row left to record as outcome unknown" in m for m in messages)
    assert not any("recorded delivery" in m for m in messages)
    assert _user_rows(db_session, task_id) == []
    _assert_run_not_resumed(db_session, task_id, TaskStatus.FAILED)

    begin_turn = AsyncMock(wraps=TaskTurnOrchestrator.begin_turn)
    with (
        _live_control_environment(outcome=ResumeReservationOutcome.RESERVED),
        patch.object(TaskTurnOrchestrator, "begin_turn", begin_turn),
    ):
        result = await _run_command(db_session, task, owner, attempt=2)
    assert result == _accepted_result(task_id)
    _assert_started_a_new_turn(db_session, task_id, begin_turn)
    assert _no_outcome_unknown(recording_reply)


@pytest.mark.asyncio
async def test_retry_that_finds_its_row_withdrawn_mid_settlement_appends_later(
    db_session,
    recording_reply: _RecordingReply,
) -> None:
    """The retry and this worker's own resume interleave.

    Attempt 2 reads the row still pending (a recovered claim on an ended run)
    and starts settling it as outcome unknown; before its row write the first
    attempt's resume withdraws the row. A missing row was never delivered:
    the retry defers instead of answering unknown, drops the unknown record it
    wrote, and the next attempt appends the message.
    """

    owner = _user(db_session, "fresh-retry-withdrawn")
    task = _expired_running_task(db_session, int(owner.id))
    task_id = int(task.id)
    _pending_row(db_session, task, int(owner.id))
    _set_command_attempt(db_session, task, owner, 1)
    _update_task(task_id, control_state=TaskControlState.RESUME_REQUESTED.value)
    _recover_to_failed(task_id)
    # The resume request and the processing command kept the row pending.
    assert _row_status(db_session, task_id) == DELIVERY_PENDING
    real_record = command_execution_service._record_command_outcome_unknown_sync

    def record_then_resume_withdraws(*args: Any, **kwargs: Any) -> bool:
        owned = real_record(*args, **kwargs)
        assert withdraw_pending_user_message_delivery_sync(task_id, TURN_ID)
        return owned

    with (
        _live_control_environment(outcome=ResumeReservationOutcome.RESERVED),
        patch.object(
            command_execution_service,
            "_record_command_outcome_unknown_sync",
            side_effect=record_then_resume_withdraws,
        ),
    ):
        with pytest.raises(TaskCommandDeferred) as deferred:
            await _run_command(db_session, task, owner, attempt=2)

    assert "new turn" in str(deferred.value)
    assert _user_rows(db_session, task_id) == []
    assert _stored_command_result(db_session, task_id) is None
    assert _no_outcome_unknown(recording_reply)

    begin_turn = AsyncMock(wraps=TaskTurnOrchestrator.begin_turn)
    with (
        _live_control_environment(outcome=ResumeReservationOutcome.RESERVED),
        patch.object(TaskTurnOrchestrator, "begin_turn", begin_turn),
    ):
        result = await _run_command(db_session, task, owner, attempt=3)
    assert result == _accepted_result(task_id)
    _assert_started_a_new_turn(db_session, task_id, begin_turn)


@pytest.mark.asyncio
async def test_non_durable_caller_is_told_to_resend(
    db_session,
) -> None:
    """Defensive only: production reaches this handler through a command.

    ``handle_task_message`` is called by the durable executor, or through
    ``handle_missing_task_message`` for a task it creates, so a live-path
    message always has a command row. A caller without one still gets a safe
    answer: not accepted, resend. Nothing of the message remains.
    """

    owner = _user(db_session, "fresh-non-durable")
    task = _expired_running_task(db_session, int(owner.id))
    task_id = int(task.id)
    agent = _deferred_resume_agent()
    background_manager = task_execution_service.BackgroundTaskManager()
    reply = _RecordingReply()
    message_data = {
        "type": "chat_message",
        "message": MESSAGE,
        "client_message_id": TURN_ID,
        "files": [],
        "user": owner,
    }
    agent_patch, manager_patch = _real_resume_environment(agent, background_manager)
    with (
        agent_patch,
        manager_patch,
        patch.object(task_execution_service, "publish_task_event", AsyncMock()),
        # COMPLETED: with no command row, lease recovery would reconcile the
        # pending row away before the resume could withdraw it.
        _end_after_transition(task_id, lambda: _complete_run(task_id)),
    ):
        await command_execution_service.handle_task_message(
            reply, task_id, dict(message_data)
        )
        await _wait_for_resume_to_finish(background_manager, task_id)

    rejected = [f for f in reply.frames if f.get("type") == "message_rejected"]
    assert len(rejected) == 1
    assert rejected[0]["rejection_outcome"] == "not_accepted"
    assert rejected[0]["retry_with_new_id"] is True
    assert rejected[0]["error_code"] == ClientErrorCode.MESSAGE_DELIVERY_FAILED.value
    assert _user_rows(db_session, task_id) == []
    _assert_run_not_resumed(db_session, task_id, TaskStatus.COMPLETED)


# --- the message reached the run before it ended -------------------------


@pytest.mark.asyncio
@ENDED_STATUSES
async def test_posted_message_whose_run_ends_before_the_transition_is_unknown(
    live_task_lease,
    db_session,
    recording_reply: _RecordingReply,
    begin_turn_spy: AsyncMock,
    ended_status: TaskStatus,
) -> None:
    """The live run accepted the message, then ended before the handoff.

    Whether it read the message is unknown: the run is not resumed, the
    message is not run again as a new turn, and the sender is told so.
    """

    owner = _user(db_session, f"posted-transition-{ended_status.value}")
    task = _live_owned_task(db_session, owner, live_task_lease)
    task_id = int(task.id)
    real_sync = controller_module.transition_task_control_state_sync

    def end_then_transition(*args: Any, **kwargs: Any):
        _end_run_directly(task_id, ended_status)
        return real_sync(*args, **kwargs)

    with (
        _live_control_environment(outcome=ResumeReservationOutcome.RESERVED) as (
            agent,
            background_manager,
        ),
        patch.object(
            controller_module,
            "transition_task_control_state_sync",
            side_effect=end_then_transition,
        ),
    ):
        result = await _run_command(db_session, task, owner)
        task_execution_service.execute_resume_background.assert_not_called()

    agent.post_user_message.assert_awaited_once()
    assert result == _outcome_unknown_result(task)
    begin_turn_spy.assert_not_awaited()
    background_manager.register_reserved_resume.assert_not_called()
    background_manager.release_resume_reservation.assert_called_once_with(task_id)
    assert _row_status(db_session, task_id) == DELIVERY_DISPATCHED
    _assert_run_not_resumed(db_session, task_id, ended_status)
    _assert_outcome_unknown_frames(recording_reply)


@pytest.mark.asyncio
@ENDED_STATUSES
async def test_posted_message_whose_run_ends_before_the_claim_keeps_its_notice(
    live_task_lease,
    db_session,
    ended_status: TaskStatus,
) -> None:
    """Injected and handed off, then the run ended: the recovered semantics.

    The command already answered accepted; the ended run is not resumed and a
    task-wide outcome-unknown notice replaces the answer no resume will give.
    """

    owner = _user(db_session, f"posted-claim-{ended_status.value}")
    task = _live_owned_task(db_session, owner, live_task_lease)
    task_id = int(task.id)
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
        _end_after_transition(
            task_id, lambda: _end_run_directly(task_id, ended_status)
        ),
    ):
        result = await _run_command(db_session, task, owner)
        assert resume_spy.await_args.kwargs["delivery_already_dispatched"] is True
        assert resume_spy.await_args.kwargs["delivery_claimed_fresh"] is True
        await _wait_for_resume_to_finish(background_manager, task_id)

    assert result == _accepted_result(task_id)
    assert _row_status(db_session, task_id) == DELIVERY_DISPATCHED
    _assert_run_not_resumed(db_session, task_id, ended_status)
    notices = [
        call.args[0]
        for call in publish.await_args_list
        if call.args[0].get("error_code")
        == ClientErrorCode.MESSAGE_OUTCOME_UNKNOWN.value
    ]
    assert len(notices) == 1
    assert notices[0]["turn_id"] == TURN_ID
    assert MESSAGE not in str(notices[0])


@pytest.mark.asyncio
@ENDED_STATUSES
@pytest.mark.parametrize("marker", ["fails", "commits_after_the_claim"])
async def test_posted_message_whose_marker_is_missing_at_the_claim_is_not_withdrawn(
    live_task_lease,
    db_session,
    begin_turn_spy: AsyncMock,
    ended_status: TaskStatus,
    marker: str,
) -> None:
    """Injected and handed off, but the row still reads ``pending``.

    The handler's ``dispatched`` marker after a posted handoff is best effort:
    it can fail, or the resume can reach its refused claim before it commits.
    The row then reads ``pending`` although the run already has the message,
    so it must not be withdrawn as never injected (which would let a retry
    append it again). The resume keeps the posted-claim notice instead.
    """

    owner = _user(db_session, f"posted-claim-marker-{marker}-{ended_status.value}")
    task = _live_owned_task(db_session, owner, live_task_lease)
    task_id = int(task.id)
    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.post_user_message = AsyncMock(
        return_value=task_execution_service.UserMessageInjectionOutcome.POSTED_FRESH
    )
    background_manager = task_execution_service.BackgroundTaskManager()
    resume_spy = AsyncMock(wraps=task_execution_service.execute_resume_background)
    resume_settled = threading.Event()
    real_mark = command_execution_service.mark_user_message_delivery_sync
    real_withdraw = task_execution_service.withdraw_pending_user_message_delivery_sync

    def handler_marker(task_id_arg: int, turn_id: str, status: str):
        if status != DELIVERY_DISPATCHED:
            return real_mark(task_id_arg, turn_id, status)
        if marker == "fails":
            raise OperationalError("UPDATE", {}, Exception("connection reset"))
        # Commit only once the resume has settled its refused claim.
        assert resume_settled.wait(timeout=DB_PROGRESS_TIMEOUT)
        return real_mark(task_id_arg, turn_id, status)

    def recording_withdraw(task_id_arg: int, turn_id: str) -> bool:
        try:
            return real_withdraw(task_id_arg, turn_id)
        finally:
            resume_settled.set()

    async def recording_publish(event: dict[str, Any], *args: Any) -> None:
        if event.get("error_code") == ClientErrorCode.MESSAGE_OUTCOME_UNKNOWN.value:
            resume_settled.set()

    publish = AsyncMock(side_effect=recording_publish)
    agent_patch, manager_patch = _real_resume_environment(agent, background_manager)
    with (
        agent_patch,
        manager_patch,
        patch.object(task_execution_service, "execute_resume_background", resume_spy),
        patch.object(task_execution_service, "publish_task_event", publish),
        patch.object(
            task_execution_service,
            "withdraw_pending_user_message_delivery_sync",
            side_effect=recording_withdraw,
        ),
        patch.object(
            command_execution_service,
            "mark_user_message_delivery_sync",
            side_effect=handler_marker,
        ),
        _end_after_transition(
            task_id, lambda: _end_run_directly(task_id, ended_status)
        ),
    ):
        result = await _run_command(db_session, task, owner)
        # Derived from ``posted``, not from the best-effort marker write.
        assert resume_spy.await_args.kwargs["delivery_already_dispatched"] is True
        assert resume_spy.await_args.kwargs["delivery_claimed_fresh"] is True
        await _wait_for_resume_to_finish(background_manager, task_id)

    # The command answered from the posted handoff (the documented
    # limitation: it completed as accepted before the claim was refused).
    assert result == _accepted_result(task_id)
    # Not withdrawn: the row the run already holds is still there.
    rows = [row for row in _user_rows(db_session, task_id) if row.turn_id == TURN_ID]
    assert len(rows) == 1
    assert _row_status(db_session, task_id) == (
        DELIVERY_PENDING if marker == "fails" else DELIVERY_DISPATCHED
    )
    _assert_run_not_resumed(db_session, task_id, ended_status)
    notices = [
        call.args[0]
        for call in publish.await_args_list
        if call.args[0].get("error_code")
        == ClientErrorCode.MESSAGE_OUTCOME_UNKNOWN.value
    ]
    assert len(notices) == 1
    assert notices[0]["turn_id"] == TURN_ID

    # Never appended as a new turn.
    begin_turn_spy.assert_not_awaited()


# --- runs that did not end are unaffected --------------------------------


@pytest.mark.asyncio
async def test_fresh_message_to_a_live_run_still_injects(
    live_task_lease,
    db_session,
    recording_reply: _RecordingReply,
    begin_turn_spy: AsyncMock,
) -> None:
    owner = _user(db_session, "fresh-live-inject")
    task = _live_owned_task(db_session, owner, live_task_lease)
    task_id = int(task.id)

    with _live_control_environment(outcome=ResumeReservationOutcome.RESERVED) as (
        agent,
        background_manager,
    ):
        result = await _run_command(db_session, task, owner)
        resume = task_execution_service.execute_resume_background
        assert resume.call_count == 1
        assert resume.call_args.kwargs["refuse_terminal_status"] is True
        assert resume.call_args.kwargs["delivery_claimed_fresh"] is True

    agent.post_user_message.assert_awaited_once()
    assert result == _accepted_result(task_id)
    background_manager.register_reserved_resume.assert_called_once()
    begin_turn_spy.assert_not_awaited()
    assert _row_status(db_session, task_id) == DELIVERY_DISPATCHED
    db_session.expire_all()
    stored = db_session.get(Task, task_id)
    assert stored.status == TaskStatus.RUNNING
    assert stored.run_id == "live-run"
    assert stored.control_state == TaskControlState.RESUME_REQUESTED.value


@pytest.mark.asyncio
async def test_fresh_message_to_a_paused_resume_request_still_resumes(
    db_session,
    begin_turn_spy: AsyncMock,
) -> None:
    """A PAUSED run with a resume pending is resumable: the fences pass it."""

    owner = _user(db_session, "fresh-paused-resume")
    task = _settled_task(
        db_session, int(owner.id), status=TaskStatus.PAUSED, run_id="live-run"
    )
    task.control_state = TaskControlState.RESUME_REQUESTED.value
    db_session.commit()
    task_id = int(task.id)

    with _live_control_environment(outcome=ResumeReservationOutcome.RESERVED) as (
        _,
        background_manager,
    ):
        with pytest.raises(TaskCommandDeferred):
            await _run_command(db_session, task, owner)
        resume = task_execution_service.execute_resume_background
        assert resume.call_count == 1
        assert resume.call_args.kwargs["refuse_terminal_status"] is True

    background_manager.register_reserved_resume.assert_called_once()
    begin_turn_spy.assert_not_awaited()
    assert _row_status(db_session, task_id) == DELIVERY_PENDING

    ended: list[TaskStatus] = []
    lease = task_execution_service._acquire_resume_task_lease(
        task_id,
        int(owner.id),
        "live-run",
        refuse_terminal_status=True,
        ended_status_out=ended,
    )
    assert lease is not None
    assert ended == []
    db_session.expire_all()
    assert db_session.get(Task, task_id).status == TaskStatus.RUNNING


@pytest.mark.asyncio
@ENDED_STATUSES
async def test_resume_request_left_on_an_ended_run_routes_to_a_new_turn(
    db_session,
    recording_reply: _RecordingReply,
    begin_turn_spy: AsyncMock,
    ended_status: TaskStatus,
) -> None:
    """The snapshot already shows the run ended, with a resume still pending."""

    owner = _user(db_session, f"ended-resume-request-{ended_status.value}")
    task = _settled_task(
        db_session, int(owner.id), status=ended_status, run_id="live-run"
    )
    task.control_state = TaskControlState.RESUME_REQUESTED.value
    db_session.commit()
    task_id = int(task.id)

    with _live_control_environment(outcome=ResumeReservationOutcome.RESERVED) as (
        agent,
        background_manager,
    ):
        result = await _run_command(db_session, task, owner)
        task_execution_service.execute_resume_background.assert_not_called()

    assert result == _accepted_result(task_id)
    # Routed before the live path: no agent built, no resume slot taken.
    agent.supports_live_control.assert_not_called()
    background_manager.try_reserve_resume.assert_not_called()
    _assert_started_a_new_turn(db_session, task_id, begin_turn_spy)
    assert _no_outcome_unknown(recording_reply)


@pytest.mark.asyncio
async def test_claim_refused_by_a_live_owner_keeps_the_failed_delivery(
    db_session,
) -> None:
    """Only an ended run withdraws a fresh row; a live owner fails it as before."""

    owner = _user(db_session, "fresh-claim-live-owner")
    task = _expired_running_task(db_session, int(owner.id))
    task_id = int(task.id)
    agent = _deferred_resume_agent()
    background_manager = task_execution_service.BackgroundTaskManager()
    publish = AsyncMock()
    agent_patch, manager_patch = _real_resume_environment(agent, background_manager)
    with (
        agent_patch,
        manager_patch,
        patch.object(task_execution_service, "publish_task_event", publish),
        _end_after_transition(
            task_id,
            lambda: _update_task(
                task_id,
                runner_id="foreign-runner",
                lease_attempt_id="foreign-attempt",
                lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            ),
        ),
    ):
        with pytest.raises(TaskCommandDeferred):
            await _run_command(db_session, task, owner)
        await _wait_for_resume_to_finish(background_manager, task_id)

    agent.post_user_message.assert_not_awaited()
    assert _row_status(db_session, task_id) == DELIVERY_FAILED
    assert any(
        call.args[0].get("error_code") == ClientErrorCode.TASK_BUSY.value
        for call in publish.await_args_list
    )
    db_session.expire_all()
    stored = db_session.get(Task, task_id)
    assert stored.status == TaskStatus.RUNNING
    assert stored.runner_id == "foreign-runner"


# --- units -----------------------------------------------------------------


def test_withdrawal_removes_only_a_pending_row(db_session) -> None:
    owner = _user(db_session, "withdraw-unit")
    task = _settled_task(
        db_session, int(owner.id), status=TaskStatus.FAILED, run_id="live-run"
    )
    task_id = int(task.id)
    for turn_id, status in (
        ("pending-turn", DELIVERY_PENDING),
        ("dispatched-turn", DELIVERY_DISPATCHED),
    ):
        db_session.add(
            TaskChatMessage(
                task_id=task_id,
                user_id=int(owner.id),
                role="user",
                message_type="user_message",
                content=MESSAGE,
                turn_id=turn_id,
                delivery_status=status,
            )
        )
    db_session.commit()

    assert withdraw_pending_user_message_delivery_sync(task_id, "pending-turn")
    assert not withdraw_pending_user_message_delivery_sync(task_id, "pending-turn")
    assert not withdraw_pending_user_message_delivery_sync(task_id, "dispatched-turn")
    assert [row.turn_id for row in _user_rows(db_session, task_id)] == [
        "dispatched-turn"
    ]


@ENDED_STATUSES
def test_resume_lease_claim_reports_the_ended_status(
    db_session, ended_status: TaskStatus
) -> None:
    owner = _user(db_session, f"claim-ended-status-{ended_status.value}")
    task = _settled_task(
        db_session, int(owner.id), status=ended_status, run_id="another-run"
    )
    ended: list[TaskStatus] = []
    not_resumable: list[bool] = []

    lease = task_execution_service._acquire_resume_task_lease(
        int(task.id),
        int(owner.id),
        "live-run",
        refuse_terminal_status=True,
        run_not_resumable_out=not_resumable,
        ended_status_out=ended,
    )

    assert lease is None
    # Reported whatever the run: a rotated run that ended is still ended.
    assert ended == [ended_status]
    assert not_resumable == [True]
