"""Live-control writes are fenced on the acquisition driving the local run.

An expired RUNNING takeover keeps ``run_id`` and mints only a new
``lease_attempt_id``. A process whose lease was taken over while its local
run kept going (a zombie of the earlier attempt) therefore still matches a
run id fence, so PAUSE_REQUESTED and the live-message RESUME_REQUESTED
handoff must also match the exact acquisition. A late result must not
resurrect a row that already settled terminal either.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select, update

from tests.web.api.test_durable_message_resume_contention import (
    _live_control_environment,
    _live_task,
    _message_command,
    _user,
)
from tests.web.api.test_durable_message_resume_contention import (
    db_session as db_session_fixture,
)
from tests.web.api.test_recovered_delivery_outcome_unknown import (
    TURN_ID,
    _pending_row,
    _row_status,
)
from tests.web.services.task_lease_shared import (
    live_task_lease as live_task_lease_fixture,
)
from xagent.core.agent.runner import UserMessageInjectionOutcome
from xagent.web.models.database import get_session_local
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services import agent_service_manager
from xagent.web.services import task_command_execution as commands
from xagent.web.services import task_coordinator_runtime
from xagent.web.services import task_execution as execution
from xagent.web.services import task_lease_service as leases
from xagent.web.services import task_setup_snapshot
from xagent.web.services.chat_history_service import (
    DELIVERY_DISPATCHED,
    DELIVERY_OUTCOME_UNKNOWN,
    DELIVERY_PENDING,
)
from xagent.web.services.client_error_messages import ClientErrorCode
from xagent.web.services.task_command_execution import (
    ClientVisibleTaskCommandDeferred,
)
from xagent.web.services.task_command_transport import (
    ClaimedTaskCommand,
    TaskCommandDeferred,
    TaskCommandKind,
)
from xagent.web.services.task_coordinator_service import TaskLease as CoordinatorLease
from xagent.web.services.task_execution import ResumeReservationOutcome
from xagent.web.services.task_execution_controller import (
    StaleTaskRunError,
    TaskControlState,
    transition_task_control_state_sync,
)
from xagent.web.services.task_lease_service import TaskLease, get_runner_id

db_session = db_session_fixture
live_task_lease = live_task_lease_fixture

SUCCESSOR_RUNNER = "successor-runner"
SUCCESSOR_ATTEMPT = "successor-attempt"


def _leased_task(db_session) -> Task:
    owner = _user(db_session, f"fence-owner-{datetime.now().timestamp()}")
    task = _live_task(db_session, int(owner.id))
    task.runner_id = get_runner_id()
    task.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=1)
    task.control_state = TaskControlState.RUNNING.value
    task.state_version = 3
    db_session.commit()
    return task


def _take_over(task_id: int) -> None:
    """What ``acquire_task_lease_no_commit`` leaves after an expired takeover."""
    with get_session_local()() as db:
        db.execute(
            update(Task)
            .where(Task.id == task_id)
            .values(
                runner_id=SUCCESSOR_RUNNER,
                lease_attempt_id=SUCCESSOR_ATTEMPT,
                lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            )
        )
        db.commit()


def _row(db_session, task_id: int) -> Task:
    db_session.expire_all()
    task = db_session.get(Task, task_id)
    assert task is not None
    return task


def _patch_pause_runtime(monkeypatch, task: Task) -> AsyncMock:
    snapshot_task = SimpleNamespace(
        user_id=int(task.user_id), run_id=str(task.run_id), status=TaskStatus.RUNNING
    )
    monkeypatch.setattr(
        task_setup_snapshot,
        "load_task_setup_snapshot_sync",
        lambda *a, **k: SimpleNamespace(task=snapshot_task, runtime_user=object()),
    )
    monkeypatch.setattr(commands, "resolve_execution_scope_off_turn", lambda *a: None)
    pause_execution = AsyncMock(return_value=True)
    service = SimpleNamespace(pause_execution=pause_execution)
    monkeypatch.setattr(
        agent_service_manager,
        "get_agent_manager",
        lambda: SimpleNamespace(get_agent_for_task=AsyncMock(return_value=service)),
    )
    return pause_execution


def _pause_message(task: Task) -> dict:
    return {"user": SimpleNamespace(id=int(task.user_id), is_admin=False)}


# ---------------------------------------------------------------------------
# PAUSE_REQUESTED
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_current_owner_pause_is_applied(
    live_task_lease, db_session, monkeypatch
) -> None:
    task = _leased_task(db_session)
    live_task_lease(db_session, task)
    pause_execution = _patch_pause_runtime(monkeypatch, task)
    publish = AsyncMock()
    monkeypatch.setattr(commands, "publish_task_event", publish)
    reply = AsyncMock()
    message = _pause_message(task)

    try:
        await commands.pause_task(reply, int(task.id), message)
        assert execution._is_task_pause_accepted(int(task.id))
    finally:
        execution._clear_task_pause_accepted(int(task.id))

    pause_execution.assert_awaited_once_with()
    assert "_durable_command_error" not in message
    assert publish.await_args.args[0]["type"] == "task_pause_requested"
    stored = _row(db_session, int(task.id))
    assert stored.control_state == TaskControlState.PAUSE_REQUESTED.value
    assert stored.state_version == 4


@pytest.mark.asyncio
async def test_zombie_pause_interrupts_locally_and_defers_for_the_successor(
    live_task_lease, db_session, monkeypatch
) -> None:
    task = _leased_task(db_session)
    # This process still heartbeats the earlier attempt; its local run is
    # alive and accepts the interrupt.
    live_task_lease(db_session, task)
    _take_over(int(task.id))
    pause_execution = _patch_pause_runtime(monkeypatch, task)
    publish = AsyncMock()
    monkeypatch.setattr(commands, "publish_task_event", publish)
    reply = AsyncMock()
    message = _pause_message(task)

    with pytest.raises(
        ClientVisibleTaskCommandDeferred, match="waiting for the active task lease"
    ):
        await commands.pause_task(reply, int(task.id), message)

    # The zombie's run is still interrupted: its settlement is attempt-fenced.
    pause_execution.assert_awaited_once_with()
    # The user's intent is retried rather than answered or reported.
    assert not execution._is_task_pause_accepted(int(task.id))
    assert "_durable_command_error" not in message
    reply.assert_not_awaited()
    publish.assert_not_awaited()
    stored = _row(db_session, int(task.id))
    assert stored.control_state == TaskControlState.RUNNING.value
    assert stored.state_version == 3
    assert stored.runner_id == SUCCESSOR_RUNNER
    assert stored.lease_attempt_id == SUCCESSOR_ATTEMPT


@pytest.mark.asyncio
async def test_zombie_pause_defers_through_the_durable_dispatcher(
    live_task_lease, db_session, monkeypatch
) -> None:
    """The deferral is the retryable transport outcome, not a rejection."""
    task = _leased_task(db_session)
    live_task_lease(db_session, task)
    # Same runner, new acquisition: the foreign-runner pre-check cannot see
    # this, only the attempt fence can.
    with get_session_local()() as db:
        db.execute(
            update(Task)
            .where(Task.id == int(task.id))
            .values(lease_attempt_id=SUCCESSOR_ATTEMPT)
        )
        db.commit()
    _patch_pause_runtime(monkeypatch, task)
    monkeypatch.setattr(
        commands, "_load_command_actor", lambda _: SimpleNamespace(id=1, is_admin=False)
    )
    command = ClaimedTaskCommand(
        id=1,
        task_id=int(task.id),
        actor_user_id=int(task.user_id),
        command_id="pause-zombie",
        kind=TaskCommandKind.PAUSE,
        payload={},
        target_run_id=str(task.run_id),
        attempt_count=1,
    )

    with pytest.raises(ClientVisibleTaskCommandDeferred):
        await commands._execute_durable_task_command(command)

    assert _row(db_session, int(task.id)).control_state == "running"


def test_pause_fence_refuses_an_expired_own_lease(db_session) -> None:
    task = _leased_task(db_session)
    task.lease_attempt_id = "own-attempt"
    task.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    db_session.commit()
    own = TaskLease(
        task_id=int(task.id),
        runner_id=str(task.runner_id),
        run_id=str(task.run_id),
        attempt_id="own-attempt",
    )

    with pytest.raises(ClientVisibleTaskCommandDeferred):
        commands._apply_pause_requested_isolated(
            int(task.id), expected_run_id=str(task.run_id), owner_leases=(own,)
        )

    assert _row(db_session, int(task.id)).control_state == "running"


def test_pause_fence_without_a_local_holder_defers(db_session) -> None:
    task = _leased_task(db_session)
    task.lease_attempt_id = "someone-else"
    db_session.commit()

    with pytest.raises(ClientVisibleTaskCommandDeferred):
        commands._apply_pause_requested_isolated(
            int(task.id), expected_run_id=str(task.run_id), owner_leases=()
        )

    assert _row(db_session, int(task.id)).state_version == 3


@pytest.mark.parametrize(
    ("status", "control_state", "outcome"),
    [
        # The interrupted run paused itself: the pause took effect.
        (
            TaskStatus.PAUSED,
            TaskControlState.PAUSED,
            commands.PauseWriteOutcome.RUN_ALREADY_PAUSED,
        ),
        (
            TaskStatus.COMPLETED,
            TaskControlState.COMPLETED,
            commands.PauseWriteOutcome.NOT_APPLIED,
        ),
    ],
    ids=["paused", "completed"],
)
def test_pause_fence_does_not_defer_a_settled_run(
    db_session, status, control_state, outcome
) -> None:
    """A settled run is not another live owner, so the pause is not deferred."""

    task = _leased_task(db_session)
    task.lease_attempt_id = "own-attempt"
    task.status = status
    task.control_state = control_state.value
    db_session.commit()

    result = commands._apply_pause_requested_isolated(
        int(task.id),
        expected_run_id=str(task.run_id),
        owner_leases=(
            TaskLease(int(task.id), str(task.runner_id), str(task.run_id), "x"),
        ),
    )

    assert result is outcome
    assert _row(db_session, int(task.id)).state_version == 3


@pytest.mark.parametrize(
    ("offset", "held"),
    [
        (timedelta(seconds=1), True),
        (timedelta(0), True),
        (-timedelta(seconds=1), False),
    ],
    ids=["before-expiry", "at-expiry", "after-expiry"],
)
def test_holder_fence_treats_the_expiry_instant_as_live(
    db_session, offset: timedelta, held: bool
) -> None:
    """Takeover and the live-owner fence expire a lease only at ``< now``;
    the holder fence must agree, or a row at exactly ``lease_expires_at``
    would be neither takeable nor pausable by its holder."""
    expires_at = datetime(2030, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    task = _leased_task(db_session)
    task.lease_attempt_id = "own-attempt"
    task.lease_expires_at = expires_at
    db_session.commit()
    own = TaskLease(int(task.id), str(task.runner_id), str(task.run_id), "own-attempt")

    with get_session_local()() as db:
        matched = db.scalar(
            select(Task.id).where(
                Task.id == int(task.id),
                leases.task_lease_holder_predicate((own,), now=expires_at - offset),
            )
        )

    assert (matched is not None) is held


class _FakeCoordinator:
    def __init__(self, lease: CoordinatorLease) -> None:
        self.lease = lease
        self.task_id = lease.task_id


@pytest.mark.asyncio
@pytest.mark.parametrize("taken_over", [False, True])
async def test_shared_mode_pause_is_fenced_on_the_coordinator_lease(
    db_session, monkeypatch, taken_over: bool
) -> None:
    task = _leased_task(db_session)
    task.lease_attempt_id = "coordinator-attempt"
    db_session.commit()
    coordinator = _FakeCoordinator(
        CoordinatorLease(int(task.id), get_runner_id(), "coordinator-attempt")
    )
    monkeypatch.setattr(
        task_coordinator_runtime,
        "current_task_coordinator",
        lambda task_id: coordinator if task_id == coordinator.task_id else None,
    )
    if taken_over:
        _take_over(int(task.id))
    _patch_pause_runtime(monkeypatch, task)
    monkeypatch.setattr(commands, "publish_task_event", AsyncMock())

    try:
        if taken_over:
            with pytest.raises(ClientVisibleTaskCommandDeferred):
                await commands.pause_task(
                    AsyncMock(), int(task.id), _pause_message(task)
                )
        else:
            await commands.pause_task(AsyncMock(), int(task.id), _pause_message(task))
    finally:
        execution._clear_task_pause_accepted(int(task.id))

    assert _row(db_session, int(task.id)).control_state == (
        TaskControlState.RUNNING.value
        if taken_over
        else TaskControlState.PAUSE_REQUESTED.value
    )


@pytest.mark.asyncio
async def test_local_holders_follow_registration_and_run(
    live_task_lease, db_session
) -> None:
    task = _leased_task(db_session)
    lease = live_task_lease(db_session, task)

    assert leases.local_task_lease_holders(int(task.id), str(task.run_id)) == (lease,)
    assert leases.local_task_lease_holders(int(task.id), "other-run") == ()
    assert leases.local_task_lease_holders(int(task.id) + 1, str(task.run_id)) == ()

    # A registration whose heartbeat already ended no longer holds the run.
    manager = leases._get_task_lease_heartbeat_manager()
    entry = manager._entries[leases._task_lease_key(lease)]
    entry.terminal_event.set()
    try:
        assert leases.local_task_lease_holders(int(task.id), str(task.run_id)) == ()
    finally:
        entry.terminal_event.clear()


def _paused_row(db_session) -> Task:
    task = _leased_task(db_session)
    task.status = TaskStatus.PAUSED
    task.control_state = TaskControlState.PAUSED.value
    task.runner_id = None
    task.lease_expires_at = None
    db_session.commit()
    return task


def _patch_paused_runtime(monkeypatch, task: Task) -> AsyncMock:
    snapshot_task = SimpleNamespace(
        user_id=int(task.user_id), run_id=str(task.run_id), status=TaskStatus.PAUSED
    )
    monkeypatch.setattr(
        task_setup_snapshot,
        "load_task_setup_snapshot_sync",
        lambda *a, **k: SimpleNamespace(task=snapshot_task, runtime_user=object()),
    )
    monkeypatch.setattr(commands, "resolve_execution_scope_off_turn", lambda *a: None)
    service = SimpleNamespace(pause_execution=AsyncMock(return_value=False))
    monkeypatch.setattr(
        agent_service_manager,
        "get_agent_manager",
        lambda: SimpleNamespace(get_agent_for_task=AsyncMock(return_value=service)),
    )
    publish = AsyncMock()
    monkeypatch.setattr(commands, "publish_task_event", publish)
    return publish


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("attempt_count", "same_run", "applied"),
    [(2, True, True), (1, True, False), (2, False, False)],
    ids=["retry-same-run", "first-attempt", "retry-other-run"],
)
async def test_pause_retry_that_finds_its_run_paused_settles_as_applied(
    db_session, monkeypatch, attempt_count: int, same_run: bool, applied: bool
) -> None:
    task = _paused_row(db_session)
    publish = _patch_paused_runtime(monkeypatch, task)
    reply = AsyncMock()
    message = {
        **_pause_message(task),
        "_durable_attempt_count": attempt_count,
        "_durable_target_run_id": str(task.run_id) if same_run else "earlier-run",
    }

    await commands.pause_task(reply, int(task.id), message)

    if applied:
        # The run's own settlement already announced PAUSED.
        assert "_durable_command_error" not in message
        reply.assert_not_awaited()
        publish.assert_not_awaited()
        assert not execution._is_task_pause_accepted(int(task.id))
    else:
        assert message["_durable_command_error"] == "Task is already paused"
        reply.assert_awaited_once()


# ---------------------------------------------------------------------------
# RESUME_REQUESTED (live message into a running run)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("row", "own", "allowed"),
    [
        ("own", True, True),
        ("successor-live", True, False),
        ("successor-live", False, False),
        ("owner-free", True, True),
        ("owner-free", False, True),
        ("successor-expired", False, True),
    ],
)
def test_live_owner_fence_refuses_only_another_live_acquisition(
    db_session, row: str, own: bool, allowed: bool
) -> None:
    task = _leased_task(db_session)
    task.lease_attempt_id = "own-attempt"
    db_session.commit()
    own_lease = TaskLease(
        int(task.id), str(task.runner_id), str(task.run_id), "own-attempt"
    )
    if row == "successor-live":
        _take_over(int(task.id))
    elif row == "owner-free":
        # A non-shared release keeps the run and clears the owner.
        task.runner_id = None
        task.lease_attempt_id = None
        task.lease_expires_at = None
        db_session.commit()
    elif row == "successor-expired":
        _take_over(int(task.id))
        with get_session_local()() as db:
            db.execute(
                update(Task)
                .where(Task.id == int(task.id))
                .values(
                    lease_expires_at=datetime.now(timezone.utc) - timedelta(seconds=1)
                )
            )
            db.commit()

    def transition():
        return transition_task_control_state_sync(
            int(task.id),
            TaskControlState.RESUME_REQUESTED,
            expected_run_id=str(task.run_id),
            fence_live_owner=True,
            owner_lease=own_lease if own else None,
        )

    if allowed:
        assert transition().control_state is TaskControlState.RESUME_REQUESTED
    else:
        with pytest.raises(StaleTaskRunError, match="another live lease"):
            transition()
        stored = _row(db_session, int(task.id))
        assert stored.control_state == TaskControlState.RUNNING.value
        assert stored.state_version == 3


@pytest.mark.asyncio
async def test_live_message_handoff_stays_off_a_successor_run(
    live_task_lease, db_session
) -> None:
    task = _leased_task(db_session)
    live_task_lease(db_session, task)
    owner = db_session.get(User, int(task.user_id))
    background_manager = execution.BackgroundTaskManager()
    with _live_control_environment(background_manager=background_manager) as (
        agent,
        _,
    ):

        async def inject_then_lose_lease(*args, **kwargs):
            # The takeover lands after routing resolved this process's lease
            # and after the injection, right before the handoff write.
            _take_over(int(task.id))
            return UserMessageInjectionOutcome.POSTED_FRESH

        agent.post_user_message = AsyncMock(side_effect=inject_then_lose_lease)

        # The refused handoff takes the same path as a rotated run: the
        # accepted injection keeps the delivery unresolved, so the command is
        # retried through recovery rather than reported delivered.
        with pytest.raises(TaskCommandDeferred, match="waiting for runtime injection"):
            await commands.execute_durable_task_command(
                _message_command(task, owner, "fenced-handoff")
            )
        agent.post_user_message.assert_awaited_once()
        # The failed handoff hands the reservation back.
        assert background_manager.resume_holder_age_seconds(int(task.id)) is None

    stored = _row(db_session, int(task.id))
    assert stored.control_state == TaskControlState.RUNNING.value
    assert stored.runner_id == SUCCESSOR_RUNNER
    assert stored.lease_attempt_id == SUCCESSOR_ATTEMPT


@pytest.mark.asyncio
async def test_live_message_handoff_follows_a_run_that_paused_itself(
    live_task_lease, db_session
) -> None:
    """The message's own interrupt can settle the local run first."""
    task = _leased_task(db_session)
    lease = live_task_lease(db_session, task)
    owner = db_session.get(User, int(task.user_id))
    background_manager = execution.BackgroundTaskManager()
    with _live_control_environment(background_manager=background_manager) as (
        agent,
        _,
    ):

        async def inject_then_settle_paused(*args, **kwargs):
            # The interrupt requested by this message pauses the run, which
            # releases its lease the non-shared way: owner cleared, run kept.
            with get_session_local()() as db:
                assert leases.release_task_lease_no_commit(
                    db, lease, status=TaskStatus.PAUSED
                )
                db.commit()
            return UserMessageInjectionOutcome.POSTED_FRESH

        agent.post_user_message = AsyncMock(side_effect=inject_then_settle_paused)

        result = await commands.execute_durable_task_command(
            _message_command(task, owner, "self-paused-handoff")
        )
        assert result["kind"] == "message"
        # The resume carries the message instead of failing it.
        execution.execute_resume_background.assert_called_once()
        await background_manager.wait_for_previous(int(task.id))

    stored = _row(db_session, int(task.id))
    assert stored.status == TaskStatus.PAUSED
    assert stored.control_state == TaskControlState.RESUME_REQUESTED.value
    assert stored.runner_id is None


@pytest.mark.asyncio
async def test_message_defers_while_a_successor_holds_the_run(
    live_task_lease, db_session
) -> None:
    """A takeover before routing leaves the message to the live owner."""
    task = _leased_task(db_session)
    live_task_lease(db_session, task)
    _take_over(int(task.id))
    owner = db_session.get(User, int(task.user_id))
    with _live_control_environment(
        background_manager=execution.BackgroundTaskManager()
    ) as (agent, _):
        with pytest.raises(
            ClientVisibleTaskCommandDeferred, match="waiting for the active task lease"
        ):
            await commands.execute_durable_task_command(
                _message_command(task, owner, "successor-message")
            )
        agent.post_user_message.assert_not_awaited()

    stored = _row(db_session, int(task.id))
    assert stored.control_state == TaskControlState.RUNNING.value
    assert stored.state_version == 3
    assert stored.lease_attempt_id == SUCCESSOR_ATTEMPT


def _holderless_task(db_session, attempt: str) -> Task:
    """This runner owns the RUNNING row live, but no heartbeat here holds it."""
    task = _leased_task(db_session)
    task.lease_attempt_id = attempt
    db_session.commit()
    return task


def _task_failure_broadcasts(publish: AsyncMock) -> list[dict]:
    return [
        call.args[0]
        for call in publish.await_args_list
        if call.args[0].get("error_code") == ClientErrorCode.TASK_EXECUTION_FAILED.value
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "attempt",
    # Its own attempt between heartbeat stop and settlement (or before its
    # heartbeat registered), and a newer attempt of this same runner.
    ["own-unregistered-attempt", SUCCESSOR_ATTEMPT],
)
async def test_message_to_a_self_owned_run_without_a_holder_defers(
    db_session, monkeypatch, attempt: str
) -> None:
    task = _holderless_task(db_session, attempt)
    owner = db_session.get(User, int(task.user_id))
    publish = AsyncMock()
    monkeypatch.setattr(commands, "publish_task_event", publish)
    with _live_control_environment(
        background_manager=execution.BackgroundTaskManager()
    ) as (agent, _):
        with pytest.raises(
            ClientVisibleTaskCommandDeferred, match="waiting for the active task lease"
        ) as deferred:
            await commands.execute_durable_task_command(
                _message_command(task, owner, "holderless-message")
            )
        agent.post_user_message.assert_not_awaited()
        execution.execute_resume_background.assert_not_called()

    # Deferred before any delivery claim: a budget exhausted here can safely
    # tell the sender to resend.
    assert deferred.value.resend_safe is True
    assert (
        commands._load_command_message_delivery_status(
            int(task.id), "holderless-message"
        )
        is None
    )
    assert _task_failure_broadcasts(publish) == []
    stored = _row(db_session, int(task.id))
    assert stored.control_state == TaskControlState.RUNNING.value
    assert stored.state_version == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("attempt_count", "defer_count", "resend_safe"),
    [
        (1, 0, True),
        (3, 2, True),
        # An attempt that ended some other way may still inject.
        (2, 0, False),
    ],
)
async def test_message_pre_check_deferral_is_resend_safe_only_when_never_delivered(
    db_session, attempt_count, defer_count, resend_safe
) -> None:
    # A claimed delivery row skips this pre-check altogether (see the
    # recovered handoff test below), so only the counters decide here.
    task = _holderless_task(db_session, "own-unregistered-attempt")
    owner = db_session.get(User, int(task.user_id))
    command = replace(
        _message_command(task, owner, TURN_ID, attempt_count=attempt_count),
        defer_count=defer_count,
    )

    with pytest.raises(ClientVisibleTaskCommandDeferred) as deferred:
        await commands._execute_durable_task_command(command)

    assert deferred.value.resend_safe is resend_safe


@pytest.mark.asyncio
@pytest.mark.parametrize("recovered", [False, True], ids=["fresh", "recovered"])
async def test_handoff_refused_after_routing_defers_or_settles_unknown(
    db_session, monkeypatch, recovered: bool
) -> None:
    """The race the pre-check cannot close: the refusal lands at the handoff."""
    task = _holderless_task(db_session, "own-unregistered-attempt")
    owner = db_session.get(User, int(task.user_id))
    if recovered:
        _pending_row(db_session, task, int(owner.id))
    # The pre-check read happened before this runner's attempt took the row.
    monkeypatch.setattr(commands, "_load_live_self_owned_attempt", lambda _: None)
    publish = AsyncMock()
    monkeypatch.setattr(commands, "publish_task_event", publish)
    background_manager = execution.BackgroundTaskManager()
    command = (
        _message_command(task, owner, TURN_ID, attempt_count=2)
        if recovered
        else _message_command(task, owner, TURN_ID)
    )
    with _live_control_environment(background_manager=background_manager) as (
        agent,
        _,
    ):
        if recovered:
            # An earlier attempt may have applied it: never resendable.
            result = await commands.execute_durable_task_command(command)
            assert result["delivery_outcome"] == DELIVERY_OUTCOME_UNKNOWN
        else:
            with pytest.raises(
                TaskCommandDeferred, match="waiting for the active task lease"
            ) as deferred:
                await commands.execute_durable_task_command(command)
            # The claimed delivery row survives for the retry to recover.
            assert deferred.value.resend_safe is False
        agent.post_user_message.assert_not_awaited()
        execution.execute_resume_background.assert_not_called()
        assert background_manager.resume_holder_age_seconds(int(task.id)) is None

    # A recovered claim settles like the recovered-claim status refusal:
    # ``dispatched`` ("do not resend") next to the command's recorded
    # outcome-unknown result.
    assert _row_status(db_session, int(task.id)) == (
        DELIVERY_DISPATCHED if recovered else DELIVERY_PENDING
    )
    # A healthy running task is never announced as failed.
    assert _task_failure_broadcasts(publish) == []
    stored = _row(db_session, int(task.id))
    assert stored.control_state == TaskControlState.RUNNING.value
    assert stored.state_version == 3


def _waiting_row(db_session) -> Task:
    """A resting row: waiting for the user, owner-free, run kept."""
    task = _leased_task(db_session)
    task.status = TaskStatus.WAITING_FOR_USER
    task.control_state = TaskControlState.WAITING_FOR_USER.value
    task.runner_id = None
    task.lease_attempt_id = None
    task.lease_expires_at = None
    db_session.commit()
    return task


def _reply_prelease(task_id: int) -> None:
    """What the HTTP reply / A2A resume prelease leaves: RUNNING, same run,
    another live acquisition, and only ``state_version`` bumped."""
    with get_session_local()() as db:
        db.execute(
            update(Task)
            .where(Task.id == task_id)
            .values(
                status=TaskStatus.RUNNING,
                control_state=TaskControlState.RUNNING.value,
                runner_id=SUCCESSOR_RUNNER,
                lease_attempt_id=SUCCESSOR_ATTEMPT,
                lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
                state_version=Task.state_version + 1,
            )
        )
        db.commit()


@pytest.mark.asyncio
@pytest.mark.parametrize("recovered", [False, True], ids=["fresh", "recovered"])
async def test_handoff_to_a_resting_row_taken_after_routing_is_refused(
    db_session, monkeypatch, recovered: bool
) -> None:
    """A prelease bypassing the durable queue starts the same run between
    this message's routing snapshot and its handoff; the run fence alone
    would land RESUME_REQUESTED on that run."""
    task = _waiting_row(db_session)
    owner = db_session.get(User, int(task.user_id))
    if recovered:
        _pending_row(db_session, task, int(owner.id))
    # The admission check runs after routing and right before the handoff.
    monkeypatch.setattr(
        commands,
        "require_execution_admission_isolated",
        lambda task_id: _reply_prelease(task_id),
    )
    publish = AsyncMock()
    monkeypatch.setattr(commands, "publish_task_event", publish)
    background_manager = execution.BackgroundTaskManager()
    command = _message_command(
        task, owner, TURN_ID, attempt_count=2 if recovered else 1
    )
    with _live_control_environment(background_manager=background_manager) as (
        agent,
        _,
    ):
        if recovered:
            result = await commands.execute_durable_task_command(command)
            assert result["delivery_outcome"] == DELIVERY_OUTCOME_UNKNOWN
        else:
            with pytest.raises(
                TaskCommandDeferred, match="waiting for the active task lease"
            ) as deferred:
                await commands.execute_durable_task_command(command)
            assert deferred.value.resend_safe is False
        agent.post_user_message.assert_not_awaited()
        execution.execute_resume_background.assert_not_called()
        assert background_manager.resume_holder_age_seconds(int(task.id)) is None

    assert _row_status(db_session, int(task.id)) == (
        DELIVERY_DISPATCHED if recovered else DELIVERY_PENDING
    )
    assert _task_failure_broadcasts(publish) == []
    # The prelease's run is untouched: no RESUME_REQUESTED stamped onto it.
    stored = _row(db_session, int(task.id))
    assert stored.status == TaskStatus.RUNNING
    assert stored.control_state == TaskControlState.RUNNING.value
    assert stored.state_version == 4
    assert stored.runner_id == SUCCESSOR_RUNNER
    assert stored.lease_attempt_id == SUCCESSOR_ATTEMPT


@pytest.mark.asyncio
async def test_handoff_to_an_unmoved_resting_row_still_lands(
    db_session, monkeypatch
) -> None:
    """The version fence does not refuse a row nobody else touched."""
    task = _waiting_row(db_session)
    owner = db_session.get(User, int(task.user_id))
    monkeypatch.setattr(
        commands, "require_execution_admission_isolated", lambda task_id: None
    )
    background_manager = execution.BackgroundTaskManager()
    with _live_control_environment(background_manager=background_manager) as (
        agent,
        _,
    ):
        # The deferred message waits for the (stubbed) resume owner to
        # dispatch it; what matters here is that the handoff was made.
        with pytest.raises(TaskCommandDeferred, match="waiting for runtime injection"):
            await commands.execute_durable_task_command(
                _message_command(task, owner, TURN_ID)
            )
        execution.execute_resume_background.assert_called_once()
        await background_manager.wait_for_previous(int(task.id))

    stored = _row(db_session, int(task.id))
    assert stored.control_state == TaskControlState.RESUME_REQUESTED.value
    assert stored.state_version == 4


@pytest.mark.asyncio
async def test_runtime_error_on_recovered_claim_records_outcome_unknown(
    db_session,
) -> None:
    owner = _user(db_session, "runtime-error-owner")
    task = _live_task(db_session, int(owner.id))
    _pending_row(db_session, task, int(owner.id))

    with (
        _live_control_environment(outcome=ResumeReservationOutcome.RESERVED),
        patch(
            "xagent.web.services.agent_service_manager.get_agent_manager",
            side_effect=RuntimeError("agent manager exploded"),
        ),
    ):
        await commands.execute_durable_task_command(
            _message_command(task, owner, TURN_ID, attempt_count=2)
        )

    # Not DELIVERY_FAILED: a resend could duplicate an applied turn.
    assert _row_status(db_session, int(task.id)) == DELIVERY_OUTCOME_UNKNOWN


@pytest.mark.asyncio
async def test_pause_retry_through_the_dispatcher_settles_as_applied(
    db_session, monkeypatch
) -> None:
    task = _paused_row(db_session)
    _patch_paused_runtime(monkeypatch, task)
    command = ClaimedTaskCommand(
        id=1,
        task_id=int(task.id),
        actor_user_id=int(task.user_id),
        command_id="pause-retry",
        kind=TaskCommandKind.PAUSE,
        payload={},
        target_run_id=str(task.run_id),
        attempt_count=2,
        defer_count=1,
    )

    result = await commands._execute_durable_task_command(command)

    assert result == {
        "task_id": int(task.id),
        "command_id": "pause-retry",
        "kind": "pause",
    }


# ---------------------------------------------------------------------------
# Late results on a row that already settled terminal
# ---------------------------------------------------------------------------


def _owned_settled_task(db_session, status: TaskStatus) -> tuple[Task, TaskLease]:
    """Terminal while the lease fence still matches, as in coordinator context."""
    task = _leased_task(db_session)
    task.lease_attempt_id = "own-attempt"
    task.status = status
    task.control_state = (
        TaskControlState.FAILED.value
        if status == TaskStatus.FAILED
        else TaskControlState.COMPLETED.value
    )
    task.error_message = "cancelled externally" if status == TaskStatus.FAILED else None
    task.output = "first answer" if status == TaskStatus.COMPLETED else None
    db_session.commit()
    return task, TaskLease(
        int(task.id), str(task.runner_id), str(task.run_id), "own-attempt"
    )


_LATE_RESULTS = {
    "interrupted": {"status": "interrupted", "success": False, "output": "stopped"},
    "success": {"status": "completed", "success": True, "output": "late answer"},
    "failure": {"status": "failed", "success": False, "output": "late failure"},
}
_SETTLED = [TaskStatus.FAILED, TaskStatus.COMPLETED]


def _assert_settled_unchanged(stored: Task, status: TaskStatus) -> None:
    assert stored.status == status
    assert stored.control_state == (
        TaskControlState.FAILED.value
        if status == TaskStatus.FAILED
        else TaskControlState.COMPLETED.value
    )
    if status == TaskStatus.FAILED:
        assert stored.error_message == "cancelled externally"
    else:
        assert stored.output == "first answer"


@pytest.mark.parametrize("status", _SETTLED, ids=lambda s: s.value)
@pytest.mark.parametrize("kind", sorted(_LATE_RESULTS))
def test_late_result_keeps_a_settled_row(
    db_session, kind: str, status: TaskStatus
) -> None:
    task, lease = _owned_settled_task(db_session, status)

    finalized = execution._finalize_task_execution_result_isolated(
        task_id=int(task.id),
        task_user_id=int(task.user_id),
        pre_run_status=TaskStatus.RUNNING,
        result=dict(_LATE_RESULTS[kind]),
        expected_run_id=lease.run_id,
        task_lease=lease,
        resolved_scope_segments=(),
        prepared_outputs=execution._PreparedTaskFileOutputs((), (), ()),
    )

    # The broadcast carries the row's actual status, not the pre-run one.
    assert finalized.final_task_status == status.value
    assert finalized.waiting_for_control is True
    assert finalized.report.control_state == {}
    stored = _row(db_session, int(task.id))
    _assert_settled_unchanged(stored, status)
    assert stored.state_version == 3


@pytest.mark.parametrize("status", _SETTLED, ids=lambda s: s.value)
@pytest.mark.parametrize("kind", sorted(_LATE_RESULTS))
def test_late_resumed_result_keeps_a_settled_row(
    db_session, kind: str, status: TaskStatus
) -> None:
    task, lease = _owned_settled_task(db_session, status)
    result = dict(_LATE_RESULTS[kind])

    finalized = execution._finalize_resumed_task(
        int(task.id),
        status=result["status"],
        success=result["success"],
        output=result["output"],
        task_owner_user_id=int(task.user_id),
        result=result,
        task_lease=lease,
        prepared_outputs=execution._PreparedTaskFileOutputs((), (), ()),
    )

    assert finalized["late_result"] is True
    stored = _row(db_session, int(task.id))
    _assert_settled_unchanged(stored, status)
    # The resumed run's lease is still released, under the row's own status.
    assert stored.runner_id is None
    assert stored.lease_attempt_id is None
    assert stored.lease_expires_at is None
