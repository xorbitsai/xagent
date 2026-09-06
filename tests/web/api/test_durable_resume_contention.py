"""Durable RESUME admission and idempotency regressions for #1499."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xagent.web.api import websocket as websocket_api
from xagent.web.api.websocket import (
    ANY_RESUME_RUN,
    BackgroundTaskManager,
    ResumeCommandOutcome,
    ResumeReservationOutcome,
    _execute_durable_task_command,
)
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.services.task_command_transport import (
    COMMAND_PENDING,
    ClaimedTaskCommand,
    TaskCommandDeferred,
    TaskCommandKind,
    TaskCommandRejected,
    dispatch_one_task_command,
    enqueue_task_command,
)
from xagent.web.services.task_execution_controller import TaskControlState


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'durable_resume_contention.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        try:
            db.close()
        finally:
            Base.metadata.drop_all(bind=get_engine())


def _user(db, username: str) -> User:
    user = User(username=username, password_hash="x", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _task(
    db,
    owner_id: int,
    *,
    status: TaskStatus = TaskStatus.PAUSED,
    control_state: TaskControlState = TaskControlState.PAUSED,
) -> Task:
    task = Task(
        user_id=owner_id,
        title="resume contention",
        description="resume contention",
        status=status,
        control_state=control_state.value,
        run_id="run-a",
        execution_mode="balanced",
        source="sdk",
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def _command(task: Task, owner: User, command_id: str) -> ClaimedTaskCommand:
    return ClaimedTaskCommand(
        id=1,
        task_id=int(task.id),
        actor_user_id=int(owner.id),
        command_id=command_id,
        kind=TaskCommandKind.RESUME,
        payload={"type": "resume_task"},
        target_run_id="run-a",
        attempt_count=1,
    )


@contextmanager
def _resume_runtime_patches(
    *, outcome: ResumeReservationOutcome
) -> Iterator[tuple[MagicMock, MagicMock, MagicMock]]:
    """Activate the resume-handler runtime patches for one test.

    A context manager rather than a factory returning a live ``ExitStack``:
    ``enter_context`` applies each patch immediately, so a factory hands back
    an object whose patches are already in effect while its cleanup only
    arms once the caller reaches its own ``with``. Anything raised in between
    would leave ``websocket_api.manager`` patched for the rest of the session.
    """

    stack = ExitStack()
    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent_manager = MagicMock()
    agent_manager.get_agent_for_task = AsyncMock(return_value=agent)
    connection_manager = MagicMock()
    connection_manager.connections_for_task.return_value = []
    connection_manager.send_personal_message = AsyncMock()
    connection_manager.broadcast_to_task = AsyncMock()
    background_manager = MagicMock()
    background_manager.try_reserve_resume.return_value = outcome
    background_manager.resume_admission_state.return_value = (
        None if outcome is ResumeReservationOutcome.RESERVED else outcome
    )
    background_manager.running_tasks = {}

    stack.enter_context(
        patch("xagent.web.api.chat.get_agent_manager", return_value=agent_manager)
    )
    stack.enter_context(patch.object(websocket_api, "manager", connection_manager))
    stack.enter_context(
        patch.object(websocket_api, "background_task_manager", background_manager)
    )
    stack.enter_context(
        patch.object(
            websocket_api, "resolve_execution_scope_off_turn", return_value=None
        )
    )
    with stack:
        yield background_manager, connection_manager, agent_manager


@pytest.mark.asyncio
async def test_resume_reservation_classifies_all_admission_outcomes() -> None:
    manager = BackgroundTaskManager()

    assert (
        manager.try_reserve_resume(1, expected_run_id=ANY_RESUME_RUN)
        is ResumeReservationOutcome.RESERVED
    )
    assert (
        manager.try_reserve_resume(1, expected_run_id=ANY_RESUME_RUN)
        is ResumeReservationOutcome.RESERVATION_HELD
    )

    coordinator_gate = asyncio.get_running_loop().create_future()
    coordinator = asyncio.ensure_future(coordinator_gate)
    try:
        manager.register_reserved_resume(1, coordinator, run_id="run-a")
        assert (
            manager.try_reserve_resume(1, expected_run_id="run-a")
            is ResumeReservationOutcome.COORDINATOR_RUNNING
        )
        assert (
            manager.try_reserve_resume(1, expected_run_id="run-b")
            is ResumeReservationOutcome.RESERVATION_HELD
        )
        # ANY_RESUME_RUN is the only way to accept a coordinator whose run is
        # not the one being asked about.
        assert (
            manager.try_reserve_resume(1, expected_run_id=ANY_RESUME_RUN)
            is ResumeReservationOutcome.COORDINATOR_RUNNING
        )
        # An explicit ``None`` asks about a task that has no run id, and must
        # not be read as a wildcard for a coordinator that has one.
        assert (
            manager.try_reserve_resume(1, expected_run_id=None)
            is ResumeReservationOutcome.RESERVATION_HELD
        )

        manager._shutting_down = True
        assert (
            manager.try_reserve_resume(2, expected_run_id=ANY_RESUME_RUN)
            is ResumeReservationOutcome.SHUTTING_DOWN
        )
    finally:
        coordinator_gate.cancel()
        await asyncio.gather(coordinator, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    [
        ResumeReservationOutcome.RESERVATION_HELD,
        ResumeReservationOutcome.SHUTTING_DOWN,
    ],
)
async def test_uncertain_resume_admission_defers_durable_command(
    db_session, outcome: ResumeReservationOutcome
) -> None:
    owner = _user(db_session, f"owner-{outcome.value}")
    task = _task(
        db_session,
        int(owner.id),
        control_state=(
            TaskControlState.RESUME_REQUESTED
            if outcome is ResumeReservationOutcome.RESERVATION_HELD
            else TaskControlState.PAUSED
        ),
    )
    with (
        _resume_runtime_patches(outcome=outcome) as (
            _background_manager,
            connection_manager,
            _agent_manager,
        ),
        pytest.raises(TaskCommandDeferred, match="resume slot"),
    ):
        await _execute_durable_task_command(
            _command(task, owner, f"resume-{outcome.value}")
        )

    connection_manager.send_personal_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_registered_coordinator_is_an_explicit_idempotent_success(
    db_session,
) -> None:
    owner = _user(db_session, "coordinator-owner")
    task = _task(db_session, int(owner.id))
    with _resume_runtime_patches(
        outcome=ResumeReservationOutcome.COORDINATOR_RUNNING
    ) as (
        background_manager,
        connection_manager,
        _agent_manager,
    ):
        result = await _execute_durable_task_command(
            _command(task, owner, "resume-already-running")
        )

    assert result is not None
    assert result["resume_outcome"] == ResumeCommandOutcome.ALREADY_IN_PROGRESS.value
    _agent_manager.get_agent_for_task.assert_not_awaited()
    background_manager.try_reserve_resume.assert_not_called()
    background_manager.register_reserved_resume.assert_not_called()

    # Deliberately silent. A registered coordinator has not reached its lease
    # claim, so the live row still reads ``paused`` -- the RESUME_REQUESTED
    # transition writes only ``control_state`` -- and any frame built here,
    # from the snapshot or from the row, would re-confirm the state it was
    # meant to correct. The coordinator's own ``task_resumed`` broadcast is
    # the correction.
    connection_manager.send_personal_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_running_control_state_is_an_explicit_idempotent_success(
    db_session,
) -> None:
    owner = _user(db_session, "running-owner")
    task = _task(
        db_session,
        int(owner.id),
        status=TaskStatus.RUNNING,
        control_state=TaskControlState.RUNNING,
    )
    task.runner_id = "runner-a"
    task.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=1)
    db_session.commit()
    with _resume_runtime_patches(outcome=ResumeReservationOutcome.RESERVED) as (
        background_manager,
        _connection_manager,
        _agent_manager,
    ):
        result = await _execute_durable_task_command(
            _command(task, owner, "resume-running")
        )

    assert result is not None
    assert result["resume_outcome"] == ResumeCommandOutcome.ALREADY_IN_PROGRESS.value
    background_manager.try_reserve_resume.assert_not_called()


@pytest.mark.asyncio
async def test_running_without_a_live_lease_defers_for_recovery(db_session) -> None:
    owner = _user(db_session, "abandoned-running-owner")
    task = _task(
        db_session,
        int(owner.id),
        status=TaskStatus.RUNNING,
        control_state=TaskControlState.RUNNING,
    )
    with (
        _resume_runtime_patches(outcome=ResumeReservationOutcome.RESERVED) as (
            background_manager,
            _connection_manager,
            _agent_manager,
        ),
        pytest.raises(TaskCommandDeferred, match="lease recovery"),
    ):
        await _execute_durable_task_command(
            _command(task, owner, "resume-abandoned-running")
        )

    background_manager.try_reserve_resume.assert_not_called()


@pytest.mark.asyncio
async def test_pending_pause_defers_resume_until_control_state_settles(
    db_session,
) -> None:
    owner = _user(db_session, "pause-requested-owner")
    task = _task(
        db_session,
        int(owner.id),
        status=TaskStatus.RUNNING,
        control_state=TaskControlState.PAUSE_REQUESTED,
    )
    with (
        _resume_runtime_patches(outcome=ResumeReservationOutcome.RESERVED) as (
            background_manager,
            _connection_manager,
            _agent_manager,
        ),
        pytest.raises(TaskCommandDeferred, match="pending pause"),
    ):
        await _execute_durable_task_command(
            _command(task, owner, "resume-during-pause")
        )

    background_manager.try_reserve_resume.assert_not_called()


@pytest.mark.asyncio
async def test_contended_resume_dispatch_stays_pending_instead_of_completed(
    db_session,
) -> None:
    owner = _user(db_session, "dispatcher-owner")
    task = _task(db_session, int(owner.id))
    enqueued = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(owner.id),
        command_id="resume-dispatch-contention",
        kind=TaskCommandKind.RESUME,
        payload={"type": "resume_task"},
    )
    with _resume_runtime_patches(outcome=ResumeReservationOutcome.RESERVATION_HELD) as (
        _background_manager,
        _connection_manager,
        _agent_manager,
    ):
        assert await dispatch_one_task_command(
            _execute_durable_task_command,
            command_db_id=enqueued.command_id,
        )

    db_session.expire_all()
    stored = db_session.get(TaskExecutionCommand, enqueued.command_id)
    assert stored is not None
    assert stored.status == COMMAND_PENDING
    assert stored.defer_count == 1
    assert stored.result is None


@pytest.mark.asyncio
async def test_reserved_resume_records_scheduled_result(db_session) -> None:
    owner = _user(db_session, "scheduled-owner")
    task = _task(
        db_session,
        int(owner.id),
        control_state=TaskControlState.RESUME_REQUESTED,
    )
    transition = AsyncMock(
        return_value=SimpleNamespace(run_id="run-a", status=TaskStatus.PAUSED)
    )
    resume_started = asyncio.Event()

    async def execute_resume_background(**_kwargs) -> None:
        resume_started.set()

    with (
        _resume_runtime_patches(outcome=ResumeReservationOutcome.RESERVED) as (
            background_manager,
            _connection_manager,
            _agent_manager,
        ),
        patch.object(
            websocket_api.task_execution_controller,
            "transition",
            new=transition,
        ),
        patch.object(
            websocket_api,
            "execute_resume_background",
            side_effect=execute_resume_background,
        ),
    ):
        result = await _execute_durable_task_command(
            _command(task, owner, "resume-scheduled")
        )
        await asyncio.wait_for(resume_started.wait(), timeout=1)

    assert result is not None
    assert result["resume_outcome"] == ResumeCommandOutcome.SCHEDULED.value
    background_manager.register_reserved_resume.assert_called_once()


@pytest.mark.asyncio
async def test_held_reservation_can_schedule_after_the_holder_releases(
    db_session,
) -> None:
    owner = _user(db_session, "released-holder-owner")
    task = _task(db_session, int(owner.id))
    transition = AsyncMock(
        return_value=SimpleNamespace(run_id="run-a", status=TaskStatus.PAUSED)
    )
    resume_started = asyncio.Event()

    async def execute_resume_background(**_kwargs) -> None:
        resume_started.set()

    with (
        _resume_runtime_patches(outcome=ResumeReservationOutcome.RESERVATION_HELD) as (
            background_manager,
            _connection_manager,
            _agent_manager,
        ),
        patch.object(
            websocket_api.task_execution_controller,
            "transition",
            new=transition,
        ),
        patch.object(
            websocket_api,
            "execute_resume_background",
            side_effect=execute_resume_background,
        ),
    ):
        background_manager.resume_admission_state.side_effect = [
            ResumeReservationOutcome.RESERVATION_HELD,
            None,
        ]
        background_manager.try_reserve_resume.return_value = (
            ResumeReservationOutcome.RESERVED
        )
        with pytest.raises(TaskCommandDeferred):
            await _execute_durable_task_command(
                _command(task, owner, "resume-held-first-attempt")
            )
        result = await _execute_durable_task_command(
            _command(task, owner, "resume-held-second-attempt")
        )
        await asyncio.wait_for(resume_started.wait(), timeout=1)

    assert result is not None
    assert result["resume_outcome"] == ResumeCommandOutcome.SCHEDULED.value
    transition.assert_awaited_once()
    background_manager.register_reserved_resume.assert_called_once()


@pytest.mark.asyncio
async def test_settling_turn_with_a_foreign_live_lease_defers_resume(
    db_session,
    caplog,
) -> None:
    """A resume must not schedule into another process's unreleased lease.

    A turn that ends in WAITING_FOR_USER commits its status before its lease
    columns are cleared -- the finalizer writes the status, and the lease is
    only released later by ``finish_turn``. Scheduling inside that window
    steals a lease the previous owner still holds, and its ownership-fenced
    settlement then matches no row and skips delivery reconciliation, which
    strands that turn's delivery row at ``pending`` for good.
    """

    owner = _user(db_session, "settling-owner")
    task = _task(
        db_session,
        int(owner.id),
        status=TaskStatus.WAITING_FOR_USER,
        control_state=TaskControlState.WAITING_FOR_USER,
    )
    task.runner_id = "another-process"
    task.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=1)
    db_session.commit()

    with (
        caplog.at_level(logging.INFO, logger="xagent.web.api.websocket"),
        _resume_runtime_patches(outcome=ResumeReservationOutcome.RESERVED) as (
            _background_manager,
            _connection_manager,
            agent_manager,
        ),
    ):
        # The wording has to survive the redaction chokepoint, exactly as it
        # does on the PAUSE and CANCEL arms of the shared guard: a terminal
        # deferral broadcast reduced to the generic string is indistinguishable
        # from an outright failure.
        with pytest.raises(
            websocket_api.ClientVisibleTaskCommandDeferred,
            match="active task lease owner",
        ):
            await _execute_durable_task_command(
                _command(task, owner, "resume-into-settling-turn")
            )

    # Deferral has to happen before the expensive scheduling prerequisites.
    agent_manager.get_agent_for_task.assert_not_awaited()

    # Deferrals are silent to the client, so this log line is the only trace a
    # stuck queue leaves outside the command table.
    assert any(
        "another process still holds a live task lease" in record.getMessage()
        and str(task.id) in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_expired_foreign_lease_does_not_defer_a_settled_resume(
    db_session,
) -> None:
    """The deferral is bounded: an expired lease is not a live owner.

    A lease on a non-RUNNING row can never be refreshed, so the window the
    test above protects closes on its own within the lease TTL.
    """

    owner = _user(db_session, "settled-owner")
    task = _task(
        db_session,
        int(owner.id),
        status=TaskStatus.WAITING_FOR_USER,
        control_state=TaskControlState.WAITING_FOR_USER,
    )
    task.runner_id = "another-process"
    task.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    db_session.commit()

    transition = AsyncMock(
        return_value=SimpleNamespace(run_id="run-a", status=TaskStatus.WAITING_FOR_USER)
    )
    resume_started = asyncio.Event()

    async def execute_resume_background(**_kwargs) -> None:
        resume_started.set()

    with (
        _resume_runtime_patches(outcome=ResumeReservationOutcome.RESERVED) as (
            _background_manager,
            _connection_manager,
            _agent_manager,
        ),
        patch.object(
            websocket_api.task_execution_controller,
            "transition",
            new=transition,
        ),
        patch.object(
            websocket_api,
            "execute_resume_background",
            side_effect=execute_resume_background,
        ),
    ):
        result = await _execute_durable_task_command(
            _command(task, owner, "resume-after-lease-expiry")
        )
        await asyncio.wait_for(resume_started.wait(), timeout=1)

    assert result is not None
    assert result["resume_outcome"] == ResumeCommandOutcome.SCHEDULED.value


@pytest.mark.asyncio
async def test_idempotent_resume_still_corrects_a_stale_client(db_session) -> None:
    """An idempotent success is durable-only; the client still needs telling.

    The resume control renders only while the client believes the task is
    paused, so a command that completes silently leaves a stale client
    clicking resume forever. The frame carries the task's state tuple, which
    is what the chat client resyncs its status from.
    """

    owner = _user(db_session, "stale-client-owner")
    task = _task(
        db_session,
        int(owner.id),
        status=TaskStatus.RUNNING,
        control_state=TaskControlState.RUNNING,
    )
    task.runner_id = "another-process"
    task.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=1)
    task.state_version = 11
    db_session.commit()

    with _resume_runtime_patches(outcome=ResumeReservationOutcome.RESERVED) as (
        _background_manager,
        connection_manager,
        _agent_manager,
    ):
        result = await _execute_durable_task_command(
            _command(task, owner, "resume-already-running")
        )

    assert result is not None
    assert result["resume_outcome"] == ResumeCommandOutcome.ALREADY_IN_PROGRESS.value

    frames = [
        call.args[0]
        for call in connection_manager.send_personal_message.await_args_list
        if call.args and call.args[0].get("type") == "task_resumed"
    ]
    assert len(frames) == 1
    # No state tuple by design: the transport attaches the live row, and the
    # handler's own setup snapshot would re-confirm a stale one. The
    # attachment itself is pinned by the enricher test below.
    assert frames[0]["task"] == {"id": int(task.id)}


@pytest.mark.asyncio
async def test_idempotent_resume_frame_is_enriched_with_the_live_row(
    db_session,
) -> None:
    """The correction frame gets its state from the DB, not from the handler.

    ``send_personal_message`` runs ``_with_current_task_control_state``, which
    attaches the current row only when the producer supplied no tuple. This
    pins the half the mocked connection manager above cannot exercise.
    """

    owner = _user(db_session, "enricher-owner")
    task = _task(
        db_session,
        int(owner.id),
        status=TaskStatus.RUNNING,
        control_state=TaskControlState.RUNNING,
    )
    task.state_version = 9
    db_session.commit()

    enriched = await websocket_api._with_current_task_control_state(
        {
            "type": "task_resumed",
            "message": "Task is already running.",
            "task": {"id": int(task.id)},
        }
    )

    assert enriched["task"] == {
        "id": int(task.id),
        "run_id": "run-a",
        "state_version": 9,
        "control_state": TaskControlState.RUNNING.value,
        "status": TaskStatus.RUNNING.value,
    }
    # The frontend reads ``task.status`` off an ``error`` frame to resync.
    assert enriched["status"] == TaskStatus.RUNNING.value


@pytest.mark.asyncio
async def test_two_concurrent_durable_resumes_schedule_one_execution(
    db_session,
) -> None:
    """End-to-end contention on the real manager, not a stubbed outcome.

    Two durable RESUME commands for the same run are executed concurrently
    against a real ``BackgroundTaskManager``. Exactly one may schedule an
    execution; the other must reach a durable outcome that does not claim a
    second resume happened.
    """

    owner = _user(db_session, "concurrent-owner")
    task = _task(db_session, int(owner.id))

    real_manager = BackgroundTaskManager()
    scheduled = 0
    first_resume_entered = asyncio.Event()
    release_first_resume = asyncio.Event()

    async def transition(*_args, **_kwargs):
        # Yield inside the window between reserving the slot and registering
        # the coordinator: this is where a second command can interleave.
        first_resume_entered.set()
        await release_first_resume.wait()
        return SimpleNamespace(run_id="run-a", status=TaskStatus.PAUSED)

    async def execute_resume_background(**_kwargs) -> None:
        nonlocal scheduled
        scheduled += 1

    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent_manager = MagicMock()
    agent_manager.get_agent_for_task = AsyncMock(return_value=agent)
    connection_manager = MagicMock()
    connection_manager.connections_for_task.return_value = []
    connection_manager.send_personal_message = AsyncMock()
    connection_manager.broadcast_to_task = AsyncMock()

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=agent_manager),
        patch.object(websocket_api, "manager", connection_manager),
        patch.object(websocket_api, "background_task_manager", real_manager),
        patch.object(
            websocket_api, "resolve_execution_scope_off_turn", return_value=None
        ),
        patch.object(
            websocket_api.task_execution_controller, "transition", new=transition
        ),
        patch.object(
            websocket_api,
            "execute_resume_background",
            side_effect=execute_resume_background,
        ),
    ):
        first = asyncio.ensure_future(
            _execute_durable_task_command(_command(task, owner, "resume-race-a"))
        )
        try:
            # These are hang guards, not latency assertions: database work and
            # thread scheduling can take over a second on a loaded CI runner.
            await asyncio.wait_for(first_resume_entered.wait(), timeout=30)

            # The slot is reserved but no coordinator is registered yet, so the
            # second command sees RESERVATION_HELD -- uncertain, not satisfied.
            with pytest.raises(TaskCommandDeferred, match="resume slot"):
                await _execute_durable_task_command(
                    _command(task, owner, "resume-race-b")
                )
        finally:
            release_first_resume.set()
            result = await asyncio.wait_for(first, timeout=30)

    assert result is not None
    assert result["resume_outcome"] == ResumeCommandOutcome.SCHEDULED.value
    assert scheduled == 1


@pytest.mark.asyncio
async def test_unresumable_rejection_still_embeds_the_control_tuple(
    db_session,
) -> None:
    """The states that still reject must keep carrying their state tuple.

    Round 1 asked for a test on whichever states still produce a rejection
    with ``{run_id, state_version, control_state, status}`` embedded, since
    the client reads those fields off an ``error`` frame. A terminal task is
    one such state: it is not resumable and never will be, so the command is
    refused outright rather than deferred.
    """

    owner = _user(db_session, "unresumable-owner")
    task = _task(
        db_session,
        int(owner.id),
        status=TaskStatus.COMPLETED,
        control_state=TaskControlState.RUNNING,
    )
    task.state_version = 4
    db_session.commit()

    with _resume_runtime_patches(outcome=ResumeReservationOutcome.RESERVED) as (
        _background_manager,
        connection_manager,
        _agent_manager,
    ):
        with pytest.raises(TaskCommandRejected) as excinfo:
            await _execute_durable_task_command(
                _command(task, owner, "resume-completed-task")
            )

    # The rejection carries a stable code, not just human-readable text.
    assert excinfo.value.reason == "not_resumable"

    frames = [
        call.args[0]
        for call in connection_manager.send_personal_message.await_args_list
        if call.args and call.args[0].get("type") == "error"
    ]
    assert len(frames) == 1
    assert frames[0]["task"] == {
        "id": int(task.id),
        "run_id": "run-a",
        "state_version": 4,
        "control_state": TaskControlState.RUNNING.value,
        "status": TaskStatus.COMPLETED.value,
    }


def _free_slot_manager() -> MagicMock:
    """A manager whose resume slot is free, so the handler reaches the
    status-based branches instead of returning on a MagicMock."""

    manager = MagicMock()
    manager.resume_admission_state.return_value = None
    manager.running_tasks = {}
    return manager


class _RecordingWebSocket:
    """Minimal websocket sink that keeps what a client would actually receive."""

    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_text(self, message: str) -> None:
        self.sent.append(json.loads(message))


@pytest.mark.asyncio
async def test_running_resume_correction_reaches_the_client_enriched(
    db_session,
) -> None:
    """Drive the real send path, not a mock, and read the client-visible frame.

    ``test_idempotent_resume_frame_is_enriched_with_the_live_row`` already
    pins enrichment, but it calls the enricher directly on a hand-built dict.
    This one goes through a real ``ConnectionManager`` and a real socket sink
    from the real durable command, so it also fails if the handler stops
    routing the frame, or starts supplying a tuple of its own and suppressing
    enrichment -- both verified by mutation.
    """

    owner = _user(db_session, "real-send-path-owner")
    task = _task(
        db_session,
        int(owner.id),
        status=TaskStatus.RUNNING,
        control_state=TaskControlState.RUNNING,
    )
    task.runner_id = "another-process"
    task.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=1)
    task.state_version = 12
    db_session.commit()

    sink = _RecordingWebSocket()
    real_manager = websocket_api.ConnectionManager()
    real_manager.register_connection(sink, int(task.id))
    # The durable executor routes personal detail to the command's recorded
    # origin, never to task membership, so the sink has to be bound as the
    # origin for this exact command id.
    command = _command(task, owner, "resume-real-send-path")

    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent_manager = MagicMock()
    agent_manager.get_agent_for_task = AsyncMock(return_value=agent)

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=agent_manager),
        patch.object(websocket_api, "manager", real_manager),
        patch.object(websocket_api, "background_task_manager", _free_slot_manager()),
    ):
        websocket_api._command_origins.register(command.command_id, sink, int(task.id))
        try:
            result = await _execute_durable_task_command(command)
        finally:
            # ``_execute_durable_task_command`` is the inner call, so the
            # discard that normally runs in ``execute_durable_task_command``
            # does not; the registry is module-global.
            websocket_api._command_origins.discard_command(
                command.command_id, int(task.id)
            )

    assert result is not None
    assert result["resume_outcome"] == ResumeCommandOutcome.ALREADY_IN_PROGRESS.value

    frames = [frame for frame in sink.sent if frame.get("type") == "task_resumed"]
    assert len(frames) == 1
    # The live row, attached by the transport because the handler supplied no
    # tuple of its own.
    assert frames[0]["task"]["status"] == TaskStatus.RUNNING.value
    assert frames[0]["task"]["state_version"] == 12
    assert frames[0]["task"]["run_id"] == "run-a"
    assert frames[0]["status"] == TaskStatus.RUNNING.value
    # Load-bearing for the frame type: the client's
    # ``taskEventMatchesControlState`` maps ``task_resumed`` to
    # ``["running"]``, so a control state of anything else -- for instance
    # ``resume_requested`` on a coordinator branch -- would make the client
    # discard the event-specific handling and re-apply the stale status
    # instead. This branch is the one place the control state is already
    # known to be RUNNING.
    assert frames[0]["control_state"] == TaskControlState.RUNNING.value
    # And it must not be an ``error`` frame: that handler appends a failed
    # chat bubble, which would report this success as a failure.
    assert not [f for f in sink.sent if f.get("type") in {"error", "task_error"}]


@pytest.mark.asyncio
async def test_resume_defers_when_the_row_moves_under_the_admission_snapshot(
    db_session,
) -> None:
    """A cancel landing mid-handler must not get a resume scheduled over it.

    The writes staged here are an A2A cancel's -- terminal status, lease
    cleared, run id preserved -- because that is the sharpest shape: neither
    ``expected_run_id`` nor the foreign-lease check notices any of it, so only
    the state-version fence can. A real A2A cancel cannot interleave at this
    exact point (the durable queue serialises commands per task), but the
    reachable writers the fence exists for -- the a2a and v1 reply preleases,
    which arrive over HTTP and preserve the run id while bumping the version
    -- move the row the same way as far as this transition can tell.
    """

    owner = _user(db_session, "cancelled-midflight-owner")
    task = _task(db_session, int(owner.id))
    task.state_version = 3
    db_session.commit()
    task_id = int(task.id)

    resume_started = asyncio.Event()

    async def execute_resume_background(**_kwargs) -> None:
        resume_started.set()

    with (
        _resume_runtime_patches(outcome=ResumeReservationOutcome.RESERVED) as (
            background_manager,
            connection_manager,
            _agent_manager,
        ),
        patch.object(
            websocket_api,
            "execute_resume_background",
            side_effect=execute_resume_background,
        ),
    ):
        # Land the cancel between the handler's snapshot read and its
        # transition, which is the window the fence exists for. Scope
        # resolution is a convenient seam inside that window, not the last
        # await in it -- ``get_agent_for_task`` runs after it.
        def land_cancel_then_resolve(*args, **kwargs):
            row = db_session.query(Task).filter(Task.id == task_id).one()
            row.status = TaskStatus.FAILED
            row.control_state = TaskControlState.FAILED.value
            row.runner_id = None
            row.lease_expires_at = None
            row.state_version = 4
            db_session.commit()
            return None

        with patch.object(
            websocket_api,
            "resolve_execution_scope_off_turn",
            side_effect=land_cancel_then_resolve,
        ):
            # Deferred, not rejected. The mover is overwhelmingly likely to
            # be a competing resume -- the same situation the RUNNING branch
            # calls an idempotent success -- so a terminal COMMAND_FAILED
            # would give one interleaving of that race the harshest outcome
            # in the handler. The retry re-reads the row and re-decides.
            with pytest.raises(
                TaskCommandDeferred,
                match="changed while it was being admitted",
            ):
                await _execute_durable_task_command(
                    _command(task, owner, "resume-over-a-cancel")
                )

    # The raw internal diagnostic must not reach the client. Asserting on the
    # exception text does not show that -- the pre-check's wording happens not
    # to contain the phrase either way. What the old code actually did was
    # fall into the runtime-error arm and send a personal
    # ``{"type": "error", "message": "Runtime error: ..."}`` frame, so the
    # absence of any frame at all is the property worth pinning.
    connection_manager.send_personal_message.assert_not_awaited()
    # The reservation has to go back, or every later RESUME reads
    # RESERVATION_HELD and defers itself into a terminal failure -- the exact
    # outcome this deferral exists to avoid.
    background_manager.release_resume_reservation.assert_called_once_with(int(task.id))
    assert not resume_started.is_set()

    db_session.expire_all()
    row = db_session.query(Task).filter(Task.id == task_id).one()
    assert row.status == TaskStatus.FAILED
    assert row.control_state == TaskControlState.FAILED.value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("late_outcome", "expected"),
    [
        (
            ResumeReservationOutcome.COORDINATOR_RUNNING,
            ResumeCommandOutcome.ALREADY_IN_PROGRESS,
        ),
        (ResumeReservationOutcome.RESERVATION_HELD, None),
        (ResumeReservationOutcome.SHUTTING_DOWN, None),
    ],
)
async def test_slot_taken_between_admission_and_reservation(
    db_session,
    late_outcome: ResumeReservationOutcome,
    expected: ResumeCommandOutcome | None,
) -> None:
    """The second reservation attempt has its own contention branches.

    Admission and reservation are two separate reads of the same slot, with
    the scope resolution and the agent build -- both of which await, and
    either of which can be slow -- in between. So the slot can be free at
    admission and taken by the time it is claimed, which is the only way to
    reach the branches at the ``try_reserve_resume`` call site rather than at
    ``resume_admission_state``. Every other test in this file short-circuits
    at the earlier call, so without this one those branches are unreached.
    """

    owner = _user(db_session, f"late-contention-{late_outcome.value}")
    task = _task(db_session, int(owner.id))

    with _resume_runtime_patches(outcome=ResumeReservationOutcome.RESERVED) as (
        background_manager,
        _connection_manager,
        agent_manager,
    ):
        # Free at admission, taken by the time the slot is claimed.
        background_manager.resume_admission_state.return_value = None
        background_manager.try_reserve_resume.return_value = late_outcome

        if expected is ResumeCommandOutcome.ALREADY_IN_PROGRESS:
            result = await _execute_durable_task_command(
                _command(task, owner, f"late-{late_outcome.value}")
            )
            assert result is not None
            assert result["resume_outcome"] == expected.value
        else:
            with pytest.raises(TaskCommandDeferred, match="resume slot"):
                await _execute_durable_task_command(
                    _command(task, owner, f"late-{late_outcome.value}")
                )

    # Reaching this call site at all means the earlier admission check passed,
    # so the expensive prerequisites were paid before the contention showed up
    # -- which is exactly why this branch has to exist separately.
    agent_manager.get_agent_for_task.assert_awaited_once()
    background_manager.try_reserve_resume.assert_called_once_with(
        int(task.id),
        expected_run_id="run-a",
    )
    # No reservation was taken, so none may be released.
    background_manager.release_resume_reservation.assert_not_called()
