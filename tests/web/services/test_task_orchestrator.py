"""Tests for the turn-lifecycle API in ``task_orchestrator``.

Covers:

  - ``TaskTurnPayload`` dual-message channel
  - ``TurnKind`` + ``force_fresh`` orthogonal kind/flag
  - ``begin_turn`` atomic claim + persist + bg schedule
  - ``finish_turn`` symmetric terminal-field writer + lease ownership guard
  - ``_schedule_bg`` lease lifecycle wrapper

Tests use SQLite in-memory + direct ORM, mocking only the bits that
require an actual agent runtime (``execute_task_background``).
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Barrier
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.orm import sessionmaker

from xagent.core.tools.adapters.vibe.connector_runtime import ConnectorRef
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services.chat_history_service import (
    DELIVERY_DISPATCHED,
    DELIVERY_FAILED,
)
from xagent.web.services.connector_runtime import (
    get_ephemeral_runtime_values,
    pop_ephemeral_runtime_values,
    store_ephemeral_runtime_values,
)
from xagent.web.services.task_execution_controller import task_execution_controller
from xagent.web.services.task_lease_service import (
    acquire_task_lease_isolated,
    get_runner_id,
)
from xagent.web.services.task_orchestrator import (
    TaskTurnError,
    TaskTurnNotFoundError,
    TaskTurnOrchestrator,
    TaskTurnPayload,
    TurnKind,
    _begin_turn_atomic_sync,
    _ClaimedTurn,
    _schedule_bg,
    finish_turn,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'orchestrator.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


def _create_user(db) -> User:
    user = User(username="orch-user", password_hash="hash", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _create_task(
    db,
    user_id: int,
    *,
    status: TaskStatus = TaskStatus.PENDING,
    input_: str | None = None,
    output: str | None = None,
    error_message: str | None = None,
) -> Task:
    task = Task(
        user_id=user_id,
        title="Orchestrator test",
        description="test",
        status=status,
        execution_mode="auto",
        input=input_,
        output=output,
        error_message=error_message,
        source="sdk",
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def _store_runtime_secret_for_turn(turn_id: str) -> None:
    store_ephemeral_runtime_values(
        turn_id,
        {
            ConnectorRef("mcp", 1): {
                "secrets": {"authorization": "Bearer cleanup-token"}
            }
        },
    )


def test_channel_and_web_claims_are_cross_process_exclusive(db_session) -> None:
    user = _create_user(db_session)
    task = _create_task(db_session, int(user.id))
    task_id = int(task.id)
    barrier = Barrier(2)

    def claim_channel() -> str | None:
        barrier.wait()
        lease = acquire_task_lease_isolated(
            task_id,
            runner_id="channel-runner",
            new_run=True,
        )
        return lease.run_id if lease is not None else None

    def claim_web() -> str | None:
        barrier.wait()
        try:
            claimed = _begin_turn_atomic_sync(
                task_id,
                int(user.id),
                payload=TaskTurnPayload("web turn"),
                kind=TurnKind.CREATE,
            )
        except TaskTurnError:
            return None
        return claimed.run_id

    with ThreadPoolExecutor(max_workers=2) as executor:
        channel_result = executor.submit(claim_channel)
        web_result = executor.submit(claim_web)
        winners = [
            run_id
            for run_id in (channel_result.result(), web_result.result())
            if run_id is not None
        ]

    assert len(winners) == 1
    db_session.expire_all()
    stored = db_session.query(Task).filter(Task.id == task_id).one()
    assert stored.status == TaskStatus.RUNNING
    assert stored.run_id == winners[0]


@pytest.fixture()
def mock_schedule_bg():
    """Stub the bg coroutine spawn so begin_turn tests don't actually run
    an agent. Opt-in: tests that drive ``_schedule_bg`` directly skip
    this fixture and patch deeper layers themselves.

    Uses ``AsyncMock()`` without an explicit ``return_value`` —
    instantiating ``asyncio.Future()`` at fixture-setup time needs a
    running event loop, which pytest-asyncio doesn't provide during
    fixture collection in CI. The default ``AsyncMock`` return is a
    plain ``MagicMock``, which begin_turn ignores anyway.
    """
    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ) as mocked:
        yield mocked


@pytest.fixture(autouse=True)
def _clear_bg_manager():
    """Reset the global bg manager between tests so _refuse_if_bg_inflight
    sees a clean slate."""
    from xagent.web.api.websocket import background_task_manager

    background_task_manager.running_tasks.clear()
    yield
    background_task_manager.running_tasks.clear()


# ---------------------------------------------------------------------------
# TaskTurnPayload
# ---------------------------------------------------------------------------


def test_payload_for_agent_falls_back_to_transcript() -> None:
    p = TaskTurnPayload(transcript_message="hi")
    assert p.for_agent == "hi"


def test_payload_uses_execution_when_provided() -> None:
    p = TaskTurnPayload(
        transcript_message="summarize this",
        execution_message="summarize this\n\n[file context]",
    )
    assert p.for_agent == "summarize this\n\n[file context]"


# ---------------------------------------------------------------------------
# begin_turn — happy paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_begin_turn_create_clears_no_terminal_fields_when_pending(
    db_session,
    mock_schedule_bg,
) -> None:
    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.PENDING)
    payload = TaskTurnPayload("first turn")

    started = await TaskTurnOrchestrator.begin_turn(
        task_id=int(task.id),
        payload=payload,
        task_owner_user_id=int(user.id),
        kind=TurnKind.CREATE,
        force_fresh=False,
    )

    db_session.refresh(task)
    assert task.status == TaskStatus.RUNNING
    assert task.input == "first turn"
    assert task.output is None
    assert task.error_message is None
    assert task.runner_id is None
    assert task.lease_expires_at is None
    assert task.run_id == started.run_id
    assert task.state_version == started.state_version == 1
    assert task.control_state == started.control_state == "running"
    delivery = (
        db_session.query(TaskChatMessage)
        .filter(TaskChatMessage.turn_id == payload.turn_id)
        .one()
    )
    assert delivery.delivery_status == DELIVERY_DISPATCHED


@pytest.mark.asyncio
async def test_begin_turn_claim_clears_stale_lease_columns(
    db_session,
    mock_schedule_bg,
) -> None:
    """Append claim must reset stale lease columns so the bg runner can
    acquire after a crashed worker left runner_id behind."""
    from xagent.web.services.task_lease_service import utc_now

    user = _create_user(db_session)
    task = _create_task(
        db_session,
        user.id,
        status=TaskStatus.COMPLETED,
        input_="done",
        output="answer",
    )
    task.runner_id = "dead-runner-from-crash"
    task.lease_expires_at = utc_now() + timedelta(hours=1)
    db_session.commit()

    await TaskTurnOrchestrator.begin_turn(
        task_id=int(task.id),
        payload=TaskTurnPayload("follow up"),
        task_owner_user_id=int(user.id),
        kind=TurnKind.APPEND,
        force_fresh=False,
    )

    db_session.refresh(task)
    assert task.status == TaskStatus.RUNNING
    assert task.runner_id is None
    assert task.lease_expires_at is None


@pytest.mark.asyncio
async def test_begin_turn_append_clears_stale_output_and_error(
    db_session,
    mock_schedule_bg,
) -> None:
    """Latest-turn snapshot invariant: appending a new turn must reset
    output / error_message from the previous turn so GET returns a
    coherent latest-turn snapshot.
    """
    user = _create_user(db_session)
    task = _create_task(
        db_session,
        user.id,
        status=TaskStatus.COMPLETED,
        input_="first question",
        output="first answer",
        error_message=None,
    )

    await TaskTurnOrchestrator.begin_turn(
        task_id=int(task.id),
        payload=TaskTurnPayload("second question"),
        task_owner_user_id=int(user.id),
        kind=TurnKind.APPEND,
        force_fresh=False,
    )

    db_session.refresh(task)
    assert task.status == TaskStatus.RUNNING
    assert task.input == "second question"
    assert task.output is None, "stale first-turn output must be cleared"
    assert task.error_message is None


@pytest.mark.asyncio
async def test_begin_turn_append_clears_stale_error_message(
    db_session,
    mock_schedule_bg,
) -> None:
    """Latest-turn snapshot invariant (FAILED side): appending after a
    failed turn must also clear the prior turn's error_message."""
    user = _create_user(db_session)
    task = _create_task(
        db_session,
        user.id,
        status=TaskStatus.FAILED,
        input_="first",
        output=None,
        error_message="first turn blew up",
    )

    await TaskTurnOrchestrator.begin_turn(
        task_id=int(task.id),
        payload=TaskTurnPayload("second"),
        task_owner_user_id=int(user.id),
        kind=TurnKind.APPEND,
        force_fresh=False,
    )

    db_session.refresh(task)
    assert task.status == TaskStatus.RUNNING
    assert task.input == "second"
    assert task.error_message is None
    assert task.output is None


@pytest.mark.asyncio
async def test_begin_turn_append_accepts_paused_task_as_new_turn(
    db_session,
    mock_schedule_bg,
) -> None:
    """A message sent after pause starts the next turn, not a checkpoint resume."""
    user = _create_user(db_session)
    task = _create_task(
        db_session,
        user.id,
        status=TaskStatus.PAUSED,
        input_="previous request",
        output="stale partial output",
        error_message="stale pause detail",
    )

    payload = TaskTurnPayload("new request after pause")
    await TaskTurnOrchestrator.begin_turn(
        task_id=int(task.id),
        payload=payload,
        task_owner_user_id=int(user.id),
        kind=TurnKind.APPEND,
        force_fresh=False,
    )

    db_session.refresh(task)
    assert task.status == TaskStatus.RUNNING
    assert task.input == "new request after pause"
    assert task.output is None
    assert task.error_message is None

    persisted = (
        db_session.query(TaskChatMessage)
        .filter(TaskChatMessage.task_id == int(task.id), TaskChatMessage.role == "user")
        .one()
    )
    assert persisted.content == "new request after pause"
    assert persisted.turn_id == payload.turn_id

    mock_schedule_bg.assert_called_once()


@pytest.mark.asyncio
async def test_begin_turn_passes_force_fresh_through_to_schedule_bg(
    db_session,
    mock_schedule_bg,
) -> None:
    """Dual-channel payload + force_fresh forwarding: begin_turn forwards
    the full ``TaskTurnPayload`` and ``force_fresh`` flag to
    ``_schedule_bg`` so the execution side receives both message
    channels and the right reconstruct-state mode."""
    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.COMPLETED)

    payload = TaskTurnPayload(
        transcript_message="show me",
        execution_message="show me\n\n[file: foo.pdf]",
    )
    await TaskTurnOrchestrator.begin_turn(
        task_id=int(task.id),
        payload=payload,
        task_owner_user_id=int(user.id),
        kind=TurnKind.APPEND,
        force_fresh=True,
    )

    mock_schedule_bg.assert_called_once()
    kwargs = mock_schedule_bg.call_args.kwargs
    assert kwargs["payload"] is payload
    assert kwargs["force_fresh"] is True

    persisted = (
        db_session.query(TaskChatMessage)
        .filter(TaskChatMessage.task_id == int(task.id), TaskChatMessage.role == "user")
        .one()
    )
    assert persisted.turn_id == payload.turn_id


@pytest.mark.asyncio
async def test_begin_turn_actor_is_audit_only_not_runtime(
    db_session,
    mock_schedule_bg,
    caplog,
) -> None:
    """``actor_user_id`` records who initiated the turn (an admin acting on
    another user's task) for audit/logging only. It must never reach the
    claim or the bg schedule -- those run as the OWNER -- and it must appear
    in the audit log line."""
    owner = _create_user(db_session)
    task = _create_task(db_session, owner.id, status=TaskStatus.COMPLETED)
    actor_id = int(owner.id) + 999  # a different principal (e.g. an admin)

    with caplog.at_level(logging.INFO, logger="xagent.web.services.task_orchestrator"):
        await TaskTurnOrchestrator.begin_turn(
            task_id=int(task.id),
            payload=TaskTurnPayload("follow-up"),
            task_owner_user_id=int(owner.id),
            actor_user_id=actor_id,
            kind=TurnKind.APPEND,
        )

    # The bg schedule runs as the owner; the actor never leaks into it.
    kwargs = mock_schedule_bg.call_args.kwargs
    assert kwargs["task_owner_user_id"] == int(owner.id)
    assert actor_id not in kwargs.values()

    # The persisted user message is attributed to the owner, not the actor.
    persisted = (
        db_session.query(TaskChatMessage)
        .filter(TaskChatMessage.task_id == int(task.id), TaskChatMessage.role == "user")
        .one()
    )
    assert persisted.user_id == int(owner.id)

    # The actor is captured in the audit log.
    assert "turn started" in caplog.text
    assert f"owner={int(owner.id)}" in caplog.text
    assert f"actor={actor_id}" in caplog.text


# ---------------------------------------------------------------------------
# begin_turn — failure modes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_begin_turn_rejects_create_with_force_fresh(
    db_session,
    mock_schedule_bg,
) -> None:
    """Invalid kind + flag combo: CREATE + force_fresh has no meaning."""
    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.PENDING)

    with pytest.raises(ValueError, match="force_fresh has no meaning"):
        await TaskTurnOrchestrator.begin_turn(
            task_id=int(task.id),
            payload=TaskTurnPayload("x"),
            task_owner_user_id=int(user.id),
            kind=TurnKind.CREATE,
            force_fresh=True,
        )


@pytest.mark.asyncio
async def test_begin_turn_rejects_task_not_owned_by_user(
    db_session,
    mock_schedule_bg,
) -> None:
    """Ownership is folded into the atomic claim predicate. A ``user_id``
    that does not own the task → ``TaskTurnNotFoundError`` (404), NOT
    ``TaskTurnError`` (409), and no row is mutated. Passing a *different*
    user id (not ``task.user_id``) proves the predicate actually guards."""
    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.PENDING)

    with pytest.raises(TaskTurnNotFoundError):
        await TaskTurnOrchestrator.begin_turn(
            task_id=int(task.id),
            task_owner_user_id=int(user.id) + 9999,
            payload=TaskTurnPayload("x"),
            kind=TurnKind.CREATE,
        )

    db_session.refresh(task)
    assert task.status == TaskStatus.PENDING, "rejected claim must not mutate"
    mock_schedule_bg.assert_not_called()


@pytest.mark.asyncio
async def test_begin_turn_marks_failed_when_schedule_raises(
    db_session,
) -> None:
    """Post-commit invariant: once the claim commits (RUNNING) but
    ``_schedule_bg`` raises, the task must be forced FAILED so it is never
    left RUNNING with no bg worker (zombie)."""
    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.PENDING)
    payload = TaskTurnPayload("x")

    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(side_effect=RuntimeError("schedule boom")),
    ):
        with pytest.raises(RuntimeError, match="schedule boom"):
            await TaskTurnOrchestrator.begin_turn(
                task_id=int(task.id),
                task_owner_user_id=int(user.id),
                payload=payload,
                kind=TurnKind.CREATE,
            )

    db_session.refresh(task)
    assert task.status == TaskStatus.FAILED
    delivery = (
        db_session.query(TaskChatMessage)
        .filter(TaskChatMessage.turn_id == payload.turn_id)
        .one()
    )
    assert delivery.delivery_status == DELIVERY_FAILED


@pytest.mark.asyncio
async def test_begin_turn_schedules_even_when_caller_cancelled(db_session) -> None:
    """Cancellation safety: if begin_turn's caller is cancelled while the
    off-loop claim is in flight (which commits RUNNING in a worker thread),
    ``asyncio.shield`` must still let the claim+schedule finish, so a committed
    RUNNING task is never left with no scheduled worker."""
    import time as _time

    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.PENDING)

    def slow_claim(task_id, task_owner_user_id, *, payload, kind):
        _time.sleep(0.15)  # window during which we cancel the caller
        return _ClaimedTurn(
            status=TaskStatus.RUNNING,
            updated_at=datetime.now(timezone.utc),
            before_message_id=1,
            task_source="sdk",
        )

    sched = MagicMock(return_value=MagicMock())
    with (
        patch(
            "xagent.web.services.task_orchestrator._begin_turn_atomic_sync",
            new=slow_claim,
        ),
        patch(
            "xagent.web.services.task_orchestrator._schedule_bg",
            new=sched,
        ),
    ):
        t = asyncio.create_task(
            TaskTurnOrchestrator.begin_turn(
                task_id=int(task.id),
                task_owner_user_id=int(user.id),
                payload=TaskTurnPayload("x"),
                kind=TurnKind.CREATE,
            )
        )
        await asyncio.sleep(0.05)  # let it enter the off-loop claim
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
        # The shielded inner keeps running; give it time to finish.
        await asyncio.sleep(0.3)

    sched.assert_called_once()  # scheduled despite the cancellation


@pytest.mark.asyncio
async def test_repeated_cancellation_keeps_turn_command_gate_until_claim_settles(
    db_session,
) -> None:
    """A second cancellation cannot release the per-task command gate while
    the shielded claim is still committing in its worker thread."""
    import threading

    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.PENDING)
    claim_started = threading.Event()
    release_claim = threading.Event()
    contender_entered = asyncio.Event()

    def blocked_claim(task_id, task_owner_user_id, *, payload, kind):
        claim_started.set()
        assert release_claim.wait(timeout=2)
        return _ClaimedTurn(
            status=TaskStatus.RUNNING,
            updated_at=datetime.now(timezone.utc),
            before_message_id=1,
            task_source="sdk",
        )

    async def contender() -> None:
        async with task_execution_controller.command(int(task.id)):
            contender_entered.set()

    sched = MagicMock(return_value=MagicMock())
    with (
        patch(
            "xagent.web.services.task_orchestrator._begin_turn_atomic_sync",
            new=blocked_claim,
        ),
        patch("xagent.web.services.task_orchestrator._schedule_bg", new=sched),
    ):
        turn = asyncio.create_task(
            TaskTurnOrchestrator.begin_turn(
                task_id=int(task.id),
                task_owner_user_id=int(user.id),
                payload=TaskTurnPayload("x"),
                kind=TurnKind.CREATE,
            )
        )
        while not claim_started.is_set():
            await asyncio.sleep(0)

        turn.cancel()
        await asyncio.sleep(0.01)
        turn.cancel()
        contender_task = asyncio.create_task(contender())
        await asyncio.sleep(0.05)

        assert not turn.done()
        assert not contender_entered.is_set()

        release_claim.set()
        with pytest.raises(asyncio.CancelledError):
            await turn
        await asyncio.wait_for(contender_task, timeout=1)

    assert contender_entered.is_set()
    sched.assert_called_once()


@pytest.mark.asyncio
async def test_begin_turn_refuses_when_bg_inflight(
    db_session,
    mock_schedule_bg,
) -> None:
    from xagent.web.api.websocket import background_task_manager

    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.COMPLETED)

    # Plant a fake "still-running" entry in the bg manager registry.
    # ``_refuse_if_bg_inflight`` only checks ``.done() is False``, so a
    # MagicMock with that one attribute is enough — we don't need a
    # real asyncio.Task (and creating one would require an extra event
    # loop, which trips up pytest-asyncio's fixture machinery in CI).
    fake_inflight = MagicMock(spec=asyncio.Task)
    fake_inflight.done.return_value = False
    background_task_manager.running_tasks[int(task.id)] = fake_inflight

    try:
        with pytest.raises(TaskTurnError) as excinfo:
            await TaskTurnOrchestrator.begin_turn(
                task_id=int(task.id),
                payload=TaskTurnPayload("x"),
                task_owner_user_id=int(user.id),
                kind=TurnKind.APPEND,
            )
        assert excinfo.value.reason == "bg_inflight"

        # Critical: the DB row must NOT have been mutated
        db_session.refresh(task)
        assert task.status == TaskStatus.COMPLETED  # unchanged
        assert task.input is None  # unchanged
    finally:
        background_task_manager.running_tasks.pop(int(task.id), None)


@pytest.mark.asyncio
async def test_begin_turn_refuses_create_against_terminal_task(
    db_session,
    mock_schedule_bg,
) -> None:
    """kind=CREATE filters status==PENDING; a COMPLETED task must reject."""
    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.COMPLETED)

    with pytest.raises(TaskTurnError) as excinfo:
        await TaskTurnOrchestrator.begin_turn(
            task_id=int(task.id),
            payload=TaskTurnPayload("x"),
            task_owner_user_id=int(user.id),
            kind=TurnKind.CREATE,
        )
    assert excinfo.value.reason == "busy"


@pytest.mark.asyncio
async def test_begin_turn_refuses_append_against_pending_task(
    db_session,
    mock_schedule_bg,
) -> None:
    """kind=APPEND filters status IN TERMINAL; a PENDING task must reject."""
    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.PENDING)

    with pytest.raises(TaskTurnError) as excinfo:
        await TaskTurnOrchestrator.begin_turn(
            task_id=int(task.id),
            payload=TaskTurnPayload("x"),
            task_owner_user_id=int(user.id),
            kind=TurnKind.APPEND,
        )
    assert excinfo.value.reason == "busy"


# ---------------------------------------------------------------------------
# finish_turn
# ---------------------------------------------------------------------------


def test_finish_turn_completed_writes_output_clears_error(db_session) -> None:
    from xagent.web.models.chat_message import TaskChatMessage

    user = _create_user(db_session)
    task = _create_task(
        db_session,
        user.id,
        status=TaskStatus.COMPLETED,
        error_message="stale",
    )
    msg = TaskChatMessage(
        task_id=task.id,
        user_id=user.id,
        role="assistant",
        content="hello world",
        message_type="assistant_message",
    )
    db_session.add(msg)
    db_session.commit()

    finish_turn(db_session, int(task.id))

    db_session.refresh(task)
    assert task.output == "hello world"
    assert task.error_message is None


def test_finish_turn_failed_writes_error_clears_stale_output(db_session) -> None:
    """Latest-turn snapshot invariant (FAILED side): a FAILED turn
    must clear the prior turn's stale ``output`` so the GET response
    doesn't show ``status='failed' + output='prior answer'``."""
    user = _create_user(db_session)
    task = _create_task(
        db_session,
        user.id,
        status=TaskStatus.FAILED,
        output="prior successful output",
        error_message=None,
    )

    finish_turn(db_session, int(task.id))

    db_session.refresh(task)
    assert task.error_message is not None
    assert "Task execution failed" in task.error_message
    assert task.output is None  # latest-turn snapshot invariant


def test_finish_turn_running_skips_when_other_worker_holds_live_lease(
    db_session,
) -> None:
    """Lease ownership guard: when another worker actively holds the
    lease, finish_turn must leave the row alone and not flip RUNNING
    to FAILED."""
    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.RUNNING)
    # Plant a live lease held by a different runner
    task.runner_id = "other-worker"
    task.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)
    task.output = "other worker's in-progress output"
    db_session.commit()

    finish_turn(db_session, int(task.id))

    db_session.refresh(task)
    # No change: status stays RUNNING, output untouched, no error injected
    assert task.status == TaskStatus.RUNNING
    assert task.runner_id == "other-worker"
    assert task.output == "other worker's in-progress output"
    assert task.error_message is None


def test_finish_turn_running_flips_failed_when_no_live_lease(db_session) -> None:
    """RUNNING + no live lease elsewhere → genuine stuck task → flip FAILED."""
    user = _create_user(db_session)
    task = _create_task(
        db_session,
        user.id,
        status=TaskStatus.RUNNING,
        output="stale partial output",
    )
    # No runner_id / lease — task is stuck
    db_session.commit()

    finish_turn(db_session, int(task.id))

    db_session.refresh(task)
    assert task.status == TaskStatus.FAILED
    assert task.error_message is not None
    assert task.output is None  # latest-turn snapshot invariant


def test_finish_turn_running_flips_failed_when_lease_expired(db_session) -> None:
    """RUNNING + lease present but expired → still flip FAILED."""
    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.RUNNING)
    task.runner_id = "other-worker"
    task.lease_expires_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    db_session.commit()

    finish_turn(db_session, int(task.id))

    db_session.refresh(task)
    assert task.status == TaskStatus.FAILED


def test_finish_turn_running_flips_failed_when_we_own_lease(db_session) -> None:
    """RUNNING + we own the lease ourselves → still our bug to finalize."""
    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.RUNNING)
    task.runner_id = get_runner_id()  # our own process
    task.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)
    db_session.commit()

    finish_turn(db_session, int(task.id))

    db_session.refresh(task)
    assert task.status == TaskStatus.FAILED


# ---------------------------------------------------------------------------
# _schedule_bg lease lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_schedule_bg_skips_finish_turn_when_lease_acquire_fails(
    db_session,
) -> None:
    """Running-elsewhere short-circuit: lease taken by another worker
    → never call execute_task_background or finish_turn; bg coroutine
    returns clean."""
    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.RUNNING)

    from xagent.web.api.websocket import background_task_manager

    payload = TaskTurnPayload("x")
    _store_runtime_secret_for_turn(payload.turn_id)
    assert get_ephemeral_runtime_values(payload.turn_id) is not None

    with (
        patch(
            "xagent.web.services.task_orchestrator.acquire_task_lease_isolated",
            return_value=None,
        ),
        patch(
            "xagent.web.api.websocket.execute_task_background",
            new=AsyncMock(),
        ) as mock_exec,
        patch(
            "xagent.web.services.task_orchestrator.finish_turn",
        ) as mock_finish,
        patch.object(background_task_manager, "register_task"),
    ):
        # Note: this test does NOT use the mock_schedule_bg fixture
        # because we're testing _schedule_bg itself. The real
        # function runs with the deeper layers patched.
        bg_task = _schedule_bg(
            task_id=int(task.id),
            task_owner_user_id=int(user.id),
            task_source=task.source,
            payload=payload,
            force_fresh=False,
            context=None,
        )
        await bg_task

    mock_exec.assert_not_awaited()
    mock_finish.assert_not_called()
    assert get_ephemeral_runtime_values(payload.turn_id) is None
    assert pop_ephemeral_runtime_values(payload.turn_id) is None


@pytest.mark.asyncio
async def test_schedule_bg_releases_lease_on_execute_task_background_exception(
    db_session,
) -> None:
    """Lease must not leak when execute_task_background raises — _runner.finally
    must still call the lease release + workforce sync helper."""
    from xagent.web.api.websocket import background_task_manager
    from xagent.web.services.task_lease_service import TaskLease

    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.RUNNING)
    fake_lease = TaskLease(task_id=int(task.id), runner_id="test-runner")
    payload = TaskTurnPayload("x")
    _store_runtime_secret_for_turn(payload.turn_id)
    assert get_ephemeral_runtime_values(payload.turn_id) is not None

    with (
        patch(
            "xagent.web.services.task_orchestrator.acquire_task_lease_isolated",
            return_value=fake_lease,
        ),
        patch(
            "xagent.web.services.task_orchestrator.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.api.websocket.execute_task_background",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ),
        patch(
            "xagent.web.services.task_orchestrator.release_current_runner_task_lease_with_workforce_sync",
        ) as mock_release,
        patch(
            "xagent.web.services.task_orchestrator.finish_turn",
        ),
        patch.object(background_task_manager, "register_task"),
        patch(
            "xagent.web.services.task_orchestrator._get_agent_manager",
            return_value=MagicMock(),
        ),
    ):
        bg_task = _schedule_bg(
            task_id=int(task.id),
            task_owner_user_id=int(user.id),
            task_source=task.source,
            payload=payload,
            force_fresh=False,
            context=None,
        )
        # Wait for the inner _runner to finish (which raises internally
        # but the wrapping create_task absorbs it). The release should
        # still have been called in _runner.finally.
        try:
            await bg_task
        except RuntimeError:
            pass

    mock_release.assert_called_once()
    assert get_ephemeral_runtime_values(payload.turn_id) is None
    assert pop_ephemeral_runtime_values(payload.turn_id) is None


@pytest.mark.asyncio
async def test_schedule_bg_forwards_execution_message_to_execute_task_background(
    db_session,
) -> None:
    """Dual-channel payload propagation through the scheduler:
    ``_schedule_bg`` must pass ``payload.execution_message`` to
    ``execute_task_background``'s ``llm_user_message=`` parameter so
    the LLM-facing variant of the turn input survives the orchestrator
    boundary.

    Together with the ``begin_turn → _schedule_bg`` test above this
    locks in the full payload chain
    (begin_turn → _schedule_bg → execute_task_background) at the
    type-signature level, so a future refactor can't silently collapse
    transcript and execution into a single string.
    """
    from xagent.web.api.websocket import background_task_manager
    from xagent.web.services.task_lease_service import TaskLease

    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.RUNNING)
    fake_lease = TaskLease(task_id=int(task.id), runner_id="test-runner")
    payload = TaskTurnPayload(
        transcript_message="summarize this",
        execution_message="summarize this\n\n[uploaded file: secret.txt]",
    )
    _store_runtime_secret_for_turn(payload.turn_id)
    assert get_ephemeral_runtime_values(payload.turn_id) is not None

    with (
        patch(
            "xagent.web.services.task_orchestrator.acquire_task_lease_isolated",
            return_value=fake_lease,
        ),
        patch(
            "xagent.web.services.task_orchestrator.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.api.websocket.execute_task_background",
            new=AsyncMock(),
        ) as mock_exec,
        patch(
            "xagent.web.services.task_orchestrator.release_current_runner_task_lease_with_workforce_sync",
        ),
        patch(
            "xagent.web.services.task_orchestrator.finish_turn",
        ),
        patch.object(background_task_manager, "register_task"),
        patch(
            "xagent.web.services.task_orchestrator._get_agent_manager",
            return_value=MagicMock(),
        ),
    ):
        bg_task = _schedule_bg(
            task_id=int(task.id),
            task_owner_user_id=int(user.id),
            task_source=task.source,
            payload=payload,
            force_fresh=False,
            context={"turn_id": "caller-turn", "existing": "value"},
        )
        await bg_task

    mock_exec.assert_awaited_once()
    kwargs = mock_exec.await_args.kwargs
    # Dual-channel payload contract: transcript and LLM-facing channels are both
    # forwarded explicitly so execute_task_background can pick the
    # right one for the agent input.
    assert kwargs["user_message"] == "summarize this", (
        "transcript_message must reach execute_task_background.user_message"
    )
    assert (
        kwargs["llm_user_message"] == "summarize this\n\n[uploaded file: secret.txt]"
    ), "execution_message must reach execute_task_background.llm_user_message"
    assert kwargs["context"]["turn_id"] == payload.turn_id
    assert kwargs["context"]["existing"] == "value"
    assert get_ephemeral_runtime_values(payload.turn_id) is None
    assert pop_ephemeral_runtime_values(payload.turn_id) is None


@pytest.mark.asyncio
async def test_schedule_bg_acquires_expired_lease_on_first_try(db_session) -> None:
    """Expired lease columns are granted by acquire_task_lease's atomic WHERE."""
    from xagent.web.api.websocket import background_task_manager
    from xagent.web.services.task_lease_service import utc_now

    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.RUNNING)
    task.runner_id = "dead-runner"
    task.lease_expires_at = utc_now() - timedelta(seconds=5)
    db_session.commit()

    fake_snapshot = MagicMock()

    with (
        patch(
            "xagent.web.services.task_orchestrator.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.services.task_orchestrator.load_task_setup_snapshot_sync",
            return_value=fake_snapshot,
        ),
        patch(
            "xagent.web.api.websocket.execute_task_background",
            new=AsyncMock(),
        ) as mock_exec,
        patch(
            "xagent.web.services.task_orchestrator.release_current_runner_task_lease_with_workforce_sync",
        ),
        patch(
            "xagent.web.services.task_orchestrator.finish_turn",
        ),
        patch.object(background_task_manager, "register_task"),
        patch(
            "xagent.web.services.task_orchestrator._get_agent_manager",
            return_value=MagicMock(),
        ),
    ):
        bg_task = _schedule_bg(
            task_id=int(task.id),
            task_owner_user_id=int(user.id),
            task_source=task.source,
            payload=TaskTurnPayload("hello"),
            force_fresh=False,
            context=None,
        )
        await bg_task

    mock_exec.assert_awaited_once()


# ---------------------------------------------------------------------------
# _runner setup-error → FAILED safety net: prevents the
# acquire_lease-sets-RUNNING-then-no-one-clears-it zombie state when
# snapshot load or execute_task_background raises.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_schedule_bg_marks_task_failed_when_snapshot_load_raises(
    db_session,
) -> None:
    """Snapshot-load exception must not leave the row visible-as-running.

    ``acquire_task_lease_isolated`` writes ``status=RUNNING`` as part
    of taking the lease. Without the outer ``except`` in ``_runner``,
    an exception out of ``load_task_setup_snapshot_sync`` propagates
    through ``_runner``'s inner ``try`` block; ``finish_turn`` and
    ``execute_task_background`` never run, and the outer release
    block reads the still-RUNNING status and writes it back --
    leaving the task displayed as running but with no worker
    executing it.

    The outer ``except`` in ``_runner`` calls
    ``_mark_task_failed_if_running`` so the row is pushed to
    ``FAILED`` before release. This test pins both halves: the
    helper is invoked, and the row is FAILED at the end.
    """
    from xagent.web.api.websocket import background_task_manager
    from xagent.web.services.task_lease_service import TaskLease

    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.RUNNING)
    fake_lease = TaskLease(task_id=int(task.id), runner_id="test-runner")
    payload = TaskTurnPayload("x")
    _store_runtime_secret_for_turn(payload.turn_id)
    assert get_ephemeral_runtime_values(payload.turn_id) is not None

    with (
        patch(
            "xagent.web.services.task_orchestrator.acquire_task_lease_isolated",
            return_value=fake_lease,
        ),
        patch(
            "xagent.web.services.task_orchestrator.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.services.task_orchestrator.load_task_setup_snapshot_sync",
            side_effect=RuntimeError("simulated snapshot load failure"),
        ),
        patch(
            "xagent.web.api.websocket.execute_task_background",
            new=AsyncMock(),
        ) as mock_exec,
        patch(
            "xagent.web.services.task_orchestrator.release_current_runner_task_lease_with_workforce_sync",
        ) as mock_release,
        patch(
            "xagent.web.services.task_orchestrator.finish_turn",
        ),
        patch.object(background_task_manager, "register_task"),
        patch(
            "xagent.web.services.task_orchestrator._get_agent_manager",
            return_value=MagicMock(),
        ),
    ):
        bg_task = _schedule_bg(
            task_id=int(task.id),
            task_owner_user_id=int(user.id),
            task_source=task.source,
            payload=payload,
            force_fresh=False,
            context=None,
        )
        try:
            await bg_task
        except RuntimeError:
            pass

    # execute_task_background must not run when snapshot load raised.
    mock_exec.assert_not_called()
    # Lease must still be released (otherwise the task TTL-stucks).
    mock_release.assert_called_once()
    # The row should now be FAILED, not the zombie RUNNING.
    db_session.refresh(task)
    assert task.status == TaskStatus.FAILED, (
        f"Expected task.status == FAILED after snapshot raise, got {task.status}. "
        "If this fails, ``_mark_task_failed_if_running`` is not running, and the "
        "zombie-RUNNING regression is back."
    )
    assert task.error_message is not None
    assert "simulated snapshot load failure" in str(task.error_message)
    assert get_ephemeral_runtime_values(payload.turn_id) is None
    assert pop_ephemeral_runtime_values(payload.turn_id) is None


@pytest.mark.asyncio
async def test_schedule_bg_cleanup_handles_missing_payload_turn_id(db_session) -> None:
    from xagent.web.api.websocket import background_task_manager
    from xagent.web.services.task_lease_service import TaskLease

    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.RUNNING)
    fake_lease = TaskLease(task_id=int(task.id), runner_id="test-runner")

    with (
        patch(
            "xagent.web.services.task_orchestrator.acquire_task_lease_isolated",
            return_value=fake_lease,
        ),
        patch(
            "xagent.web.services.task_orchestrator.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.services.task_orchestrator.load_task_setup_snapshot_sync",
            return_value=MagicMock(),
        ),
        patch(
            "xagent.web.api.websocket.execute_task_background",
            new=AsyncMock(),
        ) as mock_exec,
        patch(
            "xagent.web.services.task_orchestrator.release_current_runner_task_lease_with_workforce_sync",
        ) as mock_release,
        patch(
            "xagent.web.services.task_orchestrator.finish_turn",
        ),
        patch.object(background_task_manager, "register_task"),
        patch(
            "xagent.web.services.task_orchestrator._get_agent_manager",
            return_value=MagicMock(),
        ),
    ):
        bg_task = _schedule_bg(
            task_id=int(task.id),
            task_owner_user_id=int(user.id),
            task_source=task.source,
            payload=None,  # type: ignore[arg-type]
            force_fresh=False,
            context=None,
        )
        await bg_task

    mock_exec.assert_not_called()
    mock_release.assert_called_once()


@pytest.mark.asyncio
async def test_schedule_bg_marks_task_failed_when_execute_raises(
    db_session,
) -> None:
    """Same safety net for exceptions out of ``execute_task_background``
    that bypass its inner ``try/except``. The outer ``except`` in
    ``_runner`` catches them and routes through
    ``_mark_task_failed_if_running``.
    """
    from xagent.web.api.websocket import background_task_manager
    from xagent.web.services.task_lease_service import TaskLease

    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.RUNNING)
    fake_lease = TaskLease(task_id=int(task.id), runner_id="test-runner")

    # Snapshot loader returns a minimal sentinel snapshot so the test
    # proceeds past the snapshot-None branch.
    fake_snapshot = MagicMock()

    with (
        patch(
            "xagent.web.services.task_orchestrator.acquire_task_lease_isolated",
            return_value=fake_lease,
        ),
        patch(
            "xagent.web.services.task_orchestrator.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.services.task_orchestrator.load_task_setup_snapshot_sync",
            return_value=fake_snapshot,
        ),
        patch(
            "xagent.web.api.websocket.execute_task_background",
            new=AsyncMock(side_effect=RuntimeError("simulated agent boom")),
        ),
        patch(
            "xagent.web.services.task_orchestrator.release_current_runner_task_lease_with_workforce_sync",
        ) as mock_release,
        patch(
            "xagent.web.services.task_orchestrator.finish_turn",
        ),
        patch.object(background_task_manager, "register_task"),
        patch(
            "xagent.web.services.task_orchestrator._get_agent_manager",
            return_value=MagicMock(),
        ),
    ):
        bg_task = _schedule_bg(
            task_id=int(task.id),
            task_owner_user_id=int(user.id),
            task_source=task.source,
            payload=TaskTurnPayload("x"),
            force_fresh=False,
            context=None,
        )
        try:
            await bg_task
        except RuntimeError:
            pass

    mock_release.assert_called_once()
    db_session.refresh(task)
    assert task.status == TaskStatus.FAILED
    assert task.error_message is not None
    assert "simulated agent boom" in str(task.error_message)


@pytest.mark.asyncio
async def test_schedule_bg_does_not_overwrite_terminal_status_from_execute(
    db_session,
) -> None:
    """``_mark_task_failed_if_running`` is guarded by ``status==RUNNING``
    so it never overwrites a terminal / control status that
    ``execute_task_background`` may have set inside its own
    try/except (PAUSED, WAITING_FOR_USER, FAILED, COMPLETED).

    Simulate the inner handler setting PAUSED before raising. After
    ``_runner`` returns, the row must remain PAUSED, not be flipped
    to FAILED by the outer safety net.
    """
    from xagent.web.api.websocket import background_task_manager
    from xagent.web.services.task_lease_service import TaskLease

    user = _create_user(db_session)
    task = _create_task(db_session, user.id, status=TaskStatus.RUNNING)
    fake_lease = TaskLease(task_id=int(task.id), runner_id="test-runner")
    fake_snapshot = MagicMock()

    async def fake_execute(*args, **kwargs):
        # Inner handler decides the turn is paused, commits, then a
        # later step raises. Outer except must not undo the PAUSED.
        from xagent.web.models.task import Task as TaskModel

        with sessionmaker(bind=get_engine())() as inner:
            row = inner.query(TaskModel).filter(TaskModel.id == task.id).first()
            row.status = TaskStatus.PAUSED
            inner.commit()
        raise RuntimeError("simulated late-stage error after PAUSED")

    with (
        patch(
            "xagent.web.services.task_orchestrator.acquire_task_lease_isolated",
            return_value=fake_lease,
        ),
        patch(
            "xagent.web.services.task_orchestrator.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.services.task_orchestrator.load_task_setup_snapshot_sync",
            return_value=fake_snapshot,
        ),
        patch(
            "xagent.web.api.websocket.execute_task_background",
            new=fake_execute,
        ),
        patch(
            "xagent.web.services.task_orchestrator.release_current_runner_task_lease_with_workforce_sync",
        ),
        patch(
            "xagent.web.services.task_orchestrator.finish_turn",
        ),
        patch.object(background_task_manager, "register_task"),
        patch(
            "xagent.web.services.task_orchestrator._get_agent_manager",
            return_value=MagicMock(),
        ),
    ):
        bg_task = _schedule_bg(
            task_id=int(task.id),
            task_owner_user_id=int(user.id),
            task_source=task.source,
            payload=TaskTurnPayload("x"),
            force_fresh=False,
            context=None,
        )
        try:
            await bg_task
        except RuntimeError:
            pass

    db_session.refresh(task)
    assert task.status == TaskStatus.PAUSED, (
        f"Expected PAUSED (set by execute), got {task.status}. If this fails, "
        "``_mark_task_failed_if_running``'s status==RUNNING guard regressed."
    )
