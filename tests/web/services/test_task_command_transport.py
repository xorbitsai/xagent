from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, is_dataclass
from datetime import datetime, timedelta
from threading import Barrier, Event, get_ident
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from tests.web.pool_contention_shared import (
    GUARD_TIMEOUT,
    LOOP_LIVENESS_TICKS,
    gated_pool_checkout,
    wait_for_ticks,
)
from xagent.core.agent.runner import UserMessageInjectionOutcome
from xagent.db.sqlite import apply_sqlite_concurrency_pragmas
from xagent.web.api import websocket as websocket_api
from xagent.web.api.websocket import (
    _load_command_message_delivery_status,
    execute_durable_task_command,
)
from xagent.web.models import database as database_module
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
from xagent.web.models.task_command_terminal_event import TaskCommandTerminalEvent
from xagent.web.models.user import User
from xagent.web.services import task_command_transport as task_command_transport_module
from xagent.web.services.task_command_transport import (
    COMMAND_COMPLETED,
    COMMAND_FAILED,
    COMMAND_PENDING,
    DISPATCHER_IDLE_SECONDS,
    MAX_COMMAND_DEFERS,
    MAX_COMMAND_FAILURES,
    ClaimedTaskCommand,
    TaskCommandConflictKind,
    TaskCommandDeferred,
    TaskCommandKind,
    TaskCommandOwnerStateError,
    TaskCommandRejected,
    TaskCommandTaskMissing,
    _claim_heartbeat,
    claim_task_command,
    classify_task_command_conflict,
    defer_task_command,
    dispatch_one_task_command,
    dispatch_task_command_promptly,
    enqueue_task_command,
    fail_task_command,
    finish_task_command,
    load_task_command,
    max_command_defers,
    notify_task_command_dispatcher,
    renew_task_command_claim,
    retry_failed_task_command,
    stage_task_command,
    start_task_command_dispatcher,
    stop_task_command_dispatcher,
    task_has_live_foreign_runner,
    task_has_live_runner,
)


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'task-commands.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


@pytest.fixture()
def queue_pool_command_db(tmp_path):
    """A real one-slot QueuePool for dispatcher checkout contention."""

    engine = create_engine(
        f"sqlite:///{tmp_path / 'task-command-queue-pool.db'}",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.2,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    try:
        yield engine, SessionLocal
    finally:
        engine.dispose()


@pytest.fixture()
def low_timeout_sqlite_engine(tmp_path):
    """A file-backed SQLite engine with a short busy_timeout for lock tests.

    init_db does not expose apply_sqlite_concurrency_pragmas's busy_timeout_ms
    parameter (it always uses the 5s production default), so a lock-path test
    that needs a bounded wait builds its own engine directly instead of going
    through init_db.
    """

    engine = create_engine(
        f"sqlite:///{tmp_path / 'task-command-lock-path.db'}",
        connect_args={"check_same_thread": False},
    )
    # Long enough to stay clear of thread-scheduling jitter under a loaded
    # test run, short enough to keep the lock-path tests well under 2s.
    apply_sqlite_concurrency_pragmas(engine, busy_timeout_ms=200)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    try:
        yield engine, SessionLocal
    finally:
        engine.dispose()


@pytest.fixture()
def postgres_task_command_sessions():
    """A real PostgreSQL sessionmaker plus one running task's id.

    Used only by the truly-concurrent raced-duplicate test: SQLite serializes
    all writers through one database-wide lock, so a second writer's insert
    can never be genuinely in flight at the same instant as the first's --
    it simply cannot begin until the first's transaction ends. PostgreSQL's
    per-row locking lets two inserts targeting the same unique key both be
    open at once, with the second blocking until the first resolves.
    """

    url = os.getenv("XAGENT_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("XAGENT_TEST_POSTGRES_URL is not set")
    # init_db rebinds the module-global engine/sessionmaker in place, and
    # every other fixture and test in this file depends on that global
    # pointing at its own sqlite engine -- the prior binding must be restored
    # on exit rather than left pointed at this fixture's postgres engine.
    prior_engine = database_module._engine
    prior_session_local = database_module._SessionLocal
    init_db(db_url=url)
    engine = get_engine()
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    SessionLocal = get_session_local()
    try:
        with SessionLocal() as db:
            user, task = _create_running_task(db)
            task.runner_id = None
            task.lease_expires_at = None
            db.commit()
            task_id = int(task.id)
            actor_id = int(user.id)
        yield SessionLocal, task_id, actor_id
    finally:
        Base.metadata.drop_all(bind=engine)
        database_module._engine = prior_engine
        database_module._SessionLocal = prior_session_local


def _create_running_task(db) -> tuple[User, Task]:
    user = User(username="command-user", password_hash="hash", is_admin=False)
    db.add(user)
    db.commit()
    task = Task(
        user_id=user.id,
        title="Durable commands",
        description="Durable commands",
        status=TaskStatus.RUNNING,
        execution_mode="auto",
        run_id="run-1",
        runner_id="runner-a",
        lease_expires_at=datetime.utcnow() + timedelta(minutes=1),
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return user, task


def test_enqueue_is_committed_and_idempotent(db_session) -> None:
    user, task = _create_running_task(db_session)
    payload = {"type": "pause_task"}

    first = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="pause-1",
        kind=TaskCommandKind.PAUSE,
        payload=payload,
    )
    duplicate = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="pause-1",
        kind=TaskCommandKind.PAUSE,
        payload=payload,
    )
    conflict = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="pause-1",
        kind=TaskCommandKind.RESUME,
        payload={"type": "resume_task"},
    )

    assert first.created is True
    assert duplicate.command_id == first.command_id
    assert duplicate.created is False
    assert duplicate.payload_matches is True
    assert conflict.payload_matches is False
    assert db_session.query(TaskExecutionCommand).count() == 1


def test_reused_actor_id_does_not_match_a_legacy_command(db_session) -> None:
    user, task = _create_running_task(db_session)
    actor_id = int(user.id)
    task_id = int(task.id)
    legacy = TaskExecutionCommand(
        task_id=task_id,
        actor_user_id=actor_id,
        actor_subject=f"legacy-user-id:{actor_id}",
        command_id="legacy-actor-idempotency",
        kind=TaskCommandKind.PAUSE.value,
        payload={"type": "pause_task"},
    )
    db_session.add(legacy)
    db_session.commit()
    assert user.actor_subject != legacy.actor_subject

    duplicate = stage_task_command(
        db_session,
        task_id=task_id,
        actor_user_id=actor_id,
        command_id=legacy.command_id,
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
    )
    classification = classify_task_command_conflict(
        db_session,
        task_id=task_id,
        command_id=legacy.command_id,
        actor_user_id=actor_id,
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
    )

    assert duplicate.created is False
    assert duplicate.payload_matches is False
    assert classification.kind is TaskCommandConflictKind.RACED_DUPLICATE
    assert classification.raced is not None
    assert classification.raced.payload_matches is False


def test_live_run_command_stays_with_owner_until_task_lease_expires(
    db_session,
) -> None:
    user, task = _create_running_task(db_session)
    command = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="message-1",
        kind=TaskCommandKind.MESSAGE,
        payload={"type": "chat", "message": "new guidance"},
    )

    assert claim_task_command(db_session, runner_id="runner-b") is None
    owner_claim = claim_task_command(db_session, runner_id="runner-a")
    assert owner_claim is not None
    assert owner_claim.id == command.command_id

    # Simulate the owner crashing after claim. Once both the command claim and
    # task lease expire, another worker can replay the same durable command.
    row = db_session.query(TaskExecutionCommand).filter_by(id=command.command_id).one()
    row.claim_expires_at = datetime.utcnow() - timedelta(seconds=1)
    task.lease_expires_at = datetime.utcnow() - timedelta(seconds=1)
    db_session.commit()

    recovered = claim_task_command(db_session, runner_id="runner-b")
    assert recovered is not None
    assert recovered.id == command.command_id
    assert recovered.attempt_count == 2


def test_reassigned_command_routes_only_to_the_current_live_owner(db_session) -> None:
    user, task = _create_running_task(db_session)
    command = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="owner-takeover",
        kind=TaskCommandKind.CANCEL,
        payload={"agent_id": 1},
    )
    task.runner_id = "runner-b"
    task.lease_expires_at = datetime.utcnow() + timedelta(minutes=1)
    db_session.commit()

    assert claim_task_command(db_session, runner_id="runner-c") is None
    current_owner_claim = claim_task_command(db_session, runner_id="runner-b")

    assert current_owner_claim is not None
    assert current_owner_claim.id == command.command_id


def test_same_command_row_has_a_single_concurrent_claim_winner(db_session) -> None:
    user, task = _create_running_task(db_session)
    task.runner_id = None
    task.lease_expires_at = None
    db_session.commit()
    command = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="concurrent-claim",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
    )
    barrier = Barrier(2)

    def claim(runner_id: str) -> int | None:
        SessionLocal = get_session_local()
        with SessionLocal() as db:
            barrier.wait()
            claimed = claim_task_command(
                db,
                runner_id=runner_id,
                command_db_id=command.command_id,
            )
            return claimed.id if claimed is not None else None

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(claim, "runner-a")
        second = executor.submit(claim, "runner-b")
        winners = [value for value in (first.result(), second.result()) if value]

    assert winners == [command.command_id]


def test_stale_attempt_cannot_mutate_reclaimed_command(db_session) -> None:
    user, task = _create_running_task(db_session)
    task.runner_id = None
    task.lease_expires_at = None
    db_session.commit()
    enqueued = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="reclaimed-generation",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
    )
    first = claim_task_command(
        db_session,
        runner_id="runner-a",
        command_db_id=enqueued.command_id,
    )
    assert first is not None
    row = db_session.get(TaskExecutionCommand, enqueued.command_id)
    assert row is not None
    row.claim_expires_at = datetime.utcnow() - timedelta(seconds=1)
    db_session.commit()

    second = claim_task_command(
        db_session,
        runner_id="runner-a",
        command_db_id=enqueued.command_id,
    )
    assert second is not None
    assert second.attempt_count == first.attempt_count + 1

    assert not renew_task_command_claim(
        enqueued.command_id,
        "runner-a",
        expected_attempt_count=first.attempt_count,
    )
    assert not fail_task_command(
        enqueued.command_id,
        "runner-a",
        "stale failure",
        expected_attempt_count=first.attempt_count,
    )
    assert not defer_task_command(
        enqueued.command_id,
        "runner-a",
        "stale deferral",
        expected_attempt_count=first.attempt_count,
    )
    assert not finish_task_command(
        enqueued.command_id,
        "runner-a",
        expected_attempt_count=first.attempt_count,
    )

    db_session.expire_all()
    row = db_session.get(TaskExecutionCommand, enqueued.command_id)
    assert row is not None
    assert row.status == "processing"
    assert row.claimed_by == "runner-a"
    assert row.attempt_count == second.attempt_count
    assert (
        db_session.query(TaskCommandTerminalEvent)
        .filter(TaskCommandTerminalEvent.task_command_id == enqueued.command_id)
        .count()
        == 0
    )


def test_live_foreign_runner_is_rechecked_before_cancel(db_session) -> None:
    _user, task = _create_running_task(db_session)

    assert task_has_live_foreign_runner(int(task.id), runner_id="runner-b") is True
    assert task_has_live_foreign_runner(int(task.id), runner_id="runner-a") is False
    assert task_has_live_runner(int(task.id), expected_run_id="run-1") is True
    assert task_has_live_runner(int(task.id), expected_run_id="run-2") is False

    task.lease_expires_at = datetime.utcnow() - timedelta(seconds=1)
    db_session.commit()

    assert task_has_live_foreign_runner(int(task.id), runner_id="runner-b") is False
    assert task_has_live_runner(int(task.id), expected_run_id="run-1") is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({}, "Agent ID is missing or null in cancel command payload"),
        ({"agent_id": None}, "Agent ID is missing or null in cancel command payload"),
        ({"agent_id": "invalid"}, "Agent ID 'invalid' is invalid"),
    ],
)
async def test_cancel_command_rejects_invalid_agent_id_payload(
    db_session,
    payload,
    error: str,
) -> None:
    user, task = _create_running_task(db_session)
    task.runner_id = None
    task.lease_expires_at = None
    db_session.commit()
    command = ClaimedTaskCommand(
        id=1,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="invalid-cancel",
        kind=TaskCommandKind.CANCEL,
        payload=payload,
        target_run_id=None,
        attempt_count=1,
    )

    with pytest.raises(ValueError, match=error):
        await execute_durable_task_command(command)


@pytest.mark.asyncio
async def test_cancel_command_defers_on_a_live_foreign_owner(db_session) -> None:
    user, task = _create_running_task(db_session)
    command = ClaimedTaskCommand(
        id=1,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="misrouted-cancel",
        kind=TaskCommandKind.CANCEL,
        payload={"agent_id": 1},
        target_run_id=None,
        attempt_count=1,
    )

    with pytest.raises(TaskCommandDeferred, match="active task lease owner"):
        await execute_durable_task_command(command)


@pytest.mark.asyncio
async def test_pause_command_defers_on_a_live_foreign_owner(db_session) -> None:
    user, task = _create_running_task(db_session)
    command = ClaimedTaskCommand(
        id=1,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="misrouted-pause",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
        target_run_id=None,
        attempt_count=1,
    )

    with pytest.raises(TaskCommandDeferred, match="active task lease owner"):
        await execute_durable_task_command(command)


@pytest.mark.asyncio
async def test_resume_completes_when_the_target_run_has_a_live_foreign_owner(
    db_session,
) -> None:
    user, task = _create_running_task(db_session)
    task.control_state = "running"
    db_session.commit()
    command = ClaimedTaskCommand(
        id=1,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="idempotent-foreign-resume",
        kind=TaskCommandKind.RESUME,
        payload={"type": "resume_task"},
        target_run_id="run-1",
        attempt_count=1,
    )

    result = await execute_durable_task_command(command)

    assert result is not None
    assert result["resume_outcome"] == "already_in_progress"


@pytest.mark.asyncio
async def test_resume_defers_when_a_settling_turn_still_holds_a_foreign_lease(
    db_session,
) -> None:
    """RESUME still defers for a foreign lease in a state it cannot classify.

    The idempotency evidence only settles RUNNING rows. A turn that has
    committed WAITING_FOR_USER but whose lease columns have not been cleared
    yet is not evidence that anything is resuming, so the command must defer
    rather than schedule into the previous owner's live lease.
    """

    user, task = _create_running_task(db_session)
    task.status = TaskStatus.WAITING_FOR_USER
    task.control_state = "waiting_for_user"
    db_session.commit()
    command = ClaimedTaskCommand(
        id=1,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="resume-into-foreign-lease",
        kind=TaskCommandKind.RESUME,
        payload={"type": "resume_task"},
        target_run_id="run-1",
        attempt_count=1,
    )

    with pytest.raises(TaskCommandDeferred):
        await execute_durable_task_command(command)


@pytest.mark.asyncio
async def test_cancel_command_does_not_require_persisted_actor(db_session) -> None:
    _user, task = _create_running_task(db_session)
    task.runner_id = None
    task.lease_expires_at = None
    db_session.commit()
    command = ClaimedTaskCommand(
        id=1,
        task_id=int(task.id),
        actor_user_id=None,
        command_id="cancel-with-deleted-actor",
        kind=TaskCommandKind.CANCEL,
        payload={},
        target_run_id=None,
        attempt_count=1,
    )

    with (
        patch.object(websocket_api, "_load_command_actor") as load_actor,
        pytest.raises(ValueError, match="Agent ID is missing"),
    ):
        await execute_durable_task_command(command)
    load_actor.assert_not_called()


@pytest.mark.asyncio
async def test_pause_command_with_no_actor_fails_the_way_f1_identified(
    db_session,
) -> None:
    """PR #1060 review, F1: unlike CANCEL (see the sibling test above),
    PAUSE DOES require a persisted actor -- _load_command_actor(None)
    raises. This is exactly the failure the stale-preview-run reaper's
    dispatch hit before the fix: it called
    pause_workforce_tasks_after_archive with actor_user_id=None for every
    reaped run, so every reaped run's "cancel" never actually paused the
    running task."""
    _user, task = _create_running_task(db_session)
    task.runner_id = None
    task.lease_expires_at = None
    db_session.commit()
    command = ClaimedTaskCommand(
        id=1,
        task_id=int(task.id),
        actor_user_id=None,
        command_id="pause-with-no-actor",
        kind=TaskCommandKind.PAUSE,
        payload={},
        target_run_id=None,
        attempt_count=1,
    )

    with pytest.raises(ValueError, match="Task command has no actor"):
        await execute_durable_task_command(command)


@pytest.mark.asyncio
async def test_pause_command_with_a_real_actor_gets_past_the_actor_check(
    db_session,
) -> None:
    """PR #1060 review, F1 fix: WorkforceRunPauseTarget now carries the
    run's own owner (WorkforceRun.user_id, nullable=False) as
    actor_user_id, so the reaper's PAUSE dispatch no longer hits the
    failure above. Whatever _handle_pause_task_unserialized does next is
    exercised by other PAUSE tests; the only thing under test here is
    getting past the actor-loading gate without F1's specific failure."""
    user, task = _create_running_task(db_session)
    task.runner_id = None
    task.lease_expires_at = None
    db_session.commit()
    command = ClaimedTaskCommand(
        id=2,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="pause-with-real-actor",
        kind=TaskCommandKind.PAUSE,
        payload={},
        target_run_id=None,
        attempt_count=1,
    )

    try:
        await execute_durable_task_command(command)
    except Exception as exc:
        assert "Task command has no actor" not in str(exc)


@pytest.mark.asyncio
async def test_only_terminal_command_failure_is_broadcast(db_session) -> None:
    _user, task = _create_running_task(db_session)
    task.runner_id = None
    task.lease_expires_at = None
    db_session.commit()
    transient = ClaimedTaskCommand(
        id=1,
        task_id=int(task.id),
        actor_user_id=None,
        command_id="transient-cancel-failure",
        kind=TaskCommandKind.CANCEL,
        payload={},
        target_run_id=None,
        attempt_count=1,
        failure_count=0,
    )
    terminal = ClaimedTaskCommand(
        id=2,
        task_id=int(task.id),
        actor_user_id=None,
        command_id="terminal-cancel-failure",
        kind=TaskCommandKind.CANCEL,
        payload={},
        target_run_id=None,
        attempt_count=1,
        failure_count=MAX_COMMAND_FAILURES - 1,
    )

    with patch.object(
        websocket_api.manager,
        "broadcast_to_task",
        new=AsyncMock(),
    ) as broadcast:
        with pytest.raises(ValueError, match="Agent ID is missing"):
            await execute_durable_task_command(transient)
        broadcast.assert_not_awaited()

        with pytest.raises(ValueError, match="Agent ID is missing"):
            await execute_durable_task_command(terminal)
        broadcast.assert_awaited_once()
        event, event_task_id = broadcast.await_args.args
        assert event_task_id == int(task.id)
        assert event["type"] == "agent_error"
        assert event["command_id"] == "terminal-cancel-failure"


@pytest.mark.asyncio
async def test_final_command_deferral_is_broadcast(db_session) -> None:
    _user, task = _create_running_task(db_session)
    command = ClaimedTaskCommand(
        id=1,
        task_id=int(task.id),
        actor_user_id=None,
        command_id="terminal-cancel-defer",
        kind=TaskCommandKind.CANCEL,
        payload={"agent_id": 1},
        target_run_id=None,
        attempt_count=1,
        defer_count=max_command_defers() - 1,
    )

    with patch.object(
        websocket_api.manager,
        "broadcast_to_task",
        new=AsyncMock(),
    ) as broadcast:
        with pytest.raises(TaskCommandDeferred, match="active task lease owner"):
            await execute_durable_task_command(command)
        broadcast.assert_awaited_once()


@pytest.mark.asyncio
async def test_command_is_rejected_after_run_rotation(db_session) -> None:
    user, task = _create_running_task(db_session)
    task.run_id = "run-2"
    task.runner_id = None
    task.lease_expires_at = None
    db_session.commit()
    command = ClaimedTaskCommand(
        id=1,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="stale-pause",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
        target_run_id="run-1",
        attempt_count=1,
    )

    with pytest.raises(TaskCommandRejected, match="Task run changed"):
        await execute_durable_task_command(command)


@pytest.mark.asyncio
async def test_stale_run_rejection_reason_is_persisted(db_session) -> None:
    user, task = _create_running_task(db_session)
    enqueued = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="stale-pause-result",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
    )
    task.run_id = "run-2"
    task.runner_id = None
    task.lease_expires_at = None
    db_session.commit()

    assert await dispatch_one_task_command(
        execute_durable_task_command,
        command_db_id=enqueued.command_id,
    )
    db_session.expire_all()
    stored = db_session.get(TaskExecutionCommand, enqueued.command_id)
    assert stored is not None
    assert stored.status == COMMAND_FAILED
    assert stored.result == {"rejection_reason": "stale_run"}


def test_later_command_cannot_overtake_unfinished_command(db_session) -> None:
    user, task = _create_running_task(db_session)
    first = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="pause-ordered",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
    )
    second = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="resume-ordered",
        kind=TaskCommandKind.RESUME,
        payload={"type": "resume_task"},
    )

    claimed_first = claim_task_command(db_session, runner_id="runner-a")
    assert claimed_first is not None
    assert claimed_first.id == first.command_id
    assert (
        claim_task_command(
            db_session,
            runner_id="runner-a",
            command_db_id=second.command_id,
        )
        is None
    )

    assert finish_task_command(first.command_id, "runner-a") is True
    claimed_second = claim_task_command(db_session, runner_id="runner-a")
    assert claimed_second is not None
    assert claimed_second.id == second.command_id


@pytest.mark.asyncio
async def test_dispatch_claims_and_completes_once(db_session) -> None:
    user, task = _create_running_task(db_session)
    # Untargeted commands may be consumed by the current test process.
    task.runner_id = None
    task.lease_expires_at = None
    db_session.commit()
    enqueued = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="cancel-once",
        kind=TaskCommandKind.CANCEL,
        payload={"agent_id": 1},
    )
    seen: list[int] = []

    async def execute(command):
        seen.append(command.id)
        return {"ok": True}

    assert await dispatch_one_task_command(execute, command_db_id=enqueued.command_id)
    assert not await dispatch_one_task_command(
        execute, command_db_id=enqueued.command_id
    )
    db_session.expire_all()
    stored = (
        db_session.query(TaskExecutionCommand).filter_by(id=enqueued.command_id).one()
    )
    assert seen == [enqueued.command_id]
    assert stored.status == COMMAND_COMPLETED
    assert stored.result == {"ok": True}


@pytest.mark.asyncio
async def test_deferred_handoff_retries_without_consuming_failure_budget(
    db_session,
) -> None:
    user, task = _create_running_task(db_session)
    task.runner_id = None
    task.lease_expires_at = None
    db_session.commit()
    enqueued = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="deferred-message",
        kind=TaskCommandKind.MESSAGE,
        payload={"type": "chat", "message": "wait"},
    )

    async def defer(_command):
        raise TaskCommandDeferred("checkpoint is not ready")

    assert await dispatch_one_task_command(defer, command_db_id=enqueued.command_id)
    db_session.expire_all()
    stored = db_session.get(TaskExecutionCommand, enqueued.command_id)
    assert stored is not None
    assert stored.status == "pending"
    assert stored.attempt_count == 1
    assert stored.failure_count == 0
    assert stored.defer_count == 1
    assert stored.claim_expires_at is not None

    stored.claim_expires_at = datetime.utcnow() - timedelta(seconds=1)
    db_session.commit()

    async def finish(_command):
        return {"applied": True}

    assert await dispatch_one_task_command(finish, command_db_id=enqueued.command_id)
    db_session.expire_all()
    stored = db_session.get(TaskExecutionCommand, enqueued.command_id)
    assert stored is not None
    assert stored.status == COMMAND_COMPLETED
    assert stored.attempt_count == 2
    assert stored.failure_count == 0
    assert stored.defer_count == 1


@pytest.mark.asyncio
async def test_deferred_message_eventually_fails_and_unblocks_cancel(
    db_session,
) -> None:
    user, task = _create_running_task(db_session)
    task.runner_id = None
    task.lease_expires_at = None
    db_session.commit()
    message = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="stuck-message",
        kind=TaskCommandKind.MESSAGE,
        payload={"type": "chat", "message": "wait forever"},
    )
    cancel = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="cancel-after-stuck-message",
        kind=TaskCommandKind.CANCEL,
        payload={"agent_id": 1},
    )
    row = db_session.get(TaskExecutionCommand, message.command_id)
    assert row is not None
    row.defer_count = max_command_defers() - 1
    db_session.commit()

    async def defer(_command):
        raise TaskCommandDeferred("checkpoint never became ready")

    assert await dispatch_one_task_command(defer, command_db_id=message.command_id)
    db_session.expire_all()
    row = db_session.get(TaskExecutionCommand, message.command_id)
    assert row is not None
    assert row.status == COMMAND_FAILED
    assert row.failure_count == 0
    assert row.defer_count == max_command_defers()
    event = (
        db_session.query(TaskCommandTerminalEvent)
        .filter(TaskCommandTerminalEvent.task_command_id == message.command_id)
        .one()
    )
    assert event.outcome == COMMAND_FAILED
    assert event.outcome_version == row.attempt_count

    cancel_claim = claim_task_command(db_session)
    assert cancel_claim is not None
    assert cancel_claim.id == cancel.command_id


@pytest.mark.asyncio
async def test_real_failures_use_a_separate_bounded_budget(db_session) -> None:
    user, task = _create_running_task(db_session)
    task.runner_id = None
    task.lease_expires_at = None
    db_session.commit()
    command = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="last-failure",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
    )
    claimed = claim_task_command(db_session, runner_id="runner-a")
    assert claimed is not None
    row = db_session.get(TaskExecutionCommand, command.command_id)
    assert row is not None
    row.failure_count = MAX_COMMAND_FAILURES - 1
    db_session.commit()

    assert fail_task_command(command.command_id, "runner-a", "still broken") is True
    db_session.expire_all()
    row = db_session.get(TaskExecutionCommand, command.command_id)
    assert row is not None
    assert row.status == COMMAND_FAILED
    assert row.failure_count == MAX_COMMAND_FAILURES


def test_failed_command_retry_preserves_immutable_target(db_session) -> None:
    user, task = _create_running_task(db_session)
    enqueued = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="retry-terminal-command",
        kind=TaskCommandKind.CANCEL,
        payload={"agent_id": 1},
    )
    row = db_session.get(TaskExecutionCommand, enqueued.command_id)
    assert row is not None
    row.status = COMMAND_FAILED
    row.failure_count = MAX_COMMAND_FAILURES
    row.defer_count = max_command_defers()
    row.error = "temporary cancellation failure"
    row.completed_at = datetime.utcnow()
    db_session.commit()

    assert retry_failed_task_command(
        db_session,
        enqueued.command_id,
    )
    db_session.expire_all()
    row = db_session.get(TaskExecutionCommand, enqueued.command_id)
    assert row is not None
    assert row.status == "pending"
    assert row.failure_count == 0
    assert row.defer_count == 0
    assert row.error is None
    assert row.completed_at is None
    assert row.target_run_id == "run-1"
    assert row.target_runner_id == "runner-a"


@pytest.mark.asyncio
async def test_recovery_dispatches_committed_message_across_run_rotation(
    db_session,
) -> None:
    user, task = _create_running_task(db_session)
    task.input = "already committed"
    enqueued = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="committed-turn",
        kind=TaskCommandKind.MESSAGE,
        payload={
            "type": "chat",
            "message": "already committed",
            "client_message_id": "committed-turn",
            "files": [],
        },
    )
    first_claim = claim_task_command(db_session, runner_id="runner-a")
    assert first_claim is not None
    db_session.add(
        TaskChatMessage(
            task_id=int(task.id),
            user_id=int(user.id),
            role="user",
            content="already committed",
            message_type="user_message",
            turn_id="committed-turn",
            delivery_status="pending",
        )
    )
    row = db_session.query(TaskExecutionCommand).filter_by(id=enqueued.command_id).one()
    row.claim_expires_at = datetime.utcnow() - timedelta(seconds=1)
    # MESSAGE commands represent user intent for the task rather than a control
    # mutation on one run, so recovery deliberately applies them after rotation.
    task.run_id = "run-2"
    task.lease_expires_at = datetime.utcnow() - timedelta(seconds=1)
    db_session.commit()

    runtime_agent = MagicMock()
    runtime_agent.supports_live_control.return_value = True
    runtime_agent.post_user_message = AsyncMock(
        return_value=UserMessageInjectionOutcome.POSTED_FRESH
    )
    runtime_manager = MagicMock(
        get_agent_for_task=AsyncMock(return_value=runtime_agent)
    )
    resume = AsyncMock()

    with (
        patch(
            "xagent.web.api.chat.get_agent_manager",
            return_value=runtime_manager,
        ),
        patch.object(websocket_api, "execute_resume_background", new=resume),
    ):
        assert await dispatch_one_task_command(
            execute_durable_task_command,
            command_db_id=enqueued.command_id,
        )
        resume_task = websocket_api.background_task_manager.resume_tasks.get(
            int(task.id)
        )
        assert resume_task is not None
        await resume_task
        websocket_api.background_task_manager.cleanup_task(
            int(task.id),
            expected_task=resume_task,
        )

    db_session.expire_all()
    messages = (
        db_session.query(TaskChatMessage)
        .filter_by(task_id=int(task.id), turn_id="committed-turn")
        .all()
    )
    assert len(messages) == 1
    assert messages[0].delivery_status == "dispatched"
    command = db_session.get(TaskExecutionCommand, enqueued.command_id)
    assert command is not None
    assert command.status == COMMAND_COMPLETED
    assert task.run_id == "run-2"
    runtime_agent.post_user_message.assert_awaited_once()
    assert runtime_agent.post_user_message.await_args.kwargs["turn_id"] == (
        "committed-turn"
    )
    resume.assert_awaited_once()
    assert resume.await_args.kwargs["expected_run_id"] == "run-2"


@pytest.mark.asyncio
async def test_dispatcher_recovers_command_that_predates_worker_start(
    db_session,
) -> None:
    user, task = _create_running_task(db_session)
    task.runner_id = None
    task.lease_expires_at = None
    db_session.commit()
    enqueued = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="startup-recovery",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
    )
    applied = asyncio.Event()

    async def execute(command):
        assert command.id == enqueued.command_id
        applied.set()
        return None

    start_task_command_dispatcher(execute)
    try:
        await asyncio.wait_for(applied.wait(), timeout=2)
        for _ in range(100):
            db_session.expire_all()
            stored = db_session.get(TaskExecutionCommand, enqueued.command_id)
            if stored is not None and stored.status == COMMAND_COMPLETED:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("dispatcher did not complete recovered command")
    finally:
        await stop_task_command_dispatcher()
    db_session.expire_all()
    stored = db_session.get(TaskExecutionCommand, enqueued.command_id)
    assert stored is not None
    assert stored.status == COMMAND_COMPLETED


@pytest.mark.asyncio
async def test_dispatcher_recovers_unrelated_tasks_concurrently(db_session) -> None:
    user, first_task = _create_running_task(db_session)
    first_task.runner_id = None
    first_task.lease_expires_at = None
    second_task = Task(
        user_id=user.id,
        title="Second durable command",
        description="Second durable command",
        status=TaskStatus.RUNNING,
        execution_mode="auto",
        run_id="run-2",
        runner_id=None,
        lease_expires_at=None,
    )
    db_session.add(second_task)
    db_session.commit()
    first = enqueue_task_command(
        db_session,
        task_id=int(first_task.id),
        actor_user_id=int(user.id),
        command_id="slow-recovery",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
    )
    second = enqueue_task_command(
        db_session,
        task_id=int(second_task.id),
        actor_user_id=int(user.id),
        command_id="independent-recovery",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
    )
    release_first = asyncio.Event()
    second_started = asyncio.Event()

    async def execute(command):
        if command.id == first.command_id:
            await release_first.wait()
        elif command.id == second.command_id:
            second_started.set()
        return None

    start_task_command_dispatcher(execute)
    try:
        # This is an anti-hang bound, not a latency assertion. The dispatcher
        # normally starts both tasks immediately, but the complete web suite
        # can heavily contend for CI workers and SQLite connections.
        await asyncio.wait_for(second_started.wait(), timeout=10)
        release_first.set()
    finally:
        release_first.set()
        await stop_task_command_dispatcher()


def test_load_task_command_returns_an_immutable_primitive_snapshot(db_session) -> None:
    user, task = _create_running_task(db_session)
    task.runner_id = None
    task.lease_expires_at = None
    db_session.commit()
    enqueued = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="detached-status",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
    )
    row = db_session.get(TaskExecutionCommand, enqueued.command_id)
    assert row is not None
    row.attempt_count = 3
    row.failure_count = 2
    row.defer_count = 1
    row.result = {"rejection_reason": "stale_run", "mutable": ["ignored"]}
    db_session.commit()

    stored = load_task_command(enqueued.command_id)

    assert stored is not None
    assert is_dataclass(stored)
    assert not isinstance(stored, TaskExecutionCommand)
    assert stored.command_db_id == enqueued.command_id
    assert stored.status == "pending"
    assert stored.error is None
    assert stored.rejection_reason == "stale_run"
    assert stored.result == {"rejection_reason": "stale_run"}
    stored.result["rejection_reason"] = "mutated"
    assert stored.result == {"rejection_reason": "stale_run"}
    assert stored.attempt_count == 3
    assert stored.failure_count == 2
    assert stored.defer_count == 1
    with pytest.raises(FrozenInstanceError):
        stored.status = "failed"  # type: ignore[misc]


def test_legacy_delivery_status_preserves_none(db_session) -> None:
    user, task = _create_running_task(db_session)
    db_session.add(
        TaskChatMessage(
            task_id=int(task.id),
            user_id=int(user.id),
            role="user",
            content="legacy",
            message_type="user_message",
            turn_id="legacy-delivery",
            delivery_status=None,
        )
    )
    db_session.commit()

    assert (
        _load_command_message_delivery_status(int(task.id), "legacy-delivery") is None
    )


@pytest.mark.asyncio
async def test_dispatch_claim_pool_wait_does_not_block_event_loop(
    queue_pool_command_db,
    monkeypatch,
) -> None:
    engine, SessionLocal = queue_pool_command_db
    with SessionLocal() as seed_db:
        user, task = _create_running_task(seed_db)
        task.runner_id = None
        task.lease_expires_at = None
        seed_db.commit()
        enqueued = enqueue_task_command(
            seed_db,
            task_id=int(task.id),
            actor_user_id=int(user.id),
            command_id="contended-dispatch-claim",
            kind=TaskCommandKind.PAUSE,
            payload={"type": "pause_task"},
        )

    session_threads: list[int] = []

    def session_factory():
        session_threads.append(get_ident())
        return SessionLocal()

    monkeypatch.setattr(database_module, "get_session_local", lambda: session_factory)

    held_connection = engine.connect()
    loop_thread = get_ident()
    ticker_stop = asyncio.Event()
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while not ticker_stop.is_set():
            await asyncio.sleep(0.01)
            ticks += 1

    async def execute(command: ClaimedTaskCommand) -> None:
        assert isinstance(command, ClaimedTaskCommand)

    with gated_pool_checkout(engine) as gate:
        ticker_task = asyncio.create_task(ticker())
        dispatch_task = asyncio.create_task(
            dispatch_one_task_command(execute, command_db_id=enqueued.command_id)
        )
        try:
            await gate.wait_until_contending()
            observed = await wait_for_ticks(lambda: ticks)
            assert observed >= LOOP_LIVENESS_TICKS, (
                "command claim QueuePool checkout blocked the event loop"
            )
            assert not dispatch_task.done()
        finally:
            held_connection.close()
            gate.let_through()
            ticker_stop.set()
            await asyncio.wait_for(
                asyncio.gather(dispatch_task, ticker_task, return_exceptions=True),
                timeout=GUARD_TIMEOUT,
            )

    assert dispatch_task.result()
    assert session_threads
    assert session_threads[0] != loop_thread


@pytest.mark.asyncio
async def test_command_handler_pool_timeout_does_not_checkout_again(
    queue_pool_command_db,
    monkeypatch,
    caplog,
) -> None:
    """A handler checkout timeout leaves its durable claim for expiry/retry."""
    engine, SessionLocal = queue_pool_command_db
    with SessionLocal() as seed_db:
        user, task = _create_running_task(seed_db)
        task.runner_id = None
        task.lease_expires_at = None
        seed_db.commit()
        enqueued = enqueue_task_command(
            seed_db,
            task_id=int(task.id),
            actor_user_id=int(user.id),
            command_id="handler-pool-timeout",
            kind=TaskCommandKind.RESUME,
            payload={"type": "resume_task"},
        )
        task_id = int(task.id)

    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)
    held_connections = []
    ticker_stop = asyncio.Event()
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while not ticker_stop.is_set():
            ticks += 1
            await asyncio.sleep(0.01)

    def checkout_from_exhausted_pool() -> None:
        with SessionLocal() as db:
            db.execute(text("SELECT 1")).scalar()

    async def execute(_command: ClaimedTaskCommand) -> None:
        held_connections.append(engine.connect())
        await asyncio.to_thread(checkout_from_exhausted_pool)

    caplog.set_level(
        logging.ERROR,
        logger="xagent.web.services.task_command_transport",
    )
    ticker_task = asyncio.create_task(ticker())
    try:
        assert await dispatch_one_task_command(
            execute,
            command_db_id=enqueued.command_id,
        )
        assert ticks >= 10, (
            "command handler QueuePool timeout blocked the event loop; "
            f"ticker advanced only {ticks} times"
        )
    finally:
        ticker_stop.set()
        await ticker_task
        for connection in held_connections:
            connection.close()

    with SessionLocal() as verify_db:
        stored = verify_db.get(TaskExecutionCommand, enqueued.command_id)
        assert stored is not None
        assert stored.status == "processing"
        assert stored.claim_expires_at is not None

    assert f"task_id={task_id}" in caplog.text
    assert "component=task-command-handler" in caplog.text
    assert "retaining command claim for expiry" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("executor_outcome", "disposition_name"),
    [
        ("success", "finish_task_command"),
        ("failure", "fail_task_command"),
        ("deferred", "defer_task_command"),
    ],
)
async def test_final_disposition_pool_timeout_retains_claim_for_ttl(
    db_session,
    monkeypatch,
    caplog,
    executor_outcome: str,
    disposition_name: str,
) -> None:
    """A final write timeout must not trigger another checkout or lose fencing."""

    user, task = _create_running_task(db_session)
    task.runner_id = None
    task.lease_expires_at = None
    db_session.commit()
    enqueued = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id=f"{executor_outcome}-disposition-timeout",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
    )
    disposition_calls: list[tuple[str, str, int]] = []

    def pool_timeout(*args, **kwargs) -> bool:
        disposition_calls.append(
            (
                disposition_name,
                str(args[1]),
                int(kwargs["expected_attempt_count"]),
            )
        )
        raise SQLAlchemyTimeoutError("synthetic final disposition pool timeout")

    def unexpected_disposition(*_args, **_kwargs) -> bool:
        raise AssertionError("pool timeout triggered a second disposition checkout")

    for candidate in (
        "finish_task_command",
        "fail_task_command",
        "defer_task_command",
    ):
        monkeypatch.setattr(
            task_command_transport_module,
            candidate,
            pool_timeout if candidate == disposition_name else unexpected_disposition,
        )

    async def execute(_command: ClaimedTaskCommand) -> None:
        if executor_outcome == "failure":
            raise RuntimeError("executor failed")
        if executor_outcome == "deferred":
            raise TaskCommandDeferred("handoff not ready")

    caplog.set_level(
        logging.ERROR,
        logger="xagent.web.services.task_command_transport",
    )

    assert await dispatch_one_task_command(
        execute,
        command_db_id=enqueued.command_id,
    )

    assert len(disposition_calls) == 1
    observed_disposition, observed_runner_id, observed_attempt = disposition_calls[0]
    assert observed_disposition == disposition_name
    assert observed_attempt == 1
    db_session.expire_all()
    stored = db_session.get(TaskExecutionCommand, enqueued.command_id)
    assert stored is not None
    assert stored.status == "processing"
    assert stored.claimed_by == observed_runner_id
    assert stored.claim_expires_at is not None
    assert stored.attempt_count == 1
    assert "component=task-command-disposition" in caplog.text
    assert f"disposition={disposition_name}" in caplog.text


@pytest.mark.asyncio
async def test_real_final_disposition_pool_timeout_stays_off_event_loop(
    queue_pool_command_db,
    monkeypatch,
    caplog,
) -> None:
    """A real one-slot pool timeout leaves the exact processing claim intact."""

    engine, SessionLocal = queue_pool_command_db
    with SessionLocal() as seed_db:
        user, task = _create_running_task(seed_db)
        task.runner_id = None
        task.lease_expires_at = None
        seed_db.commit()
        enqueued = enqueue_task_command(
            seed_db,
            task_id=int(task.id),
            actor_user_id=int(user.id),
            command_id="real-finish-disposition-timeout",
            kind=TaskCommandKind.PAUSE,
            payload={"type": "pause_task"},
        )

    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)
    held_connections = []
    ticker_stop = asyncio.Event()
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while not ticker_stop.is_set():
            ticks += 1
            await asyncio.sleep(0.01)

    async def execute(_command: ClaimedTaskCommand) -> dict[str, bool]:
        held_connections.append(engine.connect())
        return {"ok": True}

    caplog.set_level(
        logging.ERROR,
        logger="xagent.web.services.task_command_transport",
    )
    ticker_task = asyncio.create_task(ticker())
    try:
        assert await dispatch_one_task_command(
            execute,
            command_db_id=enqueued.command_id,
        )
        assert ticks >= 10, (
            "final disposition QueuePool timeout blocked the event loop; "
            f"ticker advanced only {ticks} times"
        )
    finally:
        ticker_stop.set()
        await ticker_task
        for connection in held_connections:
            connection.close()

    with SessionLocal() as verify_db:
        stored = verify_db.get(TaskExecutionCommand, enqueued.command_id)
        assert stored is not None
        assert stored.status == "processing"
        assert stored.claimed_by is not None
        assert stored.claim_expires_at is not None
        assert stored.attempt_count == 1

    assert "component=task-command-disposition" in caplog.text
    assert "disposition=finish_task_command" in caplog.text
    assert "retaining command claim for expiry" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("heartbeat_failure", ["pool_timeout", "claim_lost"])
async def test_dispatch_skips_disposition_after_unhealthy_heartbeat(
    db_session,
    monkeypatch,
    caplog,
    heartbeat_failure: str,
) -> None:
    """An unresolved heartbeat cannot be followed by another DB checkout."""

    user, task = _create_running_task(db_session)
    task.runner_id = None
    task.lease_expires_at = None
    db_session.commit()
    enqueued = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id=f"heartbeat-{heartbeat_failure}",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
    )

    class HeartbeatOutcome:
        claim_lost = heartbeat_failure == "claim_lost"
        pool_timeout = (
            SQLAlchemyTimeoutError("heartbeat pool timeout")
            if heartbeat_failure == "pool_timeout"
            else None
        )
        requires_ttl_recovery = claim_lost or pool_timeout is not None

    async def unhealthy_heartbeat(
        _command_db_id: int,
        _runner_id: str,
        _attempt_count: int,
        stop_event: asyncio.Event,
    ) -> HeartbeatOutcome:
        await stop_event.wait()
        return HeartbeatOutcome()

    def unexpected_disposition(*_args, **_kwargs) -> bool:
        raise AssertionError("unhealthy heartbeat triggered a final checkout")

    monkeypatch.setattr(
        task_command_transport_module,
        "_claim_heartbeat",
        unhealthy_heartbeat,
    )
    for candidate in (
        "finish_task_command",
        "fail_task_command",
        "defer_task_command",
    ):
        monkeypatch.setattr(
            task_command_transport_module,
            candidate,
            unexpected_disposition,
        )
    caplog.set_level(
        logging.ERROR,
        logger="xagent.web.services.task_command_transport",
    )

    assert await dispatch_one_task_command(
        lambda _command: asyncio.sleep(0),
        command_db_id=enqueued.command_id,
    )

    db_session.expire_all()
    stored = db_session.get(TaskExecutionCommand, enqueued.command_id)
    assert stored is not None
    assert stored.status == "processing"
    assert stored.claimed_by is not None
    assert stored.claim_expires_at is not None
    assert "component=task-command-heartbeat" in caplog.text
    assert "retaining command claim for expiry" in caplog.text


@pytest.mark.asyncio
async def test_dispatcher_worker_isolates_one_command_error(
    monkeypatch,
    caplog,
) -> None:
    """One command failure must not terminate its long-lived dispatcher worker."""

    recovered = asyncio.Event()
    calls = 0

    async def dispatch(_executor, *, command_db_id=None) -> bool:
        nonlocal calls
        del command_db_id
        calls += 1
        if calls == 1:
            raise RuntimeError("single command failure")
        recovered.set()
        return False

    monkeypatch.setattr(
        task_command_transport_module,
        "dispatch_one_task_command",
        dispatch,
    )
    monkeypatch.setattr(
        task_command_transport_module,
        "_dispatcher_wakeup",
        asyncio.Event(),
    )
    monkeypatch.setattr(
        task_command_transport_module,
        "DISPATCHER_IDLE_SECONDS",
        0.01,
    )
    caplog.set_level(
        logging.ERROR,
        logger="xagent.web.services.task_command_transport",
    )
    worker = asyncio.create_task(
        task_command_transport_module._run_task_command_dispatcher_worker(
            lambda _command: asyncio.sleep(0)
        )
    )
    try:
        await asyncio.wait_for(recovered.wait(), timeout=0.25)
    finally:
        worker.cancel()

    with pytest.raises(asyncio.CancelledError):
        await worker
    assert calls >= 2
    assert "component=task-command-dispatcher" in caplog.text
    assert "single command failure" in caplog.text


@pytest.mark.asyncio
async def test_dispatch_cancellation_drains_inflight_completion_worker(
    db_session,
    monkeypatch,
) -> None:
    user, task = _create_running_task(db_session)
    task.runner_id = None
    task.lease_expires_at = None
    db_session.commit()
    enqueued = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="cancel-during-command-finish",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
    )
    finish_started = Event()
    allow_finish = Event()
    finish_completed = Event()

    def blocking_finish(*_args, **_kwargs) -> bool:
        finish_started.set()
        assert allow_finish.wait(timeout=2)
        finish_completed.set()
        return True

    monkeypatch.setattr(
        "xagent.web.services.task_command_transport.finish_task_command",
        blocking_finish,
    )

    async def execute(_command: ClaimedTaskCommand) -> None:
        return None

    dispatch_task = asyncio.create_task(
        dispatch_one_task_command(execute, command_db_id=enqueued.command_id)
    )
    await asyncio.wait_for(asyncio.to_thread(finish_started.wait, 1), timeout=1)
    dispatch_task.cancel()
    try:
        await asyncio.sleep(0.02)
        assert not dispatch_task.done()
    finally:
        allow_finish.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(dispatch_task, timeout=1)
    assert finish_completed.is_set()


@pytest.mark.asyncio
async def test_dispatch_cancellation_during_heartbeat_persists_completion(
    db_session,
    monkeypatch,
) -> None:
    user, task = _create_running_task(db_session)
    task.runner_id = None
    task.lease_expires_at = None
    db_session.commit()
    enqueued = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="cancel-while-stopping-command-heartbeat",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
    )
    heartbeat_stopping = asyncio.Event()
    allow_heartbeat_to_finish = asyncio.Event()

    async def delayed_heartbeat(
        _command_db_id: int,
        _runner_id: str,
        _attempt_count: int,
        stop_event: asyncio.Event,
    ):
        await stop_event.wait()
        heartbeat_stopping.set()
        await allow_heartbeat_to_finish.wait()
        return task_command_transport_module.TaskCommandClaimHeartbeatOutcome()

    monkeypatch.setattr(
        task_command_transport_module,
        "_claim_heartbeat",
        delayed_heartbeat,
    )

    dispatch = asyncio.create_task(
        dispatch_one_task_command(
            lambda _command: asyncio.sleep(0, result={"ok": True}),
            command_db_id=enqueued.command_id,
        )
    )
    await heartbeat_stopping.wait()
    dispatch.cancel()
    await asyncio.sleep(0)
    assert not dispatch.done()

    allow_heartbeat_to_finish.set()
    with pytest.raises(asyncio.CancelledError):
        await dispatch

    db_session.expire_all()
    stored = db_session.get(TaskExecutionCommand, enqueued.command_id)
    assert stored is not None
    assert stored.status == COMMAND_COMPLETED
    assert stored.result == {"ok": True}


@pytest.mark.asyncio
async def test_dispatch_cancellation_during_heartbeat_preserves_late_write_error(
    db_session,
    monkeypatch,
) -> None:
    user, task = _create_running_task(db_session)
    task.runner_id = None
    task.lease_expires_at = None
    db_session.commit()
    enqueued = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="cancel-before-command-write-error",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
    )
    heartbeat_stopping = asyncio.Event()
    allow_heartbeat_to_finish = asyncio.Event()
    persistence_error = RuntimeError("disposition write failed")

    async def delayed_heartbeat(
        _command_db_id: int,
        _runner_id: str,
        _attempt_count: int,
        stop_event: asyncio.Event,
    ):
        await stop_event.wait()
        heartbeat_stopping.set()
        await allow_heartbeat_to_finish.wait()
        return task_command_transport_module.TaskCommandClaimHeartbeatOutcome()

    def failing_finish(*_args, **_kwargs) -> bool:
        raise persistence_error

    monkeypatch.setattr(
        task_command_transport_module,
        "_claim_heartbeat",
        delayed_heartbeat,
    )
    monkeypatch.setattr(
        task_command_transport_module,
        "finish_task_command",
        failing_finish,
    )

    dispatch = asyncio.create_task(
        dispatch_one_task_command(
            lambda _command: asyncio.sleep(0, result={"ok": True}),
            command_db_id=enqueued.command_id,
        )
    )
    await heartbeat_stopping.wait()
    dispatch.cancel()
    await asyncio.sleep(0)
    assert not dispatch.done()

    allow_heartbeat_to_finish.set()
    with pytest.raises(asyncio.CancelledError) as exc_info:
        await dispatch

    assert exc_info.value.__cause__ is persistence_error


@pytest.mark.asyncio
async def test_claim_heartbeat_cancellation_drains_inflight_renewal(
    monkeypatch,
) -> None:
    renew_started = Event()
    allow_renew = Event()
    renew_completed = Event()

    def blocking_renew(*_args, **_kwargs) -> bool:
        renew_started.set()
        assert allow_renew.wait(timeout=2)
        renew_completed.set()
        return True

    monkeypatch.setattr(
        "xagent.web.services.task_command_transport.get_task_lease_heartbeat_seconds",
        lambda: 0.001,
    )
    monkeypatch.setattr(
        "xagent.web.services.task_command_transport.renew_task_command_claim",
        blocking_renew,
    )

    heartbeat = asyncio.create_task(_claim_heartbeat(7, "runner-a", 1, asyncio.Event()))
    await asyncio.wait_for(asyncio.to_thread(renew_started.wait, 1), timeout=1)
    heartbeat.cancel()
    try:
        await asyncio.sleep(0.02)
        assert not heartbeat.done()
    finally:
        allow_renew.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(heartbeat, timeout=1)
    assert renew_completed.is_set()


@pytest.mark.asyncio
async def test_claim_heartbeat_reports_unresolved_pool_timeout(monkeypatch) -> None:
    stop_event = asyncio.Event()
    timeout = SQLAlchemyTimeoutError("heartbeat checkout timed out")

    def renew(*_args, **_kwargs) -> bool:
        stop_event.set()
        raise timeout

    monkeypatch.setattr(
        task_command_transport_module,
        "get_task_lease_heartbeat_seconds",
        lambda: 0.001,
    )
    monkeypatch.setattr(
        task_command_transport_module,
        "renew_task_command_claim",
        renew,
    )

    outcome = await _claim_heartbeat(7, "runner-a", 1, stop_event)

    assert outcome.claim_lost is False
    assert outcome.pool_timeout is timeout
    assert outcome.requires_ttl_recovery is True


@pytest.mark.asyncio
async def test_claim_heartbeat_clears_pool_timeout_after_success(monkeypatch) -> None:
    stop_event = asyncio.Event()
    attempts = 0

    def renew(*_args, **_kwargs) -> bool:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise SQLAlchemyTimeoutError("transient heartbeat checkout timeout")
        stop_event.set()
        return True

    monkeypatch.setattr(
        task_command_transport_module,
        "get_task_lease_heartbeat_seconds",
        lambda: 0.001,
    )
    monkeypatch.setattr(
        task_command_transport_module,
        "renew_task_command_claim",
        renew,
    )

    outcome = await _claim_heartbeat(7, "runner-a", 1, stop_event)

    assert attempts == 2
    assert outcome.claim_lost is False
    assert outcome.pool_timeout is None
    assert outcome.requires_ttl_recovery is False


@pytest.mark.asyncio
async def test_claim_heartbeat_survives_transient_database_error(
    monkeypatch,
) -> None:
    stop_event = asyncio.Event()
    attempts = 0

    def renew(
        _command_db_id: int,
        _runner_id: str,
        *,
        expected_attempt_count: int | None = None,
    ) -> bool:
        nonlocal attempts
        assert expected_attempt_count == 3
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary database outage")
        stop_event.set()
        return True

    monkeypatch.setattr(
        "xagent.web.services.task_command_transport.get_task_lease_heartbeat_seconds",
        lambda: 0.001,
    )
    monkeypatch.setattr(
        "xagent.web.services.task_command_transport.renew_task_command_claim",
        renew,
    )

    await asyncio.wait_for(_claim_heartbeat(7, "runner-a", 3, stop_event), timeout=0.2)

    assert attempts == 2


@pytest.mark.asyncio
async def test_dispatcher_does_not_erase_wakeup_during_empty_claim(monkeypatch) -> None:
    second_claim = asyncio.Event()
    calls = 0

    async def fake_dispatch(_executor, *, command_db_id=None) -> bool:
        nonlocal calls
        del command_db_id
        calls += 1
        if calls == 1:
            notify_task_command_dispatcher()
            return False
        second_claim.set()
        return False

    monkeypatch.setattr(
        "xagent.web.services.task_command_transport.dispatch_one_task_command",
        fake_dispatch,
    )

    start_task_command_dispatcher(lambda _command: asyncio.sleep(0))
    try:
        await asyncio.wait_for(second_claim.wait(), timeout=0.25)
    finally:
        await stop_task_command_dispatcher()

    assert calls >= 2


@pytest.mark.asyncio
async def test_prompt_dispatch_observes_late_task_failure(monkeypatch, caplog) -> None:
    async def fail_after_handoff(_executor, *, command_db_id=None) -> bool:
        del command_db_id
        await asyncio.sleep(0.06)
        raise RuntimeError("late dispatch failure")

    async def execute(_command):
        return None

    monkeypatch.setattr(
        "xagent.web.services.task_command_transport.dispatch_one_task_command",
        fail_after_handoff,
    )
    caplog.set_level(logging.ERROR)

    await dispatch_task_command_promptly(execute, command_db_id=1)
    for _ in range(100):
        if "Detached prompt task command dispatch failed" in caplog.text:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("late prompt dispatch failure was not observed")

    assert "late dispatch failure" in caplog.text


def test_enqueue_detects_a_task_deleted_after_the_caller_loaded_it(
    db_session,
) -> None:
    """A caller's earlier snapshot must not let a deleted task be enqueued."""

    user, task = _create_running_task(db_session)
    task_id = int(task.id)
    # The task id was validated earlier -- as the websocket transport does
    # between its permission check and this enqueue -- and the row is gone by
    # the time the command is written.
    db_session.query(Task).filter(Task.id == task_id).delete()
    db_session.commit()

    with pytest.raises(ValueError, match=f"Task {task_id} not found"):
        enqueue_task_command(
            db_session,
            task_id=task_id,
            actor_user_id=int(user.id),
            command_id="pause-after-delete",
            kind=TaskCommandKind.PAUSE,
            payload={"type": "pause_task"},
        )

    assert db_session.query(TaskExecutionCommand).count() == 0


def test_websocket_enqueue_returns_none_when_the_task_vanishes_mid_enqueue(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """allow_missing_task routes a mid-enqueue delete to the recovery sentinel.

    The task exists at the permission check but is gone by the time the command
    is written. The caller must get None so it creates a replacement task rather
    than rejecting the delivery.
    """

    user, task = _create_running_task(db_session)
    task_id = int(task.id)

    def raise_missing(*_args, **_kwargs):
        raise TaskCommandTaskMissing(f"Task {task_id} not found")

    monkeypatch.setattr(websocket_api, "enqueue_task_command", raise_missing)

    result = websocket_api._enqueue_websocket_task_command_sync(
        task_id=task_id,
        actor_user_id=int(user.id),
        actor_is_admin=False,
        command_id="pause-vanished",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
        allow_missing_task=True,
    )

    assert result is None


def test_websocket_enqueue_rejects_missing_task_with_client_visible_wording(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without allow_missing_task the delivery must be rejected, not silently
    swallowed as if the command had been accepted."""

    user, task = _create_running_task(db_session)
    task_id = int(task.id)

    def raise_missing(*_args, **_kwargs):
        raise TaskCommandTaskMissing(f"Task {task_id} not found")

    monkeypatch.setattr(websocket_api, "enqueue_task_command", raise_missing)

    # The rejection is re-raised as ClientVisibleValidationError at this
    # boundary (still a ValueError, so pause/resume reject it the same way)
    # so the redaction chokepoint keeps "not found" wording on this race
    # instead of flattening it to the generic string (#1514 round 5).
    with pytest.raises(ValueError) as raised:
        websocket_api._enqueue_websocket_task_command_sync(
            task_id=task_id,
            actor_user_id=int(user.id),
            actor_is_admin=False,
            command_id="pause-vanished-strict",
            kind=TaskCommandKind.PAUSE,
            payload={"type": "pause_task"},
            allow_missing_task=False,
        )
    assert isinstance(raised.value, websocket_api.ClientVisibleValidationError)
    assert str(raised.value) == f"Task {task_id} not found"


def test_task_foreign_key_violation_is_reported_as_a_missing_task(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the task really is gone, an IntegrityError with no duplicate row is
    a task FK failure. It must surface as a missing task so callers reach their
    recovery path, rather than raising NoResultFound from the duplicate lookup."""

    user, task = _create_running_task(db_session)
    task_id = int(task.id)

    real_flush = db_session.flush
    flush_calls = 0

    def flush_raising_fk_error(*args, **kwargs):
        # stage_task_command's owner pre-flush is a real, harmless no-op on
        # this clean session -- let it through so only the attributable
        # command-insert flush is intercepted below.
        nonlocal flush_calls
        flush_calls += 1
        # flush_calls is coupled to stage_task_command's exact two-flush
        # sequence -- owner pre-flush first, command-insert flush second.
        # If that sequence changes, this injection lands on the wrong flush.
        if flush_calls == 1:
            return real_flush(*args, **kwargs)
        db_session.flush = real_flush
        # Delete the task so the post-rollback recheck sees it as absent, the
        # way a concurrent delete would.
        db_session.rollback()
        db_session.query(Task).filter(Task.id == task_id).delete()
        db_session.commit()
        raise IntegrityError("INSERT", {}, Exception("FOREIGN KEY constraint failed"))

    monkeypatch.setattr(db_session, "flush", flush_raising_fk_error)

    with pytest.raises(ValueError, match=f"Task {task_id} not found"):
        enqueue_task_command(
            db_session,
            task_id=task_id,
            actor_user_id=int(user.id),
            command_id="pause-fk",
            kind=TaskCommandKind.PAUSE,
            payload={"type": "pause_task"},
        )


def test_actor_foreign_key_failure_is_not_reported_as_a_missing_task(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The row references both tasks and users. An IntegrityError with no
    duplicate command does not prove the task caused it: a concurrently deleted
    actor fails the users FK while the task is still present. That must not be
    translated into missing-task recovery, which would continue with a cached
    principal instead of reloading the actor."""

    user, task = _create_running_task(db_session)
    task_id = int(task.id)

    real_flush = db_session.flush
    flush_calls = 0

    def flush_raising_fk_error(*args, **kwargs):
        # stage_task_command's owner pre-flush is a real, harmless no-op on
        # this clean session -- let it through so only the attributable
        # command-insert flush is intercepted below.
        nonlocal flush_calls
        flush_calls += 1
        # flush_calls is coupled to stage_task_command's exact two-flush
        # sequence -- owner pre-flush first, command-insert flush second.
        # If that sequence changes, this injection lands on the wrong flush.
        if flush_calls == 1:
            return real_flush(*args, **kwargs)
        db_session.flush = real_flush
        raise IntegrityError("INSERT", {}, Exception("FOREIGN KEY constraint failed"))

    monkeypatch.setattr(db_session, "flush", flush_raising_fk_error)

    with pytest.raises(IntegrityError):
        enqueue_task_command(
            db_session,
            task_id=task_id,
            actor_user_id=int(user.id),
            command_id="pause-actor-fk",
            kind=TaskCommandKind.PAUSE,
            payload={"type": "pause_task"},
        )


# --- stage_task_command / classify_task_command_conflict (#1073) ---------


def test_stage_rejects_malformed_command_id_before_any_db_write(db_session) -> None:
    """A malformed command_id must fail before touching the database, so a
    caller cannot end up with a partially-written row for an id it rejects."""

    user, task = _create_running_task(db_session)
    task_id = int(task.id)

    with pytest.raises(ValueError, match="command_id must be 1-64"):
        stage_task_command(
            db_session,
            task_id=task_id,
            actor_user_id=int(user.id),
            command_id="bad id!",
            kind=TaskCommandKind.PAUSE,
            payload={"type": "pause_task"},
        )

    assert db_session.query(TaskExecutionCommand).count() == 0


def test_stage_strips_whitespace_from_the_staged_command_id(db_session) -> None:
    """A whitespace-padded command_id must stage a row carrying the stripped
    id, not the raw padded string."""

    user, task = _create_running_task(db_session)
    task_id = int(task.id)

    staged = stage_task_command(
        db_session,
        task_id=task_id,
        actor_user_id=int(user.id),
        command_id="  cmd-1  ",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
    )
    assert staged.client_command_id == "cmd-1"

    db_session.commit()
    row = db_session.get(TaskExecutionCommand, staged.staged_db_id)
    assert row is not None
    assert row.command_id == "cmd-1"


def test_stage_does_not_commit(db_session) -> None:
    """Staging must be observable only inside the caller's own session until
    it commits -- a second session must see nothing beforehand."""

    user, task = _create_running_task(db_session)
    task_id = int(task.id)

    staged = stage_task_command(
        db_session,
        task_id=task_id,
        actor_user_id=int(user.id),
        command_id="stage-visibility",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
    )
    assert staged.created is True

    SessionLocal = get_session_local()
    with SessionLocal() as observer:
        assert observer.get(TaskExecutionCommand, staged.staged_db_id) is None

    db_session.commit()

    with SessionLocal() as observer:
        row = observer.get(TaskExecutionCommand, staged.staged_db_id)
        assert row is not None
        assert row.status == COMMAND_PENDING


def test_stage_missing_task_leaves_no_command_pending(db_session) -> None:
    """Staging against a task deleted before the call must not add any
    command row to the session -- there is no write for a caller rollback to
    even need to undo."""

    user, task = _create_running_task(db_session)
    task_id = int(task.id)
    db_session.query(Task).filter(Task.id == task_id).delete()
    db_session.commit()

    with pytest.raises(TaskCommandTaskMissing):
        stage_task_command(
            db_session,
            task_id=task_id,
            actor_user_id=int(user.id),
            command_id="stage-missing-task",
            kind=TaskCommandKind.PAUSE,
            payload={"type": "pause_task"},
        )

    assert db_session.query(TaskExecutionCommand).count() == 0


def test_outer_rollback_leaves_no_command(db_session) -> None:
    """A caller that stages a command as part of a larger unit of work and
    then rolls back must lose the command along with the rest of that work --
    there is no SAVEPOINT isolating the command insert from the caller's own
    writes. The sibling write is created before staging begins; created after,
    this would only prove a single-row rollback rather than the caller's
    whole transaction being undone."""

    user, task = _create_running_task(db_session)
    task_id = int(task.id)
    task.title = "rolled-back-sibling-write"

    staged = stage_task_command(
        db_session,
        task_id=task_id,
        actor_user_id=int(user.id),
        command_id="outer-rollback",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
    )
    assert staged.created is True

    db_session.rollback()

    SessionLocal = get_session_local()
    with SessionLocal() as observer:
        observed_task = observer.get(Task, task_id)
        assert observed_task is not None
        assert observed_task.title != "rolled-back-sibling-write"
        assert (
            observer.query(TaskExecutionCommand)
            .filter(TaskExecutionCommand.task_id == task_id)
            .count()
            == 0
        )


def test_enqueue_idempotent_hit_issues_no_commit(db_session) -> None:
    """Only the created=True path may end the caller's transaction -- a
    duplicate command_id must leave an accompanying uncommitted write
    uncommitted, observed from a second session."""

    user, task = _create_running_task(db_session)
    first = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="no-commit-duplicate",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
    )
    assert first.created is True

    task.title = "uncommitted sibling write"
    duplicate = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="no-commit-duplicate",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
    )
    assert duplicate.created is False

    SessionLocal = get_session_local()
    with SessionLocal() as observer:
        observed_task = observer.get(Task, int(task.id))
        assert observed_task is not None
        assert observed_task.title != "uncommitted sibling write"

    # stage_task_command's owner pre-flush already pushed the title change
    # above into the open transaction, so it no longer shows up as dirty by
    # this point -- a fresh, still-unflushed write on the same session is
    # what proves the session was left healthy rather than silently rolled
    # back by the idempotent-hit path.
    sibling_user = User(
        username="idempotent-hit-sibling-write",
        password_hash="hash",
        is_admin=False,
    )
    db_session.add(sibling_user)
    assert sibling_user in db_session.new

    db_session.commit()

    with SessionLocal() as observer:
        observed_task = observer.get(Task, int(task.id))
        assert observed_task is not None
        assert observed_task.title == "uncommitted sibling write"
        assert (
            observer.query(User)
            .filter(User.username == "idempotent-hit-sibling-write")
            .first()
            is not None
        )


def test_enqueue_missing_task_issues_no_commit(db_session) -> None:
    """A missing task must not commit -- an accompanying pending write on the
    same session must still be observed as uncommitted by another session."""

    user, task = _create_running_task(db_session)
    task_id = int(task.id)
    db_session.query(Task).filter(Task.id == task_id).delete()
    db_session.add(
        User(username="no-commit-sibling-write", password_hash="hash", is_admin=False)
    )

    with pytest.raises(TaskCommandTaskMissing):
        enqueue_task_command(
            db_session,
            task_id=task_id,
            actor_user_id=int(user.id),
            command_id="no-commit-missing-task",
            kind=TaskCommandKind.PAUSE,
            payload={"type": "pause_task"},
        )

    SessionLocal = get_session_local()
    with SessionLocal() as observer:
        assert (
            observer.query(User)
            .filter(User.username == "no-commit-sibling-write")
            .first()
            is None
        )


def test_owner_pre_flush_failure_raises_owner_state_error(db_session) -> None:
    """A conflict on the caller's own pending write, surfaced by staging's
    pre-flush, must be reported apart from a conflict on the command insert
    itself so the two are never confused."""

    user, task = _create_running_task(db_session)
    task_id = int(task.id)
    db_session.add(User(username=user.username, password_hash="hash", is_admin=False))

    with pytest.raises(TaskCommandOwnerStateError):
        stage_task_command(
            db_session,
            task_id=task_id,
            actor_user_id=int(user.id),
            command_id="owner-state-conflict",
            kind=TaskCommandKind.PAUSE,
            payload={"type": "pause_task"},
        )


def test_owner_state_error_is_not_an_integrity_error(db_session) -> None:
    """Pinned deliberately: TaskCommandOwnerStateError must not be an
    IntegrityError subclass, or enqueue_task_command's
    ``except IntegrityError`` would wrongly route an owner-write failure
    through duplicate/missing-task classification."""

    assert not issubclass(TaskCommandOwnerStateError, IntegrityError)


def test_enqueue_does_not_catch_owner_state_error(db_session) -> None:
    """enqueue_task_command's except IntegrityError must not swallow a
    pre-flush failure on the caller's own pending write."""

    user, task = _create_running_task(db_session)
    task_id = int(task.id)
    db_session.add(User(username=user.username, password_hash="hash", is_admin=False))

    with pytest.raises(TaskCommandOwnerStateError):
        enqueue_task_command(
            db_session,
            task_id=task_id,
            actor_user_id=int(user.id),
            command_id="owner-state-conflict-wrapper",
            kind=TaskCommandKind.PAUSE,
            payload={"type": "pause_task"},
        )


def test_stage_snapshot_sees_a_bulk_cas_update_pending_in_the_same_session(
    db_session,
) -> None:
    """A CAS-then-stage owner updates the row via a synchronize_session=False
    bulk statement, which executes immediately. The staging snapshot must
    observe it -- an ORM query here would instead return the identity map's
    pre-update cached instance."""

    user, task = _create_running_task(db_session)
    task_id = int(task.id)

    updated = (
        db_session.query(Task)
        .filter(Task.id == task_id, Task.runner_id == "runner-a")
        .update({Task.runner_id: "runner-b"}, synchronize_session=False)
    )
    assert updated == 1

    staged = stage_task_command(
        db_session,
        task_id=task_id,
        actor_user_id=int(user.id),
        command_id="cas-bulk-update",
        kind=TaskCommandKind.CANCEL,
        payload={"agent_id": 1},
    )
    db_session.commit()

    row = db_session.get(TaskExecutionCommand, staged.staged_db_id)
    assert row is not None
    assert row.target_runner_id == "runner-b"


def test_stage_snapshot_sees_an_orm_attribute_change_pending_in_the_same_session(
    db_session,
) -> None:
    """A CAS-then-stage owner instead mutates the loaded ORM instance rather
    than issuing a bulk update. With autoflush disabled this is invisible to
    a raw Core select until something flushes it -- staging's own owner
    pre-flush must be what surfaces it, not the identity map."""

    user, task = _create_running_task(db_session)
    task_id = int(task.id)

    task.runner_id = "runner-c"
    # No explicit flush here: stage_task_command's own pre-flush must be what
    # pushes this to the database before its Core select runs.

    staged = stage_task_command(
        db_session,
        task_id=task_id,
        actor_user_id=int(user.id),
        command_id="cas-orm-attribute",
        kind=TaskCommandKind.CANCEL,
        payload={"agent_id": 1},
    )
    db_session.commit()

    row = db_session.get(TaskExecutionCommand, staged.staged_db_id)
    assert row is not None
    assert row.target_runner_id == "runner-c"


def test_interleaved_duplicate_enqueue_reports_the_committed_winner(
    db_session,
) -> None:
    """Two real sessions/threads both go through the public enqueue_task_command
    entry point. B's idempotency precheck runs and finds nothing before A's
    insert commits, so B's own insert genuinely races A's at the flush --
    unlike a recipe where A commits first and B's precheck simply finds the
    row, this exercises the IntegrityError-then-rollback-then-classify path
    with a real concurrent writer."""

    user, task = _create_running_task(db_session)
    task_id = int(task.id)
    actor_id = int(user.id)
    SessionLocal = get_session_local()

    precheck_done = Event()
    release_b = Event()

    def enqueue_as_b():
        with SessionLocal() as db_b:
            real_add = db_b.add

            def paused_add(instance):
                if isinstance(instance, TaskExecutionCommand):
                    precheck_done.set()
                    assert release_b.wait(timeout=10)
                real_add(instance)

            db_b.add = paused_add
            return enqueue_task_command(
                db_b,
                task_id=task_id,
                actor_user_id=actor_id,
                command_id="interleaved-duplicate",
                kind=TaskCommandKind.PAUSE,
                payload={"type": "pause_task"},
            )

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(enqueue_as_b)
        assert precheck_done.wait(timeout=10)

        with SessionLocal() as db_a:
            winner = enqueue_task_command(
                db_a,
                task_id=task_id,
                actor_user_id=actor_id,
                command_id="interleaved-duplicate",
                kind=TaskCommandKind.PAUSE,
                payload={"type": "pause_task"},
            )
        release_b.set()
        loser = future.result(timeout=10)

    assert winner.created is True
    assert loser.created is False
    assert loser.command_id == winner.command_id
    assert loser.payload_matches is True
    assert loser.status == winner.status


@pytest.mark.postgresql
def test_postgres_concurrent_insert_blocks_until_the_winner_resolves(
    postgres_task_command_sessions,
) -> None:
    """A genuinely-concurrent variant of the raced-duplicate test: both
    writers' inserts are open against PostgreSQL at the same instant, with
    B's insert blocked on A's uncommitted row rather than merely interleaved
    by a test-code pause. Requires XAGENT_TEST_POSTGRES_URL; skipped
    otherwise."""

    SessionLocal, task_id, actor_id = postgres_task_command_sessions
    a_flushed = Event()
    allow_a_commit = Event()
    b_blocked_then_resolved = Event()

    def run_a():
        with SessionLocal() as db_a:
            staged = stage_task_command(
                db_a,
                task_id=task_id,
                actor_user_id=actor_id,
                command_id="pg-real-race",
                kind=TaskCommandKind.PAUSE,
                payload={"type": "pause_task"},
            )
            a_flushed.set()
            assert allow_a_commit.wait(timeout=10)
            db_a.commit()
            return staged

    def run_b():
        assert a_flushed.wait(timeout=10)
        with SessionLocal() as db_b:
            try:
                stage_task_command(
                    db_b,
                    task_id=task_id,
                    actor_user_id=actor_id,
                    command_id="pg-real-race",
                    kind=TaskCommandKind.PAUSE,
                    payload={"type": "pause_task"},
                )
            except IntegrityError:
                db_b.rollback()
                classification = classify_task_command_conflict(
                    db_b,
                    task_id=task_id,
                    command_id="pg-real-race",
                    actor_user_id=actor_id,
                    kind=TaskCommandKind.PAUSE,
                    payload={"type": "pause_task"},
                )
                b_blocked_then_resolved.set()
                return classification
            raise AssertionError("B's insert should have raced A's committed one")

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(run_a)
        future_b = executor.submit(run_b)
        assert a_flushed.wait(timeout=10)
        # Give B's flush time to genuinely be blocked inside PostgreSQL,
        # rather than releasing A the instant B's thread has merely started.
        time.sleep(0.2)
        allow_a_commit.set()
        staged_a = future_a.result(timeout=10)
        classification = future_b.result(timeout=10)

    assert b_blocked_then_resolved.is_set()
    assert classification.kind is TaskCommandConflictKind.RACED_DUPLICATE
    assert classification.raced is not None
    assert classification.raced.command_db_id == staged_a.staged_db_id


@pytest.mark.asyncio
async def test_dispatch_with_staged_id_before_commit_is_noop_and_converges_after_commit(
    db_session,
) -> None:
    """A staged id must not be dispatchable before its owning transaction
    commits -- the dispatcher claims through its own isolated session, which
    cannot see an uncommitted row. Once the owner commits without notifying,
    durable polling converges on it within one idle cycle. The target task
    has no earlier in-flight command, so unfinished-earlier-command ordering
    cannot also explain the delay."""

    user, task = _create_running_task(db_session)
    task.runner_id = None
    task.lease_expires_at = None
    db_session.commit()
    task_id = int(task.id)

    staged = stage_task_command(
        db_session,
        task_id=task_id,
        actor_user_id=int(user.id),
        command_id="staged-before-commit",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
    )

    async def must_not_dispatch(_command):
        raise AssertionError("must not dispatch an uncommitted staged command")

    assert not await dispatch_one_task_command(
        must_not_dispatch, command_db_id=staged.staged_db_id
    )

    db_session.commit()  # deliberately no notify_task_command_dispatcher() call

    applied = asyncio.Event()

    async def execute_after_commit(command):
        assert command.id == staged.staged_db_id
        applied.set()
        return None

    start_task_command_dispatcher(execute_after_commit)
    try:
        await asyncio.wait_for(applied.wait(), timeout=DISPATCHER_IDLE_SECONDS + 1.5)
    finally:
        await stop_task_command_dispatcher()


def test_owner_holding_a_flushed_command_blocks_a_concurrent_claim(
    low_timeout_sqlite_engine,
) -> None:
    """Staging is the last write before the owner's commit, so the row it
    flushes stays under SQLite's writer lock for the rest of the owner's
    transaction -- far longer than the microseconds an insert that committed
    itself holds it for. SQLite's writer lock is database-wide, so a concurrent claim
    of a different, already-committed command must still wait out
    busy_timeout and fail with OperationalError rather than hang."""

    engine, SessionLocal = low_timeout_sqlite_engine
    with SessionLocal() as setup_db:
        user, task = _create_running_task(setup_db)
        task.runner_id = None
        task.lease_expires_at = None
        setup_db.commit()
        task_id = int(task.id)
        actor_id = int(user.id)
        claimable = enqueue_task_command(
            setup_db,
            task_id=task_id,
            actor_user_id=actor_id,
            command_id="lock-path-claimable",
            kind=TaskCommandKind.PAUSE,
            payload={"type": "pause_task"},
        )

    with SessionLocal() as owner_db:
        stage_task_command(
            owner_db,
            task_id=task_id,
            actor_user_id=actor_id,
            command_id="lock-path-owner-holds",
            kind=TaskCommandKind.CANCEL,
            payload={"agent_id": 1},
        )
        # Deliberately not committed yet -- this is the window under test.
        started = time.monotonic()
        with SessionLocal() as claimant_db:
            with pytest.raises(OperationalError):
                claim_task_command(
                    claimant_db,
                    runner_id="lock-path-claimant",
                    command_db_id=claimable.command_id,
                )
        elapsed = time.monotonic() - started
        owner_db.commit()

    assert elapsed < 2.0


@pytest.mark.asyncio
async def test_dispatcher_worker_recovers_from_a_lock_timeout_and_claims_once_released(
    low_timeout_sqlite_engine,
    monkeypatch,
    caplog,
) -> None:
    """The dispatcher worker pool must survive the OperationalError above --
    logging it and continuing to poll -- and claim the command once the
    owner's transaction ends, instead of the worker dying or hanging."""

    engine, SessionLocal = low_timeout_sqlite_engine
    with SessionLocal() as seed_db:
        user, task = _create_running_task(seed_db)
        task.runner_id = None
        task.lease_expires_at = None
        seed_db.commit()
        task_id = int(task.id)
        actor_id = int(user.id)
        enqueued = enqueue_task_command(
            seed_db,
            task_id=task_id,
            actor_user_id=actor_id,
            command_id="dispatcher-progress-target",
            kind=TaskCommandKind.PAUSE,
            payload={"type": "pause_task"},
        )

    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)
    monkeypatch.setattr(task_command_transport_module, "DISPATCHER_IDLE_SECONDS", 0.01)
    monkeypatch.setattr(
        task_command_transport_module, "_dispatcher_wakeup", asyncio.Event()
    )

    owner_db = SessionLocal()
    stage_task_command(
        owner_db,
        task_id=task_id,
        actor_user_id=actor_id,
        command_id="dispatcher-progress-lock-holder",
        kind=TaskCommandKind.CANCEL,
        payload={"agent_id": 1},
    )
    # Deliberately uncommitted -- holds SQLite's writer lock across the
    # dispatcher worker's first claim attempt.

    caplog.set_level(logging.ERROR, logger="xagent.web.services.task_command_transport")
    applied = asyncio.Event()

    async def execute(command):
        assert command.id == enqueued.command_id
        applied.set()
        return None

    worker = asyncio.create_task(
        task_command_transport_module._run_task_command_dispatcher_worker(execute)
    )
    try:
        # Wait for the worker to actually observe the lock timeout (rather
        # than a fixed sleep) before releasing it -- releasing early would
        # let the blocked claim succeed once unblocked instead of timing out,
        # never exercising the recovery path this test is for. The bound is
        # generous because the claim itself runs on the default thread-pool
        # executor, whose scheduling delay grows under a heavily loaded test
        # run rather than staying tied to busy_timeout.
        for _ in range(2000):
            if "component=task-command-dispatcher" in caplog.text:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("worker never logged a lock-timeout failure")
        assert not applied.is_set(), "must not have claimed while the lock was held"
        owner_db.commit()
        await asyncio.wait_for(applied.wait(), timeout=20)
    finally:
        worker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker
        owner_db.close()

    assert "component=task-command-dispatcher" in caplog.text


def test_enqueue_notifies_only_after_commit(db_session, monkeypatch) -> None:
    """The dispatcher wakeup must fire only once the command is durable --
    notifying before commit could wake a dispatcher onto a row it cannot yet
    see, wasting the cycle instead of claiming it."""

    user, task = _create_running_task(db_session)
    order: list[str] = []
    real_commit = db_session.commit

    def recording_commit():
        order.append("commit")
        real_commit()

    def recording_notify():
        order.append("notify")

    monkeypatch.setattr(db_session, "commit", recording_commit)
    monkeypatch.setattr(
        task_command_transport_module,
        "notify_task_command_dispatcher",
        recording_notify,
    )

    enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="notify-order",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
    )

    assert order == ["commit", "notify"]


def test_defer_budget_is_coupled_to_the_lease_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The defer budget must outlast the configured lease TTL with margin.

    A deferral's canonical wait is an unreleased task lease, which clears
    within one TTL; a fixed budget silently smaller than a raised
    ``XAGENT_TASK_LEASE_TTL_SECONDS`` turned every long park into a terminal
    failure for an already-accepted command (xorbitsai/xagent-saas#952 B3).
    """

    monkeypatch.setenv("XAGENT_TASK_LEASE_TTL_SECONDS", "300")
    assert max_command_defers() == 600

    # The historical constant stays as the floor for short TTLs.
    monkeypatch.setenv("XAGENT_TASK_LEASE_TTL_SECONDS", "10")
    assert max_command_defers() == MAX_COMMAND_DEFERS

    # The default (no env var) doubles the historical constant.
    monkeypatch.delenv("XAGENT_TASK_LEASE_TTL_SECONDS", raising=False)
    assert max_command_defers() == 120

    # An invalid value falls back to the default TTL, not the floor.
    monkeypatch.setenv("XAGENT_TASK_LEASE_TTL_SECONDS", "not-a-number")
    assert max_command_defers() == 120
