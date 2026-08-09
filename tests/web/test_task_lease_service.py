"""Tests for task execution leases."""

import asyncio
import logging
import re
import threading
import uuid
from datetime import timedelta

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from tests.shared.db_teardown import drop_all_tables
from xagent.core.agent.checkpoint import CHECKPOINT_TYPE, LEGACY_CHECKPOINT_TYPES
from xagent.web.models import database as database_module
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import ExecutionMode, Task, TaskStatus, TraceEvent
from xagent.web.models.user import User
from xagent.web.services import task_lease_service
from xagent.web.services.db_runtime import drain_async_task_cancellation_safe
from xagent.web.services.task_lease_recovery import (
    recover_task_lease_candidate_isolated,
)
from xagent.web.services.task_lease_service import (
    TASK_RUN_ID_TRACE_FIELD,
    TaskLease,
    TaskLeaseHeartbeatOutcome,
    TaskLeaseLostError,
    TaskLeaseRefreshState,
    acquire_task_lease,
    acquire_task_lease_no_commit,
    get_expired_task_lease_candidates,
    refresh_task_lease,
    release_task_lease,
    run_task_lease_heartbeat,
    run_while_task_lease_owned,
    stop_task_lease_heartbeat,
    utc_now,
)


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'lease.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        drop_all_tables(get_engine())


@pytest.fixture()
def queue_pool_runtime_db(tmp_path):
    """A real one-slot QueuePool used to exercise checkout contention."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'lease-queue-pool.db'}",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.4,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    try:
        yield engine, SessionLocal
    finally:
        engine.dispose()


def _create_task(db, *, status=TaskStatus.PENDING) -> Task:
    user = User(username="lease-user", password_hash="hash", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    task = Task(
        user_id=user.id,
        title="Lease test",
        description="Lease test",
        status=status,
        execution_mode="auto",
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def _recover_expired_task(db, task: Task) -> TaskStatus | None:
    candidate = get_expired_task_lease_candidates(
        db,
        cutoff=utc_now(),
        limit=1,
    )[0]
    db.rollback()
    return recover_task_lease_candidate_isolated(
        candidate,
        recovered_at=utc_now(),
    )


@pytest.mark.asyncio
async def test_guard_cancels_and_drains_execution_on_lease_loss() -> None:
    operation_started = asyncio.Event()
    operation_cancelled = asyncio.Event()

    async def operation() -> None:
        operation_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            operation_cancelled.set()

    async def heartbeat() -> TaskLeaseHeartbeatOutcome:
        await operation_started.wait()
        return TaskLeaseHeartbeatOutcome(lease_lost=True)

    heartbeat_task = asyncio.create_task(heartbeat())
    with pytest.raises(TaskLeaseLostError):
        await run_while_task_lease_owned(operation(), heartbeat_task)

    assert operation_cancelled.is_set()
    assert heartbeat_task.done()


@pytest.mark.asyncio
async def test_guard_keeps_execution_running_for_settlement_ready() -> None:
    operation_started = asyncio.Event()
    allow_operation_to_finish = asyncio.Event()

    async def operation() -> str:
        operation_started.set()
        await allow_operation_to_finish.wait()
        return "completed"

    async def heartbeat() -> TaskLeaseHeartbeatOutcome:
        await operation_started.wait()
        allow_operation_to_finish.set()
        return TaskLeaseHeartbeatOutcome()

    heartbeat_task = asyncio.create_task(heartbeat())

    assert await run_while_task_lease_owned(operation(), heartbeat_task) == "completed"


@pytest.mark.asyncio
async def test_guard_does_not_cancel_for_transient_heartbeat_pool_timeout() -> None:
    timeout_observed = asyncio.Event()
    stop_heartbeat = asyncio.Event()

    async def heartbeat() -> TaskLeaseHeartbeatOutcome:
        timeout_observed.set()
        await stop_heartbeat.wait()
        return TaskLeaseHeartbeatOutcome(
            pool_timeout=SQLAlchemyTimeoutError("transient pool timeout")
        )

    async def operation() -> str:
        await timeout_observed.wait()
        return "kept-running"

    heartbeat_task = asyncio.create_task(heartbeat())
    assert (
        await run_while_task_lease_owned(operation(), heartbeat_task) == "kept-running"
    )
    assert not heartbeat_task.done()
    stop_heartbeat.set()
    outcome = await heartbeat_task
    assert outcome.pool_timeout is not None


@pytest.mark.asyncio
async def test_guard_cancels_and_drains_execution_when_heartbeat_crashes() -> None:
    operation_started = asyncio.Event()
    operation_cancelled = asyncio.Event()

    async def operation() -> None:
        operation_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            operation_cancelled.set()

    async def heartbeat() -> TaskLeaseHeartbeatOutcome:
        await operation_started.wait()
        raise RuntimeError("heartbeat crashed")

    heartbeat_task = asyncio.create_task(heartbeat())
    with pytest.raises(RuntimeError, match="heartbeat crashed"):
        await run_while_task_lease_owned(operation(), heartbeat_task)

    assert operation_cancelled.is_set()
    assert heartbeat_task.done()


def test_task_model_default_execution_mode_is_auto(db_session) -> None:
    user = User(username="default-mode-user", password_hash="hash", is_admin=False)
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    task = Task(
        user_id=user.id,
        title="Default mode",
        description="Default mode",
        status=TaskStatus.PENDING,
    )
    db_session.add(task)
    db_session.commit()
    db_session.refresh(task)

    assert task.execution_mode == "auto"
    assert task.execution_mode_enum == ExecutionMode.AUTO


def test_task_lease_acquire_refresh_and_release(db_session) -> None:
    task = _create_task(db_session)

    lease = acquire_task_lease(db_session, int(task.id), runner_id="runner-a")

    assert lease is not None
    db_session.refresh(task)
    assert task.status == TaskStatus.RUNNING
    assert task.runner_id == "runner-a"
    assert task.lease_expires_at is not None
    assert task.run_id == lease.run_id
    assert task.state_version == 1
    assert task.control_state == "running"

    assert acquire_task_lease(db_session, int(task.id), runner_id="runner-b") is None
    assert refresh_task_lease(db_session, lease) == TaskLeaseRefreshState.REFRESHED
    assert release_task_lease(db_session, lease, status=TaskStatus.COMPLETED) is True
    db_session.refresh(task)
    assert task.status == TaskStatus.COMPLETED
    assert task.state_version == 2
    assert task.control_state == "completed"
    assert task.runner_id is None
    assert task.lease_expires_at is None


def test_acquire_returns_run_id_from_update_without_followup_select(
    queue_pool_runtime_db,
) -> None:
    engine, SessionLocal = queue_pool_runtime_db
    with SessionLocal() as seed_db:
        task = _create_task(seed_db)
        task_id = int(task.id)

    statements: list[str] = []

    def record_statement(
        _conn,
        _cursor,
        statement: str,
        _parameters,
        _context,
        _executemany,
    ) -> None:
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", record_statement)
    try:
        with SessionLocal() as db:
            lease = acquire_task_lease(db, task_id, runner_id="runner-returning")
    finally:
        event.remove(engine, "before_cursor_execute", record_statement)

    assert lease is not None
    assert lease.run_id
    assert len(statements) == 1
    assert statements[0].lstrip().upper().startswith("UPDATE")
    assert "RETURNING" in statements[0].upper()


def test_refresh_batch_uses_one_pool_checkout(
    queue_pool_runtime_db,
    monkeypatch,
) -> None:
    engine, SessionLocal = queue_pool_runtime_db
    leases: list[TaskLease] = []
    with SessionLocal() as seed_db:
        user = User(
            username="batch-heartbeat-owner",
            password_hash="hash",
            is_admin=False,
        )
        seed_db.add(user)
        seed_db.flush()
        for index in range(100):
            run_id = f"run-{index}"
            task = Task(
                user_id=user.id,
                title=f"Batch lease {index}",
                description="Batch lease",
                status=TaskStatus.RUNNING,
                execution_mode="auto",
                runner_id="runner-a",
                run_id=run_id,
            )
            seed_db.add(task)
            seed_db.flush()
            leases.append(
                TaskLease(
                    task_id=int(task.id),
                    runner_id="runner-a",
                    run_id=run_id,
                )
            )
        seed_db.commit()

    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)
    checkouts = 0

    def record_checkout(*_args) -> None:
        nonlocal checkouts
        checkouts += 1

    event.listen(engine, "checkout", record_checkout)
    try:
        states = task_lease_service.refresh_task_leases_isolated(tuple(leases))
    finally:
        event.remove(engine, "checkout", record_checkout)

    assert checkouts == 1
    assert len(states) == 100
    assert set(states.values()) == {TaskLeaseRefreshState.REFRESHED}


def test_acquire_no_commit_leaves_transaction_owned_by_caller(db_session) -> None:
    task = _create_task(db_session)

    lease = acquire_task_lease_no_commit(
        db_session,
        int(task.id),
        runner_id="transaction-owner",
    )

    assert lease is not None
    db_session.rollback()
    db_session.refresh(task)
    assert task.status == TaskStatus.PENDING
    assert task.runner_id is None
    assert task.run_id is None


def test_fail_and_release_task_lease_rejects_superseded_owner(db_session) -> None:
    task = _create_task(db_session)
    stale_lease = acquire_task_lease(
        db_session,
        int(task.id),
        runner_id="old-runner",
    )
    assert stale_lease is not None

    task.runner_id = "new-runner"
    task.run_id = "new-run"
    task.error_message = None
    task.output = "new owner output"
    db_session.commit()

    changed = task_lease_service.fail_and_release_task_lease_no_commit(
        db_session,
        stale_lease,
        error_message="stale runner failed",
    )
    db_session.commit()

    assert changed is False
    db_session.refresh(task)
    assert task.status == TaskStatus.RUNNING
    assert task.runner_id == "new-runner"
    assert task.run_id == "new-run"
    assert task.error_message is None
    assert task.output == "new owner output"


def test_fail_and_release_task_lease_atomically_fails_current_owner(db_session) -> None:
    task = _create_task(db_session)
    lease = acquire_task_lease(db_session, int(task.id), runner_id="runner-a")
    assert lease is not None
    task.output = "stale output"
    db_session.commit()
    state_version = int(task.state_version)

    changed = task_lease_service.fail_and_release_task_lease_no_commit(
        db_session,
        lease,
        error_message="setup failed",
    )
    db_session.commit()

    assert changed is True
    db_session.refresh(task)
    assert task.status == TaskStatus.FAILED
    assert task.control_state == "failed"
    assert task.state_version == state_version + 1
    assert task.error_message == "setup failed"
    assert task.output is None
    assert task.runner_id is None
    assert task.lease_expires_at is None


def test_release_task_lease_refuses_ownerless_running_state(db_session) -> None:
    task = _create_task(db_session)
    lease = acquire_task_lease(db_session, int(task.id), runner_id="runner-a")
    assert lease is not None

    with pytest.raises(ValueError, match="RUNNING"):
        release_task_lease(db_session, lease, status=TaskStatus.RUNNING)

    db_session.refresh(task)
    assert task.status == TaskStatus.RUNNING
    assert task.runner_id == "runner-a"
    assert task.lease_expires_at is not None


@pytest.mark.parametrize(
    "status",
    [TaskStatus.PAUSED, TaskStatus.WAITING_FOR_USER],
)
def test_release_to_non_terminal_status_clears_stale_error_message(
    db_session, status
) -> None:
    task = _create_task(db_session)
    lease = acquire_task_lease(db_session, int(task.id), runner_id="runner-a")
    assert lease is not None
    task.error_message = "earlier failure"
    task.output = "prior answer"
    db_session.commit()

    changed = task_lease_service.release_task_lease_no_commit(
        db_session,
        lease,
        status=status,
    )
    db_session.commit()

    assert changed is True
    db_session.refresh(task)
    assert task.status == status
    assert task.error_message is None
    assert task.output == "prior answer"


@pytest.mark.parametrize(
    "status",
    [TaskStatus.COMPLETED, TaskStatus.FAILED],
)
def test_release_to_terminal_status_leaves_content_fields_alone(
    db_session, status
) -> None:
    task = _create_task(db_session)
    lease = acquire_task_lease(db_session, int(task.id), runner_id="runner-a")
    assert lease is not None
    task.error_message = "earlier failure"
    task.output = "prior answer"
    db_session.commit()

    changed = task_lease_service.release_task_lease_no_commit(
        db_session,
        lease,
        status=status,
    )
    db_session.commit()

    assert changed is True
    db_session.refresh(task)
    assert task.status == status
    assert task.error_message == "earlier failure"
    assert task.output == "prior answer"


@pytest.mark.asyncio
async def test_lease_heartbeat_keeps_loop_responsive_during_pool_checkout(
    queue_pool_runtime_db,
    monkeypatch,
) -> None:
    engine, SessionLocal = queue_pool_runtime_db
    with SessionLocal() as seed_db:
        task = _create_task(seed_db, status=TaskStatus.RUNNING)
        task_id = int(task.id)
        task.runner_id = "runner-a"
        task.run_id = "run-a"
        seed_db.commit()

    def constrained_get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    monkeypatch.setattr(
        task_lease_service,
        "get_db",
        constrained_get_db,
        raising=False,
    )
    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)
    monkeypatch.setattr(
        task_lease_service,
        "get_task_lease_heartbeat_seconds",
        lambda: 0.001,
    )

    held_connection = engine.connect()
    stop_event = asyncio.Event()
    ticker_stop = asyncio.Event()
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while not ticker_stop.is_set():
            await asyncio.sleep(0.01)
            ticks += 1

    heartbeat_task = asyncio.create_task(
        run_task_lease_heartbeat(
            TaskLease(task_id=task_id, runner_id="runner-a", run_id="run-a"),
            stop_event,
        )
    )
    ticker_task = asyncio.create_task(ticker())
    try:
        await asyncio.sleep(0.12)
        assert ticks >= 3, "QueuePool checkout blocked the asyncio event loop"
    finally:
        held_connection.close()
        stop_event.set()
        await asyncio.wait_for(heartbeat_task, timeout=1)
        ticker_stop.set()
        await ticker_task

    with SessionLocal() as verify_db:
        refreshed = verify_db.query(Task).filter(Task.id == task_id).one()
        assert refreshed.last_heartbeat_at is not None


@pytest.mark.asyncio
async def test_stop_heartbeat_waits_for_shared_batch_result(monkeypatch) -> None:
    refresh_started = threading.Event()
    allow_refresh_to_finish = threading.Event()

    def blocking_refresh(
        leases: tuple[TaskLease, ...],
    ) -> dict[tuple[int, str, str | None], TaskLeaseRefreshState]:
        refresh_started.set()
        assert allow_refresh_to_finish.wait(timeout=2)
        return {
            (lease.task_id, lease.runner_id, lease.run_id): (
                TaskLeaseRefreshState.REFRESHED
            )
            for lease in leases
        }

    monkeypatch.setattr(
        task_lease_service,
        "refresh_task_leases_isolated",
        blocking_refresh,
    )
    monkeypatch.setattr(
        task_lease_service,
        "get_task_lease_heartbeat_seconds",
        lambda: 0.001,
    )

    stop_event = asyncio.Event()
    heartbeat_task = asyncio.create_task(
        run_task_lease_heartbeat(
            TaskLease(task_id=1, runner_id="runner-a", run_id="run-a"),
            stop_event,
        )
    )
    await asyncio.wait_for(asyncio.to_thread(refresh_started.wait, 1), timeout=1)

    stopping = asyncio.create_task(
        stop_task_lease_heartbeat(heartbeat_task, stop_event)
    )
    await asyncio.sleep(0.02)
    assert not stopping.done()

    allow_refresh_to_finish.set()
    outcome = await asyncio.wait_for(stopping, timeout=1)
    await task_lease_service.wait_for_heartbeat_manager_idle()

    assert outcome.requires_ttl_recovery is False


@pytest.mark.asyncio
async def test_cancelled_heartbeat_manager_settles_active_registration(
    monkeypatch,
) -> None:
    refresh_started = threading.Event()
    allow_refresh_to_finish = threading.Event()

    def blocking_refresh(
        leases: tuple[TaskLease, ...],
    ) -> dict[tuple[int, str, str | None], TaskLeaseRefreshState]:
        refresh_started.set()
        assert allow_refresh_to_finish.wait(timeout=2)
        return {
            (lease.task_id, lease.runner_id, lease.run_id): (
                TaskLeaseRefreshState.REFRESHED
            )
            for lease in leases
        }

    monkeypatch.setattr(
        task_lease_service,
        "refresh_task_leases_isolated",
        blocking_refresh,
    )
    monkeypatch.setattr(
        task_lease_service,
        "get_task_lease_heartbeat_seconds",
        lambda: 0.001,
    )

    manager = task_lease_service._TaskLeaseHeartbeatManager(asyncio.get_running_loop())
    registration = manager.register(
        TaskLease(task_id=1, runner_id="runner-a", run_id="run-a")
    )
    await asyncio.wait_for(asyncio.to_thread(refresh_started.wait, 1), timeout=1)

    runner = manager._runner
    assert runner is not None
    runner.cancel()
    allow_refresh_to_finish.set()
    with pytest.raises(asyncio.CancelledError):
        await runner

    outcome = await asyncio.wait_for(registration.close(), timeout=1)

    assert outcome.requires_ttl_recovery is False
    assert registration._entry.refresh_waiter is None


@pytest.mark.asyncio
async def test_wait_for_heartbeat_manager_idle_drains_before_caller_cancellation(
    monkeypatch,
) -> None:
    manager = task_lease_service._TaskLeaseHeartbeatManager(asyncio.get_running_loop())
    runner_started = asyncio.Event()
    allow_runner_to_finish = asyncio.Event()

    async def runner() -> None:
        runner_started.set()
        await allow_runner_to_finish.wait()

    manager._runner = asyncio.create_task(runner())
    monkeypatch.setattr(
        task_lease_service,
        "_task_lease_heartbeat_manager",
        manager,
    )
    await runner_started.wait()

    waiter = asyncio.create_task(task_lease_service.wait_for_heartbeat_manager_idle())
    await asyncio.sleep(0)
    assert not waiter.done()
    waiter.cancel()
    await asyncio.sleep(0)

    assert not waiter.done()
    assert manager._runner is not None
    assert not manager._runner.done()

    allow_runner_to_finish.set()
    with pytest.raises(asyncio.CancelledError):
        await waiter


@pytest.mark.asyncio
async def test_repeated_cancellation_drains_heartbeat_close_and_waiters(
    monkeypatch,
) -> None:
    refresh_started = threading.Event()
    allow_refresh_to_finish = threading.Event()
    gather_started = asyncio.Event()
    allow_gather_to_finish = asyncio.Event()
    original_gather = asyncio.gather

    def blocking_refresh(
        leases: tuple[TaskLease, ...],
    ) -> dict[tuple[int, str, str | None], TaskLeaseRefreshState]:
        refresh_started.set()
        assert allow_refresh_to_finish.wait(timeout=2)
        return {
            (lease.task_id, lease.runner_id, lease.run_id): (
                TaskLeaseRefreshState.REFRESHED
            )
            for lease in leases
        }

    async def blocking_gather(*args, **kwargs):
        gather_started.set()
        await allow_gather_to_finish.wait()
        return await original_gather(*args, **kwargs)

    monkeypatch.setattr(
        task_lease_service,
        "refresh_task_leases_isolated",
        blocking_refresh,
    )
    monkeypatch.setattr(
        task_lease_service,
        "get_task_lease_heartbeat_seconds",
        lambda: 0.001,
    )
    monkeypatch.setattr(task_lease_service.asyncio, "gather", blocking_gather)

    heartbeat_task = asyncio.get_running_loop().create_task(
        run_task_lease_heartbeat(
            TaskLease(task_id=1, runner_id="runner-a", run_id="run-a"),
            asyncio.Event(),
        )
    )
    await asyncio.wait_for(asyncio.to_thread(refresh_started.wait, 1), timeout=1)

    manager = task_lease_service._get_task_lease_heartbeat_manager()
    entry = next(iter(manager._entries.values()))
    refresh_waiter = entry.refresh_waiter
    assert refresh_waiter is not None

    heartbeat_task.cancel()
    await asyncio.sleep(0.02)
    assert not heartbeat_task.done()

    allow_refresh_to_finish.set()
    await asyncio.wait_for(gather_started.wait(), timeout=1)

    heartbeat_task.cancel()
    await asyncio.sleep(0.02)
    try:
        assert not heartbeat_task.done()
    finally:
        allow_gather_to_finish.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(heartbeat_task, timeout=1)
    await task_lease_service.wait_for_heartbeat_manager_idle()

    assert refresh_waiter.done()
    assert entry.refresh_waiter is None
    assert manager._entries == {}


@pytest.mark.asyncio
async def test_heartbeat_requires_exact_run_id() -> None:
    lease = TaskLease(task_id=1, runner_id="runner-a")

    with pytest.raises(ValueError, match="exact run_id fence"):
        await run_task_lease_heartbeat(lease, asyncio.Event())


@pytest.mark.asyncio
async def test_heartbeat_batches_registered_leases(monkeypatch) -> None:
    refreshed = threading.Event()
    batches: list[tuple[TaskLease, ...]] = []

    def refresh_batch(
        leases: tuple[TaskLease, ...],
    ) -> dict[tuple[int, str, str | None], TaskLeaseRefreshState]:
        batches.append(leases)
        refreshed.set()
        return {
            (lease.task_id, lease.runner_id, lease.run_id): (
                TaskLeaseRefreshState.REFRESHED
            )
            for lease in leases
        }

    monkeypatch.setattr(
        task_lease_service,
        "refresh_task_leases_isolated",
        refresh_batch,
    )
    monkeypatch.setattr(
        task_lease_service,
        "get_task_lease_heartbeat_seconds",
        lambda: 0.001,
    )

    first_stop = asyncio.Event()
    second_stop = asyncio.Event()
    first_task = asyncio.create_task(
        run_task_lease_heartbeat(
            TaskLease(task_id=1, runner_id="runner-a", run_id="run-a"),
            first_stop,
        )
    )
    second_task = asyncio.create_task(
        run_task_lease_heartbeat(
            TaskLease(task_id=2, runner_id="runner-a", run_id="run-b"),
            second_stop,
        )
    )
    assert await asyncio.to_thread(refreshed.wait, 1)

    await stop_task_lease_heartbeat(first_task, first_stop)
    await stop_task_lease_heartbeat(second_task, second_stop)
    await task_lease_service.wait_for_heartbeat_manager_idle()

    # The 1 ms test interval may legitimately permit another refresh for the
    # second lease between the two sequential stop calls.  The batching
    # invariant is that the first interval refreshes both active leases with
    # one worker checkout, not that no later interval can run.
    assert batches
    assert {(lease.task_id, lease.run_id) for lease in batches[0]} == {
        (1, "run-a"),
        (2, "run-b"),
    }


@pytest.mark.asyncio
async def test_old_batch_result_does_not_contaminate_replacement_registration(
    monkeypatch,
) -> None:
    first_refresh_started = threading.Event()
    allow_first_refresh = threading.Event()
    second_refresh_started = threading.Event()
    allow_second_refresh = threading.Event()
    attempts = 0
    lease = TaskLease(task_id=1, runner_id="runner-a", run_id="run-a")
    key = (lease.task_id, lease.runner_id, lease.run_id)

    def refresh_batch(
        _leases: tuple[TaskLease, ...],
    ) -> dict[tuple[int, str, str | None], TaskLeaseRefreshState]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            first_refresh_started.set()
            assert allow_first_refresh.wait(timeout=2)
            return {key: TaskLeaseRefreshState.LOST}
        second_refresh_started.set()
        assert allow_second_refresh.wait(timeout=2)
        return {key: TaskLeaseRefreshState.REFRESHED}

    monkeypatch.setattr(
        task_lease_service,
        "refresh_task_leases_isolated",
        refresh_batch,
    )
    monkeypatch.setattr(
        task_lease_service,
        "get_task_lease_heartbeat_seconds",
        lambda: 0.001,
    )

    first_stop = asyncio.Event()
    first_task = asyncio.create_task(run_task_lease_heartbeat(lease, first_stop))
    assert await asyncio.to_thread(first_refresh_started.wait, 1)

    first_stopping = asyncio.create_task(
        stop_task_lease_heartbeat(first_task, first_stop)
    )
    await asyncio.sleep(0.02)
    assert not first_stopping.done()

    replacement_stop = asyncio.Event()
    replacement_task = asyncio.create_task(
        run_task_lease_heartbeat(lease, replacement_stop)
    )
    allow_first_refresh.set()

    first_outcome = await asyncio.wait_for(first_stopping, timeout=1)
    assert first_outcome.lease_lost is True
    assert not replacement_task.done()
    assert await asyncio.to_thread(second_refresh_started.wait, 1)

    replacement_stopping = asyncio.create_task(
        stop_task_lease_heartbeat(replacement_task, replacement_stop)
    )
    await asyncio.sleep(0.02)
    assert not replacement_stopping.done()

    allow_second_refresh.set()
    replacement_outcome = await asyncio.wait_for(
        replacement_stopping,
        timeout=1,
    )
    await task_lease_service.wait_for_heartbeat_manager_idle()

    assert replacement_outcome.requires_ttl_recovery is False


@pytest.mark.asyncio
async def test_stop_heartbeat_reports_shared_batch_pool_timeout(
    monkeypatch, caplog
) -> None:
    refresh_attempted = threading.Event()
    allow_timeout = threading.Event()
    heartbeat_timeout = SQLAlchemyTimeoutError("pool checkout timed out")

    def timed_out_refresh(
        _leases: tuple[TaskLease, ...],
    ) -> dict[tuple[int, str, str | None], TaskLeaseRefreshState]:
        refresh_attempted.set()
        assert allow_timeout.wait(timeout=2)
        raise heartbeat_timeout

    monkeypatch.setattr(
        task_lease_service,
        "refresh_task_leases_isolated",
        timed_out_refresh,
    )
    monkeypatch.setattr(
        task_lease_service,
        "get_task_lease_heartbeat_seconds",
        lambda: 0.001,
    )

    first_stop_event = asyncio.Event()
    second_stop_event = asyncio.Event()
    first_heartbeat_task = None
    second_heartbeat_task = None
    first_stopping = None
    second_stopping = None

    async def stop_and_drain_heartbeat(
        stopping,
        heartbeat_task,
        stop_event,
    ) -> None:
        stop_event.set()
        if heartbeat_task is None:
            return
        if stopping is None:
            stopping = asyncio.create_task(
                stop_task_lease_heartbeat(heartbeat_task, stop_event)
            )
        try:
            await drain_async_task_cancellation_safe(stopping)
        finally:
            await drain_async_task_cancellation_safe(heartbeat_task)

    try:
        with caplog.at_level(logging.WARNING):
            first_heartbeat_task = asyncio.create_task(
                run_task_lease_heartbeat(
                    TaskLease(task_id=1, runner_id="runner-a", run_id="run-a"),
                    first_stop_event,
                )
            )
            second_heartbeat_task = asyncio.create_task(
                run_task_lease_heartbeat(
                    TaskLease(task_id=2, runner_id="runner-b", run_id="run-b"),
                    second_stop_event,
                )
            )
            await asyncio.wait_for(
                asyncio.to_thread(refresh_attempted.wait, 1), timeout=1
            )

            first_stopping = asyncio.create_task(
                stop_task_lease_heartbeat(first_heartbeat_task, first_stop_event)
            )
            second_stopping = asyncio.create_task(
                stop_task_lease_heartbeat(second_heartbeat_task, second_stop_event)
            )
            await asyncio.sleep(0.02)
            assert not first_stopping.done()
            assert not second_stopping.done()

            allow_timeout.set()
            first_outcome, second_outcome = await asyncio.wait_for(
                asyncio.gather(first_stopping, second_stopping),
                timeout=1,
            )

        heartbeat_warnings = [
            record
            for record in caplog.records
            if record.levelno == logging.WARNING
            and "component=lease-heartbeat" in record.getMessage()
        ]
        assert len(heartbeat_warnings) == 1
        assert heartbeat_warnings[0].getMessage() == (
            "component=lease-heartbeat task_ids=[1,2] active_lease_count=2 "
            "database pool checkout timed out: pool checkout timed out"
        )
        for outcome in (first_outcome, second_outcome):
            assert outcome.pool_timeout is heartbeat_timeout
            assert outcome.lease_lost is False
            assert outcome.requires_ttl_recovery is True
    finally:
        allow_timeout.set()
        try:
            await stop_and_drain_heartbeat(
                first_stopping,
                first_heartbeat_task,
                first_stop_event,
            )
        finally:
            try:
                await stop_and_drain_heartbeat(
                    second_stopping,
                    second_heartbeat_task,
                    second_stop_event,
                )
            finally:
                await task_lease_service.wait_for_heartbeat_manager_idle()


@pytest.mark.asyncio
async def test_stop_heartbeat_reports_lost_ownership(monkeypatch) -> None:
    def lost_refresh(
        leases: tuple[TaskLease, ...],
    ) -> dict[tuple[int, str, str | None], TaskLeaseRefreshState]:
        return {
            (lease.task_id, lease.runner_id, lease.run_id): TaskLeaseRefreshState.LOST
            for lease in leases
        }

    monkeypatch.setattr(
        task_lease_service,
        "refresh_task_leases_isolated",
        lost_refresh,
    )
    monkeypatch.setattr(
        task_lease_service,
        "get_task_lease_heartbeat_seconds",
        lambda: 0.001,
    )

    stop_event = asyncio.Event()
    heartbeat_task = asyncio.create_task(
        run_task_lease_heartbeat(
            TaskLease(task_id=1, runner_id="runner-a", run_id="run-a"),
            stop_event,
        )
    )
    await asyncio.wait_for(heartbeat_task, timeout=1)

    outcome = await stop_task_lease_heartbeat(heartbeat_task, stop_event)
    await task_lease_service.wait_for_heartbeat_manager_idle()

    assert outcome.lease_lost is True
    assert outcome.pool_timeout is None


@pytest.mark.asyncio
async def test_heartbeat_does_not_report_owned_terminal_task_as_lease_lost(
    db_session,
    monkeypatch,
) -> None:
    task = _create_task(db_session)
    lease = acquire_task_lease(db_session, int(task.id), runner_id="runner-a")
    assert lease is not None

    task.status = TaskStatus.COMPLETED
    task.control_state = "completed"
    db_session.commit()

    monkeypatch.setattr(
        task_lease_service,
        "get_task_lease_heartbeat_seconds",
        lambda: 0.001,
    )

    outcome = await asyncio.wait_for(
        run_task_lease_heartbeat(lease, asyncio.Event()),
        timeout=1,
    )
    await task_lease_service.wait_for_heartbeat_manager_idle()

    assert outcome.lease_lost is False
    assert outcome.requires_ttl_recovery is False


@pytest.mark.asyncio
async def test_batch_heartbeat_recovers_after_transient_pool_timeout(
    monkeypatch,
) -> None:
    refresh_recovered = threading.Event()
    attempts = 0

    def recovering_refresh(
        leases: tuple[TaskLease, ...],
    ) -> dict[tuple[int, str, str | None], TaskLeaseRefreshState]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise SQLAlchemyTimeoutError("transient pool checkout timeout")
        refresh_recovered.set()
        return {
            (lease.task_id, lease.runner_id, lease.run_id): (
                TaskLeaseRefreshState.REFRESHED
            )
            for lease in leases
        }

    monkeypatch.setattr(
        task_lease_service,
        "refresh_task_leases_isolated",
        recovering_refresh,
    )
    monkeypatch.setattr(
        task_lease_service,
        "get_task_lease_heartbeat_seconds",
        lambda: 0.001,
    )

    stop_event = asyncio.Event()
    heartbeat_task = asyncio.create_task(
        run_task_lease_heartbeat(
            TaskLease(task_id=1, runner_id="runner-a", run_id="run-a"),
            stop_event,
        )
    )
    assert await asyncio.to_thread(refresh_recovered.wait, 1)

    outcome = await stop_task_lease_heartbeat(heartbeat_task, stop_event)
    await task_lease_service.wait_for_heartbeat_manager_idle()

    assert outcome.requires_ttl_recovery is False


@pytest.mark.asyncio
async def test_cancellation_safe_acquire_drains_and_cleans_returned_lease() -> None:
    acquire_started = threading.Event()
    allow_acquire_to_finish = threading.Event()
    cleanup_started = threading.Event()
    allow_cleanup_to_finish = threading.Event()
    expected_lease = TaskLease(task_id=9, runner_id="runner-a", run_id="run-a")
    cleaned_leases: list[TaskLease] = []

    def acquire() -> TaskLease:
        acquire_started.set()
        assert allow_acquire_to_finish.wait(timeout=2)
        return expected_lease

    def cleanup(lease: TaskLease) -> None:
        cleanup_started.set()
        assert allow_cleanup_to_finish.wait(timeout=2)
        cleaned_leases.append(lease)

    operation = asyncio.create_task(
        task_lease_service.acquire_task_lease_cancellation_safe(acquire, cleanup)
    )
    await asyncio.wait_for(asyncio.to_thread(acquire_started.wait, 1), timeout=1)
    operation.cancel()
    await asyncio.sleep(0.02)
    assert not operation.done()

    allow_acquire_to_finish.set()
    await asyncio.wait_for(asyncio.to_thread(cleanup_started.wait, 1), timeout=1)
    assert not operation.done()

    allow_cleanup_to_finish.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(operation, timeout=1)
    assert cleaned_leases == [expected_lease]


_CASE_WHEN_PATTERN = re.compile(r"WHEN (.+?) THEN")


def _case_when_clause(case_expr) -> str:
    """The compiled WHEN condition of a two-branch case(), with literal
    binds inlined so two structurally-identical predicates compile to the
    same string regardless of bind-parameter naming.
    """
    compiled = str(case_expr.compile(compile_kwargs={"literal_binds": True}))
    match = _CASE_WHEN_PATTERN.search(compiled)
    assert match is not None, compiled
    return match.group(1)


def test_lease_checkpoint_pointer_case_predicates_match() -> None:
    """lease_checkpoint_event_id_case and lease_checkpoint_trace_event_id_case
    must clear their column under exactly the same condition.

    Both builders share _checkpoint_pointer_clearing_predicate(), so this
    passes by construction today; the assertion is what actually enforces
    it -- a future edit that inlines a diverging predicate into one builder
    would otherwise pass every other test here (each builder's own column
    still clears/retains correctly in isolation) while silently breaking
    the recovery CAS fence's two-column conjunction, which depends on both
    columns clearing together (see recover_expired_task_lease_no_commit).
    """
    legacy_when = _case_when_clause(task_lease_service.lease_checkpoint_event_id_case())
    exact_when = _case_when_clause(
        task_lease_service.lease_checkpoint_trace_event_id_case()
    )
    assert legacy_when == exact_when


def test_new_run_lease_claim_rejects_a_second_claim(db_session) -> None:
    task = _create_task(db_session)
    task.last_checkpoint_event_id = "previous-run-checkpoint"
    previous_checkpoint = TraceEvent(
        task_id=task.id,
        event_id="previous-run-checkpoint",
        event_type="system_update_general",
        timestamp=utc_now(),
        data={"checkpoint_type": CHECKPOINT_TYPE, "snapshot": {"type": "checkpoint"}},
    )
    db_session.add(previous_checkpoint)
    db_session.flush()
    task.last_checkpoint_trace_event_id = previous_checkpoint.id
    db_session.commit()

    first = acquire_task_lease(
        db_session,
        int(task.id),
        runner_id="runner-a",
        new_run=True,
    )
    second = acquire_task_lease(
        db_session,
        int(task.id),
        runner_id="runner-a",
        new_run=True,
    )

    assert first is not None
    assert first.run_id
    assert second is None
    db_session.refresh(task)
    assert task.run_id == first.run_id
    assert task.last_checkpoint_event_id is None
    # A new run clears both pointer columns together -- see
    # lease_checkpoint_trace_event_id_case's docstring for why a single
    # column left behind would desync the recovery CAS fence.
    assert task.last_checkpoint_trace_event_id is None


def test_same_run_resume_lease_preserves_checkpoint_pointer(db_session) -> None:
    task = _create_task(db_session, status=TaskStatus.WAITING_FOR_USER)
    task.run_id = "resumable-run"
    task.last_checkpoint_event_id = "current-run-checkpoint"
    current_checkpoint = TraceEvent(
        task_id=task.id,
        event_id="current-run-checkpoint",
        event_type="system_update_general",
        timestamp=utc_now(),
        data={"checkpoint_type": CHECKPOINT_TYPE, "snapshot": {"type": "checkpoint"}},
    )
    db_session.add(current_checkpoint)
    db_session.flush()
    task.last_checkpoint_trace_event_id = current_checkpoint.id
    db_session.commit()

    lease = acquire_task_lease(
        db_session,
        int(task.id),
        runner_id="resume-runner",
        expected_run_id="resumable-run",
    )
    assert lease is not None

    assert lease == TaskLease(
        task_id=int(task.id),
        runner_id="resume-runner",
        run_id="resumable-run",
        # Not part of what this test pins: a fresh attempt id per claim is
        # acquire's contract, exercised by the stamping tests below.
        attempt_id=lease.attempt_id,
    )
    db_session.refresh(task)
    assert task.last_checkpoint_event_id == "current-run-checkpoint"
    # A same-run resume retains both pointer columns together.
    assert task.last_checkpoint_trace_event_id == current_checkpoint.id


def test_running_task_reacquire_retains_checkpoint_pointer_via_case(
    db_session,
) -> None:
    """acquire_task_lease_no_commit's case()-retain branch (no new_run, no
    expected_run_id) only fires when the row is already a live RUNNING row
    with a run id -- exercised here by a heartbeat-style reacquire on a
    task that is already RUNNING with the same runner, unlike every other
    acquire_task_lease call in this file, which starts from a non-RUNNING
    task and only ever hits the case()'s clear side.
    """
    task = _create_task(db_session, status=TaskStatus.RUNNING)
    task.runner_id = "runner-a"
    task.run_id = "existing-run"
    task.last_checkpoint_event_id = "existing-checkpoint"
    checkpoint = TraceEvent(
        task_id=task.id,
        event_id="existing-checkpoint",
        event_type="system_update_general",
        timestamp=utc_now(),
        data={"checkpoint_type": CHECKPOINT_TYPE, "snapshot": {"type": "checkpoint"}},
    )
    db_session.add(checkpoint)
    db_session.flush()
    task.last_checkpoint_trace_event_id = checkpoint.id
    db_session.commit()

    lease = acquire_task_lease(db_session, int(task.id), runner_id="runner-a")

    assert lease is not None
    assert lease.run_id == "existing-run"
    db_session.refresh(task)
    assert task.last_checkpoint_event_id == "existing-checkpoint"
    # Both pointer columns retained together via the paired case() builders.
    assert task.last_checkpoint_trace_event_id == checkpoint.id


def test_lease_acquire_rejects_a_superseded_run(db_session) -> None:
    task = _create_task(db_session, status=TaskStatus.RUNNING)
    task.run_id = "current-run"
    task.control_state = "running"
    task.state_version = 3
    db_session.commit()

    assert (
        acquire_task_lease(
            db_session,
            int(task.id),
            runner_id="runner-a",
            expected_run_id="old-run",
        )
        is None
    )
    db_session.refresh(task)
    assert task.run_id == "current-run"
    assert task.state_version == 3


def test_old_lease_cannot_refresh_or_release_a_new_run(db_session) -> None:
    task = _create_task(db_session)
    old_lease = acquire_task_lease(db_session, int(task.id), runner_id="runner-a")
    assert old_lease is not None

    task.run_id = "new-run"
    task.status = TaskStatus.RUNNING
    task.control_state = "running"
    task.runner_id = "runner-a"
    db_session.commit()

    assert refresh_task_lease(db_session, old_lease) == TaskLeaseRefreshState.LOST
    assert release_task_lease(db_session, old_lease, status=TaskStatus.FAILED) is False
    db_session.refresh(task)
    assert task.run_id == "new-run"
    assert task.status == TaskStatus.RUNNING


def test_stale_running_task_with_checkpoint_becomes_paused(db_session) -> None:
    task = _create_task(db_session, status=TaskStatus.RUNNING)
    task.runner_id = "dead-runner"
    task.run_id = "checkpoint-run"
    task.lease_attempt_id = "stale-attempt"
    task.lease_expires_at = utc_now() - timedelta(seconds=1)
    task.last_checkpoint_event_id = "checkpoint-1"
    checkpoint_event = TraceEvent(
        task_id=task.id,
        event_id="checkpoint-1",
        event_type="system_update_general",
        timestamp=utc_now(),
        step_id=None,
        parent_event_id=None,
        data={
            "checkpoint_type": CHECKPOINT_TYPE,
            "snapshot": {"type": "checkpoint"},
            TASK_RUN_ID_TRACE_FIELD: "checkpoint-run",
        },
    )
    db_session.add(checkpoint_event)
    db_session.flush()
    # Exercises the PK-anchored resolution path (not just the legacy
    # string fallback): recovery must reach the same PAUSED verdict.
    task.last_checkpoint_trace_event_id = checkpoint_event.id
    db_session.commit()

    assert _recover_expired_task(db_session, task) == TaskStatus.PAUSED
    db_session.refresh(task)
    assert task.status == TaskStatus.PAUSED
    assert task.runner_id is None
    assert task.lease_expires_at is None
    assert task.lease_attempt_id is None


def test_stale_running_task_ignores_child_agent_checkpoint(db_session) -> None:
    task = _create_task(db_session, status=TaskStatus.RUNNING)
    task.runner_id = "dead-runner"
    task.run_id = "child-checkpoint-run"
    task.lease_expires_at = utc_now() - timedelta(seconds=1)
    task.last_checkpoint_event_id = "child-checkpoint-1"
    child_checkpoint_event = TraceEvent(
        task_id=task.id,
        build_id="agent_123_child",
        event_id="child-checkpoint-1",
        event_type="system_update_general",
        timestamp=utc_now(),
        step_id=None,
        parent_event_id=None,
        data={
            "checkpoint_type": CHECKPOINT_TYPE,
            "snapshot": {"type": "checkpoint"},
            TASK_RUN_ID_TRACE_FIELD: "child-checkpoint-run",
        },
    )
    db_session.add(child_checkpoint_event)
    db_session.flush()
    # Exercises the PK-anchored resolution path: an anchored build-scoped
    # row must fail its own validation (build_id is not None), the same
    # FAILED verdict the legacy scan reaches -- and it must not fall back
    # to searching other rows for a match.
    task.last_checkpoint_trace_event_id = child_checkpoint_event.id
    db_session.commit()

    assert _recover_expired_task(db_session, task) == TaskStatus.FAILED
    db_session.refresh(task)
    assert task.status == TaskStatus.FAILED
    assert task.runner_id is None
    assert task.lease_expires_at is None


def test_stale_running_task_with_legacy_checkpoint_becomes_paused(db_session) -> None:
    task = _create_task(db_session, status=TaskStatus.RUNNING)
    task.runner_id = "dead-runner"
    task.run_id = "legacy-checkpoint-run"
    task.lease_expires_at = utc_now() - timedelta(seconds=1)
    task.last_checkpoint_event_id = "legacy-checkpoint-1"
    legacy_checkpoint_event = TraceEvent(
        task_id=task.id,
        event_id="legacy-checkpoint-1",
        event_type="system_update_general",
        timestamp=utc_now(),
        step_id=None,
        parent_event_id=None,
        data={
            "checkpoint_type": next(iter(LEGACY_CHECKPOINT_TYPES)),
            "snapshot": {"type": "checkpoint"},
            TASK_RUN_ID_TRACE_FIELD: "legacy-checkpoint-run",
        },
    )
    db_session.add(legacy_checkpoint_event)
    db_session.flush()
    task.last_checkpoint_trace_event_id = legacy_checkpoint_event.id
    db_session.commit()

    assert _recover_expired_task(db_session, task) == TaskStatus.PAUSED
    db_session.refresh(task)
    assert task.status == TaskStatus.PAUSED
    assert task.runner_id is None
    assert task.lease_expires_at is None


def test_acquire_stamps_a_fresh_attempt_id_onto_the_task_row(db_session) -> None:
    """acquire -> tasks.lease_attempt_id -> TaskLease.attempt_id round-trips.

    The lease is not stamped from RETURNING (see the values-dict read-back in
    acquire_task_lease_no_commit), so the only thing proving the returned
    identity is the one actually persisted is this three-way comparison.
    """
    task = _create_task(db_session, status=TaskStatus.PENDING)

    lease = acquire_task_lease(db_session, int(task.id), runner_id="runner-a")

    assert lease is not None
    assert lease.attempt_id is not None
    # Not a placeholder: it must parse as a uuid and fit String(64).
    uuid.UUID(lease.attempt_id)
    assert len(lease.attempt_id) <= 64

    db_session.refresh(task)
    assert task.lease_attempt_id == lease.attempt_id


def test_each_acquisition_mints_a_new_attempt_id(db_session) -> None:
    """A new value per claim -- this is what state_version cannot do."""
    task = _create_task(db_session, status=TaskStatus.PENDING)

    first = acquire_task_lease(db_session, int(task.id), runner_id="runner-a")
    assert first is not None
    release_task_lease(db_session, first, status=TaskStatus.PAUSED)

    second = acquire_task_lease(db_session, int(task.id), runner_id="runner-a")
    assert second is not None

    assert first.attempt_id != second.attempt_id
    db_session.refresh(task)
    assert task.lease_attempt_id == second.attempt_id


def test_same_run_resume_still_rotates_the_attempt_id(db_session) -> None:
    """expected_run_id keeps the run but not the attempt: this is exactly the
    shape where a successor run reacquires the SAME (runner_id, run_id)
    tuple and must still be distinguishable from the stale run holding it."""
    task = _create_task(db_session, status=TaskStatus.WAITING_FOR_USER)
    task.run_id = "resumable-run"
    db_session.commit()

    first = acquire_task_lease(
        db_session,
        int(task.id),
        runner_id="same-runner",
        expected_run_id="resumable-run",
    )
    second = acquire_task_lease(
        db_session,
        int(task.id),
        runner_id="same-runner",
        expected_run_id="resumable-run",
    )

    assert first is not None and second is not None
    assert (first.runner_id, first.run_id) == (second.runner_id, second.run_id)
    assert first.attempt_id != second.attempt_id


def test_new_run_acquisition_stamps_an_attempt_id(db_session) -> None:
    """The stamp lives in the values-dict base, not a conditional branch --
    it must fire on the new_run=True path too."""
    task = _create_task(db_session, status=TaskStatus.PENDING)

    lease = acquire_task_lease(
        db_session, int(task.id), runner_id="runner-a", new_run=True
    )

    assert lease is not None
    assert lease.attempt_id is not None
    uuid.UUID(lease.attempt_id)
    db_session.refresh(task)
    assert task.lease_attempt_id == lease.attempt_id


def test_release_clears_the_attempt_id(db_session) -> None:
    """The column must not carry a finished attempt's value."""
    task = _create_task(db_session, status=TaskStatus.PENDING)
    lease = acquire_task_lease(db_session, int(task.id), runner_id="runner-a")
    assert lease is not None
    db_session.refresh(task)
    assert task.lease_attempt_id is not None

    release_task_lease(db_session, lease, status=TaskStatus.PAUSED)

    db_session.refresh(task)
    assert task.lease_attempt_id is None


def test_expired_lease_recovery_clears_the_attempt_id(db_session) -> None:
    """Same invariant as release, exercised through expiry recovery.

    No checkpoint is set up, so recovery lands on FAILED rather than PAUSED
    -- that branch distinction is orthogonal to what this test pins.
    """
    task = _create_task(db_session, status=TaskStatus.RUNNING)
    task.runner_id = "dead-runner"
    task.run_id = "expiring-run"
    task.lease_expires_at = utc_now() - timedelta(seconds=1)
    task.lease_attempt_id = "stale-attempt"
    db_session.commit()
    assert task.lease_attempt_id is not None

    assert _recover_expired_task(db_session, task) == TaskStatus.FAILED

    db_session.refresh(task)
    assert task.lease_attempt_id is None


def test_fail_and_release_clears_the_attempt_id(db_session) -> None:
    """Same invariant, exercised through the fail-and-release writer."""
    task = _create_task(db_session)
    lease = acquire_task_lease(db_session, int(task.id), runner_id="runner-a")
    assert lease is not None
    db_session.refresh(task)
    assert task.lease_attempt_id is not None

    changed = task_lease_service.fail_and_release_task_lease_no_commit(
        db_session,
        lease,
        error_message="setup failed",
    )
    db_session.commit()

    assert changed is True
    db_session.refresh(task)
    assert task.lease_attempt_id is None


def test_release_current_runner_task_lease_clears_the_attempt_id(db_session) -> None:
    """Same invariant, exercised through the current-runner release path."""
    task = _create_task(db_session, status=TaskStatus.PENDING)
    lease = acquire_task_lease(db_session, int(task.id), runner_id="runner-a")
    assert lease is not None
    db_session.refresh(task)
    assert task.lease_attempt_id is not None

    changed = task_lease_service.release_current_runner_task_lease(
        db_session,
        int(task.id),
        status=TaskStatus.PAUSED,
        runner_id="runner-a",
    )

    assert changed is True
    db_session.refresh(task)
    assert task.lease_attempt_id is None


def test_task_lease_snapshot_never_carries_an_attempt_id() -> None:
    """The ambient snapshot rebuilt from a task row must keep attempt_id None
    even when the row has one, so a later attempt check cannot compare a
    value against itself and always pass."""
    from types import SimpleNamespace

    from xagent.web.api.websocket import _task_lease_snapshot

    row = SimpleNamespace(
        id=7,
        runner_id="runner-a",
        run_id="run-b",
        lease_attempt_id="attempt-c",
    )
    lease = _task_lease_snapshot(row)
    assert lease is not None
    assert lease.attempt_id is None
