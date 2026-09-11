import asyncio
from dataclasses import replace
from datetime import timedelta

import pytest
from sqlalchemy import select, text, update
from sqlalchemy.exc import DataError, OperationalError
from sqlalchemy.orm import sessionmaker

from tests.web.services.test_task_execution_event_store import engine as engine_fixture
from tests.web.services.test_task_execution_event_store import (
    task_id as task_id_fixture,
)
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.task import Task, TaskStatus
from xagent.web.services import task_lease_service as ls

engine = engine_fixture
task_id = task_id_fixture
DEFERRED = ls.TaskLeaseRefreshState.DEFERRED


@pytest.fixture
def scenario(engine, task_id, monkeypatch):
    if engine.dialect.name != "postgresql":
        pytest.skip("PostgreSQL row locking")
    factory = sessionmaker(engine)
    monkeypatch.setattr("xagent.web.models.database.get_session_local", lambda: factory)
    with factory() as db:
        uid = db.get(Task, task_id).user_id
        ids = [task_id]
        for title in ["B", "C"]:
            row = Task(user_id=uid, title=title)
            db.add(row)
            db.flush()
            ids.append(row.id)
        db.commit()
        batch = tuple(
            ls.acquire_task_lease(db, tid, runner_id="probe", new_run=True)
            for tid in ids
        )
        db.execute(
            update(Task)
            .where(Task.id.in_(ids))
            .values(last_heartbeat_at=ls.utc_now() - timedelta(seconds=20))
        )
        db.commit()
    return factory, batch, uid


def heartbeats(factory, batch):
    with factory() as db:
        return dict(
            db.execute(
                select(Task.id, Task.last_heartbeat_at).where(
                    Task.id.in_([lease.task_id for lease in batch])
                )
            ).all()
        )


def test_blocked_middle_row_healthy_rows_commit_then_retry(scenario):
    factory, batch, _ = scenario
    before = heartbeats(factory, batch)
    with factory() as blocker:
        blocker.execute(
            update(Task)
            .where(Task.id == batch[1].task_id)
            .values(updated_at=Task.updated_at)
        )
        states = ls.refresh_task_leases_isolated(batch)
        after = heartbeats(factory, batch)
        assert [states[ls._task_lease_key(lease)] for lease in batch] == [
            "refreshed",
            DEFERRED,
            "refreshed",
        ]
        assert after[batch[0].task_id] > before[batch[0].task_id]
        assert after[batch[1].task_id] == before[batch[1].task_id]
        assert after[batch[2].task_id] > before[batch[2].task_id]
        # Three retries must not write B while the conflicting lock remains.
        for _ in range(3):
            assert (
                ls.refresh_task_leases_isolated((batch[1],))[
                    ls._task_lease_key(batch[1])
                ]
                == DEFERRED
            )
        assert heartbeats(factory, batch) == after
        blocker.rollback()
    assert (
        ls.refresh_task_leases_isolated((batch[1],))[ls._task_lease_key(batch[1])]
        == "refreshed"
    )
    final = heartbeats(factory, batch)
    assert final[batch[1].task_id] > before[batch[1].task_id]
    assert final[batch[0].task_id] == after[batch[0].task_id]


def test_no_key_update_preserves_fk_key_share_compatibility(scenario):
    factory, batch, uid = scenario
    with factory() as child:
        child.add(
            TaskChatMessage(
                task_id=batch[1].task_id,
                user_id=uid,
                role="assistant",
                message_type="assistant_response",
                content="uncommitted",
            )
        )
        child.flush()
        assert set(ls.refresh_task_leases_isolated(batch).values()) == {"refreshed"}
        child.rollback()


@pytest.mark.parametrize("change", ["attempt", "runner", "run", "terminal", "delete"])
def test_uncommitted_changes_defer_then_classify_committed_state(scenario, change):
    factory, batch, _ = scenario
    lease = batch[1]
    with factory() as writer:
        row = writer.get(Task, lease.task_id)
        if change == "attempt":
            row.lease_attempt_id = "new-attempt"
        elif change == "runner":
            row.runner_id = "new-runner"
        elif change == "run":
            row.run_id = "new-run"
        elif change == "terminal":
            row.status = TaskStatus.COMPLETED
        else:
            writer.delete(row)
        writer.flush()
        result = ls.refresh_task_leases_isolated(batch)
        assert result[ls._task_lease_key(lease)] == DEFERRED
        writer.commit()
    result = ls.refresh_task_leases_isolated((lease,))
    assert result[ls._task_lease_key(lease)] == (
        "settlement_ready" if change == "terminal" else "lost"
    )


def test_sql_failure_rolls_back_healthy_updates(scenario, monkeypatch):
    factory, batch, _ = scenario
    before = heartbeats(factory, batch)
    original_commit = factory.class_.commit

    def fail_commit(db):
        db.execute(text("select 1/0"))
        original_commit(db)

    monkeypatch.setattr(factory.class_, "commit", fail_commit)
    with pytest.raises(DataError, match="division by zero"):
        ls.refresh_task_leases_isolated(batch)
    assert heartbeats(factory, batch) == before


def test_table_lock_is_not_skipped_and_hits_database_timeout(scenario):
    factory, batch, _ = scenario
    with factory() as blocker:
        blocker.execute(text("LOCK TABLE tasks IN ACCESS EXCLUSIVE MODE"))
        with pytest.raises(OperationalError, match="lock timeout"):
            ls.refresh_task_leases_isolated(batch)
        blocker.rollback()
    assert set(ls.refresh_task_leases_isolated(batch).values()) == {"refreshed"}


def test_long_locked_task_can_expire_while_healthy_tasks_renew(scenario):
    factory, batch, _ = scenario
    with factory() as db:
        db.execute(
            update(Task)
            .where(Task.id == batch[1].task_id)
            .values(lease_expires_at=ls.utc_now() - timedelta(seconds=1))
        )
        db.commit()
    with factory() as blocker:
        blocker.execute(
            update(Task)
            .where(Task.id == batch[1].task_id)
            .values(updated_at=Task.updated_at)
        )
        result = ls.refresh_task_leases_isolated(batch)
        assert result[ls._task_lease_key(batch[1])] == DEFERRED
        with factory() as observer:
            expired = observer.scalar(
                select(Task.lease_expires_at).where(Task.id == batch[1].task_id)
            )
            assert expired < ls.utc_now()
        blocker.rollback()


def test_old_and_new_attempt_in_same_batch_must_not_share_success(scenario):
    factory, batch, _ = scenario
    old = batch[1]
    with factory() as db:
        current = ls.acquire_task_lease(
            db, old.task_id, runner_id=old.runner_id, expected_run_id=old.run_id
        )
    states = ls.refresh_task_leases_isolated((old, current))
    assert states[ls._task_lease_key(old)] == "lost"
    assert states[ls._task_lease_key(current)] == "refreshed"


@pytest.mark.parametrize("field", ["run_id", "attempt_id"])
def test_missing_identity_must_never_renew(scenario, field):
    factory, batch, _ = scenario
    lease = replace(batch[1], **{field: None})
    with factory() as db:
        db.execute(
            update(Task)
            .where(Task.id == lease.task_id)
            .values(**{"run_id" if field == "run_id" else "lease_attempt_id": None})
        )
        db.commit()
    assert (
        ls.refresh_task_leases_isolated((lease,))[ls._task_lease_key(lease)] == "lost"
    )


def _manager_with_clock(monkeypatch):
    """Advance heartbeat deadlines without wall-clock sleeps or short SQL budgets."""
    from unittest.mock import Mock

    clock = Mock(wraps=asyncio.get_running_loop())
    clock.time.return_value = 0.0
    monkeypatch.setattr(ls, "get_task_lease_heartbeat_seconds", lambda: 20)
    manager = ls._TaskLeaseHeartbeatManager(clock)
    settled = asyncio.Event()
    settle = manager._settle_refresh_waiter

    def record(*args, **kwargs):
        settle(*args, **kwargs)
        settled.set()

    monkeypatch.setattr(manager, "_settle_refresh_waiter", record)
    return manager, clock, settled


async def _advance_heartbeat(manager, clock, settled, at):
    settled.clear()
    clock.time.return_value = at
    manager._wake_event.set()
    await asyncio.wait_for(settled.wait(), timeout=2)


@pytest.mark.asyncio
async def test_deferred_subset_retries_without_speeding_up_healthy_leases(monkeypatch):
    manager, clock, settled = _manager_with_clock(monkeypatch)
    batch = tuple(
        ls.TaskLease(task_id=i, runner_id="r", run_id="run", attempt_id="a")
        for i in range(3)
    )
    calls = []

    def refresh(items):
        calls.append(items)
        return {
            ls._task_lease_key(lease): DEFERRED
            if lease == batch[1] and len(calls) <= 2
            else ls.TaskLeaseRefreshState.REFRESHED
            for lease in items
        }

    monkeypatch.setattr(ls, "refresh_task_leases_isolated", refresh)
    registrations = [manager.register(lease) for lease in batch]
    previous_error = RuntimeError("previous checkout timeout")
    registrations[1]._entry.outcome = ls.TaskLeaseHeartbeatOutcome(
        pool_timeout=previous_error
    )
    try:
        await asyncio.sleep(0)
        await _advance_heartbeat(manager, clock, settled, 20)
        assert registrations[1]._entry.outcome.pool_timeout is previous_error
        assert not registrations[1].terminal_event.is_set()
        await _advance_heartbeat(manager, clock, settled, 21)
        assert registrations[1]._entry.outcome.pool_timeout is previous_error
        assert registrations[1]._entry.deferred_since == 20
        await _advance_heartbeat(manager, clock, settled, 22)
        assert registrations[1]._entry.outcome.pool_timeout is None
        assert registrations[1]._entry.retry_at is None
        assert registrations[1]._entry.deferred_since is None
        await _advance_heartbeat(manager, clock, settled, 40)
        assert calls == [batch, (batch[1],), (batch[1],), batch]
    finally:
        for registration in registrations:
            await registration.close()
        await manager.wait_until_idle()


@pytest.mark.asyncio
async def test_retry_database_error_waits_for_normal_tick(monkeypatch):
    manager, clock, settled = _manager_with_clock(monkeypatch)
    lease = ls.TaskLease(task_id=1, runner_id="r", run_id="run", attempt_id="a")
    calls = []

    def refresh(items):
        calls.append(items)
        if len(calls) == 2:
            raise RuntimeError("database error during retry")
        return {ls._task_lease_key(lease): DEFERRED}

    monkeypatch.setattr(ls, "refresh_task_leases_isolated", refresh)
    registration = manager.register(lease)
    try:
        await asyncio.sleep(0)
        await _advance_heartbeat(manager, clock, settled, 20)
        await _advance_heartbeat(manager, clock, settled, 21)
        assert registration._entry.retry_at is None
        assert not registration.terminal_event.is_set()
        await _advance_heartbeat(manager, clock, settled, 40)
        assert len(calls) == 3
        assert registration._entry.retry_at == 41
        await asyncio.wait_for(registration.close(), 1)
        await asyncio.wait_for(manager.wait_until_idle(), 1)
        assert len(calls) == 3  # Closing does not wait for a future retry.
    finally:
        await registration.close()
        await manager.wait_until_idle()


@pytest.mark.asyncio
async def test_late_deferred_result_cannot_schedule_replacement_registration(
    monkeypatch,
):
    from threading import Event

    manager, clock, settled = _manager_with_clock(monkeypatch)
    lease = ls.TaskLease(task_id=1, runner_id="r", run_id="run", attempt_id="a")
    ready, release = Event(), Event()
    calls = []

    def refresh(items):
        calls.append(items)
        if len(calls) == 1:
            ready.set()
            assert release.wait(5)
            return {ls._task_lease_key(lease): DEFERRED}
        return {
            ls._task_lease_key(item): ls.TaskLeaseRefreshState.REFRESHED
            for item in items
        }

    monkeypatch.setattr(ls, "refresh_task_leases_isolated", refresh)
    old = manager.register(lease)
    new = None
    closing = None
    try:
        await asyncio.sleep(0)
        clock.time.return_value = 20
        manager._wake_event.set()
        assert await asyncio.to_thread(ready.wait, 2)
        closing = asyncio.create_task(old.close())
        await asyncio.sleep(0)
        assert not closing.done()
        # Same-key replacement is stricter than merely changing attempt: the
        # old in-flight entry must not attach a retry to the new entry object.
        new = manager.register(lease)
        assert old._entry is not new._entry
        release.set()
        await asyncio.wait_for(closing, 2)
        assert new._entry.retry_at is None
        await _advance_heartbeat(manager, clock, settled, 40)
        assert calls == [(lease,), (lease,)]
        assert not new.terminal_event.is_set()
    finally:
        release.set()
        if closing is not None:
            await closing
        await old.close()
        if new is not None:
            await new.close()
        await manager.wait_until_idle()


@pytest.mark.asyncio
async def test_manager_retries_real_locked_lease_after_healthy_commit(
    scenario, monkeypatch
):
    factory, batch, _ = scenario
    manager, clock, settled = _manager_with_clock(monkeypatch)
    original_refresh = ls.refresh_task_leases_isolated
    calls = []

    def refresh(items):
        calls.append(items)
        return original_refresh(items)

    monkeypatch.setattr(ls, "refresh_task_leases_isolated", refresh)
    registrations = [manager.register(lease) for lease in batch]
    before = heartbeats(factory, batch)
    try:
        await asyncio.sleep(0)
        with factory() as blocker:
            blocker.execute(
                update(Task)
                .where(Task.id == batch[1].task_id)
                .values(updated_at=Task.updated_at)
            )
            await _advance_heartbeat(manager, clock, settled, 20)
            after = heartbeats(factory, batch)
            assert after[batch[0].task_id] > before[batch[0].task_id]
            assert after[batch[1].task_id] == before[batch[1].task_id]
            assert after[batch[2].task_id] > before[batch[2].task_id]
            assert not registrations[1].terminal_event.is_set()
            blocker.rollback()
        await _advance_heartbeat(manager, clock, settled, 21)
        final = heartbeats(factory, batch)
        assert final[batch[1].task_id] > before[batch[1].task_id]
        assert final[batch[0].task_id] == after[batch[0].task_id]
        assert final[batch[2].task_id] == after[batch[2].task_id]
        assert calls == [batch, (batch[1],)]
    finally:
        for registration in registrations:
            await registration.close()
        await manager.wait_until_idle()


def test_preexecution_validation_waits_for_definitive_ownership(scenario, monkeypatch):
    import time
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from sqlalchemy.orm import Session

    factory, batch, _ = scenario
    lease = batch[1]
    ready = Event()
    pids = []

    class ValidationSession(Session):
        def __enter__(self):
            super().__enter__()
            self.execute(text("SET LOCAL statement_timeout='5s'"))
            pids.append(self.scalar(text("select pg_backend_pid()")))
            ready.set()
            return self

    validation_factory = sessionmaker(factory.kw["bind"], class_=ValidationSession)
    monkeypatch.setattr(
        "xagent.web.models.database.get_session_local", lambda: validation_factory
    )
    with factory() as blocker, ThreadPoolExecutor(1) as pool:
        blocker.execute(
            update(Task)
            .where(Task.id == lease.task_id)
            .values(lease_attempt_id="successor")
        )
        future = pool.submit(ls.validate_preacquired_task_lease_isolated, lease)
        try:
            assert ready.wait(2)
            with factory() as observer:
                deadline = time.monotonic() + 2
                while not observer.scalar(
                    text("select pg_blocking_pids(:pid)"), {"pid": pids[0]}
                ):
                    assert not future.done(), (
                        "Admission must not return DEFERRED as successful validation"
                    )
                    assert time.monotonic() < deadline
                    time.sleep(0.005)
        finally:
            blocker.commit()
        assert future.result(timeout=5) == ls.TaskLeaseRefreshState.LOST


@pytest.mark.asyncio
async def test_slow_subset_retry_does_not_skip_normal_batch_deadline(monkeypatch):
    manager, clock, settled = _manager_with_clock(monkeypatch)
    batch = tuple(
        ls.TaskLease(task_id=i, runner_id="r", run_id="run", attempt_id="a")
        for i in range(2)
    )
    calls = []
    normal_batch = asyncio.Event()
    loop = asyncio.get_running_loop()

    def refresh(items):
        calls.append(items)
        if len(calls) == 2:
            clock.time.return_value = 42  # Retry crossed the normal tick at 40.
        if len(calls) == 3:
            loop.call_soon_threadsafe(normal_batch.set)
        return {
            ls._task_lease_key(lease): DEFERRED
            if lease == batch[1] and len(calls) == 1
            else ls.TaskLeaseRefreshState.REFRESHED
            for lease in items
        }

    monkeypatch.setattr(ls, "refresh_task_leases_isolated", refresh)
    registrations = [manager.register(lease) for lease in batch]
    try:
        await asyncio.sleep(0)
        await _advance_heartbeat(manager, clock, settled, 20)
        await _advance_heartbeat(manager, clock, settled, 21)
        await asyncio.wait_for(normal_batch.wait(), 2)
        assert calls == [batch, (batch[1],), batch]
    finally:
        for registration in registrations:
            await registration.close()
        await manager.wait_until_idle()


@pytest.mark.asyncio
async def test_repeated_cancellation_drains_inflight_retry(monkeypatch):
    from threading import Event

    manager, clock, settled = _manager_with_clock(monkeypatch)
    monkeypatch.setattr(ls, "_get_task_lease_heartbeat_manager", lambda: manager)
    lease = ls.TaskLease(task_id=1, runner_id="r", run_id="run", attempt_id="a")
    ready, release, finished = Event(), Event(), Event()
    calls = []

    def refresh(items):
        calls.append(items)
        if len(calls) == 1:
            return {ls._task_lease_key(lease): DEFERRED}
        try:
            ready.set()
            assert release.wait(5)
            return {ls._task_lease_key(lease): ls.TaskLeaseRefreshState.REFRESHED}
        finally:
            finished.set()

    monkeypatch.setattr(ls, "refresh_task_leases_isolated", refresh)
    heartbeat = asyncio.create_task(ls.run_task_lease_heartbeat(lease, asyncio.Event()))
    try:
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        await _advance_heartbeat(manager, clock, settled, 20)
        clock.time.return_value = 21
        manager._wake_event.set()
        assert await asyncio.to_thread(ready.wait, 2)
        for _ in range(2):
            heartbeat.cancel()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not heartbeat.done()
            assert not finished.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(heartbeat, 2)
        assert finished.is_set()
        await asyncio.wait_for(manager.wait_until_idle(), 2)
        assert len(calls) == 2
        assert not manager._entries
    finally:
        release.set()
        if not heartbeat.done():
            heartbeat.cancel()
        await asyncio.gather(heartbeat, return_exceptions=True)
        await manager.wait_until_idle()
