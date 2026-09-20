"""Batch owner renewals, real row locks, and shutdown transaction ordering."""

import asyncio
import os
from dataclasses import replace
from datetime import timedelta
from threading import Event
from unittest.mock import AsyncMock

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from tests.shared.postgres_disposable import psycopg2_kwargs
from tests.web.services.task_database_shared import engine as engine_fixture
from tests.web.services.task_database_shared import task_id as task_id_fixture
from xagent.web.models.task import Task, TaskStatus
from xagent.web.services import task_coordinator_runtime as runtime
from xagent.web.services import task_coordinator_service as service

engine = engine_fixture
task_id = task_id_fixture


@pytest.fixture(params=["psycopg2", "psycopg"])
def renewal_engine(engine, request):
    if engine.dialect.name != "postgresql" or request.param == "psycopg2":
        yield engine
        return
    psycopg = pytest.importorskip("psycopg")
    with engine.connect() as db:
        dbname = db.scalar(sa.text("SELECT current_database()"))
    kwargs = psycopg2_kwargs(os.environ["XAGENT_TEST_POSTGRES_URL"], dbname)
    result = sa.create_engine(
        "postgresql+psycopg://", creator=lambda: psycopg.connect(**kwargs)
    )
    try:
        yield result
    finally:
        result.dispose()


@pytest.fixture
def owners(engine, task_id):
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as db, db.begin():
        first = db.get(Task, task_id)
        ids = [task_id]
        for status in (TaskStatus.WAITING_FOR_USER, TaskStatus.COMPLETED):
            task = Task(user_id=first.user_id, title=status.value, status=status)
            db.add(task)
            db.flush()
            ids.append(task.id)
        first.status = TaskStatus.PAUSED
        leases = tuple(
            service.acquire_task_lease_no_commit(db, tid, runner_id="owner")
            for tid in ids
        )
        db.execute(
            sa.update(Task)
            .where(Task.id.in_(ids))
            .values(last_heartbeat_at=service.utc_now() - timedelta(seconds=20))
        )
    return factory, leases


def times(factory, leases):
    with factory() as db:
        return dict(
            db.execute(
                sa.select(Task.id, Task.last_heartbeat_at).where(
                    Task.id.in_([lease.task_id for lease in leases])
                )
            ).all()
        )


def test_batch_keeps_non_running_owners_and_exact_attempts(owners):
    factory, leases = owners
    before = times(factory, leases)
    with factory() as db:
        originals = {
            t.id: (t.status, t.run_id, t.updated_at)
            for t in db.scalars(sa.select(Task))
        }
    stale = replace(leases[1], attempt_id="stale")
    with factory() as db, db.begin():
        states = service.renew_task_leases_no_commit(db, (leases[0], stale, leases[2]))
    assert states == {leases[0]: "renewed", stale: "lost", leases[2]: "renewed"}
    after = times(factory, leases)
    assert after[leases[0].task_id] > before[leases[0].task_id]
    assert after[leases[1].task_id] == before[leases[1].task_id]
    assert after[leases[2].task_id] > before[leases[2].task_id]
    with factory() as db:
        assert {
            t.id: (t.status, t.run_id, t.updated_at)
            for t in db.scalars(sa.select(Task))
        } == originals


def test_postgresql_skips_locked_owner_without_rolling_back_healthy_rows(
    engine, owners
):
    if engine.dialect.name != "postgresql":
        pytest.skip("PostgreSQL row locks")
    factory, leases = owners
    before = times(factory, leases)
    with factory() as blocker:
        blocker.execute(
            sa.update(Task)
            .where(Task.id == leases[1].task_id)
            .values(updated_at=Task.updated_at)
        )
        with factory() as db, db.begin():
            states = service.renew_task_leases_no_commit(db, leases)
        assert [states[lease] for lease in leases] == ["renewed", "deferred", "renewed"]
        after = times(factory, leases)
        assert after[leases[0].task_id] > before[leases[0].task_id]
        assert after[leases[1].task_id] == before[leases[1].task_id]
        assert after[leases[2].task_id] > before[leases[2].task_id]
        # Uncommitted replacement cannot be mistaken for a committed loss.
        blocker.execute(
            sa.update(Task)
            .where(Task.id == leases[1].task_id)
            .values(lease_attempt_id="successor")
        )
        with factory() as db, db.begin():
            assert (
                service.renew_task_leases_no_commit(db, (leases[1],))[leases[1]]
                == "deferred"
            )
        blocker.commit()
    with factory() as db, db.begin():
        assert (
            service.renew_task_leases_no_commit(db, (leases[1],))[leases[1]] == "lost"
        )


async def until(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.005)


async def test_registry_single_connection_skips_lock_and_retries_subset(
    engine, task_id, monkeypatch
):
    if engine.dialect.name != "postgresql":
        pytest.skip("PostgreSQL row locks")
    monkeypatch.setattr(runtime, "get_task_lease_heartbeat_seconds", lambda: 0.4)
    app_engine = sa.create_engine(
        "postgresql://",
        creator=engine.pool._creator,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.1,
    )
    factory = sessionmaker(app_engine, expire_on_commit=False)
    admin = sessionmaker(engine)
    with admin() as db, db.begin():
        task = Task(user_id=db.get(Task, task_id).user_id, title="healthy")
        db.add(task)
        db.flush()
        healthy_id = task.id
    registry = runtime.TaskCoordinatorRegistry(factory)
    start_batch = asyncio.Event()
    run_batches = registry._run_heartbeats

    async def start_when_ready():
        await start_batch.wait()
        await run_batches()

    monkeypatch.setattr(registry, "_run_heartbeats", start_when_ready)
    batches = []
    original = registry._renew

    def renew(leases):
        result = original(leases)
        batches.append((leases, result))
        return result

    monkeypatch.setattr(registry, "_renew", renew)
    blocker = admin()
    try:
        blocked = await registry.ensure(task_id)
        healthy = await registry.ensure(healthy_id)
        blocker.execute(
            sa.update(Task).where(Task.id == task_id).values(updated_at=Task.updated_at)
        )
        start_batch.set()
        await until(lambda: blocked._retry_at is not None)
        assert blocked._healthy
        assert healthy._healthy
        assert healthy._heartbeat_error is None
        assert batches[0][1] == {blocked.lease: "deferred", healthy.lease: "renewed"}
        blocker.rollback()
        await until(
            lambda: any(
                leases == (blocked.lease,) and results[blocked.lease] == "renewed"
                for leases, results in batches
            )
        )
        assert blocked.state == runtime.CoordinatorState.ACTIVE
    finally:
        start_batch.set()
        blocker.close()
        await registry.close()
        app_engine.dispose()
    assert registry._heartbeats == {}
    assert registry._heartbeat_runner is None


async def test_close_waits_for_inflight_batch_without_cancelling_other_owner(
    engine, task_id, monkeypatch
):
    monkeypatch.setattr(runtime, "get_task_lease_heartbeat_seconds", lambda: 0.05)
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as db, db.begin():
        task = Task(user_id=db.get(Task, task_id).user_id, title="survivor")
        db.add(task)
        db.flush()
        second_id = task.id
    registry = runtime.TaskCoordinatorRegistry(factory)
    start_batch = asyncio.Event()
    run_batches = registry._run_heartbeats

    async def start_when_ready():
        await start_batch.wait()
        await run_batches()

    monkeypatch.setattr(registry, "_run_heartbeats", start_when_ready)
    entered, unblock = Event(), Event()
    survivor_renewed = asyncio.Event()
    original = registry._renew
    calls = 0
    loop = asyncio.get_running_loop()

    def renew(leases):
        nonlocal calls
        calls += 1
        result = original(leases)
        if calls == 1:
            entered.set()
            assert unblock.wait(5)
        elif any(lease.task_id == second_id for lease in leases):
            loop.call_soon_threadsafe(survivor_renewed.set)
        return result

    monkeypatch.setattr(registry, "_renew", renew)
    closing = None
    try:
        first = await registry.ensure(task_id)
        second = await registry.ensure(second_id)
        start_batch.set()
        assert await asyncio.to_thread(entered.wait, 5)
        closing = asyncio.create_task(first.close())
        await until(lambda: first.lease not in registry._heartbeats)
        assert not first._heartbeat_done.done()
        assert not first._close_task.done()
        closing.cancel()
        with factory() as db:
            assert db.get(Task, task_id).lease_attempt_id == first.lease.attempt_id
        unblock.set()
        with pytest.raises(asyncio.CancelledError):
            await closing
        await first.close()
        await asyncio.wait_for(survivor_renewed.wait(), 5)
        assert second._healthy
        with factory() as db:
            assert db.get(Task, task_id).runner_id is None
            assert db.get(Task, second_id).lease_attempt_id == second.lease.attempt_id
    finally:
        start_batch.set()
        unblock.set()
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        await registry.close()


async def test_postgresql_table_lock_defers_whole_batch(engine, owners):
    if engine.dialect.name != "postgresql":
        pytest.skip("PostgreSQL statement timeout")
    factory, leases = owners
    registry = runtime.TaskCoordinatorRegistry(factory)
    before = times(factory, leases)
    with factory() as blocker:
        blocker.execute(sa.text("LOCK TABLE tasks IN ACCESS EXCLUSIVE MODE"))
        with pytest.raises(sa.exc.DBAPIError) as raised:
            await asyncio.to_thread(registry._renew, leases)
        assert raised.value.orig.pgcode == "55P03"
        blocker.rollback()
    assert times(factory, leases) == before
    assert all(state == "renewed" for state in registry._renew(leases).values())


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("commit failed"),
        sa.exc.ProgrammingError("invalid SQL", {}, ValueError("invalid SQL")),
        sa.exc.OperationalError("invalid SQLite SQL", {}, ValueError("invalid SQL")),
    ],
)
async def test_batch_commit_failure_is_not_acknowledged(
    engine, task_id, monkeypatch, failure
):
    monkeypatch.setattr(runtime, "get_task_lease_heartbeat_seconds", lambda: 0.02)
    factory = sessionmaker(engine, expire_on_commit=False)
    registry = runtime.TaskCoordinatorRegistry(factory)
    try:
        owner = await registry.ensure(task_id)
        before = times(factory, (owner.lease,))

        def fail_commit(session):
            raise failure

        sa.event.listen(factory, "before_commit", fail_commit)
        try:
            await until(lambda: owner.state == runtime.CoordinatorState.CLOSED)
            assert owner._recovery_required
            assert not owner._healthy
            assert owner._heartbeat_done.exception() is failure
            assert times(factory, (owner.lease,)) == before
        finally:
            sa.event.remove(factory, "before_commit", fail_commit)
    finally:
        await registry.close()


async def test_brief_deferral_still_allows_command_and_close_release(
    engine, task_id, monkeypatch
):
    if engine.dialect.name != "postgresql":
        pytest.skip("PostgreSQL row locks")
    monkeypatch.setattr(runtime, "get_task_lease_heartbeat_seconds", lambda: 0.2)
    factory = sessionmaker(engine, expire_on_commit=False)
    registry = runtime.TaskCoordinatorRegistry(factory)
    blocker = factory()
    try:
        owner = await registry.ensure(task_id)
        blocker.execute(
            sa.update(Task).where(Task.id == task_id).values(updated_at=Task.updated_at)
        )
        await until(lambda: owner._retry_at is not None)
        assert owner._healthy
        # A control command need not acquire the locked task row itself.
        from types import SimpleNamespace

        execute = AsyncMock(return_value="applied")
        assert (
            await owner.execute_command(
                SimpleNamespace(kind=SimpleNamespace(value="stop")), execute
            )
            == "applied"
        )
        blocker.rollback()
        await owner.close()
        with factory() as db:
            assert db.get(Task, task_id).lease_attempt_id is None
    finally:
        blocker.close()
        await registry.close()


async def test_sustained_deferral_degrades_once_and_success_restores_admission(
    engine, task_id, monkeypatch, caplog
):
    monkeypatch.setattr(runtime, "get_task_lease_heartbeat_seconds", lambda: 0.1)
    factory = sessionmaker(engine, expire_on_commit=False)
    registry = runtime.TaskCoordinatorRegistry(factory)
    original = registry._renew
    attempts = 0

    def defer(leases):
        nonlocal attempts
        attempts += 1
        return {lease: service.TaskLeaseRenewalState.DEFERRED for lease in leases}

    monkeypatch.setattr(registry, "_renew", defer)
    try:
        owner = await registry.ensure(task_id)
        await until(lambda: owner._retry_at is not None)
        assert owner._healthy
        owner._last_renewed_at -= runtime.get_task_lease_ttl_seconds()
        await until(lambda: not owner._healthy)
        assert owner.state == runtime.CoordinatorState.ACTIVE
        assert (
            owner.submit_execution(
                admit=lambda *_: None, execute=AsyncMock(), settle=lambda *_: None
            )
            is None
        )
        degraded_at = attempts
        await until(lambda: attempts >= degraded_at + 2)
        warnings = [
            r.getMessage()
            for r in caplog.records
            if "remains deferred" in r.getMessage()
        ]
        assert len(warnings) == 1
        assert str(task_id) in warnings[0]
        monkeypatch.setattr(registry, "_renew", original)
        await until(lambda: owner._healthy)
        handle = owner.submit_execution(
            admit=lambda *_: None, execute=AsyncMock(), settle=lambda *_: None
        )
        assert handle is not None
        await handle
    finally:
        await registry.close()


@pytest.mark.parametrize("sqlstate", ["55P03", "57014", "40P01", "40001"])
async def test_batch_rollback_retries_at_normal_cadence_and_recovers(
    renewal_engine, task_id, monkeypatch, caplog, sqlstate
):
    engine = renewal_engine
    if engine.dialect.name != "postgresql":
        pytest.skip("PostgreSQL transaction errors")
    monkeypatch.setattr(runtime, "get_task_lease_heartbeat_seconds", lambda: 0.2)
    factory = sessionmaker(engine, expire_on_commit=False)
    registry = runtime.TaskCoordinatorRegistry(factory)
    original = registry._renew
    attempts = []
    failed = False
    loop = asyncio.get_running_loop()

    def fail_once(conn, cursor, statement, parameters, context, executemany):
        nonlocal failed
        if (
            not failed
            and statement.startswith("UPDATE tasks SET")
            and "last_heartbeat_at" in statement
        ):
            failed = True
            # The server aborts the real renewal transaction; it is not a
            # simulated successful per-row result.
            return (
                "DO $$ BEGIN RAISE EXCEPTION 'test rollback' USING ERRCODE = '"
                + sqlstate
                + "'; END $$;",
                {},
            )
        return statement, parameters

    def renew(leases):
        attempts.append(loop.time())
        return original(leases)

    with factory() as db, db.begin():
        second_task = Task(user_id=db.get(Task, task_id).user_id, title="sibling")
        db.add(second_task)
        db.flush()
        sibling_id = second_task.id
    try:
        owner = await registry.ensure(task_id)
        sibling = await registry.ensure(sibling_id)
        leases = (owner.lease, sibling.lease)
        before = times(factory, leases)
        monkeypatch.setattr(registry, "_renew", renew)
        sa.event.listen(engine, "before_cursor_execute", fail_once, retval=True)
        try:
            await until(lambda: failed and owner._renewal_waiter is None)
            failure_at = loop.time()
            assert owner.state == runtime.CoordinatorState.ACTIVE
            assert sibling.state == runtime.CoordinatorState.ACTIVE
            assert owner._healthy and sibling._healthy
            assert owner._heartbeat_error is None
            assert sibling._heartbeat_error is None
            assert not owner._recovery_required
            assert not sibling._recovery_required
            assert times(factory, leases) == before
            assert any(
                f"tasks=[{task_id}, {sibling_id}]" in record.getMessage()
                and f"sqlstate={sqlstate}" in record.getMessage()
                for record in caplog.records
            )
            await until(lambda: len(attempts) >= 2 and owner._renewal_waiter is None)
            assert attempts[1] - failure_at >= 0.18
            after = times(factory, leases)
            assert all(after[tid] > before[tid] for tid in before)
            assert owner._heartbeat_error is None
        finally:
            sa.event.remove(engine, "before_cursor_execute", fail_once)
    finally:
        await registry.close()


async def test_subset_failure_waits_full_interval_before_next_batch(
    engine, task_id, monkeypatch
):
    monkeypatch.setattr(runtime, "get_task_lease_heartbeat_seconds", lambda: 0.2)
    factory = sessionmaker(engine, expire_on_commit=False)
    registry = runtime.TaskCoordinatorRegistry(factory)
    original = registry._renew
    calls = []
    loop = asyncio.get_running_loop()

    def renew(leases):
        calls.append(loop.time())
        if len(calls) == 1:
            return {lease: service.TaskLeaseRenewalState.DEFERRED for lease in leases}
        if len(calls) == 2:
            raise sa.exc.TimeoutError("pool exhausted on row retry")
        return original(leases)

    monkeypatch.setattr(registry, "_renew", renew)
    try:
        owner = await registry.ensure(task_id)
        await until(lambda: owner._heartbeat_error is not None)
        failed_at = loop.time()
        assert not owner._healthy
        assert owner._retry_at is None
        await until(lambda: owner._healthy)
        assert calls[2] - failed_at >= 0.18
    finally:
        await registry.close()


@pytest.mark.parametrize("failure_point", ["before_commit", "after_commit"])
async def test_connection_failure_uses_each_owners_acknowledged_margin(
    renewal_engine, task_id, monkeypatch, failure_point
):
    engine = renewal_engine
    monkeypatch.setattr(runtime, "get_task_lease_heartbeat_seconds", lambda: 0.2)
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as db, db.begin():
        sibling = Task(user_id=db.get(Task, task_id).user_id, title="valid margin")
        db.add(sibling)
        db.flush()
        sibling_id = sibling.id
    registry = runtime.TaskCoordinatorRegistry(factory)
    gate = asyncio.Event()
    run_batches = registry._run_heartbeats

    async def gated():
        await gate.wait()
        await run_batches()

    monkeypatch.setattr(registry, "_run_heartbeats", gated)
    failed = False
    attempts = []
    original = registry._renew

    def renew(leases):
        attempts.append(registry.loop.time())
        return original(leases)

    monkeypatch.setattr(registry, "_renew", renew)

    def lost_connection(session):
        nonlocal failed
        if not failed:
            failed = True
            raise sa.exc.OperationalError(
                "COMMIT",
                {},
                engine.dialect.dbapi.OperationalError("connection lost"),
                connection_invalidated=True,
            )

    try:
        expired = await registry.ensure(task_id)
        survivor = await registry.ensure(sibling_id)
        expired._last_renewed_at -= runtime.get_task_lease_ttl_seconds()
        acknowledged_at = survivor._last_renewed_at
        children = [asyncio.create_task(asyncio.Event().wait()) for _ in range(2)]
        expired.track_execution(children[0])
        survivor.track_execution(children[1])
        leases = (expired.lease, survivor.lease)
        before = times(factory, leases)
        sa.event.listen(factory, failure_point, lost_connection)
        try:
            gate.set()
            await until(lambda: failed and survivor._renewal_waiter is None)
            failed_at = registry.loop.time()
            await expired.close()
            assert expired._recovery_required
            assert children[0].cancelled()
            assert survivor.state == runtime.CoordinatorState.ACTIVE
            assert survivor._healthy and survivor._heartbeat_error is None
            assert not children[1].done()
            # Neither real durable outcome grants unacknowledged lease time.
            assert survivor._last_renewed_at == acknowledged_at
            after = times(factory, leases)
            assert (after != before) == (failure_point == "after_commit")
            await until(lambda: survivor._last_renewed_at > acknowledged_at)
            assert attempts[1] - failed_at >= 0.18
            assert not children[1].done()
            with factory() as db:
                assert (
                    db.get(Task, task_id).lease_attempt_id == expired.lease.attempt_id
                )
                assert (
                    db.get(Task, sibling_id).lease_attempt_id
                    == survivor.lease.attempt_id
                )
        finally:
            sa.event.remove(factory, failure_point, lost_connection)
    finally:
        gate.set()
        await registry.close()


async def test_postgresql_connect_error_without_invalidation_retains_margin(
    renewal_engine, task_id, monkeypatch
):
    engine = renewal_engine
    if engine.dialect.name != "postgresql":
        pytest.skip("PostgreSQL driver connection errors")
    monkeypatch.setattr(runtime, "get_task_lease_heartbeat_seconds", lambda: 0.1)
    factory = sessionmaker(engine, expire_on_commit=False)
    registry = runtime.TaskCoordinatorRegistry(factory)
    error = sa.exc.OperationalError(
        None, None, engine.dialect.dbapi.OperationalError("connection unavailable")
    )
    assert not error.connection_invalidated
    attempted = Event()

    def fail_connect(leases):
        attempted.set()
        raise error

    monkeypatch.setattr(registry, "_renew", fail_connect)
    try:
        owner = await registry.ensure(task_id)
        acknowledged_at = owner._last_renewed_at
        await until(lambda: attempted.is_set() and owner._renewal_waiter is None)
        assert owner.state == runtime.CoordinatorState.ACTIVE
        assert owner._healthy and owner._heartbeat_error is None
        assert owner._last_renewed_at == acknowledged_at
    finally:
        await registry.close()


@pytest.mark.parametrize(
    "health", ["healthy", "margin_exhausted", "pool_timeout", "disconnect"]
)
async def test_rollback_before_execution_completion_uses_existing_lease_health(
    engine, task_id, monkeypatch, health
):
    from xagent.web.models import database
    from xagent.web.services import managed_task_lease as managed
    from xagent.web.services.task_lease_service import TaskLease as ExecutionLease

    monkeypatch.setattr(runtime, "get_task_lease_heartbeat_seconds", lambda: 0.1)
    factory = sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(database, "get_session_local", lambda: factory)
    registry = runtime.TaskCoordinatorRegistry(factory)
    failed = Event()

    class RolledBack(Exception):
        pgcode = "55P03"

    error = (
        sa.exc.OperationalError(
            "renew", {}, OSError("connection lost"), connection_invalidated=True
        )
        if health == "disconnect"
        else sa.exc.OperationalError("renew", {}, RolledBack("injected rollback"))
    )

    def renew(leases):
        failed.set()
        raise error

    monkeypatch.setattr(registry, "_renew", renew)
    owner = await registry.ensure(task_id)
    if health == "margin_exhausted":
        owner._last_renewed_at -= runtime.get_task_lease_ttl_seconds()
    previous_error = sa.exc.TimeoutError("earlier pool exhaustion")
    if health == "pool_timeout":
        owner._healthy = False
        owner._heartbeat_error = previous_error
    with factory() as db, db.begin():
        task = db.get(Task, task_id)
        task.status = TaskStatus.RUNNING
        task.run_id = "finishing-run"
    token = runtime._current_coordinator.set(owner)
    handle = managed.start_managed_task_lease(
        ExecutionLease(
            task_id=task_id,
            runner_id=owner.lease.runner_id,
            attempt_id=owner.lease.attempt_id,
            run_id="finishing-run",
        )
    )
    try:
        await until(lambda: failed.is_set() and owner._renewal_waiter is None)
        assert owner.state == runtime.CoordinatorState.ACTIVE
        if health in ("healthy", "disconnect"):
            assert owner._healthy
            assert owner._heartbeat_error is None
        elif health == "pool_timeout":
            assert owner._heartbeat_error is previous_error
        else:
            assert not owner._healthy
            assert owner._heartbeat_error is error
        # No renewal can succeed in this test. Completion must use the last
        # acknowledged lease and the real fenced finalizer, not await a retry.
        finalized = await handle.finalize_result(status=TaskStatus.COMPLETED)
        assert finalized == (health in ("healthy", "disconnect"))
        with factory() as db:
            task = db.get(Task, task_id)
            assert task.status == (
                TaskStatus.COMPLETED if finalized else TaskStatus.RUNNING
            )
            assert task.lease_attempt_id == owner.lease.attempt_id
    finally:
        await handle.close()
        runtime._current_coordinator.reset(token)
        await registry.close()
