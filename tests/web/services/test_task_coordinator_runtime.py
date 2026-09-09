"""Coordinator lifecycle contracts using real SQLite/PostgreSQL transactions."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from threading import Event
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.exc import TimeoutError as PoolTimeout

from tests.web.services.test_task_coordinator_service import (
    database as database_fixture,
)
from tests.web.services.test_task_execution_event_store import engine as engine_fixture
from tests.web.services.test_task_execution_event_store import (
    task_id as task_id_fixture,
)
from xagent.web.models.task import Task, TaskStatus
from xagent.web.services import task_coordinator_runtime as runtime
from xagent.web.services import task_coordinator_service as ownership
from xagent.web.services.task_execution_controller import task_control_snapshot

database = database_fixture
engine = engine_fixture
task_id = task_id_fixture


@pytest.fixture
async def registry(database, monkeypatch):
    monkeypatch.setattr(runtime, "get_task_lease_heartbeat_seconds", lambda: 0.02)
    result = runtime.TaskCoordinatorRegistry(database[0])
    try:
        yield result
    finally:
        await result.close()
    assert result._coordinators == {}


def admit(db, lease):
    return ownership.begin_task_execution_no_commit(
        db,
        lease,
        expected=task_control_snapshot(db.get(Task, lease.task_id)),
        new_run=True,
    )


def settle(db, context, error):
    task = db.get(Task, context.lease.task_id)
    task.status = TaskStatus.PAUSED if error else TaskStatus.COMPLETED
    task.control_state = "paused" if error else "completed"
    task.state_version += 1


async def wait_thread_event(event):
    assert await asyncio.to_thread(event.wait, 5)


async def test_repeated_wake_shares_acquisition_and_heartbeat(registry, database):
    coordinators = await asyncio.gather(
        *(registry.ensure(database[1]) for _ in range(10))
    )
    coordinator = coordinators[0]
    assert all(c is coordinator for c in coordinators)
    heartbeat = coordinator._heartbeat_task
    token = coordinator.lease.attempt_id
    coordinator.wakeup.clear()
    assert await registry.ensure(database[1]) is coordinator
    assert coordinator.wakeup.is_set()
    assert coordinator._heartbeat_task is heartbeat
    assert coordinator.lease.attempt_id == token


async def test_remote_owner_is_not_adopted(registry, database):
    other = runtime.TaskCoordinatorRegistry(database[0])
    try:
        results = await asyncio.gather(
            registry.ensure(database[1]), other.ensure(database[1])
        )
        assert sum(c is not None for c in results) == 1
        assert sum(len(r._coordinators) for r in (registry, other)) == 1
    finally:
        await other.close()


@pytest.mark.parametrize("action", ["retry", "cancel_waiter", "shutdown"])
async def test_wake_during_release_waits_for_cleanup(
    registry, database, monkeypatch, action
):
    owner = await registry.ensure(database[1])
    committed, unblock = Event(), Event()
    original = owner._release

    def blocked_release():
        result = original()
        committed.set()
        assert unblock.wait(5)
        return result

    monkeypatch.setattr(owner, "_release", blocked_release)
    closer = asyncio.create_task(owner.close())
    waiters = []
    shutdown = None
    try:
        await wait_thread_event(committed)
        with database[0]() as db:
            assert db.get(Task, database[1]).runner_id is None
        waiters = [asyncio.create_task(registry.ensure(database[1])) for _ in range(2)]
        await asyncio.sleep(0)
        assert all(not waiter.done() for waiter in waiters)
        assert registry._coordinators[database[1]] is owner
        if action == "cancel_waiter":
            waiters[0].cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiters[0]
            assert not owner._close_task.cancelled()
        elif action == "shutdown":
            shutdown = asyncio.create_task(registry.close())
            await asyncio.sleep(0)
    finally:
        unblock.set()
    await closer
    if shutdown is not None:
        await shutdown
        assert await asyncio.gather(*waiters) == [None, None]
        assert not registry._coordinators
    else:
        successor = await waiters[-1]
        assert successor is not None
        assert successor is not owner
        assert successor.lease.attempt_id != owner.lease.attempt_id
        if action == "retry":
            assert await waiters[0] is successor


async def test_cancelled_acquisition_drains_commit_and_cleans_slot(
    registry, database, monkeypatch
):
    committed, unblock = Event(), Event()
    original = runtime.TaskCoordinator._acquire

    def blocked(coordinator):
        lease = original(coordinator)
        committed.set()
        assert unblock.wait(5)
        return lease

    monkeypatch.setattr(runtime.TaskCoordinator, "_acquire", blocked)
    waiter = asyncio.create_task(registry.ensure(database[1]))
    try:
        await wait_thread_event(committed)
        waiter.cancel()
        await asyncio.sleep(0)
        waiter.cancel()
    finally:
        unblock.set()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert not registry._coordinators
    with database[0]() as db:
        assert db.get(Task, database[1]).runner_id is None
    assert await registry.ensure(database[1]) is not None


async def test_one_cancelled_waiter_does_not_cancel_shared_acquisition(
    registry, database, monkeypatch
):
    entered, unblock = Event(), Event()
    original = runtime.TaskCoordinator._acquire

    def blocked(coordinator):
        entered.set()
        assert unblock.wait(5)
        return original(coordinator)

    monkeypatch.setattr(runtime.TaskCoordinator, "_acquire", blocked)
    first = asyncio.create_task(registry.ensure(database[1]))
    second = asyncio.create_task(registry.ensure(database[1]))
    try:
        await wait_thread_event(entered)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
    finally:
        unblock.set()
    coordinator = await second
    assert coordinator.state == runtime.CoordinatorState.ACTIVE
    assert await registry.ensure(database[1]) is coordinator


async def test_startup_failure_releases_acquired_owner(registry, database, monkeypatch):
    original = runtime.TaskCoordinator._start

    async def broken(coordinator):
        coordinator.lease = await asyncio.to_thread(coordinator._acquire)
        raise RuntimeError("startup failed")

    monkeypatch.setattr(runtime.TaskCoordinator, "_start", broken)
    with pytest.raises(RuntimeError, match="startup failed"):
        await registry.ensure(database[1])
    assert not registry._coordinators
    with database[0]() as db:
        assert db.get(Task, database[1]).runner_id is None
    monkeypatch.setattr(runtime.TaskCoordinator, "_start", original)
    assert await registry.ensure(database[1]) is not None


async def test_close_during_acquisition_cannot_start_heartbeat(
    registry, database, monkeypatch
):
    entered, unblock = Event(), Event()
    original = runtime.TaskCoordinator._acquire

    def blocked(coordinator):
        entered.set()
        assert unblock.wait(5)
        return original(coordinator)

    monkeypatch.setattr(runtime.TaskCoordinator, "_acquire", blocked)
    waiter = asyncio.create_task(registry.ensure(database[1]))
    try:
        await wait_thread_event(entered)
        coordinator = registry._coordinators[database[1]]
        closer = asyncio.create_task(registry.close())
        # Let close mark the coordinator quiescing before the worker returns.
        async with asyncio.timeout(5):
            while coordinator.state == runtime.CoordinatorState.ACQUIRING:
                await asyncio.sleep(0)
    finally:
        unblock.set()
    assert await waiter is None
    await closer
    assert coordinator._heartbeat_task is None
    assert await registry.ensure(database[1]) is None


async def test_two_entry_admissions_share_one_handle_and_one_lease(registry, database):
    coordinator = await registry.ensure(database[1])
    started, finish = asyncio.Event(), asyncio.Event()
    contexts = []

    async def execute(context):
        contexts.append(context)
        started.set()
        await finish.wait()

    first = coordinator.submit_execution(admit=admit, execute=execute, settle=settle)
    assert first is not None
    assert (
        coordinator.submit_execution(admit=admit, execute=execute, settle=settle)
        is None
    )
    await asyncio.wait_for(started.wait(), 5)
    # A long Runner does not prevent another entry from waking the coordinator.
    coordinator.wakeup.clear()
    assert await registry.ensure(database[1]) is coordinator
    assert coordinator.wakeup.is_set()
    heartbeat = coordinator._heartbeat_task
    finish.set()
    await first
    assert coordinator._heartbeat_task is heartbeat
    with database[0]() as db:
        task = db.get(Task, database[1])
        assert task.status == TaskStatus.COMPLETED
        assert task.lease_attempt_id == coordinator.lease.attempt_id
    second = coordinator.submit_execution(admit=admit, execute=execute, settle=settle)
    await second
    assert contexts[0].lease is contexts[1].lease
    assert contexts[0].run_id != contexts[1].run_id


async def test_admission_rejection_rolls_back_business_writes(registry, database):
    coordinator = await registry.ensure(database[1])
    execute = AsyncMock()

    def reject(db, lease):
        db.get(Task, lease.task_id).output = "must roll back"
        return None

    await coordinator.submit_execution(admit=reject, execute=execute, settle=settle)
    execute.assert_not_awaited()
    with database[0]() as db:
        assert db.get(Task, database[1]).output is None
        assert db.get(Task, database[1]).status == TaskStatus.PENDING


async def test_cancel_during_admission_settles_late_committed_run(registry, database):
    coordinator = await registry.ensure(database[1])
    entered, unblock = Event(), Event()
    execute = AsyncMock()

    def blocked(db, lease):
        context = admit(db, lease)
        entered.set()
        assert unblock.wait(5)
        return context

    handle = coordinator.submit_execution(admit=blocked, execute=execute, settle=settle)
    try:
        await wait_thread_event(entered)
        closer = asyncio.create_task(coordinator.close())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    finally:
        unblock.set()
    await closer
    assert handle.cancelled()
    execute.assert_not_awaited()
    with database[0]() as db:
        task = db.get(Task, database[1])
        assert task.status == TaskStatus.PAUSED
        assert task.runner_id is None


async def test_close_keeps_heartbeat_until_execution_cleanup_settles(
    registry, database, monkeypatch
):
    coordinator = await registry.ensure(database[1])
    started, draining, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
    renewed = Event()
    original = coordinator._renew

    def renew():
        result = original()
        renewed.set()
        return result

    monkeypatch.setattr(coordinator, "_renew", renew)

    async def execute(context):
        started.set()
        try:
            await asyncio.Future()
        finally:
            draining.set()
            await finish.wait()

    coordinator.submit_execution(admit=admit, execute=execute, settle=settle)
    await asyncio.wait_for(started.wait(), 5)
    closer = asyncio.create_task(coordinator.close())
    try:
        await asyncio.wait_for(draining.wait(), 5)
        renewed.clear()
        closer.cancel()
        await asyncio.sleep(0)
        closer.cancel()
        await wait_thread_event(renewed)
        assert (
            coordinator.submit_execution(admit=admit, execute=execute, settle=settle)
            is None
        )
        with database[0]() as db:
            assert (
                db.get(Task, database[1]).lease_attempt_id
                == coordinator.lease.attempt_id
            )
    finally:
        finish.set()
    with pytest.raises(asyncio.CancelledError):
        await closer
    await coordinator.close()
    assert coordinator._heartbeat_task.done()
    assert coordinator._execution_task is None
    with database[0]() as db:
        assert db.get(Task, database[1]).runner_id is None


async def test_lost_owner_drains_runner_and_cannot_settle_successor(registry, database):
    coordinator = await registry.ensure(database[1])
    started, drained = asyncio.Event(), asyncio.Event()

    async def execute(context):
        started.set()
        try:
            await asyncio.Future()
        finally:
            drained.set()

    handle = coordinator.submit_execution(admit=admit, execute=execute, settle=settle)
    await asyncio.wait_for(started.wait(), 5)
    with database[0]() as db, db.begin():
        task = db.get(Task, database[1])
        task.lease_attempt_id = "successor"
        task.output = "successor output"
    await asyncio.wait_for(drained.wait(), 5)
    await coordinator.close()
    assert handle.cancelled()
    with database[0]() as db:
        task = db.get(Task, database[1])
        assert task.lease_attempt_id == "successor"
        assert task.output == "successor output"
        assert task.status == TaskStatus.RUNNING


async def test_pool_timeout_does_not_mean_lost_or_busy_retry(
    registry, database, monkeypatch
):
    coordinator = await registry.ensure(database[1])
    failed = Event()
    calls = 0

    def timeout():
        nonlocal calls
        calls += 1
        failed.set()
        raise PoolTimeout("pool exhausted")

    monkeypatch.setattr(runtime, "get_task_lease_heartbeat_seconds", lambda: 0.1)
    monkeypatch.setattr(coordinator, "_renew", timeout)
    await wait_thread_event(failed)
    # The heartbeat handles the thread result on its next event-loop turn.
    async with asyncio.timeout(5):
        while coordinator._healthy:
            await asyncio.sleep(0)
    assert coordinator.state == runtime.CoordinatorState.ACTIVE
    assert (
        coordinator.submit_execution(admit=admit, execute=AsyncMock(), settle=settle)
        is None
    )
    await asyncio.sleep(0.02)
    assert calls == 1
    await coordinator.close()
    with database[0]() as db:
        assert (
            db.get(Task, database[1]).lease_attempt_id == coordinator.lease.attempt_id
        )


async def test_heartbeat_crash_closes_and_drains_execution(
    registry, database, monkeypatch
):
    coordinator = await registry.ensure(database[1])
    started, drained = asyncio.Event(), asyncio.Event()

    async def execute(context):
        started.set()
        try:
            await asyncio.Future()
        finally:
            drained.set()

    coordinator.submit_execution(admit=admit, execute=execute, settle=settle)
    await asyncio.wait_for(started.wait(), 5)

    def crash():
        raise RuntimeError("connection failed")

    monkeypatch.setattr(coordinator, "_renew", crash)
    await asyncio.wait_for(drained.wait(), 5)
    await coordinator.close()
    assert coordinator._heartbeat_task.done()
    with database[0]() as db:
        assert (
            db.get(Task, database[1]).lease_attempt_id == coordinator.lease.attempt_id
        )


@pytest.mark.parametrize("repeat_cancel", [False, True])
async def test_shutdown_releases_successful_inflight_settlement(
    registry, database, repeat_cancel
):
    coordinator = await registry.ensure(database[1])
    entered, unblock = Event(), Event()

    def blocked_settlement(db, context, error):
        assert error is None
        entered.set()
        assert unblock.wait(5)
        settle(db, context, error)

    handle = coordinator.submit_execution(
        admit=admit, execute=AsyncMock(), settle=blocked_settlement
    )
    try:
        await wait_thread_event(entered)
        closer = asyncio.create_task(coordinator.close())
        async with asyncio.timeout(5):
            while not handle.cancelling():
                await asyncio.sleep(0)
        if repeat_cancel:
            handle.cancel()
            await asyncio.sleep(0)
        assert not closer.done()
    finally:
        unblock.set()
    await closer
    assert handle.cancelled()
    assert not coordinator._recovery_required
    with database[0]() as db:
        task = db.get(Task, database[1])
        assert task.status == TaskStatus.COMPLETED
        assert task.runner_id is None
        assert task.lease_attempt_id is None
        assert task.lease_expires_at is None
    successor = await registry.ensure(database[1])
    assert successor is not None
    assert successor.lease.attempt_id != coordinator.lease.attempt_id


@pytest.mark.parametrize("stage", ["admission", "settlement"])
@pytest.mark.parametrize("cancelled", [False, True])
async def test_unknown_commit_stops_heartbeat_and_retains_recovery_evidence(
    registry, database, monkeypatch, stage, cancelled
):
    coordinator = await registry.ensure(database[1])
    execute = AsyncMock()
    method = "_admit" if stage == "admission" else "_settle"
    original = getattr(coordinator, method)
    committed, unblock = Event(), Event()

    def uncertain(*args):
        original(*args)
        if cancelled:
            committed.set()
            assert unblock.wait(5)
        raise RuntimeError("commit acknowledgement lost")

    monkeypatch.setattr(coordinator, method, uncertain)
    handle = coordinator.submit_execution(admit=admit, execute=execute, settle=settle)
    if cancelled:
        try:
            await wait_thread_event(committed)
            handle.cancel()
        finally:
            unblock.set()
        with pytest.raises(asyncio.CancelledError):
            await handle
    else:
        with pytest.raises(RuntimeError, match="commit acknowledgement lost"):
            await handle
    await coordinator.close()
    assert coordinator._heartbeat_task.done()
    if stage == "admission":
        execute.assert_not_awaited()
    with database[0]() as db:
        task = db.get(Task, database[1])
        assert task.lease_attempt_id == coordinator.lease.attempt_id
        assert task.status == (
            TaskStatus.RUNNING if stage == "admission" else TaskStatus.COMPLETED
        )


async def test_renewal_recovers_after_pool_timeout(registry, database, monkeypatch):
    coordinator = await registry.ensure(database[1])
    original = coordinator._renew
    recovered = Event()
    calls = 0

    def renew():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise PoolTimeout("transient exhaustion")
        result = original()
        recovered.set()
        return result

    monkeypatch.setattr(coordinator, "_renew", renew)
    await wait_thread_event(recovered)
    async with asyncio.timeout(5):
        while not coordinator._healthy:
            await asyncio.sleep(0)
    assert coordinator.state == runtime.CoordinatorState.ACTIVE
    handle = coordinator.submit_execution(
        admit=admit, execute=AsyncMock(), settle=settle
    )
    assert handle is not None
    await handle
    await coordinator.close()
    with database[0]() as db:
        assert db.get(Task, database[1]).runner_id is None


async def test_acquisition_commit_ack_failure_leaves_token_for_expiry(
    registry, database, monkeypatch
):
    original = runtime.TaskCoordinator._acquire

    def uncertain(coordinator):
        original(coordinator)
        raise RuntimeError("acquisition commit acknowledgement lost")

    monkeypatch.setattr(runtime.TaskCoordinator, "_acquire", uncertain)
    with pytest.raises(RuntimeError, match="acquisition commit acknowledgement lost"):
        await registry.ensure(database[1])
    assert not registry._coordinators
    with database[0]() as db:
        task = db.get(Task, database[1])
        assert task.lease_attempt_id is not None
        assert task.status == TaskStatus.PENDING


@pytest.mark.parametrize("status", [TaskStatus.PAUSED, TaskStatus.RUNNING])
async def test_recovery_only_reclaims_expired_nonrunning_tasks(
    registry, database, status
):
    with database[0]() as db, db.begin():
        old = ownership.acquire_task_lease_no_commit(db, database[1], runner_id="dead")
        task = db.get(Task, database[1])
        task.status = status
        task.control_state = status.value
        task.run_id = "old-run"
        task.output = "retained result"
        task.lease_expires_at = ownership.utc_now() - timedelta(seconds=1)
    coordinator = await registry.ensure(database[1])
    assert (coordinator is not None) is (status != TaskStatus.RUNNING)
    with database[0]() as db:
        task = db.get(Task, database[1])
        assert task.status == status
        assert task.run_id == "old-run"
        assert task.output == "retained result"
        if coordinator:
            assert coordinator.lease.attempt_id != old.attempt_id
            assert not ownership.renew_task_lease_no_commit(db, old)
        else:
            assert task.lease_attempt_id == old.attempt_id
