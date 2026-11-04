import asyncio

import pytest

from xagent.web.services.task_dispatch_cap import DispatchPolicy
from xagent.web.services.task_execution import BackgroundTaskManager
from xagent.web.services.task_lease_service import run_while_task_lease_owned

FIVE_SLOTS = DispatchPolicy(limit=5, interactive_reserve=0)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
async def test_registered_owner_is_removed_after_lease_child_finishes(outcome):
    manager = BackgroundTaskManager()
    started = asyncio.Event()
    child_cleanup_kept_owner = []

    async def operation():
        started.set()
        try:
            if outcome == "failure":
                raise ValueError("execution failed")
            if outcome == "cancel":
                await asyncio.Event().wait()
        finally:
            manager.cleanup_task(42)
            child_cleanup_kept_owner.append(42 in manager.running_tasks)

    async def owner():
        heartbeat = asyncio.create_task(asyncio.Event().wait())
        try:
            await run_while_task_lease_owned(operation(), heartbeat)
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)

    task = asyncio.create_task(owner())
    manager.register_task(42, task)
    await started.wait()
    if outcome == "cancel":
        task.cancel()
    result = (await asyncio.gather(task, return_exceptions=True))[0]
    await asyncio.sleep(0)
    assert child_cleanup_kept_owner == [True]
    assert manager.running_tasks == {}
    if outcome == "failure":
        assert isinstance(result, ValueError)
    elif outcome == "cancel":
        assert isinstance(result, asyncio.CancelledError)
    else:
        assert result is None


@pytest.mark.asyncio
async def test_old_completion_does_not_remove_replacement():
    manager = BackgroundTaskManager()
    old = asyncio.create_task(asyncio.sleep(0))
    replacement = asyncio.create_task(asyncio.Event().wait())
    manager.register_task(42, old)
    manager.register_task(42, replacement)
    assert manager.reserve_resume(42)
    manager.register_reserved_resume(42, replacement, run_id="replacement-run")
    try:
        await old
        await asyncio.sleep(0)
        assert manager.running_tasks[42] is replacement
        assert manager.resume_tasks[42] is replacement
        assert manager._resume_run_ids[42] == "replacement-run"
        assert 42 in manager._resume_owner_started_at
    finally:
        replacement.cancel()
        await asyncio.gather(replacement, return_exceptions=True)
        await asyncio.sleep(0)
    assert manager.running_tasks == {}
    assert manager.resume_tasks == {}
    assert manager._resume_run_ids == {}
    assert manager._resume_owner_started_at == {}


@pytest.mark.asyncio
async def test_cancellation_before_owner_starts_removes_registration():
    manager = BackgroundTaskManager()
    task = asyncio.create_task(asyncio.Event().wait())
    manager.register_task(42, task)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)
    assert manager.running_tasks == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("promote", [False, True])
async def test_resume_cancelled_before_start_cleans_all_registrations(promote):
    manager = BackgroundTaskManager()
    entered = []

    async def resume():
        entered.append(True)
        await asyncio.Event().wait()

    assert manager.reserve_resume(42)
    task = asyncio.create_task(resume())
    manager.register_reserved_resume(42, task, run_id="run-a")
    if promote:
        manager.promote_resume_task(42, task)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)
    assert not entered
    assert manager.running_tasks == {}
    assert manager.resume_tasks == {}
    assert manager._resume_run_ids == {}
    assert manager._resume_owner_started_at == {}
    assert manager._resume_reservations == set()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_resume_completion_cleans_metadata(fail):
    manager = BackgroundTaskManager()

    async def resume():
        await asyncio.sleep(0)
        if fail:
            raise ValueError("resume failed")

    assert manager.reserve_resume(42)
    task = asyncio.create_task(resume())
    manager.register_reserved_resume(42, task, run_id="run-a")
    result = (await asyncio.gather(task, return_exceptions=True))[0]
    await asyncio.sleep(0)
    assert isinstance(result, ValueError) if fail else result is None
    assert manager.resume_tasks == {}
    assert manager._resume_run_ids == {}
    assert manager._resume_owner_started_at == {}


@pytest.mark.asyncio
async def test_delayed_shutdown_callbacks_cannot_remove_new_lifespan_owner(monkeypatch):
    manager = BackgroundTaskManager()
    deferred = []
    cleanup = manager.cleanup_task

    def defer_cleanup(task_id, *, expected_task=None):
        deferred.append((task_id, expected_task, manager._shutting_down))

    monkeypatch.setattr(manager, "cleanup_task", defer_cleanup)
    old = asyncio.create_task(asyncio.Event().wait())
    manager.register_task(42, old)
    assert manager.reserve_resume(42)
    manager.register_reserved_resume(42, old, run_id="old-run")
    await manager.shutdown()
    assert len(deferred) == 2
    assert all(during_shutdown for _, _, during_shutdown in deferred)
    manager.start_accepting()
    monkeypatch.setattr(manager, "cleanup_task", cleanup)
    replacement = asyncio.create_task(asyncio.Event().wait())
    manager.register_task(42, replacement)
    assert manager.reserve_resume(42)
    manager.register_reserved_resume(42, replacement, run_id="new-run")
    try:
        # Replay both actual completion callbacks after admission has reopened.
        for task_id, expected_task, _ in deferred:
            cleanup(task_id, expected_task=expected_task)
        assert manager.running_tasks[42] is replacement
        assert manager.resume_tasks[42] is replacement
        assert manager._resume_run_ids[42] == "new-run"
        assert 42 in manager._resume_owner_started_at
    finally:
        await manager.shutdown()
    assert manager.running_tasks == {}
    assert manager.resume_tasks == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("live", "resume", "tokens", "closed", "active"),
    [
        ({1, 2}, {3}, set(), True, {1, 2, 3}),
        ({1, 2}, set(), {3}, True, {1, 2, 3}),
        ({1, 2, 3}, set(), set(), True, {1, 2, 3}),
        ({1, 2}, {2}, set(), False, {1, 2}),
        ({1, 2}, set(), {2}, False, {1, 2}),
        (set(), set(), set(), False, set()),
    ],
    ids=[
        "resume-entry",
        "token",
        "live-only",
        "resume-overlap",
        "token-overlap",
        "empty",
    ],
)
async def test_cap_snapshot_counts_union_of_task_ids(
    live, resume, tokens, closed, active
):
    manager = BackgroundTaskManager()
    handle = object()
    for task_id in live:
        manager._live_executions[task_id] = {handle}
    for task_id in resume:
        manager.resume_tasks[task_id] = handle
    for task_id in tokens:
        manager._dispatch_reservations[task_id] += 1

    snapshot = manager.dispatch_cap_snapshot(
        DispatchPolicy(limit=3, interactive_reserve=0)
    )

    assert snapshot.limit == 3
    assert snapshot.active_ids == frozenset(active)
    assert snapshot.closed is closed
    for task_id in active:
        assert snapshot.admits(task_id, interactive=False)
    assert snapshot.admits(99, interactive=False) is (not closed)


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["token", "live_handle", "shutdown"])
async def test_dispatch_reservations_follow_manager_lifespan(state):
    manager = BackgroundTaskManager()
    handle = asyncio.create_task(asyncio.Event().wait())
    try:
        if state == "token":
            token = manager.reserve_dispatch_slot(
                1, policy=FIVE_SLOTS, interactive=False
            )
            with pytest.raises(RuntimeError, match="still owns background work"):
                manager.start_accepting()
            manager.release_dispatch_slot(token)
            manager.start_accepting()
        elif state == "live_handle":
            handle.cancel()
            await asyncio.gather(handle, return_exceptions=True)

            async def unwinding() -> None:
                # The execution's own finally has run: it has left every
                # registry table but is still unwinding.
                manager.cleanup_task(2)
                await asyncio.Event().wait()

            handle = asyncio.create_task(unwinding())
            manager.register_task(2, handle)
            await asyncio.sleep(0)
            assert manager.running_tasks == {}
            with pytest.raises(RuntimeError, match="still owns background work"):
                manager.start_accepting()
        else:
            manager.register_task(2, handle)
            token = manager.reserve_dispatch_slot(
                1, policy=FIVE_SLOTS, interactive=False
            )
            await manager.shutdown()
            assert manager._live_executions == {}
            assert not manager._dispatch_reservations
            manager.release_dispatch_slot(token)
            manager._drop_live_execution(2, handle)
            assert manager._live_executions == {}
            assert not manager._dispatch_reservations
            manager.start_accepting()
            assert manager._shutting_down is False
    finally:
        handle.cancel()
        await asyncio.gather(handle, return_exceptions=True)
