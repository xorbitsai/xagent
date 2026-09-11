import asyncio

import pytest

from xagent.web.api.websocket import BackgroundTaskManager
from xagent.web.services.task_lease_service import run_while_task_lease_owned


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
