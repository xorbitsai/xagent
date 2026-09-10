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
    manager.resume_tasks[42] = replacement
    try:
        await old
        await asyncio.sleep(0)
        assert manager.running_tasks[42] is replacement
        assert manager.resume_tasks[42] is replacement
    finally:
        replacement.cancel()
        await asyncio.gather(replacement, return_exceptions=True)
        await asyncio.sleep(0)
    assert manager.running_tasks == {}
    assert manager.resume_tasks == {}


@pytest.mark.asyncio
async def test_cancellation_before_owner_starts_removes_registration():
    manager = BackgroundTaskManager()
    task = asyncio.create_task(asyncio.Event().wait())
    manager.register_task(42, task)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)
    assert manager.running_tasks == {}
