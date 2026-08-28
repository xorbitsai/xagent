from __future__ import annotations

import asyncio

import pytest

from xagent.web.api.websocket import BackgroundTaskManager


@pytest.mark.asyncio
async def test_shutdown_cancels_snapshot_once_and_drains_before_clearing() -> None:
    manager = BackgroundTaskManager()
    started = [asyncio.Event(), asyncio.Event()]
    cancellation_seen = [asyncio.Event(), asyncio.Event()]
    allow_cleanup = asyncio.Event()
    cleanup_attempted = [asyncio.Event(), asyncio.Event()]
    allow_settlement = asyncio.Event()

    async def execution(index: int) -> None:
        started[index].set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_seen[index].set()
            await allow_cleanup.wait()
            manager.cleanup_task(index + 1)
            cleanup_attempted[index].set()
            await allow_settlement.wait()
            raise

    first = asyncio.create_task(execution(0))
    second = asyncio.create_task(execution(1))
    manager.register_task(1, first)
    assert manager.reserve_resume(1)
    manager.register_reserved_resume(1, first, run_id=None)
    manager.register_task(2, second)
    assert manager.reserve_resume(99)
    await asyncio.gather(*(event.wait() for event in started))

    shutdown = asyncio.create_task(manager.shutdown())
    await asyncio.gather(*(event.wait() for event in cancellation_seen))

    # The promoted task appears in both maps but must receive only one cancel.
    assert first.cancelling() == 1
    assert second.cancelling() == 1
    # Registrations remain visible until every task's cleanup has settled.
    assert manager.running_tasks == {1: first, 2: second}
    assert manager.resume_tasks == {1: first}
    assert manager._resume_reservations == {99}
    assert not shutdown.done()

    # Cancelling application shutdown must not abandon task cleanup.
    shutdown.cancel()
    await asyncio.sleep(0)
    assert not shutdown.done()

    allow_cleanup.set()
    await asyncio.gather(*(event.wait() for event in cleanup_attempted))
    # A task's own finally block must not erase shutdown's ownership snapshot.
    assert manager.running_tasks == {1: first, 2: second}
    assert manager.resume_tasks == {1: first}

    repeated_shutdown = asyncio.create_task(manager.shutdown())
    await asyncio.sleep(0)
    assert first.cancelling() == 1
    assert second.cancelling() == 1

    allow_settlement.set()
    with pytest.raises(asyncio.CancelledError):
        await shutdown
    await repeated_shutdown

    assert first.cancelled()
    assert second.cancelled()
    assert manager.running_tasks == {}
    assert manager.resume_tasks == {}
    assert manager._resume_reservations == set()


@pytest.mark.asyncio
async def test_shutdown_admission_fence_rejects_late_task_registration() -> None:
    manager = BackgroundTaskManager()
    started = asyncio.Event()
    cancellation_seen = asyncio.Event()
    allow_cleanup = asyncio.Event()

    async def existing_execution() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await allow_cleanup.wait()
            raise

    existing = asyncio.create_task(existing_execution())
    manager.register_task(1, existing)
    assert manager.reserve_resume(2)
    await started.wait()

    shutdown = asyncio.create_task(manager.shutdown())
    await cancellation_seen.wait()

    assert not manager.reserve_resume(3)

    late_execution = asyncio.create_task(asyncio.sleep(60))
    with pytest.raises(RuntimeError, match="shutting down"):
        manager.register_task(3, late_execution)

    late_resume = asyncio.create_task(asyncio.sleep(60))
    with pytest.raises(RuntimeError, match="shutting down"):
        manager.register_reserved_resume(2, late_resume, run_id=None)

    await asyncio.gather(late_execution, late_resume, return_exceptions=True)
    assert late_execution.cancelled()
    assert late_resume.cancelled()
    assert 3 not in manager.running_tasks
    assert 2 not in manager.resume_tasks

    allow_cleanup.set()
    await shutdown


@pytest.mark.asyncio
async def test_shutdown_fence_prevents_cancelled_resume_from_promoting() -> None:
    manager = BackgroundTaskManager()
    cancellation_seen = asyncio.Event()
    promotion_attempted = asyncio.Event()
    promotion_outcomes: list[str] = []

    assert manager.reserve_resume(7)

    async def resume_coordinator() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            current = asyncio.current_task()
            assert current is not None
            try:
                manager.promote_resume_task(7, current)
            except RuntimeError:
                promotion_outcomes.append("rejected")
            else:
                promotion_outcomes.append("accepted")
            promotion_attempted.set()
            raise

    resume = asyncio.create_task(resume_coordinator())
    manager.register_reserved_resume(7, resume, run_id=None)

    shutdown = asyncio.create_task(manager.shutdown())
    await cancellation_seen.wait()
    await promotion_attempted.wait()
    await shutdown

    assert promotion_outcomes == ["rejected"]
    assert resume.cancelled()
    assert manager.running_tasks == {}
    assert manager.resume_tasks == {}


@pytest.mark.asyncio
async def test_start_accepting_reopens_only_an_idle_manager() -> None:
    manager = BackgroundTaskManager()
    active = asyncio.create_task(asyncio.sleep(60))
    manager.register_task(1, active)

    try:
        with pytest.raises(RuntimeError, match="still owns background work"):
            manager.start_accepting()
    except BaseException:
        active.cancel()
        await asyncio.gather(active, return_exceptions=True)
        raise

    await manager.shutdown()
    manager.start_accepting()

    assert manager.reserve_resume(2)
    replacement = asyncio.create_task(asyncio.sleep(60))
    manager.register_task(2, replacement)
    await manager.shutdown()
    assert replacement.cancelled()


@pytest.mark.asyncio
async def test_cleanup_child_can_release_pending_owner_registration() -> None:
    manager = BackgroundTaskManager()
    allow_owner_to_finish = asyncio.Event()

    async def owner() -> None:
        await allow_owner_to_finish.wait()

    owner_task = asyncio.create_task(owner())
    manager.register_task(7, owner_task)
    assert manager.reserve_resume(7)
    manager.register_reserved_resume(7, owner_task, run_id=None)

    async def cleanup_child() -> None:
        manager.cleanup_task(7, expected_task=owner_task)

    try:
        await asyncio.create_task(cleanup_child())

        assert not owner_task.done()
        assert manager.running_tasks == {}
        assert manager.resume_tasks == {}
    finally:
        allow_owner_to_finish.set()
        await owner_task
    manager.start_accepting()


@pytest.mark.asyncio
async def test_owner_cleanup_does_not_remove_replacement_registration() -> None:
    manager = BackgroundTaskManager()
    completed_owner = asyncio.create_task(asyncio.sleep(0))
    await completed_owner
    replacement = asyncio.create_task(asyncio.sleep(60))
    manager.register_task(7, replacement)

    manager.cleanup_task(7, expected_task=completed_owner)

    assert manager.running_tasks == {7: replacement}
    await manager.shutdown()


def test_concurrent_shutdown_lock_is_recreated_for_each_event_loop() -> None:
    manager = BackgroundTaskManager()

    async def run_lifespan(task_id: int) -> None:
        manager.start_accepting()
        execution = asyncio.create_task(asyncio.sleep(60))
        manager.register_task(task_id, execution)

        await asyncio.gather(manager.shutdown(), manager.shutdown())

        assert execution.cancelled()
        assert manager.running_tasks == {}

    asyncio.run(run_lifespan(1))
    asyncio.run(run_lifespan(2))
