import asyncio

import pytest

from xagent.web.api.websocket import (
    BackgroundTaskManager,
    _clear_task_pause_accepted,
    _is_task_pause_accepted,
    _mark_task_pause_accepted,
    _task_status_uses_live_control,
)
from xagent.web.models.task import TaskStatus


def test_paused_task_user_message_is_not_live_control() -> None:
    assert not _task_status_uses_live_control(TaskStatus.PAUSED)


def test_active_task_user_messages_stay_live_control() -> None:
    assert _task_status_uses_live_control(TaskStatus.RUNNING)
    assert _task_status_uses_live_control(TaskStatus.WAITING_FOR_USER)


def test_requested_control_states_route_messages_consistently() -> None:
    assert _task_status_uses_live_control(
        TaskStatus.PAUSED,
        control_state="resume_requested",
    )
    assert not _task_status_uses_live_control(
        TaskStatus.RUNNING,
        control_state="pause_requested",
    )


def test_accepted_pause_routes_active_task_out_of_live_control() -> None:
    assert not _task_status_uses_live_control(
        TaskStatus.RUNNING,
        pause_accepted=True,
    )
    assert not _task_status_uses_live_control(
        TaskStatus.WAITING_FOR_USER,
        pause_accepted=True,
    )


def test_terminal_and_pending_statuses_are_not_live_control() -> None:
    assert not _task_status_uses_live_control(TaskStatus.PENDING)
    assert not _task_status_uses_live_control(TaskStatus.COMPLETED)
    assert not _task_status_uses_live_control(TaskStatus.FAILED)


def test_pause_accepted_marker_can_be_cleared() -> None:
    task_id = 12345
    _clear_task_pause_accepted(task_id)

    _mark_task_pause_accepted(task_id)
    assert _is_task_pause_accepted(task_id)

    _clear_task_pause_accepted(task_id)
    assert not _is_task_pause_accepted(task_id)


@pytest.mark.asyncio
async def test_resume_coordinator_does_not_replace_task_that_it_waits_for() -> None:
    manager = BackgroundTaskManager()
    allow_original_to_check = asyncio.Event()

    async def original_runner() -> None:
        await allow_original_to_check.wait()
        await manager.wait_for_previous(7)

    original = asyncio.create_task(original_runner())
    manager.register_task(7, original)
    assert manager.reserve_resume(7)

    async def resume_runner() -> None:
        await original
        current = asyncio.current_task()
        assert current is not None
        manager.promote_resume_task(7, current)

    resume = asyncio.create_task(resume_runner())
    manager.register_reserved_resume(7, resume)

    # The original remains the active execution until it finishes, so its
    # wait_for_previous call sees itself instead of the resume coordinator.
    assert manager.running_tasks[7] is original
    assert manager.resume_tasks[7] is resume
    assert not manager.reserve_resume(7)

    allow_original_to_check.set()
    await asyncio.wait_for(asyncio.gather(original, resume), timeout=1)
    assert manager.running_tasks[7] is resume


@pytest.mark.asyncio
async def test_unregistered_resume_coordinator_cannot_promote_itself() -> None:
    manager = BackgroundTaskManager()
    current = asyncio.current_task()
    assert current is not None

    with pytest.raises(RuntimeError, match="not registered"):
        manager.promote_resume_task(7, current)

    assert 7 not in manager.running_tasks
