from xagent.web.api.websocket import (
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
