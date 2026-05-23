from xagent.web.api.websocket import _task_status_uses_live_control
from xagent.web.models.task import TaskStatus


def test_paused_task_user_message_is_not_live_control() -> None:
    assert not _task_status_uses_live_control(TaskStatus.PAUSED)


def test_active_task_user_messages_stay_live_control() -> None:
    assert _task_status_uses_live_control(TaskStatus.RUNNING)
    assert _task_status_uses_live_control(TaskStatus.WAITING_FOR_USER)


def test_terminal_and_pending_statuses_are_not_live_control() -> None:
    assert not _task_status_uses_live_control(TaskStatus.PENDING)
    assert not _task_status_uses_live_control(TaskStatus.COMPLETED)
    assert not _task_status_uses_live_control(TaskStatus.FAILED)
