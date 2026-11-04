"""Real hosts/workers reuse public workflows with V2 content isolation enabled."""

import pytest

from tests.e2e import test_shared_execution as workflows

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.parametrize("shared_app", [2], indirect=True),
]


def test_v2_create_continue_and_poll(shared_app):
    workflows.test_sdk_create_append_and_poll_execute_only_in_worker(shared_app)
    from xagent.web.models.database import get_session_local
    from xagent.web.models.task import Task
    from xagent.web.models.task_execution_event import TaskExecutionEvent

    with get_session_local()() as db:
        assert {task.conversation_storage_version for task in db.query(Task)} == {2}
        assert db.query(TaskExecutionEvent).count() > 0


def test_v2_live_and_reconnect(shared_app):
    workflows.test_websocket_execute_stream_and_reconnect(shared_app)


@pytest.mark.parametrize("interface", ["sdk", "a2a"])
def test_v2_answer_after_worker_replacement(shared_app, interface):
    workflows.test_waiting_reply_restores_checkpoint_in_new_worker(
        shared_app, interface
    )


def test_v2_cancel_running_task(shared_app):
    workflows.test_a2a_cancel_reaches_running_worker(shared_app)


@pytest.mark.parametrize("interface", ["owner", "sdk"])
def test_v2_workforce_and_child(shared_app, interface):
    workflows.test_workforce_manager_and_child_execute_in_worker(shared_app, interface)


def test_v2_pause_and_resume(shared_app):
    workflows.test_websocket_pause_and_resume_with_two_workers(shared_app)


def test_v2_files_and_output_download(shared_app):
    workflows.test_sdk_file_input_tool_output_and_download(shared_app)


def test_v2_running_message_delivery(shared_app):
    workflows.test_websocket_running_message_is_delivered_once(shared_app)


def test_v2_worker_loss(shared_app):
    workflows.test_host_loss_settles_without_duplicate_execution(
        shared_app, "worker_kill"
    )


def test_v2_a2a_reconnect(shared_app):
    workflows.test_a2a_stream_disconnect_and_resubscribe(shared_app)
