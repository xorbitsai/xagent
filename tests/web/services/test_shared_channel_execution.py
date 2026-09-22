"""Channel acceptance is atomic and never acquires an ingress lease."""

# Pytest fixture imports are intentionally shadowed by test parameters.
# ruff: noqa: F811

from types import SimpleNamespace

import pytest

from tests.web.services.channel_delivery_shared import database_url as database_url
from tests.web.services.channel_delivery_shared import selected as selected
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import get_session_local
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user_channel import UserChannel
from xagent.web.services import shared_channel_execution as shared
from xagent.web.services.channel_runtime import (
    ChannelAuthorizationError,
)
from xagent.web.services.task_orchestrator import TaskTurnError, TaskTurnPayload


def test_acceptance_persists_start_and_single_transcript_without_lease(selected):
    command_id = shared._accept_channel_turn(
        selected, TaskTurnPayload("hello"), "ingress"
    )
    with get_session_local()() as db:
        task = db.get(Task, selected.selection.task_id)
        command = db.get(TaskExecutionCommand, command_id)
        messages = db.query(TaskChatMessage).filter_by(task_id=task.id).all()
        assert task.status == TaskStatus.PENDING
        assert task.run_id is None
        assert command.target_run_id == selected.run_id
        assert task.runner_id is None
        assert task.lease_attempt_id is None
        assert command.status == "pending"
        assert command.reply_host_id == "ingress"
        assert command.reply_origin == "origin-token"
        assert command.payload["channel"] == {
            "channel_id": selected.selection.channel_id,
            "external_user_id": "sender",
        }
        assert len(messages) == 1
        assert command.payload["before_message_id"] == messages[0].id
    with pytest.raises(TaskTurnError):
        shared._accept_channel_turn(selected, TaskTurnPayload("second"), "ingress")
    shared._settle_pending_selection(selected.selection)
    with get_session_local()() as db:
        assert db.get(Task, selected.selection.task_id) is not None


@pytest.mark.parametrize("changed_payload", [False, True])
def test_uncertain_commit_requires_matching_payload(
    selected, monkeypatch, changed_payload
):
    from sqlalchemy.orm import Session

    commit = Session.commit
    failure = ConnectionError("commit acknowledgement lost")

    def uncertain_commit(db):
        commit(db)
        if changed_payload:
            with get_session_local()() as other:
                command = other.query(TaskExecutionCommand).one()
                command.payload = {**command.payload, "message": "different message"}
                commit(other)
        raise failure

    monkeypatch.setattr(Session, "commit", uncertain_commit)
    if changed_payload:
        with pytest.raises(ConnectionError) as error:
            shared._accept_channel_turn(selected, TaskTurnPayload("hello"), "ingress")
        assert error.value is failure
    else:
        command_id = shared._accept_channel_turn(
            selected, TaskTurnPayload("hello"), "ingress"
        )
    with get_session_local()() as db:
        command = db.query(TaskExecutionCommand).one()
        assert command.command_id == selected.command_id
        assert command.payload["message"] == (
            "different message" if changed_payload else "hello"
        )
        if not changed_payload:
            assert command.id == command_id
        assert db.query(TaskChatMessage).count() == 1


def test_start_failure_rolls_back_message_and_run(selected, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("START failed")

    monkeypatch.setattr(shared, "stage_task_start_command", fail)
    with pytest.raises(RuntimeError, match="START failed"):
        shared._accept_channel_turn(selected, TaskTurnPayload("hello"), "ingress")
    with get_session_local()() as db:
        task = db.get(Task, selected.selection.task_id)
        assert task.status == TaskStatus.PENDING
        assert task.run_id == selected.selection.previous_run_id
        assert db.query(TaskChatMessage).count() == 0
        assert db.query(TaskExecutionCommand).count() == 0


def test_acceptance_revalidates_channel_sender(selected):
    with get_session_local()() as db:
        db.get(UserChannel, selected.selection.channel_id).config = {
            "allowed_users": ["another-sender"]
        }
        db.commit()
    with pytest.raises(ChannelAuthorizationError):
        shared._accept_channel_turn(selected, TaskTurnPayload("hello"), "ingress")
    with get_session_local()() as db:
        assert db.get(Task, selected.selection.task_id).status == TaskStatus.PENDING
        assert db.query(TaskExecutionCommand).count() == 0


@pytest.mark.asyncio
async def test_stop_cannot_target_replacement_run(selected):
    shared._accept_channel_turn(selected, TaskTurnPayload("hello"), "ingress")
    with get_session_local()() as db:
        db.get(Task, selected.selection.task_id).run_id = "replacement-run"
        db.query(TaskExecutionCommand).filter_by(
            task_id=selected.selection.task_id
        ).update({"status": "failed"})
        db.commit()
    await selected._enqueue_stop()
    with get_session_local()() as db:
        assert db.query(TaskExecutionCommand).count() == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["completed", "waiting_for_user"])
async def test_worker_handoff_and_channel_result_commit_atomically(
    selected, monkeypatch, status
):
    import asyncio
    from unittest.mock import AsyncMock, Mock

    from tests.web.services.coordinator_command_shared import claim_task_command
    from xagent.web.services import (
        agent_service_manager,
        task_command_execution,
        task_event_bridge,
        task_execution,
        task_orchestrator,
    )
    from xagent.web.services.task_execution_context_service import (
        TaskExecutionRecoverySnapshot,
    )

    monkeypatch.setenv("XAGENT_TASK_EXECUTION_ROLE", "worker")
    from xagent.web.services import task_coordinator_runtime

    monkeypatch.setattr(task_coordinator_runtime, "get_runner_id", lambda: "worker-1")
    command_id = shared._accept_channel_turn(
        selected, TaskTurnPayload("hello"), "ingress"
    )
    with get_session_local()() as db:
        command = await claim_task_command(
            db, runner_id="worker-1", command_db_id=command_id
        )
    tracer = SimpleNamespace(add_handler=Mock(), remove_handler=Mock())
    service = SimpleNamespace(
        tracer=tracer,
        workspace=None,
        set_conversation_history=Mock(),
        set_execution_context_messages=Mock(),
        set_recovered_skill_context=Mock(),
    )
    manager = SimpleNamespace(
        get_agent_for_task=AsyncMock(return_value=service),
        execute_task=AsyncMock(
            return_value={"success": True, "status": status, "output": "Worker answer"}
        ),
    )
    monkeypatch.setattr(agent_service_manager, "get_agent_manager", lambda: manager)
    bridge = Mock()
    sending = asyncio.Event()
    release = asyncio.Event()
    delivered = []

    async def send_progress(message):
        sending.set()
        await release.wait()
        assert shared._read_channel_result(command.id, selected.run_id) is None
        delivered.append(message["trace"])

    bridge.reply_for.return_value = send_progress

    async def execute_with_progress(**_):
        forwarder = tracer.add_handler.call_args.args[0]
        await forwarder.handle_event(Mock(to_dict=lambda: {"event": "A"}))
        await sending.wait()
        await forwarder.handle_event(Mock(to_dict=lambda: {"event": "B"}))
        await forwarder.handle_event(Mock(to_dict=lambda: {"event": "C"}))
        release.set()
        return {"success": True, "status": status, "output": "Worker answer"}

    manager.execute_task.side_effect = execute_with_progress
    monkeypatch.setattr(task_event_bridge, "_bridge", bridge)
    monkeypatch.setattr(shared, "get_task_event_bridge", lambda: bridge)
    snapshot = SimpleNamespace(
        runtime_user=object(),
        task=SimpleNamespace(user_id=selected.selection.user_id),
        conversation_history=(),
        conversation_watermark=None,
        execution_recovery=TaskExecutionRecoverySnapshot(),
    )
    monkeypatch.setattr(
        task_orchestrator, "load_task_setup_snapshot_sync", lambda *a, **kw: snapshot
    )
    monkeypatch.setattr(task_orchestrator, "resolve_execution_scope", lambda *a: None)
    monkeypatch.setattr(
        task_execution,
        "background_task_manager",
        task_execution.BackgroundTaskManager(),
    )
    try:
        await task_command_execution.execute_durable_task_command(command)
        async with asyncio.timeout(10):
            while shared._read_channel_result(command.id, selected.run_id) is None:
                await asyncio.sleep(0.01)
    finally:
        await task_coordinator_runtime.close_task_coordinators()
        await task_execution.background_task_manager.shutdown()
    assert (
        manager.execute_task.await_args.kwargs["task_lease"].run_id == selected.run_id
    )
    with get_session_local()() as db:
        task = db.get(Task, command.task_id)
        row = db.get(TaskExecutionCommand, command.id)
        assert task.status == TaskStatus(status)
        assert task.lease_attempt_id is None
        assert row.result["channel_result"]["status"] == status
        messages = (
            db.query(TaskChatMessage).filter_by(task_id=task.id, role="assistant").all()
        )
        assert len(messages) == 1
        assert messages[0].content == "Worker answer"
        assert messages[0].turn_id == command.command_id
        assert task.output == ("Worker answer" if status == "completed" else None)
    assert (
        shared._read_channel_result(command.id, selected.run_id)["output"]
        == "Worker answer"
    )
    tracer.remove_handler.assert_called_once_with(tracer.add_handler.call_args.args[0])
    assert delivered == [{"event": "A"}, {"event": "B"}, {"event": "C"}]
    bridge.discard_command.assert_called_with(command.command_id, command.task_id)


def test_completion_waits_for_paused_execution_lease_release(selected):
    from xagent.web.services.task_completion import TaskRunChanged, _is_run_finished

    command_id = shared._accept_channel_turn(
        selected, TaskTurnPayload("hello"), "ingress"
    )
    with get_session_local()() as db:
        task = db.get(Task, selected.selection.task_id)
        task.status = TaskStatus.PAUSED
        task.run_id = selected.run_id
        db.get(TaskExecutionCommand, command_id).status = "completed"
        task.runner_id = "worker"
        db.commit()
    assert not _is_run_finished(selected.selection.task_id, selected.run_id)
    assert shared._read_channel_result(command_id, selected.run_id) is None
    with get_session_local()() as db:
        db.get(Task, selected.selection.task_id).runner_id = None
        db.commit()
    assert _is_run_finished(selected.selection.task_id, selected.run_id)
    assert (
        shared._read_channel_result(command_id, selected.run_id)["status"]
        == "interrupted"
    )
    with pytest.raises(TaskRunChanged):
        _is_run_finished(selected.selection.task_id, "other-run")


@pytest.mark.asyncio
async def test_stop_queued_channel_turn_targets_planned_run(selected):
    shared._accept_channel_turn(selected, TaskTurnPayload("hello"), "ingress")
    await selected._enqueue_stop()
    with get_session_local()() as db:
        task = db.get(Task, selected.selection.task_id)
        pause = (
            db.query(TaskExecutionCommand)
            .filter_by(task_id=task.id, kind="pause")
            .one()
        )
        assert task.run_id is None
        assert task.status == TaskStatus.PENDING
        assert pause.target_run_id == selected.run_id


def test_wait_for_queued_channel_and_failed_acceptance(selected):
    command_id = shared._accept_channel_turn(
        selected, TaskTurnPayload("hello"), "ingress"
    )
    assert shared._read_channel_result(command_id, selected.run_id) is None
    with get_session_local()() as db:
        db.get(TaskExecutionCommand, command_id).status = "failed"
        db.commit()
    assert shared._read_channel_result(command_id, selected.run_id)["success"] is False


def test_worker_revalidates_channel_after_acceptance(selected):
    from tests.web.services.coordinator_command_shared import claim_for_owner
    from xagent.web.services import task_start_consumer
    from xagent.web.services.task_command_transport import SettledTaskCommand
    from xagent.web.services.task_coordinator_service import (
        acquire_task_lease_no_commit,
    )

    command_id = shared._accept_channel_turn(
        selected, TaskTurnPayload("hello"), "ingress"
    )
    with get_session_local()() as db:
        db.get(UserChannel, selected.selection.channel_id).is_active = False
        db.commit()
        lease = acquire_task_lease_no_commit(
            db, task_id=selected.selection.task_id, runner_id="worker-1"
        )
        db.commit()
        command = claim_for_owner(db, lease, command_id)
    result = task_start_consumer._commit_handoff(command, lease)
    assert isinstance(result, SettledTaskCommand)
    with get_session_local()() as db:
        assert db.get(Task, selected.selection.task_id).run_id is None
        assert db.get(TaskExecutionCommand, command_id).status == "failed"


def test_missing_file_rolls_back_channel_start_and_transcript(selected):
    with pytest.raises(TaskTurnError):
        shared._accept_channel_turn(
            selected, TaskTurnPayload("hello", file_ids=("missing",)), "ingress"
        )
    with get_session_local()() as db:
        assert db.query(TaskChatMessage).count() == 0
        assert db.query(TaskExecutionCommand).count() == 0
        assert db.get(Task, selected.selection.task_id).run_id is None


@pytest.mark.asyncio
async def test_cancellation_during_acceptance_keeps_task_and_queues_stop(
    selected, monkeypatch
):
    import asyncio
    import threading
    from unittest.mock import Mock

    accepted = threading.Event()
    release = threading.Event()
    accept = shared._accept_channel_turn

    def delayed(*args):
        result = accept(*args)
        accepted.set()
        assert release.wait(10)
        return result

    bridge = Mock(host_id="ingress")
    bridge.register_origin.return_value = "origin"
    monkeypatch.setattr(shared, "get_task_event_bridge", lambda: bridge)
    monkeypatch.setattr(shared, "_accept_channel_turn", delayed)
    execution = asyncio.create_task(selected.execute(TaskTurnPayload("hello"), None))
    try:
        assert await asyncio.to_thread(accepted.wait, 10)
        execution.cancel()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await execution
    await selected.close()
    assert selected.accepted
    with get_session_local()() as db:
        assert db.get(Task, selected.selection.task_id) is not None
        commands = (
            db.query(TaskExecutionCommand).order_by(TaskExecutionCommand.id).all()
        )
        assert [row.kind for row in commands] == ["start", "pause"]
        assert all(row.target_run_id == selected.run_id for row in commands)
