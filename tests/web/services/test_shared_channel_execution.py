"""Channel acceptance is atomic and never acquires an ingress lease."""

import os
from types import SimpleNamespace

import pytest
from sqlalchemy import text
from sqlalchemy.engine import make_url

from tests.shared.postgres_disposable import disposable_database_factory
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import Base, get_engine, get_session_local, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.models.user_channel import UserChannel
from xagent.web.services import shared_channel_execution as shared
from xagent.web.services.channel_runtime import (
    ChannelAuthorizationError,
    _prepare_channel_task_sync,
)
from xagent.web.services.task_orchestrator import TaskTurnError, TaskTurnPayload


@pytest.fixture(
    params=["sqlite", pytest.param("postgresql", marks=pytest.mark.postgresql)]
)
def database_url(request, tmp_path):
    if request.param == "postgresql":
        with disposable_database_factory("shared_channel") as make:
            engine = make("worker")
            with engine.connect() as connection:
                name = connection.execute(
                    text("SELECT current_database()")
                ).scalar_one()
            yield (
                make_url(os.environ["XAGENT_TEST_POSTGRES_URL"])
                .set(database=name)
                .render_as_string(hide_password=False)
            )
    else:
        yield f"sqlite:///{tmp_path / 'channel.db'}"


@pytest.fixture
def selected(database_url, monkeypatch):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    init_db(db_url=database_url)
    with get_session_local()() as db:
        user = User(username="owner", password_hash="unused")
        db.add(user)
        db.flush()
        channel = UserChannel(
            user_id=user.id,
            channel_type="feishu",
            channel_name="test",
            config={"allowed_users": ["sender"]},
            is_active=True,
        )
        db.add(channel)
        db.commit()
        channel_id = channel.id
    selection = _prepare_channel_task_sync(
        channel_id=channel_id,
        external_user_id="sender",
        active_task_id=None,
        text="hello",
        channel_name="test",
        expected_owner_user_id=None,
        defer_execution=True,
    )
    turn = shared.SharedChannelTurn(selection, workspace=SimpleNamespace())
    turn.origin = "origin-token"
    yield turn
    Base.metadata.drop_all(bind=get_engine())
    get_engine().dispose()


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

    from xagent.web.services import (
        agent_service_manager,
        task_command_execution,
        task_event_bridge,
        task_execution,
        task_orchestrator,
    )
    from xagent.web.services.task_command_transport import claim_task_command
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
        command = claim_task_command(db, runner_id="worker-1", command_db_id=command_id)
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
    from xagent.web.services import task_start_consumer
    from xagent.web.services.task_command_transport import (
        SettledTaskCommand,
        claim_task_command,
    )
    from xagent.web.services.task_coordinator_service import (
        acquire_task_lease_no_commit,
    )

    command_id = shared._accept_channel_turn(
        selected, TaskTurnPayload("hello"), "ingress"
    )
    with get_session_local()() as db:
        db.get(UserChannel, selected.selection.channel_id).is_active = False
        db.commit()
        command = claim_task_command(db, runner_id="worker-1", command_db_id=command_id)
        lease = acquire_task_lease_no_commit(
            db, task_id=selected.selection.task_id, runner_id="worker-1"
        )
        db.commit()
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
