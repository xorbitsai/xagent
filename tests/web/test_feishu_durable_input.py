"""Feishu batches use real receipt/command transactions and mocked platform IO."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy.orm import sessionmaker

from tests.web.services.task_database_shared import engine as engine_fixture
from xagent.web.channels.feishu import bot as module
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_channel_delivery import TaskChannelDelivery
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.task_input_receipt import TaskInputReceipt
from xagent.web.models.user import User
from xagent.web.models.user_channel import UserChannel
from xagent.web.services import (
    channel_delivery,
)
from xagent.web.services import channel_input_acceptance as inputs
from xagent.web.services import (
    channel_runtime,
    shared_channel_execution,
    task_event_bridge,
    uploaded_file_store,
)

engine = engine_fixture


def message(text="hello", identity="one", chat="chat", kind="text"):
    return SimpleNamespace(
        event=SimpleNamespace(
            sender=SimpleNamespace(sender_id=SimpleNamespace(open_id="sender")),
            message=SimpleNamespace(
                message_type=kind,
                chat_id=chat,
                message_id=identity,
                content=json.dumps(
                    {"text": text} if kind == "text" else {"file_key": text}
                ),
                create_time="1",
            ),
        )
    )


@pytest.fixture
def ingress(engine, monkeypatch, tmp_path):
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine)
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    for service in (
        inputs,
        channel_delivery,
        shared_channel_execution,
        uploaded_file_store,
        channel_runtime,
    ):
        monkeypatch.setattr(service, "get_session_local", lambda: sessions)
    monkeypatch.setattr(task_event_bridge, "_bridge", Mock(host_id="ingress"))

    async def complete(handler):
        result = {"success": True, "status": "completed", "output": "answer"}
        with sessions() as db:
            task = db.get(Task, handler.task_id)
            task.status = TaskStatus.COMPLETED
            for command in db.query(TaskExecutionCommand).filter_by(
                task_id=task.id, kind="start"
            ):
                command.status = "completed"
                command.result = {"channel_result": result}
            db.commit()
        return result

    observe = AsyncMock(side_effect=complete)
    monkeypatch.setattr(shared_channel_execution.SharedChannelTurn, "observe", observe)
    monkeypatch.setattr(
        module,
        "get_agent_manager",
        Mock(side_effect=AssertionError("No ingress Agent")),
    )
    monkeypatch.setattr(
        module,
        "persist_channel_user_message",
        AsyncMock(side_effect=AssertionError("Acceptance owns transcript")),
    )
    with sessions() as db:
        owner = User(username="owner", password_hash="unused")
        db.add(owner)
        db.flush()
        channel = UserChannel(
            user_id=owner.id,
            channel_type="feishu",
            channel_name="test",
            config={"allowed_users": ["sender"]},
            is_active=True,
        )
        db.add(channel)
        db.commit()
        cid = int(channel.id)

    def make_bot():
        bot = object.__new__(module.FeishuBotInstance)
        bot._initialize_batch_control()
        bot.user_active_trace_handlers = {}
        bot.control_tasks = set()
        bot.control_queues = {}
        bot.control_locks = {}
        bot._accepting = True
        bot.start_time = 100
        bot.queue_flush_delay_seconds = 0
        bot.channel_id = cid
        bot.channel_name = "test"
        bot.active_tasks_file = tmp_path / "active.json"
        bot.active_tasks = bot._load_active_tasks()
        bot._stop_lock = None
        bot._stop_loop = None
        bot._ingress_stopped = False
        bot.ws_client = None
        bot._ping_task = None
        bot.api_client = Mock()
        bot._send_text = AsyncMock(return_value="loading")
        bot._update_text = AsyncMock()
        return bot

    yield make_bot, sessions, observe
    Base.metadata.drop_all(engine)


async def run(bot, *messages):
    for item in messages:
        bot._handle_message_sync(item)
    if bot.control_tasks:
        await asyncio.gather(*tuple(bot.control_tasks))
    if bot.user_message_tasks:
        await asyncio.gather(*tuple(bot.user_message_tasks.values()))


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", [False, True], ids=["null", "missing"])
async def test_empty_content_does_not_abort_batch(ingress, missing):
    make, sessions, observe = ingress
    bot = make()
    empty = message(identity="empty")
    if missing:
        del empty.event.message.content
    else:
        empty.event.message.content = None
    await run(bot, empty, message("valid text", "valid"))
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == 2
        command = db.query(TaskExecutionCommand).filter_by(kind="start").one()
        assert command.payload["message"] == "Received a text message.\nvalid text"
    observe.assert_awaited_once()


@pytest.mark.asyncio
async def test_batch_retries_after_restart_reuse_one_command_and_loading(ingress):
    make, sessions, observe = ingress
    first = make()
    await run(first, message("A", "a"), message("B", "b"))
    second = make()
    await run(second, message("B", "b"), message("A", "a"))
    with sessions() as db:
        assert db.query(Task).count() == 1
        assert db.query(TaskExecutionCommand).count() == 1
        assert db.query(TaskInputReceipt).count() == 2
        assert (
            db.query(TaskChannelDelivery).one().destination["loading_message_id"]
            == "loading"
        )
    assert first._send_text.await_count == 1
    second._send_text.assert_not_awaited()
    assert observe.await_count == 2


@pytest.mark.asyncio
async def test_overlapping_batch_accepts_only_new_inputs(ingress):
    make, sessions, _ = ingress
    bot = make()
    await run(bot, message("A", "a"), message("B", "b"))
    await run(bot, message("B", "b"), message("C", "c"))
    with sessions() as db:
        commands = (
            db.query(TaskExecutionCommand).order_by(TaskExecutionCommand.id).all()
        )
        assert len(commands) == 2
        assert db.query(TaskInputReceipt).count() == 3
        assert db.query(Task).count() == 1
        assert commands[0].payload["message"] == "A\nB"
        assert commands[1].payload["message"] == "C"


@pytest.mark.asyncio
async def test_old_conflict_does_not_block_new_input(ingress):
    make, sessions, _ = ingress
    bot = make()
    await run(bot, message("A", "a"))
    await run(bot, message("changed", "a"), message("B", "b"))
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == 2
        assert db.query(TaskExecutionCommand).count() == 2
    assert any(
        "different content" in call.args[1] for call in bot._send_text.await_args_list
    )


@pytest.mark.asyncio
async def test_save_failure_keeps_accepted_task_and_retry_identity(ingress):
    make, sessions, _ = ingress
    bot = make()
    bot._save_active_tasks = Mock(return_value=False)
    await run(bot, message())
    selected = bot.active_tasks["sender"]
    assert any(
        "request was accepted" in call.args[1]
        for call in bot._send_text.await_args_list
    )
    await run(make(), message())
    with sessions() as db:
        assert db.query(Task).count() == 1
        assert db.query(TaskExecutionCommand).count() == 1
        assert str(db.query(TaskInputReceipt).one().task_id) == selected


@pytest.mark.asyncio
async def test_distinct_chats_do_not_mix_receipt_scopes(ingress):
    make, sessions, _ = ingress
    bot = make()
    await run(bot, message("A", "same", "chat-a"), message("B", "same", "chat-b"))
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == 2
        assert db.query(Task).count() == 1
        assert {
            row.destination["chat_id"] for row in db.query(TaskChannelDelivery)
        } == {"chat-a", "chat-b"}


@pytest.mark.asyncio
async def test_attachment_failure_has_no_acceptance(ingress):
    make, sessions, _ = ingress
    bot = make()
    bot._download_feishu_file_sync = Mock(return_value=None)
    await run(bot, message("missing", kind="file"))
    with sessions() as db:
        assert db.query(Task).count() == 0
        assert db.query(TaskInputReceipt).count() == 0
        assert db.query(TaskExecutionCommand).count() == 0
    assert "were not accepted" in bot._send_text.await_args.args[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/new", "/stop"])
async def test_control_during_lookup_prevents_acceptance(ingress, monkeypatch, command):
    make, sessions, _ = ingress
    bot = make()
    entered, release = asyncio.Event(), asyncio.Event()
    original = module.run_db_io_cancellation_safe

    async def lookup(operation):
        entered.set()
        await release.wait()
        return await original(operation)

    monkeypatch.setattr(module, "run_db_io_cancellation_safe", lookup)
    task = asyncio.create_task(bot._process_messages_batch("sender", [message()]))
    await asyncio.wait_for(entered.wait(), 5)
    await bot._handle_control("sender", message(command), command)
    release.set()
    await asyncio.wait_for(task, 5)
    with sessions() as db:
        assert db.query(TaskExecutionCommand).count() == 0
    assert bot.active_tasks == ({"sender": "-1"} if command == "/new" else {})


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/new", "/stop"])
async def test_control_during_commit_stops_late_accepted_turn(
    ingress, monkeypatch, command
):
    import threading

    make, sessions, _ = ingress
    bot = make()
    entered, release = threading.Event(), threading.Event()
    original = module.accept_channel_input

    def accept(*args, **kwargs):
        result = original(*args, **kwargs)
        entered.set()
        assert release.wait(10)
        return result

    monkeypatch.setattr(module, "accept_channel_input", accept)
    task = asyncio.create_task(bot._process_messages_batch("sender", [message()]))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        await bot._handle_control("sender", message(command), command)
    finally:
        release.set()
    await asyncio.wait_for(task, 5)
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == 1
        assert {row.kind for row in db.query(TaskExecutionCommand)} == {
            "start",
            "pause",
        }
        assert db.query(TaskChannelDelivery).one().status == (
            "discarded" if command == "/new" else "pending"
        )
    assert bot.active_tasks["sender"] == ("-1" if command == "/new" else "1")


@pytest.mark.asyncio
async def test_batch_changed_repartitions(ingress, monkeypatch):
    make, sessions, _ = ingress
    bot = make()
    original = module.accept_channel_input
    calls = []

    def accept(incoming, **kwargs):
        calls.append(incoming.message_id)
        if len(calls) == 1:
            original(incoming, **(kwargs | {"additional_inputs": ()}))
            raise inputs.ChannelInputBatchChanged()
        return original(incoming, **kwargs)

    monkeypatch.setattr(module, "accept_channel_input", accept)
    await run(bot, message("A", "a"), message("B", "b"))
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == 2
        assert db.query(TaskExecutionCommand).count() == 2
    assert calls == ["a", "b"]


@pytest.mark.asyncio
async def test_attachments_are_bound_atomically_and_not_downloaded_on_retry(
    ingress, monkeypatch, tmp_path
):
    from xagent.web.models.chat_message import TaskChatMessage
    from xagent.web.models.uploaded_file import UploadedFile
    from xagent.web.services.channel_runtime import DownloadedChannelFile
    from xagent.web.services.uploaded_file_store import StagedUploadedFile

    make, sessions, _ = ingress
    bot = make()
    source = tmp_path / "input.txt"
    source.write_text("contents")

    def download(*args):
        with sessions() as db:
            assert db.query(Task).count() == 0
        return DownloadedChannelFile("input.txt", source, "text/plain", 8, "F1")

    bot._download_feishu_file_sync = Mock(side_effect=download)

    def stage(**kwargs):
        return StagedUploadedFile(
            kwargs["file_id"],
            kwargs["user_id"],
            None,
            "input.txt",
            str(source),
            "local",
            kwargs["storage_key"],
            None,
            "checksum",
            None,
            None,
            None,
            "text/plain",
            8,
            "feishu",
        )

    monkeypatch.setattr(module, "stage_uploaded_file_from_local_path", stage)
    await run(bot, message("F1", kind="file"), message("explain", "two"))
    await run(bot, message("F1", kind="file"))
    bot._download_feishu_file_sync.assert_called_once()
    with sessions() as db:
        uploaded = db.query(UploadedFile).one()
        command = db.query(TaskExecutionCommand).one()
        assert command.payload["file_ids"] == [uploaded.file_id]
        assert uploaded.task_id == command.task_id
        assert (
            db.query(TaskChatMessage)
            .filter_by(role="user")
            .one()
            .attachments[0]["type"]
            == "text/plain"
        )


@pytest.mark.asyncio
async def test_revoked_sender_cannot_replay(ingress):
    make, sessions, observe = ingress
    bot = make()
    await run(bot, message())
    with sessions() as db:
        channel = db.get(UserChannel, bot.channel_id)
        channel.config = {"allowed_users": ["someone-else"]}
        db.commit()
    await run(bot, message())
    assert observe.await_count == 1
    assert "not authorized" in bot._send_text.await_args.args[1]


@pytest.mark.asyncio
async def test_new_does_not_restore_or_send_old_receipt(ingress):
    make, sessions, observe = ingress
    bot = make()
    await run(bot, message())
    new = message("/new", "command")
    new.event.message.create_time = "101"
    await run(bot, new)
    await run(bot, message())
    assert bot.active_tasks["sender"] == "-1"
    assert observe.await_count == 1
    with sessions() as db:
        assert db.query(TaskExecutionCommand).filter_by(kind="start").count() == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("command", [None, "/stop", "/new"])
async def test_cancel_during_commit_respects_explicit_controls(
    ingress, monkeypatch, command
):
    import threading

    make, sessions, _ = ingress
    bot = make()
    entered, release = threading.Event(), threading.Event()
    original = module.accept_channel_input

    def accept(*args, **kwargs):
        result = original(*args, **kwargs)
        entered.set()
        assert release.wait(10)
        return result

    monkeypatch.setattr(module, "accept_channel_input", accept)
    task = asyncio.create_task(bot._process_messages_batch("sender", [message()]))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        if command is not None:
            await bot._handle_control("sender", message(command), command)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == 1
        assert {row.kind for row in db.query(TaskExecutionCommand)} == (
            {"start", "pause"} if command is not None else {"start"}
        )
        assert db.query(TaskChannelDelivery).one().status == (
            "discarded" if command == "/new" else "pending"
        )
    assert bot.active_tasks["sender"] == ("-1" if command == "/new" else "1")
    assert not bot.user_preparing_executions
    assert not bot.user_active_executions


@pytest.mark.asyncio
async def test_progress_uses_persisted_loading_message(ingress):
    make, sessions, observe = ingress
    bot = make()
    complete = observe.side_effect

    async def progress(handler):
        await handler._update_message("partial answer")
        with sessions() as db:
            assert (
                db.query(TaskChannelDelivery).one().destination["loading_message_id"]
                == "loading"
            )
        return await complete(handler)

    observe.side_effect = progress
    await run(bot, message())
    assert any(
        call.args == ("chat", "loading", "partial answer ✍️")
        for call in bot._update_text.await_args_list
    )
    bot.api_client.im.v1.message.patch.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/stop", "/new"])
async def test_historical_replay_keeps_newer_pending_turn_controllable(
    ingress, monkeypatch, command
):
    make, sessions, observe = ingress
    bot = make()
    await run(bot, message("A", "a"))
    observe.side_effect = None
    observe.return_value = {"status": "accepted"}
    await run(bot, message("B", "b"))
    newer = bot.user_active_executions["sender"]
    entered, release = asyncio.Event(), asyncio.Event()

    async def deliver(*args, **kwargs):
        entered.set()
        await release.wait()
        return False

    monkeypatch.setattr(shared_channel_execution.SharedChannelTurn, "deliver", deliver)
    replay = asyncio.create_task(run(bot, message("A", "a")))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert bot.user_active_executions["sender"] == newer
        await bot._handle_control("sender", message(command), command)
        await newer[1].stop_task
    finally:
        release.set()
    await asyncio.wait_for(replay, 5)
    assert bot.user_active_executions["sender"] == newer
    assert observe.await_count == 2
    with sessions() as db:
        pause = db.query(TaskExecutionCommand).filter_by(kind="pause").one()
        assert pause.target_run_id == newer[1].run_id


@pytest.mark.asyncio
async def test_cancel_during_save_warning_preserves_owned_accepted_turn(ingress):
    make, sessions, observe = ingress
    bot = make()
    bot._save_active_tasks = Mock(return_value=False)
    entered = asyncio.Event()

    async def send(*args):
        assert "request was accepted" in args[1]
        assert bot.user_active_executions["sender"]
        entered.set()
        await asyncio.Future()

    bot._send_text = send
    task = asyncio.create_task(bot._process_messages_batch("sender", [message()]))
    await asyncio.wait_for(entered.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == 1
        assert {row.kind for row in db.query(TaskExecutionCommand)} == {
            "start",
        }
    observe.assert_not_awaited()
    assert bot.user_active_executions["sender"][1].accepted
    assert not bot.user_preparing_executions


@pytest.mark.asyncio
async def test_observation_failure_keeps_accepted_turn_controllable(ingress):
    from sqlalchemy.exc import OperationalError

    make, sessions, observe = ingress
    bot = make()
    observe.side_effect = OperationalError("attach", {}, Exception("temporary outage"))
    await run(bot, message())
    turn = bot.user_active_executions["sender"][1]
    assert turn.accepted
    await bot._handle_control("sender", message("/stop"), "/stop")
    await turn.stop_task
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == 1
        assert (
            db.query(TaskExecutionCommand).filter_by(kind="pause").one().target_run_id
            == turn.run_id
        )


@pytest.mark.asyncio
async def test_shutdown_detaches_observer_and_recovery_delivers(ingress):
    make, sessions, observe = ingress
    bot = make()
    complete = observe.side_effect
    entered = asyncio.Event()

    async def blocked(handler):
        entered.set()
        await asyncio.Future()

    observe.side_effect = blocked
    bot._handle_message_sync(message())
    await asyncio.wait_for(entered.wait(), 5)
    turn = bot.user_active_executions["sender"][1]
    await asyncio.wait_for(bot.stop(), 5)
    assert not bot.user_message_tasks
    assert not bot.user_active_executions
    with sessions() as db:
        assert [row.kind for row in db.query(TaskExecutionCommand)] == ["start"]
        assert db.query(TaskChannelDelivery).one().status == "pending"
    await complete(SimpleNamespace(task_id=turn.selection.task_id))
    restarted = make()
    await channel_delivery.recover_channel_results(
        restarted.channel_id, restarted._deliver_shared_result
    )
    with sessions() as db:
        assert db.query(TaskChannelDelivery).one().status == "delivered"


@pytest.mark.asyncio
async def test_observation_failure_recovers_without_false_error(ingress):
    from sqlalchemy.exc import OperationalError

    make, sessions, observe = ingress
    bot = make()
    complete = observe.side_effect
    observe.side_effect = OperationalError("attach", {}, Exception("temporary outage"))
    await run(bot, message())
    turn = bot.user_active_executions["sender"][1]
    assert all("Sorry" not in call.args[1] for call in bot._send_text.await_args_list)
    with sessions() as db:
        assert db.query(TaskChannelDelivery).one().status == "pending"
    await complete(SimpleNamespace(task_id=turn.selection.task_id))
    await channel_delivery.recover_channel_results(
        bot.channel_id, bot._deliver_shared_result
    )
    with sessions() as db:
        assert db.query(TaskChannelDelivery).one().status == "delivered"


@pytest.mark.asyncio
async def test_replay_observation_failure_does_not_block_new_input(ingress):
    from sqlalchemy.exc import OperationalError

    make, sessions, observe = ingress
    bot = make()
    complete = observe.side_effect
    await run(bot, message("original", "old"))
    attempts = []

    async def fail_once(handler):
        attempts.append(handler)
        if len(attempts) == 1:
            raise OperationalError("attach", {}, Exception("temporary outage"))
        return await complete(handler)

    observe.side_effect = fail_once
    await run(bot, message("original", "old"), message("new", "new"))
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == 2
        assert db.query(TaskExecutionCommand).filter_by(kind="start").count() == 2
    assert len(attempts) == 2
    assert all("Sorry" not in call.args[1] for call in bot._send_text.await_args_list)


@pytest.mark.asyncio
async def test_repartition_does_not_repeat_rejection_notice(ingress, monkeypatch):
    make, sessions, _ = ingress
    bot = make()
    await run(bot, message("original", "old"))
    original = module.accept_channel_input
    attempts = []

    def accept(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise inputs.ChannelInputBatchChanged()
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "accept_channel_input", accept)
    await run(bot, message("changed", "old"), message("new", "new"))
    assert (
        sum(
            "different content" in call.args[1]
            for call in bot._send_text.await_args_list
        )
        == 1
    )
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == 2
    assert len(attempts) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("file_key", ["", "unavailable"])
async def test_attachment_failure_is_atomic_within_chat_group(ingress, file_key):
    make, sessions, _ = ingress
    bot = make()
    bot._download_feishu_file_sync = Mock(return_value=None)
    await run(
        bot,
        message("explain attachment", "text", "chat-a"),
        message(file_key, "file", "chat-a", "file"),
        message("independent", "good", "chat-b"),
    )
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == 1
        command = db.query(TaskExecutionCommand).filter_by(kind="start").one()
        assert command.payload["message"] == "independent"
        assert db.query(TaskChannelDelivery).one().destination["chat_id"] == "chat-b"
    assert any(
        call.args[0] == "chat-a" and "were not accepted" in call.args[1]
        for call in bot._send_text.await_args_list
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/stop", "/new"])
async def test_control_during_rejection_notice_stops_later_notices(ingress, command):
    make, sessions, _ = ingress
    bot = make()
    await run(bot, message("A", "a"), message("B", "b"))
    original_send = bot._send_text
    notices = []

    async def send(chat, text):
        if "different content" in text:
            notices.append(text)
            if len(notices) == 1:
                await bot._handle_control("sender", message(command), command)
        return await original_send(chat, text)

    bot._send_text = send
    await run(bot, message("changed A", "a"), message("changed B", "b"))
    assert len(notices) == 1
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == 2
