"""Stable provider inputs cannot select a new task on retry."""

from dataclasses import replace
from unittest.mock import Mock

import pytest
from sqlalchemy.orm import sessionmaker

from tests.web.services.task_database_shared import engine as engine_fixture
from xagent.web.models.database import Base
from xagent.web.models.task import Task
from xagent.web.models.task_channel_delivery import TaskChannelDelivery
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.task_input_receipt import TaskInputReceipt
from xagent.web.models.user import User
from xagent.web.models.user_channel import UserChannel
from xagent.web.services import channel_input_acceptance as inputs
from xagent.web.services import task_event_bridge
from xagent.web.services.task_orchestrator import TaskTurnError, TaskTurnPayload

engine = engine_fixture


@pytest.fixture
def ingress(engine, monkeypatch):
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine)
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setattr(inputs, "get_session_local", lambda: sessions)
    monkeypatch.setattr(task_event_bridge, "_bridge", Mock(host_id="ingress"))
    with sessions() as db:
        owner = User(username="owner", password_hash="unused")
        db.add(owner)
        db.flush()
        channel = UserChannel(
            user_id=owner.id,
            channel_type="slack",
            channel_name="test",
            config={"allowed_users": ["sender"]},
            is_active=True,
        )
        db.add(channel)
        db.commit()
        incoming = inputs.ChannelInput(
            int(channel.id),
            "sender",
            "slack",
            ("team", "chat"),
            "123.456",
            "hello",
            (),
            {"chat_id": "chat", "thread_ts": "123.456", "loading_ts": None},
        )
    yield incoming, sessions
    Base.metadata.drop_all(engine)


def accept(incoming, **changes):
    owner, _ = inputs.lookup_channel_input(incoming)
    return inputs.accept_channel_input(
        incoming,
        **(
            dict(
                owner_id=owner,
                active_task_id=None,
                channel_name="test",
                payload=TaskTurnPayload(incoming.text),
                staged_files=(),
                host_id="host",
            )
            | changes
        ),
    )


def test_duplicate_uses_original_task_and_delivery(ingress):
    incoming, sessions = ingress
    first = accept(incoming)
    replay = accept(incoming, active_task_id=-1)
    assert replay.replayed
    assert (replay.task_id, replay.command_db_id) == (
        first.task_id,
        first.command_db_id,
    )
    with sessions() as db:
        assert db.query(Task).count() == 1
        assert db.query(TaskExecutionCommand).count() == 1
        assert db.query(TaskChannelDelivery).count() == 1
        assert db.query(TaskInputReceipt).count() == 1


def test_content_conflict_and_distinct_identity(ingress):
    incoming, _ = ingress
    first = accept(incoming)
    with pytest.raises(TaskTurnError, match="input_conflict"):
        accept(replace(incoming, text="changed"))
    second = accept(replace(incoming, message_id="different"))
    assert second.task_id != first.task_id


def test_failed_start_rolls_back_receipt_and_new_task(ingress, monkeypatch):
    incoming, sessions = ingress
    monkeypatch.setattr(
        inputs,
        "accept_channel_turn_no_commit",
        Mock(side_effect=RuntimeError("failed START")),
    )
    with pytest.raises(RuntimeError, match="failed START"):
        accept(incoming)
    with sessions() as db:
        assert db.query(Task).count() == 0
        assert db.query(TaskInputReceipt).count() == 0


def test_deleted_target_does_not_resurrect(ingress):
    incoming, sessions = ingress
    accept(incoming)
    with sessions() as db:
        db.query(Task).delete(synchronize_session=False)
        db.commit()
    with pytest.raises(TaskTurnError, match="input_unavailable"):
        accept(incoming)


def test_simultaneous_acceptance_has_one_winner(ingress):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    incoming, sessions = ingress
    owner, _ = inputs.lookup_channel_input(incoming)
    barrier = Barrier(2)

    def submit():
        barrier.wait(timeout=10)
        return inputs.accept_channel_input(
            incoming,
            owner_id=owner,
            active_task_id=None,
            channel_name="test",
            payload=TaskTurnPayload("hello"),
            staged_files=(),
            host_id="host",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(submit) for _ in range(2)]
        results = [future.result(timeout=20) for future in futures]
    assert sorted(result.replayed for result in results) == [False, True]
    assert len({result.command_db_id for result in results}) == 1
    with sessions() as db:
        assert db.query(Task).count() == db.query(TaskInputReceipt).count() == 1


def test_replay_rechecks_sender_authorization(ingress):
    from xagent.web.services.channel_runtime import ChannelAuthorizationError

    incoming, sessions = ingress
    accept(incoming)
    with sessions() as db:
        db.get(UserChannel, incoming.channel_id).config = {
            "allowed_users": ["someone-else"]
        }
        db.commit()
    with pytest.raises(ChannelAuthorizationError):
        inputs.lookup_channel_input(incoming)


def test_commit_acknowledgement_loss_replays_saved_input(ingress, monkeypatch):
    from sqlalchemy.orm import Session

    incoming, sessions = ingress
    original = Session.commit

    def commit(db):
        accepted = any(isinstance(item, TaskInputReceipt) for item in db.dirty)
        original(db)
        if accepted:
            raise ConnectionError("commit acknowledgement lost")

    monkeypatch.setattr(Session, "commit", commit)
    saved = accept(incoming)
    assert not saved.replayed
    assert saved.selection.is_new_task
    with sessions() as db:
        assert db.query(TaskExecutionCommand).count() == 1


@pytest.fixture
def slack_ingress(ingress, monkeypatch, tmp_path):
    from unittest.mock import AsyncMock

    from xagent.web.channels.slack import bot as slack
    from xagent.web.services import (
        channel_delivery,
        shared_channel_execution,
        uploaded_file_store,
    )

    incoming, sessions = ingress
    for module in (channel_delivery, shared_channel_execution, uploaded_file_store):
        monkeypatch.setattr(module, "get_session_local", lambda: sessions)
    monkeypatch.setattr(slack, "get_storage_root", lambda: tmp_path)
    monkeypatch.setattr(
        shared_channel_execution.SharedChannelTurn,
        "observe",
        AsyncMock(return_value={"status": "accepted"}),
    )
    local_agent = Mock(side_effect=AssertionError("ingress must not create Agent"))
    persist = AsyncMock(
        side_effect=AssertionError("START acceptance owns the transcript")
    )
    monkeypatch.setattr(slack, "get_agent_manager", local_agent)
    monkeypatch.setattr(slack, "persist_channel_user_message", persist)

    def bot():
        result = slack.SlackBotInstance(
            "token", None, "test", channel_id=incoming.channel_id, bot_user_id="bot"
        )
        result._send_text = AsyncMock(return_value="loading-ts")
        result._send_final_text = AsyncMock()
        result._save_active_tasks = Mock()
        result.web_client.chat_update = AsyncMock()
        return result

    yield bot, sessions
    local_agent.assert_not_called()
    persist.assert_not_awaited()


@pytest.mark.asyncio
async def test_slack_two_envelopes_and_restart_reuse_loading_and_task(slack_ingress):
    make_bot, sessions = slack_ingress
    envelope = {"team_id": "team"}
    event = {
        "type": "app_mention",
        "user": "sender",
        "channel": "chat",
        "ts": "1.0",
        "text": "<@bot> hello",
    }
    first = make_bot()
    await first._process_event("conversation", envelope, event)
    first._send_text.assert_awaited_once()
    second = make_bot()
    second.active_tasks["conversation"] = -123
    await second._process_event(
        "conversation",
        envelope | {"event_id": "different"},
        event | {"type": "message", "client_msg_id": "client-id", "text": "hello"},
    )
    second._send_text.assert_not_awaited()
    assert second.active_tasks["conversation"] == -123
    with sessions() as db:
        assert db.query(TaskExecutionCommand).count() == 1
        assert db.query(Task).count() == 1
        assert (
            db.query(TaskChannelDelivery).one().destination["loading_ts"]
            == "loading-ts"
        )


@pytest.mark.asyncio
async def test_slack_progress_and_final_use_persisted_message(slack_ingress):
    from xagent.core.agent.trace import (
        TraceAction,
        TraceCategory,
        TraceEvent,
        TraceEventType,
        TraceScope,
    )
    from xagent.web.channels.slack.bot import _DurableSlackProgress

    make_bot, sessions = slack_ingress
    bot = make_bot()
    await bot._process_event(
        "conversation",
        {"team_id": "team"},
        {
            "type": "message",
            "user": "sender",
            "channel": "chat",
            "ts": "1.0",
            "text": "hello",
        },
    )
    bot._send_final_text.assert_awaited_once()
    assert "still being processed" in bot._send_final_text.await_args.kwargs["text"]
    assert bot._send_final_text.await_args.kwargs["loading_ts"] == "loading-ts"
    bot._send_final_text.reset_mock()
    with sessions() as db:
        command_id = int(db.query(TaskExecutionCommand).one().id)
        task_id = int(db.query(Task).one().id)
    event = TraceEvent(
        TraceEventType(TraceScope.TASK, TraceAction.START, TraceCategory.TOOL),
        task_id=str(task_id),
        data={"tool_name": "search"},
    )
    observer = _DurableSlackProgress(bot, command_id)
    await observer.handle_event(event)
    assert bot.web_client.chat_update.await_args.kwargs["ts"] == "loading-ts"
    with sessions() as db:
        command = db.get(TaskExecutionCommand, command_id)
        command.status = "completed"
        command.result = {
            "channel_result": {
                "success": True,
                "status": "completed",
                "output": "answer",
            }
        }
        db.commit()
    await observer.handle_event(event)
    bot._send_final_text.assert_awaited_once_with(
        channel_id="chat", thread_ts="1.0", loading_ts="loading-ts", text="answer"
    )
    await observer.handle_event(event)
    assert bot.web_client.chat_update.await_count == 1
    bot._send_text.assert_awaited_once()
    bot._send_final_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_slack_files_stage_before_task_and_retry_does_not_download(
    slack_ingress, tmp_path, monkeypatch
):
    from unittest.mock import AsyncMock

    from xagent.web.channels.slack import bot as slack
    from xagent.web.models.uploaded_file import UploadedFile
    from xagent.web.services.channel_runtime import DownloadedChannelFile
    from xagent.web.services.uploaded_file_store import StagedUploadedFile

    make_bot, sessions = slack_ingress
    bot = make_bot()
    source = tmp_path / "input.txt"
    source.write_text("contents")

    async def download(*args):
        with sessions() as db:
            assert db.query(Task).count() == 0
        return DownloadedChannelFile("input.txt", source, "text/plain", 8, "F1")

    bot._download_slack_file = AsyncMock(side_effect=download)

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
            "slack",
        )

    monkeypatch.setattr(slack, "stage_uploaded_file_from_local_path", stage)
    event = {
        "type": "message",
        "user": "sender",
        "channel": "chat",
        "ts": "1.0",
        "files": [{"id": "F1", "url_private": "old"}],
    }
    await bot._process_event("conversation", {"team_id": "team"}, event)
    await bot._process_event(
        "conversation",
        {"team_id": "team"},
        event | {"files": [{"id": "F1", "url_private": "new"}]},
    )
    bot._download_slack_file.assert_awaited_once()
    with sessions() as db:
        uploaded = db.query(UploadedFile).one()
        command = db.query(TaskExecutionCommand).one()
        from xagent.web.models.chat_message import TaskChatMessage

        assert (
            db.query(TaskChatMessage)
            .filter_by(role="user")
            .one()
            .attachments[0]["type"]
            == "text/plain"
        )
        assert uploaded.task_id == command.task_id
        assert command.payload["file_ids"] == [uploaded.file_id]


def test_attachment_metadata_rolls_back_with_start(ingress, tmp_path, monkeypatch):
    from xagent.web.models.uploaded_file import UploadedFile
    from xagent.web.services.uploaded_file_store import StagedUploadedFile

    incoming, sessions = ingress
    owner, _ = inputs.lookup_channel_input(incoming)
    staged = StagedUploadedFile(
        "file",
        owner,
        None,
        "input.txt",
        str(tmp_path / "input.txt"),
        "local",
        "key",
        None,
        "checksum",
        None,
        None,
        None,
        "text/plain",
        8,
    )
    monkeypatch.setattr(
        inputs,
        "accept_channel_turn_no_commit",
        Mock(side_effect=RuntimeError("failed START")),
    )
    with pytest.raises(RuntimeError, match="failed START"):
        accept(incoming, staged_files=(staged,))
    with sessions() as db:
        assert (
            db.query(UploadedFile).count()
            == db.query(Task).count()
            == db.query(TaskInputReceipt).count()
            == 0
        )


@pytest.mark.asyncio
async def test_final_waits_for_loading_claim_and_never_restarts_execution(
    slack_ingress,
):
    import asyncio
    from unittest.mock import AsyncMock

    from xagent.web.channels.slack.bot import _DurableSlackProgress

    make_bot, sessions = slack_ingress
    bot = make_bot()
    entered, release = asyncio.Event(), asyncio.Event()

    async def send(*args, **kwargs):
        entered.set()
        await release.wait()
        return "persisted-loading"

    bot._send_text = AsyncMock(side_effect=send)
    processing = asyncio.create_task(
        bot._process_event(
            "conversation",
            {"team_id": "team"},
            {
                "type": "message",
                "user": "sender",
                "channel": "chat",
                "ts": "1.0",
                "text": "hello",
            },
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 10)
        with sessions() as db:
            command = db.query(TaskExecutionCommand).one()
            command_id = int(command.id)
            command.status = "completed"
            command.result = {
                "channel_result": {
                    "success": True,
                    "status": "completed",
                    "output": "answer",
                }
            }
            db.commit()
        await _DurableSlackProgress(bot, command_id).send(None)
        bot._send_final_text.assert_not_awaited()
    finally:
        release.set()
        await processing
    bot._send_text.assert_awaited_once()
    bot._send_final_text.assert_awaited_once()
    assert bot._send_final_text.await_args.kwargs["loading_ts"] == "persisted-loading"
    with sessions() as db:
        assert db.query(TaskExecutionCommand).count() == 1
        assert db.query(TaskChannelDelivery).one().status == "delivered"


@pytest.mark.asyncio
async def test_observer_attaches_to_original_command_without_new_start(
    ingress, monkeypatch
):
    from unittest.mock import AsyncMock

    from xagent.web.services import shared_channel_execution as shared

    incoming, sessions = ingress
    saved = accept(incoming)
    monkeypatch.setattr(shared, "get_session_local", lambda: sessions)
    monkeypatch.setattr(shared, "notify_task_command_dispatcher", Mock())
    monkeypatch.setattr(
        shared.SharedChannelTurn,
        "wait_result",
        AsyncMock(return_value={"status": "accepted"}),
    )
    task_event_bridge._bridge.register_origin.return_value = "new-origin"
    turn = saved.as_turn()
    try:
        await turn.observe(None)
        with sessions() as db:
            command = db.query(TaskExecutionCommand).one()
            assert command.command_id == saved.command_id
            assert command.reply_origin == "new-origin"
            assert command.reply_host_id == "ingress"
    finally:
        await turn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("chat", ["C1", "D1"])
@pytest.mark.parametrize("completion", ["lost_ack", "cancel"])
async def test_slack_late_acceptance_preserves_followup_conversation(
    slack_ingress, monkeypatch, chat, completion
):
    import asyncio
    import threading

    from sqlalchemy.orm import Session

    from xagent.web.channels.slack import bot as slack
    from xagent.web.models.task import TaskStatus

    make_bot, sessions = slack_ingress
    bot = make_bot()
    bot._save_active_tasks = slack.SlackBotInstance._save_active_tasks.__get__(bot)
    original = Session.commit

    def commit(db):
        accepted = any(isinstance(item, TaskInputReceipt) for item in db.dirty)
        original(db)
        if accepted:
            raise ConnectionError("commit acknowledgement lost")

    if completion == "lost_ack":
        monkeypatch.setattr(Session, "commit", commit)
    else:
        entered, release = threading.Event(), threading.Event()
        original_accept = slack.accept_channel_input

        def accept(*args, **kwargs):
            result = original_accept(*args, **kwargs)
            entered.set()
            assert release.wait(10)
            return result

        monkeypatch.setattr(slack, "accept_channel_input", accept)
    event = {
        "type": "app_mention",
        "user": "sender",
        "channel": chat,
        "ts": "1.0",
        "text": "hello",
    }
    envelope = {"team_id": "team", "event": event}
    await bot.handle_events_api_payload(envelope)
    if completion == "cancel":
        assert await asyncio.to_thread(entered.wait, 10)
        worker = next(iter(bot.event_tasks.values()))
        stopping = asyncio.create_task(bot.stop())
        try:
            async with asyncio.timeout(5):
                while not worker.cancelling():
                    await asyncio.sleep(0)
        finally:
            release.set()
            await stopping
        assert worker.cancelled()
    else:
        await asyncio.gather(*list(bot.event_tasks.values()))
    # Restart reads the association persisted before cancellation was propagated.
    bot = make_bot()
    key = bot._conversation_key(envelope, event)
    with sessions() as db:
        task = db.query(Task).one()
        task_id = int(task.id)
        assert bot.active_tasks[key] == task_id
        command = db.query(TaskExecutionCommand).one()
        command.status = "completed"
        command.result = {
            "channel_result": {
                "success": True,
                "status": "completed",
                "output": "answer",
            }
        }
        task.status = TaskStatus.COMPLETED
        task.run_id = command.target_run_id
        db.commit()
    followup = event | {"type": "message", "ts": "2.0", "text": "continue"}
    if chat == "C1":
        followup["thread_ts"] = "1.0"
    await bot.handle_events_api_payload(envelope | {"event": followup})
    await asyncio.gather(*list(bot.event_tasks.values()))
    with sessions() as db:
        assert db.query(Task).count() == 1
        commands = (
            db.query(TaskExecutionCommand).order_by(TaskExecutionCommand.id).all()
        )
        assert len(commands) == 2
        assert commands[1].task_id == task_id
        assert commands[1].payload["message"] == "continue"


def test_batch_receipts_survive_split_and_overlapping_retry(ingress):
    incoming, sessions = ingress
    second = replace(incoming, message_id="2", text="second")
    third = replace(incoming, message_id="3", text="third")
    first = accept(
        incoming, additional_inputs=(second,), payload=TaskTurnPayload("hello\nsecond")
    )
    _, pending, replays, unavailable = inputs.lookup_channel_inputs(
        (second, incoming, third, third)
    )
    assert not unavailable
    assert pending == (third,)
    assert [row.command_db_id for row in replays] == [first.command_db_id]
    assert accept(second).command_db_id == first.command_db_id
    with pytest.raises(inputs.ChannelInputBatchChanged):
        accept(second, additional_inputs=(third,))
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == 2
        assert db.query(TaskExecutionCommand).count() == 1


def test_batch_failure_rolls_back_every_alias(ingress, monkeypatch):
    incoming, sessions = ingress
    second = replace(incoming, message_id="2")
    monkeypatch.setattr(
        inputs,
        "accept_channel_turn_no_commit",
        Mock(side_effect=RuntimeError("failed")),
    )
    with pytest.raises(RuntimeError):
        accept(incoming, additional_inputs=(second,))
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == db.query(Task).count() == 0


@pytest.fixture(params=["feishu", "telegram"])
def provider_ingress(ingress, monkeypatch, request):
    """Exercise adapters against the real receipt transaction, without a network."""
    import importlib
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    incoming, sessions = ingress
    platform = request.param
    module = importlib.import_module(f"xagent.web.channels.{platform}.bot")
    factory = importlib.import_module(
        f"tests.web.test_{platform}_message_queue"
    ).make_bot
    bot = factory()
    sender = "sender" if platform == "feishu" else 123
    with sessions() as db:
        channel = db.get(UserChannel, incoming.channel_id)
        channel.channel_type = platform
        channel.config = {"allowed_users": [str(sender)]}
        db.commit()
    bot.channel_id = incoming.channel_id
    bot.channel_name = "test"
    bot.active_tasks = {}
    bot._active_tasks_unsaved = False
    bot._save_active_tasks = Mock(return_value=True)
    bot._observe_shared_input = AsyncMock()
    bot._send_text = AsyncMock(return_value="loading")
    bot.start_time = 1000
    bot.bot = Mock()
    monkeypatch.setattr(module, "deliver_channel_result", AsyncMock(return_value=False))
    monkeypatch.setattr(
        module,
        "get_agent_manager",
        Mock(side_effect=AssertionError("No local executor")),
    )

    def message(identity, text="hello", voice=None):
        if platform == "feishu":
            import json

            return SimpleNamespace(
                event=SimpleNamespace(
                    message=SimpleNamespace(
                        chat_id="chat",
                        message_id=str(identity),
                        message_type="text",
                        content=json.dumps({"text": text}),
                        create_time="2000",
                    )
                )
            )
        return SimpleNamespace(
            message_id=identity,
            text=text,
            caption=None,
            document=None,
            photo=None,
            audio=None,
            video=None,
            voice=voice,
            message_thread_id=None,
            reply_to_message=None,
            chat=SimpleNamespace(id=456),
            answer=AsyncMock(),
        )

    async def process(*messages):
        if platform == "feishu":
            await bot._process_messages_batch(sender, list(messages))
        else:
            await bot._process_user_messages_batch(sender, list(messages))

    return bot, message, process, sessions, platform, module


@pytest.mark.asyncio
async def test_provider_retry_repartition_only_accepts_new_messages(provider_ingress):
    bot, message, process, sessions, platform, module = provider_ingress
    a, b, c = message(1, "A"), message(2, "B"), message(3, "C")
    await process(a, b)
    with sessions() as db:
        first = db.query(TaskExecutionCommand).one()
        first_id = first.id
        assert db.query(TaskInputReceipt).count() == 2
    # A restarted/switching ingress must not associate a replay with a new task.
    bot.active_tasks.clear()
    bot._save_active_tasks.reset_mock()
    await process(b, a, b)
    assert not bot.active_tasks
    bot._save_active_tasks.assert_not_called()
    await process(b, c)
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == 3
        commands = db.query(TaskExecutionCommand).all()
        assert len(commands) == 2
        receipts = db.query(TaskInputReceipt).all()
        assert sum(r.command_db_id == first_id for r in receipts) == 2
        assert db.query(TaskChannelDelivery).count() == 2
    assert bot._observe_shared_input.await_count == 2


@pytest.mark.asyncio
async def test_provider_content_conflict_cannot_create_another_turn(provider_ingress):
    bot, message, process, sessions, platform, module = provider_ingress
    await process(message(1))
    bot.active_tasks.clear()
    await process(message(1, "changed"))
    with sessions() as db:
        assert db.query(TaskExecutionCommand).count() == 1
    assert bot._observe_shared_input.await_count == 1


@pytest.mark.asyncio
async def test_provider_download_failure_does_not_accept_partial_batch(
    provider_ingress, monkeypatch
):
    import json
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    bot, message, process, sessions, platform, module = provider_ingress
    item = message(2)
    if platform == "feishu":
        item.event.message.message_type = "file"
        item.event.message.content = json.dumps({"file_key": "file"})
        bot._download_feishu_file_sync = Mock(return_value=None)
    else:
        item.document = SimpleNamespace(file_id="file", file_unique_id="stable")
        bot._download_telegram_files = AsyncMock(
            side_effect=ConnectionError("download failed")
        )
    await process(message(1), item)
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == 0
        assert db.query(Task).count() == 0
    bot._observe_shared_input.assert_not_awaited()
    if platform == "feishu":
        bot._send_text.assert_awaited_once()
        assert "could not be accepted" in bot._send_text.await_args.args[1]
    else:
        item.answer.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_group", [0, 1])
async def test_provider_batch_error_replies_to_failed_chat(
    provider_ingress, failed_group
):
    import json
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    bot, message, process, sessions, platform, module = provider_ingress
    items = [message(1), message(2)]
    if platform == "feishu":
        for index, item in enumerate(items):
            item.event.message.chat_id = f"chat-{index}"
        failed = items[failed_group].event.message
        failed.message_type = "file"
        failed.content = json.dumps({"file_key": "file"})
        bot._download_feishu_file_sync = Mock(return_value=None)
    else:
        for index, item in enumerate(items):
            item.chat.id = 456 + index
        items[failed_group].document = SimpleNamespace(
            file_id="file", file_unique_id="stable"
        )

        async def download(files, *args, **kwargs):
            if files:
                raise ConnectionError("download failed")
            return []

        bot._download_telegram_files = AsyncMock(side_effect=download)
    await process(*items)
    if platform == "feishu":
        bot._send_text.assert_awaited_once_with(
            f"chat-{failed_group}",
            "This message could not be accepted. Please try again.",
        )
    else:
        items[failed_group].answer.assert_awaited_once_with(
            "Sorry, an error occurred while processing your request."
        )
        items[1 - failed_group].answer.assert_not_awaited()
    # Earlier groups remain accepted; the failed group has no durable receipt.
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == failed_group


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_ingress", ["telegram"], indirect=True)
async def test_telegram_batch_error_replies_to_failed_topic(provider_ingress):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    bot, message, process, sessions, platform, module = provider_ingress
    first, second = message(1), message(2)
    first.message_thread_id = 10
    second.message_thread_id = 20
    first.document = SimpleNamespace(file_id="file", file_unique_id="stable")
    bot._download_telegram_files = AsyncMock(
        side_effect=ConnectionError("download failed")
    )
    await process(first, second)
    first.answer.assert_awaited_once_with(
        "Sorry, an error occurred while processing your request."
    )
    second.answer.assert_not_awaited()


@pytest.mark.asyncio
async def test_telegram_stop_during_download_leaves_no_receipt(provider_ingress):
    from unittest.mock import AsyncMock

    bot, message, process, sessions, platform, module = provider_ingress
    if platform != "telegram":
        return

    async def download(*args, **kwargs):
        bot.user_conversation_generations[123] = 1
        return []

    bot._download_telegram_files = AsyncMock(side_effect=download)
    await process(message(1))
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == 0
    assert not bot.active_tasks
    bot._observe_shared_input.assert_not_awaited()


@pytest.mark.asyncio
async def test_feishu_old_accepted_message_replays_but_old_new_input_is_skipped(
    provider_ingress,
):
    bot, message, process, sessions, platform, module = provider_ingress
    if platform != "feishu":
        return
    a = message(1)
    await process(a)
    bot.start_time = 3000
    bot.active_tasks.clear()
    await process(a, message(2))
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == 1
    assert not bot.active_tasks
    assert module.deliver_channel_result.await_count == 1


def test_concurrent_overlapping_batches_accept_each_physical_input_once(ingress):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    incoming, sessions = ingress
    owner, _ = inputs.lookup_channel_input(incoming)
    a, b, c = (replace(incoming, message_id=str(i), text=str(i)) for i in range(3))
    barrier = Barrier(2)

    def submit(items):
        barrier.wait(timeout=10)
        while items:
            _, pending, _, _ = inputs.lookup_channel_inputs(items)
            if not pending:
                return
            try:
                inputs.accept_channel_input(
                    pending[0],
                    additional_inputs=pending[1:],
                    owner_id=owner,
                    active_task_id=None,
                    channel_name="test",
                    payload=TaskTurnPayload("\n".join(i.text for i in pending)),
                    staged_files=(),
                    host_id="host",
                )
                return
            except inputs.ChannelInputBatchChanged:
                items = pending

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(submit, items) for items in [(a, b), (c, b)]]
        for future in futures:
            future.result(timeout=20)
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == 3
        assert db.query(TaskExecutionCommand).count() == 2


@pytest.mark.asyncio
async def test_provider_cancel_after_commit_keeps_original_selection(
    provider_ingress, monkeypatch
):
    import asyncio
    from threading import Event

    bot, message, process, sessions, platform, module = provider_ingress
    entered, release = Event(), Event()
    original = inputs.accept_channel_input

    def blocked(*args, **kwargs):
        result = original(*args, **kwargs)
        entered.set()
        assert release.wait(10)
        return result

    monkeypatch.setattr(module, "accept_channel_input", blocked)
    worker = asyncio.create_task(process(message(1)))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        worker.cancel()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await worker
    with sessions() as db:
        task = db.query(Task).one()
        sender = "sender" if platform == "feishu" else 123
        assert int(bot.active_tasks[sender]) == task.id
    bot._save_active_tasks.assert_called_once()
    bot._observe_shared_input.assert_not_awaited()


@pytest.mark.asyncio
async def test_telegram_voice_retry_skips_download_and_transcription(
    provider_ingress, monkeypatch, tmp_path
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from xagent.web.models.uploaded_file import UploadedFile
    from xagent.web.services import channel_input_files
    from xagent.web.services.channel_runtime import DownloadedChannelFile
    from xagent.web.services.uploaded_file_store import StagedUploadedFile

    bot, message, process, sessions, platform, module = provider_ingress
    if platform != "telegram":
        return
    source = tmp_path / "voice.ogg"
    source.write_bytes(b"audio")
    bot._download_telegram_files = AsyncMock(
        return_value=[
            DownloadedChannelFile("voice.ogg", source, "audio/ogg", 5, "voice")
        ]
    )
    asr = SimpleNamespace(
        transcribe=AsyncMock(return_value=SimpleNamespace(text="transcribed")),
        aclose=AsyncMock(),
    )
    bot._resolve_voice_asr_model_isolated = Mock(return_value=asr)
    bot.voice_transcription_timeout_seconds = 2

    def stage(**kw):
        return StagedUploadedFile(
            kw["file_id"],
            kw["user_id"],
            None,
            "voice.ogg",
            str(source),
            "local",
            kw["storage_key"],
            None,
            "checksum",
            None,
            None,
            None,
            "audio/ogg",
            5,
            "telegram",
        )

    monkeypatch.setattr(
        channel_input_files, "stage_uploaded_file_from_local_path", stage
    )
    monkeypatch.setattr(channel_input_files, "compensate_staged_uploaded_files", Mock())
    voice = SimpleNamespace(file_id="voice", file_unique_id="stable")
    await process(message(1, "", voice))
    bot.active_tasks.clear()
    await process(
        message(1, "", SimpleNamespace(file_id="renewed", file_unique_id="stable"))
    )
    bot._download_telegram_files.assert_awaited_once()
    asr.transcribe.assert_awaited_once()
    asr.aclose.assert_awaited_once()
    with sessions() as db:
        command = db.query(TaskExecutionCommand).one()
        assert "transcribed" in command.payload["message"]
        assert command.payload["file_ids"] == [db.query(UploadedFile).one().file_id]


@pytest.mark.asyncio
async def test_provider_progress_reuses_persisted_loading_message(
    provider_ingress, monkeypatch
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from xagent.web.services import channel_delivery
    from xagent.web.services.shared_channel_execution import SharedChannelTurn

    bot, message, process, sessions, platform, module = provider_ingress
    monkeypatch.setattr(channel_delivery, "get_session_local", lambda: sessions)
    from xagent.web.services import shared_channel_execution

    monkeypatch.setattr(shared_channel_execution, "get_session_local", lambda: sessions)
    monkeypatch.setattr(
        module, "deliver_channel_result", channel_delivery.deliver_channel_result
    )
    if platform == "telegram":
        # Restore real observation but keep the bridge wait controlled.
        bot._observe_shared_input = (
            module.TelegramBotInstance._observe_shared_input.__get__(bot)
        )
    else:
        bot._observe_shared_input = (
            module.FeishuBotInstance._observe_shared_input.__get__(bot)
        )

    async def observe(turn, handler):
        await handler.send()
        with sessions() as db:
            command = db.get(TaskExecutionCommand, turn.command_db_id)
            command.status = "completed"
            command.result = {
                "channel_result": {
                    "success": True,
                    "status": "completed",
                    "output": "answer",
                }
            }
            db.commit()
        return {"success": True, "status": "completed", "output": "answer"}

    monkeypatch.setattr(SharedChannelTurn, "observe", observe)
    final_destinations = []

    async def final(delivery, result):
        final_destinations.append(dict(delivery.destination))

    bot._deliver_shared_result = final
    item = message(1)
    if platform == "telegram":
        item.answer = AsyncMock(
            return_value=SimpleNamespace(message_id=99, delete=AsyncMock())
        )
    await process(item)
    with sessions() as db:
        delivery = db.query(TaskChannelDelivery).one()
        assert delivery.status == "delivered"
        assert delivery.destination["loading_message_id"] == (
            99 if platform == "telegram" else "loading"
        )
    assert len(final_destinations) == 1
    if platform == "telegram":
        item.answer.assert_awaited_once()
    else:
        bot._send_text.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("switch", [False, True])
async def test_telegram_stop_at_acceptance_preserves_only_current_selection(
    provider_ingress, monkeypatch, switch
):
    import asyncio
    from threading import Event
    from unittest.mock import AsyncMock

    bot, message, process, sessions, platform, module = provider_ingress
    if platform != "telegram":
        return
    entered, release = Event(), Event()
    original = inputs.accept_channel_input

    def blocked(*args, **kwargs):
        result = original(*args, **kwargs)
        entered.set()
        assert release.wait(10)
        return result

    monkeypatch.setattr(module, "accept_channel_input", blocked)
    bot._settle_fenced_turn = AsyncMock()
    worker = asyncio.create_task(process(message(1)))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        if switch:
            bot._start_new_conversation(123)
        else:
            bot._stop_current_conversation(123)
    finally:
        release.set()
    await worker
    with sessions() as db:
        task = db.query(Task).one()
        assert bot.active_tasks[123] == (-1 if switch else task.id)
    bot._settle_fenced_turn.assert_awaited_once()
    assert bot._settle_fenced_turn.await_args.kwargs["already_persisted"] is True
    bot._observe_shared_input.assert_not_awaited()


@pytest.mark.asyncio
async def test_provider_observer_failure_does_not_reject_accepted_input(
    provider_ingress, monkeypatch
):
    from unittest.mock import AsyncMock

    from xagent.web.services.channel_progress import DurableChannelProgress
    from xagent.web.services.shared_channel_execution import SharedChannelTurn

    bot, message, process, sessions, platform, module = provider_ingress
    cls = (
        module.FeishuBotInstance if platform == "feishu" else module.TelegramBotInstance
    )
    bot._observe_shared_input = cls._observe_shared_input.__get__(bot)
    monkeypatch.setattr(DurableChannelProgress, "send", AsyncMock())
    monkeypatch.setattr(
        SharedChannelTurn,
        "observe",
        AsyncMock(side_effect=ConnectionError("observer lost")),
    )
    monkeypatch.setattr(SharedChannelTurn, "close", AsyncMock())
    item = message(1)
    await process(item)
    with sessions() as db:
        assert db.query(TaskExecutionCommand).count() == 1
        assert db.query(TaskChannelDelivery).one().status == "pending"
    if platform == "telegram":
        assert not bot.user_active_executions
        item.answer.assert_not_awaited()
        bot.user_preparing_executions.add(123)
        assert bot._stop_current_conversation(123)
        assert bot._consume_user_stop_request(123)
    else:
        bot._send_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_provider_post_commit_file_cleanup_failure_is_not_rejection(
    provider_ingress, monkeypatch
):
    from contextlib import asynccontextmanager

    bot, message, process, sessions, platform, module = provider_ingress

    @asynccontextmanager
    async def staging(*args, **kwargs):
        yield (), []
        raise ConnectionError("compensation unavailable")

    monkeypatch.setattr(module, "stage_channel_input_files", staging)
    item = message(1)
    await process(item)
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == 1
        assert db.query(TaskChannelDelivery).one().status == "pending"
    if platform == "feishu":
        bot._send_text.assert_not_awaited()
    else:
        item.answer.assert_not_awaited()


@pytest.mark.asyncio
async def test_provider_unavailable_old_message_does_not_block_new_input(
    provider_ingress,
):
    bot, message, process, sessions, platform, module = provider_ingress
    old = message(1, "old")
    await process(old)
    with sessions() as db:
        db.query(Task).delete(synchronize_session=False)
        db.commit()
    bot.active_tasks.clear()
    bot._observe_shared_input.reset_mock()
    fresh = message(2, "fresh")
    await process(old, fresh)
    with sessions() as db:
        assert db.query(Task).one().description == "fresh"
        assert db.query(TaskInputReceipt).count() == 2
    bot._observe_shared_input.assert_awaited_once()
    if platform == "feishu":
        assert "no longer available" in bot._send_text.await_args.args[1]
    else:
        assert "no longer available" in old.answer.await_args.args[0]
        fresh.answer.assert_not_awaited()


@pytest.fixture
def second_channel(ingress):
    incoming, sessions = ingress
    with sessions() as db:
        owner = User(username="second-owner", password_hash="unused")
        db.add(owner)
        db.flush()
        channel = UserChannel(
            user_id=owner.id,
            channel_type="slack",
            channel_name="second",
            config={"allowed_users": ["sender"]},
            is_active=True,
        )
        db.add(channel)
        db.commit()
        return int(owner.id), replace(incoming, channel_id=int(channel.id))


def test_owner_change_between_lookup_and_acceptance_rejects_input(
    ingress, second_channel
):
    incoming, sessions = ingress
    previous_owner, _ = inputs.lookup_channel_input(incoming)
    other_owner, _ = second_channel
    with sessions() as db:
        db.get(UserChannel, incoming.channel_id).user_id = other_owner
        db.commit()
    with pytest.raises(TaskTurnError, match="owner_changed"):
        inputs.accept_channel_input(
            incoming,
            owner_id=previous_owner,
            active_task_id=None,
            channel_name="test",
            payload=TaskTurnPayload("hello"),
            staged_files=(),
            host_id="host",
        )
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == db.query(Task).count() == 0


def test_batch_cannot_accept_other_owners_channel(ingress, second_channel):
    incoming, sessions = ingress
    _, other = second_channel
    with pytest.raises(TaskTurnError, match="owner_changed"):
        accept(incoming, additional_inputs=(other,))
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == db.query(Task).count() == 0


@pytest.mark.parametrize(
    "binding", ["task_owner", "channel", "command_task", "command_actor"]
)
def test_replay_rejects_changed_persisted_binding(ingress, second_channel, binding):
    incoming, sessions = ingress
    other_owner, other = second_channel
    first, second = accept(incoming), accept(other)
    with sessions() as db:
        task = db.get(Task, first.task_id)
        command = db.get(TaskExecutionCommand, first.command_db_id)
        if binding == "task_owner":
            task.user_id = other_owner
        elif binding == "channel":
            task.channel_id = other.channel_id
        elif binding == "command_task":
            command.task_id = second.task_id
        else:
            command.actor_subject = db.get(User, other_owner).actor_subject
        db.commit()
    with pytest.raises(TaskTurnError, match="input_unavailable"):
        inputs.lookup_channel_input(incoming)


def test_raced_unavailable_alias_requests_repartition(ingress):
    incoming, sessions = ingress
    old = replace(incoming, message_id="old")
    previous = accept(old)
    with sessions() as db:
        db.delete(db.get(Task, previous.task_id))
        db.commit()
    with pytest.raises(inputs.ChannelInputBatchChanged):
        accept(incoming, additional_inputs=(old,))
    _, pending, replayed, unavailable = inputs.lookup_channel_inputs((incoming, old))
    assert pending == (incoming,)
    assert not replayed
    assert unavailable == (inputs.RejectedChannelInput(old, "input_unavailable"),)
    accept(pending[0])
    with sessions() as db:
        assert db.query(Task).count() == 1
        assert db.query(TaskInputReceipt).count() == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("include_original", [False, True])
async def test_provider_conflicting_old_message_does_not_block_new_input(
    provider_ingress,
    include_original,
):
    bot, message, process, sessions, platform, module = provider_ingress
    await process(message(1, "old"))
    bot.active_tasks.clear()
    bot._observe_shared_input.reset_mock()
    changed = message(1, "changed")
    fresh = message(2, "fresh")
    batch = [changed, fresh]
    if include_original:
        batch.append(message(1, "old"))
    await process(*batch)
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == 2
        assert {task.description for task in db.query(Task)} == {"old", "fresh"}
        assert db.query(TaskExecutionCommand).count() == 2
    bot._observe_shared_input.assert_awaited_once()
    if platform == "feishu":
        assert "different content" in bot._send_text.await_args.args[1]
    else:
        assert "different content" in changed.answer.await_args.args[0]
        fresh.answer.assert_not_awaited()


def test_raced_conflicting_alias_requests_repartition(ingress):
    incoming, sessions = ingress
    old = replace(incoming, message_id="old")
    accept(old)
    changed = replace(old, text="changed")
    with pytest.raises(inputs.ChannelInputBatchChanged):
        accept(incoming, additional_inputs=(changed,))
    _, pending, replayed, rejected = inputs.lookup_channel_inputs((incoming, changed))
    assert pending == (incoming,)
    assert not replayed
    assert rejected == (inputs.RejectedChannelInput(changed, "input_conflict"),)
    accept(pending[0])
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_ingress", ["telegram"], indirect=True)
@pytest.mark.parametrize("control", ["switch", "stop"])
async def test_telegram_control_during_lookup_suppresses_stale_rejection(
    provider_ingress, monkeypatch, control
):
    import asyncio
    from threading import Event

    bot, message, process, sessions, platform, module = provider_ingress
    old = message(1)
    await process(old)
    with sessions() as db:
        db.query(Task).delete(synchronize_session=False)
        db.commit()
    bot.active_tasks.clear()
    bot._observe_shared_input.reset_mock()
    entered, release = Event(), Event()
    lookup = inputs.lookup_channel_inputs

    def blocked(items):
        result = lookup(items)
        entered.set()
        assert release.wait(10)
        return result

    monkeypatch.setattr(module, "lookup_channel_inputs", blocked)
    fresh = message(2)
    processing = asyncio.create_task(process(old, fresh))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        if control == "switch":
            bot._start_new_conversation(123)
        else:
            bot._stop_current_conversation(123)
    finally:
        release.set()
    await processing
    old.answer.assert_not_awaited()
    assert "wasn't sent" in fresh.answer.await_args.args[0]
    bot._observe_shared_input.assert_not_awaited()
    with sessions() as db:
        assert db.query(Task).count() == 0
        assert db.query(TaskInputReceipt).count() == 1


def test_batch_actor_identity_failure_is_not_a_rejected_receipt(ingress, monkeypatch):
    incoming, _ = ingress
    monkeypatch.setattr(inputs, "_resolve_actor_subject", lambda *args: None)
    with pytest.raises(TaskTurnError, match="input_unavailable"):
        inputs.lookup_channel_inputs((incoming,))
