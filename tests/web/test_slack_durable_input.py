"""Slack shared ingress accepts physical messages once and retains progress."""

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
async def test_slack_progress_and_final_use_persisted_message(
    slack_ingress, monkeypatch
):
    from xagent.core.agent.trace import (
        TraceAction,
        TraceCategory,
        TraceEvent,
        TraceEventType,
        TraceScope,
    )
    from xagent.web.services import channel_delivery, shared_channel_execution
    from xagent.web.services.channel_delivery import deliver_channel_result

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
    observer = shared_channel_execution.SharedChannelTurn.observe.await_args.args[0]
    claim = Mock(wraps=channel_delivery._claim)
    monkeypatch.setattr(channel_delivery, "_claim", claim)
    irrelevant = TraceEvent(
        TraceEventType(TraceScope.TASK, TraceAction.START, TraceCategory.MESSAGE),
        task_id=str(task_id),
        data={"role": "user", "content": "hello"},
    )
    await observer.handle_event(irrelevant)
    claim.assert_not_called()
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
    # Final delivery is independent of a throttled or duplicate progress event.
    await observer.handle_event(event)
    assert claim.call_count == 1
    await deliver_channel_result(command_id, bot._deliver_shared_result)
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


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["conflict", "unauthorized", "missing_identity"])
async def test_slack_invalid_replay_cannot_accept_another_command(
    slack_ingress, failure
):
    make_bot, sessions = slack_ingress
    bot = make_bot()
    event = {
        "type": "message",
        "user": "sender",
        "channel": "chat",
        "ts": "1",
        "text": "hello",
    }
    await bot._process_event("conversation", {"team_id": "team"}, event)
    if failure == "conflict":
        event["text"] = "changed"
    elif failure == "unauthorized":
        with sessions() as db:
            db.query(UserChannel).one().config = {"allowed_users": ["other"]}
            db.commit()
    else:
        del event["ts"]
    bot._send_text.reset_mock()
    await bot._process_event("conversation", {"team_id": "team"}, event)
    bot._send_text.assert_awaited_once()
    with sessions() as db:
        assert db.query(TaskExecutionCommand).count() == 1
        assert db.query(TaskInputReceipt).count() == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["download", "acceptance"])
async def test_slack_failed_input_compensates_staged_files(
    slack_ingress, monkeypatch, tmp_path, failure
):
    from unittest.mock import AsyncMock

    from xagent.core.file_storage.factory import get_unscoped_file_storage
    from xagent.web.channels.slack import bot as slack
    from xagent.web.models.uploaded_file import UploadedFile
    from xagent.web.services.channel_runtime import DownloadedChannelFile

    make_bot, sessions = slack_ingress
    bot = make_bot()
    storage = tmp_path / "objects"
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", storage.as_uri())
    get_unscoped_file_storage.cache_clear()
    source = tmp_path / "source.txt"
    source.write_text("payload")
    bot._download_slack_file = AsyncMock(
        side_effect=[
            DownloadedChannelFile("source.txt", source, "text/plain", 7, "F1"),
            None,
        ]
    )
    if failure == "acceptance":
        monkeypatch.setattr(
            slack,
            "accept_channel_input",
            Mock(side_effect=RuntimeError("failed accept")),
        )
    event = {
        "type": "message",
        "user": "sender",
        "channel": "chat",
        "ts": "1",
        "files": [{"id": "F1"}],
    }
    if failure == "download":
        event["files"].append({"id": "F2"})
    try:
        await bot._process_event("conversation", {"team_id": "team"}, event)
        with sessions() as db:
            assert db.query(Task).count() == 0
            assert db.query(TaskInputReceipt).count() == 0
            assert db.query(UploadedFile).count() == 0
        assert not any(path.is_file() for path in storage.rglob("*"))
        bot._send_text.assert_awaited_once()
    finally:
        get_unscoped_file_storage.cache_clear()


@pytest.mark.asyncio
async def test_slack_cancellation_waits_for_staging_before_compensation(
    slack_ingress, monkeypatch, tmp_path
):
    import asyncio
    import threading
    from unittest.mock import AsyncMock

    from xagent.core.file_storage.factory import get_unscoped_file_storage
    from xagent.web.channels.slack import bot as slack
    from xagent.web.models.uploaded_file import UploadedFile
    from xagent.web.services.channel_runtime import DownloadedChannelFile

    make_bot, sessions = slack_ingress
    bot = make_bot()
    storage = tmp_path / "objects"
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", storage.as_uri())
    get_unscoped_file_storage.cache_clear()
    entered, release = threading.Event(), threading.Event()
    original = slack.stage_uploaded_file_from_local_path

    async def download(_, directory):
        path = directory / "input.txt"
        path.write_text("payload")
        return DownloadedChannelFile("input.txt", path, "text/plain", 7, "F1")

    def stage(**kwargs):
        entered.set()
        assert release.wait(10)
        assert kwargs["local_path"].exists()
        return original(**kwargs)

    bot._download_slack_file = AsyncMock(side_effect=download)
    monkeypatch.setattr(slack, "stage_uploaded_file_from_local_path", stage)
    event = {
        "type": "message",
        "user": "sender",
        "channel": "chat",
        "ts": "1",
        "files": [{"id": "F1"}],
    }
    worker = asyncio.create_task(
        bot._process_event("conversation", {"team_id": "team"}, event)
    )
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        worker.cancel()
        await asyncio.sleep(0)
        assert not worker.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await worker
        with sessions() as db:
            assert db.query(Task).count() == 0
            assert db.query(UploadedFile).count() == 0
        assert not any(path.is_file() for path in storage.rglob("*"))
    finally:
        release.set()
        await asyncio.gather(worker, return_exceptions=True)
        get_unscoped_file_storage.cache_clear()
