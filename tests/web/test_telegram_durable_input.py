"""Telegram physical inputs use real transactions with mocked platform IO."""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram import Dispatcher, types
from sqlalchemy.orm import sessionmaker

from tests.web.services.task_database_shared import engine as engine_fixture
from xagent.web.channels.telegram import bot as module
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
from xagent.web.services.channel_runtime import DownloadedChannelFile

engine = engine_fixture


def message(text="hello", identity=1, chat=123, topic=None, **kwargs):
    return types.Message(
        message_id=identity,
        date=kwargs.pop("date", datetime.now(timezone.utc)),
        chat=types.Chat(id=chat, type="private" if chat > 0 else "supergroup"),
        from_user=types.User(id=123, is_bot=False, first_name="Sender"),
        message_thread_id=topic,
        text=text,
        **kwargs,
    )


@pytest.fixture
def ingress(engine, monkeypatch, tmp_path):
    monkeypatch.setenv("XAGENT_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(tmp_path / "uploads"))
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
            channel_type="telegram",
            channel_name="test",
            config={"allowed_users": ["123"]},
            is_active=True,
        )
        db.add(channel)
        db.commit()
        cid = int(channel.id)

    def make_bot():
        from tests.web.test_telegram_message_queue import make_bot as make_control

        bot = make_control()
        bot.channel_id = cid
        bot.channel_name = "test"
        bot.instance_id = "test"
        bot.active_tasks_file = tmp_path / "active.json"
        bot._legacy_active_tasks_file = tmp_path / "legacy.json"
        bot.active_tasks = bot._load_active_tasks()
        bot._active_tasks_unsaved = False
        bot._started_at = None
        bot._menu_generation = "current"
        bot.bot = Mock()
        bot.dp = Dispatcher()
        bot._register_handlers()
        return bot

    sent = []

    async def answer(self, text, **kwargs):
        sent.append((self.chat.id, self.message_thread_id, text))
        return message(
            identity=1000 + len(sent), chat=self.chat.id, topic=self.message_thread_id
        )

    monkeypatch.setattr(types.Message, "answer", answer)
    monkeypatch.setattr(types.Message, "edit_text", AsyncMock())
    monkeypatch.setattr(types.Message, "delete", AsyncMock())
    make_bot.sent = sent
    yield make_bot, sessions, observe
    Base.metadata.drop_all(engine)


async def run(bot, *messages):
    await bot._process_user_messages_batch(123, list(messages))


@pytest.mark.asyncio
async def test_restart_replay_reuses_command_and_loading(ingress):
    make, sessions, observe = ingress
    await run(make(), message("A", 1), message("B", 2))
    await run(make(), message("B", 2), message("A", 1))
    with sessions() as db:
        assert db.query(Task).count() == 1
        assert db.query(TaskInputReceipt).count() == 2
        assert db.query(TaskExecutionCommand).count() == 1
        assert (
            db.query(TaskChannelDelivery).one().destination["loading_message_id"]
            == 1001
        )
    assert len(make.sent) == 1
    assert observe.await_count == 2


@pytest.mark.asyncio
async def test_overlap_accepts_only_new_input(ingress):
    make, sessions, _ = ingress
    bot = make()
    await run(bot, message("A", 1), message("B", 2))
    await run(bot, message("B", 2), message("C", 3))
    with sessions() as db:
        commands = (
            db.query(TaskExecutionCommand).order_by(TaskExecutionCommand.id).all()
        )
        assert [c.payload["message"] for c in commands] == ["A\nB", "C"]
        assert db.query(TaskInputReceipt).count() == 3
        assert db.query(Task).count() == 1


@pytest.mark.asyncio
async def test_conflict_does_not_block_new_message(ingress):
    make, sessions, _ = ingress
    bot = make()
    await run(bot, message())
    await run(bot, message("changed"), message("new", 2))
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == 2
        assert db.query(TaskExecutionCommand).count() == 2
    assert any("different content" in text for _, _, text in make.sent)


@pytest.mark.asyncio
async def test_topic_and_chat_are_distinct_inputs_and_reply_destinations(ingress):
    make, sessions, _ = ingress
    await run(
        make(),
        message("one", 1, -100, 10),
        message("two", 1, -100, 20),
        message("three", 1, -200, 10),
    )
    with sessions() as db:
        deliveries = (
            db.query(TaskChannelDelivery).order_by(TaskChannelDelivery.command_id).all()
        )
        assert [
            (d.destination["chat_id"], d.destination["message_thread_id"])
            for d in deliveries
        ] == [(-100, 10), (-100, 20), (-200, 10)]
        assert db.query(TaskInputReceipt).count() == 3
    assert [(chat, topic) for chat, topic, _ in make.sent] == [
        (-100, 10),
        (-100, 20),
        (-200, 10),
    ]


@pytest.mark.asyncio
async def test_save_failure_keeps_acceptance(ingress):
    make, sessions, _ = ingress
    bot = make()
    bot._save_active_tasks = Mock(return_value=False)
    await run(bot, message())
    selected = bot.active_tasks[123]
    await run(make(), message())
    with sessions() as db:
        assert db.query(Task).count() == 1
        assert db.query(TaskExecutionCommand).count() == 1
        assert db.query(TaskInputReceipt).one().task_id == selected
    assert any("request was accepted" in text for _, _, text in make.sent)


@pytest.mark.asyncio
async def test_observation_failure_keeps_control_and_accepts_later_input(ingress):
    make, sessions, observe = ingress
    bot = make()
    complete = observe.side_effect
    observe.side_effect = RuntimeError("observation unavailable")
    await run(bot, message())
    assert 123 in bot.user_active_executions
    assert not any("error occurred" in text for _, _, text in make.sent)
    # The worker finishes independently of the disconnected observer.
    await complete(SimpleNamespace(task_id=bot.active_tasks[123]))
    observe.side_effect = complete
    await run(bot, message("next", 2))
    with sessions() as db:
        assert db.query(TaskExecutionCommand).count() == 2


def voice(file_id="download", unique="stable"):
    return types.Voice(
        file_id=file_id, file_unique_id=unique, duration=1, mime_type="audio/ogg"
    )


def mock_files(bot, tmp_path):
    async def download(file, directory):
        path = directory / (file.file_id + ".ogg")
        path.write_bytes(b"audio")
        return DownloadedChannelFile(path.name, path, "audio/ogg", 5, file.file_id)

    bot._download_telegram_file = AsyncMock(side_effect=download)
    bot._resolve_voice_asr_model_isolated = Mock(return_value=object())
    bot._close_voice_asr_model = AsyncMock()
    bot._transcribe_uploaded_voice_files = AsyncMock(
        return_value={"download": "spoken words"}
    )


@pytest.mark.asyncio
async def test_voice_replay_skips_download_and_asr_after_file_id_changes(
    ingress, tmp_path
):
    make, sessions, _ = ingress
    bot = make()
    mock_files(bot, tmp_path)
    await run(bot, message(None, voice=voice()))
    bot._resolve_voice_asr_model_isolated.side_effect = AssertionError(
        "Replay must not need ASR"
    )
    await run(bot, message(None, voice=voice("refreshed")))
    bot._download_telegram_file.assert_awaited_once()
    bot._transcribe_uploaded_voice_files.assert_awaited_once()
    with sessions() as db:
        command = db.query(TaskExecutionCommand).one()
        assert "spoken words" in command.payload["message"]
        assert db.query(TaskInputReceipt).count() == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["download", "asr", "no_model"])
async def test_preparation_failure_rejects_group_and_continues_next_topic(
    ingress, tmp_path, failure
):
    make, sessions, _ = ingress
    bot = make()
    mock_files(bot, tmp_path)
    if failure == "download":
        bot._download_telegram_file.side_effect = OSError("unavailable")
    elif failure == "asr":
        bot._transcribe_uploaded_voice_files.side_effect = (
            module.TelegramVoiceTranscriptionError("failed")
        )
    else:
        bot._resolve_voice_asr_model_isolated.return_value = None
    await run(
        bot,
        message("text", 1, -100, 10),
        message(None, 2, -100, 10, voice=voice()),
        message("next group", 3, -100, 20),
    )
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == 1
        assert db.query(TaskExecutionCommand).one().payload["message"] == "next group"
    assert any("not accepted" in text for _, _, text in make.sent)


@pytest.mark.asyncio
async def test_mixed_voice_audio_text_preserve_order(ingress, tmp_path):
    make, sessions, _ = ingress
    bot = make()
    mock_files(bot, tmp_path)
    await run(
        bot,
        message("before"),
        message(None, 2, voice=voice()),
        message("after", 3),
        message(
            None,
            4,
            caption="music",
            audio=types.Audio(
                file_id="music", file_unique_id="music-stable", duration=2
            ),
        ),
    )
    with sessions() as db:
        command = db.query(TaskExecutionCommand).one()
        assert command.payload["message"].startswith(
            "before\nspoken words\nafter\nmusic"
        )
        assert "music.ogg" in command.payload["execution_message"]
        task = db.query(Task).one()
        assert task.description.startswith(
            "before\nspoken words\nafter\nmusic\n\n[music.ogg]"
        )
        assert task.title == task.description[:50] + "..."
        assert len(command.payload["file_ids"]) == 2
        assert db.query(TaskInputReceipt).count() == 4
    assert bot._transcribe_uploaded_voice_files.await_args.args[0] == ["download"]


@pytest.mark.asyncio
async def test_recovery_creates_missing_loading(ingress):
    make, sessions, observe = ingress
    bot = make()
    incoming = await bot._shared_input(123, message("hello", 1, -100, 20))
    owner, _, _, _ = inputs.lookup_channel_inputs((incoming,))
    accepted = inputs.accept_channel_input(
        incoming,
        owner_id=owner,
        active_task_id=None,
        channel_name="test",
        payload=module.TaskTurnPayload("hello"),
        staged_files=(),
        host_id="ingress",
    )
    await observe.side_effect(SimpleNamespace(task_id=accepted.task_id))
    await accepted.as_turn().deliver(bot._deliver_shared_result)
    with sessions() as db:
        delivery = db.query(TaskChannelDelivery).one()
        assert delivery.destination["loading_message_id"] == 1001
        assert delivery.status == "delivered"
    assert make.sent[0][:2] == (-100, 20)


@pytest.mark.asyncio
async def test_new_selection_is_atomic_before_cleanup_await_and_discards_pending(
    ingress,
):
    make, sessions, observe = ingress
    bot = make()
    observe.side_effect = RuntimeError("not observing")
    await run(bot, message())
    old = bot.active_tasks[123]
    bot.selected_agents[123] = 7
    original = bot._discard_abandoned_results

    async def discard(user, task):
        assert bot.selected_agents[user] == 9
        assert bot.active_tasks[user] != old
        return await original(user, task)

    bot._discard_abandoned_results = discard
    persisted, cleaned = await bot._reset_conversation(123, 9)
    assert persisted and cleaned
    with sessions() as db:
        assert db.query(TaskChannelDelivery).one().status == "discarded"


@pytest.mark.asyncio
async def test_expired_agent_callbacks_do_not_change_selection(ingress):
    make, _, _ = ingress
    bot = make()
    bot.active_tasks[123] = 10
    bot.selected_agents[123] = 7
    for data in ["agsel:default", "agsel:old:default", "agpage:old:0"]:
        callback = SimpleNamespace(
            data=data, from_user=SimpleNamespace(id=123), answer=AsyncMock()
        )
        if data.startswith("agsel"):
            await bot._handle_agent_selection_callback(callback)
        else:
            await bot._handle_agents_page_callback(callback)
        assert "expired" in callback.answer.await_args.args[0]
    assert bot.active_tasks[123] == 10
    assert bot.selected_agents[123] == 7
    callback = SimpleNamespace(data="agsel:current:default", answer=AsyncMock())
    assert await bot._callback_payload(callback, "agsel:") == "default"


@pytest.mark.asyncio
async def test_old_controls_filtered_but_ordinary_messages_retained(ingress):
    make, _, _ = ingress
    bot = make()
    bot._started_at = datetime.now(timezone.utc)
    old = bot._started_at - timedelta(seconds=1)
    bot._enqueue_user_message = Mock()
    # Registered handlers are invoked at their routing boundary.
    handlers = {h.callback.__name__: h.callback for h in bot.dp.message.handlers}
    for name in [
        "cmd_start",
        "cmd_help",
        "cmd_new",
        "cmd_list",
        "cmd_switch",
        "cmd_agents",
        "cmd_stop",
    ]:
        await handlers[name](message("/" + name[4:], date=old))
    await handlers["handle_message"](message("stop", date=old))
    assert not make.sent
    await handlers["handle_message"](message("ordinary", date=old))
    bot._enqueue_user_message.assert_called_once()
    assert bot._stale_control(message(date=bot._started_at.replace(microsecond=0)))
    assert not bot._stale_control(message(date=bot._started_at + timedelta(seconds=1)))


@pytest.mark.asyncio
@pytest.mark.parametrize("control", [None, "stop", "new"])
async def test_cancel_during_commit_drains_and_honors_explicit_controls(
    ingress, monkeypatch, control
):
    import threading

    make, sessions, _ = ingress
    bot = make()
    entered, release = threading.Event(), threading.Event()
    original = module.accept_channel_input

    def accept(*args, **kwargs):
        accepted = original(*args, **kwargs)
        entered.set()
        assert release.wait(10)
        return accepted

    monkeypatch.setattr(module, "accept_channel_input", accept)
    pending = asyncio.create_task(run(bot, message()))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        if control == "new":
            await bot._reset_conversation(123)
        elif control == "stop":
            bot._stop_current_conversation(123)
        pending.cancel()
        await asyncio.sleep(0)
        assert not pending.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == 1
        assert {c.kind for c in db.query(TaskExecutionCommand)} == (
            {"start", "pause"} if control else {"start"}
        )
        assert db.query(TaskChannelDelivery).one().status == (
            "discarded" if control == "new" else "pending"
        )
    assert bot.active_tasks[123] == (-1 if control == "new" else 1)
    assert not bot.user_preparing_executions


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["stop", "new"])
async def test_control_during_file_preparation_leaves_no_acceptance(
    ingress, tmp_path, control
):
    make, sessions, _ = ingress
    bot = make()
    mock_files(bot, tmp_path)
    original = bot._download_telegram_file.side_effect
    entered, release = asyncio.Event(), asyncio.Event()

    async def download(*args):
        file = await original(*args)
        entered.set()
        await release.wait()
        return file

    bot._download_telegram_file.side_effect = download
    pending = asyncio.create_task(run(bot, message(None, voice=voice())))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        if control == "new":
            await bot._reset_conversation(123)
        else:
            bot._stop_current_conversation(123)
    finally:
        release.set()
    await pending
    with sessions() as db:
        assert db.query(Task).count() == 0
        assert db.query(TaskInputReceipt).count() == 0
    bot._close_voice_asr_model.assert_awaited_once()
    assert not make.sent


@pytest.mark.asyncio
async def test_progress_uses_durable_loading_and_html_fallback(ingress):
    make, sessions, observe = ingress
    bot = make()
    complete = observe.side_effect

    async def progress(handler):
        types.Message.edit_text.side_effect = [RuntimeError("HTML invalid"), None, None]
        await handler._update_message("partial answer")
        with sessions() as db:
            assert (
                db.query(TaskChannelDelivery).one().destination["loading_message_id"]
                == 1001
            )
        return await complete(handler)

    observe.side_effect = progress
    await run(bot, message())
    calls = types.Message.edit_text.await_args_list
    assert "partial answer" in calls[0].args[0]
    assert calls[1].kwargs["parse_mode"] is None


@pytest.mark.asyncio
async def test_replay_reauthorizes(ingress):
    make, sessions, observe = ingress
    bot = make()
    await run(bot, message())
    with sessions() as db:
        channel = db.get(UserChannel, bot.channel_id)
        channel.config = {"allowed_users": ["someone_else"]}
        db.commit()
    await run(bot, message())
    observe.assert_awaited_once()
    assert "not authorized" in make.sent[-1][2]


@pytest.mark.asyncio
@pytest.mark.parametrize("shared", [True, False])
async def test_startup_retains_only_shared_pending_updates(monkeypatch, shared):
    from tests.web.test_telegram_message_queue import make_bot

    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", str(shared).lower())
    bot = make_bot()
    bot.instance_id = "test"
    bot.bot = SimpleNamespace(delete_webhook=AsyncMock(), set_my_commands=AsyncMock())
    bot.dp = SimpleNamespace(start_polling=AsyncMock())
    bot._menu_generation = "previous"
    await bot.start()
    bot.bot.delete_webhook.assert_awaited_once_with(drop_pending_updates=not shared)
    bot.dp.start_polling.assert_awaited_once()
    assert bot._started_at is not None
    assert bot._menu_generation != "previous"


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["stop", "new"])
async def test_pending_notice_keeps_worker_controllable(ingress, control):
    make, sessions, observe = ingress
    bot = make()
    observe.side_effect = None
    observe.return_value = {"status": "accepted"}
    await run(bot, message())
    task_id, turn = bot.user_active_executions[123]
    assert "still being processed" in str(types.Message.edit_text.await_args_list)
    if control == "new":
        await bot._reset_conversation(123)
    else:
        assert bot._stop_current_conversation(123)
    await turn.stop_task
    with sessions() as db:
        pause = db.query(TaskExecutionCommand).filter_by(kind="pause").one()
        assert pause.task_id == task_id
        assert pause.target_run_id == turn.run_id
        assert db.query(TaskChannelDelivery).one().status == (
            "discarded" if control == "new" else "pending"
        )


@pytest.mark.asyncio
async def test_historical_replay_cannot_replace_newer_pending_handle(
    ingress, monkeypatch
):
    make, sessions, observe = ingress
    bot = make()
    await run(bot, message("first", 1))
    observe.side_effect = None
    observe.return_value = {"status": "accepted"}
    await run(bot, message("second", 2))
    newer = bot.user_active_executions[123]
    entered, release = asyncio.Event(), asyncio.Event()

    async def deliver(*args, **kwargs):
        entered.set()
        await release.wait()
        return False

    monkeypatch.setattr(shared_channel_execution.SharedChannelTurn, "deliver", deliver)
    replay = asyncio.create_task(run(bot, message("first", 1)))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert bot.user_active_executions[123] == newer
        bot._stop_current_conversation(123)
        await newer[1].stop_task
    finally:
        release.set()
    await replay
    with sessions() as db:
        assert (
            db.query(TaskExecutionCommand).filter_by(kind="pause").one().target_run_id
            == newer[1].run_id
        )


@pytest.mark.asyncio
async def test_reconstructed_delivery_routes_real_sdk_sends_to_topic():
    from aiogram.methods import SendMessage

    from tests.web.test_telegram_message_queue import make_bot

    bot = make_bot()
    bot.bot = AsyncMock(return_value=message(identity=99, chat=-100, topic=20))
    delivery = channel_delivery.ChannelDelivery(
        1,
        1,
        1,
        {
            "chat_id": -100,
            "message_thread_id": 20,
            "message_id": 1,
            "loading_message_id": None,
            "telegram_user_id": 123,
        },
        "claim",
        "123",
    )
    await bot._deliver_shared_result(
        delivery, {"success": True, "status": "completed", "output": "x" * 4100}
    )
    sends = [
        call.args[0]
        for call in bot.bot.await_args_list
        if isinstance(call.args[0], SendMessage)
    ]
    assert len(sends) == 2  # loading and overflow chunk
    assert all(send.chat_id == -100 and send.message_thread_id == 20 for send in sends)
    reply = bot._shared_reply_message(delivery)
    assert reply.answer_document("file-id").message_thread_id == 20
    assert reply.answer_photo("photo-id").message_thread_id == 20


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "control,cancel", [(None, True), ("stop", False), ("stop", True), ("new", True)]
)
async def test_controls_during_replay_lookup_settle_accepted_work(
    ingress, monkeypatch, control, cancel
):
    import threading

    make, sessions, observe = ingress
    first = make()
    observe.side_effect = RuntimeError("observer offline")
    await run(first, message())
    bot = make()  # Restart: durable acceptance exists but no control handle.
    entered, release = threading.Event(), threading.Event()
    lookup = module.lookup_channel_inputs

    def held(incoming):
        result = lookup(incoming)
        entered.set()
        assert release.wait(10)
        return result

    monkeypatch.setattr(module, "lookup_channel_inputs", held)
    task = asyncio.create_task(run(bot, message()))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        if control == "stop":
            assert bot._stop_current_conversation(123)
        elif control == "new":
            await bot._reset_conversation(123)
        if cancel:
            task.cancel()
    finally:
        release.set()
    if cancel:
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        await task
    with sessions() as db:
        assert db.query(TaskExecutionCommand).filter_by(kind="pause").count() == (
            1 if control else 0
        )
        assert db.query(TaskChannelDelivery).one().status == (
            "discarded" if control == "new" else "pending"
        )
    assert not bot.user_preparing_executions


@pytest.mark.asyncio
async def test_stop_during_conflict_notice_pauses_unobserved_replay(
    ingress, monkeypatch
):
    make, sessions, observe = ingress
    first = make()
    await run(first, message("first", 1))
    observe.side_effect = RuntimeError("observer offline")
    await run(first, message("second", 2))
    bot = make()
    entered, release = asyncio.Event(), asyncio.Event()
    answer = types.Message.answer

    async def held(self, text, **kwargs):
        if "different content" in text:
            entered.set()
            await release.wait()
        return await answer(self, text, **kwargs)

    monkeypatch.setattr(types.Message, "answer", held)
    task = asyncio.create_task(run(bot, message("changed", 1), message("second", 2)))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert bot._stop_current_conversation(123)
    finally:
        release.set()
    await task
    with sessions() as db:
        start = (
            db.query(TaskExecutionCommand)
            .filter_by(kind="start")
            .order_by(TaskExecutionCommand.id.desc())
            .first()
        )
        assert (
            db.query(TaskExecutionCommand).filter_by(kind="pause").one().target_run_id
            == start.target_run_id
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("notice", ["save", "agent"])
async def test_acceptance_notice_is_not_lost_when_worker_finishes_early(
    ingress, monkeypatch, notice
):
    make, sessions, _ = ingress
    bot = make()
    if notice == "save":
        bot._save_active_tasks = Mock(return_value=False)
    else:
        bot.selected_agents[123] = 123456
    accept = module.accept_channel_input

    def completed(*args, **kwargs):
        accepted = accept(*args, **kwargs)
        with sessions() as db:
            db.get(Task, accepted.task_id).status = TaskStatus.COMPLETED
            command = db.get(TaskExecutionCommand, accepted.command_db_id)
            command.status = "completed"
            command.result = {
                "channel_result": {
                    "success": True,
                    "status": "completed",
                    "output": "fast answer",
                }
            }
            db.commit()
        return accepted

    monkeypatch.setattr(module, "accept_channel_input", completed)
    await run(bot, message())
    expected = (
        "request was accepted"
        if notice == "save"
        else "selected agent is no longer available"
    )
    assert any(expected in text for _, _, text in make.sent)
    assert "fast answer" in str(types.Message.edit_text.await_args_list)


@pytest.mark.asyncio
@pytest.mark.parametrize("previous", ["new_sentinel", "older_task"])
async def test_stale_saved_selection_cannot_discard_accepted_reply(ingress, previous):
    make, sessions, observe = ingress
    bot = make()
    complete = observe.side_effect
    if previous == "new_sentinel":
        await bot._reset_conversation(123)
    else:
        await run(bot, message("older task", 99))
        # A missing selected Agent causes a new default task, preserving the
        # prior task on disk when the following activation save fails.
        bot.selected_agents[123] = 123456
    saved_selection = dict(bot.active_tasks)
    bot._save_active_tasks = Mock(return_value=False)
    observe.side_effect = RuntimeError("observer offline")
    await run(bot, message())
    accepted_task = bot.active_tasks[123]
    restarted = make()
    assert restarted.active_tasks == saved_selection
    await run(restarted, message())
    assert restarted.active_tasks == saved_selection
    with sessions() as db:
        assert (
            db.query(TaskChannelDelivery)
            .join(
                TaskExecutionCommand,
                TaskChannelDelivery.command_id == TaskExecutionCommand.id,
            )
            .filter(TaskExecutionCommand.task_id == accepted_task)
            .one()
            .status
            == "pending"
        )
        assert (
            db.query(TaskExecutionCommand)
            .filter_by(task_id=accepted_task, kind="start")
            .count()
            == 1
        )
    await complete(SimpleNamespace(task_id=accepted_task))
    await channel_delivery.recover_channel_results(
        restarted.channel_id, restarted._deliver_shared_result
    )
    with sessions() as db:
        assert (
            db.query(TaskChannelDelivery)
            .join(
                TaskExecutionCommand,
                TaskChannelDelivery.command_id == TaskExecutionCommand.id,
            )
            .filter(TaskExecutionCommand.task_id == accepted_task)
            .one()
            .status
            == "delivered"
        )


@pytest.mark.asyncio
async def test_explicit_new_still_discards_replayed_old_reply(ingress):
    make, sessions, observe = ingress
    bot = make()
    complete = observe.side_effect
    observe.side_effect = RuntimeError("observer offline")
    await run(bot, message())
    accepted_task = bot.active_tasks[123]
    _, turn = bot.user_active_executions[123]
    await bot._reset_conversation(123)
    await turn.stop_task
    restarted = make()
    await run(restarted, message())
    await complete(SimpleNamespace(task_id=accepted_task))
    types.Message.edit_text.reset_mock()
    await channel_delivery.recover_channel_results(
        restarted.channel_id, restarted._deliver_shared_result
    )
    with sessions() as db:
        assert db.query(TaskChannelDelivery).one().status == "discarded"
    types.Message.edit_text.assert_not_awaited()
