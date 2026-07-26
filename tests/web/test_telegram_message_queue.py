import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiogram import types

from xagent.web.channels.telegram.bot import (
    TelegramBotInstance,
    TelegramChannelManager,
    TelegramVoiceTranscriptionError,
)
from xagent.web.models.task import TaskStatus
from xagent.web.services.channel_runtime import ChannelOwnerSnapshot
from xagent.web.services.task_execution_context_service import (
    TaskExecutionRecoverySnapshot,
)
from xagent.web.services.task_lease_service import TaskLease


def make_bot() -> TelegramBotInstance:
    bot = object.__new__(TelegramBotInstance)
    bot._accepting = True
    bot._ingress_stopped = False
    bot._stop_lock = None
    bot._stop_loop = None
    bot.user_message_queues = {}
    bot.user_message_tasks = {}
    bot.user_active_executions = {}
    bot.user_preparing_executions = set()
    bot.user_stop_events = {}
    return bot


@pytest.mark.asyncio
async def test_extract_message_content_includes_voice_message() -> None:
    bot = make_bot()
    voice = SimpleNamespace(file_id="voice-file-id")
    message = SimpleNamespace(
        text=None,
        caption=None,
        document=None,
        photo=None,
        audio=None,
        voice=voice,
        video=None,
    )

    text, files = await bot._extract_message_content(message)  # type: ignore[arg-type]

    assert text == ""
    assert files == [voice]


@pytest.mark.asyncio
async def test_extract_message_content_keeps_regular_audio_as_attachment() -> None:
    bot = make_bot()
    audio = SimpleNamespace(file_id="audio-file-id")
    message = SimpleNamespace(
        text=None,
        caption=None,
        document=None,
        photo=None,
        audio=audio,
        voice=None,
        video=None,
    )

    text, files = await bot._extract_message_content(message)  # type: ignore[arg-type]

    assert text == ""
    assert files == [audio]


def test_compose_prompt_text_replaces_voice_with_transcript_in_message_order() -> None:
    voice = SimpleNamespace(file_id="voice-file-id")
    voice_message = SimpleNamespace(voice=voice)
    text_message = SimpleNamespace(voice=None)

    prompt = TelegramBotInstance._compose_prompt_text(
        [
            (voice_message, "", [voice]),
            (text_message, "请直接回答这个问题", []),
        ],
        {"voice-file-id": "今晚有世界杯比赛吗？"},
    )

    assert prompt == "今晚有世界杯比赛吗？\n请直接回答这个问题"


@pytest.mark.parametrize(
    ("file_info", "expected"),
    [
        ({"name": "voice.oga", "type": "audio/ogg"}, "ogg"),
        ({"name": "voice.mp3", "type": "audio/mpeg"}, "mp3"),
        ({"name": "voice.opus", "type": ""}, "ogg"),
    ],
)
def test_audio_format_from_file_info(file_info: dict[str, str], expected: str) -> None:
    assert TelegramBotInstance._audio_format_from_file_info(file_info) == expected


@pytest.mark.parametrize(
    ("telegram_file", "target_path", "expected"),
    [
        (
            types.Voice(
                file_id="voice-id",
                file_unique_id="voice-unique-id",
                duration=1,
                mime_type="audio/ogg",
            ),
            Path("voice.bin"),
            "audio/ogg",
        ),
        (
            SimpleNamespace(mime_type="application/pdf"),
            Path("report.jpg"),
            "image/jpeg",
        ),
        (
            SimpleNamespace(mime_type="application/pdf"),
            Path("report.bin"),
            "application/pdf",
        ),
    ],
)
def test_mime_type_for_telegram_file(
    telegram_file: object,
    target_path: Path,
    expected: str,
) -> None:
    assert (
        TelegramBotInstance._mime_type_for_telegram_file(
            telegram_file,
            target_path,
        )
        == expected
    )


def test_display_message_for_user_hides_runtime_file_links() -> None:
    assert (
        TelegramBotInstance._display_message_for_user(
            "Please summarize this document",
            has_files=True,
        )
        == "Please summarize this document"
    )
    assert (
        TelegramBotInstance._display_message_for_user("", has_files=True)
        == "Uploaded file(s)"
    )


@pytest.mark.asyncio
async def test_transcribe_uploaded_voice_files_uses_registered_input_file() -> None:
    bot = make_bot()

    class FakeASR:
        def __init__(self) -> None:
            self.calls: list[dict[str, str | None]] = []

        async def transcribe(self, *, audio: str, format: str | None = None) -> str:
            self.calls.append({"audio": audio, "format": format})
            return "今晚有世界杯比赛吗？"

    asr = FakeASR()
    uploaded_info = [
        {
            "file_id": "workspace-file-id",
            "telegram_file_id": "voice-file-id",
            "name": "voice-file-id.oga",
            "path": "/workspace/input/voice-file-id.oga",
            "type": "audio/ogg",
            "size": 123,
        }
    ]

    transcripts = await bot._transcribe_uploaded_voice_files(
        ["voice-file-id"],
        uploaded_info,
        asr,
    )

    assert transcripts == {"voice-file-id": "今晚有世界杯比赛吗？"}
    assert asr.calls == [
        {"audio": "/workspace/input/voice-file-id.oga", "format": "ogg"}
    ]
    assert uploaded_info[0]["file_id"] == "workspace-file-id"


@pytest.mark.asyncio
async def test_transcribe_uploaded_voice_files_extracts_result_text() -> None:
    bot = make_bot()

    class ResultASR:
        async def transcribe(
            self, *, audio: str, format: str | None = None
        ) -> SimpleNamespace:
            return SimpleNamespace(text="今晚有世界杯比赛吗？")

    transcripts = await bot._transcribe_uploaded_voice_files(
        ["voice-file-id"],
        [
            {
                "telegram_file_id": "voice-file-id",
                "name": "voice.oga",
                "path": "/workspace/input/voice.oga",
                "type": "audio/ogg",
            }
        ],
        ResultASR(),
    )

    assert transcripts == {"voice-file-id": "今晚有世界杯比赛吗？"}


@pytest.mark.asyncio
async def test_close_voice_asr_model_supports_sync_close() -> None:
    class SyncClosableASR:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    asr = SyncClosableASR()

    await TelegramBotInstance._close_voice_asr_model(asr)

    assert asr.closed is True


@pytest.mark.asyncio
async def test_transcribe_uploaded_voice_files_rejects_empty_result() -> None:
    bot = make_bot()

    class EmptyASR:
        async def transcribe(self, *, audio: str, format: str | None = None) -> str:
            return "  "

    with pytest.raises(
        TelegramVoiceTranscriptionError,
        match="returned empty text",
    ):
        await bot._transcribe_uploaded_voice_files(
            ["voice-file-id"],
            [
                {
                    "telegram_file_id": "voice-file-id",
                    "name": "voice.oga",
                    "path": "/workspace/input/voice.oga",
                    "type": "audio/ogg",
                }
            ],
            EmptyASR(),
        )


def test_resolve_voice_asr_model_returns_none_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi import HTTPException

    bot = make_bot()

    def no_asr_model(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise HTTPException(status_code=404, detail="No ASR model is configured")

    monkeypatch.setattr(
        "xagent.web.api.model._resolve_asr_model_for_transcription",
        no_asr_model,
    )

    assert bot._resolve_voice_asr_model(object(), object()) is None  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_voice_without_asr_is_rejected_before_channel_task_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.channels.telegram import bot as telegram_bot_module

    bot = make_bot()
    bot.channel_id = 1
    bot.channel_name = "Telegram voice preflight"
    bot.active_tasks = {}
    bot._save_active_tasks = lambda: None
    bot._consume_user_stop_request = lambda _user_id: False
    bot._clear_user_stop_request = lambda _user_id: None
    voice = SimpleNamespace(file_id="voice-id")

    async def extract_voice(_message):  # type: ignore[no-untyped-def]
        return "", [voice]

    async def authorize(**_kwargs) -> ChannelOwnerSnapshot:  # type: ignore[no-untyped-def]
        return ChannelOwnerSnapshot(user_id=5)

    prepare_calls = 0

    async def prepare(**_kwargs):  # type: ignore[no-untyped-def]
        nonlocal prepare_calls
        prepare_calls += 1
        raise AssertionError("voice preflight must run before task creation")

    monkeypatch.setattr(
        telegram_bot_module,
        "authorize_channel_sender",
        authorize,
        raising=False,
    )
    monkeypatch.setattr(telegram_bot_module, "prepare_channel_task", prepare)
    bot._resolve_voice_asr_model_isolated = lambda _user_id: None
    bot._extract_message_content = extract_voice

    class Message:
        def __init__(self) -> None:
            self.voice = voice
            self.chat = SimpleNamespace(id=456)
            self.answers: list[str] = []

        async def answer(self, text: str, **_kwargs) -> None:
            self.answers.append(text)

    message = Message()
    await bot._process_user_messages_batch(123, [message])  # type: ignore[arg-type]

    assert prepare_calls == 0
    assert len(message.answers) == 1
    assert "no speech recognition model is configured" in message.answers[0]


@pytest.mark.asyncio
async def test_stop_after_prepare_settles_preclaimed_task_instead_of_orphaning_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bot = make_bot()
    bot.channel_id = 1
    bot.channel_name = "Telegram stop after prepare"
    bot.active_tasks = {}
    bot._save_active_tasks = lambda: None
    bot._clear_user_stop_request = lambda _user_id: None
    stop_checks = 0

    def consume_stop(_user_id: int) -> bool:
        nonlocal stop_checks
        stop_checks += 1
        return stop_checks == 1

    bot._consume_user_stop_request = consume_stop
    lease = TaskLease(task_id=44, runner_id="runner-a", run_id="run-a")
    finalized: list[TaskStatus] = []

    class FakeManagedLease:
        heartbeat_task = None

        def __init__(self) -> None:
            self.lease = lease
            self.closed = False

        async def finalize_result(self, *, status: TaskStatus, **_kwargs) -> bool:
            finalized.append(status)
            return True

        async def close(self) -> bool:
            self.closed = True
            return True

    managed = FakeManagedLease()

    async def extract_text(_message):  # type: ignore[no-untyped-def]
        return "hello", []

    async def prepare(**_kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(
            user_id=5,
            task_id=44,
            is_new_task=True,
            managed_lease=managed,
        )

    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.prepare_channel_task",
        prepare,
    )
    bot._extract_message_content = extract_text

    class Message:
        voice = None
        chat = SimpleNamespace(id=456)

        async def answer(self, _text: str, **_kwargs) -> None:
            return None

    await bot._process_user_messages_batch(123, [Message()])  # type: ignore[arg-type]

    assert finalized == [TaskStatus.PAUSED]
    assert managed.closed is True


@pytest.mark.asyncio
async def test_process_user_queue_drains_messages_added_while_batch_runs() -> None:
    bot = make_bot()
    bot.queue_flush_delay_seconds = 0
    bot.user_message_queues = {123: ["first"]}

    processed_batches: list[list[str]] = []

    async def fake_process_batch(user_id: int, messages: list[str]) -> None:
        processed_batches.append(list(messages))
        if len(processed_batches) == 1:
            bot.user_message_queues.setdefault(user_id, []).append("second")

    bot._process_user_messages_batch = fake_process_batch

    queue_task = asyncio.create_task(bot._process_user_queue(123))
    bot.user_message_tasks[123] = queue_task

    await queue_task

    assert processed_batches == [["first"], ["second"]]
    assert bot.user_message_tasks == {}
    assert bot.user_message_queues == {}


@pytest.mark.asyncio
async def test_process_user_queue_drains_message_added_while_unregistering() -> None:
    bot = make_bot()
    bot.queue_flush_delay_seconds = 0
    bot.user_message_queues = {123: ["first"]}

    class RaceTaskDict(dict):
        def __init__(self, user_id: int) -> None:
            super().__init__()
            self.user_id = user_id
            self.injected = False

        def pop(self, key, default=None):  # type: ignore[no-untyped-def]
            value = super().pop(key, default)
            if key == self.user_id and not self.injected:
                self.injected = True
                bot.user_message_queues.setdefault(key, []).append("second")
            return value

    bot.user_message_tasks = RaceTaskDict(123)
    processed_batches: list[list[str]] = []

    async def fake_process_batch(user_id: int, messages: list[str]) -> None:
        processed_batches.append(list(messages))

    bot._process_user_messages_batch = fake_process_batch

    queue_task = asyncio.create_task(bot._process_user_queue(123))
    bot.user_message_tasks[123] = queue_task

    await queue_task

    assert processed_batches == [["first"], ["second"]]
    assert bot.user_message_tasks == {}
    assert bot.user_message_queues == {}


def test_start_new_conversation_clears_queue_and_pauses_active_execution() -> None:
    bot = make_bot()
    bot.user_message_queues = {123: ["old queued message"]}
    bot.active_tasks = {123: 456}
    bot.saved = False

    class FakeAgentService:
        def __init__(self) -> None:
            self.pause_calls: list[tuple[str, str | None]] = []

        def pause_execution_by_id(
            self, execution_id: str, reason: str | None = None
        ) -> bool:
            self.pause_calls.append((execution_id, reason))
            return True

    agent_service = FakeAgentService()
    bot.user_active_executions = {123: (456, agent_service)}

    def fake_save_active_tasks() -> None:
        bot.saved = True

    bot._save_active_tasks = fake_save_active_tasks

    assert bot._start_new_conversation(123) is True
    assert 123 not in bot.user_message_queues
    assert bot.active_tasks[123] == -1
    assert bot.saved is True
    assert agent_service.pause_calls == [("456", "new Telegram conversation requested")]


def test_stop_current_conversation_preserves_active_task() -> None:
    bot = make_bot()
    bot.user_message_queues = {123: ["old queued message"]}
    bot.active_tasks = {123: 456}
    bot.saved = False

    class FakeAgentService:
        def __init__(self) -> None:
            self.pause_calls: list[tuple[str, str | None]] = []

        def pause_execution_by_id(
            self, execution_id: str, reason: str | None = None
        ) -> bool:
            self.pause_calls.append((execution_id, reason))
            return True

    agent_service = FakeAgentService()
    bot.user_active_executions = {123: (456, agent_service)}

    def fake_save_active_tasks() -> None:
        bot.saved = True

    bot._save_active_tasks = fake_save_active_tasks

    assert bot._stop_current_conversation(123) is True
    assert 123 not in bot.user_message_queues
    assert bot.active_tasks[123] == 456
    assert bot.saved is False
    assert agent_service.pause_calls == [("456", "Telegram stop requested")]


def test_stop_current_conversation_clears_pending_queue_without_active_run() -> None:
    bot = make_bot()
    bot.user_message_queues = {123: ["queued before execution"]}
    bot.active_tasks = {123: 456}

    assert bot._stop_current_conversation(123) is True
    assert bot.user_message_queues == {}
    assert bot.active_tasks[123] == 456


def test_stop_current_conversation_records_stop_during_preparation() -> None:
    bot = make_bot()
    bot.active_tasks = {123: 456}
    bot.user_preparing_executions.add(123)

    assert bot._stop_current_conversation(123) is True
    assert bot.user_stop_events[123].is_set()
    assert bot.active_tasks[123] == 456


@pytest.mark.asyncio
async def test_await_execution_with_stop_monitor_pauses_pending_stop() -> None:
    bot = make_bot()

    class FakeAgentService:
        def __init__(self) -> None:
            self.pause_calls: list[tuple[str, str | None]] = []

        def pause_execution_by_id(
            self, execution_id: str, reason: str | None = None
        ) -> bool:
            self.pause_calls.append((execution_id, reason))
            return True

    agent_service = FakeAgentService()
    bot.user_active_executions = {123: (456, agent_service)}
    bot._request_user_stop(123)

    async def fake_execution() -> dict:
        await asyncio.sleep(0)
        return {"status": "interrupted"}

    result = await bot._await_execution_with_stop_monitor(
        123,
        fake_execution(),
        reason="Telegram stop requested",
    )

    assert result == {"status": "interrupted"}
    assert agent_service.pause_calls == [("456", "Telegram stop requested")]
    assert not bot.user_stop_events[123].is_set()


@pytest.mark.asyncio
async def test_stop_monitor_drains_execution_cleanup_before_cancellation() -> None:
    bot = make_bot()
    execution_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()

    async def fake_execution() -> dict:
        execution_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await allow_cleanup.wait()
            cleanup_finished.set()

    monitor = asyncio.create_task(
        bot._await_execution_with_stop_monitor(
            123,
            fake_execution(),
            reason="Telegram stop requested",
        )
    )
    await execution_started.wait()
    monitor.cancel()
    await cleanup_started.wait()
    await asyncio.sleep(0)

    assert not monitor.done()
    assert not cleanup_finished.is_set()

    allow_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(monitor, timeout=1)
    assert cleanup_finished.is_set()


class _FakeTelegramDispatcher:
    def __init__(
        self,
        *,
        stop_started: asyncio.Event | None = None,
        allow_stop: asyncio.Event | None = None,
    ) -> None:
        self.stop_started = stop_started
        self.allow_stop = allow_stop
        self.stop_calls = 0

    async def stop_polling(self) -> None:
        self.stop_calls += 1
        if self.stop_started is not None:
            self.stop_started.set()
        if self.allow_stop is not None:
            await self.allow_stop.wait()


class _FakeTelegramSession:
    def __init__(self) -> None:
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


def _attach_telegram_transport(
    bot: TelegramBotInstance,
    *,
    stop_started: asyncio.Event | None = None,
    allow_stop: asyncio.Event | None = None,
) -> tuple[_FakeTelegramDispatcher, _FakeTelegramSession]:
    dispatcher = _FakeTelegramDispatcher(
        stop_started=stop_started,
        allow_stop=allow_stop,
    )
    session = _FakeTelegramSession()
    bot.dp = dispatcher  # type: ignore[assignment]
    bot.bot = SimpleNamespace(session=session)  # type: ignore[assignment]
    return dispatcher, session


@pytest.mark.asyncio
async def test_stop_drains_each_owned_turn_once_before_clearing_runtime_state() -> None:
    bot = make_bot()
    dispatcher, session = _attach_telegram_transport(bot)
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()

    async def running_turn() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await allow_cleanup.wait()
            cleanup_finished.set()

    turn_task = asyncio.create_task(running_turn())
    await asyncio.sleep(0)
    bot.user_message_tasks = {1: turn_task, 2: turn_task}
    bot.user_message_queues = {1: ["queued"]}
    bot.user_active_executions = {1: (42, object())}
    bot.user_preparing_executions = {2}
    bot.user_stop_events = {1: asyncio.Event()}

    stop_task = asyncio.create_task(bot.stop())
    try:
        await asyncio.wait_for(cleanup_started.wait(), timeout=0.5)
        await asyncio.sleep(0)

        assert turn_task.cancelling() == 1
        assert not stop_task.done()
        assert bot.user_message_tasks == {1: turn_task, 2: turn_task}
        assert bot.user_message_queues == {1: ["queued"]}
        assert bot.user_active_executions
        assert bot.user_preparing_executions
        assert bot.user_stop_events

        allow_cleanup.set()
        await asyncio.wait_for(stop_task, timeout=1)

        assert cleanup_finished.is_set()
        assert bot.user_message_tasks == {}
        assert bot.user_message_queues == {}
        assert bot.user_active_executions == {}
        assert bot.user_preparing_executions == set()
        assert bot.user_stop_events == {}
        assert dispatcher.stop_calls == 1
        assert session.close_calls == 1
    finally:
        allow_cleanup.set()
        if not turn_task.done():
            turn_task.cancel()
        await asyncio.gather(turn_task, return_exceptions=True)
        if not stop_task.done():
            stop_task.cancel()
        await asyncio.gather(stop_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_stop_cancellation_waits_for_turn_cleanup_before_propagating() -> None:
    bot = make_bot()
    _attach_telegram_transport(bot)
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()

    async def running_turn() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await allow_cleanup.wait()

    turn_task = asyncio.create_task(running_turn())
    await asyncio.sleep(0)
    bot.user_message_tasks = {1: turn_task}
    bot.user_message_queues = {1: ["queued"]}

    stop_task = asyncio.create_task(bot.stop())
    try:
        await asyncio.wait_for(cleanup_started.wait(), timeout=0.5)
        stop_task.cancel()
        await asyncio.sleep(0)

        assert not stop_task.done()
        assert bot.user_message_tasks == {1: turn_task}
        assert bot.user_message_queues == {1: ["queued"]}

        allow_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(stop_task, timeout=1)

        assert turn_task.done()
        assert bot.user_message_tasks == {}
        assert bot.user_message_queues == {}
    finally:
        allow_cleanup.set()
        if not turn_task.done():
            turn_task.cancel()
        await asyncio.gather(turn_task, return_exceptions=True)
        if not stop_task.done():
            stop_task.cancel()
        await asyncio.gather(stop_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_stop_fence_rejects_late_telegram_ingress() -> None:
    bot = make_bot()
    stop_started = asyncio.Event()
    allow_stop = asyncio.Event()
    _attach_telegram_transport(
        bot,
        stop_started=stop_started,
        allow_stop=allow_stop,
    )

    stop_task = asyncio.create_task(bot.stop())
    try:
        await asyncio.wait_for(stop_started.wait(), timeout=0.5)

        assert bot._enqueue_user_message(1, "late") is False
        assert bot.user_message_queues == {}
        assert bot.user_message_tasks == {}
    finally:
        allow_stop.set()
        await asyncio.gather(stop_task, return_exceptions=True)


def test_stop_is_concurrent_and_cross_loop_idempotent() -> None:
    bot = make_bot()
    dispatcher, session = _attach_telegram_transport(bot)

    async def stop_concurrently() -> None:
        await asyncio.gather(bot.stop(), bot.stop())

    asyncio.run(stop_concurrently())
    asyncio.run(bot.stop())

    assert dispatcher.stop_calls == 1
    assert session.close_calls == 1


@pytest.mark.asyncio
async def test_manager_stop_cancellation_drains_telegram_polling_cleanup() -> None:
    manager = TelegramChannelManager()
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()

    async def polling() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await allow_cleanup.wait()
            cleanup_finished.set()

    polling_task = asyncio.create_task(polling())
    await asyncio.sleep(0)

    async def stop_bot() -> None:
        return None

    bot = SimpleNamespace(
        instance_id="telegram-instance",
        polling_task=polling_task,
        stop=stop_bot,
    )
    manager.bots["token"] = bot

    stop_task = asyncio.create_task(manager._stop_bot_for_token("token"))
    try:
        await asyncio.wait_for(cleanup_started.wait(), timeout=0.5)
        stop_task.cancel()
        await asyncio.sleep(0)

        assert not stop_task.done()
        assert "token" in manager.bots
        assert not cleanup_finished.is_set()

        allow_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(stop_task, timeout=1)

        assert cleanup_finished.is_set()
        assert polling_task.done()
        assert "token" not in manager.bots
    finally:
        allow_cleanup.set()
        if not polling_task.done():
            polling_task.cancel()
        await asyncio.gather(polling_task, return_exceptions=True)
        if not stop_task.done():
            stop_task.cancel()
        await asyncio.gather(stop_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_manager_stop_is_single_flight_per_telegram_bot() -> None:
    manager = TelegramChannelManager()
    stop_started = asyncio.Event()
    allow_stop = asyncio.Event()
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()
    stop_calls = 0

    async def polling() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await allow_cleanup.wait()

    polling_task = asyncio.create_task(polling())
    await asyncio.sleep(0)

    async def stop_bot() -> None:
        nonlocal stop_calls
        stop_calls += 1
        stop_started.set()
        await allow_stop.wait()

    bot = SimpleNamespace(
        instance_id="telegram-instance",
        polling_task=polling_task,
        stop=stop_bot,
    )
    manager.bots["token"] = bot

    stop_tasks = [
        asyncio.create_task(manager._stop_bot_for_token("token")),
        asyncio.create_task(manager._stop_bot_for_token("token")),
    ]
    try:
        await asyncio.wait_for(stop_started.wait(), timeout=0.5)
        await asyncio.sleep(0)
        assert stop_calls == 1

        allow_stop.set()
        await asyncio.wait_for(cleanup_started.wait(), timeout=0.5)
        assert polling_task.cancelling() == 1

        allow_cleanup.set()
        await asyncio.wait_for(asyncio.gather(*stop_tasks), timeout=1)
        assert "token" not in manager.bots
    finally:
        allow_stop.set()
        allow_cleanup.set()
        if not polling_task.done():
            polling_task.cancel()
        await asyncio.gather(polling_task, return_exceptions=True)
        for stop_task in stop_tasks:
            if not stop_task.done():
                stop_task.cancel()
        await asyncio.gather(*stop_tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_channel_failure_suppresses_stale_error_after_exact_settlement_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bot = make_bot()
    bot.channel_id = 1
    bot.channel_name = "Telegram exact settlement"
    bot.active_tasks = {}
    bot.bot = object()
    bot._save_active_tasks = lambda: None
    bot._consume_user_stop_request = lambda _user_id: False
    bot._clear_user_stop_request = lambda _user_id: None

    lease = TaskLease(task_id=44, runner_id="runner-a", run_id="shared-run")

    class FakeManagedLease:
        heartbeat_task = None

        def __init__(self) -> None:
            self.lease = lease
            self.closed = False

        async def close(self) -> bool:
            self.closed = True
            return True

        async def finalize_result(
            self,
            *,
            status: TaskStatus,
            **_kwargs,
        ) -> bool:
            settlements.append((self.lease, status))
            return False

    managed = FakeManagedLease()
    settlements: list[tuple[TaskLease, TaskStatus]] = []

    class FakeTracer:
        def __init__(self) -> None:
            self.handlers: list[object] = []

        def add_handler(self, handler: object) -> None:
            self.handlers.append(handler)

    agent_service = SimpleNamespace(
        tracer=FakeTracer(),
        set_conversation_history=lambda _messages: None,
        set_execution_context_messages=lambda _messages: None,
        set_recovered_skill_context=lambda _context: None,
    )

    class FakeAgentManager:
        async def get_agent_for_task(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return agent_service

        async def execute_task(self, **_kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("channel execution failed")

    async def extract_text(_message):  # type: ignore[no-untyped-def]
        return "hello", []

    async def await_execution(_user_id, execution, *, reason):  # type: ignore[no-untyped-def]
        return await execution

    async def fake_prepare(**_kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(
            user_id=5,
            task_id=44,
            is_new_task=True,
            managed_lease=managed,
        )

    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.prepare_channel_task",
        fake_prepare,
    )
    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.load_task_setup_snapshot_sync",
        lambda *_args: SimpleNamespace(
            runtime_user=None,
            conversation_history=(),
            execution_recovery=TaskExecutionRecoverySnapshot(),
        ),
    )

    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.get_agent_manager",
        lambda: FakeAgentManager(),
    )

    async def persist_message(**_kwargs) -> None:  # type: ignore[no-untyped-def]
        return None

    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.persist_channel_user_message",
        persist_message,
    )
    bot._extract_message_content = extract_text
    bot._await_execution_with_stop_monitor = await_execution

    class LoadingMessage:
        message_id = 77

    class Message:
        from_user = SimpleNamespace(id=123)
        chat = SimpleNamespace(id=456)

        def __init__(self) -> None:
            self.answers: list[str] = []

        async def answer(self, text: str, **_kwargs) -> LoadingMessage:
            self.answers.append(text)
            return LoadingMessage()

    message = Message()
    await bot._process_user_messages_batch(123, [message])  # type: ignore[arg-type]

    assert settlements == [(lease, TaskStatus.FAILED)]
    assert managed.closed is True
    assert "Sorry, an error occurred while processing your request." not in (
        message.answers
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("/stop", True),
        ("/stop@xagent_bot", True),
        ("/pause now", True),
        ("STOP", True),
        ("暂停", True),
        ("停止", True),
        ("请暂停一下", False),
        ("/new", False),
    ],
)
def test_stop_request_text_aliases(text: str, expected: bool) -> None:
    bot = make_bot()

    assert bot._is_stop_request_text(text) is expected
