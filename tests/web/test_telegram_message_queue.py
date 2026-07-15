import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiogram import types

from xagent.web.channels.telegram.bot import (
    TelegramBotInstance,
    TelegramVoiceTranscriptionError,
)


def make_bot() -> TelegramBotInstance:
    bot = object.__new__(TelegramBotInstance)
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
