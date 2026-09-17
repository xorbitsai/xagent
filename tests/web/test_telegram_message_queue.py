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
    bot.user_active_trace_handlers = {}
    bot.user_preparing_executions = set()
    bot.user_stop_events = {}
    bot.user_conversation_generations = {}
    bot.user_switch_locks = {}
    bot.selected_agents = {}
    bot._save_selected_agents = lambda: True
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
    bot._save_active_tasks = lambda: True
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


def _fenced_turn_bot(finalized: list, *, active_task_id: int):  # type: ignore[no-untyped-def]
    bot = make_bot()
    bot.channel_id = 1
    bot.channel_name = "Telegram fence"
    bot.active_tasks = {101: active_task_id}
    bot._save_active_tasks = lambda: True
    bot._clear_user_stop_request = lambda _user_id: None
    bot._active_tasks_unsaved = False
    return bot


class _FenceLease:
    """Stand-in for ManagedTaskLease that mirrors its settlement semantics.

    ``close()`` is gated on ``_closed`` the way the real lease is, so a caller
    that closes a still-unsettled claim is visible here as ``closed_unsettled``
    -- against the real lease that is the RUNNING -> FAILED transition pinned by
    test_closing_a_running_claim_would_fail_the_task.
    """

    heartbeat_task = None

    def __init__(
        self,
        task_id: int,
        finalized: list,  # type: ignore[type-arg]
        *,
        finalize_succeeds: bool = True,
        finalize_raises: Exception | None = None,
    ) -> None:
        self.lease = TaskLease(
            task_id=task_id, runner_id="runner-fence", run_id="run-fence"
        )
        self._finalized = finalized
        self._settled = False
        self._finalize_succeeds = finalize_succeeds
        self._finalize_raises = finalize_raises
        self.closed = False
        self.closed_unsettled = False

    async def finalize_result(self, *, status: TaskStatus, **_kwargs) -> bool:
        if self._finalize_raises is not None:
            raise self._finalize_raises
        if not self._finalize_succeeds:
            # The unhealthy-heartbeat / TTL-retention path: the run is retained
            # for recovery rather than settled, so callers see False.
            return False
        self._settled = True
        self._finalized.append((self.lease.task_id, status))
        return True

    async def close(self) -> bool:
        self.closed = True
        if not self._settled:
            self.closed_unsettled = True
        return True


class _FenceMessage:
    voice = None
    chat = SimpleNamespace(id=456)

    def __init__(self) -> None:
        self.answers: list[str] = []

    async def answer(self, text: str, **_kwargs) -> None:
        self.answers.append(text)


@pytest.mark.asyncio
async def test_message_racing_switch_pauses_rather_than_failing_the_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Switch-first ordering: the turn resolves to the task the user is now in.

    Returning silently would swallow the message, and releasing the claim with
    ManagedTaskLease.close() would persist that task as FAILED, since close()
    maps RUNNING to FAILED. It must be finalized to the resumable PAUSED
    instead. See test_closing_a_running_claim_would_fail_the_task in
    tests/web/services/test_channel_runtime.py, which pins close()'s behaviour
    against a real lease and database.
    """

    finalized: list[tuple[int, TaskStatus]] = []
    bot = _fenced_turn_bot(finalized, active_task_id=7)
    bot._consume_user_stop_request = lambda _user_id: False
    lease = _FenceLease(42, finalized)

    async def extract_text(_message):  # type: ignore[no-untyped-def]
        return "hello", []

    async def prepare(**_kwargs):  # type: ignore[no-untyped-def]
        # The /switch commits while this turn is being prepared.
        bot.user_conversation_generations[101] = (
            bot.user_conversation_generations.get(101, 0) + 1
        )
        bot.active_tasks[101] = 42
        return SimpleNamespace(
            user_id=5,
            task_id=42,
            is_new_task=False,
            managed_lease=lease,
            requested_agent_missing=False,
        )

    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.prepare_channel_task", prepare
    )
    bot._extract_message_content = extract_text

    message = _FenceMessage()
    await bot._process_user_messages_batch(101, [message])  # type: ignore[arg-type]

    # Finalized to PAUSED before the batch's cleanup calls close(). That
    # ordering is what makes it safe: close() maps only a *RUNNING* task to
    # FAILED, so settling first leaves the status resumable.
    assert finalized == [(42, TaskStatus.PAUSED)]
    assert bot.active_tasks[101] == 42
    assert len(message.answers) == 1
    assert "send it again" in message.answers[0]


@pytest.mark.asyncio
async def test_fence_replies_even_when_settling_the_claim_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed settle must not swallow the user's message.

    finalize_result returns False on the ordinary unhealthy-heartbeat / TTL
    path, which makes _finalize_requested_stop raise TaskLeaseLostError. If that
    unwound past the reply the user would get nothing -- the exact invariant the
    fence exists to guarantee.
    """

    finalized: list[tuple[int, TaskStatus]] = []
    bot = _fenced_turn_bot(finalized, active_task_id=7)
    bot._consume_user_stop_request = lambda _user_id: False
    lease = _FenceLease(42, finalized, finalize_succeeds=False)

    async def extract_text(_message):  # type: ignore[no-untyped-def]
        return "hello", []

    async def prepare(**_kwargs):  # type: ignore[no-untyped-def]
        bot.user_conversation_generations[101] = (
            bot.user_conversation_generations.get(101, 0) + 1
        )
        bot.active_tasks[101] = 42
        return SimpleNamespace(
            user_id=5,
            task_id=42,
            is_new_task=False,
            managed_lease=lease,
            requested_agent_missing=False,
        )

    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.prepare_channel_task", prepare
    )
    bot._extract_message_content = extract_text

    message = _FenceMessage()
    await bot._process_user_messages_batch(101, [message])  # type: ignore[arg-type]

    # The settle failed, so nothing was finalized -- but the user was told.
    assert finalized == []
    assert len(message.answers) == 1
    assert "send it again" in message.answers[0]


def test_prune_idle_user_state_keeps_locks_and_drops_stop_events() -> None:
    """Pruning must never remove a lock: Lock.release() only schedules the
    next waiter, so locked() can be False while a waiter is queued -- dropping
    the lock there would hand the next command a fresh one and break the
    /switch-vs-/new mutual exclusion."""

    bot = make_bot()
    lock = asyncio.Lock()
    bot.user_switch_locks[101] = lock
    stop_event = asyncio.Event()
    bot.user_stop_events[101] = stop_event

    bot._prune_idle_user_state(101)

    assert bot.user_switch_locks[101] is lock
    assert 101 not in bot.user_stop_events

    # In-flight work blocks pruning entirely.
    bot.user_stop_events[101] = stop_event
    bot.user_preparing_executions.add(101)
    bot._prune_idle_user_state(101)
    assert 101 in bot.user_stop_events


@pytest.mark.asyncio
async def test_fence_settle_exception_does_not_reach_the_generic_error_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising settle must not unwind into the generic error handler.

    That handler finalizes the task FAILED -- which _settle_fenced_turn's own
    docstring forbids -- and sends a second, contradictory "Sorry..." reply.
    The fence notice must be the only reply the user sees.
    """

    finalized: list[tuple[int, TaskStatus]] = []
    bot = _fenced_turn_bot(finalized, active_task_id=7)
    bot._consume_user_stop_request = lambda _user_id: False
    lease = _FenceLease(42, finalized, finalize_raises=RuntimeError("db write failed"))

    async def extract_text(_message):  # type: ignore[no-untyped-def]
        return "hello", []

    async def prepare(**_kwargs):  # type: ignore[no-untyped-def]
        bot.user_conversation_generations[101] = (
            bot.user_conversation_generations.get(101, 0) + 1
        )
        bot.active_tasks[101] = 42
        return SimpleNamespace(
            user_id=5,
            task_id=42,
            is_new_task=False,
            managed_lease=lease,
            requested_agent_missing=False,
        )

    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.prepare_channel_task", prepare
    )
    bot._extract_message_content = extract_text

    message = _FenceMessage()
    await bot._process_user_messages_batch(101, [message])  # type: ignore[arg-type]

    assert finalized == []
    assert len(message.answers) == 1
    assert "send it again" in message.answers[0]
    assert not any("Sorry" in answer for answer in message.answers)


@pytest.mark.asyncio
async def test_stale_fenced_turn_pauses_the_old_task_and_still_replies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A claim for a conversation the user left is paused -- but never silently."""

    finalized: list[tuple[int, TaskStatus]] = []
    bot = _fenced_turn_bot(finalized, active_task_id=99)
    bot._consume_user_stop_request = lambda _user_id: False
    lease = _FenceLease(7, finalized)

    async def extract_text(_message):  # type: ignore[no-untyped-def]
        return "hello", []

    async def prepare(**_kwargs):  # type: ignore[no-untyped-def]
        bot.user_conversation_generations[101] = (
            bot.user_conversation_generations.get(101, 0) + 1
        )
        # active_tasks already points elsewhere: task 7 is genuinely stale.
        return SimpleNamespace(
            user_id=5,
            task_id=7,
            is_new_task=False,
            managed_lease=lease,
            requested_agent_missing=False,
        )

    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.prepare_channel_task", prepare
    )
    bot._extract_message_content = extract_text

    message = _FenceMessage()
    await bot._process_user_messages_batch(101, [message])  # type: ignore[arg-type]

    assert finalized == [(7, TaskStatus.PAUSED)]
    assert len(message.answers) == 1
    assert "send it again" in message.answers[0]


@pytest.mark.asyncio
async def test_stop_during_the_loading_message_removes_it_and_replies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loading message is the first awaited send, before any trace handler.

    It cannot be protected by the cancellation latch, so a stop landing during
    that send must be caught by the manual re-check: the stale loading message
    is deleted, the task is paused once, and the user still gets a reply.
    """

    finalized: list[tuple[int, TaskStatus]] = []
    bot = _fenced_turn_bot(finalized, active_task_id=42)
    lease = _FenceLease(42, finalized)
    loading_deleted: list[bool] = []
    loading_sent = False

    def consume_stop(_user_id: int) -> bool:
        # True only once the loading message has been sent -- the exact window
        # this protection covers, since no trace handler exists yet.
        return loading_sent

    bot._consume_user_stop_request = consume_stop

    async def extract_text(_message):  # type: ignore[no-untyped-def]
        return "hello", []

    async def prepare(**_kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(
            user_id=5,
            task_id=42,
            is_new_task=False,
            managed_lease=lease,
            requested_agent_missing=False,
        )

    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.prepare_channel_task", prepare
    )
    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.load_task_setup_snapshot_sync",
        lambda *_args: SimpleNamespace(
            runtime_user=None,
            conversation_history=(),
            conversation_watermark=None,
            execution_recovery=TaskExecutionRecoverySnapshot(),
        ),
    )

    agent_service = SimpleNamespace(
        set_conversation_history=lambda _messages, *, watermark=None: None,
        set_execution_context_messages=lambda _messages: None,
        set_recovered_skill_context=lambda _context: None,
    )

    class FakeAgentManager:
        async def get_agent_for_task(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return agent_service

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

    class LoadingMessage:
        message_id = 555

        async def delete(self) -> None:
            loading_deleted.append(True)

    class Message(_FenceMessage):
        from_user = SimpleNamespace(id=101)

        async def answer(self, text: str, **_kwargs):  # type: ignore[no-untyped-def]
            nonlocal loading_sent
            self.answers.append(text)
            if not loading_sent:
                loading_sent = True
                return LoadingMessage()
            return None

    message = Message()
    await bot._process_user_messages_batch(101, [message])  # type: ignore[arg-type]

    assert loading_deleted == [True]
    assert finalized == [(42, TaskStatus.PAUSED)]
    # The loading message plus an interruption notice -- never a silent return.
    assert len(message.answers) == 2
    # This checkpoint runs after persist_channel_user_message, so it must not
    # ask for a resend: complying would duplicate the message in history.
    assert "was received" in message.answers[1]
    assert "send it again" not in message.answers[1]


@pytest.mark.asyncio
async def test_stop_request_fence_pauses_and_replies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit /stop pauses the run even when it is the live conversation.

    That is what the user asked for, so unlike the switch-away case the PAUSED
    transition is correct here -- but the message still must not vanish.
    """

    finalized: list[tuple[int, TaskStatus]] = []
    bot = _fenced_turn_bot(finalized, active_task_id=42)
    stop_checks = 0

    def consume_stop(_user_id: int) -> bool:
        nonlocal stop_checks
        stop_checks += 1
        return stop_checks == 1

    bot._consume_user_stop_request = consume_stop
    lease = _FenceLease(42, finalized)

    async def extract_text(_message):  # type: ignore[no-untyped-def]
        return "hello", []

    async def prepare(**_kwargs):  # type: ignore[no-untyped-def]
        # No generation bump: this turn is fenced by the stop request alone.
        return SimpleNamespace(
            user_id=5,
            task_id=42,
            is_new_task=False,
            managed_lease=lease,
            requested_agent_missing=False,
        )

    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.prepare_channel_task", prepare
    )
    bot._extract_message_content = extract_text

    message = _FenceMessage()
    await bot._process_user_messages_batch(101, [message])  # type: ignore[arg-type]

    # The user asked for the pause, so it is applied.
    assert finalized == [(42, TaskStatus.PAUSED)]
    assert len(message.answers) == 1
    assert "send it again" in message.answers[0]


@pytest.mark.asyncio
async def test_stop_after_prepare_settles_preclaimed_task_instead_of_orphaning_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bot = make_bot()
    bot.channel_id = 1
    bot.channel_name = "Telegram stop after prepare"
    bot.active_tasks = {}
    bot._save_active_tasks = lambda: True
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
            requested_agent_missing=False,
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


@pytest.mark.asyncio
async def test_switch_stop_request_sees_batch_during_message_extraction() -> None:
    bot = make_bot()
    extraction_started = asyncio.Event()
    finish_extraction = asyncio.Event()

    async def slow_extract(_message: object) -> tuple[str, list]:
        extraction_started.set()
        await finish_extraction.wait()
        raise RuntimeError("stop test")

    bot._extract_message_content = slow_extract  # type: ignore[method-assign]
    processing = asyncio.create_task(
        bot._process_user_messages_batch(123, [SimpleNamespace()])
    )
    await extraction_started.wait()

    assert bot._request_current_conversation_stop(
        123,
        reason="Telegram task switch requested",
    )
    assert bot.user_stop_events[123].is_set()

    finish_extraction.set()
    with pytest.raises(RuntimeError, match="stop test"):
        await processing
    assert 123 not in bot.user_preparing_executions


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

    def fake_save_active_tasks() -> bool:
        bot.saved = True
        return True

    bot._save_active_tasks = fake_save_active_tasks

    assert bot._start_new_conversation(123) == (True, True)
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
async def test_empty_output_edits_the_loading_message_with_a_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty output must not leave the loading message as the final state.

    Telegram rejects empty text with "Bad Request: message text is empty", so
    editing with "" would fail silently and strand "Got it, I'm working on
    this now." forever. The placeholder must be non-empty.
    """

    bot = make_bot()
    bot.channel_id = 1
    bot.channel_name = "Telegram empty output"
    bot.active_tasks = {}
    bot.bot = object()
    bot._save_active_tasks = lambda: True
    bot._consume_user_stop_request = lambda _user_id: False
    bot._clear_user_stop_request = lambda _user_id: None

    lease = TaskLease(task_id=44, runner_id="runner-a", run_id="run-a")

    class FakeManagedLease:
        heartbeat_task = None

        def __init__(self) -> None:
            self.lease = lease

        async def close(self) -> bool:
            return True

        async def finalize_result(self, *, status: TaskStatus, **_kwargs) -> bool:
            return True

    class FakeTracer:
        def add_handler(self, _handler: object) -> None:
            return None

        def remove_handler(self, _handler: object) -> None:
            return None

    agent_service = SimpleNamespace(
        tracer=FakeTracer(),
        set_conversation_history=lambda _messages, *, watermark=None: None,
        set_execution_context_messages=lambda _messages: None,
        set_recovered_skill_context=lambda _context: None,
    )

    class FakeAgentManager:
        async def get_agent_for_task(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return agent_service

        def execute_task(self, **_kwargs):  # type: ignore[no-untyped-def]
            async def run() -> str:
                return "raw result"

            return run()

    async def extract_text(_message):  # type: ignore[no-untyped-def]
        return "hello", []

    async def await_execution(_user_id, execution, *, reason):  # type: ignore[no-untyped-def]
        return await execution

    async def fake_prepare(**_kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(
            user_id=5,
            task_id=44,
            is_new_task=True,
            managed_lease=FakeManagedLease(),
            requested_agent_missing=False,
        )

    async def persist_message(**_kwargs) -> None:  # type: ignore[no-untyped-def]
        return None

    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.prepare_channel_task", fake_prepare
    )
    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.load_task_setup_snapshot_sync",
        lambda *_args: SimpleNamespace(
            runtime_user=None,
            conversation_history=(),
            conversation_watermark=None,
            execution_recovery=TaskExecutionRecoverySnapshot(),
        ),
    )
    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.get_agent_manager",
        lambda: FakeAgentManager(),
    )
    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.persist_channel_user_message",
        persist_message,
    )
    # The execution produced no visible text and no attachments.
    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.project_execution_result_for_channel",
        lambda _result: SimpleNamespace(
            visible_text="",
            task_status=TaskStatus.COMPLETED,
            transcript_content="",
            interactions=[],
            message_type="assistant_response",
            diagnostic_error=None,
        ),
    )
    bot._extract_message_content = extract_text
    bot._await_execution_with_stop_monitor = await_execution

    edits: list[str] = []

    class LoadingMessage:
        message_id = 77

        async def edit_text(self, text: str, **_kwargs) -> None:
            edits.append(text)

        async def delete(self) -> None:
            return None

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

    assert len(edits) == 1
    assert edits[0].strip() != ""
    assert "Task completed." in edits[0]


@pytest.mark.asyncio
async def test_successful_telegram_turn_hands_finalize_the_execution_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The waiting-branch call site forwards its own execute_task() result.

    finalize_result has no reader for execution_result yet, but the channel
    must already supply the exact object execute_task returned so a future
    reader gets the full result mapping rather than a synthesized draft.
    """

    bot = make_bot()
    bot.channel_id = 1
    bot.channel_name = "Telegram execution result"
    bot.active_tasks = {}
    bot.bot = object()
    bot._save_active_tasks = lambda: True
    bot._consume_user_stop_request = lambda _user_id: False
    bot._clear_user_stop_request = lambda _user_id: None

    lease = TaskLease(task_id=48, runner_id="runner-a", run_id="run-a")
    finalized: list[dict] = []

    class FakeManagedLease:
        heartbeat_task = None

        def __init__(self) -> None:
            self.lease = lease

        async def close(self) -> bool:
            return True

        async def finalize_result(self, **kwargs) -> bool:  # type: ignore[no-untyped-def]
            finalized.append(kwargs)
            return True

    execution_result = {"success": True, "output": "Telegram reply"}

    class FakeTracer:
        def add_handler(self, _handler: object) -> None:
            return None

        def remove_handler(self, _handler: object) -> None:
            return None

    agent_service = SimpleNamespace(
        tracer=FakeTracer(),
        set_conversation_history=lambda _messages, *, watermark=None: None,
        set_execution_context_messages=lambda _messages: None,
        set_recovered_skill_context=lambda _context: None,
    )

    class FakeAgentManager:
        async def get_agent_for_task(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return agent_service

        async def execute_task(self, **_kwargs):  # type: ignore[no-untyped-def]
            return execution_result

    async def extract_text(_message):  # type: ignore[no-untyped-def]
        return "hello", []

    async def await_execution(_user_id, execution, *, reason):  # type: ignore[no-untyped-def]
        return await execution

    async def fake_prepare(**_kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(
            user_id=5,
            task_id=48,
            is_new_task=True,
            managed_lease=FakeManagedLease(),
            requested_agent_missing=False,
        )

    async def persist_message(**_kwargs) -> None:  # type: ignore[no-untyped-def]
        return None

    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.prepare_channel_task", fake_prepare
    )
    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.load_task_setup_snapshot_sync",
        lambda *_args: SimpleNamespace(
            runtime_user=None,
            conversation_history=(),
            conversation_watermark=None,
            execution_recovery=TaskExecutionRecoverySnapshot(),
        ),
    )
    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.get_agent_manager",
        lambda: FakeAgentManager(),
    )
    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.persist_channel_user_message",
        persist_message,
    )
    bot._extract_message_content = extract_text
    bot._await_execution_with_stop_monitor = await_execution

    class LoadingMessage:
        message_id = 77

        async def edit_text(self, text: str, **_kwargs) -> None:
            return None

        async def delete(self) -> None:
            return None

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

    assert len(finalized) == 1
    assert finalized[0]["execution_result"] is execution_result


@pytest.mark.asyncio
async def test_channel_failure_suppresses_stale_error_after_exact_settlement_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bot = make_bot()
    bot.channel_id = 1
    bot.channel_name = "Telegram exact settlement"
    bot.active_tasks = {}
    bot.bot = object()
    bot._save_active_tasks = lambda: True
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
            execution_result=None,
            **_kwargs,
        ) -> bool:
            settlements.append((self.lease, status))
            settled_execution_results.append(execution_result)
            return False

    managed = FakeManagedLease()
    settlements: list[tuple[TaskLease, TaskStatus]] = []
    settled_execution_results: list[object] = []

    class FakeTracer:
        def __init__(self) -> None:
            self.handlers: list[object] = []

        def add_handler(self, handler: object) -> None:
            self.handlers.append(handler)

        def remove_handler(self, handler: object) -> None:
            if handler in self.handlers:
                self.handlers.remove(handler)

    agent_service = SimpleNamespace(
        tracer=FakeTracer(),
        set_conversation_history=lambda _messages, *, watermark=None: None,
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
            requested_agent_missing=False,
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
            conversation_watermark=None,
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
    # The except branch has no execution result in scope, so it settles
    # through the default rather than passing one along.
    assert settled_execution_results == [None]
    assert managed.closed is True
    assert "Sorry, an error occurred while processing your request." not in (
        message.answers
    )


@pytest.mark.asyncio
async def test_finalize_requested_stop_settles_with_no_execution_result() -> None:
    """A requested /stop has no channel execution result to hand over."""

    finalized: list[dict] = []

    class FakeManagedLease:
        async def finalize_result(self, **kwargs) -> bool:  # type: ignore[no-untyped-def]
            finalized.append(kwargs)
            return True

    await TelegramBotInstance._finalize_requested_stop(
        FakeManagedLease(),
        task_id=45,  # type: ignore[arg-type]
    )

    assert finalized == [{"status": TaskStatus.PAUSED}]
    assert "execution_result" not in finalized[0]


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


def _agent(agent_id: int, name: str) -> SimpleNamespace:
    return SimpleNamespace(agent_id=agent_id, name=name)


def test_build_agents_keyboard_lists_default_and_agents() -> None:
    agents = [_agent(1, "Alpha"), _agent(2, "Beta")]

    keyboard = TelegramBotInstance._build_agents_keyboard(
        agents, 0, selected_agent_id=2
    )

    rows = keyboard.inline_keyboard
    assert rows[0][0].text == "Default assistant"
    assert rows[0][0].callback_data == "agsel:default"
    assert rows[1][0].text == "Alpha"
    assert rows[1][0].callback_data == "agsel:1"
    assert rows[2][0].text == "Beta ✓"
    assert rows[2][0].callback_data == "agsel:2"
    assert len(rows) == 3
    for row in rows:
        for button in row:
            assert len(button.callback_data.encode()) <= 64


def test_build_agents_keyboard_truncates_long_names_with_ellipsis() -> None:
    long_name = "A" * 80

    keyboard = TelegramBotInstance._build_agents_keyboard(
        [_agent(1, long_name)], 0, selected_agent_id=None
    )

    label = keyboard.inline_keyboard[1][0].text
    assert label == f"{'A' * 57}..."


def test_build_agents_keyboard_marks_default_when_no_selection() -> None:
    keyboard = TelegramBotInstance._build_agents_keyboard(
        [_agent(1, "Alpha")], 0, selected_agent_id=None
    )

    assert keyboard.inline_keyboard[0][0].text == "Default assistant ✓"


def test_build_agents_keyboard_paginates() -> None:
    agents = [_agent(i, f"Agent {i}") for i in range(20)]
    page_size = TelegramBotInstance.agents_page_size

    first = TelegramBotInstance._build_agents_keyboard(
        agents, 0, selected_agent_id=None
    )
    # default row + page of agents + nav row
    assert len(first.inline_keyboard) == page_size + 2
    assert [b.text for b in first.inline_keyboard[-1]] == ["Next ➡"]
    assert first.inline_keyboard[-1][0].callback_data == "agpage:1"

    middle = TelegramBotInstance._build_agents_keyboard(
        agents, 1, selected_agent_id=None
    )
    assert [b.text for b in middle.inline_keyboard[-1]] == ["⬅ Prev", "Next ➡"]

    # An out-of-range page clamps to the last page
    last = TelegramBotInstance._build_agents_keyboard(
        agents, 99, selected_agent_id=None
    )
    assert [b.text for b in last.inline_keyboard[-1]] == ["⬅ Prev"]
    assert last.inline_keyboard[1][0].text == f"Agent {2 * page_size}"


class _FakeCallback:
    def __init__(self, user_id: int, data: str) -> None:
        self.from_user = SimpleNamespace(id=user_id)
        self.data = data
        self.answers: list[tuple[tuple, dict]] = []
        self.message = None

    async def answer(self, *args, **kwargs) -> None:
        self.answers.append((args, kwargs))


@pytest.mark.asyncio
async def test_agent_selection_callback_binds_agent_and_starts_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bot = make_bot()
    bot.channel_id = 1
    bot.active_tasks = {123: 44}
    bot._save_active_tasks = lambda: True

    async def fake_get_agent(**kwargs):  # type: ignore[no-untyped-def]
        assert kwargs == {
            "channel_id": 1,
            "external_user_id": "123",
            "agent_id": 7,
        }
        return SimpleNamespace(agent_id=7, name="Research Agent")

    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.get_channel_owner_agent",
        fake_get_agent,
    )

    callback = _FakeCallback(123, "agsel:7")
    await bot._handle_agent_selection_callback(callback)  # type: ignore[arg-type]

    assert bot.selected_agents == {123: 7}
    assert bot.active_tasks[123] == -1
    assert callback.answers == [(("Research Agent selected",), {})]


@pytest.mark.asyncio
async def test_agent_selection_callback_default_clears_selection() -> None:
    bot = make_bot()
    bot.channel_id = 1
    bot.active_tasks = {123: 44}
    bot._save_active_tasks = lambda: True
    bot.selected_agents = {123: 7}

    callback = _FakeCallback(123, "agsel:default")
    await bot._handle_agent_selection_callback(callback)  # type: ignore[arg-type]

    assert bot.selected_agents == {}
    assert bot.active_tasks[123] == -1
    assert callback.answers == [(("Default assistant selected",), {})]


@pytest.mark.asyncio
async def test_agent_selection_callback_alerts_when_agent_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bot = make_bot()
    bot.channel_id = 1
    bot.active_tasks = {123: 44}
    bot._save_active_tasks = lambda: True
    bot.selected_agents = {123: 7}

    async def fake_get_agent(**_kwargs):  # type: ignore[no-untyped-def]
        return None

    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.get_channel_owner_agent",
        fake_get_agent,
    )

    callback = _FakeCallback(123, "agsel:9")
    await bot._handle_agent_selection_callback(callback)  # type: ignore[arg-type]

    # Selection and conversation stay untouched
    assert bot.selected_agents == {123: 7}
    assert bot.active_tasks[123] == 44
    assert callback.answers == [
        (("That agent is no longer available.",), {"show_alert": True})
    ]


@pytest.mark.asyncio
async def test_agents_command_reports_empty_agent_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bot = make_bot()
    bot.channel_id = 1

    async def fake_list(**_kwargs):  # type: ignore[no-untyped-def]
        return ()

    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.list_channel_owner_agents",
        fake_list,
    )

    answers: list[tuple[str, dict]] = []

    class Message:
        from_user = SimpleNamespace(id=123)

        async def answer(self, text: str, **kwargs) -> None:
            answers.append((text, kwargs))

    await bot._handle_agents_command(Message())  # type: ignore[arg-type]

    assert len(answers) == 1
    assert "don't have any custom agents yet" in answers[0][0]


@pytest.mark.asyncio
async def test_agents_command_sends_keyboard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bot = make_bot()
    bot.channel_id = 1
    bot.selected_agents = {123: 2}

    async def fake_list(**kwargs):  # type: ignore[no-untyped-def]
        assert kwargs == {"channel_id": 1, "external_user_id": "123"}
        return (_agent(1, "Alpha"), _agent(2, "Beta"))

    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.list_channel_owner_agents",
        fake_list,
    )

    answers: list[tuple[str, dict]] = []

    class Message:
        from_user = SimpleNamespace(id=123)

        async def answer(self, text: str, **kwargs) -> None:
            answers.append((text, kwargs))

    await bot._handle_agents_command(Message())  # type: ignore[arg-type]

    assert len(answers) == 1
    keyboard = answers[0][1]["reply_markup"]
    assert keyboard.inline_keyboard[2][0].text == "Beta ✓"


@pytest.mark.asyncio
async def test_agents_command_reports_unauthorized_sender(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.services.channel_runtime import ChannelAuthorizationError

    bot = make_bot()
    bot.channel_id = 1

    async def fake_list(**_kwargs):  # type: ignore[no-untyped-def]
        raise ChannelAuthorizationError("nope")

    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.list_channel_owner_agents",
        fake_list,
    )

    answers: list[str] = []

    class Message:
        from_user = SimpleNamespace(id=123)

        async def answer(self, text: str, **_kwargs) -> None:
            answers.append(text)

    await bot._handle_agents_command(Message())  # type: ignore[arg-type]

    assert answers == ["🚫 You are not authorized to use this bot."]


def test_set_selected_agent_round_trips_persistence(tmp_path: Path) -> None:
    bot = object.__new__(TelegramBotInstance)
    bot.instance_id = "test-bot"
    bot.selected_agents_file = tmp_path / "selected_agents.json"
    bot.selected_agents = {}

    bot._set_selected_agent(123, 7)
    assert bot._load_selected_agents() == {123: 7}

    bot._set_selected_agent(123, None)
    assert bot._load_selected_agents() == {}


@pytest.mark.asyncio
async def test_batch_passes_selected_agent_and_clears_stale_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bot = make_bot()
    bot.channel_id = 1
    bot.channel_name = "Telegram agent selection"
    bot.active_tasks = {}
    bot._save_active_tasks = lambda: True
    bot._clear_user_stop_request = lambda _user_id: None
    bot.selected_agents = {123: 7}

    stop_checks = 0

    def consume_stop(_user_id: int) -> bool:
        nonlocal stop_checks
        stop_checks += 1
        return stop_checks == 1

    bot._consume_user_stop_request = consume_stop

    lease = TaskLease(task_id=44, runner_id="runner-a", run_id="run-a")

    class FakeManagedLease:
        heartbeat_task = None

        def __init__(self) -> None:
            self.lease = lease

        async def finalize_result(self, **_kwargs) -> bool:
            return True

        async def close(self) -> bool:
            return True

    prepare_kwargs: dict = {}

    async def prepare(**kwargs):  # type: ignore[no-untyped-def]
        prepare_kwargs.update(kwargs)
        return SimpleNamespace(
            user_id=5,
            task_id=44,
            is_new_task=True,
            managed_lease=FakeManagedLease(),
            requested_agent_missing=True,
        )

    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.prepare_channel_task",
        prepare,
    )

    async def extract_text(_message):  # type: ignore[no-untyped-def]
        return "hello", []

    bot._extract_message_content = extract_text

    answers: list[str] = []

    class Message:
        voice = None
        chat = SimpleNamespace(id=456)

        async def answer(self, text: str, **_kwargs) -> None:
            answers.append(text)

    await bot._process_user_messages_batch(123, [Message()])  # type: ignore[arg-type]

    assert prepare_kwargs["agent_id"] == 7
    assert bot.selected_agents == {}
    assert any("no longer available" in text for text in answers)


def test_selected_agents_store_path_is_durable_and_collision_free(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XAGENT_STORAGE_ROOT", str(tmp_path))

    by_channel = TelegramBotInstance._selected_agents_store_path(5, "12345678:token-a")
    other_channel = TelegramBotInstance._selected_agents_store_path(
        6, "12345678:token-a"
    )
    assert by_channel == tmp_path / "telegram" / "selected_agents_channel_5.json"
    assert by_channel != other_channel
    assert by_channel.is_relative_to(tmp_path)

    # Without a channel id the key is a full-token hash, so distinct bots
    # sharing an eight-character token prefix cannot collide.
    same_prefix_a = TelegramBotInstance._selected_agents_store_path(
        None, "12345678:token-a"
    )
    same_prefix_b = TelegramBotInstance._selected_agents_store_path(
        None, "12345678:token-b"
    )
    assert same_prefix_a != same_prefix_b
    assert same_prefix_a.is_relative_to(tmp_path)


@pytest.mark.asyncio
async def test_agent_selection_toast_stays_within_telegram_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bot = make_bot()
    bot.channel_id = 1
    bot.active_tasks = {}
    bot._save_active_tasks = lambda: True
    # 192-char name: the raw toast would be 201 chars, one over the API limit
    long_name = "A" * 192

    async def fake_get_agent(**_kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(agent_id=7, name=long_name)

    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.get_channel_owner_agent",
        fake_get_agent,
    )

    callback = _FakeCallback(123, "agsel:7")
    await bot._handle_agent_selection_callback(callback)  # type: ignore[arg-type]

    toast = callback.answers[0][0][0]
    assert len(toast) == 200
    assert toast.endswith("...")
    assert bot.selected_agents == {123: 7}


@pytest.mark.asyncio
async def test_batch_discards_prepared_task_after_conversation_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bot = make_bot()
    bot.channel_id = 1
    bot.channel_name = "Telegram generation fence"
    bot.active_tasks = {}
    bot._save_active_tasks = lambda: True
    bot._clear_user_stop_request = lambda _user_id: None
    bot._consume_user_stop_request = lambda _user_id: False
    bot.selected_agents = {123: 7}

    lease = TaskLease(task_id=44, runner_id="runner-a", run_id="run-a")
    finalized: list[TaskStatus] = []

    class FakeManagedLease:
        heartbeat_task = None

        def __init__(self) -> None:
            self.lease = lease

        async def finalize_result(self, *, status: TaskStatus, **_kwargs) -> bool:
            finalized.append(status)
            return True

        async def close(self) -> bool:
            return True

    async def prepare(**_kwargs):  # type: ignore[no-untyped-def]
        # The user switches agents while this preparation is in flight
        bot._set_selected_agent(123, 9)
        bot._start_new_conversation(123)
        return SimpleNamespace(
            user_id=5,
            task_id=44,
            is_new_task=True,
            managed_lease=FakeManagedLease(),
            requested_agent_missing=True,
        )

    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.prepare_channel_task",
        prepare,
    )

    async def extract_text(_message):  # type: ignore[no-untyped-def]
        return "hello", []

    bot._extract_message_content = extract_text

    answers: list[str] = []

    class Message:
        voice = None
        chat = SimpleNamespace(id=456)

        async def answer(self, text: str, **_kwargs) -> None:
            answers.append(text)

    await bot._process_user_messages_batch(123, [Message()])  # type: ignore[arg-type]

    # The stale claim is settled, and neither the new conversation marker nor
    # the newer agent selection is overwritten by the stale result.
    assert finalized == [TaskStatus.PAUSED]
    assert bot.active_tasks[123] == -1
    assert bot.selected_agents == {123: 9}
    # The fence must not swallow the message: the user is told to resend.
    assert len(answers) == 1
    assert "send it again" in answers[0]
