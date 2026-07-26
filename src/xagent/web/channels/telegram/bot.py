import asyncio
import html
import json
import logging
import mimetypes
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Coroutine, Dict, Optional, cast
from uuid import uuid4

if TYPE_CHECKING:
    from ....core.agent.service import AgentService

from aiogram import Bot, Dispatcher, types
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import FSInputFile
from sqlalchemy.orm import Session

from ....core.file_ref import build_file_id_ref
from ...api.chat import get_agent_manager
from ...models.database import get_session_local
from ...models.task import TaskStatus
from ...models.user import User
from ...services.channel_runtime import (
    ChannelAuthorizationError,
    ChannelConfigurationError,
    DownloadedChannelFile,
    authorize_channel_sender,
    load_active_channel_configs,
    load_channel_output_files,
    persist_channel_user_message,
    prepare_channel_task,
    register_channel_uploaded_files,
    update_channel_task_fields,
)
from ...services.db_runtime import (
    cancel_and_drain_async_task,
    drain_async_task_cancellation_safe,
    run_db_io_cancellation_safe,
)
from ...services.execution_result_projection import project_execution_result_for_channel
from ...services.file_turn import (
    append_uploaded_files_context,
    build_uploaded_files_context,
    normalize_attachments_for_persistence,
)
from ...services.managed_task_lease import ManagedTaskLease
from ...services.task_execution_context_service import (
    materialize_task_execution_recovery_state,
)
from ...services.task_lease_service import TaskLeaseLostError
from ...services.task_setup_snapshot import load_task_setup_snapshot_sync
from .handler import TelegramTraceHandler
from .utils import (
    TelegramFileRef,
    TelegramImageRef,
    markdown_to_tg_html,
    strip_telegram_file_refs,
    strip_telegram_image_refs,
)

logger = logging.getLogger(__name__)


class TelegramVoiceTranscriptionError(RuntimeError):
    """Raised when a Telegram voice prompt cannot be transcribed."""


class TelegramBotInstance:
    queue_flush_delay_seconds = 1.0
    voice_transcription_timeout_seconds = 180.0
    stop_text_aliases = {"/stop", "/pause", "stop", "pause", "停止", "暂停"}

    def __init__(
        self,
        token: str,
        instance_id: str,
        channel_id: Optional[int] = None,
        channel_name: Optional[str] = None,
    ):
        self.bot_token = token
        self.instance_id = instance_id
        self.channel_id = channel_id
        self.channel_name = channel_name
        self.bot: Bot
        self.dp: Dispatcher
        self.polling_task: Optional[asyncio.Task] = None
        self.user_message_queues: Dict[int, list] = {}
        self.user_message_tasks: Dict[int, asyncio.Task] = {}
        self.user_active_executions: Dict[int, tuple[int, object]] = {}
        self.user_preparing_executions: set[int] = set()
        self.user_stop_events: Dict[int, asyncio.Event] = {}
        self._accepting = True
        self._ingress_stopped = False
        self._stop_lock: asyncio.Lock | None = None
        self._stop_loop: asyncio.AbstractEventLoop | None = None

        # Load active tasks state
        self.active_tasks_file = Path(f"data/telegram_active_tasks_{instance_id}.json")
        self.active_tasks = self._load_active_tasks()

        default_props = DefaultBotProperties(parse_mode=ParseMode.HTML)

        # Support HTTP proxy from environment for domestic testing
        proxy_url = (
            os.getenv("HTTPS_PROXY")
            or os.getenv("https_proxy")
            or os.getenv("HTTP_PROXY")
            or os.getenv("http_proxy")
        )
        if proxy_url:
            logger.info(f"Using proxy for Telegram Bot {instance_id}: {proxy_url}")
            from aiogram.client.session.aiohttp import AiohttpSession

            session = AiohttpSession(proxy=proxy_url)
            self.bot = Bot(token=self.bot_token, session=session, default=default_props)
        else:
            self.bot = Bot(token=self.bot_token, default=default_props)

        self.dp = Dispatcher()
        self._register_handlers()

    def _load_active_tasks(self) -> dict:
        if self.active_tasks_file.exists():
            try:
                with open(self.active_tasks_file, "r") as f:
                    # Convert string keys back to int
                    return {int(k): v for k, v in json.load(f).items()}
            except Exception as e:
                logger.error(
                    f"Failed to load Telegram active tasks for {self.instance_id}: {e}"
                )
        return {}

    def _save_active_tasks(self) -> None:
        try:
            self.active_tasks_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.active_tasks_file, "w") as f:
                json.dump(self.active_tasks, f)
        except Exception as e:
            logger.error(
                f"Failed to save Telegram active tasks for {self.instance_id}: {e}"
            )

    def _register_handlers(self) -> None:
        from aiogram.filters import Command

        @self.dp.message(CommandStart())
        async def cmd_start(message: types.Message) -> None:
            if not self._accepting or message.from_user is None:
                return
            logger.info(
                f"Received /start from {message.from_user.id} on bot {self.instance_id}"
            )
            await message.answer(
                "Hi, I'm Xagent. Send me anything you'd like help with, or use /new when you want a fresh start."
            )

        @self.dp.message(Command("new"))
        async def cmd_new(message: types.Message) -> None:
            if not self._accepting or message.from_user is None:
                return
            logger.info(
                f"Received /new from {message.from_user.id} on bot {self.instance_id}"
            )
            self._start_new_conversation(message.from_user.id)
            await message.answer(
                "Fresh start. Send me what you'd like to work on next."
            )

        @self.dp.message(Command("stop", "pause"))
        async def cmd_stop(message: types.Message) -> None:
            if not self._accepting or message.from_user is None:
                return
            logger.info(
                f"Received stop command from {message.from_user.id} on bot {self.instance_id}"
            )
            stopped = self._stop_current_conversation(message.from_user.id)
            if stopped:
                await message.answer(
                    "Stopped the current run. Send another message to continue here, or use /new for a fresh task."
                )
            else:
                await message.answer("No active run to stop.")

        @self.dp.message()
        async def handle_message(message: types.Message) -> None:
            if not self._accepting or message.from_user is None:
                return
            msg_content = (
                message.text
                or message.caption
                or (
                    "[File/Media attached]"
                    if message.document
                    or message.photo
                    or message.audio
                    or message.voice
                    or message.video
                    else "Unknown"
                )
            )
            logger.info(
                f"Received message from {message.from_user.id} on bot {self.instance_id}: {msg_content}"
            )

            user_id = message.from_user.id
            if self._is_stop_request_text(msg_content):
                logger.info(
                    f"Received stop text from {user_id} on bot {self.instance_id}: {msg_content}"
                )
                stopped = self._stop_current_conversation(user_id)
                if stopped:
                    await message.answer(
                        "Stopped the current run. Send another message to continue here, or use /new for a fresh task."
                    )
                else:
                    await message.answer("No active run to stop.")
                return

            self._enqueue_user_message(user_id, message)

    def _enqueue_user_message(self, user_id: int, message: Any) -> bool:
        if not self._accepting:
            return False

        self.user_message_queues.setdefault(user_id, []).append(message)
        task = self.user_message_tasks.get(user_id)
        if task is None or task.done():
            self._schedule_user_queue(user_id)
        return True

    def _schedule_user_queue(self, user_id: int) -> bool:
        if not self._accepting:
            return False
        self.user_message_tasks[user_id] = asyncio.create_task(
            self._process_user_queue(user_id)
        )
        return True

    def _start_new_conversation(self, user_id: int) -> bool:
        stopped = self._request_current_conversation_stop(
            user_id, reason="new Telegram conversation requested"
        )
        self.active_tasks[user_id] = -1
        self._save_active_tasks()
        return stopped

    def _stop_current_conversation(self, user_id: int) -> bool:
        return self._request_current_conversation_stop(
            user_id, reason="Telegram stop requested"
        )

    def _request_current_conversation_stop(self, user_id: int, *, reason: str) -> bool:
        queued_messages = self.user_message_queues.pop(user_id, None)
        stopped = self._stop_user_active_execution(user_id, reason=reason)
        preparing = user_id in self.user_preparing_executions
        if preparing and not stopped:
            self._request_user_stop(user_id)
        return bool(queued_messages) or stopped or preparing

    def _stop_user_active_execution(self, user_id: int, *, reason: str) -> bool:
        active_execution = self.user_active_executions.get(user_id)
        if active_execution is None:
            return False

        task_id, agent_service = active_execution
        pause_execution_by_id = getattr(agent_service, "pause_execution_by_id", None)
        if not callable(pause_execution_by_id):
            logger.warning(
                "Telegram active task %s for user %s does not support pause",
                task_id,
                user_id,
            )
            return False

        try:
            return bool(pause_execution_by_id(str(task_id), reason=reason))
        except Exception as e:
            logger.warning(
                "Failed to pause Telegram active task %s for user %s: %s",
                task_id,
                user_id,
                e,
            )
            return False

    def _get_user_stop_event(self, user_id: int) -> asyncio.Event:
        event = self.user_stop_events.get(user_id)
        if event is None:
            event = asyncio.Event()
            self.user_stop_events[user_id] = event
        return event

    def _request_user_stop(self, user_id: int) -> None:
        self._get_user_stop_event(user_id).set()

    def _consume_user_stop_request(self, user_id: int) -> bool:
        event = self.user_stop_events.get(user_id)
        if event is None or not event.is_set():
            return False
        event.clear()
        return True

    def _clear_user_stop_request(self, user_id: int) -> None:
        event = self.user_stop_events.get(user_id)
        if event is not None:
            event.clear()

    async def _await_execution_with_stop_monitor(
        self,
        user_id: int,
        execution: Coroutine[Any, Any, dict[str, Any]],
        *,
        reason: str,
    ) -> dict[str, Any]:
        execution_task: asyncio.Task[dict[str, Any]] = asyncio.create_task(execution)
        stop_event = self._get_user_stop_event(user_id)

        try:
            while True:
                if execution_task.done():
                    return await execution_task

                if stop_event.is_set():
                    while not execution_task.done():
                        if self._stop_user_active_execution(user_id, reason=reason):
                            stop_event.clear()
                            break
                        await asyncio.sleep(0.05)
                    continue

                done, _ = await asyncio.wait(
                    {execution_task},
                    timeout=0.05,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if execution_task in done:
                    return await execution_task
        finally:
            if not execution_task.done():
                await cancel_and_drain_async_task(execution_task)

    @staticmethod
    async def _finalize_requested_stop(
        managed_lease: ManagedTaskLease,
        *,
        task_id: int,
    ) -> None:
        if not await managed_lease.finalize_result(status=TaskStatus.PAUSED):
            raise TaskLeaseLostError(
                f"task {task_id} ownership changed before Telegram stop"
            )

    def _is_stop_request_text(self, text: str) -> bool:
        normalized = text.strip().lower()
        if normalized.startswith("/"):
            normalized = normalized.split()[0].split("@", 1)[0]
        return normalized in self.stop_text_aliases

    async def _process_user_queue(self, user_id: int) -> None:
        while True:
            await asyncio.sleep(self.queue_flush_delay_seconds)
            messages = self.user_message_queues.pop(user_id, [])
            if messages:
                await self._process_user_messages_batch(user_id, messages)

            if self.user_message_queues.get(user_id):
                continue

            current_task = cast(asyncio.Task, asyncio.current_task())
            if self.user_message_tasks.get(user_id) is current_task:
                self.user_message_tasks.pop(user_id, None)

            if not self.user_message_queues.get(user_id):
                return

            self.user_message_tasks[user_id] = current_task

    async def _extract_message_content(
        self, message: types.Message
    ) -> tuple[str, list]:
        text = message.text or message.caption or ""
        files: list[Any] = []

        if message.document:
            files.append(message.document)
        elif message.photo:
            files.append(message.photo[-1])
        elif message.audio:
            files.append(message.audio)
        elif message.voice:
            files.append(message.voice)
        elif message.video:
            files.append(message.video)

        return text, files

    def _resolve_voice_asr_model(self, db: Session, user: User) -> Any | None:
        from fastapi import HTTPException

        from ....core.model.asr.adapter import get_asr_model_instance
        from ...api.model import _resolve_asr_model_for_transcription

        try:
            db_model = _resolve_asr_model_for_transcription(db, user)
        except HTTPException as exc:
            if exc.status_code == 404:
                return None
            raise
        return get_asr_model_instance(db_model)

    def _resolve_voice_asr_model_isolated(self, user_id: int) -> Any | None:
        """Resolve ASR in one worker-owned Session without leaking ORM state."""

        SessionLocal = get_session_local()
        with SessionLocal() as db:
            user = db.query(User).filter(User.id == user_id).first()
            if user is None:
                return None
            return self._resolve_voice_asr_model(db, user)

    @staticmethod
    def _audio_format_from_file_info(file_info: dict[str, Any]) -> str | None:
        mime_type = str(file_info.get("type") or "").lower()
        if mime_type.startswith("audio/"):
            subtype = mime_type.split("/", 1)[1].split(";", 1)[0].strip()
            if subtype:
                return "mp3" if subtype == "mpeg" else subtype

        suffix = Path(str(file_info.get("name") or "")).suffix.lower().lstrip(".")
        if suffix in {"oga", "opus"}:
            return "ogg"
        return suffix or None

    @staticmethod
    def _mime_type_for_telegram_file(file_input: Any, target_path: Path) -> str:
        declared_mime_type = getattr(file_input, "mime_type", None)
        guessed_mime_type, _ = mimetypes.guess_type(str(target_path))
        if isinstance(file_input, types.Voice) and declared_mime_type:
            return str(declared_mime_type)
        if guessed_mime_type and guessed_mime_type != "application/octet-stream":
            return guessed_mime_type
        return str(
            declared_mime_type or guessed_mime_type or "application/octet-stream"
        )

    @staticmethod
    def _display_message_for_user(message: str, has_files: bool) -> str:
        if message.strip():
            return message
        return "Uploaded file(s)" if has_files else message

    async def _transcribe_uploaded_voice_files(
        self,
        voice_file_ids: list[str],
        uploaded_info: list[dict[str, Any]],
        asr_model: Any,
    ) -> dict[str, str]:
        uploaded_by_source_id = {
            str(info.get("telegram_file_id")): info
            for info in uploaded_info
            if info.get("telegram_file_id")
        }
        transcripts: dict[str, str] = {}

        for voice_file_id in voice_file_ids:
            file_info = uploaded_by_source_id.get(voice_file_id)
            if file_info is None:
                raise TelegramVoiceTranscriptionError(
                    "Telegram voice input was not downloaded"
                )

            try:
                result = await asyncio.wait_for(
                    asr_model.transcribe(
                        audio=str(file_info["path"]),
                        format=self._audio_format_from_file_info(file_info),
                    ),
                    timeout=self.voice_transcription_timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                raise TelegramVoiceTranscriptionError(
                    "Telegram voice transcription timed out"
                ) from exc
            except Exception as exc:
                logger.error(
                    "Failed to transcribe Telegram voice input %s: %s",
                    voice_file_id,
                    exc,
                )
                raise TelegramVoiceTranscriptionError(
                    "Telegram voice transcription failed"
                ) from exc

            raw_text = getattr(result, "text", result)
            transcript = str(raw_text).strip()
            if not transcript:
                raise TelegramVoiceTranscriptionError(
                    "Telegram voice transcription returned empty text"
                )
            transcripts[voice_file_id] = transcript

        return transcripts

    @staticmethod
    def _compose_prompt_text(
        message_contents: list[tuple[types.Message, str, list]],
        voice_transcripts: dict[str, str],
    ) -> str:
        prompt_parts: list[str] = []
        for message, message_text, _files in message_contents:
            if message_text:
                prompt_parts.append(message_text)
            voice = getattr(message, "voice", None)
            if voice is not None:
                transcript = voice_transcripts.get(str(voice.file_id))
                if transcript:
                    prompt_parts.append(transcript)
        return "\n".join(prompt_parts)

    @staticmethod
    async def _close_voice_asr_model(asr_model: Any) -> None:
        from inspect import isawaitable

        close = getattr(asr_model, "aclose", None) or getattr(asr_model, "close", None)
        if not callable(close):
            return
        try:
            result = close()
            if isawaitable(result):
                await result
        except Exception:
            logger.warning("Failed to close Telegram voice ASR model", exc_info=True)

    async def _download_and_register_files(
        self,
        files: list,
        agent_service: "AgentService",
        task_id: int,
        user_id: int,
    ) -> list:
        if not agent_service.workspace:
            logger.warning("Agent service workspace is not available for file upload")
            return []

        target_dir = getattr(
            agent_service.workspace,
            "input_dir",
            agent_service.workspace.workspace_dir / "input",
        )

        downloaded_files: list[DownloadedChannelFile] = []
        for f in files:
            try:
                file_id = f.file_id
                tg_file = await self.bot.get_file(file_id)
                if not tg_file.file_path:
                    raise FileNotFoundError(
                        f"Telegram file {file_id} has no downloadable path"
                    )

                if hasattr(f, "file_name") and f.file_name:
                    file_name = f.file_name
                else:
                    ext = Path(tg_file.file_path).suffix
                    if not ext and type(f).__name__ == "PhotoSize":
                        ext = ".jpg"
                    file_name = f"{file_id}{ext}"

                from ...api.websocket import (
                    build_unique_target_path,
                    normalize_filename,
                )

                try:
                    normalized_file_name = normalize_filename(file_name)
                    target_path = build_unique_target_path(
                        target_dir, normalized_file_name
                    )
                except ImportError:
                    import time

                    normalized_file_name = f"{int(time.time())}_{file_name}"
                    target_path = Path(target_dir) / normalized_file_name

                target_path.parent.mkdir(parents=True, exist_ok=True)

                await self.bot.download_file(tg_file.file_path, destination=target_path)

                mime_type = self._mime_type_for_telegram_file(f, target_path)

                file_size = getattr(f, "file_size", None) or target_path.stat().st_size
                downloaded_files.append(
                    DownloadedChannelFile(
                        name=normalized_file_name,
                        path=target_path,
                        mime_type=mime_type,
                        size=int(file_size),
                        source_id=str(file_id),
                    )
                )
            except Exception as e:
                logger.error(
                    "Failed to download Telegram file %s: %s",
                    getattr(f, "file_id", "unknown"),
                    e,
                )

        registered = await register_channel_uploaded_files(
            workspace=agent_service.workspace,
            task_id=task_id,
            user_id=user_id,
            files=tuple(downloaded_files),
        )
        uploaded_files_info = [
            item.to_file_info(source_key="telegram_file_id") for item in registered
        ]
        for item in registered:
            logger.info(
                "Successfully downloaded and registered Telegram file: %s",
                item.name,
            )
        return uploaded_files_info

    async def _process_user_messages_batch(
        self, user_id: int, messages: list[types.Message]
    ) -> None:
        message_contents: list[tuple[types.Message, str, list]] = []
        files = []

        # We'll use the last message for answering
        last_message = messages[-1]

        for msg in messages:
            message_text, message_files = await self._extract_message_content(msg)
            message_contents.append((msg, message_text, message_files))
            files.extend(message_files)

        text = self._compose_prompt_text(message_contents, {})
        voice_file_ids = [
            str(voice.file_id)
            for msg in messages
            if (voice := getattr(msg, "voice", None)) is not None
        ]

        if not text and not files:
            return

        self.user_preparing_executions.add(user_id)
        self._clear_user_stop_request(user_id)
        claimed_task_id: int | None = None
        managed_lease: ManagedTaskLease | None = None
        voice_asr_model: Any | None = None
        try:
            try:
                expected_owner_user_id: int | None = None
                if voice_file_ids:
                    owner = await authorize_channel_sender(
                        channel_id=self.channel_id,
                        external_user_id=str(user_id),
                    )
                    expected_owner_user_id = owner.user_id
                    voice_asr_model = await run_db_io_cancellation_safe(
                        lambda: self._resolve_voice_asr_model_isolated(
                            expected_owner_user_id
                        )
                    )
                    if voice_asr_model is None:
                        await last_message.answer(
                            "I couldn't understand that voice message because no "
                            "speech recognition model is configured. Configure an "
                            "ASR model or send the request as text."
                        )
                        return

                active_task_id = self.active_tasks.get(user_id)
                prepared_task = await prepare_channel_task(
                    channel_id=self.channel_id,
                    external_user_id=str(user_id),
                    active_task_id=(
                        int(active_task_id) if active_task_id is not None else None
                    ),
                    text=text,
                    channel_name=self.channel_name,
                    expected_owner_user_id=expected_owner_user_id,
                )
            except ChannelAuthorizationError:
                await last_message.answer("🚫 You are not authorized to use this bot.")
                return
            except ChannelConfigurationError:
                await last_message.answer(
                    "Configuration error: Cannot find the owner of this bot."
                )
                return

            if prepared_task is None:
                await last_message.answer(
                    "I'm still working on the previous message. "
                    "Please wait for it to finish."
                )
                return

            # No await is allowed between receiving the committed claim and
            # taking ownership of its managed heartbeat in this transport.
            managed_lease = prepared_task.managed_lease
            task_id = prepared_task.task_id
            claimed_task_id = task_id
            owner_user_id = prepared_task.user_id
            is_new_task = prepared_task.is_new_task
            if is_new_task:
                self.active_tasks[user_id] = task_id
                self._save_active_tasks()

            if self._consume_user_stop_request(user_id):
                await self._finalize_requested_stop(
                    managed_lease,
                    task_id=task_id,
                )
                return

            setup_snapshot = await run_db_io_cancellation_safe(
                lambda: load_task_setup_snapshot_sync(task_id, owner_user_id)
            )
            if setup_snapshot is None:
                raise RuntimeError(f"Task {task_id} disappeared before execution")

            agent_manager = get_agent_manager()
            agent_service = await agent_manager.get_agent_for_task(
                task_id,
                user=setup_snapshot.runtime_user,
                task_setup_snapshot=setup_snapshot,
                task_owner_user_id=owner_user_id,
            )
            agent_service.set_conversation_history(
                [dict(message) for message in setup_snapshot.conversation_history]
            )
            recovery_state = await materialize_task_execution_recovery_state(
                setup_snapshot.execution_recovery
            )
            agent_service.set_execution_context_messages(
                recovery_state.get("messages", [])
            )
            agent_service.set_recovered_skill_context(
                recovery_state.get("skill_context")
            )

            message_turn_id = str(uuid4())
            context: dict = {"turn_id": message_turn_id}

            if self._consume_user_stop_request(user_id):
                await self._finalize_requested_stop(
                    managed_lease,
                    task_id=task_id,
                )
                return

            uploaded_info: list[dict[str, Any]] = []
            persisted_attachments: list[dict[str, Any]] = []
            execution_text = text
            display_message = text
            if files:
                uploaded_info = await self._download_and_register_files(
                    files=files,
                    agent_service=agent_service,
                    task_id=task_id,
                    user_id=owner_user_id,
                )
                if voice_file_ids and not uploaded_info:
                    raise TelegramVoiceTranscriptionError(
                        "Telegram voice input was not downloaded"
                    )
                if uploaded_info:
                    persisted_attachments = normalize_attachments_for_persistence(
                        uploaded_info
                    )
                    voice_transcripts: dict[str, str] = {}
                    if voice_file_ids:
                        voice_transcripts = await self._transcribe_uploaded_voice_files(
                            voice_file_ids,
                            uploaded_info,
                            voice_asr_model,
                        )
                        await self._close_voice_asr_model(voice_asr_model)
                        voice_asr_model = None

                    text = self._compose_prompt_text(
                        message_contents,
                        voice_transcripts,
                    )
                    display_message = self._display_message_for_user(
                        text,
                        bool(uploaded_info),
                    )
                    voice_file_id_set = set(voice_file_ids)
                    voice_uploaded_info = [
                        info
                        for info in uploaded_info
                        if str(info.get("telegram_file_id")) in voice_file_id_set
                    ]
                    regular_uploaded_info = [
                        info
                        for info in uploaded_info
                        if str(info.get("telegram_file_id")) not in voice_file_id_set
                    ]
                    file_info_list = [
                        f"[{info['name']}]({build_file_id_ref(info['file_id'])})"
                        for info in regular_uploaded_info
                    ]
                    if text and file_info_list:
                        text += f"\n\n{' '.join(file_info_list)}"
                    elif file_info_list:
                        text = " ".join(file_info_list)
                    execution_text = append_uploaded_files_context(
                        text,
                        build_uploaded_files_context(voice_uploaded_info),
                    )
                    if is_new_task:
                        title = None
                        if voice_file_ids:
                            title = text if len(text) <= 50 else f"{text[:50]}..."
                        await update_channel_task_fields(
                            task_id=task_id,
                            user_id=owner_user_id,
                            description=text,
                            title=title,
                        )

                    context["state"] = context.get("state", {})
                    context["state"]["file_info"] = uploaded_info
                    context["file_info"] = uploaded_info
                    context["uploaded_files"] = [
                        str(info["path"]) for info in uploaded_info
                    ]
                    context["files"] = persisted_attachments
                    context["display_message"] = display_message

            if self._consume_user_stop_request(user_id):
                await self._finalize_requested_stop(
                    managed_lease,
                    task_id=task_id,
                )
                return

            await persist_channel_user_message(
                task_id=task_id,
                user_id=owner_user_id,
                content=display_message,
                attachments=persisted_attachments or None,
                turn_id=message_turn_id,
            )

            loading_msg = await last_message.answer(
                "Got it, I'm working on this now.\n"
                "<i>I'll update this message as I make progress.</i>",
                parse_mode=ParseMode.HTML,
            )

            tg_handler = TelegramTraceHandler(
                task_id,
                self.bot,
                last_message.chat.id,
                message_id=loading_msg.message_id,
            )
            agent_service.tracer.add_handler(tg_handler)

            from ...user_isolated_memory import UserContext

            actual_task_id = str(task_id)
            active_execution = (task_id, agent_service)
            self.user_active_executions[user_id] = active_execution

            try:
                if self._consume_user_stop_request(user_id):
                    await self._finalize_requested_stop(
                        managed_lease,
                        task_id=task_id,
                    )
                    return

                with UserContext(owner_user_id):
                    result = await self._await_execution_with_stop_monitor(
                        user_id,
                        agent_manager.execute_task(
                            agent_service=agent_service,
                            task=execution_text,
                            context=context,
                            task_id=actual_task_id,
                            tracking_task_id=actual_task_id,
                            db_session=None,
                            manage_task_lease=False,
                            task_lease=managed_lease.lease,
                            task_lease_heartbeat_task=managed_lease.heartbeat_task,
                        ),
                        reason="Telegram stop requested",
                    )
            finally:
                if self.user_active_executions.get(user_id) == active_execution:
                    self.user_active_executions.pop(user_id, None)
                if tg_handler in agent_service.tracer.handlers:
                    agent_service.tracer.handlers.remove(tg_handler)

            projection = project_execution_result_for_channel(result)
            if not await managed_lease.finalize_result(
                status=projection.task_status,
                assistant_content=projection.transcript_content,
                interactions=projection.interactions,
                message_type=projection.message_type,
            ):
                raise TaskLeaseLostError(
                    f"task {task_id} ownership changed before Telegram result"
                )

            output, image_refs, file_refs = self._extract_telegram_output_refs(
                projection.visible_text,
            )
            if not output and (image_refs or file_refs):
                output = "Task completed."

            max_len = 4000
            text_chunks = [
                output[i : i + max_len] for i in range(0, len(output), max_len)
            ]

            try:
                html_chunk0 = markdown_to_tg_html(text_chunks[0])
                await loading_msg.edit_text(html_chunk0, parse_mode=ParseMode.HTML)
            except Exception as e:
                if "message is not modified" not in str(e).lower():
                    try:
                        await loading_msg.edit_text(text_chunks[0])
                    except Exception as e2:
                        if "message is not modified" not in str(e2).lower():
                            logger.warning(f"Failed to edit message: {e2}")

            for chunk in text_chunks[1:]:
                try:
                    html_chunk = markdown_to_tg_html(chunk)
                    await last_message.answer(html_chunk, parse_mode=ParseMode.HTML)
                except Exception:
                    await last_message.answer(chunk)

            if image_refs:
                failed_image_refs = await self._send_output_images(
                    image_refs=image_refs,
                    user_id=owner_user_id,
                    task_id=task_id,
                    reply_to=last_message,
                )
                if failed_image_refs:
                    await self._send_image_fallback_message(
                        image_refs=failed_image_refs,
                        reply_to=last_message,
                    )
            if file_refs:
                failed_file_refs = await self._send_output_files(
                    file_refs=file_refs,
                    user_id=owner_user_id,
                    task_id=task_id,
                    reply_to=last_message,
                )
                if failed_file_refs:
                    await self._send_file_fallback_message(
                        file_refs=failed_file_refs,
                        reply_to=last_message,
                    )
        except TaskLeaseLostError:
            logger.warning(
                "Telegram execution lost task %s lease; skipping stale result",
                claimed_task_id,
            )
        except Exception as e:
            logger.error(f"Error processing Telegram message: {e}")
            if managed_lease is not None:
                try:
                    finalized = await managed_lease.finalize_result(
                        status=TaskStatus.FAILED,
                    )
                except Exception:
                    logger.warning(
                        "Failed to finalize Telegram task %s after channel error",
                        claimed_task_id,
                        exc_info=True,
                    )
                    return
                if not finalized:
                    logger.warning(
                        "Telegram task %s ownership changed after channel error; "
                        "skipping stale error response",
                        claimed_task_id,
                    )
                    return
            if isinstance(e, TelegramVoiceTranscriptionError):
                await last_message.answer(
                    "I couldn't transcribe that voice message. Please try again "
                    "or send the request as text."
                )
            else:
                await last_message.answer(
                    "Sorry, an error occurred while processing your request."
                )
        finally:

            async def _cleanup_message_batch() -> None:
                try:
                    if managed_lease is not None:
                        await managed_lease.close()
                finally:
                    try:
                        if voice_asr_model is not None:
                            await self._close_voice_asr_model(voice_asr_model)
                    finally:
                        self.user_preparing_executions.discard(user_id)
                        self._clear_user_stop_request(user_id)

            cleanup_task = asyncio.create_task(_cleanup_message_batch())
            await drain_async_task_cancellation_safe(cleanup_task)

    async def _send_output_images(
        self,
        *,
        image_refs: list[TelegramImageRef],
        user_id: int,
        task_id: int,
        reply_to: types.Message,
    ) -> list[TelegramImageRef]:
        ordered_file_ids = list(dict.fromkeys(ref.file_id for ref in image_refs))
        failed_refs: list[TelegramImageRef] = []

        file_records = await load_channel_output_files(
            file_ids=ordered_file_ids,
            user_id=user_id,
            task_id=task_id,
        )
        file_record_by_id = {record.file_id: record for record in file_records}

        sent_file_ids: set[str] = set()
        for image_ref in image_refs:
            if image_ref.file_id in sent_file_ids:
                continue
            sent_file_ids.add(image_ref.file_id)

            file_record = file_record_by_id.get(image_ref.file_id)
            if not file_record:
                logger.warning(
                    "Telegram output image not found: file_id=%s task_id=%s",
                    image_ref.file_id,
                    task_id,
                )
                failed_refs.append(image_ref)
                continue

            mime_type = file_record.mime_type or ""
            if not mime_type.startswith("image/"):
                logger.warning(
                    "Telegram output file is not an image: file_id=%s mime_type=%s",
                    image_ref.file_id,
                    mime_type,
                )
                failed_refs.append(image_ref)
                continue

            image_path = Path(file_record.storage_path)
            if not image_path.is_file():
                logger.warning(
                    "Telegram output image path missing: file_id=%s path=%s",
                    image_ref.file_id,
                    image_path,
                )
                failed_refs.append(image_ref)
                continue

            caption = (
                html.escape(image_ref.alt_text[:512]) if image_ref.alt_text else None
            )
            try:
                await reply_to.answer_photo(
                    FSInputFile(image_path), caption=caption or None
                )
            except Exception as e:
                logger.warning(
                    "Failed to send Telegram output image: file_id=%s error=%s",
                    image_ref.file_id,
                    e,
                )
                failed_refs.append(image_ref)

        return failed_refs

    def _extract_telegram_output_refs(
        self, output: Optional[str]
    ) -> tuple[str, list[TelegramImageRef], list[TelegramFileRef]]:
        """Extract only local attachments explicitly referenced in the final answer."""
        if not output:
            return "", [], []

        output, image_refs = strip_telegram_image_refs(output)
        output, file_refs = strip_telegram_file_refs(output)
        image_refs, file_refs = self._dedupe_telegram_output_refs(
            image_refs,
            file_refs,
        )
        return output, image_refs, file_refs

    def _dedupe_telegram_output_refs(
        self,
        image_refs: list[TelegramImageRef],
        file_refs: list[TelegramFileRef],
    ) -> tuple[list[TelegramImageRef], list[TelegramFileRef]]:
        deduped_images: list[TelegramImageRef] = []
        image_file_ids: set[str] = set()
        for image_ref in image_refs:
            if image_ref.file_id in image_file_ids:
                continue
            image_file_ids.add(image_ref.file_id)
            deduped_images.append(image_ref)

        deduped_files: list[TelegramFileRef] = []
        file_ids: set[str] = set()
        for file_ref in file_refs:
            if file_ref.file_id in image_file_ids or file_ref.file_id in file_ids:
                continue
            file_ids.add(file_ref.file_id)
            deduped_files.append(file_ref)

        return deduped_images, deduped_files

    async def _send_image_fallback_message(
        self, *, image_refs: list[TelegramImageRef], reply_to: types.Message
    ) -> None:
        subject = "image" if len(image_refs) == 1 else "images"
        lines = [
            f"I couldn't send the {subject} through Telegram, but the file reference is still available:"
        ]
        for image_ref in image_refs:
            label = image_ref.alt_text or "image"
            lines.append(f"- {label}: file:{image_ref.file_id}")
        text = "\n".join(lines)
        try:
            await reply_to.answer(markdown_to_tg_html(text), parse_mode=ParseMode.HTML)
        except Exception:
            await reply_to.answer(text)

    async def _send_output_files(
        self,
        *,
        file_refs: list[TelegramFileRef],
        user_id: int,
        task_id: int,
        reply_to: types.Message,
    ) -> list[TelegramFileRef]:
        ordered_file_ids = list(dict.fromkeys(ref.file_id for ref in file_refs))
        failed_refs: list[TelegramFileRef] = []

        file_records = await load_channel_output_files(
            file_ids=ordered_file_ids,
            user_id=user_id,
            task_id=task_id,
        )
        file_record_by_id = {record.file_id: record for record in file_records}

        sent_file_ids: set[str] = set()
        for file_ref in file_refs:
            if file_ref.file_id in sent_file_ids:
                continue
            sent_file_ids.add(file_ref.file_id)

            file_record = file_record_by_id.get(file_ref.file_id)
            if not file_record:
                logger.warning(
                    "Telegram output file not found: file_id=%s task_id=%s",
                    file_ref.file_id,
                    task_id,
                )
                failed_refs.append(file_ref)
                continue

            file_path = Path(file_record.storage_path)
            if not file_path.is_file():
                logger.warning(
                    "Telegram output file path missing: file_id=%s path=%s",
                    file_ref.file_id,
                    file_path,
                )
                failed_refs.append(file_ref)
                continue

            record_filename = file_record.filename
            caption_source = file_ref.label or str(record_filename or "file")
            caption = html.escape(caption_source[:1024])
            try:
                await reply_to.answer_document(
                    FSInputFile(file_path), caption=caption or None
                )
            except Exception as e:
                logger.warning(
                    "Failed to send Telegram output file: file_id=%s error=%s",
                    file_ref.file_id,
                    e,
                )
                failed_refs.append(file_ref)

        return failed_refs

    async def _send_file_fallback_message(
        self, *, file_refs: list[TelegramFileRef], reply_to: types.Message
    ) -> None:
        subject = "file" if len(file_refs) == 1 else "files"
        lines = [
            f"I couldn't send the {subject} through Telegram, but the file reference is still available:"
        ]
        for file_ref in file_refs:
            label = file_ref.label or "file"
            lines.append(f"- {label}: file:{file_ref.file_id}")
        text = "\n".join(lines)
        try:
            await reply_to.answer(markdown_to_tg_html(text), parse_mode=ParseMode.HTML)
        except Exception:
            await reply_to.answer(text)

    async def start(self) -> None:
        if not self._accepting:
            return
        try:
            # Drop pending updates to ignore messages sent while the bot was offline/inactive
            await self.bot.delete_webhook(drop_pending_updates=True)
            if not self._accepting:
                return
            # Get bot info manually just for logging (optional, since dp.start_polling also logs)
            # We remove the duplicate log to avoid confusion
            await self.dp.start_polling(self.bot, handle_signals=False)
        except Exception as e:
            logger.error(
                f"Telegram bot polling stopped due to error for {self.instance_id}: {e}",
                exc_info=True,
            )

    def _stop_lock_for_current_loop(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        lock = self._stop_lock
        if lock is None or (self._stop_loop is not loop and not lock.locked()):
            lock = asyncio.Lock()
            self._stop_lock = lock
            self._stop_loop = loop
        elif self._stop_loop is not loop:
            raise RuntimeError("Telegram bot stop is already running on another loop")
        return lock

    async def _drain_user_message_tasks(self) -> None:
        current = asyncio.current_task()
        tasks = {
            task for task in self.user_message_tasks.values() if task is not current
        }
        for task in tasks:
            if not task.done():
                task.cancel()

        async def drain_tasks() -> None:
            await asyncio.gather(*tasks, return_exceptions=True)

        cleanup_task = asyncio.create_task(drain_tasks())
        try:
            await drain_async_task_cancellation_safe(cleanup_task)
        finally:
            self.user_message_tasks.clear()
            self.user_message_queues.clear()
            self.user_active_executions.clear()
            self.user_preparing_executions.clear()
            self.user_stop_events.clear()

    async def _stop_ingress(self) -> None:
        try:
            if self.dp:
                await self.dp.stop_polling()
        finally:
            if self.bot:
                await self.bot.session.close()

    async def _stop_once(self, lock: asyncio.Lock) -> None:
        async with lock:
            try:
                if not self._ingress_stopped:
                    await self._stop_ingress()
                    self._ingress_stopped = True
            finally:
                await self._drain_user_message_tasks()

    async def stop(self) -> None:
        self._accepting = False
        stop_task = asyncio.create_task(
            self._stop_once(self._stop_lock_for_current_loop())
        )
        await drain_async_task_cancellation_safe(stop_task)


class TelegramChannelManager:
    def __init__(self) -> None:
        self.bots: Dict[str, TelegramBotInstance] = {}
        self._bot_stop_tasks: Dict[str, asyncio.Task[None]] = {}
        self.enabled = True  # Always enabled, we load dynamically

    async def start(self) -> None:
        await self._sync_bots_async()

    async def stop(self) -> None:
        tokens = list(self.bots.keys())
        for token in tokens:
            await self._stop_bot_for_token(token)

    async def _sync_bots_async(self) -> None:
        active_tokens = set()
        channel_info_by_token: Dict[str, Dict] = {}

        try:
            channels = await load_active_channel_configs(
                channel_type="telegram",
                required_config_keys=("bot_token",),
            )
            for ch in channels:
                token = ch.config_value("bot_token")
                if token:
                    active_tokens.add(token)
                    channel_info_by_token[token] = {
                        "id": ch.channel_id,
                        "name": ch.channel_name,
                    }
        except Exception as e:
            logger.error(f"Failed to load user channels for sync: {e}")
            return  # Don't try to sync if we failed to load from db

        current_tokens = set(self.bots.keys())

        logger.info(
            f"Syncing telegram bots. Current active in db: {len(active_tokens)}, currently running: {len(current_tokens)}"
        )

        # Stop bots that are no longer active
        for token in current_tokens - active_tokens:
            await self._stop_bot_for_token(token)

        # Start bots that are newly active
        for token in active_tokens - current_tokens:
            channel_info = channel_info_by_token.get(token, {})
            ch_id = channel_info.get("id")
            ch_name = channel_info.get("name")
            await self._start_bot_for_token(
                token,
                int(ch_id) if ch_id is not None else None,
                str(ch_name) if ch_name is not None else None,
            )

    async def _start_bot_for_token(
        self,
        token: str,
        channel_id: Optional[int] = None,
        channel_name: Optional[str] = None,
    ) -> None:
        if token not in self.bots:
            instance_id = token[:8] + "..." if len(token) > 8 else "unknown"
            logger.info(f"Initializing Telegram channel {instance_id}...")
            bot = TelegramBotInstance(
                token, instance_id, channel_id=channel_id, channel_name=channel_name
            )
            self.bots[token] = bot
            bot.polling_task = asyncio.create_task(bot.start())

    async def _shutdown_bot_for_token(
        self,
        token: str,
        bot: TelegramBotInstance,
    ) -> None:
        logger.info(f"Stopping bot {bot.instance_id}...")
        try:
            try:
                # First try to stop the polling gracefully
                await bot.stop()
            except Exception as e:
                logger.error(f"Error while stopping bot {bot.instance_id}: {e}")
        finally:
            try:
                if bot.polling_task is not None:
                    await cancel_and_drain_async_task(bot.polling_task)
            finally:
                if self.bots.get(token) is bot:
                    self.bots.pop(token, None)
        logger.info(f"Successfully stopped and removed bot {bot.instance_id}")

    async def _stop_bot_for_token(self, token: str) -> None:
        stop_task = self._bot_stop_tasks.get(token)
        if stop_task is None:
            bot = self.bots.get(token)
            if bot is None:
                return
            stop_task = asyncio.create_task(self._shutdown_bot_for_token(token, bot))
            self._bot_stop_tasks[token] = stop_task

        try:
            await drain_async_task_cancellation_safe(stop_task)
        finally:
            if stop_task.done() and self._bot_stop_tasks.get(token) is stop_task:
                self._bot_stop_tasks.pop(token, None)


_telegram_manager = None


def get_telegram_channel() -> TelegramChannelManager:
    global _telegram_manager
    if _telegram_manager is None:
        _telegram_manager = TelegramChannelManager()
    return _telegram_manager
