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

from ....config import get_default_task_execution_mode
from ....core.file_ref import build_file_id_ref
from ...api.chat import get_agent_manager
from ...models.database import get_db
from ...models.task import Task, TaskStatus
from ...models.uploaded_file import UploadedFile
from ...models.user import User
from ...services.chat_history_service import persist_user_message
from ...services.execution_result_projection import project_execution_result_for_channel
from ...services.file_turn import (
    append_uploaded_files_context,
    build_uploaded_files_context,
    normalize_attachments_for_persistence,
)
from ...services.managed_task_lease import (
    ManagedTaskLease,
    claim_managed_task_lease,
)
from ...services.task_execution_controller import (
    StaleTaskRunError,
    TaskControlState,
    apply_task_control_transition,
    control_state_for_status,
    task_control_snapshot,
    task_execution_controller,
)
from .handler import TelegramTraceHandler
from .utils import (
    TelegramFileRef,
    TelegramImageRef,
    markdown_to_tg_html,
    persist_telegram_assistant_turn,
    restore_telegram_task_context,
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
            logger.info(
                f"Received /start from {message.from_user.id} on bot {self.instance_id}"
            )
            await message.answer(
                "Hi, I'm Xagent. Send me anything you'd like help with, or use /new when you want a fresh start."
            )

        @self.dp.message(Command("new"))
        async def cmd_new(message: types.Message) -> None:
            logger.info(
                f"Received /new from {message.from_user.id} on bot {self.instance_id}"
            )
            self._start_new_conversation(message.from_user.id)
            await message.answer(
                "Fresh start. Send me what you'd like to work on next."
            )

        @self.dp.message(Command("stop", "pause"))
        async def cmd_stop(message: types.Message) -> None:
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

            if user_id not in self.user_message_queues:
                self.user_message_queues[user_id] = []
            self.user_message_queues[user_id].append(message)

            if (
                user_id not in self.user_message_tasks
                or self.user_message_tasks[user_id].done()
            ):
                self._schedule_user_queue(user_id)

    def _schedule_user_queue(self, user_id: int) -> None:
        self.user_message_tasks[user_id] = asyncio.create_task(
            self._process_user_queue(user_id)
        )

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
                execution_task.cancel()

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
        files = []

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
        db: Session,
    ) -> list:
        from ...services.uploaded_file_store import UploadedFileStore

        uploaded_files_info: list[dict] = []

        if not agent_service.workspace:
            logger.warning("Agent service workspace is not available for file upload")
            return uploaded_files_info

        target_dir = getattr(
            agent_service.workspace,
            "input_dir",
            agent_service.workspace.workspace_dir / "input",
        )

        for f in files:
            try:
                file_id = f.file_id
                tg_file = await self.bot.get_file(file_id)

                if hasattr(f, "file_name") and f.file_name:
                    file_name = f.file_name
                else:
                    ext = Path(tg_file.file_path).suffix if tg_file.file_path else ""
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

                file_record = UploadedFileStore(db).create_from_local_path(
                    local_path=target_path,
                    user_id=user_id,
                    task_id=task_id,
                    filename=normalized_file_name,
                    mime_type=mime_type,
                )
                setattr(file_record, "file_size", int(file_size))
                db.flush()

                agent_service.workspace.register_file(
                    str(target_path),
                    file_id=str(file_record.file_id),
                    db_session=db,
                )

                uploaded_files_info.append(
                    {
                        "file_id": str(file_record.file_id),
                        "name": normalized_file_name,
                        "path": str(target_path),
                        "type": mime_type,
                        "size": file_size,
                        "telegram_file_id": str(file_id),
                    }
                )
                logger.info(
                    f"Successfully downloaded and registered Telegram file: {normalized_file_name}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to process Telegram file {getattr(f, 'file_id', 'unknown')}: {e}"
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
        claimed_run_id: str | None = None
        managed_lease: ManagedTaskLease | None = None
        voice_asr_model: Any | None = None
        try:
            db_gen = get_db()
            db = next(db_gen)
            try:
                user = None
                if self.channel_id:
                    from ...models.user_channel import UserChannel

                    channel = (
                        db.query(UserChannel)
                        .filter(UserChannel.id == self.channel_id)
                        .first()
                    )
                    if channel:
                        user = db.query(User).filter(User.id == channel.user_id).first()
                        if channel.config:
                            allowed_users = channel.config.get("allowed_users")
                            if allowed_users is not None:
                                if str(last_message.from_user.id) not in allowed_users:
                                    await last_message.answer(
                                        "🚫 You are not authorized to use this bot."
                                    )
                                    return

                if not user:
                    await last_message.answer(
                        "Configuration error: Cannot find the owner of this bot."
                    )
                    return

                if self._consume_user_stop_request(user_id):
                    return

                if voice_file_ids:
                    voice_asr_model = self._resolve_voice_asr_model(db, user)
                    if voice_asr_model is None:
                        await last_message.answer(
                            "I couldn't understand that voice message because no "
                            "speech recognition model is configured. Configure an "
                            "ASR model or send the request as text."
                        )
                        return

                active_task_id = self.active_tasks.get(user_id)
                task = None

                if active_task_id == -1:
                    pass
                elif active_task_id:
                    task = (
                        db.query(Task)
                        .filter(Task.id == active_task_id, Task.user_id == user.id)
                        .first()
                    )

                is_new_task = False
                if not task:
                    task_title = text if text else "Untitled Task"
                    if len(task_title) > 50:
                        task_title = task_title[:50] + "..."

                    task = Task(
                        user_id=user.id,
                        title=task_title,
                        description=text,
                        status=TaskStatus.PENDING,
                        execution_mode=get_default_task_execution_mode(),
                        channel_id=self.channel_id,
                        channel_name=self.channel_name,
                        connector_runtime_selected_refs=[],
                    )
                    db.add(task)
                    db.commit()
                    db.refresh(task)
                    self.active_tasks[user_id] = task.id
                    self._save_active_tasks()
                    is_new_task = True
                else:
                    db.refresh(task)

                async with task_execution_controller.command(int(task.id)):
                    db.refresh(task)
                    managed_lease = claim_managed_task_lease(db, int(task.id))
                    if managed_lease is None:
                        await last_message.answer(
                            "I'm still working on the previous message. "
                            "Please wait for it to finish."
                        )
                        return
                    db.refresh(task)
                    run_snapshot = task_control_snapshot(task)
                    claimed_task_id = int(task.id)
                    claimed_run_id = run_snapshot.run_id

                agent_manager = get_agent_manager()
                agent_service = await agent_manager.get_agent_for_task(
                    int(task.id),
                    db,
                    user=user,  # type: ignore
                )

                await restore_telegram_task_context(agent_service, db, int(task.id))

                message_turn_id = str(uuid4())
                context: dict = {"turn_id": message_turn_id}

                if self._consume_user_stop_request(user_id):
                    apply_task_control_transition(
                        task,
                        TaskControlState.PAUSED,
                        status=TaskStatus.PAUSED,
                        expected_run_id=run_snapshot.run_id,
                    )
                    db.commit()
                    return

                uploaded_info: list[dict[str, Any]] = []
                persisted_attachments: list[dict[str, Any]] = []
                execution_text = text
                display_message = text
                if files:
                    uploaded_info = await self._download_and_register_files(
                        files=files,
                        agent_service=agent_service,
                        task_id=int(task.id),  # type: ignore
                        user_id=int(user.id),  # type: ignore
                        db=db,
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
                            voice_transcripts = (
                                await self._transcribe_uploaded_voice_files(
                                    voice_file_ids,
                                    uploaded_info,
                                    voice_asr_model,
                                )
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
                            if str(info.get("telegram_file_id"))
                            not in voice_file_id_set
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
                            task.description = text  # type: ignore
                            if voice_file_ids or not task.title:
                                title = text if len(text) <= 50 else f"{text[:50]}..."
                                task.title = title  # type: ignore[assignment]
                            db.commit()

                        context["state"] = context.get("state", {})
                        context["state"]["file_info"] = uploaded_info
                        context["file_info"] = uploaded_info
                        context["uploaded_files"] = [
                            str(info["path"]) for info in uploaded_info
                        ]
                        context["files"] = persisted_attachments
                        context["display_message"] = display_message

                if self._consume_user_stop_request(user_id):
                    apply_task_control_transition(
                        task,
                        TaskControlState.PAUSED,
                        status=TaskStatus.PAUSED,
                        expected_run_id=run_snapshot.run_id,
                    )
                    db.commit()
                    return

                persist_user_message(
                    db=db,
                    task_id=int(task.id),  # type: ignore
                    user_id=int(user.id),  # type: ignore
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
                    int(task.id),  # type: ignore
                    self.bot,
                    last_message.chat.id,
                    message_id=loading_msg.message_id,
                )
                agent_service.tracer.add_handler(tg_handler)

                from ...user_isolated_memory import UserContext

                actual_task_id = str(task.id)
                active_execution = (int(task.id), agent_service)  # type: ignore[arg-type]
                self.user_active_executions[user_id] = active_execution

                try:
                    if self._consume_user_stop_request(user_id):
                        apply_task_control_transition(
                            task,
                            TaskControlState.PAUSED,
                            status=TaskStatus.PAUSED,
                            expected_run_id=run_snapshot.run_id,
                        )
                        db.commit()
                        return

                    with UserContext(int(user.id)):  # type: ignore
                        result = await self._await_execution_with_stop_monitor(
                            user_id,
                            agent_manager.execute_task(
                                agent_service=agent_service,
                                task=execution_text,
                                context=context,
                                task_id=actual_task_id,
                                tracking_task_id=str(task.id),
                                db_session=db,
                                manage_task_lease=False,
                            ),
                            reason="Telegram stop requested",
                        )
                finally:
                    if self.user_active_executions.get(user_id) == active_execution:
                        self.user_active_executions.pop(user_id, None)
                    if tg_handler in agent_service.tracer.handlers:
                        agent_service.tracer.handlers.remove(tg_handler)

                projection = project_execution_result_for_channel(result)
                db.refresh(task)
                projected_control_state = control_state_for_status(
                    projection.task_status
                )
                if (
                    task.status != projection.task_status
                    or task.control_state != projected_control_state.value
                ):
                    try:
                        apply_task_control_transition(
                            task,
                            projected_control_state,
                            status=projection.task_status,
                            expected_run_id=run_snapshot.run_id,
                        )
                        db.commit()
                    except StaleTaskRunError:
                        db.rollback()
                        logger.info(
                            "Skipping stale Telegram completion for task %s run %s",
                            task.id,
                            run_snapshot.run_id,
                        )

                persist_telegram_assistant_turn(
                    db=db,
                    task_id=int(task.id),  # type: ignore
                    user_id=int(user.id),  # type: ignore
                    content=projection.transcript_content,
                    interactions=projection.interactions,
                    message_type=projection.message_type,
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
                        user_id=int(user.id),  # type: ignore
                        task_id=int(task.id),  # type: ignore
                        db=db,
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
                        user_id=int(user.id),  # type: ignore
                        task_id=int(task.id),  # type: ignore
                        db=db,
                        reply_to=last_message,
                    )
                    if failed_file_refs:
                        await self._send_file_fallback_message(
                            file_refs=failed_file_refs,
                            reply_to=last_message,
                        )

            finally:
                try:
                    next(db_gen)
                except StopIteration:
                    pass
        except Exception as e:
            logger.error(f"Error processing Telegram message: {e}")
            if claimed_task_id is not None and claimed_run_id is not None:
                try:
                    snapshot = await task_execution_controller.snapshot(claimed_task_id)
                    if (
                        snapshot is not None
                        and snapshot.run_id == claimed_run_id
                        and snapshot.status == TaskStatus.RUNNING
                    ):
                        await task_execution_controller.transition(
                            claimed_task_id,
                            TaskControlState.FAILED,
                            status=TaskStatus.FAILED,
                            expected_run_id=claimed_run_id,
                        )
                except Exception:
                    logger.warning(
                        "Failed to finalize Telegram task %s after channel error",
                        claimed_task_id,
                        exc_info=True,
                    )
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
            if voice_asr_model is not None:
                await self._close_voice_asr_model(voice_asr_model)
            if managed_lease is not None:
                await managed_lease.close()
            self.user_preparing_executions.discard(user_id)
            self._clear_user_stop_request(user_id)

    async def _send_output_images(
        self,
        *,
        image_refs: list[TelegramImageRef],
        user_id: int,
        task_id: int,
        db: Session,
        reply_to: types.Message,
    ) -> list[TelegramImageRef]:
        ordered_file_ids = list(dict.fromkeys(ref.file_id for ref in image_refs))
        failed_refs: list[TelegramImageRef] = []

        file_records = (
            db.query(UploadedFile)
            .filter(
                UploadedFile.file_id.in_(ordered_file_ids),
                UploadedFile.user_id == user_id,
                UploadedFile.task_id == task_id,
            )
            .all()
            if ordered_file_ids
            else []
        )
        file_record_by_id = {str(record.file_id): record for record in file_records}

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
        db: Session,
        reply_to: types.Message,
    ) -> list[TelegramFileRef]:
        ordered_file_ids = list(dict.fromkeys(ref.file_id for ref in file_refs))
        failed_refs: list[TelegramFileRef] = []

        file_records = (
            db.query(UploadedFile)
            .filter(
                UploadedFile.file_id.in_(ordered_file_ids),
                UploadedFile.user_id == user_id,
                UploadedFile.task_id == task_id,
            )
            .all()
            if ordered_file_ids
            else []
        )
        file_record_by_id = {str(record.file_id): record for record in file_records}

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

            record_filename = getattr(file_record, "filename", "")
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
        try:
            # Drop pending updates to ignore messages sent while the bot was offline/inactive
            await self.bot.delete_webhook(drop_pending_updates=True)
            # Get bot info manually just for logging (optional, since dp.start_polling also logs)
            # We remove the duplicate log to avoid confusion
            await self.dp.start_polling(self.bot, handle_signals=False)
        except Exception as e:
            logger.error(
                f"Telegram bot polling stopped due to error for {self.instance_id}: {e}",
                exc_info=True,
            )

    async def stop(self) -> None:
        if self.dp:
            await self.dp.stop_polling()
        if self.bot:
            await self.bot.session.close()


class TelegramChannelManager:
    def __init__(self) -> None:
        self.bots: Dict[str, TelegramBotInstance] = {}
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

        db_gen = get_db()
        db = next(db_gen)
        try:
            from ...models.user_channel import UserChannel

            channels = (
                db.query(UserChannel)
                .filter(
                    UserChannel.channel_type == "telegram",
                    UserChannel.is_active.is_(True),
                )
                .all()
            )
            for ch in channels:
                token = ch.config.get("bot_token")
                if token:
                    active_tokens.add(token)
                    channel_info_by_token[token] = {
                        "id": ch.id,
                        "name": ch.channel_name,
                    }
        except Exception as e:
            logger.error(f"Failed to load user channels for sync: {e}")
            return  # Don't try to sync if we failed to load from db
        finally:
            try:
                next(db_gen)
            except StopIteration:
                pass

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

    async def _stop_bot_for_token(self, token: str) -> None:
        if token in self.bots:
            bot = self.bots[token]
            logger.info(f"Stopping bot {bot.instance_id}...")

            try:
                # First try to stop the polling gracefully
                await bot.stop()
            except Exception as e:
                logger.error(f"Error while stopping bot {bot.instance_id}: {e}")

            if bot.polling_task and not bot.polling_task.done():
                bot.polling_task.cancel()
                try:
                    await bot.polling_task
                except asyncio.CancelledError:
                    pass

            del self.bots[token]
            logger.info(f"Successfully stopped and removed bot {bot.instance_id}")


_telegram_manager = None


def get_telegram_channel() -> TelegramChannelManager:
    global _telegram_manager
    if _telegram_manager is None:
        _telegram_manager = TelegramChannelManager()
    return _telegram_manager
