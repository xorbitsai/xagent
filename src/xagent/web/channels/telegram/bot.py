import asyncio
import html
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from ....core.agent.service import AgentService

from aiogram import Bot, Dispatcher, types
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import FSInputFile
from sqlalchemy.orm import Session

from ...api.chat import get_agent_manager
from ...models.database import get_db
from ...models.task import Task, TaskStatus
from ...models.uploaded_file import UploadedFile
from ...models.user import User
from .handler import TelegramTraceHandler
from .utils import TelegramImageRef, markdown_to_tg_html, strip_telegram_image_refs

logger = logging.getLogger(__name__)


class TelegramBotInstance:
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
            self.active_tasks[message.from_user.id] = -1
            self._save_active_tasks()
            await message.answer(
                "Fresh start. Send me what you'd like to work on next."
            )

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
                    or message.video
                    else "Unknown"
                )
            )
            logger.info(
                f"Received message from {message.from_user.id} on bot {self.instance_id}: {msg_content}"
            )

            user_id = message.from_user.id
            if user_id not in self.user_message_queues:
                self.user_message_queues[user_id] = []
            self.user_message_queues[user_id].append(message)

            if (
                user_id not in self.user_message_tasks
                or self.user_message_tasks[user_id].done()
            ):
                self.user_message_tasks[user_id] = asyncio.create_task(
                    self._process_user_queue(user_id)
                )

    async def _process_user_queue(self, user_id: int) -> None:
        await asyncio.sleep(1.0)
        messages = self.user_message_queues.pop(user_id, [])
        if not messages:
            return
        await self._process_user_messages_batch(user_id, messages)

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
        elif message.video:
            files.append(message.video)

        return text, files

    async def _download_and_register_files(
        self,
        files: list,
        agent_service: "AgentService",
        task_id: int,
        user_id: int,
        db: Session,
    ) -> list:
        import mimetypes
        from pathlib import Path

        from ...models.uploaded_file import UploadedFile

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

                mime_type, _ = mimetypes.guess_type(str(target_path))
                if not mime_type:
                    mime_type = "application/octet-stream"

                file_size = getattr(f, "file_size", target_path.stat().st_size)

                file_record = UploadedFile(
                    user_id=user_id,
                    task_id=task_id,
                    filename=normalized_file_name,
                    storage_path=str(target_path),
                    mime_type=mime_type,
                    file_size=file_size,
                )
                db.add(file_record)
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
        combined_text = ""
        combined_files = []

        # We'll use the last message for answering
        last_message = messages[-1]

        for msg in messages:
            text, files = await self._extract_message_content(msg)
            if text:
                if combined_text:
                    combined_text += "\n" + text
                else:
                    combined_text = text
            if files:
                combined_files.extend(files)

        text = combined_text
        files = combined_files

        if not text and not files:
            return

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
                        channel_id=self.channel_id,
                        channel_name=self.channel_name,
                    )
                    db.add(task)
                    db.commit()
                    db.refresh(task)
                    self.active_tasks[user_id] = task.id
                    self._save_active_tasks()
                    is_new_task = True
                else:
                    task.status = TaskStatus.PENDING
                    db.commit()

                agent_manager = get_agent_manager()
                agent_service = await agent_manager.get_agent_for_task(
                    int(task.id),
                    db,
                    user=user,  # type: ignore
                )

                from ...services.task_execution_context_service import (
                    load_task_execution_recovery_state,
                )

                recovery_state = await load_task_execution_recovery_state(
                    db, int(task.id)
                )  # type: ignore
                agent_service.set_execution_context_messages(
                    recovery_state.get("messages", [])
                )
                agent_service.set_recovered_skill_context(
                    recovery_state.get("skill_context")
                )

                context: dict = {}

                if files:
                    uploaded_info = await self._download_and_register_files(
                        files=files,
                        agent_service=agent_service,
                        task_id=int(task.id),  # type: ignore
                        user_id=int(user.id),  # type: ignore
                        db=db,
                    )
                    if uploaded_info:
                        file_info_list = [
                            f"[{info['name']}](file://{info['file_id']})"
                            for info in uploaded_info
                        ]
                        if text:
                            text += f"\n\n{' '.join(file_info_list)}"
                        else:
                            text = " ".join(file_info_list)
                        if is_new_task:
                            task.description = text  # type: ignore
                            if not task.title:
                                title_str = (
                                    text if len(text) <= 50 else f"{text[:50]}..."
                                )
                                task.title = title_str  # type: ignore
                            db.commit()

                        context["state"] = context.get("state", {})
                        context["state"]["file_info"] = uploaded_info

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

                with UserContext(int(user.id)):  # type: ignore
                    result = await agent_manager.execute_task(
                        agent_service=agent_service,
                        task=text,
                        context=context,
                        task_id=actual_task_id,
                        tracking_task_id=str(task.id),
                        db_session=db,
                    )

                task.status = (
                    TaskStatus.COMPLETED
                    if result.get("success", False)
                    else TaskStatus.FAILED
                )
                db.commit()

                output = result.get("output", "")

                chat_response = result.get("chat_response")
                if isinstance(chat_response, dict):
                    interactions = chat_response.get("interactions", [])
                    if interactions:
                        interaction_texts = []
                        for interaction in interactions:
                            label = interaction.get("label") or interaction.get(
                                "field", "Input"
                            )
                            options = interaction.get("options", [])
                            if options:
                                opts = []
                                for opt in options:
                                    if isinstance(opt, dict):
                                        opts.append(
                                            str(opt.get("label", opt.get("value", "")))
                                        )
                                    else:
                                        opts.append(str(opt))
                                interaction_texts.append(
                                    f"• {label}\n  Options: {', '.join(opts)}"
                                )
                            else:
                                interaction_texts.append(f"• {label}")
                        if interaction_texts:
                            output += "\n\n" + "\n".join(interaction_texts)

                if not output or not str(output).strip():
                    output = "Task completed, but no output was generated."

                output = str(output)
                output, image_refs = strip_telegram_image_refs(output)
                if not output and image_refs:
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

            finally:
                try:
                    next(db_gen)
                except StopIteration:
                    pass
        except Exception as e:
            logger.error(f"Error processing Telegram message: {e}")
            await last_message.answer(
                "Sorry, an error occurred while processing your request."
            )

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
