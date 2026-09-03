import asyncio
import errno
import hashlib
import html
import json
import logging
import mimetypes
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Coroutine,
    Dict,
    Optional,
    Sequence,
    cast,
)
from uuid import uuid4

if TYPE_CHECKING:
    from ....core.agent.service import AgentService

from aiogram import Bot, Dispatcher, F, types
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import (
    BotCommand,
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from sqlalchemy.orm import Session

from ....core.file_ref import build_file_id_ref
from ...api.chat import get_agent_manager
from ...models.database import get_session_local
from ...models.task import TaskStatus
from ...models.user import User
from ...services.channel_runtime import (
    TELEGRAM_TASK_LIST_LIMIT,
    ChannelAgentSnapshot,
    ChannelAuthorizationError,
    ChannelConfigurationError,
    DownloadedChannelFile,
    TelegramChannelTaskSnapshot,
    authorize_channel_sender,
    get_channel_owner_agent,
    list_channel_owner_agents,
    load_active_channel_configs,
    load_channel_output_files,
    load_telegram_channel_task,
    load_telegram_channel_tasks,
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
    CancelledDelivery,
    TelegramFileRef,
    TelegramImageRef,
    deliver_cancellation_safe,
    markdown_to_tg_html,
    strip_telegram_file_refs,
    strip_telegram_image_refs,
)

logger = logging.getLogger(__name__)

# tasks.id is a 64-bit signed integer in both supported backends. A /switch
# argument above this overflows the DB bind rather than missing the lookup.
_MAX_TASK_ID = 2**63 - 1

# Opening a directory to fsync it is a POSIX idiom. Windows cannot do it at all
# -- its CRT reports EACCES for os.open() on a directory -- so the barrier is
# skipped there instead of failing every save. EACCES must stay a real
# permission error on POSIX, so this is a platform check, not an errno one.
_DIRECTORY_FSYNC_SUPPORTED = os.name != "nt"

# Some filesystems and network mounts still refuse the operation. Every other
# OSError is a real durability failure and must not be reported as a success.
_UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS = frozenset(
    getattr(errno, name)
    for name in ("EINVAL", "EISDIR", "ENOSYS", "ENOTSUP", "EOPNOTSUPP")
    if hasattr(errno, name)
)


class TelegramVoiceTranscriptionError(RuntimeError):
    """Raised when a Telegram voice prompt cannot be transcribed."""


class TelegramBotInstance:
    queue_flush_delay_seconds = 1.0
    voice_transcription_timeout_seconds = 180.0
    task_list_message_limit = 3800
    stop_text_aliases = {"/stop", "/pause", "stop", "pause", "停止", "暂停"}
    agents_page_size = 8
    bot_commands = (
        BotCommand(command="new", description="Start a new task"),
        BotCommand(command="list", description="List your previous tasks"),
        BotCommand(command="switch", description="Switch to a previous task"),
        BotCommand(command="agents", description="Choose an agent"),
        BotCommand(command="stop", description="Stop the current run"),
        BotCommand(command="help", description="Show available commands"),
    )

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
        self.user_active_trace_handlers: Dict[int, TelegramTraceHandler] = {}
        self.user_preparing_executions: set[int] = set()
        self.user_stop_events: Dict[int, asyncio.Event] = {}
        # Held for the whole /switch transition. Aiogram handles updates
        # concurrently, so a plain message arriving mid-switch would otherwise
        # join the old task's queue and be discarded by the stop path.
        self.user_switch_locks: Dict[int, asyncio.Lock] = {}
        self.user_conversation_generations: Dict[int, int] = {}
        self._accepting = True
        self._ingress_stopped = False
        self._stop_lock: asyncio.Lock | None = None
        self._stop_loop: asyncio.AbstractEventLoop | None = None

        # Load active tasks state
        self.active_tasks_file = self._active_tasks_store_path(channel_id, token)
        self._legacy_active_tasks_file = Path(
            f"data/telegram_active_tasks_{instance_id}.json"
        )
        # Set when a save failed, so the next batch retries the persist.
        # Initialized before the load: a legacy-file fallback saves the durable
        # copy from inside _load_active_tasks.
        self._active_tasks_unsaved = False
        self.active_tasks = self._load_active_tasks()

        # Per-Telegram-user custom agent selection for the next conversation
        self.selected_agents_file = self._selected_agents_store_path(channel_id, token)
        self.selected_agents: Dict[int, int] = self._load_selected_agents()

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
        # Read the legacy cwd-relative file when the durable one is absent, so
        # an upgrade keeps the selection instead of losing the only evidence
        # that can claim legacy telegram_user_id IS NULL tasks.
        for path in (self.active_tasks_file, self._legacy_active_tasks_file):
            if not path.exists():
                continue
            try:
                with open(path, "r") as f:
                    # Convert string keys back to int
                    loaded = {int(k): v for k, v in json.load(f).items()}
            except Exception as e:
                logger.error(
                    f"Failed to load Telegram active tasks for {self.instance_id} "
                    f"from {path}: {e}"
                )
                continue

            if path is self._legacy_active_tasks_file:
                # Persist the durable copy BEFORE retiring the legacy file.
                # Retiring first would leave a window -- until the next
                # /switch, /new, or task creation triggers a save -- where a
                # restart finds neither file and the mapping (the sole
                # proof-of-ownership for claiming legacy tasks) is gone for
                # good. If the save fails, keep the legacy file so the next
                # restart can try again.
                self.active_tasks = loaded
                if self._save_active_tasks():
                    self._retire_legacy_active_tasks_file()
            elif self._legacy_active_tasks_file.exists():
                # The durable copy already supersedes the legacy file, so a
                # leftover one is pure hazard: were the durable file later
                # lost, the next restart would resurrect this stale mapping
                # and could hand a task to the wrong sender. No save is
                # needed first -- the durable read just succeeded.
                self._retire_legacy_active_tasks_file()
            return loaded
        return {}

    def _retire_legacy_active_tasks_file(self) -> None:
        """Rename the legacy file so a fallback read can happen at most once.

        This mapping is proof-of-ownership for claiming pre-migration tasks
        (``telegram_user_id IS NULL``). If the durable file were later lost
        while this one survived, a stale mapping would be resurrected and could
        reassign a task to the wrong current sender.
        """

        retired = self._legacy_active_tasks_file.with_suffix(
            f"{self._legacy_active_tasks_file.suffix}.migrated"
        )
        try:
            os.replace(self._legacy_active_tasks_file, retired)
        except OSError:
            logger.warning(
                "Failed to retire the legacy Telegram active-tasks file %s; "
                "a future load could resurrect a stale mapping",
                self._legacy_active_tasks_file,
                exc_info=True,
            )

    def _save_active_tasks(self) -> bool:
        """Persist the active-task mapping atomically.

        Returns True only when the mapping is durably on disk. Callers that
        confirm a selection to the user must not report success on False: a
        truncated or missing file silently resurrects the previous task after
        a restart.
        """
        tmp_path = self.active_tasks_file.with_suffix(
            f"{self.active_tasks_file.suffix}.tmp"
        )
        self._active_tasks_unsaved = True
        try:
            self.active_tasks_file.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(self.active_tasks)
            with open(tmp_path, "w") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.active_tasks_file)
            # Fsyncing the temp file does not commit the renamed directory
            # entry, so a host crash could still lose a selection reported as
            # durable. Only a genuinely unsupported operation is tolerated:
            # any other error leaves the save dirty so it is retried.
            if not _DIRECTORY_FSYNC_SUPPORTED:
                self._active_tasks_unsaved = False
                return True
            try:
                dir_fd = os.open(str(self.active_tasks_file.parent), os.O_RDONLY)
            except OSError as e:
                if e.errno not in _UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
                    raise
                logger.debug(
                    "Directory fsync unsupported for Telegram active tasks %s",
                    self.active_tasks_file.parent,
                    exc_info=True,
                )
            else:
                try:
                    os.fsync(dir_fd)
                except OSError as e:
                    if e.errno not in _UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
                        raise
                    logger.debug(
                        "Directory fsync unsupported for Telegram active tasks %s",
                        self.active_tasks_file.parent,
                        exc_info=True,
                    )
                finally:
                    os.close(dir_fd)
            self._active_tasks_unsaved = False
            return True
        except Exception as e:
            logger.error(
                f"Failed to save Telegram active tasks for {self.instance_id}: {e}"
            )
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                logger.debug(
                    "Failed to remove temporary Telegram active-task file %s",
                    tmp_path,
                    exc_info=True,
                )
            return False

    @staticmethod
    def _active_tasks_store_path(channel_id: Optional[int], token: str) -> Path:
        """Durable per-channel active-task store.

        Lives under the storage root for the same reason as the selected-agent
        store: the cwd-relative ``data/`` path is not on the persisted volume,
        so a container recreation forgot the selection. That also destroyed the
        only evidence used to claim legacy ``telegram_user_id IS NULL`` tasks.
        """
        from ....config import get_storage_root

        if channel_id is not None:
            key = f"channel_{int(channel_id)}"
        else:
            key = hashlib.sha256(token.encode()).hexdigest()[:16]
        return Path(get_storage_root()) / "telegram" / f"active_tasks_{key}.json"

    @staticmethod
    def _selected_agents_store_path(channel_id: Optional[int], token: str) -> Path:
        """Durable per-channel selection store.

        Lives under the storage root (survives container recreation, unlike a
        cwd-relative ``data/`` path) and is keyed by the stable channel id —
        falling back to a full-token hash — so distinct bots whose tokens share
        a prefix can never collide on one file.
        """
        from ....config import get_storage_root

        if channel_id is not None:
            key = f"channel_{int(channel_id)}"
        else:
            key = hashlib.sha256(token.encode()).hexdigest()[:16]
        return Path(get_storage_root()) / "telegram" / f"selected_agents_{key}.json"

    def _load_selected_agents(self) -> Dict[int, int]:
        if self.selected_agents_file.exists():
            try:
                with open(self.selected_agents_file, "r") as f:
                    return {int(k): int(v) for k, v in json.load(f).items()}
            except Exception as e:
                logger.error(
                    f"Failed to load Telegram selected agents for {self.instance_id}: {e}"
                )
        return {}

    def _save_selected_agents(self) -> bool:
        """Persist the agent selection. Returns whether it is durable."""

        try:
            self.selected_agents_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.selected_agents_file, "w") as f:
                json.dump(self.selected_agents, f)
            return True
        except Exception as e:
            logger.error(
                f"Failed to save Telegram selected agents for {self.instance_id}: {e}"
            )
            return False

    def _set_selected_agent(self, user_id: int, agent_id: Optional[int]) -> bool:
        if agent_id is None:
            self.selected_agents.pop(user_id, None)
        else:
            self.selected_agents[user_id] = agent_id
        return self._save_selected_agents()

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
                "Hi, I'm Xagent. Send me anything you'd like help with, use /new "
                "for a fresh start, or /list to find a previous task."
            )

        @self.dp.message(Command("help"))
        async def cmd_help(message: types.Message) -> None:
            if not self._accepting or message.from_user is None:
                return
            await message.answer(self._help_text())

        @self.dp.message(Command("new"))
        async def cmd_new(message: types.Message) -> None:
            if not self._accepting or message.from_user is None:
                return
            logger.info(
                f"Received /new from {message.from_user.id} on bot {self.instance_id}"
            )
            # Same barrier /switch holds: /switch suspends on an awaited
            # lookup before it writes active_tasks, so an unguarded reset here
            # would be silently overwritten when that lookup resumes.
            async with self._switch_lock_for_user(message.from_user.id):
                _, persisted = self._start_new_conversation(message.from_user.id)
                # Inside the lock: the active-task reset and the agent clear are
                # one update, and keeping them together stays correct if agent
                # persistence ever becomes async.
                if persisted:
                    # Only clear the agent once the reset is durable, so a
                    # failed save leaves the whole previous selection intact.
                    self._set_selected_agent(message.from_user.id, None)
            if not persisted:
                await message.answer(
                    "I couldn't start a new task because the change could not "
                    "be saved. The current task is still active. Please try "
                    "again."
                )
                return
            await message.answer(
                "Fresh start. Send me what you'd like to work on next."
            )

        @self.dp.message(Command("list"))
        async def cmd_list(message: types.Message) -> None:
            if not self._accepting or message.from_user is None:
                return
            await self._handle_list_command(message)

        @self.dp.message(Command("switch"))
        async def cmd_switch(message: types.Message) -> None:
            if not self._accepting or message.from_user is None:
                return
            await self._handle_switch_command(message)

        @self.dp.message(Command("agents"))
        async def cmd_agents(message: types.Message) -> None:
            if not self._accepting or message.from_user is None:
                return
            logger.info(
                f"Received /agents from {message.from_user.id} on bot {self.instance_id}"
            )
            await self._handle_agents_command(message)

        @self.dp.callback_query(F.data.startswith("agsel:"))
        async def on_agent_selected(callback: CallbackQuery) -> None:
            if not self._accepting or callback.from_user is None:
                return
            await self._handle_agent_selection_callback(callback)

        @self.dp.callback_query(F.data.startswith("agpage:"))
        async def on_agents_page(callback: CallbackQuery) -> None:
            if not self._accepting or callback.from_user is None:
                return
            await self._handle_agents_page_callback(callback)

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

    def _switch_lock_for_user(self, user_id: int) -> asyncio.Lock:
        lock = self.user_switch_locks.get(user_id)
        if lock is None:
            lock = asyncio.Lock()
            self.user_switch_locks[user_id] = lock
        return lock

    def _enqueue_user_message(self, user_id: int, message: Any) -> bool:
        if not self._accepting:
            return False

        self.user_message_queues.setdefault(user_id, []).append(message)
        task = self.user_message_tasks.get(user_id)
        if task is None or task.done():
            self._schedule_user_queue(user_id)
        return True

    @staticmethod
    def _help_text() -> str:
        return (
            "Send a message to continue the current task.\n\n"
            "/new — start a new task\n"
            "/list — list your previous tasks\n"
            "/switch &lt;task_id&gt; — switch to a task shown by /list\n"
            "/agents — choose which agent replies\n"
            "/stop — stop the current run\n"
            "/help — show this help"
        )

    @staticmethod
    def _format_task_timestamp(timestamp: datetime | None) -> str:
        """Render a task timestamp as UTC wall time.

        Columns are DateTime(timezone=True), but SQLite returns naive values
        that this project stores in UTC, while PostgreSQL returns aware values
        in the session timezone. Aware values are converted so the "UTC" label
        is never attached to local wall time.
        """

        if timestamp is None:
            return "unknown"
        if timestamp.tzinfo is not None:
            timestamp = timestamp.astimezone(timezone.utc)
        return timestamp.strftime("%Y-%m-%d %H:%M")

    @staticmethod
    def _task_list_line(
        task: TelegramChannelTaskSnapshot,
        *,
        is_active: bool,
    ) -> str:
        raw_title = " ".join((task.title or "Untitled Task").split())
        title = html.escape(raw_title[:80] + ("…" if len(raw_title) > 80 else ""))
        date_text = TelegramBotInstance._format_task_timestamp(
            task.updated_at or task.created_at
        )
        marker = "● " if is_active else ""
        return (
            f"{marker}<code>{task.task_id}</code> · {title}\n"
            f"   {html.escape(task.status)} · {date_text} UTC"
        )

    @staticmethod
    def _telegram_text_units(text: str) -> int:
        """Count the UTF-16 code units used by Telegram's message limit."""
        return len(text.encode("utf-16-le")) // 2

    def _format_task_list_messages(
        self,
        tasks: list[TelegramChannelTaskSnapshot],
        *,
        active_task_id: int | None,
    ) -> list[str]:
        if not tasks:
            return [
                "You don't have any saved Telegram tasks yet. "
                "Send a message to create one."
            ]

        # Only claim truncation when the list actually hit the limit.
        header = "<b>Your Telegram tasks</b>"
        if len(tasks) >= TELEGRAM_TASK_LIST_LIMIT:
            header += f" ({TELEGRAM_TASK_LIST_LIMIT} most recent)"
        footer = "Use <code>/switch &lt;task_id&gt;</code> to continue one."
        messages: list[str] = []
        current = header
        for task in tasks:
            line = self._task_list_line(
                task,
                is_active=task.task_id == active_task_id,
            )
            candidate = f"{current}\n\n{line}"
            if self._telegram_text_units(candidate) > self.task_list_message_limit:
                messages.append(current)
                current = f"{header}\n\n{line}"
            else:
                current = candidate

        footer_candidate = f"{current}\n\n{footer}"
        if self._telegram_text_units(footer_candidate) > self.task_list_message_limit:
            messages.append(current)
            current = footer
        else:
            current = footer_candidate
        messages.append(current)
        return messages

    async def _handle_list_command(self, message: types.Message) -> None:
        if not self._accepting or message.from_user is None:
            return
        telegram_user_id = int(message.from_user.id)
        try:
            active_task_id = self.active_tasks.get(telegram_user_id)
            tasks = await load_telegram_channel_tasks(
                channel_id=self.channel_id,
                external_user_id=str(telegram_user_id),
                active_task_id=active_task_id,
            )
            for response in self._format_task_list_messages(
                list(tasks),
                active_task_id=active_task_id if active_task_id != -1 else None,
            ):
                await message.answer(response)
        except ChannelAuthorizationError:
            await message.answer("🚫 You are not authorized to use this bot.")
        except ChannelConfigurationError:
            await message.answer("This bot is inactive or not correctly configured.")
        except Exception:
            # Match the plain-message path: never leave the command silent.
            logger.error("Failed to list Telegram tasks", exc_info=True)
            await message.answer(
                "Sorry, an error occurred while processing your request."
            )

    @staticmethod
    def _switch_task_id(command_text: str | None) -> int | None:
        parts = (command_text or "").strip().split()
        # Require ASCII digits. isdigit() alone accepts "²", which int() then
        # rejects with an uncaught ValueError, and other Unicode digit forms
        # ("٣", "１２") would silently resolve to an unrelated task id.
        if len(parts) != 2 or not (parts[1].isascii() and parts[1].isdigit()):
            return None
        task_id = int(parts[1])
        # Bound to the column's range: a larger value overflows the SQLite and
        # PostgreSQL integer binds, surfacing a DataError instead of the normal
        # "task not found" reply.
        if not 0 < task_id <= _MAX_TASK_ID:
            return None
        return task_id

    async def _handle_switch_command(self, message: types.Message) -> None:
        if not self._accepting or message.from_user is None:
            return
        task_id = self._switch_task_id(message.text)
        if task_id is None:
            await message.answer(
                "Usage: <code>/switch &lt;task_id&gt;</code>\n"
                "Use /list to see your task IDs."
            )
            return

        telegram_user_id = int(message.from_user.id)
        # Serialize concurrent /switch commands from the same sender: two
        # transitions interleaving across their awaited lookups would race on
        # active_tasks and selected_agents. The message path deliberately does
        # not take this barrier -- a message racing a switch is handled by the
        # fences in _process_user_messages_batch, which reply to the user
        # instead of being serialized behind network I/O.
        async with self._switch_lock_for_user(telegram_user_id):
            confirmation = await self._switch_to_task(
                message, telegram_user_id, task_id
            )
        # The confirmation is a Telegram round trip (60s default timeout) and
        # the state mutation is already complete, so it must not extend the
        # lock -- matching /new and both /agents callback sites. Concurrent
        # /switch commands serialize on the lock but their confirmations are
        # unordered sends, so only confirm a selection that is still current:
        # otherwise "Switched to task A" can land last while B is active.
        if (
            confirmation is not None
            and self.active_tasks.get(telegram_user_id) == task_id
        ):
            await message.answer(confirmation)

    async def _switch_to_task(
        self,
        message: types.Message,
        telegram_user_id: int,
        task_id: int,
    ) -> str | None:
        """Perform the switch; error replies are sent here, under the lock.

        Returns the success confirmation for the caller to send after the lock
        is released, or None when an error reply was already sent.
        """

        try:
            task = await load_telegram_channel_task(
                channel_id=self.channel_id,
                external_user_id=str(telegram_user_id),
                task_id=task_id,
                active_task_id=self.active_tasks.get(telegram_user_id),
            )
            if task is None:
                await message.answer(
                    "Task not found or not accessible. Use /list to see your tasks."
                )
                return None

            # tasks.agent_id has no FK and agent deletion does not check for
            # referencing tasks, so the binding can dangle. Catch it here:
            # otherwise /switch reports success and the next message silently
            # evicts the user into a fresh task, undoing the switch.
            if task.agent_id is not None:
                bound_agent = await get_channel_owner_agent(
                    channel_id=self.channel_id,
                    external_user_id=str(telegram_user_id),
                    agent_id=task.agent_id,
                )
                if bound_agent is None:
                    await message.answer(
                        f"Task <code>{task_id}</code> is bound to an agent that "
                        "is no longer available, so it can't be resumed. "
                        "Use /new to start a fresh conversation."
                    )
                    return None

            if self.active_tasks.get(telegram_user_id) == task_id:
                # A dequeued batch is active work too, even before its
                # execution is registered.
                is_busy = (
                    telegram_user_id in self.user_active_executions
                    or telegram_user_id in self.user_preparing_executions
                )
                running_suffix = " and is still working" if is_busy else ""
                await message.answer(
                    f"Task <code>{task_id}</code> is already active{running_suffix}: "
                    f"{html.escape(task.title)}"
                )
                return None

            # Persistence is the commit point: stopping the old conversation
            # discards its queue, cancels its trace handler, and pauses its
            # run, none of which can be undone by restoring the mapping.
            #
            # The agent binding moves with the task. prepare_channel_task()
            # discards a task whose agent_id does not match the selection, so
            # keeping the old conversation's agent would silently create a new
            # task on the next message.
            previous_task_id = self.active_tasks.get(telegram_user_id)
            previous_agent_id = self.selected_agents.get(telegram_user_id)
            self.active_tasks[telegram_user_id] = task_id
            agent_persisted = True
            if previous_agent_id != task.agent_id:
                agent_persisted = self._set_selected_agent(
                    telegram_user_id, task.agent_id
                )
            if not agent_persisted or not self._save_active_tasks():
                if previous_task_id is None:
                    self.active_tasks.pop(telegram_user_id, None)
                else:
                    self.active_tasks[telegram_user_id] = previous_task_id
                self._save_active_tasks()
                if previous_agent_id != task.agent_id:
                    self._set_selected_agent(telegram_user_id, previous_agent_id)
                await message.answer(
                    "I couldn't save the switch, so the previous task is still "
                    "active. Please try again."
                )
                return None

            # Fence in-flight preparation before requesting the stop. A batch
            # awaiting prepare_channel_task() would otherwise return with a
            # still-matching generation and overwrite this confirmed selection
            # in its is_new_task branch before consuming the stop event.
            self.user_conversation_generations[telegram_user_id] = (
                self._conversation_generation(telegram_user_id) + 1
            )
            self._request_current_conversation_stop(
                telegram_user_id,
                reason="Telegram task switch requested",
            )
            return (
                f"Switched to task <code>{task_id}</code>: "
                f"{html.escape(task.title)}\n"
                "Send a message to continue it."
            )
        except ChannelAuthorizationError:
            await message.answer("🚫 You are not authorized to use this bot.")
        except ChannelConfigurationError:
            await message.answer("This bot is inactive or not correctly configured.")
        except Exception:
            # Match the plain-message path: never leave the command silent.
            logger.error("Failed to switch Telegram task %s", task_id, exc_info=True)
            await message.answer(
                "Sorry, an error occurred while processing your request."
            )
        return None

    def _schedule_user_queue(self, user_id: int) -> bool:
        if not self._accepting:
            return False
        self.user_message_tasks[user_id] = asyncio.create_task(
            self._process_user_queue(user_id)
        )
        return True

    def _conversation_generation(self, user_id: int) -> int:
        return self.user_conversation_generations.get(user_id, 0)

    def _start_new_conversation(self, user_id: int) -> tuple[bool, bool]:
        """Reset the conversation. Returns (stopped_something, persisted)."""

        # Persist before fencing or stopping: both are irreversible, so a
        # mapping that will not survive a restart must not tear the old
        # conversation down.
        previous_task_id = self.active_tasks.get(user_id)
        self.active_tasks[user_id] = -1
        if not self._save_active_tasks():
            if previous_task_id is None:
                self.active_tasks.pop(user_id, None)
            else:
                self.active_tasks[user_id] = previous_task_id
            self._save_active_tasks()
            return False, False

        # Bumping the generation fences out any in-flight preparation for the
        # previous conversation: its result must not overwrite the new state.
        self.user_conversation_generations[user_id] = (
            self._conversation_generation(user_id) + 1
        )
        stopped = self._request_current_conversation_stop(
            user_id, reason="new Telegram conversation requested"
        )
        return stopped, True

    def _stop_current_conversation(self, user_id: int) -> bool:
        # discard_output=False: /stop pauses the run but keeps the user in this
        # conversation, so the partial answer must still be delivered.
        return self._request_current_conversation_stop(
            user_id,
            reason="Telegram stop requested",
            discard_output=False,
        )

    def _request_current_conversation_stop(
        self, user_id: int, *, reason: str, discard_output: bool = True
    ) -> bool:
        queued_messages = self.user_message_queues.pop(user_id, None)
        active_trace_handler = self._active_trace_handlers().get(user_id)
        if active_trace_handler is not None:
            active_trace_handler.cancel(discard_output=discard_output)
        stopped = self._stop_user_active_execution(user_id, reason=reason)
        preparing = user_id in self.user_preparing_executions
        if preparing and not stopped:
            self._request_user_stop(user_id)
        return bool(queued_messages) or stopped or preparing

    def _active_trace_handlers(self) -> Dict[int, TelegramTraceHandler]:
        return self.user_active_trace_handlers

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

    async def _settle_fenced_turn(
        self,
        managed_lease: ManagedTaskLease,
        *,
        task_id: int,
        reply_to: types.Message,
        already_persisted: bool = False,
    ) -> None:
        """Settle a turn fenced out by a /switch, /new, or /stop, with a reply.

        A fence must never swallow the message. The claim is always finalized to
        PAUSED: the turn is no longer running, and PAUSED is resumable. Notably
        ``ManagedTaskLease.close()`` must not be used here -- it maps a RUNNING
        task to FAILED, which would corrupt the very conversation the user is in.

        ``already_persisted`` distinguishes the wording only. Once the message is
        in conversation history, asking the user to resend would duplicate it, so
        they are told it was received instead.

        This helper does not raise -- except asyncio.CancelledError, which is a
        BaseException and deliberately propagates so cancellation is never
        absorbed. The reply is the real contract: a settle failure has no
        useful handling in the caller -- settling raises TaskLeaseLostError on
        the ordinary unhealthy-heartbeat/TTL path, and letting any exception
        unwind would land in the generic error handler, which finalizes the
        task FAILED (this docstring's own "never") and sends a second,
        contradictory reply. A failed reply send must likewise not replace the
        settle outcome; it is logged instead.
        """

        try:
            await self._finalize_requested_stop(managed_lease, task_id=task_id)
        except TaskLeaseLostError:
            # The routine path, not a failure: the lease was lost either to
            # another runner (which now owns settlement) or to the unhealthy-
            # heartbeat TTL retention (recovery settles the run). Neither
            # warrants a warning with a traceback.
            logger.info(
                "Fenced Telegram task %s lost its lease before the fence "
                "could settle it; settlement belongs to the new owner or "
                "TTL recovery",
                task_id,
            )
        except Exception:
            logger.warning(
                "Failed to settle fenced Telegram task %s",
                task_id,
                exc_info=True,
            )
        try:
            if already_persisted:
                await reply_to.answer(
                    "Your message was received but the conversation was "
                    "interrupted before I could answer. Send anything to continue."
                )
            else:
                await reply_to.answer(
                    "That message wasn't sent because the conversation was "
                    "paused. Please send it again to continue."
                )
        except Exception:
            logger.warning(
                "Failed to deliver the fence notice for Telegram task %s",
                task_id,
                exc_info=True,
            )

    def _is_stop_request_text(self, text: str) -> bool:
        normalized = text.strip().lower()
        if normalized.startswith("/"):
            normalized = normalized.split()[0].split("@", 1)[0]
        return normalized in self.stop_text_aliases

    @staticmethod
    def _build_agents_keyboard(
        agents: Sequence[ChannelAgentSnapshot],
        page: int,
        *,
        selected_agent_id: Optional[int],
    ) -> InlineKeyboardMarkup:
        page_size = TelegramBotInstance.agents_page_size
        total_pages = max(1, (len(agents) + page_size - 1) // page_size)
        page = max(0, min(page, total_pages - 1))
        page_agents = agents[page * page_size : (page + 1) * page_size]

        default_label = "Default assistant"
        if selected_agent_id is None:
            default_label += " ✓"
        rows = [
            [InlineKeyboardButton(text=default_label, callback_data="agsel:default")]
        ]
        for agent in page_agents:
            label = agent.name if len(agent.name) <= 60 else f"{agent.name[:57]}..."
            if agent.agent_id == selected_agent_id:
                label += " ✓"
            rows.append(
                [
                    InlineKeyboardButton(
                        text=label, callback_data=f"agsel:{agent.agent_id}"
                    )
                ]
            )
        if total_pages > 1:
            nav_row = []
            if page > 0:
                nav_row.append(
                    InlineKeyboardButton(
                        text="⬅ Prev", callback_data=f"agpage:{page - 1}"
                    )
                )
            if page < total_pages - 1:
                nav_row.append(
                    InlineKeyboardButton(
                        text="Next ➡", callback_data=f"agpage:{page + 1}"
                    )
                )
            if nav_row:
                rows.append(nav_row)
        return InlineKeyboardMarkup(inline_keyboard=rows)

    async def _handle_agents_command(self, message: types.Message) -> None:
        user_id = message.from_user.id  # type: ignore[union-attr]
        try:
            agents = await list_channel_owner_agents(
                channel_id=self.channel_id,
                external_user_id=str(user_id),
            )
        except ChannelAuthorizationError:
            await message.answer("🚫 You are not authorized to use this bot.")
            return
        except ChannelConfigurationError:
            await message.answer("This bot is inactive or not correctly configured.")
            return

        if not agents:
            await message.answer(
                "You don't have any custom agents yet. Create one in the "
                "Agent Builder, then run /agents again."
            )
            return

        await message.answer(
            "Choose the agent for your next conversation:",
            reply_markup=self._build_agents_keyboard(
                agents, 0, selected_agent_id=self.selected_agents.get(user_id)
            ),
        )

    async def _handle_agent_selection_callback(self, callback: CallbackQuery) -> None:
        user_id = callback.from_user.id
        payload = (callback.data or "").removeprefix("agsel:")
        try:
            if payload == "default":
                # Reset first: applying the selection before the reset is
                # durable would leave it active after a reported failure, so
                # the next message would start a fresh task with it anyway.
                async with self._switch_lock_for_user(user_id):
                    _, persisted = self._start_new_conversation(user_id)
                    # Inside the lock: reset and selection are one update.
                    if persisted:
                        self._set_selected_agent(user_id, None)
                if not persisted:
                    await callback.answer(
                        "I couldn't save that change. Please try again.",
                        show_alert=True,
                    )
                    return
                await callback.answer("Default assistant selected")
                confirmation = (
                    "Default assistant selected. "
                    "Send a message to start a fresh conversation."
                )
            else:
                agent = await get_channel_owner_agent(
                    channel_id=self.channel_id,
                    external_user_id=str(user_id),
                    agent_id=int(payload),
                )
                if agent is None:
                    await callback.answer(
                        "That agent is no longer available.", show_alert=True
                    )
                    return
                async with self._switch_lock_for_user(user_id):
                    _, persisted = self._start_new_conversation(user_id)
                    # Inside the lock: reset and selection are one update.
                    if persisted:
                        self._set_selected_agent(user_id, agent.agent_id)
                if not persisted:
                    await callback.answer(
                        "I couldn't save that change. Please try again.",
                        show_alert=True,
                    )
                    return
                # answerCallbackQuery rejects texts over 200 characters
                toast = f"{agent.name} selected"
                if len(toast) > 200:
                    toast = f"{toast[:197]}..."
                await callback.answer(toast)
                confirmation = (
                    f"Agent selected: {html.escape(agent.name)}. "
                    "Send a message to start a fresh conversation. "
                    "Use /new to go back to the default assistant."
                )
        except (ChannelAuthorizationError, ChannelConfigurationError):
            await callback.answer(
                "You are not authorized to use this bot.", show_alert=True
            )
            return
        except ValueError:
            await callback.answer()
            return

        message = callback.message
        if isinstance(message, types.Message):
            try:
                # Dropping reply_markup removes the stale keyboard
                await message.edit_text(confirmation)
            except Exception as e:
                if "message is not modified" not in str(e).lower():
                    logger.warning(
                        "Failed to edit Telegram agent keyboard message: %s", e
                    )

    async def _handle_agents_page_callback(self, callback: CallbackQuery) -> None:
        user_id = callback.from_user.id
        payload = (callback.data or "").removeprefix("agpage:")
        try:
            page = int(payload)
        except ValueError:
            await callback.answer()
            return

        try:
            agents = await list_channel_owner_agents(
                channel_id=self.channel_id,
                external_user_id=str(user_id),
            )
        except (ChannelAuthorizationError, ChannelConfigurationError):
            await callback.answer(
                "You are not authorized to use this bot.", show_alert=True
            )
            return

        message = callback.message
        if isinstance(message, types.Message):
            try:
                await message.edit_reply_markup(
                    reply_markup=self._build_agents_keyboard(
                        agents,
                        page,
                        selected_agent_id=self.selected_agents.get(user_id),
                    )
                )
            except Exception as e:
                if "message is not modified" not in str(e).lower():
                    logger.warning(
                        "Failed to update Telegram agents keyboard page: %s", e
                    )
        await callback.answer()

    def _prune_idle_user_state(self, user_id: int) -> None:
        """Drop per-user bookkeeping once the sender has no work in flight.

        These dicts are keyed by Telegram user id, so a long-lived bot serving
        many senders would otherwise accumulate one entry each for the process
        lifetime. Only state that is meaningless while idle is removed: the
        conversation generation stays, because it fences turns across resets.

        ``user_switch_locks`` is deliberately never pruned here. ``Lock.release()``
        marks the lock unlocked and only *schedules* the next waiter, so a lock
        with a queued waiter still reports ``locked() is False``. Dropping it in
        that window would hand the next command a brand-new Lock and break the
        mutual exclusion between /switch, /new and the /agents callbacks. One
        Lock per sender is far cheaper than that race; it is released in bulk by
        the shutdown drain.
        """

        if user_id in self.user_preparing_executions:
            return
        if user_id in self.user_active_executions:
            return
        if self._active_trace_handlers().get(user_id) is not None:
            return
        if self.user_message_tasks.get(user_id) is not None:
            return

        stop_event = self.user_stop_events.get(user_id)
        if stop_event is not None and not stop_event.is_set():
            self.user_stop_events.pop(user_id, None)

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
                self._prune_idle_user_state(user_id)
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
        # Mark preparation before the first await so /new, /stop, and /switch
        # cannot miss a batch that has already been dequeued.
        self.user_preparing_executions.add(user_id)
        self._clear_user_stop_request(user_id)
        message_contents: list[tuple[types.Message, str, list]] = []
        files = []

        # We'll use the last message for answering
        last_message = messages[-1]

        try:
            for msg in messages:
                message_text, message_files = await self._extract_message_content(msg)
                message_contents.append((msg, message_text, message_files))
                files.extend(message_files)
        except Exception:
            self.user_preparing_executions.discard(user_id)
            self._clear_user_stop_request(user_id)
            raise

        text = self._compose_prompt_text(message_contents, {})
        voice_file_ids = [
            str(voice.file_id)
            for msg in messages
            if (voice := getattr(msg, "voice", None)) is not None
        ]

        if not text and not files:
            self.user_preparing_executions.discard(user_id)
            self._clear_user_stop_request(user_id)
            return

        # preparing/stop state is already marked at the top of this method, so
        # that /new, /stop, and /switch cannot miss an already-dequeued batch.
        conversation_generation = self._conversation_generation(user_id)
        claimed_task_id: int | None = None
        managed_lease: ManagedTaskLease | None = None
        voice_asr_model: Any | None = None
        tg_handler: TelegramTraceHandler | None = None
        agent_service: Any | None = None
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
                    agent_id=self.selected_agents.get(user_id),
                )
            except ChannelAuthorizationError:
                await last_message.answer("🚫 You are not authorized to use this bot.")
                return
            except ChannelConfigurationError:
                await last_message.answer(
                    "This bot is inactive or not correctly configured."
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

            # The user switched conversations (agent selection or /new) while
            # this turn was being prepared: settle the stale claim without
            # touching active_tasks or selected_agents.
            if self._conversation_generation(user_id) != conversation_generation:
                await self._settle_fenced_turn(
                    managed_lease,
                    task_id=task_id,
                    reply_to=last_message,
                )
                return

            if is_new_task:
                self.active_tasks[user_id] = task_id
                if not self._save_active_tasks():
                    # The row is already committed, so rolling the mapping back
                    # would orphan it. Mark it dirty instead: without this the
                    # selection stays non-durable forever, because later
                    # messages see an existing task and never reach this branch.
                    logger.warning(
                        "Telegram active task %s for user %s is not durable yet; "
                        "will retry persisting the selection",
                        task_id,
                        user_id,
                    )
            elif self._active_tasks_unsaved:
                # Retry a previously failed activation: later messages see an
                # existing task, so the branch above never runs again.
                self._save_active_tasks()

            if prepared_task.requested_agent_missing:
                self._set_selected_agent(user_id, None)
                await last_message.answer(
                    "The selected agent is no longer available, so I'm using "
                    "the default assistant for this conversation."
                )

            if self._consume_user_stop_request(user_id):
                await self._settle_fenced_turn(
                    managed_lease,
                    task_id=task_id,
                    reply_to=last_message,
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
            # Same nudge as the websocket/A2A/v1 resume paths: a cached
            # AgentService whose tools were already built (e.g. paused
            # waiting for the user to connect an app) would otherwise keep
            # its stale MCP config forever, and a Telegram reply is one of
            # the ways a connect_apps pause gets answered. Gated on
            # prior_status (this task's status just before this claim, not
            # its now-RUNNING status) - an ordinary continuing message has
            # no reason to believe connector state changed, and invalidating
            # a warm cached agent's tools on every message forces a full
            # MCP/OAuth rebuild each time instead of only on a genuine
            # resume from a connect_apps pause.
            if prepared_task.prior_status in {
                TaskStatus.PAUSED,
                TaskStatus.WAITING_FOR_USER,
            }:
                agent_manager.refresh_connector_runtime_tools(task_id)
            agent_service.set_conversation_history(
                [dict(message) for message in setup_snapshot.conversation_history],
                watermark=setup_snapshot.conversation_watermark,
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
                await self._settle_fenced_turn(
                    managed_lease,
                    task_id=task_id,
                    reply_to=last_message,
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
                await self._settle_fenced_turn(
                    managed_lease,
                    task_id=task_id,
                    reply_to=last_message,
                )
                return

            await persist_channel_user_message(
                task_id=task_id,
                user_id=owner_user_id,
                content=display_message,
                attachments=persisted_attachments or None,
                turn_id=message_turn_id,
            )

            # Persisting the user message is awaited, so consume a stop that
            # landed during it. Otherwise the abandoned conversation gets an
            # orphaned "I'm working on this" message after the switch.
            if self._consume_user_stop_request(user_id):
                await self._settle_fenced_turn(
                    managed_lease,
                    task_id=task_id,
                    reply_to=last_message,
                    already_persisted=True,
                )
                return

            loading_msg = await last_message.answer(
                "Got it, I'm working on this now.\n"
                "<i>I'll update this message as I make progress.</i>",
                parse_mode=ParseMode.HTML,
            )

            # That send is awaited too, and no trace handler exists yet to
            # cancel, so a stop landing during it would only be consumed after
            # the stale message was already delivered. Remove it on stopping.
            if self._consume_user_stop_request(user_id):
                try:
                    await loading_msg.delete()
                except Exception:
                    logger.debug(
                        "Failed to remove the Telegram loading message for "
                        "stopped task %s",
                        task_id,
                        exc_info=True,
                    )
                await self._settle_fenced_turn(
                    managed_lease,
                    task_id=task_id,
                    reply_to=last_message,
                    already_persisted=True,
                )
                return

            tg_handler = TelegramTraceHandler(
                task_id,
                self.bot,
                last_message.chat.id,
                message_id=loading_msg.message_id,
            )
            self._active_trace_handlers()[user_id] = tg_handler
            agent_service.tracer.add_handler(tg_handler)

            from ...user_isolated_memory import UserContext

            actual_task_id = str(task_id)
            active_execution = (task_id, agent_service)
            self.user_active_executions[user_id] = active_execution

            try:
                if self._consume_user_stop_request(user_id):
                    await self._settle_fenced_turn(
                        managed_lease,
                        task_id=task_id,
                        reply_to=last_message,
                        already_persisted=True,
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
                agent_service.tracer.remove_handler(tg_handler)

            projection = project_execution_result_for_channel(result)
            if not await managed_lease.finalize_result(
                status=projection.task_status,
                assistant_content=projection.transcript_content,
                interactions=projection.interactions,
                message_type=projection.message_type,
                error_message=projection.diagnostic_error,
                execution_result=result,
            ):
                raise TaskLeaseLostError(
                    f"task {task_id} ownership changed before Telegram result"
                )

            # Only skip when the user actually left this conversation
            # (/switch, /new). A /stop leaves them here waiting to read the
            # partial answer, which they got before task switching existed.
            if tg_handler.discard_output:
                logger.info(
                    "Skipping Telegram delivery for abandoned execution of task %s "
                    "for user %s",
                    task_id,
                    user_id,
                )
                return

            output, image_refs, file_refs = self._extract_telegram_output_refs(
                projection.visible_text,
            )
            if not output and (image_refs or file_refs):
                output = "Task completed."

            def is_cancelled() -> bool:
                # discard_output, not cancelled: a /stop must still deliver
                # (and keep) this answer for the user who is still here.
                return bool(tg_handler is not None and tg_handler.discard_output)

            max_len = 4000
            text_chunks = [
                output[i : i + max_len] for i in range(0, len(output), max_len)
            ]
            # An empty output with no attachments yields no chunks, and the
            # sends below index [0] unconditionally. The resulting IndexError
            # lands in a handler that can no longer reply -- the lease is
            # already settled -- so the user would get nothing at all. The
            # placeholder must be non-empty: Telegram rejects empty text with
            # "Bad Request: message text is empty", which would strand the
            # loading message as the permanent final state. Reuse the
            # attachment-only wording so both no-text paths converge.
            if not text_chunks:
                text_chunks = ["Task completed."]

            # Every send below is an awaited round trip, so a cancellation can
            # land mid-flight. deliver_cancellation_safe() re-checks afterwards
            # and removes a late success, raising CancelledDelivery so the rest
            # of the output sequence is abandoned.
            async def delete_loading(_result: Any) -> None:
                await loading_msg.delete()

            async def delete_answer(msg: Any) -> None:
                await msg.delete()

            try:
                try:
                    html_chunk0 = markdown_to_tg_html(text_chunks[0])
                    await deliver_cancellation_safe(
                        lambda: loading_msg.edit_text(
                            html_chunk0, parse_mode=ParseMode.HTML
                        ),
                        is_cancelled=is_cancelled,
                        delete=delete_loading,
                        description=f"final text for task {task_id}",
                    )
                except CancelledDelivery:
                    return
                except Exception as e:
                    if "message is not modified" not in str(e).lower():
                        try:
                            await deliver_cancellation_safe(
                                lambda: loading_msg.edit_text(text_chunks[0]),
                                is_cancelled=is_cancelled,
                                delete=delete_loading,
                                description=f"final text for task {task_id}",
                            )
                        except CancelledDelivery:
                            return
                        except Exception as e2:
                            if "message is not modified" not in str(e2).lower():
                                logger.warning(f"Failed to edit message: {e2}")

                for chunk in text_chunks[1:]:
                    html_chunk = markdown_to_tg_html(chunk)
                    try:
                        await deliver_cancellation_safe(
                            lambda: last_message.answer(
                                html_chunk, parse_mode=ParseMode.HTML
                            ),
                            is_cancelled=is_cancelled,
                            delete=delete_answer,
                            description=f"output chunk for task {task_id}",
                        )
                    except CancelledDelivery:
                        return
                    except Exception:
                        await deliver_cancellation_safe(
                            lambda: last_message.answer(chunk),
                            is_cancelled=is_cancelled,
                            delete=delete_answer,
                            description=f"output chunk for task {task_id}",
                        )
            except CancelledDelivery:
                return

            try:
                if image_refs:
                    failed_image_refs = await self._send_output_images(
                        image_refs=image_refs,
                        user_id=owner_user_id,
                        task_id=task_id,
                        reply_to=last_message,
                        is_cancelled=is_cancelled,
                    )
                    if failed_image_refs:
                        await deliver_cancellation_safe(
                            lambda: self._send_image_fallback_message(
                                image_refs=failed_image_refs,
                                reply_to=last_message,
                            ),
                            is_cancelled=is_cancelled,
                            delete=lambda msg: msg.delete(),
                            description=f"image fallback for task {task_id}",
                        )

                if file_refs:
                    failed_file_refs = await self._send_output_files(
                        file_refs=file_refs,
                        user_id=owner_user_id,
                        task_id=task_id,
                        reply_to=last_message,
                        is_cancelled=is_cancelled,
                    )
                    if failed_file_refs:
                        await deliver_cancellation_safe(
                            lambda: self._send_file_fallback_message(
                                file_refs=failed_file_refs,
                                reply_to=last_message,
                            ),
                            is_cancelled=is_cancelled,
                            delete=lambda msg: msg.delete(),
                            description=f"file fallback for task {task_id}",
                        )
            except CancelledDelivery:
                return
        except TaskLeaseLostError:
            logger.warning(
                "Telegram execution lost task %s lease; skipping stale result",
                claimed_task_id,
            )
        except Exception as e:
            logger.error(f"Error processing Telegram message: {e}")
            # A /stop, /new, or /switch can land during awaited setup, before
            # tg_handler exists. Without this the run is finalized FAILED and a
            # stale error reaches the conversation the user already left.
            handler_cancelled = tg_handler is not None and tg_handler.cancelled
            stop_requested = handler_cancelled or self._consume_user_stop_request(
                user_id
            )
            if stop_requested and managed_lease is not None:
                try:
                    await self._finalize_requested_stop(
                        managed_lease,
                        task_id=claimed_task_id if claimed_task_id is not None else -1,
                    )
                except Exception:
                    logger.warning(
                        "Failed to finalize paused Telegram task %s after stop",
                        claimed_task_id,
                        exc_info=True,
                    )
                return
            if stop_requested and managed_lease is None:
                # No lease means setup itself failed, so there is no stale
                # conversation to protect -- only a real error the user has not
                # been told about. A pending stop must not silence it.
                logger.warning(
                    "Telegram turn for user %s failed during setup while a stop "
                    "was pending",
                    user_id,
                    exc_info=True,
                )
            if managed_lease is not None:
                try:
                    finalized = await managed_lease.finalize_result(
                        status=TaskStatus.FAILED,
                        error_message=str(e),
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
            if tg_handler is not None:
                active_trace_handlers = self._active_trace_handlers()
                if active_trace_handlers.get(user_id) is tg_handler:
                    active_trace_handlers.pop(user_id, None)

                # Backstop: the handler is attached before the try/finally that
                # detaches it, so a raise in between would otherwise leave it
                # on the tracer for the process lifetime. remove_handler is a
                # no-op when the inner finally already detached it.
                if agent_service is not None:
                    agent_service.tracer.remove_handler(tg_handler)

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
        is_cancelled: Callable[[], bool],
    ) -> list[TelegramImageRef]:
        ordered_file_ids = list(dict.fromkeys(ref.file_id for ref in image_refs))
        failed_refs: list[TelegramImageRef] = []

        file_records = await load_channel_output_files(
            file_ids=ordered_file_ids,
            user_id=user_id,
            task_id=task_id,
        )
        file_record_by_id = {record.file_id: record for record in file_records}

        # Loading the records is awaited, so re-check before the first send.
        if is_cancelled():
            return []

        sent_file_ids: set[str] = set()
        for image_ref in image_refs:
            if is_cancelled():
                return []
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
                await deliver_cancellation_safe(
                    lambda: reply_to.answer_photo(
                        FSInputFile(image_path), caption=caption or None
                    ),
                    is_cancelled=is_cancelled,
                    delete=lambda msg: msg.delete(),
                    description=f"output image {image_ref.file_id}",
                )
            except CancelledDelivery:
                # The late attachment is already removed. Abandon the rest
                # instead of reporting them as failures, which would emit a
                # fallback message into the conversation the user left.
                return []
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
    ) -> Any:
        subject = "image" if len(image_refs) == 1 else "images"
        lines = [
            f"I couldn't send the {subject} through Telegram, but the file reference is still available:"
        ]
        for image_ref in image_refs:
            label = image_ref.alt_text or "image"
            lines.append(f"- {label}: file:{image_ref.file_id}")
        text = "\n".join(lines)
        # Returned so a caller whose cancellation lands mid-send can remove
        # this notice instead of stranding it in an abandoned conversation.
        try:
            return await reply_to.answer(
                markdown_to_tg_html(text), parse_mode=ParseMode.HTML
            )
        except Exception:
            return await reply_to.answer(text)

    async def _send_output_files(
        self,
        *,
        file_refs: list[TelegramFileRef],
        user_id: int,
        task_id: int,
        reply_to: types.Message,
        is_cancelled: Callable[[], bool],
    ) -> list[TelegramFileRef]:
        ordered_file_ids = list(dict.fromkeys(ref.file_id for ref in file_refs))
        failed_refs: list[TelegramFileRef] = []

        file_records = await load_channel_output_files(
            file_ids=ordered_file_ids,
            user_id=user_id,
            task_id=task_id,
        )
        file_record_by_id = {record.file_id: record for record in file_records}

        # Loading the records is awaited, so re-check before the first send.
        if is_cancelled():
            return []

        sent_file_ids: set[str] = set()
        for file_ref in file_refs:
            if is_cancelled():
                return []
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
                await deliver_cancellation_safe(
                    lambda: reply_to.answer_document(
                        FSInputFile(file_path), caption=caption or None
                    ),
                    is_cancelled=is_cancelled,
                    delete=lambda msg: msg.delete(),
                    description=f"output file {file_ref.file_id}",
                )
            except CancelledDelivery:
                # The late attachment is already removed. Abandon the rest
                # instead of reporting them as failures, which would emit a
                # fallback message into the conversation the user left.
                return []
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
    ) -> Any:
        subject = "file" if len(file_refs) == 1 else "files"
        lines = [
            f"I couldn't send the {subject} through Telegram, but the file reference is still available:"
        ]
        for file_ref in file_refs:
            label = file_ref.label or "file"
            lines.append(f"- {label}: file:{file_ref.file_id}")
        text = "\n".join(lines)
        # Returned so a caller whose cancellation lands mid-send can remove
        # this notice instead of stranding it in an abandoned conversation.
        try:
            return await reply_to.answer(
                markdown_to_tg_html(text), parse_mode=ParseMode.HTML
            )
        except Exception:
            return await reply_to.answer(text)

    async def start(self) -> None:
        if not self._accepting:
            return
        try:
            # Drop pending updates to ignore messages sent while the bot was offline/inactive
            await self.bot.delete_webhook(drop_pending_updates=True)
            if not self._accepting:
                return
            try:
                await self.bot.set_my_commands(list(self.bot_commands))
            except Exception:
                logger.warning(
                    "Failed to register Telegram command menu for %s",
                    self.instance_id,
                    exc_info=True,
                )
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
            self._active_trace_handlers().clear()
            self.user_preparing_executions.clear()
            self.user_stop_events.clear()
            self.user_conversation_generations.clear()
            self.user_switch_locks.clear()

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
        # Channel CRUD endpoints fire sync as a background task, so two syncs
        # can interleave. _stop_bot_for_token awaits the shutdown drain and
        # only removes the bot from self.bots afterwards, so a second sync
        # entering that window sees a token that is still present but already
        # being torn down: it skips starting it, the first sync completes the
        # removal, and a re-enabled channel ends up neither running nor
        # tracked until some later sync happens to run.
        self._sync_lock = asyncio.Lock()

    async def start(self) -> None:
        await self._sync_bots_async()

    async def stop(self) -> None:
        tokens = list(self.bots.keys())
        for token in tokens:
            await self._stop_bot_for_token(token)

    async def _sync_bots_async(self) -> None:
        async with self._sync_lock:
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
