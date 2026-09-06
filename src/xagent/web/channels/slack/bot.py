from __future__ import annotations

import asyncio
import html
import json
import logging
import mimetypes
import re
from collections import deque
from contextlib import suppress
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import httpx
from slack_sdk.socket_mode.aiohttp import SocketModeClient
from slack_sdk.socket_mode.async_client import AsyncBaseSocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse
from slack_sdk.web.async_client import AsyncWebClient

from ....config import get_slack_app_token, get_storage_root
from ....core.file_ref import build_file_id_ref
from ...api.chat import get_agent_manager
from ...models.task import TaskStatus
from ...services.channel_runtime import (
    ChannelAuthorizationError,
    ChannelConfigurationError,
    DownloadedChannelFile,
    authorize_channel_sender,
    deactivate_channel_sync,
    load_active_channel_configs,
    load_channel_output_files,
    persist_channel_user_message,
    prepare_channel_task,
    register_channel_uploaded_files,
    update_channel_task_fields,
)
from ...services.client_error_messages import CLIENT_SAFE_AUTO_MODEL_UNAVAILABLE
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
from ...services.llm_utils import AutoModelUnavailableError
from ...services.managed_task_lease import ManagedTaskLease
from ...services.task_execution_context_service import (
    materialize_task_execution_recovery_state,
)
from ...services.task_lease_service import TaskLeaseLostError
from ...services.task_setup_snapshot import load_task_setup_snapshot_sync
from .trace_handler import SlackTraceHandler
from .utils import (
    SlackFileRef,
    SlackFileUrlError,
    markdown_to_slack,
    strip_slack_file_refs,
    validate_slack_file_url,
)

logger = logging.getLogger(__name__)

_DIRECT_MESSAGE_CHANNEL_TYPES = {"im", "mpim"}
_SUPPORTED_MESSAGE_SUBTYPES = {None, "file_share", "thread_broadcast"}
_CONTROL_START_COMMANDS = {"/start", "start"}
_CONTROL_NEW_COMMANDS = {"/new", "new", "new task"}
_MAX_FILE_DOWNLOAD_REDIRECTS = 5
_MAX_RECENT_EVENT_IDS = 1000
# Slack rejects message text over 4000 characters. Chunk the source below that
# and clamp again after conversion, since entity escaping can expand text.
_MAX_SOURCE_CHUNK_CHARS = 3500
_MAX_MESSAGE_CHARS = 3900
# Connection retries back off from 10s to 5min so a permanently broken token
# or an unreachable gateway stops emitting a warning every 10 seconds.
_RETRY_INITIAL_SECONDS = 10.0
_RETRY_MAX_SECONDS = 300.0


class SlackFileDownloadError(RuntimeError):
    """Slack supplied attachments but none could be registered for the turn."""


class _RetryBackoff:
    """Exponential backoff with a ceiling for Slack connection retry loops.

    A flat retry interval makes a permanently broken config (a revoked app
    token, an unreachable gateway) indistinguishable from a transient blip:
    it just logs at the same rate forever. Growing the delay keeps a transient
    failure fast to recover from while letting a persistent one fall quiet.
    """

    def __init__(
        self,
        *,
        initial_seconds: float = _RETRY_INITIAL_SECONDS,
        max_seconds: float = _RETRY_MAX_SECONDS,
    ) -> None:
        self._initial: float = initial_seconds
        self._max: float = max_seconds
        self.attempts: int = 0

    @property
    def delay(self) -> float:
        return min(self._initial * float(2**self.attempts), self._max)

    async def sleep(self) -> None:
        await asyncio.sleep(self.delay)
        self.attempts += 1

    def reset(self) -> None:
        self.attempts = 0

    @property
    def exhausted_quiet_threshold(self) -> bool:
        """Whether the delay has reached its ceiling (log less past this)."""
        return self.delay >= self._max


def _payload_team_id(payload: dict[str, Any]) -> str | None:
    """Resolve the workspace a Socket Mode payload belongs to."""
    team_id = payload.get("team_id")
    if team_id:
        return str(team_id)
    event = payload.get("event")
    if isinstance(event, dict) and event.get("team"):
        return str(event["team"])
    authorizations = payload.get("authorizations")
    if isinstance(authorizations, list):
        for authorization in authorizations:
            if isinstance(authorization, dict) and authorization.get("team_id"):
                return str(authorization["team_id"])
    return None


class SlackBotInstance:
    """One Slack app connected to one workspace through Socket Mode."""

    def __init__(
        self,
        bot_token: str,
        app_token: str | None,
        instance_id: str,
        channel_id: int | None = None,
        channel_name: str | None = None,
        bot_user_id: str | None = None,
    ) -> None:
        self.bot_token = bot_token
        self.app_token = app_token
        self.instance_id = instance_id
        self.channel_id = channel_id
        self.channel_name = channel_name
        self.web_client = AsyncWebClient(token=bot_token)
        self.socket_client: SocketModeClient | None = None
        self.bot_user_id = bot_user_id
        self.polling_task: asyncio.Task[None] | None = None

        state_dir = get_storage_root() / "channel_state"
        self.active_tasks_file = state_dir / f"slack_active_tasks_{instance_id}.json"
        self.active_tasks = self._load_active_tasks()

        self.event_queues: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
        self.event_tasks: dict[str, asyncio.Task[None]] = {}
        self._recent_event_ids: deque[str] = deque(maxlen=_MAX_RECENT_EVENT_IDS)
        self._recent_event_id_set: set[str] = set()
        self._accepting = True
        self._deactivation_sync_task: asyncio.Task[None] | None = None
        self._run_forever = asyncio.Event()
        self._stop_lock: asyncio.Lock | None = None
        self._stop_loop: asyncio.AbstractEventLoop | None = None

    def _load_active_tasks(self) -> dict[str, int]:
        if not self.active_tasks_file.exists():
            return {}
        try:
            data = json.loads(self.active_tasks_file.read_text())
            if not isinstance(data, dict):
                return {}
            return {str(key): int(value) for key, value in data.items()}
        except Exception:
            logger.warning(
                "Failed to load Slack active-task state for %s",
                self.instance_id,
                exc_info=True,
            )
            return {}

    def _save_active_tasks(self) -> None:
        try:
            self.active_tasks_file.parent.mkdir(parents=True, exist_ok=True)
            self.active_tasks_file.write_text(json.dumps(self.active_tasks))
        except Exception:
            logger.warning(
                "Failed to save Slack active-task state for %s",
                self.instance_id,
                exc_info=True,
            )

    async def _handle_socket_request(
        self,
        client: AsyncBaseSocketModeClient,
        request: SocketModeRequest,
    ) -> None:
        if request.envelope_id:
            await client.send_socket_mode_response(
                SocketModeResponse(envelope_id=request.envelope_id)
            )
        if not self._accepting or request.type != "events_api":
            return

        payload = request.payload if isinstance(request.payload, dict) else {}
        await self.handle_events_api_payload(payload)

    async def handle_events_api_payload(self, payload: dict[str, Any]) -> None:
        """Queue one already-acknowledged Events API payload for this workspace."""
        if not self._accepting:
            return
        event = payload.get("event")
        if not isinstance(event, dict):
            return
        event_type = str(event.get("type") or "")
        if event_type == "app_uninstalled" or (
            event_type == "tokens_revoked" and self._revokes_bot_token(event)
        ):
            await self._deactivate_after_workspace_removal(event_type)
            return
        if not self._should_handle_event(event):
            return

        dedup_key = self._event_dedup_key(payload, event)
        if dedup_key and not self._remember_event_id(dedup_key):
            return

        conversation_key = self._conversation_key(payload, event)
        self.event_queues.setdefault(conversation_key, []).append((payload, event))
        worker = self.event_tasks.get(conversation_key)
        if worker is None or worker.done():
            self.event_tasks[conversation_key] = asyncio.create_task(
                self._process_event_queue(conversation_key)
            )

    @staticmethod
    def _revokes_bot_token(event: dict[str, Any]) -> bool:
        """Report whether a ``tokens_revoked`` event kills this bot's token.

        Slack also sends this event when only a user token is revoked, which
        leaves the bot token working. Tearing the channel down then would
        destroy a live integration, so require a bot entry in the payload.
        No ``user_scope`` is requested today, so in practice every such event
        carries ``tokens.bot`` — but the check keeps that assumption explicit.
        """
        tokens = event.get("tokens")
        if not isinstance(tokens, dict):
            return False
        bot_tokens = tokens.get("bot")
        return bool(bot_tokens)

    async def _deactivate_after_workspace_removal(self, event_type: str) -> None:
        """React to the workspace uninstalling the app or revoking its token.

        The stored bot token is dead, so keep the row from advertising a
        working connection: stop accepting events, deactivate the channel and
        drop the revoked token, then let the manager sync tear this instance
        down. Reinstalling through OAuth reuses the same row via team_id.
        """
        logger.warning(
            "Slack workspace removed the app (%s); deactivating channel %s",
            event_type,
            self.channel_id,
        )
        channel_id = self.channel_id
        if channel_id is None:
            return
        try:
            await run_db_io_cancellation_safe(
                lambda: deactivate_channel_sync(
                    channel_id=channel_id,
                    clear_config_keys=("bot_token", "team_id"),
                )
            )
        except Exception:
            # The row is still active and this bot's token may still work, so
            # keep accepting events: a permanently non-accepting instance is
            # indistinguishable from a healthy one to the manager's sync diff
            # and would only recover on a process restart.
            logger.exception(
                "Failed to deactivate Slack channel %s after %s",
                self.channel_id,
                event_type,
            )
            return
        # Only stop accepting once the row is durably deactivated.
        self._accepting = False
        from ...api.channel import trigger_slack_sync

        # The sync stops this very bot instance, so it must not be awaited
        # from inside this instance's own dispatch path. Keep a reference so
        # the task is not garbage-collected while still running.
        self._deactivation_sync_task = asyncio.create_task(trigger_slack_sync())

    @staticmethod
    def _event_dedup_key(payload: dict[str, Any], event: dict[str, Any]) -> str:
        """Identify the physical Slack message rather than the event envelope.

        A mention inside an already-active thread is delivered twice — once as
        ``app_mention`` and once as ``message`` — with distinct ``event_id``
        values, so deduplication must key on the message itself. Slack retry
        redeliveries also share this key, so they stay covered.
        """
        client_msg_id = str(event.get("client_msg_id") or "")
        if client_msg_id:
            return f"msg:{client_msg_id}"
        channel_id = str(event.get("channel") or "")
        ts = str(event.get("ts") or "")
        if channel_id and ts:
            return f"ts:{channel_id}:{ts}"
        return str(payload.get("event_id") or "")

    def _remember_event_id(self, event_id: str) -> bool:
        if event_id in self._recent_event_id_set:
            return False
        if len(self._recent_event_ids) == _MAX_RECENT_EVENT_IDS:
            # deque(maxlen=...) drops the oldest entry on append, so read it
            # before appending to keep the companion set in step.
            self._recent_event_id_set.discard(self._recent_event_ids[0])
        self._recent_event_ids.append(event_id)
        self._recent_event_id_set.add(event_id)
        return True

    def _should_handle_event(self, event: dict[str, Any]) -> bool:
        event_type = str(event.get("type") or "")
        if event_type not in {"app_mention", "message"}:
            return False
        if event.get("bot_id") or event.get("app_id"):
            return False
        if self.bot_user_id and str(event.get("user") or "") == self.bot_user_id:
            return False
        if event.get("subtype") not in _SUPPORTED_MESSAGE_SUBTYPES:
            return False
        if event_type == "app_mention":
            return True
        channel_type = str(event.get("channel_type") or "")
        channel_id = str(event.get("channel") or "")
        if channel_type in _DIRECT_MESSAGE_CHANNEL_TYPES or channel_id.startswith("D"):
            return True

        # In shared channels, accept unmentioned replies only inside a thread
        # that this user already started with the bot. This keeps broad message
        # subscriptions from turning the app into an ambient channel listener.
        thread_ts = str(event.get("thread_ts") or "")
        user_id = str(event.get("user") or "")
        active_suffix = f":{channel_id}:{user_id}:{thread_ts}"
        return bool(thread_ts) and any(
            key.endswith(active_suffix) for key in self.active_tasks
        )

    @staticmethod
    def _conversation_key(
        payload: dict[str, Any],
        event: dict[str, Any],
    ) -> str:
        team_id = str(payload.get("team_id") or event.get("team") or "workspace")
        channel_id = str(event.get("channel") or "channel")
        user_id = str(event.get("user") or "user")
        channel_type = str(event.get("channel_type") or "")
        if channel_type in _DIRECT_MESSAGE_CHANNEL_TYPES or channel_id.startswith("D"):
            scope = str(event.get("thread_ts") or "direct")
        else:
            scope = str(event.get("thread_ts") or event.get("ts") or "thread")
        return ":".join((team_id, channel_id, user_id, scope))

    async def _process_event_queue(self, conversation_key: str) -> None:
        try:
            while self._accepting:
                queue = self.event_queues.get(conversation_key)
                if not queue:
                    self.event_queues.pop(conversation_key, None)
                    return
                payload, event = queue.pop(0)
                try:
                    await self._process_event(conversation_key, payload, event)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # A failure notifying Slack about one event must not strand
                    # the rest of this conversation's queued events.
                    logger.exception(
                        "Error processing queued Slack event for %s",
                        conversation_key,
                    )
        finally:
            current = asyncio.current_task()
            if self.event_tasks.get(conversation_key) is current:
                self.event_tasks.pop(conversation_key, None)
            if not self.event_queues.get(conversation_key):
                self.event_queues.pop(conversation_key, None)

    def _message_text(self, event: dict[str, Any]) -> str:
        text = str(event.get("text") or "")
        if self.bot_user_id:
            text = re.sub(rf"<@{re.escape(self.bot_user_id)}>", " ", text)
        return re.sub(r"[ \t]+", " ", html.unescape(text)).strip()

    @staticmethod
    def _reply_thread_ts(event: dict[str, Any]) -> str | None:
        channel_id = str(event.get("channel") or "")
        channel_type = str(event.get("channel_type") or "")
        if channel_type in _DIRECT_MESSAGE_CHANNEL_TYPES or channel_id.startswith("D"):
            value = event.get("thread_ts")
        else:
            value = event.get("thread_ts") or event.get("ts")
        return str(value) if value else None

    async def _process_event(
        self,
        conversation_key: str,
        payload: dict[str, Any],
        event: dict[str, Any],
    ) -> None:
        del payload
        slack_user_id = str(event.get("user") or "")
        slack_channel_id = str(event.get("channel") or "")
        thread_ts = self._reply_thread_ts(event)
        text = self._message_text(event)
        raw_files = event.get("files")
        files = (
            [item for item in raw_files if isinstance(item, dict)]
            if isinstance(raw_files, list)
            else []
        )
        if not slack_user_id or not slack_channel_id or (not text and not files):
            return

        normalized_command = text.casefold()
        if normalized_command in _CONTROL_START_COMMANDS | _CONTROL_NEW_COMMANDS:
            await self._handle_control_command(
                conversation_key=conversation_key,
                user_id=slack_user_id,
                channel_id=slack_channel_id,
                thread_ts=thread_ts,
                command=normalized_command,
            )
            return

        claimed_task_id: int | None = None
        managed_lease: ManagedTaskLease | None = None
        loading_ts: str | None = None
        try:
            prompt_text = text or "Please process the attached Slack file(s)."
            active_task_id = self.active_tasks.get(conversation_key)
            try:
                prepared_task = await prepare_channel_task(
                    channel_id=self.channel_id,
                    external_user_id=slack_user_id,
                    active_task_id=active_task_id,
                    text=prompt_text,
                    channel_name=self.channel_name,
                )
            except ChannelAuthorizationError:
                await self._send_text(
                    slack_channel_id,
                    "🚫 You are not authorized to use this bot.",
                    thread_ts=thread_ts,
                )
                return
            except ChannelConfigurationError:
                await self._send_text(
                    slack_channel_id,
                    "Configuration error: Cannot find the owner of this bot.",
                    thread_ts=thread_ts,
                )
                return

            if prepared_task is None:
                await self._send_text(
                    slack_channel_id,
                    "I'm still working on the previous message. "
                    "Please wait for it to finish.",
                    thread_ts=thread_ts,
                )
                return

            managed_lease = prepared_task.managed_lease
            task_id = prepared_task.task_id
            claimed_task_id = task_id
            owner_user_id = prepared_task.user_id
            is_new_task = prepared_task.is_new_task
            if is_new_task:
                self.active_tasks[conversation_key] = task_id
                self._save_active_tasks()

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

            turn_id = str(uuid4())
            context: dict[str, Any] = {"turn_id": turn_id}
            display_message = text
            execution_text = prompt_text
            persisted_attachments: list[dict[str, Any]] = []

            if files:
                uploaded_info = await self._download_and_register_files(
                    files=files,
                    agent_service=agent_service,
                    task_id=task_id,
                    user_id=owner_user_id,
                )
                if not uploaded_info:
                    raise SlackFileDownloadError(
                        "Slack attachments could not be downloaded"
                    )
                persisted_attachments = normalize_attachments_for_persistence(
                    uploaded_info
                )
                if not display_message:
                    names = ", ".join(str(item["name"]) for item in uploaded_info)
                    display_message = f"Attached file(s): {names}"
                execution_text = append_uploaded_files_context(
                    prompt_text,
                    build_uploaded_files_context(uploaded_info),
                )
                if is_new_task:
                    await update_channel_task_fields(
                        task_id=task_id,
                        user_id=owner_user_id,
                        description=display_message,
                    )
                context["state"] = {"file_info": uploaded_info}
                context["file_info"] = uploaded_info
                context["uploaded_files"] = [
                    str(item["path"]) for item in uploaded_info
                ]
                context["files"] = persisted_attachments
                context["display_message"] = display_message

            await persist_channel_user_message(
                task_id=task_id,
                user_id=owner_user_id,
                content=display_message or prompt_text,
                attachments=persisted_attachments or None,
                turn_id=turn_id,
            )

            loading_ts = await self._send_text(
                slack_channel_id,
                "Got it, I'm working on this now.\n"
                "_I'll update this message as I make progress._",
                thread_ts=thread_ts,
            )
            trace_handler: SlackTraceHandler | None = None
            if loading_ts:
                trace_handler = SlackTraceHandler(
                    task_id,
                    self.web_client,
                    slack_channel_id,
                    loading_ts,
                )
                agent_service.tracer.add_handler(trace_handler)

            from ...user_isolated_memory import UserContext

            actual_task_id = str(task_id)
            try:
                with UserContext(owner_user_id):
                    result = await agent_manager.execute_task(
                        agent_service=agent_service,
                        task=execution_text,
                        context=context,
                        task_id=actual_task_id,
                        tracking_task_id=actual_task_id,
                        db_session=None,
                        manage_task_lease=False,
                        task_lease=managed_lease.lease,
                        task_lease_heartbeat_task=managed_lease.heartbeat_task,
                    )
            finally:
                if (
                    trace_handler is not None
                    and trace_handler in agent_service.tracer.handlers
                ):
                    agent_service.tracer.handlers.remove(trace_handler)

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
                    f"task {task_id} ownership changed before Slack result"
                )

            output, output_files = strip_slack_file_refs(projection.visible_text)
            if not output and output_files:
                output = "Task completed."
            await self._send_final_text(
                channel_id=slack_channel_id,
                thread_ts=thread_ts,
                loading_ts=loading_ts,
                text=output,
            )
            if output_files:
                failed_refs = await self._send_output_files(
                    refs=output_files,
                    channel_id=slack_channel_id,
                    thread_ts=thread_ts,
                    user_id=owner_user_id,
                    task_id=task_id,
                )
                if failed_refs:
                    await self._send_file_fallback_message(
                        refs=failed_refs,
                        channel_id=slack_channel_id,
                        thread_ts=thread_ts,
                    )
        except TaskLeaseLostError:
            logger.warning(
                "Slack execution lost task %s lease; skipping stale result",
                claimed_task_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.exception("Error processing Slack message")
            if managed_lease is not None:
                try:
                    finalized = await managed_lease.finalize_result(
                        status=TaskStatus.FAILED,
                        error_message=str(error),
                    )
                except Exception:
                    logger.warning(
                        "Failed to finalize Slack task %s after channel error",
                        claimed_task_id,
                        exc_info=True,
                    )
                    return
                if not finalized:
                    logger.warning(
                        "Slack task %s ownership changed after channel error; "
                        "skipping stale error response",
                        claimed_task_id,
                    )
                    return
            if isinstance(error, AutoModelUnavailableError):
                error_text = CLIENT_SAFE_AUTO_MODEL_UNAVAILABLE
            elif isinstance(error, SlackFileDownloadError):
                error_text = (
                    "I couldn't download the attached Slack file(s). "
                    "Please try uploading them again."
                )
            else:
                error_text = "Sorry, an error occurred while processing your request."
            if loading_ts:
                await self._update_text(slack_channel_id, loading_ts, error_text)
            else:
                await self._send_text(
                    slack_channel_id,
                    error_text,
                    thread_ts=thread_ts,
                )
        finally:
            if managed_lease is not None:
                close_task = asyncio.create_task(managed_lease.close())
                await drain_async_task_cancellation_safe(close_task)

    async def _handle_control_command(
        self,
        *,
        conversation_key: str,
        user_id: str,
        channel_id: str,
        thread_ts: str | None,
        command: str,
    ) -> None:
        try:
            await authorize_channel_sender(
                channel_id=self.channel_id,
                external_user_id=user_id,
            )
        except ChannelAuthorizationError:
            await self._send_text(
                channel_id,
                "🚫 You are not authorized to use this bot.",
                thread_ts=thread_ts,
            )
            return
        except ChannelConfigurationError:
            await self._send_text(
                channel_id,
                "Configuration error: Cannot find the owner of this bot.",
                thread_ts=thread_ts,
            )
            return

        if command in _CONTROL_START_COMMANDS:
            text = "Welcome to Xagent! Send `new` to start a new task."
        else:
            self.active_tasks[conversation_key] = -1
            self._save_active_tasks()
            text = "Started a new task. Please describe your request."
        await self._send_text(channel_id, text, thread_ts=thread_ts)

    async def _download_and_register_files(
        self,
        *,
        files: list[dict[str, Any]],
        agent_service: Any,
        task_id: int,
        user_id: int,
    ) -> list[dict[str, Any]]:
        if not agent_service.workspace:
            logger.warning("Agent service workspace is unavailable for Slack upload")
            return []
        target_dir = getattr(
            agent_service.workspace,
            "input_dir",
            agent_service.workspace.workspace_dir / "input",
        )

        downloaded_files: list[DownloadedChannelFile] = []
        for file_info in files:
            try:
                downloaded = await self._download_slack_file(file_info, target_dir)
            except Exception:
                logger.warning(
                    "Failed to download Slack file %s",
                    file_info.get("id", "unknown"),
                    exc_info=True,
                )
                continue
            if downloaded is not None:
                downloaded_files.append(downloaded)

        registered = await register_channel_uploaded_files(
            workspace=agent_service.workspace,
            task_id=task_id,
            user_id=user_id,
            files=tuple(downloaded_files),
        )
        return [item.to_file_info(source_key="slack_file_id") for item in registered]

    async def _download_slack_file(
        self,
        file_info: dict[str, Any],
        target_dir: Path,
    ) -> DownloadedChannelFile | None:
        resolved = file_info
        file_id = str(file_info.get("id") or "")
        download_url = str(
            file_info.get("url_private_download") or file_info.get("url_private") or ""
        )
        if not download_url and file_id:
            response = await self.web_client.files_info(file=file_id)
            response_file = response.get("file")
            if isinstance(response_file, dict):
                resolved = response_file
                download_url = str(
                    resolved.get("url_private_download")
                    or resolved.get("url_private")
                    or ""
                )
        if not download_url:
            return None
        # The URL comes from an inbound Slack event and is fetched with the bot
        # token attached, so validate the host before any request is made.
        validate_slack_file_url(download_url)

        from ...api.websocket import build_unique_target_path, normalize_filename

        filename = normalize_filename(
            str(resolved.get("name") or f"{file_id or 'slack-file'}.bin")
        )
        target_path = build_unique_target_path(target_dir, filename)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Redirects are followed manually so every hop is re-validated
            # against the Slack host allowlist; httpx's own redirect handling
            # would let one off-Slack hop receive the bot token.
            async with httpx.AsyncClient(
                headers={"Authorization": f"Bearer {self.bot_token}"},
                follow_redirects=False,
                timeout=60.0,
            ) as client:
                current_url = download_url
                for _ in range(_MAX_FILE_DOWNLOAD_REDIRECTS + 1):
                    async with client.stream("GET", current_url) as http_response:
                        if http_response.is_redirect:
                            location = http_response.headers.get("location") or ""
                            if not location:
                                raise SlackFileUrlError(
                                    "Slack file redirect is missing a location"
                                )
                            current_url = validate_slack_file_url(
                                str(httpx.URL(current_url).join(location))
                            )
                            continue
                        http_response.raise_for_status()
                        with target_path.open("wb") as target:
                            async for chunk in http_response.aiter_bytes():
                                target.write(chunk)
                        break
                else:
                    raise SlackFileUrlError("Slack file download redirected too often")
        except BaseException:
            target_path.unlink(missing_ok=True)
            raise

        mime_type = str(resolved.get("mimetype") or "")
        if not mime_type:
            mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        return DownloadedChannelFile(
            name=filename,
            path=target_path,
            mime_type=mime_type,
            size=target_path.stat().st_size,
            source_id=file_id or None,
        )

    async def _send_text(
        self,
        channel_id: str,
        text: str,
        *,
        thread_ts: str | None,
    ) -> str | None:
        return await self._send_mrkdwn(
            channel_id,
            markdown_to_slack(text),
            thread_ts=thread_ts,
        )

    async def _send_mrkdwn(
        self,
        channel_id: str,
        text: str,
        *,
        thread_ts: str | None,
    ) -> str | None:
        response = await self.web_client.chat_postMessage(
            channel=channel_id,
            text=text,
            thread_ts=thread_ts,
            link_names=False,
        )
        message_ts = response.get("ts")
        return str(message_ts) if message_ts else None

    async def _update_text(
        self,
        channel_id: str,
        message_ts: str,
        text: str,
    ) -> None:
        await self._update_mrkdwn(
            channel_id,
            message_ts,
            markdown_to_slack(text),
        )

    async def _update_mrkdwn(
        self,
        channel_id: str,
        message_ts: str,
        text: str,
    ) -> None:
        await self.web_client.chat_update(
            channel=channel_id,
            ts=message_ts,
            text=text,
            link_names=False,
        )

    async def _send_final_text(
        self,
        *,
        channel_id: str,
        thread_ts: str | None,
        loading_ts: str | None,
        text: str,
    ) -> None:
        # Split the source text and convert each chunk, never the other way
        # round: splitting converted mrkdwn can bisect a <url|label> token or
        # an escaped entity and emit visibly broken output. Entity escaping can
        # still expand a chunk past Slack's limit, so clamp after converting.
        chunks = [
            markdown_to_slack(chunk)[:_MAX_MESSAGE_CHARS]
            for chunk in self._split_message(text, max_length=_MAX_SOURCE_CHUNK_CHARS)
        ]
        if loading_ts:
            await self._update_mrkdwn(channel_id, loading_ts, chunks[0])
        else:
            await self._send_mrkdwn(channel_id, chunks[0], thread_ts=thread_ts)
        for chunk in chunks[1:]:
            await self._send_mrkdwn(channel_id, chunk, thread_ts=thread_ts)

    @staticmethod
    def _split_message(text: str, max_length: int = 3900) -> list[str]:
        normalized = text.strip() or "Task completed."
        chunks: list[str] = []
        while normalized:
            if len(normalized) <= max_length:
                chunks.append(normalized)
                break
            split_at = normalized.rfind("\n", 0, max_length)
            if split_at < max_length // 2:
                split_at = normalized.rfind(" ", 0, max_length)
            if split_at < max_length // 2:
                split_at = max_length
            chunks.append(normalized[:split_at].rstrip())
            normalized = normalized[split_at:].lstrip()
        return chunks

    async def _send_output_files(
        self,
        *,
        refs: list[SlackFileRef],
        channel_id: str,
        thread_ts: str | None,
        user_id: int,
        task_id: int,
    ) -> list[SlackFileRef]:
        file_records = await load_channel_output_files(
            file_ids=[ref.file_id for ref in refs],
            user_id=user_id,
            task_id=task_id,
        )
        records_by_id = {record.file_id: record for record in file_records}
        failed: list[SlackFileRef] = []
        for ref in refs:
            record = records_by_id.get(ref.file_id)
            if record is None:
                failed.append(ref)
                continue
            file_path = Path(record.storage_path)
            if not file_path.is_file():
                failed.append(ref)
                continue
            try:
                await self.web_client.files_upload_v2(
                    channel=channel_id,
                    thread_ts=thread_ts,
                    file=file_path,
                    filename=record.filename,
                    title=ref.label or record.filename,
                )
            except Exception:
                logger.warning(
                    "Failed to upload Slack output file %s",
                    ref.file_id,
                    exc_info=True,
                )
                failed.append(ref)
        return failed

    async def _send_file_fallback_message(
        self,
        *,
        refs: list[SlackFileRef],
        channel_id: str,
        thread_ts: str | None,
    ) -> None:
        subject = "file" if len(refs) == 1 else "files"
        lines = [
            f"I couldn't send the {subject} through Slack, "
            "but the internal file reference is still available:"
        ]
        lines.extend(f"- {ref.label}: {build_file_id_ref(ref.file_id)}" for ref in refs)
        await self._send_text(channel_id, "\n".join(lines), thread_ts=thread_ts)

    async def start(self) -> None:
        if not self._accepting:
            return
        if not self.app_token:
            raise RuntimeError("Manual Slack Socket Mode requires an app-level token")
        backoff = _RetryBackoff()
        while self._accepting:
            try:
                auth = await self.web_client.auth_test()
                bot_user_id = auth.get("user_id")
                self.bot_user_id = str(bot_user_id) if bot_user_id else None
                break
            except asyncio.CancelledError:
                raise
            except Exception:
                # A revoked app token never recovers, so once the delay has
                # grown to its ceiling report that plainly instead of
                # repeating an identical warning forever.
                logger.warning(
                    "Slack bot %s authentication failed (attempt %d); "
                    "retrying in %.0fs%s",
                    self.instance_id,
                    backoff.attempts + 1,
                    backoff.delay,
                    " -- check whether the bot token is still valid"
                    if backoff.exhausted_quiet_threshold
                    else "",
                    exc_info=not backoff.exhausted_quiet_threshold,
                )
                await backoff.sleep()
        if not self._accepting:
            return

        self.socket_client = SocketModeClient(
            app_token=self.app_token,
            web_client=self.web_client,
        )
        self.socket_client.socket_mode_request_listeners.append(
            self._handle_socket_request
        )
        logger.info("Starting Slack bot %s", self.instance_id)
        await self.socket_client.connect()
        try:
            await self._run_forever.wait()
        finally:
            if self.socket_client is not None:
                await self.socket_client.close()

    def _stop_lock_for_current_loop(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        lock = self._stop_lock
        if lock is None or (self._stop_loop is not loop and not lock.locked()):
            lock = asyncio.Lock()
            self._stop_lock = lock
            self._stop_loop = loop
        elif self._stop_loop is not loop:
            raise RuntimeError("Slack bot stop is already running on another loop")
        return lock

    async def _stop_once(self, lock: asyncio.Lock) -> None:
        async with lock:
            self._accepting = False
            self._run_forever.set()
            if self.socket_client is not None:
                await self.socket_client.close()
                self.socket_client = None
            current = asyncio.current_task()
            workers = {
                task for task in self.event_tasks.values() if task is not current
            }
            for worker in workers:
                if not worker.done():
                    worker.cancel()
            if workers:
                await asyncio.gather(*workers, return_exceptions=True)
            self.event_tasks.clear()
            self.event_queues.clear()

    async def stop(self) -> None:
        stop_task = asyncio.create_task(
            self._stop_once(self._stop_lock_for_current_loop())
        )
        await drain_async_task_cancellation_safe(stop_task)


class SlackOAuthSocketGateway:
    """One shared Socket Mode connection for every OAuth-installed workspace."""

    def __init__(
        self,
        app_token: str,
        bot_lookup: Callable[[dict[str, Any]], SlackBotInstance | None],
    ) -> None:
        self.app_token = app_token
        self.bot_lookup = bot_lookup
        self.socket_client: SocketModeClient | None = None
        self._accepting = True
        self._run_forever = asyncio.Event()

    async def _handle_socket_request(
        self,
        client: AsyncBaseSocketModeClient,
        request: SocketModeRequest,
    ) -> None:
        if request.envelope_id:
            await client.send_socket_mode_response(
                SocketModeResponse(envelope_id=request.envelope_id)
            )
        if not self._accepting or request.type != "events_api":
            return
        payload = request.payload if isinstance(request.payload, dict) else {}
        bot = self.bot_lookup(payload)
        if bot is None:
            logger.warning(
                "Ignoring Slack OAuth event for unknown workspace %s",
                _payload_team_id(payload) or "unknown",
            )
            return
        await bot.handle_events_api_payload(payload)

    async def start(self) -> None:
        # The socket is closed in finally so a normal return or a cancellation
        # cannot leak an open websocket, which would let a restarted gateway
        # race a still-connected one for the same app token.
        backoff = _RetryBackoff()
        try:
            while self._accepting:
                try:
                    self.socket_client = SocketModeClient(
                        app_token=self.app_token,
                        web_client=AsyncWebClient(),
                    )
                    self.socket_client.socket_mode_request_listeners.append(
                        self._handle_socket_request
                    )
                    logger.info("Starting shared Slack OAuth Socket Mode gateway")
                    await self.socket_client.connect()
                    # Connected: a later drop restarts from the short delay.
                    backoff.reset()
                    await self._run_forever.wait()
                    return
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # This gateway serves every OAuth workspace, so a
                    # persistent failure here is the more likely of the two
                    # loops to sit broken for a long time.
                    logger.warning(
                        "Shared Slack OAuth Socket Mode connection failed "
                        "(attempt %d); retrying in %.0fs%s",
                        backoff.attempts + 1,
                        backoff.delay,
                        " -- check XAGENT_SLACK_APP_TOKEN and network egress"
                        if backoff.exhausted_quiet_threshold
                        else "",
                        exc_info=not backoff.exhausted_quiet_threshold,
                    )
                    if self.socket_client is not None:
                        with suppress(Exception):
                            await self.socket_client.close()
                        self.socket_client = None
                    await backoff.sleep()
        finally:
            if self.socket_client is not None:
                with suppress(Exception):
                    await self.socket_client.close()
                self.socket_client = None

    async def stop(self) -> None:
        self._accepting = False
        self._run_forever.set()
        if self.socket_client is not None:
            await self.socket_client.close()
            self.socket_client = None


class SlackChannelManager:
    enabled = True

    def __init__(self) -> None:
        # Manual mode keeps one user-owned Slack app and Socket connection per
        # bot token. OAuth mode uses one Xagent-owned app connection and routes
        # events to these workspace-specific bot clients by team_id.
        self.bots: dict[str, SlackBotInstance] = {}
        self.oauth_bots: dict[str, SlackBotInstance] = {}
        self.oauth_gateway: SlackOAuthSocketGateway | None = None
        self.oauth_gateway_task: asyncio.Task[None] | None = None
        self._bot_stop_tasks: dict[str, asyncio.Task[None]] = {}
        self._sync_lock = asyncio.Lock()

    async def start(self) -> None:
        await self._sync_bots_async()

    async def stop(self) -> None:
        await self._stop_oauth_gateway()
        for team_id in list(self.oauth_bots):
            await self._stop_oauth_bot(team_id)
        for bot_token in list(self.bots):
            await self._stop_bot_for_token(bot_token)

    async def _sync_bots_async(self) -> None:
        async with self._sync_lock:
            try:
                # One load, partitioned on the declared installation_mode.
                # Capability probing (which keys happen to be present) is not
                # mutually exclusive — a row carrying both app_token and OAuth
                # identity would otherwise start two bot instances.
                channels = await load_active_channel_configs(
                    channel_type="slack",
                    required_config_keys=("bot_token",),
                    optional_config_keys=(
                        "app_token",
                        "team_id",
                        "bot_user_id",
                        "installation_mode",
                    ),
                )
            except Exception:
                logger.exception("Failed to load Slack channels for sync")
                return

            manual_channels = tuple(
                channel
                for channel in channels
                if channel.config_value("installation_mode") != "oauth"
            )
            oauth_channels = tuple(
                channel
                for channel in channels
                if channel.config_value("installation_mode") == "oauth"
            )
            await self._sync_manual_bots(manual_channels)
            await self._sync_oauth_bots(oauth_channels)

    async def _sync_manual_bots(
        self,
        channels: tuple[Any, ...],
    ) -> None:
        channel_info_by_token: dict[str, dict[str, Any]] = {}
        for channel in channels:
            bot_token = channel.config_value("bot_token")
            app_token = channel.config_value("app_token")
            if bot_token and app_token:
                channel_info_by_token[bot_token] = {
                    "app_token": app_token,
                    "id": channel.channel_id,
                    "name": channel.channel_name,
                }

        active_tokens = set(channel_info_by_token)
        current_tokens = set(self.bots)
        for bot_token in current_tokens - active_tokens:
            await self._stop_bot_for_token(bot_token)

        for bot_token in active_tokens:
            info = channel_info_by_token[bot_token]
            existing = self.bots.get(bot_token)
            if existing is not None and (
                existing.app_token != info["app_token"]
                or existing.channel_id != info["id"]
                or (existing.polling_task is not None and existing.polling_task.done())
            ):
                await self._stop_bot_for_token(bot_token)
                existing = None
            if existing is None:
                await self._start_bot_for_token(
                    bot_token=bot_token,
                    app_token=str(info["app_token"]),
                    channel_id=int(info["id"]),
                    channel_name=str(info["name"]),
                )
            else:
                existing.channel_name = str(info["name"])

    async def _sync_oauth_bots(
        self,
        channels: tuple[Any, ...],
    ) -> None:
        app_token = get_slack_app_token()
        oauth_info_by_team: dict[str, dict[str, Any]] = {}
        for channel in channels:
            team_id = channel.config_value("team_id")
            bot_token = channel.config_value("bot_token")
            bot_user_id = channel.config_value("bot_user_id")
            if not (team_id and bot_token and bot_user_id):
                continue
            if team_id in oauth_info_by_team:
                # Channels arrive ordered by row id, so the oldest row keeps
                # the workspace; duplicates are a data problem worth surfacing
                # rather than a silent last-writer-wins overwrite. The loser
                # can never receive events, so deactivate it instead of
                # leaving a row the API keeps reporting as connected.
                logger.error(
                    "Multiple active Slack OAuth channels claim team %s; "
                    "keeping channel %s and deactivating channel %s",
                    team_id,
                    oauth_info_by_team[team_id]["id"],
                    channel.channel_id,
                )
                losing_channel_id = channel.channel_id
                try:
                    await run_db_io_cancellation_safe(
                        lambda: deactivate_channel_sync(
                            channel_id=losing_channel_id,
                            clear_config_keys=("team_id",),
                        )
                    )
                except Exception:
                    logger.exception(
                        "Failed to deactivate duplicate Slack channel %s",
                        losing_channel_id,
                    )
                continue
            oauth_info_by_team[team_id] = {
                "bot_token": bot_token,
                "bot_user_id": bot_user_id,
                "id": channel.channel_id,
                "name": channel.channel_name,
            }

        if oauth_info_by_team and not app_token:
            logger.error(
                "Slack OAuth channels are active but XAGENT_SLACK_APP_TOKEN is unset"
            )
            oauth_info_by_team.clear()

        for team_id in set(self.oauth_bots) - set(oauth_info_by_team):
            await self._stop_oauth_bot(team_id)

        for team_id, info in oauth_info_by_team.items():
            existing = self.oauth_bots.get(team_id)
            if existing is not None and (
                existing.bot_token != info["bot_token"]
                or existing.channel_id != info["id"]
                or existing.bot_user_id != info["bot_user_id"]
            ):
                await self._stop_oauth_bot(team_id)
                existing = None
            if existing is None:
                self.oauth_bots[team_id] = SlackBotInstance(
                    bot_token=str(info["bot_token"]),
                    app_token=None,
                    instance_id=f"oauth-channel-{info['id']}",
                    channel_id=int(info["id"]),
                    channel_name=str(info["name"]),
                    bot_user_id=str(info["bot_user_id"]),
                )
            else:
                existing.channel_name = str(info["name"])

        if self.oauth_bots and app_token:
            await self._ensure_oauth_gateway(app_token)
        else:
            await self._stop_oauth_gateway()

    def _oauth_bot_for_payload(
        self,
        payload: dict[str, Any],
    ) -> SlackBotInstance | None:
        team_id = _payload_team_id(payload)
        return self.oauth_bots.get(team_id) if team_id else None

    async def _ensure_oauth_gateway(self, app_token: str) -> None:
        if (
            self.oauth_gateway is not None
            and self.oauth_gateway.app_token == app_token
            and self.oauth_gateway_task is not None
            and not self.oauth_gateway_task.done()
        ):
            return
        await self._stop_oauth_gateway()
        self.oauth_gateway = SlackOAuthSocketGateway(
            app_token=app_token,
            bot_lookup=self._oauth_bot_for_payload,
        )
        self.oauth_gateway_task = asyncio.create_task(self.oauth_gateway.start())

    async def _stop_oauth_gateway(self) -> None:
        gateway = self.oauth_gateway
        task = self.oauth_gateway_task
        self.oauth_gateway = None
        self.oauth_gateway_task = None
        if gateway is not None:
            with suppress(Exception):
                await gateway.stop()
        if task is not None:
            await cancel_and_drain_async_task(task)

    async def _stop_oauth_bot(self, team_id: str) -> None:
        bot = self.oauth_bots.pop(team_id, None)
        if bot is not None:
            with suppress(Exception):
                await bot.stop()

    async def _start_bot_for_token(
        self,
        *,
        bot_token: str,
        app_token: str,
        channel_id: int,
        channel_name: str,
    ) -> None:
        if bot_token in self.bots:
            return
        bot = SlackBotInstance(
            bot_token=bot_token,
            app_token=app_token,
            instance_id=f"channel-{channel_id}",
            channel_id=channel_id,
            channel_name=channel_name,
        )
        self.bots[bot_token] = bot
        bot.polling_task = asyncio.create_task(bot.start())

    async def _shutdown_bot_for_token(
        self,
        bot_token: str,
        bot: SlackBotInstance,
    ) -> None:
        try:
            with suppress(Exception):
                await bot.stop()
        finally:
            if bot.polling_task is not None:
                await cancel_and_drain_async_task(bot.polling_task)
            if self.bots.get(bot_token) is bot:
                self.bots.pop(bot_token, None)

    async def _stop_bot_for_token(self, bot_token: str) -> None:
        stop_task = self._bot_stop_tasks.get(bot_token)
        if stop_task is None:
            bot = self.bots.get(bot_token)
            if bot is None:
                return
            stop_task = asyncio.create_task(
                self._shutdown_bot_for_token(bot_token, bot)
            )
            self._bot_stop_tasks[bot_token] = stop_task
        try:
            await drain_async_task_cancellation_safe(stop_task)
        finally:
            if stop_task.done() and self._bot_stop_tasks.get(bot_token) is stop_task:
                self._bot_stop_tasks.pop(bot_token, None)


_slack_manager: SlackChannelManager | None = None


def get_slack_channel() -> SlackChannelManager:
    global _slack_manager
    if _slack_manager is None:
        _slack_manager = SlackChannelManager()
    return _slack_manager
