import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateMessageRequest,
    CreateMessageRequestBody,
    PatchMessageRequest,
    PatchMessageRequestBody,
)

from ....core.file_ref import build_file_id_ref
from ...api.chat import get_agent_manager
from ...models.task import TaskStatus
from ...services.channel_runtime import (
    ChannelAuthorizationError,
    ChannelConfigurationError,
    DownloadedChannelFile,
    authorize_channel_sender,
    load_active_channel_configs,
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
from ...services.file_turn import normalize_attachments_for_persistence
from ...services.managed_task_lease import ManagedTaskLease
from ...services.task_execution_context_service import (
    materialize_task_execution_recovery_state,
)
from ...services.task_lease_service import TaskLeaseLostError
from ...services.task_setup_snapshot import load_task_setup_snapshot_sync
from .trace_handler import FeishuTraceHandler

logger = logging.getLogger(__name__)


class FeishuBotInstance:
    def __init__(
        self,
        app_id: str,
        app_secret: str,
        instance_id: str,
        channel_id: Optional[int] = None,
        channel_name: Optional[str] = None,
    ):
        self.app_id = app_id
        self.app_secret = app_secret
        self.instance_id = instance_id
        self.channel_id = channel_id
        self.channel_name = channel_name

        self.active_tasks_file = Path(f"data/feishu_active_tasks_{instance_id}.json")
        self.active_tasks = self._load_active_tasks()

        self.ws_client: Any = None
        self.api_client = (
            lark.Client.builder().app_id(app_id).app_secret(app_secret).build()
        )
        self.polling_task: Optional[asyncio.Task] = None
        self.user_message_queues: Dict[str, list] = {}
        self.user_message_tasks: Dict[str, asyncio.Task] = {}
        self._ping_task: asyncio.Task | None = None
        self._accepting = True
        self._ingress_stopped = False
        self._stop_lock: asyncio.Lock | None = None
        self._stop_loop: asyncio.AbstractEventLoop | None = None

        import time

        self.start_time = int(time.time() * 1000)

    def _load_active_tasks(self) -> Dict[str, str]:
        if self.active_tasks_file.exists():
            try:
                with open(self.active_tasks_file, "r") as f:
                    data = json.load(f)
                    return {str(k): str(v) for k, v in data.items()}
            except Exception as e:
                logger.error(f"Error loading feishu active tasks: {e}")
        return {}

    def _save_active_tasks(self) -> None:
        try:
            self.active_tasks_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.active_tasks_file, "w") as f:
                json.dump(self.active_tasks, f)
        except Exception as e:
            logger.error(f"Error saving feishu active tasks: {e}")

    def _handle_message_sync(self, data: Any) -> None:
        if not self._accepting:
            return

        loop = asyncio.get_event_loop()
        if loop.is_running():
            event = data.event
            if not event or not event.message or not event.sender:
                return

            # Ignore messages that were created before this bot instance started
            if hasattr(event.message, "create_time") and event.message.create_time:
                try:
                    if int(event.message.create_time) < self.start_time:
                        logger.info(
                            f"Ignoring stale message from {event.message.create_time} (bot started at {self.start_time})"
                        )
                        return
                except (ValueError, TypeError):
                    pass

            open_id = event.sender.sender_id.open_id

            if open_id not in self.user_message_queues:
                self.user_message_queues[open_id] = []
            self.user_message_queues[open_id].append(data)

            if (
                open_id not in self.user_message_tasks
                or self.user_message_tasks[open_id].done()
            ):
                self.user_message_tasks[open_id] = loop.create_task(
                    self._process_user_queue(open_id)
                )
        else:
            logger.error("No running event loop to schedule Feishu message processing")

    async def _process_user_queue(self, open_id: str) -> None:
        await asyncio.sleep(1.0)
        messages_data = self.user_message_queues.pop(open_id, [])
        if not messages_data:
            return
        await self._process_messages_batch(open_id, messages_data)

    async def _process_messages_batch(
        self, open_id: str, messages_data: list[Any]
    ) -> None:
        chat_id = messages_data[0].event.message.chat_id
        claimed_task_id: int | None = None
        managed_lease: ManagedTaskLease | None = None
        try:
            combined_text = ""
            files_info = []
            message_types = []

            for data in messages_data:
                event = data.event
                message_id = event.message.message_id
                message_type = event.message.message_type
                content_str = event.message.content
                message_types.append(message_type)

                text = ""
                try:
                    content_json = json.loads(content_str)
                    if message_type == "text":
                        text = content_json.get("text", "").strip()
                    elif message_type in ("image", "audio", "media", "file"):
                        if message_type == "image":
                            file_key = content_json.get("image_key")
                        else:
                            file_key = content_json.get("file_key")

                        if file_key:
                            files_info.append(
                                {
                                    "type": message_type,
                                    "file_key": file_key,
                                    "message_id": message_id,
                                }
                            )
                    elif message_type != "text":
                        text = f"Please process this {message_type}."
                except Exception:
                    text = content_str.strip()

                if text:
                    if combined_text:
                        combined_text += "\n" + text
                    else:
                        combined_text = text

            text = combined_text

            if not text and not files_info:
                if message_types:
                    text = f"Received a {message_types[-1]} message."
                else:
                    return

            if text in {"/start", "/new"}:
                try:
                    await authorize_channel_sender(
                        channel_id=self.channel_id,
                        external_user_id=str(open_id),
                    )
                except ChannelAuthorizationError:
                    await self._send_text(
                        chat_id,
                        "\ud83d\udeab You are not authorized to use this bot.",
                    )
                    return
                except ChannelConfigurationError:
                    await self._send_text(
                        chat_id,
                        "This bot is inactive or not correctly configured.",
                    )
                    return

                if text == "/start":
                    await self._send_text(
                        chat_id,
                        "Welcome to Xagent! You can send /new to start a new task.",
                    )
                else:
                    self.active_tasks[open_id] = "-1"
                    self._save_active_tasks()
                    await self._send_text(
                        chat_id,
                        "Started a new task. Please describe your request.",
                    )
                return

            active_task_id = self.active_tasks.get(open_id)
            try:
                prepared_task = await prepare_channel_task(
                    channel_id=self.channel_id,
                    external_user_id=str(open_id),
                    active_task_id=(
                        int(active_task_id) if active_task_id is not None else None
                    ),
                    text=text,
                    channel_name=self.channel_name,
                )
            except ChannelAuthorizationError:
                await self._send_text(
                    chat_id,
                    "\ud83d\udeab You are not authorized to use this bot.",
                )
                return
            except ChannelConfigurationError:
                await self._send_text(
                    chat_id,
                    "This bot is inactive or not correctly configured.",
                )
                return

            if prepared_task is None:
                await self._send_text(
                    chat_id,
                    "I'm still working on the previous message. "
                    "Please wait for it to finish.",
                )
                return

            # Take ownership synchronously after the atomic DB claim so any
            # later transport failure settles or TTL-recovers this exact run.
            managed_lease = prepared_task.managed_lease
            task_id = prepared_task.task_id
            claimed_task_id = task_id
            owner_user_id = prepared_task.user_id
            is_new_task = prepared_task.is_new_task
            if is_new_task:
                self.active_tasks[open_id] = str(task_id)
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
            # Same nudge as the websocket/A2A/v1 resume paths: a cached
            # AgentService whose tools were already built (e.g. paused
            # waiting for the user to connect an app) would otherwise keep
            # its stale MCP config forever, and a Feishu reply is one of
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
            persisted_attachments: list[dict[str, Any]] = []

            if files_info:
                uploaded_info = await self._download_and_register_files(
                    files_info=files_info,
                    agent_service=agent_service,
                    task_id=task_id,
                    user_id=owner_user_id,
                )
                if uploaded_info:
                    persisted_attachments = normalize_attachments_for_persistence(
                        uploaded_info
                    )
                    file_info_list = [
                        f"[{info['name']}]({build_file_id_ref(info['file_id'])})"
                        for info in uploaded_info
                    ]
                    if text:
                        text += f"\n\n{' '.join(file_info_list)}"
                    else:
                        text = " ".join(file_info_list)
                    if is_new_task:
                        await update_channel_task_fields(
                            task_id=task_id,
                            user_id=owner_user_id,
                            description=text,
                        )

                    context["state"] = context.get("state", {})
                    context["state"]["file_info"] = uploaded_info

            await persist_channel_user_message(
                task_id=task_id,
                user_id=owner_user_id,
                content=text,
                attachments=persisted_attachments or None,
                turn_id=message_turn_id,
            )

            loading_msg_id = await self._send_text(
                chat_id,
                f"⏳ **Task #{task_id} is processing...**\n_Please wait for the result._",
            )

            fs_handler = None
            if loading_msg_id:
                fs_handler = FeishuTraceHandler(
                    task_id, self.api_client, chat_id, loading_msg_id
                )
                agent_service.tracer.add_handler(fs_handler)

            from ...user_isolated_memory import UserContext

            actual_task_id = str(task_id)
            try:
                with UserContext(owner_user_id):
                    result = await agent_manager.execute_task(
                        agent_service=agent_service,
                        task=text,
                        context=context,
                        task_id=actual_task_id,
                        tracking_task_id=actual_task_id,
                        db_session=None,
                        manage_task_lease=False,
                        task_lease=managed_lease.lease,
                        task_lease_heartbeat_task=managed_lease.heartbeat_task,
                    )
            finally:
                if fs_handler is not None:
                    agent_service.tracer.remove_handler(fs_handler)

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
                    f"task {task_id} ownership changed before Feishu result"
                )

            output = projection.visible_text

            max_len = 4000
            text_chunks = [
                output[i : i + max_len] for i in range(0, len(output), max_len)
            ]

            if loading_msg_id:
                await self._update_text(chat_id, loading_msg_id, text_chunks[0])
            else:
                await self._send_text(chat_id, text_chunks[0])

            for chunk in text_chunks[1:]:
                await self._send_text(chat_id, chunk)

        except TaskLeaseLostError:
            logger.warning(
                "Feishu execution lost task %s lease; skipping stale result",
                claimed_task_id,
            )
        except Exception as e:
            logger.error(f"Error processing Feishu message: {e}", exc_info=True)
            if managed_lease is not None:
                try:
                    finalized = await managed_lease.finalize_result(
                        status=TaskStatus.FAILED,
                        error_message=str(e),
                    )
                except Exception:
                    logger.warning(
                        "Failed to finalize Feishu task %s after channel error",
                        claimed_task_id,
                        exc_info=True,
                    )
                    return
                if not finalized:
                    logger.warning(
                        "Feishu task %s ownership changed after channel error; "
                        "skipping stale error response",
                        claimed_task_id,
                    )
                    return
            await self._send_text(
                chat_id, "Sorry, an error occurred while processing your request."
            )
        finally:
            if managed_lease is not None:
                await managed_lease.close()

    async def _download_and_register_files(
        self,
        files_info: list,
        agent_service: "Any",
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
        for file_info in files_info:
            worker = asyncio.create_task(
                asyncio.to_thread(
                    self._download_feishu_file_sync,
                    file_info,
                    target_dir,
                )
            )
            try:
                downloaded = await drain_async_task_cancellation_safe(worker)
            except Exception:
                logger.exception(
                    "Failed to download Feishu file %s",
                    file_info.get("file_key", "unknown"),
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
        uploaded_files_info = [item.to_file_info() for item in registered]
        for item in registered:
            logger.info(
                "Successfully downloaded and registered Feishu file: %s",
                item.name,
            )
        return uploaded_files_info

    def _download_feishu_file_sync(
        self,
        file_info: dict[str, Any],
        target_dir: Path,
    ) -> DownloadedChannelFile | None:
        import mimetypes

        from lark_oapi.api.im.v1 import GetMessageResourceRequest

        message_id = file_info["message_id"]
        file_key = file_info["file_key"]
        msg_type = file_info["type"]
        req = (
            GetMessageResourceRequest.builder()
            .message_id(message_id)
            .file_key(file_key)
            .type(msg_type)
            .build()
        )

        resp = self.api_client.im.v1.message_resource.get(req)
        if not resp.success():
            logger.error(
                "Failed to download Feishu file: %s, %s, %s",
                resp.code,
                resp.msg,
                resp.error,
            )
            return None

        if hasattr(resp, "file_name") and resp.file_name:
            file_name = resp.file_name
        else:
            ext = ".jpg" if msg_type == "image" else ".bin"
            file_name = f"{file_key}{ext}"

        from ...api.websocket import build_unique_target_path, normalize_filename

        normalized_file_name = normalize_filename(file_name)
        target_path = build_unique_target_path(target_dir, normalized_file_name)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if not hasattr(resp, "file") or not resp.file:
            logger.error("No file content in Feishu response for %s", file_key)
            return None
        with open(target_path, "wb") as target:
            target.write(resp.file.read())

        mime_type, _ = mimetypes.guess_type(str(target_path))
        return DownloadedChannelFile(
            name=normalized_file_name,
            path=target_path,
            mime_type=mime_type or "application/octet-stream",
            size=target_path.stat().st_size,
            source_id=str(file_key),
        )

    async def _send_text(self, chat_id: str, text: str) -> Optional[str]:
        try:
            # We use "interactive" msg_type instead of "text" to allow patching later.
            # "patch" endpoint only supports cards (interactive).
            card_content = {
                "config": {"wide_screen_mode": True},
                "elements": [{"tag": "markdown", "content": text}],
            }
            req = (
                CreateMessageRequest.builder()
                .receive_id_type("chat_id")
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(chat_id)
                    .msg_type("interactive")
                    .content(json.dumps(card_content))
                    .build()
                )
                .build()
            )
            resp = await asyncio.get_event_loop().run_in_executor(
                None, self.api_client.im.v1.message.create, req
            )
            if not resp.success():
                logger.error(
                    f"Failed to send Feishu message: {resp.code}, {resp.msg}, {resp.error}"
                )
                return None
            if resp.data and resp.data.message_id:
                return resp.data.message_id  # type: ignore
            return None
        except Exception as e:
            logger.error(f"Error sending Feishu message: {e}")
            return None

    async def _update_text(self, chat_id: str, message_id: str, text: str) -> None:
        try:
            card_content = {
                "config": {"wide_screen_mode": True},
                "elements": [{"tag": "markdown", "content": text}],
            }
            req = (
                PatchMessageRequest.builder()
                .message_id(message_id)
                .request_body(
                    PatchMessageRequestBody.builder()
                    .content(json.dumps(card_content))
                    .build()
                )
                .build()
            )
            resp = await asyncio.get_event_loop().run_in_executor(
                None, self.api_client.im.v1.message.patch, req
            )
            if not resp.success():
                # Fallback to normal send if patch fails (e.g., if original msg wasn't patchable)
                logger.error(
                    f"Failed to update Feishu message: {resp.code}, {resp.msg}, {resp.error}"
                )
                if resp.code == 230001:  # "This message is NOT a card." error
                    logger.info("Falling back to send_text instead of update_text")
                    await self._send_text(chat_id, text)
        except Exception as e:
            logger.error(f"Error updating Feishu message: {e}")

    async def start(self) -> None:
        if not self._accepting:
            return
        event_handler = (
            lark.EventDispatcherHandler.builder("", "")
            .register_p2_im_message_receive_v1(self._handle_message_sync)
            .build()
        )

        self.ws_client = lark.ws.Client(
            self.app_id,
            self.app_secret,
            event_handler=event_handler,
            log_level=lark.LogLevel.INFO,
        )
        logger.info(f"Starting Feishu bot {self.instance_id}")

        # We cannot use ws_client.start() because it uses loop.run_until_complete()
        # which fails when the event loop is already running.
        # So we directly call the underlying async methods.
        try:
            await self.ws_client._connect()
        except Exception as e:
            logger.error(f"Feishu bot {self.instance_id} connect failed, err: {e}")
            await self.ws_client._disconnect()
            if self.ws_client._auto_reconnect:
                await self.ws_client._reconnect()
            else:
                raise e

        if not self._accepting:
            await self.ws_client._disconnect()
            return

        self._ping_task = asyncio.create_task(self.ws_client._ping_loop())

        # To keep the start task alive like ws_client.start() did with _select()
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass

    def _stop_lock_for_current_loop(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        lock = self._stop_lock
        if lock is None or (self._stop_loop is not loop and not lock.locked()):
            lock = asyncio.Lock()
            self._stop_lock = lock
            self._stop_loop = loop
        elif self._stop_loop is not loop:
            raise RuntimeError("Feishu bot stop is already running on another loop")
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

    async def _stop_ingress(self) -> None:
        try:
            if self.ws_client:
                # Keep auto_reconnect=True but override _reconnect to gracefully
                # swallow the disconnect exception and stop the receive loop cleanly.
                async def noop_reconnect() -> None:
                    pass

                self.ws_client._auto_reconnect = True
                self.ws_client._reconnect = noop_reconnect

                # Suppress the harmless normal-closure error logged by the Lark SDK
                lark_logger = logging.getLogger("Lark")

                class DisconnectFilter(logging.Filter):
                    def filter(self, record: logging.LogRecord) -> bool:
                        return "receive message loop exit" not in record.getMessage()

                log_filter = DisconnectFilter()
                lark_logger.addFilter(log_filter)

                try:
                    await self.ws_client._disconnect()
                    # Give the receive loop a moment to exit and process the suppressed log
                    await asyncio.sleep(0.1)
                finally:
                    lark_logger.removeFilter(log_filter)
        finally:
            if self._ping_task is not None:
                ping_task = self._ping_task
                self._ping_task = None
                await cancel_and_drain_async_task(ping_task)

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


class FeishuChannelManager:
    enabled = True  # Always enabled, we load dynamically

    def __init__(self) -> None:
        self.bots: Dict[str, FeishuBotInstance] = {}
        self._bot_stop_tasks: Dict[str, asyncio.Task[None]] = {}
        # Channel CRUD endpoints fire sync as a background task, so two syncs
        # can interleave. _stop_bot_for_appid awaits the shutdown drain and
        # only removes the bot from self.bots afterwards, so a second sync
        # entering that window sees an app_id that is still present but
        # already being torn down: it skips starting it, the first sync
        # completes the removal, and a re-enabled channel ends up neither
        # running nor tracked until some later sync happens to run.
        self._sync_lock = asyncio.Lock()

    async def start(self) -> None:
        await self._sync_bots_async()

    async def stop(self) -> None:
        for app_id in list(self.bots.keys()):
            await self._stop_bot_for_appid(app_id)

    async def _sync_bots_async(self) -> None:
        async with self._sync_lock:
            active_app_ids = set()
            channel_info_by_appid: Dict[str, Dict] = {}

            try:
                channels = await load_active_channel_configs(
                    channel_type="feishu",
                    required_config_keys=("app_id", "app_secret"),
                )
                for ch in channels:
                    app_id = ch.config_value("app_id")
                    app_secret = ch.config_value("app_secret")
                    if app_id and app_secret:
                        active_app_ids.add(app_id)
                        channel_info_by_appid[app_id] = {
                            "app_secret": app_secret,
                            "id": ch.channel_id,
                            "name": ch.channel_name,
                        }
            except Exception as e:
                logger.error(f"Failed to load feishu channels for sync: {e}")
                return

            current_app_ids = set(self.bots.keys())

            for app_id in current_app_ids - active_app_ids:
                await self._stop_bot_for_appid(app_id)

            for app_id in active_app_ids - current_app_ids:
                info = channel_info_by_appid[app_id]
                await self._start_bot_for_appid(
                    app_id, info["app_secret"], info["id"], info["name"]
                )

    async def _start_bot_for_appid(
        self, app_id: str, app_secret: str, channel_id: int, channel_name: str
    ) -> None:
        if app_id not in self.bots:
            instance_id = app_id[:8] + "..." if len(app_id) > 8 else "unknown"
            bot = FeishuBotInstance(
                app_id, app_secret, instance_id, channel_id, channel_name
            )
            self.bots[app_id] = bot
            bot.polling_task = asyncio.create_task(bot.start())

    async def _shutdown_bot_for_appid(
        self,
        app_id: str,
        bot: FeishuBotInstance,
    ) -> None:
        try:
            try:
                await bot.stop()
            except Exception as e:
                logger.error(f"Error while stopping feishu bot: {e}")
        finally:
            try:
                if bot.polling_task is not None:
                    await cancel_and_drain_async_task(bot.polling_task)
            finally:
                if self.bots.get(app_id) is bot:
                    self.bots.pop(app_id, None)

    async def _stop_bot_for_appid(self, app_id: str) -> None:
        stop_task = self._bot_stop_tasks.get(app_id)
        if stop_task is None:
            bot = self.bots.get(app_id)
            if bot is None:
                return
            stop_task = asyncio.create_task(self._shutdown_bot_for_appid(app_id, bot))
            self._bot_stop_tasks[app_id] = stop_task

        try:
            await drain_async_task_cancellation_safe(stop_task)
        finally:
            if stop_task.done() and self._bot_stop_tasks.get(app_id) is stop_task:
                self._bot_stop_tasks.pop(app_id, None)


_feishu_manager = None


def get_feishu_channel() -> FeishuChannelManager:
    global _feishu_manager
    if _feishu_manager is None:
        _feishu_manager = FeishuChannelManager()
    return _feishu_manager
