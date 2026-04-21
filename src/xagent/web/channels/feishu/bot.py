import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateMessageRequest,
    CreateMessageRequestBody,
    PatchMessageRequest,
    PatchMessageRequestBody,
)

from ...api.chat import get_agent_manager
from ...models.database import get_db
from ...models.task import Task, TaskStatus
from ...models.user import User
from ...models.user_channel import UserChannel
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
        # Get chat_id from the first message
        chat_id = messages_data[0].event.message.chat_id

        db_gen = get_db()
        db = next(db_gen)
        try:
            user = None
            if self.channel_id:
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
                            if str(open_id) not in allowed_users:
                                await self._send_text(
                                    chat_id,
                                    "\ud83d\udeab You are not authorized to use this bot.",
                                )
                                return

            if not user:
                await self._send_text(
                    chat_id,
                    "Configuration error: Cannot find the owner of this bot.",
                )
                return

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

            if text == "/start":
                await self._send_text(
                    chat_id, "Welcome to Xagent! You can send /new to start a new task."
                )
                return
            elif text == "/new":
                self.active_tasks[open_id] = "-1"
                self._save_active_tasks()
                await self._send_text(
                    chat_id, "Started a new task. Please describe your request."
                )
                return

            active_task_id = self.active_tasks.get(open_id)
            task = None

            if active_task_id == "-1":
                pass
            elif active_task_id:
                task = (
                    db.query(Task)
                    .filter(Task.id == int(active_task_id), Task.user_id == user.id)
                    .first()
                )

            was_completed_or_failed = False
            if not task:
                is_new_task = True

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
                self.active_tasks[open_id] = str(task.id)
                self._save_active_tasks()
            else:
                is_new_task = False
                was_completed_or_failed = task.status in [
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                ]
                task.status = TaskStatus.PENDING
                db.commit()

            agent_manager = get_agent_manager()
            agent_service = await agent_manager.get_agent_for_task(
                int(task.id), db, user=user
            )

            from ...services.task_execution_context_service import (
                load_task_execution_recovery_state,
            )

            recovery_state = await load_task_execution_recovery_state(db, int(task.id))
            agent_service.set_execution_context_messages(
                recovery_state.get("messages", [])
            )
            agent_service.set_recovered_skill_context(
                recovery_state.get("skill_context")
            )

            context: dict = {}

            if files_info:
                uploaded_info = await self._download_and_register_files(
                    files_info=files_info,
                    agent_service=agent_service,
                    task_id=int(task.id),
                    user_id=int(user.id),
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
                            task.title = text if len(text) <= 50 else f"{text[:50]}..."  # type: ignore
                        db.commit()

                    context["state"] = context.get("state", {})
                    context["state"]["file_info"] = uploaded_info

            loading_msg_id = await self._send_text(
                chat_id,
                f"⏳ **Task #{task.id} is processing...**\n_Please wait for the result._",
            )

            if loading_msg_id:
                fs_handler = FeishuTraceHandler(
                    int(task.id), self.api_client, chat_id, loading_msg_id
                )
                agent_service.tracer.add_handler(fs_handler)

            from ...user_isolated_memory import UserContext

            force_fresh_execution = not is_new_task and was_completed_or_failed
            actual_task_id = None if force_fresh_execution else str(task.id)

            with UserContext(int(user.id)):
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

        except Exception as e:
            logger.error(f"Error processing Feishu message: {e}", exc_info=True)
            await self._send_text(
                chat_id, "Sorry, an error occurred while processing your request."
            )
        finally:
            try:
                next(db_gen)
            except StopIteration:
                pass

    async def _download_and_register_files(
        self,
        files_info: list,
        agent_service: "Any",
        task_id: int,
        user_id: int,
        db: Any,
    ) -> list:
        import mimetypes
        from pathlib import Path

        from lark_oapi.api.im.v1 import GetMessageResourceRequest

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

        for f_info in files_info:
            try:
                message_id = f_info["message_id"]
                file_key = f_info["file_key"]
                msg_type = f_info["type"]

                # Call Lark API to get file resource
                req = (
                    GetMessageResourceRequest.builder()
                    .message_id(message_id)
                    .file_key(file_key)
                    .type(msg_type)
                    .build()
                )

                resp = await asyncio.get_event_loop().run_in_executor(
                    None, self.api_client.im.v1.message_resource.get, req
                )

                if not resp.success():
                    logger.error(
                        f"Failed to download Feishu file: {resp.code}, {resp.msg}, {resp.error}"
                    )
                    continue

                if hasattr(resp, "file_name") and resp.file_name:
                    file_name = resp.file_name
                else:
                    ext = ".jpg" if msg_type == "image" else ".bin"
                    file_name = f"{file_key}{ext}"

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

                # Write file content
                if hasattr(resp, "file") and resp.file:
                    with open(target_path, "wb") as f:
                        f.write(resp.file.read())
                else:
                    logger.error(f"No file content in Feishu response for {file_key}")
                    continue

                mime_type, _ = mimetypes.guess_type(str(target_path))
                if not mime_type:
                    mime_type = "application/octet-stream"

                file_size = target_path.stat().st_size

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
                    f"Successfully downloaded and registered Feishu file: {normalized_file_name}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to process Feishu file {f_info.get('file_key', 'unknown')}: {e}"
                )

        return uploaded_files_info

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

        self._ping_task = asyncio.create_task(self.ws_client._ping_loop())

        # To keep the start task alive like ws_client.start() did with _select()
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
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

        if hasattr(self, "_ping_task") and self._ping_task:
            self._ping_task.cancel()


class FeishuChannelManager:
    enabled = True  # Always enabled, we load dynamically

    def __init__(self) -> None:
        self.bots: Dict[str, FeishuBotInstance] = {}

    async def start(self) -> None:
        await self._sync_bots_async()

    async def stop(self) -> None:
        for app_id in list(self.bots.keys()):
            await self._stop_bot_for_appid(app_id)

    async def _sync_bots_async(self) -> None:
        active_app_ids = set()
        channel_info_by_appid: Dict[str, Dict] = {}

        db_gen = get_db()
        db = next(db_gen)
        try:
            channels = (
                db.query(UserChannel)
                .filter(
                    UserChannel.channel_type == "feishu",
                    UserChannel.is_active.is_(True),
                )
                .all()
            )
            for ch in channels:
                app_id = ch.config.get("app_id")
                app_secret = ch.config.get("app_secret")
                if app_id and app_secret:
                    active_app_ids.add(app_id)
                    channel_info_by_appid[app_id] = {
                        "app_secret": app_secret,
                        "id": ch.id,
                        "name": ch.channel_name,
                    }
        except Exception as e:
            logger.error(f"Failed to load feishu channels for sync: {e}")
            return
        finally:
            try:
                next(db_gen)
            except StopIteration:
                pass

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

    async def _stop_bot_for_appid(self, app_id: str) -> None:
        if app_id in self.bots:
            bot = self.bots[app_id]
            try:
                await bot.stop()
            except Exception as e:
                logger.error(f"Error while stopping feishu bot: {e}")
            if bot.polling_task and not bot.polling_task.done():
                bot.polling_task.cancel()
            del self.bots[app_id]


_feishu_manager = None


def get_feishu_channel() -> FeishuChannelManager:
    global _feishu_manager
    if _feishu_manager is None:
        _feishu_manager = FeishuChannelManager()
    return _feishu_manager
