import asyncio
import json
import logging
import os
from collections import deque
from contextlib import suppress
from itertools import groupby
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, Optional, cast
from uuid import uuid4

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateMessageRequest,
    CreateMessageRequestBody,
    PatchMessageRequest,
    PatchMessageRequestBody,
)

from ....config import get_channel_ingress_enabled, get_shared_task_execution_enabled
from ....core.file_ref import build_file_id_ref
from ....core.file_storage.keys import build_upload_storage_key
from ...models.task import TaskStatus
from ...services.agent_service_manager import get_agent_manager
from ...services.channel_delivery import (
    ChannelDelivery,
    ChannelDeliveryDiscarded,
    discard_channel_task_results,
    recover_channel_results,
)
from ...services.channel_input_acceptance import (
    AcceptedChannelInput,
    ChannelInput,
    ChannelInputBatchChanged,
    accept_channel_input,
    lookup_channel_inputs,
)
from ...services.channel_progress import DurableChannelProgress
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
from ...services.client_error_messages import CLIENT_SAFE_AUTO_MODEL_UNAVAILABLE
from ...services.db_runtime import (
    await_task_settlement,
    cancel_and_drain_async_task,
    drain_async_task_cancellation_safe,
    run_db_io_cancellation_safe,
)
from ...services.execution_result_projection import project_execution_result_for_channel
from ...services.file_turn import normalize_attachments_for_persistence
from ...services.llm_utils import AutoModelUnavailableError
from ...services.managed_task_lease import ManagedTaskLease
from ...services.shared_channel_execution import SharedChannelTurn
from ...services.task_event_bridge import get_task_event_bridge
from ...services.task_execution_context_service import (
    materialize_task_execution_recovery_state,
)
from ...services.task_lease_service import TaskLeaseLostError
from ...services.task_orchestrator import TaskTurnError, TaskTurnPayload
from ...services.task_setup_snapshot import load_task_setup_snapshot_sync
from ...services.uploaded_file_store import (
    StagedUploadedFile,
    compensate_staged_uploaded_files,
    stage_uploaded_file_from_local_path,
)
from ..batch_control import BatchChannelControl
from .trace_handler import FeishuTraceHandler

logger = logging.getLogger(__name__)


class FeishuBotInstance(BatchChannelControl[str]):
    control_label = "Feishu"

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
        self._initialize_batch_control()
        self.user_active_trace_handlers: dict[str, FeishuTraceHandler] = {}
        self.control_tasks: set[asyncio.Task] = set()
        self.control_queues: dict[str, deque[tuple[Any, str | None]]] = {}
        self.control_locks: dict[str, asyncio.Lock] = {}
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

    def _save_active_tasks(self) -> bool:
        temporary = self.active_tasks_file.with_suffix(".json.tmp")
        try:
            self.active_tasks_file.parent.mkdir(parents=True, exist_ok=True)
            with temporary.open("w") as f:
                json.dump(self.active_tasks, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, self.active_tasks_file)
            return True
        except Exception:
            logger.exception("Error saving Feishu active tasks")
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
            return False

    def _active_trace_handlers(self) -> dict[str, FeishuTraceHandler]:
        return self.user_active_trace_handlers

    @staticmethod
    def _control_command(data: Any) -> str | None:
        message = data.event.message
        if message.message_type != "text":
            return None
        try:
            text = json.loads(message.content).get("text", "").strip()
        except (ValueError, AttributeError, TypeError):
            return None
        return text if text in {"/start", "/help", "/new", "/stop", "/pause"} else None

    async def _handle_control(self, open_id: str, data: Any, command: str) -> None:
        chat_id = data.event.message.chat_id
        lock = self.control_locks.setdefault(open_id, asyncio.Lock())
        try:
            async with lock:
                await authorize_channel_sender(
                    channel_id=self.channel_id, external_user_id=open_id
                )
                if not self._accepting:
                    return
                if command in {"/start", "/help"}:
                    reply = "Send a message to begin, /new for a fresh task, or /stop to pause the current run."
                elif command == "/new":
                    previous = self.active_tasks.get(open_id)
                    self.active_tasks[open_id] = "-1"
                    if not self._save_active_tasks():
                        if previous is None:
                            self.active_tasks.pop(open_id, None)
                        else:
                            self.active_tasks[open_id] = previous
                        reply = "I couldn't save the new conversation. The current task is still active. Please try again."
                    else:
                        self.user_conversation_generations[open_id] = (
                            self._conversation_generation(open_id) + 1
                        )
                        self._request_current_conversation_stop(
                            open_id, reason="new Feishu conversation requested"
                        )
                        if (
                            previous is not None
                            and int(previous) > 0
                            and self.channel_id is not None
                        ):
                            try:
                                await discard_channel_task_results(
                                    channel_id=self.channel_id,
                                    external_user_id=open_id,
                                    task_id=int(previous),
                                )
                            except Exception:
                                logger.exception("Failed to discard old Feishu replies")
                                await self._send_text(
                                    chat_id,
                                    "The new conversation is selected, but I couldn't finish cleaning up the previous replies. You can send your new request.",
                                )
                                return
                        reply = "Started a new task. Please describe your request."
                else:
                    stopped = self._stop_current_conversation(open_id)
                    reply = (
                        "Stopped the current run. Send another message to continue here, or /new for a fresh task."
                        if stopped
                        else "No active run to stop."
                    )
            await self._send_text(chat_id, reply)
        except ChannelAuthorizationError:
            await self._send_text(chat_id, "🚫 You are not authorized to use this bot.")
        except ChannelConfigurationError:
            await self._send_text(
                chat_id, "This bot is inactive or not correctly configured."
            )
        except Exception:
            logger.exception("Failed to handle Feishu control command")
            await self._send_text(
                chat_id, "I couldn't complete that command. Please try again."
            )

    def _handle_message_sync(self, data: Any) -> None:
        if not self._accepting:
            return

        loop = asyncio.get_event_loop()
        if loop.is_running():
            event = data.event
            if not event or not event.message or not event.sender:
                return

            command = self._control_command(data)
            # Shared ordinary inputs resolve durable receipts after restart.
            # Controls have no receipt and must not be replayed from before startup.
            if (
                (command is not None or not get_shared_task_execution_enabled())
                and hasattr(event.message, "create_time")
                and event.message.create_time
            ):
                try:
                    if int(event.message.create_time) < self.start_time:
                        logger.info(
                            f"Ignoring stale message from {event.message.create_time} (bot started at {self.start_time})"
                        )
                        return
                except (ValueError, TypeError):
                    pass

            open_id = event.sender.sender_id.open_id

            if command is not None or open_id in self.control_queues:
                if open_id not in self.control_queues:
                    self.control_queues[open_id] = deque()
                    task = loop.create_task(self._process_control_queue(open_id))
                    self.control_tasks.add(task)
                    task.add_done_callback(self.control_tasks.discard)
                self.control_queues[open_id].append((data, command))
                return
            self._enqueue_user_message(open_id, data)
        else:
            logger.error("No running event loop to schedule Feishu message processing")

    async def _process_control_queue(self, open_id: str) -> None:
        # Serialize authorization and following inputs, never the execution itself.
        # Later messages cannot enter the old queue while a command awaits the DB.
        queue = self.control_queues[open_id]
        try:
            while queue and self._accepting:
                data, command = queue.popleft()
                if command is None:
                    self._enqueue_user_message(open_id, data)
                else:
                    await self._handle_control(open_id, data, command)
        finally:
            self.control_queues.pop(open_id, None)

    async def _process_queued_batch(self, open_id: str, messages: list[Any]) -> None:
        await self._process_messages_batch(open_id, messages)

    def _shared_input(self, open_id: str, data: Any) -> ChannelInput:
        message = data.event.message
        if self.channel_id is None:
            raise ChannelConfigurationError("Channel is not configured")
        if not message.message_id or not message.chat_id:
            raise TaskTurnError("input_identity_missing")
        text, files = self._message_content(message)
        if message.message_type in ("image", "audio", "media", "file") and not files:
            raise TaskTurnError("file_unavailable")
        return ChannelInput(
            self.channel_id,
            open_id,
            "feishu",
            (str(message.chat_id),),
            str(message.message_id),
            text,
            tuple(f"{item['type']}:{item['file_key']}" for item in files),
            {"chat_id": message.chat_id, "loading_message_id": None},
        )

    @staticmethod
    def _message_content(message: Any) -> tuple[str, list[dict[str, Any]]]:
        text = ""
        files = []
        content_str = getattr(message, "content", None) or ""
        try:
            content = json.loads(content_str)
            if message.message_type == "text":
                text = content.get("text", "").strip()
            elif message.message_type in ("image", "audio", "media", "file"):
                key = content.get(
                    "image_key" if message.message_type == "image" else "file_key"
                )
                if key:
                    files.append(
                        {
                            "type": message.message_type,
                            "file_key": key,
                            "message_id": message.message_id,
                        }
                    )
            else:
                text = f"Please process this {message.message_type}."
        except (ValueError, AttributeError, TypeError):
            text = content_str.strip()
        if not text and not files:
            text = f"Received a {message.message_type} message."
        return text, files

    async def _process_shared_messages_batch(
        self, open_id: str, messages: list[Any]
    ) -> None:
        generation = self._conversation_generation(open_id)
        self.user_preparing_executions.add(open_id)
        self._clear_user_stop_request(open_id)
        stop_event = self._get_user_stop_event(open_id)
        chat_id = messages[0].event.message.chat_id

        def stopped() -> bool:
            return (
                stop_event.is_set()
                or self._conversation_generation(open_id) != generation
            )

        try:
            # Preserve arrival order and per-user task selection, but never mix
            # provider scopes inside one atomic acceptance batch.
            for chat_id, group in groupby(
                messages, key=lambda item: item.event.message.chat_id
            ):
                if stopped():
                    return
                unobserved_replays: deque[AcceptedChannelInput] = deque()
                try:
                    raw = list(group)
                    incoming = tuple(self._shared_input(open_id, item) for item in raw)
                    files = {
                        str(item.event.message.message_id): self._message_content(
                            item.event.message
                        )[1]
                        for item in raw
                    }
                    observed: set[int] = set()
                    notified_rejections: set[tuple[str, str]] = set()
                    while not stopped():
                        lookup, cancellation = await await_task_settlement(
                            asyncio.create_task(
                                asyncio.to_thread(lookup_channel_inputs, incoming)
                            )
                        )
                        owner_id, pending, replays, rejected = lookup
                        unobserved_replays.extend(replays)
                        if cancellation is not None:
                            raise cancellation
                        if stopped():
                            return
                        for rejection in rejected:
                            if stopped():
                                return
                            key = (rejection.incoming.message_id, rejection.reason)
                            if key not in notified_rejections:
                                await self._send_text(
                                    chat_id, self._input_error_message(rejection.reason)
                                )
                                notified_rejections.add(key)
                        while unobserved_replays:
                            if stopped():
                                return
                            accepted = unobserved_replays.popleft()
                            if accepted.command_db_id not in observed:
                                observed.add(accepted.command_db_id)
                                await self._observe_shared_input(
                                    open_id, accepted, generation
                                )
                        if not pending or stopped():
                            break
                        get_task_event_bridge().require_ready()
                        staged: list[StagedUploadedFile] = []
                        try:
                            with TemporaryDirectory(
                                prefix="xagent-feishu-input-"
                            ) as staging_dir:
                                for item in pending:
                                    for file_info in files[item.message_id]:
                                        downloaded = (
                                            await drain_async_task_cancellation_safe(
                                                asyncio.create_task(
                                                    asyncio.to_thread(
                                                        self._download_feishu_file_sync,
                                                        file_info,
                                                        Path(staging_dir),
                                                    )
                                                )
                                            )
                                        )
                                        if downloaded is None:
                                            raise TaskTurnError("file_unavailable")
                                        if stopped():
                                            return
                                        file_id = str(uuid4())
                                        (
                                            uploaded,
                                            cancellation,
                                        ) = await await_task_settlement(
                                            asyncio.create_task(
                                                asyncio.to_thread(
                                                    stage_uploaded_file_from_local_path,
                                                    local_path=downloaded.path,
                                                    user_id=owner_id,
                                                    filename=downloaded.name,
                                                    file_id=file_id,
                                                    mime_type=downloaded.mime_type,
                                                    storage_key=build_upload_storage_key(
                                                        owner_id,
                                                        file_id,
                                                        downloaded.name,
                                                    ),
                                                    upload_source="feishu",
                                                    execution_scope=None,
                                                )
                                            )
                                        )
                                        staged.append(uploaded)
                                        if cancellation is not None:
                                            raise cancellation
                                        if stopped():
                                            return
                                attachments = normalize_attachments_for_persistence(
                                    [
                                        {
                                            "file_id": item.file_id,
                                            "name": item.filename,
                                            "type": item.mime_type,
                                            "size": item.file_size,
                                        }
                                        for item in staged
                                    ]
                                )
                                text = "\n".join(
                                    item.text for item in pending if item.text
                                )
                                links = " ".join(
                                    f"[{item.filename}]({build_file_id_ref(item.file_id)})"
                                    for item in staged
                                )
                                if links:
                                    text = f"{text}\n\n{links}" if text else links
                                if stopped():
                                    return
                                active_task_id = self.active_tasks.get(open_id)
                                accepted, cancellation = await await_task_settlement(
                                    asyncio.create_task(
                                        asyncio.to_thread(
                                            accept_channel_input,
                                            pending[0],
                                            additional_inputs=pending[1:],
                                            owner_id=owner_id,
                                            active_task_id=int(active_task_id)
                                            if active_task_id is not None
                                            else None,
                                            channel_name=self.channel_name,
                                            payload=TaskTurnPayload(
                                                text,
                                                execution_message=text,
                                                attachments=attachments or None,
                                                file_ids=tuple(
                                                    item.file_id for item in staged
                                                ),
                                            ),
                                            staged_files=tuple(staged),
                                            host_id=get_task_event_bridge().host_id,
                                        )
                                    )
                                )
                                # Inspect late commits before propagating cancellation:
                                # /new must never restore an older task selection.
                                saved = True
                                if (
                                    self._conversation_generation(open_id) == generation
                                    and not accepted.replayed
                                    and accepted.selection.is_new_task
                                ):
                                    self.active_tasks[open_id] = str(accepted.task_id)
                                    saved = self._save_active_tasks()
                                if stopped():
                                    turn = accepted.as_turn()
                                    turn.discard_output = (
                                        self._conversation_generation(open_id)
                                        != generation
                                    )
                                    await drain_async_task_cancellation_safe(
                                        asyncio.create_task(turn.stop())
                                    )
                                if cancellation is not None:
                                    raise cancellation
                                if stopped():
                                    return
                                await self._observe_shared_input(
                                    open_id,
                                    accepted,
                                    generation,
                                    save_failure_chat=chat_id if not saved else None,
                                )
                                break
                        except ChannelInputBatchChanged:
                            # A competing acceptance changed the partition. Reload
                            # all physical inputs; never replay a new START ourselves.
                            # Compensate this attempt's unused uploads below, then
                            # download only still-pending inputs again. Keeping files
                            # across repartition would require separate ownership.
                            continue
                        finally:
                            if staged:
                                await run_db_io_cancellation_safe(
                                    lambda: compensate_staged_uploaded_files(
                                        tuple(staged)
                                    )
                                )
                except TaskTurnError as error:
                    if error.reason != "file_unavailable":
                        raise
                    if stopped():
                        return
                    await self._send_text(
                        chat_id,
                        "I couldn't process the attached Feishu file(s). "
                        "New messages in this group were not accepted. "
                        "Please resend them together.",
                    )
                finally:
                    # Keep responsibility for accepted work until observation
                    # takes over, even when lookup or a notice is cancelled.
                    if stopped() and unobserved_replays:

                        async def stop_replays() -> None:
                            for replay in unobserved_replays:
                                turn = replay.as_turn()
                                turn.discard_output = (
                                    self._conversation_generation(open_id) != generation
                                )
                                await turn.stop()

                        await drain_async_task_cancellation_safe(
                            asyncio.create_task(stop_replays())
                        )
        except ChannelAuthorizationError:
            await self._send_text(chat_id, "🚫 You are not authorized to use this bot.")
        except ChannelConfigurationError:
            await self._send_text(
                chat_id, "This bot is inactive or not correctly configured."
            )
        except TaskTurnError as error:
            if not stopped():
                await self._send_text(chat_id, self._input_error_message(error.reason))
        except Exception:
            logger.exception("Error accepting or observing Feishu input")
            if not stopped():
                await self._send_text(
                    chat_id, "Sorry, an error occurred while processing your request."
                )
        finally:
            self.user_preparing_executions.discard(open_id)
            self._clear_user_stop_request(open_id)

    @staticmethod
    def _input_error_message(reason: str) -> str:
        return {
            "busy": "I'm still working on the previous message. Please wait for it to finish.",
            "input_conflict": "This Feishu message was already accepted with different content. Please send a new message.",
            "input_unavailable": "The original task is no longer available. Please send a new message.",
            "file_unavailable": "I couldn't download the attached Feishu file(s). Please try uploading them again.",
        }.get(reason, "This message could not be accepted. Please try again.")

    async def _observe_shared_input(
        self,
        open_id: str,
        accepted: AcceptedChannelInput,
        generation: int,
        *,
        save_failure_chat: str | None = None,
    ) -> None:
        turn = accepted.as_turn()
        active = (accepted.task_id, turn)
        retained = False
        handler = None

        def current() -> bool:
            selected = self.active_tasks.get(open_id)
            return self._conversation_generation(open_id) == generation and (
                selected is None or selected == str(accepted.task_id)
            )

        async def deliver(delivery: ChannelDelivery, result: dict[str, Any]) -> None:
            await self._deliver_shared_result(delivery, result, is_current=current)

        async def progress(text: str | None) -> None:
            async def send(delivery: ChannelDelivery, _: Any) -> None:
                if not current():
                    raise ChannelDeliveryDiscarded
                destination = delivery.destination
                if not destination["loading_message_id"]:
                    destination["loading_message_id"] = await self._send_text(
                        destination["chat_id"],
                        f"⏳ **Task #{accepted.task_id} is processing...**\n_Please wait for the result._",
                    )
                    if not destination["loading_message_id"]:
                        raise ConnectionError("Feishu loading message was not accepted")
                if (
                    text is not None
                    and current()
                    and handler is not None
                    and not handler.cancelled
                ):
                    await self._update_text(
                        destination["chat_id"],
                        destination["loading_message_id"],
                        text,
                        require_delivery=True,
                        is_current=current,
                    )

            await DurableChannelProgress(accepted.command_db_id, send, deliver).send()

        try:
            if (
                self._conversation_generation(open_id) != generation
                or self._get_user_stop_event(open_id).is_set()
            ):
                turn.discard_output = (
                    self._conversation_generation(open_id) != generation
                )
                await turn.stop()
                return
            if not current():
                await turn.discard_delivery()
                return
            previous = self.user_active_executions.get(open_id)
            if (
                accepted.replayed
                and previous is not None
                and isinstance(previous[1], SharedChannelTurn)
                and previous[1].command_db_id is not None
                and previous[1].command_db_id > accepted.command_db_id
            ):
                # A historical receipt may retry delivery, but cannot take
                # trace or control ownership from a newer retained command.
                await turn.deliver(deliver)
                return
            self.user_active_executions[open_id] = active
            # An accepted command remains controllable if observation fails
            # before its terminal outcome is known.
            retained = True
            handler = FeishuTraceHandler(
                accepted.task_id, self.api_client, "", send_update=progress
            )
            self.user_active_trace_handlers[open_id] = handler
            if save_failure_chat is not None:
                await self._send_text(
                    save_failure_chat,
                    f"Your request was accepted as task #{accepted.task_id}, but I couldn't save the current conversation locally. "
                    "Retrying this same message will reuse the accepted task.",
                )
            await progress(None)
            result = await turn.observe(handler)
            if not current():
                await turn.discard_delivery()
                retained = False
                return
            retained = result.get("status") == "accepted"
            delivered = await turn.deliver(
                deliver, pending_notice=result.get("status") == "accepted"
            )
            retained = (
                result.get("status") == "accepted" and not delivered and current()
            )
        except asyncio.CancelledError:
            # Ending an ingress observer does not cancel durable worker work.
            # Explicit controls request their own stop; close drains that request.
            raise
        except Exception:
            logger.exception(
                "Error observing accepted Feishu input command_id=%s",
                accepted.command_db_id,
            )
        finally:
            if self.user_active_trace_handlers.get(open_id) is handler:
                self.user_active_trace_handlers.pop(open_id, None)
            if self.user_active_executions.get(open_id) == active and (
                not retained or not current()
            ):
                self.user_active_executions.pop(open_id, None)
            await drain_async_task_cancellation_safe(asyncio.create_task(turn.close()))

    async def _process_messages_batch(
        self, open_id: str, messages_data: list[Any]
    ) -> None:
        if get_shared_task_execution_enabled():
            await self._process_shared_messages_batch(open_id, messages_data)
            return
        chat_id = messages_data[0].event.message.chat_id
        claimed_task_id: int | None = None
        managed_lease: ManagedTaskLease | None = None
        agent_service = None
        active_execution: tuple[int, object] | None = None
        fs_handler: FeishuTraceHandler | None = None
        self.user_preparing_executions.add(open_id)
        self._clear_user_stop_request(open_id)
        generation = self._conversation_generation(open_id)

        async def interrupted() -> bool:
            stopped = self._consume_user_stop_request(open_id)
            if not stopped and self._conversation_generation(open_id) == generation:
                return False
            try:
                if managed_lease is not None:
                    await managed_lease.finalize_result(status=TaskStatus.PAUSED)
            except Exception:
                logger.warning(
                    "Failed to settle interrupted Feishu preparation", exc_info=True
                )
            return True

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
            selected_task = prepared_task
            managed_lease = prepared_task.managed_lease
            if await interrupted():
                return
            task_id = selected_task.task_id
            claimed_task_id = task_id
            owner_user_id = selected_task.user_id
            is_new_task = selected_task.is_new_task
            if is_new_task:
                self.active_tasks[open_id] = str(task_id)
                if not self._save_active_tasks():
                    if active_task_id is None:
                        self.active_tasks.pop(open_id, None)
                    else:
                        self.active_tasks[open_id] = active_task_id
                    if managed_lease is not None:
                        await managed_lease.finalize_result(status=TaskStatus.PAUSED)
                    await self._send_text(
                        chat_id,
                        "I couldn't save the new conversation. Your request wasn't started. Please try again.",
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

            if await interrupted():
                return
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
                if await interrupted():
                    return
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

            if await interrupted():
                return
            await persist_channel_user_message(
                task_id=task_id,
                user_id=owner_user_id,
                content=text,
                attachments=persisted_attachments or None,
                turn_id=message_turn_id,
            )

            if await interrupted():
                return
            loading_msg_id = await self._send_text(
                chat_id,
                f"⏳ **Task #{task_id} is processing...**\n_Please wait for the result._",
            )

            if await interrupted():
                return
            if loading_msg_id:
                fs_handler = FeishuTraceHandler(
                    task_id, self.api_client, chat_id, loading_msg_id
                )
                self.user_active_trace_handlers[open_id] = fs_handler
                if agent_service is not None:
                    agent_service.tracer.add_handler(fs_handler)

            active_execution = (
                task_id,
                agent_service,
            )
            self.user_active_executions[open_id] = active_execution
            local_service: Any = agent_service
            local_lease = cast(ManagedTaskLease, managed_lease)
            from ...user_isolated_memory import UserContext

            actual_task_id = str(task_id)
            try:
                with UserContext(owner_user_id):
                    result = await self._await_execution_with_stop_monitor(
                        open_id,
                        agent_manager.execute_task(
                            agent_service=local_service,
                            task=text,
                            context=context,
                            task_id=actual_task_id,
                            tracking_task_id=actual_task_id,
                            db_session=None,
                            manage_task_lease=False,
                            task_lease=local_lease.lease,
                            task_lease_heartbeat_task=local_lease.heartbeat_task,
                        ),
                        reason="Feishu stop requested",
                    )
            finally:
                if fs_handler is not None:
                    local_service.tracer.remove_handler(fs_handler)

            projection = project_execution_result_for_channel(result)
            if managed_lease is not None and not await managed_lease.finalize_result(
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

            if self._conversation_generation(open_id) != generation:
                return
            output = projection.visible_text

            max_len = 4000
            text_chunks = [
                output[i : i + max_len] for i in range(0, len(output), max_len)
            ]

            if loading_msg_id:
                await self._update_text(
                    chat_id,
                    loading_msg_id,
                    text_chunks[0],
                    is_current=lambda: self._conversation_generation(open_id)
                    == generation,
                )
            else:
                await self._send_text(chat_id, text_chunks[0])

            for chunk in text_chunks[1:]:
                if self._conversation_generation(open_id) != generation:
                    return
                await self._send_text(chat_id, chunk)

        except TaskLeaseLostError:
            logger.warning(
                "Feishu execution lost task %s lease; skipping stale result",
                claimed_task_id,
            )
        except Exception as e:
            logger.error(f"Error processing Feishu message: {e}", exc_info=True)
            if active_execution is None and await interrupted():
                return
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
            if self._conversation_generation(open_id) != generation:
                return
            await self._send_text(
                chat_id,
                CLIENT_SAFE_AUTO_MODEL_UNAVAILABLE
                if isinstance(e, AutoModelUnavailableError)
                else "Sorry, an error occurred while processing your request.",
            )
        finally:
            if fs_handler is not None:
                self.user_active_trace_handlers.pop(open_id, None)
            if (
                active_execution is not None
                and self.user_active_executions.get(open_id) == active_execution
            ):
                self.user_active_executions.pop(open_id, None)

            async def cleanup() -> None:
                try:
                    if managed_lease is not None:
                        await managed_lease.close()
                finally:
                    self.user_preparing_executions.discard(open_id)
                    self._clear_user_stop_request(open_id)

            await drain_async_task_cancellation_safe(asyncio.create_task(cleanup()))

    async def _deliver_shared_result(
        self,
        delivery: ChannelDelivery,
        result: dict[str, Any],
        *,
        is_current: Callable[[], bool] | None = None,
    ) -> None:
        if is_current is None:
            selected_task = self.active_tasks.get(delivery.external_user_id)
            if selected_task is not None and selected_task != str(delivery.task_id):
                raise ChannelDeliveryDiscarded
            generation = self._conversation_generation(delivery.external_user_id)

            def is_current() -> bool:
                return (
                    self._conversation_generation(delivery.external_user_id)
                    == generation
                )

        if not is_current():
            raise ChannelDeliveryDiscarded
        projection = project_execution_result_for_channel(result)
        chat_id = delivery.destination["chat_id"]
        loading_id = delivery.destination["loading_message_id"]
        chunks = [
            projection.visible_text[i : i + 4000]
            for i in range(0, len(projection.visible_text), 4000)
        ]
        if loading_id:
            await self._update_text(
                chat_id,
                loading_id,
                chunks[0],
                require_delivery=True,
                is_current=is_current,
            )
        elif not await self._send_text(chat_id, chunks[0]):
            raise ConnectionError("Feishu final message was not accepted")
        if not is_current():
            raise ChannelDeliveryDiscarded
        for chunk in chunks[1:]:
            if not is_current():
                raise ChannelDeliveryDiscarded
            if not await self._send_text(chat_id, chunk):
                raise ConnectionError("Feishu final message was not accepted")

    async def _download_and_register_files(
        self,
        files_info: list,
        agent_service: "Any",
        task_id: int,
        user_id: int,
        workspace: Any = None,
    ) -> list:
        workspace = workspace if workspace is not None else agent_service.workspace
        if not workspace:
            logger.warning("Agent service workspace is not available for file upload")
            return []

        target_dir = getattr(
            workspace,
            "input_dir",
            workspace.workspace_dir / "input",
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
            workspace=workspace,
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

        from ...api.websocket import build_unique_target_path
        from ...services.task_execution import normalize_filename

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

    async def _update_text(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        *,
        require_delivery: bool = False,
        is_current: Callable[[], bool] | None = None,
    ) -> None:
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
                if is_current is not None and not is_current():
                    return
                if resp.code == 230001:  # "This message is NOT a card." error
                    logger.info("Falling back to send_text instead of update_text")
                    sent = await self._send_text(chat_id, text)
                    if sent:
                        return
                if require_delivery:
                    raise ConnectionError(
                        "Feishu final message update was not accepted"
                    )
        except Exception as e:
            if require_delivery:
                raise
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
            task
            for task in (*self.user_message_tasks.values(), *self.control_tasks)
            if task is not current
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
            self.control_tasks.clear()
            self.control_queues.clear()
            self.control_locks.clear()
            self.user_active_executions.clear()
            self.user_active_trace_handlers.clear()
            self.user_preparing_executions.clear()
            self.user_stop_events.clear()
            self.user_conversation_generations.clear()

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
        if not get_channel_ingress_enabled():
            logger.info("Feishu channel ingress disabled on this host")
            return
        while get_channel_ingress_enabled():
            await self._sync_bots_async()
            if not get_shared_task_execution_enabled():
                return
            for bot in tuple(self.bots.values()):
                if bot.channel_id is not None:
                    await recover_channel_results(
                        bot.channel_id, bot._deliver_shared_result
                    )
            # CRUD may reach another web replica. The designated ingress
            # observes its committed configuration without opening extra bots.
            await asyncio.sleep(30)

    async def stop(self) -> None:
        for app_id in list(self.bots.keys()):
            await self._stop_bot_for_appid(app_id)

    async def _sync_bots_async(self) -> None:
        if not get_channel_ingress_enabled():
            return
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
