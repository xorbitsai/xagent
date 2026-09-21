"""Channel ingress preparation and worker execution, joined by durable START."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, cast
from uuid import uuid4

from sqlalchemy import exists, select
from sqlalchemy.exc import InterfaceError, OperationalError
from sqlalchemy.exc import TimeoutError as DatabaseTimeoutError
from sqlalchemy.orm import Session

from ...config import get_task_reply_wait_timeout_seconds
from ...core.agent.trace import (
    TraceAction,
    TraceCategory,
    TraceEvent,
    TraceEventType,
    TraceHandler,
    TraceScope,
)
from ...core.execution_scope import resolve_execution_scope
from ...core.runtime_performance import increment_counter
from ...core.workspace import TaskWorkspace
from ..models.database import get_session_local
from ..models.task import Task, TaskStatus
from ..models.task_channel_delivery import TaskChannelDelivery
from ..models.task_command import TaskExecutionCommand
from ..models.uploaded_file import UploadedFile
from .channel_delivery import (
    PENDING_CHANNEL_RESULT,
    ChannelSender,
    deliver_channel_result,
)
from .channel_runtime import (
    SelectedChannelTask,
    _load_channel_owner_sync,
    _prepare_channel_task_sync,
)
from .chat_history_service import persist_user_message_no_commit
from .db_runtime import await_task_settlement, run_db_io_cancellation_safe
from .task_command_transport import (
    ClaimedTaskCommand,
    TaskCommandKind,
    notify_task_command_dispatcher,
    stage_task_command,
)
from .task_event_bridge import TaskReplyRouteUnavailable, get_task_event_bridge
from .task_lease_service import TaskLease, TaskLeaseLostError
from .task_orchestrator import (
    TaskTurnError,
    TaskTurnPayload,
    reserve_task_start_no_commit,
)
from .task_start_protocol import (
    ChannelExecutionContext,
    TaskStartPayload,
    stage_task_start_command,
)
from .workspace_binding import canonical_workspace_base

logger = logging.getLogger(__name__)


def _settle_pending_selection(selection: SelectedChannelTask) -> None:
    if not selection.is_new_task:
        return
    with get_session_local()() as db:
        pending = db.query(Task).filter(
            Task.id == selection.task_id,
            Task.user_id == selection.user_id,
            Task.status == TaskStatus.PENDING,
            Task.run_id == selection.previous_run_id,
            Task.state_version == selection.state_version,
            Task.runner_id.is_(None),
            ~exists(select(1).where(TaskExecutionCommand.task_id == selection.task_id)),
        )
        # Selection may already own uploaded files before START is accepted.
        # Both stop and failed acceptance must keep the conversation resumable.
        pending.update(
            {
                Task.status: TaskStatus.PAUSED,
                Task.control_state: "paused",
                Task.run_id: str(uuid4()),
                Task.state_version: Task.state_version + 1,
            },
            synchronize_session=False,
        )
        db.commit()


def _channel_workspace(selection: SelectedChannelTask) -> TaskWorkspace:
    scope = resolve_execution_scope(selection.task_id)
    segments = scope.workspace_segments if scope is not None else ()
    workspace = TaskWorkspace(
        f"web_task_{selection.task_id}",
        base_dir=canonical_workspace_base(selection.user_id, segments),
        db_task_id=selection.task_id,
        scope_segments=segments,
        durable_storage_segments=scope.durable_storage_segments
        if scope is not None
        else (),
    )
    workspace.owner_user_id = selection.user_id
    return workspace


@dataclass
class SharedChannelTurn:
    """An ingress-owned selection and stop signal, never an Agent or lease."""

    selection: SelectedChannelTask
    workspace: TaskWorkspace | None
    run_id: str = field(default_factory=lambda: str(uuid4()))
    command_id: str = field(default_factory=lambda: uuid4().hex)
    accepted: bool = False
    stop_requested: bool = False
    origin: str | None = None
    stop_task: asyncio.Task | None = None
    delivery_destination: dict[str, Any] | None = None
    command_db_id: int | None = None
    discard_output: bool = False

    async def deliver(
        self, sender: ChannelSender, *, pending_notice: bool = False
    ) -> bool:
        assert self.command_db_id is not None
        return await deliver_channel_result(
            self.command_db_id, sender, pending_notice=pending_notice
        )

    async def discard_delivery(self) -> None:
        if self.command_db_id is None:
            return

        def discard() -> None:
            with get_session_local()() as db:
                db.query(TaskChannelDelivery).filter(
                    TaskChannelDelivery.command_id == self.command_db_id,
                    TaskChannelDelivery.status == "pending",
                ).update(
                    {"status": "discarded", "claim_token": None},
                    synchronize_session=False,
                )
                db.commit()

        await run_db_io_cancellation_safe(discard)

    def request_stop(self) -> bool:
        self.stop_requested = True
        if self.accepted and (self.stop_task is None or self.stop_task.done()):
            self.stop_task = asyncio.create_task(self._enqueue_stop())
        return True

    async def _enqueue_stop(self) -> None:
        if self.discard_output:
            await self.discard_delivery()

        def enqueue() -> None:
            with get_session_local()() as db:
                task = db.execute(
                    select(Task)
                    .where(Task.id == self.selection.task_id)
                    .with_for_update()
                ).scalar_one_or_none()
                if task is None:
                    return
                if task.run_id != self.run_id:
                    queued = (
                        db.query(TaskExecutionCommand.id)
                        .filter(
                            TaskExecutionCommand.task_id == task.id,
                            TaskExecutionCommand.command_id == self.command_id,
                            TaskExecutionCommand.target_run_id == self.run_id,
                            TaskExecutionCommand.status.in_(("pending", "processing")),
                        )
                        .first()
                    )
                    if queued is None:
                        return
                stage_task_command(
                    db,
                    task_id=int(task.id),
                    actor_user_id=self.selection.user_id,
                    command_id=f"stop:{self.command_id}",
                    kind=TaskCommandKind.PAUSE,
                    payload={},
                    target_run_id=self.run_id,
                )
                db.commit()

        await run_db_io_cancellation_safe(enqueue)
        notify_task_command_dispatcher()

    async def stop(self) -> None:
        self.request_stop()
        if self.stop_task is not None:
            await self.stop_task
        elif not self.accepted:
            await run_db_io_cancellation_safe(
                lambda: _settle_pending_selection(self.selection)
            )

    async def close(self) -> None:
        if self.origin is not None:
            get_task_event_bridge().discard_origin(self.origin)
        if self.stop_task is not None:
            await asyncio.gather(self.stop_task, return_exceptions=True)
        if not self.accepted:
            await run_db_io_cancellation_safe(
                lambda: _settle_pending_selection(self.selection)
            )

    def register_trace_handler(self, trace_handler: TraceHandler | None) -> None:
        bridge = get_task_event_bridge()
        bridge.require_ready()

        async def receive(message: dict[str, Any]) -> None:
            if message.get("run_id") != self.run_id or trace_handler is None:
                return
            if message.get("type") == "channel_trace":
                raw = message["trace"]
                event = TraceEvent(
                    TraceEventType(
                        TraceScope(raw["scope"]),
                        TraceAction(raw["action"]),
                        TraceCategory(raw["category"]),
                    ),
                    task_id=raw.get("task_id"),
                    step_id=raw.get("step_id"),
                    timestamp=raw["timestamp"],
                    data=raw.get("data"),
                    parent_id=raw.get("parent_id"),
                )
                event.id = raw["id"]
                await trace_handler.handle_event(event)

        self.origin = bridge.register_origin(
            self.selection.task_id, self.command_id, receive, recipient=self
        )

    async def observe(self, trace_handler: TraceHandler | None) -> dict[str, Any]:
        """Attach to an already accepted command without accepting it again."""
        self.register_trace_handler(trace_handler)

        def attach() -> None:
            with get_session_local()() as db:
                db.query(TaskExecutionCommand).filter(
                    TaskExecutionCommand.id == self.command_db_id,
                    TaskExecutionCommand.command_id == self.command_id,
                ).update(
                    {
                        "reply_host_id": get_task_event_bridge().host_id,
                        "reply_origin": self.origin,
                    },
                    synchronize_session=False,
                )
                db.commit()

        await run_db_io_cancellation_safe(attach)
        notify_task_command_dispatcher()
        return await self.wait_result()

    async def execute(
        self, payload: TaskTurnPayload, trace_handler: TraceHandler | None
    ) -> dict[str, Any]:
        self.register_trace_handler(trace_handler)
        bridge = get_task_event_bridge()
        if self.stop_requested:
            return {"success": True, "status": "interrupted"}
        acceptance = asyncio.create_task(
            asyncio.to_thread(_accept_channel_turn, self, payload, bridge.host_id)
        )
        command_db_id, cancellation = await await_task_settlement(acceptance)
        self.accepted = True
        self.command_db_id = command_db_id
        notify_task_command_dispatcher()
        if cancellation is not None:
            self.request_stop()
            raise cancellation
        if self.stop_requested:
            self.request_stop()
        return await self.wait_result()

    async def wait_result(self) -> dict[str, Any]:
        assert self.command_db_id is not None
        command_db_id = self.command_db_id
        unavailable_since: float | None = None
        retry_delay = 0.25
        loop = asyncio.get_running_loop()
        deadline = loop.time() + get_task_reply_wait_timeout_seconds()
        while True:
            try:
                result = await run_db_io_cancellation_safe(
                    lambda: _read_channel_result(command_db_id, self.run_id)
                )
            except (DatabaseTimeoutError, OperationalError, InterfaceError):
                if unavailable_since is None:
                    unavailable_since = loop.time()
                    logger.warning(
                        "Channel result query unavailable; retrying task_id=%s",
                        self.selection.task_id,
                    )
                if loop.time() >= deadline:
                    return dict(PENDING_CHANNEL_RESULT)
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 5.0)
                continue
            except TaskLeaseLostError:
                return {"success": True, "status": "interrupted"}
            unavailable_since = None
            retry_delay = 0.25
            if result is not None:
                return result
            if loop.time() >= deadline:
                return dict(PENDING_CHANNEL_RESULT)
            await asyncio.sleep(0.25)


async def prepare_shared_channel_turn(
    *,
    channel_id: int | None,
    external_user_id: str,
    active_task_id: int | None,
    text: str,
    channel_name: str | None,
    expected_owner_user_id: int | None = None,
    agent_id: int | None = None,
) -> SharedChannelTurn | None:
    get_task_event_bridge().require_ready()
    worker = asyncio.create_task(
        asyncio.to_thread(
            _prepare_channel_task_sync,
            channel_id=channel_id,
            external_user_id=external_user_id,
            active_task_id=active_task_id,
            text=text,
            channel_name=channel_name,
            expected_owner_user_id=expected_owner_user_id,
            agent_id=agent_id,
            defer_execution=True,
        )
    )
    selection, cancellation = await await_task_settlement(worker)
    assert selection is None or isinstance(selection, SelectedChannelTask)
    if selection is None:
        if cancellation is not None:
            raise cancellation
        return None
    if cancellation is not None:
        await run_db_io_cancellation_safe(lambda: _settle_pending_selection(selection))
        raise cancellation
    try:
        workspace = await run_db_io_cancellation_safe(
            lambda: _channel_workspace(selection)
        )
        return SharedChannelTurn(selection, workspace)
    except BaseException:
        await run_db_io_cancellation_safe(lambda: _settle_pending_selection(selection))
        raise


def accept_channel_turn_no_commit(
    db: Session, turn: SharedChannelTurn, payload: TaskTurnPayload, host_id: str
) -> int:
    """Stage START, transcript and delivery inside the caller transaction."""
    selection = turn.selection
    owner = _load_channel_owner_sync(
        db,
        channel_id=selection.channel_id,
        external_user_id=selection.external_user_id,
    )
    if owner.user_id != selection.user_id:
        raise TaskTurnError("owner_changed")
    task = db.execute(
        select(Task).where(Task.id == selection.task_id).with_for_update()
    ).scalar_one_or_none()
    if (
        task is None
        or task.channel_id != selection.channel_id
        or task.user_id != owner.user_id
    ):
        raise TaskTurnError("task_not_found")
    if (
        task.state_version != selection.state_version
        or task.run_id != selection.previous_run_id
        or not reserve_task_start_no_commit(
            db,
            task_id=int(task.id),
            task_owner_user_id=selection.user_id,
            statuses=(
                TaskStatus.PENDING,
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.PAUSED,
                TaskStatus.WAITING_FOR_USER,
            ),
        )
    ):
        raise TaskTurnError("busy")
    db.refresh(task)
    files = (
        db.query(UploadedFile)
        .filter(
            UploadedFile.file_id.in_(payload.file_ids),
            UploadedFile.user_id == owner.user_id,
            UploadedFile.task_id == task.id,
        )
        .all()
    )
    if len(files) != len(set(payload.file_ids)):
        raise TaskTurnError("file_unavailable")
    message = persist_user_message_no_commit(
        db,
        task_id=int(task.id),
        user_id=owner.user_id,
        content=payload.transcript_message,
        attachments=payload.attachments,
        turn_id=turn.command_id,
    )
    db.flush()
    start = TaskStartPayload(
        version=1,
        run_id=turn.run_id,
        expected_run_id=selection.previous_run_id,
        state_version=int(task.state_version),
        turn_id=turn.command_id,
        kind="channel",
        message=payload.transcript_message,
        execution_message=payload.execution_message,
        file_ids=list(payload.file_ids),
        before_message_id=int(message.id) if message is not None else None,
        channel=ChannelExecutionContext(
            channel_id=selection.channel_id,
            external_user_id=selection.external_user_id,
        ),
    )
    staged = stage_task_start_command(
        db,
        task_id=int(task.id),
        actor_user_id=owner.user_id,
        start=start,
        reply_host_id=host_id,
        reply_origin=turn.origin,
    )
    if turn.delivery_destination is not None:
        db.add(
            TaskChannelDelivery(
                command_id=staged.staged_db_id,
                channel_id=selection.channel_id,
                destination=turn.delivery_destination,
            )
        )
    return staged.staged_db_id


def _accept_channel_turn(
    turn: SharedChannelTurn, payload: TaskTurnPayload, host_id: str
) -> int:
    selection = turn.selection
    with get_session_local()() as db:
        command_db_id = accept_channel_turn_no_commit(db, turn, payload, host_id)
        expected_payload = cast(
            TaskExecutionCommand, db.get(TaskExecutionCommand, command_db_id)
        ).payload
        try:
            db.commit()
        except Exception:
            db.close()
            with get_session_local()() as check:
                saved = check.get(TaskExecutionCommand, command_db_id)
                if (
                    saved is None
                    or saved.command_id != turn.command_id
                    or saved.payload != expected_payload
                ):
                    raise
            logger.warning(
                "Channel acceptance recovered after uncertain commit task_id=%s command_id=%s",
                selection.task_id,
                turn.command_id,
            )
            increment_counter("xagent.channel.acceptance.commit_recovered")
        return command_db_id


def _read_channel_result(command_id: int, run_id: str) -> dict[str, Any] | None:
    with get_session_local()() as db:
        command = db.get(TaskExecutionCommand, command_id)
        if command is None or command.target_run_id != run_id:
            raise TaskLeaseLostError("Channel execution identity changed")
        result = cast(dict[str, Any], command.result or {})
        if "channel_result" in result:
            return cast(dict[str, Any], result["channel_result"])
        if command.status == "failed":
            return {
                "success": False,
                "status": "failed",
                "output": "Task could not be started.",
            }
        task = db.get(Task, command.task_id)
        if (
            task is not None
            and task.run_id != run_id
            and command.status in ("pending", "processing")
        ):
            return None
        if task is None or task.run_id != run_id:
            raise TaskLeaseLostError("Channel task changed before delivery")
        if task.runner_id is not None or task.status in (
            TaskStatus.RUNNING,
            TaskStatus.PENDING,
        ):
            return None
        # Crash recovery or a control command can settle without reaching the
        # execution leaf. Its persisted state still terminates this exact wait.
        if task.status == TaskStatus.PAUSED:
            return {"success": True, "status": "interrupted"}
        return {
            "success": task.status == TaskStatus.COMPLETED,
            "status": task.status.value,
            "output": task.output or "",
        }


class ChannelProgressForwarder(TraceHandler):
    def __init__(self, command: ClaimedTaskCommand, run_id: str) -> None:
        self.command = command
        self.run_id = run_id
        self._unavailable = False
        self._next_route_attempt = 0.0
        self._route_retry_delay = 1.0

    async def handle_event(self, event: TraceEvent) -> None:
        if self._unavailable or monotonic() < self._next_route_attempt:
            return
        try:
            await get_task_event_bridge().reply_for(
                self.command.command_id, self.command.task_id, require_ack=True
            )(
                {
                    "type": "channel_trace",
                    "run_id": self.run_id,
                    "trace": event.to_dict(),
                }
            )
            self._next_route_attempt = 0.0
            self._route_retry_delay = 1.0
        except TaskReplyRouteUnavailable:
            # Ingress may attach after acceptance. Retry on later traces, but
            # do not query/log every event if ingress never installs a route.
            self._next_route_attempt = monotonic() + self._route_retry_delay
            self._route_retry_delay = min(self._route_retry_delay * 2, 30.0)
            return
        except ConnectionError:
            self._unavailable = True
            logger.warning(
                "Channel progress forwarding stopped after delivery failure task_id=%s",
                self.command.task_id,
            )
            increment_counter("xagent.channel.progress.unavailable")
        except Exception:
            self._unavailable = True
            logger.exception(
                "Channel progress forwarding failed task_id=%s", self.command.task_id
            )
            increment_counter("xagent.channel.progress.error")


async def execute_channel_background(
    *,
    command: ClaimedTaskCommand,
    lease: TaskLease,
    heartbeat_task: asyncio.Task,
    snapshot: Any,
    payload: TaskTurnPayload,
) -> None:
    from ..user_isolated_memory import UserContext
    from .agent_service_manager import get_agent_manager
    from .execution_result_projection import project_execution_result_for_channel
    from .managed_task_lease import finalize_managed_task_lease_result
    from .task_execution_context_service import (
        materialize_task_execution_recovery_state,
    )

    manager = get_agent_manager()
    service = await manager.get_agent_for_task(
        command.task_id,
        user=snapshot.runtime_user,
        task_setup_snapshot=snapshot,
        task_owner_user_id=snapshot.task.user_id,
    )
    service.set_conversation_history(
        [dict(message) for message in snapshot.conversation_history],
        watermark=snapshot.conversation_watermark,
    )
    recovery = await materialize_task_execution_recovery_state(
        snapshot.execution_recovery
    )
    service.set_execution_context_messages(recovery.get("messages", []))
    service.set_recovered_skill_context(recovery.get("skill_context"))
    from .file_turn import resolve_turn_file_infos

    def resolve_files() -> list[dict[str, Any]]:
        with get_session_local()() as db:
            infos, missing = resolve_turn_file_infos(
                file_ids=list(payload.file_ids),
                owner_user_id=snapshot.task.user_id,
                task_id=command.task_id,
                db=db,
            )
            if missing:
                raise TaskTurnError("file_unavailable")
            return infos

    file_infos = (
        await run_db_io_cancellation_safe(resolve_files) if payload.file_ids else []
    )
    if file_infos:
        from .task_execution import _register_uploaded_files_for_agent

        await run_db_io_cancellation_safe(
            lambda: _register_uploaded_files_for_agent(service, file_infos)
        )
    from .file_turn import normalize_attachments_for_persistence

    context: dict[str, Any] = {
        "turn_id": command.command_id,
        "files": payload.attachments
        or normalize_attachments_for_persistence(file_infos),
        "display_message": payload.transcript_message,
    }
    if file_infos:
        context.update(
            {
                "file_info": file_infos,
                "uploaded_files": [info["path"] for info in file_infos],
                "state": {"file_info": file_infos},
            }
        )
    assert lease.run_id is not None
    forwarder = ChannelProgressForwarder(command, lease.run_id)
    service.tracer.add_handler(forwarder)
    try:
        with UserContext(snapshot.task.user_id):
            result = await manager.execute_task(
                agent_service=service,
                task=payload.for_agent,
                context=context,
                task_id=str(command.task_id),
                tracking_task_id=str(command.task_id),
                db_session=None,
                manage_task_lease=False,
                task_lease=lease,
                task_lease_heartbeat_task=heartbeat_task,
            )
        projection = project_execution_result_for_channel(result)
        durable_result = {
            "success": projection.task_status != TaskStatus.FAILED,
            "status": str(result.get("status") or projection.task_status.value),
            "output": projection.transcript_content,
            "chat_response": {
                "message": projection.transcript_content,
                "interactions": projection.interactions,
            },
        }

        def finalize() -> bool:
            with get_session_local()() as db:
                return finalize_managed_task_lease_result(
                    db,
                    lease,
                    status=projection.task_status,
                    assistant_content=projection.transcript_content,
                    turn_id=command.command_id,
                    interactions=projection.interactions,
                    message_type=projection.message_type,
                    error_message=projection.diagnostic_error,
                    completion=(command.id, durable_result),
                )

        if not await run_db_io_cancellation_safe(finalize):
            raise TaskLeaseLostError("Channel result no longer owns its execution")
    finally:
        service.tracer.remove_handler(forwarder)
        get_task_event_bridge().discard_command(command.command_id, command.task_id)
