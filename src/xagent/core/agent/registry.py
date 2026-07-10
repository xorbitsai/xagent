from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from .context import ExecutionContext

if TYPE_CHECKING:
    from .runner import AgentRunner

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ExecutionLifecycleStatus(str, Enum):
    """Lifecycle states exposed to callers of the execution registry."""

    REGISTERED = "registered"
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    WAITING_FOR_USER = "waiting_for_user"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExecutionHandle:
    """Live execution handle stored by the registry."""

    execution_id: str
    runner: "AgentRunner"
    task: asyncio.Task[dict[str, Any]] | None = None
    requested_task: str | None = None
    status: ExecutionLifecycleStatus = ExecutionLifecycleStatus.REGISTERED
    metadata: dict[str, Any] = field(default_factory=dict)
    last_result: dict[str, Any] | None = None
    last_error: str | None = None
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)

    @property
    def is_running(self) -> bool:
        return self.task is not None and not self.task.done()

    @property
    def is_resumable(self) -> bool:
        return self.status in {
            ExecutionLifecycleStatus.INTERRUPTED,
            ExecutionLifecycleStatus.WAITING_FOR_USER,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "requested_task": self.requested_task,
            "status": self.status.value,
            "is_running": self.is_running,
            "is_resumable": self.is_resumable,
            "metadata": dict(self.metadata),
            "last_error": self.last_error,
            "has_result": self.last_result is not None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class ExecutionRegistry:
    """Tracks active execution handles and routes control operations."""

    def __init__(self) -> None:
        self._handles: dict[str, ExecutionHandle] = {}
        self._subscribers: dict[int, Callable[[dict[str, Any]], Any]] = {}
        self._next_subscription_id = 1

    def subscribe(self, callback: Callable[[dict[str, Any]], Any]) -> int:
        subscription_id = self._next_subscription_id
        self._next_subscription_id += 1
        self._subscribers[subscription_id] = callback
        return subscription_id

    def unsubscribe(self, subscription_id: int) -> bool:
        return self._subscribers.pop(subscription_id, None) is not None

    def register(
        self,
        execution_id: str,
        runner: "AgentRunner",
        *,
        task: asyncio.Task[dict[str, Any]] | None = None,
        requested_task: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionHandle:
        handle = ExecutionHandle(
            execution_id=execution_id,
            runner=runner,
            task=task,
            requested_task=requested_task,
            status=(
                ExecutionLifecycleStatus.RUNNING
                if task is not None
                else ExecutionLifecycleStatus.REGISTERED
            ),
            metadata=dict(metadata or {}),
        )
        self._handles[execution_id] = handle
        if task is not None:
            task.add_done_callback(self._make_task_done_callback(execution_id))
        self._emit_event(
            "execution.started" if task is not None else "execution.registered",
            handle,
        )
        return handle

    def start(
        self,
        runner: "AgentRunner",
        *,
        execution_id: str,
        task: str | None,
        metadata: dict[str, Any] | None = None,
        **run_kwargs: Any,
    ) -> ExecutionHandle:
        run_task = asyncio.create_task(
            runner.run(
                task=task,
                execution_id=execution_id,
                metadata=metadata,
                **run_kwargs,
            )
        )
        return self.register(
            execution_id,
            runner,
            task=run_task,
            requested_task=task,
            metadata=metadata,
        )

    def get(self, execution_id: str) -> ExecutionHandle | None:
        return self._handles.get(execution_id)

    def get_status(self, execution_id: str) -> dict[str, Any] | None:
        handle = self.get(execution_id)
        if handle is None:
            return None
        return handle.to_dict()

    def unregister(self, execution_id: str) -> ExecutionHandle | None:
        return self._handles.pop(execution_id, None)

    def list_handles(self) -> list[ExecutionHandle]:
        return list(self._handles.values())

    def list_statuses(self) -> list[dict[str, Any]]:
        return [handle.to_dict() for handle in self.list_handles()]

    def pause(self, execution_id: str, reason: str | None = None) -> bool:
        handle = self.get(execution_id)
        if handle is None:
            return False
        handle.updated_at = _utcnow()
        return handle.runner.pause(execution_id, reason=reason)

    def cancel(self, execution_id: str, reason: str | None = None) -> bool:
        handle = self.get(execution_id)
        if handle is None:
            return False

        handle.updated_at = _utcnow()
        handle.runner.cancel(execution_id, reason=reason)
        if handle.task is not None and not handle.task.done():
            handle.status = ExecutionLifecycleStatus.CANCELLED
            handle.last_error = reason or "Execution was cancelled."
            handle.task.cancel()
            return True

        handle.status = ExecutionLifecycleStatus.CANCELLED
        handle.last_error = reason or "Execution was cancelled."
        self._emit_event("execution.cancelled", handle)
        self.unregister(execution_id)
        return True

    async def resume(self, execution_id: str, **kwargs: Any) -> dict[str, Any] | None:
        handle = self.get(execution_id)
        if handle is None:
            return None
        if handle.status == ExecutionLifecycleStatus.CANCELLED:
            return None
        result = await handle.runner.resume(execution_id, **kwargs)
        self._apply_result(handle, result)
        if not handle.is_resumable:
            self.unregister(execution_id)
        return result

    async def inject_user_message(
        self,
        execution_id: str,
        message: str | None = None,
        *,
        execution_message: str | None = None,
        display_message: str | None = None,
        files: list[dict[str, Any]] | None = None,
        turn_id: str | None = None,
        request_interrupt: bool = True,
        reason: str | None = None,
    ) -> ExecutionContext | None:
        handle = self.get(execution_id)
        if handle is None:
            return None
        handle.updated_at = _utcnow()
        return await handle.runner.inject_user_message(
            execution_id,
            message,
            execution_message=execution_message,
            display_message=display_message,
            files=files,
            turn_id=turn_id,
            request_interrupt=request_interrupt,
            reason=reason,
        )

    async def post_user_message(
        self,
        execution_id: str,
        message: str | None = None,
        *,
        execution_message: str | None = None,
        display_message: str | None = None,
        files: list[dict[str, Any]] | None = None,
        turn_id: str | None = None,
        request_interrupt: bool = True,
        reason: str | None = None,
    ) -> ExecutionContext | None:
        context = await self.inject_user_message(
            execution_id,
            message,
            execution_message=execution_message,
            display_message=display_message,
            files=files,
            turn_id=turn_id,
            request_interrupt=request_interrupt,
            reason=reason,
        )
        handle = self.get(execution_id)
        if handle is not None and context is not None:
            resolved_execution_message = (
                execution_message if execution_message is not None else message
            )
            resolved_display_message = (
                display_message if display_message is not None else message
            )
            self._emit_event(
                "execution.message_posted",
                handle,
                {
                    "message": resolved_display_message,
                    "display_message": resolved_display_message,
                    "execution_message": resolved_execution_message,
                    "files": files,
                    "turn_id": turn_id,
                    "request_interrupt": request_interrupt,
                },
            )
        return context

    def _on_task_done(
        self,
        execution_id: str,
        task: asyncio.Task[dict[str, Any]],
    ) -> None:
        handle = self.get(execution_id)
        if handle is None:
            return

        handle.task = None
        handle.updated_at = _utcnow()
        if task.cancelled():
            handle.status = ExecutionLifecycleStatus.CANCELLED
            handle.last_error = "Execution task was cancelled."
            self._emit_event("execution.cancelled", handle)
            self.unregister(execution_id)
            return

        try:
            result = task.result()
        except Exception:  # noqa: BLE001
            handle.status = ExecutionLifecycleStatus.FAILED
            handle.last_error = "Execution task failed without a normalized result."
            self._emit_event("execution.failed", handle)
            self.unregister(execution_id)
            return

        self._apply_result(handle, result)
        if not handle.is_resumable:
            self.unregister(execution_id)

    def _make_task_done_callback(
        self,
        execution_id: str,
    ) -> Callable[[asyncio.Task[dict[str, Any]]], None]:
        def callback(task: asyncio.Task[dict[str, Any]]) -> None:
            self._on_task_done(execution_id, task)

        return callback

    def _apply_result(
        self,
        handle: ExecutionHandle,
        result: dict[str, Any],
    ) -> None:
        handle.last_result = result
        handle.updated_at = _utcnow()
        handle.last_error = None

        status = result.get("status")
        if status == "interrupted":
            handle.status = ExecutionLifecycleStatus.INTERRUPTED
            error = result.get("error")
            handle.last_error = str(error) if error is not None else None
            self._emit_event("execution.interrupted", handle)
            return
        if status == "waiting_for_user":
            handle.status = ExecutionLifecycleStatus.WAITING_FOR_USER
            self._emit_event("execution.waiting_for_user", handle)
            return
        if result.get("success"):
            handle.status = ExecutionLifecycleStatus.COMPLETED
            self._emit_event("execution.completed", handle)
            return

        handle.status = ExecutionLifecycleStatus.FAILED
        error = result.get("error")
        handle.last_error = str(error) if error is not None else None
        self._emit_event("execution.failed", handle)

    def _emit_event(
        self,
        event_type: str,
        handle: ExecutionHandle,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not self._subscribers:
            return

        payload = {
            "type": event_type,
            "execution_id": handle.execution_id,
            "timestamp": _utcnow().isoformat(),
            "handle": handle.to_dict(),
        }
        if extra:
            payload.update(extra)

        for callback in self._subscribers.values():
            try:
                result = callback(dict(payload))
            except Exception:
                logger.exception("Execution registry subscriber failed")
                continue
            if inspect.isawaitable(result):
                task = asyncio.ensure_future(result)
                task.add_done_callback(self._log_subscriber_error)

    def _log_subscriber_error(self, task: asyncio.Future[Any]) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("Execution registry subscriber failed")
