"""Web tracer factory helpers."""

from __future__ import annotations

from typing import Any, Optional

from ..core.agent.checkpoint import READABLE_CHECKPOINT_TYPES
from ..core.agent.trace import (
    BaseTraceHandler,
    ConsoleTraceHandler,
)
from ..core.agent.trace import TraceEvent as CoreTraceEvent
from ..core.agent.trace import (
    TraceHandler,
    Tracer,
)
from ..core.tracing import create_agent_tracer
from .api.trace_handlers import DatabaseTraceHandler
from .models.user import User


class EphemeralCheckpointTraceHandler(BaseTraceHandler):
    """In-memory checkpoint storage for websocket-scoped preview executions."""

    def __init__(self, store: dict[str, dict[str, Any]]) -> None:
        super().__init__()
        self.store = store

    async def _handle_system_event(self, event: CoreTraceEvent) -> None:
        data = event.data if isinstance(event.data, dict) else {}
        if data.get("checkpoint_type") not in READABLE_CHECKPOINT_TYPES:
            return

        raw_id = (
            data.get("root_execution_id") or data.get("execution_id") or event.task_id
        )
        if raw_id is None:
            return

        execution_id = str(raw_id)
        snapshot = data.get("snapshot")
        if not execution_id or not isinstance(snapshot, dict):
            return

        self.store[execution_id] = dict(snapshot)

    async def load_latest_checkpoint(
        self,
        execution_id: str,
    ) -> dict[str, Any] | None:
        snapshot = self.store.get(str(execution_id))
        return dict(snapshot) if isinstance(snapshot, dict) else None


def create_task_tracer(
    task_id: int,
    user: Optional[User] = None,
    user_id: Optional[int] = None,
) -> Tracer:
    """Build the standard tracer stack for persisted web task execution."""
    from .api.ws_trace_handlers import WebSocketTraceHandler

    resolved_user_id = user_id
    if user is not None and user.id is not None:
        resolved_user_id = int(user.id)

    return create_agent_tracer(
        handlers=[
            ConsoleTraceHandler(),
            DatabaseTraceHandler(task_id),
            WebSocketTraceHandler(task_id),
        ],
        task_id=str(task_id),
        user_id=resolved_user_id,
        trace_name=f"xagent-web-task-{task_id}",
        session_id=f"task:{task_id}",
        tags=["xagent", "web", "task"],
        metadata={
            "source": "xagent-web",
            "task_id": task_id,
            "is_preview": False,
        },
    )


def create_ephemeral_tracer(
    *,
    task_id: str,
    websocket_handler: TraceHandler,
    checkpoint_store: dict[str, dict[str, Any]] | None = None,
    user: Optional[User] = None,
    is_preview: bool = False,
) -> Tracer:
    """Build a tracer for websocket-only flows such as builder preview."""
    handlers: list[TraceHandler] = []
    if checkpoint_store is not None:
        handlers.append(EphemeralCheckpointTraceHandler(checkpoint_store))
    handlers.append(websocket_handler)

    return create_agent_tracer(
        handlers=handlers,
        task_id=task_id,
        user_id=int(user.id) if user and user.id is not None else None,
        trace_name=f"xagent-web-{task_id}",
        session_id=task_id,
        tags=["xagent", "web", "preview" if is_preview else "builder"],
        metadata={
            "source": "xagent-web",
            "task_id": task_id,
            "is_preview": is_preview,
        },
    )
