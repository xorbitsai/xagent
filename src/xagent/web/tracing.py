"""Web tracer factory helpers."""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from ..core.agent.checkpoint import READABLE_CHECKPOINT_TYPES
from ..core.agent.trace import (
    BaseTraceHandler,
    ConsoleTraceHandler,
    ExecutionEventPersistenceError,
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


class ExecutionEventTraceAdapter(DatabaseTraceHandler):
    """Strict fact writer plus the transitional checkpoint reader.

    Normal observer dispatch must never write the same event a second time.
    """

    def __init__(self, task_id: int, build_id: str | None = None) -> None:
        super().__init__(task_id, build_id=build_id)
        self.authoritative = True

    async def handle_event(self, event: CoreTraceEvent) -> None:
        pass

    async def commit_event(self, event: CoreTraceEvent) -> None:
        required = event.require_persisted
        event.require_persisted = True
        try:
            await asyncio.to_thread(self._sync_save_to_database, event)
        except Exception as exc:
            raise ExecutionEventPersistenceError(
                "Conversation event commit failed"
            ) from exc
        finally:
            event.require_persisted = required


def task_database_handler(
    task_id: int, build_id: str | None = None
) -> DatabaseTraceHandler:
    from .models.database import get_session_local
    from .services.task_execution_event_writer import uses_execution_events

    with get_session_local()() as db:
        canonical = uses_execution_events(db, task_id)
    if canonical:
        return ExecutionEventTraceAdapter(task_id, build_id=build_id)
    return DatabaseTraceHandler(task_id, build_id=build_id)


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

    database_handler = task_database_handler(task_id)
    tracer = create_agent_tracer(
        handlers=[
            ConsoleTraceHandler(),
            database_handler,
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

    if isinstance(database_handler, ExecutionEventTraceAdapter):
        tracer.event_writer = database_handler.commit_event
    return tracer


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
