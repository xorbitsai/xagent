"""Web tracer factory helpers."""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..core.agent.checkpoint import (
    READABLE_CHECKPOINT_TYPES,
    ExecutionEventPersistenceError,
)
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
from .models.user import User
from .services.db_write_retry import retry_db_write_async
from .services.execution_event_signals import (
    report_commit_failure,
    report_recovery_refusal,
)
from .services.task_lease_service import current_task_lease
from .services.trace_handlers import CommitState, DatabaseTraceHandler

logger = logging.getLogger(__name__)


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
    """Strict fact writer and event-backed recovery reader.

    Normal observer dispatch must never write the same event a second time.
    """

    def __init__(self, task_id: int, build_id: str | None = None) -> None:
        super().__init__(task_id, build_id=build_id)
        self.authoritative = True

    def _sync_load_latest_checkpoint(self, execution_id: str) -> Any:
        from sqlalchemy.exc import SQLAlchemyError

        from ..core.agent.checkpoint import CheckpointUnavailableError
        from .services.ops_signals import (
            CHECKPOINT_LOAD_UNAVAILABLE,
            clear_degradation,
            register_degradation,
        )

        try:
            result = super()._sync_load_latest_checkpoint(execution_id)
        except SQLAlchemyError as exc:
            register_degradation(
                CHECKPOINT_LOAD_UNAVAILABLE,
                f"task {self.task_id}: execution-event checkpoint read failed",
            )
            lease = current_task_lease()
            report_recovery_refusal(
                "checkpoint",
                "unavailable",
                task_id=self.task_id,
                run_id=lease.run_id if lease is not None else None,
            )
            raise CheckpointUnavailableError(
                "Execution-event checkpoint read could not complete"
            ) from exc
        clear_degradation(CHECKPOINT_LOAD_UNAVAILABLE)
        return result

    def _task_has_run_tagged_checkpoint(self, db: Any) -> bool:
        from sqlalchemy import select

        from ..core.agent.checkpoint import CheckpointUnavailableError
        from .models.task_execution_event import TaskExecutionEvent
        from .services.ops_signals import (
            CHECKPOINT_LOAD_UNAVAILABLE,
            register_degradation,
        )

        # Mirrors the base probe's failure handling: a DB error must surface
        # as an incomplete read, never as "no tagged row" and never as a raw
        # driver exception that the caller would report a second time.
        try:
            return (
                db.scalar(
                    select(TaskExecutionEvent.id)
                    .where(
                        TaskExecutionEvent.task_id == self.task_id,
                        TaskExecutionEvent.scope_id == "root",
                        TaskExecutionEvent.kind == "recovery_state",
                        TaskExecutionEvent.run_id.is_not(None),
                    )
                    .limit(1)
                )
                is not None
            )
        except Exception as exc:
            register_degradation(
                CHECKPOINT_LOAD_UNAVAILABLE,
                f"task {self.task_id}: checkpoint partition probe failed",
            )
            # Type only: a driver message or statement parameters can carry
            # task content.
            logger.error(
                "task %s: run-tagged checkpoint probe failed: %s",
                self.task_id,
                type(exc).__name__,
            )
            raise CheckpointUnavailableError(
                f"task {self.task_id}: could not determine whether a "
                "run-tagged checkpoint exists"
            ) from exc

    def _sync_load_latest_checkpoint_unguarded(
        self, db: Any, execution_id: str, partition: Any
    ) -> Any:
        from .services.task_execution_event_recovery import (
            check_recovery_owner,
            read_event_checkpoint,
        )

        check_recovery_owner(db, self.task_id)
        result = read_event_checkpoint(
            db,
            task_id=self.task_id,
            scope_id=self.build_id or "root",
            execution_id=execution_id,
            run_id=partition.run_id if partition else None,
            filter_run=partition is not None,
        )
        check_recovery_owner(db, self.task_id)
        return result["snapshot"] if result is not None else None

    async def load_committed_tool_outcome(
        self, tool_call: dict[str, Any]
    ) -> dict[str, Any] | None:
        import asyncio

        from .models.database import get_session_local
        from .services.task_execution_event_recovery import (
            check_recovery_owner,
            read_committed_tool_outcome,
        )

        # The lease context is a contextvar; read it here, not in the worker
        # thread that runs ``load``.
        lease = current_task_lease()

        def load() -> dict[str, Any] | None:
            from sqlalchemy.exc import SQLAlchemyError

            from ..core.agent.checkpoint import CheckpointUnavailableError

            try:
                with get_session_local()() as db:
                    check_recovery_owner(db, self.task_id)
                    result = read_committed_tool_outcome(
                        db,
                        task_id=self.task_id,
                        scope_id=self.build_id or "root",
                        tool_call=tool_call,
                        run_id=lease.run_id if lease is not None else None,
                    )
                    check_recovery_owner(db, self.task_id)
                    return result
            except SQLAlchemyError as exc:
                attempt_id = tool_call.get("tool_attempt_id")
                report_recovery_refusal(
                    "tool_outcome",
                    "unavailable",
                    task_id=self.task_id,
                    run_id=lease.run_id if lease is not None else None,
                    tool_attempt_id=str(attempt_id) if attempt_id else None,
                )
                raise CheckpointUnavailableError(
                    "Committed tool outcome read could not complete"
                ) from exc

        return await asyncio.to_thread(load)

    async def handle_event(self, event: CoreTraceEvent) -> None:
        pass

    async def commit_event(self, event: CoreTraceEvent) -> None:
        required = event.require_persisted
        event.require_persisted = True
        # A retryable failure may have committed. Every attempt reruns the
        # whole transaction for this same event, so its fact identity is
        # unchanged and a committed first attempt replays instead of
        # appending again. Backoff runs outside the trace runtime's slot.
        commit_state = CommitState()

        async def attempt() -> None:
            commit_state.attempt += 1
            await self._save_to_database(event, commit_state=commit_state)

        try:
            await retry_db_write_async(attempt, site="trace_fact")
        except Exception as exc:
            # Retryable attempts were only warned about; report the write once.
            lease = current_task_lease()
            data = event.data if isinstance(event.data, dict) else {}
            attempt_id = data.get("tool_attempt_id")
            report_commit_failure(
                "trace_fact",
                exc,
                task_id=self.task_id,
                run_id=lease.run_id if lease is not None else None,
                retried=commit_state.attempt > 1,
                tool_attempt_id=str(attempt_id) if attempt_id else None,
            )
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
    from .services.task_event_trace_handler import TaskEventTraceHandler

    resolved_user_id = user_id
    if user is not None and user.id is not None:
        resolved_user_id = int(user.id)

    database_handler = task_database_handler(task_id)
    tracer = create_agent_tracer(
        handlers=[
            ConsoleTraceHandler(),
            database_handler,
            TaskEventTraceHandler(
                task_id,
                authoritative=isinstance(database_handler, ExecutionEventTraceAdapter),
            ),
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
