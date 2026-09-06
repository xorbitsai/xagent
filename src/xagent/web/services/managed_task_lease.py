"""Managed lease lifecycle for inline task transports."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

from sqlalchemy.orm import Session

from ..models.database import release_db_connection_if_clean
from ..models.task import Task, TaskStatus
from .assistant_history_safety import (
    ASSISTANT_RESPONSE_MESSAGE_TYPE,
    assistant_history_values_for_persistence,
)
from .client_error_messages import CLIENT_SAFE_TASK_FAILURE
from .db_runtime import (
    drain_async_task_cancellation_safe,
    run_db_io_cancellation_safe,
)
from .task_lease_service import (
    TaskLease,
    TaskLeaseHeartbeatOutcome,
    acquire_task_lease_cancellation_safe,
    acquire_task_lease_no_commit,
    release_task_lease_no_commit,
    run_task_lease_heartbeat,
    stop_task_lease_heartbeat,
)
from .workforce_runtime import sync_workforce_run_status

logger = logging.getLogger(__name__)


def finalize_managed_task_lease_result(
    db: Session,
    lease: TaskLease,
    *,
    status: TaskStatus,
    assistant_content: str | None = None,
    turn_id: str | None = None,
    interactions: list[dict[str, Any]] | None = None,
    message_type: str = ASSISTANT_RESPONSE_MESSAGE_TYPE,
    error_message: str | None = None,
    # Reserved for a future reader and unconsumed today: this function
    # never reads it, and resolve_publishable_clarification -- the reader
    # it is reserved for -- has no production caller anywhere yet. The
    # change that wires that resolver is what will read it. Never pass
    # this to a logger or an exception.
    # finalize_managed_task_lease_result_isolated
    # hands its call off to a worker thread, so this mapping is held by a
    # closure across that thread boundary for a while, and it can carry
    # large structures such as file_outputs with no truncation applied.
    execution_result: Mapping[str, Any] | None = None,
) -> bool:
    """Atomically persist one inline transport result under its exact lease."""

    if status == TaskStatus.RUNNING:
        raise ValueError("Cannot finalize a managed lease with RUNNING status")

    from .chat_history_service import persist_assistant_message_no_commit
    from .task_execution_event_writer import stage_result_fact_no_commit
    from .task_orchestrator import invalidate_task_cache_best_effort

    try:
        if not release_task_lease_no_commit(db, lease, status=status):
            db.rollback()
            return False

        db.expire_all()
        task = db.query(Task).filter(Task.id == lease.task_id).one()
        history_content, history_message_type = (
            assistant_history_values_for_persistence(
                content=assistant_content or "",
                message_type=message_type,
                is_failure=status == TaskStatus.FAILED,
            )
        )
        if status == TaskStatus.FAILED:
            diagnostic_error = (error_message or "").strip()
            setattr(
                task,
                "error_message",
                diagnostic_error or CLIENT_SAFE_TASK_FAILURE,
            )
        sync_workforce_run_status(db, task, status)
        if task.user_id is not None and (
            (assistant_content is not None and assistant_content.strip())
            or interactions
            or status == TaskStatus.FAILED
        ):
            persist_assistant_message_no_commit(
                db,
                task_id=lease.task_id,
                user_id=int(task.user_id),
                content=history_content,
                interactions=(interactions if status != TaskStatus.FAILED else None),
                message_type=history_message_type,
                turn_id=turn_id,
            )
        stage_result_fact_no_commit(
            db, task, dict(execution_result or {"error": error_message})
        )
        db.commit()
    except Exception:
        db.rollback()
        raise

    invalidate_task_cache_best_effort(lease.task_id)
    return True


def _finalize_managed_task_lease_result_sync(
    lease: TaskLease,
    *,
    status: TaskStatus,
    assistant_content: str | None = None,
    turn_id: str | None = None,
    interactions: list[dict[str, Any]] | None = None,
    message_type: str = ASSISTANT_RESPONSE_MESSAGE_TYPE,
    error_message: str | None = None,
    execution_result: Mapping[str, Any] | None = None,
) -> bool:
    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        return finalize_managed_task_lease_result(
            db,
            lease,
            status=status,
            assistant_content=assistant_content,
            turn_id=turn_id,
            interactions=interactions,
            message_type=message_type,
            error_message=error_message,
            execution_result=execution_result,
        )


async def finalize_managed_task_lease_result_isolated(
    lease: TaskLease,
    *,
    status: TaskStatus,
    assistant_content: str | None = None,
    turn_id: str | None = None,
    interactions: list[dict[str, Any]] | None = None,
    message_type: str = ASSISTANT_RESPONSE_MESSAGE_TYPE,
    error_message: str | None = None,
    execution_result: Mapping[str, Any] | None = None,
) -> bool:
    """Settle one exact managed lease using a worker-owned short Session."""

    return await run_db_io_cancellation_safe(
        lambda: _finalize_managed_task_lease_result_sync(
            lease,
            status=status,
            assistant_content=assistant_content,
            turn_id=turn_id,
            interactions=interactions,
            message_type=message_type,
            error_message=error_message,
            execution_result=execution_result,
        )
    )


def _release_managed_task_lease_sync(lease: TaskLease) -> bool:
    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = db.query(Task).filter(Task.id == lease.task_id).first()
        if task is None or task.run_id != lease.run_id:
            return False
        final_status = (
            TaskStatus.FAILED if task.status == TaskStatus.RUNNING else task.status
        )
        return finalize_managed_task_lease_result(
            db,
            lease,
            status=final_status,
        )


@dataclass
class ManagedTaskLease:
    """Keep a pre-acquired lease alive and release it exactly once."""

    lease: TaskLease
    stop_event: asyncio.Event
    heartbeat_task: asyncio.Task[TaskLeaseHeartbeatOutcome]
    _closed: bool = field(default=False, init=False)

    async def close(self) -> bool:
        if self._closed:
            return False
        self._closed = True
        cleanup_task = asyncio.create_task(self._close_resources())
        return await drain_async_task_cancellation_safe(cleanup_task)

    async def finalize_result(
        self,
        *,
        status: TaskStatus,
        assistant_content: str | None = None,
        turn_id: str | None = None,
        interactions: list[dict[str, Any]] | None = None,
        message_type: str = ASSISTANT_RESPONSE_MESSAGE_TYPE,
        error_message: str | None = None,
        execution_result: Mapping[str, Any] | None = None,
    ) -> bool:
        """Stop heartbeating, then atomically persist this owner's result."""

        if self._closed:
            return False
        self._closed = True
        cleanup_task = asyncio.create_task(
            self._finalize_resources(
                status=status,
                assistant_content=assistant_content,
                turn_id=turn_id,
                interactions=interactions,
                message_type=message_type,
                error_message=error_message,
                execution_result=execution_result,
            )
        )
        return await drain_async_task_cancellation_safe(cleanup_task)

    async def _stop_heartbeat_for_settlement(self) -> bool:
        heartbeat_outcome = await stop_task_lease_heartbeat(
            self.heartbeat_task, self.stop_event
        )
        if not heartbeat_outcome.requires_ttl_recovery:
            return True
        logger.error(
            "Task %s managed lease heartbeat unhealthy at shutdown; "
            "retaining run %s for TTL recovery (lost=%s, pool_timeout=%s)",
            self.lease.task_id,
            self.lease.run_id,
            heartbeat_outcome.lease_lost,
            heartbeat_outcome.pool_timeout is not None,
        )
        return False

    async def _finalize_resources(
        self,
        *,
        status: TaskStatus,
        assistant_content: str | None,
        turn_id: str | None,
        interactions: list[dict[str, Any]] | None,
        message_type: str,
        error_message: str | None,
        execution_result: Mapping[str, Any] | None = None,
    ) -> bool:
        if not await self._stop_heartbeat_for_settlement():
            return False
        return await finalize_managed_task_lease_result_isolated(
            self.lease,
            status=status,
            assistant_content=assistant_content,
            turn_id=turn_id,
            interactions=interactions,
            message_type=message_type,
            error_message=error_message,
            execution_result=execution_result,
        )

    async def _close_resources(self) -> bool:
        if not await self._stop_heartbeat_for_settlement():
            return False
        try:
            return await run_db_io_cancellation_safe(
                lambda: _release_managed_task_lease_sync(self.lease)
            )
        except Exception:
            logger.error(
                "Failed to release managed task lease for task %s run %s",
                self.lease.task_id,
                self.lease.run_id,
                exc_info=True,
            )
            return False


def start_managed_task_lease(lease: TaskLease) -> ManagedTaskLease:
    """Start heartbeating a lease that the caller already claimed."""

    stop_event = asyncio.Event()
    heartbeat_task = asyncio.create_task(run_task_lease_heartbeat(lease, stop_event))
    return ManagedTaskLease(
        lease=lease,
        stop_event=stop_event,
        heartbeat_task=heartbeat_task,
    )


def claim_managed_task_lease(
    db: Session,
    task_id: int,
) -> ManagedTaskLease | None:
    """Atomically claim a new run, project RUNNING, and start its heartbeat."""

    lease = _claim_managed_task_lease_in_session(db, task_id)
    return start_managed_task_lease(lease) if lease is not None else None


def _claim_managed_task_lease_in_session(
    db: Session,
    task_id: int,
) -> TaskLease | None:
    """Commit one managed claim in the Session owned by the current caller."""

    lease = acquire_task_lease_no_commit(db, task_id, new_run=True)
    if lease is None:
        db.rollback()
        return None
    try:
        db.expire_all()
        task = db.query(Task).filter(Task.id == task_id).one()
        sync_workforce_run_status(db, task, TaskStatus.RUNNING)
        db.commit()
    except Exception:
        db.rollback()
        raise
    return lease


def _claim_managed_task_lease_sync(task_id: int) -> TaskLease | None:
    """Claim using a short Session owned entirely by one database worker."""

    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        return _claim_managed_task_lease_in_session(db, task_id)


async def claim_managed_task_lease_isolated(
    task_id: int,
    *,
    caller_db: Session | None = None,
) -> ManagedTaskLease | None:
    """Claim off-loop and compensate a commit that wins caller cancellation."""

    if caller_db is not None and not release_db_connection_if_clean(caller_db):
        raise RuntimeError(
            "Cannot claim a managed task lease while the caller database "
            "session has pending writes"
        )

    lease = await acquire_task_lease_cancellation_safe(
        lambda: _claim_managed_task_lease_sync(task_id),
        _release_managed_task_lease_sync,
    )
    return start_managed_task_lease(lease) if lease is not None else None
