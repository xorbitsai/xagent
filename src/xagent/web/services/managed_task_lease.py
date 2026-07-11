"""Managed lease lifecycle for inline task transports."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from ..models.task import Task, TaskStatus
from .task_lease_service import (
    TaskLease,
    acquire_task_lease,
    run_task_lease_heartbeat,
    stop_task_lease_heartbeat,
)
from .workforce_runtime import release_task_lease_with_workforce_sync

logger = logging.getLogger(__name__)


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
        return release_task_lease_with_workforce_sync(
            db,
            lease,
            status=final_status,
        )


@dataclass
class ManagedTaskLease:
    """Keep a pre-acquired lease alive and release it exactly once."""

    lease: TaskLease
    stop_event: asyncio.Event
    heartbeat_task: asyncio.Task[None]
    _closed: bool = field(default=False, init=False)

    async def close(self) -> bool:
        if self._closed:
            return False
        self._closed = True
        await stop_task_lease_heartbeat(self.heartbeat_task, self.stop_event)
        try:
            return await asyncio.to_thread(_release_managed_task_lease_sync, self.lease)
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
    """Atomically claim a new run and start its lease heartbeat."""

    lease = acquire_task_lease(db, task_id, new_run=True)
    return start_managed_task_lease(lease) if lease is not None else None
