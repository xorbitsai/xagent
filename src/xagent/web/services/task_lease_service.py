"""Task execution leases for multi-process agent runners."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import uuid
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, cast

from sqlalchemy import or_, update
from sqlalchemy.orm import Session

from ...config import (
    get_task_lease_heartbeat_seconds,
    get_task_lease_ttl_seconds,
)
from ...core.agent_v2.checkpoint import CHECKPOINT_TYPE
from ..models.database import get_db
from ..models.task import Task, TaskStatus, TraceEvent

logger = logging.getLogger(__name__)

_RUNNER_ID = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:12]}"


def _rowcount(result: Any) -> int:
    return int(getattr(result, "rowcount", 0) or 0)


@dataclass(frozen=True)
class TaskLease:
    task_id: int
    runner_id: str


def get_runner_id() -> str:
    """Return the current process runner id."""
    return _RUNNER_ID


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _expires_at(now: datetime | None = None) -> datetime:
    return (now or utc_now()) + timedelta(seconds=get_task_lease_ttl_seconds())


def has_agent_v2_checkpoint(db: Session, task_id: int) -> bool:
    """Return whether the task has a persisted agent_v2 checkpoint."""
    rows = (
        db.query(TraceEvent)
        .filter(
            TraceEvent.task_id == task_id,
            TraceEvent.event_type == "system_update_general",
        )
        .order_by(TraceEvent.id.desc())
        .limit(100)
        .all()
    )
    for row in rows:
        data: dict[str, Any] = (
            cast(dict[str, Any], row.data) if isinstance(row.data, dict) else {}
        )
        if data.get("checkpoint_type") == CHECKPOINT_TYPE:
            return True
    return False


def acquire_task_lease(
    db: Session,
    task_id: int,
    *,
    runner_id: str | None = None,
) -> TaskLease | None:
    """Acquire the task execution lease if no live runner owns it."""
    runner = runner_id or get_runner_id()
    now = utc_now()
    expires_at = _expires_at(now)
    stmt = (
        update(Task)
        .where(Task.id == task_id)
        .where(
            or_(
                Task.status != TaskStatus.RUNNING,
                Task.runner_id == runner,
                Task.runner_id.is_(None),
                Task.lease_expires_at.is_(None),
                Task.lease_expires_at < now,
            )
        )
        .values(
            status=TaskStatus.RUNNING,
            runner_id=runner,
            last_heartbeat_at=now,
            lease_expires_at=expires_at,
        )
    )
    result = db.execute(stmt.execution_options(synchronize_session=False))
    db.commit()
    if _rowcount(result) != 1:
        logger.info("Task %s lease acquisition denied for runner %s", task_id, runner)
        return None
    logger.info(
        "Task %s lease acquired by runner %s until %s",
        task_id,
        runner,
        expires_at.isoformat(),
    )
    return TaskLease(task_id=task_id, runner_id=runner)


def refresh_task_lease(db: Session, lease: TaskLease) -> bool:
    """Refresh a live task lease owned by this runner."""
    now = utc_now()
    expires_at = _expires_at(now)
    stmt = (
        update(Task)
        .where(Task.id == lease.task_id)
        .where(Task.runner_id == lease.runner_id)
        .where(Task.status == TaskStatus.RUNNING)
        .values(last_heartbeat_at=now, lease_expires_at=expires_at)
    )
    result = db.execute(stmt.execution_options(synchronize_session=False))
    db.commit()
    return _rowcount(result) == 1


def release_task_lease(
    db: Session,
    lease: TaskLease | None,
    *,
    status: TaskStatus,
) -> bool:
    """Release a task lease and set its final visible status."""
    if lease is None:
        return False
    stmt = (
        update(Task)
        .where(Task.id == lease.task_id)
        .where(Task.runner_id == lease.runner_id)
        .values(
            status=status,
            runner_id=None,
            lease_expires_at=None,
            last_heartbeat_at=utc_now(),
        )
    )
    result = db.execute(stmt.execution_options(synchronize_session=False))
    db.commit()
    return _rowcount(result) == 1


def release_current_runner_task_lease(
    db: Session,
    task_id: int,
    *,
    status: TaskStatus,
    runner_id: str | None = None,
) -> bool:
    """Release the current runner's lease for a task."""
    runner = runner_id or get_runner_id()
    stmt = (
        update(Task)
        .where(Task.id == task_id)
        .where(Task.runner_id == runner)
        .values(
            status=status,
            runner_id=None,
            lease_expires_at=None,
            last_heartbeat_at=utc_now(),
        )
    )
    result = db.execute(stmt.execution_options(synchronize_session=False))
    db.commit()
    return _rowcount(result) == 1


def mark_task_paused_if_stale(db: Session, task: Task) -> bool:
    """Convert a stale RUNNING task into a recoverable terminal state."""
    if task.status != TaskStatus.RUNNING:
        return False

    now = utc_now()
    lease_expires_at = task.lease_expires_at
    if lease_expires_at is not None and lease_expires_at.tzinfo is None:
        lease_expires_at = lease_expires_at.replace(tzinfo=timezone.utc)

    if lease_expires_at is not None and lease_expires_at >= now:
        return False

    setattr(
        task,
        "status",
        TaskStatus.PAUSED
        if has_agent_v2_checkpoint(db, int(task.id))
        else TaskStatus.FAILED,
    )
    setattr(task, "runner_id", None)
    setattr(task, "lease_expires_at", None)
    setattr(task, "last_heartbeat_at", now)
    db.commit()
    logger.info("Marked stale task %s as %s", task.id, task.status.value)
    return True


async def run_task_lease_heartbeat(
    lease: TaskLease,
    stop_event: asyncio.Event,
) -> None:
    """Keep a task lease alive until the execution finishes."""
    interval = get_task_lease_heartbeat_seconds()
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
            break
        except asyncio.TimeoutError:
            pass

        db_gen = get_db()
        db = next(db_gen)
        try:
            if not refresh_task_lease(db, lease):
                logger.warning(
                    "Task %s lease heartbeat lost for runner %s",
                    lease.task_id,
                    lease.runner_id,
                )
                return
        except Exception as e:
            logger.warning(
                "Task %s lease heartbeat failed for runner %s: %s",
                lease.task_id,
                lease.runner_id,
                e,
            )
        finally:
            db.close()


async def stop_task_lease_heartbeat(
    task: asyncio.Task[Any] | None,
    stop_event: asyncio.Event | None,
) -> None:
    if stop_event is not None:
        stop_event.set()
    if task is not None:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
