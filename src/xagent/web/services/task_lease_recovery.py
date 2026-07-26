"""Automatic, fenced recovery for expired task execution leases."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from ..models.task import Task, TaskStatus
from .db_runtime import is_database_pool_timeout, run_db_io_cancellation_safe
from .task_lease_service import (
    TaskLeaseRecoveryCandidate,
    get_expired_task_lease_candidates,
    get_next_expired_task_lease_candidate_for_update,
    has_recoverable_agent_checkpoint,
    recover_expired_task_lease_no_commit,
    utc_now,
)
from .task_orchestrator import (
    invalidate_task_cache_best_effort,
    sync_trigger_run_status,
)
from .workforce_runtime import sync_workforce_run_status

logger = logging.getLogger(__name__)

TASK_LEASE_EXPIRED_ERROR = (
    "Task execution lease expired before the runner completed the task."
)
TASK_LEASE_PAUSED_TRIGGER_ERROR = (
    "Task paused after execution lease expiry; manual resume is required."
)

TaskLeaseRecoveryCursor = tuple[datetime, int]


@dataclass(frozen=True)
class TaskLeaseRecoveryBatch:
    """Result of scanning and attempting one bounded candidate page."""

    scanned: int
    recovered: int
    next_cursor: TaskLeaseRecoveryCursor | None


def recover_task_lease_candidate_no_commit(
    db: Session,
    candidate: TaskLeaseRecoveryCandidate,
    *,
    recovered_at: datetime,
) -> TaskStatus | None:
    """Stage one recovery and all lifecycle projections without committing."""

    next_status = (
        TaskStatus.PAUSED
        if has_recoverable_agent_checkpoint(db, candidate)
        else TaskStatus.FAILED
    )
    task_error = None if next_status == TaskStatus.PAUSED else TASK_LEASE_EXPIRED_ERROR
    recovered = recover_expired_task_lease_no_commit(
        db,
        candidate,
        status=next_status,
        recovered_at=recovered_at,
        error_message=task_error,
    )
    if not recovered:
        return None

    db.expire_all()
    task = db.query(Task).filter(Task.id == candidate.task_id).one()
    sync_workforce_run_status(db, task, next_status)
    sync_trigger_run_status(
        db,
        task,
        next_status,
        error_message=(
            TASK_LEASE_PAUSED_TRIGGER_ERROR
            if next_status == TaskStatus.PAUSED
            else TASK_LEASE_EXPIRED_ERROR
        ),
    )
    return next_status


def recover_task_lease_candidate_isolated(
    candidate: TaskLeaseRecoveryCandidate,
    *,
    recovered_at: datetime,
) -> TaskStatus | None:
    """Own the short recovery Session, transaction, and post-commit cache write."""

    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        try:
            recovered = recover_task_lease_candidate_no_commit(
                db,
                candidate,
                recovered_at=recovered_at,
            )
            if recovered is None:
                db.rollback()
                return None
            db.commit()
        except Exception:
            db.rollback()
            raise
    _record_recovered_task_lease(candidate, recovered)
    return recovered


def _record_recovered_task_lease(
    candidate: TaskLeaseRecoveryCandidate,
    recovered: TaskStatus,
) -> None:
    """Publish post-commit effects for one successfully recovered lease."""

    invalidate_task_cache_best_effort(candidate.task_id)
    logger.warning(
        "Recovered expired task lease: task=%s status=%s runner=%s run=%s",
        candidate.task_id,
        recovered.value,
        candidate.runner_id,
        candidate.run_id,
    )


def _recover_expired_task_leases_batch_with_cas(
    *,
    cutoff: datetime,
    batch_size: int,
    after: TaskLeaseRecoveryCursor | None,
) -> TaskLeaseRecoveryBatch:
    """Scan one bounded page and recover every candidate independently."""

    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    with SessionLocal() as scan_db:
        candidates = get_expired_task_lease_candidates(
            scan_db,
            cutoff=cutoff,
            limit=batch_size,
            after=after,
        )

    recovered = 0
    for candidate in candidates:
        try:
            if (
                recover_task_lease_candidate_isolated(
                    candidate,
                    recovered_at=utc_now(),
                )
                is not None
            ):
                recovered += 1
        except Exception as exc:
            if is_database_pool_timeout(exc):
                raise
            logger.exception(
                "Task lease recovery failed for task %s; retrying next tick",
                candidate.task_id,
            )

    return TaskLeaseRecoveryBatch(
        scanned=len(candidates),
        recovered=recovered,
        next_cursor=candidates[-1].cursor if candidates else None,
    )


def _use_postgresql_recovery_partitioning() -> bool:
    """Return whether candidate row locking can partition recovery workers."""

    from ..models.database import get_engine

    return get_engine().dialect.name == "postgresql"


def _recover_expired_task_leases_batch_with_postgresql_locks(
    *,
    cutoff: datetime,
    batch_size: int,
    after: TaskLeaseRecoveryCursor | None,
) -> TaskLeaseRecoveryBatch:
    """Recover one page using a short locked transaction per candidate."""

    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    scanned = 0
    recovered_count = 0
    cursor = after
    while scanned < batch_size:
        candidate: TaskLeaseRecoveryCandidate | None = None
        recovered: TaskStatus | None = None
        with SessionLocal() as db:
            try:
                candidate = get_next_expired_task_lease_candidate_for_update(
                    db,
                    cutoff=cutoff,
                    after=cursor,
                )
            except Exception:
                db.rollback()
                raise

            if candidate is None:
                db.rollback()
                break

            try:
                recovered = recover_task_lease_candidate_no_commit(
                    db,
                    candidate,
                    recovered_at=utc_now(),
                )
                if recovered is None:
                    db.rollback()
                else:
                    db.commit()
            except Exception as exc:
                db.rollback()
                if is_database_pool_timeout(exc):
                    raise
                recovered = None
                logger.exception(
                    "Task lease recovery failed for task %s; retrying next tick",
                    candidate.task_id,
                )

        scanned += 1
        cursor = candidate.cursor
        if recovered is not None:
            recovered_count += 1
            _record_recovered_task_lease(candidate, recovered)

    return TaskLeaseRecoveryBatch(
        scanned=scanned,
        recovered=recovered_count,
        next_cursor=cursor if scanned else None,
    )


def recover_expired_task_leases_batch_isolated(
    *,
    cutoff: datetime,
    batch_size: int,
    after: TaskLeaseRecoveryCursor | None,
) -> TaskLeaseRecoveryBatch:
    """Recover one bounded page with backend-appropriate worker partitioning."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if _use_postgresql_recovery_partitioning():
        return _recover_expired_task_leases_batch_with_postgresql_locks(
            cutoff=cutoff,
            batch_size=batch_size,
            after=after,
        )
    return _recover_expired_task_leases_batch_with_cas(
        cutoff=cutoff,
        batch_size=batch_size,
        after=after,
    )


async def recover_expired_task_leases_until_cutoff(
    *,
    cutoff: datetime,
    batch_size: int,
) -> int:
    """Drain all pages older than one fixed cutoff without starving the loop."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    recovered = 0
    cursor: TaskLeaseRecoveryCursor | None = None
    while True:
        batch = await run_db_io_cancellation_safe(
            lambda: recover_expired_task_leases_batch_isolated(
                cutoff=cutoff,
                batch_size=batch_size,
                after=cursor,
            )
        )
        recovered += batch.recovered
        if batch.scanned < batch_size:
            return recovered
        cursor = batch.next_cursor
        await asyncio.sleep(0)


async def run_task_lease_recovery_loop(
    *,
    poll_interval_seconds: int,
    batch_size: int,
) -> None:
    """Continuously recover expired leases while the backend process is alive."""

    while True:
        try:
            recovered = await recover_expired_task_leases_until_cutoff(
                cutoff=datetime.now(timezone.utc),
                batch_size=batch_size,
            )
            if recovered:
                logger.info("Recovered %s expired task lease(s)", recovered)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if is_database_pool_timeout(exc):
                logger.warning(
                    "Task lease recovery skipped after database pool timeout"
                )
            else:
                logger.exception("Task lease recovery tick failed")

        await asyncio.sleep(poll_interval_seconds)
