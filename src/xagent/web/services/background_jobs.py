from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlsplit

from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from ...config import (
    get_background_job_max_retries,
    get_background_job_stale_seconds,
    get_celery_broker_url,
    get_celery_enabled,
)
from ..models.background_job import (
    BackgroundJob,
    BackgroundJobStatus,
    BackgroundJobType,
)

logger = logging.getLogger(__name__)

QUEUE_DEFAULT = "default"
QUEUE_KB = "kb"
QUEUE_TRIGGERS = "triggers"

NON_TERMINAL_JOB_STATUSES = frozenset(
    {
        BackgroundJobStatus.PENDING.value,
        BackgroundJobStatus.ENQUEUED.value,
        BackgroundJobStatus.RUNNING.value,
    }
)
TERMINAL_JOB_STATUSES = frozenset(
    {
        BackgroundJobStatus.SUCCEEDED.value,
        BackgroundJobStatus.FAILED.value,
        BackgroundJobStatus.CANCELLED.value,
    }
)


def _is_redis_broker_reachable(broker_url: str) -> bool:
    try:
        import redis  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("Redis Celery broker configured but redis package is missing")
        return False

    try:
        client = redis.Redis.from_url(
            broker_url,
            socket_connect_timeout=1,
            socket_timeout=1,
        )
        client.ping()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Celery Redis broker is unreachable: %s", exc)
        return False
    return True


def is_background_job_enqueue_available(*, check_worker: bool = False) -> bool:
    """Return whether a new durable job can be sent to Celery now."""
    if not get_celery_enabled():
        return False

    broker_url = get_celery_broker_url()
    if broker_url is None:
        return False

    broker_scheme = urlsplit(broker_url).scheme
    if broker_scheme in {"redis", "rediss"} and not _is_redis_broker_reachable(
        broker_url
    ):
        return False

    if not check_worker:
        return True

    try:
        from ..jobs.celery_app import celery_app

        if celery_app.conf.task_always_eager:
            return True
        return bool(celery_app.control.ping(timeout=0.5))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Celery worker health check failed: %s", exc)
        return False


def queue_for_job_type(job_type: str) -> str:
    if job_type.startswith("kb."):
        return QUEUE_KB
    if job_type.startswith("trigger."):
        return QUEUE_TRIGGERS
    return QUEUE_DEFAULT


def create_background_job(
    db: Session,
    *,
    user_id: int,
    job_type: str | BackgroundJobType,
    payload: dict[str, Any],
    queue: str | None = None,
    idempotency_key: str | None = None,
    max_attempts: int | None = None,
    reuse_terminal_idempotency_key: bool = True,
) -> BackgroundJob:
    resolved_job_type = (
        job_type.value if isinstance(job_type, BackgroundJobType) else job_type
    )

    if idempotency_key:
        existing_query = db.query(BackgroundJob).filter(
            BackgroundJob.idempotency_key == idempotency_key
        )
        if not reuse_terminal_idempotency_key:
            existing_query = existing_query.filter(
                BackgroundJob.status.in_(NON_TERMINAL_JOB_STATUSES)
            )
        existing = existing_query.first()
        if existing is not None:
            return existing
        if not reuse_terminal_idempotency_key:
            release_terminal_background_job_idempotency_key(db, idempotency_key)

    job = BackgroundJob(
        user_id=user_id,
        job_type=resolved_job_type,
        queue=queue or queue_for_job_type(resolved_job_type),
        status=BackgroundJobStatus.PENDING.value,
        payload=payload,
        progress={"message": "Queued", "completed": 0, "total": 1},
        idempotency_key=idempotency_key,
        max_attempts=max_attempts or get_background_job_max_retries(),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_non_terminal_background_job_by_idempotency_key(
    db: Session,
    idempotency_key: str,
) -> BackgroundJob | None:
    return (
        db.query(BackgroundJob)
        .filter(BackgroundJob.idempotency_key == idempotency_key)
        .filter(BackgroundJob.status.in_(NON_TERMINAL_JOB_STATUSES))
        .first()
    )


def release_terminal_background_job_idempotency_key(
    db: Session,
    idempotency_key: str,
) -> None:
    terminal_jobs = (
        db.query(BackgroundJob)
        .filter(BackgroundJob.idempotency_key == idempotency_key)
        .filter(BackgroundJob.status.in_(TERMINAL_JOB_STATUSES))
        .all()
    )
    if not terminal_jobs:
        return
    for job in terminal_jobs:
        setattr(job, "idempotency_key", None)
        db.add(job)
    db.commit()


def enqueue_background_job(db: Session, job: BackgroundJob) -> BackgroundJob:
    if not get_celery_enabled():
        logger.info("Background job %s created but Celery enqueue is disabled", job.id)
        return job
    if get_celery_broker_url() is None:
        raise RuntimeError(
            "Celery background jobs are enabled but no broker URL is set"
        )

    from ..jobs.tasks import execute_background_job

    setattr(job, "status", BackgroundJobStatus.ENQUEUED.value)
    db.add(job)
    db.commit()
    db.refresh(job)

    async_result = execute_background_job.apply_async(
        args=[job.id],
        queue=str(job.queue or QUEUE_DEFAULT),
    )
    db.refresh(job)
    setattr(job, "celery_task_id", async_result.id)
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_background_job(db: Session, job_id: str) -> BackgroundJob | None:
    return db.query(BackgroundJob).filter(BackgroundJob.id == job_id).first()


def list_background_jobs(
    db: Session,
    *,
    user_id: int,
    is_admin: bool,
    status: str | None = None,
    job_type: str | None = None,
    limit: int = 50,
) -> list[BackgroundJob]:
    query = db.query(BackgroundJob)
    if not is_admin:
        query = query.filter(BackgroundJob.user_id == user_id)
    if status:
        query = query.filter(BackgroundJob.status == status)
    if job_type:
        query = query.filter(BackgroundJob.job_type == job_type)
    return (
        query.order_by(BackgroundJob.created_at.desc())
        .limit(max(1, min(limit, 200)))
        .all()
    )


def mark_job_running(db: Session, job: BackgroundJob) -> BackgroundJob:
    setattr(job, "status", BackgroundJobStatus.RUNNING.value)
    setattr(job, "attempts", int(job.attempts or 0) + 1)
    setattr(job, "started_at", datetime.now(timezone.utc))
    setattr(job, "error_message", None)
    setattr(job, "progress", {"message": "Running", "completed": 0, "total": 1})
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def update_job_progress(
    db: Session,
    job: BackgroundJob,
    *,
    message: str,
    completed: int | None = None,
    total: int | None = None,
    extra: dict[str, Any] | None = None,
) -> BackgroundJob:
    progress = dict(job.progress or {})
    progress["message"] = message
    if completed is not None:
        progress["completed"] = completed
    if total is not None:
        progress["total"] = total
    if extra:
        progress.update(extra)
    setattr(job, "progress", progress)
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def requeue_stale_background_jobs(
    db: Session,
    *,
    stale_after_seconds: int | None = None,
    limit: int = 100,
) -> list[BackgroundJob]:
    """Requeue non-terminal jobs whose durable DB state is stale.

    Redis/Celery can lose in-flight delivery state during broker loss or worker
    crashes. The database row remains authoritative, so the scheduler can safely
    put old pending/enqueued/running jobs back on the broker.
    """
    stale_seconds = stale_after_seconds or get_background_job_stale_seconds()
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=stale_seconds)
    requeue_statuses = {
        BackgroundJobStatus.PENDING.value,
        BackgroundJobStatus.ENQUEUED.value,
        BackgroundJobStatus.RUNNING.value,
    }

    stale_jobs = (
        db.query(BackgroundJob)
        .filter(BackgroundJob.status.in_(requeue_statuses))
        .filter(
            or_(
                and_(
                    BackgroundJob.status == BackgroundJobStatus.RUNNING.value,
                    BackgroundJob.started_at.is_not(None),
                    BackgroundJob.started_at <= cutoff,
                ),
                and_(
                    BackgroundJob.status != BackgroundJobStatus.RUNNING.value,
                    BackgroundJob.updated_at.is_not(None),
                    BackgroundJob.updated_at <= cutoff,
                ),
                and_(
                    BackgroundJob.updated_at.is_(None),
                    BackgroundJob.created_at.is_not(None),
                    BackgroundJob.created_at <= cutoff,
                ),
            )
        )
        .order_by(BackgroundJob.created_at.asc())
        .limit(max(1, min(limit, 500)))
        .all()
    )

    requeued: list[BackgroundJob] = []
    for job in stale_jobs:
        logger.warning(
            "Requeueing stale background job %s type=%s status=%s",
            job.id,
            job.job_type,
            job.status,
        )
        setattr(job, "status", BackgroundJobStatus.PENDING.value)
        setattr(job, "celery_task_id", None)
        setattr(job, "started_at", None)
        setattr(job, "error_message", "Requeued stale background job")
        setattr(
            job,
            "progress",
            {"message": "Requeued stale background job", "completed": 0, "total": 1},
        )
        db.add(job)

    if not stale_jobs:
        return requeued

    db.commit()
    for job in stale_jobs:
        db.refresh(job)

    if not get_celery_enabled():
        return stale_jobs

    if get_celery_broker_url() is None:
        error_message = "Celery background jobs are enabled but no broker URL is set"
        for job in stale_jobs:
            setattr(
                job, "error_message", f"Failed to requeue stale job: {error_message}"
            )
            db.add(job)
        db.commit()
        for job in stale_jobs:
            db.refresh(job)
        return stale_jobs

    from ..jobs.tasks import execute_background_job

    for job in stale_jobs:
        setattr(job, "status", BackgroundJobStatus.ENQUEUED.value)
        db.add(job)
    db.commit()
    for job in stale_jobs:
        db.refresh(job)

    for job in stale_jobs:
        try:
            async_result = execute_background_job.apply_async(
                args=[job.id],
                queue=str(job.queue or QUEUE_DEFAULT),
            )
            setattr(job, "celery_task_id", async_result.id)
            requeued.append(job)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to requeue stale background job %s", job.id)
            setattr(job, "status", BackgroundJobStatus.PENDING.value)
            setattr(job, "error_message", f"Failed to requeue stale job: {exc}")
            requeued.append(job)
        db.add(job)

    db.commit()
    for job in requeued:
        db.refresh(job)

    return requeued


def mark_job_succeeded(
    db: Session,
    job: BackgroundJob,
    *,
    result: dict[str, Any] | None = None,
) -> BackgroundJob:
    setattr(job, "status", BackgroundJobStatus.SUCCEEDED.value)
    setattr(job, "result", result)
    setattr(job, "error_message", None)
    setattr(job, "finished_at", datetime.now(timezone.utc))
    setattr(job, "progress", {"message": "Completed", "completed": 1, "total": 1})
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def mark_job_failed(
    db: Session,
    job: BackgroundJob,
    *,
    error_message: str,
    result: dict[str, Any] | None = None,
) -> BackgroundJob:
    setattr(job, "status", BackgroundJobStatus.FAILED.value)
    setattr(job, "error_message", error_message)
    setattr(job, "result", result)
    setattr(job, "finished_at", datetime.now(timezone.utc))
    setattr(job, "progress", {"message": error_message, "completed": 0, "total": 1})
    db.add(job)
    db.commit()
    db.refresh(job)
    return job
