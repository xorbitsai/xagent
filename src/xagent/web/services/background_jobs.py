from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from ...config import (
    get_background_job_max_retries,
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
) -> BackgroundJob:
    resolved_job_type = (
        job_type.value if isinstance(job_type, BackgroundJobType) else job_type
    )

    if idempotency_key:
        existing = (
            db.query(BackgroundJob)
            .filter(BackgroundJob.idempotency_key == idempotency_key)
            .first()
        )
        if existing is not None:
            return existing

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
