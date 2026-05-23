from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session

from ..models.background_job import (
    BackgroundJob,
    BackgroundJobStatus,
    BackgroundJobType,
)
from ..models.database import get_session_local, init_db
from ..services.background_jobs import (
    mark_job_failed,
    mark_job_running,
    mark_job_succeeded,
)
from .celery_app import celery_app
from .exceptions import BackgroundJobHandlerError

logger = logging.getLogger(__name__)


def _open_worker_session() -> Session:
    try:
        SessionLocal = get_session_local()
    except RuntimeError:
        init_db()
        SessionLocal = get_session_local()
    return SessionLocal()


def _execute_job_handler(db: Session, job: BackgroundJob) -> dict[str, Any]:
    if job.job_type == BackgroundJobType.KB_INGEST_DOCUMENT.value:
        from .kb_tasks import handle_kb_ingest_document

        return handle_kb_ingest_document(db, job)
    if job.job_type == BackgroundJobType.KB_INGEST_WEB.value:
        from .kb_tasks import handle_kb_ingest_web

        return handle_kb_ingest_web(db, job)
    if job.job_type == BackgroundJobType.TRIGGER_EVENT.value:
        from .trigger_tasks import handle_trigger_event

        return handle_trigger_event(db, job)
    if job.job_type == BackgroundJobType.TRIGGER_SCAN.value:
        from .trigger_tasks import handle_trigger_scan

        return handle_trigger_scan(db, job)

    raise ValueError(f"Unsupported background job type: {job.job_type}")


@celery_app.task(
    bind=True,
    name="xagent.web.jobs.tasks.execute_background_job",
    retry_backoff=True,
    retry_jitter=True,
)
def execute_background_job(self: Any, job_id: str) -> dict[str, Any]:
    db = _open_worker_session()
    try:
        job = db.query(BackgroundJob).filter(BackgroundJob.id == job_id).first()
        if job is None:
            raise ValueError(f"Background job not found: {job_id}")
        if job.status == BackgroundJobStatus.CANCELLED.value:
            logger.info("Skipping cancelled background job %s", job_id)
            return {"status": "cancelled"}
        if job.status == BackgroundJobStatus.SUCCEEDED.value:
            logger.info("Skipping already completed background job %s", job_id)
            return dict(job.result or {"status": "succeeded"})

        mark_job_running(db, job)
        try:
            result = _execute_job_handler(db, job)
        except BackgroundJobHandlerError as exc:
            db.refresh(job)
            if exc.retryable and int(job.attempts or 0) < int(job.max_attempts or 1):
                setattr(job, "status", BackgroundJobStatus.ENQUEUED.value)
                setattr(job, "error_message", str(exc))
                setattr(job, "result", exc.result)
                db.add(job)
                db.commit()
                raise self.retry(exc=exc, max_retries=int(job.max_attempts or 1))
            mark_job_failed(db, job, error_message=str(exc), result=exc.result)
            raise
        except Exception as exc:  # noqa: BLE001
            db.refresh(job)
            if int(job.attempts or 0) < int(job.max_attempts or 1):
                setattr(job, "status", BackgroundJobStatus.ENQUEUED.value)
                setattr(job, "error_message", str(exc))
                db.add(job)
                db.commit()
                raise self.retry(exc=exc, max_retries=int(job.max_attempts or 1))
            mark_job_failed(db, job, error_message=str(exc))
            raise

        db.refresh(job)
        mark_job_succeeded(db, job, result=result)
        return result
    finally:
        db.close()
