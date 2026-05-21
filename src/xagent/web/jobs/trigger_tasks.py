from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from ..models.background_job import BackgroundJob
from ..services.background_jobs import update_job_progress
from .celery_app import celery_app

logger = logging.getLogger(__name__)


def handle_trigger_event(db: Session, job: BackgroundJob) -> dict[str, Any]:
    """Persisted trigger-event processing hook.

    This intentionally stops before agent execution. The next layer can create
    ready trigger runs or call the existing web/task scheduler from the FastAPI
    process without moving the agent runner into Celery.
    """
    payload = dict(job.payload or {})
    update_job_progress(db, job, message="Processing trigger event")
    logger.info(
        "Processed trigger event job=%s source=%s event=%s",
        job.id,
        payload.get("source_type"),
        payload.get("event_type"),
    )
    return {
        "status": "accepted",
        "source_type": payload.get("source_type"),
        "event_type": payload.get("event_type"),
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }


def handle_trigger_scan(db: Session, job: BackgroundJob) -> dict[str, Any]:
    payload = dict(job.payload or {})
    update_job_progress(db, job, message="Scanning scheduled triggers")
    return {
        "status": "scanned",
        "scan_scope": payload.get("scope", "all"),
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }


@celery_app.task(name="xagent.web.jobs.trigger_tasks.scan_due_triggers")
def scan_due_triggers() -> dict[str, Any]:
    """Celery Beat entrypoint placeholder for scheduled trigger scans.

    Full trigger definitions and agent handoff are kept outside Celery. This task
    exists so deployments can run beat/worker now without pulling agent execution
    into the worker process.
    """
    logger.info("Scheduled trigger scan tick")
    return {"status": "ok"}
