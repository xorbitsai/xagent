from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from ..models.background_job import BackgroundJob
from ..models.database import get_session_local, init_db
from ..services.background_jobs import (
    requeue_stale_background_jobs,
    update_job_progress,
)
from ..services.triggers import prepare_trigger_run, scan_due_scheduled_triggers
from ..services.workforce_runtime import (
    pause_workforce_tasks_after_archive,
    reap_stale_preview_workforce_runs,
)
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
    trigger_id = payload.get("trigger_id")
    if trigger_id:
        from ..models.trigger import AgentTrigger

        trigger = (
            db.query(AgentTrigger).filter(AgentTrigger.id == int(trigger_id)).first()
        )
        if trigger is None:
            raise ValueError(f"Trigger not found: {trigger_id}")
        run, created = prepare_trigger_run(
            db,
            trigger=trigger,
            event_payload=dict(payload.get("event_payload") or {}),
            source_event_id=payload.get("source_event_id"),
            background_job_id=str(job.id),
        )
        return {
            "status": "prepared" if created else "duplicate",
            "trigger_id": int(trigger.id),
            "trigger_run_id": int(run.id),
            "task_id": int(run.task_id) if run.task_id is not None else None,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }

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
    requeued_jobs = requeue_stale_background_jobs(db)
    runs = scan_due_scheduled_triggers(db)
    return {
        "status": "scanned",
        "scan_scope": payload.get("scope", "all"),
        "requeued_stale_jobs": len(requeued_jobs),
        "trigger_runs_created": len(runs),
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }


@celery_app.task(name="xagent.web.jobs.trigger_tasks.scan_due_triggers")
def scan_due_triggers() -> dict[str, Any]:
    """Celery Beat entrypoint for scheduled trigger scans and job recovery.

    Full trigger definitions and agent handoff are kept outside Celery. This task
    also requeues stale DB-backed jobs after broker loss or worker crashes, and
    reaps abandoned workforce-builder preview runs (see
    ``reap_stale_preview_workforce_runs``).
    """
    logger.info("Scheduled trigger scan tick")
    try:
        SessionLocal = get_session_local()
    except RuntimeError:
        init_db()
        SessionLocal = get_session_local()

    db = SessionLocal()
    try:
        requeued_jobs = requeue_stale_background_jobs(db)
        runs = scan_due_scheduled_triggers(db)
        reaped_pause_targets = reap_stale_preview_workforce_runs(db)
        if reaped_pause_targets:
            asyncio.run(
                pause_workforce_tasks_after_archive(
                    reaped_pause_targets,
                    # Preview runs have no workforce_id; only used for the
                    # log message on a failed dispatch.
                    workforce_id=0,
                    actor_user_id=None,
                    reason="preview-reap",
                )
            )
        return {
            "status": "ok",
            "requeued_stale_jobs": len(requeued_jobs),
            "trigger_runs_created": len(runs),
            # Only counts reaped runs whose Task was still RUNNING (i.e. that
            # needed an explicit PAUSE dispatch), not every reaped run.
            "reaped_preview_run_pause_dispatches": len(reaped_pause_targets),
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }
    finally:
        db.close()
