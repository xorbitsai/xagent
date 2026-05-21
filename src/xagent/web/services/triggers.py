from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from ..models.background_job import BackgroundJob, BackgroundJobType
from .background_jobs import create_background_job, enqueue_background_job


def enqueue_trigger_event_job(
    db: Session,
    *,
    user_id: int,
    source_type: str,
    event_type: str,
    event_payload: dict[str, Any],
    source_event_id: str | None = None,
) -> BackgroundJob:
    """Persist and enqueue a trigger event without running the agent in Celery."""
    idempotency_key = (
        f"trigger:{source_type}:{source_event_id}" if source_event_id else None
    )
    job = create_background_job(
        db,
        user_id=user_id,
        job_type=BackgroundJobType.TRIGGER_EVENT,
        payload={
            "user_id": user_id,
            "source_type": source_type,
            "event_type": event_type,
            "source_event_id": source_event_id,
            "event_payload": event_payload,
        },
        idempotency_key=idempotency_key,
    )
    return enqueue_background_job(db, job)
