from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ...config import get_celery_broker_url, get_celery_enabled
from ..auth_dependencies import get_current_user
from ..models.background_job import BackgroundJob
from ..models.database import get_db
from ..models.user import User
from ..schemas.background_job import BackgroundJobResponse
from ..services.background_jobs import (
    get_background_job,
    is_background_job_enqueue_available,
    list_background_jobs,
)

jobs_router = APIRouter(prefix="/api/jobs", tags=["jobs"])


def _authorize_job(job: BackgroundJob | None, user: User) -> BackgroundJob:
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if not bool(user.is_admin) and int(job.user_id) != int(user.id):
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@jobs_router.get("", response_model=list[BackgroundJobResponse])
def list_jobs(
    status: str | None = Query(None),
    job_type: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[BackgroundJob]:
    return list_background_jobs(
        db,
        user_id=int(user.id),
        is_admin=bool(user.is_admin),
        status=status,
        job_type=job_type,
        limit=limit,
    )


@jobs_router.get("/capabilities")
def get_job_capabilities(
    _user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Return how clients should submit background-capable work."""
    celery_enabled = get_celery_enabled()
    broker_configured = get_celery_broker_url() is not None
    broker_reachable = is_background_job_enqueue_available(check_worker=False)
    worker_available = (
        is_background_job_enqueue_available(check_worker=True)
        if broker_reachable
        else False
    )
    return {
        "kb_ingest_mode": "celery" if worker_available else "sync",
        "celery_enabled": celery_enabled,
        "broker_configured": broker_configured,
        "broker_reachable": broker_reachable,
        "worker_available": worker_available,
    }


@jobs_router.get("/{job_id}", response_model=BackgroundJobResponse)
def get_job(
    job_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> BackgroundJob:
    return _authorize_job(get_background_job(db, job_id), user)
