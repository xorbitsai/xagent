"""Admin diagnostic endpoint for one task's V2 execution-event health.

Read-only and admin-authenticated. The response never carries event payloads
or other free text; see ``services.execution_event_diagnostics``.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ..auth_dependencies import get_current_user
from ..models.database import get_db
from ..models.user import User
from ..services.execution_event_diagnostics import task_execution_event_diagnostics

router = APIRouter(
    prefix="/api/admin/execution-events", tags=["admin-execution-events"]
)
logger = logging.getLogger(__name__)


@router.get("/tasks/{task_id}")
def get_task_execution_event_diagnostics(
    task_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Event counts, runs, open tool attempts and recovery state for a task.

    A plain ``def`` so FastAPI runs it in the threadpool: the session is
    synchronous. A database failure returns 503, never zeroed sections --
    an empty event log and an unreadable one must stay distinguishable. The
    log line carries the task id and exception class only, since a
    SQLAlchemy message can embed statement parameters.
    """
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        diagnostics = task_execution_event_diagnostics(db, task_id)
    except SQLAlchemyError as exc:
        logger.warning(
            "Execution event diagnostics failed for task %s: %s",
            task_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=503, detail="Execution event diagnostics unavailable"
        ) from None
    if diagnostics is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return diagnostics
