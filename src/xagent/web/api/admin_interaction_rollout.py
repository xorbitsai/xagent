"""Admin diagnostic endpoint for the interaction rollout policy and gate state.

Read-only, admin-authenticated. This is a diagnostic surface, not part of
the gate itself: it queries active-row stats on demand rather than caching
them, so a stale reading is never possible, at the cost of a query per
request.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..auth_dependencies import get_current_user
from ..models.database import get_db
from ..models.user import User
from ..services.interaction_rollout import (
    counters_snapshot,
    get_interaction_rollout_policy,
)
from ..services.ops_signals import active_degradations
from ..services.task_interaction_schema import interaction_requests_table_exists

router = APIRouter(
    prefix="/api/admin/interaction-rollout", tags=["admin-interaction-rollout"]
)
logger = logging.getLogger(__name__)


@router.get("")
async def get_interaction_rollout_diagnostics(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    """Resolved policy, active-row stats, gate counters, and degradations.

    Failure shape is three-way, not the usual two: table absent is a normal
    pre-migration deployment state and returns 200 with ``schema_absent``
    true and no count fields (an empty allow list read as "0 active rows"
    would be misleading -- there is no table to have rows). A stats query
    that raises (a transient DB error) returns 503, not a fabricated
    ``active_count: 0`` -- a 0 here is the specific number operators watch
    to confirm a rollback has drained, and a fake 0 during an outage would
    read as "drained" when the truth is "unknown."
    """
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    policy = get_interaction_rollout_policy()
    response: dict[str, object] = {
        "policy": {
            "mode": policy.mode,
            "native_sources": sorted(policy.native_sources),
            "native_protocol_version": policy.native_protocol_version,
        },
        "counters": counters_snapshot(),
        "degradations": active_degradations(),
    }

    if not interaction_requests_table_exists(db):
        response["schema_absent"] = True
        return response

    try:
        row = db.execute(
            text(
                "SELECT count(*) AS active_count, "
                "min(created_at) AS oldest_created_at "
                "FROM task_interaction_requests "
                "WHERE status = 'active' AND active_slot IS NOT NULL"
            )
        ).one()

        # Result-type marshaling for the raw column value differs by
        # backend: PostgreSQL hands back a tz-aware datetime, SQLite hands
        # back a plain str (its DATETIME type has no native temporal type).
        # Both shapes -- and any further marshaling surprise -- are handled
        # in this same try so they land in the deliberate 503 branch below,
        # never as an unhandled 500.
        oldest_created_at = row.oldest_created_at
        oldest_age_seconds: float | None = None
        if oldest_created_at is not None:
            if isinstance(oldest_created_at, str):
                oldest_created_at = datetime.fromisoformat(oldest_created_at)
            if oldest_created_at.tzinfo is None:
                oldest_created_at = oldest_created_at.replace(tzinfo=timezone.utc)
            oldest_age_seconds = (
                datetime.now(timezone.utc) - oldest_created_at
            ).total_seconds()
    except Exception as exc:
        logger.exception("Interaction rollout diagnostics stats query failed")
        raise HTTPException(
            status_code=503,
            detail="Interaction rollout diagnostics stats query failed",
        ) from exc

    response["active_count"] = row.active_count
    response["oldest_age_seconds"] = oldest_age_seconds
    return response
