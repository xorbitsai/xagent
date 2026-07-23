from __future__ import annotations

import logging
from typing import Any, cast

from fastapi import HTTPException
from sqlalchemy import update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from xagent.web.models.agent import Agent, is_workforce_generated_manager_agent
from xagent.web.models.database import release_db_connection_if_clean
from xagent.web.models.user import User
from xagent.web.models.workforce import Workforce, WorkforceAgent, WorkforceRun

from .agent_store import AgentStore
from .hot_path_cache import invalidate_agent_cache
from .workforce_access import can_edit_workforce

logger = logging.getLogger(__name__)


def is_workforce_manager_discard_safe(
    workforce: Workforce,
    manager: Agent | None,
    *,
    used_as_other_manager: bool,
    used_as_worker: bool,
) -> bool:
    """Return whether discard may also remove this Workforce's manager."""
    if manager is None or not is_workforce_generated_manager_agent(manager):
        return True
    return bool(
        int(manager.user_id) == int(workforce.owner_user_id)
        and not used_as_other_manager
        and not used_as_worker
    )


def _conflict(code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=409,
        detail={"code": code, "message": message},
    )


def _ensure_discard_access(
    db: Session,
    user: User,
    workforce: Workforce | None,
) -> Workforce:
    if workforce is None:
        raise HTTPException(status_code=404, detail="Workforce not found")
    if not can_edit_workforce(db, user, workforce):
        raise HTTPException(status_code=403, detail="Access denied")
    return workforce


def acquire_workforce_lifecycle_fence(
    db: Session,
    workforce_id: int,
) -> Workforce | None:
    """Serialize lifecycle decisions and return a current Workforce row.

    The no-op UPDATE is intentional: it takes a row lock on server databases
    and a writer lock on SQLite, where SELECT FOR UPDATE is ignored. Ending the
    caller's initial read transaction first also avoids a stale WAL snapshot
    when SQLite upgrades the operation to a write.
    """

    release_db_connection_if_clean(db)
    result = cast(
        "CursorResult[Any]",
        db.execute(
            update(Workforce)
            .where(Workforce.id == workforce_id)
            .values(id=Workforce.id, updated_at=Workforce.updated_at)
            .execution_options(synchronize_session=False)
        ),
    )
    if result.rowcount == 0:
        return None

    db.expire_all()
    return (
        db.query(Workforce)
        .populate_existing()
        .filter(Workforce.id == workforce_id)
        .first()
    )


def _lock_generated_manager_for_discard(
    db: Session,
    workforce: Workforce,
) -> Agent | None:
    manager = (
        db.query(Agent)
        .filter(Agent.id == int(workforce.manager_agent_id))
        .with_for_update()
        .first()
    )
    if manager is None or not is_workforce_generated_manager_agent(manager):
        return None

    manager_id = int(manager.id)
    used_as_other_manager = (
        db.query(Workforce.id)
        .filter(
            Workforce.manager_agent_id == manager_id,
            Workforce.id != int(workforce.id),
        )
        .first()
        is not None
    )
    used_as_worker = (
        db.query(WorkforceAgent.id)
        .filter(WorkforceAgent.agent_id == manager_id)
        .first()
        is not None
    )
    if not is_workforce_manager_discard_safe(
        workforce,
        manager,
        used_as_other_manager=used_as_other_manager,
        used_as_worker=used_as_worker,
    ):
        raise _conflict(
            "workforce_not_discardable",
            "The generated manager cannot be safely discarded.",
        )
    return manager


def discard_draft_workforce(
    db: Session,
    user: User,
    workforce: Workforce | None,
) -> None:
    """Atomically discard one run-free draft and its owned manager, if any."""

    workforce = _ensure_discard_access(db, user, workforce)
    workforce_id = int(workforce.id)
    deleted_manager_identity: tuple[int, int] | None = None

    try:
        workforce = _ensure_discard_access(
            db,
            user,
            acquire_workforce_lifecycle_fence(db, workforce_id),
        )
        if workforce.status != "draft":
            raise _conflict(
                "workforce_not_discardable",
                "Only draft workforces can be discarded.",
            )
        if (
            db.query(WorkforceRun.id)
            .filter(WorkforceRun.workforce_id == workforce_id)
            .first()
            is not None
        ):
            raise _conflict(
                "workforce_has_runs",
                "Workforces with run history cannot be discarded.",
            )

        generated_manager = _lock_generated_manager_for_discard(db, workforce)
        db.delete(workforce)
        db.flush()

        if generated_manager is not None:
            deleted_manager_identity = (
                int(generated_manager.user_id),
                int(generated_manager.id),
            )
            AgentStore(db).stage_delete_agent(generated_manager)

        db.commit()
    except Exception:
        db.rollback()
        raise

    if deleted_manager_identity is not None:
        try:
            invalidate_agent_cache(*deleted_manager_identity)
        except Exception:
            logger.warning(
                "Failed to invalidate the discarded Workforce manager cache",
                exc_info=True,
            )
