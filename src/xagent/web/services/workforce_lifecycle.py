from __future__ import annotations

import logging
from typing import Any, cast

from fastapi import HTTPException
from sqlalchemy import update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from xagent.web.models.agent import Agent, is_workforce_generated_manager_agent
from xagent.web.models.agent_api_key import AgentApiKey
from xagent.web.models.database import release_db_connection_if_clean
from xagent.web.models.deployment import Deployment, DeploymentOwnerType
from xagent.web.models.trigger import AgentTrigger
from xagent.web.models.user import User
from xagent.web.models.workforce import Workforce, WorkforceAgent, WorkforceRun

from .agent_store import AgentStore
from .hot_path_cache import invalidate_agent_cache
from .workforce_access import can_edit_workforce
from .workforce_runtime import (
    WorkforceRunPauseTarget,
    cancel_active_workforce_runs,
)

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


def _lock_generated_manager(
    db: Session,
    workforce: Workforce,
) -> tuple[Agent | None, bool]:
    """Lock the manager row and report whether it may be deleted alongside.

    Returns ``(None, True)`` when the manager is not a generated one (there
    is nothing extra to delete), otherwise the locked generated manager and
    whether removing it is safe per :func:`is_workforce_manager_discard_safe`.
    """
    manager = (
        db.query(Agent)
        .filter(Agent.id == int(workforce.manager_agent_id))
        .with_for_update()
        .first()
    )
    if manager is None or not is_workforce_generated_manager_agent(manager):
        return None, True

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
    safe = is_workforce_manager_discard_safe(
        workforce,
        manager,
        used_as_other_manager=used_as_other_manager,
        used_as_worker=used_as_worker,
    )
    return manager, safe


def _lock_generated_manager_for_discard(
    db: Session,
    workforce: Workforce,
) -> Agent | None:
    manager, safe = _lock_generated_manager(db, workforce)
    if not safe:
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


def delete_workforce_permanently(
    db: Session,
    user: User,
    workforce: Workforce | None,
) -> tuple[
    list[WorkforceRunPauseTarget], list[tuple[AgentTrigger, str, dict[str, Any]]]
]:
    """Atomically hard-delete a workforce and everything that hangs off it.

    Unlike :func:`discard_draft_workforce`, this accepts any status and any
    run history: in-flight runs are cancelled in the same transaction (the
    caller must dispatch PAUSE to the returned targets after commit, exactly
    like archive), and a generated manager that is still referenced by other
    workforces is kept instead of blocking the delete.

    Returns the PAUSE targets alongside the cascade-deleted triggers'
    teardown data (trigger, type, config) -- this function stays sync and
    does not perform the actual provider unregister calls itself: it runs
    directly on the request's event-loop thread (its caller is an ``async
    def`` route, not one FastAPI offloads to a worker thread), so any
    provider network I/O has to happen through the caller's own
    ``asyncio.to_thread`` the same way :func:`pause_workforce_tasks_after_archive`
    already does, or it would block that thread for every concurrent request.
    """

    workforce = _ensure_discard_access(db, user, workforce)
    workforce_id = int(workforce.id)
    deleted_manager_identity: tuple[int, int] | None = None

    try:
        workforce = _ensure_discard_access(
            db,
            user,
            acquire_workforce_lifecycle_fence(db, workforce_id),
        )
        # Same reasoning as archive: flipping/deleting the row alone leaves
        # in-flight runs executing, so cancel them under the fence and let
        # the caller dispatch PAUSE after this transaction commits.
        pause_targets = cancel_active_workforce_runs(db, workforce_id)
        generated_manager, manager_deletable = _lock_generated_manager(db, workforce)

        # The ORM cascade below deletes trigger rows without the trigger CRUD
        # path's provider teardown, which would leak provider-side bindings
        # (Gmail watches etc.). Capture what teardown needs while the rows
        # are still readable; the actual unregister runs after commit, same
        # ordering as _delete_trigger.
        trigger_teardowns: list[tuple[AgentTrigger, str, dict[str, Any]]] = []
        for trigger in (
            db.query(AgentTrigger)
            .filter(AgentTrigger.workforce_id == workforce_id)
            .all()
        ):
            trigger_type = str(trigger.type)
            config = dict(trigger.config or {})
            # Expunge now, before this session's cascade-delete and commit
            # expire every instance it still tracks: an expunged object
            # keeps its already-loaded column values in memory and is never
            # touched by this session again, so it stays safely readable
            # from the background thread that runs the actual provider
            # teardown post-commit -- accessing an attribute on a still-
            # tracked, expired, deleted-row instance there would raise
            # DetachedInstanceError/ObjectDeletedError instead. Cascade
            # delete is unaffected: Workforce.triggers hasn't been loaded
            # yet (the fence's expire_all() above cleared any earlier
            # load), so flush resolves it with its own fresh query,
            # independent of this separately-queried, now-detached copy.
            db.expunge(trigger)
            trigger_teardowns.append((trigger, trigger_type, config))

        # Neither table is reachable through the ORM cascades on Workforce,
        # and SQLite runs without foreign-key enforcement, so the DB-level
        # ON DELETE CASCADE on agent_api_keys (and the FK-less deployments
        # table) cannot be relied on here.
        db.query(AgentApiKey).filter(AgentApiKey.workforce_id == workforce_id).delete(
            synchronize_session=False
        )
        db.query(Deployment).filter(
            Deployment.owner_type == DeploymentOwnerType.WORKFORCE.value,
            Deployment.owner_id == workforce_id,
        ).delete(synchronize_session=False)

        db.delete(workforce)
        db.flush()

        if generated_manager is not None and manager_deletable:
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
                "Failed to invalidate the deleted Workforce manager cache",
                exc_info=True,
            )
    return pause_targets, trigger_teardowns
