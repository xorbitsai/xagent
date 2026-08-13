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
from xagent.web.models.task import Task
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


def is_workforce_manager_removal_safe(
    workforce: Workforce,
    manager: Agent | None,
    *,
    used_as_other_manager: bool,
    used_as_worker: bool,
) -> bool:
    """Return whether removing this Workforce may also remove its manager.

    Shared by discard (undo a run-free draft) and permanent delete (remove
    any workforce outright) -- both may cascade into deleting a generated,
    exclusively-owned manager agent alongside the workforce itself.
    """
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


def _ensure_workforce_lifecycle_access(
    db: Session,
    user: User,
    workforce: Workforce | None,
) -> Workforce:
    """Shared 404/403 gate for lifecycle mutations (discard, permanent delete)."""
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
    whether removing it is safe per :func:`is_workforce_manager_removal_safe`.
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
    safe = is_workforce_manager_removal_safe(
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

    workforce = _ensure_workforce_lifecycle_access(db, user, workforce)
    workforce_id = int(workforce.id)
    deleted_manager_identity: tuple[int, int] | None = None

    try:
        workforce = _ensure_workforce_lifecycle_access(
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

    Stays a plain sync function -- its caller (an ``async def`` route) is
    expected to run it via ``asyncio.to_thread`` rather than call it
    directly, since the cascade below can walk a large run/trigger history
    and would otherwise block that thread for every concurrent request.
    Returns the PAUSE targets alongside the cascade-deleted triggers'
    teardown data (trigger, type, config); it does not perform the actual
    provider unregister calls itself, so that network I/O also has to
    happen through the caller's own ``asyncio.to_thread`` dispatch, the same
    way :func:`pause_workforce_tasks_after_archive` already does its own.
    """

    workforce = _ensure_workforce_lifecycle_access(db, user, workforce)
    workforce_id = int(workforce.id)
    deleted_manager_identity: tuple[int, int] | None = None

    try:
        workforce = _ensure_workforce_lifecycle_access(
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
        #
        # Known race, accepted rather than closed: trigger creation
        # (create_workforce_trigger / _create_trigger) does not acquire this
        # function's lifecycle fence, so a trigger created for this
        # workforce between this SELECT and this transaction's commit is
        # cascade-deleted at the DB level (workforce.triggers is re-resolved
        # fresh at flush time) but never appears in trigger_teardowns --
        # its provider-side binding silently leaks. Widening the fence to
        # cover trigger creation would close it, but that path is shared
        # with unrelated, actively-used trigger CRUD far outside this
        # workforce lifecycle feature; the window is also narrow (a create
        # request racing the exact commit of a delete on the same
        # workforce), so it's left as a known, documented gap rather than
        # risking new contention on that shared path.
        trigger_teardowns: list[tuple[AgentTrigger, str, dict[str, Any]]] = []
        for trigger in (
            db.query(AgentTrigger)
            .filter(AgentTrigger.workforce_id == workforce_id)
            .all()
        ):
            trigger_type = str(trigger.type)
            config = dict(trigger.config or {})
            # Expunge now, before this session's cascade-delete and commit:
            # not required for safety on the pinned SQLAlchemy version -- a
            # deleted-and-committed instance goes `detached` while keeping
            # its already-loaded values, so a plain attribute read on it
            # from the background thread would not itself raise -- but
            # expunging removes any dependency on that behavior and on
            # Workforce.triggers staying unloaded through this point (today
            # true because the fence's expire_all() above cleared any
            # earlier load, together with _load_workforce's eager-load set
            # not including .triggers), so a later change to either can't
            # quietly reintroduce a footgun here.
            db.expunge(trigger)
            trigger_teardowns.append((trigger, trigger_type, config))

        # WorkforceRun.task_id is ondelete="SET NULL" with a non-cascading
        # Task relationship, and workforce-run tasks default to
        # is_visible=True -- so without this, the conversations/traces a
        # "permanent delete" promises to remove would stay fully visible in
        # history/search after the workforce and its runs are gone. Hiding
        # rather than hard-deleting the Task rows only shares "never
        # hard-delete a Task" with stage_delete_agent's own cleanup (which
        # detaches by nulling Task.agent_id, not hides) -- other subsystems
        # (trace storage, workspace files) still key off the task id.
        task_ids_query = db.query(WorkforceRun.task_id).filter(
            WorkforceRun.workforce_id == workforce_id,
            WorkforceRun.task_id.isnot(None),
        )
        db.query(Task).filter(Task.id.in_(task_ids_query)).update(
            {Task.is_visible: False}, synchronize_session=False
        )

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

        # This ORM cascade (workers/runs/builder_messages/triggers, all
        # cascade="all, delete-orphan" on Workforce -- models/workforce.py)
        # loads every child row into Python and deletes them one by one
        # rather than a single DB-side statement, which is the same
        # already-accepted cost discard_draft_workforce's own db.delete(
        # workforce) has always paid for the same relationships. Adding
        # passive_deletes=True to lean on DB-level ON DELETE CASCADE instead
        # would fix that here, but this project's SQLite path never enables
        # foreign-key enforcement (no PRAGMA foreign_keys=ON in
        # models/database.py, the same reason AgentApiKey/Deployment above
        # need an explicit delete) -- passive_deletes would silently stop
        # cleaning up these rows on SQLite instead of just being slower,
        # which is worse than the N+1 cost it would remove. Left as a known,
        # bounded-scale cost instead; the caller offloading this whole
        # function via asyncio.to_thread (see the docstring above) keeps it
        # off the event loop regardless of workforce size.
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
