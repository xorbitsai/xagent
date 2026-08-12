"""Retire a run's active interaction row when a legacy resume path answers it.

Three production sites inject a WebSocket or A2A user message straight into
a checkpoint instead of going through the native interaction protocol's
answer path: the online WebSocket injection, the deferred WebSocket
injection (``execute_resume_background``), and the A2A resume-input path.
Each of those sites, once its own message write has succeeded, must retire
the run's active ``task_interaction_requests`` row (if any) as
``terminated`` / ``answered_via_legacy_resume`` and clear
``tasks.interaction_protocol_version`` back to ``NULL`` in the same
transaction as that retirement -- otherwise a stale marker would keep
pointing a reader at a question the legacy path already answered by other
means.

Every rowcount the close statement below produces is classified the same
way, at the one place the classification happens
(``_classify_close_rowcount``): exactly one row closed is the expected
case and logs at info; zero rows is the overwhelmingly common case today
(no production writer has inserted into ``task_interaction_requests``
yet) and logs at debug; more than one row is impossible under
``uq_task_interaction_active_slot`` (it allows at most one row per task
with a non-NULL ``active_slot``), but if that schema invariant were ever
broken, this module logs at error and registers a degradation signal
rather than raising -- a resume injection or an already-durable input
write must not be turned into a failure by a check meant only to catch
schema corruption after the fact.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import sqlalchemy as sa
from sqlalchemy.orm import Session

from ..models.database import get_session_local
from ..models.task import Task
from ..models.task_interaction import TaskInteractionRequest
from .ops_signals import (
    INTERACTION_LEGACY_RESUME_CLOSE_ROWCOUNT_ANOMALY,
    register_degradation,
)
from .task_interaction_schema import interaction_requests_table_exists
from .task_lease_service import _rowcount

logger = logging.getLogger(__name__)


def _classify_close_rowcount(rowcount: int, *, task_id: int, run_id: str) -> None:
    if rowcount == 1:
        logger.info(
            "legacy resume closed the active interaction row task_id=%s run_id=%s",
            task_id,
            run_id,
        )
    elif rowcount == 0:
        logger.debug(
            "legacy resume close matched no active interaction row "
            "task_id=%s run_id=%s",
            task_id,
            run_id,
        )
    else:
        logger.error(
            "legacy resume close matched %s active interaction rows for one "
            "task, expected at most one task_id=%s run_id=%s",
            rowcount,
            task_id,
            run_id,
        )
        register_degradation(
            INTERACTION_LEGACY_RESUME_CLOSE_ROWCOUNT_ANOMALY,
            f"task_id={task_id} run_id={run_id} matched {rowcount} rows",
        )


def close_legacy_resume_interaction(
    db: Session,
    *,
    task_id: int,
    run_id: str,
) -> int:
    """Retire the run's active interaction row and clear the task's marker.

    Caller obligations, because neither happens here: the caller has
    already confirmed ``interaction_requests_table_exists(db)`` -- this
    function issues no catalog check of its own and unconditionally
    targets both tables -- and the caller owns the transaction; this
    function never commits or rolls back.

    Both statements always run, regardless of the close statement's
    rowcount: the clear is not conditioned on having actually closed a
    row, because a task's marker can legitimately need clearing even when
    this run never staged an interaction row at all (today's 100% case,
    since this table has no production writer yet).

    Returns the close statement's rowcount, classified and logged by
    ``_classify_close_rowcount`` -- see the module docstring for why a
    rowcount greater than 1 is logged, not raised.
    """
    now = datetime.now(timezone.utc)
    close_result = db.execute(
        sa.update(TaskInteractionRequest)
        .where(
            TaskInteractionRequest.task_id == task_id,
            TaskInteractionRequest.run_id == run_id,
            TaskInteractionRequest.status == "active",
            TaskInteractionRequest.active_slot.isnot(None),
        )
        .values(
            status="terminated",
            active_slot=None,
            terminal_reason="answered_via_legacy_resume",
            terminated_at=now,
            # updated_at has onupdate=func.now(); bind it explicitly to
            # this same caller-clock value instead of letting the server
            # clock fill it, matching interaction_handoff's explicit
            # created_at binding (task_interaction_staging.py). Do not
            # delete this as redundant -- removing it would let this row's
            # timestamp source diverge from every other writer's.
            updated_at=now,
        )
    )
    rowcount = _rowcount(close_result)
    _classify_close_rowcount(rowcount, task_id=task_id, run_id=run_id)
    db.execute(
        sa.update(Task)
        .where(Task.id == task_id, Task.run_id == run_id)
        .values(interaction_protocol_version=None)
    )
    return rowcount


def close_legacy_resume_interaction_sync(task_id: int, run_id: str) -> int:
    """Close + clear from a synchronous or ``asyncio.to_thread`` caller.

    Shared by both WebSocket legacy-resume injection sites (the online
    handler and the deferred ``execute_resume_background`` path): each
    calls this via ``run_db_io_cancellation_safe`` with no transaction of
    its own open first, and this function commits before returning.

    Skips entirely, without opening the lock read below, when
    ``task_interaction_requests`` does not exist on this deployment -- a
    deployment not yet migrated to that table must not pay for, or fail
    on, a lock read against a table it does not have.
    """
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        if not interaction_requests_table_exists(db):
            return 0
        # First statement this transaction directs at tasks or
        # task_interaction_requests. purge_task_rows NULLs this task's checkpoint
        # pointer columns on tasks (task_deletion.py:45) before it deletes the
        # task's interaction rows, so closing the interaction row first and
        # clearing the marker second would close a lock cycle with it -- at every
        # lock level purge itself could take, because the cycle is closed by that
        # UPDATE, not by purge's own locking read. FOR NO KEY UPDATE, not
        # FOR UPDATE: a concurrent stager's replacement INSERT needs KEY SHARE on
        # this same parent row for its foreign key, and FOR UPDATE would block it,
        # closing a second cycle. Both halves are required; see the caller
        # obligation in task_interaction_staging.py's module docstring. The gate
        # call above only inspects the catalog and takes no row lock, so it does
        # not precede this in the sense the obligation means. On SQLite the clause
        # is dropped and the single-writer lock serializes instead.
        db.execute(
            sa.select(Task.id).where(Task.id == task_id).with_for_update(key_share=True)
        ).first()
        rowcount = close_legacy_resume_interaction(db, task_id=task_id, run_id=run_id)
        db.commit()
        return rowcount
