"""Retire a run's active interaction row when a legacy resume path answers it.

Four production sites inject a WebSocket, A2A, or v1 SDK user message
straight into a checkpoint instead of going through the native interaction
protocol's answer path: the online WebSocket injection, the deferred
WebSocket injection (``execute_resume_background``), the A2A resume-input
path, and the v1 ``POST .../reply`` resume-input path (``task_reply.py``).
Each of those sites, once its own message write has succeeded, must retire
the run's active ``task_interaction_requests`` row (if any) as
``terminated`` / ``answered_via_legacy_resume`` and clear
``tasks.interaction_protocol_version`` back to ``NULL`` in the same
transaction as that retirement -- otherwise a stale marker would keep
pointing a reader at a question the legacy path already answered by other
means.

Three resume-abandonment paths (the WebSocket lease restore, the A2A
prelease restore, and the v1 reply prelease restore) do the mirror-image
cleanup when a resume is undone instead of completed: if the marker no
longer corresponds to any active row, clear it. That clear is conditioned
on ``NOT EXISTS`` rather than unconditional, because an abandonment can
also race a resume that never even reached the injection call -- clearing
unconditionally there would erase a marker that is still correct for a
question that is still active.

A fourth WebSocket exit path does not clear the marker at all:
``_settle_resumed_task_lease`` (``websocket.py``) settles a cancelled or
failed resume by way of ``settle_task_lease_isolated`` ->
``fail_and_release_task_lease_no_commit``, neither of which touches
``interaction_protocol_version`` or ``task_interaction_requests``. This is
not an oversight to fix here -- it is the same lazy-clear posture the
marker's ownership comment on ``Task.interaction_protocol_version``
already documents: every exit path with no injection point to hang a
close or a clear on is allowed to leave a stale marker behind, because
readers filter on ``status`` before ever consulting it.

The two WebSocket sites and the A2A and v1 reply sites do not give this
close the same atomicity guarantee. At both WebSocket sites
(``websocket.py``), ``close_legacy_resume_interaction_sync`` opens its own
short transaction only after the message write has already committed --
the close is a best-effort step that follows the message write, not part
of it. If that separate transaction fails, both call sites only log the
failure; they do not retry, raise, or register a degradation signal. That
is a deliberate choice, not a gap: a stale marker degrades to the legacy
fallback question by design -- the same guarantee documented in the marker
comment on ``Task.interaction_protocol_version`` (``models/task.py``) -- so
there is nothing left to protect by escalating. Both WebSocket call sites
handle the failure of the delivery marker write immediately beside each
close call (``mark_user_message_delivery_sync``) the identical log-only
way, for the identical reason; see the inline comments beside each pair.
The A2A and v1 reply sites are different: each close call runs inside its
own resume-input fence transaction (``_update_a2a_resume_input_sync`` in
``a2a.py``, ``_update_reply_input_sync`` in ``task_reply.py``), committed
together with the ownership fence UPDATE that precedes it -- that is what
makes those two sites stronger than the WebSocket sites, not a claim that
the close is atomic with the message injection itself. The message
injection (``post_user_message`` at ``a2a.py:464`` / ``task_reply.py:512``)
commits on its own, earlier, in ``AgentRunner._persist_injected_context``;
only after that commit has already landed does the caller open the fence
transaction that writes the resumed input and closes the interaction row
together. A crash between those two commits leaves the interaction row
``active`` and the marker still set to ``1`` even though the injected
message already answered the question. That window is benign for the same
reason as the reader fallback documented above: a reader keys off
``status`` first, so a stale active row here just makes it fall back to
asking the legacy question again -- the same family of harmless window
as the message-injection replay that
``AgentRunner.inject_user_message`` short-circuits on a repeated turn
id.

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
            "legacy resume closing the active interaction row task_id=%s run_id=%s",
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

    A constraint on the batch that adds a production writer for
    ``task_interaction_requests``: the close statement matches on
    ``(task_id, run_id, active)`` alone -- nothing binds it to the
    specific question the injected user message answered. That is sound
    only while this run cannot stage a new row between the message write
    and this close. Injecting the message is exactly what resumes the
    agent, so once a production writer exists, the resumed agent may
    stage a fresh question in that window and this statement would
    retire it as if it had been answered. The writer batch must either
    key this close on the row observed before injection (read the
    primary key first) or add a staleness fence; it must not ship the
    statement as-is.
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


def clear_interaction_marker_if_unpaired(
    db: Session,
    *,
    task_id: int,
    run_id: str,
) -> None:
    """Zero the protocol marker if it no longer names any active row.

    Used by the three resume-abandonment paths (the WebSocket lease
    restore, the A2A prelease restore, and the v1 reply prelease restore)
    instead of ``close_legacy_resume_interaction``: an abandonment can
    happen before the injection call ever ran, so there is no row to close
    here -- only a marker to reconcile.

    The semantics are not "clear the marker" but "if the marker no longer
    corresponds to any active row, zero it": the UPDATE below only matches
    when NOT EXISTS an active row for this exact (task_id, run_id) pair,
    so a marker that still names a live question survives untouched. The
    outer ``Task.run_id == run_id`` predicate is what keeps this scoped to
    the run being abandoned -- if another run now owns the task (a new
    turn already started), this UPDATE matches zero rows regardless of the
    NOT EXISTS check, so an abandoned resume can never clear a marker that
    belongs to a run other than its own.

    Caller obligations: the caller owns the transaction (this function
    never commits or rolls back) and has already confirmed the resume is
    actually being abandoned (e.g. the lease restore's own ``restored``
    flag) before calling this. No lock read precedes this statement --
    unlike the close path, the caller's own non-key-column ``tasks`` UPDATE
    (``release_task_lease_no_commit``) already is the first statement this
    transaction directs at ``tasks`` or ``task_interaction_requests``, and
    it satisfies the same ordering and strength obligation a dedicated
    lock read would.

    Skipped entirely when ``task_interaction_requests`` does not exist.
    """
    if not interaction_requests_table_exists(db):
        return
    still_active = (
        sa.select(sa.literal(1))
        .where(
            TaskInteractionRequest.task_id == task_id,
            TaskInteractionRequest.run_id == run_id,
            TaskInteractionRequest.status == "active",
            TaskInteractionRequest.active_slot.isnot(None),
        )
        .exists()
    )
    db.execute(
        sa.update(Task)
        .where(
            Task.id == task_id,
            Task.run_id == run_id,
            ~still_active,
        )
        .values(interaction_protocol_version=None)
    )
