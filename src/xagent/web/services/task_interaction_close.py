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
cleanup when a resume is undone instead of completed: there is no row to
retire, only a marker to reconcile. Both clears -- theirs and the
injection paths' -- are conditioned on the same ``NOT EXISTS``
(``_active_row_exists``), for the same reason from two directions. An
abandonment can race a resume that never even reached the injection
call, and an injection can be handed no id because the pre-injection
read failed while a question was in fact live; in either case an
unconditional clear would erase a marker that is still correct for a
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
asking the legacy question again.

The replay ``AgentRunner.inject_user_message`` short-circuits on a
repeated turn id is not a window of that same benign family, and is
not covered by the argument above. A replay persists nothing and still
reports success to its caller, so an injection site cannot tell it from
a first attempt; the row that site observed before calling is then not
the question the replayed message answered but whatever the resumed
agent has asked since, and closing it retires a live question instead
of leaving a stale one behind. That is the opposite failure from the
one this paragraph describes, and it is not fixed here -- see the
comment at the online WebSocket injection site (``websocket.py``) for
the precondition it puts on the change that wires the first production
writer.

The close statement binds to one primary key, read before the injection
by ``active_interaction_id_sync`` and carried to the close by whichever
path is doing the injecting. It is not enough to match on
``(task_id, run_id, active)``: injecting the message is exactly what
resumes the agent, and a resumed agent can stage a fresh question before
the close runs, which an unbound statement would retire as though the
injected message had answered it. Reading the id at close time instead of
before the injection leaves that window exactly as wide as it is with no
key at all -- the value has to be the one observed before the message
went in.

Every rowcount the close statement below produces is classified the same
way, at the one place the classification happens
(``_classify_close_rowcount``): exactly one row closed is the expected
case and logs at info; zero rows is the overwhelmingly common case today
(no production writer has inserted into ``task_interaction_requests``
yet) and logs at debug, carrying with it a short description of why
nothing matched (``_unmatched_close_row``) -- "nothing was ever there"
and "something was there and is no longer this run's live question" are
the same rowcount and different situations; more than one row is
impossible twice over. The close statement binds to a primary key
(``id == interaction_id``), which matches at most one row on its own, and
its ``active_slot`` predicate falls under
``uq_task_interaction_active_slot``, which allows at most one row per task
with a non-NULL ``active_slot``. Both are named because a rowcount above
one means a different thing under each: the unique constraint gone, or the
primary key itself no longer unique. If either schema invariant were ever
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


def _unmatched_close_row(db: Session, *, interaction_id: int | None) -> str:
    """Why the close statement matched nothing, for the zero-rowcount log.

    A zero rowcount folds together outcomes an operator has to tell apart:
    nothing was ever there to close, and something was there but is no
    longer this run's live question -- another path retired it first, or it
    belongs to a run this close is not finishing. The rowcount alone cannot
    say which, and the difference decides whether a stale marker is the
    expected 100% case or the trace of a close that lost a race.

    Three answers, in increasing cost. ``no_id_read`` costs nothing and is
    every call today: the pre-injection read produced no id, either because
    no row was active or because the read could not be made at all (see
    ``active_interaction_id_sync``). Only a real id makes this issue a
    statement, a primary-key lookup on a table the caller's transaction has
    already written to: ``row_absent`` when no row carries that id at all,
    and ``row_status=<status>`` when one does -- which is the case that
    used to be invisible. ``status`` is NOT NULL, so a NULL scalar here
    means no row, never a row with an unset status.

    A row reported as ``row_status=active`` is not a contradiction: the
    close also fences on ``task_id`` and ``run_id``, so an id belonging to
    another task or another run of this task reads back active and still
    matches nothing.
    """
    if interaction_id is None:
        return "no_id_read"
    status = db.execute(
        sa.select(TaskInteractionRequest.status).where(
            TaskInteractionRequest.id == interaction_id
        )
    ).scalar()
    if status is None:
        return "row_absent"
    return f"row_status={status}"


def _classify_close_rowcount(
    rowcount: int, *, task_id: int, run_id: str, unmatched_row: str | None
) -> None:
    """Log one close statement's rowcount at the level its case deserves.

    ``unmatched_row`` describes why nothing matched and is read only on the
    zero branch; the caller passes ``None`` on the other two, where there
    is nothing unmatched to describe. It is a required argument rather than
    a defaulted one so that a caller has to decide, instead of silently
    getting a log line that has lost the distinction.
    """
    if rowcount == 1:
        logger.info(
            "legacy resume closing the active interaction row task_id=%s run_id=%s",
            task_id,
            run_id,
        )
    elif rowcount == 0:
        logger.debug(
            "legacy resume close matched no active interaction row "
            "task_id=%s run_id=%s unmatched_row=%s",
            task_id,
            run_id,
            unmatched_row,
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


def active_interaction_id_sync(task_id: int) -> int | None:
    """The id of this task's active native interaction row, or ``None``.

    Read this *before* injecting the user message, and pass what it
    returns to ``close_legacy_resume_interaction`` -- see this module's
    docstring for why that ordering is the whole point.

    Opens and closes its own short session. Three legacy-resume injection
    paths call it from a point where they hold no session of their own: the
    two fence-transaction paths (``a2a.py``, ``v1/task_reply.py``) open
    their fence only after the injection has already committed, and the
    WebSocket online chat injection holds none at all. The fourth
    legacy-resume path -- the WebSocket deferred injection reached through
    ``execute_resume_background`` -- does not call this function itself; it
    reuses the id the online handler already read, carried through
    ``pending_user_message["interaction_id"]``
    (``tests/web/api/test_websocket_owner_actor.py`` pins that it must not
    read again). A fourth call site does exist in ``src/``, but it is not
    an injection path at all -- see the resume command seam paragraph
    below.

    Keys off ``_active_native_row_criteria``
    (``task_interaction_service.py``), the same four-field predicate every
    other reader of "this task's one live row" uses, rather than a
    predicate of its own -- the close this feeds and the readers that show
    the question must not disagree about which row is live.

    Returns ``None`` when the read cannot be made at all -- no table yet,
    no session, a failing query. ``None`` closes nothing: the close
    statement matches zero rows and the active row survives -- and so
    does the marker, because the clear beside the close is conditioned on
    no active row remaining for this ``(task_id, run_id)`` pair (see
    ``close_legacy_resume_interaction``). A reader keeps seeing the live
    native question the read could not see, instead of falling back to
    the legacy transcript question. The alternative -- closing on the old
    unbound predicate when the read fails -- is the retire-the-wrong-row
    bug this function exists to prevent, so an unreadable id must never
    widen what the close matches.

    Two branches decide "no id" before the row lookup runs, and both mean
    the same thing as an empty lookup, not a failure:
    ``get_optional_session_local()`` returning ``None`` (no database
    configured in this process -- expected in tests, never in production),
    and ``tasks.interaction_protocol_version`` being ``NULL``. The marker
    gate is the write side of the same first step
    ``get_pending_interaction_question`` (``task_interaction_read.py``)
    takes on the read side: a NULL marker means no native row was ever
    staged for this task's current wait, so the interaction table goes
    unqueried. Keeping the two sides on one judgment is the point -- a
    question the read surface will not show must not be a question this
    close retires -- and the cost it removes is real: this function runs on
    every A2A resume, every v1 chat reply and every live WebSocket chat
    message, and under a NULL marker it now costs one primary-key lookup
    instead of an uncached catalog inspection plus a two-table join.

    That gate decides every call today: the only statements in ``src/``
    that write the column are this module's two clears, and both of them
    write ``NULL``, so the marker is NULL for every task and this branch
    returns ``None`` before anything below it runs.

    This is also the resume command seam's reader (``websocket.py``'s
    refusal gate for a resume whose payload cannot prove it answered the
    active question). That seam used to carry a byte-identical copy of the
    query; one reader is what keeps the close, the refusal gate and the
    read surface from drifting into three notions of "the live row".
    """
    from ..models.database import get_optional_session_local
    from .task_interaction_service import _active_native_row_criteria

    SessionLocal = get_optional_session_local()
    if SessionLocal is None:
        return None
    try:
        db = SessionLocal()
    except Exception:
        logger.warning(
            "could not open a session to read the active interaction row "
            "for task_id=%s; the close will match no row",
            task_id,
            exc_info=True,
        )
        return None
    try:
        marker = (
            db.query(Task.interaction_protocol_version)
            .filter(Task.id == task_id)
            .scalar()
        )
        if marker is None:
            return None
        if not interaction_requests_table_exists(db):
            return None
        row = (
            db.query(TaskInteractionRequest.id)
            .join(Task, Task.id == TaskInteractionRequest.task_id)
            .filter(
                TaskInteractionRequest.task_id == task_id,
                *_active_native_row_criteria(),
            )
            .first()
        )
        return int(row[0]) if row is not None else None
    except Exception:
        logger.warning(
            "the active interaction row lookup failed for task_id=%s; "
            "the close will match no row",
            task_id,
            exc_info=True,
        )
        return None
    finally:
        db.close()


def _active_row_exists(*, task_id: int, run_id: str) -> sa.Exists:
    """The EXISTS clause both marker clears in this module are conditioned on.

    "This ``(task_id, run_id)`` pair still has a live interaction row":
    the row's own lifecycle state (``status == "active"`` and
    ``active_slot IS NOT NULL``) plus the pair itself. Written once and
    referenced from both UPDATE statements instead of twice, because the
    two clears mean the same thing by "no longer names any active row"
    and must not drift into two readings of it.

    Negated at both use sites -- neither clear runs while a live row
    remains. The ``run_id`` leg compares against the value the caller
    passed, which in both cases is the run the caller is finishing or
    abandoning, so a live row belonging to some other run of the same
    task is out of scope in both directions: it does not hold this
    clear back, and this clear does not touch its marker.

    Not the same predicate as ``_active_native_row_criteria``
    (``task_interaction_service.py``), which this module also imports, in
    ``active_interaction_id_sync``. That one's third leg is
    ``TaskInteractionRequest.run_id == Task.run_id`` -- a comparison
    against the task row a reader has already joined -- while this one
    compares against a caller-supplied value. Today the two select the
    same rows at both use sites, but only because each outer UPDATE
    already pins ``Task.run_id == run_id``; the agreement is a
    consequence of the surrounding statement, not a property of either
    predicate. Swapping this clause for that one is therefore not a
    simplification: it would turn both subqueries into correlated
    subqueries against the row being updated, and it would re-point the
    compensation paths' clear at whichever run the task currently
    carries instead of at the run being abandoned.
    """

    return (
        sa.select(sa.literal(1))
        .where(
            TaskInteractionRequest.task_id == task_id,
            TaskInteractionRequest.run_id == run_id,
            TaskInteractionRequest.status == "active",
            TaskInteractionRequest.active_slot.isnot(None),
        )
        .exists()
    )


def close_legacy_resume_interaction(
    db: Session,
    *,
    task_id: int,
    run_id: str,
    interaction_id: int | None,
) -> int:
    """Retire the run's active interaction row and clear the task's marker.

    ``interaction_id`` is the row observed *before* the user message was
    injected, from ``active_interaction_id_sync``; the close binds to it,
    so it retires that exact question and nothing else. This module's
    docstring carries the argument for why the value has to come from
    before the injection.

    ``interaction_id=None`` means the pre-injection read produced no id --
    either because no row was active at injection time, or because the
    read could not be made at all (see ``active_interaction_id_sync``).
    SQLAlchemy renders it as ``id IS NULL``, which is never true of a
    primary key, so the close matches zero rows. What happens to the
    marker then is not fixed by this argument and depends on which of
    those two cases it was: the clear below runs its own check, and
    zeroes the marker only if nothing active is left. Nothing was ever
    there in the first case, so the marker clears; a live row the read
    could not see is still there in the second, so it does not. Do not
    "simplify" this into skipping the close statement or dropping its id
    predicate: both change which rows the close can touch.

    Caller obligations, because neither happens here: the caller has
    already confirmed ``interaction_requests_table_exists(db)`` -- this
    function issues no catalog check of its own and unconditionally
    targets both tables -- and the caller owns the transaction; this
    function never commits or rolls back.

    Both statements always run, and neither is conditioned on the other's
    rowcount -- but the clear carries a condition of its own: it zeroes
    the marker only when no active row for this ``(task_id, run_id)``
    pair remains once the close above has had its turn. That covers the
    case this function is normally in (the close retired the one live
    row, so nothing is left and the marker goes) and today's 100% case
    (this run never staged a row at all, so nothing was ever there and
    the marker still goes, since the table has no production writer yet),
    while refusing the case that used to lose a question: the
    pre-injection read came back ``None`` -- or came back with an id the
    close did not match -- while a live row is in fact still sitting
    there. Zeroing the marker then would point every reader at the legacy
    transcript question while the native row it named is still active and
    unanswered.

    The condition is ``_active_row_exists``, the same clause
    ``clear_interaction_marker_if_unpaired`` is built on: "no active row
    for this pair" has to mean one thing in this module, not two.

    Returns the close statement's rowcount, classified and logged by
    ``_classify_close_rowcount`` -- see the module docstring for why a
    rowcount greater than 1 is logged, not raised.
    """
    now = datetime.now(timezone.utc)
    close_result = db.execute(
        sa.update(TaskInteractionRequest)
        .where(
            TaskInteractionRequest.id == interaction_id,
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
    _classify_close_rowcount(
        rowcount,
        task_id=task_id,
        run_id=run_id,
        # Only the zero branch has anything to describe, and only a
        # non-None id makes this reach the database at all, so today's
        # every-call path (no id read, zero rows) still issues nothing
        # extra.
        unmatched_row=(
            _unmatched_close_row(db, interaction_id=interaction_id)
            if rowcount == 0
            else None
        ),
    )
    db.execute(
        sa.update(Task)
        .where(
            Task.id == task_id,
            Task.run_id == run_id,
            ~_active_row_exists(task_id=task_id, run_id=run_id),
        )
        .values(interaction_protocol_version=None)
    )
    return rowcount


def close_legacy_resume_interaction_sync(
    *, task_id: int, run_id: str, interaction_id: int | None
) -> int:
    """Close + clear from a synchronous or ``asyncio.to_thread`` caller.

    Shared by both WebSocket legacy-resume injection sites (the online
    handler and the deferred ``execute_resume_background`` path): each
    calls this via ``run_db_io_cancellation_safe`` with no transaction of
    its own open first, and this function commits before returning.

    Keyword-only, matching the function it wraps. The three arguments are
    an int, a str and an optional int, and the two that are ints are a task
    primary key and an interaction primary key -- a positional call that
    transposed them would be accepted by every type in the signature and
    would close some other task's row.

    ``interaction_id`` is passed straight through -- see
    ``close_legacy_resume_interaction`` for where it has to come from and
    what ``None`` means.

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
        rowcount = close_legacy_resume_interaction(
            db, task_id=task_id, run_id=run_id, interaction_id=interaction_id
        )
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
    db.execute(
        sa.update(Task)
        .where(
            Task.id == task_id,
            Task.run_id == run_id,
            ~_active_row_exists(task_id=task_id, run_id=run_id),
        )
        .values(interaction_protocol_version=None)
    )
