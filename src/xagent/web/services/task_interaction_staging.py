"""Stage one interaction request row into a session the caller already owns.

``stage_interaction_request`` does the plain-Python validation, the
caller's own flush, a keyed idempotency pre-read, the reclaim UPDATE that
retires a stale or superseded active row, and the active-row INSERT under a
savepoint this function owns.

Caller obligations, because none of them happen here:

* This function joins the ``Session`` passed in; it never calls
  ``db.commit()`` or the outer ``db.rollback()``. It manages only its own
  inner savepoint (see below) -- the caller owns the transaction and
  decides when to commit or roll it back.
* It never notifies, prunes, or writes any column of ``tasks``.
* If this function raises ``InteractionOwnerStateError``, the session is
  left mid-transaction with no savepoint of its own to roll back to (the
  failure happened before this function opened one) -- the caller must
  roll back the whole transaction before issuing another statement on it.
  Every other exception this function raises is contained by its own inner
  savepoint.

Zero production callers as of this module's introduction: a static test
(``tests/web/services/test_interaction_staging_production_gate.py``) asserts
that no production module calls this function. See that test's docstring
for the removal condition.

Every rejection the database's 23 CHECK constraints could raise on the
INSERT is rejected in plain Python first, inside this function's own
validation block, because the post-conflict re-check (step 7 below)
classifies any ``IntegrityError`` that does not match the caller's own
identity key as a slot conflict -- ``InteractionSlotTaken``. That re-check
cannot tell a real slot conflict apart from a programming error (an
out-of-vocabulary ``origin``, a non-positive TTL, a ``None`` payload past
``JSON(none_as_null=True))``) that also violates a CHECK: both land on the
same ``else`` branch. The backstop is the database's constraint set, not
the primary defense; the primary defense is this validation block, and it
must reject everything the INSERT could reject before the INSERT is ever
issued. Adding a column CHECK to ``TaskInteractionRequest`` without adding
the matching Python check here reopens that misclassification funnel by
one more case.

Concurrency note (inherited, not introduced here): on PostgreSQL, an
``IntegrityError`` poisons the rest of the transaction
(``InFailedSqlTransaction``) until something rolls back at least to the
savepoint that was open when it fired; SQLite instead lets the session keep
issuing statements. This function's own inner ``db.begin_nested()`` around
the INSERT is what makes the post-conflict re-check possible on both
backends -- see the function's docstring for why a savepoint shared with
the caller does not work.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..models.task_interaction import TaskInteractionRequest
from .task_command_transport import COMMAND_ID_PATTERN

_KIND_VOCABULARY = frozenset({"clarification"})
_ORIGIN_VOCABULARY = frozenset(
    {"internal", "sdk", "a2a", "trigger", "widget", "shared_link"}
)
_RESUME_LOCATOR_FORMAT = "trace_event_pk_v1"
_RESUME_CHECKPOINT_TYPE = "agent_execution_checkpoint"
_PROTOCOL_VERSION = 1


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class InteractionHandoffError(RuntimeError):
    """Common base for every exception this module raises.

    Deliberately not the dispatch key ``interaction_handoff`` uses to decide
    what to swallow -- see ``_SWALLOWED`` below for why an explicit tuple is
    used instead of ``isinstance(exc, InteractionHandoffError)``.
    """


class InteractionSlotTaken(InteractionHandoffError):
    """The active-row INSERT collided with another request's active slot.

    Raised only when the post-conflict re-check (step 7) does not find the
    caller's own ``(task_id, run_id, request_idempotency_key)`` among the
    surviving rows after an ``IntegrityError``. Because that re-check cannot
    distinguish a real slot conflict from any other ``IntegrityError`` the
    INSERT could have raised, this exception is also what a programming
    error that skipped the plain-Python validation block would surface as.
    See the module docstring's misclassification-funnel paragraph.
    """


class InteractionRequestClosed(InteractionHandoffError):
    """The idempotency pre-read found this identity key already terminal.

    Raised when ``(task_id, run_id, request_idempotency_key)`` names a row
    whose ``status`` is ``answered`` or ``terminated`` rather than
    ``active``. Identity is scoped by ``run_id``: a key that was reclaimed
    to ``terminated`` earlier in the *same* run still raises this on reuse
    (there is no cross-run leakage to guard against, because a different
    run never shares this row at all).
    """


class InteractionAnchorCorrupt(InteractionHandoffError):
    """The resume anchor handed to this call is not self-consistent.

    Raised by the shared field-validation helper both
    ``stage_interaction_request`` and ``interaction_handoff`` call: an empty
    or wrong-vocabulary anchor field, or a missing ``trace_event_id``. This
    is the plain-Python half of anchor validation -- see
    ``InteractionAnchor``'s docstring for what the other half (a database
    read resolving absence/unavailable/corrupt against the live
    ``trace_events`` row) is and why it is not this module's job.
    """


class InteractionAttemptMismatch(InteractionHandoffError):
    """The caller's lease no longer names the task's current attempt.

    Raised by ``interaction_handoff.__enter__`` when
    ``lease.attempt_id is not None`` and it disagrees with
    ``task.lease_attempt_id``. See ``TaskLease.attempt_id``'s docstring
    (``task_lease_service.py``) for the ``is not None`` gate: a permanent
    ``None`` lease (the ambient snapshot ``_task_lease_snapshot`` builds)
    must read as "cannot prove attempt identity", never as "matches" or "does
    not match".
    """


class InteractionRunPartitionMismatch(InteractionHandoffError):
    """The interaction's run identity does not match its resume anchor's.

    Raised by ``stage_interaction_request``'s validation block when
    ``run_id != anchor.resume_run_partition``. There is deliberately no
    database CHECK for this comparison
    (``ck_task_interaction_requests_run_partition_matches`` was written and
    then deleted from the model -- see ``task_interaction.py``'s
    ``__table_args__`` comment): forcing it into a CHECK would turn the kind
    of corruption #1071 exists to detect into a row that cannot be written
    at all, which the post-conflict re-check would then misclassify as a
    slot conflict instead of surfacing as the corruption it actually is.

    PR-C2a deliberately swallows this exception -- see ``_SWALLOWED`` and
    ``interaction_handoff``'s docstring for the reasoning and for why that
    is a scoped override of this primitive's own default, not this
    primitive's design intent.
    """


class InteractionOwnerStateError(InteractionHandoffError):
    """The caller's own pending writes failed to flush before staging began.

    Raised by ``stage_interaction_request``'s first ``db.flush()`` when
    called directly, mirroring ``TaskCommandOwnerStateError``
    (``task_command_transport.py``) for the identical reason: the wrapper's
    ``except IntegrityError`` around the INSERT must classify conflicts on
    that INSERT, and must not also catch a failure that has nothing to do
    with it. Called through ``interaction_handoff`` instead, it can also
    fire one statement earlier, from that context manager's own
    ``db.begin_nested()`` -- ``Session.begin_nested()`` always flushes
    pending state first, to establish the SAVEPOINT after whatever is
    already pending, so the same caller-write failure can surface there
    before ``stage_interaction_request`` is ever reached. After this is
    raised the session is unusable until the caller rolls back -- and,
    because it fires before any savepoint in this module has been opened,
    there is no savepoint of this module's own to roll back to. This
    exception is never swallowed:
    the caller's own write failed, not the interaction's, and on PostgreSQL
    the transaction is already poisoned by the time this fires, so
    degrading in place is not physically possible.
    """


# Explicit tuple, not `isinstance(exc, InteractionHandoffError)`: dispatching
# off the common base class would make any *future* subclass of it
# automatically swallowed the moment it is defined -- a fail-open default.
# Naming each swallowed type here means a new one has to be added on
# purpose, in the same line reviewers already read, before it starts being
# swallowed.
#
# PR-C2a note: InteractionRunPartitionMismatch's presence in this tuple is a
# deliberate override of this primitive's literal design default (which
# called for propagating it unchanged, the same as InteractionOwnerStateError
# below it). See interaction_handoff's docstring for the reasoning and the
# re-adjudication obligation this override carries forward.
# ---------------------------------------------------------------------------
# Data carriers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InteractionAnchor:
    """The resume anchor a caller is staging an interaction request against.

    This is not ``trace_event_staging.StagedCheckpointAnchor``. That type is
    PR-A's return value for the shell's pointer UPDATE and carries only
    ``checkpoint_event_id`` / ``trace_event_id``; PR-A's own review already
    confirmed this module does not consume it, and it is not reused or
    subclassed here.

    Represents only half of anchor resolution. The other half -- a database
    read against ``trace_events`` by primary key, layered into
    absence/unavailable/corrupt outcomes with a legacy-column fallback -- is
    a caller obligation that belongs to the batch that wires a production
    reader, because it is a DB read that must happen outside any savepoint
    this module opens. What this dataclass carries is the *result* of that
    read once the caller already has one in hand: the fields both
    ``stage_interaction_request`` and ``interaction_handoff`` validate for
    self-consistency (non-empty strings, the two closed vocabularies, and a
    real primary key) before ever building SQL from them. A caller that
    passes an anchor whose fields fail that validation gets
    ``InteractionAnchorCorrupt``, not a database round-trip.
    """

    trace_event_id: int
    resume_event_id: str
    resume_execution_id: str
    resume_run_partition: str
    resume_locator_format: str = _RESUME_LOCATOR_FORMAT
    resume_checkpoint_type: str = _RESUME_CHECKPOINT_TYPE


@dataclass(frozen=True)
class StagedInteractionRequest:
    """One interaction request row added to a caller-owned session.

    Meaningful only once the caller commits the transaction this row was
    staged into -- ``staged_db_id`` is the primary key SQLAlchemy assigned
    during this call's flush (whether from a fresh INSERT or from an
    existing row this call replayed), but it names a real, durable row only
    after that commit succeeds.

    ``created`` is ``True`` only on the clean-INSERT path. It is ``False``
    on every replay path: a pre-existing ``active`` row hit by the
    idempotency pre-read (step 4, before any reclaim or INSERT was even
    attempted) and a pre-existing ``active`` row hit by the post-conflict
    re-check (step 7, after this call's own INSERT lost a race). Both
    replay paths return the *other* request's row, not a new one --
    ``status`` and ``active_slot`` describe that row as it stood at the
    moment this call read it, which on the step-4 path may already be
    expired (see ``stage_interaction_request``'s docstring on why step 4
    does not consult ``expires_at``).
    """

    staged_db_id: int
    created: bool
    status: str
    active_slot: int | None


# ---------------------------------------------------------------------------
# Shared validation
# ---------------------------------------------------------------------------


def _validate_anchor_fields(anchor: InteractionAnchor) -> None:
    """The plain-Python half of anchor validation, shared by
    ``stage_interaction_request``'s own pre-INSERT check and
    ``interaction_handoff``'s ``__enter__`` precondition, so the two never
    drift into checking different things for the same dataclass.

    Every field checked here backs a real CHECK constraint on
    ``task_interaction_requests`` (see the audit in this PR's design
    record); a corrupt anchor caught here never reaches SQL.
    """

    if not anchor.resume_event_id:
        raise InteractionAnchorCorrupt("anchor.resume_event_id must not be empty")
    if not anchor.resume_execution_id:
        raise InteractionAnchorCorrupt("anchor.resume_execution_id must not be empty")
    if not anchor.resume_run_partition:
        raise InteractionAnchorCorrupt("anchor.resume_run_partition must not be empty")
    if anchor.resume_locator_format != _RESUME_LOCATOR_FORMAT:
        raise InteractionAnchorCorrupt(
            f"anchor.resume_locator_format must be {_RESUME_LOCATOR_FORMAT!r}, "
            f"got {anchor.resume_locator_format!r}"
        )
    if anchor.resume_checkpoint_type != _RESUME_CHECKPOINT_TYPE:
        raise InteractionAnchorCorrupt(
            f"anchor.resume_checkpoint_type must be {_RESUME_CHECKPOINT_TYPE!r}, "
            f"got {anchor.resume_checkpoint_type!r}"
        )
    if anchor.trace_event_id is None:
        raise InteractionAnchorCorrupt("anchor.trace_event_id must not be None")


def _validate_request_fields(
    *,
    run_id: str,
    anchor: InteractionAnchor,
    kind: str,
    protocol_version: int,
    origin: str,
    request_payload: Any,
    request_idempotency_key: str,
    expires_at: datetime,
    now: datetime,
) -> str:
    """Reject, in plain Python, every condition that could otherwise only be
    caught by a CHECK constraint on the INSERT. See the module docstring's
    misclassification-funnel paragraph for why this list must stay complete.

    Returns the normalized (stripped) idempotency key.
    """

    if kind not in _KIND_VOCABULARY:
        raise ValueError(
            f"kind must be one of {sorted(_KIND_VOCABULARY)}, got {kind!r}"
        )
    if protocol_version != _PROTOCOL_VERSION:
        raise ValueError(
            f"protocol_version must be {_PROTOCOL_VERSION}, got {protocol_version!r}"
        )
    if origin not in _ORIGIN_VOCABULARY:
        raise ValueError(
            f"origin must be one of {sorted(_ORIGIN_VOCABULARY)}, got {origin!r}"
        )
    normalized_key = request_idempotency_key.strip()
    if COMMAND_ID_PATTERN.fullmatch(normalized_key) is None:
        raise ValueError("request_idempotency_key must be 1-64 URL-safe characters")
    if request_payload is None:
        raise ValueError("request_payload must not be None")
    utc_offset = expires_at.utcoffset()
    if expires_at.tzinfo is None or utc_offset is None:
        raise ValueError("expires_at must be an aware UTC datetime")
    if utc_offset.total_seconds() != 0:
        raise ValueError("expires_at must be UTC (utcoffset must be zero)")
    if expires_at <= now:
        raise ValueError("expires_at must be after now")
    if not run_id:
        raise ValueError("run_id must not be empty")

    _validate_anchor_fields(anchor)

    if run_id != anchor.resume_run_partition:
        raise InteractionRunPartitionMismatch(
            f"run_id {run_id!r} does not match anchor.resume_run_partition "
            f"{anchor.resume_run_partition!r}"
        )

    return normalized_key


# ---------------------------------------------------------------------------
# Statement helpers
# ---------------------------------------------------------------------------


def _identity_lookup_stmt(
    *, task_id: int, run_id: str, request_idempotency_key: str
) -> sa.Select[Any]:
    """The identical Core SELECT used by both the idempotency pre-read (step
    4) and the post-conflict re-check (step 7). Written once so the two
    statements cannot drift apart at the edges of what they match."""

    return sa.select(
        TaskInteractionRequest.id,
        TaskInteractionRequest.status,
        TaskInteractionRequest.active_slot,
    ).where(
        TaskInteractionRequest.task_id == task_id,
        TaskInteractionRequest.run_id == run_id,
        TaskInteractionRequest.request_idempotency_key == request_idempotency_key,
    )


def _reclaim_stale_slot_stmt(*, task_id: int, run_id: str, now: datetime) -> sa.Update:
    """The reclaim UPDATE (step 5). Stays in the caller's outer transaction,
    not this function's inner savepoint -- it dies with the whole handoff on
    rollback, and the post-conflict re-check must not have undone it.

    Branch order is load-bearing: cross-run supersession takes priority over
    plain expiry when a row is both (``run_id <> :reclaim_run_id`` is
    checked before falling through to ``deadline_elapsed``).

    The one ``sa.text`` in this module, and it must stay parameterized via
    ``.bindparams`` -- see the module docstring's statement-discipline note.
    """

    return (
        sa.update(TaskInteractionRequest)
        .where(
            TaskInteractionRequest.task_id == task_id,
            TaskInteractionRequest.active_slot.isnot(None),
            sa.or_(
                TaskInteractionRequest.expires_at <= now,
                TaskInteractionRequest.run_id != run_id,
            ),
        )
        .values(
            status="terminated",
            terminal_reason=sa.text(
                "CASE WHEN run_id <> :reclaim_run_id THEN 'run_superseded' "
                "ELSE 'deadline_elapsed' END"
            ).bindparams(reclaim_run_id=run_id),
            active_slot=None,
            terminated_at=now,
            updated_at=now,
        )
    )


# ---------------------------------------------------------------------------
# The staging primitive
# ---------------------------------------------------------------------------


def stage_interaction_request(
    db: Session,
    *,
    task_id: int,
    run_id: str,
    anchor: InteractionAnchor,
    kind: str,
    protocol_version: int,
    origin: str,
    request_payload: Any,
    request_idempotency_key: str,
    expires_at: datetime,
    now: datetime,
) -> StagedInteractionRequest:
    """Add one active interaction request row to ``db``, without ending its
    transaction. See the module docstring for what the caller still owns.

    Statement order is load-bearing:

    1. Reject, in plain Python, everything the INSERT could otherwise only
       be rejected for by a CHECK constraint (see the module docstring).
       Nothing below this point issues SQL until this block has returned
       cleanly.
    2. Flush the caller's own pending writes first, so a conflict there is
       reported as ``InteractionOwnerStateError`` rather than folded into
       the ``IntegrityError`` this function raises for a slot conflict on
       its own INSERT below.
    3. Read for an existing row at this identity -- ``(task_id, run_id,
       request_idempotency_key)`` -- before touching anything else. A hit
       whose ``status`` is ``active`` is replayed as-is (``created=False``);
       this branch does not consult ``expires_at``, so a replayed row may
       already be expired -- callers must not assume otherwise (see
       ``StagedInteractionRequest``'s docstring). A hit whose ``status`` is
       ``answered`` or ``terminated`` raises ``InteractionRequestClosed``:
       identity is scoped by ``run_id``, so a key reclaimed to
       ``terminated`` earlier in *this same run* still raises here on reuse.
    4. Reclaim: retire this task's stale or run-superseded active row, if
       any, in the same outer transaction as everything else -- not inside
       the inner savepoint opened next, so a post-conflict re-check below
       cannot undo it.
    5. Insert the new active row inside a savepoint this function opens and
       owns (``db.begin_nested()``), then flush. A savepoint shared with a
       caller (or with ``interaction_handoff``'s own savepoint) does not
       work here: ``ResourceClosedError: This transaction is closed`` fires
       on both backends, on both the exception exit path and the successful
       REPLAY-after-conflict exit path, because a savepoint that something
       else already rolled back cannot be committed or rolled back again.
       Measured directly against this tree; see this PR's mutation test for
       the primitive's own savepoint (M-1), which reproduces the same
       failure by making that same mistake on purpose.
    6. On ``IntegrityError``, roll back only this inner savepoint and
       re-check identity with the same SELECT step 3 used. A hit whose
       ``status`` is ``active`` is a REPLAY-after-conflict: another session
       won the race between this call's step-3 read and its own INSERT, so
       the row this call now owns is that winner's row, not a fresh one.
       No hit (or a non-active hit) raises ``InteractionSlotTaken`` --
       which, because this re-check cannot distinguish a genuine slot
       conflict from a programming error that also violates some other
       CHECK, is why step 1's validation list must be complete.

    This function's reclaim UPDATE assumes the caller has already proven it
    holds the task's current attempt; bypassing that precondition to call it
    directly lets a dead run silently displace a live run's question.
    """

    resolved_task_id = int(task_id)
    normalized_key = _validate_request_fields(
        run_id=run_id,
        anchor=anchor,
        kind=kind,
        protocol_version=protocol_version,
        origin=origin,
        request_payload=request_payload,
        request_idempotency_key=request_idempotency_key,
        expires_at=expires_at,
        now=now,
    )

    try:
        db.flush()
    except IntegrityError as exc:
        raise InteractionOwnerStateError(
            "caller's pending writes failed to flush before interaction staging"
        ) from exc

    existing = db.execute(
        _identity_lookup_stmt(
            task_id=resolved_task_id,
            run_id=run_id,
            request_idempotency_key=normalized_key,
        )
    ).first()
    if existing is not None:
        if existing.status == "active":
            return StagedInteractionRequest(
                staged_db_id=int(existing.id),
                created=False,
                status=existing.status,
                active_slot=existing.active_slot,
            )
        raise InteractionRequestClosed(
            f"request {normalized_key!r} on task {resolved_task_id} run "
            f"{run_id!r} is already {existing.status}"
        )

    db.execute(
        _reclaim_stale_slot_stmt(task_id=resolved_task_id, run_id=run_id, now=now)
    )

    row = {
        "task_id": resolved_task_id,
        "run_id": run_id,
        "kind": kind,
        "protocol_version": protocol_version,
        "status": "active",
        "active_slot": 1,
        "origin": origin,
        "request_payload": request_payload,
        "response_payload": None,
        "request_idempotency_key": normalized_key,
        "resume_trace_event_id": anchor.trace_event_id,
        "resume_event_id": anchor.resume_event_id,
        "resume_execution_id": anchor.resume_execution_id,
        "resume_locator_format": anchor.resume_locator_format,
        "resume_checkpoint_type": anchor.resume_checkpoint_type,
        "resume_run_partition": anchor.resume_run_partition,
        "responder_user_id": None,
        "responder_identity": None,
        "terminal_reason": None,
        "expires_at": expires_at,
        "responded_at": None,
        "terminated_at": None,
    }

    inner = db.begin_nested()
    try:
        new_row = TaskInteractionRequest(**row)
        db.add(new_row)
        db.flush()
    except IntegrityError:
        inner.rollback()
        again = db.execute(
            _identity_lookup_stmt(
                task_id=resolved_task_id,
                run_id=run_id,
                request_idempotency_key=normalized_key,
            )
        ).first()
        if again is not None and again.status == "active":
            return StagedInteractionRequest(
                staged_db_id=int(again.id),
                created=False,
                status=again.status,
                active_slot=again.active_slot,
            )
        raise InteractionSlotTaken(
            f"task {resolved_task_id} already has an active interaction "
            "request in a different slot"
        )
    else:
        inner.commit()
        return StagedInteractionRequest(
            staged_db_id=int(new_row.id),
            created=True,
            status="active",
            active_slot=1,
        )
