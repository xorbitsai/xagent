"""Stage one blocking task interaction request into a session the caller
already owns, and hand it off under a savepoint this module controls.

Two production entry points, sharing one module because they share every
exception type and one nesting invariant that must never be split across a
file boundary:

* ``stage_interaction_request`` -- plain-Python validation, the caller's own
  flush, a keyed idempotency pre-read, the reclaim UPDATE that retires a
  stale or superseded active row, and the active-row INSERT under a
  savepoint this function owns.
* ``interaction_handoff`` -- a context manager that opens the outer
  savepoint the whole handoff lives in, asserts the caller's lease still
  owns the task row and that the anchor it was given is self-consistent,
  and degrades on a closed set of expected failures instead of losing the
  caller's turn.

Caller obligations, because none of them happen here:

* Both entry points join the ``Session`` passed in; neither ever calls
  ``db.commit()`` or the outer ``db.rollback()``. ``stage_interaction_request``
  manages only its own inner savepoint (see below); ``interaction_handoff``
  manages only the outer savepoint it opens in ``__enter__``. The caller
  owns the transaction and decides when to commit or roll it back.
* Neither one writes any *data* column of ``tasks``, and neither notifies
  or prunes anything. The one exception is structural, not data: on
  SQLite, ``interaction_handoff`` issues a single self-assigning,
  zero-row ``UPDATE tasks SET id = id WHERE id = -1`` immediately before
  opening its own outer savepoint (see that function's docstring for why).
  ``id = -1`` can never match a row an autoincrement primary key minted,
  so this can never touch a real row's data, and it is skipped entirely
  on PostgreSQL, which does not need it. A degraded handoff still leaves
  whatever the caller was about to persist for its own reasons (a stale
  attempt continuing to write its own task row, for instance) untouched --
  that is a fact the three eventual finalizer call sites must each account
  for, not something this module can prevent. Those three finalizers are
  the complete writer set this module assumes: a fourth, legacy release
  path (``AgentServiceManager.execute_task`` in ``web/api/chat.py``) also
  formally writes ``TaskStatus.WAITING_FOR_USER`` on a task row when
  ``manage_task_lease`` is true, but every production call site
  (``websocket.py``, and the Feishu/Slack/Telegram bot channels) passes
  ``manage_task_lease=False`` -- that write path is confirmed unreachable
  today, not merely unlikely. A caller added later that reaches it with
  ``manage_task_lease=True`` would make it a fourth writer this module's
  finalizer accounting does not yet account for.
* If ``stage_interaction_request`` raises ``InteractionOwnerStateError``,
  the session is left mid-transaction with no savepoint of its own to roll
  back to (the failure happened before this function opened one) -- the
  caller must roll back the whole transaction before issuing another
  statement on it. Every other exception this module raises during a call
  wrapped by ``interaction_handoff`` is contained by that context manager's
  own savepoint; called outside it, the same rollback obligation applies.

Zero production callers as of this module's introduction: a static test
(``tests/web/services/test_interaction_staging_production_gate.py``) asserts
that no production module imports or calls either entry point. See that
test's docstring for the removal condition.

Every rejection the database's 23 CHECK constraints could raise on the
INSERT is rejected in plain Python first, inside
``stage_interaction_request``'s validation block, because the post-conflict
re-check (step 7 below) classifies any ``IntegrityError`` for which no row
exists at the caller's own identity as a slot conflict --
``InteractionSlotTaken`` (a re-check hit that does exist there, but is
already ``answered`` or ``terminated``, raises ``InteractionRequestClosed``
instead -- see that exception's docstring -- because the identity row itself
explains the conflict). A CHECK-violating programming error (an
out-of-vocabulary ``origin``, a non-positive TTL, a ``None`` payload past
``JSON(none_as_null=True))``) never inserted a row in the first place, so it
always lands on the no-identity-row branch, indistinguishable from a real
slot conflict. The backstop is the database's constraint set, not the
primary defense; the primary defense is this validation block, and it must
reject everything the INSERT could reject except the two
concurrent-parent-deletion cells documented below, before the INSERT is
ever issued. Adding a column CHECK to ``TaskInteractionRequest`` without
adding the matching Python check here reopens that misclassification
funnel by one more case.

Three mechanisms, not one. A CHECK violation surfaces as an
``IntegrityError`` and lands in the misclassification funnel above. A
column-length violation surfaces as a ``DataError`` on PostgreSQL -- not an
``IntegrityError``, so the conflict classifier never sees it, and not in
``_SWALLOWED``, so it would escape the handoff entirely -- and is silently
accepted on SQLite, which does not enforce ``VARCHAR`` length. That
asymmetry is why the length caps are derived from the model's own column
types rather than written as literals. A value ``request_payload`` cannot
serialize at all is the third: SQLAlchemy's JSON type serializes at bind
time, inside the INSERT, well past this validation block, and a value that
fails there raises ``StatementError`` on both backends -- not
``IntegrityError``, so it would also escape the conflict classifier and
``_SWALLOWED`` the same way a length violation does. ``request_payload`` is
probed with ``json.dumps(..., allow_nan=False)`` in this validation block
for exactly this reason, before the INSERT is ever issued (see that check's
own comment for why ``allow_nan=False`` is load-bearing: the two backends
diverge on ``NaN``/``Infinity`` in a way plain ``json.dumps`` would not
catch).

Two cells this validation block cannot close, by construction: the parent
``tasks`` row (``fk_task_interaction_requests_task_id``) or the anchor's
``trace_events`` row (``fk_task_interaction_requests_resume_trace_event_id``)
being deleted concurrently, some time between whatever read gave the
caller its ``task_id`` / the anchor's ``trace_event_id`` and this call's
own INSERT. Neither has a Python precondition here -- nothing in this
function's own arguments could detect a row disappearing out from under
it -- and both surface as a foreign-key ``IntegrityError`` that step 7's
re-check, finding no matching identity row, classifies as
``InteractionSlotTaken`` along with everything else in the
misclassification funnel above. Closing that gap is not this validation
block's job: it belongs to whatever serializes deletion against staging
(the task-deletion path's own locking, and the trace-event pruning job's
own protection set for rows a live anchor could still name), not to a
plain-Python check inside this primitive.

Concurrency note (inherited, not introduced here): on PostgreSQL, an
``IntegrityError`` poisons the rest of the transaction
(``InFailedSqlTransaction``) until something rolls back at least to the
savepoint that was open when it fired; SQLite instead lets the session keep
issuing statements. ``stage_interaction_request``'s own inner
``db.begin_nested()`` around the INSERT is what makes the post-conflict
re-check possible on both backends -- see the function's docstring for why
a savepoint shared with the caller (or with ``interaction_handoff``'s own
savepoint) does not work.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterator

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..models.task import Task
from ..models.task_interaction import TaskInteractionRequest
from .ops_signals import (
    INTERACTION_HANDOFF_DEGRADED,
    INTERACTION_RUN_PARTITION_MISMATCH_DEGRADED,
    register_degradation,
)
from .task_command_transport import COMMAND_ID_PATTERN
from .task_lease_service import TaskLease

logger = logging.getLogger(__name__)

_KIND_VOCABULARY = frozenset({"clarification"})
_ORIGIN_VOCABULARY = frozenset(
    {"internal", "sdk", "a2a", "trigger", "widget", "shared_link"}
)
_RESUME_LOCATOR_FORMAT = "trace_event_pk_v1"
_RESUME_CHECKPOINT_TYPE = "agent_execution_checkpoint"
_PROTOCOL_VERSION = 1

_MAX_LENGTHS: dict[str, int] = {
    # Derived from the model so they cannot drift from the schema. These
    # are column types, not CHECK constraints: an over-length value is
    # silently stored on SQLite and raises DataError -- not IntegrityError
    # -- on PostgreSQL, so it never enters the conflict classifier and
    # never reaches the swallowed set.
    name: TaskInteractionRequest.__table__.c[name].type.length
    for name in (
        "run_id",
        "resume_event_id",
        "resume_execution_id",
        "resume_run_partition",
    )
}


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

    Raised only when the post-conflict re-check (step 7) finds no row at all
    at the caller's own ``(task_id, run_id, request_idempotency_key)`` after
    an ``IntegrityError`` -- there is no identity row to explain the
    conflict, so it must have been the active-slot unique. A re-check hit
    whose ``status`` is ``answered`` or ``terminated`` raises
    ``InteractionRequestClosed`` instead (see that exception's docstring):
    it is the same identity key, just already closed, not a slot race.
    Because a no-hit re-check still cannot distinguish a real slot conflict
    from any other ``IntegrityError`` the INSERT could have raised, this
    exception is also what a programming error that skipped the
    plain-Python validation block would surface as. See the module
    docstring's misclassification-funnel paragraph.
    """


class InteractionRequestClosed(InteractionHandoffError):
    """This identity key already names a terminal row.

    Raised when ``(task_id, run_id, request_idempotency_key)`` names a row
    whose ``status`` is ``answered`` or ``terminated`` rather than
    ``active`` -- both by the idempotency pre-read (step 4, before this
    call's own INSERT is ever attempted) and by the post-conflict re-check
    (step 7, when this call's own INSERT collided with a row that turns out
    to already be closed rather than merely active elsewhere). Both reads
    share ``_identity_lookup_stmt``, so the two call sites classify the same
    row state identically. Identity is scoped by ``run_id``: a key that was
    reclaimed to ``terminated`` earlier in the *same* run still raises this
    on reuse (there is no cross-run leakage to guard against, because a
    different run never shares this row at all).
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

    Raised by ``_InteractionHandoff.stage()``, at the start of the call,
    before any staging SQL is issued -- not by ``interaction_handoff``'s
    ``__enter__`` (see ``stage()``'s own docstring for why this
    precondition has to live there). Raised when
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

    The handoff layer deliberately swallows this exception -- see ``_SWALLOWED`` and
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


class InteractionHandoffMisuse(InteractionHandoffError):
    """Raised for either of two ways a caller can misuse the ``with``
    block, both caught only once the block's body has finished running:

    * ``stage()`` was called more than once inside one handoff. The schema
      allows exactly one active interaction row per task, so a second
      ``stage()`` in the same handoff can only fail -- and the handler that
      would swallow it rolls back the outer savepoint, discarding the first
      call's row while reporting success.
    * The caller committed or rolled back the session from inside the
      ``with`` block -- violating ``interaction_handoff``'s "no I/O in
      between" obligation -- after ``stage()`` had already succeeded. Both
      operations end the whole transaction, deactivating this context
      manager's own outer savepoint along with it; by the time the block
      exits normally, that savepoint no longer exists to commit. Raising
      here instead of silently skipping the now-impossible commit matters
      because skipping would report success for a row whose containment is
      gone -- the caller's own commit already decided that row's fate one
      way or the other, and this context manager has no way left to tell
      which.

    Deliberately absent from ``_SWALLOWED`` in both cases: these are caller
    bugs, classified the same way ``InteractionOwnerStateError`` is --
    propagate, do not degrade.
    """


class InteractionOriginUnknown(InteractionHandoffError):
    """``task.source`` names a value outside the frozen origin vocabulary.

    ``origin`` is a frozen copy of ``task.source`` and an audit column: it
    records which source surface an interaction was raised on behalf of.
    Substituting ``"internal"`` for an unrecognised value would write a
    false provenance into that column, so this degrades instead -- the row
    is not written, and the drift is reported. Unlike the ``ValueError``
    ``stage_interaction_request`` raises for a caller-supplied ``origin``,
    this names a value read out of a persisted row: data drift, not a
    programming error.
    """


# Explicit tuple, not `isinstance(exc, InteractionHandoffError)`: dispatching
# off the common base class would make any *future* subclass of it
# automatically swallowed the moment it is defined -- a fail-open default.
# Naming each swallowed type here means a new one has to be added on
# purpose, in the same line reviewers already read, before it starts being
# swallowed.
#
# Note: InteractionRunPartitionMismatch's presence in this tuple is a
# deliberate override of this primitive's literal design default (which
# called for propagating it unchanged, the same as InteractionOwnerStateError
# below it). See interaction_handoff's docstring for the reasoning and the
# re-adjudication obligation this override carries forward.
_SWALLOWED: tuple[type[BaseException], ...] = (
    InteractionSlotTaken,
    InteractionRequestClosed,
    InteractionAnchorCorrupt,
    InteractionAttemptMismatch,
    InteractionRunPartitionMismatch,
    InteractionOriginUnknown,
)

# Which ops_signals name a given swallowed exception type registers.
# InteractionRunPartitionMismatch gets its own signal, distinct from the
# other five's shared INTERACTION_HANDOFF_DEGRADED, precisely because it is
# the one override that cannot be validated at this layer -- see
# interaction_handoff's docstring. Keeping it separately addressable in
# /health lets the change that wires the first production caller tell "a
# run-partition anchor went stale" apart from every other reason a handoff
# degraded, without having to reparse log lines to do it.
_DEGRADATION_SIGNALS: dict[type[BaseException], str] = {
    InteractionSlotTaken: INTERACTION_HANDOFF_DEGRADED,
    InteractionRequestClosed: INTERACTION_HANDOFF_DEGRADED,
    InteractionAnchorCorrupt: INTERACTION_HANDOFF_DEGRADED,
    InteractionAttemptMismatch: INTERACTION_HANDOFF_DEGRADED,
    InteractionRunPartitionMismatch: INTERACTION_RUN_PARTITION_MISMATCH_DEGRADED,
    InteractionOriginUnknown: INTERACTION_HANDOFF_DEGRADED,
}


# ---------------------------------------------------------------------------
# Data carriers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InteractionAnchor:
    """The resume anchor a caller is staging an interaction request against.

    This is not ``trace_event_staging.StagedCheckpointAnchor``. That type
    exists for the trace shell's pointer UPDATE and carries only
    ``checkpoint_event_id`` / ``trace_event_id``; this module does not
    consume it, and it is not reused or subclassed here.

    Represents only half of anchor resolution. The other half -- a database
    read against ``trace_events`` by primary key, layered into
    absence/unavailable/corrupt outcomes with a legacy-column fallback -- is
    a caller obligation that belongs to the change that wires a production
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
    ``_InteractionHandoff.stage()``'s own precondition check -- both now run
    at the start of their respective calls, not in ``__enter__`` -- so the
    two never drift into checking different things for the same dataclass.

    Every field checked here backs a real CHECK constraint on
    ``task_interaction_requests``; a corrupt anchor caught here never
    reaches SQL.
    """

    if not isinstance(anchor, InteractionAnchor):
        raise InteractionAnchorCorrupt(
            f"anchor must be an InteractionAnchor, got {type(anchor).__name__}"
        )
    for field_name in (
        "resume_event_id",
        "resume_execution_id",
        "resume_run_partition",
    ):
        value = getattr(anchor, field_name)
        if not value:
            raise InteractionAnchorCorrupt(f"anchor.{field_name} must not be empty")
        if len(value) > _MAX_LENGTHS[field_name]:
            raise InteractionAnchorCorrupt(
                f"{field_name} exceeds the column limit "
                f"{_MAX_LENGTHS[field_name]} (got {len(value)}): a locator "
                "this long cannot have come from a valid trace row"
            )
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
    if isinstance(anchor.trace_event_id, bool) or not isinstance(
        anchor.trace_event_id, int
    ):
        raise InteractionAnchorCorrupt(
            "anchor.trace_event_id must be an integer row id, got "
            f"{anchor.trace_event_id!r}"
        )
    if anchor.trace_event_id <= 0:
        raise InteractionAnchorCorrupt(
            "anchor.trace_event_id must be a positive row id, got "
            f"{anchor.trace_event_id!r}"
        )


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

    Row 13 of that list: ``now`` must be validated as an aware UTC datetime
    the same way ``expires_at`` is (row 6) -- not normalized, only rejected
    if it fails. ``now`` has no CHECK constraint of its own to back this up
    (unlike every other row here), but it is not a free pass: the reclaim
    UPDATE (``_reclaim_stale_slot_stmt``) persists this exact value into
    ``terminated_at`` and ``updated_at`` on every row it reclaims, so a
    naive or non-UTC ``now`` would otherwise reach SQL and get silently
    mis-stored the same way an unvalidated ``expires_at`` would (see
    ``TaskInteractionRequest``'s own docstring on the aware-UTC obligation
    for that column).

    Returns the normalized (stripped) idempotency key.
    """

    if not isinstance(kind, str):
        raise ValueError(f"kind must be a str, got {type(kind).__name__}")
    if kind not in _KIND_VOCABULARY:
        raise ValueError(
            f"kind must be one of {sorted(_KIND_VOCABULARY)}, got {kind!r}"
        )
    if (
        not isinstance(protocol_version, int)
        or isinstance(protocol_version, bool)
        or protocol_version != _PROTOCOL_VERSION
    ):
        raise ValueError(
            f"protocol_version must be {_PROTOCOL_VERSION}, got {protocol_version!r}"
        )
    if not isinstance(origin, str):
        raise ValueError(f"origin must be a str, got {type(origin).__name__}")
    if origin not in _ORIGIN_VOCABULARY:
        raise ValueError(
            f"origin must be one of {sorted(_ORIGIN_VOCABULARY)}, got {origin!r}"
        )
    if not isinstance(request_idempotency_key, str):
        raise ValueError(
            "request_idempotency_key must be a str, got "
            f"{type(request_idempotency_key).__name__}"
        )
    normalized_key = request_idempotency_key.strip()
    if COMMAND_ID_PATTERN.fullmatch(normalized_key) is None:
        raise ValueError("request_idempotency_key must be 1-64 URL-safe characters")
    if request_payload is None:
        raise ValueError("request_payload must not be None")
    # A value that cannot be JSON-serialized at all never reaches this
    # block's other checks -- it would sail through every one of them (it is
    # not None, and nothing else here inspects its shape) and fail instead
    # at bind time, inside the INSERT, as a StatementError on both backends.
    # StatementError is not IntegrityError, so the post-conflict re-check
    # would never see it, and it is not in _SWALLOWED either, so it would
    # escape interaction_handoff entirely instead of degrading like every
    # other expected failure this module knows how to name. Probing here,
    # before the INSERT is ever issued, turns that crash into the same
    # ValueError every other row-13-and-earlier violation raises.
    #
    # allow_nan=False is load-bearing, not a strictness knob: the default
    # json.dumps happily renders float('nan') / float('inf') as the bare
    # (non-JSON) tokens NaN / Infinity, which round-trips through Python's
    # own json.loads but is not valid JSON text. The two backends then
    # diverge on what a real INSERT does with it -- SQLite has no native
    # JSON type and stores the column as TEXT, so it stores that invalid
    # JSON verbatim with no complaint at all; PostgreSQL's jsonb parser
    # rejects it, raising DataError, not StatementError. Passing
    # allow_nan=False makes this probe raise ValueError for NaN/Infinity
    # before either of those divergent, backend-specific outcomes is ever
    # reached, so this validation block rejects the payload the same way
    # regardless of which backend the caller happens to be running against.
    #
    # A payload that is valid JSON but serializes twice (e.g. a dict whose
    # value is itself already a JSON string) is accepted here: this probe
    # only checks that request_payload itself is JSON-serializable, not that
    # its values are not already-serialized JSON text -- double-serialization
    # is a caller-shape question this primitive does not adjudicate, and it
    # is well within the size this table's clarification payloads are
    # expected to carry.
    #
    # If this engine is ever configured with a custom json_serializer (it is
    # not, today -- see xagent/db/sqlite.py and the engine construction this
    # module's callers use), this probe must derive from that serializer
    # rather than calling json.dumps directly, or the two could accept
    # different payloads.
    try:
        json.dumps(request_payload, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"request_payload is not JSON-serializable: {exc}") from exc
    if not isinstance(expires_at, datetime):
        raise ValueError(
            f"expires_at must be a datetime, got {type(expires_at).__name__}"
        )
    utc_offset = expires_at.utcoffset()
    if expires_at.tzinfo is None or utc_offset is None:
        raise ValueError("expires_at must be an aware UTC datetime")
    if utc_offset.total_seconds() != 0:
        raise ValueError("expires_at must be UTC (utcoffset must be zero)")
    if not isinstance(now, datetime):
        raise ValueError(f"now must be a datetime, got {type(now).__name__}")
    now_utc_offset = now.utcoffset()
    if now.tzinfo is None or now_utc_offset is None:
        raise ValueError("now must be an aware UTC datetime")
    if now_utc_offset.total_seconds() != 0:
        raise ValueError("now must be UTC (utcoffset must be zero)")
    if expires_at <= now:
        raise ValueError("expires_at must be after now")
    if not isinstance(run_id, str):
        raise ValueError(f"run_id must be a str, got {type(run_id).__name__}")
    if not run_id:
        raise ValueError("run_id must not be empty")
    if len(run_id) > _MAX_LENGTHS["run_id"]:
        raise ValueError(
            f"run_id must be at most {_MAX_LENGTHS['run_id']} characters, "
            f"got {len(run_id)}"
        )

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

    ``terminal_reason`` is built with ``sa.case``, not ``sa.text`` -- Core
    compiles and parameterizes ``run_id`` on both backends without any raw
    SQL fragment, verified equivalent to the earlier hand-written ``CASE
    WHEN run_id <> :reclaim_run_id THEN 'run_superseded' ELSE
    'deadline_elapsed' END``. The only remaining ``sa.text`` usage in this
    module is ``interaction_handoff``'s SQLite-only savepoint guard, which
    needs no parameter because its predicate is a hardcoded constant, not
    caller input.

    This UPDATE is also the shape the next status-writing statement on this
    table needs to follow: it writes ``status`` and ``terminated_at``
    together, because
    ``ck_task_interaction_requests_terminated_at_pairs_status`` enforces
    ``(status = 'terminated') = (terminated_at IS NOT NULL)`` on every row
    -- setting one without the other fails at the database, not just in
    review. A future answering statement (writing ``status='answered'``) is
    bound by the same discipline in the other direction: it must write
    neither ``terminated_at`` nor ``terminal_reason``, since neither column
    pairs with ``answered`` in any CHECK on this table.
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
            terminal_reason=sa.case(
                (TaskInteractionRequest.run_id != run_id, "run_superseded"),
                else_="deadline_elapsed",
            ),
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
       Measured directly against this tree; pinned by the primitive's
       savepoint mutation test, which reproduces the same failure by making
       that same mistake on purpose.
    6. On ``IntegrityError``, roll back only this inner savepoint and
       re-check identity with the same SELECT step 3 used. A hit whose
       ``status`` is ``active`` is a REPLAY-after-conflict: another session
       won the race between this call's step-3 read and its own INSERT, so
       the row this call now owns is that winner's row, not a fresh one. A
       hit whose ``status`` is ``answered`` or ``terminated`` raises
       ``InteractionRequestClosed`` -- the same classification step 3 gives
       that state, since both reads share ``_identity_lookup_stmt``: the
       identity row this call collided with had already closed by the time
       the INSERT fired, which is a fact about that row's own lifecycle, not
       an active-slot race. No hit at all raises ``InteractionSlotTaken`` --
       the INSERT collided with some other row's active slot instead, and
       there is no identity row here to explain the conflict any other way
       -- which, because this re-check cannot distinguish a genuine slot
       conflict from a programming error that also violates some other
       CHECK, is why step 1's validation list must be complete.

    The post-conflict re-check assumes READ COMMITTED. It re-reads the
    identity key after rolling back the inner savepoint, and needs a fresh
    snapshot to see the row the winning session committed between this
    call's pre-read and its own INSERT. Under REPEATABLE READ or
    SERIALIZABLE the re-read reuses this transaction's original snapshot,
    does not see that row, and classifies a legitimate replay as
    ``InteractionSlotTaken`` -- measured on PostgreSQL 16 at both levels.
    That is a degradation, not a corruption: ``InteractionSlotTaken`` is
    swallowed, so the caller's turn survives and the question is lost.
    READ COMMITTED is PostgreSQL's default and this codebase sets no
    ``isolation_level`` on its engine; the change that wires the first
    production caller should assert that rather than continue to assume it.

    This function's reclaim UPDATE assumes the caller has already proven it
    holds the task's current attempt; bypassing that precondition to call it
    directly lets a dead run silently displace a live run's question -- the
    live run's own next staging attempt then surfaces as
    ``InteractionSlotTaken`` or ``InteractionRequestClosed``, not as the
    precondition violation that actually caused it.
    """

    if isinstance(task_id, bool) or not isinstance(task_id, int):
        raise ValueError(f"task_id must be a positive int, got {task_id!r}")
    if task_id <= 0:
        raise ValueError(f"task_id must be a positive int, got {task_id!r}")
    resolved_task_id = task_id
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
        _reclaim_stale_slot_stmt(task_id=resolved_task_id, run_id=run_id, now=now),
        execution_options={"synchronize_session": False},
    )

    row = {
        "task_id": resolved_task_id,
        "created_at": now,
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
        if again is not None:
            if again.status == "active":
                return StagedInteractionRequest(
                    staged_db_id=int(again.id),
                    created=False,
                    status=again.status,
                    active_slot=again.active_slot,
                )
            raise InteractionRequestClosed(
                f"request {normalized_key!r} on task {resolved_task_id} run "
                f"{run_id!r} is already {again.status}"
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
    finally:
        # A no-op on both handled paths above (the except branch's own
        # inner.rollback() and the else branch's inner.commit() have already
        # deactivated the savepoint by the time this runs). What it actually
        # guards is any exception this try block does not otherwise catch --
        # IntegrityError is the only one classified above, so anything else
        # db.flush() or the constructor could raise (a StatementError from a
        # bind-time serialization failure, for instance) would previously
        # propagate straight out with this savepoint still open, leaking it
        # into whatever the caller does next. Closing it here regardless of
        # which path was taken keeps this function's own savepoint scoped to
        # this function on every exit, not just the two it names explicitly.
        if inner.is_active:
            inner.rollback()


# ---------------------------------------------------------------------------
# The handoff context manager
# ---------------------------------------------------------------------------


class _InteractionHandoff:
    """The object ``interaction_handoff`` yields. Constructed once per
    ``with`` block; ``stage`` may be called exactly once (zero calls is
    legal: a caller may decide not to ask).
    """

    def __init__(
        self,
        db: Session,
        lease: TaskLease,
        *,
        task: Task,
        anchor: InteractionAnchor,
        now: datetime,
    ) -> None:
        self.db = db
        self.lease = lease
        self.task = task
        self.anchor = anchor
        self.now = now
        self._staged = False

    def _assert_current_attempt(self) -> None:
        """``lease.attempt_id is None`` is a fail-open sentinel (a
        pre-attempt-column lease, or the permanent-``None`` ambient snapshot
        ``_task_lease_snapshot`` builds in ``websocket.py``) and must be
        treated as "cannot prove attempt identity", not as "matches" --
        skipping this assertion, never failing it. See
        ``TaskLease.attempt_id``'s docstring (``task_lease_service.py``) for
        the two distinct sources of that ``None``.

        This is a plain Python object comparison, not a SQL expression: the
        `` == None`` -> ``IS NULL`` folding that affects a SQLAlchemy
        column comparison compiled to SQL does not apply here.

        This reads ``self.task.lease_attempt_id`` from whatever snapshot of
        the task row the caller already loaded. Whether that attribute
        access itself touches the database is conditional on the object's
        own session-bound state, not a fixed property of this line: if
        SQLAlchemy has expired ``task`` (its default behavior after every
        commit) the access issues its own lazy-load ``SELECT`` before
        returning a value; if ``task`` is detached from any session
        entirely it raises ``DetachedInstanceError`` instead -- neither is
        swallowed here. When the attribute is already loaded and unexpired,
        this line takes no lock and issues no SQL of its own. Whether the
        comparison is safe
        against a concurrent attempt change depends on the backend, not on
        this line: on PostgreSQL, a caller that loaded ``task`` with its own
        ``SELECT ... FOR UPDATE`` genuinely blocks a concurrent writer for
        as long as that lock is held. On SQLite, SQLAlchemy silently drops
        ``with_for_update()`` -- the dialect does not support it -- so no
        row lock is ever actually taken there (the same fact
        ``get_next_expired_task_lease_candidate_for_update`` in
        ``task_lease_service.py`` documents for its own row-lock read).
        What keeps this comparison's window closed on SQLite instead is
        single-writer semantics: SQLite's database-wide writer lock
        serializes every write regardless of which row it targets. That
        makes this a TOCTOU (time-of-check-to-time-of-use) gap that happens
        to stay closed because nothing else can be writing concurrently,
        not a genuine row-level lock -- a caller that assumes real row
        locking here is assuming something SQLite does not provide.
        """

        if (
            self.lease.attempt_id is not None
            and self.task.lease_attempt_id != self.lease.attempt_id
        ):
            raise InteractionAttemptMismatch(
                f"task {self.task.id}'s current attempt "
                f"({self.task.lease_attempt_id!r}) does not match this "
                f"lease's attempt ({self.lease.attempt_id!r})"
            )

    def _assert_anchor_consistent(self) -> None:
        """Re-validates the same structural rules
        ``stage_interaction_request`` validates before its own INSERT (see
        ``_validate_anchor_fields``), so a corrupt anchor is caught before
        any staging SQL is issued, even on the very first ``stage`` call."""

        _validate_anchor_fields(self.anchor)

    def stage(
        self,
        *,
        kind: str,
        protocol_version: int,
        request_payload: Any,
        request_idempotency_key: str,
        expires_at: datetime,
    ) -> StagedInteractionRequest:
        """Stage one interaction request for this handoff's task, run, and
        anchor. ``run_id`` is always this handoff's ``lease.run_id`` --
        callers cannot stage a request under a different run through this
        method, which is what keeps the reclaim UPDATE's cross-run
        supersession branch meaningful. ``origin`` is always
        ``task.source or "internal"``, computed here rather than left to the
        caller (see ``TaskInteractionRequest.origin``'s column comment).

        The attempt and anchor assertions run first, in that order, before
        any statement this call could issue -- see ``interaction_handoff``'s
        docstring for why they live here rather than in ``__enter__``.
        """

        if self._staged:
            raise InteractionHandoffMisuse(
                "interaction_handoff supports exactly one stage() call per "
                f"handoff; task {self.task.id} run {self.lease.run_id} "
                "attempted a second"
            )
        self._staged = True
        if self.lease.task_id != int(self.task.id):
            raise ValueError(
                f"lease names task {self.lease.task_id} but this handoff was "
                f"given task {int(self.task.id)}"
            )
        self._assert_current_attempt()
        self._assert_anchor_consistent()
        if self.lease.run_id is None:
            raise ValueError(
                "lease has no run_id; cannot stage an interaction request without one"
            )
        origin = str(self.task.source) if self.task.source else "internal"
        if origin not in _ORIGIN_VOCABULARY:
            raise InteractionOriginUnknown(
                f"task {self.task.id}'s source {origin!r} is outside the "
                f"origin vocabulary {sorted(_ORIGIN_VOCABULARY)}"
            )
        return stage_interaction_request(
            self.db,
            task_id=int(self.task.id),
            run_id=self.lease.run_id,
            anchor=self.anchor,
            kind=kind,
            protocol_version=protocol_version,
            origin=origin,
            request_payload=request_payload,
            request_idempotency_key=request_idempotency_key,
            expires_at=expires_at,
            now=self.now,
        )


@contextmanager
def interaction_handoff(
    db: Session,
    lease: TaskLease,
    *,
    task: Task,
    anchor: InteractionAnchor,
    now: datetime,
) -> Iterator[_InteractionHandoff]:
    """Open the savepoint one interaction handoff lives in and degrade --
    instead of losing the caller's turn -- on a closed set of expected
    failures, including the caller's lease no longer owning the task's
    current attempt and the resume anchor it was given being inconsistent.

    Caller obligations: none. Unlike ``stage_interaction_request``, this
    context manager leaves nothing for its caller to do beyond the
    surrounding transaction's own commit/rollback -- the anchor's database
    read (the other half described in ``InteractionAnchor``'s docstring)
    has already happened before this is ever called, and the attempt and
    anchor assertions that used to be caller obligations are now checked by
    ``handoff.stage()`` itself, on every call. The one obligation that
    remains is structural, not sequential: the ``with`` block must be the
    transaction's last word before the caller commits -- no I/O in between
    -- because every write inside it, including the reclaim UPDATE, holds
    SQLite's database-wide writer lock (or, on PostgreSQL, ordinary row
    locks) for the whole span, not just an instant.

    This handoff fails closed, on purpose, and does not inherit the
    silent-discard shape a sibling primitive in this package uses.
    ``trace_event_staging.stage_trace_event_row`` / its caller
    ``DatabaseTraceHandler._save_trace_event`` (``web/api/trace_handlers.py``)
    has a branch that discovers the parent task row is gone and, for a
    non-required event, logs at debug and returns without raising -- a
    deliberate silent discard, scoped to that primitive's best-effort trace
    events. This module has no equivalent branch anywhere: every one of the
    six swallowed exceptions below is swallowed because it is a named,
    expected outcome with its own degradation signal, never because the
    underlying data (the task, the anchor, the request row) turned out to
    be missing or unreadable. A caller with a missing task or a broken
    anchor gets an exception here -- ``InteractionAnchorCorrupt`` or a
    database error, not a quiet no-op -- because a blocking interaction
    request that is silently never staged strands the caller's turn
    exactly the way this module's degrade-instead-of-losing-the-turn design
    exists to prevent.

    Nesting, exactly:

        savepoint = db.begin_nested()        # this function's own
        yield handoff                        # caller calls handoff.stage()
                                              # exactly once; that call
                                              # asserts attempt and anchor
                                              # validity first, then opens
                                              # and closes its *own* inner
                                              # savepoint (see
                                              # stage_interaction_request)
        normal exit         -> savepoint.commit()
        one of six expected failures
                             -> savepoint.rollback(), register a
                                degradation signal, log, and swallow
        anything else        -> savepoint.rollback(), re-raise unchanged

    The attempt and anchor assertions live inside ``handoff.stage()``,
    called from the caller's ``with``-body, not in this function before
    ``yield``. That placement is forced by what a ``@contextmanager``
    generator can and cannot do, not a style preference: a ``with``
    statement always runs its body once ``__enter__`` succeeds -- there is
    no way for ``__enter__`` to succeed and have the body silently skipped.
    ``__enter__`` succeeding, for a generator-based context manager, means
    the generator reached its ``yield``. If an assertion raised and was
    caught *before* ``yield``, the only way to make ``with`` proceed as if
    nothing happened would be to swallow it and fall out of the generator
    without yielding -- but a generator that returns instead of yielding
    makes ``next()`` raise ``StopIteration``, and ``contextlib`` turns that
    into ``RuntimeError("generator didn't yield")`` from ``__enter__``. That
    error is not one of the six swallowed types, so it would propagate to
    the caller anyway, defeating the whole point of swallowing it in the
    first place -- and because it fires from ``__enter__``, the caller's own
    code after the ``with`` block (its commit, most importantly) never
    runs. Checking both preconditions inside ``stage()`` instead means a
    failure surfaces from *inside* the ``with``-body, at a point the
    surrounding ``try``/``except`` here is actually able to intercept via
    the generator's normal exception-resumption path, so ``with`` completes
    normally and everything after it, including the caller's commit, still
    executes. Both assertions run first, before any statement ``stage``
    could otherwise issue -- mutation tests pin that ordering.

    Six expected failures are swallowed:
    ``InteractionSlotTaken``, ``InteractionRequestClosed``,
    ``InteractionAnchorCorrupt``, ``InteractionAttemptMismatch``,
    ``InteractionRunPartitionMismatch``, and ``InteractionOriginUnknown``.
    Every other exception -- including ``InteractionOwnerStateError`` and
    ``InteractionHandoffMisuse`` -- propagates unchanged, after this
    savepoint is rolled back.

    ``InteractionRunPartitionMismatch`` is a deliberate, scoped override of
    this primitive's own default. Read literally, the design that produced
    ``stage_interaction_request`` calls for this exception to propagate
    unchanged, the same as ``InteractionOwnerStateError`` -- on the
    reasoning that a confirmed identity mismatch between a run and its own
    resume anchor is closer to "something is structurally wrong" than to an
    ordinary race. The handoff overrides that default and swallows it here
    instead, registering its own dedicated signal
    (``INTERACTION_RUN_PARTITION_MISMATCH_DEGRADED``, kept separate from the
    other five's shared ``INTERACTION_HANDOFF_DEGRADED`` so it stays
    individually addressable on ``/health``) and a structured log line
    naming the task, the lease's run, and the anchor's run partition. This
    override ships with zero production callers and therefore zero
    real-world reachability data for it: **the change that wires the first
    production caller -- the one that removes the zero-production-caller
    AST gate -- must re-adjudicate this policy once a real reachability
    picture exists**, not carry it forward unexamined.

    Every degradation signal this module registers is sticky: nothing
    clears it once set (``ops_signals.py`` has no automatic-clear path, by
    design -- see that module's docstring). ``/health`` reports a
    degradation from an hour-old, already-recovered handoff exactly the
    same as one from a second ago. This module inherits that behavior; it
    does not introduce it.

    A SQLite-only correctness gap, found and fixed while building this
    module: a session whose first write-adjacent statement is a bare
    SAVEPOINT breaks pysqlite's transaction tracking badly enough that the
    savepoint's later release becomes a permanent commit and a subsequent
    ``Session.rollback()`` does not undo it, confirmed directly (not even
    from that same session's own point of view). This function's own outer
    savepoint is exactly that first statement whenever a caller has issued
    no DML of its own before entering this context manager, which would
    make every one of the six swallowed exceptions' rollback silently do
    nothing -- the half-written garbage this context manager exists to
    prevent. The zero-row, single-column, SQLite-only ``UPDATE`` issued
    immediately below, before the savepoint opens, is the fix: see the
    comment there for the reproduction, the SQLAlchemy-documented pysqlite
    recipe that was considered and rejected instead, and why. It is gated
    on the session's own dialect and skipped entirely on PostgreSQL, which
    was checked directly and does not share this behavior at all.

    In the wiring-window while ``lease.attempt_id is None`` is still
    possible (see ``_assert_current_attempt``'s docstring), this handoff's
    identity key is already run-scoped (``uq_task_interaction_request_identity``
    covers ``task_id, run_id, request_idempotency_key``), which is weaker
    protection against a stale worker than the task-scoped identity an
    earlier design considered: both the attempt fence and the stale-worker
    protection a task-scoped key would have given are absent for the same
    rolling-restart window at once.
    """

    # A zero-row, SQLite-only, single-column UPDATE, issued deliberately
    # before this function's own savepoint opens. This statement's only
    # purpose is to be a real DML statement: it touches no row (id = -1 can
    # never match one an autoincrement primary key minted) and its rowcount
    # is never read. Removing it silently breaks the swallow-and-rollback
    # contract this function exists to provide, whenever a caller enters
    # this context manager without a prior write of its own on the same
    # session -- exactly this file's own
    # test_cm4_with_exit_does_not_commit_outer_transaction shape.
    #
    # Reproduction: on SQLite, a session whose first write-adjacent
    # statement is a bare SAVEPOINT breaks pysqlite's transaction tracking.
    # The savepoint's later release becomes a permanent commit at the
    # engine level, and a subsequent Session.rollback() does not undo it --
    # confirmed directly, including from that same session's own point of
    # view. A caller's own prior write avoids this (and so does
    # stage_interaction_request's own inner savepoint, whose reclaim UPDATE
    # -- a real DML -- always runs first); this function's outer savepoint
    # has no such statement ahead of it otherwise. PostgreSQL does not
    # share this behavior at all, so this guard is gated to SQLite; issuing
    # it unconditionally on PostgreSQL would do nothing useful there while
    # adding a statement to every handoff.
    #
    # The standard fix, and why it was not used: SQLAlchemy's documented
    # pysqlite recipe (disable pysqlite's own isolation_level, issue an
    # explicit BEGIN on the engine's "begin" event -- see
    # https://docs.sqlalchemy.org/en/20/dialects/sqlite.html#serializable-isolation-savepoints-transactional-ddl)
    # does fix this, confirmed directly. It also breaks a scenario this
    # module's own tests require: two sessions racing on the same identity
    # key (REPLAY-after-conflict, T-P-9/T-SP-2), where the first session reads
    # before the second commits and then tries to write afterward, fails
    # with sqlite3.OperationalError ("database is locked", sqlite_errorcode
    # 517 / SQLITE_BUSY_SNAPSHOT) instead of the IntegrityError this
    # primitive is built to classify -- a WAL snapshot conflict, not
    # ordinary lock contention, so busy_timeout cannot wait it out. That
    # recipe is also not present in this codebase's actual SQLite engine
    # configuration (apply_sqlite_concurrency_pragmas, xagent/db/sqlite.py),
    # and applying it there instead -- the only way to make it available
    # without every caller opting in individually -- would change
    # transaction behavior for every other user of db.begin_nested() in
    # this codebase, a blast radius well beyond one staging primitive, and
    # would need its own resolution for the busy_timeout/snapshot-conflict
    # tradeoff for genuine concurrent callers before it could ship anywhere.
    # That is a SQLite concurrency-model decision, not a staging-primitive
    # one. The dummy-DML fix below is the scoped alternative: local to this
    # function, changes nothing about how any other caller's session
    # behaves, and does not touch engine configuration at all. The wiring
    # batch, once real caller topology and deployment concurrency
    # assumptions exist, should revisit whether the engine-level recipe
    # becomes worth its cost at that point.
    #
    # Written as raw SQL via sa.text(), not sa.update(Task), specifically to
    # keep its SET clause to exactly the one column named ("id = id", a
    # self-assignment): sa.update(Task).where(Task.id == -1), with no
    # .values() at all, compiles to a SET clause naming every column
    # SQLAlchemy maps on Task -- all 44 of them, including runner_id and
    # lease_attempt_id -- because Core has nothing to narrow the SET clause
    # to without an explicit .values(). Even sa.update(Task).where(...)
    # .values(id=-1) does not get down to one column: Task.updated_at
    # carries onupdate=func.now(), which SQLAlchemy Core appends to any
    # UPDATE on this table whenever the column is not given an explicit
    # value, so that form still compiles to two SET columns (id,
    # updated_at) on both backends -- confirmed directly. Raw text is the
    # only way to name exactly one column and nothing else; the same
    # bypass-the-onupdate-handler technique is already used for the same
    # reason by acquire_runtime_key_transition_fence in
    # services/api_keys.py. This module's own module-docstring invariant
    # ("neither one writes any data column of tasks") depends on the SET
    # clause never growing back to include a real data column -- do not
    # replace this with sa.update(Task) in any form without re-verifying
    # what it compiles to.
    if db.get_bind(Task).dialect.name == "sqlite":
        db.execute(sa.text(f"UPDATE {Task.__tablename__} SET id = id WHERE id = -1"))

    # Session.begin_nested() always flushes first (it has to, in order to
    # establish the SAVEPOINT after whatever is already pending): a doomed
    # caller write that stage_interaction_request's own flush would
    # otherwise catch and turn into InteractionOwnerStateError can just as
    # well surface right here, before this function's own savepoint even
    # exists to roll back. Same exception, same reasoning, only fired one
    # statement earlier than the primitive's own docstring describes.
    try:
        savepoint = db.begin_nested()
    except IntegrityError as exc:
        raise InteractionOwnerStateError(
            "caller's pending writes failed to flush while opening the "
            "interaction handoff's savepoint"
        ) from exc
    handoff = _InteractionHandoff(db, lease, task=task, anchor=anchor, now=now)
    try:
        yield handoff
    except _SWALLOWED as exc:
        # The except clause matches subclasses (isinstance semantics), so the
        # signal lookup must too: an exact type(exc) key lookup would raise
        # KeyError for a future subclass of a swallowed type, and KeyError is
        # not swallowed -- the degradation path would become a crash path.
        signal = next(
            (sig for cls, sig in _DEGRADATION_SIGNALS.items() if isinstance(exc, cls)),
            INTERACTION_HANDOFF_DEGRADED,
        )
        try:
            register_degradation(
                signal,
                f"task {task.id} run {lease.run_id}: {type(exc).__name__}: {exc}",
            )
            logger.error(
                "interaction handoff degraded",
                extra={
                    "task_id": task.id,
                    "lease_run_id": lease.run_id,
                    "lease_attempt_id": lease.attempt_id,
                    "anchor_run_partition": anchor.resume_run_partition,
                    "exception_type": type(exc).__name__,
                    "degradation_signal": signal,
                },
            )
        finally:
            # Registration and logging run first, in the try; the rollback
            # lives in finally so it still runs even if logger.error itself
            # raises (a misconfigured handler, for instance) -- otherwise
            # that would leak this savepoint open on top of replacing the
            # swallowed exception. The rollback stays guarded: a caller that
            # violated the never-commit-or-rollback-inside-this-block
            # contract has already deactivated this savepoint, and an
            # unguarded rollback would raise ResourceClosedError, replacing
            # whatever exception is already in flight and skipping the
            # degradation signal entirely.
            if savepoint.is_active:
                savepoint.rollback()
        return
    except BaseException:
        if savepoint.is_active:
            savepoint.rollback()
        raise
    else:
        if not savepoint.is_active:
            raise InteractionHandoffMisuse(
                "the caller committed or rolled back inside the "
                "interaction_handoff block; the savepoint that contains the "
                "staged interaction row no longer exists"
            )
        savepoint.commit()
