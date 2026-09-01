"""Stage one trace row (and, on the live-lease checkpoint path, its
exact-row anchor) into a session the caller already owns.

``stage_trace_event_row`` is the construction step factored out of
``DatabaseTraceHandler._save_trace_event`` (``web/api/trace_handlers.py``):
building the row, encoding checkpoint blobs, stamping the run partition, and
flushing to get a primary key for the anchor column. Everything that used to
follow that construction in the same function -- the pointer UPDATE and its
rowcount classification, tool-usage counters, the commit, the prune trigger,
and both rollback handlers -- stays in the caller.

Caller obligations, because none of them happen here:

* This function joins the ``Session`` passed in; it never commits, never
  rolls back, never signals anyone else about the row, and never prunes.
  The caller owns the transaction and must commit it (or roll it back)
  itself.
* If this function raises ``IntegrityError`` (a stale FK on ``task_id``, for
  instance), the session is left mid-transaction. The caller must roll back
  before issuing another statement on it.

``StagedTraceRow.row_id`` and ``.anchor`` are only meaningful once the
caller commits: ``row_id`` is the primary key SQLAlchemy assigned during
this call's flush, but it names a real, durable row only after commit
succeeds. ``anchor`` is non-``None`` only on the root-checkpoint-with-a-live-
lease path (flush happens there because the caller needs the anchor's
primary key to write ``tasks.last_checkpoint_trace_event_id`` before it
commits); every other path returns ``anchor=None`` and does not flush.

Known bypass: ``_persist_agent_outbound_event`` (``web/api/websocket.py``)
constructs its own ``TraceEvent`` row and adds it to the session directly,
without going through ``_save_trace_event`` or this function. This function
is the sole construction point on the ``_save_trace_event`` path, not the
only place a trace row is ever built in the codebase.

A third writer would have to sanitize for itself too (#1248): every payload
reaching these columns must pass ``sanitize_json_payload`` first, and that
is a convention here rather than something the type system enforces. It is
not enforced structurally because the ordering is load-bearing -- the
sanitize has to happen before ``encode_checkpoint_data_for_storage`` hashes
the payload, which a column-level ``TypeDecorator`` (running at bind time,
long after the hash) could not guarantee. On PostgreSQL a missed sanitize
fails loudly at INSERT, since the columns are ``jsonb``; on SQLite it would
store silently, so a new writer is worth checking by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from ...core.agent.checkpoint import (
    CHECKPOINT_EVENT_TYPE,
    CHECKPOINT_TYPE,
    READABLE_CHECKPOINT_TYPES,
    checkpoint_execution_id,
)
from ..models.task import TraceEvent as DatabaseTraceEvent
from ..utils.json_payload_sanitizer import sanitize_json_payload
from .task_lease_service import TASK_RUN_ID_TRACE_FIELD, TaskLease
from .trace_message_storage import encode_checkpoint_data_for_storage


@dataclass(frozen=True)
class StagedCheckpointAnchor:
    """The exact-row anchor for a checkpoint staged under a live task lease.

    Non-``None`` only on the root-checkpoint-with-a-live-lease path.
    ``trace_event_id`` is the primary key the caller writes into
    ``tasks.last_checkpoint_trace_event_id``; it names a real row only once
    the caller commits the transaction this row was staged into.
    """

    checkpoint_event_id: str
    trace_event_id: int


@dataclass(frozen=True)
class StagedTraceRow:
    """A trace row added to a caller-owned session, not yet committed.

    ``row_id`` is ``None`` on every path that does not flush (see the
    module docstring); where it is set, it names a real row only after the
    caller's commit succeeds. ``anchor`` is ``None`` on every path except
    the root-checkpoint-with-a-live-lease path.

    ``stored_data`` is the payload as written to the row: the caller's own
    ``data``, unchanged except on the checkpoint paths, which merge the
    run-id field and encode blobs. It carries whatever type the caller
    passed, which is why it is not annotated more tightly than the
    ``data`` parameter itself.
    """

    row_id: int | None
    stored_data: Any
    anchor: StagedCheckpointAnchor | None


def stage_trace_event_row(
    db: Session,
    *,
    task_id: int,
    build_id: str | None,
    event_id: str,
    event_type: str,
    timestamp: datetime,
    step_id: str | None,
    parent_event_id: str | None,
    data: Any,
    checkpoint_lease: TaskLease | None,
) -> StagedTraceRow:
    """Add one trace row to ``db`` and, on the live-lease checkpoint path,
    flush it to stage its exact-row anchor. See the module docstring for
    what the caller still owns.

    ``checkpoint_lease`` is the caller's own ``current_task_lease()``
    result, already gated on ``build_id is None`` before this is called --
    a sub-agent checkpoint (``build_id`` set) must be passed
    ``checkpoint_lease=None`` regardless of whether a lease is live.
    """
    # Sanitize before anything derives from the payload: the checkpoint
    # branches below hash and deduplicate blob rows out of ``data``, and the
    # stored hash must be computed over what actually lands in the column.
    # PostgreSQL's jsonb rejects NUL and unpaired-surrogate code points at
    # INSERT (#1248); on other dialects the same cleaning keeps stored
    # payloads identical across backends.
    data = sanitize_json_payload(data)

    is_checkpoint = (
        event_type == "system_update_general"
        and isinstance(data, dict)
        and data.get("checkpoint_type") == CHECKPOINT_TYPE
    )
    if is_checkpoint and checkpoint_lease is not None:
        data = {
            **data,
            TASK_RUN_ID_TRACE_FIELD: checkpoint_lease.run_id,
        }
    if is_checkpoint:
        data = encode_checkpoint_data_for_storage(
            db,
            task_id=task_id,
            data=data,
        )

    trace_event = DatabaseTraceEvent(
        task_id=task_id,
        build_id=build_id,
        event_id=event_id,
        event_type=event_type,
        timestamp=timestamp,
        step_id=step_id,
        parent_event_id=parent_event_id,
        data=data,
    )
    db.add(trace_event)

    if is_checkpoint and build_id is None and checkpoint_lease is not None:
        # Flush the pending insert so trace_event.id (the row's primary
        # key) is assigned before the caller builds the exact-row anchor
        # column from it. autoflush is off for this session, so without
        # this the anchor would be built from an unassigned id and
        # silently written NULL.
        db.flush()
        if trace_event.id is None:
            raise RuntimeError(
                f"Task {task_id} checkpoint {event_id} row has no "
                "primary key after flush; refusing to write a NULL "
                "checkpoint anchor"
            )
        row_id = int(trace_event.id)
        return StagedTraceRow(
            row_id=row_id,
            stored_data=data,
            anchor=StagedCheckpointAnchor(
                checkpoint_event_id=event_id,
                trace_event_id=row_id,
            ),
        )

    return StagedTraceRow(row_id=None, stored_data=data, anchor=None)


def checkpoint_run_partition_filter(run_id: str | None) -> Any:
    """The run-partition predicate the legacy checkpoint scan uses to filter
    ``trace_events`` rows by run partition
    (``DatabaseTraceHandler._checkpoint_run_partition_filter``,
    ``web/api/trace_handlers.py``).

    Moved here, unchanged, from that ``trace_handlers.py`` staticmethod: this
    module already imports every name the predicate needs
    (``DatabaseTraceEvent``, ``TASK_RUN_ID_TRACE_FIELD``), and services must
    not import from api, so the move means a future services-layer consumer
    of this predicate never has to reverse-import the api layer to get it.

    The interaction anchor resolver (``resolve_interaction_anchor``,
    ``services/task_interaction_anchor.py``) is not such a consumer -- it
    never calls this function. It does its own plain-Python equality check
    against a single already-fetched row instead of building a SQL predicate
    for a query's ``WHERE`` clause, and is mentioned here only as this
    function's semantic sibling: the two answer the same run-partition-match
    question, but disagree on what ``run_id IS NULL`` means. ``run_id is
    None`` is a legitimate partition here -- the root-checkpoint read path
    this predicate was written for can genuinely have no run id yet --
    which is why it compiles to ``IS NULL`` rather than being rejected. The
    resolver instead treats ``task.run_id IS NULL`` as absence outright,
    before any row is even read (see that module's own docstring for why).
    """
    run_field = DatabaseTraceEvent.data[TASK_RUN_ID_TRACE_FIELD].as_string()
    return run_field == run_id if run_id is not None else run_field.is_(None)


# The six conditions below are named rather than anonymous so a caller can
# ask which one failed, not merely whether any did. Order here is the order
# the judgment reads in prose; it is not itself a contract, because every
# condition is a pure read with no side effect and all six are evaluated on
# every call. The ordering contract that does exist belongs to the callers:
# the corrupt judgment must run before any legacy-checkpoint-type judgment
# (see task_interaction_anchor.py's module docstring on steps 4 and 5).
CHECKPOINT_ROW_TASK_OWNERSHIP = "task_ownership"
CHECKPOINT_ROW_EVENT_TYPE = "event_type"
CHECKPOINT_ROW_BUILD_SCOPE = "build_scope"
CHECKPOINT_ROW_CHECKPOINT_TYPE = "checkpoint_type"
CHECKPOINT_ROW_RUN_PARTITION = "run_partition"
CHECKPOINT_ROW_EXECUTION_IDENTITY = "execution_identity"

_CHECKPOINT_ROW_EVENT_TYPE_NAME = str(CHECKPOINT_EVENT_TYPE)


def failed_checkpoint_row_conditions(
    row: DatabaseTraceEvent,
    row_data: dict[str, Any],
    *,
    task_id: int,
    run_id: str | None,
    execution_id: str,
) -> frozenset[str]:
    """Which self-consistency conditions ``row`` fails as the checkpoint a
    task's exact-row pointer names. An empty result means the row is a
    legitimate checkpoint for this ``(task_id, run_id, execution_id)``.

    One definition, three consumers. ``_load_pk_anchored_checkpoint``
    (``api/trace_handlers.py``), ``resolve_interaction_anchor``
    (``services/task_interaction_anchor.py``) and lease recovery's
    ``_candidate_row_failures`` (``services/task_lease_service.py``) all read
    it. The first two ask the same question about the same row from opposite
    directions -- one decides whether an already-written anchor still
    resolves, the other decides what may be anchored to -- and an answer
    that differs between them is a bug by construction, not a judgment call.
    They previously kept two hand-copied disjunctions aligned through a
    static AST comparison; that comparison could only see the operands, not
    how ``partition_matches`` was computed above them, and the two sides did
    in fact differ there. Lease recovery asks the same question from a third
    direction: whether an expired lease's checkpoint pointer still names a
    checkpoint to resume from, rather than a task to fail. Its import of
    this function lives inside the function that calls it, not at module
    level -- this module already imports ``task_lease_service`` for
    ``TASK_RUN_ID_TRACE_FIELD`` and ``TaskLease``, so a module-level import
    in the other direction is a cycle, not a style choice; do not move it to
    the top of the file. ``_resolve_read_direction_anchor``
    (``services/task_interaction_service.py``) carries a related but not
    identical judgment and is not a caller -- see the note at the end.

    What is deliberately NOT decided here: whether a legacy
    ``checkpoint_type`` is acceptable. Condition
    ``CHECKPOINT_ROW_CHECKPOINT_TYPE`` admits every member of
    ``READABLE_CHECKPOINT_TYPES``, the current type and the legacy ones, on
    every caller's behalf. ``resolve_interaction_anchor`` rejects the legacy
    ones in a separate later step of its own; the read paths accept them.
    That difference stays at the call sites, where the reason for it lives.

    ``row_data`` is passed in rather than read off ``row`` because every
    caller has already normalized a non-dict ``row.data`` to an empty dict
    and needs the normalized value afterwards; taking it as an argument
    keeps one normalization instead of two that could disagree.

    ``run_id`` of ``None`` is a legitimate partition, matched by a row whose
    own run field is also absent -- the root-checkpoint read path can
    genuinely have no run id yet. Callers that treat a missing ``run_id`` as
    a terminal outcome of their own must do so before calling this, not by
    expecting a failure here.

    An empty ``row_execution_id`` (a legacy row carrying no identity field)
    passes ``CHECKPOINT_ROW_EXECUTION_IDENTITY`` on purpose. That leniency
    is part of the condition, not a gap in it: such a row is anchored but
    not scanned, and rejecting it here would make an unreadable checkpoint
    look like no checkpoint.

    Not consumed by ``_resolve_read_direction_anchor``, which needs a
    seventh condition this one has no input for (the trace row's
    ``event_id`` must equal the interaction row's ``resume_event_id``) and
    compares the partition against a non-null ``resume_run_partition``
    rather than a task's possibly-null ``run_id``. Folding it in would mean
    adding an optional condition and a second partition rule for one caller;
    it stays separate until something makes the two genuinely the same
    question.
    """

    failed: set[str] = set()

    if row.task_id != task_id:
        failed.add(CHECKPOINT_ROW_TASK_OWNERSHIP)
    if row.event_type != _CHECKPOINT_ROW_EVENT_TYPE_NAME:
        failed.add(CHECKPOINT_ROW_EVENT_TYPE)
    if row.build_id is not None:
        failed.add(CHECKPOINT_ROW_BUILD_SCOPE)
    if row_data.get("checkpoint_type") not in READABLE_CHECKPOINT_TYPES:
        failed.add(CHECKPOINT_ROW_CHECKPOINT_TYPE)

    run_field = row_data.get(TASK_RUN_ID_TRACE_FIELD)
    partition_matches = run_field == run_id if run_id is not None else run_field is None
    if not partition_matches:
        failed.add(CHECKPOINT_ROW_RUN_PARTITION)

    row_execution_id = checkpoint_execution_id(row_data)
    if row_execution_id and row_execution_id != execution_id:
        failed.add(CHECKPOINT_ROW_EXECUTION_IDENTITY)

    return frozenset(failed)


def is_missing_run_partition_only(
    failed: frozenset[str], row_data: dict[str, Any]
) -> bool:
    """True when the only condition ``row_data``'s row failed is the
    run-partition match, and it failed because the run field reads as
    absent rather than holding the wrong value.

    "Reads as absent" is literal: this is ``row_data.get(...) is None``, so
    a key that is missing and a key explicitly stored as JSON ``null`` are
    the same answer here. That is deliberate rather than unnoticed. The
    writer never stores ``null`` -- ``stage_trace_event_row`` above only
    writes this field for a current-type checkpoint, and only from a lease
    whose ``run_id`` it already has -- so an explicit ``null`` is a shape
    nothing in this system produces. Distinguishing the two would put such a
    row in the corrupt branch, which is the worse direction to fail for a
    shape we have no evidence about; treating it as pre-existing costs at
    most one deferral to the legacy scan, which validates the partition
    itself.

    That shape is a pre-existing checkpoint row, not a corrupt one: the
    checkpoint pointer column is backfilled from the legacy event-id column
    (the 20260804 migration), and the ``trace_events`` row it matches can
    predate the run-partition field; a legacy-type checkpoint row never
    carries the field at all, because ``stage_trace_event_row`` above only
    writes it for current-type checkpoints.

    Both conditions are load-bearing. Dropping the "only" makes a row that
    is wrong in some other way as well look pre-existing; dropping the
    "absent" makes a row carrying a genuinely wrong partition look
    pre-existing. Each has its own test.

    ``failed`` must be the result of ``failed_checkpoint_row_conditions``
    called on this same ``row_data``; nothing in the signature enforces that
    pairing. Every caller computes the two on adjacent lines from one row,
    which is the only supported use -- passing a ``failed`` set derived from
    a different row would produce an answer about neither.
    """

    return failed == {CHECKPOINT_ROW_RUN_PARTITION} and (
        row_data.get(TASK_RUN_ID_TRACE_FIELD) is None
    )
