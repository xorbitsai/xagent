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
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from ...core.agent.checkpoint import CHECKPOINT_TYPE
from ..models.task import TraceEvent as DatabaseTraceEvent
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
