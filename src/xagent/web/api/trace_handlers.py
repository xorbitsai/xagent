"""Web-specific trace handlers for database operations."""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import func, update
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from ...config import get_checkpoint_history_limit
from ...core.agent.checkpoint import (
    CHECKPOINT_TYPE,
    READABLE_CHECKPOINT_TYPES,
    CheckpointAccessRefusedError,
    CheckpointCorruptError,
    CheckpointReadError,
    CheckpointUnavailableError,
    checkpoint_execution_id,
)
from ...core.agent.trace import BaseTraceHandler
from ...core.agent.trace import TraceEvent as CoreTraceEvent
from ...core.tools.adapters.vibe.connector_runtime import (
    redact_runtime_sensitive_payload,
)
from ...web.models.database import get_db
from ...web.models.task import Task, TaskStatus
from ...web.models.task import TraceEvent as DatabaseTraceEvent
from ...web.models.task_interaction import TaskInteractionRequest
from ...web.models.tool_config import ToolUsage
from ...web.services.ops_signals import (
    CHECKPOINT_DECODE_FALLBACK,
    CHECKPOINT_LOAD_UNAVAILABLE,
    CHECKPOINT_PK_ANCHOR_DANGLING,
    CHECKPOINT_PRUNE_FAILED,
    clear_degradation,
    register_degradation,
)
from ...web.services.task_interaction_schema import interaction_requests_table_exists
from ...web.services.task_lease_service import (
    TASK_RUN_ID_TRACE_FIELD,
    current_task_lease,
)
from ...web.services.trace_event_staging import stage_trace_event_row
from ...web.services.trace_message_storage import (
    SQL_IN_CLAUSE_CHUNK_SIZE,
    CheckpointMessageDecodeError,
    chunks,
    decode_trace_event_data,
)

logger = logging.getLogger(__name__)

# Page size for one batch of the checkpoint read scan. A read does not stop
# at the first page: it keeps paging through the matching set (see
# _sync_load_latest_checkpoint below) until a readable row is found or the
# set is proven exhausted (a page shorter than this). The constant only
# bounds the cost of one query, not how many candidate rows a read may
# examine before ruling on unavailable vs. corrupt.
CHECKPOINT_ROW_SCAN_LIMIT = 100

# Operational bound on the scan loop. With history pruning disabled
# (XAGENT_CHECKPOINT_HISTORY_LIMIT=0) and a large backlog of matching rows,
# the loop would otherwise issue one query per CHECKPOINT_ROW_SCAN_LIMIT
# rows before proving the matching set exhausted. This caps that cost: a
# scan that reaches the cap without a resolution is treated as unavailable
# rather than continuing indefinitely.
CHECKPOINT_SCAN_MAX_PAGES = 50


@dataclass(frozen=True)
class _AnchorFallback:
    """Why the PK anchor deferred to the legacy scan.

    Distinguishes "no PK anchor to try" from a resolved snapshot, which may
    legitimately be an empty dict. An unset or dangling pointer defers with
    nothing to carry. A correctly identified anchor row whose payload could
    not be read defers *and* hands the scan the same verdict flag the scan
    would have set for that row itself: the scan's candidate filter can
    legitimately exclude that row (a row carrying no execution identity is
    anchored but not scanned, see _load_pk_anchored_checkpoint), and an
    excluded row must never let an unreadable checkpoint become "no
    checkpoint".
    """

    undecodable: bool = False
    generic_failure: bool = False


def _checkpoint_execution_id_predicate(execution_id: str) -> Any:
    """SQL mirror of ``checkpoint_execution_id()``: root wins, then the flat
    field, then the snapshot's own id -- so legacy rows that only set one of
    them are not skipped. The read query and history pruning must agree on
    which rows belong to one execution, or pruning could drop a row the read
    path still considers current (or vice versa); both consume this one
    predicate rather than keeping independently maintained copies in sync by
    hand.
    """
    return (
        func.coalesce(
            func.nullif(
                DatabaseTraceEvent.data["root_execution_id"].as_string(),
                "",
            ),
            func.nullif(
                DatabaseTraceEvent.data["execution_id"].as_string(),
                "",
            ),
            DatabaseTraceEvent.data["snapshot"]["execution_id"].as_string(),
        )
        == execution_id
    )


def _convert_float_to_datetime(timestamp: Any) -> datetime:
    """Convert float timestamp to datetime for database storage."""
    if isinstance(timestamp, (int, float)):
        return datetime.fromtimestamp(timestamp, timezone.utc)
    elif isinstance(timestamp, datetime):
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=timezone.utc)
        return timestamp
    else:
        return datetime.now(timezone.utc)


class DatabaseTraceHandler(BaseTraceHandler):
    """Enhanced trace handler that saves events to database with clear scope handling."""

    def __init__(self, task_id: int, build_id: Optional[str] = None):
        super().__init__()
        self.task_id = task_id
        self.build_id = build_id

    async def _handle_task_event(self, event: CoreTraceEvent) -> None:
        """Handle task-level events for database storage."""
        await self._save_to_database(event)

    async def _handle_step_event(self, event: CoreTraceEvent) -> None:
        """Handle step-level events for database storage."""
        await self._save_to_database(event)

    async def _handle_action_event(self, event: CoreTraceEvent) -> None:
        """Handle action-level events for database storage."""
        await self._save_to_database(event)

    async def _handle_system_event(self, event: CoreTraceEvent) -> None:
        """Handle system-level events for database storage."""
        await self._save_to_database(event)

    async def _save_to_database(self, event: CoreTraceEvent) -> None:
        """Save trace event to database."""
        try:
            # Run synchronous database operations in a thread pool to avoid blocking event loop
            await asyncio.to_thread(self._sync_save_to_database, event)
        except Exception as e:
            # Don't catch required field validation errors - let them propagate
            if isinstance(e, ValueError) and ("missing required" in str(e)):
                logger.error(f"Re-raising required field validation error: {e}")
                raise
            if getattr(event, "require_persisted", False):
                logger.error(
                    "Required trace event persistence failed for task %s: %s",
                    self.task_id,
                    e,
                )
                raise

            logger.warning(
                f"Failed to save trace event to database for task {self.task_id}: {e}"
            )

    async def load_latest_checkpoint(
        self, execution_id: str
    ) -> Optional[Dict[str, Any]]:
        """Load the latest agent checkpoint persisted as a trace event.

        ``None`` means the query completed and found nothing -- an
        authoritative fact. Anything that prevented that determination
        (query failure, refused partition, undecodable rows) is translated
        to a ``CheckpointReadError`` subclass inside the sync worker below
        and propagates through here unchanged; it must never collapse back
        to ``None``, or a transient failure would be indistinguishable from
        "no checkpoint" to every caller up the stack.
        """
        return await asyncio.to_thread(
            self._sync_load_latest_checkpoint,
            execution_id,
        )

    def _sync_load_latest_checkpoint(
        self,
        execution_id: str,
    ) -> Optional[Dict[str, Any]]:
        try:
            db = next(get_db())
        except Exception as exc:
            register_degradation(
                CHECKPOINT_LOAD_UNAVAILABLE,
                f"task {self.task_id}: checkpoint session checkout failed",
            )
            raise CheckpointUnavailableError(
                f"task {self.task_id}: could not open a database session "
                "to read the checkpoint"
            ) from exc
        try:
            query = db.query(DatabaseTraceEvent).filter(
                DatabaseTraceEvent.task_id == self.task_id,
                DatabaseTraceEvent.event_type == "system_update_general",
                DatabaseTraceEvent.data["checkpoint_type"]
                .as_string()
                .in_(sorted(READABLE_CHECKPOINT_TYPES)),
                # The page size below bounds this predicate's matching set,
                # not an unfiltered row scan, so a page shorter than it
                # proves the matching set exhausted (see the scan loop
                # below), and a zero-row first page is authoritative.
                _checkpoint_execution_id_predicate(str(execution_id)),
            )
            anchored_fallback = _AnchorFallback()
            if self.build_id is None:
                try:
                    run_id = self._root_checkpoint_read_partition(db)
                except CheckpointReadError:
                    # Already a partition verdict (refused, or the task row
                    # is missing); it carries its own classification.
                    raise
                except Exception as exc:
                    # Resolving the partition is part of the read. A DB
                    # failure here leaves the partition unknown, so the read
                    # could not be completed -- same translation the main
                    # query below gets, or the failure would escape the
                    # contract as a raw driver exception no consumer catches.
                    register_degradation(
                        CHECKPOINT_LOAD_UNAVAILABLE,
                        f"task {self.task_id}: checkpoint partition resolution failed",
                    )
                    raise CheckpointUnavailableError(
                        f"task {self.task_id}: could not resolve the "
                        "checkpoint read partition"
                    ) from exc
                anchored = self._load_pk_anchored_checkpoint(
                    db, run_id, str(execution_id)
                )
                if not isinstance(anchored, _AnchorFallback):
                    return anchored
                anchored_fallback = anchored
                query = query.filter(
                    DatabaseTraceEvent.build_id.is_(None),
                    self._checkpoint_run_partition_filter(run_id),
                )
            else:
                query = query.filter(DatabaseTraceEvent.build_id == self.build_id)

            ordered_query = query.order_by(
                DatabaseTraceEvent.timestamp.desc(),
                DatabaseTraceEvent.id.desc(),
            )

            saw_generic_failure = anchored_fallback.generic_failure
            saw_undecodable_row = anchored_fallback.undecodable
            saw_any_row = (
                anchored_fallback.undecodable or anchored_fallback.generic_failure
            )
            offset = 0
            page_count = 0
            while True:
                page_count += 1
                if page_count > CHECKPOINT_SCAN_MAX_PAGES:
                    # The matching set is not proven exhausted, but scanning
                    # further is not bounded work anymore -- treat it the
                    # same as any other read that could not be completed.
                    register_degradation(
                        CHECKPOINT_LOAD_UNAVAILABLE,
                        f"task {self.task_id}: checkpoint scan reached the "
                        f"{CHECKPOINT_SCAN_MAX_PAGES}-page cap without "
                        "resolving",
                    )
                    raise CheckpointUnavailableError(
                        f"task {self.task_id}: checkpoint scan exceeded "
                        f"{CHECKPOINT_SCAN_MAX_PAGES} pages without "
                        "resolving"
                    )
                try:
                    rows = (
                        ordered_query.offset(offset)
                        .limit(CHECKPOINT_ROW_SCAN_LIMIT)
                        .all()
                    )
                except Exception as exc:
                    register_degradation(
                        CHECKPOINT_LOAD_UNAVAILABLE,
                        f"task {self.task_id}: checkpoint query failed",
                    )
                    raise CheckpointUnavailableError(
                        f"task {self.task_id}: checkpoint query failed"
                    ) from exc
                # This page succeeded -- whatever the decode loop below
                # concludes about the rows it found, the read infrastructure
                # is healthy again.
                clear_degradation(CHECKPOINT_LOAD_UNAVAILABLE)
                if not rows:
                    break
                saw_any_row = True

                for row in rows:
                    data: Dict[str, Any] = (
                        row.data if isinstance(row.data, dict) else {}
                    )
                    try:
                        data = decode_trace_event_data(
                            db,
                            task_id=self.task_id,
                            data=data,
                            strict=True,
                        )
                    except CheckpointMessageDecodeError as exc:
                        saw_undecodable_row = True
                        logger.warning(
                            "Skipping unreadable checkpoint trace event %s "
                            "for task %s: %s",
                            row.event_id,
                            self.task_id,
                            exc,
                        )
                        continue
                    except Exception:
                        # E.g. a transient DB error from the blob prefetch.
                        # Fall back to an older readable checkpoint instead
                        # of letting the error abort loading for the whole
                        # task. Surface the degradation on /health so a
                        # systemic decode failure is observable instead of
                        # only a per-row warning log; the signal self-clears
                        # on the next successful decode.
                        saw_generic_failure = True
                        register_degradation(
                            CHECKPOINT_DECODE_FALLBACK,
                            f"task {self.task_id}: checkpoint decode failed, "
                            f"fell back past event {row.event_id}",
                        )
                        logger.warning(
                            "Skipping checkpoint trace event %s for task %s "
                            "after decode failure",
                            row.event_id,
                            self.task_id,
                            exc_info=True,
                        )
                        continue
                    clear_degradation(CHECKPOINT_DECODE_FALLBACK)
                    snapshot = data.get("snapshot")
                    if not isinstance(snapshot, dict):
                        # The row claims to be a checkpoint but carries no
                        # payload: a permanent failure for this row, the same
                        # class as an undecodable one. Per-row failures never
                        # abort the scan -- an older row may still carry a
                        # usable checkpoint -- and the verdict for the whole
                        # matching set is decided once, after exhaustion.
                        saw_undecodable_row = True
                        logger.warning(
                            "Skipping checkpoint trace event %s for task %s: "
                            "readable checkpoint_type but no snapshot",
                            row.event_id,
                            self.task_id,
                        )
                        continue
                    return dict(snapshot)

                if len(rows) < CHECKPOINT_ROW_SCAN_LIMIT:
                    # A short page proves the matching set is exhausted --
                    # no further rows exist beyond this one.
                    break
                # OFFSET paging over (timestamp DESC, id DESC): a row inserted
                # mid-scan shifts later pages by one. Root reads are fenced to a
                # single run partition, which excludes concurrent writers there;
                # build-scoped histories are append-only per build.
                offset += CHECKPOINT_ROW_SCAN_LIMIT

            if not saw_any_row:
                return None
            # The matching set is exhausted and every candidate row failed.
            # A generic (transient) failure anywhere in the scan is
            # conservatively unavailable -- retryable. Only a fully scanned
            # set that is exclusively permanent decode failures is corrupt.
            if saw_generic_failure:
                register_degradation(
                    CHECKPOINT_LOAD_UNAVAILABLE,
                    f"task {self.task_id}: checkpoint scan exhausted the "
                    "matching set with a generic decode failure among the "
                    "candidate rows",
                )
                raise CheckpointUnavailableError(
                    f"task {self.task_id}: checkpoint read could not be "
                    "completed for all candidate rows"
                )
            if saw_undecodable_row:
                raise CheckpointCorruptError(
                    f"task {self.task_id}: all matching checkpoint rows are undecodable"
                )
            return None
        finally:
            db.close()

    def _root_checkpoint_read_partition(
        self,
        db: Session,
    ) -> str | None:
        """Resolve the run partition this reader may read, or refuse.

        Exact executions read only checkpoints tagged with their bound run.
        Legacy callers can read only untagged rows, and only while the task
        has no active run and no run has ever been tagged. Build-scoped
        checkpoints retain their historical build-only partitioning and do
        not call this helper. A refusal means the checkpoint may exist but
        this reader is not authoritative for it right now -- distinct from
        a query that completed and found nothing.
        """

        lease = current_task_lease()
        if lease is not None:
            if lease.task_id != self.task_id or lease.run_id is None:
                raise CheckpointAccessRefusedError(
                    f"task {self.task_id}: active lease is not bound to this reader",
                    reason="lease_mismatch",
                )
            return lease.run_id

        task_run = db.query(Task.run_id).filter(Task.id == self.task_id).one_or_none()
        if task_run is None:
            # The task row itself is gone -- an exceptional condition, not
            # a partition policy decision.
            # No register_degradation: the signal is process-wide and is only
            # cleared once a page query succeeds later in the read, which this
            # branch never reaches -- one absent task would latch it for the
            # whole process.
            raise CheckpointUnavailableError(
                f"task {self.task_id}: task row is missing"
            )
        if task_run[0] is not None:
            raise CheckpointAccessRefusedError(
                f"task {self.task_id}: an active run is in progress under "
                "a different lease",
                reason="active_run",
            )
        tagged_checkpoint_exists = (
            db.query(DatabaseTraceEvent.id)
            .filter(
                DatabaseTraceEvent.task_id == self.task_id,
                DatabaseTraceEvent.build_id.is_(None),
                DatabaseTraceEvent.event_type == "system_update_general",
                DatabaseTraceEvent.data["checkpoint_type"]
                .as_string()
                .in_(sorted(READABLE_CHECKPOINT_TYPES)),
                DatabaseTraceEvent.data[TASK_RUN_ID_TRACE_FIELD]
                .as_string()
                .is_not(None),
            )
            .first()
            is not None
        )
        if tagged_checkpoint_exists:
            # Positive proof a checkpoint exists in a partition this legacy
            # reader is not allowed to read -- a refusal, not an absence.
            raise CheckpointAccessRefusedError(
                f"task {self.task_id}: a tagged run has already superseded "
                "legacy checkpoints",
                reason="superseded_legacy",
            )
        return None

    @staticmethod
    def _checkpoint_run_partition_filter(run_id: str | None) -> Any:
        run_field = DatabaseTraceEvent.data[TASK_RUN_ID_TRACE_FIELD].as_string()
        return run_field == run_id if run_id is not None else run_field.is_(None)

    def _load_pk_anchored_checkpoint(
        self,
        db: Session,
        run_id: str | None,
        execution_id: str,
    ) -> Dict[str, Any] | _AnchorFallback:
        """Resolve the checkpoint through the task's exact-row pointer.

        Returns an ``_AnchorFallback`` when the read defers to the legacy
        scan. An empty one means there was nothing to anchor on: the
        pointer is unset, or it names a row that no longer exists (only
        possible on a database upgraded through Alembic rather than created
        fresh, since that path has no DB-level FK -- see the migration that
        adds this column).

        Once a target row is found, its *identity* is authoritative: a
        validation mismatch raises rather than falling back to search other
        rows. Its *payload* is not. A row that is correctly identified but
        whose payload cannot be read defers to the scan as well, so the
        older rows history pruning deliberately retains can still answer
        the read (see _prune_checkpoint_history). That fallback carries the
        row's own verdict flag on the returned ``_AnchorFallback``, because
        the scan may legitimately exclude the very row the pointer named,
        and a scan that then finds nothing must not report "no checkpoint"
        for a checkpoint that exists and is unreadable.

        The execution-identity check here is verification, not the legacy
        scan's filtering: that scan excludes non-matching rows from its
        candidate set via ``_checkpoint_execution_id_predicate`` before it
        ever sees them, but the pointer names one row unconditionally, so
        the row's own claimed identity (if it has one) has to be checked
        against the caller's after the fact. A row carrying no execution
        identity at all passes this check, because
        ``checkpoint_execution_id()`` returns "" for it and the conjunct
        short-circuits. The legacy scan does the opposite: its
        ``coalesce(nullif(...))`` predicate yields NULL for such a row and
        ``NULL = :execution_id`` is never true, so the scan drops it from
        the candidate set. The anchor path is deliberately the more
        permissive of the two -- a pointer that names a row is a stronger
        identity claim than a JSON field match, and the rows without the
        field are legacy rows written before it existed, exactly the rows
        the migration's backfill anchors. Nothing depends on the two
        agreeing: web's ``execution_id`` is the task id.
        """
        # Cleared before the pointer is even read, not only when one
        # resolves to a row: the registry is process-wide, so a process
        # where no task currently has an anchor would otherwise keep a
        # stale dangling signal set forever. The coarseness is real and
        # accepted -- one healthy task's read clears another task's
        # dangling signal -- and is the same cross-task coarseness the
        # signal already has in the registering direction.
        clear_degradation(CHECKPOINT_PK_ANCHOR_DANGLING)
        try:
            pointer = (
                db.query(Task.last_checkpoint_trace_event_id)
                .filter(Task.id == self.task_id)
                .one_or_none()
            )
        except Exception as exc:
            register_degradation(
                CHECKPOINT_LOAD_UNAVAILABLE,
                f"task {self.task_id}: checkpoint pointer lookup failed",
            )
            raise CheckpointUnavailableError(
                f"task {self.task_id}: checkpoint pointer lookup failed"
            ) from exc
        if pointer is None or pointer[0] is None:
            return _AnchorFallback()

        pointer_id = pointer[0]
        try:
            row = db.get(DatabaseTraceEvent, pointer_id)
        except Exception as exc:
            register_degradation(
                CHECKPOINT_LOAD_UNAVAILABLE,
                f"task {self.task_id}: checkpoint pointer row fetch failed",
            )
            raise CheckpointUnavailableError(
                f"task {self.task_id}: checkpoint pointer row fetch failed"
            ) from exc
        if row is None:
            register_degradation(
                CHECKPOINT_PK_ANCHOR_DANGLING,
                f"task {self.task_id}: checkpoint pointer {pointer_id} has "
                "no matching trace_events row; falling back to the legacy "
                "scan",
            )
            return _AnchorFallback()

        row_data: Dict[str, Any] = row.data if isinstance(row.data, dict) else {}
        run_field = row_data.get(TASK_RUN_ID_TRACE_FIELD)
        partition_matches = (
            run_field == run_id if run_id is not None else run_field is None
        )
        row_execution_id = checkpoint_execution_id(row_data)
        if (
            row.task_id != self.task_id
            or row.event_type != "system_update_general"
            or row.build_id is not None
            or row_data.get("checkpoint_type") not in READABLE_CHECKPOINT_TYPES
            or not partition_matches
            or (row_execution_id and row_execution_id != execution_id)
        ):
            raise CheckpointCorruptError(
                f"task {self.task_id}: checkpoint pointer {pointer_id} does "
                "not match the row it anchors"
            )

        try:
            decoded = decode_trace_event_data(
                db,
                task_id=self.task_id,
                data=dict(row_data),
                strict=True,
            )
        except CheckpointMessageDecodeError as exc:
            # Permanent per-row failure, classified exactly as the scan
            # classifies one: log, no signal, and defer -- an older row may
            # still carry a usable checkpoint.
            logger.warning(
                "Checkpoint pointer %s row for task %s is undecodable; "
                "falling back to the legacy scan: %s",
                pointer_id,
                self.task_id,
                exc,
            )
            return _AnchorFallback(undecodable=True)
        except Exception:
            # E.g. a transient DB error from the blob prefetch: the same
            # generic case the scan registers CHECKPOINT_DECODE_FALLBACK
            # for. CHECKPOINT_LOAD_UNAVAILABLE belongs to the exhausted-set
            # verdict, not to one row.
            register_degradation(
                CHECKPOINT_DECODE_FALLBACK,
                f"task {self.task_id}: checkpoint decode failed, fell back "
                f"past pointer {pointer_id}",
            )
            logger.warning(
                "Checkpoint pointer %s row for task %s failed to decode; "
                "falling back to the legacy scan",
                pointer_id,
                self.task_id,
                exc_info=True,
            )
            return _AnchorFallback(generic_failure=True)
        # The scan retires these two signals on two different facts: the
        # load signal once a page query returns, the decode-fallback signal
        # once a row decodes. An anchored read establishes both in one round
        # trip, so both clears land here -- and here rather than at the top
        # of this function, where the dangling clear sits, because that one
        # reports on the pointer every attempt re-reads while these report
        # that the read actually got through. An anchored read returns
        # without ever entering the scan, so without these a decode-fallback
        # signal set by one bad row could never retire in the steady state
        # this anchor exists to produce.
        clear_degradation(CHECKPOINT_LOAD_UNAVAILABLE)
        clear_degradation(CHECKPOINT_DECODE_FALLBACK)

        snapshot = decoded.get("snapshot") if isinstance(decoded, dict) else None
        if not isinstance(snapshot, dict):
            # Readable checkpoint_type but no payload: the same permanent
            # per-row class as an undecodable row, and deferred the same way.
            logger.warning(
                "Checkpoint pointer %s row for task %s has a readable "
                "checkpoint_type but no snapshot; falling back to the "
                "legacy scan",
                pointer_id,
                self.task_id,
            )
            return _AnchorFallback(undecodable=True)
        return dict(snapshot)

    def _sync_save_to_database(self, event: CoreTraceEvent) -> None:
        """Synchronous database save operation (runs in thread pool)."""
        # Create database session
        db = next(get_db())
        try:
            # Save unified trace event to database
            self._save_trace_event(db, event)
        finally:
            db.close()

    def _save_trace_event(self, db: Session, event: CoreTraceEvent) -> None:
        """Save trace event in unified format to database."""
        from ...web.api.ws_trace_handlers import get_event_type_mapping

        try:
            # Map the trace event to the unified event type
            event_type_str = get_event_type_mapping(event)

            # Convert timestamp
            timestamp = _convert_float_to_datetime(event.timestamp)

            # Serialize data to ensure JSON compatibility
            data = self._serialize_data_for_json(event.data or {})
            if event_type_str in {
                "tool_execution_start",
                "tool_execution_end",
                "tool_execution_failed",
            }:
                data = redact_runtime_sensitive_payload(data)
            if self._is_duplicate_user_message_turn(db, event_type_str, data):
                logger.debug(
                    "Skipping duplicate user_message turn_id=%s for task %s",
                    data.get("turn_id") if isinstance(data, dict) else None,
                    self.task_id,
                )
                return
            if (
                event_type_str == "system_update_general"
                and isinstance(data, dict)
                and data.get("checkpoint_type") == CHECKPOINT_TYPE
            ):
                checkpoint_lease = (
                    current_task_lease() if self.build_id is None else None
                )
            else:
                checkpoint_lease = None

            staged = stage_trace_event_row(
                db,
                task_id=self.task_id,
                build_id=self.build_id,
                event_id=str(event.id),
                event_type=event_type_str,
                timestamp=timestamp,
                step_id=event.step_id,
                parent_event_id=str(event.parent_id) if event.parent_id else None,
                data=data,
                checkpoint_lease=checkpoint_lease,
            )
            data = staged.stored_data

            if staged.anchor is not None and checkpoint_lease is not None:
                # staged.anchor is set on exactly the path that needs this
                # pointer UPDATE, so it is read here rather than re-derived.
                # Re-deriving would test the post-encode payload rebound at
                # the line above, not the pre-encode payload the flush
                # decision was actually made from; the two agree today only
                # because no encoder touches checkpoint_type. The lease is
                # re-tested only to narrow it for the fence below -- it is
                # never None when the anchor is set.
                pointer_update = db.execute(
                    update(Task)
                    .where(
                        Task.id == self.task_id,
                        Task.status == TaskStatus.RUNNING,
                        Task.runner_id == checkpoint_lease.runner_id,
                        Task.run_id == checkpoint_lease.run_id,
                    )
                    .values(
                        last_checkpoint_event_id=str(event.id),
                        last_checkpoint_trace_event_id=staged.anchor.trace_event_id,
                    )
                    .execution_options(synchronize_session=False)
                )
                if int(getattr(pointer_update, "rowcount", 0) or 0) != 1:
                    task_still_exists = (
                        db.query(Task.id).filter(Task.id == self.task_id).first()
                        is not None
                    )
                    if not task_still_exists:
                        # The task row is gone, not merely leased elsewhere --
                        # the same outcome the trace_events.task_id FK
                        # produces for the row itself wherever that FK is
                        # enforced. Discard the staged row so this path
                        # leaves nothing behind, then classify it exactly as
                        # that FK violation is classified below: a required
                        # event still fails loudly, a best-effort event is
                        # dropped quietly.
                        db.rollback()
                        if getattr(event, "require_persisted", False):
                            logger.error(
                                "Required trace event references missing task %s: %s",
                                self.task_id,
                                event.id,
                            )
                            raise RuntimeError(
                                f"Task {self.task_id} no longer exists; "
                                f"checkpoint {event.id} was not persisted"
                            )
                        logger.debug(
                            "Skip checkpoint pointer update for missing task %s: %s",
                            self.task_id,
                            event.id,
                        )
                        return
                    raise RuntimeError(
                        f"Task {self.task_id} lease changed before checkpoint "
                        f"{event.id} could be persisted"
                    )

            # Update tool usage statistics if this is a tool execution event
            if event_type_str == "tool_execution_end":
                tool_name = data.get("tool_name") if isinstance(data, dict) else None
                if tool_name:
                    try:
                        tool_usage: Any = (
                            db.query(ToolUsage)
                            .filter(ToolUsage.tool_name == tool_name)
                            .first()
                        )
                        if not tool_usage:
                            tool_usage = ToolUsage(
                                tool_name=tool_name,
                                usage_count=0,
                                success_count=0,
                                error_count=0,
                            )
                            db.add(tool_usage)

                        tool_usage.usage_count += 1
                        # We assume success for tool_execution_end events as errors are typically handled separately
                        # and react pattern emits this event on success
                        if isinstance(data, dict) and data.get("success", True):
                            tool_usage.success_count += 1
                        else:
                            tool_usage.error_count += 1

                        tool_usage.last_used_at = timestamp
                        logger.debug(f"Updated usage stats for tool {tool_name}")
                    except Exception as e:
                        logger.error(f"Failed to update tool usage stats: {e}")

            db.commit()

            if (
                event_type_str == "system_update_general"
                and isinstance(data, dict)
                and data.get("checkpoint_type") == CHECKPOINT_TYPE
            ):
                self._prune_checkpoint_history(db, data)

            logger.debug(
                f"Saved trace event {event.id} of type {event_type_str} to database"
            )

        except IntegrityError as e:
            db.rollback()
            error_text = str(e)
            if (
                "trace_events_task_id_fkey" in error_text
                or "ForeignKeyViolation" in error_text
            ):
                if getattr(event, "require_persisted", False):
                    logger.error(
                        "Required trace event references missing task %s: %s",
                        self.task_id,
                        event.id,
                    )
                    raise
                logger.debug(
                    f"Skip trace event for missing task {self.task_id}: {event.id}"
                )
                return
            logger.error(f"Failed to save trace event to database: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to save trace event to database: {e}")
            db.rollback()
            raise

    def _prune_checkpoint_history(self, db: Session, data: Dict[str, Any]) -> None:
        """Drop checkpoint rows beyond the retention limit for one execution.

        Resume only reads the most recent readable checkpoint; a few older
        rows are kept so an unreadable latest can fall back. Runs in its own
        transaction after the checkpoint commit so a prune failure can never
        take the checkpoint write down with it. Blobs are not touched: most
        stay referenced by the surviving checkpoints thanks to content
        dedup, but a blob referenced only by pruned rows (e.g. a message
        later dropped by context compaction) is orphaned until whole-task
        deletion cleans it up.

        The entry guard below is a caller-contract check, not one of those
        prune failures: it fires only when a caller hands this method a
        session with uncommitted work, which the sole call site -- right
        after the checkpoint commit -- never does. It raises outside the
        degrade-and-log block deliberately, because that block's rollback
        would discard the caller's pending writes and turn a misuse bug into
        silent data loss.

        Retention is exactly ``limit`` rows, with up to two exceptions:
        protecting a row named by the task's exact-row pointer, or by an
        active interaction request's resume anchor, when that row itself
        ranks outside the window keeps up to ``limit + 2`` (one pointer per
        task, and at most one active interaction row per task -- see
        ``uq_task_interaction_active_slot``).
        """
        if db.new or db.dirty or db.deleted:
            raise RuntimeError(
                "checkpoint prune must not run with pending writes on the session"
            )
        limit = get_checkpoint_history_limit()
        if limit <= 0:
            return
        execution_id = checkpoint_execution_id(data)
        if not execution_id:
            return
        try:
            build_filter = (
                DatabaseTraceEvent.build_id == self.build_id
                if self.build_id is not None
                else DatabaseTraceEvent.build_id.is_(None)
            )
            partition_filters = [build_filter]
            if self.build_id is None:
                run_id = data.get(TASK_RUN_ID_TRACE_FIELD)
                partition_filters.append(
                    self._checkpoint_run_partition_filter(
                        run_id if isinstance(run_id, str) and run_id else None
                    )
                )
            anchor = (
                db.query(Task.last_checkpoint_trace_event_id)
                .filter(Task.id == self.task_id)
                .one_or_none()
            )
            anchor_id = anchor[0] if anchor is not None else None
            stale_rows = (
                db.query(DatabaseTraceEvent.id)
                .filter(
                    DatabaseTraceEvent.task_id == self.task_id,
                    *partition_filters,
                    DatabaseTraceEvent.event_type == "system_update_general",
                    DatabaseTraceEvent.data["checkpoint_type"]
                    .as_string()
                    .in_(sorted(READABLE_CHECKPOINT_TYPES)),
                    _checkpoint_execution_id_predicate(execution_id),
                )
                .order_by(
                    DatabaseTraceEvent.timestamp.desc(),
                    DatabaseTraceEvent.id.desc(),
                )
                .offset(limit)
                .all()
            )
            # This query completing is the proof the retention path is
            # healthy, whatever it found. Clearing only after a successful
            # delete would leave a steady state with nothing to prune unable
            # to clear the signal at all.
            clear_degradation(CHECKPOINT_PRUNE_FAILED)
            if not stale_rows:
                # Nothing ranks outside the window, so no protection set is
                # needed to decide there is nothing to delete: skip the
                # schema gate and the interaction-anchor query below.
                # stale_ids is a filtered subset of stale_rows, so an empty
                # stale_rows always reaches the same early return further
                # down. Placed after the clear above so the healthy-attempt
                # signal still fires on every write.
                return
            # Protection set, second source: a trace row anchored by an
            # ACTIVE interaction row must survive retention, or answering
            # that interaction later would find no checkpoint to resume
            # from. Only active_slot IS NOT NULL rows protect: terminal rows
            # are not resumable. Deliberately no expires_at filter -- an
            # expired request is still answerable, so its anchor must stay.
            #
            # Placed after clear_degradation above, not before: on a
            # deployment upgraded to a revision before this table exists the
            # query raises OperationalError on SQLite and ProgrammingError
            # on PostgreSQL, which would land in two different handlers and
            # -- on SQLite -- register a CHECKPOINT_PRUNE_FAILED signal on
            # every checkpoint that no operator could ever clear. The
            # has_table gate makes that unreachable; the placement keeps the
            # blast radius small if the gate is ever removed.
            interaction_anchor_ids: set[int] = set()
            if interaction_requests_table_exists(db):
                interaction_anchor_ids = {
                    row_id
                    for (row_id,) in db.query(
                        TaskInteractionRequest.resume_trace_event_id
                    ).filter(
                        TaskInteractionRequest.task_id == self.task_id,
                        TaskInteractionRequest.active_slot.isnot(None),
                        TaskInteractionRequest.resume_trace_event_id.isnot(None),
                    )
                }
            # Rank first, protect after. The row this task's exact-row
            # pointer references, and any row an active interaction request
            # anchors, are never deleted, whatever their position in the
            # retention ranking; excluding them from the candidate set
            # instead would shift every remaining row's OFFSET rank and turn
            # the retained count into an unpredictable limit + k. The
            # exact-row pointer's protection is structurally unreachable for
            # the steady-state writer -- the pointer always names the row
            # just written, which ranks ahead of the offset -- but it guards
            # the backfill-vs-prune window and back-pointing anchors that
            # point at an older row. The interaction anchor is exactly that
            # kind of back-pointing anchor, arrived: a task can keep writing
            # checkpoints after an interaction is raised, so the row the
            # interaction resumes from steadily sinks in the retention
            # ranking while the interaction stays open.
            protected_ids = interaction_anchor_ids | (
                {anchor_id} if anchor_id is not None else set()
            )
            stale_ids = [
                row_id for (row_id,) in stale_rows if row_id not in protected_ids
            ]
            if not stale_ids:
                return
            # Chunk the IN clause: a backlog from previously-disabled pruning
            # can exceed SQLite's bind-parameter limit in one statement.
            for chunk in chunks(stale_ids, SQL_IN_CLAUSE_CHUNK_SIZE):
                db.query(DatabaseTraceEvent).filter(
                    DatabaseTraceEvent.id.in_(chunk)
                ).delete(synchronize_session=False)
            db.commit()
            logger.debug(
                "Pruned %d checkpoint rows for task %s execution %s",
                len(stale_ids),
                self.task_id,
                execution_id,
            )
        except (IntegrityError, OperationalError):
            # PostgreSQL enforces the anchor FK, so a race that lets a
            # candidate row become the active anchor between selection and
            # delete surfaces here as a restrict violation (IntegrityError).
            # Two anchors can do that: the task's checkpoint pointer, and an
            # active interaction row whose resume_trace_event_id lands on a
            # candidate -- the latter trips
            # ck_task_interaction_requests_active_anchor on both backends
            # (the ON DELETE SET NULL fires and the CHECK forbids the NULL
            # on active rows). For the pointer FK, a database upgraded
            # through Alembic on SQLite has no DB-level constraint today
            # and cannot raise this -- that changes when the interaction
            # table's migration lands, whose rows enforce their CHECK on
            # upgraded SQLite too; a freshly created SQLite database
            # (create_all, e.g. tests) has the FK and can. Under stricter
            # isolation levels the same race can instead
            # surface as psycopg2's SerializationFailure or DeadlockDetected,
            # both of which SQLAlchemy wraps in OperationalError, not
            # IntegrityError -- catch both so this retention path degrades
            # the same way regardless of which one the database raises. The
            # signal is named for the outcome rather than for either cause:
            # OperationalError also covers lock timeouts and dropped
            # connections, /health publishes only the signal name, and the
            # operator's response to all of them is the same -- retention
            # stopped, rows are accumulating, read the logged traceback for
            # which one it was.
            db.rollback()
            register_degradation(
                CHECKPOINT_PRUNE_FAILED,
                f"task {self.task_id}: checkpoint prune could not delete stale rows",
            )
            logger.warning(
                "Checkpoint prune failed for task %s",
                self.task_id,
                exc_info=True,
            )
        except Exception:
            db.rollback()
            logger.warning(
                "Failed to prune checkpoint history for task %s",
                self.task_id,
                exc_info=True,
            )

    def _is_duplicate_user_message_turn(
        self,
        db: Session,
        event_type: str,
        data: Any,
    ) -> bool:
        if event_type != "user_message" or not isinstance(data, dict):
            return False
        turn_id = data.get("turn_id")
        if not isinstance(turn_id, str) or not turn_id:
            return False
        build_filter = (
            DatabaseTraceEvent.build_id == self.build_id
            if self.build_id is not None
            else DatabaseTraceEvent.build_id.is_(None)
        )
        return (
            db.query(DatabaseTraceEvent.id)
            .filter(
                DatabaseTraceEvent.task_id == self.task_id,
                build_filter,
                DatabaseTraceEvent.event_type == "user_message",
                DatabaseTraceEvent.data["turn_id"].as_string() == turn_id,
            )
            .first()
            is not None
        )

    def _serialize_data_for_json(self, data: Any) -> Any:
        """Recursively serialize data to ensure JSON compatibility and clean problematic characters."""
        import json
        from datetime import datetime

        def clean_string(value: str) -> str:
            """Clean string data to remove problematic characters for PostgreSQL JSON."""
            if not isinstance(value, str):
                return value

            # Remove NULL characters and other problematic control characters
            cleaned = value.replace("\x00", "")  # Remove NULL character
            cleaned = cleaned.replace("\u0000", "")  # Remove Unicode NULL
            # Remove other control characters that might cause issues
            cleaned = "".join(
                char for char in cleaned if ord(char) >= 32 or char in "\n\r\t"
            )
            return cleaned

        def serialize_value(value: Any) -> Any:
            # Handle Pydantic models (BaseModel)
            if hasattr(value, "model_dump"):
                # Convert Pydantic model to dict
                return serialize_value(value.model_dump())
            elif callable(getattr(value, "to_dict", None)):
                return serialize_value(value.to_dict())
            elif isinstance(value, datetime):
                if value.tzinfo is None:
                    value = value.replace(tzinfo=timezone.utc)
                return value.timestamp()
            elif isinstance(value, str):
                return clean_string(value)
            elif isinstance(value, dict):
                return {k: serialize_value(v) for k, v in value.items()}
            elif isinstance(value, (list, tuple)):
                return [serialize_value(item) for item in value]
            elif isinstance(value, bytes):
                # Convert bytes to string, cleaning problematic characters
                try:
                    decoded = value.decode("utf-8")
                    return clean_string(decoded)
                except UnicodeDecodeError:
                    # If decode fails, use safe representation
                    return f"<bytes: {len(value)}>"
            else:
                return value

        try:
            # First clean and serialize the data
            cleaned_data = serialize_value(data)

            # Test if cleaned data is JSON serializable
            json.dumps(cleaned_data)
            return cleaned_data
        except (TypeError, ValueError) as e:
            # If still not serializable, log the error and return a safe fallback
            logger.warning(
                f"Failed to serialize data for JSON: {e}, data type: {type(data)}"
            )
            return {
                "_serialization_error": f"Failed to serialize {type(data).__name__}",
                "_original_type": type(data).__name__,
                "_error": str(e),
            }
