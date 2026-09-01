"""Task execution leases for multi-process agent runners."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Iterator, TypeVar, cast

from sqlalchemy import and_, case, func, or_, update
from sqlalchemy.exc import MultipleResultsFound
from sqlalchemy.orm import Query, Session

from ...config import (
    get_task_lease_heartbeat_seconds,
    get_task_lease_ttl_seconds,
)
from ..models.task import Task, TaskStatus, TraceEvent, task_status_predicate
from .db_runtime import (
    await_task_settlement,
    cancel_and_drain_async_task,
    drain_async_task_cancellation_safe,
    is_database_pool_timeout,
    run_db_io_cancellation_safe,
)
from .ops_signals import (
    CHECKPOINT_LEGACY_POINTER_AMBIGUOUS,
    register_degradation,
)
from .task_execution_controller import control_state_for_status

logger = logging.getLogger(__name__)

_RUNNER_ID = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:12]}"
TASK_RUN_ID_TRACE_FIELD = "_task_run_id"
_T = TypeVar("_T")


def _rowcount(result: Any) -> int:
    return int(getattr(result, "rowcount", 0) or 0)


@dataclass(frozen=True)
class TaskLease:
    """One runner's claim on one task row.

    ``attempt_id`` names the exact acquisition that produced this lease. It
    is minted fresh by ``acquire_task_lease_no_commit`` on every successful
    claim and mirrored into ``tasks.lease_attempt_id`` in the same UPDATE, so
    a holder can later prove the row still belongs to *its* attempt rather
    than to a successor that reused the same ``(runner_id, run_id)`` tuple.

    ``attempt_id is None`` is a fail-open sentinel with two distinct sources:

    * a lease acquired by a worker running code from before this column
      existed -- a rolling-restart window that closes on its own once every
      worker has restarted; and
    * ``_task_lease_snapshot`` in ``websocket.py``, which reconstructs an
      ambient lease from an already-loaded task row. That one carries None
      permanently and on purpose: every one of its fields is read from the
      task row, so populating ``attempt_id`` from the same row would make any
      later ``task.lease_attempt_id != lease.attempt_id`` check compare a
      value against itself -- an always-passing fence, which is strictly
      worse than an explicit None that callers can detect and skip. Today
      that snapshot only feeds checkpoint fencing via
      ``bind_task_lease_context`` and never reaches a settlement path, so
      the permanent None also has no behavioural cost.

    Consumers must therefore treat None as "cannot prove attempt identity"
    and fall back to whatever fence they already had, never as "matches".

    The existing lease writers (refresh, release, fail-and-release) do not
    fence on this column yet, deliberately: rejecting a stale attempt there
    turns its refresh into a LOST classification and an active cancellation,
    which is a behavior change that belongs to the first consumer of this
    column, with its own tests. Until then the pre-existing
    ``(runner_id, run_id)`` fences are the only guards on those paths.
    """

    task_id: int
    runner_id: str
    run_id: str | None = None
    attempt_id: str | None = None


# ---------------------------------------------------------------------------
# Fence predicates. Three plain booleans over a task row and the lease that
# claims it, shared by every caller that has to prove a settlement still
# belongs to the run that produced it.
#
# Only the boolean is shared. What a caller does when one of these is False
# stays at the caller: the clarification resolver classifies each into its
# own fail-closed reason, the interaction handoff raises, and the WebSocket
# finalizers report a late result. Those three consequences are genuinely
# different and must not be folded together here.
#
# These are plain Python comparisons against an already-loaded task row.
# They are NOT the shape the WebSocket finalizers use: those compile the
# same ownership condition into the WHERE clause of a locking SELECT, so
# "no row came back" is what tells them ownership changed, and the row lock
# they take is scoped by that same condition. Rewriting them to load a row
# and then call the predicate below would move the lock and change what a
# late result means, so they keep their own form. That the two forms agree
# is asserted statically, in
# tests/web/services/test_interaction_fence_equivalence.py.
# ---------------------------------------------------------------------------


def lease_is_fenced(lease: TaskLease) -> bool:
    """Whether ``lease`` can name the run that produced a result at all.

    A lease with no ``run_id`` predates run fencing; it cannot prove which
    run a result came from, so a caller holding one may not settle
    anything that depends on run identity.
    """

    return lease.run_id is not None


def task_row_matches_lease_owner(task: Task, lease: TaskLease) -> bool:
    """Whether ``task`` is still owned by the runner and run in ``lease``.

    Both fields are compared, never one: a successor runner can reuse a
    ``run_id`` and a single runner can move between runs, so either field
    alone leaves a real ownership change looking like a match.
    """

    return bool(task.runner_id == lease.runner_id and task.run_id == lease.run_id)


def task_row_matches_lease_attempt(task: Task, lease: TaskLease) -> bool:
    """Whether ``task``'s current attempt is the one ``lease`` names.

    ``lease.attempt_id is None`` returns True. That is a fail-open
    sentinel and it is deliberate: ``None`` means this lease cannot prove
    attempt identity at all (a pre-attempt-column lease, or the
    permanently-``None`` ambient snapshot ``_task_lease_snapshot`` builds
    in ``websocket.py`` -- see ``TaskLease.attempt_id`` for both sources),
    and the established reading is "skip this check", never "treat it as a
    mismatch". A caller that needs attempt identity proven rather than
    merely not-contradicted has to check ``lease.attempt_id is not None``
    itself; this predicate does not make that distinction for it.

    This is a plain Python object comparison, not a SQL expression: the
    ``== None`` -> ``IS NULL`` folding that affects a SQLAlchemy column
    comparison compiled to SQL does not apply here.
    """

    return bool(lease.attempt_id is None or task.lease_attempt_id == lease.attempt_id)


_CURRENT_TASK_LEASE: ContextVar[TaskLease | None] = ContextVar(
    "xagent_current_task_lease",
    default=None,
)


class TaskLeaseLostError(RuntimeError):
    """Raised after the current runner definitively loses task ownership."""


@contextmanager
def bind_task_lease_context(lease: TaskLease) -> Iterator[None]:
    """Bind one exact run/runner lease to all work spawned by this context."""

    if lease.run_id is None:
        raise ValueError("task lease context requires an exact run_id fence")
    token = _CURRENT_TASK_LEASE.set(lease)
    try:
        yield
    finally:
        _CURRENT_TASK_LEASE.reset(token)


def current_task_lease() -> TaskLease | None:
    """Return the exact lease bound to the current execution context."""

    return _CURRENT_TASK_LEASE.get()


@dataclass(frozen=True)
class TaskLeaseRecoveryCandidate:
    """Immutable ownership snapshot for one expired task lease."""

    task_id: int
    runner_id: str | None
    run_id: str | None
    lease_expires_at: datetime
    state_version: int
    last_checkpoint_event_id: str | None
    last_checkpoint_trace_event_id: int | None

    @property
    def cursor(self) -> tuple[datetime, int]:
        return self.lease_expires_at, self.task_id


@dataclass(frozen=True)
class TaskLeaseHeartbeatOutcome:
    """Lease state observed when a heartbeat loop stops.

    A shared batch pool timeout is retained for each registration in that
    batch until a later successful refresh proves ownership is healthy again.
    Callers use an unresolved timeout (or definite ownership loss) to avoid
    starting a second settlement checkout while the pool is exhausted.
    """

    lease_lost: bool = False
    pool_timeout: BaseException | None = None

    @property
    def requires_ttl_recovery(self) -> bool:
        return self.lease_lost or self.pool_timeout is not None


class TaskLeaseRefreshState(str, Enum):
    """Result of refreshing one exact task-run lease."""

    REFRESHED = "refreshed"
    SETTLEMENT_READY = "settlement_ready"
    LOST = "lost"


TaskLeaseKey = tuple[int, str, str | None]


def _task_lease_key(lease: TaskLease) -> TaskLeaseKey:
    return lease.task_id, lease.runner_id, lease.run_id


async def run_while_task_lease_owned(
    operation: Coroutine[Any, Any, _T],
    heartbeat_task: asyncio.Task[TaskLeaseHeartbeatOutcome],
) -> _T:
    """Run one operation until completion or definitive lease ownership loss.

    The heartbeat remains owned by the caller and is not stopped when the
    operation completes. A transient pool timeout does not complete the
    heartbeat's terminal waiter, so it never cancels execution here. Only a
    structured ``lease_lost`` result cancels and fully drains the operation.
    """

    operation_task = asyncio.create_task(operation)
    try:
        done, _ = await asyncio.wait(
            {operation_task, heartbeat_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if heartbeat_task in done:
            try:
                outcome = heartbeat_task.result()
            except BaseException:
                # The guard owns the child operation. Never let a crashed or
                # externally-cancelled heartbeat orphan it in the background.
                await cancel_and_drain_async_task(operation_task)
                raise
            if isinstance(outcome, TaskLeaseHeartbeatOutcome) and outcome.lease_lost:
                await cancel_and_drain_async_task(operation_task)
                raise TaskLeaseLostError(
                    "task execution stopped after lease ownership was lost"
                )
        return await operation_task
    except asyncio.CancelledError:
        await cancel_and_drain_async_task(operation_task)
        raise


async def acquire_task_lease_cancellation_safe(
    acquire: Callable[[], TaskLease | None],
    cleanup: Callable[[TaskLease], Any],
) -> TaskLease | None:
    """Acquire a lease and clean up a late result before propagating cancel.

    The acquisition callback and cleanup callback each execute in their own
    worker thread. When cancellation arrives during acquisition, the acquire
    worker is drained first; if it committed and returned a lease, cleanup is
    then drained as well. Only after both operations settle is cancellation
    delivered to the caller.
    """
    worker = asyncio.get_running_loop().create_task(asyncio.to_thread(acquire))
    lease, cancellation = await await_task_settlement(worker)
    if cancellation is None:
        return lease

    if lease is not None:
        try:
            await run_db_io_cancellation_safe(lambda: cleanup(lease))
        except asyncio.CancelledError:
            # A repeated caller cancellation was recorded and propagated only
            # after cleanup settled. Preserve the original cancellation below.
            pass
        except Exception:
            logger.exception(
                "Failed to clean up task %s lease after cancelled acquisition",
                lease.task_id,
            )
    raise cancellation


def get_runner_id() -> str:
    """Return the current process runner id."""
    return _RUNNER_ID


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def task_lease_expires_at(now: datetime | None = None) -> datetime:
    return (now or utc_now()) + timedelta(seconds=get_task_lease_ttl_seconds())


class CheckpointRecoveryVerdict(str, Enum):
    """Result of resolving one recovery candidate's checkpoint pointer.

    ``RECOVERABLE`` and ``NOT_RECOVERABLE`` both mean the checkpoint's row
    identity was authoritatively resolved (found-and-valid, or found-invalid,
    or provably absent); the caller acts on the candidate immediately,
    recovering to PAUSED or FAILED respectively. ``INDETERMINATE`` means row
    identity itself could not be established this round (an ambiguous legacy
    match) -- the caller must leave the candidate's lease and status
    untouched so the next sweep can retry, never fold this into FAILED.
    """

    RECOVERABLE = "recoverable"
    NOT_RECOVERABLE = "not_recoverable"
    INDETERMINATE = "indeterminate"


def _checkpoint_row_matches_candidate(
    row: TraceEvent, candidate: TaskLeaseRecoveryCandidate
) -> bool:
    """Validate an already-identified trace_events row against the fields a
    recoverable checkpoint must satisfy for this candidate's run.

    Shared by both the PK-anchored and legacy resolution paths: the legacy
    path's query already filters task_id/build_id/event_type, so those
    checks are redundant there, but the PK path loads a row by raw primary
    key with no such filter, so it needs the full set. A mismatch here is
    corruption of the row the pointer names, not a cue to search elsewhere.

    A ``False`` here always resolves the candidate to NOT_RECOVERABLE
    (lease recovery fails the task). The conditions themselves are no longer
    written out here: this function and both by-primary-key read paths read
    one predicate (``failed_checkpoint_row_conditions``,
    ``trace_event_staging.py``), so there is nothing left for the three to
    disagree about. Two rules stay this function's own and are applied
    around that predicate rather than inside it: a candidate with no
    ``run_id`` fails closed before the predicate is consulted, and the
    missing-run-partition reclassification is handled by the caller
    (``resolve_checkpoint_recovery``) rather than here, because only the
    exact-pointer path has a second candidate set to defer to.

    The vocabularies still differ -- this path calls the outcome a mismatch
    and the reader calls it corrupt (``CheckpointCorruptError``) -- and that
    difference is still deliberate, because each names the outcome for its
    own caller. What is no longer true is that the two could drift on *what*
    they judge.
    """

    if candidate.run_id is None:
        return False
    _row_data, failed = _candidate_row_failures(row, candidate)
    return not failed


def _candidate_row_failures(
    row: TraceEvent, candidate: TaskLeaseRecoveryCandidate
) -> tuple[dict[str, Any], frozenset[str]]:
    """The shared row-validity conditions ``row`` fails for this candidate,
    with the normalized payload the caller needs alongside it.

    Imported inside the function on purpose: ``trace_event_staging`` already
    imports this module (``TASK_RUN_ID_TRACE_FIELD``, ``TaskLease``), so a
    module-level import in this direction is a cycle, not a style choice.

    ``candidate.run_id is None`` never reaches here. That case is this
    module's own terminal outcome -- an exact pointer alone cannot prove a
    checkpoint belongs to the expired run (see ``resolve_checkpoint_recovery``)
    -- and the shared predicate deliberately does not encode it: a ``None``
    ``run_id`` is a legitimate partition there, matched by a row whose own
    run field is also absent. Both callers fail closed on it first, which is
    exactly what that predicate's docstring tells callers with a terminal
    ``run_id`` rule of their own to do.

    ``execution_id`` is ``str(candidate.task_id)`` because web's execution id
    *is* the task id -- the same value the other two callers pass
    (``_load_pk_anchored_checkpoint``, ``resolve_interaction_anchor``).
    """

    from .trace_event_staging import failed_checkpoint_row_conditions

    data: dict[str, Any] = (
        cast(dict[str, Any], row.data) if isinstance(row.data, dict) else {}
    )
    return data, failed_checkpoint_row_conditions(
        row,
        data,
        task_id=candidate.task_id,
        run_id=candidate.run_id,
        execution_id=str(candidate.task_id),
    )


def _resolve_legacy_checkpoint_recovery(
    db: Session,
    candidate: TaskLeaseRecoveryCandidate,
) -> CheckpointRecoveryVerdict:
    """Resolve the candidate's legacy ``event_id`` string against the row it
    names, within the same partition (task, root scope, checkpoint type)
    today's query has always used.

    An unset pointer or a zero-row match is an authoritative absence: the
    candidate's run has no checkpoint to recover to, so this is
    NOT_RECOVERABLE (FAILED), not INDETERMINATE. Only two-or-more rows
    sharing the same event_id inside this partition make the row's identity
    itself ambiguous -- that is the case this resolver cannot decide and
    defers to the next sweep instead of guessing or failing the task.
    """

    if candidate.last_checkpoint_event_id is None:
        return CheckpointRecoveryVerdict.NOT_RECOVERABLE
    query = db.query(TraceEvent).filter(
        TraceEvent.task_id == candidate.task_id,
        TraceEvent.build_id.is_(None),
        TraceEvent.event_id == candidate.last_checkpoint_event_id,
        TraceEvent.event_type == "system_update_general",
    )
    try:
        row = query.one_or_none()
    except MultipleResultsFound:
        logger.warning(
            "Task %s legacy checkpoint event_id %s matches more than one "
            "trace_events row; skipping this recovery candidate for the "
            "next sweep",
            candidate.task_id,
            candidate.last_checkpoint_event_id,
        )
        # An ambiguity nothing in this process can resolve: every later
        # sweep re-selects the same candidate and re-hits it. A log line
        # alone leaves that invisible to monitoring, so it rides /health
        # until a sweep completes without seeing one.
        register_degradation(
            CHECKPOINT_LEGACY_POINTER_AMBIGUOUS,
            f"task {candidate.task_id}: legacy checkpoint event_id "
            f"{candidate.last_checkpoint_event_id} matches more than one row",
        )
        return CheckpointRecoveryVerdict.INDETERMINATE
    if row is None:
        return CheckpointRecoveryVerdict.NOT_RECOVERABLE
    return (
        CheckpointRecoveryVerdict.RECOVERABLE
        if _checkpoint_row_matches_candidate(row, candidate)
        else CheckpointRecoveryVerdict.NOT_RECOVERABLE
    )


def resolve_checkpoint_recovery(
    db: Session,
    candidate: TaskLeaseRecoveryCandidate,
) -> CheckpointRecoveryVerdict:
    """Resolve whether the candidate's checkpoint pointer identifies a
    recoverable checkpoint for its current run.

    Recovery fails closed for events written before run provenance was
    introduced: an exact pointer alone cannot prove that the checkpoint
    belongs to the expired run. New-run claims clear both pointer columns
    before execution starts.

    The exact-row pointer is tried first when set: a hit is authoritative
    (validated, never searched past). A pointer whose row is gone is not a
    validation failure -- it is only reachable on a database upgraded
    without this column's FK (a compatibility-window state, see the
    migration) -- so it falls back to the legacy string resolution below
    instead of failing the candidate outright.
    """

    if candidate.last_checkpoint_trace_event_id is not None:
        row = db.get(TraceEvent, candidate.last_checkpoint_trace_event_id)
        if row is not None:
            from .trace_event_staging import is_missing_run_partition_only

            if candidate.run_id is None:
                return CheckpointRecoveryVerdict.NOT_RECOVERABLE
            row_data, failed = _candidate_row_failures(row, candidate)
            if not failed:
                return CheckpointRecoveryVerdict.RECOVERABLE
            if not is_missing_run_partition_only(failed, row_data):
                return CheckpointRecoveryVerdict.NOT_RECOVERABLE
            # A pre-existing row, not a mismatched one: the 20260804 backfill
            # can point this column at a trace_events row written before the
            # run-partition field existed. Treating it as "this pointer did
            # not answer" rather than "this pointer named a bad row" is the
            # same verdict the two by-primary-key read paths reach for this
            # exact shape, and it reaches the legacy scan the same way a
            # dangling pointer does. Nothing is loosened: that scan validates
            # whatever row it finds through the same predicate, so a
            # RECOVERABLE verdict still requires a real run-partition match.
            logger.info(
                "Task %s's checkpoint pointer %s is missing its "
                "run-partition field; deferring to the legacy event_id scan "
                "rather than treating the row as a mismatch",
                candidate.task_id,
                candidate.last_checkpoint_trace_event_id,
            )
        else:
            logger.warning(
                "Task %s checkpoint pointer trace_event_id=%s has no matching "
                "trace_events row; falling back to the legacy event_id scan",
                candidate.task_id,
                candidate.last_checkpoint_trace_event_id,
            )
    return _resolve_legacy_checkpoint_recovery(db, candidate)


def _nullable_match(column: Any, value: Any) -> Any:
    return column.is_(None) if value is None else column == value


def lease_run_id_case(candidate_run_id: str) -> Any:
    """SET run_id for a lease acquisition: keep the existing run unless the
    row is not RUNNING."""
    return case(
        (task_status_predicate.ne(TaskStatus.RUNNING), candidate_run_id),
        else_=func.coalesce(Task.run_id, candidate_run_id),
    )


def lease_state_version_case(
    status: TaskStatus, control_state: str, current_version: Any
) -> Any:
    """SET state_version: bump only when this write actually changes the
    row's (status, control_state) pair."""
    return case(
        (
            or_(
                task_status_predicate.ne(status),
                Task.control_state != control_state,
            ),
            current_version + 1,
        ),
        else_=current_version,
    )


def _checkpoint_pointer_clearing_predicate() -> Any:
    """WHERE condition shared by both checkpoint-pointer SET case() builders:
    true when the row is not a live RUNNING row with a run id, i.e. when
    both pointer columns must be cleared.

    Extracted so the legacy and exact-row builders below can never drift
    apart on which rows clear their pointer -- a per-builder copy would let
    one column retain a stale pointer the other clears, which would make
    the recovery CAS fence's conjunction over both columns never match (see
    test_lease_checkpoint_pointer_case_predicates_match in
    test_task_lease_service.py).
    """
    return or_(
        task_status_predicate.ne(TaskStatus.RUNNING),
        Task.run_id.is_(None),
    )


def lease_checkpoint_event_id_case() -> Any:
    """SET last_checkpoint_event_id: clear it when the row is not a live
    RUNNING row with a run id."""
    return case(
        (_checkpoint_pointer_clearing_predicate(), None),
        else_=Task.last_checkpoint_event_id,
    )


def lease_checkpoint_trace_event_id_case() -> Any:
    """SET last_checkpoint_trace_event_id: mirrors
    lease_checkpoint_event_id_case's clearing condition exactly, so the two
    pointer columns are always set or cleared together."""
    return case(
        (_checkpoint_pointer_clearing_predicate(), None),
        else_=Task.last_checkpoint_trace_event_id,
    )


def _expired_task_lease_candidates_query(
    db: Session,
    *,
    cutoff: datetime,
    after: tuple[datetime, int] | None = None,
) -> Query[Any]:
    """Build the shared ordered query for expired RUNNING lease snapshots."""

    query = (
        db.query(
            Task.id,
            Task.runner_id,
            Task.run_id,
            Task.lease_expires_at,
            Task.state_version,
            Task.last_checkpoint_event_id,
            Task.last_checkpoint_trace_event_id,
        )
        .filter(
            task_status_predicate.eq(TaskStatus.RUNNING),
            Task.lease_expires_at.is_not(None),
            Task.lease_expires_at < cutoff,
        )
        .order_by(Task.lease_expires_at.asc(), Task.id.asc())
    )
    if after is not None:
        after_expiry, after_task_id = after
        query = query.filter(
            or_(
                Task.lease_expires_at > after_expiry,
                and_(
                    Task.lease_expires_at == after_expiry,
                    Task.id > after_task_id,
                ),
            )
        )
    return query


def _task_lease_recovery_candidate_from_row(
    row: Any,
) -> TaskLeaseRecoveryCandidate | None:
    lease_expires_at = row.lease_expires_at
    if lease_expires_at is None:
        return None
    return TaskLeaseRecoveryCandidate(
        task_id=int(row.id),
        runner_id=str(row.runner_id) if row.runner_id is not None else None,
        run_id=str(row.run_id) if row.run_id is not None else None,
        lease_expires_at=lease_expires_at,
        state_version=int(row.state_version or 0),
        last_checkpoint_event_id=(
            str(row.last_checkpoint_event_id)
            if row.last_checkpoint_event_id is not None
            else None
        ),
        last_checkpoint_trace_event_id=(
            int(row.last_checkpoint_trace_event_id)
            if row.last_checkpoint_trace_event_id is not None
            else None
        ),
    )


def get_expired_task_lease_candidates(
    db: Session,
    *,
    cutoff: datetime,
    limit: int,
    after: tuple[datetime, int] | None = None,
) -> tuple[TaskLeaseRecoveryCandidate, ...]:
    """Load one ordered, bounded page of expired RUNNING lease snapshots."""

    if limit < 1:
        raise ValueError("limit must be positive")

    candidates: list[TaskLeaseRecoveryCandidate] = []
    query = _expired_task_lease_candidates_query(
        db,
        cutoff=cutoff,
        after=after,
    )
    for row in query.limit(limit).all():
        candidate = _task_lease_recovery_candidate_from_row(row)
        if candidate is not None:
            candidates.append(candidate)
    return tuple(candidates)


def get_next_expired_task_lease_candidate_for_update(
    db: Session,
    *,
    cutoff: datetime,
    after: tuple[datetime, int] | None = None,
) -> TaskLeaseRecoveryCandidate | None:
    """Lock and return one expired candidate without waiting on peer workers.

    This entry point is owned by the PostgreSQL recovery path. Keeping the
    row lock and recovery mutation in the same short transaction partitions
    candidates across backend workers. SQLite callers must continue to rely on
    the exact compare-and-swap recovery fence because SQLite ignores
    ``SELECT ... FOR UPDATE``.
    """

    row = (
        _expired_task_lease_candidates_query(
            db,
            cutoff=cutoff,
            after=after,
        )
        .with_for_update(of=Task, skip_locked=True)
        .first()
    )
    if row is None:
        return None
    return _task_lease_recovery_candidate_from_row(row)


def recover_expired_task_lease_no_commit(
    db: Session,
    candidate: TaskLeaseRecoveryCandidate,
    *,
    status: TaskStatus,
    recovered_at: datetime,
    error_message: str | None,
) -> bool:
    """Stage one fully fenced expired-lease transition without committing."""

    if status not in {TaskStatus.PAUSED, TaskStatus.FAILED}:
        raise ValueError("expired task leases can only recover to PAUSED or FAILED")

    values: dict[str, Any] = {
        "status": task_status_predicate.value(status),
        "runner_id": None,
        "lease_expires_at": None,
        "last_heartbeat_at": recovered_at,
        "control_state": control_state_for_status(status).value,
        "state_version": func.coalesce(Task.state_version, 0) + 1,
        "error_message": error_message,
        # Defence in depth: the column must never outlive the attempt that
        # wrote it. This writer already NULLs runner_id, so no live fence
        # depends on the value -- clearing it keeps the column honest for
        # anyone who reads it directly.
        "lease_attempt_id": None,
    }
    if status == TaskStatus.FAILED:
        values["output"] = None

    statement = (
        update(Task)
        .where(
            Task.id == candidate.task_id,
            task_status_predicate.eq(TaskStatus.RUNNING),
            _nullable_match(Task.runner_id, candidate.runner_id),
            _nullable_match(Task.run_id, candidate.run_id),
            Task.lease_expires_at == candidate.lease_expires_at,
            Task.lease_expires_at < recovered_at,
            Task.state_version == candidate.state_version,
            _nullable_match(
                Task.last_checkpoint_event_id,
                candidate.last_checkpoint_event_id,
            ),
            _nullable_match(
                Task.last_checkpoint_trace_event_id,
                candidate.last_checkpoint_trace_event_id,
            ),
        )
        .values(**values)
    )
    result = db.execute(statement.execution_options(synchronize_session=False))
    return _rowcount(result) == 1


def acquire_task_lease(
    db: Session,
    task_id: int,
    *,
    runner_id: str | None = None,
    expected_run_id: str | None = None,
    new_run: bool = False,
) -> TaskLease | None:
    """Acquire the task execution lease if no live runner owns it.

    ``new_run=True`` atomically requires a non-running task and assigns a new
    run id in the same UPDATE. This is the durable claim used by channel
    transports so another worker cannot rotate the run between a status check
    and lease acquisition.
    """
    lease = acquire_task_lease_no_commit(
        db,
        task_id,
        runner_id=runner_id,
        expected_run_id=expected_run_id,
        new_run=new_run,
    )
    db.commit()
    if lease is None:
        logger.info(
            "Task %s lease acquisition denied for runner %s",
            task_id,
            runner_id or get_runner_id(),
        )
        return None
    logger.info(
        "Task %s lease acquired by runner %s",
        task_id,
        lease.runner_id,
    )
    return lease


def acquire_task_lease_no_commit(
    db: Session,
    task_id: int,
    *,
    runner_id: str | None = None,
    expected_run_id: str | None = None,
    new_run: bool = False,
) -> TaskLease | None:
    """Stage one atomic lease claim; the caller owns commit/rollback."""
    runner = runner_id or get_runner_id()
    now = utc_now()
    expires_at = task_lease_expires_at(now)
    candidate_run_id = expected_run_id or str(uuid.uuid4())
    attempt_id = str(uuid.uuid4())
    current_version = func.coalesce(Task.state_version, 0)
    running_control_state = control_state_for_status(TaskStatus.RUNNING).value
    values: dict[str, Any] = {
        "status": task_status_predicate.value(TaskStatus.RUNNING),
        "runner_id": runner,
        "last_heartbeat_at": now,
        "lease_expires_at": expires_at,
        "run_id": (
            candidate_run_id if new_run else lease_run_id_case(candidate_run_id)
        ),
        "control_state": running_control_state,
        "state_version": lease_state_version_case(
            TaskStatus.RUNNING, running_control_state, current_version
        ),
        # A fresh identity for this exact claim. Every successful acquisition
        # mints a new one; no other writer ever assigns a value here (the
        # release/recovery writers only clear it). That is precisely what
        # state_version cannot offer -- _apply_pause_requested_isolated bumps
        # state_version mid-run without changing attempt ownership, so a
        # state_version fence pinned at acquisition would reject a legitimate
        # finalizer that had been paused.
        "lease_attempt_id": attempt_id,
    }
    if new_run:
        values["last_checkpoint_event_id"] = None
        values["last_checkpoint_trace_event_id"] = None
        values["output"] = None
        values["error_message"] = None
    elif expected_run_id is None:
        values["last_checkpoint_event_id"] = lease_checkpoint_event_id_case()
        values["last_checkpoint_trace_event_id"] = (
            lease_checkpoint_trace_event_id_case()
        )

    stmt = (
        update(Task)
        .where(Task.id == task_id)
        .where(
            or_(
                task_status_predicate.ne(TaskStatus.RUNNING),
                Task.runner_id == runner,
                Task.runner_id.is_(None),
                Task.lease_expires_at.is_(None),
                Task.lease_expires_at < now,
            )
        )
        .values(**values)
    )
    if new_run:
        stmt = stmt.where(task_status_predicate.ne(TaskStatus.RUNNING))
    if expected_run_id is not None:
        stmt = stmt.where(Task.run_id == expected_run_id)
    result = db.execute(
        stmt.returning(Task.run_id).execution_options(synchronize_session=False)
    )
    stored_run_id = result.scalar_one_or_none()
    if stored_run_id is None:
        return None
    return TaskLease(
        task_id=task_id,
        runner_id=runner,
        run_id=str(stored_run_id) if stored_run_id is not None else None,
        attempt_id=values["lease_attempt_id"],
    )


def acquire_task_lease_isolated(
    task_id: int,
    *,
    runner_id: str | None = None,
    expected_run_id: str | None = None,
    new_run: bool = False,
) -> TaskLease | None:
    """Same semantics as :func:`acquire_task_lease` but opens, commits,
    and closes its own ``SessionLocal``.

    Safe to call from ``asyncio.to_thread`` -- the inline call in
    ``_runner`` measured 3.75s of synchronous DB write on the main
    event loop (issue #427). Wrapping the existing helper preserves
    every transactional detail (the conditional UPDATE + rowcount
    guard) while letting the loop continue.
    """
    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        return acquire_task_lease(
            db,
            task_id,
            runner_id=runner_id,
            expected_run_id=expected_run_id,
            new_run=new_run,
        )
    finally:
        db.close()


def _refresh_task_lease_no_commit(
    db: Session,
    lease: TaskLease,
    *,
    now: datetime,
    expires_at: datetime,
) -> TaskLeaseRefreshState:
    """Stage one exact lease refresh without ending the caller's transaction."""

    stmt = (
        update(Task)
        .where(Task.id == lease.task_id)
        .where(Task.runner_id == lease.runner_id)
        .where(task_status_predicate.eq(TaskStatus.RUNNING))
        .values(last_heartbeat_at=now, lease_expires_at=expires_at)
    )
    if lease.run_id is not None:
        stmt = stmt.where(Task.run_id == lease.run_id)
    result = db.execute(stmt.execution_options(synchronize_session=False))
    if _rowcount(result) == 1:
        return TaskLeaseRefreshState.REFRESHED

    owned_query = db.query(Task.status).filter(
        Task.id == lease.task_id,
        Task.runner_id == lease.runner_id,
    )
    if lease.run_id is not None:
        owned_query = owned_query.filter(Task.run_id == lease.run_id)
    owned_status = owned_query.scalar()
    if owned_status is not None and owned_status != TaskStatus.RUNNING:
        return TaskLeaseRefreshState.SETTLEMENT_READY
    return TaskLeaseRefreshState.LOST


def refresh_task_lease(db: Session, lease: TaskLease) -> TaskLeaseRefreshState:
    """Refresh one exact lease or classify why it no longer needs refresh.

    A task finalizer commits its terminal status before post-result broadcasts
    complete, while the scheduler still owns and later releases the lease. A
    heartbeat in that window must distinguish the same terminal run from a
    lease that another runner or run actually replaced.
    """
    now = utc_now()
    state = _refresh_task_lease_no_commit(
        db,
        lease,
        now=now,
        expires_at=task_lease_expires_at(now),
    )
    db.commit()
    return state


def refresh_task_leases_isolated(
    leases: tuple[TaskLease, ...],
) -> dict[TaskLeaseKey, TaskLeaseRefreshState]:
    """Refresh a process-local lease batch with one Session and one checkout.

    Individual task heartbeat coroutines register with the shared manager
    below. The manager snapshots all live registrations and invokes this
    helper once per interval, bounding pool contention to one heartbeat
    waiter per backend process instead of one waiter per active task.
    """
    if not leases:
        return {}

    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    now = utc_now()
    expires_at = task_lease_expires_at(now)
    with SessionLocal() as db:
        try:
            states = {
                _task_lease_key(lease): _refresh_task_lease_no_commit(
                    db,
                    lease,
                    now=now,
                    expires_at=expires_at,
                )
                for lease in leases
            }
            db.commit()
            return states
        except Exception:
            db.rollback()
            raise


def validate_preacquired_task_lease_isolated(
    lease: TaskLease,
) -> TaskLeaseRefreshState:
    """Refresh and validate an exact lease committed before local scheduling."""

    states = refresh_task_leases_isolated((lease,))
    return states.get(_task_lease_key(lease), TaskLeaseRefreshState.LOST)


_NON_TERMINAL_RELEASE_STATUSES = frozenset(
    {TaskStatus.PAUSED, TaskStatus.WAITING_FOR_USER}
)


def release_task_lease(
    db: Session,
    lease: TaskLease | None,
    *,
    status: TaskStatus,
) -> bool:
    """Release a task lease and set its final visible status."""
    released = release_task_lease_no_commit(db, lease, status=status)
    if lease is None:
        return False
    db.commit()
    return released


def release_task_lease_no_commit(
    db: Session,
    lease: TaskLease | None,
    *,
    status: TaskStatus,
) -> bool:
    """Stage release of one exact lease; the caller owns commit/rollback.

    Releasing to a non-terminal resting state also clears ``error_message``:
    the row is healthy again, and a message left by an earlier failed attempt
    would otherwise keep surfacing to clients. ``output`` is left alone --
    only terminal transitions own it (see
    ``fail_and_release_task_lease_no_commit`` and the FAILED branch of
    ``recover_expired_task_lease_no_commit``).
    """
    if status == TaskStatus.RUNNING:
        raise ValueError("Cannot release a task lease with RUNNING status")
    if lease is None:
        return False
    control_state = control_state_for_status(status).value
    current_version = func.coalesce(Task.state_version, 0)
    values: dict[str, Any] = {
        "status": task_status_predicate.value(status),
        "runner_id": None,
        "lease_expires_at": None,
        "last_heartbeat_at": utc_now(),
        "control_state": control_state,
        "state_version": lease_state_version_case(
            status, control_state, current_version
        ),
        "lease_attempt_id": None,
    }
    if status in _NON_TERMINAL_RELEASE_STATUSES:
        values["error_message"] = None
    stmt = (
        update(Task)
        .where(Task.id == lease.task_id)
        .where(Task.runner_id == lease.runner_id)
        .values(**values)
    )
    if lease.run_id is not None:
        stmt = stmt.where(Task.run_id == lease.run_id)
    result = db.execute(stmt.execution_options(synchronize_session=False))
    return _rowcount(result) == 1


def fail_and_release_task_lease_no_commit(
    db: Session,
    lease: TaskLease,
    *,
    error_message: str,
) -> bool:
    """Atomically fail and release the exact live lease without committing.

    The caller owns the transaction so related lifecycle projections can be
    synchronized before one final commit. A lease without a run id is not a
    sufficient ownership fence and is therefore never allowed to mutate the
    task row.
    """
    if lease.run_id is None:
        return False

    failed_control_state = control_state_for_status(TaskStatus.FAILED).value
    stmt = (
        update(Task)
        .where(Task.id == lease.task_id)
        .where(Task.runner_id == lease.runner_id)
        .where(Task.run_id == lease.run_id)
        .where(task_status_predicate.eq(TaskStatus.RUNNING))
        .values(
            status=task_status_predicate.value(TaskStatus.FAILED),
            runner_id=None,
            lease_expires_at=None,
            last_heartbeat_at=utc_now(),
            control_state=failed_control_state,
            state_version=func.coalesce(Task.state_version, 0) + 1,
            error_message=error_message,
            output=None,
            lease_attempt_id=None,
        )
    )
    result = db.execute(stmt.execution_options(synchronize_session=False))
    return _rowcount(result) == 1


def release_current_runner_task_lease(
    db: Session,
    task_id: int,
    *,
    status: TaskStatus,
    runner_id: str | None = None,
    expected_run_id: str | None = None,
) -> bool:
    """Release the current runner's lease for a task."""
    if status == TaskStatus.RUNNING:
        raise ValueError("Cannot release a task lease with RUNNING status")
    runner = runner_id or get_runner_id()
    control_state = control_state_for_status(status).value
    current_version = func.coalesce(Task.state_version, 0)
    stmt = (
        update(Task)
        .where(Task.id == task_id)
        .where(Task.runner_id == runner)
        .values(
            status=task_status_predicate.value(status),
            runner_id=None,
            lease_expires_at=None,
            last_heartbeat_at=utc_now(),
            control_state=control_state,
            state_version=lease_state_version_case(
                status, control_state, current_version
            ),
            lease_attempt_id=None,
        )
    )
    if expected_run_id is not None:
        stmt = stmt.where(Task.run_id == expected_run_id)
    result = db.execute(stmt.execution_options(synchronize_session=False))
    db.commit()
    return _rowcount(result) == 1


@dataclass
class _TaskLeaseHeartbeatEntry:
    lease: TaskLease
    registrations: int = 0
    outcome: TaskLeaseHeartbeatOutcome = field(
        default_factory=TaskLeaseHeartbeatOutcome
    )
    terminal_event: asyncio.Event = field(default_factory=asyncio.Event)
    refresh_waiter: asyncio.Future[TaskLeaseHeartbeatOutcome] | None = None


class _TaskLeaseHeartbeatRegistration:
    """One task's reference to the process-local heartbeat batch."""

    def __init__(
        self,
        manager: "_TaskLeaseHeartbeatManager",
        key: TaskLeaseKey,
        entry: _TaskLeaseHeartbeatEntry,
    ) -> None:
        self._manager = manager
        self._key = key
        self._entry = entry
        self._closed = False

    @property
    def terminal_event(self) -> asyncio.Event:
        return self._entry.terminal_event

    async def close(self) -> TaskLeaseHeartbeatOutcome:
        refresh_waiter = self._entry.refresh_waiter
        if not self._closed:
            self._closed = True
            refresh_waiter = self._manager.unregister(
                self._key,
                self._entry,
            )
        if refresh_waiter is not None:
            return await asyncio.shield(refresh_waiter)
        return self._entry.outcome


class _TaskLeaseHeartbeatManager:
    """Refresh all leases through one pool waiter per backend process."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._entries: dict[TaskLeaseKey, _TaskLeaseHeartbeatEntry] = {}
        self._wake_event = asyncio.Event()
        self._runner: asyncio.Task[None] | None = None

    def register(self, lease: TaskLease) -> _TaskLeaseHeartbeatRegistration:
        if lease.run_id is None:
            raise ValueError("task lease heartbeat requires an exact run_id fence")
        key = _task_lease_key(lease)
        entry = self._entries.get(key)
        if entry is None:
            entry = _TaskLeaseHeartbeatEntry(lease=lease)
            self._entries[key] = entry
        entry.registrations += 1
        if self._runner is None or self._runner.done():
            self._runner = self._loop.create_task(self._run())
        self._wake_event.set()
        return _TaskLeaseHeartbeatRegistration(self, key, entry)

    def unregister(
        self,
        key: TaskLeaseKey,
        expected_entry: _TaskLeaseHeartbeatEntry,
    ) -> asyncio.Future[TaskLeaseHeartbeatOutcome] | None:
        entry = self._entries.get(key)
        if entry is expected_entry:
            entry.registrations -= 1
            if entry.registrations <= 0:
                self._entries.pop(key, None)
        self._wake_event.set()
        return expected_entry.refresh_waiter

    @staticmethod
    def _settle_refresh_waiter(
        entry: _TaskLeaseHeartbeatEntry,
        waiter: asyncio.Future[TaskLeaseHeartbeatOutcome],
        outcome: TaskLeaseHeartbeatOutcome,
        *,
        terminal: bool = False,
    ) -> None:
        entry.outcome = outcome
        if terminal:
            entry.terminal_event.set()
        if entry.refresh_waiter is waiter:
            entry.refresh_waiter = None
        if not waiter.done():
            waiter.set_result(outcome)

    async def wait_until_idle(self) -> None:
        runner = self._runner
        if runner is not None and runner is not asyncio.current_task():
            await drain_async_task_cancellation_safe(runner)

    async def _run(self) -> None:
        active_refresh_waiters: tuple[
            tuple[
                TaskLeaseKey,
                _TaskLeaseHeartbeatEntry,
                asyncio.Future[TaskLeaseHeartbeatOutcome],
            ],
            ...,
        ] = ()
        try:
            next_refresh_at = self._loop.time() + get_task_lease_heartbeat_seconds()
            while self._entries:
                delay = max(0.0, next_refresh_at - self._loop.time())
                if delay > 0:
                    self._wake_event.clear()
                    try:
                        await asyncio.wait_for(
                            self._wake_event.wait(),
                            timeout=delay,
                        )
                    except asyncio.TimeoutError:
                        pass
                    if not self._entries:
                        return
                    if self._loop.time() < next_refresh_at:
                        continue

                snapshot_entries = tuple(self._entries.items())
                snapshot = tuple(entry.lease for _, entry in snapshot_entries)
                refresh_waiters = tuple(
                    (key, entry, self._loop.create_future())
                    for key, entry in snapshot_entries
                )
                active_refresh_waiters = refresh_waiters
                for _, entry, waiter in refresh_waiters:
                    entry.refresh_waiter = waiter
                try:
                    states = await run_db_io_cancellation_safe(
                        lambda: refresh_task_leases_isolated(snapshot)
                    )
                except Exception as error:
                    if is_database_pool_timeout(error):
                        task_ids = ",".join(
                            str(task_id)
                            for task_id in sorted({lease.task_id for lease in snapshot})
                        )
                        logger.warning(
                            "component=lease-heartbeat task_ids=[%s] "
                            "active_lease_count=%s database pool checkout "
                            "timed out: %s",
                            task_ids,
                            len(snapshot),
                            error,
                        )
                    else:
                        logger.warning(
                            "Task lease heartbeat batch failed for %s active "
                            "leases: %s",
                            len(snapshot),
                            error,
                            exc_info=True,
                        )
                    for _, entry, waiter in refresh_waiters:
                        outcome = (
                            TaskLeaseHeartbeatOutcome(pool_timeout=error)
                            if is_database_pool_timeout(error)
                            else entry.outcome
                        )
                        self._settle_refresh_waiter(
                            entry,
                            waiter,
                            outcome,
                        )
                else:
                    for key, entry, waiter in refresh_waiters:
                        state = states.get(key, TaskLeaseRefreshState.LOST)
                        if state == TaskLeaseRefreshState.LOST:
                            self._settle_refresh_waiter(
                                entry,
                                waiter,
                                TaskLeaseHeartbeatOutcome(lease_lost=True),
                                terminal=True,
                            )
                        elif state == TaskLeaseRefreshState.SETTLEMENT_READY:
                            self._settle_refresh_waiter(
                                entry,
                                waiter,
                                TaskLeaseHeartbeatOutcome(),
                                terminal=True,
                            )
                        else:
                            self._settle_refresh_waiter(
                                entry,
                                waiter,
                                TaskLeaseHeartbeatOutcome(),
                            )
                active_refresh_waiters = ()
                interval = get_task_lease_heartbeat_seconds()
                next_refresh_at += interval
                now = self._loop.time()
                while next_refresh_at <= now:
                    next_refresh_at += interval
        finally:
            for _, entry, waiter in active_refresh_waiters:
                self._settle_refresh_waiter(
                    entry,
                    waiter,
                    entry.outcome,
                )
            self._runner = None


_task_lease_heartbeat_manager: _TaskLeaseHeartbeatManager | None = None


def _get_task_lease_heartbeat_manager() -> _TaskLeaseHeartbeatManager:
    global _task_lease_heartbeat_manager

    loop = asyncio.get_running_loop()
    manager = _task_lease_heartbeat_manager
    if manager is None or manager._loop is not loop:
        manager = _TaskLeaseHeartbeatManager(loop)
        _task_lease_heartbeat_manager = manager
    return manager


async def wait_for_heartbeat_manager_idle() -> None:
    """Wait for the current loop's shared heartbeat worker to become idle."""
    manager = _task_lease_heartbeat_manager
    if manager is not None and manager._loop is asyncio.get_running_loop():
        await manager.wait_until_idle()


async def run_task_lease_heartbeat(
    lease: TaskLease,
    stop_event: asyncio.Event,
) -> TaskLeaseHeartbeatOutcome:
    """Keep a task lease in the process-local heartbeat batch until stopped."""
    registration = _get_task_lease_heartbeat_manager().register(lease)
    stop_waiter = asyncio.create_task(stop_event.wait())
    terminal_waiter = asyncio.create_task(registration.terminal_event.wait())
    outcome: TaskLeaseHeartbeatOutcome
    try:
        await asyncio.wait(
            {stop_waiter, terminal_waiter},
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:

        async def close_registration_and_waiters() -> TaskLeaseHeartbeatOutcome:
            try:
                return await registration.close()
            finally:
                for waiter in (stop_waiter, terminal_waiter):
                    if not waiter.done():
                        waiter.cancel()
                await asyncio.gather(
                    stop_waiter,
                    terminal_waiter,
                    return_exceptions=True,
                )

        cleanup_task = asyncio.create_task(close_registration_and_waiters())
        outcome = await drain_async_task_cancellation_safe(cleanup_task)
    if outcome.lease_lost:
        logger.warning(
            "Task %s lease heartbeat lost for runner %s",
            lease.task_id,
            lease.runner_id,
        )
    return outcome


async def stop_task_lease_heartbeat(
    task: asyncio.Task[Any] | None,
    stop_event: asyncio.Event | None,
) -> TaskLeaseHeartbeatOutcome:
    if stop_event is not None:
        stop_event.set()
    if task is None:
        return TaskLeaseHeartbeatOutcome()

    try:
        outcome, cancellation = await await_task_settlement(task)
    except asyncio.CancelledError:
        current = asyncio.current_task()
        if current is not None and current.cancelling() > 0:
            raise
        # The heartbeat task itself had already been cancelled. Its worker I/O
        # is cancellation-safe, so there is nothing left to drain here.
        return TaskLeaseHeartbeatOutcome()
    if cancellation is not None:
        raise cancellation
    if isinstance(outcome, TaskLeaseHeartbeatOutcome):
        return outcome
    # Compatibility with externally supplied/mocked heartbeat tasks that
    # predate the structured result.
    return TaskLeaseHeartbeatOutcome()
