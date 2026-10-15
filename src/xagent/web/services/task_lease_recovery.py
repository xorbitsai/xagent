"""Automatic, fenced recovery for expired task execution leases."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import or_, select, update
from sqlalchemy.orm import Session

from ..models.chat_message import TaskChatMessage
from ..models.task import Task, TaskStatus, task_status_predicate
from .chat_history_service import DELIVERY_DISPATCHED, DELIVERY_PENDING
from .db_runtime import is_database_pool_timeout, run_db_io_cancellation_safe
from .ops_signals import (
    CHECKPOINT_LEGACY_POINTER_AMBIGUOUS,
    CHECKPOINT_RECOVERY_UNAVAILABLE,
    clear_degradation,
)
from .task_execution_controller import TaskControlState
from .task_lease_service import (
    CheckpointRecoveryVerdict,
    TaskLeaseRecoveryCandidate,
    get_expired_task_lease_candidates,
    get_next_expired_task_lease_candidate_for_update,
    recover_expired_task_lease_no_commit,
    resolve_checkpoint_recovery_with_data,
    utc_now,
)
from .task_orchestrator import (
    _APPENDABLE_STATUSES,
    invalidate_task_cache_best_effort,
    publish_settlement_pause,
    sync_trigger_run_status,
)
from .task_settlement_report import SettlementReport
from .workforce_runtime import sync_workforce_run_status

logger = logging.getLogger(__name__)

TASK_LEASE_EXPIRED_ERROR = (
    "Task execution lease expired before the runner completed the task."
)
TASK_UNKNOWN_TOOL_EFFECT_ERROR = (
    "Task execution lease expired with an unknown tool effect; "
    "automatic replay is unsafe."
)
TASK_LEASE_PAUSED_TRIGGER_ERROR = (
    "Task paused after execution lease expiry; manual resume is required."
)

TaskLeaseRecoveryCursor = tuple[datetime, int]


@dataclass(frozen=True)
class LeaseRecoveryPause:
    """A committed PAUSED recovery to announce as settlement announces its
    pauses (``publish_settlement_pause``): ``task_paused``, then what
    ``report.recovery`` planned for automatic recovery."""

    task_id: int
    report: SettlementReport


@dataclass(frozen=True)
class TaskLeaseRecovery:
    """One staged recovery: the status written and, for a pause, what to
    announce once its transaction commits."""

    status: TaskStatus
    pause: LeaseRecoveryPause | None = None


@dataclass(frozen=True)
class TaskLeaseRecoveryBatch:
    """Result of scanning and attempting one bounded candidate page."""

    scanned: int
    recovered: int
    next_cursor: TaskLeaseRecoveryCursor | None
    # The committed pauses, announced by the async drain.
    paused: tuple[LeaseRecoveryPause, ...] = ()


def _orphaned_pending_delivery_predicates(
    *,
    now: datetime,
    task_id: int | None,
    created_before: datetime | None,
) -> list[Any]:
    """Predicates selecting pending user rows that no owner can still settle.

    A row qualifies only while its task is quiescent -- appendable (never
    PENDING or WAITING_FOR_USER), no pause/resume in flight, and no live
    lease -- and the task has no unfinished command that could still redrive
    or inject the turn. A same-id MESSAGE that failed is excluded as well:
    its sender was told the message was not accepted, so the row must not
    start reading as delivered.
    """

    from ..models.task_command import TaskExecutionCommand
    from .task_command_transport import (
        COMMAND_FAILED,
        COMMAND_PENDING,
        COMMAND_PROCESSING,
    )

    predicates: list[Any] = [
        TaskChatMessage.role == "user",
        TaskChatMessage.delivery_status == DELIVERY_PENDING,
        TaskChatMessage.turn_id.is_not(None),
        select(Task.id)
        .where(
            Task.id == TaskChatMessage.task_id,
            task_status_predicate.in_(_APPENDABLE_STATUSES),
            or_(
                Task.control_state.is_(None),
                Task.control_state.not_in(
                    (
                        TaskControlState.PAUSE_REQUESTED.value,
                        TaskControlState.RESUME_REQUESTED.value,
                    )
                ),
            ),
            or_(
                Task.runner_id.is_(None),
                Task.lease_expires_at.is_(None),
                Task.lease_expires_at < now,
            ),
        )
        .exists(),
        ~select(TaskExecutionCommand.id)
        .where(
            TaskExecutionCommand.task_id == TaskChatMessage.task_id,
            TaskExecutionCommand.status.in_((COMMAND_PENDING, COMMAND_PROCESSING)),
        )
        .exists(),
        ~select(TaskExecutionCommand.id)
        .where(
            TaskExecutionCommand.task_id == TaskChatMessage.task_id,
            TaskExecutionCommand.command_id == TaskChatMessage.turn_id,
            TaskExecutionCommand.status == COMMAND_FAILED,
        )
        .exists(),
    ]
    if task_id is not None:
        predicates.append(TaskChatMessage.task_id == task_id)
    if created_before is not None:
        predicates.append(TaskChatMessage.created_at < created_before)
    return predicates


def reconcile_orphaned_pending_deliveries_no_commit(
    db: Session,
    *,
    now: datetime,
    task_id: int | None = None,
    created_before: datetime | None = None,
) -> int:
    """Advance orphaned ``pending`` user rows to ``dispatched``; no commit.

    A row stays ``pending`` forever when the run that accepted it crashed
    before settling it, or when its coroutine was not the authoritative
    closer at finalize (see ``_reconcile_finalized_turn_delivery``). Nothing
    will settle it later, and a ``pending`` row makes every same-id resend
    wait on it. ``dispatched`` means "do not resend", not "applied": whether
    the turn ran is unknown, and resending could run it twice. One guarded
    UPDATE, so a row that a concurrent new turn claims, or whose task starts
    running, is left alone.
    """

    from .task_execution_event_store import lock_task_execution_events_no_commit
    from .task_execution_event_writer import stage_delivery_fact_no_commit

    # Match the event writers' task-before-chat lock order. Recheck orphan
    # eligibility in the UPDATE after waiting for any in-flight task writer.
    event_tasks = list(
        db.scalars(
            select(Task.id)
            .where(
                Task.conversation_storage_version == 2,
                Task.id.in_(
                    select(TaskChatMessage.task_id).where(
                        *_orphaned_pending_delivery_predicates(
                            now=now,
                            task_id=task_id,
                            created_before=created_before,
                        )
                    )
                ),
            )
            .order_by(Task.id)
        )
    )
    for event_task_id in event_tasks:
        lock_task_execution_events_no_commit(db, event_task_id)
    result = db.execute(
        update(TaskChatMessage)
        .where(
            or_(
                TaskChatMessage.task_id.in_(event_tasks),
                TaskChatMessage.task_id.in_(
                    select(Task.id).where(Task.conversation_storage_version == 1)
                ),
            ),
            *_orphaned_pending_delivery_predicates(
                now=now,
                task_id=task_id,
                created_before=created_before,
            ),
        )
        .values(delivery_status=DELIVERY_DISPATCHED)
        .returning(TaskChatMessage.task_id, TaskChatMessage.turn_id)
        .execution_options(synchronize_session=False)
    )
    changed = result.all()
    for changed_task_id, turn_id in changed:
        if changed_task_id in event_tasks and turn_id:
            stage_delivery_fact_no_commit(
                db,
                task_id=changed_task_id,
                turn_id=turn_id,
                status=DELIVERY_DISPATCHED,
            )
    return len(changed)


def reconcile_orphaned_pending_deliveries_isolated(*, batch_size: int) -> int:
    """Sweep a bounded page of tasks with orphaned pending delivery rows.

    Rows younger than one lease TTL are left for their own run to settle.
    Each task gets its own short transaction so one busy task cannot hold
    locks for the whole page.
    """

    from ...config import get_task_lease_ttl_seconds
    from ..models.database import get_session_local

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    now = utc_now()
    created_before = now - timedelta(seconds=get_task_lease_ttl_seconds())
    SessionLocal = get_session_local()
    with SessionLocal() as scan_db:
        task_ids = [
            int(value)
            for value in scan_db.execute(
                select(TaskChatMessage.task_id)
                .where(
                    *_orphaned_pending_delivery_predicates(
                        now=now,
                        task_id=None,
                        created_before=created_before,
                    )
                )
                .distinct()
                .order_by(TaskChatMessage.task_id)
                .limit(batch_size)
            ).scalars()
        ]

    reconciled = 0
    for candidate_task_id in task_ids:
        with SessionLocal() as db:
            try:
                reconciled += reconcile_orphaned_pending_deliveries_no_commit(
                    db,
                    now=utc_now(),
                    task_id=candidate_task_id,
                    created_before=created_before,
                )
                db.commit()
            except Exception as exc:
                db.rollback()
                if is_database_pool_timeout(exc):
                    raise
                logger.exception(
                    "Orphaned delivery reconciliation failed for task %s; "
                    "retrying next tick",
                    candidate_task_id,
                )
    return reconciled


def recover_task_lease_candidate_no_commit(
    db: Session,
    candidate: TaskLeaseRecoveryCandidate,
    *,
    recovered_at: datetime,
) -> TaskStatus | None:
    """Stage one recovery and all lifecycle projections without committing."""

    staged = stage_task_lease_recovery_no_commit(
        db, candidate, recovered_at=recovered_at
    )
    return staged.status if staged is not None else None


def stage_task_lease_recovery_no_commit(
    db: Session,
    candidate: TaskLeaseRecoveryCandidate,
    *,
    recovered_at: datetime,
) -> TaskLeaseRecovery | None:
    """Stage one recovery and its projections; ``None`` writes nothing.

    A PAUSED recovery also returns the report its committer announces:
    ``task_paused`` with the control identity and reason, then what was
    recorded for automatic recovery.
    """

    resolution = resolve_checkpoint_recovery_with_data(db, candidate)
    verdict = resolution.verdict
    if verdict is CheckpointRecoveryVerdict.INDETERMINATE:
        # Checkpoint identity/content could not be resolved this round
        # (an ambiguous legacy event_id or unavailable event read). Leave
        # the lease and status untouched -- no recovery statement runs below --
        # so the next sweep re-selects it and can retry cleanly instead of
        # failing a task whose checkpoint may in fact be readable.
        return None
    next_status = (
        TaskStatus.PAUSED
        if verdict is CheckpointRecoveryVerdict.RECOVERABLE
        else TaskStatus.FAILED
    )
    task_error = (
        None
        if next_status == TaskStatus.PAUSED
        else TASK_UNKNOWN_TOOL_EFFECT_ERROR
        if verdict is CheckpointRecoveryVerdict.UNKNOWN_TOOL_EFFECT
        else TASK_LEASE_EXPIRED_ERROR
    )
    recovered = recover_expired_task_lease_no_commit(
        db,
        candidate,
        status=next_status,
        recovered_at=recovered_at,
        error_message=task_error,
    )
    if not recovered:
        return None

    # Same transaction as the fenced status write: the expired run cannot
    # settle its own delivery rows any more, so close them here.
    reconciled = reconcile_orphaned_pending_deliveries_no_commit(
        db,
        now=recovered_at,
        task_id=candidate.task_id,
    )
    if reconciled:
        logger.info(
            "Lease recovery of task %s reconciled %s pending delivery row(s)",
            candidate.task_id,
            reconciled,
        )
    db.expire_all()
    task = db.query(Task).filter(Task.id == candidate.task_id).one()
    from .task_auto_recovery import record_lease_expiry_interruption_no_commit

    # After the fenced write so it sees the new state_version. A failure
    # inside it is skipped; it cannot change the status, error, or the
    # projections staged here except the TriggerRun one below.
    recorded = record_lease_expiry_interruption_no_commit(
        db,
        task=task,
        candidate=candidate,
        resolution=resolution,
        task_status=next_status,
        recovered_at=recovered_at,
    )
    from .task_auto_recovery import lease_expiry_interruption_reason
    from .task_auto_resume import trigger_error_for_recorded
    from .task_execution_controller import task_control_snapshot
    from .task_execution_event_writer import stage_result_fact_no_commit

    stage_result_fact_no_commit(db, task, {"error": task_error})
    sync_workforce_run_status(db, task, next_status)
    # A scheduled automatic resume keeps the TriggerRun RUNNING: the resumed
    # run settles it.
    trigger_error = (
        trigger_error_for_recorded(recorded, default=TASK_LEASE_PAUSED_TRIGGER_ERROR)
        if next_status == TaskStatus.PAUSED
        else TASK_LEASE_EXPIRED_ERROR
    )
    if trigger_error is not None:
        sync_trigger_run_status(db, task, next_status, error_message=trigger_error)
    if next_status != TaskStatus.PAUSED:
        return TaskLeaseRecovery(next_status)
    return TaskLeaseRecovery(
        next_status,
        LeaseRecoveryPause(
            task_id=candidate.task_id,
            report=SettlementReport(
                control_state=task_control_snapshot(task).as_dict(),
                paused_for=lease_expiry_interruption_reason(candidate, verdict),
                recovery=recorded,
            ),
        ),
    )


def recover_task_lease_candidate_isolated(
    candidate: TaskLeaseRecoveryCandidate,
    *,
    recovered_at: datetime,
) -> TaskStatus | None:
    """Own the short recovery Session, transaction, and post-commit cache write."""

    committed = _commit_task_lease_recovery(candidate, recovered_at=recovered_at)
    return committed.status if committed is not None else None


def _commit_task_lease_recovery(
    candidate: TaskLeaseRecoveryCandidate,
    *,
    recovered_at: datetime,
) -> TaskLeaseRecovery | None:
    """Recover one candidate in its own transaction; the result once committed."""

    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        try:
            staged = stage_task_lease_recovery_no_commit(
                db,
                candidate,
                recovered_at=recovered_at,
            )
            if staged is None:
                db.rollback()
                return None
            db.commit()
        except Exception:
            db.rollback()
            raise
    _record_recovered_task_lease(candidate, staged.status)
    return staged


def _record_recovered_task_lease(
    candidate: TaskLeaseRecoveryCandidate,
    recovered: TaskStatus,
) -> None:
    """Publish post-commit effects for one successfully recovered lease."""

    invalidate_task_cache_best_effort(candidate.task_id)
    logger.warning(
        "Recovered expired task lease: task=%s status=%s runner=%s run=%s",
        candidate.task_id,
        recovered.value,
        candidate.runner_id,
        candidate.run_id,
    )


_COMMITTED_PAUSES_ATTR = "_xagent_committed_lease_recovery_pauses"


def _carry_committed_pauses(
    exc: BaseException, paused: list[LeaseRecoveryPause]
) -> None:
    """Keep the pauses this batch committed on the error that ended it, so
    the drain still announces them (as the auto-resume tick does)."""

    if paused:
        exc.__dict__[_COMMITTED_PAUSES_ATTR] = tuple(paused)


def committed_lease_recovery_pauses(
    exc: BaseException,
) -> tuple[LeaseRecoveryPause, ...]:
    """What a batch that raised ``exc`` had committed before it failed."""

    pauses = getattr(exc, _COMMITTED_PAUSES_ATTR, ())
    return pauses if isinstance(pauses, tuple) else ()


def _recover_expired_task_leases_batch_with_cas(
    *,
    cutoff: datetime,
    batch_size: int,
    after: TaskLeaseRecoveryCursor | None,
) -> TaskLeaseRecoveryBatch:
    """Scan one bounded page and recover every candidate independently."""

    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    with SessionLocal() as scan_db:
        candidates = get_expired_task_lease_candidates(
            scan_db,
            cutoff=cutoff,
            limit=batch_size,
            after=after,
        )

    recovered = 0
    paused: list[LeaseRecoveryPause] = []
    try:
        for candidate in candidates:
            try:
                committed = _commit_task_lease_recovery(
                    candidate, recovered_at=utc_now()
                )
                if committed is not None:
                    recovered += 1
                    if committed.pause is not None:
                        paused.append(committed.pause)
            except Exception as exc:
                if is_database_pool_timeout(exc):
                    raise
                logger.exception(
                    "Task lease recovery failed for task %s; retrying next tick",
                    candidate.task_id,
                )
    except Exception as exc:
        _carry_committed_pauses(exc, paused)
        raise

    return TaskLeaseRecoveryBatch(
        scanned=len(candidates),
        recovered=recovered,
        next_cursor=candidates[-1].cursor if candidates else None,
        paused=tuple(paused),
    )


def _use_postgresql_recovery_partitioning() -> bool:
    """Return whether candidate row locking can partition recovery workers."""

    from ..models.database import get_engine

    return get_engine().dialect.name == "postgresql"


def _recover_expired_task_leases_batch_with_postgresql_locks(
    *,
    cutoff: datetime,
    batch_size: int,
    after: TaskLeaseRecoveryCursor | None,
) -> TaskLeaseRecoveryBatch:
    """Recover one page using a short locked transaction per candidate."""

    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    scanned = 0
    recovered_count = 0
    paused: list[LeaseRecoveryPause] = []
    cursor = after
    try:
        while scanned < batch_size:
            candidate: TaskLeaseRecoveryCandidate | None = None
            recovered: TaskLeaseRecovery | None = None
            with SessionLocal() as db:
                try:
                    candidate = get_next_expired_task_lease_candidate_for_update(
                        db,
                        cutoff=cutoff,
                        after=cursor,
                    )
                except Exception:
                    db.rollback()
                    raise

                if candidate is None:
                    db.rollback()
                    break

                try:
                    recovered = stage_task_lease_recovery_no_commit(
                        db,
                        candidate,
                        recovered_at=utc_now(),
                    )
                    if recovered is None:
                        db.rollback()
                    else:
                        db.commit()
                except Exception as exc:
                    db.rollback()
                    if is_database_pool_timeout(exc):
                        raise
                    recovered = None
                    logger.exception(
                        "Task lease recovery failed for task %s; retrying next tick",
                        candidate.task_id,
                    )

            scanned += 1
            cursor = candidate.cursor
            if recovered is not None:
                recovered_count += 1
                _record_recovered_task_lease(candidate, recovered.status)
                if recovered.pause is not None:
                    paused.append(recovered.pause)
    except Exception as exc:
        _carry_committed_pauses(exc, paused)
        raise

    return TaskLeaseRecoveryBatch(
        scanned=scanned,
        recovered=recovered_count,
        next_cursor=cursor if scanned else None,
        paused=tuple(paused),
    )


def recover_expired_task_leases_batch_isolated(
    *,
    cutoff: datetime,
    batch_size: int,
    after: TaskLeaseRecoveryCursor | None,
) -> TaskLeaseRecoveryBatch:
    """Recover one bounded page with backend-appropriate worker partitioning."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if _use_postgresql_recovery_partitioning():
        return _recover_expired_task_leases_batch_with_postgresql_locks(
            cutoff=cutoff,
            batch_size=batch_size,
            after=after,
        )
    return _recover_expired_task_leases_batch_with_cas(
        cutoff=cutoff,
        batch_size=batch_size,
        after=after,
    )


async def recover_expired_task_leases_until_cutoff(
    *,
    cutoff: datetime,
    batch_size: int,
) -> int:
    """Drain all pages older than one fixed cutoff without starving the loop."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    # One drain is the unit this signal reports on: "the most recent
    # completed sweep saw at least one ambiguous legacy pointer". Clearing
    # per candidate would let a later clean candidate erase an earlier
    # ambiguity inside the same sweep; clearing per page would let page 2
    # erase what page 1 registered, since the cursor has already advanced
    # past it. At either finer grain the signal could never stay set.
    clear_degradation(CHECKPOINT_LEGACY_POINTER_AMBIGUOUS)
    clear_degradation(CHECKPOINT_RECOVERY_UNAVAILABLE)

    async def announce(pauses: tuple[LeaseRecoveryPause, ...]) -> None:
        # Committed pauses are announced as settlement announces its own:
        # ``task_paused``, then any automatic-resume notice. Best effort.
        for pause in pauses:
            await publish_settlement_pause(pause.task_id, pause.report)

    recovered = 0
    cursor: TaskLeaseRecoveryCursor | None = None
    while True:
        try:
            batch = await run_db_io_cancellation_safe(
                lambda: recover_expired_task_leases_batch_isolated(
                    cutoff=cutoff,
                    batch_size=batch_size,
                    after=cursor,
                )
            )
        except Exception as exc:
            # A later candidate failed the batch (a pool timeout, the scan);
            # what it committed before that is still announced.
            await announce(committed_lease_recovery_pauses(exc))
            raise
        recovered += batch.recovered
        await announce(batch.paused)
        if batch.scanned < batch_size:
            return recovered
        cursor = batch.next_cursor
        await asyncio.sleep(0)


async def run_task_lease_recovery_loop(
    *,
    poll_interval_seconds: int,
    batch_size: int,
) -> None:
    """Continuously recover expired leases while the backend process is alive."""

    secret_cursor: int | None = None
    while True:
        try:
            recovered = await recover_expired_task_leases_until_cutoff(
                cutoff=datetime.now(timezone.utc),
                batch_size=batch_size,
            )
            if recovered:
                logger.info("Recovered %s expired task lease(s)", recovered)
            from ...config import get_shared_task_execution_enabled

            if get_shared_task_execution_enabled():
                from .task_runtime_secrets import clean_finished_runtime_values

                await run_db_io_cancellation_safe(
                    lambda: recover_expired_idle_task_leases(batch_size=batch_size)
                )
                secret_cursor = await run_db_io_cancellation_safe(
                    lambda: clean_finished_runtime_values(
                        after_id=secret_cursor, batch_size=batch_size
                    )
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if is_database_pool_timeout(exc):
                logger.warning(
                    "Task lease recovery skipped after database pool timeout"
                )
            else:
                logger.exception("Task lease recovery tick failed")

        # Its own guard so a lease-recovery fault cannot starve it. The first
        # tick runs before any sleep, which also backfills historical rows.
        try:
            reconciled = await run_db_io_cancellation_safe(
                lambda: reconcile_orphaned_pending_deliveries_isolated(
                    batch_size=batch_size
                )
            )
            if reconciled:
                logger.info(
                    "Reconciled %s orphaned pending delivery row(s) to dispatched",
                    reconciled,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if is_database_pool_timeout(exc):
                logger.warning(
                    "Orphaned delivery reconciliation skipped after database "
                    "pool timeout"
                )
            else:
                logger.exception("Orphaned delivery reconciliation tick failed")

        await asyncio.sleep(poll_interval_seconds)


def recover_expired_idle_task_leases(*, batch_size: int) -> int:
    """Recover a bounded page of expired owners whose execution already settled."""
    from sqlalchemy import select

    from ..models.database import get_session_local
    from .task_coordinator_service import recover_expired_idle_task_lease_no_commit

    with get_session_local()() as db:
        candidates = (
            db.execute(
                select(Task.id)
                .where(
                    Task.status != TaskStatus.RUNNING,
                    Task.lease_expires_at < utc_now(),
                )
                .order_by(Task.lease_expires_at, Task.id)
                .limit(batch_size)
            )
            .scalars()
            .all()
        )
    recovered = 0
    for task_id in candidates:
        with get_session_local()() as db, db.begin():
            recovered += recover_expired_idle_task_lease_no_commit(db, task_id)
    return recovered
