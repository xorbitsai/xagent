"""Durable, cross-worker transport for task execution commands.

Ingress commits a command before acknowledging it: enqueue_task_command is
the committing wrapper most callers use, built on stage_task_command, the
staging primitive that deliberately does not commit so a caller can fold a
command into a larger transaction of its own. Every web worker runs the
same dispatcher; a database claim chooses one consumer while per-task ordering
prevents a later command from overtaking an earlier unfinished command.
"""

from __future__ import annotations

import asyncio
import enum
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

from sqlalchemy import and_, exists, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Query, Session, aliased

from ...config import (
    get_task_lease_heartbeat_seconds,
    get_task_lease_ttl_seconds,
)
from ..models.task import Task, TaskStatus
from ..models.task_command import TaskExecutionCommand
from .db_runtime import (
    await_task_settlement,
    is_database_pool_timeout,
    propagate_deferred_cancellation,
    run_db_io_cancellation_safe,
)
from .task_lease_service import get_runner_id

logger = logging.getLogger(__name__)

COMMAND_PENDING = "pending"
COMMAND_PROCESSING = "processing"
COMMAND_COMPLETED = "completed"
COMMAND_FAILED = "failed"
COMMAND_TERMINAL = (COMMAND_COMPLETED, COMMAND_FAILED)

COMMAND_ID_PATTERN = re.compile(r"[A-Za-z0-9._:-]{1,64}")
MAX_COMMAND_FAILURES = 5
MAX_COMMAND_DEFERS = 60
DISPATCHER_IDLE_SECONDS = 0.5
DISPATCHER_CONCURRENCY = 4


class TaskCommandKind(str, enum.Enum):
    MESSAGE = "message"
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"


class TaskCommandTaskMissing(ValueError):
    """The target task no longer exists.

    Raised when the row is absent, including when a concurrent delete lands
    between the existence check and the insert. Callers that tolerate a missing
    task treat this as the missing-task sentinel rather than a rejection, so a
    deleted task can be replaced instead of losing the message.
    """


class TaskCommandDeferred(RuntimeError):
    """The command is durable but its downstream handoff is still pending."""


class TaskCommandRejected(RuntimeError):
    """The downstream handoff reached a durable failed state."""

    def __init__(self, message: str, *, reason: str | None = None) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class EnqueuedTaskCommand:
    command_id: int
    client_command_id: str
    created: bool
    payload_matches: bool
    status: str


@dataclass(frozen=True)
class StagedTaskCommand:
    """A command row added to a caller-owned session, not yet committed.

    ``staged_db_id`` is deliberately not named ``command_id`` -- unlike
    ``EnqueuedTaskCommand.command_id``, this id is only guaranteed to exist in
    the database once the caller commits the session it was staged on. Code
    that dispatches or notifies by id must never accept a StagedTaskCommand.
    """

    staged_db_id: int
    client_command_id: str
    created: bool
    payload_matches: bool
    status: str

    def to_enqueued(self) -> EnqueuedTaskCommand:
        """Project this staged row into the committing wrapper's result shape.

        ``staged_db_id`` becomes ``EnqueuedTaskCommand.command_id`` -- valid
        only once the caller has committed the transaction this row was
        staged on.
        """

        return EnqueuedTaskCommand(
            command_id=self.staged_db_id,
            client_command_id=self.client_command_id,
            created=self.created,
            payload_matches=self.payload_matches,
            status=self.status,
        )


class TaskCommandOwnerStateError(RuntimeError):
    """The caller's own pending writes failed to flush before staging began.

    Deliberately not an IntegrityError subclass: the wrapper's
    ``except IntegrityError`` classifies conflicts on the command insert
    itself, and must not also catch a failure that has nothing to do with
    that insert. After this is raised, the session is unusable -- the caller
    must roll back before issuing any further statement on it.
    """


class TaskCommandConflictKind(enum.Enum):
    RACED_DUPLICATE = "raced_duplicate"
    TASK_MISSING = "task_missing"
    UNRELATED = "unrelated"


@dataclass(frozen=True)
class RacedTaskCommandProjection:
    """Immutable snapshot of the row that won a duplicate-insert race."""

    command_db_id: int
    status: str
    payload_matches: bool


@dataclass(frozen=True)
class TaskCommandConflictClassification:
    kind: TaskCommandConflictKind
    raced: RacedTaskCommandProjection | None = None


@dataclass(frozen=True)
class ClaimedTaskCommand:
    id: int
    task_id: int
    actor_user_id: int | None
    command_id: str
    kind: TaskCommandKind
    payload: dict[str, Any]
    target_run_id: str | None
    attempt_count: int
    failure_count: int = 0
    defer_count: int = 0


@dataclass(frozen=True)
class TaskCommandStateSnapshot:
    """Immutable command state safe to return across a worker boundary."""

    command_db_id: int
    status: str
    error: str | None
    rejection_reason: str | None
    attempt_count: int
    failure_count: int
    defer_count: int

    @property
    def result(self) -> dict[str, Any]:
        """Return the legacy result shape without exposing mutable snapshot state."""

        if self.rejection_reason is None:
            return {}
        return {"rejection_reason": self.rejection_reason}


@dataclass(frozen=True)
class TaskCommandClaimHeartbeatOutcome:
    """Last confirmed health of a processing-command claim."""

    claim_lost: bool = False
    pool_timeout: BaseException | None = None

    @property
    def requires_ttl_recovery(self) -> bool:
        return self.claim_lost or self.pool_timeout is not None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _canonical_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _normalize_command_id(command_id: str) -> str:
    """Strip and validate a caller-supplied command_id.

    Shared by every normalization site in this module so the accepted format
    -- 1-64 URL-safe characters -- cannot drift between them.
    """

    normalized = command_id.strip()
    if COMMAND_ID_PATTERN.fullmatch(normalized) is None:
        raise ValueError("command_id must be 1-64 URL-safe characters")
    return normalized


def _matches_existing(
    command: TaskExecutionCommand,
    *,
    actor_user_id: int | None,
    kind: TaskCommandKind,
    payload: dict[str, Any],
) -> bool:
    stored_payload: dict[str, Any] = (
        command.payload if isinstance(command.payload, dict) else {}
    )
    stored_actor_user_id = getattr(command, "actor_user_id", None)
    return (
        stored_actor_user_id == actor_user_id
        and str(command.kind) == kind.value
        and _canonical_payload(stored_payload) == _canonical_payload(payload)
    )


def stage_task_command(
    db: Session,
    *,
    task_id: int,
    actor_user_id: int | None,
    command_id: str,
    kind: TaskCommandKind,
    payload: dict[str, Any],
) -> StagedTaskCommand:
    """Add an idempotent command row to the session without ending its transaction.

    Never calls ``db.commit()``, ``db.rollback()``, or the dispatcher notify --
    the caller owns the transaction boundary and decides what durability means
    for its own writes alongside this command. Statement order is load-bearing:

    1. Reject a malformed command_id before touching the database at all.
    2. Flush the caller's own pending writes first, so a conflict there is
       reported as TaskCommandOwnerStateError rather than folded into the
       IntegrityError this function raises for a duplicate command insert.
    3. Read the task through a single Core select rather than the ORM. This
       both closes the window a separate existence check and a separate
       snapshot read would leave between them, and -- unlike an ORM query --
       is not served from the identity map, so it observes a caller's own
       uncommitted CAS-style update to the same row instead of a stale cached
       instance.
    4. Check for an existing command with this (task_id, command_id) before
       adding a new row, so a repeated call is idempotent without relying on
       the unique constraint to reject it.
    5. Add and flush the new row last, so any remaining IntegrityError is
       attributable to the command insert itself.

    Caller obligations:

    a. After the caller's own commit, it should invoke
       notify_task_command_dispatcher(). Skipping it does not lose the
       command -- the dispatcher's idle poll still recovers it -- but
       delivery silently degrades to up to DISPATCHER_IDLE_SECONDS of added
       latency instead of an immediate wakeup.
    b. On IntegrityError from the command-insert flush in step 5, the caller
       must roll back the whole transaction before issuing any further
       statement on this session -- classify_task_command_conflict below
       assumes that rollback has already happened.
    c. On every exit path -- a created row, an idempotent hit, or
       TaskCommandTaskMissing -- this call must be the last write a caller
       issues before it commits, with no I/O or other slow work in between.
       A caller that instead stays in a long transaction after staging holds
       SQLite's database-wide writer lock for the entire transaction, not
       the microseconds an immediate commit implies; every concurrent
       writer, including the dispatcher's claim, blocks for that whole span.
    """

    normalized_id = _normalize_command_id(command_id)
    resolved_task_id = int(task_id)

    try:
        db.flush()
    except IntegrityError as exc:
        raise TaskCommandOwnerStateError(
            "caller's pending writes failed to flush before command staging"
        ) from exc

    # A task snapshot a caller loaded earlier would widen the concurrent-delete
    # window across everything it did in between, so existence is re-checked
    # here, by query, before inserting a command that references the row.
    snapshot = db.execute(
        select(Task.status, Task.runner_id, Task.run_id).where(
            Task.id == resolved_task_id
        )
    ).one_or_none()
    if snapshot is None:
        raise TaskCommandTaskMissing(f"Task {task_id} not found")

    existing = (
        db.query(TaskExecutionCommand)
        .filter(
            TaskExecutionCommand.task_id == resolved_task_id,
            TaskExecutionCommand.command_id == normalized_id,
        )
        .first()
    )
    if existing is not None:
        matches = _matches_existing(
            existing,
            actor_user_id=actor_user_id,
            kind=kind,
            payload=payload,
        )
        return StagedTaskCommand(
            staged_db_id=int(existing.id),
            client_command_id=normalized_id,
            created=False,
            payload_matches=matches,
            status=str(existing.status),
        )

    active_runner_id = (
        snapshot.runner_id
        if snapshot.status == TaskStatus.RUNNING and snapshot.runner_id is not None
        else None
    )
    command = TaskExecutionCommand(
        task_id=resolved_task_id,
        actor_user_id=actor_user_id,
        command_id=normalized_id,
        kind=kind.value,
        payload=payload,
        target_run_id=snapshot.run_id,
        target_runner_id=active_runner_id,
        status=COMMAND_PENDING,
    )
    db.add(command)
    db.flush()
    return StagedTaskCommand(
        staged_db_id=int(command.id),
        client_command_id=normalized_id,
        created=True,
        payload_matches=True,
        status=COMMAND_PENDING,
    )


def classify_task_command_conflict(
    db: Session,
    *,
    task_id: int,
    command_id: str,
    actor_user_id: int | None,
    kind: TaskCommandKind,
    payload: dict[str, Any],
) -> TaskCommandConflictClassification:
    """Classify an IntegrityError from staging a command insert.

    Normalizes command_id the same way stage_task_command does -- strip plus
    full-format validation, not strip alone -- so a lookup here targets the
    same row a duplicate-command race would have inserted.

    Call only after the caller has rolled back the transaction the failed
    stage happened on -- this queries post-rollback state to tell apart a
    duplicate command_id that won the race (RACED_DUPLICATE, carrying a
    frozen projection of the winning row computed in this same query so the
    caller does not need a second, possibly inconsistent, read), a task
    deleted concurrently with the insert (TASK_MISSING), and a conflict on
    neither -- the row references two foreign keys, so absence of a duplicate
    does not prove the task caused it: a concurrently deleted actor fails the
    users FK while the task is still present (UNRELATED). Rolling back to a
    savepoint opened before the failing insert is equivalent to rolling back
    the whole transaction for this purpose, as long as that savepoint
    predates the insert: either way, the post-rollback state this function
    queries no longer contains the failed statement's own effects.

    RACED_DUPLICATE means the command survived the race, not that the
    caller's work did: the rollback that had to precede this call also undid
    whatever else the caller had written. A caller that had such writes must
    redo them rather than report success. Opening the savepoint after those
    writes is the way out of redoing them: a rollback to a savepoint that
    postdates them leaves them pending in the still-open outer transaction,
    so the caller decides whether to commit or discard them once it knows
    what this function found. ``task_interaction_service.respond()`` is
    that shape. A caller with nothing else pending
    -- enqueue_task_command is one -- has nothing to redo and reports the
    raced row as created=False.
    """

    resolved_task_id = int(task_id)
    normalized_id = _normalize_command_id(command_id)
    raced = (
        db.query(TaskExecutionCommand)
        .filter(
            TaskExecutionCommand.task_id == resolved_task_id,
            TaskExecutionCommand.command_id == normalized_id,
        )
        .one_or_none()
    )
    if raced is not None:
        matches = _matches_existing(
            raced,
            actor_user_id=actor_user_id,
            kind=kind,
            payload=payload,
        )
        return TaskCommandConflictClassification(
            kind=TaskCommandConflictKind.RACED_DUPLICATE,
            raced=RacedTaskCommandProjection(
                command_db_id=int(raced.id),
                status=str(raced.status),
                payload_matches=matches,
            ),
        )
    if db.query(Task.id).filter(Task.id == resolved_task_id).scalar() is None:
        return TaskCommandConflictClassification(
            kind=TaskCommandConflictKind.TASK_MISSING
        )
    return TaskCommandConflictClassification(kind=TaskCommandConflictKind.UNRELATED)


def enqueue_task_command(
    db: Session,
    *,
    task_id: int,
    actor_user_id: int | None,
    command_id: str,
    kind: TaskCommandKind,
    payload: dict[str, Any],
) -> EnqueuedTaskCommand:
    """Commit an idempotent command and return only after it is durable.

    TaskCommandOwnerStateError from stage_task_command's owner pre-flush
    propagates uncaught -- it is deliberately not caught alongside
    IntegrityError below. A caller managing its own session lifetime must
    treat it as its own write failure, not a command-staging failure, and
    roll back before issuing any further statement on the session.
    """

    try:
        staged = stage_task_command(
            db,
            task_id=task_id,
            actor_user_id=actor_user_id,
            command_id=command_id,
            kind=kind,
            payload=payload,
        )
        # Only a newly created row is worth committing, and the commit stays
        # inside this try so that a constraint failure surfacing there rather
        # than at the staging flush is still rolled back and classified.
        if staged.created:
            db.commit()
    except IntegrityError:
        db.rollback()
        # A constraint that only fires at commit time on the caller's own row
        # (rather than on the command insert) would still land here and enter
        # conflict classification below. No deferrable constraint exists in
        # the current schema, so this residual conflation cannot yet trigger
        # in practice; it is left as-is to keep parity with the original
        # single-function behavior this was extracted from.
        classification = classify_task_command_conflict(
            db,
            task_id=task_id,
            command_id=command_id,
            actor_user_id=actor_user_id,
            kind=kind,
            payload=payload,
        )
        if classification.kind is TaskCommandConflictKind.RACED_DUPLICATE:
            raced = classification.raced
            assert raced is not None
            return EnqueuedTaskCommand(
                command_id=raced.command_db_id,
                client_command_id=_normalize_command_id(command_id),
                created=False,
                payload_matches=raced.payload_matches,
                status=raced.status,
            )
        if classification.kind is TaskCommandConflictKind.TASK_MISSING:
            raise TaskCommandTaskMissing(f"Task {task_id} not found") from None
        raise

    if not staged.created:
        return staged.to_enqueued()

    notify_task_command_dispatcher()
    return staged.to_enqueued()


def _claim_availability_predicate(now: datetime) -> Any:
    return or_(
        and_(
            TaskExecutionCommand.status == COMMAND_PENDING,
            or_(
                TaskExecutionCommand.claim_expires_at.is_(None),
                TaskExecutionCommand.claim_expires_at < now,
            ),
        ),
        and_(
            TaskExecutionCommand.status == COMMAND_PROCESSING,
            TaskExecutionCommand.claim_expires_at < now,
        ),
    )


def _command_routing_predicate(runner_id: str, now: datetime) -> Any:
    return or_(
        TaskExecutionCommand.target_runner_id.is_(None),
        TaskExecutionCommand.target_runner_id == runner_id,
        Task.runner_id == runner_id,
        Task.runner_id.is_(None),
        Task.lease_expires_at.is_(None),
        Task.lease_expires_at < now,
    )


def _unfinished_earlier_command() -> Any:
    earlier = aliased(TaskExecutionCommand)
    return exists(
        select(1).where(
            earlier.task_id == TaskExecutionCommand.task_id,
            earlier.id < TaskExecutionCommand.id,
            earlier.status.notin_(COMMAND_TERMINAL),
        )
    )


def _claimable_query(
    db: Session, *, runner_id: str, command_db_id: int | None
) -> Query[Any]:
    now = _utc_now()
    query = (
        db.query(TaskExecutionCommand)
        .join(Task, Task.id == TaskExecutionCommand.task_id)
        .filter(
            _claim_availability_predicate(now),
            ~_unfinished_earlier_command(),
            _command_routing_predicate(runner_id, now),
        )
    )
    if command_db_id is not None:
        query = query.filter(TaskExecutionCommand.id == command_db_id)
    return query.order_by(TaskExecutionCommand.id.asc())


def claim_task_command(
    db: Session,
    *,
    runner_id: str | None = None,
    command_db_id: int | None = None,
) -> ClaimedTaskCommand | None:
    """Atomically claim the oldest eligible command for this worker."""

    resolved_runner_id = runner_id or get_runner_id()
    candidate = _claimable_query(
        db,
        runner_id=resolved_runner_id,
        command_db_id=command_db_id,
    ).first()
    if candidate is None:
        return None

    now = _utc_now()
    expires = now + timedelta(seconds=get_task_lease_ttl_seconds())
    routable_task = exists(
        select(1).where(
            Task.id == TaskExecutionCommand.task_id,
            _command_routing_predicate(resolved_runner_id, now),
        )
    )
    claimed = (
        db.query(TaskExecutionCommand)
        .filter(
            TaskExecutionCommand.id == int(candidate.id),
            _claim_availability_predicate(now),
            ~_unfinished_earlier_command(),
            routable_task,
        )
        .update(
            {
                TaskExecutionCommand.status: COMMAND_PROCESSING,
                TaskExecutionCommand.claimed_by: resolved_runner_id,
                TaskExecutionCommand.claim_expires_at: expires,
                TaskExecutionCommand.attempt_count: (
                    TaskExecutionCommand.attempt_count + 1
                ),
                TaskExecutionCommand.error: None,
                TaskExecutionCommand.updated_at: now,
            },
            synchronize_session=False,
        )
    )
    if claimed != 1:
        db.rollback()
        return None
    db.commit()
    fresh = (
        db.query(TaskExecutionCommand)
        .filter(TaskExecutionCommand.id == int(candidate.id))
        .one()
    )
    payload: dict[str, Any] = fresh.payload if isinstance(fresh.payload, dict) else {}
    return ClaimedTaskCommand(
        id=int(fresh.id),
        task_id=int(fresh.task_id),
        actor_user_id=(
            int(fresh.actor_user_id) if fresh.actor_user_id is not None else None
        ),
        command_id=str(fresh.command_id),
        kind=TaskCommandKind(str(fresh.kind)),
        payload=payload,
        target_run_id=(str(fresh.target_run_id) if fresh.target_run_id else None),
        attempt_count=int(fresh.attempt_count or 0),
        failure_count=int(fresh.failure_count or 0),
        defer_count=int(fresh.defer_count or 0),
    )


def _claim_task_command_isolated(
    *,
    runner_id: str,
    command_db_id: int | None,
) -> ClaimedTaskCommand | None:
    """Claim in a worker-owned short session and return a detached snapshot."""

    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        return claim_task_command(
            db,
            runner_id=runner_id,
            command_db_id=command_db_id,
        )


def renew_task_command_claim(
    command_db_id: int,
    runner_id: str,
    *,
    expected_attempt_count: int | None = None,
) -> bool:
    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    now = _utc_now()
    with SessionLocal() as db:
        query = db.query(TaskExecutionCommand).filter(
            TaskExecutionCommand.id == command_db_id,
            TaskExecutionCommand.status == COMMAND_PROCESSING,
            TaskExecutionCommand.claimed_by == runner_id,
        )
        if expected_attempt_count is not None:
            query = query.filter(
                TaskExecutionCommand.attempt_count == expected_attempt_count
            )
        updated = query.update(
            {
                TaskExecutionCommand.claim_expires_at: now
                + timedelta(seconds=get_task_lease_ttl_seconds()),
                TaskExecutionCommand.updated_at: now,
            },
            synchronize_session=False,
        )
        db.commit()
        return updated == 1


async def _claim_heartbeat(
    command_db_id: int,
    runner_id: str,
    attempt_count: int,
    stop_event: asyncio.Event,
) -> TaskCommandClaimHeartbeatOutcome:
    interval = get_task_lease_heartbeat_seconds()
    pool_timeout: BaseException | None = None
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
            return TaskCommandClaimHeartbeatOutcome(pool_timeout=pool_timeout)
        except asyncio.TimeoutError:
            pass
        try:
            renewed = await run_db_io_cancellation_safe(
                lambda: renew_task_command_claim(
                    command_db_id,
                    runner_id,
                    expected_attempt_count=attempt_count,
                )
            )
        except Exception as exc:  # noqa: BLE001
            if is_database_pool_timeout(exc):
                pool_timeout = exc
            logger.warning(
                "Failed to renew task command claim %s: %s",
                command_db_id,
                exc,
                exc_info=True,
            )
            continue
        pool_timeout = None
        if not renewed:
            return TaskCommandClaimHeartbeatOutcome(claim_lost=True)
    return TaskCommandClaimHeartbeatOutcome(pool_timeout=pool_timeout)


def finish_task_command(
    command_db_id: int,
    runner_id: str,
    *,
    result: dict[str, Any] | None = None,
    expected_attempt_count: int | None = None,
) -> bool:
    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    now = _utc_now()
    with SessionLocal() as db:
        query = db.query(TaskExecutionCommand).filter(
            TaskExecutionCommand.id == command_db_id,
            TaskExecutionCommand.status == COMMAND_PROCESSING,
            TaskExecutionCommand.claimed_by == runner_id,
        )
        if expected_attempt_count is not None:
            query = query.filter(
                TaskExecutionCommand.attempt_count == expected_attempt_count
            )
        updated = query.update(
            {
                TaskExecutionCommand.status: COMMAND_COMPLETED,
                TaskExecutionCommand.result: result,
                TaskExecutionCommand.error: None,
                TaskExecutionCommand.claimed_by: None,
                TaskExecutionCommand.claim_expires_at: None,
                TaskExecutionCommand.completed_at: now,
                TaskExecutionCommand.updated_at: now,
            },
            synchronize_session=False,
        )
        db.commit()
        return updated == 1


def fail_task_command(
    command_db_id: int,
    runner_id: str,
    error: str,
    *,
    force_terminal: bool = False,
    expected_attempt_count: int | None = None,
    result: dict[str, Any] | None = None,
) -> bool:
    """Retry a failed claim, or make it terminal after bounded attempts."""

    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    now = _utc_now()
    with SessionLocal() as db:
        snapshot_query = db.query(
            TaskExecutionCommand.failure_count,
            TaskExecutionCommand.attempt_count,
        ).filter(
            TaskExecutionCommand.id == command_db_id,
            TaskExecutionCommand.status == COMMAND_PROCESSING,
            TaskExecutionCommand.claimed_by == runner_id,
        )
        if expected_attempt_count is not None:
            snapshot_query = snapshot_query.filter(
                TaskExecutionCommand.attempt_count == expected_attempt_count
            )
        snapshot = snapshot_query.first()
        if snapshot is None:
            return False
        observed_failure_count = int(snapshot.failure_count or 0)
        observed_attempt_count = int(snapshot.attempt_count or 0)
        failure_count = observed_failure_count + 1
        terminal = force_terminal or failure_count >= MAX_COMMAND_FAILURES
        updated = (
            db.query(TaskExecutionCommand)
            .filter(
                TaskExecutionCommand.id == command_db_id,
                TaskExecutionCommand.status == COMMAND_PROCESSING,
                TaskExecutionCommand.claimed_by == runner_id,
                TaskExecutionCommand.failure_count == observed_failure_count,
                TaskExecutionCommand.attempt_count == observed_attempt_count,
            )
            .update(
                {
                    TaskExecutionCommand.status: (
                        COMMAND_FAILED if terminal else COMMAND_PENDING
                    ),
                    TaskExecutionCommand.failure_count: failure_count,
                    TaskExecutionCommand.error: error[:4000],
                    TaskExecutionCommand.result: result,
                    TaskExecutionCommand.claimed_by: None,
                    TaskExecutionCommand.claim_expires_at: (
                        None
                        if terminal
                        else now + timedelta(seconds=min(2**failure_count, 30))
                    ),
                    TaskExecutionCommand.updated_at: now,
                    TaskExecutionCommand.completed_at: now if terminal else None,
                },
                synchronize_session=False,
            )
        )
        db.commit()
        if updated != 1:
            return False
    if not terminal:
        notify_task_command_dispatcher()
    return True


def defer_task_command(
    command_db_id: int,
    runner_id: str,
    reason: str,
    *,
    expected_attempt_count: int | None = None,
) -> bool:
    """Release a claim for retry without consuming the failure budget."""

    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    now = _utc_now()
    with SessionLocal() as db:
        snapshot_query = db.query(
            TaskExecutionCommand.defer_count,
            TaskExecutionCommand.attempt_count,
        ).filter(
            TaskExecutionCommand.id == command_db_id,
            TaskExecutionCommand.status == COMMAND_PROCESSING,
            TaskExecutionCommand.claimed_by == runner_id,
        )
        if expected_attempt_count is not None:
            snapshot_query = snapshot_query.filter(
                TaskExecutionCommand.attempt_count == expected_attempt_count
            )
        snapshot = snapshot_query.first()
        if snapshot is None:
            return False
        observed_defer_count = int(snapshot.defer_count or 0)
        observed_attempt_count = int(snapshot.attempt_count or 0)
        defer_count = observed_defer_count + 1
        terminal = defer_count >= MAX_COMMAND_DEFERS
        updated = (
            db.query(TaskExecutionCommand)
            .filter(
                TaskExecutionCommand.id == command_db_id,
                TaskExecutionCommand.status == COMMAND_PROCESSING,
                TaskExecutionCommand.claimed_by == runner_id,
                TaskExecutionCommand.defer_count == observed_defer_count,
                TaskExecutionCommand.attempt_count == observed_attempt_count,
            )
            .update(
                {
                    TaskExecutionCommand.status: (
                        COMMAND_FAILED if terminal else COMMAND_PENDING
                    ),
                    TaskExecutionCommand.defer_count: defer_count,
                    TaskExecutionCommand.error: (
                        (
                            f"Deferred command exceeded retry budget: {reason}"
                            if terminal
                            else reason
                        )[:4000]
                    ),
                    TaskExecutionCommand.claimed_by: None,
                    TaskExecutionCommand.claim_expires_at: (
                        None if terminal else now + timedelta(seconds=1)
                    ),
                    TaskExecutionCommand.updated_at: now,
                    TaskExecutionCommand.completed_at: now if terminal else None,
                },
                synchronize_session=False,
            )
        )
        db.commit()
        if updated != 1:
            return False
    if not terminal:
        notify_task_command_dispatcher()
    return True


def retry_failed_task_command(
    db: Session,
    command_db_id: int,
) -> bool:
    """Reset one failed command without retargeting its immutable execution."""

    now = _utc_now()
    updated = (
        db.query(TaskExecutionCommand)
        .filter(
            TaskExecutionCommand.id == command_db_id,
            TaskExecutionCommand.status == COMMAND_FAILED,
        )
        .update(
            {
                TaskExecutionCommand.status: COMMAND_PENDING,
                TaskExecutionCommand.failure_count: 0,
                TaskExecutionCommand.defer_count: 0,
                TaskExecutionCommand.error: None,
                TaskExecutionCommand.result: None,
                TaskExecutionCommand.claimed_by: None,
                TaskExecutionCommand.claim_expires_at: None,
                TaskExecutionCommand.completed_at: None,
                TaskExecutionCommand.updated_at: now,
            },
            synchronize_session=False,
        )
    )
    db.commit()
    if updated == 1:
        notify_task_command_dispatcher()
        return True
    return False


def task_has_live_foreign_runner(
    task_id: int,
    *,
    runner_id: str | None = None,
) -> bool:
    """Return whether another process currently owns the task lease."""

    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    resolved_runner_id = runner_id or get_runner_id()
    now = _utc_now()
    with SessionLocal() as db:
        return (
            db.query(Task.id)
            .filter(
                Task.id == task_id,
                Task.runner_id.is_not(None),
                Task.runner_id != resolved_runner_id,
                Task.lease_expires_at.is_not(None),
                Task.lease_expires_at >= now,
            )
            .first()
            is not None
        )


CommandExecutor = Callable[[ClaimedTaskCommand], Awaitable[dict[str, Any] | None]]
CommandDisposition = Callable[[], bool]
_dispatcher_wakeup: asyncio.Event | None = None
_dispatcher_task: asyncio.Task[Any] | None = None
_dispatcher_loop: asyncio.AbstractEventLoop | None = None
_prompt_dispatch_tasks: set[asyncio.Task[bool]] = set()


def notify_task_command_dispatcher() -> None:
    wakeup = _dispatcher_wakeup
    loop = _dispatcher_loop
    if wakeup is not None and loop is not None and not loop.is_closed():
        loop.call_soon_threadsafe(wakeup.set)


async def _persist_task_command_disposition(
    command: ClaimedTaskCommand,
    *,
    disposition: str,
    operation: CommandDisposition,
) -> bool:
    """Persist one final state without retrying an exhausted pool checkout."""

    try:
        persisted = await run_db_io_cancellation_safe(operation)
    except Exception as exc:  # noqa: BLE001
        if not is_database_pool_timeout(exc):
            raise
        # The command is still fenced by claimed_by + attempt_count. Avoid an
        # immediate second checkout; expiry is the durable recovery boundary.
        logger.error(
            "task_id=%s component=task-command-disposition disposition=%s "
            "database pool checkout timed out; retaining command claim for "
            "expiry (command_id=%s, kind=%s, attempt=%s): %s",
            command.task_id,
            disposition,
            command.command_id,
            command.kind.value,
            command.attempt_count,
            exc,
            exc_info=True,
        )
        return False
    if not persisted:
        logger.warning(
            "Lost claim while persisting task command %s disposition=%s attempt=%s",
            command.id,
            disposition,
            command.attempt_count,
        )
    return persisted


async def dispatch_one_task_command(
    executor: CommandExecutor,
    *,
    command_db_id: int | None = None,
) -> bool:
    runner_id = get_runner_id()
    command = await run_db_io_cancellation_safe(
        lambda: _claim_task_command_isolated(
            runner_id=runner_id,
            command_db_id=command_db_id,
        )
    )
    if command is None:
        return False

    stop_event = asyncio.Event()
    heartbeat = asyncio.get_running_loop().create_task(
        _claim_heartbeat(command.id, runner_id, command.attempt_count, stop_event)
    )
    disposition_name: str | None = None
    disposition_operation: CommandDisposition | None = None
    heartbeat_outcome = TaskCommandClaimHeartbeatOutcome()
    heartbeat_cancellation: asyncio.CancelledError | None = None
    try:
        result = await executor(command)
    except asyncio.CancelledError:
        # Leave the processing claim intact. Another worker may reclaim it only
        # after the claim expires, which avoids concurrent replay on shutdown.
        raise
    except TaskCommandDeferred as exc:
        reason = str(exc)

        def persist_deferral() -> bool:
            return defer_task_command(
                command.id,
                runner_id,
                reason,
                expected_attempt_count=command.attempt_count,
            )

        disposition_name = "defer_task_command"
        disposition_operation = persist_deferral
    except TaskCommandRejected as exc:
        error = str(exc)
        rejection_result = (
            {"rejection_reason": exc.reason} if exc.reason is not None else None
        )

        def persist_rejection() -> bool:
            return fail_task_command(
                command.id,
                runner_id,
                error,
                force_terminal=True,
                expected_attempt_count=command.attempt_count,
                result=rejection_result,
            )

        disposition_name = "fail_task_command"
        disposition_operation = persist_rejection
    except Exception as exc:  # noqa: BLE001
        if is_database_pool_timeout(exc):
            # The handler already waited for an exhausted checkout. Persisting
            # command failure here would immediately request a second
            # connection. Keep the fenced processing claim intact; another
            # dispatcher may reclaim it only after claim expiry.
            logger.error(
                "task_id=%s component=task-command-handler database pool "
                "checkout timed out; retaining command claim for expiry "
                "(command_id=%s, kind=%s, attempt=%s): %s",
                command.task_id,
                command.command_id,
                command.kind.value,
                command.attempt_count,
                exc,
                exc_info=True,
            )
        else:
            logger.exception(
                "Task command %s (%s) failed on attempt %s",
                command.command_id,
                command.kind.value,
                command.attempt_count,
            )
            error = str(exc)

            def persist_failure() -> bool:
                return fail_task_command(
                    command.id,
                    runner_id,
                    error,
                    expected_attempt_count=command.attempt_count,
                )

            disposition_name = "fail_task_command"
            disposition_operation = persist_failure
    else:

        def persist_completion() -> bool:
            return finish_task_command(
                command.id,
                runner_id,
                result=result,
                expected_attempt_count=command.attempt_count,
            )

        disposition_name = "finish_task_command"
        disposition_operation = persist_completion
    finally:
        stop_event.set()
        heartbeat_outcome, heartbeat_cancellation = await await_task_settlement(
            heartbeat
        )

    with propagate_deferred_cancellation(heartbeat_cancellation):
        if heartbeat_outcome.requires_ttl_recovery:
            logger.error(
                "task_id=%s component=task-command-heartbeat claim is unresolved; "
                "retaining command claim for expiry (command_id=%s, kind=%s, "
                "attempt=%s, claim_lost=%s, pool_timeout=%s)",
                command.task_id,
                command.command_id,
                command.kind.value,
                command.attempt_count,
                heartbeat_outcome.claim_lost,
                heartbeat_outcome.pool_timeout,
            )
            return True
        if disposition_name is not None and disposition_operation is not None:
            await _persist_task_command_disposition(
                command,
                disposition=disposition_name,
                operation=disposition_operation,
            )
        return True


async def dispatch_task_command_promptly(
    executor: CommandExecutor,
    *,
    command_db_id: int,
) -> None:
    """Kick local consumption without tying WS receive to command execution.

    Fast commands usually finish during the short handoff window. Slow file or
    checkpoint work continues in its own task, allowing the socket receive loop
    to accept the next pause/message command instead of blocking behind it.
    """

    task = asyncio.get_running_loop().create_task(
        dispatch_one_task_command(executor, command_db_id=command_db_id)
    )
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=0.05)
    except asyncio.TimeoutError:
        _prompt_dispatch_tasks.add(task)
        task.add_done_callback(_consume_prompt_dispatch_result)
        return


def _consume_prompt_dispatch_result(task: asyncio.Task[bool]) -> None:
    """Retain and observe timed-out prompt dispatches until they finish."""

    _prompt_dispatch_tasks.discard(task)
    if task.cancelled():
        return
    try:
        task.result()
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Detached prompt task command dispatch failed: %s",
            exc,
            exc_info=(type(exc), exc, exc.__traceback__),
        )


async def _run_task_command_dispatcher_worker(executor: CommandExecutor) -> None:
    while True:
        wakeup = _dispatcher_wakeup
        if wakeup is None:
            return
        # Clear before checking for work. A notify that arrives during the DB
        # claim remains set, so an empty claim cannot erase that wakeup and
        # sleep while work is waiting.
        wakeup.clear()
        try:
            processed = await dispatch_one_task_command(executor)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "component=task-command-dispatcher command dispatch failed; "
                "worker continuing: %s",
                exc,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            try:
                await asyncio.wait_for(wakeup.wait(), timeout=DISPATCHER_IDLE_SECONDS)
            except asyncio.TimeoutError:
                pass
            continue
        if processed:
            continue
        try:
            await asyncio.wait_for(wakeup.wait(), timeout=DISPATCHER_IDLE_SECONDS)
        except asyncio.TimeoutError:
            pass


async def run_task_command_dispatcher(executor: CommandExecutor) -> None:
    """Recover queued commands without serializing unrelated tasks."""

    global _dispatcher_loop, _dispatcher_wakeup
    _dispatcher_loop = asyncio.get_running_loop()
    _dispatcher_wakeup = asyncio.Event()
    workers = [
        asyncio.create_task(_run_task_command_dispatcher_worker(executor))
        for _ in range(DISPATCHER_CONCURRENCY)
    ]
    try:
        await asyncio.gather(*workers)
    finally:
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)


def start_task_command_dispatcher(executor: CommandExecutor) -> asyncio.Task[Any]:
    global _dispatcher_task
    if _dispatcher_task is not None and not _dispatcher_task.done():
        return _dispatcher_task
    _dispatcher_task = asyncio.create_task(run_task_command_dispatcher(executor))
    return _dispatcher_task


async def stop_task_command_dispatcher() -> None:
    global _dispatcher_loop, _dispatcher_task, _dispatcher_wakeup
    prompt_tasks = list(_prompt_dispatch_tasks)
    for prompt_task in prompt_tasks:
        prompt_task.cancel()
    if prompt_tasks:
        await asyncio.gather(*prompt_tasks, return_exceptions=True)
    _prompt_dispatch_tasks.clear()

    task = _dispatcher_task
    _dispatcher_task = None
    _dispatcher_wakeup = None
    _dispatcher_loop = None
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def load_task_command(command_db_id: int) -> TaskCommandStateSnapshot | None:
    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        row = (
            db.query(
                TaskExecutionCommand.id,
                TaskExecutionCommand.status,
                TaskExecutionCommand.error,
                TaskExecutionCommand.result,
                TaskExecutionCommand.attempt_count,
                TaskExecutionCommand.failure_count,
                TaskExecutionCommand.defer_count,
            )
            .filter(TaskExecutionCommand.id == command_db_id)
            .first()
        )
        if row is None:
            return None
        result = row.result if isinstance(row.result, dict) else {}
        rejection_reason = result.get("rejection_reason")
        return TaskCommandStateSnapshot(
            command_db_id=int(row.id),
            status=str(row.status),
            error=str(row.error) if row.error is not None else None,
            rejection_reason=(
                rejection_reason if isinstance(rejection_reason, str) else None
            ),
            attempt_count=int(row.attempt_count or 0),
            failure_count=int(row.failure_count or 0),
            defer_count=int(row.defer_count or 0),
        )
