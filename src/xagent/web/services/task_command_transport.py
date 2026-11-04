"""Durable, cross-worker transport for task execution commands.

Ingress commits a command before acknowledging it: enqueue_task_command is
the committing wrapper most callers use, built on stage_task_command, the
staging primitive that deliberately does not commit so a caller can fold a
command into a larger transaction of its own. Every web worker runs the
same dispatcher; a database claim chooses one consumer while per-task ordering
keeps a later command behind an earlier unfinished one. Controls are the
exception: a cancel or pause may pass a command that waits for an admission
ticket, and a process at its dispatch cap lets a cancel or pause pass a
never-claimed command it holds for want of a slot, or a never-claimed batch
command it holds at its batch line (_unfinished_earlier_command).
"""

from __future__ import annotations

import asyncio
import enum
import json
import logging
import os
import re
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Awaitable, Callable, NamedTuple, cast

from sqlalchemy import and_, exists, false, func, or_, select, text, true
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Query, Session, aliased

from ...config import (
    DISPATCHER_FAIR_SHARE,
    get_dispatcher_fair_share_enabled,
    get_shared_task_execution_enabled,
    get_task_lease_heartbeat_seconds,
    get_task_lease_ttl_seconds,
)
from ..models.task import Task, TaskStatus, task_status_predicate
from ..models.task_admission import TaskAdmissionTicket
from ..models.task_command import (
    _CLAIMABLE_STATUS_CLAUSE,
    CLAIMABLE_COMMAND_STATUSES,
    TaskExecutionCommand,
)
from ..models.user import User
from .db_runtime import (
    await_task_settlement,
    caller_is_cancelling,
    is_database_pool_timeout,
    propagate_deferred_cancellation,
    run_db_io_cancellation_safe,
)
from .release_hold import release_hold
from .task_admission_execution import (
    AdmissionWaiting,
    admission_execution,
    return_to_admission_queue,
)
from .task_command_terminal_events import (
    TerminalTaskEventDraft,
    stage_terminal_event,
    terminal_event_draft_for_error,
)
from .task_dispatch_cap import (
    CAP_OVERTAKING_CONTROL_KINDS,
    CONTROL_KINDS,
    DISPATCH_CLASS_BATCH,
    DISPATCH_CLASS_INTERACTIVE,
    EXECUTION_OPENING_KINDS,
    DispatchCapSnapshot,
    DispatchClass,
    TenantListingRequired,
    build_dispatch_cap_snapshot,
)
from .task_execution_host import consumes_task_commands
from .task_lease_service import get_runner_id

if TYPE_CHECKING:
    from .task_coordinator_service import TaskLease as TaskOwnerLease

logger = logging.getLogger(__name__)

COMMAND_PENDING = "pending"
COMMAND_PROCESSING = "processing"
COMMAND_COMPLETED = "completed"
COMMAND_FAILED = "failed"
COMMAND_TERMINAL = (COMMAND_COMPLETED, COMMAND_FAILED)

COMMAND_ID_PATTERN = re.compile(r"[A-Za-z0-9._:-]{1,64}")
MAX_COMMAND_FAILURES = 5
# Since the budget was coupled to the lease TTL, this constant is the FLOOR
# of the effective defer budget, not the budget itself — compare against
# max_command_defers(), never against this constant directly.
MAX_COMMAND_DEFERS = 60


def max_command_defers() -> int:
    """The defer budget, a TTL-coupled heuristic rather than a guarantee.

    A deferral's shortest canonical wait is a task lease another holder has
    not released yet: on a non-RUNNING row the lease cannot be refreshed, so
    it clears within one TTL. The budget (spent roughly once per second —
    ``defer_task_command`` re-arms the claim one second out) must outlast
    that with real margin: a fixed 60 was silently smaller than any raised
    ``XAGENT_TASK_LEASE_TTL_SECONDS``, turning every long park into a
    terminal failure for an already-accepted command
    (xorbitsai/xagent-saas#952 B3). Twice the TTL keeps 100% headroom at
    every setting; the historical constant stays as the floor so short-TTL
    deployments keep their existing patience. Read per call, not cached:
    the TTL is env-driven and tests vary it.

    This mitigates rather than closes the exhaustion: a RUNNING row's lease
    is heartbeat-refreshed for the whole turn, so a command deferring behind
    a turn longer than the budget still fails terminally. The structural
    fix — patience as a wall-clock deadline against the command's
    ``created_at`` instead of a defer count — is #1995.
    """

    return max(MAX_COMMAND_DEFERS, get_task_lease_ttl_seconds() * 2)


DISPATCHER_IDLE_SECONDS = 0.5
DISPATCHER_CONCURRENCY = 4
DISPATCHER_RESTART_LIMIT = 3
DISPATCHER_RESTART_WINDOW_SECONDS = 600.0
# A failing claim (typically an unreachable database) backs off exponentially
# from DISPATCHER_IDLE_SECONDS up to this ceiling instead of retrying every
# idle tick, and repeats of the failure are logged at most once per interval.
# Both are per worker: with DISPATCHER_CONCURRENCY workers an outage logs up
# to that many lines per interval, and each line's consecutive_failures counts
# only that worker's failed claims.
DISPATCHER_FAILURE_BACKOFF_MAX_SECONDS = 30.0
DISPATCHER_FAILURE_LOG_INTERVAL_SECONDS = 60.0


class TaskCommandKind(str, enum.Enum):
    START = "start"
    RESUME_INPUT = "resume_input"
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
    """The command is durable but its downstream handoff is still pending.

    ``resend_safe`` is true only when the failed handoff proves this command
    never reached the downstream operation. The terminal-event projection can
    then tell the sender to retry without risking a duplicate.
    """

    def __init__(self, message: str, *, resend_safe: bool = False) -> None:
        super().__init__(message)
        self.resend_safe = resend_safe


class TaskCommandRejected(RuntimeError):
    """The downstream handoff reached a durable failed state.

    ``settle_cap_held`` asks the failing write to settle the commands this
    control passed while a dispatch cap held them
    (``settle_cap_held_commands_no_commit``); only a pause rejected because
    nothing of its task runs sets it. Such a pause still ends ``failed``
    although it stopped those commands; when the START it targets waits for
    an admission ticket instead, ``settle_queued_start_for_pause`` fails that
    START and completes the pause.
    """

    def __init__(
        self, message: str, *, reason: str | None = None, settle_cap_held: bool = False
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.settle_cap_held = settle_cap_held


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
    admission_required: bool = False


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


# Command ids reserved for the auto-resume sweeper. Episode bookkeeping does
# not count a RESUME with this prefix as a person acting (and the sweeper's
# claim-time check will trust one as its own), so no ingress may stage one.
# ``reserved=True`` is the sweeper's door into the namespace; it has no
# production caller until the sweeper lands, and tests exercise it.
AUTO_RESUME_COMMAND_PREFIX = "auto-resume:"


def _normalize_command_id(command_id: str, *, reserved: bool = False) -> str:
    """Strip and validate a caller-supplied command_id.

    Shared by every normalization site in this module so the accepted format
    -- 1-64 URL-safe characters -- cannot drift between them. The
    ``auto-resume:`` namespace is refused unless ``reserved`` says the caller
    is the auto-resume sweeper.
    """

    normalized = command_id.strip()
    if COMMAND_ID_PATTERN.fullmatch(normalized) is None:
        raise ValueError("command_id must be 1-64 URL-safe characters")
    if not reserved and normalized.startswith(AUTO_RESUME_COMMAND_PREFIX):
        raise ValueError(
            f"command_id prefix {AUTO_RESUME_COMMAND_PREFIX!r} is reserved"
        )
    return normalized


def _matches_existing(
    command: TaskExecutionCommand,
    *,
    actor_user_id: int | None,
    actor_subject: str | None,
    kind: TaskCommandKind,
    payload: dict[str, Any],
) -> bool:
    stored_payload: dict[str, Any] = (
        command.payload if isinstance(command.payload, dict) else {}
    )
    stored_actor_user_id = getattr(command, "actor_user_id", None)
    stored_actor_subject = getattr(command, "actor_subject", None)
    actors_match = (
        stored_actor_user_id is None and stored_actor_subject is None
        if actor_user_id is None
        else actor_subject is not None and stored_actor_subject == actor_subject
    )
    return (
        actors_match
        and str(command.kind) == kind.value
        and _canonical_payload(stored_payload) == _canonical_payload(payload)
    )


def _load_actor_subject(db: Session, actor_user_id: int | None) -> str | None:
    if actor_user_id is None:
        return None
    stored_subject = db.execute(
        select(User.actor_subject).where(User.id == actor_user_id)
    ).scalar_one_or_none()
    return str(stored_subject) if stored_subject is not None else None


def existing_task_command_payload_matches(
    db: Session,
    *,
    task_id: int,
    command_id: str,
    actor_user_id: int | None,
    kind: TaskCommandKind,
    payload: dict[str, Any],
) -> bool | None:
    """Read an existing command's payload identity without staging any writes.

    None means no command exists; False means its actor, kind or payload
    conflicts. Missing actor identities are not created by this read path.
    """
    with db.no_autoflush:
        existing = (
            db.query(TaskExecutionCommand)
            .filter(
                TaskExecutionCommand.task_id == task_id,
                TaskExecutionCommand.command_id == command_id,
            )
            .first()
        )
        if existing is None:
            return None
        return _matches_existing(
            existing,
            actor_user_id=actor_user_id,
            actor_subject=_load_actor_subject(db, actor_user_id),
            kind=kind,
            payload=payload,
        )


def _resolve_actor_subject(db: Session, actor_user_id: int | None) -> str | None:
    stored_subject = _load_actor_subject(db, actor_user_id)
    if actor_user_id is None or stored_subject is not None:
        return stored_subject
    candidate_subject = str(uuid.uuid4())
    (
        db.query(User)
        .filter(
            User.id == actor_user_id,
            User.actor_subject.is_(None),
        )
        .update(
            {User.actor_subject: candidate_subject},
            synchronize_session=False,
        )
    )
    return _load_actor_subject(db, actor_user_id)


def command_identity_matches_task(
    db: Session, task: Task, command: TaskExecutionCommand
) -> bool:
    """Revalidate the accepted actor and owner separately at execution time."""
    from ..models.agent import Agent

    owner = db.get(User, task.user_id)
    actor = (
        db.get(User, command.actor_user_id)
        if command.actor_user_id is not None
        else None
    )
    return bool(
        owner is not None
        and owner.actor_subject == command.task_owner_subject
        and owner.id == command.task_owner_user_id
        and actor is not None
        and actor.actor_subject == command.actor_subject
        and (
            actor.id == owner.id
            or actor.is_admin
            or (
                task.source == "sdk"
                and db.execute(
                    select(Agent.id).where(
                        Agent.id == task.agent_id, Agent.user_id == actor.id
                    )
                ).scalar_one_or_none()
                is not None
            )
        )
    )


def stage_task_command(
    db: Session,
    *,
    task_id: int,
    actor_user_id: int | None,
    command_id: str,
    kind: TaskCommandKind,
    payload: dict[str, Any],
    reply_host_id: str | None = None,
    reply_origin: str | None = None,
    target_run_id: str | None = None,
    reserved: bool = False,
) -> StagedTaskCommand:
    """Add an idempotent command row to the session without ending its transaction.

    ``reserved`` admits the auto-resume sweeper's command id namespace;
    every other caller leaves it False, so no ingress can stage one.

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
    4. Resolve the actor and task owner's stable internal subjects in this
       transaction, so idempotency and terminal delivery never treat a reused
       integer user id as the same identity.
    5. Check for an existing command with this (task_id, command_id) before
       adding a new row, so a repeated call is idempotent without relying on
       the unique constraint to reject it.
    6. Add and flush the new row last, so any remaining IntegrityError is
       attributable to the command insert itself. Classification runs once per
       new execution command, after the row is flushed. The deployment
       classifier's dispatch class is written to the row in the same
       transaction, before the admission ticket is staged. The fairness key
       is set on the inserted row (the owner key, or for a control command the
       key of the task's latest command); a classifier's key replaces it in
       the flush that writes the dispatch class. The priority is the one the
       classifier returned, else the task's declared value (NULL reads 0), and
       is written in that same flush. A control command of a task
       with no earlier command takes the owner key, which can differ from a
       deployment classifier's key for that task.

    Caller obligations:

    a. After the caller's own commit, it should invoke
       notify_task_command_dispatcher(). Skipping it does not lose the
       command -- the dispatcher's idle poll still recovers it -- but
       delivery silently degrades to up to DISPATCHER_IDLE_SECONDS of added
       latency instead of an immediate wakeup. While a worker is backing off
       after failed claims, its poll interval grows to at most
       DISPATCHER_FAILURE_BACKOFF_MAX_SECONDS, which then bounds that latency;
       a notify only reaches workers in the process that sent it.
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

    normalized_id = _normalize_command_id(command_id, reserved=reserved)
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
        select(
            Task.status,
            Task.runner_id,
            Task.run_id,
            Task.state_version,
            Task.user_id,
            Task.source,
            Task.channel_name,
            Task.requested_dispatch_class,
            Task.dispatch_priority,
            User.id.label("task_owner_row_id"),
            User.actor_subject.label("task_owner_subject"),
        )
        .select_from(Task)
        .outerjoin(
            User,
            and_(
                User.id == Task.user_id,
                User.created_at.is_not(None),
                Task.created_at.is_not(None),
                User.created_at <= Task.created_at,
            ),
        )
        .where(Task.id == resolved_task_id)
    ).one_or_none()
    if snapshot is None:
        raise TaskCommandTaskMissing(f"Task {task_id} not found")

    actor_subject = _resolve_actor_subject(db, actor_user_id)
    task_owner_subject = (
        str(snapshot.task_owner_subject)
        if snapshot.task_owner_subject is not None
        else None
    )
    if snapshot.task_owner_row_id is not None and task_owner_subject is None:
        task_owner_subject = (
            actor_subject
            if actor_user_id == int(snapshot.user_id)
            else _resolve_actor_subject(db, int(snapshot.user_id))
        )
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
            actor_subject=actor_subject,
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
    fairness_key = f"user:{int(snapshot.user_id)}"
    if kind.value not in EXECUTION_OPENING_KINDS:
        latest_key = db.execute(
            select(TaskExecutionCommand.fairness_key)
            .where(TaskExecutionCommand.task_id == resolved_task_id)
            .order_by(TaskExecutionCommand.id.desc())
            .limit(1)
        ).scalar_one_or_none()
        if latest_key is not None:
            fairness_key = str(latest_key)
    command = TaskExecutionCommand(
        task_id=resolved_task_id,
        actor_user_id=actor_user_id,
        actor_subject=actor_subject,
        task_owner_user_id=int(snapshot.user_id),
        task_owner_subject=task_owner_subject,
        command_id=normalized_id,
        kind=kind.value,
        payload=payload,
        target_run_id=target_run_id if target_run_id is not None else snapshot.run_id,
        target_state_version=int(snapshot.state_version or 0),
        target_runner_id=active_runner_id,
        reply_host_id=reply_host_id,
        reply_origin=reply_origin,
        status=COMMAND_PENDING,
        fairness_key=fairness_key,
    )
    db.add(command)
    db.flush()
    from .task_execution_event_writer import (
        append_fact_no_commit,
        uses_execution_events,
    )

    if uses_execution_events(db, resolved_task_id):
        append_fact_no_commit(
            db,
            task_id=resolved_task_id,
            kind="command_accepted",
            key=f"command:{normalized_id}",
            run_id=snapshot.run_id,
            payload={
                "command_id": normalized_id,
                "kind": kind.value,
                "payload": payload,
                "actor_user_id": actor_user_id,
                "target_state_version": int(snapshot.state_version or 0),
            },
        )
    from .task_execution_admission import (
        TaskClassificationView,
        classify_command_no_commit,
        stage_task_admission,
    )

    classification = None
    if command.kind in EXECUTION_OPENING_KINDS:
        # A background command is one the system stages itself (``reserved``).
        # A classifier may use the class of the task's previous command, so it
        # is read for every background command and for no other. The row just
        # flushed is excluded: it exists with the default class.
        previous_dispatch_class: DispatchClass | None = None
        if reserved:
            previous_dispatch_class = cast(
                "DispatchClass | None",
                db.execute(
                    select(TaskExecutionCommand.dispatch_class)
                    .where(
                        TaskExecutionCommand.task_id == resolved_task_id,
                        TaskExecutionCommand.kind.in_(EXECUTION_OPENING_KINDS),
                        TaskExecutionCommand.id != command.id,
                    )
                    .order_by(TaskExecutionCommand.id.desc())
                    .limit(1)
                ).scalar_one_or_none(),
            )
        classification = classify_command_no_commit(
            db,
            command,
            TaskClassificationView(
                task_id=resolved_task_id,
                source=snapshot.source,
                channel_name=snapshot.channel_name,
                owner_user_id=int(snapshot.user_id),
                background=reserved,
                previous_dispatch_class=previous_dispatch_class,
                requested_dispatch_class=snapshot.requested_dispatch_class,
                dispatch_priority=snapshot.dispatch_priority,
            ),
        )
        changed = False
        if classification.dispatch_class != command.dispatch_class:
            command.dispatch_class = classification.dispatch_class  # type: ignore[assignment]
            changed = True
        classified_key = classification.fairness_key
        if classified_key is not None and classified_key != command.fairness_key:
            command.fairness_key = classified_key  # type: ignore[assignment]
            changed = True
        priority = (
            classification.priority
            if classification.priority is not None
            else int(snapshot.dispatch_priority or 0)
        )
        if priority != command.priority:
            command.priority = priority
            changed = True
        if changed:
            # The session does not autoflush; write the class, the key and the
            # priority before anything else in the acceptance transaction can
            # read the row, whatever the caller does next.
            db.flush()
    stage_task_admission(
        db, command, classification.admission if classification else None
    )
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
        actor_subject = _load_actor_subject(db, actor_user_id)
        matches = _matches_existing(
            raced,
            actor_user_id=actor_user_id,
            actor_subject=actor_subject,
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
    reply_host_id: str | None = None,
    reply_origin: str | None = None,
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
            reply_host_id=reply_host_id,
            reply_origin=reply_origin,
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
    if get_shared_task_execution_enabled():
        return and_(
            TaskExecutionCommand.status.in_(CLAIMABLE_COMMAND_STATUSES),
            or_(
                TaskExecutionCommand.retry_available_at.is_(None),
                TaskExecutionCommand.retry_available_at <= now,
            ),
        )
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


# Presence prechecks: whether any command in a claimable status exists, so an
# idle scan builds no candidate query. The status set is a superset of what
# ``_claim_availability_predicate`` accepts in either mode (a test guards
# this), which makes the precheck a necessary condition for any candidate.
# Both read ``TaskExecutionCommand`` through the ORM, so an unflushed command
# in the session is flushed first, as it is for a ``Query.first()`` lookup, and
# both can use the claimable partial indexes.
_OPEN_COMMAND_PRESENCE = select(
    exists(select(1).select_from(TaskExecutionCommand).where(_CLAIMABLE_STATUS_CLAUSE))
)
# The same, plus whether any of them is a reply (RESUME_INPUT) to a waiting task.
_OPEN_COMMAND_AND_REPLY_PRESENCE = select(
    exists(select(1).select_from(TaskExecutionCommand).where(_CLAIMABLE_STATUS_CLAUSE)),
    exists(
        select(1)
        .select_from(TaskExecutionCommand)
        .where(
            _CLAIMABLE_STATUS_CLAUSE,
            TaskExecutionCommand.kind == TaskCommandKind.RESUME_INPUT.value,
        )
    ),
)


def _command_routing_predicate(runner_id: str, now: datetime) -> Any:
    if get_shared_task_execution_enabled():
        # An empty immutable target must not let another worker steal live
        # owner commands and spend the defer budget waiting for that owner.
        return or_(
            Task.runner_id == runner_id,
            Task.runner_id.is_(None),
            Task.lease_expires_at.is_(None),
            Task.lease_expires_at < now,
        )
    return or_(
        TaskExecutionCommand.target_runner_id.is_(None),
        TaskExecutionCommand.target_runner_id == runner_id,
        Task.runner_id == runner_id,
        Task.runner_id.is_(None),
        Task.lease_expires_at.is_(None),
        Task.lease_expires_at < now,
    )


def _held_below_cap(earlier: Any) -> Any:
    """Whether this process, at the batch line, will not start ``earlier``.

    That is a batch command, or a command that waits behind a pending,
    never-claimed batch command of its task that would open an execution:
    commands of a task run in order. ``earlier`` is an alias of
    ``TaskExecutionCommand``; callers add the never-claimed condition for
    ``earlier`` itself.
    """

    head = aliased(TaskExecutionCommand)
    return or_(
        earlier.dispatch_class == DISPATCH_CLASS_BATCH,
        exists(
            select(1).where(
                head.task_id == earlier.task_id,
                head.id <= earlier.id,
                head.status == COMMAND_PENDING,
                head.attempt_count == 0,
                head.kind.in_(sorted(EXECUTION_OPENING_KINDS)),
                head.dispatch_class == DISPATCH_CLASS_BATCH,
            )
        ),
    )


def _unfinished_earlier_command(cap: DispatchCapSnapshot | None = None) -> Any:
    """An earlier command of the same task that still blocks this one.

    A cancel or pause may pass an earlier command that waits for an
    admission ticket. While ``cap`` is closed (the process starts no new
    execution) or batch-closed (it starts no batch command), a control in
    ``CAP_OVERTAKING_CONTROL_KINDS`` may also pass an earlier pending command
    that would open an execution, was never claimed, and whose task holds no
    slot in this process, when this process would not start it now. At the
    cap that is every such command. Below the cap a command counts as held
    when it is a batch command, or when it waits behind a held batch command
    of its task, because this process will not start it either; a control
    passes both. The control is applied first and its completion, or a
    pause's rejection because nothing of its task runs, settles the passed
    commands (``settle_cap_held_commands_no_commit``), which does not read
    the snapshot; a rejected cancel, or a pause rejected for any other reason
    (for example access denied, ``stale_run`` or invalid data), settles
    nothing. A command already claimed once and back in pending is not
    passed: it may already have had an effect. Callers that claim must pass
    the same snapshot to the candidate query and the claim UPDATE. Neither
    this SQL nor the owner lease keeps the passed command unclaimed while the
    control is processing: the lease lock lasts only for the claim
    transaction, and the lease owner is the process applying the control.
    What does is that an owner applies one command of a task at a time
    (``busy_command_tasks`` and ``_command_tasks`` in
    ``dispatch_one_task_command``, ``_command_lock`` in ``execute_command``).
    A claim path that bypasses the task coordinator must keep that rule.
    """
    from .task_execution_admission import waiting_admission

    earlier = aliased(TaskExecutionCommand)
    clauses = [
        earlier.task_id == TaskExecutionCommand.task_id,
        earlier.id < TaskExecutionCommand.id,
        earlier.status.notin_(COMMAND_TERMINAL),
        ~and_(
            TaskExecutionCommand.kind.in_(("cancel", "pause")),
            earlier.status == COMMAND_PENDING,
            waiting_admission(earlier.id),
        ),
    ]
    if cap is not None and cap.batch_closed:
        held = true() if cap.closed else _held_below_cap(earlier)
        clauses.append(
            ~and_(
                TaskExecutionCommand.kind.in_(sorted(CAP_OVERTAKING_CONTROL_KINDS)),
                earlier.status == COMMAND_PENDING,
                earlier.kind.in_(sorted(EXECUTION_OPENING_KINDS)),
                earlier.attempt_count == 0,
                held,
                TaskExecutionCommand.task_id.notin_(sorted(cap.active_ids)),
            )
        )
    return exists(select(1).where(*clauses))


def _claimable_query(
    db: Session,
    *,
    runner_id: str,
    command_db_id: int | None,
    cap: DispatchCapSnapshot | None = None,
    now: datetime | None = None,
) -> Query[Any]:
    """The commands this process could claim, oldest first.

    ``now`` is the time the availability and routing checks are judged at; a
    caller running several queries for one decision passes the same value to
    all of them. ``cap`` narrows the result to what the process can start: at
    its limit, controls and commands of tasks it already runs; between its
    batch line and its limit, those plus interactive commands. Callers that
    claim must pass the same snapshot to the candidate query and the claim
    UPDATE.
    """
    from .task_execution_admission import admission_eligible

    if now is None:
        now = _utc_now()
    query = (
        db.query(TaskExecutionCommand)
        .join(Task, Task.id == TaskExecutionCommand.task_id)
        .filter(
            # Keep START dormant until the entire shared host is enabled.
            (
                true()
                if get_shared_task_execution_enabled()
                else TaskExecutionCommand.kind.notin_(
                    (TaskCommandKind.START.value, TaskCommandKind.RESUME_INPUT.value)
                )
            ),
            _claim_availability_predicate(now),
            admission_eligible(),
            ~_unfinished_earlier_command(cap),
            _command_routing_predicate(runner_id, now),
        )
    )
    if cap is not None and cap.batch_closed:
        # A control of any task, or a command of a task that already holds a
        # slot, needs no new slot; at the limit only those are worth claiming.
        controls_or_active = or_(
            TaskExecutionCommand.kind.in_(sorted(CONTROL_KINDS)),
            TaskExecutionCommand.task_id.in_(sorted(cap.active_ids)),
        )
        if cap.closed:
            query = query.filter(controls_or_active)
        else:
            # On the batch line only interactive commands may also take a new
            # slot, so a batch command of a task this process does not run is
            # left for a process with room.
            query = query.filter(
                or_(
                    controls_or_active,
                    TaskExecutionCommand.dispatch_class == DISPATCH_CLASS_INTERACTIVE,
                )
            )
    if command_db_id is not None:
        query = query.filter(TaskExecutionCommand.id == command_db_id)
    return query.order_by(TaskExecutionCommand.id.asc())


def command_processing_predicates(
    db: Session,
    command_db_id: int,
    runner_id: str,
    *,
    expected_attempt_count: int | None = None,
    require_live_claim: bool = False,
    owner_lease: TaskOwnerLease | None = None,
) -> tuple[Any, ...]:
    """Authorize a command write in the same transaction as its task owner.

    Attempt count isolates retries inside an owner; only the task has a lease.
    Legacy local execution retains its existing command claim contract.
    """
    predicates = [
        TaskExecutionCommand.id == command_db_id,
        TaskExecutionCommand.status == COMMAND_PROCESSING,
    ]
    if get_shared_task_execution_enabled():
        from .task_coordinator_runtime import current_task_coordinator
        from .task_coordinator_service import lock_task_lease_no_commit

        task_id = db.scalar(
            select(TaskExecutionCommand.task_id).where(
                TaskExecutionCommand.id == command_db_id
            )
        )
        if owner_lease is None:
            coordinator = (
                current_task_coordinator(task_id) if task_id is not None else None
            )
            owner_lease = coordinator.lease if coordinator is not None else None
        if (
            owner_lease is None
            or owner_lease.task_id != task_id
            or owner_lease.runner_id != runner_id
            or not lock_task_lease_no_commit(db, owner_lease)
        ):
            return (false(),)
    else:
        predicates.append(TaskExecutionCommand.claimed_by == runner_id)
        if require_live_claim:
            predicates.append(TaskExecutionCommand.claim_expires_at > _utc_now())
    if expected_attempt_count is not None:
        predicates.append(TaskExecutionCommand.attempt_count == expected_attempt_count)
    return tuple(predicates)


def claim_task_command(
    db: Session,
    *,
    runner_id: str | None = None,
    command_db_id: int | None = None,
    cap: DispatchCapSnapshot | None = None,
) -> ClaimedTaskCommand | None:
    """Atomically claim the oldest eligible command for this worker.

    ``cap`` is the snapshot the candidate was picked under; the lookup applies
    the same narrowing to it. Without ``command_db_id``, one presence statement
    first checks that any claimable-status command exists, and nothing else is
    built when none does.
    """

    if not consumes_task_commands():
        return None
    if command_db_id is None and not db.execute(_OPEN_COMMAND_PRESENCE).scalar_one():
        return None
    resolved_runner_id = runner_id or get_runner_id()
    candidate = _claimable_query(
        db,
        runner_id=resolved_runner_id,
        command_db_id=command_db_id,
        cap=cap,
    ).first()
    if candidate is None:
        return None

    shared = get_shared_task_execution_enabled()
    if shared:
        from .task_coordinator_runtime import current_task_coordinator
        from .task_coordinator_service import lock_task_lease_no_commit

        coordinator = current_task_coordinator(int(candidate.task_id))
        if (
            coordinator is None
            or coordinator.lease is None
            or coordinator.lease.runner_id != resolved_runner_id
            or not lock_task_lease_no_commit(db, coordinator.lease)
        ):
            return None
        from .task_execution_admission import reserve_task_admission

        if not reserve_task_admission(db, int(candidate.id), coordinator.lease):
            db.rollback()
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
            ~_unfinished_earlier_command(cap),
            routable_task,
        )
        .update(
            {
                TaskExecutionCommand.status: COMMAND_PROCESSING,
                TaskExecutionCommand.claimed_by: None if shared else resolved_runner_id,
                TaskExecutionCommand.claim_expires_at: None if shared else expires,
                TaskExecutionCommand.retry_available_at: None,
                # The only write that moves a command to processing, and it
                # always counts the attempt: _unfinished_earlier_command and
                # settle_cap_held_commands_no_commit read attempt_count == 0 as
                # "never claimed". A new path that claims or executes a
                # command must count it the same way.
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
    fresh = (
        db.query(TaskExecutionCommand)
        .filter(TaskExecutionCommand.id == int(candidate.id))
        .populate_existing()
        .one()
    )
    admission_required = db.get(TaskAdmissionTicket, fresh.id) is not None
    payload: dict[str, Any] = fresh.payload if isinstance(fresh.payload, dict) else {}
    command = ClaimedTaskCommand(
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
        admission_required=admission_required,
        failure_count=int(fresh.failure_count or 0),
        defer_count=int(fresh.defer_count or 0),
    )
    accepted_at = cast(datetime, fresh.created_at)
    db.commit()
    if admission_required:
        from .task_admission_observation import record_command_admission

        record_command_admission(command.id, command.attempt_count, accepted_at)
    return command


def _claim_task_command_isolated(
    *,
    runner_id: str,
    command_db_id: int | None,
    cap: DispatchCapSnapshot | None = None,
) -> ClaimedTaskCommand | None:
    """Claim in a worker-owned short session and return a detached snapshot."""

    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        return claim_task_command(
            db,
            runner_id=runner_id,
            command_db_id=command_db_id,
            cap=cap,
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


def finish_task_command_no_commit(
    db: Session,
    command_db_id: int,
    runner_id: str,
    *,
    result: dict[str, Any] | None = None,
    expected_attempt_count: int | None = None,
    require_live_claim: bool = False,
    owner_lease: TaskOwnerLease | None = None,
) -> bool:
    """Stage completion; START commits this together with its execution lease."""
    now = _utc_now()
    query = db.query(TaskExecutionCommand).filter(
        *command_processing_predicates(
            db,
            command_db_id,
            runner_id,
            expected_attempt_count=expected_attempt_count,
            require_live_claim=require_live_claim,
            owner_lease=owner_lease,
        ),
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
    if updated == 1:
        from .task_execution_admission import (
            settle_cancelled_admissions,
            settle_cap_held_commands_no_commit,
        )

        settle_cancelled_admissions(db, command_db_id)
        settle_cap_held_commands_no_commit(db, command_db_id)
        stage_terminal_event(db, command_db_id=command_db_id)
    return updated == 1


def finish_task_command(
    command_db_id: int,
    runner_id: str,
    *,
    result: dict[str, Any] | None = None,
    expected_attempt_count: int | None = None,
) -> bool:
    from ..models.database import get_session_local

    with get_session_local()() as db:
        updated = finish_task_command_no_commit(
            db,
            command_db_id,
            runner_id,
            result=result,
            expected_attempt_count=expected_attempt_count,
        )
        db.commit()
        return updated


def _records_message_outcome_unknown(result: Any) -> bool:
    from .chat_history_service import DELIVERY_OUTCOME_UNKNOWN

    return (
        isinstance(result, dict)
        and result.get("delivery_outcome") == DELIVERY_OUTCOME_UNKNOWN
    )


def fail_task_command(
    command_db_id: int,
    runner_id: str,
    error: str,
    *,
    force_terminal: bool = False,
    expected_attempt_count: int | None = None,
    result: dict[str, Any] | None = None,
    terminal_event: TerminalTaskEventDraft | None = None,
    settle_cap_held: bool = False,
) -> bool:
    """Retry a failed claim, or make it terminal after bounded attempts.

    ``settle_cap_held`` settles, in the same transaction as a terminal
    failure, the commands the control passed while a dispatch cap held them.
    """

    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    now = _utc_now()
    with SessionLocal() as db:
        handoff = None
        if get_shared_task_execution_enabled():
            handoff = (
                db.query(TaskExecutionCommand)
                .filter(
                    TaskExecutionCommand.id == command_db_id,
                    TaskExecutionCommand.kind.in_(
                        (
                            TaskCommandKind.START.value,
                            TaskCommandKind.RESUME_INPUT.value,
                        )
                    ),
                )
                .first()
            )
        if handoff is not None:
            # Match acceptance/handoff lock order: task, then command.
            db.execute(
                select(Task.id).where(Task.id == handoff.task_id).with_for_update()
            ).first()
        ownership = command_processing_predicates(
            db,
            command_db_id,
            runner_id,
            expected_attempt_count=expected_attempt_count,
        )
        snapshot_query = db.query(
            TaskExecutionCommand.failure_count,
            TaskExecutionCommand.attempt_count,
            TaskExecutionCommand.result,
        ).filter(
            TaskExecutionCommand.id == command_db_id,
            TaskExecutionCommand.status == COMMAND_PROCESSING,
            *ownership,
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
        if result is None and _records_message_outcome_unknown(snapshot.result):
            # A MESSAGE attempt recorded that its turn's ``dispatched`` row
            # means "outcome unknown" before it failed. That record is the
            # retry's only way to tell it from an accepted turn; keep it.
            result = snapshot.result
        failure_count = observed_failure_count + 1
        terminal = force_terminal or failure_count >= MAX_COMMAND_FAILURES
        updated = (
            db.query(TaskExecutionCommand)
            .filter(
                TaskExecutionCommand.id == command_db_id,
                TaskExecutionCommand.status == COMMAND_PROCESSING,
                *ownership,
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
                    (
                        TaskExecutionCommand.retry_available_at
                        if get_shared_task_execution_enabled()
                        else TaskExecutionCommand.claim_expires_at
                    ): (
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
        if updated == 1 and terminal:
            if handoff is not None:
                if handoff.kind == TaskCommandKind.START.value:
                    from .task_start_consumer import settle_failed_start_no_commit

                    settle_failed_start_no_commit(db, handoff)
            if settle_cap_held:
                from .task_execution_admission import (
                    settle_cap_held_commands_no_commit,
                )

                settle_cap_held_commands_no_commit(db, command_db_id)
            stage_terminal_event(
                db,
                command_db_id=command_db_id,
                draft=terminal_event,
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
    terminal_event: TerminalTaskEventDraft | None = None,
) -> bool:
    """Release a claim for retry without consuming the failure budget."""

    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    now = _utc_now()
    with SessionLocal() as db:
        ownership = command_processing_predicates(
            db,
            command_db_id,
            runner_id,
            expected_attempt_count=expected_attempt_count,
        )
        snapshot_query = db.query(
            TaskExecutionCommand.defer_count,
            TaskExecutionCommand.attempt_count,
        ).filter(
            TaskExecutionCommand.id == command_db_id,
            TaskExecutionCommand.status == COMMAND_PROCESSING,
            *ownership,
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
        terminal = defer_count >= max_command_defers()
        updated = (
            db.query(TaskExecutionCommand)
            .filter(
                TaskExecutionCommand.id == command_db_id,
                TaskExecutionCommand.status == COMMAND_PROCESSING,
                *ownership,
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
                    (
                        TaskExecutionCommand.retry_available_at
                        if get_shared_task_execution_enabled()
                        else TaskExecutionCommand.claim_expires_at
                    ): (None if terminal else now + timedelta(seconds=1)),
                    TaskExecutionCommand.updated_at: now,
                    TaskExecutionCommand.completed_at: now if terminal else None,
                },
                synchronize_session=False,
            )
        )
        if updated == 1 and terminal:
            stage_terminal_event(
                db,
                command_db_id=command_db_id,
                draft=terminal_event,
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

    from .task_execution_admission import prepare_task_admission_retry

    prepare_task_admission_retry(db, command_db_id)

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
                TaskExecutionCommand.retry_available_at: None,
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


def task_has_live_runner(
    task_id: int,
    *,
    expected_run_id: str | None,
) -> bool:
    """Return whether the target run currently has an unexpired task lease.

    ``expected_run_id`` is required and has no wildcard: an omitted default
    would let a lease belonging to a different run stand in as idempotency
    evidence for the run actually being asked about. An explicit ``None``
    means the row's own run id is NULL, matching how ``expected_run_id`` reads
    on ``BackgroundTaskManager.resume_admission_state``. Callers that really
    do want "any run" should ask ``task_has_live_foreign_runner`` instead.
    """

    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    now = _utc_now()
    with SessionLocal() as db:
        query = db.query(Task.id).filter(
            Task.id == task_id,
            Task.status == TaskStatus.RUNNING,
            Task.runner_id.is_not(None),
            Task.lease_expires_at.is_not(None),
            Task.lease_expires_at >= now,
        )
        return query.filter(Task.run_id == expected_run_id).first() is not None


@dataclass(frozen=True)
class SettledTaskCommand:
    """The handler already committed its terminal command disposition."""

    result: dict[str, Any] | None = None


CommandExecutor = Callable[
    [ClaimedTaskCommand], Awaitable[dict[str, Any] | None | SettledTaskCommand]
]
CommandDisposition = Callable[[], bool]
_dispatcher_wakeup: asyncio.Event | None = None
_dispatcher_slot_freed: asyncio.Event | None = None
_dispatcher_task: asyncio.Task[Any] | None = None
_dispatcher_loop: asyncio.AbstractEventLoop | None = None
_prompt_dispatch_tasks: set[asyncio.Task[bool]] = set()


def notify_task_command_dispatcher() -> None:
    wakeup = _dispatcher_wakeup
    loop = _dispatcher_loop
    if wakeup is not None and loop is not None and not loop.is_closed():
        loop.call_soon_threadsafe(wakeup.set)


def notify_dispatcher_slot_freed() -> None:
    """Tell the dispatcher an execution ended or was cancelled.

    Separate from ``notify_task_command_dispatcher``: a freed slot says nothing
    about the database, so only an idle wait listens to it.
    """

    slot_freed = _dispatcher_slot_freed
    loop = _dispatcher_loop
    if slot_freed is not None and loop is not None and not loop.is_closed():
        loop.call_soon_threadsafe(slot_freed.set)


async def _wait_for_work(
    wakeup: asyncio.Event, slot_freed: asyncio.Event | None, timeout: float
) -> None:
    """Sleep until a new command is announced, a slot frees, or ``timeout``."""

    waiters = {asyncio.ensure_future(wakeup.wait())}
    if slot_freed is not None:
        waiters.add(asyncio.ensure_future(slot_freed.wait()))
    try:
        await asyncio.wait(
            waiters, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
        )
    finally:
        for waiter in waiters:
            waiter.cancel()


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


def _owner_lease_recovery_settled(runner_id: str, now: datetime) -> Any:
    """Skip tasks whose RUNNING owner is another runner with an expired lease.

    Such a task needs lease recovery before acquisition. Our own owner can
    still process commands: expiry alone does not revoke its immutable lease
    identity.
    """
    return or_(
        Task.runner_id == runner_id,
        task_status_predicate.ne(TaskStatus.RUNNING),
        Task.lease_expires_at.is_(None),
        Task.lease_expires_at >= now,
    )


def _dispatcher_candidate_query(
    db: Session,
    *,
    runner_id: str,
    command_db_id: int | None,
    shared: bool,
    cap: DispatchCapSnapshot | None = None,
    now: datetime | None = None,
) -> Query[Any]:
    """The database part of the dispatcher's candidate selection.

    ``_claimable_query``, plus the owner lease-recovery filter when
    ``shared``. The shared dispatcher's scan and
    ``count_claimable_task_commands`` both build on this, so a database
    predicate added here reaches both. Process-local state stays with the
    caller: the scan passes its dispatch-cap snapshot through ``cap`` and adds
    the busy-task list and the reply-first kind order itself; the count passes
    neither.
    Without shared execution ``claim_task_command`` looks up with
    ``_claimable_query`` alone, so a predicate for both modes belongs there.
    The ``now`` and ``cap`` a caller passes are used by every query it runs for
    one decision, and by both the claimable query and the lease-recovery
    filter; without a ``now`` the query reads the clock once.
    """
    now = now if now is not None else _utc_now()
    query = _claimable_query(
        db,
        runner_id=runner_id,
        command_db_id=command_db_id,
        cap=cap,
        now=now,
    )
    if shared:
        query = query.filter(_owner_lease_recovery_settled(runner_id, now))
    return query


# The last raw value of the fair-share switch and what it parsed to.
_fair_share_memo: tuple[str | None, bool] | None = None


def _fair_share_enabled() -> bool:
    """Whether dispatch ranks tenants, parsed again only when the raw value changes.

    Each call reads the environment variable once. A value equal to the one the
    last parse saw returns that parse's result without parsing or warning, so a
    wrong value is reported once however often it is read. Threads that read a
    new value at the same moment may each parse it, which at worst repeats the
    warning. The dispatch policy line parses the value on its own when it changes,
    so an invalid value is warned about there too.
    """
    global _fair_share_memo
    raw = os.getenv(DISPATCHER_FAIR_SHARE)
    memo = _fair_share_memo
    if memo is None or memo[0] != raw:
        memo = (raw, get_dispatcher_fair_share_enabled())
        _fair_share_memo = memo
    return memo[1]


# With the fair-share switch on, a scan with room asks each of the first
# FAIR_SHARE_TENANTS_PER_SCAN ranked tenants for the best command it can claim among
# the FAIR_SHARE_WINDOW highest-priority, then oldest, queued commands of one
# (tenant key, dispatch class). A (tenant key, class) that has none is not asked
# again for FAIR_SHARE_SKIP_SECONDS, unless its tenant is the only one ranked: no
# other tenant could take that turn, so every scan asks it again.
#
# A fallback round reads the queue as it always did, so that a command only the full
# scan can reach is not left behind. Fallback rounds aim at one in every
# FAIR_SHARE_FALLBACK_EVERY new slots that counted attempts open: a fair round that
# opens a slot adds one to a balance, a fallback round the balance brought that
# opens one takes FAIR_SHARE_FALLBACK_EVERY - 1 off, and attempts that read the
# balance together may all become fallback rounds, so the share holds over time
# rather than slot by slot.
# An attempt that opens no slot leaves the balance alone: one that joins a task this
# process already executes, takes a control command, has its slot refused or picks
# nothing. So every counted attempt whose scan finds a command also adds one to an
# attempt count, and only a fallback round that dispatches a command holding a slot
# token sets that count back to zero. An attempt that finds nothing is left out: as
# a fallback round it would have made the same oldest-first lookup and found nothing
# either, so it passed no command over, and polling a queue with nothing claimable
# must not bring a fallback round. Once FAIR_SHARE_FALLBACK_AFTER_ATTEMPTS attempts
# have been counted, every attempt is a fallback round until one does; an attempt
# decides its round before it is counted, so up to DISPATCHER_CONCURRENCY - 1 more
# may go by first. Only a fallback round whose first scan read the balance at its
# line takes FAIR_SHARE_FALLBACK_EVERY - 1 off when it opens a slot; one the attempt
# count alone brings leaves the balance alone, so a stretch of attempts that open no
# slot leaves no debt for the fair rounds to pay back. The balance goes below zero
# only when attempts that read it at its line together each open a slot, so it stays
# at or above -(FAIR_SHARE_FALLBACK_EVERY - 1) * (DISPATCHER_CONCURRENCY - 1).
FAIR_SHARE_WINDOW = 16
FAIR_SHARE_TENANTS_PER_SCAN = 4
FAIR_SHARE_SKIP_SECONDS = 2.0
FAIR_SHARE_FALLBACK_EVERY = 8
FAIR_SHARE_FALLBACK_AFTER_ATTEMPTS = 64

# The oldest queued command of each (tenant key, dispatch class), by skipping from
# one pair to the next through ix_task_commands_claimable_fair; a scan of the queue
# would read every queued row. Queued means pending or processing, claimable or not.
# Only commands that open an execution count (_OPENING_KINDS_SQL): a pause or cancel
# needs no slot, and the key it carries may be one its task no longer has, so it must
# not make a tenant of its own. The scan still reaches it, through the step that
# takes an older command needing no new slot.
_OPENING_KINDS_SQL = ", ".join(f"'{kind}'" for kind in sorted(EXECUTION_OPENING_KINDS))
_LIST_QUEUED_TENANTS_POSTGRESQL = text(
    f"""
    WITH RECURSIVE s(k, c, id) AS (
      (SELECT fairness_key, dispatch_class, id FROM task_execution_commands
        WHERE status IN ('pending', 'processing') AND kind IN ({_OPENING_KINDS_SQL})
        ORDER BY fairness_key, dispatch_class, id LIMIT 1)
      UNION ALL
      SELECT n.k, n.c, n.id FROM s CROSS JOIN LATERAL (
        SELECT fairness_key AS k, dispatch_class AS c, id
          FROM task_execution_commands
         WHERE status IN ('pending', 'processing')
           AND kind IN ({_OPENING_KINDS_SQL})
           AND (fairness_key, dispatch_class) > (s.k, s.c)
         ORDER BY fairness_key, dispatch_class, id LIMIT 1) n)
    SELECT k, c, id FROM s
    """
)


def _list_queued_tenants(db: Session) -> dict[str, dict[str, int]]:
    """The oldest queued command that opens an execution, per class, per tenant key."""
    if db.get_bind().dialect.name == "postgresql":
        rows = db.execute(_LIST_QUEUED_TENANTS_POSTGRESQL).all()
    else:
        rows = db.execute(
            select(
                TaskExecutionCommand.fairness_key,
                TaskExecutionCommand.dispatch_class,
                func.min(TaskExecutionCommand.id),
            )
            .where(
                TaskExecutionCommand.status.in_((COMMAND_PENDING, COMMAND_PROCESSING)),
                TaskExecutionCommand.kind.in_(sorted(EXECUTION_OPENING_KINDS)),
            )
            .group_by(
                TaskExecutionCommand.fairness_key, TaskExecutionCommand.dispatch_class
            )
        ).all()
    tenants: dict[str, dict[str, int]] = {}
    for key, dispatch_class, oldest in rows:
        tenants.setdefault(str(key), {})[str(dispatch_class)] = int(oldest)
    return tenants


def _tenants_in_band(
    tenants: dict[str, dict[str, int]], cap: DispatchCapSnapshot | None
) -> dict[str, dict[str, int]]:
    """The tenants a scan can take a new slot for under ``cap``.

    Between the batch line and the limit only interactive commands may open an
    execution, so only the interactive class of each tenant counts and a tenant
    without one drops out.
    """
    if cap is None or not cap.batch_closed:
        return tenants
    interactive: dict[str, dict[str, int]] = {
        key: {DISPATCH_CLASS_INTERACTIVE: classes[DISPATCH_CLASS_INTERACTIVE]}
        for key, classes in tenants.items()
        if DISPATCH_CLASS_INTERACTIVE in classes
    }
    return interactive


def _queued_tenant_window(key: str, dispatch_class: str) -> Any:
    """The ids of the highest-priority queued commands of one tenant key and class.

    These are the FAIR_SHARE_WINDOW commands with the highest priority, oldest
    first among equals. The key is bounded with ``>=`` and ``<=`` rather than
    ``=``: with an equality PostgreSQL has preferred the primary-key order to an
    index on (key, class, id).
    """
    return (
        select(TaskExecutionCommand.id)
        .where(
            TaskExecutionCommand.status.in_((COMMAND_PENDING, COMMAND_PROCESSING)),
            TaskExecutionCommand.fairness_key >= key,
            TaskExecutionCommand.fairness_key <= key,
            TaskExecutionCommand.dispatch_class == dispatch_class,
        )
        .order_by(
            TaskExecutionCommand.fairness_key,
            TaskExecutionCommand.dispatch_class,
            TaskExecutionCommand.priority.desc(),
            TaskExecutionCommand.id,
        )
        .limit(FAIR_SHARE_WINDOW)
    )


def _in_tenant_window(query: Query[Any], key: str, dispatch_class: str) -> Query[Any]:
    """``query`` narrowed to the window of one tenant key and class, best row first.

    The claimable order by id is replaced by priority then id, so the first row is
    the highest-priority, then oldest, claimable command of the window.
    """
    return (
        query.filter(
            TaskExecutionCommand.id.in_(_queued_tenant_window(key, dispatch_class))
        )
        .order_by(None)
        .order_by(TaskExecutionCommand.priority.desc(), TaskExecutionCommand.id)
    )


def _older_controls(query: Query[Any], before_id: int) -> Query[Any]:
    """``query`` narrowed to pause and cancel commands older than ``before_id``.

    ix_task_commands_claimable_kind serves it and holds claimable rows only, so it
    reads the claimable controls, not the backlog.
    """
    return query.filter(
        TaskExecutionCommand.kind.in_(sorted(CONTROL_KINDS)),
        TaskExecutionCommand.id < before_id,
    )


def _older_of_tasks(
    claimable: Callable[[], Query[Any]], before_id: int, task_ids: frozenset[int]
) -> Query[Any]:
    """The claimable commands of ``task_ids`` older than ``before_id``, narrowed
    to the oldest one.

    Every running task is asked for its own oldest claimable command and the
    smallest of those ids is taken, so each task's lookup reads
    ix_task_commands_unfinished_task in id order, and the check that a command
    has no earlier unfinished command of its task reads that index too. The
    command-index rows each lookup reads are its own task's unfinished
    commands, whatever the length of the backlog or of the settled history.
    """
    running = aliased(Task, name="running_task")
    oldest_of_task = (
        claimable()
        .filter(
            TaskExecutionCommand.task_id == running.id,
            TaskExecutionCommand.id < before_id,
        )
        .order_by(None)
        .order_by(TaskExecutionCommand.id)
        .limit(1)
        .with_entities(TaskExecutionCommand.id)
        .correlate(running)
        .scalar_subquery()
    )
    oldest = (
        select(func.min(oldest_of_task))
        .select_from(running)
        .where(running.id.in_(sorted(task_ids)))
        .scalar_subquery()
    )
    return claimable().filter(TaskExecutionCommand.id == oldest)


def _older_needing_no_new_slot(
    claimable: Callable[[], Query[Any]], before_id: int, active: frozenset[int]
) -> Any:
    """The oldest claimable command older than ``before_id`` that needs no new slot.

    That is a control of any task, or a command of a task this process already
    runs. The two are looked up separately and the older found row is taken, so
    the control lookup keeps its index whatever the other costs; a control of a
    running task matches both lookups and is the same row. The second lookup is
    not sent when the process runs no task. None when neither finds a row.
    """
    found = [_older_controls(claimable(), before_id).first()]
    if active:
        found.append(_older_of_tasks(claimable, before_id, active).first())
    rows = [row for row in found if row is not None]
    return min(rows, key=lambda row: row.id) if rows else None


def _counted_tenants(
    tenants: dict[str, dict[str, int]], skipped: frozenset[tuple[str, str]]
) -> Mapping[str, int]:
    """The queued tenants the tenant cap counts, each with its oldest queued id.

    A tenant whose every queued class is in ``skipped`` is left out; the id is
    the oldest across all of its classes. A new read-only view; ``tenants`` is
    not changed.
    """
    return MappingProxyType(
        {
            key: min(classes.values())
            for key, classes in tenants.items()
            if any((key, name) not in skipped for name in classes)
        }
    )


def _not_capped(key: str, active_ids: frozenset[int]) -> Any:
    """The condition that leaves out the new work of the tenant ``key`` holds back.

    Controls and commands of tasks in ``active_ids`` need no new slot and pass.
    """
    return ~and_(
        TaskExecutionCommand.fairness_key == key,
        TaskExecutionCommand.kind.notin_(sorted(CONTROL_KINDS)),
        TaskExecutionCommand.task_id.notin_(sorted(active_ids)),
    )


class _FairState:
    """What fair-share selection remembers between scans of one process.

    Every field is guarded by ``_fair_state_lock``, which is held for memory reads
    and writes only, never across a database call or a log line.
    """

    def __init__(self) -> None:
        # The fallback rounds' balance: each new slot a counted fair round opens
        # adds one, each new slot a fallback round that read it at its line opens
        # takes off FAIR_SHARE_FALLBACK_EVERY - 1. It is negative only after
        # attempts that read it at its line together each opened a slot.
        self.since_fallback = 0
        # Counted attempts that found a command since a fallback round last
        # dispatched a command holding a slot token, whatever they then dispatched
        # and whether they dispatched at all.
        self.attempts_since_fallback = 0
        # Until when (``_fair_clock``) a (tenant key, dispatch class) is not asked.
        # When only one tenant is ranked, ranking neither reads nor writes it.
        self.skips: dict[tuple[str, str], float] = {}
        # Window queries that found nothing since the diagnostics line last asked.
        # Attempts that read the skip table before one of them records a miss each
        # query and count the same (tenant key, class), so this counts lookups.
        self.misses = 0
        # Since the diagnostics line last asked: counted attempts decided as
        # fallback rounds, and scans that ranked at least two tenants but took the
        # plain lookup because no tenant asked had a claimable command or none
        # could be asked.
        self.fallbacks = 0
        self.all_missed = 0


_fair_state = _FairState()
_fair_state_lock = threading.Lock()


def _fair_clock() -> float:
    return time.monotonic()


def drain_fair_misses() -> int:
    """The window queries that found nothing since the last call, and start again.

    The diagnostics line reports the count for its window.
    """
    with _fair_state_lock:
        misses = _fair_state.misses
        _fair_state.misses = 0
    return misses


def drain_fair_bypasses() -> tuple[int, int]:
    """The fallback rounds and the all-missed scans since the last call, and start again.

    The diagnostics line reports both counts for its window.
    """
    with _fair_state_lock:
        bypasses = (_fair_state.fallbacks, _fair_state.all_missed)
        _fair_state.fallbacks = 0
        _fair_state.all_missed = 0
    return bypasses


def _fair_live_skips() -> frozenset[tuple[str, str]]:
    """The (tenant key, class) pairs not to ask now, dropping the lapsed ones."""
    now = _fair_clock()
    with _fair_state_lock:
        skips = _fair_state.skips
        for pair in [pair for pair, until in skips.items() if until <= now]:
            del skips[pair]
        return frozenset(skips)


def _fair_record_miss(key: str, dispatch_class: str, skip: bool = True) -> None:
    """Count a window query that found nothing and, with ``skip``, skip the pair."""
    until = _fair_clock() + FAIR_SHARE_SKIP_SECONDS
    with _fair_state_lock:
        if skip:
            _fair_state.skips[(key, dispatch_class)] = until
        _fair_state.misses += 1


def _fair_count_attempt(attempt: _FairAttempt) -> None:
    """Add a counted attempt whose deciding scan found a command to the attempt count.

    Called by that scan before it returns the command, so once per attempt: a
    rescan happens only after a scan found one. An attempt that finds nothing is
    not added (see the comment above FAIR_SHARE_WINDOW). The lock is held for the
    memory update only.
    """
    if not attempt.counted:
        return
    with _fair_state_lock:
        _fair_state.attempts_since_fallback += 1


def _fair_count_dispatch(attempt: _FairAttempt, opened: bool) -> None:
    """Record the dispatch of a counted attempt that is about to apply its command.

    Called on the event loop once the attempt holds its slot token and the task's
    ownership check has passed, with no await in between; a control command takes
    no slot token and is not recorded here. ``opened`` is whether
    no execution of this process held the task's slot at that moment, so that
    applying it is expected to take a new one; an execution that ends before the
    command is applied leaves it counted as joining. An attempt whose slot is refused or
    released, or whose ownership check raises, never gets here; one that does is
    recorded even if applying the command then fails. A dispatch that opens a slot
    moves the balance: a fallback round whose first scan read the balance at its
    line takes FAIR_SHARE_FALLBACK_EVERY - 1 off, one the attempt count alone
    brought leaves it alone, and any other round adds one. A fallback round's
    dispatch also sets the attempt count back to zero, whether or not it opens a
    slot; the attempt itself was counted by the scan that found its command
    (``_fair_count_attempt``). The lock is held for the memory update only.
    """
    if not attempt.counted:
        return
    with _fair_state_lock:
        if attempt.fallback:
            _fair_state.attempts_since_fallback = 0
            if opened and attempt.at_balance_line:
                _fair_state.since_fallback -= FAIR_SHARE_FALLBACK_EVERY - 1
        elif opened:
            _fair_state.since_fallback += 1


@dataclass
class _FairAttempt:
    """What one dispatch attempt carries from one scan to the next.

    ``decided`` is set by the first scan, which fixes whether the attempt is a
    fallback round; a rescan keeps that decision. ``counted`` is set when that
    scan had room: a counted attempt is added to the attempt count once that scan
    finds a command (``_fair_count_attempt``), and only a counted attempt is
    recorded once it dispatches (``_fair_count_dispatch``). A scan the
    presence precheck ends because no command is open never reaches that
    decision, so it leaves ``decided`` unset. ``at_balance_line`` is set when that
    scan read the balance at its line, so that the attempt's dispatch takes its
    share off the balance. ``tenants`` is the listing of the queue, taken by the
    first scan that ranks or that is between the batch line and the limit, and
    reused by the rescans of the same attempt.
    """

    decided: bool = False
    counted: bool = False
    fallback: bool = False
    at_balance_line: bool = False
    tenants: dict[str, dict[str, int]] | None = None


def _new_fair_attempt() -> _FairAttempt | None:
    """The attempt record for one dispatch attempt, or None with the switch off."""
    return _FairAttempt() if _fair_share_enabled() else None


def _ranks_tenants(
    attempt: _FairAttempt, command_db_id: int | None, cap: DispatchCapSnapshot | None
) -> bool:
    """Whether this scan ranks tenants; the first scan fixes the attempt's round.

    A scan for one command, or with no room, reads the queue as it always did
    and is not counted. Otherwise the first scan marks the attempt counted and,
    under the lock, makes the attempt a fallback round for all its scans when the
    balance has reached FAIR_SHARE_FALLBACK_EVERY - 1 or the attempt count has
    reached FAIR_SHARE_FALLBACK_AFTER_ATTEMPTS. So once that many counted attempts
    have found a command since a fallback round last dispatched a command holding
    a slot token, every attempt is a fallback round until one does; only an
    attempt that read the balance at its line is marked to take its share off. It
    only reads both: ``_fair_count_attempt`` adds the attempt to the attempt count
    once the scan finds a command, and ``_fair_count_dispatch`` moves the balance
    and sets the attempt count back. A scan the presence precheck already ended
    never calls this, so it is not counted and does not set ``decided``.
    """
    has_room = command_db_id is None and (cap is None or not cap.closed)
    if not attempt.decided:
        attempt.decided = True
        if has_room:
            attempt.counted = True
            with _fair_state_lock:
                attempt.at_balance_line = (
                    _fair_state.since_fallback >= FAIR_SHARE_FALLBACK_EVERY - 1
                )
                attempt.fallback = (
                    attempt.at_balance_line
                    or _fair_state.attempts_since_fallback
                    >= FAIR_SHARE_FALLBACK_AFTER_ATTEMPTS
                )
                if attempt.fallback:
                    _fair_state.fallbacks += 1
    return has_room and not attempt.fallback


def _fair_pick(
    db: Session,
    attempt: _FairAttempt,
    cap: DispatchCapSnapshot | None,
    claimable: Callable[[], Query[Any]],
    oldest_claimable: Callable[[], Any],
    capped: str | None,
    skips: Callable[[], frozenset[tuple[str, str]]],
) -> Any:
    """The command a scan with room takes while tenants are ranked, or None.

    ``claimable()`` is the claimable query of the scan, ``oldest_claimable()`` its
    plain lookup of the oldest row. Tenants are ranked by the slots they hold in
    this process, then by their oldest queued command that opens an execution, and
    the first FAIR_SHARE_TENANTS_PER_SCAN that are not skipped and not ``capped``
    are asked in that order, the live skip table coming from ``skips()``: the
    first with a claimable command among its window supplies the best candidate,
    and the oldest claimable command older than that which needs no new slot
    (``_older_needing_no_new_slot``) is taken in its place. A (tenant key, class)
    with no claimable command in its window is missed and skipped for a while.
    With no tenant listed, the plain lookup decides. A single listed tenant is
    ranked too, so its windows are read by priority, but ranking neither reads
    nor writes the skip table for it: no other tenant could take its turn, so
    every scan asks each of its classes again. So it returns None only when
    ``oldest_claimable()`` does, which a fallback round reads instead: the attempt
    count leaves out the attempts that find nothing on that ground.

    The ranking uses the slots of this process only. Every process reads one queue,
    so on a tie they all prefer the tenant with the older oldest queued command. A
    later ranking that sees that tenant's new slot breaks the tie in this process,
    so the tenant tends to lead by one slot per process. That is not a strict bound:
    ``cap.occupancy`` is copied when the attempt takes its snapshot, so concurrent
    attempts can rank from the same older counts and prefer the same tenant before
    any of them sees the slot another one took. Fallback rounds take the oldest
    claimable command whatever its tenant and can add to the lead.

    Within a tenant key and class the window is read by priority. When both
    classes have a candidate, the one with the older side wins: the batch side by
    its candidate's id, the interactive side by the smallest id among the
    claimable interactive commands of its window. With all priorities equal this
    is the same order as comparing the two candidates' ids. Commands that need no
    new slot and are older than the winner (``_older_needing_no_new_slot``) still
    run first, and a class in the skip table is not asked, so the other class
    alone decides.

    The listing is taken once per attempt, so a rescan may rank a queue that has
    moved on; at worst that costs one more window query or one more recorded miss.
    A task the attempt already set aside (``busy``) that fills the window of a tenant
    makes that tenant miss and, unless it is the only tenant ranked, be skipped, as a
    task that cannot be claimed does.
    """
    if attempt.tenants is None:
        attempt.tenants = _list_queued_tenants(db)
    tenants = _tenants_in_band(attempt.tenants, cap)
    if not tenants:
        return oldest_claimable()
    alone = len(tenants) == 1
    skipped = frozenset() if alone else skips()
    occupancy = cap.occupancy if cap is not None else {}
    ranked = sorted(
        (
            occupancy.get(key, 0),
            min(classes.values()),
            key,
            [name for name in sorted(classes) if (key, name) not in skipped],
        )
        for key, classes in tenants.items()
    )
    best = None
    for _, _, key, classes in [
        entry for entry in ranked if entry[3] and entry[2] != capped
    ][:FAIR_SHARE_TENANTS_PER_SCAN]:
        hits = []
        ages: dict[str, int] = {}
        for dispatch_class in classes:
            window = _in_tenant_window(claimable(), key, dispatch_class)
            # Only the interactive side needs every claimable row of its window.
            if dispatch_class == DISPATCH_CLASS_INTERACTIVE:
                rows = window.all()
            else:
                rows = [row] if (row := window.first()) is not None else []
            if not rows:
                _fair_record_miss(key, dispatch_class, skip=not alone)
                continue
            hits.append(rows[0])
            ages[dispatch_class] = min(row.id for row in rows)
        if hits:
            best = min(hits, key=lambda row: ages[row.dispatch_class])
            break
    if best is None:
        # A tenant ranked alone falls back with no other tenant passed over, so
        # only a scan that ranked at least two is counted as all missed.
        if not alone:
            with _fair_state_lock:
                _fair_state.all_missed += 1
        return oldest_claimable()
    active = cap.active_ids if cap is not None else frozenset()
    older = _older_needing_no_new_slot(claimable, best.id, active)
    return best if older is None else older


class CommandCandidate(NamedTuple):
    """The command a dispatch scan picked and what its slot decision needs."""

    id: int
    task_id: int
    kind: TaskCommandKind
    interactive: bool
    fairness_key: str
    # The oldest queued command id of each queued tenant the selection counted,
    # for the slot decision; None when the selection listed no tenants.
    queued_tenants: Mapping[str, int] | None = None


def _find_command_candidate(
    command_db_id: int | None,
    busy_task_ids: tuple[int, ...] = (),
    cap: DispatchCapSnapshot | None = None,
    attempt: _FairAttempt | None = None,
) -> CommandCandidate | None:
    """A reply to a waiting task, else the oldest command this process could claim now.

    Returns None when there is none. Every query here judges availability at
    one time, read once at the start, and narrows by the dispatch-cap snapshot
    ``cap`` the caller took; they are separate SELECTs, so they do not share
    one database snapshot. What the process may select depends on ``cap``: with
    room, everything claimable; between its batch line and its limit, controls,
    commands of tasks it already runs, and interactive commands; at its limit,
    controls and commands of tasks it already runs. Without ``command_db_id``
    and below the limit, one presence statement decides whether any
    claimable-status command exists and whether any of them is a reply; nothing
    is built when none exists, and the reply-first lookup runs only when a reply
    is open. At the limit the scan runs its single narrow query as before. The
    precheck is a necessary condition only: when commands are open but none is
    claimable, the full scan still runs, adding one indexed round trip. A
    command committed after the precheck is found by the next poll or the
    in-process notify, within the bound ``stage_task_command`` documents.
    With the fair-share switch on, the command ``_fair_pick`` chooses from the
    first-ranked tenant that has one is taken instead, and between the batch line
    and the limit the new work of the tenant the cap holds back is left out of
    every query.
    """
    from ..models.database import get_session_local

    runner_id = get_runner_id()
    now = _utc_now()
    capped: str | None = None
    live_skips: frozenset[tuple[str, str]] | None = None

    def skips() -> frozenset[tuple[str, str]]:
        # One copy of the skip table per scan.
        nonlocal live_skips
        if live_skips is None:
            live_skips = _fair_live_skips()
        return live_skips

    with get_session_local()() as db:

        def first_candidate(kind: TaskCommandKind | None = None) -> Any:
            # This scan serves only shared execution (dispatch_one_task_command).
            query = _dispatcher_candidate_query(
                db,
                runner_id=runner_id,
                command_db_id=command_db_id,
                shared=True,
                cap=cap,
                now=now,
            )
            if kind is not None:
                query = query.filter(TaskExecutionCommand.kind == kind.value)
            if capped is not None and cap is not None:
                query = query.filter(_not_capped(capped, cap.active_ids))
            return query.filter(
                TaskExecutionCommand.task_id.notin_(busy_task_ids)
            ).first()

        def claimable() -> Query[Any]:
            query = _dispatcher_candidate_query(
                db,
                runner_id=runner_id,
                command_db_id=None,
                shared=True,
                cap=cap,
                now=now,
            ).filter(TaskExecutionCommand.task_id.notin_(busy_task_ids))
            if capped is not None and cap is not None:
                query = query.filter(_not_capped(capped, cap.active_ids))
            return query

        row = None
        any_reply = False
        if command_db_id is None and (cap is None or not cap.closed):
            any_open, any_reply = db.execute(_OPEN_COMMAND_AND_REPLY_PRESENCE).one()
            if not any_open:
                return None
        deciding = attempt is not None and not attempt.decided
        ranking = attempt is not None and _ranks_tenants(attempt, command_db_id, cap)
        counted: Mapping[str, int] | None = None
        if (
            attempt is not None
            and command_db_id is None
            and cap is not None
            and cap.in_reserve_band
        ):
            # Between the lines the tenant cap needs the queued tenants before
            # any query is built, replies included.
            if attempt.tenants is None:
                attempt.tenants = _list_queued_tenants(db)
            counted = _counted_tenants(attempt.tenants, skips())
            capped = cap.capped_tenant(counted)
        if any_reply:
            # A reply to a waiting task is served before any other claimable
            # command, oldest first. That deliberately includes older pause
            # and cancel commands of other tasks; each waits behind at most
            # one reply per waiting task, because the shared claimable query
            # still enforces each task's own order. This order applies to the
            # scan only: an admission ticket can still keep a reply behind
            # older work waiting in the same bucket. On the batch line a reply
            # classified batch is left out unless its task already runs here,
            # like any batch command. At the limit a reply that needs a new
            # slot is not claimable, so the narrow query decides. A reply of
            # the tenant the cap holds back waits unless its task already runs.
            row = first_candidate(TaskCommandKind.RESUME_INPUT)
        if row is None:
            if attempt is not None and ranking:
                row = _fair_pick(
                    db, attempt, cap, claimable, first_candidate, capped, skips
                )
            else:
                row = first_candidate()
        if row is None:
            return None
        if attempt is not None and deciding:
            _fair_count_attempt(attempt)
        listed = None if attempt is None else attempt.tenants
        if command_db_id is None and listed is not None and counted is None:
            counted = _counted_tenants(listed, skips())
        return CommandCandidate(
            int(row.id),
            int(row.task_id),
            TaskCommandKind(row.kind),
            row.dispatch_class == DISPATCH_CLASS_INTERACTIVE,
            str(row.fairness_key),
            counted,
        )


def count_claimable_task_commands(db: Session, *, runner_id: str) -> int:
    """Count the commands that pass the dispatcher's database checks.

    Builds on the same query as the dispatcher's candidate scan
    (``_dispatcher_candidate_query``), seen from ``runner_id``: claim
    availability, team admission, the earlier-unfinished-command gate and
    runner routing, plus the START/RESUME_INPUT exclusion without shared
    execution and the owner lease-recovery filter with it. Whether shared
    execution is on, and so which variant of each predicate applies, comes
    from the calling process's ``get_shared_task_execution_enabled()``, not
    from the worker that would claim. Without shared execution the
    dispatcher claims through ``claim_task_command``, whose lookup is
    ``_claimable_query`` alone, which is what this applies in that mode.

    The result is an upper bound on what a dispatcher in ``runner_id`` could
    claim now. Process-local state is not applied: the per-process dispatch
    cap (``DispatchCapSnapshot``), which at the cap, or for batch commands at
    its batch line, narrows the shared scan, or at the cap or on the batch line
    swaps a held command within one task for a later cancel or pause of that
    task that this count does not include, and reserves a slot once a
    candidate that opens an execution is picked; and the busy-task list.
    Neither are the checks made after a candidate is picked: whether the
    process consumes commands at all (``consumes_task_commands()``), task
    owner acquisition and lease checks, and the admission reservation that
    rechecks capacity under lock. Each of those checks can only turn a
    counted command away, and the cap's swap is one for one per task, so the
    result stays an upper bound on how many commands could be claimed.
    The tenant cap's prefilter, which leaves out the new work of the tenant the
    cap holds back between the batch line and the limit, belongs to the dispatch
    scan only and is not applied here either.

    This is not a backlog depth either. Only the first unfinished command of
    each task passes, except that a cancel or pause may pass an earlier
    command that still waits for admission. Under team admission a ticketed
    command passes when its ticket already holds a slot under the task's
    current owner, or when no older ticket of its bucket waits for the next
    slot, the bucket has a free slot and startup pacing allows it. An older
    ticket queued behind its own task's earlier command, or guidance joining
    a running execution, does not wait for a slot (``_older_waiter``). A
    command without a ticket is not gated by admission, and a guidance
    message that joins its task's running execution passes whatever its
    ticket says.

    The query runs in ``db``. Unlike the dispatcher's ``first()`` lookup, a
    count evaluates the correlated admission and earlier-command subqueries
    for every candidate row, so callers that need a bound should set a
    statement timeout on that session.
    """
    return (
        _dispatcher_candidate_query(
            db,
            runner_id=runner_id,
            command_db_id=None,
            shared=get_shared_task_execution_enabled(),
        )
        .order_by(None)
        .count()
    )


async def dispatch_one_task_command(
    executor: CommandExecutor,
    *,
    command_db_id: int | None = None,
) -> bool:
    if not get_shared_task_execution_enabled():
        # Combined mode dispatches without a cap, except while a planned
        # restart holds new execution: the held snapshot then applies the
        # closed-cap claim rules, so commands that would open an execution
        # stay pending for the next process while controls and commands of
        # active tasks are still applied here. Its host never seals, but a
        # seal reported by ``execution_status`` is enforced here too;
        # ``_dispatch_task_command`` already counts the whole attempt.
        if release_hold.sealed:
            return False
        return await _dispatch_task_command(
            executor,
            command_db_id=command_db_id,
            cap=build_dispatch_cap_snapshot() if release_hold.closed else None,
        )
    if not consumes_task_commands():
        return False
    # Shared execution: one token covers the whole attempt -- candidate query,
    # dispatch reservation, the ``registry.ensure`` wait (which may wait for an
    # old owner's close before creating a new one) and the apply. Checked and
    # registered in one synchronous step, so a seal either sees this attempt
    # in the ledger until it returns or refuses it here, before any side
    # effect.
    attempt = release_hold.try_begin_attempt(
        "command_attempt", str(command_db_id or "next")
    )
    if attempt is None:
        return False
    with attempt:
        return await _dispatch_shared_task_command(
            executor, command_db_id=command_db_id
        )


async def _dispatch_shared_task_command(
    executor: CommandExecutor,
    *,
    command_db_id: int | None,
) -> bool:
    from types import SimpleNamespace

    from .task_coordinator_runtime import (
        _CoordinatorClosed,
        _CoordinatorInterrupted,
        get_task_coordinator_registry,
    )
    from .task_execution import background_task_manager

    registry = get_task_coordinator_registry()
    skipped_task_ids = set(registry.busy_command_tasks())
    attempt = _new_fair_attempt()
    listing_rescanned = False
    while True:
        cap = build_dispatch_cap_snapshot()
        candidate = await run_db_io_cancellation_safe(
            lambda: _find_command_candidate(
                command_db_id, tuple(skipped_task_ids), cap, attempt=attempt
            )
        )
        if candidate is None:
            return False
        selected_id, task_id, kind, interactive, fairness_key, queued = candidate
        reservation = None
        if kind.value in EXECUTION_OPENING_KINDS:
            try:
                reservation = background_task_manager.reserve_dispatch_slot(
                    task_id,
                    policy=cap.policy,
                    interactive=interactive,
                    fairness_key=fairness_key,
                    queued_tenants=queued,
                    require_listing=attempt is not None and command_db_id is None,
                )
            except TenantListingRequired:
                # The snapshot this candidate was picked under sat below the
                # batch line, so the selection listed no tenants; the process
                # is between the lines now and the tenant cap needs the listing.
                # Nothing was reserved, so select once more under a fresh
                # snapshot, which lists them. The task is not set aside and the
                # attempt keeps its fallback decision; a second miss ends it.
                if listing_rescanned:
                    return False
                listing_rescanned = True
                continue
            if reservation is None:
                # A candidate picked under a snapshot is one this process could
                # start, so the slot is refused only when the state changed
                # since: more tasks became active (to the batch line for a batch
                # command, to the limit for any), the candidate's task left the
                # active set, or the tenant cap, read from the active set now,
                # holds the candidate's tenant back. An execution end signals
                # the dispatcher, which retries then. Releasing another
                # attempt's reservation signals nothing, so when the slot was
                # held by an attempt that ends without starting an execution,
                # the retry waits for the idle interval or the next notify.
                return False
        try:
            coordinator = await registry.ensure(task_id)
        except BaseException:
            background_task_manager.release_dispatch_slot(reservation)
            raise
        if coordinator is not None and not coordinator._command_tasks:
            break
        # Ownership can change after candidate selection. Skip this task
        # without spending its business retry budget; keep the exclusion
        # local to this scan so a later dispatch can reconsider it.
        background_task_manager.release_dispatch_slot(reservation)
        skipped_task_ids.add(task_id)

    # A control command holds no slot token: its attempt was counted when its scan
    # found it, and its dispatch moves neither the balance nor the attempt count.
    if attempt is not None and reservation is not None:
        _fair_count_dispatch(
            attempt, not background_task_manager.holds_execution(task_id)
        )

    async def apply() -> bool:
        try:
            return await _dispatch_task_command(
                executor, command_db_id=selected_id, cap=cap
            )
        except BaseException:
            coordinator.require_recovery()
            raise

    try:
        return await coordinator.execute_command(
            SimpleNamespace(task_id=task_id, kind=kind), apply
        )
    except (TaskCommandDeferred, _CoordinatorClosed):
        # Ownership health is not a business attempt or deferral.
        return False
    except _CoordinatorInterrupted:
        logger.warning(
            "task_id=%s component=task-command-dispatcher command application "
            "was cancelled by something other than the dispatcher (task owner "
            "close or a cancellation inside the command); any claim it held is "
            "left for recovery (command_db_id=%s, kind=%s)",
            task_id,
            selected_id,
            kind.value,
        )
        release_hold.record_unsettled(
            "command_claim",
            f"command:{selected_id}",
            f"{kind.value} application interrupted; claim left for recovery",
        )
        return False
    finally:
        background_task_manager.release_dispatch_slot(reservation)


async def _dispatch_task_command(
    executor: CommandExecutor,
    *,
    command_db_id: int | None = None,
    cap: DispatchCapSnapshot | None = None,
) -> bool:
    """Claim one command and apply it, counted as a release responsibility.

    Counted from before the claim: the claim commits in a worker thread, and
    the command it claims -- for example input to a task that is active when
    the cap snapshot was taken -- can open an execution only after that
    execution's predecessor has already left the manager's tables. Counting
    the whole attempt leaves no window where a claimed command is invisible.
    """
    with release_hold.track("command_dispatch", str(command_db_id or "next")):
        return await _claim_and_apply_task_command(
            executor, command_db_id=command_db_id, cap=cap
        )


async def _claim_and_apply_task_command(
    executor: CommandExecutor,
    *,
    command_db_id: int | None = None,
    cap: DispatchCapSnapshot | None = None,
) -> bool:
    runner_id = get_runner_id()
    command = await run_db_io_cancellation_safe(
        lambda: _claim_task_command_isolated(
            runner_id=runner_id,
            command_db_id=command_db_id,
            cap=cap,
        )
    )
    if command is None:
        return False

    stop_event = asyncio.Event()
    heartbeat = (
        None
        if get_shared_task_execution_enabled()
        else asyncio.get_running_loop().create_task(
            _claim_heartbeat(command.id, runner_id, command.attempt_count, stop_event)
        )
    )
    already_completed = False
    disposition_name: str | None = None
    disposition_operation: CommandDisposition | None = None
    heartbeat_outcome = TaskCommandClaimHeartbeatOutcome()
    heartbeat_cancellation: asyncio.CancelledError | None = None
    try:
        with admission_execution(
            command.id, command.task_id, command.admission_required
        ):
            result = await executor(command)
    except AdmissionWaiting:
        disposition_name = "return_to_admission_queue"

        def persist_admission_wait() -> bool:
            return return_to_admission_queue(
                command.id, runner_id, command.attempt_count
            )

        disposition_operation = persist_admission_wait
    except asyncio.CancelledError:
        # Leave the processing claim intact. Another worker may reclaim it only
        # after the claim expires, which avoids concurrent replay on shutdown.
        raise
    except TaskCommandDeferred as exc:
        reason = str(exc)
        terminal_event = terminal_event_draft_for_error(exc)

        def persist_deferral() -> bool:
            return defer_task_command(
                command.id,
                runner_id,
                reason,
                expected_attempt_count=command.attempt_count,
                terminal_event=terminal_event,
            )

        disposition_name = "defer_task_command"
        disposition_operation = persist_deferral
    except TaskCommandRejected as exc:
        error = str(exc)
        rejection_result = (
            {"rejection_reason": exc.reason} if exc.reason is not None else None
        )
        settle_cap_held = exc.settle_cap_held

        # An executor that bound its own presentation draft wins; the neutral
        # literal is only the fallback. The other two dispositions already
        # read the bound draft — hardcoding here silently discarded the
        # external input executor's identity-withholding draft
        # (xorbitsai/xagent-saas#952 M3).
        rejection_event = terminal_event_draft_for_error(exc) or TerminalTaskEventDraft(
            message_code=None,
            resend_safe=False,
        )

        def persist_rejection() -> bool:
            return fail_task_command(
                command.id,
                runner_id,
                error,
                force_terminal=True,
                expected_attempt_count=command.attempt_count,
                result=rejection_result,
                terminal_event=rejection_event,
                settle_cap_held=settle_cap_held,
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
            terminal_event = terminal_event_draft_for_error(exc)

            def persist_failure() -> bool:
                return fail_task_command(
                    command.id,
                    runner_id,
                    error,
                    expected_attempt_count=command.attempt_count,
                    terminal_event=terminal_event,
                )

            disposition_name = "fail_task_command"
            disposition_operation = persist_failure
    else:

        def persist_completion() -> bool:
            assert not isinstance(result, SettledTaskCommand)
            return finish_task_command(
                command.id,
                runner_id,
                result=result,
                expected_attempt_count=command.attempt_count,
            )

        if isinstance(result, SettledTaskCommand):
            already_completed = True
        else:
            disposition_name = "finish_task_command"
            disposition_operation = persist_completion
    finally:
        stop_event.set()
        if heartbeat is not None:
            heartbeat_outcome, heartbeat_cancellation = await await_task_settlement(
                heartbeat
            )

    with propagate_deferred_cancellation(heartbeat_cancellation):
        if heartbeat_outcome.requires_ttl_recovery and not already_completed:
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
            persisted = await _persist_task_command_disposition(
                command,
                disposition=disposition_name,
                operation=disposition_operation,
            )
            if not persisted:
                from .task_coordinator_runtime import current_task_coordinator

                coordinator = current_task_coordinator(command.task_id)
                if coordinator is not None:
                    coordinator.require_recovery()
        elif not already_completed and get_shared_task_execution_enabled():
            from .task_coordinator_runtime import current_task_coordinator

            coordinator = current_task_coordinator(command.task_id)
            assert coordinator is not None
            coordinator.require_recovery()
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

    if not consumes_task_commands():
        notify_task_command_dispatcher()
        return

    task = asyncio.get_running_loop().create_task(
        dispatch_one_task_command(executor, command_db_id=command_db_id)
    )
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=0.05)
    except asyncio.TimeoutError:
        _prompt_dispatch_tasks.add(task)
        task.add_done_callback(_consume_prompt_dispatch_result)
        return


def prompt_dispatch_count() -> int:
    """Detached prompt dispatches still running (``dispatch_task_command_promptly``)."""

    return sum(1 for task in _prompt_dispatch_tasks if not task.done())


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


def _dispatcher_clock() -> float:
    return time.monotonic()


def _dispatcher_failure_backoff_seconds(consecutive_failures: int) -> float:
    """Return the retry delay after ``consecutive_failures`` (>= 1) failures."""

    exponent = min(max(consecutive_failures, 1) - 1, 16)
    return min(
        DISPATCHER_IDLE_SECONDS * float(1 << exponent),
        DISPATCHER_FAILURE_BACKOFF_MAX_SECONDS,
    )


async def _run_task_command_dispatcher_worker(executor: CommandExecutor) -> None:
    # Invariant: this loop exits only when the worker task itself is cancelled
    # (or when stop_task_command_dispatcher cleared the wakeup event).
    # A CancelledError raised from inside dispatch that nobody aimed at this
    # task (for example a shared await on another task that was cancelled)
    # must not end the loop. It is not a failed claim, so it does not feed the
    # failure backoff below.
    async def idle_wait(
        wakeup: asyncio.Event, timeout: float = DISPATCHER_IDLE_SECONDS
    ) -> None:
        try:
            await asyncio.wait_for(wakeup.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass

    consecutive_failures = 0
    suppressed_failures = 0
    last_failure_logged_at: float | None = None
    while True:
        wakeup = _dispatcher_wakeup
        slot_freed = _dispatcher_slot_freed
        if wakeup is None:
            return
        # Clear before checking for work. A notify that arrives during the DB
        # claim remains set, so an empty claim cannot erase that wakeup and
        # sleep while work is waiting. The slot-freed event follows the same
        # rule.
        wakeup.clear()
        if slot_freed is not None:
            slot_freed.clear()
        try:
            processed = await dispatch_one_task_command(executor)
        except asyncio.CancelledError as exc:
            if caller_is_cancelling():
                raise
            logger.error(
                "component=task-command-dispatcher dispatch raised a "
                "cancellation not aimed at this worker; worker continuing",
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            await idle_wait(wakeup)
            continue
        except Exception as exc:  # noqa: BLE001
            consecutive_failures += 1
            retry_in = _dispatcher_failure_backoff_seconds(consecutive_failures)
            now = _dispatcher_clock()
            if (
                last_failure_logged_at is None
                or now - last_failure_logged_at
                >= DISPATCHER_FAILURE_LOG_INTERVAL_SECONDS
            ):
                logger.error(
                    "component=task-command-dispatcher command dispatch failed; "
                    "worker continuing (consecutive_failures=%s "
                    "suppressed_since_last_log=%s retry_in=%.2fs): %s",
                    consecutive_failures,
                    suppressed_failures,
                    retry_in,
                    exc,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
                last_failure_logged_at = now
                suppressed_failures = 0
            else:
                suppressed_failures += 1
            # A notify still cuts the backoff short: notifies follow a
            # committed command write, so new work is a strong hint that the
            # database is reachable again.
            await idle_wait(wakeup, retry_in)
            continue
        if consecutive_failures:
            logger.info(
                "component=task-command-dispatcher dispatch recovered after "
                "%s consecutive failure(s)",
                consecutive_failures,
            )
            consecutive_failures = 0
            suppressed_failures = 0
            last_failure_logged_at = None
        if processed:
            continue
        await _wait_for_work(wakeup, slot_freed, DISPATCHER_IDLE_SECONDS)


async def run_task_command_dispatcher(executor: CommandExecutor) -> None:
    """Recover queued commands without serializing unrelated tasks."""

    global _dispatcher_loop, _dispatcher_wakeup, _dispatcher_slot_freed
    _dispatcher_loop = asyncio.get_running_loop()
    _dispatcher_wakeup = asyncio.Event()
    _dispatcher_slot_freed = asyncio.Event()
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


def _observe_dispatcher_exit(task: asyncio.Task[Any]) -> None:
    if task is not _dispatcher_task:
        return  # stopped through stop_task_command_dispatcher
    try:
        task.result()
    except asyncio.CancelledError as error:
        logger.critical(
            "component=task-command-dispatcher dispatcher ended by a cancellation "
            "outside stop_task_command_dispatcher",
            exc_info=(type(error), error, error.__traceback__),
        )
    except BaseException as error:  # noqa: BLE001
        logger.critical(
            "component=task-command-dispatcher dispatcher exited unexpectedly",
            exc_info=(type(error), error, error.__traceback__),
        )
    else:
        logger.critical(
            "component=task-command-dispatcher dispatcher returned while still registered"
        )


def start_task_command_dispatcher(
    executor: CommandExecutor,
) -> asyncio.Task[Any] | None:
    global _dispatcher_task
    if not consumes_task_commands():
        return None
    if _dispatcher_task is not None and not _dispatcher_task.done():
        return _dispatcher_task
    _dispatcher_task = asyncio.create_task(run_task_command_dispatcher(executor))
    _dispatcher_task.add_done_callback(_observe_dispatcher_exit)
    return _dispatcher_task


async def supervise_task_command_dispatcher(
    executor: CommandExecutor, stop: asyncio.Event
) -> None:
    """Keep the already-started dispatcher running until ``stop`` is set.

    For worker entry points only. It never performs the first start (callers
    start the dispatcher first). An unexpected exit is restarted in place.
    An unexpected exit after DISPATCHER_RESTART_LIMIT restarts within
    DISPATCHER_RESTART_WINDOW_SECONDS raises RuntimeError instead. Whether the
    process is restarted afterwards is up to whatever runs it: the built-in
    worker pool stops the whole group when any child exits.
    """

    loop = asyncio.get_running_loop()
    restarts: list[float] = []
    while True:
        dispatcher = _dispatcher_task
        if dispatcher is None or dispatcher.done() or dispatcher.get_loop() is not loop:
            if consumes_task_commands():
                logger.warning(
                    "component=task-command-dispatcher no running dispatcher to "
                    "supervise; command consumption is not supervised in this "
                    "process"
                )
            await stop.wait()
            return
        waiter = asyncio.create_task(stop.wait())
        try:
            await asyncio.wait(
                {waiter, dispatcher}, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            if not waiter.done():
                waiter.cancel()
                await asyncio.gather(waiter, return_exceptions=True)
        if stop.is_set():
            return
        if _dispatcher_task is not dispatcher:
            logger.info(
                "component=task-command-dispatcher dispatcher stopped explicitly; "
                "supervision ends"
            )
            await stop.wait()
            return
        now = loop.time()
        restarts = [t for t in restarts if now - t < DISPATCHER_RESTART_WINDOW_SECONDS]
        if len(restarts) >= DISPATCHER_RESTART_LIMIT:
            raise RuntimeError(
                f"Task command dispatcher exited unexpectedly {len(restarts) + 1} "
                f"times within {DISPATCHER_RESTART_WINDOW_SECONDS:g} seconds; exiting"
            )
        restarts.append(now)
        logger.error(
            "component=task-command-dispatcher restarting dispatcher "
            "(restart %s of %s within %gs)",
            len(restarts),
            DISPATCHER_RESTART_LIMIT,
            DISPATCHER_RESTART_WINDOW_SECONDS,
        )
        start_task_command_dispatcher(executor)


async def stop_task_command_dispatcher() -> None:
    global _dispatcher_loop, _dispatcher_task, _dispatcher_wakeup
    global _dispatcher_slot_freed
    prompt_tasks = list(_prompt_dispatch_tasks)
    for prompt_task in prompt_tasks:
        prompt_task.cancel()
    if prompt_tasks:
        await asyncio.gather(*prompt_tasks, return_exceptions=True)
    _prompt_dispatch_tasks.clear()

    task = _dispatcher_task
    _dispatcher_task = None
    _dispatcher_wakeup = None
    _dispatcher_slot_freed = None
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
