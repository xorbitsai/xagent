"""Durable capacity waiting, separate from command failure/retry accounting.

Hosts classify commands at acceptance; workers enforce the persisted bucket.
Disjoint buckets can reserve capacity for different traffic classes. A ticket
holds capacity through the task owner's actual execution cleanup, including
after a terminal business status has already been published. Expiry alone is
never permission to reclaim capacity: the existing owner recovery must fence
out that acquisition first.

Capacity is charged per active execution, not per stamped ticket row. Every
ticket the task's current owner holds in one bucket is the same execution, so
a RESUME admitted while the previous incarnation's cleanup still holds its
ticket does not take a second slot there. Live guidance joins the running
execution without a slot from its own bucket, because the bucket classified a
queue position and the execution it joins already paid for its slot; it must
reserve one only if it becomes a new turn. Tickets carry no execution
identity, so the coordinator releases the ticket of each execution handle as
that handle finishes: an incarnation classified into another bucket frees the
previous bucket's slot once the previous incarnation has drained, while the
owner's settled and idle releases remain the backstop for tickets no handle
ever carried. Counting distinct tasks is serialized by the bucket row lock
taken before every stamp and by the exclusive owner stamp per task. The scan
and claim predicates below must agree on this contract, so every executor must
run the same revision before a host enables a policy. The snapshot counts every
held task, including the waiter's own; it is display only (#2700).
"""

from __future__ import annotations

import functools
import inspect
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, cast

from sqlalchemy import and_, delete, exists, func, or_, select, update
from sqlalchemy.orm import Session, aliased
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.selectable import ScalarSelect

from ...config import (
    get_dispatch_accept_requested_class_enabled,
    get_shared_task_execution_enabled,
)
from ..models.task import Task, TaskStatus
from ..models.task_admission import TaskAdmissionBucket, TaskAdmissionTicket
from ..models.task_command import TaskExecutionCommand
from . import task_admission_pacing
from .task_admission_observation import record_queue_full
from .task_admission_pacing import (
    StartupPacing,
    reserve_startup,
    stage_startup_pacing,
    startup_eligible,
)
from .task_coordinator_service import TaskLease
from .task_dispatch_cap import (
    DISPATCH_CLASS_BATCH,
    DISPATCH_CLASS_INTERACTIVE,
    DISPATCH_CLASSES,
    DispatchClass,
)

if TYPE_CHECKING:
    from .task_command_transport import ClaimedTaskCommand, SettledTaskCommand

logger = logging.getLogger(__name__)

# The inclusive range of a command priority; higher goes first.
PRIORITY_MIN = -10
PRIORITY_MAX = 10

_EXECUTION_KINDS = frozenset({"start", "resume", "resume_input", "message"})
_TERMINAL = ("completed", "failed")
_STOPPED_BEFORE_START = "Task stopped before execution started."
_CONTROL_STOPPED_COMMAND = "Task command stopped by a control request."


@dataclass(frozen=True)
class AdmissionPolicy:
    """One server-owned budget; all hosts must use the same configuration."""

    bucket: str
    capacity: int
    max_pending: int
    pacing: StartupPacing | None = None

    def __post_init__(self) -> None:
        if not self.bucket or len(self.bucket) > 255:
            raise ValueError("Admission bucket must contain 1-255 characters")
        if self.capacity < 1 or self.max_pending < 1:
            raise ValueError("Admission capacity and pending budget must be positive")


class AdmissionQueueFull(RuntimeError):
    """The host must translate this retryable refusal at its ingress boundary."""


class ExecutionHeld(AdmissionQueueFull):
    """New execution is paused on this process for a planned restart.

    Raised by ``release_hold.admit`` before any side effect. A subclass of the
    queue-full refusal on purpose: every ingress that already projects that
    retryable "not accepted" outcome onto its own protocol handles this one
    the same way, without a new wire code. The argument names the entry kind.
    """


@dataclass(frozen=True, kw_only=True)
class CommandClassification:
    """What a deployment decided for one new execution command at acceptance.

    ``fairness_key`` is the tenant the command is accounted to; None means the
    acceptance step gives it the task owner's default key. ``priority`` is an
    integer from -10 to 10, higher first, that orders the command among the
    commands of its tenant key and class; None means the command inherits the
    priority its task was created with, or 0 when the task has none.
    """

    admission: AdmissionPolicy | None = None
    dispatch_class: DispatchClass = DISPATCH_CLASS_BATCH
    fairness_key: str | None = None
    priority: int | None = None

    def __post_init__(self) -> None:
        if self.dispatch_class not in DISPATCH_CLASSES:
            raise ValueError(f"Unknown dispatch class: {self.dispatch_class!r}")
        key = self.fairness_key
        if key is not None and not (isinstance(key, str) and 1 <= len(key) <= 64):
            raise ValueError(f"Invalid fairness key: {self.fairness_key!r}")
        priority = self.priority
        if priority is not None and not (
            type(priority) is int and PRIORITY_MIN <= priority <= PRIORITY_MAX
        ):
            raise ValueError(f"Invalid priority: {self.priority!r}")


@dataclass(frozen=True, kw_only=True)
class TaskClassificationView:
    """What a classifier may know about the task a new command belongs to.

    ``background`` is true for a command the system stages on its own (the
    automatic resume sweeper) rather than one a caller asked for.
    ``previous_dispatch_class`` is the class of the task's latest earlier
    execution-opening command, or None when there is none. It is read for
    every background command and is None for any other command.
    ``requested_dispatch_class`` is the value the caller declared when it created
    the task, exactly as stored: it may not be a valid class, and it is present
    whether or not the deployment accepts declarations. ``upstream_dispatch_class``
    decides whether to adopt it and only adopts a valid class.
    ``dispatch_priority`` is the priority the caller declared at creation, as
    stored: None means none.
    """

    task_id: int
    source: str | None
    channel_name: str | None
    owner_user_id: int
    background: bool
    previous_dispatch_class: DispatchClass | None
    requested_dispatch_class: str | None
    dispatch_priority: int | None


CommandClassifier = Callable[
    [Session, TaskExecutionCommand, TaskClassificationView], CommandClassification
]
AdmissionHook = Callable[[Session, TaskExecutionCommand], AdmissionPolicy | None]

_UPSTREAM_INTERACTIVE_SOURCES = frozenset({"internal", "widget", "shared_link"})
_UPSTREAM_BATCH_SOURCES = frozenset({"sdk", "trigger", "a2a"})


def upstream_source_class(source: str | None) -> DispatchClass | None:
    """The class the upstream rule gives a task source, or None if unrecognized.

    A missing source (None or the empty string) means ``internal``, as it does
    for ``normalize_interaction_origin`` and automatic recovery. Any other match
    is exact: no stripping and no case folding.
    """
    if not source:
        return DISPATCH_CLASS_INTERACTIVE
    if source in _UPSTREAM_INTERACTIVE_SOURCES:
        return DISPATCH_CLASS_INTERACTIVE
    if source in _UPSTREAM_BATCH_SOURCES:
        return DISPATCH_CLASS_BATCH
    return None


def _accepted_declaration(view: TaskClassificationView) -> DispatchClass | None:
    """The caller's declared class if this deployment adopts it, else None.

    A declaration is adopted only when it is a valid class, the acceptance
    setting is on, and fair-share dispatch is on. The checks run in that order
    and stop at the first that fails, so a task without a declaration reads
    neither setting. A stored value that is not a valid class is treated as no
    declaration and logged.
    """
    declared = view.requested_dispatch_class
    if declared is None:
        return None
    if declared not in DISPATCH_CLASSES:
        logger.warning(
            "Task %s has an unknown requested dispatch class %r; ignoring it",
            view.task_id,
            declared,
        )
        return None
    if not get_dispatch_accept_requested_class_enabled():
        return None
    # Imported here: the transport module imports this one.
    from .task_command_transport import _fair_share_enabled

    if not _fair_share_enabled():
        return None
    return cast(DispatchClass, declared)


def upstream_dispatch_class(view: TaskClassificationView, kind: str) -> DispatchClass:
    """The upstream default class of one execution-opening command.

    An answer to a waiting task (a ``resume_input`` command that is not
    background) is always interactive and never looks at the task's
    declaration. Any other command takes the caller's
    declared class when the task has a valid one and both the acceptance setting and
    fair-share dispatch are on. Without an accepted declaration, a background
    command follows its task source; a source the rule does not recognize keeps
    the interactive class of the task's previous command, and is batch
    otherwise. Any other command follows its task source, batch if unrecognized.
    """
    if kind == "resume_input" and not view.background:
        return DISPATCH_CLASS_INTERACTIVE
    accepted = _accepted_declaration(view)
    if accepted is not None:
        return accepted
    if view.background:
        by_source = upstream_source_class(view.source)
        if by_source is not None:
            return by_source
        if view.previous_dispatch_class == DISPATCH_CLASS_INTERACTIVE:
            return DISPATCH_CLASS_INTERACTIVE
        return DISPATCH_CLASS_BATCH
    return upstream_source_class(view.source) or DISPATCH_CLASS_BATCH


def upstream_command_classifier(
    db: Session, command: TaskExecutionCommand, view: TaskClassificationView
) -> CommandClassification:
    return CommandClassification(
        dispatch_class=upstream_dispatch_class(view, str(command.kind))
    )


@dataclass(frozen=True)
class _InstalledClassifier:
    """The classifier and its admission declaration, replaced as one unit."""

    classifier: CommandClassifier
    admission_enabled: bool
    legacy: bool = False


_UPSTREAM = _InstalledClassifier(upstream_command_classifier, False)
_installed = _UPSTREAM


def _describe(classifier: CommandClassifier) -> str:
    return getattr(classifier, "__qualname__", None) or repr(classifier)


def describe_installed_classifier() -> tuple[str, bool]:
    """The installed classifier's name and whether it declared admission."""
    installed = _installed
    return _describe(installed.classifier), installed.admission_enabled


def _install(
    classifier: CommandClassifier, admission_enabled: bool, *, legacy: bool
) -> None:
    global _installed
    if not callable(classifier):
        raise TypeError("command classifier must be callable")
    try:
        # What is called is the callable itself, not a function it wraps.
        signature = inspect.signature(classifier, follow_wrapped=False)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        try:
            signature.bind(None, None, None)
        except TypeError:
            raise TypeError(
                "command classifier must accept (db, command, view)"
            ) from None
    if not isinstance(admission_enabled, bool):
        raise TypeError("admission_enabled must be a bool")
    _installed = _InstalledClassifier(classifier, admission_enabled, legacy)
    logger.info(
        "Command classifier installed: %s admission_enabled=%s",
        _describe(classifier),
        admission_enabled,
    )


def set_command_classifier(
    classifier: CommandClassifier, *, admission_enabled: bool
) -> None:
    """Install the trusted deployment classifier for new execution commands.

    Install it in every process that accepts commands and every process that
    executes them. It is called once per new execution command, after the
    command row is flushed and before the acceptance transaction commits.
    The classifier runs inside the acceptance transaction: it must not commit
    or roll back, and must not do slow I/O, because under SQLite that
    transaction holds the database-wide writer lock.
    It receives the session, the flushed command and a read-only view of its
    task. Construct every field of its return value by keyword; the
    ``dispatch_class`` field defaults to batch, so a classifier that only sets
    ``admission`` makes every command batch. To keep the upstream class it can
    return ``dispatch_class=upstream_dispatch_class(view, command.kind)``.
    It may also return a ``priority`` from -10 to 10 for the command; None keeps
    the priority the task was created with.
    The view carries the caller's declaration as stored; a classifier that
    decides for itself whether to adopt it can clear it before calling the
    upstream rule with ``dataclasses.replace(view, requested_dispatch_class=None)``.
    ``admission_enabled`` (a ``bool``) declares whether it may return an
    admission policy.
    """
    _install(classifier, admission_enabled, legacy=False)


def reset_command_classifier() -> None:
    """Restore the upstream classifier; for tests and for clearing a legacy hook."""
    global _installed
    _installed = _UPSTREAM
    logger.info("Command classifier reset to the upstream default")


def set_task_admission_hook(hook: AdmissionHook | None) -> None:
    """Legacy entry point: install an admission-only classifier.

    ``None`` clears a classifier installed through this function; a classifier
    installed with ``set_command_classifier`` is left in place. Otherwise the
    hook is called with ``(db, command)`` and its policy is staged as the
    ticket; the command's dispatch class follows the upstream rule.
    """
    if hook is None:
        if _installed.legacy:
            reset_command_classifier()
        return

    @functools.wraps(hook)
    def legacy_admission_hook(
        db: Session, command: TaskExecutionCommand, view: TaskClassificationView
    ) -> CommandClassification:
        return CommandClassification(
            dispatch_class=upstream_dispatch_class(view, str(command.kind)),
            admission=hook(db, command),
        )

    _install(legacy_admission_hook, True, legacy=True)


def admission_enabled() -> bool:
    """Whether the installed classifier declared that it stages admission tickets."""
    return _installed.admission_enabled


def classify_command_no_commit(
    db: Session, command: TaskExecutionCommand, view: TaskClassificationView
) -> CommandClassification:
    """Run the installed classifier for one new execution command, exactly once.

    The classifier receives the session, the flushed command and ``view``.
    Classifier exceptions propagate to the acceptance transaction unchanged.
    A result of the wrong type, or an admission policy from a classifier that
    declared ``admission_enabled=False``, is rejected before its admission
    ticket is staged; the caller's rollback discards the flushed command row.
    """
    installed = _installed
    result = installed.classifier(db, command, view)
    if not isinstance(result, CommandClassification):
        raise TypeError("command classifier must return CommandClassification")
    if not installed.admission_enabled and result.admission is not None:
        raise TypeError(
            "command classifier returned an admission policy but was installed "
            "with admission_enabled=False"
        )
    return result


def stage_task_admission(
    db: Session, command: TaskExecutionCommand, policy: AdmissionPolicy | None
) -> None:
    """Stage a new command's ticket in its acceptance transaction, exactly once."""
    if policy is None or command.kind not in _EXECUTION_KINDS:
        return
    if not get_shared_task_execution_enabled():
        raise RuntimeError("Execution admission requires shared task execution")
    dialect = db.get_bind().dialect.name
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    elif dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert  # type: ignore[assignment]
    else:
        raise RuntimeError("Execution admission requires PostgreSQL or SQLite")
    db.execute(
        insert(TaskAdmissionBucket)
        .values(
            key=policy.bucket, capacity=policy.capacity, max_pending=policy.max_pending
        )
        .on_conflict_do_nothing(index_elements=["key"])
    )
    bucket = _lock_bucket(db, policy.bucket)
    if (bucket.capacity, bucket.max_pending) != (policy.capacity, policy.max_pending):
        raise ValueError("Drain the admission bucket before changing its policy")
    stage_startup_pacing(db, policy.bucket, policy.pacing)
    if _pending_count(db, policy.bucket) >= policy.max_pending:
        record_queue_full(db, policy.bucket)
        raise AdmissionQueueFull("Execution queue is full")
    db.add(
        TaskAdmissionTicket(
            command_id=command.id, task_id=command.task_id, bucket_key=policy.bucket
        )
    )
    db.flush()


def _pending_count(db: Session, key: str) -> int:
    return int(
        db.scalar(
            select(func.count())
            .select_from(TaskAdmissionTicket)
            .join(
                TaskExecutionCommand,
                TaskExecutionCommand.id == TaskAdmissionTicket.command_id,
            )
            .where(
                TaskAdmissionTicket.bucket_key == key,
                ~_held_ticket(TaskAdmissionTicket),
                TaskExecutionCommand.status.notin_(_TERMINAL),
            )
        )
        or 0
    )


def prepare_task_admission_retry(db: Session, command_id: int) -> None:
    """Recheck pending capacity while preserving the original classification."""
    ticket = db.get(TaskAdmissionTicket, command_id)
    if ticket is None:
        return
    # Keep the task-before-bucket lock order used by acceptance and dispatch.
    db.execute(select(Task.id).where(Task.id == ticket.task_id).with_for_update())
    bucket = _lock_bucket(db, str(ticket.bucket_key))
    command = db.get(TaskExecutionCommand, command_id, populate_existing=True)
    if command is None or command.status != "failed":
        return
    if _pending_count(db, str(bucket.key)) >= int(bucket.max_pending):
        record_queue_full(db, str(bucket.key))
        raise AdmissionQueueFull("Execution queue is full")


def _lock_bucket(db: Session, key: str) -> TaskAdmissionBucket:
    # A write also serializes SQLite transactions; SELECT FOR UPDATE alone does not.
    db.execute(
        update(TaskAdmissionBucket)
        .where(TaskAdmissionBucket.key == key)
        .values(capacity=TaskAdmissionBucket.capacity)
    )
    return db.execute(
        select(TaskAdmissionBucket)
        .where(TaskAdmissionBucket.key == key)
        .execution_options(populate_existing=True)
    ).scalar_one()


def _live_count(bucket_key: object, *, excluding_task_id: object) -> ScalarSelect[int]:
    """Other tasks holding a slot; one task's held tickets are one execution."""
    ticket, task = aliased(TaskAdmissionTicket), aliased(Task)
    return (
        select(func.count(func.distinct(ticket.task_id)))
        .select_from(ticket)
        .join(task, task.id == ticket.task_id)
        .where(
            ticket.bucket_key == bucket_key,
            ticket.task_id != excluding_task_id,
            ticket.owner_attempt_id == task.lease_attempt_id,
            ticket.runner_id == task.runner_id,
        )
        .correlate(TaskAdmissionTicket, Task)
        .scalar_subquery()
    )


def _held_ticket(ticket: type[TaskAdmissionTicket]) -> ColumnElement[bool]:
    task = aliased(Task)
    return exists(
        select(1)
        .where(
            task.id == ticket.task_id,
            task.lease_attempt_id == ticket.owner_attempt_id,
            task.runner_id == ticket.runner_id,
        )
        .correlate(ticket)
    )


_cached_admission_eligible: tuple[object, ColumnElement[bool]] | None = None


def admission_eligible() -> ColumnElement[bool]:
    """Advisory queue scan; the transaction below rechecks after locking.

    The expression is built once per process and shared by every caller. It is
    immutable and holds no per-call state, so every time dependency inside it
    must go through ``task_admission_pacing.database_time``; the cache is keyed
    on that function object, read from the module on each call.

    The cached expression is rebuilt only when
    ``task_admission_pacing.database_time`` is replaced. Replacing the helper
    functions in this module that build the predicate (``startup_eligible``,
    ``_live_count``, ``_older_waiter``, ``_held_ticket``, ``_behind_own_task``,
    ``_joins_running_execution``) does not rebuild it; call
    ``reset_admission_eligible_cache()`` first. Replacing the clock must happen
    on a single thread.
    """
    global _cached_admission_eligible
    clock = task_admission_pacing.database_time
    cached = _cached_admission_eligible
    if cached is None or cached[0] is not clock:
        cached = (clock, _build_admission_eligible())
        _cached_admission_eligible = cached
    return cached[1]


def reset_admission_eligible_cache() -> None:
    """Drop the cached expression so the next call rebuilds it."""
    global _cached_admission_eligible
    _cached_admission_eligible = None


def _build_admission_eligible() -> ColumnElement[bool]:
    blocked = exists(
        select(1)
        .select_from(TaskAdmissionTicket)
        .join(
            TaskAdmissionBucket,
            TaskAdmissionBucket.key == TaskAdmissionTicket.bucket_key,
        )
        .where(
            TaskAdmissionTicket.command_id == TaskExecutionCommand.id,
            # A resumed claim under the same owner already holds its slot.
            or_(
                TaskAdmissionTicket.owner_attempt_id.is_(None),
                Task.lease_attempt_id.is_(None),
                TaskAdmissionTicket.owner_attempt_id != Task.lease_attempt_id,
                TaskAdmissionTicket.runner_id != Task.runner_id,
            ),
            or_(
                _live_count(TaskAdmissionTicket.bucket_key, excluding_task_id=Task.id)
                >= TaskAdmissionBucket.capacity,
                _older_waiter(),
                ~startup_eligible(),
            ),
        )
        .correlate(TaskExecutionCommand, Task)
    )
    return or_(~blocked, _joins_running_execution())


def _joins_running_execution(
    command: type[TaskExecutionCommand] = TaskExecutionCommand,
    task: type[Task] = Task,
) -> ColumnElement[bool]:
    """Guidance continues the held execution, whichever bucket classified it."""
    ticket = aliased(TaskAdmissionTicket)
    return and_(
        command.kind == "message",
        task.status == TaskStatus.RUNNING,
        task.control_state == "running",
        exists(
            select(1)
            .select_from(ticket)
            .where(
                ticket.task_id == task.id,
                ticket.runner_id == task.runner_id,
                ticket.owner_attempt_id == task.lease_attempt_id,
            )
            .correlate(task, command)
        ),
    )


def _behind_own_task(command: type[TaskExecutionCommand]) -> ColumnElement[bool]:
    """Per-task order keeps this command behind an earlier open one of its task.

    This is the transport's ``_unfinished_earlier_command`` for a ticketed
    command: its cancel/pause carve-out never applies because those kinds
    stage no ticket.
    """
    earlier = aliased(TaskExecutionCommand)
    return exists(
        select(1)
        .where(
            earlier.task_id == command.task_id,
            earlier.id < command.id,
            earlier.status.notin_(_TERMINAL),
        )
        .correlate(command)
    )


def _older_waiter() -> ColumnElement[bool]:
    """An older unheld ticket in the bucket whose command needs the next slot.

    The bucket head is the oldest waiting command that will take this
    bucket's next slot, even while a retry deadline, startup pacing, or
    routing still holds it. A command behind its own task's earlier open
    command, whichever bucket that command waits in or runs under, cannot
    use this bucket's capacity yet, and guidance that joins its task's
    running execution never needs a slot of its own; neither is the head,
    so neither delays other tasks.
    """
    ticket, command = aliased(TaskAdmissionTicket), aliased(TaskExecutionCommand)
    task = aliased(Task)
    return exists(
        select(1)
        .select_from(ticket)
        .join(command, command.id == ticket.command_id)
        .join(task, task.id == ticket.task_id)
        .where(
            ticket.bucket_key == TaskAdmissionTicket.bucket_key,
            ticket.command_id < TaskAdmissionTicket.command_id,
            ~_held_ticket(ticket),
            command.status.notin_(_TERMINAL),
            ~_behind_own_task(command),
            ~_joins_running_execution(command, task),
        )
        .correlate(TaskAdmissionTicket)
    )


def waiting_for_capacity(db: Session, command_id: int) -> bool:
    """A pending command the claim scan would skip right now for admission.

    Ingress uses this to acknowledge durable acceptance instead of holding a
    request until capacity opens; it never predicts the later execution. A
    host whose classifier did not declare admission staged no ticket, so it has
    nothing to check.
    """
    if not admission_enabled():
        return False
    return bool(
        db.scalar(
            select(~admission_eligible())
            .select_from(TaskExecutionCommand)
            .join(Task, Task.id == TaskExecutionCommand.task_id)
            .where(
                TaskExecutionCommand.id == command_id,
                TaskExecutionCommand.status == "pending",
            )
        )
    )


def reserve_task_admission(db: Session, command_id: int, lease: TaskLease) -> bool:
    """Reserve with the command claim; caller already holds the exact task owner."""
    ticket = db.get(TaskAdmissionTicket, command_id)
    if ticket is None:
        return True
    bucket = _lock_bucket(db, str(ticket.bucket_key))
    if ticket.task_id != lease.task_id:
        raise ValueError("Admission ticket belongs to another task")
    if db.scalar(
        select(_joins_running_execution())
        .select_from(TaskExecutionCommand)
        .join(Task, Task.id == TaskExecutionCommand.task_id)
        .where(TaskExecutionCommand.id == command_id)
    ):
        return True
    if (ticket.runner_id, ticket.owner_attempt_id) == (
        lease.runner_id,
        lease.attempt_id,
    ):
        return True
    if db.scalar(
        select(_older_waiter()).where(TaskAdmissionTicket.command_id == command_id)
    ):
        return False
    active = db.scalar(
        select(_live_count(ticket.bucket_key, excluding_task_id=ticket.task_id))
    )
    if active is not None and active >= bucket.capacity:
        return False
    if not reserve_startup(db, str(ticket.bucket_key)):
        return False
    setattr(ticket, "runner_id", lease.runner_id)
    setattr(ticket, "owner_attempt_id", lease.attempt_id)
    db.flush()
    return True


def _owned_tickets(lease: TaskLease) -> tuple[ColumnElement[bool], ...]:
    return (
        TaskAdmissionTicket.task_id == lease.task_id,
        TaskAdmissionTicket.runner_id == lease.runner_id,
        TaskAdmissionTicket.owner_attempt_id == lease.attempt_id,
    )


def _release_tickets(db: Session, *owned: ColumnElement[bool]) -> None:
    """Drop the tickets of completed commands; unstamp the rest for their retry."""
    db.execute(
        delete(TaskAdmissionTicket).where(
            *owned,
            TaskAdmissionTicket.command_id.in_(
                select(TaskExecutionCommand.id).where(
                    TaskExecutionCommand.status == "completed"
                )
            ),
        )
    )
    db.execute(
        update(TaskAdmissionTicket)
        .where(*owned)
        .values(runner_id=None, owner_attempt_id=None)
    )


def release_task_admissions(db: Session, lease: TaskLease) -> None:
    """Release only after the owner drained execution, never merely on status flip."""
    _release_tickets(db, *_owned_tickets(lease))


def release_command_admission(db: Session, lease: TaskLease, command_id: int) -> None:
    """Release one command's ticket, versus every ticket the lease holds.

    Compare ``release_task_admissions``, which releases every ticket of the
    lease; this releases only the one drained execution's ticket, since the
    owner may still run another. Fenced by the ticket's current owner, so a
    stale owner never frees a slot a successor acquisition stamped.
    """
    _release_tickets(
        db, TaskAdmissionTicket.command_id == command_id, *_owned_tickets(lease)
    )


def waiting_admission(command_id: object) -> ColumnElement[bool]:
    """An unclaimed command that waits for an admission ticket; controls may pass it."""
    return exists(
        select(1).where(
            TaskAdmissionTicket.command_id == command_id,
            TaskAdmissionTicket.owner_attempt_id.is_(None),
        )
    )


def settle_queued_start_for_pause(
    command: ClaimedTaskCommand,
) -> SettledTaskCommand | None:
    """Atomically stop the exact unreserved START targeted by a claimed PAUSE.

    The transaction proves the START, ticket, PAUSE claim, and live owner.
    """
    from ..models.database import get_session_local
    from ..models.user import User
    from .task_command_terminal_events import (
        TerminalTaskEventDraft,
        stage_terminal_event,
    )
    from .task_command_transport import (
        SettledTaskCommand,
        command_identity_matches_task,
        command_processing_predicates,
        finish_task_command_no_commit,
    )
    from .task_coordinator_runtime import current_task_coordinator
    from .task_start_consumer import settle_failed_start_no_commit
    from .task_start_protocol import TaskStartPayload

    if command.kind.value != "pause" or command.target_run_id is None:
        return None
    coordinator = current_task_coordinator(command.task_id)
    lease = coordinator.lease if coordinator is not None else None
    if lease is None or lease.task_id != command.task_id:
        return None

    with get_session_local()() as db, db.begin():
        ownership = command_processing_predicates(
            db,
            command.id,
            lease.runner_id,
            expected_attempt_count=command.attempt_count,
            require_live_claim=True,
            owner_lease=lease,
        )
        pause = (
            db.query(TaskExecutionCommand)
            .filter(
                *ownership,
                TaskExecutionCommand.task_id == command.task_id,
                TaskExecutionCommand.actor_user_id == command.actor_user_id,
                TaskExecutionCommand.command_id == command.command_id,
                TaskExecutionCommand.kind == "pause",
                TaskExecutionCommand.target_run_id == command.target_run_id,
            )
            .with_for_update()
            .one_or_none()
        )
        task = db.get(Task, command.task_id, populate_existing=True)
        if task is None:
            return None
        if pause is None or not command_identity_matches_task(db, task, pause):
            return None
        actor = db.get(User, pause.actor_user_id)
        if actor is None or not (actor.id == task.user_id or actor.is_admin):
            return None
        if (
            pause.target_state_version != task.state_version
            or task.run_id == command.target_run_id
        ):
            return None

        candidates = (
            db.query(TaskExecutionCommand, TaskAdmissionTicket)
            .join(
                TaskAdmissionTicket,
                TaskAdmissionTicket.command_id == TaskExecutionCommand.id,
            )
            .filter(
                TaskExecutionCommand.task_id == task.id,
                TaskExecutionCommand.id < pause.id,
                TaskExecutionCommand.kind == "start",
                TaskExecutionCommand.status == "pending",
                TaskExecutionCommand.target_run_id == command.target_run_id,
                TaskExecutionCommand.target_state_version == task.state_version,
                TaskAdmissionTicket.task_id == task.id,
                TaskAdmissionTicket.runner_id.is_(None),
                TaskAdmissionTicket.owner_attempt_id.is_(None),
            )
            .order_by(TaskExecutionCommand.id.desc())
            .limit(2)
            .with_for_update()
            .all()
        )
        if len(candidates) != 1:
            return None
        start_row, _ticket = candidates[0]
        if not command_identity_matches_task(db, task, start_row):
            return None
        try:
            start = TaskStartPayload.model_validate(start_row.payload)
        except ValueError:
            return None
        if (
            start.run_id != command.target_run_id
            or start.run_id != start_row.target_run_id
            or start.turn_id != start_row.command_id
            or start.expected_run_id != task.run_id
            or start.state_version != task.state_version
        ):
            return None

        cancelled = db.scalar(
            update(TaskExecutionCommand)
            .where(
                TaskExecutionCommand.id == start_row.id,
                TaskExecutionCommand.status == "pending",
                waiting_admission(TaskExecutionCommand.id),
            )
            .values(
                status="failed",
                error=_STOPPED_BEFORE_START,
                completed_at=func.now(),
                result={"rejection_reason": "cancelled_before_admission"},
            )
            .returning(TaskExecutionCommand.id)
        )
        if cancelled is None:
            return None
        db.refresh(start_row)
        settle_failed_start_no_commit(db, start_row)
        stage_terminal_event(
            db,
            command_db_id=int(start_row.id),
            draft=TerminalTaskEventDraft(message_code=None, resend_safe=False),
        )
        result = {
            "task_id": command.task_id,
            "command_id": command.command_id,
            "kind": command.kind.value,
        }
        if not finish_task_command_no_commit(
            db,
            command.id,
            lease.runner_id,
            result=result,
            expected_attempt_count=command.attempt_count,
            require_live_claim=True,
            owner_lease=lease,
        ):
            raise RuntimeError("PAUSE claim changed while settling queued START")
        return SettledTaskCommand(result)


def settle_cancelled_admissions(db: Session, command_id: int) -> None:
    """Settle unreserved completions and commands invalidated by a control."""
    from .task_command_terminal_events import (
        TerminalTaskEventDraft,
        stage_terminal_event,
    )
    from .task_start_consumer import settle_failed_start_no_commit

    db.execute(
        delete(TaskAdmissionTicket).where(
            TaskAdmissionTicket.command_id == command_id,
            TaskAdmissionTicket.owner_attempt_id.is_(None),
        )
    )
    control = db.get(TaskExecutionCommand, command_id)
    if control is None or control.kind not in {"cancel", "pause"}:
        return
    task = db.get(Task, control.task_id, populate_existing=True)
    if task is None:
        return
    cancelled = db.scalars(
        update(TaskExecutionCommand)
        .where(
            TaskExecutionCommand.task_id == task.id,
            TaskExecutionCommand.id < command_id,
            TaskExecutionCommand.status == "pending",
            TaskExecutionCommand.target_state_version != task.state_version,
            waiting_admission(TaskExecutionCommand.id),
        )
        .values(
            status="failed",
            error=_CONTROL_STOPPED_COMMAND,
            completed_at=func.now(),
            result={"rejection_reason": "cancelled_before_admission"},
        )
        .returning(TaskExecutionCommand.id)
    ).all()
    for cancelled_id in cancelled:
        command = db.get(TaskExecutionCommand, cancelled_id, populate_existing=True)
        assert command is not None
        scope = (
            command.payload.get("scope") if isinstance(command.payload, dict) else None
        )
        if command.kind == "start":
            settle_failed_start_no_commit(db, command)
        stage_terminal_event(
            db,
            command_db_id=cancelled_id,
            draft=TerminalTaskEventDraft(
                message_code=None,
                resend_safe=False,
                include_command_identity=scope != "external",
            ),
        )


def cap_held_command_predicates(
    task_id: int, control_db_id: int
) -> tuple[ColumnElement[bool], ...]:
    """Earlier commands of a task that a control may have passed while a
    dispatch cap held them: pending, never claimed, opening an execution and
    not waiting for an admission ticket (those stay with
    ``settle_cancelled_admissions``)."""
    from .task_dispatch_cap import EXECUTION_OPENING_KINDS

    return (
        TaskExecutionCommand.task_id == task_id,
        TaskExecutionCommand.id < control_db_id,
        TaskExecutionCommand.status == "pending",
        TaskExecutionCommand.attempt_count == 0,
        TaskExecutionCommand.kind.in_(sorted(EXECUTION_OPENING_KINDS)),
        ~waiting_admission(TaskExecutionCommand.id),
    )


def settle_cap_held_commands_no_commit(db: Session, control_command_id: int) -> None:
    """Fail the never-claimed commands a control passed while a cap held them.

    Runs in the control's terminal-write transaction: after
    ``settle_cancelled_admissions`` when the control completes, and when a
    pause is rejected because nothing of its task runs. Writes what that
    function writes: the command row, the START follow-up and one terminal
    event per row. It does not touch message delivery rows; a held message
    never ran, so it has none of its own, and its terminal event tells the
    client. Rows matched: ``cap_held_command_predicates``, and for a cancel
    only those recorded at a state version other than the task's current
    one, because a cancel that stopped nothing leaves the version as it was.
    A pause has no version condition: a pause that finds nothing running
    leaves the version as it was and still stops what it passed. A control
    passes such rows only at the cap or on the batch line; in every other case,
    and for an interactive row on the batch line that does not wait behind a
    held batch row, they exist only when a command inserted before the control
    committed after the control was claimed. Only those committed before
    this UPDATE are failed; a later one stays ``pending`` and is claimed
    after the control, where a start or a reply is rejected by its
    state-version check when the control moved the version, and a message
    is not.
    """
    if not get_shared_task_execution_enabled():
        return
    from collections import Counter

    from .task_command_terminal_events import (
        TerminalTaskEventDraft,
        stage_terminal_event,
    )
    from .task_dispatch_cap import CAP_OVERTAKING_CONTROL_KINDS
    from .task_start_consumer import settle_failed_start_no_commit

    control = db.get(TaskExecutionCommand, control_command_id)
    if control is None or control.kind not in CAP_OVERTAKING_CONTROL_KINDS:
        return
    task = db.get(Task, control.task_id, populate_existing=True)
    if task is None:
        return
    predicates = list(cap_held_command_predicates(int(task.id), control_command_id))
    if control.kind == "cancel":
        predicates.append(
            TaskExecutionCommand.target_state_version != task.state_version
        )
    settled = db.scalars(
        update(TaskExecutionCommand)
        .where(*predicates)
        .values(
            status="failed",
            error=_CONTROL_STOPPED_COMMAND,
            completed_at=func.now(),
            result={"rejection_reason": "cancelled_before_admission"},
        )
        .returning(TaskExecutionCommand.id)
    ).all()
    kinds: Counter[str] = Counter()
    for settled_id in settled:
        row = db.get(TaskExecutionCommand, settled_id, populate_existing=True)
        assert row is not None
        kinds[str(row.kind)] += 1
        scope = row.payload.get("scope") if isinstance(row.payload, dict) else None
        if row.kind == "start":
            settle_failed_start_no_commit(db, row)
        stage_terminal_event(
            db,
            command_db_id=settled_id,
            draft=TerminalTaskEventDraft(
                message_code=None,
                resend_safe=False,
                include_command_identity=scope != "external",
            ),
        )
    if settled:
        logger.info(
            "task_id=%s component=dispatch-cap control=%s control_command_id=%s "
            "settled_commands=%s kinds=%s",
            task.id,
            control.kind,
            control_command_id,
            len(settled),
            dict(kinds),
        )
