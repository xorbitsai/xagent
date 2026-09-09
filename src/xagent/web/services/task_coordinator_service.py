"""Task ownership primitives for the Phase B coordinator.

The command dispatcher is not wired to these primitives yet. A coordinator
owns one TaskLease; execution writers additionally carry a fixed run in a
TaskExecutionContext. All database operations participate in the caller's
transaction, including command receipts and execution-state transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from sqlalchemy import and_, func, update
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from ..models.task import Task, TaskStatus, task_status_predicate
from .task_execution_controller import TaskControlSnapshot, TaskControlState
from .task_lease_service import (
    lease_state_version_case,
    task_lease_expires_at,
    utc_now,
)


@dataclass(frozen=True)
class TaskLease:
    """The acquisition held by a task coordinator, independent of a run."""

    task_id: int
    runner_id: str
    attempt_id: str


@dataclass(frozen=True)
class TaskExecutionContext:
    """A fixed run authorized by a lease; it has no separate lifecycle."""

    lease: TaskLease
    run_id: str


def task_lease_predicate(lease: TaskLease) -> ColumnElement[bool]:
    """Authorize only the acquisition originally held by the caller."""
    return and_(
        Task.id == lease.task_id,
        Task.runner_id == lease.runner_id,
        Task.lease_attempt_id == lease.attempt_id,
    )


def task_execution_predicate(
    execution: TaskExecutionContext,
) -> ColumnElement[bool]:
    """An earlier run cannot write even if its coordinator still owns the task."""
    return and_(
        task_lease_predicate(execution.lease),
        Task.run_id == execution.run_id,
    )


def acquire_task_lease_no_commit(
    db: Session, task_id: int, *, runner_id: str
) -> TaskLease | None:
    """Acquire an unowned task without starting or resuming execution.

    Expiry alone is not permission to restart a task: recovery must classify
    and clear an expired acquisition first. Same-runner acquisition is not
    reentrant, including when the task is paused or waiting for an answer.
    """
    lease = TaskLease(task_id, runner_id, str(uuid4()))
    now = utc_now()
    acquired = db.execute(
        update(Task)
        .where(
            Task.id == task_id,
            Task.runner_id.is_(None),
            Task.lease_attempt_id.is_(None),
            Task.lease_expires_at.is_(None),
        )
        .values(
            runner_id=lease.runner_id,
            lease_attempt_id=lease.attempt_id,
            lease_expires_at=task_lease_expires_at(now),
            last_heartbeat_at=now,
            updated_at=Task.updated_at,
        )
        .returning(Task.id)
        .execution_options(synchronize_session=False)
    ).scalar_one_or_none()
    return lease if acquired is not None else None


def renew_task_lease_no_commit(db: Session, lease: TaskLease) -> bool:
    """Keep the coordinator alive during execution, control, and settlement."""
    now = utc_now()
    renewed = db.execute(
        update(Task)
        .where(task_lease_predicate(lease))
        .values(
            lease_expires_at=task_lease_expires_at(now),
            last_heartbeat_at=now,
            updated_at=Task.updated_at,
        )
        .returning(Task.id)
        .execution_options(synchronize_session=False)
    ).scalar_one_or_none()
    return renewed is not None


def lock_task_lease_no_commit(db: Session, lease: TaskLease) -> bool:
    """Lock before command/state reads, including SQLite's write transaction."""
    locked = db.execute(
        update(Task)
        .where(task_lease_predicate(lease))
        .values(updated_at=Task.updated_at)
        .returning(Task.id)
        .execution_options(synchronize_session=False)
    ).scalar_one_or_none()
    return locked is not None


def lock_task_execution_no_commit(db: Session, execution: TaskExecutionContext) -> bool:
    """Lock the original run before staging execution results or projections."""
    locked = db.execute(
        update(Task)
        .where(task_execution_predicate(execution))
        .values(updated_at=Task.updated_at)
        .returning(Task.id)
        .execution_options(synchronize_session=False)
    ).scalar_one_or_none()
    return locked is not None


def begin_task_execution_no_commit(
    db: Session,
    lease: TaskLease,
    *,
    expected: TaskControlSnapshot,
    new_run: bool,
) -> TaskExecutionContext | None:
    """Enter an admitted run without replacing the coordinator's acquisition.

    The caller admits the command and drains any previous execution first.
    The snapshot fences concurrent control changes. A new run clears the
    checkpoint pointers; a resume retains them and the original run id.
    """
    if not new_run and expected.run_id is None:
        raise ValueError("Resuming execution requires an existing run_id")
    run_id = str(uuid4()) if new_run else expected.run_id
    control_state = TaskControlState.RUNNING.value
    values = {
        "status": task_status_predicate.value(TaskStatus.RUNNING),
        "control_state": control_state,
        "state_version": lease_state_version_case(
            TaskStatus.RUNNING, control_state, func.coalesce(Task.state_version, 0)
        ),
        "run_id": run_id,
    }
    if new_run:
        values.update(
            last_checkpoint_event_id=None,
            last_checkpoint_trace_event_id=None,
            output=None,
            error_message=None,
        )
    started_run = db.execute(
        update(Task)
        .where(
            task_lease_predicate(lease),
            Task.id == expected.task_id,
            Task.run_id == expected.run_id,
            task_status_predicate.eq(expected.status),
            task_status_predicate.ne(TaskStatus.RUNNING),
            Task.control_state == expected.control_state.value,
            Task.state_version == expected.state_version,
        )
        .values(**values)
        .returning(Task.run_id)
        .execution_options(synchronize_session=False)
    ).scalar_one_or_none()
    if started_run is None:
        return None
    return TaskExecutionContext(lease=lease, run_id=str(started_run))


def release_task_lease_no_commit(db: Session, lease: TaskLease) -> bool:
    """Release a settled coordinator without changing the task's business state.

    The coordinator must drain callbacks and check pending work in this same
    transaction before calling. RUNNING tasks must be settled first.
    """
    released = db.execute(
        update(Task)
        .where(
            task_lease_predicate(lease),
            task_status_predicate.ne(TaskStatus.RUNNING),
        )
        .values(
            runner_id=None,
            lease_attempt_id=None,
            lease_expires_at=None,
            last_heartbeat_at=utc_now(),
            updated_at=Task.updated_at,
        )
        .returning(Task.id)
        .execution_options(synchronize_session=False)
    ).scalar_one_or_none()
    return released is not None


def recover_expired_idle_task_lease_no_commit(db: Session, task_id: int) -> bool:
    """Clear an expired non-running owner without changing execution evidence.

    RUNNING recovery still requires checkpoint classification. Command effects
    are deliberately retained for the successor's receipt reconciliation.
    The conditional UPDATE races renewal under the same task write lock.
    """
    recovered = db.execute(
        update(Task)
        .where(
            Task.id == task_id,
            task_status_predicate.ne(TaskStatus.RUNNING),
            Task.lease_expires_at < utc_now(),
        )
        .values(
            runner_id=None,
            lease_attempt_id=None,
            lease_expires_at=None,
            updated_at=Task.updated_at,
        )
        .returning(Task.id)
        .execution_options(synchronize_session=False)
    ).scalar_one_or_none()
    return recovered is not None
