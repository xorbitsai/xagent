"""Fence a guidance claim from turning into an unadmitted execution.

The claim counter remains monotonic because it is also a write fence. Returning
speculative guidance to the queue spends neither failure nor deferral budget.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from ..models.task import Task
from ..models.task_admission import TaskAdmissionTicket
from ..models.task_command import TaskExecutionCommand

_GUIDANCE_RETRY_MAX_EXPONENT = 6
_GUIDANCE_RETRY_MAX_SECONDS = 60


class AdmissionWaiting(RuntimeError):
    """Guidance became a new execution and must pass admission again."""


@dataclass(frozen=True)
class _AdmissionExecution:
    command_id: int
    task_id: int
    injected_run_id: str | None = None


_current: ContextVar[_AdmissionExecution | None] = ContextVar(
    "task_admission_execution", default=None
)


@contextmanager
def admission_execution(
    command_id: int, task_id: int, governed: bool
) -> Iterator[None]:
    token = _current.set(_AdmissionExecution(command_id, task_id) if governed else None)
    try:
        yield
    finally:
        _current.reset(token)


def allow_injected_guidance(task_id: int, run_id: str | None) -> None:
    """Only a confirmed injection may continue the exact original execution."""
    context = _current.get()
    if context is not None and context.task_id == task_id and run_id is not None:
        _current.set(replace(context, injected_run_id=run_id))


def require_execution_admission(
    db: Session, task_id: int, *, continuing_run_id: str | None = None
) -> None:
    """Check before handoff or inside the transaction entering RUNNING."""
    context = _current.get()
    if context is None:
        return
    if task_id != context.task_id:
        raise ValueError("Command admission belongs to another task")
    if continuing_run_id is not None and continuing_run_id == context.injected_run_id:
        return
    held = db.scalar(
        select(TaskAdmissionTicket.command_id)
        .join(Task, Task.id == TaskAdmissionTicket.task_id)
        .where(
            TaskAdmissionTicket.command_id == context.command_id,
            Task.id == task_id,
            TaskAdmissionTicket.runner_id == Task.runner_id,
            TaskAdmissionTicket.owner_attempt_id == Task.lease_attempt_id,
        )
    )
    if held is None:
        raise AdmissionWaiting("Waiting for execution admission")


def require_execution_admission_isolated(task_id: int) -> None:
    from ..models.database import get_session_local

    with get_session_local()() as db:
        require_execution_admission(db, task_id)


def return_to_admission_queue(command_id: int, runner_id: str, attempt: int) -> bool:
    from ..models.database import get_session_local
    from .task_command_transport import command_processing_predicates

    # Claims are monotonic even though this path spends no business retry budget.
    # Back off as 1, 2, 4, ... seconds and clamp before exponentiation so a
    # long-lived command cannot create an expensive or unbounded integer.
    exponent = min(max(attempt - 1, 0), _GUIDANCE_RETRY_MAX_EXPONENT)
    retry_delay = min(2**exponent, _GUIDANCE_RETRY_MAX_SECONDS)
    with get_session_local()() as db, db.begin():
        owned = command_processing_predicates(
            db, command_id, runner_id, expected_attempt_count=attempt
        )
        return (
            db.execute(
                update(TaskExecutionCommand)
                .where(*owned)
                .values(
                    status="pending",
                    claimed_by=None,
                    claim_expires_at=None,
                    retry_available_at=datetime.now(timezone.utc)
                    + timedelta(seconds=retry_delay),
                    error=None,
                )
                .returning(TaskExecutionCommand.id)
            ).scalar_one_or_none()
            is not None
        )
