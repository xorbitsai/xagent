"""Process-local task gate and versioned execution state.

The durable P2 inbox lives in :mod:`task_command_transport`; after one worker
claims a command, this controller remains the local reentrant guard around the
state transition. The database state tuple written here is the ordering
contract shared by every transport and by the frontend.
"""

from __future__ import annotations

import asyncio
import enum
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator, cast
from uuid import uuid4

from sqlalchemy import and_, false, func, or_, select, update
from sqlalchemy.orm import object_session

from ..models.task import Task, TaskStatus, task_status_predicate

if TYPE_CHECKING:
    from .task_lease_service import TaskLease


class TaskControlState(str, enum.Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSE_REQUESTED = "pause_requested"
    PAUSED = "paused"
    RESUME_REQUESTED = "resume_requested"
    WAITING_FOR_USER = "waiting_for_user"
    COMPLETED = "completed"
    FAILED = "failed"


class StaleTaskRunError(RuntimeError):
    """Raised when a late transition targets an execution that is no longer current."""


class StaleTaskStateVersionError(StaleTaskRunError):
    """The run still matches, but the row moved since the caller read it.

    Separated from its parent because the two mean opposite things to a
    caller. A rotated run is terminal: the execution being targeted no longer
    exists and no retry can make it exist. A moved version is not: the row is
    still this run's, someone simply wrote to it first, so re-reading and
    re-deciding is the correct response.

    Raised for an ambiguous rowcount miss too. There the UPDATE carried both
    fences and cannot say which one rejected it, and deferring is the safe
    half of that ambiguity: the retry reads a fresh row, and a genuinely
    rotated run is then caught precisely by the run check above.
    """


# Statuses a message delivery must never resume its original run out of:
# the run ended for good (settled FAILED by recovery, or finished). A fresh
# message still reaches such a task through APPEND as a new run.
NON_RESUMABLE_STATUSES: tuple[TaskStatus, ...] = (
    TaskStatus.FAILED,
    TaskStatus.COMPLETED,
)


class TaskStatusRefusedError(RuntimeError):
    """The row's run has ended, and the caller asked not to transition it.

    Not a stale-run error: the run may still match. A caller opts in with
    ``refuse_terminal_status`` when the target state must never be entered
    from a :data:`NON_RESUMABLE_STATUSES` row (a message handoff must not
    resume an ended run), and the refusal is part of the same conditional
    UPDATE, so a status committed after the caller's snapshot is still caught.
    """


@dataclass(frozen=True)
class TaskControlSnapshot:
    task_id: int
    run_id: str | None
    state_version: int
    control_state: TaskControlState
    status: TaskStatus

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "state_version": self.state_version,
            "control_state": self.control_state.value,
            "status": self.status.value,
        }


def control_state_for_status(status: TaskStatus) -> TaskControlState:
    control_state = {
        TaskStatus.PENDING: TaskControlState.IDLE,
        TaskStatus.RUNNING: TaskControlState.RUNNING,
        TaskStatus.PAUSED: TaskControlState.PAUSED,
        TaskStatus.WAITING_FOR_USER: TaskControlState.WAITING_FOR_USER,
        TaskStatus.COMPLETED: TaskControlState.COMPLETED,
        TaskStatus.FAILED: TaskControlState.FAILED,
    }.get(status)
    if control_state is None:
        raise ValueError(f"Unsupported task status: {status!r}")
    return control_state


def task_control_snapshot(task: Task) -> TaskControlSnapshot:
    raw_state = str(getattr(task, "control_state", None) or "")
    try:
        control_state = TaskControlState(raw_state)
    except ValueError:
        control_state = control_state_for_status(task.status)
    task_id = getattr(task, "id", None)
    if task_id is None:
        raise ValueError("Cannot create a task control snapshot for a task with no ID")
    return TaskControlSnapshot(
        task_id=int(task_id),
        run_id=getattr(task, "run_id", None),
        state_version=int(getattr(task, "state_version", 0) or 0),
        control_state=control_state,
        status=task.status,
    )


def live_foreign_owner_absent(owner_lease: "TaskLease | None") -> Any:
    """SQL condition: no acquisition but ``owner_lease`` owns the row live.

    True when the row has no owner, its lease has expired, or it is still
    held by exactly ``owner_lease`` (runner and attempt). An owner-free row is
    the normal state after a local run settled itself -- a non-shared release
    clears ``runner_id`` and the attempt but keeps ``run_id`` -- and an
    expired one is what lease acquisition may take over, so neither is a
    reason to refuse. Only a live, different acquisition is.
    """
    from .task_lease_service import utc_now

    now = utc_now()
    is_owner = (
        and_(
            Task.runner_id == owner_lease.runner_id,
            Task.lease_attempt_id == owner_lease.attempt_id,
        )
        if owner_lease is not None and owner_lease.attempt_id is not None
        else false()
    )
    return or_(
        Task.runner_id.is_(None),
        Task.lease_expires_at.is_(None),
        Task.lease_expires_at < now,
        is_owner,
    )


def apply_task_control_transition(
    task: Task,
    control_state: TaskControlState,
    *,
    status: TaskStatus | None = None,
    new_run: bool = False,
    expected_run_id: str | None = None,
    expected_state_version: int | None = None,
    fence_live_owner: bool = False,
    owner_lease: "TaskLease | None" = None,
    refuse_terminal_status: bool = False,
) -> TaskControlSnapshot:
    """Mutate one ORM task with a monotonic control-state transition.

    The caller owns the transaction. This lets terminal task status and its
    assistant transcript row continue to commit atomically.

    ``fence_live_owner`` refuses the write while an acquisition other than
    ``owner_lease`` (the caller's own, or ``None`` when it holds none) owns
    the row unexpired -- see :func:`live_foreign_owner_absent`. An expired
    RUNNING takeover keeps ``run_id``, so the run fence alone would let a
    caller stamp the write onto a successor's run. A refusal raises
    :class:`StaleTaskRunError`, as a rotated run does.

    ``refuse_terminal_status`` raises :class:`TaskStatusRefusedError` instead
    of transitioning a :data:`NON_RESUMABLE_STATUSES` row, checked in the
    UPDATE itself.
    """

    current_run_id = getattr(task, "run_id", None)
    current_state_version = int(getattr(task, "state_version", 0) or 0)
    refused = frozenset(NON_RESUMABLE_STATUSES if refuse_terminal_status else ())
    if task.status in refused:
        raise TaskStatusRefusedError(
            f"task {task.id} is {task.status.value}; refusing {control_state.value}"
        )
    if expected_run_id is not None and current_run_id != expected_run_id:
        raise StaleTaskRunError(
            f"task {task.id} run changed from {expected_run_id} to {current_run_id}"
        )
    if (
        expected_state_version is not None
        and current_state_version != expected_state_version
    ):
        raise StaleTaskStateVersionError(
            f"task {task.id} state changed from version "
            f"{expected_state_version} to {current_state_version}"
        )
    if new_run:
        current_run_id = str(uuid4())
    elif current_run_id is None and control_state not in {
        TaskControlState.IDLE,
        TaskControlState.COMPLETED,
        TaskControlState.FAILED,
    }:
        current_run_id = str(uuid4())

    session = object_session(task)
    task_id = getattr(task, "id", None)
    if session is not None and task_id is not None:
        # Preserve caller-owned pending fields (for example A2A cancellation
        # metadata) before the Core UPDATE + refresh below.
        session.flush([task])
        values: dict[Any, Any] = {
            Task.control_state: control_state.value,
            Task.state_version: func.coalesce(Task.state_version, 0) + 1,
        }
        if status is not None:
            values[Task.status] = status
        if current_run_id != getattr(task, "run_id", None):
            values[Task.run_id] = current_run_id

        statement = update(Task).where(Task.id == int(task_id))
        if expected_run_id is not None:
            statement = statement.where(Task.run_id == expected_run_id)
        if expected_state_version is not None:
            statement = statement.where(
                func.coalesce(Task.state_version, 0) == expected_state_version
            )
        if fence_live_owner:
            statement = statement.where(live_foreign_owner_absent(owner_lease))
        if refused:
            statement = statement.where(task_status_predicate.not_in(refused))
        # Keep unrelated caller-owned pending objects out of this helper's
        # atomic UPDATE and refresh. ``Session.execute`` and ``refresh`` can
        # otherwise trigger another session-wide autoflush.
        with session.no_autoflush:
            result = session.execute(
                statement.values(values).execution_options(synchronize_session=False)
            )
            if int(getattr(result, "rowcount", 0) or 0) != 1:
                if refused:
                    # Name the status fence when it is what tripped: a
                    # status committed after the pre-check above means
                    # something different to the caller than a moved run.
                    raced_status = session.scalar(
                        select(Task.status).where(Task.id == int(task_id))
                    )
                    if raced_status in refused:
                        raise TaskStatusRefusedError(
                            f"task {task_id} became {raced_status.value}; "
                            f"refusing {control_state.value}"
                        )
                # The Python pre-check above reads the row before this
                # UPDATE, so a commit landing in between arrives here
                # instead. Name both fences rather than only the run id:
                # this is the line an operator reads while diagnosing a
                # rejected transition, and blaming the run id when only the
                # version moved sends them the wrong way.
                error_type = (
                    StaleTaskStateVersionError
                    if expected_state_version is not None
                    else StaleTaskRunError
                )
                raise error_type(
                    f"task {task_id} no longer matches run {expected_run_id} "
                    f"at state version {expected_state_version}"
                    + (
                        ", or another live lease acquisition owns it"
                        if fence_live_owner
                        else ""
                    )
                )
            session.refresh(task)
        snapshot = task_control_snapshot(task)
        if task.conversation_storage_version == 2:
            from .task_execution_event_writer import append_fact_no_commit

            append_fact_no_commit(
                session,
                task_id=int(task.id),
                kind="control_state_changed",
                key=f"control:{task.state_version}",
                run_id=cast(str | None, task.run_id),
                payload=snapshot.as_dict(),
            )
        return snapshot

    # Fallback for detached/transient objects. Persistent task rows use the
    # atomic UPDATE above so concurrent lifecycle writers cannot reuse a
    # version number.
    if current_run_id != getattr(task, "run_id", None):
        setattr(task, "run_id", current_run_id)
    if status is not None:
        setattr(task, "status", status)
    setattr(task, "control_state", control_state.value)
    setattr(task, "state_version", int(getattr(task, "state_version", 0) or 0) + 1)
    return task_control_snapshot(task)


def transition_task_control_state_sync(
    task_id: int,
    control_state: TaskControlState,
    *,
    status: TaskStatus | None = None,
    new_run: bool = False,
    expected_run_id: str | None = None,
    expected_state_version: int | None = None,
    fence_live_owner: bool = False,
    owner_lease: "TaskLease | None" = None,
    refuse_terminal_status: bool = False,
) -> TaskControlSnapshot:
    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = db.query(Task).filter(Task.id == task_id).first()
        if task is None:
            raise ValueError(f"Task {task_id} not found")
        snapshot = apply_task_control_transition(
            task,
            control_state,
            status=status,
            new_run=new_run,
            expected_run_id=expected_run_id,
            expected_state_version=expected_state_version,
            fence_live_owner=fence_live_owner,
            owner_lease=owner_lease,
            refuse_terminal_status=refuse_terminal_status,
        )
        db.commit()
        return snapshot


def load_task_control_snapshot_sync(task_id: int) -> TaskControlSnapshot | None:
    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = db.query(Task).filter(Task.id == task_id).first()
        return task_control_snapshot(task) if task is not None else None


class _ReentrantCommandGate:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.owner: asyncio.Task[Any] | None = None
        self.depth = 0
        # Includes the owner and tasks waiting to acquire the gate.  A plain
        # ``lock.locked()`` check is not enough for cleanup: ``release()``
        # wakes a waiter before that waiter gets CPU time to mark the lock as
        # held again, which can otherwise let a third command create a second
        # gate for the same task.
        self.users = 0

    async def acquire(self) -> None:
        current = asyncio.current_task()
        if current is None:
            raise RuntimeError("Task execution command has no current asyncio task")
        if self.owner is current:
            self.depth += 1
            return
        await self.lock.acquire()
        self.owner = current
        self.depth = 1

    def release(self) -> None:
        current = asyncio.current_task()
        if current is None or self.owner is not current:
            raise RuntimeError("Task execution command gate released by non-owner")
        self.depth -= 1
        if self.depth == 0:
            self.owner = None
            self.lock.release()


_command_owners: ContextVar[tuple[tuple[int, int, asyncio.Task[Any]], ...]] = (
    ContextVar("task_execution_command_owners", default=())
)


class TaskExecutionController:
    """Per-task serial command gate plus versioned state transitions."""

    def __init__(self) -> None:
        self._gates: dict[int, _ReentrantCommandGate] = {}

    @asynccontextmanager
    async def command(self, task_id: int) -> AsyncIterator[None]:
        normalized_task_id = int(task_id)
        gate = self._gates.setdefault(normalized_task_id, _ReentrantCommandGate())
        current = asyncio.current_task()
        if current is None:
            raise RuntimeError("Task execution command has no current asyncio task")

        # Reentry is safe only in the same asyncio Task. Context is inherited by
        # child Tasks, so detect the otherwise-self-deadlocking pattern before
        # the child waits on a gate that its awaiting parent still owns.
        inherited_owner = next(
            (
                owner
                for controller_id, held_task_id, owner in _command_owners.get()
                if controller_id == id(self) and held_task_id == normalized_task_id
            ),
            None,
        )
        if (
            inherited_owner is not None
            and inherited_owner is not current
            and gate.owner is inherited_owner
        ):
            raise RuntimeError(
                f"Task execution command for task {normalized_task_id} cannot be "
                "reentered from a child asyncio task while its parent holds the gate"
            )

        gate.users += 1
        acquired = False
        owner_token = None
        try:
            await gate.acquire()
            acquired = True
            owner_token = _command_owners.set(
                _command_owners.get() + ((id(self), normalized_task_id, current),)
            )
            yield
        finally:
            if owner_token is not None:
                _command_owners.reset(owner_token)
            if acquired:
                gate.release()
            gate.users -= 1
            if gate.users == 0:
                self._gates.pop(normalized_task_id, None)

    async def transition(
        self,
        task_id: int,
        control_state: TaskControlState,
        *,
        status: TaskStatus | None = None,
        new_run: bool = False,
        expected_run_id: str | None = None,
        expected_state_version: int | None = None,
        fence_live_owner: bool = False,
        owner_lease: "TaskLease | None" = None,
        refuse_terminal_status: bool = False,
    ) -> TaskControlSnapshot:
        """Apply one control transition, optionally fenced on an exact row.

        ``expected_run_id`` alone cannot detect a writer that moved the row
        while preserving its run id, and several do: the a2a and v1 reply
        preleases bump ``state_version`` while
        ``acquire_task_lease_no_commit`` keeps the existing run, and an A2A
        cancel finalizes to FAILED with ``run_id`` untouched. Callers that
        read the row before deciding must therefore fence on
        ``expected_state_version`` too, or their write lands on a row that
        has since moved underneath them. The underlying sync path has always
        supported it; this wrapper simply did not pass it through.
        """

        return await asyncio.to_thread(
            transition_task_control_state_sync,
            int(task_id),
            control_state,
            status=status,
            new_run=new_run,
            expected_run_id=expected_run_id,
            expected_state_version=expected_state_version,
            fence_live_owner=fence_live_owner,
            owner_lease=owner_lease,
            refuse_terminal_status=refuse_terminal_status,
        )

    async def snapshot(self, task_id: int) -> TaskControlSnapshot | None:
        return await asyncio.to_thread(load_task_control_snapshot_sync, int(task_id))


task_execution_controller = TaskExecutionController()
