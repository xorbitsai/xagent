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
from typing import Any, AsyncIterator
from uuid import uuid4

from sqlalchemy import func, update
from sqlalchemy.orm import object_session

from ..models.task import Task, TaskStatus


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


def apply_task_control_transition(
    task: Task,
    control_state: TaskControlState,
    *,
    status: TaskStatus | None = None,
    new_run: bool = False,
    expected_run_id: str | None = None,
    expected_state_version: int | None = None,
) -> TaskControlSnapshot:
    """Mutate one ORM task with a monotonic control-state transition.

    The caller owns the transaction. This lets terminal task status and its
    assistant transcript row continue to commit atomically.
    """

    current_run_id = getattr(task, "run_id", None)
    current_state_version = int(getattr(task, "state_version", 0) or 0)
    if expected_run_id is not None and current_run_id != expected_run_id:
        raise StaleTaskRunError(
            f"task {task.id} run changed from {expected_run_id} to {current_run_id}"
        )
    if (
        expected_state_version is not None
        and current_state_version != expected_state_version
    ):
        raise StaleTaskRunError(
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
        # Keep unrelated caller-owned pending objects out of this helper's
        # atomic UPDATE and refresh. ``Session.execute`` and ``refresh`` can
        # otherwise trigger another session-wide autoflush.
        with session.no_autoflush:
            result = session.execute(
                statement.values(values).execution_options(synchronize_session=False)
            )
            if int(getattr(result, "rowcount", 0) or 0) != 1:
                raise StaleTaskRunError(
                    f"task {task_id} no longer belongs to run {expected_run_id}"
                )
            session.refresh(task)
        return task_control_snapshot(task)

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
    ) -> TaskControlSnapshot:
        return await asyncio.to_thread(
            transition_task_control_state_sync,
            int(task_id),
            control_state,
            status=status,
            new_run=new_run,
            expected_run_id=expected_run_id,
        )

    async def snapshot(self, task_id: int) -> TaskControlSnapshot | None:
        return await asyncio.to_thread(load_task_control_snapshot_sync, int(task_id))


task_execution_controller = TaskExecutionController()
