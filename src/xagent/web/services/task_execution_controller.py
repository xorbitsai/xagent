"""Serialized task control commands and versioned execution state.

P1 keeps command serialization process-local. Cross-worker command transport is
deliberately a later concern; the database state tuple written here is the
authoritative ordering contract shared by every transport and by the frontend.
"""

from __future__ import annotations

import asyncio
import enum
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator
from uuid import uuid4

from sqlalchemy import func, update
from sqlalchemy.orm import object_session

from ..models.task import Task, TaskStatus


class TaskCommand(str, enum.Enum):
    START_TURN = "start_turn"
    MESSAGE = "message"
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"
    CHANNEL_MESSAGE = "channel_message"


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
    return {
        TaskStatus.PENDING: TaskControlState.IDLE,
        TaskStatus.RUNNING: TaskControlState.RUNNING,
        TaskStatus.PAUSED: TaskControlState.PAUSED,
        TaskStatus.WAITING_FOR_USER: TaskControlState.WAITING_FOR_USER,
        TaskStatus.COMPLETED: TaskControlState.COMPLETED,
        TaskStatus.FAILED: TaskControlState.FAILED,
    }[status]


def task_control_snapshot(task: Task) -> TaskControlSnapshot:
    raw_state = str(getattr(task, "control_state", None) or "")
    try:
        control_state = TaskControlState(raw_state)
    except ValueError:
        control_state = control_state_for_status(task.status)
    return TaskControlSnapshot(
        task_id=int(task.id),
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
) -> TaskControlSnapshot:
    """Mutate one ORM task with a monotonic control-state transition.

    The caller owns the transaction. This lets terminal task status and its
    assistant transcript row continue to commit atomically.
    """

    current_run_id = getattr(task, "run_id", None)
    if expected_run_id is not None and current_run_id != expected_run_id:
        raise StaleTaskRunError(
            f"task {task.id} run changed from {expected_run_id} to {current_run_id}"
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
        session.flush()
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


class TaskExecutionController:
    """Per-task serial command gate plus versioned state transitions."""

    def __init__(self) -> None:
        self._gates: dict[int, _ReentrantCommandGate] = {}

    @asynccontextmanager
    async def command(self, task_id: int, command: TaskCommand) -> AsyncIterator[None]:
        del command  # retained for tracing/debug call sites and future queue metrics
        normalized_task_id = int(task_id)
        gate = self._gates.setdefault(normalized_task_id, _ReentrantCommandGate())
        gate.users += 1
        acquired = False
        try:
            await gate.acquire()
            acquired = True
            yield
        finally:
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
