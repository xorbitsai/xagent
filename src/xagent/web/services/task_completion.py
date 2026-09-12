"""Wait for a particular durable execution, independently of its host."""

import asyncio

from sqlalchemy import select

from ..models.database import get_session_local
from ..models.task import Task, TaskStatus
from .db_runtime import run_db_io_cancellation_safe


class TaskRunChanged(RuntimeError):
    """The requested execution is no longer the task's current run."""


def _is_run_finished(task_id: int, run_id: str) -> bool:
    with get_session_local()() as db:
        row = db.execute(
            select(Task.run_id, Task.status, Task.runner_id).where(Task.id == task_id)
        ).first()
        if row is None or row.run_id != run_id:
            raise TaskRunChanged(
                f"Task {task_id} no longer represents the requested run"
            )
        # Paused and waiting-for-user executions have exited too. Waiting here
        # follows the old background-task contract, not the whole conversation.
        return row.runner_id is None and row.status not in (
            TaskStatus.RUNNING,
            TaskStatus.PENDING,
        )


async def wait_for_task_run(task_id: int, run_id: str) -> None:
    """Wait without retaining a DB connection or mistaking a new run for this one."""
    while not await run_db_io_cancellation_safe(
        lambda: _is_run_finished(task_id, run_id)
    ):
        await asyncio.sleep(0.25)
