"""Detached A2A task snapshots for execution and protocol readers."""

from ..models.database import get_session_local
from ..models.task import Task
from .a2a_protocol import A2ATaskSnapshot
from .db_runtime import run_db_io_cancellation_safe


def _load_a2a_task_snapshot_sync(
    agent_id: int,
    task_id: int,
) -> A2ATaskSnapshot | None:
    """Load one detached A2A task snapshot in a worker-owned Session."""

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        fresh = (
            db.query(Task)
            .filter(
                Task.id == task_id,
                Task.agent_id == agent_id,
                Task.source == "a2a",
            )
            .first()
        )
        return A2ATaskSnapshot.from_task(fresh) if fresh is not None else None


async def load_a2a_task_snapshot(
    agent_id: int,
    task_id: int,
) -> A2ATaskSnapshot | None:
    """Load an A2A snapshot off the event loop without retaining a Session."""
    return await run_db_io_cancellation_safe(
        lambda: _load_a2a_task_snapshot_sync(agent_id, task_id)
    )
