"""Read-only reconciliation for lossy shared event streams."""

from typing import Any

from ..models.database import get_session_local
from ..models.task import Task
from .task_execution_controller import task_control_snapshot


def load_task_stream_snapshots(task_ids: list[int]) -> list[dict[str, Any]]:
    """Capture identity and persisted output from the same task row read.

    Callers supply only task IDs with an already-authorized local audience.
    There is no token replay: output is the completed turn's durable text.
    """
    if not task_ids:
        return []
    with get_session_local()() as db:
        snapshots = []
        for task in db.query(Task).filter(Task.id.in_(task_ids)).all():
            snapshot = {
                "type": "task_stream_snapshot",
                "task_id": int(task.id),
                **task_control_snapshot(task).as_dict(),
                "output": task.output if task.status.value == "completed" else None,
                "lease_attempt_id": task.lease_attempt_id,
            }
            if task.status.value == "waiting_for_user":
                from .task_interaction_read import get_pending_interaction_question

                question, interactions = get_pending_interaction_question(db, task)
                snapshot["question"] = question
                snapshot["interactions"] = interactions
            snapshots.append(snapshot)
        return snapshots
