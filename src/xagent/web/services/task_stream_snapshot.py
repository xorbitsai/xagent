"""Read-only reconciliation for lossy shared event streams."""

import logging
from typing import Any

from sqlalchemy import select

from ..models.database import get_session_local
from ..models.task import Task
from .client_error_messages import CLIENT_SAFE_TASK_FAILURE
from .task_execution_controller import task_control_snapshot

logger = logging.getLogger(__name__)


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
                "output": task.output
                if task.conversation_storage_version == 1
                and task.status.value == "completed"
                else None,
                "completion_outcome": task.completion_outcome,
                "lease_attempt_id": task.lease_attempt_id,
            }
            if task.status.value == "waiting_for_user":
                from .task_interaction_read import get_pending_interaction_question

                question, interactions = get_pending_interaction_question(db, task)
                snapshot["question"] = question
                snapshot["interactions"] = interactions
            snapshots.append(snapshot)
            if task.conversation_storage_version == 2 and task.status.value in {
                "completed",
                "failed",
            }:
                from ..models.task_execution_event import TaskExecutionEvent
                from .task_event_display import settlement_display_events

                try:
                    settlement = db.scalar(
                        select(TaskExecutionEvent).where(
                            TaskExecutionEvent.task_id == task.id,
                            TaskExecutionEvent.scope_id == "root",
                            TaskExecutionEvent.idempotency_key
                            == f"result:{task.run_id}:{task.state_version}:{task.status.value}",
                        )
                    )
                    if settlement is None:
                        raise ValueError(
                            f"Missing display settlement (task_id={task.id}, "
                            f"run_id={task.run_id}, state_version={task.state_version})"
                        )
                    snapshots.extend(settlement_display_events(db, settlement))
                except ValueError:
                    logger.warning(
                        "Task stream display integrity gap task_id=%s",
                        task.id,
                        exc_info=True,
                    )
                    snapshots.append(
                        {
                            "type": "error",
                            "task_id": int(task.id),
                            "message": CLIENT_SAFE_TASK_FAILURE,
                        }
                    )
        return snapshots
