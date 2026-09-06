"""Version-two facts and their transitional legacy projections.

All helpers participate in the caller's transaction. Version one remains the
creation default; there is deliberately no API for switching existing tasks.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, cast
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models.chat_message import TaskChatMessage
from ..models.task import Task, TaskStatus
from ..models.task_execution_event import TaskExecutionEvent
from .task_execution_event_store import append_task_execution_event_no_commit


def uses_execution_events(db: Session, task_id: int) -> bool:
    return (
        db.scalar(select(Task.conversation_storage_version).where(Task.id == task_id))
        == 2
    )


def _fact_json_default(value: Any) -> Any:
    """Serialize the existing execution-result protocols without a lossy fallback."""
    if callable(getattr(value, "model_dump", None)):
        return value.model_dump(mode="json")
    if callable(getattr(value, "to_dict", None)):
        return value.to_dict()
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Unsupported execution fact value: {type(value).__name__}")


def append_fact_no_commit(
    db: Session,
    *,
    task_id: int,
    kind: str,
    key: str,
    payload: dict[str, Any],
    scope_id: str = "root",
    run_id: str | None = None,
    turn_id: str | None = None,
    assistant_message_id: str | None = None,
    tool_attempt_id: str | None = None,
    occurred_at: datetime | None = None,
) -> TaskExecutionEvent:
    return append_task_execution_event_no_commit(
        db,
        task_id=task_id,
        scope_id=scope_id,
        kind=kind,
        idempotency_key=key,
        payload=json.loads(
            json.dumps(payload, default=_fact_json_default, allow_nan=False)
        ),
        run_id=run_id,
        turn_id=turn_id,
        assistant_message_id=assistant_message_id,
        tool_attempt_id=tool_attempt_id,
        occurred_at=occurred_at or datetime.now(timezone.utc),
    )


def stage_chat_message_no_commit(
    db: Session, message: TaskChatMessage
) -> TaskChatMessage:
    """Commit the message fact first, then materialize its old-protocol row."""
    if not uses_execution_events(db, int(message.task_id)):
        db.add(message)
        return message
    task = cast(Any, db.get(Task, message.task_id))
    turn_id = cast(str | None, message.turn_id)
    if message.role == "user" and not turn_id:
        turn_id = str(uuid4())
    identity = turn_id if message.role == "user" else str(uuid4())
    if message.role == "assistant" and task.status in {
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.PAUSED,
        TaskStatus.WAITING_FOR_USER,
    }:
        identity = f"assistant:{task.run_id}:{task.state_version}:{message.message_type}:{turn_id}"
    payload = {
        "user_id": message.user_id,
        "role": message.role,
        "content": message.content,
        "message_type": message.message_type,
        "interactions": message.interactions,
        "attachments": message.attachments,
        "turn_id": turn_id,
        "delivery_status": message.delivery_status,
    }
    event = append_fact_no_commit(
        db,
        task_id=int(message.task_id),
        kind="input_accepted" if message.role == "user" else "assistant_message",
        key=f"message:{identity}",
        payload=payload,
        run_id=cast(str | None, task.run_id),
        turn_id=turn_id,
    )
    # Reuse the committed envelope, including an empty attachments list and the
    # actual accepted turn identity. The old row is only a compatibility reader.
    existing = db.scalar(
        select(TaskChatMessage).where(
            TaskChatMessage.execution_event_id == event.event_id,
        )
    )
    if existing is not None:
        return existing
    message.execution_event_id = event.event_id
    for field, value in event.payload.items():
        setattr(message, field, value)
    db.add(message)
    return message


def stage_delivery_fact_no_commit(
    db: Session, *, task_id: int, turn_id: str, status: str
) -> None:
    if uses_execution_events(db, task_id):
        append_fact_no_commit(
            db,
            task_id=task_id,
            kind="input_delivery_changed",
            key=f"delivery:{turn_id}:{status}",
            turn_id=turn_id,
            payload={"status": status},
        )


def stage_result_fact_no_commit(
    db: Session, task: Task, result: dict[str, Any]
) -> None:
    if task.conversation_storage_version == 2:
        append_fact_no_commit(
            db,
            task_id=int(task.id),
            kind="execution_settled",
            key=f"result:{task.run_id}:{task.status.value}",
            run_id=cast(str | None, task.run_id),
            payload={"status": task.status.value, "result": result},
        )


def stage_applied_inputs_no_commit(db: Session, state: TaskExecutionEvent) -> None:
    """Record first application alongside the durable state that proves it."""
    snapshot = state.payload["data"]["snapshot"]
    context = snapshot.get("context") or {}
    for message in context.get("messages", []):
        if message.get("role") != "user":
            continue
        turn_id = (message.get("metadata") or {}).get("turn_id")
        if not turn_id:
            continue
        key = f"input-applied:{turn_id}"
        existing = db.scalar(
            select(TaskExecutionEvent.id).where(
                TaskExecutionEvent.task_id == state.task_id,
                TaskExecutionEvent.scope_id == state.scope_id,
                TaskExecutionEvent.idempotency_key == key,
            )
        )
        if existing is None:
            append_fact_no_commit(
                db,
                task_id=int(state.task_id),
                scope_id=str(state.scope_id),
                run_id=cast(str | None, state.run_id),
                turn_id=turn_id,
                kind="input_applied",
                key=key,
                payload={"recovery_event_id": state.event_id},
            )
