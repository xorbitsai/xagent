"""Unwired event-store primitives for the conversation migration.

The caller owns the transaction. Runtime integration and event-specific payload
contracts ship in stage 3.2; no existing task producer calls this module.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from ..models.task import Task
from ..models.task_execution_event import TaskExecutionEvent
from ..utils.json_payload_sanitizer import sanitize_json_payload

MAX_EXECUTION_EVENT_PAGE_SIZE = 100


class ExecutionEventConflict(ValueError):
    """An idempotency key was reused for a different fact."""


def append_task_execution_event_no_commit(
    db: Session,
    *,
    task_id: int,
    scope_id: str,
    idempotency_key: str,
    kind: str,
    payload: dict[str, Any],
    occurred_at: datetime,
    run_id: str | None = None,
    turn_id: str | None = None,
    assistant_message_id: str | None = None,
    tool_attempt_id: str | None = None,
    payload_version: int = 1,
) -> TaskExecutionEvent:
    """Stage one fact, serializing appends until the caller commits/rolls back.

    An UPDATE acquires a task-row lock on PostgreSQL and a writer lock on
    SQLite, including when SQLite SELECT FOR UPDATE would do nothing. The
    counter belongs to the same transaction as the event: no later append can
    commit past an uncommitted sequence. A replay keeps the first occurrence
    timestamp and event id, without allocating another sequence.
    """
    # Freeze a JSON value at the persistence boundary. Python equality would
    # otherwise equate {"value": True} with {"value": 1} on an idempotent replay.
    serialized_payload = json.dumps(
        sanitize_json_payload(payload), sort_keys=True, allow_nan=False
    )
    values = {
        "kind": kind,
        "payload_version": payload_version,
        "payload": json.loads(serialized_payload),
        "run_id": run_id,
        "turn_id": turn_id,
        "assistant_message_id": assistant_message_id,
        "tool_attempt_id": tool_attempt_id,
    }
    sequence = db.execute(
        update(Task)
        .where(Task.id == task_id)
        .values(
            conversation_event_sequence=Task.conversation_event_sequence,
            # Allocating an internal cursor must not reorder the task list.
            updated_at=Task.updated_at,
        )
        .returning(Task.conversation_event_sequence)
        .execution_options(synchronize_session=False)
    ).scalar_one_or_none()
    if sequence is None:
        raise ValueError(f"Task {task_id} does not exist")

    existing = db.scalars(
        select(TaskExecutionEvent).where(
            TaskExecutionEvent.task_id == task_id,
            TaskExecutionEvent.scope_id == scope_id,
            TaskExecutionEvent.idempotency_key == idempotency_key,
        )
    ).first()
    if existing is not None:
        if json.dumps(
            existing.payload, sort_keys=True, allow_nan=False
        ) != serialized_payload or any(
            getattr(existing, key) != value
            for key, value in values.items()
            if key != "payload"
        ):
            raise ExecutionEventConflict("Idempotency key identifies a different event")
        return existing

    db.execute(
        update(Task)
        .where(Task.id == task_id)
        .values(conversation_event_sequence=sequence + 1, updated_at=Task.updated_at)
        .execution_options(synchronize_session=False)
    )
    event = TaskExecutionEvent(
        event_id=str(uuid4()),
        task_id=task_id,
        scope_id=scope_id,
        idempotency_key=idempotency_key,
        sequence=sequence + 1,
        occurred_at=occurred_at,
        **values,
    )
    db.add(event)
    db.flush()
    return event


def load_task_execution_events(
    db: Session,
    *,
    task_id: int,
    scope_id: str,
    after_sequence: int = 0,
    limit: int = MAX_EXECUTION_EVENT_PAGE_SIZE,
) -> list[TaskExecutionEvent]:
    """Read one scope in commit order; reject page sizes outside 1–100."""
    if not 1 <= limit <= MAX_EXECUTION_EVENT_PAGE_SIZE:
        raise ValueError(f"limit must be between 1 and {MAX_EXECUTION_EVENT_PAGE_SIZE}")
    return list(
        db.scalars(
            select(TaskExecutionEvent)
            .where(
                TaskExecutionEvent.task_id == task_id,
                TaskExecutionEvent.scope_id == scope_id,
                TaskExecutionEvent.sequence > after_sequence,
            )
            .order_by(TaskExecutionEvent.sequence)
            .limit(limit)
        )
    )
