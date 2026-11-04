"""One metrics input per task: V1 traces or V2 root execution facts."""

from typing import Any

from sqlalchemy import String, cast, null, select, union_all
from sqlalchemy.orm import aliased

from ..models.task import Task, TraceEvent
from ..models.task_execution_event import TaskExecutionEvent


def monitoring_trace_source(*, user_id: int | None = None) -> Any:
    """Retain the existing metrics SQL while excluding V2 compatibility rows.

    The negative ID is only an ORM identity-map key in this read-only union;
    it is never exposed as a cursor, message identity, or stored identifier.
    """
    legacy = (
        select(*TraceEvent.__table__.c)
        .join(Task, Task.id == TraceEvent.task_id)
        .where(Task.conversation_storage_version == 1)
    )
    events = (
        select(
            (-TaskExecutionEvent.id).label("id"),
            TaskExecutionEvent.task_id,
            cast(null(), String).label("build_id"),
            TaskExecutionEvent.payload["protocol_event_id"]
            .as_string()
            .label("event_id"),
            TaskExecutionEvent.kind.label("event_type"),
            TaskExecutionEvent.occurred_at.label("timestamp"),
            TaskExecutionEvent.payload["step_id"].as_string().label("step_id"),
            TaskExecutionEvent.payload["parent_event_id"]
            .as_string()
            .label("parent_event_id"),
            TaskExecutionEvent.payload["data"].label("data"),
        )
        .join(Task, Task.id == TaskExecutionEvent.task_id)
        .where(
            Task.conversation_storage_version == 2,
            TaskExecutionEvent.scope_id == "root",
            TaskExecutionEvent.payload_version == 1,
            TaskExecutionEvent.payload["protocol_event_id"].as_string().isnot(None),
            TaskExecutionEvent.kind != "recovery_state",
        )
    )
    if user_id is not None:
        legacy = legacy.where(Task.user_id == user_id)
        events = events.where(Task.user_id == user_id)
    return aliased(
        TraceEvent, union_all(legacy, events).subquery(), adapt_on_names=True
    )
