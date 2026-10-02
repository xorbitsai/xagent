"""Authorized callers share one projection of a delegated task scope."""

from datetime import timezone
from typing import Any

from fastapi import HTTPException
from sqlalchemy.orm import Session

from ..models.task import Task, TaskStatus, TraceEvent
from .public_trace_events import (
    DELEGATED_AGENT_TRACE_SOURCE,
    is_audit_only_trace_data,
    normalize_public_trace_event,
)
from .trace_event_types import GENERAL_ERROR_EVENT_TYPES
from .trace_message_storage import decode_trace_events_data


def _trace_timestamp(value: Any) -> float | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return float(value.timestamp())


_AGENT_EXECUTION_METADATA_FIELDS = (
    "agent_id",
    "agent_name",
    "worker_member_id",
    "worker_alias",
)


def _merge_agent_execution_metadata(
    metadata: dict[str, Any], data: dict[str, Any]
) -> None:
    """Keep the first immutable delegation identity observed in trace order."""

    for key in _AGENT_EXECUTION_METADATA_FIELDS:
        if key in data and key not in metadata:
            metadata[key] = data[key]


def _derive_agent_execution_status(
    trace_events: list[dict[str, Any]],
) -> str | None:
    """Infer a terminal worker status when its parent summary is missing."""

    status_aliases = {
        "completed": "completed",
        "failed": "failed",
        "interrupted": "interrupted",
        "waiting_for_user": "waiting_for_user",
    }
    for event in reversed(trace_events):
        event_type = str(event.get("event_type") or "")
        if event_type in GENERAL_ERROR_EVENT_TYPES:
            return "failed"
        if event_type not in {
            "react_task_end",
            "task_completion",
            "dag_execute_end",
        }:
            continue
        data = event.get("data")
        if not isinstance(data, dict):
            data = {}
        result = data.get("result")
        if not isinstance(result, dict):
            result = {}
        for candidate in (result.get("status"), data.get("status")):
            if isinstance(candidate, str):
                normalized = status_aliases.get(candidate.strip().lower())
                if normalized is not None:
                    return normalized
        success = result.get("success", data.get("success"))
        if success is False:
            return "failed"
        if success is True or event_type == "task_completion":
            return "completed"
    return None


def _serialize_agent_execution_traces(
    db: Session,
    *,
    task_id: int,
    worker_task_id: str,
    event_horizon: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if event_horizon is not None:
        from .task_event_display import (
            convert_display_event,
            load_display_facts,
        )

        facts = load_display_facts(
            db,
            task_id,
            scope_id=worker_task_id,
            through_sequence=event_horizon,
            source=DELEGATED_AGENT_TRACE_SOURCE,
        )
        converted = [convert_display_event(row, replay=True) for row in facts]
        canonical_events = [
            {
                key: event[key]
                for key in (
                    "event_id",
                    "event_type",
                    "step_id",
                    "timestamp",
                    "data",
                    "parent_event_id",
                )
            }
            for event in converted
            if event is not None and event["type"] == "trace_event"
        ]
        if not canonical_events:
            raise HTTPException(status_code=404, detail="Agent execution not found")
        canonical_metadata: dict[str, Any] = {"worker_task_id": worker_task_id}
        for event in canonical_events:
            _merge_agent_execution_metadata(canonical_metadata, event["data"])
        return canonical_events, canonical_metadata
    events = (
        db.query(TraceEvent)
        .filter(
            TraceEvent.task_id == task_id,
            TraceEvent.build_id == worker_task_id,
            TraceEvent.data["source"].as_string() == DELEGATED_AGENT_TRACE_SOURCE,
            TraceEvent.event_type != "agent_checkpoint",
        )
        .order_by(TraceEvent.timestamp, TraceEvent.id)
        .all()
    )
    if not events:
        raise HTTPException(status_code=404, detail="Agent execution not found")

    decoded = decode_trace_events_data(
        db,
        task_id=task_id,
        data_items=[event.data for event in events],
        strict=False,
    )
    public_events: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {"worker_task_id": worker_task_id}
    for event, data in zip(events, decoded):
        if is_audit_only_trace_data(data):
            continue
        if isinstance(data, dict):
            _merge_agent_execution_metadata(metadata, data)
        event_type, public_data = normalize_public_trace_event(
            str(event.event_type), data
        )
        public_events.append(
            {
                "event_id": event.event_id,
                "event_type": event_type,
                "step_id": event.step_id,
                "timestamp": _trace_timestamp(event.timestamp),
                "data": public_data,
                "parent_event_id": event.parent_event_id,
            }
        )
    return public_events, metadata


def load_agent_execution_detail(
    db: Session, task: Task, worker_task_id: str
) -> dict[str, Any]:
    """Read a child scope after the route has authorized its parent task."""
    task_id = int(task.id)
    event_horizon = None
    if task.conversation_storage_version == 2:
        from .task_event_display import display_horizon

        event_horizon = display_horizon(db, task_id)
    trace_events, metadata = _serialize_agent_execution_traces(
        db,
        task_id=task_id,
        worker_task_id=worker_task_id,
        event_horizon=event_horizon,
    )

    status = _derive_agent_execution_status(trace_events) or "running"
    if status == "running" and task.status in {TaskStatus.COMPLETED, TaskStatus.FAILED}:
        status = "interrupted"
    if event_horizon is not None:
        from ..models.task_execution_event import TaskExecutionEvent
        from .task_event_display import convert_display_event

        facts = (
            db.query(TaskExecutionEvent)
            .filter(
                TaskExecutionEvent.task_id == task_id,
                TaskExecutionEvent.scope_id == "root",
                TaskExecutionEvent.sequence <= event_horizon,
                TaskExecutionEvent.kind == "task_update_general",
                TaskExecutionEvent.payload["data"]["worker_task_id"].as_string()
                == worker_task_id,
            )
            .order_by(TaskExecutionEvent.sequence)
            .all()
        )
        summary_data = [
            {**event["data"], "event_type": event["event_type"]}
            for fact in facts
            if (event := convert_display_event(fact)) is not None
        ]
    else:
        summary_events = (
            db.query(TraceEvent)
            .filter(
                TraceEvent.task_id == task_id,
                TraceEvent.build_id.is_(None),
                TraceEvent.data["worker_task_id"].as_string() == worker_task_id,
            )
            .order_by(TraceEvent.id)
            .all()
        )
        summary_data = [
            event.data if isinstance(event.data, dict) else {}
            for event in summary_events
        ]
    for data in summary_data:
        summary_type = data.get("event_type")
        _merge_agent_execution_metadata(metadata, data)
        if summary_type == "workforce_delegation_end":
            status = "completed"
        elif summary_type == "workforce_delegation_error":
            status = "failed"

    return {
        **metadata,
        "task_id": task_id,
        "status": status,
        "trace_events": trace_events,
    }
