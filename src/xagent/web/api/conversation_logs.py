from __future__ import annotations

import logging
import math
from datetime import timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import and_, case, func, or_
from sqlalchemy.orm import Session, aliased, selectinload

from ..auth_dependencies import get_current_user
from ..models.agent import Agent
from ..models.chat_message import TaskChatMessage
from ..models.database import get_db
from ..models.task import Task, TraceEvent
from ..models.trigger import AgentTrigger, TriggerRun
from ..models.user import User
from ..utils.db_timezone import format_datetime_for_api

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/conversation-logs", tags=["conversation-logs"])

SOURCE_REST_API = "rest_api"
SOURCE_WEBHOOK = "webhook"
SOURCE_WIDGET = "widget"
SOURCE_SHARED_LINK = "shared_link"

SOURCE_LABELS = {
    SOURCE_REST_API: "REST API",
    SOURCE_WEBHOOK: "Webhook",
    SOURCE_WIDGET: "Widget",
    SOURCE_SHARED_LINK: "Shareable Link",
}
SOURCE_ORDER = [
    SOURCE_WIDGET,
    SOURCE_REST_API,
    SOURCE_SHARED_LINK,
    SOURCE_WEBHOOK,
]
DIRECT_SOURCE_TO_UI_SOURCE = {
    "sdk": SOURCE_REST_API,
    "widget": SOURCE_WIDGET,
    "shared_link": SOURCE_SHARED_LINK,
}
TRIGGER_TYPE_TO_UI_SOURCE = {
    "webhook": SOURCE_WEBHOOK,
}
EXTERNAL_TASK_SOURCES = {*DIRECT_SOURCE_TO_UI_SOURCE, "trigger"}


def _status_value(task: Task) -> str:
    status = getattr(task, "status", None)
    if status is None:
        return "unknown"
    return str(getattr(status, "value", status) or "unknown")


def _agent_config(task: Task) -> dict[str, Any]:
    config = getattr(task, "agent_config", None)
    return config if isinstance(config, dict) else {}


def _trigger_run_for_task(db: Session, task_id: int) -> TriggerRun | None:
    return (
        db.query(TriggerRun)
        .filter(TriggerRun.task_id == task_id)
        .order_by(TriggerRun.id.desc())
        .first()
    )


def _trigger_type_for_task(db: Session, task: Task) -> str | None:
    config_type = _agent_config(task).get("trigger_type")
    if config_type:
        return str(config_type)
    run = _trigger_run_for_task(db, int(task.id))
    if run and run.trigger:
        return str(run.trigger.type)
    return None


def _ui_source_from_values(source: str, trigger_type: str | None) -> str | None:
    if source in DIRECT_SOURCE_TO_UI_SOURCE:
        return DIRECT_SOURCE_TO_UI_SOURCE[source]
    if source == "trigger" and trigger_type:
        return TRIGGER_TYPE_TO_UI_SOURCE.get(trigger_type)
    return None


def _ui_source_for_task(db: Session, task: Task) -> str | None:
    source = str(getattr(task, "source", "") or "")
    return _ui_source_from_values(source, _trigger_type_for_task(db, task))


def _message_sort_key(message: TaskChatMessage) -> tuple[bool, Any, int]:
    return (
        message.created_at is not None,
        message.created_at,
        int(message.id),
    )


def _apply_external_task_scope(query: Any, user: User) -> Any:
    """Admins can inspect hidden external conversation logs across all users."""
    query = query.filter(
        Task.is_visible.is_(False),
        Task.source.in_(sorted(EXTERNAL_TASK_SOURCES)),
    )
    if not bool(user.is_admin):
        query = query.filter(Task.user_id == int(user.id))
    return query


def _apply_task_filters(
    query: Any,
    *,
    agent_id: int | None,
    search: str | None,
) -> Any:
    if agent_id is not None:
        query = query.filter(Task.agent_id == agent_id)
    if search:
        like = f"%{search}%"
        query = query.filter(
            or_(
                Task.title.ilike(like),
                Task.description.ilike(like),
                Task.input.ilike(like),
                Task.output.ilike(like),
            )
        )
    return query


def _base_task_query(db: Session, user: User) -> Any:
    query = db.query(Task).options(
        selectinload(Task.agent),
        selectinload(Task.chat_messages),
    )
    return _apply_external_task_scope(query, user)


def _conversation_source_query(
    db: Session,
    user: User,
    *,
    agent_id: int | None,
    search: str | None,
) -> tuple[Any, Any]:
    latest_run_ids = (
        db.query(
            TriggerRun.task_id.label("task_id"),
            func.max(TriggerRun.id).label("trigger_run_id"),
        )
        .filter(TriggerRun.task_id.isnot(None))
        .group_by(TriggerRun.task_id)
        .subquery()
    )
    latest_run = aliased(TriggerRun)
    latest_trigger = aliased(AgentTrigger)

    trigger_type = func.coalesce(
        Task.agent_config["trigger_type"].as_string(),
        latest_trigger.type,
    )
    ui_source = case(
        *[
            (Task.source == source, ui_source)
            for source, ui_source in DIRECT_SOURCE_TO_UI_SOURCE.items()
        ],
        *[
            (
                and_(Task.source == "trigger", trigger_type == trigger_type_value),
                ui_source,
            )
            for trigger_type_value, ui_source in TRIGGER_TYPE_TO_UI_SOURCE.items()
        ],
        else_=None,
    ).label("ui_source")

    query = (
        _apply_external_task_scope(db.query(Task), user)
        .outerjoin(latest_run_ids, latest_run_ids.c.task_id == Task.id)
        .outerjoin(latest_run, latest_run.id == latest_run_ids.c.trigger_run_id)
        .outerjoin(latest_trigger, latest_trigger.id == latest_run.trigger_id)
    )
    query = _apply_task_filters(query, agent_id=agent_id, search=search)
    return query, ui_source


def _last_activity_at(task: Task) -> Any:
    messages = list(getattr(task, "chat_messages", []) or [])
    if messages:
        latest_message = max(messages, key=_message_sort_key)
        return latest_message.created_at or task.updated_at or task.created_at
    return task.updated_at or task.created_at


def _serialize_log_summary(task: Task, ui_source: str) -> dict[str, Any]:
    agent = task.agent if isinstance(task.agent, Agent) else None
    return {
        "task_id": int(task.id),
        "title": task.title,
        "description": task.description,
        "status": _status_value(task),
        "source": ui_source,
        "source_label": SOURCE_LABELS.get(ui_source, ui_source),
        "stored_source": task.source,
        "agent_id": int(task.agent_id) if task.agent_id is not None else None,
        "agent_name": agent.name if agent else None,
        "agent_logo_url": agent.logo_url if agent else None,
        "created_at": format_datetime_for_api(task.created_at),
        "updated_at": format_datetime_for_api(task.updated_at),
        "last_activity_at": format_datetime_for_api(_last_activity_at(task)),
        "input_tokens": task.input_tokens or 0,
        "output_tokens": task.output_tokens or 0,
        "total_tokens": task.total_tokens or 0,
        "llm_calls": task.llm_calls or 0,
        "message_count": len(getattr(task, "chat_messages", []) or []),
    }


def _serialize_transcript(messages: list[TaskChatMessage]) -> list[dict[str, Any]]:
    return [
        {
            "id": int(message.id),
            "role": message.role,
            "content": message.content,
            "message_type": message.message_type,
            "interactions": message.interactions,
            "turn_id": message.turn_id,
            "attachments": message.attachments or [],
            "created_at": format_datetime_for_api(message.created_at),
        }
        for message in sorted(messages, key=_message_sort_key)
    ]


def _event_epoch(dt: Any) -> float | None:
    """Epoch seconds for a trace timestamp. Trace events are stored UTC, but
    SQLite hands them back naive; treat naive as UTC so the epoch is correct
    regardless of the server's local timezone."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return float(dt.timestamp())


def _serialize_trace_events(db: Session, task_id: int) -> list[dict[str, Any]]:
    """Historical trace events for a task, decoded and shaped for the frontend
    TraceEventRenderer (event_id / event_type / step_id / timestamp / data)."""
    from ..services.trace_message_storage import decode_trace_events_data

    events = (
        db.query(TraceEvent)
        .filter(TraceEvent.task_id == task_id, TraceEvent.build_id.is_(None))
        .order_by(TraceEvent.id.asc())
        .all()
    )
    if not events:
        return []
    # Trace rendering is a non-essential enrichment: a blob-load / decode failure
    # must not 500 the whole detail page (transcript + metadata), so fail soft.
    try:
        decoded = decode_trace_events_data(
            db, task_id=task_id, data_items=[e.data for e in events], strict=False
        )
    except Exception:
        logger.warning(
            "Failed to decode trace events for task %s", task_id, exc_info=True
        )
        return []
    return [
        {
            "event_id": e.event_id,
            "event_type": e.event_type,
            "step_id": e.step_id,
            "timestamp": _event_epoch(e.timestamp),
            "data": data,
            "parent_event_id": e.parent_event_id,
        }
        for e, data in zip(events, decoded)
    ]


def _serialize_trigger_metadata(
    db: Session,
    task: Task,
) -> dict[str, Any] | None:
    run = _trigger_run_for_task(db, int(task.id))
    trigger = run.trigger if run else None
    config = _agent_config(task)
    trigger_type = str(
        getattr(trigger, "type", None) or config.get("trigger_type") or ""
    )
    if trigger_type != "webhook":
        return None

    return {
        "trigger_id": int(trigger.id)
        if isinstance(trigger, AgentTrigger)
        else config.get("trigger_id"),
        "trigger_run_id": int(run.id) if run else config.get("trigger_run_id"),
        "trigger_type": "webhook",
        "source_event_id": run.source_event_id if run else None,
        "status": str(run.status) if run else None,
        "test": bool(config.get("trigger_test", False)),
    }


def _serialize_public_context(task: Task, ui_source: str) -> dict[str, Any] | None:
    config = _agent_config(task)
    if ui_source == SOURCE_WIDGET:
        return {
            "guest_id": config.get("guest_id"),
            "auth_mode": config.get("auth_mode") or "widget",
            "channel_name": task.channel_name,
            "widget_agent_id": config.get("widget_agent_id"),
        }
    if ui_source == SOURCE_SHARED_LINK:
        return {
            "auth_mode": config.get("auth_mode") or "share",
            "channel_name": task.channel_name,
            "share_agent_id": config.get("share_agent_id") or task.agent_id,
        }
    return None


def _source_summary_from_query(
    query: Any, ui_source: Any
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    counts = {source: 0 for source in SOURCE_ORDER}
    options: dict[int, dict[str, Any]] = {}
    rows = (
        query.outerjoin(Agent, Agent.id == Task.agent_id)
        .with_entities(
            ui_source,
            Task.agent_id,
            Agent.name,
            Agent.logo_url,
            func.count(Task.id),
        )
        .filter(ui_source.isnot(None))
        .group_by(ui_source, Task.agent_id, Agent.name, Agent.logo_url)
        .all()
    )
    for source, task_agent_id, agent_name, agent_logo_url, count in rows:
        if source in counts:
            counts[source] += int(count)
        if task_agent_id is None:
            continue
        option_agent_id = int(task_agent_id)
        if option_agent_id in options:
            continue
        options[option_agent_id] = {
            "agent_id": option_agent_id,
            "agent_name": agent_name or f"Agent {option_agent_id}",
            "agent_logo_url": agent_logo_url,
        }
    return {"all": sum(counts.values()), **counts}, sorted(
        options.values(), key=lambda item: item["agent_name"].casefold()
    )


def _latest_message_activity_subquery(db: Session) -> Any:
    return (
        db.query(
            TaskChatMessage.task_id.label("task_id"),
            func.max(TaskChatMessage.created_at).label("last_message_at"),
        )
        .group_by(TaskChatMessage.task_id)
        .subquery()
    )


def _load_tasks_by_id(db: Session, user: User, task_ids: list[int]) -> dict[int, Task]:
    if not task_ids:
        return {}
    tasks = _base_task_query(db, user).filter(Task.id.in_(task_ids)).all()
    return {int(task.id): task for task in tasks}


@router.get("")
async def list_conversation_logs(
    source: str = Query("all"),
    agent_id: int | None = Query(None),
    search: str | None = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """List hidden external conversation logs.

    Admin users can inspect hidden external conversation logs across all users.
    Non-admin users are limited to their own logs.
    """
    normalized_source = source.strip().lower()
    allowed_sources = {"all", *SOURCE_ORDER}
    if normalized_source not in allowed_sources:
        raise HTTPException(status_code=400, detail="Unsupported conversation source")

    search_value = search.strip() if search else None
    source_query, ui_source = _conversation_source_query(
        db,
        user,
        agent_id=agent_id,
        search=search_value,
    )
    source_counts, agent_options = _source_summary_from_query(source_query, ui_source)

    filtered_query = source_query.filter(ui_source.isnot(None))
    if normalized_source != "all":
        filtered_query = filtered_query.filter(ui_source == normalized_source)

    total = int(source_counts[normalized_source])
    start = (page - 1) * per_page
    latest_message_activity = _latest_message_activity_subquery(db)
    last_activity_at = func.coalesce(
        latest_message_activity.c.last_message_at,
        Task.updated_at,
        Task.created_at,
    )
    page_rows = (
        filtered_query.outerjoin(
            latest_message_activity,
            latest_message_activity.c.task_id == Task.id,
        )
        .with_entities(Task.id, ui_source)
        .order_by(last_activity_at.desc(), Task.id.desc())
        .offset(start)
        .limit(per_page)
        .all()
    )
    task_ids = [int(task_id) for task_id, _source in page_rows]
    source_by_task_id = {int(task_id): str(source) for task_id, source in page_rows}
    tasks_by_id = _load_tasks_by_id(db, user, task_ids)

    return {
        "logs": [
            _serialize_log_summary(tasks_by_id[task_id], source_by_task_id[task_id])
            for task_id in task_ids
            if task_id in tasks_by_id
        ],
        "source_counts": source_counts,
        "agents": agent_options,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": max(1, math.ceil(total / per_page)),
        },
    }


@router.get("/{task_id}")
async def get_conversation_log_detail(
    task_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Return one hidden external conversation log.

    Admin users can inspect hidden external conversation logs across all users.
    Non-admin users are limited to their own logs.
    """
    query = _base_task_query(db, user).filter(Task.id == task_id)
    task = query.first()
    if task is None:
        raise HTTPException(status_code=404, detail="Conversation log not found")

    ui_source = _ui_source_for_task(db, task)
    if ui_source is None:
        raise HTTPException(status_code=404, detail="Conversation log not found")

    messages = list(task.chat_messages or [])
    return {
        "log": _serialize_log_summary(task, ui_source),
        "transcript": _serialize_transcript(messages),
        "trace_events": _serialize_trace_events(db, int(task.id)),
        "metadata": {
            "task": {
                "task_id": int(task.id),
                "input": task.input,
                "output": task.output,
                "error_message": task.error_message,
                "description": task.description,
            },
            "trigger": _serialize_trigger_metadata(db, task),
            "public_context": _serialize_public_context(task, ui_source),
        },
        "read_only": True,
    }
