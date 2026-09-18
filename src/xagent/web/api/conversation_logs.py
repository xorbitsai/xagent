from __future__ import annotations

import logging
import math
from datetime import timezone
from typing import Any, Sequence

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import and_, case, func, or_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, aliased, selectinload
from sqlalchemy.sql import ColumnElement, visitors
from sqlalchemy.sql.elements import ColumnClause, TextClause
from sqlalchemy.types import Boolean

from ..auth_dependencies import get_current_user
from ..models.agent import Agent
from ..models.chat_message import TaskChatMessage
from ..models.database import get_db
from ..models.task import Task, TraceEvent
from ..models.trigger import AgentTrigger, TriggerRun
from ..models.uploaded_file import UploadedFile
from ..models.user import User
from ..services.conversation_log_sources import (
    EXTERNAL_TASK_SOURCE,
    get_external_task_public_context,
    get_external_task_source_branches,
)
from ..services.file_reference_output_service import (
    load_assistant_file_reference_records,
    reconcile_assistant_file_references,
)
from ..services.public_trace_events import (
    is_audit_only_trace_data,
    normalize_public_trace_event,
    public_task_trace_filter,
)
from ..services.task_runtime import MCP_RUNTIME_AUTHORIZATION_POLICY_REQUIRED_KEY
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
# ``external`` is stamped by the deployment layer's session transport and
# covers both REST/SDK and widget-session tasks; the deployment classifies it
# through ``services.conversation_log_sources`` and unclassified rows default
# to REST API so they are never dropped from the page.
EXTERNAL_DEFAULT_UI_SOURCE = SOURCE_REST_API
# Webhook is excluded: its detail view expects a TriggerRun, which external
# rows never have.
EXTERNAL_HOOK_UI_SOURCES = {SOURCE_WIDGET, SOURCE_REST_API, SOURCE_SHARED_LINK}
EXTERNAL_TASK_SOURCES = {
    *DIRECT_SOURCE_TO_UI_SOURCE,
    "trigger",
    EXTERNAL_TASK_SOURCE,
}


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


def _validated_external_source_branches(
    db: Session,
) -> list[tuple[ColumnElement[bool], str]]:
    """Deployment ``(predicate, ui_source)`` pairs, in the order the hook returns them.

    Two-tier fail-soft, matching the hook contract in
    ``services.conversation_log_sources``: a hook that raises is treated as
    unregistered, and an entry that is not a ``(boolean SQL predicate over
    tasks, known ui_source)`` pair is skipped on its own so the
    deployment's other branches still apply. Rows that only a skipped branch
    would have matched fall back to the REST API default instead of the page
    going down.
    """
    try:
        entries = get_external_task_source_branches(db)
    except Exception as exc:
        logger.exception("External task source hook failed; using default")
        _rollback_after_hook_failure(db, exc)
        return []
    branches: list[tuple[ColumnElement[bool], str]] = []
    for entry in entries:
        try:
            branch = _validated_source_branch(entry)
        except Exception as exc:
            _rollback_after_hook_failure(db, exc)
            # Hook-supplied objects run their own code inside the checks
            # (``__clause_element__``, SQL compilation); a failure there is a
            # malformed entry, not a page outage.
            logger.warning(
                "Ignoring external task source branch %r: validation raised",
                entry,
                exc_info=True,
            )
            continue
        if branch is not None:
            branches.append(branch)
    return branches


def _validated_source_branch(entry: Any) -> tuple[ColumnElement[bool], str] | None:
    """Return ``entry`` as a validated branch, or ``None`` after logging why not."""
    if not (isinstance(entry, (tuple, list)) and len(entry) == 2):
        logger.warning(
            "Ignoring malformed external task source branch %r: expected a "
            "(predicate, ui_source) pair",
            entry,
        )
        return None
    predicate, ui_source = entry
    # ORM attributes such as ``Task.is_visible`` are InstrumentedAttribute
    # proxies, not ColumnElements; unwrap them so bare columns are judged by
    # their SQL type like any other expression.
    clause_element = getattr(predicate, "__clause_element__", None)
    if callable(clause_element):
        predicate = clause_element()
    if not isinstance(predicate, ColumnElement):
        logger.warning(
            "Ignoring external task source branch: predicate %r is not a "
            "SQL expression",
            predicate,
        )
        return None
    raw_sql = _raw_sql_fragments(predicate)
    if raw_sql:
        # text() and literal_column() declare no FROM entries, so the check
        # below cannot see the tables they name; they fail at execute time
        # instead ("missing FROM-clause entry" on PostgreSQL).
        logger.warning(
            "Ignoring external task source branch: predicate %s embeds raw SQL "
            "%s; build predicates from Task columns and bound parameters",
            predicate,
            ", ".join(repr(fragment) for fragment in raw_sql),
        )
        return None
    if not isinstance(predicate.type, Boolean):
        # SQLite coerces a non-boolean CASE condition; PostgreSQL raises
        # "argument of CASE/WHEN must be type boolean" at execute time.
        logger.warning(
            "Ignoring external task source branch: predicate %s is not "
            "boolean-typed (wrap functions with type_=Boolean or cast(Boolean))",
            predicate,
        )
        return None
    foreign_froms = _foreign_from_names(predicate)
    if foreign_froms:
        # A bare cross-table comparison (or a Task alias) adds that FROM entry
        # to both the list query and the detail query and cartesian-joins
        # them; other tables must be reached through exists() or in_(subquery).
        logger.warning(
            "Ignoring external task source branch: predicate %s adds %s to "
            "FROM; only the tasks table may appear",
            predicate,
            ", ".join(foreign_froms),
        )
        return None
    if not isinstance(ui_source, str) or ui_source not in EXTERNAL_HOOK_UI_SOURCES:
        logger.warning(
            "Ignoring external task source branch with unsupported ui_source %r",
            ui_source,
        )
        return None
    return predicate, ui_source


def _rollback_after_hook_failure(db: Session, exc: BaseException) -> None:
    """Restore the request session after a deployment hook failed on it.

    A hook statement that errors leaves the PostgreSQL transaction aborted,
    so every later statement in the request would be refused and the
    fail-soft default could never be rendered. Both endpoints are read-only
    GETs and call their hooks before any write, so nothing durable is
    discarded (same reasoning as ``connector_team_scope``'s hook seam). A
    failure that never reached the database leaves the transaction usable,
    so only SQLAlchemy errors roll back; this keeps the identity map warm
    for the rest of the request.
    """
    if not isinstance(exc, SQLAlchemyError):
        return
    try:
        db.rollback()
    except Exception:
        logger.warning(
            "Rolling back after a failed external task hook failed", exc_info=True
        )


def _raw_sql_fragments(predicate: ColumnElement[bool]) -> list[str]:
    """Raw SQL embedded in ``predicate``: ``text()`` and table-less columns.

    ``literal_column()`` and ``column()`` both produce a ``ColumnClause`` bound
    to no table, so they contribute no FROM entries and name whatever they
    like. ``exists()`` renders its projection as ``literal_column("*")``; that
    one literal is structural and allowed.
    """
    fragments: list[str] = []
    for element in visitors.iterate(predicate):
        if isinstance(element, TextClause):
            fragments.append(element.text)
        elif (
            isinstance(element, ColumnClause)
            and element.table is None
            and element.name != "*"
        ):
            fragments.append(str(element.name))
    return fragments


def _foreign_from_names(predicate: ColumnElement[bool]) -> list[str]:
    """Names of FROM entries other than the ``tasks`` table that ``predicate`` adds.

    Correlated ``exists()`` contributes no FROM entries and ``in_(subquery)``
    contributes only ``tasks``; a bare ``Task.x == Other.y`` contributes both,
    and an ``aliased(Task)`` contributes a second ``tasks`` alias.
    """
    return [
        str(getattr(source, "description", None) or source)
        for source in select(predicate).get_final_froms()
        if source is not Task.__table__
    ]


def _external_ui_source_case(
    branches: Sequence[tuple[ColumnElement[bool], str]],
) -> Any:
    """Classification of a ``source="external"`` row; shared by list and detail.

    Returns the plain default when no branch is registered: ``case()`` needs
    at least one WHEN arm, and a bare bound parameter is what the existing
    arms already use.
    """
    if not branches:
        return EXTERNAL_DEFAULT_UI_SOURCE
    return case(*branches, else_=EXTERNAL_DEFAULT_UI_SOURCE)


def _external_ui_source_for_task(db: Session, task: Task) -> str:
    branches = _validated_external_source_branches(db)
    if not branches:
        return EXTERNAL_DEFAULT_UI_SOURCE
    ui_source = (
        db.query(_external_ui_source_case(branches))
        .select_from(Task)
        .filter(Task.id == int(task.id))
        .scalar()
    )
    return str(ui_source or EXTERNAL_DEFAULT_UI_SOURCE)


def _ui_source_for_task(db: Session, task: Task) -> str | None:
    source = str(task.source or "")
    if source == EXTERNAL_TASK_SOURCE:
        return _external_ui_source_for_task(db, task)
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
        # MCP actor/OAuth-negotiation turns are stored as hidden external
        # tasks too (``channel_runtime`` selects them by this key). They are
        # channel plumbing, not conversations, so keep them off this page as
        # they were before ``external`` entered the scope. NULL IS NOT TRUE
        # holds, so rows without the key are unaffected.
        Task.agent_config[MCP_RUNTIME_AUTHORIZATION_POLICY_REQUIRED_KEY]
        .as_boolean()
        .isnot(True),
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
        (
            Task.source == EXTERNAL_TASK_SOURCE,
            _external_ui_source_case(_validated_external_source_branches(db)),
        ),
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


def _serialize_transcript_with_events(
    db: Session,
    task: Task,
    messages: list[TaskChatMessage],
    file_reference_records: Sequence[UploadedFile] | None = None,
) -> list[dict[str, Any]]:
    """Transcript with successful context-compaction notices interleaved.

    Compaction events are not persisted chat messages, so we pull the
    ``action_end_compact`` trace events and merge them in by timestamp to
    surface the same "context compacted" notice the live chat shows.
    """

    if file_reference_records is None:
        file_reference_records = load_assistant_file_reference_records(
            db,
            task_id=int(task.id),
            user_id=int(task.user_id),
        )

    # (sort_epoch, kind, payload); on ties order user (0) -> compaction (1) ->
    # assistant (2), since compaction happens before the assistant reply.
    rows: list[tuple[float, int, dict[str, Any]]] = []
    for message in sorted(messages, key=_message_sort_key):
        content = message.content
        if message.role == "assistant":
            content = reconcile_assistant_file_references(
                db,
                task_id=int(task.id),
                user_id=int(task.user_id),
                content=content,
                records=file_reference_records,
            )
        rows.append(
            (
                _event_epoch(message.created_at) or 0.0,
                0 if message.role == "user" else 2,
                {
                    "id": int(message.id),
                    "role": message.role,
                    "content": content,
                    "message_type": message.message_type,
                    "interactions": message.interactions,
                    "turn_id": message.turn_id,
                    "attachments": message.attachments or [],
                    "created_at": format_datetime_for_api(message.created_at),
                },
            )
        )

    events = (
        db.query(TraceEvent)
        .filter(
            TraceEvent.task_id == int(task.id),
            TraceEvent.event_type == "action_end_compact",
            TraceEvent.build_id.is_(None),
        )
        .all()
    )
    for event in events:
        data: dict[str, Any] = event.data if isinstance(event.data, dict) else {}
        rows.append(
            (
                _event_epoch(event.timestamp) or 0.0,
                1,
                {
                    "id": f"compact-{int(event.id)}",
                    "role": "system",
                    "message_type": "compaction",
                    "content": "",
                    "compaction": {
                        "original_tokens": data.get("original_tokens"),
                        "compacted_tokens": data.get("compacted_tokens"),
                        "compression_ratio": data.get("compression_ratio"),
                    },
                    "created_at": format_datetime_for_api(event.timestamp),
                },
            )
        )

    rows.sort(key=lambda row: (row[0], row[1]))
    return [payload for _, _, payload in rows]


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
        .filter(
            TraceEvent.task_id == task_id,
            public_task_trace_filter(TraceEvent),
        )
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
    serialized: list[dict[str, Any]] = []
    for event, data in zip(events, decoded):
        if is_audit_only_trace_data(data):
            continue
        event_type, public_data = normalize_public_trace_event(
            str(event.event_type), data
        )
        serialized.append(
            {
                "event_id": event.event_id,
                "event_type": event_type,
                "step_id": event.step_id,
                "timestamp": _event_epoch(event.timestamp),
                "data": public_data,
                "parent_event_id": event.parent_event_id,
            }
        )
    return serialized


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


def _serialize_public_context(
    db: Session, task: Task, ui_source: str
) -> dict[str, Any] | None:
    if str(task.source or "") == EXTERNAL_TASK_SOURCE:
        # The session transport does not populate agent_config with widget or
        # share details, so external rows carry only deployment-provided context.
        try:
            return get_external_task_public_context(db, task, ui_source)
        except Exception as exc:
            logger.exception("External task context hook failed; omitting context")
            _rollback_after_hook_failure(db, exc)
            return None
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
    file_reference_records = load_assistant_file_reference_records(
        db,
        task_id=int(task.id),
        user_id=int(task.user_id),
    )
    return {
        "log": _serialize_log_summary(task, ui_source),
        "transcript": _serialize_transcript_with_events(
            db,
            task,
            messages,
            file_reference_records,
        ),
        "trace_events": _serialize_trace_events(db, int(task.id)),
        "metadata": {
            "task": {
                "task_id": int(task.id),
                "input": task.input,
                "output": reconcile_assistant_file_references(
                    db,
                    task_id=int(task.id),
                    user_id=int(task.user_id),
                    content=task.output,
                    records=file_reference_records,
                ),
                "error_message": task.error_message,
                "description": task.description,
            },
            "trigger": _serialize_trigger_metadata(db, task),
            "public_context": _serialize_public_context(db, task, ui_source),
        },
        "read_only": True,
    }
