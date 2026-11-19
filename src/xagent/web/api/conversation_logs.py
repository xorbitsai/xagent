from __future__ import annotations

import csv
import io
import logging
import math
import tempfile
from contextlib import suppress
from datetime import datetime, timezone
from itertools import islice
from typing import Annotated, Any, Iterable, Iterator, Sequence
from typing import cast as typing_cast

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import (
    JSON,
    String,
    and_,
    case,
    cast,
    func,
    literal,
    or_,
    select,
    union_all,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, aliased, defer, selectinload
from sqlalchemy.sql import ColumnElement, visitors
from sqlalchemy.sql.elements import ColumnClause, TextClause
from sqlalchemy.types import Boolean
from starlette.background import BackgroundTask

from ..auth_dependencies import get_current_user
from ..models.agent import Agent
from ..models.chat_message import TaskChatMessage
from ..models.database import get_db
from ..models.expired_task import ExpiredTaskTombstone
from ..models.task import Task, TraceEvent
from ..models.task_execution_event import TaskExecutionEvent
from ..models.trigger import AgentTrigger, TriggerRun
from ..models.uploaded_file import UploadedFile
from ..models.user import User
from ..services.assistant_question_display import question_content_for_display
from ..services.assistant_question_replay import QUESTION_MESSAGE_TYPES
from ..services.conversation_log_sources import (
    EXTERNAL_TASK_SOURCE,
    get_external_task_public_context,
    get_external_task_source_branches,
)
from ..services.expired_tasks import find_expired_task
from ..services.file_reference_output_service import (
    load_assistant_file_reference_records,
    reconcile_assistant_file_references,
)
from ..services.message_metadata import (
    displayed_message_clause,
    is_display_none_metadata,
    is_display_none_user_event,
)
from ..services.public_trace_events import (
    is_audit_only_trace_data,
    normalize_public_trace_event,
    public_task_trace_filter,
)
from ..services.task_event_display import project_event_message
from ..services.task_runtime import mcp_runtime_authorization_policy_required_clause
from ..utils.db_timezone import format_datetime_for_api, normalize_datetime_from_db
from .expired_task_errors import task_expired_http_error

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

# One export is built in a temporary file before any byte is sent, so both the
# row count and the byte size are capped; an over-limit export fails instead
# of being truncated.
EXPORT_ROW_LIMIT = 10_000
EXPORT_BYTE_LIMIT = 50 * 1024 * 1024
EXPORT_SPOOL_BYTES = 8 * 1024 * 1024
EXPORT_READ_CHUNK_BYTES = 64 * 1024
EXPORT_FETCH_BATCH = 500
# Excel holds at most 32,767 UTF-16 code units in a cell; the conversation cell
# is cut below that so the truncation marker still fits.
EXPORT_CELL_CHAR_LIMIT = 32_000
EXPORT_TOO_LARGE_CODE = "conversation_log_export_too_large"
EXPORT_COLUMNS = (
    "conversation_id",
    "agent",
    "channel",
    "status",
    "title",
    "started_at",
    "last_activity_at",
    "message_count",
    "total_tokens",
    "first_message",
    "latest_answer",
    "error_message",
    "conversation",
)
EXPORT_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
EXPORT_MESSAGE_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M"
EXPORT_TIME_UNKNOWN = "[time unknown]"
EXPORT_CONVERSATION_UNAVAILABLE = "[conversation unavailable]"
# Only these event kinds can project to a message; the others are skipped in SQL.
_EXPORT_MESSAGE_EVENT_KINDS = ("input_accepted", "assistant_message", "agent_message")
# A spreadsheet reads a cell starting with one of these as a formula.
_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _status_text(status: Any) -> str:
    if status is None:
        return "unknown"
    return str(getattr(status, "value", status) or "unknown")


def _status_value(task: Task) -> str:
    return _status_text(getattr(task, "status", None))


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
    """The second live gate: ``None`` here is a 404 even for an in-scope task.

    Mirrored by ``_tombstone_has_ui_source`` below for a task retention
    purged; see ``test_live_and_tombstone_predicates_agree`` in
    ``tests/web/api/test_conversation_logs.py`` -- change the two together.
    """
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
    """Admins can inspect hidden external conversation logs across all users.

    Mirrored by ``_tombstone_in_external_task_scope`` below for a task
    retention purged; ``tests/web/api/test_conversation_logs.py``'s
    ``test_live_and_tombstone_predicates_agree`` runs both live and
    post-purge over the same task shapes and checks they agree, so a term
    changed here without changing there fails that test rather than silently
    disclosing (or hiding) an expired task. Change the two together.
    """
    query = query.filter(
        Task.is_visible.is_(False),
        Task.source.in_(sorted(EXTERNAL_TASK_SOURCES)),
        # MCP actor/OAuth-negotiation turns are stored as hidden external
        # tasks too (``channel_runtime`` selects them by this key). They are
        # channel plumbing, not conversations, so keep them off this page as
        # they were before ``external`` entered the scope. NULL IS NOT TRUE
        # holds, so rows without the key are unaffected. Spelled through the
        # shared helper so the expired-task tombstone's stored flag
        # (``services.expired_tasks``) is this same predicate, not a second
        # implementation of it.
        mcp_runtime_authorization_policy_required_clause().isnot(True),
    )
    if not bool(user.is_admin):
        query = query.filter(Task.user_id == int(user.id))
    return query


def _tombstone_in_external_task_scope(
    tombstone: ExpiredTaskTombstone, user: User
) -> bool:
    """``_apply_external_task_scope``, applied to an expired task's tombstone.

    The tombstone stores exactly the inputs that filter reads, so this is the
    same predicate term for term: hidden, an external source, not MCP channel
    plumbing, and the caller's own unless the caller is an admin. Keep the two
    in step -- a term added there and not here would let an expired task be
    disclosed to a caller who could never have seen it live.

    ``test_live_and_tombstone_predicates_agree``
    (``tests/web/api/test_conversation_logs.py``) is what would catch the two
    drifting apart; change it alongside ``_apply_external_task_scope``.
    """
    if tombstone.is_visible:
        return False
    if tombstone.source not in EXTERNAL_TASK_SOURCES:
        return False
    if tombstone.is_channel_plumbing:
        return False
    return bool(user.is_admin) or int(tombstone.user_id) == int(user.id)


def _tombstone_has_ui_source(tombstone: ExpiredTaskTombstone) -> bool:
    """The detail route's second gate, ``_ui_source_for_task``, on a tombstone.

    Mirrors ``_ui_source_for_task`` by running the tombstone's stored
    ``source`` and ``trigger_type`` through the same ``_ui_source_from_values``
    mapping the live task uses: direct sources map to a UI source
    unconditionally, and a ``trigger`` row passes only for a webhook trigger
    type, exactly like the live task's ``AgentTrigger.type``. ``external`` rows
    are handled separately here because live, ``_external_ui_source_for_task``
    always resolves to a UI source -- it falls back to the REST API default
    when no deployment branch matches -- so an external tombstone always
    passes too.

    ``test_live_and_tombstone_predicates_agree``
    (``tests/web/api/test_conversation_logs.py``) is what would catch the two
    drifting apart; change it alongside ``_ui_source_for_task``.
    """
    source = str(tombstone.source or "")
    if source == EXTERNAL_TASK_SOURCE:
        return True
    return _ui_source_from_values(source, tombstone.trigger_type) is not None


def _missing_conversation_log_error(
    db: Session, user: User, task_id: int
) -> HTTPException:
    """The error for a detail request whose task is not served live.

    ``410 task_expired`` when retention expired the task and the tombstone
    passes the same gates the live task would have (see
    ``expired_task_errors``); otherwise this route's ordinary 404, so the
    answer never reveals that an id the caller could not see ever existed.
    """
    tombstone = find_expired_task(db, task_id)
    if (
        tombstone is not None
        and _tombstone_in_external_task_scope(tombstone, user)
        and _tombstone_has_ui_source(tombstone)
    ):
        return task_expired_http_error(
            tombstone,
            message="Conversation log expired under the retention policy",
        )
    return HTTPException(status_code=404, detail="Conversation log not found")


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


def _validated_source_filter(source: str) -> str:
    """The normalized ``source`` query value; unsupported values are a 400."""
    normalized = source.strip().lower()
    if normalized not in {"all", *SOURCE_ORDER}:
        raise HTTPException(status_code=400, detail="Unsupported conversation source")
    return normalized


def _activity_window(
    active_from: datetime | None, active_to: datetime | None
) -> tuple[datetime | None, datetime | None]:
    """Validate the optional last-activity window; the interval is half-open.

    Both bounds must carry a timezone offset: a naive value would be read in
    the server's zone, which the caller cannot know. A window with
    ``active_from >= active_to`` selects nothing, so it is rejected instead of
    returning an empty page.
    """
    for name, value in (("active_from", active_from), ("active_to", active_to)):
        if value is not None and value.utcoffset() is None:
            raise HTTPException(
                status_code=422, detail=f"{name} must include a timezone offset"
            )
    if active_from is not None and active_to is not None and active_from >= active_to:
        raise HTTPException(
            status_code=422, detail="active_from must be earlier than active_to"
        )
    return active_from, active_to


def _window_bound_for_db(db: Session, value: datetime) -> datetime:
    """A window bound in the form the connected database compares correctly.

    Stored timestamps are UTC. PostgreSQL compares timezone-aware values by
    instant; SQLite stores naive UTC text, so an aware bound would be compared
    as a different string and match the wrong rows.
    """
    utc_value = value.astimezone(timezone.utc)
    if db.get_bind().dialect.name == "sqlite":
        return utc_value.replace(tzinfo=None)
    return utc_value


def _last_activity_expression(activity: Any) -> ColumnElement[Any]:
    """The last-activity instant shown on a log row.

    ``activity`` is ``_latest_message_activity_subquery``'s result. This is the
    single definition used for ordering, the time window and the export.
    """
    return func.coalesce(activity.c.last_message_at, Task.updated_at, Task.created_at)


def _last_activity_window_clause(
    db: Session,
    candidate_ids: Any,
    start: datetime | None,
    end: datetime | None,
) -> ColumnElement[bool]:
    """``Task.id`` is among ``candidate_ids`` and its last activity is in
    ``[start, end)``.

    ``candidate_ids`` is a ``select`` of the task ids that already passed the
    scope and filters; the activity subquery is summarised over exactly those.
    """
    activity = _latest_message_activity_subquery(db, candidate_ids)
    last_activity = _last_activity_expression(activity)
    window = select(Task.id).outerjoin(activity, activity.c.task_id == Task.id)
    window = window.where(Task.id.in_(candidate_ids))
    if start is not None:
        window = window.where(last_activity >= _window_bound_for_db(db, start))
    if end is not None:
        window = window.where(last_activity < _window_bound_for_db(db, end))
    return Task.id.in_(window.correlate(None))


def _base_task_query(db: Session, user: User) -> Any:
    query = db.query(Task).options(
        selectinload(Task.agent),
        selectinload(
            Task.chat_messages.and_(
                TaskChatMessage.task_id.in_(
                    select(Task.id).where(Task.conversation_storage_version == 1)
                )
            )
        ),
    )
    return _apply_external_task_scope(query, user)


def _conversation_source_query(
    db: Session,
    user: User,
    *,
    agent_id: int | None,
    search: str | None,
    active_from: datetime | None = None,
    active_to: datetime | None = None,
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
    if active_from is not None or active_to is not None:
        query = query.filter(
            _last_activity_window_clause(
                db,
                query.with_entities(Task.id).statement.correlate(None),
                active_from,
                active_to,
            )
        )
    return query, ui_source


def _visible_chat_messages(task: Task) -> list[TaskChatMessage]:
    return [
        message
        for message in getattr(task, "chat_messages", []) or []
        if not is_display_none_metadata(message.message_metadata)
    ]


def _last_activity_at(task: Task) -> Any:
    messages = _visible_chat_messages(task)
    if messages:
        latest_message = max(messages, key=_message_sort_key)
        return latest_message.created_at or task.updated_at or task.created_at
    return task.updated_at or task.created_at


def _source_label(ui_source: str) -> str:
    return SOURCE_LABELS.get(ui_source, ui_source)


def _serialize_log_summary(
    task: Task, ui_source: str, *, event_activity: tuple[int, Any] | None = None
) -> dict[str, Any]:
    agent = task.agent if isinstance(task.agent, Agent) else None
    return {
        "task_id": int(task.id),
        "title": task.title,
        "description": task.description,
        "status": _status_value(task),
        "source": ui_source,
        "source_label": _source_label(ui_source),
        "stored_source": task.source,
        "agent_id": int(task.agent_id) if task.agent_id is not None else None,
        "agent_name": agent.name if agent else None,
        "agent_logo_url": agent.logo_url if agent else None,
        "created_at": format_datetime_for_api(task.created_at),
        "updated_at": format_datetime_for_api(task.updated_at),
        "last_activity_at": format_datetime_for_api(
            (event_activity[1] or task.updated_at or task.created_at)
            if event_activity is not None
            else _last_activity_at(task)
        ),
        "input_tokens": task.input_tokens or 0,
        "output_tokens": task.output_tokens or 0,
        "total_tokens": task.total_tokens or 0,
        "llm_calls": task.llm_calls or 0,
        "message_count": event_activity[0]
        if event_activity is not None
        else len(_visible_chat_messages(task)),
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
        content = str(message.content or "")
        if message.role == "assistant":
            if message.message_type in QUESTION_MESSAGE_TYPES:
                content = question_content_for_display(content, message.interactions)
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
        if is_audit_only_trace_data(data) or is_display_none_user_event(
            str(event.event_type), data
        ):
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


def _event_log_message_predicate(db: Session) -> Any:
    data = TaskExecutionEvent.payload["data"]
    expects_response: ColumnElement[bool]
    # Match the transcript reader's Python identity/truthiness checks without
    # casting arbitrary persisted JSON strings to PostgreSQL booleans.
    if db.get_bind().dialect.name == "postgresql":
        audit_only = cast(data["__audit_only__"], JSONB) == literal(True, type_=JSONB)
        hidden = cast(data["visible"], JSONB) == literal(False, type_=JSONB)
        expects_response = cast(data["expect_response"], JSONB).not_in(
            [JSON.NULL, False, 0, "", [], {}]
        )
    else:
        audit_only = (
            func.json_type(TaskExecutionEvent.payload, "$.data.__audit_only__")
            == "true"
        )
        hidden = func.json_type(TaskExecutionEvent.payload, "$.data.visible") == "false"
        response_type = func.json_type(
            TaskExecutionEvent.payload, "$.data.expect_response"
        )
        expects_response = case(
            (
                response_type.in_(["integer", "real"]),
                data["expect_response"].as_float() != 0,
            ),
            else_=cast(data["expect_response"], String).not_in(
                ["null", "0", '""', "[]", "{}"]
            ),
        )
    return and_(
        audit_only.is_not(True),
        # Hidden user messages are plumbing: the detail view omits them, so the
        # list's count and last activity must not include them either.
        displayed_message_clause(TaskExecutionEvent.payload["message_metadata"]),
        or_(
            TaskExecutionEvent.kind.in_(["input_accepted", "assistant_message"]),
            and_(
                TaskExecutionEvent.kind == "agent_message",
                expects_response,
                hidden.is_not(True),
                data["message"].as_string().isnot(None),
                data["message"].as_string() != "",
            ),
        ),
    )


def _latest_message_activity_subquery(db: Session, task_ids: Any) -> Any:
    legacy = (
        select(
            TaskChatMessage.task_id.label("task_id"),
            TaskChatMessage.created_at.label("created_at"),
        )
        .join(Task, Task.id == TaskChatMessage.task_id)
        .where(
            Task.conversation_storage_version == 1,
            Task.id.in_(task_ids),
            displayed_message_clause(TaskChatMessage.message_metadata),
        )
    )
    canonical = (
        select(
            TaskExecutionEvent.task_id.label("task_id"),
            TaskExecutionEvent.occurred_at.label("created_at"),
        )
        .join(Task, Task.id == TaskExecutionEvent.task_id)
        .where(
            Task.conversation_storage_version == 2,
            TaskExecutionEvent.scope_id == "root",
            Task.id.in_(task_ids),
            _event_log_message_predicate(db),
        )
    )
    messages = union_all(legacy, canonical).subquery()
    return (
        select(
            messages.c.task_id,
            func.max(messages.c.created_at).label("last_message_at"),
            func.count().label("message_count"),
        )
        .group_by(messages.c.task_id)
        .subquery()
    )


def _load_tasks_by_id(db: Session, user: User, task_ids: list[int]) -> dict[int, Task]:
    if not task_ids:
        return {}
    # Loaded without structured_output_record: reading the column from these objects, or
    # from the same rows reached again through Session.get or a relationship in this
    # session, raises; do not keep using these Task objects after a commit, because the
    # reload after a commit reads the column back.
    tasks = (
        _base_task_query(db, user)
        .options(defer(typing_cast(Any, Task.structured_output_record), raiseload=True))
        .filter(Task.id.in_(task_ids))
        .all()
    )
    return {int(task.id): task for task in tasks}


@router.get("")
async def list_conversation_logs(
    source: str = Query("all"),
    agent_id: int | None = Query(None),
    search: str | None = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    active_from: Annotated[datetime | None, Query()] = None,
    active_to: Annotated[datetime | None, Query()] = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """List hidden external conversation logs.

    Admin users can inspect hidden external conversation logs across all users.
    Non-admin users are limited to their own logs. ``active_from`` /
    ``active_to`` restrict the rows, the per-source counts and the agent
    options to conversations whose last activity is in ``[active_from,
    active_to)``.
    """
    normalized_source = _validated_source_filter(source)
    window_start, window_end = _activity_window(active_from, active_to)
    search_value = search.strip() if search else None
    source_query, ui_source = _conversation_source_query(
        db,
        user,
        agent_id=agent_id,
        search=search_value,
        active_from=window_start,
        active_to=window_end,
    )
    source_counts, agent_options = _source_summary_from_query(source_query, ui_source)

    filtered_query = source_query.filter(ui_source.isnot(None))
    if normalized_source != "all":
        filtered_query = filtered_query.filter(ui_source == normalized_source)

    total = int(source_counts[normalized_source])
    start = (page - 1) * per_page
    latest_message_activity = _latest_message_activity_subquery(
        db, filtered_query.with_entities(Task.id).statement.correlate(None)
    )
    last_activity_at = _last_activity_expression(latest_message_activity)
    page_rows = (
        filtered_query.outerjoin(
            latest_message_activity,
            latest_message_activity.c.task_id == Task.id,
        )
        .with_entities(
            Task.id,
            ui_source,
            latest_message_activity.c.message_count,
            latest_message_activity.c.last_message_at,
        )
        .order_by(last_activity_at.desc(), Task.id.desc())
        .offset(start)
        .limit(per_page)
        .all()
    )
    task_ids = [int(row[0]) for row in page_rows]
    source_by_task_id = {int(row[0]): str(row[1]) for row in page_rows}
    activity_by_task_id = {int(row[0]): (int(row[2] or 0), row[3]) for row in page_rows}
    tasks_by_id = _load_tasks_by_id(db, user, task_ids)

    return {
        "logs": [
            _serialize_log_summary(
                tasks_by_id[task_id],
                source_by_task_id[task_id],
                event_activity=activity_by_task_id[task_id]
                if tasks_by_id[task_id].conversation_storage_version == 2
                else None,
            )
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


def _csv_text(value: Any) -> str:
    """A text cell safe to open in a spreadsheet.

    A value starting with a formula prefix gets a leading ``'`` so it is read
    as text; ``None`` is an empty cell.
    """
    if value is None:
        return ""
    text = str(value)
    if text.startswith(_CSV_FORMULA_PREFIXES):
        return "'" + text
    return text


def _csv_line(cells: Iterable[Any]) -> str:
    """One CSV record, with every text cell passed through ``_csv_text``."""
    buffer = io.StringIO()
    csv.writer(buffer, quoting=csv.QUOTE_ALL).writerow(
        [
            _csv_text(cell) if cell is None or isinstance(cell, str) else cell
            for cell in cells
        ]
    )
    return buffer.getvalue()


def _export_timestamp(
    value: datetime | None, fmt: str = EXPORT_TIMESTAMP_FORMAT
) -> str | None:
    """A stored time as UTC text in ``fmt``, which a spreadsheet reads as a date.

    ``None`` stays ``None``.
    """
    if value is None:
        return None
    return normalize_datetime_from_db(value).strftime(fmt)


def _utf16_units(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def _truncate_export_cell(text: str) -> str:
    """``text`` cut to ``EXPORT_CELL_CHAR_LIMIT`` UTF-16 units, with a marker.

    Excel counts a cell in UTF-16 code units, so a character outside the BMP
    counts twice and is never split. The marker states how many units were
    dropped.
    """
    total = _utf16_units(text)
    if total <= EXPORT_CELL_CHAR_LIMIT:
        return text
    kept_units = 0
    kept_chars = 0
    for char in text:
        units = _utf16_units(char)
        if kept_units + units > EXPORT_CELL_CHAR_LIMIT:
            break
        kept_units += units
        kept_chars += 1
    return f"{text[:kept_chars]}\n\n[truncated: {total - kept_units:,} more characters]"


def _export_role_label(role: str | None) -> str:
    return (role or "").capitalize() or "Unknown"


def _export_message_block(role: str | None, content: str | None, at: Any) -> str:
    stamp = _export_timestamp(at, EXPORT_MESSAGE_TIMESTAMP_FORMAT)
    when = f"[{stamp} UTC]" if stamp is not None else EXPORT_TIME_UNKNOWN
    return f"{when} {_export_role_label(role)}:\n{content or ''}"


def _load_export_transcripts(db: Session, task_ids: Sequence[int]) -> dict[int, str]:
    """The conversation cell for each of ``task_ids`` that has displayed messages.

    A message shows when the detail page shows it, in the detail page's order,
    with its stored text. A task whose v2 events cannot be projected gets
    ``EXPORT_CONVERSATION_UNAVAILABLE``. A task without messages has no key.
    """
    ids = list(task_ids)
    if not ids:
        return {}
    blocks: dict[int, list[str]] = {}

    v1_rows = db.execute(
        select(
            TaskChatMessage.task_id,
            TaskChatMessage.role,
            TaskChatMessage.content,
            TaskChatMessage.created_at,
        )
        .join(Task, Task.id == TaskChatMessage.task_id)
        .where(
            Task.id.in_(ids),
            Task.conversation_storage_version == 1,
            displayed_message_clause(TaskChatMessage.message_metadata),
        )
        .order_by(
            TaskChatMessage.task_id,
            TaskChatMessage.created_at.isnot(None),
            TaskChatMessage.created_at,
            TaskChatMessage.role != "user",
            TaskChatMessage.id,
        )
    )
    for task_id, role, content, created_at in v1_rows:
        blocks.setdefault(int(task_id), []).append(
            _export_message_block(role, content, created_at)
        )

    unavailable: set[int] = set()
    v2_events = db.scalars(
        select(TaskExecutionEvent)
        .join(Task, Task.id == TaskExecutionEvent.task_id)
        .where(
            Task.id.in_(ids),
            Task.conversation_storage_version == 2,
            TaskExecutionEvent.scope_id == "root",
            TaskExecutionEvent.sequence <= Task.conversation_event_sequence,
            TaskExecutionEvent.kind.in_(_EXPORT_MESSAGE_EVENT_KINDS),
        )
        .order_by(TaskExecutionEvent.task_id, TaskExecutionEvent.sequence)
    )
    for event in v2_events:
        task_id = int(event.task_id)
        if task_id in unavailable:
            continue
        try:
            message = project_event_message(event)
        except ValueError:
            logger.warning(
                "Conversation export could not project the events of task %s",
                task_id,
            )
            unavailable.add(task_id)
            continue
        if message is None or is_display_none_metadata(message["message_metadata"]):
            continue
        blocks.setdefault(task_id, []).append(
            _export_message_block(
                message["role"], message["content"], message["created_at"]
            )
        )

    transcripts = {
        task_id: _truncate_export_cell("\n\n".join(parts))
        for task_id, parts in blocks.items()
    }
    for task_id in unavailable:
        transcripts[task_id] = EXPORT_CONVERSATION_UNAVAILABLE
    return transcripts


def _export_rows(db: Session, result_rows: Iterator[Any]) -> Iterator[tuple[Any, ...]]:
    """Export cells for every main-query row; conversations load one batch at a time."""
    while batch := list(islice(result_rows, EXPORT_FETCH_BATCH)):
        transcripts = _load_export_transcripts(db, [int(row[0]) for row in batch])
        for (
            task_id,
            agent_name,
            row_ui_source,
            status,
            title,
            created_at,
            last_activity_at,
            message_count,
            total_tokens,
            description,
            output,
            error_message,
        ) in batch:
            yield (
                int(task_id),
                agent_name,
                _source_label(row_ui_source),
                _status_text(status),
                title,
                _export_timestamp(created_at),
                _export_timestamp(last_activity_at),
                int(message_count or 0),
                int(total_tokens or 0),
                description,
                output,
                error_message,
                transcripts.get(int(task_id), ""),
            )


def _export_too_large(reason: str, limit: int) -> HTTPException:
    return HTTPException(
        status_code=422,
        detail={"code": EXPORT_TOO_LARGE_CODE, "reason": reason, "limit": limit},
    )


def _write_export_file(
    rows: Iterable[Sequence[Any]],
) -> tempfile.SpooledTemporaryFile[bytes]:
    """The CSV (UTF-8 with BOM, header first) in a rewound temporary file.

    Raises a 422 ``_export_too_large`` once more than ``EXPORT_ROW_LIMIT`` rows
    or ``EXPORT_BYTE_LIMIT`` bytes would be written. On any failure the file is
    closed before the error propagates.
    """
    raw = tempfile.SpooledTemporaryFile(max_size=EXPORT_SPOOL_BYTES)
    wrapper = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
    try:
        written = 0
        wrapper.write(header := _csv_line(EXPORT_COLUMNS))
        written += len(header.encode("utf-8"))
        for row_count, cells in enumerate(rows, start=1):
            if row_count > EXPORT_ROW_LIMIT:
                raise _export_too_large("rows", EXPORT_ROW_LIMIT)
            line = _csv_line(cells)
            written += len(line.encode("utf-8"))
            if written > EXPORT_BYTE_LIMIT:
                raise _export_too_large("bytes", EXPORT_BYTE_LIMIT)
            wrapper.write(line)
        wrapper.flush()
        # Detach so collecting the wrapper does not close the file.
        wrapper.detach()
        raw.seek(0)
    except BaseException:
        with suppress(ValueError):
            wrapper.close()
        raw.close()
        raise
    return raw


def _iter_export_file(
    raw: tempfile.SpooledTemporaryFile[bytes],
) -> Iterator[bytes]:
    while chunk := raw.read(EXPORT_READ_CHUNK_BYTES):
        yield chunk


@router.get("/export")
def export_conversation_logs(
    source: str = Query("all"),
    agent_id: int | None = Query(None),
    search: str | None = Query(None),
    active_from: Annotated[datetime | None, Query()] = None,
    active_to: Annotated[datetime | None, Query()] = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Export the filtered conversation logs as a CSV, one row per conversation.

    Takes the list's filters and returns exactly the rows the list shows when
    paged to the end, in the same order. More than ``EXPORT_ROW_LIMIT`` rows or
    ``EXPORT_BYTE_LIMIT`` bytes is a 422 with no file. This is a plain ``def``
    so FastAPI runs the blocking reads and writes in a worker thread. The
    database session is released before the file is streamed.
    """
    normalized_source = _validated_source_filter(source)
    window_start, window_end = _activity_window(active_from, active_to)
    search_value = search.strip() if search else None
    source_query, ui_source = _conversation_source_query(
        db,
        user,
        agent_id=agent_id,
        search=search_value,
        active_from=window_start,
        active_to=window_end,
    )
    filtered_query = source_query.filter(ui_source.isnot(None))
    if normalized_source != "all":
        filtered_query = filtered_query.filter(ui_source == normalized_source)

    activity = _latest_message_activity_subquery(
        db, filtered_query.with_entities(Task.id).statement.correlate(None)
    )
    last_activity = _last_activity_expression(activity)
    # One row more than the limit is read so exceeding it is observable.
    rows = (
        filtered_query.outerjoin(activity, activity.c.task_id == Task.id)
        .outerjoin(Agent, Agent.id == Task.agent_id)
        .with_entities(
            Task.id,
            Agent.name,
            ui_source,
            Task.status,
            Task.title,
            Task.created_at,
            last_activity,
            activity.c.message_count,
            Task.total_tokens,
            Task.description,
            Task.output,
            Task.error_message,
        )
        .order_by(last_activity.desc(), Task.id.desc())
        .limit(EXPORT_ROW_LIMIT + 1)
        .yield_per(EXPORT_FETCH_BATCH)
    )
    # The result is closed here, on every path: a cursor left to garbage
    # collection would close after the connection was back in the pool.
    result_rows = iter(rows)
    try:
        raw = _write_export_file(_export_rows(db, result_rows))
    finally:
        result_rows.close()
    # Return the connection to the pool now; ``get_db`` would otherwise hold it
    # until the response has been sent.
    db.close()
    filename = datetime.now(timezone.utc).strftime("conversation-logs-%Y%m%d-%H%M%S")
    return StreamingResponse(
        _iter_export_file(raw),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}.csv"',
            "Cache-Control": "no-store",
        },
        background=BackgroundTask(raw.close),
    )


@router.get("/{task_id}")
async def get_conversation_log_detail(
    task_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Return one hidden external conversation log.

    Admin users can inspect hidden external conversation logs across all users.
    Non-admin users are limited to their own logs. A log the retention purge
    expired answers ``410 task_expired`` to exactly those callers and 404 to
    everyone else; the list never includes it.
    """
    query = _base_task_query(db, user).filter(Task.id == task_id)
    task = query.first()
    if task is None:
        raise _missing_conversation_log_error(db, user, task_id)

    ui_source = _ui_source_for_task(db, task)
    if ui_source is None:
        raise HTTPException(status_code=404, detail="Conversation log not found")

    # Hidden messages (a client's resume signal) are plumbing, not conversation.
    messages = (
        _visible_chat_messages(task) if task.conversation_storage_version == 1 else []
    )
    file_reference_records = load_assistant_file_reference_records(
        db,
        task_id=int(task.id),
        user_id=int(task.user_id),
    )
    if task.conversation_storage_version == 2:
        from ..services.task_event_display import load_event_display_snapshot

        view = load_event_display_snapshot(db, int(task.id))
        transcript = []
        shown = [
            message
            for message in view.messages
            if not is_display_none_metadata(message.get("message_metadata"))
        ]
        for message in shown:
            item = dict(message)
            item.pop("message_metadata", None)
            if item["role"] == "assistant":
                item["content"] = reconcile_assistant_file_references(
                    db,
                    task_id=int(task.id),
                    user_id=int(task.user_id),
                    content=item["content"],
                    records=file_reference_records,
                )
            item["created_at"] = format_datetime_for_api(item["created_at"])
            transcript.append(item)
        traces = [
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
            for event in view.events
            if event["type"] == "trace_event"
            and event["event_type"] not in {"user_message", "agent_message"}
        ]
        positioned = [(item["id"], item) for item in transcript]
        for notice in view.compactions:
            item = dict(notice)
            position = item.pop("sequence")
            item["created_at"] = format_datetime_for_api(item["created_at"])
            positioned.append((position, item))
        transcript = [item for _, item in sorted(positioned, key=lambda pair: pair[0])]
        event_activity = (
            len(shown),
            max((item["created_at"] for item in shown), default=None),
        )
    else:
        transcript = _serialize_transcript_with_events(
            db, task, messages, file_reference_records
        )
        traces = _serialize_trace_events(db, int(task.id))
        event_activity = None
    return {
        "log": _serialize_log_summary(task, ui_source, event_activity=event_activity),
        "transcript": transcript,
        "trace_events": traces,
        # When retention last removed this task's trace (#2565). Later turns
        # write new trace rows, so a non-null value does not mean
        # ``trace_events`` is empty: it means the events from before this
        # moment are gone and the timeline may be incomplete.
        "trace_events_expired_at": format_datetime_for_api(task.traces_expired_at)
        if task.conversation_storage_version == 1
        else None,
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
