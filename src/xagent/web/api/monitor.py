"""Monitoring management API route handlers"""

import logging
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import Text
from sqlalchemy import cast as sql_cast
from sqlalchemy import func
from sqlalchemy.orm import Session
from sqlalchemy.sql import expression

from ..auth_dependencies import get_current_user
from ..models.database import get_db
from ..models.task import Task
from ..models.user import User
from ..utils.db_timezone import safe_timestamp_to_unix

logger = logging.getLogger(__name__)

# Create router
monitor_router = APIRouter(prefix="/api/monitor", tags=["monitor"])

# The guard below matches on the payload's *text* form, where a doubled
# backslash is an escaped backslash rather than the start of an escape.
# Those are neutralized first, so every backslash that survives opens a
# real JSON escape. That is what makes the match exact in both directions:
# without it, text that merely looks like an escape is dropped, and worse,
# a real unpaired surrogate sitting right after such text is missed.
PG_ESCAPED_BACKSLASH = r"\\"
PG_ESCAPED_BACKSLASH_STANDIN = "__"

# A valid surrogate pair converts to text without complaint, so pairs are
# removed before matching -- any surrogate escape still standing after that
# is unpaired by construction. Stripping is also what keeps this cheap:
# the lookahead/lookbehind form this replaces cost roughly an order of
# magnitude more on a table the monitoring dashboard scans.
PG_SURROGATE_PAIR_PATTERN = (
    r"\\u[dD][89abAB][0-9a-fA-F]{2}\\u[dD][c-fC-F][0-9a-fA-F]{2}"
)

# What PostgreSQL stores happily in a json column but refuses to convert to
# text: the NUL escape, and either half of an unpaired surrogate. Matched
# against the normalized, pair-stripped text produced above.
#
# This assumes a UTF8 server encoding, which is what the shipped compose file
# runs. On a server in another encoding ->> also rejects any escape naming a
# character outside that charset -- including a valid surrogate pair, which
# the step above deliberately strips -- so the list is not exhaustive there.
PG_UNSAFE_ESCAPE_PATTERN = (
    r"\\u0000"
    r"|\\u[dD][89abAB][0-9a-fA-F]{2}"
    r"|\\u[dD][c-fC-F][0-9a-fA-F]{2}"
)


def is_admin_user(user: User) -> bool:
    """Check if user is an administrator"""
    return bool(user.is_admin)


def get_user_filter_condition() -> None:
    """Get user filter condition - used for administrators to view all data"""
    return None  # Administrators can view all data


def get_user_specific_filter(user_id: int) -> Any:
    """Get filter condition for specific user"""
    return Task.user_id == user_id


def get_json_field_expression(column: Any, field_path: str, db_session: Session) -> Any:
    """
    Cross-database JSON field extraction expression

    Args:
        column: SQLAlchemy column object
        field_path: JSON field path, such as 'tool_name' or '$.tool_name'
        db_session: Database session used to detect database dialect

    Returns:
        JSON field extraction expression suitable for the current database
    """
    # Ensure field path format is correct
    if field_path.startswith("$."):
        field_name = field_path[2:]  # Remove '$.' prefix
    else:
        field_name = field_path

    # Detect database dialect
    if db_session.bind is None:
        raise ValueError("Database session bind is None")

    dialect_name = db_session.bind.dialect.name

    if dialect_name == "postgresql":
        # PostgreSQL extracts a JSON field as text with ->>. A json value can
        # legally carry escape sequences that ->> then refuses to convert --
        # NUL raises "unsupported Unicode escape sequence", an unpaired UTF-16
        # surrogate raises "invalid input syntax for type json" -- and one such
        # row fails the entire query. Null those payloads out before extracting
        # so the row drops instead.
        #
        # These columns are jsonb since #1248, and jsonb rejects such payloads
        # at INSERT, so on a migrated database this guard can never match. It
        # stays because it is not this module's business to assume the
        # migration has run: a database still on the json type -- one upgraded
        # by hand, or mid-rollout -- carries exactly the rows it defends
        # against.
        #
        # Retirement condition, so this does not linger on an unmeasurable
        # "while any deployment might still hold one": it can go once no
        # supported upgrade path reaches this code with a json column -- that
        # is, once skipping the 20260813 migration is no longer supported.
        # What to check on a given database before removing it:
        #
        #     SELECT data_type FROM information_schema.columns
        #     WHERE table_name = 'trace_events' AND column_name = 'data';
        #
        # Because jsonb makes this branch unreachable over the model's own
        # table, its drop path is exercised against a throwaway native-json
        # table by tests/web/api/test_monitor_postgresql.py's
        # TestReadGuardAgainstNativeJson -- delete that alongside this.
        #
        # Matching runs on the column's text form because valid JSON never
        # holds a raw control character or a bare surrogate (RFC 8259 requires
        # escaping) -- the escape sequence is what has to be found. ~ is the
        # regex-match operator and applies to text; the ~? used here before
        # does not exist in PostgreSQL at all, so every query through this
        # branch failed.
        #
        # Two normalizations run before the match, and both are load-bearing:
        # escaped backslashes become an inert stand-in so nothing that merely
        # looks like an escape is treated as one, and valid surrogate pairs are
        # deleted so whatever surrogate escape remains is unpaired. Only then
        # is a plain alternation enough -- no lookaround, which is what made an
        # earlier version of this guard cost an order of magnitude more on a
        # table these dashboard endpoints scan.
        #
        # The MySQL/SQLite branches strip the escape and keep the row. Doing
        # that here would mean casting the edited text back to json, which
        # raises whenever the edit leaves invalid JSON -- one bad row would
        # again fail the request.
        payload_text = func.replace(
            sql_cast(column, Text),
            PG_ESCAPED_BACKSLASH,
            PG_ESCAPED_BACKSLASH_STANDIN,
        )
        unpaired_only = func.regexp_replace(
            payload_text, PG_SURROGATE_PAIR_PATTERN, "", "g"
        )
        valid_data = expression.case(
            (
                unpaired_only.op("~")(PG_UNSAFE_ESCAPE_PATTERN),
                expression.null(),
            ),
            else_=column,
        )
        return valid_data.op("->>")(field_name)
    elif dialect_name == "mysql":
        # MySQL uses JSON_EXTRACT function, also cleans NULL characters
        cleaned_json = func.replace(
            func.replace(func.replace(column, "\\u0000", ""), "\x00", ""), "\\n", " "
        )
        return func.json_extract(cleaned_json, f"$.{field_name}")
    else:
        # SQLite and other databases use json_extract function, also cleans NULL characters
        cleaned_json = func.replace(
            func.replace(func.replace(column, "\\u0000", ""), "\x00", ""), "\\n", " "
        )
        return func.json_extract(cleaned_json, f"$.{field_name}")


def safe_get_json_field(column: Any, field_path: str, db_session: Session) -> Any:
    """JSON field extraction expression for the session's dialect.

    Args:
        column: SQLAlchemy column object
        field_path: JSON field path
        db_session: Database session

    Returns:
        JSON field extraction expression suitable for the current database
    """
    return get_json_field_expression(column, field_path, db_session)


@monitor_router.get("/tools")
async def get_tools() -> Dict[str, Any]:
    """Get list of available tools"""
    try:
        from ...core.agent.service import AgentService
        from ...core.memory.in_memory import InMemoryMemoryStore

        # Create AgentService with auto tool config
        agent_service = AgentService(name="monitor_tools", memory=InMemoryMemoryStore())

        # Trigger tool initialization
        await agent_service._ensure_tools_initialized()

        return {
            "tools": [
                {
                    "name": tool.metadata.name,
                    "description": tool.metadata.description,
                    "schema": tool.metadata.schema
                    if hasattr(tool.metadata, "schema")
                    else None,
                }
                for tool in agent_service.tools
            ],
            "count": len(agent_service.tools),
        }
    except Exception as e:
        logger.error(f"Get tools failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@monitor_router.get("/agents")
async def get_agents(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """Get list of agents"""
    try:
        # Build query based on user permissions
        query = db.query(Task)

        if not is_admin_user(current_user):
            # Regular users can only view their own tasks
            query = query.filter(Task.user_id == current_user.id)

        # Get recent tasks
        recent_tasks = query.order_by(Task.created_at.desc()).limit(10).all()

        return [
            {
                "task_id": task.id,
                "title": task.title,
                "status": task.status.value,
                "created_at": safe_timestamp_to_unix(task.created_at)
                if task.created_at
                else None,
                "updated_at": safe_timestamp_to_unix(task.updated_at)
                if task.updated_at
                else None,
            }
            for task in recent_tasks
        ]
    except Exception as e:
        logger.error(f"Get agents failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@monitor_router.get("/stats")
async def get_monitoring_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get monitoring statistics"""
    try:
        from ..models.task import TraceEvent

        # Build TraceEvent query filter based on user permissions
        trace_event_filter = []
        # Only count trace events from VIBE phase (exclude BUILD phase)
        trace_event_filter.append(TraceEvent.build_id.is_(None))

        if not is_admin_user(current_user):
            # Regular users can only view TraceEvents from their own tasks
            trace_event_filter.append(
                TraceEvent.task_id.in_(
                    db.query(Task.id).filter(Task.user_id == current_user.id)
                )
            )

        # Get total call count (LLM calls + tool executions)
        llm_calls_start = (
            db.query(TraceEvent)
            .filter(TraceEvent.event_type == "llm_call_start", *trace_event_filter)
            .count()
        )
        llm_calls_end = (
            db.query(TraceEvent)
            .filter(TraceEvent.event_type == "llm_call_end", *trace_event_filter)
            .count()
        )
        tool_executions_start = (
            db.query(TraceEvent)
            .filter(
                TraceEvent.event_type == "tool_execution_start", *trace_event_filter
            )
            .count()
        )
        tool_executions_end = (
            db.query(TraceEvent)
            .filter(TraceEvent.event_type == "tool_execution_end", *trace_event_filter)
            .count()
        )
        total_calls = llm_calls_end + tool_executions_end

        # Get today's call count
        today = datetime.now().date()
        today_start = datetime.combine(today, datetime.min.time())
        today_calls = (
            db.query(TraceEvent)
            .filter(
                TraceEvent.event_type.in_(["llm_call_start", "tool_execution_start"]),
                TraceEvent.timestamp >= today_start,
                *trace_event_filter,
            )
            .count()
        )

        # Calculate success rate (based on successfully completed tasks)
        # Count task completion from TraceEvents
        completed_tasks = (
            db.query(TraceEvent.task_id)
            .filter(
                TraceEvent.event_type.in_(["task_completion", "task_end_react"]),
                TraceEvent.task_id.isnot(None),
                *trace_event_filter,
            )
            .distinct()
            .count()
        )

        total_tasks_with_events = (
            db.query(TraceEvent.task_id)
            .filter(
                TraceEvent.task_id.isnot(None),
                *trace_event_filter,
            )
            .distinct()
            .count()
        )

        success_rate = (
            (completed_tasks / total_tasks_with_events * 100)
            if total_tasks_with_events > 0
            else 0
        )

        # Calculate average processing time (based on actual execution time of LLM calls)
        # Use more precise matching logic: match start and end events by step_id and attempt
        llm_starts = (
            db.query(TraceEvent)
            .filter(
                TraceEvent.event_type == "llm_call_start",
                TraceEvent.data.isnot(None),
                TraceEvent.task_id.isnot(None),
                *trace_event_filter,
            )
            .all()
        )

        llm_ends = (
            db.query(TraceEvent)
            .filter(
                TraceEvent.event_type == "llm_call_end",
                TraceEvent.data.isnot(None),
                TraceEvent.task_id.isnot(None),
                *trace_event_filter,
            )
            .all()
        )

        # Build lookup table for end events
        end_events: dict[tuple[int, str, int], datetime] = {}
        for end_event in llm_ends:
            # Try to match by step_id and attempt
            if end_event.data and isinstance(end_event.data, dict):
                step_id = end_event.data.get("step_id")
                attempt = end_event.data.get("attempt")
                task_id = end_event.task_id

                if step_id and attempt:
                    key = (task_id, step_id, attempt)
                    end_events[key] = end_event.timestamp

        # Match start events with end events
        valid_durations: list[float] = []
        for start_event in llm_starts:
            if start_event.data and isinstance(start_event.data, dict):
                step_id = start_event.data.get("step_id")
                attempt = start_event.data.get("attempt")
                task_id = start_event.task_id

                if step_id and attempt:
                    key = (task_id, step_id, attempt)
                    if key in end_events:
                        duration = (
                            end_events[key] - start_event.timestamp
                        ).total_seconds()
                        # Exclude outliers: less than 0 or greater than 1 hour
                        if 0 < duration <= 3600:
                            valid_durations.append(duration)

        avg_response_time = (
            round(sum(valid_durations) / len(valid_durations), 2)
            if valid_durations
            else None
        )

        # Get active model count
        try:
            # Use safe JSON field extraction to avoid NULL character issues
            model_name_expr = safe_get_json_field(TraceEvent.data, "model_name", db)
            active_models = (
                db.query(model_name_expr)
                .filter(
                    TraceEvent.event_type == "llm_call_start",
                    TraceEvent.timestamp >= today_start,
                    TraceEvent.data.isnot(None),
                    model_name_expr.isnot(None),
                    *trace_event_filter,
                )
                .distinct()
                .count()
            )
        except Exception as e:
            logger.error(f"Failed to query active models: {e}")
            active_models = 0

        # Get total token count
        total_tokens: int | None = 0
        tokens_found = False
        llm_end_events = (
            db.query(TraceEvent)
            .filter(
                TraceEvent.event_type == "llm_call_end",
                TraceEvent.data.isnot(None),
                *trace_event_filter,
            )
            .all()
        )

        for event in llm_end_events:
            if event.data:
                if "total_tokens" in event.data and isinstance(
                    event.data["total_tokens"], int
                ):
                    total_tokens += event.data["total_tokens"]
                    tokens_found = True
                elif "usage" in event.data and isinstance(event.data["usage"], dict):
                    usage = event.data["usage"]
                    if "total_tokens" in usage and isinstance(
                        usage["total_tokens"], int
                    ):
                        total_tokens += usage["total_tokens"]
                        tokens_found = True
                    elif (
                        "prompt_tokens" in usage
                        and "completion_tokens" in usage
                        and isinstance(usage["prompt_tokens"], int)
                        and isinstance(usage["completion_tokens"], int)
                    ):
                        total_tokens += (
                            usage["prompt_tokens"] + usage["completion_tokens"]
                        )
                        tokens_found = True

        # If no token information found, set to None
        if not tokens_found:
            total_tokens = None

        return {
            "totalCalls": total_calls,
            "successRate": round(success_rate, 1),
            "avgResponseTime": avg_response_time,
            "activeModels": active_models,
            "totalTokens": total_tokens,
            "todayCalls": today_calls,
            "totalTasks": total_tasks_with_events,
            "completedTasks": completed_tasks,
            "failedTasks": total_tasks_with_events - completed_tasks,
            "runningTasks": None,
            "totalAgents": None,
            "llmCalls": llm_calls_start,
            "toolExecutions": tool_executions_start,
        }
    except Exception as e:
        logger.error(f"Get monitoring stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@monitor_router.get("/popular-tools")
async def get_popular_tools(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """Get popular tools statistics"""
    try:
        from ..models.task import TraceEvent

        # Build filter conditions based on user permissions
        trace_event_filter = []
        if not is_admin_user(current_user):
            # Regular users can only view TraceEvents from their own tasks
            trace_event_filter.append(
                TraceEvent.task_id.in_(
                    db.query(Task.id).filter(Task.user_id == current_user.id)
                )
            )

        # Count tool usage from TraceEvents
        try:
            # Use safe JSON field extraction
            tool_name_expr = safe_get_json_field(TraceEvent.data, "tool_name", db)
            tool_usage_stats = (
                db.query(
                    tool_name_expr.label("tool_name"),
                    func.count(TraceEvent.event_id).label("usage_count"),
                )
                .filter(
                    TraceEvent.event_type == "tool_execution_start",
                    TraceEvent.data.isnot(None),
                    tool_name_expr.isnot(None),
                    *trace_event_filter,
                )
                .group_by(tool_name_expr)
                .all()
            )
        except Exception as e:
            logger.error(f"Failed to query tool usage stats: {e}")
            tool_usage_stats = []

        # Convert to list format
        result = []
        for tool_name, usage_count in tool_usage_stats:
            if tool_name:
                result.append(
                    {
                        "name": tool_name,
                        "description": f"Tool: {tool_name}",
                        "usage_count": usage_count,
                        "avg_duration": 0,  # Simplified handling
                    }
                )

        # Sort by usage count
        result.sort(key=lambda x: x["usage_count"], reverse=True)

        # If no data, return empty list, do not create any data
        return result[:10]  # Return top 10
    except Exception as e:
        logger.error(f"Get popular tools failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@monitor_router.get("/model-stats")
async def get_model_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """Get model usage statistics"""
    try:
        from ..models.task import TraceEvent

        # Build filter conditions based on user permissions
        trace_event_filter = []
        if not is_admin_user(current_user):
            # Regular users can only view TraceEvents from their own tasks
            trace_event_filter.append(
                TraceEvent.task_id.in_(
                    db.query(Task.id).filter(Task.user_id == current_user.id)
                )
            )

        # Get real LLM call statistics from TraceEvents
        # Count usage for each model (based on llm_call_start events)
        try:
            # Use safe JSON field extraction
            model_name_expr = safe_get_json_field(TraceEvent.data, "model_name", db)
            model_stats = (
                db.query(
                    model_name_expr.label("model_name"),
                    func.count(TraceEvent.event_id).label("total_calls"),
                )
                .filter(
                    TraceEvent.event_type == "llm_call_start",
                    TraceEvent.data.isnot(None),
                    model_name_expr.isnot(None),
                    *trace_event_filter,
                )
                .group_by(model_name_expr)
                .all()
            )
        except Exception as e:
            logger.error(f"Failed to query model stats: {e}")
            model_stats = []

        # Rows the response will carry. Filtered here rather than inside the
        # loop so a dropped row stays out of the denominator too: the query
        # rejects a NULL model name but not an empty one, and counting calls
        # that are never reported would deflate every rate that is.
        counted_models = [
            (model_name, model_calls)
            for model_name, model_calls in model_stats
            if model_name and model_calls > 0
        ]

        # Denominator for each model's share of the traffic. Must stay distinct
        # from the loop's per-model variable: while the two shared a name the
        # rate divided a model's calls by itself and every model reported 100%
        # (#1245).
        all_model_calls = sum(model_calls for _, model_calls in counted_models)

        result = []
        for model_name, model_calls in counted_models:
            # No zero-guard needed: the sum runs over these same rows, each of
            # which the comprehension above established is positive.
            usage_rate = model_calls / all_model_calls * 100

            result.append(
                {
                    "name": model_name,
                    "status": "running",
                    "usage_rate": round(usage_rate, 1),
                    "success_rate": None,  # Simplified, do not calculate success rate
                    "total_tasks": model_calls,
                    "successful_tasks": None,
                    "failed_tasks": None,
                }
            )

        return result
    except Exception as e:
        logger.error(f"Get model stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@monitor_router.get("/dashboard-stats")
async def get_dashboard_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get dashboard statistics"""
    try:
        from ..models.task import Task, TaskStatus, TraceEvent, task_status_predicate

        # Build filter conditions based on user permissions
        task_filter = []
        if not is_admin_user(current_user):
            task_filter.append(Task.user_id == current_user.id)

        # Get total task count
        total_tasks = db.query(Task).filter(*task_filter).count()

        # Get active agent count (based on tasks with recent activity)
        recent_active_time = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        active_agents = (
            db.query(Task)
            .filter(
                Task.updated_at >= recent_active_time,
                task_status_predicate.in_([TaskStatus.RUNNING, TaskStatus.PENDING]),
                *task_filter,
            )
            .count()
        )

        # Get deployed application count (temporarily set to 0, waiting for Deploy feature implementation)
        deployed_apps = 0

        # Get today's call count
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        # Build TraceEvent filter conditions
        trace_event_filter = []
        if not is_admin_user(current_user):
            trace_event_filter.append(
                TraceEvent.task_id.in_(
                    db.query(Task.id).filter(Task.user_id == current_user.id)
                )
            )

        today_calls = (
            db.query(TraceEvent)
            .filter(
                TraceEvent.event_type.in_(["llm_call_start", "tool_execution_start"]),
                TraceEvent.timestamp >= today_start,
                *trace_event_filter,
            )
            .count()
        )

        return {
            "totalTasks": total_tasks,
            "activeAgents": active_agents,
            "deployedApps": deployed_apps,
            "todayCalls": today_calls,
        }
    except Exception as e:
        logger.error(f"Get dashboard stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
