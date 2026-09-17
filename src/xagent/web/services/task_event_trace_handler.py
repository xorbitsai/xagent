"""Publish task trace events through the host event delivery adapter."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ...core.agent.trace import (
    TraceAction,
    TraceCategory,
    TraceEvent,
    TraceHandler,
    TraceScope,
)
from ...core.runtime_performance import (
    increment_counter,
    observe_duration,
    run_in_thread_with_telemetry,
)
from .public_trace_events import is_audit_only_trace_data, normalize_public_trace_event
from .task_events import publish_task_event, task_has_audience
from .task_execution import create_stream_event
from .trace_event_types import (
    LEGACY_GENERAL_ERROR_EVENT_TYPE,
    STEP_GENERAL_ERROR_EVENT_TYPE,
    TASK_GENERAL_ERROR_EVENT_TYPE,
)


# Helper function to map new event types to old-style handling for compatibility
def get_event_type_mapping(event: TraceEvent) -> str:
    """Map new trace event types to old-style string identifiers for compatibility."""
    scope = event.event_type.scope
    action = event.event_type.action
    category = event.event_type.category

    # Map to old-style event type names
    if (
        scope == TraceScope.TASK
        and action == TraceAction.START
        and category == TraceCategory.DAG_PLAN
    ):
        return "dag_plan_start"
    elif (
        scope == TraceScope.TASK
        and action == TraceAction.END
        and category == TraceCategory.DAG_PLAN
    ):
        return "dag_plan_end"
    elif (
        scope == TraceScope.TASK
        and action == TraceAction.START
        and category == TraceCategory.DAG
    ):
        return "dag_execute_start"
    elif (
        scope == TraceScope.TASK
        and action == TraceAction.UPDATE
        and category == TraceCategory.DAG
    ):
        return "dag_execution"
    elif (
        scope == TraceScope.TASK
        and action == TraceAction.END
        and category == TraceCategory.DAG
    ):
        return "dag_execute_end"
    elif (
        scope == TraceScope.STEP
        and action == TraceAction.START
        and category == TraceCategory.DAG
    ):
        return "dag_step_start"
    elif (
        scope == TraceScope.STEP
        and action == TraceAction.END
        and category == TraceCategory.DAG
    ):
        return "dag_step_end"
    elif (
        scope == TraceScope.ACTION
        and action == TraceAction.START
        and category == TraceCategory.LLM
    ):
        return "llm_call_start"
    elif (
        scope == TraceScope.ACTION
        and action == TraceAction.END
        and category == TraceCategory.LLM
    ):
        return "llm_call_end"
    elif (
        scope == TraceScope.ACTION
        and action == TraceAction.ERROR
        and category == TraceCategory.LLM
    ):
        return "llm_call_failed"
    elif (
        scope == TraceScope.ACTION
        and action == TraceAction.START
        and category == TraceCategory.TOOL
    ):
        return "tool_execution_start"
    elif (
        scope == TraceScope.ACTION
        and action == TraceAction.END
        and category == TraceCategory.TOOL
    ):
        return "tool_execution_end"
    elif (
        scope == TraceScope.ACTION
        and action == TraceAction.ERROR
        and category == TraceCategory.TOOL
    ):
        return "tool_execution_failed"
    elif (
        scope == TraceScope.TASK
        and action == TraceAction.ERROR
        and category == TraceCategory.GENERAL
    ):
        return TASK_GENERAL_ERROR_EVENT_TYPE
    elif (
        scope == TraceScope.STEP
        and action == TraceAction.ERROR
        and category == TraceCategory.GENERAL
    ):
        return STEP_GENERAL_ERROR_EVENT_TYPE
    elif action == TraceAction.ERROR:
        return LEGACY_GENERAL_ERROR_EVENT_TYPE
    elif (
        scope == TraceScope.ACTION
        and action == TraceAction.START
        and category == TraceCategory.COMPACT
    ):
        return "action_start_compact"
    elif (
        scope == TraceScope.ACTION
        and action == TraceAction.END
        and category == TraceCategory.COMPACT
    ):
        return "action_end_compact"
    elif (
        scope == TraceScope.SYSTEM
        and action == TraceAction.UPDATE
        and category == TraceCategory.VISUALIZATION
    ):
        return "visualization_update"
    elif (
        scope == TraceScope.TASK
        and action == TraceAction.START
        and category == TraceCategory.MESSAGE
    ):
        # 特殊处理 user_message 事件，记录日志以调试重复显示问题
        logger.info(
            f"📨 Mapping trace event to 'user_message': scope={scope.value}, action={action.value}, category={category.value}"
        )
        return "user_message"
    elif (
        scope == TraceScope.TASK
        and action == TraceAction.END
        and category == TraceCategory.MESSAGE
    ):
        return "ai_message"
    elif (
        scope == TraceScope.TASK
        and action == TraceAction.END
        and category == TraceCategory.GENERAL
    ):
        return "task_completion"
    # Skill selection events
    elif (
        scope == TraceScope.TASK
        and action == TraceAction.START
        and category == TraceCategory.SKILL
    ):
        return "skill_select_start"
    elif (
        scope == TraceScope.TASK
        and action == TraceAction.END
        and category == TraceCategory.SKILL
    ):
        return "skill_select_end"
    # ReAct pattern events (for BUILD phase)
    elif (
        scope == TraceScope.TASK
        and action == TraceAction.START
        and category == TraceCategory.REACT
    ):
        return "react_task_start"
    elif (
        scope == TraceScope.TASK
        and action == TraceAction.END
        and category == TraceCategory.REACT
    ):
        return "react_task_end"
    elif (
        scope == TraceScope.STEP
        and action == TraceAction.START
        and category == TraceCategory.REACT
    ):
        return "react_step_start"
    elif (
        scope == TraceScope.STEP
        and action == TraceAction.END
        and category == TraceCategory.REACT
    ):
        return "react_step_end"
    elif (
        scope == TraceScope.ACTION
        and action == TraceAction.START
        and category == TraceCategory.REACT
    ):
        return "react_action_start"
    elif (
        scope == TraceScope.ACTION
        and action == TraceAction.END
        and category == TraceCategory.REACT
    ):
        return "react_action_end"
    else:
        # Fallback to event type value
        return event.event_type.value


logger = logging.getLogger(__name__)


def is_agent_checkpoint_data(data: Any) -> bool:
    """Return True for internal agent checkpoint payloads.

    These events are persisted for resume/recovery, but they are too large and
    too low-level for the user-facing execution log stream.
    """
    if not isinstance(data, dict):
        return False
    try:
        from ...core.agent.checkpoint import READABLE_CHECKPOINT_TYPES
    except Exception:
        READABLE_CHECKPOINT_TYPES = frozenset(
            {"agent_execution_checkpoint", "agent_v2_execution_checkpoint"}
        )
    return data.get("checkpoint_type") in READABLE_CHECKPOINT_TYPES or (
        data.get("type") == "checkpoint"
        and isinstance(data.get("pattern_state"), dict)
        and isinstance(data.get("context"), dict)
    )


# Maps every control character (codepoint 0-31) to deletion, except tab
# (9), newline (10), and carriage return (13), which are kept -- same
# survivor set the old per-character loop in ``clean_string`` kept via
# ``ord(char) >= 32 or char in "\n\r\t"``. DEL (127) and the C1 control
# range (128-159) are untouched by this table (and were untouched by the
# old loop too, since both are >= 32) -- this table only ever removes,
# never rewrites, so leaving them out is the same as mapping them to
# themselves. The table's contents never vary, and ``clean_string`` needs
# it for every string value in every trace event this module serializes,
# so it lives at module scope rather than inside the function.
_CONTROL_CHAR_TRANSLATION_TABLE = str.maketrans(
    {codepoint: None for codepoint in range(32) if codepoint not in (9, 10, 13)}
)


def serialize_trace_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively serialize trace event data to ensure JSON compatibility.

    Module-level so the v1 SSE content-projection layer
    (``v1/_events_stream.py``) can reuse the exact same pass on live
    broadcast frames that the WebSocket handler applies before
    broadcasting -- this function never reads ``self``, so lifting it
    out of ``TaskEventTraceHandler`` changes nothing about its
    behavior for the existing caller below.

    Truncation, field pruning, or any wholesale replacement of the
    payload added here changes how this pass relates to the two agent
    checkpoint checks in ``_convert_trace_event_to_stream_event``: one
    runs on the raw payload and one on this function's output, and they
    agree only while this pass either preserves the checked fields or
    replaces the payload outright. The fallback below already does the
    latter, which is why the raw check exists. Any change to what this
    function may return must be reviewed together with both checks. It
    must also be reviewed against ``v1/_events_stream.py``, which reuses
    this pass on live broadcast frames but carries no checkpoint check of
    its own -- it relies on this handler having already dropped them.

    The mechanical guard for this is the parametrized case in
    ``test_raw_and_serialized_checkpoint_checks_drop_the_same_set`` that
    feeds a checkpoint payload this pass cannot serialize: it asserts the
    raw check catches it before this fallback runs.
    """
    import json
    from datetime import datetime

    def clean_string(value: str) -> str:
        """Clean string data to remove problematic characters for JSON.

        ``str.translate`` with the module-level
        ``_CONTROL_CHAR_TRANSLATION_TABLE`` filters the whole string in
        one C-level call. The previous form ran one Python-level
        generator step per character and fed the survivors to
        ``"".join`` -- the join was already a C builtin; the
        per-character iteration was not. Same survivor set either way:
        codepoints 0-31 are dropped except tab, newline, and carriage
        return.
        """
        if not isinstance(value, str):
            return value
        return value.translate(_CONTROL_CHAR_TRANSLATION_TABLE)

    def serialize_value(value: Any) -> Any:
        # Handle Pydantic models (BaseModel)
        if hasattr(value, "model_dump"):
            return serialize_value(value.model_dump())
        elif callable(getattr(value, "to_dict", None)):
            return serialize_value(value.to_dict())
        elif hasattr(value, "dict"):  # Fallback for older Pydantic
            return serialize_value(value.dict())
        elif isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.timestamp()
        elif isinstance(value, str):
            return clean_string(value)
        elif isinstance(value, dict):
            return {k: serialize_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [serialize_value(item) for item in value]
        elif isinstance(value, bytes):
            try:
                return clean_string(value.decode("utf-8"))
            except UnicodeDecodeError:
                return f"<bytes: {len(value)}>"
        else:
            return value

    try:
        # First clean and serialize the data
        cleaned_data = serialize_value(data)

        # Test if cleaned data is JSON serializable
        json.dumps(cleaned_data)
        return cleaned_data  # type: ignore[no-any-return]
    except (TypeError, ValueError) as e:
        # If still not serializable, return a safe fallback
        logger.warning(
            f"Failed to serialize data for JSON: {e}, data type: {type(data)}"
        )
        return {
            "_serialization_error": f"Failed to serialize {type(data).__name__}",
            "_original_type": type(data).__name__,
            "_error": str(e),
        }


def _convert_timestamp_to_utc_timestamp(timestamp: Any) -> float:
    """Convert timestamp to Unix timestamp for WebSocket compatibility."""
    if timestamp is None:
        return datetime.now(timezone.utc).timestamp()
    elif isinstance(timestamp, (int, float)):
        return float(timestamp)
    elif isinstance(timestamp, datetime):
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.timestamp()  # type: ignore[no-any-return]
    else:
        # Fallback to current UTC time
        return datetime.now(timezone.utc).timestamp()


class TaskEventTraceHandler(TraceHandler):
    """Trace handler that publishes events through the host event delivery adapter."""

    def __init__(self, task_id: int):
        self.task_id = task_id
        self._task_description: Optional[str] = None
        self._task_description_loaded = False

    async def handle_event(self, event: TraceEvent) -> None:
        """Publish a trace event through the host adapter using unified stream format."""
        with observe_duration("xagent.websocket.trace_handler.duration"):
            await self._handle_event(event)

    async def _handle_event(self, event: TraceEvent) -> None:
        try:
            # Nothing is listening for this task, so every frame the rest of
            # this method could build would be discarded downstream anyway.
            # Ask that question first and skip the two database reads and the
            # recursive serialization pass between here and the publish call.
            # The answer is an optimization hint, not a delivery decision:
            # ``publish_task_event`` and whatever sink the host registered
            # still decide that. Both failure modes of the probe answer
            # "yes" -- no probe registered, and a probe that raises -- so a
            # failing probe costs work rather than frames.
            #
            # What a correct "no" still costs is a client that attaches
            # while this call is running: the broadcast path used to
            # re-read the registry after the frame was built and include
            # it, and this call has already returned by then. The paths
            # that replay history on attach recover it; the two that do
            # not replay are named in the change description.
            #
            # A client that attaches later replays this task's history from
            # ``trace_events``. ``DatabaseTraceHandler`` runs earlier on the
            # same dispatch and commits the row on every path that reaches
            # its commit; the paths that return before it (missing task,
            # duplicate turn, checkpoint pointer miss) write no row at all,
            # so a later attach misses nothing this skip took away -- the
            # frame those paths drop was already dropped downstream before
            # this change.
            if not task_has_audience(self.task_id):
                increment_counter(
                    "xagent.websocket.trace.events",
                    attributes={"outcome": "no_audience"},
                )
                return

            # Debug: Log the event being handled (reduced verbosity)
            logger.debug(
                f"TaskEventTraceHandler handling event: {event.event_type.value} for task {self.task_id}"
            )

            # Load task description if not already loaded
            await self._load_task_description()

            # Convert trace event to unified stream format
            with observe_duration("xagent.websocket.trace_serialization.duration"):
                stream_event = self._convert_trace_event_to_stream_event(event)
            if stream_event:
                stream_data = stream_event.get("data")
                if isinstance(stream_data, dict) and await run_in_thread_with_telemetry(
                    "websocket_prior_user_message_check",
                    self._has_prior_user_message_turn,
                    str(stream_event.get("event_type") or ""),
                    stream_data,
                    str(event.id),
                ):
                    stream_event = None

            # Publish through the host event delivery adapter for this task
            if stream_event:
                increment_counter(
                    "xagent.websocket.trace.events",
                    attributes={"outcome": "broadcast"},
                )
                logger.debug(
                    f"TaskEventTraceHandler sending stream event: {stream_event.get('event_type')} (id: {stream_event.get('event_id')}) to task {self.task_id}"
                )
                await publish_task_event(stream_event, self.task_id)
            else:
                increment_counter(
                    "xagent.websocket.trace.events",
                    attributes={"outcome": "dropped"},
                )
                logger.debug(
                    f"TaskEventTraceHandler no stream event to send for event: {event.event_type.value}"
                )

        except Exception as e:
            logger.warning(
                f"Failed to send trace event to WebSocket for task {self.task_id}: {e}"
            )

    async def _load_task_description(self) -> None:
        """Load task description from database."""
        if self._task_description_loaded:
            return

        try:
            # Run synchronous database operations in a thread pool to avoid blocking event loop
            await run_in_thread_with_telemetry(
                "websocket_task_description_load",
                self._sync_load_task_description,
            )
        except Exception as e:
            logger.warning(
                f"Failed to load task description for task {self.task_id}: {e}"
            )

        self._task_description_loaded = True

    def _sync_load_task_description(self) -> None:
        """Synchronous database query (runs in thread pool)."""
        from sqlalchemy.orm import Session

        from ..models.database import get_db
        from ..models.task import Task

        db_gen = get_db()
        db: Session = next(db_gen)

        try:
            task = db.query(Task).filter(Task.id == self.task_id).first()
            if task:
                self._task_description = (
                    str(task.description) if task.description else None
                )
                logger.info(
                    f"Loaded task description for task {self.task_id}: {task.description[:50]}..."
                )
            else:
                logger.warning(f"Task not found for task_id {self.task_id}")
        finally:
            db.close()

    def _convert_trace_event_to_stream_event(
        self, event: TraceEvent
    ) -> Optional[Dict[str, Any]]:
        """Convert trace event to unified stream format.

        Returns ``None`` to skip the broadcast entirely. The caller
        (``handle_event``) checks ``if stream_event:`` before sending,
        so a ``None`` here means the event still reaches
        ``DatabaseTraceHandler`` (audit row persisted) but never reaches
        any WebSocket client. Two reasons we drop:

          1. ``__audit_only__=True`` in ``event.data`` -- server-only
             RCA payload (raw LLM messages / response) that must not
             leak to clients.
          2. Agent checkpoint data -- internal runtime state the
             frontend doesn't render (checked on the raw payload first,
             then again after serialization; see the comment there for
             why both checks are kept).
        """
        # Server-only audit traces: drop early so we don't waste effort
        # serializing payloads we're about to discard.
        if is_audit_only_trace_data(event.data):
            logger.debug(
                f"Dropping audit-only trace event from WS broadcast: "
                f"step_id={event.step_id} task={self.task_id}"
            )
            return None

        # Checkpoints never reach a client either, so decide that on the raw
        # payload too -- the same reason the check directly above runs here
        # rather than after serialization. Both checkpoint checks stay,
        # because neither one subsumes the other:
        #   * a payload that is not a checkpoint here can become one
        #     below, because ``serialize_trace_data`` turns objects
        #     exposing ``model_dump`` / ``to_dict`` into plain dicts. No
        #     producer on the standard execution path emits that shape
        #     today -- ``TraceCheckpointStore`` nests the legacy payload
        #     under ``snapshot``. The check stays so this copy of the
        #     predicate agrees with the other copy, which historical
        #     replay also calls;
        #   * a payload that IS a checkpoint here can stop looking like
        #     one below, because a payload ``json.dumps`` rejects is
        #     replaced wholesale by a three-key ``_serialization_error``
        #     placeholder that carries none of the checked fields.
        # The second case is the one behavioral difference this check
        # introduces: an unserializable checkpoint used to reach clients
        # as a content-free error placeholder and no longer does. The
        # ``DatabaseTraceHandler`` copy of the same serialization pass
        # still logs that failure on the same payload.
        if is_agent_checkpoint_data(event.data):
            return None

        event_type_str = get_event_type_mapping(event)
        logger.debug(
            f"Converting trace event to stream event: {event_type_str} for task {self.task_id}"
        )

        # Make a deep copy of data and serialize non-JSON-serializable objects
        data = self._serialize_data(event.data)
        if is_agent_checkpoint_data(data):
            return None
        if self._task_description:
            data["task_description"] = self._task_description
        event_type_str, data = normalize_public_trace_event(event_type_str, data)

        # Create the base stream event
        stream_event = create_stream_event(
            event_type_str, self.task_id, data, event.timestamp
        )

        # Add step_id if present (required for tool/LLM events)
        if event.step_id:
            stream_event["step_id"] = event.step_id

        # Add parent_id if present (for event correlation)
        if event.parent_id:
            stream_event["parent_id"] = event.parent_id

        return stream_event

    def _has_prior_user_message_turn(
        self,
        event_type: str,
        data: Dict[str, Any],
        event_id: str,
    ) -> bool:
        if event_type != "user_message" or not isinstance(data, dict):
            return False
        turn_id = data.get("turn_id")
        if not isinstance(turn_id, str) or not turn_id:
            return False

        try:
            from contextlib import closing

            from ..models.database import get_db
            from ..models.task import TraceEvent as DatabaseTraceEvent

            with closing(get_db()) as db_gen:
                db = next(db_gen)
                return (
                    db.query(DatabaseTraceEvent.id)
                    .filter(
                        DatabaseTraceEvent.task_id == self.task_id,
                        DatabaseTraceEvent.event_type == "user_message",
                        DatabaseTraceEvent.event_id != event_id,
                        DatabaseTraceEvent.data["turn_id"].as_string() == turn_id,
                    )
                    .first()
                    is not None
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Could not check prior user_message turn_id=%s for task %s: %s",
                turn_id,
                self.task_id,
                exc,
            )
            return False

    def _serialize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively serialize data to ensure JSON compatibility.

        Thin instance wrapper -- the actual pass is ``serialize_trace_data``
        (module level, doesn't read ``self``) so the v1 SSE content
        projector can reuse the identical logic on live broadcast frames
        without going through this class.
        """
        return serialize_trace_data(data)
