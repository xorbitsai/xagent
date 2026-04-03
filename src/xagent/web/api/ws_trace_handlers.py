"""WebSocket trace handlers for real-time updates."""

import asyncio
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
from .websocket import create_stream_event, manager


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
    elif action == TraceAction.ERROR:
        return "trace_error"
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


class WebSocketTraceHandler(TraceHandler):
    """Trace handler that sends events to WebSocket clients."""

    def __init__(self, task_id: int):
        self.task_id = task_id
        self._task_description: Optional[str] = None
        self._task_description_loaded = False

    async def handle_event(self, event: TraceEvent) -> None:
        """Send trace event to WebSocket clients using unified stream format."""
        try:
            # Debug: Log the event being handled (reduced verbosity)
            logger.debug(
                f"WebSocketTraceHandler handling event: {event.event_type.value} for task {self.task_id}"
            )

            # Load task description if not already loaded
            await self._load_task_description()

            # Convert trace event to unified stream format
            stream_event = self._convert_trace_event_to_stream_event(event)

            # Send to all connected WebSocket clients for this task
            if stream_event:
                logger.debug(
                    f"WebSocketTraceHandler sending stream event: {stream_event.get('event_type')} (id: {stream_event.get('event_id')}) to task {self.task_id}"
                )
                await manager.broadcast_to_task(stream_event, self.task_id)
            else:
                logger.debug(
                    f"WebSocketTraceHandler no stream event to send for event: {event.event_type.value}"
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
            await asyncio.to_thread(self._sync_load_task_description)
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

    def _convert_trace_event_to_stream_event(self, event: TraceEvent) -> Dict[str, Any]:
        """Convert trace event to unified stream format."""
        event_type_str = get_event_type_mapping(event)
        logger.debug(
            f"Converting trace event to stream event: {event_type_str} for task {self.task_id}"
        )

        # Make a deep copy of data and serialize non-JSON-serializable objects
        data = self._serialize_data(event.data)

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

        # Add task description if available
        if self._task_description:
            stream_event["data"]["task_description"] = self._task_description

        return stream_event

    def _serialize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively serialize data to ensure JSON compatibility."""
        import json
        from datetime import datetime

        def clean_string(value: str) -> str:
            """Clean string data to remove problematic characters for JSON."""
            if not isinstance(value, str):
                return value

            # Remove NULL characters and other problematic control characters
            cleaned = value.replace("\x00", "")  # Remove NULL character
            cleaned = cleaned.replace("\u0000", "")  # Remove Unicode NULL
            # Remove other control characters that might cause issues
            cleaned = "".join(
                char for char in cleaned if ord(char) >= 32 or char in "\n\r\t"
            )
            return cleaned

        def serialize_value(value: Any) -> Any:
            # Handle Pydantic models (BaseModel)
            if hasattr(value, "model_dump"):
                return serialize_value(value.model_dump())
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
