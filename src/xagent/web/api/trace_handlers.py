"""Web-specific trace handlers for database operations."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ...core.agent.trace import BaseTraceHandler
from ...core.agent.trace import TraceEvent as CoreTraceEvent
from ...web.models.database import get_db
from ...web.models.task import TraceEvent as DatabaseTraceEvent
from ...web.models.tool_config import ToolUsage

logger = logging.getLogger(__name__)


def _convert_float_to_datetime(timestamp: Any) -> datetime:
    """Convert float timestamp to datetime for database storage."""
    if isinstance(timestamp, (int, float)):
        return datetime.fromtimestamp(timestamp, timezone.utc)
    elif isinstance(timestamp, datetime):
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=timezone.utc)
        return timestamp
    else:
        return datetime.now(timezone.utc)


class DatabaseTraceHandler(BaseTraceHandler):
    """Enhanced trace handler that saves events to database with clear scope handling."""

    def __init__(self, task_id: int, build_id: Optional[str] = None):
        super().__init__()
        self.task_id = task_id
        self.build_id = build_id

    async def _handle_task_event(self, event: CoreTraceEvent) -> None:
        """Handle task-level events for database storage."""
        await self._save_to_database(event)

    async def _handle_step_event(self, event: CoreTraceEvent) -> None:
        """Handle step-level events for database storage."""
        await self._save_to_database(event)

    async def _handle_action_event(self, event: CoreTraceEvent) -> None:
        """Handle action-level events for database storage."""
        await self._save_to_database(event)

    async def _handle_system_event(self, event: CoreTraceEvent) -> None:
        """Handle system-level events for database storage."""
        await self._save_to_database(event)

    async def _save_to_database(self, event: CoreTraceEvent) -> None:
        """Save trace event to database."""
        try:
            # Run synchronous database operations in a thread pool to avoid blocking event loop
            await asyncio.to_thread(self._sync_save_to_database, event)
        except Exception as e:
            # Don't catch required field validation errors - let them propagate
            if isinstance(e, ValueError) and ("missing required" in str(e)):
                logger.error(f"Re-raising required field validation error: {e}")
                raise

            logger.warning(
                f"Failed to save trace event to database for task {self.task_id}: {e}"
            )

    def _sync_save_to_database(self, event: CoreTraceEvent) -> None:
        """Synchronous database save operation (runs in thread pool)."""
        # Create database session
        db = next(get_db())
        try:
            # Save unified trace event to database
            self._save_trace_event(db, event)
        finally:
            db.close()

    def _save_trace_event(self, db: Session, event: CoreTraceEvent) -> None:
        """Save trace event in unified format to database."""
        from ...web.api.ws_trace_handlers import get_event_type_mapping

        try:
            # Map the trace event to the unified event type
            event_type_str = get_event_type_mapping(event)

            # Convert timestamp
            timestamp = _convert_float_to_datetime(event.timestamp)

            # Serialize data to ensure JSON compatibility
            data = self._serialize_data_for_json(event.data or {})

            # Create trace event record
            trace_event = DatabaseTraceEvent(
                task_id=self.task_id,
                build_id=self.build_id,  # ← 添加 build_id
                event_id=str(event.id),
                event_type=event_type_str,
                timestamp=timestamp,
                step_id=event.step_id,
                parent_event_id=str(event.parent_id) if event.parent_id else None,
                data=data,
            )

            db.add(trace_event)

            # Update tool usage statistics if this is a tool execution event
            if event_type_str == "tool_execution_end":
                tool_name = data.get("tool_name") if isinstance(data, dict) else None
                if tool_name:
                    try:
                        tool_usage: Any = (
                            db.query(ToolUsage)
                            .filter(ToolUsage.tool_name == tool_name)
                            .first()
                        )
                        if not tool_usage:
                            tool_usage = ToolUsage(
                                tool_name=tool_name,
                                usage_count=0,
                                success_count=0,
                                error_count=0,
                            )
                            db.add(tool_usage)

                        tool_usage.usage_count += 1
                        # We assume success for tool_execution_end events as errors are typically handled separately
                        # and react pattern emits this event on success
                        if isinstance(data, dict) and data.get("success", True):
                            tool_usage.success_count += 1
                        else:
                            tool_usage.error_count += 1

                        tool_usage.last_used_at = timestamp
                        logger.debug(f"Updated usage stats for tool {tool_name}")
                    except Exception as e:
                        logger.error(f"Failed to update tool usage stats: {e}")

            db.commit()

            logger.debug(
                f"Saved trace event {event.id} of type {event_type_str} to database"
            )

        except IntegrityError as e:
            db.rollback()
            error_text = str(e)
            if (
                "trace_events_task_id_fkey" in error_text
                or "ForeignKeyViolation" in error_text
            ):
                logger.debug(
                    f"Skip trace event for missing task {self.task_id}: {event.id}"
                )
                return
            logger.error(f"Failed to save trace event to database: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to save trace event to database: {e}")
            db.rollback()
            raise

    def _serialize_data_for_json(self, data: Any) -> Any:
        """Recursively serialize data to ensure JSON compatibility and clean problematic characters."""
        import json
        from datetime import datetime

        def clean_string(value: str) -> str:
            """Clean string data to remove problematic characters for PostgreSQL JSON."""
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
                # Convert Pydantic model to dict
                return serialize_value(value.model_dump())
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
                # Convert bytes to string, cleaning problematic characters
                try:
                    decoded = value.decode("utf-8")
                    return clean_string(decoded)
                except UnicodeDecodeError:
                    # If decode fails, use safe representation
                    return f"<bytes: {len(value)}>"
            else:
                return value

        try:
            # First clean and serialize the data
            cleaned_data = serialize_value(data)

            # Test if cleaned data is JSON serializable
            json.dumps(cleaned_data)
            return cleaned_data
        except (TypeError, ValueError) as e:
            # If still not serializable, log the error and return a safe fallback
            logger.warning(
                f"Failed to serialize data for JSON: {e}, data type: {type(data)}"
            )
            return {
                "_serialization_error": f"Failed to serialize {type(data).__name__}",
                "_original_type": type(data).__name__,
                "_error": str(e),
            }
