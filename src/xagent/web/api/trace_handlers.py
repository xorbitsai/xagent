"""Web-specific trace handlers for database operations."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ...config import get_checkpoint_history_limit
from ...core.agent.checkpoint import (
    CHECKPOINT_TYPE,
    READABLE_CHECKPOINT_TYPES,
    checkpoint_execution_id,
)
from ...core.agent.trace import BaseTraceHandler
from ...core.agent.trace import TraceEvent as CoreTraceEvent
from ...core.tools.adapters.vibe.connector_runtime import (
    redact_runtime_sensitive_payload,
)
from ...web.models.database import get_db
from ...web.models.task import Task
from ...web.models.task import TraceEvent as DatabaseTraceEvent
from ...web.models.tool_config import ToolUsage
from ...web.services.ops_signals import (
    CHECKPOINT_DECODE_FALLBACK,
    clear_degradation,
    register_degradation,
)
from ...web.services.trace_message_storage import (
    SQL_IN_CLAUSE_CHUNK_SIZE,
    CheckpointMessageDecodeError,
    chunks,
    decode_trace_event_data,
    encode_checkpoint_data_for_storage,
)

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
            if getattr(event, "require_persisted", False):
                logger.error(
                    "Required trace event persistence failed for task %s: %s",
                    self.task_id,
                    e,
                )
                raise

            logger.warning(
                f"Failed to save trace event to database for task {self.task_id}: {e}"
            )

    async def load_latest_checkpoint(
        self, execution_id: str
    ) -> Optional[Dict[str, Any]]:
        """Load the latest agent checkpoint persisted as a trace event."""
        try:
            return await asyncio.to_thread(
                self._sync_load_latest_checkpoint,
                execution_id,
            )
        except Exception as e:
            logger.warning(
                "Failed to load latest checkpoint for task %s execution %s: %s",
                self.task_id,
                execution_id,
                e,
            )
            return None

    def _sync_load_latest_checkpoint(
        self,
        execution_id: str,
    ) -> Optional[Dict[str, Any]]:
        db = next(get_db())
        try:
            query = db.query(DatabaseTraceEvent).filter(
                DatabaseTraceEvent.task_id == self.task_id,
                DatabaseTraceEvent.event_type == "system_update_general",
            )
            if self.build_id is None:
                query = query.filter(DatabaseTraceEvent.build_id.is_(None))
            else:
                query = query.filter(DatabaseTraceEvent.build_id == self.build_id)

            rows = (
                query.order_by(
                    DatabaseTraceEvent.timestamp.desc(),
                    DatabaseTraceEvent.id.desc(),
                )
                .limit(100)
                .all()
            )
            for row in rows:
                data: Dict[str, Any] = row.data if isinstance(row.data, dict) else {}
                if data.get("checkpoint_type") not in READABLE_CHECKPOINT_TYPES:
                    continue
                if checkpoint_execution_id(data) != str(execution_id):
                    continue
                try:
                    data = decode_trace_event_data(
                        db,
                        task_id=self.task_id,
                        data=data,
                        strict=True,
                    )
                except CheckpointMessageDecodeError as exc:
                    logger.warning(
                        "Skipping unreadable checkpoint trace event %s for task %s: %s",
                        row.event_id,
                        self.task_id,
                        exc,
                    )
                    continue
                except Exception:
                    # E.g. a transient DB error from the blob prefetch. Fall
                    # back to an older readable checkpoint instead of letting
                    # the error abort loading for the whole task. Surface the
                    # degradation on /health so a systemic decode failure is
                    # observable instead of only a per-row warning log; the
                    # signal self-clears on the next successful decode.
                    register_degradation(
                        CHECKPOINT_DECODE_FALLBACK,
                        f"task {self.task_id}: checkpoint decode failed, "
                        f"fell back past event {row.event_id}",
                    )
                    logger.warning(
                        "Skipping checkpoint trace event %s for task %s after "
                        "decode failure",
                        row.event_id,
                        self.task_id,
                        exc_info=True,
                    )
                    continue
                clear_degradation(CHECKPOINT_DECODE_FALLBACK)
                snapshot = data.get("snapshot")
                return dict(snapshot) if isinstance(snapshot, dict) else None
            return None
        finally:
            db.close()

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
            if event_type_str in {
                "tool_execution_start",
                "tool_execution_end",
                "tool_execution_failed",
            }:
                data = redact_runtime_sensitive_payload(data)
            if self._is_duplicate_user_message_turn(db, event_type_str, data):
                logger.debug(
                    "Skipping duplicate user_message turn_id=%s for task %s",
                    data.get("turn_id") if isinstance(data, dict) else None,
                    self.task_id,
                )
                return
            if (
                event_type_str == "system_update_general"
                and isinstance(data, dict)
                and data.get("checkpoint_type") == CHECKPOINT_TYPE
            ):
                data = encode_checkpoint_data_for_storage(
                    db,
                    task_id=self.task_id,
                    data=data,
                )

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

            if (
                event_type_str == "system_update_general"
                and isinstance(data, dict)
                and data.get("checkpoint_type") == CHECKPOINT_TYPE
                and self.build_id is None
            ):
                task = db.query(Task).filter(Task.id == self.task_id).first()
                if task:
                    setattr(task, "last_checkpoint_event_id", str(event.id))

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

            if (
                event_type_str == "system_update_general"
                and isinstance(data, dict)
                and data.get("checkpoint_type") == CHECKPOINT_TYPE
            ):
                self._prune_checkpoint_history(db, data)

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
                if getattr(event, "require_persisted", False):
                    logger.error(
                        "Required trace event references missing task %s: %s",
                        self.task_id,
                        event.id,
                    )
                    raise
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

    def _prune_checkpoint_history(self, db: Session, data: Dict[str, Any]) -> None:
        """Drop checkpoint rows beyond the retention limit for one execution.

        Resume only reads the most recent readable checkpoint; a few older
        rows are kept so an unreadable latest can fall back. Runs in its own
        transaction after the checkpoint commit so a prune failure can never
        take the checkpoint write down with it. Blobs are not touched: most
        stay referenced by the surviving checkpoints thanks to content
        dedup, but a blob referenced only by pruned rows (e.g. a message
        later dropped by context compaction) is orphaned until whole-task
        deletion cleans it up.
        """
        limit = get_checkpoint_history_limit()
        if limit <= 0:
            return
        execution_id = checkpoint_execution_id(data)
        if not execution_id:
            return
        try:
            build_filter = (
                DatabaseTraceEvent.build_id == self.build_id
                if self.build_id is not None
                else DatabaseTraceEvent.build_id.is_(None)
            )
            stale_rows = (
                db.query(DatabaseTraceEvent.id)
                .filter(
                    DatabaseTraceEvent.task_id == self.task_id,
                    build_filter,
                    DatabaseTraceEvent.event_type == "system_update_general",
                    DatabaseTraceEvent.data["checkpoint_type"]
                    .as_string()
                    .in_(sorted(READABLE_CHECKPOINT_TYPES)),
                    # SQL mirror of checkpoint_execution_id(): root wins,
                    # then the flat field, then the snapshot's own id, so
                    # legacy rows that only set one of them are not skipped.
                    func.coalesce(
                        func.nullif(
                            DatabaseTraceEvent.data["root_execution_id"].as_string(),
                            "",
                        ),
                        func.nullif(
                            DatabaseTraceEvent.data["execution_id"].as_string(),
                            "",
                        ),
                        DatabaseTraceEvent.data["snapshot"]["execution_id"].as_string(),
                    )
                    == execution_id,
                )
                .order_by(
                    DatabaseTraceEvent.timestamp.desc(),
                    DatabaseTraceEvent.id.desc(),
                )
                .offset(limit)
                .all()
            )
            if not stale_rows:
                return
            stale_ids = [row_id for (row_id,) in stale_rows]
            # Chunk the IN clause: a backlog from previously-disabled pruning
            # can exceed SQLite's bind-parameter limit in one statement.
            for chunk in chunks(stale_ids, SQL_IN_CLAUSE_CHUNK_SIZE):
                db.query(DatabaseTraceEvent).filter(
                    DatabaseTraceEvent.id.in_(chunk)
                ).delete(synchronize_session=False)
            db.commit()
            logger.debug(
                "Pruned %d checkpoint rows for task %s execution %s",
                len(stale_ids),
                self.task_id,
                execution_id,
            )
        except Exception:
            db.rollback()
            logger.warning(
                "Failed to prune checkpoint history for task %s",
                self.task_id,
                exc_info=True,
            )

    def _is_duplicate_user_message_turn(
        self,
        db: Session,
        event_type: str,
        data: Any,
    ) -> bool:
        if event_type != "user_message" or not isinstance(data, dict):
            return False
        turn_id = data.get("turn_id")
        if not isinstance(turn_id, str) or not turn_id:
            return False
        build_filter = (
            DatabaseTraceEvent.build_id == self.build_id
            if self.build_id is not None
            else DatabaseTraceEvent.build_id.is_(None)
        )
        return (
            db.query(DatabaseTraceEvent.id)
            .filter(
                DatabaseTraceEvent.task_id == self.task_id,
                build_filter,
                DatabaseTraceEvent.event_type == "user_message",
                DatabaseTraceEvent.data["turn_id"].as_string() == turn_id,
            )
            .first()
            is not None
        )

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
            elif callable(getattr(value, "to_dict", None)):
                return serialize_value(value.to_dict())
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
