"""WebSocket real-time communication handler"""

import asyncio
import json
import logging
import re
import shutil
import uuid
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast
from urllib.parse import unquote

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import RedirectResponse
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from ...config import (
    get_default_task_execution_mode,
    get_external_upload_dirs,
    get_uploads_dir,
)
from ...core.agent.checkpoint import CHECKPOINT_EVENT_TYPE
from ...core.agent.trace import TraceEvent, TraceHandler, trace_user_message
from ...core.execution_scope import resolve_execution_scope, turn_execution_scope
from ...core.file_ref import FILE_REF_MODEL_INSTRUCTIONS, build_file_ref
from ..auth_dependencies import get_user_from_websocket_token
from ..models.chat_message import TaskChatMessage
from ..models.database import get_db, get_session_local
from ..models.task import Task, TaskStatus
from ..models.uploaded_file import UploadedFile
from ..models.user import User

if TYPE_CHECKING:
    from ..services.task_setup_snapshot import TaskSetupSnapshot

from ...core.file_storage.keys import build_task_output_storage_key
from ..services.chat_history_service import (
    DELIVERY_COMPLETED,
    DELIVERY_DISPATCHED,
    DELIVERY_FAILED,
    DELIVERY_PENDING,
    UserMessageDeliveryClaim,
    claim_user_message_delivery,
    get_latest_waiting_question,
    inspect_user_message_delivery,
    mark_user_message_delivery,
    mark_user_message_delivery_sync,
)
from ..services.file_reference_output_service import (
    load_assistant_file_reference_records,
    reconcile_assistant_file_references,
)
from ..services.file_turn import (
    append_uploaded_files_context as _append_uploaded_files_context_to_message,
)
from ..services.file_turn import (
    bind_turn_files,
)
from ..services.file_turn import (
    build_uploaded_files_context as _build_uploaded_files_context,
)
from ..services.file_turn import (
    normalize_attachments_for_persistence as _normalize_attachments_for_persistence,
)
from ..services.file_turn import (
    normalize_filename,
    resolve_turn_file_infos,
)
from ..services.hot_path_cache import (
    cache_get,
    cache_set,
    cache_version_token,
    task_cache_ttl_seconds,
    web_task_history_key,
)
from ..services.managed_file_ref import (
    DurableStorageOperationError,
)
from ..services.task_command_transport import (
    COMMAND_FAILED,
    COMMAND_ID_PATTERN,
    MAX_COMMAND_DEFERS,
    MAX_COMMAND_FAILURES,
    ClaimedTaskCommand,
    EnqueuedTaskCommand,
    TaskCommandDeferred,
    TaskCommandKind,
    TaskCommandRejected,
    dispatch_task_command_promptly,
    enqueue_task_command,
    task_has_live_foreign_runner,
)
from ..services.task_execution_controller import (
    StaleTaskRunError,
    TaskControlState,
    apply_task_control_transition,
    task_control_snapshot,
    task_execution_controller,
)
from ..services.task_lease_service import (
    acquire_task_lease,
    mark_task_paused_if_stale,
    run_task_lease_heartbeat,
    stop_task_lease_heartbeat,
)
from ..services.uploaded_file_store import UploadedFileStore
from ..services.workforce_runtime import (
    mark_workforce_task_status,
    release_current_runner_task_lease_with_workforce_sync,
    release_task_lease_with_workforce_sync,
    sync_workforce_run_status,
)
from ..tracing import create_ephemeral_tracer
from ..user_isolated_memory import UserContext
from ..utils.db_timezone import safe_timestamp_to_unix
from .public_trace_events import is_audit_only_trace_data, normalize_public_trace_event

logger = logging.getLogger(__name__)

CHECKPOINT_EVENT_TYPE_NAME = str(CHECKPOINT_EVENT_TYPE)

_pause_accepted_task_ids: set[int] = set()


def _mark_task_pause_accepted(task_id: int) -> None:
    _pause_accepted_task_ids.add(int(task_id))


def _clear_task_pause_accepted(task_id: int) -> None:
    _pause_accepted_task_ids.discard(int(task_id))


def _is_task_pause_accepted(task_id: int) -> bool:
    return int(task_id) in _pause_accepted_task_ids


def _task_status_uses_live_control(
    status: TaskStatus,
    *,
    control_state: str | None = None,
    pause_accepted: bool = False,
) -> bool:
    """Return True when a user message should be delivered to an active run."""

    if pause_accepted or control_state == TaskControlState.PAUSE_REQUESTED.value:
        return False
    if control_state == TaskControlState.RESUME_REQUESTED.value:
        return True
    return status in {TaskStatus.WAITING_FOR_USER, TaskStatus.RUNNING}


def _task_status_payload(db: Session, task_id: int) -> dict[str, Any] | None:
    task = db.query(Task).filter(Task.id == task_id).first()
    if task is None:
        return None
    return {
        "id": task_id,
        "status": task.status.value,
    }


def _task_error_payload(
    db: Session,
    task_id: int,
    message: str,
    *,
    event_type: str = "error",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": event_type,
        "message": message,
    }
    task_payload = _task_status_payload(db, task_id)
    if task_payload is not None:
        payload["task"] = task_payload
    return payload


def _client_message_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if COMMAND_ID_PATTERN.fullmatch(normalized) is None:
        return None
    return normalized


async def _send_message_delivery(
    websocket: WebSocket,
    *,
    client_message_id: str | None,
    turn_id: str,
    accepted: bool,
    message: str | None = None,
    retry_with_new_id: bool = False,
) -> None:
    if client_message_id is None:
        return
    payload: dict[str, Any] = {
        "type": "message_accepted" if accepted else "message_rejected",
        "client_message_id": client_message_id,
        "turn_id": turn_id,
        "timestamp": datetime.now(timezone.utc).timestamp(),
    }
    if message:
        payload["message"] = message
    if retry_with_new_id:
        payload["retry_with_new_id"] = True
    await manager.send_personal_message(payload, websocket)


def _terminal_task_error_payload(
    task_id: int,
    message: str,
    *,
    event_type: str = "agent_error",
    expected_run_id: str | None = None,
) -> dict[str, Any]:
    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        current_task = db.query(Task).filter(Task.id == task_id).first()
        if (
            current_task is not None
            and expected_run_id is not None
            and current_task.run_id != expected_run_id
        ):
            logger.info(
                "Ignoring late terminal error for task %s run %s; current run is %s",
                task_id,
                expected_run_id,
                current_task.run_id,
            )
            return _task_error_payload(db, task_id, message, event_type=event_type)
        released = release_current_runner_task_lease_with_workforce_sync(
            db,
            task_id,
            status=TaskStatus.FAILED,
            expected_run_id=expected_run_id,
        )
        task = db.query(Task).filter(Task.id == task_id).first()
        if task is not None:
            if not released:
                orm_task = cast(Any, task)
                orm_task.runner_id = None
                orm_task.lease_expires_at = None
                orm_task.last_heartbeat_at = datetime.now(timezone.utc)
            mark_workforce_task_status(
                db,
                task,
                TaskStatus.FAILED,
                error_message=message,
            )
            db.commit()
            # Persist the error as an assistant message so failures that
            # happen before agent execution starts (no trace events, e.g.
            # sandbox capacity rejection) survive a history reload instead
            # of degrading to a generic "Unknown error" bubble.
            task_user_id = getattr(task, "user_id", None)
            if task_user_id is not None:
                from ..services.chat_history_service import persist_assistant_message

                try:
                    persist_assistant_message(
                        db,
                        task_id=task_id,
                        user_id=int(task_user_id),
                        content=message,
                        message_type="chat_response",
                    )
                except Exception:
                    logger.warning(
                        "Failed to persist terminal error chat message",
                        exc_info=True,
                    )
        return _task_error_payload(
            db,
            task_id,
            message,
            event_type=event_type,
        )
    except Exception:
        db.rollback()
        logger.warning("Failed to persist terminal task error", exc_info=True)
        return {
            "type": event_type,
            "message": message,
            "task": {
                "id": task_id,
                "status": TaskStatus.FAILED.value,
            },
        }
    finally:
        db.close()


def _resolve_task_llm_ids(
    task: Any, db: Session
) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Best-effort resolve internal model_id identifiers for a task."""
    from ..models.model import Model as DBModel
    from ..services.llm_utils import CoreStorage, make_normalize_model_id

    core_storage = CoreStorage(db, DBModel)

    _normalize = make_normalize_model_id(core_storage)

    return (
        _normalize(getattr(task, "model_id", None), getattr(task, "model_name", None)),
        _normalize(
            getattr(task, "small_fast_model_id", None),
            getattr(task, "small_fast_model_name", None),
        ),
        _normalize(
            getattr(task, "visual_model_id", None),
            getattr(task, "visual_model_name", None),
        ),
        _normalize(
            getattr(task, "compact_model_id", None),
            getattr(task, "compact_model_name", None),
        ),
    )


def build_unique_target_path(target_dir: Any, filename: str) -> Any:
    from pathlib import Path

    base_path = Path(target_dir) / filename
    if not base_path.exists():
        return base_path

    stem = base_path.stem
    suffix = base_path.suffix
    counter = 1
    while True:
        candidate = base_path.parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _display_message_for_user(user_message: str, has_files: bool) -> str:
    """Return the user-visible message for chat history and trace events."""
    if user_message.strip():
        return user_message
    if has_files:
        return "Uploaded file(s)"
    return user_message


def _display_file_refs_from_file_info(
    file_info_list: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return display-safe file refs without runtime paths."""
    refs: list[dict[str, Any]] = []
    for file_info in file_info_list:
        file_id = str(file_info.get("file_id") or "").strip()
        if not file_id:
            continue
        ref: dict[str, Any] = {"file_id": file_id}
        name = file_info.get("name") or file_info.get("original_name")
        if name is not None:
            ref["name"] = str(name)
        size = file_info.get("size")
        if size is not None:
            ref["size"] = size
        file_type = file_info.get("type")
        if file_type is not None:
            ref["type"] = str(file_type)
        refs.append(ref)
    return refs


def _selected_file_ids_from_task_config(task: Any) -> list[str]:
    """Return unique selected file ids stored during task creation."""
    agent_config = getattr(task, "agent_config", None)
    if not isinstance(agent_config, dict):
        return []

    raw_file_ids = agent_config.get("selected_file_ids")
    if not isinstance(raw_file_ids, list):
        return []

    file_ids = []
    seen = set()
    for raw_file_id in raw_file_ids:
        if not isinstance(raw_file_id, str):
            continue
        file_id = raw_file_id.strip()
        if file_id and file_id not in seen:
            seen.add(file_id)
            file_ids.append(file_id)
    return file_ids


def _uploaded_file_ref(file_record: UploadedFile) -> dict[str, Any]:
    """Build a websocket file ref from an authorized UploadedFile record."""
    return {
        "file_id": str(file_record.file_id),
        "name": str(file_record.filename),
        "size": int(file_record.file_size or 0),
        "type": file_record.mime_type,
    }


def _selected_file_refs_from_task(task: Any, db: Session) -> list[dict[str, Any]]:
    """Recover task-selected file refs after revalidating DB ownership/binding."""
    selected_file_ids = _selected_file_ids_from_task_config(task)
    if not selected_file_ids:
        return []

    task_id = getattr(task, "id", None)
    task_owner_id = getattr(task, "user_id", None)
    if task_id is None or task_owner_id is None:
        logger.warning("Cannot recover selected files without task id and owner id")
        return []

    task_id_int = int(task_id)
    task_owner_id_int = int(task_owner_id)
    records = (
        db.query(UploadedFile)
        .filter(
            UploadedFile.file_id.in_(selected_file_ids),
            UploadedFile.user_id == task_owner_id_int,
            or_(UploadedFile.task_id == task_id_int, UploadedFile.task_id.is_(None)),
        )
        .all()
    )
    records_by_file_id = {str(record.file_id): record for record in records}

    refs: list[dict[str, Any]] = []
    for file_id in selected_file_ids:
        record = records_by_file_id.get(file_id)
        if record is None:
            logger.warning(
                "Skipping selected file %s for task %s: not found, wrong owner, "
                "or bound to another task",
                file_id,
                task_id_int,
            )
            continue
        refs.append(_uploaded_file_ref(record))
    return refs


def _attachment_fingerprint(attachments: Any) -> str:
    """Order-independent fingerprint of a chip-shaped attachment list.

    Used by the replay dedup key so two user turns with the same typed
    text but different uploaded files don't collapse into one. We
    fingerprint on ``file_id`` only — the field is stable across the
    trace event payload and the persisted ``TaskChatMessage.attachments``
    column, and the order of items isn't meaningful for identity.
    """
    if not isinstance(attachments, list):
        return ""
    file_ids: list[str] = []
    for item in attachments:
        if not isinstance(item, dict):
            continue
        file_id = item.get("file_id")
        if isinstance(file_id, str) and file_id.strip():
            file_ids.append(file_id.strip())
    return "|".join(sorted(file_ids))


def _trace_user_message_turn_id(event_type: str, data: Any) -> str | None:
    if event_type != "user_message" or not isinstance(data, dict):
        return None
    turn_id = data.get("turn_id")
    return turn_id if isinstance(turn_id, str) and turn_id else None


def _is_duplicate_user_message_turn(
    event_type: str,
    data: Any,
    seen_turn_ids: set[str],
) -> bool:
    turn_id = _trace_user_message_turn_id(event_type, data)
    if turn_id is None:
        return False
    if turn_id in seen_turn_ids:
        return True
    seen_turn_ids.add(turn_id)
    return False


def create_stream_event(
    event_type: str,
    task_id: Union[int, str],
    data: Dict[str, Any],
    timestamp: Optional[Any] = None,
) -> Dict[str, Any]:
    """Create unified stream event format"""
    return {
        "type": "trace_event",
        "event_id": str(uuid.uuid4()),
        "event_type": event_type,
        "task_id": task_id,
        "timestamp": _stream_timestamp(timestamp),
        "data": data,
    }


def create_final_answer_stream_event(
    event_type: str,
    task_id: Union[int, str],
    data: Dict[str, Any],
    timestamp: Optional[Any] = None,
) -> Dict[str, Any]:
    """Create non-persistent final-answer UI stream events."""

    payload = dict(data)
    payload.pop("type", None)
    payload.pop("event_id", None)
    payload.pop("task_id", None)
    return {
        "type": event_type,
        "event_id": str(uuid.uuid4()),
        "task_id": task_id,
        "timestamp": _stream_timestamp(timestamp),
        **payload,
    }


def _stream_timestamp(timestamp: Optional[Any] = None) -> float:
    # Convert timestamp to Unix timestamp if it's a datetime
    if timestamp is None:
        return datetime.now(timezone.utc).timestamp()
    if isinstance(timestamp, datetime):
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.timestamp()
    if not isinstance(timestamp, (int, float)):
        return datetime.now(timezone.utc).timestamp()
    return float(timestamp)


def _persist_agent_outbound_event(task_id: int, event: Dict[str, Any]) -> None:
    """Persist agent outbound events and durable waiting prompts."""

    from ..models.task import Task as DatabaseTask
    from ..models.task import TraceEvent as DatabaseTraceEvent
    from ..services.chat_history_service import persist_assistant_message

    db_gen = get_db()
    db = next(db_gen)
    try:
        event_data = event.get("data")
        data: Dict[str, Any] = cast(
            Dict[str, Any], event_data if isinstance(event_data, dict) else {}
        )
        timestamp = event.get("timestamp")
        if isinstance(timestamp, (int, float)):
            event_time = datetime.fromtimestamp(float(timestamp), timezone.utc)
        else:
            event_time = datetime.now(timezone.utc)

        trace_event = DatabaseTraceEvent(
            task_id=task_id,
            event_id=str(data.get("event_id") or event.get("event_id") or uuid.uuid4()),
            event_type=str(
                event.get("event_type") or event.get("type") or "agent_message"
            ),
            timestamp=event_time,
            step_id=str(data["step_id"]) if data.get("step_id") else None,
            parent_event_id=None,
            data=data,
        )
        db.add(trace_event)

        if bool(data.get("expect_response")):
            task = db.query(DatabaseTask).filter(DatabaseTask.id == task_id).first()
            message = str(data.get("message") or "")
            task_user_id = _task_user_id(task) if task else None
            if task and task_user_id is not None and message:
                metadata = data.get("metadata") if isinstance(data, dict) else {}
                interactions = (
                    metadata.get("interactions")
                    if isinstance(metadata, dict)
                    and isinstance(metadata.get("interactions"), list)
                    else None
                )
                persist_assistant_message(
                    db,
                    task_id=task_id,
                    user_id=task_user_id,
                    content=message,
                    message_type="question",
                    interactions=interactions,
                )

        db.commit()
    except Exception:
        db.rollback()
        logger.exception(
            "Failed to persist agent outbound message for task %s", task_id
        )
    finally:
        db.close()


def _agent_outbound_event_type(payload: Dict[str, Any]) -> str:
    message_type = str(payload.get("message_type") or "info")
    if bool(payload.get("expect_response")) or message_type == "question":
        return "agent_message"
    return "agent_progress"


def _reconcile_streamed_final_answer(task_id: int, content: str) -> str:
    """Repair the completed stream payload using task-scoped durable FileRefs."""
    db_gen = get_db()
    db = next(db_gen)
    try:
        task = db.query(Task).filter(Task.id == int(task_id)).first()
        task_user_id = _task_user_id(task) if task is not None else None
        if task_user_id is None:
            return content
        return str(
            reconcile_assistant_file_references(
                db,
                task_id=int(task_id),
                user_id=task_user_id,
                content=content,
            )
        )
    finally:
        db.close()


def make_agent_outbound_handler(task_id: int) -> Any:
    """Create a web bridge for agent agent-to-user messages."""

    async def handle_outbound_message(payload: Dict[str, Any]) -> None:
        payload_type = str(payload.get("type") or "")
        if payload_type in {
            "final_answer_start",
            "final_answer_delta",
            "final_answer_end",
            "final_answer_error",
        }:
            if payload_type == "final_answer_end" and isinstance(
                payload.get("content"), str
            ):
                payload = dict(payload)
                payload["content"] = await asyncio.to_thread(
                    _reconcile_streamed_final_answer,
                    task_id,
                    str(payload["content"]),
                )
            await manager.broadcast_to_task(
                create_final_answer_stream_event(payload_type, task_id, dict(payload)),
                task_id,
            )
            return

        if payload.get("visible") is False:
            return

        event_type = _agent_outbound_event_type(payload)
        event = create_stream_event(
            event_type,
            task_id,
            {
                "event_id": payload.get("event_id"),
                "step_id": payload.get("step_id"),
                "execution_id": payload.get("execution_id"),
                "message": payload.get("message"),
                "message_type": payload.get("message_type", "info"),
                "expect_response": bool(payload.get("expect_response", False)),
                "display": "chat" if event_type == "agent_message" else "timeline",
                "visible": bool(payload.get("visible", True)),
                "metadata": payload.get("metadata") or {},
            },
        )
        await asyncio.to_thread(_persist_agent_outbound_event, task_id, event)
        await manager.broadcast_to_task(event, task_id)

    return handle_outbound_message


def _is_agent_checkpoint_data(data: Any) -> bool:
    """Return True for internal agent checkpoint payloads."""
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


def _is_audit_only_trace_data(data: Any) -> bool:
    """Return True for trace payloads that should stay server-side."""
    return is_audit_only_trace_data(data)


def convert_to_local_time(utc_dt: Any) -> datetime:
    """Convert UTC datetime to local time for consistent display."""
    if utc_dt.tzinfo is None:
        # If naive datetime, assume UTC
        utc_dt = utc_dt.replace(tzinfo=timezone.utc)

    # Convert to local time
    local_dt = utc_dt.astimezone()
    # Remove timezone info to avoid frontend confusion
    return local_dt.replace(tzinfo=None)  # type: ignore[no-any-return]


def _build_output_file_id(relative_path: str) -> str:
    del relative_path
    return str(uuid.uuid4())


def _resolve_output_storage_path(raw_path: str) -> Optional[tuple[Any, str]]:
    if not raw_path:
        return None

    path_candidate = Path(raw_path)
    if path_candidate.exists() and path_candidate.is_file():
        resolved = path_candidate.resolve()
    else:
        resolved = (get_uploads_dir() / raw_path.lstrip("/")).resolve()
        if not resolved.exists() or not resolved.is_file():
            return None

    uploads_root = get_uploads_dir().resolve()
    try:
        relative_path = str(resolved.relative_to(uploads_root))
    except ValueError:
        return None

    return resolved, relative_path


def _resolve_legacy_preview_storage_path(raw_path: str) -> Optional[tuple[Path, str]]:
    candidates: list[str] = []

    def _append_candidate(value: str) -> None:
        normalized = value.strip()
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    _append_candidate(raw_path)
    _append_candidate(unquote(raw_path))

    current = list(candidates)
    for candidate in current:
        for prefix in ("file:", "/preview/", "preview/", "/uploads/", "uploads/"):
            if candidate.startswith(prefix):
                _append_candidate(candidate[len(prefix) :])

    for candidate in candidates:
        resolved = _resolve_output_storage_path(candidate)
        if resolved is not None:
            resolved_path, relative_path = resolved
            return Path(resolved_path), relative_path

    for candidate in candidates:
        normalized = candidate.lstrip("/")
        if not normalized:
            continue
        glob_matches = list(get_uploads_dir().glob(f"user_*/{normalized}"))
        if glob_matches:
            resolved_path = glob_matches[0].resolve()
            relative_path = str(resolved_path.relative_to(get_uploads_dir().resolve()))
            return resolved_path, relative_path

    return None


def _infer_owner_from_relative_path(
    db: Session, relative_path: str
) -> Optional[tuple[int, Optional[int]]]:
    path_parts = Path(relative_path).parts
    if not path_parts:
        return None

    user_id: Optional[int] = None
    task_id: Optional[int] = None

    first = path_parts[0]
    remaining = path_parts[1:] if len(path_parts) > 1 else []

    if first.startswith("user_"):
        try:
            user_id = int(first.replace("user_", "", 1))
        except ValueError:
            return None
        if remaining:
            task_segment = remaining[0]
            if task_segment.startswith("web_task_"):
                try:
                    task_id = int(task_segment.replace("web_task_", "", 1))
                except ValueError:
                    task_id = None
            elif task_segment.startswith("task_"):
                try:
                    task_id = int(task_segment.replace("task_", "", 1))
                except ValueError:
                    task_id = None
        return user_id, task_id

    if first.startswith("web_task_"):
        try:
            task_id = int(first.replace("web_task_", "", 1))
        except ValueError:
            return None
    elif first.startswith("task_"):
        try:
            task_id = int(first.replace("task_", "", 1))
        except ValueError:
            return None

    if task_id is not None:
        task_row = db.query(Task).filter(Task.id == task_id).first()
        if task_row and getattr(task_row, "user_id", None) is not None:
            return int(getattr(task_row, "user_id")), task_id

    return None


def _map_link_token_to_file_id(
    token: str, path_to_file_id: Dict[str, str]
) -> Optional[str]:
    raw = token.strip()
    if not raw:
        return None

    direct_candidates = [
        raw,
        raw.lstrip("/"),
        raw.replace("%2F", "/").lstrip("/"),
        unquote(raw),
    ]

    expanded_candidates: list[str] = []
    for candidate in direct_candidates:
        if not candidate:
            continue
        if candidate not in expanded_candidates:
            expanded_candidates.append(candidate)
        if candidate.startswith("file:"):
            stripped = candidate[5:].lstrip("/")
            if stripped and stripped not in expanded_candidates:
                expanded_candidates.append(stripped)
        for prefix in ("preview/", "/preview/", "uploads/", "/uploads/"):
            if candidate.startswith(prefix):
                stripped = candidate[len(prefix) :].lstrip("/")
                if stripped and stripped not in expanded_candidates:
                    expanded_candidates.append(stripped)

    for candidate in expanded_candidates:
        mapped = path_to_file_id.get(candidate)
        if mapped:
            return mapped
    return None


def _rewrite_file_links_to_file_id(
    output_text: Any, path_to_file_id: Dict[str, str]
) -> Any:
    if not isinstance(output_text, str) or not output_text:
        return output_text

    def replace_link(match: re.Match[str]) -> str:
        token = match.group(1).strip()
        mapped_file_id = _map_link_token_to_file_id(token, path_to_file_id)
        if mapped_file_id:
            return f"(file:{mapped_file_id})"
        return match.group(0)

    def replace_legacy_link(match: re.Match[str]) -> str:
        token = match.group(1).strip()
        mapped_file_id = _map_link_token_to_file_id(token, path_to_file_id)
        if mapped_file_id:
            return f"(file:{mapped_file_id})"
        return match.group(0)

    rewritten_output = re.sub(r"\(file:([^)]+)\)", replace_link, output_text)
    rewritten_output = re.sub(
        r"\(((?:/?preview|/?uploads)/[^)\s]+)\)",
        replace_legacy_link,
        rewritten_output,
    )
    rewritten_output = re.sub(
        r"\((/?(?:input|output|temp)/[^)\s]+|/?(?:user_\d+/)?(?:web_task_\d+|task_\d+)/(?:input|output|temp)/[^)\s]+)\)",
        replace_legacy_link,
        rewritten_output,
    )
    return rewritten_output


def _add_file_link_aliases(
    path_to_file_id: Dict[str, str], relative_path: str, file_id: str
) -> None:
    normalized_relative_path = relative_path.lstrip("/")
    if not normalized_relative_path:
        return

    for prefix in ("", "/", "preview/", "/preview/", "uploads/", "/uploads/"):
        _set_file_link_alias(
            path_to_file_id, f"{prefix}{normalized_relative_path}", file_id
        )

    basename = Path(normalized_relative_path).name
    if basename and basename != normalized_relative_path:
        _set_file_link_alias(path_to_file_id, basename, file_id)

    parts = Path(normalized_relative_path).parts
    task_local_parts: tuple[str, ...] = ()
    if (
        len(parts) >= 3
        and parts[0].startswith("user_")
        and (parts[1].startswith("web_task_") or parts[1].startswith("task_"))
    ):
        without_user = "/".join(parts[1:])
        if without_user:
            _add_file_link_aliases(path_to_file_id, without_user, file_id)
        task_local_parts = parts[2:]
    elif len(parts) >= 2 and (
        parts[0].startswith("web_task_") or parts[0].startswith("task_")
    ):
        task_local_parts = parts[1:]

    if task_local_parts and task_local_parts[0] in {"input", "output", "temp"}:
        task_local_path = "/".join(task_local_parts)
        _set_file_link_alias(path_to_file_id, task_local_path, file_id)
        _set_file_link_alias(path_to_file_id, f"/{task_local_path}", file_id)


def _set_file_link_alias(
    path_to_file_id: Dict[str, str], alias: str, file_id: str
) -> None:
    existing_file_id = path_to_file_id.get(alias)
    if existing_file_id is None or existing_file_id == file_id:
        path_to_file_id[alias] = file_id
        return

    # A bare ``file:report.txt`` link is ambiguous when multiple outputs can
    # claim the same alias. Keep scoped aliases but disable ambiguous rewriting
    # so we never point the user at the wrong artifact. The empty string is a
    # sticky sentinel for this alias: once ambiguous, later registrations cannot
    # reclaim it for a single file.
    path_to_file_id[alias] = ""


def _uploaded_file_record_in_task_scope(
    file_record: Any, task_id: int, task_user_id: int
) -> bool:
    try:
        record_user_id = int(getattr(file_record, "user_id"))
    except (TypeError, ValueError):
        return False

    if record_user_id != int(task_user_id):
        return False

    record_task_id = getattr(file_record, "task_id", None)
    if record_task_id is None:
        return True

    try:
        return int(record_task_id) == int(task_id)
    except (TypeError, ValueError):
        return False


def _scope_segments_for_task(task_id: Any) -> tuple[str, ...]:
    """workspace_segments of the task's resolved ExecutionScope ((),
    when unscoped) — for storage-key composition outside the turn context.

    A None ``task_id`` (e.g. the legacy-preview backfill, whose owner
    inference may find a user but no task) means there is no task identity
    to resolve a scope from — unscoped, never the string ``"None"``.
    """
    if task_id is None:
        return ()
    scope = resolve_execution_scope(task_id)
    return scope.workspace_segments if scope is not None else ()


def _output_path_in_current_task_scope(
    relative_path: str, task_id: int, task_user_id: int
) -> bool:
    parts = Path(relative_path.lstrip("/")).parts
    task_dirs = {f"web_task_{task_id}", f"task_{task_id}"}

    if len(parts) >= 4 and parts[0] == f"user_{task_user_id}":
        # Scoped workspaces insert ExecutionScope.workspace_segments between
        # the user root and the task dir
        # (user_{id}/{segment}.../web_task_{id}/output/...); accept the task
        # dir at any depth after the user root so scoped outputs are not
        # misclassified as foreign. Keep scanning past a component that
        # merely LOOKS like the task dir — a scope segment may legitimately
        # be named like one (the segment charset allows it), and an early
        # verdict on it would reject the real task dir further down.
        for index in range(1, len(parts) - 2):
            if parts[index] in task_dirs and parts[index + 1] == "output":
                return True

    return len(parts) >= 3 and parts[0] in task_dirs and parts[1] == "output"


def _normalize_workspace_relative_path(relative_path: str) -> str:
    normalized = relative_path.strip().lstrip("/")
    path_parts = [part for part in Path(normalized).parts if part not in ("", ".")]
    if not path_parts or ".." in path_parts:
        return Path(normalized).name or "output"

    if path_parts[0].startswith("user_"):
        path_parts = path_parts[1:]

    if path_parts and (
        path_parts[0].startswith("web_task_") or path_parts[0].startswith("task_")
    ):
        path_parts = path_parts[1:]

    return "/".join(path_parts) if path_parts else "output"


def _workspace_category_from_relative_path(relative_path: str) -> str:
    path_parts = Path(relative_path).parts
    return path_parts[0] if path_parts else "output"


def _normalize_file_outputs(
    db: Session,
    task_id: int,
    task_user_id: int,
    file_outputs: Any,
) -> tuple[list[Dict[str, Any]], Dict[str, str]]:
    from ..models.uploaded_file import UploadedFile

    if isinstance(file_outputs, str):
        file_outputs = [file_outputs] if file_outputs.strip() else []
    if not isinstance(file_outputs, list):
        return [], {}

    normalized_outputs: list[Dict[str, Any]] = []
    path_to_file_id: Dict[str, str] = {}
    changed = False
    # Resolved once per normalization pass: per-storage-key resolution would
    # re-query the snapshot loader / resolver once per output file (N+1 with
    # the output-file count — workforce runs can emit dozens).
    scope_segments = _scope_segments_for_task(task_id)

    def add_normalized_output(
        file_record: UploadedFile,
        fallback_filename: str,
        raw_paths: list[str],
    ) -> None:
        final_file_id = str(file_record.file_id)
        final_filename = fallback_filename or str(file_record.filename)

        normalized_outputs.append(
            build_file_ref(
                file_id=final_file_id,
                filename=final_filename,
                mime_type=getattr(file_record, "mime_type", None),
                size=getattr(file_record, "file_size", None),
            )
        )

        for raw_path in raw_paths:
            stripped = raw_path.strip()
            if stripped:
                _set_file_link_alias(path_to_file_id, stripped, final_file_id)
                _set_file_link_alias(
                    path_to_file_id, stripped.lstrip("/"), final_file_id
                )

        storage_path = getattr(file_record, "storage_path", None)
        if storage_path:
            _set_file_link_alias(path_to_file_id, str(storage_path), final_file_id)

        workspace_relative_path = getattr(file_record, "workspace_relative_path", None)
        if isinstance(workspace_relative_path, str) and workspace_relative_path.strip():
            _add_file_link_aliases(
                path_to_file_id, workspace_relative_path, final_file_id
            )

    for item in file_outputs:
        item_file_id = ""
        item_filename = ""
        item_relative_path = ""
        raw_paths: list[str] = []

        if isinstance(item, str):
            raw_paths = [item]
        elif isinstance(item, dict):
            if isinstance(item.get("file_id"), str):
                item_file_id = str(item.get("file_id"))
            if isinstance(item.get("filename"), str):
                item_filename = str(item.get("filename"))
            for key in ("file_path", "download_path", "relative_path", "path"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    raw_paths.append(value)
                    if key == "relative_path":
                        item_relative_path = value
        else:
            continue

        resolved_info = None
        for raw_path in raw_paths:
            resolved_info = _resolve_output_storage_path(raw_path)
            if resolved_info is not None:
                break

        if resolved_info is None:
            if item_file_id:
                file_record = (
                    db.query(UploadedFile)
                    .filter(
                        UploadedFile.file_id == item_file_id,
                        UploadedFile.user_id == task_user_id,
                        or_(
                            UploadedFile.task_id == task_id,
                            UploadedFile.task_id.is_(None),
                        ),
                    )
                    .first()
                )
                if file_record is None:
                    logger.warning(
                        "Skipping file output outside task/user scope: %s",
                        item_file_id,
                    )
                    continue
                normalized_outputs.append(
                    build_file_ref(
                        file_id=str(file_record.file_id),
                        filename=item_filename or str(file_record.filename),
                        mime_type=getattr(file_record, "mime_type", None),
                        size=getattr(file_record, "file_size", None),
                    )
                )
            continue

        resolved_path, relative_path = resolved_info
        normalized_relative_path = relative_path.lstrip("/")
        file_record = (
            db.query(UploadedFile)
            .filter(UploadedFile.storage_path == str(resolved_path))
            .first()
        )
        if file_record is not None and not _uploaded_file_record_in_task_scope(
            file_record, task_id, task_user_id
        ):
            logger.warning(
                "Skipping file output record outside task/user scope: %s",
                getattr(file_record, "file_id", str(resolved_path)),
            )
            continue

        if file_record is not None and not _output_path_in_current_task_scope(
            normalized_relative_path, task_id, task_user_id
        ):
            if getattr(file_record, "workspace_category", None) != "output":
                logger.warning(
                    "Skipping registered file output outside output category: %s",
                    getattr(file_record, "file_id", str(resolved_path)),
                )
                continue
            add_normalized_output(file_record, item_filename, raw_paths)
            continue

        if not _output_path_in_current_task_scope(
            normalized_relative_path, task_id, task_user_id
        ):
            logger.warning(
                "Skipping file output outside current task output scope: %s",
                relative_path,
            )
            continue

        workspace_relative_path = _normalize_workspace_relative_path(
            item_relative_path or normalized_relative_path
        )
        workspace_category = _workspace_category_from_relative_path(
            workspace_relative_path
        )
        expected_file_id = item_file_id or _build_output_file_id(
            workspace_relative_path
        )

        if file_record is None and item_file_id:
            file_record = (
                db.query(UploadedFile)
                .filter(
                    UploadedFile.file_id == item_file_id,
                    UploadedFile.user_id == task_user_id,
                    or_(
                        UploadedFile.task_id == task_id, UploadedFile.task_id.is_(None)
                    ),
                )
                .first()
            )

        if file_record is None:
            try:
                file_record = UploadedFileStore(db).create_from_local_path(
                    local_path=resolved_path,
                    user_id=task_user_id,
                    file_id=expected_file_id,
                    task_id=task_id,
                    filename=item_filename or resolved_path.name,
                    mime_type=None,
                    storage_key=build_task_output_storage_key(
                        task_user_id,
                        task_id,
                        expected_file_id,
                        workspace_relative_path,
                        scope_segments=scope_segments,
                    ),
                    workspace_relative_path=workspace_relative_path,
                    workspace_category=workspace_category,
                )
                db.flush()
                changed = True
            except DurableStorageOperationError:
                db.rollback()
                raise

        else:
            try:
                file_record = UploadedFileStore(db).upsert_by_storage_path(
                    user_id=task_user_id,
                    filename=item_filename or resolved_path.name,
                    storage_path=resolved_path,
                    mime_type=None,
                    file_size=resolved_path.stat().st_size,
                    storage_key=build_task_output_storage_key(
                        task_user_id,
                        task_id,
                        str(file_record.file_id),
                        workspace_relative_path,
                        scope_segments=scope_segments,
                    ),
                    task_id=task_id,
                    workspace_relative_path=workspace_relative_path,
                    workspace_category=workspace_category,
                )
                changed = True
            except DurableStorageOperationError:
                db.rollback()
                raise

        if item_file_id:
            path_to_file_id[item_file_id] = str(file_record.file_id)
        add_normalized_output(file_record, item_filename, raw_paths)
        _add_file_link_aliases(
            path_to_file_id, normalized_relative_path, str(file_record.file_id)
        )

        if workspace_relative_path != normalized_relative_path:
            final_file_id = str(file_record.file_id)
            _add_file_link_aliases(
                path_to_file_id,
                workspace_relative_path,
                final_file_id,
            )

    if changed:
        db.commit()

    return normalized_outputs, path_to_file_id


def _normalize_task_file_outputs(
    db: Session,
    task: Any,
    file_outputs: Any,
    *,
    task_id: Optional[int] = None,
    task_user_id: Optional[int] = None,
) -> tuple[list[Dict[str, Any]], Dict[str, str]]:
    """Resolve and persist ``file_outputs`` produced by an agent run.

    Two callsite shapes:
      1. WS / legacy paths still hold the ORM ``task`` row in-scope —
         pass it as ``task`` and the user_id / task_id come from there.
      2. Snapshot path (``execute_task_background`` with off-loop
         loader) sets ``task=None`` to avoid ORM session crossings,
         and supplies ``task_id`` + ``task_user_id`` directly. Without
         this overload the persistence step silently no-ops because
         ``_task_user_id(None)`` returns ``None``.
    """
    resolved_user_id: Optional[int]
    resolved_task_id: Optional[int]
    if task is not None:
        resolved_user_id = _task_user_id(task)
        resolved_task_id = int(cast(Any, task.id))
    else:
        resolved_user_id = task_user_id
        resolved_task_id = task_id

    if resolved_user_id is None or resolved_task_id is None:
        return [], {}

    return _normalize_file_outputs(
        db,
        task_id=resolved_task_id,
        task_user_id=resolved_user_id,
        file_outputs=file_outputs,
    )


def _rewrite_links_in_payload(payload: Any, path_to_file_id: Dict[str, str]) -> Any:
    if isinstance(payload, str):
        return _rewrite_file_links_to_file_id(payload, path_to_file_id)
    if isinstance(payload, list):
        return [_rewrite_links_in_payload(item, path_to_file_id) for item in payload]
    if isinstance(payload, dict):
        return {
            key: _rewrite_links_in_payload(value, path_to_file_id)
            for key, value in payload.items()
        }
    return payload


def _task_user_id(task: Any) -> int | None:
    user_id = getattr(task, "user_id", None)
    if user_id is None:
        return None
    return int(cast(Any, user_id))


def _task_run_id(task: Any) -> str | None:
    run_id = getattr(task, "run_id", None)
    return str(run_id) if run_id is not None else None


def _task_control_state_value(task: Any) -> str | None:
    control_state = getattr(task, "control_state", None)
    return str(control_state) if control_state is not None else None


async def execute_task_background(
    task_id: int,
    user_message: str,
    context: Dict[str, Any] | None,
    agent_manager: Any,
    task_owner_user_id: int | None,
    before_message_id: int | None = None,
    llm_user_message: Optional[str] = None,
    task_setup_snapshot: Optional["TaskSetupSnapshot"] = None,
    expected_run_id: str | None = None,
) -> None:
    """Execute task in background without blocking WebSocket message loop.

    ``task_setup_snapshot`` is the off-loop snapshot loaded by
    ``_schedule_bg._runner``. When provided, the Task SELECT is
    skipped (saves a synchronous DB read measured at 3.33s on the
    main event loop under contention, issue #427) and downstream
    consumers pull task fields from the snapshot. The User SELECT is
    kept because ``get_user_tool_overrides`` is a hook
    (``Callable[[Session, Any], dict]``, ``services/tool_credentials.py``)
    that may read arbitrary ORM fields off the user object;
    constructing a primitive shim there would be a quiet BC break.

    WS callers (and any caller that has not yet adopted the snapshot
    plumbing) pass ``None`` and the legacy Task SELECT runs as
    before.
    """
    from ..models.database import get_db
    from ..models.task import Task, TaskStatus
    from ..models.user import User
    from ..services.chat_history_service import (
        load_task_transcript,
        persist_assistant_message,
    )
    from ..services.task_execution_context_service import (
        load_task_execution_recovery_state,
    )

    # Wait for previous background task to complete
    await background_task_manager.wait_for_previous(task_id)

    db_gen = get_db()
    try:
        db = next(db_gen)
        context_dict = context if isinstance(context, dict) else {}
        logger.info(f"Background task execution started for task {task_id}")

        task_user_id: Optional[int]
        if task_setup_snapshot is not None:
            # Snapshot path: skip the Task SELECT.
            task_user_id = task_setup_snapshot.task.user_id
            task = None
        else:
            task = db.query(Task).filter(Task.id == task_id).first()
            if task is None:
                raise ValueError(f"Task {task_id} not found")
            task_user_id = _task_user_id(task)

        # The task OWNER (from snapshot / DB) is the runtime identity. A passed
        # ``task_owner_user_id`` must equal it -- it may never override the
        # owner, or the task would run as the wrong user (e.g. an admin acting
        # on someone else's task would get the admin's models / tools / OAuth).
        # All callers pass the owner; a mismatch is a programming error, so
        # reject it rather than silently continue.
        if (
            task_owner_user_id is not None
            and task_user_id is not None
            and task_owner_user_id != task_user_id
        ):
            raise ValueError(
                f"execute_task_background: passed task_owner_user_id "
                f"{task_owner_user_id} does not match task {task_id} owner "
                f"{task_user_id}; refusing to run as the wrong user"
            )
        effective_user_id = (
            task_user_id if task_user_id is not None else task_owner_user_id
        )
        user = (
            db.query(User).filter(User.id == effective_user_id).first()
            if effective_user_id is not None
            else None
        )

        # The execution scope resolves per turn alongside the acting user, so
        # a resumed/restarted task re-derives the same scope from the resolver.
        with UserContext(effective_user_id), turn_execution_scope(task_id):
            # Get agent service. ``effective_user_id`` is the task owner
            # (authoritative above); pass it as the runtime identity so the
            # agent's models / tools resolve as the owner, not any acting admin.
            agent_service = await agent_manager.get_agent_for_task(
                task_id,
                db,
                user=user,
                task_setup_snapshot=task_setup_snapshot,
                task_owner_user_id=effective_user_id,
                connector_runtime_turn_id=context_dict.get("turn_id")
                if isinstance(context_dict.get("turn_id"), str)
                else None,
            )
            if hasattr(agent_service, "set_outbound_message_handler"):
                agent_service.set_outbound_message_handler(
                    make_agent_outbound_handler(task_id)
                )
            if before_message_id is not None:
                conversation_history = load_task_transcript(
                    db,
                    task_id,
                    before_message_id=before_message_id,
                )
                agent_service.set_conversation_history(conversation_history)
            recovery_state = await load_task_execution_recovery_state(db, task_id)
            execution_context_messages = recovery_state.get("messages", [])
            agent_service.set_execution_context_messages(execution_context_messages)
            agent_service.set_recovered_skill_context(
                recovery_state.get("skill_context")
            )
            _register_uploaded_files_for_agent(
                agent_service,
                context_dict.get("file_info", []),
                db,
            )

            # Execute the next turn under the same task/thread id.
            actual_task_id = str(task_id)
            task_for_agent = llm_user_message or user_message
            result = await agent_manager.execute_task(
                agent_service=agent_service,
                task=task_for_agent,
                context=context,
                task_id=actual_task_id,
                tracking_task_id=str(task_id),
                db_session=db,
                manage_task_lease=False,
            )

        normalized_outputs, path_to_file_id = _normalize_task_file_outputs(
            db,
            task,
            result.get("file_outputs", []),
            task_id=int(task_id) if task is None else None,
            task_user_id=task_user_id if task is None else None,
        )
        if normalized_outputs:
            result["file_outputs"] = normalized_outputs

        # Get AI response. A failed turn has no assistant reply, so it must not
        # inherit the "Task completed" success sentinel: the frontend reads
        # `output` as the failure reason, so the sentinel would render (and
        # persist) a misleading "Task completed" bubble on a failed task. Fall
        # back to the sentinel only when the turn actually succeeded.
        default_response = "Task completed" if result.get("success", False) else ""
        chat_response = result.get("chat_response")
        if isinstance(chat_response, dict):
            ai_response = chat_response.get("message") or result.get(
                "output", default_response
            )
        else:
            ai_response = result.get("output", default_response)

        # Rewrite file links to file_id
        ai_response = _rewrite_file_links_to_file_id(
            ai_response,
            path_to_file_id,
        )
        if effective_user_id is not None:
            ai_response = reconcile_assistant_file_references(
                db,
                task_id=int(task_id),
                user_id=int(effective_user_id),
                content=ai_response,
            )
            if isinstance(chat_response, dict) and chat_response.get("message"):
                chat_response = {**chat_response, "message": ai_response}

        # Task execution result is logged by ConsoleTraceHandler, no need for duplicate logs

        db_new_gen = get_db()
        final_control_snapshot = None
        try:
            db_new = next(db_new_gen)
            waiting_for_control = False
            # ``task`` is ``None`` on the snapshot path; pull the
            # pre-run status from the snapshot in that case. The
            # ``task_updated`` query just below normally overwrites
            # this with the post-run value, but we still need a
            # sensible default for the rare case where the row went
            # missing between snapshot load and finalize.
            if task is not None:
                final_task_status = task.status.value
            elif task_setup_snapshot is not None:
                final_task_status = task_setup_snapshot.task.status.value
            else:
                final_task_status = TaskStatus.PENDING.value
            task_updated = db_new.query(Task).filter(Task.id == task_id).first()
            if task_updated:
                if (
                    expected_run_id is not None
                    and task_updated.run_id != expected_run_id
                ):
                    logger.info(
                        "Ignoring late task result for task %s run %s; "
                        "current run is %s",
                        task_id,
                        expected_run_id,
                        task_updated.run_id,
                    )
                    return
                # Caller is responsible for the lease lifecycle (acquire +
                # release); this function only writes ``status``. The
                # orchestrator's ``_schedule_bg`` wraps the call in
                # acquire/release; chat.py and WS continuation paths
                # acquire and release the lease directly themselves.
                #
                # Previously this branch called
                # ``release_current_runner_task_lease(status=...)``, which
                # bundled status update with lease release in one UPDATE
                # filtered on ``runner_id == get_runner_id()``. That hid a
                # bug for callers that never acquired the lease: the
                # filter didn't match, so status was silently never
                # written either (a quiet "stuck RUNNING" outcome).
                task_agent_config: dict[str, Any] = (
                    task_updated.agent_config
                    if isinstance(task_updated.agent_config, dict)
                    else {}
                )
                # A2A cancellation is durable even when the cancelled coroutine
                # ignores cancellation and produces a result after the timeout.
                if task_agent_config.get("a2a_state") == "TASK_STATE_CANCELED":
                    waiting_for_control = True
                    logger.info(
                        "Task %s was canceled while execution was in flight; "
                        "ignoring the late result",
                        task_id,
                    )
                elif result.get("status") == "waiting_for_user":
                    next_control_state = (
                        TaskControlState.RESUME_REQUESTED
                        if task_updated.control_state
                        == TaskControlState.RESUME_REQUESTED.value
                        else TaskControlState.WAITING_FOR_USER
                    )
                    final_control_snapshot = apply_task_control_transition(
                        task_updated,
                        next_control_state,
                        status=TaskStatus.WAITING_FOR_USER,
                        expected_run_id=expected_run_id,
                    )
                    sync_workforce_run_status(db_new, task_updated, task_updated.status)
                    db_new.commit()
                    waiting_for_control = True
                    logger.info(
                        f"Updated task {task_id} status to WAITING_FOR_USER for v2 control state"
                    )
                elif result.get("status") == "interrupted":
                    next_control_state = (
                        TaskControlState.RESUME_REQUESTED
                        if task_updated.control_state
                        == TaskControlState.RESUME_REQUESTED.value
                        else TaskControlState.PAUSED
                    )
                    final_control_snapshot = apply_task_control_transition(
                        task_updated,
                        next_control_state,
                        status=TaskStatus.PAUSED,
                        expected_run_id=expected_run_id,
                    )
                    sync_workforce_run_status(db_new, task_updated, task_updated.status)
                    db_new.commit()
                    waiting_for_control = True
                    logger.info(
                        f"Updated task {task_id} status to PAUSED for v2 interrupt state"
                    )
                elif task_updated.status not in {
                    TaskStatus.PAUSED,
                    TaskStatus.WAITING_FOR_USER,
                }:
                    final_control_state = (
                        TaskControlState.COMPLETED
                        if result.get("success", False)
                        else TaskControlState.FAILED
                    )
                    final_status = (
                        TaskStatus.COMPLETED
                        if result.get("success", False)
                        else TaskStatus.FAILED
                    )
                    final_control_snapshot = apply_task_control_transition(
                        task_updated,
                        final_control_state,
                        status=final_status,
                        expected_run_id=expected_run_id,
                    )
                    sync_workforce_run_status(db_new, task_updated, task_updated.status)
                    # Do NOT commit the terminal status here. Leave it
                    # pending so the assistant-message persistence below
                    # commits it atomically: the task is marked terminal
                    # only once the turn is durably complete. If that write
                    # fails, the status stays RUNNING and the outer except
                    # surfaces a real failure -- instead of leaving a
                    # COMPLETED row with no assistant message. Control
                    # statuses (PAUSED / WAITING_FOR_USER) above commit
                    # themselves; they have no assistant message to persist.
                    logger.info(
                        f"Task {task_id} marked {task_updated.status.value} "
                        "(pending commit with assistant message)"
                    )
                else:
                    waiting_for_control = True
                    logger.info(
                        f"Task {task_id} is paused, not updating status to {result.get('success')}"
                    )
                final_task_status = task_updated.status.value

                if not waiting_for_control:
                    # ``persist_assistant_message`` requires a real
                    # user_id (FK into ``users.id``). Prefer
                    # ``effective_user_id`` -- it already folded the
                    # function-parameter ``user_id`` and the
                    # snapshot/legacy ``task_user_id`` together earlier.
                    # If both were None we cannot persist; fail loudly
                    # rather than writing an orphan row with user_id=0.
                    if effective_user_id is None:
                        raise ValueError(
                            f"Task {task_id}: cannot persist assistant "
                            "message without a resolved user_id "
                            "(both function param and task.user_id were None)"
                        )
                    persist_assistant_message(
                        db_new,
                        task_id=task_id,
                        user_id=int(effective_user_id),
                        content=str(ai_response),
                        message_type="chat_response"
                        if isinstance(chat_response, dict)
                        else "final_answer",
                        interactions=chat_response.get("interactions")
                        if isinstance(chat_response, dict)
                        else None,
                        content_is_reconciled=True,
                    )
                    # Commit the pending terminal status. ``persist_assistant_message``
                    # commits internally when it writes a row, but it
                    # early-returns WITHOUT committing when the assistant
                    # content is empty (a valid empty-reply turn). This
                    # explicit commit lands the terminal status in that
                    # case too, so an empty successful turn stays COMPLETED
                    # rather than being left RUNNING (and later flipped to
                    # FAILED by finish_turn). If persistence raised, control
                    # never reaches here -- the status stays uncommitted and
                    # the outer except surfaces a real failure.
                    db_new.commit()

            # Materialize broadcast metadata into primitives BEFORE the
            # ``finally`` block closes ``db_new``. ``task_updated`` is
            # bound to that session; accessing its attributes after
            # close raises ``DetachedInstanceError``. Title /
            # description / execution_mode / updated_at don't change
            # during a turn, so this snapshot is consistent with what
            # the legacy code emitted.
            if task_updated is not None:
                broadcast_meta = {
                    "id": int(task_updated.id),
                    "title": task_updated.title,
                    "description": task_updated.description,
                    "execution_mode": getattr(task_updated, "execution_mode", None),
                    "updated_at": task_updated.updated_at,
                }
            else:
                # Task row deleted between turn start and finalize.
                # Broadcasts below will emit nulls for title /
                # description; log here so the gap is visible in
                # incident triage instead of having to reconstruct it
                # from the silent-null payload.
                logger.warning(
                    "Task %s row missing at finalize; broadcasting partial "
                    "task metadata (title/description/execution_mode null)",
                    task_id,
                )
                broadcast_meta = {
                    "id": task_id,
                    "title": None,
                    "description": None,
                    "execution_mode": None,
                    "updated_at": None,
                }

            # Snapshot agent metadata before the request-scoped ORM
            # session closes. Snapshot callers intentionally set
            # ``task=None``, so we fall back to the off-loop snapshot.
            if task is not None:
                broadcast_agent_meta = {
                    "agent_id": task.agent_id,
                    "agent_name": task.agent.name if task.agent else None,
                    "agent_logo_url": task.agent.logo_url if task.agent else None,
                }
            elif task_setup_snapshot is not None:
                broadcast_agent_meta = {
                    "agent_id": task_setup_snapshot.task.agent_id,
                    "agent_name": (
                        task_setup_snapshot.agent.name
                        if task_setup_snapshot.agent is not None
                        else None
                    ),
                    "agent_logo_url": None,
                }
            else:
                broadcast_agent_meta = {
                    "agent_id": None,
                    "agent_name": None,
                    "agent_logo_url": None,
                }
        finally:
            try:
                next(db_new_gen)
            except StopIteration:
                pass

        # Note: trace_task_completion is handled by the agent execution logic (e.g., dag_plan_execute.py)

        control_event_state = (
            final_control_snapshot.as_dict()
            if final_control_snapshot is not None
            else {}
        )

        if waiting_for_control:
            await manager.broadcast_to_task(
                create_stream_event(
                    "task_info",
                    task_id,
                    {
                        "id": broadcast_meta["id"],
                        "title": broadcast_meta["title"],
                        "description": broadcast_meta["description"],
                        "status": final_task_status,
                        "execution_mode": broadcast_meta["execution_mode"],
                        "agent_id": broadcast_agent_meta["agent_id"],
                        "agent_name": broadcast_agent_meta["agent_name"],
                        "agent_logo_url": broadcast_agent_meta["agent_logo_url"],
                        **control_event_state,
                    },
                    broadcast_meta["updated_at"] or None,
                ),
                task_id,
            )
            logger.info(f"Background task {task_id} paused for v2 control")
            return

        # Send task completion event (includes agent response info)
        await manager.broadcast_to_task(
            {
                "type": "task_completed",
                "task": {
                    "id": broadcast_meta["id"],
                    "title": broadcast_meta["title"],
                    "status": final_task_status,
                    "description": broadcast_meta["description"],
                },
                "result": ai_response,
                "output": ai_response,
                "file_outputs": normalized_outputs,
                "success": result.get("success", False),
                # Machine-readable failure classification (e.g. "quota_exceeded")
                # plus its structured details, so the client can localise and
                # branch instead of parsing the message. Absent for normal turns.
                "error_code": result.get("error_code"),
                "error_details": result.get("error_details"),
                **control_event_state,
                "chat_response": chat_response
                if isinstance(chat_response, dict)
                else None,
                "timestamp": datetime.now(timezone.utc).timestamp(),
            },
            task_id,
        )
        logger.info(f"Background task {task_id} execution completed")

    except Exception as e:
        # The outer try also spans the post-terminal steps -- assistant
        # message persistence and the completion / paused broadcasts --
        # that run *after* the task status was already committed terminal
        # (COMPLETED above). ``_terminal_task_error_payload`` writes FAILED
        # + the real error_message unconditionally, so gate it on the
        # task's current status: only a task still RUNNING is a genuine
        # execution failure. Otherwise a failed post-completion broadcast
        # would rewrite an already-COMPLETED task as FAILED and store the
        # broadcast error as the task's failure cause.
        status_db = get_session_local()()
        try:
            current = status_db.query(Task).filter(Task.id == task_id).first()
            still_running = current is not None and (
                current.status == TaskStatus.RUNNING
                and (expected_run_id is None or current.run_id == expected_run_id)
            )
        finally:
            status_db.close()

        if not still_running:
            # Terminal state already committed; the exception came from a
            # best-effort post-completion step. Observe it without touching
            # the row or emitting a contradictory task_error. ``finish_turn``
            # still reconciles the terminal fields afterward.
            logger.warning(
                f"Background task {task_id} post-terminal step failed; "
                f"task state left unchanged: {e}",
                exc_info=True,
            )
        else:
            logger.error(
                f"Background task {task_id} execution failed: {e}", exc_info=True
            )
            # Genuine failure: _terminal_task_error_payload persists FAILED
            # + the real error_message and builds the notification payload.
            try:
                message = str(e)
                await manager.broadcast_to_task(
                    {
                        **_terminal_task_error_payload(
                            task_id,
                            message,
                            event_type="task_error",
                            expected_run_id=expected_run_id,
                        ),
                        "task_id": task_id,
                        "error": message,
                        "timestamp": datetime.now(timezone.utc).timestamp(),
                    },
                    task_id,
                )
            except Exception as broadcast_error:
                logger.error(f"Failed to send error notification: {broadcast_error}")
    except asyncio.CancelledError:
        logger.info(f"Background task {task_id} cancelled")
        raise
    finally:
        # Clean up background task record
        _clear_task_pause_accepted(task_id)
        background_task_manager.cleanup_task(task_id)
        try:
            next(db_gen)
        except StopIteration:
            pass


def _latest_result_user_turn_id(result: Dict[str, Any]) -> str | None:
    agent_result = result.get("agent_result")
    if not isinstance(agent_result, dict):
        return None
    context = agent_result.get("context")
    messages = (
        context.get("messages")
        if isinstance(context, dict)
        else getattr(context, "messages", None)
    )
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        role = (
            message.get("role")
            if isinstance(message, dict)
            else getattr(message, "role", None)
        )
        if role != "user":
            continue
        metadata = (
            message.get("metadata")
            if isinstance(message, dict)
            else getattr(message, "metadata", None)
        )
        if isinstance(metadata, dict):
            turn_id = metadata.get("turn_id")
            if isinstance(turn_id, str) and turn_id:
                return turn_id
    return None


async def execute_resume_background(
    task_id: int,
    agent_service: Any,
    task_owner_user_id: int | None,
    previous_task: Optional[asyncio.Task] = None,
    pending_user_message: Optional[Dict[str, Any]] = None,
    delivery_turn_id: str | None = None,
    delivery_already_dispatched: bool = False,
    delivery_websocket: WebSocket | None = None,
    delivery_client_message_id: str | None = None,
    expected_run_id: str | None = None,
) -> None:
    """Resume an agent execution after an interrupt/user-message checkpoint.

    ``task_owner_user_id`` is the task OWNER's id -- the runtime identity the
    resume executes as (``UserContext``), not the acting principal.
    """
    from ..models.agent import Agent
    from ..models.database import get_db
    from ..models.task import Task, TaskStatus

    lease_stop_event = None
    lease_heartbeat_task = None
    lease = None
    lease_released = False
    result: Dict[str, Any] | None = None
    # Token tracking + mid-run quota gate for the resumed segment (resume had
    # neither before, so a resumed run escaped mid-run enforcement entirely).
    resume_tracker = None
    resume_tracker_db = None
    resume_tracker_db_gen = None
    normalized_outputs: list[Dict[str, str]] = []
    output = ""
    success = False
    final_status = TaskStatus.RUNNING.value
    task_title: str | None = None
    task_description: str | None = None
    task_execution_mode: str | None = None
    task_agent_id: int | None = None
    agent_name: str | None = None
    agent_logo_url: str | None = None
    delivery_was_dispatched = delivery_already_dispatched
    final_control_snapshot = None

    async def notify_deferred_delivery(
        accepted: bool,
        message: str | None = None,
        *,
        retry_with_new_id: bool = False,
    ) -> None:
        if delivery_websocket is None or delivery_client_message_id is None:
            return
        try:
            await _send_message_delivery(
                delivery_websocket,
                client_message_id=delivery_client_message_id,
                turn_id=delivery_turn_id or delivery_client_message_id,
                accepted=accepted,
                message=message,
                retry_with_new_id=retry_with_new_id,
            )
        except Exception:
            # Delivery state is durable; a disconnected client will retry the
            # same id and recover the result from that state.
            logger.warning(
                "Could not send deferred delivery acknowledgement for task %s",
                task_id,
                exc_info=True,
            )

    try:
        if previous_task is not None and not previous_task.done():
            try:
                await previous_task
            except Exception as e:
                logger.warning(
                    f"Previous background task {task_id} ended before resume: {e}"
                )

        current_task = asyncio.current_task()
        if current_task is None:
            raise RuntimeError(f"Task {task_id} resume has no asyncio task")
        background_task_manager.promote_resume_task(task_id, current_task)

        db_gen = get_db()
        db_lease = next(db_gen)
        try:
            lease = acquire_task_lease(
                db_lease,
                task_id,
                expected_run_id=expected_run_id,
            )
            if lease is not None:
                task_for_sync = db_lease.query(Task).filter(Task.id == task_id).first()
                # Same owner guard as ``execute_task_background``: the resume
                # runs under ``UserContext(task_owner_user_id)`` below, so a
                # passed owner that disagrees with the task row would resume as
                # the wrong user. All callers pass the owner; a mismatch is a
                # programming error -- fail loudly rather than run as the wrong
                # identity.
                if (
                    task_for_sync is not None
                    and task_owner_user_id is not None
                    and int(task_for_sync.user_id) != task_owner_user_id
                ):
                    raise ValueError(
                        f"execute_resume_background: passed task_owner_user_id "
                        f"{task_owner_user_id} does not match task {task_id} "
                        f"owner {int(task_for_sync.user_id)}; refusing to resume "
                        "as the wrong user"
                    )
                if task_for_sync is not None and sync_workforce_run_status(
                    db_lease, task_for_sync, TaskStatus.RUNNING
                ):
                    db_lease.commit()
                elif task_for_sync is not None:
                    db_lease.commit()
        finally:
            db_lease.close()
        if lease is None:
            logger.info(
                "Task %s resume skipped; another runner owns the lease", task_id
            )
            if delivery_turn_id is not None and not delivery_was_dispatched:
                await asyncio.to_thread(
                    mark_user_message_delivery_sync,
                    task_id,
                    delivery_turn_id,
                    DELIVERY_FAILED,
                )
                await notify_deferred_delivery(
                    False,
                    "The deferred message could not be delivered. Please retry.",
                    retry_with_new_id=True,
                )
            await manager.broadcast_to_task(
                {
                    "type": "agent_error",
                    "message": "Task is already running on another worker.",
                    "task": {"id": task_id, "status": TaskStatus.RUNNING.value},
                    "timestamp": datetime.now(timezone.utc).timestamp(),
                },
                task_id,
            )
            return
        lease_stop_event = asyncio.Event()
        lease_heartbeat_task = asyncio.create_task(
            run_task_lease_heartbeat(lease, lease_stop_event)
        )

        # The task row can become RUNNING before the original AgentRunner has
        # created a context/checkpoint. Retry an early failed injection only
        # after that original execution has settled and persisted its state.
        # Acquire the execution lease first: otherwise a non-owner worker could
        # persist the injection and acknowledge it, then discover that it is
        # not allowed to run the resume.
        if pending_user_message is not None:
            posted = await agent_service.post_user_message(
                str(task_id),
                execution_message=pending_user_message.get("execution_message"),
                display_message=pending_user_message.get("display_message"),
                files=pending_user_message.get("files"),
                turn_id=pending_user_message.get("turn_id"),
                request_interrupt=False,
                reason="deferred websocket user message",
            )
            if not posted:
                raise RuntimeError(
                    "The user message was saved, but no resumable execution "
                    "checkpoint became available."
                )
            delivery_was_dispatched = True
            if delivery_turn_id is not None:
                await asyncio.to_thread(
                    mark_user_message_delivery_sync,
                    task_id,
                    delivery_turn_id,
                    DELIVERY_DISPATCHED,
                )
            await notify_deferred_delivery(True)

        # Resume is now durable: lease acquisition committed RUNNING. Do not
        # announce it earlier from the WebSocket request handler.
        await manager.broadcast_to_task(
            {
                "type": "task_resumed",
                "task_id": task_id,
                "message": "Task resumed",
                "timestamp": datetime.now(timezone.utc).timestamp(),
            },
            task_id,
        )

        # Track tokens and enforce the mid-run quota gate on the resumed segment
        # too. Best-effort: a tracking hiccup must never block the resume.
        try:
            from ..tracking.task_tracker import TaskTracker

            resume_tracker_db_gen = get_db()
            resume_tracker_db = next(resume_tracker_db_gen)
            resume_tracker = TaskTracker(
                task_id=int(task_id), db_session=resume_tracker_db
            )
            await resume_tracker.start_tracking()
            agent_service.set_interrupt_checker(
                resume_tracker.interrupt_reason_for_quota
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"execute_resume_background: token tracking unavailable "
                f"for task {task_id}: {e}"
            )
            resume_tracker = None

        with UserContext(task_owner_user_id), turn_execution_scope(task_id):
            result = await agent_service.resume_execution_by_id(str(task_id))

        if result is None:
            raise RuntimeError(
                f"No resumable execution checkpoint was found for task {task_id}."
            )

        # If the mid-run quota gate stopped the resumed run, surface the reason
        # the way the start gate does instead of a silent flip to PAUSED.
        if resume_tracker is not None and isinstance(result, dict):
            _quota_reason = getattr(resume_tracker, "quota_interrupt_reason", None)
            if _quota_reason:
                result = {
                    **result,
                    "success": False,
                    "status": "quota_exceeded",
                    "output": _quota_reason,
                    "error": _quota_reason,
                    # A mid-run interrupt is always the quota checker, so forward
                    # the code the way the start gate does (see chat.py).
                    "error_code": "quota_exceeded",
                }

        status = str(result.get("status") or "")
        success = bool(result.get("success", False))
        output = str(result.get("output") or result.get("error") or "")

        if task_owner_user_id is not None:
            db_gen = get_db()
            db_normalize = next(db_gen)
            try:
                task_for_normalize = (
                    db_normalize.query(Task).filter(Task.id == task_id).first()
                )
                if task_for_normalize is not None:
                    normalized_outputs, path_to_file_id = _normalize_task_file_outputs(
                        db_normalize,
                        task_for_normalize,
                        result.get("file_outputs", []),
                    )
                    if normalized_outputs:
                        result["file_outputs"] = normalized_outputs
                        output = _rewrite_file_links_to_file_id(output, path_to_file_id)
                    output = reconcile_assistant_file_references(
                        db_normalize,
                        task_id=int(task_id),
                        user_id=int(task_owner_user_id),
                        content=output,
                    )
            finally:
                db_normalize.close()

        db_gen = get_db()
        db_new = next(db_gen)
        try:
            task_updated = db_new.query(Task).filter(Task.id == task_id).first()
            if task_updated:
                task_title = cast(Any, task_updated.title)
                task_description = cast(Any, task_updated.description)
                task_execution_mode = cast(Any, task_updated.execution_mode)
                task_agent_id = cast(Any, task_updated.agent_id)
                if task_updated.agent_id is not None:
                    agent = (
                        db_new.query(Agent)
                        .filter(Agent.id == task_updated.agent_id)
                        .first()
                    )
                    if agent is not None:
                        agent_name = cast(Any, agent.name)
                        agent_logo_url = cast(Any, agent.logo_url)
                if status == "waiting_for_user":
                    final_task_status = TaskStatus.WAITING_FOR_USER
                elif status == "interrupted":
                    final_task_status = TaskStatus.PAUSED
                elif success:
                    final_task_status = TaskStatus.COMPLETED
                else:
                    final_task_status = TaskStatus.FAILED

                final_control_snapshot = apply_task_control_transition(
                    task_updated,
                    {
                        TaskStatus.WAITING_FOR_USER: TaskControlState.WAITING_FOR_USER,
                        TaskStatus.PAUSED: TaskControlState.PAUSED,
                        TaskStatus.COMPLETED: TaskControlState.COMPLETED,
                        TaskStatus.FAILED: TaskControlState.FAILED,
                    }[final_task_status],
                    status=final_task_status,
                    expected_run_id=expected_run_id,
                )

                if success and output.strip() and task_owner_user_id is not None:
                    from ..services.chat_history_service import (
                        persist_assistant_message_no_commit,
                    )

                    persist_assistant_message_no_commit(
                        db_new,
                        task_id=task_id,
                        user_id=int(task_owner_user_id),
                        content=output,
                        message_type="final_answer",
                        turn_id=_latest_result_user_turn_id(result),
                        content_is_reconciled=True,
                    )
                    orm_task_updated = cast(Any, task_updated)
                    orm_task_updated.output = output
                    orm_task_updated.error_message = None
                elif final_task_status == TaskStatus.FAILED:
                    orm_task_updated = cast(Any, task_updated)
                    orm_task_updated.output = None
                    orm_task_updated.error_message = output or "Task execution failed."
                lease_released = release_current_runner_task_lease_with_workforce_sync(
                    db_new,
                    task_id,
                    status=final_task_status,
                    expected_run_id=expected_run_id,
                )
                db_new.refresh(task_updated)
                final_status = task_updated.status.value
        finally:
            db_new.close()

        if delivery_turn_id is not None:
            await asyncio.to_thread(
                mark_user_message_delivery_sync,
                task_id,
                delivery_turn_id,
                DELIVERY_COMPLETED,
            )

        control_event_state = (
            final_control_snapshot.as_dict()
            if final_control_snapshot is not None
            else {}
        )

        if status in {"interrupted", "waiting_for_user"}:
            await manager.broadcast_to_task(
                create_stream_event(
                    "task_info",
                    task_id,
                    {
                        "id": task_id,
                        "title": task_title,
                        "description": task_description,
                        "status": final_status,
                        "execution_mode": task_execution_mode,
                        "agent_id": task_agent_id,
                        "agent_name": agent_name,
                        "agent_logo_url": agent_logo_url,
                        **control_event_state,
                    },
                ),
                task_id,
            )
            return

        await manager.broadcast_to_task(
            {
                "type": "task_completed",
                "task": {
                    "id": task_id,
                    "title": task_title,
                    "status": final_status,
                    "description": task_description,
                },
                "result": output,
                "output": output,
                "file_outputs": normalized_outputs,
                "success": success,
                # Forward the coded reason so a mid-run quota interrupt on a
                # resumed run pops the same dialog as the start-gate path.
                "error_code": result.get("error_code"),
                "error_details": result.get("error_details"),
                **control_event_state,
                "metadata": result.get("metadata", {}),
                "timestamp": datetime.now(timezone.utc).timestamp(),
            },
            task_id,
        )
    except asyncio.CancelledError:
        logger.info(f"V2 resume background task {task_id} cancelled")
        if delivery_turn_id is not None and not delivery_was_dispatched:
            await asyncio.to_thread(
                mark_user_message_delivery_sync,
                task_id,
                delivery_turn_id,
                DELIVERY_FAILED,
            )
            await notify_deferred_delivery(
                False,
                "The deferred message was cancelled. Please retry.",
                retry_with_new_id=True,
            )
        raise
    except Exception as e:
        logger.error(f"V2 resume background task {task_id} failed: {e}", exc_info=True)
        error_message = str(e)
        if delivery_turn_id is not None and not delivery_was_dispatched:
            await asyncio.to_thread(
                mark_user_message_delivery_sync,
                task_id,
                delivery_turn_id,
                DELIVERY_FAILED,
            )
            await notify_deferred_delivery(
                False,
                error_message,
                retry_with_new_id=True,
            )
        current_snapshot = (
            await task_execution_controller.snapshot(task_id)
            if expected_run_id is not None
            else None
        )
        if current_snapshot is not None and current_snapshot.run_id != expected_run_id:
            logger.info(
                "Suppressing late resume error for task %s run %s; current run is %s",
                task_id,
                expected_run_id,
                current_snapshot.run_id,
            )
            return
        await manager.broadcast_to_task(
            {
                **_terminal_task_error_payload(
                    task_id,
                    error_message,
                    event_type="task_error",
                    expected_run_id=expected_run_id,
                ),
                "task_id": task_id,
                "error": error_message,
                "timestamp": datetime.now(timezone.utc).timestamp(),
            },
            task_id,
        )
    finally:
        # Finalize the resumed segment's tracking: drop the checker, meter the
        # partial usage, and close the tracker's dedicated session. Best-effort.
        if resume_tracker is not None:
            agent_service.set_interrupt_checker(None)
            try:
                await resume_tracker.complete_tracking()
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    f"execute_resume_background: token tracking completion "
                    f"failed for task {task_id}: {e}"
                )
        if resume_tracker_db is not None:
            resume_tracker_db.close()
        await stop_task_lease_heartbeat(lease_heartbeat_task, lease_stop_event)
        if lease is not None and not lease_released:
            db_gen = get_db()
            db_cleanup = next(db_gen)
            try:
                release_task_lease_with_workforce_sync(
                    db_cleanup, lease, status=TaskStatus.FAILED
                )
            finally:
                db_cleanup.close()
        _clear_task_pause_accepted(task_id)
        background_task_manager.cleanup_task(task_id)


# Background task manager: ensures only one active background execution per task
class BackgroundTaskManager:
    """Manages background task execution, ensuring only one background process per task at a time"""

    def __init__(self) -> None:
        # task_id -> asyncio.Task
        self.running_tasks: Dict[int, asyncio.Task] = {}
        # Resume coordinators are deliberately tracked separately while they
        # wait for the current execution. Replacing ``running_tasks[task_id]``
        # too early creates a cycle: the original execution waits for the new
        # resume task while that resume task waits for the original execution.
        self.resume_tasks: Dict[int, asyncio.Task] = {}
        self._resume_reservations: set[int] = set()

    async def wait_for_previous(self, task_id: int) -> None:
        """Wait for previous background task of this task to complete"""
        if task_id in self.running_tasks:
            old_task = self.running_tasks[task_id]
            current_task = asyncio.current_task()
            if current_task is not None and old_task is current_task:
                return
            if not old_task.done():
                logger.info(
                    f"Waiting for previous background task {task_id} to complete..."
                )
                try:
                    await old_task
                    logger.info(f"Previous background task {task_id} completed")
                except Exception as e:
                    logger.warning(
                        f"Previous background task {task_id} ended with error: {e}"
                    )

    def register_task(self, task_id: int, task: asyncio.Task) -> None:
        """Register new background task"""
        self.running_tasks[task_id] = task
        logger.info(f"Registered background task for task {task_id}")

    def reserve_resume(self, task_id: int) -> bool:
        """Atomically reserve the single live-control resume slot."""

        # Keep this check-and-add block synchronous: asyncio task switches can
        # only happen at ``await``, so it is the in-process atomic guard.
        existing = self.resume_tasks.get(task_id)
        if task_id in self._resume_reservations or (
            existing is not None and not existing.done()
        ):
            return False
        self._resume_reservations.add(task_id)
        return True

    def register_reserved_resume(self, task_id: int, task: asyncio.Task) -> None:
        if task_id not in self._resume_reservations:
            raise RuntimeError(f"Task {task_id} has no reserved resume slot")
        self._resume_reservations.discard(task_id)
        self.resume_tasks[task_id] = task
        logger.info("Registered resume coordinator for task %s", task_id)

    def release_resume_reservation(self, task_id: int) -> None:
        self._resume_reservations.discard(task_id)

    def promote_resume_task(self, task_id: int, task: asyncio.Task) -> None:
        existing = self.resume_tasks.get(task_id)
        if existing is not task:
            raise RuntimeError(
                f"Task {task_id} resume coordinator is not registered or no longer current"
            )
        self.running_tasks[task_id] = task
        logger.info("Promoted resume coordinator for task %s", task_id)

    def cleanup_task(self, task_id: int) -> None:
        """Clean up completed background task"""
        current = asyncio.current_task()
        task = self.running_tasks.get(task_id)
        if task is not None and (task.done() or task is current):
            self.running_tasks.pop(task_id, None)
            logger.info(f"Cleaned up background task for task {task_id}")
        resume_task = self.resume_tasks.get(task_id)
        if resume_task is not None and (resume_task.done() or resume_task is current):
            self.resume_tasks.pop(task_id, None)
            logger.info("Cleaned up resume coordinator for task %s", task_id)

    async def cancel_task(self, task_id: int, timeout_seconds: float = 0.5) -> None:
        tasks = {
            task
            for task in (
                self.running_tasks.get(task_id),
                self.resume_tasks.get(task_id),
            )
            if task is not None
        }
        if not tasks:
            return

        for task in tasks:
            if task.done():
                continue
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=timeout_seconds)
            except asyncio.CancelledError:
                logger.info(f"Cancelled background task for task {task_id}")
            except asyncio.TimeoutError:
                logger.info(
                    f"Cancellation timeout for task {task_id}; continuing cleanup"
                )
            except RuntimeError as e:
                logger.warning(
                    f"Background task {task_id} cancellation runtime warning: {e}"
                )
            except Exception as e:
                logger.warning(
                    f"Background task {task_id} raised during cancellation: {e}"
                )

        self.running_tasks.pop(task_id, None)
        self.resume_tasks.pop(task_id, None)
        self._resume_reservations.discard(task_id)


# Global background task manager
background_task_manager = BackgroundTaskManager()


class SharedWebSocketTracer(TraceHandler):
    """Shared WebSocket tracer that sends events directly to WebSocket with proper JSON serialization."""

    def __init__(self, ws: WebSocket, task_id: str, is_preview: bool = False):
        self.ws = ws
        self.task_id = task_id
        self.is_preview = is_preview
        self._closed = False

    def _serialize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively serialize data to ensure JSON compatibility."""

        def clean_string(value: str) -> str:
            if not isinstance(value, str):
                return value
            cleaned = value.replace("\x00", "").replace("\u0000", "")
            cleaned = "".join(
                char for char in cleaned if ord(char) >= 32 or char in "\n\r\t"
            )
            return cleaned

        def serialize_value(value: Any) -> Any:
            if hasattr(value, "model_dump"):
                return serialize_value(value.model_dump())
            elif callable(getattr(value, "to_dict", None)):
                return serialize_value(value.to_dict())
            elif hasattr(value, "dict"):
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
            cleaned_data = cast(Dict[str, Any], serialize_value(data))
            json.dumps(cleaned_data)
            return cleaned_data
        except Exception as e:
            logger.warning(f"Failed to serialize data for JSON: {e}")
            return {"_serialization_error": str(e)}

    async def handle_event(self, event: TraceEvent) -> None:
        """Convert and send trace event to WebSocket."""
        # Skip if WebSocket is already closed
        if self._closed:
            return

        try:
            from .ws_trace_handlers import get_event_type_mapping

            if _is_audit_only_trace_data(event.data):
                return

            # Convert trace event to stream format
            event_type_str = get_event_type_mapping(event)
            serialized_data = self._serialize_data(event.data)
            if _is_agent_checkpoint_data(serialized_data):
                return
            event_type_str, serialized_data = normalize_public_trace_event(
                event_type_str, serialized_data
            )

            stream_event = create_stream_event(
                event_type_str,
                0 if self.is_preview else self.task_id,
                serialized_data,
                event.timestamp,
            )

            if event.step_id:
                stream_event["step_id"] = event.step_id
            if event.parent_id:
                stream_event["parent_id"] = event.parent_id
            if self.is_preview:
                stream_event["is_preview"] = True

            await self.ws.send_text(json.dumps(stream_event))

        except (RuntimeError, ConnectionError) as e:
            error_msg = str(e)
            if (
                "close" in error_msg.lower()
                or "response already completed" in error_msg.lower()
            ):
                self._closed = True
                logger.debug(f"WebSocket connection closed: {e}")
            else:
                logger.warning(f"WebSocket error in tracer: {e}")
        except Exception as e:
            logger.warning(f"Failed to send trace event: {e}")


# WebSocket router
ws_router = APIRouter()


@ws_router.get("/preview/{legacy_path:path}", response_model=None)
async def redirect_legacy_preview(
    legacy_path: str,
    db: Session = Depends(get_db),
) -> Any:
    resolved_info = _resolve_legacy_preview_storage_path(legacy_path)
    if resolved_info is None:
        raise HTTPException(status_code=404, detail="Legacy preview target not found")

    resolved_path, relative_path = resolved_info
    file_record = (
        db.query(UploadedFile)
        .filter(UploadedFile.storage_path == str(resolved_path))
        .first()
    )

    if file_record is None:
        owner_info = _infer_owner_from_relative_path(db, relative_path)
        if owner_info is None:
            raise HTTPException(
                status_code=404, detail="Cannot infer owner for legacy preview path"
            )

        owner_user_id, task_id = owner_info
        generated_file_id = _build_output_file_id(relative_path)
        file_record = UploadedFileStore(db).create_from_local_path(
            local_path=resolved_path,
            user_id=owner_user_id,
            file_id=generated_file_id,
            task_id=task_id,
            filename=resolved_path.name,
            mime_type=None,
            storage_key=build_task_output_storage_key(
                owner_user_id,
                cast(int, task_id),
                generated_file_id,
                relative_path,
                scope_segments=_scope_segments_for_task(task_id),
            ),
        )
        db.commit()
        db.refresh(file_record)

    return RedirectResponse(
        url=f"/api/files/public/preview/{file_record.file_id}",
        status_code=307,
    )


_VERSIONED_TASK_EVENT_TYPES = {
    "agent_error",
    "error",
    "task_completed",
    "task_error",
    "task_pause_requested",
    "task_paused",
    "task_resumed",
    "task_started",
    "task_waiting_for_user",
}


def _is_versioned_task_event(message: dict[str, Any]) -> bool:
    message_type = str(message.get("type") or "")
    if message_type in _VERSIONED_TASK_EVENT_TYPES:
        return True
    return (
        message_type == "trace_event"
        and str(
            message.get("event_type")
            or (
                message.get("data", {}).get("event_type")
                if isinstance(message.get("data"), dict)
                else ""
            )
        )
        == "task_info"
    )


def _event_task_id(message: dict[str, Any]) -> int | None:
    candidates = [message.get("task_id")]
    task_data = message.get("task")
    if isinstance(task_data, dict):
        candidates.append(task_data.get("id"))
        candidates.append(task_data.get("task_id"))
    data = message.get("data")
    if isinstance(data, dict):
        candidates.append(data.get("id"))
        candidates.append(data.get("task_id"))
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            return int(candidate)
        except (TypeError, ValueError):
            continue
    return None


def _event_task_control_state(message: dict[str, Any]) -> dict[str, Any] | None:
    sources = [message]
    task_data = message.get("task")
    if isinstance(task_data, dict):
        sources.append(task_data)
    data = message.get("data")
    if isinstance(data, dict):
        sources.append(data)

    for source in sources:
        version = source.get("state_version")
        control_state = source.get("control_state")
        status = source.get("status")
        if (
            isinstance(version, int)
            and not isinstance(version, bool)
            and version >= 0
            and isinstance(control_state, str)
            and isinstance(status, str)
            and (isinstance(source.get("run_id"), str) or source.get("run_id") is None)
        ):
            return {
                "run_id": source.get("run_id"),
                "state_version": version,
                "control_state": control_state,
                "status": status,
            }
    return None


async def _with_current_task_control_state(
    message: dict[str, Any],
    *,
    fallback_task_id: int | None = None,
) -> dict[str, Any]:
    """Attach one canonical DB state tuple to a state-bearing event.

    Event producers can finish out of order. Preserve a producer-captured
    state tuple when present; otherwise attach the current row snapshot.
    Clients compare the resulting ``run_id`` / ``state_version`` before
    applying the event.
    """

    if not _is_versioned_task_event(message):
        return message
    task_id = _event_task_id(message) or fallback_task_id
    if task_id is None:
        return message
    state = _event_task_control_state(message)
    if state is None:
        snapshot = await task_execution_controller.snapshot(task_id)
        if snapshot is None:
            return message
        state = snapshot.as_dict()
    enriched = dict(message)
    enriched.update(state)
    enriched["task_id"] = task_id

    if enriched.get("type") == "trace_event":
        data = enriched.get("data")
        enriched["data"] = {**(data if isinstance(data, dict) else {}), **state}

    task_data = enriched.get("task")
    if isinstance(task_data, dict):
        enriched["task"] = {**task_data, **state, "id": task_id}
    return enriched


# Connection manager
class ConnectionManager:
    def __init__(self) -> None:
        # task_id -> List[WebSocket]
        self.active_connections: Dict[int, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, task_id: int) -> None:
        await websocket.accept()
        self.register_connection(websocket, task_id)

    def register_connection(self, websocket: WebSocket, task_id: int) -> None:
        """Register an already-accepted websocket for task broadcasts."""
        if task_id not in self.active_connections:
            self.active_connections[task_id] = []
        if websocket not in self.active_connections[task_id]:
            self.active_connections[task_id].append(websocket)

    def disconnect(self, websocket: WebSocket, task_id: int) -> None:
        if task_id in self.active_connections:
            try:
                self.active_connections[task_id].remove(websocket)
                if not self.active_connections[task_id]:
                    del self.active_connections[task_id]
            except ValueError:
                pass

    def move_connection(
        self, websocket: WebSocket, old_task_id: int, new_task_id: int
    ) -> None:
        """Move a WebSocket connection from one task_id to another"""
        if old_task_id in self.active_connections:
            try:
                self.active_connections[old_task_id].remove(websocket)
                if not self.active_connections[old_task_id]:
                    del self.active_connections[old_task_id]
            except ValueError:
                pass

        if new_task_id not in self.active_connections:
            self.active_connections[new_task_id] = []
        self.active_connections[new_task_id].append(websocket)
        logger.info(
            f"Moved WebSocket connection from task {old_task_id} to {new_task_id}"
        )

    async def send_personal_message(self, message: dict, websocket: WebSocket) -> None:
        versioned_message = await _with_current_task_control_state(message)
        await websocket.send_text(json.dumps(versioned_message))

    async def broadcast_to_task(self, message: dict, task_id: int) -> None:
        if task_id in self.active_connections:
            versioned_message = await _with_current_task_control_state(
                message,
                fallback_task_id=task_id,
            )
            for connection in self.active_connections[task_id].copy():
                try:
                    await connection.send_text(json.dumps(versioned_message))
                except (ConnectionError, WebSocketDisconnect, RuntimeError) as e:
                    # Network connection error, remove disconnected connection
                    logger.warning(f"Connection error for task {task_id}: {e}")
                    self.disconnect(connection, task_id)
                except Exception as e:
                    # Other errors should not be silently handled, log and re-raise
                    logger.error(
                        f"Unexpected error broadcasting to task {task_id}: {e}"
                    )
                    # Remove disconnected connection but preserve error propagation
                    self.disconnect(connection, task_id)
                    raise


# Global connection manager
manager = ConnectionManager()


async def handle_file_upload_for_task(
    task_id: int,
    files: list,
    db: Session,
    user: Optional[User] = None,
    task_owner_id: Optional[int] = None,
) -> dict:
    """Handle file upload for task.

    Thin transport wrapper over the shared ``services.file_turn`` pipeline:
    resolve the requested file ids to file-info dicts, then bind the ones
    that resolved to this task. WS keeps its lenient behavior — files that
    don't resolve are logged and skipped, not raised.
    """
    try:
        logger.info(f"📁 Starting file upload for task {task_id}, files: {len(files)}")

        authorized_owner_id = task_owner_id
        if authorized_owner_id is None and user is not None:
            authorized_owner_id = int(user.id)
        if authorized_owner_id is None:
            logger.warning(
                "Cannot handle uploaded files for task %s without an authorized owner",
                task_id,
            )
            return {"uploaded_files": [], "file_info_list": []}

        file_ids = [str(f.get("file_id")) for f in files if f.get("file_id")]
        file_info_list, missing = resolve_turn_file_infos(
            file_ids=file_ids,
            owner_user_id=int(authorized_owner_id),
            db=db,
            task_id=int(task_id),
        )
        for missing_id in missing:
            logger.warning(
                "File record not accessible for task %s: %s", task_id, missing_id
            )

        bind_turn_files(
            file_ids=[info["file_id"] for info in file_info_list],
            task_id=int(task_id),
            owner_user_id=int(authorized_owner_id),
            db=db,
        )

        uploaded_files = [info["path"] for info in file_info_list]
        logger.info(f"🎉 File upload completed, uploaded {len(uploaded_files)} files")
        return {"uploaded_files": uploaded_files, "file_info_list": file_info_list}

    except Exception as e:
        logger.error(f"Error handling file upload for task {task_id}: {e}")
        raise


def _register_uploaded_files_for_agent(
    agent_service: Any,
    file_info_list: List[Dict[str, Any]],
    db: Session,
) -> None:
    """Expose staged upload records to the agent workspace under its DB session."""
    workspace = getattr(agent_service, "workspace", None)
    if not workspace:
        return

    input_dir = Path(workspace.input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    for file_info in file_info_list:
        file_id = str(file_info.get("file_id") or "")
        source_path = Path(str(file_info.get("path") or ""))
        if not file_id or not source_path.exists():
            logger.warning(
                "Skipping unavailable uploaded file for workspace: %s", file_info
            )
            continue

        normalized_file_name = normalize_filename(
            Path(str(file_info.get("name") or source_path.name)).name
        )
        candidate = input_dir / normalized_file_name
        suffix_idx = 1
        stem, ext = candidate.stem, candidate.suffix
        while candidate.exists() or candidate.is_symlink():
            try:
                if candidate.resolve() == source_path.resolve():
                    break
            except OSError:
                pass
            candidate = input_dir / f"{stem}_{suffix_idx}{ext}"
            suffix_idx += 1

        workspace_link_path: Path | None
        if candidate.exists() or candidate.is_symlink():
            workspace_link_path = candidate
        else:
            try:
                candidate.symlink_to(source_path.resolve())
                workspace_link_path = candidate
            except OSError as link_err:
                logger.warning(
                    f"symlink failed ({link_err}); copying "
                    f"{source_path.name} into workspace"
                )
                shutil.copy2(source_path, candidate)
                workspace_link_path = candidate

        registration_path = source_path.resolve()
        workspace.register_file(
            str(registration_path),
            file_id=file_id,
            db_session=db,
        )
        file_info["path"] = str(registration_path)
        file_info["workspace_path"] = str(workspace_link_path)
        logger.info(
            "File registered for agent workspace: storage=%s input_link=%s",
            registration_path,
            workspace_link_path,
        )


async def get_authenticated_user(
    websocket: WebSocket, token: Optional[str] = None
) -> Optional[User]:
    """
    Get authenticated user from WebSocket connection

    Args:
        websocket: WebSocket connection
        token: Optional authentication token

    Returns:
        User if authenticated, None otherwise
    """
    if not token:
        return None

    try:
        from ..models.database import get_db

        db_gen = get_db()
        db = next(db_gen)

        try:
            return get_user_from_websocket_token(token, db)
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error authenticating WebSocket user: {e}")
        return None


async def handle_chat_message(
    websocket: WebSocket, task_id: int, message_data: dict
) -> None:
    """Durably accept a chat command before acknowledging the client."""

    try:
        enqueued = await _enqueue_websocket_task_command(
            task_id=task_id,
            message_data=message_data,
            kind=TaskCommandKind.MESSAGE,
            command_id=_client_message_id(message_data.get("client_message_id")),
            allow_missing_task=True,
        )
    except (PermissionError, ValueError) as exc:
        client_message_id = _client_message_id(message_data.get("client_message_id"))
        await _send_message_delivery(
            websocket,
            client_message_id=client_message_id,
            turn_id=client_message_id or str(uuid.uuid4()),
            accepted=False,
            message=str(exc),
        )
        await manager.send_personal_message(
            {"type": "error", "message": str(exc)}, websocket
        )
        return
    if enqueued is None:
        # Legacy recovery path for a client still connected to a task that was
        # deleted. The existing handler creates the replacement task first;
        # subsequent commands use the durable transport normally.
        async with task_execution_controller.command(task_id):
            await _handle_chat_message_unserialized(websocket, task_id, message_data)
        return
    if not enqueued.payload_matches:
        await _send_message_delivery(
            websocket,
            client_message_id=_client_message_id(message_data.get("client_message_id")),
            turn_id=enqueued.client_command_id,
            accepted=False,
            message="Message id was already used for different content or files.",
            retry_with_new_id=True,
        )
        return
    if enqueued.status == COMMAND_FAILED:
        await _send_message_delivery(
            websocket,
            client_message_id=_client_message_id(message_data.get("client_message_id")),
            turn_id=enqueued.client_command_id,
            accepted=False,
            message="The previous delivery attempt failed. Please retry the draft.",
            retry_with_new_id=True,
        )
        return
    await _send_message_delivery(
        websocket,
        client_message_id=_client_message_id(message_data.get("client_message_id")),
        turn_id=enqueued.client_command_id,
        accepted=True,
    )
    if enqueued.command_id:
        await dispatch_task_command_promptly(
            execute_durable_task_command,
            command_db_id=enqueued.command_id,
        )


def _enqueue_websocket_task_command_sync(
    *,
    task_id: int,
    actor_user_id: int,
    actor_is_admin: bool,
    command_id: str,
    kind: TaskCommandKind,
    payload: dict[str, Any],
    allow_missing_task: bool,
) -> EnqueuedTaskCommand | None:
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = db.query(Task).filter(Task.id == task_id).first()
        if task is None:
            if allow_missing_task:
                return None
            raise ValueError(f"Task {task_id} not found")
        if not actor_is_admin and int(task.user_id) != actor_user_id:
            raise PermissionError(
                f"Access denied: Task {task_id} does not belong to you"
            )
        if kind == TaskCommandKind.MESSAGE:
            from ..services.chat_history_service import (
                inspect_user_message_delivery,
            )

            existing_delivery = inspect_user_message_delivery(
                db,
                task_id,
                str(payload.get("message") or ""),
                attachments=(
                    payload.get("files")
                    if isinstance(payload.get("files"), list)
                    else None
                ),
                turn_id=command_id,
            )
            if (
                existing_delivery is not None
                and not existing_delivery.pending
                and not payload.get("files")
            ):
                return EnqueuedTaskCommand(
                    command_id=0,
                    client_command_id=command_id,
                    created=False,
                    payload_matches=existing_delivery.payload_matches,
                    status=(
                        DELIVERY_FAILED
                        if existing_delivery.failed
                        else DELIVERY_COMPLETED
                    ),
                )
        result = enqueue_task_command(
            db,
            task_id=task_id,
            actor_user_id=actor_user_id,
            command_id=command_id,
            kind=kind,
            payload=payload,
        )
        return result


async def _enqueue_websocket_task_command(
    *,
    task_id: int,
    message_data: dict[str, Any],
    kind: TaskCommandKind,
    command_id: str | None = None,
    allow_missing_task: bool = False,
) -> EnqueuedTaskCommand | None:
    user = message_data.get("user")
    if user is None:
        raise ValueError("User authentication required for task command")
    resolved_command_id = command_id or f"{kind.value}:{uuid.uuid4()}"
    # User ORM instances and server-only authentication fields are never put
    # into the JSON inbox. The consumer re-resolves the actor by id.
    payload = {
        key: value
        for key, value in message_data.items()
        if key not in {"user", "user_id"} and not key.startswith("_durable_")
    }
    if kind == TaskCommandKind.MESSAGE:
        # The durable command identity is also the delivery/turn identity.
        # This remains stable across retries even when an API client omitted
        # or supplied an invalid client_message_id.
        payload["client_message_id"] = resolved_command_id
    return await asyncio.to_thread(
        _enqueue_websocket_task_command_sync,
        task_id=int(task_id),
        actor_user_id=int(user.id),
        actor_is_admin=bool(user.is_admin),
        command_id=resolved_command_id,
        kind=kind,
        payload=payload,
        allow_missing_task=allow_missing_task,
    )


async def _handle_chat_message_unserialized(
    websocket: WebSocket, task_id: int, message_data: dict
) -> None:
    """Handle chat message"""
    client_message_id = _client_message_id(message_data.get("client_message_id"))
    turn_id = client_message_id or str(uuid.uuid4())
    suppress_delivery_ack = bool(message_data.get("_durable_ack_sent"))
    delivery_finished = False
    delivery_dispatched = False
    delivery_claimed = False
    recovered_delivery: UserMessageDeliveryClaim | None = None

    async def finish_delivery(
        accepted: bool,
        message: str | None = None,
        *,
        retry_with_new_id: bool = False,
    ) -> None:
        nonlocal delivery_finished
        if delivery_finished:
            return
        delivery_finished = True
        if not accepted:
            message_data["_durable_command_error"] = message or "Message was rejected"
        if suppress_delivery_ack:
            return
        await _send_message_delivery(
            websocket,
            client_message_id=client_message_id,
            turn_id=turn_id,
            accepted=accepted,
            message=message,
            retry_with_new_id=retry_with_new_id,
        )

    async def finish_delivery_failure(message: str) -> None:
        """Reject pre-dispatch failures; never confuse persistence with delivery."""

        if delivery_finished:
            return
        if delivery_claimed and not delivery_dispatched:
            await asyncio.to_thread(
                mark_user_message_delivery_sync,
                task_id,
                turn_id,
                DELIVERY_FAILED,
            )
        await finish_delivery(
            delivery_dispatched,
            None if delivery_dispatched else message,
        )

    async def finish_existing_delivery(
        claim: UserMessageDeliveryClaim,
    ) -> None:
        if not claim.payload_matches:
            await finish_delivery(
                False,
                "Message id was already used for different content or files.",
                retry_with_new_id=True,
            )
        elif claim.failed:
            await finish_delivery(
                False,
                "The previous delivery attempt failed. Please retry the draft.",
                retry_with_new_id=True,
            )
        elif claim.pending:
            await finish_delivery(
                False,
                "The message is still being applied. Please retry shortly.",
            )
        else:
            await finish_delivery(True)

    try:
        user_message = message_data.get("message", "")

        context = message_data.get("context", {})
        files = message_data.get("files", [])
        user = message_data.get("user")
        authorized_task_id: int | None = None

        # Race-condition fallback: when the message arrives without `files`
        # in its payload, the frontend may still have uploaded files via the
        # HTTP /api/files/upload endpoint a moment earlier. Look those up in
        # the DB and treat them as if they had been declared inline. This
        # fixes the task-36 scenario where the agent's first turn answered
        # "I don't see any documents" despite a successful HTTP upload.
        if not files and user is not None:
            try:
                with closing(get_db()) as _db_iter:
                    _db: Session = next(_db_iter)
                    cutoff = datetime.now(timezone.utc).replace(
                        tzinfo=None
                    ) - timedelta(minutes=5)
                    pending = (
                        _db.query(UploadedFile)
                        .filter(
                            UploadedFile.user_id == int(user.id),
                            UploadedFile.task_id == int(task_id),
                            UploadedFile.created_at >= cutoff,
                        )
                        .order_by(UploadedFile.created_at.desc())
                        .all()
                    )
                    if pending:
                        files = [
                            {
                                "file_id": str(record.file_id),
                                "name": str(record.filename),
                                "size": int(record.file_size or 0),
                                "type": record.mime_type,
                            }
                            for record in pending
                        ]
                        logger.info(
                            f"📁 Race fallback: recovered {len(files)} "
                            f"uploaded file(s) from DB for task {task_id}"
                        )
            except Exception as _e:  # noqa: BLE001
                logger.warning(
                    f"Race fallback file lookup failed for task {task_id}: {_e}"
                )

        logger.info(f"Received chat message for task {task_id}")
        logger.info(f"👤 User: {user.id if user else 'unknown'}")
        logger.info(f"📄 Message: {user_message}")
        logger.info(f"📁 Files received from websocket/fallback: {len(files)}")

        # Call Agent to handle - use same agent manager as chat API
        try:
            from .chat import get_agent_manager

            # Get database session
            db_gen = get_db()
            db: Session = next(db_gen)

            try:
                # Verify user permissions and get task
                if not user:
                    raise ValueError("User authentication required for task access")

                # Check if task exists and belongs to current user, unless admin
                if user.is_admin:
                    task = db.query(Task).filter(Task.id == task_id).first()
                else:
                    task = (
                        db.query(Task)
                        .filter(Task.id == task_id, Task.user_id == user.id)
                        .first()
                    )

                if not task:
                    # Check if task exists but doesn't belong to current user
                    existing_task = db.query(Task).filter(Task.id == task_id).first()
                    if existing_task:
                        # Task exists but doesn't belong to current user, deny access
                        logger.warning(
                            f"User {user.id} attempted to access task {task_id} belonging to user {existing_task.user_id}"
                        )
                        raise ValueError(
                            f"Access denied: Task {task_id} does not belong to you"
                        )
                    else:
                        # Task doesn't exist (may have been deleted), create new task
                        # This is a fresh start, don't use continuation logic
                        logger.info(
                            f"Task {task_id} not found (may have been deleted). Creating new task."
                        )
                        task_title = f"Chat: {user_message}"
                        if len(task_title) > 50:
                            task_title = task_title[:50] + "..."

                        task = Task(
                            user_id=int(user.id),  # Use authenticated user ID
                            title=task_title,
                            description=user_message,
                            status=TaskStatus.PENDING,  # Use PENDING instead of RUNNING
                            execution_mode=get_default_task_execution_mode(),
                            connector_runtime_selected_refs=[],
                        )
                        db.add(task)
                        db.commit()
                        db.refresh(task)

                        # Update task_id to newly created task ID
                        old_task_id = task_id
                        task_id = int(task.id)
                        logger.info(
                            f"Created new task with ID {task_id}, replacing old task_id {old_task_id}"
                        )

                        # Move WebSocket connection to new task_id
                        manager.move_connection(websocket, old_task_id, task_id)

                        # Send task ID update event to notify frontend
                        await manager.send_personal_message(
                            {
                                "type": "task_id_updated",
                                "old_task_id": old_task_id,
                                "new_task_id": task_id,
                            },
                            websocket,
                        )

                        # Send task info event to update frontend state
                        logger.info(
                            f"Sending task_info event for new task {task_id}, status: {task.status.value}"
                        )

                        # Determine is_dag from agent config if agent_id exists
                        is_dag = None
                        if task.agent_id:
                            from ..models.agent import Agent

                            agent = (
                                db.query(Agent)
                                .filter(Agent.id == task.agent_id)
                                .first()
                            )
                            if agent:
                                is_dag = agent.execution_mode == "think"

                        (
                            model_id,
                            small_fast_model_id,
                            visual_model_id,
                            compact_model_id,
                        ) = _resolve_task_llm_ids(task, db)

                        task_event = create_stream_event(
                            "task_info",
                            task_id,
                            {
                                "id": task.id,
                                "title": task.title,
                                "description": task.description,
                                "status": task.status.value,
                                "model_id": model_id,
                                "small_fast_model_id": small_fast_model_id,
                                "visual_model_id": visual_model_id,
                                "compact_model_id": compact_model_id,
                                "model_name": task.model_name,
                                "small_fast_model_name": task.small_fast_model_name,
                                "visual_model_name": task.visual_model_name,
                                "compact_model_name": task.compact_model_name,
                                "execution_mode": task.execution_mode,
                                "agent_id": task.agent_id,
                                "agent_name": task.agent.name if task.agent else None,
                                "agent_logo_url": task.agent.logo_url
                                if task.agent
                                else None,
                                "is_dag": is_dag,
                                "created_at": safe_timestamp_to_unix(task.created_at)
                                if task.created_at
                                else None,
                                "updated_at": safe_timestamp_to_unix(task.updated_at)
                                if task.updated_at
                                else None,
                            },
                            task.created_at if task.created_at else None,
                        )
                        await manager.broadcast_to_task(task_event, task_id)
                        logger.info(f"task_info event sent for task {task_id}")

                authorized_task_id = int(task.id)

                if not files and task.status == TaskStatus.PENDING:
                    files = _selected_file_refs_from_task(task, db)
                    if files:
                        logger.info(
                            f"📁 Recovered {len(files)} selected file(s) from task "
                            f"{task_id} for initial chat turn"
                        )

                logger.info(f"📁 Files used for execution: {len(files)}")
                for i, file_info in enumerate(files):
                    logger.info(
                        f"📄 File {i}: {file_info.get('name', 'unknown')} ({file_info.get('size', 0)} bytes)"
                    )

                # Handle file upload if files present
                uploaded_file_paths = []
                file_info_list = []
                uploaded_files_context = ""
                if files:
                    # Process file upload
                    upload_result = await handle_file_upload_for_task(
                        task_id,
                        files,
                        db,
                        user,
                        task_owner_id=int(task.user_id),
                    )
                    uploaded_file_paths = upload_result.get("uploaded_files", [])
                    file_info_list = upload_result.get("file_info_list", [])

                    if file_info_list:
                        context["uploaded_files"] = uploaded_file_paths
                        context["file_info"] = file_info_list
                        file_ids = [f["file_id"] for f in file_info_list]
                        file_names = [f["name"] for f in file_info_list]
                        file_id_list_str = ", ".join(f'"{fid}"' for fid in file_ids)

                        # Check if this task is an agent-builder task to inject KB instructions
                        is_agent_builder = False
                        if task.agent_id:
                            from ..models.agent import Agent

                            agent_record = (
                                db.query(Agent)
                                .filter(Agent.id == task.agent_id)
                                .first()
                            )
                            if agent_record and agent_record.skills:
                                if isinstance(agent_record.skills, list):
                                    is_agent_builder = any(
                                        s == "agent-builder"
                                        for s in agent_record.skills
                                    )
                                elif isinstance(agent_record.skills, str):
                                    is_agent_builder = (
                                        "agent-builder" in agent_record.skills
                                    )

                        uploaded_files_context = _build_uploaded_files_context(
                            file_info_list,
                            is_agent_builder=is_agent_builder,
                        )
                        file_prompt = (
                            "## UPLOADED FILES\n"
                            f"The user has uploaded {len(file_info_list)} file(s): {file_names}\n\n"
                            f"{FILE_REF_MODEL_INSTRUCTIONS}\n\n"
                        )

                        if is_agent_builder:
                            file_prompt += (
                                f"Use these exact file_ids (UUIDs) with `create_knowledge_base_from_file`:\n"
                                f"  file_ids = [{file_id_list_str}]\n\n"
                                "IMPORTANT: The file_ids above are UUIDs (e.g. '5d983e39-a83b-...'). "
                                "Do NOT use file paths as file_ids. "
                                "Call `create_knowledge_base_from_file` with the file_ids listed above, "
                                "then create or update the agent with the returned collection_name. "
                                "Do NOT generate a 'wait for upload' step — the files are already uploaded."
                            )
                        else:
                            file_prompt += (
                                "These files have been successfully uploaded to the workspace and are ready for processing.\n"
                                "You can use standard workspace tools to read, analyze, or process them."
                            )

                        existing_prompt = context.get("system_prompt")
                        if existing_prompt:
                            context["system_prompt"] = (
                                f"{existing_prompt}\n\n{file_prompt}"
                            )
                        else:
                            context["system_prompt"] = file_prompt

                user_message_for_llm = _append_uploaded_files_context_to_message(
                    user_message,
                    uploaded_files_context,
                )
                display_user_message = _display_message_for_user(
                    user_message,
                    bool(file_info_list),
                )
                display_file_refs = _display_file_refs_from_file_info(file_info_list)
                context["display_message"] = display_user_message
                context["files"] = display_file_refs

                persisted_attachments = _normalize_attachments_for_persistence(
                    file_info_list
                )

                # Retry inspection happens after file normalization so the
                # same text with different attachments cannot alias an older
                # durable turn. Legacy rows with no delivery status are
                # treated as delivered; failed handoffs are explicitly
                # rejected so the frontend can retry with a fresh id.
                if client_message_id is not None:
                    existing_delivery = inspect_user_message_delivery(
                        db,
                        task_id,
                        display_user_message,
                        attachments=persisted_attachments or None,
                        turn_id=turn_id,
                    )
                    if existing_delivery is not None:
                        durable_replay = (
                            int(message_data.get("_durable_attempt_count") or 0) > 1
                        )
                        if (
                            durable_replay
                            and existing_delivery.pending
                            and existing_delivery.payload_matches
                        ):
                            # The command claim is the exclusive replay lease,
                            # so it may safely adopt a PENDING transcript row
                            # left by a worker that died mid-application.
                            recovered_delivery = UserMessageDeliveryClaim(
                                message=existing_delivery.message,
                                claimed=True,
                                payload_matches=True,
                            )
                            delivery_claimed = True
                            if (
                                task.status == TaskStatus.RUNNING
                                and str(task.input or "").strip()
                                == display_user_message.strip()
                            ):
                                # New-turn claim committed before the old
                                # worker died. Do not start or inject it again.
                                mark_user_message_delivery(
                                    db,
                                    task_id=task_id,
                                    turn_id=turn_id,
                                    status=DELIVERY_DISPATCHED,
                                )
                                delivery_dispatched = True
                                await finish_delivery(True)
                                return
                        else:
                            await finish_existing_delivery(existing_delivery)
                            return

                # DAG plan-execute will automatically send user_message trace event

                # The user message is persisted inside
                # ``TaskTurnOrchestrator.begin_turn`` as part of the atomic
                # transition (claim + persist + schedule commit together).

                # Messages to an actively executing task are control-plane
                # input. A PAUSED task plus a fresh user message is a new
                # turn on the same task/thread; only an explicit resume event
                # should continue the paused checkpoint.
                pause_accepted = _is_task_pause_accepted(task_id)
                task_uses_live_control = _task_status_uses_live_control(
                    task.status,
                    control_state=_task_control_state_value(task),
                    pause_accepted=pause_accepted,
                )
                if (
                    recovered_delivery is not None
                    and message_data.get("_durable_target_run_id") == task.run_id
                ):
                    # A crashed owner may have already interrupted the run and
                    # persisted PAUSED before its command claim was completed.
                    # This is the same durable guidance command, not a new turn.
                    task_uses_live_control = True
                agent_service = None
                dag_pattern = None
                supports_live_control = False
                has_continuation = False
                if task_uses_live_control:
                    agent_service = await get_agent_manager().get_agent_for_task(
                        task_id,
                        db,
                        user=user,
                        task_owner_user_id=int(task.user_id),
                    )
                    if hasattr(agent_service, "set_outbound_message_handler"):
                        agent_service.set_outbound_message_handler(
                            make_agent_outbound_handler(task_id)
                        )
                    dag_pattern = (
                        agent_service.get_dag_pattern()
                        if hasattr(agent_service, "get_dag_pattern")
                        else None
                    )
                    supports_live_control = getattr(
                        agent_service, "supports_live_control", lambda: False
                    )()
                    has_continuation = bool(
                        dag_pattern and hasattr(dag_pattern, "request_continuation")
                    )

                if (
                    task_uses_live_control
                    and has_continuation
                    and not supports_live_control
                ):
                    # Use continuation: old task will handle at appropriate time
                    logger.info(f"Using continuation for running task {task_id}")
                    assert dag_pattern is not None  # for mypy type checking

                    delivery_claim = recovered_delivery or claim_user_message_delivery(
                        db,
                        task_id=task_id,
                        user_id=int(task.user_id),
                        content=display_user_message,
                        attachments=persisted_attachments or None,
                        turn_id=turn_id,
                    )
                    recovered_delivery = None
                    if not delivery_claim.claimed:
                        await finish_existing_delivery(delivery_claim)
                        return
                    delivery_claimed = True

                    # Immediately send trace_user_message to display user message on interface
                    if hasattr(dag_pattern, "tracer") and hasattr(
                        dag_pattern, "task_id"
                    ):
                        trace_data: Dict[str, Any] = {
                            "context": context,
                            "pattern": "DAG Plan-Execute Continuation",
                            "continuation": "true",
                            "files": display_file_refs,
                            "turn_id": turn_id,
                        }
                        # Surface uploaded files at the top level so the
                        # frontend user-message renderer can show clickable
                        # file chips alongside the continuation bubble
                        # (matches what historical replay shows on reload).
                        # ``files`` is already populated above via #455's
                        # display_file_refs; mirror it under ``attachments``
                        # for the historical-replay client contract.
                        if display_file_refs:
                            trace_data["attachments"] = display_file_refs
                        await trace_user_message(
                            dag_pattern.tracer,
                            str(dag_pattern.task_id),
                            display_user_message,
                            trace_data,
                        )

                    dag_pattern.request_continuation(user_message_for_llm, context)
                    mark_user_message_delivery(
                        db,
                        task_id=task_id,
                        turn_id=turn_id,
                        status=DELIVERY_DISPATCHED,
                    )
                    # The existing DAG worker owns terminal execution and has
                    # no per-continuation completion callback. For this path,
                    # DISPATCHED is the terminal delivery state: it means the
                    # continuation was accepted, not that the whole DAG ended.
                    delivery_dispatched = True

                    # If previously PAUSED/WAITING_FOR_USER, update status to RUNNING
                    if task.status in {TaskStatus.PAUSED, TaskStatus.WAITING_FOR_USER}:
                        if acquire_task_lease(db, task_id) is None:
                            await manager.send_personal_message(
                                {
                                    "type": "error",
                                    "message": (
                                        "Task is already running on another worker"
                                    ),
                                },
                                websocket,
                            )
                            await finish_delivery(True)
                            return
                        db.refresh(task)
                        if sync_workforce_run_status(db, task, task.status):
                            db.commit()

                        (
                            model_id,
                            small_fast_model_id,
                            visual_model_id,
                            compact_model_id,
                        ) = _resolve_task_llm_ids(task, db)

                        # Send task status update event
                        task_event = create_stream_event(
                            "task_info",
                            task_id,
                            {
                                "id": task.id,
                                "title": task.title,
                                "description": task.description,
                                "status": task.status.value,
                                "model_id": model_id,
                                "small_fast_model_id": small_fast_model_id,
                                "visual_model_id": visual_model_id,
                                "compact_model_id": compact_model_id,
                                "model_name": task.model_name,
                                "small_fast_model_name": task.small_fast_model_name,
                                "visual_model_name": task.visual_model_name,
                                "compact_model_name": task.compact_model_name,
                                "execution_mode": task.execution_mode,
                                "created_at": safe_timestamp_to_unix(task.created_at)
                                if task.created_at
                                else None,
                                "updated_at": safe_timestamp_to_unix(task.updated_at)
                                if task.updated_at
                                else None,
                            },
                            task.created_at if task.created_at else None,
                        )
                        await manager.broadcast_to_task(task_event, task_id)
                        logger.info(f"Task {task_id} status updated to RUNNING")

                    # Continuation will be handled by old task, return directly
                    await finish_delivery(True)
                    return
                if task_uses_live_control and supports_live_control:
                    logger.info(f"Using agent message control for task {task_id}")
                    assert agent_service is not None
                    if not background_task_manager.reserve_resume(task_id):
                        await finish_delivery(
                            False,
                            "A previous guidance message is still being applied. "
                            "Please wait for it to finish.",
                        )
                        return
                    # Pass the user-typed bubble text + display-safe file refs
                    # alongside the LLM-augmented execution text. The runner
                    # persists them onto Message.metadata so its tracing
                    # callback can emit the bubble with the typed content +
                    # file chips rather than the inflated prompt; matches what
                    # historical replay shows on reload.
                    # ``post_user_message`` routes into ``AgentRunner.inject_user_message``,
                    # which dispatches ``on_user_message_posted`` — that callback
                    # is the single emission point for the live-control
                    # continuation user-message trace. Do not emit a second
                    # ``trace_user_message`` here; doing so would render the
                    # bubble twice in the live UI. The DAG Plan-Execute
                    # continuation path above is a separate code path and
                    # keeps its own immediate trace.
                    bg_task: asyncio.Task[None] | None = None
                    try:
                        delivery_claim = (
                            recovered_delivery
                            or claim_user_message_delivery(
                                db,
                                task_id=task_id,
                                user_id=int(task.user_id),
                                content=display_user_message,
                                attachments=persisted_attachments or None,
                                turn_id=turn_id,
                            )
                        )
                        recovered_delivery = None
                        if not delivery_claim.claimed:
                            background_task_manager.release_resume_reservation(task_id)
                            await finish_existing_delivery(delivery_claim)
                            return
                        delivery_claimed = True

                        posted = await agent_service.post_user_message(
                            str(task_id),
                            execution_message=user_message_for_llm,
                            display_message=display_user_message,
                            files=display_file_refs,
                            turn_id=turn_id,
                            request_interrupt=task.status == TaskStatus.RUNNING,
                            reason="new websocket user message",
                        )
                        if not posted:
                            logger.warning(
                                "Agent execution %s was not live; deferring the "
                                "durable user message until its checkpoint is ready",
                                task_id,
                            )
                        else:
                            mark_user_message_delivery(
                                db,
                                task_id=task_id,
                                turn_id=turn_id,
                                status=DELIVERY_DISPATCHED,
                            )
                        # ``post_user_message`` has already durably injected the
                        # turn when it returns True. Preserve that fact even if
                        # the resume reservation is concurrently withdrawn
                        # before the coordinator can be registered.
                        delivery_dispatched = posted

                        await task_execution_controller.transition(
                            task_id,
                            TaskControlState.RESUME_REQUESTED,
                            expected_run_id=_task_run_id(task),
                        )

                        previous_task = background_task_manager.running_tasks.get(
                            task_id
                        )
                        bg_task = asyncio.create_task(
                            execute_resume_background(
                                task_id=task_id,
                                agent_service=agent_service,
                                task_owner_user_id=int(task.user_id),
                                expected_run_id=_task_run_id(task),
                                previous_task=previous_task,
                                pending_user_message=(
                                    None
                                    if posted
                                    else {
                                        "execution_message": user_message_for_llm,
                                        "display_message": display_user_message,
                                        "files": display_file_refs,
                                        "turn_id": turn_id,
                                    }
                                ),
                                delivery_turn_id=turn_id,
                                delivery_already_dispatched=posted,
                                delivery_websocket=(
                                    None
                                    if posted or suppress_delivery_ack
                                    else websocket
                                ),
                                delivery_client_message_id=(
                                    None
                                    if posted or suppress_delivery_ack
                                    else client_message_id
                                ),
                            )
                        )
                        background_task_manager.register_reserved_resume(
                            task_id, bg_task
                        )
                    except BaseException:
                        if bg_task is not None:
                            bg_task.cancel()
                        background_task_manager.release_resume_reservation(task_id)
                        raise

                    if posted:
                        await finish_delivery(True)
                    return
                elif task_uses_live_control and not has_continuation:
                    # Task is running but doesn't support continuation (shouldn't happen)
                    logger.error(
                        f"Task {task_id} is running but does not support continuation"
                    )
                    await manager.send_personal_message(
                        {
                            "type": "error",
                            "message": "Task does not support message continuation",
                        },
                        websocket,
                    )
                    await finish_delivery(
                        False, "Task does not support message continuation."
                    )
                    return
                else:
                    # New task/turn (PENDING/COMPLETED/FAILED/PAUSED), execute normally
                    if pause_accepted and task.status in {
                        TaskStatus.RUNNING,
                        TaskStatus.WAITING_FOR_USER,
                    }:
                        logger.info(
                            "Task %s has an accepted pause request; waiting for "
                            "the active run to persist its control state before "
                            "routing the follow-up message",
                            task_id,
                        )
                        await background_task_manager.wait_for_previous(task_id)
                        db.refresh(task)
                        if task.status in {
                            TaskStatus.RUNNING,
                            TaskStatus.WAITING_FOR_USER,
                        }:
                            await manager.broadcast_to_task(
                                {
                                    **_task_error_payload(
                                        db,
                                        task_id,
                                        (
                                            "Task pause is still being applied; "
                                            "please retry shortly."
                                        ),
                                        event_type="agent_error",
                                    ),
                                    "timestamp": datetime.now(timezone.utc).timestamp(),
                                },
                                task_id,
                            )
                            await finish_delivery(
                                False,
                                "Task pause is still being applied; please retry shortly.",
                            )
                            return
                        _clear_task_pause_accepted(task_id)

                    logger.info(
                        f"Task {task_id} starting new execution turn (status: {task.status.value})"
                    )

                    # The execution wrapper acquires the lease just before it
                    # starts running. Avoid acquiring it during setup so setup
                    # failures cannot leave the task locked.
                    if task.status != TaskStatus.RUNNING:
                        logger.info(
                            f"Sending task_info event for existing task {task_id}, status: {task.status.value}"
                        )

                        # Determine is_dag from agent config if agent_id exists
                        is_dag = None
                        if task.agent_id:
                            from ..models.agent import Agent

                            agent = (
                                db.query(Agent)
                                .filter(Agent.id == task.agent_id)
                                .first()
                            )
                            if agent:
                                is_dag = agent.execution_mode == "think"

                        (
                            model_id,
                            small_fast_model_id,
                            visual_model_id,
                            compact_model_id,
                        ) = _resolve_task_llm_ids(task, db)

                        task_event = create_stream_event(
                            "task_info",
                            task_id,
                            {
                                "id": task.id,
                                "title": task.title,
                                "description": task.description,
                                "status": task.status.value,
                                "model_id": model_id,
                                "small_fast_model_id": small_fast_model_id,
                                "visual_model_id": visual_model_id,
                                "compact_model_id": compact_model_id,
                                "model_name": task.model_name,
                                "small_fast_model_name": task.small_fast_model_name,
                                "visual_model_name": task.visual_model_name,
                                "compact_model_name": task.compact_model_name,
                                "execution_mode": task.execution_mode,
                                "agent_id": task.agent_id,
                                "agent_name": task.agent.name if task.agent else None,
                                "agent_logo_url": task.agent.logo_url
                                if task.agent
                                else None,
                                "is_dag": is_dag,
                                "created_at": safe_timestamp_to_unix(task.created_at)
                                if task.created_at
                                else None,
                                "updated_at": safe_timestamp_to_unix(task.updated_at)
                                if task.updated_at
                                else None,
                            },
                            task.created_at if task.created_at else None,
                        )
                        await manager.broadcast_to_task(task_event, task_id)
                        logger.info(f"task_info event sent for existing task {task_id}")

                    # Build context with vibe mode information if available
                    if hasattr(task, "execution_mode") and task.execution_mode:
                        context["execution_mode"] = task.execution_mode
                    if (
                        hasattr(task, "process_description")
                        and task.process_description
                    ):
                        context["process_description"] = task.process_description
                    if hasattr(task, "examples") and task.examples:
                        context["examples"] = task.examples

                    # WS builds the display/execution payload here and
                    # delegates the full new-turn transition to the
                    # shared orchestrator. ``begin_turn`` owns the
                    # atomic claim (status flip + input set + terminal-
                    # field reset), the transcript persist, the
                    # single-commit transaction, and the lease-aware bg
                    # schedule -- so WS and /v1 SDK use one turn-
                    # lifecycle state machine.
                    from ..services.task_orchestrator import (
                        TaskTurnError,
                        TaskTurnNotFoundError,
                        TaskTurnOrchestrator,
                        TaskTurnPayload,
                        TurnKind,
                    )

                    # Strip absolute filesystem paths before the row hits
                    # disk — the attachments column is exposed to historical-
                    # replay clients, so paths must not leak.
                    payload = TaskTurnPayload(
                        transcript_message=display_user_message,
                        execution_message=user_message_for_llm,
                        attachments=persisted_attachments or None,
                        turn_id=turn_id,
                    )
                    # WS path has these legal entries into begin_turn:
                    #   PENDING                  → CREATE
                    #   COMPLETED / FAILED       → APPEND
                    #   PAUSED + user message    → APPEND (new turn)
                    # WAITING_FOR_USER / RUNNING should have been intercepted
                    # by the live-control path above. Reaching this branch
                    # with either is an upstream-dispatch bug; surface it as
                    # an agent_error rather than silently letting begin_turn
                    # 409 on the wrong status.
                    if task.status == TaskStatus.PENDING:
                        turn_kind = TurnKind.CREATE
                        turn_force_fresh = False
                    elif task.status in (
                        TaskStatus.COMPLETED,
                        TaskStatus.FAILED,
                    ):
                        turn_kind = TurnKind.APPEND
                        turn_force_fresh = False
                    elif task.status == TaskStatus.PAUSED:
                        turn_kind = TurnKind.APPEND
                        turn_force_fresh = False
                    else:
                        logger.error(
                            f"WS schedule reached for task {task_id} with "
                            f"unexpected status={task.status}; expected "
                            "PENDING, PAUSED, or terminal. Live-control path "
                            "should have intercepted."
                        )
                        await manager.broadcast_to_task(
                            {
                                **_task_error_payload(
                                    db,
                                    task_id,
                                    "Internal dispatch error; please retry.",
                                    event_type="agent_error",
                                ),
                                "timestamp": datetime.now(timezone.utc).timestamp(),
                            },
                            task_id,
                        )
                        await finish_delivery(
                            False, "Internal dispatch error; please retry."
                        )
                        return

                    try:
                        await TaskTurnOrchestrator.begin_turn(
                            task_id=int(task.id),
                            # Owner, not the acting principal: ``task`` was
                            # already authorized above (admin bypass / owner
                            # check), and the turn must run as the task owner,
                            # not an admin acting on someone else's task.
                            task_owner_user_id=int(task.user_id),
                            # The acting principal (the admin when acting on
                            # another user's task) -- audit/logging only.
                            actor_user_id=int(user.id),
                            payload=payload,
                            kind=turn_kind,
                            force_fresh=turn_force_fresh,
                            context=context,
                        )
                        logger.info(f"Task {task_id} started in background")
                        await finish_delivery(True)
                    except TaskTurnNotFoundError:
                        # Task vanished or changed ownership between the
                        # resolve above and the atomic claim — surface it the
                        # same way as a busy refusal (no row was mutated).
                        logger.warning(
                            "begin_turn: task %s not found / not owned at claim",
                            task_id,
                        )
                        await manager.broadcast_to_task(
                            {
                                **_task_error_payload(
                                    db,
                                    task_id,
                                    "Task is no longer available.",
                                    event_type="agent_error",
                                ),
                                "timestamp": datetime.now(timezone.utc).timestamp(),
                            },
                            task_id,
                        )
                        await finish_delivery(False, "Task is no longer available.")
                    except TaskTurnError as busy_err:
                        # begin_turn's atomic transaction rolls back on
                        # bg_inflight / busy — neither the status flip
                        # nor the user message persists, so no transcript
                        # cleanup is needed here. The rejected-turn-leaves-
                        # no-side-effect contract makes the previous
                        # best-effort delete unnecessary.
                        logger.warning(
                            f"Refused to schedule bg for task {task_id}: "
                            f"{busy_err.reason}"
                        )
                        await manager.broadcast_to_task(
                            {
                                **_task_error_payload(
                                    db,
                                    task_id,
                                    (
                                        "Task is currently busy; please wait for "
                                        "the previous turn to finish before sending "
                                        "another message."
                                    ),
                                    event_type="agent_error",
                                ),
                                "timestamp": datetime.now(timezone.utc).timestamp(),
                            },
                            task_id,
                        )
                        await finish_delivery(
                            False,
                            "Task is currently busy; please wait for the previous "
                            "turn to finish before sending another message.",
                        )

            finally:
                db.close()

        except (ValueError, KeyError, TypeError) as e:
            # Data validation and format error
            message = f"Data validation error: {str(e)}"
            logger.error(f"Data validation error in agent execution: {e}")
            await finish_delivery_failure(message)
            timestamp = datetime.now(timezone.utc).timestamp()
            if authorized_task_id is not None:
                await manager.broadcast_to_task(
                    {
                        **_terminal_task_error_payload(authorized_task_id, message),
                        "timestamp": timestamp,
                    },
                    authorized_task_id,
                )
            else:
                await manager.send_personal_message(
                    {
                        "type": "error",
                        "message": message,
                        "timestamp": timestamp,
                    },
                    websocket,
                )
        except RuntimeError as e:
            # Runtime error
            message = f"Runtime error: {str(e)}"
            logger.error(f"Runtime error in agent execution: {e}")
            await finish_delivery_failure(message)
            timestamp = datetime.now(timezone.utc).timestamp()
            if authorized_task_id is not None:
                await manager.broadcast_to_task(
                    {
                        **_terminal_task_error_payload(authorized_task_id, message),
                        "timestamp": timestamp,
                    },
                    authorized_task_id,
                )
            else:
                await manager.send_personal_message(
                    {
                        "type": "error",
                        "message": message,
                        "timestamp": timestamp,
                    },
                    websocket,
                )
        except Exception as e:
            # Other unknown errors, re-raise
            logger.error(f"Unexpected error in agent execution: {e}")
            await finish_delivery_failure(str(e))
            raise

    except (ValueError, KeyError, TypeError) as e:
        # Message format error
        logger.error(f"Message format error: {e}")
        await finish_delivery_failure(f"Message format error: {str(e)}")
        await manager.send_personal_message(
            {"type": "error", "message": f"Message format error: {str(e)}"}, websocket
        )
    except (ConnectionError, WebSocketDisconnect) as e:
        # Connection error
        logger.error(f"Connection error handling chat message: {e}")
        raise
    except Exception as e:
        # Other errors, re-raise
        logger.error(f"Unexpected error handling chat message: {e}")
        await finish_delivery_failure(str(e))
        raise


async def handle_execute_task(
    websocket: WebSocket, task_id: int, message_data: dict
) -> None:
    """Handle task execution request"""
    try:
        user = message_data.get("user")
        authorized_task_id: int | None = None
        if not user:
            raise ValueError("User authentication required for task execution")

        # Send execution start confirmation
        await manager.send_personal_message(
            {
                "type": "execution_started",
                "task_id": task_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            websocket,
        )

        # Get database session
        from ..models.database import get_db
        from ..models.task import Task, TaskStatus
        from ..services.task_execution_context_service import (
            load_task_execution_recovery_state,
        )
        from .chat import get_agent_manager

        db_gen = get_db()
        db: Session = next(db_gen)

        try:
            # Get task - admin can access any task
            if user.is_admin:
                task = db.query(Task).filter(Task.id == task_id).first()
            else:
                task = (
                    db.query(Task)
                    .filter(Task.id == task_id, Task.user_id == user.id)
                    .first()
                )
            if not task:
                raise Exception(f"Task {task_id} not found or access denied")
            authorized_task_id = int(task.id)

            (
                model_id,
                small_fast_model_id,
                visual_model_id,
                compact_model_id,
            ) = _resolve_task_llm_ids(task, db)

            # Send task info event to update frontend state
            task_event = create_stream_event(
                "task_info",
                task_id,
                {
                    "id": task.id,
                    "title": task.title,
                    "description": task.description,
                    "status": task.status.value,
                    "model_id": model_id,
                    "small_fast_model_id": small_fast_model_id,
                    "visual_model_id": visual_model_id,
                    "compact_model_id": compact_model_id,
                    "model_name": task.model_name,
                    "small_fast_model_name": task.small_fast_model_name,
                    "visual_model_name": task.visual_model_name,
                    "compact_model_name": task.compact_model_name,
                    "execution_mode": task.execution_mode,
                    "agent_id": task.agent_id,
                    "agent_name": task.agent.name if task.agent else None,
                    "agent_logo_url": task.agent.logo_url if task.agent else None,
                    "created_at": safe_timestamp_to_unix(task.created_at)
                    if task.created_at
                    else None,
                    "updated_at": safe_timestamp_to_unix(task.updated_at)
                    if task.updated_at
                    else None,
                },
                task.created_at if task.created_at else None,
            )
            await manager.broadcast_to_task(task_event, task_id)

            # DAG plan-execute will automatically send user_message trace event

            # DAG plan-execute also sends trace events, but may not forward in real-time

            # Get agent and execute task
            from .chat import get_agent_manager

            agent_manager = get_agent_manager()
            agent_service = await agent_manager.get_agent_for_task(
                task_id, db, user=user, task_owner_user_id=int(task.user_id)
            )
            if hasattr(agent_service, "set_outbound_message_handler"):
                agent_service.set_outbound_message_handler(
                    make_agent_outbound_handler(task_id)
                )
            recovery_state = await load_task_execution_recovery_state(db, task_id)
            agent_service.set_execution_context_messages(
                recovery_state.get("messages", [])
            )
            agent_service.set_recovered_skill_context(
                recovery_state.get("skill_context")
            )

            # Set up user context as the task OWNER (runtime identity), not the
            # acting principal -- an admin executing another user's task must
            # run with the owner's identity. ``task`` is already loaded and
            # authorized above.
            with UserContext(int(task.user_id)), turn_execution_scope(task_id):
                # Build context with vibe mode information if available
                task_context = {}
                if hasattr(task, "execution_mode") and task.execution_mode:
                    task_context["execution_mode"] = task.execution_mode
                if hasattr(task, "process_description") and task.process_description:
                    task_context["process_description"] = task.process_description
                if hasattr(task, "examples") and task.examples:
                    task_context["examples"] = task.examples

                # Execute task with automatic token tracking
                result = await agent_manager.execute_task(
                    agent_service=agent_service,
                    task=str(task.description),
                    context=task_context,
                    task_id=str(task_id),
                    db_session=db,
                )

                # Update task status
                if result.get("success", False):
                    release_current_runner_task_lease_with_workforce_sync(
                        db, task_id, status=TaskStatus.COMPLETED
                    )
                else:
                    release_current_runner_task_lease_with_workforce_sync(
                        db, task_id, status=TaskStatus.FAILED
                    )
                db.refresh(task)

                # Send task completion event (don't duplicate result as trace system already sent)

            # Workspace cleanup now only happens on task deletion, so users can view result files

            # Note: trace_task_completion is handled by handle_chat_message to avoid duplicates

            # Extract file output info
            file_outputs, path_to_file_id = _normalize_task_file_outputs(
                db,
                task,
                result.get("file_outputs", []),
            )
            result["output"] = _rewrite_file_links_to_file_id(
                result.get("output", ""),
                path_to_file_id,
            )
            result["output"] = reconcile_assistant_file_references(
                db,
                task_id=int(task_id),
                user_id=int(task.user_id),
                content=result["output"],
            )

            # Send task completion event (don't duplicate result as trace system already sent)
            await manager.broadcast_to_task(
                {
                    "type": "task_completed",
                    "task": {
                        "id": task.id,
                        "title": task.title,
                        "status": task.status.value,
                        "description": task.description,
                    },
                    "success": result.get("success", False),
                    "run_id": _task_run_id(task),
                    "state_version": int(task.state_version or 0),
                    "control_state": _task_control_state_value(task) or "idle",
                    "status": task.status.value,
                    "result": result.get("output", ""),
                    "output": result.get("output", ""),
                    # Same coded-reason forwarding as the primary completion
                    # broadcast, so a coded failure on this path (e.g. a mid-run
                    # quota interrupt) still reaches the app-layer dialog.
                    "error_code": result.get("error_code"),
                    "error_details": result.get("error_details"),
                    "chat_response": result.get("chat_response"),
                    "metadata": result.get("metadata", {}),
                    "file_outputs": file_outputs,  # Add file output info
                    "timestamp": datetime.now(timezone.utc).timestamp(),
                },
                task_id,
            )

        finally:
            db.close()

    except (ValueError, KeyError, TypeError) as e:
        # Data validation and format error
        message = f"Data validation error: {str(e)}"
        logger.error(f"Data validation error in task execution: {e}")
        timestamp = datetime.now(timezone.utc).isoformat()
        if authorized_task_id is not None:
            await manager.broadcast_to_task(
                {
                    **_terminal_task_error_payload(authorized_task_id, message),
                    "timestamp": timestamp,
                },
                authorized_task_id,
            )
        else:
            await manager.send_personal_message(
                {
                    "type": "error",
                    "message": message,
                    "timestamp": timestamp,
                },
                websocket,
            )
    except RuntimeError as e:
        # Runtime error
        message = f"Runtime error: {str(e)}"
        logger.error(f"Runtime error in task execution: {e}")
        timestamp = datetime.now(timezone.utc).isoformat()
        if authorized_task_id is not None:
            await manager.broadcast_to_task(
                {
                    **_terminal_task_error_payload(authorized_task_id, message),
                    "timestamp": timestamp,
                },
                authorized_task_id,
            )
        else:
            await manager.send_personal_message(
                {
                    "type": "error",
                    "message": message,
                    "timestamp": timestamp,
                },
                websocket,
            )
    except Exception as e:
        # Other unknown errors, re-raise
        logger.error(f"Unexpected error in task execution: {e}")
        raise


async def send_historical_data_as_stream(
    websocket: WebSocket, task_id: int, user: User
) -> None:
    """Send historical data as stream messages - using unified trace event format"""
    try:
        # Load historical data directly from database
        from ..models.agent import Agent
        from ..models.database import get_db
        from ..models.task import Task, TaskStatus, TraceEvent

        db_gen = get_db()
        db = next(db_gen)

        try:
            # Get task basic info
            task = db.query(Task).filter(Task.id == task_id).first()
            if not task:
                logger.warning(f"Task {task_id} not found")
                return

            # Verify user permissions
            if not task.user_id:
                logger.warning(f"Task {task_id} has no user association")
                return

            # Verify user permissions - admin can access any task
            if not user.is_admin and task.user_id != int(user.id):
                logger.warning(
                    f"User {user.id} attempted to access task {task_id} belonging to user {task.user_id}"
                )
                return

            if mark_task_paused_if_stale(db, task):
                db.refresh(task)
                if sync_workforce_run_status(db, task, task.status):
                    db.commit()

            max_trace_event_id = (
                db.query(func.max(TraceEvent.id))
                .filter(
                    TraceEvent.task_id == task_id,
                    TraceEvent.build_id.is_(None),
                )
                .scalar()
                or 0
            )
            max_chat_message_id = (
                db.query(func.max(TaskChatMessage.id))
                .filter(TaskChatMessage.task_id == task_id)
                .scalar()
                or 0
            )
            cache_key = web_task_history_key(task_id)
            task_updated_at = cache_version_token(task.updated_at)
            cached = cache_get(cache_key)
            if (
                isinstance(cached, dict)
                and cached.get("updated_at") == task_updated_at
                and cached.get("max_trace_event_id") == int(max_trace_event_id)
                and cached.get("max_chat_message_id") == int(max_chat_message_id)
                and isinstance(cached.get("events"), list)
            ):
                for cached_event in cached["events"]:
                    if isinstance(cached_event, dict):
                        await manager.send_personal_message(cached_event, websocket)
                return

            cached_stream_events: list[dict[str, Any]] = []

            # Determine is_dag from agent config if agent_id exists
            is_dag = None
            if task.agent_id:
                agent = db.query(Agent).filter(Agent.id == task.agent_id).first()
                if agent:
                    is_dag = agent.execution_mode == "think"

            (
                model_id,
                small_fast_model_id,
                visual_model_id,
                compact_model_id,
            ) = _resolve_task_llm_ids(task, db)
            waiting_question = None
            waiting_interactions = None
            if task.status == TaskStatus.WAITING_FOR_USER:
                waiting_question, waiting_interactions = get_latest_waiting_question(
                    db, task_id
                )

            # Send task basic info
            task_event = create_stream_event(
                "task_info",
                task_id,
                {
                    "id": task.id,
                    "title": task.title,
                    "description": task.description,
                    "status": task.status.value,
                    "model_id": model_id,
                    "small_fast_model_id": small_fast_model_id,
                    "visual_model_id": visual_model_id,
                    "compact_model_id": compact_model_id,
                    "model_name": task.model_name,
                    "small_fast_model_name": task.small_fast_model_name,
                    "visual_model_name": task.visual_model_name,
                    "compact_model_name": task.compact_model_name,
                    "execution_mode": task.execution_mode,
                    "agent_id": task.agent_id,
                    "agent_name": task.agent.name if task.agent else None,
                    "agent_logo_url": task.agent.logo_url if task.agent else None,
                    "is_dag": is_dag,
                    "waiting_question": waiting_question,
                    "waiting_interactions": waiting_interactions,
                    "created_at": safe_timestamp_to_unix(task.created_at)
                    if task.created_at
                    else None,
                    "updated_at": safe_timestamp_to_unix(task.updated_at)
                    if task.updated_at
                    else None,
                },
                task.created_at if task.created_at else None,
            )
            await manager.send_personal_message(task_event, websocket)
            cached_stream_events.append(task_event)

            # Get unified trace events (only VIBE phase, exclude BUILD phase)
            trace_events = (
                db.query(TraceEvent)
                .filter(
                    TraceEvent.task_id == task_id,
                    TraceEvent.build_id.is_(None),  # ← Only get VIBE events
                    # Agent checkpoints are persisted as trace rows for
                    # resume/recovery, but they are internal snapshots and can
                    # be megabytes each. Filtering them in SQL avoids loading
                    # hundreds of large JSON blobs just to discard them below.
                    TraceEvent.event_type != CHECKPOINT_EVENT_TYPE_NAME,
                )
                .order_by(TraceEvent.timestamp, TraceEvent.id)
                .all()
            )

            # DAG execution info is now directly provided by DAG plan-execute trace events

            # DAG execution events are now directly sent by DAG plan-execute, no need to rebuild

            # DAG step info is now directly provided by DAG plan-execute trace events

            # DAG step rebuild code removed, DAG plan-execute now directly sends trace events

            # Merge all time-sensitive events and sort by timestamp
            historical_events: list[dict[str, Any]] = []

            historical_path_to_file_id: Dict[str, str] = {}
            normalized_trace_data_by_event_id: Dict[str, Any] = {}
            # Dedup key for "is this chat_messages row already covered by a
            # trace event?". Includes an attachment fingerprint so two
            # user turns with the same typed text but different uploaded
            # files no longer collapse into one — the second row used to
            # be dropped and its file chips disappeared on reload.
            trace_message_keys: set[tuple[str, str, str]] = set()
            trace_user_turn_ids: set[str] = set()
            seen_trace_user_turn_ids: set[str] = set()

            for trace_event in trace_events:
                normalized_event_data = trace_event.data
                if isinstance(trace_event.data, dict):
                    normalized_event_data = dict(trace_event.data)
                    if _is_audit_only_trace_data(normalized_event_data):
                        normalized_trace_data_by_event_id[str(trace_event.event_id)] = (
                            normalized_event_data
                        )
                        continue
                    normalized_outputs, path_to_file_id = _normalize_task_file_outputs(
                        db,
                        task,
                        normalized_event_data.get("file_outputs", []),
                    )
                    if normalized_outputs:
                        normalized_event_data["file_outputs"] = normalized_outputs
                    if path_to_file_id:
                        historical_path_to_file_id.update(path_to_file_id)
                normalized_trace_data_by_event_id[str(trace_event.event_id)] = (
                    normalized_event_data
                )
                if isinstance(normalized_event_data, dict):
                    content = normalized_event_data.get(
                        "message"
                    ) or normalized_event_data.get("content")
                    event_attachments = normalized_event_data.get(
                        "files"
                    ) or normalized_event_data.get("attachments")
                    attachment_key = _attachment_fingerprint(event_attachments)
                    if trace_event.event_type == "user_message":
                        trace_turn_id = _trace_user_message_turn_id(
                            "user_message", normalized_event_data
                        )
                        if trace_turn_id:
                            trace_user_turn_ids.add(trace_turn_id)
                        elif isinstance(content, str) and content.strip():
                            trace_message_keys.add(
                                ("user", content.strip(), attachment_key)
                            )
                    elif (
                        trace_event.event_type in {"agent_message", "ai_message"}
                        and isinstance(content, str)
                        and content.strip()
                    ):
                        trace_message_keys.add(
                            ("assistant", content.strip(), attachment_key)
                        )

            for trace_event in trace_events:
                normalized_event_data = normalized_trace_data_by_event_id.get(
                    str(trace_event.event_id), trace_event.data
                )
                if _is_audit_only_trace_data(normalized_event_data):
                    continue
                if _is_duplicate_user_message_turn(
                    str(trace_event.event_type),
                    normalized_event_data,
                    seen_trace_user_turn_ids,
                ):
                    continue
                if _is_agent_checkpoint_data(normalized_event_data):
                    continue
                if historical_path_to_file_id and isinstance(
                    normalized_event_data, dict
                ):
                    normalized_event_data = _rewrite_links_in_payload(
                        normalized_event_data,
                        historical_path_to_file_id,
                    )
                public_event_type, public_event_data = normalize_public_trace_event(
                    str(trace_event.event_type),
                    normalized_event_data,
                )
                historical_events.append(
                    {
                        "type": "trace_event",
                        "data": {
                            "event_id": trace_event.event_id,
                            "event_type": public_event_type,
                            "step_id": trace_event.step_id,
                            "parent_event_id": trace_event.parent_event_id,
                            "data": public_event_data,
                        },
                        "timestamp": safe_timestamp_to_unix(trace_event.timestamp)
                        if trace_event.timestamp
                        else None,
                    }
                )

            chat_messages = (
                db.query(TaskChatMessage)
                .filter(TaskChatMessage.task_id == task_id)
                .order_by(TaskChatMessage.created_at, TaskChatMessage.id)
                .all()
            )
            file_reference_records = load_assistant_file_reference_records(
                db,
                task_id=int(task_id),
                user_id=int(task.user_id),
            )
            for chat_message in chat_messages:
                role = str(chat_message.role)
                content = str(chat_message.content or "").strip()
                if role == "assistant":
                    content = reconcile_assistant_file_references(
                        db,
                        task_id=int(task_id),
                        user_id=int(task.user_id),
                        content=content,
                        records=file_reference_records,
                    )
                # Read attachments off the row so file-only turns (empty
                # content + non-empty attachments) survive replay and so the
                # chip metadata reaches the synthesized user_message event.
                _attachments_raw = chat_message.attachments
                row_attachments: Optional[list] = (
                    _attachments_raw
                    if isinstance(_attachments_raw, list) and _attachments_raw
                    else None
                )
                # Drop only when there's nothing to render — empty text *and*
                # no attachments. A row with attachments but no text is a real
                # turn (user uploaded files without typing) and must be kept.
                if not content and not row_attachments:
                    continue

                if role == "user":
                    row_turn_id = getattr(chat_message, "turn_id", None)
                    if isinstance(row_turn_id, str):
                        row_turn_id = row_turn_id.strip() or None
                    else:
                        row_turn_id = None

                    if row_turn_id:
                        if row_turn_id in trace_user_turn_ids:
                            continue
                    elif (
                        content
                        and (role, content, _attachment_fingerprint(row_attachments))
                        in trace_message_keys
                    ):
                        continue

                    event_type = "user_message"
                    data: dict[str, Any] = {"message": content, "content": content}
                    if row_turn_id:
                        data["turn_id"] = row_turn_id
                    if row_attachments:
                        # Surface the persisted chip payload at the top level
                        # so the frontend user-message renderer can show
                        # clickable file chips on reload, matching the live
                        # event shape emitted by the agent tracing callback.
                        data["files"] = row_attachments
                        data["attachments"] = row_attachments
                elif role == "assistant":
                    if (
                        content
                        and (role, content, _attachment_fingerprint(row_attachments))
                        in trace_message_keys
                    ):
                        continue
                    interactions = chat_message.interactions
                    data = {
                        "message": content,
                        "content": content,
                        "role": "assistant",
                        "source": "chat_history",
                        "display": "chat",
                        # Historical assistant questions are transcript entries.
                        # The current WAITING_FOR_USER state is reasserted separately
                        # after replay, so old questions must not flip status back.
                        "expect_response": False,
                        "visible": True,
                    }
                    if isinstance(interactions, list):
                        data["metadata"] = {"interactions": interactions}
                    event_type = "agent_message"
                else:
                    continue

                historical_events.append(
                    {
                        "type": "trace_event",
                        "data": {
                            "event_id": f"chat_message_{chat_message.id}",
                            "event_type": event_type,
                            "step_id": None,
                            "parent_event_id": None,
                            "data": data,
                        },
                        "timestamp": chat_message.created_at,
                    }
                )

            # Sort historical events by timestamp
            min_datetime = datetime.min.replace(tzinfo=timezone.utc)

            def sort_key(x: dict[str, Any]) -> datetime:
                timestamp = x["timestamp"]
                if isinstance(timestamp, datetime):
                    if timestamp.tzinfo is None:
                        return timestamp.replace(tzinfo=timezone.utc)
                    return timestamp
                if isinstance(timestamp, (int, float)):
                    return datetime.fromtimestamp(timestamp, timezone.utc)
                return min_datetime

            historical_events.sort(key=sort_key)

            # Filter dag_plan_end events: keep only the latest one
            # This is because continuation generates new plans, we don't want old plans to overwrite new ones
            dag_plan_end_events = []
            other_events = []
            for event in historical_events:
                if event["type"] == "trace_event":
                    event_data = event["data"]
                    if isinstance(event_data, dict):
                        event_type = event_data.get("event_type", "")
                        if event_type == "dag_plan_end":
                            dag_plan_end_events.append(event)
                            continue
                other_events.append(event)

            # Keep only the latest dag_plan_end event
            if dag_plan_end_events:
                latest_plan_event = dag_plan_end_events[
                    -1
                ]  # Already sorted by time, last one is latest
                logger.info(
                    f"Filtered {len(dag_plan_end_events) - 1} old dag_plan_end events from history"
                )
                other_events.append(latest_plan_event)

            # Send sorted historical events
            for event in other_events:
                if event["type"] == "trace_event":
                    # For trace events, send directly in unified format
                    event_data = event["data"]
                    if not isinstance(event_data, dict):
                        continue

                    event_timestamp = event["timestamp"]
                    timestamp_val = safe_timestamp_to_unix(event_timestamp)

                    stream_event = {
                        "type": "trace_event",
                        "event_id": str(event_data.get("event_id", "")),
                        "event_type": str(event_data.get("event_type", "")),
                        "task_id": task_id,
                        "timestamp": int(timestamp_val),
                        "data": dict(event_data.get("data", {})),
                    }

                    # Add step_id at the top level if present (consistent with WebSocketTraceHandler)
                    if event_data.get("step_id"):
                        stream_event["step_id"] = str(event_data["step_id"])
                    await manager.send_personal_message(stream_event, websocket)
                    cached_stream_events.append(stream_event)
                else:
                    # For other events, use original format
                    event_data = event["data"]
                    if isinstance(event_data, dict):
                        event_obj = create_stream_event(
                            str(event["type"]),
                            task_id,
                            event_data,
                            event["timestamp"],
                        )
                        await manager.send_personal_message(event_obj, websocket)
                        cached_stream_events.append(event_obj)

            # Send historical data completion marker
            completion_event = create_stream_event(
                "historical_data_complete",
                task_id,
                {
                    "message": "Historical data loading complete",
                    "total_trace_events": len(trace_events),
                },
            )
            await manager.send_personal_message(completion_event, websocket)
            cached_stream_events.append(completion_event)

            # Historical trace replay can end with an in-flight event from before a
            # crash/restart, such as llm_call_start. Re-assert the current DB task
            # state after replay so stale running trace events do not keep the UI in
            # a running state.
            if task.status in {TaskStatus.PAUSED, TaskStatus.WAITING_FOR_USER}:
                event_type = (
                    "task_waiting_for_user"
                    if task.status == TaskStatus.WAITING_FOR_USER
                    else "task_paused"
                )
                question_message = None
                question_interactions = None
                if task.status == TaskStatus.WAITING_FOR_USER:
                    question_message, question_interactions = (
                        get_latest_waiting_question(db, task_id)
                    )

                message = (
                    question_message or "Task waiting for user response"
                    if task.status == TaskStatus.WAITING_FOR_USER
                    else "Task paused"
                )
                status_event = {
                    "type": event_type,
                    "task_id": task_id,
                    "message": message,
                    "timestamp": datetime.now(timezone.utc).timestamp(),
                    **task_control_snapshot(task).as_dict(),
                }
                if question_message:
                    status_event["question"] = question_message
                if isinstance(question_interactions, list):
                    status_event["interactions"] = question_interactions
                await manager.send_personal_message(status_event, websocket)
                cached_stream_events.append(status_event)

            cache_set(
                cache_key,
                {
                    "updated_at": task_updated_at,
                    "max_trace_event_id": int(max_trace_event_id),
                    "max_chat_message_id": int(max_chat_message_id),
                    "events": cached_stream_events,
                },
                ttl_seconds=task_cache_ttl_seconds(),
            )

        except (ValueError, KeyError, TypeError) as e:
            # Data format error
            logger.error(
                f"Data format error loading historical data for task {task_id}: {e}"
            )
            raise
        except RuntimeError as e:
            # Runtime error
            logger.error(
                f"Runtime error loading historical data for task {task_id}: {e}"
            )
            raise
        except Exception as e:
            # Other unknown errors, re-raise
            logger.error(
                f"Unexpected error loading historical data for task {task_id}: {e}"
            )
            raise
        finally:
            db.close()

    except (ValueError, KeyError, TypeError) as e:
        # Data format error
        logger.error(f"Data format error sending historical data stream: {e}")
        error_event = create_stream_event(
            "error",
            task_id,
            {
                "message": f"Data format error: {str(e)}",
            },
        )
        await manager.send_personal_message(error_event, websocket)
        raise
    except (ConnectionError, WebSocketDisconnect) as e:
        # Connection error
        logger.error(f"Connection error sending historical data stream: {e}")
        raise
    except Exception as e:
        # Other unknown errors, re-raise
        logger.error(f"Unexpected error sending historical data stream: {e}")
        raise


async def handle_status_request(websocket: WebSocket, task_id: int, user: User) -> None:
    """Handle status request - send historical data as stream messages"""
    await send_historical_data_as_stream(websocket, task_id, user)


@ws_router.websocket("/ws/chat/{task_id}")
async def websocket_chat_endpoint(
    websocket: WebSocket,
    task_id: int,
    token: Optional[str] = Query(None, description="Authentication token"),
) -> None:
    """WebSocket unified endpoint - handle chat, execution status, and DAG intervention"""
    # Verify user identity
    user = await get_authenticated_user(websocket, token)
    if not user:
        await websocket.close(code=4001, reason="Authentication required")
        return

    await manager.connect(websocket, task_id)

    try:
        # Send initial state
        await handle_status_request(websocket, task_id, user)

        while True:
            # Receive client message
            data = await websocket.receive_text()
            logger.info(
                f"📨 Received WebSocket message for task {task_id}: {data[:200]}"
            )  # Log first 200 chars
            message_data = json.loads(data)
            logger.info(f"📋 Parsed message type: {message_data.get('type')}")

            # Add user info to message data
            message_data["user_id"] = user.id
            message_data["user"] = user

            if message_data.get("type") == "chat":
                await handle_chat_message(websocket, task_id, message_data)
            elif message_data.get("type") == "execute_task":
                await handle_execute_task(websocket, task_id, message_data)
            elif message_data.get("type") == "intervention":
                await handle_intervention(websocket, task_id, message_data)
            elif message_data.get("type") == "status_request":
                await handle_status_request(websocket, task_id, user)
            elif message_data.get("type") == "pause_task":
                logger.info(f"📥 Received pause_task message for task {task_id}")
                await handle_pause_task(websocket, task_id, message_data)
            elif message_data.get("type") == "resume_task":
                await handle_resume_task(websocket, task_id, message_data)
            else:
                await manager.send_personal_message(
                    {"type": "error", "message": "Unknown message type"}, websocket
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket, task_id)
    except (ConnectionError, RuntimeError) as e:
        # Connection error
        logger.error(f"Connection error in WebSocket: {e}")
        manager.disconnect(websocket, task_id)
    except Exception as e:
        # Other errors, re-raise
        logger.error(f"Unexpected error in WebSocket: {e}")
        manager.disconnect(websocket, task_id)
        raise


async def handle_intervention(
    websocket: WebSocket, task_id: int, message_data: dict
) -> None:
    """Handle manual intervention"""
    try:
        intervention_data = {
            "step_id": message_data.get("step_id"),
            "action": message_data.get("action"),
            "data": message_data.get("data", {}),
        }

        # Simulate handling intervention
        await manager.broadcast_to_task(
            {
                "type": "intervention_processed",
                "message": f"Manual intervention processed: {intervention_data['action']}",
                "intervention_id": intervention_data["step_id"],
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),  # Send UTC timestamp directly
            },
            task_id,
        )

    except (ValueError, KeyError, TypeError) as e:
        # Data validation error
        logger.error(f"Data validation error in intervention: {e}")
        await manager.send_personal_message(
            {"type": "error", "message": f"Data validation error: {str(e)}"}, websocket
        )
    except RuntimeError as e:
        # Runtime error
        logger.error(f"Runtime error in intervention: {e}")
        await manager.send_personal_message(
            {"type": "error", "message": f"Runtime error: {str(e)}"}, websocket
        )
    except Exception as e:
        # Other errors, re-raise
        logger.error(f"Unexpected error in intervention: {e}")
        raise


async def handle_pause_task(
    websocket: WebSocket, task_id: int, message_data: dict
) -> None:
    """Persist a pause request; the lease owner applies it in command order."""

    try:
        enqueued = await _enqueue_websocket_task_command(
            task_id=task_id,
            message_data=message_data,
            kind=TaskCommandKind.PAUSE,
            command_id=_client_message_id(message_data.get("command_id")),
        )
    except (PermissionError, ValueError) as exc:
        await manager.send_personal_message(
            {"type": "error", "message": str(exc)}, websocket
        )
        return
    assert enqueued is not None
    if not enqueued.payload_matches:
        await manager.send_personal_message(
            {
                "type": "error",
                "message": "Command id was already used for a different request.",
            },
            websocket,
        )
        return
    await manager.send_personal_message(
        {
            "type": "task_command_accepted",
            "task_id": task_id,
            "command_id": enqueued.client_command_id,
            "command": TaskCommandKind.PAUSE.value,
        },
        websocket,
    )
    await dispatch_task_command_promptly(
        execute_durable_task_command,
        command_db_id=enqueued.command_id,
    )


async def _handle_pause_task_unserialized(
    websocket: WebSocket, task_id: int, message_data: dict
) -> None:
    """Handle task pause request"""
    db: Session | None = None
    try:
        logger.info(f"🔘 handle_pause_task called for task {task_id}")
        user = message_data.get("user")
        if not user:
            logger.error("No user in message_data")
            raise ValueError("User authentication required for task pause")

        logger.info(f"User {user.id} authenticated for pause")

        # Get database session
        from ..models.database import get_db

        db_gen = get_db()
        db = next(db_gen)

        # Authorize the task BEFORE building any runtime: an admin may pause any
        # task; a non-admin only their own. Loading the task here also gives us
        # the OWNER, so the agent runtime runs as the owner, not the actor.
        from ..models.task import Task as _PauseTask

        if user.is_admin:
            task = db.query(_PauseTask).filter(_PauseTask.id == task_id).first()
        else:
            task = (
                db.query(_PauseTask)
                .filter(_PauseTask.id == task_id, _PauseTask.user_id == int(user.id))
                .first()
            )
        if task is None:
            logger.warning(
                "pause: task %s not found or not owned by user %s", task_id, user.id
            )
            raise ValueError(f"Access denied: task {task_id} is not available")
        task_owner_user_id = int(task.user_id)

        # Get agent service (as the task owner)
        from .chat import get_agent_manager

        logger.info(f"Getting agent service for task {task_id}")
        agent_service = await get_agent_manager().get_agent_for_task(
            task_id, db, user=user, task_owner_user_id=task_owner_user_id
        )
        logger.info(f"Agent service obtained: {type(agent_service).__name__}")

        # Check if agent supports pause functionality
        if hasattr(agent_service, "pause_execution"):
            logger.info("Agent supports pause_execution, calling it...")
            pause_result = await agent_service.pause_execution()
            if pause_result is False:
                message_data["_durable_command_error"] = (
                    "No live execution found to pause"
                )
                await manager.send_personal_message(
                    _task_error_payload(
                        db,
                        task_id,
                        "No live execution found to pause",
                    ),
                    websocket,
                )
                logger.warning(f"No live execution found to pause for task {task_id}")
                return
            logger.info("Agent pause_execution completed")
            db.refresh(task)
            if task.status != TaskStatus.RUNNING:
                message_data["_durable_command_error"] = (
                    "Task finished before the pause request was applied"
                )
                await manager.send_personal_message(
                    _task_error_payload(
                        db,
                        task_id,
                        "Task finished before the pause request was applied",
                    ),
                    websocket,
                )
                return
            apply_task_control_transition(
                task,
                TaskControlState.PAUSE_REQUESTED,
                expected_run_id=_task_run_id(task),
            )
            db.commit()
            _mark_task_pause_accepted(task_id)

            # This confirms only that the control request was accepted. The
            # frontend deliberately waits for the later durable ``task_info``
            # PAUSED state before changing its pause UI; treating this event
            # as ``task_paused`` would reintroduce the optimistic-state bug.
            await manager.broadcast_to_task(
                {
                    "type": "task_pause_requested",
                    "task_id": task_id,
                    "message": "Task pause requested",
                    "timestamp": datetime.now(timezone.utc).timestamp(),
                },
                task_id,
            )
            logger.info(f"Task {task_id} pause requested successfully")
        else:
            # If pause not supported, send error message
            message_data["_durable_command_error"] = (
                "Current agent does not support pause functionality"
            )
            await manager.send_personal_message(
                _task_error_payload(
                    db,
                    task_id,
                    "Current agent does not support pause functionality",
                ),
                websocket,
            )
            logger.warning(
                f"Agent for task {task_id} does not support pause functionality"
            )

    except (ValueError, KeyError, TypeError) as e:
        # Data validation error
        message_data["_durable_command_error"] = str(e)
        logger.error(f"Data validation error pausing task {task_id}: {e}")
        await manager.send_personal_message(
            {"type": "error", "message": f"Data validation error: {str(e)}"}, websocket
        )
    except RuntimeError as e:
        # Runtime error
        logger.error(f"Runtime error pausing task {task_id}: {e}")
        await manager.send_personal_message(
            {"type": "error", "message": f"Runtime error: {str(e)}"}, websocket
        )
        raise
    except Exception as e:
        # Other errors, re-raise
        logger.error(f"Unexpected error pausing task {task_id}: {e}")
        raise
    finally:
        if db is not None:
            db.close()


async def handle_resume_task(
    websocket: WebSocket, task_id: int, message_data: dict
) -> None:
    """Persist a resume request; a worker applies it in command order."""

    try:
        enqueued = await _enqueue_websocket_task_command(
            task_id=task_id,
            message_data=message_data,
            kind=TaskCommandKind.RESUME,
            command_id=_client_message_id(message_data.get("command_id")),
        )
    except (PermissionError, ValueError) as exc:
        await manager.send_personal_message(
            {"type": "error", "message": str(exc)}, websocket
        )
        return
    assert enqueued is not None
    if not enqueued.payload_matches:
        await manager.send_personal_message(
            {
                "type": "error",
                "message": "Command id was already used for a different request.",
            },
            websocket,
        )
        return
    await manager.send_personal_message(
        {
            "type": "task_command_accepted",
            "task_id": task_id,
            "command_id": enqueued.client_command_id,
            "command": TaskCommandKind.RESUME.value,
        },
        websocket,
    )
    await dispatch_task_command_promptly(
        execute_durable_task_command,
        command_db_id=enqueued.command_id,
    )


async def _handle_resume_task_unserialized(
    websocket: WebSocket, task_id: int, message_data: dict
) -> None:
    """Handle task resume request"""
    try:
        user = message_data.get("user")
        if not user:
            raise ValueError("User authentication required for task resume")

        # Get database session
        from ..models.database import get_db

        db_gen = get_db()
        db = next(db_gen)

        from ..models.task import Task

        task: Any | None = None
        agent_service: Any = None
        try:
            # Authorize BEFORE building any runtime: an admin may resume any
            # task, a non-admin only their own. Loading the task here also
            # yields the OWNER, so the agent runs as the owner (not the actor)
            # and nothing is built / cached for an unauthorized request.
            if user.is_admin:
                task = db.query(Task).filter(Task.id == task_id).first()
            else:
                task = (
                    db.query(Task)
                    .filter(Task.id == task_id, Task.user_id == int(user.id))
                    .first()
                )
            if task is None:
                logger.warning(
                    f"Task {task_id} not found or access denied for user {user.id}"
                )
            else:
                from .chat import get_agent_manager

                agent_service = await get_agent_manager().get_agent_for_task(
                    task_id,
                    db,
                    user=user,
                    task_owner_user_id=int(task.user_id),
                )
        finally:
            db.close()

        if task is None:
            message_data["_durable_command_error"] = "Task not found or access denied"
            await manager.send_personal_message(
                {"type": "error", "message": "Task not found or access denied"},
                websocket,
            )
            return

        resume_control_state = task_control_snapshot(task).as_dict()
        if getattr(agent_service, "supports_live_control", lambda: False)():
            if task.status not in {TaskStatus.PAUSED, TaskStatus.WAITING_FOR_USER}:
                message_data["_durable_command_error"] = (
                    "Task is not paused and cannot be resumed."
                )
                await manager.send_personal_message(
                    {
                        "type": "error",
                        "message": "Task is not paused and cannot be resumed.",
                        "task": {"id": task_id, **resume_control_state},
                    },
                    websocket,
                )
                return
            if not background_task_manager.reserve_resume(task_id):
                await manager.send_personal_message(
                    {
                        "type": "error",
                        "message": "Task resume is already in progress.",
                        "task": {"id": task_id, **resume_control_state},
                    },
                    websocket,
                )
                return
            resume_snapshot: Any | None = None
            bg_task: asyncio.Task[None] | None = None
            try:
                resume_snapshot = await task_execution_controller.transition(
                    task_id,
                    TaskControlState.RESUME_REQUESTED,
                    expected_run_id=_task_run_id(task),
                )
                previous_task = background_task_manager.running_tasks.get(task_id)
                bg_task = asyncio.create_task(
                    execute_resume_background(
                        task_id=task_id,
                        agent_service=agent_service,
                        task_owner_user_id=int(task.user_id),
                        expected_run_id=resume_snapshot.run_id,
                        previous_task=previous_task,
                    )
                )
                background_task_manager.register_reserved_resume(task_id, bg_task)
            except BaseException:
                if bg_task is not None:
                    bg_task.cancel()
                background_task_manager.release_resume_reservation(task_id)
                if resume_snapshot is not None:
                    await asyncio.shield(
                        task_execution_controller.transition(
                            task_id,
                            (
                                TaskControlState.WAITING_FOR_USER
                                if resume_snapshot.status == TaskStatus.WAITING_FOR_USER
                                else TaskControlState.PAUSED
                            ),
                            expected_run_id=resume_snapshot.run_id,
                        )
                    )
                raise
            logger.info(f"Task {task_id} v2 resume scheduled")
            return

        # Check if agent supports resume functionality
        if hasattr(agent_service, "resume_execution"):
            resume_snapshot = await task_execution_controller.transition(
                task_id,
                TaskControlState.RESUME_REQUESTED,
                expected_run_id=_task_run_id(task),
            )
            await agent_service.resume_execution()
            await task_execution_controller.transition(
                task_id,
                TaskControlState.RUNNING,
                status=TaskStatus.RUNNING,
                expected_run_id=resume_snapshot.run_id,
            )

            # Send resume confirmation
            await manager.broadcast_to_task(
                {
                    "type": "task_resumed",
                    "task_id": task_id,
                    "message": "Task resumed",
                    "timestamp": datetime.now(timezone.utc).timestamp(),
                },
                task_id,
            )
            logger.info(f"Task {task_id} resumed successfully")
        else:
            # If resume not supported, send error message
            message_data["_durable_command_error"] = (
                "Current agent does not support resume functionality"
            )
            await manager.send_personal_message(
                {
                    "type": "error",
                    "message": "Current agent does not support resume functionality",
                },
                websocket,
            )
            logger.warning(
                f"Agent for task {task_id} does not support resume functionality"
            )

    except (ValueError, KeyError, TypeError) as e:
        # Data validation error
        message_data["_durable_command_error"] = str(e)
        logger.error(f"Data validation error resuming task {task_id}: {e}")
        await manager.send_personal_message(
            {"type": "error", "message": f"Data validation error: {str(e)}"}, websocket
        )
    except RuntimeError as e:
        # Runtime error
        logger.error(f"Runtime error resuming task {task_id}: {e}")
        await manager.send_personal_message(
            {"type": "error", "message": f"Runtime error: {str(e)}"}, websocket
        )
        raise
    except Exception as e:
        # Other errors, re-raise
        logger.error(f"Unexpected error resuming task {task_id}: {e}")
        raise


class _DiscardingCommandWebSocket:
    """Minimal sink used when a recovered command has no originating socket."""

    async def send_text(self, _message: str) -> None:
        return None


def _load_command_actor(actor_user_id: int | None) -> User:
    if actor_user_id is None:
        raise ValueError("Task command has no actor")
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        user = db.query(User).filter(User.id == actor_user_id).first()
        if user is None:
            raise ValueError(f"Task command actor {actor_user_id} no longer exists")
        db.expunge(user)
        return user


async def _execute_durable_task_command(
    command: ClaimedTaskCommand,
) -> dict[str, Any] | None:
    """Apply one DB-claimed command using the existing transport adapters.

    The handler is independent of the originating connection. If the socket is
    still connected, personal validation errors go there; after a crash or on a
    different worker they are discarded while task-level state/error events are
    still broadcast normally.
    """

    connections = manager.active_connections.get(command.task_id, [])
    websocket: Any = connections[0] if connections else _DiscardingCommandWebSocket()
    message_data = dict(command.payload)
    message_data.update(
        {
            "_durable_ack_sent": True,
            "_durable_attempt_count": command.attempt_count,
            "_durable_target_run_id": command.target_run_id,
        }
    )
    if command.kind != TaskCommandKind.CANCEL:
        user = await asyncio.to_thread(_load_command_actor, command.actor_user_id)
        message_data.update({"user": user, "user_id": int(user.id)})
    if command.kind != TaskCommandKind.MESSAGE and command.target_run_id is not None:
        current_run_id = await asyncio.to_thread(
            _load_command_task_run_id, command.task_id
        )
        if current_run_id != command.target_run_id:
            raise TaskCommandRejected(
                f"Task run changed before {command.kind.value} command "
                f"{command.command_id} was applied",
                reason="stale_run",
            )
    if command.kind in {
        TaskCommandKind.PAUSE,
        TaskCommandKind.RESUME,
        TaskCommandKind.CANCEL,
    } and await asyncio.to_thread(
        task_has_live_foreign_runner,
        command.task_id,
    ):
        raise TaskCommandDeferred(
            f"{command.kind.value.title()} command {command.command_id} is waiting "
            "for the active task lease owner"
        )

    if command.kind == TaskCommandKind.MESSAGE:
        await _handle_chat_message_unserialized(
            websocket, command.task_id, message_data
        )
        delivery_status = await asyncio.to_thread(
            _load_command_message_delivery_status,
            command.task_id,
            command.command_id,
        )
        if delivery_status == DELIVERY_PENDING:
            raise TaskCommandDeferred(
                f"Message {command.command_id} is waiting for runtime injection"
            )
        if delivery_status == DELIVERY_FAILED:
            raise TaskCommandRejected(
                f"Message {command.command_id} could not be applied"
            )
    elif command.kind == TaskCommandKind.PAUSE:
        await _handle_pause_task_unserialized(websocket, command.task_id, message_data)
    elif command.kind == TaskCommandKind.RESUME:
        await _handle_resume_task_unserialized(websocket, command.task_id, message_data)
    elif command.kind == TaskCommandKind.CANCEL:
        from ..models.agent import Agent
        from .a2a import _cancel_task_unserialized

        agent_id_value = message_data.get("agent_id")
        if agent_id_value is None:
            raise ValueError("Agent ID is missing or null in cancel command payload")
        try:
            agent_id = int(agent_id_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Agent ID {agent_id_value!r} is invalid in cancel command payload"
            ) from exc
        SessionLocal = get_session_local()
        with SessionLocal() as db:
            agent = db.query(Agent).filter(Agent.id == agent_id).first()
            if agent is None:
                raise ValueError(f"Agent {agent_id} not found for cancel command")
            try:
                await _cancel_task_unserialized(
                    task_id=command.task_id,
                    agent=agent,
                    db=db,
                )
            except StaleTaskRunError as exc:
                raise TaskCommandRejected(str(exc), reason="stale_run") from exc
    else:  # pragma: no cover - enum construction rejects this earlier
        raise ValueError(f"Unsupported task command kind: {command.kind}")
    durable_error = message_data.get("_durable_command_error")
    if isinstance(durable_error, str) and durable_error:
        raise TaskCommandRejected(durable_error)
    return {
        "task_id": command.task_id,
        "command_id": command.command_id,
        "kind": command.kind.value,
    }


async def _broadcast_terminal_command_error(
    command: ClaimedTaskCommand,
    error: BaseException,
) -> None:
    await manager.broadcast_to_task(
        {
            "type": "agent_error",
            "message": (f"Task command {command.kind.value} failed: {error}"),
            "task_id": command.task_id,
            "command_id": command.command_id,
            "timestamp": datetime.now(timezone.utc).timestamp(),
        },
        command.task_id,
    )


async def execute_durable_task_command(
    command: ClaimedTaskCommand,
) -> dict[str, Any] | None:
    """Apply one command and expose only terminal transport failures to clients."""

    try:
        return await _execute_durable_task_command(command)
    except TaskCommandDeferred as exc:
        if command.defer_count + 1 >= MAX_COMMAND_DEFERS:
            await _broadcast_terminal_command_error(command, exc)
        raise
    except TaskCommandRejected:
        # Rejections come from handlers that already expose their durable
        # domain-level outcome. The dispatcher makes them terminal immediately.
        raise
    except Exception as exc:
        if command.failure_count + 1 >= MAX_COMMAND_FAILURES:
            await _broadcast_terminal_command_error(command, exc)
        raise


def _load_command_message_delivery_status(
    task_id: int,
    turn_id: str,
) -> str | None:
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        message = (
            db.query(TaskChatMessage)
            .filter(
                TaskChatMessage.task_id == task_id,
                TaskChatMessage.role == "user",
                TaskChatMessage.turn_id == turn_id,
            )
            .first()
        )
        if message is None:
            return None
        delivery_status = getattr(message, "delivery_status", None)
        return delivery_status if isinstance(delivery_status, str) else None


def _load_command_task_run_id(task_id: int) -> str | None:
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = db.query(Task).filter(Task.id == task_id).first()
        if task is None:
            raise ValueError(f"Task {task_id} no longer exists")
        return str(task.run_id) if task.run_id is not None else None


@ws_router.websocket("/ws/build/chat")
async def websocket_builder_chat_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None, description="Authentication token"),
) -> None:
    """WebSocket endpoint for AI Agent Builder Assistant chat."""
    user = await get_authenticated_user(websocket, token)
    if not user:
        await websocket.close(code=4001, reason="Authentication required")
        return

    await websocket.accept()
    logger.info(f"Builder chat WebSocket connection established for user {user.id}")

    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"📨 Received builder chat message: {data[:200]}")

            message_data = json.loads(data)

            # Run in background to not block receiving
            if (
                hasattr(websocket.state, "chat_task")
                and websocket.state.chat_task
                and not websocket.state.chat_task.done()
            ):
                websocket.state.chat_task.cancel()

            websocket.state.chat_task = asyncio.create_task(
                handle_builder_chat(websocket, message_data, user)
            )

    except WebSocketDisconnect:
        logger.info(f"Builder chat WebSocket disconnected for user {user.id}")
    except (ConnectionError, RuntimeError) as e:
        logger.error(f"Connection error in builder chat WebSocket: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in builder chat WebSocket: {e}")


async def handle_builder_chat(
    websocket: WebSocket,
    message_data: dict,
    user: User,
) -> None:
    """Handle individual builder chat requests via WebSocket using an in-memory ReAct agent.

    This creates an agent that only has access to the 'create_agent' tool, allowing
    dynamic agent creation during the conversation.

    Sends messages in the format expected by the frontend:
    - message_delta: Streaming text chunks
    - message_end: Final message with optional config_updates
    - error: Error messages

    Performance optimizations:
    - Reuses AgentService across messages (only creates on first message)
    - Pre-creates CreateAgentTool directly without full tool loading
    - Caches LLM configuration in websocket state
    """
    import uuid

    from ...core.agent.context.enrichment import build_skill_context
    from ...core.agent.service import AgentService
    from ...core.memory.in_memory import InMemoryMemoryStore
    from ...skills.utils import create_skill_manager
    from ..models.database import get_db
    from ..services.llm_utils import UserAwareModelStorage

    db_gen = get_db()
    db = next(db_gen)

    # Generate task_id for builder chat (reuse if exists)
    if not hasattr(websocket.state, "builder_task_id"):
        websocket.state.builder_task_id = f"builder_chat_{uuid.uuid4().hex[:8]}"
    builder_task_id = websocket.state.builder_task_id

    builder_tracer = create_ephemeral_tracer(
        task_id=builder_task_id,
        websocket_handler=SharedWebSocketTracer(
            websocket, builder_task_id, is_preview=False
        ),
        user=user,
        is_preview=False,
    )

    try:
        user_message = message_data.get("message", "")
        if (
            not user_message
            and "messages" in message_data
            and isinstance(message_data["messages"], list)
            and len(message_data["messages"]) > 0
        ):
            last_msg = message_data["messages"][-1]
            if isinstance(last_msg, dict) and last_msg.get("role") == "user":
                user_message = last_msg.get("content", "")

        # Handle uploaded files: upload to server and inject file_ids into message
        files = message_data.get("files", [])
        if files:
            from ..models.uploaded_file import UploadedFile as _UploadedFile

            file_ids = []
            for file_info in files:
                file_id = file_info.get("file_id")
                if not file_id:
                    continue
                record = (
                    db.query(_UploadedFile)
                    .filter(
                        _UploadedFile.file_id == file_id,
                        _UploadedFile.user_id == int(user.id),
                    )
                    .first()
                )
                if record:
                    file_ids.append(file_id)

            if file_ids:
                user_message += (
                    f"\n\n[Uploaded file_ids: {file_ids}. "
                    "Use file_id as the canonical file handle and do not guess storage paths. "
                    "Please call `create_knowledge_base_from_file` with these file_ids immediately, "
                    "then create or update the agent with the resulting collection_name.]"
                )

        # Build current_config back from top-level keys
        models = message_data.get("models")
        if not isinstance(models, dict):
            models = {}
        current_config = {
            "id": message_data.get("id"),
            "name": message_data.get("name", ""),
            "description": message_data.get("description", ""),
            "instructions": message_data.get("instructions", ""),
            "model": models.get("general"),
            "compact_model": models.get("compact"),
            "tool_categories": message_data.get("tool_categories", []),
            "skills": message_data.get("selectedSkills", []),
            "knowledge_bases": message_data.get("selectedKbs", []),
            "execution_mode": message_data.get("executionMode", "balanced"),
        }

        skill_manager = create_skill_manager()
        agent_builder_skill = await skill_manager.get_skill("agent-builder")
        agent_builder_skill_context = (
            build_skill_context(agent_builder_skill) if agent_builder_skill else None
        )

        # Build system prompt with runtime state only. The behavioral workflow comes
        # from the forced agent-builder skill context below.
        system_prompt = f"""You are the runtime wrapper for the Xagent builder chat.
Follow the selected `agent-builder` skill as the authoritative workflow.

Current Agent Configuration:
{current_config}

Builder chat tools available in this runtime:
- create_agent: Create a new agent with specific capabilities
- update_agent: Update an existing agent with specific capabilities
- list_available_skills: Query the list of skills you can assign to an agent
- list_tool_categories: Query the list of tool categories you can assign to an agent
- list_knowledge_bases: Query the list of knowledge bases you can associate with an agent
- ask_user_question: Ask the user a question with a clarification form when you need their input or decision (e.g., about creating a knowledge base)
- create_knowledge_base_from_url: Create a knowledge base by crawling a given website URL (use this automatically if the user provided a URL)
- create_knowledge_base_from_file: Create a knowledge base from already-uploaded files using their file_ids (use this when the user has uploaded files)

Use native `ask_user_question` for structured user input. Do not ask required
clarification questions as plain assistant text.
"""

        async def send_builder_outbound_message(payload: Dict[str, Any]) -> None:
            """Bridge agent agent-to-user messages to the builder chat socket."""
            await websocket.send_text(
                json.dumps(
                    create_stream_event(
                        _agent_outbound_event_type(payload),
                        builder_task_id,
                        {
                            "event_id": payload.get("event_id"),
                            "step_id": payload.get("step_id"),
                            "execution_id": payload.get("execution_id"),
                            "message": payload.get("message"),
                            "message_type": payload.get("message_type", "info"),
                            "expect_response": bool(
                                payload.get("expect_response", False)
                            ),
                            "visible": bool(payload.get("visible", True)),
                            "metadata": payload.get("metadata") or {},
                        },
                    )
                )
            )

        # Get LLM configuration
        model_name = current_config.get("model")
        compact_model_name = current_config.get("compact_model")
        resolver = UserAwareModelStorage(db)
        llm = None
        compact_llm = None

        if model_name:
            llm = resolver.get_llm_by_name_with_access(
                model_name,
                user_id=user.id,  # type: ignore[arg-type]
            )

        if compact_model_name:
            compact_llm = resolver.get_llm_by_name_with_access(
                compact_model_name,
                user_id=user.id,  # type: ignore[arg-type]
            )

        if not llm or compact_llm is None:
            default_llm, _fast_llm, _vision_llm, default_compact_llm = (
                resolver.get_configured_defaults(
                    user_id=user.id  # type: ignore[arg-type]
                )
            )
            if not llm:
                llm = default_llm
            if compact_llm is None:
                compact_llm = default_compact_llm

        if not llm:
            await websocket.send_text(
                json.dumps(
                    {"type": "error", "message": "No LLM configured for builder chat"}
                )
            )
            return

        # Create or reuse agent service (only create once)
        if not hasattr(websocket.state, "builder_agent_service"):
            # Create or get memory for builder chat
            if not hasattr(websocket.state, "builder_memory"):
                websocket.state.builder_memory = InMemoryMemoryStore()
            memory = websocket.state.builder_memory

            # Initialize chat history
            websocket.state.builder_chat_history = []

            from ...core.tools.adapters.vibe.agent_tool import (
                CreateAgentTool,
                ListAvailableSkillsTool,
                ListToolCategoriesTool,
                UpdateAgentTool,
            )
            from ...core.tools.adapters.vibe.document_search import (
                ListKnowledgeBasesTool,
            )
            from ...core.tools.adapters.vibe.file_ingestion_tool import (
                CreateKnowledgeBaseFromFileTool,
            )
            from ...core.tools.adapters.vibe.web_ingestion_tool import (
                CreateKnowledgeBaseFromUrlTool,
            )

            # Create only the necessary tools directly (much faster than loading all tools)
            create_agent_tool = CreateAgentTool(
                session_factory=get_session_local(),
                user_id=int(user.id),
                task_id=builder_task_id,
                workspace_base_dir=str(get_uploads_dir() / "builder_chat"),
            )
            update_agent_tool = UpdateAgentTool(
                session_factory=get_session_local(),
                user_id=int(user.id),
                task_id=builder_task_id,
                workspace_base_dir=str(get_uploads_dir() / "builder_chat"),
            )
            list_skills_tool = ListAvailableSkillsTool()
            list_tool_categories_tool = ListToolCategoriesTool()
            list_kbs_tool = ListKnowledgeBasesTool(
                user_id=int(user.id), is_admin=bool(user.is_admin)
            )
            create_kb_url_tool = CreateKnowledgeBaseFromUrlTool(
                user_id=int(user.id), is_admin=bool(user.is_admin)
            )
            create_kb_file_tool = CreateKnowledgeBaseFromFileTool(
                user_id=int(user.id), is_admin=bool(user.is_admin)
            )

            # Build allowed external directories
            allowed_external_dirs = []
            if user and user.id:
                from ...core.workspace import scoped_user_root

                user_upload_dir = scoped_user_root(get_uploads_dir(), int(user.id))
                allowed_external_dirs.append(str(user_upload_dir))
            allowed_external_dirs.extend([str(d) for d in get_external_upload_dirs()])

            # Create agent service with pre-built tool (no WebToolConfig needed)
            agent_service = AgentService(
                name="builder_chat_agent",
                llm=llm,
                fast_llm=None,  # No fast llm for builder chat
                vision_llm=None,
                compact_llm=compact_llm,
                memory=memory,
                tools=[
                    create_agent_tool,
                    update_agent_tool,
                    list_skills_tool,
                    list_tool_categories_tool,
                    list_kbs_tool,
                    create_kb_url_tool,
                    create_kb_file_tool,
                ],
                pattern="react",
                id=builder_task_id,
                enable_workspace=True,
                workspace_base_dir=str(get_uploads_dir() / "builder_chat"),
                allowed_external_dirs=allowed_external_dirs,
                task_id=builder_task_id,
                tracer=builder_tracer,  # Using common websocket tracer
            )

            # Save agent service to websocket state for reuse. Builder chat has a
            # fixed product workflow: force the agent-builder skill and do not
            # allow generic skill auto-selection to choose anything else.
            agent_service.set_allowed_skills(["agent-builder"])
            agent_service.set_recovered_skill_context(agent_builder_skill_context)
            agent_service.set_outbound_message_handler(send_builder_outbound_message)
            websocket.state.builder_agent_service = agent_service
            logger.info(
                f"Created new builder chat agent service with task_id: {builder_task_id}"
            )
        else:
            agent_service = websocket.state.builder_agent_service
            agent_service.set_allowed_skills(["agent-builder"])
            agent_service.set_recovered_skill_context(agent_builder_skill_context)
            agent_service.set_outbound_message_handler(send_builder_outbound_message)
            # Update tracer to the new connection
            agent_service.tracer = builder_tracer
            # Defensive initialization for service reuse
            if not hasattr(websocket.state, "builder_chat_history"):
                websocket.state.builder_chat_history = []
            if not hasattr(websocket.state, "builder_memory"):
                websocket.state.builder_memory = InMemoryMemoryStore()
            if hasattr(agent_service, "agent") and hasattr(
                agent_service.agent, "patterns"
            ):
                for pattern in agent_service.agent.patterns:
                    if hasattr(pattern, "tracer"):
                        pattern.tracer = builder_tracer
            logger.info(
                f"Reusing existing builder chat agent service with task_id: {builder_task_id}"
            )

        # Execute task with the agent
        if user_message:
            # Build execution context with system prompt
            execution_context: dict[str, Any] = {
                "system_prompt": system_prompt,
            }

            # Set chat history before execution
            if hasattr(websocket.state, "builder_chat_history") and hasattr(
                agent_service, "set_conversation_history"
            ):
                agent_service.set_conversation_history(
                    websocket.state.builder_chat_history
                )

            # Execute task with the agent
            with UserContext(int(user.id)):
                result = await agent_service.execute_task(
                    task=user_message,
                    context=execution_context,
                    task_id=builder_task_id,
                )

            if result.get("status") == "waiting_for_user":
                result["chat_response"] = {
                    "message": result.get("message", ""),
                    "interactions": result.get("interactions", []),
                }
                result.setdefault("output", result.get("message", ""))

            # Append interaction to chat history
            if hasattr(websocket.state, "builder_chat_history"):
                # Make sure we don't end up with consecutive user messages
                if (
                    websocket.state.builder_chat_history
                    and websocket.state.builder_chat_history[-1]["role"] == "user"
                ):
                    logger.warning(
                        "Found consecutive user messages in builder_chat_history. Appending a placeholder assistant message."
                    )
                    # If last message was also user, insert a placeholder assistant message
                    # instead of dropping the previous user message (which causes data loss)
                    websocket.state.builder_chat_history.append(
                        {
                            "role": "assistant",
                            "content": "I apologize, but my previous process was interrupted. Let's continue.",
                        }
                    )

                websocket.state.builder_chat_history.append(
                    {"role": "user", "content": user_message}
                )
                output_content = result.get("output", "")

                # If there's a structured chat_response, serialize it to JSON
                # so the LLM retains the original structured interaction context
                chat_response = result.get("chat_response")
                if chat_response:
                    try:
                        # Reconstruct the expected JSON block that was stripped by react.py
                        structured_content = json.dumps(
                            {"type": "chat", "chat": chat_response}, ensure_ascii=False
                        )
                        output_content = f"```json\n{structured_content}\n```"
                    except Exception as e:
                        logger.warning(
                            f"Failed to serialize chat_response for history: {e}"
                        )

                if output_content:
                    websocket.state.builder_chat_history.append(
                        {"role": "assistant", "content": output_content}
                    )
                else:
                    # Provide a fallback assistant message to prevent consecutive user messages
                    websocket.state.builder_chat_history.append(
                        {
                            "role": "assistant",
                            "content": "I encountered an issue and couldn't generate a proper response.",
                        }
                    )

                # Keep history size manageable (e.g. last 20 messages)
                websocket.state.builder_chat_history = (
                    websocket.state.builder_chat_history[-20:]
                )

            # Send task_completed event to match the preview flow behavior
            # which relies on Trace events but might need a final completion indicator
            try:
                # We need to pass the chat_response if it exists, along with content
                # so the frontend can receive the structured data instead of trying to parse markdown
                task_completion_result = {"content": result.get("output", "")}
                if result.get("chat_response"):
                    task_completion_result["chat_response"] = result.get(
                        "chat_response"
                    )

                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "task_completed",
                            "task_id": builder_task_id,
                            "result": task_completion_result,
                            "success": result.get("success", True),
                            "timestamp": datetime.now(timezone.utc).timestamp(),
                        }
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to send task_completed: {e}")

    except Exception as e:
        logger.error(f"Error handling builder chat: {e}", exc_info=True)
        await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
    finally:
        db.close()


@ws_router.websocket("/ws/build/preview")
async def websocket_build_preview_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None, description="Authentication token"),
) -> None:
    """WebSocket endpoint for build page agent preview using normal task execution."""
    # Verify user identity
    user = await get_authenticated_user(websocket, token)
    if not user:
        await websocket.close(code=4001, reason="Authentication required")
        return

    await websocket.accept()
    logger.info(f"Build preview WebSocket connection established for user {user.id}")

    try:
        while True:
            # Receive client message
            data = await websocket.receive_text()
            logger.info(f"📨 Received build preview WebSocket message: {data[:200]}")

            message_data = json.loads(data)
            message_type = message_data.get("type")

            if message_type == "preview":
                await handle_build_preview_execution(websocket, message_data, user)
            elif message_type == "pause":
                task_id = getattr(websocket.state, "preview_task_id", None)
                if isinstance(task_id, (int, str)) and str(task_id).isdigit():
                    await handle_pause_task(
                        websocket,
                        int(task_id),
                        {"type": "pause_task", "user": user},
                    )
                else:
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "error",
                                "message": "No active agent to pause",
                            }
                        )
                    )
            elif message_type == "resume":
                task_id = getattr(websocket.state, "preview_task_id", None)
                if isinstance(task_id, (int, str)) and str(task_id).isdigit():
                    await handle_resume_task(
                        websocket,
                        int(task_id),
                        {"type": "resume_task", "user": user},
                    )
                else:
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "error",
                                "message": "No active agent to resume",
                            }
                        )
                    )
            elif message_type == "clear_context":
                preview_task_id = getattr(websocket.state, "preview_task_id", None)
                if (
                    isinstance(preview_task_id, (int, str))
                    and str(preview_task_id).isdigit()
                ):
                    manager.disconnect(websocket, int(preview_task_id))
                websocket.state.preview_task_id = None
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "context_cleared",
                            "timestamp": datetime.now(timezone.utc).timestamp(),
                        }
                    )
                )
                logger.info(f"Cleared build preview context for user {user.id}")
            else:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "message": f"Unknown message type: {message_type}",
                        }
                    )
                )

    except WebSocketDisconnect:
        preview_task_id = getattr(websocket.state, "preview_task_id", None)
        if isinstance(preview_task_id, (int, str)) and str(preview_task_id).isdigit():
            manager.disconnect(websocket, int(preview_task_id))
        logger.info(f"Build preview WebSocket disconnected for user {user.id}")
    except (ConnectionError, RuntimeError) as e:
        logger.error(f"Connection error in build preview WebSocket: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in build preview WebSocket: {e}")


async def handle_build_preview_execution(
    websocket: WebSocket,
    message_data: dict,
    user: User,
) -> None:
    """Create a normal preview task and schedule it through the chat task flow."""
    from ..schemas.chat import TaskCreateRequest
    from .chat import create_task

    user_message = message_data.get("message", "")
    files_data = message_data.get("files", [])
    if not user_message and not files_data:
        await websocket.send_text(
            json.dumps(
                {
                    "type": "error",
                    "message": "Message or files are required for preview",
                }
            )
        )
        return

    agent_config = {
        "instructions": message_data.get("instructions", ""),
        "knowledge_bases": message_data.get("knowledge_bases", []),
        "skills": message_data.get("skills", []),
        "tool_categories": message_data.get("tool_categories", []),
        "is_preview": True,
        "preview_agent_id": message_data.get("agent_id"),
    }
    models = message_data.get("models", {})

    def _model_ref(key: str) -> Optional[str]:
        value = models.get(key)
        if value is None or value == "":
            return None
        return str(value)

    llm_ids = [
        _model_ref("general"),
        _model_ref("small_fast"),
        _model_ref("visual"),
        _model_ref("compact"),
    ]
    execution_mode = message_data.get("execution_mode")

    preview_task_id = getattr(websocket.state, "preview_task_id", None)
    has_preview_task = (
        isinstance(preview_task_id, (int, str)) and str(preview_task_id).isdigit()
    )
    if not has_preview_task:
        task_request = TaskCreateRequest(
            title=(user_message or "Build preview")[:80],
            description=user_message,
            agent_id=None,
            files=None,
            llm_ids=llm_ids,
            agent_config=agent_config,
            execution_mode=execution_mode,
            is_visible=False,
        )

        from ..models import database as database_module

        db_gen = database_module.get_db()
        preview_db = next(db_gen)
        try:
            task_response = await create_task(task_request, db=preview_db, user=user)
            preview_task_id = int(task_response.task_id)
        finally:
            preview_db.close()

        websocket.state.preview_task_id = preview_task_id
        manager.register_connection(websocket, preview_task_id)
    else:
        preview_task_id = int(str(preview_task_id))

    await handle_chat_message(
        websocket,
        preview_task_id,
        {
            "type": "chat",
            "message": user_message,
            "files": files_data,
            "user": user,
            "user_id": user.id,
            "context": {},
        },
    )
    return
