"""WebSocket real-time communication handler"""

import asyncio
import json
import logging
import re
import shutil
import unicodedata
import uuid
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast
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
from ...core.file_ref import FILE_REF_MODEL_INSTRUCTIONS, build_file_ref
from ..auth_dependencies import get_user_from_websocket_token
from ..models.database import get_db
from ..models.task import Task, TaskStatus
from ..models.uploaded_file import UploadedFile
from ..models.user import User
from ..services.chat_history_service import get_latest_waiting_question
from ..services.hot_path_cache import (
    cache_get,
    cache_set,
    cache_version_token,
    task_cache_ttl_seconds,
    web_task_history_key,
)
from ..services.managed_file_ref import (
    DurableStorageOperationError,
    build_task_output_storage_key,
    ensure_uploaded_file_local_path,
)
from ..services.task_lease_service import (
    acquire_task_lease,
    mark_task_paused_if_stale,
    run_task_lease_heartbeat,
    stop_task_lease_heartbeat,
)
from ..services.uploaded_file_store import UploadedFileStore
from ..services.workforce_runtime import (
    release_current_runner_task_lease_with_workforce_sync,
    release_task_lease_with_workforce_sync,
    sync_workforce_run_status,
)
from ..tracing import create_ephemeral_tracer
from ..user_isolated_memory import UserContext
from ..utils.db_timezone import safe_timestamp_to_unix

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
    status: TaskStatus, *, pause_accepted: bool = False
) -> bool:
    """Return True when a user message should be delivered to an active run."""

    if pause_accepted:
        return False
    return status in {TaskStatus.WAITING_FOR_USER, TaskStatus.RUNNING}


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


def normalize_filename(filename: str) -> str:
    """
    Normalize filename by removing special characters and spaces.

    Args:
        filename: Original filename

    Returns:
        Normalized filename safe for file operations
    """
    from pathlib import Path

    # Keep file extension
    name_part = Path(filename).stem
    extension = Path(filename).suffix

    # Unicode normalize (NFD to NFC, remove diacritics)
    name_part = unicodedata.normalize("NFC", name_part)

    # Replace spaces with underscores
    name_part = re.sub(r"\s+", "_", name_part)

    # Remove special characters, keep only letters, numbers, underscores, Chinese characters
    name_part = re.sub(r"[^\w\u4e00-\u9fff\-_.]", "", name_part)

    # Remove consecutive underscores
    name_part = re.sub(r"_+", "_", name_part)

    # Remove leading and trailing underscores
    name_part = name_part.strip("_")

    # Use default name if filename is empty
    if not name_part:
        name_part = "file"

    # Reassemble filename
    normalized_name = name_part + extension

    # Ensure filename doesn't start with a dot (hidden file)
    if normalized_name.startswith("."):
        normalized_name = "_" + normalized_name

    return normalized_name


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


def _build_uploaded_files_context(
    file_info_list: List[Dict[str, Any]], *, is_agent_builder: bool = False
) -> str:
    """Build stable LLM context for files already uploaded for this turn."""
    if not file_info_list:
        return ""

    file_summaries = []
    file_ids = []
    for file_info in file_info_list:
        file_id = str(file_info.get("file_id") or "").strip()
        if not file_id:
            continue
        name = str(
            file_info.get("original_name") or file_info.get("name") or "uploaded file"
        )
        file_ids.append(file_id)
        file_summaries.append(f"- {name}: file_id={file_id}")

    if not file_ids:
        return ""

    lines = [
        "## UPLOADED FILES",
        "The user has uploaded file(s) for this turn. Use these exact file_id values:",
        *file_summaries,
        "",
        FILE_REF_MODEL_INSTRUCTIONS,
    ]
    if is_agent_builder:
        joined_file_ids = ", ".join(f'"{file_id}"' for file_id in file_ids)
        lines.extend(
            [
                "",
                "For knowledge-base creation, call `create_knowledge_base_from_file` with:",
                f"  file_ids = [{joined_file_ids}]",
                "Do NOT ask the user to upload again unless these file_ids fail.",
            ]
        )
    return "\n".join(lines)


def _append_uploaded_files_context_to_message(
    message: str, uploaded_files_context: str
) -> str:
    if not uploaded_files_context:
        return message
    if uploaded_files_context in message:
        return message
    return f"{message.rstrip()}\n\n{uploaded_files_context}"


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


def _normalize_attachments_for_persistence(
    file_info_list: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Project file_info_list to the minimal shape we persist on chat rows.

    Thin wrapper around the shared
    ``core.agent.attachments.project_file_info_to_chip`` so the trace
    callback and the persistence path can't drift on what fields the
    browser sees (paths must never leak — the attachments column and the
    user_message trace events both reach the UI).
    """
    from ...core.agent.attachments import project_file_info_to_chip

    return project_file_info_to_chip(file_info_list)


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
    """Persist v2 agent-to-user messages so waiting prompts survive reloads."""

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
            await manager.broadcast_to_task(
                create_final_answer_stream_event(payload_type, task_id, dict(payload)),
                task_id,
            )
            return

        event = create_stream_event(
            "agent_message",
            task_id,
            {
                "event_id": payload.get("event_id"),
                "step_id": payload.get("step_id"),
                "execution_id": payload.get("execution_id"),
                "message": payload.get("message"),
                "message_type": payload.get("message_type", "info"),
                "expect_response": bool(payload.get("expect_response", False)),
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
    return isinstance(data, dict) and data.get("__audit_only__") is True


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

    path_to_file_id[normalized_relative_path] = file_id
    path_to_file_id[f"/{normalized_relative_path}"] = file_id
    path_to_file_id[f"preview/{normalized_relative_path}"] = file_id
    path_to_file_id[f"/preview/{normalized_relative_path}"] = file_id
    path_to_file_id[f"uploads/{normalized_relative_path}"] = file_id
    path_to_file_id[f"/uploads/{normalized_relative_path}"] = file_id

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
        path_to_file_id[task_local_path] = file_id
        path_to_file_id[f"/{task_local_path}"] = file_id


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


def _output_path_in_current_task_scope(
    relative_path: str, task_id: int, task_user_id: int
) -> bool:
    parts = Path(relative_path.lstrip("/")).parts
    task_dirs = {f"web_task_{task_id}", f"task_{task_id}"}

    if (
        len(parts) >= 4
        and parts[0] == f"user_{task_user_id}"
        and parts[1] in task_dirs
        and parts[2] == "output"
    ):
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
                path_to_file_id[stripped] = final_file_id
                path_to_file_id[stripped.lstrip("/")] = final_file_id

        storage_path = getattr(file_record, "storage_path", None)
        if storage_path:
            path_to_file_id[str(storage_path)] = final_file_id

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
            path_to_file_id[workspace_relative_path] = final_file_id
            path_to_file_id[f"/{workspace_relative_path}"] = final_file_id
            path_to_file_id[f"preview/{workspace_relative_path}"] = final_file_id
            path_to_file_id[f"/preview/{workspace_relative_path}"] = final_file_id
            path_to_file_id[f"uploads/{workspace_relative_path}"] = final_file_id
            path_to_file_id[f"/uploads/{workspace_relative_path}"] = final_file_id

    if changed:
        db.commit()

    return normalized_outputs, path_to_file_id


def _normalize_task_file_outputs(
    db: Session,
    task: Any,
    file_outputs: Any,
) -> tuple[list[Dict[str, Any]], Dict[str, str]]:
    task_user_id = _task_user_id(task)
    if task_user_id is None:
        return [], {}

    task_id = int(cast(Any, task.id))
    return _normalize_file_outputs(
        db,
        task_id=task_id,
        task_user_id=task_user_id,
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


async def execute_task_background(
    task_id: int,
    user_message: str,
    context: Dict[str, Any],
    agent_manager: Any,
    user_id: int | None,
    before_message_id: int | None = None,
    llm_user_message: Optional[str] = None,
) -> None:
    """Execute task in background without blocking WebSocket message loop"""
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
        logger.info(f"Background task execution started for task {task_id}")

        task = db.query(Task).filter(Task.id == task_id).first()
        if task is None:
            raise ValueError(f"Task {task_id} not found")

        task_user_id = _task_user_id(task)
        effective_user_id = user_id if user_id is not None else task_user_id
        user = (
            db.query(User).filter(User.id == effective_user_id).first()
            if effective_user_id is not None
            else None
        )

        with UserContext(effective_user_id):
            # Get agent service
            agent_service = await agent_manager.get_agent_for_task(
                task_id, db, user=user
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
                context.get("file_info", []),
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
            )

        normalized_outputs, path_to_file_id = _normalize_task_file_outputs(
            db,
            task,
            result.get("file_outputs", []),
        )
        if normalized_outputs:
            result["file_outputs"] = normalized_outputs

        # Get AI response
        chat_response = result.get("chat_response")
        if isinstance(chat_response, dict):
            ai_response = chat_response.get("message") or result.get(
                "output", "Task completed"
            )
        else:
            ai_response = result.get("output", "Task completed")

        # Rewrite file links to file_id
        ai_response = _rewrite_file_links_to_file_id(
            ai_response,
            path_to_file_id,
        )

        # Task execution result is logged by ConsoleTraceHandler, no need for duplicate logs

        db_new_gen = get_db()
        try:
            db_new = next(db_new_gen)
            waiting_for_control = False
            final_task_status = task.status.value
            task_updated = db_new.query(Task).filter(Task.id == task_id).first()
            if task_updated:
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
                if result.get("status") == "waiting_for_user":
                    task_updated.status = TaskStatus.WAITING_FOR_USER
                    sync_workforce_run_status(db_new, task_updated, task_updated.status)
                    db_new.commit()
                    waiting_for_control = True
                    logger.info(
                        f"Updated task {task_id} status to WAITING_FOR_USER for v2 control state"
                    )
                elif result.get("status") == "interrupted":
                    task_updated.status = TaskStatus.PAUSED
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
                    if result.get("success", False):
                        task_updated.status = TaskStatus.COMPLETED
                    else:
                        task_updated.status = TaskStatus.FAILED
                    sync_workforce_run_status(db_new, task_updated, task_updated.status)
                    db_new.commit()
                    logger.info(
                        f"Updated task {task_id} status to {task_updated.status.value}"
                    )
                else:
                    logger.info(
                        f"Task {task_id} is paused, not updating status to {result.get('success')}"
                    )
                final_task_status = task_updated.status.value

                if not waiting_for_control:
                    persist_assistant_message(
                        db_new,
                        task_id=task_id,
                        user_id=int(task.user_id),
                        content=str(
                            chat_response.get("message", ai_response)
                            if isinstance(chat_response, dict)
                            else ai_response
                        ),
                        message_type="chat_response"
                        if isinstance(chat_response, dict)
                        else "final_answer",
                        interactions=chat_response.get("interactions")
                        if isinstance(chat_response, dict)
                        else None,
                    )
        finally:
            try:
                next(db_new_gen)
            except StopIteration:
                pass

        # Note: trace_task_completion is handled by the agent execution logic (e.g., dag_plan_execute.py)

        if waiting_for_control:
            await manager.broadcast_to_task(
                create_stream_event(
                    "task_info",
                    task_id,
                    {
                        "id": task.id,
                        "title": task.title,
                        "description": task.description,
                        "status": final_task_status,
                        "execution_mode": task.execution_mode,
                    },
                    task.updated_at if task.updated_at else None,
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
                    "id": task.id,
                    "title": task.title,
                    "status": final_task_status,
                    "description": task.description,
                },
                "result": ai_response,
                "output": ai_response,
                "file_outputs": normalized_outputs,
                "success": result.get("success", False),
                "chat_response": chat_response
                if isinstance(chat_response, dict)
                else None,
                "timestamp": datetime.now(timezone.utc).timestamp(),
            },
            task_id,
        )
        logger.info(f"Background task {task_id} execution completed")

    except Exception as e:
        logger.error(f"Background task {task_id} execution failed: {e}", exc_info=True)
        # Send error event
        try:
            await manager.broadcast_to_task(
                {
                    "type": "task_error",
                    "task_id": task_id,
                    "error": str(e),
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


async def execute_resume_background(
    task_id: int,
    agent_service: Any,
    user: Any,
    task: Any,
    previous_task: Optional[asyncio.Task] = None,
) -> None:
    """Resume an agent execution after an interrupt/user-message checkpoint."""
    from ..models.database import get_db
    from ..models.task import Task, TaskStatus

    lease_stop_event = None
    lease_heartbeat_task = None
    lease = None
    lease_released = False
    result: Dict[str, Any] | None = None
    normalized_outputs: list[Dict[str, str]] = []
    output = ""
    success = False
    final_status = getattr(task.status, "value", str(task.status))
    try:
        if previous_task is not None and not previous_task.done():
            try:
                await previous_task
            except Exception as e:
                logger.warning(
                    f"Previous background task {task_id} ended before resume: {e}"
                )

        db_gen = get_db()
        db_lease = next(db_gen)
        try:
            lease = acquire_task_lease(db_lease, task_id)
            if lease is not None:
                task_for_sync = db_lease.query(Task).filter(Task.id == task_id).first()
                if task_for_sync is not None and sync_workforce_run_status(
                    db_lease, task_for_sync, TaskStatus.RUNNING
                ):
                    db_lease.commit()
        finally:
            db_lease.close()
        if lease is None:
            logger.info(
                "Task %s resume skipped; another runner owns the lease", task_id
            )
            return
        lease_stop_event = asyncio.Event()
        lease_heartbeat_task = asyncio.create_task(
            run_task_lease_heartbeat(lease, lease_stop_event)
        )

        user_id = int(user.id) if user else None
        with UserContext(user_id):
            result = await agent_service.resume_execution_by_id(str(task_id))

        if result is None:
            logger.warning(f"No resumable agent execution found for task {task_id}")
            return

        status = str(result.get("status") or "")
        success = bool(result.get("success", False))
        output = str(result.get("output") or result.get("error") or "")

        if _task_user_id(task) is not None:
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
            finally:
                db_normalize.close()

        db_gen = get_db()
        db_new = next(db_gen)
        try:
            task_updated = db_new.query(Task).filter(Task.id == task_id).first()
            if task_updated:
                if status == "waiting_for_user":
                    final_task_status = TaskStatus.WAITING_FOR_USER
                elif status == "interrupted":
                    final_task_status = TaskStatus.PAUSED
                elif success:
                    final_task_status = TaskStatus.COMPLETED
                else:
                    final_task_status = TaskStatus.FAILED
                lease_released = release_current_runner_task_lease_with_workforce_sync(
                    db_new, task_id, status=final_task_status
                )
                db_new.refresh(task_updated)
                final_status = task_updated.status.value
            else:
                final_status = task.status.value
        finally:
            db_new.close()

        if status in {"interrupted", "waiting_for_user"}:
            await manager.broadcast_to_task(
                create_stream_event(
                    "task_info",
                    task_id,
                    {
                        "id": task.id,
                        "title": task.title,
                        "description": task.description,
                        "status": final_status,
                        "execution_mode": task.execution_mode,
                    },
                ),
                task_id,
            )
            return

        await manager.broadcast_to_task(
            {
                "type": "task_completed",
                "task": {
                    "id": task.id,
                    "title": task.title,
                    "status": final_status,
                    "description": task.description,
                },
                "result": output,
                "output": output,
                "file_outputs": normalized_outputs,
                "success": success,
                "metadata": result.get("metadata", {}),
                "timestamp": datetime.now(timezone.utc).timestamp(),
            },
            task_id,
        )
    except asyncio.CancelledError:
        logger.info(f"V2 resume background task {task_id} cancelled")
        raise
    except Exception as e:
        logger.error(f"V2 resume background task {task_id} failed: {e}", exc_info=True)
        await manager.broadcast_to_task(
            {
                "type": "task_error",
                "task_id": task_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).timestamp(),
            },
            task_id,
        )
    finally:
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

    def cleanup_task(self, task_id: int) -> None:
        """Clean up completed background task"""
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            if task.done():
                del self.running_tasks[task_id]
                logger.info(f"Cleaned up background task for task {task_id}")

    async def cancel_task(self, task_id: int, timeout_seconds: float = 0.5) -> None:
        task = self.running_tasks.get(task_id)
        if not task:
            return

        if not task.done():
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

            # Convert trace event to stream format
            event_type_str = get_event_type_mapping(event)
            serialized_data = self._serialize_data(event.data)

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
            ),
        )
        db.commit()
        db.refresh(file_record)

    return RedirectResponse(
        url=f"/api/files/public/preview/{file_record.file_id}",
        status_code=307,
    )


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
        await websocket.send_text(json.dumps(message))

    async def broadcast_to_task(self, message: dict, task_id: int) -> None:
        if task_id in self.active_connections:
            for connection in self.active_connections[task_id].copy():
                try:
                    await connection.send_text(json.dumps(message))
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
    """Handle file upload for task"""
    try:
        from ..models.uploaded_file import UploadedFile

        uploaded_files = []
        file_info_list = []

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

        for file_info in files:
            file_id = file_info.get("file_id")
            if not file_id:
                logger.warning(f"No file_id provided in file info: {file_info}")
                continue

            file_record = (
                db.query(UploadedFile)
                .filter(
                    UploadedFile.file_id == file_id,
                    UploadedFile.user_id == int(authorized_owner_id),
                    or_(
                        UploadedFile.task_id == int(task_id),
                        UploadedFile.task_id.is_(None),
                    ),
                )
                .first()
            )
            if not file_record:
                logger.warning(
                    "File record not accessible for task %s: %s",
                    task_id,
                    file_id,
                )
                continue

            file_name = file_record.filename
            file_size = file_record.file_size
            file_type = file_record.mime_type
            source_path = ensure_uploaded_file_local_path(file_record)

            if not source_path.exists():
                logger.warning(f"Physical file not found: {source_path}")
                continue

            try:
                # Use normalized filename instead of original
                original_file_name = Path(file_name).name
                normalized_file_name = normalize_filename(original_file_name)

                # Keep the real file in the user-level uploads dir, but expose
                # a symlink inside the task workspace's input/ directory so
                # the agent's `list_files` tool can see it without needing
                # additional tool calls. Without this link, `list_files`
                # returns an empty workspace and the agent often gives up
                # before falling back to `list_all_user_files`.
                target_path = source_path
                uploaded_files.append(str(target_path))

                if file_record.task_id is None:
                    file_record.task_id = task_id

                db.flush()

                # Build file info using normalized filename
                file_info_list.append(
                    {
                        "file_id": file_record.file_id,
                        "name": normalized_file_name,
                        "original_name": original_file_name,
                        "size": file_size,
                        "type": file_type,
                        "path": str(target_path),
                        "workspace_path": None,
                    }
                )

                logger.info(
                    f"File staged: storage={target_path} "
                    f"(original={original_file_name} normalized={normalized_file_name})"
                )

            except Exception as e:
                logger.error(f"Error handling file {file_info.get('name')}: {e}")
                raise

        logger.info(f"🎉 File upload completed, uploaded {len(uploaded_files)} files")
        db.commit()
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
    """Handle chat message"""
    try:
        user_message = message_data.get("message", "")

        context = message_data.get("context", {})
        files = message_data.get("files", [])
        user = message_data.get("user")

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
                    pause_accepted=pause_accepted,
                )
                agent_service = None
                dag_pattern = None
                supports_live_control = False
                has_continuation = False
                if task_uses_live_control:
                    agent_service = await get_agent_manager().get_agent_for_task(
                        task_id, db, user=user
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

                    # Immediately send trace_user_message to display user message on interface
                    if hasattr(dag_pattern, "tracer") and hasattr(
                        dag_pattern, "task_id"
                    ):
                        trace_data: Dict[str, Any] = {
                            "context": context,
                            "pattern": "DAG Plan-Execute Continuation",
                            "continuation": "true",
                            "files": display_file_refs,
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
                    return
                if task_uses_live_control and supports_live_control:
                    logger.info(f"Using agent message control for task {task_id}")
                    assert agent_service is not None
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
                    posted = await agent_service.post_user_message(
                        str(task_id),
                        execution_message=user_message_for_llm,
                        display_message=display_user_message,
                        files=display_file_refs,
                        request_interrupt=task.status == TaskStatus.RUNNING,
                        reason="new websocket user message",
                    )
                    if not posted:
                        logger.warning(
                            f"agent execution {task_id} was not live; attempting resume from checkpoint"
                        )

                    previous_task = background_task_manager.running_tasks.get(task_id)
                    bg_task = asyncio.create_task(
                        execute_resume_background(
                            task_id=task_id,
                            agent_service=agent_service,
                            user=user,
                            task=task,
                            previous_task=previous_task,
                        )
                    )
                    background_task_manager.register_task(task_id, bg_task)

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
                                    "type": "agent_error",
                                    "message": (
                                        "Task pause is still being applied; "
                                        "please retry shortly."
                                    ),
                                    "timestamp": datetime.now(timezone.utc).timestamp(),
                                },
                                task_id,
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
                        TaskTurnOrchestrator,
                        TaskTurnPayload,
                        TurnKind,
                    )

                    # Strip absolute filesystem paths before the row hits
                    # disk — the attachments column is exposed to historical-
                    # replay clients, so paths must not leak.
                    persisted_attachments = _normalize_attachments_for_persistence(
                        file_info_list
                    )
                    payload = TaskTurnPayload(
                        transcript_message=display_user_message,
                        execution_message=user_message_for_llm,
                        attachments=persisted_attachments or None,
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
                                "type": "agent_error",
                                "message": ("Internal dispatch error; please retry."),
                                "timestamp": datetime.now(timezone.utc).timestamp(),
                            },
                            task_id,
                        )
                        return

                    try:
                        await TaskTurnOrchestrator.begin_turn(
                            task=task,
                            payload=payload,
                            user=user,
                            db=db,
                            kind=turn_kind,
                            force_fresh=turn_force_fresh,
                            context=context,
                        )
                        logger.info(f"Task {task_id} started in background")
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
                                "type": "agent_error",
                                "message": (
                                    "Task is currently busy; please wait for "
                                    "the previous turn to finish before sending "
                                    "another message."
                                ),
                                "timestamp": datetime.now(timezone.utc).timestamp(),
                            },
                            task_id,
                        )

            finally:
                db.close()

        except (ValueError, KeyError, TypeError) as e:
            # Data validation and format error
            logger.error(f"Data validation error in agent execution: {e}")
            await manager.broadcast_to_task(
                {
                    "type": "agent_error",
                    "message": f"Data validation error: {str(e)}",
                    "timestamp": datetime.now(timezone.utc).timestamp(),
                },
                task_id,
            )
        except RuntimeError as e:
            # Runtime error
            logger.error(f"Runtime error in agent execution: {e}")
            await manager.broadcast_to_task(
                {
                    "type": "agent_error",
                    "message": f"Runtime error: {str(e)}",
                    "timestamp": datetime.now(timezone.utc).timestamp(),
                },
                task_id,
            )
        except Exception as e:
            # Other unknown errors, re-raise
            logger.error(f"Unexpected error in agent execution: {e}")
            raise

    except (ValueError, KeyError, TypeError) as e:
        # Message format error
        logger.error(f"Message format error: {e}")
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
        raise


async def handle_execute_task(
    websocket: WebSocket, task_id: int, message_data: dict
) -> None:
    """Handle task execution request"""
    try:
        user = message_data.get("user")
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
                task_id, db, user=user
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

            # Set up user context
            with UserContext(user.id):
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
                    "result": result.get("output", ""),
                    "output": result.get("output", ""),
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
        logger.error(f"Data validation error in task execution: {e}")
        await manager.broadcast_to_task(
            {
                "type": "agent_error",
                "message": f"Data validation error: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            task_id,
        )
    except RuntimeError as e:
        # Runtime error
        logger.error(f"Runtime error in task execution: {e}")
        await manager.broadcast_to_task(
            {
                "type": "agent_error",
                "message": f"Runtime error: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            task_id,
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
        from ..models.chat_message import TaskChatMessage
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
                historical_events.append(
                    {
                        "type": "trace_event",
                        "data": {
                            "event_id": trace_event.event_id,
                            "event_type": trace_event.event_type,
                            "step_id": trace_event.step_id,
                            "parent_event_id": trace_event.parent_event_id,
                            "data": normalized_event_data,
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
            for chat_message in chat_messages:
                role = str(chat_message.role)
                content = str(chat_message.content or "").strip()
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
                        # Historical assistant questions are transcript entries.
                        # The current WAITING_FOR_USER state is reasserted separately
                        # after replay, so old questions must not flip status back.
                        "expect_response": False,
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
    """Handle task pause request"""
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

        # Get agent service
        from .chat import get_agent_manager

        logger.info(f"Getting agent service for task {task_id}")
        agent_service = await get_agent_manager().get_agent_for_task(
            task_id, db, user=user
        )
        logger.info(f"Agent service obtained: {type(agent_service).__name__}")

        # Check if agent supports pause functionality
        if hasattr(agent_service, "pause_execution"):
            logger.info("Agent supports pause_execution, calling it...")
            pause_result = await agent_service.pause_execution()
            if pause_result is False:
                await manager.send_personal_message(
                    {
                        "type": "error",
                        "message": "No live execution found to pause",
                    },
                    websocket,
                )
                logger.warning(f"No live execution found to pause for task {task_id}")
                return
            logger.info("Agent pause_execution completed")
            _mark_task_pause_accepted(task_id)

            # Send pause confirmation
            await manager.broadcast_to_task(
                {
                    "type": "task_paused",
                    "task_id": task_id,
                    "message": "Task paused",
                    "timestamp": datetime.now(timezone.utc).timestamp(),
                },
                task_id,
            )
            logger.info(f"Task {task_id} paused successfully")
        else:
            # If pause not supported, send error message
            await manager.send_personal_message(
                {
                    "type": "error",
                    "message": "Current agent does not support pause functionality",
                },
                websocket,
            )
            logger.warning(
                f"Agent for task {task_id} does not support pause functionality"
            )

    except (ValueError, KeyError, TypeError) as e:
        # Data validation error
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
    except Exception as e:
        # Other errors, re-raise
        logger.error(f"Unexpected error pausing task {task_id}: {e}")
        raise


async def handle_resume_task(
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

        # Get agent service
        from .chat import get_agent_manager

        try:
            agent_service = await get_agent_manager().get_agent_for_task(
                task_id, db, user=user
            )
        finally:
            db.close()

        from ..models.task import Task

        task: Any | None = None
        db_update_gen = get_db()
        db_update = next(db_update_gen)
        try:
            if user.is_admin:
                task = db_update.query(Task).filter(Task.id == task_id).first()
            else:
                task = (
                    db_update.query(Task)
                    .filter(Task.id == task_id, Task.user_id == user.id)
                    .first()
                )
            if not task:
                logger.warning(
                    f"Task {task_id} not found or access denied for user {user.id}"
                )
        finally:
            db_update.close()

        if task is None:
            await manager.send_personal_message(
                {"type": "error", "message": "Task not found or access denied"},
                websocket,
            )
            return

        if getattr(agent_service, "supports_live_control", lambda: False)():
            await manager.broadcast_to_task(
                {
                    "type": "task_resumed",
                    "task_id": task_id,
                    "message": "Task resumed",
                    "timestamp": datetime.now(timezone.utc).timestamp(),
                },
                task_id,
            )
            previous_task = background_task_manager.running_tasks.get(task_id)
            bg_task = asyncio.create_task(
                execute_resume_background(
                    task_id=task_id,
                    agent_service=agent_service,
                    user=user,
                    task=task,
                    previous_task=previous_task,
                )
            )
            background_task_manager.register_task(task_id, bg_task)
            logger.info(f"Task {task_id} v2 resume scheduled")
            return

        # Check if agent supports resume functionality
        if hasattr(agent_service, "resume_execution"):
            await agent_service.resume_execution()

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
    except Exception as e:
        # Other errors, re-raise
        logger.error(f"Unexpected error resuming task {task_id}: {e}")
        raise


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
                        "agent_message",
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
                db=db,
                user_id=int(user.id),
                task_id=builder_task_id,
                workspace_base_dir=str(get_uploads_dir() / "builder_chat"),
            )
            update_agent_tool = UpdateAgentTool(
                db=db,
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
                user_upload_dir = get_uploads_dir() / f"user_{user.id}"
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
