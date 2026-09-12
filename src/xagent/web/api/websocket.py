"""WebSocket real-time communication handler"""

import asyncio
import json
import logging
import uuid
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Union,
    cast,
)
from urllib.parse import unquote

from anyio import BrokenResourceError, ClosedResourceError
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
    get_external_upload_dirs,
    get_shared_task_execution_enabled,
    get_uploads_dir,
)
from ...core.agent.checkpoint import (
    CHECKPOINT_EVENT_TYPE,
)
from ...core.agent.trace import TraceEvent, TraceHandler
from ...core.execution_scope import (
    ExecutionScope,
    resolve_execution_scope,
)
from ...core.file_ref import build_file_ref
from ...core.file_storage.keys import (
    build_task_output_storage_key,
    build_upload_storage_key,
)
from ...core.runtime_performance import (
    increment_counter as increment_performance_counter,
)
from ...core.runtime_performance import (
    observe_duration,
    observe_value,
)
from ..models.chat_message import TaskChatMessage
from ..models.database import (
    get_db,
    get_session_local,
    release_db_connection_if_clean,
)
from ..models.task import Task
from ..models.uploaded_file import UploadedFile
from ..models.user import User
from ..services import task_command_execution as command_execution_service
from ..services import task_start as task_start_service
from ..services.assistant_history_safety import (
    assistant_history_has_safe_ancillary_payload,
    client_safe_assistant_history_content,
)
from ..services.assistant_question_replay import load_transcript_replay
from ..services.chat_history_service import (
    DELIVERY_COMPLETED,
    DELIVERY_FAILED,
)
from ..services.client_error_messages import (
    CLIENT_SAFE_TASK_FAILURE,
    ClientErrorCode,
    client_error_message,
)
from ..services.db_runtime import (
    cancel_and_drain_async_task,
    run_db_io_cancellation_safe,
)
from ..services.file_reference_output_service import (
    reconcile_assistant_file_references,
)
from ..services.file_turn import (
    bind_turn_files,
    resolve_turn_file_infos,
)
from ..services.hot_path_cache import (
    cache_get,
    cache_set,
    cache_version_token,
    task_cache_ttl_seconds,
    web_task_history_key,
)
from ..services.llm_utils import AutoModelUnavailableError
from ..services.managed_file_ref import (
    DurableObjectIntegrityError,
    DurableStorageOperationError,
    log_durable_storage_fault,
)
from ..services.mcp_runtime import (
    MCPBuiltinOAuthActorPolicyRequiredError,
)
from ..services.public_trace_events import (
    is_audit_only_trace_data,
    normalize_public_trace_event,
    public_task_trace_filter,
)
from ..services.task_command_execution import _read_task_error_payload_offloop
from ..services.task_command_transport import (
    COMMAND_FAILED,
    EnqueuedTaskCommand,
    TaskCommandKind,
    TaskCommandTaskMissing,
    dispatch_task_command_promptly,
    enqueue_task_command,
)

# The v1 SSE endpoint imports this shared predicate from this module.
from ..services.task_event_state import (
    _is_versioned_task_event as _is_versioned_task_event,
)
from ..services.task_event_state import (
    _with_current_task_control_state,
    _with_task_control_state_snapshot,
)
from ..services.task_events import (
    CommandReply,
    discard_command_reply,
    set_task_audience_probe,
    set_task_command_delivery,
    set_task_event_sink,
)
from ..services.task_execution import (
    ClientVisibleError,
    ClientVisibleValidationError,
    _add_file_link_aliases,
    _agent_outbound_event_type,
    _build_output_file_id,
    _normalize_workspace_relative_path,
    _output_path_in_current_task_scope,
    _resolve_output_storage_path,
    _rewrite_file_links_to_file_id,
    _set_file_link_alias,
    _task_user_id,
    _uploaded_file_record_in_task_scope,
    _waiting_or_paused_event_fields,
    _workspace_category_from_relative_path,
    client_safe_error_message,
    create_stream_event,
)
from ..services.task_execution_controller import (
    task_control_snapshot,
)
from ..services.task_execution_controller import (
    task_execution_controller as task_execution_controller,  # Tests patch this API module's controller.snapshot.
)
from ..services.task_interaction_read import get_pending_interaction_question
from ..services.task_runtime import (
    mcp_runtime_authorization_policy_required,
    task_extension_bindings_from_agent_config,
)
from ..services.task_socket_writer import TaskSocketWriter
from ..services.uploaded_file_store import (
    UploadedFileStore,
    UploadedFileVersionConflict,
    compensate_staged_uploaded_files,
    stage_uploaded_file_from_local_path,
)
from ..tracing import create_ephemeral_tracer
from ..user_isolated_memory import UserContext
from ..utils.db_timezone import safe_timestamp_to_unix
from .websocket_auth import (
    WebSocketPrincipal,
    _WebSocketAuthenticationTerminated,
    get_authenticated_user,
    send_websocket_authentication_infrastructure_failure,
)

logger = logging.getLogger(__name__)


def _make_command_reply(websocket: WebSocket) -> CommandReply:
    async def reply(message: dict[str, Any]) -> None:
        try:
            if message.get("type") == "task_id_updated":
                manager.move_connection(websocket, int(message["new_task_id"]))
            await manager.send_personal_message(message, websocket)
        except WebSocketDisconnect as exc:
            raise ConnectionError(str(exc)) from exc

    return reply


async def send_message_delivery(
    websocket: WebSocket,
    *,
    client_message_id: str | None,
    turn_id: str,
    accepted: bool,
    message: str | None = None,
    error_code: str | None = None,
    retry_with_new_id: bool = False,
    rejection_outcome: Literal["not_accepted", "outcome_unknown"] | None = None,
) -> None:
    await command_execution_service.send_message_delivery(
        _make_command_reply(websocket),
        client_message_id=client_message_id,
        turn_id=turn_id,
        accepted=accepted,
        message=message,
        error_code=error_code,
        retry_with_new_id=retry_with_new_id,
        rejection_outcome=rejection_outcome,
    )


CHECKPOINT_EVENT_TYPE_NAME = str(CHECKPOINT_EVENT_TYPE)


# Exception text can carry file paths, SQL fragments, provider payloads and
# other internals. It reaches anonymous widget/share visitors through both the
# error bubble and the message_rejected ack, so an *incidental* validation
# failure only ever surfaces as this fixed string; the detail stays in the log.
#
# RuntimeError text follows the same rule for every audience. The task-wide
# broadcast, rejection ack, and personal error bubble expose stable safe
# fields; provider responses and internal paths remain operator-only logs.


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


def _scope_segments_for_task(task_id: Any) -> tuple[str, ...]:
    """workspace_segments of the task's resolved ExecutionScope ((),
    when unscoped) — for storage-key composition outside the turn context.

    A None ``task_id`` (e.g. the legacy-preview backfill, whose owner
    inference may find a user but no task) means there is no task identity
    to resolve a scope from — unscoped, never the string ``"None"``.

    Fails closed (its only caller, ``_register_legacy_preview_isolated``,
    uses these segments to compose the storage key for a brand-new durable
    object): choosing that namespace is an authority decision, and a
    resolver/snapshot mismatch here must not be downgraded to either side's
    guess -- ``ExecutionScopeAuthorityError`` propagates instead.
    """
    if task_id is None:
        return ()
    scope = resolve_execution_scope(task_id)
    return scope.workspace_segments if scope is not None else ()


def _normalize_file_outputs(
    db: Session,
    task_id: int,
    task_user_id: int,
    file_outputs: Any,
) -> tuple[list[Any], Dict[str, str]]:
    """Project historical file outputs without filesystem or durable writes."""

    if isinstance(file_outputs, str):
        file_outputs = [file_outputs] if file_outputs.strip() else []
    if not isinstance(file_outputs, list):
        return [], {}
    if not file_outputs:
        return [], {}

    parsed_outputs: list[tuple[Any, str, str, tuple[str, ...], str]] = []
    candidate_file_ids: set[str] = set()
    candidate_storage_paths: set[str] = set()
    candidate_workspace_paths: set[str] = set()
    for item in file_outputs:
        item_file_id = ""
        item_filename = ""
        item_relative_path = ""
        raw_paths: list[str] = []
        if isinstance(item, str):
            raw_paths = [item]
        elif isinstance(item, dict):
            if isinstance(item.get("file_id"), str):
                item_file_id = str(item["file_id"]).strip()
            if isinstance(item.get("filename"), str):
                item_filename = str(item["filename"])
            for key in ("file_path", "download_path", "relative_path", "path"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    raw_paths.append(value.strip())
                    if key == "relative_path":
                        item_relative_path = value
        else:
            parsed_outputs.append((item, "", "", (), ""))
            continue

        normalized_raw_paths = tuple(dict.fromkeys(raw_paths))
        parsed_outputs.append(
            (
                item,
                item_file_id,
                item_filename,
                normalized_raw_paths,
                item_relative_path,
            )
        )
        if item_file_id:
            candidate_file_ids.add(item_file_id)
        candidate_storage_paths.update(normalized_raw_paths)
        if item_relative_path.strip():
            candidate_workspace_paths.add(
                _normalize_workspace_relative_path(item_relative_path)
            )

    identity_filters = []
    if candidate_file_ids:
        identity_filters.append(UploadedFile.file_id.in_(candidate_file_ids))
    if candidate_storage_paths:
        identity_filters.append(UploadedFile.storage_path.in_(candidate_storage_paths))
    if candidate_workspace_paths:
        identity_filters.append(
            UploadedFile.workspace_relative_path.in_(candidate_workspace_paths)
        )
    candidate_records = (
        db.query(UploadedFile).filter(or_(*identity_filters)).all()
        if identity_filters
        else []
    )
    by_file_id = {
        str(record.file_id): record
        for record in candidate_records
        if record.file_id is not None
    }
    by_storage_path = {
        str(record.storage_path): record
        for record in candidate_records
        if record.storage_path is not None
    }
    by_workspace_path: dict[str, list[UploadedFile]] = {}
    for record in candidate_records:
        workspace_relative_path = getattr(record, "workspace_relative_path", None)
        if isinstance(workspace_relative_path, str) and workspace_relative_path:
            by_workspace_path.setdefault(workspace_relative_path, []).append(record)

    normalized_outputs: list[Any] = []
    path_to_file_id: Dict[str, str] = {}

    def add_normalized_output(
        file_record: UploadedFile,
        fallback_filename: str,
        raw_paths: tuple[str, ...],
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

    def path_claims_current_task_output(value: str) -> bool:
        parts = Path(value).parts
        return any(
            _output_path_in_current_task_scope(
                "/".join(parts[index:]),
                task_id,
                task_user_id,
            )
            for index in range(len(parts))
        )

    for (
        original_item,
        item_file_id,
        item_filename,
        candidate_raw_paths,
        item_relative_path,
    ) in parsed_outputs:
        file_record: UploadedFile | None = None
        found_by_file_id = False
        if item_file_id:
            file_record = by_file_id.get(item_file_id)
            found_by_file_id = file_record is not None

        if file_record is None:
            file_record = next(
                (
                    by_storage_path[raw_path]
                    for raw_path in candidate_raw_paths
                    if raw_path in by_storage_path
                ),
                None,
            )

        if file_record is None and item_relative_path:
            workspace_relative_path = _normalize_workspace_relative_path(
                item_relative_path
            )
            scoped_workspace_records = [
                record
                for record in by_workspace_path.get(workspace_relative_path, ())
                if _uploaded_file_record_in_task_scope(
                    record,
                    task_id,
                    task_user_id,
                )
                and record.storage_status != "compensating"
            ]
            if len(scoped_workspace_records) == 1:
                file_record = scoped_workspace_records[0]

        if file_record is None:
            normalized_outputs.append(deepcopy(original_item))
            continue

        if (
            not _uploaded_file_record_in_task_scope(
                file_record,
                task_id,
                task_user_id,
            )
            or file_record.storage_status == "compensating"
        ):
            logger.warning(
                "Skipping historical file output outside task/user scope: %s",
                item_file_id or candidate_raw_paths,
            )
            continue

        if not found_by_file_id:
            workspace_category = getattr(file_record, "workspace_category", None)
            if workspace_category not in (None, "output") or (
                workspace_category is None
                and not any(
                    path_claims_current_task_output(path)
                    for path in (
                        *candidate_raw_paths,
                        str(getattr(file_record, "storage_path", "") or ""),
                    )
                    if path
                )
            ):
                logger.warning(
                    "Skipping registered file output outside output category: %s",
                    getattr(file_record, "file_id", item_file_id),
                )
                continue

        if item_file_id:
            path_to_file_id[item_file_id] = str(file_record.file_id)
        add_normalized_output(
            file_record,
            item_filename,
            candidate_raw_paths,
        )

        if item_relative_path:
            _add_file_link_aliases(
                path_to_file_id,
                _normalize_workspace_relative_path(item_relative_path),
                str(file_record.file_id),
            )

    return normalized_outputs, path_to_file_id


def _normalize_task_file_outputs(
    db: Session,
    task: Any,
    file_outputs: Any,
    *,
    task_id: Optional[int] = None,
    task_user_id: Optional[int] = None,
) -> tuple[list[Any], Dict[str, str]]:
    """Project only already-registered ``file_outputs`` for historical replay.

    History is a read path. Missing legacy metadata is deliberately left
    unresolved here; durable backfill belongs to a separately fenced
    reconciler, never to a cache-miss replay.
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
            from ..services.task_event_trace_handler import get_event_type_mapping

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


class _LegacyPreviewRegistrationError(RuntimeError):
    """A legacy preview path cannot be registered for a public redirect."""


def _register_legacy_preview_isolated(legacy_path: str) -> str:
    """Register one legacy local preview without overlapping DB and file I/O.

    Owner discovery and the final insert each own a short Session. Durable
    staging happens between those phases, with no Session alive. The final
    transaction revalidates the unique ``storage_path`` and task ownership
    before using :class:`UploadedFileStore`'s optimistic insert contract.
    """

    resolved_info = _resolve_legacy_preview_storage_path(legacy_path)
    if resolved_info is None:
        raise _LegacyPreviewRegistrationError("Legacy preview target not found")
    resolved_path, relative_path = resolved_info

    SessionLocal = get_session_local()
    with SessionLocal() as lookup_db:
        existing = (
            lookup_db.query(UploadedFile)
            .filter(UploadedFile.storage_path == str(resolved_path))
            .first()
        )
        if existing is not None:
            return str(existing.file_id)

        owner_info = _infer_owner_from_relative_path(lookup_db, relative_path)
        if owner_info is None:
            raise _LegacyPreviewRegistrationError(
                "Cannot infer owner for legacy preview path"
            )
        owner_user_id, task_id = owner_info

    generated_file_id = _build_output_file_id(relative_path)
    scope_segments = _scope_segments_for_task(task_id)
    workspace_relative_path = _normalize_workspace_relative_path(relative_path)
    workspace_category = _workspace_category_from_relative_path(workspace_relative_path)
    storage_key = (
        build_task_output_storage_key(
            owner_user_id,
            task_id,
            generated_file_id,
            (
                f"_versions/{uuid.uuid4().hex}/"
                f"{workspace_relative_path or resolved_path.name}"
            ),
            scope_segments=scope_segments,
        )
        if task_id is not None
        else build_upload_storage_key(
            owner_user_id,
            generated_file_id,
            resolved_path.name,
            scope_segments=scope_segments,
        )
    )
    staged = stage_uploaded_file_from_local_path(
        local_path=resolved_path,
        user_id=owner_user_id,
        file_id=generated_file_id,
        task_id=task_id,
        filename=resolved_path.name,
        mime_type=None,
        storage_key=storage_key,
        workspace_relative_path=workspace_relative_path,
        workspace_category=workspace_category,
        execution_scope=(
            ExecutionScope(
                workspace_segments=scope_segments,
                isolate_external_dirs=bool(scope_segments),
            )
            if scope_segments
            else None
        ),
    )

    metadata_committed = False
    try:
        with SessionLocal() as write_db:
            current = (
                write_db.query(UploadedFile)
                .filter(UploadedFile.storage_path == str(resolved_path))
                .with_for_update()
                .first()
            )
            if current is not None:
                return str(current.file_id)

            if task_id is not None:
                current_owner = (
                    write_db.query(Task.user_id)
                    .filter(Task.id == task_id)
                    .with_for_update()
                    .scalar()
                )
                if current_owner is None or int(current_owner) != owner_user_id:
                    raise _LegacyPreviewRegistrationError(
                        "Legacy preview task ownership changed during registration"
                    )
            elif (
                write_db.query(User.id).filter(User.id == owner_user_id).scalar()
                is None
            ):
                raise _LegacyPreviewRegistrationError(
                    "Legacy preview owner no longer exists"
                )

            try:
                applied = UploadedFileStore(write_db).upsert_already_durable(
                    staged,
                    expected=None,
                )
                write_db.commit()
            except UploadedFileVersionConflict:
                # A competing request can win the unique storage_path insert
                # after our revalidation. Re-read that winner and discard only
                # our own immutable staged object.
                write_db.rollback()
                winner = (
                    write_db.query(UploadedFile)
                    .filter(UploadedFile.storage_path == str(resolved_path))
                    .first()
                )
                if winner is None:
                    raise
                return str(winner.file_id)
            metadata_committed = True
            return applied.snapshot.file_id
    finally:
        if not metadata_committed:
            try:
                failed_file_ids = compensate_staged_uploaded_files((staged,))
            except Exception:
                logger.exception(
                    "Failed to compensate staged legacy preview object %s",
                    staged.file_id,
                )
            else:
                if failed_file_ids:
                    logger.warning(
                        "Retained staged legacy preview object %s because "
                        "reference or deletion state was unknown",
                        staged.file_id,
                    )


@ws_router.get("/preview/{legacy_path:path}", response_model=None)
async def redirect_legacy_preview(
    legacy_path: str,
    db: Session = Depends(get_db),
) -> Any:
    if not release_db_connection_if_clean(db):
        raise RuntimeError(
            "Cannot register a legacy preview while the request database "
            "session has pending writes"
        )
    try:
        file_id = await run_db_io_cancellation_safe(
            lambda: _register_legacy_preview_isolated(legacy_path)
        )
    except _LegacyPreviewRegistrationError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return RedirectResponse(
        url=f"/api/files/public/preview/{file_id}",
        status_code=307,
    )


# Connection manager
class _CommandOriginRegistry:
    """(task_id, command_id) -> the exact socket that submitted the command.

    Recorded at the ingress handler, where the connection is the verified
    origin, and consulted by the durable executor in place of any guess:
    origin is never inferred from task membership, actor id, guest id, or
    connection order. Same-worker only, by design - when the command executes
    after a worker restart or on a different worker, ``resolve`` finds nothing
    and the executor degrades to a discarding socket, so personal detail is
    dropped rather than sent to an unverified connection.

    ``command_id`` is client-supplied and only unique per task (the DB carries
    a ``(task_id, command_id)`` uniqueness constraint), so the key is the pair,
    never the id alone - otherwise a command_id shared across two tasks would
    let one void or overwrite the other's entry.

    First registration wins. A second connection on the same task cannot
    overwrite an existing origin by resubmitting the same command_id: the
    enqueue dedupe returns the in-flight row for such a resubmission, so
    without this rule a co-tenant on a public/share task could redirect
    another sender's error detail to itself. Re-registering the *same* socket
    is idempotent.

    Only the ingress that *created* the durable row registers (callers gate on
    ``EnqueuedTaskCommand.created``); a payload-matching duplicate never binds,
    which is what keeps a co-tenant, a post-disconnect resubmission, or a
    duplicate handled on another worker from acquiring the origin.

    Lifecycle: an entry dies with its socket (``discard_socket`` from
    ``ConnectionManager.disconnect`` and ``detach_task_connections``) or with
    its command's terminal outcome (``discard_command`` from the durable
    dispatch wrapper), whichever comes first. A deferred command that will
    retry keeps its entry. An entry whose command is claimed by another worker
    is never resolved here (wrong worker) and its local cleanup never runs, so
    to bound that case the store is an LRU capped at ``_MAX_ORIGINS``: an
    eviction just makes ``resolve`` miss and the executor degrade to the safe
    discard, so a socket that never disconnects while its commands always run
    elsewhere can cost at most the wording on the oldest few, never unbounded
    memory.
    """

    _MAX_ORIGINS = 4096

    def __init__(self) -> None:
        self._origins: OrderedDict[tuple[int, str], Any] = OrderedDict()

    def register(self, command_id: str, websocket: Any, task_id: int) -> None:
        if not command_id:
            return
        key = (int(task_id), command_id)
        existing = self._origins.get(key)
        if existing is not None and existing is not websocket:
            # First registration wins; a resubmission from another socket
            # must not capture this command's origin.
            return
        self._origins[key] = websocket
        self._origins.move_to_end(key)
        while len(self._origins) > self._MAX_ORIGINS:
            # Oldest first: eviction degrades that command to the safe discard,
            # it never reroutes detail.
            self._origins.popitem(last=False)

    def resolve(self, command_id: str, task_id: int) -> Any | None:
        websocket = self._origins.get((int(task_id), command_id))
        if websocket is None:
            return None
        if not manager.is_connection_registered(websocket, int(task_id)):
            return None
        return websocket

    def reply_for(self, command_id: str, task_id: int) -> CommandReply:
        websocket = self.resolve(command_id, task_id)
        if websocket is None:
            return discard_command_reply
        return _make_command_reply(websocket)

    def discard_command(self, command_id: str, task_id: int) -> None:
        self._origins.pop((int(task_id), command_id), None)

    def has(self, command_id: str, task_id: int) -> bool:
        """Whether an origin is currently recorded for this command."""
        return (int(task_id), command_id) in self._origins

    def discard_socket(self, websocket: Any) -> None:
        for key in [k for k, ws in self._origins.items() if ws is websocket]:
            del self._origins[key]


_command_origins = _CommandOriginRegistry()


class ConnectionManager:
    def __init__(self) -> None:
        # task_id -> List[WebSocket]
        self.active_connections: Dict[int, List[WebSocket]] = {}
        self._writers: dict[WebSocket, TaskSocketWriter] = {}
        self._stream_reconciler: asyncio.Task[None] | None = None
        # WebSocket -> current task_id
        self._connection_task_ids: Dict[WebSocket, int] = {}

    async def connect(self, websocket: WebSocket, task_id: int) -> None:
        await websocket.accept()
        self.register_connection(websocket, task_id)

    def register_connection(self, websocket: WebSocket, task_id: int) -> None:
        """Register an already-accepted websocket for task broadcasts."""
        current_task_id = self._connection_task_ids.get(websocket)
        if current_task_id is not None and current_task_id != task_id:
            self._remove_from_task(websocket, current_task_id)
        if task_id not in self.active_connections:
            self.active_connections[task_id] = []
        if websocket not in self.active_connections[task_id]:
            self.active_connections[task_id].append(websocket)
        self._connection_task_ids[websocket] = task_id
        if get_shared_task_execution_enabled() and websocket not in self._writers:
            self._writers[websocket] = TaskSocketWriter(
                websocket, lambda: self.disconnect(websocket)
            )

    def _remove_from_task(self, websocket: WebSocket, task_id: int) -> None:
        if task_id in self.active_connections:
            try:
                self.active_connections[task_id].remove(websocket)
                if not self.active_connections[task_id]:
                    del self.active_connections[task_id]
            except ValueError:
                pass

    def disconnect(self, websocket: WebSocket) -> None:
        writer = self._writers.pop(websocket, None)
        if writer is not None:
            writer.stop()
        if get_shared_task_execution_enabled():
            from ..services.task_event_bridge import get_task_event_bridge

            get_task_event_bridge().discard_recipient(websocket)
        task_id = self._connection_task_ids.pop(websocket, None)
        if task_id is not None:
            self._remove_from_task(websocket, task_id)
        _command_origins.discard_socket(websocket)

    def detach_task_connections(self, task_id: int) -> List[WebSocket]:
        """Remove and return every connection currently owned by a task."""
        connections = self.active_connections.pop(task_id, [])
        for connection in connections:
            if self._connection_task_ids.get(connection) == task_id:
                del self._connection_task_ids[connection]
            self.disconnect(connection)
        return connections

    def connections_for_task(self, task_id: int) -> List[WebSocket]:
        """Return a stable snapshot of a task's current connections."""
        return self.active_connections.get(task_id, []).copy()

    def has_connections_for_task(self, task_id: int) -> bool:
        """Return whether a task currently has a registered connection."""

        return bool(self.active_connections.get(task_id))

    def connection_count(self) -> int:
        """Return the current number of task WebSocket registrations."""

        return len(self._connection_task_ids)

    def is_connection_registered(self, websocket: WebSocket, task_id: int) -> bool:
        """Return whether a connection is still owned by the given task."""
        return self._connection_task_ids.get(websocket) == task_id

    def move_connection(self, websocket: WebSocket, new_task_id: int) -> None:
        """Move a WebSocket connection from one task_id to another"""
        old_task_id = self._connection_task_ids.get(websocket)
        self.register_connection(websocket, new_task_id)
        logger.info(
            f"Moved WebSocket connection from task {old_task_id} to {new_task_id}"
        )

    async def send_personal_message(self, message: dict, websocket: WebSocket) -> None:
        versioned_message = await _with_current_task_control_state(message)
        writer = self._writers.get(websocket)
        if writer is not None:
            delivery = writer.enqueue(json.dumps(versioned_message), acknowledge=True)
            await cast(asyncio.Future[None], delivery)
        else:
            await websocket.send_text(json.dumps(versioned_message))

    async def deliver_shared_event(self, message: dict[str, Any], task_id: int) -> None:
        encoded = json.dumps(message)
        for connection in self.connections_for_task(task_id):
            writer = self._writers.get(connection)
            if writer is None:
                logger.warning(
                    "Shared task event skipped: connection has no writer task_id=%s",
                    task_id,
                )
            else:
                try:
                    writer.enqueue(encoded)
                except ConnectionError:
                    self.disconnect(connection)

    def start_stream_reconciliation(self) -> None:
        self._stream_reconciler = asyncio.create_task(self._reconcile_streams())

    async def stop_stream_reconciliation(self) -> None:
        if self._stream_reconciler is not None:
            self._stream_reconciler.cancel()
            await asyncio.gather(self._stream_reconciler, return_exceptions=True)
            self._stream_reconciler = None

    async def _reconcile_streams(self) -> None:
        from ..services.task_stream_snapshot import load_task_stream_snapshots

        while True:
            try:
                task_ids = list(self.active_connections)
                snapshots = await run_db_io_cancellation_safe(
                    lambda: load_task_stream_snapshots(task_ids)
                )
                for snapshot in snapshots:
                    await self.deliver_shared_event(snapshot, snapshot["task_id"])
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "Shared stream state reconciliation failed", exc_info=True
                )
            await asyncio.sleep(5)

    async def shared_stream_status(self, kind: str) -> None:
        for task_id in list(self.active_connections):
            await self.deliver_shared_event({"type": kind, "task_id": task_id}, task_id)

    async def broadcast_to_task(self, message: dict, task_id: int) -> None:
        if get_shared_task_execution_enabled():
            from ..services.task_event_bridge import get_task_event_bridge

            await get_task_event_bridge().publish(message, task_id)
            return
        has_connections = self.has_connections_for_task(task_id)
        increment_performance_counter(
            "xagent.websocket.broadcast.calls",
            attributes={"outcome": "connected" if has_connections else "empty"},
        )
        if not has_connections:
            return

        with observe_duration("xagent.websocket.broadcast.duration"):
            versioned_message = await _with_current_task_control_state(
                message,
                fallback_task_id=task_id,
            )
            # Registrations can change while message enrichment yields. Take
            # the delivery snapshot afterwards so newly connected clients are
            # included; membership is checked again before every send below.
            connections = self.connections_for_task(task_id)
            # Fanout measures intended recipients, not successful deliveries.
            observe_value(
                "xagent.websocket.broadcast.fanout",
                len(connections),
                unit="{connection}",
            )
            try:
                encoded_message = json.dumps(versioned_message)
            except Exception:
                # Invalid message data does not imply a broken connection.
                logger.error("Failed to serialize WebSocket broadcast", exc_info=True)
                raise
            observe_value(
                "xagent.websocket.payload.size",
                # ensure_ascii=True makes character count equal byte count.
                len(encoded_message),
                unit="By",
            )
            for connection in connections:
                if not self.is_connection_registered(connection, task_id):
                    continue
                try:
                    await connection.send_text(encoded_message)
                    increment_performance_counter("xagent.websocket.messages.sent")
                except (
                    BrokenResourceError,
                    ClosedResourceError,
                    ConnectionError,
                    WebSocketDisconnect,
                    RuntimeError,
                ) as e:
                    increment_performance_counter(
                        "xagent.websocket.send.errors",
                        attributes={"error.type": "connection"},
                    )
                    # Network connection error, remove disconnected connection
                    logger.warning(f"Connection error for task {task_id}: {e}")
                    self.disconnect(connection)
                except Exception as e:
                    increment_performance_counter(
                        "xagent.websocket.send.errors",
                        attributes={"error.type": "unexpected"},
                    )
                    # Other errors should not be silently handled, log and re-raise
                    logger.error(
                        f"Unexpected error broadcasting to task {task_id}: {e}"
                    )
                    # Remove disconnected connection but preserve error propagation
                    self.disconnect(connection)
                    raise


# Global connection manager
manager = ConnectionManager()


# The execution service publishes events; this process owns socket delivery.
async def _deliver_task_event(message: dict[str, Any], task_id: int) -> None:
    await manager.broadcast_to_task(message, task_id)


# The execution service asks whether anyone is listening; this process owns
# the registry that knows. Registered after the sink above, and never before
# it: installing a sink clears any probe attached to a previous one.
def _task_event_audience(task_id: int) -> bool:
    return manager.has_connections_for_task(task_id)


set_task_event_sink(_deliver_task_event)
set_task_audience_probe(_task_event_audience)
set_task_command_delivery(_command_origins)


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


async def handle_chat_message(
    websocket: WebSocket, task_id: int, message_data: dict
) -> None:
    """Durably accept a chat command before acknowledging the client."""

    try:
        enqueued = await _enqueue_websocket_task_command(
            websocket=websocket,
            task_id=task_id,
            message_data=message_data,
            kind=TaskCommandKind.MESSAGE,
            command_id=command_execution_service._client_message_id(
                message_data.get("client_message_id")
            ),
            allow_missing_task=True,
        )
    except (
        MCPBuiltinOAuthActorPolicyRequiredError,
        PermissionError,
        ValueError,
    ) as exc:
        command_execution_service.log_client_facing_failure(
            exc, "Chat command rejected for task %s: %s", task_id
        )
        client_message_id = command_execution_service._client_message_id(
            message_data.get("client_message_id")
        )
        error_code = (
            exc.error_code
            if isinstance(exc, ClientVisibleError)
            else ClientErrorCode.MESSAGE_PROCESSING_FAILED
        )
        message = client_error_message(error_code)
        await send_message_delivery(
            websocket,
            client_message_id=client_message_id,
            turn_id=client_message_id or str(uuid.uuid4()),
            accepted=False,
            message=message,
            error_code=error_code.value,
            rejection_outcome="not_accepted",
        )
        await manager.send_personal_message(
            {"type": "error", "message": message, "error_code": error_code.value},
            websocket,
        )
        return
    if enqueued is None:
        # Legacy recovery path for a client still connected to a task that was
        # deleted. The existing handler creates the replacement task first;
        # subsequent commands use the durable transport normally.
        await command_execution_service.handle_missing_task_message(
            _make_command_reply(websocket), task_id, message_data
        )
        return
    if not enqueued.payload_matches:
        await send_message_delivery(
            websocket,
            client_message_id=command_execution_service._client_message_id(
                message_data.get("client_message_id")
            ),
            turn_id=enqueued.client_command_id,
            accepted=False,
            message=client_error_message(ClientErrorCode.MESSAGE_ID_CONFLICT),
            error_code=ClientErrorCode.MESSAGE_ID_CONFLICT.value,
            retry_with_new_id=True,
            rejection_outcome="not_accepted",
        )
        return
    if enqueued.status == COMMAND_FAILED:
        await send_message_delivery(
            websocket,
            client_message_id=command_execution_service._client_message_id(
                message_data.get("client_message_id")
            ),
            turn_id=enqueued.client_command_id,
            accepted=False,
            message=client_error_message(ClientErrorCode.MESSAGE_DELIVERY_FAILED),
            error_code=ClientErrorCode.MESSAGE_DELIVERY_FAILED.value,
            retry_with_new_id=True,
            rejection_outcome="not_accepted",
        )
        return
    await send_message_delivery(
        websocket,
        client_message_id=command_execution_service._client_message_id(
            message_data.get("client_message_id")
        ),
        turn_id=enqueued.client_command_id,
        accepted=True,
    )
    if enqueued.command_id:
        if enqueued.created and not get_shared_task_execution_enabled():
            # Only the ingress that created the durable row owns the origin.
            # A payload-matching duplicate (created=False) - a co-tenant
            # resubmission, one arriving after the creator disconnected, or one
            # handled on another worker - must never bind, or it could receive
            # the creator's raw error detail. Registered before dispatch so
            # local execution cannot outrun the binding.
            _command_origins.register(enqueued.client_command_id, websocket, task_id)
        await dispatch_task_command_promptly(
            command_execution_service.execute_durable_task_command,
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
    reply_host_id: str | None = None,
    reply_origin: str | None = None,
) -> EnqueuedTaskCommand | None:
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = db.query(Task).filter(Task.id == task_id).first()
        if task is None:
            if allow_missing_task:
                return None
            raise ClientVisibleValidationError(
                f"Task {task_id} not found",
                error_code=ClientErrorCode.TASK_UNAVAILABLE,
            )
        if not actor_is_admin and int(task.user_id) != actor_user_id:
            # Keep the permission-specific text for operator logs, but expose
            # the same code as a missing task so task IDs cannot be probed.
            raise command_execution_service.ClientVisiblePermissionError(
                f"Access denied: Task {task_id} does not belong to you",
                error_code=ClientErrorCode.TASK_UNAVAILABLE,
            )
        if mcp_runtime_authorization_policy_required(task.agent_config):
            raise MCPBuiltinOAuthActorPolicyRequiredError(
                f"Task {task_id} is actor-marked; generic task commands are unsupported"
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
        try:
            result = enqueue_task_command(
                db,
                task_id=task_id,
                actor_user_id=actor_user_id,
                command_id=command_id,
                kind=kind,
                payload=payload,
                **(
                    {"reply_host_id": reply_host_id, "reply_origin": reply_origin}
                    if reply_host_id is not None
                    else {}
                ),
            )
        except TaskCommandTaskMissing as exc:
            # The row was deleted after the check above, so route this through
            # the same sentinel as a task that was already gone. Otherwise the
            # caller rejects the delivery instead of creating a replacement.
            if allow_missing_task:
                return None
            # Same answer as the direct lookup above, in the same wording.
            # The sentinel is a bare ValueError, so re-raising it as-is let
            # the pause/resume catch redact "not found" on this race alone.
            # Converted at this boundary only; transport semantics unchanged.
            raise ClientVisibleValidationError(
                str(exc),
                error_code=ClientErrorCode.TASK_UNAVAILABLE,
            ) from exc
        return result


async def _enqueue_websocket_task_command(
    *,
    websocket: WebSocket | None = None,
    task_id: int,
    message_data: dict[str, Any],
    kind: TaskCommandKind,
    command_id: str | None = None,
    allow_missing_task: bool = False,
) -> EnqueuedTaskCommand | None:
    user = message_data.get("user")
    if user is None:
        raise ClientVisibleValidationError(
            "User authentication required for task command",
            error_code=ClientErrorCode.AUTHENTICATION_REQUIRED,
        )
    resolved_command_id = command_id or f"{kind.value}:{uuid.uuid4()}"
    # User ORM instances and server-only authentication fields are never put
    # into the JSON inbox. The consumer re-resolves the actor by id.
    payload = {
        key: value
        for key, value in message_data.items()
        if key not in {"user", "user_id"} and not key.startswith("_durable_")
    }
    if "scope" in payload:
        # ``scope`` routes a durable command to a non-first-party execution
        # core (see ``command_execution_service._execute_durable_task_command``); only server-side
        # producers may name one. A client frame that carries it is refused
        # rather than silently stripped, so the sender learns the frame was
        # not accepted as written.
        raise ClientVisibleValidationError(
            "Reserved field 'scope' is not accepted from clients",
            error_code=ClientErrorCode.INVALID_MESSAGE,
        )
    if kind == TaskCommandKind.MESSAGE:
        # The durable command identity is also the delivery/turn identity.
        # This remains stable across retries even when an API client omitted
        # or supplied an invalid client_message_id.
        payload["client_message_id"] = resolved_command_id
    route = {}
    bridge = None
    token = None
    if get_shared_task_execution_enabled():
        from ..services.task_event_bridge import TaskEventBridge, get_task_event_bridge

        bridge = get_task_event_bridge()
        if kind == TaskCommandKind.MESSAGE and not bridge.ready.is_set():
            raise ClientVisibleValidationError(
                "Task event transport is unavailable; retry after reconnection",
                error_code=ClientErrorCode.TASK_UNAVAILABLE,
            )
        if websocket is not None and bridge.ready.is_set():

            async def reply(message: dict[str, Any]) -> None:
                if not manager.is_connection_registered(websocket, task_id):
                    raise ConnectionError("Original task connection is unavailable")
                await _make_command_reply(websocket)(message)

            token = bridge.register_origin(
                task_id, resolved_command_id, reply, recipient=websocket
            )
            route = {"reply_host_id": bridge.host_id, "reply_origin": token}
    result = await asyncio.to_thread(
        _enqueue_websocket_task_command_sync,
        task_id=int(task_id),
        actor_user_id=int(user.id),
        actor_is_admin=bool(user.is_admin),
        command_id=resolved_command_id,
        kind=kind,
        payload=payload,
        allow_missing_task=allow_missing_task,
        **route,
    )
    if token is not None and (result is None or not result.created):
        cast(TaskEventBridge, bridge).discard_origin(token)
    return result


@dataclass(frozen=True)
class _LegacyExecuteTaskRequest:
    """Detached input needed before scheduling an existing task execution."""

    task_id: int
    task_owner_user_id: int
    task_source: str | None
    task_description: str
    task_context: dict[str, Any]
    task_info: dict[str, Any]
    created_at: datetime | None


def _load_legacy_execute_task_request_sync(
    task_id: int,
    *,
    actor_user_id: int,
    actor_is_admin: bool,
) -> _LegacyExecuteTaskRequest | None:
    """Authorize and detach legacy execution metadata in one short Session."""
    from ..models.agent import Agent

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task_query = db.query(Task).filter(Task.id == task_id)
        if not actor_is_admin:
            task_query = task_query.filter(Task.user_id == actor_user_id)
        task = task_query.first()
        if task is None:
            return None

        (
            model_id,
            small_fast_model_id,
            visual_model_id,
            compact_model_id,
        ) = command_execution_service._resolve_task_llm_ids(task, db)
        agent_name: str | None = None
        agent_logo_url: str | None = None
        if task.agent_id is not None:
            agent_fields = (
                db.query(Agent.name, Agent.logo_url)
                .filter(Agent.id == task.agent_id)
                .first()
            )
            if agent_fields is not None:
                agent_name = str(agent_fields[0])
                agent_logo_url = (
                    str(agent_fields[1]) if agent_fields[1] is not None else None
                )

        task_context: dict[str, Any] = {}
        if task.execution_mode:
            task_context["execution_mode"] = str(task.execution_mode)
        if task.process_description:
            task_context["process_description"] = str(task.process_description)
        if task.examples:
            task_context["examples"] = deepcopy(task.examples)

        created_at = cast(datetime | None, task.created_at)
        return _LegacyExecuteTaskRequest(
            task_id=int(task.id),
            task_owner_user_id=int(task.user_id),
            task_source=str(task.source) if task.source is not None else None,
            task_description=str(task.description),
            task_context=task_context,
            task_info={
                "id": int(task.id),
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
                "agent_name": agent_name,
                "agent_logo_url": agent_logo_url,
                "runtime_extension_bindings": list(
                    task_extension_bindings_from_agent_config(task.agent_config)
                ),
                "created_at": safe_timestamp_to_unix(task.created_at)
                if task.created_at
                else None,
                "updated_at": safe_timestamp_to_unix(task.updated_at)
                if task.updated_at
                else None,
            },
            created_at=created_at,
        )


async def handle_execute_task(
    websocket: WebSocket, task_id: int, message_data: dict
) -> None:
    """Handle the legacy execution command without retaining a DB Session."""
    try:
        user = message_data.get("user")
        authorized_task_id: int | None = None
        if not user:
            raise ClientVisibleValidationError(
                "User authentication required for task execution",
                error_code=ClientErrorCode.AUTHENTICATION_REQUIRED,
            )
        actor_user_id = int(user.id)
        actor_is_admin = bool(user.is_admin)

        # Preserve the legacy protocol acknowledgement before task lookup.
        await manager.send_personal_message(
            {
                "type": "execution_started",
                "task_id": task_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            websocket,
        )

        request = await run_db_io_cancellation_safe(
            lambda: _load_legacy_execute_task_request_sync(
                task_id,
                actor_user_id=actor_user_id,
                actor_is_admin=actor_is_admin,
            )
        )
        if request is None:
            raise ClientVisibleValidationError(
                f"Task {task_id} not found or access denied",
                error_code=ClientErrorCode.TASK_UNAVAILABLE,
            )
        authorized_task_id = request.task_id

        await manager.broadcast_to_task(
            create_stream_event(
                "task_info",
                request.task_id,
                request.task_info,
                request.created_at,
            ),
            request.task_id,
        )

        await task_start_service.execute_existing_task(
            task_id=request.task_id,
            task_owner_user_id=request.task_owner_user_id,
            task_source=request.task_source,
            task_description=request.task_description,
            context=request.task_context,
            actor_user_id=actor_user_id,
        )

    except (
        MCPBuiltinOAuthActorPolicyRequiredError,
        ValueError,
        KeyError,
        TypeError,
    ) as e:
        # Data validation and actor-policy errors are client-safe only when
        # explicitly marked with a stable code.
        error_code = (
            e.error_code
            if isinstance(e, ClientVisibleError)
            else ClientErrorCode.MESSAGE_PROCESSING_FAILED
        )
        message = client_error_message(error_code)
        command_execution_service.log_client_facing_failure(
            e, "Task execution rejected: %s"
        )
        timestamp = datetime.now(timezone.utc).isoformat()
        if authorized_task_id is not None:
            error_payload = await _read_task_error_payload_offloop(
                authorized_task_id,
                message,
                error_code=error_code.value,
            )
            await manager.broadcast_to_task(
                {
                    **error_payload,
                    "timestamp": timestamp,
                },
                authorized_task_id,
            )
        else:
            await manager.send_personal_message(
                {
                    "type": "error",
                    "message": message,
                    "error_code": error_code.value,
                    "timestamp": timestamp,
                },
                websocket,
            )
    except RuntimeError as e:
        # Runtime failures can contain provider responses, paths, and other
        # operator-only details. Keep those in the log and expose only the
        # stable client contract to every audience.
        message = CLIENT_SAFE_TASK_FAILURE
        logger.error("Runtime error in task execution: %s", e, exc_info=True)
        timestamp = datetime.now(timezone.utc).isoformat()
        if authorized_task_id is not None:
            error_payload = await _read_task_error_payload_offloop(
                authorized_task_id,
                CLIENT_SAFE_TASK_FAILURE,
                error_code=ClientErrorCode.TASK_EXECUTION_FAILED.value,
            )
            await manager.broadcast_to_task(
                {
                    **error_payload,
                    "timestamp": timestamp,
                },
                authorized_task_id,
            )
            await manager.send_personal_message(
                {
                    "type": "error",
                    "error_code": ClientErrorCode.TASK_EXECUTION_FAILED.value,
                    "message": client_error_message(
                        ClientErrorCode.TASK_EXECUTION_FAILED
                    ),
                    "timestamp": timestamp,
                },
                websocket,
            )
        else:
            await manager.send_personal_message(
                {
                    "type": "error",
                    "error_code": ClientErrorCode.TASK_EXECUTION_FAILED.value,
                    "message": client_error_message(
                        ClientErrorCode.TASK_EXECUTION_FAILED
                    ),
                    "timestamp": timestamp,
                },
                websocket,
            )
    except Exception as e:
        # Re-raised, but the callers do not own the stack: the chat endpoint
        # logs without exc_info and the public endpoints swallow entirely.
        logger.error("Unexpected error in task execution: %s", e, exc_info=True)
        raise


@dataclass(frozen=True)
class _HistoricalStreamSnapshot:
    """A complete, detached historical replay ready for network delivery."""

    events: tuple[dict[str, Any], ...]


def _load_historical_stream_snapshot_sync(
    task_id: int,
    *,
    actor_user_id: int,
    actor_is_admin: bool,
) -> _HistoricalStreamSnapshot | None:
    """Load, normalize, and cache one historical replay in a short Session."""
    try:
        # Load historical data directly from database
        from ..models.agent import Agent
        from ..models.database import get_db
        from ..models.task import Task, TaskStatus, TraceEvent
        from ..models.workforce import WorkforceRun

        db_gen = get_db()
        db = next(db_gen)

        try:
            # Get task basic info
            task = db.query(Task).filter(Task.id == task_id).first()
            if not task:
                logger.warning(f"Task {task_id} not found")
                return None

            # Verify user permissions
            if not task.user_id:
                logger.warning(f"Task {task_id} has no user association")
                return None

            # Verify user permissions - admin can access any task
            if not actor_is_admin and task.user_id != actor_user_id:
                logger.warning(
                    "User %s attempted to access task %s belonging to user %s",
                    actor_user_id,
                    task_id,
                    task.user_id,
                )
                return None

            is_workforce_run = (
                db.query(WorkforceRun.id)
                .filter(WorkforceRun.task_id == task_id)
                .first()
                is not None
            )
            trace_scope_filter = (
                TraceEvent.build_id.is_(None)
                if is_workforce_run
                else public_task_trace_filter(TraceEvent)
            )
            trace_scope = "workforce-top-level-v2" if is_workforce_run else "public-v2"

            max_trace_event_id = (
                db.query(func.max(TraceEvent.id))
                .filter(
                    TraceEvent.task_id == task_id,
                    trace_scope_filter,
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
            control_state = task_control_snapshot(task).as_dict()
            # Redis is synchronous I/O. Never spend its timeout budget while
            # pinning a database pool slot: this phase is read-only, so return
            # the connection before consulting the cache. The Session remains
            # usable and transparently re-checks out for a cache miss replay.
            cached = (
                cache_get(cache_key) if release_db_connection_if_clean(db) else None
            )
            if (
                isinstance(cached, dict)
                and cached.get("trace_scope") == trace_scope
                and cached.get("updated_at") == task_updated_at
                and cached.get("max_trace_event_id") == int(max_trace_event_id)
                and cached.get("max_chat_message_id") == int(max_chat_message_id)
                and isinstance(cached.get("events"), list)
            ):
                cached_events = tuple(
                    _with_task_control_state_snapshot(
                        cached_event,
                        task_id=task_id,
                        state=control_state,
                    )
                    for cached_event in cached["events"]
                    if isinstance(cached_event, dict)
                )
                return _HistoricalStreamSnapshot(events=cached_events)

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
            ) = command_execution_service._resolve_task_llm_ids(task, db)
            waiting_question = None
            waiting_interactions = None
            if task.status == TaskStatus.WAITING_FOR_USER:
                waiting_question, waiting_interactions = (
                    get_pending_interaction_question(db, task)
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
            cached_stream_events.append(task_event)

            # Replay only top-level task events. Delegated Agent internals can
            # be much larger than the manager trace and are loaded on demand by
            # the Workforce Agent-execution drawer.
            trace_events = (
                db.query(TraceEvent)
                .filter(
                    TraceEvent.task_id == task_id,
                    trace_scope_filter,
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
                    trace_file_outputs = normalized_event_data.get("file_outputs", [])
                    normalized_outputs, path_to_file_id = _normalize_task_file_outputs(
                        db,
                        None,
                        trace_file_outputs,
                        task_id=task_id,
                        task_user_id=int(task.user_id),
                    )
                    if "file_outputs" in normalized_event_data:
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

            # A waiting question is persisted as both a trace event and a
            # transcript row, and replay used to ship both (#2292). Keep the
            # row and drop the trace twin it claims; see
            # ``services.assistant_question_replay`` for why the row wins.
            transcript = load_transcript_replay(
                db,
                task_id=int(task_id),
                task_user_id=int(task.user_id),
                trace_events=trace_events,
                trace_data_by_event_id=normalized_trace_data_by_event_id,
            )

            for trace_event in trace_events:
                normalized_event_data = normalized_trace_data_by_event_id.get(
                    str(trace_event.event_id), trace_event.data
                )
                if _is_audit_only_trace_data(normalized_event_data):
                    continue
                if transcript.superseded(trace_event.event_id):
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

            for chat_message in transcript.rows:
                role = str(chat_message.role)
                content = str(chat_message.content or "").strip()
                if role == "assistant":
                    content = client_safe_assistant_history_content(
                        content=content,
                        message_type=str(chat_message.message_type),
                    )
                    content = reconcile_assistant_file_references(
                        db,
                        task_id=int(task_id),
                        user_id=int(task.user_id),
                        content=content,
                        records=transcript.file_reference_records,
                    )
                # Read attachments off the row so file-only turns (empty
                # content + non-empty attachments) survive replay and so the
                # chip metadata reaches the synthesized user_message event.
                assistant_ancillary_is_safe = role != "assistant" or (
                    assistant_history_has_safe_ancillary_payload(
                        str(chat_message.message_type)
                    )
                )
                _attachments_raw = (
                    chat_message.attachments if assistant_ancillary_is_safe else None
                )
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
                    if not transcript.paired(chat_message.id) and (
                        content
                        and (role, content, _attachment_fingerprint(row_attachments))
                        in trace_message_keys
                    ):
                        continue
                    interactions = (
                        chat_message.interactions
                        if assistant_ancillary_is_safe
                        else None
                    )
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
                            "event_id": transcript.event_id_for(
                                chat_message.id, f"chat_message_{chat_message.id}"
                            ),
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

                    # Add step_id at the top level if present (consistent with TaskEventTraceHandler)
                    if event_data.get("step_id"):
                        stream_event["step_id"] = str(event_data["step_id"])
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
            cached_stream_events.append(completion_event)

            # Historical trace replay can end with an in-flight event from before a
            # crash/restart, such as llm_call_start. Re-assert the current DB task
            # state after replay so stale running trace events do not keep the UI in
            # a running state.
            if task.status in {TaskStatus.PAUSED, TaskStatus.WAITING_FOR_USER}:
                event_type, default_message = _waiting_or_paused_event_fields(
                    task.status
                )
                question_message = None
                question_interactions = None
                if task.status == TaskStatus.WAITING_FOR_USER:
                    # Same task, same db session, no await between this branch
                    # and the task_info block above: reuse its already-fetched
                    # result instead of querying get_pending_interaction_question
                    # a second time for a value that cannot have changed.
                    question_message, question_interactions = (
                        waiting_question,
                        waiting_interactions,
                    )

                message = question_message or default_message
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
                cached_stream_events.append(status_event)

            detached_events = [
                _with_task_control_state_snapshot(
                    event,
                    task_id=task_id,
                    state=control_state,
                )
                for event in cached_stream_events
            ]
            # The replay is fully detached before cache serialization. If an
            # unexpected pending mutation prevents a clean release, skip this
            # optional cache write instead of holding the connection across
            # remote cache I/O.
            if release_db_connection_if_clean(db):
                cache_set(
                    cache_key,
                    {
                        "trace_scope": trace_scope,
                        "updated_at": task_updated_at,
                        "max_trace_event_id": int(max_trace_event_id),
                        "max_chat_message_id": int(max_chat_message_id),
                        "events": detached_events,
                    },
                    ttl_seconds=task_cache_ttl_seconds(),
                )
            return _HistoricalStreamSnapshot(events=tuple(detached_events))

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
        logger.error(f"Data format error building historical data stream: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error building historical data stream: {e}")
        raise


async def send_historical_data_as_stream(
    websocket: WebSocket,
    task_id: int,
    user: Union[User, WebSocketPrincipal],
) -> None:
    """Send one detached historical snapshot in stream-event order."""

    try:
        snapshot = await run_db_io_cancellation_safe(
            lambda: _load_historical_stream_snapshot_sync(
                task_id,
                actor_user_id=int(user.id),
                actor_is_admin=bool(user.is_admin),
            )
        )
        if snapshot is None:
            return
        for event in snapshot.events:
            await manager.send_personal_message(deepcopy(event), websocket)
    except (ValueError, KeyError, TypeError) as e:
        logger.error(f"Data format error sending historical data stream: {e}")
        error_event = create_stream_event(
            "error",
            task_id,
            {
                "message": client_safe_error_message(
                    e,
                    fallback="Task history could not be loaded. Please try again.",
                ),
            },
        )
        await manager.send_personal_message(error_event, websocket)
        raise
    except (ConnectionError, WebSocketDisconnect) as e:
        logger.error(f"Connection error sending historical data stream: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error sending historical data stream: {e}")
        raise


async def handle_status_request(
    websocket: WebSocket,
    task_id: int,
    user: Union[User, WebSocketPrincipal],
) -> None:
    """Handle status request - send historical data as stream messages"""
    await send_historical_data_as_stream(websocket, task_id, user)


# Operation labels for the endpoint-level fault arms, keyed by message type.
# A closed map rather than interpolation: ``type`` is client-supplied, and
# ``operation`` is meant to be a bounded, aggregatable value -- it is also not
# sanitised the way rendered fields are (#1520), so a client must not be able to
# reach it.
#
# Only the message types whose handler lets a fault propagate are listed.
# ``execute_task`` and ``intervention`` end in ``except RuntimeError``
# with no re-raise, so a durable fault from either is swallowed there and can
# never reach the arms below; giving them a label would claim a reachability
# that does not exist, and the label would read as covered while never being
# emitted. Making them reachable means giving those two handlers a durable arm
# of their own, which is absorber work and belongs to #1515 -- at which point
# they get a label here. ``_SWALLOWED_DISPATCH_TYPES`` in the tests pins the
# omission against the handlers, so this cannot silently become wrong.
# ``chat`` is absent for a third reason, distinct from the two above: every
# fault arm of its handler that re-raises reports through
# ``log_durable_storage_fault`` first, and the logger marks the instance, so
# the call here is a no-op rather than a second record. A label for it would
# name a line that is never emitted.
# ``test_chat_is_unlabelled_only_because_its_arms_report_first`` pins the
# report-before-re-raise half against the handler's own arms.
_DISPATCH_OPERATIONS = {
    "status_request": "websocket status request",
    "pause_task": "websocket pause_task",
    "resume_task": "websocket resume_task",
}
_UNKNOWN_DISPATCH_OPERATION = "websocket unknown message type"


def _private_websocket_task_access_sync(
    *,
    task_id: int,
    actor_user_id: int,
    actor_is_admin: bool,
) -> Literal["authorized", "missing", "foreign"]:
    """Classify private task access before a socket joins its live audience."""

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        owner_id = db.query(Task.user_id).filter(Task.id == task_id).scalar()
        if owner_id is None:
            return "missing"
        if actor_is_admin or int(owner_id) == actor_user_id:
            return "authorized"
        return "foreign"


@ws_router.websocket("/ws/chat/{task_id}")
async def websocket_chat_endpoint(
    websocket: WebSocket,
    task_id: int,
    token: Optional[str] = Query(None, description="Authentication token"),
) -> None:
    """WebSocket unified endpoint - handle chat, execution status, and DAG intervention"""
    # Verify user identity
    try:
        user = await get_authenticated_user(websocket, token)
    except _WebSocketAuthenticationTerminated:
        return
    if not user:
        await websocket.close(code=4001, reason="Authentication required")
        return

    # Accept before closing an access denial so the fixed close reason survives
    # the WebSocket handshake.  Do not register until ownership has been
    # checked: task broadcasts are an owner-scoped private audience.
    await websocket.accept()
    try:
        access = await run_db_io_cancellation_safe(
            lambda: _private_websocket_task_access_sync(
                task_id=task_id,
                actor_user_id=int(user.id),
                actor_is_admin=bool(user.is_admin),
            )
        )
    except Exception as exc:
        await send_websocket_authentication_infrastructure_failure(websocket, exc)
        return
    if access == "foreign":
        await websocket.close(
            code=4003,
            reason=client_error_message(ClientErrorCode.TASK_UNAVAILABLE),
        )
        return
    if access == "authorized":
        manager.register_connection(websocket, task_id)
    # A missing id deliberately stays accepted but unregistered.  The legacy
    # first-chat recovery path creates a replacement task, then
    # ``move_connection`` registers this already-accepted socket on that id.

    # Which message the loop is currently applying, for the fault arms below:
    # they guard the whole dispatch, so a fixed label would report a resume or
    # an execute_task fault as a chat turn in the one line meant to name it.
    # Initialised here, not in the loop, because the initial status request runs
    # before the first message is ever parsed.
    dispatching = "websocket initial status request"

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

            # ``str()`` before the lookup, not for the ``None`` case -- a
            # missing type misses the map either way -- but because ``type``
            # is client-supplied and need not be hashable: ``{"type": []}``
            # would raise ``TypeError`` from ``dict.get`` itself. Every value
            # that does not name a handler lands on the bounded fallback.
            dispatching = _DISPATCH_OPERATIONS.get(
                str(message_data.get("type")), _UNKNOWN_DISPATCH_OPERATION
            )

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
        pass
    except DurableObjectIntegrityError:
        # Precedes the parent arm: permanent corruption, already recorded at
        # ERROR with both checksums where it is raised, so it must not also be
        # logged as a transient outage. Swallowed exactly as the parent arm
        # swallows -- the socket is going away either way.
        pass
    except DurableStorageOperationError as exc:
        # Must precede the RuntimeError arm below, which this subclasses. A
        # storage fault reaching here would otherwise be logged as "Connection
        # error in WebSocket" and swallowed -- mislabelled and cause-less on
        # the very path #1467 was filed about. Still swallowed, as before:
        # the socket is going away regardless and the client has already been
        # answered; only the diagnosis changes.
        log_durable_storage_fault(logger, dispatching, exc, task_id=task_id)
    except (ConnectionError, RuntimeError) as e:
        # Connection error
        logger.error(f"Connection error in WebSocket: {e}")
    except Exception as e:
        # Other errors, re-raise
        logger.error(f"Unexpected error in WebSocket: {e}")
        raise
    finally:
        manager.disconnect(websocket)


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
                # The action is client-supplied and reaches every connection on
                # the task, so it travels as a structured field only.
                "message": "Manual intervention processed",
                "action": intervention_data["action"],
                "intervention_id": intervention_data["step_id"],
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),  # Send UTC timestamp directly
            },
            task_id,
        )

    except (ValueError, KeyError, TypeError) as e:
        # Data validation error
        command_execution_service.log_client_facing_failure(
            e, "Data validation error in intervention: %s"
        )
        await manager.send_personal_message(
            {
                "type": "error",
                "message": client_safe_error_message(e),
            },
            websocket,
        )
    except RuntimeError as e:
        # RuntimeError is incidental server detail, never display prose. A
        # stable code lets current clients localize the fixed fallback while
        # old clients still receive safe English text.
        logger.error("Runtime error in intervention: %s", e, exc_info=True)
        await manager.send_personal_message(
            {
                "type": "error",
                "error_code": ClientErrorCode.MESSAGE_PROCESSING_FAILED.value,
                "message": client_error_message(
                    ClientErrorCode.MESSAGE_PROCESSING_FAILED
                ),
            },
            websocket,
        )
    except Exception as e:
        # Re-raised, but the callers do not own the stack: the chat endpoint
        # logs without exc_info and the public endpoints swallow entirely.
        logger.error("Unexpected error in intervention: %s", e, exc_info=True)
        raise


async def handle_pause_task(
    websocket: WebSocket, task_id: int, message_data: dict
) -> None:
    """Persist a pause request; the lease owner applies it in command order."""

    try:
        enqueued = await _enqueue_websocket_task_command(
            websocket=websocket,
            task_id=task_id,
            message_data=message_data,
            kind=TaskCommandKind.PAUSE,
            command_id=command_execution_service._client_message_id(
                message_data.get("command_id")
            ),
        )
    except (
        MCPBuiltinOAuthActorPolicyRequiredError,
        PermissionError,
        ValueError,
    ) as exc:
        command_execution_service.log_client_facing_failure(
            exc, "Pause command rejected for task %s: %s", task_id
        )
        error_code = (
            exc.error_code
            if isinstance(exc, ClientVisibleError)
            else ClientErrorCode.MESSAGE_PROCESSING_FAILED
        )
        await manager.send_personal_message(
            {
                "type": "error",
                "message": client_error_message(error_code),
                "error_code": error_code.value,
            },
            websocket,
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
    if enqueued.created and not get_shared_task_execution_enabled():
        # Only the creating ingress owns the origin; a payload-matching
        # duplicate must never bind (see handle_chat_message). Registered
        # before dispatch so local execution cannot outrun the binding.
        _command_origins.register(enqueued.client_command_id, websocket, task_id)
    await dispatch_task_command_promptly(
        command_execution_service.execute_durable_task_command,
        command_db_id=enqueued.command_id,
    )


async def handle_resume_task(
    websocket: WebSocket, task_id: int, message_data: dict
) -> None:
    """Persist a resume request; a worker applies it in command order."""

    try:
        enqueued = await _enqueue_websocket_task_command(
            websocket=websocket,
            task_id=task_id,
            message_data=message_data,
            kind=TaskCommandKind.RESUME,
            command_id=command_execution_service._client_message_id(
                message_data.get("command_id")
            ),
        )
    except (
        MCPBuiltinOAuthActorPolicyRequiredError,
        PermissionError,
        ValueError,
    ) as exc:
        command_execution_service.log_client_facing_failure(
            exc, "Resume command rejected for task %s: %s", task_id
        )
        error_code = (
            exc.error_code
            if isinstance(exc, ClientVisibleError)
            else ClientErrorCode.MESSAGE_PROCESSING_FAILED
        )
        await manager.send_personal_message(
            {
                "type": "error",
                "message": client_error_message(error_code),
                "error_code": error_code.value,
            },
            websocket,
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
    if enqueued.created and not get_shared_task_execution_enabled():
        # Only the creating ingress owns the origin; a payload-matching
        # duplicate must never bind (see handle_chat_message). Registered
        # before dispatch so local execution cannot outrun the binding.
        _command_origins.register(enqueued.client_command_id, websocket, task_id)
    await dispatch_task_command_promptly(
        command_execution_service.execute_durable_task_command,
        command_db_id=enqueued.command_id,
    )


@ws_router.websocket("/ws/build/chat")
async def websocket_builder_chat_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None, description="Authentication token"),
) -> None:
    """WebSocket endpoint for AI Agent Builder Assistant chat."""
    try:
        user = await get_authenticated_user(websocket, token)
    except _WebSocketAuthenticationTerminated:
        return
    if not user:
        await websocket.close(code=4001, reason="Authentication required")
        return

    await websocket.accept()
    logger.info(f"Builder chat WebSocket connection established for user {user.id}")
    active_chat_task: asyncio.Task[None] | None = None

    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"📨 Received builder chat message: {data[:200]}")

            message_data = json.loads(data)

            # Run in background to not block receiving
            if active_chat_task is not None:
                await cancel_and_drain_async_task(active_chat_task)

            active_chat_task = asyncio.create_task(
                handle_builder_chat(websocket, message_data, user)
            )
            websocket.state.chat_task = active_chat_task

    except WebSocketDisconnect:
        logger.info(f"Builder chat WebSocket disconnected for user {user.id}")
    except (ConnectionError, RuntimeError) as e:
        logger.error(f"Connection error in builder chat WebSocket: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in builder chat WebSocket: {e}")
    finally:
        if active_chat_task is not None:
            await cancel_and_drain_async_task(active_chat_task)
        websocket.state.chat_task = None


async def handle_builder_chat(
    websocket: WebSocket,
    message_data: dict,
    user: Union[User, WebSocketPrincipal],
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
    from ..services.agent_prompt import apply_user_voice, voice_from_runtime_user
    from ..services.builder_chat_runtime import load_builder_chat_runtime_inputs

    user_id = int(user.id)
    is_admin = bool(user.is_admin)

    # Generate task_id for builder chat (reuse if exists)
    if not hasattr(websocket.state, "builder_task_id"):
        websocket.state.builder_task_id = f"builder_chat_{uuid.uuid4().hex[:8]}"
    builder_task_id = websocket.state.builder_task_id

    builder_tracer = create_ephemeral_tracer(
        task_id=builder_task_id,
        websocket_handler=SharedWebSocketTracer(
            websocket, builder_task_id, is_preview=False
        ),
        # The tracer only consumes ``user.id``. The cast keeps compatibility
        # with its HTTP-oriented annotation while WebSockets carry a frozen
        # principal instead of a detached ORM row.
        user=cast(User, user),
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

        # Resolve all database-backed inputs in one worker-owned short Session.
        files = message_data.get("files", [])
        requested_file_ids: list[str] = []
        if isinstance(files, list):
            for file_info in files:
                if not isinstance(file_info, dict):
                    continue
                file_id = file_info.get("file_id")
                if file_id:
                    requested_file_ids.append(str(file_id))

        runtime_inputs = await load_builder_chat_runtime_inputs(
            user_id=user_id,
            requested_file_ids=requested_file_ids,
            model_name=current_config.get("model"),
            compact_model_name=current_config.get("compact_model"),
        )
        if runtime_inputs.authorized_file_ids:
            user_message += (
                f"\n\n[Uploaded file_ids: {list(runtime_inputs.authorized_file_ids)}. "
                "Use file_id as the canonical file handle and do not guess storage paths. "
                "Please call `create_knowledge_base_from_file` with these file_ids immediately, "
                "then create or update the agent with the resulting collection_name.]"
            )

        skill_manager = create_skill_manager()
        agent_builder_skill = await skill_manager.get_skill("agent-builder")
        agent_builder_skill_context = (
            build_skill_context(agent_builder_skill) if agent_builder_skill else None
        )

        # Build system prompt with runtime state only. The behavioral workflow comes
        # from the forced agent-builder skill context below.
        system_prompt: Optional[
            str
        ] = f"""You are the runtime wrapper for the Xagent builder chat.
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
        # apply_user_voice's own scoping caveat covers create_agent/
        # update_agent's persisted name/description/instructions here -
        # see apply_output_voice's docstring.
        system_prompt = apply_user_voice(system_prompt, voice_from_runtime_user(user))

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
                        event_id=payload.get("event_id"),
                    )
                )
            )

        llm = runtime_inputs.llm
        compact_llm = runtime_inputs.compact_llm

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
            session_factory = get_session_local()
            create_agent_tool = CreateAgentTool(
                session_factory=session_factory,
                user_id=user_id,
                task_id=builder_task_id,
                workspace_base_dir=str(get_uploads_dir() / "builder_chat"),
            )
            update_agent_tool = UpdateAgentTool(
                session_factory=session_factory,
                user_id=user_id,
                task_id=builder_task_id,
                workspace_base_dir=str(get_uploads_dir() / "builder_chat"),
            )
            list_skills_tool = ListAvailableSkillsTool()
            list_tool_categories_tool = ListToolCategoriesTool()
            list_kbs_tool = ListKnowledgeBasesTool(user_id=user_id, is_admin=is_admin)
            create_kb_url_tool = CreateKnowledgeBaseFromUrlTool(
                user_id=user_id, is_admin=is_admin
            )
            create_kb_file_tool = CreateKnowledgeBaseFromFileTool(
                user_id=user_id, is_admin=is_admin
            )

            # Build allowed external directories
            allowed_external_dirs = []
            if user_id:
                from ...core.workspace import scoped_user_root

                user_upload_dir = scoped_user_root(get_uploads_dir(), user_id)
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
            with UserContext(user_id):
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
        logger.error("Error handling builder chat: %s", e, exc_info=True)
        error_metadata = {}
        if isinstance(e, AutoModelUnavailableError):
            error_metadata["error_code"] = ClientErrorCode.AUTO_MODEL_UNAVAILABLE.value
        await websocket.send_text(
            json.dumps(
                {
                    **error_metadata,
                    "type": "error",
                    "message": client_safe_error_message(e),
                }
            )
        )


@ws_router.websocket("/ws/build/preview")
async def websocket_build_preview_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None, description="Authentication token"),
) -> None:
    """WebSocket endpoint for build page agent preview using normal task execution."""
    # Verify user identity
    try:
        user = await get_authenticated_user(websocket, token)
    except _WebSocketAuthenticationTerminated:
        return
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
                manager.disconnect(websocket)
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
                            # Not echoed back: matches the main loop at the
                            # "Unknown message type" site above.
                            "message": "Unknown message type",
                        }
                    )
                )

    except WebSocketDisconnect:
        logger.info(f"Build preview WebSocket disconnected for user {user.id}")
    except (ConnectionError, RuntimeError) as e:
        logger.error(f"Connection error in build preview WebSocket: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in build preview WebSocket: {e}")
    finally:
        manager.disconnect(websocket)


async def handle_build_preview_execution(
    websocket: WebSocket,
    message_data: dict,
    user: Union[User, WebSocketPrincipal],
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
            # create_task's implementation only consumes ``user.id``; keep its
            # HTTP dependency annotation local instead of widening the
            # WebSocket principal back into an ORM object.
            task_response = await create_task(
                task_request,
                db=preview_db,
                user=cast(User, user),
            )
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
