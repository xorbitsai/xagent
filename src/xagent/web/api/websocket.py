"""WebSocket real-time communication handler"""

import asyncio
import enum
import json
import logging
import re
import shutil
import time
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Union,
    cast,
    overload,
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
from sqlalchemy import case, func, or_, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ...config import (
    get_default_task_execution_mode,
    get_external_upload_dirs,
    get_uploads_dir,
)
from ...core.agent.checkpoint import (
    CHECKPOINT_EVENT_TYPE,
    CheckpointAccessRefusedError,
    CheckpointReadError,
    CheckpointUnavailableError,
)
from ...core.agent.runner import UserMessageInjectionOutcome
from ...core.agent.trace import TraceEvent, TraceHandler
from ...core.execution_scope import (
    EXECUTION_SCOPE_NOT_PROVIDED,
    ExecutionScope,
    ExecutionScopeContext,
    ExecutionScopeNotProvided,
    resolve_execution_scope,
    resolve_execution_scope_off_turn,
)
from ...core.file_ref import FILE_REF_MODEL_INSTRUCTIONS, build_file_ref
from ..models.chat_message import TaskChatMessage
from ..models.database import (
    get_db,
    get_session_local,
    release_db_connection_if_clean,
)
from ..models.task import Task, TaskStatus
from ..models.uploaded_file import UploadedFile
from ..models.user import User

if TYPE_CHECKING:
    from ..services.task_setup_snapshot import TaskSetupSnapshot
    from ..services.task_orchestrator import _ClaimedTurn, TaskTurnPayload

from ...core.file_storage.keys import (
    build_task_output_storage_key,
    build_upload_storage_key,
)
from ..services.assistant_history_safety import (
    ASSISTANT_RESPONSE_MESSAGE_TYPE,
    TASK_FAILURE_MESSAGE_TYPE,
    assistant_history_has_safe_ancillary_payload,
    assistant_history_values_for_persistence,
    client_safe_assistant_history_content,
    safe_str,
)
from ..services.chat_history_service import (
    DELIVERY_COMPLETED,
    DELIVERY_DISPATCHED,
    DELIVERY_FAILED,
    DELIVERY_PENDING,
    UserMessageDeliveryClaim,
    claim_user_message_delivery_no_commit,
    inspect_user_message_delivery,
    mark_user_message_delivery_sync,
)
from ..services.client_error_messages import (
    CLIENT_SAFE_GUIDANCE_IN_PROGRESS,
    CLIENT_SAFE_TASK_FAILURE,
    CLIENT_SAFE_VALIDATION_ERROR,
    ClientErrorCode,
    client_error_message,
)
from ..services.db_runtime import (
    await_task_settlement,
    cancel_and_drain_async_task,
    drain_async_task_cancellation_safe,
    is_database_pool_timeout,
    propagate_deferred_cancellation,
    run_db_io_cancellation_safe,
)
from ..services.external_task_cancel import (
    EXTERNAL_COMMAND_SCOPE,
    cancel_external_task_unserialized,
    external_cancel_exhausted_message,
)
from ..services.external_task_input import execute_external_task_input_command
from ..services.file_reference_output_service import (
    load_assistant_file_reference_records,
    reconcile_assistant_file_references,
)
from ..services.file_turn import (
    append_uploaded_files_context as _append_uploaded_files_context_to_message,
)
from ..services.file_turn import (
    bind_turn_files,
    bind_turn_files_no_commit,
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
    DurableObjectIntegrityError,
    DurableStorageOperationError,
    log_durable_storage_fault,
)
from ..services.mcp_runtime import (
    MCPBuiltinOAuthActorPolicy,
    MCPBuiltinOAuthActorPolicyRequiredError,
)
from ..services.task_command_terminal_events import (
    TerminalTaskEventDraft,
    TerminalTaskEventMessageCode,
    bind_terminal_event_draft,
    is_external_cancel_command,
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
    TaskCommandTaskMissing,
    dispatch_task_command_promptly,
    enqueue_task_command,
    task_has_live_foreign_runner,
    task_has_live_runner,
)
from ..services.task_execution_controller import (
    StaleTaskRunError,
    StaleTaskStateVersionError,
    TaskControlSnapshot,
    TaskControlState,
    apply_task_control_transition,
    control_state_for_status,
    task_control_snapshot,
    task_execution_controller,
)
from ..services.task_interaction_close import (
    active_interaction_id_sync,
    clear_interaction_marker_if_unpaired,
    close_legacy_resume_interaction_sync,
)
from ..services.task_interaction_read import get_pending_interaction_question
from ..services.task_lease_service import (
    TaskLease,
    TaskLeaseHeartbeatOutcome,
    TaskLeaseLostError,
    acquire_task_lease_cancellation_safe,
    acquire_task_lease_no_commit,
    bind_task_lease_context,
    release_task_lease_no_commit,
    run_task_lease_heartbeat,
    run_while_task_lease_owned,
    stop_task_lease_heartbeat,
)
from ..services.task_runtime import (
    SELECTED_FILE_IDS_AGENT_CONFIG_KEY,
    mcp_runtime_authorization_policy_required,
    task_extension_bindings_from_agent_config,
)
from ..services.uploaded_file_store import (
    StagedUploadedFile,
    SupersededObjectCleanupClaim,
    UploadedFileStore,
    UploadedFileVersionConflict,
    UploadedFileVersionSnapshot,
    cleanup_superseded_uploaded_file_objects,
    compensate_staged_uploaded_files,
    snapshot_uploaded_file_version,
    stage_uploaded_file_from_local_path,
)
from ..services.workforce_runtime import (
    sync_workforce_run_status,
)
from ..tracing import create_ephemeral_tracer
from ..user_isolated_memory import UserContext
from ..utils.db_timezone import safe_timestamp_to_unix
from ..utils.json_payload_sanitizer import sanitize_json_payload
from .public_trace_events import (
    is_audit_only_trace_data,
    normalize_public_trace_event,
    public_task_trace_filter,
)
from .websocket_auth import (
    WebSocketPrincipal,
    _WebSocketAuthenticationTerminated,
    get_authenticated_user,
    send_websocket_authentication_infrastructure_failure,
)

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


def _waiting_or_paused_event_fields(status: TaskStatus) -> tuple[str, str]:
    """Event type and default message for a task settled at WAITING_FOR_USER
    or PAUSED. Shared by the live-lease restore broadcast and the
    historical-replay status reassertion so both present identical labels
    for the same status."""

    if status == TaskStatus.WAITING_FOR_USER:
        return "task_waiting_for_user", "Task waiting for user response"
    return "task_paused", "Task paused"


# Non-transient turn rejections must not fall back to retryable busy guidance.
_TURN_REJECTION_CODES = {
    "actor_task_reuse_unsupported": (ClientErrorCode.MESSAGE_CONTINUATION_UNSUPPORTED),
    "workforce_archived": ClientErrorCode.WORKFORCE_ARCHIVED,
    "workforce_config_changed": ClientErrorCode.WORKFORCE_UNAVAILABLE,
    "workforce_run_not_found": ClientErrorCode.WORKFORCE_UNAVAILABLE,
    "workforce_run_not_active": ClientErrorCode.WORKFORCE_UNAVAILABLE,
}


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
    error_code: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": event_type,
        "message": message,
    }
    if error_code is not None:
        payload["error_code"] = error_code
    task_payload = _task_status_payload(db, task_id)
    if task_payload is not None:
        payload["task"] = task_payload
    return payload


def create_terminal_task_error_event(
    task_id: int,
    message: str,
) -> dict[str, Any]:
    """Shape an error event after the exact lease owner commits FAILED."""

    return {
        "type": "task_error",
        "message": message,
        "task_id": task_id,
        "task": {
            "id": task_id,
            "status": TaskStatus.FAILED.value,
        },
        "error": message,
        "timestamp": datetime.now(timezone.utc).timestamp(),
    }


def _client_message_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if COMMAND_ID_PATTERN.fullmatch(normalized) is None:
        return None
    return normalized


# Exception text can carry file paths, SQL fragments, provider payloads and
# other internals. It reaches anonymous widget/share visitors through both the
# error bubble and the message_rejected ack, so an *incidental* validation
# failure only ever surfaces as this fixed string; the detail stays in the log.
#
# RuntimeError text follows the same rule for every audience. The task-wide
# broadcast, rejection ack, and personal error bubble expose stable safe
# fields; provider responses and internal paths remain operator-only logs.
class ClientVisibleError(Exception):
    """Marker: this exception's text was written for the end user.

    Raise a subclass - never a bare builtin - when the message itself is the
    actionable answer ("authentication required", "access denied"). Everything
    else reaching a client-facing handler is treated as incidental and
    redacted, so forgetting the marker fails closed.
    """

    def __init__(
        self,
        *args: object,
        error_code: ClientErrorCode = ClientErrorCode.MESSAGE_PROCESSING_FAILED,
    ) -> None:
        if type(self) is ClientVisibleError:
            raise TypeError("ClientVisibleError must be subclassed")
        self.error_code = error_code
        super().__init__(*args)


class ClientVisibleValidationError(ClientVisibleError, ValueError):
    """A validation failure whose text is safe to show the sender."""


class ClientVisiblePermissionError(ClientVisibleError, PermissionError):
    """An authorization refusal whose text is safe to show the sender."""


class ClientVisibleTaskCommandDeferred(ClientVisibleError, TaskCommandDeferred):
    """A deferral whose wording this module wrote for the sender.

    Terminal deferral broadcasts go through the chokepoint like everything
    else, so without the marker "waiting for the active task lease owner"
    would reach the client as the generic string and become
    indistinguishable from an outright failure.
    """


def client_safe_error_message(
    error: BaseException,
    *,
    fallback: str = CLIENT_SAFE_VALIDATION_ERROR,
) -> str:
    """The only way an exception may become text a chat client can see.

    ``tests/web/api/test_websocket_client_safe_errors.py`` enforces this for
    the shapes it recognizes: delivery producers and known error-event payloads
    handed to ``send_personal_message``, ``broadcast_to_task`` or ``send_text``.

    The sweep recognizes the client egress shapes used by this module,
    including terminal task helpers, dict-spread overrides, both ``message``
    and ``error`` fields, and the deferred-delivery wrapper. It is still a
    deliberately small static check rather than general data-flow analysis;
    for example, a payload ``type`` built from a variable remains outside its
    scope (#1547).

    Read a passing sweep as "the recognized egress shapes are clean", never
    as "arbitrary Python data flow cannot reach a client raw".
    """
    if not isinstance(error, ClientVisibleError):
        return fallback
    message = str(error)
    return message if message.strip() else fallback


def client_safe_task_command_failure(
    kind: TaskCommandKind,
    error: BaseException,
    *,
    scope: str | None = None,
    task_status: TaskStatus | None = None,
) -> str:
    """Terminal command failure: server-owned kind prefix + redacted detail.

    The frontend renders ``message`` verbatim for ``agent_error``, so dropping
    the prefix entirely removed user-visible context. The kind comes from our
    own enum, never from the exception, which is what makes the prefix safe.

    An external-scope cancel is the one command a task's audience issues
    without any account behind it, and its whole meaning is "stop this
    response". That audience gets neither the command identity nor the
    exception detail - and it gets a sentence about the turn rather than
    about the command, which is why the caller reads the task and hands the
    status in. Saying the response was interrupted when the task is still
    running would be false, and the visitor would keep waiting on a turn
    nobody stopped.
    """
    if is_external_cancel_command(kind=kind.value, scope=scope):
        return external_cancel_exhausted_message(task_status)
    return f"Task command {kind.value} failed: {client_safe_error_message(error)}"


def log_client_facing_failure(error: Exception, template: str, *args: object) -> None:
    """Record a failure whose text the client will not see in full.

    A ``ClientVisibleError`` is an answer written for the sender - an
    unauthenticated frame, a task that no longer exists - so it is routine
    and gets no traceback; otherwise any visitor could emit stack dumps on
    demand. Anything else is incidental, and once its text is redacted the
    traceback is the only record left.

    ``template`` ends in the ``%s`` that receives ``error``; ``args`` fill the
    placeholders before it.
    """
    rendered_message: str | None = None
    try:
        if str.endswith(template, "%s"):
            rendered_message = str.__str__(template % (*args, error))
    except Exception:
        pass
    if rendered_message is None:
        safe_template = _safe_log_argument(template)
        safe_args = tuple(_safe_log_argument(arg) for arg in args)
        safe_error = _safe_log_argument(error)
        logger.log(
            logging.WARNING if isinstance(error, ClientVisibleError) else logging.ERROR,
            "Malformed client-facing log template %r with args=%r; original error: %s",
            safe_template,
            safe_args,
            safe_error,
            exc_info=None if isinstance(error, ClientVisibleError) else True,
        )
        return
    if isinstance(error, ClientVisibleError):
        logger.warning(rendered_message)
    else:
        logger.error(rendered_message, exc_info=True)


def _safe_log_argument(value: object) -> object:
    """Snapshot malformed-log values without trusting hostile string methods."""
    # Exact types preserve builtin logging representations without admitting subclasses.
    if type(value) in (str, int, float, bytes):
        return value
    try:
        # The unbound call strips any surviving ``str``-subclass overrides.
        return str.__str__(str(value))
    except Exception as rendering_error:
        return f"<unprintable {type(value).__name__}: {type(rendering_error).__name__}>"


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
    if client_message_id is None:
        return
    if not accepted and rejection_outcome is None:
        raise ValueError("Rejected delivery requires an explicit rejection outcome")
    if accepted and rejection_outcome is not None:
        raise ValueError("Accepted delivery cannot include a rejection outcome")
    payload: dict[str, Any] = {
        "type": "message_accepted" if accepted else "message_rejected",
        "client_message_id": client_message_id,
        "turn_id": turn_id,
        "timestamp": datetime.now(timezone.utc).timestamp(),
    }
    if message:
        payload["message"] = message
    if error_code is not None:
        payload["error_code"] = error_code
    if retry_with_new_id:
        payload["retry_with_new_id"] = True
    if rejection_outcome is not None:
        payload["rejection_outcome"] = rejection_outcome
    await manager.send_personal_message(payload, websocket)


@overload
def _terminal_task_error_payload(
    task_id: int,
    message: str,
    *,
    event_type: str = "agent_error",
    expected_run_id: str | None = None,
    only_if_running: Literal[False] = False,
) -> dict[str, Any]: ...


@overload
def _terminal_task_error_payload(
    task_id: int,
    message: str,
    *,
    event_type: str = "agent_error",
    expected_run_id: str | None = None,
    only_if_running: Literal[True],
) -> dict[str, Any] | None: ...


def _terminal_task_error_payload(
    task_id: int,
    message: str,
    *,
    event_type: str = "agent_error",
    expected_run_id: str | None = None,
    only_if_running: bool = False,
) -> dict[str, Any] | None:
    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        failed_control_state = TaskControlState.FAILED.value
        current_version = func.coalesce(Task.state_version, 0)
        statement = (
            update(Task)
            .where(Task.id == task_id)
            # This legacy helper has no concrete TaskLease. It may only settle
            # an ownerless row; a RUNNING row with any owner belongs to the
            # lease-aware orchestrator and must be left untouched.
            .where(Task.runner_id.is_(None))
            .values(
                status=TaskStatus.FAILED,
                lease_expires_at=None,
                last_heartbeat_at=datetime.now(timezone.utc),
                control_state=failed_control_state,
                state_version=case(
                    (
                        or_(
                            Task.status != TaskStatus.FAILED,
                            Task.control_state != failed_control_state,
                        ),
                        current_version + 1,
                    ),
                    else_=current_version,
                ),
                error_message=message,
            )
        )
        if expected_run_id is not None:
            statement = statement.where(Task.run_id == expected_run_id)
        if only_if_running:
            statement = statement.where(Task.status == TaskStatus.RUNNING)

        result = db.execute(statement.execution_options(synchronize_session=False))
        if int(getattr(result, "rowcount", 0) or 0) != 1:
            db.rollback()
            current_payload = _task_error_payload(
                db,
                task_id,
                CLIENT_SAFE_TASK_FAILURE,
                event_type=event_type,
            )
            logger.info(
                "Ignoring unfenced terminal error for task %s run %s; "
                "current status is %s",
                task_id,
                expected_run_id,
                (current_payload.get("task") or {}).get("status"),
            )
            return None if only_if_running else current_payload

        task = db.query(Task).filter(Task.id == task_id).first()
        if task is not None:
            sync_workforce_run_status(db, task, TaskStatus.FAILED)
            # Persist the error as an assistant message so failures that
            # happen before agent execution starts (no trace events, e.g.
            # sandbox capacity rejection) survive a history reload instead
            # of degrading to a generic "Unknown error" bubble.
            task_user_id = getattr(task, "user_id", None)
            if task_user_id is not None:
                from ..services.chat_history_service import (
                    persist_assistant_message_no_commit,
                )

                try:
                    persist_assistant_message_no_commit(
                        db,
                        task_id=task_id,
                        user_id=int(task_user_id),
                        content=CLIENT_SAFE_TASK_FAILURE,
                        message_type=TASK_FAILURE_MESSAGE_TYPE,
                    )
                except Exception:
                    logger.warning(
                        "Failed to persist terminal error chat message",
                        exc_info=True,
                    )
            db.commit()
        return _task_error_payload(
            db,
            task_id,
            CLIENT_SAFE_TASK_FAILURE,
            event_type=event_type,
        )
    except Exception:
        db.rollback()
        logger.warning("Failed to persist terminal task error", exc_info=True)
        return {
            "type": event_type,
            "message": CLIENT_SAFE_TASK_FAILURE,
            "task": {
                "id": task_id,
                "status": TaskStatus.FAILED.value,
            },
        }
    finally:
        db.close()


def _read_task_error_payload_isolated(
    task_id: int,
    message: str,
    *,
    event_type: str = "agent_error",
    error_code: str | None = None,
) -> dict[str, Any]:
    """Read a task error payload in a short Session owned by this worker."""
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        try:
            return _task_error_payload(
                db,
                task_id,
                message,
                event_type=event_type,
                error_code=error_code,
            )
        except Exception:
            db.rollback()
            logger.warning("Failed to read terminal task error payload", exc_info=True)
            payload = {"type": event_type, "message": message}
            if error_code is not None:
                payload["error_code"] = error_code
            return payload


async def _read_task_error_payload_offloop(
    task_id: int,
    message: str,
    *,
    event_type: str = "agent_error",
    error_code: str | None = None,
) -> dict[str, Any]:
    """Keep a potentially blocked pool checkout off the asyncio event loop."""
    return await run_db_io_cancellation_safe(
        lambda: _read_task_error_payload_isolated(
            task_id,
            message,
            event_type=event_type,
            error_code=error_code,
        )
    )


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

    raw_file_ids = agent_config.get(SELECTED_FILE_IDS_AGENT_CONFIG_KEY)
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
            UploadedFile.storage_status != "compensating",
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
    *,
    event_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a stream event, preserving a producer-supplied event identity."""
    resolved_event_id = (
        event_id if isinstance(event_id, str) and event_id else str(uuid.uuid4())
    )
    return {
        "type": "trace_event",
        "event_id": resolved_event_id,
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
        # This function builds its own TraceEvent row instead of going
        # through stage_trace_event_row (see that module's "known bypass"
        # note), so it must sanitize for itself: PostgreSQL's jsonb rejects
        # NUL and unpaired-surrogate code points at INSERT (#1248).
        data = sanitize_json_payload(data)
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
            event_id=payload.get("event_id"),
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


@dataclass(frozen=True)
class _OutputFileRecordSnapshot:
    """Detached durable metadata used to plan one output registration."""

    version: UploadedFileVersionSnapshot
    file_id: str
    filename: str
    storage_key: str | None
    mime_type: str | None
    file_size: int
    workspace_relative_path: str | None
    workspace_category: str | None


@dataclass(frozen=True)
class _TaskOutputStageRequest:
    """One validated local output whose bytes still need durable staging."""

    item_index: int
    resolved_path: Path
    raw_paths: tuple[str, ...]
    item_file_id: str
    filename: str
    normalized_relative_path: str
    workspace_relative_path: str
    workspace_category: str
    existing: _OutputFileRecordSnapshot | None


@dataclass(frozen=True)
class _ResolvedTaskOutputInput:
    """Filesystem-resolved input that is safe to inspect with a short Session."""

    item_index: int
    item_file_id: str
    item_filename: str
    item_relative_path: str
    raw_paths: tuple[str, ...]
    resolved_info: tuple[Path, str] | None


@dataclass(frozen=True)
class _PreparedTaskOutputMutation:
    """One already-durable object awaiting the fenced metadata transaction."""

    staged: StagedUploadedFile
    expected: UploadedFileVersionSnapshot | None


@dataclass(frozen=True)
class _PreparedTaskFileOutputs:
    """Detached result of the no-Session durable-output phase."""

    normalized_outputs: tuple[dict[str, Any], ...]
    path_to_file_id: tuple[tuple[str, str], ...]
    mutations: tuple[_PreparedTaskOutputMutation, ...]

    @property
    def staged_files(self) -> tuple[StagedUploadedFile, ...]:
        return tuple(mutation.staged for mutation in self.mutations)


def _snapshot_output_file(record: UploadedFile) -> _OutputFileRecordSnapshot:
    return _OutputFileRecordSnapshot(
        version=snapshot_uploaded_file_version(record),
        file_id=str(record.file_id),
        filename=str(record.filename),
        storage_key=(
            str(record.storage_key) if record.storage_key is not None else None
        ),
        mime_type=str(record.mime_type) if record.mime_type is not None else None,
        file_size=int(record.file_size or 0),
        workspace_relative_path=(
            str(record.workspace_relative_path)
            if record.workspace_relative_path is not None
            else None
        ),
        workspace_category=(
            str(record.workspace_category)
            if record.workspace_category is not None
            else None
        ),
    )


def _prepared_output_ref(
    *,
    file_id: str,
    filename: str,
    mime_type: str | None,
    file_size: int,
) -> dict[str, Any]:
    return build_file_ref(
        file_id=file_id,
        filename=filename,
        mime_type=mime_type,
        size=file_size,
    )


def _prepare_task_file_outputs_isolated(
    *,
    task_id: int,
    task_user_id: int | None,
    file_outputs: Any,
    resolved_scope_segments: tuple[str, ...],
) -> _PreparedTaskFileOutputs:
    """Stage task output bytes without retaining a Session or task-row lock.

    The first phase only snapshots existing metadata.  The Session is closed
    before checksum/object-storage work begins.  A later exact-run transaction
    applies these detached mutations together with the terminal task state.
    """

    if isinstance(file_outputs, str):
        file_outputs = [file_outputs] if file_outputs.strip() else []
    if not isinstance(file_outputs, list) or not file_outputs:
        return _PreparedTaskFileOutputs((), (), ())

    SessionLocal = get_session_local()
    resolved_task_user_id = task_user_id
    if resolved_task_user_id is None:
        with SessionLocal() as db:
            resolved_task_user_id = (
                db.query(Task.user_id).filter(Task.id == task_id).scalar()
            )
        if resolved_task_user_id is None:
            return _PreparedTaskFileOutputs((), (), ())
    owner_user_id = int(resolved_task_user_id)

    # Parse and resolve every candidate before opening the metadata Session.
    # Local files may be backed by slow network mounts; even existence checks
    # must not pin a database connection.
    resolved_inputs: list[_ResolvedTaskOutputInput] = []
    for item_index, item in enumerate(file_outputs):
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
                    raw_paths.append(value)
                    if key == "relative_path":
                        item_relative_path = value
        else:
            continue

        resolved_info: tuple[Path, str] | None = None
        for raw_path in raw_paths:
            candidate = _resolve_output_storage_path(raw_path)
            if candidate is not None:
                resolved_info = (Path(candidate[0]), str(candidate[1]))
                break
        resolved_inputs.append(
            _ResolvedTaskOutputInput(
                item_index=item_index,
                item_file_id=item_file_id,
                item_filename=item_filename,
                item_relative_path=item_relative_path,
                raw_paths=tuple(raw_paths),
                resolved_info=resolved_info,
            )
        )

    stage_requests: list[_TaskOutputStageRequest] = []
    immediate_outputs: list[
        tuple[int, dict[str, Any], tuple[str, ...], str | None]
    ] = []
    with SessionLocal() as db:
        for resolved_input in resolved_inputs:
            if resolved_input.resolved_info is None:
                if not resolved_input.item_file_id:
                    continue
                record = (
                    db.query(UploadedFile)
                    .filter(
                        UploadedFile.file_id == resolved_input.item_file_id,
                        UploadedFile.user_id == owner_user_id,
                        or_(
                            UploadedFile.task_id == task_id,
                            UploadedFile.task_id.is_(None),
                        ),
                        UploadedFile.storage_status != "compensating",
                    )
                    .first()
                )
                if record is None:
                    logger.warning(
                        "Skipping file output outside task/user scope: %s",
                        resolved_input.item_file_id,
                    )
                    continue
                snapshot = _snapshot_output_file(record)
                immediate_outputs.append(
                    (
                        resolved_input.item_index,
                        _prepared_output_ref(
                            file_id=snapshot.file_id,
                            filename=(
                                resolved_input.item_filename or snapshot.filename
                            ),
                            mime_type=snapshot.mime_type,
                            file_size=snapshot.file_size,
                        ),
                        resolved_input.raw_paths,
                        snapshot.workspace_relative_path,
                    )
                )
                continue

            resolved_path, relative_path = resolved_input.resolved_info
            normalized_relative_path = relative_path.lstrip("/")
            record = (
                db.query(UploadedFile)
                .filter(
                    UploadedFile.storage_path == str(resolved_path),
                    UploadedFile.storage_status != "compensating",
                )
                .first()
            )
            if record is not None and not _uploaded_file_record_in_task_scope(
                record,
                task_id,
                owner_user_id,
            ):
                logger.warning(
                    "Skipping file output record outside task/user scope: %s",
                    getattr(record, "file_id", str(resolved_path)),
                )
                continue

            if record is not None and not _output_path_in_current_task_scope(
                normalized_relative_path,
                task_id,
                owner_user_id,
            ):
                snapshot = _snapshot_output_file(record)
                if snapshot.workspace_category != "output":
                    logger.warning(
                        "Skipping registered file output outside output category: %s",
                        snapshot.file_id,
                    )
                    continue
                immediate_outputs.append(
                    (
                        resolved_input.item_index,
                        _prepared_output_ref(
                            file_id=snapshot.file_id,
                            filename=(
                                resolved_input.item_filename or snapshot.filename
                            ),
                            mime_type=snapshot.mime_type,
                            file_size=snapshot.file_size,
                        ),
                        resolved_input.raw_paths,
                        snapshot.workspace_relative_path,
                    )
                )
                continue

            if not _output_path_in_current_task_scope(
                normalized_relative_path,
                task_id,
                owner_user_id,
            ):
                logger.warning(
                    "Skipping file output outside current task output scope: %s",
                    relative_path,
                )
                continue

            workspace_relative_path = _normalize_workspace_relative_path(
                resolved_input.item_relative_path or normalized_relative_path
            )
            workspace_category = _workspace_category_from_relative_path(
                workspace_relative_path
            )
            if record is None and resolved_input.item_file_id:
                record = (
                    db.query(UploadedFile)
                    .filter(
                        UploadedFile.file_id == resolved_input.item_file_id,
                        UploadedFile.user_id == owner_user_id,
                        or_(
                            UploadedFile.task_id == task_id,
                            UploadedFile.task_id.is_(None),
                        ),
                        UploadedFile.storage_status != "compensating",
                    )
                    .first()
                )
            existing = _snapshot_output_file(record) if record is not None else None
            stage_requests.append(
                _TaskOutputStageRequest(
                    item_index=resolved_input.item_index,
                    resolved_path=resolved_path,
                    raw_paths=resolved_input.raw_paths,
                    item_file_id=resolved_input.item_file_id,
                    filename=(resolved_input.item_filename or resolved_path.name),
                    normalized_relative_path=normalized_relative_path,
                    workspace_relative_path=workspace_relative_path,
                    workspace_category=workspace_category,
                    existing=existing,
                )
            )

    normalized_by_index: dict[int, dict[str, Any]] = {
        index: output for index, output, _raw_paths, _relative in immediate_outputs
    }
    path_to_file_id: dict[str, str] = {}
    for (
        index,
        output,
        immediate_raw_paths,
        immediate_relative_path,
    ) in immediate_outputs:
        del index
        file_id = str(output["file_id"])
        for raw_path in immediate_raw_paths:
            stripped = raw_path.strip()
            if stripped:
                _set_file_link_alias(path_to_file_id, stripped, file_id)
                _set_file_link_alias(path_to_file_id, stripped.lstrip("/"), file_id)
        if immediate_relative_path:
            _add_file_link_aliases(
                path_to_file_id,
                immediate_relative_path,
                file_id,
            )

    mutations: list[_PreparedTaskOutputMutation] = []
    staged_by_path: dict[str, StagedUploadedFile] = {}
    try:
        for request in stage_requests:
            path_key = str(request.resolved_path)
            staged = staged_by_path.get(path_key)
            if staged is None:
                existing_file_id = (
                    request.existing.file_id if request.existing is not None else ""
                )
                file_id = (
                    existing_file_id
                    or request.item_file_id
                    or _build_output_file_id(request.workspace_relative_path)
                )
                # Every staged output gets an immutable generation key,
                # including first insert. Competing preparations can therefore
                # compensate only their own object and committed cleanup never
                # needs to reuse a superseded key.
                key_relative_path = (
                    f"_versions/{uuid.uuid4().hex}/{request.workspace_relative_path}"
                )
                staged = stage_uploaded_file_from_local_path(
                    local_path=request.resolved_path,
                    user_id=int(resolved_task_user_id),
                    task_id=task_id,
                    file_id=file_id,
                    filename=request.filename,
                    mime_type=None,
                    storage_key=build_task_output_storage_key(
                        int(resolved_task_user_id),
                        task_id,
                        file_id,
                        key_relative_path,
                        scope_segments=resolved_scope_segments,
                    ),
                    workspace_relative_path=request.workspace_relative_path,
                    workspace_category=request.workspace_category,
                    execution_scope=ExecutionScope(
                        workspace_segments=resolved_scope_segments,
                        isolate_external_dirs=bool(resolved_scope_segments),
                    ),
                )
                staged_by_path[path_key] = staged
                mutations.append(
                    _PreparedTaskOutputMutation(
                        staged=staged,
                        expected=(
                            request.existing.version
                            if request.existing is not None
                            else None
                        ),
                    )
                )

            normalized_by_index[request.item_index] = _prepared_output_ref(
                file_id=staged.file_id,
                filename=request.filename or staged.filename,
                mime_type=staged.mime_type,
                file_size=staged.file_size,
            )
            if request.item_file_id:
                path_to_file_id[request.item_file_id] = staged.file_id
            for raw_path in request.raw_paths:
                stripped = raw_path.strip()
                if stripped:
                    _set_file_link_alias(path_to_file_id, stripped, staged.file_id)
                    _set_file_link_alias(
                        path_to_file_id,
                        stripped.lstrip("/"),
                        staged.file_id,
                    )
            _set_file_link_alias(
                path_to_file_id,
                str(request.resolved_path),
                staged.file_id,
            )
            _add_file_link_aliases(
                path_to_file_id,
                request.normalized_relative_path,
                staged.file_id,
            )
            if request.workspace_relative_path != request.normalized_relative_path:
                _add_file_link_aliases(
                    path_to_file_id,
                    request.workspace_relative_path,
                    staged.file_id,
                )
    except Exception:
        compensate_staged_uploaded_files(tuple(staged_by_path.values()))
        raise

    return _PreparedTaskFileOutputs(
        normalized_outputs=tuple(
            normalized_by_index[index] for index in sorted(normalized_by_index)
        ),
        path_to_file_id=tuple(path_to_file_id.items()),
        mutations=tuple(mutations),
    )


def _apply_prepared_task_file_outputs(
    db: Session,
    prepared: _PreparedTaskFileOutputs,
) -> tuple[
    list[dict[str, Any]],
    dict[str, str],
    tuple[SupersededObjectCleanupClaim, ...],
]:
    """Apply metadata only; the caller owns the exact-run transaction."""

    store = UploadedFileStore(db)
    cleanup_claims: list[SupersededObjectCleanupClaim] = []
    for mutation in prepared.mutations:
        applied = store.upsert_already_durable(
            mutation.staged,
            expected=mutation.expected,
        )
        if applied.superseded_cleanup_claim is not None:
            cleanup_claims.append(applied.superseded_cleanup_claim)
    return (
        [deepcopy(output) for output in prepared.normalized_outputs],
        dict(prepared.path_to_file_id),
        tuple(cleanup_claims),
    )


def _settle_prepared_task_file_outputs(
    prepared: _PreparedTaskFileOutputs,
    *,
    metadata_committed: bool,
    cleanup_claims: tuple[SupersededObjectCleanupClaim, ...] = (),
) -> None:
    """Complete the object-storage side of the two-phase registration."""

    if metadata_committed:
        try:
            failed_cleanup_claims = cleanup_superseded_uploaded_file_objects(
                cleanup_claims
            )
        except Exception:
            # The metadata transaction is already committed. Object cleanup is
            # post-commit garbage collection and must never reclassify that
            # durable success as a failed task execution.
            logger.exception(
                "Failed to clean up superseded task output objects after commit"
            )
            return
        if failed_cleanup_claims:
            logger.warning(
                "Retained %s superseded task output object(s) because reference, "
                "backend, or deletion state was unknown",
                len(failed_cleanup_claims),
            )
        return
    try:
        failed_staged_files = compensate_staged_uploaded_files(prepared.staged_files)
    except Exception:
        # Settlement runs from finalizers' ``finally`` blocks. A best-effort
        # compensation failure must not replace the original transaction error
        # that caused rollback.
        logger.exception("Failed to compensate staged task output objects")
        return
    if failed_staged_files:
        logger.warning(
            "Failed to compensate %s staged task output object(s)",
            len(failed_staged_files),
        )


async def _prepare_task_file_outputs_cancellation_safe(
    *,
    task_id: int,
    task_user_id: int | None,
    file_outputs: Any,
    resolved_scope_segments: tuple[str, ...],
) -> _PreparedTaskFileOutputs:
    """Drain staging and compensate its late result before propagating cancel."""

    worker = asyncio.create_task(
        asyncio.to_thread(
            _prepare_task_file_outputs_isolated,
            task_id=task_id,
            task_user_id=task_user_id,
            file_outputs=file_outputs,
            resolved_scope_segments=resolved_scope_segments,
        )
    )
    prepared, cancellation = await await_task_settlement(worker)
    if cancellation is None:
        return prepared
    try:
        await run_db_io_cancellation_safe(
            lambda: _settle_prepared_task_file_outputs(
                prepared,
                metadata_committed=False,
            )
        )
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception(
            "Failed to compensate task %s outputs after cancelled staging",
            task_id,
        )
    raise cancellation


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


def _task_lease_snapshot(task: Any) -> TaskLease | None:
    """Detach the exact active run identity needed by live-control writes."""

    task_id = getattr(task, "id", None)
    runner_id = getattr(task, "runner_id", None)
    run_id = getattr(task, "run_id", None)
    if task_id is None or runner_id is None or run_id is None:
        return None
    return TaskLease(
        # attempt_id is left at its None default on purpose: every field here
        # comes from the task row, so reading lease_attempt_id from that same
        # row would make any later attempt check compare a value against
        # itself. See TaskLease's docstring in task_lease_service.py.
        task_id=int(task_id),
        runner_id=str(runner_id),
        run_id=str(run_id),
    )


def _task_control_state_value(task: Any) -> str | None:
    control_state = getattr(task, "control_state", None)
    return str(control_state) if control_state is not None else None


@dataclass(frozen=True)
class _TaskExecutionFinalization:
    normalized_outputs: list[dict[str, Any]]
    ai_response: Any
    chat_response: Any
    waiting_for_control: bool
    terminal_state_committed: bool
    final_control_snapshot: TaskControlSnapshot | None
    final_task_status: str
    broadcast_meta: dict[str, Any]
    late_result: bool = False


def _finalize_task_execution_result_isolated(
    *,
    task_id: int,
    task_user_id: int | None,
    pre_run_status: TaskStatus,
    result: dict[str, Any],
    expected_run_id: str | None,
    task_lease: TaskLease | None,
    resolved_scope_segments: tuple[str, ...],
    prepared_outputs: _PreparedTaskFileOutputs | None = None,
) -> _TaskExecutionFinalization:
    """Persist one task result in a worker-owned, ownership-fenced session."""
    from ..services.chat_history_service import persist_assistant_message_no_commit

    if prepared_outputs is None:
        resolved_output_user_id = task_user_id
        if resolved_output_user_id is None:
            SessionLocal = get_session_local()
            with SessionLocal() as lookup_db:
                resolved_output_user_id = (
                    lookup_db.query(Task.user_id).filter(Task.id == task_id).scalar()
                )
        prepared_outputs = (
            _prepare_task_file_outputs_isolated(
                task_id=task_id,
                task_user_id=int(resolved_output_user_id),
                file_outputs=result.get("file_outputs", []),
                resolved_scope_segments=resolved_scope_segments,
            )
            if resolved_output_user_id is not None
            else _PreparedTaskFileOutputs((), (), ())
        )

    SessionLocal = get_session_local()
    finalize_db = SessionLocal()
    metadata_committed = False
    cleanup_claims: tuple[SupersededObjectCleanupClaim, ...] = ()
    try:
        default_response = "Task completed" if result.get("success", False) else ""
        chat_response = result.get("chat_response")
        if isinstance(chat_response, dict):
            ai_response = chat_response.get("message") or result.get(
                "output", default_response
            )
        else:
            ai_response = result.get("output", default_response)

        task_query = finalize_db.query(Task).filter(Task.id == task_id)
        if task_lease is not None:
            if task_lease.run_id is None:
                logger.warning(
                    "Task %s result has an unfenced lease; refusing finalization",
                    task_id,
                )
                task_updated = None
                late_result = True
            else:
                task_updated = (
                    task_query.filter(
                        Task.runner_id == task_lease.runner_id,
                        Task.run_id == task_lease.run_id,
                    )
                    .with_for_update()
                    .first()
                )
                late_result = task_updated is None
        else:
            # Legacy callers have no concrete owner identity and may only
            # finalize an ownerless task row.
            task_query = task_query.filter(Task.runner_id.is_(None))
            if expected_run_id is not None:
                task_query = task_query.filter(Task.run_id == expected_run_id)
            task_updated = task_query.with_for_update().first()
            late_result = task_updated is None

        if late_result:
            finalize_db.rollback()
            logger.info(
                "Ignoring late task result for task %s run %s; ownership changed",
                task_id,
                expected_run_id,
            )
            return _TaskExecutionFinalization(
                normalized_outputs=[],
                ai_response=ai_response,
                chat_response=chat_response,
                waiting_for_control=False,
                terminal_state_committed=False,
                final_control_snapshot=None,
                final_task_status=pre_run_status.value,
                broadcast_meta={},
                late_result=True,
            )

        (
            normalized_outputs,
            path_to_file_id,
            cleanup_claims,
        ) = _apply_prepared_task_file_outputs(finalize_db, prepared_outputs)
        ai_response = _rewrite_file_links_to_file_id(ai_response, path_to_file_id)
        if task_user_id is not None:
            ai_response = reconcile_assistant_file_references(
                finalize_db,
                task_id=task_id,
                user_id=task_user_id,
                content=ai_response,
            )
            if isinstance(chat_response, dict) and chat_response.get("message"):
                chat_response = {**chat_response, "message": ai_response}

        waiting_for_control = False
        terminal_state_committed = False
        final_control_snapshot: TaskControlSnapshot | None = None
        final_task_status = pre_run_status.value

        if task_updated is not None:
            task_agent_config: dict[str, Any] = (
                task_updated.agent_config
                if isinstance(task_updated.agent_config, dict)
                else {}
            )
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
                sync_workforce_run_status(
                    finalize_db,
                    task_updated,
                    task_updated.status,
                )
                finalize_db.commit()
                metadata_committed = True
                terminal_state_committed = True
                waiting_for_control = True
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
                sync_workforce_run_status(
                    finalize_db,
                    task_updated,
                    task_updated.status,
                )
                finalize_db.commit()
                metadata_committed = True
                terminal_state_committed = True
                waiting_for_control = True
            elif task_updated.status not in {
                TaskStatus.PAUSED,
                TaskStatus.WAITING_FOR_USER,
            }:
                final_status = (
                    TaskStatus.COMPLETED
                    if result.get("success", False)
                    else TaskStatus.FAILED
                )
                final_control_snapshot = apply_task_control_transition(
                    task_updated,
                    TaskControlState.COMPLETED
                    if final_status == TaskStatus.COMPLETED
                    else TaskControlState.FAILED,
                    status=final_status,
                    expected_run_id=expected_run_id,
                )
                if final_status == TaskStatus.FAILED:
                    diagnostic_error = safe_str(result.get("error")).strip()
                    setattr(
                        task_updated,
                        "error_message",
                        diagnostic_error
                        or safe_str(ai_response).strip()
                        or CLIENT_SAFE_TASK_FAILURE,
                    )
                sync_workforce_run_status(
                    finalize_db,
                    task_updated,
                    task_updated.status,
                )
            else:
                waiting_for_control = True
                terminal_state_committed = True

            final_task_status = task_updated.status.value
            if not waiting_for_control:
                if task_user_id is None:
                    raise ValueError(
                        f"Task {task_id}: cannot persist assistant message "
                        "without a resolved user_id"
                    )
                history_content, history_message_type = (
                    assistant_history_values_for_persistence(
                        content=safe_str(ai_response),
                        message_type=ASSISTANT_RESPONSE_MESSAGE_TYPE,
                        is_failure=task_updated.status == TaskStatus.FAILED,
                    )
                )
                persist_assistant_message_no_commit(
                    finalize_db,
                    task_id=task_id,
                    user_id=task_user_id,
                    content=history_content,
                    message_type=history_message_type,
                    interactions=(
                        chat_response.get("interactions")
                        if isinstance(chat_response, dict)
                        and task_updated.status != TaskStatus.FAILED
                        else None
                    ),
                    content_is_reconciled=True,
                )
                finalize_db.commit()
                metadata_committed = True
                terminal_state_committed = True

            broadcast_meta = {
                "id": int(task_updated.id),
                "title": task_updated.title,
                "description": task_updated.description,
                "execution_mode": getattr(task_updated, "execution_mode", None),
                "updated_at": task_updated.updated_at,
            }
        else:
            broadcast_meta = {
                "id": task_id,
                "title": None,
                "description": None,
                "execution_mode": None,
                "updated_at": None,
            }

        return _TaskExecutionFinalization(
            normalized_outputs=normalized_outputs,
            ai_response=ai_response,
            chat_response=chat_response,
            waiting_for_control=waiting_for_control,
            terminal_state_committed=terminal_state_committed,
            final_control_snapshot=final_control_snapshot,
            final_task_status=final_task_status,
            broadcast_meta=broadcast_meta,
        )
    finally:
        try:
            finalize_db.close()
        finally:
            _settle_prepared_task_file_outputs(
                prepared_outputs,
                metadata_committed=metadata_committed,
                cleanup_claims=cleanup_claims,
            )


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
    task_lease: TaskLease | None = None,
    resolved_execution_scope: Union[
        ExecutionScope, None, ExecutionScopeNotProvided
    ] = EXECUTION_SCOPE_NOT_PROVIDED,
    mcp_runtime_authorization_policy: MCPBuiltinOAuthActorPolicy | None = None,
) -> None:
    """Execute one task without checking out a DB connection on the event loop.

    Setup and finalization use worker-owned short Sessions. The long-running
    agent await receives only detached runtime state and primitive identifiers.
    """
    from ..services.task_execution_context_service import (
        materialize_task_execution_recovery_state,
    )
    from ..services.task_setup_snapshot import load_task_setup_snapshot_sync

    terminal_state_committed = False
    try:
        if resolved_execution_scope is EXECUTION_SCOPE_NOT_PROVIDED:
            execution_scope = await run_db_io_cancellation_safe(
                lambda: resolve_execution_scope(task_id)
            )
        else:
            execution_scope = cast(
                Optional[ExecutionScope],
                resolved_execution_scope,
            )

        snapshot = task_setup_snapshot
        if snapshot is None:
            snapshot = await run_db_io_cancellation_safe(
                lambda: load_task_setup_snapshot_sync(
                    task_id,
                    task_owner_user_id,
                    before_message_id=before_message_id,
                )
            )
        if snapshot is None:
            raise ClientVisibleValidationError(
                f"Task {task_id} not found",
                error_code=ClientErrorCode.TASK_UNAVAILABLE,
            )

        context_dict = context if isinstance(context, dict) else {}
        logger.info(f"Background task execution started for task {task_id}")
        task_user_id = snapshot.task.user_id
        user = snapshot.runtime_user

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
        effective_user_id = task_user_id

        with UserContext(effective_user_id), ExecutionScopeContext(execution_scope):
            # Get agent service. ``effective_user_id`` is the task owner
            # (authoritative above); pass it as the runtime identity so the
            # agent's models / tools resolve as the owner, not any acting admin.
            agent_service = await agent_manager.get_agent_for_task(
                task_id,
                None,
                user=user,
                task_setup_snapshot=snapshot,
                task_owner_user_id=effective_user_id,
                connector_runtime_turn_id=context_dict.get("turn_id")
                if isinstance(context_dict.get("turn_id"), str)
                else None,
                mcp_runtime_authorization_policy=(mcp_runtime_authorization_policy),
                resolved_execution_scope=execution_scope,
            )
            if hasattr(agent_service, "set_outbound_message_handler"):
                agent_service.set_outbound_message_handler(
                    make_agent_outbound_handler(task_id)
                )
            agent_service.set_conversation_history(
                [dict(message) for message in snapshot.conversation_history]
            )
            recovery_state = await materialize_task_execution_recovery_state(
                snapshot.execution_recovery
            )
            execution_context_messages = recovery_state.get("messages", [])
            agent_service.set_execution_context_messages(execution_context_messages)
            agent_service.set_recovered_skill_context(
                recovery_state.get("skill_context")
            )
            await run_db_io_cancellation_safe(
                lambda: _register_uploaded_files_for_agent(
                    agent_service,
                    context_dict.get("file_info", []),
                )
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
                db_session=None,
                manage_task_lease=False,
                task_lease=task_lease,
            )

        finalize_run_id = (
            task_lease.run_id if task_lease is not None else expected_run_id
        )
        finalization_worker = asyncio.create_task(
            asyncio.to_thread(
                lambda: _finalize_task_execution_result_isolated(
                    task_id=task_id,
                    task_user_id=effective_user_id,
                    pre_run_status=cast(TaskStatus, snapshot.task.status),
                    result=result,
                    expected_run_id=finalize_run_id,
                    task_lease=task_lease,
                    resolved_scope_segments=(
                        execution_scope.workspace_segments
                        if execution_scope is not None
                        else ()
                    ),
                )
            )
        )
        finalized, finalization_cancellation = await await_task_settlement(
            finalization_worker
        )
        with propagate_deferred_cancellation(finalization_cancellation):
            if finalized.late_result:
                return

            normalized_outputs = finalized.normalized_outputs
            if normalized_outputs:
                result["file_outputs"] = normalized_outputs
            ai_response = finalized.ai_response
            chat_response = finalized.chat_response
            waiting_for_control = finalized.waiting_for_control
            terminal_state_committed = finalized.terminal_state_committed
            final_control_snapshot = finalized.final_control_snapshot
            final_task_status = finalized.final_task_status
            broadcast_meta = finalized.broadcast_meta
            broadcast_agent_meta = {
                "agent_id": snapshot.task.agent_id,
                "agent_name": (
                    snapshot.agent.name if snapshot.agent is not None else None
                ),
                "agent_logo_url": None,
            }

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
                    "type": "task_completed",
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
        if task_lease is not None:
            if terminal_state_committed:
                logger.warning(
                    "Background task %s post-terminal step failed; "
                    "task state left unchanged: %s",
                    task_id,
                    e,
                    exc_info=True,
                )
                return

            if is_database_pool_timeout(e):
                # The orchestrator owns the concrete lease and will retain it
                # for TTL recovery. Broadcasting FAILED here would contradict
                # the durable RUNNING + fenced-lease state.
                logger.error(
                    "task_id=%s component=execution database pool checkout "
                    "timed out; retaining exact lease for TTL recovery without "
                    "broadcasting task_error: %s",
                    task_id,
                    e,
                    exc_info=True,
                )
                raise

            logger.error(
                "Background task %s execution failed: %s",
                task_id,
                e,
                exc_info=True,
            )
            # The concrete run/runner lease belongs to the orchestrator. Let
            # its single worker-owned settlement transaction persist failure
            # and release the lease before it emits any terminal event. Doing
            # either DB work or a broadcast here could race a replacement run.
            raise

        error_message = str(e)
        error_code = (
            e.error_code
            if isinstance(e, ClientVisibleError)
            else ClientErrorCode.TASK_EXECUTION_FAILED
        )
        safe_error_message = client_error_message(error_code)
        terminal_payload = await run_db_io_cancellation_safe(
            lambda: _terminal_task_error_payload(
                task_id,
                error_message,
                event_type="task_error",
                expected_run_id=expected_run_id,
                only_if_running=True,
            )
        )

        if terminal_payload is None:
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
            # + the real error_message for diagnostics. Replace every
            # client-visible copy in the notification payload: the spread
            # already carries ``message``, while older clients also read
            # ``error``.
            try:
                await manager.broadcast_to_task(
                    {
                        **terminal_payload,
                        "task_id": task_id,
                        "message": safe_error_message,
                        "error": safe_error_message,
                        "error_code": error_code.value,
                        "timestamp": datetime.now(timezone.utc).timestamp(),
                    },
                    task_id,
                )
            except Exception as broadcast_error:
                logger.error(f"Failed to send error notification: {broadcast_error}")
    except asyncio.CancelledError as cancellation:
        deferred_error = cancellation.__cause__
        if deferred_error is not None and not isinstance(
            deferred_error, asyncio.CancelledError
        ):
            logger.warning(
                "Background task %s cancelled after deferred work failed: %s",
                task_id,
                deferred_error,
                exc_info=(
                    type(deferred_error),
                    deferred_error,
                    deferred_error.__traceback__,
                ),
            )
        else:
            logger.info("Background task %s cancelled", task_id)
        raise
    finally:
        # Clean up background task record
        _clear_task_pause_accepted(task_id)
        background_task_manager.cleanup_task(task_id)


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


def _acquire_resume_task_lease(
    task_id: int,
    task_owner_user_id: int | None,
    expected_run_id: str | None,
    *,
    prior_status_out: list[TaskStatus] | None = None,
) -> TaskLease | None:
    """Validate and claim a resume lease in one worker transaction.

    ``prior_status_out``, when given, receives the task's status as read
    here -- before the lease-acquiring update below flips it to RUNNING.
    A checkpoint read failure later in the resume attempt needs this to
    restore the task instead of falling through to a terminal FAILED. It
    is an out parameter rather than part of the return value because this
    function is called through ``acquire_task_lease_cancellation_safe``,
    whose acquire/cleanup pair is typed for a bare ``TaskLease``.
    """
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = db.query(Task).filter(Task.id == task_id).first()
        if (
            task is not None
            and task_owner_user_id is not None
            and int(task.user_id) != task_owner_user_id
        ):
            raise ValueError(
                f"execute_resume_background: passed task_owner_user_id "
                f"{task_owner_user_id} does not match task {task_id} "
                f"owner {int(task.user_id)}; refusing to resume as the "
                "wrong user"
            )
        if task is not None and prior_status_out is not None:
            prior_status_out.append(TaskStatus(task.status))
        lease = acquire_task_lease_no_commit(
            db,
            task_id,
            expected_run_id=expected_run_id,
        )
        if lease is None:
            db.commit()
            return None
        if task is not None:
            db.expire(task)
            db.refresh(task)
            sync_workforce_run_status(db, task, TaskStatus.RUNNING)
        db.commit()
        return lease


def _restore_resumed_task_lease_to_prior_status(
    lease: TaskLease,
    *,
    status: TaskStatus,
) -> bool:
    """Release the exact resume lease back to its pre-acquisition status.

    A checkpoint read failure during resume must not silently downgrade a
    paused/waiting task to a terminal FAILED. Uses the same exact-lease
    WHERE fence (task id + runner id + run id) as the TTL reaper, so the
    two can never both release the same row -- whichever loses the race
    affects zero rows instead of double-releasing. The commit below is
    unconditional, unlike the A2A prelease restore: when the fence excludes
    every row, the UPDATE affects zero rows and this commits that no-op
    rather than rolling back. That unconditional commit is unrelated to the
    protocol-marker clear below, which is conditioned on ``restored``: a
    fence miss means this call lost the race for the row and must not touch
    a marker some other winner now owns.
    """
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        restored = release_task_lease_no_commit(db, lease, status=status)
        if restored:
            # This is a resume abandonment, not a completion: no injection
            # ran, so there is no interaction row to close here, only a
            # marker to reconcile if it no longer names an active row. See
            # clear_interaction_marker_if_unpaired's docstring for the
            # NOT EXISTS semantics. No lock read precedes this statement --
            # release_task_lease_no_commit's own tasks UPDATE writes only
            # non-key columns and is already the first statement this
            # transaction directs at tasks or task_interaction_requests, so
            # it already satisfies the ordering and strength obligation a
            # dedicated lock read would.
            assert lease.run_id is not None
            clear_interaction_marker_if_unpaired(
                db, task_id=lease.task_id, run_id=lease.run_id
            )
        db.commit()
        return restored


def _finalize_resumed_task(
    task_id: int,
    *,
    status: str,
    success: bool,
    output: str,
    task_owner_user_id: int | None,
    result: Dict[str, Any],
    task_lease: TaskLease,
    prepared_outputs: _PreparedTaskFileOutputs,
) -> dict[str, Any]:
    """Persist one fenced resumed result in a single worker transaction."""
    from ..models.agent import Agent
    from ..services.chat_history_service import persist_assistant_message_no_commit

    finalized: dict[str, Any] = {
        "task_title": None,
        "task_description": None,
        "task_execution_mode": None,
        "task_agent_id": None,
        "agent_name": None,
        "agent_logo_url": None,
        "final_status": TaskStatus.RUNNING.value,
        "lease_released": False,
        "control_event_state": {},
        "normalized_outputs": [],
        "output": output,
        "late_result": False,
    }
    if task_lease.run_id is None:
        _settle_prepared_task_file_outputs(
            prepared_outputs,
            metadata_committed=False,
        )
        finalized["late_result"] = True
        return finalized
    SessionLocal = get_session_local()
    db = SessionLocal()
    metadata_committed = False
    cleanup_claims: tuple[SupersededObjectCleanupClaim, ...] = ()
    try:
        task = (
            db.query(Task)
            .filter(
                Task.id == task_id,
                Task.runner_id == task_lease.runner_id,
                Task.run_id == task_lease.run_id,
            )
            .with_for_update()
            .first()
        )
        if task is None:
            finalized["late_result"] = True
            return finalized

        (
            normalized_outputs,
            path_to_file_id,
            cleanup_claims,
        ) = _apply_prepared_task_file_outputs(db, prepared_outputs)
        if normalized_outputs:
            output = _rewrite_file_links_to_file_id(output, path_to_file_id)
        if task_owner_user_id is not None:
            output = reconcile_assistant_file_references(
                db,
                task_id=task_id,
                user_id=task_owner_user_id,
                content=output,
            )
        finalized["normalized_outputs"] = normalized_outputs
        finalized["output"] = output

        finalized["task_title"] = cast(Any, task.title)
        finalized["task_description"] = cast(Any, task.description)
        finalized["task_execution_mode"] = cast(Any, task.execution_mode)
        finalized["task_agent_id"] = cast(Any, task.agent_id)
        if task.agent_id is not None:
            agent = db.query(Agent).filter(Agent.id == task.agent_id).first()
            if agent is not None:
                finalized["agent_name"] = cast(Any, agent.name)
                finalized["agent_logo_url"] = cast(Any, agent.logo_url)

        if status == "waiting_for_user":
            final_task_status = TaskStatus.WAITING_FOR_USER
        elif status == "interrupted":
            final_task_status = TaskStatus.PAUSED
        elif success:
            final_task_status = TaskStatus.COMPLETED
        else:
            final_task_status = TaskStatus.FAILED

        control_snapshot = apply_task_control_transition(
            task,
            {
                TaskStatus.WAITING_FOR_USER: TaskControlState.WAITING_FOR_USER,
                TaskStatus.PAUSED: TaskControlState.PAUSED,
                TaskStatus.COMPLETED: TaskControlState.COMPLETED,
                TaskStatus.FAILED: TaskControlState.FAILED,
            }[final_task_status],
            status=final_task_status,
            expected_run_id=task_lease.run_id,
        )

        if success and output.strip() and task_owner_user_id is not None:
            persist_assistant_message_no_commit(
                db,
                task_id=task_id,
                user_id=task_owner_user_id,
                content=output,
                message_type=ASSISTANT_RESPONSE_MESSAGE_TYPE,
                turn_id=_latest_result_user_turn_id(result),
                content_is_reconciled=True,
            )
            orm_task = cast(Any, task)
            orm_task.output = output
            orm_task.error_message = None
        elif final_task_status == TaskStatus.FAILED:
            if task_owner_user_id is not None:
                persist_assistant_message_no_commit(
                    db,
                    task_id=task_id,
                    user_id=task_owner_user_id,
                    content=CLIENT_SAFE_TASK_FAILURE,
                    message_type=TASK_FAILURE_MESSAGE_TYPE,
                    turn_id=_latest_result_user_turn_id(result),
                    content_is_reconciled=True,
                )
            orm_task = cast(Any, task)
            orm_task.output = None
            orm_task.error_message = (
                str(result.get("error") or "").strip()
                or output
                or CLIENT_SAFE_TASK_FAILURE
            )

        sync_workforce_run_status(db, task, final_task_status)
        lease_released = release_task_lease_no_commit(
            db,
            task_lease,
            status=final_task_status,
        )
        if not lease_released:
            db.rollback()
            finalized["late_result"] = True
            return finalized
        db.commit()
        metadata_committed = True
        finalized["lease_released"] = True
        finalized["final_status"] = final_task_status.value
        finalized["control_event_state"] = control_snapshot.as_dict()
        return finalized
    finally:
        try:
            db.close()
        finally:
            _settle_prepared_task_file_outputs(
                prepared_outputs,
                metadata_committed=metadata_committed,
                cleanup_claims=cleanup_claims,
            )


def _settle_resumed_task_lease(
    lease: TaskLease,
    *,
    error_message: str | None,
) -> bool:
    """Delegate resume cleanup to the shared run/runner-fenced lifecycle."""
    from ..services.task_orchestrator import settle_task_lease_isolated

    return settle_task_lease_isolated(lease, error_message=error_message)


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
    # Defaulting to None is a structurally open door, not exercised by any
    # caller today: acquire_task_lease_no_commit (task_lease_service.py)
    # mints a fresh uuid for a None here, so a future call site that
    # forgets to pass its own run id would silently claim a lease under a
    # run nobody else knows about instead of failing loudly.
    expected_run_id: str | None = None,
    resolved_execution_scope: Union[
        ExecutionScope, None, ExecutionScopeNotProvided
    ] = EXECUTION_SCOPE_NOT_PROVIDED,
    preacquired_lease: TaskLease | None = None,
    preacquired_heartbeat_stop: asyncio.Event | None = None,
    preacquired_heartbeat_task: (asyncio.Task[TaskLeaseHeartbeatOutcome] | None) = None,
    preacquired_prior_status: TaskStatus | None = None,
) -> None:
    """Resume an agent execution after an interrupt/user-message checkpoint.

    ``task_owner_user_id`` is the task OWNER's id -- the runtime identity the
    resume executes as (``UserContext``), not the acting principal.
    """
    resume_owner_task = asyncio.current_task()
    if resume_owner_task is None:
        raise RuntimeError(f"Task {task_id} resume has no asyncio task")

    lease_stop_event = preacquired_heartbeat_stop
    lease_heartbeat_task = preacquired_heartbeat_task
    lease: TaskLease | None = preacquired_lease
    lease_released = False
    settlement_error: str | None = None
    broadcast_error_message: str | None = None
    defer_db_cleanup_to_ttl_recovery = False
    # The status this task held before a lease claim flipped it to RUNNING;
    # the checkpoint-unavailable/refused recovery path below restores to
    # this instead of a terminal FAILED. Captured at acquisition when this
    # call claims the lease, and handed over by the claimant when the lease
    # was preacquired -- a caller that claims the lease elsewhere owns the
    # same obligation, or its resume would answer a transient read failure
    # by downgrading a still-resumable task.
    resume_prior_status: TaskStatus | None = None
    restore_lease_to_prior_status: TaskStatus | None = None
    result: Dict[str, Any] | None = None
    prepared_outputs: _PreparedTaskFileOutputs | None = None
    # Token tracking + mid-run quota gate for the resumed segment (resume had
    # neither before, so a resumed run escaped mid-run enforcement entirely).
    resume_tracker = None
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
    control_event_state: dict[str, Any] = {}

    async def notify_deferred_delivery(
        accepted: bool,
        message: str | None = None,
        *,
        error_code: ClientErrorCode | None = None,
        retry_with_new_id: bool = False,
        rejection_outcome: Literal["not_accepted", "outcome_unknown"] | None = None,
    ) -> None:
        if delivery_websocket is None or delivery_client_message_id is None:
            return
        try:
            await send_message_delivery(
                delivery_websocket,
                client_message_id=delivery_client_message_id,
                turn_id=delivery_turn_id or delivery_client_message_id,
                accepted=accepted,
                message=message,
                error_code=error_code.value if error_code is not None else None,
                retry_with_new_id=retry_with_new_id,
                rejection_outcome=rejection_outcome,
            )
        except Exception:
            # Delivery state is durable; a disconnected client will retry the
            # same id and recover the result from that state.
            logger.warning(
                "Could not send deferred delivery acknowledgement for task %s",
                task_id,
                exc_info=True,
            )

    async def mark_deferred_delivery_failed() -> bool:
        """Persist a failed delivery without amplifying pool exhaustion."""
        nonlocal defer_db_cleanup_to_ttl_recovery
        if delivery_turn_id is None or delivery_was_dispatched:
            return True
        try:
            await run_db_io_cancellation_safe(
                lambda: mark_user_message_delivery_sync(
                    task_id,
                    delivery_turn_id,
                    DELIVERY_FAILED,
                )
            )
            return True
        except Exception as delivery_error:
            if not is_database_pool_timeout(delivery_error):
                raise
            defer_db_cleanup_to_ttl_recovery = lease is not None and not lease_released
            logger.error(
                "task_id=%s component=resume-delivery database pool checkout "
                "timed out; skipping immediate settlement and retaining lease "
                "for TTL recovery: %s",
                task_id,
                delivery_error,
                exc_info=True,
            )
            return False

    try:
        preacquired_resources = (
            preacquired_lease,
            preacquired_heartbeat_stop,
            preacquired_heartbeat_task,
        )
        if any(resource is not None for resource in preacquired_resources) and not all(
            resource is not None for resource in preacquired_resources
        ):
            raise ValueError(
                "A preacquired resume lease, heartbeat stop event, and heartbeat "
                "task must be transferred together"
            )
        if preacquired_lease is not None:
            if preacquired_lease.task_id != task_id or preacquired_lease.run_id is None:
                raise ValueError(
                    "A preacquired resume lease must match the task and exact run"
                )
            if (
                expected_run_id is not None
                and preacquired_lease.run_id != expected_run_id
            ):
                raise ValueError(
                    "The preacquired resume lease does not match expected_run_id"
                )
            # Adopt the claimant's pre-acquisition status so the recovery
            # path below is wired on this entry too, not only when this
            # call claimed the lease itself.
            resume_prior_status = preacquired_prior_status
        if previous_task is not None and not previous_task.done():
            try:
                await previous_task
            except Exception as e:
                logger.warning(
                    f"Previous background task {task_id} ended before resume: {e}"
                )

        background_task_manager.promote_resume_task(task_id, resume_owner_task)

        if resolved_execution_scope is EXECUTION_SCOPE_NOT_PROVIDED:
            execution_scope = await run_db_io_cancellation_safe(
                lambda: resolve_execution_scope(task_id)
            )
        else:
            execution_scope = cast(
                Optional[ExecutionScope],
                resolved_execution_scope,
            )

        if lease is None:
            prior_status_box: list[TaskStatus] = []
            lease = await acquire_task_lease_cancellation_safe(
                lambda: _acquire_resume_task_lease(
                    task_id,
                    task_owner_user_id,
                    expected_run_id,
                    prior_status_out=prior_status_box,
                ),
                lambda acquired: _settle_resumed_task_lease(
                    acquired,
                    error_message="resume cancelled during lease acquisition",
                ),
            )
            if prior_status_box:
                resume_prior_status = prior_status_box[0]
            if lease is None:
                logger.info(
                    "Task %s resume skipped; another runner owns the lease", task_id
                )
                if delivery_turn_id is not None and not delivery_was_dispatched:
                    await run_db_io_cancellation_safe(
                        lambda: mark_user_message_delivery_sync(
                            task_id,
                            delivery_turn_id,
                            DELIVERY_FAILED,
                        )
                    )
                    await notify_deferred_delivery(
                        False,
                        client_error_message(ClientErrorCode.MESSAGE_DELIVERY_FAILED),
                        error_code=ClientErrorCode.MESSAGE_DELIVERY_FAILED,
                        retry_with_new_id=True,
                        rejection_outcome="not_accepted",
                    )
                await manager.broadcast_to_task(
                    {
                        "type": "agent_error",
                        "message": client_error_message(ClientErrorCode.TASK_BUSY),
                        "error_code": ClientErrorCode.TASK_BUSY.value,
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
        else:
            # The caller acquired and committed this exact lease before
            # injecting a checkpoint message. Ownership of both lease and
            # heartbeat transfers atomically to this background task; a second
            # acquisition would either self-block on a size-1 pool or create a
            # second runner identity for the same resume.
            assert lease_stop_event is not None
            assert lease_heartbeat_task is not None

        # The task row can become RUNNING before the original AgentRunner has
        # created a context/checkpoint. Retry an early failed injection only
        # after that original execution has settled and persisted its state.
        # Acquire the execution lease first: otherwise a non-owner worker could
        # persist the injection and acknowledge it, then discover that it is
        # not allowed to run the resume.
        if pending_user_message is not None:
            assert lease_heartbeat_task is not None
            with bind_task_lease_context(lease):
                posted = await run_while_task_lease_owned(
                    agent_service.post_user_message(
                        str(task_id),
                        execution_message=pending_user_message.get("execution_message"),
                        display_message=pending_user_message.get("display_message"),
                        files=pending_user_message.get("files"),
                        turn_id=pending_user_message.get("turn_id"),
                        request_interrupt=False,
                        reason="deferred websocket user message",
                    ),
                    lease_heartbeat_task,
                )
            if not posted:
                raise RuntimeError(
                    "The user message was saved, but no resumable execution "
                    "checkpoint became available."
                )
            delivery_was_dispatched = True
            # Unconditional and not nested inside the delivery_turn_id branch
            # below: retiring this run's active interaction row and clearing
            # the task's protocol marker has nothing to do with whether a
            # delivery-ack turn id is present. Borrowing that condition would
            # give the close a gate it has no reason to have. Bound to a
            # plain local first, not read from lease inside the lambda below:
            # a narrowing assert on an enclosing-scope variable does not
            # apply inside a nested closure.
            assert lease.run_id is not None
            close_run_id = lease.run_id
            # Read by the online handler before this message was injected,
            # and carried here rather than read now for the opposite reason
            # to the one it looks like: the injection is not still to come,
            # it is the post_user_message call above and has already
            # committed by this line. Injecting is what resumes the agent,
            # so a read here could name a question the resumed agent has
            # staged since, not the one the message answered. Bound to a
            # plain local before the lambda below, like close_run_id above.
            close_interaction_id = pending_user_message.get("interaction_id")
            # Distinct from the "unconditional" argument above, which is
            # only about not borrowing the delivery_turn_id branch's
            # condition: this task can itself be retried across runs with
            # the same pending_user_message, and post_user_message reports
            # that retry explicitly as a replay instead of a bare truthy
            # `posted`. What the guard buys here is narrower than at the
            # sites that read their own id: the id carried above is a
            # primary key the first attempt already retired, and the close
            # statement binds to it, so on a replay the close would be a
            # no-op rather than a retirement of a live question. The guard
            # stays as defense in depth -- it is what keeps this site safe
            # if it ever stops carrying the id forward and starts deriving
            # its own. See task_interaction_close's module docstring for
            # the rule, the other sites, and why the v1 reply resume-input
            # path needs no guard at all.
            if posted is UserMessageInjectionOutcome.POSTED_FRESH:
                try:
                    await run_db_io_cancellation_safe(
                        lambda: close_legacy_resume_interaction_sync(
                            task_id=task_id,
                            run_id=close_run_id,
                            interaction_id=close_interaction_id,
                        )
                    )
                except Exception:
                    logger.warning(
                        "legacy resume interaction close failed after deferred "
                        "message seal for task %s run %s",
                        task_id,
                        close_run_id,
                        exc_info=True,
                    )
                except asyncio.CancelledError:
                    # See run_db_io_cancellation_safe's docstring: it drains
                    # its worker to completion before propagating a
                    # cancellation raised while awaiting it, so the
                    # close-and-clear transaction has already committed or
                    # failed by the time this branch runs. Only the log
                    # statement was interrupted.
                    logger.warning(
                        "legacy resume interaction close was cancelled after "
                        "deferred message seal for task %s run %s; continuing "
                        "resume",
                        task_id,
                        close_run_id,
                    )
            if delivery_turn_id is not None:
                try:
                    await run_db_io_cancellation_safe(
                        lambda: mark_user_message_delivery_sync(
                            task_id,
                            delivery_turn_id,
                            DELIVERY_DISPATCHED,
                        )
                    )
                except Exception:
                    logger.warning(
                        "delivery marker failed after deferred message seal "
                        "for task %s turn %s",
                        task_id,
                        delivery_turn_id,
                        exc_info=True,
                    )
                except asyncio.CancelledError:
                    # Once the checkpoint write has accepted this turn, task
                    # cancellation must not turn that durable success into a
                    # failed delivery. Continue the registered resume; the
                    # marker is monotonic and can be reconciled from the
                    # checkpoint on retry.
                    logger.warning(
                        "delivery marker was cancelled after deferred message "
                        "seal for task %s turn %s; continuing resume",
                        task_id,
                        delivery_turn_id,
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

            resume_tracker = TaskTracker(
                task_id=int(task_id),
                expected_run_id=lease.run_id,
                expected_runner_id=lease.runner_id,
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

        assert lease_heartbeat_task is not None
        with (
            UserContext(task_owner_user_id),
            ExecutionScopeContext(execution_scope),
            bind_task_lease_context(lease),
        ):
            result = await run_while_task_lease_owned(
                agent_service.resume_execution_by_id(str(task_id)),
                lease_heartbeat_task,
            )

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

        # Final usage belongs to this exact run. Persist it while the run still
        # owns the lease, then stop heartbeat before the atomic result/lease
        # finalizer. If the usage checkout itself times out, its exception is
        # handled below and the lease is deliberately retained for TTL recovery
        # instead of performing a second checkout against the exhausted pool.
        if resume_tracker is not None:
            tracker_to_complete = resume_tracker
            resume_tracker = None
            agent_service.set_interrupt_checker(None)
            await tracker_to_complete.complete_tracking()

        # Output object storage can be arbitrarily slow. Stage and checksum it
        # while this exact runner's heartbeat is still active; after heartbeat
        # shutdown only the short fenced metadata/lease transaction remains.
        prepared_outputs = await _prepare_task_file_outputs_cancellation_safe(
            task_id=task_id,
            task_user_id=task_owner_user_id,
            file_outputs=result.get("file_outputs", []),
            resolved_scope_segments=(
                execution_scope.workspace_segments
                if execution_scope is not None
                else ()
            ),
        )

        heartbeat_outcome = await stop_task_lease_heartbeat(
            lease_heartbeat_task, lease_stop_event
        )
        lease_heartbeat_task = None
        lease_stop_event = None
        if (
            isinstance(heartbeat_outcome, TaskLeaseHeartbeatOutcome)
            and heartbeat_outcome.requires_ttl_recovery
        ):
            defer_db_cleanup_to_ttl_recovery = True
            logger.error(
                "task_id=%s component=resume-heartbeat unhealthy before "
                "finalization; retaining lease for TTL recovery (lost=%s, "
                "pool_timeout=%s)",
                task_id,
                heartbeat_outcome.lease_lost,
                heartbeat_outcome.pool_timeout is not None,
            )
            if heartbeat_outcome.pool_timeout is not None:
                raise heartbeat_outcome.pool_timeout
            # A replacement owner already holds the task. Do not persist or
            # broadcast this stale runner's result.
            return

        outputs_for_finalizer = prepared_outputs
        try:
            finalized = await run_db_io_cancellation_safe(
                lambda: _finalize_resumed_task(
                    task_id,
                    status=status,
                    success=success,
                    output=output,
                    task_owner_user_id=task_owner_user_id,
                    result=result,
                    task_lease=lease,
                    prepared_outputs=outputs_for_finalizer,
                )
            )
        finally:
            # Once invoked, the fenced finalizer owns success cleanup or
            # compensation, including cancellation-safe late completion.
            prepared_outputs = None
        if finalized["late_result"]:
            logger.info(
                "Ignoring late resume result for task %s run %s; ownership changed",
                task_id,
                lease.run_id,
            )
            # Ownership was already checked inside the fenced finalizer. There
            # is no lease from this run left to settle, and another checkout
            # would only race the replacement run.
            lease_released = True
            return
        normalized_outputs = finalized["normalized_outputs"]
        output = finalized["output"]
        if normalized_outputs:
            result["file_outputs"] = normalized_outputs
        task_title = finalized["task_title"]
        task_description = finalized["task_description"]
        task_execution_mode = finalized["task_execution_mode"]
        task_agent_id = finalized["task_agent_id"]
        agent_name = finalized["agent_name"]
        agent_logo_url = finalized["agent_logo_url"]
        final_status = finalized["final_status"]
        lease_released = bool(finalized["lease_released"])
        control_event_state = finalized["control_event_state"]

        if delivery_turn_id is not None:
            await run_db_io_cancellation_safe(
                lambda: mark_user_message_delivery_sync(
                    task_id,
                    delivery_turn_id,
                    DELIVERY_COMPLETED,
                )
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
                "type": "task_completed",
                "metadata": result.get("metadata", {}),
                "timestamp": datetime.now(timezone.utc).timestamp(),
            },
            task_id,
        )
    except TaskLeaseLostError:
        defer_db_cleanup_to_ttl_recovery = lease is not None and not lease_released
        logger.warning(
            "Task %s resume execution cancelled after lease ownership loss",
            task_id,
        )
        return
    except asyncio.CancelledError:
        settlement_error = "resume execution cancelled"
        logger.info(f"V2 resume background task {task_id} cancelled")
        if delivery_turn_id is not None and not delivery_was_dispatched:
            if await mark_deferred_delivery_failed():
                await notify_deferred_delivery(
                    False,
                    client_error_message(ClientErrorCode.MESSAGE_DELIVERY_FAILED),
                    error_code=ClientErrorCode.MESSAGE_DELIVERY_FAILED,
                    retry_with_new_id=True,
                    rejection_outcome="not_accepted",
                )
        raise
    except Exception as e:
        error_message = str(e)
        if is_database_pool_timeout(e):
            # The failed operation already waited on an exhausted checkout.
            # Any delivery/status/settlement write here would immediately
            # request another connection. Keep the exact lease fenced until
            # TTL recovery and leave a pending delivery reclaimable.
            defer_db_cleanup_to_ttl_recovery = lease is not None and not lease_released
            recovery_action = (
                "retaining lease for TTL recovery"
                if defer_db_cleanup_to_ttl_recovery
                else "leaving durable state for retry"
            )
            logger.error(
                "task_id=%s component=resume database pool checkout timed out; "
                "skipping immediate DB cleanup and %s: %s",
                task_id,
                recovery_action,
                e,
                exc_info=True,
            )
            # The durable state remains RUNNING under the exact lease (or is
            # otherwise left reclaimable when no lease was acquired). Do not
            # emit the generic FAILED/task_error payload below.
            return
        elif (
            isinstance(e, (CheckpointUnavailableError, CheckpointAccessRefusedError))
            and lease is not None
            and not lease_released
            and resume_prior_status in {TaskStatus.PAUSED, TaskStatus.WAITING_FOR_USER}
        ):
            # Not pool exhaustion (handled above) but still a read that
            # could not be completed, or a partition this reader was not
            # authoritative for -- retryable, not a policy decision this
            # task made. Restore it to whatever it was before this resume
            # attempt claimed the lease instead of a terminal FAILED; the
            # finally block below performs the actual write once the
            # heartbeat has stopped. RUNNING is never a restore target:
            # ``release_task_lease_no_commit`` refuses to release a lease
            # back to RUNNING, so a prior status of RUNNING (an abandoned
            # lease this attempt stole via TTL expiry) falls through to the
            # settle/FAILED branch below instead of dead-ending here.
            restore_lease_to_prior_status = resume_prior_status
            logger.error(
                "task_id=%s component=resume checkpoint could not be read; "
                "restoring prior status %s: %s",
                task_id,
                resume_prior_status.value,
                e,
                exc_info=True,
            )
            if delivery_turn_id is not None and not delivery_was_dispatched:
                if await mark_deferred_delivery_failed():
                    await notify_deferred_delivery(
                        False,
                        client_error_message(ClientErrorCode.MESSAGE_DELIVERY_FAILED),
                        error_code=ClientErrorCode.MESSAGE_DELIVERY_FAILED,
                        retry_with_new_id=True,
                        rejection_outcome="not_accepted",
                    )
        else:
            logger.error(
                "V2 resume background task %s failed: %s",
                task_id,
                e,
                exc_info=True,
            )
            settlement_error = error_message
            broadcast_error_message = client_safe_error_message(
                e,
                fallback=CLIENT_SAFE_TASK_FAILURE,
            )
            if delivery_turn_id is not None and not delivery_was_dispatched:
                if await mark_deferred_delivery_failed():
                    await notify_deferred_delivery(
                        False,
                        CLIENT_SAFE_VALIDATION_ERROR,
                        retry_with_new_id=True,
                        rejection_outcome="not_accepted",
                    )
            current_snapshot = None
            if (
                lease is None
                and not defer_db_cleanup_to_ttl_recovery
                and expected_run_id is not None
            ):
                current_snapshot = await task_execution_controller.snapshot(task_id)
            if (
                current_snapshot is not None
                and current_snapshot.run_id != expected_run_id
            ):
                logger.info(
                    "Suppressing late resume error for task %s run %s; "
                    "current run is %s",
                    task_id,
                    expected_run_id,
                    current_snapshot.run_id,
                )
                return
        if lease is None:
            if broadcast_error_message is not None:
                await manager.broadcast_to_task(
                    create_terminal_task_error_event(
                        task_id,
                        broadcast_error_message,
                    ),
                    task_id,
                )
            else:
                await manager.broadcast_to_task(
                    create_terminal_task_error_event(
                        task_id,
                        CLIENT_SAFE_TASK_FAILURE,
                    ),
                    task_id,
                )
    finally:

        async def finalize_resume_resources() -> None:
            nonlocal defer_db_cleanup_to_ttl_recovery, lease_released
            nonlocal prepared_outputs

            try:
                # Finalize any tracker that did not reach the normal completion
                # point, then release any unfinished lease through worker-owned
                # short Sessions.
                if resume_tracker is not None:
                    agent_service.set_interrupt_checker(None)
                    try:
                        if defer_db_cleanup_to_ttl_recovery:
                            # Do not initiate a final usage checkout immediately
                            # after a pool timeout. Stop/drain only the periodic
                            # loop.
                            await resume_tracker.stop_periodic_updates()
                        else:
                            await resume_tracker.complete_tracking()
                    except Exception as e:  # noqa: BLE001
                        if is_database_pool_timeout(e):
                            defer_db_cleanup_to_ttl_recovery = (
                                lease is not None and not lease_released
                            )
                            logger.error(
                                "task_id=%s component=resume-tracker database "
                                "pool checkout timed out; retaining lease for "
                                "TTL recovery: %s",
                                task_id,
                                e,
                                exc_info=True,
                            )
                        else:
                            logger.warning(
                                "execute_resume_background: token tracking "
                                "completion failed for task %s: %s",
                                task_id,
                                e,
                            )
                if lease_heartbeat_task is not None or lease_stop_event is not None:
                    try:
                        heartbeat_outcome = await stop_task_lease_heartbeat(
                            lease_heartbeat_task,
                            lease_stop_event,
                        )
                        if (
                            isinstance(heartbeat_outcome, TaskLeaseHeartbeatOutcome)
                            and heartbeat_outcome.requires_ttl_recovery
                        ):
                            defer_db_cleanup_to_ttl_recovery = (
                                lease is not None and not lease_released
                            )
                            logger.error(
                                "task_id=%s component=resume-heartbeat unhealthy "
                                "during cleanup; retaining lease for TTL "
                                "recovery (lost=%s, pool_timeout=%s)",
                                task_id,
                                heartbeat_outcome.lease_lost,
                                heartbeat_outcome.pool_timeout is not None,
                            )
                    except Exception:
                        logger.warning(
                            "resume heartbeat shutdown failed for task %s",
                            task_id,
                            exc_info=True,
                        )
                if prepared_outputs is not None:
                    outputs_to_compensate = prepared_outputs
                    prepared_outputs = None
                    await run_db_io_cancellation_safe(
                        lambda: _settle_prepared_task_file_outputs(
                            outputs_to_compensate,
                            metadata_committed=False,
                        )
                    )
                if (
                    lease is not None
                    and not lease_released
                    and not defer_db_cleanup_to_ttl_recovery
                    and restore_lease_to_prior_status is not None
                ):
                    try:
                        restored = await run_db_io_cancellation_safe(
                            lambda: _restore_resumed_task_lease_to_prior_status(
                                lease,
                                status=restore_lease_to_prior_status,
                            )
                        )
                        if restored:
                            lease_released = True
                    except Exception:
                        logger.error(
                            "resume lease restore-to-prior-status failed for "
                            "task %s; retaining lease for TTL recovery",
                            task_id,
                            exc_info=True,
                        )
                    else:
                        if restored:
                            # Correct the optimistic RUNNING state a client
                            # may still be showing after the lease claim
                            # above flipped it, before this failure restored
                            # the prior status. Best-effort: a missed
                            # broadcast does not change the restore result
                            # that already committed.
                            try:
                                restored_snapshot = (
                                    await task_execution_controller.snapshot(task_id)
                                )
                                event_type, message = _waiting_or_paused_event_fields(
                                    restore_lease_to_prior_status
                                )
                                await manager.broadcast_to_task(
                                    {
                                        "task_id": task_id,
                                        "message": message,
                                        "timestamp": datetime.now(
                                            timezone.utc
                                        ).timestamp(),
                                        **(
                                            restored_snapshot.as_dict()
                                            if restored_snapshot is not None
                                            else {}
                                        ),
                                        "type": event_type,
                                    },
                                    task_id,
                                )
                            except Exception:
                                logger.warning(
                                    "resume lease restore-to-prior-status "
                                    "broadcast failed for task %s",
                                    task_id,
                                    exc_info=True,
                                )
                        else:
                            logger.warning(
                                "task_id=%s component=resume restore to prior "
                                "status %s affected no rows; the task row no "
                                "longer matches this lease fence (runner_id=%s "
                                "run_id=%s), so another releaser now owns its "
                                "status",
                                task_id,
                                restore_lease_to_prior_status.value,
                                lease.runner_id,
                                lease.run_id,
                            )
                elif (
                    lease is not None
                    and not lease_released
                    and not defer_db_cleanup_to_ttl_recovery
                ):
                    try:
                        settled = await run_db_io_cancellation_safe(
                            lambda: _settle_resumed_task_lease(
                                lease,
                                error_message=settlement_error,
                            )
                        )
                        if settled:
                            lease_released = True
                            if broadcast_error_message is not None:
                                try:
                                    await manager.broadcast_to_task(
                                        create_terminal_task_error_event(
                                            task_id,
                                            broadcast_error_message,
                                        ),
                                        task_id,
                                    )
                                except Exception:
                                    logger.warning(
                                        "task %s resume failure was committed but "
                                        "its terminal broadcast failed",
                                        task_id,
                                        exc_info=True,
                                    )
                    except Exception:
                        logger.error(
                            "resume lease settlement failed for task %s; "
                            "retaining lease for TTL recovery",
                            task_id,
                            exc_info=True,
                        )
            finally:
                _clear_task_pause_accepted(task_id)
                background_task_manager.cleanup_task(
                    task_id,
                    expected_task=resume_owner_task,
                )

        cleanup_task = asyncio.create_task(finalize_resume_resources())
        await drain_async_task_cancellation_safe(cleanup_task)


@dataclass(frozen=True)
class BackgroundTaskCancelOutcome:
    """Whether cancellation was requested from live process-local task work."""

    requested: bool


class ResumeReservationOutcome(str, enum.Enum):
    """Result of trying to take the single live-control resume slot."""

    RESERVED = "reserved"
    # Another caller owns the pre-registration window. Its transition or
    # coordinator registration may still fail, so this is not yet proof that
    # the task is resuming.
    RESERVATION_HELD = "reservation_held"
    # A registered resume coordinator is already responsible for the task.
    COORDINATOR_RUNNING = "coordinator_running"
    # This process no longer admits new background work.
    SHUTTING_DOWN = "shutting_down"


class AnyResumeRun:
    """Marker admitting a coordinator for *any* run as idempotency evidence.

    Distinguishes "do not check the run" from an explicit ``None``, which
    means "this task has no run id, so only a coordinator registered without
    one is evidence". Without the marker a caller that simply had no run id
    to hand would silently accept a coordinator belonging to a different run.
    """

    __slots__ = ()


ANY_RESUME_RUN = AnyResumeRun()


class ResumeCommandOutcome(str, enum.Enum):
    """Durable meaning of one handled RESUME command."""

    SCHEDULED = "scheduled"
    ALREADY_IN_PROGRESS = "already_in_progress"
    DEFERRED = "deferred"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ResumeCommandResult:
    outcome: ResumeCommandOutcome
    # Human-readable text. Lands in the command row's ``error`` column and,
    # for deferrals, in the message a budget exhaustion reports.
    reason: str | None = None
    # Stable machine-readable code, mirroring the ``stale_run`` code the
    # CANCEL branch already emits. Populates ``result["rejection_reason"]``
    # so a client can branch without matching human-readable text.
    reason_code: str | None = None
    # Whether ``reason`` is wording this module wrote for the sender. Terminal
    # deferral broadcasts go through the redaction chokepoint, so without this
    # the text is replaced by the generic string and the deferral becomes
    # indistinguishable from an outright failure -- see
    # ``ClientVisibleTaskCommandDeferred``.
    client_visible: bool = False


# Background task manager: ensures only one active background execution per task
class BackgroundTaskManager:
    """Manages background task execution, ensuring only one background process per task at a time"""

    def __init__(self) -> None:
        # task_id -> asyncio.Task
        self.running_tasks: dict[int, asyncio.Task] = {}
        # Resume coordinators are deliberately tracked separately while they
        # wait for the current execution. Replacing ``running_tasks[task_id]``
        # too early creates a cycle: the original execution waits for the new
        # resume task while that resume task waits for the original execution.
        self.resume_tasks: dict[int, asyncio.Task] = {}
        # The coordinator is evidence only for the exact run it was created
        # to resume. A lingering old-run task must not complete a command for
        # a newer run as an idempotent success.
        self._resume_run_ids: dict[int, str | None] = {}
        self._resume_reservations: set[int] = set()
        self._resume_owner_started_at: dict[int, float] = {}
        self._shutting_down = False
        self._shutdown_lock = asyncio.Lock()

    def start_accepting(self) -> None:
        """Reopen admission for a new application lifespan."""

        if (
            self._shutdown_lock.locked()
            or self.running_tasks
            or self.resume_tasks
            or self._resume_run_ids
            or self._resume_reservations
            or self._resume_owner_started_at
        ):
            raise RuntimeError("Background task manager still owns background work")
        # asyncio synchronization primitives are bound to the event loop that
        # first contends on them. A new application lifespan may use a new loop,
        # so an idle manager must not retain the previous lifespan's lock.
        self._shutdown_lock = asyncio.Lock()
        self._shutting_down = False

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
                    await asyncio.shield(old_task)
                    logger.info(f"Previous background task {task_id} completed")
                except Exception as e:
                    logger.warning(
                        f"Previous background task {task_id} ended with error: {e}"
                    )

    def register_task(self, task_id: int, task: asyncio.Task) -> None:
        """Register new background task"""
        if self._shutting_down:
            task.cancel()
            raise RuntimeError("Background task manager is shutting down")
        self.running_tasks[task_id] = task
        logger.info(f"Registered background task for task {task_id}")

    def resume_admission_state(
        self,
        task_id: int,
        *,
        expected_run_id: str | None | AnyResumeRun,
    ) -> ResumeReservationOutcome | None:
        """Classify existing resume ownership without taking an empty slot.

        Returns ``None`` when the slot is free. Not a pure read: a coordinator
        that has already finished is reclaimed here, dropping both its task
        and its registered run id, so a finished registration never reports
        the slot as occupied.

        ``expected_run_id`` is the evidence axis. Pass :data:`ANY_RESUME_RUN`
        to accept a coordinator for any run; an explicit ``None`` means the
        task has no run id and only a coordinator registered without one
        counts.
        """

        if self._shutting_down:
            return ResumeReservationOutcome.SHUTTING_DOWN
        if task_id in self._resume_reservations:
            return ResumeReservationOutcome.RESERVATION_HELD
        existing = self.resume_tasks.get(task_id)
        if existing is not None and not existing.done():
            registered_run_id = self._resume_run_ids.get(task_id)
            if (
                isinstance(expected_run_id, AnyResumeRun)
                or registered_run_id == expected_run_id
            ):
                return ResumeReservationOutcome.COORDINATOR_RUNNING
            # The task id is still locally occupied, but by a coordinator for
            # another run. It is not evidence that this run is resuming and it
            # is not safe to overwrite its registration.
            return ResumeReservationOutcome.RESERVATION_HELD
        if existing is not None:
            self.resume_tasks.pop(task_id, None)
            self._resume_run_ids.pop(task_id, None)
            self._resume_owner_started_at.pop(task_id, None)
        return None

    def resume_holder_age_seconds(self, task_id: int) -> float | None:
        """Return the local resume-slot holder's monotonic age, if known."""

        started_at = self._resume_owner_started_at.get(task_id)
        if started_at is None:
            return None
        return max(0.0, time.monotonic() - started_at)

    def try_reserve_resume(
        self,
        task_id: int,
        *,
        expected_run_id: str | None | AnyResumeRun,
    ) -> ResumeReservationOutcome:
        """Atomically classify admission to the live-control resume slot."""

        # Keep this inspect-and-add block synchronous: asyncio task switches
        # can only happen at ``await``, so it is the in-process atomic guard.
        existing_state = self.resume_admission_state(
            task_id,
            expected_run_id=expected_run_id,
        )
        if existing_state is not None:
            return existing_state
        self._resume_reservations.add(task_id)
        self._resume_owner_started_at[task_id] = time.monotonic()
        return ResumeReservationOutcome.RESERVED

    def reserve_resume(self, task_id: int) -> bool:
        """Boolean compatibility wrapper for callers that cannot classify.

        Keeps the pre-classification contract: any unfinished coordinator
        reports the slot as taken, whichever run it belongs to.
        """

        return (
            self.try_reserve_resume(task_id, expected_run_id=ANY_RESUME_RUN)
            is ResumeReservationOutcome.RESERVED
        )

    def register_reserved_resume(
        self,
        task_id: int,
        task: asyncio.Task,
        *,
        run_id: str | None,
    ) -> None:
        if self._shutting_down:
            task.cancel()
            raise RuntimeError("Background task manager is shutting down")
        if task_id not in self._resume_reservations:
            raise RuntimeError(f"Task {task_id} has no reserved resume slot")
        self._resume_reservations.discard(task_id)
        self._resume_owner_started_at.setdefault(task_id, time.monotonic())
        self.resume_tasks[task_id] = task
        self._resume_run_ids[task_id] = run_id
        logger.info("Registered resume coordinator for task %s", task_id)

    def release_resume_reservation(self, task_id: int) -> None:
        if self._shutting_down:
            return
        self._resume_reservations.discard(task_id)
        self._resume_owner_started_at.pop(task_id, None)

    def promote_resume_task(self, task_id: int, task: asyncio.Task) -> None:
        if self._shutting_down:
            raise RuntimeError("Background task manager is shutting down")
        existing = self.resume_tasks.get(task_id)
        if existing is not task:
            raise RuntimeError(
                f"Task {task_id} resume coordinator is not registered or no longer current"
            )
        self.running_tasks[task_id] = task
        logger.info("Promoted resume coordinator for task %s", task_id)

    def cleanup_task(
        self,
        task_id: int,
        *,
        expected_task: asyncio.Task | None = None,
    ) -> None:
        """Clean up completed background task"""
        if self._shutting_down:
            return
        current = expected_task or asyncio.current_task()

        def owns_registration(task: asyncio.Task) -> bool:
            if expected_task is not None:
                return task is expected_task
            return task.done() or task is current

        task = self.running_tasks.get(task_id)
        if task is not None and owns_registration(task):
            self.running_tasks.pop(task_id, None)
            logger.info(f"Cleaned up background task for task {task_id}")
        resume_task = self.resume_tasks.get(task_id)
        if resume_task is not None and owns_registration(resume_task):
            self.resume_tasks.pop(task_id, None)
            self._resume_run_ids.pop(task_id, None)
            self._resume_owner_started_at.pop(task_id, None)
            logger.info("Cleaned up resume coordinator for task %s", task_id)

    async def cancel_task(
        self,
        task_id: int,
        timeout_seconds: float = 0.5,
    ) -> BackgroundTaskCancelOutcome:
        tasks = {
            task
            for task in (
                self.running_tasks.get(task_id),
                self.resume_tasks.get(task_id),
            )
            if task is not None
        }
        if not self._shutting_down:
            # A cancel can race the await between reservation and coordinator
            # registration. Clear that pre-registration owner even when there
            # is no asyncio task to cancel yet.
            self._resume_reservations.discard(task_id)
            self._resume_owner_started_at.pop(task_id, None)
        if not tasks:
            return BackgroundTaskCancelOutcome(requested=False)

        requested = False
        for task in tasks:
            if task.done():
                continue
            requested = task.cancel() or requested
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

        if not self._shutting_down:
            self.running_tasks.pop(task_id, None)
            self.resume_tasks.pop(task_id, None)
            self._resume_run_ids.pop(task_id, None)
            self._resume_owner_started_at.pop(task_id, None)
        return BackgroundTaskCancelOutcome(requested=requested)

    async def shutdown(self) -> None:
        """Fence new work, cancel every owned task, and drain its cleanup."""

        self._shutting_down = True
        async with self._shutdown_lock:
            current = asyncio.current_task()
            tasks = {
                task
                for task in (*self.running_tasks.values(), *self.resume_tasks.values())
                if task is not current
            }
            for task in tasks:
                if not task.done():
                    task.cancel()

            async def drain_tasks() -> None:
                await asyncio.gather(*tasks, return_exceptions=True)

            cleanup_task = asyncio.create_task(drain_tasks())
            try:
                await drain_async_task_cancellation_safe(cleanup_task)
            finally:
                # ``drain_async_task_cancellation_safe`` reaches this block only
                # after the owned cleanup task has settled, even when shutdown's
                # caller is cancelled.
                self.running_tasks.clear()
                self.resume_tasks.clear()
                self._resume_run_ids.clear()
                self._resume_reservations.clear()
                self._resume_owner_started_at.clear()


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


def _with_task_control_state_snapshot(
    message: dict[str, Any],
    *,
    task_id: int,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Attach one already-loaded control-state tuple without database I/O."""

    if not _is_versioned_task_event(message):
        return deepcopy(message)
    resolved_state = _event_task_control_state(message) or state
    enriched = deepcopy(message)
    enriched.update(resolved_state)
    enriched["task_id"] = task_id

    if enriched.get("type") == "trace_event":
        data = enriched.get("data")
        enriched["data"] = {
            **(data if isinstance(data, dict) else {}),
            **resolved_state,
        }

    task_data = enriched.get("task")
    if isinstance(task_data, dict):
        enriched["task"] = {
            **task_data,
            **resolved_state,
            "id": task_id,
        }
    return enriched


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
    return _with_task_control_state_snapshot(
        message,
        task_id=task_id,
        state=state,
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

    def _remove_from_task(self, websocket: WebSocket, task_id: int) -> None:
        if task_id in self.active_connections:
            try:
                self.active_connections[task_id].remove(websocket)
                if not self.active_connections[task_id]:
                    del self.active_connections[task_id]
            except ValueError:
                pass

    def disconnect(self, websocket: WebSocket) -> None:
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
            _command_origins.discard_socket(connection)
        return connections

    def connections_for_task(self, task_id: int) -> List[WebSocket]:
        """Return a stable snapshot of a task's current connections."""
        return self.active_connections.get(task_id, []).copy()

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
        await websocket.send_text(json.dumps(versioned_message))

    async def broadcast_to_task(self, message: dict, task_id: int) -> None:
        if self.connections_for_task(task_id):
            versioned_message = await _with_current_task_control_state(
                message,
                fallback_task_id=task_id,
            )
            for connection in self.connections_for_task(task_id):
                if not self.is_connection_registered(connection, task_id):
                    continue
                try:
                    await connection.send_text(json.dumps(versioned_message))
                except (
                    BrokenResourceError,
                    ClosedResourceError,
                    ConnectionError,
                    WebSocketDisconnect,
                    RuntimeError,
                ) as e:
                    # Network connection error, remove disconnected connection
                    logger.warning(f"Connection error for task {task_id}: {e}")
                    self.disconnect(connection)
                except Exception as e:
                    # Other errors should not be silently handled, log and re-raise
                    logger.error(
                        f"Unexpected error broadcasting to task {task_id}: {e}"
                    )
                    # Remove disconnected connection but preserve error propagation
                    self.disconnect(connection)
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


@dataclass(frozen=True)
class _UserMessageDeliverySnapshot:
    """Primitive delivery result safe to carry outside its DB Session."""

    claimed: bool
    payload_matches: bool
    failed: bool
    pending: bool


class _WebSocketCommitOutcomeUnknown(RuntimeError):
    """A WebSocket acceptance COMMIT may still be visible to a later retry."""


def _snapshot_user_message_delivery(
    claim: UserMessageDeliveryClaim,
) -> _UserMessageDeliverySnapshot:
    return _UserMessageDeliverySnapshot(
        claimed=bool(claim.claimed),
        payload_matches=bool(claim.payload_matches),
        failed=bool(claim.failed),
        pending=bool(claim.pending),
    )


def _retire_websocket_session_best_effort(
    db: Session,
    *,
    task_id: int,
) -> None:
    """Release an owned Session without replacing its primary error."""

    try:
        db.close()
        return
    except Exception:
        logger.warning(
            "failed to close websocket turn session for task %s",
            task_id,
            exc_info=True,
        )
    try:
        db.invalidate()
    except Exception:
        logger.warning(
            "failed to invalidate websocket turn session for task %s",
            task_id,
            exc_info=True,
        )


@contextmanager
def _owned_websocket_session(*, task_id: int) -> Iterator[Session]:
    SessionLocal = get_session_local()
    resource = SessionLocal()
    enter = getattr(resource, "__enter__", None)
    exit_context = getattr(resource, "__exit__", None)
    db = enter() if callable(enter) else resource
    try:
        yield db
    finally:
        if callable(exit_context):
            try:
                exit_context(None, None, None)
            except Exception:
                _retire_websocket_session_best_effort(db, task_id=task_id)
        else:
            _retire_websocket_session_best_effort(db, task_id=task_id)


def _reconcile_websocket_acceptance_graph(
    *,
    task_id: int,
    task_owner_user_id: int,
    turn_id: str,
    content: str,
    file_ids: list[str],
    expected_run_id: str | None,
    expected_status: TaskStatus,
) -> bool:
    """Boundedly inspect an ambiguous acceptance COMMIT via fresh Sessions."""

    for attempt in range(3):
        reconcile_db: Session | None = None
        try:
            SessionLocal = get_session_local()
            reconcile_db = SessionLocal()
            task_query = reconcile_db.query(Task).filter(
                Task.id == task_id,
                Task.user_id == task_owner_user_id,
                Task.status == expected_status,
                Task.run_id == expected_run_id,
            )
            if task_query.first() is None:
                pass
            else:
                message = (
                    reconcile_db.query(TaskChatMessage)
                    .filter(
                        TaskChatMessage.task_id == task_id,
                        TaskChatMessage.role == "user",
                        TaskChatMessage.turn_id == turn_id,
                        TaskChatMessage.content == content.strip(),
                        TaskChatMessage.delivery_status.in_(
                            (
                                DELIVERY_PENDING,
                                DELIVERY_DISPATCHED,
                                DELIVERY_COMPLETED,
                            )
                        ),
                    )
                    .first()
                )
                if message is not None:
                    if not file_ids:
                        return True
                    bound = (
                        reconcile_db.query(UploadedFile.file_id)
                        .filter(
                            UploadedFile.file_id.in_(file_ids),
                            UploadedFile.user_id == task_owner_user_id,
                            UploadedFile.task_id == task_id,
                        )
                        .count()
                    )
                    if bound == len(set(file_ids)):
                        return True
        except Exception:
            logger.warning(
                "websocket commit reconciliation attempt %s failed for task %s",
                attempt + 1,
                task_id,
                exc_info=True,
            )
        finally:
            if reconcile_db is not None:
                _retire_websocket_session_best_effort(
                    reconcile_db,
                    task_id=task_id,
                )
        if attempt < 2:
            time.sleep(0.01)
    return False


def _claim_user_message_delivery_isolated(
    *,
    task_id: int,
    task_owner_user_id: int,
    content: str,
    attachments: list[dict[str, Any]] | None,
    file_ids: list[str],
    turn_id: str,
    expected_run_id: str | None,
    expected_status: TaskStatus,
) -> _UserMessageDeliverySnapshot:
    """Claim a live-control message in one worker-owned short Session."""

    with _owned_websocket_session(task_id=task_id) as db:
        try:
            try:
                claim = claim_user_message_delivery_no_commit(
                    db,
                    task_id=task_id,
                    user_id=task_owner_user_id,
                    content=content,
                    attachments=attachments,
                    turn_id=turn_id,
                )
            except IntegrityError:
                db.rollback()
                _retire_websocket_session_best_effort(db, task_id=task_id)
                with _owned_websocket_session(task_id=task_id) as winner_db:
                    winner = inspect_user_message_delivery(
                        winner_db,
                        task_id,
                        content,
                        attachments=attachments,
                        turn_id=turn_id,
                    )
                    if winner is None:
                        raise
                    if winner.payload_matches and file_ids:
                        bound_count = (
                            winner_db.query(UploadedFile.file_id)
                            .filter(
                                UploadedFile.file_id.in_(file_ids),
                                UploadedFile.user_id == task_owner_user_id,
                                UploadedFile.task_id == task_id,
                            )
                            .count()
                        )
                        if bound_count != len(set(file_ids)):
                            raise
                    return _snapshot_user_message_delivery(winner)
            if claim.claimed:
                missing = bind_turn_files_no_commit(
                    file_ids=file_ids,
                    task_id=task_id,
                    owner_user_id=task_owner_user_id,
                    db=db,
                )
                if missing:
                    raise ClientVisibleValidationError(
                        "Files are no longer bindable: " + ", ".join(missing),
                        error_code=ClientErrorCode.MESSAGE_ATTACHMENT_UNAVAILABLE,
                    )
            claim_snapshot = _snapshot_user_message_delivery(claim)
            try:
                db.flush()
                db.commit()
            except Exception as commit_error:
                _retire_websocket_session_best_effort(db, task_id=task_id)
                if not _reconcile_websocket_acceptance_graph(
                    task_id=task_id,
                    task_owner_user_id=task_owner_user_id,
                    turn_id=turn_id,
                    content=content,
                    file_ids=file_ids,
                    expected_run_id=expected_run_id,
                    expected_status=expected_status,
                ):
                    raise _WebSocketCommitOutcomeUnknown(
                        f"live delivery {turn_id} has an unknown commit outcome"
                    ) from commit_error
            return claim_snapshot
        except Exception:
            db.rollback()
            raise


def _register_uploaded_files_for_agent(
    agent_service: Any,
    file_info_list: List[Dict[str, Any]],
) -> None:
    """Bind already-durable inputs to the workspace without another upload."""

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

        registration = workspace.describe_file_registration(str(source_path.resolve()))
        workspace.bind_already_durable_file(
            registration,
            file_id=file_id,
        )
        file_info["path"] = str(registration.path)
        file_info["workspace_path"] = str(workspace_link_path)
        logger.info(
            "File registered for agent workspace: storage=%s input_link=%s",
            registration.path,
            workspace_link_path,
        )


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
    except (
        MCPBuiltinOAuthActorPolicyRequiredError,
        PermissionError,
        ValueError,
    ) as exc:
        log_client_facing_failure(exc, "Chat command rejected for task %s: %s", task_id)
        client_message_id = _client_message_id(message_data.get("client_message_id"))
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
        async with task_execution_controller.command(task_id):
            await _handle_chat_message_unserialized(websocket, task_id, message_data)
        return
    if not enqueued.payload_matches:
        await send_message_delivery(
            websocket,
            client_message_id=_client_message_id(message_data.get("client_message_id")),
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
            client_message_id=_client_message_id(message_data.get("client_message_id")),
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
        client_message_id=_client_message_id(message_data.get("client_message_id")),
        turn_id=enqueued.client_command_id,
        accepted=True,
    )
    if enqueued.command_id:
        if enqueued.created:
            # Only the ingress that created the durable row owns the origin.
            # A payload-matching duplicate (created=False) - a co-tenant
            # resubmission, one arriving after the creator disconnected, or one
            # handled on another worker - must never bind, or it could receive
            # the creator's raw error detail. Registered before dispatch so
            # local execution cannot outrun the binding.
            _command_origins.register(enqueued.client_command_id, websocket, task_id)
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
            raise ClientVisibleValidationError(
                f"Task {task_id} not found",
                error_code=ClientErrorCode.TASK_UNAVAILABLE,
            )
        if not actor_is_admin and int(task.user_id) != actor_user_id:
            # Keep the permission-specific text for operator logs, but expose
            # the same code as a missing task so task IDs cannot be probed.
            raise ClientVisiblePermissionError(
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
        # core (see ``_execute_durable_task_command``); only server-side
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


@dataclass(frozen=True)
class _WebSocketTaskRoutingSnapshot:
    """Detached task state used by WebSocket turn routing and presentation."""

    task_id: int
    task_owner_user_id: int
    status: TaskStatus
    control_state: str | None
    run_id: str | None
    task_lease: TaskLease | None
    task_input: str
    task_info: dict[str, Any]
    task_context: dict[str, Any]
    created_at: datetime | None


@dataclass(frozen=True)
class _WebSocketTurnPreparation:
    """All synchronous state needed before WebSocket turn orchestration.

    The preparation owner opens and closes its own Session in a worker thread.
    Only primitives and frozen application-layer values cross back to asyncio;
    no ORM row or Session may survive into a network wait, broadcast, agent
    construction, or turn claim.
    """

    requested_task_id: int
    routing: _WebSocketTaskRoutingSnapshot
    task_created: bool
    execution_context: dict[str, Any]
    user_message_for_llm: str
    display_user_message: str
    display_file_refs: tuple[dict[str, Any], ...]
    persisted_attachments: tuple[dict[str, Any], ...]
    turn_payload: "TaskTurnPayload"
    claimed_created_turn: "_ClaimedTurn | None"
    existing_delivery: _UserMessageDeliverySnapshot | None
    recovered_delivery: _UserMessageDeliverySnapshot | None
    delivery_claimed: bool
    delivery_dispatched: bool
    uses_live_control: bool


def _agent_builder_skill_enabled(skills: Any) -> bool:
    if isinstance(skills, list):
        return any(skill == "agent-builder" for skill in skills)
    return isinstance(skills, str) and "agent-builder" in skills


def _load_websocket_task_routing_snapshot(
    db: Session,
    task: Task,
) -> tuple[_WebSocketTaskRoutingSnapshot, bool]:
    """Project one authorized Task row without leaking ORM state."""

    from ..models.agent import Agent

    agent_name: str | None = None
    agent_logo_url: str | None = None
    agent_execution_mode: str | None = None
    agent_skills: Any = None
    if task.agent_id is not None:
        agent_fields = (
            db.query(
                Agent.name,
                Agent.logo_url,
                Agent.execution_mode,
                Agent.skills,
            )
            .filter(Agent.id == task.agent_id)
            .first()
        )
        if agent_fields is not None:
            agent_name = str(agent_fields[0]) if agent_fields[0] is not None else None
            agent_logo_url = (
                str(agent_fields[1]) if agent_fields[1] is not None else None
            )
            agent_execution_mode = (
                str(agent_fields[2]) if agent_fields[2] is not None else None
            )
            agent_skills = deepcopy(agent_fields[3])

    (
        model_id,
        small_fast_model_id,
        visual_model_id,
        compact_model_id,
    ) = _resolve_task_llm_ids(task, db)

    task_context: dict[str, Any] = {}
    if task.execution_mode:
        task_context["execution_mode"] = str(task.execution_mode)
    if task.process_description:
        task_context["process_description"] = str(task.process_description)
    if task.examples:
        task_context["examples"] = deepcopy(task.examples)

    created_at = cast(datetime | None, task.created_at)
    status = cast(TaskStatus, task.status)
    return (
        _WebSocketTaskRoutingSnapshot(
            task_id=int(task.id),
            task_owner_user_id=int(task.user_id),
            status=status,
            control_state=_task_control_state_value(task),
            run_id=_task_run_id(task),
            task_lease=_task_lease_snapshot(task),
            task_input=str(task.input or ""),
            task_info={
                "id": int(task.id),
                "title": task.title,
                "description": task.description,
                "status": status.value,
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
                "is_dag": (
                    agent_execution_mode == "think"
                    if agent_execution_mode is not None
                    else None
                ),
                "created_at": (
                    safe_timestamp_to_unix(task.created_at) if task.created_at else None
                ),
                "updated_at": (
                    safe_timestamp_to_unix(task.updated_at) if task.updated_at else None
                ),
            },
            task_context=task_context,
            created_at=created_at,
        ),
        _agent_builder_skill_enabled(agent_skills),
    )


def _load_websocket_task_routing_snapshot_sync(
    task_id: int,
    *,
    task_owner_user_id: int,
    actor_user_id: int,
    actor_is_admin: bool,
) -> _WebSocketTaskRoutingSnapshot | None:
    """Reload routing state after an async wait in a fresh worker Session."""

    if not actor_is_admin and actor_user_id != task_owner_user_id:
        return None
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = (
            db.query(Task)
            .filter(
                Task.id == task_id,
                Task.user_id == task_owner_user_id,
            )
            .first()
        )
        if task is None:
            return None
        routing, _is_agent_builder = _load_websocket_task_routing_snapshot(db, task)
        return routing


def _recover_recent_websocket_file_refs(
    db: Session,
    *,
    task_id: int,
    actor_user_id: int,
) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=5)
    pending = (
        db.query(UploadedFile)
        .filter(
            UploadedFile.user_id == actor_user_id,
            UploadedFile.task_id == task_id,
            UploadedFile.created_at >= cutoff,
        )
        .order_by(UploadedFile.created_at.desc())
        .all()
    )
    return [_uploaded_file_ref(record) for record in pending]


def _prepare_websocket_turn_sync(
    *,
    requested_task_id: int,
    actor_user_id: int,
    actor_is_admin: bool,
    user_message: str,
    raw_context: dict[str, Any],
    raw_files: list[dict[str, Any]],
    client_message_id: str | None,
    turn_id: str,
    durable_attempt_count: int,
    durable_target_run_id: str | None,
    pause_accepted: bool,
) -> _WebSocketTurnPreparation:
    """Authorize, normalize, and detach one WebSocket turn off the event loop."""

    with _owned_websocket_session(task_id=requested_task_id) as db:
        files = deepcopy(raw_files)
        if not files:
            try:
                files = _recover_recent_websocket_file_refs(
                    db,
                    task_id=requested_task_id,
                    actor_user_id=actor_user_id,
                )
                if files:
                    logger.info(
                        "📁 Race fallback: recovered %s uploaded file(s) from DB "
                        "for task %s",
                        len(files),
                        requested_task_id,
                    )
            except Exception as error:  # noqa: BLE001
                logger.warning(
                    "Race fallback file lookup failed for task %s: %s",
                    requested_task_id,
                    error,
                )

        task_query = db.query(Task).filter(Task.id == requested_task_id)
        if not actor_is_admin:
            task_query = task_query.filter(Task.user_id == actor_user_id)
        task = task_query.first()
        task_created = False
        if task is None:
            existing_task = db.query(Task).filter(Task.id == requested_task_id).first()
            if existing_task is not None:
                logger.warning(
                    "User %s attempted to access task %s belonging to user %s",
                    actor_user_id,
                    requested_task_id,
                    existing_task.user_id,
                )
                # Match the public opacity contract at command enqueue above.
                raise ClientVisiblePermissionError(
                    f"Access denied: Task {requested_task_id} does not belong to you",
                    error_code=ClientErrorCode.TASK_UNAVAILABLE,
                )
            task_created = True

        if task is not None and not files and task.status == TaskStatus.PENDING:
            files = _selected_file_refs_from_task(task, db)
            if files:
                logger.info(
                    "📁 Recovered %s selected file(s) from task %s for initial "
                    "chat turn",
                    len(files),
                    task.id,
                )

        routing: _WebSocketTaskRoutingSnapshot | None = None
        if task is not None:
            routing, is_agent_builder = _load_websocket_task_routing_snapshot(db, task)
            file_owner_user_id = routing.task_owner_user_id
            file_task_id: int | None = routing.task_id
        else:
            # A missing task has no persisted execution scope or binding yet.
            # Resolve and materialize its unbound uploads while this Session is
            # still read-only; only then create and claim the task atomically.
            is_agent_builder = False
            file_owner_user_id = actor_user_id
            file_task_id = None
        logger.info("📁 Files used for execution: %s", len(files))
        for index, file_ref in enumerate(files):
            logger.info(
                "📄 File %s: %s (%s bytes)",
                index,
                file_ref.get("name", "unknown"),
                file_ref.get("size", 0),
            )

        file_info_list: list[dict[str, Any]] = []
        execution_context = deepcopy(raw_context)
        if files:
            file_ids = [
                str(file_ref.get("file_id"))
                for file_ref in files
                if file_ref.get("file_id")
            ]
            file_info_list, missing = resolve_turn_file_infos(
                file_ids=file_ids,
                owner_user_id=file_owner_user_id,
                db=db,
                task_id=file_task_id,
            )
            for missing_id in missing:
                logger.warning(
                    "File record not accessible for task %s: %s",
                    requested_task_id,
                    missing_id,
                )
        if task_created:
            task_title = f"Chat: {user_message}"
            if len(task_title) > 50:
                task_title = task_title[:50] + "..."
            task = Task(
                user_id=actor_user_id,
                title=task_title,
                description=user_message,
                status=TaskStatus.PENDING,
                execution_mode=get_default_task_execution_mode(),
                connector_runtime_selected_refs=[],
            )
            db.add(task)
            db.flush()
            assert task is not None
            routing, is_agent_builder = _load_websocket_task_routing_snapshot(db, task)

        assert routing is not None
        uploaded_files_context = _build_uploaded_files_context(
            file_info_list,
            is_agent_builder=is_agent_builder,
        )
        if file_info_list:
            uploaded_file_paths = [
                str(file_info["path"]) for file_info in file_info_list
            ]
            execution_context["uploaded_files"] = uploaded_file_paths
            execution_context["file_info"] = deepcopy(file_info_list)
            file_ids = [str(file_info["file_id"]) for file_info in file_info_list]
            file_names = [file_info["name"] for file_info in file_info_list]
            file_id_list_str = ", ".join(f'"{file_id}"' for file_id in file_ids)
            file_prompt = (
                "## UPLOADED FILES\n"
                f"The user has uploaded {len(file_info_list)} file(s): "
                f"{file_names}\n\n"
                f"{FILE_REF_MODEL_INSTRUCTIONS}\n\n"
            )
            if is_agent_builder:
                file_prompt += (
                    "Use these exact file_ids (UUIDs) with "
                    "`create_knowledge_base_from_file`:\n"
                    f"  file_ids = [{file_id_list_str}]\n\n"
                    "IMPORTANT: The file_ids above are UUIDs (e.g. "
                    "'5d983e39-a83b-...'). Do NOT use file paths as file_ids. "
                    "Call `create_knowledge_base_from_file` with the file_ids "
                    "listed above, then create or update the agent with the "
                    "returned collection_name. Do NOT generate a 'wait for "
                    "upload' step — the files are already uploaded."
                )
            else:
                file_prompt += (
                    "These files have been successfully uploaded to the workspace "
                    "and are ready for processing.\nYou can use standard workspace "
                    "tools to read, analyze, or process them."
                )
            existing_prompt = execution_context.get("system_prompt")
            execution_context["system_prompt"] = (
                f"{existing_prompt}\n\n{file_prompt}"
                if existing_prompt
                else file_prompt
            )

        user_message_for_llm = _append_uploaded_files_context_to_message(
            user_message,
            uploaded_files_context,
        )
        display_user_message = _display_message_for_user(
            user_message,
            bool(file_info_list),
        )
        display_file_refs = _display_file_refs_from_file_info(file_info_list)
        execution_context["display_message"] = display_user_message
        execution_context["files"] = deepcopy(display_file_refs)
        persisted_attachments = _normalize_attachments_for_persistence(file_info_list)

        from ..services.task_orchestrator import (
            TaskTurnOrchestrator,
            TaskTurnPayload,
        )

        turn_payload = TaskTurnPayload(
            transcript_message=display_user_message,
            execution_message=user_message_for_llm,
            attachments=deepcopy(persisted_attachments) or None,
            file_ids=tuple(str(file_info["file_id"]) for file_info in file_info_list),
            turn_id=turn_id,
        )
        claimed_created_turn = None
        existing_delivery_snapshot: _UserMessageDeliverySnapshot | None = None
        recovered_delivery: _UserMessageDeliverySnapshot | None = None
        delivery_claimed = False
        delivery_dispatched = False
        if task_created:
            assert task is not None
            claimed_created_turn = TaskTurnOrchestrator.claim_created_turn_no_commit(
                db,
                task_id=routing.task_id,
                task_owner_user_id=routing.task_owner_user_id,
                payload=turn_payload,
            )
            # The atomic claim uses a bulk UPDATE. Refresh the Task before
            # projecting the detached routing snapshot so the first
            # task_info event reflects the committed RUNNING lease.
            db.expire(task)
            db.refresh(task)
            routing, _is_agent_builder = _load_websocket_task_routing_snapshot(
                db,
                task,
            )
            try:
                db.flush()
                db.commit()
            except Exception as commit_error:
                _retire_websocket_session_best_effort(
                    db,
                    task_id=routing.task_id,
                )
                if not _reconcile_websocket_acceptance_graph(
                    task_id=routing.task_id,
                    task_owner_user_id=routing.task_owner_user_id,
                    turn_id=turn_id,
                    content=display_user_message,
                    file_ids=list(turn_payload.file_ids),
                    expected_run_id=routing.run_id,
                    expected_status=TaskStatus.RUNNING,
                ):
                    raise _WebSocketCommitOutcomeUnknown(
                        f"created task turn {turn_id} has an unknown commit outcome"
                    ) from commit_error
            delivery_claimed = True
            logger.info(
                "Created and claimed task %s, replacing old task_id %s",
                routing.task_id,
                requested_task_id,
            )
        elif client_message_id is not None:
            existing_delivery = inspect_user_message_delivery(
                db,
                routing.task_id,
                display_user_message,
                attachments=persisted_attachments or None,
                turn_id=turn_id,
            )
            if existing_delivery is not None:
                if (
                    durable_attempt_count > 1
                    and existing_delivery.pending
                    and existing_delivery.payload_matches
                ):
                    recovered_delivery = _UserMessageDeliverySnapshot(
                        claimed=True,
                        payload_matches=True,
                        failed=False,
                        pending=True,
                    )
                    delivery_claimed = True
                else:
                    existing_delivery_snapshot = _snapshot_user_message_delivery(
                        existing_delivery
                    )

        uses_live_control = _task_status_uses_live_control(
            routing.status,
            control_state=routing.control_state,
            pause_accepted=pause_accepted,
        )
        if claimed_created_turn is not None:
            # This RUNNING row is the just-claimed first turn, not a
            # continuation into an already-running agent.
            uses_live_control = False
        if recovered_delivery is not None and durable_target_run_id == routing.run_id:
            uses_live_control = True

        return _WebSocketTurnPreparation(
            requested_task_id=requested_task_id,
            routing=routing,
            task_created=task_created,
            execution_context=deepcopy(execution_context),
            user_message_for_llm=user_message_for_llm,
            display_user_message=display_user_message,
            display_file_refs=tuple(deepcopy(display_file_refs)),
            persisted_attachments=tuple(deepcopy(persisted_attachments)),
            turn_payload=turn_payload,
            claimed_created_turn=claimed_created_turn,
            existing_delivery=existing_delivery_snapshot,
            recovered_delivery=recovered_delivery,
            delivery_claimed=delivery_claimed,
            delivery_dispatched=delivery_dispatched,
            uses_live_control=uses_live_control,
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
    delivery_injected = False
    delivery_claimed = False
    delivery_failure_persist_attempted = False
    delivery_failure_pool_timeout = False
    recovered_delivery: _UserMessageDeliverySnapshot | None = None

    async def finish_delivery(
        accepted: bool,
        message: str | None = None,
        *,
        error_code: str | None = None,
        retry_with_new_id: bool = False,
        rejection_outcome: Literal["not_accepted", "outcome_unknown"] | None = None,
    ) -> None:
        nonlocal delivery_finished
        if delivery_finished:
            return
        delivery_finished = True
        if not accepted:
            message_data["_durable_command_error"] = message or "Message was rejected"
        if suppress_delivery_ack:
            return
        await send_message_delivery(
            websocket,
            client_message_id=client_message_id,
            turn_id=turn_id,
            accepted=accepted,
            message=message,
            error_code=error_code,
            retry_with_new_id=retry_with_new_id,
            rejection_outcome=rejection_outcome,
        )

    async def finish_delivery_failure(
        message: str,
        *,
        error_code: str | None = None,
    ) -> bool:
        """Reject pre-dispatch failures; never confuse persistence with delivery."""

        nonlocal delivery_failure_persist_attempted, delivery_failure_pool_timeout
        if delivery_finished:
            return not delivery_failure_pool_timeout
        if (
            delivery_claimed
            and not delivery_dispatched
            and not delivery_failure_persist_attempted
            and not delivery_injected
        ):
            # Set before awaiting the worker. If its checkout times out and the
            # exception reaches another handler layer, that layer must not
            # issue the same write again against the exhausted pool.
            delivery_failure_persist_attempted = True
            try:
                await run_db_io_cancellation_safe(
                    lambda: mark_user_message_delivery_sync(
                        task_id,
                        turn_id,
                        DELIVERY_FAILED,
                    )
                )
            except Exception as delivery_error:
                if not is_database_pool_timeout(delivery_error):
                    raise
                delivery_failure_pool_timeout = True
                logger.error(
                    "task_id=%s component=live-control-delivery database pool "
                    "checkout timed out; not retrying failure persistence: %s",
                    task_id,
                    delivery_error,
                    exc_info=True,
                )
        if delivery_dispatched:
            await finish_delivery(True)
        else:
            await finish_delivery(
                False,
                message,
                error_code=error_code,
                rejection_outcome=(
                    "outcome_unknown"
                    if delivery_failure_pool_timeout or delivery_injected
                    else "not_accepted"
                ),
            )
        return not delivery_failure_pool_timeout

    async def answer_durable_turn_failure(sender_error_code: ClientErrorCode) -> bool:
        """Answer a durable turn failure, addressing each audience as #1514 does.

        The rejection ack and its suppressed-ack replacement bubble carry the
        specific sender code. The task-wide broadcast also reaches widget and
        share subscribers who did not initiate the turn, so it carries only the
        neutral task-failure code and fallback.

        The durable arms below precede the ``RuntimeError`` arm they subclass.
        On main those faults reached that arm and were broadcast from it, so
        answering with the ack alone would quietly stop notifying the task.
        This keeps that notification while giving the sender the specific
        wording each fault deserves.

        Returns ``False`` when the delivery layer says the caller must stop.
        """
        ack_message = client_error_message(sender_error_code)
        if not await finish_delivery_failure(
            ack_message,
            error_code=sender_error_code.value,
        ):
            return False
        timestamp = datetime.now(timezone.utc).timestamp()
        if authorized_task_id is not None:
            safe_error_payload = await _read_task_error_payload_offloop(
                authorized_task_id,
                CLIENT_SAFE_TASK_FAILURE,
                error_code=ClientErrorCode.TASK_EXECUTION_FAILED.value,
            )
            await manager.broadcast_to_task(
                {**safe_error_payload, "timestamp": timestamp},
                authorized_task_id,
            )
            if suppress_delivery_ack:
                await manager.send_personal_message(
                    {
                        "type": "error",
                        "message": ack_message,
                        "error_code": sender_error_code.value,
                        "timestamp": timestamp,
                    },
                    websocket,
                )
        else:
            await manager.send_personal_message(
                {
                    "type": "error",
                    "message": ack_message,
                    "error_code": sender_error_code.value,
                    "timestamp": timestamp,
                },
                websocket,
            )
        return True

    async def finish_existing_delivery(
        claim: Union[UserMessageDeliveryClaim, _UserMessageDeliverySnapshot],
    ) -> None:
        if not claim.payload_matches:
            await finish_delivery(
                False,
                client_error_message(ClientErrorCode.MESSAGE_ID_CONFLICT),
                error_code=ClientErrorCode.MESSAGE_ID_CONFLICT.value,
                retry_with_new_id=True,
                rejection_outcome="not_accepted",
            )
        elif claim.failed:
            await finish_delivery(
                False,
                client_error_message(ClientErrorCode.MESSAGE_DELIVERY_FAILED),
                error_code=ClientErrorCode.MESSAGE_DELIVERY_FAILED.value,
                retry_with_new_id=True,
                rejection_outcome="not_accepted",
            )
        elif claim.pending:
            await finish_delivery(
                False,
                client_error_message(ClientErrorCode.GUIDANCE_IN_PROGRESS),
                error_code=ClientErrorCode.GUIDANCE_IN_PROGRESS.value,
                rejection_outcome="outcome_unknown",
            )
        else:
            await finish_delivery(True)

    try:
        user_message = message_data.get("message", "")
        raw_context = message_data.get("context", {})
        raw_files = message_data.get("files", [])
        user = message_data.get("user")
        authorized_task_id: int | None = None

        if user is None:
            raise ClientVisibleValidationError(
                "User authentication required for task access",
                error_code=ClientErrorCode.AUTHENTICATION_REQUIRED,
            )
        if not isinstance(user_message, str):
            raise ClientVisibleValidationError(
                "Chat message must be a string",
                error_code=ClientErrorCode.INVALID_MESSAGE,
            )
        if not isinstance(raw_context, dict):
            raise ClientVisibleValidationError(
                "Chat context must be an object",
                error_code=ClientErrorCode.INVALID_MESSAGE,
            )
        if not isinstance(raw_files, list):
            raise ClientVisibleValidationError(
                "Chat files must be a list",
                error_code=ClientErrorCode.INVALID_MESSAGE,
            )

        actor_user_id = int(user.id)
        actor_is_admin = bool(user.is_admin)
        pause_accepted = _is_task_pause_accepted(task_id)
        preparation = await run_db_io_cancellation_safe(
            lambda: _prepare_websocket_turn_sync(
                requested_task_id=task_id,
                actor_user_id=actor_user_id,
                actor_is_admin=actor_is_admin,
                user_message=user_message,
                raw_context=deepcopy(raw_context),
                raw_files=deepcopy(raw_files),
                client_message_id=client_message_id,
                turn_id=turn_id,
                durable_attempt_count=int(
                    message_data.get("_durable_attempt_count") or 0
                ),
                durable_target_run_id=message_data.get("_durable_target_run_id"),
                pause_accepted=pause_accepted,
            )
        )
        routing = preparation.routing
        task_id = routing.task_id
        authorized_task_id = task_id
        context = deepcopy(preparation.execution_context)
        user_message_for_llm = preparation.user_message_for_llm
        display_user_message = preparation.display_user_message
        display_file_refs = [
            deepcopy(file_ref) for file_ref in preparation.display_file_refs
        ]
        persisted_attachments = [
            deepcopy(attachment) for attachment in preparation.persisted_attachments
        ]
        turn_payload = preparation.turn_payload
        recovered_delivery = preparation.recovered_delivery
        delivery_claimed = preparation.delivery_claimed
        delivery_dispatched = preparation.delivery_dispatched

        logger.info(f"Received chat message for task {task_id}")
        logger.info(f"👤 User: {actor_user_id}")
        logger.info(f"📄 Message: {user_message}")
        logger.info(
            "📁 Files received from websocket/fallback: %s",
            len(display_file_refs),
        )

        # Call Agent to handle - use same agent manager as chat API
        try:
            from .chat import get_agent_manager

            if preparation.task_created:
                old_task_id = preparation.requested_task_id
                manager.move_connection(websocket, task_id)
                await manager.send_personal_message(
                    {
                        "type": "task_id_updated",
                        "old_task_id": old_task_id,
                        "new_task_id": task_id,
                    },
                    websocket,
                )
                await manager.broadcast_to_task(
                    create_stream_event(
                        "task_info",
                        task_id,
                        deepcopy(routing.task_info),
                        routing.created_at,
                    ),
                    task_id,
                )

            if preparation.claimed_created_turn is not None:
                from ..services.task_orchestrator import TaskTurnOrchestrator

                await TaskTurnOrchestrator.schedule_claimed_create_turn(
                    task_id=task_id,
                    task_owner_user_id=routing.task_owner_user_id,
                    actor_user_id=actor_user_id,
                    payload=turn_payload,
                    claimed=preparation.claimed_created_turn,
                    context=context,
                )
                # Scheduling owns the durable PENDING -> DISPATCHED update.
                # Keep the local flag aligned so a later WebSocket ack failure
                # cannot rewrite an already-running turn as failed delivery.
                delivery_dispatched = True
                message_data["_registered_turn_handoff"] = turn_id
                await finish_delivery(True)
                return

            if preparation.existing_delivery is not None:
                await finish_existing_delivery(preparation.existing_delivery)
                return
            if delivery_dispatched:
                await finish_delivery(True)
                return

            # DAG plan-execute will automatically send the user_message trace
            # event. The transcript write for a new turn is owned atomically by
            # TaskTurnOrchestrator.begin_turn.
            task_uses_live_control = preparation.uses_live_control
            task_owner_user_id = routing.task_owner_user_id
            task_status = routing.status
            task_run_id = routing.run_id
            live_task_lease = (
                routing.task_lease if task_status == TaskStatus.RUNNING else None
            )
            agent_service = None
            supports_live_control = False
            if task_uses_live_control:
                resolved_execution_scope = await run_db_io_cancellation_safe(
                    lambda: resolve_execution_scope(task_id)
                )
                from ..services.task_setup_snapshot import (
                    load_task_setup_snapshot_sync,
                )

                task_setup_snapshot = await run_db_io_cancellation_safe(
                    lambda: load_task_setup_snapshot_sync(
                        task_id,
                        task_owner_user_id,
                        actor_user_id=actor_user_id,
                        actor_is_admin=actor_is_admin,
                    )
                )
                if task_setup_snapshot is None:
                    raise ValueError(f"Task {task_id} is no longer available")
                agent_service = await get_agent_manager().get_agent_for_task(
                    task_id,
                    None,
                    user=task_setup_snapshot.runtime_user,
                    task_setup_snapshot=task_setup_snapshot,
                    task_owner_user_id=task_owner_user_id,
                    resolved_execution_scope=resolved_execution_scope,
                )
                if hasattr(agent_service, "set_outbound_message_handler"):
                    agent_service.set_outbound_message_handler(
                        make_agent_outbound_handler(task_id)
                    )
                supports_live_control = getattr(
                    agent_service, "supports_live_control", lambda: False
                )()

            if task_uses_live_control and supports_live_control:
                logger.info(f"Using agent message control for task {task_id}")
                assert agent_service is not None
                reservation = background_task_manager.try_reserve_resume(
                    task_id,
                    expected_run_id=task_run_id,
                )
                if reservation is not ResumeReservationOutcome.RESERVED:
                    if suppress_delivery_ack:
                        # Durable commands own their retry budget. All three
                        # occupied states can clear without this attempt doing
                        # anything: a reservation can register or release, a
                        # coordinator can finish, and shutdown is retained as
                        # a defensive state even though normal shutdown stops
                        # the dispatcher before setting the manager flag.
                        holder_age_seconds = (
                            background_task_manager.resume_holder_age_seconds(task_id)
                        )
                        holder_age_text = (
                            f"{holder_age_seconds:.3f}"
                            if holder_age_seconds is not None
                            else "unknown"
                        )
                        logger.info(
                            "Deferring message %s for task %s: resume slot "
                            "unavailable (%s), holder_age_seconds=%s",
                            turn_id,
                            task_id,
                            reservation.value,
                            holder_age_text,
                        )
                        message_data["_durable_command_defer"] = turn_id
                        message_data["_durable_command_defer_reason"] = (
                            f"Message {turn_id} is waiting for the live-control "
                            f"resume slot ({reservation.value})"
                        )
                        if recovered_delivery is not None:
                            # A prior attempt claimed the durable delivery and
                            # may have injected it. Task/run-local occupancy is
                            # not evidence that this command owns the live
                            # coordinator after a worker handoff.
                            message_data["_durable_command_defer_unsafe"] = turn_id
                        return
                    await finish_delivery(
                        False,
                        CLIENT_SAFE_GUIDANCE_IN_PROGRESS,
                        error_code=ClientErrorCode.GUIDANCE_IN_PROGRESS.value,
                        rejection_outcome="not_accepted",
                    )
                    return
                # Pass the user-typed bubble text + display-safe file refs
                # alongside the LLM-augmented execution text. The runner
                # persists them onto Message.metadata so its tracing
                # callback can emit the bubble with the typed content +
                # file chips rather than the inflated prompt; matches what
                # historical replay shows on reload.
                # ``post_user_message`` routes into
                # ``AgentRunner.inject_user_message``, whose
                # ``on_user_message_posted`` callback is the single trace
                # emission point. Do not emit a second user-message trace.
                bg_task: asyncio.Task[None] | None = None
                handoff_registered = False
                try:
                    if recovered_delivery is not None:
                        delivery_claim = recovered_delivery
                    else:
                        delivery_claim = await run_db_io_cancellation_safe(
                            lambda: _claim_user_message_delivery_isolated(
                                task_id=task_id,
                                task_owner_user_id=task_owner_user_id,
                                content=display_user_message,
                                attachments=persisted_attachments or None,
                                file_ids=list(turn_payload.file_ids),
                                turn_id=turn_id,
                                expected_run_id=routing.run_id,
                                expected_status=task_status,
                            )
                        )
                    recovered_delivery = None
                    if not delivery_claim.claimed:
                        background_task_manager.release_resume_reservation(task_id)
                        await finish_existing_delivery(delivery_claim)
                        return
                    delivery_claimed = True

                    # Read before the injection below and before the posted
                    # fork, so both branches carry the same observation --
                    # see task_interaction_close's module docstring for why
                    # it has to precede the injection. The two branches are
                    # mutually exclusive: posted true closes below with this
                    # local, posted false hands the same value to
                    # execute_resume_background through pending_user_message,
                    # so one observation only ever serves one close.
                    active_interaction_id = await run_db_io_cancellation_safe(
                        lambda: active_interaction_id_sync(task_id)
                    )

                    posted = UserMessageInjectionOutcome.NOT_POSTED
                    if live_task_lease is not None:
                        with bind_task_lease_context(live_task_lease):
                            try:
                                posted = await agent_service.post_user_message(
                                    str(task_id),
                                    execution_message=user_message_for_llm,
                                    display_message=display_user_message,
                                    files=display_file_refs,
                                    turn_id=turn_id,
                                    request_interrupt=True,
                                    reason="new websocket user message",
                                )
                            except CheckpointUnavailableError:
                                # Fold into the existing not-posted path
                                # below: the durable message is deferred to
                                # the resume owner instead of injected live,
                                # exactly as when there was no exact lease
                                # or checkpoint to inject into. Distinct
                                # from corrupt/refused, which are not
                                # retryable by simply deferring.
                                posted = UserMessageInjectionOutcome.NOT_POSTED
                            except CheckpointReadError:
                                # Corrupt and refused reach here today. The
                                # base class is deliberate: a read failure
                                # that is not the retryable-by-deferring
                                # unavailable case must reject the claimed
                                # delivery rather than escape this handler
                                # and orphan it. Use finish_delivery_failure,
                                # not finish_delivery, so the row is actually
                                # persisted DELIVERY_FAILED -- otherwise it
                                # stays DELIVERY_PENDING forever and a retry
                                # with the same client_message_id loops on
                                # "still being applied".
                                background_task_manager.release_resume_reservation(
                                    task_id
                                )
                                await answer_durable_turn_failure(
                                    ClientErrorCode.TASK_CHECKPOINT_UNREADABLE
                                )
                                return
                    delivery_injected = bool(posted)
                    if not posted:
                        logger.warning(
                            "Agent execution %s had no exact live lease or "
                            "checkpoint; deferring the durable user message "
                            "until the resume owner is ready",
                            task_id,
                        )
                    handoff_snapshot = await task_execution_controller.transition(
                        task_id,
                        TaskControlState.RESUME_REQUESTED,
                        expected_run_id=task_run_id,
                    )

                    previous_task = background_task_manager.running_tasks.get(task_id)
                    bg_task = asyncio.create_task(
                        execute_resume_background(
                            task_id=task_id,
                            agent_service=agent_service,
                            task_owner_user_id=task_owner_user_id,
                            # Also the transition's run, for the same reason
                            # as the registration below: a ``None`` here
                            # reaches ``acquire_task_lease_no_commit``, whose
                            # ``candidate_run_id = expected_run_id or uuid4()``
                            # would mint a *second* run and claim the lease
                            # under it -- leaving the row, the coordinator
                            # registration, and the execution on three
                            # different answers for one resume.
                            #
                            # Two further effects on that formerly-NULL path,
                            # both intended. The claim now carries
                            # ``WHERE run_id = :expected``, so if the run
                            # rotates before it lands -- the window includes
                            # the unbounded ``await previous_task`` -- the
                            # claim returns None and the delivery fails
                            # cleanly instead of stealing the lease. And the
                            # ``expected_run_id is None`` branch that clears
                            # the checkpoint pointers is now skipped, which
                            # is what a resume wants: those pointers are the
                            # anchor it is resuming from.
                            expected_run_id=handoff_snapshot.run_id,
                            previous_task=previous_task,
                            resolved_execution_scope=resolved_execution_scope,
                            pending_user_message=(
                                None
                                if posted
                                else {
                                    "execution_message": user_message_for_llm,
                                    "display_message": display_user_message,
                                    "files": display_file_refs,
                                    "turn_id": turn_id,
                                    # The pre-injection observation, carried
                                    # rather than re-read: the deferred path
                                    # injects later still, so a read there
                                    # would be even further past the point
                                    # where the answered row is identifiable.
                                    "interaction_id": active_interaction_id,
                                }
                            ),
                            delivery_turn_id=turn_id,
                            delivery_already_dispatched=bool(posted),
                            delivery_websocket=(
                                None if posted or suppress_delivery_ack else websocket
                            ),
                            delivery_client_message_id=(
                                None
                                if posted or suppress_delivery_ack
                                else client_message_id
                            ),
                        )
                    )
                    background_task_manager.register_reserved_resume(
                        task_id,
                        bg_task,
                        # The transition's run id, not the routing snapshot's.
                        # ``apply_task_control_transition`` mints a fresh run
                        # for a legacy row whose ``run_id`` is NULL, so the
                        # pre-transition value would register this coordinator
                        # under ``None`` while the task runs under a uuid. A
                        # later RESUME asking about that uuid would then read
                        # a live resume as RESERVATION_HELD and defer itself
                        # to a terminal failure. The other three registration
                        # sites already use their post-transition value.
                        run_id=handoff_snapshot.run_id,
                    )
                    handoff_registered = True
                    if posted:
                        # Registration completes the local resume handoff.
                        # The delivery marker is a best-effort projection and
                        # must not reject a turn that is already resumable.
                        delivery_dispatched = True
                        message_data["_registered_turn_handoff"] = turn_id
                        try:
                            await run_db_io_cancellation_safe(
                                lambda: mark_user_message_delivery_sync(
                                    task_id,
                                    turn_id,
                                    DELIVERY_DISPATCHED,
                                )
                            )
                        except Exception:
                            logger.warning(
                                "delivery marker failed after registered resume handoff "
                                "for task %s turn %s",
                                task_id,
                                turn_id,
                                exc_info=True,
                            )
                        # For a first attempt, retiring the interaction row
                        # observed before the injection and clearing the
                        # task's marker in the same short transaction is
                        # correct: the message went in outside the native
                        # interaction protocol's answer path, so a question
                        # this run had open under that protocol was
                        # answered by other means.
                        #
                        # That reading breaks on a replay. This site reads
                        # its own id fresh on every attempt, so on a replay
                        # the id above is not the question the replayed
                        # message answered -- the first attempt retired
                        # that one -- it is whatever the resumed agent has
                        # staged since, and closing on it would retire a
                        # live question. `posted` alone cannot tell a fresh
                        # write from a replay; AgentRunner.inject_user_message
                        # reports the distinction explicitly and this guard
                        # reads that report. See task_interaction_close's
                        # module docstring for the rule, the other sites,
                        # and why the v1 reply resume-input path needs no
                        # guard at all.
                        #
                        # The run fence
                        # is live_task_lease.run_id, not task_run_id: posted
                        # being true only happens by way of the
                        # live_task_lease is not None branch above, which is
                        # what makes this attribute access safe. Bound to a
                        # plain local first, not read from live_task_lease
                        # inside the lambda below: a narrowing assert on an
                        # enclosing-scope variable does not apply inside a
                        # nested closure.
                        assert live_task_lease is not None
                        assert live_task_lease.run_id is not None
                        close_run_id = live_task_lease.run_id
                        close_interaction_id = active_interaction_id
                        if posted is UserMessageInjectionOutcome.POSTED_FRESH:
                            try:
                                await run_db_io_cancellation_safe(
                                    lambda: close_legacy_resume_interaction_sync(
                                        task_id=task_id,
                                        run_id=close_run_id,
                                        interaction_id=close_interaction_id,
                                    )
                                )
                            except Exception:
                                logger.warning(
                                    "legacy resume interaction close failed after "
                                    "registered resume handoff for task %s run %s",
                                    task_id,
                                    close_run_id,
                                    exc_info=True,
                                )
                            except asyncio.CancelledError:
                                # run_db_io_cancellation_safe drains its worker
                                # thread to completion before propagating a
                                # cancellation raised while awaiting it, so by the
                                # time this branch runs the close-and-clear
                                # transaction has already committed or failed on
                                # its own; there is nothing left in flight to
                                # protect. This only logs the interruption instead
                                # of letting it escape as an unhandled
                                # cancellation. Unlike the delivery marker above,
                                # this branch is deliberate, not a gap to copy.
                                logger.warning(
                                    "legacy resume interaction close was cancelled "
                                    "for task %s run %s; the resume proceeds "
                                    "unaffected",
                                    task_id,
                                    close_run_id,
                                )
                except BaseException:
                    if bg_task is not None and not handoff_registered:
                        bg_task.cancel()
                    if not handoff_registered:
                        background_task_manager.release_resume_reservation(task_id)
                    raise

                if posted:
                    await finish_delivery(True)
                return
            elif task_uses_live_control:
                # A runtime without the durable checkpoint/live-control
                # contract cannot safely accept a continuation: there is no
                # exact lease handoff or completion owner to fence it.
                logger.error(
                    "Task %s does not support durable message continuation",
                    task_id,
                )
                await manager.send_personal_message(
                    {
                        "type": "error",
                        "message": client_error_message(
                            ClientErrorCode.MESSAGE_CONTINUATION_UNSUPPORTED
                        ),
                        "error_code": (
                            ClientErrorCode.MESSAGE_CONTINUATION_UNSUPPORTED.value
                        ),
                    },
                    websocket,
                )
                await finish_delivery(
                    False,
                    client_error_message(
                        ClientErrorCode.MESSAGE_CONTINUATION_UNSUPPORTED
                    ),
                    error_code=ClientErrorCode.MESSAGE_CONTINUATION_UNSUPPORTED.value,
                    rejection_outcome="not_accepted",
                )
                return
            else:
                # New task/turn (PENDING/COMPLETED/FAILED/PAUSED), execute normally
                if pause_accepted and routing.status in {
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
                    refreshed_routing = await run_db_io_cancellation_safe(
                        lambda: _load_websocket_task_routing_snapshot_sync(
                            task_id,
                            task_owner_user_id=routing.task_owner_user_id,
                            actor_user_id=actor_user_id,
                            actor_is_admin=actor_is_admin,
                        )
                    )
                    if refreshed_routing is None:
                        raise ValueError(f"Task {task_id} is no longer available")
                    routing = refreshed_routing
                    if routing.status in {
                        TaskStatus.RUNNING,
                        TaskStatus.WAITING_FOR_USER,
                    }:
                        error_payload = await _read_task_error_payload_offloop(
                            task_id,
                            client_error_message(
                                ClientErrorCode.TASK_PAUSE_IN_PROGRESS
                            ),
                            event_type="agent_error",
                            error_code=ClientErrorCode.TASK_PAUSE_IN_PROGRESS.value,
                        )
                        await manager.broadcast_to_task(
                            {
                                **error_payload,
                                "timestamp": datetime.now(timezone.utc).timestamp(),
                            },
                            task_id,
                        )
                        await finish_delivery(
                            False,
                            client_error_message(
                                ClientErrorCode.TASK_PAUSE_IN_PROGRESS
                            ),
                            error_code=ClientErrorCode.TASK_PAUSE_IN_PROGRESS.value,
                            rejection_outcome="not_accepted",
                        )
                        return
                    _clear_task_pause_accepted(task_id)

                logger.info(
                    "Task %s starting new execution turn (status: %s)",
                    task_id,
                    routing.status.value,
                )

                # The execution wrapper acquires the lease just before it
                # starts running. Avoid acquiring it during setup so setup
                # failures cannot leave the task locked.
                if routing.status != TaskStatus.RUNNING:
                    logger.info(
                        "Sending task_info event for task %s, status: %s",
                        task_id,
                        routing.status.value,
                    )
                    task_event = create_stream_event(
                        "task_info",
                        task_id,
                        deepcopy(routing.task_info),
                        routing.created_at,
                    )
                    await manager.broadcast_to_task(task_event, task_id)
                    logger.info(f"task_info event sent for existing task {task_id}")

                context.update(deepcopy(routing.task_context))

                # WS builds the display/execution payload here and
                # delegates the full new-turn transition to the
                # shared orchestrator. ``begin_turn`` owns the
                # atomic claim (status flip + input set + terminal-
                # field reset), the transcript persist, the
                # single-commit transaction, and the lease-aware bg
                # schedule -- so WS and /v1 SDK use one turn-
                # lifecycle state machine.
                from ..services.task_orchestrator import (
                    TaskTurnCommitOutcomeUnknown,
                    TaskTurnError,
                    TaskTurnNotFoundError,
                    TaskTurnOrchestrator,
                    TurnKind,
                )

                # Preparation already built the shared transcript/execution
                # payload after stripping absolute paths from persisted
                # attachments. Both the missing-task atomic CREATE path and
                # existing-task begin_turn path use this same value.
                payload = turn_payload
                # WS path has these legal entries into begin_turn:
                #   PENDING                  → CREATE
                #   COMPLETED / FAILED       → APPEND
                #   PAUSED + user message    → APPEND (new turn)
                # WAITING_FOR_USER / RUNNING should have been intercepted
                # by the live-control path above. Reaching this branch
                # with either is an upstream-dispatch bug; surface it as
                # an agent_error rather than silently letting begin_turn
                # 409 on the wrong status.
                if routing.status == TaskStatus.PENDING:
                    turn_kind = TurnKind.CREATE
                    turn_force_fresh = False
                elif routing.status in (
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                ):
                    turn_kind = TurnKind.APPEND
                    turn_force_fresh = False
                elif routing.status == TaskStatus.PAUSED:
                    turn_kind = TurnKind.APPEND
                    turn_force_fresh = False
                else:
                    logger.error(
                        f"WS schedule reached for task {task_id} with "
                        f"unexpected status={routing.status}; expected "
                        "PENDING, PAUSED, or terminal. Live-control path "
                        "should have intercepted."
                    )
                    error_payload = await _read_task_error_payload_offloop(
                        task_id,
                        client_error_message(ClientErrorCode.MESSAGE_PROCESSING_FAILED),
                        event_type="agent_error",
                        error_code=ClientErrorCode.MESSAGE_PROCESSING_FAILED.value,
                    )
                    await manager.broadcast_to_task(
                        {
                            **error_payload,
                            "timestamp": datetime.now(timezone.utc).timestamp(),
                        },
                        task_id,
                    )
                    await finish_delivery(
                        False,
                        client_error_message(ClientErrorCode.MESSAGE_PROCESSING_FAILED),
                        error_code=ClientErrorCode.MESSAGE_PROCESSING_FAILED.value,
                        rejection_outcome="not_accepted",
                    )
                    return

                turn_task_id = routing.task_id
                turn_owner_user_id = routing.task_owner_user_id
                turn_actor_user_id = actor_user_id
                try:
                    await TaskTurnOrchestrator.begin_turn(
                        task_id=turn_task_id,
                        # Owner, not the acting principal: ``task`` was
                        # already authorized above (admin bypass / owner
                        # check), and the turn must run as the task owner,
                        # not an admin acting on someone else's task.
                        task_owner_user_id=turn_owner_user_id,
                        # The acting principal (the admin when acting on
                        # another user's task) -- audit/logging only.
                        actor_user_id=turn_actor_user_id,
                        payload=payload,
                        kind=turn_kind,
                        force_fresh=turn_force_fresh,
                        context=context,
                    )
                    message_data["_registered_turn_handoff"] = turn_id
                    logger.info(f"Task {task_id} started in background")
                    await finish_delivery(True)
                except TaskTurnCommitOutcomeUnknown:
                    message_data["_commit_outcome_unknown"] = turn_id
                    await finish_delivery(
                        False,
                        client_error_message(
                            ClientErrorCode.MESSAGE_ACCEPTANCE_PENDING
                        ),
                        error_code=ClientErrorCode.MESSAGE_ACCEPTANCE_PENDING.value,
                        rejection_outcome="outcome_unknown",
                    )
                except TaskTurnNotFoundError:
                    # Task vanished or changed ownership between the
                    # resolve above and the atomic claim — surface it the
                    # same way as a busy refusal (no row was mutated).
                    logger.warning(
                        "begin_turn: task %s not found / not owned at claim",
                        task_id,
                    )
                    error_payload = await _read_task_error_payload_offloop(
                        task_id,
                        client_error_message(ClientErrorCode.TASK_UNAVAILABLE),
                        event_type="agent_error",
                        error_code=ClientErrorCode.TASK_UNAVAILABLE.value,
                    )
                    await manager.broadcast_to_task(
                        {
                            **error_payload,
                            "timestamp": datetime.now(timezone.utc).timestamp(),
                        },
                        task_id,
                    )
                    await finish_delivery(
                        False,
                        client_error_message(ClientErrorCode.TASK_UNAVAILABLE),
                        error_code=ClientErrorCode.TASK_UNAVAILABLE.value,
                        rejection_outcome="not_accepted",
                    )
                except TaskTurnError as busy_err:
                    # begin_turn's atomic transaction rolls back on
                    # bg_inflight / busy — neither the status flip
                    # nor the user message persists, so no transcript
                    # cleanup is needed here. The rejected-turn-leaves-
                    # no-side-effect contract makes the previous
                    # best-effort delete unnecessary.
                    logger.warning(
                        f"Refused to schedule bg for task {task_id}: {busy_err.reason}"
                    )
                    rejection_code = _TURN_REJECTION_CODES.get(
                        busy_err.reason, ClientErrorCode.TASK_BUSY
                    )
                    rejection_message = client_error_message(rejection_code)
                    error_payload = await _read_task_error_payload_offloop(
                        task_id,
                        rejection_message,
                        event_type="agent_error",
                        error_code=rejection_code.value,
                    )
                    await manager.broadcast_to_task(
                        {
                            **error_payload,
                            "timestamp": datetime.now(timezone.utc).timestamp(),
                        },
                        task_id,
                    )
                    await finish_delivery(
                        False,
                        rejection_message,
                        error_code=rejection_code.value,
                        rejection_outcome="not_accepted",
                    )

        except _WebSocketCommitOutcomeUnknown:
            message_data["_commit_outcome_unknown"] = turn_id
            await finish_delivery(
                False,
                client_error_message(ClientErrorCode.MESSAGE_ACCEPTANCE_PENDING),
                error_code=ClientErrorCode.MESSAGE_ACCEPTANCE_PENDING.value,
                rejection_outcome="outcome_unknown",
            )
        except (ValueError, KeyError, TypeError) as e:
            # Data validation and format error
            if (
                isinstance(e, ClientVisibleError)
                and e.error_code == ClientErrorCode.MESSAGE_ATTACHMENT_UNAVAILABLE
            ):
                # Preparation verified the attachment before the later atomic
                # bind lost its race.  The specific state belongs only to the
                # verified origin; task subscribers receive the same neutral
                # execution failure as every other sender-specific durable
                # attachment fault.
                log_client_facing_failure(
                    e,
                    "Attachment bind race in agent execution: %s",
                )
                if not await answer_durable_turn_failure(e.error_code):
                    return
                return
            error_code = (
                e.error_code
                if isinstance(e, ClientVisibleError)
                else ClientErrorCode.MESSAGE_PROCESSING_FAILED
            )
            message = client_error_message(error_code)
            log_client_facing_failure(e, "Data validation error in agent execution: %s")
            if not await finish_delivery_failure(
                message,
                error_code=error_code.value,
            ):
                return
            timestamp = datetime.now(timezone.utc).timestamp()
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
        except DurableObjectIntegrityError:
            # Precedes the durable-fault arm below, which this subclasses. A
            # checksum mismatch is permanent corruption, already recorded at
            # ERROR with both checksums where it is raised, so it must not also
            # be logged as a transient outage. It still owes the client an
            # answer, and a distinct one: retrying cannot help, the stored copy
            # has to be replaced.
            #
            # The allowlisted code selects a fixed fallback for every audience;
            # no exception text crosses the boundary.
            if not await answer_durable_turn_failure(
                ClientErrorCode.MESSAGE_ATTACHMENT_CORRUPT
            ):
                return
        except DurableStorageOperationError as exc:
            # Must precede the RuntimeError arm below, which this subclasses.
            # This is the selected-file attachment path: the fault arrives here
            # first, is answered to the client, and is swallowed -- so this is
            # both the only place its provider cause can be recorded and the
            # last place its text could escape.
            #
            # Fixed contract fallback rather than ``str(exc)``: the wrap's own
            # text is not what a client should read, and this arm is also the
            # sole logging owner for this path -- the fault does not re-raise,
            # so the endpoint-level arm never sees it and cannot double-record.
            log_durable_storage_fault(
                logger,
                "websocket agent execution",
                exc,
                task_id=authorized_task_id,
            )
            if not await answer_durable_turn_failure(
                ClientErrorCode.MESSAGE_ATTACHMENT_UNAVAILABLE
            ):
                return
        except RuntimeError as e:
            # RuntimeError is incidental server detail. Reuse the same
            # audience split as the durable-failure arms above: the initiator
            # gets the message-processing code while task subscribers get the
            # neutral task-failure code.
            logger.error("Runtime error in agent execution: %s", e, exc_info=True)
            if not await answer_durable_turn_failure(
                ClientErrorCode.MESSAGE_PROCESSING_FAILED
            ):
                return
        except Exception as e:
            # Other unknown errors, re-raise
            # The branch that withholds the detail logs it, rather than
            # relying on a caller: the durable dispatcher does record a stack
            # (task_command_transport.py:1100, logger.exception) but
            # websocket_chat_endpoint and both public_chat_access.py endpoints
            # log without exc_info, so the record depends on who called.
            logger.error("Unexpected error in agent execution: %s", e, exc_info=True)
            await finish_delivery_failure(client_safe_error_message(e))
            raise

    except ClientVisiblePermissionError as e:
        log_client_facing_failure(e, "Message permission error: %s")
        message = client_error_message(e.error_code)
        await finish_delivery_failure(message, error_code=e.error_code.value)
        await manager.send_personal_message(
            {"type": "error", "message": message, "error_code": e.error_code.value},
            websocket,
        )
    except (ValueError, KeyError, TypeError) as e:
        # Message format error
        log_client_facing_failure(e, "Message format error: %s")
        error_code = (
            e.error_code
            if isinstance(e, ClientVisibleError)
            else ClientErrorCode.MESSAGE_PROCESSING_FAILED
        )
        message = client_error_message(error_code)
        await finish_delivery_failure(message, error_code=error_code.value)
        await manager.send_personal_message(
            {"type": "error", "message": message, "error_code": error_code.value},
            websocket,
        )
    except (ConnectionError, WebSocketDisconnect) as e:
        # Connection error
        logger.error("Connection error handling chat message: %s", e)
        raise
    except DurableObjectIntegrityError:
        # Attachment preparation runs in this outer scope (see the
        # ``_prepare_websocket_turn_sync`` call above), *before* the inner
        # agent-execution try. So a stored-file fault surfaces here, not in the
        # arms guarding that inner block -- which is why the fixed detail has to
        # be applied at this level too.
        #
        # Corruption is permanent: the copy has to be replaced, so the client is
        # told that rather than to retry. No exception text goes outbound; the
        # integrity ERROR with both checksums is already logged where it is
        # raised.
        # The allowlisted code selects the fixed re-upload guidance.
        await finish_delivery_failure(
            client_error_message(ClientErrorCode.MESSAGE_ATTACHMENT_CORRUPT),
            error_code=ClientErrorCode.MESSAGE_ATTACHMENT_CORRUPT.value,
        )
        raise
    except DurableStorageOperationError as exc:
        # Same scope reasoning as above. ``str(exc)`` is the wrap's message and
        # carries the storage key, whose scope segments encode the owning user's
        # id -- it must not reach a socket frame, a persisted rejection, or a
        # broadcast, any more than it may reach an HTTP body or a model (#1467).
        #
        # Logging here rather than leaving it to the endpoint arm: the
        # durable-command route invokes this handler directly and never reaches
        # that arm. Double-recording is prevented at the logger, which marks the
        # fault, so both arms are safe to write independently.
        log_durable_storage_fault(
            logger,
            "websocket chat turn preparation",
            exc,
            task_id=task_id,
        )
        await finish_delivery_failure(
            client_error_message(ClientErrorCode.MESSAGE_ATTACHMENT_UNAVAILABLE),
            error_code=ClientErrorCode.MESSAGE_ATTACHMENT_UNAVAILABLE.value,
        )
        raise
    except Exception as e:
        # Other errors, re-raise
        # Redacted below, so the traceback is the only record left.
        logger.error("Unexpected error handling chat message: %s", e, exc_info=True)
        await finish_delivery_failure(client_safe_error_message(e))
        raise


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
        ) = _resolve_task_llm_ids(task, db)
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

        from ..services.task_orchestrator import (
            TaskTurnOrchestrator,
            TaskTurnPayload,
        )

        background_task = await TaskTurnOrchestrator.schedule_existing_task_execution(
            task_id=request.task_id,
            task_owner_user_id=request.task_owner_user_id,
            task_source=request.task_source,
            payload=TaskTurnPayload(
                transcript_message=request.task_description,
                execution_message=request.task_description,
            ),
            context=request.task_context,
            actor_user_id=actor_user_id,
        )
        # The legacy command did not return until execution finished. Keep that
        # ordering while the scheduled coroutine owns all runtime DB work.
        await background_task

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
        log_client_facing_failure(e, "Task execution rejected: %s")
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
            trace_scope = "workforce-top-level-v1" if is_workforce_run else "public-v1"

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
            ) = _resolve_task_llm_ids(task, db)
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
                    content = client_safe_assistant_history_content(
                        content=content,
                        message_type=str(chat_message.message_type),
                    )
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
                    if (
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
        log_client_facing_failure(e, "Data validation error in intervention: %s")
        await manager.send_personal_message(
            {"type": "error", "message": client_safe_error_message(e)}, websocket
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
            task_id=task_id,
            message_data=message_data,
            kind=TaskCommandKind.PAUSE,
            command_id=_client_message_id(message_data.get("command_id")),
        )
    except (
        MCPBuiltinOAuthActorPolicyRequiredError,
        PermissionError,
        ValueError,
    ) as exc:
        log_client_facing_failure(
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
    if enqueued.created:
        # Only the creating ingress owns the origin; a payload-matching
        # duplicate must never bind (see handle_chat_message). Registered
        # before dispatch so local execution cannot outrun the binding.
        _command_origins.register(enqueued.client_command_id, websocket, task_id)
    await dispatch_task_command_promptly(
        execute_durable_task_command,
        command_db_id=enqueued.command_id,
    )


def _apply_pause_requested_isolated(
    task_id: int,
    *,
    expected_run_id: str | None,
) -> bool:
    """Persist PAUSE_REQUESTED for the exact RUNNING run in a short Session."""

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        values: dict[str, Any] = {
            "control_state": TaskControlState.PAUSE_REQUESTED.value,
            "state_version": func.coalesce(Task.state_version, 0) + 1,
        }
        if expected_run_id is None:
            # Preserve ``apply_task_control_transition`` semantics for legacy
            # RUNNING rows that predate run ids.
            values["run_id"] = str(uuid.uuid4())

        statement = update(Task).where(
            Task.id == task_id,
            Task.status == TaskStatus.RUNNING,
        )
        statement = (
            statement.where(Task.run_id.is_(None))
            if expected_run_id is None
            else statement.where(Task.run_id == expected_run_id)
        )
        result = db.execute(
            statement.values(**values).execution_options(synchronize_session=False)
        )
        if int(getattr(result, "rowcount", 0) or 0) == 1:
            db.commit()
            return True

        current = db.query(Task.run_id, Task.status).filter(Task.id == task_id).first()
        if current is not None:
            current_run_id = str(current[0]) if current[0] is not None else None
            if current_run_id != expected_run_id:
                raise StaleTaskRunError(
                    f"task {task_id} run changed from {expected_run_id} "
                    f"to {current_run_id}"
                )
        return False


async def _handle_pause_task_unserialized(
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

        from ..services.task_setup_snapshot import load_task_setup_snapshot_sync
        from .chat import get_agent_manager

        task_setup_snapshot = await run_db_io_cancellation_safe(
            lambda: load_task_setup_snapshot_sync(
                task_id,
                None,
                actor_user_id=int(user.id),
                actor_is_admin=bool(user.is_admin),
            )
        )
        if task_setup_snapshot is None:
            logger.warning(
                "pause: task %s not found or not owned by user %s", task_id, user.id
            )
            raise ValueError(f"Access denied: task {task_id} is not available")

        task_fields = task_setup_snapshot.task
        task_owner_user_id = int(task_fields.user_id)
        expected_run_id = task_fields.run_id
        # Off-turn: on an agent-cache hit this only locates the already-
        # running agent's existing workspace/sandbox to pause it. On a miss,
        # get_agent_for_task below builds a fresh agent from this value,
        # which can materialize a workspace directory tree and acquire a
        # sandbox lease. resolve_execution_scope_off_turn resolves this value
        # through three distinct outcomes:
        # - resolver authoritative, snapshot disagrees on a namespace field:
        #   downgrades to the resolver's own answer (with a warning) instead
        #   of raising, so the pause still proceeds -- the value here is the
        #   trusted resolver answer, not the snapshot.
        # - resolver abstains, snapshot widens the abstention's fallback:
        #   ExecutionScopeAbstentionMismatchError is re-raised rather than
        #   downgraded, so the pause is refused outright -- an abstention
        #   never produced an authoritative value to fall back to.
        # - resolver abstains, snapshot narrows the abstention's fallback:
        #   the returned value IS the snapshot (policy fields overlaid from
        #   the fallback). That is persisted, client-influenceable data, and
        #   it is trusted here only because it was already validated as a
        #   narrowing of what the resolver granted, so anything the build
        #   below materializes from it still lands inside the authorised
        #   subtree.
        # Pause schedules no turn, so nothing downstream re-resolves or
        # corrects a build that happens here.
        execution_scope = await run_db_io_cancellation_safe(
            lambda: resolve_execution_scope_off_turn(task_id)
        )

        # Get agent service (as the task owner)
        logger.info(f"Getting agent service for task {task_id}")
        agent_service = await get_agent_manager().get_agent_for_task(
            task_id,
            None,
            user=task_setup_snapshot.runtime_user,
            task_setup_snapshot=task_setup_snapshot,
            task_owner_user_id=task_owner_user_id,
            resolved_execution_scope=execution_scope,
        )
        logger.info(f"Agent service obtained: {type(agent_service).__name__}")

        # Check if agent supports pause functionality
        if hasattr(agent_service, "pause_execution"):
            logger.info("Agent supports pause_execution, calling it...")
            pause_result = await agent_service.pause_execution()
            if pause_result is False:
                # ``pause_execution`` reports on the live run only, so it says
                # "no" both for a task that is already paused and for one that
                # is not running at all. Those read very differently to a user,
                # so the persisted status picks the message.
                pause_failure = (
                    "Task is already paused"
                    if task_fields.status == TaskStatus.PAUSED
                    else "No live execution found to pause"
                )
                message_data["_durable_command_error"] = pause_failure
                error_payload = await _read_task_error_payload_offloop(
                    task_id,
                    pause_failure,
                )
                await manager.send_personal_message(
                    error_payload,
                    websocket,
                )
                logger.warning("%s for task %s", pause_failure, task_id)
                return
            logger.info("Agent pause_execution completed")
            pause_applied = await run_db_io_cancellation_safe(
                lambda: _apply_pause_requested_isolated(
                    task_id,
                    expected_run_id=expected_run_id,
                )
            )
            if not pause_applied:
                message_data["_durable_command_error"] = (
                    "Task finished before the pause request was applied"
                )
                error_payload = await _read_task_error_payload_offloop(
                    task_id,
                    "Task finished before the pause request was applied",
                )
                await manager.send_personal_message(
                    error_payload,
                    websocket,
                )
                return
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
            error_payload = await _read_task_error_payload_offloop(
                task_id,
                "Current agent does not support pause functionality",
            )
            await manager.send_personal_message(
                error_payload,
                websocket,
            )
            logger.warning(
                f"Agent for task {task_id} does not support pause functionality"
            )

    except (ValueError, KeyError, TypeError) as e:
        # Data validation error
        message_data["_durable_command_error"] = str(e)
        logger.error(
            "Data validation error pausing task %s: %s", task_id, e, exc_info=True
        )
        await manager.send_personal_message(
            {"type": "error", "message": client_safe_error_message(e)}, websocket
        )
    except RuntimeError as e:
        logger.error("Runtime error pausing task %s: %s", task_id, e, exc_info=True)
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
        raise
    except Exception as e:
        # Other errors, re-raise
        logger.error("Unexpected error pausing task %s: %s", task_id, e)
        raise


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
    except (
        MCPBuiltinOAuthActorPolicyRequiredError,
        PermissionError,
        ValueError,
    ) as exc:
        log_client_facing_failure(
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
    if enqueued.created:
        # Only the creating ingress owns the origin; a payload-matching
        # duplicate must never bind (see handle_chat_message). Registered
        # before dispatch so local execution cannot outrun the binding.
        _command_origins.register(enqueued.client_command_id, websocket, task_id)
    await dispatch_task_command_promptly(
        execute_durable_task_command,
        command_db_id=enqueued.command_id,
    )


async def _handle_resume_task_unserialized(
    websocket: WebSocket, task_id: int, message_data: dict
) -> ResumeCommandResult:
    """Handle task resume request"""
    try:
        user = message_data.get("user")
        if not user:
            raise ValueError("User authentication required for task resume")

        from ..services.task_setup_snapshot import load_task_setup_snapshot_sync
        from .chat import get_agent_manager

        task_setup_snapshot = await run_db_io_cancellation_safe(
            lambda: load_task_setup_snapshot_sync(
                task_id,
                None,
                actor_user_id=int(user.id),
                actor_is_admin=bool(user.is_admin),
            )
        )
        if task_setup_snapshot is None:
            logger.warning(
                "Task %s not found or access denied for user %s",
                task_id,
                user.id,
            )
            reason = "Task not found or access denied"
            await manager.send_personal_message(
                {"type": "error", "message": "Task not found or access denied"},
                websocket,
            )
            return ResumeCommandResult(
                ResumeCommandOutcome.REJECTED,
                reason,
                reason_code="task_not_found",
            )

        task_fields = task_setup_snapshot.task
        task_owner_user_id = int(task_fields.user_id)
        task_status = cast(TaskStatus, task_fields.status)
        raw_control_state = task_fields.control_state
        try:
            control_state = TaskControlState(str(raw_control_state))
        except ValueError:
            control_state = control_state_for_status(task_status)
        resume_control_state = TaskControlSnapshot(
            task_id=task_id,
            run_id=task_fields.run_id,
            state_version=task_fields.state_version,
            control_state=control_state,
            status=task_status,
        ).as_dict()

        # Compatibility seam into the interaction lifecycle service: both
        # resume paths below (the supports_live_control branch and the
        # bare resume_execution fallback) reach the durable RESUME
        # transition unconditionally, with no notion of a pending question.
        # A task that still has an active native interaction row has one:
        # if this resume's own command payload cannot prove it is the
        # continuation respond() staged, refuse rather than let either path
        # append to or replan around an unanswered question. This runs
        # before agent_service is built (below) so a refused request never
        # pays for constructing one. Gated on tasks.interaction_protocol_
        # version first, though: under a NULL marker the read below returns
        # None regardless of whether an active row exists, so this refusal
        # never fires for that state -- deliberately, matching what the
        # read surface would show for the same task (see
        # active_interaction_id_sync's own docstring).
        #
        # Residual window, named here rather than closed here: this lookup
        # opens and closes its own session, and no lock spans it and either
        # RESUME transition below, so the row can in principle change
        # between this read and the transition. Nothing can drive that
        # change until respond()'s finalizer exists, and that finalizer --
        # not this seam -- is what must own the window when it lands.
        #
        # The read itself is task_interaction_close.active_interaction_id_sync
        # -- the same reader the three legacy-resume injection sites use, so
        # this gate and the close cannot disagree about which row is live.
        active_interaction_id = await run_db_io_cancellation_safe(
            lambda: active_interaction_id_sync(task_id)
        )
        if active_interaction_id is not None:
            receipt_interaction_id = message_data.get("interaction_id")
            receipt_responder_identity = message_data.get("responder_identity")
            # isinstance before comparing, the same shape the cancel
            # command's own state-version guard uses below: `True == 1` and
            # `1.0 == 1` both hold in Python, so a bare `!=` against the
            # row's int id accepts a JSON `true` as the receipt for row 1
            # and a JSON `5.0` as the receipt for row 5. A receipt this
            # seam cannot recognize as the exact int respond() staged is no
            # receipt at all, so the type check is part of the comparison,
            # not a separate validation step a caller could skip.
            if (
                isinstance(receipt_interaction_id, bool)
                or not isinstance(receipt_interaction_id, int)
                or receipt_interaction_id != active_interaction_id
                or not receipt_responder_identity
            ):
                from ..services import ops_signals

                ops_signals.register_degradation(
                    ops_signals.INTERACTION_LEGACY_RESUME_SHIM,
                    f"task {task_id} run {task_fields.run_id}: legacy resume "
                    f"refused, active interaction {active_interaction_id} has "
                    "not been answered through respond()",
                )
                logger.warning(
                    "legacy resume refused for task_id=%s run_id=%s "
                    "interaction_id=%s: active interaction has not been "
                    "answered through respond()",
                    task_id,
                    task_fields.run_id,
                    active_interaction_id,
                )
                reason = (
                    "This task has an unanswered question; answer it before resuming."
                )
                await manager.send_personal_message(
                    {
                        "type": "error",
                        "message": (
                            "This task has an unanswered question; answer it "
                            "before resuming."
                        ),
                        "task": {"id": task_id, **resume_control_state},
                    },
                    websocket,
                )
                return ResumeCommandResult(
                    ResumeCommandOutcome.REJECTED,
                    reason,
                    reason_code="interaction_pending",
                )

        attempt_count = message_data.get("_durable_attempt_count")

        def _log_resume_deferral(classification: str) -> None:
            # Deferrals are deliberately silent to the client (a retry that
            # usually resolves within a second is not a failure), so the
            # command row's ``error`` column is otherwise the only trace. A
            # stuck queue has to be diagnosable from application logs alone.
            logger.info(
                "Deferring resume for task %s run %s: %s (attempt %s)",
                task_id,
                task_fields.run_id,
                classification,
                attempt_count,
            )

        def _defer_for_slot(outcome: ResumeReservationOutcome) -> ResumeCommandResult:
            _log_resume_deferral(
                f"live-control resume slot is unavailable ({outcome.value})"
            )
            return ResumeCommandResult(
                ResumeCommandOutcome.DEFERRED,
                "Resume command is waiting for the live-control resume slot "
                f"({outcome.value})",
            )

        async def _resync_client_to_running_task() -> None:
            """Correct a stale client on the one path nothing else corrects.

            Used only by the already-RUNNING branch. There the row genuinely
            reads ``running``, and no resume is starting, so no
            ``task_resumed`` broadcast will ever arrive to clear the client's
            belief that the task is paused -- the resume control renders on
            local status alone. The coordinator branches are deliberately
            silent instead: their row still reads ``paused`` until the lease
            claim, and their coordinator broadcasts the correction itself.

            No state tuple is supplied. ``send_personal_message`` runs
            ``_with_current_task_control_state``, which attaches the live row
            exactly when the producer supplied none; passing this handler's
            setup snapshot would ship a value already stale by construction.

            ``task_resumed`` rather than ``error``: the command succeeded, and
            ``case "error"`` in ``app-context-chat.tsx`` unconditionally
            appends a failed chat bubble, so an error frame would report a
            success as a failure. ``case "task_resumed"`` is the codebase's
            control-only shape -- it dispatches ``UPDATE_TASK_STATUS`` with
            status, run id, state version and control state, and adds no
            message.

            This type is only usable *here*. ``taskEventMatchesControlState``
            maps ``task_resumed`` to ``["running"]``, and this branch is the
            one place that has already established ``control_state`` is
            ``RUNNING`` -- via a fresh ``task_has_live_runner`` read that also
            requires an unexpired lease on this exact run. On a branch whose
            control state is ``resume_requested`` the same frame would fail
            that match and re-apply the stale status instead.

            A ``task_info`` trace event cannot stand in: the client rebuilds
            the whole task record from that frame, so a partial payload
            blanks the title, description, and model ids.

            Best-effort by construction: the origin socket is same-worker
            only, so a command claimed after a restart or by another worker
            sends this into the discarding sink. The durable outcome, not
            this frame, is the authoritative record.
            """

            try:
                await manager.send_personal_message(
                    {
                        "type": "task_resumed",
                        "message": "Task is already running.",
                        "task": {"id": task_id},
                    },
                    websocket,
                )
            except Exception:
                # A half-open socket must not turn an idempotent success into
                # a durable command failure: the resume really is in flight.
                logger.warning(
                    "Could not deliver the resume-already-in-progress notice "
                    "for task %s",
                    task_id,
                    exc_info=True,
                )

        if control_state is TaskControlState.PAUSE_REQUESTED:
            _log_resume_deferral("pending pause has not settled")
            return ResumeCommandResult(
                ResumeCommandOutcome.DEFERRED,
                "Resume command is waiting for the pending pause to settle",
            )

        admission_state = background_task_manager.resume_admission_state(
            task_id,
            expected_run_id=task_fields.run_id,
        )
        if admission_state is ResumeReservationOutcome.COORDINATOR_RUNNING:
            logger.info(
                "Task %s already has a coordinator for run %s",
                task_id,
                task_fields.run_id,
            )
            # No client frame here, and this is a trade rather than a pure
            # win. A registration lasts the whole resumed execution, so for
            # most of this window the row already reads ``running`` and a
            # frame would have carried the correction. Only the slice before
            # the lease claim reads ``paused`` -- the RESUME_REQUESTED
            # transition writes just ``control_state`` -- and there a frame
            # re-confirms the state it was meant to correct.
            #
            # The correction is instead the coordinator's own ``task_resumed``
            # broadcast at lease commit. That is a single unrepeated event,
            # where re-clicking Resume used to be retriable, so a client that
            # was momentarily not in ``connections_for_task`` when it fired
            # stays stale until it reloads. Tracked with the rest of the
            # coordinator-evidence gaps in #1781.
            return ResumeCommandResult(ResumeCommandOutcome.ALREADY_IN_PROGRESS)
        if admission_state is not None:
            # Only RESERVATION_HELD and SHUTTING_DOWN remain: RESERVED is
            # never returned by an inspection and COORDINATOR_RUNNING
            # returned above. Both are uncertain rather than terminal, so
            # they defer for a durable retry.
            return _defer_for_slot(admission_state)

        if task_status is TaskStatus.RUNNING:
            live_runner = await run_db_io_cancellation_safe(
                lambda: task_has_live_runner(
                    task_id,
                    expected_run_id=task_fields.run_id,
                )
            )
            if control_state is TaskControlState.RUNNING and live_runner:
                logger.info(
                    "Task %s run %s has an active execution lease",
                    task_id,
                    task_fields.run_id,
                )
                await _resync_client_to_running_task()
                return ResumeCommandResult(ResumeCommandOutcome.ALREADY_IN_PROGRESS)
            if live_runner:
                # The lease is live; it is the control state that has not
                # settled. Naming the lease here would send whoever reads the
                # log after the wrong thing.
                _log_resume_deferral(
                    "running task holds a live lease but its control state is "
                    f"{control_state.value}"
                )
                return ResumeCommandResult(
                    ResumeCommandOutcome.DEFERRED,
                    "Resume command is waiting for the running task's control "
                    "state to settle",
                )
            _log_resume_deferral("running task has no live lease yet")
            return ResumeCommandResult(
                ResumeCommandOutcome.DEFERRED,
                "Resume command is waiting for running-task lease recovery",
            )
        if task_status in {
            TaskStatus.PAUSED,
            TaskStatus.WAITING_FOR_USER,
        } and await run_db_io_cancellation_safe(
            lambda: task_has_live_foreign_runner(task_id)
        ):
            # The idempotency evidence above only classifies RUNNING rows,
            # but a settling turn commits PAUSED/WAITING_FOR_USER while still
            # holding its lease: the finalizer writes the status and the lease
            # columns are only cleared later, by ``finish_turn``. Scheduling
            # into that window steals a live lease, and the previous owner's
            # ownership-fenced settlement then matches no row and silently
            # skips its delivery reconciliation. Deferring is bounded: a lease
            # on a non-RUNNING row cannot be refreshed, so it expires within
            # ``XAGENT_TASK_LEASE_TTL_SECONDS``. Same-process holds are not
            # foreign and are already serialised through ``previous_task``.
            _log_resume_deferral("another process still holds a live task lease")
            return ResumeCommandResult(
                ResumeCommandOutcome.DEFERRED,
                "Resume command is waiting for the active task lease owner",
                # Same wording, and the same reason for it, as the PAUSE and
                # CANCEL arms of the shared guard this branch replaces for
                # RESUME. It has to survive the redaction chokepoint for the
                # same reason theirs does.
                client_visible=True,
            )

        # Scope resolution is a scheduling prerequisite, not evidence that an
        # execution already exists. Idempotent and deferred outcomes above
        # deliberately avoid this potentially expensive off-turn work: on an
        # agent-cache miss, or a cached-scope-fingerprint mismatch,
        # ``get_agent_for_task`` below builds a fresh agent from this value,
        # which can materialize a workspace directory tree and acquire a
        # sandbox lease.
        #
        # ``resolve_execution_scope_off_turn`` has three distinct outcomes:
        # resolver authoritative with a snapshot disagreement downgrades to
        # the resolver's own answer; resolver abstention with a widening
        # snapshot re-raises, refusing the resume outright; resolver
        # abstention with a narrowing snapshot returns the snapshot itself,
        # which is persisted, client-influenceable data trusted only because
        # it was already validated as a narrowing of what the resolver
        # granted. The turn scheduled below is a different consumer and gets
        # ``EXECUTION_SCOPE_NOT_PROVIDED`` instead, so it resolves its own
        # scope fail-closed rather than inheriting this off-turn result. The
        # equivalent call in ``_handle_pause_task_unserialized`` carries the
        # same reasoning in full.
        resolved_execution_scope = await run_db_io_cancellation_safe(
            lambda: resolve_execution_scope_off_turn(task_id)
        )

        agent_service = await get_agent_manager().get_agent_for_task(
            task_id,
            None,
            user=task_setup_snapshot.runtime_user,
            task_setup_snapshot=task_setup_snapshot,
            task_owner_user_id=task_owner_user_id,
            resolved_execution_scope=resolved_execution_scope,
        )
        if getattr(agent_service, "supports_live_control", lambda: False)():
            if task_status not in {TaskStatus.PAUSED, TaskStatus.WAITING_FOR_USER}:
                reason = "Task is not paused and cannot be resumed."
                await manager.send_personal_message(
                    {
                        "type": "error",
                        "message": "Task is not paused and cannot be resumed.",
                        "task": {"id": task_id, **resume_control_state},
                    },
                    websocket,
                )
                return ResumeCommandResult(
                    ResumeCommandOutcome.REJECTED,
                    reason,
                    reason_code="not_resumable",
                )
            reservation = background_task_manager.try_reserve_resume(
                task_id,
                expected_run_id=task_fields.run_id,
            )
            if reservation is ResumeReservationOutcome.COORDINATOR_RUNNING:
                logger.info(
                    "Task %s already has a registered resume coordinator",
                    task_id,
                )
                # Silent for the same reason, and with the same trade, as
                # the admission-state branch above.
                return ResumeCommandResult(ResumeCommandOutcome.ALREADY_IN_PROGRESS)
            if reservation is not ResumeReservationOutcome.RESERVED:
                return _defer_for_slot(reservation)
            resume_snapshot: Any | None = None
            bg_task: asyncio.Task[None] | None = None
            try:
                resume_snapshot = await task_execution_controller.transition(
                    task_id,
                    TaskControlState.RESUME_REQUESTED,
                    expected_run_id=task_fields.run_id,
                    # Every admission decision above was made from the setup
                    # snapshot. ``expected_run_id`` alone cannot notice a
                    # writer that moved the row while preserving its run id,
                    # and the reachable such writer is a competing resume,
                    # not a cancel: ``_acquire_reply_prelease_sync`` and
                    # ``_acquire_a2a_resume_prelease_sync`` come in over HTTP,
                    # bypassing the durable queue entirely, and
                    # ``acquire_task_lease_no_commit`` keeps the existing run
                    # (``candidate_run_id = expected_run_id or uuid4()``)
                    # while bumping ``state_version``. Without this fence a
                    # v1 reply and a websocket Resume landing together both
                    # transition the row and schedule two coordinators
                    # against one lease.
                    #
                    # An A2A cancel writes the same shape -- FAILED, lease
                    # cleared, run id preserved -- but cannot actually
                    # interleave here: cancels reach the DB only through the
                    # durable queue, and ``_unfinished_earlier_command``
                    # serialises commands per task, so a PROCESSING resume
                    # blocks the cancel from being claimed at all.
                    expected_state_version=task_fields.state_version,
                )
            except StaleTaskStateVersionError as exc:
                # Only the version fence lands here. A rotated run raises the
                # base class and keeps its old meaning -- the command targets
                # an execution that no longer exists, nothing will make it
                # valid, so it propagates and stays terminal.
                #
                # The fence added a trigger with the opposite meaning: the row
                # is still this run's, someone simply wrote first, and the
                # writer is overwhelmingly a competing resume -- the same
                # situation the RUNNING branch calls an idempotent success.
                # Letting that through as terminal would hand one interleaving
                # of that race the harshest outcome in the handler, and
                # (because these are RuntimeErrors) leak the raw diagnostic to
                # the client through the arm below on the way out. That arm
                # still does so for the rotated-run raise -- deliberate, per
                # the #1479 note on it -- so what this closes is the leak the
                # fence itself introduced, not the arm.
                #
                # Deferring re-runs the whole admission decision against a
                # fresh row, so it lands on whichever outcome is actually true
                # rather than guessing from a snapshot already known to be
                # stale.
                #
                # One imprecision is deliberate: a rotated run caught by the
                # SQL fence rather than the pre-check cannot be told apart
                # from a moved version -- the UPDATE carried both predicates
                # and reports only that it matched nothing -- so it lands here
                # too and defers once. That costs a single retry, because the
                # dispatcher re-reads the run before re-entering this handler
                # and rejects a genuinely rotated one terminally.
                background_task_manager.release_resume_reservation(task_id)
                _log_resume_deferral(f"row moved under the admission snapshot ({exc})")
                return ResumeCommandResult(
                    ResumeCommandOutcome.DEFERRED,
                    "Resume command is waiting to re-read a task row that "
                    "changed while it was being admitted",
                )
            except BaseException:
                # Everything else the transition can fail with -- a rotated
                # run, a deleted row, a DB error -- keeps its previous
                # meaning and propagates. The reservation still has to go
                # back: this arm exists only because splitting the deferral
                # case out of the block below would otherwise let these
                # escape without releasing it.
                background_task_manager.release_resume_reservation(task_id)
                raise
            try:
                previous_task = background_task_manager.running_tasks.get(task_id)
                bg_task = asyncio.create_task(
                    execute_resume_background(
                        task_id=task_id,
                        agent_service=agent_service,
                        task_owner_user_id=task_owner_user_id,
                        expected_run_id=resume_snapshot.run_id,
                        previous_task=previous_task,
                        # Not `resolved_execution_scope`: that value is the
                        # off-turn downgrade used above to obtain
                        # `agent_service` (which, on an agent-cache miss, may
                        # itself have built the workspace/sandbox from it --
                        # see the comment above `resolved_execution_scope`).
                        # The scheduled turn selects the namespace its own
                        # output lands under, so it explicitly gets
                        # `EXECUTION_SCOPE_NOT_PROVIDED` and runs its own
                        # fail-closed resolution instead of inheriting a
                        # disputed answer.
                        resolved_execution_scope=EXECUTION_SCOPE_NOT_PROVIDED,
                    )
                )
                background_task_manager.register_reserved_resume(
                    task_id,
                    bg_task,
                    run_id=resume_snapshot.run_id,
                )
            except BaseException:
                if bg_task is not None:
                    bg_task.cancel()
                background_task_manager.release_resume_reservation(task_id)
                if resume_snapshot is not None:
                    try:
                        await asyncio.shield(
                            task_execution_controller.transition(
                                task_id,
                                (
                                    TaskControlState.WAITING_FOR_USER
                                    if resume_snapshot.status
                                    == TaskStatus.WAITING_FOR_USER
                                    else TaskControlState.PAUSED
                                ),
                                expected_run_id=resume_snapshot.run_id,
                                expected_state_version=resume_snapshot.state_version,
                            )
                        )
                    except (StaleTaskRunError, ValueError) as rollback_exc:
                        # Someone else moved the row since the transition this
                        # is undoing -- a cancel, the coordinator's own lease
                        # claim, or a hard delete, which surfaces as the bare
                        # ValueError ``transition_task_control_state_sync``
                        # raises for a missing row. Their outcome wins;
                        # rolling back would resurrect the state we are
                        # abandoning. Swallowed rather than raised so it
                        # cannot mask the failure that brought us here, which
                        # is the whole point of this arm -- so it has to
                        # cover every way the rollback can legitimately fail
                        # to find its row, not just the version fence.
                        # The reason has to come from the exception, not from
                        # the version fence: this arm also catches a row that
                        # was deleted outright, and reporting that as an
                        # ordinary version-fence skip sends whoever reads the
                        # log after a race that did not happen.
                        logger.info(
                            "Skipped resume rollback for task %s (expected "
                            "state version %s): %s",
                            task_id,
                            resume_snapshot.state_version,
                            rollback_exc,
                            exc_info=True,
                        )
                raise
            logger.info(f"Task {task_id} v2 resume scheduled")
            return ResumeCommandResult(ResumeCommandOutcome.SCHEDULED)

        # Unreachable: ``supports_live_control`` is defined once, on
        # ``AgentService``, and returns True unconditionally, so the block above
        # always returns or raises. Kept as a loud failure rather than a silent
        # fallthrough in case a future agent type opts out of live control.
        raise RuntimeError(
            f"Agent for task {task_id} does not support live execution control"
        )

    except (ValueError, KeyError, TypeError) as e:
        # Data validation error
        reason = str(e)
        logger.error(
            "Data validation error resuming task %s: %s", task_id, e, exc_info=True
        )
        await manager.send_personal_message(
            {"type": "error", "message": client_safe_error_message(e)}, websocket
        )
        return ResumeCommandResult(
            ResumeCommandOutcome.REJECTED,
            reason,
            reason_code="invalid_command_payload",
        )
    except RuntimeError as e:
        logger.error("Runtime error resuming task %s: %s", task_id, e, exc_info=True)
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
        raise
    except Exception as e:
        # Other errors, re-raise
        logger.error("Unexpected error resuming task %s: %s", task_id, e)
        raise


class _DiscardingCommandWebSocket:
    """Minimal sink used when a recovered command has no originating socket."""

    async def send_text(self, _message: str) -> None:
        return None


@dataclass(frozen=True)
class _CommandActor:
    id: int
    is_admin: bool


def _load_command_actor(actor_user_id: int | None) -> _CommandActor:
    if actor_user_id is None:
        raise ValueError("Task command has no actor")
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        user_row = (
            db.query(User.id, User.is_admin).filter(User.id == actor_user_id).first()
        )
        if user_row is None:
            raise ValueError(f"Task command actor {actor_user_id} no longer exists")
        return _CommandActor(id=int(user_row[0]), is_admin=bool(user_row[1]))


async def _execute_durable_task_command(
    command: ClaimedTaskCommand,
) -> dict[str, Any] | None:
    """Apply one DB-claimed command using the existing transport adapters.

    Personal replies go only to the registered origin socket, verified to
    still be connected to this task. Origin is never inferred from task
    membership, actor id, or connection order: after a crash, a handoff to a
    different worker, or a disconnect, personal detail is discarded while
    task-level state/error events are still broadcast normally.
    """

    if command.kind == TaskCommandKind.MESSAGE and "scope" in command.payload:
        # The first-party chat adapter below resolves its actor, origin
        # socket, and delivery ledger from first-party rows. A MESSAGE that
        # names a scope is not that command: "external" belongs to the
        # embedding application's execution core, reached through the
        # registered seam, and any other value names a core that does not
        # exist here -- running the first-party core against it would inject
        # the message while attributing it to the wrong audience. Equality
        # rather than set membership so an unhashable payload value lands in
        # the terminal rejection instead of raising ``TypeError`` into the
        # retry path.
        scope_value = command.payload["scope"]
        if scope_value != EXTERNAL_COMMAND_SCOPE:
            raise TaskCommandRejected(
                f"Message command {command.command_id} names task scope "
                f"{scope_value!r}, which has no execution core",
                reason="unsupported_scope",
            )
        return await execute_external_task_input_command(command)

    websocket: Any = _command_origins.resolve(command.command_id, command.task_id)
    if websocket is None:
        websocket = _DiscardingCommandWebSocket()
    message_data = dict(command.payload)
    message_data.update(
        {
            "_durable_ack_sent": True,
            "_durable_attempt_count": command.attempt_count,
            "_durable_target_run_id": command.target_run_id,
        }
    )
    if command.kind != TaskCommandKind.CANCEL:
        user = await run_db_io_cancellation_safe(
            lambda: _load_command_actor(command.actor_user_id)
        )
        message_data.update({"user": user, "user_id": int(user.id)})
    if command.kind != TaskCommandKind.MESSAGE and command.target_run_id is not None:
        current_run_id = await run_db_io_cancellation_safe(
            lambda: _load_command_task_run_id(command.task_id)
        )
        if current_run_id != command.target_run_id:
            raise TaskCommandRejected(
                f"Task run changed before {command.kind.value} command "
                f"{command.command_id} was applied",
                reason="stale_run",
            )
    if command.kind in {
        TaskCommandKind.PAUSE,
        TaskCommandKind.CANCEL,
    } and await run_db_io_cancellation_safe(
        lambda: task_has_live_foreign_runner(command.task_id)
    ):
        raise ClientVisibleTaskCommandDeferred(
            f"{command.kind.value.title()} command {command.command_id} is waiting "
            "for the active task lease owner"
        )

    resume_result: ResumeCommandResult | None = None
    if command.kind == TaskCommandKind.MESSAGE:
        await _handle_chat_message_unserialized(
            websocket, command.task_id, message_data
        )
        if message_data.get("_durable_command_defer") == command.command_id:
            # This marker is mutually exclusive with commit-outcome-unknown:
            # the handler returns immediately after recording contention.
            raise TaskCommandDeferred(
                str(message_data["_durable_command_defer_reason"]),
                resend_safe=(
                    # Every settled contention increments defer_count once.
                    # Equality proves there is no extra expired/failed claim
                    # whose worker might still resume and inject after this
                    # attempt observed no delivery row.
                    #
                    # The equality couples two write sites: claiming is the
                    # only writer of attempt_count and defer_task_command the
                    # only writer of defer_count. retry_failed_task_command
                    # resets defer_count but not attempt_count, so an
                    # operator-retried command can never prove safety again
                    # (the safe direction for a duplicate-send decision).
                    command.attempt_count == command.defer_count + 1
                    and message_data.get("_durable_command_defer_unsafe")
                    != command.command_id
                ),
            )
        if message_data.get("_commit_outcome_unknown") == command.command_id:
            raise ClientVisibleTaskCommandDeferred(
                f"Message {command.command_id} has an unknown commit outcome"
            )
        if message_data.get("_registered_turn_handoff") == command.command_id:
            return {
                "task_id": command.task_id,
                "command_id": command.command_id,
                "kind": command.kind.value,
            }
        delivery_status = await run_db_io_cancellation_safe(
            lambda: _load_command_message_delivery_status(
                command.task_id,
                command.command_id,
            )
        )
        if delivery_status == DELIVERY_PENDING:
            raise ClientVisibleTaskCommandDeferred(
                f"Message {command.command_id} is waiting for runtime injection"
            )
        if delivery_status == DELIVERY_FAILED:
            raise TaskCommandRejected(
                f"Message {command.command_id} could not be applied"
            )
    else:
        try:
            if command.kind == TaskCommandKind.PAUSE:
                await _handle_pause_task_unserialized(
                    websocket,
                    command.task_id,
                    message_data,
                )
            elif command.kind == TaskCommandKind.RESUME:
                resume_result = await _handle_resume_task_unserialized(
                    websocket,
                    command.task_id,
                    message_data,
                )
                if resume_result.outcome is ResumeCommandOutcome.DEFERRED:
                    deferral_message = (
                        resume_result.reason or "Resume command will be retried"
                    )
                    if resume_result.client_visible:
                        raise ClientVisibleTaskCommandDeferred(deferral_message)
                    raise TaskCommandDeferred(deferral_message)
                if resume_result.outcome is ResumeCommandOutcome.REJECTED:
                    raise TaskCommandRejected(
                        resume_result.reason or "Resume command was rejected",
                        reason=resume_result.reason_code,
                    )
            elif command.kind == TaskCommandKind.CANCEL:
                agent_id_value = message_data.get("agent_id")
                if agent_id_value is None:
                    raise ValueError(
                        "Agent ID is missing or null in cancel command payload"
                    )
                try:
                    agent_id = int(agent_id_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Agent ID {agent_id_value!r} is invalid in cancel "
                        "command payload"
                    ) from exc
                target_state_version = message_data.get("target_state_version")
                if isinstance(target_state_version, bool) or not isinstance(
                    target_state_version,
                    int,
                ):
                    raise TaskCommandRejected(
                        f"Cancel command {command.command_id} has no exact "
                        "state-version target",
                        reason="stale_run",
                    )
                # The A2A execution core loads its target as an A2A task, so
                # a cancel for any other task source needs its own core. The
                # scope names which one, and the absence of the key is itself
                # a value: it is the only shape this command had before the
                # external core existed, so it stays on the A2A path. Any
                # other value names a core that does not exist here, and
                # silently running the A2A one against it would cancel
                # nothing while reporting success.
                if "scope" not in message_data:
                    scope_value = EXTERNAL_COMMAND_SCOPE_ABSENT
                else:
                    scope_value = message_data["scope"]
                # Identity and equality checks rather than set membership:
                # an unhashable payload value (a dict or list) must land in
                # the same terminal rejection, not raise ``TypeError`` into
                # the retry path.
                if (
                    scope_value is not EXTERNAL_COMMAND_SCOPE_ABSENT
                    and scope_value != EXTERNAL_COMMAND_SCOPE
                ):
                    raise TaskCommandRejected(
                        f"Cancel command {command.command_id} names task scope "
                        f"{scope_value!r}, which has no execution core",
                        reason="unsupported_scope",
                    )
                async with task_execution_controller.command(command.task_id):
                    if scope_value == EXTERNAL_COMMAND_SCOPE:
                        await cancel_external_task_unserialized(
                            task_id=command.task_id,
                            agent_id=agent_id,
                            expected_run_id=command.target_run_id,
                            expected_state_version=target_state_version,
                            turn_id=_command_turn_id(command.task_id, message_data),
                        )
                    else:
                        from .a2a import _cancel_task_unserialized

                        await _cancel_task_unserialized(
                            task_id=command.task_id,
                            agent_id=agent_id,
                            expected_run_id=command.target_run_id,
                            expected_state_version=target_state_version,
                        )
            else:  # pragma: no cover - enum construction rejects this earlier
                raise ValueError(f"Unsupported task command kind: {command.kind}")
        except StaleTaskRunError as exc:
            raise TaskCommandRejected(str(exc), reason="stale_run") from exc
    durable_error = message_data.get("_durable_command_error")
    if isinstance(durable_error, str) and durable_error:
        raise TaskCommandRejected(durable_error)
    result = {
        "task_id": command.task_id,
        "command_id": command.command_id,
        "kind": command.kind.value,
    }
    if resume_result is not None:
        result["resume_outcome"] = resume_result.outcome.value
    return result


# "no scope key at all" needs a value the scope check can compare against
# and no payload can ever carry. A JSON payload cannot hold this object, so
# a producer cannot forge the pre-external shape by writing a string.
EXTERNAL_COMMAND_SCOPE_ABSENT = object()


def _command_scope(command: ClaimedTaskCommand) -> str | None:
    """The scope a command payload names, or ``None`` when it names none."""

    scope = command.payload.get("scope")
    return scope if isinstance(scope, str) else None


def _command_turn_id(task_id: int, message_data: dict[str, Any]) -> str | None:
    """The turn a command names, or ``None`` when it names none usably.

    The value only picks which delivery row a cancel closes. A producer that
    writes something other than a non-empty string is a bug, but refusing
    the stop over it would leave the visitor's turn running, so the target
    falls back to the running turn and the bug is logged rather than raised.
    """

    if "turn_id" not in message_data:
        return None
    raw = message_data["turn_id"]
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    logger.warning(
        "task %s cancel command carries an unusable turn_id of type %s; "
        "falling back to the running turn's delivery row",
        task_id,
        type(raw).__name__,
    )
    return None


async def _broadcast_terminal_command_error(
    command: ClaimedTaskCommand,
    error: BaseException,
) -> None:
    scope = _command_scope(command)
    # Two things separate an external-scope cancel from every other command
    # that exhausts its budget, and both come from who reads the frame. The
    # wording has to be true about the turn, which takes reading the task.
    # And ``command_kind``/``command_id`` are operator handles: an anonymous
    # visitor cannot act on them and should not be shown the durable command
    # identity of a task they do not own. Two payload literals rather than
    # one built and trimmed: the client-safe guard only inspects dict
    # literals passed straight to the sink, and a payload assembled in a
    # variable would drop this site out of its view entirely.
    if is_external_cancel_command(kind=command.kind.value, scope=scope):
        task_status = await _load_terminal_command_task_status(command.task_id)
        await manager.broadcast_to_task(
            {
                "type": "agent_error",
                "message": client_safe_task_command_failure(
                    command.kind,
                    error,
                    scope=scope,
                    task_status=task_status,
                ),
                "task_id": command.task_id,
                "timestamp": datetime.now(timezone.utc).timestamp(),
            },
            command.task_id,
        )
        return
    await manager.broadcast_to_task(
        {
            "type": "agent_error",
            # A blessed constructor rather than an f-string at the call
            # site: the guard cannot see inside an interpolation. The kind
            # also travels as a structured field for consumers that want it.
            "message": client_safe_task_command_failure(
                command.kind,
                error,
                scope=scope,
            ),
            "command_kind": command.kind.value,
            "task_id": command.task_id,
            "command_id": command.command_id,
            "timestamp": datetime.now(timezone.utc).timestamp(),
        },
        command.task_id,
    )


async def _terminal_command_event_draft(
    command: ClaimedTaskCommand,
    error: BaseException,
) -> TerminalTaskEventDraft:
    """Build safe presentation metadata; disposition code persists it."""

    scope = _command_scope(command)
    if is_external_cancel_command(kind=command.kind.value, scope=scope):
        try:
            task_status = await _load_terminal_command_task_status(command.task_id)
        except Exception as exc:
            logger.warning(
                "Could not classify external terminal command outcome; "
                "using conservative client message task_id=%s error_type=%s",
                command.task_id,
                type(exc).__name__,
            )
            task_status = None
        return TerminalTaskEventDraft(
            message_code=(
                TerminalTaskEventMessageCode.EXTERNAL_TURN_INTERRUPTED
                if task_status in {TaskStatus.COMPLETED, TaskStatus.FAILED}
                else TerminalTaskEventMessageCode.EXTERNAL_CANCEL_NOT_APPLIED
            ),
            resend_safe=False,
            include_command_identity=False,
        )
    return TerminalTaskEventDraft(
        message_code=(
            TerminalTaskEventMessageCode.TASK_COMMAND_DEFERRED
            if isinstance(error, TaskCommandDeferred)
            else TerminalTaskEventMessageCode.TASK_COMMAND_FAILED
        ),
        resend_safe=(
            error.resend_safe if isinstance(error, TaskCommandDeferred) else False
        ),
        # The disclosure rule the cancel branch above states — an anonymous
        # external audience cannot act on durable command identity and is not
        # shown it — holds for every external-scope command, including the
        # external input MESSAGEs routed through the registered seam. This
        # rebind runs after the executor's own draft and would otherwise
        # silently restore the identity the executor withheld.
        include_command_identity=scope != EXTERNAL_COMMAND_SCOPE,
    )


async def execute_durable_task_command(
    command: ClaimedTaskCommand,
) -> dict[str, Any] | None:
    """Apply one command and expose only terminal transport failures to clients."""

    try:
        result = await _execute_durable_task_command(command)
    except TaskCommandDeferred as exc:
        if command.defer_count + 1 >= MAX_COMMAND_DEFERS:
            _command_origins.discard_command(command.command_id, command.task_id)
            bind_terminal_event_draft(
                exc,
                await _terminal_command_event_draft(command, exc),
            )
            await _broadcast_terminal_command_error(command, exc)
        # A deferral that will retry keeps its origin entry.
        raise
    except TaskCommandRejected:
        # Rejections come from handlers that already expose their durable
        # domain-level outcome. The dispatcher makes them terminal immediately.
        _command_origins.discard_command(command.command_id, command.task_id)
        raise
    except Exception as exc:
        if command.failure_count + 1 >= MAX_COMMAND_FAILURES:
            _command_origins.discard_command(command.command_id, command.task_id)
            bind_terminal_event_draft(
                exc,
                await _terminal_command_event_draft(command, exc),
            )
            await _broadcast_terminal_command_error(command, exc)
        raise
    _command_origins.discard_command(command.command_id, command.task_id)
    return result


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


async def _load_terminal_command_task_status(task_id: int) -> TaskStatus | None:
    """The task's status right now, or ``None`` when it cannot be read.

    This read only chooses wording for a notification that is already the
    last act of a terminal command, and it runs inside the ``except`` bodies
    of ``execute_durable_task_command``. An exception raised here would
    replace the failure that dispatcher is handling, turning "the command
    failed" into "the database failed", so an unreadable row - deleted, pool
    exhausted, database down - is answered as ``None`` and logged.
    ``CancelledError`` is deliberately not caught: a cancelled dispatcher
    still has to unwind.
    """

    def _read() -> TaskStatus | None:
        SessionLocal = get_session_local()
        with SessionLocal() as db:
            task = db.query(Task).filter(Task.id == task_id).first()
            return task.status if task is not None else None

    try:
        return await run_db_io_cancellation_safe(_read)
    except Exception:
        logger.warning(
            "could not read task %s status while wording a terminal command broadcast",
            task_id,
            exc_info=True,
        )
        return None


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
    from ..services.builder_chat_runtime import load_builder_chat_runtime_inputs
    from .agents import apply_user_voice, voice_from_runtime_user

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
        await websocket.send_text(
            json.dumps({"type": "error", "message": client_safe_error_message(e)})
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
