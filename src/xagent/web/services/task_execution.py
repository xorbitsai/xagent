"""Background task execution, resume, and settlement without API route dependencies."""

import asyncio
import enum
import logging
import re
import shutil
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Union,
    cast,
    overload,
)
from urllib.parse import unquote

from sqlalchemy import case, func, or_, update
from sqlalchemy.orm import Session

from ...config import (
    get_uploads_dir,
)
from ...core.agent.checkpoint import (
    CheckpointAccessRefusedError,
    CheckpointUnavailableError,
)
from ...core.agent.runner import UserMessageInjectionOutcome
from ...core.execution_scope import (
    EXECUTION_SCOPE_NOT_PROVIDED,
    ExecutionScope,
    ExecutionScopeContext,
    ExecutionScopeNotProvided,
    resolve_execution_scope,
)
from ...core.file_ref import build_file_ref
from ..models.database import (
    get_db,
    get_session_local,
)
from ..models.task import Task, TaskStatus
from ..models.uploaded_file import UploadedFile
from .llm_utils import AutoModelUnavailableError
from .task_events import DeliveryNotifier, publish_task_event
from .task_lease_service import (
    lock_task_lease_for_settlement_no_commit,
    lock_task_lease_no_commit,
    task_lease_attempt_predicate,
)

if TYPE_CHECKING:
    from .task_setup_snapshot import TaskSetupSnapshot
from ...core.file_storage.keys import (
    build_task_output_storage_key,
)
from ..user_isolated_memory import UserContext
from ..utils.json_payload_sanitizer import sanitize_json_payload
from .assistant_history_safety import (
    ASSISTANT_RESPONSE_MESSAGE_TYPE,
    TASK_FAILURE_MESSAGE_TYPE,
    assistant_history_values_for_persistence,
    safe_str,
)
from .chat_history_service import (
    DELIVERY_COMPLETED,
    DELIVERY_DISPATCHED,
    DELIVERY_FAILED,
    mark_user_message_delivery_sync,
)
from .client_error_messages import (
    CLIENT_SAFE_TASK_FAILURE,
    CLIENT_SAFE_VALIDATION_ERROR,
    CONNECTOR_RUNTIME_CLIENT_ERROR_CODES,
    ClientErrorCode,
    client_error_message,
)
from .db_runtime import (
    await_task_settlement,
    drain_async_task_cancellation_safe,
    is_database_pool_timeout,
    propagate_deferred_cancellation,
    run_db_io_cancellation_safe,
)
from .file_reference_output_service import (
    reconcile_assistant_file_references,
)
from .file_turn import (
    normalize_filename,
)
from .mcp_runtime import (
    MCPActorExecutionIdentity,
    MCPBuiltinOAuthActorPolicy,
)
from .task_execution_controller import (
    TaskControlSnapshot,
    TaskControlState,
    apply_task_control_transition,
    task_execution_controller,
)
from .task_interaction_close import (
    clear_interaction_marker_if_unpaired,
    close_legacy_resume_interaction_sync,
)
from .task_lease_service import (
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
from .uploaded_file_store import (
    StagedUploadedFile,
    SupersededObjectCleanupClaim,
    UploadedFileStore,
    UploadedFileVersionSnapshot,
    cleanup_superseded_uploaded_file_objects,
    compensate_staged_uploaded_files,
    snapshot_uploaded_file_version,
    stage_uploaded_file_from_local_path,
)
from .workforce_runtime import (
    sync_workforce_run_status,
)

logger = logging.getLogger(__name__)


_pause_accepted_task_ids: set[int] = set()


def _mark_task_pause_accepted(task_id: int) -> None:
    _pause_accepted_task_ids.add(int(task_id))


def _clear_task_pause_accepted(task_id: int) -> None:
    _pause_accepted_task_ids.discard(int(task_id))


def _is_task_pause_accepted(task_id: int) -> bool:
    return int(task_id) in _pause_accepted_task_ids


def _waiting_or_paused_event_fields(status: TaskStatus) -> tuple[str, str]:
    """Event type and default message for a task settled at WAITING_FOR_USER
    or PAUSED. Shared by the live-lease restore broadcast and the
    historical-replay status reassertion so both present identical labels
    for the same status."""

    if status == TaskStatus.WAITING_FOR_USER:
        return "task_waiting_for_user", "Task waiting for user response"
    return "task_paused", "Task paused"


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
    *,
    code: str | None = None,
    run_id: str | None = None,
    state_version: int | None = None,
    control_state: str | None = None,
) -> dict[str, Any]:
    """Shape an error event after the exact lease owner commits FAILED.

    ``code`` is written only when it survives validation, so a caller that
    passes none still gets the same six-key frame, and a caller that passes
    something unusable gets that same frame rather than an exception. This
    runs on the reporting path of an already-failed task, and the one call
    site that passes this argument evaluates it inside the ``except
    Exception`` that only logs a failed broadcast -- so raising here would
    cost the terminal frame outright and leave the user on the silent
    failure this path exists to remove. A bad optional argument costs that
    argument and nothing else. The rejection is logged with its stack.

    ``code`` must be a connector-runtime code or ``AUTO_MODEL_UNAVAILABLE``.
    """

    # Python annotations are not enforced at run time, so the mypy gate on the
    # signature above is not the whole door: a caller that routes through Any
    # (a dict from JSON, a **kwargs splat) type-checks clean and would reach
    # this function with a value of the wrong shape. Name the contract here
    # instead.
    #
    # ConnectorRuntimeError types its code as a bare str and stores it
    # unvalidated, so "only the eight module constants reach here" is a fact
    # about today's raise sites, not a property the code holds. The type
    # check comes first for the same reason: annotations are not enforced,
    # and an unhashable value would raise inside the membership test on a
    # path whose whole point is that it never raises.
    if code is not None and (
        not isinstance(code, str)
        or (
            code not in CONNECTOR_RUNTIME_CLIENT_ERROR_CODES
            and code != ClientErrorCode.AUTO_MODEL_UNAVAILABLE.value
        )
    ):
        logger.error(
            "task_id=%s component=terminal-error-frame dropped=code "
            "value=%r; the frame is still sent without it",
            task_id,
            code,
            stack_info=True,
        )
        code = None

    event: dict[str, Any] = {
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
    if code is not None:
        event["code"] = code
    if run_id is not None:
        event["task"]["run_id"] = run_id
    if state_version is not None:
        event["task"]["state_version"] = state_version
    if control_state is not None:
        event["task"]["control_state"] = control_state
    return event


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
    if isinstance(error, AutoModelUnavailableError):
        return client_error_message(ClientErrorCode.AUTO_MODEL_UNAVAILABLE)
    if not isinstance(error, ClientVisibleError):
        return fallback
    message = str(error)
    return message if message.strip() else fallback


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
                from .chat_history_service import (
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
    from .chat_history_service import persist_assistant_message

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
        from .task_lease_service import current_task_lease

        lease = current_task_lease()
        if lease is not None and (
            lease.task_id != task_id or not lock_task_lease_no_commit(db, lease)
        ):
            raise TaskLeaseLostError("Outbound event producer lost its task lease")
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
                    source_event_id=str(trace_event.event_id),
                )

        db.commit()
    except Exception as exc:
        db.rollback()
        if isinstance(exc, TaskLeaseLostError):
            raise
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
            await publish_task_event(
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
        await run_db_io_cancellation_safe(
            lambda: _persist_agent_outbound_event(task_id, event)
        )
        await publish_task_event(event, task_id)

    return handle_outbound_message


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


def _task_user_id(task: Any) -> int | None:
    user_id = getattr(task, "user_id", None)
    if user_id is None:
        return None
    return int(cast(Any, user_id))


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
    from .chat_history_service import persist_assistant_message_no_commit

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
                lock_task_lease_for_settlement_no_commit(finalize_db, task_lease)
                task_updated = (
                    task_query.filter(
                        Task.runner_id == task_lease.runner_id,
                        task_lease_attempt_predicate(task_lease),
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
    from .task_execution_context_service import (
        materialize_task_execution_recovery_state,
    )
    from .task_setup_snapshot import load_task_setup_snapshot_sync

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
        mcp_actor_execution_identity: MCPActorExecutionIdentity | None = None
        if (
            mcp_runtime_authorization_policy is not None
            and task_lease is not None
            and task_lease.task_id == task_id
        ):
            try:
                mcp_actor_execution_identity = MCPActorExecutionIdentity(
                    task_id=task_id,
                    run_id=task_lease.run_id,  # type: ignore[arg-type]
                    turn_id=context_dict.get("turn_id"),  # type: ignore[arg-type]
                    lease_attempt_id=task_lease.attempt_id,  # type: ignore[arg-type]
                )
            except ValueError:
                # Only execution-scoped actor stdio requires this complete
                # fence. Per-call stdio and existing OAuth remain available.
                mcp_actor_execution_identity = None
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
                mcp_actor_execution_identity=mcp_actor_execution_identity,
                resolved_execution_scope=execution_scope,
            )
            if hasattr(agent_service, "set_outbound_message_handler"):
                agent_service.set_outbound_message_handler(
                    make_agent_outbound_handler(task_id)
                )
            agent_service.set_conversation_history(
                [dict(message) for message in snapshot.conversation_history],
                watermark=snapshot.conversation_watermark,
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
                await publish_task_event(
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
            await publish_task_event(
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
        if isinstance(e, AutoModelUnavailableError):
            error_code = ClientErrorCode.AUTO_MODEL_UNAVAILABLE
        elif isinstance(e, ClientVisibleError):
            error_code = e.error_code
        else:
            error_code = ClientErrorCode.TASK_EXECUTION_FAILED
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
                await publish_task_event(
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
    from .chat_history_service import persist_assistant_message_no_commit

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
        lock_task_lease_for_settlement_no_commit(db, task_lease)
        task = (
            db.query(Task)
            .filter(
                Task.id == task_id,
                Task.runner_id == task_lease.runner_id,
                task_lease_attempt_predicate(task_lease),
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
    from .task_orchestrator import settle_task_lease_isolated

    return settle_task_lease_isolated(lease, error_message=error_message)


async def execute_resume_background(
    task_id: int,
    agent_service: Any,
    task_owner_user_id: int | None,
    previous_task: Optional[asyncio.Task] = None,
    pending_user_message: Optional[Dict[str, Any]] = None,
    delivery_turn_id: str | None = None,
    delivery_already_dispatched: bool = False,
    delivery_notifier: DeliveryNotifier | None = None,
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
        if delivery_notifier is None:
            return
        try:
            await delivery_notifier(
                turn_id=delivery_turn_id,
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
                await publish_task_event(
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
        await publish_task_event(
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
                expected_attempt_id=lease.attempt_id,
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
            await publish_task_event(
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

        await publish_task_event(
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
                await publish_task_event(
                    create_terminal_task_error_event(
                        task_id,
                        broadcast_error_message,
                    ),
                    task_id,
                )
            else:
                await publish_task_event(
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
                                await publish_task_event(
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
                                    await publish_task_event(
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
        # Execution may run in a lease-guard child task, whose finally block
        # cannot remove this still-running outer owner. Release the registration
        # when its actual owner finishes, including failure or early cancellation.
        # Fence by identity so an old completion cannot remove a newer run.
        task.add_done_callback(
            lambda finished: self.cleanup_task(task_id, expected_task=finished)
        )
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
        # Cancellation before the coroutine starts skips its finally block.
        # Use the same owner fence as execution completion, including after
        # this coordinator has been promoted into running_tasks.
        task.add_done_callback(
            lambda finished: self.cleanup_task(task_id, expected_task=finished)
        )
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


background_task_manager = BackgroundTaskManager()


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
