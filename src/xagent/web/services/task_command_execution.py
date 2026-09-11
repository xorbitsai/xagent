"""Consume task control commands without importing API routes or connections."""

import asyncio
import enum
import logging
import time
import uuid
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import (
    TYPE_CHECKING,
    Any,
    Iterator,
    Literal,
    Optional,
    Union,
    assert_never,
    cast,
)

from sqlalchemy import func, or_, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ...config import (
    get_default_task_execution_mode,
)
from ...core.agent.checkpoint import (
    CheckpointReadError,
    CheckpointUnavailableError,
)
from ...core.agent.runner import UserMessageInjectionOutcome
from ...core.execution_scope import (
    EXECUTION_SCOPE_NOT_PROVIDED,
    resolve_execution_scope,
    resolve_execution_scope_off_turn,
)
from ...core.file_ref import FILE_REF_MODEL_INSTRUCTIONS
from ..models.chat_message import TaskChatMessage
from ..models.database import (
    get_session_local,
)
from ..models.task import Task, TaskStatus
from ..models.uploaded_file import UploadedFile
from ..models.user import User
from . import task_execution as task_execution_service
from .task_execution import (
    ClientVisibleError,
    ClientVisibleValidationError,
    ResumeReservationOutcome,
    _clear_task_pause_accepted,
    _is_task_pause_accepted,
    _mark_task_pause_accepted,
    _task_error_payload,
    client_safe_error_message,
    create_stream_event,
)
from .task_interaction_close import (
    ActiveInteractionAbsent,
    ActiveInteractionFound,
    ActiveInteractionUnavailable,
)
from .task_lease_service import registered_task_lease

if TYPE_CHECKING:
    from .task_orchestrator import TaskTurnPayload, _ClaimedTurn

from ..utils.db_timezone import safe_timestamp_to_unix
from .chat_history_service import (
    DELIVERY_COMPLETED,
    DELIVERY_DISPATCHED,
    DELIVERY_FAILED,
    DELIVERY_PENDING,
    UserMessageDeliveryClaim,
    claim_user_message_delivery_no_commit,
    inspect_user_message_delivery,
    mark_user_message_delivery_sync,
)
from .client_error_messages import (
    CLIENT_SAFE_GUIDANCE_IN_PROGRESS,
    CLIENT_SAFE_TASK_FAILURE,
    ClientErrorCode,
    client_error_message,
)
from .db_runtime import (
    is_database_pool_timeout,
    run_db_io_cancellation_safe,
)
from .external_task_cancel import (
    EXTERNAL_CANCEL_BROADCAST_REJECTION_REASONS,
    EXTERNAL_COMMAND_SCOPE,
    cancel_external_task_unserialized,
    external_cancel_exhausted_message,
)
from .external_task_input import (
    execute_external_task_input_command,
    external_input_terminal_message,
)
from .file_turn import (
    append_uploaded_files_context as _append_uploaded_files_context_to_message,
)
from .file_turn import (
    bind_turn_files_no_commit,
)
from .file_turn import build_uploaded_files_context as _build_uploaded_files_context
from .file_turn import (
    normalize_attachments_for_persistence as _normalize_attachments_for_persistence,
)
from .file_turn import (
    resolve_turn_file_infos,
)
from .managed_file_ref import (
    DurableObjectIntegrityError,
    DurableStorageOperationError,
    log_durable_storage_fault,
)
from .task_command_terminal_events import (
    TerminalTaskEventDraft,
    TerminalTaskEventMessageCode,
    bind_terminal_event_draft,
    first_party_message_terminal_text,
    is_external_cancel_command,
    terminal_event_draft_for_error,
)
from .task_command_transport import (
    COMMAND_ID_PATTERN,
    MAX_COMMAND_FAILURES,
    ClaimedTaskCommand,
    TaskCommandDeferred,
    TaskCommandKind,
    TaskCommandRejected,
    max_command_defers,
    task_has_live_foreign_runner,
    task_has_live_runner,
)
from .task_events import (
    CommandReply,
    DeliveryNotifier,
    command_reply,
    finish_task_command_delivery,
    publish_task_event,
)
from .task_execution_controller import (
    StaleTaskRunError,
    StaleTaskStateVersionError,
    TaskControlSnapshot,
    TaskControlState,
    control_state_for_status,
    task_execution_controller,
)
from .task_interaction_close import (
    active_interaction_id_sync,
    close_legacy_resume_interaction_sync,
)
from .task_lease_service import (
    TaskLease,
    bind_task_lease_context,
)
from .task_runtime import (
    SELECTED_FILE_IDS_AGENT_CONFIG_KEY,
    task_extension_bindings_from_agent_config,
)

logger = logging.getLogger(__name__)

# Non-transient turn rejections must not fall back to retryable busy guidance.
_TURN_REJECTION_CODES = {
    "actor_task_reuse_unsupported": (ClientErrorCode.MESSAGE_CONTINUATION_UNSUPPORTED),
    "workforce_archived": ClientErrorCode.WORKFORCE_ARCHIVED,
    "workforce_config_changed": ClientErrorCode.WORKFORCE_UNAVAILABLE,
    "workforce_run_not_found": ClientErrorCode.WORKFORCE_UNAVAILABLE,
    "workforce_run_not_active": ClientErrorCode.WORKFORCE_UNAVAILABLE,
}


EXTERNAL_COMMAND_SCOPE_ABSENT = object()


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


def _client_message_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if COMMAND_ID_PATTERN.fullmatch(normalized) is None:
        return None
    return normalized


class ClientVisiblePermissionError(ClientVisibleError, PermissionError):
    """An authorization refusal whose text is safe to show the sender."""


class ClientVisibleTaskCommandDeferred(ClientVisibleError, TaskCommandDeferred):
    """A deferral whose wording this module wrote for the sender.

    Terminal deferral broadcasts go through the chokepoint like everything
    else, so without the marker "waiting for the active task lease owner"
    would reach the client as the generic string and become
    indistinguishable from an outright failure.
    """


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

    An external-scope MESSAGE gets the same courtesy for the opposite
    reason: the generic fallback ends in "Please try again.", which is
    false for the non-retryable rejections this broadcast exists to
    surface (revoked principal, stale request, spent id). Its wording is
    picked by what the terminal exception proves -- non-application is
    asserted only when it is established, uncertainty otherwise -- and
    needs no task status, so the caller does not read the task for it.

    A first-party MESSAGE drops the prefix for the same proof rule: its
    sender is deciding whether to resend a durably accepted reply, so the
    sentence comes from the bound terminal-event draft instead of the
    exception text (#1500).
    """
    if is_external_cancel_command(kind=kind.value, scope=scope):
        return external_cancel_exhausted_message(task_status)
    if scope == EXTERNAL_COMMAND_SCOPE and kind == TaskCommandKind.MESSAGE:
        return external_input_terminal_message(error)
    if kind == TaskCommandKind.MESSAGE:
        # A first-party MESSAGE follows the external rule above rather than
        # the generic fallback: restating the deferral's last wait condition
        # under a "failed" prefix tells the sender nothing about whether the
        # accepted websocket command was applied (#1500). The sentence is derived from
        # the bound terminal-event draft, so it asserts non-application only
        # when the persisted outcome proves it.
        return first_party_message_terminal_text(terminal_event_draft_for_error(error))
    # kind.value in the text is safe only while every external-scope kind is
    # handled above; a new external-scope kind needs its own branch first.
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


def make_delivery_notifier(
    reply: CommandReply, client_message_id: str | None
) -> DeliveryNotifier | None:
    """Bind a client connection to the execution service's delivery callback."""
    if client_message_id is None:
        return None

    async def notify(*, turn_id: str | None, **outcome: Any) -> None:
        await send_message_delivery(
            reply,
            client_message_id=client_message_id,
            turn_id=turn_id or client_message_id,
            **outcome,
        )

    return notify


async def send_message_delivery(
    reply: CommandReply,
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
    await reply(payload)


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
    from .llm_utils import CoreStorage, make_normalize_model_id

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
    "Build a websocket file ref from an authorized UploadedFile record."
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


def _task_run_id(task: Any) -> str | None:
    run_id = getattr(task, "run_id", None)
    return str(run_id) if run_id is not None else None


def _task_lease_snapshot(task: Any) -> TaskLease | None:
    """Detach a routing key; resolve its registered holder before live writes."""

    task_id = getattr(task, "id", None)
    runner_id = getattr(task, "runner_id", None)
    run_id = getattr(task, "run_id", None)
    if task_id is None or runner_id is None or run_id is None:
        return None
    return TaskLease(
        # Routing observation only. The async caller resolves this key to
        # a registered acquisition before binding it to an execution writer.
        attempt_id=getattr(task, "lease_attempt_id", None),
        task_id=int(task_id),
        runner_id=str(runner_id),
        run_id=str(run_id),
    )


def _task_control_state_value(task: Any) -> str | None:
    control_state = getattr(task, "control_state", None)
    return str(control_state) if control_state is not None else None


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


@dataclass(frozen=True)
class _UserMessageDeliverySnapshot:
    """Primitive delivery result safe to carry outside its DB Session."""

    claimed: bool
    payload_matches: bool
    failed: bool
    pending: bool


class _TaskCommandCommitOutcomeUnknown(RuntimeError):
    """A Task message acceptance COMMIT may still be visible to a later retry."""


def _snapshot_user_message_delivery(
    claim: UserMessageDeliveryClaim,
) -> _UserMessageDeliverySnapshot:
    return _UserMessageDeliverySnapshot(
        claimed=bool(claim.claimed),
        payload_matches=bool(claim.payload_matches),
        failed=bool(claim.failed),
        pending=bool(claim.pending),
    )


def _retire_command_session_best_effort(
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
def _owned_command_session(*, task_id: int) -> Iterator[Session]:
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
                _retire_command_session_best_effort(db, task_id=task_id)
        else:
            _retire_command_session_best_effort(db, task_id=task_id)


def _reconcile_command_acceptance_graph(
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
                _retire_command_session_best_effort(
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

    with _owned_command_session(task_id=task_id) as db:
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
                _retire_command_session_best_effort(db, task_id=task_id)
                with _owned_command_session(task_id=task_id) as winner_db:
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
                _retire_command_session_best_effort(db, task_id=task_id)
                if not _reconcile_command_acceptance_graph(
                    task_id=task_id,
                    task_owner_user_id=task_owner_user_id,
                    turn_id=turn_id,
                    content=content,
                    file_ids=file_ids,
                    expected_run_id=expected_run_id,
                    expected_status=expected_status,
                ):
                    raise _TaskCommandCommitOutcomeUnknown(
                        f"live delivery {turn_id} has an unknown commit outcome"
                    ) from commit_error
            return claim_snapshot
        except Exception:
            db.rollback()
            raise


@dataclass(frozen=True)
class _TaskCommandRoutingSnapshot:
    """Detached task state used by task command routing and presentation."""

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
class _TaskMessagePreparation:
    """All synchronous state needed before task message orchestration.

    The preparation owner opens and closes its own Session in a worker thread.
    Only primitives and frozen application-layer values cross back to asyncio;
    no ORM row or Session may survive into a network wait, broadcast, agent
    construction, or turn claim.
    """

    requested_task_id: int
    routing: _TaskCommandRoutingSnapshot
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


def _load_task_command_routing_snapshot(
    db: Session,
    task: Task,
) -> tuple[_TaskCommandRoutingSnapshot, bool]:
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
        _TaskCommandRoutingSnapshot(
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


def _load_task_command_routing_snapshot_sync(
    task_id: int,
    *,
    task_owner_user_id: int,
    actor_user_id: int,
    actor_is_admin: bool,
) -> _TaskCommandRoutingSnapshot | None:
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
        routing, _is_agent_builder = _load_task_command_routing_snapshot(db, task)
        return routing


def _recover_recent_task_file_refs(
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


def _prepare_task_message_sync(
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
) -> _TaskMessagePreparation:
    """Authorize, normalize, and detach one task message off the event loop."""

    with _owned_command_session(task_id=requested_task_id) as db:
        files = deepcopy(raw_files)
        if not files:
            try:
                files = _recover_recent_task_file_refs(
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

        routing: _TaskCommandRoutingSnapshot | None = None
        if task is not None:
            routing, is_agent_builder = _load_task_command_routing_snapshot(db, task)
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
            routing, is_agent_builder = _load_task_command_routing_snapshot(db, task)

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

        from .task_orchestrator import (
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
            routing, _is_agent_builder = _load_task_command_routing_snapshot(
                db,
                task,
            )
            try:
                db.flush()
                db.commit()
            except Exception as commit_error:
                _retire_command_session_best_effort(
                    db,
                    task_id=routing.task_id,
                )
                if not _reconcile_command_acceptance_graph(
                    task_id=routing.task_id,
                    task_owner_user_id=routing.task_owner_user_id,
                    turn_id=turn_id,
                    content=display_user_message,
                    file_ids=list(turn_payload.file_ids),
                    expected_run_id=routing.run_id,
                    expected_status=TaskStatus.RUNNING,
                ):
                    raise _TaskCommandCommitOutcomeUnknown(
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

        return _TaskMessagePreparation(
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


async def handle_task_message(
    reply: CommandReply, task_id: int, message_data: dict
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
            reply,
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
            await publish_task_event(
                {**safe_error_payload, "timestamp": timestamp},
                authorized_task_id,
            )
            if suppress_delivery_ack:
                await reply(
                    {
                        "type": "error",
                        "message": ack_message,
                        "error_code": sender_error_code.value,
                        "timestamp": timestamp,
                    },
                )
        else:
            await reply(
                {
                    "type": "error",
                    "message": ack_message,
                    "error_code": sender_error_code.value,
                    "timestamp": timestamp,
                },
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
            lambda: _prepare_task_message_sync(
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
            from .agent_service_manager import get_agent_manager

            if preparation.task_created:
                old_task_id = preparation.requested_task_id
                await reply(
                    {
                        "type": "task_id_updated",
                        "old_task_id": old_task_id,
                        "new_task_id": task_id,
                    },
                )
                await publish_task_event(
                    create_stream_event(
                        "task_info",
                        task_id,
                        deepcopy(routing.task_info),
                        routing.created_at,
                    ),
                    task_id,
                )

            if preparation.claimed_created_turn is not None:
                from .task_orchestrator import TaskTurnOrchestrator

                await TaskTurnOrchestrator.schedule_claimed_create_turn(
                    task_id=task_id,
                    task_owner_user_id=routing.task_owner_user_id,
                    actor_user_id=actor_user_id,
                    payload=turn_payload,
                    claimed=preparation.claimed_created_turn,
                    context=context,
                )
                # Scheduling owns the durable PENDING -> DISPATCHED update.
                # Keep the local flag aligned so a later reply delivery failure
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
                registered_task_lease(routing.task_lease)
                if task_status == TaskStatus.RUNNING
                else None
            )
            agent_service = None
            supports_live_control = False
            if task_uses_live_control:
                resolved_execution_scope = await run_db_io_cancellation_safe(
                    lambda: resolve_execution_scope(task_id)
                )
                from .task_setup_snapshot import (
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
                        task_execution_service.make_agent_outbound_handler(task_id)
                    )
                supports_live_control = getattr(
                    agent_service, "supports_live_control", lambda: False
                )()

            if task_uses_live_control and supports_live_control:
                logger.info(f"Using agent message control for task {task_id}")
                assert agent_service is not None
                reservation = (
                    task_execution_service.background_task_manager.try_reserve_resume(
                        task_id,
                        expected_run_id=task_run_id,
                    )
                )
                if reservation is not ResumeReservationOutcome.RESERVED:
                    if suppress_delivery_ack:
                        # Durable commands own their retry budget. All three
                        # occupied states can clear without this attempt doing
                        # anything: a reservation can register or release, a
                        # coordinator can finish, and shutdown is retained as
                        # a defensive state even though normal shutdown stops
                        # the dispatcher before setting the manager flag.
                        holder_age_seconds = task_execution_service.background_task_manager.resume_holder_age_seconds(
                            task_id
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
                        task_execution_service.background_task_manager.release_resume_reservation(
                            task_id
                        )
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
                    active_interaction_read = await run_db_io_cancellation_safe(
                        lambda: active_interaction_id_sync(task_id)
                    )
                    # Translated here, on the read side, for both branches at
                    # once: the deferred path (posted false) does not re-read
                    # -- it carries whatever this site puts in
                    # pending_user_message["interaction_id"] and is pinned by
                    # tests/web/api/test_websocket_owner_actor.py not to call
                    # active_interaction_id_sync itself -- so translating the
                    # three-state read anywhere past this point would leave
                    # that path with a three-state value it has no way to
                    # judge (it never re-reads, so it cannot tell Unavailable
                    # apart from a fresh Absent). Absent and Unavailable both
                    # become `None` for the same reason as the other two
                    # close sites: `None` binds the close to no primary key,
                    # which matches zero rows and retires no question
                    # either way -- the safe outcome for a read that could
                    # not be made, not a claim that nothing was ever
                    # active. What then happens to the task's marker
                    # differs between the two, and this value is not what
                    # decides it: the clear beside the close runs its own
                    # check (see active_interaction_id_sync's docstring).
                    # Three branches, not a two-way isinstance fold, so
                    # Unavailable stays visible on its own line.
                    if isinstance(active_interaction_read, ActiveInteractionFound):
                        active_interaction_id = active_interaction_read.interaction_id
                    elif isinstance(active_interaction_read, ActiveInteractionAbsent):
                        active_interaction_id = None
                    elif isinstance(
                        active_interaction_read, ActiveInteractionUnavailable
                    ):
                        active_interaction_id = None
                        logger.info(
                            "active interaction read unavailable (reason=%s) "
                            "for task_id=%s; the legacy resume close will "
                            "match no row",
                            active_interaction_read.reason,
                            task_id,
                        )
                    else:
                        assert_never(active_interaction_read)

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
                                task_execution_service.background_task_manager.release_resume_reservation(
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

                    previous_task = task_execution_service.background_task_manager.running_tasks.get(
                        task_id
                    )
                    bg_task = asyncio.create_task(
                        task_execution_service.execute_resume_background(
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
                            delivery_notifier=(
                                None
                                if posted or suppress_delivery_ack
                                else make_delivery_notifier(reply, client_message_id)
                            ),
                        )
                    )
                    task_execution_service.background_task_manager.register_reserved_resume(
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
                        task_execution_service.background_task_manager.release_resume_reservation(
                            task_id
                        )
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
                await reply(
                    {
                        "type": "error",
                        "message": client_error_message(
                            ClientErrorCode.MESSAGE_CONTINUATION_UNSUPPORTED
                        ),
                        "error_code": (
                            ClientErrorCode.MESSAGE_CONTINUATION_UNSUPPORTED.value
                        ),
                    },
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
                    await task_execution_service.background_task_manager.wait_for_previous(
                        task_id
                    )
                    refreshed_routing = await run_db_io_cancellation_safe(
                        lambda: _load_task_command_routing_snapshot_sync(
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
                        await publish_task_event(
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
                    await publish_task_event(task_event, task_id)
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
                from .task_orchestrator import (
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
                    await publish_task_event(
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
                    await publish_task_event(
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
                    await publish_task_event(
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

        except _TaskCommandCommitOutcomeUnknown:
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
                await publish_task_event(
                    {
                        **error_payload,
                        "timestamp": timestamp,
                    },
                    authorized_task_id,
                )
            else:
                await reply(
                    {
                        "type": "error",
                        "message": message,
                        "error_code": error_code.value,
                        "timestamp": timestamp,
                    },
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
        await reply(
            {"type": "error", "message": message, "error_code": e.error_code.value},
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
        await reply(
            {"type": "error", "message": message, "error_code": error_code.value},
        )
    except ConnectionError as e:
        # Connection error
        logger.error("Connection error handling chat message: %s", e)
        raise
    except DurableObjectIntegrityError:
        # Attachment preparation runs in this outer scope (see the
        # ``_prepare_task_message_sync`` call above), *before* the inner
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


async def pause_task(reply: CommandReply, task_id: int, message_data: dict) -> None:
    """Handle task pause request"""
    try:
        logger.info(f"🔘 handle_pause_task called for task {task_id}")
        user = message_data.get("user")
        if not user:
            logger.error("No user in message_data")
            raise ValueError("User authentication required for task pause")

        logger.info(f"User {user.id} authenticated for pause")

        from .agent_service_manager import get_agent_manager
        from .task_setup_snapshot import load_task_setup_snapshot_sync

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
                await reply(
                    error_payload,
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
                await reply(
                    error_payload,
                )
                return
            _mark_task_pause_accepted(task_id)

            # This confirms only that the control request was accepted. The
            # frontend deliberately waits for the later durable ``task_info``
            # PAUSED state before changing its pause UI; treating this event
            # as ``task_paused`` would reintroduce the optimistic-state bug.
            await publish_task_event(
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
            await reply(
                error_payload,
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
        await reply(
            {
                "type": "error",
                "message": client_safe_error_message(e),
            },
        )
    except RuntimeError as e:
        logger.error("Runtime error pausing task %s: %s", task_id, e, exc_info=True)
        await reply(
            {
                "type": "error",
                "error_code": ClientErrorCode.MESSAGE_PROCESSING_FAILED.value,
                "message": client_error_message(
                    ClientErrorCode.MESSAGE_PROCESSING_FAILED
                ),
            },
        )
        raise
    except Exception as e:
        # Other errors, re-raise
        logger.error("Unexpected error pausing task %s: %s", task_id, e)
        raise


async def resume_task(
    reply: CommandReply, task_id: int, message_data: dict
) -> ResumeCommandResult:
    """Handle task resume request"""
    try:
        user = message_data.get("user")
        if not user:
            raise ValueError("User authentication required for task resume")

        from .agent_service_manager import get_agent_manager
        from .task_setup_snapshot import load_task_setup_snapshot_sync

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
            await reply(
                {"type": "error", "message": "Task not found or access denied"},
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
        # version first, though: under a NULL marker the read below reports
        # ActiveInteractionAbsent regardless of whether an active row
        # exists, so this refusal never fires for that state -- deliberately,
        # matching what the read surface would show for the same task (see
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
        active_interaction_read = await run_db_io_cancellation_safe(
            lambda: active_interaction_id_sync(task_id)
        )
        if isinstance(active_interaction_read, ActiveInteractionFound):
            active_interaction_id = active_interaction_read.interaction_id
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
                from . import ops_signals

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
                await reply(
                    {
                        "type": "error",
                        "message": (
                            "This task has an unanswered question; answer it "
                            "before resuming."
                        ),
                        "task": {"id": task_id, **resume_control_state},
                    },
                )
                return ResumeCommandResult(
                    ResumeCommandOutcome.REJECTED,
                    reason,
                    reason_code="interaction_pending",
                )
        elif isinstance(active_interaction_read, ActiveInteractionUnavailable):
            # Same action as the ActiveInteractionAbsent branch below -- the
            # resume proceeds -- but deliberately its own branch rather than
            # a shared one. This is the arm that turns into a refusal, and
            # refusing here is wired by the change that first writes
            # tasks.interaction_protocol_version = 1. Until that marker can
            # be 1, nothing in src/ having written it to anything but NULL,
            # a read that could not be made cannot be hiding a live native
            # interaction row: refusing today would cost a refused resume on
            # every waiting task, including the ones that provably carry no
            # question at all. Keeping the branch separate is what lets that
            # later change edit one branch's action and touch neither of the
            # other two.
            #
            # Logged at info, not warning: the read itself already logs one
            # warning for this same incident (see
            # active_interaction_id_sync), and a second warning-level line
            # would double-count one read failure.
            logger.info(
                "the active interaction read was unavailable (reason=%s) for "
                "task_id=%s run_id=%s; the resume proceeds",
                active_interaction_read.reason,
                task_id,
                task_fields.run_id,
            )
        elif isinstance(active_interaction_read, ActiveInteractionAbsent):
            # Nothing to do: no native interaction row is waiting on an
            # answer, so this gate lets the resume through.
            pass
        else:
            assert_never(active_interaction_read)

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

        def _defer_for_slot(
            outcome: ResumeReservationOutcome,
        ) -> ResumeCommandResult:
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
                await reply(
                    {
                        "type": "task_resumed",
                        "message": "Task is already running.",
                        "task": {"id": task_id},
                    },
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

        admission_state = (
            task_execution_service.background_task_manager.resume_admission_state(
                task_id,
                expected_run_id=task_fields.run_id,
            )
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
        # equivalent call in ``pause_task`` carries the
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
                await reply(
                    {
                        "type": "error",
                        "message": "Task is not paused and cannot be resumed.",
                        "task": {"id": task_id, **resume_control_state},
                    },
                )
                return ResumeCommandResult(
                    ResumeCommandOutcome.REJECTED,
                    reason,
                    reason_code="not_resumable",
                )
            reservation = (
                task_execution_service.background_task_manager.try_reserve_resume(
                    task_id,
                    expected_run_id=task_fields.run_id,
                )
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
                    # v1 reply and a WebSocket Resume landing together both
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
                task_execution_service.background_task_manager.release_resume_reservation(
                    task_id
                )
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
                task_execution_service.background_task_manager.release_resume_reservation(
                    task_id
                )
                raise
            try:
                previous_task = (
                    task_execution_service.background_task_manager.running_tasks.get(
                        task_id
                    )
                )
                bg_task = asyncio.create_task(
                    task_execution_service.execute_resume_background(
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
                task_execution_service.background_task_manager.register_reserved_resume(
                    task_id,
                    bg_task,
                    run_id=resume_snapshot.run_id,
                )
            except BaseException:
                if bg_task is not None:
                    bg_task.cancel()
                task_execution_service.background_task_manager.release_resume_reservation(
                    task_id
                )
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
        await reply(
            {
                "type": "error",
                "message": client_safe_error_message(e),
            },
        )
        return ResumeCommandResult(
            ResumeCommandOutcome.REJECTED,
            reason,
            reason_code="invalid_command_payload",
        )
    except RuntimeError as e:
        logger.error("Runtime error resuming task %s: %s", task_id, e, exc_info=True)
        await reply(
            {
                "type": "error",
                "error_code": ClientErrorCode.MESSAGE_PROCESSING_FAILED.value,
                "message": client_error_message(
                    ClientErrorCode.MESSAGE_PROCESSING_FAILED
                ),
            },
        )
        raise
    except Exception as e:
        # Other errors, re-raise
        logger.error("Unexpected error resuming task %s: %s", task_id, e)
        raise


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
    """Apply one DB-claimed command; personal replies use the host callback.

    A runner without an originating connection discards personal replies while
    task-level state/error events continue through the task event publisher.
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

    reply = command_reply(command.command_id, command.task_id)
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
        await handle_task_message(reply, command.task_id, message_data)
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
                await pause_task(
                    reply,
                    command.task_id,
                    message_data,
                )
            elif command.kind == TaskCommandKind.RESUME:
                resume_result = await resume_task(
                    reply,
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
                        from .a2a_task_cancel import cancel_a2a_task

                        await cancel_a2a_task(
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
    # identity of a task they do not own. Three payload literals rather than
    # one built and trimmed: the client-safe guard only inspects dict
    # literals passed straight to the sink, and a payload assembled in a
    # variable would drop this site out of its view entirely.
    if is_external_cancel_command(kind=command.kind.value, scope=scope):
        task_status = await _load_terminal_command_task_status(command.task_id)
        if task_status is TaskStatus.COMPLETED:
            # Every wording this branch can pick asserts the turn did not
            # finish cleanly, which is false here - the run completed and
            # its own completion frame already answered the audience. The
            # persisted terminal-event draft keeps its audit classification;
            # nothing renders it to a client.
            return
        await publish_task_event(
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
    if scope == EXTERNAL_COMMAND_SCOPE:
        # Every other external-scope command mirrors the persisted-event
        # rule (``include_command_identity=scope != EXTERNAL_COMMAND_SCOPE``):
        # the live frame must not disclose what the durable record withholds,
        # because an embedding application's stream projection may forward
        # ``agent_error`` frames verbatim to the anonymous audience. No task
        # status read either -- the wording asserts nothing about the turn,
        # and this branch runs inside exception handlers where an unguarded
        # database read would escape the disposition that is being reported.
        await publish_task_event(
            {
                "type": "agent_error",
                "message": client_safe_task_command_failure(
                    command.kind,
                    error,
                    scope=scope,
                ),
                "task_id": command.task_id,
                "timestamp": datetime.now(timezone.utc).timestamp(),
            },
            command.task_id,
        )
        return
    # ``outcome``/``resend_safe``/``message_code`` expose the persisted
    # terminal disposition structurally (#1500), so the sender can decide
    # whether resending the command is safe without parsing ``message``.
    # The field names match the durable terminal-event projection (#1904),
    # including its two disambiguators: ``task_run_id`` (the acceptance
    # snapshot's run) and ``outcome_version`` (the attempt count, which the
    # terminal CAS write pins to this same value), because an operator retry
    # can send one ``command_id`` through a terminal broadcast twice.
    # Values come from the draft the dispatcher binds before broadcasting;
    # a missing draft degrades to the unsafe/unknown reading. Only this
    # identity-bearing frame carries them: the two external frames above
    # deliberately expose nothing the anonymous audience cannot act on,
    # and a retry decision needs the ``command_id`` they withhold.
    #
    # ``resend_safe`` is a proof of non-application, not a retryability
    # rating: the only producer of ``True`` is the MESSAGE contention
    # deferral. PAUSE/RESUME/CANCEL terminals therefore always carry
    # ``False`` even though those commands are idempotent by design -- a
    # consumer deciding whether to offer a retry for them must reason from
    # ``command_kind``, never from this flag.
    draft = terminal_event_draft_for_error(error)
    await publish_task_event(
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
            "outcome": "failed",
            "resend_safe": bool(draft and draft.resend_safe),
            "message_code": (
                draft.message_code.value if draft and draft.message_code else None
            ),
            "command_kind": command.kind.value,
            "task_id": command.task_id,
            "command_id": command.command_id,
            "task_run_id": command.target_run_id,
            "outcome_version": int(command.attempt_count or 0),
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
        if command.defer_count + 1 >= max_command_defers():
            finish_task_command_delivery(command.command_id, command.task_id)
            bind_terminal_event_draft(
                exc,
                await _terminal_command_event_draft(command, exc),
            )
            await _broadcast_terminal_command_error(command, exc)
        # A deferral that will retry keeps its origin entry.
        raise
    except TaskCommandRejected as exc:
        # Rejections come from handlers that already expose their durable
        # domain-level outcome. The dispatcher makes them terminal immediately.
        finish_task_command_delivery(command.command_id, command.task_id)
        scope = _command_scope(command)
        # Two external-scope commands answer an audience with no channel of
        # its own, so their terminal rejections broadcast here. First-party
        # rejections keep their handler-owned notifications and are
        # deliberately not re-broadcast.
        #
        # MESSAGE: the external audience's answer travels through the seam
        # executor, which reports outcomes only as these exceptions. Without
        # a broadcast, a deferred answer that is later terminally rejected
        # (revoked principal, stale request, spent id, quota) vanishes
        # silently — the task stays parked and nobody is told
        # (xorbitsai/xagent-saas#952 B2).
        #
        # CANCEL: per-reason policy (#2009). ``stale_run`` is the one
        # rejection the widget's stop press can reach — the target's
        # run/version moved between the producer's read and this dispatch —
        # and silence there leaves the press doing nothing while the
        # unwanted run keeps producing. The reason set and the rationale
        # for what stays silent live with the wording constants in
        # ``external_task_cancel.py``.
        if (
            command.kind == TaskCommandKind.MESSAGE and scope == EXTERNAL_COMMAND_SCOPE
        ) or (
            is_external_cancel_command(kind=command.kind.value, scope=scope)
            and exc.reason in EXTERNAL_CANCEL_BROADCAST_REJECTION_REASONS
        ):
            # An executor-bound presentation draft is preserved; the
            # standard one is derived only when none was bound, so the
            # persisted terminal event is classified either way.
            if terminal_event_draft_for_error(exc) is None:
                bind_terminal_event_draft(
                    exc,
                    await _terminal_command_event_draft(command, exc),
                )
            try:
                await _broadcast_terminal_command_error(command, exc)
            except Exception:
                # The rejection is already classified and its draft is
                # bound; a failed notification must not supersede the
                # terminal rejection into a retried failure (the finalize
                # broadcast in external_task_cancel.py keeps the same rule).
                # Exception, never BaseException: cancellation propagates.
                logger.warning(
                    "task %s external %s rejection is terminal but its "
                    "broadcast failed",
                    command.task_id,
                    command.kind.value,
                    exc_info=True,
                )
        raise
    except Exception as exc:
        if command.failure_count + 1 >= MAX_COMMAND_FAILURES:
            finish_task_command_delivery(command.command_id, command.task_id)
            bind_terminal_event_draft(
                exc,
                await _terminal_command_event_draft(command, exc),
            )
            await _broadcast_terminal_command_error(command, exc)
        raise
    finish_task_command_delivery(command.command_id, command.task_id)
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
