"""SDK task endpoints: ``/v1/chat/tasks/*`` family.

Phase 1 surface this module owns:

  - POST /v1/chat/tasks
  - POST /v1/chat/tasks/{id}/messages
  - GET  /v1/chat/tasks/{id}
  - GET  /v1/chat/tasks/{id}/steps

All endpoints authenticate via ``get_principal_from_api_key`` -- an
agent-bound key scopes to its agent's SDK tasks, a workforce-bound key
scopes to the tasks behind its workforce's runs (``WorkforceRun.task_id``
is a 1:1 unique binding; runs are created via
``POST /v1/workforces/{id}/runs``), and a personal key scopes to the SDK
tasks its holder created with a personal key. A runtime key never reaches
a task a personal key created. Tasks are created with an agent key for
its own agent, or with a personal key for any agent the holder can see;
the task then runs as the holder.
All responses use the stable ``V1ApiError`` envelope. Task turn
lifecycle (claim RUNNING, persist messages, schedule bg, sync output)
is delegated to ``services.task_orchestrator.TaskTurnOrchestrator``,
which is also used by the WebSocket UI path so both transports share
one state machine.
"""

import asyncio
import logging
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Literal, NoReturn, Optional, cast

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import TypeAdapter
from sqlalchemy import func
from sqlalchemy.orm import Session

from ....core.runtime_performance import increment_counter
from ....core.tools.adapters.vibe.connector_runtime import ConnectorRuntimeError
from ...config import is_allowed_file
from ...models.database import get_session_local
from ...models.task import Task, TaskStatus, TraceEvent
from ...models.uploaded_file import UploadedFile
from ...schemas.v1 import (
    AppendMessageRequest,
    AppendMessageResponse,
    CancelQuestionResponse,
    CreateTaskRequest,
    CreateTaskResponse,
    DeleteTaskResponse,
    PendingInteraction,
    PublicStep,
    ReplyRequest,
    ReplyResponse,
    StepsResponse,
    TaskFileInfo,
    TaskFilesResponse,
    TaskInfoResponse,
    UploadedFileInfo,
    UploadFilesResponse,
)
from ...services import task_start as task_start_service
from ...services.db_runtime import (
    run_db_io_cancellation_safe,
)
from ...services.hot_path_cache import (
    cache_get,
    cache_set,
    cache_version_token,
    task_cache_ttl_seconds,
    task_snapshot_key,
    task_steps_key,
)
from ...services.managed_file_ref import (
    DurableStorageOperationError,
)
from ...services.prompt_context import validate_signed_prompt_context
from ...services.task_auto_resume import TerminalView, status_only_terminal_view
from ...services.task_delete import (
    TaskDeleteBusy,
    TaskDeleteSnapshot,
    TaskExtensionCleanupRejected,
    commit_task_deletion,
    finish_task_deletion,
    task_delete_snapshot,
    task_is_active_for_delete,
)
from ...services.task_execution_controller import TaskControlState
from ...services.task_interaction_read import get_pending_interaction_question
from ...services.task_orchestrator import (
    TaskTurnError,
    TaskTurnFileBindingError,
    TaskTurnNotFoundError,
)
from ...services.task_start_protocol import has_queued_start
from ...services.uploaded_file_store import (
    UploadedFileVersionSnapshot,
    snapshot_uploaded_file_version,
)
from ...utils.db_timezone import normalize_datetime_from_db
from . import _events_stream
from ._step_mapping import map_trace_events_to_public_steps
from .deps import (
    ApiKeyPrincipal,
    get_principal_from_api_key,
    record_principal_usage,
)
from .errors import V1ApiError, V1ErrorCode, raise_for_turn_rejection

logger = logging.getLogger(__name__)

router = APIRouter()

_CONNECTOR_RUNTIME_SETUP_FAILED_MESSAGE = "Connector runtime setup failed."

# ``datetime.isoformat()`` renders a UTC offset as ``+00:00``; pydantic's own
# JSON mode renders it as ``Z``. ``StepsResponse.steps_expired_at`` goes out
# through pydantic, so the 410 body's ``details.expired_at`` has to be
# serialized the same way to be byte-identical for the same instant.
_DATETIME_JSON: TypeAdapter[datetime] = TypeAdapter(datetime)


def _resolve_upload_owner_user_id_isolated(
    *,
    task_id: int,
    principal: ApiKeyPrincipal,
) -> int:
    """Authorize an existing upload target in one short worker Session."""

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = _resolve_task_or_404(task_id, principal, db)
        return int(task.user_id)


@router.post("/chat/files", response_model=UploadFilesResponse)
async def upload_task_files(
    files: list[UploadFile] = File(...),
    task_id: Optional[int] = Query(
        default=None,
        gt=0,
        description=(
            "Existing SDK task whose persisted runtime owner should own "
            "the uploaded files."
        ),
    ),
    principal: ApiKeyPrincipal = Depends(get_principal_from_api_key),
) -> UploadFilesResponse:
    """Store files for later attachment to a task turn.

    API-key-gated counterpart to the JWT-only ``POST /api/files/upload``.
    Files are stored unbound (``UploadedFile.task_id`` NULL); the returned
    ``file_id`` values are passed back in ``message.files`` on
    ``POST /v1/chat/tasks`` / ``POST /v1/workforces/{id}/runs`` (or
    ``.../messages``), where they get bound to the task and exposed to
    the agent.

    When ``task_id`` is omitted, the upload is owned by the key owner's
    current user for a future create request. When ``task_id`` is provided,
    the task is authorized through the key-bound owner and its persisted
    ``Task.user_id`` owns the upload. This keeps historical tasks usable
    after owner changes without transferring file ownership during append.
    """
    from ..files import store_uploaded_files

    upload_owner_user_id = principal.owner_user_id
    if task_id is not None:
        upload_owner_user_id = await run_db_io_cancellation_safe(
            lambda: _resolve_upload_owner_user_id_isolated(
                task_id=task_id,
                principal=principal,
            )
        )

    # Reject unsupported types up front with a clean v1 400. ``store_uploaded_files``
    # would otherwise raise a bare HTTPException (a 500 for unsupported type) that
    # bypasses the v1 error envelope and leaks the internal ``task_type`` wording.
    for uploaded in files:
        if not is_allowed_file(uploaded.filename or "", "general"):
            raise V1ApiError(
                V1ErrorCode.INVALID_INPUT,
                400,
                message=f"Unsupported file type: {uploaded.filename}",
            )

    try:
        result = await store_uploaded_files(
            upload_items=list(files),
            task_type="general",
            task_id=None,
            folder=None,
            user_id=upload_owner_user_id,
            single_file_mode=False,
        )
    except HTTPException as exc:
        # ``store_uploaded_files`` is shared with the JWT upload route and raises
        # bare HTTPExceptions; translate to the v1 envelope so SDK clients keep a
        # stable {"error": {"code": ...}} shape. 503 (durable storage) stays 503 so
        # callers can retry; 413 (too large) stays 413; other client errors -> 400.
        if exc.status_code == 503:
            _raise_v1_storage_unavailable(exc)
        if 400 <= exc.status_code < 500:
            raise V1ApiError(
                V1ErrorCode.INVALID_INPUT,
                413 if exc.status_code == 413 else 400,
                message="File upload rejected.",
            ) from exc
        raise V1ApiError(V1ErrorCode.INTERNAL_ERROR, 500) from exc
    return UploadFilesResponse(
        files=[
            UploadedFileInfo(
                file_id=f["file_id"],
                filename=f["filename"],
                file_size=f["file_size"],
                mime_type=f.get("mime_type"),
            )
            for f in result.get("files", [])
        ]
    )


def _sdk_task_scope(principal: ApiKeyPrincipal) -> task_start_service.SdkTaskScope:
    """The task scope of ``principal``: its bound agent or workforce, or the
    holder of a personal key.

    Any other principal has no task scope and gets the same 404 a task
    outside its scope gets.
    """
    if principal.agent is not None:
        return task_start_service.SdkTaskScope(
            agent_id=int(principal.agent.id), workforce_id=None
        )
    if principal.workforce is not None:
        return task_start_service.SdkTaskScope(
            agent_id=None, workforce_id=int(principal.workforce.id)
        )
    if principal.user is not None:
        return task_start_service.SdkTaskScope(
            agent_id=None, workforce_id=None, user_id=int(principal.user.id)
        )
    raise V1ApiError(V1ErrorCode.TASK_NOT_FOUND, 404)


def _raise_start_rejection(exc: task_start_service.TaskStartRejected) -> NoReturn:
    if exc.reason == "agent_id_required":
        raise V1ApiError(
            V1ErrorCode.INVALID_INPUT, 422, message="agent_id is required"
        ) from exc
    if exc.reason == "prompt_context_fields_rejected":
        # Names only, without the reason or the declared limit.
        raise V1ApiError(
            V1ErrorCode.INVALID_INPUT,
            422,
            message="prompt_context fields this agent does not accept: "
            + ", ".join(exc.fields),
        ) from exc
    if exc.reason == "connector_runtime_setup_failed":
        raise V1ApiError(
            V1ErrorCode.INTERNAL_ERROR,
            500,
            message=_CONNECTOR_RUNTIME_SETUP_FAILED_MESSAGE,
        ) from exc
    if exc.reason == "task_expired":
        # 410, not 404: an unaware client still handles it as the 4xx it
        # already gets for a missing task, and one that knows the code can
        # stop polling an id that will never answer again.
        assert exc.task_id is not None and exc.expired_at is not None
        raise V1ApiError(
            V1ErrorCode.TASK_EXPIRED,
            410,
            details={
                "task_id": exc.task_id,
                "expired_at": _DATETIME_JSON.dump_python(exc.expired_at, mode="json"),
            },
        ) from exc
    code, status = {
        "agent_not_found": (V1ErrorCode.AGENT_NOT_FOUND, 404),
        "workforce_not_found": (V1ErrorCode.WORKFORCE_NOT_FOUND, 404),
        "task_not_found": (V1ErrorCode.TASK_NOT_FOUND, 404),
        "runtime_agent_missing": (V1ErrorCode.INTERNAL_ERROR, 500),
    }[exc.reason]
    raise V1ApiError(code, status) from exc


def _raise_v1_storage_unavailable(exc: Exception) -> NoReturn:
    """The retryable-503 envelope every durable-fault arm in this module answers with.

    One helper rather than the envelope inline at each arm, matching this
    module's other ``_raise_v1_*`` helpers. It always chains ``from exc``: the
    ``v1_api_error_handler`` does not render ``exc_info`` today, so no arm loses
    anything by chaining, and an arm that dropped the cause would lose it
    silently the day that boundary starts logging one (#1521).
    """
    raise V1ApiError(
        V1ErrorCode.INTERNAL_ERROR,
        503,
        message="File storage is temporarily unavailable.",
    ) from exc


def _raise_v1_file_binding_error(exc: TaskTurnFileBindingError) -> NoReturn:
    raise V1ApiError(
        V1ErrorCode.INVALID_INPUT,
        400,
        message=(
            "These file ids are not accessible: " + ", ".join(exc.missing_file_ids)
        ),
    ) from exc


def _validate_prompt_context(prompt_context: dict[str, Any] | None) -> None:
    if prompt_context is None:
        return
    try:
        validate_signed_prompt_context(prompt_context)
    except ValueError as exc:
        raise V1ApiError(V1ErrorCode.INVALID_INPUT, 422, message=str(exc)) from exc


def _raise_v1_connector_runtime_error(exc: ConnectorRuntimeError) -> NoReturn:
    try:
        code = V1ErrorCode(exc.code)
    except ValueError:
        code = V1ErrorCode.INVALID_RUNTIME_CONTEXT
    raise V1ApiError(
        code,
        exc.status_code,
        message=exc.safe_message,
        details=exc.to_public_error().get("details"),
    ) from exc


def _validate_owner_scope(
    principal: ApiKeyPrincipal,
    *,
    request_agent_id: int | None,
    request_workforce_id: int | None,
    task_agent_id: int,
) -> None:
    try:
        task_start_service.validate_sdk_owner_scope(
            _sdk_task_scope(principal),
            request_agent_id=request_agent_id,
            request_workforce_id=request_workforce_id,
            task_agent_id=task_agent_id,
        )
    except task_start_service.TaskStartRejected as exc:
        _raise_start_rejection(exc)


def _ensure_agent_reachable(
    principal: ApiKeyPrincipal, task: Task, db: Session
) -> None:
    """404 ``agent_not_found`` once a personal key's holder can no longer run
    the task's agent; see ``ensure_sdk_task_agent_reachable``."""
    try:
        task_start_service.ensure_sdk_task_agent_reachable(
            db, _sdk_task_scope(principal), task
        )
    except task_start_service.TaskStartRejected as exc:
        _raise_start_rejection(exc)


@router.post(
    "/chat/tasks",
    status_code=202,
    response_model=CreateTaskResponse,
)
async def create_chat_task(
    request: CreateTaskRequest,
    principal: ApiKeyPrincipal = Depends(get_principal_from_api_key),
) -> CreateTaskResponse:
    """Create a new SDK-driven task and kick off its first turn.

    Single endpoint does three things atomically from the caller's
    perspective:

      1. Verifies the body's ``agent_id``: with an agent key it must be
         the agent bound to the key; with a personal key it must be an
         agent the key holder can see (the same set ``GET /v1/agents``
         lists). Otherwise -> 404 ``agent_not_found`` (404 not 403, so
         the existence of unrelated agents isn't leaked via error code),
         decided before any file or connector input is examined.
      2. Persists a new :class:`Task` row owned by the agent's user for
         an agent key, or by the key holder for a personal key (the turn
         then runs with the holder's files, connectors and skills), with
         ``source='sdk'``, ``is_visible=False``, the credential kind,
         and ``input`` set to the user message. Also persists the first
         user message to
         ``task_chat_messages`` so the existing background execution
         path can consume it without special-casing this entry point.
      3. Schedules background execution via
         ``start_task_in_background`` (which uses the same coroutine
         the WebSocket handler does). Returns 202 immediately --
         callers poll ``GET /v1/chat/tasks/{task_id}`` to observe the
         eventual ``completed`` / ``failed`` status. A ``failed`` status
         can be derived: when automatic resume of an interrupted task
         gives up, the task stays ``paused`` in storage and is reported
         as ``failed``. It can change if the task's owner resumes it by
         hand or the caller appends a message.

    Args:
        request: Validated :class:`CreateTaskRequest`. ``message.content``
            is guaranteed non-empty by Pydantic; ``agent_id`` is the
            target agent the SDK caller wants to invoke.
        principal: Key owner from the auth dependency: an agent key's
            bound agent, or a personal key's holder. A workforce key
            cannot create tasks here.
    Returns:
        :class:`CreateTaskResponse` with the new ``task_id``,
        ``agent_id``, ``status='running'`` (the atomic claim inside
        the handler flips the row from PENDING to RUNNING before the
        response is sent), and ``created_at`` for the caller to
        start polling from.

    Raises:
        V1ApiError 401: missing/invalid/revoked key (raised inside
            ``get_principal_from_api_key``; envelope is uniform with
            other auth failures).
        V1ApiError 404: ``agent_not_found`` -- the agent is not the one
            bound to the agent key, or not one the personal key's holder
            can see.
        V1ApiError 422: ``prompt_context`` is malformed, or a personal
            key sent a field the agent does not declare or longer than its
            declared length.
        500 (V1 envelope): any unexpected exception -- the global
            handler in ``web/app.py`` translates to
            ``{"error": {"code": "internal_error", ...}}`` and the raw
            exception message stays out of the response.
    """
    agent_identity = principal.agent
    _validate_prompt_context(request.prompt_context)
    credential_kind: task_start_service.SdkCredentialKind
    if principal.user is not None:
        # create_sdk_task checks that the holder can see this agent.
        agent_id = int(request.agent_id)
        task_owner_user_id = int(principal.user.id)
        credential_kind = "personal"
    elif agent_identity is not None and request.agent_id == agent_identity.id:
        agent_id = int(agent_identity.id)
        task_owner_user_id = int(agent_identity.user_id)
        credential_kind = "runtime"
    else:
        raise V1ApiError(V1ErrorCode.AGENT_NOT_FOUND, 404)

    try:
        started = await task_start_service.create_sdk_task(
            agent_id=agent_id,
            task_owner_user_id=task_owner_user_id,
            actor_user_id=task_owner_user_id,
            message=request.message.content,
            file_ids=tuple(request.message.files or ()),
            connector_runtime_context=tuple(
                item.model_dump() for item in request.connector_runtime_context or ()
            ),
            timezone=request.timezone,
            requested_dispatch_class=request.dispatch_class,
            requested_priority=request.priority,
            prompt_context=request.prompt_context,
            credential_kind=credential_kind,
        )
    except task_start_service.TaskStartRejected as exc:
        _raise_start_rejection(exc)
    except task_start_service.SdkCreateTurnRejected as exc:
        raise_for_turn_rejection(exc.reason)
    except TaskTurnFileBindingError as exc:
        _raise_v1_file_binding_error(exc)
    except ConnectorRuntimeError as exc:
        _raise_v1_connector_runtime_error(exc)
    except DurableStorageOperationError as exc:
        _raise_v1_storage_unavailable(exc)

    await record_principal_usage(
        principal,
        route="create_task",
        task_id=started.task_id,
        agent_id=started.agent_id,
    )

    return CreateTaskResponse(
        task_id=started.task_id,
        agent_id=started.agent_id,
        status=started.status.value,
        created_at=cast(datetime, started.accepted_at),
        run_id=started.run_id,
        state_version=started.state_version,
        control_state=started.control_state,
        dispatch_class=started.dispatch_class,
    )


# Terminal task statuses for ``completed_at`` derivation in GET task.
# A task in any of these states is no longer running; ``updated_at``
# is the last DB write and thus the closest proxy to "when did the
# task end". For non-terminal states we return ``None`` so SDK
# clients can disambiguate "still running" from "ended at <time>".
_TERMINAL_STATUSES = (TaskStatus.COMPLETED, TaskStatus.FAILED)


@dataclass(frozen=True)
class _TaskInfoSnapshot:
    """Detached task fields required by the public status response."""

    task_id: int
    agent_id: int
    status: TaskStatus
    run_id: str | None
    state_version: int
    control_state: str
    input: str | None
    output: str | None
    error: str | None
    created_at: datetime | None
    updated_at: datetime | None
    pending_question: str | None
    pending_interactions: list[dict[str, Any]] | None
    # Set when a status-only caller's task rests PAUSED after automatic resume
    # stopped for good. Every other field keeps the stored value;
    # :func:`effective_task_snapshot` and the GET response apply the view.
    view: TerminalView | None = None
    # Set when an appended turn's START is queued and the row still holds the
    # previous turn (see ``_QUEUED_START_STATUSES``). Applied like ``view``.
    start_queued: bool = False


# Terminal statuses an append is accepted from. With shared execution the row
# keeps one of them, and the previous turn's output and error, until the
# queued START is handed off, so a caller waiting for a terminal status would
# take the previous turn as the new one. A PAUSED row keeps reading as paused,
# which callers already wait on. A RUNNING or WAITING_FOR_USER row has moved
# since the START was queued, and a PENDING row already reads as pending.
_QUEUED_START_STATUSES = (TaskStatus.COMPLETED, TaskStatus.FAILED)


def effective_task_snapshot(snapshot: _TaskInfoSnapshot) -> _TaskInfoSnapshot:
    """``snapshot`` as the caller must read it: ``pending`` with no output or
    error while an appended turn's START is queued, ``failed`` when it carries
    a terminal view, the stored values otherwise.

    ``run_id``, ``state_version`` and ``updated_at`` stay the stored ones, so
    the stream's change detection compares what the task row says. A queued
    START also keeps the stored ``control_state``, which is what the append
    response reported.
    """

    if snapshot.start_queued:
        return replace(
            snapshot,
            status=TaskStatus.PENDING,
            output=None,
            error=None,
        )
    view = snapshot.view
    if view is None:
        return snapshot
    return replace(
        snapshot,
        status=TaskStatus.FAILED,
        control_state=TaskStatus.FAILED.value,
        output=None,
        error=view.error,
    )


@dataclass(frozen=True)
class _TaskStepsVersionSnapshot:
    """Authorized task identity and the trace version used for cache lookup.

    The version is the pair (``max_event_id``, ``traces_expired_at``).
    A trace purge does not always move ``max_event_id``: rows outside
    this read (build-scoped events, checkpoint blobs) can be all it
    removed, and a dialect that reuses ids can put a later turn back on
    the same maximum. An entry cached before the purge must not be served
    without the ``steps_expired`` flag it now owes.
    """

    task_id: int
    agent_id: int
    max_event_id: int
    traces_expired_at: datetime | None
    storage_version: int = 1


@dataclass(frozen=True)
class _TraceEventSnapshot:
    """Trace fields consumed by the pure public-step mapper."""

    task_id: int
    event_id: str
    event_type: str
    timestamp: datetime
    step_id: str | None
    data: Any


@dataclass(frozen=True)
class _TaskStepsSnapshot:
    """Revalidated task identity and detached ordered trace rows."""

    task_id: int
    agent_id: int
    max_event_id: int
    traces_expired_at: datetime | None
    events: tuple[_TraceEventSnapshot, ...]
    storage_version: int = 1


def _read_traces_expired_at(task_id: int, db: Session) -> datetime | None:
    """``tasks.traces_expired_at``, read *after* the trace rows it qualifies.

    The trace purge deletes the rows and stamps this column in one
    transaction, but a steps read sees each statement's own committed
    snapshot. Reading the column first could pair a pre-purge "never
    expired" with the post-purge empty list -- a truncated history
    presented as complete. Reading it last errs the other way: at worst
    full steps flagged as possibly incomplete.
    """
    return cast(
        datetime | None,
        db.query(Task.traces_expired_at).filter(Task.id == task_id).scalar(),
    )


def _resolve_task_or_404(task_id: int, principal: ApiKeyPrincipal, db: Session) -> Task:
    try:
        return task_start_service.resolve_sdk_task(
            task_id, _sdk_task_scope(principal), db
        )
    except task_start_service.TaskStartRejected as exc:
        _raise_start_rejection(exc)


# A task an executor may still be writing to is refused rather than deleted
# out from under it; a finished/paused/waiting task is deletable.
def _raise_if_task_active_for_delete(db: Session, task: Task) -> None:
    if task_is_active_for_delete(db, task):
        raise V1ApiError(
            V1ErrorCode.TASK_BUSY,
            409,
            message=(
                "Task is running or has queued work; wait for it to finish before "
                "deleting it."
            ),
        )


def _load_task_delete_snapshot(
    task_id: int,
    principal: ApiKeyPrincipal,
) -> TaskDeleteSnapshot:
    """Authorize and guard one task inside a worker-owned short Session."""

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = _resolve_task_or_404(task_id, principal, db)
        _raise_if_task_active_for_delete(db, task)
        return task_delete_snapshot(task)


def _task_file_source(file_record: UploadedFile) -> Literal["output", "upload"]:
    # ``workspace_category`` is set by workspace registration; uploads leave
    # it NULL, so anything but the output category is an attachment.
    return "output" if file_record.workspace_category == "output" else "upload"


def _list_chat_task_files_sync(
    task_id: int, principal: ApiKeyPrincipal
) -> TaskFilesResponse:
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = _resolve_task_or_404(task_id, principal, db)
        records = (
            db.query(UploadedFile)
            .filter(UploadedFile.task_id == int(task.id))
            .order_by(UploadedFile.created_at.asc(), UploadedFile.id.asc())
            .all()
        )
        files: list[TaskFileInfo] = []
        for record in records:
            if (
                record.detached_reason is not None
                or record.storage_status == "compensating"
            ):
                continue
            created_at = (
                normalize_datetime_from_db(record.created_at)
                if record.created_at is not None
                else None
            )
            files.append(
                TaskFileInfo(
                    file_id=str(record.file_id),
                    filename=str(record.filename),
                    file_size=int(record.file_size or 0),
                    mime_type=record.mime_type,
                    created_at=created_at,
                    source=_task_file_source(record),
                )
            )
        return TaskFilesResponse(files=files)


def _load_download_file_snapshot(
    file_id: str, principal: ApiKeyPrincipal
) -> UploadedFileVersionSnapshot:
    """Authorize one SDK download and detach its storage metadata.

    Task-bound files require the task to be reachable through this key (the
    same scope every other v1 task route uses). A never-attached upload is
    allowed only to the user who uploaded it. Every refusal is a 404 so file
    existence is not disclosed.
    """

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        record = db.query(UploadedFile).filter(UploadedFile.file_id == file_id).first()
        if (
            record is None
            or record.detached_reason is not None
            or record.storage_status == "compensating"
        ):
            raise V1ApiError(V1ErrorCode.FILE_NOT_FOUND, 404)
        if record.task_id is not None:
            try:
                _resolve_task_or_404(int(record.task_id), principal, db)
            except V1ApiError as exc:
                raise V1ApiError(V1ErrorCode.FILE_NOT_FOUND, 404) from exc
        elif int(record.user_id) != principal.owner_user_id:
            raise V1ApiError(V1ErrorCode.FILE_NOT_FOUND, 404)
        return snapshot_uploaded_file_version(record)


def _raise_v1_file_download_error(exc: HTTPException) -> NoReturn:
    if exc.status_code >= 500:
        _raise_v1_storage_unavailable(exc)
    if exc.status_code == 409:
        raise V1ApiError(
            V1ErrorCode.INTERNAL_ERROR,
            409,
            message="File integrity verification failed. Please re-upload this file.",
        ) from exc
    # 403 (path containment) and 404 both stay non-disclosing on the SDK surface.
    raise V1ApiError(V1ErrorCode.FILE_NOT_FOUND, 404) from exc


def _load_task_info_snapshot(
    task_id: int,
    principal: ApiKeyPrincipal,
) -> _TaskInfoSnapshot:
    """Authorize and detach one task inside a worker-owned short Session."""

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        # Read before the row. The handoff ends the START and moves the row in
        # one commit, so a START no longer queued here means the row read next
        # is already the handed-off one; read in the other order, a handoff
        # committing in between would leave the previous turn's row with no
        # queued START. A RUNNING row read next is excluded by its status.
        queued_before_row = has_queued_start(db, task_id)
        task = _resolve_task_or_404(task_id, principal, db)
        pending_question: str | None = None
        pending_interactions: list[dict[str, Any]] | None = None
        if task.status == TaskStatus.WAITING_FOR_USER:
            pending_question, pending_interactions = get_pending_interaction_question(
                db, task
            )
        view: TerminalView | None = None
        start_queued = queued_before_row and task.status in _QUEUED_START_STATUSES
        # The view re-checks both inside its own statement; this only skips
        # the query for reads that cannot be terminal.
        if (
            task.status == TaskStatus.PAUSED
            and task.control_state == TaskControlState.PAUSED.value
        ):
            view = status_only_terminal_view(db, int(task.id))
        if view is not None:
            increment_counter(
                "xagent.v1.task.projected_failed", attributes={"state": view.state}
            )
        return _TaskInfoSnapshot(
            task_id=int(task.id),
            agent_id=int(task.agent_id),
            status=task.status,
            run_id=str(task.run_id) if task.run_id is not None else None,
            state_version=int(task.state_version or 0),
            control_state=str(task.control_state or "idle"),
            input=cast(str | None, task.input),
            output=cast(str | None, task.output),
            error=cast(str | None, task.error_message),
            created_at=cast(datetime | None, task.created_at),
            updated_at=cast(datetime | None, task.updated_at),
            pending_question=pending_question,
            pending_interactions=pending_interactions,
            view=view,
            start_queued=start_queued,
        )


def _load_task_steps_version_snapshot(
    task_id: int,
    principal: ApiKeyPrincipal,
) -> _TaskStepsVersionSnapshot:
    """Authorize a task and detach the trace version for cache lookup."""

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = _resolve_task_or_404(task_id, principal, db)
        if task.conversation_storage_version == 2:
            from ...services.task_event_display import display_horizon

            return _TaskStepsVersionSnapshot(
                task_id=int(task.id),
                agent_id=int(task.agent_id),
                max_event_id=display_horizon(db, task_id),
                traces_expired_at=None,
                storage_version=2,
            )
        max_event_id = (
            db.query(func.max(TraceEvent.id))
            .filter(
                TraceEvent.task_id == task_id,
                TraceEvent.build_id.is_(None),
            )
            .scalar()
            or 0
        )
        return _TaskStepsVersionSnapshot(
            task_id=int(task.id),
            agent_id=int(task.agent_id),
            max_event_id=int(max_event_id),
            traces_expired_at=_read_traces_expired_at(task_id, db),
        )


def _load_task_steps_snapshot(
    task_id: int,
    principal: ApiKeyPrincipal,
) -> _TaskStepsSnapshot:
    """Reauthorize and detach the ordered trace rows after a cache miss."""

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = _resolve_task_or_404(task_id, principal, db)
        if task.conversation_storage_version == 2:
            from ...services.task_event_display import load_event_display_snapshot

            view = load_event_display_snapshot(db, task_id)
            return _TaskStepsSnapshot(
                task_id=int(task.id),
                agent_id=int(task.agent_id),
                max_event_id=view.horizon,
                traces_expired_at=None,
                storage_version=2,
                events=tuple(
                    _TraceEventSnapshot(
                        task_id=task_id,
                        event_id=event["event_id"],
                        event_type=event["event_type"],
                        timestamp=datetime.fromtimestamp(
                            event["timestamp"], timezone.utc
                        ),
                        step_id=event["step_id"],
                        data=event["data"],
                    )
                    for event in view.events
                    if event["type"] == "trace_event"
                ),
            )
        rows = (
            db.query(TraceEvent)
            .filter(
                TraceEvent.task_id == task_id,
                TraceEvent.build_id.is_(None),
            )
            .order_by(TraceEvent.id.asc())
            .all()
        )
        events = tuple(
            _TraceEventSnapshot(
                task_id=int(row.task_id),
                event_id=str(row.event_id),
                event_type=str(row.event_type),
                timestamp=cast(datetime, row.timestamp),
                step_id=str(row.step_id) if row.step_id is not None else None,
                data=row.data,
            )
            for row in rows
        )
        return _TaskStepsSnapshot(
            task_id=int(task.id),
            agent_id=int(task.agent_id),
            max_event_id=int(rows[-1].id) if rows else 0,
            traces_expired_at=_read_traces_expired_at(task_id, db),
            events=events,
        )


def _filter_interaction_descriptors(
    interactions: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """Drop non-mapping elements from a stored interactions list.

    ``PendingInteraction.interactions`` is typed ``list[dict[str, Any]] |
    None``. A row written by an older or buggy writer can carry a list
    with non-dict elements (e.g. a bare string); passing that straight
    into the schema fails Pydantic validation and 500s the GET
    permanently, for as long as the dirty row exists. Filtering here --
    not in the shared transcript reader, and not in
    ``task_interaction_read``'s tuple adapter either -- keeps the fix
    scoped to this read path's output contract; the other four consumers
    (websocket.py, chat.py) go through the same adapter untouched. Same
    semantics as react.py's ``_normalize_ask_user_interactions``: a
    non-dict element is dropped, not coerced or repaired.
    """
    if interactions is None:
        return None
    return [item for item in interactions if isinstance(item, dict)]


def _get_chat_task_sync(
    task_id: int,
    principal: ApiKeyPrincipal,
) -> TaskInfoResponse:
    """Build one task response without retaining a Session during cache I/O.

    ``pending_interaction`` is deliberately kept OUT of the cached
    response body and stitched on after every cache read/write. The
    cache entry's freshness token is ``tasks.updated_at`` (see
    ``task_updated_at`` below), but the pending question lives in
    ``task_chat_messages`` -- a write there does not touch
    ``tasks.updated_at``, so a cached entry has no way to prove its
    ``pending_interaction`` is still current. Caching it anyway would
    let a stale question survive a cache hit indefinitely.

    The terminal view of a status-only caller's task is kept out of the
    cache for the same reason: automatic resume stops by writing only the
    recovery row, and an append queues a START without touching the task
    row's ``updated_at``, so a cached body cannot prove the view is still
    current. The cached body holds the stored values; the view is applied
    to the response after every cache read or write. A queued START is
    applied the same way, for the same reason.
    """

    task = _load_task_info_snapshot(task_id, principal)
    completed_at = task.updated_at if task.status in _TERMINAL_STATUSES else None
    pending_interaction = (
        PendingInteraction(
            question=task.pending_question,
            interactions=_filter_interaction_descriptors(task.pending_interactions),
        )
        if task.pending_question is not None
        else None
    )
    overlay: dict[str, Any] = {"pending_interaction": pending_interaction}
    if task.start_queued:
        effective = effective_task_snapshot(task)
        overlay.update(
            status=effective.status.value,
            output=effective.output,
            error=effective.error,
            completed_at=None,
        )
    if task.view is not None:
        effective = effective_task_snapshot(task)
        overlay.update(
            status=effective.status.value,
            control_state=effective.control_state,
            output=effective.output,
            error=effective.error,
            completed_at=task.view.stopped_at,
        )
    cache_key = task_snapshot_key(task_id)
    task_updated_at = cache_version_token(task.updated_at)
    cached = cache_get(cache_key)
    if isinstance(cached, dict) and cached.get("updated_at") == task_updated_at:
        response = TaskInfoResponse.model_validate(cached["response"])
        return response.model_copy(update=overlay)

    response = TaskInfoResponse(
        task_id=task.task_id,
        agent_id=task.agent_id,
        workforce_id=(
            int(principal.workforce.id) if principal.workforce is not None else None
        ),
        status=task.status.value,
        run_id=task.run_id,
        state_version=task.state_version,
        control_state=task.control_state,
        input=task.input,
        output=task.output,
        error=task.error,
        created_at=task.created_at,
        completed_at=completed_at,
    )
    cache_set(
        cache_key,
        {
            "updated_at": task_updated_at,
            "response": response.model_dump(mode="json"),
        },
        ttl_seconds=task_cache_ttl_seconds(),
    )
    return response.model_copy(update=overlay)


def _get_chat_task_steps_sync(
    task_id: int,
    principal: ApiKeyPrincipal,
) -> StepsResponse:
    """Build public steps using two short authorization/read transactions."""

    version = _load_task_steps_version_snapshot(task_id, principal)
    cache_key = task_steps_key(task_id)
    cached = cache_get(cache_key)
    # An entry written before ``traces_expired_at`` existed has no such key
    # and reads as ``None``, which matches exactly the tasks retention never
    # touched -- the ones whose cached body is still right without the flag.
    if (
        isinstance(cached, dict)
        and cached.get("storage_version", 1) == version.storage_version
        and cached.get("max_event_id") == version.max_event_id
        and cached.get("traces_expired_at")
        == cache_version_token(version.traces_expired_at)
    ):
        return StepsResponse.model_validate(cached["response"])

    snapshot = _load_task_steps_snapshot(task_id, principal)
    public_steps_data = map_trace_events_to_public_steps(list(snapshot.events))
    response = StepsResponse(
        task_id=snapshot.task_id,
        agent_id=snapshot.agent_id,
        steps=[PublicStep(**step) for step in public_steps_data],
        steps_expired=snapshot.traces_expired_at is not None,
        steps_expired_at=snapshot.traces_expired_at,
    )
    cache_set(
        cache_key,
        {
            "storage_version": snapshot.storage_version,
            "max_event_id": snapshot.max_event_id,
            "traces_expired_at": cache_version_token(snapshot.traces_expired_at),
            "response": response.model_dump(mode="json"),
        },
        ttl_seconds=task_cache_ttl_seconds(),
    )
    return response


@router.post(
    "/chat/tasks/{task_id}/messages",
    status_code=202,
    response_model=AppendMessageResponse,
)
async def append_message_to_task(
    task_id: int,
    request: AppendMessageRequest,
    principal: ApiKeyPrincipal = Depends(get_principal_from_api_key),
) -> AppendMessageResponse:
    """Append the next user message to an existing task and kick off its next turn.

    Phase 1 multi-turn model is task-centric: subsequent user inputs
    extend the same ``task_id`` rather than creating a new task or a
    new ``conversation_id``. Works for every key type: an agent-bound
    key appends to its agent's SDK tasks, a workforce-bound key appends
    to the manager tasks behind its workforce's SDK runs (the workforce
    turn guard revalidates archive/config drift on every append), and a
    personal key appends to the tasks its holder created with a personal
    key. This endpoint:

      1. Validates the path ``task_id`` exists and belongs to the
         key-bound owner (404 ``task_not_found`` otherwise).
      2. Validates ``body.agent_id`` matches the key-bound agent
         (404 ``agent_not_found`` otherwise; required for agent keys,
         rejected for workforce keys, optional for personal keys, where
         it must be the task's agent). ``body.workforce_id``, when
         provided with a workforce key, must match the bound workforce
         (404 ``workforce_not_found`` otherwise); a personal key never
         passes one. For a personal key, the holder must still be able
         to see the task's agent (404 ``agent_not_found`` otherwise),
         checked before ``prompt_context`` and connector input.
      3. Rejects the call with 409 ``task_busy`` if the task is
         currently ``RUNNING``, or 409 ``interaction_response_required``
         if it is ``WAITING_FOR_USER`` -- that status answers a pending
         agent question and is handled by
         ``POST /v1/chat/tasks/{id}/reply`` instead. Workforce turn
         rejections map to their own stable codes (``workforce_archived``
         / ``workforce_config_changed``, 409 -- NOT retryable) so clients
         aren't told to retry a permanently-rejected conversation.
      4. Otherwise persists the new user message to
         ``task_chat_messages``, updates ``task.input`` to record
         this turn's input, and kicks off the next background turn
         via the same helper POST uses.

    Args:
        task_id: Path parameter; the target task's primary key.
        request: Validated :class:`AppendMessageRequest`. ``message.content``
            is guaranteed non-empty by Pydantic.
        principal: Key-bound owner from the auth dependency.
    Returns:
        :class:`AppendMessageResponse` with the task identity and an
        ``accepted_at`` timestamp.

    Raises:
        V1ApiError 401: missing / invalid / revoked key.
        V1ApiError 404: task not found OR not owned by the key OR
            body.agent_id / body.workforce_id doesn't match the bound
            owner OR a personal key's holder can no longer see the agent.
        V1ApiError 410: ``task_expired`` -- the retention policy expired
            a task this key could have seen; ``details`` carries
            ``task_id`` and ``expired_at``.
        V1ApiError 422: ``prompt_context`` is malformed, sent with a
            workforce key, or, with a personal key, has a field the agent
            does not declare or longer than its declared length.
        V1ApiError 409: ``task_busy`` (retryable), a workforce rejection
            code (not retryable), or ``interaction_response_required``
            (use ``reply`` instead).
        500: any other unexpected error (V1 envelope via global handler).
    """
    _validate_prompt_context(request.prompt_context)
    if request.prompt_context is not None and principal.workforce is not None:
        raise V1ApiError(
            V1ErrorCode.INVALID_INPUT,
            422,
            message="prompt_context is not supported for workforce keys.",
        )
    try:
        started = await task_start_service.append_sdk_turn(
            task_id=task_id,
            scope=_sdk_task_scope(principal),
            actor_user_id=principal.owner_user_id,
            request_agent_id=request.agent_id,
            request_workforce_id=request.workforce_id,
            message=request.message.content,
            file_ids=tuple(request.message.files or ()),
            connector_runtime_context=tuple(
                item.model_dump() for item in request.connector_runtime_context or ()
            ),
            prompt_context=request.prompt_context,
        )
    except task_start_service.TaskStartRejected as exc:
        _raise_start_rejection(exc)
    except TaskTurnNotFoundError:
        raise V1ApiError(V1ErrorCode.TASK_NOT_FOUND, 404)
    except TaskTurnError as exc:
        raise_for_turn_rejection(exc.reason)
    except TaskTurnFileBindingError as exc:
        _raise_v1_file_binding_error(exc)
    except ConnectorRuntimeError as exc:
        _raise_v1_connector_runtime_error(exc)
    except DurableStorageOperationError as exc:
        _raise_v1_storage_unavailable(exc)

    await record_principal_usage(
        principal,
        route="append_message",
        task_id=started.task_id,
        agent_id=started.agent_id,
    )

    return AppendMessageResponse(
        task_id=started.task_id,
        agent_id=started.agent_id,
        workforce_id=(
            int(principal.workforce.id) if principal.workforce is not None else None
        ),
        status=started.status.value,
        accepted_at=started.accepted_at,
        run_id=started.run_id,
        state_version=started.state_version,
        control_state=started.control_state,
    )


@router.post(
    "/chat/tasks/{task_id}/reply",
    status_code=202,
    response_model=ReplyResponse,
)
async def reply_to_waiting_task(
    task_id: int,
    request: ReplyRequest,
    principal: ApiKeyPrincipal = Depends(get_principal_from_api_key),
) -> ReplyResponse:
    """Answer the agent's pending question on a ``waiting_for_user`` task.

    Route declaration lives here alongside the other task endpoints;
    the implementation is in ``task_reply.py`` because it resumes an
    existing run rather than claiming a new turn -- a different
    lifecycle from create/append that deserves its own module. See
    :func:`xagent.web.api.v1.task_reply.reply_to_task` for the full
    contract (accepted states, error codes, fail-closed checkpoint
    handling).

    Imported inside the handler (not at module level) because
    ``task_reply`` imports ownership-resolution helpers back from this
    module; deferring the import here breaks that cycle without
    duplicating those helpers.
    """
    from .task_reply import reply_to_task

    return await reply_to_task(task_id=task_id, request=request, principal=principal)


@router.delete("/chat/tasks/{task_id}", response_model=DeleteTaskResponse)
async def delete_chat_task(
    task_id: int,
    principal: ApiKeyPrincipal = Depends(get_principal_from_api_key),
) -> DeleteTaskResponse:
    """Delete one SDK task and everything tied to it.

    The task is resolved through the same ownership rule as
    ``GET /v1/chat/tasks/{task_id}``: a task of another agent, another
    user's agent, or a nonexistent id all answer the same 404. A task
    that is executing, holds an unexpired execution lease, or has an
    accepted command still queued answers 409 ``task_busy`` instead of
    being deleted out from under its executor; wait for it to finish
    and retry. The final re-check runs under a row lock, which is
    effective on PostgreSQL only; SQLite leaves a small window between
    the check and the delete.

    The owned rows (trace events/blobs/checkpoints, chat messages, DAG
    executions, interaction requests, connector runtime context, runtime
    extension bindings), the uploaded-file bindings, and the workspace
    are removed through the same helper the Web UI delete uses. External
    cleanup that fails is recorded as a retryable obligation, never a
    half state reported as success: ``workspace_cleanup_pending`` and
    ``external_cleanup_pending`` in the response say whether anything is
    still owed. Deleting twice answers 404.

    Raises:
        V1ApiError 401: missing / invalid / revoked key.
        V1ApiError 404: task missing or not owned by the calling key.
        V1ApiError 409: ``task_busy`` -- running or queued, retry after it ends.
        V1ApiError 410: ``task_expired`` for a retention-expired task.
        V1ApiError 503: a bound runtime extension could not release its
            state; the task was not deleted.
    """
    snapshot = await run_db_io_cancellation_safe(
        lambda: _load_task_delete_snapshot(task_id, principal)
    )
    try:
        committed = await commit_task_deletion(snapshot, refuse_if_active=True)
    except TaskDeleteBusy as exc:
        raise V1ApiError(
            V1ErrorCode.TASK_BUSY,
            409,
            message=(
                "Task became active during deletion; wait for it to finish "
                "before deleting it."
            ),
        ) from exc
    except TaskExtensionCleanupRejected as exc:
        raise V1ApiError(
            V1ErrorCode.INTERNAL_ERROR,
            503,
            message="Runtime extension cleanup failed; the task was not deleted.",
        ) from exc
    if committed is None:
        raise V1ApiError(V1ErrorCode.TASK_NOT_FOUND, 404)

    # The rows are committed as gone: detach this task's WebSocket
    # connections before the next await, then stop any in-process executor and
    # close the sockets before the workspace is reclaimed, as the Web UI
    # delete does. The guards above refuse a task that is leased or has queued
    # work, so this only reaches executors that started in the window the
    # pre-checks cannot close. The cleanup runs as its own task and is
    # shielded, so a cancelled request cannot leave sockets detached but open.
    from ...services.task_execution import background_task_manager
    from ..websocket import manager as websocket_manager

    connections = websocket_manager.detach_task_connections(task_id)

    async def _cleanup_runtime_state() -> None:
        await background_task_manager.cancel_task(task_id, timeout_seconds=0.05)
        for connection in connections:
            try:
                await connection.close()
            except Exception:
                logger.warning("Failed to close WebSocket connection", exc_info=True)

    await asyncio.shield(asyncio.create_task(_cleanup_runtime_state()))
    outcome = await finish_task_deletion(committed)
    await record_principal_usage(
        principal, route="delete_task", task_id=task_id, agent_id=None
    )
    return DeleteTaskResponse(
        task_id=task_id,
        workspace_cleanup_pending=outcome.workspace_cleanup_pending,
        external_cleanup_pending=outcome.external_cleanup_pending,
    )


@router.post(
    "/chat/tasks/{task_id}/cancel-question",
    status_code=202,
    response_model=CancelQuestionResponse,
)
async def cancel_waiting_task_question(
    task_id: int,
    principal: ApiKeyPrincipal = Depends(get_principal_from_api_key),
) -> CancelQuestionResponse:
    """Abandon the pending question on a ``waiting_for_user`` task.

    Reuses ``GET``'s ownership rule (another agent's task, another user's
    agent, a missing id, or a non-SDK task all answer 404). It does not
    re-check, for a personal key, that the holder can still see the
    task's agent: ending a question starts no work with it. The pending
    interaction row is terminated rather than deleted, the paused run is
    finalized to ``completed``, and the same task then accepts
    ``POST /v1/chat/tasks/{id}/messages`` (fresh connector context, same
    conversation). Repeating the call on the already-cancelled run returns
    the same 202 body; any other non-waiting state answers 409.

    A concurrent ``POST .../reply`` that claimed the answer slot first makes
    this answer 409 ``task_busy``. Once this endpoint wins, a reply that
    admitted against the old waiting state can still answer 409
    ``task_busy`` when its own compare-and-swap loses; one that reads the
    task afterwards answers 409 ``no_pending_interaction``.

    Raises:
        V1ApiError 401: missing / invalid / revoked key.
        V1ApiError 404: task missing, not owned by the key, or not SDK-created.
        V1ApiError 409: ``task_busy`` while a reply is in flight;
            ``task_not_waiting`` for any other non-waiting state.
        V1ApiError 410: ``task_expired`` for a retention-expired task.
    """
    from ...services.task_question_cancel import (
        TaskQuestionCancelBusy,
        TaskQuestionCancelNotWaiting,
        cancel_task_question,
    )

    try:
        result = await cancel_task_question(
            task_id=task_id,
            scope=_sdk_task_scope(principal),
        )
    except task_start_service.TaskStartRejected as exc:
        _raise_start_rejection(exc)
    except TaskQuestionCancelBusy as exc:
        raise V1ApiError(
            V1ErrorCode.TASK_BUSY,
            409,
            message="A reply is already in flight for this task; retry later.",
        ) from exc
    except TaskQuestionCancelNotWaiting as exc:
        raise V1ApiError(V1ErrorCode.TASK_NOT_WAITING, 409) from exc
    except TaskTurnError as exc:
        raise_for_turn_rejection(exc.reason)

    await record_principal_usage(
        principal, route="cancel_question", task_id=result.task_id, agent_id=None
    )
    return CancelQuestionResponse(
        task_id=result.task_id,
        status=result.status,
        run_id=result.run_id,
        state_version=result.state_version,
    )


@router.get("/chat/tasks/{task_id}", response_model=TaskInfoResponse)
async def get_chat_task(
    task_id: int,
    principal: ApiKeyPrincipal = Depends(get_principal_from_api_key),
) -> TaskInfoResponse:
    """Return a snapshot of one task's current state.

    SDK clients call this to poll a previously-submitted task for
    its status, latest output, or failure reason. The shape is
    deliberately flat -- detailed step-by-step execution data lives
    behind ``GET /v1/chat/tasks/{task_id}/steps``.

    Args:
        task_id: Path parameter; the target task's primary key.
        authed: ``(Agent, AgentApiKey)`` tuple.
    Returns:
        :class:`TaskInfoResponse` with ``task_id``, ``agent_id``,
        ``status``, latest-turn ``input`` / ``output`` / ``error``,
        ``created_at``, and ``completed_at`` (set only when the task
        has reached a terminal state).

    Raises:
        V1ApiError 401: missing / invalid / revoked key.
        V1ApiError 404: task missing or not owned by the calling key.
        V1ApiError 410: ``task_expired`` -- the retention policy expired
            a task this key could have seen; ``details`` carries
            ``task_id`` and ``expired_at``.
    """
    return await run_db_io_cancellation_safe(
        lambda: _get_chat_task_sync(task_id, principal)
    )


@router.get("/chat/tasks/{task_id}/steps", response_model=StepsResponse)
async def get_chat_task_steps(
    task_id: int,
    principal: ApiKeyPrincipal = Depends(get_principal_from_api_key),
) -> StepsResponse:
    """Return the public-timeline steps for a task.

    Pulls all :class:`TraceEvent` rows for the task in DB order, then
    collapses them via :func:`map_trace_events_to_public_steps` into
    the 4 stable public step types: ``thinking``, ``tool_call``,
    ``agent_delegation``, ``message``.

    The internal trace event taxonomy has ~32 ``event_type`` strings
    today; SDK callers see only the 4 types listed above. Internal
    events not on the public allow-list (LLM calls, memory ops,
    visualization ticks, most DAG bookkeeping) are silently dropped --
    intentionally, so internal trace evolution doesn't break the SDK
    contract; the exceptions are dag_execution's planning/replanning/
    executing phase transitions, which project onto a planning
    thinking step, and a literal ``status="failed"`` on
    dag_execute_end, which closes a still-open planning step as
    ``failed`` instead of leaving it running forever (a plan-generation
    exception that escapes DAGPattern.run() before it emits
    dag_execute_end still leaves the planning step running -- see
    ``_step_mapping.py``'s module docstring).

    Args:
        task_id: Path parameter; the target task's primary key.
        authed: ``(Agent, AgentApiKey)`` tuple resolved by the auth
            dependency. The agent here is the key-bound agent.
    Returns:
        :class:`StepsResponse` with ``task_id``, ``agent_id``, and the
        steps array in ``started_at`` ascending order. In-flight steps
        appear with ``status='running'`` and ``completed_at=null`` so
        SDK clients can poll this endpoint and observe progress.
        ``steps_expired`` / ``steps_expired_at`` report that the
        retention policy removed the task's historical steps, so the
        array may be incomplete -- it can still hold steps from turns
        taken after that removal.

    Raises:
        V1ApiError 401: missing / invalid / revoked key.
        V1ApiError 404: task missing or not owned by the calling key.
        V1ApiError 410: ``task_expired`` -- the retention policy expired
            the whole task (not just its steps); ``details`` carries
            ``task_id`` and ``expired_at``.
    """
    return await run_db_io_cancellation_safe(
        lambda: _get_chat_task_steps_sync(task_id, principal)
    )


@router.get("/chat/tasks/{task_id}/files", response_model=TaskFilesResponse)
async def get_chat_task_files(
    task_id: int,
    principal: ApiKeyPrincipal = Depends(get_principal_from_api_key),
) -> TaskFilesResponse:
    """List the files bound to a task.

    Covers both files the agent produced in the task workspace and files
    attached to one of its turns. Authorization is the same task scope every
    other ``/v1/chat/tasks/{task_id}/*`` route uses.

    Raises:
        V1ApiError 401: missing / invalid / revoked key.
        V1ApiError 404: task missing or not accessible with this key.
        V1ApiError 410: the task was expired by the retention policy.
    """
    return await run_db_io_cancellation_safe(
        lambda: _list_chat_task_files_sync(task_id, principal)
    )


@router.get("/chat/files/{file_id}/download", response_model=None)
async def download_chat_file(
    file_id: str,
    principal: ApiKeyPrincipal = Depends(get_principal_from_api_key),
) -> Response:
    """Download one file as an attachment.

    A task-bound file is served only when its task is reachable through this
    key; an unbound upload only to the user who uploaded it. Failures use the
    stable v1 error envelope and answer 404 rather than 403.
    """
    from ..files import build_registered_file_download_response

    snapshot = await run_db_io_cancellation_safe(
        lambda: _load_download_file_snapshot(file_id, principal)
    )
    try:
        return await build_registered_file_download_response(
            snapshot,
            owner_user_id=int(snapshot.user_id),
            disposition="attachment",
        )
    except HTTPException as exc:
        _raise_v1_file_download_error(exc)


@router.get("/chat/tasks/{task_id}/events")
async def stream_chat_task_events(
    task_id: int,
    principal: ApiKeyPrincipal = Depends(get_principal_from_api_key),
) -> StreamingResponse:
    """Stream a task's lifecycle and step/message content as Server-Sent
    Events.

    This endpoint declares no ``responses=`` schema, so the field lists
    below are the only OpenAPI-visible contract for its 8 SSE event
    types (each a ``event: <name>`` / ``data: <json>`` frame pair). Four
    are lifecycle-only:
      - ``task.status``: ``{status}``.
      - ``task.completed``: ``{status, output, error}``. A failed task
        also ends the stream with this event rather than a separate
        one -- ``status`` is ``"failed"`` and ``error`` is set.
      - ``task.input_required``: ``{task_id, prompt}``. ``prompt`` is the
        agent's pending question when one is on record (the same value
        ``GET /v1/chat/tasks/{task_id}`` returns as
        ``pending_interaction.question``), or ``null`` if none is.
        Earlier versions of this endpoint sent ``null`` here in every
        case, so a client that assumed the field was never populated
        must accept a string: the field's name and type (nullable
        string) are unchanged, only the value can now be present.
        Answer it with ``POST /v1/chat/tasks/{task_id}/reply`` -- calling
        ``POST .../messages`` (append) on a task in this state 409s.
      - ``stream.error``: ``{code, message}``, a flat shape distinct
        from the nested error envelope ``V1ApiError`` HTTP responses use
        elsewhere in this router; the two error-code vocabularies do
        not overlap.

      On the two attach-time fast paths described under Behavior below,
      whichever of these two conclusion frames the path emits carries two
      extra fields, ``snapshot_truncated: true`` and
      ``snapshot_total_steps`` (the task's full public step count at
      attach time), whenever that attach's one-shot step snapshot sent
      fewer steps than the task has -- absent otherwise, and absent on
      every other producer of these two frames. They ride the conclusion
      rather than a ``step.*`` frame because the snapshot's byte budget
      can admit no steps at all, leaving no ``step.*`` frame to carry
      them.

    The other four project the task's step-by-step execution and
    streamed message text onto the same connection, from the same
    ``PublicStep`` shape and public step-type list ``GET
    .../steps`` uses (``thinking``/``tool_call``/``agent_delegation``/
    ``message``):
      - ``step.started``: ``{step}``, a running ``PublicStep``.
      - ``step.completed``: ``{step}``, the same step once its end event
        (or failure) resolves it -- ``status`` is ``"completed"`` or
        ``"failed"``. On the two attach-time fast paths below (the task
        is already finished, or already waiting on user input), these
        frames are that attach's one-shot step snapshot: in
        ``started_at`` order, drawn from the task's most recent steps
        up to a step-count cap (512) and a total-wire-bytes budget over the
        snapshot's serialized frames. When the byte budget is what binds,
        it keeps that window's oldest contiguous run, so the very latest
        steps may be the ones missing. Whether the snapshot was cut short
        is reported on the conclusion frame, not on these -- ``GET
        .../steps`` is the authoritative full history either way.
      - ``message.delta``: ``{message_id, text}``, one chunk of a
        streamed final answer as the agent generates it.
      - ``message.completed``: ``{message_id, content}``, that same
        message's full text once the stream ends successfully. A
        ``message.delta`` sequence is not guaranteed a matching
        ``message.completed`` -- see Behavior below.
        Each frame is capped on its own and the delta sequence has no
        aggregate cap, so on a long answer a ``message.completed``
        carrying ``"truncated": true`` can hold less text than the
        deltas the client already accumulated. It is not authoritative
        over them: prefer the accumulated deltas, or ``steps()`` for the
        full persisted text.
      A ``message`` ``PublicStep``'s ``id`` cannot be correlated between
      this stream and ``GET .../steps``: the live frame mints a fresh id
      per broadcast, while ``steps()`` returns the persisted trace
      event's own id as that step's id.
      A ``thinking`` step whose id begins with ``thinking:plan:`` or
      ``thinking:planning:`` cannot be correlated either, and is the one
      case where trying to actively corrupts client state. Both ids end
      in a count of the planning cycles the projection has seen, and
      this stream's projection starts empty at attach: it numbers the
      first planning cycle it observes ``:1`` however many already
      happened, while ``steps()`` counts from the start of the task's
      persisted history. A client attaching during a task's second
      planning cycle is therefore sent ``thinking:plan:{task_id}:1`` for
      a step ``steps()`` calls ``thinking:plan:{task_id}:2`` -- and
      ``thinking:plan:{task_id}:1`` already exists in ``steps()`` as the
      first cycle, a different step. Merging the two surfaces by id
      overwrites one with the other. These ids are stable within one
      connection and nowhere else: a re-attach after ``stream_expired``
      (the 1-hour cap) or ``resync_required`` (queue overflow) opens a
      new connection whose count starts over. Reconcile planning steps
      against ``steps()`` by ``started_at`` and content, not by id.
      Every other public step type -- ``tool_call``,
      ``agent_delegation``, and ``thinking`` steps whose id carries the
      originating step id rather than a count -- keeps the same id
      across both surfaces. That holds because those ids are derived
      from a key the event itself carries: a tool invocation's own id,
      or the id of the step that produced the event. Both surfaces read
      that same key off that same event, which is what makes the two
      ids equal rather than merely similar. An event carrying neither
      key would instead be identified by a per-delivery id that differs
      between the two surfaces and so would not correlate; no event
      type this stream projects reaches that state.
      A final answer the agent streams live reaches the client twice on
      this stream: once as its ``message.delta``/``message.completed``
      sequence, and again as a ``message``-type ``step.completed`` once
      the underlying ``ai_message`` trace event folds in -- the same
      step ``steps()`` already shows for that row. This is duplication,
      not loss: a client that already rendered the delta/completed
      sequence can ignore the matching step, or use it as the
      authoritative record instead, but should not expect the two to be
      mutually exclusive.
      A ``step.*``/``message.*`` frame whose content-bearing field would
      exceed a per-frame byte cap has that field truncated (or, for a
      step's structured ``data``, replaced) and flagged with
      ``"truncated": true`` rather than sent whole. The cap is 64 KiB of
      the field's own JSON wire form. A replaced step ``data`` carries
      ``truncated: true``, ``original_bytes`` (what the original
      sub-object would have measured on the wire), and -- as long as
      they fit -- that step type's identifying key: ``name`` for a
      ``tool_call``, ``sub_agent_name`` for an ``agent_delegation``,
      ``phase`` for a ``thinking`` step, ``role`` for a ``message``. So
      a client can render "the ``search`` tool ran and its result was
      too large" rather than "a step was too large". A truncated
      ``message.delta``/``message.completed`` carries the shortened text
      itself plus ``truncated: true`` and no ``original_bytes``. A raw
      broadcast frame that runs past 256 KiB -- measured without the
      internal task-description stamp such frames carry -- is dropped
      whole instead of truncated (see Behavior below): no marker, no
      content.
      This cap is specific to this stream -- ``GET .../steps`` applies
      no such cap and always returns a step's full, untruncated
      ``data``, so the two surfaces can disagree on content for the
      same step once it's large enough to trip this stream's cap.

    Behavior:
      - Normal attach: opens the stream, emits ``task.status`` for the
        task's current state, then goes live -- a status update each
        time the task's status changes (consecutive duplicates are
        suppressed), ``step.*``/``message.*`` frames as new trace
        events and streamed answer chunks arrive, and
        ``task.completed`` when the task finishes. A ``: ping`` comment
        line is sent whenever 15s pass without any other frame going
        out, to keep the connection alive -- not on a fixed 15s
        cadence regardless of activity.
      - A stream that goes live carries only what happens after the
        connection opens; it never resends the steps that already
        happened. (The two attach-time fast paths below are the
        exception: they send a one-shot snapshot of the task's steps
        and close, rather than going live at all.) That has one
        consequence a client must handle: a step that was already
        running at attach time does not appear on this stream at all.
        Its ``step.started`` was broadcast before this connection
        existed, and its end event arrives here with no matching
        start, so that end is dropped rather than sent -- the client
        sees neither half of the step. ``GET .../steps`` reads the
        database and is unaffected by when the stream was opened, so a
        client attaching to an already-running task reconciles there
        rather than expecting this stream to fill the gap.
      - Frame sequence into ``task.completed``: a task that fails emits
        ``task.status`` (``"failed"``) from the failure broadcast first,
        then the watchdog's authoritative ``task.completed`` close
        frame. Delivery of every ``step.*``/``message.*`` content frame
        on this stream is best-effort, not guaranteed, for four
        separate reasons: (1) closing a stream for any reason drains its
        queued backlog before inserting the close frame, so any
        already-queued frame -- a ``task.status``, a ``step.*``, a
        ``message.*`` -- can be dropped rather than delivered; this is
        not specific to a failure closing the stream, it's the
        queue-drain behavior every close goes through. (2) one content
        frame that fails to project (a malformed field the projector
        can't fold) is dropped on its own -- it does not close the
        stream or affect any other frame, so a step or message can go
        missing from an otherwise unremarkable stream. (3) an oversized
        raw broadcast frame is dropped whole rather than projected,
        so its content never reaches the per-field truncation cap
        described above and leaves no ``"truncated": true`` marker --
        this is a full drop, not a truncation. The size measured
        excludes the task description that the broadcast conversion
        stamps onto every trace event, so a task's description alone
        can no longer blank out its step stream this way -- a frame is
        only dropped here when its content, apart from that field, is
        still over the cap. (4) a planning step can be left unresolved:
        when a planning round ends without emitting its own terminal
        event (a plan-generation error escaping before it), the next
        round clears the pairing key that step was waiting on, so the
        step's ``step.completed`` is never produced and the client
        keeps a ``thinking`` step at ``"running"`` for the rest of the
        stream's life (``_step_mapping.py``'s ``dag_execute_start``
        branch). This one is not a stream-only artifact: ``GET
        .../steps`` folds the same events and shows that step the same
        way. (2), (3) and (4) are silent: none of them ever produces a
        ``stream.error`` frame or any other signal on the wire. (1) is
        not silent in the same way -- the
        close frame that follows a queue-drain is often itself a
        ``stream.error`` (``resync_required``/``unauthorized``/
        ``task_deleted``/``task_expired``/``stream_expired``) -- but that
        close frame carries no record of what it drained on its way out, so its
        presence or absence proves nothing about whether a content frame
        was lost this way: a ``stream.error`` does not mean content was
        dropped, and its absence does not mean nothing was. Only
        reconciling against ``steps()`` answers that. No failure
        information is lost when a status frame is dropped this way --
        ``task.completed`` still carries ``status: "failed"`` and a
        populated ``error`` -- but a dropped content frame has no such
        backstop: a ``message.delta`` sequence can end with no
        ``message.completed``, and a step can be left ``"running"`` on
        this stream even though it actually resolved. Reconcile via
        ``steps()`` for the authoritative picture; don't treat "no
        content frame arrived" as "nothing happened". A task that
        succeeds has no intermediate status broadcast, so it goes
        straight to ``task.completed`` with no preceding
        ``task.status``. Attaching to a task that's already terminal
        (the fast path below) always sends the conclusion frame; the
        generation reread below decides only whether step content
        accompanies it, sending ``resync_required`` in place of the
        steps on a confirmed change.
      - Attaching to an already-finished task is not an error: the
        stream opens, emits ``task.status``, the task's steps, then
        ``task.completed``, and closes immediately. If reading those
        steps fails, this fast path (and the waiting-for-user one
        below) still sends its conclusion frame -- ``task.completed``
        here, ``task.input_required`` there -- before closing with a
        ``stream.error`` frame naming why -- ``task_deleted`` when the
        row is gone by the time the steps are read (``task_expired``
        when the retention policy is what removed it),
        ``resync_required`` otherwise: a step-read failure on either
        fast path never costs the client the lifecycle conclusion it already had in
        hand from the snapshot that picked this path. Step content goes
        out at all only once the task row has been read once more and
        confirmed to still be the same run/state generation as the
        snapshot that picked this path, *and* the steps cursor
        (``max_event_id``) is confirmed unmoved since -- a task that
        restarted in between (a ``POST reply`` resuming a waiting task,
        or a WS append resuming a finished one) moves the row to a new
        generation, which means the steps just read may already belong
        to it while the conclusion still describes the old one; a trace
        row landing in that same window can advance the cursor without
        touching ``run_id``/``state_version`` at all (trace rows write
        through their own commit, which can update the task row's own
        checkpoint-pointer columns while the task is running but never
        touches those two fields), so the generation check alone cannot
        see it. Both checks have to pass for the steps to go out: a
        confirmed match on both sends the steps, then the conclusion; a
        confirmed change on either still sends the conclusion but
        withholds the steps, closing with ``resync_required`` instead;
        and a reread that fails outright is treated like the steps-read
        failure above -- no step content goes out, but the conclusion
        (already known-good, independent of this reread) still does,
        followed by ``stream.error`` naming why -- ``task_deleted``
        (``task_expired`` when retention expired the task) or
        ``resync_required`` by the same rule. A failure while
        serializing an individual step is handled the same way: the
        conclusion frame goes out, followed by
        ``stream.error(resync_required)``, and the step list sent
        before the failure may be partial.
      - The stream force-closes with ``task.input_required`` if the task
        is found waiting on user input (same steps-then-close shape:
        ``task.status``, the task's steps, then ``task.input_required``);
        with ``stream.error`` if the API key is revoked/paused (a personal
        key: revoked or expired), the task
        row disappears (within one watchdog cycle, 30s in production, of
        the delete: ``task_expired`` when the retention policy expired
        it, ``task_deleted`` for any other removal), the client can't
        keep up (``resync_required``), or the stream has been open for the 1-hour maximum
        (``stream_expired``, emitted before the connection closes so
        it's distinguishable from a clean end). After any
        ``stream.error``, re-attaching is the supported recovery path
        (no replay of missed frames -- reconcile via ``steps()`` first).
      - A task that's ``paused`` does not close the stream (matching
        SDK ``wait()`` semantics: it may still be resumed). A paused
        task whose automatic resume has stopped for good, with nothing
        queued that could move it, is read as ``failed`` and closes
        with ``task.completed`` within one watchdog cycle; the 1-hour
        cap ends a paused stream that nothing resumes or stops.
      - No ``task.status`` frame is guaranteed to be fresh or in order,
        at any point in the stream's life, not only at attach: this
        endpoint never buffers or reconciles ``task.status`` frame
        order against anything. Accepted because only the three close
        frames above are treated as authoritative; each comes from a
        direct read of the task row, never from frame ordering.

    Args:
        task_id: Path parameter; the target task's primary key.
        principal: Resolved by the auth dependency; also used (not as
            an authorization gate) for the per-key concurrency count
            and the periodic key-validity check.

    Raises:
        V1ApiError 401: missing / invalid / revoked key (plain JSON;
            the stream never opens).
        V1ApiError 404: task missing or not owned by the calling key
            (plain JSON; the stream never opens).
        V1ApiError 410 ``task_expired``: the retention policy expired a
            task this key could have seen (plain JSON with ``details``
            ``task_id`` / ``expired_at``; the stream never opens).
        V1ApiError 429 ``rate_limited``: 2 or more concurrent streams
            already open on this task, or 32 or more already open for
            this key across all tasks (the per-key cap). Both caps are
            checked once, before the attach's own stream registers, so
            both are best-effort under a concurrent attach burst:
            several attaches can pass the check at the same instant,
            so the number of streams actually open can briefly exceed
            the cap -- the per-key cap's own counter never does, since
            it checks before it increments; it's specifically the
            count of open streams that can run ahead of it. No stream
            is aborted once opened; both counts self-heal as open
            streams close. Both caps are per-process: a multi-worker
            deployment counts independently in each worker process, so
            the effective cluster-wide limit is the per-process cap
            times the worker count. An attach that takes the terminal
            or waiting-for-user fast path (see Behavior above) never
            registers a sink at all, so it never counts toward either
            cap -- its step snapshot read goes through the same
            ``max_event_id``-keyed cache ``GET .../steps`` uses. That
            snapshot is bounded by a step-count cap and a
            total-wire-bytes budget; a snapshot cut short by either
            carries the ``snapshot_truncated``/``snapshot_total_steps``
            marker on its conclusion frame as described above. When a
            shared cache backend is configured (Redis; see
            ``hot_path_cache.get_cache_backend``), a burst of fast-path
            attaches on one task (or repeated attaches after it's
            already terminal) usually collapses into cache hits rather
            than each one re-reading and re-projecting the task's full
            trace history. Without one configured -- ``NoOpCache``, what
            this reads through whenever Redis is disabled -- every
            attach re-reads and re-projects the task's full trace
            history regardless of this shared code path; the cache-hit
            collapsing only actually happens with a real backend behind
            it. Broadcast delivery is per-process
            too: a sink only receives the broadcasts fanned out by its
            own worker
            process's connection manager, so a task transition driven
            by a different worker reaches this stream only once the
            watchdog's 30s database poll picks it up, not via broadcast.
            That poll only ever reads task status, though, never step
            content: an attach routed to a different worker than the one
            executing the task still gets the authoritative close frame
            (the watchdog reads the task row), but receives none of the
            live ``step.*``/``message.*`` frames produced in between --
            there is no equivalent fallback delivery for content the way
            there is for a status transition, so such an attach
            degrades to the lifecycle-only shape. Some transitions --
            lease-expiry recovery among them -- never broadcast at all,
            in any deployment shape, so the watchdog poll is their only
            delivery path regardless of worker count.
    """
    snapshot = await run_db_io_cancellation_safe(
        lambda: effective_task_snapshot(_load_task_info_snapshot(task_id, principal))
    )
    return await _events_stream.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=lambda tid, p: effective_task_snapshot(
            _load_task_info_snapshot(tid, p)
        ),
        read_task_steps_response=_get_chat_task_steps_sync,
        read_task_steps_version=_load_task_steps_version_snapshot,
    )
