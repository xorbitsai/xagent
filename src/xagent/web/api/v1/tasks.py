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
``POST /v1/workforces/{id}/runs``). Task creation stays agent-only.
All responses use the stable ``V1ApiError`` envelope. Task turn
lifecycle (claim RUNNING, persist messages, schedule bg, sync output)
is delegated to ``services.task_orchestrator.TaskTurnOrchestrator``,
which is also used by the WebSocket UI path so both transports share
one state machine.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, NoReturn, Optional, cast

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from ....core.tools.adapters.vibe.connector_runtime import ConnectorRuntimeError
from ...config import is_allowed_file
from ...models.database import get_session_local
from ...models.task import Task, TaskStatus, TraceEvent
from ...schemas.v1 import (
    AppendMessageRequest,
    AppendMessageResponse,
    CreateTaskRequest,
    CreateTaskResponse,
    PendingInteraction,
    PublicStep,
    ReplyRequest,
    ReplyResponse,
    StepsResponse,
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
from ...services.task_interaction_read import get_pending_interaction_question
from ...services.task_orchestrator import (
    TaskTurnError,
    TaskTurnFileBindingError,
    TaskTurnNotFoundError,
)
from . import _events_stream
from ._step_mapping import map_trace_events_to_public_steps
from .deps import ApiKeyPrincipal, get_principal_from_api_key, record_key_usage
from .errors import V1ApiError, V1ErrorCode, raise_for_turn_rejection

router = APIRouter()

_CONNECTOR_RUNTIME_SETUP_FAILED_MESSAGE = "Connector runtime setup failed."


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
    return task_start_service.SdkTaskScope(
        agent_id=int(principal.agent.id) if principal.agent is not None else None,
        workforce_id=int(principal.workforce.id)
        if principal.workforce is not None
        else None,
    )


def _raise_start_rejection(exc: task_start_service.TaskStartRejected) -> NoReturn:
    if exc.reason == "agent_id_required":
        raise V1ApiError(
            V1ErrorCode.INVALID_INPUT, 422, message="agent_id is required"
        ) from exc
    if exc.reason == "connector_runtime_setup_failed":
        raise V1ApiError(
            V1ErrorCode.INTERNAL_ERROR,
            500,
            message=_CONNECTOR_RUNTIME_SETUP_FAILED_MESSAGE,
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
) -> None:
    try:
        task_start_service.validate_sdk_owner_scope(
            _sdk_task_scope(principal),
            request_agent_id=request_agent_id,
            request_workforce_id=request_workforce_id,
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

      1. Verifies the body's ``agent_id`` matches the agent bound to
         the presented API key. Mismatch -> 404 ``agent_not_found``
         (404 not 403, so the existence of unrelated agents isn't
         leaked via error code).
      2. Persists a new :class:`Task` row owned by the agent's user,
         with ``source='sdk'``, ``is_visible=False``, and ``input`` set
         to the user message. Also persists the first user message to
         ``task_chat_messages`` so the existing background execution
         path can consume it without special-casing this entry point.
      3. Schedules background execution via
         ``start_task_in_background`` (which uses the same coroutine
         the WebSocket handler does). Returns 202 immediately --
         callers poll ``GET /v1/chat/tasks/{task_id}`` to observe the
         eventual ``completed`` / ``failed`` status.

    Args:
        request: Validated :class:`CreateTaskRequest`. ``message.content``
            is guaranteed non-empty by Pydantic; ``agent_id`` is the
            target agent the SDK caller wants to invoke.
        authed: ``(Agent, AgentApiKey)`` tuple resolved by the auth
            dependency. The agent here is the *key-bound* agent, the
            single source of truth for what this caller may touch.
    Returns:
        :class:`CreateTaskResponse` with the new ``task_id``,
        ``agent_id``, ``status='running'`` (the atomic claim inside
        the handler flips the row from PENDING to RUNNING before the
        response is sent), and ``created_at`` for the caller to
        start polling from.

    Raises:
        V1ApiError 401: missing/invalid/revoked key (raised inside
            ``get_agent_from_api_key``; envelope is uniform with
            other auth failures).
        V1ApiError 404: ``request.agent_id != authed_agent.id``.
        500 (V1 envelope): any unexpected exception -- the global
            handler in ``web/app.py`` translates to
            ``{"error": {"code": "internal_error", ...}}`` and the raw
            exception message stays out of the response.
    """
    agent_identity = principal.agent
    if agent_identity is None or request.agent_id != agent_identity.id:
        raise V1ApiError(V1ErrorCode.AGENT_NOT_FOUND, 404)

    try:
        started = await task_start_service.create_sdk_task(
            agent_id=int(agent_identity.id),
            task_owner_user_id=int(agent_identity.user_id),
            actor_user_id=int(agent_identity.user_id),
            message=request.message.content,
            file_ids=tuple(request.message.files or ()),
            connector_runtime_context=tuple(
                item.model_dump() for item in request.connector_runtime_context or ()
            ),
            timezone=request.timezone,
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

    await record_key_usage(str(principal.key.key_prefix))

    return CreateTaskResponse(
        task_id=started.task_id,
        agent_id=started.agent_id,
        status=started.status.value,
        created_at=cast(datetime, started.accepted_at),
        run_id=started.run_id,
        state_version=started.state_version,
        control_state=started.control_state,
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


@dataclass(frozen=True)
class _TaskStepsVersionSnapshot:
    """Authorized task identity and the trace version used for cache lookup."""

    task_id: int
    agent_id: int
    max_event_id: int


@dataclass(frozen=True)
class _TraceEventSnapshot:
    """Trace fields consumed by the pure public-step mapper.

    ``id`` is the row's numeric primary key (``TraceEvent.id``) --
    distinct from ``event_id`` below (the row's own string identifier,
    exposed to SDK clients as part of a public step's derived id).
    Kept here specifically so ``v1/_events_stream.py``'s warm-up replay
    can bound itself to a watermark (``id <= some max_event_id``,
    the same column ``_load_task_steps_version_snapshot`` reads) without
    a second, id-only read of the same rows.
    """

    id: int
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
    events: tuple[_TraceEventSnapshot, ...]


def _resolve_task_or_404(task_id: int, principal: ApiKeyPrincipal, db: Session) -> Task:
    try:
        return task_start_service.resolve_sdk_task(
            task_id, _sdk_task_scope(principal), db
        )
    except task_start_service.TaskStartRejected as exc:
        _raise_start_rejection(exc)


def _load_task_info_snapshot(
    task_id: int,
    principal: ApiKeyPrincipal,
) -> _TaskInfoSnapshot:
    """Authorize and detach one task inside a worker-owned short Session."""

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = _resolve_task_or_404(task_id, principal, db)
        pending_question: str | None = None
        pending_interactions: list[dict[str, Any]] | None = None
        if task.status == TaskStatus.WAITING_FOR_USER:
            pending_question, pending_interactions = get_pending_interaction_question(
                db, task
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
        )


def _load_task_steps_version_snapshot(
    task_id: int,
    principal: ApiKeyPrincipal,
) -> _TaskStepsVersionSnapshot:
    """Authorize a task and detach the trace version for cache lookup."""

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = _resolve_task_or_404(task_id, principal, db)
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
        )


def _load_task_steps_snapshot(
    task_id: int,
    principal: ApiKeyPrincipal,
) -> _TaskStepsSnapshot:
    """Reauthorize and detach the ordered trace rows after a cache miss."""

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = _resolve_task_or_404(task_id, principal, db)
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
                id=int(row.id),
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
    cache_key = task_snapshot_key(task_id)
    task_updated_at = cache_version_token(task.updated_at)
    cached = cache_get(cache_key)
    if isinstance(cached, dict) and cached.get("updated_at") == task_updated_at:
        response = TaskInfoResponse.model_validate(cached["response"])
        return response.model_copy(update={"pending_interaction": pending_interaction})

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
    return response.model_copy(update={"pending_interaction": pending_interaction})


def _get_chat_task_steps_sync(
    task_id: int,
    principal: ApiKeyPrincipal,
) -> StepsResponse:
    """Build public steps using two short authorization/read transactions."""

    version = _load_task_steps_version_snapshot(task_id, principal)
    cache_key = task_steps_key(task_id)
    cached = cache_get(cache_key)
    if isinstance(cached, dict) and cached.get("max_event_id") == version.max_event_id:
        return StepsResponse.model_validate(cached["response"])

    snapshot = _load_task_steps_snapshot(task_id, principal)
    public_steps_data = map_trace_events_to_public_steps(list(snapshot.events))
    response = StepsResponse(
        task_id=snapshot.task_id,
        agent_id=snapshot.agent_id,
        steps=[PublicStep(**step) for step in public_steps_data],
    )
    cache_set(
        cache_key,
        {
            "max_event_id": snapshot.max_event_id,
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
    new ``conversation_id``. Works for both key owner types: an
    agent-bound key appends to its agent's SDK tasks, a workforce-bound
    key appends to the manager tasks behind its workforce's SDK runs
    (the workforce turn guard revalidates archive/config drift on every
    append). This endpoint:

      1. Validates the path ``task_id`` exists and belongs to the
         key-bound owner (404 ``task_not_found`` otherwise).
      2. Validates ``body.agent_id`` matches the key-bound agent
         (404 ``agent_not_found`` otherwise; required for agent keys,
         rejected for workforce keys). ``body.workforce_id``, when
         provided with a workforce key, must match the bound workforce
         (404 ``workforce_not_found`` otherwise).
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
            owner.
        V1ApiError 409: ``task_busy`` (retryable), a workforce rejection
            code (not retryable), or ``interaction_response_required``
            (use ``reply`` instead).
        500: any other unexpected error (V1 envelope via global handler).
    """
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

    await record_key_usage(str(principal.key.key_prefix))

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

    Raises:
        V1ApiError 401: missing / invalid / revoked key.
        V1ApiError 404: task missing or not owned by the calling key.
    """
    return await run_db_io_cancellation_safe(
        lambda: _get_chat_task_steps_sync(task_id, principal)
    )


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
        ``"failed"``. Every attach opens with a batch of these frames
        describing the steps the task already has, in ``started_at``
        order: a replay of the task's history on the normal path, and a
        one-shot snapshot on the two attach-time fast paths below (the
        task is already finished, or already waiting on user input).
        Both batches are drawn from the task's most recent steps and are
        bounded by two caps applied in series, not whichever binds first:
        a step-count cap (512) keeps the most recent steps, then a
        total-wire-bytes budget (4 MiB) over that window's serialized
        frames trims it further. Either cap can be the one that actually
        removes something, and both can fire on the same batch. When the
        byte budget is what trims, it keeps that window's oldest
        contiguous run, so the very latest steps may be the ones missing;
        a batch can also come out empty. Whether the batch was cut short
        is reported on the conclusion frame on the fast paths, and is not
        reported at all on the normal path, which has no conclusion frame
        at attach time -- these ``step.*`` frames carry no **batch-level**
        truncation field of their own on either path (a step's own
        ``data`` can still carry its own ``truncated: true`` when that
        step's content overflows the per-frame cap -- see below). ``GET
        .../steps`` is the authoritative full history either way, and is
        where a client that needs every step reconciles.
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
      this stream and ``GET .../steps`` for a step this connection
      projects *live*: the live frame mints a fresh id per broadcast,
      while ``steps()`` returns the persisted trace event's own id as
      that step's id. This does not hold for the attach-time replay
      batch above: a replayed step is folded from the same persisted
      ``TraceEvent`` rows ``steps()`` reads, through the same
      ``PublicStepProjector`` fold, so its id is that same row's own id
      and matches ``steps()`` exactly.
      A ``thinking`` step whose id begins with ``thinking:plan:`` or
      ``thinking:planning:`` is correlated the same way: the replay
      batch's ids match ``steps()`` exactly, because the projector that
      builds that batch is warmed by replaying the task's full
      persisted history rather than starting empty, so the planning-
      cycle count it lands on is the same one ``steps()`` computes from
      the same rows.
      A client library may carry a stricter rule than this contract. The
      Python SDK's ``events()`` guidance tells callers to reconcile a
      planning step by ``started_at`` plus content, "never by id", with
      no exception for a replayed step -- wording that predates the
      attach-time replay batch. A caller that follows it is safe rather
      than wrong: it forgoes a correlation this endpoint does offer for
      replayed steps instead of making one this endpoint does not
      support. Narrowing that guidance to the live-projected case is
      SDK-side work, not a difference in what this endpoint sends.
      The gap that remains is narrower than the replay batch, and is
      real in two cases. First, a ``thinking:plan:``/``thinking:planning:``
      id minted for a live frame *after* the replay batch: its count
      continues from whatever this connection's own projector had
      observed by then, and a client must not assume that always equals
      ``steps()``'s own count. Second, the narrow race window between
      this connection's registration with the broadcast manager and the
      watermark read that bounds the replay (see ``_build_warm_up_frames``):
      a row landing inside that window is folded in twice -- once by the
      replay, and once more as a live broadcast that carries no
      persisted id to dedupe it against -- which can push a live
      planning-cycle count out of step with ``steps()`` from that point
      on. A planning step opened inside that window can also be left at
      ``status: "running"`` for the rest of the connection: the row is
      folded twice, so it produces two public step ids, and the row's
      single end event closes only the later one. Tracked as #1776.
      This is a defect, unlike the final-answer duplication described
      below, which is deliberate.
      Trying to actively correlate a step id across either of these
      two live cases corrupts client state rather than merely missing a
      match. These live ids are stable within one connection and nowhere
      else: a re-attach after ``stream_expired`` (the 1-hour cap) or
      ``resync_required`` (queue overflow) opens a new connection whose
      count starts over. Reconcile a live planning step against
      ``steps()`` by ``started_at`` and content, not by id.
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
      step ``steps()`` and a fresh attach's replay already show for
      that row. This is duplication, not loss: a client that already
      rendered the delta/completed sequence can ignore the matching
      step, or use it as the authoritative record instead, but should
      not expect the two to be mutually exclusive. Unlike the
      race-window duplication described above (tracked as #1776),
      this one is by design on every attach, not a narrow-window
      defect.
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
      task's own description column, which such frames carry under one
      of two keys depending on the frame family -- is dropped whole
      instead of truncated (see Behavior below): no marker, no
      content.
      This cap is specific to this stream -- ``GET .../steps`` applies
      no such cap and always returns a step's full, untruncated
      ``data``, so the two surfaces can disagree on content for the
      same step once it's large enough to trip this stream's cap.

    Behavior:
      - Normal attach: opens the stream, emits ``task.status`` for the
        task's current state, then replays the task's already-known
        steps as ``step.started``/``step.completed`` frames in
        ``started_at`` order -- the same order ``steps()`` returns them
        in -- then goes live: a status update each time the task's
        status changes (consecutive duplicates are suppressed), further
        ``step.*``/``message.*`` frames as new trace events and streamed
        answer chunks arrive, and ``task.completed`` when the task
        finishes. A ``: ping`` comment line is sent whenever 15s pass
        without any other frame going out, to keep the connection
        alive -- not on a fixed 15s cadence regardless of activity.
      - The replay is what lets a client attach to a task that is
        already running and still see that task's in-flight steps
        resolve: a step whose ``step.started`` was broadcast before the
        connection existed is replayed here, so its end event arriving
        later has a start to pair with and reaches the client as
        ``step.completed`` rather than being dropped as an orphan --
        unless the replay's byte budget trimmed that step out of the
        batch, or the batch ended early at a step this stream could not
        serialize -- reasons (5) and (6) below.
      - The replay is bounded, and a client must not treat it as the
        task's complete history. It carries the same two caps, applied
        in the same order, that ``step.started``/``step.completed``
        above states for every attach-time batch, including which end
        of the window each one drops. A task with
        more history than that gets a partial replay, with no marker
        frame saying so, and a replay can legitimately be empty. This is
        a bound, not a failure:
        nothing about it closes the stream or emits ``stream.error``.
        A step whose data this stream
        cannot serialize ends the batch at that step, which shortens the
        replay the same way without closing the stream either; that one
        is reason (6) below.
        ``GET .../steps`` reads the database, is unaffected by when the
        stream was opened or by what the replay admitted, and is the
        authoritative full history -- a client that needs every step
        reconciles there. That is true of a replay shortened by either
        cap. It is not unconditionally true of a replay shortened by a
        step it could not serialize: reason (6) below says why.
      - Frame sequence into ``task.completed``: a task that fails emits
        ``task.status`` (``"failed"``) from the failure broadcast first,
        then the watchdog's authoritative ``task.completed`` close
        frame. Delivery of every ``step.*``/``message.*`` content frame
        on this stream is best-effort, not guaranteed, for six
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
        excludes the task's own description column -- which arrives
        under one of two keys depending on the frame family -- so a
        task's description alone can no longer blank out its step
        stream this way -- a frame is only dropped here when its
        content, apart from that column, is still over the cap. (4) a
        planning step can be left unresolved:
        when a planning round ends without emitting its own terminal
        event (a plan-generation error escaping before it), the next
        round clears the pairing key that step was waiting on, so the
        step's ``step.completed`` is never produced and the client
        keeps a ``thinking`` step at ``"running"`` for the rest of the
        stream's life (``_step_mapping.py``'s ``dag_execute_start``
        branch). This one is not a stream-only artifact: ``GET
        .../steps`` folds the same events and shows that step the same
        way. (5) the replay's byte budget can trim a still-running step
        out of the batch it sends at attach time: the projector that
        feeds the live view is still warmed from the task's whole
        history regardless, so that step's ``step.completed`` still
        arrives once it resolves, but this stream never sent the
        ``step.started`` the replay would otherwise have given it, so
        the end arrives with no start on this stream to pair with. This
        one is a stream-only artifact, unlike (4): ``GET .../steps``
        reads the task's full history directly and is unaffected by
        what one stream attach's replay admitted. (6) a step whose data
        cannot be turned into JSON ends the replay batch at that step:
        the steps serialized before it still go out and the stream
        still goes live, but that step and anything the batch had not
        reached are missing from the replay, leaving the same
        start-without-end shape reason (5) describes. Reconciling this
        one against ``GET .../steps`` is not guaranteed to work the way
        it does for (5): that endpoint serializes the same step data on
        its own read path, so data that defeats the replay can make
        that read fail as well. That is a property of the data, not of
        this stream, and it is tracked separately. (2), (3), (4), (5)
        and (6) are silent: none of them ever produces a
        ``stream.error`` frame or any other signal on the wire. (1) is
        not silent in the same way -- the
        close frame that follows a queue-drain is often itself a
        ``stream.error`` (``resync_required``/``unauthorized``/
        ``task_deleted``/``stream_expired``) -- but that close frame
        carries no record of what it drained on its way out, so its
        presence or absence proves nothing about whether a content frame
        was lost this way: a ``stream.error`` does not mean content was
        dropped, and its absence does not mean nothing was. Only
        reconciling against ``steps()`` answers that. A related but
        separate mechanism is the warm-up staging buffer (used only
        during a fresh attach's replay window, before this sink goes
        live): if it overflows, the still-unfed staged messages are
        dropped and the stream closes with ``resync_required`` for that
        specific reason -- a bound on that buffer, not the outbound
        queue (1) describes. No failure
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
        row is gone by the time the steps are read, ``resync_required``
        otherwise: a step-read failure on either fast path never
        costs the client the lifecycle conclusion it already had in
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
        followed by ``stream.error`` naming why -- ``task_deleted`` or
        ``resync_required`` by the same rule. A failure while
        serializing an individual step is handled the same way: the
        conclusion frame goes out, followed by
        ``stream.error(resync_required)``, and the step list sent
        before the failure may be partial.
      - The stream force-closes with ``task.input_required`` if the task
        is found waiting on user input (same steps-then-close shape:
        ``task.status``, the task's steps, then ``task.input_required``);
        with ``stream.error`` if the API key is revoked/paused, the task
        row disappears (within one watchdog cycle, 30s in production, of
        the delete), the client can't keep up (``resync_required``), or
        the stream has been open for the 1-hour maximum
        (``stream_expired``, emitted before the connection closes so
        it's distinguishable from a clean end). After any
        ``stream.error``, re-attaching is the supported recovery path
        (no replay of missed frames -- reconcile via ``steps()`` first).
      - A task that's ``paused`` does not close the stream (matching
        SDK ``wait()`` semantics: another process may resume it); the
        1-hour cap is what eventually ends an orphaned paused stream.
      - No ``task.status`` frame is guaranteed to be fresh or in order,
        at any point in the stream's life, not only at attach: this
        endpoint never buffers or reconciles ``task.status`` frame
        order against anything (``step.*``/``message.*`` frames are a
        separate story -- attaching mid-task replays known steps before
        going live specifically so a step's end event isn't misread as
        an orphan, unless the replay's byte budget trimmed that step
        out of the batch, or the batch ended early at a step this
        stream could not serialize -- reasons (5) and (6) below; that
        ordering
        guarantee doesn't extend to ``task.status``). Accepted because
        only the three close frames
        above are treated as authoritative; each comes from a direct
        read of the task row, never from frame ordering.

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
            snapshot is bounded by the same step-count cap and
            total-wire-bytes budget the normal streaming path's own
            history replay is; a snapshot cut short by either carries
            the ``snapshot_truncated``/``snapshot_total_steps`` marker
            on its conclusion frame as described above. When a
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
            executing the task still gets a correct history replay (it
            reads the database) and the authoritative close frame (the
            watchdog reads the task row), but receives none of the live
            ``step.*``/``message.*`` frames produced in between -- there
            is no equivalent fallback delivery for content the way there
            is for a status transition, so such an attach's live portion
            degrades to the lifecycle-only shape. Some transitions --
            lease-expiry recovery among them -- never broadcast at all,
            in any deployment shape, so the watchdog poll is their only
            delivery path regardless of worker count.
    """
    snapshot = await run_db_io_cancellation_safe(
        lambda: _load_task_info_snapshot(task_id, principal)
    )
    return await _events_stream.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=_load_task_info_snapshot,
        read_task_steps=_load_task_steps_snapshot,
        read_task_steps_response=_get_chat_task_steps_sync,
        read_task_steps_version=_load_task_steps_version_snapshot,
    )
