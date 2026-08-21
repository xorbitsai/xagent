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

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, NoReturn, Optional, cast

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from ....core.tools.adapters.vibe.connector_runtime import ConnectorRuntimeError
from ...config import is_allowed_file
from ...models.agent import Agent
from ...models.database import get_session_local
from ...models.task import Task, TaskStatus, TraceEvent
from ...models.workforce import WorkforceRun
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
from ...services.connector_runtime import (
    bind_create_connector_runtime_plan,
    persist_create_connector_runtime_context,
    pop_ephemeral_runtime_values,
    prepare_append_connector_runtime,
    prepare_create_connector_runtime,
    store_ephemeral_runtime_values,
)
from ...services.db_runtime import (
    drain_async_task_cancellation_safe,
    run_db_io_cancellation_safe,
)
from ...services.file_turn import (
    append_uploaded_files_context,
    build_uploaded_files_context,
    normalize_attachments_for_persistence,
    resolve_turn_file_infos,
)
from ...services.hot_path_cache import (
    cache_get,
    cache_set,
    cache_version_token,
    task_cache_ttl_seconds,
    task_snapshot_key,
    task_steps_key,
)
from ...services.managed_file_ref import DurableStorageOperationError
from ...services.task_interaction_read import get_pending_interaction_question
from ...services.task_orchestrator import (
    TaskTurnError,
    TaskTurnFileBindingError,
    TaskTurnNotFoundError,
    TaskTurnOrchestrator,
    TaskTurnPayload,
    TurnKind,
    TurnStarted,
    _ClaimedTurn,
    _retire_turn_session_best_effort,
    commit_claimed_turn_or_reconcile,
)
from . import _events_stream
from ._step_mapping import map_trace_events_to_public_steps
from .deps import ApiKeyPrincipal, get_principal_from_api_key, record_key_usage
from .errors import V1ApiError, V1ErrorCode, raise_for_turn_rejection

router = APIRouter()
logger = logging.getLogger(__name__)

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
            raise V1ApiError(
                V1ErrorCode.INTERNAL_ERROR,
                503,
                message="File storage is temporarily unavailable.",
            ) from exc
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


def _resolve_turn_files_or_400(
    *,
    file_ids: list[str],
    owner_user_id: int,
    db: Session,
    task_id: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Resolve every ``file_id`` up front (all-or-nothing) WITHOUT binding.

    Called before the task is committed (create) or the turn is claimed
    (append), so a bad/unowned/already-bound id fails with 400 before any
    task row is created or mutated -- no orphan task, no binding stuck to a
    turn that later 409s. Actual binding happens via :func:`bind_turn_files`
    only after the turn is committed to running.
    """
    if not file_ids:
        return []
    try:
        file_infos, missing = resolve_turn_file_infos(
            file_ids=file_ids,
            owner_user_id=owner_user_id,
            db=db,
            task_id=task_id,
        )
    except DurableStorageOperationError as exc:
        # Transient storage fault, not a client error -- 503 so SDK can retry.
        raise V1ApiError(
            V1ErrorCode.INTERNAL_ERROR,
            503,
            message="File storage is temporarily unavailable.",
        ) from exc
    if missing:
        raise V1ApiError(
            V1ErrorCode.INVALID_INPUT,
            400,
            message="These file ids are not accessible: " + ", ".join(missing),
        )
    return file_infos


def _turn_payload(content: str, file_infos: list[dict[str, Any]]) -> TaskTurnPayload:
    """Build a :class:`TaskTurnPayload`, file-enriching the execution channel.

    Consolidates the payload construction shared by create and append so the
    transcript-vs-execution split can't drift between the two entry points.
    """
    if not file_infos:
        return TaskTurnPayload(transcript_message=content)
    context = build_uploaded_files_context(file_infos)
    return TaskTurnPayload(
        transcript_message=content,
        execution_message=append_uploaded_files_context(content, context),
        attachments=normalize_attachments_for_persistence(file_infos) or None,
        file_ids=tuple(str(info["file_id"]) for info in file_infos),
    )


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


def _store_connector_runtime_values_or_fail(
    *,
    task_id: int,
    turn_id: str,
    values_by_ref: dict,
) -> None:
    try:
        store_ephemeral_runtime_values(turn_id, values_by_ref)
    except Exception as exc:
        pop_ephemeral_runtime_values(turn_id)
        logger.warning(
            "Connector runtime setup failed for task %s turn %s",
            task_id,
            turn_id,
        )
        raise V1ApiError(
            V1ErrorCode.INTERNAL_ERROR,
            500,
            message=_CONNECTOR_RUNTIME_SETUP_FAILED_MESSAGE,
        ) from exc


@dataclass(frozen=True)
class _PreparedCreateTaskStart:
    """Detached result of the atomic create-and-claim transaction."""

    task_id: int
    agent_id: int
    task_owner_user_id: int
    created_at: datetime
    payload: TaskTurnPayload
    claimed_turn: _ClaimedTurn


@dataclass(frozen=True)
class _PreparedAppendTurn:
    """Detached result of the atomic append claim transaction."""

    task_id: int
    agent_id: int
    task_owner_user_id: int
    payload: TaskTurnPayload
    claimed_turn: _ClaimedTurn


def _prepare_created_task_isolated(
    *,
    agent_id: int,
    task_owner_user_id: int,
    message: str,
    file_ids: tuple[str, ...],
    connector_runtime_context: tuple[dict[str, Any], ...],
) -> _PreparedCreateTaskStart:
    """Create, claim, and commit the first turn in one DB transaction."""

    SessionLocal = get_session_local()
    db = SessionLocal()
    payload: TaskTurnPayload | None = None
    owned_task_id = 0
    try:
        agent = db.get(Agent, agent_id)
        if agent is None:
            raise V1ApiError(V1ErrorCode.AGENT_NOT_FOUND, 404)

        file_infos = _resolve_turn_files_or_400(
            file_ids=list(file_ids),
            owner_user_id=task_owner_user_id,
            db=db,
            task_id=None,
        )
        task = Task(
            user_id=task_owner_user_id,
            title=message[:50] or "SDK task",
            description=message,
            status=TaskStatus.PENDING,
            agent_id=agent_id,
            input=message,
            source="sdk",
            is_visible=False,
        )
        try:
            runtime_plan = prepare_create_connector_runtime(
                db=db,
                agent=agent,
                task_source="sdk",
                connector_user_id=task_owner_user_id,
                payload_items=connector_runtime_context,
            )
            bind_create_connector_runtime_plan(task=task, plan=runtime_plan)
        except ConnectorRuntimeError as exc:
            _raise_v1_connector_runtime_error(exc)

        db.add(task)
        db.flush()
        task_id = int(task.id)
        owned_task_id = task_id
        persist_create_connector_runtime_context(
            db=db,
            task_id=task_id,
            plan=runtime_plan,
        )
        payload = _turn_payload(message, file_infos)
        try:
            claimed_turn = TaskTurnOrchestrator.claim_created_turn_no_commit(
                db,
                task_id=task_id,
                task_owner_user_id=task_owner_user_id,
                payload=payload,
            )
        except TaskTurnFileBindingError as exc:
            _raise_v1_file_binding_error(exc)
        _store_connector_runtime_values_or_fail(
            task_id=task_id,
            turn_id=payload.turn_id,
            values_by_ref=runtime_plan.ephemeral_by_ref,
        )
        created_at = db.query(Task.created_at).filter(Task.id == task_id).scalar()
        if created_at is None:
            raise RuntimeError("created task has no creation timestamp")
        commit_claimed_turn_or_reconcile(
            db,
            task_id=task_id,
            task_owner_user_id=task_owner_user_id,
            payload=payload,
            claimed=claimed_turn,
        )
        return _PreparedCreateTaskStart(
            task_id=task_id,
            agent_id=agent_id,
            task_owner_user_id=task_owner_user_id,
            created_at=created_at,
            payload=payload,
            claimed_turn=claimed_turn,
        )
    except Exception:
        db.rollback()
        if payload is not None:
            pop_ephemeral_runtime_values(payload.turn_id)
        raise
    finally:
        _retire_turn_session_best_effort(db, task_id=owned_task_id)


def _validate_owner_scope(
    principal: ApiKeyPrincipal,
    *,
    request_agent_id: int | None,
    request_workforce_id: int | None,
) -> None:
    """Validate a request body's owner-scoping fields against the key.

    Shared by every ``/v1/chat/tasks/{task_id}/*`` write that reuses the
    ``AppendMessageRequest``-shaped owner fields (append, reply): an
    agent-bound key must pass a matching ``agent_id`` and no
    ``workforce_id``; a workforce-bound key must not pass ``agent_id``
    and, if it passes ``workforce_id``, it must match the bound
    workforce. Task ownership itself must already be resolved (e.g. via
    :func:`_resolve_task_or_404`) before calling this, so an unrelated
    task stays an opaque ``task_not_found`` even when the body also
    names the wrong owner.
    """
    if principal.agent is not None:
        if request_agent_id is None:
            raise V1ApiError(
                V1ErrorCode.INVALID_INPUT,
                422,
                message="agent_id is required",
            )
        if request_agent_id != principal.agent.id:
            raise V1ApiError(V1ErrorCode.AGENT_NOT_FOUND, 404)
        if request_workforce_id is not None:
            raise V1ApiError(V1ErrorCode.WORKFORCE_NOT_FOUND, 404)
    else:
        assert principal.workforce is not None
        if request_agent_id is not None:
            raise V1ApiError(V1ErrorCode.AGENT_NOT_FOUND, 404)
        if (
            request_workforce_id is not None
            and request_workforce_id != principal.workforce.id
        ):
            raise V1ApiError(V1ErrorCode.WORKFORCE_NOT_FOUND, 404)


def _prepare_append_turn_isolated(
    *,
    task_id: int,
    principal: ApiKeyPrincipal,
    request_agent_id: int | None,
    request_workforce_id: int | None,
    message: str,
    file_ids: tuple[str, ...],
    connector_runtime_context: tuple[dict[str, Any], ...],
) -> _PreparedAppendTurn:
    """Resolve append authorization and runtime inputs into detached values."""

    SessionLocal = get_session_local()
    db = SessionLocal()
    payload: TaskTurnPayload | None = None
    try:
        task = _resolve_task_or_404(task_id, principal, db)
        task_owner_user_id = int(task.user_id)

        # Task ownership is resolved first so an unrelated task remains an
        # opaque task_not_found even when the body also names the wrong owner.
        _validate_owner_scope(
            principal,
            request_agent_id=request_agent_id,
            request_workforce_id=request_workforce_id,
        )

        runtime_agent = db.get(Agent, int(task.agent_id))
        if runtime_agent is None:
            raise V1ApiError(V1ErrorCode.INTERNAL_ERROR, 500)
        try:
            runtime_plan = prepare_append_connector_runtime(
                db=db,
                agent=runtime_agent,
                task=task,
                connector_user_id=task_owner_user_id,
                payload_items=connector_runtime_context,
            )
        except ConnectorRuntimeError as exc:
            _raise_v1_connector_runtime_error(exc)

        file_infos = _resolve_turn_files_or_400(
            file_ids=list(file_ids),
            owner_user_id=task_owner_user_id,
            db=db,
            task_id=int(task.id),
        )
        payload = _turn_payload(message, file_infos)
        try:
            claimed_turn = TaskTurnOrchestrator.claim_append_turn_no_commit(
                db,
                task_id=int(task.id),
                task_owner_user_id=task_owner_user_id,
                payload=payload,
            )
        except TaskTurnFileBindingError as exc:
            _raise_v1_file_binding_error(exc)
        _store_connector_runtime_values_or_fail(
            task_id=int(task.id),
            turn_id=payload.turn_id,
            values_by_ref=runtime_plan.ephemeral_by_ref,
        )
        prepared = _PreparedAppendTurn(
            task_id=int(task.id),
            agent_id=int(task.agent_id),
            task_owner_user_id=task_owner_user_id,
            payload=payload,
            claimed_turn=claimed_turn,
        )
        commit_claimed_turn_or_reconcile(
            db,
            task_id=int(task.id),
            task_owner_user_id=task_owner_user_id,
            payload=payload,
            claimed=claimed_turn,
        )
        return prepared
    except Exception:
        db.rollback()
        if payload is not None:
            pop_ephemeral_runtime_values(payload.turn_id)
        raise
    finally:
        _retire_turn_session_best_effort(db, task_id=task_id)


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

    task_owner_user_id = int(agent_identity.user_id)
    actor_user_id = int(agent_identity.user_id)
    runtime_context = tuple(
        item.model_dump() for item in request.connector_runtime_context or ()
    )

    async def _prepare_and_schedule() -> tuple[_PreparedCreateTaskStart, TurnStarted]:
        prepared = await run_db_io_cancellation_safe(
            lambda: _prepare_created_task_isolated(
                agent_id=int(agent_identity.id),
                task_owner_user_id=task_owner_user_id,
                message=request.message.content,
                file_ids=tuple(request.message.files or ()),
                connector_runtime_context=runtime_context,
            )
        )
        try:
            started = await TaskTurnOrchestrator.schedule_claimed_create_turn(
                task_id=prepared.task_id,
                task_owner_user_id=prepared.task_owner_user_id,
                actor_user_id=actor_user_id,
                payload=prepared.payload,
                claimed=prepared.claimed_turn,
            )
        except TaskTurnError as exc:
            pop_ephemeral_runtime_values(prepared.payload.turn_id)
            raise_for_turn_rejection(exc.reason)
        except BaseException:
            pop_ephemeral_runtime_values(prepared.payload.turn_id)
            raise
        return prepared, started

    # The owned child starts before the first await. If the caller is cancelled
    # after RUNNING commits, scheduling (or its exact-lease compensation) still
    # settles before cancellation propagates.
    start_task = asyncio.create_task(_prepare_and_schedule())
    prepared, started = await drain_async_task_cancellation_safe(start_task)

    await record_key_usage(str(principal.key.key_prefix))

    return CreateTaskResponse(
        task_id=prepared.task_id,
        agent_id=prepared.agent_id,
        status=started.status.value,
        created_at=prepared.created_at,
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
    events: tuple[_TraceEventSnapshot, ...]


def _resolve_task_or_404(task_id: int, principal: ApiKeyPrincipal, db: Session) -> Task:
    """Resolve a task_id against the calling key's ownership AND
    SDK-source scope.

    Returns the :class:`Task` row when the task:

      1. Exists.
      2. Belongs to the key's owner -- directly (``Task.agent_id`` for
         an agent-bound key) or through a workforce run
         (``WorkforceRun.task_id`` for a workforce-bound key; the
         binding is 1:1 unique).
      3. Was created by the SDK (``source == "sdk"``).

    Any other case — missing row, row belongs to a different owner,
    or row was created by the Web UI / internal paths — raises
    :class:`V1ApiError` with ``task_not_found`` (404 not 403, so the
    existence of tasks under other owners / other surfaces isn't
    observable through error code).

    The ``source == "sdk"`` filter exists because an SDK API key
    binds to an owner, not to a particular product surface. Without
    it, an SDK client could read or append to any task the Web UI
    created under the same owner (the user's own historical Web UI
    chats or workforce runs, for example). Whether that's intentional
    is a product decision, but the safe default for a public SDK is to
    scope lookups to tasks the SDK itself created — ``POST
    /v1/chat/tasks`` and ``POST /v1/workforces/{id}/runs`` both write
    ``source="sdk"`` so this is well-defined.

    Args:
        task_id: Path parameter from the route.
        principal: The key-bound owner resolved by
            ``get_principal_from_api_key``.
        db: SQLAlchemy session.

    Raises:
        V1ApiError(TASK_NOT_FOUND, 404): task missing, not owned by
            the calling key, or not created by the SDK.
    """
    query = db.query(Task).filter(
        Task.id == task_id,
        Task.source == "sdk",
    )
    if principal.agent is not None:
        query = query.filter(Task.agent_id == principal.agent.id)
    else:
        assert principal.workforce is not None
        query = query.join(WorkforceRun, WorkforceRun.task_id == Task.id).filter(
            WorkforceRun.workforce_id == principal.workforce.id
        )
    task = query.first()
    if task is None:
        raise V1ApiError(V1ErrorCode.TASK_NOT_FOUND, 404)
    return task


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
    actor_user_id = principal.owner_user_id
    runtime_context = tuple(
        item.model_dump() for item in request.connector_runtime_context or ()
    )

    async def _prepare_and_begin() -> tuple[_PreparedAppendTurn, TurnStarted]:
        # A runner may still be finishing local cleanup after its terminal
        # status is visible in the database. Reject that tail window before
        # the domain-owned transaction stages any append mutation. The atomic
        # status predicate inside the transaction remains authoritative across
        # workers and concurrent requests.
        TaskTurnOrchestrator.ensure_no_background_turn(task_id)
        prepared = await run_db_io_cancellation_safe(
            lambda: _prepare_append_turn_isolated(
                task_id=task_id,
                principal=principal,
                request_agent_id=request.agent_id,
                request_workforce_id=request.workforce_id,
                message=request.message.content,
                file_ids=tuple(request.message.files or ()),
                connector_runtime_context=runtime_context,
            )
        )
        try:
            started = await TaskTurnOrchestrator.schedule_claimed_turn(
                task_id=prepared.task_id,
                task_owner_user_id=prepared.task_owner_user_id,
                actor_user_id=actor_user_id,
                payload=prepared.payload,
                claimed=prepared.claimed_turn,
                kind=TurnKind.APPEND,
            )
        except BaseException:
            pop_ephemeral_runtime_values(prepared.payload.turn_id)
            raise
        return prepared, started

    start_task = asyncio.create_task(_prepare_and_begin())
    try:
        prepared, started = await drain_async_task_cancellation_safe(start_task)
    except TaskTurnNotFoundError:
        raise V1ApiError(V1ErrorCode.TASK_NOT_FOUND, 404)
    except TaskTurnError as exc:
        raise_for_turn_rejection(exc.reason)

    await record_key_usage(str(principal.key.key_prefix))

    return AppendMessageResponse(
        task_id=prepared.task_id,
        agent_id=prepared.agent_id,
        workforce_id=(
            int(principal.workforce.id) if principal.workforce is not None else None
        ),
        status=started.status.value,
        accepted_at=started.updated_at,
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

    The other four project the task's step-by-step execution and
    streamed message text onto the same connection, from the same
    ``PublicStep`` shape and public step-type list ``GET
    .../steps`` uses (``thinking``/``tool_call``/``agent_delegation``/
    ``message``):
      - ``step.started``: ``{step}``, a running ``PublicStep``.
      - ``step.completed``: ``{step}``, the same step once its end event
        (or failure) resolves it -- ``status`` is ``"completed"`` or
        ``"failed"``.
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
      - This stream carries only what happens after the connection
        opens; it never sends the steps that already happened, on any
        path. That has two consequences a client must handle. First, a
        step that was already running at attach time does not appear on
        this stream at all: its ``step.started`` was broadcast before
        this connection existed, and its end event arrives here with no
        matching start, so that end is dropped rather than sent. The
        client sees neither half of the step -- it is absent, not left
        at ``"running"``. Second, an attach that arrives once the task
        has already finished (or is already waiting on user input) has
        no live content left to carry, so it closes with the lifecycle
        frames alone. Either way, ``GET .../steps`` reads the database
        and is unaffected by when the stream was opened, so that is
        where a client reconciles what this stream did not carry.
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
        ``task_deleted``/``stream_expired``) -- but that close frame
        carries no record of what it drained on its way out, so its
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
        ``task.status``.
      - Attaching to an already-finished task is not an error: the
        stream opens, emits ``task.status`` then ``task.completed``,
        and closes immediately, with no ``step.*`` frames in between --
        that task's steps are read through ``GET .../steps``.
      - The stream force-closes with ``task.input_required`` if the task
        is found waiting on user input (same shape: ``task.status``,
        then ``task.input_required``, then close); with ``stream.error``
        if the API key is revoked/paused, the task row disappears
        (within one watchdog cycle, 30s in production, of the delete),
        the client can't keep up (``resync_required``), or the stream
        has been open for the 1-hour maximum (``stream_expired``,
        emitted before the connection closes so it's distinguishable
        from a clean end). After any ``stream.error``, re-attaching is
        the supported recovery path (no replay of missed frames --
        reconcile via ``steps()`` first).
      - A task that's ``paused`` does not close the stream (matching
        SDK ``wait()`` semantics: another process may resume it); the
        1-hour cap is what eventually ends an orphaned paused stream.
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
            cap. Broadcast delivery is per-process too: a sink only
            receives the broadcasts fanned out by its own worker
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
        lambda: _load_task_info_snapshot(task_id, principal)
    )
    return await _events_stream.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=_load_task_info_snapshot,
    )
