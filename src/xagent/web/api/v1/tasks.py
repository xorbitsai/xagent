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

import logging
from typing import Any, NoReturn, Optional, cast

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy import func
from sqlalchemy.orm import Session

from ....core.tools.adapters.vibe.connector_runtime import ConnectorRuntimeError
from ...config import is_allowed_file
from ...models.agent import Agent
from ...models.database import get_db
from ...models.task import Task, TaskStatus, TraceEvent
from ...models.user import User
from ...models.workforce import WorkforceRun
from ...schemas.v1 import (
    AppendMessageRequest,
    AppendMessageResponse,
    CreateTaskRequest,
    CreateTaskResponse,
    PublicStep,
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
from ...services.file_turn import (
    append_uploaded_files_context,
    bind_turn_files,
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
from ...services.task_execution_controller import (
    TaskControlState,
    apply_task_control_transition,
)
from ...services.task_orchestrator import (
    TaskTurnError,
    TaskTurnNotFoundError,
    TaskTurnOrchestrator,
    TaskTurnPayload,
    TurnKind,
)
from ._step_mapping import map_trace_events_to_public_steps
from .deps import ApiKeyPrincipal, get_principal_from_api_key, record_key_usage
from .errors import V1ApiError, V1ErrorCode, raise_for_turn_rejection

router = APIRouter()
logger = logging.getLogger(__name__)

_CONNECTOR_RUNTIME_SETUP_FAILED_MESSAGE = "Connector runtime setup failed."


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
    db: Session = Depends(get_db),
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
        task = _resolve_task_or_404(task_id, principal, db)
        upload_owner_user_id = int(task.user_id)

    owner = db.query(User).filter(User.id == upload_owner_user_id).first()
    if owner is None:
        raise V1ApiError(V1ErrorCode.INTERNAL_ERROR, 500)

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
            user=owner,
            db=db,
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
    )


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


def _rollback_runtime_setup_mark_failure(db: Session, task_id: int) -> None:
    try:
        db.rollback()
    except Exception:
        logger.warning(
            "Failed to roll back task %s session after connector runtime setup error",
            task_id,
            exc_info=True,
        )


def _mark_task_failed_after_runtime_setup_error(db: Session, task_id: int) -> None:
    """Best-effort terminal mark after pre-schedule runtime setup fails."""

    try:
        db.rollback()
        task = db.query(Task).filter(Task.id == task_id).first()
        if task is not None:
            orm_task = cast(Any, task)
            apply_task_control_transition(
                task,
                TaskControlState.FAILED,
                status=TaskStatus.FAILED,
            )
            orm_task.error_message = _CONNECTOR_RUNTIME_SETUP_FAILED_MESSAGE
            db.commit()
    except Exception:
        _rollback_runtime_setup_mark_failure(db, task_id)
        logger.warning(
            "Failed to mark task %s failed after connector runtime setup error",
            task_id,
            exc_info=True,
        )


def _store_connector_runtime_values_or_fail(
    *,
    db: Session,
    task_id: int,
    turn_id: str,
    values_by_ref: dict,
    mark_task_failed: bool,
) -> None:
    try:
        store_ephemeral_runtime_values(turn_id, values_by_ref)
    except Exception as exc:
        pop_ephemeral_runtime_values(turn_id)
        if mark_task_failed:
            _mark_task_failed_after_runtime_setup_error(db, task_id)
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


@router.post(
    "/chat/tasks",
    status_code=202,
    response_model=CreateTaskResponse,
)
async def create_chat_task(
    request: CreateTaskRequest,
    principal: ApiKeyPrincipal = Depends(get_principal_from_api_key),
    db: Session = Depends(get_db),
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
        db: SQLAlchemy session.

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
    # Task creation is agent-only: workforce runs are created via
    # ``POST /v1/workforces/{id}/runs`` (the manager-delegation flow
    # cannot be entered through a bare agent task). A workforce-bound
    # key therefore never matches ``request.agent_id`` -- same 404 as
    # any other unbound agent_id.
    agent = principal.agent
    _key = principal.key_row
    task_source = "sdk"

    # Server-side agent_id consistency check. The key already binds an
    # agent; ``body.agent_id`` is required by the SDK contract for
    # forward-compat (and Python/TS SDK symmetry), but the bound
    # agent is the only authority. Mismatch is a 404 -- never a 403
    # -- so the existence of agent_id=N elsewhere in the system isn't
    # observable to this caller.
    if agent is None or request.agent_id != agent.id:
        raise V1ApiError(V1ErrorCode.AGENT_NOT_FOUND, 404)

    task_owner_user_id = int(agent.user_id)
    actor_user_id = int(agent.user_id)

    # Validate any attached file ids BEFORE creating the task, so a bad id
    # fails with 400 without leaving an orphan PENDING task behind. Binding
    # happens after the turn is claimed (below).
    file_infos = _resolve_turn_files_or_400(
        file_ids=request.message.files or [],
        owner_user_id=task_owner_user_id,
        db=db,
        task_id=None,
    )

    # title is what the web UI shows in its task list. Truncate to
    # 50 chars (matches the WS handler convention) so very long
    # user inputs don't fill the sidebar with a one-line wall of
    # text. The full message is preserved in ``description`` /
    # ``input`` / ``task_chat_messages``.
    title = request.message.content[:50] or "SDK task"

    # Create the Task row with SDK-specific fields populated.
    # ``source='sdk'`` lets adoption metrics queries split SDK traffic
    # from web/widget; ``is_visible=False`` keeps SDK/REST runs out of
    # Web UI discovery surfaces while preserving exact task-id access
    # for SDK polling and audit views; ``input`` records this turn's
    # user message so GET endpoint can return it without going through
    # task_chat_messages.
    task = Task(
        user_id=task_owner_user_id,
        title=title,
        description=request.message.content,
        status=TaskStatus.PENDING,
        agent_id=agent.id,
        input=request.message.content,
        source=task_source,
        is_visible=False,
    )
    try:
        runtime_plan = prepare_create_connector_runtime(
            db=db,
            agent=agent,
            task_source=task_source,
            connector_user_id=task_owner_user_id,
            payload_items=request.connector_runtime_context,
        )
        bind_create_connector_runtime_plan(task=task, plan=runtime_plan)
    except ConnectorRuntimeError as exc:
        _raise_v1_connector_runtime_error(exc)

    db.add(task)
    db.flush()
    persist_create_connector_runtime_context(
        db=db, task_id=int(task.id), plan=runtime_plan
    )
    db.commit()
    db.refresh(task)

    # Orchestrator's begin_turn handles the full new-turn transition:
    # bg-inflight guard, atomic status flip + transcript persist in one
    # commit, and bg coroutine scheduling under a lease lifecycle.
    # A brand-new task shouldn't ever hit busy -- but we map it
    # anyway for defense.
    payload = _turn_payload(request.message.content, file_infos)
    _store_connector_runtime_values_or_fail(
        db=db,
        task_id=int(task.id),
        turn_id=payload.turn_id,
        values_by_ref=runtime_plan.ephemeral_by_ref,
        mark_task_failed=True,
    )
    try:
        started = await TaskTurnOrchestrator.begin_turn(
            task_id=int(task.id),
            task_owner_user_id=task_owner_user_id,
            # SDK key resolves to the agent owner; actor == owner here.
            actor_user_id=actor_user_id,
            payload=payload,
            kind=TurnKind.CREATE,
            force_fresh=False,
        )
    except TaskTurnNotFoundError:
        pop_ephemeral_runtime_values(payload.turn_id)
        raise V1ApiError(V1ErrorCode.TASK_NOT_FOUND, 404)
    except TaskTurnError as exc:
        pop_ephemeral_runtime_values(payload.turn_id)
        raise_for_turn_rejection(exc.reason)
    except Exception:
        pop_ephemeral_runtime_values(payload.turn_id)
        raise

    # Bind files only after the turn is committed to running. This bind can
    # race the background runner (begin_turn schedules it via create_task and
    # may await further steps before returning here), but that's harmless: the
    # runner's file query tolerates a NULL task_id (task_id == this OR IS NULL,
    # owned by the user), so the just-uploaded files are readable whether or
    # not this bind has landed. The bind is for durable task<->file association.
    bind_turn_files(
        file_ids=[info["file_id"] for info in file_infos],
        task_id=int(task.id),
        owner_user_id=task_owner_user_id,
        db=db,
    )

    # Record usage here (not in the shared auth dependency) so read-only
    # status/steps polling below doesn't count as a "call". Runtime validation
    # and turn claim have already accepted this as a real task invocation.
    record_key_usage(str(_key.key_prefix))

    # ``status`` comes from the orchestrator's committed-row snapshot
    # (``started.status`` == RUNNING), NOT the caller's ``task`` object --
    # ``begin_turn`` now commits on an isolated worker-thread session and
    # never refreshes this request's ORM row, so ``task.status`` is still
    # the stale PENDING. ``created_at`` is set at task creation and isn't
    # touched by the turn, so the in-memory value is correct.
    return CreateTaskResponse(
        task_id=int(task.id),
        agent_id=int(agent.id),
        status=started.status.value,
        created_at=task.created_at,
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


@router.post(
    "/chat/tasks/{task_id}/messages",
    status_code=202,
    response_model=AppendMessageResponse,
)
async def append_message_to_task(
    task_id: int,
    request: AppendMessageRequest,
    principal: ApiKeyPrincipal = Depends(get_principal_from_api_key),
    db: Session = Depends(get_db),
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
         currently ``RUNNING`` -- the SDK client should poll
         ``GET /v1/chat/tasks/{id}`` until status leaves RUNNING and
         retry. Workforce turn rejections map to their own stable
         codes (``workforce_archived`` / ``workforce_config_changed``,
         409 -- NOT retryable) so clients aren't told to retry a
         permanently-rejected conversation.
      4. Otherwise persists the new user message to
         ``task_chat_messages``, updates ``task.input`` to record
         this turn's input, and kicks off the next background turn
         via the same helper POST uses.

    Args:
        task_id: Path parameter; the target task's primary key.
        request: Validated :class:`AppendMessageRequest`. ``message.content``
            is guaranteed non-empty by Pydantic.
        principal: Key-bound owner from the auth dependency.
        db: SQLAlchemy session.

    Returns:
        :class:`AppendMessageResponse` with the task identity and an
        ``accepted_at`` timestamp.

    Raises:
        V1ApiError 401: missing / invalid / revoked key.
        V1ApiError 404: task not found OR not owned by the key OR
            body.agent_id / body.workforce_id doesn't match the bound
            owner.
        V1ApiError 409: ``task_busy`` (retryable) or a workforce
            rejection code (not retryable).
        500: any other unexpected error (V1 envelope via global handler).
    """
    # Resolve task first so cross-owner leak protection (404 instead
    # of 403 for "not yours") fires before any body-level checks.
    task = _resolve_task_or_404(task_id, principal, db)
    task_owner_user_id = int(task.user_id)
    actor_user_id = principal.owner_user_id
    _key = principal.key_row

    if principal.agent is not None:
        # ``body.agent_id`` was a required field before workforce keys
        # existed; keep that contract for agent keys (422, matching the
        # old Pydantic required-field failure) now that the schema field
        # is optional for the workforce case.
        if request.agent_id is None:
            raise V1ApiError(
                V1ErrorCode.INVALID_INPUT,
                422,
                message="agent_id is required",
            )
        # body.agent_id mismatch is also a 404 -- but agent_not_found,
        # not task_not_found, because that's the field the caller got
        # wrong. Choosing AGENT_NOT_FOUND keeps it consistent with the
        # POST /v1/chat/tasks behavior for the same condition.
        if request.agent_id != principal.agent.id:
            raise V1ApiError(V1ErrorCode.AGENT_NOT_FOUND, 404)
        if request.workforce_id is not None:
            raise V1ApiError(V1ErrorCode.WORKFORCE_NOT_FOUND, 404)
    else:
        assert principal.workforce is not None
        # A workforce key never speaks for a bare agent: naming one is
        # the same "field you got wrong" 404 as an agent-key mismatch.
        if request.agent_id is not None:
            raise V1ApiError(V1ErrorCode.AGENT_NOT_FOUND, 404)
        # ``workforce_id`` is optional (the key already binds one) but
        # must match when provided -- symmetry with the agent contract.
        if (
            request.workforce_id is not None
            and request.workforce_id != principal.workforce.id
        ):
            raise V1ApiError(V1ErrorCode.WORKFORCE_NOT_FOUND, 404)

    # Connector runtime preparation always runs against the task's
    # executing agent: the key-bound agent for agent keys, the
    # workforce's manager agent (== task.agent_id) for workforce keys.
    runtime_agent = principal.agent
    if runtime_agent is None:
        runtime_agent = db.get(Agent, int(task.agent_id))
        if runtime_agent is None:
            raise V1ApiError(V1ErrorCode.INTERNAL_ERROR, 500)

    try:
        runtime_plan = prepare_append_connector_runtime(
            db=db,
            agent=runtime_agent,
            task=task,
            connector_user_id=task_owner_user_id,
            payload_items=request.connector_runtime_context,
        )
    except ConnectorRuntimeError as exc:
        _raise_v1_connector_runtime_error(exc)

    # Validate attached file ids before claiming the turn: an unresolvable id
    # is a 400 that must not mutate anything, and files must not be bound to a
    # turn that then 409s (task busy). ``task_id`` is passed so files already
    # bound to this task re-resolve idempotently. Binding runs after the claim.
    file_infos = _resolve_turn_files_or_400(
        file_ids=request.message.files or [],
        owner_user_id=task_owner_user_id,
        db=db,
        task_id=int(task.id),
    )

    # Orchestrator does the atomic claim (status must be terminal --
    # COMPLETED or FAILED -- to be appendable, so PENDING/RUNNING both
    # 409), persists the new user message, and schedules the bg turn
    # with a single-flight guard against concurrent kickoffs.
    payload = _turn_payload(request.message.content, file_infos)
    _store_connector_runtime_values_or_fail(
        db=db,
        task_id=int(task.id),
        turn_id=payload.turn_id,
        values_by_ref=runtime_plan.ephemeral_by_ref,
        mark_task_failed=False,
    )
    try:
        started = await TaskTurnOrchestrator.begin_turn(
            task_id=int(task.id),
            task_owner_user_id=task_owner_user_id,
            # The key-bound agent authorizes access; the persisted task owner
            # remains the runtime identity when those identities have drifted.
            actor_user_id=actor_user_id,
            payload=payload,
            kind=TurnKind.APPEND,
            force_fresh=False,
        )
    except TaskTurnNotFoundError:
        pop_ephemeral_runtime_values(payload.turn_id)
        raise V1ApiError(V1ErrorCode.TASK_NOT_FOUND, 404)
    except TaskTurnError as exc:
        pop_ephemeral_runtime_values(payload.turn_id)
        raise_for_turn_rejection(exc.reason)
    except Exception:
        pop_ephemeral_runtime_values(payload.turn_id)
        raise

    # Bind files only after the turn is claimed -- a 409 above leaves them
    # unbound and reusable. The bind can race the background runner, but the
    # runner's file query tolerates a NULL task_id (see create_chat_task), so
    # visibility doesn't depend on this commit landing first.
    bind_turn_files(
        file_ids=[info["file_id"] for info in file_infos],
        task_id=int(task.id),
        owner_user_id=task_owner_user_id,
        db=db,
    )

    # See the matching comment in create_chat_task: recorded here, not in
    # the shared auth dependency, so status polling elsewhere never counts.
    record_key_usage(str(_key.key_prefix))

    # ``status`` / ``accepted_at`` come from the orchestrator's committed-row
    # snapshot (``started``), not a post-call ``db.refresh(task)``. The
    # refresh was itself a blocking SELECT on the event loop; ``begin_turn``
    # already SELECTed ``updated_at`` (set by ``onupdate=func.now()`` on the
    # atomic UPDATE) inside its off-loop transaction, so SDK clients still see
    # a value matching what GET /v1/chat/tasks/{id} would return.
    return AppendMessageResponse(
        task_id=int(task.id),
        agent_id=int(task.agent_id),
        workforce_id=(
            int(principal.workforce.id) if principal.workforce is not None else None
        ),
        status=started.status.value,
        accepted_at=started.updated_at,
        run_id=started.run_id,
        state_version=started.state_version,
        control_state=started.control_state,
    )


@router.get("/chat/tasks/{task_id}", response_model=TaskInfoResponse)
async def get_chat_task(
    task_id: int,
    principal: ApiKeyPrincipal = Depends(get_principal_from_api_key),
    db: Session = Depends(get_db),
) -> TaskInfoResponse:
    """Return a snapshot of one task's current state.

    SDK clients call this to poll a previously-submitted task for
    its status, latest output, or failure reason. The shape is
    deliberately flat -- detailed step-by-step execution data lives
    behind ``GET /v1/chat/tasks/{task_id}/steps`` (commit F).

    Args:
        task_id: Path parameter; the target task's primary key.
        authed: ``(Agent, AgentApiKey)`` tuple.
        db: SQLAlchemy session.

    Returns:
        :class:`TaskInfoResponse` with ``task_id``, ``agent_id``,
        ``status``, latest-turn ``input`` / ``output`` / ``error``,
        ``created_at``, and ``completed_at`` (set only when the task
        has reached a terminal state).

    Raises:
        V1ApiError 401: missing / invalid / revoked key.
        V1ApiError 404: task missing or not owned by the calling key.
    """
    task = _resolve_task_or_404(task_id, principal, db)

    # completed_at is derived from updated_at when the task is in a
    # terminal state. Pre-terminal states return None so SDK clients
    # don't mis-interpret an in-flight task's last write timestamp as
    # a completion time.
    completed_at = task.updated_at if task.status in _TERMINAL_STATUSES else None
    cache_key = task_snapshot_key(task_id)
    task_updated_at = cache_version_token(task.updated_at)
    cached = cache_get(cache_key)
    if isinstance(cached, dict) and cached.get("updated_at") == task_updated_at:
        return TaskInfoResponse.model_validate(cached["response"])

    response = TaskInfoResponse(
        task_id=int(task.id),
        agent_id=int(task.agent_id),
        # A task is reachable by exactly one owner type (manager agents
        # can't hold keys), so caching the workforce identity inside the
        # per-task snapshot can't leak across principals.
        workforce_id=(
            int(principal.workforce.id) if principal.workforce is not None else None
        ),
        status=task.status.value,
        run_id=task.run_id,
        state_version=int(task.state_version or 0),
        control_state=str(task.control_state or "idle"),
        input=task.input,
        output=task.output,
        error=task.error_message,
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
    return response


@router.get("/chat/tasks/{task_id}/steps", response_model=StepsResponse)
async def get_chat_task_steps(
    task_id: int,
    principal: ApiKeyPrincipal = Depends(get_principal_from_api_key),
    db: Session = Depends(get_db),
) -> StepsResponse:
    """Return the public-timeline steps for a task.

    Pulls all :class:`TraceEvent` rows for the task in DB order, then
    collapses them via :func:`map_trace_events_to_public_steps` into
    the 4 stable public step types: ``thinking``, ``tool_call``,
    ``agent_delegation``, ``message``.

    The internal trace event taxonomy has ~32 ``event_type`` strings
    today; SDK callers see only the 4 types listed above. Internal
    events not on the public allow-list (LLM calls, memory ops,
    visualization ticks, DAG bookkeeping) are silently dropped --
    intentionally, so internal trace evolution doesn't break the SDK
    contract.

    Args:
        task_id: Path parameter; the target task's primary key.
        authed: ``(Agent, AgentApiKey)`` tuple resolved by the auth
            dependency. The agent here is the key-bound agent.
        db: SQLAlchemy session.

    Returns:
        :class:`StepsResponse` with ``task_id``, ``agent_id``, and the
        steps array in ``started_at`` ascending order. In-flight steps
        appear with ``status='running'`` and ``completed_at=null`` so
        SDK clients can poll this endpoint and observe progress.

    Raises:
        V1ApiError 401: missing / invalid / revoked key.
        V1ApiError 404: task missing or not owned by the calling key.
    """
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
    cache_key = task_steps_key(task_id)
    cached = cache_get(cache_key)
    if isinstance(cached, dict) and cached.get("max_event_id") == int(max_event_id):
        return StepsResponse.model_validate(cached["response"])

    # Ordered ASC by ``id`` -- the trace_events PK is monotonically
    # increasing per write, so ordering by it gives us the same
    # temporal ordering as ``timestamp`` but without depending on
    # clock-skew within a single task's write fan-out.
    events = (
        db.query(TraceEvent)
        .filter(
            TraceEvent.task_id == task_id,
            TraceEvent.build_id.is_(None),
        )
        .order_by(TraceEvent.id.asc())
        .all()
    )

    # Pure mapping -- testable in isolation via
    # tests/web/api/v1/test_steps_mapping.py without spinning up a
    # FastAPI app or DB session.
    public_steps_data = map_trace_events_to_public_steps(events)

    response = StepsResponse(
        task_id=int(task.id),
        agent_id=int(task.agent_id),
        steps=[PublicStep(**step) for step in public_steps_data],
    )
    cache_set(
        cache_key,
        {
            "max_event_id": int(max_event_id),
            "response": response.model_dump(mode="json"),
        },
        ttl_seconds=task_cache_ttl_seconds(),
    )
    return response
