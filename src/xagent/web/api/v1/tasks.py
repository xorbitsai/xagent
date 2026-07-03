"""SDK task endpoints: ``/v1/chat/tasks/*`` family.

Phase 1 surface this module owns:

  - POST /v1/chat/tasks
  - POST /v1/chat/tasks/{id}/messages
  - GET  /v1/chat/tasks/{id}
  - GET  /v1/chat/tasks/{id}/steps

All endpoints authenticate via ``get_agent_from_api_key`` and use the
stable ``V1ApiError`` envelope. Task turn lifecycle (claim RUNNING,
persist messages, schedule bg, sync output) is delegated to
``services.task_orchestrator.TaskTurnOrchestrator``, which is also used
by the WebSocket UI path so both transports share one state machine.
"""

from typing import Tuple

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from ...models.agent import Agent
from ...models.agent_api_key import AgentApiKey
from ...models.database import get_db
from ...models.task import Task, TaskStatus, TraceEvent
from ...schemas.v1 import (
    AppendMessageRequest,
    AppendMessageResponse,
    CreateTaskRequest,
    CreateTaskResponse,
    PublicStep,
    StepsResponse,
    TaskInfoResponse,
)
from ...services.hot_path_cache import (
    cache_get,
    cache_set,
    cache_version_token,
    task_cache_ttl_seconds,
    task_snapshot_key,
    task_steps_key,
)
from ...services.task_orchestrator import (
    TaskTurnError,
    TaskTurnNotFoundError,
    TaskTurnOrchestrator,
    TaskTurnPayload,
    TurnKind,
)
from ._step_mapping import map_trace_events_to_public_steps
from .deps import get_agent_from_api_key, record_key_usage
from .errors import V1ApiError, V1ErrorCode

router = APIRouter()


@router.post(
    "/chat/tasks",
    status_code=202,
    response_model=CreateTaskResponse,
)
async def create_chat_task(
    request: CreateTaskRequest,
    authed: Tuple[Agent, AgentApiKey] = Depends(get_agent_from_api_key),
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
    agent, _key = authed

    # Server-side agent_id consistency check. The key already binds an
    # agent; ``body.agent_id`` is required by the SDK contract for
    # forward-compat (and Python/TS SDK symmetry), but the bound
    # agent is the only authority. Mismatch is a 404 -- never a 403
    # -- so the existence of agent_id=N elsewhere in the system isn't
    # observable to this caller.
    if request.agent_id != agent.id:
        raise V1ApiError(V1ErrorCode.AGENT_NOT_FOUND, 404)

    # Record usage here (not in the shared auth dependency) so read-only
    # status/steps polling below doesn't count as a "call" -- only actual
    # task-creating/message-appending requests do.
    record_key_usage(str(_key.key_prefix))

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
        user_id=agent.user_id,
        title=title,
        description=request.message.content,
        status=TaskStatus.PENDING,
        agent_id=agent.id,
        input=request.message.content,
        source="sdk",
        is_visible=False,
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    # Orchestrator's begin_turn handles the full new-turn transition:
    # bg-inflight guard, atomic status flip + transcript persist in one
    # commit, and bg coroutine scheduling under a lease lifecycle.
    # A brand-new task shouldn't ever hit busy -- but we map it
    # anyway for defense.
    try:
        started = await TaskTurnOrchestrator.begin_turn(
            task_id=int(task.id),
            task_owner_user_id=int(agent.user_id),
            # SDK key resolves to the agent owner; actor == owner here.
            actor_user_id=int(agent.user_id),
            payload=TaskTurnPayload(transcript_message=request.message.content),
            kind=TurnKind.CREATE,
            force_fresh=False,
        )
    except TaskTurnNotFoundError:
        raise V1ApiError(V1ErrorCode.TASK_NOT_FOUND, 404)
    except TaskTurnError:
        raise V1ApiError(V1ErrorCode.TASK_BUSY, 409)

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
    )


# Terminal task statuses for ``completed_at`` derivation in GET task.
# A task in any of these states is no longer running; ``updated_at``
# is the last DB write and thus the closest proxy to "when did the
# task end". For non-terminal states we return ``None`` so SDK
# clients can disambiguate "still running" from "ended at <time>".
_TERMINAL_STATUSES = (TaskStatus.COMPLETED, TaskStatus.FAILED)


def _resolve_task_or_404(task_id: int, agent: Agent, db: Session) -> Task:
    """Resolve a task_id against the calling agent's ownership AND
    SDK-source scope.

    Returns the :class:`Task` row when the task:

      1. Exists.
      2. Belongs to ``agent``.
      3. Was created by the SDK (``source == "sdk"``).

    Any other case — missing row, row belongs to a different agent,
    or row was created by the Web UI / internal paths — raises
    :class:`V1ApiError` with ``task_not_found`` (404 not 403, so the
    existence of tasks under other agents / other surfaces isn't
    observable through error code).

    The ``source == "sdk"`` filter exists because an SDK API key
    binds to an agent, not to a particular product surface. Without
    it, an SDK client could read or append to any task the Web UI
    created under the same agent (the user's own historical Web UI
    chats, for example). Whether that's intentional is a product
    decision, but the safe default for a public SDK is to scope
    lookups to tasks the SDK itself created — ``POST /v1/chat/tasks``
    writes ``source="sdk"`` so this is well-defined.

    Args:
        task_id: Path parameter from the route.
        agent: The key-bound agent resolved by
            ``get_agent_from_api_key``.
        db: SQLAlchemy session.

    Raises:
        V1ApiError(TASK_NOT_FOUND, 404): task missing, not owned by
            the calling agent, or not created by the SDK.
    """
    task = (
        db.query(Task)
        .filter(
            Task.id == task_id,
            Task.agent_id == agent.id,
            Task.source == "sdk",
        )
        .first()
    )
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
    authed: Tuple[Agent, AgentApiKey] = Depends(get_agent_from_api_key),
    db: Session = Depends(get_db),
) -> AppendMessageResponse:
    """Append the next user message to an existing task and kick off its next turn.

    Phase 1 multi-turn model is task-centric: subsequent user inputs
    extend the same ``task_id`` rather than creating a new task or a
    new ``conversation_id``. This endpoint:

      1. Validates the path ``task_id`` exists and belongs to the
         key-bound agent (404 ``task_not_found`` otherwise).
      2. Validates ``body.agent_id`` matches the key-bound agent
         (404 ``agent_not_found`` otherwise).
      3. Rejects the call with 409 ``task_busy`` if the task is
         currently ``RUNNING`` -- the SDK client should poll
         ``GET /v1/chat/tasks/{id}`` until status leaves RUNNING and
         retry.
      4. Otherwise persists the new user message to
         ``task_chat_messages``, updates ``task.input`` to record
         this turn's input, and kicks off the next background turn
         via the same helper POST uses.

    Args:
        task_id: Path parameter; the target task's primary key.
        request: Validated :class:`AppendMessageRequest`. ``message.content``
            is guaranteed non-empty by Pydantic.
        authed: ``(Agent, AgentApiKey)`` from the auth dependency.
        db: SQLAlchemy session.

    Returns:
        :class:`AppendMessageResponse` with the task identity and an
        ``accepted_at`` timestamp.

    Raises:
        V1ApiError 401: missing / invalid / revoked key.
        V1ApiError 404: task not found OR not owned by the agent OR
            body.agent_id doesn't match the bound agent.
        V1ApiError 409: ``task_busy`` -- task currently RUNNING.
        500: any other unexpected error (V1 envelope via global handler).
    """
    agent, _key = authed

    # Resolve task first so cross-agent leak protection (404 instead
    # of 403 for "not yours") fires before any body-level checks.
    task = _resolve_task_or_404(task_id, agent, db)

    # body.agent_id mismatch is also a 404 -- but agent_not_found,
    # not task_not_found, because that's the field the caller got
    # wrong. Choosing AGENT_NOT_FOUND keeps it consistent with the
    # POST /v1/chat/tasks behavior for the same condition.
    if request.agent_id != agent.id:
        raise V1ApiError(V1ErrorCode.AGENT_NOT_FOUND, 404)

    # See the matching comment in create_chat_task: recorded here, not in
    # the shared auth dependency, so status polling elsewhere never counts.
    record_key_usage(str(_key.key_prefix))

    # Orchestrator does the atomic claim (status must be terminal --
    # COMPLETED or FAILED -- to be appendable, so PENDING/RUNNING both
    # 409), persists the new user message, and schedules the bg turn
    # with a single-flight guard against concurrent kickoffs.
    try:
        started = await TaskTurnOrchestrator.begin_turn(
            task_id=int(task.id),
            task_owner_user_id=int(agent.user_id),
            # SDK key resolves to the agent owner; actor == owner here.
            actor_user_id=int(agent.user_id),
            payload=TaskTurnPayload(transcript_message=request.message.content),
            kind=TurnKind.APPEND,
            force_fresh=False,
        )
    except TaskTurnNotFoundError:
        raise V1ApiError(V1ErrorCode.TASK_NOT_FOUND, 404)
    except TaskTurnError:
        raise V1ApiError(V1ErrorCode.TASK_BUSY, 409)

    # ``status`` / ``accepted_at`` come from the orchestrator's committed-row
    # snapshot (``started``), not a post-call ``db.refresh(task)``. The
    # refresh was itself a blocking SELECT on the event loop; ``begin_turn``
    # already SELECTed ``updated_at`` (set by ``onupdate=func.now()`` on the
    # atomic UPDATE) inside its off-loop transaction, so SDK clients still see
    # a value matching what GET /v1/chat/tasks/{id} would return.
    return AppendMessageResponse(
        task_id=int(task.id),
        agent_id=int(agent.id),
        status=started.status.value,
        accepted_at=started.updated_at,
    )


@router.get("/chat/tasks/{task_id}", response_model=TaskInfoResponse)
async def get_chat_task(
    task_id: int,
    authed: Tuple[Agent, AgentApiKey] = Depends(get_agent_from_api_key),
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
        V1ApiError 404: task missing or not owned by the calling agent.
    """
    agent, _key = authed
    task = _resolve_task_or_404(task_id, agent, db)

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
        status=task.status.value,
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
    authed: Tuple[Agent, AgentApiKey] = Depends(get_agent_from_api_key),
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
        V1ApiError 404: task missing or not owned by the calling agent.
    """
    agent, _key = authed
    task = _resolve_task_or_404(task_id, agent, db)

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
