from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from time import monotonic
from typing import Any, Mapping, Tuple

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from ..models.agent import Agent
from ..models.agent_api_key import AgentApiKey
from ..models.database import get_db, get_session_local
from ..models.task import Task, TaskStatus
from ..services.a2a_protocol import (
    A2A_VERSION,
    ALL_TASK_STATES,
    a2a_error,
    a2a_json_response,
    build_agent_card,
    extract_message_text,
    is_published_agent,
    message_context_id,
    message_task_id,
    new_context_id,
    sse_task_artifacts,
    sse_task_snapshot,
    sse_task_update,
    task_context_id,
    task_state,
    task_to_a2a,
)
from ..services.task_orchestrator import (
    TaskTurnError,
    TaskTurnNotFoundError,
    TaskTurnOrchestrator,
    TaskTurnPayload,
    TurnKind,
)
from .v1.deps import get_agent_from_api_key, record_key_usage
from .v1.errors import V1ApiError

router = APIRouter(prefix="/api/a2a", tags=["a2a"])
_bearer = HTTPBearer(auto_error=False)
_TERMINAL_STATUSES = {TaskStatus.COMPLETED, TaskStatus.FAILED}
_STREAM_END_STATUSES = _TERMINAL_STATUSES | {
    TaskStatus.PAUSED,
    TaskStatus.WAITING_FOR_USER,
}
A2A_BLOCKING_WAIT_TIMEOUT_SECONDS = 60.0
A2A_STREAM_MAX_DURATION_SECONDS = 60.0 * 60.0


async def _get_a2a_agent_from_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> Tuple[Agent, AgentApiKey]:
    _validate_a2a_version(request)
    try:
        return await get_agent_from_api_key(credentials, db)
    except V1ApiError as exc:
        raise a2a_error(
            "invalid_api_key",
            exc.message,
            status_code=exc.http_status,
        ) from exc


def _resolve_published_agent(db: Session, agent_id: int) -> Agent:
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if agent is None or not is_published_agent(agent):
        raise a2a_error("agent_not_found", "Agent not found.", status_code=404)
    return agent


def _resolve_a2a_task(db: Session, task_id: int, agent: Agent) -> Task:
    task = (
        db.query(Task)
        .filter(
            Task.id == task_id,
            Task.agent_id == int(agent.id),
            Task.source == "a2a",
        )
        .first()
    )
    if task is None:
        raise a2a_error("task_not_found", "Task not found.", status_code=404)
    return task


def _require_bound_agent(path_agent_id: int, agent: Agent) -> None:
    if int(agent.id) != int(path_agent_id) or not is_published_agent(agent):
        raise a2a_error("agent_not_found", "Agent not found.", status_code=404)


def _validate_a2a_version(request: Request) -> None:
    requested = request.headers.get("A2A-Version")
    if requested is None:
        requested = request.query_params.get("A2A-Version")
    if requested is None or not requested.strip():
        raise a2a_error(
            "version_not_supported",
            "A2A-Version header or query parameter is required.",
            status_code=400,
            details={"supportedVersions": A2A_VERSION},
        )
    requested = requested.strip()
    version_parts = requested.split(".")
    compatible = (
        len(version_parts) in {2, 3}
        and all(part.isdecimal() for part in version_parts)
        and version_parts[0] == A2A_VERSION.split(".", maxsplit=1)[0]
    )
    if not compatible:
        raise a2a_error(
            "version_not_supported",
            f"A2A protocol version {requested!r} is not supported.",
            status_code=400,
            details={"supportedVersions": A2A_VERSION},
        )


def _validate_send_configuration(body: Mapping[str, Any]) -> bool:
    configuration = body.get("configuration")
    if configuration is None:
        return False
    if not isinstance(configuration, Mapping):
        raise a2a_error(
            "invalid_argument",
            "configuration must be a JSON object.",
            status_code=400,
            details={"field": "configuration"},
        )
    if configuration.get("taskPushNotificationConfig") is not None:
        raise a2a_error(
            "push_notification_not_supported",
            "This agent does not support A2A push notifications.",
            status_code=400,
        )
    accepted_modes = configuration.get("acceptedOutputModes")
    if accepted_modes is not None:
        if not isinstance(accepted_modes, list) or not all(
            isinstance(mode, str) for mode in accepted_modes
        ):
            raise a2a_error(
                "invalid_argument",
                "acceptedOutputModes must be an array of media types.",
                status_code=400,
                details={"field": "configuration.acceptedOutputModes"},
            )
        if accepted_modes and "text/plain" not in accepted_modes:
            raise a2a_error(
                "content_type_not_supported",
                "This agent currently returns text/plain output only.",
                status_code=400,
                details={"supportedMediaType": "text/plain"},
            )
    return_immediately = configuration.get("returnImmediately", False)
    if not isinstance(return_immediately, bool):
        raise a2a_error(
            "invalid_argument",
            "returnImmediately must be a boolean.",
            status_code=400,
            details={"field": "configuration.returnImmediately"},
        )
    return return_immediately


async def _start_a2a_turn(
    *,
    db: Session,
    agent: Agent,
    text: str,
    context_id: str | None,
    task_id: int | None,
) -> Task:
    created_task = task_id is None
    normalized_waiting_task = False
    if task_id is None:
        context_id = context_id or new_context_id()
        task = Task(
            user_id=int(agent.user_id),
            title=(text[:50] or "A2A task"),
            description=text,
            status=TaskStatus.PENDING,
            agent_id=int(agent.id),
            input=text,
            source="a2a",
            is_visible=False,
            execution_mode=agent.execution_mode,
            agent_config={"a2a_context_id": context_id},
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        kind = TurnKind.CREATE
    else:
        task = _resolve_a2a_task(db, task_id, agent)
        if task.status in _TERMINAL_STATUSES:
            raise a2a_error(
                "unsupported_operation",
                "Messages cannot be appended to a terminal A2A task.",
                status_code=400,
                details={"taskId": task.id},
            )
        stored_context_id = task_context_id(task)
        if context_id is not None and context_id != stored_context_id:
            raise a2a_error(
                "invalid_argument",
                "The supplied contextId does not match the referenced task.",
                status_code=400,
                details={"taskId": task.id, "contextId": context_id},
            )
        context_id = stored_context_id
        agent_config: dict[str, Any] = (
            dict(task.agent_config) if isinstance(task.agent_config, dict) else {}
        )
        if not agent_config.get("a2a_context_id"):
            agent_config["a2a_context_id"] = context_id
            setattr(task, "agent_config", agent_config)
            db.commit()
            db.refresh(task)
        if task.status == TaskStatus.WAITING_FOR_USER:
            # The web runtime's new-message path resumes PAUSED tasks as a new
            # turn. Normalize WAITING_FOR_USER to that same durable path so an
            # A2A follow-up does not depend on an in-memory WebSocket session.
            task.status = TaskStatus.PAUSED
            db.commit()
            db.refresh(task)
            normalized_waiting_task = True
        kind = TurnKind.APPEND

    payload = TaskTurnPayload(transcript_message=text)
    try:
        await TaskTurnOrchestrator.begin_turn(
            task_id=int(task.id),
            task_owner_user_id=int(agent.user_id),
            actor_user_id=int(agent.user_id),
            payload=payload,
            kind=kind,
            force_fresh=False,
        )
    except TaskTurnNotFoundError as exc:
        _recover_failed_turn_start(
            db,
            int(task.id),
            created_task=created_task,
            restore_waiting=normalized_waiting_task,
        )
        raise a2a_error("task_not_found", "Task not found.", status_code=404) from exc
    except TaskTurnError as exc:
        _recover_failed_turn_start(
            db,
            int(task.id),
            created_task=created_task,
            restore_waiting=normalized_waiting_task,
        )
        raise a2a_error(
            "unsupported_operation",
            "Task is currently running and cannot accept a new message.",
            status_code=400,
            details={"taskId": task.id},
        ) from exc
    except Exception:
        _recover_failed_turn_start(
            db,
            int(task.id),
            created_task=created_task,
            restore_waiting=normalized_waiting_task,
        )
        raise

    db.expire_all()
    return _resolve_a2a_task(db, int(task.id), agent)


def _recover_failed_turn_start(
    db: Session,
    task_id: int,
    *,
    created_task: bool,
    restore_waiting: bool,
) -> None:
    db.expire_all()
    task = db.query(Task).filter(Task.id == task_id).first()
    if task is None:
        return
    if created_task and task.status == TaskStatus.PENDING:
        db.delete(task)
        db.commit()
        return
    if restore_waiting and task.status == TaskStatus.PAUSED:
        task.status = TaskStatus.WAITING_FOR_USER
        db.commit()


async def _json_body(request: Request) -> Mapping[str, Any]:
    try:
        body = await request.json()
    except ValueError as exc:
        raise a2a_error(
            "invalid_request",
            "Request body must be valid JSON.",
            status_code=400,
        ) from exc
    if not isinstance(body, Mapping):
        raise a2a_error(
            "invalid_argument", "Request body must be a JSON object.", status_code=400
        )
    return body


def _message_payload(body: Mapping[str, Any]) -> Mapping[str, Any]:
    message = body.get("message")
    if not isinstance(message, Mapping):
        raise a2a_error(
            "invalid_argument",
            "Request body must include a message object.",
            status_code=400,
        )
    return message


def _fetch_fresh_a2a_task(agent_id: int, task_id: int) -> Task | None:
    session_local = get_session_local()
    local_db = session_local()
    try:
        fresh = (
            local_db.query(Task)
            .filter(
                Task.id == task_id,
                Task.agent_id == agent_id,
                Task.source == "a2a",
            )
            .first()
        )
        if fresh is not None:
            local_db.expunge(fresh)
        return fresh
    finally:
        local_db.close()


def _task_stream_response(agent: Agent, task: Task) -> StreamingResponse:
    started_task_id = int(task.id)

    async def _events() -> Any:
        deadline = monotonic() + A2A_STREAM_MAX_DURATION_SECONDS
        yield sse_task_snapshot(task)
        if task.status in _STREAM_END_STATUSES:
            return
        previous_state = task_state(task)
        previous_output = str(task.output or "")
        previous_error = str(task.error_message or "")
        while True:
            remaining = deadline - monotonic()
            if remaining <= 0:
                return
            await asyncio.sleep(min(0.5, remaining))
            fresh = _fetch_fresh_a2a_task(int(agent.id), started_task_id)
            if fresh is None:
                return
            fresh_output = str(fresh.output or "")
            fresh_state = task_state(fresh)
            fresh_error = str(fresh.error_message or "")
            if fresh_output and fresh_output != previous_output:
                artifacts = sse_task_artifacts(fresh)
                if artifacts:
                    yield artifacts
            if fresh_state != previous_state or fresh_error != previous_error:
                yield sse_task_update(fresh)
            previous_state = fresh_state
            previous_output = fresh_output
            previous_error = fresh_error
            if fresh.status in _STREAM_END_STATUSES:
                return

    return StreamingResponse(
        _events(),
        media_type="text/event-stream",
        headers={"A2A-Version": A2A_VERSION},
    )


async def _wait_for_task(agent: Agent, task: Task) -> Task:
    if task.status in _STREAM_END_STATUSES:
        return task
    task_id = int(task.id)
    deadline = monotonic() + A2A_BLOCKING_WAIT_TIMEOUT_SECONDS
    fresh = task
    while True:
        remaining = deadline - monotonic()
        if remaining <= 0:
            return fresh
        await asyncio.sleep(min(0.25, remaining))
        fetched = _fetch_fresh_a2a_task(int(agent.id), task_id)
        if fetched is None:
            raise a2a_error("task_not_found", "Task not found.", status_code=404)
        fresh = fetched
        if fresh.status in _STREAM_END_STATUSES:
            return fresh


def _page_offset(page_token: str | None) -> int:
    if page_token is None or page_token == "":
        return 0
    if page_token.isdecimal():
        return int(page_token)
    raise a2a_error(
        "invalid_argument",
        "pageToken is invalid.",
        status_code=400,
        details={"field": "pageToken"},
    )


def _is_after(value: Any, threshold: datetime) -> bool:
    if not isinstance(value, datetime):
        return False
    timestamp: datetime = value
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    if threshold.tzinfo is None:
        threshold = threshold.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc) > threshold.astimezone(timezone.utc)


@router.get("/agents/{agent_id}/.well-known/agent-card.json")
async def get_agent_card_well_known(
    agent_id: int,
    request: Request,
    db: Session = Depends(get_db),
) -> Any:
    agent = _resolve_published_agent(db, agent_id)
    return a2a_json_response(build_agent_card(agent, request))


@router.post("/agents/{agent_id}/message:send")
async def send_message(
    agent_id: int,
    request: Request,
    authed: Tuple[Agent, AgentApiKey] = Depends(_get_a2a_agent_from_api_key),
    db: Session = Depends(get_db),
) -> Any:
    agent, key = authed
    _require_bound_agent(agent_id, agent)
    body = await _json_body(request)
    return_immediately = _validate_send_configuration(body)
    message = _message_payload(body)
    text = extract_message_text(message)
    context_id = message_context_id(message, body)
    task_id = message_task_id(message, body)
    task = await _start_a2a_turn(
        db=db,
        agent=agent,
        text=text,
        context_id=context_id,
        task_id=task_id,
    )
    record_key_usage(str(key.key_prefix))
    if not return_immediately:
        task = await _wait_for_task(agent, task)
    return a2a_json_response({"task": task_to_a2a(task)})


@router.post("/agents/{agent_id}/message:stream")
async def stream_message(
    agent_id: int,
    request: Request,
    authed: Tuple[Agent, AgentApiKey] = Depends(_get_a2a_agent_from_api_key),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    agent, key = authed
    _require_bound_agent(agent_id, agent)
    body = await _json_body(request)
    _validate_send_configuration(body)
    message = _message_payload(body)
    text = extract_message_text(message)
    context_id = message_context_id(message, body)
    task_id = message_task_id(message, body)
    task = await _start_a2a_turn(
        db=db,
        agent=agent,
        text=text,
        context_id=context_id,
        task_id=task_id,
    )
    record_key_usage(str(key.key_prefix))
    return _task_stream_response(agent, task)


@router.get("/agents/{agent_id}/tasks/{task_id}")
async def get_task(
    agent_id: int,
    task_id: int,
    authed: Tuple[Agent, AgentApiKey] = Depends(_get_a2a_agent_from_api_key),
    db: Session = Depends(get_db),
) -> Any:
    agent, _key = authed
    _require_bound_agent(agent_id, agent)
    task = _resolve_a2a_task(db, task_id, agent)
    return a2a_json_response(task_to_a2a(task))


@router.get("/agents/{agent_id}/tasks")
async def list_tasks(
    agent_id: int,
    context_id: str | None = Query(default=None, alias="contextId"),
    status: str | None = Query(default=None),
    page_size: int = Query(default=50, ge=1, le=100, alias="pageSize"),
    page_token: str | None = Query(default=None, alias="pageToken"),
    include_artifacts: bool = Query(default=False, alias="includeArtifacts"),
    status_timestamp_after: datetime | None = Query(
        default=None, alias="statusTimestampAfter"
    ),
    authed: Tuple[Agent, AgentApiKey] = Depends(_get_a2a_agent_from_api_key),
    db: Session = Depends(get_db),
) -> Any:
    agent, _key = authed
    _require_bound_agent(agent_id, agent)
    tasks = (
        db.query(Task)
        .filter(Task.agent_id == int(agent.id), Task.source == "a2a")
        .order_by(Task.id.desc())
        .all()
    )
    if context_id is not None:
        tasks = [task for task in tasks if task_context_id(task) == context_id]
    if status is not None:
        if status not in ALL_TASK_STATES:
            raise a2a_error(
                "invalid_argument",
                f"Unknown A2A task status: {status}",
                status_code=400,
                details={"field": "status"},
            )
        tasks = [task for task in tasks if task_state(task) == status]
    if status_timestamp_after is not None:
        tasks = [
            task for task in tasks if _is_after(task.updated_at, status_timestamp_after)
        ]

    offset = _page_offset(page_token)
    total_size = len(tasks)
    page = tasks[offset : offset + page_size]
    next_offset = offset + len(page)
    next_page_token = str(next_offset) if next_offset < total_size else ""
    return a2a_json_response(
        {
            "tasks": [
                task_to_a2a(task, include_artifacts=include_artifacts) for task in page
            ],
            "nextPageToken": next_page_token,
            "pageSize": page_size,
            "totalSize": total_size,
        }
    )


@router.api_route(
    "/agents/{agent_id}/tasks/{task_id}:subscribe", methods=["GET", "POST"]
)
async def subscribe_task(
    agent_id: int,
    task_id: int,
    authed: Tuple[Agent, AgentApiKey] = Depends(_get_a2a_agent_from_api_key),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    agent, _key = authed
    _require_bound_agent(agent_id, agent)
    task = _resolve_a2a_task(db, task_id, agent)
    if task.status in _TERMINAL_STATUSES:
        raise a2a_error(
            "unsupported_operation",
            "A terminal task cannot be subscribed to.",
            status_code=400,
            details={"taskId": task.id},
        )
    return _task_stream_response(agent, task)


@router.post("/agents/{agent_id}/tasks/{task_id}:cancel")
async def cancel_task(
    agent_id: int,
    task_id: int,
    authed: Tuple[Agent, AgentApiKey] = Depends(_get_a2a_agent_from_api_key),
    db: Session = Depends(get_db),
) -> Any:
    agent, _key = authed
    _require_bound_agent(agent_id, agent)
    task = _resolve_a2a_task(db, task_id, agent)
    agent_config: dict[str, Any] = (
        dict(task.agent_config) if isinstance(task.agent_config, dict) else {}
    )
    if agent_config.get("a2a_state") == "TASK_STATE_CANCELED":
        return a2a_json_response(task_to_a2a(task))
    if task.status in _TERMINAL_STATUSES:
        raise a2a_error(
            "task_not_cancelable",
            "Task is not in a cancelable state.",
            status_code=400,
            details={"taskId": task.id},
        )

    from .websocket import background_task_manager

    await background_task_manager.cancel_task(int(task.id))
    agent_config["a2a_state"] = "TASK_STATE_CANCELED"
    setattr(task, "agent_config", agent_config)
    task.status = TaskStatus.FAILED
    setattr(task, "output", None)
    setattr(task, "error_message", "Task canceled by A2A client.")
    db.commit()
    db.refresh(task)
    return a2a_json_response(task_to_a2a(task))
