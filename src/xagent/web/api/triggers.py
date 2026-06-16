from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Literal, cast

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..auth_dependencies import get_current_user
from ..models.database import get_db
from ..models.trigger import AgentTrigger, TriggerRun
from ..models.user import User
from ..services.triggers import (
    TriggerNotFoundError,
    TriggerSecretError,
    TriggerServiceError,
    create_agent_trigger,
    delete_agent_trigger,
    find_webhook_trigger,
    fire_trigger,
    get_owned_agent,
    get_owned_trigger,
    update_agent_trigger,
    verify_webhook_secret,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["triggers"])


class TriggerCreateRequest(BaseModel):
    type: Literal["webhook", "scheduled"]
    name: str | None = Field(default=None, max_length=200)
    enabled: bool = True
    config: dict[str, Any] = Field(default_factory=dict)
    prompt_template: str | None = None
    secret: str | None = None


class TriggerUpdateRequest(BaseModel):
    name: str | None = Field(default=None, max_length=200)
    enabled: bool | None = None
    config: dict[str, Any] | None = None
    prompt_template: str | None = None
    secret: str | None = None
    rotate_secret: bool = False


class TriggerTestRequest(BaseModel):
    payload: dict[str, Any] = Field(default_factory=dict)
    source_event_id: str | None = None


class TriggerResponse(BaseModel):
    id: int
    user_id: int
    agent_id: int
    type: str
    name: str
    enabled: bool
    config: dict[str, Any]
    prompt_template: str | None
    webhook_token: str | None
    webhook_secret: str | None = None
    next_run_at: str | None
    last_run_at: str | None
    last_error: str | None
    created_at: str | None
    updated_at: str | None


class TriggerRunResponse(BaseModel):
    id: int
    trigger_id: int
    task_id: int | None
    background_job_id: str | None
    status: str
    source_event_id: str | None
    payload_snapshot: dict[str, Any] | None
    idempotency_key: str
    error_message: str | None
    started_at: str | None
    finished_at: str | None
    created_at: str | None
    updated_at: str | None


class TriggerFireResponse(BaseModel):
    trigger_run: TriggerRunResponse
    duplicate: bool = False


class PublicTriggerFireResponse(BaseModel):
    trigger_run_id: int
    status: str
    duplicate: bool = False


def _dt(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _serialize_trigger(
    trigger: AgentTrigger, *, webhook_secret: str | None = None
) -> TriggerResponse:
    return TriggerResponse(
        id=int(trigger.id),
        user_id=int(trigger.user_id),
        agent_id=int(trigger.agent_id),
        type=str(trigger.type),
        name=str(trigger.name),
        enabled=bool(trigger.enabled),
        config=dict(trigger.config or {}),
        prompt_template=trigger.prompt_template,
        webhook_token=trigger.webhook_token,
        webhook_secret=webhook_secret,
        next_run_at=_dt(cast(datetime | None, trigger.next_run_at)),
        last_run_at=_dt(cast(datetime | None, trigger.last_run_at)),
        last_error=trigger.last_error,
        created_at=_dt(cast(datetime | None, trigger.created_at)),
        updated_at=_dt(cast(datetime | None, trigger.updated_at)),
    )


def _serialize_run(run: TriggerRun) -> TriggerRunResponse:
    payload = run.payload_snapshot if isinstance(run.payload_snapshot, dict) else None
    return TriggerRunResponse(
        id=int(run.id),
        trigger_id=int(run.trigger_id),
        task_id=int(run.task_id) if run.task_id is not None else None,
        background_job_id=run.background_job_id,
        status=str(run.status),
        source_event_id=run.source_event_id,
        payload_snapshot=payload,
        idempotency_key=str(run.idempotency_key),
        error_message=run.error_message,
        started_at=_dt(getattr(run, "started_at", None)),
        finished_at=_dt(getattr(run, "finished_at", None)),
        created_at=_dt(getattr(run, "created_at", None)),
        updated_at=_dt(getattr(run, "updated_at", None)),
    )


def _agent_or_404(db: Session, *, user_id: int, agent_id: int) -> None:
    if get_owned_agent(db, user_id=user_id, agent_id=agent_id) is None:
        raise HTTPException(status_code=404, detail="Agent not found")


def _trigger_or_404(
    db: Session,
    *,
    user_id: int,
    agent_id: int,
    trigger_id: int,
) -> AgentTrigger:
    trigger = get_owned_trigger(
        db, user_id=user_id, agent_id=agent_id, trigger_id=trigger_id
    )
    if trigger is None:
        raise HTTPException(status_code=404, detail="Trigger not found")
    return trigger


def _handle_service_error(exc: Exception) -> HTTPException:
    if isinstance(exc, TriggerNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, TriggerSecretError):
        return HTTPException(status_code=401, detail=str(exc))
    if isinstance(exc, TriggerServiceError):
        return HTTPException(status_code=400, detail=str(exc))
    logger.exception("Unhandled trigger API error")
    return HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/api/agents/{agent_id}/triggers",
    response_model=list[TriggerResponse],
)
async def list_triggers(
    agent_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[TriggerResponse]:
    user_id = int(current_user.id)
    _agent_or_404(db, user_id=user_id, agent_id=agent_id)
    rows = (
        db.query(AgentTrigger)
        .filter(AgentTrigger.user_id == user_id, AgentTrigger.agent_id == agent_id)
        .order_by(AgentTrigger.created_at.desc(), AgentTrigger.id.desc())
        .all()
    )
    return [_serialize_trigger(row) for row in rows]


@router.post(
    "/api/agents/{agent_id}/triggers",
    response_model=TriggerResponse,
)
async def create_trigger(
    agent_id: int,
    request: TriggerCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TriggerResponse:
    try:
        trigger, secret = create_agent_trigger(
            db,
            user_id=int(current_user.id),
            agent_id=agent_id,
            trigger_type=request.type,
            name=request.name,
            enabled=request.enabled,
            config=request.config,
            prompt_template=request.prompt_template,
            secret=request.secret,
        )
        return _serialize_trigger(trigger, webhook_secret=secret)
    except Exception as exc:
        raise _handle_service_error(exc)


@router.patch(
    "/api/agents/{agent_id}/triggers/{trigger_id}",
    response_model=TriggerResponse,
)
async def update_trigger(
    agent_id: int,
    trigger_id: int,
    request: TriggerUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TriggerResponse:
    try:
        trigger, secret = update_agent_trigger(
            db,
            user_id=int(current_user.id),
            agent_id=agent_id,
            trigger_id=trigger_id,
            updates=request.model_dump(exclude_unset=True),
        )
        return _serialize_trigger(trigger, webhook_secret=secret)
    except Exception as exc:
        raise _handle_service_error(exc)


@router.delete("/api/agents/{agent_id}/triggers/{trigger_id}")
async def delete_trigger(
    agent_id: int,
    trigger_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    try:
        delete_agent_trigger(
            db,
            user_id=int(current_user.id),
            agent_id=agent_id,
            trigger_id=trigger_id,
        )
        return {"message": "Trigger deleted"}
    except Exception as exc:
        raise _handle_service_error(exc)


@router.get(
    "/api/agents/{agent_id}/triggers/{trigger_id}/runs",
    response_model=list[TriggerRunResponse],
)
async def list_trigger_runs(
    agent_id: int,
    trigger_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[TriggerRunResponse]:
    trigger = _trigger_or_404(
        db,
        user_id=int(current_user.id),
        agent_id=agent_id,
        trigger_id=trigger_id,
    )
    rows = (
        db.query(TriggerRun)
        .filter(TriggerRun.trigger_id == int(trigger.id))
        .order_by(TriggerRun.created_at.desc(), TriggerRun.id.desc())
        .limit(100)
        .all()
    )
    return [_serialize_run(row) for row in rows]


@router.post(
    "/api/agents/{agent_id}/triggers/{trigger_id}/test",
    response_model=TriggerFireResponse,
)
async def test_trigger(
    agent_id: int,
    trigger_id: int,
    request: TriggerTestRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TriggerFireResponse:
    trigger = _trigger_or_404(
        db,
        user_id=int(current_user.id),
        agent_id=agent_id,
        trigger_id=trigger_id,
    )
    try:
        run, created = await fire_trigger(
            db,
            trigger=trigger,
            event_payload=request.payload,
            source_event_id=request.source_event_id,
            test=True,
        )
        return TriggerFireResponse(
            trigger_run=_serialize_run(run), duplicate=not created
        )
    except Exception as exc:
        raise _handle_service_error(exc)


async def _read_payload(request: Request) -> dict[str, Any]:
    body = await request.body()
    if not body:
        return {}
    try:
        decoded = json.loads(body.decode("utf-8"))
    except ValueError:
        return {"body": body.decode("utf-8", errors="replace")}
    if isinstance(decoded, dict):
        return decoded
    return {"value": decoded}


@router.post(
    "/api/triggers/webhook/{webhook_token}",
    response_model=PublicTriggerFireResponse,
)
async def receive_webhook_trigger(
    webhook_token: str,
    request: Request,
    db: Session = Depends(get_db),
) -> PublicTriggerFireResponse:
    trigger = find_webhook_trigger(db, webhook_token)
    if trigger is None:
        raise HTTPException(status_code=404, detail="Trigger not found")
    if not trigger.enabled:
        raise HTTPException(status_code=409, detail="Trigger is disabled")

    secret = request.headers.get("x-xagent-trigger-secret")
    try:
        verify_webhook_secret(trigger, secret)
        payload = await _read_payload(request)
        source_event_id = (
            request.headers.get("x-xagent-event-id")
            or request.headers.get("x-event-id")
            or request.headers.get("x-request-id")
        )
        run, created = await fire_trigger(
            db,
            trigger=trigger,
            event_payload=payload,
            source_event_id=source_event_id,
        )
        return PublicTriggerFireResponse(
            trigger_run_id=int(run.id),
            status=str(run.status),
            duplicate=not created,
        )
    except Exception as exc:
        logger.warning("Webhook trigger %s rejected: %s", trigger.id, exc)
        raise _handle_service_error(exc)
