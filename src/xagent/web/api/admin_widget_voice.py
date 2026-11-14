"""Administrator switch for widget voice on an agent or a workforce.

This is the only writer of ``agents.widget_voice_enabled`` and
``deployments.widget_voice_enabled``: owners cannot turn widget voice on for
their own widgets. ``available`` reports whether the deployment's voice
models resolve for the entity's owner, so a switch that is on but has no
effect can be explained.
"""

from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from ..models.agent import Agent
from ..models.database import get_db
from ..models.deployment import Deployment, DeploymentOwnerType
from ..models.user import User
from ..models.workforce import Workforce
from ..services.deployments import get_deployment
from ..services.widget_voice import (
    WidgetVoiceResolution,
    resolve_widget_voice_models,
)
from .admin_memory_embedding_authority import require_admin

router = APIRouter(prefix="/api/admin/widget-voice", tags=["admin-widget-voice"])


class WidgetVoiceSwitchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    enabled: bool


class WidgetVoiceSwitchResponse(BaseModel):
    enabled: bool
    available: bool
    reason: str | None = None


def _response(
    enabled: bool, resolution: WidgetVoiceResolution
) -> WidgetVoiceSwitchResponse:
    return WidgetVoiceSwitchResponse(
        enabled=enabled,
        available=resolution.models is not None,
        reason=resolution.reason,
    )


def _agent(db: Session, agent_id: int) -> Agent:
    agent = db.get(Agent, agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


def _workforce_deployment(
    db: Session, workforce_id: int
) -> tuple[Workforce, Deployment]:
    # Read only: a workforce without a deployment row has no widget to voice.
    workforce = db.get(Workforce, workforce_id)
    deployment = (
        get_deployment(db, DeploymentOwnerType.WORKFORCE, workforce_id)
        if workforce is not None
        else None
    )
    if workforce is None or deployment is None:
        raise HTTPException(status_code=404, detail="Workforce deployment not found")
    return workforce, deployment


@router.get("/agents/{agent_id}", response_model=WidgetVoiceSwitchResponse)
def get_agent_widget_voice(
    agent_id: int,
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> WidgetVoiceSwitchResponse:
    agent = _agent(db, agent_id)
    resolution = resolve_widget_voice_models(int(cast(Any, agent.user_id)))
    return _response(bool(agent.widget_voice_enabled), resolution)


@router.put("/agents/{agent_id}", response_model=WidgetVoiceSwitchResponse)
def put_agent_widget_voice(
    agent_id: int,
    request: WidgetVoiceSwitchRequest,
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> WidgetVoiceSwitchResponse:
    agent = _agent(db, agent_id)
    # Resolve first: a failed read leaves the switch as it was.
    resolution = resolve_widget_voice_models(int(cast(Any, agent.user_id)))
    cast(Any, agent).widget_voice_enabled = request.enabled
    db.commit()
    return _response(request.enabled, resolution)


@router.get("/workforces/{workforce_id}", response_model=WidgetVoiceSwitchResponse)
def get_workforce_widget_voice(
    workforce_id: int,
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> WidgetVoiceSwitchResponse:
    workforce, deployment = _workforce_deployment(db, workforce_id)
    resolution = resolve_widget_voice_models(int(cast(Any, workforce.owner_user_id)))
    return _response(bool(deployment.widget_voice_enabled), resolution)


@router.put("/workforces/{workforce_id}", response_model=WidgetVoiceSwitchResponse)
def put_workforce_widget_voice(
    workforce_id: int,
    request: WidgetVoiceSwitchRequest,
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> WidgetVoiceSwitchResponse:
    workforce, deployment = _workforce_deployment(db, workforce_id)
    # Resolve first: a failed read leaves the switch as it was.
    resolution = resolve_widget_voice_models(int(cast(Any, workforce.owner_user_id)))
    cast(Any, deployment).widget_voice_enabled = request.enabled
    db.commit()
    return _response(request.enabled, resolution)
