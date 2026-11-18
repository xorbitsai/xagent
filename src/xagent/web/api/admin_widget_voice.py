"""Administrator switch for widget voice on an agent or a workforce.

Writes ``agents.widget_voice_enabled`` and ``deployments.widget_voice_enabled``
for any agent or workforce. Owners write the same columns for the ones they can
edit through ``/api/agents/{id}/widget-voice`` and
``/api/workforces/{id}/widget-voice``; the last write wins. ``available``
reports whether both of the deployment's voice models (speech recognition and
speech synthesis) are usable for the entity's owner, so a switch that is on but
has no effect can be explained. Unlike the owner routes, these always ask the
voice provider, whether the switch is on or off.

The request body, the target row view and the write log are the owner routes'
(``services.widget_voice_switch``): each write logs the same fields. Unlike an
owner write, an administrator write takes no row lock, because it does not
depend on the value it replaces, and it may turn voice on for a workforce that
is not active, so the switch can be set before the workforce is activated.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, cast

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..models.agent import Agent
from ..models.database import get_db, get_session_local
from ..models.deployment import Deployment, DeploymentOwnerType
from ..models.user import User
from ..models.workforce import Workforce
from ..services.db_runtime import run_db_io_cancellation_safe
from ..services.deployments import get_deployment
from ..services.widget_voice import (
    WidgetVoiceAvailability,
    release_request_connection,
    resolve_widget_voice_availability,
)
from ..services.widget_voice_switch import (
    WidgetVoiceSwitchRequest,
    WidgetVoiceTarget,
    log_switch_write,
)
from .admin_memory_embedding_authority import require_admin

router = APIRouter(prefix="/api/admin/widget-voice", tags=["admin-widget-voice"])


class WidgetVoiceSwitchResponse(BaseModel):
    enabled: bool
    available: bool
    reason: str | None = None


def _response(
    enabled: bool, availability: WidgetVoiceAvailability
) -> WidgetVoiceSwitchResponse:
    return WidgetVoiceSwitchResponse(
        enabled=enabled,
        available=availability.available,
        reason=availability.reason,
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


def _read_agent(agent_id: int) -> WidgetVoiceTarget:
    with get_session_local()() as db:
        agent = _agent(db, agent_id)
        return WidgetVoiceTarget(
            int(cast(Any, agent.user_id)), bool(agent.widget_voice_enabled)
        )


def _write_agent(agent_id: int, enabled: bool) -> bool:
    """Write the switch; return the value read just before it (no row lock)."""
    with get_session_local()() as db:
        agent = _agent(db, agent_id)
        previous = bool(agent.widget_voice_enabled)
        cast(Any, agent).widget_voice_enabled = enabled
        db.commit()
        return previous


def _read_workforce(workforce_id: int) -> WidgetVoiceTarget:
    with get_session_local()() as db:
        workforce, deployment = _workforce_deployment(db, workforce_id)
        return WidgetVoiceTarget(
            int(cast(Any, workforce.owner_user_id)),
            bool(deployment.widget_voice_enabled),
        )


def _write_workforce(workforce_id: int, enabled: bool) -> bool:
    """Write the switch; return the value read just before it (no row lock)."""
    with get_session_local()() as db:
        _, deployment = _workforce_deployment(db, workforce_id)
        previous = bool(deployment.widget_voice_enabled)
        cast(Any, deployment).widget_voice_enabled = enabled
        db.commit()
        return previous


async def _target(
    db: Session, read: Callable[[], WidgetVoiceTarget]
) -> WidgetVoiceTarget:
    """Read the target in a short session of its own.

    ``db`` is the request session the admin check ran on; its connection goes
    back to the pool first, so this request never waits for a second one. A
    session that cannot return it raises ``RuntimeError``.
    """
    release_request_connection(db)
    return await run_db_io_cancellation_safe(read)


async def _get(
    db: Session, read: Callable[[], WidgetVoiceTarget]
) -> WidgetVoiceSwitchResponse:
    target = await _target(db, read)
    availability = await resolve_widget_voice_availability(
        target.owner_user_id, request_db=db
    )
    return _response(target.enabled, availability)


async def _put(
    db: Session,
    read: Callable[[], WidgetVoiceTarget],
    write: Callable[[], bool],
    enabled: bool,
    *,
    actor_user_id: int,
    id_key: Literal["agent_id", "workforce_id"],
    entity_id: int,
) -> WidgetVoiceSwitchResponse:
    target = await _target(db, read)
    # Resolve before writing: a lookup that raises leaves the switch as it was.
    availability = await resolve_widget_voice_availability(
        target.owner_user_id, request_db=db
    )
    # The row is read again: it may have been deleted meanwhile (404). A new
    # owner does not matter to the write: the switch belongs to the row, not to
    # its owner. ``available`` in the response describes the owner read before
    # the lookup.
    previous = await run_db_io_cancellation_safe(write)
    log_switch_write(
        set_by="administrator",
        id_key=id_key,
        entity_id=entity_id,
        actor_user_id=actor_user_id,
        actor_is_admin=True,
        enabled=enabled,
        previous=previous,
        available=availability.available,
    )
    return _response(enabled, availability)


@router.get("/agents/{agent_id}", response_model=WidgetVoiceSwitchResponse)
async def get_agent_widget_voice(
    agent_id: int,
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> WidgetVoiceSwitchResponse:
    return await _get(db, lambda: _read_agent(agent_id))


@router.put("/agents/{agent_id}", response_model=WidgetVoiceSwitchResponse)
async def put_agent_widget_voice(
    agent_id: int,
    request: WidgetVoiceSwitchRequest,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> WidgetVoiceSwitchResponse:
    # Plain value first: the request session returns its connection next.
    actor_user_id = int(cast(Any, admin.id))
    return await _put(
        db,
        lambda: _read_agent(agent_id),
        lambda: _write_agent(agent_id, request.enabled),
        request.enabled,
        actor_user_id=actor_user_id,
        id_key="agent_id",
        entity_id=agent_id,
    )


@router.get("/workforces/{workforce_id}", response_model=WidgetVoiceSwitchResponse)
async def get_workforce_widget_voice(
    workforce_id: int,
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> WidgetVoiceSwitchResponse:
    return await _get(db, lambda: _read_workforce(workforce_id))


@router.put("/workforces/{workforce_id}", response_model=WidgetVoiceSwitchResponse)
async def put_workforce_widget_voice(
    workforce_id: int,
    request: WidgetVoiceSwitchRequest,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> WidgetVoiceSwitchResponse:
    # Plain value first: the request session returns its connection next.
    actor_user_id = int(cast(Any, admin.id))
    return await _put(
        db,
        lambda: _read_workforce(workforce_id),
        lambda: _write_workforce(workforce_id, request.enabled),
        request.enabled,
        actor_user_id=actor_user_id,
        id_key="workforce_id",
        entity_id=workforce_id,
    )
