"""Owner switch for widget voice on an agent or a workforce.

Whoever may edit an agent (``AgentStore.get_owned_agent``) or a workforce
(``can_edit_workforce``) can read and write its widget voice switch here. The
switch is the same column the administrator API writes: the last write wins.

``available`` reports whether both of the deployment's voice models are usable
for the entity's owner, so a switch that is on but has no effect can be
explained. It is ``None`` while the switch is off: nothing was checked, and the
voice provider is not asked.

Every route frees the request session's database connection before the voice
provider is asked, because the lookup takes connections from the same pool.
Once it is freed, nothing here touches the request session or the ORM user
object again: the caller is carried as plain values (``OwnerCaller``) and every
ownership check runs in short sessions of its own, once before the provider is
asked and once more on the row that is written.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal, cast

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from ..models.agent import Agent
from ..models.database import get_session_local
from ..models.deployment import Deployment, DeploymentOwnerType
from ..models.user import User
from ..models.workforce import Workforce
from .agent_store import AgentStore
from .agent_team_scope import get_agent_team_scope
from .db_runtime import run_db_io_cancellation_safe
from .widget_voice import (
    WidgetVoiceAvailability,
    release_request_connection,
    resolve_widget_voice_availability,
)
from .workforce_access import can_edit_workforce

logger = logging.getLogger(__name__)


class OwnerWidgetVoiceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    enabled: bool


class OwnerWidgetVoiceResponse(BaseModel):
    enabled: bool
    available: bool | None = None
    reason: (
        Literal["voice_models_not_configured", "voice_models_unavailable"] | None
    ) = None


@dataclass(frozen=True)
class OwnerCaller:
    """The calling user as plain values, safe to use after the request session
    has returned its connection."""

    user_id: int
    is_admin: bool

    @classmethod
    def of(cls, user: User) -> OwnerCaller:
        return cls(user_id=int(cast(Any, user.id)), is_admin=bool(user.is_admin))


@dataclass(frozen=True)
class _Target:
    """What a route needs from the target row, detached from its session."""

    owner_user_id: int
    enabled: bool


def _off() -> OwnerWidgetVoiceResponse:
    return OwnerWidgetVoiceResponse(enabled=False, available=None, reason=None)


def _checked(
    enabled: bool, availability: WidgetVoiceAvailability
) -> OwnerWidgetVoiceResponse:
    return OwnerWidgetVoiceResponse(
        enabled=enabled,
        available=availability.available,
        reason=cast(Any, availability.reason),
    )


# ===== Agents =====


def _owned_agent(
    db: Session, caller: OwnerCaller, agent_id: int, *, for_update: bool = False
) -> Agent:
    team_scope = get_agent_team_scope(db, caller.user_id)
    agent = AgentStore(db).get_owned_agent(
        caller.user_id, agent_id, team_scope, for_update=for_update
    )
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


def _read_agent(caller: OwnerCaller, agent_id: int) -> _Target:
    with get_session_local()() as db:
        agent = _owned_agent(db, caller, agent_id)
        return _Target(int(cast(Any, agent.user_id)), bool(agent.widget_voice_enabled))


def _write_agent(caller: OwnerCaller, agent_id: int, enabled: bool) -> bool:
    """Write the switch on the row read again with ownership; return the old value."""
    with get_session_local()() as db:
        agent = _owned_agent(db, caller, agent_id, for_update=True)
        previous = bool(agent.widget_voice_enabled)
        cast(Any, agent).widget_voice_enabled = enabled
        db.commit()
        return previous


async def get_agent_voice(
    db: Session, caller: OwnerCaller, agent_id: int
) -> OwnerWidgetVoiceResponse:
    release_request_connection(db)
    target = await run_db_io_cancellation_safe(lambda: _read_agent(caller, agent_id))
    if not target.enabled:
        return _off()
    availability = await resolve_widget_voice_availability(
        target.owner_user_id, request_db=db
    )
    return _checked(True, availability)


async def put_agent_voice(
    db: Session, caller: OwnerCaller, agent_id: int, enabled: bool
) -> OwnerWidgetVoiceResponse:
    release_request_connection(db)
    target = await run_db_io_cancellation_safe(lambda: _read_agent(caller, agent_id))
    availability = (
        await resolve_widget_voice_availability(target.owner_user_id, request_db=db)
        if enabled
        else None
    )
    previous = await run_db_io_cancellation_safe(
        lambda: _write_agent(caller, agent_id, enabled)
    )
    _log_write("agent_id", agent_id, caller, enabled, previous, availability)
    return _checked(True, availability) if availability is not None else _off()


# ===== Workforces =====


def _editable_workforce(
    db: Session, caller: OwnerCaller, workforce_id: int
) -> Workforce:
    user = db.get(User, caller.user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    workforce = db.get(Workforce, workforce_id)
    if workforce is None:
        raise HTTPException(status_code=404, detail="Workforce not found")
    if not can_edit_workforce(db, user, workforce):
        raise HTTPException(status_code=403, detail="Access denied")
    return workforce


def _deployment(
    db: Session, workforce_id: int, *, for_update: bool = False
) -> Deployment | None:
    query = db.query(Deployment).filter(
        Deployment.owner_type == DeploymentOwnerType.WORKFORCE.value,
        Deployment.owner_id == int(workforce_id),
    )
    if for_update:
        query = query.with_for_update()
    return query.first()


def _guard_write(
    workforce: Workforce, deployment: Deployment | None, enabled: bool
) -> Deployment:
    if deployment is None:
        raise HTTPException(status_code=404, detail="Workforce widget not enabled")
    if enabled and workforce.status != "active":
        raise HTTPException(
            status_code=400, detail="Only active workforces can enable voice"
        )
    return deployment


def _read_workforce(
    caller: OwnerCaller, workforce_id: int, write_value: bool | None
) -> _Target | None:
    """The workforce's owner and switch, or ``None`` when it has no deployment
    row (reads only; a write to such a workforce is a 404)."""
    with get_session_local()() as db:
        workforce = _editable_workforce(db, caller, workforce_id)
        deployment = _deployment(db, workforce_id)
        if write_value is not None:
            _guard_write(workforce, deployment, write_value)
        if deployment is None:
            return None
        return _Target(
            int(cast(Any, workforce.owner_user_id)),
            bool(deployment.widget_voice_enabled),
        )


def _write_workforce(caller: OwnerCaller, workforce_id: int, enabled: bool) -> bool:
    """Write the switch on the row read again with ownership; return the old value."""
    with get_session_local()() as db:
        workforce = _editable_workforce(db, caller, workforce_id)
        deployment = _guard_write(
            workforce, _deployment(db, workforce_id, for_update=True), enabled
        )
        previous = bool(deployment.widget_voice_enabled)
        cast(Any, deployment).widget_voice_enabled = enabled
        db.commit()
        return previous


async def get_workforce_voice(
    db: Session, caller: OwnerCaller, workforce_id: int
) -> OwnerWidgetVoiceResponse:
    release_request_connection(db)
    target = await run_db_io_cancellation_safe(
        lambda: _read_workforce(caller, workforce_id, None)
    )
    if target is None or not target.enabled:
        return _off()
    availability = await resolve_widget_voice_availability(
        target.owner_user_id, request_db=db
    )
    return _checked(True, availability)


async def put_workforce_voice(
    db: Session, caller: OwnerCaller, workforce_id: int, enabled: bool
) -> OwnerWidgetVoiceResponse:
    release_request_connection(db)
    target = await run_db_io_cancellation_safe(
        lambda: _read_workforce(caller, workforce_id, enabled)
    )
    if target is None:
        raise HTTPException(status_code=404, detail="Workforce widget not enabled")
    availability = (
        await resolve_widget_voice_availability(target.owner_user_id, request_db=db)
        if enabled
        else None
    )
    previous = await run_db_io_cancellation_safe(
        lambda: _write_workforce(caller, workforce_id, enabled)
    )
    _log_write("workforce_id", workforce_id, caller, enabled, previous, availability)
    return _checked(True, availability) if availability is not None else _off()


def _log_write(
    id_key: str,
    entity_id: int,
    caller: OwnerCaller,
    enabled: bool,
    previous: bool,
    availability: WidgetVoiceAvailability | None,
) -> None:
    logger.info(
        "Widget voice switch set by owner",
        extra={
            id_key: entity_id,
            "actor_user_id": caller.user_id,
            "actor_is_admin": caller.is_admin,
            "widget_voice_enabled": enabled,
            "previous_widget_voice_enabled": previous,
            "available": availability.available if availability else None,
        },
    )
