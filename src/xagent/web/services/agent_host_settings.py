"""Let a host stage settings in the same transaction as an Agent save."""

from collections.abc import Callable
from typing import Any

from fastapi import HTTPException
from sqlalchemy.orm import Session

from ..models.agent import Agent
from ..models.user import User

AgentHostSettingsWriter = Callable[[Session, Agent, User, dict[str, Any]], None]
_writer: AgentHostSettingsWriter | None = None


def set_agent_host_settings_writer(writer: AgentHostSettingsWriter | None) -> None:
    global _writer
    _writer = writer


def stage_agent_host_settings(
    db: Session, agent: Agent, user: User, settings: dict[str, Any]
) -> None:
    """Stage host settings without committing the caller's transaction."""
    if _writer is None:
        raise HTTPException(422, "This deployment does not support Agent host settings")
    _writer(db, agent, user, settings)
