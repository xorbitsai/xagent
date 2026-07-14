"""Team-scope seam for agent ownership and visibility.

Standalone xagent leaves the hook unset, so ``get_agent_team_scope`` returns
``None`` and agents stay purely user-owned with no visibility gating. The SaaS
overlay installs a hook that maps a user id to the team that owns their agents
and whether that user is a team admin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, cast

from sqlalchemy import and_, or_
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from ..models.agent import Agent


@dataclass(frozen=True)
class AgentTeamScope:
    team_id: int
    is_team_admin: bool


_agent_team_scope_hook = None  # (db, user_id) -> AgentTeamScope | None


def set_agent_team_scope_hook(hook: Any) -> None:
    """Install (or clear, with ``None``) the user-id -> AgentTeamScope resolver."""
    global _agent_team_scope_hook
    _agent_team_scope_hook = hook


def get_agent_team_scope(
    db: Session, user_id: Optional[int]
) -> Optional[AgentTeamScope]:
    """Return the caller's team scope, or ``None`` when unscoped."""
    if _agent_team_scope_hook is None or user_id is None:
        return None
    return cast(Optional[AgentTeamScope], _agent_team_scope_hook(db, user_id))


def team_id_of(scope: Optional[AgentTeamScope]) -> Optional[int]:
    """Extract the team id from a scope (for stamping/cache keys)."""
    return scope.team_id if scope is not None else None


def owns_agent(agent: Agent, user_id: int, scope: Optional[AgentTeamScope]) -> bool:
    """Per-agent (in-memory) mirror of :func:`owned_agent_clause`.

    Use this wherever a single loaded ``Agent`` is authorized/serialized so the
    workforce gates and ownership serialization stay in lockstep with the list
    query. Keep the branches identical to ``owned_agent_clause``.
    """
    if scope is None or agent.team_id is None:
        return int(agent.user_id) == int(user_id)
    if int(agent.team_id) != int(scope.team_id):
        return False
    return bool(scope.is_team_admin or agent.visibility == "team")


def owned_agent_clause(
    user_id: int, scope: Optional[AgentTeamScope]
) -> ColumnElement[bool]:
    """Predicate for agents *user_id* may see/manage.

    - No scope (standalone / no hook): exactly ``Agent.user_id == user_id``.
    - Scoped team admin: every agent in the team, regardless of visibility.
    - Scoped non-admin: team agents whose ``visibility == 'team'`` only.
    A legacy ``team_id IS NULL`` row still resolves via its own ``user_id``.
    """
    if scope is None:
        return Agent.user_id == user_id
    if scope.is_team_admin:
        team_clause = Agent.team_id == scope.team_id
    else:
        team_clause = and_(Agent.team_id == scope.team_id, Agent.visibility == "team")
    return or_(
        and_(Agent.team_id.is_(None), Agent.user_id == user_id),
        team_clause,
    )
