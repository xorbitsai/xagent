"""Team-scope seam for agent ownership and visibility.

Standalone xagent leaves the hook unset, so ``get_agent_team_scope`` returns
``None`` and agents stay purely user-owned with no visibility gating. The SaaS
overlay installs a hook that maps a user id to the team that owns their agents
and whether that user is a team admin.
"""

from __future__ import annotations

import re
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

# (db, user_id, team_id, tool_categories) -> list of unshared connector dicts.
_team_agent_connector_validator = None

# (db, user_id, team_id, knowledge_bases) -> list of unshared KB dicts.
_team_agent_knowledge_base_validator = None

_MAX_POSTGRES_INTEGER = 2**31 - 1
_MAX_POSTGRES_INTEGER_DECIMAL_DIGITS = len(str(_MAX_POSTGRES_INTEGER))
_CANONICAL_AGENT_ID = re.compile(r"[1-9][0-9]*")


def set_agent_team_hooks(
    *,
    scope: Any = None,
    connector_validator: Any = None,
    knowledge_base_validator: Any = None,
) -> None:
    """Install or clear the complete application-owned agent team hook set."""
    global _agent_team_scope_hook
    global _team_agent_connector_validator, _team_agent_knowledge_base_validator
    _agent_team_scope_hook = scope
    _team_agent_connector_validator = connector_validator
    _team_agent_knowledge_base_validator = knowledge_base_validator


def set_agent_team_scope_hook(hook: Any) -> None:
    """Install (or clear, with ``None``) the user-id -> AgentTeamScope resolver."""
    global _agent_team_scope_hook
    _agent_team_scope_hook = hook


def validate_team_agent_connectors(
    db: Session, user_id: int, team_id: int, tool_categories: Any
) -> list:
    """Return the agent's selected connectors that are not shared with the team.

    Empty list when no validator is installed (standalone) or nothing is
    unshared. Each item is a ``{"type", "id", "name"}`` dict.
    """
    if _team_agent_connector_validator is None:
        return []
    return cast(
        list,
        _team_agent_connector_validator(db, user_id, team_id, tool_categories),
    )


def validate_team_agent_knowledge_bases(
    db: Session, user_id: int, team_id: int, knowledge_bases: Any
) -> list:
    """Return selected knowledge bases not owned by the agent's team."""
    if _team_agent_knowledge_base_validator is None:
        return []
    return cast(
        list,
        _team_agent_knowledge_base_validator(db, user_id, team_id, knowledge_bases),
    )


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


def resolve_authorized_agent(
    db: Session, owner_user_id: int, candidate_id: Any
) -> Agent | None:
    """Resolve a canonical Agent id within the task owner's team scope.

    Persisted task config is untrusted at every read boundary. Reject malformed
    values before constructing the SQL predicate, then reuse the same owner/team
    clause as all normal Agent reads.

    Lifecycle status is caller-owned; each use case must apply its required status policy after resolution.
    """
    if isinstance(candidate_id, bool):
        return None
    if isinstance(candidate_id, int):
        agent_id = candidate_id
    elif isinstance(candidate_id, str):
        # Bound untrusted text before regex scanning or integer conversion.
        if len(candidate_id) > _MAX_POSTGRES_INTEGER_DECIMAL_DIGITS:
            return None
        if not _CANONICAL_AGENT_ID.fullmatch(candidate_id):
            return None
        agent_id = int(candidate_id)
    else:
        return None
    if not 0 < agent_id <= _MAX_POSTGRES_INTEGER:
        return None
    return (
        db.query(Agent)
        .filter(
            Agent.id == agent_id,
            owned_agent_clause(owner_user_id, get_agent_team_scope(db, owner_user_id)),
        )
        .first()
    )
