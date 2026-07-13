"""Agent visibility helpers shared by agent and workforce APIs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from sqlalchemy.orm import Session

from ..models.agent import Agent, AgentOrigin, AgentStatus
from ..models.user import User
from .agent_team_scope import get_agent_team_scope, owned_agent_clause
from .workforce_access import get_visible_agent_ids

AgentAccessLevel = Literal["owner", "policy"]


@dataclass(frozen=True)
class AccessibleAgent:
    agent: Agent
    access: AgentAccessLevel
    can_edit: bool
    can_publish: bool
    can_delete: bool

    @property
    def readonly(self) -> bool:
        return not self.can_edit


def accessible_agent_permissions(accessible_agent: AccessibleAgent) -> dict[str, Any]:
    return {
        "access": accessible_agent.access,
        "readonly": accessible_agent.readonly,
        "can_edit": accessible_agent.can_edit,
        "can_publish": accessible_agent.can_publish,
        "can_delete": accessible_agent.can_delete,
    }


def _normalize_excluded_agent_ids(values: Iterable[int] | None) -> set[int]:
    normalized: set[int] = set()
    for value in values or []:
        if isinstance(value, int):
            normalized.add(value)
    return normalized


def _owned_accessible_agent(agent: Agent) -> AccessibleAgent:
    return AccessibleAgent(
        agent=agent,
        access="owner",
        can_edit=True,
        can_publish=True,
        can_delete=True,
    )


def _policy_accessible_agent(agent: Agent) -> AccessibleAgent:
    return AccessibleAgent(
        agent=agent,
        access="policy",
        can_edit=False,
        can_publish=False,
        can_delete=False,
    )


def _sort_by_created_desc(items: list[AccessibleAgent]) -> list[AccessibleAgent]:
    def sort_key(item: AccessibleAgent) -> tuple[float, int]:
        created_at = item.agent.created_at
        timestamp = created_at.timestamp() if created_at else 0.0
        return (timestamp, int(item.agent.id or 0))

    return sorted(
        items,
        key=sort_key,
        reverse=True,
    )


def _is_published(agent: Agent) -> bool:
    return getattr(agent.status, "value", agent.status) == AgentStatus.PUBLISHED.value


def _exclude_private_workforce_manager_agents(query: Any) -> Any:
    return query.filter(Agent.origin != AgentOrigin.WORKFORCE_GENERATED_MANAGER.value)


def list_accessible_agents(
    db: Session,
    user: User,
    *,
    purpose: str = "agent_list",
    exclude_agent_ids: Iterable[int] | None = None,
    include_private_workforce_manager_agents: bool = False,
) -> list[AccessibleAgent]:
    """List agents visible to the user, including owned and policy-visible drafts."""
    excluded = _normalize_excluded_agent_ids(exclude_agent_ids)
    items_by_id: dict[int, AccessibleAgent] = {}

    team_scope = get_agent_team_scope(db, int(user.id))
    owned_query = db.query(Agent).filter(owned_agent_clause(int(user.id), team_scope))
    if excluded:
        owned_query = owned_query.filter(Agent.id.notin_(excluded))
    if not include_private_workforce_manager_agents:
        owned_query = _exclude_private_workforce_manager_agents(owned_query)
    for agent in owned_query.all():
        items_by_id[int(agent.id)] = _owned_accessible_agent(agent)

    if user.is_admin:
        policy_query = db.query(Agent)
    else:
        visible_agent_ids = get_visible_agent_ids(db, user, purpose)
        if not visible_agent_ids:
            return _sort_by_created_desc(list(items_by_id.values()))
        # Admins-only agents surface only through the owned clause (team admin);
        # the policy/published path must never leak them to non-admins.
        policy_query = db.query(Agent).filter(
            Agent.id.in_(visible_agent_ids), Agent.visibility != "admins"
        )

    if excluded:
        policy_query = policy_query.filter(Agent.id.notin_(excluded))
    if not include_private_workforce_manager_agents:
        policy_query = _exclude_private_workforce_manager_agents(policy_query)

    for agent in policy_query.all():
        agent_id = int(agent.id)
        if agent_id in items_by_id:
            continue
        items_by_id[agent_id] = _policy_accessible_agent(agent)

    return _sort_by_created_desc(list(items_by_id.values()))


def list_accessible_published_agent_items(
    db: Session,
    user: User,
    *,
    purpose: str = "workforce_select",
    exclude_agent_ids: Iterable[int] | None = None,
    include_private_workforce_manager_agents: bool = False,
) -> list[AccessibleAgent]:
    return sorted(
        [
            item
            for item in list_accessible_agents(
                db,
                user,
                purpose=purpose,
                exclude_agent_ids=exclude_agent_ids,
                include_private_workforce_manager_agents=include_private_workforce_manager_agents,
            )
            if _is_published(item.agent)
        ],
        key=lambda item: int(item.agent.id or 0),
    )


def list_accessible_published_agents(
    db: Session,
    user: User,
    *,
    purpose: str = "workforce_select",
    exclude_agent_ids: Iterable[int] | None = None,
    include_private_workforce_manager_agents: bool = False,
) -> list[Agent]:
    return [
        item.agent
        for item in list_accessible_published_agent_items(
            db,
            user,
            purpose=purpose,
            exclude_agent_ids=exclude_agent_ids,
            include_private_workforce_manager_agents=include_private_workforce_manager_agents,
        )
    ]
