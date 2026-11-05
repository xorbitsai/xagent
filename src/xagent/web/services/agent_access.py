"""Agent visibility helpers shared by agent and workforce APIs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from sqlalchemy import or_
from sqlalchemy.orm import Query, Session

from ..models.agent import Agent, AgentOrigin, AgentStatus
from ..models.user import User
from .agent_team_scope import get_agent_team_scope, owned_agent_clause, owns_agent
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


def query_accessible_agents(
    db: Session,
    user: User,
    *,
    purpose: str = "agent_list",
    exclude_agent_ids: Iterable[int] | None = None,
    include_private_workforce_manager_agents: bool = False,
) -> Query[Agent]:
    """Build the canonical query for Agents visible to ``user``.

    Keep ownership and policy visibility in one SQL predicate so paginated and
    bounded list callers cannot drift from the normal Agent list semantics.
    """
    query = db.query(Agent)
    if not user.is_admin:
        team_scope = get_agent_team_scope(db, int(user.id))
        access_clauses = [owned_agent_clause(int(user.id), team_scope)]
        visible_agent_ids = get_visible_agent_ids(db, user, purpose)
        if visible_agent_ids:
            # Admins-only Agents surface only through the owned clause (team
            # admin); policy visibility must never leak them to non-admins.
            access_clauses.append(
                (Agent.id.in_(visible_agent_ids)) & (Agent.visibility != "admins")
            )
        query = query.filter(or_(*access_clauses))

    excluded = _normalize_excluded_agent_ids(exclude_agent_ids)
    if excluded:
        query = query.filter(Agent.id.notin_(excluded))
    if not include_private_workforce_manager_agents:
        query = _exclude_private_workforce_manager_agents(query)
    return query


def list_accessible_agents(
    db: Session,
    user: User,
    *,
    purpose: str = "agent_list",
    exclude_agent_ids: Iterable[int] | None = None,
    include_private_workforce_manager_agents: bool = False,
) -> list[AccessibleAgent]:
    """List agents visible to the user, including owned and policy-visible drafts."""
    team_scope = get_agent_team_scope(db, int(user.id))
    items = [
        (
            _owned_accessible_agent(agent)
            if owns_agent(agent, int(user.id), team_scope)
            else _policy_accessible_agent(agent)
        )
        for agent in query_accessible_agents(
            db,
            user,
            purpose=purpose,
            exclude_agent_ids=exclude_agent_ids,
            include_private_workforce_manager_agents=include_private_workforce_manager_agents,
        ).all()
    ]
    return _sort_by_created_desc(items)


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
    limit: int | None = None,
) -> list[Agent]:
    if limit is not None:
        bounded_limit = max(1, int(limit))
        access_query = query_accessible_agents(
            db,
            user,
            purpose=purpose,
            exclude_agent_ids=exclude_agent_ids,
            include_private_workforce_manager_agents=include_private_workforce_manager_agents,
        )
        access_query = access_query.filter(
            Agent.__table__.c.status == AgentStatus.PUBLISHED.value
        )
        return list(access_query.order_by(Agent.id).limit(bounded_limit).all())

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


def can_user_run_agent_now(db: Session, user_id: int, agent: Agent) -> bool:
    """Whether ``user_id`` may run ``agent`` directly, checked against live state.

    The same two branches the direct-run task path accepts an agent through:
    the caller's own or team-visible agent (``owns_agent``, the in-memory
    mirror of ``owned_agent_clause``), or a published agent the caller can
    see through the canonical accessible-agent query. The workforce-manager
    exemption that path also has is deliberately not included: callers use
    this to re-check access that was granted earlier and not re-verified
    since, which is exactly what that exemption skips.

    Hook failures (team scope, visibility policy) propagate; they are never
    read as access. Workforce-generated manager agents are not excluded here
    (the direct-run loader rejects them outside their own workforce); the
    caller's agent query is expected to have filtered them out already, as
    the delegation query does.
    """
    if owns_agent(agent, int(user_id), get_agent_team_scope(db, int(user_id))):
        return True
    if not _is_published(agent):
        return False
    user = db.query(User).filter(User.id == int(user_id)).first()
    if user is None:
        return False
    return (
        query_accessible_agents(db, user, purpose="agent_list")
        .filter(Agent.id == int(agent.id))
        .first()
        is not None
    )
