"""Agent DB/cache boundary for web and builder write paths."""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import Any, cast

from sqlalchemy.orm import Session

from ...core.utils.type_check import ensure_list
from ..models.agent import Agent, AgentOrigin, AgentStatus
from ..models.agent_api_key import AgentApiKey
from ..models.task import Task
from .agent_team_scope import (
    get_agent_team_scope,
    owned_agent_clause,
    team_id_of,
)
from .hot_path_cache import (
    agent_detail_key,
    agent_list_key,
    cache_get,
    cache_set,
    invalidate_agent_cache,
)

_AGENT_UPDATE_FIELDS = {
    "name",
    "description",
    "instructions",
    "execution_mode",
    "models",
    "knowledge_bases",
    "skills",
    "tool_categories",
    "suggested_prompts",
    "logo_url",
    "status",
    "widget_enabled",
    "allowed_domains",
    "widget_key",
    "share_enabled",
    "share_token",
    "share_updated_at",
    "visibility",
}


# Sentinel so callers can pass an already-resolved scope (even a real ``None``)
# and skip the second team-scope lookup within a single write method.
_UNRESOLVED = object()


_VALID_VISIBILITIES = {"team", "admins"}


def _assert_can_set_visibility(
    team_scope: Any,
    visibility: str | None,
    current_visibility: str | None = None,
) -> None:
    """Guard visibility writes: value-domain (#14) + admins-only gate (#8/#9).

    - Unknown values raise ``ValueError`` (store-layer domain check, not just
      the API Pydantic Literal).
    - Setting ``admins`` requires a resolved team scope with admin rights;
      no scope (standalone / legacy) is fail-closed, not fail-open (#8).
    - A team admin is also required to change an agent that is *currently*
      admins-only, so a non-admin cannot downgrade it (#9).
    """
    is_team_admin = bool(team_scope and getattr(team_scope, "is_team_admin", False))
    if visibility is not None and visibility not in _VALID_VISIBILITIES:
        raise ValueError(f"Unsupported agent visibility: {visibility}")
    if visibility == "admins" and not is_team_admin:
        raise PermissionError(
            "Only team admins can set an agent to admins-only visibility"
        )
    if visibility is not None and current_visibility == "admins" and not is_team_admin:
        raise PermissionError(
            "Only team admins can change the visibility of an admins-only agent"
        )


def clean_tool_categories(categories: Any) -> list[str]:
    return [c for c in (ensure_list(categories) or []) if c != "other"]


def new_widget_key() -> str:
    """Generate an unguessable widget embed credential for an agent."""
    return secrets.token_urlsafe(32)


class AgentStore:
    """Owns Agent reads/writes that participate in hot-path cache policy."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def agent_to_response_dict(self, agent: Agent) -> dict[str, Any]:
        return {
            "id": agent.id,
            "user_id": agent.user_id,
            "name": agent.name,
            "description": agent.description,
            "instructions": agent.instructions,
            "execution_mode": agent.execution_mode or "graph",
            "models": agent.models,
            "knowledge_bases": ensure_list(agent.knowledge_bases) or [],
            "skills": ensure_list(agent.skills) or [],
            "tool_categories": clean_tool_categories(agent.tool_categories),
            "suggested_prompts": ensure_list(agent.suggested_prompts) or [],
            "logo_url": agent.logo_url,
            "status": agent.status.value,
            "visibility": agent.visibility,
            "published_at": agent.published_at.isoformat()
            if agent.published_at
            else None,
            "created_at": agent.created_at.isoformat(),
            "updated_at": agent.updated_at.isoformat(),
            "widget_enabled": agent.widget_enabled,
            "allowed_domains": ensure_list(agent.allowed_domains) or [],
            "share_enabled": agent.share_enabled,
            "share_updated_at": agent.share_updated_at.isoformat()
            if agent.share_updated_at
            else None,
        }

    def agent_to_list_item_dict(self, agent: Agent) -> dict[str, Any]:
        return {
            "id": agent.id,
            "name": agent.name,
            "description": agent.description,
            "logo_url": agent.logo_url,
            "status": agent.status.value,
            "visibility": agent.visibility,
            "created_at": agent.created_at.isoformat(),
            "updated_at": agent.updated_at.isoformat()
            if agent.updated_at
            else agent.created_at.isoformat(),
            "widget_enabled": agent.widget_enabled,
            "allowed_domains": ensure_list(agent.allowed_domains) or [],
            "share_enabled": agent.share_enabled,
            "share_updated_at": agent.share_updated_at.isoformat()
            if agent.share_updated_at
            else None,
        }

    def list_agent_items(self, user_id: int) -> list[dict[str, Any]]:
        team_scope = get_agent_team_scope(self.db, user_id)
        cache_key = agent_list_key(
            user_id,
            team_id_of(team_scope),
            bool(team_scope and team_scope.is_team_admin),
        )
        cached = cache_get(cache_key)
        if isinstance(cached, list):
            return cached

        agents = (
            self.db.query(Agent)
            .filter(owned_agent_clause(user_id, team_scope))
            .filter(Agent.origin != AgentOrigin.WORKFORCE_GENERATED_MANAGER.value)
            .order_by(Agent.created_at.desc())
            .all()
        )
        response = [self.agent_to_list_item_dict(agent) for agent in agents]
        cache_set(cache_key, response)
        return response

    def get_agent_response(self, user_id: int, agent_id: int) -> dict[str, Any] | None:
        team_scope = get_agent_team_scope(self.db, user_id)
        cache_key = agent_detail_key(
            user_id,
            agent_id,
            team_id_of(team_scope),
            bool(team_scope and team_scope.is_team_admin),
        )
        cached = cache_get(cache_key)
        if isinstance(cached, dict):
            return cached

        agent = self.get_owned_agent(user_id, agent_id, team_scope)
        if agent is None:
            return None

        response = self.agent_to_response_dict(agent)
        cache_set(cache_key, response)
        return response

    def get_owned_agent(
        self, user_id: int, agent_id: int, team_scope: Any = _UNRESOLVED
    ) -> Agent | None:
        if team_scope is _UNRESOLVED:
            team_scope = get_agent_team_scope(self.db, user_id)
        return (
            self.db.query(Agent)
            .filter(
                Agent.id == agent_id,
                owned_agent_clause(user_id, team_scope),
                Agent.origin != AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
            )
            .first()
        )

    def get_agent_response_for_admin(self, agent_id: int) -> dict[str, Any] | None:
        """Read any agent's detail regardless of owner (admin-only path).

        Deliberately bypasses the hot-path cache: the detail cache is keyed by
        (viewer_user_id, agent_id) and only invalidated for the owner on write,
        so caching an admin's cross-user read would go stale on owner edits.
        """
        agent = (
            self.db.query(Agent)
            .filter(
                Agent.id == agent_id,
                Agent.origin != AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
            )
            .first()
        )
        if agent is None:
            return None
        return self.agent_to_response_dict(agent)

    def agent_name_exists(
        self,
        user_id: int,
        name: str,
        *,
        exclude_agent_id: int | None = None,
        team_scope: Any = _UNRESOLVED,
    ) -> bool:
        if team_scope is _UNRESOLVED:
            team_scope = get_agent_team_scope(self.db, user_id)
        query = self.db.query(Agent.id).filter(
            owned_agent_clause(user_id, team_scope),
            Agent.name == name,
            Agent.origin != AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
        )
        if exclude_agent_id is not None:
            query = query.filter(Agent.id != exclude_agent_id)
        return query.first() is not None

    def create_agent(
        self,
        *,
        user_id: int,
        name: str,
        description: str | None,
        instructions: str | None,
        execution_mode: str | None = None,
        models: dict[str, Any] | None = None,
        knowledge_bases: list[str] | None = None,
        skills: list[str] | None = None,
        tool_categories: list[str] | None = None,
        suggested_prompts: list[str] | None = None,
        origin: str = AgentOrigin.USER.value,
        status: AgentStatus = AgentStatus.DRAFT,
        published_at: datetime | None = None,
        widget_enabled: bool = True,
        allowed_domains: list[str] | None = None,
        share_enabled: bool = False,
        share_token: str | None = None,
        share_updated_at: datetime | None = None,
        visibility: str | None = None,
    ) -> Agent:
        agent = self.add_agent(
            user_id=user_id,
            name=name,
            description=description,
            instructions=instructions,
            execution_mode=execution_mode,
            models=models,
            knowledge_bases=knowledge_bases,
            skills=skills,
            tool_categories=tool_categories,
            suggested_prompts=suggested_prompts,
            origin=origin,
            status=status,
            published_at=published_at,
            widget_enabled=widget_enabled,
            allowed_domains=allowed_domains,
            share_enabled=share_enabled,
            share_token=share_token,
            share_updated_at=share_updated_at,
            visibility=visibility,
        )
        self.db.commit()
        self.db.refresh(agent)
        # add_agent already stamped agent.team_id from the resolved scope.
        invalidate_agent_cache(
            user_id, int(agent.id), cast("int | None", agent.team_id)
        )
        return agent

    def add_agent(
        self,
        *,
        user_id: int,
        name: str,
        description: str | None,
        instructions: str | None,
        execution_mode: str | None = None,
        models: dict[str, Any] | None = None,
        knowledge_bases: list[str] | None = None,
        skills: list[str] | None = None,
        tool_categories: list[str] | None = None,
        suggested_prompts: list[str] | None = None,
        origin: str = AgentOrigin.USER.value,
        status: AgentStatus = AgentStatus.DRAFT,
        published_at: datetime | None = None,
        widget_enabled: bool = True,
        allowed_domains: list[str] | None = None,
        share_enabled: bool = False,
        share_token: str | None = None,
        share_updated_at: datetime | None = None,
        visibility: str | None = None,
    ) -> Agent:
        if status == AgentStatus.PUBLISHED and published_at is None:
            published_at = datetime.now(timezone.utc)
        # Widget-enabled agents always carry an embed credential; agents
        # created disabled get one when the widget is first enabled.
        team_scope = get_agent_team_scope(self.db, user_id)
        _assert_can_set_visibility(team_scope, visibility)
        widget_key = new_widget_key() if widget_enabled else None
        agent = Agent(
            user_id=user_id,
            team_id=team_id_of(team_scope),
            name=name,
            description=description,
            instructions=instructions,
            execution_mode=execution_mode or "graph",
            models=models,
            knowledge_bases=knowledge_bases or [],
            skills=skills or [],
            tool_categories=clean_tool_categories(tool_categories),
            suggested_prompts=suggested_prompts or [],
            origin=origin,
            status=status,
            published_at=published_at,
            widget_enabled=widget_enabled,
            allowed_domains=allowed_domains or [],
            widget_key=widget_key,
            share_enabled=share_enabled,
            share_token=share_token,
            share_updated_at=share_updated_at,
            visibility=visibility or "team",
        )
        self.db.add(agent)
        self.db.flush()
        return agent

    def update_agent_fields(
        self,
        user_id: int,
        agent_id: int,
        updates: dict[str, Any],
        *,
        team_scope: Any = _UNRESOLVED,
        agent: Agent | None = None,
    ) -> Agent | None:
        if team_scope is _UNRESOLVED:
            team_scope = get_agent_team_scope(self.db, user_id)
        if agent is None:
            agent = self.get_owned_agent(user_id, agent_id, team_scope)
        if agent is None:
            return None

        if "visibility" in updates:
            _assert_can_set_visibility(
                team_scope,
                updates.get("visibility"),
                cast("str | None", agent.visibility),
            )

        unknown_fields = set(updates) - _AGENT_UPDATE_FIELDS
        if unknown_fields:
            raise ValueError(
                f"Unsupported agent update fields: {', '.join(sorted(unknown_fields))}"
            )

        for field, value in updates.items():
            if field == "tool_categories":
                value = clean_tool_categories(value)
            setattr(agent, field, value)

        self.db.commit()
        self.db.refresh(agent)
        invalidate_agent_cache(user_id, agent_id, team_id_of(team_scope))
        return agent

    def delete_agent(self, user_id: int, agent_id: int) -> Agent | None:
        team_scope = get_agent_team_scope(self.db, user_id)
        agent = self.get_owned_agent(user_id, agent_id, team_scope)
        if agent is None:
            return None

        self.db.query(AgentApiKey).filter(AgentApiKey.agent_id == agent_id).delete()
        self.db.query(Task).filter(Task.agent_id == agent_id).update(
            {Task.agent_id: None}
        )
        self.db.delete(agent)
        self.db.commit()
        invalidate_agent_cache(user_id, agent_id, team_id_of(team_scope))
        return agent

    def publish_agent(self, user_id: int, agent_id: int) -> Agent | None:
        team_scope = get_agent_team_scope(self.db, user_id)
        agent = self.get_owned_agent(user_id, agent_id, team_scope)
        if agent is None:
            return None
        if agent.status == AgentStatus.PUBLISHED:
            return agent

        agent.status = AgentStatus.PUBLISHED
        agent.published_at = datetime.now(timezone.utc)  # type: ignore[assignment]
        self.db.commit()
        self.db.refresh(agent)
        invalidate_agent_cache(user_id, agent_id, team_id_of(team_scope))
        return agent

    def unpublish_agent(self, user_id: int, agent_id: int) -> Agent | None:
        team_scope = get_agent_team_scope(self.db, user_id)
        agent = self.get_owned_agent(user_id, agent_id, team_scope)
        if agent is None:
            return None
        if agent.status != AgentStatus.PUBLISHED:
            return agent

        agent.status = AgentStatus.DRAFT
        agent.published_at = None  # type: ignore[assignment]
        self.db.commit()
        self.db.refresh(agent)
        invalidate_agent_cache(user_id, agent_id, team_id_of(team_scope))
        return agent
