"""Agent DB/cache boundary for web and builder write paths."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, cast

from sqlalchemy.orm import Session

from ...core.tools.adapters.vibe.base import AGENT_CONFIG_UNASSIGNABLE_CATEGORIES
from ...core.utils.type_check import ensure_list
from ..models.agent import Agent, AgentOrigin, AgentStatus
from ..models.agent_api_key import AgentApiKey
from ..models.task import Task
from .agent_team_scope import (
    get_agent_team_scope,
    owned_agent_clause,
    team_id_of,
    validate_team_agent_connectors,
    validate_team_agent_knowledge_bases,
)

# Single source of truth for the widget embed credential; re-exported here so
# existing ``from ..services.agent_store import new_widget_key`` callers keep
# working (the agent and workforce channels mint the same kind of key).
from .deployments import new_widget_key
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


class UnsharedConnectorsError(Exception):
    """A team agent selects connectors not shared with or resolvable by its team.

    Carries the offending connectors so the API layer can surface them
    (mapped to HTTP 422) and the frontend can prompt the user to share or
    correct the references.
    """

    def __init__(self, connectors: list) -> None:
        self.connectors = connectors
        super().__init__(
            "agent selects connectors not shared with or resolvable by the team"
        )


class UnsharedKnowledgeBasesError(Exception):
    """A team agent selects knowledge bases not owned by its team."""

    def __init__(self, knowledge_bases: list) -> None:
        self.knowledge_bases = knowledge_bases
        super().__init__("agent selects knowledge bases not shared with the team")


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


def normalize_tool_categories(categories: Any) -> list[str] | None:
    """Drop non-assignable categories while preserving ``None``.

    ``None`` and ``[]`` are distinct at runtime
    (:meth:`ToolSelectionSpec.from_raw`): ``None`` is the legacy
    "unconfigured" value that keeps the full default tool set, while
    ``[]`` means the caller explicitly selected zero tools. Write paths
    must use this helper so an omitted selection is not persisted as
    "zero tools" (issue #944).
    """
    parsed = ensure_list(categories)
    if parsed is None:
        return None
    return [c for c in parsed if c not in AGENT_CONFIG_UNASSIGNABLE_CATEGORIES]


def clean_tool_categories(categories: Any) -> list[str]:
    """``normalize_tool_categories`` coerced for response payloads.

    Response schemas type ``tool_categories`` as a non-optional list,
    so ``None`` renders as ``[]`` here. Never use this on a write path.
    """
    return normalize_tool_categories(categories) or []


class AgentStore:
    """Owns Agent reads/writes that participate in hot-path cache policy."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def agent_to_response_dict(self, agent: Agent) -> dict[str, Any]:
        return {
            "id": agent.id,
            "user_id": agent.user_id,
            "team_id": agent.team_id,
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
            "team_id": agent.team_id,
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
        self,
        user_id: int,
        agent_id: int,
        team_scope: Any = _UNRESOLVED,
        *,
        for_update: bool = False,
    ) -> Agent | None:
        if team_scope is _UNRESOLVED:
            team_scope = get_agent_team_scope(self.db, user_id)
        query = self.db.query(Agent).filter(
            Agent.id == agent_id,
            owned_agent_clause(user_id, team_scope),
            Agent.origin != AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
        )
        if for_update:
            query = query.with_for_update()
        return query.first()

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
        # Agents are created personal (team_id NULL); promotion is explicit.
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
        if visibility is not None and visibility not in _VALID_VISIBILITIES:
            raise ValueError(f"Unsupported agent visibility: {visibility}")
        widget_key = new_widget_key() if widget_enabled else None
        # Agents are created personal (team_id NULL). Team ownership is granted
        # only by an explicit promote (see ``promote_agent_to_team``); a create
        # never stamps the caller's team. ``visibility`` is stored but only
        # takes effect once the agent is a team agent.
        agent = Agent(
            user_id=user_id,
            team_id=None,
            name=name,
            description=description,
            instructions=instructions,
            execution_mode=execution_mode or "graph",
            models=models,
            knowledge_bases=knowledge_bases or [],
            skills=skills or [],
            tool_categories=normalize_tool_categories(tool_categories),
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

        # A team agent (team_id set) may only select connectors already shared
        # with its team. Re-validate when its connector selection changes.
        if agent.team_id is not None and "tool_categories" in updates:
            unshared = validate_team_agent_connectors(
                self.db,
                user_id,
                int(agent.team_id),
                clean_tool_categories(updates.get("tool_categories")),
            )
            if unshared:
                raise UnsharedConnectorsError(unshared)

        if agent.team_id is not None and "knowledge_bases" in updates:
            unshared_kbs = validate_team_agent_knowledge_bases(
                self.db,
                user_id,
                int(agent.team_id),
                ensure_list(updates.get("knowledge_bases")) or [],
            )
            if unshared_kbs:
                raise UnsharedKnowledgeBasesError(unshared_kbs)

        unknown_fields = set(updates) - _AGENT_UPDATE_FIELDS
        if unknown_fields:
            raise ValueError(
                f"Unsupported agent update fields: {', '.join(sorted(unknown_fields))}"
            )

        for field, value in updates.items():
            if field == "tool_categories":
                value = normalize_tool_categories(value)
            setattr(agent, field, value)

        self.db.commit()
        self.db.refresh(agent)
        invalidate_agent_cache(user_id, agent_id, team_id_of(team_scope))
        return agent

    def promote_agent_to_team(
        self,
        user_id: int,
        agent_id: int,
        scope: Any,
        *,
        visibility: str = "team",
    ) -> Agent | None:
        """Make a personal agent team-owned, gated on shared connectors.

        Validates that every connector the agent selects is already shared
        with ``scope.team_id`` (raising :class:`UnsharedConnectorsError` with
        the offenders otherwise), then stamps ``team_id`` + ``visibility``.
        """
        team_scope = get_agent_team_scope(self.db, user_id)
        agent = self.get_owned_agent(user_id, agent_id, team_scope)
        if agent is None:
            return None
        _assert_can_set_visibility(
            scope, visibility, cast("str | None", agent.visibility)
        )
        if (
            agent.team_id is not None
            and int(agent.team_id) == int(scope.team_id)
            and str(agent.visibility) == visibility
        ):
            return agent
        unshared = validate_team_agent_connectors(
            self.db,
            user_id,
            int(scope.team_id),
            clean_tool_categories(agent.tool_categories),
        )
        if unshared:
            raise UnsharedConnectorsError(unshared)
        unshared_kbs = validate_team_agent_knowledge_bases(
            self.db,
            user_id,
            int(scope.team_id),
            ensure_list(agent.knowledge_bases) or [],
        )
        if unshared_kbs:
            raise UnsharedKnowledgeBasesError(unshared_kbs)
        agent.team_id = scope.team_id
        agent.visibility = visibility  # type: ignore[assignment]
        self.db.commit()
        self.db.refresh(agent)
        invalidate_agent_cache(user_id, agent_id, team_id_of(team_scope))
        return agent

    def demote_agent_to_personal(self, user_id: int, agent_id: int) -> Agent | None:
        """Revert a team agent to personal (team_id NULL)."""
        team_scope = get_agent_team_scope(self.db, user_id)
        agent = self.get_owned_agent(user_id, agent_id, team_scope)
        if agent is None:
            return None
        if (
            team_scope is not None
            and agent.team_id is not None
            and not team_scope.is_team_admin
            and int(agent.user_id) != int(user_id)
        ):
            raise PermissionError(
                "Only a team admin or the agent creator can make it personal"
            )
        agent.team_id = None  # type: ignore[assignment]
        # Reset to the default so a restrictive value (e.g. "admins") cannot
        # survive on a personal agent, where the visibility control is hidden.
        agent.visibility = "team"  # type: ignore[assignment]
        self.db.commit()
        self.db.refresh(agent)
        invalidate_agent_cache(user_id, agent_id, team_id_of(team_scope))
        return agent

    def stage_delete_agent(self, agent: Agent) -> None:
        """Stage the Agent aggregate deletion without committing or caching."""
        agent_id = int(agent.id)
        self.db.query(AgentApiKey).filter(AgentApiKey.agent_id == agent_id).delete(
            synchronize_session=False
        )
        self.db.query(Task).filter(Task.agent_id == agent_id).update(
            {Task.agent_id: None}, synchronize_session=False
        )
        self.db.delete(agent)
        self.db.flush()

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
