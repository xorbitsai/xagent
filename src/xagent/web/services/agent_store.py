"""Agent DB/cache boundary for web and builder write paths."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from ...core.utils.type_check import ensure_list
from ..models.agent import Agent, AgentStatus
from ..models.agent_api_key import AgentApiKey
from ..models.task import Task
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
}


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
            "tool_categories": ensure_list(agent.tool_categories) or [],
            "suggested_prompts": ensure_list(agent.suggested_prompts) or [],
            "logo_url": agent.logo_url,
            "status": agent.status.value,
            "published_at": agent.published_at.isoformat()
            if agent.published_at
            else None,
            "created_at": agent.created_at.isoformat(),
            "updated_at": agent.updated_at.isoformat(),
            "widget_enabled": agent.widget_enabled,
            "allowed_domains": ensure_list(agent.allowed_domains) or [],
        }

    def agent_to_list_item_dict(self, agent: Agent) -> dict[str, Any]:
        return {
            "id": agent.id,
            "name": agent.name,
            "description": agent.description,
            "logo_url": agent.logo_url,
            "status": agent.status.value,
            "created_at": agent.created_at.isoformat(),
            "updated_at": agent.updated_at.isoformat()
            if agent.updated_at
            else agent.created_at.isoformat(),
            "widget_enabled": agent.widget_enabled,
            "allowed_domains": ensure_list(agent.allowed_domains) or [],
        }

    def list_agent_items(self, user_id: int) -> list[dict[str, Any]]:
        cache_key = agent_list_key(user_id)
        cached = cache_get(cache_key)
        if isinstance(cached, list):
            return cached

        agents = (
            self.db.query(Agent)
            .filter(Agent.user_id == user_id)
            .order_by(Agent.created_at.desc())
            .all()
        )
        response = [self.agent_to_list_item_dict(agent) for agent in agents]
        cache_set(cache_key, response)
        return response

    def get_agent_response(self, user_id: int, agent_id: int) -> dict[str, Any] | None:
        cache_key = agent_detail_key(user_id, agent_id)
        cached = cache_get(cache_key)
        if isinstance(cached, dict):
            return cached

        agent = self.get_owned_agent(user_id, agent_id)
        if agent is None:
            return None

        response = self.agent_to_response_dict(agent)
        cache_set(cache_key, response)
        return response

    def get_owned_agent(self, user_id: int, agent_id: int) -> Agent | None:
        return (
            self.db.query(Agent)
            .filter(Agent.id == agent_id, Agent.user_id == user_id)
            .first()
        )

    def agent_name_exists(
        self, user_id: int, name: str, *, exclude_agent_id: int | None = None
    ) -> bool:
        query = self.db.query(Agent.id).filter(
            Agent.user_id == user_id, Agent.name == name
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
        status: AgentStatus = AgentStatus.DRAFT,
        widget_enabled: bool = True,
        allowed_domains: list[str] | None = None,
    ) -> Agent:
        agent = Agent(
            user_id=user_id,
            name=name,
            description=description,
            instructions=instructions,
            execution_mode=execution_mode or "graph",
            models=models,
            knowledge_bases=knowledge_bases or [],
            skills=skills or [],
            tool_categories=tool_categories or [],
            suggested_prompts=suggested_prompts or [],
            status=status,
            widget_enabled=widget_enabled,
            allowed_domains=allowed_domains or [],
        )
        self.db.add(agent)
        self.db.commit()
        self.db.refresh(agent)
        invalidate_agent_cache(user_id, int(agent.id))
        return agent

    def update_agent_fields(
        self, user_id: int, agent_id: int, updates: dict[str, Any]
    ) -> Agent | None:
        agent = self.get_owned_agent(user_id, agent_id)
        if agent is None:
            return None

        unknown_fields = set(updates) - _AGENT_UPDATE_FIELDS
        if unknown_fields:
            raise ValueError(
                f"Unsupported agent update fields: {', '.join(sorted(unknown_fields))}"
            )

        for field, value in updates.items():
            setattr(agent, field, value)

        self.db.commit()
        self.db.refresh(agent)
        invalidate_agent_cache(user_id, agent_id)
        return agent

    def delete_agent(self, user_id: int, agent_id: int) -> Agent | None:
        agent = self.get_owned_agent(user_id, agent_id)
        if agent is None:
            return None

        self.db.query(AgentApiKey).filter(AgentApiKey.agent_id == agent_id).delete()
        self.db.query(Task).filter(Task.agent_id == agent_id).update(
            {Task.agent_id: None}
        )
        self.db.delete(agent)
        self.db.commit()
        invalidate_agent_cache(user_id, agent_id)
        return agent

    def publish_agent(self, user_id: int, agent_id: int) -> Agent | None:
        agent = self.get_owned_agent(user_id, agent_id)
        if agent is None:
            return None
        if agent.status == AgentStatus.PUBLISHED:
            return agent

        agent.status = AgentStatus.PUBLISHED
        agent.published_at = datetime.now(timezone.utc)  # type: ignore[assignment]
        self.db.commit()
        self.db.refresh(agent)
        invalidate_agent_cache(user_id, agent_id)
        return agent

    def unpublish_agent(self, user_id: int, agent_id: int) -> Agent | None:
        agent = self.get_owned_agent(user_id, agent_id)
        if agent is None:
            return None
        if agent.status != AgentStatus.PUBLISHED:
            return agent

        agent.status = AgentStatus.DRAFT
        agent.published_at = None  # type: ignore[assignment]
        self.db.commit()
        self.db.refresh(agent)
        invalidate_agent_cache(user_id, agent_id)
        return agent
