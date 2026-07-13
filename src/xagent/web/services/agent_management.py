"""Reusable agent management operations for web, SDK, and SaaS adapters."""

from __future__ import annotations

from typing import Any, cast

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ...core.tools.core.document_search import find_missing_knowledge_bases
from ...templates.manager import TemplateManager
from ..models.agent import Agent
from ..models.model import Model as DBModel
from ..schemas.agent_api_key import APIKeyGenerateResponse
from ..services.agent_store import AgentStore, invalidate_agent_cache
from .api_keys import AgentApiKeyService, KeyRotationConflict

# Agent-builder tool category that gates knowledge-base access. A KB
# selection is only valid when this category is also enabled.
KNOWLEDGE_TOOL_CATEGORY = "knowledge"


class DuplicateAgentNameError(ValueError):
    """Raised when a user already owns an agent with the requested name."""


class TemplateNotFoundError(LookupError):
    """Raised when a template id cannot be resolved."""


class InvalidAgentModelConfigError(ValueError):
    """Raised when the agent model slot payload does not match DB id shape."""


class InvalidKnowledgeBaseError(ValueError):
    """Raised when KB selection fails the knowledge-tool or visibility rule."""


class AgentManagementService:
    """High-level user-owned agent management workflow boundary."""

    MODEL_SLOTS = frozenset({"general", "small_fast", "visual", "compact"})

    def __init__(self, db: Session, template_manager: TemplateManager | None = None):
        self.db = db
        self.store = AgentStore(db)
        self.template_manager = template_manager
        self.key_service = AgentApiKeyService(db)

    def list_agents_for_user(self, user_id: int) -> list[dict[str, Any]]:
        return self.store.list_agent_items(user_id)

    async def validate_knowledge_bases(
        self,
        *,
        knowledge_bases: list[str] | None,
        tool_categories: list[str] | None,
        user_id: int,
        is_admin: bool,
    ) -> None:
        """Enforce the knowledge-base invariant shared with ``/api/agents``.

        A non-empty KB selection requires the ``knowledge`` tool category
        and every named KB must be visible to the user. Raises
        :class:`InvalidKnowledgeBaseError` on either violation. This is
        async (KB visibility is an I/O lookup), so it lives on the async
        :meth:`create_agent` entry point rather than the sync transaction
        executor.
        """
        if not knowledge_bases:
            return
        if KNOWLEDGE_TOOL_CATEGORY not in (tool_categories or []):
            raise InvalidKnowledgeBaseError(
                "Knowledge bases are selected but the Knowledge tool "
                "category is not enabled."
            )
        missing = await find_missing_knowledge_bases(
            knowledge_bases, user_id=user_id, is_admin=is_admin
        )
        if missing:
            raise InvalidKnowledgeBaseError(
                "Knowledge base(s) not found or not visible to this user: "
                + ", ".join(missing)
            )

    async def create_agent(
        self,
        *,
        user_id: int,
        is_admin: bool,
        name: str,
        description: str | None,
        instructions: str | None,
        execution_mode: str | None = "balanced",
        models: dict[str, Any] | None = None,
        knowledge_bases: list[str] | None = None,
        skills: list[str] | None = None,
        tool_categories: list[str] | None = None,
        suggested_prompts: list[str] | None = None,
        generate_runtime_key: bool = True,
    ) -> tuple[Agent, APIKeyGenerateResponse | None]:
        """Sole external create entry point: validate KBs (async) then
        run the transactional create. Every public create path
        (``POST /v1/agents`` and ``POST /v1/agents/from-template``) goes
        through here, so the KB invariant has a single enforcement point.
        """
        await self.validate_knowledge_bases(
            knowledge_bases=knowledge_bases,
            tool_categories=tool_categories,
            user_id=user_id,
            is_admin=is_admin,
        )
        return self.create_agent_with_optional_key(
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
            generate_runtime_key=generate_runtime_key,
        )

    def create_agent_with_optional_key(
        self,
        *,
        user_id: int,
        name: str,
        description: str | None,
        instructions: str | None,
        execution_mode: str | None = "balanced",
        models: dict[str, Any] | None = None,
        knowledge_bases: list[str] | None = None,
        skills: list[str] | None = None,
        tool_categories: list[str] | None = None,
        suggested_prompts: list[str] | None = None,
        generate_runtime_key: bool = True,
    ) -> tuple[Agent, APIKeyGenerateResponse | None]:
        """Create an agent and (optionally) its first runtime key in a
        single transaction. Internal transaction executor: assumes
        knowledge-base inputs were already validated by the async
        :meth:`create_agent` entry point; this method only validates
        models and owns the commit boundary.

        Committing the agent and its first key separately would leave a
        persisted agent behind if the key step fails, so a client retry
        would hit duplicate-name even though the create appeared to
        fail. This method stages both writes (flush, no commit) and
        commits once at the boundary, rolling back atomically on any
        failure.

        Conflict contract: the only IntegrityError this path can raise
        comes from the runtime key's ``key_prefix`` unique constraint
        (the ``uq_agent_api_keys_agent_active`` partial index that used
        to also live here was dropped for multi-key support -- an agent
        may hold more than one active key now); the agent table has no
        unique constraint and therefore does not contribute one. If a
        ``(user_id, name)`` unique constraint is ever added to agents,
        the conflict translation here must be split by source (agent ->
        duplicate-name 400, key -> 409).
        """
        if self.store.agent_name_exists(user_id, name):
            raise DuplicateAgentNameError(name)

        models = self._validate_models(models, user_id=user_id)

        agent = self.store.add_agent(  # flush, no commit
            user_id=user_id,
            name=name,
            description=description,
            instructions=instructions,
            execution_mode=execution_mode or "balanced",
            models=models,
            knowledge_bases=knowledge_bases or [],
            skills=skills or [],
            tool_categories=tool_categories or [],
            suggested_prompts=suggested_prompts or [],
        )

        # The runtime key is the only write that can raise IntegrityError
        # here (agent has no unique constraint), so the conflict-to-409
        # translation wraps key staging + commit together. See the
        # contract note in the docstring above.
        staged_key = None
        try:
            if generate_runtime_key:
                staged_key = self.key_service.stage_rotated_key(int(agent.id))
            self.db.commit()  # single transaction boundary for both writes
        except IntegrityError as exc:
            self.db.rollback()
            raise KeyRotationConflict(str(exc)) from exc

        self.db.refresh(agent)
        invalidate_agent_cache(
            user_id, int(agent.id), cast("int | None", agent.team_id)
        )

        key_resp: APIKeyGenerateResponse | None = None
        if staged_key is not None:
            new_row, full_key = staged_key
            self.db.refresh(new_row)
            key_resp = APIKeyGenerateResponse(
                full_key=full_key,
                key_prefix=new_row.key_prefix,
                created_at=new_row.created_at,
            )
        return agent, key_resp

    async def create_agent_from_template(
        self,
        *,
        user_id: int,
        is_admin: bool,
        template_id: str,
        name: str | None = None,
        description: str | None = None,
        instructions: str | None = None,
        execution_mode: str | None = None,
        models: dict[str, Any] | None = None,
        knowledge_bases: list[str] | None = None,
        skills: list[str] | None = None,
        tool_categories: list[str] | None = None,
        suggested_prompts: list[str] | None = None,
        generate_runtime_key: bool = True,
    ) -> tuple[Agent, APIKeyGenerateResponse | None]:
        """Resolve a template (async I/O) then create the agent through
        :meth:`create_agent`, so KB validation and the single commit
        boundary are shared with the plain create path.
        """
        if self.template_manager is None:
            raise TemplateNotFoundError(template_id)

        template = await self.template_manager.get_template(template_id)
        if template is None:
            raise TemplateNotFoundError(template_id)

        agent_config = template.get("agent_config") or {}
        final_name = name or template.get("name") or template_id
        final_description = description
        if final_description is None:
            descriptions = template.get("descriptions") or {}
            if isinstance(descriptions, dict):
                final_description = descriptions.get("en") or ""
            elif isinstance(descriptions, str):
                final_description = descriptions

        return await self.create_agent(
            user_id=user_id,
            is_admin=is_admin,
            generate_runtime_key=generate_runtime_key,
            name=final_name,
            description=final_description,
            instructions=(
                instructions
                if instructions is not None
                else agent_config.get("instructions")
            ),
            execution_mode=execution_mode or agent_config.get("execution_mode"),
            models=models if models is not None else agent_config.get("models"),
            knowledge_bases=(
                knowledge_bases
                if knowledge_bases is not None
                else agent_config.get("knowledge_bases") or []
            ),
            skills=skills if skills is not None else agent_config.get("skills") or [],
            tool_categories=(
                tool_categories
                if tool_categories is not None
                else agent_config.get("tool_categories") or []
            ),
            suggested_prompts=(
                suggested_prompts
                if suggested_prompts is not None
                else agent_config.get("suggested_prompts") or []
            ),
        )

    def generate_agent_runtime_key(
        self, *, user_id: int, agent_id: int
    ) -> APIKeyGenerateResponse | None:
        agent = self.store.get_owned_agent(user_id, agent_id)
        if agent is None:
            return None
        return self.key_service.rotate_key(agent_id)

    def _validate_models(
        self, models: dict[str, Any] | None, *, user_id: int
    ) -> dict[str, Any] | None:
        if models is None:
            return None

        from .model_service import _is_model_visible_to_user

        normalized: dict[str, Any] = {}
        for slot, model_id in models.items():
            if slot not in self.MODEL_SLOTS:
                raise InvalidAgentModelConfigError(slot)
            if model_id is None:
                normalized[slot] = None
                continue
            if isinstance(model_id, bool) or not isinstance(model_id, int):
                raise InvalidAgentModelConfigError(slot)
            exists = (
                self.db.query(DBModel.id)
                .filter(DBModel.id == model_id, DBModel.is_active.is_(True))
                .first()
            )
            if exists is None or not _is_model_visible_to_user(
                self.db, model_id, user_id
            ):
                raise InvalidAgentModelConfigError(slot)
            normalized[slot] = model_id
        return normalized
