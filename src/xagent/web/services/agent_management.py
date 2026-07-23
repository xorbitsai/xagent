"""Reusable agent management operations for web, SDK, and SaaS adapters."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal, cast

from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, aliased

from ...core.tools.core.document_search import find_missing_knowledge_bases
from ...templates.manager import TemplateManager
from ..models.agent import Agent
from ..models.model import Model as DBModel
from ..models.user import User
from ..models.workforce import Workforce, WorkforceAgent, WorkforceRun
from ..schemas.agent_api_key import APIKeyGenerateResponse
from ..services.agent_store import AgentStore, invalidate_agent_cache
from .api_keys import AgentApiKeyService, KeyRotationConflict
from .workforce_access import can_edit_workforce, filter_visible_workforces
from .workforce_lifecycle import is_workforce_manager_discard_safe

logger = logging.getLogger(__name__)

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


@dataclass(frozen=True)
class AgentWorkforceReference:
    workforce_id: int
    name: str
    status: str
    roles: tuple[Literal["manager", "worker"], ...]
    can_edit: bool
    can_discard: bool


@dataclass(frozen=True)
class _AgentWorkforceReferenceSnapshot:
    workforce_id: int
    roles: tuple[Literal["manager", "worker"], ...]
    is_visible: bool


@dataclass(frozen=True)
class AgentDeleteResult:
    logo_url: str | None


class AgentWorkforceConflictError(RuntimeError):
    """Raised when a Workforce FK prevents deletion of an Agent."""

    def __init__(
        self,
        references: tuple[AgentWorkforceReference, ...],
        *,
        has_hidden_references: bool,
    ) -> None:
        if not references and not has_hidden_references:
            raise ValueError("Workforce conflict requires blocker evidence.")
        if any(not reference.roles for reference in references):
            raise ValueError("Visible Workforce references require at least one role.")
        super().__init__("Agent is used by one or more workforces.")
        self.references = references
        self.has_hidden_references = has_hidden_references


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

    def _workforce_reference_snapshot(
        self,
        *,
        actor: User,
        agent_id: int,
    ) -> tuple[_AgentWorkforceReferenceSnapshot, ...]:
        """Capture blocker roles and policy-owned visibility in one statement."""
        blocker_workforce = aliased(Workforce)
        manager_reference = blocker_workforce.manager_agent_id == agent_id
        worker_reference = (
            select(WorkforceAgent.id)
            .where(
                WorkforceAgent.workforce_id == blocker_workforce.id,
                WorkforceAgent.agent_id == agent_id,
            )
            .exists()
        )
        visible_reference = (
            filter_visible_workforces(
                self.db,
                actor,
                self.db.query(Workforce),
            )
            .filter(Workforce.id == blocker_workforce.id)
            .exists()
            .correlate(blocker_workforce)
        )
        rows = (
            self.db.query(
                blocker_workforce.id,
                manager_reference.label("is_manager_reference"),
                worker_reference.label("is_worker_reference"),
                visible_reference.label("is_visible"),
            )
            .filter(or_(manager_reference, worker_reference))
            .order_by(blocker_workforce.id)
            .all()
        )

        snapshot: list[_AgentWorkforceReferenceSnapshot] = []
        for workforce_id, is_manager, is_worker, is_visible in rows:
            roles: list[Literal["manager", "worker"]] = []
            if is_manager:
                roles.append("manager")
            if is_worker:
                roles.append("worker")
            snapshot.append(
                _AgentWorkforceReferenceSnapshot(
                    workforce_id=int(workforce_id),
                    roles=tuple(roles),
                    is_visible=bool(is_visible),
                )
            )
        return tuple(snapshot)

    def _visible_workforce_references(
        self,
        *,
        actor: User,
        snapshot: tuple[_AgentWorkforceReferenceSnapshot, ...],
    ) -> tuple[AgentWorkforceReference, ...]:
        snapshot_by_id = {
            reference.workforce_id: reference
            for reference in snapshot
            if reference.is_visible
        }
        snapshot_ids = tuple(snapshot_by_id)
        if not snapshot_ids:
            return ()
        visible_rows = (
            self.db.query(Workforce).filter(Workforce.id.in_(snapshot_ids)).all()
        )
        workforces_by_id = {int(workforce.id): workforce for workforce in visible_rows}
        workforce_ids = tuple(sorted(workforces_by_id))
        if not workforce_ids:
            return ()

        run_counts = {
            int(workforce_id): int(count)
            for workforce_id, count in (
                self.db.query(WorkforceRun.workforce_id, func.count(WorkforceRun.id))
                .filter(WorkforceRun.workforce_id.in_(workforce_ids))
                .group_by(WorkforceRun.workforce_id)
                .all()
            )
        }
        manager_ids = tuple(
            {int(workforce.manager_agent_id) for workforce in visible_rows}
        )
        managers_by_id = {
            int(manager.id): manager
            for manager in self.db.query(Agent).filter(Agent.id.in_(manager_ids)).all()
        }
        manager_reference_counts = {
            int(manager_id): int(count)
            for manager_id, count in (
                self.db.query(Workforce.manager_agent_id, func.count(Workforce.id))
                .filter(Workforce.manager_agent_id.in_(manager_ids))
                .group_by(Workforce.manager_agent_id)
                .all()
            )
        }
        managers_used_as_workers = {
            int(manager_id)
            for (manager_id,) in (
                self.db.query(WorkforceAgent.agent_id)
                .filter(WorkforceAgent.agent_id.in_(manager_ids))
                .distinct()
                .all()
            )
        }

        references: list[AgentWorkforceReference] = []
        for workforce_id in workforce_ids:
            workforce = workforces_by_id[workforce_id]
            status = str(workforce.status)
            can_edit = bool(
                status != "archived" and can_edit_workforce(self.db, actor, workforce)
            )
            manager_id = int(workforce.manager_agent_id)
            manager_discard_safe = is_workforce_manager_discard_safe(
                workforce,
                managers_by_id.get(manager_id),
                used_as_other_manager=manager_reference_counts.get(manager_id, 0) > 1,
                used_as_worker=manager_id in managers_used_as_workers,
            )
            references.append(
                AgentWorkforceReference(
                    workforce_id=workforce_id,
                    name=str(workforce.name),
                    status=status,
                    roles=snapshot_by_id[workforce_id].roles,
                    can_edit=can_edit,
                    can_discard=bool(
                        can_edit
                        and status == "draft"
                        and run_counts.get(workforce_id, 0) == 0
                        and manager_discard_safe
                    ),
                )
            )
        return tuple(references)

    def _workforce_conflict(
        self, *, actor: User, agent_id: int
    ) -> AgentWorkforceConflictError | None:
        snapshot = self._workforce_reference_snapshot(actor=actor, agent_id=agent_id)
        if not snapshot:
            return None
        references = self._visible_workforce_references(
            actor=actor,
            snapshot=snapshot,
        )
        has_hidden_references = any(not reference.is_visible for reference in snapshot)
        if not references and not has_hidden_references:
            return None
        return AgentWorkforceConflictError(
            references,
            has_hidden_references=has_hidden_references,
        )

    def delete_agent(
        self,
        *,
        actor: User,
        agent_id: int,
    ) -> AgentDeleteResult | None:
        """Delete an owned Agent unless any Workforce still references it."""
        actor_user_id = int(actor.id)
        agent = self.store.get_owned_agent(
            actor_user_id,
            agent_id,
            for_update=True,
        )
        if agent is None:
            return None
        conflict = self._workforce_conflict(actor=actor, agent_id=agent_id)
        if conflict is not None:
            raise conflict

        logo_url = cast("str | None", agent.logo_url)
        agent_owner_user_id = int(agent.user_id)
        agent_team_id = cast("int | None", agent.team_id)
        try:
            self.store.stage_delete_agent(agent)
            self.db.commit()
        except IntegrityError:
            self.db.rollback()
            conflict = self._workforce_conflict(actor=actor, agent_id=agent_id)
            if conflict is not None:
                raise conflict from None
            raise
        except Exception:
            self.db.rollback()
            raise

        try:
            invalidate_agent_cache(
                agent_owner_user_id,
                agent_id,
                agent_team_id,
            )
        except Exception:
            logger.warning(
                "Failed to invalidate cache after deleting agent %s",
                agent_id,
                exc_info=True,
            )
        return AgentDeleteResult(logo_url=logo_url)

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
