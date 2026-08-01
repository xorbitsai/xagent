"""Reusable agent management operations for web, SDK, and SaaS adapters."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from types import TracebackType
from typing import Any, Callable, Generic, Literal, TypeVar, cast

from sqlalchemy import func, or_, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, aliased

from ...core.tools.core.document_search import find_missing_knowledge_bases
from ...core.utils.api_key import (
    PREFIX_COLLISION_RETRIES,
    ApiKeyKind,
    generate_api_key,
)
from ...templates.manager import TemplateManager
from ..models.agent import (
    AGENT_NAME_UNIQUE_INDEX,
    AGENT_TEMPLATE_QUICK_ACCESS_UNIQUE_INDEX,
    Agent,
    AgentOrigin,
)
from ..models.agent_api_key import AgentApiKey
from ..models.database import get_session_local, release_db_connection_if_clean
from ..models.model import Model as DBModel
from ..models.user import User
from ..models.workforce import Workforce, WorkforceAgent, WorkforceRun
from ..schemas.agent_api_key import APIKeyGenerateResponse
from ..services.agent_store import AgentStore, invalidate_agent_cache
from .api_keys import (
    AgentApiKeyService,
    ApiKeyCandidate,
    KeyRotationConflict,
    RuntimeKeyDeliveryError,
    RuntimeKeyReceipt,
    acquire_runtime_key_transition_fence,
)
from .db_runtime import (
    await_task_settlement,
    is_process_control_exception,
    propagate_deferred_cancellation,
    run_db_io_cancellation_safe,
)
from .workforce_access import can_edit_workforce, filter_visible_workforces
from .workforce_lifecycle import is_workforce_manager_discard_safe
from .workforce_names import resolve_unique_agent_name

logger = logging.getLogger(__name__)
_RuntimeKeyResultT = TypeVar("_RuntimeKeyResultT")

# Agent-builder tool category that gates knowledge-base access. A KB
# selection is only valid when this category is also enabled.
KNOWLEDGE_TOOL_CATEGORY = "knowledge"

# Select-then-insert retries for resolve_agent_from_template: each retry
# re-selects, so a concurrent insert of this same template's agent converges
# to reuse and an unrelated name race picks a fresh candidate name.
TEMPLATE_RESOLVE_RACE_RETRIES = 3


class DuplicateAgentNameError(ValueError):
    """Raised when a user already owns an agent with the requested name."""


class TemplateQuickAccessRaceError(ValueError):
    """Raised when a concurrent insert wins the (user_id, template_id)
    quick-access race - see AGENT_TEMPLATE_QUICK_ACCESS_UNIQUE_INDEX and
    _resolve_agent_from_template_sync's retry loop."""


class TemplateNotFoundError(LookupError):
    """Raised when a template id cannot be resolved."""


class InvalidAgentModelConfigError(ValueError):
    """Raised when the agent model slot payload does not match DB id shape."""


class InvalidKnowledgeBaseError(ValueError):
    """Raised when KB selection fails the knowledge-tool or visibility rule."""


@dataclass(frozen=True)
class AgentCreateSpec:
    """Detached input for one agent-management create transaction."""

    name: str
    description: str | None
    instructions: str | None
    execution_mode: str | None
    models: tuple[tuple[str, int | None], ...] | None
    knowledge_bases: tuple[str, ...]
    skills: tuple[str, ...]
    tool_categories: tuple[str, ...]
    suggested_prompts: tuple[str, ...]
    generate_runtime_key: bool
    template_id: str | None = None

    @classmethod
    def from_values(
        cls,
        *,
        name: str,
        description: str | None,
        instructions: str | None,
        execution_mode: str | None,
        models: dict[str, Any] | None,
        knowledge_bases: list[str] | tuple[str, ...] | None,
        skills: list[str] | tuple[str, ...] | None,
        tool_categories: list[str] | tuple[str, ...] | None,
        suggested_prompts: list[str] | tuple[str, ...] | None,
        generate_runtime_key: bool,
        template_id: str | None = None,
    ) -> AgentCreateSpec:
        frozen_models: tuple[tuple[str, int | None], ...] | None = None
        if models is not None:
            frozen_models = tuple(
                (str(slot), cast("int | None", model_id))
                for slot, model_id in models.items()
            )
        return cls(
            name=name,
            description=description,
            instructions=instructions,
            execution_mode=execution_mode,
            models=frozen_models,
            knowledge_bases=tuple(knowledge_bases or ()),
            skills=tuple(skills or ()),
            tool_categories=tuple(tool_categories or ()),
            suggested_prompts=tuple(suggested_prompts or ()),
            template_id=template_id,
            generate_runtime_key=generate_runtime_key,
        )

    def model_mapping(self) -> dict[str, Any] | None:
        return dict(self.models) if self.models is not None else None


@dataclass(frozen=True)
class AgentSummarySnapshot:
    """Frozen V1 list item detached from its worker-owned Session."""

    id: int
    name: str
    description: str | None
    logo_url: str | None
    status: str
    created_at: str
    updated_at: str
    widget_enabled: bool
    allowed_domains: tuple[str, ...]
    share_enabled: bool
    share_updated_at: str | None


@dataclass(frozen=True)
class AgentResponseSnapshot:
    """Frozen agent detail detached from its worker-owned Session."""

    id: int
    user_id: int
    team_id: int | None
    name: str
    description: str | None
    instructions: str | None
    execution_mode: str
    models: tuple[tuple[str, int | None], ...] | None
    knowledge_bases: tuple[str, ...]
    skills: tuple[str, ...]
    tool_categories: tuple[str, ...]
    suggested_prompts: tuple[str, ...]
    logo_url: str | None
    status: str
    visibility: str
    published_at: str | None
    created_at: str
    updated_at: str
    widget_enabled: bool
    allowed_domains: tuple[str, ...]
    share_enabled: bool
    share_updated_at: str | None
    template_id: str | None = None

    def model_mapping(self) -> dict[str, Any] | None:
        return dict(self.models) if self.models is not None else None

    def to_response_dict(self) -> dict[str, Any]:
        """Return the shared detached payload consumed by web and V1 schemas."""

        return {
            "id": self.id,
            "user_id": self.user_id,
            "team_id": self.team_id,
            "name": self.name,
            "description": self.description,
            "instructions": self.instructions,
            "execution_mode": self.execution_mode,
            "models": self.model_mapping(),
            "knowledge_bases": list(self.knowledge_bases),
            "skills": list(self.skills),
            "tool_categories": list(self.tool_categories),
            "suggested_prompts": list(self.suggested_prompts),
            "logo_url": self.logo_url,
            "status": self.status,
            "visibility": self.visibility,
            "published_at": self.published_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "widget_enabled": self.widget_enabled,
            "allowed_domains": list(self.allowed_domains),
            "share_enabled": self.share_enabled,
            "share_updated_at": self.share_updated_at,
            "template_id": self.template_id,
        }


@dataclass(frozen=True)
class RuntimeKeySnapshot:
    """One-shot runtime key returned after its write transaction commits."""

    full_key: str
    key_prefix: str
    created_at: datetime


@dataclass(frozen=True)
class AgentCreateSnapshot:
    """Detached result of atomic agent + optional runtime-key creation."""

    agent: AgentResponseSnapshot
    api_key: RuntimeKeySnapshot | None


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


@dataclass(frozen=True)
class _RuntimeKeyDeliveryOutcome(Generic[_RuntimeKeyResultT]):
    """Detached worker result that retains an exact key receipt on failure."""

    result: _RuntimeKeyResultT | None
    receipt: RuntimeKeyReceipt | None
    error: BaseException | None
    traceback: TracebackType | None


@dataclass(frozen=True)
class _RuntimeKeyCompensationResult:
    """Auditable row counts from one fenced compensation transaction."""

    new_key_revoked: int
    prior_keys_restored: int


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
        self.runtime_key_receipt: RuntimeKeyReceipt | None = None

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
        await _validate_agent_knowledge_bases(
            knowledge_bases=tuple(knowledge_bases or ()),
            tool_categories=tuple(tool_categories or ()),
            user_id=user_id,
            is_admin=is_admin,
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
        template_id: str | None = None,
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
            template_id=template_id,
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
        runtime_key_candidate: ApiKeyCandidate | None = None,
        template_id: str | None = None,
        origin: str = AgentOrigin.USER.value,
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

        Conflict contract: ``agent_name_exists`` is a fast-path pre-check,
        not the source of truth -- the ``uq_agents_user_id_name_active``
        partial unique index (excludes workforce-generated-manager agents)
        is what actually prevents a same-user concurrent-request race, so
        the agent insert's flush can itself raise IntegrityError and is
        translated to ``DuplicateAgentNameError`` below. The index is keyed
        on ``(user_id, name)`` only, so it matches ``agent_name_exists``
        exactly in standalone xagent (no team-scope hook, where the check is
        also purely per-user). When a team-scope hook is installed,
        ``agent_name_exists`` widens to a team-wide check (any teammate's
        ``visibility="team"`` agent, or all of them for an admin) that this
        per-user index does not mirror -- two different users on the same
        team can still race past it with identical names; only a single
        user's own double-submit is guaranteed to be caught here. A second
        partial unique index, ``uq_agents_user_id_template_id_quick_access``
        (scoped to ``origin=template_quick_access``), similarly turns an
        insert into ``TemplateQuickAccessRaceError`` below -- see
        ``AgentManagementService._resolve_agent_from_template_sync``. The
        only remaining IntegrityError source is the runtime key's
        ``key_prefix`` unique constraint (the ``uq_agent_api_keys_agent_active``
        partial index that used to also live here was dropped for multi-key
        support -- an agent may hold more than one active key now), handled
        separately inside ``complete_runtime_key_delivery``.

        Delivery contract: once a runtime-key receipt exists, an ambiguous
        commit result or post-commit response failure is wrapped in
        :class:`RuntimeKeyDeliveryError`. The runtime boundary is the sole
        consumer and compensates the exact transition in a fresh Session.
        """
        self.runtime_key_receipt = None
        if self.store.agent_name_exists(user_id, name):
            raise DuplicateAgentNameError(name)

        models = self._validate_models(models, user_id=user_id)

        try:
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
                template_id=template_id,
                origin=origin,
            )
        except IntegrityError as exc:
            self.db.rollback()
            if is_agent_name_unique_violation(exc):
                raise DuplicateAgentNameError(name) from exc
            if is_agent_template_quick_access_unique_violation(exc):
                raise TemplateQuickAccessRaceError(template_id) from exc
            raise

        # The agent-name race is handled above via the add_agent flush; the
        # runtime key is the only remaining write that can raise
        # IntegrityError here, so its conflict-to-409 translation happens
        # inside complete_runtime_key_delivery. See the contract note in the
        # docstring above.
        def stage_runtime_key() -> tuple[AgentApiKey, str] | None:
            if generate_runtime_key:
                staged_key = self.key_service.stage_rotated_key(
                    int(agent.id),
                    candidate=runtime_key_candidate,
                )
                self.runtime_key_receipt = self.key_service.runtime_key_receipt
                return staged_key
            return None

        def build_response(
            staged_key: tuple[AgentApiKey, str] | None,
        ) -> tuple[Agent, APIKeyGenerateResponse | None]:
            key_resp: APIKeyGenerateResponse | None = None
            self.db.refresh(agent)
            agent_id = int(agent.id)
            agent_team_id = cast("int | None", agent.team_id)
            if staged_key is not None:
                new_row, full_key = staged_key
                self.db.refresh(new_row)
                key_resp = APIKeyGenerateResponse(
                    full_key=full_key,
                    key_prefix=new_row.key_prefix,
                    created_at=new_row.created_at,
                )
            # Preserve the fully-loaded response row as a detached object. The
            # read-only transaction release below expires attached ORM state.
            self.db.expunge(agent)
            # Refreshes above open a new read-only transaction. End it before
            # cache invalidation, which may perform synchronous remote I/O.
            release_db_connection_if_clean(self.db)
            try:
                invalidate_agent_cache(
                    user_id,
                    agent_id,
                    agent_team_id,
                )
            except Exception:
                logger.warning(
                    "Failed to invalidate cache after creating agent %s",
                    agent_id,
                    exc_info=True,
                )
            return agent, key_resp

        return self.key_service.complete_runtime_key_delivery(
            stage=stage_runtime_key,
            build_response=build_response,
        )

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
            template_id=template_id,
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
        self,
        *,
        user_id: int,
        agent_id: int,
        runtime_key_candidate: ApiKeyCandidate | None = None,
    ) -> APIKeyGenerateResponse | None:
        agent = self.store.get_owned_agent(user_id, agent_id)
        if agent is None:
            return None
        response = self.key_service.rotate_key_for_runtime_delivery(
            agent_id,
            candidate=runtime_key_candidate,
        )
        self.runtime_key_receipt = self.key_service.runtime_key_receipt
        return response

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


def _agent_summary_snapshot(payload: dict[str, Any]) -> AgentSummarySnapshot:
    return AgentSummarySnapshot(
        id=int(payload["id"]),
        name=str(payload["name"]),
        description=cast("str | None", payload.get("description")),
        logo_url=cast("str | None", payload.get("logo_url")),
        status=str(payload["status"]),
        created_at=str(payload["created_at"]),
        updated_at=str(payload["updated_at"]),
        widget_enabled=bool(payload["widget_enabled"]),
        allowed_domains=tuple(payload.get("allowed_domains") or ()),
        share_enabled=bool(payload["share_enabled"]),
        share_updated_at=cast("str | None", payload.get("share_updated_at")),
    )


def _agent_response_snapshot(payload: dict[str, Any]) -> AgentResponseSnapshot:
    raw_models = payload.get("models")
    frozen_models = (
        tuple(
            (str(slot), cast("int | None", model_id))
            for slot, model_id in raw_models.items()
        )
        if isinstance(raw_models, dict)
        else None
    )
    return AgentResponseSnapshot(
        id=int(payload["id"]),
        user_id=int(payload["user_id"]),
        team_id=(
            int(payload["team_id"]) if payload.get("team_id") is not None else None
        ),
        name=str(payload["name"]),
        description=cast("str | None", payload.get("description")),
        instructions=cast("str | None", payload.get("instructions")),
        execution_mode=str(payload["execution_mode"]),
        models=frozen_models,
        knowledge_bases=tuple(payload.get("knowledge_bases") or ()),
        skills=tuple(payload.get("skills") or ()),
        tool_categories=tuple(payload.get("tool_categories") or ()),
        suggested_prompts=tuple(payload.get("suggested_prompts") or ()),
        logo_url=cast("str | None", payload.get("logo_url")),
        status=str(payload["status"]),
        visibility=str(payload["visibility"]),
        published_at=cast("str | None", payload.get("published_at")),
        created_at=str(payload["created_at"]),
        updated_at=str(payload["updated_at"]),
        widget_enabled=bool(payload["widget_enabled"]),
        allowed_domains=tuple(payload.get("allowed_domains") or ()),
        share_enabled=bool(payload["share_enabled"]),
        share_updated_at=cast("str | None", payload.get("share_updated_at")),
        template_id=cast("str | None", payload.get("template_id")),
    )


def _runtime_key_snapshot(
    response: APIKeyGenerateResponse | None,
) -> RuntimeKeySnapshot | None:
    if response is None:
        return None
    return RuntimeKeySnapshot(
        full_key=response.full_key,
        key_prefix=response.key_prefix,
        created_at=response.created_at,
    )


def _is_runtime_key_prefix_collision(error: BaseException) -> bool:
    """Recognize the authoritative key-prefix unique constraint failure."""

    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).lower()
        if "key_prefix" in message and (
            "agent_api_keys" in message or "unique" in message or "duplicate" in message
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def is_agent_name_unique_violation(error: BaseException) -> bool:
    """Recognize the authoritative (user_id, name) unique index failure.

    Postgres includes the index name (``uq_agents_user_id_name_active``) in
    its error message; sqlite instead names the columns
    (``agents.user_id, agents.name``). Matching either keeps this from
    misclassifying an unrelated IntegrityError (e.g. a ``widget_key``
    collision or a foreign-key violation) as a duplicate-name conflict.
    """

    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).lower()
        if AGENT_NAME_UNIQUE_INDEX.lower() in message:
            return True
        if (
            "agents.user_id" in message
            and "agents.name" in message
            and ("unique" in message or "duplicate" in message)
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def is_agent_template_quick_access_unique_violation(error: BaseException) -> bool:
    """Recognize the authoritative (user_id, template_id) quick-access
    unique index failure - the counterpart to is_agent_name_unique_violation
    above, for AGENT_TEMPLATE_QUICK_ACCESS_UNIQUE_INDEX. Fires when a
    concurrent request wins the resolve flow's create race even when the two
    inserts picked different (disambiguated) names, so
    is_agent_name_unique_violation alone would miss it (PR review findings
    B1/B2).
    """

    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).lower()
        if AGENT_TEMPLATE_QUICK_ACCESS_UNIQUE_INDEX.lower() in message:
            return True
        if (
            "agents.user_id" in message
            and "agents.template_id" in message
            and ("unique" in message or "duplicate" in message)
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


async def _validate_agent_knowledge_bases(
    *,
    knowledge_bases: tuple[str, ...],
    tool_categories: tuple[str, ...],
    user_id: int,
    is_admin: bool,
) -> None:
    """Validate detached KB inputs without retaining a SQLAlchemy Session."""

    if not knowledge_bases:
        return
    if KNOWLEDGE_TOOL_CATEGORY not in tool_categories:
        raise InvalidKnowledgeBaseError(
            "Knowledge bases are selected but the Knowledge tool "
            "category is not enabled."
        )
    missing = await find_missing_knowledge_bases(
        list(knowledge_bases),
        user_id=user_id,
        is_admin=is_admin,
    )
    if missing:
        raise InvalidKnowledgeBaseError(
            "Knowledge base(s) not found or not visible to this user: "
            + ", ".join(missing)
        )


class AgentManagementRuntime:
    """Async owner for V1 agent management.

    The runtime itself owns no Session. Async template and knowledge-base
    materialization operate on detached values, while every synchronous SQL
    transaction runs in a worker that creates and closes its own Session.
    """

    def __init__(self, template_manager: TemplateManager | None = None) -> None:
        self.template_manager = template_manager

    async def list_agents(
        self,
        *,
        user_id: int,
    ) -> tuple[AgentSummarySnapshot, ...]:
        return await run_db_io_cancellation_safe(
            lambda: self._list_agents_sync(user_id=user_id)
        )

    @staticmethod
    def _list_agents_sync(*, user_id: int) -> tuple[AgentSummarySnapshot, ...]:
        SessionLocal = get_session_local()
        with SessionLocal() as db:
            payloads = AgentManagementService(db).list_agents_for_user(user_id)
            return tuple(_agent_summary_snapshot(payload) for payload in payloads)

    @staticmethod
    def _run_runtime_key_session_operation(
        operation: Callable[
            [AgentManagementService],
            _RuntimeKeyResultT | None,
        ],
    ) -> _RuntimeKeyDeliveryOutcome[_RuntimeKeyResultT]:
        """Detach runtime-key outcomes before closing their worker Session.

        A committed key receipt is needed even when the service reports a
        post-commit failure.  ``Session.__exit__`` can itself fail, so this
        boundary converts the service failure to detached data before close;
        an operational close failure cannot replace that earlier failure.
        """

        db: Session | None = None
        service: AgentManagementService | None = None
        outcome: _RuntimeKeyDeliveryOutcome[_RuntimeKeyResultT] | None = None
        try:
            SessionLocal = get_session_local()
            db = SessionLocal()
            service = AgentManagementService(db)
            try:
                result = operation(service)
                outcome = _RuntimeKeyDeliveryOutcome(
                    result=result,
                    receipt=service.runtime_key_receipt,
                    error=None,
                    traceback=None,
                )
            except RuntimeKeyDeliveryError as exc:
                outcome = _RuntimeKeyDeliveryOutcome(
                    result=None,
                    receipt=exc.receipt,
                    error=exc.error,
                    traceback=exc.traceback,
                )
            except BaseException as exc:
                outcome = _RuntimeKeyDeliveryOutcome(
                    result=None,
                    receipt=(
                        None
                        if isinstance(exc, KeyRotationConflict)
                        else service.runtime_key_receipt
                    ),
                    error=exc,
                    traceback=exc.__traceback__,
                )
        except BaseException as exc:
            outcome = _RuntimeKeyDeliveryOutcome(
                result=None,
                receipt=None,
                error=exc,
                traceback=exc.__traceback__,
            )
        finally:
            if db is not None:
                try:
                    db.close()
                except BaseException as close_error:
                    if outcome is None or outcome.error is None:
                        outcome = _RuntimeKeyDeliveryOutcome(
                            result=None,
                            receipt=None if outcome is None else outcome.receipt,
                            error=close_error,
                            traceback=close_error.__traceback__,
                        )
                    elif is_process_control_exception(close_error):
                        outcome = _RuntimeKeyDeliveryOutcome(
                            result=None,
                            receipt=outcome.receipt,
                            error=close_error,
                            traceback=close_error.__traceback__,
                        )
                    else:
                        logger.warning(
                            "Failed to close runtime-key worker session after a "
                            "delivery failure",
                            exc_info=(
                                type(close_error),
                                close_error,
                                close_error.__traceback__,
                            ),
                        )
        assert outcome is not None
        return outcome

    async def create_agent(
        self,
        *,
        user_id: int,
        is_admin: bool,
        spec: AgentCreateSpec,
    ) -> AgentCreateSnapshot:
        await _validate_agent_knowledge_bases(
            knowledge_bases=spec.knowledge_bases,
            tool_categories=spec.tool_categories,
            user_id=user_id,
            is_admin=is_admin,
        )
        result = await self._run_runtime_key_delivery(
            lambda: self._create_agent_with_retry_sync(
                user_id=user_id,
                spec=spec,
            )
        )
        assert result is not None
        return result

    @staticmethod
    def _create_agent_with_retry_sync(
        *,
        user_id: int,
        spec: AgentCreateSpec,
    ) -> _RuntimeKeyDeliveryOutcome[AgentCreateSnapshot]:
        attempts = PREFIX_COLLISION_RETRIES if spec.generate_runtime_key else 1
        for attempt in range(attempts):
            # Candidate generation includes bcrypt. It deliberately precedes
            # Session creation so CPU work never pins a pool checkout.
            candidate = (
                generate_api_key(None, kind=ApiKeyKind.AGENT)
                if spec.generate_runtime_key
                else None
            )

            def create_attempt(
                service: AgentManagementService,
                candidate: ApiKeyCandidate | None = candidate,
            ) -> AgentCreateSnapshot:
                agent, key_response = service.create_agent_with_optional_key(
                    user_id=user_id,
                    name=spec.name,
                    description=spec.description,
                    instructions=spec.instructions,
                    execution_mode=spec.execution_mode,
                    models=spec.model_mapping(),
                    knowledge_bases=list(spec.knowledge_bases),
                    skills=list(spec.skills),
                    tool_categories=list(spec.tool_categories),
                    suggested_prompts=list(spec.suggested_prompts),
                    generate_runtime_key=spec.generate_runtime_key,
                    runtime_key_candidate=candidate,
                    template_id=spec.template_id,
                )
                agent_snapshot = _agent_response_snapshot(
                    service.store.agent_to_response_dict(agent)
                )
                return AgentCreateSnapshot(
                    agent=agent_snapshot,
                    api_key=_runtime_key_snapshot(key_response),
                )

            outcome = AgentManagementRuntime._run_runtime_key_session_operation(
                create_attempt
            )
            if (
                isinstance(outcome.error, KeyRotationConflict)
                and candidate is not None
                and _is_runtime_key_prefix_collision(outcome.error)
                and attempt + 1 < attempts
            ):
                continue
            return outcome
        error = KeyRotationConflict(
            "Failed to generate a unique runtime key prefix after retrying."
        )
        return _RuntimeKeyDeliveryOutcome(
            result=None,
            receipt=None,
            error=error,
            traceback=error.__traceback__,
        )

    async def _spec_from_template(
        self,
        *,
        template_id: str,
        name: str | None,
        description: str | None,
        instructions: str | None,
        execution_mode: str | None,
        models: dict[str, Any] | None,
        knowledge_bases: list[str] | None,
        skills: list[str] | None,
        tool_categories: list[str] | None,
        suggested_prompts: list[str] | None,
        generate_runtime_key: bool,
    ) -> AgentCreateSpec:
        """Resolve a template (async I/O) into a detached create spec."""
        if self.template_manager is None:
            raise TemplateNotFoundError(template_id)
        template = await self.template_manager.get_template(template_id)
        if template is None:
            raise TemplateNotFoundError(template_id)

        agent_config = template.get("agent_config") or {}
        final_description = description
        if final_description is None:
            descriptions = template.get("descriptions") or {}
            if isinstance(descriptions, dict):
                final_description = descriptions.get("en") or ""
            elif isinstance(descriptions, str):
                final_description = descriptions

        return AgentCreateSpec.from_values(
            name=name or template.get("name") or template_id,
            description=final_description,
            template_id=template_id,
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
            generate_runtime_key=generate_runtime_key,
        )

    async def create_agent_from_template(
        self,
        *,
        user_id: int,
        is_admin: bool,
        template_id: str,
        name: str | None,
        description: str | None,
        instructions: str | None,
        execution_mode: str | None,
        models: dict[str, Any] | None,
        knowledge_bases: list[str] | None,
        skills: list[str] | None,
        tool_categories: list[str] | None,
        suggested_prompts: list[str] | None,
        generate_runtime_key: bool,
    ) -> AgentCreateSnapshot:
        spec = await self._spec_from_template(
            template_id=template_id,
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
        return await self.create_agent(
            user_id=user_id,
            is_admin=is_admin,
            spec=spec,
        )

    async def resolve_agent_from_template(
        self,
        *,
        user_id: int,
        is_admin: bool,
        template_id: str,
        name: str | None = None,
    ) -> tuple[AgentResponseSnapshot, bool]:
        """Atomic server-side get-or-create keyed on (user_id, template_id,
        origin=template_quick_access).

        Backs flows with reuse semantics (the /task template quick-access):
        return the caller's own existing quick-access agent for this
        template as-is, or create and publish a fresh one. Unlike the plain
        create path this never raises on a duplicate default name; it
        disambiguates server-side instead. Reuse never republishes a found
        agent that isn't currently published - see
        :meth:`_resolve_agent_from_template_sync` for why.

        Contrast with :meth:`create_agent_from_template`, which is a pure
        create used by flows that deliberately mint multiple instances of one
        template under user-chosen names (workforce workers, via the plain
        ``origin=user`` ``POST /from-template``); those agents are invisible
        to this method's reuse query (origin-scoped, see
        AGENT_TEMPLATE_QUICK_ACCESS_UNIQUE_INDEX) and a DB-level uniqueness
        constraint on plain (user_id, template_id) would have broken them, so
        the constraint backing this method's idempotency is scoped to the
        quick-access origin specifically.

        ``name`` is create-only: it seeds the name for a freshly minted
        agent (disambiguated server-side on collision) but is not consulted
        on the reuse path. A caller passing a different ``name`` on a repeat
        call for a template it already has a quick-access agent for gets
        that existing agent back under its original name, silently -- the
        response's ``agent.name`` is the source of truth for what a caller
        should reconcile against.

        Returns the detached agent snapshot and whether it was newly created.
        """
        spec = await self._spec_from_template(
            template_id=template_id,
            name=name,
            description=None,
            instructions=None,
            execution_mode=None,
            models=None,
            knowledge_bases=None,
            skills=None,
            tool_categories=None,
            suggested_prompts=None,
            # The quick-access flow talks to the agent through the normal
            # chat session, never through a runtime API key.
            generate_runtime_key=False,
        )
        await _validate_agent_knowledge_bases(
            knowledge_bases=spec.knowledge_bases,
            tool_categories=spec.tool_categories,
            user_id=user_id,
            is_admin=is_admin,
        )
        return await run_db_io_cancellation_safe(
            lambda: self._resolve_agent_from_template_sync(
                user_id=user_id,
                template_id=template_id,
                spec=spec,
            )
        )

    @staticmethod
    def _resolve_agent_from_template_sync(
        *,
        user_id: int,
        template_id: str,
        spec: AgentCreateSpec,
    ) -> tuple[AgentResponseSnapshot, bool]:
        SessionLocal = get_session_local()
        with SessionLocal() as db:
            service = AgentManagementService(db)
            # Sticky: once a TemplateQuickAccessRaceError is seen, it is never
            # overwritten by a later DuplicateAgentNameError - retry
            # exhaustion must surface 409 even when a later attempt's
            # collision happened to be a plain name collision instead,
            # otherwise the two are indistinguishable to the caller (PR
            # review finding m6).
            last_error: (
                DuplicateAgentNameError | TemplateQuickAccessRaceError | None
            ) = None
            for _ in range(TEMPLATE_RESOLVE_RACE_RETRIES):
                # Strictly the caller's own rows (user_id filter, not the
                # team-wide owned_agent_clause): this path may publish what it
                # finds, and publishing a teammate's in-progress draft on this
                # user's behalf is exactly the bug this server-side resolve
                # exists to prevent (PR review finding F1). Also strictly
                # scoped to the quick-access origin (PR review finding B4):
                # without it, this query could adopt (and publish) an
                # unrelated agent the workforce-builder UI built from the
                # same template under a user-chosen name, since that flow
                # writes template_id too. Lowest id wins so concurrent
                # duplicates resolve deterministically (F3).
                existing = (
                    db.query(Agent)
                    .filter(
                        Agent.user_id == user_id,
                        Agent.template_id == template_id,
                        Agent.origin == AgentOrigin.TEMPLATE_QUICK_ACCESS.value,
                    )
                    .order_by(Agent.id)
                    .first()
                )
                if existing is not None:
                    # Deliberately does not auto-publish a found draft (PR
                    # review finding B3): status/published_at alone can't
                    # distinguish "never published yet" (the create-then-
                    # publish below failed) from "the user explicitly
                    # unpublished it" - both look identical. Silently
                    # republishing on an unrelated later call would revert
                    # that choice with zero visible signal. Returning it
                    # as-is is honest either way; the response's own
                    # status/published_at fields tell the caller the truth.
                    return (
                        _agent_response_snapshot(
                            service.store.agent_to_response_dict(existing)
                        ),
                        False,
                    )

                candidate_name = resolve_unique_agent_name(
                    db, user_id=user_id, name=spec.name
                )
                try:
                    agent, _ = service.create_agent_with_optional_key(
                        user_id=user_id,
                        name=candidate_name,
                        description=spec.description,
                        instructions=spec.instructions,
                        execution_mode=spec.execution_mode,
                        models=spec.model_mapping(),
                        knowledge_bases=list(spec.knowledge_bases),
                        skills=list(spec.skills),
                        tool_categories=list(spec.tool_categories),
                        suggested_prompts=list(spec.suggested_prompts),
                        generate_runtime_key=False,
                        template_id=template_id,
                        origin=AgentOrigin.TEMPLATE_QUICK_ACCESS.value,
                    )
                except (DuplicateAgentNameError, TemplateQuickAccessRaceError) as exc:
                    # A concurrent request inserted between our select and our
                    # insert. DuplicateAgentNameError means it took the exact
                    # name we picked; TemplateQuickAccessRaceError means it
                    # won this same (user, template)'s quick-access row even
                    # under a *different*, disambiguated name - the case the
                    # name-collision-only retry used to miss entirely (PR
                    # review findings B1/B2). Either way, re-select: if it was
                    # this template's quick-access agent we reuse it, else
                    # resolve_unique_agent_name picks a fresh name next pass.
                    if not isinstance(last_error, TemplateQuickAccessRaceError):
                        last_error = exc
                    db.rollback()
                    continue

                # Publish in a second commit. If this fails the create still
                # stands as an unpublished draft; per the note on the reuse
                # branch above, a later resolve call now returns it as-is
                # rather than silently republishing it.
                agent = service.store.publish_agent(user_id, int(agent.id)) or agent
                return (
                    _agent_response_snapshot(
                        service.store.agent_to_response_dict(agent)
                    ),
                    True,
                )
            raise last_error or DuplicateAgentNameError(spec.name)

    async def rotate_agent_runtime_key(
        self,
        *,
        user_id: int,
        agent_id: int,
    ) -> RuntimeKeySnapshot | None:
        return await self._run_runtime_key_delivery(
            lambda: self._rotate_agent_runtime_key_with_retry_sync(
                user_id=user_id,
                agent_id=agent_id,
            )
        )

    @staticmethod
    def _rotate_agent_runtime_key_with_retry_sync(
        *,
        user_id: int,
        agent_id: int,
    ) -> _RuntimeKeyDeliveryOutcome[RuntimeKeySnapshot]:
        for attempt in range(PREFIX_COLLISION_RETRIES):
            candidate = generate_api_key(None, kind=ApiKeyKind.AGENT)

            def rotate_attempt(
                service: AgentManagementService,
                candidate: ApiKeyCandidate = candidate,
            ) -> RuntimeKeySnapshot | None:
                response = service.generate_agent_runtime_key(
                    user_id=user_id,
                    agent_id=agent_id,
                    runtime_key_candidate=candidate,
                )
                return _runtime_key_snapshot(response)

            outcome = AgentManagementRuntime._run_runtime_key_session_operation(
                rotate_attempt
            )
            if (
                isinstance(outcome.error, KeyRotationConflict)
                and _is_runtime_key_prefix_collision(outcome.error)
                and attempt + 1 < PREFIX_COLLISION_RETRIES
            ):
                continue
            return outcome
        error = KeyRotationConflict(
            "Failed to generate a unique runtime key prefix after retrying."
        )
        return _RuntimeKeyDeliveryOutcome(
            result=None,
            receipt=None,
            error=error,
            traceback=error.__traceback__,
        )

    async def _run_runtime_key_delivery(
        self,
        operation: Callable[[], _RuntimeKeyDeliveryOutcome[_RuntimeKeyResultT]],
    ) -> _RuntimeKeyResultT | None:
        """Settle key delivery before exposing cancellation or worker failure."""

        worker = asyncio.create_task(asyncio.to_thread(operation))
        outcome, cancellation = await await_task_settlement(worker)
        with propagate_deferred_cancellation(cancellation):
            compensation_error: BaseException | None = None
            if outcome.receipt is not None and (
                cancellation is not None or outcome.error is not None
            ):
                compensation_error = await self._compensate_runtime_key(outcome.receipt)

            if outcome.error is not None:
                error = outcome.error.with_traceback(outcome.traceback)
                if compensation_error is not None:
                    error.add_note(
                        "Runtime-key compensation also failed: "
                        f"{type(compensation_error).__name__}: {compensation_error}"
                    )
                raise error
            if compensation_error is not None:
                # Compensation only runs without a worker error when caller
                # cancellation was captured. Raising here lets the shared
                # boundary preserve that cancellation with this failure as its
                # cause.
                raise compensation_error
        return outcome.result

    @staticmethod
    def _compensate_runtime_key_sync(
        receipt: RuntimeKeyReceipt,
    ) -> _RuntimeKeyCompensationResult:
        """Fence a failed transition and restore only its replaced key rows."""

        SessionLocal = get_session_local()
        with SessionLocal() as db:
            now = datetime.now(timezone.utc)
            agent_exists = acquire_runtime_key_transition_fence(
                db,
                receipt.agent_id,
            )
            new_key_revoked = 0
            prior_keys_restored = 0
            if agent_exists:
                revoke_result = cast(
                    "CursorResult[Any]",
                    db.execute(
                        update(AgentApiKey)
                        .where(
                            AgentApiKey.id == receipt.key_id,
                            AgentApiKey.agent_id == receipt.agent_id,
                            AgentApiKey.key_prefix == receipt.key_prefix,
                            AgentApiKey.revoked_at.is_(None),
                            AgentApiKey.paused_at.is_(None),
                        )
                        .values(revoked_at=now, updated_at=now)
                        .execution_options(synchronize_session=False)
                    ),
                )
                new_key_revoked = max(int(revoke_result.rowcount or 0), 0)
                if (
                    new_key_revoked == 1
                    and receipt.replaced_key_ids
                    and receipt.rotation_timestamp is not None
                ):
                    restore_result = cast(
                        "CursorResult[Any]",
                        db.execute(
                            update(AgentApiKey)
                            .where(
                                AgentApiKey.agent_id == receipt.agent_id,
                                AgentApiKey.id.in_(receipt.replaced_key_ids),
                                AgentApiKey.revoked_at == receipt.rotation_timestamp,
                            )
                            .values(revoked_at=None, updated_at=now)
                            .execution_options(synchronize_session=False)
                        ),
                    )
                    prior_keys_restored = max(
                        int(restore_result.rowcount or 0),
                        0,
                    )
            db.commit()

        result = _RuntimeKeyCompensationResult(
            new_key_revoked=new_key_revoked,
            prior_keys_restored=prior_keys_restored,
        )
        logger.info(
            "Compensated undelivered runtime key id=%s agent_id=%s prefix=%s "
            "(new_key_revoked=%d, prior_keys_restored=%d, expected_prior=%d)",
            receipt.key_id,
            receipt.agent_id,
            receipt.key_prefix,
            result.new_key_revoked,
            result.prior_keys_restored,
            len(receipt.replaced_key_ids),
        )
        return result

    async def _compensate_runtime_key(
        self,
        receipt: RuntimeKeyReceipt,
    ) -> BaseException | None:
        """Run exact revocation in a fresh worker-owned Session."""

        worker = asyncio.create_task(
            asyncio.to_thread(self._compensate_runtime_key_sync, receipt)
        )
        try:
            _result, cancellation = await await_task_settlement(worker)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            if is_process_control_exception(exc):
                raise
            logger.warning(
                "Failed to compensate undelivered runtime key "
                "id=%s agent_id=%s prefix=%s",
                receipt.key_id,
                receipt.agent_id,
                receipt.key_prefix,
                exc_info=True,
            )
            return exc
        with propagate_deferred_cancellation(cancellation):
            pass
        return None
