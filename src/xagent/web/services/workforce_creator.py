import json
import logging
import re
from dataclasses import dataclass
from typing import Any, cast

from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from xagent.web.models.agent import Agent, AgentOrigin, AgentStatus
from xagent.web.models.user import User
from xagent.web.services.agent_management import (
    TEMPLATE_RESOLVE_RACE_RETRIES,
    is_agent_name_unique_violation,
    is_agent_template_quick_access_unique_violation,
)
from xagent.web.services.llm_utils import UserAwareModelStorage

from ..models.workforce import Workforce, WorkforceBuilderMessage
from .agent_access import list_accessible_published_agents
from .agent_store import AgentStore
from .hot_path_cache import invalidate_agent_cache
from .workforce_access import (
    can_create_workforce,
    resolve_create_scope,
)
from .workforce_names import (
    is_workforce_name_unique_violation,
    resolve_unique_agent_name,
    resolve_unique_workforce_name,
)
from .workforce_snapshot import normalize_text
from .workforce_workers import create_workforce_worker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkforcePromptCreationResult:
    workforce: Workforce
    plan: dict[str, Any]
    messages: list[WorkforceBuilderMessage]


def _serialize_available_agents(agents: list[Agent]) -> list[dict[str, Any]]:
    return [
        {
            "agent_id": agent.id,
            "name": agent.name,
            "description": agent.description,
            "status": getattr(agent.status, "value", str(agent.status)),
        }
        for agent in agents
    ]


def _short_name_from_prompt(prompt: str) -> str:
    words = re.findall(r"[\w-]+", prompt, flags=re.UNICODE)
    if not words:
        return "New Workforce"
    name = " ".join(words[:6]).strip()
    if len(name) > 80:
        name = name[:80].rstrip()
    return f"{name} Workforce" if "workforce" not in name.lower() else name


def _clean_creation_plan(
    candidate: dict[str, Any],
    available_agent_ids: set[int],
    prompt: str,
) -> dict[str, Any]:
    name = str(candidate.get("name") or "").strip() or _short_name_from_prompt(prompt)
    description = str(candidate.get("description") or "").strip() or prompt.strip()

    manager = candidate.get("manager")
    if not isinstance(manager, dict):
        manager = {}
    manager_name = str(manager.get("name") or "").strip() or f"{name} Manager"
    manager_description = (
        str(manager.get("description") or "").strip()
        or "Coordinates this Workforce and synthesizes worker outputs."
    )
    manager_instructions = str(manager.get("instructions") or "").strip() or (
        "Understand the user's goal, delegate focused work to the available workers, "
        "compare their outputs, and return one coherent final answer."
    )

    workers = candidate.get("workers")
    if not isinstance(workers, list):
        workers = []
    clean_workers: list[dict[str, Any]] = []
    seen_agent_ids: set[int] = set()
    for item in workers:
        if not isinstance(item, dict):
            continue
        agent_id = item.get("agent_id")
        if not isinstance(agent_id, int):
            continue
        if agent_id not in available_agent_ids or agent_id in seen_agent_ids:
            continue
        instructions = str(item.get("assignment_instructions") or "").strip()
        if not instructions:
            continue
        enabled = item.get("enabled", True)
        clean_workers.append(
            {
                "agent_id": agent_id,
                "alias": str(item.get("alias") or "").strip() or None,
                "assignment_instructions": instructions,
                "enabled": enabled if isinstance(enabled, bool) else True,
            }
        )
        seen_agent_ids.add(agent_id)

    warnings = candidate.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
    clean_warnings = [str(item).strip() for item in warnings if str(item).strip()]
    if not clean_workers:
        clean_warnings.append(
            "No published worker agents were selected. Add workers before running."
        )

    return {
        "name": name[:200],
        "description": description,
        "manager": {
            "name": manager_name[:200],
            "description": manager_description,
            "instructions": manager_instructions,
        },
        "workers": clean_workers,
        "warnings": clean_warnings,
    }


def _fallback_creation_plan(prompt: str, agents: list[Agent]) -> dict[str, Any]:
    lower_prompt = prompt.lower()
    selected: list[Agent] = []
    for agent in agents:
        haystack = f"{agent.name} {agent.description or ''}".lower()
        if any(
            token and token in haystack
            for token in re.findall(r"[a-z0-9]+", lower_prompt)
        ):
            selected.append(agent)
        if len(selected) >= 3:
            break
    if not selected:
        selected = agents[: min(3, len(agents))]

    name = _short_name_from_prompt(prompt)
    return _clean_creation_plan(
        {
            "name": name,
            "description": prompt,
            "manager": {
                "name": f"{name} Manager",
                "description": "Coordinates this Workforce and synthesizes worker outputs.",
                "instructions": (
                    "Break the user request into worker assignments, call the right "
                    "workers, reconcile their outputs, and provide a concise final answer."
                ),
            },
            "workers": [
                {
                    "agent_id": int(agent.id),
                    "alias": agent.name,
                    "assignment_instructions": (
                        f"Contribute to the Workforce goal using the strengths of {agent.name}."
                    ),
                    "enabled": True,
                }
                for agent in selected
            ],
            "warnings": [
                "Created from a rule-based fallback because no LLM plan was available."
            ],
        },
        {int(agent.id) for agent in agents},
        prompt,
    )


async def generate_workforce_creation_plan(
    db: Session,
    user: User,
    prompt: str,
) -> dict[str, Any]:
    normalized_prompt = normalize_text(prompt, "prompt", required=True)
    agents = list_accessible_published_agents(db, user)
    available_agent_ids = {int(agent.id) for agent in agents}

    try:
        storage = UserAwareModelStorage(db)
        default_llm, _, _, _ = storage.get_configured_defaults(int(user.id))
        llm = default_llm
        if not llm:
            default_llm, _, _, _ = storage.get_configured_defaults(None)
            llm = default_llm
        if not llm:
            return _fallback_creation_plan(normalized_prompt, agents)

        system_prompt = (
            "You create initial AI Workforce drafts from a user's goal. "
            "A Workforce is AI orchestration, not a human team. "
            "Create a manager spec that can coordinate workers and synthesize results. "
            "Select workers only from available_published_agents. "
            "Do not create worker agents. If no worker fits, return no workers and add a warning. "
            "Return JSON only with keys: name, description, manager, workers, warnings. "
            "manager has name, description, instructions. "
            "Each worker has agent_id, alias, assignment_instructions, enabled. "
            "Keep instructions concrete and task-oriented."
        )
        user_prompt = json.dumps(
            {
                "request": normalized_prompt,
                "available_published_agents": _serialize_available_agents(agents),
            },
            ensure_ascii=False,
        )
        response = await llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = (
            response["content"]
            if isinstance(response, dict) and "content" in response
            else response
        )
        if not isinstance(content, str):
            content = str(content)
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            return _fallback_creation_plan(normalized_prompt, agents)
        return _clean_creation_plan(parsed, available_agent_ids, normalized_prompt)
    except Exception as exc:
        logger.warning("Failed to generate Workforce creation plan with LLM: %s", exc)
        return _fallback_creation_plan(normalized_prompt, agents)


def _create_manager_agent_from_plan(
    db: Session,
    user: User,
    name: str,
    manager_plan: dict[str, Any],
) -> Agent:
    return AgentStore(db).add_agent(
        user_id=int(user.id),
        name=resolve_unique_agent_name(
            db,
            user_id=int(user.id),
            name=str(manager_plan.get("name") or f"{name} Manager"),
        ),
        description=normalize_text(
            cast(str | None, manager_plan.get("description")),
            "description",
        ),
        instructions=normalize_text(
            cast(str | None, manager_plan.get("instructions")),
            "instructions",
        ),
        execution_mode="think",
        models=None,
        knowledge_bases=[],
        skills=[],
        tool_categories=[],
        suggested_prompts=[],
        origin=AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
        status=AgentStatus.PUBLISHED,
        widget_enabled=False,
        allowed_domains=[],
    )


def _find_quick_access_worker_agent(
    db: Session, *, user_id: int, template_id: str
) -> Agent | None:
    return (
        db.query(Agent)
        .filter(
            Agent.user_id == user_id,
            Agent.template_id == template_id,
            Agent.origin == AgentOrigin.TEMPLATE_QUICK_ACCESS.value,
        )
        .order_by(Agent.id)
        .first()
    )


# Machine-readable error codes for the workforce-from-template flow,
# following the structured-detail pattern already used by agents.py
# ("agent_in_use_by_workforce") and mcp.py. The frontend maps these codes
# to translated messages instead of byte-matching human-readable English
# detail strings, which silently degraded to a generic toast on any
# backend wording change. `params` carries the variable parts (e.g. the
# agent's name) separately so translations can interpolate them.
WORKFORCE_CREATE_ACCESS_DENIED_CODE = "workforce_create_access_denied"
WORKFORCE_CREATE_CONFLICT_CODE = "workforce_create_conflict"
WORKFORCE_WORKER_UNPUBLISHED_CODE = "workforce_worker_unpublished"


def _ensure_published_quick_access_agent(agent: Agent) -> Agent:
    """A quick-access worker agent the caller separately unpublished (a
    supported, deliberate user action - see
    `AgentManagementService._resolve_agent_from_template_sync`'s B3
    rationale for why the /task quick-access flow never auto-republishes a
    found draft either) must not be silently reused as-is: it would pass
    this check, then fail downstream in `create_workforce_worker`'s
    `ensure_agent_access(..., require_published=True)` with a generic 400
    the frontend can only render as "please retry" - which can never
    succeed, since retrying resolves to the same unpublished agent every
    time. Raise here instead, with a message specific enough to actually
    be actionable.
    """
    if agent.status != AgentStatus.PUBLISHED:
        raise HTTPException(
            status_code=400,
            detail={
                "code": WORKFORCE_WORKER_UNPUBLISHED_CODE,
                "message": f"This workforce needs an agent that is "
                f"currently unpublished: {agent.name}. Republish it from "
                "your Agents list, then try this workforce template again.",
                "params": {"agent_name": agent.name},
            },
        )
    return agent


async def _get_or_create_quick_access_worker_agent(
    db: Session,
    template_manager: Any,
    *,
    user_id: int,
    template_id: str,
) -> Agent:
    """Reuse the caller's existing quick-access instance of `template_id` if
    one exists, else create+publish a new one. Mirrors the get-or-create
    query `AgentManagementService.resolve_agent_from_template` uses for the
    /task quick-access flow, so instantiating the same workforce template
    twice does not mint duplicate worker agents.

    A plain select-then-insert here would race: two concurrent calls (e.g. a
    double-clicked "Use") can both miss the SELECT and then collide on
    `uq_agents_user_id_template_id_quick_access` at INSERT. Each insert
    attempt therefore runs in its own SAVEPOINT (`db.begin_nested()`) so a
    collision only unwinds that attempt - not the manager agent / Workforce
    already staged in the caller's outer transaction - and we retry by
    re-reading the row the winning concurrent request just committed - which
    depends on the connection's isolation level being READ COMMITTED (the
    default for both SQLite and PostgreSQL here; nothing in this codebase
    overrides it). Under REPEATABLE READ the re-select could still miss the
    just-committed row and this would exhaust its retries.

    Same intent as `AgentManagementService._resolve_agent_from_template_sync`,
    which hardens the same (user_id, template_id, quick-access origin)
    uniqueness for the /task quick-access flow, but not the same mechanism:
    that resolver does a full `db.rollback()` per attempt (it owns its own
    session), while this one uses a SAVEPOINT (`db.begin_nested()`) so a
    collision only unwinds this attempt - not the manager agent / Workforce
    already staged in the caller's outer transaction. Both discriminate the
    same two constraints via `is_agent_template_quick_access_unique_violation`
    / `is_agent_name_unique_violation` before deciding how to retry.
    """
    existing = _find_quick_access_worker_agent(
        db, user_id=user_id, template_id=template_id
    )
    if existing is not None:
        return _ensure_published_quick_access_agent(existing)

    worker_template = await template_manager.get_template(template_id)
    if not worker_template:
        raise HTTPException(
            status_code=400,
            detail=f"Workforce template references an unknown template: {template_id}",
        )
    if worker_template.get("type", "agent") != "agent":
        # A workforce template's agents[] must reference single-agent
        # templates. TemplateManager._enrich_template nulls agent_config for
        # any non-agent template, so without this check a workforce
        # referencing another workforce would crash below on
        # `None.get(...)` instead of failing with a clear message.
        raise HTTPException(
            status_code=400,
            detail=f"Workforce template references a non-agent template as a "
            f"worker: {template_id}",
        )
    agent_config = worker_template.get("agent_config") or {}
    # Populate the same fields the /task quick-access resolver
    # (`AgentManagementRuntime._spec_from_template`) does for a freshly
    # minted quick-access agent - both write the same (user_id, template_id,
    # quick-access-origin) row, so leaving these hardcoded to empty here
    # would silently strand them the first time this path (rather than
    # /task) happens to create the row first.
    # Deliberate exception: `knowledge_bases` stays empty. KB ids are
    # user-scoped runtime entities the /task path validates per-user via
    # `_validate_agent_knowledge_bases` (rejecting the template with a 400
    # if they don't resolve) - this path has no equivalent check, so
    # passing them through raw would silently create an agent with
    # dangling KB references instead. No built-in template declares
    # knowledge_bases today; if one ever does, this path needs the same
    # validation before it may forward them.
    worker_descriptions = worker_template.get("descriptions") or {}
    worker_description = (
        worker_descriptions.get("en") if isinstance(worker_descriptions, dict) else None
    )

    for attempt in range(TEMPLATE_RESOLVE_RACE_RETRIES):
        try:
            with db.begin_nested():
                agent = AgentStore(db).add_agent(
                    user_id=user_id,
                    name=resolve_unique_agent_name(
                        db,
                        user_id=user_id,
                        name=str(worker_template.get("name") or template_id),
                    ),
                    description=worker_description,
                    instructions=agent_config.get("instructions", ""),
                    execution_mode=agent_config.get("execution_mode", "balanced"),
                    models=agent_config.get("models"),
                    knowledge_bases=[],
                    skills=agent_config.get("skills", []),
                    tool_categories=agent_config.get("tool_categories", []),
                    suggested_prompts=agent_config.get("suggested_prompts") or [],
                    origin=AgentOrigin.TEMPLATE_QUICK_ACCESS.value,
                    status=AgentStatus.PUBLISHED,
                    widget_enabled=False,
                    allowed_domains=[],
                    template_id=template_id,
                )
            return agent
        except IntegrityError as exc:
            # The savepoint rollback above already discarded our failed
            # insert. Which constraint fired determines what a retry should
            # even do - conflating them previously misdiagnosed a genuine
            # name collision as this template's quick-access race, which
            # burns every retry on a re-select that can never find a row
            # (nothing else raced for *this* template_id) and returns a
            # misleading 409.
            if is_agent_template_quick_access_unique_violation(exc):
                # A concurrent request won the (user_id, template_id)
                # quick-access race; its row should now be visible.
                existing = _find_quick_access_worker_agent(
                    db, user_id=user_id, template_id=template_id
                )
                if existing is not None:
                    return _ensure_published_quick_access_agent(existing)
                logger.warning(
                    "Quick-access worker agent insert collided on the "
                    "quick-access index without a resolvable row (attempt "
                    "%s/%s) for template_id=%s",
                    attempt + 1,
                    TEMPLATE_RESOLVE_RACE_RETRIES,
                    template_id,
                )
            elif is_agent_name_unique_violation(exc):
                # resolve_unique_agent_name's own check-then-insert lost a
                # race for the name it picked - not a quick-access race at
                # all. Retrying picks a fresh name via the next iteration's
                # resolve_unique_agent_name call.
                logger.warning(
                    "Quick-access worker agent name collided (attempt "
                    "%s/%s) for template_id=%s; retrying with a new name",
                    attempt + 1,
                    TEMPLATE_RESOLVE_RACE_RETRIES,
                    template_id,
                )
            else:
                # An unrecognized constraint (e.g. a widget_key collision or
                # an unrelated FK failure) - don't misdiagnose it as either
                # race above.
                raise

    raise HTTPException(
        status_code=409,
        detail={
            "code": WORKFORCE_CREATE_CONFLICT_CODE,
            "message": "Could not create the workforce's worker agents due "
            "to a concurrent request; please try again.",
        },
    )


async def create_workforce_from_template(
    db: Session,
    user: User,
    template_manager: Any,
    template: dict[str, Any],
    lang: str | None = None,
) -> Workforce:
    """Instantiate a workforce-type template: a fresh manager agent (from
    `workforce_config.manager`) plus one worker agent per
    `workforce_config.agents[]` entry, reused/created from that entry's own
    `template_id` (see `_get_or_create_quick_access_worker_agent`), then
    assembled into a new `Workforce`. Mirrors the transaction shape of
    `create_workforce_from_prompt` above: everything commits together, or
    rolls back together.
    """
    workforce_config = template.get("workforce_config") or {}
    manager_spec = cast(dict[str, Any], workforce_config.get("manager") or {})
    agent_specs = cast(list[dict[str, Any]], workforce_config.get("agents") or [])
    if not manager_spec.get("instructions") or not agent_specs:
        raise HTTPException(
            status_code=400, detail="Template is missing workforce configuration"
        )

    scope_type, scope_id = resolve_create_scope(db, user)
    if not can_create_workforce(db, user, scope_type, scope_id):
        raise HTTPException(
            status_code=403,
            detail={
                "code": WORKFORCE_CREATE_ACCESS_DENIED_CODE,
                "message": "Access denied",
            },
        )

    owner_user_id = int(user.id)
    try:
        base_name = str(template.get("name") or "Workforce")
        name = resolve_unique_workforce_name(
            db,
            scope_type=scope_type,
            scope_id=scope_id,
            name=base_name,
        )
        # The manager-agent insert (unlike the Workforce insert just below)
        # is NOT wrapped in its own `db.begin_nested()` - that is safe only
        # because `uq_agents_user_id_name_active` (`AGENT_NAME_UNIQUE_INDEX`
        # in `xagent/web/models/agent.py`) is a partial index whose
        # predicate excludes `origin='workforce_generated_manager'` rows
        # entirely, so two concurrent managers can even share a name
        # without ever hitting a unique-constraint collision in the first
        # place. If that predicate ever changes, this insert would need the
        # same SAVEPOINT-retry treatment as the Workforce insert below.
        manager_agent = AgentStore(db).add_agent(
            user_id=owner_user_id,
            name=resolve_unique_agent_name(
                db,
                user_id=owner_user_id,
                name=str(manager_spec.get("name") or f"{name} Manager"),
            ),
            description=normalize_text(
                cast(str | None, manager_spec.get("description")), "description"
            ),
            instructions=str(manager_spec["instructions"]),
            execution_mode=str(manager_spec.get("execution_mode") or "think"),
            models=None,
            knowledge_bases=[],
            skills=cast(list[str], manager_spec.get("skills") or []),
            tool_categories=cast(list[str], manager_spec.get("tool_categories") or []),
            suggested_prompts=[],
            origin=AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
            status=AgentStatus.PUBLISHED,
            widget_enabled=False,
            allowed_domains=[],
        )

        workforce_description = normalize_text(
            cast(str | None, get_localized_description(template, lang)),
            "description",
        )
        # `name` came from an unlocked check-then-insert
        # (`resolve_unique_workforce_name`), same shape as the worker-agent
        # race below - two concurrent instantiations of the SAME template
        # (this template's name is fixed, unlike `create_workforce_from_prompt`'s
        # LLM-generated one, so this collision is not just theoretical) can
        # both resolve the same name and race to insert it. Retry inside a
        # SAVEPOINT so a collision only unwinds this insert, not the
        # manager agent already created above.
        workforce: Workforce | None = None
        for attempt in range(TEMPLATE_RESOLVE_RACE_RETRIES):
            try:
                with db.begin_nested():
                    workforce = Workforce(
                        owner_user_id=owner_user_id,
                        scope_type=scope_type,
                        scope_id=scope_id,
                        name=name,
                        description=workforce_description,
                        manager_agent_id=int(manager_agent.id),
                        status="draft",
                    )
                    db.add(workforce)
                    db.flush()
                break
            except IntegrityError as exc:
                if not is_workforce_name_unique_violation(exc):
                    raise
                logger.warning(
                    "Workforce name %r collided (attempt %s/%s) for "
                    "scope_type=%s scope_id=%s; retrying with a new name",
                    name,
                    attempt + 1,
                    TEMPLATE_RESOLVE_RACE_RETRIES,
                    scope_type,
                    scope_id,
                )
                # Re-resolve from the template's RAW name, never from a
                # resolved (possibly suffixed) one - resolving from a
                # suffixed name compounds suffixes ("X 2" -> "X 2 2")
                # instead of advancing to "X 3". That holds both for the
                # collided `name` from this loop and for the initial
                # resolution above, which itself already carries a suffix
                # whenever the user instantiated this template before.
                name = resolve_unique_workforce_name(
                    db, scope_type=scope_type, scope_id=scope_id, name=base_name
                )
        else:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": WORKFORCE_CREATE_CONFLICT_CODE,
                    "message": "Could not create the workforce due to a "
                    "concurrent request; please try again.",
                },
            )
        assert workforce is not None

        for index, agent_spec in enumerate(agent_specs):
            worker_agent = await _get_or_create_quick_access_worker_agent(
                db,
                template_manager,
                user_id=owner_user_id,
                template_id=str(agent_spec["template_id"]),
            )
            create_workforce_worker(
                db,
                workforce,
                user,
                source_type="existing",
                agent_id=int(worker_agent.id),
                alias=cast(str | None, agent_spec.get("alias")),
                assignment_instructions=str(agent_spec["assignment_instructions"]),
                enabled=True,
                sort_order=index + 1,
                template_id=str(agent_spec["template_id"]),
            )

        manager_agent_id = int(manager_agent.id)
        workforce_id = int(workforce.id)
        db.commit()
    except Exception:
        db.rollback()
        raise
    else:
        invalidate_workforce_creation_cache(
            owner_user_id=owner_user_id,
            manager_agent_id=manager_agent_id,
            workforce_id=workforce_id,
        )
    db.refresh(workforce)
    return workforce


def get_localized_description(
    template: dict[str, Any], lang: str | None = None
) -> str | None:
    """Best-effort description for a newly created Workforce's
    `description` column. Unlike the template gallery response, this was
    never locale-aware at all - `create_workforce_from_template` had no way
    to know the caller's locale, so every Workforce got its English
    description regardless of the creating user's UI language. `lang`
    threads the same query param the sibling GET endpoints already accept.
    `descriptions` is always a {en, zh, ...} dict
    here - `TemplateManager._parse_yaml_file` raises ValueError for any
    other shape, so a template dict reaching this function can't carry a
    plain string instead.

    Preference order: the caller's `lang` if populated, else English, else
    any other populated locale rather than leaving the Workforce's
    description empty - the key check in `_parse_yaml_file` only requires
    an 'en' key to be *present*, not a non-empty value.
    """
    descriptions = template.get("descriptions")
    if not isinstance(descriptions, dict):
        return None
    if lang and descriptions.get(lang):
        return str(descriptions[lang])
    value = descriptions.get("en")
    if value:
        return str(value)
    for other_value in descriptions.values():
        if other_value:
            return str(other_value)
    return None


def invalidate_workforce_creation_cache(
    *,
    owner_user_id: int,
    manager_agent_id: int,
    workforce_id: int,
) -> None:
    del workforce_id
    invalidate_agent_cache(owner_user_id, manager_agent_id)


async def create_workforce_from_prompt(
    db: Session,
    user: User,
    *,
    prompt: str,
) -> WorkforcePromptCreationResult:
    normalized_prompt = normalize_text(prompt, "prompt", required=True)
    scope_type, scope_id = resolve_create_scope(db, user)
    if not can_create_workforce(db, user, scope_type, scope_id):
        raise HTTPException(status_code=403, detail="Access denied")

    owner_user_id = int(user.id)
    try:
        plan = await generate_workforce_creation_plan(db, user, normalized_prompt)
        name = resolve_unique_workforce_name(
            db,
            scope_type=scope_type,
            scope_id=scope_id,
            name=str(plan["name"]),
        )
        manager_plan = cast(dict[str, Any], plan["manager"])
        manager_agent = _create_manager_agent_from_plan(db, user, name, manager_plan)

        workforce = Workforce(
            owner_user_id=int(user.id),
            scope_type=scope_type,
            scope_id=scope_id,
            name=name,
            description=normalize_text(
                cast(str | None, plan.get("description")),
                "description",
            ),
            manager_agent_id=int(manager_agent.id),
            status="draft",
        )
        db.add(workforce)
        db.flush()

        for index, worker in enumerate(
            cast(list[dict[str, Any]], plan.get("workers") or [])
        ):
            create_workforce_worker(
                db,
                workforce,
                user,
                source_type="existing",
                agent_id=cast(int, worker["agent_id"]),
                alias=cast(str | None, worker.get("alias")),
                assignment_instructions=str(worker["assignment_instructions"]),
                enabled=bool(worker.get("enabled", True)),
                sort_order=index + 1,
            )

        user_message = WorkforceBuilderMessage(
            workforce_id=int(workforce.id),
            user_id=int(user.id),
            role="user",
            content=normalized_prompt,
            status="message",
        )
        db.add(user_message)
        warnings = cast(list[str], plan.get("warnings") or [])
        assistant_content = "Created an initial Workforce draft from your prompt."
        if warnings:
            assistant_content = f"{assistant_content}\n\n" + "\n".join(
                f"- {warning}" for warning in warnings
            )
        assistant_message = WorkforceBuilderMessage(
            workforce_id=int(workforce.id),
            user_id=int(user.id),
            role="assistant",
            content=assistant_content,
            status="message",
        )
        db.add(assistant_message)
        manager_agent_id = int(manager_agent.id)
        workforce_id = int(workforce.id)
        db.commit()
    except Exception:
        db.rollback()
        raise
    else:
        invalidate_workforce_creation_cache(
            owner_user_id=owner_user_id,
            manager_agent_id=manager_agent_id,
            workforce_id=workforce_id,
        )
    db.refresh(workforce)
    db.refresh(user_message)
    db.refresh(assistant_message)
    return WorkforcePromptCreationResult(
        workforce=workforce,
        plan=plan,
        messages=[user_message, assistant_message],
    )
