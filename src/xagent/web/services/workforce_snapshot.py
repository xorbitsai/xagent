import hashlib
import json
from typing import Any, Literal, cast, overload

from fastapi import HTTPException
from sqlalchemy.orm import Session

from xagent.core.execution_scope import (
    EXECUTION_SCOPE_AGENT_CONFIG_KEY,
    get_execution_scope,
)
from xagent.core.tools.adapters.vibe.agent_tool_names import (
    gen_workforce_agent_tool_name,
)
from xagent.web.models.agent import Agent
from xagent.web.models.user import User

from ..models.workforce import Workforce, WorkforceAgent
from .workforce_access import ensure_workforce_access, ensure_workforce_agent_run_access

WORKFORCE_STATUSES = {"draft", "active", "archived"}
RUN_STATUSES = {"pending", "running", "paused", "completed", "failed", "cancelled"}


def normalize_workforce_status(status: str | None) -> str:
    normalized = (status or "draft").strip().lower()
    if normalized not in WORKFORCE_STATUSES:
        raise HTTPException(status_code=400, detail="Invalid workforce status")
    return normalized


def normalize_workforce_run_status(status: str | None) -> str:
    normalized = (status or "pending").strip().lower()
    if normalized not in RUN_STATUSES:
        raise HTTPException(status_code=400, detail="Invalid workforce run status")
    return normalized


@overload
def normalize_text(
    value: str | None, field_name: str, required: Literal[True]
) -> str: ...


@overload
def normalize_text(
    value: str | None, field_name: str, required: Literal[False] = False
) -> str | None: ...


def normalize_text(
    value: str | None, field_name: str, required: bool = False
) -> str | None:
    if value is None:
        if required:
            raise HTTPException(status_code=400, detail=f"{field_name} is required")
        return None
    normalized = value.strip()
    if required and not normalized:
        raise HTTPException(status_code=400, detail=f"{field_name} is required")
    return normalized or None


def build_worker_tool_name(agent_id: int, alias: str | None = None) -> str:
    return gen_workforce_agent_tool_name(agent_id, alias)


def _sorted_workers(workforce: Workforce) -> list[WorkforceAgent]:
    return sorted(
        workforce.workers, key=lambda item: (item.sort_order or 0, item.id or 0)
    )


def validate_workforce_for_run(
    db: Session,
    user: User,
    workforce: Workforce,
    *,
    is_preview: bool = False,
) -> tuple[Agent, list[WorkforceAgent]]:
    workforce = ensure_workforce_access(db, user, workforce, action="run")
    if workforce.status == "archived":
        raise HTTPException(status_code=400, detail="Archived workforce cannot run")
    if workforce.status != "active" and not is_preview:
        raise HTTPException(status_code=400, detail="Workforce must be active to run")

    manager_agent = ensure_workforce_agent_run_access(
        workforce.manager_agent, user, db, workforce
    )
    workers = _sorted_workers(workforce)
    enabled_workers = [worker for worker in workers if worker.enabled]
    if not enabled_workers:
        raise HTTPException(
            status_code=400, detail="Workforce requires at least one enabled worker"
        )

    for worker in enabled_workers:
        ensure_workforce_agent_run_access(worker.agent, user, db, workforce)
        instructions = normalize_text(
            cast(str | None, worker.assignment_instructions),
            "assignment_instructions",
            required=True,
        )
        if instructions is None:
            raise HTTPException(
                status_code=400, detail="assignment_instructions is required"
            )
        if int(worker.agent_id) == int(workforce.manager_agent_id):
            raise HTTPException(
                status_code=400, detail="Manager agent cannot also be a worker"
            )

    return manager_agent, enabled_workers


def build_manager_system_prompt(snapshot: dict[str, Any]) -> str:
    workforce = snapshot["workforce"]
    workers = snapshot["workers"]
    lines = [
        f'You are the Workforce Manager for "{workforce["name"]}".',
        "",
        "You are the only agent that talks to the user. You may delegate work only to the Worker Agents exposed as tools in this Workforce.",
        "",
        "Rules:",
        "1. Decide which Worker Agents are needed for the user's request.",
        "2. Give each Worker Agent a focused task with enough context.",
        "3. Do not delegate outside this Workforce.",
        "4. Consolidate Worker results into one final answer.",
        "5. If Worker outputs conflict, resolve the conflict or explain uncertainty.",
        "6. Do not expose internal tool names unless necessary.",
        "7. Use the exact tool name shown for each Worker Agent. Never infer a "
        "tool name from worker order or agent ids.",
        "",
        "Available Worker Agents:",
    ]
    for worker in workers:
        alias = worker.get("alias") or worker["name"]
        lines.append(
            f"- {alias} (tool: {worker['tool_name']}): "
            f"{worker['assignment_instructions']}"
        )
    return "\n".join(lines)


def build_worker_system_prompt(
    workforce_name: str, assignment_instructions: str
) -> str:
    return "\n".join(
        [
            f'You are being called as part of Workforce "{workforce_name}".',
            "",
            "Your assignment in this Workforce:",
            assignment_instructions,
            "",
            "Stay within this assignment. Return your result to the Workforce Manager. Do not address the end user directly unless asked.",
        ]
    )


def build_agent_tool_overrides(
    snapshot: dict[str, Any],
    workforce_run_id: int | None = None,
) -> dict[int, dict[str, Any]]:
    workforce_name = snapshot["workforce"]["name"]
    workforce_id = snapshot["workforce"]["id"]
    overrides: dict[int, dict[str, Any]] = {}
    for worker in snapshot["workers"]:
        alias = worker.get("alias") or worker["name"]
        tool_name = worker.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name.strip():
            tool_name = build_worker_tool_name(int(worker["agent_id"]), alias)
        description_parts = []
        if worker.get("description"):
            description_parts.append(worker["description"])
        description_parts.append(f"Workforce role: {alias}.")
        description_parts.append(f"Assignment: {worker['assignment_instructions']}")
        overrides[int(worker["agent_id"])] = {
            "tool_name": tool_name,
            "description": " ".join(description_parts),
            "extra_system_prompt": build_worker_system_prompt(
                workforce_name, worker["assignment_instructions"]
            ),
            "allowed_agent_ids": [],
            "enable_global_agent_tools": False,
            "workforce_run_id": workforce_run_id,
            "workforce_id": workforce_id,
            "workforce_name": workforce_name,
            "worker_member_id": worker.get("member_id"),
            "worker_alias": alias,
        }
    return overrides


# Version of the fingerprint ALGORITHM, stored alongside the pinned value in
# the run snapshot. The turn-entry guard only compares pinned vs live when the
# pinned version matches; a version bump therefore exempts runs pinned under an
# older algorithm instead of spuriously rejecting them with
# "workforce_config_changed" on deploy. Bump whenever the payload shape or
# canonicalization below changes. History: 1 = unsorted list fields (never
# released); 2 = list fields canonicalized via sort.
WORKFORCE_CONFIG_FINGERPRINT_VERSION = 2


def _fingerprint_agent_payload(agent: Agent) -> dict[str, Any]:
    # knowledge_bases / skills / tool_categories are order-insensitive sets
    # persisted verbatim from the frontend's multi-selects, which append in
    # click order. Canonicalize (sort) them so re-saving the same set in a
    # different array order doesn't change the fingerprint and force-reject
    # in-flight sessions. json.dumps(sort_keys=True) sorts dict keys only,
    # not list contents.
    return {
        "instructions": agent.instructions,
        "execution_mode": agent.execution_mode,
        "models": agent.models or {},
        "knowledge_bases": sorted(agent.knowledge_bases or [], key=str),
        "skills": sorted(agent.skills or [], key=str),
        "tool_categories": sorted(agent.tool_categories or [], key=str),
    }


def compute_workforce_config_fingerprint(
    workforce: Workforce,
    manager_agent: Agent,
    enabled_workers: list[WorkforceAgent],
) -> str:
    """Hash the live config that shapes a run's execution.

    The run snapshot only freezes prompt-building data; worker execution
    re-reads the live Agent per call (instructions, models, KBs, skills,
    tool categories). A full deep-freeze is impossible — KB content, MCP
    config and model keys are inherently live — so instead this fingerprint
    is stored at run creation and re-checked at each new turn's entry:
    a mismatch rejects the turn ("config changed, start a new session")
    rather than silently executing with drifted config.
    """
    payload = {
        "version": WORKFORCE_CONFIG_FINGERPRINT_VERSION,
        "workforce": {"id": workforce.id, "name": workforce.name},
        "manager": {
            "agent_id": manager_agent.id,
            **_fingerprint_agent_payload(manager_agent),
        },
        "workers": [
            {
                "member_id": worker.id,
                "agent_id": worker.agent_id,
                "alias": worker.alias,
                "assignment_instructions": worker.assignment_instructions,
                "agent": _fingerprint_agent_payload(worker.agent),
            }
            for worker in enabled_workers
        ],
    }
    canonical = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_live_workforce_config_fingerprint(workforce: Workforce) -> str | None:
    """Recompute the fingerprint from the current DB state of a workforce.

    Applies the same enabled-worker filter and ordering as
    ``build_workforce_snapshot``. Returns ``None`` when the live state cannot
    produce a comparable fingerprint (e.g. the manager relationship is gone),
    which callers should treat as a mismatch.
    """
    manager_agent = workforce.manager_agent
    if manager_agent is None:
        return None
    enabled_workers = [
        worker
        for worker in _sorted_workers(workforce)
        if worker.enabled and worker.agent is not None
    ]
    return compute_workforce_config_fingerprint(
        workforce, cast(Agent, manager_agent), enabled_workers
    )


def build_workforce_snapshot(
    db: Session,
    user: User,
    workforce: Workforce,
    *,
    is_preview: bool = False,
) -> dict[str, Any]:
    manager_agent, enabled_workers = validate_workforce_for_run(
        db,
        user,
        workforce,
        is_preview=is_preview,
    )
    snapshot_workers: list[dict[str, Any]] = []
    for worker in enabled_workers:
        alias = (
            normalize_text(cast(str | None, worker.alias), "alias") or worker.agent.name
        )
        assignment_instructions = normalize_text(
            cast(str | None, worker.assignment_instructions),
            "assignment_instructions",
            required=True,
        )
        if assignment_instructions is None:
            raise HTTPException(
                status_code=400, detail="assignment_instructions is required"
            )

        snapshot_workers.append(
            {
                "member_id": worker.id,
                "agent_id": worker.agent_id,
                "name": worker.agent.name,
                "alias": alias,
                "description": worker.agent.description,
                "assignment_instructions": assignment_instructions,
                "execution_mode": worker.agent.execution_mode,
                "tool_name": build_worker_tool_name(cast(int, worker.agent_id), alias),
                "enabled": bool(worker.enabled),
            }
        )

    snapshot: dict[str, Any] = {
        "version": 1,
        "workforce": {
            "id": workforce.id,
            "name": workforce.name,
            "description": workforce.description,
            "status": workforce.status,
            "scope_type": workforce.scope_type,
            "scope_id": workforce.scope_id,
            "owner_user_id": workforce.owner_user_id,
        },
        "manager": {
            "agent_id": manager_agent.id,
            "name": manager_agent.name,
            "description": manager_agent.description,
            "instructions": manager_agent.instructions,
            "execution_mode": manager_agent.execution_mode,
            "models": manager_agent.models or {},
        },
        "workers": snapshot_workers,
    }
    snapshot["manager"]["runtime_prompt"] = build_manager_system_prompt(snapshot)
    snapshot["config_fingerprint"] = compute_workforce_config_fingerprint(
        workforce, manager_agent, enabled_workers
    )
    snapshot["config_fingerprint_version"] = WORKFORCE_CONFIG_FINGERPRINT_VERSION
    return snapshot


def build_workforce_task_config(
    snapshot: dict[str, Any],
    selected_file_ids: list[str] | None = None,
    workforce_run_id: int | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "workforce_id": snapshot["workforce"]["id"],
        "workforce_snapshot": snapshot,
    }
    if workforce_run_id is not None:
        config["workforce_run_id"] = workforce_run_id
    if selected_file_ids:
        config["selected_file_ids"] = selected_file_ids
    # Workforce runs create fresh Task rows whose ids the embedder's scope
    # resolver cannot map. Persist the creating context's ExecutionScope
    # snapshot into agent_config (no schema migration); per-turn activation
    # prefers this snapshot over the resolver, so the run executes fully
    # scoped even after a process restart.
    scope = get_execution_scope()
    if scope is not None:
        config[EXECUTION_SCOPE_AGENT_CONFIG_KEY] = scope.to_dict()
    return config
