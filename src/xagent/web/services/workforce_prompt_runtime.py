"""ReAct runtime for building a Workforce and all of its required agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, Type
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from ...core.agent.language import (
    detect_prose_script_mismatch,
    output_language_policy,
    response_language_rules,
)
from ...core.agent.result import extract_assistant_message
from ...core.agent.service import AgentService
from ...core.agent.voice_policy import apply_output_voice
from ...core.model.chat.basic.base import BaseLLM
from ...core.tools.adapters.vibe.agent_tool import (
    ListAvailableSkillsTool,
    ListToolCategoriesTool,
)
from ...core.tools.adapters.vibe.base import (
    AbstractBaseTool,
    ToolCategory,
    ToolVisibility,
)
from ...core.utils.type_check import ensure_list

_EXECUTION_MODES = frozenset({"flash", "balanced", "think", "auto"})
MAX_WORKFORCE_PROMPT_LENGTH = 12_000
MAX_WORKFORCE_BUILDER_AGENTS = 16
MAX_WORKFORCE_BUILDER_WORKERS = 32
MAX_WORKFORCE_BUILDER_EXISTING_AGENTS = 200
MAX_WORKFORCE_BUILDER_AGENT_RESULTS = 50
WORKFORCE_BUILDER_MAX_ITERATIONS = 48
# Named list of the persisted-configuration fields the language-scoping
# instruction in workforce_prompt_builder_system_prompt (below) exempts
# from the request language policy. Voice-scoping used to enumerate the
# same list a second time here; that's now handled once, centrally, by
# apply_output_voice's own caveat instead.
_WORKFORCE_BUILDER_PERSISTED_FIELDS = (
    "Workforce and agent names, descriptions, instructions, aliases, and "
    "assignment text"
)


class WorkforcePromptBuilderError(RuntimeError):
    """The ReAct builder stopped before producing a valid Workforce."""


class WorkforcePromptBuilderUnavailableError(RuntimeError):
    """The builder runtime could not start or complete its execution boundary."""


@dataclass(frozen=True)
class StagedAgentSpec:
    ref: str
    name: str
    description: str
    instructions: str
    tool_categories: list[str]
    skills: list[str] | None
    execution_mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_ref": self.ref,
            "name": self.name,
            "description": self.description,
            "instructions": self.instructions,
            "tool_categories": self.tool_categories,
            "skills": self.skills,
            "execution_mode": self.execution_mode,
        }


@dataclass(frozen=True)
class StagedWorkforceWorker:
    agent_ref: str
    alias: str | None
    assignment_instructions: str
    enabled: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_ref": self.agent_ref,
            "alias": self.alias,
            "assignment_instructions": self.assignment_instructions,
            "enabled": self.enabled,
        }


@dataclass(frozen=True)
class StagedWorkforceSpec:
    name: str
    description: str
    manager_agent_ref: str
    workers: tuple[StagedWorkforceWorker, ...]


@dataclass
class WorkforcePromptBuilderState:
    """In-memory staging area shared by the Workforce builder tools."""

    existing_agents: dict[str, dict[str, Any]]
    created_agents: dict[str, StagedAgentSpec] = field(default_factory=dict)
    finalized_workforce: StagedWorkforceSpec | None = None
    listed_existing_agents: bool = False
    _next_agent_number: int = 1

    @classmethod
    def from_agents(
        cls, agents: Sequence[Mapping[str, Any]]
    ) -> WorkforcePromptBuilderState:
        return cls(
            existing_agents={
                f"existing:{int(agent['agent_id'])}": {
                    "agent_ref": f"existing:{int(agent['agent_id'])}",
                    "agent_id": int(agent["agent_id"]),
                    "name": str(agent["name"]),
                    "description": str(agent.get("description") or ""),
                    "status": str(agent.get("status") or ""),
                }
                for agent in agents
            }
        )

    def stage_agent(self, args: Mapping[str, Any]) -> dict[str, Any]:
        if self.finalized_workforce is not None:
            return {
                "status": "error",
                "message": "The Workforce is already finalized; no more agents may be staged.",
            }

        name = str(args.get("name") or "").strip()[:200].rstrip()
        description = str(args.get("description") or "").strip()
        instructions = str(args.get("instructions") or "").strip()
        if not name or not description or not instructions:
            return {
                "status": "error",
                "message": "name, description, and instructions are required.",
            }
        if any(
            spec.name.casefold() == name.casefold()
            for spec in self.created_agents.values()
        ):
            return {
                "status": "error",
                "message": f"An agent named {name!r} is already staged.",
            }
        if len(self.created_agents) >= MAX_WORKFORCE_BUILDER_AGENTS:
            return {
                "status": "error",
                "message": (
                    "The Workforce builder reached its staged-agent limit of "
                    f"{MAX_WORKFORCE_BUILDER_AGENTS}."
                ),
            }

        execution_mode = str(args.get("execution_mode") or "balanced").strip()
        if execution_mode not in _EXECUTION_MODES:
            return {
                "status": "error",
                "message": ("execution_mode must be flash, balanced, think, or auto."),
            }

        ref = f"new:{self._next_agent_number}"
        self._next_agent_number += 1
        spec = StagedAgentSpec(
            ref=ref,
            name=name,
            description=description,
            instructions=instructions,
            tool_categories=ensure_list(args.get("tool_categories")) or [],
            skills=ensure_list(args.get("skills")),
            execution_mode=execution_mode,
        )
        self.created_agents[ref] = spec
        return {
            "status": "success",
            "agent_ref": ref,
            "agent_name": name,
            "message": (
                f"Staged {name!r}. Use agent_ref {ref!r} when finalizing the Workforce."
            ),
        }

    def finalize(self, args: Mapping[str, Any]) -> dict[str, Any]:
        if self.finalized_workforce is not None:
            return {
                "status": "error",
                "message": "The Workforce has already been finalized.",
            }
        if not self.listed_existing_agents:
            return {
                "status": "error",
                "message": "Call list_available_agents before finalizing the Workforce.",
            }

        name = str(args.get("name") or "").strip()[:200].rstrip()
        description = str(args.get("description") or "").strip()
        manager_ref = str(args.get("manager_agent_ref") or "").strip()
        if not name or not description:
            return {
                "status": "error",
                "message": "Workforce name and description are required.",
            }
        if manager_ref not in self.created_agents:
            return {
                "status": "error",
                "message": (
                    "manager_agent_ref must reference a dedicated agent staged by "
                    "create_agent in this builder run."
                ),
            }

        raw_workers = args.get("workers")
        if not isinstance(raw_workers, list) or not raw_workers:
            return {
                "status": "error",
                "message": "At least one worker is required.",
            }
        if len(raw_workers) > MAX_WORKFORCE_BUILDER_WORKERS:
            return {
                "status": "error",
                "message": (
                    "The Workforce builder supports at most "
                    f"{MAX_WORKFORCE_BUILDER_WORKERS} workers."
                ),
            }

        workers: list[StagedWorkforceWorker] = []
        seen_refs: set[str] = set()
        all_refs = self.existing_agents.keys() | self.created_agents.keys()
        for raw_worker in raw_workers:
            if not isinstance(raw_worker, Mapping):
                return {"status": "error", "message": "Each worker must be an object."}
            agent_ref = str(raw_worker.get("agent_ref") or "").strip()
            assignment = str(raw_worker.get("assignment_instructions") or "").strip()
            if agent_ref not in all_refs:
                return {
                    "status": "error",
                    "message": f"Unknown worker agent_ref: {agent_ref!r}.",
                }
            if agent_ref == manager_ref:
                return {
                    "status": "error",
                    "message": "The manager agent cannot also be a worker.",
                }
            if agent_ref in seen_refs:
                return {
                    "status": "error",
                    "message": f"Worker agent_ref {agent_ref!r} is duplicated.",
                }
            if not assignment:
                return {
                    "status": "error",
                    "message": (
                        f"assignment_instructions is required for {agent_ref!r}."
                    ),
                }
            alias_value = str(raw_worker.get("alias") or "").strip()
            workers.append(
                StagedWorkforceWorker(
                    agent_ref=agent_ref,
                    alias=alias_value[:200].rstrip() or None,
                    assignment_instructions=assignment,
                    enabled=bool(raw_worker.get("enabled", True)),
                )
            )
            seen_refs.add(agent_ref)

        used_created_refs = {manager_ref} | {
            worker.agent_ref
            for worker in workers
            if worker.agent_ref in self.created_agents
        }
        unused_created_refs = set(self.created_agents) - used_created_refs
        if unused_created_refs:
            return {
                "status": "error",
                "message": (
                    "Every staged agent must be used by the Workforce. Unused refs: "
                    + ", ".join(sorted(unused_created_refs))
                ),
            }

        self.finalized_workforce = StagedWorkforceSpec(
            name=name,
            description=description,
            manager_agent_ref=manager_ref,
            workers=tuple(workers),
        )
        return {
            "status": "success",
            "message": (
                "Workforce finalized in memory. The caller may now persist the "
                "Workforce and all staged agents atomically."
            ),
            "created_agent_count": len(self.created_agents),
            "worker_count": len(workers),
        }

    def to_plan(self) -> dict[str, Any]:
        workforce = self.finalized_workforce
        if workforce is None:
            raise WorkforcePromptBuilderError(
                "The ReAct builder did not call create_workforce successfully."
            )
        manager = self.created_agents[workforce.manager_agent_ref]
        return {
            "name": workforce.name,
            "description": workforce.description,
            "manager": manager.to_dict(),
            "workers": [worker.to_dict() for worker in workforce.workers],
            "created_agents": [spec.to_dict() for spec in self.created_agents.values()],
            "warnings": [],
        }


class ListAvailableAgentsArgs(BaseModel):
    query: str | None = Field(
        default=None,
        description="Optional name or description filter for published agents.",
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=MAX_WORKFORCE_BUILDER_AGENT_RESULTS,
        description="Maximum matching agents to return for this search.",
    )


class ListAvailableAgentsResult(BaseModel):
    agents: list[dict[str, Any]]
    total_matches: int
    has_more: bool


class ListAvailableAgentsTool(AbstractBaseTool):
    category = ToolCategory.AGENT
    decision_group = "workforce_list_available_agents"

    def __init__(self, state: WorkforcePromptBuilderState) -> None:
        self._state = state
        self._visibility = ToolVisibility.PUBLIC

    @property
    def name(self) -> str:
        return "list_available_agents"

    @property
    def description(self) -> str:
        return (
            "List accessible published agents that may be reused as Workforce "
            "workers. You must call this before finalizing. Reuse an existing "
            "agent only when its capability is a clear fit; create every missing "
            "specialist with create_agent."
        )

    def args_type(self) -> Type[BaseModel]:
        return ListAvailableAgentsArgs

    def return_type(self) -> Type[BaseModel]:
        return ListAvailableAgentsResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        raise NotImplementedError("Only async execution is supported.")

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        self._state.listed_existing_agents = True
        query = str(args.get("query") or "").strip().casefold()
        agents = list(self._state.existing_agents.values())
        if query:
            agents = [
                agent
                for agent in agents
                if query
                in f"{agent.get('name', '')} {agent.get('description', '')}".casefold()
            ]
        limit = int(args.get("limit") or 20)
        return ListAvailableAgentsResult(
            agents=agents[:limit],
            total_matches=len(agents),
            has_more=len(agents) > limit,
        ).model_dump()


class StageAgentArgs(BaseModel):
    name: str = Field(description="Short, descriptive agent name.")
    description: str = Field(
        description=(
            "User-facing description of when this agent should be used. Persist "
            "this in the same language and Chinese script as the user's request."
        )
    )
    instructions: str = Field(
        description=(
            "Complete system instructions for this agent. Persist this in the "
            "same language and Chinese script as the user's request."
        )
    )
    tool_categories: list[str] = Field(
        default_factory=list,
        description="Tool categories assigned to this agent.",
    )
    skills: list[str] | None = Field(
        default=None,
        description="Built-in skills assigned to this agent.",
    )
    execution_mode: str = Field(
        default="balanced",
        description="flash, balanced, think, or auto.",
    )

    @field_validator("tool_categories", "skills", mode="before")
    @classmethod
    def parse_stringified_lists(cls, value: Any) -> Any:
        if isinstance(value, str):
            parsed = ensure_list(value)
            if parsed is not None:
                return parsed
        return value


class StageAgentResult(BaseModel):
    status: str
    agent_ref: str | None = None
    agent_name: str | None = None
    message: str


class StageAgentTool(AbstractBaseTool):
    category = ToolCategory.AGENT
    decision_group = "workforce_stage_agent"

    def __init__(self, state: WorkforcePromptBuilderState) -> None:
        self._state = state
        self._visibility = ToolVisibility.PUBLIC

    @property
    def name(self) -> str:
        return "create_agent"

    @property
    def description(self) -> str:
        return (
            "Stage one new agent for this Workforce. Call this tool separately for "
            "the dedicated manager and for every missing worker role. A successful "
            "call returns an agent_ref used by create_workforce. No database row is "
            "written until the whole Workforce is finalized, so a failed build "
            "leaves no partial agents. All names, descriptions, and instructions "
            "must use the request's language and preserve Simplified versus "
            "Traditional Chinese."
        )

    def args_type(self) -> Type[BaseModel]:
        return StageAgentArgs

    def return_type(self) -> Type[BaseModel]:
        return StageAgentResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        raise NotImplementedError("Only async execution is supported.")

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        return self._state.stage_agent(args)


class WorkforceWorkerArgs(BaseModel):
    agent_ref: str = Field(
        description="A new:<n> or existing:<database-id> reference returned by tools."
    )
    alias: str | None = None
    assignment_instructions: str = Field(
        description=(
            "Specific responsibility in the Workforce, in the request's language "
            "and Chinese script."
        )
    )
    enabled: bool = True


class FinalizeWorkforceArgs(BaseModel):
    name: str = Field(description="User-facing Workforce name.")
    description: str = Field(description="User-facing Workforce description.")
    manager_agent_ref: str = Field(
        description="Reference returned by create_agent for the dedicated manager."
    )
    workers: list[WorkforceWorkerArgs] = Field(
        description="Every worker required to fulfill the user's goal."
    )


class FinalizeWorkforceResult(BaseModel):
    status: str
    message: str
    created_agent_count: int | None = None
    worker_count: int | None = None


class FinalizeWorkforceTool(AbstractBaseTool):
    category = ToolCategory.AGENT
    decision_group = "workforce_finalize"

    def __init__(self, state: WorkforcePromptBuilderState) -> None:
        self._state = state
        self._visibility = ToolVisibility.PUBLIC

    @property
    def name(self) -> str:
        return "create_workforce"

    @property
    def description(self) -> str:
        return (
            "Finalize the staged Workforce only after list_available_agents and "
            "all required create_agent calls succeeded. The manager must be a "
            "dedicated newly staged agent, at least one distinct worker is required, "
            "and every reference and assignment is validated before persistence."
        )

    def args_type(self) -> Type[BaseModel]:
        return FinalizeWorkforceArgs

    def return_type(self) -> Type[BaseModel]:
        return FinalizeWorkforceResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        raise NotImplementedError("Only async execution is supported.")

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        raw_args = dict(args)
        raw_args["workers"] = [
            worker.model_dump() if isinstance(worker, WorkforceWorkerArgs) else worker
            for worker in raw_args.get("workers") or []
        ]
        return self._state.finalize(raw_args)


def workforce_prompt_builder_system_prompt() -> str:
    return f"""You are the Xagent Workforce Builder. Build the complete multi-agent
Workforce requested by the user through the provided tools. This is an execution
workflow, not a one-shot JSON planning task.

Required workflow:
1. Call list_available_agents to inspect reusable published agents.
2. Call create_agent once for a dedicated manager. The manager should normally use
   think mode and must be able to decompose work, delegate, compare results, recover
   from worker failure, and synthesize one final answer.
3. Identify every distinct specialist role needed for the goal. Reuse an existing
   agent only when it clearly fits. For every missing role, call create_agent
   separately. Do not collapse multiple distinct roles into one generic worker just
   to reduce tool calls.
4. If you need capability metadata, call list_available_skills and
   list_tool_categories before creating the affected agent.
5. Only after every create_agent call succeeded, call create_workforce exactly once
   with the dedicated manager ref and all worker refs. Never finish with prose before
   create_workforce returns status=success.

create_agent and create_workforce stage validated state in memory. The caller writes
all created agents and the Workforce in one transaction only after successful
finalization. A tool error must be corrected with another tool call; never claim
success when finalization did not happen.

All persisted user-facing prose passed to tools, including {_WORKFORCE_BUILDER_PERSISTED_FIELDS},
must follow the user's request language. English tool names, schemas, and tool
results do not authorize changing it.
{output_language_policy()}
{response_language_rules(subject="current user request")}
"""


async def build_workforce_prompt_plan(
    *,
    prompt: str,
    llm: BaseLLM,
    available_agents: Sequence[Mapping[str, Any]],
    compact_llm: BaseLLM | None = None,
    voice: str | None = None,
) -> dict[str, Any]:
    """Run the ReAct builder and return its validated in-memory plan."""

    if len(prompt) > MAX_WORKFORCE_PROMPT_LENGTH:
        raise WorkforcePromptBuilderError(
            f"The Workforce prompt exceeds {MAX_WORKFORCE_PROMPT_LENGTH} characters."
        )

    state = WorkforcePromptBuilderState.from_agents(available_agents)
    execution_id = f"workforce-prompt-builder-{uuid4().hex}"
    # apply_output_voice's own scoping caveat covers create_agent/
    # create_workforce's persisted arguments here - see its docstring.
    system_prompt = apply_output_voice(workforce_prompt_builder_system_prompt(), voice)
    service = AgentService(
        name="Workforce Prompt Builder",
        id=execution_id,
        task_id=execution_id,
        pattern="react",
        llm=llm,
        compact_llm=compact_llm,
        tools=[
            ListAvailableAgentsTool(state),
            StageAgentTool(state),
            FinalizeWorkforceTool(state),
            ListAvailableSkillsTool(),
            ListToolCategoriesTool(),
        ],
        system_prompt=system_prompt,
        memory_enabled=False,
        enable_workspace=False,
        react_max_iterations=WORKFORCE_BUILDER_MAX_ITERATIONS,
    )
    # This is a closed staging runtime. Generic skill discovery/load_skill is
    # intentionally disabled; capability metadata is exposed only by the two
    # explicit listing tools above.
    service.set_allowed_skills([])
    try:
        result = await service.execute_task(prompt, task_id=execution_id)
    except Exception as exc:
        raise WorkforcePromptBuilderUnavailableError(
            "The ReAct Workforce builder runtime is unavailable."
        ) from exc
    if not result.get("success"):
        status = str(result.get("status") or "failed")
        error = str(result.get("error") or result.get("output") or "").strip()
        raise WorkforcePromptBuilderError(
            f"The ReAct Workforce builder stopped with status {status}: {error}"
        )
    completion_outcome = result.get("completion_outcome")
    if completion_outcome in {"partial", "blocked"}:
        raise WorkforcePromptBuilderError(
            "The ReAct Workforce builder did not complete all requested work: "
            f"{completion_outcome}."
        )
    builder_response = extract_assistant_message(result)
    if not builder_response or not builder_response.strip():
        raise WorkforcePromptBuilderError(
            "The ReAct Workforce builder completed without a user-facing final answer."
        )
    plan = state.to_plan()
    plan["builder_response"] = builder_response.strip()
    _validate_builder_plan_language(prompt=prompt, plan=plan)
    return plan


def _validate_builder_plan_language(
    *,
    prompt: str,
    plan: Mapping[str, Any],
) -> None:
    """Reject high-confidence script drift before any builder prose is stored."""
    fields: list[tuple[str, str]] = [
        ("workforce.name", str(plan.get("name") or "")),
        ("workforce.description", str(plan.get("description") or "")),
        ("builder_response", str(plan.get("builder_response") or "")),
    ]
    for index, agent in enumerate(plan.get("created_agents") or []):
        if not isinstance(agent, Mapping):
            continue
        for field_name in ("name", "description", "instructions"):
            fields.append(
                (
                    f"created_agents[{index}].{field_name}",
                    str(agent.get(field_name) or ""),
                )
            )
    for index, worker in enumerate(plan.get("workers") or []):
        if not isinstance(worker, Mapping):
            continue
        for field_name in ("alias", "assignment_instructions"):
            fields.append(
                (
                    f"workers[{index}].{field_name}",
                    str(worker.get(field_name) or ""),
                )
            )

    for field_name, value in fields:
        mismatch = detect_prose_script_mismatch(prompt, value)
        if mismatch is None:
            continue
        raise WorkforcePromptBuilderError(
            f"The ReAct Workforce builder returned {mismatch.observed_script}-script "
            f"text in {field_name}, which does not match the request language."
        )
