from __future__ import annotations

import inspect
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


class PlanValidationError(ValueError):
    """Raised when a DAG execution plan is structurally invalid."""


@dataclass
class PlanStep:
    """Serializable DAG step used by the v2 execution runtime."""

    id: str
    task: str
    dependencies: list[str] = field(default_factory=list)
    description: str | None = None
    tool_names: list[str] = field(default_factory=list)
    status: str = "pending"
    result: Any = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task,
            "dependencies": list(self.dependencies),
            "description": self.description,
            "tool_names": list(self.tool_names),
            "status": self.status,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanStep":
        tool_names = normalize_tool_names(data)
        return cls(
            id=str(data["id"]),
            task=str(data["task"]),
            dependencies=list(data.get("dependencies", [])),
            description=(
                str(data["description"])
                if data.get("description") is not None
                else None
            ),
            tool_names=tool_names,
            status=str(data.get("status", "pending")),
            result=data.get("result"),
            error=data.get("error"),
        )


@dataclass
class ExecutionPlan:
    """Minimal execution plan for DAGPattern."""

    steps: list[PlanStep]

    def validate(self) -> "ExecutionPlan":
        if not self.steps:
            raise PlanValidationError(
                "DAG execution plan must contain at least one step."
            )

        seen: set[str] = set()
        duplicates: set[str] = set()
        for step in self.steps:
            if not step.id.strip():
                raise PlanValidationError("DAG step id must not be empty.")
            if not step.task.strip():
                raise PlanValidationError(f"DAG step {step.id} task must not be empty.")
            if step.id in seen:
                duplicates.add(step.id)
            seen.add(step.id)
        if duplicates:
            duplicate_list = ", ".join(sorted(duplicates))
            raise PlanValidationError(f"DAG step ids must be unique: {duplicate_list}.")

        for step in self.steps:
            for dependency in step.dependencies:
                if dependency not in seen:
                    raise PlanValidationError(
                        f"DAG step {step.id} depends on unknown step {dependency}."
                    )

        graph = {step.id: list(step.dependencies) for step in self.steps}
        visited: set[str] = set()

        for step in self.steps:
            if step.id in visited:
                continue
            visiting: set[str] = set()
            stack: list[tuple[str, bool]] = [(step.id, False)]
            while stack:
                step_id, expanded = stack.pop()
                if expanded:
                    visiting.discard(step_id)
                    visited.add(step_id)
                    continue
                if step_id in visited:
                    continue
                if step_id in visiting:
                    raise PlanValidationError(
                        f"DAG execution plan contains a dependency cycle at {step_id}."
                    )
                visiting.add(step_id)
                stack.append((step_id, True))
                for dependency in reversed(graph[step_id]):
                    stack.append((dependency, False))
        return self

    def to_dict(self) -> dict[str, Any]:
        return {"steps": [step.to_dict() for step in self.steps]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionPlan":
        return cls(steps=[PlanStep.from_dict(item) for item in data.get("steps", [])])


@dataclass
class PlanGenerationRequest:
    """Structured input for DAG plan generation and replan flows."""

    context: Any
    execution_id: str | None = None
    replan: bool = False
    completed_step_results: dict[str, Any] = field(default_factory=dict)
    previous_plan: ExecutionPlan | None = None
    available_tool_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "replan": self.replan,
            "completed_step_results": dict(self.completed_step_results),
            "previous_plan": (
                self.previous_plan.to_dict() if self.previous_plan is not None else None
            ),
            "available_tool_names": list(self.available_tool_names),
        }


class PlanGenerator(ABC):
    """Abstract plan generator used by DAGPattern."""

    @abstractmethod
    async def generate_plan(
        self,
        *,
        request: PlanGenerationRequest,
        llm: Any,
    ) -> ExecutionPlan:
        """Build an execution plan from the current root context."""


class CallablePlanGenerator(PlanGenerator):
    """Wraps a simple callable as a PlanGenerator."""

    def __init__(self, fn: Callable[..., Any]) -> None:
        self.fn = fn

    async def generate_plan(
        self,
        *,
        request: PlanGenerationRequest,
        llm: Any,
    ) -> ExecutionPlan:
        payload = self.fn(request=request, llm=llm)
        if inspect.isawaitable(payload):
            payload = await payload
        return coerce_execution_plan(payload)


class LLMPlanGenerator(PlanGenerator):
    """Minimal LLM-backed plan generator for the v2 DAG runtime."""

    PLAN_TOOL_NAME = "generate_execution_plan"

    async def generate_plan(
        self,
        *,
        request: PlanGenerationRequest,
        llm: Any,
    ) -> ExecutionPlan:
        plan_tools = [self._plan_tool_schema()]
        response = await llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a DAG execution plan by calling the "
                        f"{self.PLAN_TOOL_NAME} tool exactly once. Each step requires "
                        '"id", "task", "dependencies", and "tool_names"; '
                        '"description" is optional but strongly recommended. '
                        "dependencies is required for every step; "
                        "use an empty array only for true entry steps that do not "
                        "need any prior output. If a step uses data, files, decisions, "
                        "analysis, or artifacts produced by another step, it must "
                        "depend on that producing step. For example, screenshot or "
                        "render steps must depend on the step that creates the HTML "
                        "or file they render, and final synthesis steps must depend "
                        "on the research or build steps they summarize. Use task as "
                        "the short node title, "
                        "description for the concrete work to perform, and tool_names "
                        "for the step's suggested execution tool scope. tool_names "
                        "must only contain exact names from available_tool_names. "
                        "Include the best matching available tools for every step "
                        "that needs tool use. Leave tool_names empty only for pure "
                        "reasoning, summarization, or formatting steps that can be "
                        "completed from provided context and dependency results. Do "
                        "not put skill names, artifact types, programming languages, "
                        "or made-up tools in tool_names. tool_names are not hard "
                        "limits, but they define the expected tool scope for the "
                        "step executor; choose them carefully so the executor does "
                        "not need to perform sibling or downstream step work. "
                        "Keep ids stable "
                        "across replans when a completed step can be reused."
                    ),
                },
                {"role": "user", "content": self._build_prompt(request)},
            ],
            tools=plan_tools,
            tool_choice="required",
            thinking={"type": "disabled", "enable": False},
        )
        plan = coerce_execution_plan(
            self._extract_tool_arguments(response, self.PLAN_TOOL_NAME)
        )
        return self._filter_suggested_tools(
            plan=plan,
            available_tool_names=request.available_tool_names,
        )

    def _filter_suggested_tools(
        self,
        *,
        plan: ExecutionPlan,
        available_tool_names: list[str],
    ) -> ExecutionPlan:
        available = {name.strip() for name in available_tool_names if name.strip()}
        if not available:
            for step in plan.steps:
                step.tool_names = []
            return plan

        for step in plan.steps:
            original_tool_names = list(step.tool_names)
            step.tool_names = [name for name in step.tool_names if name in available]
            dropped = [
                name for name in original_tool_names if name not in step.tool_names
            ]
            if dropped:
                logger.info(
                    "Dropped invalid DAG suggested tool names for step %s: %s. "
                    "Available tools: %s",
                    step.id,
                    dropped,
                    sorted(available),
                )
        return plan

    def _plan_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.PLAN_TOOL_NAME,
                "description": "Submit the DAG execution plan.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "task": {"type": "string"},
                                    "dependencies": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": (
                                            "Concrete step instructions shown in the "
                                            "execution plan."
                                        ),
                                    },
                                    "tool_names": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": (
                                            "Suggested execution tool scope for this "
                                            "step. Each name must exactly match one of the "
                                            "available_tool_names from the prompt. "
                                            "Use an empty array only when the step can "
                                            "be completed without tools."
                                        ),
                                    },
                                },
                                "required": [
                                    "id",
                                    "task",
                                    "dependencies",
                                    "tool_names",
                                ],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["steps"],
                    "additionalProperties": False,
                },
            },
        }

    def _build_prompt(self, request: PlanGenerationRequest) -> str:
        latest_messages = [
            {"role": message.role, "content": message.content}
            for message in request.context.messages
            if getattr(message, "role", None) in {"user", "assistant", "tool"}
        ]
        payload = {
            "execution_id": request.execution_id,
            "replan": request.replan,
            "messages": latest_messages,
            "retrieved_memory_context": request.context.metadata.get(
                "retrieved_memory_context"
            ),
            "selected_skill": request.context.metadata.get("selected_skill"),
            "selected_skill_context": request.context.metadata.get(
                "selected_skill_context"
            ),
            "completed_step_results": request.completed_step_results,
            "previous_plan": (
                request.previous_plan.to_dict()
                if request.previous_plan is not None
                else None
            ),
            "available_tool_names": list(request.available_tool_names),
        }
        return json.dumps(payload, ensure_ascii=False)

    def _extract_tool_arguments(self, response: Any, tool_name: str) -> dict[str, Any]:
        tool_calls = self._response_tool_calls(response)
        for tool_call in tool_calls:
            function_payload = self._function_payload(tool_call)
            if not function_payload:
                continue
            if function_payload.get("name") != tool_name:
                continue
            return self._coerce_arguments(function_payload.get("arguments", {}))
        raise ValueError(f"LLMPlanGenerator requires a {tool_name} tool call response.")

    def _response_tool_calls(self, response: Any) -> list[Any]:
        if isinstance(response, dict):
            return list(response.get("tool_calls") or [])
        return list(getattr(response, "tool_calls", []) or [])

    def _function_payload(self, tool_call: Any) -> dict[str, Any] | None:
        if isinstance(tool_call, dict):
            function_payload = tool_call.get("function")
            return function_payload if isinstance(function_payload, dict) else None
        function_payload = getattr(tool_call, "function", None)
        if function_payload is None:
            return None
        return {
            "name": getattr(function_payload, "name", None),
            "arguments": getattr(function_payload, "arguments", {}),
        }

    def _coerce_arguments(self, arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        if not isinstance(arguments, str):
            raise TypeError("Tool call arguments must be an object or JSON string.")
        try:
            payload = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError("Tool call arguments must be valid JSON.") from exc
        if not isinstance(payload, dict):
            raise TypeError("Tool call arguments must decode to an object.")
        return payload


def normalize_tool_names(data: dict[str, Any]) -> list[str]:
    raw_tools = data.get("tool_names")
    if raw_tools is None and data.get("tool_name") is not None:
        raw_tools = [data.get("tool_name")]
    if raw_tools is None:
        raw_tools = data.get("tools", [])
    if isinstance(raw_tools, str):
        raw_items: list[Any] = [raw_tools]
    elif isinstance(raw_tools, list):
        raw_items = raw_tools
    else:
        raw_items = []

    names: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        name: str | None = None
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict):
            if isinstance(item.get("function"), dict):
                name = item["function"].get("name")
            else:
                value = item.get("name") or item.get("tool_name")
                name = str(value) if value is not None else None
        if name is None:
            continue
        stripped = name.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            names.append(stripped)
    return names


def coerce_execution_plan(payload: Any) -> ExecutionPlan:
    """Normalize common plan payloads into ExecutionPlan."""
    if isinstance(payload, ExecutionPlan):
        return payload.validate()
    if isinstance(payload, dict):
        if "steps" in payload:
            return ExecutionPlan.from_dict(payload).validate()
        if {"id", "task"} <= set(payload):
            return ExecutionPlan(steps=[PlanStep.from_dict(payload)]).validate()
    if isinstance(payload, list):
        return ExecutionPlan(
            steps=[
                item
                if isinstance(item, PlanStep)
                else PlanStep.from_dict(item)
                if isinstance(item, dict)
                else PlanStep(id=f"step_{index}", task=str(item))
                for index, item in enumerate(payload)
            ]
        ).validate()
    raise TypeError(f"Unsupported plan payload: {type(payload).__name__}")
