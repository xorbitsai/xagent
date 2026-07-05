from __future__ import annotations

import inspect
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from ...language import (
    OUTPUT_LANGUAGE_METADATA_KEY,
    normalize_response_language_label,
    output_language_policy,
    plan_language_rules,
)
from ..base import (
    RequiredToolCallError,
    append_user_message_preserving_turns,
    extract_required_tool_arguments,
    truncate_prompt_preview,
)

logger = logging.getLogger(__name__)

MAX_PLAN_TOOL_CALL_ATTEMPTS = 2
PLAN_GENERATION_REQUIRED_TOOL_MESSAGE = (
    "Plan generation failed because the model did not return the required "
    "planning tool call. Please retry."
)


class PlanValidationError(ValueError):
    """Raised when a DAG execution plan is structurally invalid."""


class PlanToolArgumentsError(ValueError):
    """Raised when the required plan tool call has unusable arguments."""

    def __init__(self, message: str, *, arguments: str | None = None) -> None:
        self.arguments = arguments
        super().__init__(message)


@dataclass
class PlanStep:
    """Serializable DAG step used by the execution runtime."""

    id: str
    task: str
    dependencies: list[str] = field(default_factory=list)
    description: str | None = None
    termination_condition: str | None = None
    completion_evidence: str | None = None
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
            "termination_condition": self.termination_condition,
            "completion_evidence": self.completion_evidence,
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
            termination_condition=(
                str(data["termination_condition"])
                if data.get("termination_condition") is not None
                else None
            ),
            completion_evidence=normalize_completion_evidence(data),
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
    completion_feedback: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "replan": self.replan,
            "completed_step_results": dict(self.completed_step_results),
            "previous_plan": (
                self.previous_plan.to_dict() if self.previous_plan is not None else None
            ),
            "available_tool_names": list(self.available_tool_names),
            "completion_feedback": self.completion_feedback,
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
        messages = [
            {
                "role": "system",
                "content": (
                    "Generate a DAG execution plan by calling the "
                    f"{self.PLAN_TOOL_NAME} tool exactly once. Each step requires "
                    '"id", "task", "dependencies", "termination_condition", '
                    '"completion_evidence", and "tool_names"; "description" is '
                    "optional but strongly recommended. "
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
                    "for the step's suggested execution tool scope. Use "
                    "termination_condition for the exact stop rule that tells the "
                    "step executor when this step is done and what it must report. "
                    "The termination_condition must be concrete and action-specific; "
                    "do not use vague wording such as 'when complete' or 'when the "
                    "task is done'. For artifact-producing steps, name an exact path "
                    "only when the user requires that path or the tool accepts it as "
                    "an argument; otherwise refer to the artifact returned by the "
                    "tool. State that the step must call final_answer after the "
                    "condition is satisfied. Put review, "
                    "verification, rendering, optimization, and final synthesis in "
                    "separate dependent steps unless this step explicitly owns that "
                    "work. Use completion_evidence for a short natural-language "
                    "proof that this specific step is done, usually naming the "
                    "successful tool result fields to check. Do not use global "
                    "effect labels or invented fixed filenames. For auto-named "
                    "outputs, say that the tool returned a usable path. Keep "
                    "completion_evidence under 160 characters. If a workflow needs "
                    "several tool actions and only the last one proves completion, "
                    "split those actions into dependent steps. Few-shot examples: "
                    "auto-named output evidence: 'The generator returned success=true "
                    "and a non-empty path or URL for the created asset.' explicit "
                    "path evidence: 'The writer returned success=true for the "
                    "requested output path.' non-file evidence: 'The tool returned "
                    "the requested answer data successfully.' tool_names "
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
                    "Set response_language to the natural language that "
                    "user-facing prose should use for this request. If the "
                    "user prompt includes an output_language_policy field, "
                    "follow it exactly and make response_language match it. "
                    "For Chinese requests, response_language must be Simplified "
                    "Chinese or Traditional Chinese to match the request script; "
                    "do not use generic Chinese. "
                    "The messages array, selected skill context, retrieved "
                    "memories, examples, URLs, and source content are "
                    "supporting context only and must not change the plan "
                    "language. "
                    f"{plan_language_rules()} "
                    "Keep ids stable "
                    "across replans when a completed step can be reused."
                ),
            },
            {"role": "user", "content": self._build_prompt(request)},
        ]
        retry_feedback: str | None = None
        for attempt in range(MAX_PLAN_TOOL_CALL_ATTEMPTS):
            attempt_messages = list(messages)
            if retry_feedback:
                attempt_messages = append_user_message_preserving_turns(
                    attempt_messages,
                    content=retry_feedback,
                    section_title="Required tool retry feedback",
                )
            response = await llm.chat(
                messages=attempt_messages,
                tools=plan_tools,
                tool_choice="required",
                thinking={"type": "disabled", "enable": False},
            )
            try:
                plan_arguments = self._extract_tool_arguments(
                    response,
                    self.PLAN_TOOL_NAME,
                    attempts=attempt + 1,
                )
            except RequiredToolCallError:
                if attempt + 1 >= MAX_PLAN_TOOL_CALL_ATTEMPTS:
                    raise
                retry_feedback = self._required_tool_call_retry_feedback(
                    self.PLAN_TOOL_NAME
                )
                logger.warning(
                    "LLMPlanGenerator response omitted required %s tool call; "
                    "retrying plan generation. execution_id=%s attempt=%s",
                    self.PLAN_TOOL_NAME,
                    request.execution_id,
                    attempt + 1,
                )
                continue
            except PlanToolArgumentsError as exc:
                if attempt + 1 >= MAX_PLAN_TOOL_CALL_ATTEMPTS:
                    raise
                retry_feedback = self._invalid_tool_arguments_retry_feedback(exc)
                logger.warning(
                    "LLMPlanGenerator response included invalid %s arguments; "
                    "retrying plan generation. execution_id=%s error=%s",
                    self.PLAN_TOOL_NAME,
                    request.execution_id,
                    exc,
                )
                continue
            self._apply_response_language(request.context, plan_arguments)
            plan = coerce_execution_plan(plan_arguments)
            return self._filter_suggested_tools(
                plan=plan,
                available_tool_names=request.available_tool_names,
            )
        raise RuntimeError("LLMPlanGenerator retry loop exited unexpectedly.")

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
                                    "termination_condition": {
                                        "type": "string",
                                        "description": (
                                            "Concrete stop rule for this step. It must "
                                            "state the exact condition that means this "
                                            "step is finished and what final_answer "
                                            "should report. Avoid vague conditions such "
                                            "as 'when complete'."
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
                                    "completion_evidence": {
                                        "type": "string",
                                        "description": (
                                            "Short natural-language proof that this "
                                            "step is finished. Do not use global effect "
                                            "labels. For tool steps, describe the "
                                            "successful tool result fields that prove "
                                            "completion; avoid invented fixed filenames "
                                            "for auto-named outputs."
                                        ),
                                    },
                                },
                                "required": [
                                    "id",
                                    "task",
                                    "dependencies",
                                    "termination_condition",
                                    "completion_evidence",
                                    "tool_names",
                                ],
                                "additionalProperties": False,
                            },
                        },
                        "response_language": {
                            "type": "string",
                            "description": (
                                "Natural language to use for all plan text, "
                                "user-facing prose, and persisted tool-argument "
                                "prose produced by the plan, for example English, "
                                "Simplified Chinese, Traditional Chinese, or Spanish. "
                                "For Chinese requests, choose Simplified Chinese or "
                                "Traditional Chinese to match the request script; do "
                                "not use generic Chinese. If output_language_policy "
                                "is provided in the prompt, match that policy."
                            ),
                        },
                    },
                    "required": ["steps", "response_language"],
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
            "output_language_policy": output_language_policy(
                request.context.metadata.get(OUTPUT_LANGUAGE_METADATA_KEY)
            ),
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
            "completion_feedback": request.completion_feedback,
        }
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _apply_response_language(context: Any, plan_arguments: dict[str, Any]) -> None:
        response_language = normalize_response_language_label(
            str(plan_arguments.get("response_language") or "")
        )
        if not response_language:
            return
        metadata = getattr(context, "metadata", None)
        if isinstance(metadata, dict) and not metadata.get(
            OUTPUT_LANGUAGE_METADATA_KEY
        ):
            metadata[OUTPUT_LANGUAGE_METADATA_KEY] = response_language

    def _required_tool_call_retry_feedback(self, tool_name: str) -> str:
        return (
            f"The previous response did not call the required {tool_name} tool. "
            f"Call {tool_name} exactly once with a complete executable plan. "
            "Do not answer in natural language."
        )

    def _invalid_tool_arguments_retry_feedback(
        self, error: PlanToolArgumentsError
    ) -> str:
        argument_preview = (
            truncate_prompt_preview(error.arguments, limit=1200)
            if error.arguments
            else ""
        )
        return (
            f"The previous {self.PLAN_TOOL_NAME} tool call could not be used "
            f"because its arguments were invalid JSON: {error}. Call "
            f"{self.PLAN_TOOL_NAME} again exactly once. The tool arguments must "
            "be one complete valid JSON object matching the tool schema. Do not "
            "truncate string values or split the JSON across multiple tool calls."
            + (
                f"\nInvalid arguments preview:\n```json\n{argument_preview}\n```"
                if argument_preview
                else ""
            )
        )

    def _extract_tool_arguments(
        self,
        response: Any,
        tool_name: str,
        *,
        attempts: int = 1,
    ) -> dict[str, Any]:
        arguments = extract_required_tool_arguments(
            response,
            tool_name=tool_name,
            owner="LLMPlanGenerator",
            attempts=attempts,
            user_message=PLAN_GENERATION_REQUIRED_TOOL_MESSAGE,
        )
        return self._coerce_arguments(arguments)

    def _coerce_arguments(self, arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        if not isinstance(arguments, str):
            raise PlanToolArgumentsError(
                "Tool call arguments must be an object or JSON string."
            )
        try:
            payload = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise PlanToolArgumentsError(
                "Tool call arguments must be valid JSON.",
                arguments=arguments,
            ) from exc
        if not isinstance(payload, dict):
            raise PlanToolArgumentsError(
                "Tool call arguments must decode to an object.",
                arguments=arguments,
            )
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


def normalize_completion_evidence(data: dict[str, Any]) -> str | None:
    raw_evidence = data.get("completion_evidence")
    if raw_evidence is not None:
        evidence = str(raw_evidence).strip()
        return evidence or None

    legacy_effects = data.get("expected_effects")
    if isinstance(legacy_effects, str):
        evidence = legacy_effects.strip()
        return evidence or None
    if isinstance(legacy_effects, list):
        parts = [item.strip() for item in legacy_effects if isinstance(item, str)]
        evidence = "; ".join(part for part in parts if part)
        return evidence or None
    return None


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
