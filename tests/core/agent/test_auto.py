from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from xagent.core.agent import (
    Agent,
    AgentRunner,
    AutoAction,
    AutoPattern,
    DAGPattern,
    ExecutionContext,
    LLMPlanGenerator,
    PatternRuntime,
    ReActPattern,
)
from xagent.core.agent.context.enrichment import MEMORY_CONTEXT_METADATA_KEY
from xagent.core.agent.language import (
    OUTPUT_LANGUAGE_METADATA_KEY,
    OUTPUT_LANGUAGE_SOURCE_METADATA_KEY,
    OUTPUT_LANGUAGE_SOURCE_PLAN,
    response_language_rules,
)
from xagent.core.agent.pattern.auto.auto import DECISION_TOOL_NAME, _AutoChildRuntime
from xagent.core.model.chat.basic.router import RouterLLM
from xagent.core.model.chat.exceptions import LLMToolProtocolError
from xagent.core.model.chat.tool_protocol import (
    ToolProtocolViolation,
    tool_protocol_error_response,
)
from xagent.core.model.chat.types import ChunkType, StreamChunk

DAG_COMPLETION_TOOL_NAME = "assess_dag_completion"


class SearchArgs(BaseModel):
    query: str
    count: int = 10


class FakeWorkspace:
    def __init__(self, task_id: str, tmp_path: Path) -> None:
        workspace_dir = tmp_path / task_id
        self.id = task_id
        self.workspace_dir = workspace_dir
        self.input_dir = workspace_dir / "input"
        self.output_dir = workspace_dir / "output"
        self.temp_dir = workspace_dir / "temp"
        self.allowed_external_dirs: list[Path] = []


class FakeWorkspaceManager:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path

    def get_or_create_workspace(
        self,
        base_dir: str,
        task_id: str,
        allowed_external_dirs: list[str] | None = None,
        scope_segments: tuple[str, ...] = (),
    ) -> FakeWorkspace:
        del base_dir, allowed_external_dirs
        return FakeWorkspace(task_id, self.tmp_path)


class FakeLLM:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self.responses and has_tool(kwargs, DAG_COMPLETION_TOOL_NAME):
            return default_completion_assessment_response(kwargs)
        return self.responses.pop(0)


class RaisingFakeLLM(FakeLLM):
    async def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class StreamingDecisionLLM:
    def __init__(self, argument_snapshots: list[str]) -> None:
        self.argument_snapshots = argument_snapshots
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        raise AssertionError("streaming decision should not call chat()")

    async def stream_chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        for arguments in self.argument_snapshots:
            yield StreamChunk(
                type=ChunkType.TOOL_CALL,
                tool_calls=[
                    {
                        "id": f"call_{DECISION_TOOL_NAME}",
                        "type": "function",
                        "function": {
                            "name": DECISION_TOOL_NAME,
                            "arguments": arguments,
                        },
                    }
                ],
            )


class OutboundCollector:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def __call__(self, payload: dict[str, Any]) -> None:
        self.events.append(payload)


class TimeoutLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        raise TimeoutError("read timed out")


class MemoryNote:
    content = "Answer simple follow-ups using the project memory."
    keywords = ["follow-up"]
    metadata = {"source": "test"}
    category = "react_memory"


class FakeMemoryStore:
    def __init__(self) -> None:
        self.searches: list[dict[str, Any]] = []

    def search(self, **kwargs: Any) -> list[MemoryNote]:
        self.searches.append(kwargs)
        return [MemoryNote()]


class FakeSkillManager:
    async def list_skills(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "auto-skill",
                "description": "Auto skill",
                "when_to_use": "Auto tasks",
            }
        ]

    async def get_skill(self, name: str) -> dict[str, Any] | None:
        if name != "auto-skill":
            return None
        return {
            "name": "auto-skill",
            "description": "Auto skill",
            "content": "Use the Auto skill instructions.",
        }


class FlakySkillManager(FakeSkillManager):
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.get_skill_calls = 0

    async def get_skill(self, name: str) -> dict[str, Any] | None:
        self.get_skill_calls += 1
        if self.get_skill_calls <= self.failures:
            raise RuntimeError("temporary skill store failure")
        return await super().get_skill(name)


class QueryMemoryNote:
    keywords: list[str] = []
    metadata = {"source": "test"}
    category = "react_memory"

    def __init__(self, content: str) -> None:
        self.content = content


class QueryMemoryStore:
    def __init__(self) -> None:
        self.searches: list[dict[str, Any]] = []

    def search(self, **kwargs: Any) -> list[QueryMemoryNote]:
        self.searches.append(kwargs)
        query = str(kwargs.get("query") or "")
        return [QueryMemoryNote(f"memory for {query}")]


def has_tool(kwargs: dict[str, Any], tool_name: str) -> bool:
    for tool_schema in kwargs.get("tools") or []:
        function = tool_schema.get("function")
        if isinstance(function, dict) and function.get("name") == tool_name:
            return True
    return False


def default_completion_assessment_response(kwargs: dict[str, Any]) -> dict[str, Any]:
    answer = "done"
    messages = kwargs.get("messages") or []
    if messages:
        try:
            payload = json.loads(str(messages[-1].get("content", "{}")))
            candidate = payload.get("candidate_output")
            if isinstance(candidate, str):
                answer = candidate
            elif candidate is not None:
                answer = json.dumps(candidate, ensure_ascii=False, default=str)
        except (AttributeError, json.JSONDecodeError):
            answer = "done"
    return {
        "tool_calls": [
            {
                "id": "call_assess_dag_completion",
                "type": "function",
                "function": {
                    "name": DAG_COMPLETION_TOOL_NAME,
                    "arguments": json.dumps(
                        {
                            "status": "completed",
                            "reason": "Completion assessment.",
                            "answer": answer,
                            "missing_work": "",
                            "replan_instruction": "",
                        }
                    ),
                },
            }
        ]
    }


class CapturingChildPattern:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        self.kwargs = kwargs
        return {"success": True, "output": "child done"}

    def get_state(self) -> dict[str, Any]:
        return {"captured": True}


class FakeSearchTool:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

        class Metadata:
            name = "zhipu_web_search"
            description = "Search the web."

        self.metadata = Metadata()

    def args_type(self) -> type[BaseModel]:
        return SearchArgs

    async def run_json_async(self, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(args)
        return {"results": [{"title": args["query"], "link": "https://example.com"}]}


def test_auto_child_runtime_forwards_clear_interrupt() -> None:
    parent_runtime = PatternRuntime()
    parent_runtime.request_interrupt("resume with user guidance")

    child_runtime = _AutoChildRuntime(
        parent=parent_runtime,
        auto_pattern=AutoPattern(),
        root_context=ExecutionContext(execution_id="auto-clear-interrupt"),
    )

    child_runtime.clear_interrupt()

    assert parent_runtime._interrupt_requested is False
    assert parent_runtime.interrupt_reason is None


@pytest.mark.asyncio
async def test_auto_child_runtime_forwards_llm_errors() -> None:
    parent_runtime = RecordingRuntime()
    context = ExecutionContext(execution_id="auto-child-llm-error")
    child_runtime = _AutoChildRuntime(
        parent=parent_runtime,
        auto_pattern=AutoPattern(),
        root_context=context,
    )

    await child_runtime.on_llm_error(
        context=context,
        error=RuntimeError("provider failed"),
        metadata={"phase": "dag_step"},
    )

    assert parent_runtime.hooks == [
        (
            "llm_error",
            {
                "error": "provider failed",
                "metadata": {"phase": "dag_step"},
            },
        )
    ]


def test_auto_child_runtime_covers_pattern_runtime_async_interface() -> None:
    missing = [
        name
        for name, method in inspect.getmembers(PatternRuntime)
        if not name.startswith("_")
        and inspect.iscoroutinefunction(method)
        and not hasattr(_AutoChildRuntime, name)
    ]

    assert missing == []


def decision_tool_response(
    action: str,
    reason: str,
    answer: str | None = None,
    requires_current_or_external_facts: bool = False,
    existing_context_sufficient: bool = True,
    evidence_basis: str = "current conversation",
    missing_verification: str = "",
) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "action": action,
        "reason": reason,
        "requires_current_or_external_facts": requires_current_or_external_facts,
        "existing_context_sufficient": existing_context_sufficient,
        "evidence_basis": evidence_basis,
        "missing_verification": missing_verification,
    }
    if answer is not None:
        arguments["answer"] = answer
    return {
        "tool_calls": [
            {
                "id": f"call_{DECISION_TOOL_NAME}",
                "type": "function",
                "function": {
                    "name": DECISION_TOOL_NAME,
                    "arguments": json.dumps(arguments),
                },
            }
        ]
    }


def load_skill_tool_response(skill_name: str) -> dict[str, Any]:
    return {
        "tool_calls": [
            {
                "id": "call_load_skill",
                "type": "function",
                "function": {
                    "name": "load_skill",
                    "arguments": json.dumps({"skill_name": skill_name}),
                },
            }
        ]
    }


def malformed_empty_missing_verification_decision_tool_response() -> dict[str, Any]:
    return {
        "tool_calls": [
            {
                "id": f"call_{DECISION_TOOL_NAME}",
                "type": "function",
                "function": {
                    "name": DECISION_TOOL_NAME,
                    "arguments": (
                        '{"action":"plan_execute","reason":"Needs DAG.",'
                        '"requires_current_or_external_facts":false,'
                        '"existing_context_sufficient":true,'
                        '"evidence_basis":"current conversation",'
                        '"missing_verification":}'
                    ),
                },
            }
        ]
    }


def truncated_final_answer_decision_tool_response() -> dict[str, Any]:
    return {
        "tool_calls": [
            {
                "id": f"call_{DECISION_TOOL_NAME}",
                "type": "function",
                "function": {
                    "name": DECISION_TOOL_NAME,
                    "arguments": (
                        '{"action":"final_answer","reason":"simple reply",'
                        '"requires_current_or_external_facts":false,'
                        '"existing_context_sufficient":true,'
                        '"evidence_basis":"current conversation",'
                        '"missing_verification":"",'
                        '"answer":"Recovered answer'
                    ),
                },
            }
        ]
    }


def unrepairable_decision_tool_response() -> dict[str, Any]:
    return {
        "tool_calls": [
            {
                "id": f"call_{DECISION_TOOL_NAME}",
                "type": "function",
                "function": {
                    "name": DECISION_TOOL_NAME,
                    "arguments": "not json at all",
                },
            }
        ]
    }


@pytest.mark.asyncio
async def test_auto_decision_sees_memory_context() -> None:
    llm = FakeLLM(
        responses=[
            decision_tool_response(
                AutoAction.FINAL_ANSWER.value,
                "simple",
                "Done from context.",
            )
        ]
    )
    context = ExecutionContext(execution_id="auto-context")
    context.add_user_message(
        "Answer from context\n\nAttached file: /private/runtime/input.txt",
        metadata={"display_message": "Answer from context"},
    )
    memory_store = FakeMemoryStore()

    result = await AutoPattern().run(
        context=context,
        tools=[],
        llm=llm,
        memory_store=memory_store,
        skill_manager=FakeSkillManager(),
    )

    assert result["success"] is True
    assert [search["filters"]["category"] for search in memory_store.searches] == [
        "react_memory",
        "general",
    ]
    assert [search["query"] for search in memory_store.searches] == [
        "Answer from context",
        "Answer from context",
    ]
    decision_messages = llm.calls[0]["messages"]
    system_context = next(
        message["content"]
        for message in decision_messages
        if message["role"] == "system"
    )
    assert "Answer simple follow-ups using the project memory." in system_context
    assert [tool["function"]["name"] for tool in llm.calls[0]["tools"]] == [
        DECISION_TOOL_NAME,
        "load_skill",
    ]


@pytest.mark.asyncio
async def test_auto_loads_matching_skill_before_selecting_pattern() -> None:
    llm = FakeLLM(
        responses=[
            load_skill_tool_response("auto-skill"),
            decision_tool_response("react", "Skill guidance favors ReAct."),
        ]
    )
    child = CapturingChildPattern()
    pattern = AutoPattern(react_pattern=child)  # type: ignore[arg-type]
    context = ExecutionContext(execution_id="auto-skill-routing")
    context.add_user_message("Use the auto skill for this task")
    manager = FakeSkillManager()
    runtime = RecordingRuntime()

    result = await pattern.run(
        context=context,
        tools=[],
        llm=llm,
        runtime=runtime,
        skill_manager=manager,
    )

    assert result["success"] is True
    assert pattern.selected_pattern == "react"
    assert pattern.routing_skill_loads == 1
    assert context.metadata["loaded_skills"] == ["auto-skill"]
    assert context.metadata["selected_skill"]["name"] == "auto-skill"
    assert (
        "Use the Auto skill instructions." in context.metadata["selected_skill_context"]
    )
    assert child.kwargs is not None
    assert child.kwargs["skill_manager"] is manager

    first_tool_names = [tool["function"]["name"] for tool in llm.calls[0]["tools"]]
    second_tool_names = [tool["function"]["name"] for tool in llm.calls[1]["tools"]]
    assert first_tool_names == [DECISION_TOOL_NAME, "load_skill"]
    assert second_tool_names == [DECISION_TOOL_NAME]
    first_prompt = llm.calls[0]["messages"][-1]["content"]
    assert "Before choosing an execution pattern" in first_prompt
    assert "call load_skill as the only tool call" in first_prompt
    second_system = next(
        message["content"]
        for message in llm.calls[1]["messages"]
        if message["role"] == "system"
    )
    assert "Selected skill guidance" in second_system
    assert "Use the Auto skill instructions." in second_system
    assert pattern.get_state()["routing_skill_loads"] == 1


@pytest.mark.asyncio
async def test_auto_retries_transient_routing_skill_load_failure() -> None:
    llm = FakeLLM(
        responses=[
            load_skill_tool_response("auto-skill"),
            load_skill_tool_response("auto-skill"),
            decision_tool_response("react", "Skill guidance favors ReAct."),
        ]
    )
    child = CapturingChildPattern()
    pattern = AutoPattern(react_pattern=child)  # type: ignore[arg-type]
    context = ExecutionContext(execution_id="auto-skill-load-retry")
    context.add_user_message("Use the auto skill for this task")
    manager = FlakySkillManager(failures=1)
    runtime = RecordingRuntime()

    result = await pattern.run(
        context=context,
        tools=[],
        llm=llm,
        runtime=runtime,
        skill_manager=manager,
    )

    assert result["success"] is True
    assert pattern.selected_pattern == "react"
    assert manager.get_skill_calls == 2
    assert pattern.routing_skill_loads == 1
    assert pattern.routing_skill_load_failures == 1
    assert [tool["function"]["name"] for tool in llm.calls[1]["tools"]] == [
        DECISION_TOOL_NAME,
        "load_skill",
    ]
    assert [checkpoint["label"] for checkpoint in runtime.checkpoints].count(
        "auto_skill_load_failed"
    ) == 1
    assert [checkpoint["label"] for checkpoint in runtime.checkpoints].count(
        "auto_skill_loaded"
    ) == 1


@pytest.mark.asyncio
async def test_auto_continues_after_routing_skill_load_retries_are_exhausted() -> None:
    llm = FakeLLM(
        responses=[
            load_skill_tool_response("auto-skill"),
            load_skill_tool_response("auto-skill"),
            decision_tool_response("react", "Continue without skill guidance."),
        ]
    )
    child = CapturingChildPattern()
    pattern = AutoPattern(react_pattern=child)  # type: ignore[arg-type]
    context = ExecutionContext(execution_id="auto-skill-load-exhausted")
    context.add_user_message("Use the auto skill for this task")
    manager = FlakySkillManager(failures=2)

    result = await pattern.run(
        context=context,
        tools=[],
        llm=llm,
        runtime=RecordingRuntime(),
        skill_manager=manager,
    )

    assert result["success"] is True
    assert pattern.selected_pattern == "react"
    assert manager.get_skill_calls == 2
    assert pattern.routing_skill_loads == 0
    assert pattern.routing_skill_load_failures == 2
    assert [tool["function"]["name"] for tool in llm.calls[2]["tools"]] == [
        DECISION_TOOL_NAME
    ]
    assert "continue without it" in llm.calls[2]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_auto_loaded_skill_guidance_reaches_dag_before_planning() -> None:
    llm = FakeLLM(
        responses=[
            load_skill_tool_response("auto-skill"),
            decision_tool_response("plan_execute", "Skill requires a plan."),
        ]
    )
    child = CapturingChildPattern()
    pattern = AutoPattern(dag_pattern=child)  # type: ignore[arg-type]
    context = ExecutionContext(execution_id="auto-skill-dag")
    context.add_user_message("Plan this auto skill task")

    result = await pattern.run(
        context=context,
        tools=[],
        llm=llm,
        runtime=RecordingRuntime(),
        skill_manager=FakeSkillManager(),
    )

    assert result["success"] is True
    assert pattern.selected_pattern == "plan_execute"
    assert child.kwargs is not None
    child_context = child.kwargs["context"]
    assert (
        "Use the Auto skill instructions."
        in child_context.metadata["selected_skill_context"]
    )
    child_system = child_context.get_messages_for_llm()[0]["content"]
    assert "Selected skill guidance" in child_system
    assert "Use the Auto skill instructions." in child_system


def plan_tool_response(
    steps: list[dict[str, Any]], response_language: str = "English"
) -> dict[str, Any]:
    return {
        "tool_calls": [
            {
                "id": "call_generate_execution_plan",
                "type": "function",
                "function": {
                    "name": "generate_execution_plan",
                    "arguments": json.dumps(
                        {"steps": steps, "response_language": response_language}
                    ),
                },
            }
        ]
    }


class TracerCheckpointStore:
    def __init__(self) -> None:
        self.by_execution_id: dict[str, dict[str, Any]] = {}
        self.checkpoints: list[dict[str, Any]] = []

    async def checkpoint(self, **payload: Any) -> None:
        self.by_execution_id[str(payload["execution_id"])] = dict(payload)
        self.checkpoints.append(dict(payload))

    async def load_latest_checkpoint(self, execution_id: str) -> dict[str, Any] | None:
        payload = self.by_execution_id.get(execution_id)
        return dict(payload) if payload is not None else None


class RecordingTracer:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def trace_event(
        self,
        event_type: Any,
        *,
        task_id: str | None = None,
        step_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> str:
        self.events.append(
            {
                "event_type": getattr(event_type, "value", str(event_type)),
                "task_id": task_id,
                "step_id": step_id,
                "data": data or {},
            }
        )
        return str(len(self.events))


class RecordingRuntime(PatternRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.hooks: list[tuple[str, dict[str, Any]]] = []

    async def on_llm_start(
        self,
        *,
        context: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        del context
        self.hooks.append(
            (
                "llm_start",
                {
                    "message_count": len(messages),
                    "tools_count": len(tools or []),
                    "metadata": metadata or {},
                },
            )
        )

    async def on_llm_end(
        self,
        *,
        context: Any,
        response: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        del context
        self.hooks.append(
            (
                "llm_end",
                {
                    "response": response,
                    "metadata": metadata or {},
                },
            )
        )

    async def on_llm_error(
        self,
        *,
        context: Any,
        error: Exception,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        del context
        self.hooks.append(
            (
                "llm_error",
                {
                    "error": str(error),
                    "metadata": metadata or {},
                },
            )
        )

    async def on_dag_step_start(
        self,
        *,
        context: Any,
        step_id: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        del context
        self.hooks.append(("dag_step_start", {"step_id": step_id, "data": data or {}}))

    async def on_dag_step_end(
        self,
        *,
        context: Any,
        step_id: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        del context
        self.hooks.append(("dag_step_end", {"step_id": step_id, "data": data or {}}))

    async def on_dag_execution(
        self,
        *,
        context: Any,
        phase: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        del context
        self.hooks.append(("dag_execution", {"phase": phase, "data": data or {}}))


@pytest.mark.asyncio
async def test_auto_pattern_final_answer_completes_without_child_pattern() -> None:
    llm = FakeLLM(
        [decision_tool_response("final_answer", "Greeting only.", answer="hi")]
    )
    pattern = AutoPattern()
    context = ExecutionContext()
    context.add_user_message("hi")
    runtime = PatternRuntime()

    result = await pattern.run(context=context, tools=[], llm=llm, runtime=runtime)

    assert result["success"] is True
    assert result["status"] == "completed"
    assert result["output"] == "hi"
    assert pattern.decision is not None
    assert pattern.decision.action == AutoAction.FINAL_ANSWER
    assert OUTPUT_LANGUAGE_METADATA_KEY not in context.metadata
    assert context.messages[-1].role == "assistant"
    assert context.messages[-1].content == "hi"
    assert len(llm.calls) == 1
    assert llm.calls[0]["tools"][0]["function"]["name"] == DECISION_TOOL_NAME
    assert llm.calls[0]["tool_choice"] == "required"
    assert llm.calls[0]["thinking"] == {"type": "disabled", "enable": False}
    assert "response_format" not in llm.calls[0]
    assert [message["role"] for message in llm.calls[0]["messages"]].count(
        "system"
    ) == 1
    first_call_roles = [message["role"] for message in llm.calls[0]["messages"]]
    assert not any(
        current == previous == "user"
        for previous, current in zip(first_call_roles, first_call_roles[1:])
    )
    decision_prompt = llm.calls[0]["messages"][-1]["content"]
    assert llm.calls[0]["messages"][-1]["role"] == "user"
    assert "must include a complete non-empty answer field" in decision_prompt
    assert (
        "available retrieved context already provide enough evidence" in decision_prompt
    )
    assert "knowledge base or RAG results" in decision_prompt
    assert "do not choose final_answer" in decision_prompt
    assert "explicitly asks to call or use an available tool" in decision_prompt
    assert "no tool or other work will happen after this routing decision" in (
        decision_prompt
    )
    assert "future tool action" in decision_prompt
    assert "ask the user to wait" in decision_prompt
    assert "not already present in the conversation" in decision_prompt
    assert "even when the user did not explicitly mention a tool" in decision_prompt
    assert "does not mean the file contents have been inspected" in decision_prompt
    assert "no prior tool result explicitly contains" in decision_prompt
    assert "Never claim to see, read, or hear attachment contents" in decision_prompt
    assert "pause for user input" in decision_prompt
    assert "Use react as the default tool-use mode" in decision_prompt
    assert "For follow-up requests" in decision_prompt
    assert "Do not choose plan_execute merely because" in decision_prompt
    assert "user-visible DAG execution" in decision_prompt
    assert "execution tools are available" in decision_prompt
    assert "response_language" not in decision_prompt
    assert "Available execution tool names" not in decision_prompt
    tool_schema = llm.calls[0]["tools"][0]["function"]
    assert "answer argument is mandatory" in tool_schema["description"]
    assert "response_language" not in tool_schema["parameters"]["properties"]
    assert "response_language" not in tool_schema["parameters"]["required"]
    assert "answer" in tool_schema["parameters"]["required"]
    answer_schema = tool_schema["parameters"]["properties"]["answer"]
    assert "Required for every decision" in answer_schema["description"]
    assert (
        "Use an empty string for react or plan_execute" in answer_schema["description"]
    )
    assert runtime.last_checkpoint is not None
    assert runtime.last_checkpoint["pattern"] == "AutoPattern"
    assert (
        "same natural language as the current user request"
        in tool_schema["description"]
    )
    assert "tool results, source documents" in answer_schema["description"]


@pytest.mark.asyncio
async def test_auto_pattern_clears_stale_output_language_before_routing() -> None:
    llm = FakeLLM(
        [decision_tool_response("final_answer", "Greeting only.", answer="hi")]
    )
    pattern = AutoPattern()
    context = ExecutionContext()
    context.metadata[OUTPUT_LANGUAGE_METADATA_KEY] = "Spanish"
    context.metadata[OUTPUT_LANGUAGE_SOURCE_METADATA_KEY] = OUTPUT_LANGUAGE_SOURCE_PLAN
    context.add_user_message("hi")
    runtime = PatternRuntime()

    result = await pattern.run(context=context, tools=[], llm=llm, runtime=runtime)

    assert result["success"] is True
    assert OUTPUT_LANGUAGE_METADATA_KEY not in context.metadata
    assert OUTPUT_LANGUAGE_SOURCE_METADATA_KEY not in context.metadata
    decision_context = "\n".join(
        str(message.get("content", "")) for message in llm.calls[0]["messages"]
    )
    assert "Output language: Spanish" not in decision_context


@pytest.mark.asyncio
async def test_auto_pattern_keeps_request_context_output_language() -> None:
    llm = FakeLLM(
        [decision_tool_response("final_answer", "Greeting only.", answer="hi")]
    )
    pattern = AutoPattern()
    context = ExecutionContext()
    context.metadata["request_context"] = {OUTPUT_LANGUAGE_METADATA_KEY: "French"}
    context.metadata[OUTPUT_LANGUAGE_METADATA_KEY] = "French"
    context.metadata[OUTPUT_LANGUAGE_SOURCE_METADATA_KEY] = OUTPUT_LANGUAGE_SOURCE_PLAN
    context.add_user_message("hi")

    result = await pattern.run(
        context=context, tools=[], llm=llm, runtime=PatternRuntime()
    )

    assert result["success"] is True
    assert context.metadata[OUTPUT_LANGUAGE_METADATA_KEY] == "French"
    assert OUTPUT_LANGUAGE_SOURCE_METADATA_KEY not in context.metadata


@pytest.mark.asyncio
async def test_auto_pattern_streams_direct_final_answer_as_tool_args_arrive() -> None:
    prefix = (
        '{"action":"final_answer","reason":"simple",'
        '"requires_current_or_external_facts":false,'
        '"existing_context_sufficient":true,'
        '"evidence_basis":"current conversation",'
        '"missing_verification":"",'
        '"answer":"'
    )
    llm = StreamingDecisionLLM(
        [
            prefix + "Hi",
            prefix + "Hi there",
            prefix + "Hi there.",
            prefix + 'Hi there."}',
        ]
    )
    collector = OutboundCollector()
    runtime = PatternRuntime(
        execution_id="auto-stream",
        outbound_message_handler=collector,
    )
    pattern = AutoPattern()
    context = ExecutionContext(execution_id="auto-stream")
    context.add_user_message("Say hello")

    result = await pattern.run(context=context, tools=[], llm=llm, runtime=runtime)

    assert result["success"] is True
    assert result["output"] == "Hi there."
    assert [event["type"] for event in collector.events] == [
        "final_answer_start",
        "final_answer_delta",
        "final_answer_delta",
        "final_answer_delta",
        "final_answer_end",
    ]
    assert [event["delta"] for event in collector.events[1:-1]] == [
        "Hi",
        " there",
        ".",
    ]
    assert collector.events[-1]["content"] == "Hi there."
    assert len({event["message_id"] for event in collector.events}) == 1
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_auto_pattern_does_not_stream_non_final_decision() -> None:
    arguments = json.dumps(
        {
            "action": "react",
            "reason": "Needs a tool.",
            "requires_current_or_external_facts": False,
            "existing_context_sufficient": True,
            "evidence_basis": "current conversation",
            "missing_verification": "",
        }
    )
    llm = StreamingDecisionLLM([arguments[:80], arguments])
    collector = OutboundCollector()
    runtime = PatternRuntime(
        execution_id="auto-react-stream",
        outbound_message_handler=collector,
    )
    child = CapturingChildPattern()
    pattern = AutoPattern(react_pattern=child)  # type: ignore[arg-type]
    context = ExecutionContext(execution_id="auto-react-stream")
    context.add_user_message("Use a tool")

    result = await pattern.run(context=context, tools=[], llm=llm, runtime=runtime)

    assert result["success"] is True
    assert result["output"] == "child done"
    assert collector.events == []
    assert pattern.selected_pattern == "react"
    assert child.kwargs is not None
    assert "allow_auto_reroute" not in child.kwargs


@pytest.mark.asyncio
async def test_auto_decision_prompt_exposes_execution_tool_names() -> None:
    llm = FakeLLM([decision_tool_response("react", "Needs an execution tool.")])
    child = CapturingChildPattern()
    pattern = AutoPattern(react_pattern=child)  # type: ignore[arg-type]
    context = ExecutionContext()
    context.add_user_message("Create an agent from a knowledge base")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "list_knowledge_bases",
                "description": "List knowledge bases",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        FakeSearchTool(),
    ]

    result = await pattern.run(
        context=context,
        tools=tools,
        llm=llm,
        runtime=PatternRuntime(),
    )

    assert result["success"] is True
    decision_call = llm.calls[0]
    assert [tool["function"]["name"] for tool in decision_call["tools"]] == [
        DECISION_TOOL_NAME
    ]
    decision_prompt = decision_call["messages"][-1]["content"]
    assert "2 execution tools are available" in decision_prompt
    assert (
        "Available execution tool names: list_knowledge_bases, zhipu_web_search."
        in decision_prompt
    )


@pytest.mark.asyncio
async def test_auto_decision_prompt_includes_grounding_rule() -> None:
    llm = FakeLLM([decision_tool_response("react", "Needs an execution tool.")])
    child = CapturingChildPattern()
    pattern = AutoPattern(react_pattern=child)  # type: ignore[arg-type]
    context = ExecutionContext()
    context.add_user_message("Build a KPI report")

    result = await pattern.run(
        context=context,
        tools=[FakeSearchTool()],
        llm=llm,
        runtime=PatternRuntime(),
    )

    assert result["success"] is True
    decision_prompt = llm.calls[0]["messages"][-1]["content"]
    assert "quantitative data" in decision_prompt
    assert "illustrative placeholders" in decision_prompt
    assert "invented values" in decision_prompt
    assert decision_prompt.count("## FINAL DELIVERABLE FILE REFERENCES") == 1
    assert decision_prompt.index(
        "If the answer would need such unsupported specifics"
    ) < decision_prompt.index("## FINAL DELIVERABLE FILE REFERENCES")
    assert "get_workspace_output_files" not in decision_prompt
    assert "You must classify whether" in decision_prompt
    assert "You must also classify whether" not in decision_prompt
    answer_description = pattern._decision_tool_schema()["function"]["parameters"][
        "properties"
    ]["answer"]["description"]
    assert "## FINAL DELIVERABLE FILE REFERENCES" not in answer_description
    assert "exact markdown_link" in answer_description
    assert "get_workspace_output_files" not in answer_description
    # Routes through the classification field so _normalize_decision's
    # deterministic fallback catches it, not just the model's routing choice.
    assert "set existing_context_sufficient=false and choose react" in decision_prompt
    assert "use an appropriate tool" not in decision_prompt


@pytest.mark.asyncio
async def test_auto_pattern_interrupt_before_decision_skips_llm_call() -> None:
    llm = FakeLLM(
        [decision_tool_response("final_answer", "Should not be called.", answer="hi")]
    )
    pattern = AutoPattern()
    context = ExecutionContext()
    context.add_user_message("hi")
    runtime = PatternRuntime()
    runtime.request_interrupt("paused by test")

    result = await pattern.run(context=context, tools=[], llm=llm, runtime=runtime)

    assert result["success"] is False
    assert result["status"] == "interrupted"
    assert result["interrupt_reason"] == "paused by test"
    assert len(llm.calls) == 0
    assert runtime.last_checkpoint is not None
    assert runtime.last_checkpoint["label"] == "auto_interrupted"
    assert runtime.last_checkpoint["metadata"] == {
        "safe_point": "auto_before_decision",
        "reason": "paused by test",
    }


@pytest.mark.asyncio
async def test_auto_pattern_does_not_emit_general_task_start_or_completion() -> None:
    llm = FakeLLM(
        [decision_tool_response("final_answer", "Greeting only.", answer="hi")]
    )
    tracer = RecordingTracer()
    pattern = AutoPattern()
    context = ExecutionContext(execution_id="auto-final")
    context.add_user_message("hi")

    result = await pattern.run(
        context=context,
        tools=[],
        llm=llm,
        runtime=PatternRuntime(tracer=tracer),
    )

    assert result["success"] is True
    assert {event["event_type"] for event in tracer.events} == {
        "action_start_llm",
        "action_end_llm",
        "task_update_general",
    }


@pytest.mark.asyncio
async def test_auto_react_repetition_stays_in_single_react_trace() -> None:
    llm = FakeLLM(
        [
            decision_tool_response("react", "Needs current search."),
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "search_1",
                        "function": {
                            "name": "zhipu_web_search",
                            "arguments": '{"query":"AI news","count":10}',
                        },
                    }
                ],
            },
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "search_2",
                        "function": {
                            "name": "zhipu_web_search",
                            "arguments": '{"query":"AI news latest","count":5}',
                        },
                    }
                ],
            },
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "decision_1",
                        "function": {
                            "name": "react_decision",
                            "arguments": (
                                '{"action":"final_answer",'
                                '"reason":"已有结果足够回答。"}'
                            ),
                        },
                    }
                ],
            },
            "可以基于已有搜索结果回答。",
        ]
    )
    tracer = RecordingTracer()
    runtime = PatternRuntime(tracer=tracer)
    pattern = AutoPattern(
        react_pattern=ReActPattern(
            max_iterations=4,
            repeated_tool_decision_after_consecutive_tool_calls=2,
        )
    )
    context = ExecutionContext(execution_id="auto-react-repeat")
    context.add_user_message("总结最近 AI 新闻")
    tool = FakeSearchTool()

    result = await pattern.run(
        context=context,
        tools=[tool],
        llm=llm,
        runtime=runtime,
    )

    event_types = [event["event_type"] for event in tracer.events]
    assert result["success"] is True
    assert result["response"] == "可以基于已有搜索结果回答。"
    assert len(tool.calls) == 2
    assert event_types.count("task_start_react") == 1
    assert event_types.count("task_end_react") == 1
    assert "auto_child_reroute" not in [
        checkpoint["label"] for checkpoint in runtime.checkpoints
    ]


@pytest.mark.asyncio
async def test_auto_interrupt_cancels_child_react_tool() -> None:
    class SlowVisionTool:
        name = "understand_images"
        description = "Analyze an image."

        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()

        async def ainvoke(self, _args: dict[str, Any]) -> Any:
            self.started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
            return {"success": True, "answer": "never"}

    llm = FakeLLM(
        [
            decision_tool_response("react", "Image inspection needs a tool."),
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "vision-1",
                        "function": {
                            "name": "understand_images",
                            "arguments": json.dumps(
                                {
                                    "images": "file-id",
                                    "question": "What is shown?",
                                }
                            ),
                        },
                    }
                ],
            },
        ]
    )
    runtime = PatternRuntime(execution_id="auto-cancel-tool")
    pattern = AutoPattern(react_pattern=ReActPattern(max_iterations=2))
    context = ExecutionContext(execution_id="auto-cancel-tool")
    context.add_user_message("Inspect this image.")
    tool = SlowVisionTool()

    task = asyncio.create_task(
        pattern.run(context=context, tools=[tool], llm=llm, runtime=runtime)
    )
    await tool.started.wait()
    runtime.request_interrupt("paused by websocket")
    result = await task

    assert result["success"] is False
    assert result["status"] == "interrupted"
    assert result["interrupt_reason"] == "paused by websocket"
    assert tool.cancelled.is_set()


@pytest.mark.asyncio
async def test_auto_pattern_react_decision_delegates_to_react() -> None:
    llm = FakeLLM(
        [
            decision_tool_response("react", "Ordinary response."),
            "react done",
        ]
    )
    pattern = AutoPattern()
    context = ExecutionContext()
    context.add_user_message("Say done")
    runtime = RecordingRuntime()

    result = await pattern.run(context=context, tools=[], llm=llm, runtime=runtime)

    assert result["success"] is True
    assert result["output"] == "react done"
    assert result["auto_decision"] == {
        "action": "react",
        "reason": "Ordinary response.",
        "requires_current_or_external_facts": False,
        "existing_context_sufficient": True,
        "evidence_basis": "current conversation",
        "missing_verification": "",
    }
    assert OUTPUT_LANGUAGE_METADATA_KEY not in context.metadata
    assert pattern.selected_pattern == "react"
    assert pattern.react_state is not None
    assert runtime.last_checkpoint is not None
    assert runtime.last_checkpoint["pattern_state"]["selected_pattern"] == "react"
    assert [hook for hook, _ in runtime.hooks] == [
        "llm_start",
        "llm_end",
        "llm_start",
        "llm_end",
    ]
    assert runtime.hooks[0][1]["metadata"] == {"phase": "auto_decision"}


@pytest.mark.asyncio
async def test_auto_pattern_falls_back_to_the_main_llm_for_compaction() -> None:
    """Deliberate reversal of a previous policy.

    This test used to assert the opposite -- that an unconfigured compact slot
    left the main model untouched, so compaction cost nothing. What it cost
    instead was the summary: ``PatternRuntime.compact_context_if_needed``
    skips summarization without a compact LLM and drops all but the last few
    messages, so the resumed conversation loses the tool observations that
    explain what the agent already did. Spending main-model tokens is the
    lesser price, and an empty slot is ordinary rather than exceptional --
    agent preview and delegated sub-agents resolve it themselves and validate
    only the default model.
    """
    llm = FakeLLM(
        [
            {"content": "summary for the routing decision"},
            decision_tool_response("react", "Ordinary response."),
            {"content": "summary for the child pattern"},
            "react done",
        ]
    )
    pattern = AutoPattern()
    context = ExecutionContext()
    context.compact_config.threshold = 1
    context.add_user_message("Say done " + "x" * 200)
    runtime = RecordingRuntime()

    result = await pattern.run(context=context, tools=[], llm=llm, runtime=runtime)

    assert result["success"] is True
    assert result["output"] == "react done"
    # Four calls: Auto compacts before routing, routes, then the child ReAct
    # pattern compacts again (this fixture's threshold of 1 keeps every
    # context over budget) before its own call.
    assert len(llm.calls) == 4
    compaction_calls = [
        call
        for call in llm.calls
        if "Compress agent conversation history"
        in call["messages"][0].get("content", "")
    ]
    assert len(compaction_calls) == 2
    assert has_tool(llm.calls[1], DECISION_TOOL_NAME)


@pytest.mark.asyncio
async def test_auto_pattern_passes_memory_to_child_pattern() -> None:
    child = CapturingChildPattern()
    memory_store = FakeMemoryStore()
    llm = FakeLLM(
        [
            decision_tool_response("react", "Needs child execution."),
        ]
    )
    pattern = AutoPattern(react_pattern=child)  # type: ignore[arg-type]
    context = ExecutionContext()
    context.add_user_message("Use context and then execute")

    result = await pattern.run(
        context=context,
        tools=[],
        llm=llm,
        memory_store=memory_store,
        memory_similarity_threshold=0.42,
    )

    assert result["success"] is True
    assert child.kwargs is not None
    assert child.kwargs["memory_store"] is memory_store
    assert child.kwargs["memory_similarity_threshold"] == 0.42


@pytest.mark.asyncio
async def test_auto_pattern_emits_llm_error_for_decision_failure() -> None:
    llm = TimeoutLLM()
    pattern = AutoPattern()
    context = ExecutionContext()
    context.add_user_message("Use Python")
    runtime = RecordingRuntime()

    with pytest.raises(TimeoutError):
        await pattern.run(context=context, tools=[], llm=llm, runtime=runtime)

    assert [hook for hook, _ in runtime.hooks] == ["llm_start", "llm_error"]
    assert runtime.hooks[0][1]["metadata"] == {"phase": "auto_decision"}
    assert runtime.hooks[1][1]["metadata"] == {"phase": "auto_decision"}
    assert "read timed out" in runtime.hooks[1][1]["error"]


@pytest.mark.asyncio
async def test_auto_pattern_plan_execute_decision_delegates_to_dag() -> None:
    llm = FakeLLM(
        [
            decision_tool_response("plan_execute", "Needs a plan."),
            plan_tool_response([{"id": "answer", "task": "Answer directly"}]),
            "dag done",
        ]
    )
    pattern = AutoPattern(dag_pattern=DAGPattern(LLMPlanGenerator()))
    context = ExecutionContext()
    context.add_user_message("Plan then answer")
    runtime = RecordingRuntime()

    result = await pattern.run(context=context, tools=[], llm=llm, runtime=runtime)

    assert result["success"] is True
    assert result["output"] == "dag done"
    assert result["step_results"] == {"answer": "dag done"}
    assert result["auto_decision"] == {
        "action": "plan_execute",
        "reason": "Needs a plan.",
        "requires_current_or_external_facts": False,
        "existing_context_sufficient": True,
        "evidence_basis": "current conversation",
        "missing_verification": "",
    }
    assert OUTPUT_LANGUAGE_METADATA_KEY not in context.metadata
    assert OUTPUT_LANGUAGE_SOURCE_METADATA_KEY not in context.metadata
    assert pattern.selected_pattern == "plan_execute"
    assert pattern.dag_state is not None
    assert runtime.last_checkpoint is not None
    assert runtime.last_checkpoint["pattern_state"]["selected_pattern"] == (
        "plan_execute"
    )
    hook_names = [hook for hook, _ in runtime.hooks]
    assert "dag_execution" in hook_names
    assert "dag_step_start" in hook_names
    assert "dag_step_end" in hook_names
    assert hook_names.count("llm_start") >= 1
    assert hook_names.count("llm_end") >= 1


@pytest.mark.asyncio
async def test_auto_pattern_repairs_empty_missing_verification_argument() -> None:
    llm = FakeLLM(
        [
            malformed_empty_missing_verification_decision_tool_response(),
            plan_tool_response([{"id": "answer", "task": "Answer directly"}]),
            "dag done",
        ]
    )
    pattern = AutoPattern(dag_pattern=DAGPattern(LLMPlanGenerator()))
    context = ExecutionContext()
    context.add_user_message("Plan then answer")

    result = await pattern.run(context=context, tools=[], llm=llm)

    assert result["success"] is True
    assert result["auto_decision"]["action"] == "plan_execute"
    assert result["auto_decision"]["missing_verification"] == ""


@pytest.mark.asyncio
async def test_auto_pattern_retries_truncated_final_answer_arguments() -> None:
    llm = FakeLLM(
        [
            truncated_final_answer_decision_tool_response(),
            decision_tool_response(
                "final_answer",
                "Retry produced the full answer.",
                answer="Complete answer after retry.",
            ),
        ]
    )
    pattern = AutoPattern()
    context = ExecutionContext()
    context.add_user_message("Continue")
    runtime = RecordingRuntime()

    result = await pattern.run(context=context, tools=[], llm=llm, runtime=runtime)

    assert result["success"] is True
    assert result["output"] == "Complete answer after retry."
    assert result["auto_decision"]["action"] == "final_answer"
    assert len(llm.calls) == 2
    retry_messages = llm.calls[1]["messages"]
    assert "truncated" in retry_messages[-1]["content"]
    assert "Recovered answer" in retry_messages[-1]["content"]
    assert any(
        checkpoint["label"] == "auto_decision_retry"
        for checkpoint in runtime.checkpoints
    )


@pytest.mark.asyncio
async def test_auto_pattern_does_not_stream_rejected_final_answer_candidate() -> None:
    llm = FakeLLM(
        [
            truncated_final_answer_decision_tool_response(),
            decision_tool_response(
                "final_answer",
                "Retry produced the full answer.",
                answer="Complete answer after retry.",
            ),
        ]
    )
    pattern = AutoPattern()
    context = ExecutionContext(execution_id="auto-retry-stream")
    context.add_user_message("Continue")
    collector = OutboundCollector()
    runtime = PatternRuntime(
        execution_id="auto-retry-stream",
        outbound_message_handler=collector,
    )

    result = await pattern.run(context=context, tools=[], llm=llm, runtime=runtime)

    assert result["success"] is True
    assert result["output"] == "Complete answer after retry."
    assert [event["type"] for event in collector.events] == [
        "final_answer_start",
        "final_answer_delta",
        "final_answer_end",
    ]
    assert collector.events[1]["delta"] == "Complete answer after retry."


@pytest.mark.asyncio
async def test_auto_pattern_retries_unrepairable_decision_arguments() -> None:
    llm = FakeLLM(
        [
            unrepairable_decision_tool_response(),
            decision_tool_response(
                "final_answer",
                "Retry produced valid arguments.",
                answer="after retry",
            ),
        ]
    )
    pattern = AutoPattern()
    context = ExecutionContext()
    context.add_user_message("Continue")
    runtime = RecordingRuntime()

    result = await pattern.run(context=context, tools=[], llm=llm, runtime=runtime)

    assert result["success"] is True
    assert result["output"] == "after retry"
    assert len(llm.calls) == 2
    retry_messages = llm.calls[1]["messages"]
    assert "invalid JSON" in retry_messages[-1]["content"]
    assert "not json at all" in retry_messages[-1]["content"]
    llm_start_metadata = [
        hook[1]["metadata"] for hook in runtime.hooks if hook[0] == "llm_start"
    ]
    assert llm_start_metadata == [
        {"phase": "auto_decision"},
        {"phase": "auto_decision", "attempt": 2},
    ]
    assert any(
        checkpoint["label"] == "auto_decision_retry"
        for checkpoint in runtime.checkpoints
    )


@pytest.mark.asyncio
async def test_auto_pattern_resume_reuses_existing_decision() -> None:
    llm = FakeLLM(["react after resume"])
    pattern = AutoPattern(react_pattern=ReActPattern())
    pattern.load_state(
        {
            "status": "running",
            "decision": {"action": "react", "reason": "Already decided."},
            "selected_pattern": "react",
        }
    )
    context = ExecutionContext()
    context.add_user_message("Continue")

    result = await pattern.run(
        context=context,
        tools=[],
        llm=llm,
        runtime=PatternRuntime(),
    )

    assert result["success"] is True
    assert result["output"] == "react after resume"
    assert len(llm.calls) == 1
    assert llm.calls[0]["tools"] is not None


@pytest.mark.asyncio
async def test_auto_pattern_final_answer_resume_redecides_after_new_user_message() -> (
    None
):
    first_llm = FakeLLM(
        [decision_tool_response("final_answer", "Original answer.", answer="old")]
    )
    first_pattern = AutoPattern()
    context = ExecutionContext()
    context.add_user_message("first question")
    runtime = PatternRuntime()

    def interrupt_after_decision() -> bool:
        return bool(
            runtime.last_checkpoint
            and runtime.last_checkpoint.get("label") == "auto_after_decision"
        )

    runtime.interrupt_checker = interrupt_after_decision

    interrupted = await first_pattern.run(
        context=context,
        tools=[],
        llm=first_llm,
        runtime=runtime,
    )

    assert interrupted["status"] == "interrupted"
    assert runtime.last_checkpoint is not None
    assert runtime.last_checkpoint["label"] == "auto_interrupted"

    resumed_context = ExecutionContext.from_dict(runtime.last_checkpoint["context"])
    resumed_context.add_user_message("replacement question")
    resumed_pattern = AutoPattern()
    resumed_pattern.load_state(runtime.last_checkpoint["pattern_state"])
    resumed_llm = FakeLLM(
        [decision_tool_response("final_answer", "Replacement answer.", answer="new")]
    )

    resumed = await resumed_pattern.run(
        context=resumed_context,
        tools=[],
        llm=resumed_llm,
        runtime=PatternRuntime(),
    )

    assert resumed["success"] is True
    assert resumed["output"] == "new"
    assert len(resumed_llm.calls) == 1


@pytest.mark.asyncio
async def test_auto_pattern_final_answer_redecision_refreshes_enrichment() -> None:
    first_llm = FakeLLM(
        [decision_tool_response("final_answer", "Original answer.", answer="old")]
    )
    first_pattern = AutoPattern()
    context = ExecutionContext()
    context.add_user_message("first question")
    runtime = PatternRuntime()
    memory_store = QueryMemoryStore()

    def interrupt_after_decision() -> bool:
        return bool(
            runtime.last_checkpoint
            and runtime.last_checkpoint.get("label") == "auto_after_decision"
        )

    runtime.interrupt_checker = interrupt_after_decision

    interrupted = await first_pattern.run(
        context=context,
        tools=[],
        llm=first_llm,
        runtime=runtime,
        memory_store=memory_store,
    )

    assert interrupted["status"] == "interrupted"
    assert runtime.last_checkpoint is not None

    resumed_context = ExecutionContext.from_dict(runtime.last_checkpoint["context"])
    resumed_context.add_user_message("replacement question")
    resumed_pattern = AutoPattern()
    resumed_pattern.load_state(runtime.last_checkpoint["pattern_state"])
    resumed_llm = FakeLLM(
        [decision_tool_response("final_answer", "Replacement answer.", answer="new")]
    )

    resumed = await resumed_pattern.run(
        context=resumed_context,
        tools=[],
        llm=resumed_llm,
        runtime=PatternRuntime(),
        memory_store=memory_store,
    )

    assert resumed["success"] is True
    assert resumed["output"] == "new"
    assert [search["query"] for search in memory_store.searches] == [
        "first question",
        "first question",
        "replacement question",
        "replacement question",
    ]
    resumed_system_context = next(
        message["content"]
        for message in resumed_llm.calls[0]["messages"]
        if message["role"] == "system"
    )
    assert "memory for replacement question" in resumed_system_context
    assert "memory for first question" not in resumed_system_context


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "protocol_code",
    ["malformed_tool_arguments", "unavailable_tool_call"],
)
async def test_auto_pattern_retries_provider_routing_protocol_errors(
    protocol_code: str,
) -> None:
    llm = RaisingFakeLLM(
        [
            LLMToolProtocolError(
                provider="deepseek",
                code=protocol_code,
                message="invalid routing tool call",
            ),
            decision_tool_response(
                "final_answer",
                "Recovered after protocol error.",
                answer="Recovered routing answer.",
            ),
        ]
    )
    pattern = AutoPattern()
    context = ExecutionContext(execution_id="auto-protocol-retry")
    context.add_user_message("Answer from the current context")
    runtime = RecordingRuntime()

    result = await pattern.run(
        context=context,
        tools=[],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is True
    assert result["response"] == "Recovered routing answer."
    assert len(llm.calls) == 2
    retry_message = llm.calls[1]["messages"][-1]["content"]
    assert protocol_code.split("_", 1)[0] in retry_message
    assert "one complete JSON object" in retry_message
    assert DECISION_TOOL_NAME in retry_message
    llm_error_hooks = [
        payload for name, payload in runtime.hooks if name == "llm_error"
    ]
    assert llm_error_hooks[-1]["metadata"]["protocol_code"] == protocol_code


@pytest.mark.asyncio
async def test_auto_pattern_retries_missing_decision_tool_call() -> None:
    llm = FakeLLM(
        [
            "not a tool call",
            decision_tool_response(
                "final_answer",
                "Greeting only.",
                answer="Complete answer after retry.",
            ),
        ]
    )
    pattern = AutoPattern()
    context = ExecutionContext()
    context.add_user_message("Continue")

    result = await pattern.run(
        context=context,
        tools=[],
        llm=llm,
        runtime=PatternRuntime(),
    )

    assert result["success"] is True
    assert result["response"] == "Complete answer after retry."
    assert len(llm.calls) == 2
    retry_roles = [message["role"] for message in llm.calls[1]["messages"]]
    assert not any(
        current == previous == "user"
        for previous, current in zip(retry_roles, retry_roles[1:])
    )
    retry_message = llm.calls[1]["messages"][-1]["content"]
    assert f"did not call the required {DECISION_TOOL_NAME} tool" in retry_message


@pytest.mark.asyncio
async def test_auto_pattern_retries_unavailable_tool_call_as_routing_decision() -> None:
    llm = FakeLLM(
        [
            tool_protocol_error_response(
                ToolProtocolViolation(
                    provider="deepseek",
                    code="unavailable_tool_call",
                    message="DeepSeek returned unavailable tool call 'web_search'.",
                )
            ),
            decision_tool_response(
                "final_answer",
                "No tool work is required.",
                answer="Recovered routing answer.",
            ),
        ]
    )
    pattern = AutoPattern()
    context = ExecutionContext()
    context.add_user_message("Answer from the current context")

    result = await pattern.run(context=context, tools=[], llm=llm)

    assert result["success"] is True
    assert result["response"] == "Recovered routing answer."
    assert len(llm.calls) == 2
    retry_message = llm.calls[1]["messages"][-1]["content"]
    assert f"did not call the required {DECISION_TOOL_NAME} tool" in retry_message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "protocol_code",
    ["malformed_tool_arguments", "unavailable_tool_call"],
)
async def test_auto_pattern_retries_provider_routing_protocol_errors_after_skill_load(
    protocol_code: str,
) -> None:
    llm = RaisingFakeLLM(
        [
            load_skill_tool_response("auto-skill"),
            {"tool_calls": []},
            LLMToolProtocolError(
                provider="deepseek",
                code=protocol_code,
                message="invalid routing tool call",
            ),
            decision_tool_response("react", "Recovered after protocol error."),
        ]
    )
    child = CapturingChildPattern()
    pattern = AutoPattern(react_pattern=child)  # type: ignore[arg-type]
    context = ExecutionContext(execution_id="auto-protocol-retry")
    context.add_user_message("Use the auto skill for this task")
    runtime = RecordingRuntime()

    result = await pattern.run(
        context=context,
        tools=[],
        llm=llm,
        runtime=runtime,
        skill_manager=FakeSkillManager(),
    )

    assert result["success"] is True
    assert pattern.selected_pattern == "react"
    assert pattern.routing_skill_loads == 1
    assert len(llm.calls) == 4
    assert [tool["function"]["name"] for tool in llm.calls[1]["tools"]] == [
        DECISION_TOOL_NAME
    ]
    assert [tool["function"]["name"] for tool in llm.calls[2]["tools"]] == [
        DECISION_TOOL_NAME
    ]
    assert [tool["function"]["name"] for tool in llm.calls[3]["tools"]] == [
        DECISION_TOOL_NAME
    ]
    retry_message = llm.calls[3]["messages"][-1]["content"]
    assert protocol_code.split("_", 1)[0] in retry_message
    assert "one complete JSON object" in retry_message
    assert DECISION_TOOL_NAME in retry_message
    assert any(
        checkpoint["label"] == "auto_decision_retry"
        and checkpoint["metadata"].get("protocol_code") == protocol_code
        for checkpoint in runtime.checkpoints
    )
    llm_error_hooks = [
        payload for name, payload in runtime.hooks if name == "llm_error"
    ]
    assert llm_error_hooks[-1]["metadata"]["protocol_code"] == protocol_code


@pytest.mark.asyncio
async def test_auto_pattern_missing_decision_tool_call_fails() -> None:
    llm = FakeLLM(["not a tool call", {"tool_calls": []}])
    pattern = AutoPattern()
    context = ExecutionContext()
    context.add_user_message("Continue")
    runtime = PatternRuntime()

    result = await pattern.run(
        context=context,
        tools=[],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is False
    assert result["status"] == "failed"
    assert result["failure_reason"] == "missing_required_tool_call"
    assert result["required_tool_name"] == DECISION_TOOL_NAME
    assert result["attempts"] == 2
    assert result["error"] == (
        "Auto routing failed because the model did not return the required "
        "decision tool call. Please retry."
    )
    assert "AutoPattern decision requires" not in result["error"]
    assert pattern.last_result == result
    assert runtime.last_checkpoint is not None
    assert runtime.last_checkpoint["label"] == "auto_decision_failed"
    assert runtime.last_checkpoint["metadata"]["failure_reason"] == (
        "missing_required_tool_call"
    )
    assert runtime.last_checkpoint["metadata"]["required_tool_name"] == (
        DECISION_TOOL_NAME
    )
    assert runtime.last_checkpoint["metadata"]["attempts"] == 2


@pytest.mark.asyncio
async def test_auto_pattern_unknown_action_fails() -> None:
    llm = FakeLLM(
        [
            decision_tool_response("unknown", "Bad action."),
        ]
    )
    pattern = AutoPattern()
    context = ExecutionContext()
    context.add_user_message("Continue")

    with pytest.raises(ValueError, match="Invalid AutoPattern action: unknown"):
        await pattern.run(
            context=context,
            tools=[],
            llm=llm,
            runtime=PatternRuntime(),
        )


@pytest.mark.asyncio
async def test_auto_pattern_empty_final_answer_falls_back_to_react() -> None:
    llm = FakeLLM(
        [
            decision_tool_response("final_answer", "No answer.", answer="  "),
            "react fallback",
        ]
    )
    pattern = AutoPattern()
    context = ExecutionContext(execution_id="auto-empty-final-candidate")
    context.add_user_message("Continue")
    collector = OutboundCollector()
    runtime = PatternRuntime(
        execution_id="auto-empty-final-candidate",
        outbound_message_handler=collector,
    )

    result = await pattern.run(
        context=context,
        tools=[],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is True
    assert result["output"] == "react fallback"
    assert result["auto_decision"] == {
        "action": "react",
        "reason": (
            "AutoPattern selected final_answer without a non-empty answer; "
            "falling back to react."
        ),
        "requires_current_or_external_facts": False,
        "existing_context_sufficient": True,
        "evidence_basis": "",
        "missing_verification": "",
    }
    assert OUTPUT_LANGUAGE_METADATA_KEY not in context.metadata
    assert pattern.selected_pattern == "react"
    assert len(llm.calls) == 2
    assert collector.events == []


@pytest.mark.asyncio
async def test_auto_pattern_final_answer_requiring_external_facts_falls_back_to_react() -> (
    None
):
    llm = FakeLLM(
        [
            decision_tool_response(
                "final_answer",
                "Recent public facts can be answered from memory.",
                answer="Unsupported factual answer.",
                requires_current_or_external_facts=True,
                existing_context_sufficient=False,
                evidence_basis="memory only",
                missing_verification="Need current public-source verification.",
            ),
            "verified through react",
        ]
    )
    pattern = AutoPattern()
    context = ExecutionContext(execution_id="auto-external-facts-candidate")
    context.add_user_message("总结最近 AI 圈子的供应链攻击")
    collector = OutboundCollector()
    runtime = PatternRuntime(
        execution_id="auto-external-facts-candidate",
        outbound_message_handler=collector,
    )

    result = await pattern.run(
        context=context,
        tools=[],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is True
    assert result["output"] == "verified through react"
    assert result["auto_decision"] == {
        "action": "react",
        "reason": (
            "AutoPattern selected final_answer for a request requiring current or "
            "external facts without sufficient supporting context; falling back to "
            "react."
        ),
        "requires_current_or_external_facts": True,
        "existing_context_sufficient": False,
        "evidence_basis": "memory only",
        "missing_verification": "Need current public-source verification.",
    }
    assert OUTPUT_LANGUAGE_METADATA_KEY not in context.metadata
    assert pattern.selected_pattern == "react"
    assert len(llm.calls) == 2
    assert collector.events == []


@pytest.mark.asyncio
async def test_auto_pattern_plan_execute_without_dag_fails() -> None:
    llm = FakeLLM(
        [
            decision_tool_response("plan_execute", "Needs DAG."),
        ]
    )
    pattern = AutoPattern()
    context = ExecutionContext()
    context.add_user_message("Continue")

    with pytest.raises(ValueError, match="DAGPattern"):
        await pattern.run(
            context=context,
            tools=[],
            llm=llm,
            runtime=PatternRuntime(),
        )


def test_auto_pattern_get_execution_snapshot_builds_react_child_frame() -> None:
    pattern = AutoPattern()
    pattern.load_state(
        {
            "status": "running",
            "decision": {"action": "react", "reason": "Needs ReAct."},
            "selected_pattern": "react",
            "react_state": {"iteration": 1},
        }
    )
    context = ExecutionContext(execution_id="snap-1")
    context.add_user_message("Continue")

    snapshot = pattern.get_execution_snapshot(context)

    assert snapshot["root_execution_id"] == "snap-1"
    assert snapshot["status"] == "running"
    assert snapshot["active_frame_ids"] == ["snap-1:auto", "snap-1:auto:react"]
    assert snapshot["control_state"] == {"selected_pattern": "react"}
    root_frame = snapshot["frames"]["snap-1:auto"]
    assert root_frame["pattern_type"] == "auto"
    assert root_frame["children"] == ["snap-1:auto:react"]
    assert root_frame["active_child_id"] == "snap-1:auto:react"
    child_frame = snapshot["frames"]["snap-1:auto:react"]
    assert child_frame["parent_frame_id"] == "snap-1:auto"
    assert child_frame["pattern_type"] == "react"
    assert child_frame["pattern_state"] == {"iteration": 1}


@pytest.mark.asyncio
async def test_auto_pattern_react_resume_from_tracer_does_not_redecide(
    tmp_path: Path,
) -> None:
    tracer = TracerCheckpointStore()
    execution_id = "auto-react-restart"
    first_agent = Agent(
        name="auto",
        patterns=[AutoPattern()],
        llm=None,
    )
    first_runner = AgentRunner(
        agent=first_agent,
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    first_llm = FakeLLM(
        responses=[
            decision_tool_response("react", "Needs ReAct."),
            {
                "content": "Need input.",
                "tool_calls": [
                    {
                        "id": "ask",
                        "function": {
                            "name": "send_message",
                            "arguments": (
                                '{"message":"Need input","expect_response":true}'
                            ),
                        },
                    }
                ],
            },
        ],
    )
    first_agent.llm = first_llm

    interrupted = await first_runner.run(
        task="Answer through auto react",
        execution_id=execution_id,
    )

    assert interrupted["status"] == "waiting_for_user"
    latest = await tracer.load_latest_checkpoint(execution_id)
    assert latest is not None
    assert latest["pattern"] == "AutoPattern"
    assert latest["pattern_state"]["selected_pattern"] == "react"

    resumed_llm = FakeLLM(["resumed react"])
    resumed_agent = Agent(
        name="auto",
        patterns=[AutoPattern()],
        llm=resumed_llm,
    )
    resumed_runner = AgentRunner(
        agent=resumed_agent,
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    await resumed_runner.post_user_message(
        execution_id,
        "User input",
        request_interrupt=False,
    )

    resumed = await resumed_runner.resume(execution_id)

    assert resumed["success"] is True
    assert resumed["status"] == "completed"
    assert resumed["output"] == "resumed react"
    assert resumed["pattern"] == "AutoPattern"
    assert len(resumed_llm.calls) == 1
    assert resumed_llm.calls[0]["tools"] is not None


@pytest.mark.asyncio
async def test_auto_pattern_dag_resume_from_tracer_does_not_redecide(
    tmp_path: Path,
) -> None:
    tracer = TracerCheckpointStore()
    execution_id = "auto-dag-restart"
    first_agent = Agent(
        name="auto",
        patterns=[AutoPattern(dag_pattern=DAGPattern(LLMPlanGenerator()))],
        llm=None,
    )
    first_runner = AgentRunner(
        agent=first_agent,
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    first_llm = FakeLLM(
        responses=[
            decision_tool_response("plan_execute", "Needs DAG."),
            plan_tool_response([{"id": "answer", "task": "Answer directly"}]),
            {
                "content": "Need input.",
                "tool_calls": [
                    {
                        "id": "ask",
                        "function": {
                            "name": "send_message",
                            "arguments": (
                                '{"message":"Need input","expect_response":true}'
                            ),
                        },
                    }
                ],
            },
        ],
    )
    first_agent.llm = first_llm

    interrupted = await first_runner.run(
        task="Plan through auto dag",
        execution_id=execution_id,
    )

    assert interrupted["status"] == "waiting_for_user"
    latest = await tracer.load_latest_checkpoint(execution_id)
    assert latest is not None
    assert latest["pattern"] == "AutoPattern"
    assert latest["pattern_state"]["selected_pattern"] == "plan_execute"
    assert latest["pattern_state"]["dag_state"] is not None

    resumed_llm = FakeLLM(["resumed dag"])
    resumed_agent = Agent(
        name="auto",
        patterns=[AutoPattern(dag_pattern=DAGPattern(LLMPlanGenerator()))],
        llm=resumed_llm,
    )
    resumed_runner = AgentRunner(
        agent=resumed_agent,
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    await resumed_runner.post_user_message(
        execution_id,
        "User input",
        request_interrupt=False,
    )

    resumed = await resumed_runner.resume(execution_id)

    assert resumed["success"] is True
    assert resumed["status"] == "completed"
    assert resumed["output"] == "resumed dag"
    assert resumed["pattern"] == "AutoPattern"
    assert len(resumed_llm.calls) == 2
    assert has_tool(resumed_llm.calls[1], DAG_COMPLETION_TOOL_NAME)


@pytest.mark.asyncio
async def test_auto_decision_prompt_includes_memory_rule_only_with_store() -> None:
    async def run_and_get_routing_prompt(memory_store: Any | None) -> str:
        llm = FakeLLM(
            responses=[
                decision_tool_response(
                    AutoAction.FINAL_ANSWER.value,
                    "simple",
                    "Done.",
                )
            ]
        )
        context = ExecutionContext()
        context.add_user_message("记住：我喜欢坂本龙一的音乐")
        result = await AutoPattern().run(
            context=context,
            tools=[],
            llm=llm,
            memory_store=memory_store,
        )
        assert result["success"] is True
        routing_messages = [
            message["content"]
            for message in llm.calls[0]["messages"]
            if message["role"] == "user"
            and "Auto routing instruction" in str(message["content"])
        ]
        assert routing_messages
        return str(routing_messages[-1])

    with_memory = await run_and_get_routing_prompt(FakeMemoryStore())
    without_memory = await run_and_get_routing_prompt(None)

    assert "memory tools can persist" in with_memory
    assert "memory tools can persist" not in without_memory


def test_clearing_request_scoped_enrichment_drops_the_image_edit_flag() -> None:
    from xagent.core.agent.context.enrichment import (
        IMAGE_EDIT_UNAVAILABLE_METADATA_KEY,
    )

    context = ExecutionContext(system_prompt="Base prompt.")
    context.metadata[IMAGE_EDIT_UNAVAILABLE_METADATA_KEY] = True

    AutoPattern()._clear_request_scoped_enrichment(context)

    assert IMAGE_EDIT_UNAVAILABLE_METADATA_KEY not in context.metadata


@pytest.mark.asyncio
async def test_stale_memory_language_does_not_reach_child_as_hard_policy() -> None:
    llm = FakeLLM([decision_tool_response("react", "Needs tools.")])
    child = CapturingChildPattern()
    pattern = AutoPattern(react_pattern=child)  # type: ignore[arg-type]
    context = ExecutionContext(execution_id="auto-memory-language-leak")
    context.metadata[MEMORY_CONTEXT_METADATA_KEY] = "请始终使用中文回答。"
    context.add_user_message("Summarize the quarterly revenue trend in one paragraph.")

    result = await pattern.run(
        context=context,
        tools=[],
        llm=llm,
        runtime=RecordingRuntime(),
    )

    assert result["success"] is True
    assert OUTPUT_LANGUAGE_METADATA_KEY not in context.metadata
    assert child.kwargs is not None
    child_system = child.kwargs["context"].get_messages_for_llm()[0]["content"]
    assert "请始终使用中文回答。" in child_system
    assert "Output language:" not in child_system
    assert "Output language policy:" not in child_system
    assert "Summarize the quarterly revenue trend in one paragraph." in child_system
    assert response_language_rules() in child_system


@pytest.mark.asyncio
async def test_direct_final_answer_allows_an_explicit_target_language() -> None:
    request = "Reply in French: what is the capital of Italy?"
    llm = FakeLLM(
        [
            decision_tool_response(
                "final_answer",
                "Simple factual reply.",
                answer="La capitale de l'Italie est Rome.",
            )
        ]
    )
    pattern = AutoPattern()
    context = ExecutionContext(execution_id="auto-explicit-target-language")
    context.add_user_message(request)

    result = await pattern.run(
        context=context, tools=[], llm=llm, runtime=PatternRuntime()
    )

    assert result["success"] is True
    assert result["output"] == "La capitale de l'Italie est Rome."
    assert OUTPUT_LANGUAGE_METADATA_KEY not in context.metadata
    target_rule = (
        "If the current user request explicitly asks to translate, rewrite, or "
        "answer in another language, use that requested target language."
    )
    tool_schema = llm.calls[0]["tools"][0]["function"]
    assert target_rule in tool_schema["description"]
    assert (
        target_rule in tool_schema["parameters"]["properties"]["answer"]["description"]
    )
    system_content = context.get_messages_for_llm()[0]["content"]
    assert request in system_content
    assert target_rule in system_content


class RoutedDecisionLLM:
    """Downstream selection behind a router, for the Auto decision path.

    Both entry points are needed: compaction goes through ``run_llm_call`` ->
    ``chat``, while the routing decision streams (``_ResolvedRouterLLM``
    defines ``stream_chat``, so the runtime takes the native streaming path).
    """

    def __init__(self, chat_responses: list[Any], decision: dict[str, Any]) -> None:
        self.chat_responses = chat_responses
        self.decision = decision
        self.calls: list[dict[str, Any]] = []

    async def chat(self, messages: Any = None, **kwargs: Any) -> Any:
        self.calls.append({"messages": messages, **kwargs})
        return self.chat_responses.pop(0)

    async def stream_chat(self, messages: Any = None, **kwargs: Any) -> Any:
        self.calls.append({"messages": messages, **kwargs})
        yield StreamChunk(
            type=ChunkType.TOOL_CALL,
            tool_calls=self.decision["tool_calls"],
        )


def _auto_routing_router(downstream: Any, route_prompts: list[str]) -> RouterLLM:
    """A real ``RouterLLM`` with its selection stubbed to record the prompt.

    ``context_window`` is set, as production always does via ``adapter.py``;
    4 gives a compaction threshold of 3, so any context compacts.
    """
    router = RouterLLM(downstream_resolver=lambda _model_id: downstream)
    router.context_window = 4

    async def select_model(prompt: str) -> str:
        route_prompts.append(prompt)
        return "test/model"

    router._select_model = select_model  # type: ignore[assignment]
    return router


@pytest.mark.asyncio
async def test_auto_summarizes_with_the_main_model_when_no_compact_model() -> None:
    """Same substitution as ReAct, and the resolve-before-compact order.

    Auto compacted before resolving the virtual model, the reverse of what
    ``prepare_llm_for_context`` documents: the resolver recomputes the
    compaction threshold from the selected model's window, which is useless
    once compaction has run, and compaction would otherwise route a second
    time on the compaction prompt -- whose only user message is the whole
    transcript.
    """
    downstream = RoutedDecisionLLM(
        [{"content": "summary of prior work"}],
        decision=decision_tool_response("final_answer", "Greeting only.", answer="hi"),
    )
    route_prompts: list[str] = []
    router = _auto_routing_router(downstream, route_prompts)
    context = ExecutionContext()
    context.add_user_message("hi")
    context.add_tool_result("read_file", {"output": "x" * 200}, tool_call_id="call-1")

    result = await AutoPattern().run(
        context=context,
        tools=[],
        llm=router,
        compact_llm=None,
        runtime=PatternRuntime(),
    )

    assert result["success"] is True
    # The summary call happened, and it went to the main model.
    assert len(downstream.calls) == 2
    # Exactly two routing decisions -- the one hoisted above compaction and
    # the per-attempt one in the decision loop. Never on the transcript.
    assert len(route_prompts) == 2
    assert not any(
        "Conversation history to compact" in prompt for prompt in route_prompts
    )
