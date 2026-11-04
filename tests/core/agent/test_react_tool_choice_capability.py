from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from xagent.core.agent import ExecutionContext, PatternRuntime, ReActPattern
from xagent.core.agent.context.execution import TOOL_EVIDENCE_REMOVED_METADATA_KEY
from xagent.core.agent.service import AgentService
from xagent.core.model.chat.basic.bedrock import BedrockLLM
from xagent.core.model.chat.basic.router import RouterLLM


def _text_events(text: str) -> list[dict[str, Any]]:
    return [
        {"messageStart": {"role": "assistant"}},
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"text": text},
            }
        },
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 4, "outputTokens": 2}}},
    ]


def _tool_events(
    name: str, arguments: str, *, call_id: str = "call-1"
) -> list[dict[str, Any]]:
    return [
        {"messageStart": {"role": "assistant"}},
        {
            "contentBlockStart": {
                "contentBlockIndex": 0,
                "start": {"toolUse": {"toolUseId": call_id, "name": name}},
            }
        },
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": arguments}},
            }
        },
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "tool_use"}},
        {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 3}}},
    ]


class _EventStream:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events

    def __iter__(self):
        return iter(self._events)

    def close(self) -> None:
        pass


class _ScriptedBedrockClient:
    def __init__(self, responses: list[list[dict[str, Any]]]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def converse_stream(self, **request: Any) -> dict[str, Any]:
        self.requests.append(request)
        return {"stream": _EventStream(self.responses.pop(0))}


class _CalculatorArgs(BaseModel):
    expression: str


class _Calculator:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

        class Metadata:
            name = "calculator"
            description = "Evaluate a basic arithmetic expression."

        self.metadata = Metadata()

    def args_type(self) -> type[BaseModel]:
        return _CalculatorArgs

    async def run_json_async(self, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(args)
        return {"result": 4}


@pytest.mark.asyncio
async def test_agent_service_auto_only_bedrock_completes_as_plain_text() -> None:
    client = _ScriptedBedrockClient([_text_events("A tomato-shaped hello.")])
    llm = BedrockLLM(
        "au.anthropic.claude-opus-5-5-v1:0",
        region_name="ap-southeast-2",
        client=client,
    )
    service = AgentService(
        name="Tomato guide",
        id="auto-only-plain-text",
        pattern="react",
        llm=llm,
        tools=[],
        enable_default_tools=False,
        enable_workspace=False,
        memory_enabled=False,
        skills_enabled=False,
        user_interaction_enabled=False,
        react_max_iterations=2,
    )

    result = await service.execute_task("Say hello with a tomato analogy.")

    assert result["success"] is True
    assert result["output"] == "A tomato-shaped hello."
    assert len(client.requests) == 1
    request = client.requests[0]
    assert request["toolConfig"]["toolChoice"] == {"auto": {}}
    assert [tool["toolSpec"]["name"] for tool in request["toolConfig"]["tools"]] == [
        "final_answer"
    ]
    system_prompt = str(request["system"])
    assert "prefer calling the final_answer tool exactly once" in system_prompt
    assert "plain assistant text only when the task is fully complete" in system_prompt
    assert "Never use plain assistant text for a progress update" in system_prompt
    assert "partial result, a blocked outcome" in system_prompt


@pytest.mark.asyncio
async def test_agent_service_auto_only_bedrock_runs_tool_then_text_completion() -> None:
    client = _ScriptedBedrockClient(
        [
            _tool_events("calculator", '{"expression":"2+2"}'),
            _text_events("The result is 4."),
        ]
    )
    calculator = _Calculator()
    service = AgentService(
        name="Calculator",
        id="auto-only-tool-turn",
        pattern="react",
        llm=BedrockLLM(
            "au.anthropic.claude-opus-5-5-v1:0",
            region_name="ap-southeast-2",
            client=client,
        ),
        tools=[calculator],
        enable_default_tools=False,
        enable_workspace=False,
        memory_enabled=False,
        skills_enabled=False,
        user_interaction_enabled=False,
        react_max_iterations=3,
    )

    result = await service.execute_task("Calculate 2+2.")

    assert result["success"] is True
    assert result["output"] == "The result is 4."
    assert calculator.calls == [{"expression": "2+2"}]
    assert [request["toolConfig"]["toolChoice"] for request in client.requests] == [
        {"auto": {}},
        {"auto": {}},
    ]


@pytest.mark.asyncio
async def test_react_uses_resolved_router_auto_only_capability(monkeypatch) -> None:
    client = _ScriptedBedrockClient([_text_events("Resolved naturally.")])
    downstream = BedrockLLM(
        "anthropic.claude-opus-5-5-v1:0",
        region_name="us-east-1",
        client=client,
    )
    router = RouterLLM(
        downstream_resolver=lambda _model_id: downstream,
        candidate_models=("bedrock-opus",),
    )

    async def select_model(_prompt: str, **_kwargs: Any) -> str:
        return "bedrock-opus"

    monkeypatch.setattr(router, "_select_model", select_model)
    monkeypatch.setattr(router, "_profile_context_window", lambda _model_id: None)
    monkeypatch.setattr(router, "_profile_input_modalities", lambda _model_id: ())
    context = ExecutionContext(execution_id="resolved-router-auto-only")
    context.add_user_message("Reply naturally.")

    result = await ReActPattern(max_iterations=1).run(
        context=context, tools=[], llm=router
    )

    assert result["success"] is True
    assert result["output"] == "Resolved naturally."
    assert client.requests[0]["toolConfig"]["toolChoice"] == {"auto": {}}
    assert "plain assistant text only when the task is fully complete" in str(
        client.requests[0]["system"]
    )


class _ScriptedLLM:
    def __init__(
        self, responses: list[Any], *, supports_forced_tool_choice: bool
    ) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.supports_forced_tool_choice = supports_forced_tool_choice

    async def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _tool_call(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {"tool_calls": [{"id": call_id, "name": name, "args": args}]}


@pytest.mark.asyncio
async def test_auto_only_react_retry_and_forced_finalization_stay_in_auto_mode() -> (
    None
):
    llm = _ScriptedLLM(
        [
            _tool_call("calculator", {"expression": "2+2"}, "work"),
            _tool_call("final_answer", {"answer": " "}, "empty"),
            _tool_call(
                "final_answer",
                {
                    "response_language": "English",
                    "answer": "Recovered through the control tool.",
                    "outcome": "completed",
                },
                "recovered",
            ),
        ],
        supports_forced_tool_choice=False,
    )
    context = ExecutionContext(execution_id="auto-only-retry")
    context.add_user_message("Calculate 2+2.")

    result = await ReActPattern(max_iterations=3, finalize_after_tool_result=True).run(
        context=context, tools=[_Calculator()], llm=llm
    )

    assert result["success"] is True
    assert result["output"] == "Recovered through the control tool."
    assert [call["tool_choice"] for call in llm.calls] == ["auto", "auto", "auto"]
    assert "plain assistant text" not in llm.calls[1]["messages"][0]["content"]
    assert "plain assistant text" not in llm.calls[2]["messages"][0]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("supports_forced_tool_choice", [True, False])
async def test_flash_forced_turn_accepts_nonempty_plain_text(
    supports_forced_tool_choice: bool,
) -> None:
    llm = _ScriptedLLM(
        [
            _tool_call("calculator", {"expression": "2+2"}, "work"),
            {"content": "2+2 = 4.", "done": True},
        ],
        supports_forced_tool_choice=supports_forced_tool_choice,
    )
    context = ExecutionContext(execution_id="flash-plain-text")
    context.metadata[TOOL_EVIDENCE_REMOVED_METADATA_KEY] = False
    context.add_user_message("Calculate 2+2.")

    result = await ReActPattern(max_iterations=2, finalize_after_tool_result=True).run(
        context=context, tools=[_Calculator()], llm=llm
    )

    assert result["success"] is True
    assert result["output"] == "2+2 = 4."
    assert len(llm.calls) == 2
    assert llm.calls[-1]["tool_choice"] == (
        "required" if supports_forced_tool_choice else "auto"
    )


@pytest.mark.asyncio
async def test_missing_evidence_requires_structured_partial_outcome() -> None:
    llm = _ScriptedLLM(
        [
            {"content": "Some evidence is unavailable.", "done": True},
            _tool_call(
                "final_answer",
                {
                    "response_language": "English",
                    "answer": "Some evidence is unavailable.",
                    "outcome": "partial",
                },
                "partial",
            ),
        ],
        supports_forced_tool_choice=False,
    )
    pattern = ReActPattern(max_iterations=1)
    pattern.force_final_answer_next = True
    context = ExecutionContext(execution_id="missing-evidence")
    context.metadata[TOOL_EVIDENCE_REMOVED_METADATA_KEY] = True
    context.add_user_message("Summarize the completed work.")

    result = await pattern.run(context=context, tools=[], llm=llm)

    assert result["success"] is True
    assert result["completion_outcome"] == "partial"
    assert "Call final_answer" in llm.calls[-1]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_auto_only_settlement_fence_retries_plain_text_and_preserves_outcome() -> (
    None
):
    llm = _ScriptedLLM(
        [
            {"content": "The write was refused.", "done": True},
            _tool_call(
                "final_answer",
                {
                    "response_language": "English",
                    "answer": "The write was refused.",
                    "outcome": "blocked",
                },
                "blocked",
            ),
        ],
        supports_forced_tool_choice=False,
    )
    pattern = ReActPattern(max_iterations=1)
    pattern.force_final_answer_next = True
    pattern.settlement_final_answer_fence = True
    context = ExecutionContext(execution_id="auto-only-settlement-fence")
    context.add_user_message("Perform the protected write.")

    result = await pattern.run(context=context, tools=[], llm=llm)

    assert result["success"] is True
    assert result["completion_outcome"] == "blocked"
    assert result["output"] == "The write was refused."
    assert [call["tool_choice"] for call in llm.calls] == ["auto", "auto"]
    assert all(
        "plain assistant text" not in call["messages"][0]["content"]
        for call in llm.calls
    )


@pytest.mark.asyncio
async def test_auto_only_blank_plain_text_uses_protocol_retry() -> None:
    llm = _ScriptedLLM(
        [
            {"content": "  ", "done": True},
            {"content": "Recovered with text.", "done": True},
        ],
        supports_forced_tool_choice=False,
    )
    context = ExecutionContext(execution_id="auto-only-blank-text")
    context.add_user_message("Answer without using the calculator.")

    result = await ReActPattern(max_iterations=1).run(
        context=context, tools=[_Calculator()], llm=llm
    )

    assert result["success"] is True
    assert result["output"] == "Recovered with text."
    assert [call["tool_choice"] for call in llm.calls] == ["auto", "auto"]


@pytest.mark.asyncio
async def test_forced_capable_blank_plain_text_retry_requires_tool_call() -> None:
    llm = _ScriptedLLM(
        [
            {"content": "  ", "done": True},
            _tool_call(
                "final_answer",
                {
                    "response_language": "English",
                    "answer": "Recovered through final_answer.",
                    "outcome": "completed",
                },
                "recovered",
            ),
        ],
        supports_forced_tool_choice=True,
    )
    context = ExecutionContext(execution_id="forced-capable-blank-text")
    context.add_user_message("Answer without using the calculator.")
    runtime = PatternRuntime(execution_id=context.execution_id)

    result = await ReActPattern(max_iterations=1).run(
        context=context, tools=[_Calculator()], llm=llm, runtime=runtime
    )

    assert result["success"] is True
    assert result["output"] == "Recovered through final_answer."
    assert [call["tool_choice"] for call in llm.calls] == ["required", "required"]
    assert "plain-text answer" not in llm.calls[-1]["messages"][0]["content"]
    assert any(
        checkpoint["label"] == "empty_plain_text_recovery"
        for checkpoint in runtime.checkpoints
    )


@pytest.mark.asyncio
async def test_auto_only_repeated_tool_decision_uses_auto_choice() -> None:
    llm = _ScriptedLLM(
        [
            _tool_call(
                "react_decision",
                {
                    "action": "final_answer",
                    "reason": "Enough work has been completed.",
                    "missing_verification": "",
                },
                "decision",
            )
        ],
        supports_forced_tool_choice=False,
    )
    pattern = ReActPattern()
    pattern.repeated_tool_decision = {
        "tool_name": "calculator",
        "consecutive_tool_calls": 4,
    }
    context = ExecutionContext(execution_id="auto-only-repeated-decision")
    context.add_user_message("Calculate several values.")

    await pattern._run_repeated_tool_decision(
        context=context,
        llm=llm,
        runtime=PatternRuntime(execution_id=context.execution_id),
    )

    assert llm.calls[0]["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_other_provider_keeps_required_tool_choice() -> None:
    llm = _ScriptedLLM(
        [
            _tool_call(
                "final_answer",
                {
                    "response_language": "English",
                    "answer": "Done.",
                    "outcome": "completed",
                },
                "final",
            )
        ],
        supports_forced_tool_choice=True,
    )
    context = ExecutionContext(execution_id="forced-supported")
    context.add_user_message("Reply when done.")

    result = await ReActPattern(max_iterations=1).run(
        context=context, tools=[], llm=llm
    )

    assert result["success"] is True
    assert llm.calls[0]["tool_choice"] == "required"
    assert (
        "instead of calling another work tool or returning plain assistant text"
        in (llm.calls[0]["messages"][0]["content"])
    )


@pytest.mark.asyncio
async def test_forced_supported_bedrock_still_sends_any() -> None:
    client = _ScriptedBedrockClient(
        [
            _tool_events(
                "final_answer",
                '{"response_language":"English","answer":"Done.",'
                '"outcome":"completed"}',
                call_id="final",
            )
        ]
    )
    context = ExecutionContext(execution_id="bedrock-forced-supported")
    context.add_user_message("Reply when done.")

    result = await ReActPattern(max_iterations=1).run(
        context=context,
        tools=[],
        llm=BedrockLLM("us.anthropic.claude-haiku-5-5-v1:0", client=client),
    )

    assert result["success"] is True
    assert client.requests[0]["toolConfig"]["toolChoice"] == {"any": {}}
