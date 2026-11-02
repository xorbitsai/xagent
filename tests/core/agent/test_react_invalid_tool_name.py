"""A wrong tool name must come back to the model as a correctable tool error.

Providers only accept tool names matching ``[A-Za-z0-9_-]{1,64}``, so a name
outside that set has to leave every LLM request in a safe spelling. The
rewritten spelling is only ever shown to the provider: lookup and the error
keep what the model actually emitted, so a rewrite can never run a real tool.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from xagent.core.agent.context.execution import ExecutionContext
from xagent.core.agent.context.message import Message
from xagent.core.agent.pattern.react import ReActPattern
from xagent.core.model.chat.basic.bedrock import BedrockLLM

PROVIDER_TOOL_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")


class RecordingTool:
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = f"{name} tool"
        self.calls = 0

    async def execute(self, **_kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        return {"success": True, "output": "ok"}


class ScriptedLLM:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _tool_call_response(name: str) -> dict[str, Any]:
    return {
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
        ],
    }


async def _run_one_tool_call(
    name: str, tools: list[RecordingTool]
) -> tuple[dict[str, Any], ExecutionContext, list[dict[str, Any]]]:
    """Run a call to ``name`` then a plain answer; return the second request."""
    llm = ScriptedLLM([_tool_call_response(name), {"content": "Recovered."}])
    context = ExecutionContext()
    context.add_user_message("Go")
    result = await ReActPattern(max_iterations=3).run(
        context=context, tools=tools, llm=llm
    )
    assert len(llm.calls) == 2, "the model must get a second turn"
    return result, context, llm.calls[1]["messages"]


def _replayed_tool_names(messages: list[dict[str, Any]]) -> list[str]:
    return [
        tool_call["function"]["name"]
        for message in messages
        for tool_call in message.get("tool_calls") or []
    ]


def _tool_error(context: ExecutionContext) -> str:
    return context.get_messages_by_role("tool")[0].metadata["raw_result"]["error"]


@pytest.mark.asyncio
async def test_name_with_newline_reaches_the_model_as_an_error() -> None:
    tool = RecordingTool("create_report_suggestion")

    result, context, messages = await _run_one_tool_call(
        "create_report_su\nsuggestion", [tool]
    )

    assert result["success"] is True
    assert result["response"] == "Recovered."
    assert tool.calls == 0
    assert _replayed_tool_names(messages) == ["create_report_su_suggestion"]
    assert _tool_error(context) == (
        "Tool not found: 'create_report_su\\nsuggestion'. "
        "Did you mean: create_report_suggestion?"
    )
    tool_message = next(m for m in messages if m["role"] == "tool")
    assert "Did you mean: create_report_suggestion?" in tool_message["content"]
    assert tool_message["tool_call_id"] == "call_1"


@pytest.mark.asyncio
async def test_replayed_name_is_accepted_by_a_real_provider_conversion() -> None:
    _, _, messages = await _run_one_tool_call(
        "create_report_su\nsuggestion", [RecordingTool("create_report_suggestion")]
    )

    _, converted = BedrockLLM._convert_messages(messages)

    tool_uses = [
        block["toolUse"]
        for message in converted
        for block in message["content"]
        if "toolUse" in block
    ]
    assert [tool_use["name"] for tool_use in tool_uses] == [
        "create_report_su_suggestion"
    ]
    assert PROVIDER_TOOL_NAME.fullmatch(tool_uses[0]["name"])


@pytest.mark.asyncio
async def test_rewritten_name_equal_to_a_real_tool_is_not_executed() -> None:
    tool = RecordingTool("create_report")

    result, context, messages = await _run_one_tool_call("create report", [tool])

    assert result["success"] is True
    assert tool.calls == 0
    assert _replayed_tool_names(messages) == ["create_report"]
    assert _tool_error(context) == (
        "Tool not found: 'create report'. Did you mean: create_report?"
    )


@pytest.mark.asyncio
async def test_rewritten_name_equal_to_a_control_tool_does_not_end_the_run() -> None:
    llm = ScriptedLLM(
        [
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "final answer",
                            "arguments": json.dumps(
                                {"response_language": "English", "answer": "Early."}
                            ),
                        },
                    }
                ],
            },
            {"content": "Recovered."},
        ]
    )
    context = ExecutionContext()
    context.add_user_message("Go")

    result = await ReActPattern(max_iterations=3).run(
        context=context, tools=[], llm=llm
    )

    assert result["response"] == "Recovered."
    assert "Did you mean: final_answer?" in _tool_error(context)


@pytest.mark.asyncio
async def test_valid_unknown_name_is_replayed_unchanged_with_a_hint() -> None:
    tool = RecordingTool("calculator")

    result, context, messages = await _run_one_tool_call("weather_lookup", [tool])

    assert result["success"] is True
    assert tool.calls == 0
    assert _replayed_tool_names(messages) == ["weather_lookup"]
    error = _tool_error(context)
    assert error.startswith("Tool not found: 'weather_lookup'. Available tools: ")
    assert "calculator" in error


@pytest.mark.asyncio
async def test_valid_known_name_is_unchanged_and_executed() -> None:
    tool = RecordingTool("calculator")

    result, context, messages = await _run_one_tool_call("calculator", [tool])

    assert result["success"] is True
    assert tool.calls == 1
    assert _replayed_tool_names(messages) == ["calculator"]
    assert context.get_messages_by_role("tool")[0].metadata["raw_result"] == {
        "success": True,
        "output": "ok",
    }


def test_not_found_error_suggests_close_names() -> None:
    tool_name = "billing_create_invoice_suggestion"
    tools = [RecordingTool(tool_name), RecordingTool("calculator")]

    with pytest.raises(ValueError) as raised:
        ReActPattern()._find_tool(f"{tool_name}_placeholder", tools)

    assert str(raised.value) == (
        f"Tool not found: '{tool_name}_placeholder'. Did you mean: {tool_name}?"
    )


def test_not_found_error_lists_tools_when_nothing_is_close() -> None:
    with pytest.raises(ValueError) as raised:
        ReActPattern(user_interaction_enabled=False)._find_tool(
            "zzz", [RecordingTool("calculator")]
        )

    error = str(raised.value)
    assert error.startswith("Tool not found: 'zzz'. Available tools: calculator, ")
    assert "final_answer" in error


def test_not_found_error_caps_a_long_tool_list() -> None:
    tools = [RecordingTool(f"tool_{index:03d}") for index in range(100)]
    pattern = ReActPattern()

    with pytest.raises(ValueError) as raised:
        pattern._find_tool("zzz", tools)

    error = str(raised.value)
    assert "tool_019" in error
    assert "tool_020" not in error
    builtin = pattern._schema_tool_names(pattern._builtin_tool_schemas())
    hidden = 100 + len(builtin) - 20
    assert error.endswith(f"(and {hidden} more).")


def test_not_found_error_does_not_offer_the_hidden_spill_reader() -> None:
    from xagent.core.tools.tool_result_spill import SPILL_READ_TOOL_NAME

    with pytest.raises(ValueError) as raised:
        ReActPattern()._find_tool(
            "zzz", [RecordingTool(SPILL_READ_TOOL_NAME), RecordingTool("calculator")]
        )

    assert SPILL_READ_TOOL_NAME not in str(raised.value)


def _context_with_tool_call(name: Any) -> ExecutionContext:
    context = ExecutionContext()
    context.add_user_message("Go")
    context.add_assistant_message(
        "",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
        ],
    )
    context.add_tool_result(tool_name=str(name), result="x", tool_call_id="call_1")
    return context


@pytest.mark.parametrize(
    ("stored", "sent"),
    [
        ("calculator", "calculator"),
        ("with-dash_and_9", "with-dash_and_9"),
        ("a\nb", "a_b"),
        ("a b.c", "a_b_c"),
        ("工具", "__"),
        ("x" * 70, "x" * 64),
        ("", "invalid_tool_name"),
        (None, "invalid_tool_name"),
    ],
)
def test_requests_only_carry_provider_safe_names(stored: Any, sent: str) -> None:
    messages = _context_with_tool_call(stored).get_messages_for_llm()

    assert _replayed_tool_names(messages) == [sent]
    assert PROVIDER_TOOL_NAME.fullmatch(sent)


def test_stored_history_keeps_what_the_model_emitted() -> None:
    context = _context_with_tool_call("create report")

    context.get_messages_for_llm()

    [assistant] = context.get_messages_by_role("assistant")
    assert assistant.tool_calls is not None
    assert assistant.tool_calls[0]["function"]["name"] == "create report"


def test_restored_checkpoint_with_invalid_name_is_also_safe() -> None:
    restored = ExecutionContext.from_dict(
        _context_with_tool_call("a\nb").to_dict(),
    )

    assert _replayed_tool_names(restored.get_messages_for_llm()) == ["a_b"]


def test_flat_tool_call_shape_is_also_safe() -> None:
    context = ExecutionContext()
    context.messages.append(
        Message(
            role="assistant",
            content="",
            tool_calls=[{"id": "call_1", "name": "a\nb", "args": {}}],
        )
    )
    context.messages.append(Message(role="tool", content="x", tool_call_id="call_1"))

    [message] = [m for m in context.get_messages_for_llm() if m.get("tool_calls")]

    assert message["tool_calls"][0]["name"] == "a_b"
