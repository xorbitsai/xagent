"""A wrong tool name must come back to the model as a correctable tool error.

Providers only accept tool names matching ``[A-Za-z0-9_-]{1,64}``, so a name
outside that set has to leave every LLM request in a safe spelling. The
rewritten spelling is never used for lookup, so a rewrite can never run a
real tool. The error reports what the model emitted, and names the rewritten
spelling only as a suggestion when it is a tool the model is offered.
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

    error = ReActPattern()._tool_not_found_message(
        f"{tool_name}_placeholder", tools, ExecutionContext()
    )

    assert error == (
        f"Tool not found: '{tool_name}_placeholder'. Did you mean: {tool_name}?"
    )


def test_not_found_error_lists_tools_when_nothing_is_close() -> None:
    error = ReActPattern(user_interaction_enabled=False)._tool_not_found_message(
        "zzz", [RecordingTool("calculator")], ExecutionContext()
    )

    assert error.startswith("Tool not found: 'zzz'. Available tools: calculator, ")
    assert "final_answer" in error


def test_not_found_error_caps_a_long_tool_list() -> None:
    tools = [RecordingTool(f"tool_{index:03d}") for index in range(100)]
    pattern = ReActPattern()

    error = pattern._tool_not_found_message("zzz", tools, ExecutionContext())

    assert "tool_019" in error
    assert "tool_020" not in error
    builtin = pattern._schema_tool_names(pattern._builtin_tool_schemas())
    hidden = 100 + len(builtin) - 20
    assert error.endswith(f"(and {hidden} more).")


def test_replayed_spelling_of_a_real_tool_is_named_when_nothing_is_close() -> None:
    tools = [RecordingTool(f"tool_{index:03d}") for index in range(25)]
    tools.append(RecordingTool("a_b"))

    error = ReActPattern()._tool_not_found_message("a b", tools, ExecutionContext())

    assert error == "Tool not found: 'a b'. Did you mean: a_b?"


def test_replayed_spelling_of_a_real_tool_is_named_first() -> None:
    tools = [RecordingTool("create_report"), RecordingTool("create_report_")]

    error = ReActPattern()._tool_not_found_message(
        "create_report!", tools, ExecutionContext()
    )

    assert error == (
        "Tool not found: 'create_report!'. Did you mean: create_report_, create_report?"
    )


def test_suggestions_are_capped_at_three() -> None:
    tools = [
        RecordingTool(name)
        for name in (
            "create_report_",
            "create_report",
            "create_reports",
            "create_reporta",
        )
    ]

    error = ReActPattern()._tool_not_found_message(
        "create_report!", tools, ExecutionContext()
    )

    assert error == (
        "Tool not found: 'create_report!'. "
        "Did you mean: create_report_, create_report, create_reports?"
    )


def test_rewritten_name_is_suggested_only_when_offered() -> None:
    tools = [RecordingTool("read_tool_result"), RecordingTool("calculator")]

    error = ReActPattern()._tool_not_found_message(
        "read tool result", tools, ExecutionContext()
    )

    assert error == (
        "Tool not found: 'read tool result'. "
        "Available tools: calculator, final_answer, send_message, ask_user_question."
    )


def _context_with_stored_result() -> ExecutionContext:
    from xagent.core.agent.context.components import SpillRegistryComponent

    context = ExecutionContext()
    context.set_component(
        "spilled_results",
        SpillRegistryComponent(
            records=[
                {
                    "relative_path": "output/result.json",
                    "kind": "array",
                    "item_count": 1,
                    "original_chars": 10,
                    "value_path": "output",
                    "record_fields": None,
                    "truncated_after_items": None,
                }
            ]
        ),
    )
    context.add_user_message("Go")
    return context


@pytest.mark.asyncio
@pytest.mark.parametrize("stored", [False, True])
async def test_hint_candidates_are_the_tools_offered_on_that_turn(
    stored: bool,
) -> None:
    from xagent.core.tools.tool_result_spill import SPILL_READ_TOOL_NAME

    if stored:
        context = _context_with_stored_result()
    else:
        context = ExecutionContext()
        context.add_user_message("Go")
    llm = ScriptedLLM([_tool_call_response("zzz"), {"content": "Recovered."}])
    tools = [RecordingTool(SPILL_READ_TOOL_NAME), RecordingTool("calculator")]

    await ReActPattern(max_iterations=3).run(context=context, tools=tools, llm=llm)

    offered = [schema["function"]["name"] for schema in llm.calls[0]["tools"]]
    assert (SPILL_READ_TOOL_NAME in offered) is stored
    assert _tool_error(context) == (
        f"Tool not found: 'zzz'. Available tools: {', '.join(offered)}."
    )


@pytest.mark.asyncio
async def test_mistyped_reader_is_suggested_once_a_result_is_stored() -> None:
    from xagent.core.tools.tool_result_spill import SPILL_READ_TOOL_NAME

    context = _context_with_stored_result()
    llm = ScriptedLLM(
        [_tool_call_response("read_tool_reslt"), {"content": "Recovered."}]
    )
    tools = [RecordingTool(SPILL_READ_TOOL_NAME), RecordingTool("calculator")]

    await ReActPattern(max_iterations=3).run(context=context, tools=tools, llm=llm)

    assert _tool_error(context) == (
        "Tool not found: 'read_tool_reslt'. Did you mean: read_tool_result?"
    )


@pytest.mark.asyncio
async def test_one_invalid_call_among_valid_ones_keeps_ids_and_order() -> None:
    tool = RecordingTool("calculator")
    llm = ScriptedLLM(
        [
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "calculator", "arguments": '{"x": 1}'},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "weather lookup", "arguments": "{}"},
                    },
                    {
                        "id": "call_3",
                        "type": "function",
                        "function": {"name": "calculator", "arguments": '{"x": 2}'},
                    },
                ],
            },
            {"content": "Recovered."},
        ]
    )
    context = ExecutionContext()
    context.add_user_message("Go")

    result = await ReActPattern(max_iterations=3).run(
        context=context, tools=[tool], llm=llm
    )

    assert result["response"] == "Recovered."
    assert tool.calls == 2
    messages = llm.calls[1]["messages"]
    replayed = [
        (tool_call["id"], tool_call["function"]["name"])
        for message in messages
        for tool_call in message.get("tool_calls") or []
    ]
    assert replayed == [
        ("call_1", "calculator"),
        ("call_2", "weather_lookup"),
        ("call_3", "calculator"),
    ]
    assert [m["tool_call_id"] for m in messages if m["role"] == "tool"] == [
        "call_1",
        "call_2",
        "call_3",
    ]
    results = {
        message.tool_call_id: message.metadata["raw_result"]
        for message in context.get_messages_by_role("tool")
    }
    assert results["call_1"] == {"success": True, "output": "ok"}
    assert results["call_3"] == {"success": True, "output": "ok"}
    assert results["call_2"]["error"].startswith("Tool not found: 'weather lookup'.")
    [assistant] = [m for m in context.get_messages_by_role("assistant") if m.tool_calls]
    assert [c["function"]["name"] for c in assistant.tool_calls] == [
        "calculator",
        "weather lookup",
        "calculator",
    ]


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
        ("y" * 64, "y" * 64),
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
