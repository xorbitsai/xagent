"""The request prefix (system message, then tools) must not change mid-run.

Providers cache by prefix, and the system message comes first. A tool joining
the schema mid-run (a loaded deferred tool, or the stored-result reader after a
spill) may only append to the tools array; it must never alter the system text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.core.agent.test_react import (
    FakeLLM,
    FakeReadToolResultTool,
    FakeTool,
    _tool_call_response,
)
from tests.core.agent.test_react_deferred_builtin_schemas import (
    BuiltinTool,
    CATEGORIES,
)
from tests.core.agent.test_react_deferred_connector_schemas import ConnectorTool
from xagent.core.agent.context.execution import ExecutionContext
from xagent.core.agent.deferred_tools import LOAD_TOOLS_NAME
from xagent.core.agent.pattern.react import ReActPattern
from xagent.core.tools.adapters.vibe.base import ToolCategory
from xagent.core.tools.tool_result_spill import (
    SPILL_READ_TOOL_NAME,
    SPILL_RESERVED_RESULT_KEY,
    SpillTarget,
    spill_dir_for_workspace,
    spill_oversized_values,
)


def _system(call: dict[str, Any]) -> str:
    return "\n".join(
        str(m["content"]) for m in call["messages"] if m.get("role") == "system"
    )


def _tools(call: dict[str, Any]) -> list[str]:
    return [json.dumps(t, sort_keys=True) for t in call.get("tools") or []]


def _names(call: dict[str, Any]) -> list[str]:
    return [t["function"]["name"] for t in call.get("tools") or []]


def _spilling_tool(tmp_path: Path) -> FakeTool:
    target = SpillTarget(spill_dir=spill_dir_for_workspace(tmp_path), max_chars=100)

    class SpillingTool(FakeTool):
        async def run_json_async(self, args: dict[str, Any]) -> Any:
            spilled, records = spill_oversized_values(
                {"output": "x" * 300},
                target,
                tool_name="calculator",
                max_recursion=20,
            )
            return {**spilled, SPILL_RESERVED_RESULT_KEY: records}

    return SpillingTool()


def _ctx(tmp_path: Path) -> ExecutionContext:
    context = ExecutionContext()
    context.attach_workspace("ws", str(tmp_path))
    context.add_user_message("q")
    return context


def _assert_prefix_stable(before: dict[str, Any], after: dict[str, Any]) -> None:
    assert _system(before) == _system(after)
    old, new = _tools(before), _tools(after)
    assert new[: len(old)] == old


def _spill_call() -> dict[str, Any]:
    return _tool_call_response("s", "calculator", '{"expression": "1+1"}')


def _final() -> dict[str, Any]:
    return _tool_call_response("f", "final_answer", '{"answer": "done"}')


@pytest.mark.asyncio
async def test_spill_reader_joining_mid_run_leaves_the_system_message_alone(
    tmp_path: Path,
) -> None:
    llm = FakeLLM(responses=[_spill_call(), _final()])
    await ReActPattern(max_iterations=4).run(
        context=_ctx(tmp_path),
        tools=[_spilling_tool(tmp_path), FakeReadToolResultTool()],
        llm=llm,
    )

    assert SPILL_READ_TOOL_NAME not in _names(llm.calls[0])
    assert _names(llm.calls[1])[-1] == SPILL_READ_TOOL_NAME
    _assert_prefix_stable(llm.calls[0], llm.calls[1])
    # The reader is still announced, by a sentence that never changes.
    assert SPILL_READ_TOOL_NAME in _system(llm.calls[0])


@pytest.mark.asyncio
async def test_loads_then_a_spill_never_touch_the_system_message(
    tmp_path: Path,
) -> None:
    tools = [
        _spilling_tool(tmp_path),
        FakeReadToolResultTool(),
        ConnectorTool("notion_search", "notion"),
        BuiltinTool("generate_image", ToolCategory.IMAGE),
    ]
    llm = FakeLLM(
        responses=[
            _tool_call_response("l1", LOAD_TOOLS_NAME, '{"names": ["notion_search"]}'),
            _tool_call_response("l2", LOAD_TOOLS_NAME, '{"names": ["image"]}'),
            _spill_call(),
            _final(),
        ]
    )
    await ReActPattern(
        max_iterations=8,
        defer_connector_schemas=True,
        defer_builtin_tool_categories=CATEGORIES,
    ).run(context=_ctx(tmp_path), tools=tools, llm=llm)

    assert len(llm.calls) == 4
    assert _names(llm.calls[1])[-1] == "notion_search"
    assert _names(llm.calls[2])[-1] == "generate_image"
    assert SPILL_READ_TOOL_NAME in _names(llm.calls[3])
    for before, after in zip(llm.calls, llm.calls[1:]):
        assert _system(before) == _system(after)
    # Each load only appends. (The reader is placed ahead of loaded tools, so
    # the spill call is checked on the system message alone.)
    assert _tools(llm.calls[2])[: len(_tools(llm.calls[1]))] == _tools(llm.calls[1])
