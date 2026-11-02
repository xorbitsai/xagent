"""Per-call prompt size breakdown: sizes are right and content never leaks."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from tests.core.agent.concurrency_harness import make_react
from xagent.core.agent.context import ExecutionContext
from xagent.core.agent.prompt_composition import (
    CALL_KIND_COMPACTION,
    CALL_KIND_DECISION,
    CALL_KIND_MAIN,
    CALL_KIND_PROTOCOL_RETRY,
    PROMPT_COMPOSITION_KEY,
    build_prompt_composition,
    call_kind_from_metadata,
)
from xagent.core.agent.runtime import PatternRuntime

SECRET = "TOPSECRET-user-text"
TOOL_SECRET = "TOPSECRET-tool-arg"


def _schema(name: str, description: str = "") -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}},
        },
    }


class _Tracer:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def trace_event(self, event_type: Any, **kwargs: Any) -> None:
        self.events.append(
            {
                "type": getattr(event_type, "value", str(event_type)),
                "data": kwargs.get("data") or {},
            }
        )


def test_sizes_by_role_and_origin() -> None:
    messages = [
        {"role": "system", "content": "s" * 100},
        {"role": "user", "content": "u" * 30},
        {"role": "user", "content": [{"type": "text", "text": "v" * 20}]},
        {
            "role": "assistant",
            "content": "a" * 5,
            "tool_calls": [
                {"function": {"name": "abc", "arguments": "x" * 40}},
            ],
        },
    ]
    tools = [_schema("t1", "d" * 10), _schema("t2"), _schema("m1")]
    origins = {"t1": "file", "t2": "file", "m1": "mcp:slack"}

    comp = build_prompt_composition(
        messages,
        tools,
        call_kind=CALL_KIND_MAIN,
        iteration=3,
        tool_origins=origins,
        context_tokens=77,
    )

    assert comp["system_chars"] == 100
    assert comp["role_chars"] == {"system": 100, "user": 50, "assistant": 5 + 40 + 3}
    assert comp["role_count"] == {"system": 1, "user": 2, "assistant": 1}
    assert comp["message_count"] == 4
    assert comp["tools_count"] == 3
    assert comp["tools_count_by_origin"] == {"file": 2, "mcp:slack": 1}
    expected = {
        o: sum(
            len(json.dumps(s, ensure_ascii=False, separators=(",", ":")))
            for s in tools
            if origins[s["function"]["name"]] == o
        )
        for o in ("file", "mcp:slack")
    }
    assert comp["tools_chars_by_origin"] == expected
    assert comp["tools_chars"] == sum(expected.values())
    assert comp["largest_tools"][0]["name"] == "t1"
    assert comp["iteration"] == 3
    assert comp["context_tokens"] == 77


def test_unmapped_tool_is_unknown_and_largest_is_capped() -> None:
    tools = [_schema(f"t{i}", "d" * i) for i in range(10)]
    comp = build_prompt_composition([], tools, call_kind=CALL_KIND_MAIN)
    assert comp["tools_count_by_origin"] == {"unknown": 10}
    assert len(comp["largest_tools"]) == 5


def test_composition_never_contains_content() -> None:
    messages = [
        {"role": "user", "content": SECRET},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "f", "arguments": TOOL_SECRET}}],
        },
    ]
    tools = [_schema("f", description=SECRET)]
    comp = build_prompt_composition(
        messages, tools, call_kind=CALL_KIND_MAIN, tool_origins={"f": "basic"}
    )
    dumped = json.dumps(comp)
    assert SECRET not in dumped
    assert TOOL_SECRET not in dumped


@pytest.mark.parametrize(
    ("metadata", "kind"),
    [
        ({}, CALL_KIND_MAIN),
        ({"iteration": 1}, CALL_KIND_MAIN),
        ({"purpose": "context_compaction"}, CALL_KIND_COMPACTION),
        ({"phase": "repeated_tool_decision_requested"}, CALL_KIND_DECISION),
        ({"phase": "invalid_tool_protocol_retry"}, CALL_KIND_PROTOCOL_RETRY),
    ],
)
def test_call_kind_from_metadata(metadata: dict[str, Any], kind: str) -> None:
    assert call_kind_from_metadata(metadata) == kind


@pytest.mark.asyncio
async def test_runtime_attaches_composition_to_start_end_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tracer = _Tracer()
    runtime = PatternRuntime(tracer=tracer, execution_id="task-1")
    ctx = ExecutionContext()
    messages = [
        {"role": "system", "content": "s" * 10},
        {"role": "user", "content": SECRET},
    ]
    metadata: dict[str, Any] = {"iteration": 2}

    with caplog.at_level(logging.INFO, logger="xagent.core.agent.runtime"):
        await runtime.on_llm_start(
            context=ctx, messages=messages, tools=[_schema("f")], metadata=metadata
        )
        await runtime.on_llm_end(
            context=ctx,
            response={
                "content": "ok",
                "usage": {"prompt_tokens": 9, "completion_tokens": 1},
            },
            metadata=metadata,
        )

    start = tracer.events[0]["data"][PROMPT_COMPOSITION_KEY]
    end = tracer.events[-1]["data"]
    assert start["call_kind"] == CALL_KIND_MAIN
    assert start["iteration"] == 2
    assert start["system_chars"] == 10
    assert start["tools_count"] == 1
    # The matching llm_call_end carries the same breakdown beside the tokens.
    assert end[PROMPT_COMPOSITION_KEY] == start
    assert end["input_tokens"] == 9

    logged = [
        r.getMessage() for r in caplog.records if "prompt composition" in r.getMessage()
    ]
    assert len(logged) == 1
    assert SECRET not in logged[0]
    assert '"system_chars":10' in logged[0]


@pytest.mark.asyncio
async def test_runtime_keeps_caller_supplied_composition() -> None:
    tracer = _Tracer()
    runtime = PatternRuntime(tracer=tracer, execution_id="task-1")
    supplied = {"call_kind": CALL_KIND_PROTOCOL_RETRY, "marker": 1}
    await runtime.on_llm_start(
        context=ExecutionContext(),
        messages=[],
        metadata={PROMPT_COMPOSITION_KEY: supplied},
    )
    assert tracer.events[0]["data"][PROMPT_COMPOSITION_KEY] == supplied


@pytest.mark.asyncio
async def test_runtime_compaction_call_kind() -> None:
    tracer = _Tracer()
    runtime = PatternRuntime(tracer=tracer, execution_id="task-1")
    await runtime.on_llm_start(
        context=ExecutionContext(),
        messages=[{"role": "user", "content": "x"}],
        metadata={"purpose": "context_compaction"},
    )
    comp = tracer.events[0]["data"][PROMPT_COMPOSITION_KEY]
    assert comp["call_kind"] == CALL_KIND_COMPACTION


def test_react_origin_map_and_composition_buckets_builtin_controls() -> None:
    def tool(name: str, category: Any, server: str | None = None) -> Any:
        return SimpleNamespace(
            metadata=SimpleNamespace(name=name, category=category, source_server=server)
        )

    pattern = make_react()
    origins = pattern._tool_origin_map(
        [
            tool("read_file", SimpleNamespace(value="file")),
            tool("slack_search", SimpleNamespace(value="mcp"), "slack"),
            tool("no_category", None),
        ]
    )
    assert origins == {
        "read_file": "file",
        "slack_search": "mcp:slack",
        "no_category": "other",
    }

    pattern._tool_origins = origins
    ctx = ExecutionContext()
    schemas = [_schema("read_file"), _schema("slack_search"), _schema("final_answer")]
    comp = pattern._prompt_composition(
        ctx,
        [{"role": "system", "content": "abc"}],
        schemas,
        call_kind=CALL_KIND_MAIN,
        iteration=4,
    )
    assert comp["tools_count_by_origin"] == {"file": 1, "mcp:slack": 1, "builtin": 1}
    assert comp["iteration"] == 4
    assert isinstance(comp["context_tokens"], int)
