"""Run-level facts the runtime records for the turn settlement notification."""

from __future__ import annotations

from datetime import datetime

import pytest

from xagent.core.agent import Agent, ExecutionContext, PatternRuntime
from xagent.core.agent.runner import AgentRunner
from xagent.core.model.chat.token_context import (
    TokenUsage,
    get_token_usage,
    set_token_usage,
)


@pytest.mark.asyncio
async def test_on_tool_error_counts_one_tool_error() -> None:
    usage = set_token_usage(TokenUsage())
    runtime = PatternRuntime()
    tool_call = {"id": "call-1", "name": "search", "args": {}}

    await runtime.on_tool_start(tool_call=tool_call)
    assert get_token_usage().tool_errors == 0

    await runtime.on_tool_error(tool_call=tool_call, error=RuntimeError("boom"))

    assert usage.tool_calls == 1
    assert usage.tool_errors == 1


@pytest.mark.asyncio
async def test_first_final_answer_delta_time_is_the_first_non_empty_chunk() -> None:
    runtime = PatternRuntime()
    message_id = "final_answer_1"

    await runtime.emit_final_answer_delta(message_id, "")
    assert runtime.first_final_answer_delta_at is None

    await runtime.emit_final_answer_delta(message_id, "Hi ")
    first = runtime.first_final_answer_delta_at
    assert isinstance(first, datetime)
    assert first.tzinfo is not None

    await runtime.emit_final_answer_delta(message_id, "there")
    assert runtime.first_final_answer_delta_at == first


@pytest.mark.asyncio
async def test_finish_run_writes_first_delta_time_into_result_metadata() -> None:
    runner = AgentRunner(agent=Agent(name="facts", patterns=[], llm=None))
    context = ExecutionContext(execution_id="facts-1", system_prompt="s")
    runtime = PatternRuntime()
    message_id = "final_answer_1"
    await runtime.emit_final_answer_delta(message_id, "Hi")
    result: dict = {"success": False, "output": "Hi"}

    await runner._finish_run(context, result, runtime=runtime)

    assert result["metadata"]["first_answer_delta_at"] == (
        runtime.first_final_answer_delta_at.isoformat()
    )


@pytest.mark.asyncio
async def test_finish_run_keeps_existing_metadata_and_skips_unstreamed_runs() -> None:
    runner = AgentRunner(agent=Agent(name="facts", patterns=[], llm=None))
    context = ExecutionContext(execution_id="facts-2", system_prompt="s")
    streamed = PatternRuntime()
    message_id = "final_answer_2"
    await streamed.emit_final_answer_delta(message_id, "Hi")
    result: dict = {"success": False, "metadata": {"finish_reason": "stop"}}

    await runner._finish_run(context, result, runtime=streamed)
    assert result["metadata"]["finish_reason"] == "stop"
    assert "first_answer_delta_at" in result["metadata"]

    unstreamed: dict = {"success": False}
    await runner._finish_run(context, unstreamed, runtime=PatternRuntime())
    assert "first_answer_delta_at" not in unstreamed.get("metadata", {})
