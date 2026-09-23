from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from xagent.core.agent import (
    AgentExecutionAdapter,
    AgentExecutionConfig,
    ExecutionContext,
    PatternRuntime,
    ReActPattern,
)
from xagent.core.agent.pattern.react import react as react_module

from .test_react import FakeLLM, FakeTool


def tool_call(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {"tool_calls": [{"id": call_id, "name": name, "args": args}]}


@pytest.mark.asyncio
async def test_iteration_limit_delivers_existing_result_without_more_work() -> None:
    llm = FakeLLM(
        [
            tool_call("calculator", {"expression": "2+2"}, "work"),
            tool_call(
                "final_answer",
                {
                    "answer": "2+2 = 4. The remaining calculations were not run.",
                    "outcome": "partial",
                    "response_language": "en",
                },
                "delivery",
            ),
        ]
    )
    tool = FakeTool()
    context = ExecutionContext(execution_id="iteration-limit-test")
    context.add_user_message("Calculate 2+2, then 3+3.")
    pattern = ReActPattern(max_iterations=1)

    result = await pattern.run(
        context=context, tools=[tool], llm=llm, runtime=PatternRuntime()
    )

    assert result["completion_outcome"] == "partial"
    assert result["termination_reason"] == "max_iterations"
    assert "2+2 = 4" in result["output"]
    assert "iteration limit" in result["output"]
    assert tool.calls == [{"expression": "2+2"}]
    assert len(llm.calls) == 2
    assert llm.calls[-1]["messages"][-1]["role"] == "user"
    assert "work phase has now stopped" in llm.calls[-1]["messages"][-1]["content"]
    assert llm.calls[-1]["tool_choice"] == {
        "type": "function",
        "function": {"name": "final_answer"},
    }
    assert [t["function"]["name"] for t in llm.calls[-1]["tools"]] == ["final_answer"]
    assert llm.calls[-1]["tools"][0]["function"]["parameters"]["properties"]["outcome"][
        "enum"
    ] == ["partial", "blocked"]
    assert any(
        m["role"] == "tool" and "4" in m["content"] for m in llm.calls[-1]["messages"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["partial", "blocked"])
async def test_iteration_limit_preserves_file_links_and_outcome(outcome: str) -> None:
    answer = "Saved [report.csv](file:registered-report). Analysis is unfinished."
    llm = FakeLLM(
        [
            tool_call(
                "final_answer",
                {"answer": answer, "outcome": outcome, "response_language": "en"},
                "end",
            )
        ]
    )
    context = ExecutionContext(execution_id="delivery")
    context.add_user_message("Create a report and analyze it.")
    context.add_assistant_message(
        "",
        tool_calls=[
            {
                "id": "write",
                "type": "function",
                "function": {"name": "write_file", "arguments": "{}"},
            }
        ],
    )
    context.add_tool_result(
        tool_call_id="write",
        result={
            "success": True,
            "markdown_link": "[report.csv](file:registered-report)",
        },
        tool_name="write_file",
    )
    pattern = ReActPattern(max_iterations=1)
    pattern.current_iteration = 1
    runtime = PatternRuntime()
    runtime.compact_context_if_needed = AsyncMock()

    result = await pattern.run(context=context, tools=[], llm=llm, runtime=runtime)

    assert answer in result["output"]
    assert result["completion_outcome"] == outcome
    assert "[report.csv](file:registered-report)" in str(llm.calls[0]["messages"])
    runtime.compact_context_if_needed.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        tool_call("calculator", {"expression": "9+9"}, "extra-work"),
        tool_call("final_answer", {"answer": " "}, "empty"),
        {
            "tool_calls": [
                {"name": "final_answer", "args": {"answer": "not accepted"}},
                {"name": "calculator", "args": {"expression": "9+9"}},
            ]
        },
        {"content": "Unstructured answer is not accepted on the delivery-only turn."},
    ],
)
async def test_iteration_limit_never_retries_or_executes_extra_tools(
    response: Any,
) -> None:
    llm = FakeLLM([tool_call("calculator", {"expression": "2+2"}, "work"), response])
    tool = FakeTool()
    context = ExecutionContext(execution_id="refused-delivery")
    context.add_user_message("Do several calculations.")
    pattern = ReActPattern(max_iterations=1)

    result = await pattern.run(context=context, tools=[tool], llm=llm)

    assert result["success"] is False
    assert result["status"] == "max_iterations"
    assert "output" not in result
    assert tool.calls == [{"expression": "2+2"}]
    assert len(llm.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("answer", None),
        ("answer", 42),
        ("outcome", None),
        ("outcome", "completed"),
        ("outcome", "unknown"),
        ("outcome", []),
        ("response_language", None),
        ("response_language", 42),
        ("extra", "not allowed"),
    ],
)
async def test_iteration_limit_rejects_schema_invalid_delivery(
    field: str, value: Any
) -> None:
    args = {"answer": "Useful result", "outcome": "partial", "response_language": "en"}
    if value is None:
        args.pop(field)
    else:
        args[field] = value
    llm = FakeLLM([tool_call("final_answer", args, "delivery")])
    pattern = ReActPattern(max_iterations=1)
    pattern.current_iteration = 1

    result = await pattern.run(context=ExecutionContext(), tools=[], llm=llm)

    assert result["success"] is False
    assert result["status"] == "max_iterations"
    assert "output" not in result
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_iteration_limit_timeout_cancels_delivery_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(react_module, "ITERATION_LIMIT_DELIVERY_TIMEOUT_SECONDS", 0.01)
    cancelled = asyncio.Event()

    class HangingLLM:
        async def chat(self, **kwargs: Any) -> Any:
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    context = ExecutionContext(execution_id="timeout")
    context.add_user_message("Continue.")
    pattern = ReActPattern(max_iterations=1)
    pattern.current_iteration = 1
    runtime = PatternRuntime()

    result = await pattern.run(
        context=context, tools=[], llm=HangingLLM(), runtime=runtime
    )

    assert result["status"] == "max_iterations"
    assert result["success"] is False
    assert cancelled.is_set()
    assert not runtime._active_llm_tasks


@pytest.mark.asyncio
async def test_iteration_limit_delivery_error_is_not_user_output() -> None:
    class FailingLLM:
        async def chat(self, **kwargs: Any) -> Any:
            raise RuntimeError("provider token=secret")

    context = ExecutionContext(execution_id="failure")
    context.add_user_message("Continue.")
    pattern = ReActPattern(max_iterations=1)
    pattern.current_iteration = 1
    result = await pattern.run(context=context, tools=[], llm=FailingLLM())

    assert result["status"] == "max_iterations"
    assert "secret" not in str(result)


@pytest.mark.asyncio
async def test_iteration_limit_cancellation_during_delivery_propagates() -> None:
    started = asyncio.Event()

    class HangingLLM:
        async def chat(self, **kwargs: Any) -> Any:
            started.set()
            await asyncio.Event().wait()

    pattern = ReActPattern(max_iterations=1)
    pattern.current_iteration = 1
    runtime = PatternRuntime()
    task = asyncio.create_task(
        pattern.run(
            context=ExecutionContext(execution_id="cancel"),
            tools=[],
            llm=HangingLLM(),
            runtime=runtime,
        )
    )
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not runtime._active_llm_tasks


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["prepare", "trace"])
@pytest.mark.parametrize("via_checker", [False, True])
async def test_iteration_limit_setup_interrupt_does_not_start_provider(
    phase: str,
    via_checker: bool,
) -> None:
    runtime = PatternRuntime()
    checker_requested = False

    async def checker() -> bool:
        return checker_requested

    if via_checker:
        runtime.interrupt_checker = checker
    started = asyncio.Event()
    release = asyncio.Event()
    provider = FakeLLM(
        [tool_call("final_answer", {"answer": "Must not deliver"}, "late")]
    )

    async def pause_setup(**_: Any) -> None:
        started.set()
        await release.wait()

    class PreparedLLM:
        async def prepare_for_call(self, *_: Any, **__: Any) -> Any:
            if phase == "prepare":
                await pause_setup()
            return provider

    if phase == "trace":
        runtime.on_llm_start = AsyncMock(side_effect=pause_setup)
    pattern = ReActPattern(max_iterations=1)
    pattern.current_iteration = 1
    context = ExecutionContext(execution_id="stop-during-delivery-setup")
    context.add_user_message("Continue.")
    task = asyncio.create_task(
        pattern.run(context=context, tools=[], llm=PreparedLLM(), runtime=runtime)
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    if via_checker:
        checker_requested = True
    else:
        runtime.request_interrupt("stop during delivery setup")
    release.set()
    result = await task

    assert result["status"] == "interrupted"
    assert "output" not in result
    assert not provider.calls
    assert not runtime._active_llm_tasks


@pytest.mark.asyncio
@pytest.mark.parametrize("during_call", [False, True])
async def test_iteration_limit_respects_user_stop(during_call: bool) -> None:
    runtime = PatternRuntime()
    llm = FakeLLM([])
    if during_call:

        async def stop(**kwargs: Any) -> Any:
            runtime.request_interrupt("User stopped")
            return tool_call("final_answer", {"answer": "Must not deliver"}, "stop")

        llm.chat = stop
    else:
        runtime.request_interrupt("User stopped")
    context = ExecutionContext(execution_id="stopped")
    context.add_user_message("Continue.")
    pattern = ReActPattern(max_iterations=1)
    pattern.current_iteration = 1

    result = await pattern.run(context=context, tools=[], llm=llm, runtime=runtime)

    assert result["status"] == "interrupted"
    assert "output" not in result


@pytest.mark.asyncio
async def test_iteration_limit_does_not_convert_failed_dag_step_to_success() -> None:
    llm = FakeLLM([])
    pattern = ReActPattern(max_iterations=1)
    pattern.current_iteration = 1
    context = ExecutionContext(execution_id="dag-step", metadata={"dag_step_id": "s1"})

    result = await pattern.run(context=context, tools=[], llm=llm)

    assert result["success"] is False
    assert result["status"] == "max_iterations"
    assert not llm.calls


@pytest.mark.asyncio
async def test_iteration_limit_attempt_survives_checkpoint_restore() -> None:
    pattern = ReActPattern(max_iterations=1)
    pattern.current_iteration = 1
    context = ExecutionContext(execution_id="restore")
    context.add_user_message("Continue.")
    await pattern.run(context=context, tools=[], llm=FakeLLM([{}]))
    restored = ReActPattern()
    restored.load_state(pattern.get_state())
    llm = FakeLLM([])

    result = await restored.run(context=context, tools=[], llm=llm)

    assert result["status"] == "max_iterations"
    assert not llm.calls


@pytest.mark.asyncio
async def test_normal_completion_does_not_add_delivery_call() -> None:
    llm = FakeLLM([tool_call("final_answer", {"answer": "Done."}, "end")])
    context = ExecutionContext(execution_id="normal")
    pattern = ReActPattern(max_iterations=1)

    result = await pattern.run(context=context, tools=[], llm=llm)

    assert result["completion_outcome"] == "completed"
    assert result["output"] == "Done."
    assert len(llm.calls) == 1
    assert not pattern.iteration_limit_delivery_attempted


@pytest.mark.asyncio
async def test_iteration_limit_delivery_reaches_execution_adapter(
    tmp_path: Path,
) -> None:
    llm = FakeLLM(
        [
            tool_call("calculator", {"expression": "2+2"}, "work"),
            tool_call(
                "final_answer",
                {
                    "answer": "4; second calculation not run.",
                    "outcome": "partial",
                    "response_language": "en",
                },
                "delivery",
            ),
        ]
    )
    adapter = AgentExecutionAdapter(
        AgentExecutionConfig(
            name="bounded-delivery",
            pattern="react",
            llm=llm,
            tools=[FakeTool()],
            react_max_iterations=1,
            workspace_base_dir=str(tmp_path),
            skills_enabled=False,
        )
    )

    result = await adapter.execute(
        task="Calculate 2+2, then 3+3.", task_id="bounded-delivery"
    )

    assert result["success"] is True
    assert result["completion_outcome"] == "partial"
    assert result["metadata"]["completion_outcome"] == "partial"
    assert result["agent_result"]["termination_reason"] == "max_iterations"
    assert result["termination_reason"] == "max_iterations"
    assert result["metadata"]["termination_reason"] == "max_iterations"
    assert "4; second calculation not run." in result["output"]
