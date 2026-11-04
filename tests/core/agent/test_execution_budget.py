import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from xagent.core.agent.budget import (
    ExecutionBudget,
    ExecutionBudgetPolicy,
    active_execution_budget,
    budget_llm_kwargs,
    budget_warning_handler,
)
from xagent.core.agent.context import ExecutionContext
from xagent.core.agent.runner import AgentRunner
from xagent.core.agent.runtime import (
    LLMCallInterrupted,
    PatternRuntime,
    ToolCallInterrupted,
)
from xagent.core.model.chat.token_context import (
    TokenContextManager,
    add_token_usage,
    token_usage_observer,
)
from xagent.core.model.chat.types import ChunkType, StreamChunk


class MeteredLLM:
    def __init__(self, tokens=30):
        self.tokens = tokens
        self.calls = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        add_token_usage(
            input_tokens=self.tokens - 1, output_tokens=1, cached_input_tokens=10
        )
        return {"content": "result"}


class LoopPattern:
    status = "running"

    async def run(self, context, runtime, llm=None, **kwargs):
        self.runtime = runtime
        self.context = context
        while not await runtime.should_interrupt():
            await runtime.run_llm_call(
                llm, messages=[{"role": "user", "content": "work"}]
            )
            await runtime.checkpoint("after_call", context=context, pattern=self)
        return {"status": "interrupted", "success": False}


def runner(pattern, llm, policy):
    return AgentRunner(
        SimpleNamespace(patterns=[pattern], tools=[], llm=llm),
        workspace_enabled=False,
        budget_policy_provider=lambda: policy,
    )


@pytest.mark.asyncio
async def test_real_runner_stops_and_warns_once_without_extra_summary_call():
    llm = MeteredLLM()
    pattern = LoopPattern()
    result = await runner(
        pattern, llm, ExecutionBudgetPolicy(max_tokens=100, soft_limit_percent=50)
    ).run("work", outbound_message_handler=lambda payload: None)
    assert len(llm.calls) == 4  # last admitted call overshoots; no fifth call
    assert "Execution token budget" not in str(llm.calls[0])
    assert "Execution token budget" in str(llm.calls[2])
    assert len(pattern.runtime.outbound_messages) == 1
    assert result["termination_reason"] == "token_budget"
    assert result["completion_outcome"] == "blocked"
    assert result["context"].execution_budget["used_tokens"] == 120
    assert pattern.runtime.last_checkpoint["context"]["execution_budget"]["closed"]
    assert active_execution_budget.get() is None
    assert token_usage_observer.get() is None
    assert budget_warning_handler.get() is None


@pytest.mark.asyncio
async def test_unlimited_run_does_not_add_budget_only_tail_checkpoint():
    class RecordingPattern:
        async def run(self, context, runtime, llm, **kwargs):
            self.runtime = runtime
            await runtime.run_llm_call(llm, messages=[])
            context.add_assistant_message("recorded")
            await runtime.checkpoint("final", context=context, pattern=self)
            return {"success": True, "output": "recorded"}

    pattern = RecordingPattern()
    llm = MeteredLLM()
    result = await runner(pattern, llm, ExecutionBudgetPolicy()).run("work")

    assert result["success"]
    assert len(llm.calls) == 1
    assert result["context"].execution_budget is None
    assert pattern.runtime.last_checkpoint["label"] == "final"
    assert pattern.runtime.last_checkpoint["context"]["execution_budget"] is None
    assert not pattern.runtime.outbound_messages


@pytest.mark.asyncio
async def test_resume_cannot_reset_or_relax_budget_but_new_completed_turn_can():
    old_context = ExecutionContext()
    old_context.add_user_message("work", metadata={"turn_id": "one"})
    old_context.execution_budget = ExecutionBudget(
        policy=ExecutionBudgetPolicy(max_tokens=100), used_tokens=100, turn_id="one"
    ).model_dump()
    llm = MeteredLLM()
    pattern = LoopPattern()
    first = await runner(pattern, llm, ExecutionBudgetPolicy(max_tokens=1000)).run(
        None, checkpoint={"context": old_context.to_dict()}
    )
    assert not llm.calls
    assert first["context"].execution_budget["policy"]["max_tokens"] == 100
    completed_context = first["context"]
    completed_context.add_user_message("next", metadata={"turn_id": "two"})
    second = await runner(LoopPattern(), llm, ExecutionBudgetPolicy(max_tokens=60)).run(
        None, checkpoint={"context": completed_context.to_dict()}
    )
    assert len(llm.calls) == 2
    assert second["context"].execution_budget["used_tokens"] == 60


@pytest.mark.asyncio
async def test_user_metadata_cannot_supply_budget_or_reset_consumption():
    pattern = LoopPattern()
    result = await runner(
        pattern, MeteredLLM(), ExecutionBudgetPolicy(max_tokens=30)
    ).run(
        "work",
        metadata={
            "execution_budget": {"used_tokens": 0, "policy": {"max_tokens": 99999}},
            "request_context": {"execution_budget": None},
        },
    )
    assert result["context"].execution_budget["used_tokens"] == 30


@pytest.mark.asyncio
async def test_existing_quota_gate_takes_precedence():
    llm = MeteredLLM()
    result = await runner(LoopPattern(), llm, ExecutionBudgetPolicy(max_tokens=30)).run(
        "work", interrupt_checker=lambda: "quota unavailable"
    )
    assert result["status"] == "interrupted"
    assert "termination_reason" not in result
    assert not llm.calls


@pytest.mark.asyncio
async def test_observer_counts_nested_contexts_and_cached_input_once():
    budget = ExecutionBudget(policy=ExecutionBudgetPolicy(max_tokens=100))
    token = token_usage_observer.set(budget.record_usage)
    try:
        with TokenContextManager():
            add_token_usage(input_tokens=30, output_tokens=5, cached_input_tokens=20)
            with TokenContextManager():
                add_token_usage(input_tokens=40, output_tokens=10)
        assert budget.used_tokens == 85
    finally:
        token_usage_observer.reset(token)


@pytest.mark.asyncio
async def test_exhausted_budget_blocks_llm_and_tool_before_invocation():
    budget = ExecutionBudget(policy=ExecutionBudgetPolicy(max_tokens=1), used_tokens=1)
    token = active_execution_budget.set(budget)
    try:
        llm = MeteredLLM()
        with pytest.raises(LLMCallInterrupted):
            await PatternRuntime().run_llm_call(llm, messages=[])
        with pytest.raises(ToolCallInterrupted):
            await PatternRuntime().run_tool_call(
                lambda: pytest.fail("tool must not run")
            )
        assert not llm.calls
    finally:
        active_execution_budget.reset(token)


@pytest.mark.parametrize(
    "values",
    [
        {"max_tokens": 0},
        {"max_tokens": -1},
        {"max_tokens": True},
        {"soft_limit_percent": 100},
        {"soft_limit_percent": 0},
    ],
)
def test_invalid_policies_are_rejected(values):
    with pytest.raises(ValidationError):
        ExecutionBudgetPolicy(**values)


def test_context_snapshot_does_not_share_budget_containers():
    context = ExecutionContext(
        execution_budget=ExecutionBudget(
            policy=ExecutionBudgetPolicy(max_tokens=100)
        ).model_dump()
    )
    payload = context.to_dict()
    context.execution_budget["used_tokens"] = 50
    assert ExecutionContext.from_dict(payload).execution_budget["used_tokens"] == 0


@pytest.mark.asyncio
async def test_nested_runner_cannot_get_a_separate_allowance():
    llm = MeteredLLM(tokens=30)

    class ParentPattern:
        async def run(self, context, runtime, **kwargs):
            await runtime.run_llm_call(llm, messages=[])
            child = runner(LoopPattern(), llm, ExecutionBudgetPolicy(max_tokens=99999))
            child_result = await child.run("child work")
            assert child_result["termination_reason"] == "token_budget"
            assert not active_execution_budget.get().closed
            with pytest.raises(LLMCallInterrupted):
                await runtime.run_llm_call(llm, messages=[])
            return {"success": False, "status": "interrupted"}

    result = await runner(
        ParentPattern(), llm, ExecutionBudgetPolicy(max_tokens=60)
    ).run("parent work")
    assert len(llm.calls) == 2
    assert result["context"].execution_budget["used_tokens"] == 60


@pytest.mark.asyncio
async def test_parallel_calls_share_usage_and_next_batch_is_stopped():
    llm = MeteredLLM(tokens=30)

    class ParallelPattern:
        async def run(self, context, runtime, **kwargs):
            await asyncio.gather(
                *(runtime.run_llm_call(llm, messages=[]) for _ in range(3))
            )
            with pytest.raises(LLMCallInterrupted):
                await runtime.run_llm_call(llm, messages=[])
            return {"success": False, "status": "interrupted"}

    result = await runner(
        ParallelPattern(), llm, ExecutionBudgetPolicy(max_tokens=60)
    ).run("parallel")
    assert len(llm.calls) == 3  # all three were admitted before usage was reported
    assert result["context"].execution_budget["used_tokens"] == 90


@pytest.mark.asyncio
async def test_budget_blocks_native_stream_before_provider_is_called():
    class StreamLLM:
        async def stream_chat(self, **kwargs):
            pytest.fail("exhausted budget must not call streaming provider")
            yield None

    token = active_execution_budget.set(
        ExecutionBudget(policy=ExecutionBudgetPolicy(max_tokens=1), used_tokens=1)
    )
    try:
        with pytest.raises(LLMCallInterrupted):
            await PatternRuntime().run_streaming_llm_call(StreamLLM(), messages=[])
    finally:
        active_execution_budget.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["react", "dag"])
async def test_real_patterns_stop_before_executing_over_budget_tool(mode):
    from tests.core.agent.test_react import FakeTool
    from xagent.core.agent import DAGPattern, ExecutionPlan, PlanStep, ReActPattern

    class WorkingLLM(MeteredLLM):
        async def chat(self, **kwargs):
            await super().chat(**kwargs)
            return {
                "tool_calls": [
                    {
                        "id": str(len(self.calls)),
                        "name": "calculator",
                        "args": {"expression": "1 + 1"},
                    }
                ]
            }

    llm = WorkingLLM(tokens=30)
    tool = FakeTool()
    if mode == "react":
        pattern = ReActPattern(max_iterations=10)
    else:
        pattern = DAGPattern(
            plan_generator=lambda **kwargs: ExecutionPlan(
                steps=[
                    PlanStep(id="first", task="Calculate", tool_names=["calculator"])
                ]
            ),
            max_concurrency=1,
        )
    agent_runner = runner(
        pattern, llm, ExecutionBudgetPolicy(max_tokens=60, soft_limit_percent=50)
    )
    agent_runner.agent.tools = [tool]
    result = await agent_runner.run(
        "Use calculator repeatedly", metadata={"output_language": "Chinese"}
    )
    assert result.get("termination_reason") == "token_budget", result
    assert len(llm.calls) == 2
    assert len(tool.calls) == 1
    assert result["context"].execution_budget["used_tokens"] == 60
    assert "本次执行" in result["output"]


@pytest.mark.asyncio
async def test_budget_stop_hands_over_registered_files_only():
    from unittest.mock import AsyncMock

    agent_runner = runner(
        LoopPattern(), MeteredLLM(), ExecutionBudgetPolicy(max_tokens=100)
    )
    runtime = PatternRuntime(budget_owner=True)
    context = ExecutionContext()
    workspace = SimpleNamespace(
        get_output_files=lambda: [
            {"file_id": "saved-file", "filename": "result.csv"},
            {"file_id": None, "filename": "unregistered.csv"},
        ]
    )
    runtime.checkpoint = AsyncMock()
    runtime.checkpoint_context_tail = AsyncMock()
    result = await agent_runner._finish_budget_stop(
        context, runtime, LoopPattern(), workspace
    )
    assert "[result.csv](file:saved-file)" in result["output"]
    assert "unregistered.csv" not in result["output"]
    assert result["completion_outcome"] == "partial"


@pytest.mark.asyncio
async def test_budget_stop_includes_real_workspace_root_but_not_inputs_or_temp(
    tmp_path,
):
    from unittest.mock import AsyncMock

    from xagent.core.workspace import TaskWorkspace

    workspace = TaskWorkspace("budget-stop", base_dir=str(tmp_path))
    for directory, name in (
        (workspace.workspace_dir, "root-result.csv"),
        (workspace.output_dir, "output-result.csv"),
        (workspace.input_dir, "source.csv"),
        (workspace.temp_dir, "scratch.csv"),
    ):
        path = directory / name
        path.write_text("value\n1\n")
        workspace.register_file(str(path))
    (workspace.workspace_dir / "unregistered.csv").write_text("value\n2\n")
    agent_runner = runner(
        LoopPattern(), MeteredLLM(), ExecutionBudgetPolicy(max_tokens=100)
    )
    runtime = PatternRuntime(budget_owner=True)
    runtime.checkpoint = AsyncMock()
    runtime.checkpoint_context_tail = AsyncMock()

    result = await agent_runner._finish_budget_stop(
        ExecutionContext(), runtime, LoopPattern(), workspace
    )

    for name in ("root-result.csv", "output-result.csv"):
        assert f"[{name}](file:" in result["output"]
    for name in ("source.csv", "scratch.csv", "unregistered.csv"):
        assert name not in result["output"]
    assert result["completion_outcome"] == "partial"


@pytest.mark.asyncio
async def test_final_answer_already_paid_for_is_not_discarded_at_limit():
    from xagent.core.agent import ReActPattern

    class AnswerLLM(MeteredLLM):
        async def chat(self, **kwargs):
            await super().chat(**kwargs)
            return {
                "tool_calls": [
                    {
                        "id": "final",
                        "name": "final_answer",
                        "args": {"answer": "The answer is 42.", "outcome": "completed"},
                    }
                ]
            }

    llm = AnswerLLM(tokens=30)
    result = await runner(
        ReActPattern(), llm, ExecutionBudgetPolicy(max_tokens=20)
    ).run("Answer directly")
    assert result["success"]
    assert result.get("termination_reason") != "token_budget"
    assert "42" in str(result)
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_policy_failure_stops_before_any_calls_and_cleans_context():
    llm = MeteredLLM()
    agent_runner = runner(LoopPattern(), llm, ExecutionBudgetPolicy())

    async def unavailable():
        raise RuntimeError("policy unavailable")

    agent_runner.budget_policy_provider = unavailable
    with pytest.raises(RuntimeError, match="policy unavailable"):
        await agent_runner.run("work")
    assert not llm.calls
    assert active_execution_budget.get() is None
    assert token_usage_observer.get() is None


@pytest.mark.asyncio
async def test_pause_and_new_runner_resume_keep_recorded_consumption():
    llm = MeteredLLM(tokens=30)

    class PausePattern:
        async def run(self, context, runtime, **kwargs):
            self.runtime = runtime
            await runtime.run_llm_call(llm, messages=[])
            await runtime.checkpoint(
                "paused", context=context, pattern=self, status="interrupted"
            )
            return {"success": False, "status": "interrupted"}

    pattern = PausePattern()
    await runner(pattern, llm, ExecutionBudgetPolicy(max_tokens=60)).run("work")
    saved = pattern.runtime.last_checkpoint
    assert saved["context"]["execution_budget"]["used_tokens"] == 30
    result = await runner(LoopPattern(), llm, ExecutionBudgetPolicy(max_tokens=60)).run(
        None, checkpoint=saved
    )
    assert len(llm.calls) == 2
    assert result["context"].execution_budget["used_tokens"] == 60


@pytest.mark.asyncio
async def test_new_user_turn_after_failed_checkpoint_gets_fresh_budget():
    context = ExecutionContext()
    context.add_user_message("new request", metadata={"turn_id": "new-turn"})
    context.execution_budget = ExecutionBudget(
        policy=ExecutionBudgetPolicy(max_tokens=30), used_tokens=30, turn_id="old-turn"
    ).model_dump()
    llm = MeteredLLM()
    result = await runner(LoopPattern(), llm, ExecutionBudgetPolicy(max_tokens=30)).run(
        None, checkpoint={"context": context.to_dict(), "status": "failed"}
    )
    assert len(llm.calls) == 1
    assert result["context"].execution_budget["turn_id"] == "new-turn"


@pytest.mark.asyncio
@pytest.mark.parametrize("held_kind", ["llm", "tool", "stream"])
async def test_budget_refusal_does_not_cancel_an_admitted_sibling(held_kind):
    entered, release = asyncio.Event(), asyncio.Event()
    budget = ExecutionBudget(policy=ExecutionBudgetPolicy(max_tokens=30))
    token = active_execution_budget.set(budget)
    usage_token = token_usage_observer.set(budget.record_usage)
    runtime = PatternRuntime()

    async def held_call(**kwargs):
        entered.set()
        await release.wait()
        add_token_usage(input_tokens=10, output_tokens=1)
        return "paid result"

    class HeldStream:
        async def stream_chat(self, **kwargs):
            result = await held_call(**kwargs)
            yield StreamChunk(type=ChunkType.TOKEN, delta=result)
            yield StreamChunk(type=ChunkType.END)

    try:
        if held_kind == "stream":
            call = runtime.run_streaming_llm_call(HeldStream(), messages=[])
        elif held_kind == "llm":
            call = runtime.run_llm_call(SimpleNamespace(chat=held_call), messages=[])
        else:
            call = runtime.run_tool_call(held_call)
        held = asyncio.create_task(call)
        await asyncio.wait_for(entered.wait(), timeout=5)
        await runtime.run_llm_call(MeteredLLM(), messages=[])
        with pytest.raises(LLMCallInterrupted):
            await runtime.run_llm_call(MeteredLLM(), messages=[])
        with pytest.raises(ToolCallInterrupted):
            await runtime.run_tool_call(lambda: pytest.fail("new work admitted"))
        assert runtime.budget_stopped
        assert not await runtime.should_interrupt()
        assert not held.done()
        release.set()
        result = await held
        assert result == "paid result"
        assert budget.used_tokens == 41
    finally:
        release.set()
        await asyncio.gather(held, return_exceptions=True)
        token_usage_observer.reset(usage_token)
        active_execution_budget.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "language", ["English", "Simplified Chinese", "Traditional Chinese"]
)
async def test_nested_soft_warning_is_delivered_by_owner_in_its_language(language):
    messages = []

    class ChildPattern:
        async def run(self, context, runtime, **kwargs):
            assert runtime.outbound_message_handler is None
            await asyncio.gather(
                runtime.run_tool_call(lambda: None),
                runtime.run_tool_call(lambda: None),
            )
            assert not runtime.outbound_messages
            return {"success": True, "output": "child done"}

    class ParentPattern:
        async def run(self, context, runtime, llm, **kwargs):
            await runtime.run_llm_call(llm, messages=[])
            await runner(ChildPattern(), llm, ExecutionBudgetPolicy()).run("child")
            await runtime.run_tool_call(lambda: None)
            assert active_execution_budget.get().soft_notified
            return {"success": True, "output": "done"}

    async def deliver(payload):
        await asyncio.sleep(0)
        messages.append(payload)

    await runner(
        ParentPattern(),
        MeteredLLM(60),
        ExecutionBudgetPolicy(max_tokens=100, soft_limit_percent=50),
    ).run(
        "work", metadata={"output_language": language}, outbound_message_handler=deliver
    )
    assert len(messages) == 1
    assert messages[0]["metadata"]["kind"] == "budget_warning"
    assert "60" in messages[0]["message"]
    expected = {
        "English": "approaching",
        "Simplified Chinese": "本次执行",
        "Traditional Chinese": "本次執行",
    }
    assert expected[language] in messages[0]["message"]
    assert "Prioritize" not in messages[0]["message"]


@pytest.mark.asyncio
async def test_undelivered_soft_warning_does_not_fail_work_or_consume_notice(caplog):
    budget = ExecutionBudget(
        policy=ExecutionBudgetPolicy(max_tokens=100), used_tokens=80
    )
    token = active_execution_budget.set(budget)
    delivered = []

    def broken_handler(payload):
        raise RuntimeError("publish failed")

    runtime = PatternRuntime()
    try:
        assert await runtime.run_tool_call(lambda: "first") == "first"
        assert not budget.soft_notified
        runtime.outbound_message_handler = broken_handler
        assert await runtime.run_tool_call(lambda: "second") == "second"
        assert not budget.soft_notified
        assert "Could not deliver execution budget warning" in caplog.text
        runtime.outbound_message_handler = delivered.append
        assert await runtime.run_tool_call(lambda: "third") == "third"
        assert budget.soft_notified
        assert len(delivered) == 1
    finally:
        active_execution_budget.reset(token)


def test_soft_limit_prompt_prefix_stays_stable_as_usage_grows():
    budget = ExecutionBudget(
        policy=ExecutionBudgetPolicy(max_tokens=100), used_tokens=80
    )
    token = active_execution_budget.set(budget)
    original = {
        "messages": [
            {"role": "system", "content": "stable"},
            {"role": "user", "content": "work"},
        ]
    }
    try:
        first = budget_llm_kwargs(original)
        budget.record_usage(5, 5)
        assert budget_llm_kwargs(original) == first
        assert original["messages"][0]["content"] == "stable"
    finally:
        active_execution_budget.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["user_stop", "sibling_usage"])
async def test_warning_delivery_rechecks_admission_after_yield(reason):
    entered, release = asyncio.Event(), asyncio.Event()
    budget = ExecutionBudget(
        policy=ExecutionBudgetPolicy(max_tokens=100), used_tokens=80
    )
    token = active_execution_budget.set(budget)

    async def deliver(payload):
        entered.set()
        await release.wait()

    runtime = PatternRuntime(outbound_message_handler=deliver)
    work = asyncio.create_task(
        runtime.run_tool_call(lambda: pytest.fail("work admitted after stop"))
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        if reason == "user_stop":
            runtime.request_interrupt("user stopped")
        else:
            budget.record_usage(20, 0)
        release.set()
        with pytest.raises(ToolCallInterrupted):
            await work
        assert runtime.budget_stopped == (reason == "sibling_usage")
    finally:
        release.set()
        await asyncio.gather(work, return_exceptions=True)
        active_execution_budget.reset(token)


@pytest.mark.asyncio
async def test_budget_handoff_excludes_old_turn_files_but_keeps_resumed_turn_files(
    tmp_path,
):
    import os
    from unittest.mock import AsyncMock

    from xagent.core.workspace import TaskWorkspace

    workspace = TaskWorkspace("turn-files", base_dir=str(tmp_path))
    path = workspace.output_dir / "result.csv"
    path.write_text("old\n")
    workspace.register_file(str(path))
    os.utime(path, (100, 100))
    budget = ExecutionBudget(
        policy=ExecutionBudgetPolicy(max_tokens=10), started_at=200
    )
    runtime = PatternRuntime(budget_owner=True)
    runtime.checkpoint = AsyncMock()
    runtime.checkpoint_context_tail = AsyncMock()
    agent_runner = runner(LoopPattern(), MeteredLLM(), budget.policy)
    token = active_execution_budget.set(budget)
    try:
        result = await agent_runner._finish_budget_stop(
            ExecutionContext(), runtime, LoopPattern(), workspace
        )
        assert result["completion_outcome"] == "blocked"
        assert "result.csv" not in result["output"]
        path.write_text("updated\n")
        os.utime(path, (300, 300))
        # A new process restores the original turn boundary, not the resume time.
        resumed = ExecutionBudget.model_validate(budget.model_dump())
        active_execution_budget.set(resumed)
        result = await agent_runner._finish_budget_stop(
            ExecutionContext(), runtime, LoopPattern(), workspace
        )
        assert result["completion_outcome"] == "partial"
        assert "[result.csv](file:" in result["output"]
        assert resumed.started_at == 200
    finally:
        active_execution_budget.reset(token)
