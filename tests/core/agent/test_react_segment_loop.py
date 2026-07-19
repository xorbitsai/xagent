"""Inc.5 — segment loop integrated into _execute_pending_tool_calls (§4.2.3).

Drives _execute_pending_tool_calls directly with fakes to pin the integrated
behavior:
- I4: concurrency-safe tools before a control tool run as a batch, then the
  control tool short-circuits (final_answer -> completed; ask_user_question ->
  waiting_for_user); later tools do not run.
- I5 (interrupt half): an interrupt at the segment boundary stops the loop with
  pending_tool_calls preserved.
- mixed S/U ordering: a concurrent batch, then a serial unsafe tool, results
  back-filled in input order.
- batch checkpoints: before_tool_batch / after_tool_batch are emitted for a
  concurrent segment; before_tool / after_tool for a serial one.
- repeated-tool-decision is evaluated once per segment.
"""

from __future__ import annotations

import asyncio

import pytest

from tests.core.agent.concurrency_harness import (
    FakeRuntime,
    FakeTool,
    RecordingContext,
    make_react,
    make_tool_call,
)
from xagent.core.agent import PatternRuntime, ToolCallInterrupted


def _checkpoint_statuses(runtime: FakeRuntime) -> list[str]:
    return [payload["status"] for payload in runtime.events_of("checkpoint")]


def _make_pattern(**kwargs):
    # Disable repeated-tool-decision thresholds so it never fires unless a test
    # opts in; set task_text so final_answer's memory step is a no-op.
    pattern = make_react(
        parallel=True,
        repeated_tool_decision_after_consecutive_tool_calls=None,
        repeated_tool_decision_after_consecutive_work_tool_calls=None,
        **kwargs,
    )
    pattern.task_text = "t"
    return pattern


async def test_concurrent_batch_then_final_answer_short_circuits() -> None:
    tools = [FakeTool("s1", read_only=True), FakeTool("s2", read_only=True)]
    pattern = _make_pattern()
    pattern.pending_tool_calls = [
        make_tool_call("s1"),
        make_tool_call("s2"),
        make_tool_call("final_answer", {"answer": "done"}),
    ]
    context = RecordingContext()
    runtime = FakeRuntime()

    result = await pattern._execute_pending_tool_calls(
        context=context, tools=tools, llm=None, runtime=runtime
    )

    assert result is not None
    assert result.get("status") == "completed"
    assert pattern.pending_tool_calls == []
    # s1, s2 ran (in order) then final_answer recorded its answer.
    assert [r["tool_name"] for r in context.tool_results] == [
        "s1",
        "s2",
        "final_answer",
    ]
    statuses = _checkpoint_statuses(runtime)
    assert "before_tool_batch" in statuses
    assert "after_tool_batch" in statuses


async def test_serial_then_ask_user_waits() -> None:
    # A lone safe tool degrades to serial; ask_user_question then waits.
    tools = [FakeTool("s1", read_only=True)]
    pattern = _make_pattern()
    pattern.pending_tool_calls = [
        make_tool_call("s1"),
        make_tool_call("ask_user_question", {"message": "which?"}),
        make_tool_call("s2"),  # must NOT run after waiting_for_user
    ]
    context = RecordingContext()
    runtime = FakeRuntime()

    result = await pattern._execute_pending_tool_calls(
        context=context, tools=tools, llm=None, runtime=runtime
    )

    assert result is not None
    assert result.get("status") == "waiting_for_user"
    assert [r["tool_name"] for r in context.tool_results] == [
        "s1",
        "ask_user_question",
    ]
    statuses = _checkpoint_statuses(runtime)
    assert "before_tool" in statuses  # serial path used for the lone safe tool


async def test_interrupt_before_batch_preserves_pending() -> None:
    tools = [FakeTool("s1", read_only=True), FakeTool("s2", read_only=True)]
    pattern = _make_pattern()
    pending = [make_tool_call("s1"), make_tool_call("s2")]
    pattern.pending_tool_calls = list(pending)
    context = RecordingContext()
    runtime = FakeRuntime(interrupt=True, interrupt_reason="stop")

    result = await pattern._execute_pending_tool_calls(
        context=context, tools=tools, llm=None, runtime=runtime
    )

    assert result is not None
    assert result.get("status") == "interrupted"
    # Nothing executed; the whole batch is still pending for resume.
    assert context.tool_results == []
    assert [tc["name"] for tc in pattern.pending_tool_calls] == ["s1", "s2"]


async def test_interrupt_after_capped_batch_preserves_remaining() -> None:
    # The batch cap (= max_concurrency) turns a long safe run into multiple
    # batches, so an interrupt requested after the first batch leaves the rest
    # pending. Without the cap all four would run as one uninterruptible batch.
    class _InterruptAfterFirstBatch(FakeRuntime):
        async def should_interrupt(self) -> bool:
            return len(self.events_of("on_tool_end")) >= 2

    tools = [FakeTool(name, read_only=True) for name in ("s1", "s2", "s3", "s4")]
    pattern = _make_pattern(max_concurrency=2)
    pattern.pending_tool_calls = [
        make_tool_call(name) for name in ("s1", "s2", "s3", "s4")
    ]
    context = RecordingContext()
    runtime = _InterruptAfterFirstBatch()

    result = await pattern._execute_pending_tool_calls(
        context=context, tools=tools, llm=None, runtime=runtime
    )

    assert result is not None
    assert result.get("status") == "interrupted"
    # Only the first capped batch ran; the remainder is preserved for resume.
    assert [r["tool_name"] for r in context.tool_results] == ["s1", "s2"]
    assert [tc["name"] for tc in pattern.pending_tool_calls] == ["s3", "s4"]


async def test_interrupt_during_batch_preserves_completed_results() -> None:
    blocked = asyncio.Event()
    tools = [
        FakeTool("s1", read_only=True),
        FakeTool("s2", read_only=True, gate=blocked),
    ]
    pattern = _make_pattern(max_concurrency=2)
    calls = [make_tool_call("s1"), make_tool_call("s2")]
    pattern.pending_tool_calls = list(calls)
    context = RecordingContext()
    runtime = PatternRuntime(execution_id="partial-batch-interrupt")

    task = asyncio.create_task(
        pattern._execute_pending_tool_calls(
            context=context,
            tools=tools,
            llm=None,
            runtime=runtime,
        )
    )
    while (
        pattern.tool_ledger.get(calls[0]["id"]) is None
        or pattern.tool_ledger[calls[0]["id"]].status != "completed"
        or not tools[1].calls
    ):
        await asyncio.sleep(0)

    runtime.request_interrupt("pause partial batch")
    with pytest.raises(ToolCallInterrupted, match="pause partial batch"):
        await task

    assert [result["tool_name"] for result in context.tool_results] == ["s1"]
    assert [call["name"] for call in pattern.pending_tool_calls] == ["s2"]
    assert pattern.tool_ledger[calls[0]["id"]].status == "completed"
    assert pattern.tool_ledger[calls[1]["id"]].status == "interrupted"


async def test_concurrent_batch_then_unsafe_serial_preserves_order() -> None:
    tools = [
        FakeTool("s1", read_only=True),
        FakeTool("s2", read_only=True),
        FakeTool("u1", concurrency_safe=False),
    ]
    pattern = _make_pattern()
    pattern.pending_tool_calls = [
        make_tool_call("s1"),
        make_tool_call("s2"),
        make_tool_call("u1"),
    ]
    context = RecordingContext()
    runtime = FakeRuntime()

    result = await pattern._execute_pending_tool_calls(
        context=context, tools=tools, llm=None, runtime=runtime
    )

    assert result is None  # no control tool, loop drains pending
    assert pattern.pending_tool_calls == []
    assert [r["tool_name"] for r in context.tool_results] == ["s1", "s2", "u1"]
    statuses = _checkpoint_statuses(runtime)
    assert "before_tool_batch" in statuses  # s1+s2 batch
    assert "before_tool" in statuses  # u1 serial


async def test_concurrent_mcp_error_results_do_not_force_final_answer() -> None:
    mcp_error = {"content": [{"text": "failed"}], "is_error": True}
    tools = [
        FakeTool("s1", read_only=True, result=mcp_error),
        FakeTool("s2", read_only=True, result=mcp_error),
    ]
    pattern = _make_pattern(finalize_after_tool_result=True)
    pattern.pending_tool_calls = [make_tool_call("s1"), make_tool_call("s2")]

    result = await pattern._execute_pending_tool_calls(
        context=RecordingContext(), tools=tools, llm=None, runtime=FakeRuntime()
    )

    assert result is None
    assert pattern.force_final_answer_next is False


async def test_flag_off_runs_everything_serially() -> None:
    # With the flag off, two safe tools each run as their own serial segment.
    tools = [FakeTool("s1", read_only=True), FakeTool("s2", read_only=True)]
    pattern = make_react(
        parallel=False,
        repeated_tool_decision_after_consecutive_tool_calls=None,
        repeated_tool_decision_after_consecutive_work_tool_calls=None,
    )
    pattern.task_text = "t"
    pattern.pending_tool_calls = [make_tool_call("s1"), make_tool_call("s2")]
    context = RecordingContext()
    runtime = FakeRuntime()

    await pattern._execute_pending_tool_calls(
        context=context, tools=tools, llm=None, runtime=runtime
    )

    statuses = _checkpoint_statuses(runtime)
    assert "before_tool_batch" not in statuses
    assert statuses.count("before_tool") == 2
    assert statuses.count("after_tool") == 2
    assert [r["tool_name"] for r in context.tool_results] == ["s1", "s2"]


# --- Inc.6: checkpoint / resume of the concurrency knobs --------------------


def test_get_state_load_state_round_trips_concurrency_fields() -> None:
    pattern = make_react(parallel=True, max_concurrency=5)
    state = pattern.get_state()
    assert state["tool_parallel_enabled"] is True
    assert state["tool_max_concurrency"] == 5

    restored = make_react(parallel=False, max_concurrency=1)
    restored.load_state(state)
    assert restored.tool_parallel_enabled is True
    assert restored.tool_max_concurrency == 5


async def test_crash_during_batch_keeps_segment_pending_for_resume() -> None:
    # If the process dies mid-batch (before backfill/dequeue), the whole segment
    # must remain pending so resume re-executes it (concurrency-safe tools are
    # idempotent). Backfill is where the crash is simulated.
    class ExplodingContext(RecordingContext):
        def add_tool_result(self, *args, **kwargs):
            raise RuntimeError("crash during backfill")

    tools = [FakeTool("s1", read_only=True), FakeTool("s2", read_only=True)]
    pattern = _make_pattern()
    pattern.pending_tool_calls = [make_tool_call("s1"), make_tool_call("s2")]

    with pytest.raises(RuntimeError, match="crash during backfill"):
        await pattern._execute_pending_tool_calls(
            context=ExplodingContext(), tools=tools, llm=None, runtime=FakeRuntime()
        )

    # Segment was not dequeued -> available for re-execution on resume.
    assert [tc["name"] for tc in pattern.pending_tool_calls] == ["s1", "s2"]
