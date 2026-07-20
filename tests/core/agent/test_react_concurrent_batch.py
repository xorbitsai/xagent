"""Inc.3 — concurrent batch execution + ordered backfill (design §4.2.2, §5.5).

``_run_concurrent_batch`` runs a segment of concurrency-safe tool calls under a
Semaphore via ``asyncio.gather`` and back-fills their results into the context
in the original tool-call order. Invariants pinned here:

- I1: ``add_tool_result`` order == input order, even when tools finish out of
  order.
- I2: every ``tool_call_id`` gets exactly one result (including failures).
- real concurrency: the batch overlaps (peak == batch size) rather than running
  serially, and the Semaphore caps the peak.
- exception isolation: one failing tool yields an error result for that call
  while the rest succeed.
"""

from __future__ import annotations

import asyncio

import pytest

from tests.core.agent.concurrency_harness import (
    ConcurrencyTracker,
    FakeRuntime,
    FakeTool,
    RecordingContext,
    make_react,
    make_tool_call,
)
from xagent.core.agent import PatternRuntime, ToolCallInterrupted


async def test_ordered_backfill_despite_out_of_order_completion() -> None:
    names = ["s1", "s2", "s3"]
    tracker = ConcurrencyTracker()
    gates = {name: asyncio.Event() for name in names}
    tools = [
        FakeTool(name, concurrency_safe=True, gate=gates[name], tracker=tracker)
        for name in names
    ]
    pattern = make_react(parallel=True, max_concurrency=3)
    runtime = FakeRuntime()
    context = RecordingContext()
    batch = [make_tool_call(name) for name in names]

    task = asyncio.create_task(
        pattern._run_concurrent_batch(batch, tools, runtime, context)
    )
    # Wait until all three are in-flight, then release them in REVERSE order so
    # completion order is the opposite of input order.
    while tracker.active < 3:
        await asyncio.sleep(0)
    for index, name in enumerate(["s3", "s2", "s1"], start=1):
        gates[name].set()
        while len(tracker.leave_order) < index:
            await asyncio.sleep(0)
    await task

    # Completion was reverse, but backfill preserves input order (I1).
    assert tracker.leave_order == ["s3", "s2", "s1"]
    assert [r["tool_name"] for r in context.tool_results] == names
    assert [r["tool_call_id"] for r in context.tool_results] == [
        tc["id"] for tc in batch
    ]


async def test_every_tool_call_id_gets_exactly_one_result() -> None:
    names = ["s1", "s2", "s3"]
    tools = [FakeTool(name, concurrency_safe=True) for name in names]
    pattern = make_react(parallel=True, max_concurrency=3)
    context = RecordingContext()
    batch = [make_tool_call(name) for name in names]

    await pattern._run_concurrent_batch(batch, tools, FakeRuntime(), context)

    ids = [r["tool_call_id"] for r in context.tool_results]
    assert sorted(ids) == sorted(tc["id"] for tc in batch)
    assert len(ids) == len(set(ids)) == 3


async def test_batch_runs_concurrently() -> None:
    names = ["s1", "s2", "s3"]
    tracker = ConcurrencyTracker()
    tools = [
        FakeTool(name, concurrency_safe=True, delay=0.02, tracker=tracker)
        for name in names
    ]
    pattern = make_react(parallel=True, max_concurrency=3)
    context = RecordingContext()
    batch = [make_tool_call(name) for name in names]

    await pattern._run_concurrent_batch(batch, tools, FakeRuntime(), context)

    # All three overlapped (a serial run would never exceed 1).
    assert tracker.peak == 3


async def test_semaphore_caps_concurrency() -> None:
    names = ["s1", "s2", "s3", "s4"]
    tracker = ConcurrencyTracker()
    tools = [
        FakeTool(name, concurrency_safe=True, delay=0.02, tracker=tracker)
        for name in names
    ]
    pattern = make_react(parallel=True, max_concurrency=2)
    context = RecordingContext()
    batch = [make_tool_call(name) for name in names]

    await pattern._run_concurrent_batch(batch, tools, FakeRuntime(), context)

    assert tracker.peak <= 2
    assert len(context.tool_results) == 4


async def test_exception_isolation_within_batch() -> None:
    tools = [
        FakeTool("s1", concurrency_safe=True),
        FakeTool("boom", concurrency_safe=True, raises=RuntimeError("kaboom")),
        FakeTool("s3", concurrency_safe=True),
    ]
    pattern = make_react(parallel=True, max_concurrency=3)
    context = RecordingContext()
    batch = [make_tool_call(name) for name in ["s1", "boom", "s3"]]

    results = await pattern._run_concurrent_batch(batch, tools, FakeRuntime(), context)

    # All three back-filled in order; the middle one is an error result.
    assert [r["tool_name"] for r in context.tool_results] == ["s1", "boom", "s3"]
    assert results[0]["success"] is True
    assert results[1]["success"] is False
    assert "kaboom" in results[1]["error"]
    assert results[2]["success"] is True


async def test_infra_callback_failure_marks_ledger_terminal() -> None:
    # If an infra callback (on_tool_start) raises after the "running" record is
    # written, the ledger must reach a terminal state. The consecutive-count
    # walks only count {completed, failed}, so a record stuck at "running" would
    # be silently skipped and undercount the repeated-tool-decision triggers.
    class _BoomOnStart(FakeRuntime):
        async def on_tool_start(self, *, tool_call: dict) -> None:
            raise RuntimeError("trace backend down")

    tools = [FakeTool("s1", concurrency_safe=True)]
    pattern = make_react(parallel=True)
    call = make_tool_call("s1")

    with pytest.raises(RuntimeError, match="trace backend down"):
        await pattern._execute_tool_safely(call, tools, _BoomOnStart())

    assert pattern.tool_ledger[call["id"]].status == "failed"


async def test_concurrent_batch_assigns_ids_to_id_less_calls() -> None:
    # A tool call without an id must get a stable one stamped on the *original*
    # dict before _execute_tool_safely's _with_* transforms run. _record_tool_call
    # only generates a fallback key internally; if it is not written back, the
    # key drifts between the running/completed writes (len(tool_ledger) grows in
    # between) and never matches the still-id-less dict that _backfill_result and
    # _reorder_ledger_for_batch read, desyncing the ledger from the context
    # (I2/I3).
    tools = [
        FakeTool("s1", concurrency_safe=True),
        FakeTool("s2", concurrency_safe=True),
    ]
    pattern = make_react(parallel=True, max_concurrency=2)
    runtime = FakeRuntime()
    # An active ReAct step makes _with_runtime_step return a *copy*, so stamping
    # the id after the transform (rather than before) would miss the original
    # batch dict that backfill/reorder iterate over.
    runtime.active_react_step_id = "step-x"
    context = RecordingContext()
    batch = [make_tool_call("s1", id=""), make_tool_call("s2", id="")]

    await pattern._run_concurrent_batch(batch, tools, runtime, context)

    # Exactly one terminal record per call (no orphan stuck at "running").
    assert len(pattern.tool_ledger) == 2
    assert all(record.status == "completed" for record in pattern.tool_ledger.values())
    # Context tool_call_ids are non-empty and match the ledger keys in input
    # order (I2 + I3).
    ctx_ids = [r["tool_call_id"] for r in context.tool_results]
    assert all(ctx_ids)
    assert ctx_ids == list(pattern.tool_ledger.keys())


async def test_concurrent_batch_propagates_infra_callback_failure() -> None:
    # An infra callback failure (on_tool_start) is a real exception, not a tool
    # failure. The serial path lets it propagate and halt the turn; the
    # concurrent path must do the same instead of mis-reporting infrastructure
    # breakage to the model as a tool-failure result. The exception is re-raised
    # after successful neighbors are back-filled so an explicit retry cannot
    # repeat work that already completed.
    class _BoomOnFirst(FakeRuntime):
        async def on_tool_start(self, *, tool_call: dict) -> None:
            if tool_call["name"] == "boom":
                raise RuntimeError("trace down")
            await super().on_tool_start(tool_call=tool_call)

    tools = [
        FakeTool("s1", concurrency_safe=True),
        FakeTool("boom", concurrency_safe=True),
        FakeTool("s3", concurrency_safe=True),
    ]
    pattern = make_react(parallel=True, max_concurrency=3)
    context = RecordingContext()
    batch = [make_tool_call(name) for name in ["s1", "boom", "s3"]]

    with pytest.raises(RuntimeError, match="trace down"):
        await pattern._run_concurrent_batch(batch, tools, _BoomOnFirst(), context)

    assert [result["tool_name"] for result in context.tool_results] == ["s1", "s3"]
    # The failing call still reaches a terminal ledger state (I3 walk).
    assert pattern.tool_ledger[batch[1]["id"]].status == "failed"


async def test_mixed_infra_failure_and_interrupt_reconciles_batch_before_raise() -> (
    None
):
    class _BoomOnStart(PatternRuntime):
        async def on_tool_start(self, *, tool_call: dict) -> None:
            if tool_call["name"] == "boom":
                raise RuntimeError("trace down")
            await super().on_tool_start(tool_call=tool_call)

    blocked = asyncio.Event()
    tools = [
        FakeTool("done", concurrency_safe=True),
        FakeTool("boom", concurrency_safe=True),
        FakeTool("paused", concurrency_safe=True, gate=blocked),
    ]
    pattern = make_react(parallel=True, max_concurrency=3)
    context = RecordingContext()
    batch = [make_tool_call(name) for name in ("done", "boom", "paused")]
    pattern.pending_tool_calls = list(batch)
    runtime = _BoomOnStart(execution_id="mixed-batch-failure")

    task = asyncio.create_task(
        pattern._run_concurrent_batch(batch, tools, runtime, context)
    )
    while not tools[0].calls or not tools[2].calls:
        await asyncio.sleep(0)
    runtime.request_interrupt("pause mixed batch")

    with pytest.raises(RuntimeError, match="trace down"):
        await task

    assert [result["tool_name"] for result in context.tool_results] == ["done"]
    assert [call["name"] for call in pattern.pending_tool_calls] == [
        "boom",
        "paused",
    ]
    assert pattern.tool_ledger[batch[0]["id"]].status == "completed"
    assert pattern.tool_ledger[batch[1]["id"]].status == "failed"
    assert pattern.tool_ledger[batch[2]["id"]].status == "interrupted"


async def test_interrupt_filter_uses_batch_position_when_ids_repeat() -> None:
    blocked = asyncio.Event()
    tools = [
        FakeTool("done", concurrency_safe=True),
        FakeTool("paused", concurrency_safe=True, gate=blocked),
    ]
    pattern = make_react(parallel=True, max_concurrency=2)
    context = RecordingContext()
    batch = [
        make_tool_call("done", id="provider-duplicate"),
        make_tool_call("paused", id="provider-duplicate"),
    ]
    pattern.pending_tool_calls = list(batch)
    runtime = PatternRuntime(execution_id="duplicate-provider-ids")

    task = asyncio.create_task(
        pattern._run_concurrent_batch(batch, tools, runtime, context)
    )
    while not tools[0].calls or not tools[1].calls:
        await asyncio.sleep(0)
    runtime.request_interrupt("pause duplicate ids")

    with pytest.raises(ToolCallInterrupted, match="pause duplicate ids"):
        await task

    assert [result["tool_name"] for result in context.tool_results] == ["done"]
    assert pattern.pending_tool_calls == [batch[1]]


# --- Inc.4: tool_ledger ordering after a concurrent batch (I3) -------------


def test_reorder_ledger_for_batch_restores_input_order() -> None:
    # Simulate interleaved completion writing batch records out of order, with
    # a pre-existing record before the batch that must keep its position.
    pattern = make_react(parallel=True)
    pattern._record_tool_call(
        make_tool_call("earlier", id="e0"), status="completed", result={"success": True}
    )
    batch = [
        make_tool_call("s1", id="b1"),
        make_tool_call("s2", id="b2"),
        make_tool_call("s3", id="b3"),
    ]
    for tool_call in (batch[2], batch[0], batch[1]):  # scrambled order
        pattern._record_tool_call(
            tool_call, status="completed", result={"success": True}
        )
    assert list(pattern.tool_ledger.keys()) == ["e0", "b3", "b1", "b2"]

    pattern._reorder_ledger_for_batch(batch)

    assert list(pattern.tool_ledger.keys()) == ["e0", "b1", "b2", "b3"]


async def test_concurrent_batch_keeps_ledger_in_input_order() -> None:
    names = ["s1", "s2", "s3"]
    tracker = ConcurrencyTracker()
    gates = {name: asyncio.Event() for name in names}
    tools = [
        FakeTool(name, concurrency_safe=True, gate=gates[name], tracker=tracker)
        for name in names
    ]
    pattern = make_react(parallel=True, max_concurrency=3)
    context = RecordingContext()
    batch = [make_tool_call(name) for name in names]

    task = asyncio.create_task(
        pattern._run_concurrent_batch(batch, tools, FakeRuntime(), context)
    )
    while tracker.active < 3:
        await asyncio.sleep(0)
    for index, name in enumerate(["s3", "s2", "s1"], start=1):  # reverse completion
        gates[name].set()
        while len(tracker.leave_order) < index:
            await asyncio.sleep(0)
    await task

    assert [tc["id"] for tc in batch] == list(pattern.tool_ledger.keys())


async def test_concurrent_batch_updates_consecutive_group_count() -> None:
    group = "search"
    names = ["s1", "s2", "s3"]
    tools = [
        FakeTool(name, concurrency_safe=True, decision_group=group) for name in names
    ]
    pattern = make_react(parallel=True, max_concurrency=3)
    pattern._tool_decision_groups_by_name = pattern._tool_decision_groups_for_tools(
        tools
    )
    context = RecordingContext()
    batch = [make_tool_call(name) for name in names]

    await pattern._run_concurrent_batch(batch, tools, FakeRuntime(), context)

    assert pattern._consecutive_successful_tool_group_count(group) == 3


async def test_mcp_error_result_stops_consecutive_successful_group_count() -> None:
    group = "search"
    tool = FakeTool("s1", concurrency_safe=True, decision_group=group)
    pattern = make_react(parallel=True)
    pattern._tool_decision_groups_by_name = pattern._tool_decision_groups_for_tools(
        [tool]
    )
    pattern._record_tool_call(
        make_tool_call("s1"),
        status="completed",
        result={"content": [{"text": "failed"}], "is_error": True},
    )

    assert pattern._consecutive_successful_tool_group_count(group) == 0


async def test_concurrent_batch_with_failure_counts_work_calls() -> None:
    tools = [
        FakeTool("s1", concurrency_safe=True),
        FakeTool("boom", concurrency_safe=True, raises=RuntimeError("x")),
        FakeTool("s3", concurrency_safe=True),
    ]
    pattern = make_react(parallel=True, max_concurrency=3)
    pattern._tool_decision_groups_by_name = pattern._tool_decision_groups_for_tools(
        tools
    )
    context = RecordingContext()
    batch = [make_tool_call(name) for name in ["s1", "boom", "s3"]]

    await pattern._run_concurrent_batch(batch, tools, FakeRuntime(), context)

    # completed + failed both count as work calls.
    assert pattern._consecutive_work_tool_call_count() == 3
