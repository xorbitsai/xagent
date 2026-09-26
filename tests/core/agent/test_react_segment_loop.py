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
- forced-answer reads: on a forced turn that offers read_tool_result, each
  read in the batch is admitted or refused on its own before anything runs,
  and results still reach the context in call order.
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
from xagent.core.agent.context.components import SpillRegistryComponent
from xagent.core.agent.pattern.react.react import FORCED_ANSWER_READS_USED_UP_TEXT
from xagent.core.tools.tool_result_spill import SPILL_READ_TOOL_NAME


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
        "s2",
    ]
    assert context.tool_results[-1]["result"]["status"] == "cancelled"
    assert pattern.pending_tool_calls == []
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


# --- forced-answer turn: read policy inside one batch ------------------------

_STORED_PATH = "tool-results/calculator-0123456789abcdef0123456789abcdef.json"
_UNLISTED_PATH = "tool-results/unlisted-000000000000.json"


class _StoredResultsContext(RecordingContext):
    """A RecordingContext whose run has stored one result."""

    def __init__(self) -> None:
        super().__init__()
        self._registry = SpillRegistryComponent(
            records=[{"relative_path": _STORED_PATH, "kind": "array"}]
        )

    def get_component(self, name: str):
        return self._registry if name == "spilled_results" else None


class _InterruptAfterTools(FakeRuntime):
    """Asks to stop once a given number of tool calls have finished."""

    def __init__(self, after: int) -> None:
        super().__init__()
        self.after = after

    async def should_interrupt(self) -> bool:
        return len(self.events_of("on_tool_end")) >= self.after


def _read_calls(*paths: str | None) -> list[dict]:
    """One read_tool_result call per path; None makes an "s1" call instead."""
    return [
        make_tool_call("s1")
        if path is None
        else make_tool_call(SPILL_READ_TOOL_NAME, {"path": path})
        for path in paths
    ]


def _forced_read_pattern(*, parallel: bool, read_open: bool = True, counts=(0, 0, 0)):
    """A pattern mid forced turn with the reads open or closed and the two
    read counters and the extra iterations preset to counts."""
    pattern = make_react(
        parallel=parallel,
        max_concurrency=3,
        repeated_tool_decision_after_consecutive_tool_calls=None,
        repeated_tool_decision_after_consecutive_work_tool_calls=None,
    )
    pattern.task_text = "t"
    pattern.force_final_answer_next = True
    pattern._forced_answer_read_open = read_open
    (
        pattern.forced_answer_reads_used,
        pattern.forced_answer_reads_rejected,
        pattern.forced_answer_extra_iterations,
    ) = counts
    return pattern


def _resumed(pattern, *, parallel: bool):
    """A fresh pattern restored from pattern's checkpoint state; its own
    read-open flag starts False so only the checkpoint can open it."""
    restored = _forced_read_pattern(parallel=parallel, read_open=False)
    restored.load_state(pattern.get_state())
    return restored


def _forced_read_counts(pattern) -> tuple[int, int, int]:
    return (
        pattern.forced_answer_reads_used,
        pattern.forced_answer_reads_rejected,
        pattern.forced_answer_extra_iterations,
    )


async def _drain(pattern, tools, *, context=None, runtime=None):
    """Execute the pattern's pending calls on a stored-results context with a
    recording runtime; returns the result and the context."""
    context = context if context is not None else _StoredResultsContext()
    result = await pattern._execute_pending_tool_calls(
        context=context,
        tools=tools,
        llm=None,
        runtime=runtime if runtime is not None else FakeRuntime(),
    )
    return result, context


def _reader_and_other() -> tuple[FakeTool, FakeTool]:
    """A concurrency-safe read_tool_result and a concurrency-safe "s1"."""
    return (
        FakeTool(SPILL_READ_TOOL_NAME, read_only=True),
        FakeTool("s1", read_only=True),
    )


@pytest.mark.parametrize(
    "cell", ["serial", "concurrent", "mixed_with_other_call", "resumed"]
)
async def test_forced_turn_policy_applies_within_one_batch(cell: str) -> None:
    """Eight reads in one forced-turn response: the first three run, the
    other five get the used-up observation without running, and every call
    gets its result in call order."""
    reader, other = _reader_and_other()
    pattern = _forced_read_pattern(parallel=cell != "serial")
    calls = _read_calls(*[_STORED_PATH] * 8)
    if cell == "mixed_with_other_call":
        calls.insert(2, make_tool_call("s1"))
    if cell == "resumed":
        pattern = _resumed(pattern, parallel=True)
    pattern.pending_tool_calls = list(calls)

    result, context = await _drain(pattern, [reader, other])

    assert result is None
    assert pattern.pending_tool_calls == []
    assert len(reader.calls) == 3
    assert len(other.calls) == (1 if cell == "mixed_with_other_call" else 0)
    assert [entry["tool_call_id"] for entry in context.tool_results] == [
        call["id"] for call in calls
    ]
    refused = [
        entry["tool_call_id"]
        for entry in context.tool_results
        if entry["result"] == {"output": FORCED_ANSWER_READS_USED_UP_TEXT}
    ]
    read_ids = [call["id"] for call in calls if call["name"] == SPILL_READ_TOOL_NAME]
    assert refused == read_ids[3:]
    assert _forced_read_counts(pattern) == (3, 0, 3)


@pytest.mark.parametrize("parallel", [False, True])
async def test_ordinary_turn_reads_are_not_counted(parallel: bool) -> None:
    """Outside a forced turn that offers reads, every read runs and nothing
    is counted."""
    reader, _ = _reader_and_other()
    pattern = _forced_read_pattern(parallel=parallel, read_open=False)
    pattern.force_final_answer_next = False
    pattern.pending_tool_calls = _read_calls(*[_STORED_PATH] * 8)

    result, _ = await _drain(pattern, [reader])

    assert result is None
    assert len(reader.calls) == 8
    assert _forced_read_counts(pattern) == (0, 0, 0)


# --- forced-answer turn: extra iterations ------------------------------------


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "concurrent"])
@pytest.mark.parametrize(
    ("interrupt_after", "remaining_path", "resumed_counts"),
    [
        (1, _STORED_PATH, (2, 0, 2)),
        (2, _UNLISTED_PATH, (2, 1, 3)),
        (3, _STORED_PATH, (3, 0, 3)),
        (3, _UNLISTED_PATH, (3, 0, 3)),
    ],
)
async def test_forced_answer_extra_iterations_survive_interrupt_and_resume(
    parallel: bool,
    interrupt_after: int,
    remaining_path: str,
    resumed_counts: tuple[int, int, int],
) -> None:
    """Every admitted read has already counted when the batch is interrupted,
    and the resumed turn counts only what it does itself: a read refused
    because the allowance is used up adds nothing, a path that matches no
    stored result adds a reject and an extra iteration."""
    reader, _ = _reader_and_other()
    pattern = _forced_read_pattern(parallel=parallel)
    pattern.tool_max_concurrency = interrupt_after
    pending_after = _read_calls(remaining_path)
    pattern.pending_tool_calls = [
        *_read_calls(*[_STORED_PATH] * interrupt_after),
        *pending_after,
    ]

    result, context = await _drain(
        pattern, [reader], runtime=_InterruptAfterTools(interrupt_after)
    )

    assert result is not None
    assert result.get("status") == "interrupted"
    assert len(reader.calls) == interrupt_after
    assert _forced_read_counts(pattern) == (interrupt_after, 0, interrupt_after)
    assert pattern.pending_tool_calls == pending_after

    resumed = _resumed(pattern, parallel=parallel)
    result, _ = await _drain(resumed, [reader], context=context)

    assert result is None
    assert resumed.pending_tool_calls == []
    assert _forced_read_counts(resumed) == resumed_counts


async def test_forced_answer_extra_iterations_survive_a_second_interrupt() -> None:
    """A resumed batch that is interrupted again keeps every count it made,
    and the next resume carries on from there."""
    reader, _ = _reader_and_other()
    pattern = _forced_read_pattern(parallel=False)
    pattern.pending_tool_calls = _read_calls(*[_STORED_PATH] * 3)

    _, context = await _drain(pattern, [reader], runtime=_InterruptAfterTools(1))
    assert _forced_read_counts(pattern) == (1, 0, 1)

    resumed = _resumed(pattern, parallel=False)
    result, _ = await _drain(
        resumed, [reader], context=context, runtime=_InterruptAfterTools(1)
    )
    assert result is not None
    assert result.get("status") == "interrupted"
    assert _forced_read_counts(resumed) == (2, 0, 2)
    assert len(resumed.pending_tool_calls) == 1

    again = _resumed(resumed, parallel=False)
    result, _ = await _drain(again, [reader], context=context)
    assert result is None
    assert _forced_read_counts(again) == (3, 0, 3)
    assert len(reader.calls) == 3


async def test_batches_without_reads_add_no_extra_iterations() -> None:
    _, other = _reader_and_other()
    pattern = _forced_read_pattern(parallel=True)
    pattern.pending_tool_calls = _read_calls(None, None, None, None)

    result, _ = await _drain(pattern, [other])

    assert result is None
    assert len(other.calls) == 4
    assert _forced_read_counts(pattern) == (0, 0, 0)


@pytest.mark.parametrize(
    ("cell", "start", "paths", "read_open"),
    [
        ("all_admitted", (0, 0, 0), [_STORED_PATH] * 3, True),
        ("all_unlisted", (0, 0, 0), [_UNLISTED_PATH] * 3, True),
        (
            "mixed",
            (0, 0, 0),
            [_STORED_PATH, _UNLISTED_PATH, _STORED_PATH, _UNLISTED_PATH],
            True,
        ),
        ("allowance_used_up", (3, 1, 4), [_STORED_PATH] * 3, True),
        ("with_a_non_read_call", (0, 0, 0), [_STORED_PATH, None, _STORED_PATH], True),
        ("ordinary_turn", (0, 0, 0), [_STORED_PATH] * 3, False),
    ],
)
async def test_extra_iterations_lock_step_with_read_counters(
    cell: str, start: tuple[int, int, int], paths: list, read_open: bool
) -> None:
    """The extra iterations grow exactly as much as the two read counters
    together, and never past their combined per-run caps."""
    pattern = _forced_read_pattern(parallel=True, read_open=read_open, counts=start)
    pattern.pending_tool_calls = _read_calls(*paths)

    await _drain(pattern, list(_reader_and_other()))

    used, rejected, extra = _forced_read_counts(pattern)
    assert extra - start[2] == (used - start[0]) + (rejected - start[1])
    assert extra >= start[2]
    assert extra <= 6
    if cell in {"allowance_used_up", "ordinary_turn"}:
        assert (used, rejected, extra) == start
