"""Inc.0 sanity test: the concurrency harness imports and behaves as designed.

This is the acceptance gate for the test scaffolding (design §9 Inc.0): it does
not exercise any production concurrency path, only that the fakes are wired
correctly and reusable by later increments.
"""

from __future__ import annotations

import asyncio

from tests.core.agent.concurrency_harness import (
    ConcurrencyTracker,
    FakeRuntime,
    FakeTool,
    RecordingContext,
    make_react,
    make_tool_call,
)


def test_make_tool_call_assigns_unique_ids() -> None:
    a = make_tool_call("calculator", {"expression": "1+1"})
    b = make_tool_call("calculator", {"expression": "2+2"})
    assert a["name"] == "calculator"
    assert a["args"] == {"expression": "1+1"}
    assert a["id"] and b["id"] and a["id"] != b["id"]

    explicit = make_tool_call("web_search", id="fixed")
    assert explicit["id"] == "fixed"


def test_fake_tool_metadata_implication() -> None:
    read_only = FakeTool("web_search", read_only=True)
    assert read_only.metadata.read_only is True
    assert read_only.metadata.concurrency_safe is True

    safe = FakeTool("calculator", concurrency_safe=True)
    assert safe.metadata.read_only is False
    assert safe.metadata.concurrency_safe is True

    unsafe = FakeTool("write_file")
    assert unsafe.metadata.concurrency_safe is False


async def test_fake_tool_execute_returns_default_envelope() -> None:
    tool = FakeTool("calculator", concurrency_safe=True)
    result = await tool.execute(expression="1+1")
    assert result == {
        "success": True,
        "tool": "calculator",
        "args": {"expression": "1+1"},
    }
    assert tool.calls == [{"expression": "1+1"}]


async def test_fake_tool_raises_is_propagated() -> None:
    boom = RuntimeError("boom")
    tool = FakeTool("broken", raises=boom)
    try:
        await tool.execute()
    except RuntimeError as exc:
        assert exc is boom
    else:  # pragma: no cover - defensive
        raise AssertionError("FakeTool(raises=...) should have raised")


async def test_concurrency_tracker_observes_real_overlap() -> None:
    tracker = ConcurrencyTracker()
    gate = asyncio.Event()
    tools = [
        FakeTool(f"t{i}", concurrency_safe=True, gate=gate, tracker=tracker)
        for i in range(3)
    ]

    async def run(tool: FakeTool) -> None:
        await tool.execute()

    tasks = [asyncio.create_task(run(tool)) for tool in tools]
    # Let all three enter (and block on the gate) before any can leave.
    while tracker.active < 3:
        await asyncio.sleep(0)
    assert tracker.peak == 3
    gate.set()
    await asyncio.gather(*tasks)
    assert tracker.active == 0
    assert len(tracker.leave_order) == 3


def test_recording_context_preserves_order_and_ids() -> None:
    context = RecordingContext()
    context.add_tool_result(tool_name="a", result={"ok": 1}, tool_call_id="id-a")
    context.add_tool_result(tool_name="b", result={"ok": 2}, tool_call_id="id-b")
    context.add_assistant_message("done")

    assert [r["tool_call_id"] for r in context.tool_results] == ["id-a", "id-b"]
    assert [r["tool_name"] for r in context.tool_results] == ["a", "b"]
    assert context.messages[-1].role == "assistant"
    assert context.messages[-1].content == "done"


async def test_fake_runtime_records_event_sequence() -> None:
    runtime = FakeRuntime()
    assert await runtime.should_interrupt() is False

    tool_call = make_tool_call("calculator", id="cc-1")
    await runtime.on_tool_start(tool_call=tool_call)
    await runtime.checkpoint("before_tool", metadata={"tool_call": tool_call})
    await runtime.on_tool_end(tool_call=tool_call, result={"success": True})

    kinds = [kind for kind, _ in runtime.events]
    assert kinds == ["on_tool_start", "checkpoint", "on_tool_end"]
    assert runtime.events_of("on_tool_start")[0]["tool_call_id"] == "cc-1"


async def test_fake_runtime_returns_question_event_identity() -> None:
    runtime = FakeRuntime()

    outbound = await runtime.send_message(
        message="which?",
        message_type="question",
        expect_response=True,
        visible=True,
    )

    assert outbound["event_id"] == "fake-agent-message-1"
    assert runtime.events_of("send_message") == [outbound]


async def test_fake_runtime_interrupt_flag() -> None:
    runtime = FakeRuntime(interrupt=True, interrupt_reason="stop")
    assert await runtime.should_interrupt() is True
    assert runtime.interrupt_reason == "stop"


def test_make_react_sets_concurrency_knobs() -> None:
    serial = make_react()
    assert serial.tool_parallel_enabled is False
    assert serial.tool_max_concurrency == 3

    parallel = make_react(parallel=True, max_concurrency=2, max_iterations=5)
    assert parallel.tool_parallel_enabled is True
    assert parallel.tool_max_concurrency == 2
    assert parallel.max_iterations == 5
