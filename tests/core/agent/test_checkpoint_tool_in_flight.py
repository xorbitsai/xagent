"""``checkpoint_tool_call_in_flight``: may a tool call have started after a
checkpoint was written, so that resuming from it would run the call again?"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests.core.agent.concurrency_harness import (
    FakeTool,
    RecordingContext,
    make_react,
    make_tool_call,
)
from xagent.core.agent import PatternRuntime
from xagent.core.agent.checkpoint import checkpoint_tool_call_in_flight

_PENDING = {"pending_tool_calls": [{"id": "call-1", "name": "send", "args": {}}]}
_IDLE = {"pending_tool_calls": []}


def _pattern(**kwargs: Any):
    pattern = make_react(
        repeated_tool_decision_after_consecutive_tool_calls=None,
        repeated_tool_decision_after_consecutive_work_tool_calls=None,
        **kwargs,
    )
    pattern.task_text = "t"
    return pattern


async def _wait_for_call(tool: FakeTool) -> None:
    while not tool.calls:
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_serial_tool_is_in_flight_from_before_tool_until_after_tool() -> None:
    first_gate, second_gate = asyncio.Event(), asyncio.Event()
    tools = [FakeTool("u1", gate=first_gate), FakeTool("u2", gate=second_gate)]
    pattern = _pattern()
    pattern.pending_tool_calls = [make_tool_call("u1"), make_tool_call("u2")]
    runtime = PatternRuntime(execution_id="serial-in-flight")

    run = asyncio.create_task(
        pattern._execute_pending_tool_calls(
            context=RecordingContext(), tools=tools, llm=None, runtime=runtime
        )
    )
    await _wait_for_call(tools[0])
    assert runtime.last_checkpoint["label"] == "before_tool"
    assert checkpoint_tool_call_in_flight(runtime.last_checkpoint) is True

    first_gate.set()
    await _wait_for_call(tools[1])
    # u1's after_tool was followed by u2's own before_tool.
    assert runtime.last_checkpoint["label"] == "before_tool"
    assert checkpoint_tool_call_in_flight(runtime.last_checkpoint) is True

    second_gate.set()
    await run
    assert runtime.last_checkpoint["label"] == "after_tool"
    assert checkpoint_tool_call_in_flight(runtime.last_checkpoint) is False


@pytest.mark.asyncio
async def test_concurrent_batch_is_in_flight_until_the_whole_segment_ends() -> None:
    gate = asyncio.Event()
    tools = [
        FakeTool("s1", read_only=True),
        FakeTool("s2", read_only=True, gate=gate),
    ]
    pattern = _pattern(parallel=True, max_concurrency=2)
    pattern.pending_tool_calls = [make_tool_call("s1"), make_tool_call("s2")]
    runtime = PatternRuntime(execution_id="batch-in-flight")

    run = asyncio.create_task(
        pattern._execute_pending_tool_calls(
            context=RecordingContext(), tools=tools, llm=None, runtime=runtime
        )
    )
    await _wait_for_call(tools[1])
    # s1 may already be done; the batch stays pending until s2 returns.
    assert runtime.last_checkpoint["label"] == "before_tool_batch"
    assert checkpoint_tool_call_in_flight(runtime.last_checkpoint) is True

    gate.set()
    await run
    assert runtime.last_checkpoint["label"] == "after_tool_batch"
    assert checkpoint_tool_call_in_flight(runtime.last_checkpoint) is False


@pytest.mark.asyncio
async def test_calls_planned_after_llm_have_not_started() -> None:
    pattern = _pattern()
    pattern.pending_tool_calls = [make_tool_call("u1")]
    runtime = PatternRuntime(execution_id="planned")

    checkpoint = await runtime.checkpoint(
        "after_llm", context=RecordingContext(), pattern=pattern
    )

    assert checkpoint["pattern_state"]["pending_tool_calls"]
    assert checkpoint_tool_call_in_flight(checkpoint) is False


@pytest.mark.parametrize(
    "label, expected",
    [
        ("auto_child_before_tool", True),
        ("auto_child_before_tool_batch", True),
        ("auto_child_after_llm", False),
        ("auto_child_after_tool", False),
    ],
)
def test_auto_child_uses_its_own_label(label: str, expected: bool) -> None:
    checkpoint = {
        "label": label,
        "metadata": {"child_label": label.removeprefix("auto_child_")},
        "pattern_state": {"react_state": _PENDING, "dag_state": None},
    }
    assert checkpoint_tool_call_in_flight(checkpoint) is expected


def _dag_checkpoint(
    label: str, writer: str, step_states: dict[str, Any]
) -> dict[str, Any]:
    return {
        "label": label,
        "metadata": {"active_step_id": writer, "child_label": label[len("dag_") :]},
        "pattern_state": {
            "plan": {"steps": [{"id": step} for step in step_states]},
            "active_step_pattern_states": step_states,
        },
    }


@pytest.mark.parametrize(
    "label, expected",
    [
        ("dag_before_tool", True),
        ("dag_before_tool_batch", True),
        ("dag_after_llm", False),
        ("dag_after_tool", False),
    ],
)
def test_dag_writer_step_uses_the_label(label: str, expected: bool) -> None:
    checkpoint = _dag_checkpoint(label, "a", {"a": _PENDING, "b": _IDLE})
    assert checkpoint_tool_call_in_flight(checkpoint) is expected


def test_dag_sibling_with_pending_calls_counts_as_in_flight() -> None:
    # Steps run concurrently: "b" may be inside a tool call even though the
    # latest checkpoint was written by "a" after its LLM call.
    checkpoint = _dag_checkpoint("dag_after_llm", "a", {"a": _IDLE, "b": _PENDING})
    assert checkpoint_tool_call_in_flight(checkpoint) is True


def test_dag_legacy_singular_step_state_is_read() -> None:
    checkpoint = {
        "label": "dag_before_tool",
        "metadata": {"active_step_id": "a"},
        "pattern_state": {
            "active_step_id": "a",
            "active_step_pattern_state": _PENDING,
        },
    }
    assert checkpoint_tool_call_in_flight(checkpoint) is True


def test_auto_dag_step_label_carries_both_prefixes() -> None:
    checkpoint = {
        "label": "auto_child_dag_before_tool",
        "metadata": {"active_step_id": "a", "child_label": "before_tool"},
        "pattern_state": {
            "react_state": None,
            "dag_state": {"active_step_pattern_states": {"a": _PENDING}},
        },
    }
    assert checkpoint_tool_call_in_flight(checkpoint) is True


@pytest.mark.parametrize(
    "checkpoint",
    [
        None,
        {},
        {"label": "before_tool", "pattern_state": None},
        {"label": "before_tool", "pattern_state": _IDLE},
        {"label": None, "pattern_state": _PENDING},
        {"label": "before_tool", "pattern_state": {"pending_tool_calls": "bad"}},
        {
            "label": "dag_before_tool",
            "metadata": "bad",
            "pattern_state": {"active_step_pattern_states": ["bad"]},
        },
    ],
)
def test_missing_or_malformed_fields_mean_no_call_in_flight(checkpoint: Any) -> None:
    assert checkpoint_tool_call_in_flight(checkpoint) is False
