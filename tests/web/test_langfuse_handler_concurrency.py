"""Inc.1.5 — Langfuse action-span pairing under in-turn tool concurrency.

When several tool calls in the same ReAct turn share a step_id and tool name
(e.g. three concurrent ``web_search`` calls), the handler must pair each
START with its own END/ERROR by ``tool_call_id`` — not by "last observation
pushed for this name" (LIFO), which mis-attributes output/duration/status the
moment completion order differs from reverse-start order.

This is the prerequisite fix that must land before tool concurrency is enabled
(design §5.4). It introduces no concurrency itself; it drives the handler with
interleaved events directly.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.utils.mock_helpers import create_langfuse_mock
from xagent.core.agent.trace import (
    TraceAction,
    TraceCategory,
    TraceEventType,
    Tracer,
    TraceScope,
    trace_action_end,
    trace_action_start,
)
from xagent.core.tracing.langfuse import create_langfuse_trace_handler


def _make_observation(mocker, span_id: str) -> Any:
    observation = mocker.Mock()
    observation.trace_id = "trace-concurrency"
    observation.id = span_id
    return observation


def _enable_langfuse_env(monkeypatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test-secret")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://langfuse.example")


async def _trace_tool_error(
    tracer: Tracer,
    task_id: str,
    step_id: str,
    data: dict[str, Any],
) -> None:
    """Emit a per-tool ACTION/ERROR/TOOL event, as runtime.on_tool_error does."""
    await tracer.trace_event(
        TraceEventType(TraceScope.ACTION, TraceAction.ERROR, TraceCategory.TOOL),
        task_id=task_id,
        step_id=step_id,
        data=data,
    )


@pytest.mark.asyncio
async def test_same_name_concurrent_tools_pair_by_tool_call_id(
    mocker, monkeypatch, langfuse_client_reset
):
    _enable_langfuse_env(monkeypatch)
    _, mock_langfuse = create_langfuse_mock(mocker)
    root = _make_observation(mocker, "root")
    obs_a = _make_observation(mocker, "obs-a")
    obs_b = _make_observation(mocker, "obs-b")
    mock_langfuse.start_observation.return_value = root
    # First START (call A) gets obs_a, second START (call B) gets obs_b.
    root.start_observation.side_effect = [obs_a, obs_b]

    handler = create_langfuse_trace_handler(task_id="task-1")
    assert handler is not None
    tracer = Tracer()
    tracer.add_handler(handler)

    # Two same-name tool calls start; the first-started one completes first
    # (FIFO completion), which is exactly what LIFO pairing gets wrong.
    await trace_action_start(
        tracer,
        "task-1",
        "step-1",
        TraceCategory.TOOL,
        data={"tool_name": "web_search", "tool_call_id": "A", "tool_args": {"q": "a"}},
    )
    await trace_action_start(
        tracer,
        "task-1",
        "step-1",
        TraceCategory.TOOL,
        data={"tool_name": "web_search", "tool_call_id": "B", "tool_args": {"q": "b"}},
    )
    await trace_action_end(
        tracer,
        "task-1",
        "step-1",
        TraceCategory.TOOL,
        data={
            "tool_name": "web_search",
            "tool_call_id": "A",
            "result": "RES_A",
            "success": True,
        },
    )
    await trace_action_end(
        tracer,
        "task-1",
        "step-1",
        TraceCategory.TOOL,
        data={
            "tool_name": "web_search",
            "tool_call_id": "B",
            "result": "RES_B",
            "success": True,
        },
    )

    assert obs_a.update.call_args.kwargs["output"]["result"] == "RES_A"
    assert obs_b.update.call_args.kwargs["output"]["result"] == "RES_B"
    obs_a.end.assert_called_once()
    obs_b.end.assert_called_once()


@pytest.mark.asyncio
async def test_concurrent_tool_error_pairs_by_tool_call_id(
    mocker, monkeypatch, langfuse_client_reset
):
    _enable_langfuse_env(monkeypatch)
    _, mock_langfuse = create_langfuse_mock(mocker)
    root = _make_observation(mocker, "root")
    obs_a = _make_observation(mocker, "obs-a")
    obs_b = _make_observation(mocker, "obs-b")
    mock_langfuse.start_observation.return_value = root
    root.start_observation.side_effect = [obs_a, obs_b]

    handler = create_langfuse_trace_handler(task_id="task-2")
    assert handler is not None
    tracer = Tracer()
    tracer.add_handler(handler)

    # A errors while B is still running, then B ends normally.
    await trace_action_start(
        tracer,
        "task-2",
        "step-1",
        TraceCategory.TOOL,
        data={"tool_name": "web_search", "tool_call_id": "A", "tool_args": {"q": "a"}},
    )
    await trace_action_start(
        tracer,
        "task-2",
        "step-1",
        TraceCategory.TOOL,
        data={"tool_name": "web_search", "tool_call_id": "B", "tool_args": {"q": "b"}},
    )
    await _trace_tool_error(
        tracer,
        "task-2",
        "step-1",
        data={
            "tool_name": "web_search",
            "tool_call_id": "A",
            "error_message": "boom-A",
            "success": False,
        },
    )
    await trace_action_end(
        tracer,
        "task-2",
        "step-1",
        TraceCategory.TOOL,
        data={
            "tool_name": "web_search",
            "tool_call_id": "B",
            "result": "RES_B",
            "success": True,
        },
    )

    # A's observation is the one flagged ERROR with A's message.
    assert obs_a.update.call_args.kwargs["level"] == "ERROR"
    assert obs_a.update.call_args.kwargs["status_message"] == "boom-A"
    obs_a.end.assert_called_once()
    # B's observation ends normally with B's result.
    assert obs_b.update.call_args.kwargs.get("level") != "ERROR"
    assert obs_b.update.call_args.kwargs["output"]["result"] == "RES_B"
    obs_b.end.assert_called_once()


@pytest.mark.asyncio
async def test_reused_tool_call_id_pairs_by_invocation_id(
    mocker, monkeypatch, langfuse_client_reset
):
    _enable_langfuse_env(monkeypatch)
    _, mock_langfuse = create_langfuse_mock(mocker)
    root = _make_observation(mocker, "root")
    obs_a = _make_observation(mocker, "obs-a")
    obs_b = _make_observation(mocker, "obs-b")
    mock_langfuse.start_observation.return_value = root
    root.start_observation.side_effect = [obs_a, obs_b]

    handler = create_langfuse_trace_handler(task_id="task-reused-id")
    assert handler is not None
    tracer = Tracer()
    tracer.add_handler(handler)

    for invocation_id in ("invocation-a", "invocation-b"):
        await trace_action_start(
            tracer,
            "task-reused-id",
            "step-1",
            TraceCategory.TOOL,
            data={
                "tool_name": "web_search",
                "tool_call_id": "tool_call_0",
                "invocation_id": invocation_id,
            },
        )
    for invocation_id, result in (
        ("invocation-a", "RES_A"),
        ("invocation-b", "RES_B"),
    ):
        await trace_action_end(
            tracer,
            "task-reused-id",
            "step-1",
            TraceCategory.TOOL,
            data={
                "tool_name": "web_search",
                "tool_call_id": "tool_call_0",
                "invocation_id": invocation_id,
                "result": result,
                "success": True,
            },
        )

    assert obs_a.update.call_args.kwargs["output"]["result"] == "RES_A"
    assert obs_b.update.call_args.kwargs["output"]["result"] == "RES_B"


@pytest.mark.asyncio
async def test_single_tool_without_tool_call_id_still_pairs(
    mocker, monkeypatch, langfuse_client_reset
):
    # Backward compatibility: events without tool_call_id (legacy / external)
    # still degrade to name-based pairing for a single in-flight call.
    _enable_langfuse_env(monkeypatch)
    _, mock_langfuse = create_langfuse_mock(mocker)
    root = _make_observation(mocker, "root")
    obs = _make_observation(mocker, "obs")
    mock_langfuse.start_observation.return_value = root
    root.start_observation.side_effect = [obs]

    handler = create_langfuse_trace_handler(task_id="task-3")
    assert handler is not None
    tracer = Tracer()
    tracer.add_handler(handler)

    await trace_action_start(
        tracer,
        "task-3",
        "step-1",
        TraceCategory.TOOL,
        data={"tool_name": "calculator", "tool_args": {"expression": "1+1"}},
    )
    await trace_action_end(
        tracer,
        "task-3",
        "step-1",
        TraceCategory.TOOL,
        data={"tool_name": "calculator", "result": "2", "success": True},
    )

    assert obs.update.call_args.kwargs["output"]["result"] == "2"
    obs.end.assert_called_once()


@pytest.mark.asyncio
async def test_same_name_tools_in_different_steps_remain_independent(
    mocker, monkeypatch, langfuse_client_reset
):
    # DAG step-level concurrency (already supported): same tool name in
    # different steps must stay independent. Guards against regression.
    _enable_langfuse_env(monkeypatch)
    _, mock_langfuse = create_langfuse_mock(mocker)
    root = _make_observation(mocker, "root")
    obs_s1 = _make_observation(mocker, "obs-s1")
    obs_s2 = _make_observation(mocker, "obs-s2")
    mock_langfuse.start_observation.return_value = root
    root.start_observation.side_effect = [obs_s1, obs_s2]

    handler = create_langfuse_trace_handler(task_id="task-4")
    assert handler is not None
    tracer = Tracer()
    tracer.add_handler(handler)

    await trace_action_start(
        tracer,
        "task-4",
        "step-1",
        TraceCategory.TOOL,
        data={"tool_name": "web_search", "tool_args": {"q": "s1"}},
    )
    await trace_action_start(
        tracer,
        "task-4",
        "step-2",
        TraceCategory.TOOL,
        data={"tool_name": "web_search", "tool_args": {"q": "s2"}},
    )
    await trace_action_end(
        tracer,
        "task-4",
        "step-1",
        TraceCategory.TOOL,
        data={"tool_name": "web_search", "result": "RES_S1", "success": True},
    )
    await trace_action_end(
        tracer,
        "task-4",
        "step-2",
        TraceCategory.TOOL,
        data={"tool_name": "web_search", "result": "RES_S2", "success": True},
    )

    assert obs_s1.update.call_args.kwargs["output"]["result"] == "RES_S1"
    assert obs_s2.update.call_args.kwargs["output"]["result"] == "RES_S2"
