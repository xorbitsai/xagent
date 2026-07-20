"""Unit tests for ``map_trace_events_to_public_steps``.

The function is intentionally pure (no DB / FastAPI / async) so this
suite can drive synthetic events directly. We use a tiny ``_Event``
namedtuple-ish dict to mimic the attributes the ORM row exposes
(``event_type``, ``data``, ``step_id``, ``timestamp``, ``event_id``,
``task_id``). Anywhere the helper falls back to ``isinstance(event,
dict)`` works too -- but the attribute form is closer to production
behavior.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from xagent.web.api.v1._step_mapping import map_trace_events_to_public_steps


def _ev(
    event_type: str,
    *,
    data: Optional[Dict[str, Any]] = None,
    step_id: Optional[str] = None,
    task_id: Optional[str] = "t1",
    event_id: Optional[str] = None,
    timestamp: Optional[datetime] = None,
) -> SimpleNamespace:
    """Build a synthetic event row.

    Defaults to a deterministic-ish timestamp so test ordering is
    stable. ``event_id`` falls back to a synthetic value if not given.
    """
    return SimpleNamespace(
        event_type=event_type,
        data=data or {},
        step_id=step_id,
        task_id=task_id,
        event_id=event_id or f"evt-{event_type}-{step_id or 'na'}",
        timestamp=timestamp or datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )


# ===== happy paths: each public type appears at least once =====


def test_react_action_pair_becomes_thinking_action():
    events: List[SimpleNamespace] = [
        _ev("react_action_start", step_id="s1"),
        _ev(
            "react_action_end",
            step_id="s1",
            timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
        ),
    ]
    steps = map_trace_events_to_public_steps(events)
    assert len(steps) == 1
    s = steps[0]
    assert s["type"] == "thinking"
    assert s["data"]["phase"] == "action"
    assert s["status"] == "completed"
    assert s["completed_at"] > s["started_at"]


def test_dag_step_pair_becomes_thinking_step():
    events = [
        _ev("dag_step_start", step_id="s2"),
        _ev(
            "dag_step_end",
            step_id="s2",
            timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
        ),
    ]
    steps = map_trace_events_to_public_steps(events)
    assert len(steps) == 1
    assert steps[0]["type"] == "thinking"
    assert steps[0]["data"]["phase"] == "step"


def test_dag_plan_pair_becomes_thinking_planning():
    events = [
        _ev("dag_plan_start", task_id="t1"),
        _ev(
            "dag_plan_end",
            task_id="t1",
            timestamp=datetime(2026, 1, 1, 12, 0, 2, tzinfo=timezone.utc),
        ),
    ]
    steps = map_trace_events_to_public_steps(events)
    assert len(steps) == 1
    assert steps[0]["type"] == "thinking"
    assert steps[0]["data"]["phase"] == "planning"


def test_two_dag_plan_pairs_produce_two_separate_steps():
    """Replan in one task: two dag_plan_start/end pairs must produce
    two distinct planning steps (regression for the previous
    pair-key collision where both used ``plan:{task_id}`` and the
    second start silently overwrote the first's pending entry).
    """
    events = [
        _ev(
            "dag_plan_start",
            task_id="t1",
            timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        ),
        _ev(
            "dag_plan_end",
            task_id="t1",
            timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
        ),
        _ev(
            "dag_plan_start",
            task_id="t1",
            timestamp=datetime(2026, 1, 1, 12, 0, 2, tzinfo=timezone.utc),
        ),
        _ev(
            "dag_plan_end",
            task_id="t1",
            timestamp=datetime(2026, 1, 1, 12, 0, 3, tzinfo=timezone.utc),
        ),
    ]
    steps = map_trace_events_to_public_steps(events)
    assert len(steps) == 2
    assert all(
        s["type"] == "thinking" and s["data"]["phase"] == "planning" for s in steps
    )
    # IDs must differ so SDK clients can dedupe across re-polls.
    assert steps[0]["id"] != steps[1]["id"]
    # Sorted by started_at -> first pair before second.
    assert steps[0]["started_at"] < steps[1]["started_at"]


def test_tool_execution_pair_becomes_tool_call_with_args_and_result():
    events = [
        _ev(
            "tool_execution_start",
            step_id="s3",
            data={
                "tool_name": "execute_python",
                "tool_args": {"code": "print(1)"},
                "tool_execution_id": "tx-1",
            },
        ),
        _ev(
            "tool_execution_end",
            step_id="s3",
            data={
                "tool_name": "execute_python",
                "tool_args": {"code": "print(1)"},
                "tool_execution_id": "tx-1",
                "result": {"output": "1\n"},
                "success": True,
            },
            timestamp=datetime(2026, 1, 1, 12, 0, 3, tzinfo=timezone.utc),
        ),
    ]
    steps = map_trace_events_to_public_steps(events)
    assert len(steps) == 1
    s = steps[0]
    assert s["type"] == "tool_call"
    assert s["status"] == "completed"
    assert s["data"]["name"] == "execute_python"
    assert s["data"]["args"] == {"code": "print(1)"}
    assert s["data"]["result"] == {"output": "1\n"}
    assert "error" not in s["data"]
    # Contract: tool_call uses ``result``, not ``output``. Only
    # agent_delegation steps rename it to ``output``.
    assert "output" not in s["data"]


def test_tool_execution_agent_id_tool_becomes_agent_delegation():
    """A canonical ``agent_<id>`` tool routes to agent_delegation."""
    events = [
        _ev(
            "tool_execution_start",
            step_id="s4",
            data={
                "tool_name": "agent_42",
                "tool_args": {"text": "hello"},
                "tool_execution_id": "tx-2",
            },
        ),
        _ev(
            "tool_execution_end",
            step_id="s4",
            data={
                "tool_name": "agent_42",
                "tool_args": {"text": "hello"},
                "tool_execution_id": "tx-2",
                "result": {"translated": "你好"},
                "success": True,
            },
            timestamp=datetime(2026, 1, 1, 12, 0, 4, tzinfo=timezone.utc),
        ),
    ]
    steps = map_trace_events_to_public_steps(events)
    assert len(steps) == 1
    s = steps[0]
    assert s["type"] == "agent_delegation"
    assert s["data"]["sub_agent_name"] == "agent_42"
    assert s["data"]["input"] == {"text": "hello"}
    # Contract: agent_delegation uses ``output`` on the public surface
    # (mirrors ``input`` on the start side). The internal field is
    # still ``data['result']`` on the end event, but the mapping
    # function renames it to ``output`` for this public type.
    # ``result`` must NOT appear on agent_delegation steps.
    assert s["data"]["output"] == {"translated": "你好"}
    assert "result" not in s["data"]


def test_tool_execution_semantic_workforce_tool_becomes_agent_delegation():
    """Semantic Workforce tool names remain agent delegation steps."""
    events = [
        _ev(
            "tool_execution_start",
            step_id="s4",
            data={
                "tool_name": "worker_preproduction_qa__a18",
                "tool_args": {"task": "review"},
                "tool_execution_id": "tx-workforce",
            },
        ),
        _ev(
            "tool_execution_end",
            step_id="s4",
            data={
                "tool_name": "worker_preproduction_qa__a18",
                "tool_args": {"task": "review"},
                "tool_execution_id": "tx-workforce",
                "result": {"status": "approved"},
                "success": True,
            },
            timestamp=datetime(2026, 1, 1, 12, 0, 4, tzinfo=timezone.utc),
        ),
    ]

    steps = map_trace_events_to_public_steps(events)

    assert len(steps) == 1
    assert steps[0]["type"] == "agent_delegation"
    assert steps[0]["data"]["sub_agent_name"] == ("worker_preproduction_qa__a18")


def test_tool_execution_legacy_call_agent_trace_still_maps_to_delegation():
    """Historical traces with ``call_agent_<name>`` remain readable."""
    events = [
        _ev(
            "tool_execution_start",
            step_id="s4",
            data={
                "tool_name": "call_agent_translator",
                "tool_args": {"text": "hello"},
                "tool_execution_id": "tx-legacy",
            },
        ),
        _ev(
            "tool_execution_end",
            step_id="s4",
            data={
                "tool_name": "call_agent_translator",
                "tool_args": {"text": "hello"},
                "tool_execution_id": "tx-legacy",
                "result": {"translated": "你好"},
                "success": True,
            },
            timestamp=datetime(2026, 1, 1, 12, 0, 4, tzinfo=timezone.utc),
        ),
    ]

    steps = map_trace_events_to_public_steps(events)

    assert len(steps) == 1
    assert steps[0]["type"] == "agent_delegation"
    assert steps[0]["data"]["sub_agent_name"] == "translator"


def test_tool_execution_failure_marks_failed_with_error():
    events = [
        _ev(
            "tool_execution_start",
            step_id="s5",
            data={
                "tool_name": "broken_tool",
                "tool_args": {},
                "tool_execution_id": "tx-3",
            },
        ),
        _ev(
            "tool_execution_end",
            step_id="s5",
            data={
                "tool_name": "broken_tool",
                "tool_args": {},
                "tool_execution_id": "tx-3",
                "success": False,
                "error": "exploded",
            },
            timestamp=datetime(2026, 1, 1, 12, 0, 5, tzinfo=timezone.utc),
        ),
    ]
    steps = map_trace_events_to_public_steps(events)
    assert len(steps) == 1
    assert steps[0]["type"] == "tool_call"
    assert steps[0]["status"] == "failed"
    assert steps[0]["data"]["error"] == "exploded"
    assert "result" not in steps[0]["data"]


def test_tool_execution_failed_event_marks_step_failed():
    """Regression for v2 runtime which emits a dedicated
    ``tool_execution_failed`` event (TraceCategory.TOOL + ERROR action)
    instead of ``tool_execution_end`` with ``success=False``.

    Without explicit handling, the pending start was never finalized
    and ``GET /steps`` would report the step as still ``running``
    forever after a tool failure.
    """
    events = [
        _ev(
            "tool_execution_start",
            step_id="s_fail",
            data={
                "tool_name": "execute_python",
                "tool_params": {"code": "1 / 0"},
                "tool_call_id": "call-fail",
            },
        ),
        _ev(
            "tool_execution_failed",
            step_id="s_fail",
            data={
                "tool_name": "execute_python",
                "tool_call_id": "call-fail",
                "error": "division by zero",
            },
            timestamp=datetime(2026, 1, 1, 12, 0, 5, tzinfo=timezone.utc),
        ),
    ]
    steps = map_trace_events_to_public_steps(events)
    assert len(steps) == 1
    s = steps[0]
    assert s["type"] == "tool_call"
    assert s["status"] == "failed"
    assert s["data"]["error"] == "division by zero"
    assert s["completed_at"] > s["started_at"]


def test_v2_tool_params_alias_reads_args():
    """v2 runtime writes ``tool_params`` where v1 writes ``tool_args``.

    Mapper should fall back to ``tool_params`` so the public step's
    ``data.args`` field is populated regardless of which runtime ran.
    """
    events = [
        _ev(
            "tool_execution_start",
            step_id="s_v2",
            data={
                "tool_name": "web_search",
                "tool_params": {"query": "xagent"},
                "tool_call_id": "call-v2",
            },
        ),
        _ev(
            "tool_execution_end",
            step_id="s_v2",
            data={
                "tool_name": "web_search",
                "tool_call_id": "call-v2",
                "result": "search results",
                "success": True,
            },
            timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
        ),
    ]
    steps = map_trace_events_to_public_steps(events)
    assert len(steps) == 1
    s = steps[0]
    assert s["status"] == "completed"
    assert s["data"]["name"] == "web_search"
    assert s["data"]["args"] == {"query": "xagent"}
    assert s["data"]["result"] == "search results"


def test_v2_tool_start_preserves_assistant_content():
    events = [
        _ev(
            "tool_execution_start",
            step_id="s_note",
            data={
                "tool_name": "web_search",
                "tool_params": {"query": "ai news"},
                "tool_call_id": "call-note",
                "assistant_content": "I need current search results first.",
            },
        ),
    ]

    steps = map_trace_events_to_public_steps(events)

    assert len(steps) == 1
    assert steps[0]["type"] == "tool_call"
    assert steps[0]["status"] == "running"
    assert steps[0]["data"]["assistant_content"] == (
        "I need current search results first."
    )


def test_v2_tool_call_id_does_not_collide_within_same_step():
    """Two v2 tool calls under the same ``step_id`` must each get their
    own pending entry — pair key has to be the per-invocation
    ``tool_call_id``, not the shared ``step_id``.

    Without this, the second start overwrites the first pending and
    the public timeline either loses the first call or pairs an end
    event with the wrong pending entry.
    """
    events = [
        _ev(
            "tool_execution_start",
            step_id="shared_step",
            event_id="evt-1",
            data={
                "tool_name": "tool_a",
                "tool_params": {"q": "a"},
                "tool_call_id": "call-a",
            },
        ),
        _ev(
            "tool_execution_start",
            step_id="shared_step",
            event_id="evt-2",
            data={
                "tool_name": "tool_b",
                "tool_params": {"q": "b"},
                "tool_call_id": "call-b",
            },
            timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
        ),
        _ev(
            "tool_execution_end",
            step_id="shared_step",
            event_id="evt-3",
            data={
                "tool_name": "tool_a",
                "tool_call_id": "call-a",
                "result": "result_a",
                "success": True,
            },
            timestamp=datetime(2026, 1, 1, 12, 0, 2, tzinfo=timezone.utc),
        ),
        _ev(
            "tool_execution_end",
            step_id="shared_step",
            event_id="evt-4",
            data={
                "tool_name": "tool_b",
                "tool_call_id": "call-b",
                "result": "result_b",
                "success": True,
            },
            timestamp=datetime(2026, 1, 1, 12, 0, 3, tzinfo=timezone.utc),
        ),
    ]
    steps = map_trace_events_to_public_steps(events)
    assert len(steps) == 2, (
        f"expected 2 distinct tool_call steps, got {len(steps)}; the second "
        f"start likely overwrote the first pending entry"
    )
    by_name = {s["data"]["name"]: s for s in steps}
    assert "tool_a" in by_name and "tool_b" in by_name
    assert by_name["tool_a"]["data"]["args"] == {"q": "a"}
    assert by_name["tool_a"]["data"]["result"] == "result_a"
    assert by_name["tool_b"]["data"]["args"] == {"q": "b"}
    assert by_name["tool_b"]["data"]["result"] == "result_b"


def test_user_and_ai_messages_emit_two_message_steps_in_order():
    events = [
        _ev(
            "user_message",
            data={"message": "hello"},
            timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        ),
        _ev(
            "ai_message",
            data={"content": "hi back"},
            timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
        ),
    ]
    steps = map_trace_events_to_public_steps(events)
    assert [s["type"] for s in steps] == ["message", "message"]
    assert steps[0]["data"] == {"role": "user", "content": "hello"}
    assert steps[1]["data"] == {"role": "assistant", "content": "hi back"}
    # Messages have started_at == completed_at and status='completed'
    assert steps[0]["started_at"] == steps[0]["completed_at"]
    assert steps[0]["status"] == "completed"


# ===== filtering: non-exposed event types are dropped =====


def test_unexposed_event_types_are_dropped():
    """LLM, memory, dag_execute, react_task, etc. don't surface on the SDK."""
    events = [
        _ev("llm_call_start", step_id="s1"),
        _ev("llm_call_end", step_id="s1"),
        _ev("dag_execute_start"),
        _ev("dag_execute_end"),
        _ev("react_task_start"),
        _ev("react_task_end"),
        _ev("react_step_start", step_id="s1"),
        _ev("react_step_end", step_id="s1"),
        _ev("visualization_update"),
        _ev("task_completion"),
        _ev("trace_error"),
    ]
    steps = map_trace_events_to_public_steps(events)
    assert steps == []


# ===== partial pairs =====


def test_start_without_end_is_running():
    """SDK polled mid-step: start seen, no end yet -> status='running'."""
    events = [_ev("react_action_start", step_id="s1")]
    steps = map_trace_events_to_public_steps(events)
    assert len(steps) == 1
    assert steps[0]["type"] == "thinking"
    assert steps[0]["status"] == "running"
    assert steps[0]["completed_at"] is None


def test_orphan_end_is_dropped():
    """End event with no matching start -> nothing emitted (malformed data)."""
    events = [_ev("react_action_end", step_id="s1")]
    steps = map_trace_events_to_public_steps(events)
    assert steps == []


# ===== ordering / multiple steps =====


def test_steps_are_sorted_by_started_at():
    """Out-of-order writes still produce monotonic output."""
    events = [
        # action that starts LATER but is logged first in this synthetic
        # list (e.g. due to async flush ordering in the runtime)
        _ev(
            "react_action_start",
            step_id="late",
            timestamp=datetime(2026, 1, 1, 12, 0, 5, tzinfo=timezone.utc),
        ),
        _ev(
            "react_action_end",
            step_id="late",
            timestamp=datetime(2026, 1, 1, 12, 0, 6, tzinfo=timezone.utc),
        ),
        _ev(
            "react_action_start",
            step_id="early",
            timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
        ),
        _ev(
            "react_action_end",
            step_id="early",
            timestamp=datetime(2026, 1, 1, 12, 0, 2, tzinfo=timezone.utc),
        ),
    ]
    steps = map_trace_events_to_public_steps(events)
    assert len(steps) == 2
    assert steps[0]["id"].endswith(":early")
    assert steps[1]["id"].endswith(":late")
    assert steps[0]["started_at"] < steps[1]["started_at"]


def test_pending_runs_appear_after_completed_when_started_later():
    """An in-flight step at the tail still sorts last."""
    events = [
        _ev(
            "react_action_start",
            step_id="done",
            timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        ),
        _ev(
            "react_action_end",
            step_id="done",
            timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
        ),
        _ev(
            "react_action_start",
            step_id="running",
            timestamp=datetime(2026, 1, 1, 12, 0, 2, tzinfo=timezone.utc),
        ),
    ]
    steps = map_trace_events_to_public_steps(events)
    assert [s["status"] for s in steps] == ["completed", "running"]


# ===== empty input =====


def test_empty_event_list_returns_empty_steps():
    assert map_trace_events_to_public_steps([]) == []


# ===== skill_select_*: surface as tool_call =====


def test_skill_select_pair_becomes_tool_call():
    events = [
        _ev(
            "skill_select_start",
            data={"skill_name": "presentation"},
            timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        ),
        _ev(
            "skill_select_end",
            data={"skill_name": "presentation", "result": "ok"},
            timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
        ),
    ]
    steps = map_trace_events_to_public_steps(events)
    assert len(steps) == 1
    s = steps[0]
    assert s["type"] == "tool_call"
    assert s["data"]["name"] == "presentation"
    assert s["status"] == "completed"


# ===== content-key normalization =====


def test_message_content_falls_back_when_message_key_absent():
    """ai_message uses ``data.content``, user_message uses ``data.message``.
    Helper normalizes both into the public ``content`` field.
    """
    events = [
        _ev("user_message", data={"message": "msg-key"}),
        _ev(
            "ai_message",
            data={"content": "content-key"},
            timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
        ),
    ]
    steps = map_trace_events_to_public_steps(events)
    assert steps[0]["data"]["content"] == "msg-key"
    assert steps[1]["data"]["content"] == "content-key"


# ===== timestamp coercion =====


def test_float_timestamp_is_coerced_to_datetime():
    """If the ORM returns a float (epoch) instead of datetime, handle it."""
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp()
    events = [
        _ev(
            "react_action_start",
            step_id="s1",
            timestamp=base,  # type: ignore[arg-type]
        ),
        _ev(
            "react_action_end",
            step_id="s1",
            timestamp=base + 1,  # type: ignore[arg-type]
        ),
    ]
    steps = map_trace_events_to_public_steps(events)
    assert isinstance(steps[0]["started_at"], datetime)
    assert isinstance(steps[0]["completed_at"], datetime)
    assert steps[0]["completed_at"] - steps[0]["started_at"] == timedelta(seconds=1)
