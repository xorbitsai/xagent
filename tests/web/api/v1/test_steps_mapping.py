"""Unit tests for ``map_trace_events_to_public_steps``.

The function is intentionally pure (no DB / FastAPI / async) so this
suite can drive synthetic events directly. We use a tiny ``_Event``
namedtuple-ish dict to mimic the attributes the ORM row exposes
(``event_type``, ``data``, ``step_id``, ``timestamp``, ``event_id``,
``task_id``). Anywhere the helper falls back to ``isinstance(event,
dict)`` works too -- but the attribute form is closer to production
behavior.
"""

import copy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from xagent.web.api.v1._step_mapping import (
    PublicStepProjector,
    map_trace_events_to_public_steps,
)


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


def _dag_exec_ev(
    phase: str,
    *,
    task_id: Optional[str] = "t1",
    timestamp: Optional[datetime] = None,
    **extra_data: Any,
) -> SimpleNamespace:
    """Build a synthetic ``dag_execution`` event.

    ``phase`` is the only field the projector reads out of ``data``;
    ``extra_data`` lets callers attach the other fields a real
    emission carries (``completed_step_count``, ``plan_step_count``,
    ``steps``, ...) so a test can prove none of them leak into the
    public step.
    """
    data: Dict[str, Any] = {"phase": phase, **extra_data}
    return _ev("dag_execution", task_id=task_id, data=data, timestamp=timestamp)


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


def test_dag_step_end_with_failed_status_projects_failed():
    events = [
        _ev("dag_step_start", step_id="s2"),
        _ev(
            "dag_step_end",
            step_id="s2",
            data={"status": "failed", "error": "boom"},
            timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
        ),
    ]
    steps = map_trace_events_to_public_steps(events)
    assert len(steps) == 1
    assert steps[0]["status"] == "failed"


def test_dag_step_end_with_completed_status_projects_completed():
    events = [
        _ev("dag_step_start", step_id="s2"),
        _ev(
            "dag_step_end",
            step_id="s2",
            data={"status": "completed"},
            timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
        ),
    ]
    steps = map_trace_events_to_public_steps(events)
    assert len(steps) == 1
    assert steps[0]["status"] == "completed"


def test_dag_step_end_missing_status_key_defaults_to_completed():
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
    assert steps[0]["status"] == "completed"


@pytest.mark.parametrize("raw_status", ["interrupted", "cancelled", None])
def test_dag_step_end_unrecognized_status_falls_back_to_completed(raw_status):
    """Any ``data['status']`` other than the literal ``"failed"`` must
    fold to ``"completed"`` -- not pass through verbatim. This pins the
    fallback against a refactor that swaps the dedicated helper for a
    direct ``_data_get(event, "status", "completed")`` passthrough:
    the three existing status tests above (``"failed"``, ``"completed"``,
    missing key) all coincide with a passthrough's output and would
    stay green even if an unrecognized value like ``"interrupted"``
    leaked straight into ``PublicStepStatus``, which does not accept it.
    """
    data = {} if raw_status is None else {"status": raw_status}
    events = [
        _ev("dag_step_start", step_id="s2"),
        _ev(
            "dag_step_end",
            step_id="s2",
            data=data,
            timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
        ),
    ]
    steps = map_trace_events_to_public_steps(events)
    assert len(steps) == 1
    assert steps[0]["status"] == "completed"


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


# ===== PublicStepProjector <-> batch driver equivalence pin =====
#
# ``map_trace_events_to_public_steps`` is a thin batch driver over
# ``PublicStepProjector`` (from_history + materialized_steps + the
# started_at sort): it holds no folding logic of its own. These cases
# cover every pairing family the module docstring documents
# (thinking/planning incl. replan, tool_call, agent_delegation,
# skill_select, message, orphan start/end, unexposed-type filtering,
# out-of-order timestamps) so the equivalence assertion below pins
# that invariant -- the batch driver can never grow a second copy of
# the folding logic without this test catching the divergence.

_PROJECTOR_EQUIVALENCE_CASES: Dict[str, List[SimpleNamespace]] = {
    "react_action_pair": [
        _ev("react_action_start", step_id="s1"),
        _ev(
            "react_action_end",
            step_id="s1",
            timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
        ),
    ],
    "dag_step_pair": [
        _ev("dag_step_start", step_id="s2"),
        _ev(
            "dag_step_end",
            step_id="s2",
            timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
        ),
    ],
    "dag_step_failed": [
        _ev("dag_step_start", step_id="s2f"),
        _ev(
            "dag_step_end",
            step_id="s2f",
            data={"status": "failed", "error": "boom"},
            timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
        ),
    ],
    "dag_plan_pair": [
        _ev("dag_plan_start", task_id="t1"),
        _ev(
            "dag_plan_end",
            task_id="t1",
            timestamp=datetime(2026, 1, 1, 12, 0, 2, tzinfo=timezone.utc),
        ),
    ],
    "replan_two_plan_pairs": [
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
    ],
    "dag_plan_orphan_end_no_open_plan": [
        _ev("dag_plan_end", task_id="t1"),
    ],
    "tool_call_success": [
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
    ],
    "tool_call_failure": [
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
    ],
    "tool_execution_failed_event": [
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
    ],
    "agent_delegation": [
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
    ],
    "two_tool_calls_same_step_id": [
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
    ],
    "tool_start_key_collision": [
        _ev(
            "tool_execution_start",
            step_id="collide",
            data={"tool_name": "tool_x", "tool_args": {"a": 1}},
        ),
        _ev(
            "tool_execution_start",
            step_id="collide",
            data={"tool_name": "tool_y", "tool_args": {"b": 2}},
            timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
        ),
        _ev(
            "tool_execution_end",
            step_id="collide",
            data={"tool_name": "tool_y", "result": "result_y", "success": True},
            timestamp=datetime(2026, 1, 1, 12, 0, 2, tzinfo=timezone.utc),
        ),
    ],
    "skill_select_pair": [
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
    ],
    "user_and_ai_messages": [
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
    ],
    "start_without_end_is_running": [
        _ev("react_action_start", step_id="s1"),
    ],
    "orphan_end_is_dropped": [
        _ev("react_action_end", step_id="s1"),
    ],
    "out_of_order_timestamps": [
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
    ],
    "pending_after_completed": [
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
    ],
    "dag_execution_planning_pair": [
        _dag_exec_ev(
            "planning",
            task_id="t1",
            timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            completed_step_count=0,
            previous_step_count=0,
        ),
        _dag_exec_ev(
            "executing",
            task_id="t1",
            timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
            plan_step_count=2,
            replan=False,
            steps=[{"id": "s1"}, {"id": "s2"}],
        ),
    ],
    "dag_execution_resume_sequence": [
        _dag_exec_ev(
            "planning",
            task_id="t1",
            timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        ),
        _dag_exec_ev(
            "planning",
            task_id="t1",
            timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
        ),
        _dag_exec_ev(
            "executing",
            task_id="t1",
            timestamp=datetime(2026, 1, 1, 12, 0, 2, tzinfo=timezone.utc),
        ),
    ],
    "dag_execution_planning_failure": [
        _dag_exec_ev(
            "planning",
            task_id="t1",
            timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        ),
        _ev(
            "dag_execute_end",
            task_id="t1",
            data={"status": "failed"},
            timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
        ),
    ],
    "dag_plan_and_dag_execution_coexist": [
        _ev(
            "dag_plan_start",
            task_id="t1",
            timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        ),
        _dag_exec_ev(
            "planning",
            task_id="t1",
            timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
        ),
        _ev(
            "dag_plan_end",
            task_id="t1",
            timestamp=datetime(2026, 1, 1, 12, 0, 2, tzinfo=timezone.utc),
        ),
        _dag_exec_ev(
            "executing",
            task_id="t1",
            timestamp=datetime(2026, 1, 1, 12, 0, 3, tzinfo=timezone.utc),
        ),
    ],
    "unexposed_event_types_are_dropped": [
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
    ],
    "dag_execute_end_interrupted_does_not_leak_key_to_next_round": [
        _ev(
            "dag_execute_start",
            task_id="t1",
            timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        ),
        _dag_exec_ev(
            "planning",
            task_id="t1",
            timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
        ),
        _ev(
            "dag_execute_end",
            task_id="t1",
            data={"status": "interrupted"},
            timestamp=datetime(2026, 1, 1, 12, 0, 2, tzinfo=timezone.utc),
        ),
        # Round two's dag_execute_start/planning events are lost --
        # only the terminal failure survives. The interrupted round
        # one end above must have already cleared the open key, or
        # this failure would reach back and close round one's
        # stranded planning step instead of being a no-op.
        _ev(
            "dag_execute_end",
            task_id="t1",
            data={"status": "failed"},
            timestamp=datetime(2026, 1, 1, 12, 0, 3, tzinfo=timezone.utc),
        ),
    ],
    "empty_event_list": [],
}


@pytest.mark.parametrize(
    "events",
    _PROJECTOR_EQUIVALENCE_CASES.values(),
    ids=list(_PROJECTOR_EQUIVALENCE_CASES.keys()),
)
def test_projector_equivalent_to_batch(events):
    """``PublicStepProjector.from_history(events).materialized_steps()``,
    sorted by the same ``started_at`` key the batch driver's trailing
    sort applies, must equal ``map_trace_events_to_public_steps``
    exactly. This is the byte-for-byte pin that the batch function
    stays a pure driver over the projector, never a second copy of
    the folding logic.

    Literal output shapes per event family are already pinned by the
    direct-call tests above, which now exercise the projector itself.
    What this test guards instead is that the batch driver never grows
    a second, independent folding implementation.

    It is also the pin on the retention default: this path goes through
    ``from_history(events).materialized_steps()`` with no override, so a
    projector that stopped retaining finished steps by default would
    raise rather than return the task's whole timeline here.
    """
    expected = map_trace_events_to_public_steps(events)
    projected = PublicStepProjector.from_history(events).materialized_steps()
    projected.sort(key=lambda s: s["started_at"])
    assert projected == expected


@pytest.mark.parametrize(
    "events",
    _PROJECTOR_EQUIVALENCE_CASES.values(),
    ids=list(_PROJECTOR_EQUIVALENCE_CASES.keys()),
)
def test_projector_incremental_feed_matches_materialized_steps(events):
    """Folding the per-event increments ``feed()`` returns (keyed by
    step id, last write wins) must reconstruct exactly the same step
    objects ``materialized_steps()`` holds afterwards -- pins that the
    incremental ``feed()`` view and the projector's own materialized
    state can never drift apart.
    """
    projector = PublicStepProjector()
    by_id: Dict[str, Dict[str, Any]] = {}
    for event in events:
        for step in projector.feed(event):
            by_id[step["id"]] = step

    reconstructed = sorted(by_id.values(), key=lambda s: s["id"])
    materialized = sorted(projector.materialized_steps(), key=lambda s: s["id"])
    assert reconstructed == materialized


@pytest.mark.parametrize(
    "events",
    _PROJECTOR_EQUIVALENCE_CASES.values(),
    ids=list(_PROJECTOR_EQUIVALENCE_CASES.keys()),
)
def test_projector_without_retention_returns_the_same_steps_and_keeps_none(events):
    """``retain_finished=False`` changes only what the projector keeps
    after ``feed()`` returns, never what ``feed()`` itself returns.

    Feeds the same event sequence into a retaining and a non-retaining
    projector in lockstep, deep-copying each ``feed()`` result
    immediately (the same reason ``test_feed_return_value_reflects_step_
    state_at_call_time`` does -- a returned dict is mutated in place
    when its end event later arrives, and a non-retaining projector
    still hands back that same live object, it just doesn't also keep
    it in ``_finished``). The two sequences must be equal event-for-
    event. Afterwards, the non-retaining projector's ``_finished`` is
    ``None`` and ``materialized_steps()`` raises rather than returning a
    silently partial timeline.
    """
    retaining = PublicStepProjector()
    non_retaining = PublicStepProjector(retain_finished=False)
    retaining_results = []
    non_retaining_results = []
    for event in events:
        retaining_results.append(copy.deepcopy(retaining.feed(event)))
        non_retaining_results.append(copy.deepcopy(non_retaining.feed(event)))
    assert retaining_results == non_retaining_results
    assert non_retaining._finished is None
    with pytest.raises(RuntimeError):
        non_retaining.materialized_steps()


def test_feed_return_value_reflects_step_state_at_call_time():
    """``feed()``'s return is the same dict object later mutated in place
    when the matching end event finalizes the step -- so a caller that
    wants to know what a step looked like *at the moment of a given
    feed() call* must deep-copy the return value immediately, which is
    exactly what this test does. Without the ``copy.deepcopy`` calls
    below, ``start_steps[0]`` would silently become the finalized
    ``completed`` step by the time the last assertion runs, and this
    test could only ever observe the terminal state.
    """
    projector = PublicStepProjector()

    start_steps = copy.deepcopy(
        projector.feed(
            _ev(
                "react_action_start",
                step_id="s1",
                timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            )
        )
    )
    assert len(start_steps) == 1
    assert start_steps[0]["status"] == "running"
    assert start_steps[0]["completed_at"] is None

    end_steps = copy.deepcopy(
        projector.feed(
            _ev(
                "react_action_end",
                step_id="s1",
                timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
            )
        )
    )
    assert len(end_steps) == 1
    assert end_steps[0]["status"] == "completed"
    assert end_steps[0]["completed_at"] is not None

    # Proof the deep copy above actually decoupled the snapshot: the
    # start-time copy must still read "running" now that the live
    # object backing the un-copied return value has been mutated to
    # "completed" by the end event.
    assert start_steps[0]["status"] == "running"
    assert start_steps[0]["completed_at"] is None

    # Orphan end (no matching start): dropped silently, nothing to feed back.
    assert PublicStepProjector().feed(_ev("react_action_end", step_id="orphan")) == []

    # Unexposed event type: not part of the public step contract.
    assert PublicStepProjector().feed(_ev("llm_call_start", step_id="s1")) == []

    # Empty/falsy event_type: nothing identifiable to fold.
    assert PublicStepProjector().feed(_ev("", step_id="s1")) == []


# ===== dag_execution: planning-phase visibility from an existing signal =====


def test_dag_execution_planning_pair_becomes_thinking_planning():
    events = [
        _dag_exec_ev(
            "planning",
            task_id="t1",
            timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            completed_step_count=0,
            previous_step_count=0,
        ),
        _dag_exec_ev(
            "executing",
            task_id="t1",
            timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
            plan_step_count=2,
            replan=False,
            steps=[{"id": "s1"}, {"id": "s2"}],
        ),
    ]
    steps = map_trace_events_to_public_steps(events)
    assert len(steps) == 1
    s = steps[0]
    assert s["type"] == "thinking"
    assert s["data"]["phase"] == "planning"
    assert s["status"] == "completed"
    # id shape is pinned separately from the legacy dag_plan_* family
    # so the two counters can never collide.
    assert s["id"] == "thinking:planning:t1:1"


def test_dag_execution_planning_pair_with_integer_task_id():
    events = [
        _dag_exec_ev(
            "planning",
            task_id=42,
            timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        ),
        _dag_exec_ev(
            "executing",
            task_id=42,
            timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
        ),
    ]
    steps = map_trace_events_to_public_steps(events)
    assert len(steps) == 1
    s = steps[0]
    # An int task_id must interpolate into the key the same way a str
    # one does.
    assert s["id"] == "thinking:planning:42:1"
    assert s["status"] == "completed"
    assert s["completed_at"] == events[1].timestamp


def test_two_dag_execution_planning_entries_produce_two_separate_steps():
    """Each entry into planning -- not just replan -- produces its own
    step; a replanning entry reports the same public phase as a first
    planning entry.
    """
    events = [
        _dag_exec_ev(
            "planning",
            task_id="t1",
            timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        ),
        _dag_exec_ev(
            "executing",
            task_id="t1",
            timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
        ),
        _dag_exec_ev(
            "replanning",
            task_id="t1",
            timestamp=datetime(2026, 1, 1, 12, 0, 2, tzinfo=timezone.utc),
        ),
        _dag_exec_ev(
            "executing",
            task_id="t1",
            timestamp=datetime(2026, 1, 1, 12, 0, 3, tzinfo=timezone.utc),
        ),
    ]
    steps = map_trace_events_to_public_steps(events)
    assert len(steps) == 2
    assert all(s["data"]["phase"] == "planning" for s in steps)
    assert all(s["status"] == "completed" for s in steps)
    assert steps[0]["id"] == "thinking:planning:t1:1"
    assert steps[1]["id"] == "thinking:planning:t1:2"


def test_dag_execution_planning_without_executing_stays_running():
    """No executing row (planning failed or task is still in flight):
    the step stays running -- and the output is already non-empty with
    the first step reporting phase=planning, i.e. visible without
    waiting for the executing row.
    """
    events = [_dag_exec_ev("planning", task_id="t1")]
    steps = map_trace_events_to_public_steps(events)
    assert len(steps) == 1
    assert steps[0]["type"] == "thinking"
    assert steps[0]["data"]["phase"] == "planning"
    assert steps[0]["status"] == "running"
    assert steps[0]["completed_at"] is None


def test_dag_execution_planning_failure_does_not_disturb_other_steps():
    """A planning step left running does not block other, unrelated
    steps in the same stream from folding normally."""
    events = [
        _dag_exec_ev(
            "planning",
            task_id="t1",
            timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        ),
        _ev(
            "react_action_start",
            step_id="s1",
            timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
        ),
        _ev(
            "react_action_end",
            step_id="s1",
            timestamp=datetime(2026, 1, 1, 12, 0, 2, tzinfo=timezone.utc),
        ),
    ]
    steps = map_trace_events_to_public_steps(events)
    assert len(steps) == 2
    planning_step = next(s for s in steps if s["data"]["phase"] == "planning")
    action_step = next(s for s in steps if s["data"]["phase"] == "action")
    assert planning_step["status"] == "running"
    assert action_step["status"] == "completed"


def test_dag_execution_resume_sequence_first_running_second_completed():
    """A resumed-after-interruption sequence (planning, planning,
    executing -- no executing between the two plannings) leaves the
    first step running and closes only the second."""
    events = [
        _dag_exec_ev(
            "planning",
            task_id="t1",
            timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        ),
        _dag_exec_ev(
            "planning",
            task_id="t1",
            timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
        ),
        _dag_exec_ev(
            "executing",
            task_id="t1",
            timestamp=datetime(2026, 1, 1, 12, 0, 2, tzinfo=timezone.utc),
        ),
    ]
    steps = map_trace_events_to_public_steps(events)
    assert len(steps) == 2
    assert steps[0]["status"] == "running"
    assert steps[0]["completed_at"] is None
    assert steps[1]["status"] == "completed"
    assert steps[1]["completed_at"] is not None
    assert steps[0]["id"] != steps[1]["id"]


def test_dag_execution_translated_step_data_is_phase_only():
    """The translated step's ``data`` is exactly ``{"phase": ...}`` --
    none of the source event's other fields (completed_step_count,
    plan_step_count, replan, the full plan step list) leak through,
    whether the step is still running or already completed.
    """
    running_only = map_trace_events_to_public_steps(
        [
            _dag_exec_ev(
                "planning",
                task_id="t1",
                completed_step_count=1,
                previous_step_count=2,
            )
        ]
    )
    assert set(running_only[0]["data"].keys()) == {"phase"}

    events = [
        _dag_exec_ev(
            "planning",
            task_id="t1",
            timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            completed_step_count=1,
            previous_step_count=2,
        ),
        _dag_exec_ev(
            "executing",
            task_id="t1",
            timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
            plan_step_count=3,
            replan=True,
            steps=[{"id": "s1", "task": "do x"}],
        ),
    ]
    completed = map_trace_events_to_public_steps(events)
    assert len(completed) == 1
    assert set(completed[0]["data"].keys()) == {"phase"}
    assert completed[0]["data"]["phase"] == "planning"


@pytest.mark.parametrize("phase", ["completion_assessment", "some_future_phase"])
def test_dag_execution_other_phases_are_dropped(phase):
    """Phases outside {planning, replanning, executing} are not
    translated -- current behavior for dag_execution is preserved."""
    events = [_dag_exec_ev(phase, task_id="t1")]
    assert map_trace_events_to_public_steps(events) == []


def test_dag_execution_missing_or_malformed_phase_is_dropped():
    """Fail-safe: a missing/non-string ``phase``, or a ``data``
    payload that isn't even a dict, falls into the discard branch
    instead of raising."""
    no_phase = _dag_exec_ev("planning", task_id="t1")
    no_phase.data = {}
    non_str_phase = _dag_exec_ev("planning", task_id="t1")
    non_str_phase.data = {"phase": 123}
    none_phase = _dag_exec_ev("planning", task_id="t1")
    none_phase.data = {"phase": None}
    non_dict_data = _dag_exec_ev("planning", task_id="t1")
    non_dict_data.data = None
    for event in (no_phase, non_str_phase, none_phase, non_dict_data):
        assert PublicStepProjector().feed(event) == []


def test_dag_execution_executing_without_open_planning_is_noop():
    """Orphan executing (no preceding planning start): drops silently,
    same policy as an orphan end elsewhere in this module."""
    events = [_dag_exec_ev("executing", task_id="t1", plan_step_count=1, steps=[])]
    assert map_trace_events_to_public_steps(events) == []


def test_dag_plan_and_dag_execution_coexist_without_id_collision():
    """Legacy dag_plan_* and dag_execution-derived planning steps
    interleaved in one stream fold independently: separate counters,
    separate open keys, each end pairs with its own family's start,
    and the resulting ids never collide.
    """
    events = [
        _ev(
            "dag_plan_start",
            task_id="t1",
            timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        ),
        _dag_exec_ev(
            "planning",
            task_id="t1",
            timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
        ),
        _ev(
            "dag_plan_end",
            task_id="t1",
            timestamp=datetime(2026, 1, 1, 12, 0, 2, tzinfo=timezone.utc),
        ),
        _dag_exec_ev(
            "executing",
            task_id="t1",
            timestamp=datetime(2026, 1, 1, 12, 0, 3, tzinfo=timezone.utc),
        ),
    ]
    steps = map_trace_events_to_public_steps(events)
    assert len(steps) == 2
    assert all(
        s["type"] == "thinking" and s["data"] == {"phase": "planning"} for s in steps
    )
    assert all(s["status"] == "completed" for s in steps)
    ids = {s["id"] for s in steps}
    assert len(ids) == 2
    assert ids == {"thinking:plan:t1:1", "thinking:planning:t1:1"}


# ===== dag_execute_end: planning failure closes an open planning step =====


@pytest.mark.parametrize(
    "status",
    ["failed", "interrupted", "waiting_for_user", "completed", "__missing__", None],
)
def test_dag_execute_end_only_failed_closes_open_planning_step(status):
    """Only the literal string "failed" closes the open planning step;
    every other observed value -- interrupted, waiting_for_user,
    completed, a missing key, or None -- leaves it running. This is
    the opposite fallback direction from _terminal_status_from_event,
    which this branch does not reuse."""
    start = _dag_exec_ev(
        "planning",
        task_id="t1",
        timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )
    data = {} if status == "__missing__" else {"status": status}
    end = _ev(
        "dag_execute_end",
        task_id="t1",
        data=data,
        timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
    )
    steps = map_trace_events_to_public_steps([start, end])
    assert len(steps) == 1
    step = steps[0]
    assert step["type"] == "thinking"
    assert step["data"] == {"phase": "planning"}
    if status == "failed":
        assert step["status"] == "failed"
        assert step["completed_at"] is not None
    else:
        assert step["status"] == "running"
        assert step["completed_at"] is None


def test_dag_execute_end_failed_data_is_phase_only():
    """Anti-leak gate: dag_execute_end's data['result'] carries the
    full DAG pattern output. The closed step's data must be exactly
    {"phase": "planning"}, not a superset."""
    start = _dag_exec_ev(
        "planning",
        task_id="t1",
        timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )
    end = _ev(
        "dag_execute_end",
        task_id="t1",
        data={
            "status": "failed",
            "result": {
                "success": False,
                "error": "plan validation failed",
                "step_results": {"s1": {"output": "secret intermediate output"}},
                "output": "should never leak",
            },
        },
        timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
    )
    steps = map_trace_events_to_public_steps([start, end])
    assert len(steps) == 1
    assert set(steps[0]["data"].keys()) == {"phase"}


def test_dag_execute_end_without_open_planning_is_noop():
    """No open planning step to close -- same orphan-end policy as
    elsewhere in this module."""
    events = [
        _ev(
            "dag_execute_end",
            task_id="t1",
            data={"status": "failed"},
        )
    ]
    assert map_trace_events_to_public_steps(events) == []


def test_dag_execute_start_clears_stale_open_planning_key():
    """A fresh dag_execute_start clears any planning key left open by
    a prior round that never reached a terminal event, so a failure
    in the new round can't reach back and close the old round's
    stranded planning step (left as-is, still "running")."""
    round_one_start = _dag_exec_ev(
        "planning",
        task_id="t1",
        timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )
    dag_execute_start_round_two = _ev(
        "dag_execute_start",
        task_id="t1",
        timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
    )
    round_two_failure = _ev(
        "dag_execute_end",
        task_id="t1",
        data={"status": "failed"},
        timestamp=datetime(2026, 1, 1, 12, 0, 2, tzinfo=timezone.utc),
    )
    steps = map_trace_events_to_public_steps(
        [round_one_start, dag_execute_start_round_two, round_two_failure]
    )
    # Round 1's planning step is still open/running -- the round 2
    # failure must not have reached back and closed it.
    assert len(steps) == 1
    assert steps[0]["status"] == "running"
    assert steps[0]["completed_at"] is None


def test_dag_execute_start_planning_end_failed_closes_within_round():
    """Pin the in-round production order: start -> planning ->
    end(failed) closes the planning step opened by that same round.
    Guards the failed-closes path; complements the non-failed
    key-clearing coverage in the surrounding tests.
    """
    start = _ev(
        "dag_execute_start",
        task_id="t1",
        timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )
    planning = _dag_exec_ev(
        "planning",
        task_id="t1",
        timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
    )
    end = _ev(
        "dag_execute_end",
        task_id="t1",
        data={"status": "failed"},
        timestamp=datetime(2026, 1, 1, 12, 0, 2, tzinfo=timezone.utc),
    )
    steps = map_trace_events_to_public_steps([start, planning, end])
    assert len(steps) == 1
    assert steps[0]["status"] == "failed"
    assert steps[0]["completed_at"] is not None


@pytest.mark.parametrize("round_one_status", ["interrupted", "waiting_for_user"])
def test_dag_execute_end_interrupted_does_not_misattribute_next_rounds_failure(
    round_one_status,
):
    """Round one ends non-"failed" (its dag_execute_end status is not
    the literal "failed", so its planning step is deliberately left
    running), and round two's own dag_execute_start and planning
    events are lost -- only round two's terminal dag_execute_end
    (failed) survives.

    ``_open_dag_execution_key`` is scoped to the round that opened it:
    any dag_execute_end -- close or not -- clears it, since that
    round has now ended one way or another. Without that clearing, a
    non-"failed" end would leave the key pointing at round one's
    still-pending planning step, and round two's failure would reach
    back through it and close round one's step as "failed" --
    attributing round two's failure to round one's stranded step.
    With the key cleared, round two's failure finds no open key and is
    a no-op, leaving round one's step running as designed. Parametrized
    over the status, since the key clearing is unconditional on status
    -- a status-scoped clear (e.g. "interrupted" only) would have
    passed this test just as well.
    """
    round_one_start = _ev(
        "dag_execute_start",
        task_id="t1",
        timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )
    round_one_planning = _dag_exec_ev(
        "planning",
        task_id="t1",
        timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
    )
    round_one_end = _ev(
        "dag_execute_end",
        task_id="t1",
        data={"status": round_one_status},
        timestamp=datetime(2026, 1, 1, 12, 0, 2, tzinfo=timezone.utc),
    )
    # Round two's start and planning events are lost -- only its
    # terminal failure arrives.
    round_two_end = _ev(
        "dag_execute_end",
        task_id="t1",
        data={"status": "failed"},
        timestamp=datetime(2026, 1, 1, 12, 0, 3, tzinfo=timezone.utc),
    )
    steps = map_trace_events_to_public_steps(
        [round_one_start, round_one_planning, round_one_end, round_two_end]
    )
    assert len(steps) == 1
    assert steps[0]["status"] == "running"
    assert steps[0]["completed_at"] is None


def test_two_full_rounds_interrupted_then_failed_close_independently():
    """Two complete rounds -- neither drops any event -- each close on
    their own terms: round one (interrupted) stays running, round two
    (failed) closes as failed. Guards the case where both rounds emit
    their full start/planning/end triple, which the key clearing must
    leave untouched.
    """
    round_one_start = _ev(
        "dag_execute_start",
        task_id="t1",
        timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )
    round_one_planning = _dag_exec_ev(
        "planning",
        task_id="t1",
        timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
    )
    round_one_end = _ev(
        "dag_execute_end",
        task_id="t1",
        data={"status": "interrupted"},
        timestamp=datetime(2026, 1, 1, 12, 0, 2, tzinfo=timezone.utc),
    )
    round_two_start = _ev(
        "dag_execute_start",
        task_id="t1",
        timestamp=datetime(2026, 1, 1, 12, 0, 3, tzinfo=timezone.utc),
    )
    round_two_planning = _dag_exec_ev(
        "planning",
        task_id="t1",
        timestamp=datetime(2026, 1, 1, 12, 0, 4, tzinfo=timezone.utc),
    )
    round_two_end = _ev(
        "dag_execute_end",
        task_id="t1",
        data={"status": "failed"},
        timestamp=datetime(2026, 1, 1, 12, 0, 5, tzinfo=timezone.utc),
    )
    steps = map_trace_events_to_public_steps(
        [
            round_one_start,
            round_one_planning,
            round_one_end,
            round_two_start,
            round_two_planning,
            round_two_end,
        ]
    )
    assert len(steps) == 2
    round_one_step = next(s for s in steps if s["id"] == "thinking:planning:t1:1")
    round_two_step = next(s for s in steps if s["id"] == "thinking:planning:t1:2")
    assert round_one_step["status"] == "running"
    assert round_one_step["completed_at"] is None
    assert round_two_step["status"] == "failed"
    assert round_two_step["completed_at"] is not None
