"""Unit tests for the LLM-payload truncation helper used by the audit
trace infrastructure.

The v1 runtime (which used to host per-pattern audit emit sites
covered here) was removed in upstream PR #403
(``feat: [v2 part8] remove agent v1 runtime``); per-site audit tests
have been dropped along with their targets. This file keeps the
transport-agnostic infrastructure tests that still apply on v2.

The v2-runtime audit injection (centralized in
``agent/runtime.py:on_llm_start / on_llm_end``) is covered by a
follow-up PR.
"""

import copy
import re
from typing import Any, Callable, Dict, List

import pytest


def test_truncate_for_trace_short_string_passthrough() -> None:
    from xagent.core.agent.trace import truncate_for_trace

    assert truncate_for_trace("hi", max_bytes=100) == "hi"


def test_truncate_for_trace_long_string_truncated() -> None:
    from xagent.core.agent.trace import truncate_for_trace

    out = truncate_for_trace("x" * 1000, max_bytes=100)
    assert isinstance(out, str)
    assert "[truncated" in out
    # Head of original preserved
    assert out.startswith("x" * 100)


def test_truncate_for_trace_walks_dict_and_list() -> None:
    """Per-leaf truncation: dict shape preserved, only oversized string
    leaves get the ``[truncated N chars]`` marker.

    Uses ``max_bytes=4000`` so the post-trim serialized payload stays
    inside the hard-cap envelope and the original dict shape survives.
    The over-budget collapse path is covered by the separate
    ``..._dict_total_bounded_by_max_bytes`` test below.
    """
    from xagent.core.agent.trace import truncate_for_trace

    payload = {
        "messages": [
            {"role": "user", "content": "x" * 5000},
            {"role": "assistant", "content": "short"},
        ],
        "response": "y" * 5000,
        "model_name": "stub",
        "attempt": 1,
    }
    out = truncate_for_trace(payload, max_bytes=4000)
    # Dict survives: shape preserved, not collapsed to placeholder
    assert isinstance(out, dict)
    assert "__truncated__" not in out
    # Scalars unchanged
    assert out["model_name"] == "stub"
    assert out["attempt"] == 1
    # Large string truncated
    assert "[truncated" in out["response"]
    # Nested list element truncated
    assert "[truncated" in out["messages"][0]["content"]
    # Short nested element unchanged
    assert out["messages"][1]["content"] == "short"


def test_truncate_for_trace_walks_dict_per_field() -> None:
    """Multi-field dict: every oversized value gets the trim marker;
    shape is preserved.

    Per-field budget is ``max_bytes // N_fields``, so a single field can
    overshoot by its truncation-suffix overhead (~25 bytes). The overall
    cap is enforced one level up by
    :func:`normalize_llm_trace_payload`, which only routes the
    truncatable subset of fields through this helper.
    """
    from xagent.core.agent.trace import truncate_for_trace

    big = "z" * 5000
    payload = {"a": big, "b": big, "c": big, "d": big}
    out = truncate_for_trace(payload, max_bytes=200)

    assert isinstance(out, dict)
    assert set(out.keys()) == {"a", "b", "c", "d"}
    for key in ("a", "b", "c", "d"):
        assert "[truncated" in out[key], (
            f"expected per-field trim marker on {key!r}, got {out[key]!r}"
        )


def test_truncate_for_trace_multibyte_head_no_replacement_chars() -> None:
    """Multi-byte UTF-8 truncation must not produce U+FFFD chars.

    Regression: decoding the byte-sliced head with ``errors="replace"``
    inserts a replacement char whenever the slice ends mid-codepoint,
    which inflates ``len(head)`` and makes the reported truncated
    count inaccurate (can go negative for small budgets).
    """
    from xagent.core.agent.trace import truncate_for_trace

    # 100 CJK chars = 300 UTF-8 bytes; slice at 50 lands mid-codepoint.
    value = "中" * 100
    out = truncate_for_trace(value, max_bytes=50)
    assert isinstance(out, str)
    assert "�" not in out, f"replacement char leaked into head: {out!r}"
    assert "[truncated" in out


def test_truncate_for_trace_zero_disables() -> None:
    from xagent.core.agent.trace import truncate_for_trace

    long = "z" * 10_000
    assert truncate_for_trace(long, max_bytes=0) == long


def test_truncate_for_trace_deep_nesting_collapses() -> None:
    """Pathologically nested structures must not hit Python's recursion limit.

    Builds a 100-deep dict (well above the 50-frame guard, well below
    Python's default 1000-frame limit). Without the guard, sufficiently
    deep + large payloads could still blow the stack since each level
    eats a frame for both the dict comprehension and the recursive call.
    """
    from xagent.core.agent.trace import truncate_for_trace

    deep: Any = "leaf"
    for _ in range(100):
        deep = {"nested": deep}

    out = truncate_for_trace(deep, max_bytes=10_000)

    cur: Any = out
    depth = 0
    while isinstance(cur, dict) and "nested" in cur:
        cur = cur["nested"]
        depth += 1
        if depth > 200:
            pytest.fail("recursion guard never collapsed deep payload")

    assert isinstance(cur, str)
    assert "depth exceeds" in cur, (
        f"expected depth-guard placeholder at leaf, got {cur!r}"
    )


def test_ws_handler_drops_audit_only_events() -> None:
    """Server-only audit traces with ``__audit_only__: True`` must be
    dropped before reaching WebSocket clients.

    This is a security-critical assertion: the audit pipeline persists
    raw LLM I/O (messages, response) via DatabaseTraceHandler, and the
    drop in WebSocketTraceHandler is the only barrier preventing that
    same payload from being broadcast to connected clients.
    """
    from xagent.core.agent.trace import ACTION_START_LLM, TraceEvent
    from xagent.web.api.ws_trace_handlers import WebSocketTraceHandler

    handler = WebSocketTraceHandler(task_id=1)

    audit_event = TraceEvent(
        event_type=ACTION_START_LLM,
        task_id="t1",
        step_id="dag_skill_selection",
        data={
            "__audit_only__": True,
            "messages": [{"role": "user", "content": "raw prompt body"}],
            "action": "LLM call started",
        },
    )

    result = handler._convert_trace_event_to_stream_event(audit_event)
    assert result is None, (
        "audit_only event must be dropped before WS broadcast; "
        "got non-None stream event"
    )


def test_ws_handler_passes_non_audit_events() -> None:
    """Regression: dropping ``__audit_only__`` must not affect normal events."""
    from xagent.core.agent.trace import ACTION_START_LLM, TraceEvent
    from xagent.web.api.ws_trace_handlers import WebSocketTraceHandler

    handler = WebSocketTraceHandler(task_id=1)

    event = TraceEvent(
        event_type=ACTION_START_LLM,
        task_id="t1",
        step_id="step1",
        data={"action": "LLM call started", "step_name": "test_step"},
    )

    result = handler._convert_trace_event_to_stream_event(event)
    assert result is not None, "non-audit event was incorrectly dropped"
    assert result.get("step_id") == "step1"


@pytest.mark.asyncio
async def test_trace_action_end_truncates_llm_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Integration: trace_action_end with category=LLM applies the cap."""
    from xagent.core.agent.trace import (
        TraceCategory,
        Tracer,
        trace_action_end,
    )

    captured: List[Dict[str, Any]] = []

    class _RecordingTracer(Tracer):
        async def trace_event(  # type: ignore[override]
            self,
            event_type: Any,
            task_id: Any = None,
            step_id: Any = None,
            data: Any = None,
            parent_id: Any = None,
        ) -> str:
            captured.append(data or {})
            return "evt"

    monkeypatch.setenv("XAGENT_MAX_TRACE_PAYLOAD_BYTES", "2000")

    await trace_action_end(
        _RecordingTracer(),
        "t",
        "s",
        TraceCategory.LLM,
        data={"response": "x" * 10_000, "model_name": "m"},
    )

    assert len(captured) == 1
    # response is the only truncatable field; at 2000-byte cap it goes
    # through _reduce_response → _reduce_text and emits the [truncated]
    # marker. model_name (reserved) passes through verbatim.
    assert "[truncated" in captured[0]["response"]
    assert captured[0]["model_name"] == "m"


# ---------------------------------------------------------------------------
# normalize_llm_trace_payload — reserved-field preservation
# ---------------------------------------------------------------------------


def test_normalize_preserves_reserved_under_truncation() -> None:
    """Reserved control / routing / metrics fields must pass through
    untouched even when truncatable content fields are trimmed.

    Regression for the bug rogercloud flagged: hard-cap collapse used
    to drop ``__audit_only__`` and break WS visibility filtering.
    """
    from xagent.core.agent.trace import normalize_llm_trace_payload

    payload = {
        # routing / metadata / metrics — must survive verbatim
        "__audit_only__": True,
        "model_name": "gpt-4",
        "task_type": "dag_skill_selection",
        "step_id": "step-1",
        "step_name": "skill_selection",
        "action": "LLM call completed",
        "attempt": 1,
        "json_mode_failed": False,
        "success": True,
        "usage": {"input_tokens": 12, "output_tokens": 34, "total_tokens": 46},
        "messages_count": 2,
        # bulky content — must be trimmed
        "messages": [{"role": "user", "content": "x" * 200_000}],
        "response": "y" * 200_000,
    }
    out = normalize_llm_trace_payload(payload, max_bytes=4_000)

    assert isinstance(out, dict)
    assert out["__audit_only__"] is True
    assert out["model_name"] == "gpt-4"
    assert out["task_type"] == "dag_skill_selection"
    assert out["step_id"] == "step-1"
    assert out["step_name"] == "skill_selection"
    assert out["action"] == "LLM call completed"
    assert out["attempt"] == 1
    assert out["json_mode_failed"] is False
    assert out["success"] is True
    assert out["usage"] == {
        "input_tokens": 12,
        "output_tokens": 34,
        "total_tokens": 46,
    }
    assert out["messages_count"] == 2
    # Content fields hit the reducer at 4 KB; messages get the
    # semantic-reducer treatment (role preserved, content trimmed)
    # and response gets the text-reducer treatment.
    assert "[truncated" in str(out["messages"])
    assert "[truncated" in out["response"]


def test_normalize_passthrough_when_no_content_fields() -> None:
    """All-reserved payload returns unchanged (no spurious trim).

    Important so ``_emit_trace_event`` calling normalize on every LLM
    event is cheap when the event only carries metadata.
    """
    from xagent.core.agent.trace import normalize_llm_trace_payload

    payload = {"__audit_only__": True, "model_name": "gpt-4", "attempt": 2}
    out = normalize_llm_trace_payload(payload, max_bytes=100)
    assert out is payload or out == payload


def test_normalize_passes_through_non_dict() -> None:
    """Non-dict input returns as-is — defensive for unusual callers."""
    from xagent.core.agent.trace import normalize_llm_trace_payload

    assert normalize_llm_trace_payload("not a dict") == "not a dict"
    assert normalize_llm_trace_payload(None) is None


def test_normalize_zero_disables() -> None:
    """``max_bytes=0`` (XAGENT_MAX_TRACE_PAYLOAD_BYTES=0) disables truncation."""
    from xagent.core.agent.trace import normalize_llm_trace_payload

    long_response = "x" * 10_000
    payload = {"response": long_response, "model_name": "m"}
    out = normalize_llm_trace_payload(payload, max_bytes=0)
    assert out["response"] == long_response


def test_normalize_unknown_fields_pass_through() -> None:
    """Unknown fields (neither reserved nor truncatable) pass through.

    Future-proofs against silently truncating a new routing flag added
    by an audit emit that this list hasn't been updated for yet.
    """
    from xagent.core.agent.trace import normalize_llm_trace_payload

    payload = {
        "future_routing_flag": True,
        "future_metric": 42,
        "response": "y" * 10_000,  # known truncatable
    }
    out = normalize_llm_trace_payload(payload, max_bytes=2_000)
    assert out["future_routing_flag"] is True
    assert out["future_metric"] == 42
    assert "[truncated" in out["response"]


# ---------------------------------------------------------------------------
# PatternRuntime trace-boundary cap (Finding 3 regression)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v2_runtime_emit_trace_caps_llm_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: PatternRuntime.on_llm_end with a 100 KB response gets
    capped at the trace boundary. Previously the runtime bypassed
    truncate_for_trace entirely and emitted the raw payload to the
    tracer.
    """
    from xagent.core.agent.runtime import PatternRuntime

    events: List[Dict[str, Any]] = []

    class _CaptureTracer:
        async def trace_event(
            self,
            event_type: Any,
            task_id: Any = None,
            step_id: Any = None,
            data: Any = None,
            parent_id: Any = None,
        ) -> str:
            events.append(
                {
                    "event_type": getattr(event_type, "value", str(event_type)),
                    "task_id": task_id,
                    "step_id": step_id,
                    "data": dict(data or {}),
                }
            )
            return "evt"

    class _FakeContext:
        execution_id = "task-x"
        messages: List[Any] = []

        def record_llm_usage(self, **_: Any) -> None:
            pass

    monkeypatch.setenv("XAGENT_MAX_TRACE_PAYLOAD_BYTES", "1000")
    runtime = PatternRuntime(tracer=_CaptureTracer(), execution_id="task-x")

    await runtime.on_llm_end(context=_FakeContext(), response="x" * 100_000)

    assert events, "no trace event captured"
    data = events[-1]["data"]
    assert isinstance(data.get("response"), str)
    assert len(data["response"]) < 5_000, (
        f"response should be capped well under 100k, got {len(data['response'])}"
    )
    assert "[truncated" in data["response"]
    # Reserved control field survives the boundary cap
    assert data["success"] is True


@pytest.mark.asyncio
async def test_v2_runtime_emit_trace_does_not_cap_tool_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: only LLM-category events get normalized at the
    boundary. TOOL / DAG / REACT / COMPACT / GENERAL events must pass
    through their data unchanged so we don't silently truncate tool
    output, DAG plans, etc.
    """
    from xagent.core.agent.runtime import PatternRuntime
    from xagent.core.agent.trace import (
        TraceAction,
        TraceCategory,
        TraceEventType,
        TraceScope,
    )

    events: List[Dict[str, Any]] = []

    class _CaptureTracer:
        async def trace_event(
            self,
            event_type: Any,
            task_id: Any = None,
            step_id: Any = None,
            data: Any = None,
            parent_id: Any = None,
        ) -> str:
            events.append({"data": dict(data or {})})
            return "evt"

    monkeypatch.setenv("XAGENT_MAX_TRACE_PAYLOAD_BYTES", "1000")
    runtime = PatternRuntime(tracer=_CaptureTracer(), execution_id="task-x")

    tool_end_event = TraceEventType(
        TraceScope.ACTION, TraceAction.END, TraceCategory.TOOL
    )
    huge_tool_output = "z" * 100_000
    await runtime._emit_trace_event(
        tool_end_event,
        task_id="task-x",
        step_id="step-1",
        data={"tool_output": huge_tool_output, "tool_name": "noop"},
    )

    assert len(events[-1]["data"]["tool_output"]) == 100_000
    assert "[truncated" not in events[-1]["data"]["tool_output"]


# ---------------------------------------------------------------------------
# Per-field semantic reducers (Finding 4 — Roger 2026-05-16)
# ---------------------------------------------------------------------------


def test_normalize_messages_keeps_head_tail_under_cap() -> None:
    """Regression for Roger's exact example: 1000 messages × 5 KB at
    50 KB cap. Old equal-split implementation produced ~83 KB output and
    decayed every message to a 50-byte head + suffix. New semantic
    reducer preserves head + tail with full role metadata and meaningful
    content prefix, replaces middle with a single placeholder, and
    keeps total serialized ≤ cap.
    """
    import json

    from xagent.core.agent.trace import normalize_llm_trace_payload

    payload = {
        "__audit_only__": True,
        "model_name": "gpt-4o",
        "task_type": "dag_skill_selection",
        "messages": [{"role": "user", "content": "x" * 5000} for _ in range(1000)],
    }
    out = normalize_llm_trace_payload(payload, max_bytes=50_000)

    total = len(json.dumps(out, ensure_ascii=False).encode("utf-8"))
    assert total <= 50_000, f"envelope cap broken: {total} bytes > 50000"

    # Reserved metadata intact
    assert out["__audit_only__"] is True
    assert out["model_name"] == "gpt-4o"
    assert out["task_type"] == "dag_skill_selection"

    # messages: head + tail + middle placeholder, not 1000 broken entries
    msgs = out["messages"]
    assert len(msgs) < 10, f"expected head/tail summary, got {len(msgs)} entries"

    # First and last messages keep their role
    assert msgs[0]["role"] == "user"
    assert msgs[-1]["role"] == "user"

    # First/last messages keep substantial content prefix (not a 50-byte stub)
    assert len(msgs[0]["content"]) > 1000, (
        f"head message content too short: {len(msgs[0]['content'])}"
    )

    # Middle placeholder describes omitted count
    middle = [m for m in msgs if isinstance(m, dict) and "__truncated__" in m]
    assert middle, "expected middle placeholder for omitted messages"
    assert "messages omitted" in middle[0]["__truncated__"]


def test_normalize_messages_passthrough_when_small() -> None:
    """Short messages list well under budget passes through unchanged."""
    from xagent.core.agent.trace import normalize_llm_trace_payload

    payload = {
        "model_name": "m",
        "messages": [
            {"role": "system", "content": "hi"},
            {"role": "user", "content": "yo"},
        ],
    }
    out = normalize_llm_trace_payload(payload, max_bytes=50_000)
    assert out["messages"] == payload["messages"]


def test_normalize_tools_keeps_name_description() -> None:
    """tools: tool name + description preserved, only big parameters
    schema gets trimmed/collapsed."""
    from xagent.core.agent.trace import normalize_llm_trace_payload

    huge_schema = {
        "type": "object",
        "properties": {
            f"prop_{i}": {"type": "string", "description": "x" * 200}
            for i in range(100)
        },
    }
    payload = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for information",
                    "parameters": huge_schema,
                },
            },
        ],
    }
    out = normalize_llm_trace_payload(payload, max_bytes=2_000)

    tool = out["tools"][0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "web_search"
    assert tool["function"]["description"] == "Search the web for information"
    # parameters schema got collapsed; name + description survive
    assert isinstance(tool["function"]["parameters"], dict)


def test_normalize_tool_calls_keeps_id_name() -> None:
    """tool_calls: call id + function.name preserved, only arguments trimmed."""
    from xagent.core.agent.trace import normalize_llm_trace_payload

    payload = {
        "tool_calls": [
            {
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": '{"query":"' + "x" * 5000 + '"}',
                },
            },
        ],
    }
    out = normalize_llm_trace_payload(payload, max_bytes=2_000)

    call = out["tool_calls"][0]
    assert call["id"] == "call_abc"
    assert call["type"] == "function"
    assert call["function"]["name"] == "search"
    # arguments truncated
    assert "[truncated" in call["function"]["arguments"]


def test_normalize_response_dict_truncates_content() -> None:
    """response: dict shape preserved (e.g. _short_response output);
    text fields (content/answer/output) trimmed, scalars unchanged."""
    from xagent.core.agent.trace import normalize_llm_trace_payload

    payload = {
        "response": {
            "content": "x" * 10_000,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f"}}],
        },
    }
    out = normalize_llm_trace_payload(payload, max_bytes=2_000)

    resp = out["response"]
    assert isinstance(resp, dict)
    assert "[truncated" in resp["content"]
    # tool_calls scalar metadata preserved
    assert resp["tool_calls"][0]["id"] == "c1"


def test_normalize_envelope_bounded_under_mixed_oversized_fields() -> None:
    """All four heavy field types present and oversized — total
    serialized envelope still <= max_bytes."""
    import json

    from xagent.core.agent.trace import normalize_llm_trace_payload

    payload = {
        "__audit_only__": True,
        "model_name": "gpt-4o",
        "messages": [{"role": "user", "content": "x" * 5000}] * 500,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": f"tool_{i}",
                    "description": "d" * 200,
                    "parameters": {"big": "y" * 1000},
                },
            }
            for i in range(50)
        ],
        "tool_calls": [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": "f", "arguments": "z" * 1000},
            }
            for i in range(50)
        ],
        "response": {"content": "r" * 20_000},
    }
    out = normalize_llm_trace_payload(payload, max_bytes=50_000)
    total = len(json.dumps(out, ensure_ascii=False).encode("utf-8"))
    assert total <= 50_000, f"envelope cap broken under mixed payload: {total}"
    # Reserved metadata always survives
    assert out["__audit_only__"] is True
    assert out["model_name"] == "gpt-4o"


def test_normalize_extreme_payload_collapses_largest_field() -> None:
    """Pathological budget where even semantic reducers can't fit —
    envelope-level guard collapses the largest remaining truncatable
    field to placeholder. Reserved metadata still survives.
    """
    import json

    from xagent.core.agent.trace import normalize_llm_trace_payload

    payload = {
        "__audit_only__": True,
        "model_name": "gpt-4o",
        "messages": [{"role": "user", "content": "x" * 5000}] * 1000,
    }
    # Very small cap: reducer's per-field budget already too small
    out = normalize_llm_trace_payload(payload, max_bytes=500)

    total = len(json.dumps(out, ensure_ascii=False).encode("utf-8"))
    assert total <= 500 + 200, (  # small slack for edge case
        f"envelope cap broken under extreme cap: {total}"
    )
    # Reserved metadata still survives
    assert out["__audit_only__"] is True
    assert out["model_name"] == "gpt-4o"


def test_normalize_response_string_falls_back_to_text_reducer() -> None:
    """response can be a raw string (not dict) — should go through
    _reduce_text and emit the truncated marker."""
    from xagent.core.agent.trace import normalize_llm_trace_payload

    payload = {"model_name": "m", "response": "y" * 10_000}
    out = normalize_llm_trace_payload(payload, max_bytes=2_000)
    assert isinstance(out["response"], str)
    assert "[truncated" in out["response"]


# ---------------------------------------------------------------------------
# Skill selector audit emit coverage
# ---------------------------------------------------------------------------


class _RecordingTracer:
    """Capture-only tracer used by selector audit emit tests."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    async def trace_event(
        self,
        event_type: Any,
        task_id: Any = None,
        step_id: Any = None,
        data: Any = None,
        parent_id: Any = None,
    ) -> str:
        self.events.append(
            {
                "event_type": getattr(event_type, "value", str(event_type)),
                "task_id": task_id,
                "step_id": step_id,
                "data": dict(data or {}),
            }
        )
        return "evt"


# ---------------------------------------------------------------------------
# ConsoleTraceHandler log-render cap
# ---------------------------------------------------------------------------


def _checkpoint_payload(n_messages: int, chars_per_message: int) -> Dict[str, Any]:
    """A checkpoint-shaped payload: bulky content nested under ``snapshot``.

    Mirrors what ``TraceCheckpointStore`` emits — the top-level keys carry
    no truncatable field name, so the LLM-payload normalizer leaves this
    shape untouched and the console renderer is the only thing standing
    between it and the log line.
    """
    return {
        "checkpoint_type": "step_complete",
        "sequence": 42,
        "snapshot": {
            "task_id": 12345,
            "context": {
                "messages": [
                    {"role": "assistant", "content": "y" * chars_per_message}
                    for _ in range(n_messages)
                ],
                "variables": {"scratch": "z" * chars_per_message},
            },
        },
    }


def test_render_event_data_small_payload_unchanged() -> None:
    """A payload inside the budget renders exactly as plain interpolation."""
    from xagent.core.agent.trace import _render_event_data_for_log

    payload = {
        "step": 3,
        "status": "running",
        "tool": "search",
        "args": {"q": "hello"},
        "history": [1, 2, [3, 4]],
    }
    assert _render_event_data_for_log(payload, max_bytes=50_000) == f"{payload}"


def test_render_event_data_multibyte_small_payload_unchanged() -> None:
    """Multi-byte content inside the budget is not sliced."""
    from xagent.core.agent.trace import _render_event_data_for_log

    payload = {"content": "中文测试" * 20}
    assert _render_event_data_for_log(payload, max_bytes=50_000) == f"{payload}"


def test_render_event_data_multibyte_dict_verbatim_at_budget_boundary() -> None:
    """Pins byte-accurate accounting for a multi-byte payload sitting just
    under the budget, where a char-count-based width estimate would wrongly
    truncate it.

    Each ``中`` char is 3 UTF-8 bytes. A dict holding 16,500 of them renders
    to 49,515 bytes -- inside the 50,000 byte budget with the
    ``_LOG_CONTAINER_SLACK`` reserve (256 bytes) still unspent (probed
    boundary: this shape stays verbatim through 16,575 chars / 49,740
    bytes and starts truncating at 16,576 chars / 49,743 bytes). This test
    pins that a payload comfortably inside that boundary renders
    byte-for-byte identical to plain interpolation.

    This is a regression guard for charging a multi-byte leaf's width by
    Python character count instead of UTF-8 byte count -- e.g. estimating
    non-ASCII text at up to 4 bytes/char, which would count this payload
    as needing far more than the budget and truncate it hard even though
    it fits with room to spare. Verified by temporarily changing
    ``_rendered_width`` to ``len(text) * 4``: this payload's rendered
    output then shrinks from the true 49,515 bytes down to 37,204 bytes
    and this assertion goes red.
    """
    from xagent.core.agent.trace import _render_event_data_for_log

    payload = {"content": "中" * 16_500}
    assert _render_event_data_for_log(payload, max_bytes=50_000) == f"{payload}"


@pytest.mark.parametrize(
    "n_messages,chars_per_message",
    [
        (20, 200_000),  # few very large messages
        (2_000, 2_000),  # many medium messages
        (50_000, 100),  # very wide message list
    ],
)
def test_render_event_data_bounds_multi_megabyte_checkpoint(
    n_messages: int, chars_per_message: int
) -> None:
    """Multi-MB checkpoint payloads render within the cap.

    Covers the shape ``truncate_for_trace`` alone cannot bound: it splits
    the budget per leaf, so a wide message list still renders megabytes.

    Also pins the list-omission marker itself: the omitted count is not
    just "some number" but exactly the messages dropped from the list,
    i.e. the original length minus however many entries the shrink walk
    kept before the budget ran out. Probed real values for the three
    parametrizations above: (20, 200_000) omits 19, (2_000, 2_000) omits
    1_974, (50_000, 100) omits 49_645.
    """
    from xagent.core.agent.trace import (
        _render_event_data_for_log,
        _shrink_within_budget,
    )

    payload = _checkpoint_payload(n_messages, chars_per_message)
    assert len(f"{payload}") > 500_000, "fixture should be far over the cap"

    out = _render_event_data_for_log(payload, max_bytes=50_000)

    # Slack covers the trailing "...[truncated N chars]" marker.
    out_bytes = len(out.encode("utf-8"))
    assert out_bytes <= 50_000 + 100, f"log render not bounded: {out_bytes}"
    assert out.startswith("{'checkpoint_type': 'step_complete'")

    # The dropped messages are accounted for, not silently swallowed: the
    # last list entry is the "...[N more items]" marker, and N must equal
    # the original message count minus however many messages were kept
    # ahead of it.
    shrunk_messages = _shrink_within_budget(payload, 50_000)["snapshot"]["context"][
        "messages"
    ]
    marker = shrunk_messages[-1]
    assert isinstance(marker, str) and marker.startswith("...["), (
        f"expected a list-omission marker as the last entry, got {marker!r}"
    )
    match = re.fullmatch(r"\.\.\.\[(\d+) more items\]", marker)
    assert match, f"expected '...[N more items]' marker, got {marker!r}"
    kept = len(shrunk_messages) - 1
    assert int(match.group(1)) == n_messages - kept, (
        f"omitted count {match.group(1)} should equal "
        f"{n_messages} total - {kept} kept = {n_messages - kept}"
    )


def test_render_event_data_bounds_wide_dict() -> None:
    """Key-count explosion is bounded too: the budget is shared, so the
    remaining keys collapse into one marker instead of each getting a
    minimum slice of their own.

    ``truncate_for_trace`` cannot do this — its per-leaf split gives every
    one of the 100k keys a floor of its own, so its output stays in the
    megabytes.
    """
    from xagent.core.agent.trace import (
        _render_event_data_for_log,
        _shrink_within_budget,
    )

    payload = {"snapshot": {f"k{i}": "v" * 200 for i in range(100_000)}}

    # truncate_for_trace's per-leaf budget gives every one of the 100k keys
    # a floor of its own, so it renders this payload at well over 1 MB --
    # this is the reason _shrink_within_budget (a shared, not per-leaf,
    # budget) exists. Consolidating the two helpers is tracked in #623 and
    # is out of scope here.
    out = _render_event_data_for_log(payload, max_bytes=50_000)
    out_bytes = len(out.encode("utf-8"))
    assert out_bytes <= 50_000 + 100, f"log render not bounded: {out_bytes}"

    # The dropped keys are accounted for, not silently swallowed.
    shrunk = _shrink_within_budget(payload, 50_000)
    assert "more keys" in shrunk["snapshot"]["__omitted_keys__"]


def _chinese_step_results_payload(n_steps: int, id_len: int) -> Dict[str, Any]:
    """A checkpoint-shaped payload with Chinese plan step ids as dict keys.

    Mirrors ``step_results`` in a real execution checkpoint, where each key
    is the plan's own step id — often a short Chinese phrase — rather than
    an ASCII index.
    """
    return {
        "checkpoint_type": "step_complete",
        "step_results": {
            f"步骤{i}-{'验证与执行' * id_len}": {
                "status": "done",
                "output": "结果" * 50,
            }
            for i in range(n_steps)
        },
    }


def test_render_event_data_bounds_wide_chinese_keyed_dict() -> None:
    """A dict with many Chinese keys must stay within the byte budget.

    ``_shrink_node`` charges each dict key's rendered width with
    ``len(f"{key!r}: , ")``, which counts Python characters, not the UTF-8
    bytes the log line actually spends once encoded. Before the fix this
    payload rendered 144,594 bytes against a 50,000 byte budget — a 189%
    overrun — because every 200-character Chinese key was charged as if it
    cost 200 bytes when it actually costs about 600.

    ``_shrink_within_budget`` is asserted on directly (not only through
    ``_render_event_data_for_log``) because the final hard byte-cut in
    ``_render_event_data_for_log`` bounds total output size on its own —
    it would mask a regression in the per-key accounting. Checking the
    shrink step's own output is what actually pins the accounting fix.
    """
    from xagent.core.agent.trace import (
        _render_event_data_for_log,
        _shrink_within_budget,
    )

    payload = {f"键{i}" + "中" * 200: "v" for i in range(3_000)}

    shrunk = _shrink_within_budget(payload, 50_000)
    shrunk_bytes = len(f"{shrunk}".encode("utf-8"))
    assert shrunk_bytes <= 50_000 + 1_000, (
        f"per-key byte accounting broken: {shrunk_bytes}"
    )

    out = _render_event_data_for_log(payload, max_bytes=50_000)
    out_bytes = len(out.encode("utf-8"))
    assert out_bytes <= 50_000 + 100, f"log render not bounded: {out_bytes}"


def test_render_event_data_bounds_chinese_step_results_checkpoint() -> None:
    """A checkpoint-shaped payload with Chinese plan step ids as dict keys
    renders within the byte budget end to end, matching the shape a real
    execution checkpoint produces once a plan's step ids are Chinese text.
    """
    from xagent.core.agent.trace import _render_event_data_for_log

    payload = _chinese_step_results_payload(n_steps=800, id_len=8)

    out = _render_event_data_for_log(payload, max_bytes=50_000)
    out_bytes = len(out.encode("utf-8"))
    assert out_bytes <= 50_000 + 100, f"log render not bounded: {out_bytes}"


@pytest.mark.parametrize(
    "payload",
    [
        tuple("中文内容测试" * 4 for _ in range(20_000)),
        {f"中文集合元素{i}" * 4 for i in range(20_000)},
    ],
    ids=["tuple", "set"],
)
def test_render_event_data_bounds_chinese_unwalked_container(payload: Any) -> None:
    """Tuples and sets are not walked by ``_shrink_node`` — only str/dict/list
    are — so they pass through unshrunk and the final byte-domain hard cut in
    ``_render_event_data_for_log`` is the only thing bounding them.

    Before the fix this payload rendered 135,743 bytes against a 50,000
    byte budget because the cap sliced with ``rendered[:max_bytes]``, a
    Python character offset, instead of a UTF-8 byte offset.
    """
    from xagent.core.agent.trace import _render_event_data_for_log

    out = _render_event_data_for_log(payload, max_bytes=50_000)
    out_bytes = len(out.encode("utf-8"))
    assert out_bytes <= 50_000 + 100, f"log render not bounded: {out_bytes}"


@pytest.mark.parametrize(
    "payload,check",
    [
        (None, lambda out: out == "None"),
        ("plain string", lambda out: out == "plain string"),
        (42, lambda out: out == "42"),
        ([], lambda out: out == "[]"),
        ({}, lambda out: out == "{}"),
        (
            object(),
            lambda out: out.startswith("<object object at ") and out.endswith(">"),
        ),
        (
            {"obj": object()},
            lambda out: (
                out.startswith("{'obj': <object object at ") and out.endswith(">}")
            ),
        ),
        (("a", "b"), lambda out: out == "('a', 'b')"),
    ],
)
def test_render_event_data_tolerates_unusual_payloads(
    payload: Any, check: Callable[[str], bool]
) -> None:
    """Non-dict and non-serializable payloads render to their exact expected
    text, not just to *some* string.

    A bare payload at depth 0 renders through ``str()`` (no quotes, no
    escapes); ``object()`` two levels above depth 0 renders through the
    container's ``repr()``. The ``object()`` cases use ``startswith`` /
    ``endswith`` because the memory address in the default ``repr`` varies
    per run.
    """
    from xagent.core.agent.trace import _render_event_data_for_log

    out = _render_event_data_for_log(payload, max_bytes=50_000)
    assert check(out), out


def test_render_event_data_keeps_wide_scalar_dict_verbatim() -> None:
    """A dict of many small-int leaves that fits the budget must render
    byte-identical to plain interpolation.

    Before the fix, the scalar branch charged a flat 16 bytes per leaf
    regardless of what the leaf actually renders as, so this 45,780-byte
    payload — inside the 50,000 byte budget — was shrunk down to 23,378
    bytes with 2,107 keys collapsed into ``__omitted_keys__``.
    """
    from xagent.core.agent.trace import _render_event_data_for_log

    payload = {i: i for i in range(4_000)}
    # Precise byte count (45,780) is pinned in this test's docstring above;
    # the check here only needs to confirm the fixture is inside the
    # budget, since the fixture's exact repr length is CPython-format
    # dependent and the byte-for-byte equality assertion below is what
    # actually catches the regression.
    assert len(f"{payload}".encode("utf-8")) < 50_000
    assert _render_event_data_for_log(payload, max_bytes=50_000) == f"{payload}"


def test_render_event_data_keeps_llm_call_records_verbatim() -> None:
    """The real ``llm_calls`` shape from ``ExecutionContext.to_dict()`` —
    a list of dicts of six integer fields plus an ISO timestamp string —
    must render verbatim when it fits the budget.

    Before the fix this 39,918-byte payload was shrunk to 36,267 bytes by
    the same flat per-leaf charge as the scalar dict case above.
    """
    from xagent.core.agent.trace import _render_event_data_for_log

    payload = {
        "checkpoint_type": "step_complete",
        "snapshot": {
            "context": {
                "llm_calls": [
                    {
                        "input_tokens": 12_000 + i,
                        "output_tokens": 300 + i,
                        "total_tokens": 12_300 + 2 * i,
                        "message_index": i,
                        "prompt_message_count": i % 40,
                        "prompt_content_chars": 48_000 + i,
                        "timestamp": "2026-08-24T05:33:21.123456+00:00",
                    }
                    for i in range(200)
                ]
            }
        },
    }
    # Precise byte count (39,918) is pinned in this test's docstring above;
    # the check here only needs to confirm the fixture is inside the
    # budget, since the fixture's exact repr length is CPython-format
    # dependent and the byte-for-byte equality assertion below is what
    # actually catches the regression.
    assert len(f"{payload}".encode("utf-8")) < 50_000
    assert _render_event_data_for_log(payload, max_bytes=50_000) == f"{payload}"


def test_render_event_data_bounds_escape_heavy_strings() -> None:
    """A dict of many multi-line traceback strings must fit the budget in
    the shrink step itself, so the final hard cut never fires.

    Before the fix, a string leaf was charged its raw UTF-8 length while
    the log line renders it through ``repr()`` escaping, so this payload's
    shrunk copy reached 52,501 bytes against the same 50,000 byte budget
    and the final hard cut landed in the middle of a string value instead
    of on a container boundary.
    """
    from xagent.core.agent.trace import (
        _render_event_data_for_log,
        _shrink_within_budget,
    )

    traceback_text = (
        "Traceback (most recent call last):\n"
        '  File "/a/b/c.py", line 42, in run\n'
        "    raise ValueError('boom')\n"
        "ValueError: boom\n"
    ) * 2
    payload = {f"e{i}": traceback_text for i in range(400)}

    shrunk = _shrink_within_budget(payload, 50_000)
    shrunk_bytes = len(f"{shrunk}".encode("utf-8"))
    assert shrunk_bytes <= 50_000, f"escape-heavy accounting broken: {shrunk_bytes}"

    # No hard cut fired, so the output is exactly the shrunk copy's
    # rendering -- a truncation marker sliced mid-marker is not possible.
    out = _render_event_data_for_log(payload, max_bytes=50_000)
    assert out == f"{shrunk}", "hard cut fired: the shrunk copy overshot the budget"


def test_render_event_data_fits_single_leaf_inflated_by_escaping() -> None:
    """A single string leaf whose raw byte length is well inside the budget
    but whose ``repr()``-escaped rendering is not must still fit, and the
    fit loop's first slice must not be wider than the string itself.

    ``"\\x00" * 20_000`` is only 20,000 raw bytes, but every ``\\x00`` repr's
    as the four-character escape ``\\x00``, so the escaped rendering (about
    80,000 bytes) is the one that overflows the budget. The fit loop's
    starting slice was computed from the *budget*, not from the string's
    own length: when the budget-derived slice was wider than the 20,000
    raw bytes available, slicing did nothing, and the loop appended a
    truncation marker reporting 0 chars removed onto the untouched string
    -- an internally contradictory result that also overshot the budget
    (the escaped string plus the marker suffix is wider than either alone).
    """
    from xagent.core.agent.trace import (
        _render_event_data_for_log,
        _shrink_within_budget,
    )

    payload = {"blob": "\x00" * 20_000, "step": "s3"}

    shrunk = _shrink_within_budget(payload, 50_000)
    shrunk_bytes = len(f"{shrunk}".encode("utf-8"))
    assert shrunk_bytes <= 50_000, f"single-leaf accounting broken: {shrunk_bytes}"
    assert "truncated 0 chars" not in f"{shrunk}", (
        "marker claims nothing was cut from a leaf that had to be cut"
    )

    out = _render_event_data_for_log(payload, max_bytes=50_000)
    assert out == f"{shrunk}", "hard cut fired: the shrunk copy overshot the budget"


def test_render_event_data_marks_cyclic_dict() -> None:
    """A dict that references one of its own ancestors must be replaced by
    a cycle marker instead of being re-expanded.

    Before the fix there was no cycle detection: this self-referential dict
    rendered at 967 bytes (plain ``repr()`` renders the same dict at 23
    bytes, marking the cycle with ``{...}``).
    """
    from xagent.core.agent.trace import _render_event_data_for_log

    payload: Dict[str, Any] = {"a": 1}
    payload["self"] = payload

    out = _render_event_data_for_log(payload, max_bytes=50_000)
    assert "...[cyclic reference]" in out
    assert len(out.encode("utf-8")) < 200, len(out.encode("utf-8"))
    assert "depth exceeds" not in out, "cycle should be caught before the depth guard"


def test_render_event_data_marks_cyclic_list() -> None:
    """A list that contains itself must be replaced by a cycle marker.

    Before the fix this rendered at 5,266 bytes instead of being cut short.
    """
    from xagent.core.agent.trace import _render_event_data_for_log

    payload: List[Any] = ["x" * 100]
    payload.append(payload)

    out = _render_event_data_for_log(payload, max_bytes=50_000)
    assert "...[cyclic reference]" in out
    assert len(out.encode("utf-8")) < 300, len(out.encode("utf-8"))


def test_render_event_data_marks_mutually_referencing_dicts() -> None:
    """Two dicts that reference each other must be replaced by a cycle
    marker on the back-edge, not re-expanded indefinitely.

    Before the fix this rendered at 1,215 bytes instead of being cut short.
    """
    from xagent.core.agent.trace import _render_event_data_for_log

    first: Dict[str, Any] = {"name": "a"}
    second: Dict[str, Any] = {"name": "b", "peer": first}
    first["peer"] = second

    out = _render_event_data_for_log(first, max_bytes=50_000)
    assert "...[cyclic reference]" in out
    assert len(out.encode("utf-8")) < 200, len(out.encode("utf-8"))


def test_render_event_data_renders_shared_subtree_at_every_position() -> None:
    """The same acyclic container referenced from two places is not a
    cycle -- both positions must render it in full.

    This is the reverse guard for cycle detection: a "visited" set that
    only ever grows (instead of dropping ids on the way back up) would
    mistake this shape for a cycle at the second occurrence.
    """
    from xagent.core.agent.trace import _render_event_data_for_log

    shared = {"x": 1, "y": "hello"}
    payload = {"first": shared, "second": shared}
    assert _render_event_data_for_log(payload, max_bytes=50_000) == f"{payload}"

    message = {"role": "user", "content": "hi"}
    wide = {"messages": [message] * 500, "tail": "END"}
    out = _render_event_data_for_log(wide, max_bytes=50_000)
    assert "...[cyclic reference]" not in out
    assert "'tail'" in out


def test_render_event_data_keeps_real_omitted_keys_key() -> None:
    """A real payload key literally named ``__omitted_keys__`` must survive
    truncation instead of being overwritten by the omission-count sentinel.

    Real ``event.data`` keys are the tracer's own literals
    (``checkpoint_type``, ``snapshot``, ``sequence``), so this is a
    defensive guard against a payload the tracer doesn't control, not a
    scenario observed in production. Data outranks the omission count: if
    the sentinel key already holds real payload data, no omission count is
    reported at all.
    """
    from xagent.core.agent.trace import _render_event_data_for_log

    payload = {
        "__omitted_keys__": "REAL DATA",
        "big": "x" * 100_000,
        "z": "zz",
    }
    out = _render_event_data_for_log(payload, max_bytes=50_000)
    assert "REAL DATA" in out, "sentinel clobbered a real payload key"


@pytest.mark.parametrize(
    "payload",
    ["z" * 49_999, "a\n" * 24_999],
    ids=["plain", "newlines"],
)
def test_render_event_data_keeps_top_level_string_verbatim(payload: str) -> None:
    """A bare string payload renders through ``str()``, not ``repr()``, so
    it must be charged without the quotes and escapes ``repr()`` would add.

    Before the fix, a top-level string was charged as if it would be
    wrapped in quotes like a nested one: ``"z" * 49_999`` (49,999 bytes,
    inside the budget) was charged as ``49,999 + 2 = 50,001`` bytes,
    truncated, and then truncated again by the final hard cut, ending up
    at 50,023 bytes even though ``f"{data}"`` renders it whole.
    """
    from xagent.core.agent.trace import _render_event_data_for_log

    assert len(payload.encode("utf-8")) <= 50_000
    assert _render_event_data_for_log(payload, max_bytes=50_000) == payload


def test_render_event_data_zero_cap_disables_truncation() -> None:
    """``XAGENT_MAX_TRACE_PAYLOAD_BYTES=0`` keeps the old behaviour."""
    from xagent.core.agent.trace import _render_event_data_for_log

    payload = _checkpoint_payload(5, 10_000)
    assert _render_event_data_for_log(payload, max_bytes=0) == f"{payload}"


def test_render_event_data_small_budget_hard_cut() -> None:
    """This is the only case in this file that runs with a budget below
    the default 50_000 -- everything else exercises the shrink walk at
    production scale. A ``max_bytes=200`` budget against a dict with
    several 1000-char string fields is too small for ``_shrink_node`` to
    land under the cap on its own (each field's own truncation marker,
    plus the ``__omitted_keys__`` marker for the fields it can't fit at
    all, already eats past 200 bytes), so ``_render_event_data_for_log``'s
    final hard byte-cut fires on top of the shrink pass. That hard cut
    slices the rendered string to ``max_bytes`` and appends its own
    ``...[truncated N chars]`` marker, and the marker itself is what pushes
    the result over budget -- measured at 222 bytes for this fixture, 22
    bytes over the 200-byte budget.

    The lower-bound assertion checks for the hard-cut marker itself, not
    the overshoot amount: if the overshoot is ever fixed so the result
    lands at or under 200 bytes, this test should not go red for no
    longer overshooting -- only for the marker disappearing.
    """
    from xagent.core.agent.trace import _render_event_data_for_log

    payload = {"a": "x" * 1000, "b": "y" * 1000, "c": "z" * 1000}
    out = _render_event_data_for_log(payload, max_bytes=200)
    out_bytes = len(out.encode("utf-8"))
    assert out_bytes <= 230, (
        f"expected the hard cut to stay near budget, got {out_bytes}"
    )
    assert re.search(r"\.\.\.\[truncated \d+ chars\]$", out), (
        f"expected the hard-cut marker at the end of the output, got {out!r}"
    )


@pytest.mark.asyncio
async def test_console_cap_does_not_shrink_the_persisted_payload(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The safety promise this PR exists to guarantee: the byte cap applies
    only to the string the console handler builds for its own log line. Any
    other handler on the same ``Tracer`` -- a database writer, a checkpoint
    store, a websocket broadcaster -- must still receive ``event.data``
    exactly as it was passed in, unshrunk.

    Wires a real ``Tracer`` with two handlers, ``ConsoleTraceHandler`` and a
    recording handler standing in for a persistence sink, and fires one
    event with a payload whose rendered width is far past the 50_000-byte
    console budget. The recording handler's copy of ``event.data`` must be
    the identical object (not a shrunk copy) and equal to the original, and
    the console log line must still have been capped.
    """
    import logging

    from xagent.core.agent.trace import (
        SYSTEM_INFO,
        BaseTraceHandler,
        ConsoleTraceHandler,
        Tracer,
    )

    class _RecordingPersistenceHandler(BaseTraceHandler):
        """Stands in for a database/checkpoint handler: keeps whatever
        ``event.data`` object it was handed, unmodified."""

        def __init__(self) -> None:
            super().__init__()
            self.received: List[Any] = []

        async def _handle_system_event(self, event: Any) -> None:
            self.received.append(event.data)

    monkeypatch.setenv("XAGENT_MAX_TRACE_PAYLOAD_BYTES", "50000")

    tracer = Tracer()
    tracer.add_handler(ConsoleTraceHandler())
    persistence_handler = _RecordingPersistenceHandler()
    tracer.add_handler(persistence_handler)

    original_payload = {"snapshot": {f"k{i}": "v" * 200 for i in range(2_000)}}
    assert len(f"{original_payload}".encode("utf-8")) > 50_000, (
        "fixture should be far over the console budget"
    )
    # Snapshot the content BEFORE dispatch: the tracer passes ``data`` by
    # reference, so comparing received[0] against original_payload would be
    # comparing an object with itself and could never catch an in-place
    # mutation by the console handler.
    expected_content = copy.deepcopy(original_payload)

    with caplog.at_level(logging.INFO, logger="xagent.core.agent.trace"):
        await tracer.trace_event(SYSTEM_INFO, data=original_payload)

    # Persistence side: same object, untouched.
    assert len(persistence_handler.received) == 1
    assert persistence_handler.received[0] is original_payload
    assert persistence_handler.received[0] == expected_content

    # Console side: the log line is still capped. ``Tracer.trace_event``
    # itself logs several diagnostic lines through the same module logger
    # (dispatch bookkeeping), so filter down to the one line
    # ``ConsoleTraceHandler._handle_system_event`` actually renders.
    console_records = [
        r for r in caplog.records if r.getMessage().startswith("[SYSTEM]")
    ]
    assert len(console_records) == 1
    message_bytes = len(console_records[0].getMessage().encode("utf-8"))
    assert message_bytes <= 50_000 + 30, f"log line not bounded: {message_bytes}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope_kwargs",
    [
        {"scope": "TASK", "action": "START", "task_id": "t1"},
        {"scope": "STEP", "action": "UPDATE", "step_id": "s1"},
        {"scope": "ACTION", "action": "INFO", "step_id": "s1"},
        {"scope": "SYSTEM", "action": "UPDATE"},
    ],
)
async def test_console_handler_caps_every_scope(
    scope_kwargs: Dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """All four ConsoleTraceHandler scopes cap the payload they log."""
    import logging

    from xagent.core.agent.trace import (
        ConsoleTraceHandler,
        TraceAction,
        TraceCategory,
        TraceEvent,
        TraceEventType,
        TraceScope,
    )

    attribution = dict(scope_kwargs)
    scope_name = attribution.pop("scope")
    action_name = attribution.pop("action")

    event_type = TraceEventType(
        getattr(TraceScope, scope_name),
        getattr(TraceAction, action_name),
        TraceCategory.GENERAL,
    )
    event = TraceEvent(
        event_type, data=_checkpoint_payload(2_000, 2_000), **attribution
    )

    with caplog.at_level(logging.INFO, logger="xagent.core.agent.trace"):
        await ConsoleTraceHandler().handle_event(event)

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    message_bytes = len(message.encode("utf-8"))
    assert message_bytes <= 50_000 + 200, f"log line not bounded: {message_bytes}"
    assert "[truncated" in message
