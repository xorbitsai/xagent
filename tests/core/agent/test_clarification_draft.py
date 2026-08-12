from __future__ import annotations

from typing import Any

import pytest

from xagent.core.agent import ExecutionContext, ReActPattern
from xagent.core.agent.checkpoint import CHECKPOINT_TYPE, TraceCheckpointStore
from xagent.core.agent.clarification import (
    CLARIFICATION_SOURCES,
    ClarificationDraft,
    draft_from_waiting_request,
)
from xagent.core.agent.pattern.react.react import _normalize_ask_user_interactions
from xagent.web.api.trace_handlers import DatabaseTraceHandler


class FakeLLM:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.responses.pop(0)


class RecordingTraceBackend:
    """Fake tracer that captures the real ``_event_payload`` output.

    Mirrors ``PersistentTraceBackend`` in ``tests/core/agent/test_checkpoint.py``:
    ``TraceCheckpointStore.checkpoint`` builds the event payload with its own
    real ``_event_payload`` method, this fake only records what it is handed.
    """

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def trace_event(
        self,
        event_type: Any,
        *,
        task_id: str | None = None,
        data: dict[str, Any] | None = None,
        require_persisted: bool = False,
    ) -> str:
        del event_type, require_persisted
        self.events.append({"task_id": task_id, "data": dict(data or {})})
        return f"event-{len(self.events)}"


def _send_message_request(
    *, message_count: int = 2, tool_call_id: str = "call-send"
) -> dict[str, Any]:
    return {
        "tool_call_id": tool_call_id,
        "tool_name": "send_message",
        "message": "Choose A or B",
        "message_type": "question",
        "task_text": None,
        "message_count": message_count,
    }


def _ask_user_question_request(
    *,
    message: str = "Pick one",
    interactions: list[dict[str, Any]] | None = None,
    message_count: int = 3,
    tool_call_id: str = "call-ask",
) -> dict[str, Any]:
    normalized = _normalize_ask_user_interactions(
        interactions
        if interactions is not None
        else [{"type": "select_one", "field": "choice", "label": "Choice"}]
    )
    return {
        "tool_call_id": tool_call_id,
        "tool_name": "ask_user_question",
        "message": message,
        "message_type": "question",
        "interactions": normalized,
        "task_text": None,
        "message_count": message_count,
    }


def _tool_waiting_request(
    *,
    requests: list[dict[str, Any]] | None = None,
    message_count: int = 5,
) -> dict[str, Any]:
    default_requests = [
        {
            "tool_call_id": "wait-a",
            "tool_name": "approval_gate",
            "interaction_id": "interaction-a",
            "message": "Continue?",
            "message_type": "confirmation",
            "interactions": _normalize_ask_user_interactions(
                [{"type": "select_one", "field": "decision", "label": "Decision"}]
            ),
        }
    ]
    resolved_requests = requests if requests is not None else default_requests
    return {
        "kind": "tool_waiting_for_user",
        "requests": resolved_requests,
        "message": resolved_requests[0]["message"] if resolved_requests else "",
        "message_type": "confirmation",
        "interactions": [],
        "task_text": None,
        "message_count": message_count,
    }


def test_send_message_draft_classifies_and_shapes_requests() -> None:
    """A send_message waiting request classifies as source send_message with one request item."""

    request = _send_message_request()

    draft = draft_from_waiting_request(request, execution_id="exec-1", step_id=None)

    assert draft is not None
    assert draft.source == "send_message"
    # Pins the constant's actual value rather than restating the equality
    # assert above: ``draft.source in CLARIFICATION_SOURCES`` would pass
    # for any of the three literals and catches nothing on its own.
    assert CLARIFICATION_SOURCES == {
        "send_message",
        "ask_user_question",
        "tool_waiting",
    }
    assert len(draft.requests) == 1
    assert draft.requests[0].tool_call_id == "call-send"
    assert draft.requests[0].interaction_id == "call-send"
    assert draft.requests[0].tool_name == "send_message"
    assert draft.origin_execution_id == "exec-1"
    assert draft.interactions == ()
    assert draft.message == "Choose A or B"


def test_ask_user_question_draft_classifies_and_matches_normalized_interactions() -> (
    None
):
    """An ask_user_question waiting request classifies as ask_user_question, and its
    interactions equal the real normalizer's output for the same raw input.
    """

    raw_interactions = [
        {"type": "select_one", "field": "choice", "label": "Choice"},
        {"type": "text_input", "field": "note", "label": "Note"},
    ]
    normalized = _normalize_ask_user_interactions(raw_interactions)
    request = _ask_user_question_request(interactions=raw_interactions)

    draft = draft_from_waiting_request(request, execution_id="exec-1", step_id=None)

    assert draft is not None
    assert draft.source == "ask_user_question"
    assert len(draft.requests) == 1
    assert draft.requests[0].tool_call_id == "call-ask"
    assert draft.requests[0].interaction_id == "call-ask"
    assert draft.requests[0].tool_name == "ask_user_question"
    assert draft.origin_execution_id == "exec-1"
    assert draft.interactions == tuple(normalized)


def test_tool_waiting_draft_classifies_and_shapes_requests() -> None:
    """A tool_waiting request classifies as source tool_waiting with one request item."""

    request = _tool_waiting_request()

    draft = draft_from_waiting_request(request, execution_id="exec-1", step_id=None)

    assert draft is not None
    assert draft.source == "tool_waiting"
    assert len(draft.requests) == 1
    assert draft.requests[0].tool_call_id == "wait-a"
    assert draft.requests[0].interaction_id == "interaction-a"
    assert draft.requests[0].tool_name == "approval_gate"
    assert draft.origin_execution_id == "exec-1"


def test_empty_form_ask_user_question_still_classifies_as_ask_user_question() -> None:
    """An empty-form ask_user_question is still ``ask_user_question``, not send_message.

    The classifier reads key presence (``"interactions" in request``), not
    truthiness -- an empty ``interactions`` list must not fall through to the
    ``send_message`` branch.
    """

    request = _ask_user_question_request(message="", interactions=[])

    draft = draft_from_waiting_request(request, execution_id="exec-1", step_id=None)

    assert draft is not None
    assert draft.source == "ask_user_question"
    assert draft.interactions == ()


def test_tool_waiting_multi_tool_requests_and_interaction_id_fallback() -> None:
    """Multiple waiting tools produce multiple requests; a missing
    ``interaction_id`` falls back to ``tool_call_id``, mirroring react.py's
    own fallback rule at the ``_pause_for_tool_results`` request-building site.
    """

    request = _tool_waiting_request(
        requests=[
            {
                "tool_call_id": "wait-a",
                "tool_name": "approval_gate",
                "interaction_id": "interaction-a",
                "message": "Continue A?",
                "message_type": "confirmation",
                "interactions": [],
            },
            {
                "tool_call_id": "wait-b",
                "tool_name": "second_gate",
                "interaction_id": "",
                "message": "Continue B?",
                "message_type": "confirmation",
                "interactions": [],
            },
        ]
    )

    draft = draft_from_waiting_request(request, execution_id="exec-1", step_id=None)

    assert draft is not None
    assert len(draft.requests) == 2
    assert draft.requests[0].interaction_id == "interaction-a"
    assert draft.requests[1].tool_call_id == "wait-b"
    assert draft.requests[1].interaction_id == "wait-b"


def test_missing_message_and_interactions_returns_none_without_raising() -> None:
    """An old-schema waiting request with neither field degrades to no draft."""

    request = {
        "tool_call_id": "call-legacy",
        "task_text": None,
        "message_count": 0,
    }

    draft = draft_from_waiting_request(request, execution_id="exec-1", step_id=None)

    assert draft is None


@pytest.mark.asyncio
async def test_checkpoint_snapshot_survives_real_trace_serializer() -> None:
    """A real checkpoint snapshot -- built from a real waiting run's
    real ``pattern.get_state()`` -- round-trips through the real
    ``DatabaseTraceHandler._serialize_data_for_json`` (not a copy of it).
    """

    pattern = ReActPattern(max_iterations=2)
    context = ExecutionContext(execution_id="exec-checkpoint")
    context.add_user_message("Ask")
    llm = FakeLLM(
        [
            {
                "tool_calls": [
                    {
                        "id": "call-ask",
                        "function": {
                            "name": "ask_user_question",
                            "arguments": (
                                '{"message":"Pick one","interactions":'
                                '[{"type":"select_one","field":"choice","label":"Choice"}]}'
                            ),
                        },
                    }
                ]
            }
        ]
    )

    result = await pattern.run(context=context, tools=[], llm=llm)
    assert result["status"] == "waiting_for_user"
    waiting_request_before = dict(pattern.waiting_for_user_request or {})

    backend = RecordingTraceBackend()
    store = TraceCheckpointStore(backend)
    await store.checkpoint(
        type="checkpoint",
        label="waiting_for_user",
        execution_id="exec-checkpoint",
        pattern="ReActPattern",
        pattern_state=pattern.get_state(),
        status="waiting_for_user",
    )
    event_payload = backend.events[0]["data"]

    handler = DatabaseTraceHandler(1)
    out = handler._serialize_data_for_json(event_payload)

    assert out["checkpoint_type"] == CHECKPOINT_TYPE
    assert (
        out["snapshot"]["pattern_state"]["waiting_for_user_request"]
        == waiting_request_before
    )
    assert "_serialization_error" not in out


def test_with_origin_step_recomputes_marker_for_different_steps() -> None:
    """``with_origin_step`` is a frozen copy that also redoes the marker.

    Guards the invariant that a plain ``dataclasses.replace`` (origin_step_id
    only) would silently violate: two different steps resolving the same
    turn must not collapse onto the same marker.
    """

    base = draft_from_waiting_request(
        _send_message_request(), execution_id="exec-1", step_id=None
    )
    assert base is not None

    draft_a = base.with_origin_step("step-a")
    draft_b = base.with_origin_step("step-b")

    assert isinstance(draft_a, ClarificationDraft)
    assert draft_a.origin_step_id == "step-a"
    assert draft_b.origin_step_id == "step-b"
    assert draft_a.turn_marker != draft_b.turn_marker

    # Flattening origin_step_id back out through the same with_origin_step()
    # method (rather than string surgery on the encoded marker) leaves the
    # two markers equal to each other and to the original unattributed
    # draft -- origin_step_id was the only field that varied between them.
    flattened_a = draft_a.with_origin_step("").turn_marker
    flattened_b = draft_b.with_origin_step("").turn_marker
    assert flattened_a == flattened_b == base.turn_marker


def test_distinct_nonempty_tool_call_ids_produce_distinct_markers() -> None:
    """Two waiting requests that differ only in ``tool_call_id`` produce
    different markers, as long as both ids are non-empty -- the guarantee
    ``ReActPattern._normalize_tool_calls`` enforces in production by
    falling back to ``f"tool_call_{index}"`` for any missing or empty id.

    An empty-``tool_call_id`` collision is reachable only from a corrupted
    checkpoint: ``load_state`` restores ``waiting_for_user_request``
    verbatim from whatever dict it is given, with no re-normalization, so a
    hand-edited or pre-normalization legacy checkpoint could still carry an
    empty id.
    """

    request_a = _send_message_request(tool_call_id="call-a")
    request_b = _send_message_request(tool_call_id="call-b")

    draft_a = draft_from_waiting_request(request_a, execution_id="exec-1", step_id=None)
    draft_b = draft_from_waiting_request(request_b, execution_id="exec-1", step_id=None)

    assert draft_a is not None and draft_b is not None
    assert draft_a.turn_marker != draft_b.turn_marker
