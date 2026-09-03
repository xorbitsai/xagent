from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from xagent.core.agent import ExecutionContext, PatternRuntime, ReActPattern
from xagent.core.agent.clarification import (
    ClarificationDraft,
    draft_from_waiting_request,
)
from xagent.core.agent.trace import TraceAction
from xagent.web.api.trace_handlers import DatabaseTraceHandler
from xagent.web.api.websocket import SharedWebSocketTracer
from xagent.web.api.ws_trace_handlers import WebSocketTraceHandler


class CalculatorArgs(BaseModel):
    expression: str


class FakeLLM:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.responses.pop(0)


class ResumableTool:
    """A tool with its own suspended interaction state, resolved by a follow-up reply."""

    def __init__(self) -> None:
        self.metadata = SimpleNamespace(
            name="approval_gate",
            description="Run an action after the user responds.",
        )
        self.user_response: str | None = None

    def args_type(self) -> type[BaseModel]:
        return CalculatorArgs

    def resume_user_interaction(self, *, interaction_id: str, response: str) -> None:
        self.user_response = response

    async def run_json_async(self, args: dict[str, Any]) -> Any:
        if self.user_response is None:
            return {
                "success": False,
                "status": "waiting_for_user",
                "interaction_id": "interaction-1",
                "message": "Should the action continue?",
                "message_type": "confirmation",
                "interactions": [
                    {
                        "type": "select_one",
                        "field": "decision",
                        "label": "Decision",
                        "options": [
                            {"label": "Continue", "value": "continue"},
                            {"label": "Stop", "value": "stop"},
                        ],
                    }
                ],
            }
        return {"success": True, "expression": args["expression"]}


def _send_message_llm() -> FakeLLM:
    return FakeLLM(
        [
            {
                "tool_calls": [
                    {
                        "id": "call-send",
                        "function": {
                            "name": "send_message",
                            "arguments": (
                                '{"message":"Choose A or B",'
                                '"message_type":"question","expect_response":true}'
                            ),
                        },
                    }
                ]
            }
        ]
    )


def _ask_user_question_llm() -> FakeLLM:
    return FakeLLM(
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("llm_factory", "tools_factory", "expected_source"),
    [
        (_send_message_llm, lambda: [], "send_message"),
        (_ask_user_question_llm, lambda: [], "ask_user_question"),
    ],
)
async def test_waiting_return_carries_draft_matching_recomputed_value(
    llm_factory: Any, tools_factory: Any, expected_source: str
) -> None:
    """The returned ``clarification_draft`` equals the draft recomputed
    from the same ``waiting_for_user_request`` right after the run.
    """

    pattern = ReActPattern(max_iterations=2)
    runtime = PatternRuntime(execution_id="exec-tp1")
    context = ExecutionContext(execution_id="exec-tp1")
    context.add_user_message("Ask")

    result = await pattern.run(
        context=context, tools=tools_factory(), llm=llm_factory(), runtime=runtime
    )

    assert result["status"] == "waiting_for_user"
    expected = draft_from_waiting_request(
        pattern.waiting_for_user_request,
        execution_id=context.execution_id,
        step_id=None,
    )
    assert result["clarification_draft"] == expected
    assert result["clarification_draft"].source == expected_source
    assert (
        result["clarification_draft"].event_id
        == runtime.outbound_messages[-1]["event_id"]
    )
    assert (
        pattern.waiting_for_user_request["event_id"]
        == result["clarification_draft"].event_id
    )


@pytest.mark.asyncio
async def test_waiting_return_via_tool_carries_tool_waiting_draft() -> None:
    """The multi-tool waiting point also carries
    a draft equal to the one recomputed from the same request.
    """

    tool = ResumableTool()
    pattern = ReActPattern(max_iterations=2)
    runtime = PatternRuntime(execution_id="exec-tp1-tool")
    context = ExecutionContext(execution_id="exec-tp1-tool")
    context.add_user_message("Run the gated action.")
    llm = FakeLLM(
        [
            {
                "tool_calls": [
                    {
                        "id": "wait-call",
                        "function": {
                            "name": "approval_gate",
                            "arguments": '{"expression":"2+2"}',
                        },
                    }
                ]
            }
        ]
    )

    result = await pattern.run(context=context, tools=[tool], llm=llm, runtime=runtime)

    assert result["status"] == "waiting_for_user"
    expected = draft_from_waiting_request(
        pattern.waiting_for_user_request,
        execution_id=context.execution_id,
        step_id=None,
    )
    assert result["clarification_draft"] == expected
    assert result["clarification_draft"].source == "tool_waiting"
    assert (
        result["clarification_draft"].event_id
        == runtime.outbound_messages[-1]["event_id"]
    )
    assert (
        pattern.waiting_for_user_request["event_id"]
        == result["clarification_draft"].event_id
    )


@pytest.mark.asyncio
async def test_empty_message_send_message_reaches_waiting_with_no_draft() -> None:
    """A ``send_message`` call with an empty ``message`` and
    ``expect_response=True`` is schema-valid (the tool only requires the
    ``message`` key to be present, not non-empty) and reaches
    ``waiting_for_user`` with no derivable draft.

    This is the reachable production case documented on
    ``draft_from_waiting_request``: the waiting request carries no message,
    no ``"interactions"`` key, and no ``"requests"`` list, so
    ``clarification_draft`` is ``None`` rather than a typed draft.
    """

    llm = FakeLLM(
        [
            {
                "tool_calls": [
                    {
                        "id": "call-send-empty",
                        "function": {
                            "name": "send_message",
                            "arguments": (
                                '{"message":"","message_type":"question",'
                                '"expect_response":true}'
                            ),
                        },
                    }
                ]
            }
        ]
    )
    pattern = ReActPattern(max_iterations=2)
    context = ExecutionContext(execution_id="exec-empty-message")
    context.add_user_message("Ask")

    result = await pattern.run(context=context, tools=[], llm=llm)

    assert result["status"] == "waiting_for_user"
    assert result["clarification_draft"] is None


@pytest.mark.asyncio
async def test_completed_run_after_resume_carries_no_draft() -> None:
    """Once ``waiting_for_user_request`` is cleared by
    a real user reply, the returned result no longer carries a draft.
    """

    pattern = ReActPattern(max_iterations=2)
    context = ExecutionContext(execution_id="exec-tp1-clear")
    context.add_user_message("Ask")

    first = await pattern.run(context=context, tools=[], llm=_send_message_llm())
    assert first["status"] == "waiting_for_user"

    context.add_user_message("B")
    resumed_pattern = ReActPattern(max_iterations=2)
    resumed_pattern.load_state(pattern.get_state())
    resumed_llm = FakeLLM([{"content": "Continuing with B."}])

    resumed = await resumed_pattern.run(context=context, tools=[], llm=resumed_llm)

    assert resumed["success"] is True
    assert resumed_pattern.waiting_for_user_request is None
    assert "clarification_draft" not in resumed


@pytest.mark.asyncio
async def test_reentry_without_new_message_returns_stable_draft_and_sends_nothing() -> (
    None
):
    """Re-entering with no new user message stays waiting, carries the
    same draft as the first run, and sends zero new outbound messages.
    """

    pattern = ReActPattern(max_iterations=2)
    context = ExecutionContext(execution_id="exec-tp3")
    context.add_user_message("Ask")

    first = await pattern.run(context=context, tools=[], llm=_send_message_llm())
    assert first["status"] == "waiting_for_user"

    resumed_pattern = ReActPattern(max_iterations=2)
    resumed_pattern.load_state(pattern.get_state())
    resumed_runtime = PatternRuntime()
    resumed_llm = FakeLLM([{"content": "Should not run"}])

    resumed = await resumed_pattern.run(
        context=context, tools=[], llm=resumed_llm, runtime=resumed_runtime
    )

    assert resumed["status"] == "waiting_for_user"
    assert resumed_llm.calls == []
    assert len(resumed_runtime.outbound_messages) == 0
    assert resumed["clarification_draft"] == first["clarification_draft"]
    assert resumed["clarification_draft"].event_id
    assert (
        resumed["clarification_draft"].event_id == first["clarification_draft"].event_id
    )


@pytest.mark.asyncio
async def test_marker_survives_trace_serialization_of_a_dirty_interaction_id() -> None:
    """A marker computed before a control-character injection matches
    the marker recomputed after the injected request round-trips through the
    real trace serializer and a JSON encode/decode cycle -- ``_marker_clean``
    and ``clean_string`` share a domain, so persistence cannot desync them.

    The injected id carries both a byte the domain rejects (``\\x01``) and
    one it keeps (``\\t``), so the before/after comparison actually depends
    on the two filters agreeing on ``\\t``. That comparison alone only
    catches a *web*-side narrowing, though: both markers finish with a call
    to this module's own ``_marker_clean``, so a *core*-side narrowing of
    ``_MARKER_KEEP`` would apply identically on both sides of the comparison
    and never show up as a mismatch. The direct equality assertion below
    closes that gap by pinning ``_MARKER_KEEP`` against the literal
    keep-set read out of ``clean_string``'s real source.
    """

    import ast
    import inspect
    import json
    import re

    from xagent.core.agent.checkpoint import TraceCheckpointStore
    from xagent.core.agent.clarification import _MARKER_KEEP
    from xagent.web.api.trace_handlers import DatabaseTraceHandler

    class RecordingTraceBackend:
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

    tool = ResumableTool()
    pattern = ReActPattern(max_iterations=2)
    context = ExecutionContext(execution_id="exec-tp4")
    context.add_user_message("Run the gated action.")
    llm = FakeLLM(
        [
            {
                "tool_calls": [
                    {
                        "id": "wait-call",
                        "function": {
                            "name": "approval_gate",
                            "arguments": '{"expression":"2+2"}',
                        },
                    }
                ]
            }
        ]
    )

    result = await pattern.run(context=context, tools=[tool], llm=llm)
    assert result["status"] == "waiting_for_user"

    assert pattern.waiting_for_user_request is not None
    pattern.waiting_for_user_request["requests"][0]["interaction_id"] = (
        "wait\tcall\x01x"
    )

    marker_before = draft_from_waiting_request(
        pattern.waiting_for_user_request,
        execution_id=context.execution_id,
        step_id=None,
    ).turn_marker

    backend = RecordingTraceBackend()
    store = TraceCheckpointStore(backend)
    await store.checkpoint(
        type="checkpoint",
        label="waiting_for_user",
        execution_id="exec-tp4",
        pattern="ReActPattern",
        pattern_state=pattern.get_state(),
        status="waiting_for_user",
    )
    event_payload = backend.events[0]["data"]
    handler = DatabaseTraceHandler(1)
    serialized = handler._serialize_data_for_json(event_payload)
    assert "_serialization_error" not in serialized

    restored_event_payload = json.loads(json.dumps(serialized))
    restored_state = restored_event_payload["snapshot"]["pattern_state"]

    resumed_pattern = ReActPattern(max_iterations=2)
    resumed_pattern.load_state(restored_state)

    marker_after = draft_from_waiting_request(
        resumed_pattern.waiting_for_user_request,
        execution_id="exec-tp4",
        step_id=None,
    ).turn_marker

    assert marker_after == marker_before

    # Direct pin, independent of the before/after comparison above: read
    # ``clean_string``'s keep-set literal out of the real
    # ``DatabaseTraceHandler._serialize_data_for_json`` source and assert it
    # matches ``_MARKER_KEEP`` character-for-character. A core-side edit
    # that narrows or widens ``_MARKER_KEEP`` shows up here even though it
    # cannot show up in the before/after comparison.
    web_source = inspect.getsource(DatabaseTraceHandler._serialize_data_for_json)
    keep_set_match = re.search(r'char in ("(?:[^"\\]|\\.)*")', web_source)
    assert keep_set_match is not None, "clean_string keep-set literal not found"
    web_keep_set = ast.literal_eval(keep_set_match.group(1))
    assert _MARKER_KEEP == web_keep_set


def test_marker_distinguishes_requests_with_ambiguous_raw_concatenation() -> None:
    """Two request sets whose raw values would collide under naive
    ``"|"``-joining (a ``tool_call_id`` containing ``|`` vs. an
    ``interaction_id`` containing ``|``) still produce distinct markers,
    because every component carries its own length prefix.
    """

    request_a = {
        "kind": "tool_waiting_for_user",
        "requests": [
            {
                "tool_call_id": "a|b",
                "tool_name": "t",
                "interaction_id": "c\nd",
                "message": "m",
                "message_type": "question",
                "interactions": [],
            }
        ],
        "message": "m",
        "message_type": "question",
        "interactions": [],
        "task_text": None,
        "message_count": 1,
    }
    request_b = {
        "kind": "tool_waiting_for_user",
        "requests": [
            {
                "tool_call_id": "a",
                "tool_name": "t",
                "interaction_id": "b|c\nd",
                "message": "m",
                "message_type": "question",
                "interactions": [],
            }
        ],
        "message": "m",
        "message_type": "question",
        "interactions": [],
        "task_text": None,
        "message_count": 1,
    }

    draft_a = draft_from_waiting_request(request_a, execution_id="e", step_id=None)
    draft_b = draft_from_waiting_request(request_b, execution_id="e", step_id=None)

    assert draft_a is not None and draft_b is not None
    # Naive "|"-joining without a length prefix would collapse both raw
    # concatenations to "a|b|c\nd" -- the markers must still differ.
    assert draft_a.turn_marker != draft_b.turn_marker


class RecordingTracer:
    """Fake tracer that records every ``trace_event`` call verbatim.

    Unlike ``RecordingTraceBackend`` in ``test_clarification_draft.py``
    (which sits behind ``TraceCheckpointStore`` and only ever sees
    checkpoint-shaped payloads), this attaches directly to
    ``PatternRuntime`` so it also captures the pattern-start/pattern-end
    trace events that ``ReActPattern.run`` emits on its own -- including
    the ``on_pattern_end`` event, whose ``data["result"]`` is the same
    result dict ``pattern.run`` returns, with the live
    ``ClarificationDraft`` still attached.
    """

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def trace_event(
        self,
        event_type: Any,
        *,
        task_id: str | None = None,
        step_id: str | None = None,
        data: dict[str, Any] | None = None,
        require_persisted: bool = False,
    ) -> str:
        del require_persisted
        self.events.append(
            {
                "event_type": event_type,
                "task_id": task_id,
                "step_id": step_id,
                "data": dict(data or {}),
            }
        )
        return f"event-{len(self.events)}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("handler_factory", "serialize_method"),
    [
        (lambda: DatabaseTraceHandler(1), "_serialize_data_for_json"),
        (lambda: WebSocketTraceHandler(1), "_serialize_data"),
        (lambda: SharedWebSocketTracer(ws=None, task_id=1), "_serialize_data"),
    ],
    ids=[
        "database_trace_handler",
        "websocket_trace_handler",
        "shared_websocket_tracer",
    ],
)
async def test_pattern_end_trace_payload_with_draft_survives_real_serializer(
    handler_factory: Any, serialize_method: str
) -> None:
    """The actual trace-event payload built by ``PatternRuntime.on_pattern_end``
    (``{"pattern": ..., "result": result, "status": ...}``, see
    ``runtime.py``'s ``on_pattern_end`` / ``_emit_pattern_trace``) round-trips
    through the real database and WebSocket trace-handler serializers without
    falling back to the ``_serialization_error`` stub, and ``result``/
    ``status`` survive the round trip with their content intact.

    The companion cell
    ``test_pattern_end_trace_payload_without_to_dict_collapses_to_stub``
    guards the failure direction: it removes ``ClarificationDraft.to_dict``
    and asserts the payload collapses to the ``_serialization_error`` stub.
    """

    tracer = RecordingTracer()
    runtime = PatternRuntime(execution_id="exec-trace-payload", tracer=tracer)
    pattern = ReActPattern(max_iterations=2)
    context = ExecutionContext(execution_id="exec-trace-payload")
    context.add_user_message("Ask")

    result = await pattern.run(
        context=context, tools=[], llm=_send_message_llm(), runtime=runtime
    )

    assert result["status"] == "waiting_for_user"
    assert result["clarification_draft"] is not None

    end_events = [
        event
        for event in tracer.events
        if event["event_type"].action == TraceAction.END and "result" in event["data"]
    ]
    assert len(end_events) == 1
    payload = end_events[0]["data"]
    assert payload["result"] is result

    handler = handler_factory()
    serialized = getattr(handler, serialize_method)(payload)

    assert "_serialization_error" not in serialized
    assert serialized["status"] == "waiting_for_user"
    assert serialized["result"]["status"] == "waiting_for_user"
    assert serialized["result"]["clarification_draft"]["source"] == "send_message"
    assert serialized["result"]["clarification_draft"]["message"] == "Choose A or B"
    assert (
        serialized["result"]["clarification_draft"]["event_id"]
        == runtime.outbound_messages[-1]["event_id"]
    )
    requests = serialized["result"]["clarification_draft"]["requests"]
    assert isinstance(requests, list)
    assert requests and all(isinstance(item, dict) for item in requests)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("handler_factory", "serialize_method"),
    [
        (lambda: DatabaseTraceHandler(1), "_serialize_data_for_json"),
        (lambda: WebSocketTraceHandler(1), "_serialize_data"),
        (lambda: SharedWebSocketTracer(ws=None, task_id=1), "_serialize_data"),
    ],
    ids=[
        "database_trace_handler",
        "websocket_trace_handler",
        "shared_websocket_tracer",
    ],
)
async def test_pattern_end_trace_payload_without_to_dict_collapses_to_stub(
    handler_factory: Any, serialize_method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removing ``ClarificationDraft.to_dict`` collapses the whole trace
    payload to the ``_serialization_error`` stub.

    This is the failure-direction guard for the survival cell above: the
    serializers have no branch for a bare dataclass, so without ``to_dict``
    the draft reaches ``json.dumps`` unconverted, the dump raises, and the
    handler replaces the ENTIRE event payload with the three-key stub.
    """

    tracer = RecordingTracer()
    runtime = PatternRuntime(execution_id="exec-trace-stub", tracer=tracer)
    pattern = ReActPattern(max_iterations=2)
    context = ExecutionContext(execution_id="exec-trace-stub")
    context.add_user_message("Ask")

    result = await pattern.run(
        context=context, tools=[], llm=_send_message_llm(), runtime=runtime
    )
    assert result["status"] == "waiting_for_user"

    end_events = [
        event
        for event in tracer.events
        if event["event_type"].action == TraceAction.END and "result" in event["data"]
    ]
    payload = end_events[0]["data"]

    monkeypatch.delattr(ClarificationDraft, "to_dict", raising=True)

    handler = handler_factory()
    serialized = getattr(handler, serialize_method)(payload)

    assert "_serialization_error" in serialized
    assert "result" not in serialized
