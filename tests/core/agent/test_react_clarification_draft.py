from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from xagent.core.agent import ExecutionContext, PatternRuntime, ReActPattern
from xagent.core.agent.clarification import draft_from_waiting_request


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
    context = ExecutionContext(execution_id="exec-tp1")
    context.add_user_message("Ask")

    result = await pattern.run(
        context=context, tools=tools_factory(), llm=llm_factory()
    )

    assert result["status"] == "waiting_for_user"
    expected = draft_from_waiting_request(
        pattern.waiting_for_user_request,
        execution_id=context.execution_id,
        step_id=None,
    )
    assert result["clarification_draft"] == expected
    assert result["clarification_draft"].source == expected_source


@pytest.mark.asyncio
async def test_waiting_return_via_tool_carries_tool_waiting_draft() -> None:
    """The multi-tool waiting point also carries
    a draft equal to the one recomputed from the same request.
    """

    tool = ResumableTool()
    pattern = ReActPattern(max_iterations=2)
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

    result = await pattern.run(context=context, tools=[tool], llm=llm)

    assert result["status"] == "waiting_for_user"
    expected = draft_from_waiting_request(
        pattern.waiting_for_user_request,
        execution_id=context.execution_id,
        step_id=None,
    )
    assert result["clarification_draft"] == expected
    assert result["clarification_draft"].source == "tool_waiting"


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


@pytest.mark.asyncio
async def test_marker_survives_trace_serialization_of_a_dirty_interaction_id() -> None:
    """A marker computed before a control-character injection matches
    the marker recomputed after the injected request round-trips through the
    real trace serializer and a JSON encode/decode cycle -- ``_marker_clean``
    and ``clean_string`` share a domain, so persistence cannot desync them.
    """

    import json

    from xagent.core.agent.checkpoint import TraceCheckpointStore
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
    pattern.waiting_for_user_request["requests"][0]["interaction_id"] += "\x01"

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
