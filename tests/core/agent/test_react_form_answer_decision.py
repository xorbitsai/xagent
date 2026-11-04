"""``FormAnswerDecision`` and ``_form_answer_decision`` (react.py): the one
per-call decision that both selects whether the form-answer-continuation
text renders (``form_answer_continuation=decision.applied``) and what trace
keys are recorded for that same call (``decision.trace_metadata()``).

The unit-level table below pins the formula directly against
``ExecutionContext``. The integration tests further down drive a real
``ReActPattern.run()`` end to end -- the routing build, the main call-build
site, the iteration-limit delivery, a compaction that removes the answer
before the decision runs, an LLM-summary compaction that keeps it, and the
protocol-retry site -- to confirm the wiring (not just the formula) puts
matching trace keys and message text on the same call, and puts neither on
any other call.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from xagent.config import FORM_ANSWER_CONTINUATION_ENABLED
from xagent.core.agent import ExecutionContext, PatternRuntime, ReActPattern
from xagent.core.agent.context.execution import (
    FORM_ANSWER_CONTINUATION_INSTRUCTION,
    FORM_ANSWER_FRAMING,
)
from xagent.core.agent.pattern.react.react import (
    FormAnswerDecision,
    _form_answer_decision,
)
from xagent.core.model.chat.exceptions import LLMToolProtocolError
from xagent.core.model.chat.types import ChunkType, StreamChunk

from .test_react import FakeTool


@pytest.fixture(autouse=True)
def _switch_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts from the default (switch unset, so on); a test that
    needs it off sets it explicitly."""
    monkeypatch.delenv(FORM_ANSWER_CONTINUATION_ENABLED, raising=False)


def _form_marker() -> dict[str, object]:
    return {
        "question": "What is the shift name?",
        "message_type": "question",
        "form": True,
    }


def _context_with_form_answer() -> ExecutionContext:
    context = ExecutionContext()
    context.add_user_message("Do the thing")
    context.add_user_message(
        "Morning shift", metadata={"response_to_waiting_for_user": _form_marker()}
    )
    return context


def _context_without_form_answer() -> ExecutionContext:
    context = ExecutionContext()
    context.add_user_message("Do the thing")
    return context


def _dag_step_context() -> ExecutionContext:
    context = _context_with_form_answer()
    context.metadata["dag_step_id"] = "step-1"
    return context


def _empty_task_context() -> ExecutionContext:
    # The only user message is itself the answer: top_level_user_request
    # skips it, so execution_text is empty.
    context = ExecutionContext()
    context.add_user_message(
        "Morning shift", metadata={"response_to_waiting_for_user": _form_marker()}
    )
    return context


# ---------------------------------------------------------------------------
# trace_metadata: the dict shape alone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("turn", "applied", "expected"),
    [
        (False, False, {}),
        (True, False, {"form_answer_turn": True}),
        (True, True, {"form_answer_turn": True, "form_answer_continuation": True}),
        # Not reachable from _form_answer_decision (applied is computed as
        # turn and ...), but trace_metadata must not silently invent a turn
        # key it wasn't told about.
        (False, True, {"form_answer_continuation": True}),
    ],
)
def test_trace_metadata_shape(
    turn: bool, applied: bool, expected: dict[str, bool]
) -> None:
    assert FormAnswerDecision(turn=turn, applied=applied).trace_metadata() == expected


# ---------------------------------------------------------------------------
# _form_answer_decision: the formula. How the switch's environment value is
# parsed is pinned at the getter (tests/core/test_config.py) and its effect
# on the decision in test_react_form_answer_gate.py; only "on" (unset) and one
# "off" row appear here.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("build_context", "forced", "switch", "expected"),
    [
        pytest.param(
            _context_with_form_answer,
            False,
            None,
            FormAnswerDecision(turn=True, applied=True),
            id="everything-lines-up",
        ),
        pytest.param(
            _context_with_form_answer,
            True,
            None,
            FormAnswerDecision(turn=True, applied=False),
            id="forced",
        ),
        pytest.param(
            _context_with_form_answer,
            False,
            "false",
            FormAnswerDecision(turn=True, applied=False),
            id="switch-off",
        ),
        pytest.param(
            _context_without_form_answer,
            False,
            None,
            FormAnswerDecision(turn=False, applied=False),
            id="no-form-answer",
        ),
        pytest.param(
            _dag_step_context,
            False,
            None,
            FormAnswerDecision(turn=False, applied=False),
            id="dag-step",
        ),
        pytest.param(
            _empty_task_context,
            False,
            None,
            FormAnswerDecision(turn=False, applied=False),
            id="empty-task",
        ),
    ],
)
def test_decision_formula(
    build_context: Any,
    forced: bool,
    switch: str | None,
    expected: FormAnswerDecision,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dag-step and empty-task rows pin that ``turn`` reads the target,
    not just the latest form answer: both contexts do end in a form answer,
    but neither renders a "Current user request" block, so neither records a
    trace key. The forced and switch-off rows pin that ``turn`` is still
    recorded when the text is not applied."""
    if switch is not None:
        monkeypatch.setenv(FORM_ANSWER_CONTINUATION_ENABLED, switch)
    decision = _form_answer_decision(build_context(), force_final_answer=forced)
    assert decision == expected


# ---------------------------------------------------------------------------
# Integration. Trace keys and rendered text must agree, because both come
# from the one decision computed at the real call's build site.
# ---------------------------------------------------------------------------


class _CaptureTracer:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def trace_event(
        self,
        event_type: Any,
        task_id: Any = None,
        step_id: Any = None,
        data: Any = None,
        parent_id: Any = None,
        **_: Any,
    ) -> str:
        self.events.append({"type": str(event_type), "data": dict(data or {})})
        return "evt"

    def llm_start_events(self) -> list[dict[str, Any]]:
        return [e["data"] for e in self.events if e["type"] == "action_start_llm"]


def _final_answer_call(answer: str, *, outcome: str = "completed") -> dict[str, Any]:
    return {
        "content": "",
        "tool_calls": [
            {
                "id": "call_final",
                "function": {
                    "name": "final_answer",
                    "arguments": json.dumps(
                        {
                            "response_language": "English",
                            "answer": answer,
                            "outcome": outcome,
                        }
                    ),
                },
            }
        ],
    }


class _NamedFakeLLM:
    """A non-streaming FakeLLM with a ``model_name``.

    Implements ``prepare_for_call`` (returning itself) so every test using it
    also exercises -- and can assert on -- the routing-input messages a
    router would have sent, distinct from the real call's own messages.
    """

    def __init__(
        self,
        responses: list[Any],
        *,
        model_name: str = "fake/model",
        context_window: int | None = None,
    ) -> None:
        self.responses = responses
        self.model_name = model_name
        if context_window is not None:
            self.context_window = context_window
        self.calls: list[dict[str, Any]] = []
        self.routing_calls: list[dict[str, Any]] = []

    async def prepare_for_call(
        self, messages: list[dict[str, Any]], *, preferred_input_modalities: Any = ()
    ) -> "_NamedFakeLLM":
        self.routing_calls.append({"messages": messages})
        return self

    async def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.responses.pop(0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "switch",
        "build_context",
        "force_first_turn",
        "expected_keys",
        "expect_text",
    ),
    [
        pytest.param(
            None,
            _context_with_form_answer,
            False,
            {"form_answer_turn": True, "form_answer_continuation": True},
            True,
            id="applied",
        ),
        pytest.param(
            "false",
            _context_with_form_answer,
            False,
            {"form_answer_turn": True},
            False,
            id="switch-off",
        ),
        pytest.param(
            None,
            _context_without_form_answer,
            False,
            {},
            False,
            id="no-form-answer",
        ),
        pytest.param(
            None,
            _context_with_form_answer,
            True,
            {"form_answer_turn": True},
            False,
            id="forced",
        ),
        pytest.param(
            None,
            _dag_step_context,
            False,
            {},
            False,
            id="dag-step",
        ),
    ],
)
async def test_main_path_keys_and_text_come_from_one_decision(
    monkeypatch: pytest.MonkeyPatch,
    switch: str | None,
    build_context: Any,
    force_first_turn: bool,
    expected_keys: dict[str, bool],
    expect_text: bool,
) -> None:
    if switch is not None:
        monkeypatch.setenv(FORM_ANSWER_CONTINUATION_ENABLED, switch)
    llm = _NamedFakeLLM([_final_answer_call("Booked.")])
    context = build_context()
    tracer = _CaptureTracer()
    runtime = PatternRuntime(execution_id="exec-decision-main", tracer=tracer)
    pattern = ReActPattern(max_iterations=2)
    if force_first_turn:
        pattern.force_final_answer_next = True

    result = await pattern.run(context=context, tools=[], llm=llm, runtime=runtime)

    assert result["success"] is True
    llm_events = tracer.llm_start_events()
    assert len(llm_events) == 1
    for key in ("form_answer_turn", "form_answer_continuation"):
        if key in expected_keys:
            assert llm_events[0][key] is expected_keys[key]
        else:
            assert key not in llm_events[0]

    # The routing build never applies, whatever the row -- prove it
    # directly, including on the one row where the real call does apply,
    # rather than leaving it to be inferred from that row's own outcome.
    assert len(llm.routing_calls) == 1
    routing_system_content = llm.routing_calls[0]["messages"][0]["content"]
    assert FORM_ANSWER_CONTINUATION_INSTRUCTION not in routing_system_content

    sent_messages = llm.calls[0]["messages"]
    if expect_text:
        assert FORM_ANSWER_CONTINUATION_INSTRUCTION in sent_messages[0]["content"]
        assert FORM_ANSWER_FRAMING in sent_messages[-1]["content"]
    else:
        assert FORM_ANSWER_CONTINUATION_INSTRUCTION not in sent_messages[0]["content"]
        assert FORM_ANSWER_FRAMING not in sent_messages[-1]["content"]


@pytest.mark.asyncio
async def test_iteration_limit_build_never_applies_even_with_a_form_answer() -> None:
    """The iteration-limit delivery is always forced and never computes a
    decision at all (it does not pass ``form_answer_continuation``, so it gets
    that parameter's default, False); prove it renders neither text even when
    the context otherwise qualifies and the switch is on."""
    context = _context_with_form_answer()
    tool = FakeTool()
    llm = _NamedFakeLLM(
        [
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_work",
                        "function": {
                            "name": "calculator",
                            "arguments": json.dumps({"expression": "2+2"}),
                        },
                    }
                ],
            },
            _final_answer_call(
                "2+2 = 4. The remaining work was not run.", outcome="partial"
            ),
        ],
    )
    pattern = ReActPattern(max_iterations=1)

    result = await pattern.run(
        context=context,
        tools=[tool],
        llm=llm,
        runtime=PatternRuntime(execution_id="exec-decision-iteration-limit"),
    )

    assert result["termination_reason"] == "max_iterations"
    delivery_messages = llm.calls[-1]["messages"]
    assert FORM_ANSWER_CONTINUATION_INSTRUCTION not in delivery_messages[0]["content"]
    # The answer message is followed by the work turn's assistant call and
    # tool result here, so find it by its content rather than by position.
    answer_messages = [
        message
        for message in delivery_messages
        if str(message.get("content") or "").endswith(
            "Execution-enriched message content follows:\nMorning shift"
        )
    ]
    assert len(answer_messages) == 1
    assert FORM_ANSWER_FRAMING not in answer_messages[0]["content"]


@pytest.mark.asyncio
async def test_decision_reflects_state_after_compaction_removes_the_answer() -> None:
    """The decision is computed after ``compact_context_if_needed`` runs, not
    before: when compaction removes the answer message, the real call must
    see turn=False and applied=False, not the stale pre-compaction state. A
    low ``max_messages`` plus a summarization response compaction treats as
    unusable forces the truncate fallback (``_drop_oldest_messages``) to drop
    every message, including the answer.
    """
    context = _context_with_form_answer()
    context.compact_config.threshold = 1
    context.compact_config.max_messages = 0
    llm = _NamedFakeLLM(
        [
            {"content": ""},  # compaction's own summarization call: unusable
            _final_answer_call("Done."),
        ],
        context_window=32_000,
    )
    tracer = _CaptureTracer()
    runtime = PatternRuntime(execution_id="exec-decision-compaction", tracer=tracer)
    pattern = ReActPattern(max_iterations=2)

    result = await pattern.run(context=context, tools=[], llm=llm, runtime=runtime)

    assert result["success"] is True
    # Two llm_start events: compaction's own call, then the real turn.
    real_turn_events = [
        event for event in tracer.llm_start_events() if "purpose" not in event
    ]
    assert len(real_turn_events) == 1
    assert "form_answer_turn" not in real_turn_events[0]
    assert "form_answer_continuation" not in real_turn_events[0]
    # The real call's own messages carry no prior history at all -- proof
    # compaction had already wiped it (including the answer) before this
    # call was built, not merely that the keys happen to be absent.
    sent_messages = llm.calls[-1]["messages"]
    assert [message["role"] for message in sent_messages] == ["system"]
    assert FORM_ANSWER_CONTINUATION_INSTRUCTION not in sent_messages[0]["content"]


@pytest.mark.asyncio
async def test_text_still_applies_after_llm_summary_compaction() -> None:
    """The production compaction path (``compact_with_llm_response``) replaces
    the history with a summary and re-appends the latest visible user message
    -- the same object, which rendering matches with ``is``. When that message
    is the form answer, the real call built after compaction must still carry
    both texts, and the framing must land on that re-appended answer."""
    context = _context_with_form_answer()
    answer_message = context.messages[-1]
    context.compact_config.threshold = 1
    llm = _NamedFakeLLM(
        [
            # compaction's own summarization call: a usable summary
            {"content": "The user asked to do the thing and was asked for a shift."},
            _final_answer_call("Done."),
        ],
        context_window=32_000,
    )
    tracer = _CaptureTracer()
    runtime = PatternRuntime(
        execution_id="exec-decision-summary-compaction", tracer=tracer
    )
    pattern = ReActPattern(max_iterations=2)

    result = await pattern.run(context=context, tools=[], llm=llm, runtime=runtime)

    assert result["success"] is True
    # The summary replaced the history and kept the answer object itself.
    assert context.messages[0].metadata.get("compacted_context") is True
    assert context.messages[1] is answer_message
    real_turn_events = [
        event for event in tracer.llm_start_events() if "purpose" not in event
    ]
    assert len(real_turn_events) == 1
    assert real_turn_events[0]["form_answer_turn"] is True
    assert real_turn_events[0]["form_answer_continuation"] is True
    sent_messages = llm.calls[-1]["messages"]
    assert "Compacted conversation summary:" in "".join(
        str(message.get("content") or "") for message in sent_messages
    )
    assert FORM_ANSWER_CONTINUATION_INSTRUCTION in sent_messages[0]["content"]
    assert FORM_ANSWER_FRAMING in sent_messages[-1]["content"]
    assert str(sent_messages[-1]["content"]).endswith("Morning shift")


def _final_answer_stream_chunks(answer: str) -> list[StreamChunk]:
    return [
        StreamChunk(
            type=ChunkType.TOOL_CALL,
            tool_calls=[
                {
                    "id": "call_final",
                    "function": {
                        "name": "final_answer",
                        "arguments": json.dumps(
                            {
                                "response_language": "English",
                                "answer": answer,
                                "outcome": "completed",
                            }
                        ),
                    },
                }
            ],
        ),
        StreamChunk(type=ChunkType.END),
    ]


class _RetryTriggeringLLM:
    """Fails the tool protocol on the very first turn with ``first_error``,
    forcing ``_retry_tool_protocol_response`` to build and send the retry
    turn, which answers."""

    def __init__(self, first_error: LLMToolProtocolError) -> None:
        self.model_name = "fake/model"
        self.first_error = first_error
        self.stream_calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        raise AssertionError("expected the streaming path")

    async def stream_chat(self, **kwargs: Any):
        self.stream_calls.append(kwargs)
        if len(self.stream_calls) == 1:
            raise self.first_error
        for chunk in _final_answer_stream_chunks("Booked after retry."):
            yield chunk


def _malformed_arguments_error() -> LLMToolProtocolError:
    return LLMToolProtocolError(
        provider="deepseek",
        code="malformed_tool_arguments",
        message="malformed arguments",
        details={
            "original_arguments_preview": '{"answer":',
            "original_arguments_length": 10,
            "repair_status": "skipped_incomplete",
        },
    )


def _unavailable_tool_error() -> LLMToolProtocolError:
    return LLMToolProtocolError(
        provider="deepseek",
        code="unavailable_tool_call",
        message="DeepSeek returned unavailable tool call 'calculator'.",
    )


@pytest.mark.asyncio
async def test_retry_path_carries_matching_keys_and_text_when_applied() -> None:
    llm = _RetryTriggeringLLM(_malformed_arguments_error())
    context = _context_with_form_answer()
    tracer = _CaptureTracer()
    runtime = PatternRuntime(execution_id="exec-decision-retry", tracer=tracer)
    pattern = ReActPattern(max_iterations=2)

    result = await pattern.run(context=context, tools=[], llm=llm, runtime=runtime)

    assert result["success"] is True
    llm_events = tracer.llm_start_events()
    # One event for the main (failed) turn, one for the retry.
    assert len(llm_events) == 2
    retry_event = llm_events[1]
    assert retry_event["form_answer_turn"] is True
    assert retry_event["form_answer_continuation"] is True

    retry_messages = llm.stream_calls[1]["messages"]
    system_content = retry_messages[0]["content"]
    assert FORM_ANSWER_CONTINUATION_INSTRUCTION in system_content
    assert FORM_ANSWER_FRAMING in retry_messages[-1]["content"]


@pytest.mark.asyncio
async def test_forced_main_call_then_unforced_retry_applies_only_on_the_retry() -> None:
    """A forced main call that the provider refuses as an unavailable tool
    call is retried unforced with the full tool set restored (react.py's
    ``restore_full_tool_set``). The retry makes its own decision with its own
    force flag, so the main call records only the turn key and carries no
    text, while the retry records both keys and carries both texts."""
    llm = _RetryTriggeringLLM(_unavailable_tool_error())
    context = _context_with_form_answer()
    tracer = _CaptureTracer()
    runtime = PatternRuntime(execution_id="exec-decision-forced-retry", tracer=tracer)
    pattern = ReActPattern(max_iterations=2)
    pattern.force_final_answer_next = True

    result = await pattern.run(context=context, tools=[], llm=llm, runtime=runtime)

    assert result["success"] is True
    llm_events = tracer.llm_start_events()
    assert len(llm_events) == 2
    main_event, retry_event = llm_events
    assert main_event["form_answer_turn"] is True
    assert "form_answer_continuation" not in main_event
    assert retry_event["recovery_reason"] == "unavailable_tool_call"
    assert retry_event["form_answer_turn"] is True
    assert retry_event["form_answer_continuation"] is True

    main_messages = llm.stream_calls[0]["messages"]
    assert FORM_ANSWER_CONTINUATION_INSTRUCTION not in main_messages[0]["content"]
    assert FORM_ANSWER_FRAMING not in main_messages[-1]["content"]
    retry_messages = llm.stream_calls[1]["messages"]
    assert FORM_ANSWER_CONTINUATION_INSTRUCTION in retry_messages[0]["content"]
    assert FORM_ANSWER_FRAMING in retry_messages[-1]["content"]


@pytest.mark.asyncio
async def test_retry_path_records_only_the_turn_key_when_not_applied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(FORM_ANSWER_CONTINUATION_ENABLED, "false")
    llm = _RetryTriggeringLLM(_malformed_arguments_error())
    context = _context_with_form_answer()
    tracer = _CaptureTracer()
    runtime = PatternRuntime(execution_id="exec-decision-retry-off", tracer=tracer)
    pattern = ReActPattern(max_iterations=2)

    result = await pattern.run(context=context, tools=[], llm=llm, runtime=runtime)

    assert result["success"] is True
    llm_events = tracer.llm_start_events()
    assert len(llm_events) == 2
    retry_event = llm_events[1]
    assert retry_event["form_answer_turn"] is True
    assert "form_answer_continuation" not in retry_event

    retry_messages = llm.stream_calls[1]["messages"]
    assert FORM_ANSWER_CONTINUATION_INSTRUCTION not in retry_messages[0]["content"]
    assert FORM_ANSWER_FRAMING not in retry_messages[-1]["content"]
