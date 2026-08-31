"""Contract tests for billed-attempt propagation (``usage_attempts``).

Contract (review N2 on PR #1787):

- ``usage`` on a chat() envelope is the *last* attempt's payload -- it is
  the context-freshness baseline and its semantics do not change.
- ``usage_attempts`` is the ordered list of *all billed* attempts of one
  logical call (including the last), set only when more than one attempt
  was billed. A single-attempt envelope never carries the key.
- Attempts billed before a retryable failure travel on the exception
  (``LLMRetryableError.usage_attempts``); the retry wrapper merges them
  into the eventual envelope, or onto the final exception when every
  attempt fails. Nothing is ever derived by diffing the shared task
  ledger -- the ledger stays the adapter's own ``add_token_usage`` writes,
  so the trace totals and the ledger must agree exactly.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

from xagent.core.agent import ExecutionContext, PatternRuntime
from xagent.core.model.chat.basic.base import BaseLLM
from xagent.core.model.chat.basic.claude import ClaudeLLM
from xagent.core.model.chat.basic.gemini import GeminiLLM
from xagent.core.model.chat.basic.openai import OpenAILLM
from xagent.core.model.chat.basic.zhipu import ZhipuLLM
from xagent.core.model.chat.error import retry_on
from xagent.core.model.chat.exceptions import LLMEmptyContentError
from xagent.core.model.chat.token_context import get_token_usage, reset_token_usage
from xagent.core.retry import create_retry_wrapper
from xagent.core.retry.strategy import ExponentialBackoff

# Distinct per-attempt usages: the first (failed) attempt and the final one.
P1, C1 = 10, 5
P2, C2 = 20, 7

U1 = {"prompt_tokens": P1, "completion_tokens": C1, "total_tokens": P1 + C1}
U2 = {"prompt_tokens": P2, "completion_tokens": C2, "total_tokens": P2 + C2}


class TraceEventRecorder:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def trace_event(
        self,
        event_type: Any,
        *,
        task_id: str | None = None,
        step_id: str | None = None,
        data: dict[str, Any] | None = None,
        **_: Any,
    ) -> str:
        self.events.append(
            {
                "event_type": getattr(event_type, "value", str(event_type)),
                "task_id": task_id,
                "step_id": step_id,
                "data": data or {},
            }
        )
        return str(len(self.events))


def _wrapped(llm: Any, max_retries: int = 3) -> Any:
    """The production retry boundary, with zero-delay backoff for tests."""
    return create_retry_wrapper(
        llm,
        BaseLLM,
        retry_methods={"chat"},
        max_retries=max_retries,
        retry_on=retry_on,
        strategy=ExponentialBackoff(base_delay_ms=0),
    )


def _openai_completion(
    content: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    finish_reason: str = "stop",
    reasoning_content: str | None = None,
) -> ChatCompletion:
    kwargs: dict[str, Any] = {}
    if reasoning_content is not None:
        kwargs["reasoning_content"] = reasoning_content
    return ChatCompletion(
        id="usage-attempts-completion",
        choices=[
            Choice(
                finish_reason=finish_reason,  # type: ignore[arg-type]
                index=0,
                message=ChatCompletionMessage(
                    content=content,
                    role="assistant",
                    tool_calls=None,
                    **kwargs,  # type: ignore[arg-type]
                ),
            )
        ],
        created=1234567890,
        model="gpt-4o-mini",
        object="chat.completion",
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _openai_llm(mocker: Any, side_effects: list[Any]) -> OpenAILLM:
    llm = OpenAILLM(
        model_name="gpt-4o-mini",
        api_key="test-key",
        abilities=["chat", "tool_calling", "thinking_mode"],
    )
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = side_effects
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    return llm


def _attempt_prompts(response: dict[str, Any]) -> list[int]:
    return [attempt["prompt_tokens"] for attempt in response["usage_attempts"]]


@pytest.mark.asyncio
async def test_openai_outer_retry_merges_usage_attempts(mocker: Any) -> None:
    """First attempt bills (empty content -> retryable), second succeeds:
    the envelope keeps the final attempt in ``usage`` and lists both in
    ``usage_attempts``; the task ledger must equal the attempt sum."""
    reset_token_usage()
    llm = _openai_llm(
        mocker,
        [
            _openai_completion("", P1, C1),
            _openai_completion("recovered", P2, C2),
        ],
    )

    response = await _wrapped(llm).chat([{"role": "user", "content": "hi"}])

    assert response["type"] == "text"
    assert response["usage"]["prompt_tokens"] == P2
    assert response["usage"]["completion_tokens"] == C2
    assert _attempt_prompts(response) == [P1, P2]
    ledger = get_token_usage()
    assert ledger.llm_calls == 2
    assert ledger.input_tokens == P1 + P2
    assert ledger.output_tokens == C1 + C2


@pytest.mark.asyncio
async def test_openai_single_attempt_envelope_has_no_usage_attempts(
    mocker: Any,
) -> None:
    reset_token_usage()
    llm = _openai_llm(mocker, [_openai_completion("clean", P2, C2)])

    response = await _wrapped(llm).chat([{"role": "user", "content": "hi"}])

    assert response["usage"]["prompt_tokens"] == P2
    assert "usage_attempts" not in response


@pytest.mark.asyncio
async def test_openai_structured_output_second_request_collects_attempts(
    mocker: Any,
) -> None:
    """The thinking-disabled JSON retry is a second billed request inside
    one chat() call: both attempts must surface on the envelope without any
    outer retry wrapper being involved."""
    reset_token_usage()
    llm = _openai_llm(
        mocker,
        [
            # Thinking active + non-JSON content triggers the degrade retry.
            _openai_completion(
                "not json at all{",
                P1,
                C1,
                reasoning_content="some reasoning trace",
            ),
            _openai_completion('{"ok": true}', P2, C2),
        ],
    )

    response = await llm.chat(
        [{"role": "user", "content": "hi"}],
        response_format={"type": "json_object"},
    )

    assert response["type"] == "text"
    assert response["usage"]["prompt_tokens"] == P2
    assert _attempt_prompts(response) == [P1, P2]
    ledger = get_token_usage()
    assert ledger.llm_calls == 2
    assert ledger.input_tokens == P1 + P2


@pytest.mark.asyncio
async def test_openai_all_attempts_failed_exception_carries_attempts(
    mocker: Any,
) -> None:
    """Every attempt fails: the final exception must carry the full ordered
    attempt list so the runtime can still book the billed tokens."""
    reset_token_usage()
    llm = _openai_llm(
        mocker,
        [
            _openai_completion("", P1, C1),
            _openai_completion("", P2, C2),
        ],
    )

    with pytest.raises(LLMEmptyContentError) as exc_info:
        await _wrapped(llm, max_retries=2).chat([{"role": "user", "content": "hi"}])

    attempts = exc_info.value.usage_attempts
    assert [attempt["prompt_tokens"] for attempt in attempts] == [P1, P2]
    ledger = get_token_usage()
    assert ledger.llm_calls == 2
    assert ledger.input_tokens == P1 + P2


def _claude_response(
    text: str | None, input_tokens: int, output_tokens: int
) -> SimpleNamespace:
    content = [SimpleNamespace(type="text", text=text)] if text is not None else []
    return SimpleNamespace(
        stop_reason="end_turn",
        content=content,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )


@pytest.mark.asyncio
async def test_claude_outer_retry_merges_usage_attempts() -> None:
    reset_token_usage()
    llm = ClaudeLLM(model_name="claude-3-5-sonnet-20241022", api_key="test-key")
    llm._client = SimpleNamespace(
        messages=SimpleNamespace(
            create=AsyncMock(
                side_effect=[
                    _claude_response(None, P1, C1),
                    _claude_response("recovered", P2, C2),
                ]
            )
        )
    )

    response = await _wrapped(llm).chat([{"role": "user", "content": "hi"}])

    assert response["type"] == "text"
    assert response["usage"]["prompt_tokens"] == P2
    assert _attempt_prompts(response) == [P1, P2]
    ledger = get_token_usage()
    assert ledger.llm_calls == 2
    assert ledger.input_tokens == P1 + P2


def _gemini_response(
    text: str | None, prompt_tokens: int, completion_tokens: int
) -> SimpleNamespace:
    parts = [SimpleNamespace(text=text, function_call=None)] if text is not None else []
    return SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt_tokens,
            candidates_token_count=completion_tokens,
            cached_content_token_count=0,
        ),
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts))],
    )


@pytest.mark.asyncio
async def test_gemini_outer_retry_merges_usage_attempts() -> None:
    """Gemini wraps its retryable errors into RuntimeError (``from e``);
    the billed attempts must survive that wrap onto the surfaced exception."""
    reset_token_usage()
    llm = GeminiLLM(model_name="gemini-2.5-flash", api_key="test-key")
    llm._client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(
                generate_content=AsyncMock(
                    side_effect=[
                        _gemini_response(None, P1, C1),
                        _gemini_response("recovered", P2, C2),
                    ]
                )
            )
        )
    )

    response = await _wrapped(llm).chat([{"role": "user", "content": "hi"}])

    assert response["type"] == "text"
    assert response["usage"]["prompt_tokens"] == P2
    assert _attempt_prompts(response) == [P1, P2]
    ledger = get_token_usage()
    assert ledger.llm_calls == 2
    assert ledger.input_tokens == P1 + P2


class TestRuntimeAttemptRecording:
    """The runtime books every billed attempt (final last, so the freshness
    baseline stays the final attempt) and reports billing totals on the
    trace -- the same totals the monitor aggregates from end events."""

    def setup_method(self) -> None:
        self.tracer = TraceEventRecorder()
        self.runtime = PatternRuntime(tracer=self.tracer)
        self.context = ExecutionContext(execution_id="usage-attempts-runtime")
        self.context.add_user_message("hi")

    def _end_event(self) -> dict[str, Any]:
        return next(
            event
            for event in self.tracer.events
            if event["event_type"] == "action_end_llm"
        )

    @pytest.mark.asyncio
    async def test_on_llm_end_records_each_attempt_and_trace_totals(self) -> None:
        envelope = {
            "type": "text",
            "content": "ok",
            "usage": U2,
            "usage_attempts": [U1, U2],
        }

        await self.runtime.on_llm_end(context=self.context, response=envelope)

        records = self.context.llm_calls
        assert [(r.input_tokens, r.output_tokens) for r in records] == [
            (P1, C1),
            (P2, C2),
        ]
        # The final attempt sits at the tail: freshness-baseline safe.
        assert records[-1].input_tokens == P2
        data = self._end_event()["data"]
        assert data["input_tokens"] == P1 + P2
        assert data["output_tokens"] == C1 + C2
        assert data["total_tokens"] == P1 + P2 + C1 + C2
        assert data["llm_attempt_count"] == 2
        assert data["final_prompt_tokens"] == P2
        assert data["final_output_tokens"] == C2

    @pytest.mark.asyncio
    async def test_on_llm_end_single_attempt_shape_unchanged(self) -> None:
        envelope = {"type": "text", "content": "ok", "usage": U2}

        await self.runtime.on_llm_end(context=self.context, response=envelope)

        assert len(self.context.llm_calls) == 1
        data = self._end_event()["data"]
        assert data["input_tokens"] == P2
        assert data["output_tokens"] == C2
        assert data["total_tokens"] == P2 + C2
        assert "llm_attempt_count" not in data
        assert "final_prompt_tokens" not in data

    @pytest.mark.asyncio
    async def test_on_llm_end_marks_all_attempt_records_synthetic(self) -> None:
        envelope = {
            "type": "text",
            "content": "ok",
            "usage": U2,
            "usage_attempts": [U1, U2],
        }

        await self.runtime.on_llm_end(
            context=self.context,
            response=envelope,
            metadata={"purpose": "context_compaction"},
        )

        assert [record.synthetic_purpose for record in self.context.llm_calls] == [
            "context_compaction",
            "context_compaction",
        ]

    @pytest.mark.asyncio
    async def test_on_llm_error_records_attempts_and_trace_tokens(self) -> None:
        error = LLMEmptyContentError("all attempts failed")
        error.usage_attempts = [U1, U2]

        await self.runtime.on_llm_error(context=self.context, error=error)

        records = self.context.llm_calls
        assert [(r.input_tokens, r.output_tokens) for r in records] == [
            (P1, C1),
            (P2, C2),
        ]
        error_event = next(
            event
            for event in self.tracer.events
            if event["event_type"] == "action_error_llm"
        )
        data = error_event["data"]
        assert data["input_tokens"] == P1 + P2
        assert data["output_tokens"] == C1 + C2
        assert data["total_tokens"] == P1 + P2 + C1 + C2
        assert data["llm_attempt_count"] == 2
        assert data["final_prompt_tokens"] == P2
        assert data["final_output_tokens"] == C2

    @pytest.mark.asyncio
    async def test_on_llm_error_without_attempts_keeps_current_shape(self) -> None:
        await self.runtime.on_llm_error(
            context=self.context, error=RuntimeError("plain failure")
        )

        assert self.context.llm_calls == []
        error_event = next(
            event
            for event in self.tracer.events
            if event["event_type"] == "action_error_llm"
        )
        assert "input_tokens" not in error_event["data"]
        assert "llm_attempt_count" not in error_event["data"]


# ---------------------------------------------------------------------------
# Round-2 regressions (review R1-xx on PR #1787)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_generic_error_after_internal_retry_carries_attempts(
    mocker: Any,
) -> None:
    """R1-01: the first (billed, superseded) attempt is followed by a
    thinking-disabled resend that raises BadRequestError. The surfaced
    RuntimeError wraps the SDK error -- it must still carry the known billed
    attempt so error-path accounting does not lose it."""
    import httpx
    import openai as openai_pkg

    reset_token_usage()
    llm = _openai_llm(
        mocker,
        [
            _openai_completion(
                "not json at all{",
                P1,
                C1,
                reasoning_content="some reasoning trace",
            ),
            openai_pkg.BadRequestError(
                "second request rejected",
                response=httpx.Response(
                    400, request=httpx.Request("POST", "https://api.openai.com/x")
                ),
                body=None,
            ),
        ],
    )

    with pytest.raises(RuntimeError) as exc_info:
        await llm.chat(
            [{"role": "user", "content": "hi"}],
            response_format={"type": "json_object"},
        )

    attempts = getattr(exc_info.value, "usage_attempts", None)
    assert attempts is not None
    assert [attempt["prompt_tokens"] for attempt in attempts] == [P1]
    # The ledger booked the first attempt; the error carrier must agree.
    assert get_token_usage().input_tokens == P1

    # And the runtime books it through on_llm_error.
    tracer = TraceEventRecorder()
    runtime = PatternRuntime(tracer=tracer)
    context = ExecutionContext(execution_id="r1-01")
    context.add_user_message("hi")
    await runtime.on_llm_error(context=context, error=exc_info.value)

    assert context.get_total_token_usage() == {
        "total": P1 + C1,
        "input": P1,
        "output": C1,
        "call_count": 1,
    }
    error_event = next(
        event for event in tracer.events if event["event_type"] == "action_error_llm"
    )
    assert error_event["data"]["input_tokens"] == P1


@pytest.mark.asyncio
async def test_retry_wrapper_preserves_singleton_history_when_final_unmetered() -> None:
    """R1-02: one collected billed failure followed by an unmetered success
    must still surface the known attempt -- a one-element history is not
    suppressed, and nothing is promoted into ``usage``."""

    class _FlakyLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, *args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            if self.calls == 1:
                error = LLMEmptyContentError("empty first attempt")
                error.usage_attempts = [U1]
                raise error
            return {"type": "text", "content": "ok"}  # deliberately unmetered

    response = await _wrapped(_FlakyLLM()).chat([{"role": "user", "content": "hi"}])

    assert response.get("usage") is None
    assert response["usage_attempts"] == [U1]

    # And the runtime books that single known attempt from the envelope.
    tracer = TraceEventRecorder()
    runtime = PatternRuntime(tracer=tracer)
    context = ExecutionContext(execution_id="r1-02")
    context.add_user_message("hi")
    await runtime.on_llm_end(context=context, response=response)

    assert context.get_total_token_usage() == {
        "total": P1 + C1,
        "input": P1,
        "output": C1,
        "call_count": 1,
    }
    end_event = next(
        event for event in tracer.events if event["event_type"] == "action_end_llm"
    )
    assert end_event["data"]["llm_attempt_count"] == 1
    assert end_event["data"]["input_tokens"] == P1


def test_deepseek_violation_rebuild_preserves_usage_attempts() -> None:
    """R1-03: the protocol-error rebuild is a response transformation, not a
    new provider request -- it must carry the full ordered attempt list, not
    just the final usage stamp."""
    from xagent.core.model.chat.basic.deepseek_tool_protocol import (
        normalize_deepseek_response,
    )

    response = {
        "type": "text",
        "content": "Sure: <｜｜DSML｜｜tool_calls>",
        "usage": U2,
        "usage_attempts": [U1, U2],
    }
    tool = {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Call final_answer.",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    normalized = normalize_deepseek_response(response, tools=[tool])

    assert normalized["type"] == "tool_protocol_error"
    assert normalized["usage"] == U2
    assert normalized["usage_attempts"] == [U1, U2]


@pytest.mark.asyncio
async def test_deepseek_rebuild_attempts_booked_through_on_llm_end() -> None:
    """R1-03 end-to-end: the rebuilt error envelope books both attempts."""
    from xagent.core.model.chat.basic.deepseek_tool_protocol import (
        normalize_deepseek_response,
    )

    response = {
        "type": "text",
        "content": "Sure: <｜｜DSML｜｜tool_calls>",
        "usage": U2,
        "usage_attempts": [U1, U2],
    }
    tool = {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Call final_answer.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    rebuilt = normalize_deepseek_response(response, tools=[tool])

    tracer = TraceEventRecorder()
    runtime = PatternRuntime(tracer=tracer)
    context = ExecutionContext(execution_id="r1-03")
    context.add_user_message("hi")
    await runtime.on_llm_end(context=context, response=rebuilt)

    assert [
        (record.input_tokens, record.output_tokens) for record in context.llm_calls
    ] == [(P1, C1), (P2, C2)]
    end_event = next(
        event for event in tracer.events if event["event_type"] == "action_end_llm"
    )
    assert end_event["data"]["llm_attempt_count"] == 2
    assert end_event["data"]["input_tokens"] == P1 + P2


@pytest.mark.asyncio
async def test_zhipu_blank_response_error_carries_booked_usage() -> None:
    """R1-05: billing is independent of retryability -- the non-retryable
    blank-response RuntimeError must carry the usage the adapter booked."""
    reset_token_usage()
    llm = ZhipuLLM(model_name="glm-4.5", api_key="test-key")
    llm._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content=None, tool_calls=None),
                            finish_reason="stop",
                        )
                    ],
                    usage=SimpleNamespace(
                        prompt_tokens=P1,
                        completion_tokens=C1,
                    ),
                )
            )
        )
    )

    with pytest.raises(RuntimeError) as exc_info:
        await llm.chat([{"role": "user", "content": "hi"}])

    # The surfaced error is the outer "Zhipu API error" wrap; the payload
    # must survive both the raise and the wrap.
    attempts = getattr(exc_info.value, "usage_attempts", None)
    assert attempts is not None
    assert [attempt["prompt_tokens"] for attempt in attempts] == [P1]
    assert get_token_usage().input_tokens == P1

    tracer = TraceEventRecorder()
    runtime = PatternRuntime(tracer=tracer)
    context = ExecutionContext(execution_id="r1-05")
    context.add_user_message("hi")
    await runtime.on_llm_error(context=context, error=exc_info.value)

    assert context.get_total_token_usage()["total"] == P1 + C1
    error_event = next(
        event for event in tracer.events if event["event_type"] == "action_error_llm"
    )
    assert error_event["data"]["input_tokens"] == P1


class TestAttemptCacheScope:
    """R1-04: cache metrics share the aggregate scope of the token totals --
    summed over every billed attempt on success, and extracted on the
    all-failed error path too."""

    def setup_method(self) -> None:
        self.tracer = TraceEventRecorder()
        self.runtime = PatternRuntime(tracer=self.tracer)
        self.context = ExecutionContext(execution_id="attempt-cache-scope")
        self.context.add_user_message("hi")

    @pytest.mark.asyncio
    async def test_end_trace_sums_cache_over_attempts(self) -> None:
        u1 = {**U1, "cached_input_tokens": 6}
        u2 = {**U2, "prompt_tokens_details": {"cached_tokens": 2}}
        envelope = {
            "type": "text",
            "content": "ok",
            "usage": u2,
            "usage_attempts": [u1, u2],
        }

        await self.runtime.on_llm_end(context=self.context, response=envelope)

        end_event = next(
            event
            for event in self.tracer.events
            if event["event_type"] == "action_end_llm"
        )
        assert end_event["data"]["cached_input_tokens"] == 8
        assert end_event["data"]["llm_attempt_count"] == 2

    @pytest.mark.asyncio
    async def test_error_trace_extracts_cache_from_attempts(self) -> None:
        error = LLMEmptyContentError("all attempts failed")
        error.usage_attempts = [{**U1, "cached_input_tokens": 6}, U2]

        await self.runtime.on_llm_error(context=self.context, error=error)

        error_event = next(
            event
            for event in self.tracer.events
            if event["event_type"] == "action_error_llm"
        )
        assert error_event["data"]["cached_input_tokens"] == 6


class TestMalformedAttemptRows:
    """N3 residual: unusable attempt rows are dropped so the trace attempt
    count stays in lockstep with the booked records, and a wholly unusable
    list falls back to the valid top-level usage."""

    def setup_method(self) -> None:
        self.tracer = TraceEventRecorder()
        self.runtime = PatternRuntime(tracer=self.tracer)
        self.context = ExecutionContext(execution_id="malformed-attempts")
        self.context.add_user_message("hi")

    @pytest.mark.asyncio
    async def test_malformed_row_dropped_everywhere(self) -> None:
        envelope = {
            "type": "text",
            "content": "ok",
            "usage": U2,
            "usage_attempts": [
                {"prompt_tokens": True, "completion_tokens": float("nan")},
                U2,
            ],
        }

        await self.runtime.on_llm_end(context=self.context, response=envelope)

        # Exactly one record booked, and the trace counts exactly one attempt.
        assert len(self.context.llm_calls) == 1
        end_event = next(
            event
            for event in self.tracer.events
            if event["event_type"] == "action_end_llm"
        )
        assert end_event["data"]["llm_attempt_count"] == 1
        assert end_event["data"]["input_tokens"] == P2

    @pytest.mark.asyncio
    async def test_all_invalid_attempts_fall_back_to_top_level_usage(self) -> None:
        envelope = {
            "type": "text",
            "content": "ok",
            "usage": U2,
            "usage_attempts": [{"prompt_tokens": "ten", "completion_tokens": None}],
        }

        await self.runtime.on_llm_end(context=self.context, response=envelope)

        assert self.context.get_total_token_usage()["total"] == P2 + C2
        end_event = next(
            event
            for event in self.tracer.events
            if event["event_type"] == "action_end_llm"
        )
        # Single-attempt historical shape restored: no attempt fields.
        assert "llm_attempt_count" not in end_event["data"]
        assert end_event["data"]["input_tokens"] == P2


@pytest.mark.asyncio
async def test_retry_wrapper_non_retryable_terminal_carries_collected_attempts() -> (
    None
):
    """C1: a non-retryable error after a billed retryable one must carry the
    collected history out of the wrapper -- the ``not retry_on: raise``
    short-circuit is a terminal carrier too."""

    class _FlakyLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, *args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            if self.calls == 1:
                error = LLMEmptyContentError("empty first attempt")
                error.usage_attempts = [U1]
                raise error
            raise RuntimeError("second attempt fails non-retryably")

    with pytest.raises(RuntimeError) as exc_info:
        await _wrapped(_FlakyLLM()).chat([{"role": "user", "content": "hi"}])

    assert not isinstance(exc_info.value, LLMEmptyContentError)
    assert exc_info.value.usage_attempts == [U1]

    # And the runtime books the carried attempt through on_llm_error.
    tracer = TraceEventRecorder()
    runtime = PatternRuntime(tracer=tracer)
    context = ExecutionContext(execution_id="c1-terminal")
    context.add_user_message("hi")
    await runtime.on_llm_error(context=context, error=exc_info.value)

    assert context.get_total_token_usage()["total"] == P1 + C1
    error_event = next(
        event for event in tracer.events if event["event_type"] == "action_error_llm"
    )
    assert error_event["data"]["input_tokens"] == P1
