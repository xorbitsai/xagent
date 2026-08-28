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
