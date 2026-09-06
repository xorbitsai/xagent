"""Contract tests for compact-path token usage accounting (#520) and
text-extraction of chat envelopes (#1714).

Mock-boundary philosophy: the adapter under test runs its real code end to
end; only the provider SDK's transport object is replaced (``AsyncOpenAI``
for the OpenAI family, a duck-typed attribute stub for Zhipu/Gemini whose
adapters read attributes rather than construct SDK types). Everything from
the SDK response object inward -- envelope construction, the top-level
``usage`` stamp, the runtime extractor, the context ledger and the
contextvar ledger -- is production code, so a regression anywhere in that
chain fails these tests. This mirrors test_vector_index_contract.py: the
rest of the adapter suites mock the adapter's own return value, which is
exactly why usage buried in ``raw.usage`` (#520) and repr()ed tool_call
envelopes (#1714) survived unnoticed.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

from xagent.core.agent import ExecutionContext, PatternRuntime, ReActPattern
from xagent.core.agent.utils.context_builder import ContextBuilder, StepExecutionResult
from xagent.core.agent.utils.llm_utils import unwrap_chat_text
from xagent.core.model.chat.basic.deepseek import DeepSeekLLM
from xagent.core.model.chat.basic.gemini import GeminiLLM
from xagent.core.model.chat.basic.openai import OpenAILLM
from xagent.core.model.chat.basic.zhipu import ZhipuLLM
from xagent.core.model.chat.exceptions import (
    LLMEmptyContentError,
    LLMNoTextContentError,
)
from xagent.core.model.chat.token_context import get_token_usage, reset_token_usage

PROMPT_TOKENS = 10
COMPLETION_TOKENS = 5
CACHED_TOKENS = 6
COMPACT_SUMMARY = "summarized tool result"


class FakeLLM:
    """Queue-based fake for the main-pattern LLM (no usage accounting)."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    model_name = "fake-model"

    async def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.responses.pop(0)


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


def _snapshot_ledger() -> tuple[int, int, int]:
    """Value snapshot of the contextvar ledger (the object itself is live)."""
    ledger = get_token_usage()
    return ledger.llm_calls, ledger.input_tokens, ledger.output_tokens


def _openai_completion(content: str) -> ChatCompletion:
    """A real SDK response carrying real ``CompletionUsage``."""
    return ChatCompletion(
        id="compact-contract-completion",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content=content,
                    role="assistant",
                    tool_calls=None,
                ),
            )
        ],
        created=1234567890,
        model="gpt-4o-mini",
        object="chat.completion",
        usage=CompletionUsage(
            prompt_tokens=PROMPT_TOKENS,
            completion_tokens=COMPLETION_TOKENS,
            total_tokens=PROMPT_TOKENS + COMPLETION_TOKENS,
        ),
    )


def _zhipu_response(content: str) -> SimpleNamespace:
    """Duck-typed stand-in for the zai-sdk response: the adapter only reads
    attributes (``choices[0].message.content``, ``usage.prompt_tokens`` ...),
    so a namespace at the transport boundary exercises the real adapter code."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=PROMPT_TOKENS,
            completion_tokens=COMPLETION_TOKENS,
        ),
    )


def _gemini_response(content: str) -> SimpleNamespace:
    """Duck-typed stand-in for the google-genai response, same reasoning as
    the Zhipu stub: attribute reads only (``usage_metadata``,
    ``candidates[0].content.parts[*].text``)."""
    return SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=PROMPT_TOKENS,
            candidates_token_count=COMPLETION_TOKENS,
            cached_content_token_count=0,
        ),
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[SimpleNamespace(text=content, function_call=None)]
                )
            )
        ],
    )


def _react_compact_scenario() -> tuple[TraceEventRecorder, ExecutionContext]:
    """threshold=1 context that forces one LLM compaction on the first turn."""
    tracer = TraceEventRecorder()
    context = ExecutionContext(execution_id="compact-usage-contract")
    context.compact_config.threshold = 1
    context.add_user_message("current request")
    context.add_assistant_message(
        "",
        tool_calls=[
            {"id": "call-1", "type": "function", "function": {"name": "read_file"}}
        ],
    )
    context.add_tool_result("read_file", {"output": "x" * 200}, tool_call_id="call-1")
    return tracer, context


async def _run_compact_with(compact_llm: Any) -> tuple[TraceEventRecorder, Any]:
    tracer, context = _react_compact_scenario()
    runtime = PatternRuntime(tracer=tracer)
    result = await ReActPattern(max_iterations=1).run(
        context=context,
        tools=[],
        llm=FakeLLM([{"content": "done"}]),
        compact_llm=compact_llm,
        runtime=runtime,
    )
    assert result["success"] is True
    return tracer, context


def _assert_compact_usage_accounted(
    tracer: TraceEventRecorder, context: Any, ledger_before: tuple[int, int, int]
) -> None:
    """One compact call must surface exactly once in every accounting view."""
    compact_llm_events = [
        event
        for event in tracer.events
        if event["event_type"] in {"action_start_llm", "action_end_llm"}
        and event["data"].get("purpose") == "context_compaction"
    ]
    assert [event["event_type"] for event in compact_llm_events] == [
        "action_start_llm",
        "action_end_llm",
    ]
    end_data = compact_llm_events[1]["data"]
    assert end_data["input_tokens"] == PROMPT_TOKENS
    assert end_data["output_tokens"] == COMPLETION_TOKENS

    # The execution-context ledger sees exactly one call with exact numbers.
    assert context.get_total_token_usage() == {
        "total": PROMPT_TOKENS + COMPLETION_TOKENS,
        "input": PROMPT_TOKENS,
        "output": COMPLETION_TOKENS,
        "call_count": 1,
    }

    # The contextvar ledger also records exactly one call -- the adapter's own
    # ``add_token_usage`` write; the stamp/extractor path must not double it.
    calls_before, input_before, output_before = ledger_before
    ledger = get_token_usage()
    assert ledger.llm_calls - calls_before == 1
    assert ledger.input_tokens - input_before == PROMPT_TOKENS
    assert ledger.output_tokens - output_before == COMPLETION_TOKENS


@pytest.mark.asyncio
async def test_openai_chat_stamps_top_level_usage(mocker) -> None:
    """The adapter envelope itself carries the stamp, not just the trace:
    this is what lets consumers read usage without opening ``raw``."""
    reset_token_usage()
    llm = OpenAILLM(model_name="gpt-4o-mini", api_key="test-key")
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = _openai_completion(
        COMPACT_SUMMARY
    )
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )

    response = await llm.chat([{"role": "user", "content": "hi"}])

    assert response["type"] == "text"
    assert response["usage"]["prompt_tokens"] == PROMPT_TOKENS
    assert response["usage"]["completion_tokens"] == COMPLETION_TOKENS
    assert response["usage"]["total_tokens"] == PROMPT_TOKENS + COMPLETION_TOKENS


@pytest.mark.asyncio
async def test_openai_tool_call_envelope_stamps_top_level_usage(mocker) -> None:
    reset_token_usage()
    llm = OpenAILLM(model_name="gpt-4o-mini", api_key="test-key")
    mock_client = mocker.AsyncMock()
    completion = ChatCompletion(
        id="compact-contract-tool-call",
        choices=[
            Choice(
                finish_reason="tool_calls",
                index=0,
                message=ChatCompletionMessage(
                    content=None,
                    role="assistant",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": "{}",
                            },
                        }
                    ],
                ),
            )
        ],
        created=1234567890,
        model="gpt-4o-mini",
        object="chat.completion",
        usage=CompletionUsage(
            prompt_tokens=PROMPT_TOKENS,
            completion_tokens=COMPLETION_TOKENS,
            total_tokens=PROMPT_TOKENS + COMPLETION_TOKENS,
        ),
    )
    mock_client.chat.completions.create.return_value = completion
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )

    response = await llm.chat([{"role": "user", "content": "hi"}])

    assert response["type"] == "tool_call"
    assert response["usage"]["prompt_tokens"] == PROMPT_TOKENS
    assert response["usage"]["completion_tokens"] == COMPLETION_TOKENS


@pytest.mark.asyncio
async def test_openai_compact_usage_flows_to_trace_and_context(mocker) -> None:
    reset_token_usage()
    ledger_before = _snapshot_ledger()
    llm = OpenAILLM(model_name="gpt-4o-mini", api_key="test-key")
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = _openai_completion(
        COMPACT_SUMMARY
    )
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )

    tracer, context = await _run_compact_with(llm)

    assert any(
        COMPACT_SUMMARY in (message.content or "") for message in context.messages
    )
    _assert_compact_usage_accounted(tracer, context, ledger_before)


@pytest.mark.asyncio
async def test_deepseek_compact_usage_flows_to_trace_and_context(mocker) -> None:
    reset_token_usage()
    ledger_before = _snapshot_ledger()
    llm = DeepSeekLLM(model_name="deepseek-v4-flash", api_key="test-key")
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = _openai_completion(
        COMPACT_SUMMARY
    )
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )

    tracer, context = await _run_compact_with(llm)

    assert any(
        COMPACT_SUMMARY in (message.content or "") for message in context.messages
    )
    _assert_compact_usage_accounted(tracer, context, ledger_before)


@pytest.mark.asyncio
async def test_zhipu_compact_usage_flows_to_trace_and_context() -> None:
    reset_token_usage()
    ledger_before = _snapshot_ledger()
    llm = ZhipuLLM(model_name="glm-4.5", api_key="test-key")
    llm._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: _zhipu_response(COMPACT_SUMMARY)
            )
        )
    )

    tracer, context = await _run_compact_with(llm)

    assert any(
        COMPACT_SUMMARY in (message.content or "") for message in context.messages
    )
    _assert_compact_usage_accounted(tracer, context, ledger_before)


@pytest.mark.asyncio
async def test_gemini_compact_usage_flows_to_trace_and_context() -> None:
    reset_token_usage()
    ledger_before = _snapshot_ledger()
    llm = GeminiLLM(model_name="gemini-2.5-flash", api_key="test-key")
    llm._client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(
                generate_content=AsyncMock(
                    return_value=_gemini_response(COMPACT_SUMMARY)
                )
            )
        )
    )

    tracer, context = await _run_compact_with(llm)

    assert any(
        COMPACT_SUMMARY in (message.content or "") for message in context.messages
    )
    _assert_compact_usage_accounted(tracer, context, ledger_before)


@pytest.mark.asyncio
async def test_compact_dependency_falls_back_when_compact_llm_returns_tool_call_envelope() -> (
    None
):
    """#1714: a tool_call envelope from the compact model must never be
    repr()ed into the rebuilt context; compaction fails explicitly and the
    existing truncation/error fallbacks engage instead."""
    tool_call_envelope = {
        "type": "tool_call",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }
        ],
    }
    compact_llm = FakeLLM([tool_call_envelope, tool_call_envelope])
    builder = ContextBuilder(
        llm=FakeLLM([]), compact_threshold=1, compact_llm=compact_llm
    )
    dep_result = StepExecutionResult(
        step_id="dep-1",
        messages=[
            {"role": "user", "content": "u" * 100},
            {"role": "assistant", "content": "a" * 100},
        ],
        final_result={},
        agent_name="dep-agent",
    )

    messages = await builder.build_context_for_step(
        step_name="target",
        step_description="do something with the dependency output",
        dependencies=["dep-1"],
        dependency_results={"dep-1": dep_result},
    )

    # Both the individual and the whole-context compaction consulted the
    # compact model and both had to fall back.
    assert len(compact_llm.calls) == 2
    rendered = json.dumps(messages, ensure_ascii=False)
    assert "tool_calls" not in rendered
    # The explicit truncation fallback keeps real history rather than a repr.
    assert any("a" * 100 in m["content"] for m in messages)


@pytest.mark.asyncio
async def test_execution_context_compact_tolerates_tool_call_envelope() -> None:
    """#1714 on the runtime path: a text-less envelope yields an empty summary,
    so the LLM-summary strategy declines and truncation takes over."""
    tracer, context = _react_compact_scenario()
    runtime = PatternRuntime(tracer=tracer)
    compact_llm = FakeLLM(
        [
            {
                "type": "tool_call",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            }
        ]
    )

    result = await ReActPattern(max_iterations=1).run(
        context=context,
        tools=[],
        llm=FakeLLM([{"content": "done"}]),
        compact_llm=compact_llm,
        runtime=runtime,
    )

    assert result["success"] is True
    rendered = json.dumps(
        [m.content for m in context.messages], ensure_ascii=False, default=str
    )
    assert "'tool_calls'" not in rendered
    assert '"tool_calls"' not in rendered


def test_deepseek_violation_rebuild_preserves_usage_stamp() -> None:
    """normalize_deepseek_response rebuilds the envelope on a protocol
    violation; the adapter's top-level usage stamp must survive the rebuild."""
    from xagent.core.model.chat.basic.deepseek_tool_protocol import (
        normalize_deepseek_response,
    )

    usage = {"prompt_tokens": PROMPT_TOKENS, "completion_tokens": COMPLETION_TOKENS}
    response = {
        "type": "text",
        "content": "Sure: <｜｜DSML｜｜tool_calls>",
        "usage": usage,
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
    assert normalized["usage"] == usage


class TestResolveUsagePayload:
    """The runtime extractor reads top-level stamps first, then ``raw``."""

    def setup_method(self) -> None:
        self.runtime = PatternRuntime()

    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            # OpenAI-family envelope without a stamp: usage lives in raw.
            (
                {
                    "type": "text",
                    "content": "x",
                    "raw": {"usage": {"prompt_tokens": 10, "completion_tokens": 5}},
                },
                (10, 5),
            ),
            # Legacy plain-string response: fail open.
            ("plain string", None),
            # Explicitly null usage in raw: fail open, never raise.
            (
                {"type": "text", "content": "x", "raw": {"usage": None}},
                None,
            ),
            # Top-level stamp (backwards compatible, and preferred over raw).
            (
                {
                    "type": "text",
                    "content": "x",
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                    "raw": {"usage": {"prompt_tokens": 10, "completion_tokens": 5}},
                },
                (3, 2),
            ),
            # Gemini-style usage_metadata one level down in raw.
            (
                {
                    "type": "text",
                    "content": "x",
                    "raw": {
                        "usage_metadata": {
                            "prompt_token_count": 7,
                            "candidates_token_count": 4,
                        }
                    },
                },
                (7, 4),
            ),
            # Zhipu tool_call fallback shape: raw is a plain stringified
            # response -- fail open, never raise or probe the string.
            (
                {
                    "type": "tool_call",
                    "tool_calls": [],
                    "raw": "ChatCompletion(choices=[...])",
                },
                None,
            ),
            # Top-level usage_metadata (legacy Gemini-style attribute shape
            # preserved as-is on the response): the original key order applies.
            (
                {
                    "type": "text",
                    "content": "x",
                    "usage_metadata": {
                        "prompt_token_count": 8,
                        "candidates_token_count": 3,
                    },
                },
                (8, 3),
            ),
            # Usage present but all-zero: not a measurement, fail open.
            (
                {
                    "type": "text",
                    "content": "x",
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                },
                None,
            ),
            # All-zero usage in raw behaves the same one level down.
            (
                {
                    "type": "text",
                    "content": "x",
                    "raw": {"usage": {"prompt_tokens": 0, "completion_tokens": 0}},
                },
                None,
            ),
        ],
    )
    def test_extract_token_usage_shapes(self, response: Any, expected: Any) -> None:
        assert self.runtime._extract_token_usage(response) == expected

    def test_extract_cached_tokens_fails_open_on_string_raw(self) -> None:
        response = {
            "type": "tool_call",
            "tool_calls": [],
            "raw": "ChatCompletion(choices=[...])",
        }
        assert self.runtime._extract_cached_tokens(response) == 0

    def test_extract_cached_tokens_reads_raw_usage(self) -> None:
        response = {
            "type": "text",
            "content": "x",
            "raw": {
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "prompt_tokens_details": {"cached_tokens": 6},
                }
            },
        }
        assert self.runtime._extract_cached_tokens(response) == 6
        assert self.runtime._extract_cached_tokens("plain string") == 0


class TestStrictUsageIntCoercion:
    """Usage counters are billing inputs, so coercion must be strict:
    bools, NaN/inf, negatives, non-integral floats, and numeric strings are
    rejected rather than silently truncated or crashed on, and an invalid
    earlier alias must not shadow a valid later candidate."""

    def setup_method(self) -> None:
        self.runtime = PatternRuntime()

    @pytest.mark.parametrize(
        "bad",
        [
            float("nan"),
            float("inf"),
            float("-inf"),
            True,
            False,
            -5,
            10.5,
            "10",
            None,
        ],
    )
    def test_first_int_rejects_invalid_values(self, bad: Any) -> None:
        assert self.runtime._first_int({"prompt_tokens": bad}, ("prompt_tokens",)) == 0

    @pytest.mark.parametrize(
        ("good", "expected"),
        [(10, 10), (0, 0), (10.0, 10), (2**40, 2**40)],
    )
    def test_first_int_accepts_valid_values(self, good: Any, expected: int) -> None:
        assert (
            self.runtime._first_int({"prompt_tokens": good}, ("prompt_tokens",))
            == expected
        )

    def test_invalid_first_alias_falls_through_to_valid_candidate(self) -> None:
        usage = {
            "prompt_tokens": float("nan"),
            "input_tokens": 7,
            "completion_tokens": 3,
        }
        response = {"type": "text", "content": "x", "usage": usage}
        assert self.runtime._extract_token_usage(response) == (7, 3)

    def test_extract_token_usage_rejects_bool_and_negative(self) -> None:
        response = {
            "type": "text",
            "content": "x",
            "usage": {"prompt_tokens": True, "completion_tokens": -5},
        }
        assert self.runtime._extract_token_usage(response) is None

    def test_extract_cached_tokens_rejects_non_finite(self) -> None:
        response = {
            "type": "text",
            "content": "x",
            "usage": {"cached_input_tokens": float("inf")},
        }
        assert self.runtime._extract_cached_tokens(response) == 0


class TestUsageStampCachedTokens:
    """The stamp must carry cache metrics so ``_extract_cached_tokens`` sees
    prompt-cache hits on non-streaming calls (review findings on PR #1787).
    Cached keys are stamped only when non-zero, so a default 0 never shadows
    fallback fields downstream."""

    @pytest.mark.asyncio
    async def test_zhipu_chat_stamp_includes_cached_tokens(self) -> None:
        reset_token_usage()
        llm = ZhipuLLM(model_name="glm-4.5", api_key="test-key")
        response = _zhipu_response("cached reply")
        response.usage.prompt_tokens_details = SimpleNamespace(cached_tokens=4)
        llm._client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: response)
            )
        )

        result = await llm.chat(messages=[{"role": "user", "content": "hi"}])

        assert result["usage"]["cached_input_tokens"] == 4

    @pytest.mark.asyncio
    async def test_zhipu_chat_stamp_omits_zero_cached_tokens(self) -> None:
        reset_token_usage()
        llm = ZhipuLLM(model_name="glm-4.5", api_key="test-key")
        llm._client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: _zhipu_response("plain reply")
                )
            )
        )

        result = await llm.chat(messages=[{"role": "user", "content": "hi"}])

        assert "cached_input_tokens" not in result["usage"]

    @pytest.mark.asyncio
    async def test_gemini_chat_stamp_includes_cached_tokens(self) -> None:
        reset_token_usage()
        llm = GeminiLLM(model_name="gemini-2.5-flash", api_key="test-key")
        response = _gemini_response("cached reply")
        response.usage_metadata.cached_content_token_count = 4
        llm._client = SimpleNamespace(
            aio=SimpleNamespace(
                models=SimpleNamespace(
                    generate_content=AsyncMock(return_value=response)
                )
            )
        )

        result = await llm.chat(messages=[{"role": "user", "content": "hi"}])

        assert result["usage"]["cached_input_tokens"] == 4


class TestUnwrapChatText:
    """``unwrap_chat_text`` distinguishes "no text shape" from "empty text"
    so retry/fallback semantics stay aligned with the adapters' own classes."""

    def test_plain_string_passthrough(self) -> None:
        assert unwrap_chat_text("hello") == "hello"

    def test_envelope_content(self) -> None:
        assert unwrap_chat_text({"type": "text", "content": "hi"}) == "hi"

    @pytest.mark.parametrize("content", ["", "   \n\t"])
    def test_empty_envelope_content_raises_empty_content(self, content: str) -> None:
        with pytest.raises(LLMEmptyContentError):
            unwrap_chat_text({"type": "text", "content": content})

    def test_tool_call_envelope_raises_no_text_content(self) -> None:
        with pytest.raises(LLMNoTextContentError):
            unwrap_chat_text({"type": "tool_call", "tool_calls": []})

    def test_unrecognized_shape_raises_no_text_content(self) -> None:
        with pytest.raises(LLMNoTextContentError):
            unwrap_chat_text(42)

    def test_empty_plain_string_raises_empty_content(self) -> None:
        """An empty legacy plain string is the same transient "no content"
        as an empty envelope -- ``classify_chat_response`` is the single
        source for that distinction."""
        with pytest.raises(LLMEmptyContentError):
            unwrap_chat_text("")


@pytest.mark.asyncio
async def test_openai_compact_cached_tokens_flow_to_trace_event(mocker) -> None:
    """End-to-end cache loop: a real CompletionUsage carrying
    ``prompt_tokens_details.cached_tokens`` must surface as
    ``cached_input_tokens`` on the compact trace event -- the stamp alone is
    not enough, the extractor must read it back."""
    from openai.types.completion_usage import PromptTokensDetails

    reset_token_usage()
    llm = OpenAILLM(model_name="gpt-4o-mini", api_key="test-key")
    completion = _openai_completion(COMPACT_SUMMARY)
    completion.usage = CompletionUsage(
        prompt_tokens=PROMPT_TOKENS,
        completion_tokens=COMPLETION_TOKENS,
        total_tokens=PROMPT_TOKENS + COMPLETION_TOKENS,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=CACHED_TOKENS),
    )
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = completion
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )

    tracer, context = await _run_compact_with(llm)

    compact_end = next(
        event
        for event in tracer.events
        if event["event_type"] == "action_end_llm"
        and event["data"].get("purpose") == "context_compaction"
    )
    assert compact_end["data"]["cached_input_tokens"] == CACHED_TOKENS
    assert compact_end["data"]["input_tokens"] == PROMPT_TOKENS
    assert context.get_total_token_usage()["total"] == (
        PROMPT_TOKENS + COMPLETION_TOKENS
    )


class TestSyntheticUsageFreshnessBaseline:
    """A synthetic (context-compaction) usage record must never become the
    freshness baseline of ``_get_total_tokens``: when an LLM compaction
    declines (e.g. the compact model returns a tool_call envelope) the
    messages are unchanged, so the record's fingerprint still matches and a
    small compact-prompt token count would otherwise be mistaken for the
    live context size -- suppressing the truncation fallback and letting
    oversized history flow to the main model."""

    def test_synthetic_record_does_not_hijack_freshness_baseline(self) -> None:
        context = ExecutionContext()
        context.compact_config.threshold = 100
        context.add_user_message("u" * 2000)  # ~500 est tokens, over threshold
        context.record_llm_usage(
            input_tokens=10,
            output_tokens=5,
            synthetic_purpose="context_compaction",
        )
        # The fingerprint matches (messages unchanged) and 10 < threshold,
        # but the record is synthetic: fall back to the char estimate.
        assert context.estimate_context_tokens() > 100

    def test_real_record_still_serves_as_freshness_baseline(self) -> None:
        context = ExecutionContext()
        context.add_user_message("u" * 2000)
        context.record_llm_usage(input_tokens=42, output_tokens=5)
        assert context.estimate_context_tokens() == 42

    def test_synthetic_record_after_real_record_keeps_real_baseline(self) -> None:
        context = ExecutionContext()
        context.add_user_message("u" * 2000)
        context.record_llm_usage(input_tokens=42, output_tokens=5)
        context.record_llm_usage(
            input_tokens=10,
            output_tokens=5,
            synthetic_purpose="context_compaction",
        )
        assert context.estimate_context_tokens() == 42

    def test_synthetic_purpose_survives_checkpoint_roundtrip(self) -> None:
        context = ExecutionContext()
        context.add_user_message("hi")
        context.record_llm_usage(
            input_tokens=10,
            output_tokens=5,
            synthetic_purpose="context_compaction",
        )
        context.record_llm_usage(input_tokens=7, output_tokens=3)

        restored = ExecutionContext.from_dict(context.to_dict())

        assert [call.synthetic_purpose for call in restored.llm_calls] == [
            "context_compaction",
            None,
        ]

    def test_old_checkpoint_without_synthetic_purpose_defaults_none(self) -> None:
        context = ExecutionContext()
        context.add_user_message("hi")
        context.record_llm_usage(input_tokens=10, output_tokens=5)
        data = context.to_dict()
        for call in data["llm_calls"]:
            call.pop("synthetic_purpose", None)

        restored = ExecutionContext.from_dict(data)

        assert [call.synthetic_purpose for call in restored.llm_calls] == [None]

    @pytest.mark.asyncio
    async def test_failed_llm_compaction_still_truncates_oversized_history(
        self,
    ) -> None:
        """Reviewer scenario: oversized history (chars/4 > threshold) and a
        compact model returning a *stamped* tool_call envelope whose
        prompt_tokens < threshold. LLM compaction declines, and the
        truncation fallback must really truncate; the compact usage is
        recorded exactly once and must not feed the freshness estimate."""
        tracer = TraceEventRecorder()
        context = ExecutionContext(execution_id="compact-freshness-contract")
        context.compact_config.threshold = 100
        context.compact_config.max_messages = 1
        context.add_user_message("current request")
        context.add_assistant_message(
            "",
            tool_calls=[
                {"id": "call-1", "type": "function", "function": {"name": "read_file"}}
            ],
        )
        context.add_tool_result(
            "read_file", {"output": "x" * 2000}, tool_call_id="call-1"
        )
        assert context.estimate_context_tokens() > 100

        compact_llm = FakeLLM(
            [
                {
                    "type": "tool_call",
                    "tool_calls": [
                        {
                            "id": "call_9",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"},
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }
            ]
        )

        result = await ReActPattern(max_iterations=1).run(
            context=context,
            tools=[],
            llm=FakeLLM([{"content": "done"}]),
            compact_llm=compact_llm,
            runtime=PatternRuntime(tracer=tracer),
        )

        assert result["success"] is True
        # Truncation really happened: the declined LLM compaction left the
        # messages untouched, so only the fallback's compact event proves the
        # small compact-prompt count did not hijack the estimate. (Message
        # counts alone cannot show it: the pattern appends afterwards.)
        compact_end = next(
            (
                event
                for event in tracer.events
                if event["event_type"] == "action_end_compact"
            ),
            None,
        )
        assert compact_end is not None, "truncation fallback did not fire"
        assert compact_end["data"]["strategy"] == "truncate"
        assert compact_end["data"]["removed_count"] >= 1
        # The compact call is billed exactly once, marked synthetic.
        assert context.get_total_token_usage() == {
            "total": 15,
            "input": 10,
            "output": 5,
            "call_count": 1,
        }
        assert context.llm_calls[-1].synthetic_purpose == "context_compaction"

    @pytest.mark.asyncio
    async def test_failed_llm_compaction_empty_envelope_still_truncates(self) -> None:
        """Empty-text-envelope variant of the freshness scenario: the empty
        summary declines the LLM compaction and the fallback must truncate
        regardless of the small stamped usage."""
        tracer = TraceEventRecorder()
        context = ExecutionContext(execution_id="compact-freshness-empty")
        context.compact_config.threshold = 100
        context.compact_config.max_messages = 1
        context.add_user_message("current request")
        context.add_assistant_message(
            "",
            tool_calls=[
                {"id": "call-1", "type": "function", "function": {"name": "read_file"}}
            ],
        )
        context.add_tool_result(
            "read_file", {"output": "x" * 2000}, tool_call_id="call-1"
        )
        assert context.estimate_context_tokens() > 100

        compact_llm = FakeLLM(
            [
                {
                    "type": "text",
                    "content": "",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }
            ]
        )

        result = await ReActPattern(max_iterations=1).run(
            context=context,
            tools=[],
            llm=FakeLLM([{"content": "done"}]),
            compact_llm=compact_llm,
            runtime=PatternRuntime(tracer=tracer),
        )

        assert result["success"] is True
        compact_end = next(
            (
                event
                for event in tracer.events
                if event["event_type"] == "action_end_compact"
            ),
            None,
        )
        assert compact_end is not None, "truncation fallback did not fire"
        assert compact_end["data"]["strategy"] == "truncate"
        assert compact_end["data"]["removed_count"] >= 1
        assert context.get_total_token_usage()["call_count"] == 1
        assert context.llm_calls[-1].synthetic_purpose == "context_compaction"


class TestVisionChatUsageStamp:
    """``vision_chat`` envelopes carry the same top-level usage stamp as
    ``chat()`` -- image inputs account tokens identically (#520)."""

    @pytest.mark.asyncio
    async def test_openai_vision_chat_stamps_top_level_usage(self, mocker) -> None:
        llm = OpenAILLM(
            model_name="gpt-4o-mini",
            api_key="test-key",
            abilities=["chat", "vision"],
        )
        mock_client = mocker.AsyncMock()
        mock_client.chat.completions.create.return_value = _openai_completion(
            "A diagram on a whiteboard."
        )
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI",
            return_value=mock_client,
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,ZmFrZV9pbWFnZV9kYXRh"
                        },
                    },
                ],
            }
        ]

        response = await llm.vision_chat(messages)

        assert response["type"] == "text"
        assert response["content"] == "A diagram on a whiteboard."
        assert response["usage"]["prompt_tokens"] == PROMPT_TOKENS
        assert response["usage"]["completion_tokens"] == COMPLETION_TOKENS

    @pytest.mark.asyncio
    async def test_zhipu_vision_chat_stamps_usage_with_cached_tokens(self) -> None:
        llm = ZhipuLLM(model_name="glm-4.5v", api_key="test-key")
        response = _zhipu_response("A flowchart with three boxes.")
        response.usage.prompt_tokens_details = SimpleNamespace(
            cached_tokens=CACHED_TOKENS
        )
        llm._client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: response)
            )
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,ZmFrZV9pbWFnZV9kYXRh"
                        },
                    },
                ],
            }
        ]

        result = await llm.vision_chat(messages=messages)

        assert result["type"] == "text"
        assert result["content"] == "A flowchart with three boxes."
        assert result["usage"] == {
            "prompt_tokens": PROMPT_TOKENS,
            "completion_tokens": COMPLETION_TOKENS,
            "cached_input_tokens": CACHED_TOKENS,
        }


@pytest.mark.asyncio
async def test_openai_reasoning_truncation_branch_stamps_usage(mocker) -> None:
    """The reasoning-content early return (content empty, finish_reason
    "length") is a separate result-construction branch; its envelope must
    carry the stamp too."""
    llm = OpenAILLM(model_name="gpt-4o-mini", api_key="test-key")
    completion = ChatCompletion(
        id="compact-contract-reasoning",
        choices=[
            Choice(
                finish_reason="length",
                index=0,
                message=ChatCompletionMessage(
                    content="",
                    role="assistant",
                    tool_calls=None,
                    reasoning_content="partial reasoning trace",  # type: ignore[call-arg]
                ),
            )
        ],
        created=1234567890,
        model="gpt-4o-mini",
        object="chat.completion",
        usage=CompletionUsage(
            prompt_tokens=PROMPT_TOKENS,
            completion_tokens=COMPLETION_TOKENS,
            total_tokens=PROMPT_TOKENS + COMPLETION_TOKENS,
        ),
    )
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = completion
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )

    response = await llm.chat([{"role": "user", "content": "hi"}])

    assert response["type"] == "text"
    assert response["content"] == "partial reasoning trace"
    assert response["usage"]["prompt_tokens"] == PROMPT_TOKENS
    assert response["usage"]["completion_tokens"] == COMPLETION_TOKENS


@pytest.mark.asyncio
async def test_compact_usage_survives_checkpoint_roundtrip(mocker) -> None:
    """Invariant I5: usage recorded during compaction must survive
    ``to_dict``/``from_dict`` unchanged -- checkpoints are how executions
    resume, and a lossy roundtrip would silently drop billed tokens."""
    reset_token_usage()
    llm = OpenAILLM(model_name="gpt-4o-mini", api_key="test-key")
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = _openai_completion(
        COMPACT_SUMMARY
    )
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )

    _, context = await _run_compact_with(llm)

    expected = {
        "total": PROMPT_TOKENS + COMPLETION_TOKENS,
        "input": PROMPT_TOKENS,
        "output": COMPLETION_TOKENS,
        "call_count": 1,
    }
    assert context.get_total_token_usage() == expected

    restored = ExecutionContext.from_dict(context.to_dict())

    assert restored.get_total_token_usage() == expected


@pytest.mark.asyncio
async def test_compact_estimate_reflects_current_messages_not_stale_record(
    mocker,
) -> None:
    """Invariant I4 + the implicit ordering contract in
    ``PatternRuntime.compact_context_if_needed``: ``on_llm_end`` (which calls
    ``record_llm_usage``) runs *before* ``compact_with_llm_response`` rewrites
    ``context.messages``. If someone swaps them, the record's
    ``prompt_message_count``/``prompt_content_chars`` describe the *rewritten*
    message list, the staleness check in ``_get_total_tokens`` then passes,
    and ``estimate_context_tokens`` silently reports the compact call's
    prompt tokens (~10) instead of estimating the summary text (~hundreds).
    """
    reset_token_usage()
    llm = OpenAILLM(model_name="gpt-4o-mini", api_key="test-key")
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = _openai_completion(
        COMPACT_SUMMARY
    )
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )

    tracer = TraceEventRecorder()
    context = ExecutionContext(execution_id="compact-estimate-contract")
    context.compact_config.threshold = 1
    context.add_user_message("current request")
    context.add_assistant_message(
        "",
        tool_calls=[
            {"id": "call-1", "type": "function", "function": {"name": "read_file"}}
        ],
    )
    # Large enough that the summary is genuinely smaller than the original.
    context.add_tool_result("read_file", {"output": "x" * 4000}, tool_call_id="call-1")
    estimate_before = context.estimate_context_tokens()
    message_count_before = len(context.messages)

    result = await ReActPattern(max_iterations=1).run(
        context=context,
        tools=[],
        llm=FakeLLM([{"content": "done"}]),
        compact_llm=llm,
        runtime=PatternRuntime(tracer=tracer),
    )

    assert result["success"] is True
    # Compaction really shrank the message list.
    compact_end = next(
        event for event in tracer.events if event["event_type"] == "action_end_compact"
    )
    assert compact_end["data"]["original_count"] == message_count_before
    assert compact_end["data"]["final_count"] < message_count_before

    # The post-compaction estimate is a live estimate of the current
    # messages: smaller than before, positive, and far above the compact
    # call's own prompt-token count (which a swapped record would leak).
    estimate_after = context.estimate_context_tokens()
    assert 0 < estimate_after < estimate_before
    assert estimate_after > 3 * PROMPT_TOKENS


@pytest.mark.asyncio
async def test_compact_dependency_falls_back_when_compact_llm_returns_empty_content() -> (
    None
):
    """#1714 empty-content path: an empty text envelope now raises
    ``LLMEmptyContentError`` from ``unwrap_chat_text``; compaction must fail
    explicitly and engage the truncation fallback, with no residue of the
    envelope in the rebuilt context."""
    empty_envelope = {"type": "text", "content": ""}
    compact_llm = FakeLLM([empty_envelope, empty_envelope])
    builder = ContextBuilder(
        llm=FakeLLM([]), compact_threshold=1, compact_llm=compact_llm
    )
    dep_result = StepExecutionResult(
        step_id="dep-1",
        messages=[
            {"role": "user", "content": "u" * 100},
            {"role": "assistant", "content": "a" * 100},
        ],
        final_result={},
        agent_name="dep-agent",
    )

    messages = await builder.build_context_for_step(
        step_name="target",
        step_description="do something with the dependency output",
        dependencies=["dep-1"],
        dependency_results={"dep-1": dep_result},
    )

    # Both compaction levels consulted the compact model and both fell back.
    assert len(compact_llm.calls) == 2
    rendered = json.dumps(messages, ensure_ascii=False)
    assert "'content': ''" not in rendered
    # The explicit truncation fallback keeps real history rather than a repr.
    assert any("a" * 100 in m["content"] for m in messages)


@pytest.mark.asyncio
async def test_zhipu_tool_call_envelope_uses_model_dump_raw() -> None:
    """D8: the real zai-sdk response is a pydantic model, so the production
    ``raw`` always goes through ``response.model_dump()`` -- the stand-in
    must exercise that branch, not the ``str(response)`` fallback."""

    class _ZaiLikeResponse(SimpleNamespace):
        def model_dump(self) -> dict[str, Any]:
            return {"id": "zai-1", "object": "chat.completion"}

    tool_call = SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name="calculator", arguments='{"expression": "2+2"}'),
    )
    response = _ZaiLikeResponse(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None, tool_calls=[tool_call]),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=PROMPT_TOKENS, completion_tokens=5),
    )
    reset_token_usage()
    llm = ZhipuLLM(model_name="glm-4.5", api_key="test-key")
    llm._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: response)
        )
    )

    result = await llm.chat(
        [{"role": "user", "content": "2+2?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "calc",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    assert result["type"] == "tool_call"
    assert result["raw"] == {"id": "zai-1", "object": "chat.completion"}
    assert result["usage"]["prompt_tokens"] == PROMPT_TOKENS
