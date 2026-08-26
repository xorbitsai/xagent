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
from xagent.core.model.chat.basic.deepseek import DeepSeekLLM
from xagent.core.model.chat.basic.gemini import GeminiLLM
from xagent.core.model.chat.basic.openai import OpenAILLM
from xagent.core.model.chat.basic.zhipu import ZhipuLLM
from xagent.core.model.chat.token_context import get_token_usage, reset_token_usage

PROMPT_TOKENS = 10
COMPLETION_TOKENS = 5
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
        ],
    )
    def test_extract_token_usage_shapes(self, response: Any, expected: Any) -> None:
        assert self.runtime._extract_token_usage(response) == expected

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
