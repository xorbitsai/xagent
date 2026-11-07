"""Streamed usage must carry the provider's cached-token count to the trace.

The trace's ``llm_call_end`` reads ``cached_input_tokens`` from the usage the
stream hands back. The adapter recorded the count in the token context but left
it out of that usage, so the trace never showed a cache hit. The count has two
places to get lost: the usage-only chunk, and the fallback that reads usage off
the last delta when a provider folds it into that delta.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from xagent.core.agent.context.execution import ExecutionContext
from xagent.core.agent.runtime import PatternRuntime
from xagent.core.model.chat.basic.openai import OpenAILLM
from xagent.core.model.chat.types import ChunkType, StreamChunk


def _usage(**extra: Any) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=1000, completion_tokens=20, total_tokens=1020, **extra
    )


def _parse(llm: OpenAILLM, usage: SimpleNamespace) -> StreamChunk:
    chunk = SimpleNamespace(choices=[], usage=usage)
    (parsed,) = llm._parse_stream_chunks(chunk, {})
    assert parsed.type is ChunkType.USAGE
    return parsed


class _CaptureTracer:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events

    async def trace_event(
        self,
        event_type: Any,
        task_id: Any = None,
        step_id: Any = None,
        data: Any = None,
        parent_id: Any = None,
    ) -> str:
        self._events.append({"data": dict(data or {})})
        return "evt"


@pytest.mark.parametrize(
    "extra",
    [
        {"prompt_tokens_details": SimpleNamespace(cached_tokens=400)},
        {"prompt_cache_hit_tokens": 400},
    ],
)
@pytest.mark.asyncio
async def test_usage_chunk_cache_hits_reach_the_llm_call_end_trace(
    openai_llm_config: dict[str, Any], extra: dict[str, Any]
) -> None:
    llm = OpenAILLM(**openai_llm_config)
    usage_chunk = _parse(llm, _usage(**extra))
    assert usage_chunk.usage["cached_input_tokens"] == 400

    class Streaming:
        async def stream_chat(self, **_: Any) -> Any:
            yield StreamChunk(type=ChunkType.TOKEN, delta="hello")
            # The adapter's usage, without its raw SDK chunk.
            yield StreamChunk(type=ChunkType.USAGE, usage=usage_chunk.usage)
            yield StreamChunk(type=ChunkType.END)

    events: list[dict[str, Any]] = []
    runtime = PatternRuntime(execution_id="cached", tracer=_CaptureTracer(events))
    result = await runtime.run_streaming_llm_call(
        Streaming(), messages=[], on_chunk=lambda _chunk: None
    )
    await runtime.on_llm_end(
        context=ExecutionContext(execution_id="cached"), response=result
    )

    assert events[-1]["data"]["cached_input_tokens"] == 400
    assert events[-1]["data"]["input_tokens"] == 1000


def test_usage_chunk_without_a_cache_report_has_no_cached_field(
    openai_llm_config: dict[str, Any],
) -> None:
    llm = OpenAILLM(**openai_llm_config)

    assert "cached_input_tokens" not in _parse(llm, _usage()).usage


@pytest.mark.parametrize(
    "usage_extra, expected",
    [
        ({"prompt_tokens_details": {"cached_tokens": 400}}, 400),
        ({"prompt_cache_hit_tokens": 400}, 400),
        ({}, None),
    ],
)
@pytest.mark.asyncio
async def test_usage_folded_into_the_last_delta_keeps_cache_hits(
    openai_llm_config: dict[str, Any],
    usage_extra: dict[str, Any],
    expected: int | None,
) -> None:
    """Some providers send usage on the final delta, with no usage-only chunk."""

    async def sdk_stream() -> Any:
        yield ChatCompletionChunk(
            id="folded-usage",
            created=0,
            model="synthetic-model",
            object="chat.completion.chunk",
            choices=[
                {
                    "index": 0,
                    "delta": {"content": "hello"},
                    "finish_reason": "stop",
                }
            ],
            usage={
                "prompt_tokens": 1000,
                "completion_tokens": 20,
                "total_tokens": 1020,
                **usage_extra,
            },
        )

    client = AsyncMock()
    client.chat.completions.create.return_value = sdk_stream()
    with patch("xagent.core.model.chat.basic.openai.AsyncOpenAI", return_value=client):
        llm = OpenAILLM(**openai_llm_config)
        chunks = [
            chunk
            async for chunk in llm.stream_chat(
                messages=[{"role": "user", "content": "hi"}]
            )
        ]

    (usage_chunk,) = [c for c in chunks if c.type is ChunkType.USAGE]
    assert usage_chunk.usage.get("cached_input_tokens") == expected
