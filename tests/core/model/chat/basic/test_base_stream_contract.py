"""Contract tests for ``BaseLLM.stream_chat``'s default shape handling.

The default (non-streaming) implementation maps the ``chat()`` response
union onto a single chunk. It must distinguish a text envelope from a
tool-call envelope from a legacy plain string, and must fail explicitly --
never silently emit an empty TOOL_CALL chunk -- on shapes it does not
recognize. Concrete adapters all override ``stream_chat``; this pins the
inherited default via a minimal subclass.
"""

from __future__ import annotations

from typing import Any

import pytest

from xagent.core.model.chat.basic.base import BaseLLM
from xagent.core.model.chat.types import ChunkType, StreamChunk


class _MinimalLLM(BaseLLM):
    """Smallest concrete BaseLLM: inherits the default ``stream_chat``."""

    def __init__(self, result: Any) -> None:
        self._result = result

    @property
    def abilities(self) -> list[str]:
        return ["chat"]

    @property
    def model_name(self) -> str:
        return "minimal-model"

    @property
    def supports_thinking_mode(self) -> bool:
        return False

    async def chat(self, messages: Any, **kwargs: Any) -> Any:
        return self._result


async def _collect(result: Any) -> list[StreamChunk]:
    llm = _MinimalLLM(result)
    return [
        chunk async for chunk in llm.stream_chat([{"role": "user", "content": "hi"}])
    ]


@pytest.mark.asyncio
async def test_text_envelope_yields_token_chunk() -> None:
    chunks = await _collect({"type": "text", "content": "hi", "usage": {}})
    assert len(chunks) == 1
    assert chunks[0].type == ChunkType.TOKEN
    assert chunks[0].content == "hi"
    assert chunks[0].delta == "hi"


@pytest.mark.asyncio
async def test_legacy_plain_string_yields_token_chunk() -> None:
    chunks = await _collect("plain reply")
    assert len(chunks) == 1
    assert chunks[0].type == ChunkType.TOKEN
    assert chunks[0].content == "plain reply"


@pytest.mark.asyncio
async def test_tool_call_envelope_yields_tool_call_chunk() -> None:
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "read_file", "arguments": "{}"},
        }
    ]
    envelope = {"type": "tool_call", "tool_calls": tool_calls}
    chunks = await _collect(envelope)
    assert len(chunks) == 1
    assert chunks[0].type == ChunkType.TOOL_CALL
    assert chunks[0].tool_calls == tool_calls
    assert chunks[0].raw is envelope


@pytest.mark.asyncio
async def test_unknown_dict_shape_yields_error_chunk_not_empty_tool_call() -> None:
    chunks = await _collect({"unexpected": "shape"})
    assert len(chunks) == 1
    assert chunks[0].type == ChunkType.ERROR
    assert chunks[0].content
    # The pre-contract behavior emitted a TOOL_CALL chunk with no tool calls.
    assert not (chunks[0].type == ChunkType.TOOL_CALL and not chunks[0].tool_calls)


@pytest.mark.asyncio
async def test_empty_envelope_yields_error_chunk() -> None:
    chunks = await _collect({"type": "text", "content": ""})
    assert len(chunks) == 1
    assert chunks[0].type == ChunkType.ERROR


@pytest.mark.asyncio
async def test_none_result_yields_error_chunk() -> None:
    chunks = await _collect(None)
    assert len(chunks) == 1
    assert chunks[0].type == ChunkType.ERROR


@pytest.mark.asyncio
async def test_usage_bearing_text_envelope_emits_usage_chunk() -> None:
    """R1-06: the chat-to-stream adaptation must keep the billing metadata
    the envelope carries -- raw on the content chunk and a USAGE chunk."""
    envelope = {
        "type": "text",
        "content": "hi",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    chunks = await _collect(envelope)
    assert [chunk.type for chunk in chunks] == [ChunkType.TOKEN, ChunkType.USAGE]
    assert chunks[0].raw is envelope
    assert chunks[1].usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    assert chunks[1].raw is envelope


@pytest.mark.asyncio
async def test_usage_bearing_tool_call_envelope_emits_usage_chunk() -> None:
    envelope = {
        "type": "tool_call",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    chunks = await _collect(envelope)
    assert [chunk.type for chunk in chunks] == [ChunkType.TOOL_CALL, ChunkType.USAGE]
    assert chunks[1].usage == {"prompt_tokens": 10, "completion_tokens": 5}


@pytest.mark.asyncio
async def test_usage_chunk_feeds_runtime_extraction() -> None:
    """R1-06 through the runtime's own chunk reader: the USAGE chunk the
    default emits is what ``_chunk_usage``/reconstruction consumes."""
    from xagent.core.agent import PatternRuntime

    envelope = {
        "type": "text",
        "content": "hi",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    chunks = await _collect(envelope)
    runtime = PatternRuntime()
    usage_payload: dict[str, Any] = {}
    for chunk in chunks:
        usage_payload.update(runtime._chunk_usage(chunk))
    assert usage_payload["prompt_tokens"] == 10
    assert usage_payload["completion_tokens"] == 5
