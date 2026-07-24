from __future__ import annotations

from typing import Any

import pytest

from xagent.core.agent.runtime import PatternRuntime
from xagent.core.computer.materializer import materialize_messages
from xagent.core.context_ref import (
    CONTEXT_REFS_KEY,
    ContextReference,
    ContextReferencePurpose,
)
from xagent.core.model.chat.types import ChunkType, StreamChunk


def image_reference() -> ContextReference:
    return ContextReference(
        file_ref={
            "file_id": "image-1",
            "filename": "frame.png",
            "mime_type": "image/png",
        },
        purpose=ContextReferencePurpose.OBSERVATION,
        frame_id="frame-1",
        text_fallback="Screen containing a settings dialog",
    )


class Resolver:
    async def resolve_image(self, reference: ContextReference) -> str:
        assert reference.file_id == "image-1"
        return "data:image/png;base64,c2NyZWVuc2hvdA=="


class VisionLLM:
    abilities = ["chat", "vision"]

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] | None = None

    async def chat(self, *, messages: list[dict[str, Any]], **_: Any) -> str:
        self.messages = messages
        return "ok"


class TextLLM:
    abilities = ["chat"]


class StreamingVisionLLM(VisionLLM):
    async def stream_chat(
        self,
        *,
        messages: list[dict[str, Any]],
        **_: Any,
    ) -> Any:
        self.messages = messages
        yield StreamChunk(type=ChunkType.TOKEN, delta="visible")
        yield StreamChunk(type=ChunkType.END)


@pytest.mark.asyncio
async def test_vision_materializer_expands_user_image() -> None:
    messages = [
        {
            "role": "user",
            "content": "What is visible?",
            CONTEXT_REFS_KEY: [image_reference().durable_dict()],
        }
    ]

    result = await materialize_messages(
        llm=VisionLLM(),
        messages=messages,
        resolver=Resolver(),
    )

    assert result[0]["content"][0] == {
        "type": "text",
        "text": "What is visible?",
    }
    assert result[0]["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    assert CONTEXT_REFS_KEY not in result[0]
    assert isinstance(messages[0]["content"], str)


@pytest.mark.asyncio
async def test_tool_observation_image_follows_complete_tool_group() -> None:
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call-1", "type": "function", "function": {}},
                {"id": "call-2", "type": "function", "function": {}},
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "captured",
            CONTEXT_REFS_KEY: [image_reference().durable_dict()],
        },
        {"role": "tool", "tool_call_id": "call-2", "content": "metadata"},
        {"role": "user", "content": "continue"},
    ]

    result = await materialize_messages(
        llm=VisionLLM(),
        messages=messages,
        resolver=Resolver(),
    )

    assert [message["role"] for message in result] == [
        "assistant",
        "tool",
        "tool",
        "user",
        "user",
    ]
    assert isinstance(result[3]["content"], list)
    assert result[3]["content"][0] == {
        "type": "text",
        "text": "Image context for the preceding message.",
    }
    assert "captured" not in str(result[3]["content"])
    assert result[4]["content"] == "continue"


@pytest.mark.asyncio
async def test_nonvision_materializer_uses_safe_text_fallback() -> None:
    result = await materialize_messages(
        llm=TextLLM(),
        messages=[
            {
                "role": "user",
                "content": "Inspect this",
                CONTEXT_REFS_KEY: [image_reference().durable_dict()],
            }
        ],
        resolver=None,
    )

    assert "file_id=image-1" in result[0]["content"]
    assert "Screen containing a settings dialog" in result[0]["content"]
    assert "base64" not in result[0]["content"]


@pytest.mark.asyncio
async def test_runtime_materializes_without_mutating_durable_messages() -> None:
    llm = VisionLLM()
    runtime = PatternRuntime(context_ref_resolver=Resolver())
    messages = [
        {
            "role": "user",
            "content": "Inspect",
            CONTEXT_REFS_KEY: [image_reference().durable_dict()],
        }
    ]

    result = await runtime.run_llm_call(llm, messages=messages)

    assert result == "ok"
    assert llm.messages is not None
    assert isinstance(llm.messages[0]["content"], list)
    assert isinstance(messages[0]["content"], str)
    assert CONTEXT_REFS_KEY in messages[0]


@pytest.mark.asyncio
async def test_streaming_runtime_uses_same_materialization_boundary() -> None:
    llm = StreamingVisionLLM()
    runtime = PatternRuntime(context_ref_resolver=Resolver())
    messages = [
        {
            "role": "user",
            "content": "Inspect",
            CONTEXT_REFS_KEY: [image_reference().durable_dict()],
        }
    ]

    result = await runtime.run_streaming_llm_call(llm, messages=messages)

    assert result == "visible"
    assert llm.messages is not None
    assert isinstance(llm.messages[0]["content"], list)
    assert CONTEXT_REFS_KEY in messages[0]
