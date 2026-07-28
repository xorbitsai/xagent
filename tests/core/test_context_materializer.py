from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from xagent.core.agent import (
    Agent,
    AgentRunner,
    ExecutionContext,
    PatternRuntime,
    ReActPattern,
)
from xagent.core.context_materializer import (
    WorkspaceContextReferenceResolver,
    materialize_messages,
)
from xagent.core.context_ref import (
    CONTEXT_REFS_KEY,
    ContextReference,
    ImageDetail,
)
from xagent.core.model.chat.types import ChunkType, StreamChunk


def image_reference(*, detail: ImageDetail = ImageDetail.AUTO) -> ContextReference:
    return ContextReference(
        file_ref={
            "file_id": "image-1",
            "filename": "frame.png",
            "mime_type": "image/png",
        },
        detail=detail,
        text_fallback="A settings dialog",
    )


class Resolver:
    async def resolve_image(self, reference: ContextReference) -> str:
        assert reference.file_id == "image-1"
        return "data:image/png;base64,c2NyZWVuc2hvdA=="


class MissingResolver:
    async def resolve_image(self, reference: ContextReference) -> str:
        raise FileNotFoundError(reference.file_id)


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
async def test_vision_materializer_expands_user_image_without_mutating_input() -> None:
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
    assert "detail" not in result[0]["content"][1]["image_url"]
    assert CONTEXT_REFS_KEY not in result[0]
    assert isinstance(messages[0]["content"], str)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("detail", "expected"),
    [
        (ImageDetail.LOW, "low"),
        (ImageDetail.HIGH, "high"),
        (ImageDetail.ORIGINAL, "high"),
    ],
)
async def test_vision_materializer_preserves_explicit_image_detail(
    detail: ImageDetail,
    expected: str,
) -> None:
    result = await materialize_messages(
        llm=VisionLLM(),
        messages=[
            {
                "role": "user",
                "content": "What is visible?",
                CONTEXT_REFS_KEY: [image_reference(detail=detail).durable_dict()],
            }
        ],
        resolver=Resolver(),
    )

    assert result[0]["content"][1]["image_url"]["detail"] == expected


@pytest.mark.asyncio
async def test_tool_image_follows_complete_tool_result_group() -> None:
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
    assert result[4]["content"] == "continue"


@pytest.mark.asyncio
@pytest.mark.parametrize("resolver", [None, MissingResolver()])
async def test_unresolved_vision_reference_degrades_to_text(resolver: Any) -> None:
    result = await materialize_messages(
        llm=VisionLLM(),
        messages=[
            {
                "role": "user",
                "content": "Inspect this",
                CONTEXT_REFS_KEY: [image_reference().durable_dict()],
            }
        ],
        resolver=resolver,
    )

    assert "file_id=image-1" in result[0]["content"]
    assert "A settings dialog" in result[0]["content"]
    assert "base64" not in result[0]["content"]


@pytest.mark.asyncio
async def test_nonvision_model_receives_file_ref_text_fallback() -> None:
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
    assert "A settings dialog" in result[0]["content"]
    assert "base64" not in result[0]["content"]


@pytest.mark.asyncio
async def test_workspace_resolver_enforces_size_limit_and_caches(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(b"image")

    class Workspace:
        calls = 0

        def resolve_file_id(self, file_id: str) -> Path:
            assert file_id == "image-1"
            self.calls += 1
            return image_path

    workspace = Workspace()
    resolver = WorkspaceContextReferenceResolver(
        workspace,
        cache_size=1,
        max_image_bytes=5,
    )

    first = await resolver.resolve_image(image_reference())
    second = await resolver.resolve_image(image_reference())

    assert first == second
    assert workspace.calls == 1

    image_path.write_bytes(b"too-large")
    resolver.clear()
    with pytest.raises(RuntimeError, match="exceeds"):
        await resolver.resolve_image(image_reference())


@pytest.mark.asyncio
async def test_runtime_materializes_chat_and_streaming_calls() -> None:
    messages = [
        {
            "role": "user",
            "content": "Inspect",
            CONTEXT_REFS_KEY: [image_reference().durable_dict()],
        }
    ]

    chat_llm = VisionLLM()
    runtime = PatternRuntime(context_ref_resolver=Resolver())
    assert await runtime.run_llm_call(chat_llm, messages=messages) == "ok"
    assert chat_llm.messages is not None
    assert isinstance(chat_llm.messages[0]["content"], list)

    stream_llm = StreamingVisionLLM()
    assert (
        await runtime.run_streaming_llm_call(stream_llm, messages=messages) == "visible"
    )
    assert stream_llm.messages is not None
    assert isinstance(stream_llm.messages[0]["content"], list)
    assert isinstance(messages[0]["content"], str)
    assert CONTEXT_REFS_KEY in messages[0]


@pytest.mark.asyncio
async def test_react_materializes_refs_only_at_llm_boundary() -> None:
    class ReactLLM(VisionLLM):
        async def chat(
            self,
            *,
            messages: list[dict[str, Any]],
            **_: Any,
        ) -> dict[str, Any]:
            self.messages = messages
            return {"content": "The dialog is visible.", "done": True}

    context = ExecutionContext(system_prompt="Inspect the image.")
    context.add_user_message("What is visible?", context_refs=[image_reference()])
    runtime = PatternRuntime(context_ref_resolver=Resolver())
    llm = ReactLLM()

    result = await ReActPattern(max_iterations=1).run(
        context=context,
        tools=[],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is True
    assert llm.messages is not None
    provider_user_message = next(
        message for message in llm.messages if message["role"] == "user"
    )
    assert provider_user_message["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    durable_payload = json.dumps(context.to_dict())
    assert "image-1" in durable_payload
    assert "base64" not in durable_payload


@pytest.mark.asyncio
async def test_agent_runner_supplies_workspace_resolver(tmp_path: Path) -> None:
    captured: list[Any] = []

    class Pattern:
        async def run(self, *, runtime: PatternRuntime, **_: Any) -> dict[str, Any]:
            captured.append(runtime.context_ref_resolver)
            return {"success": True, "output": "done"}

    agent = Agent(name="resolver-test", patterns=[Pattern()])
    runner = AgentRunner(
        agent,
        workspace_base_dir=str(tmp_path),
    )

    result = await runner.run("inspect an attachment", execution_id="resolver-test")

    assert result["success"] is True
    assert isinstance(captured[0], WorkspaceContextReferenceResolver)
