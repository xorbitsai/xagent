from __future__ import annotations

import asyncio
import base64
import json
import threading
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

import xagent.core.context_materializer as context_materializer
from xagent.core.agent import (
    Agent,
    AgentRunner,
    ExecutionContext,
    PatternRuntime,
    ReActPattern,
)
from xagent.core.agent.attachments import build_image_context_references
from xagent.core.context_materializer import (
    ContextReferenceResolutionError,
    WorkspaceContextReferenceResolver,
    materialize_messages,
)
from xagent.core.context_ref import (
    CONTEXT_REFS_KEY,
    ContextReference,
    ImageDetail,
)
from xagent.core.model.chat.types import ChunkType, StreamChunk


def image_reference(
    *,
    detail: ImageDetail = ImageDetail.AUTO,
    file_id: str = "image-1",
) -> ContextReference:
    return ContextReference(
        file_ref={
            "file_id": file_id,
            "filename": f"{file_id}.png",
            "mime_type": "image/png",
        },
        detail=detail,
        text_fallback="A settings dialog",
    )


def uploaded_image_reference(file_id: str) -> ContextReference:
    return build_image_context_references(
        [{"file_id": file_id, "name": f"{file_id}.png", "type": "image/png"}]
    )[0]


def encoded_image_bytes(*, image_format: str = "PNG", color: str = "red") -> bytes:
    output = BytesIO()
    Image.new("RGB", (2, 2), color=color).save(output, format=image_format)
    return output.getvalue()


class Resolver:
    async def resolve_image(self, reference: ContextReference) -> str:
        assert reference.file_id == "image-1"
        return "data:image/png;base64,c2NyZWVuc2hvdA=="


class MissingResolver:
    async def resolve_image(self, reference: ContextReference) -> str:
        raise FileNotFoundError(reference.file_id)


class RecordingResolver:
    def __init__(
        self,
        *,
        missing: set[str] | None = None,
        urls: dict[str, str] | None = None,
    ) -> None:
        self.missing = missing or set()
        self.urls = urls or {}
        self.file_ids: list[str] = []

    async def resolve_image(self, reference: ContextReference) -> str:
        self.file_ids.append(reference.file_id)
        if reference.file_id in self.missing:
            raise FileNotFoundError(reference.file_id)
        return self.urls.get(
            reference.file_id,
            "data:image/png;base64,c2NyZWVuc2hvdA==",
        )


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
@pytest.mark.parametrize(
    ("resolver", "expected_reason"),
    [
        (None, "not available for direct viewing"),
        (MissingResolver(), "image could not be loaded"),
    ],
)
async def test_unresolved_vision_reference_degrades_to_text(
    resolver: Any,
    expected_reason: str,
) -> None:
    reference = build_image_context_references(
        [{"file_id": "image-1", "name": "settings.png", "type": "image/png"}]
    )[0]
    result = await materialize_messages(
        llm=VisionLLM(),
        messages=[
            {
                "role": "user",
                "content": "Inspect this",
                CONTEXT_REFS_KEY: [reference.durable_dict()],
            }
        ],
        resolver=resolver,
    )

    assert "file_id=image-1" in result[0]["content"]
    assert "not available as native visual context" in result[0]["content"]
    assert expected_reason in result[0]["content"]
    assert "base64" not in result[0]["content"]


@pytest.mark.asyncio
async def test_nonvision_model_receives_file_ref_text_fallback() -> None:
    reference = build_image_context_references(
        [{"file_id": "image-1", "name": "settings.png", "type": "image/png"}]
    )[0]
    result = await materialize_messages(
        llm=TextLLM(),
        messages=[
            {
                "role": "user",
                "content": "Inspect this",
                CONTEXT_REFS_KEY: [reference.durable_dict()],
            }
        ],
        resolver=None,
    )

    assert "file_id=image-1" in result[0]["content"]
    assert "not available as native visual context" in result[0]["content"]
    assert "cannot view the image directly" in result[0]["content"]
    assert "base64" not in result[0]["content"]


@pytest.mark.asyncio
async def test_workspace_resolver_enforces_size_limit_and_caches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "frame.png"
    image_bytes = encoded_image_bytes()
    image_path.write_bytes(image_bytes)

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
        max_image_bytes=len(image_bytes),
    )
    read_calls = 0
    original_read_generation = resolver._read_generation

    def counted_read_generation(*args: Any) -> bytes:
        nonlocal read_calls
        read_calls += 1
        return original_read_generation(*args)

    monkeypatch.setattr(resolver, "_read_generation", counted_read_generation)

    first = await resolver.resolve_image(image_reference())
    second = await resolver.resolve_image(image_reference())

    assert first == second
    assert workspace.calls == 2
    assert read_calls == 1

    image_path.write_bytes(image_bytes + b"too-large")
    with pytest.raises(RuntimeError, match="exceeds"):
        await resolver.resolve_image(image_reference())


@pytest.mark.asyncio
async def test_workspace_resolver_rejects_malformed_image_bytes(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "malformed.png"
    image_path.write_bytes(b"not-an-image")

    class Workspace:
        def resolve_file_id(self, file_id: str) -> Path:
            assert file_id == "image-1"
            return image_path

    with pytest.raises(
        ContextReferenceResolutionError,
        match="valid supported image",
    ):
        await WorkspaceContextReferenceResolver(Workspace()).resolve_image(
            image_reference()
        )


@pytest.mark.asyncio
async def test_workspace_resolver_rejects_mismatched_image_mime_type(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "mislabeled.png"
    image_path.write_bytes(encoded_image_bytes(image_format="JPEG"))

    class Workspace:
        def resolve_file_id(self, file_id: str) -> Path:
            assert file_id == "image-1"
            return image_path

    with pytest.raises(
        ContextReferenceResolutionError,
        match="does not match its encoded format",
    ):
        await WorkspaceContextReferenceResolver(Workspace()).resolve_image(
            image_reference()
        )


@pytest.mark.asyncio
async def test_workspace_resolver_uses_canonical_detected_mime_type(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(encoded_image_bytes(image_format="JPEG"))

    class Workspace:
        def resolve_file_id(self, file_id: str) -> Path:
            assert file_id == "image-1"
            return image_path

    reference = ContextReference(
        file_ref={
            "file_id": "image-1",
            "filename": "photo.jpg",
            "mime_type": "image/jpg",
        }
    )
    resolver = WorkspaceContextReferenceResolver(Workspace())
    result = await resolver.resolve_image(reference)

    assert result.startswith("data:image/jpeg;base64,")
    with pytest.raises(
        ContextReferenceResolutionError,
        match="does not match its encoded format",
    ):
        await resolver.resolve_image(image_reference())


@pytest.mark.asyncio
async def test_workspace_resolver_bounds_cache_by_encoded_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    first_bytes = encoded_image_bytes(color="red")
    second_bytes = encoded_image_bytes(color="blue")
    first_path.write_bytes(first_bytes)
    second_path.write_bytes(second_bytes)
    paths = {"image-1": first_path, "image-2": second_path}

    max_encoded_bytes = max(
        len("data:image/png;base64,") + len(base64.b64encode(image_bytes))
        for image_bytes in (first_bytes, second_bytes)
    )

    class Workspace:
        def resolve_file_id(self, file_id: str) -> Path:
            return paths[file_id]

    resolver = WorkspaceContextReferenceResolver(
        Workspace(),
        cache_size=8,
        max_cache_bytes=max_encoded_bytes,
    )
    read_calls = 0
    original_read_generation = resolver._read_generation

    def counted_read_generation(*args: Any) -> bytes:
        nonlocal read_calls
        read_calls += 1
        return original_read_generation(*args)

    monkeypatch.setattr(resolver, "_read_generation", counted_read_generation)

    await resolver.resolve_image(image_reference(file_id="image-1"))
    await resolver.resolve_image(image_reference(file_id="image-2"))
    await resolver.resolve_image(image_reference(file_id="image-1"))

    assert read_calls == 3
    assert len(resolver._cache) == 1
    assert resolver._cache_bytes <= max_encoded_bytes


@pytest.mark.asyncio
async def test_workspace_resolver_counts_concurrent_same_key_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(encoded_image_bytes())
    read_barrier = threading.Barrier(2)

    class Workspace:
        def resolve_file_id(self, file_id: str) -> Path:
            assert file_id == "image-1"
            return image_path

    resolver = WorkspaceContextReferenceResolver(Workspace())
    original_read_generation = resolver._read_generation

    def synchronized_read_generation(*args: Any) -> bytes:
        read_barrier.wait(timeout=5)
        return original_read_generation(*args)

    monkeypatch.setattr(
        resolver,
        "_read_generation",
        synchronized_read_generation,
    )

    first, second = await asyncio.gather(
        resolver.resolve_image(image_reference()),
        resolver.resolve_image(image_reference()),
    )

    assert first == second
    assert len(resolver._cache) == 1
    assert resolver._cache_bytes == sum(
        entry.encoded_bytes for entry in resolver._cache.values()
    )


@pytest.mark.asyncio
async def test_workspace_resolver_invalidates_cache_when_file_id_generation_changes(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    first_path.write_bytes(encoded_image_bytes(color="red"))
    second_path.write_bytes(encoded_image_bytes(color="blue"))

    class Workspace:
        current_path = first_path

        def resolve_file_id(self, file_id: str) -> Path:
            assert file_id == "image-1"
            return self.current_path

    workspace = Workspace()
    resolver = WorkspaceContextReferenceResolver(workspace)

    first = await resolver.resolve_image(image_reference())
    workspace.current_path = second_path
    second = await resolver.resolve_image(image_reference())

    assert first.startswith("data:image/png;base64,")
    assert second.startswith("data:image/png;base64,")
    assert first != second


@pytest.mark.asyncio
async def test_workspace_resolver_default_cache_covers_history_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths: dict[str, Path] = {}
    for index in range(17):
        path = tmp_path / f"image-{index}.png"
        path.write_bytes(encoded_image_bytes(color=f"#{index:06x}"))
        paths[f"image-{index}"] = path

    class Workspace:
        def resolve_file_id(self, file_id: str) -> Path:
            return paths[file_id]

    resolver = WorkspaceContextReferenceResolver(Workspace())
    read_calls = 0
    original_read_generation = resolver._read_generation

    def counted_read_generation(*args: Any) -> bytes:
        nonlocal read_calls
        read_calls += 1
        return original_read_generation(*args)

    monkeypatch.setattr(resolver, "_read_generation", counted_read_generation)
    references = [image_reference(file_id=f"image-{index}") for index in range(17)]

    for reference in (*references, *references):
        await resolver.resolve_image(reference)

    assert read_calls == 17


@pytest.mark.asyncio
async def test_uploaded_refs_degrade_within_materializer_token_budget() -> None:
    class VisionLLMWithContextWindow(VisionLLM):
        context_window = 4_096

    references = build_image_context_references(
        [
            {
                "file_id": f"image-{index}",
                "name": f"image-{index}.png",
                "type": "image/png",
            }
            for index in range(5)
        ]
    )
    resolver = RecordingResolver()
    result = await materialize_messages(
        llm=VisionLLMWithContextWindow(),
        messages=[
            {
                "role": "user",
                "content": "Inspect every image",
                CONTEXT_REFS_KEY: [
                    reference.durable_dict() for reference in references
                ],
            }
        ],
        resolver=resolver,
    )

    assert resolver.file_ids == ["image-4", "image-3", "image-2"]
    assert sum(part["type"] == "image_url" for part in result[0]["content"]) == 3
    assert "file_id=image-0" in result[0]["content"][-1]["text"]
    assert "more images than the model can accept" in result[0]["content"][-1]["text"]


@pytest.mark.asyncio
async def test_materializer_degrades_aggregate_materialized_image_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        context_materializer,
        "_MAX_CONTEXT_IMAGE_BYTES_PER_REQUEST",
        32,
    )

    result = await materialize_messages(
        llm=VisionLLM(),
        messages=[
            {
                "role": "user",
                "content": "Inspect this image",
                CONTEXT_REFS_KEY: [image_reference().durable_dict()],
            }
        ],
        resolver=Resolver(),
    )

    assert isinstance(result[0]["content"], str)
    assert "more image data than the model can accept" in result[0]["content"]


@pytest.mark.asyncio
async def test_materializer_prioritizes_current_and_newest_unique_history() -> None:
    current = uploaded_image_reference("current")
    newest = uploaded_image_reference("history-newest")
    middle = uploaded_image_reference("history-middle")
    oldest = uploaded_image_reference("history-oldest")
    token_budget = sum(
        reference.estimated_tokens() for reference in (current, newest, middle)
    )
    llm = VisionLLM()
    llm.context_window = token_budget * 4
    resolver = RecordingResolver()

    result = await materialize_messages(
        llm=llm,
        messages=[
            {
                "role": "user",
                "content": "oldest",
                CONTEXT_REFS_KEY: [oldest.durable_dict()],
            },
            {
                "role": "user",
                "content": "middle",
                CONTEXT_REFS_KEY: [middle.durable_dict()],
            },
            {
                "role": "user",
                "content": "duplicate current",
                CONTEXT_REFS_KEY: [current.durable_dict()],
            },
            {
                "role": "user",
                "content": "newest",
                CONTEXT_REFS_KEY: [newest.durable_dict()],
            },
            {
                "role": "user",
                "content": "current turn",
                CONTEXT_REFS_KEY: [current.durable_dict()],
            },
        ],
        resolver=resolver,
    )

    assert resolver.file_ids == ["current", "history-newest", "history-middle"]
    assert "file_id=history-oldest" in result[0]["content"]
    assert "same image is included with a more recent message" in result[2]["content"]
    assert isinstance(result[4]["content"], list)


@pytest.mark.asyncio
async def test_materializer_without_user_message_treats_refs_as_history() -> None:
    reference = uploaded_image_reference("tool-image")
    llm = VisionLLM()
    llm.context_window = 4
    resolver = RecordingResolver()
    messages = [
        {
            "role": "tool",
            "content": "captured",
            CONTEXT_REFS_KEY: [reference.durable_dict()],
        }
    ]

    current, historical, duplicates = (
        context_materializer._prioritized_reference_positions(
            messages,
            [(reference,)],
        )
    )
    assert current == []
    assert [position.reference.file_id for position in historical] == ["tool-image"]
    assert duplicates == set()

    result = await materialize_messages(
        llm=llm,
        messages=messages,
        resolver=resolver,
    )

    assert resolver.file_ids == []
    assert "more images than the model can accept" in result[0]["content"]


@pytest.mark.asyncio
async def test_stale_history_does_not_consume_materialization_budget() -> None:
    current = uploaded_image_reference("current")
    stale = uploaded_image_reference("history-stale")
    older = uploaded_image_reference("history-older")
    token_budget = current.estimated_tokens() + older.estimated_tokens()
    llm = VisionLLM()
    llm.context_window = token_budget * 4
    resolver = RecordingResolver(missing={"history-stale"})

    result = await materialize_messages(
        llm=llm,
        messages=[
            {
                "role": "user",
                "content": "older",
                CONTEXT_REFS_KEY: [older.durable_dict()],
            },
            {
                "role": "user",
                "content": "stale",
                CONTEXT_REFS_KEY: [stale.durable_dict()],
            },
            {
                "role": "user",
                "content": "current",
                CONTEXT_REFS_KEY: [current.durable_dict()],
            },
        ],
        resolver=resolver,
    )

    assert resolver.file_ids == ["current", "history-stale", "history-older"]
    assert isinstance(result[0]["content"], list)
    assert "file_id=history-stale" in result[1]["content"]


@pytest.mark.asyncio
async def test_oversized_history_degrades_with_current_byte_priority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = "data:image/png;base64,"
    current_url = prefix + ("c" * 20)
    newest_url = prefix + ("n" * 30)
    older_url = prefix + ("o" * 5)
    monkeypatch.setattr(
        context_materializer,
        "_MAX_CONTEXT_IMAGE_BYTES_PER_REQUEST",
        len(current_url) + len(older_url),
    )
    resolver = RecordingResolver(
        urls={
            "current": current_url,
            "history-newest": newest_url,
            "history-older": older_url,
        }
    )

    result = await materialize_messages(
        llm=VisionLLM(),
        messages=[
            {
                "role": "user",
                "content": "older",
                CONTEXT_REFS_KEY: [
                    uploaded_image_reference("history-older").durable_dict()
                ],
            },
            {
                "role": "user",
                "content": "newest",
                CONTEXT_REFS_KEY: [
                    uploaded_image_reference("history-newest").durable_dict()
                ],
            },
            {
                "role": "user",
                "content": "current",
                CONTEXT_REFS_KEY: [uploaded_image_reference("current").durable_dict()],
            },
        ],
        resolver=resolver,
    )

    assert resolver.file_ids == ["current", "history-newest", "history-older"]
    assert isinstance(result[0]["content"], list)
    assert "file_id=history-newest" in result[1]["content"]
    assert isinstance(result[2]["content"], list)


@pytest.mark.asyncio
async def test_workspace_resolver_prefers_detached_resolution_off_event_loop(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(encoded_image_bytes())
    event_loop_thread = threading.get_ident()

    class Workspace:
        resolver_thread: int | None = None

        def resolve_file_id(self, file_id: str) -> Path:
            raise AssertionError("bound-session resolver must not be used")

        def resolve_file_id_detached(self, file_id: str) -> Path:
            assert file_id == "image-1"
            self.resolver_thread = threading.get_ident()
            return image_path

    workspace = Workspace()
    resolver = WorkspaceContextReferenceResolver(workspace)

    await resolver.resolve_image(image_reference())

    assert workspace.resolver_thread is not None
    assert workspace.resolver_thread != event_loop_thread


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
async def test_llm_compaction_retains_and_materializes_older_context_refs() -> None:
    context = ExecutionContext(system_prompt="Inspect the image.")
    context.add_user_message("What was visible?", context_refs=[image_reference()])
    context.add_assistant_message("I will inspect it.")
    context.add_user_message("Continue from the prior image.")

    result = context.compact_with_llm_response("The image still needs inspection.")

    assert result.compacted is True
    assert context.messages[0].context_refs == (image_reference(),)
    assert context.messages[1].context_refs == ()
    assert "base64" not in json.dumps(context.to_dict())

    materialized = await materialize_messages(
        llm=VisionLLM(),
        messages=context.get_messages_for_llm(),
        resolver=Resolver(),
    )

    continuity = next(
        message for message in materialized if isinstance(message.get("content"), list)
    )
    assert continuity["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )


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
