"""Screenshots must not accumulate unbounded cost in the conversation.

Every observation kept alive is re-encoded and re-uploaded on every subsequent
model call, so only the newest frames stay images and superseded observations
shrink to a summary.
"""

from __future__ import annotations

from typing import Any

import pytest

from xagent.core.agent.context import ExecutionContext
from xagent.core.computer.materializer import materialize_messages
from xagent.core.context_ref import (
    CONTEXT_REFS_KEY,
    SUPERSEDES_SCOPE_KEY,
    ContextReference,
    ContextReferencePurpose,
    ImageDetail,
)


class VisionLLM:
    abilities = ("vision",)

    def has_ability(self, name: str) -> bool:
        return name in self.abilities


class RecordingResolver:
    def __init__(self) -> None:
        self.resolved: list[str] = []

    async def resolve_image(self, reference: ContextReference) -> str:
        self.resolved.append(reference.file_id)
        return f"data:image/png;base64,{reference.file_id}"


class MissingFileResolver:
    async def resolve_image(self, reference: ContextReference) -> str:
        raise FileNotFoundError(reference.file_id)


def _observation_ref(index: int) -> dict[str, Any]:
    return ContextReference(
        file_ref={
            "file_id": f"image-{index}",
            "filename": f"frame-{index}.png",
            "mime_type": "image/png",
        },
        purpose=ContextReferencePurpose.OBSERVATION,
        frame_id=f"frame-{index}",
        text_fallback=f"Screenshot of frame {index}.",
        metadata={"sha256": f"digest-{index}"},
    ).durable_dict()


def _observation_messages(count: int) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for index in range(1, count + 1):
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": f"call-{index}", "name": "computer"}],
            }
        )
        messages.append(
            {
                "role": "tool",
                "content": f"observation {index}",
                "tool_call_id": f"call-{index}",
                CONTEXT_REFS_KEY: [_observation_ref(index)],
            }
        )
    return messages


@pytest.mark.asyncio
async def test_only_the_newest_frames_are_sent_as_images() -> None:
    resolver = RecordingResolver()

    await materialize_messages(
        llm=VisionLLM(),
        messages=_observation_messages(5),
        resolver=resolver,
        max_live_frames=2,
    )

    assert resolver.resolved == ["image-4", "image-5"]


@pytest.mark.asyncio
async def test_superseded_frames_keep_their_text_description() -> None:
    materialized = await materialize_messages(
        llm=VisionLLM(),
        messages=_observation_messages(3),
        resolver=RecordingResolver(),
        max_live_frames=1,
    )

    stale = next(
        message for message in materialized if message.get("tool_call_id") == "call-1"
    )
    assert "Screenshot of frame 1." in stale["content"]
    assert isinstance(stale["content"], str)


@pytest.mark.asyncio
async def test_a_pruned_screenshot_degrades_instead_of_failing() -> None:
    """Retention and cross-process resumes make missing frames normal."""
    materialized = await materialize_messages(
        llm=VisionLLM(),
        messages=_observation_messages(1),
        resolver=MissingFileResolver(),
        max_live_frames=2,
    )

    contents = [str(message.get("content") or "") for message in materialized]
    assert any("Screenshot of frame 1." in content for content in contents)
    assert all(isinstance(message.get("content"), str) for message in materialized)


def test_full_viewport_screenshots_are_not_costed_like_thumbnails() -> None:
    reference = ContextReference(
        file_ref={
            "file_id": "image-1",
            "filename": "frame-1.png",
            "mime_type": "image/png",
        },
        purpose=ContextReferencePurpose.OBSERVATION,
        frame_id="frame-1",
        metadata={"viewport": {"width": 1280, "height": 720}},
    )
    low_detail = reference.model_copy(update={"detail": ImageDetail.LOW})

    # 1280x720 scales to 1365x768 -> a 3x2 tile grid.
    assert reference.estimated_tokens() > 1_000
    assert low_detail.estimated_tokens() < 200


def test_a_new_observation_shrinks_the_previous_one() -> None:
    context = ExecutionContext(execution_id="exec-1")

    for index in (1, 2):
        context.add_tool_result(
            "computer",
            {
                "success": True,
                "frame_id": f"frame-{index}",
                "observation": {
                    "frame_id": f"frame-{index}",
                    "active_url": "https://example.com",
                    "elements": [{"element_id": "dom-1"} for _ in range(100)],
                },
                CONTEXT_REFS_KEY: [_observation_ref(index)],
                SUPERSEDES_SCOPE_KEY: "computer:task-1",
            },
            tool_call_id=f"call-{index}",
        )

    first, second = context.messages
    assert "superseded by a later observation" in first.content
    assert "frame_id=frame-1" in first.content
    assert "dom-1" not in first.content
    assert "dom-1" in second.content
