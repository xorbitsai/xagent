"""Shape-matrix tests for ``classify_chat_response`` (#1714-class).

The classifier is the single structural source of truth for the
chat()/vision_chat() response union; ``unwrap_chat_text`` (agent layer),
``VisionCore`` (tools layer), and the default ``stream_chat`` all delegate
to it. The matrix pins every shape so no consumer ever falls back to
``str(response)`` (repr leak) or reports success on a text-less result.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from xagent.core.model.chat.basic.base import BaseLLM
from xagent.core.model.chat.response_shape import (
    ChatResponseShape,
    classify_chat_response,
)
from xagent.core.tools.core.vision_tool import VisionCore


class TestClassifyChatResponseMatrix:
    @pytest.mark.parametrize(
        ("response", "kind", "text"),
        [
            # Legacy plain-string reply.
            ("hello", "text", "hello"),
            # Typed text envelope from the adapter contract.
            ({"type": "text", "content": "hi"}, "text", "hi"),
            # Content-only dict without a type tag (duck-typed text).
            ({"content": "hi"}, "text", "hi"),
            # Text envelope with extra keys (usage stamp, raw payload).
            (
                {
                    "type": "text",
                    "content": "hi",
                    "usage": {"prompt_tokens": 1},
                    "raw": {},
                },
                "text",
                "hi",
            ),
            # Empty text shapes: empty string content, whitespace-only,
            # and a legacy empty plain string.
            ({"type": "text", "content": ""}, "empty", None),
            ({"type": "text", "content": "  \n\t"}, "empty", None),
            ({"content": ""}, "empty", None),
            ("", "empty", None),
            ("   ", "empty", None),
            # Tool-call envelope (with and without arguments).
            (
                {
                    "type": "tool_call",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"},
                        }
                    ],
                },
                "tool_call",
                None,
            ),
            ({"type": "tool_call", "tool_calls": []}, "tool_call", None),
            # Unknown shapes: unrecognized dict, non-string content,
            # non-dict/non-str payloads.
            ({"foo": 1}, "unknown", None),
            ({"type": "text", "content": 123}, "unknown", None),
            ({"type": "text", "content": ["a", "b"]}, "unknown", None),
            ({"type": "text"}, "unknown", None),
            # An unrecognized type tag does not disqualify usable string
            # content: classification is structural (duck-typed), matching
            # the long-standing unwrap_chat_text acceptance.
            ({"type": "mystery", "content": "hi"}, "text", "hi"),
            (None, "unknown", None),
            (42, "unknown", None),
            (["a", "b"], "unknown", None),
        ],
    )
    def test_shape_matrix(self, response: Any, kind: str, text: str | None) -> None:
        shape = classify_chat_response(response)
        assert shape.kind == kind
        assert shape.text == text

    def test_result_is_named_tuple(self) -> None:
        shape = classify_chat_response("x")
        assert isinstance(shape, ChatResponseShape)
        assert tuple(shape) == ("text", "x")


_MEDIA = "data:image/jpeg;base64,ZmFrZV9pbWFnZV9kYXRh"
_TOOL_CALL_ENVELOPE = {
    "type": "tool_call",
    "tool_calls": [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "read_file", "arguments": "{}"},
        }
    ],
}


def _vision_core(tmp_path: Any, response: Any) -> VisionCore:
    model = Mock(spec=BaseLLM)
    model.vision_chat = AsyncMock(return_value=response)
    model.has_ability = Mock(return_value=True)
    model.supports_native_video_input = False
    return VisionCore(model, output_directory=str(tmp_path))


class TestVisionCoreResponseShapes:
    """VisionCore consumes the classifier: text-less shapes fail explicitly
    -- never a repr leak, never "success with zero results"."""

    @pytest.mark.asyncio
    async def test_understand_media_tool_call_envelope_fails_without_repr(
        self, tmp_path: Any
    ) -> None:
        core = _vision_core(tmp_path, _TOOL_CALL_ENVELOPE)
        result = await core.understand_media(_MEDIA, "What is shown?")
        assert result.success is False
        assert result.error is not None
        assert "'tool_calls'" not in result.error
        assert "'type': 'tool_call'" not in result.error

    @pytest.mark.asyncio
    async def test_understand_media_unknown_shape_fails_without_repr(
        self, tmp_path: Any
    ) -> None:
        core = _vision_core(tmp_path, {"unexpected": {"nested": 1}})
        result = await core.understand_media(_MEDIA, "What is shown?")
        assert result.success is False
        assert result.error is not None
        assert "'nested'" not in result.error

    @pytest.mark.asyncio
    async def test_understand_media_empty_envelope_fails(self, tmp_path: Any) -> None:
        core = _vision_core(tmp_path, {"type": "text", "content": "  "})
        result = await core.understand_media(_MEDIA, "What is shown?")
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_understand_media_text_envelope_succeeds(self, tmp_path: Any) -> None:
        core = _vision_core(
            tmp_path,
            {
                "type": "text",
                "content": "A labelled diagram.",
                "usage": {"prompt_tokens": 3},
            },
        )
        result = await core.understand_media(_MEDIA, "What is shown?")
        assert result.success is True
        assert result.answer == "A labelled diagram."

    @pytest.mark.asyncio
    async def test_detect_objects_tool_call_envelope_fails_without_zero_success(
        self, tmp_path: Any
    ) -> None:
        core = _vision_core(tmp_path, _TOOL_CALL_ENVELOPE)
        result = await core.detect_objects(_MEDIA, task="Find things")
        assert result.success is False
        assert result.error is not None
        assert "'tool_calls'" not in result.error
        assert result.total_detections == 0

    @pytest.mark.asyncio
    async def test_detect_objects_unknown_shape_fails(self, tmp_path: Any) -> None:
        core = _vision_core(tmp_path, {"unexpected": "shape"})
        result = await core.detect_objects(_MEDIA, task="Find things")
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_detect_objects_empty_envelope_fails(self, tmp_path: Any) -> None:
        core = _vision_core(tmp_path, {"type": "text", "content": ""})
        result = await core.detect_objects(_MEDIA, task="Find things")
        assert result.success is False
        assert result.error is not None
