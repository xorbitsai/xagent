"""The vision tool accepts a configured Auto only when every candidate reads images."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Any

from xagent.core.model.chat.basic.base import BaseLLM
from xagent.core.model.chat.basic.router import RouterLLM
from xagent.core.tools.core.vision_tool import VisionCore

# 1x1 PNG
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)


class _Downstream(BaseLLM):
    abilities = ["chat", "vision"]
    model_name = "downstream"
    supports_thinking_mode = False
    calls: list[list[dict[str, Any]]] = []

    async def chat(self, messages, *a, **k):  # type: ignore[no-untyped-def]
        return "chat"

    async def vision_chat(self, messages, *a, **k):  # type: ignore[no-untyped-def]
        type(self).calls.append(messages)
        return "a red pixel"

    async def stream_chat(self, *a, **k):  # type: ignore[no-untyped-def]
        raise NotImplementedError


def _auto(modalities: dict[str, list[str]]) -> RouterLLM:
    return RouterLLM(
        model_name="auto",
        abilities=["chat", "tool_calling", "vision"],
        mode="sequential_failover",
        candidate_models=list(modalities),
        candidate_input_modalities=modalities,
        downstream_resolver=lambda *a, **k: _Downstream(),
        use_environment_fallback=False,
    )


def _core(tmp_path: Path, model: BaseLLM) -> tuple[VisionCore, str]:
    image = tmp_path / "p.png"
    image.write_bytes(_PNG)
    return VisionCore(model, str(tmp_path), str(tmp_path)), str(image)


def test_accepts_image_input_needs_every_candidate() -> None:
    both = ["text", "image"]
    assert _auto({"candidate:1": both, "candidate:2": both}).accepts_image_input
    assert not _auto({"candidate:1": both, "candidate:2": ["text"]}).accepts_image_input
    assert not RouterLLM(
        model_name="auto", use_environment_fallback=False
    ).accepts_image_input


def test_router_still_does_not_advertise_vision() -> None:
    router = _auto({"candidate:1": ["text", "image"]})
    assert not router.has_ability("vision")


def test_understand_images_runs_on_all_vision_auto(tmp_path: Path) -> None:
    _Downstream.calls = []
    core, image = _core(tmp_path, _auto({"candidate:1": ["text", "image"]}))
    result = asyncio.run(core.understand_images([image], "what colour?"))
    assert result.success, result.error
    assert _Downstream.calls, "the image never reached the downstream model"


def test_understand_images_refuses_auto_with_a_text_only_candidate(
    tmp_path: Path,
) -> None:
    _Downstream.calls = []
    core, image = _core(
        tmp_path, _auto({"candidate:1": ["text", "image"], "candidate:2": ["text"]})
    )
    result = asyncio.run(core.understand_images([image], "what colour?"))
    assert not result.success
    assert "does not support vision" in (result.error or "")
    assert not _Downstream.calls


def test_detect_objects_gate_accepts_all_vision_auto_and_refuses_mixed(
    tmp_path: Path,
) -> None:
    core, image = _core(tmp_path, _auto({"candidate:1": ["text", "image"]}))
    ok = asyncio.run(core.detect_objects(image, "pixel"))
    assert "does not support vision" not in (ok.error or "")
    core, image = _core(
        tmp_path, _auto({"candidate:1": ["text", "image"], "candidate:2": ["text"]})
    )
    refused = asyncio.run(core.detect_objects(image, "pixel"))
    assert "does not support vision" in (refused.error or "")
