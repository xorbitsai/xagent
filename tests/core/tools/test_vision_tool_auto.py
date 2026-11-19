"""The vision tool accepts a configured Auto with at least one image-reading candidate."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Any

import pytest

from xagent.core.model.chat.basic.base import BaseLLM
from xagent.core.model.chat.basic.router import RouterLLM, _ProfileLookupError
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


def test_accepts_image_input_needs_one_image_candidate() -> None:
    both = ["text", "image"]
    assert _auto({"candidate:1": both, "candidate:2": both}).accepts_image_input
    assert _auto({"candidate:1": both, "candidate:2": ["text"]}).accepts_image_input
    assert _auto({"candidate:1": ["text"], "candidate:2": both}).accepts_image_input
    assert not _auto({"candidate:1": ["text"]}).accepts_image_input
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


def test_understand_images_refuses_auto_without_an_image_candidate(
    tmp_path: Path,
) -> None:
    _Downstream.calls = []
    core, image = _core(
        tmp_path, _auto({"candidate:1": ["text"], "candidate:2": ["text"]})
    )
    result = asyncio.run(core.understand_images([image], "what colour?"))
    assert not result.success
    assert "does not support vision" in (result.error or "")
    assert not _Downstream.calls


def test_detect_objects_gate_accepts_an_image_candidate_and_refuses_none(
    tmp_path: Path,
) -> None:
    core, image = _core(
        tmp_path, _auto({"candidate:1": ["text", "image"], "candidate:2": ["text"]})
    )
    ok = asyncio.run(core.detect_objects(image, "pixel"))
    assert "does not support vision" not in (ok.error or "")
    core, image = _core(tmp_path, _auto({"candidate:1": ["text"]}))
    refused = asyncio.run(core.detect_objects(image, "pixel"))
    assert "does not support vision" in (refused.error or "")


class _Recording(BaseLLM):
    """A downstream that records who served it; a text-only one refuses images."""

    abilities = ["chat"]
    supports_thinking_mode = False
    served: list[str] = []

    def __init__(self, name: str, *, reads_images: bool) -> None:
        self._name = name
        self._reads_images = reads_images

    @property
    def model_name(self) -> str:
        return self._name

    async def chat(self, messages, *a, **k):  # type: ignore[no-untyped-def]
        return "chat"

    async def vision_chat(self, messages, *a, **k):  # type: ignore[no-untyped-def]
        if not self._reads_images:
            raise AssertionError(f"{self.model_name} cannot read images")
        type(self).served.append(self.model_name)
        return "a red pixel"

    async def stream_chat(self, *a, **k):  # type: ignore[no-untyped-def]
        raise NotImplementedError


def test_the_image_reading_candidate_serves_the_call_and_the_text_only_one_is_skipped(
    tmp_path: Path,
) -> None:
    # The text-only candidate comes first, so a router that did not skip it would fail.
    _Recording.served = []
    modalities = {"text-only": ["text"], "reads-images": ["text", "image"]}
    router = RouterLLM(
        model_name="auto",
        abilities=["chat", "tool_calling", "vision"],
        mode="sequential_failover",
        candidate_models=list(modalities),
        candidate_input_modalities=modalities,
        downstream_resolver=lambda model_id, **_k: _Recording(
            model_id, reads_images=model_id == "reads-images"
        ),
        use_environment_fallback=False,
    )
    core, image = _core(tmp_path, router)
    result = asyncio.run(core.understand_images([image], "what colour?"))
    assert result.success, result.error
    assert _Recording.served == ["reads-images"]


def _routing(candidates: list[str], **kwargs: Any) -> RouterLLM:
    """Routing mode as production builds it: no configured candidate modalities."""
    return RouterLLM(
        model_name="auto",
        abilities=["chat", "tool_calling", "vision"],
        mode="routing",
        candidate_models=candidates,
        use_environment_fallback=False,
        **kwargs,
    )


def _profiles(
    monkeypatch: pytest.MonkeyPatch, table: dict[str, tuple[str, ...]]
) -> None:
    monkeypatch.setattr(
        RouterLLM,
        "_profile_input_modalities",
        staticmethod(lambda model_id: table[model_id]),
    )


def test_routing_mode_with_a_mixed_profile_accepts_image_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _profiles(monkeypatch, {"a": ("text", "image"), "b": ("text",)})
    assert _routing(["a", "b"]).accepts_image_input
    assert _routing(["b", "a"]).accepts_image_input
    _profiles(monkeypatch, {"a": ("text",), "b": ("text",)})
    assert not _routing(["a", "b"]).accepts_image_input


def test_routing_mode_refuses_when_the_profile_lookup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(_model_id: str) -> tuple[str, ...]:
        raise _ProfileLookupError("profiles unavailable")

    monkeypatch.setattr(
        RouterLLM, "_profile_input_modalities", staticmethod(unavailable)
    )
    assert not _routing(["a", "b"]).accepts_image_input


def test_an_excluded_image_candidate_does_not_make_the_auto_image_capable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _profiles(monkeypatch, {"a": ("text", "image"), "b": ("text",)})
    monkeypatch.setenv("XAGENT_XROUTER_EXCLUDED_MODELS", "a")
    assert not _routing(["a", "b"]).accepts_image_input
    modalities = {"a": ["text", "image"], "b": ["text"]}
    sequential = RouterLLM(
        model_name="auto",
        abilities=["chat", "tool_calling", "vision"],
        mode="sequential_failover",
        candidate_models=list(modalities),
        candidate_input_modalities=modalities,
        use_environment_fallback=False,
    )
    assert not sequential.accepts_image_input


def test_an_image_reading_fallback_keeps_an_otherwise_excluded_auto_image_capable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _profiles(
        monkeypatch, {"a": ("text", "image"), "b": ("text",), "c": ("text", "image")}
    )
    monkeypatch.setenv("XAGENT_XROUTER_EXCLUDED_MODELS", "a")
    assert _routing(["a", "b", "c"], fallback_model="c").accepts_image_input
    assert not _routing(["a", "b"], fallback_model="b").accepts_image_input
