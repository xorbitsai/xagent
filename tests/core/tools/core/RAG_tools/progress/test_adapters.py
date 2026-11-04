"""Unit tests for progress adapters."""

from __future__ import annotations

from typing import Any, Dict, Optional

from xagent.core.tools.core.RAG_tools.progress.adapters import DeepDocProgressAdapter


class _DummyProgressCallback:
    """Simple callback collector for adapter testing."""

    def __init__(self) -> None:
        self.messages: list[str] = []
        self.details: list[Optional[Dict[str, Any]]] = []

    def on_status_update(
        self, status: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Collect status updates."""
        self.messages.append(status)
        self.details.append(details)


def test_deepdoc_adapter_supports_single_argument_callback() -> None:
    """Adapter should accept DeepDoc's progress-only callback shape."""
    callback = _DummyProgressCallback()
    adapter = DeepDocProgressAdapter(callback)
    deepdoc_callback = adapter.get_callback()

    deepdoc_callback(0.3)

    assert callback.messages == ["OCR processing"]


def test_deepdoc_adapter_normalizes_message_and_deduplicates() -> None:
    """Adapter should strip duration suffix and avoid duplicate updates."""
    callback = _DummyProgressCallback()
    adapter = DeepDocProgressAdapter(callback)
    deepdoc_callback = adapter.get_callback()

    deepdoc_callback(0.4, "OCR finished (2.34s)")
    deepdoc_callback(0.5, "OCR finished (3.01s)")
    deepdoc_callback(0.7, "Layout analysis (1.23s)")

    assert callback.messages == ["OCR finished", "Layout analysis"]
