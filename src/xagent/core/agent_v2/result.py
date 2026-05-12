from __future__ import annotations

from typing import Any


def extract_assistant_message(result: dict[str, Any]) -> str | None:
    """Return the assistant-facing output from a normalized pattern result."""

    for key in ("response", "answer", "output", "content", "message"):
        value = result.get(key)
        if isinstance(value, str) and value:
            return value
    return None
