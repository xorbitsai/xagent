"""Structural classification of ``chat()``/``vision_chat()`` response shapes.

Adapters return a small union from the non-streaming chat methods:

- a legacy plain string (the assistant reply),
- a text envelope ``{"type": "text", "content": ...}`` (optionally with a
  top-level ``usage`` stamp and a ``raw`` provider payload),
- a tool-call envelope ``{"type": "tool_call", "tool_calls": [...]}``.

This module is the single, dependency-neutral source of truth for telling
those shapes apart. Consumers that need the text (``unwrap_chat_text`` in
the agent layer, ``VisionCore`` in the tools layer, the default
``stream_chat`` in this package) classify here instead of re-implementing
isinstance chains -- and none of them ever falls back to ``str(response)``,
which would leak an internal dict repr as if it were model output (#1714).
"""

from typing import Any, Literal, NamedTuple


class ChatResponseShape(NamedTuple):
    """Structural reading of a chat response.

    ``kind`` is one of:

    - ``"text"``: usable text is present; ``text`` carries it.
    - ``"empty"``: a text-bearing shape with no usable text (empty or
      whitespace-only) -- the same transient condition adapters raise
      ``LLMEmptyContentError`` for.
    - ``"tool_call"``: a tool-call envelope; there is no text by design.
    - ``"unknown"``: any other payload (unrecognized dict, non-string
      content, non-dict/non-string value).
    """

    kind: Literal["text", "empty", "tool_call", "unknown"]
    text: str | None


def classify_chat_response(response: Any) -> ChatResponseShape:
    """Classify a ``chat()``/``vision_chat()`` response by structure.

    Classification is purely structural and never raises: a legacy plain
    string is text (or empty when whitespace-only); a dict tagged
    ``type == "tool_call"`` is a tool call; any other dict with string
    ``content`` is text (or empty when whitespace-only) regardless of its
    ``type`` tag, matching the duck-typed acceptance ``unwrap_chat_text``
    has always had; everything else is unknown.
    """
    if isinstance(response, str):
        if response.strip():
            return ChatResponseShape("text", response)
        return ChatResponseShape("empty", None)
    if isinstance(response, dict):
        if response.get("type") == "tool_call":
            return ChatResponseShape("tool_call", None)
        content = response.get("content")
        if isinstance(content, str):
            if content.strip():
                return ChatResponseShape("text", content)
            return ChatResponseShape("empty", None)
        return ChatResponseShape("unknown", None)
    return ChatResponseShape("unknown", None)
