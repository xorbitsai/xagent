"""Size breakdown of one LLM request, for diagnosing prompt-token cost.

Counts and character lengths only: never message text, tool arguments or
schema bodies. Tool names appear solely as the keys of the per-origin counts.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

PROMPT_COMPOSITION_KEY = "prompt_composition"
BUILTIN_TOOL_ORIGIN = "builtin"
UNKNOWN_ORIGIN = "unknown"
# Top contributors are named so a single oversized schema is visible; the cap
# keeps the trace payload small however many tools are attached.
LARGEST_TOOLS_LIMIT = 5

CALL_KIND_MAIN = "main"
CALL_KIND_PROTOCOL_RETRY = "protocol_retry"
CALL_KIND_DECISION = "decision"
CALL_KIND_COMPACTION = "compaction"


def _content_chars(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for part in content:
            if isinstance(part, str):
                total += len(part)
            elif isinstance(part, Mapping) and isinstance(part.get("text"), str):
                total += len(part["text"])
        return total
    return 0


def _message_chars(message: Mapping[str, Any]) -> int:
    total = _content_chars(message.get("content"))
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            function = call.get("function") if isinstance(call, Mapping) else None
            if isinstance(function, Mapping):
                arguments = function.get("arguments")
                total += len(arguments) if isinstance(arguments, str) else 0
                total += len(str(function.get("name") or ""))
    return total


def _schema_chars(schema: Any) -> int:
    try:
        return len(json.dumps(schema, ensure_ascii=False, separators=(",", ":")))
    except (TypeError, ValueError):
        return 0


def _schema_name(schema: Any) -> str:
    function = schema.get("function") if isinstance(schema, Mapping) else None
    name = function.get("name") if isinstance(function, Mapping) else None
    return name if isinstance(name, str) else ""


def call_kind_from_metadata(metadata: Mapping[str, Any] | None) -> str:
    """Classify a call from the metadata its call site already passes."""
    metadata = metadata or {}
    if metadata.get("purpose") == "context_compaction":
        return CALL_KIND_COMPACTION
    phase = str(metadata.get("phase") or "")
    if phase.startswith("repeated_tool_decision"):
        return CALL_KIND_DECISION
    if phase:
        return CALL_KIND_PROTOCOL_RETRY
    return CALL_KIND_MAIN


def build_prompt_composition(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None,
    *,
    call_kind: str,
    iteration: int | None = None,
    tool_origins: Mapping[str, str] | None = None,
    context_tokens: int | None = None,
) -> dict[str, Any]:
    """Summarise sizes of the request about to be sent.

    ``tool_origins`` maps a tool name to its origin (tool category, ``mcp``,
    ``builtin`` for the react control tools); names not in it are bucketed as
    ``unknown``.
    """
    origins = tool_origins or {}
    role_chars: dict[str, int] = defaultdict(int)
    role_count: dict[str, int] = defaultdict(int)
    for message in messages:
        role = str(message.get("role") or "unknown")
        role_chars[role] += _message_chars(message)
        role_count[role] += 1
    system_chars = role_chars.get("system", 0)

    tool_chars: dict[str, int] = defaultdict(int)
    tool_count: dict[str, int] = defaultdict(int)
    per_tool: list[tuple[int, str]] = []
    for schema in tools or ():
        name = _schema_name(schema)
        origin = origins.get(name) or UNKNOWN_ORIGIN
        chars = _schema_chars(schema)
        tool_chars[origin] += chars
        tool_count[origin] += 1
        per_tool.append((chars, name))
    per_tool.sort(reverse=True)

    composition: dict[str, Any] = {
        "call_kind": call_kind,
        "message_count": len(messages),
        "system_chars": system_chars,
        "role_chars": dict(role_chars),
        "role_count": dict(role_count),
        "tools_count": sum(tool_count.values()),
        "tools_chars": sum(tool_chars.values()),
        "tools_count_by_origin": dict(tool_count),
        "tools_chars_by_origin": dict(tool_chars),
        "largest_tools": [
            {"name": name, "chars": chars}
            for chars, name in per_tool[:LARGEST_TOOLS_LIMIT]
        ],
    }
    if iteration is not None:
        composition["iteration"] = iteration
    if context_tokens is not None:
        composition["context_tokens"] = context_tokens
    return composition
