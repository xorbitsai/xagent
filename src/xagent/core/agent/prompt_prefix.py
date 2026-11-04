"""Compare traced runtime requests without retaining prompt text.

These are message/schema comparisons before provider templating and reference
materialization, not token-prefix lengths or explanations of engine cache misses.
Only provider-reported cached tokens establish an actual cache hit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

PROMPT_PREFIX_KEY = "prompt_prefix"


def _digest(value: Any) -> bytes:
    # Preserve list and mapping order: provider templates may serialize tool
    # schemas in that order. Do not mutate/canonicalize the outgoing request.
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).digest()


@dataclass
class PromptPrefixObserver:
    # One snapshot per call kind (main, retry, decision, compaction), so an
    # auxiliary call cannot replace the main loop's comparison baseline.
    _previous: dict[str, tuple[tuple[str, ...], bytes, tuple[bytes, ...]]] = field(
        default_factory=dict, repr=False
    )

    def observe(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        scope: tuple[str, ...],
        call_kind: str,
    ) -> dict[str, Any]:
        try:
            schemas = _digest(tools or [])
            fingerprints = tuple(_digest(message) for message in messages)
        except (TypeError, ValueError, RecursionError):
            self._previous.pop(call_kind, None)
            return {"change": "unavailable"}
        previous = self._previous.get(call_kind)
        self._previous[call_kind] = (scope, schemas, fingerprints)
        if previous is None or previous[0] != scope:
            return {"change": "initial" if previous is None else "scope_changed"}
        old = previous[2]
        common = 0
        for before, after in zip(old, fingerprints):
            if before != after:
                break
            common += 1
        tools_changed = previous[1] != schemas
        if tools_changed:
            change = "tools_changed"
        elif common < min(len(old), len(fingerprints)):
            change = "messages_changed"
        elif len(fingerprints) < len(old):
            change = "messages_removed"
        elif len(fingerprints) > len(old):
            change = "append_only"
        else:
            change = "identical"
        return {
            "change": change,
            "tools_changed": tools_changed,
            "common_prefix_messages": common,
            "previous_message_count": len(old),
            "message_count": len(fingerprints),
            "first_changed_message_index": (
                common if common < min(len(old), len(fingerprints)) else None
            ),
        }
