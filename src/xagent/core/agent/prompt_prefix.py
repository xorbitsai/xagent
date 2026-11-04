"""Compare traced runtime requests without retaining prompt text.

These are message/schema comparisons before provider templating and reference
materialization, not token-prefix lengths or explanations of engine cache misses.
Only provider-reported cached tokens establish an actual cache hit.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

PROMPT_PREFIX_KEY = "prompt_prefix"
_MAX_BASELINES = 32


def _digest(value: Any) -> bytes:
    # Preserve list and mapping order: provider templates may serialize tool
    # schemas in that order. Do not mutate/canonicalize the outgoing request.
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).digest()


@dataclass
class PromptPrefixObserver:
    # Parallel steps and auxiliary phases keep separate comparison baselines.
    _previous: OrderedDict[
        tuple[tuple[str, ...], str, str], tuple[bytes, tuple[bytes, ...]]
    ] = field(default_factory=OrderedDict, repr=False)

    def observe(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        scope: tuple[str, ...],
        call_kind: str,
        phase: str = "",
    ) -> dict[str, Any]:
        key = (scope, call_kind, phase)
        try:
            schemas = _digest(tools or [])
            fingerprints = tuple(_digest(message) for message in messages)
        except (TypeError, ValueError, RecursionError):
            self._previous.pop(key, None)
            return {"change": "unavailable"}
        previous = self._previous.get(key)
        self._previous[key] = (schemas, fingerprints)
        self._previous.move_to_end(key)
        if len(self._previous) > _MAX_BASELINES:
            self._previous.popitem(last=False)
        if previous is None:
            return {"change": "initial"}
        previous_schemas, old = previous
        common = 0
        for before, after in zip(old, fingerprints):
            if before != after:
                break
            common += 1
        tools_changed = previous_schemas != schemas
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
