"""A reference rendering of a schema error path, written the slow way, for
tests that check the bounded implementation against it."""

from __future__ import annotations

import json

from xagent.core.agent.structured_schema import ERROR_PATH_MAX_CHARS


def reference_render(segments: list[str | int]) -> str:
    """Escape every segment completely, join, then cut the whole path."""
    parts = []
    for seg in segments:
        if isinstance(seg, int):
            parts.append(str(seg))
        else:
            escaped = seg.replace("~", "~0").replace("/", "~1")
            parts.append(json.dumps(escaped, ensure_ascii=True)[1:-1])
    path = "".join("/" + part for part in parts)
    if len(path) > ERROR_PATH_MAX_CHARS:
        path = "..." + path[-(ERROR_PATH_MAX_CHARS - 3) :]
    return path
