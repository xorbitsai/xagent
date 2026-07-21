"""Stable OpenAI-compatible names for agent delegation tools."""

import re
import unicodedata
from typing import Any

from pypinyin import lazy_pinyin  # type: ignore[import-not-found]

AGENT_TOOL_NAME_PREFIX = "agent_"
LEGACY_AGENT_TOOL_NAME_PREFIX = "call_agent_"
WORKFORCE_AGENT_TOOL_NAME_PREFIX = "worker_"
AGENT_TOOL_NAME_ID_SEPARATOR = "__a"
MAX_AGENT_TOOL_NAME_LENGTH = 64


def _normalize_agent_id(agent_id: Any) -> int:
    if isinstance(agent_id, bool):
        raise ValueError("agent_id must be an integer")
    try:
        normalized_agent_id = int(agent_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("agent_id must be an integer") from exc
    if normalized_agent_id <= 0:
        raise ValueError("agent_id must be positive")
    return normalized_agent_id


def _semantic_slug(value: Any) -> str:
    """Return an ASCII-only semantic slug accepted by tool-call providers."""
    raw_name = value if isinstance(value, str) else ""
    # Tool schemas from OpenAI-compatible providers generally require ASCII
    # names. Romanize Han characters before stripping the remaining Unicode so
    # Chinese Agent names retain useful semantics instead of all collapsing to
    # the same generic fallback.
    normalized_name = unicodedata.normalize("NFKD", raw_name.strip())
    romanized_name = (
        normalized_name
        if normalized_name.isascii()
        else "_".join(lazy_pinyin(normalized_name))
    )
    ascii_name = romanized_name.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_name.lower())
    return re.sub(r"_+", "_", slug).strip("_-") or "agent"


def _semantic_agent_tool_name(prefix: str, agent_id: Any, semantic_name: Any) -> str:
    normalized_agent_id = _normalize_agent_id(agent_id)
    suffix = f"{AGENT_TOOL_NAME_ID_SEPARATOR}{normalized_agent_id}"
    semantic_budget = MAX_AGENT_TOOL_NAME_LENGTH - len(prefix) - len(suffix)
    normalized_name = _semantic_slug(semantic_name)
    if len(normalized_name) > semantic_budget:
        normalized_name = normalized_name[:semantic_budget].rstrip("_-")
        if "_" in normalized_name:
            normalized_name = normalized_name.rsplit("_", 1)[0]
    normalized_name = normalized_name or "agent"
    return f"{prefix}{normalized_name}{suffix}"


def gen_agent_tool_name(agent_id: Any, semantic_name: Any = None) -> str:
    """Return a stable Agent tool name.

    Passing a name returns the current semantic form. Omitting it preserves the
    historical ``agent_<id>`` form for stored traces and compatibility paths.
    """
    normalized_agent_id = _normalize_agent_id(agent_id)
    if semantic_name is None:
        return f"{AGENT_TOOL_NAME_PREFIX}{normalized_agent_id}"
    return _semantic_agent_tool_name(
        AGENT_TOOL_NAME_PREFIX, normalized_agent_id, semantic_name
    )


def gen_workforce_agent_tool_name(agent_id: Any, semantic_name: Any) -> str:
    """Return a unique, semantic name for a Workforce delegation tool.

    The readable portion helps the manager choose the correct worker while the
    agent id suffix guarantees uniqueness when names or aliases collide.
    """
    return _semantic_agent_tool_name(
        WORKFORCE_AGENT_TOOL_NAME_PREFIX, agent_id, semantic_name
    )


def parse_agent_tool_id(tool_name: Any) -> int | None:
    if not isinstance(tool_name, str):
        return None
    if tool_name.startswith(LEGACY_AGENT_TOOL_NAME_PREFIX):
        suffix = tool_name.removeprefix(LEGACY_AGENT_TOOL_NAME_PREFIX)
        return int(suffix) if suffix.isdecimal() and int(suffix) > 0 else None
    if tool_name.startswith(AGENT_TOOL_NAME_PREFIX):
        suffix = tool_name.removeprefix(AGENT_TOOL_NAME_PREFIX)
        if suffix.isdecimal():
            return int(suffix) if int(suffix) > 0 else None
    if not tool_name.startswith(
        (AGENT_TOOL_NAME_PREFIX, WORKFORCE_AGENT_TOOL_NAME_PREFIX)
    ):
        return None
    semantic_part, separator, agent_id = tool_name.rpartition(
        AGENT_TOOL_NAME_ID_SEPARATOR
    )
    if not separator or not agent_id.isdecimal() or int(agent_id) <= 0:
        return None
    prefix = (
        AGENT_TOOL_NAME_PREFIX
        if tool_name.startswith(AGENT_TOOL_NAME_PREFIX)
        else WORKFORCE_AGENT_TOOL_NAME_PREFIX
    )
    semantic_name = semantic_part.removeprefix(prefix)
    if not semantic_name or not re.fullmatch(r"[a-zA-Z0-9_]+", semantic_name):
        return None
    return int(agent_id)


def is_agent_tool_name(tool_name: Any) -> bool:
    if parse_agent_tool_id(tool_name) is not None:
        return True
    return isinstance(tool_name, str) and tool_name.startswith(
        LEGACY_AGENT_TOOL_NAME_PREFIX
    )
