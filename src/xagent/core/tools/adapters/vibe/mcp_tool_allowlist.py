"""Optional per-server allowlist of raw MCP tool names.

``readOnlyHint`` (see ``MCPWriteHint``) is a server-supplied hint, not a trust
boundary, so it cannot be used to keep write tools away from the model. A host
that needs a server to expose only an operator-reviewed set of tools declares
that set in the connection under :data:`READ_ONLY_TOOLS_KEY`.

Contract
--------
* Key absent: no filtering. Existing servers are unaffected.
* A list (or a comma-separated string) of raw MCP tool names: only exactly
  those tools are loaded; every other tool the server lists is dropped before
  an adapter is built. An empty list therefore loads nothing.
* Any other value, including an explicit ``None``, is malformed and fails
  closed, like an empty list.

The filter runs on the tool listing, before any adapter exists, so the direct
and the sandboxed loading paths share it and the model can never be handed a
tool that is not listed.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

READ_ONLY_TOOLS_KEY = "read_only_tools"

_ToolT = TypeVar("_ToolT")


def read_only_tool_names(connection: Mapping[str, Any]) -> frozenset[str] | None:
    """Return the allowed raw tool names, or ``None`` when unrestricted."""

    if READ_ONLY_TOOLS_KEY not in connection:
        return None
    value = connection[READ_ONLY_TOOLS_KEY]
    if isinstance(value, str):
        return frozenset(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, (list, tuple, set, frozenset)):
        return frozenset(
            item.strip() for item in value if isinstance(item, str) and item.strip()
        )
    return frozenset()


def filter_tools_by_read_only_allowlist(
    tools: Sequence[_ToolT],
    connection: Mapping[str, Any],
    *,
    server_name: str = "",
) -> list[_ToolT]:
    """Keep only allowlisted tools; return all of them when no list is set."""

    allowed = read_only_tool_names(connection)
    if allowed is None:
        return list(tools)
    kept = [tool for tool in tools if getattr(tool, "name", None) in allowed]
    if len(kept) != len(tools):
        logger.info(
            "MCP server %s: read-only allowlist kept %d of %d tools",
            server_name,
            len(kept),
            len(tools),
        )
    return kept
