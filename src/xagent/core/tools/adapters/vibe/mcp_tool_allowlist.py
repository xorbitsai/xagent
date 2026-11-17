"""Optional per-server allowlist of raw MCP tool names.

``readOnlyHint`` (see ``MCPWriteHint``) is a server-supplied hint, not a trust
boundary, so it cannot be used to keep write tools away from the model. A host
that needs a server to expose only an operator-reviewed set of tools declares
that set in the connection under :data:`READ_ONLY_TOOLS_KEY`. A catalog app
declares it as ``launch_config["read_only_tools"]``; built-in apps declare it in
the code registry. It is a plain allowlist and does not change the approval
gate's handling of read-only calls.

Contract
--------
* Key absent: no filtering. Existing servers are unaffected.
* A list of non-empty strings: only exactly those raw MCP tool names are
  loaded; every other tool the server lists is dropped before an adapter is
  built. An empty list therefore loads nothing.
* Anything else (``None``, a string, a list with a non-string or blank item) is
  malformed and fails closed, like an empty list.

The filter runs on the tool listing, so the direct, sandboxed and execution
scoped (Chrome) loaders share it and the model can never be handed a tool that
is not listed. This is the one parser: the catalog lookup passes the declared
value through unchanged.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

READ_ONLY_TOOLS_KEY = "read_only_tools"

_ToolT = TypeVar("_ToolT")


def is_valid_read_only_tools(value: object) -> bool:
    """Whether ``value`` is a well-formed declaration (list of non-blank str)."""

    return isinstance(value, list) and all(
        isinstance(item, str) and item.strip() for item in value
    )


def read_only_tool_names(connection: Mapping[str, Any]) -> frozenset[str] | None:
    """Return the allowed raw tool names, or ``None`` when unrestricted."""

    if READ_ONLY_TOOLS_KEY not in connection:
        return None
    value = connection[READ_ONLY_TOOLS_KEY]
    return frozenset(value) if is_valid_read_only_tools(value) else frozenset()


def filter_tools_by_read_only_allowlist(
    tools: Sequence[_ToolT], connection: Mapping[str, Any]
) -> list[_ToolT]:
    """Keep only allowlisted tools; return all of them when no list is set."""

    allowed = read_only_tool_names(connection)
    if allowed is None:
        return list(tools)
    kept = [tool for tool in tools if getattr(tool, "name", None) in allowed]
    if len(kept) != len(tools):
        logger.info(
            "MCP read-only allowlist kept %d of %d listed tools", len(kept), len(tools)
        )
    return kept
