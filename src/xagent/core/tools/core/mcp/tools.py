"""MCP tools.
This module provides MCP tools
"""

from collections.abc import Mapping
from typing import Any

from mcp import ClientSession
from mcp.types import (
    ClientRequest,
    ListToolsRequest,
    ListToolsResult,
    PaginatedRequestParams,
)
from mcp.types import Tool as MCPTool
from pydantic import BaseModel, ConfigDict, model_validator

from .sessions import Connection, create_session

MAX_ITERATIONS = 1000

# The wire name of the annotations object on a tool listing.
RAW_ANNOTATIONS_KEY = "annotations"
# Private key the sandbox runner adds to each serialized tool so the wire
# annotations survive the JSON hop out of the sandbox. Namespaced because it
# travels inside a payload otherwise shaped exactly like the MCP wire tool.
SANDBOX_RAW_ANNOTATIONS_KEY = "_xagentRawAnnotations"


_RAW_ANNOTATIONS_ATTR = "_xagent_raw_annotations"


def _list_tools_request(cursor: str | None) -> "ClientRequest":
    """Build the same ``tools/list`` request ``ClientSession`` would send."""
    params = PaginatedRequestParams(cursor=cursor) if cursor is not None else None
    return ClientRequest(ListToolsRequest(params=params))


def _tools_with_raw_annotations(payload: Any) -> list[MCPTool]:
    """Parse one ``tools/list`` payload into tools carrying wire annotations.

    Two reads of the same payload, deliberately: ``ListToolsResult`` for the
    tools themselves, so the protocol keeps exactly one definition and this
    module never re-implements the SDK's parsing; then the raw mapping for
    each tool's ``annotations``, which the parsed model can no longer answer
    because ``bool | None`` under non-strict validation has already turned
    ``1`` and ``"true"`` into ``True``.

    A plain function rather than a validator on a result subclass. The
    validator version needed class-level state to carry the raw values from
    the "before" pass to the "after" pass, and MCP servers are loaded
    concurrently here -- two validations interleaving on one class would
    attach one server's annotations to another's tools. Nothing here is
    shared: the alignment is a local zip over one payload.

    Alignment is positional, so it is verified rather than assumed. If the
    SDK's parse yields a different number of tools than the payload listed,
    every tool is left without wire evidence, which classifies as undeclared
    -- the safe reading -- instead of shifting annotations onto neighbours.
    """
    parsed = ListToolsResult.model_validate(payload)
    raw_items: list[Any] = []
    if isinstance(payload, Mapping):
        listed = payload.get("tools")
        if isinstance(listed, list):
            raw_items = listed

    if len(raw_items) != len(parsed.tools):
        for tool in parsed.tools:
            attach_raw_annotations(tool, None)
        return list(parsed.tools)

    for tool, item in zip(parsed.tools, raw_items):
        raw = item.get(RAW_ANNOTATIONS_KEY) if isinstance(item, Mapping) else None
        attach_raw_annotations(tool, raw if isinstance(raw, dict) else None)
    return list(parsed.tools)


class _RawListToolsResult(BaseModel):
    """A ``tools/list`` result that keeps the undecoded payload.

    Passed to ``send_request`` as the result type so the response survives
    the SDK's own validation untouched; ``_tools_with_raw_annotations`` then
    reads it twice. ``extra="allow"`` and the model-level capture together
    mean a field this model does not name can never make a listing
    unparsable here when the SDK itself accepted it.
    """

    model_config = ConfigDict(extra="allow")

    nextCursor: str | None = None  # noqa: N815 - MCP wire field name
    payload: dict[str, Any] = {}

    @model_validator(mode="before")
    @classmethod
    def _keep_payload(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            # Copied into a field on this instance -- no class-level state, so
            # concurrent validations cannot see each other's payloads.
            return {**data, "payload": dict(data)}
        return data


def raw_annotations_for(tool: MCPTool) -> dict[str, Any] | None:
    """Return the wire annotations captured for ``tool``, if any.

    The mapping is attached at load time by the functions below. It is read
    back through this accessor rather than off the attribute so that a tool
    that never went through a capturing loader answers ``None`` instead of
    raising -- a caller then classifies it as undeclared, which is the safe
    reading, rather than crashing on a shape it did not choose.
    """
    captured = getattr(tool, _RAW_ANNOTATIONS_ATTR, None)
    return captured if isinstance(captured, dict) else None


def attach_raw_annotations(tool: MCPTool, raw: dict[str, Any] | None) -> MCPTool:
    """Carry one tool's wire annotations alongside the SDK's parsed model.

    Set with ``object.__setattr__`` because ``Tool`` is a Pydantic model that
    does not declare this field: assigning normally would either be rejected
    or land in ``__pydantic_extra__`` and travel into ``model_dump``, which
    would put a private sidecar into the payload the sandbox runner
    serializes. This keeps it a plain Python attribute -- invisible to
    serialization, readable through ``raw_annotations_for``.
    """
    object.__setattr__(tool, _RAW_ANNOTATIONS_ATTR, raw)
    return tool


async def _list_all_tools(session: ClientSession) -> list[MCPTool]:
    """List all available tools from an MCP session with pagination support.

    Each returned tool carries its wire ``annotations`` mapping (see
    ``_attach_raw_annotations``) so a consumer can tell a declared boolean
    from a coercible non-boolean the SDK's non-strict models have already
    flattened. One request per page still: the page is validated twice --
    once by the SDK for the tools everything consumes, once by the shadow
    model for the raw annotations -- but it is the same response either way.

    Args:
        session: The MCP client session.

    Returns:
        A list of all available MCP tools.

    Raises:
        RuntimeError: If maximum iterations exceeded while listing tools.
    """
    current_cursor: str | None = None
    all_tools: list[MCPTool] = []

    iterations = 0

    while True:
        iterations += 1
        if iterations > MAX_ITERATIONS:
            msg = "Reached max of 1000 iterations while listing tools."
            raise RuntimeError(msg)

        # ``_RawListToolsResult`` rather than ``session.list_tools()``: the
        # SDK validates straight into ``ListToolsResult`` and the raw wire
        # types are gone by the time it returns. This asks for the same
        # request and keeps the payload, then hands it to the helper which
        # produces SDK-parsed tools *and* their wire annotations.
        page = await session.send_request(
            _list_tools_request(current_cursor),
            _RawListToolsResult,
        )

        page_tools = _tools_with_raw_annotations(page.payload)
        if page_tools:
            all_tools.extend(page_tools)

        # Pagination spec: https://modelcontextprotocol.io/specification/2025-06-18/server/utilities/pagination
        # compatible with None or ""
        if not page.nextCursor:
            break

        current_cursor = page.nextCursor
    return all_tools


async def load_mcp_tools(
    session: ClientSession | None,
    *,
    connection: Connection | None = None,
) -> list[MCPTool]:
    """Load all available MCP tools.

    Args:
        session: The MCP client session. If None, connection must be provided.
        connection: Connection config to create a new session if session is None.

    Returns:
        List of MCP tools. Tool annotations are returned as part
        of the tool metadata object.

    Raises:
        ValueError: If neither session nor connection is provided.
    """
    if session is None and connection is None:
        msg = "Either a session or a connection config must be provided"
        raise ValueError(msg)

    if session is None:
        # At this point, connection must be non-None since we checked above
        if connection is None:
            msg = "Connection cannot be None when session is None"
            raise ValueError(msg)
        # If a session is not provided, we will create one on the fly
        async with create_session(connection) as tool_session:
            await tool_session.initialize()
            tools = await _list_all_tools(tool_session)
    else:
        tools = await _list_all_tools(session)

    return tools
