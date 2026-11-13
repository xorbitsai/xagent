"""Report the Composio requests a connector tool call made.

Composio bills each Proxy Execute request, and one connector tool call can make
several: a Gmail search lists once and then reads one message per request, and
a batch sends one request per item. The connector subprocess has no usage
tracker bound, so it counts the requests Composio accepted during the call and
puts that count on the MCP tool result's ``_meta``. The agent process reads it
in ``MCPToolAdapter`` and records one row per accepted request.

Only the connector servers that can run on a Composio grant use this class; a
native grant runs on the same module and makes no proxy call, so its count is
zero.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult

from ....core.tools.core.mcp.composio_requests import COMPOSIO_REQUESTS_META_KEY
from .composio_proxy import begin_request_count, end_request_count


class ComposioMeteringFastMCP(FastMCP):
    """A FastMCP server that reports how many Composio requests a call made."""

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        token = begin_request_count()
        try:
            result = await super().call_tool(name, arguments)
        finally:
            accepted = end_request_count(token)
        meta = {COMPOSIO_REQUESTS_META_KEY: accepted}
        if isinstance(result, CallToolResult):
            result.meta = {**(result.meta or {}), **meta}
            return result
        if isinstance(result, tuple):
            content, structured = result
            tool_result = CallToolResult(
                content=list(content), structuredContent=structured
            )
        else:
            tool_result = CallToolResult(content=list(result))
        tool_result.meta = meta
        return tool_result
