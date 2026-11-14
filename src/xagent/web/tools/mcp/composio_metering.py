"""Report the Composio requests a connector tool call made.

Composio bills each Proxy Execute request, and one connector tool call can make
several: a Gmail search lists once and then reads one message per request, and
a batch sends one request per item. The connector subprocess has no usage
tracker bound, so it counts the requests Composio accepted during the call and
puts that count on the MCP tool result's ``_meta``. The agent process reads it
in ``MCPToolAdapter`` and records one row per accepted request.

A batch returns at its deadline without waiting for the requests it has
already sent, and each of those can still be accepted by Composio afterwards.
The count therefore waits for those requests to finish before it is taken, so
a late acceptance is still reported.

Only the connector servers that can run on a Composio grant use this class; a
native grant runs on the same module and makes no proxy call, so its count is
zero.
"""

from __future__ import annotations

import asyncio
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult

from ....core.tools.core.mcp.composio_requests import COMPOSIO_REQUESTS_META_KEY
from .composio_proxy import (
    begin_request_count,
    end_request_count,
    pending_batch_requests,
)


class ComposioMeteringFastMCP(FastMCP):
    """A FastMCP server that reports how many Composio requests a call made."""

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        token = begin_request_count()
        try:
            result = await super().call_tool(name, arguments)
            # A batch that hit its deadline has already-sent requests still
            # running; each is bounded by its own proxy timeout and can still
            # be accepted, so they have to finish before the count is taken.
            # One that fails is not this tool call's failure.
            await asyncio.gather(
                *(asyncio.wrap_future(future) for future in pending_batch_requests()),
                return_exceptions=True,
            )
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
