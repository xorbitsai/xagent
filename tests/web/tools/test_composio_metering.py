"""The ``_meta`` count a Composio-backed connector reports on a tool result.

Composio bills each Proxy Execute request, and one tool call can make several.
The connector subprocess reports how many were accepted so the agent process
can record that many rows; these tests pin what the server puts on the result,
including what a call that raises cannot report.
"""

from __future__ import annotations

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from xagent.core.tools.core.mcp.composio_requests import COMPOSIO_REQUESTS_META_KEY
from xagent.web.tools.mcp import composio_proxy
from xagent.web.tools.mcp.composio_metering import ComposioMeteringFastMCP


def _server() -> ComposioMeteringFastMCP:
    mcp = ComposioMeteringFastMCP("metering-test")

    @mcp.tool()
    def accepted_twice() -> str:
        composio_proxy.record_accepted_request()
        composio_proxy.record_accepted_request()
        return "ok"

    @mcp.tool()
    def accepted_then_raises() -> str:
        composio_proxy.record_accepted_request()
        raise RuntimeError("failed after Composio accepted the request")

    @mcp.tool()
    def accepted_nothing() -> str:
        return "ok"

    @mcp.tool(structured_output=False)
    def accepted_without_an_output_schema() -> str:
        composio_proxy.record_accepted_request()
        return "ok"

    return mcp


@pytest.mark.asyncio
async def test_the_result_reports_every_accepted_request() -> None:
    async with create_connected_server_and_client_session(
        _server()._mcp_server
    ) as session:
        result = await session.call_tool("accepted_twice", {})

    assert result.isError is False
    assert result.meta == {COMPOSIO_REQUESTS_META_KEY: 2}


@pytest.mark.asyncio
async def test_a_call_that_accepts_nothing_reports_zero() -> None:
    async with create_connected_server_and_client_session(
        _server()._mcp_server
    ) as session:
        result = await session.call_tool("accepted_nothing", {})

    assert result.isError is False
    assert result.meta == {COMPOSIO_REQUESTS_META_KEY: 0}


@pytest.mark.asyncio
async def test_a_tool_without_an_output_schema_still_reports_its_count() -> None:
    """A tool that declares no output schema gets only a content list back.

    FastMCP returns ``(content, structured)`` only when the tool declares an
    output schema and returns the bare content list otherwise, so both shapes
    have to come back as a result carrying the count.
    """
    async with create_connected_server_and_client_session(
        _server()._mcp_server
    ) as session:
        result = await session.call_tool("accepted_without_an_output_schema", {})

    assert result.isError is False
    assert result.meta == {COMPOSIO_REQUESTS_META_KEY: 1}


@pytest.mark.asyncio
async def test_a_call_that_raises_loses_its_count() -> None:
    """A raised tool error becomes an ``isError`` result without the count.

    FastMCP builds that result itself, so the accepted requests before the
    failure cannot be reported and are under-counted. The tool contract keeps
    errors as returned payloads where that matters (every Composio connector
    catches its own errors and returns them).
    """
    async with create_connected_server_and_client_session(
        _server()._mcp_server
    ) as session:
        result = await session.call_tool("accepted_then_raises", {})

    assert result.isError is True
    assert result.meta is None
