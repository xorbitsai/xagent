"""The ``_meta`` count a Composio-backed connector reports on a tool result.

Composio bills each Proxy Execute request, and one tool call can make several.
The connector subprocess reports how many were accepted so the agent process
can record that many rows; these tests pin what the server puts on the result,
including what a call that raises cannot report, and that a batch request
accepted after the batch's deadline still reaches the count.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest
from googleapiclient.discovery import build
from mcp.shared.memory import create_connected_server_and_client_session

from tests.web.tools.conftest import GMAIL_HOSTS, _Response
from xagent.core.tools.core.mcp.composio_requests import COMPOSIO_REQUESTS_META_KEY
from xagent.web.tools.mcp import composio_proxy, google_api_proxy
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


class _TimedProxy:
    """The proxy answering each message read after its own delay."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._delays: dict[str, float] = {}
        self._refusals: set[str] = set()
        self._lock = threading.Lock()

    def answers_after(self, message_id: str, delay: float) -> None:
        self._delays[message_id] = delay

    def refuses_after(self, message_id: str, delay: float) -> None:
        self._delays[message_id] = delay
        self._refusals.add(message_id)

    def request(self, **kwargs: Any) -> _Response:
        message_id = kwargs["json"]["endpoint"].rsplit("/", 1)[-1]
        with self._lock:
            self.calls.append(message_id)
        time.sleep(self._delays.get(message_id, 0.0))
        if message_id in self._refusals:
            return _Response(500, {"error": {"message": "refused"}})
        return _Response(200, {"status": 200, "data": {"id": message_id}})


@pytest.fixture
def timed_proxy(monkeypatch, composio_env) -> _TimedProxy:
    """A proxy whose per-message answer delay the test controls."""
    fake = _TimedProxy()
    monkeypatch.setattr(composio_proxy.requests, "request", fake.request)
    return fake


def _batch_server(
    *, deadline: float, max_workers: int = 8
) -> tuple[ComposioMeteringFastMCP, list[tuple[str, Any, Any]]]:
    """A metered server whose only tool reads the given messages as one batch."""
    mcp = ComposioMeteringFastMCP("batch-metering-test")
    seen: list[tuple[str, Any, Any]] = []

    @mcp.tool()
    def read_messages(messages: list[str]) -> str:
        """Read the named messages with one proxied batch."""
        service = build(
            "gmail",
            "v1",
            http=google_api_proxy.GoogleApiProxyHttp(allowed_hosts=GMAIL_HOSTS),
            static_discovery=True,
        )
        batch = service.new_batch_http_request(
            callback=lambda request_id, response, exception: seen.append(
                (request_id, response, exception)
            )
        )
        for message_id in messages:
            batch.add(
                service.users()
                .messages()
                .get(userId="me", id=message_id, format="metadata")
            )
        google_api_proxy.execute_batch(
            batch, max_workers=max_workers, deadline=deadline
        )
        return "ok"

    return mcp, seen


@pytest.mark.asyncio
async def test_a_request_accepted_after_the_deadline_is_counted(timed_proxy):
    """The batch returns at its deadline, but a request it already sent can
    still be accepted: that late acceptance must reach the reported count."""
    timed_proxy.answers_after("m1", 0.0)
    timed_proxy.answers_after("m2", 0.6)
    mcp, seen = _batch_server(deadline=0.2)

    async with create_connected_server_and_client_session(mcp._mcp_server) as session:
        result = await session.call_tool("read_messages", {"messages": ["m1", "m2"]})

    assert result.isError is False
    assert result.meta == {COMPOSIO_REQUESTS_META_KEY: 2}
    # The tool's own answer is unchanged: the unanswered request timed out.
    assert [item[0] for item in seen] == ["1", "2"]
    assert seen[1][1] is None
    assert isinstance(seen[1][2], TimeoutError)


@pytest.mark.asyncio
async def test_a_request_refused_after_the_deadline_is_not_counted(timed_proxy):
    """A late answer that is a Composio refusal is not an accepted request."""
    timed_proxy.answers_after("m1", 0.0)
    timed_proxy.refuses_after("m2", 0.6)
    mcp, seen = _batch_server(deadline=0.2)

    async with create_connected_server_and_client_session(mcp._mcp_server) as session:
        result = await session.call_tool("read_messages", {"messages": ["m1", "m2"]})

    assert result.isError is False
    assert result.meta == {COMPOSIO_REQUESTS_META_KEY: 1}
    assert isinstance(seen[1][2], TimeoutError)


@pytest.mark.asyncio
async def test_a_request_that_never_started_is_not_waited_for(timed_proxy):
    """Behind one running request the rest of the batch is cancelled and never
    sent: those requests cannot be accepted, so the call neither waits for
    them nor counts them."""
    timed_proxy.answers_after("m1", 0.5)
    timed_proxy.answers_after("m2", 1.0)
    timed_proxy.answers_after("m3", 1.0)
    mcp, _ = _batch_server(deadline=0.2, max_workers=1)

    started = time.monotonic()
    async with create_connected_server_and_client_session(mcp._mcp_server) as session:
        result = await session.call_tool(
            "read_messages", {"messages": ["m1", "m2", "m3"]}
        )
    elapsed = time.monotonic() - started

    assert timed_proxy.calls == ["m1"]
    assert result.meta == {COMPOSIO_REQUESTS_META_KEY: 1}
    assert elapsed < 0.9, "the cancelled requests must not be waited for"
