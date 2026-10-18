import asyncio
import socket

import anyio
import pytest

from xagent.core.tools.core.mcp import websocket as transport
from xagent.core.utils.security import PrivateNetworkHostError


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "ws://192.168.1.10:8080/",
        "wss://[::1]/",
        "ws://169.254.169.254/",
        "wss://user:secret@example.com/",
        "https://example.com/",
    ],
)
async def test_invalid_websocket_destinations_never_connect(url, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("a rejected MCP destination opened a socket")

    monkeypatch.setattr(transport, "_NoRedirectConnect", unexpected)
    with pytest.raises((PrivateNetworkHostError, ValueError)):
        async with transport.public_websocket_client(url):
            pytest.fail("rejected MCP destination was accepted")


@pytest.mark.asyncio
async def test_websocket_pins_ip_keeps_tls_name_and_preserves_protocol(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )
    options = []
    closed = []

    class FakeWebSocket:
        def __init__(self, url, **kwargs):
            options.append((url, kwargs))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            closed.append(True)

        async def __aiter__(self):
            yield '{"jsonrpc":"2.0","id":1,"result":{}}'
            await asyncio.Event().wait()

        async def send(self, message):
            pass

    monkeypatch.setattr(transport, "_NoRedirectConnect", FakeWebSocket)
    async with transport.public_websocket_client("wss://example.com:8443/mcp") as (
        read,
        _,
    ):
        message = await asyncio.wait_for(read.receive(), 1)
        assert message.message.root.id == 1
    url, kwargs = options[0]
    assert url == "wss://example.com:8443/mcp"
    assert kwargs["host"] == "93.184.216.34"
    assert kwargs["port"] == 8443
    assert kwargs["server_hostname"] == "example.com"
    assert kwargs["proxy"] is None
    assert kwargs["subprotocols"] == ["mcp"]
    assert closed == [True]


def test_handshake_redirect_is_never_followed():
    error = RuntimeError("redirect to private host")
    assert transport._NoRedirectConnect.process_redirect(None, error) is error


@pytest.mark.asyncio
async def test_peer_eof_closes_incoming_stream_without_waiting_for_outer_exit(
    monkeypatch,
):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )

    class ClosedPeer:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def __aiter__(self):
            if False:
                yield None

        async def send(self, message):
            pass

    monkeypatch.setattr(transport, "_NoRedirectConnect", ClosedPeer)
    async with transport.public_websocket_client("wss://example.com/") as (read, _):
        with pytest.raises(anyio.EndOfStream):
            await asyncio.wait_for(read.receive(), 1)
