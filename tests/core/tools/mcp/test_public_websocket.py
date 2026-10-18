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
@pytest.mark.parametrize(
    "configured_url, expected_url, hostname",
    [
        ("wss://example.com:8443/mcp", "wss://example.com:8443/mcp", "example.com"),
        (
            "wss://faß.example:8443/mcp",
            "wss://xn--fa-hia.example:8443/mcp",
            "xn--fa-hia.example",
        ),
    ],
)
async def test_websocket_pins_ip_keeps_tls_name_and_preserves_protocol(
    monkeypatch, configured_url, expected_url, hostname
):
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
    async with transport.public_websocket_client(configured_url) as (
        read,
        _,
    ):
        message = await asyncio.wait_for(read.receive(), 1)
        assert message.message.root.id == 1
    url, kwargs = options[0]
    assert url == expected_url
    assert kwargs["host"] == "93.184.216.34"
    assert kwargs["port"] == 8443
    assert kwargs["server_hostname"] == hostname
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


@pytest.mark.asyncio
async def test_websocket_address_fallback_has_one_budget(monkeypatch):
    attempts = []

    class Connection:
        def __init__(self, url, **kwargs):
            attempts.append(kwargs)

        async def __aenter__(self):
            if len(attempts) == 1:
                raise OSError("unreachable")
            return self

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(transport, "_NoRedirectConnect", Connection)
    async with transport._connect_public(
        "wss://example.com/", ["2606:4700:4700::1111", "1.1.1.1"], 443, {}
    ):
        pass
    assert [item["host"] for item in attempts] == ["2606:4700:4700::1111", "1.1.1.1"]
    assert 0 < attempts[0]["open_timeout"] <= 5
    assert attempts[1]["open_timeout"] <= 10


@pytest.mark.asyncio
async def test_trusted_websocket_proxy_uses_owned_gate_and_closes_it(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://trusted.example:3128")
    monkeypatch.setenv("XAGENT_TRUSTED_EGRESS_PROXY", "1")
    seen = []
    closed = []

    class Gate:
        def __init__(self, **kwargs):
            assert kwargs["upstream_proxy_url"] == "http://trusted.example:3128"

        async def start(self):
            return "http://owned:password@127.0.0.1:1234"

        async def close(self):
            closed.append(True)

    class Connection:
        def __init__(self, url, **kwargs):
            seen.append(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(transport, "PublicNetworkEgressProxy", Gate)
    monkeypatch.setattr(transport, "_NoRedirectConnect", Connection)
    async with transport._connect_public("wss://example.com/", ["1.1.1.1"], 443, {}):
        assert not closed
    assert seen[0]["proxy"] == "http://owned:password@127.0.0.1:1234"
    assert "host" not in seen[0]
    assert closed == [True]
