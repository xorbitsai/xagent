"""MCP clients cannot bypass the public-IP or configured-origin boundary."""

import socket

import httpx
import pytest

from xagent.core.tools.core.mcp.http import create_mcp_http_client
from xagent.core.utils.public_http_transport import UnsafePublicHttpTarget


@pytest.fixture
def network(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )
    seen = []
    replies = []

    def handler(request):
        seen.append(request)
        return replies.pop(0) if replies else httpx.Response(200, json={})

    monkeypatch.setattr(
        httpx, "AsyncHTTPTransport", lambda **kwargs: httpx.MockTransport(handler)
    )
    return seen, replies


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target",
    ["http://192.168.1.10:8080/", "http://169.254.169.254/", "http://[::1]/"],
)
async def test_private_mcp_is_not_supported(target, network):
    seen, _ = network
    async with create_mcp_http_client(endpoint_url=target) as client:
        with pytest.raises(UnsafePublicHttpTarget):
            await client.get(target)
    assert not seen


@pytest.mark.asyncio
async def test_mcp_http_is_pinned_and_still_sends_credentials_to_configured_origin(
    network,
):
    seen, _ = network
    async with create_mcp_http_client(
        headers={"X-Key": "key"}, endpoint_url="https://example.com/mcp"
    ) as client:
        await client.post("https://example.com/messages", json={})
    assert seen[0].url.host == "93.184.216.34"
    assert seen[0].headers["Host"] == "example.com"
    assert seen[0].headers["X-Key"] == "key"
    assert seen[0].extensions["sni_hostname"] == "example.com"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target",
    [
        "https://other.example/messages",
        "http://example.com/messages",
        "https://example.com:8443/messages",
        "http://example.com:443/messages",
    ],
)
async def test_mcp_cannot_replay_credentials_across_origins(target, network):
    seen, replies = network
    replies.append(httpx.Response(307, headers={"Location": target}))
    async with create_mcp_http_client(endpoint_url="https://example.com/mcp") as client:
        # The SDK follows same-origin redirects itself. Exercise the real
        # transport's binding when HTTPX is explicitly asked to follow too.
        with pytest.raises(UnsafePublicHttpTarget, match="origin"):
            await client.post(
                "https://example.com/mcp", content=b"secret", follow_redirects=True
            )
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_mcp_http_to_https_upgrade_is_not_an_origin_exception(network):
    seen, replies = network
    replies.append(httpx.Response(307, headers={"Location": "https://example.com/mcp"}))
    async with create_mcp_http_client(endpoint_url="http://example.com/mcp") as client:
        with pytest.raises(UnsafePublicHttpTarget, match="origin"):
            await client.post(
                "http://example.com/mcp", content=b"secret", follow_redirects=True
            )
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_configured_idna_hostname_has_the_same_origin_as_httpx_request(network):
    seen, _ = network
    async with create_mcp_http_client(
        endpoint_url="https://b\u00fccher.example/mcp"
    ) as client:
        await client.get("https://xn--bcher-kva.example/messages")
    assert seen[0].headers["Host"] == "xn--bcher-kva.example"


@pytest.mark.asyncio
async def test_hostname_resolving_to_private_is_blocked(network, monkeypatch):
    seen, _ = network
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 443))],
    )
    async with create_mcp_http_client(endpoint_url="https://example.com/mcp") as client:
        with pytest.raises(UnsafePublicHttpTarget):
            await client.post("https://example.com/mcp", content=b"credentials")
    assert not seen


@pytest.mark.asyncio
async def test_trusted_proxy_is_forwarded_but_untrusted_ambient_proxy_is_not(
    network, monkeypatch
):
    from xagent.core.utils.security import PrivateNetworkHostError

    monkeypatch.setenv("HTTPS_PROXY", "http://trusted.example:3128")
    with pytest.raises(PrivateNetworkHostError):
        create_mcp_http_client(endpoint_url="https://example.com/mcp")
    monkeypatch.setenv("XAGENT_TRUSTED_EGRESS_PROXY", "1")
    options = []
    monkeypatch.setattr(
        httpx,
        "AsyncHTTPTransport",
        lambda **kwargs: options.append(kwargs)
        or httpx.MockTransport(lambda request: httpx.Response(200)),
    )
    async with create_mcp_http_client(endpoint_url="https://example.com/mcp") as client:
        await client.get("https://example.com/mcp")
    assert options[0]["proxy"] == "http://trusted.example:3128"
    assert options[0]["trust_env"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["sse", "streamable_http"])
async def test_default_session_factory_binds_configured_endpoint(monkeypatch, kind):
    from contextlib import asynccontextmanager

    from xagent.core.tools.core.mcp import sessions

    class Captured(Exception):
        pass

    @asynccontextmanager
    async def endpoint(*args, **kwargs):
        factory = kwargs["httpx_client_factory"]
        async with factory() as client:
            assert client._transport._origin == "https://example.com:443"
        raise Captured
        yield  # pragma: no cover

    monkeypatch.setattr(
        sessions, "sse_client" if kind == "sse" else "streamablehttp_client", endpoint
    )
    with pytest.raises(Captured):
        async with sessions.create_session(
            {"transport": kind, "url": "https://example.com/mcp"}
        ):
            pytest.fail("Stub unexpectedly created a session")
