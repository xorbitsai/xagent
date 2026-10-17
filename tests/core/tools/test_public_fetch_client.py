"""Real-client regressions for public fetches behind a corporate proxy."""

import asyncio
import socket
from contextlib import asynccontextmanager

import httpx
import pytest

from xagent.core.tools.core.web_content import public_fetch_client
from xagent.core.utils.public_network_proxy import PublicNetworkEgressProxy
from xagent.core.utils.security import fetch_public_http_bytes

PUBLIC_IP = "93.184.216.34"


@asynccontextmanager
async def upstream():
    seen = []
    writers = set()

    async def serve(reader, writer):
        writers.add(writer)
        try:
            connect = await reader.readuntil(b"\r\n\r\n")
            seen.append(connect)
            writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            await writer.drain()
            request = await reader.readuntil(b"\r\n\r\n")
            seen.append(request)
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            writers.discard(writer)

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    try:
        port = server.sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}", seen
    finally:
        server.close()
        for writer in list(writers):
            writer.close()
        await server.wait_closed()


def fake_dns(monkeypatch, addresses):
    original = socket.getaddrinfo
    calls = []

    def resolve(host, port, *args, **kwargs):
        if host != "public.example":
            return original(host, port, *args, **kwargs)
        calls.append(host)
        address = addresses[min(len(calls) - 1, len(addresses) - 1)]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    return calls


async def test_upstream_gets_checked_literal_ip_and_logical_host(monkeypatch):
    calls = fake_dns(monkeypatch, [PUBLIC_IP])
    async with upstream() as (proxy_url, seen):
        async with public_fetch_client(proxy_url) as client:
            response = await fetch_public_http_bytes(
                client,
                "http://public.example/data",
                max_content_bytes=10,
                timeout=2,
                via_proxy=True,
            )
    assert response.content == b"ok"
    assert len(calls) == 2
    assert seen[0].startswith(f"CONNECT {PUBLIC_IP}:80 HTTP/1.1\r\n".encode())
    assert b"CONNECT public.example" not in seen[0]
    assert b"host: public.example" in seen[1].lower()
    assert b"proxy-authorization" not in seen[1].lower()


async def test_rebinding_after_preflight_never_reaches_upstream(monkeypatch):
    calls = fake_dns(monkeypatch, [PUBLIC_IP, "169.254.169.254"])
    async with upstream() as (proxy_url, seen):
        async with public_fetch_client(proxy_url) as client:
            with pytest.raises(httpx.HTTPStatusError) as caught:
                await fetch_public_http_bytes(
                    client,
                    "http://public.example/data",
                    max_content_bytes=10,
                    timeout=2,
                    via_proxy=True,
                )
        assert seen == []
    assert len(calls) == 2
    assert "403" in str(caught.value)


async def test_gate_closes_on_cancellation(monkeypatch):
    closed = []
    original = PublicNetworkEgressProxy.close

    async def close(gate):
        await original(gate)
        closed.append(gate)

    monkeypatch.setattr(PublicNetworkEgressProxy, "close", close)
    with pytest.raises(asyncio.CancelledError):
        async with public_fetch_client("http://127.0.0.1:1"):
            raise asyncio.CancelledError()
    assert len(closed) == 1
    assert not closed[0]._client_tasks
    assert closed[0]._server is None


@pytest.mark.parametrize(
    "proxy_url", ["socks5://proxy.example", "http://proxy.example:0"]
)
async def test_invalid_proxy_has_no_direct_fallback(proxy_url):
    with pytest.raises(ValueError):
        async with public_fetch_client(proxy_url):
            pytest.fail("Invalid proxy must not yield a directly connected client")
