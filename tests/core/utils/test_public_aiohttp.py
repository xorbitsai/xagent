import asyncio
import socket
from types import SimpleNamespace
from unittest.mock import Mock

import aiohttp
import pytest
from yarl import URL

from xagent.core.utils.public_aiohttp import (
    PublicTCPConnector,
    _check_redirect,
    create_public_aiohttp_session,
)
from xagent.core.utils.security import PrivateNetworkHostError


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "host", ["127.0.0.1", "192.168.1.10", "169.254.169.254", "::1", "::ffff:127.0.0.1"]
)
async def test_connector_literal_fast_path_is_guarded(host):
    async with PublicTCPConnector(use_dns_cache=False) as connector:
        with pytest.raises(PrivateNetworkHostError):
            await connector._resolve_host(host, 443)


@pytest.mark.asyncio
async def test_connector_validates_all_answers_and_returns_exact_checked_addresses():
    addresses = [
        {
            "hostname": "example.com",
            "host": "93.184.216.34",
            "port": 443,
            "family": socket.AF_INET,
            "proto": 0,
            "flags": 0,
        }
    ]

    class Resolver(aiohttp.abc.AbstractResolver):
        async def resolve(self, *args, **kwargs):
            return addresses

        async def close(self):
            pass

    async with PublicTCPConnector(
        resolver=Resolver(), use_dns_cache=False
    ) as connector:
        assert await connector._resolve_host("example.com", 443) == addresses
        addresses.append({**addresses[0], "host": "10.0.0.1"})
        with pytest.raises(PrivateNetworkHostError):
            await connector._resolve_host("example.com", 443)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://192.168.1.10:8080/",
        "https://user:secret@example.com/",
        "https://example.com:0/",
    ],
)
async def test_real_session_rejects_unsafe_inputs_before_connection(url):
    async with create_public_aiohttp_session() as session:
        with pytest.raises(ValueError):
            await session.get(url)
        assert session.trust_env is False


@pytest.mark.asyncio
async def test_real_session_cannot_use_per_request_proxy():
    async with create_public_aiohttp_session() as session:
        with pytest.raises(ValueError, match="proxy"):
            await session.get("https://example.com/", proxy="http://127.0.0.1:3128")


@pytest.mark.asyncio
async def test_cross_origin_redirect_is_rejected_without_replaying_credentials():
    params = SimpleNamespace(
        url=URL("https://example.com/v1"),
        response=SimpleNamespace(
            headers={"Location": "https://other.example/"}, release=Mock()
        ),
    )
    with pytest.raises(ValueError, match="Cross-origin"):
        await _check_redirect(None, SimpleNamespace(), params)
    params.response.headers["Location"] = "http://169.254.169.254/"
    with pytest.raises(PrivateNetworkHostError):
        await _check_redirect(None, SimpleNamespace(), params)
    params.response.headers["Location"] = "/v2"
    context = SimpleNamespace()
    for _ in range(5):
        await _check_redirect(None, context, params)
    with pytest.raises(ValueError, match="Too many"):
        await _check_redirect(None, context, params)
    assert params.response.release.call_count == 3


@pytest.mark.asyncio
async def test_real_session_rejects_private_resolver_answers_before_socket(monkeypatch):
    class Resolver(aiohttp.abc.AbstractResolver):
        async def resolve(self, host, port, family=socket.AF_UNSPEC):
            return [
                {
                    "hostname": host,
                    "host": "10.0.0.1",
                    "port": port,
                    "family": socket.AF_INET,
                    "proto": 0,
                    "flags": 0,
                }
            ]

        async def close(self):
            pass

    def no_socket(*args, **kwargs):
        pytest.fail("Private DNS answer reached the socket connector")

    monkeypatch.setattr("aiohappyeyeballs.start_connection", no_socket)
    async with create_public_aiohttp_session() as session:
        # Exercise the real session/request/connector path, not _resolve_host
        # in isolation. The owned connector's resolver is the only seam.
        session.connector._resolver = Resolver()
        with pytest.raises(PrivateNetworkHostError):
            await session.get("https://example.com/")


@pytest.mark.asyncio
async def test_owned_aiohttp_proxy_does_not_allow_private_origin_or_overrides(
    monkeypatch,
):
    original = socket.getaddrinfo

    def resolve(host, port, *args, **kwargs):
        if host == "public.example":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", port))]
        return original(host, port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    seen = []

    async def serve(reader, writer):
        try:
            seen.append(await reader.readuntil(b"\r\n\r\n"))
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok"
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    try:
        proxy = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        async with create_public_aiohttp_session(proxy_url=proxy) as session:
            async with session.get("http://public.example/") as response:
                assert await response.text() == "ok"
            with pytest.raises(PrivateNetworkHostError):
                await session.get("http://169.254.169.254/")
            with pytest.raises(ValueError, match="proxy"):
                await session.get("http://public.example/", proxy="http://127.0.0.1:1")
        assert len(seen) == 1
        assert seen[0].startswith(b"GET http://public.example/")
    finally:
        server.close()
        await server.wait_closed()
