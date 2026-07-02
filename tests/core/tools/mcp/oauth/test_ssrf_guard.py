"""Tests for the SSRF guard used before OAuth discovery/DCR/token-exchange
requests (see ``xagent.core.tools.core.mcp.oauth.ssrf_guard``).

DNS resolution is mocked via ``loop.getaddrinfo`` so these tests don't depend
on real network/DNS availability, except for the disallowed-IP-literal cases
where no DNS lookup is actually needed to resolve a bare IP address.
"""

import socket

import pytest

from xagent.core.tools.core.mcp.oauth.ssrf_guard import (
    UnsafeOAuthEndpointError,
    assert_public_endpoint,
)


def _addrinfo(ip: str, family=socket.AF_INET):
    return [(family, socket.SOCK_STREAM, 6, "", (ip, 0))]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/",  # cloud metadata / link-local
        "http://127.0.0.1/",  # loopback
        "http://10.0.0.5/",  # private
        "http://100.64.0.1/",  # CGNAT / shared address space (RFC 6598)
        "http://192.0.0.1/",  # IETF Protocol Assignments (non-global)
        "http://224.0.0.1/",  # multicast (note: is_global is True for this)
    ],
)
async def test_rejects_disallowed_ip_literals(url):
    with pytest.raises(UnsafeOAuthEndpointError):
        await assert_public_endpoint(url)


@pytest.mark.asyncio
async def test_rejects_localhost_hostname(monkeypatch):
    async def fake_getaddrinfo(host, port):
        assert host == "localhost"
        return _addrinfo("127.0.0.1")

    monkeypatch.setattr(
        "asyncio.get_event_loop",
        lambda: type("L", (), {"getaddrinfo": staticmethod(fake_getaddrinfo)})(),
    )

    with pytest.raises(UnsafeOAuthEndpointError):
        await assert_public_endpoint("http://localhost/")


@pytest.mark.asyncio
async def test_allows_public_looking_host(monkeypatch):
    async def fake_getaddrinfo(host, port):
        assert host == "auth.example.com"
        return _addrinfo("93.184.216.34")

    monkeypatch.setattr(
        "asyncio.get_event_loop",
        lambda: type("L", (), {"getaddrinfo": staticmethod(fake_getaddrinfo)})(),
    )

    await assert_public_endpoint("https://auth.example.com/token")


@pytest.mark.asyncio
async def test_rejects_unresolvable_scheme():
    with pytest.raises(UnsafeOAuthEndpointError):
        await assert_public_endpoint("ftp://example.com/")


@pytest.mark.asyncio
async def test_rejects_when_dns_fails(monkeypatch):
    async def fake_getaddrinfo(host, port):
        raise OSError("name resolution failed")

    monkeypatch.setattr(
        "asyncio.get_event_loop",
        lambda: type("L", (), {"getaddrinfo": staticmethod(fake_getaddrinfo)})(),
    )

    with pytest.raises(UnsafeOAuthEndpointError):
        await assert_public_endpoint("http://does-not-resolve.example/")
