"""Exercise both real crawler clients through the pinned egress boundary."""

import asyncio
from urllib.parse import urlsplit

import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import WebCrawlConfig
from xagent.core.tools.core.RAG_tools.web_crawler.crawler import WebCrawler
from xagent.core.tools.core.RAG_tools.web_crawler.url_filter import URLFilter
from xagent.core.utils.public_network_proxy import PublicNetworkEgressProxy


def test_constructing_filter_never_fetches_robots(monkeypatch):
    import httpx

    def forbidden(*args, **kwargs):
        raise AssertionError("constructor performed network I/O")

    monkeypatch.setattr(httpx, "Client", forbidden)
    filter = URLFilter("http://127.0.0.1/private")
    assert filter.robots_parser is not None
    assert not filter.is_allowed("http://127.0.0.1/private")


@pytest.mark.parametrize("fingerprint", [None, "chrome116"])
@pytest.mark.parametrize(
    "target",
    [
        "http://127.0.0.1/metadata",
        "http://169.254.169.254/latest/meta-data/",
        "file:///etc/passwd",
    ],
)
async def test_real_clients_cannot_follow_redirects_to_private_targets(
    monkeypatch,
    fingerprint,
    target,
):
    if fingerprint:
        pytest.importorskip("curl_cffi")
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("no_proxy", "*")
    monkeypatch.setenv("HTTP_PROXY", "http://untrusted-proxy.invalid:9")
    connections = []

    async def serve(reader, writer):
        try:
            await reader.readuntil(b"\r\n\r\n")
            writer.write(
                f"HTTP/1.1 302 Found\r\nLocation: {target}\r\n"
                "Content-Length: 0\r\nConnection: close\r\n\r\n".encode()
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    original_resolve = PublicNetworkEgressProxy._resolve

    async def resolve(self, url):
        if urlsplit(url).hostname == "public.example":
            return ["93.184.216.34"]
        return await original_resolve(self, url)

    async def open_validated(addresses, destination_port):
        connections.append((addresses, destination_port))
        assert addresses == ["93.184.216.34"]
        return await asyncio.open_connection("127.0.0.1", port)

    monkeypatch.setattr(PublicNetworkEgressProxy, "_resolve", resolve)
    monkeypatch.setattr(
        PublicNetworkEgressProxy, "_open_validated", staticmethod(open_validated)
    )
    crawler = WebCrawler(
        WebCrawlConfig(
            start_url="http://public.example",
            tls_impersonate=fingerprint,
            respect_robots_txt=False,
            timeout=3,
        )
    )
    try:
        async with crawler._open_sessions(None) as sessions:
            result = await crawler._fetch_with_fallback(
                sessions, "http://public.example"
            )
        assert result.fingerprint_used is None
        assert len(connections) == 1
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/file",
        "http://user:password@example.com/",
    ],
)
async def test_unsupported_crawler_urls_never_reach_a_client(url):
    crawler = WebCrawler(
        WebCrawlConfig(
            start_url="https://example.com",
            respect_robots_txt=False,
        )
    )
    with pytest.raises(ValueError, match="only accepts HTTP"):
        await crawler._fetch_with_fallback({}, url)
