"""Exercise both real crawler clients through the pinned egress boundary."""

import asyncio
import datetime
import logging
import ssl
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import urlsplit

import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from pydantic import ValidationError

from xagent.core.tools.core.RAG_tools.core.schemas import WebCrawlConfig
from xagent.core.tools.core.RAG_tools.web_crawler.crawler import WebCrawler
from xagent.core.tools.core.RAG_tools.web_crawler.url_filter import URLFilter
from xagent.core.utils.public_network_proxy import (
    PublicNetworkEgressProxy,
    PublicNetworkPolicyError,
)


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
        "https://127.0.0.1/metadata",
        "file:///etc/passwd",
        "ftp://public.example/file",
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
    monkeypatch.setenv("XAGENT_TRUSTED_EGRESS_PROXY", "false")
    connections = []
    seen = []

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
        seen.append(urlsplit(url).hostname)
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
            # Exercise the real clients' automatic redirects independently of
            # the crawler's additional upfront per-hop validation.
            client = sessions[fingerprint]
            try:
                if fingerprint:
                    response = await client.get(
                        "http://public.example", allow_redirects=True
                    )
                else:
                    response = await client.get(
                        "http://public.example", follow_redirects=True
                    )
            except Exception:
                # CONNECT policy errors and unsupported schemes are client
                # exceptions; below assertions prove which proxy hops ran.
                pass
            else:
                assert response.status_code == 403
        if target.startswith(("http://", "https://")):
            assert seen == ["public.example", urlsplit(target).hostname]
        else:
            assert seen == ["public.example"]
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


@pytest.mark.parametrize("url", ["https://u:p@example.com/", "http://@example.com/"])
def test_start_url_rejects_userinfo(url):
    with pytest.raises(ValidationError, match="embedded credentials"):
        WebCrawlConfig(start_url=url)


async def test_crawl_rejects_private_start_before_opening_sessions(monkeypatch):
    crawler = WebCrawler(WebCrawlConfig(start_url="http://127.0.0.1"))
    opened = MagicMock(side_effect=AssertionError("sessions opened"))
    monkeypatch.setattr(crawler, "_open_sessions", opened)
    with pytest.raises(ValueError, match="private network"):
        await crawler.crawl()
    opened.assert_not_called()


@pytest.mark.parametrize(
    "target", ["http://127.0.0.1/", "https://127.0.0.1/", "ftp://example.com/file"]
)
async def test_redirect_policy_failure_is_terminal(monkeypatch, target):
    async def validate(url):
        if url != "http://public.example":
            raise ValueError("target denied")
        return ["93.184.216.34"]

    monkeypatch.setattr(
        "xagent.core.tools.core.RAG_tools.web_crawler.crawler.validate_public_http_url",
        validate,
    )
    crawler = WebCrawler(WebCrawlConfig(start_url="http://public.example"))
    crawler._tls_chain = (None, "backup")
    first = AsyncMock()
    first.get.return_value = httpx.Response(302, headers={"Location": target})
    backup = AsyncMock()
    with pytest.raises(PublicNetworkPolicyError, match="target denied"):
        await crawler._fetch_with_fallback(
            {None: first, "backup": backup}, "http://public.example"
        )
    first.get.assert_awaited_once_with("http://public.example", follow_redirects=False)
    backup.get.assert_not_awaited()


@pytest.mark.parametrize("connect_error", [False, True])
@pytest.mark.parametrize(
    "url,hostname",
    [
        ("https://public.example", "public.example"),
        ("https://bücher.example", "xn--bcher-kva.example"),
        ("https://faß.example", "xn--fa-hia.example"),
    ],
)
async def test_proxy_rebinding_denial_is_terminal(
    monkeypatch, connect_error, url, hostname
):
    monkeypatch.setattr(
        "xagent.core.tools.core.RAG_tools.web_crawler.crawler.validate_public_http_url",
        AsyncMock(return_value=["93.184.216.34"]),
    )
    crawler = WebCrawler(WebCrawlConfig(start_url="https://public.example"))
    crawler._tls_chain = (None, "backup")
    crawler._egress_proxy = PublicNetworkEgressProxy()

    async def denied(*args, **kwargs):
        crawler._egress_proxy.denied_hosts.add(hostname)
        if connect_error:
            raise httpx.ProxyError("403")
        return httpx.Response(403)

    first, backup = AsyncMock(), AsyncMock()
    first.get.side_effect = denied
    with pytest.raises(PublicNetworkPolicyError):
        await crawler._fetch_with_fallback({None: first, "backup": backup}, url)
    first.get.assert_awaited_once_with(f"https://{hostname}", follow_redirects=False)
    backup.get.assert_not_awaited()


@pytest.mark.parametrize(
    "partial",
    [
        b"HTTP/1.1 200 OK\r\nContent-Length: 6\r\n\r\nab",
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n4\r\nab",
    ],
)
async def test_mid_response_timeout_never_appends_another_response(
    monkeypatch, partial
):
    proxy = PublicNetworkEgressProxy(response_timeout=0.02)
    remote = asyncio.StreamReader()
    remote.feed_data(partial)
    remote_writer = MagicMock(drain=AsyncMock(), wait_closed=AsyncMock())
    monkeypatch.setattr(proxy, "_resolve", AsyncMock(return_value=["93.184.216.34"]))
    monkeypatch.setattr(
        proxy, "_open_validated", AsyncMock(return_value=(remote, remote_writer))
    )
    request = asyncio.StreamReader()
    request.feed_data(
        b"GET http://public.example/ HTTP/1.1\r\nHost: public.example\r\n"
        + b"Proxy-Authorization: Basic "
        + proxy._auth_token
        + b"\r\n\r\n"
    )
    client = MagicMock(drain=AsyncMock(), wait_closed=AsyncMock())
    client.is_closing.return_value = False
    await proxy._handle_client(request, client)
    output = b"".join(call.args[0] for call in client.write.call_args_list)
    assert output.startswith(b"HTTP/1.1 200 OK\r\n")
    assert output.count(b"HTTP/1.1") == 1
    assert b"502" not in output


async def test_configured_proxy_timeouts_override_short_defaults(monkeypatch):
    proxy = PublicNetworkEgressProxy(connect_timeout=0.5, response_timeout=0.5)
    remote = asyncio.StreamReader()
    writer = MagicMock(drain=AsyncMock())

    async def delayed_response():
        await asyncio.sleep(0.04)
        remote.feed_data(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")

    task = asyncio.create_task(delayed_response())
    monkeypatch.setattr(
        "xagent.core.utils.public_network_proxy._PROXY_RESPONSE_TIMEOUT", 0.001
    )
    await proxy._relay_response(remote, writer)
    await task
    assert b"".join(call.args[0] for call in writer.write.call_args_list).endswith(
        b"ok"
    )


async def test_connect_timeout_uses_constructor_setting(monkeypatch):
    proxy = PublicNetworkEgressProxy(connect_timeout=0.02)

    async def slow_connect(*args):
        await asyncio.sleep(10)

    connect = AsyncMock(side_effect=slow_connect)
    monkeypatch.setattr(proxy, "_connect_address", connect)
    with pytest.raises(TimeoutError):
        await proxy._open_validated(["93.184.216.34"], 443)
    connect.assert_awaited_once_with("93.184.216.34", 443)


async def test_complete_chunked_response_does_not_wait_for_extra_terminator():
    proxy = PublicNetworkEgressProxy(response_timeout=0.02)
    remote = asyncio.StreamReader()
    body = b"2\r\nok\r\n0\r\nX-Trailer: yes\r\n\r\n"
    remote.feed_data(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n" + body)
    client = MagicMock(drain=AsyncMock())
    await proxy._relay_response(remote, client)
    assert b"".join(call.args[0] for call in client.write.call_args_list).endswith(body)


@pytest.mark.parametrize(
    "method,status", [("HEAD", 200), ("GET", 304), ("GET", 204), ("GET", 200)]
)
@pytest.mark.parametrize(
    "informational",
    [b"", b"HTTP/1.1 103 Early Hints\r\nLink: </style.css>; rel=preload\r\n\r\n"],
)
async def test_bodyless_response_finishes_without_waiting_for_origin_eof(
    monkeypatch, method, status, informational
):
    proxy = PublicNetworkEgressProxy()
    remote = asyncio.StreamReader()
    body = b"ok" if method == "GET" and status == 200 else b""
    length = 2 if body else 42
    remote.feed_data(
        informational
        + f"HTTP/1.1 {status} OK\r\nContent-Length: {length}\r\n\r\n".encode()
        + body
    )
    writer = MagicMock(drain=AsyncMock())
    remote_writer = MagicMock(drain=AsyncMock(), wait_closed=AsyncMock())
    monkeypatch.setattr(proxy, "_resolve", AsyncMock(return_value=["93.184.216.34"]))
    monkeypatch.setattr(
        proxy, "_open_validated", AsyncMock(return_value=(remote, remote_writer))
    )
    await asyncio.wait_for(
        proxy._handle_forward(
            asyncio.StreamReader(),
            writer,
            method,
            "http://public.example/",
            "HTTP/1.1",
            {},
        ),
        0.1,
    )
    output = b"".join(call.args[0] for call in writer.write.call_args_list)
    assert output.startswith(informational + f"HTTP/1.1 {status} OK\r\n".encode())
    assert output.endswith(b"\r\n\r\n" + body)


async def test_proxy_stops_accepting_before_cancelling_active_clients(monkeypatch):
    proxy = PublicNetworkEgressProxy()
    original_handler = proxy._handle_client
    accepted, cancelled = asyncio.Event(), []

    async def handle(reader, writer):
        accepted.set()
        try:
            await original_handler(reader, writer)
        finally:
            cancelled.append(proxy._server is not None and proxy._server.is_serving())

    monkeypatch.setattr(proxy, "_handle_client", handle)
    url = urlsplit(await proxy.start())
    _, client = await asyncio.open_connection(url.hostname, url.port)
    try:
        await asyncio.wait_for(accepted.wait(), 1)
        await asyncio.wait_for(proxy.close(), 1)
        assert cancelled == [False]
        assert not proxy._client_tasks
    finally:
        client.close()
        await client.wait_closed()
        await proxy.close()


async def test_late_proxy_handler_does_not_read_after_close():
    proxy = PublicNetworkEgressProxy()
    await proxy.close()
    reader = AsyncMock()
    writer = MagicMock(drain=AsyncMock(), wait_closed=AsyncMock())
    await proxy._handle_client(reader, writer)
    reader.readuntil.assert_not_awaited()
    writer.write.assert_not_called()
    writer.close.assert_called_once()


async def test_origin_reason_phrase_cannot_masquerade_as_local_policy(monkeypatch):
    monkeypatch.setattr(
        "xagent.core.tools.core.RAG_tools.web_crawler.crawler.validate_public_http_url",
        AsyncMock(return_value=["93.184.216.34"]),
    )
    crawler = WebCrawler(WebCrawlConfig(start_url="https://public.example"))
    crawler._tls_chain = (None, "backup")
    crawler._egress_proxy = PublicNetworkEgressProxy()
    first, backup = AsyncMock(), AsyncMock()
    first.get.return_value = httpx.Response(
        403, extensions={"reason_phrase": b"Public Network Policy Denied"}
    )
    backup.get.return_value = httpx.Response(404)
    result = await crawler._fetch_with_fallback(
        {None: first, "backup": backup}, "https://public.example"
    )
    assert result.response.status_code == 404
    backup.get.assert_awaited_once()


@pytest.mark.parametrize(
    "address,authority",
    [
        ("93.184.216.34", "93.184.216.34:80"),
        ("2606:4700:4700::1111", "[2606:4700:4700::1111]:80"),
    ],
)
async def test_trusted_upstream_tunnels_validated_ip_with_isolated_credentials(
    address, authority
):
    received = []

    async def upstream(reader, writer):
        try:
            received.append(await reader.readuntil(b"\r\n\r\n"))
            writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            await writer.drain()
            received.append(await reader.readuntil(b"\r\n\r\n"))
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(upstream, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    proxy = PublicNetworkEgressProxy(
        upstream_proxy_url=f"http://user:p%40ss@127.0.0.1:{port}"
    )
    proxy._resolve = AsyncMock(return_value=[address])
    local_url = await proxy.start()
    try:
        async with httpx.AsyncClient(proxy=local_url, trust_env=False) as client:
            response = await client.get("http://public.example/path")
        assert response.text == "ok"
        assert (
            received[0]
            == f"CONNECT {authority} HTTP/1.1\r\nHost: {authority}\r\nProxy-Authorization: Basic dXNlcjpwQHNz\r\n\r\n".encode()
        )
        assert b"host: public.example\r\n" in received[1].lower()
        assert received[1].startswith(b"GET /path HTTP/1.1\r\n")
        assert b"proxy-authorization" not in received[1].lower()
    finally:
        await proxy.close()
        server.close()
        await server.wait_closed()


async def test_sessions_share_gate_and_honor_ca_and_trusted_proxy(monkeypatch):
    from xagent.core.tools.core.RAG_tools.web_crawler import crawler as module

    monkeypatch.setenv("XAGENT_TRUSTED_EGRESS_PROXY", "true")
    monkeypatch.setenv("HTTPS_PROXY", "http://trusted.example:8080")
    context = ssl.create_default_context()
    monkeypatch.setattr(module, "build_ca_bundle_ssl_context", lambda: context)
    proxy = MagicMock(
        start=AsyncMock(return_value="http://local-gate:1234"), close=AsyncMock()
    )
    factory = MagicMock(return_value=proxy)
    monkeypatch.setattr(module, "PublicNetworkEgressProxy", factory)
    client = AsyncMock()
    client_factory = MagicMock(return_value=client)
    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    crawler = WebCrawler(WebCrawlConfig(start_url="https://example.com", timeout=37))
    load = MagicMock()
    monkeypatch.setattr(crawler.url_filter, "load_robots_txt", load)
    async with crawler._open_sessions(None):
        load.assert_called_once_with(
            "http://local-gate:1234", timeout=37, is_denied=proxy.is_denied
        )
        kwargs = client_factory.call_args.kwargs
        assert kwargs["proxy"] == "http://local-gate:1234"
        assert kwargs["trust_env"] is False
        assert kwargs["verify"] is context
    factory.assert_called_once_with(
        connect_timeout=37,
        response_timeout=37,
        upstream_proxy_url="http://trusted.example:8080",
    )
    proxy.close.assert_awaited_once()


def test_robots_policy_denial_stays_closed_and_preserves_ca(monkeypatch):
    from xagent.core.tools.core.RAG_tools.web_crawler import url_filter as module

    context = ssl.create_default_context()
    monkeypatch.setattr(module, "build_ca_bundle_ssl_context", lambda: context)
    client = MagicMock()
    client.return_value.__enter__.return_value.get.return_value = httpx.Response(
        403, request=httpx.Request("GET", "https://example.com/robots.txt")
    )
    monkeypatch.setattr(httpx, "Client", client)
    filter = URLFilter("https://example.com")
    filter.load_robots_txt(
        "http://local-gate:1234",
        timeout=37,
        is_denied=lambda url: url == "https://example.com/robots.txt",
    )
    assert not filter.is_allowed("https://example.com/docs")
    assert client.call_args.kwargs == dict(
        timeout=37,
        follow_redirects=True,
        max_redirects=5,
        proxy="http://local-gate:1234",
        trust_env=False,
        verify=context,
    )


def test_origin_robots_reason_phrase_is_not_authoritative(monkeypatch):
    client = MagicMock()
    client.return_value.__enter__.return_value.get.return_value = httpx.Response(
        404,
        request=httpx.Request("GET", "https://example.com/robots.txt"),
        extensions={"reason_phrase": b"Public Network Policy Denied"},
    )
    monkeypatch.setattr(httpx, "Client", client)
    filter = URLFilter("https://example.com")
    filter.load_robots_txt("http://local-gate:1234", is_denied=lambda url: False)
    assert filter.is_allowed("https://example.com/docs")


@pytest.mark.parametrize("fingerprint", [None, "test-curl"])
async def test_manual_redirect_relative_missing_location_and_limit(
    monkeypatch, fingerprint
):
    validate = AsyncMock(return_value=["93.184.216.34"])
    monkeypatch.setattr(
        "xagent.core.tools.core.RAG_tools.web_crawler.crawler.validate_public_http_url",
        validate,
    )
    crawler = WebCrawler(WebCrawlConfig(start_url="https://public.example/dir/start"))
    client = AsyncMock()
    client.get.side_effect = [
        httpx.Response(302, headers={"Location": "../next"}),
        httpx.Response(200),
    ]
    result = await crawler._get_public_response(
        client, fingerprint, crawler.config.start_url
    )
    assert result.status_code == 200
    assert [call.args[0] for call in client.get.await_args_list] == [
        "https://public.example/dir/start",
        "https://public.example/next",
    ]
    assert validate.await_count == 2
    client.get.reset_mock(side_effect=True)
    client.get.return_value = httpx.Response(302)
    assert (
        await crawler._get_public_response(
            client, fingerprint, crawler.config.start_url
        )
    ).status_code == 302
    client.get.assert_awaited_once()
    client.get.reset_mock()
    client.get.return_value = httpx.Response(302, headers={"Location": "/again"})
    with pytest.raises(ValueError, match="redirect limit"):
        await crawler._get_public_response(
            client, fingerprint, crawler.config.start_url
        )
    assert client.get.await_count == 6


async def test_page_policy_denial_has_explicit_warning_and_failed_reason(
    monkeypatch, caplog
):
    crawler = WebCrawler(WebCrawlConfig(start_url="https://public.example"))
    monkeypatch.setattr(
        crawler,
        "_fetch_with_fallback",
        AsyncMock(side_effect=PublicNetworkPolicyError("private target")),
    )
    with caplog.at_level(logging.WARNING):
        result, links = await crawler._crawl_page(
            {}, crawler.config.start_url, 0, asyncio.Semaphore(1)
        )
    assert result is None and not links
    assert crawler.failed_urls[crawler.config.start_url].startswith(
        "Blocked by public-network egress policy:"
    )
    assert "Unexpected error" not in caplog.text
    assert not any(record.levelno >= logging.ERROR for record in caplog.records)


@pytest.mark.parametrize("outcome", ["exception", "cancelled", "setup-error"])
async def test_proxy_and_sessions_close_on_setup_failure_or_interrupted_crawl(
    monkeypatch, outcome
):
    from xagent.core.tools.core.RAG_tools.web_crawler import crawler as module

    proxy = PublicNetworkEgressProxy()
    monkeypatch.setattr(module, "PublicNetworkEgressProxy", lambda **kwargs: proxy)
    client = AsyncMock()
    client.__aenter__.return_value = client
    if outcome == "setup-error":
        client.__aenter__.side_effect = RuntimeError("setup failed")
    monkeypatch.setattr(httpx, "AsyncClient", MagicMock(return_value=client))
    crawler = WebCrawler(
        WebCrawlConfig(start_url="https://example.com", respect_robots_txt=False)
    )
    ready = asyncio.Event()

    async def crawl():
        async with crawler._open_sessions(None):
            ready.set()
            if outcome == "exception":
                raise RuntimeError("crawl failed")
            await asyncio.Future()

    task = asyncio.create_task(crawl())
    if outcome == "cancelled":
        await asyncio.wait_for(ready.wait(), 1)
        task.cancel()
    with pytest.raises(
        asyncio.CancelledError if outcome == "cancelled" else RuntimeError
    ):
        await task
    assert proxy._server is None and not proxy._client_tasks
    assert crawler._egress_proxy is None
    if outcome != "setup-error":
        client.__aexit__.assert_awaited_once()


@pytest.mark.parametrize("fingerprint", [None, "chrome116"])
@pytest.mark.parametrize("upstream_scheme", ["http", "https"])
async def test_real_tls_through_trusted_chain_preserves_host_sni_and_ca(
    tmp_path, monkeypatch, fingerprint, upstream_scheme
):
    if fingerprint:
        pytest.importorskip("curl_cffi")
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "public.example")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName("public.example"), x509.DNSName("localhost")]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_path, key_path = tmp_path / "ca.pem", tmp_path / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    monkeypatch.setenv("SSL_CERT_FILE", str(cert_path))
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    # curl_cffi's supported CA override is separate from httpx's SSL_CERT_FILE.
    monkeypatch.setenv("CURL_CA_BUNDLE", str(cert_path))
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    monkeypatch.setenv("XAGENT_TRUSTED_EGRESS_PROXY", "true")
    sni, requests, tunnels = [], [], []
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.load_cert_chain(cert_path, key_path)
    tls.set_servername_callback(lambda sock, name, ctx: sni.append(name))

    async def origin(reader, writer):
        try:
            requests.append(await reader.readuntil(b"\r\n\r\n"))
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok"
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    origin_server = await asyncio.start_server(origin, "127.0.0.1", 0, ssl=tls)
    origin_port = origin_server.sockets[0].getsockname()[1]

    async def upstream(reader, writer):
        remote_writer = None
        try:
            tunnels.append(await reader.readuntil(b"\r\n\r\n"))
            remote_reader, remote_writer = await asyncio.open_connection(
                "127.0.0.1", origin_port
            )
            writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            await writer.drain()
            await PublicNetworkEgressProxy._bridge(
                reader, writer, remote_reader, remote_writer
            )
        finally:
            if remote_writer is not None:
                remote_writer.close()
                await remote_writer.wait_closed()
            writer.close()
            await writer.wait_closed()

    upstream_server = await asyncio.start_server(
        upstream, "127.0.0.1", 0, ssl=tls if upstream_scheme == "https" else None
    )
    upstream_port = upstream_server.sockets[0].getsockname()[1]
    monkeypatch.setenv("HTTPS_PROXY", f"{upstream_scheme}://localhost:{upstream_port}")
    monkeypatch.setattr(
        PublicNetworkEgressProxy, "_resolve", AsyncMock(return_value=["93.184.216.34"])
    )
    crawler = WebCrawler(
        WebCrawlConfig(
            start_url="https://public.example",
            respect_robots_txt=False,
            tls_impersonate=fingerprint,
        )
    )
    try:
        async with crawler._open_sessions(None) as sessions:
            response = await sessions[fingerprint].get(
                f"https://public.example:{origin_port}/test"
            )
            assert response.text == "ok"
        assert tunnels[0].startswith(
            f"CONNECT 93.184.216.34:{origin_port} HTTP/1.1\r\n".encode()
        )
        assert b"public.example" not in tunnels[0]
        assert "public.example" in sni
        if upstream_scheme == "https":
            assert "localhost" in sni
        assert f"host: public.example:{origin_port}\r\n".encode() in requests[0].lower()
    finally:
        upstream_server.close()
        origin_server.close()
        await upstream_server.wait_closed()
        await origin_server.wait_closed()
