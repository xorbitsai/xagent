"""Protocol, authentication and lifecycle regressions for the shared gate."""

import asyncio
import logging
import ssl
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import urlsplit

import pytest

from xagent.core.utils import public_network_proxy as module
from xagent.core.utils.public_network_proxy import (
    PublicNetworkEgressProxy,
    PublicNetworkPolicyError,
)


def test_upstream_gate_credentials_are_unique_per_instance():
    first = PublicNetworkEgressProxy(upstream_proxy_url="http://proxy.example")
    second = PublicNetworkEgressProxy(upstream_proxy_url="http://proxy.example")
    assert first._auth_token != second._auth_token
    assert first._username != second._username and first._password != second._password


@pytest.mark.parametrize(
    "method,target",
    [("GET", "http://public.example/"), ("CONNECT", "public.example:443")],
)
@pytest.mark.parametrize("auth", [b"", b"Basic wrong", b"Basic \xff"])
async def test_unauthenticated_clients_never_resolve_or_use_upstream(
    monkeypatch, method, target, auth
):
    proxy = PublicNetworkEgressProxy(
        upstream_proxy_url="http://corporate:secret@proxy.example"
    )
    resolve, connect = AsyncMock(), AsyncMock()
    monkeypatch.setattr(proxy, "_resolve", resolve)
    monkeypatch.setattr(proxy, "_open_validated", connect)
    url = urlsplit(await proxy.start())
    assert url.username and url.password
    reader, writer = await asyncio.open_connection(url.hostname, url.port)
    try:
        writer.write(
            f"{method} {target} HTTP/1.1\r\n".encode()
            + b"Proxy-Authorization: "
            + auth
            + b"\r\n\r\n"
        )
        await writer.drain()
        reply = await asyncio.wait_for(reader.read(), 1)
        assert reply.startswith(b"HTTP/1.1 407 ")
        assert b"Proxy-Authenticate: Basic " in reply
        assert b"corporate" not in reply and b"secret" not in reply
        resolve.assert_not_awaited()
        connect.assert_not_awaited()
    finally:
        writer.close()
        await writer.wait_closed()
        await asyncio.wait_for(proxy.close(), 1)


async def test_close_cancels_idle_connect_tunnel_without_waiting_for_peer(monkeypatch):
    proxy = PublicNetworkEgressProxy()
    remote = asyncio.StreamReader()
    remote_writer = MagicMock(drain=AsyncMock(), wait_closed=AsyncMock())
    remote_writer.is_closing.return_value = False
    monkeypatch.setattr(proxy, "_resolve", AsyncMock(return_value=["93.184.216.34"]))
    monkeypatch.setattr(
        proxy, "_open_validated", AsyncMock(return_value=(remote, remote_writer))
    )
    url = urlsplit(await proxy.start())
    reader, writer = await asyncio.open_connection(url.hostname, url.port)
    try:
        writer.write(
            b"CONNECT public.example:443 HTTP/1.1\r\nProxy-Authorization: Basic "
            + proxy._auth_token
            + b"\r\n\r\n"
        )
        await writer.drain()
        assert (await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 1)).startswith(
            b"HTTP/1.1 200 "
        )
        await asyncio.wait_for(proxy.close(), 1)
        assert not proxy._client_tasks
        assert await asyncio.wait_for(reader.read(), 1) == b""
        remote_writer.close.assert_called_once()
        remote_writer.wait_closed.assert_awaited_once()
    finally:
        writer.close()
        await writer.wait_closed()
        await proxy.close()


@pytest.mark.parametrize(
    "reply",
    [
        b"",
        b"HTTP/1.1 200",
        b"garbage\r\n\r\n",
        b"HTTP/1.1\r\n\r\n",
        b"HTTP/1.1 nope\r\n\r\n",
        b"HTTP/1.1 407 Refused\r\n\r\n",
    ],
)
@pytest.mark.parametrize("cleanup_error", [False, True])
async def test_upstream_bad_reply_falls_back_without_masking_error(
    monkeypatch, reply, cleanup_error
):
    first, second = asyncio.StreamReader(), asyncio.StreamReader()
    first.feed_data(reply)
    first.feed_eof()
    second.feed_data(b"HTTP/1.1 200 Connection Established\r\n\r\n")
    failed_writer = MagicMock(
        drain=AsyncMock(),
        wait_closed=AsyncMock(
            side_effect=OSError("cleanup") if cleanup_error else None
        ),
    )
    good_writer = MagicMock(drain=AsyncMock())
    open_connection = AsyncMock(
        side_effect=[(first, failed_writer), (second, good_writer)]
    )
    monkeypatch.setattr(module.asyncio, "open_connection", open_connection)
    proxy = PublicNetworkEgressProxy(upstream_proxy_url="http://proxy.example:8080")
    assert await proxy._open_validated(["93.184.216.34", "1.1.1.1"], 443) == (
        second,
        good_writer,
    )
    assert open_connection.await_count == 2
    assert failed_writer.write.call_args.args[0].startswith(
        b"CONNECT 93.184.216.34:443 "
    )
    assert good_writer.write.call_args.args[0].startswith(b"CONNECT 1.1.1.1:443 ")
    failed_writer.close.assert_called_once()


@pytest.mark.parametrize(
    "url",
    [
        "ftp://proxy.example",
        "socks5://proxy.example",
        "http://",
        "http:///path",
        "http://proxy.example/path",
        "http://proxy.example?query",
        "https://proxy.example#fragment",
        "http://proxy.example:invalid",
        "http://proxy.example:70000",
        "http://proxy.example:0",
    ],
)
def test_invalid_upstream_configuration_fails_before_start(url):
    with pytest.raises(ValueError):
        PublicNetworkEgressProxy(upstream_proxy_url=url)


async def test_https_upstream_reuses_one_ca_context(monkeypatch):
    context = ssl.create_default_context()
    build = MagicMock(return_value=context)
    monkeypatch.setattr(module, "build_ca_bundle_ssl_context", build)
    reader = MagicMock(readuntil=AsyncMock(return_value=b"HTTP/1.1 200 OK\r\n\r\n"))
    writer = MagicMock(drain=AsyncMock())
    connect = AsyncMock(return_value=(reader, writer))
    monkeypatch.setattr(module.asyncio, "open_connection", connect)
    proxy = PublicNetworkEgressProxy(upstream_proxy_url="https://proxy.example:8443")
    await proxy._open_validated(["1.1.1.1"], 443)
    await proxy._open_validated(["8.8.8.8"], 80)
    build.assert_called_once_with()
    assert all(call.kwargs["ssl"] is context for call in connect.await_args_list)


@pytest.mark.parametrize(
    "headers",
    [
        b"Content-Length: 1\r\nTransfer-Encoding: chunked",
        b"Transfer-Encoding: chunked\r\nContent-Length: 1",
        b"Transfer-Encoding: chunked\r\nContent-Length: invalid",
    ],
)
async def test_chunked_response_overrides_and_removes_content_length(headers):
    proxy = PublicNetworkEgressProxy(response_timeout=0.1)
    reader = asyncio.StreamReader()
    body = b"2\r\nok\r\n0\r\n\r\n"
    reader.feed_data(b"HTTP/1.1 200 OK\r\n" + headers + b"\r\n\r\n" + body)
    writer = MagicMock(drain=AsyncMock())
    await proxy._relay_response(reader, writer)
    output = b"".join(call.args[0] for call in writer.write.call_args_list)
    assert b"Content-Length" not in output
    assert b"Transfer-Encoding: chunked" in output
    assert output.endswith(body)


async def test_unused_preconnect_eof_is_debug_only(caplog):
    proxy = PublicNetworkEgressProxy()
    reader = asyncio.StreamReader()
    reader.feed_eof()
    writer = MagicMock(drain=AsyncMock(), wait_closed=AsyncMock())
    writer.is_closing.return_value = False
    with caplog.at_level(logging.DEBUG, logger=module.__name__):
        await proxy._handle_client(reader, writer)
    records = [r for r in caplog.records if r.name == module.__name__]
    assert records and all(r.levelno == logging.DEBUG for r in records)


async def test_policy_warning_contains_only_denied_hostname(caplog):
    proxy = PublicNetworkEgressProxy()
    with caplog.at_level(logging.WARNING, logger=module.__name__):
        with pytest.raises(PublicNetworkPolicyError):
            await proxy._resolve("http://127.0.0.1/private?token=secret")
    assert "127.0.0.1" in caplog.text
    assert "secret" not in caplog.text and "/private" not in caplog.text


async def test_real_browser_session_preserves_direct_egress_with_cdp_guards(
    monkeypatch,
):
    playwright = pytest.importorskip("playwright.async_api")
    from xagent.core.tools.core import browser_use

    async with playwright.async_playwright() as api:
        if not Path(api.chromium.executable_path).is_file():
            pytest.skip("Chromium is not installed")
        requests = []

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

        server = await asyncio.start_server(origin, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        proxy = PublicNetworkEgressProxy()

        async def resolve(url):
            if urlsplit(url).hostname != "public.example":
                raise PublicNetworkPolicyError("test only allows its origin")
            return ["93.184.216.34"]

        monkeypatch.setattr(proxy, "_resolve", resolve)
        monkeypatch.setattr(
            proxy,
            "_connect_address",
            lambda *args: asyncio.open_connection("127.0.0.1", port),
        )
        launch = api.chromium.launch

        async def launch_installed(**kwargs):
            return await launch(executable_path=api.chromium.executable_path, **kwargs)

        monkeypatch.setattr(api.chromium, "launch", launch_installed)
        monkeypatch.setattr(browser_use, "PublicNetworkEgressProxy", lambda: proxy)
        monkeypatch.setattr(browser_use, "validate_public_http_url", resolve)
        monkeypatch.setattr(
            browser_use,
            "async_playwright",
            lambda: SimpleNamespace(
                start=AsyncMock(
                    return_value=SimpleNamespace(
                        chromium=api.chromium, stop=AsyncMock()
                    )
                )
            ),
        )
        session = browser_use.BrowserSession("gate-auth-test", headless=True)
        try:
            # Direct browser gates have no upstream identity. Keep their
            # production CDP guards and existing proxy behavior unchanged.
            page = await session.get_page()
            response = await page.goto("http://public.example/", timeout=10000)
            assert response.status == 200
            assert await response.text() == "ok"
            assert requests and all(
                b"proxy-authorization" not in request.lower() for request in requests
            )
        finally:
            await session.close()
            await asyncio.wait_for(proxy.close(), 1)
            server.close()
            await server.wait_closed()
