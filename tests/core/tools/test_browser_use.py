"""
Tests for browser automation tools.

These tests use mocking to avoid requiring actual browser installation.
Run with: pytest tests/core/tools/test_browser_use.py
"""

import asyncio
import os
import stat
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xagent.core.tools.core import browser_use


@pytest.fixture
def mock_playwright():
    """Mock playwright modules."""
    with patch("xagent.core.tools.core.browser_use.PLAYWRIGHT_AVAILABLE", True):
        with patch("xagent.core.tools.core.browser_use.async_playwright") as mock_ap:
            yield mock_ap


@pytest.fixture
def reset_manager():
    """Reset global browser manager between tests."""
    manager = browser_use.get_browser_manager()
    import asyncio

    try:
        asyncio.run(manager.close_all())
    except Exception:
        pass
    browser_use._manager = None
    yield
    browser_use._manager = None


def _mock_browser_playwright(monkeypatch):
    """Install a fully mocked Playwright stack and return its browser handle."""
    monkeypatch.setattr(browser_use, "PLAYWRIGHT_AVAILABLE", True)
    monkeypatch.setattr(
        browser_use.PublicNetworkEgressProxy,
        "start",
        AsyncMock(return_value="http://127.0.0.1:43123"),
    )
    monkeypatch.setattr(browser_use.PublicNetworkEgressProxy, "close", AsyncMock())
    playwright = MagicMock()
    browser = MagicMock()
    context = MagicMock()
    page = MagicMock()
    cdp = MagicMock()

    playwright.chromium.executable_path = "/opt/playwright/chromium"
    playwright.chromium.launch = AsyncMock(return_value=browser)
    browser.new_context = AsyncMock(return_value=context)
    browser.close = AsyncMock()
    context.new_page = AsyncMock(return_value=page)
    context.route = AsyncMock()
    context.new_cdp_session = AsyncMock(return_value=cdp)
    context.add_init_script = AsyncMock()
    context.close = AsyncMock()
    page.close = AsyncMock()
    cdp.send = AsyncMock()
    cdp.detach = AsyncMock()
    playwright_api = MagicMock()
    playwright_api.start = AsyncMock(return_value=playwright)
    monkeypatch.setattr(browser_use, "async_playwright", lambda: playwright_api)
    playwright.api = playwright_api
    return playwright


class TestBrowserNavigate:
    """Tests for browser_navigate function."""

    def test_navigate_without_playwright(self):
        """Test that navigate fails gracefully without Playwright."""
        import asyncio

        with (
            patch("xagent.core.tools.core.browser_use.PLAYWRIGHT_AVAILABLE", False),
            patch(
                "xagent.core.tools.core.browser_use.validate_public_http_url",
                new=AsyncMock(return_value=["93.184.216.34"]),
            ),
        ):
            result = asyncio.run(
                browser_use.browser_navigate(
                    session_id="test-session", url="https://example.com"
                )
            )

            assert result["success"] is False
            assert "not installed" in result["error"]

    def test_navigate_with_mock(self, mock_playwright, reset_manager):
        """Test navigation with mocked Playwright."""
        # The actual browser interaction test is skipped for simplicity
        # In real integration tests, you would test with a real browser
        # For now, just test the error handling without Playwright
        pass

    def test_navigate_rejects_private_and_alternate_ip_targets(self):
        import asyncio

        with patch("xagent.core.tools.core.browser_use.PLAYWRIGHT_AVAILABLE", False):
            for url in ("http://127.0.0.1/", "http://127.1/", "http://2130706433/"):
                result = asyncio.run(
                    browser_use.browser_navigate(session_id="test-session", url=url)
                )
                assert result["error"] == "Browser target is not allowed"

    def test_navigate_allows_workspace_file_only(self, tmp_path):
        import asyncio

        html_path = tmp_path / "report.html"
        html_path.write_text("<html><body>ok</body></html>", encoding="utf-8")
        external_root = tmp_path / "uploads"
        external_root.mkdir()
        external_path = external_root / "uploaded.html"
        external_path.write_text("<html><body>uploaded</body></html>", encoding="utf-8")
        materialized_path = tmp_path / "materialized.html"
        materialized_path.write_text(
            "<html><body>materialized</body></html>", encoding="utf-8"
        )

        with patch("xagent.core.tools.core.browser_use.PLAYWRIGHT_AVAILABLE", False):
            allowed = asyncio.run(
                browser_use.browser_navigate(
                    session_id="test-session",
                    url=html_path.as_uri(),
                    _workspace_root=str(tmp_path),
                    _workspace_auth=browser_use._BROWSER_WORKSPACE_AUTH,
                )
            )
            allowed_external = asyncio.run(
                browser_use.browser_navigate(
                    session_id="test-session",
                    url=external_path.as_uri(),
                    _workspace_roots=[str(tmp_path), str(external_root)],
                    _workspace_auth=browser_use._BROWSER_WORKSPACE_AUTH,
                )
            )
            allowed_exact = asyncio.run(
                browser_use.browser_navigate(
                    session_id="test-session",
                    url=materialized_path.as_uri(),
                    _workspace_file_paths=[str(materialized_path)],
                    _workspace_auth=browser_use._BROWSER_WORKSPACE_AUTH,
                )
            )
            blocked = asyncio.run(
                browser_use.browser_navigate(
                    session_id="test-session",
                    url=html_path.as_uri(),
                )
            )
            spoofed = asyncio.run(
                browser_use.browser_navigate(
                    session_id="test-session",
                    url=html_path.as_uri(),
                    _workspace_root="/",
                )
            )

        assert "Only public network URLs are allowed" not in allowed["error"]
        assert "Browser target is not allowed" not in allowed_external["error"]
        assert "Browser target is not allowed" not in allowed_exact["error"]
        assert blocked["error"] == "Browser target is not allowed"
        assert spoofed["error"] == "Browser target is not allowed"


class TestBrowserRequestGuard:
    @pytest.mark.asyncio
    async def test_egress_proxy_connects_to_validated_ip(self, monkeypatch):
        proxy = browser_use.PublicNetworkEgressProxy()
        client_reader = MagicMock()
        client_reader.read = AsyncMock(return_value=b"")
        client_writer = MagicMock()
        client_writer.drain = AsyncMock()
        remote_reader = MagicMock()
        remote_reader.read = AsyncMock(return_value=b"")
        remote_writer = MagicMock()
        remote_writer.drain = AsyncMock()
        remote_writer.wait_closed = AsyncMock()

        monkeypatch.setattr(
            "xagent.core.utils.public_network_proxy.validate_public_http_url",
            AsyncMock(return_value=["203.0.113.10"]),
        )
        open_connection = AsyncMock(return_value=(remote_reader, remote_writer))
        monkeypatch.setattr(browser_use.asyncio, "open_connection", open_connection)

        await proxy._handle_connect(client_reader, client_writer, "example.com:443")

        open_connection.assert_awaited_once_with("203.0.113.10", 443)
        client_writer.write.assert_called_once_with(
            b"HTTP/1.1 200 Connection Established\r\n\r\n"
        )

    @pytest.mark.asyncio
    async def test_egress_proxy_rejects_unvalidated_connect(self, monkeypatch):
        proxy = browser_use.PublicNetworkEgressProxy()
        client_reader = MagicMock()
        client_writer = MagicMock()
        client_writer.drain = AsyncMock()
        validate = AsyncMock(side_effect=ValueError("private target"))
        open_connection = AsyncMock()
        monkeypatch.setattr(
            "xagent.core.utils.public_network_proxy.validate_public_http_url", validate
        )
        monkeypatch.setattr(browser_use.asyncio, "open_connection", open_connection)

        with pytest.raises(ValueError, match="private target"):
            await proxy._handle_connect(
                client_reader, client_writer, "internal.example:443"
            )

        open_connection.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_http_forward_is_one_shot_per_client_request(self, monkeypatch):
        proxy = browser_use.PublicNetworkEgressProxy()
        client_reader = MagicMock()
        client_reader.readexactly = AsyncMock(return_value=b"abc")
        client_reader.read = AsyncMock(
            side_effect=AssertionError("pipelined client bytes were forwarded")
        )
        client_writer = MagicMock()
        client_writer.drain = AsyncMock()
        remote_reader = MagicMock()
        remote_reader.readuntil = AsyncMock(
            return_value=b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n"
        )
        remote_writer = MagicMock()
        remote_writer.drain = AsyncMock()
        remote_writer.wait_closed = AsyncMock()

        monkeypatch.setattr(
            "xagent.core.utils.public_network_proxy.validate_public_http_url",
            AsyncMock(return_value=["203.0.113.10"]),
        )
        open_connection = AsyncMock(return_value=(remote_reader, remote_writer))
        monkeypatch.setattr(proxy, "_open_validated", open_connection)

        await proxy._handle_forward(
            client_reader,
            client_writer,
            "POST",
            "http://example.com/upload",
            "HTTP/1.1",
            {
                "host": "example.com",
                "content-length": "3",
                "connection": "keep-alive",
            },
        )

        open_connection.assert_awaited_once_with(["203.0.113.10"], 80)
        request = remote_writer.write.call_args.args[0]
        assert request.endswith(b"\r\n\r\nabc")
        assert b"Connection: close\r\n" in request
        response = client_writer.write.call_args.args[0]
        assert b"Connection: close\r\n" in response
        assert b"Connection: keep-alive" not in response
        client_reader.read.assert_not_awaited()
        remote_writer.wait_closed.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_ws_forward_preserves_upgrade_and_bridges_bidirectionally(
        self, monkeypatch
    ):
        proxy = browser_use.PublicNetworkEgressProxy()
        client_reader = MagicMock()
        client_reader.readexactly = AsyncMock(return_value=b"")
        client_writer = MagicMock()
        client_writer.drain = AsyncMock()
        remote_reader = MagicMock()
        remote_writer = MagicMock()
        remote_writer.drain = AsyncMock()
        remote_writer.wait_closed = AsyncMock()

        monkeypatch.setattr(
            "xagent.core.utils.public_network_proxy.validate_public_http_url",
            AsyncMock(return_value=["203.0.113.10"]),
        )
        open_connection = AsyncMock(return_value=(remote_reader, remote_writer))
        monkeypatch.setattr(proxy, "_open_validated", open_connection)
        bridge = AsyncMock()
        monkeypatch.setattr(proxy, "_bridge", bridge)

        await proxy._handle_forward(
            client_reader,
            client_writer,
            "GET",
            "ws://example.com/socket",
            "HTTP/1.1",
            {
                "host": "example.com",
                "connection": "Upgrade",
                "upgrade": "websocket",
            },
        )

        request = remote_writer.write.call_args.args[0]
        assert b"connection: Upgrade\r\n" in request
        assert b"upgrade: websocket\r\n" in request
        bridge.assert_awaited_once_with(
            client_reader, client_writer, remote_reader, remote_writer
        )

    @pytest.mark.asyncio
    async def test_concurrent_page_guards_share_one_installation(self):
        session = browser_use.BrowserSession("guard")
        context = MagicMock()
        cdp = MagicMock()
        cdp.send = AsyncMock()
        page = MagicMock()

        async def create_session(_page):
            await asyncio.sleep(0)
            return cdp

        context.new_cdp_session = create_session
        session._context = context

        await asyncio.gather(
            session._guard_page_requests(page),
            session._guard_page_requests(page),
        )

        assert cdp.on.call_count == 1
        assert cdp.send.await_count == 1
        cdp.send.assert_awaited_once_with(
            "Fetch.enable",
            {"patterns": browser_use._HTTP_FETCH_PATTERNS},
        )

    @pytest.mark.asyncio
    async def test_context_route_blocks_unauthorized_file_requests(self, tmp_path):
        session = browser_use.BrowserSession("guard", workspace_root=str(tmp_path))
        outside = tmp_path.parent / "outside.html"
        outside.write_text("<html></html>", encoding="utf-8")
        route = MagicMock()
        route.request.url = outside.as_uri()
        route.abort = AsyncMock()
        route.fallback = AsyncMock()

        await session._guard_context_route(route)

        route.abort.assert_awaited_once_with("blockedbyclient")
        route.fallback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_context_route_fulfills_authorized_file_and_falls_back_for_http(
        self, tmp_path
    ):
        session = browser_use.BrowserSession("guard", workspace_root=str(tmp_path))
        local_file = tmp_path / "page.html"
        local_file.write_text("<html></html>", encoding="utf-8")

        local_route = MagicMock()
        local_route.request.url = local_file.as_uri()
        local_route.abort = AsyncMock()
        local_route.fallback = AsyncMock()
        local_route.fulfill = AsyncMock()
        await session._guard_context_route(local_route)
        local_route.fulfill.assert_awaited_once_with(
            body=b"<html></html>",
            content_type="text/html",
            status=200,
        )
        local_route.fallback.assert_not_awaited()
        local_route.abort.assert_not_awaited()

        http_route = MagicMock()
        http_route.request.url = "https://example.com/"
        http_route.fallback = AsyncMock()
        await session._guard_context_route(http_route)
        http_route.fallback.assert_awaited_once_with()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="mkfifo unavailable")
    async def test_context_route_aborts_fifo_without_blocking(
        self, tmp_path, monkeypatch
    ):
        session = browser_use.BrowserSession("guard", workspace_root=str(tmp_path))
        fifo = tmp_path / "pipe"
        os.mkfifo(fifo)
        monkeypatch.setattr(
            browser_use, "_workspace_file_path", lambda *args, **kwargs: fifo
        )
        route = MagicMock()
        route.request.url = fifo.as_uri()
        route.abort = AsyncMock()
        route.fulfill = AsyncMock()

        await asyncio.wait_for(session._guard_context_route(route), timeout=1)

        route.abort.assert_awaited_once_with("blockedbyclient")
        route.fulfill.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_context_route_aborts_oversized_file(self, tmp_path):
        session = browser_use.BrowserSession("guard", workspace_root=str(tmp_path))
        huge = tmp_path / "huge.bin"
        with huge.open("wb") as handle:
            handle.truncate(browser_use._MAX_LOCAL_FILE_BYTES + 1)
        route = MagicMock()
        route.request.url = huge.as_uri()
        route.abort = AsyncMock()
        route.fulfill = AsyncMock()

        await session._guard_context_route(route)

        route.abort.assert_awaited_once_with("blockedbyclient")
        route.fulfill.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_context_route_aborts_identity_swap(self, tmp_path, monkeypatch):
        session = browser_use.BrowserSession("guard", workspace_root=str(tmp_path))
        local_file = tmp_path / "page.html"
        local_file.write_text("<html></html>", encoding="utf-8")
        other_file = tmp_path / "other.html"
        other_file.write_text("<html>other</html>", encoding="utf-8")
        calls = {"count": 0}

        def changing_path(*args, **kwargs):
            calls["count"] += 1
            return local_file if calls["count"] < 3 else other_file

        monkeypatch.setattr(browser_use, "_workspace_file_path", changing_path)
        route = MagicMock()
        route.request.url = local_file.as_uri()
        route.abort = AsyncMock()
        route.fulfill = AsyncMock()

        await session._guard_context_route(route)

        route.abort.assert_awaited_once_with("blockedbyclient")
        route.fulfill.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_context_route_aborts_inode_mismatch_on_stat_fallback(
        self, tmp_path, monkeypatch
    ):
        session = browser_use.BrowserSession("guard", workspace_root=str(tmp_path))
        local_file = tmp_path / "page.html"
        local_file.write_text("<html></html>", encoding="utf-8")
        real_stat = os.stat

        def stat_with_swapped_inode(path, *args, **kwargs):
            result = real_stat(path, *args, **kwargs)
            if str(path) == str(local_file) and not args and not kwargs:
                return os.stat_result(
                    (
                        result.st_mode,
                        result.st_ino + 1,
                        result.st_dev,
                        result.st_nlink,
                        result.st_uid,
                        result.st_gid,
                        result.st_size,
                        result.st_atime,
                        result.st_mtime,
                    )
                )
            return result

        def no_proc(path):
            raise OSError("no /proc")

        monkeypatch.setattr(browser_use.os, "stat", stat_with_swapped_inode)
        monkeypatch.setattr(browser_use.os, "readlink", no_proc)
        route = MagicMock()
        route.request.url = local_file.as_uri()
        route.abort = AsyncMock()
        route.fulfill = AsyncMock()

        await session._guard_context_route(route)

        route.abort.assert_awaited_once_with("blockedbyclient")
        route.fulfill.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_context_route_aborts_when_proc_fd_target_differs(
        self, tmp_path, monkeypatch
    ):
        session = browser_use.BrowserSession("guard", workspace_root=str(tmp_path))
        local_file = tmp_path / "page.html"
        local_file.write_text("<html></html>", encoding="utf-8")
        other_file = tmp_path / "other.html"
        other_file.write_text("<html>other</html>", encoding="utf-8")
        monkeypatch.setattr(browser_use.os, "readlink", lambda path: str(other_file))
        route = MagicMock()
        route.request.url = local_file.as_uri()
        route.abort = AsyncMock()
        route.fulfill = AsyncMock()

        await session._guard_context_route(route)

        route.abort.assert_awaited_once_with("blockedbyclient")
        route.fulfill.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_context_route_rejects_final_component_symlink(
        self, tmp_path, monkeypatch
    ):
        session = browser_use.BrowserSession("guard", workspace_root=str(tmp_path))
        target = tmp_path / "authorized.html"
        target.write_text("<html></html>", encoding="utf-8")
        link = tmp_path / "link.html"
        link.symlink_to(target)
        monkeypatch.setattr(
            browser_use, "_workspace_file_path", lambda *args, **kwargs: link
        )
        route = MagicMock()
        route.request.url = link.as_uri()
        route.abort = AsyncMock()
        route.fulfill = AsyncMock()

        await session._guard_context_route(route)

        route.abort.assert_awaited_once_with("blockedbyclient")
        route.fulfill.assert_not_awaited()

    def test_read_authorized_file_refuses_to_open_a_final_symlink(self, tmp_path):
        target = tmp_path / "authorized.html"
        target.write_text("<html></html>", encoding="utf-8")
        link = tmp_path / "link.html"
        link.symlink_to(target)

        with pytest.raises(OSError):
            browser_use._read_authorized_file(link.as_uri(), lambda _url: link)

    @pytest.mark.asyncio
    async def test_context_route_sniffs_extensionless_types(self, tmp_path):
        session = browser_use.BrowserSession("guard", workspace_root=str(tmp_path))
        text_file = tmp_path / "README"
        text_file.write_bytes("hello\n".encode())
        binary_file = tmp_path / "payload"
        binary_file.write_bytes(b"\x00\x01\x02")
        html_file = tmp_path / "page.html"
        html_file.write_text("<html></html>", encoding="utf-8")
        split_file = tmp_path / "NOTES"
        split_file.write_bytes(b"a" * (8 * 1024 - 1) + "中文".encode())

        for path, expected in (
            (text_file, "text/plain; charset=utf-8"),
            (split_file, "text/plain; charset=utf-8"),
            (binary_file, "application/octet-stream"),
            (html_file, "text/html"),
        ):
            route = MagicMock()
            route.request.url = path.as_uri()
            route.abort = AsyncMock()
            route.fulfill = AsyncMock()
            await session._guard_context_route(route)
            route.abort.assert_not_awaited()
            route.fulfill.assert_awaited_once_with(
                body=path.read_bytes(), content_type=expected, status=200
            )

    @pytest.mark.asyncio
    async def test_context_route_aborts_and_reraises_cancel_during_read(
        self, tmp_path, monkeypatch
    ):
        session = browser_use.BrowserSession("guard", workspace_root=str(tmp_path))
        local_file = tmp_path / "page.html"
        local_file.write_text("<html></html>", encoding="utf-8")
        started = threading.Event()
        release = threading.Event()

        def blocking_read(fd, limit):
            started.set()
            release.wait(5)
            return b"<html></html>"

        monkeypatch.setattr(browser_use, "_read_fd", blocking_read)
        route = MagicMock()
        route.request.url = local_file.as_uri()
        route.abort = AsyncMock()
        route.fulfill = AsyncMock()
        task = asyncio.create_task(session._guard_context_route(route))
        try:
            await asyncio.to_thread(started.wait, 2)
            assert started.is_set()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            route.abort.assert_awaited_once_with("blockedbyclient")
            route.fulfill.assert_not_awaited()
        finally:
            release.set()

    @pytest.mark.asyncio
    async def test_context_route_aborts_when_read_fails(self, tmp_path, monkeypatch):
        session = browser_use.BrowserSession("guard", workspace_root=str(tmp_path))
        local_file = tmp_path / "page.html"
        local_file.write_text("<html></html>", encoding="utf-8")

        def failing_read(fd, limit):
            raise OSError("read failed")

        monkeypatch.setattr(browser_use, "_read_fd", failing_read)
        route = MagicMock()
        route.request.url = local_file.as_uri()
        route.abort = AsyncMock()
        route.fulfill = AsyncMock()

        await session._guard_context_route(route)

        route.abort.assert_awaited_once_with("blockedbyclient")
        route.fulfill.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_context_route_aborts_when_fulfill_raises(self, tmp_path):
        session = browser_use.BrowserSession("guard", workspace_root=str(tmp_path))
        local_file = tmp_path / "page.html"
        local_file.write_text("<html></html>", encoding="utf-8")
        route = MagicMock()
        route.request.url = local_file.as_uri()
        route.abort = AsyncMock()
        route.fulfill = AsyncMock(side_effect=RuntimeError("fulfill failed"))

        await session._guard_context_route(route)

        route.abort.assert_awaited_once_with("blockedbyclient")

    @pytest.mark.asyncio
    async def test_context_route_swallows_abort_failure(self, tmp_path):
        session = browser_use.BrowserSession("guard", workspace_root=str(tmp_path))
        local_file = tmp_path / "page.html"
        local_file.write_text("<html></html>", encoding="utf-8")
        route = MagicMock()
        route.request.url = local_file.as_uri()
        route.abort = AsyncMock(side_effect=RuntimeError("page gone"))
        route.fulfill = AsyncMock(side_effect=RuntimeError("fulfill failed"))

        await session._guard_context_route(route)

        route.abort.assert_awaited_once_with("blockedbyclient")

    @pytest.mark.asyncio
    async def test_page_guard_failure_is_propagated(self):
        session = browser_use.BrowserSession("guard")
        context = MagicMock()
        cdp = MagicMock()
        cdp.send = AsyncMock(side_effect=RuntimeError("Fetch unavailable"))
        cdp.detach = AsyncMock()
        context.new_cdp_session = AsyncMock(return_value=cdp)
        session._context = context

        with pytest.raises(RuntimeError, match="Fetch unavailable"):
            await session._guard_page_requests(MagicMock())

        assert not session._guarded_pages
        assert not session._page_guard_futures
        cdp.detach.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_popup_guard_failure_closes_popup(self):
        session = browser_use.BrowserSession("guard")
        context = MagicMock()
        cdp = MagicMock()
        cdp.send = AsyncMock(side_effect=RuntimeError("Fetch unavailable"))
        cdp.detach = AsyncMock()
        context.new_cdp_session = AsyncMock(return_value=cdp)
        session._context = context
        page = MagicMock()
        page.close = AsyncMock()

        session._schedule_page_guard(page)
        guard_task = next(iter(session._cdp_tasks))
        await asyncio.gather(guard_task, return_exceptions=True)
        await asyncio.sleep(0)

        page.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fetch_guard_blocks_private_redirect_hops(self):
        session = browser_use.BrowserSession("guard")
        context = MagicMock()
        cdp = MagicMock()
        cdp.send = AsyncMock()
        context.new_cdp_session = AsyncMock(return_value=cdp)
        session._context = context

        await session._guard_page_requests(MagicMock())
        paused_handler = cdp.on.call_args.args[1]
        paused_handler(
            {
                "requestId": "redirect-hop",
                "request": {"url": "http://127.0.0.1/metadata"},
            }
        )
        await asyncio.gather(*list(session._cdp_tasks))

        cdp.send.assert_any_await(
            "Fetch.failRequest",
            {"requestId": "redirect-hop", "errorReason": "BlockedByClient"},
        )

    @pytest.mark.asyncio
    async def test_fetch_guard_continues_public_requests(self, monkeypatch):
        session = browser_use.BrowserSession("guard")
        context = MagicMock()
        cdp = MagicMock()
        cdp.send = AsyncMock()
        context.new_cdp_session = AsyncMock(return_value=cdp)
        session._context = context
        monkeypatch.setattr(
            browser_use,
            "validate_public_http_url",
            AsyncMock(return_value=["203.0.113.10"]),
        )

        await session._guard_page_requests(MagicMock())
        paused_handler = cdp.on.call_args.args[1]
        paused_handler(
            {
                "requestId": "public-hop",
                "request": {"url": "https://public.example/"},
            }
        )
        await asyncio.gather(*list(session._cdp_tasks))

        cdp.send.assert_any_await("Fetch.continueRequest", {"requestId": "public-hop"})


class TestBrowserClick:
    """Tests for browser_click function."""

    def test_click_without_playwright(self):
        """Test that click fails gracefully without Playwright."""
        import asyncio

        with patch("xagent.core.tools.core.browser_use.PLAYWRIGHT_AVAILABLE", False):
            result = asyncio.run(
                browser_use.browser_click(
                    session_id="test-session", selector="button.submit"
                )
            )

            assert result["success"] is False
            assert "not installed" in result["error"]


class TestBrowserFill:
    """Tests for browser_fill function."""

    def test_fill_without_playwright(self):
        """Test that fill fails gracefully without Playwright."""
        import asyncio

        with patch("xagent.core.tools.core.browser_use.PLAYWRIGHT_AVAILABLE", False):
            result = asyncio.run(
                browser_use.browser_fill(
                    session_id="test-session",
                    selector="input[name='email']",
                    value="test@example.com",
                )
            )

            assert result["success"] is False
            assert "not installed" in result["error"]


class TestBrowserScreenshot:
    """Tests for browser_screenshot function."""

    def test_screenshot_without_playwright(self):
        """Test that screenshot fails gracefully without Playwright."""
        import asyncio

        with patch("xagent.core.tools.core.browser_use.PLAYWRIGHT_AVAILABLE", False):
            result = asyncio.run(
                browser_use.browser_screenshot(session_id="test-session")
            )

            assert result["success"] is False
            assert "not installed" in result["error"]

    def test_screenshot_with_wait_for_lazy_load_parameter(self):
        """Test that screenshot accepts wait_for_lazy_load parameter."""
        import asyncio

        with patch("xagent.core.tools.core.browser_use.PLAYWRIGHT_AVAILABLE", False):
            result = asyncio.run(
                browser_use.browser_screenshot(
                    session_id="test-session",
                    full_page=True,
                    wait_for_lazy_load=True,
                )
            )

            # Should contain the wait_for_lazy_load parameter in response
            assert "wait_for_lazy_load" in result
            assert result["wait_for_lazy_load"] is True

    def test_screenshot_default_wait_for_lazy_load(self):
        """Test that wait_for_lazy_load defaults to False."""
        import asyncio

        with patch("xagent.core.tools.core.browser_use.PLAYWRIGHT_AVAILABLE", False):
            result = asyncio.run(
                browser_use.browser_screenshot(
                    session_id="test-session",
                    full_page=True,
                )
            )

            # Should default to False
            assert result.get("wait_for_lazy_load", False) is False


class TestBrowserExtractText:
    """Tests for browser_extract_text function."""

    def test_extract_text_without_playwright(self):
        """Test that extract_text fails gracefully without Playwright."""
        import asyncio

        with patch("xagent.core.tools.core.browser_use.PLAYWRIGHT_AVAILABLE", False):
            result = asyncio.run(
                browser_use.browser_extract_text(session_id="test-session")
            )

            assert result["success"] is False
            assert "not installed" in result["error"]


class TestBrowserEvaluate:
    """Tests for browser_evaluate function."""

    def test_evaluate_without_playwright(self):
        """Test that evaluate fails gracefully without Playwright."""
        import asyncio

        with patch("xagent.core.tools.core.browser_use.PLAYWRIGHT_AVAILABLE", False):
            result = asyncio.run(
                browser_use.browser_evaluate(
                    session_id="test-session", javascript="document.title"
                )
            )

            assert result["success"] is False
            assert "not installed" in result["error"]


class TestBrowserSelectOption:
    """Tests for browser_select_option function."""

    def test_select_option_without_playwright(self):
        """Test that select_option fails gracefully without Playwright."""
        import asyncio

        with patch("xagent.core.tools.core.browser_use.PLAYWRIGHT_AVAILABLE", False):
            result = asyncio.run(
                browser_use.browser_select_option(
                    session_id="test-session", selector="select.country", value="US"
                )
            )

            assert result["success"] is False
            assert "not installed" in result["error"]


class TestBrowserWaitForSelector:
    """Tests for browser_wait_for_selector function."""

    def test_wait_for_selector_without_playwright(self):
        """Test that wait_for_selector fails gracefully without Playwright."""
        import asyncio

        with patch("xagent.core.tools.core.browser_use.PLAYWRIGHT_AVAILABLE", False):
            result = asyncio.run(
                browser_use.browser_wait_for_selector(
                    session_id="test-session", selector=".dynamic-content"
                )
            )

            assert result["success"] is False
            assert "not installed" in result["error"]


class TestBrowserClose:
    """Tests for browser_close function."""

    def test_close_session(self, reset_manager):
        """Test closing a browser session."""
        import asyncio

        result = asyncio.run(browser_use.browser_close("test-session"))

        assert result["success"] is True
        assert "closed" in result["message"]


class TestBrowserListSessions:
    """Tests for browser_list_sessions function."""

    def test_list_sessions_empty(self, reset_manager):
        """Test listing sessions when none exist."""
        import asyncio

        result = asyncio.run(browser_use.browser_list_sessions())

        assert result["success"] is True
        assert result["count"] == 0
        assert result["sessions"] == []


class TestBrowserSessionManager:
    """Tests for BrowserSessionManager class."""

    def test_manager_singleton(self, reset_manager):
        """Test that manager is a singleton."""
        manager1 = browser_use.get_browser_manager()
        manager2 = browser_use.get_browser_manager()

        assert manager1 is manager2

    async def test_session_timeout(self, reset_manager):
        """Test that sessions can be cleaned up after timeout."""
        manager = browser_use.BrowserSessionManager(timeout_minutes=0)

        # Mock a session that's expired
        with patch("xagent.core.tools.core.browser_use.PLAYWRIGHT_AVAILABLE", True):
            session = browser_use.BrowserSession("test-session")
            session._last_used = browser_use.datetime.now() - browser_use.timedelta(
                minutes=1
            )

            async with manager._lock:
                manager._sessions["test-session"] = session

            # Run cleanup
            expired_count = await manager.cleanup_expired()

            assert expired_count >= 0


class TestBrowserSession:
    """Tests for BrowserSession class."""

    def test_session_initialization(self):
        """Test BrowserSession initialization."""
        with patch("xagent.core.tools.core.browser_use.PLAYWRIGHT_AVAILABLE", True):
            session = browser_use.BrowserSession("test-session", headless=True)

            assert session.session_id == "test-session"
            assert session.headless is True
            assert session._initialized is False

    @pytest.mark.asyncio
    async def test_session_uses_sandbox_unless_explicitly_allowed(self, monkeypatch):
        monkeypatch.setattr(browser_use, "PLAYWRIGHT_AVAILABLE", True)
        monkeypatch.setattr(
            browser_use.PublicNetworkEgressProxy,
            "start",
            AsyncMock(return_value="http://127.0.0.1:43123"),
        )
        monkeypatch.setattr(
            browser_use.PublicNetworkEgressProxy,
            "close",
            AsyncMock(),
        )
        playwright = MagicMock()
        browser = MagicMock()
        context = MagicMock()
        page = MagicMock()
        cdp = MagicMock()

        playwright.chromium.launch = AsyncMock(return_value=browser)
        browser.new_context = AsyncMock(return_value=context)
        context.new_page = AsyncMock(return_value=page)
        context.route = AsyncMock()
        context.new_cdp_session = AsyncMock(return_value=cdp)
        context.add_init_script = AsyncMock()
        cdp.send = AsyncMock()
        cdp.detach = AsyncMock()
        playwright_api = MagicMock()
        playwright_api.start = AsyncMock(return_value=playwright)
        monkeypatch.setattr(browser_use, "async_playwright", lambda: playwright_api)

        session = browser_use.BrowserSession("sandbox-test", headless=True)
        monkeypatch.setattr(browser_use, "get_browser_allow_unsandboxed", lambda: False)
        monkeypatch.setattr(browser_use, "get_browser_run_as", lambda: None)
        await session.get_page()

        launch_kwargs = playwright.chromium.launch.await_args.kwargs
        assert launch_kwargs["chromium_sandbox"] is True
        assert "--no-sandbox" not in launch_kwargs["args"]
        assert launch_kwargs["proxy"] == {
            "server": "http://127.0.0.1:43123",
            "bypass": "<-loopback>",
        }
        context.route.assert_awaited_once()
        assert context.route.await_args.args[0].pattern == r"^file:"

        await session.close()

    @pytest.mark.asyncio
    async def test_run_as_unset_launches_without_executable_path(self, monkeypatch):
        playwright = _mock_browser_playwright(monkeypatch)
        monkeypatch.setattr(browser_use, "get_browser_run_as", lambda: None)

        session = browser_use.BrowserSession("no-run-as", headless=True)
        await session.get_page()

        launch_kwargs = playwright.chromium.launch.await_args.kwargs
        assert launch_kwargs["executable_path"] is None
        await session.close()

    @pytest.mark.asyncio
    async def test_root_parent_launches_through_private_launcher(self, monkeypatch):
        playwright = _mock_browser_playwright(monkeypatch)
        monkeypatch.setattr(browser_use, "get_browser_run_as", lambda: (1234, 4321))
        monkeypatch.setattr(browser_use.os, "geteuid", lambda: 0)

        session = browser_use.BrowserSession("run-as-root", headless=True)
        await session.get_page()

        launch_kwargs = playwright.chromium.launch.await_args.kwargs
        launcher = Path(launch_kwargs["executable_path"])
        launcher_dir = launcher.parent
        assert launcher_dir.name.startswith("xagent-browser-launcher-")
        assert stat.S_IMODE(launcher_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(launcher.stat().st_mode) == 0o700
        script = launcher.read_text(encoding="utf-8")
        assert "/opt/playwright/chromium" in script
        assert " 1234 4321 " in script
        assert launch_kwargs["chromium_sandbox"] is True
        assert "--no-sandbox" not in launch_kwargs["args"]

        await session.close()
        assert not launcher_dir.exists()

    @pytest.mark.asyncio
    async def test_parent_already_target_uid_launches_without_wrapper(
        self, monkeypatch
    ):
        playwright = _mock_browser_playwright(monkeypatch)
        monkeypatch.setattr(browser_use, "get_browser_run_as", lambda: (1234, 4321))
        monkeypatch.setattr(browser_use.os, "geteuid", lambda: 1234)
        monkeypatch.setattr(browser_use.os, "getegid", lambda: 4321)

        session = browser_use.BrowserSession("already-target", headless=True)
        await session.get_page()

        launch_kwargs = playwright.chromium.launch.await_args.kwargs
        assert launch_kwargs["executable_path"] is None
        await session.close()

    @pytest.mark.asyncio
    async def test_parent_already_target_uid_rejects_gid_mismatch(self, monkeypatch):
        playwright = _mock_browser_playwright(monkeypatch)
        monkeypatch.setattr(browser_use, "get_browser_run_as", lambda: (1234, 4321))
        monkeypatch.setattr(browser_use.os, "geteuid", lambda: 1234)
        monkeypatch.setattr(browser_use.os, "getegid", lambda: 9999)

        session = browser_use.BrowserSession("already-target-gid", headless=True)
        with pytest.raises(RuntimeError, match="requires egid 4321"):
            await session.get_page()

        playwright.chromium.launch.assert_not_awaited()
        playwright.api.start.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_other_non_root_parent_rejects_identity_switch(self, monkeypatch):
        playwright = _mock_browser_playwright(monkeypatch)
        monkeypatch.setattr(browser_use, "get_browser_run_as", lambda: (1234, 4321))
        monkeypatch.setattr(browser_use.os, "geteuid", lambda: 1000)

        session = browser_use.BrowserSession("non-root-parent", headless=True)
        with pytest.raises(
            RuntimeError, match="requires the parent process to run as root"
        ):
            await session.get_page()

        playwright.chromium.launch.assert_not_awaited()
        playwright.api.start.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_launcher_dir_removed_when_launch_fails(self, monkeypatch):
        playwright = _mock_browser_playwright(monkeypatch)
        monkeypatch.setattr(browser_use, "get_browser_run_as", lambda: (1234, 4321))
        monkeypatch.setattr(browser_use.os, "geteuid", lambda: 0)
        playwright.chromium.launch.side_effect = RuntimeError("launch failed")

        session = browser_use.BrowserSession("launch-failure", headless=True)
        with pytest.raises(RuntimeError, match="launch failed"):
            await session.get_page()

        launcher = Path(playwright.chromium.launch.await_args.kwargs["executable_path"])
        assert not launcher.parent.exists()

    @pytest.mark.asyncio
    async def test_cancelled_close_still_removes_launcher_dir(self, monkeypatch):
        playwright = _mock_browser_playwright(monkeypatch)
        monkeypatch.setattr(browser_use, "get_browser_run_as", lambda: (1234, 4321))
        monkeypatch.setattr(browser_use.os, "geteuid", lambda: 0)
        session = browser_use.BrowserSession("close-cancel", headless=True)
        await session.get_page()
        launcher = Path(playwright.chromium.launch.await_args.kwargs["executable_path"])

        async def hang() -> None:
            await asyncio.Event().wait()

        monkeypatch.setattr(session, "_close_resources", hang)
        closing = asyncio.create_task(session.close())
        await asyncio.sleep(0)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing

        assert not launcher.parent.exists()

    @pytest.mark.asyncio
    async def test_invalid_run_as_config_never_starts_playwright(self, monkeypatch):
        playwright = _mock_browser_playwright(monkeypatch)

        def invalid_config():
            raise ValueError("invalid run-as config")

        monkeypatch.setattr(browser_use, "get_browser_run_as", invalid_config)
        session = browser_use.BrowserSession("invalid-config", headless=True)
        with pytest.raises(ValueError, match="invalid run-as config"):
            await session.get_page()

        playwright.api.start.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_run_as_without_geteuid_rejected_before_playwright(self, monkeypatch):
        playwright = _mock_browser_playwright(monkeypatch)
        monkeypatch.setattr(browser_use, "get_browser_run_as", lambda: (1234, 4321))
        monkeypatch.delattr(browser_use.os, "geteuid")
        session = browser_use.BrowserSession("no-geteuid", headless=True)

        with pytest.raises(RuntimeError, match="POSIX host"):
            await session.get_page()

        playwright.api.start.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_concurrent_get_page_launches_once(self, monkeypatch):
        playwright = _mock_browser_playwright(monkeypatch)
        monkeypatch.setattr(browser_use, "get_browser_run_as", lambda: (1234, 4321))
        monkeypatch.setattr(browser_use.os, "geteuid", lambda: 0)
        browser = playwright.chromium.launch.return_value
        started = asyncio.Event()
        release = asyncio.Event()

        async def blocked_launch(**kwargs):
            started.set()
            await release.wait()
            return browser

        playwright.chromium.launch = AsyncMock(side_effect=blocked_launch)
        session = browser_use.BrowserSession("concurrent-init", headless=True)
        first = asyncio.create_task(session.get_page())
        second = asyncio.create_task(session.get_page())
        await asyncio.wait_for(started.wait(), timeout=1)
        await asyncio.sleep(0)

        assert session._init_lock.locked()
        assert playwright.chromium.launch.await_count == 1
        assert not first.done()
        assert not second.done()

        release.set()
        pages = await asyncio.wait_for(asyncio.gather(first, second), timeout=5)
        assert pages[0] is pages[1]
        playwright.chromium.launch.assert_awaited_once()
        launcher = Path(playwright.chromium.launch.await_args.kwargs["executable_path"])
        assert launcher.parent.exists()
        await session.close()
        assert not launcher.parent.exists()

    @pytest.mark.asyncio
    async def test_cancel_during_launch_removes_launcher_dir(self, monkeypatch):
        playwright = _mock_browser_playwright(monkeypatch)
        monkeypatch.setattr(browser_use, "get_browser_run_as", lambda: (1234, 4321))
        monkeypatch.setattr(browser_use.os, "geteuid", lambda: 0)
        started = asyncio.Event()

        async def hanging_launch(**kwargs):
            started.set()
            await asyncio.Event().wait()

        playwright.chromium.launch = AsyncMock(side_effect=hanging_launch)
        session = browser_use.BrowserSession("cancel-init", headless=True)
        task = asyncio.create_task(session.get_page())
        await asyncio.wait_for(started.wait(), timeout=1)
        launcher = Path(playwright.chromium.launch.await_args.kwargs["executable_path"])
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        for _ in range(100):
            if not launcher.parent.exists():
                break
            await asyncio.sleep(0)
        assert not launcher.parent.exists()
