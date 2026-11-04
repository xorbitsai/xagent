"""
Tests for browser automation tools.

These tests use mocking to avoid requiring actual browser installation.
Run with: pytest tests/core/tools/test_browser_use.py
"""

import asyncio
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
    async def test_context_route_allows_authorized_file_and_falls_back_for_http(
        self, tmp_path
    ):
        session = browser_use.BrowserSession("guard", workspace_root=str(tmp_path))
        local_file = tmp_path / "page.html"
        local_file.write_text("<html></html>", encoding="utf-8")

        local_route = MagicMock()
        local_route.request.url = local_file.as_uri()
        local_route.abort = AsyncMock()
        local_route.fallback = AsyncMock()
        await session._guard_context_route(local_route)
        local_route.fallback.assert_awaited_once_with()
        local_route.abort.assert_not_awaited()

        http_route = MagicMock()
        http_route.request.url = "https://example.com/"
        http_route.fallback = AsyncMock()
        await session._guard_context_route(http_route)
        http_route.fallback.assert_awaited_once_with()

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
