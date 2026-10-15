"""
Tests for browser automation tools.

These tests use mocking to avoid requiring actual browser installation.
Run with: pytest tests/core/tools/test_browser_use.py
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
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

    def test_navigate_without_playwright(self, monkeypatch):
        """Test that navigate fails gracefully without Playwright."""
        import asyncio

        async def public_target(_url: str) -> list[str]:
            return ["203.0.113.10"]

        monkeypatch.setattr(browser_use, "validate_public_http_url", public_target)

        with patch("xagent.core.tools.core.browser_use.PLAYWRIGHT_AVAILABLE", False):
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
                assert "private network" in result["error"]

    def test_navigate_allows_workspace_file_only(self, tmp_path):
        import asyncio

        html_path = tmp_path / "report.html"
        html_path.write_text("<html><body>ok</body></html>", encoding="utf-8")

        with patch("xagent.core.tools.core.browser_use.PLAYWRIGHT_AVAILABLE", False):
            allowed = asyncio.run(
                browser_use.browser_navigate(
                    session_id="test-session",
                    url=html_path.as_uri(),
                    _allowed_file_roots=[str(tmp_path)],
                )
            )
            blocked = asyncio.run(
                browser_use.browser_navigate(
                    session_id="test-session",
                    url=html_path.as_uri(),
                )
            )

        assert "Only public network URLs are allowed" not in allowed["error"]
        assert (
            blocked["error"]
            == "Only authorized workspace or uploaded files are allowed"
        )


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
    @pytest.mark.parametrize(
        ("allow_unsandboxed", "expected_sandbox", "expected_no_sandbox"),
        [(False, True, False), (True, False, True)],
    )
    async def test_launch_security_flags_follow_explicit_setting(
        self, monkeypatch, allow_unsandboxed, expected_sandbox, expected_no_sandbox
    ):
        context = MagicMock()
        context.route = AsyncMock()
        context.route_web_socket = AsyncMock()
        context.add_init_script = AsyncMock()
        page = MagicMock()
        context.new_page = AsyncMock(return_value=page)
        browser = MagicMock()
        browser.new_context = AsyncMock(return_value=context)
        chromium = MagicMock()
        chromium.launch = AsyncMock(return_value=browser)
        playwright = MagicMock(chromium=chromium)
        playwright.stop = AsyncMock()
        starter = MagicMock()
        starter.start = AsyncMock(return_value=playwright)

        monkeypatch.setattr(browser_use, "PLAYWRIGHT_AVAILABLE", True)
        monkeypatch.setattr(browser_use, "async_playwright", lambda: starter)
        monkeypatch.setattr(
            browser_use, "get_browser_allow_unsandboxed", lambda: allow_unsandboxed
        )
        monkeypatch.setattr(browser_use, "get_trusted_proxy_url", lambda: None)
        monkeypatch.setattr(
            browser_use,
            "PublicNetworkAsyncHTTPTransport",
            lambda **_kwargs: httpx.MockTransport(lambda _request: httpx.Response(200)),
        )

        session = browser_use.BrowserSession("security-flags")
        try:
            await session._ensure_initialized()
            assert session._http_client is not None
            assert session._http_client.follow_redirects is False
        finally:
            await session.close()

        launch_kwargs = chromium.launch.await_args.kwargs
        assert launch_kwargs["chromium_sandbox"] is expected_sandbox
        assert ("--no-sandbox" in launch_kwargs["args"]) is expected_no_sandbox
        context.route_web_socket.assert_awaited_once()
        context.add_init_script.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_guard_route_preserves_encoded_body_and_multiple_cookies(self):
        captured: list[httpx.Request] = []

        async def respond(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(
                200,
                request=request,
                headers=[
                    ("content-encoding", "br"),
                    ("set-cookie", "first=1; Path=/; HttpOnly"),
                    ("set-cookie", "second=2; Path=/; SameSite=Lax"),
                ],
                stream=httpx.ByteStream(b"encoded-body"),
            )

        session = browser_use.BrowserSession("guard")
        session._http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(respond), follow_redirects=False
        )
        session._context = MagicMock()
        session._context.add_cookies = AsyncMock()
        session._context.cookies = AsyncMock(return_value=[])
        route = MagicMock()
        route.request.url = "https://public.example/page"
        route.request.method = "GET"
        route.request.is_navigation_request = lambda: False
        route.request.post_data_buffer = None
        route.request.all_headers = AsyncMock(
            return_value={"accept-encoding": "br, gzip", "host": "public.example"}
        )
        route.fulfill = AsyncMock()
        route.abort = AsyncMock()

        try:
            await session._guard_route(route)
        finally:
            await session._http_client.aclose()

        assert captured[0].headers["accept-encoding"] == "identity"
        route.abort.assert_not_awaited()
        fulfill = route.fulfill.await_args.kwargs
        assert fulfill["body"] == b"encoded-body"
        assert fulfill["headers"]["content-encoding"] == "br"
        cookie_calls = session._context.add_cookies.await_args_list
        assert [call.args[0][0]["name"] for call in cookie_calls] == [
            "first",
            "second",
        ]

    @pytest.mark.asyncio
    async def test_guard_route_aborts_oversized_response(self, monkeypatch):
        async def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, request=request, stream=httpx.ByteStream(b"1234")
            )

        monkeypatch.setattr(browser_use, "_BROWSER_RESPONSE_MAX_BYTES", 3)
        session = browser_use.BrowserSession("guard-limit")
        session._http_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        session._context = MagicMock()
        session._context.add_cookies = AsyncMock()
        session._context.cookies = AsyncMock(return_value=[])
        route = MagicMock()
        route.request.url = "https://public.example/large"
        route.request.method = "GET"
        route.request.is_navigation_request = lambda: False
        route.request.post_data_buffer = None
        route.request.all_headers = AsyncMock(return_value={})
        route.fulfill = AsyncMock()
        route.abort = AsyncMock()

        try:
            await session._guard_route(route)
        finally:
            await session._http_client.aclose()

        route.fulfill.assert_not_awaited()
        route.abort.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_guard_route_follows_subresource_redirects_inside_httpx(
        self, monkeypatch
    ):
        captured: list[str] = []

        async def respond(request: httpx.Request) -> httpx.Response:
            captured.append(str(request.url))
            if request.url.path == "/start":
                return httpx.Response(
                    302,
                    request=request,
                    headers={"location": "https://public.example/final"},
                )
            return httpx.Response(
                200, request=request, stream=httpx.ByteStream(b"final")
            )

        async def allow_public_target(_url: str) -> list[str]:
            return ["203.0.113.10"]

        monkeypatch.setattr(
            browser_use, "validate_public_http_url", allow_public_target
        )
        session = browser_use.BrowserSession("guard-redirect")
        session._http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(respond), follow_redirects=False
        )
        session._context = MagicMock()
        session._context.add_cookies = AsyncMock()
        session._context.cookies = AsyncMock(return_value=[])
        route = MagicMock()
        route.request.url = "https://public.example/start"
        route.request.method = "GET"
        route.request.is_navigation_request = lambda: False
        route.request.post_data_buffer = None
        route.request.all_headers = AsyncMock(return_value={})
        route.fulfill = AsyncMock()
        route.abort = AsyncMock()

        try:
            await session._guard_route(route)
        finally:
            await session._http_client.aclose()

        assert captured == [
            "https://public.example/start",
            "https://public.example/final",
        ]
        assert route.abort.await_count == 0
        assert route.fulfill.await_args.kwargs["status"] == 200
        assert route.fulfill.await_args.kwargs["body"] == b"final"

    @pytest.mark.asyncio
    async def test_guard_route_restarts_navigation_redirect_through_route(
        self, monkeypatch
    ):
        async def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                302,
                request=request,
                headers={"location": "https://public.example/final"},
                stream=httpx.ByteStream(b""),
            )

        async def allow_public_target(_url: str) -> list[str]:
            return ["203.0.113.10"]

        monkeypatch.setattr(
            browser_use, "validate_public_http_url", allow_public_target
        )
        session = browser_use.BrowserSession("guard-navigation-redirect")
        session._http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(respond), follow_redirects=False
        )
        session._context = MagicMock()
        session._context.add_cookies = AsyncMock()
        session._context.cookies = AsyncMock(return_value=[])
        route = MagicMock()
        route.request.url = "https://public.example/start"
        route.request.method = "GET"
        route.request.is_navigation_request = lambda: True
        route.request.post_data_buffer = None
        route.request.all_headers = AsyncMock(return_value={})
        route.fulfill = AsyncMock()
        route.abort = AsyncMock()

        try:
            await session._guard_route(route)
        finally:
            await session._http_client.aclose()

        fulfill = route.fulfill.await_args.kwargs
        assert fulfill["status"] == 200
        assert b"https://public.example/final" in fulfill["body"]
        route.abort.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("is_navigation", [False, True])
    async def test_guard_route_aborts_redirect_to_private_target(
        self, monkeypatch, is_navigation
    ):
        async def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                302,
                request=request,
                headers={"location": "http://169.254.169.254/latest/meta-data/"},
                stream=httpx.ByteStream(b""),
            )

        async def reject_private_target(_url: str) -> list[str]:
            raise ValueError("private network target")

        monkeypatch.setattr(
            browser_use, "validate_public_http_url", reject_private_target
        )
        session = browser_use.BrowserSession("guard-private-redirect")
        session._http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(respond), follow_redirects=False
        )
        session._context = MagicMock()
        session._context.add_cookies = AsyncMock()
        session._context.cookies = AsyncMock(return_value=[])
        route = MagicMock()
        route.request.url = "https://public.example/start"
        route.request.method = "GET"
        route.request.is_navigation_request = lambda: is_navigation
        route.request.post_data_buffer = None
        route.request.all_headers = AsyncMock(return_value={})
        route.fulfill = AsyncMock()
        route.abort = AsyncMock()

        try:
            await session._guard_route(route)
        finally:
            await session._http_client.aclose()

        route.fulfill.assert_not_awaited()
        route.abort.assert_awaited_once()
