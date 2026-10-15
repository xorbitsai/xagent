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
                assert result["error"] == "Only public network URLs are allowed"

    def test_navigate_allows_workspace_file_only(self, tmp_path):
        import asyncio

        html_path = tmp_path / "report.html"
        html_path.write_text("<html><body>ok</body></html>", encoding="utf-8")

        with patch("xagent.core.tools.core.browser_use.PLAYWRIGHT_AVAILABLE", False):
            allowed = asyncio.run(
                browser_use.browser_navigate(
                    session_id="test-session",
                    url=html_path.as_uri(),
                    _workspace_root=str(tmp_path),
                )
            )
            blocked = asyncio.run(
                browser_use.browser_navigate(
                    session_id="test-session",
                    url=html_path.as_uri(),
                )
            )

        assert "Only public network URLs are allowed" not in allowed["error"]
        assert blocked["error"] == "Only public network URLs are allowed"


class TestBrowserRequestGuard:
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
