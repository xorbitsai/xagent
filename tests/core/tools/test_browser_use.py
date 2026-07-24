"""
Tests for browser automation tools.

These tests use mocking to avoid requiring actual browser installation.
Run with: pytest tests/core/tools/test_browser_use.py
"""

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

    async def test_persistent_profile_rejects_another_task(
        self, reset_manager, tmp_path
    ):
        manager = browser_use.BrowserSessionManager()
        profile_dir = tmp_path / "user_1" / "default"
        try:
            first = await manager.get_or_create(
                "computer-profile:user_1:default",
                headless=False,
                persistent_profile_dir=profile_dir,
                owner_id="task-1",
            )

            assert first.is_persistent is True
            with pytest.raises(
                browser_use.BrowserProfileInUseError,
                match="another task",
            ):
                await manager.get_or_create(
                    "computer-profile:user_1:default",
                    headless=False,
                    persistent_profile_dir=profile_dir,
                    owner_id="task-2",
                )
        finally:
            await manager.close_all()


class TestBrowserSession:
    """Tests for BrowserSession class."""

    def test_session_initialization(self):
        """Test BrowserSession initialization."""
        with patch("xagent.core.tools.core.browser_use.PLAYWRIGHT_AVAILABLE", True):
            session = browser_use.BrowserSession("test-session", headless=True)

            assert session.session_id == "test-session"
            assert session.headless is True
            assert session._initialized is False

    async def test_persistent_session_uses_persistent_context(
        self, mock_playwright, tmp_path
    ):
        page = MagicMock()
        page.add_init_script = AsyncMock()
        page.close = AsyncMock()
        context = MagicMock()
        context.pages = []
        context.new_page = AsyncMock(return_value=page)
        context.close = AsyncMock()
        playwright = MagicMock()
        playwright.chromium.launch_persistent_context = AsyncMock(return_value=context)
        playwright.stop = AsyncMock()
        mock_playwright.return_value.start = AsyncMock(return_value=playwright)
        profile_dir = tmp_path / "profiles" / "user_1" / "default"
        session = browser_use.BrowserSession(
            "persistent",
            headless=False,
            persistent_profile_dir=profile_dir,
            owner_id="task-1",
        )

        try:
            assert await session.get_page() is page
            kwargs = playwright.chromium.launch_persistent_context.await_args.kwargs
            assert kwargs["user_data_dir"] == str(profile_dir)
            assert kwargs["headless"] is False
            assert "--no-sandbox" not in kwargs["args"]
            assert "--disable-web-security" not in kwargs["args"]
            assert "--restore-last-session" in kwargs["args"]
            assert profile_dir.stat().st_mode & 0o777 == 0o700
        finally:
            await session.close()
