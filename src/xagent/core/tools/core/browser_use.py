"""
Browser automation tool with lazy initialization.

This module provides browser automation capabilities using Playwright.
Browser sessions are created on first use (lazy initialization) and
automatically cleaned up after a timeout period.
"""

import asyncio
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from playwright.async_api import Browser, BrowserContext, Page, async_playwright

# Optional import for Playwright
try:
    from playwright.async_api import Browser, BrowserContext, Page, async_playwright

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

    # Define stub types when Playwright is not available
    # These are only used when PLAYWRIGHT_AVAILABLE is False
    Browser: Any = None  # type: ignore[no-redef]
    BrowserContext: Any = None  # type: ignore[no-redef]
    Page: Any = None  # type: ignore[no-redef]
    async_playwright: Any = None  # type: ignore[no-redef]


def _format_error_with_traceback(error: Exception, context: str = "") -> str:
    """
    Format an error with full traceback for better debugging.

    Args:
        error: The exception to format
        context: Optional context string describing what operation failed

    Returns:
        Formatted error message with traceback
    """
    import traceback

    tb_str = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )

    if context:
        return f"{context}\nError: {str(error)}\n\nTraceback:\n{tb_str}"
    else:
        return f"Error: {str(error)}\n\nTraceback:\n{tb_str}"


class BrowserSession:
    """Browser session with lazy initialization."""

    def __init__(self, session_id: str, headless: bool = True):
        """
        Initialize a browser session.

        Args:
            session_id: Unique identifier for this session
            headless: Whether to run browser in headless mode
        """
        self.session_id = session_id
        self.headless = headless
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._playwright: Any = None
        self._created_at = datetime.now()
        self._last_used = datetime.now()
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Lazy initialization: create browser on first use."""
        if not self._initialized:
            if not PLAYWRIGHT_AVAILABLE:
                raise RuntimeError(
                    "Playwright is not installed or browsers not downloaded. "
                    "Install with: pip install playwright. "
                    "Then download browsers: playwright install chromium"
                )

            self._playwright = await async_playwright().start()

            # Launch browser with anti-detection settings
            self._browser = await self._playwright.chromium.launch(
                headless=self.headless,
                args=[
                    # Disable WebDriver detection
                    "--disable-blink-features=AutomationControlled",
                    # Other anti-detection flags
                    "--disable-infobars",
                    "--window-size=1920,1080",
                    # Allow local file access
                    "--allow-file-access-from-files",
                    "--allow-file-access",
                    # No sandbox for local file access in some environments
                    "--no-sandbox",
                    # Disable web security for file:// URLs (required for local files)
                    "--disable-web-security",
                ],
            )

            # Create context with realistic settings
            self._context = await self._browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                # Set locale and timezone
                locale="zh-CN",
                timezone_id="Asia/Shanghai",
            )

            self._page = await self._context.new_page()

            # Inject script to hide webdriver property
            await self._page.add_init_script("""
                // Override webdriver detection
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });

                // Add chrome property
                window.chrome = {
                    runtime: {}
                };

                // Override permissions
                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications' ?
                        Promise.resolve({ state: Notification.permission }) :
                        originalQuery(parameters)
                );

                // Override plugins
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5]
                });

                // Override languages
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['zh-CN', 'zh', 'en-US', 'en']
                });
            """)

            self._initialized = True

    async def get_page(self) -> Page:
        """
        Get the page instance, creating browser if needed.

        Returns:
            The Playwright Page instance
        """
        await self._ensure_initialized()
        assert self._page is not None
        return self._page

    async def close(self) -> None:
        """Close the browser and cleanup resources."""
        try:
            # Close page first
            if self._page:
                await self._page.close()
                self._page = None
        except Exception:
            pass  # Ignore page close errors

        try:
            # Close context
            if self._context:
                await self._context.close()
                self._context = None
        except Exception:
            pass  # Ignore context close errors

        try:
            # Close browser
            if self._browser:
                await self._browser.close()
                self._browser = None
        except Exception:
            pass  # Ignore browser close errors

        try:
            # Stop playwright
            if self._playwright:
                await self._playwright.stop()
                self._playwright = None
        except Exception:
            pass  # Ignore playwright stop errors

        self._initialized = False


class BrowserSessionManager:
    """Global browser session manager with automatic cleanup."""

    def __init__(self, timeout_minutes: int = 30):
        """
        Initialize the browser session manager.

        Args:
            timeout_minutes: Minutes of inactivity before auto-closing a session
        """
        self._sessions: Dict[str, BrowserSession] = {}
        self._lock: asyncio.Lock = (
            asyncio.Lock()
        )  # Use asyncio.Lock instead of threading.Lock
        self._timeout = timedelta(minutes=timeout_minutes)
        self._cleanup_task: Optional[asyncio.Task[None]] = None  # Track cleanup task

    async def get_or_create(
        self, session_id: str, headless: bool = False
    ) -> BrowserSession:
        """
        Get or create a browser session (async-safe).

        Args:
            session_id: Unique session identifier
            headless: Whether to use headless mode (only used when creating new session)

        Returns:
            BrowserSession instance
        """
        async with self._lock:
            # Check if session already exists
            if session_id in self._sessions:
                # Update last used time and return existing session
                # Ignore headless parameter for existing sessions (can't change mode after creation)
                self._sessions[session_id]._last_used = datetime.now()
                return self._sessions[session_id]

            # Start cleanup task on first session creation
            if self._cleanup_task is None:
                self._cleanup_task = asyncio.create_task(self._cleanup_loop())

            # Create new session with specified headless mode
            self._sessions[session_id] = BrowserSession(session_id, headless)
            self._sessions[session_id]._last_used = datetime.now()
            return self._sessions[session_id]

    async def close(self, session_id: str) -> None:
        """
        Close a specific browser session.

        Args:
            session_id: Session to close
        """
        async with self._lock:
            if session_id in self._sessions:
                await self._sessions[session_id].close()
                del self._sessions[session_id]

    async def cleanup_expired(self) -> int:
        """Clean up expired sessions based on timeout."""
        now = datetime.now()
        async with self._lock:
            expired = [
                sid
                for sid, sess in self._sessions.items()
                if now - sess._last_used > self._timeout
            ]
            for sid in expired:
                await self._sessions[sid].close()
                del self._sessions[sid]
            return len(expired)

    async def _cleanup_loop(self) -> None:
        """Background cleanup task (async instead of threaded)."""
        try:
            while True:
                try:
                    await asyncio.sleep(300)  # Check every 5 minutes
                    expired_count = await self.cleanup_expired()
                    if expired_count > 0:
                        pass  # Could log cleanup here
                except Exception:
                    pass  # Silently handle errors in cleanup task
        except asyncio.CancelledError:
            # Task was cancelled, exit gracefully
            pass

    async def close_all(self) -> None:
        """Close all browser sessions (for shutdown)."""
        # Cancel cleanup task if running
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        async with self._lock:
            for session in list(self._sessions.values()):
                await session.close()
            self._sessions.clear()


# Global singleton with async-safe initialization
_manager: Optional[BrowserSessionManager] = None


def get_browser_manager() -> BrowserSessionManager:
    """
    Get the global browser session manager (async-safe singleton).

    Returns:
        BrowserSessionManager instance
    """
    global _manager
    if _manager is None:
        # No lock needed in async environment (no race condition in single-threaded async execution)
        _manager = BrowserSessionManager()
    return _manager


# ============== Browser Tool Functions ==============


async def browser_navigate(**kwargs: Any) -> Dict[str, Any]:
    """
    Navigate to a URL. Browser session is created automatically on first use.

    Args:
        session_id: Session ID (typically from AgentContext.session_id)
        url: Target URL to navigate to
        headless: Whether to run browser in headless mode (default: True)
        wait_until: Wait condition (default: "networkidle")

    Returns:
        Dictionary with navigation result including URL, page title, and success status
    """
    # Extract parameters with validation
    session_id = kwargs.get("session_id")
    url = kwargs.get("url")
    if not session_id or not url:
        return {
            "success": False,
            "session_id": session_id or "",
            "url": url or "",
            "title": "",
            "message": "",
            "error": "Missing required parameters: session_id and url are required",
        }
    headless = kwargs.get("headless", False)
    wait_until = kwargs.get("wait_until", "networkidle")

    if not PLAYWRIGHT_AVAILABLE:
        return {
            "success": False,
            "session_id": session_id,
            "url": url,
            "title": "",
            "message": "",
            "error": "Playwright is not installed or browsers not downloaded. Install with: pip install playwright. Then download browsers: playwright install chromium",
        }

    manager = get_browser_manager()
    session = await manager.get_or_create(session_id, headless)
    page = await session.get_page()

    try:
        # For local files, use 'domcontentloaded' or 'load' instead of 'networkidle'
        # because local files don't have network activity
        if url.startswith("file://"):
            # Override wait_until for local files
            wait_until_local = (
                wait_until if wait_until != "networkidle" else "domcontentloaded"
            )
        else:
            wait_until_local = wait_until

        # Log navigation attempt for debugging
        import logging

        logger = logging.getLogger(__name__)
        logger.info(
            f"[browser_navigate] Navigating to {url} (headless={headless}, wait_until={wait_until_local})"
        )

        # Add timeout to prevent hanging (default 30 seconds)
        # Using wait_until parameter (default: 'networkidle', but can be 'domcontentloaded' or 'load')
        # networkidle can hang on pages with continuous background requests
        await page.goto(url, wait_until=wait_until_local, timeout=30000)  # type: ignore[arg-type]
        # Get title (no timeout parameter available for title())
        title = await page.title()

        logger.info(
            f"[browser_navigate] Successfully navigated to {url}, title: {title}"
        )

        return {
            "success": True,
            "session_id": session_id,
            "url": url,
            "title": title,
            "message": f"Navigated to {url}. IMPORTANT: Session ID is '{session_id}'. You MUST use this exact session_id='{session_id}' in all subsequent browser calls (browser_click, browser_extract_text, etc.) to continue using this browser.",
        }
    except Exception as e:
        import logging
        import traceback

        logger = logging.getLogger(__name__)
        logger.error(f"[browser_navigate] Failed to navigate to {url}: {str(e)}")
        logger.error(f"[browser_navigate] Traceback: {traceback.format_exc()}")

        return {
            "success": False,
            "session_id": session_id,
            "url": url,
            "title": "",
            "message": "",
            "error": _format_error_with_traceback(
                e, f"Navigation failed for URL '{url}'"
            ),
        }


async def browser_click(**kwargs: Any) -> Dict[str, Any]:
    """
    Click an element on the page. Browser session is created automatically if needed.

    Args:
        session_id: Session ID
        selector: CSS selector or XPath for the element to click
        headless: Whether to run browser in headless mode (default: True)
        timeout: Timeout in milliseconds (default: 30000)

    Returns:
        Dictionary with click result
    """
    # Extract parameters with validation
    session_id = kwargs.get("session_id")
    selector = kwargs.get("selector")
    if not session_id or not selector:
        return {
            "success": False,
            "session_id": session_id or "",
            "selector": selector or "",
            "message": "",
            "error": "Missing required parameters: session_id and selector are required",
        }
    timeout = kwargs.get("timeout", 30000)

    if not PLAYWRIGHT_AVAILABLE:
        return {
            "success": False,
            "session_id": session_id,
            "selector": selector,
            "message": "",
            "error": "Playwright is not installed or browsers not downloaded. Install with: pip install playwright. Then download browsers: playwright install chromium",
        }

    manager = get_browser_manager()
    session = await manager.get_or_create(session_id)
    page = await session.get_page()

    try:
        await page.click(selector, timeout=timeout)
        return {
            "success": True,
            "session_id": session_id,
            "selector": selector,
            "message": f"Successfully clicked element: {selector}. Session ID: {session_id}",
        }
    except Exception as e:
        return {
            "success": False,
            "session_id": session_id,
            "selector": selector,
            "message": "",
            "error": _format_error_with_traceback(
                e, f"Click failed for selector '{selector}'"
            ),
        }


async def browser_fill(**kwargs: Any) -> Dict[str, Any]:
    """
    Fill an input field with text. Browser session is created automatically if needed.

    Args:
        session_id: Session ID
        selector: CSS selector or XPath for the input element
        value: Text value to fill
        headless: Whether to run browser in headless mode (default: True)

    Returns:
        Dictionary with fill result
    """
    # Extract parameters with validation
    session_id = kwargs.get("session_id")
    selector = kwargs.get("selector")
    value = kwargs.get("value")
    if not session_id or not selector or value is None:
        return {
            "success": False,
            "session_id": session_id or "",
            "selector": selector or "",
            "value": "",
            "message": "",
            "error": "Missing required parameters: session_id, selector, and value are required",
        }

    if not PLAYWRIGHT_AVAILABLE:
        return {
            "success": False,
            "session_id": session_id,
            "selector": selector,
            "value": "",
            "message": "",
            "error": "Playwright is not installed or browsers not downloaded. Install with: pip install playwright. Then download browsers: playwright install chromium",
        }

    manager = get_browser_manager()
    session = await manager.get_or_create(session_id)
    page = await session.get_page()

    try:
        # Add timeout to prevent hanging (default 30 seconds)
        await page.fill(selector, value, timeout=30000)
        preview = value[:50] + "..." if len(value) > 50 else value
        return {
            "success": True,
            "session_id": session_id,
            "selector": selector,
            "value": preview,
            "message": f"Filled {selector} with: {preview}. Session ID: {session_id}",
        }
    except Exception as e:
        return {
            "success": False,
            "session_id": session_id,
            "selector": selector,
            "value": "",
            "message": "",
            "error": _format_error_with_traceback(
                e, f"Fill failed for selector '{selector}'"
            ),
        }


async def browser_screenshot(**kwargs: Any) -> Dict[str, Any]:
    """
    Take a screenshot of the current page. Browser session is created automatically if needed.

    Args:
        session_id: Session ID
        full_page: Whether to capture the full page (default: False).
            Set to True to capture the entire scrolling page, not just the visible area.
        headless: Whether to run browser in headless mode (default: True)
        width: Desired viewport width in pixels (default: 1920)
        height: Desired viewport height in pixels (default: 1080)
        wait_for_lazy_load: Whether to scroll and wait for lazy-loaded content (default: False).
            Only effective when full_page=True. Use this for pages with infinite scroll,
            lazy-loaded images, or dynamic content loading. The page will be scrolled
            gradually to trigger all lazy-loaded content before capturing.

    Returns:
        Dictionary with screenshot data (base64 encoded) and metadata

    Examples:
        # Capture visible area only
        browser_screenshot(session_id="my-session")

        # Capture full page (may miss lazy-loaded content)
        browser_screenshot(session_id="my-session", full_page=True)

        # Capture full page with lazy-loaded content
        browser_screenshot(session_id="my-session", full_page=True, wait_for_lazy_load=True)
    """
    # Extract parameters with validation
    session_id = kwargs.get("session_id")
    if not session_id:
        return {
            "success": False,
            "session_id": "",
            "screenshot": "",
            "format": "base64",
            "full_page": False,
            "wait_for_lazy_load": False,
            "message": "",
            "error": "Missing required parameter: session_id is required",
        }
    full_page = kwargs.get("full_page", False)
    width = kwargs.get("width", 1920)
    height = kwargs.get("height", 1080)
    wait_for_lazy_load = kwargs.get("wait_for_lazy_load", False)

    if not PLAYWRIGHT_AVAILABLE:
        return {
            "success": False,
            "session_id": session_id,
            "screenshot": "",
            "format": "base64",
            "full_page": full_page,
            "wait_for_lazy_load": wait_for_lazy_load,
            "message": "",
            "error": "Playwright is not installed or browsers not downloaded. Install with: pip install playwright. Then download browsers: playwright install chromium",
        }

    import logging

    logger = logging.getLogger(__name__)
    logger.info(
        f"[browser_screenshot] Starting screenshot function (session={session_id}, width={width}, height={height})"
    )

    manager = get_browser_manager()
    logger.info(f"[browser_screenshot] Got browser manager (session={session_id})")

    session = await manager.get_or_create(session_id)
    logger.info(f"[browser_screenshot] Got/created session (session={session_id})")

    page = await session.get_page()
    logger.info(f"[browser_screenshot] Got page object (session={session_id})")

    # Set viewport size if width or height is specified
    current_viewport = page.viewport_size
    if current_viewport is not None and (
        current_viewport.get("width") != width
        or current_viewport.get("height") != height
    ):
        logger.info(
            f"[browser_screenshot] Setting viewport size to {width}x{height} (session={session_id})"
        )
        await page.set_viewport_size({"width": width, "height": height})

    # Track actual screenshot mode (may differ from requested if fallback occurs)
    actual_full_page = full_page

    try:
        # Screenshot returns bytes, convert to base64 with data URI prefix
        import base64

        logger = logging.getLogger(__name__)

        logger.info(
            f"[browser_screenshot] Starting screenshot (session={session_id}, full_page={full_page})"
        )

        if not full_page:
            # Viewport-only screenshot (fast, reliable)
            try:
                logger.info(
                    f"[browser_screenshot] Taking viewport screenshot (session={session_id})"
                )

                screenshot_bytes = await page.screenshot(
                    full_page=False,
                    type="png",
                )

                logger.info(
                    f"[browser_screenshot] Viewport screenshot captured (session={session_id}, size={len(screenshot_bytes)} bytes)"
                )
            except Exception as screenshot_error:
                raise screenshot_error
        else:
            # full_page=True: Use Playwright's full_page (works fine in clean asyncio)
            logger.info(
                f"[browser_screenshot] Using Playwright full_page (session={session_id}, wait_for_lazy_load={wait_for_lazy_load})"
            )

            if wait_for_lazy_load and full_page:
                # Trigger lazy loading by scrolling the page before screenshot
                logger.info(
                    f"[browser_screenshot] Triggering lazy loading before screenshot (session={session_id})"
                )

                # Get page height
                page_height = await page.evaluate("document.body.scrollHeight")
                logger.info(
                    f"[browser_screenshot] Page height: {page_height}px (session={session_id})"
                )

                # Scroll through the page in chunks to trigger lazy loading
                viewport_size = page.viewport_size
                viewport_height = (
                    viewport_size.get("height", 1080) if viewport_size else 1080
                )
                scroll_position = 0
                scroll_count = 0
                max_scrolls = 50  # Prevent infinite loops

                while scroll_position < page_height and scroll_count < max_scrolls:
                    logger.info(
                        f"[browser_screenshot] Scrolling to y={scroll_position}/{page_height} (session={session_id}, scroll {scroll_count + 1}/{max_scrolls})"
                    )
                    await page.evaluate(f"window.scrollTo(0, {scroll_position})")
                    await page.wait_for_timeout(300)  # Wait 300ms for content to load
                    scroll_position += viewport_height
                    scroll_count += 1

                # Scroll back to top before screenshot
                logger.info(
                    f"[browser_screenshot] Scrolled back to top (session={session_id})"
                )
                await page.evaluate("window.scrollTo(0, 0)")
                await page.wait_for_timeout(500)  # Wait for page to stabilize

            try:
                logger.info(
                    f"[browser_screenshot] Calling page.screenshot(full_page=True) (session={session_id})"
                )
                screenshot_bytes = await page.screenshot(
                    full_page=True,
                    type="png",
                )
                logger.info(
                    f"[browser_screenshot] Full page screenshot successful: {len(screenshot_bytes)} bytes (session={session_id})"
                )
            except Exception as native_error:
                logger.warning(
                    f"[browser_screenshot] Full page failed: {native_error}, falling back to viewport (session={session_id})"
                )
                # Fallback to viewport
                screenshot_bytes = await page.screenshot(
                    full_page=False,
                    type="png",
                )
                actual_full_page = False

        screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
        # Add data URI prefix for compatibility with vision tools
        screenshot_data_uri = f"data:image/png;base64,{screenshot_b64}"

        return {
            "success": True,
            "session_id": session_id,
            "screenshot": screenshot_data_uri,
            "format": "base64",
            "full_page": actual_full_page,  # Will be False if we fell back
            "wait_for_lazy_load": wait_for_lazy_load,
            "message": f"Screenshot captured successfully (full_page={actual_full_page}). Session ID: {session_id}",
        }
    except Exception as e:
        return {
            "success": False,
            "session_id": session_id,
            "screenshot": "",
            "format": "base64",
            "full_page": actual_full_page,
            "wait_for_lazy_load": wait_for_lazy_load,
            "message": "",
            "error": _format_error_with_traceback(
                e, f"Screenshot failed for session '{session_id}'"
            ),
        }


async def browser_extract_text(**kwargs: Any) -> Dict[str, Any]:
    """
    Extract text content from an element. Browser session is created automatically if needed.

    Args:
        session_id: Session ID
        selector: CSS selector or XPath (default: "body" for full page text)
        headless: Whether to run browser in headless mode (default: True)

    Returns:
        Dictionary with extracted text and metadata
    """
    # Extract parameters with validation
    session_id = kwargs.get("session_id")
    if not session_id:
        return {
            "success": False,
            "session_id": "",
            "selector": "",
            "text": "",
            "length": 0,
            "message": "",
            "error": "Missing required parameter: session_id is required",
        }
    selector = kwargs.get("selector", "body")

    if not PLAYWRIGHT_AVAILABLE:
        return {
            "success": False,
            "session_id": session_id,
            "selector": selector,
            "text": "",
            "length": 0,
            "message": "",
            "error": "Playwright is not installed or browsers not downloaded. Install with: pip install playwright. Then download browsers: playwright install chromium",
        }

    manager = get_browser_manager()
    session = await manager.get_or_create(session_id)
    page = await session.get_page()

    try:
        # Try inner_text() first with timeout
        element = page.locator(selector)
        try:
            # Wait for element to be attached with short timeout
            await element.wait_for(state="attached", timeout=5000)
            text = await element.inner_text(timeout=10000)
        except Exception as inner_error:
            # Fallback: use JavaScript textContent which is more reliable
            import json

            escaped_selector = json.dumps(selector)
            try:
                text = await page.evaluate(
                    f'document.querySelector({escaped_selector})?.textContent || ""'
                )
                if not text or not isinstance(text, str):
                    raise inner_error  # Re-raise original error if JS also fails
            except Exception:
                raise inner_error  # Re-raise original error

        return {
            "success": True,
            "session_id": session_id,
            "selector": selector,
            "text": text,
            "length": len(text),
            "message": f"Extracted {len(text)} characters from {selector}. Session ID: {session_id}",
        }
    except Exception as e:
        return {
            "success": False,
            "session_id": session_id,
            "selector": selector,
            "text": "",
            "length": 0,
            "message": "",
            "error": _format_error_with_traceback(
                e, f"Text extraction failed for selector '{selector}'"
            ),
        }


async def browser_evaluate(**kwargs: Any) -> Dict[str, Any]:
    """
    Execute JavaScript code in the browser. Browser session is created automatically if needed.

    Args:
        session_id: Session ID
        javascript: JavaScript code to execute
        headless: Whether to run browser in headless mode (default: True)

    Returns:
        Dictionary with execution result
    """
    # Extract parameters with validation
    session_id = kwargs.get("session_id")
    javascript = kwargs.get("javascript")
    if not session_id or not javascript:
        return {
            "success": False,
            "session_id": session_id or "",
            "result": None,
            "message": "",
            "error": "Missing required parameters: session_id and javascript are required",
        }

    if not PLAYWRIGHT_AVAILABLE:
        return {
            "success": False,
            "session_id": session_id,
            "result": None,
            "message": "",
            "error": "Playwright is not installed or browsers not downloaded. Install with: pip install playwright. Then download browsers: playwright install chromium",
        }

    manager = get_browser_manager()
    session = await manager.get_or_create(session_id)
    page = await session.get_page()

    try:
        result = await page.evaluate(javascript)

        return {
            "success": True,
            "session_id": session_id,
            "result": result,
            "message": f"JavaScript executed successfully. Session ID: {session_id}",
        }
    except Exception as e:
        return {
            "success": False,
            "session_id": session_id,
            "result": None,
            "message": "",
            "error": _format_error_with_traceback(e, "JavaScript evaluation failed"),
        }


async def browser_select_option(**kwargs: Any) -> Dict[str, Any]:
    """
    Select an option from a select dropdown. Browser session is created automatically if needed.

    Args:
        session_id: Session ID
        selector: CSS selector for the select element
        value: Option value to select
        index: Option index to select
        headless: Whether to run browser in headless mode (default: True)

    Returns:
        Dictionary with selection result
    """
    # Extract parameters with validation
    session_id = kwargs.get("session_id")
    selector = kwargs.get("selector")
    if not session_id or not selector:
        return {
            "success": False,
            "session_id": session_id or "",
            "selector": selector or "",
            "selected_value": "",
            "selected_index": None,
            "message": "",
            "error": "Missing required parameters: session_id and selector are required",
        }
    value = kwargs.get("value")
    index = kwargs.get("index")
    if value is None and index is None:
        return {
            "success": False,
            "session_id": session_id,
            "selector": selector,
            "selected_value": "",
            "selected_index": None,
            "message": "",
            "error": "Either value or index must be provided",
        }

    if not PLAYWRIGHT_AVAILABLE:
        return {
            "success": False,
            "session_id": session_id,
            "selector": selector,
            "selected_value": "",
            "selected_index": None,
            "message": "",
            "error": "Playwright is not installed or browsers not downloaded. Install with: pip install playwright. Then download browsers: playwright install chromium",
        }

    manager = get_browser_manager()
    session = await manager.get_or_create(session_id)
    page = await session.get_page()

    try:
        if value is not None:
            # Add timeout to prevent hanging (default 30 seconds)
            await page.select_option(selector, value=value, timeout=30000)
            return {
                "success": True,
                "session_id": session_id,
                "selector": selector,
                "selected_value": value,
                "message": f"Selected option with value: {value}. Session ID: {session_id}",
            }
        elif index is not None:
            # Add timeout to prevent hanging (default 30 seconds)
            await page.select_option(selector, index=index, timeout=30000)
            return {
                "success": True,
                "session_id": session_id,
                "selector": selector,
                "selected_index": index,
                "message": f"Selected option at index: {index}. Session ID: {session_id}",
            }
        else:
            return {
                "success": False,
                "session_id": session_id,
                "selector": selector,
                "selected_value": "",
                "selected_index": None,
                "message": "",
                "error": "Either value or index must be provided",
            }
    except Exception as e:
        return {
            "success": False,
            "session_id": session_id,
            "selector": selector,
            "selected_value": "",
            "selected_index": None,
            "message": "",
            "error": _format_error_with_traceback(
                e, f"Select option failed for selector '{selector}'"
            ),
        }


async def browser_wait_for_selector(**kwargs: Any) -> Dict[str, Any]:
    """
    Wait for an element to appear on the page. Browser session is created automatically if needed.

    Args:
        session_id: Session ID
        selector: CSS selector or XPath to wait for
        timeout: Timeout in milliseconds (default: 30000)
        headless: Whether to run browser in headless mode (default: True)

    Returns:
        Dictionary with wait result
    """
    # Extract parameters with validation
    session_id = kwargs.get("session_id")
    selector = kwargs.get("selector")
    if not session_id or not selector:
        return {
            "success": False,
            "session_id": session_id or "",
            "selector": selector or "",
            "message": "",
            "error": "Missing required parameters: session_id and selector are required",
        }
    timeout = kwargs.get("timeout", 30000)

    if not PLAYWRIGHT_AVAILABLE:
        return {
            "success": False,
            "session_id": session_id,
            "selector": selector,
            "message": "",
            "error": "Playwright is not installed or browsers not downloaded. Install with: pip install playwright. Then download browsers: playwright install chromium",
        }

    manager = get_browser_manager()
    session = await manager.get_or_create(session_id)
    page = await session.get_page()

    try:
        await page.wait_for_selector(selector, timeout=timeout)
        return {
            "success": True,
            "session_id": session_id,
            "selector": selector,
            "message": f"Element found: {selector}. Session ID: {session_id}",
        }
    except Exception as e:
        return {
            "success": False,
            "session_id": session_id,
            "selector": selector,
            "message": "",
            "error": _format_error_with_traceback(
                e, f"Timeout waiting for selector '{selector}'"
            ),
        }


async def browser_close(session_id: str) -> Dict[str, Any]:
    """
    Explicitly close a browser session and free resources.

    Args:
        session_id: Session ID to close

    Returns:
        Dictionary with close result
    """
    manager = get_browser_manager()
    await manager.close(session_id)

    return {
        "success": True,
        "session_id": session_id,
        "message": f"Browser session {session_id} closed. Session ID: {session_id}",
    }


async def browser_list_sessions() -> Dict[str, Any]:
    """
    List all active browser sessions (for debugging and monitoring).

    Returns:
        Dictionary with list of active sessions and their metadata
    """
    manager = get_browser_manager()
    async with manager._lock:
        sessions = [
            {
                "session_id": sid,
                "created_at": sess._created_at.isoformat(),
                "last_used": sess._last_used.isoformat(),
                "initialized": sess._initialized,
                "headless": sess.headless,
            }
            for sid, sess in manager._sessions.items()
        ]

    return {
        "success": True,
        "count": len(sessions),
        "sessions": sessions,
        "message": f"Found {len(sessions)} active session(s)",
    }


async def browser_pdf(**kwargs: Any) -> Dict[str, Any]:
    """
    Save current page as PDF. Browser session is created automatically if needed.

    Args:
        session_id: Session ID
        headless: Whether to run browser in headless mode (default: True)
        landscape: PDF orientation (default: False for portrait)
        format: Paper format (default: "A4"). Options: A4, Letter, etc.
        print_background: Include background graphics (default: True)

    Returns:
        Dictionary with PDF generation result (base64 encoded PDF data)
    """
    # Extract parameters with validation
    session_id = kwargs.get("session_id")
    if not session_id:
        return {
            "success": False,
            "session_id": "",
            "pdf": "",
            "message": "",
            "error": "Missing required parameter: session_id is required",
        }
    landscape = kwargs.get("landscape", False)
    format = kwargs.get("format", "A4")
    print_background = kwargs.get("print_background", True)

    if not PLAYWRIGHT_AVAILABLE:
        return {
            "success": False,
            "session_id": session_id,
            "pdf": "",
            "message": "",
            "error": "Playwright is not installed or browsers not downloaded. Install with: pip install playwright. Then download browsers: playwright install chromium",
        }

    manager = get_browser_manager()
    session = await manager.get_or_create(session_id)
    page = await session.get_page()

    try:
        import logging

        logger = logging.getLogger(__name__)
        logger.info(
            f"[browser_pdf] Generating PDF (session={session_id}, format={format}, landscape={landscape})"
        )

        # Generate PDF from current page
        pdf_bytes = await page.pdf(
            landscape=landscape,
            format=format,
            print_background=print_background,
        )

        # Encode to base64
        import base64

        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
        file_size = len(pdf_bytes)

        logger.info(f"[browser_pdf] PDF generated successfully ({file_size} bytes)")

        return {
            "success": True,
            "session_id": session_id,
            "pdf": pdf_base64,
            "format": "base64",
            "size": file_size,
            "message": f"PDF generated successfully ({file_size} bytes). Session ID: {session_id}",
        }
    except Exception as e:
        return {
            "success": False,
            "session_id": session_id,
            "pdf": "",
            "size": 0,
            "message": "",
            "error": _format_error_with_traceback(
                e, f"PDF generation failed for session '{session_id}'"
            ),
        }
