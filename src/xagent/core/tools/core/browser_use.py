"""
Browser automation tool with lazy initialization.

This module provides browser automation capabilities using Playwright.
Browser sessions are created on first use (lazy initialization) and
automatically cleaned up after a timeout period.
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Sequence, cast
from urllib.parse import unquote, urlsplit

from ....config import (
    get_browser_allow_unsandboxed,
    get_browser_tool_default_locale,
    get_browser_tool_default_timezone,
)
from ...utils.public_network_proxy import PublicNetworkEgressProxy
from ...utils.security import validate_public_http_url

logger = logging.getLogger(__name__)

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


def _workspace_file_path(
    url: str,
    workspace_root: str | None = None,
    workspace_roots: Sequence[str] | None = None,
    workspace_file_paths: Sequence[str] | None = None,
) -> Path | None:
    """Return a file contained by an authorized root, or ``None`` if unsafe."""
    roots = [workspace_root] if workspace_root else []
    roots.extend(workspace_roots or ())
    if not roots and not workspace_file_paths:
        return None
    try:
        parsed = urlsplit(url)
        if parsed.scheme.lower() != "file" or parsed.netloc:
            return None
        candidate = Path(unquote(parsed.path)).expanduser().resolve()
    except (OSError, ValueError):
        return None
    if not candidate.is_file():
        return None
    for file_path in workspace_file_paths or ():
        if not isinstance(file_path, str) or not file_path:
            continue
        try:
            if candidate == Path(file_path).expanduser().resolve():
                return candidate
        except (OSError, ValueError):
            continue
    for root_value in roots:
        if not isinstance(root_value, str) or not root_value:
            continue
        try:
            candidate.relative_to(Path(root_value).expanduser().resolve())
        except (OSError, ValueError):
            continue
        return candidate
    return None


async def _validate_browser_target(
    url: str,
    *,
    workspace_root: str | None = None,
    workspace_roots: Sequence[str] | None = None,
    workspace_file_paths: Sequence[str] | None = None,
) -> None:
    """Validate a browser target and every request made by its page."""
    try:
        parsed = urlsplit(url)
    except ValueError as exc:
        raise ValueError("Malformed browser URL") from exc

    scheme = parsed.scheme.lower()
    if scheme in {"about", "data", "blob"}:
        # These URLs do not perform a network connection by themselves. Any
        # HTTP(S) fetches initiated by their document still pass through the
        # CDP guard and the browser-wide egress proxy.
        return
    if scheme == "file":
        if (
            _workspace_file_path(
                url,
                workspace_root=workspace_root,
                workspace_roots=workspace_roots,
                workspace_file_paths=workspace_file_paths,
            )
            is None
        ):
            raise ValueError("Only workspace files are allowed")
        return
    if scheme not in {"http", "https"}:
        raise ValueError("Only HTTP(S) URLs are allowed")
    try:
        await validate_public_http_url(url)
    except OSError as exc:
        raise ValueError("Browser target could not be resolved") from exc


_BROWSER_WORKSPACE_AUTH = object()

_HTTP_FETCH_PATTERNS = [
    {"urlPattern": "http://*", "requestStage": "Request"},
    {"urlPattern": "https://*", "requestStage": "Request"},
    {"urlPattern": "ws://*", "requestStage": "Request"},
    {"urlPattern": "wss://*", "requestStage": "Request"},
]


class BrowserSession:
    """Browser session with lazy initialization."""

    def __init__(
        self,
        session_id: str,
        headless: bool = True,
        locale: Optional[str] = None,
        workspace_root: str | None = None,
        workspace_roots: Sequence[str] | None = None,
        workspace_file_paths: Sequence[str] | None = None,
    ):
        """
        Initialize a browser session.

        Args:
            session_id: Unique identifier for this session
            headless: Whether to run browser in headless mode
            locale: Playwright context locale (e.g. "en-US"). Falls back to
                the deployment default (see get_browser_tool_default_locale)
                when not supplied by the caller.
        """
        self.session_id = session_id
        self.headless = headless
        self.locale = locale or get_browser_tool_default_locale()
        self.workspace_root = workspace_root
        self.workspace_roots = tuple(
            dict.fromkeys(
                root
                for root in ((workspace_root,) if workspace_root else ())
                + tuple(workspace_roots or ())
                if isinstance(root, str) and root
            )
        )
        self.workspace_file_paths = tuple(
            dict.fromkeys(
                file_path
                for file_path in (workspace_file_paths or ())
                if isinstance(file_path, str) and file_path
            )
        )
        # No caller ever supplies a per-request timezone (no tool schema
        # exposes it, and _with_default_session only injects locale) -- this
        # is deployment-configured only, unlike locale.
        self.timezone_id = get_browser_tool_default_timezone()
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._playwright: Any = None
        self._cdp_sessions: list[Any] = []
        self._cdp_tasks: set[asyncio.Task[Any]] = set()
        self._guarded_pages: set[int] = set()
        self._page_guard_futures: dict[int, asyncio.Future[None]] = {}
        self._cdp_sessions_by_page: dict[int, Any] = {}
        self._egress_proxy: PublicNetworkEgressProxy | None = None
        self._created_at = datetime.now()
        self._last_used = datetime.now()
        self._initialized = False

    async def _guard_page_requests(self, page: Page) -> None:
        """Validate network requests, including HTTP redirect hops.

        Playwright's route API invokes a handler only for the first request in
        a redirect chain. The Chromium Fetch domain emits a pause for each HTTP
        hop, so using it here keeps Chromium's cookie and redirect semantics
        while applying the URL policy to network subresources and redirects.
        Local-file requests are gated by the context route installed before
        the first page exists.
        """
        page_key = id(page)
        if self._context is None:
            return
        existing = self._page_guard_futures.get(page_key)
        if existing is not None:
            await asyncio.shield(existing)
            return
        if page_key in self._guarded_pages:
            return
        loop = asyncio.get_running_loop()
        ready = loop.create_future()
        ready.add_done_callback(
            lambda future: (
                future.exception() if not future.cancelled() and future.done() else None
            )
        )
        self._page_guard_futures[page_key] = ready
        self._guarded_pages.add(page_key)
        cdp = None
        try:
            cdp = await self._context.new_cdp_session(page)
        except BaseException as exc:
            self._guarded_pages.discard(page_key)
            self._page_guard_futures.pop(page_key, None)
            if isinstance(exc, asyncio.CancelledError):
                ready.cancel()
            elif not ready.done():
                ready.set_exception(exc)
            raise

        async def handle_paused(params: dict[str, Any]) -> None:
            request_id = params.get("requestId")
            request = params.get("request") or {}
            request_url = request.get("url", "")
            try:
                await _validate_browser_target(
                    request_url,
                    workspace_root=self.workspace_root,
                    workspace_roots=self.workspace_roots,
                    workspace_file_paths=self.workspace_file_paths,
                )
            except Exception:
                # Fail closed if validation or DNS resolution fails. Leaving
                # a paused request unresolved would stall the whole page.
                await cdp.send(
                    "Fetch.failRequest",
                    {"requestId": request_id, "errorReason": "BlockedByClient"},
                )
                return
            await cdp.send("Fetch.continueRequest", {"requestId": request_id})

        def on_paused(params: dict[str, Any]) -> None:
            task = asyncio.create_task(handle_paused(params))
            self._cdp_tasks.add(task)

            def finish_request(completed: asyncio.Task[Any]) -> None:
                self._cdp_tasks.discard(completed)
                if completed.cancelled():
                    return
                error = completed.exception()
                if error is not None:
                    logger.debug(
                        "Browser CDP request guard failed",
                        exc_info=(type(error), error, error.__traceback__),
                    )

            task.add_done_callback(finish_request)

        cdp.on("Fetch.requestPaused", on_paused)
        try:
            await cdp.send(
                "Fetch.enable",
                {"patterns": _HTTP_FETCH_PATTERNS},
            )
        except BaseException as exc:
            self._guarded_pages.discard(page_key)
            self._page_guard_futures.pop(page_key, None)
            try:
                await cdp.detach()
            except Exception:
                pass
            if isinstance(exc, asyncio.CancelledError):
                ready.cancel()
            elif not ready.done():
                ready.set_exception(exc)
            raise
        self._cdp_sessions.append(cdp)
        self._cdp_sessions_by_page[page_key] = cdp

        def forget_page(*_args: Any) -> None:
            self._guarded_pages.discard(page_key)
            ready = self._page_guard_futures.pop(page_key, None)
            if ready is not None and not ready.done():
                ready.cancel()
            page_cdp = self._cdp_sessions_by_page.pop(page_key, None)
            if page_cdp is None:
                return
            try:
                self._cdp_sessions.remove(page_cdp)
            except ValueError:
                pass

            async def detach_page_cdp() -> None:
                try:
                    await page_cdp.send("Fetch.disable")
                except Exception:
                    pass
                try:
                    await page_cdp.detach()
                except Exception:
                    pass

            cleanup = asyncio.create_task(detach_page_cdp())
            self._cdp_tasks.add(cleanup)

            def finish_cleanup(done: asyncio.Task[Any]) -> None:
                self._cdp_tasks.discard(done)
                if not done.cancelled():
                    done.exception()

            cleanup.add_done_callback(finish_cleanup)

        page.on("close", forget_page)
        if not ready.done():
            ready.set_result(None)

    async def _guard_context_route(self, route: Any) -> None:
        """Gate local-file requests before popup or frame navigation starts.

        A page-level CDP session is installed asynchronously after Chromium
        creates a popup target.  BrowserContext routing is registered before
        the first page exists, so it also covers a popup's first request and
        requests from out-of-process iframes.  HTTP(S)/WebSocket traffic is
        intentionally left to the browser-wide pinned egress proxy and the
        CDP redirect guard; this route is the synchronous local-file boundary
        that the proxy cannot enforce.
        """
        request_url = route.request.url
        try:
            scheme = urlsplit(request_url).scheme.lower()
        except ValueError:
            scheme = ""
        if scheme != "file":
            await route.fallback()
            return
        try:
            await _validate_browser_target(
                request_url,
                workspace_root=self.workspace_root,
                workspace_roots=self.workspace_roots,
                workspace_file_paths=self.workspace_file_paths,
            )
        except Exception:
            logger.info("Blocked browser local-file request: %s", request_url)
            await route.abort("blockedbyclient")
            return
        await route.fallback()

    def _schedule_page_guard(self, page: Page) -> None:
        task = asyncio.create_task(self._guard_page_requests(page))
        self._cdp_tasks.add(task)

        def finish(completed: asyncio.Task[Any]) -> None:
            self._cdp_tasks.discard(completed)
            if completed.cancelled():
                return
            error = completed.exception()
            if error is not None:
                logger.error(
                    "Failed to guard popup page requests",
                    exc_info=(type(error), error, error.__traceback__),
                )
                close_task = asyncio.create_task(page.close())
                self._cdp_tasks.add(close_task)

                def finish_close(done: asyncio.Task[Any]) -> None:
                    self._cdp_tasks.discard(done)
                    if not done.cancelled():
                        done.exception()

                close_task.add_done_callback(finish_close)

        task.add_done_callback(finish)

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
            browser_args = [
                # Disable WebDriver detection
                "--disable-blink-features=AutomationControlled",
                "--disable-infobars",
                "--window-size=1920,1080",
                # Prevent browser-managed transports from bypassing the
                # request guard. Browser tasks do not require WebRTC/QUIC.
                "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
                "--disable-quic",
            ]
            allow_unsandboxed = get_browser_allow_unsandboxed()
            if allow_unsandboxed:
                browser_args.append("--no-sandbox")
            try:
                self._egress_proxy = PublicNetworkEgressProxy()
                proxy_server = await self._egress_proxy.start()
                self._browser = await self._playwright.chromium.launch(
                    headless=self.headless,
                    args=browser_args,
                    chromium_sandbox=not allow_unsandboxed,
                    # Chromium otherwise has an implicit loopback bypass on
                    # some versions.  Make the proxy boundary explicit so
                    # loopback/private destinations cannot skip validation.
                    proxy={"server": proxy_server, "bypass": "<-loopback>"},
                )
            except Exception:
                await self.close()
                raise

            # Create context with realistic settings. locale comes from the
            # caller (ultimately the requesting user/task's own language);
            # timezone_id is deployment-configured only. Both fall back to
            # the deployment default -- previously hardcoded to
            # zh-CN/Asia/Shanghai, which forced every automated session to
            # request Chinese-localized pages regardless of who asked.
            try:
                self._context = await self._browser.new_context(
                    viewport={"width": 1920, "height": 1080},
                    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    locale=self.locale,
                    timezone_id=self.timezone_id,
                    # Service workers can own requests without going through
                    # Playwright routing. Browser security checks must see every
                    # page request, so keep them disabled for this context.
                    service_workers="block",
                )
            except Exception:
                await self.close()
                raise

            try:
                await self._context.route(
                    re.compile(r"^file:"), self._guard_context_route
                )
                self._context.on("page", self._schedule_page_guard)
                self._page = await self._context.new_page()
                await self._guard_page_requests(self._page)
            except Exception:
                await self.close()
                raise

            # Languages mirror the context's own locale instead of a fixed
            # list, so the spoofed navigator.languages stays consistent with
            # what the locale/Accept-Language header already imply. Built via
            # plain string substitution (not an f-string) since the rest of
            # this script is full of literal JS braces that an f-string would
            # require escaping.
            base_language = self.locale.split("-")[0]
            spoofed_languages = json.dumps(
                list(dict.fromkeys([self.locale, base_language, "en-US", "en"]))
            )

            # Inject script to hide webdriver property
            init_script = """
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
                    get: () => __XAGENT_LANGUAGES__
                });

                // These APIs can open sockets outside page request routing.
                // Browser tasks do not need them, so keep them unavailable.
                Object.defineProperty(globalThis, 'SharedWorker', {
                    value: undefined,
                    configurable: false,
                    writable: false
                });
                Object.defineProperty(globalThis, 'RTCPeerConnection', {
                    value: undefined,
                    configurable: false,
                    writable: false
                });
            """.replace("__XAGENT_LANGUAGES__", spoofed_languages)
            await self._context.add_init_script(init_script)

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
        while self._cdp_tasks:
            tasks = list(self._cdp_tasks)
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        self._cdp_tasks.clear()
        for cdp in self._cdp_sessions:
            try:
                await cdp.send("Fetch.disable")
            except Exception:
                pass
            try:
                await cdp.detach()
            except Exception:
                pass
        self._cdp_sessions.clear()
        self._cdp_sessions_by_page.clear()
        self._guarded_pages.clear()
        self._page_guard_futures.clear()

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

        if self._egress_proxy is not None:
            try:
                await self._egress_proxy.close()
            except Exception:
                pass
            self._egress_proxy = None

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
        self,
        session_id: str,
        headless: bool = False,
        locale: Optional[str] = None,
        workspace_root: str | None = None,
        workspace_roots: Sequence[str] | None = None,
        workspace_file_paths: Sequence[str] | None = None,
    ) -> BrowserSession:
        """
        Get or create a browser session (async-safe).

        Args:
            session_id: Unique session identifier
            headless: Whether to use headless mode (only used when creating new session)
            locale: Playwright context locale (only used when creating new session)

        Returns:
            BrowserSession instance
        """
        import logging

        logger = logging.getLogger(__name__)

        async with self._lock:
            # Check if session already exists
            if session_id in self._sessions:
                # Update last used time and return existing session
                # Ignore headless/locale for existing sessions (can't change
                # after creation)
                existing = self._sessions[session_id]
                if locale and existing.locale != locale:
                    logger.warning(
                        f"Browser session {session_id!r} already exists with locale "
                        f"{existing.locale!r}; requested locale {locale!r} ignored "
                        "(a session's locale is frozen at creation)"
                    )
                if workspace_root and not existing.workspace_root:
                    existing.workspace_root = workspace_root
                if workspace_roots and not existing.workspace_roots:
                    existing.workspace_roots = tuple(workspace_roots)
                if workspace_file_paths:
                    existing.workspace_file_paths = tuple(
                        dict.fromkeys(
                            (*existing.workspace_file_paths, *workspace_file_paths)
                        )
                    )
                existing._last_used = datetime.now()
                return existing

            # Start cleanup task on first session creation
            if self._cleanup_task is None:
                self._cleanup_task = asyncio.create_task(self._cleanup_loop())

            # Create new session with specified settings
            new_session = BrowserSession(
                session_id,
                headless,
                locale=locale,
                workspace_root=workspace_root,
                workspace_roots=workspace_roots,
                workspace_file_paths=workspace_file_paths,
            )
            self._sessions[session_id] = new_session
            new_session._last_used = datetime.now()
            # Log the resolved locale (falls back to the deployment default
            # when `locale` is None), not the raw argument -- otherwise the
            # one case an operator most wants visible here, the deployment
            # default kicking in, would read as "locale=None".
            logger.info(
                f"Creating browser session {session_id!r} (locale={new_session.locale!r})"
            )
            return new_session

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
        locale: Playwright context locale (only applied when the session is
            first created); falls back to the deployment default

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
    trusted_call = kwargs.get("_workspace_auth") is _BROWSER_WORKSPACE_AUTH
    workspace_root = kwargs.get("_workspace_root") if trusted_call else None
    workspace_roots = kwargs.get("_workspace_roots") if trusted_call else None
    workspace_file_paths = kwargs.get("_workspace_file_paths") if trusted_call else None
    trusted_workspace_roots = (
        tuple(root for root in workspace_roots if isinstance(root, str) and root)
        if isinstance(workspace_roots, (list, tuple))
        else None
    )
    trusted_workspace_file_paths = (
        tuple(path for path in workspace_file_paths if isinstance(path, str) and path)
        if isinstance(workspace_file_paths, (list, tuple))
        else None
    )
    try:
        await _validate_browser_target(
            str(url),
            workspace_root=workspace_root if isinstance(workspace_root, str) else None,
            workspace_roots=trusted_workspace_roots,
            workspace_file_paths=trusted_workspace_file_paths,
        )
    except (TypeError, ValueError):
        return {
            "success": False,
            "session_id": session_id,
            "url": url,
            "title": "",
            "message": "",
            "error": "Browser target is not allowed",
        }
    headless = kwargs.get("headless", False)
    wait_until = cast(
        Literal["commit", "domcontentloaded", "load", "networkidle"],
        kwargs.get("wait_until", "networkidle"),
    )
    locale = kwargs.get("locale")

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
    session = await manager.get_or_create(
        session_id,
        headless,
        locale=locale,
        workspace_root=workspace_root if isinstance(workspace_root, str) else None,
        workspace_roots=trusted_workspace_roots,
        workspace_file_paths=trusted_workspace_file_paths,
    )
    page = await session.get_page()

    try:
        wait_until_local: Literal[
            "commit", "domcontentloaded", "load", "networkidle"
        ] = (
            "domcontentloaded"
            if url.startswith("file://") and wait_until == "networkidle"
            else wait_until
        )

        # Log navigation attempt for debugging
        import logging

        logger = logging.getLogger(__name__)
        logger.info(
            f"[browser_navigate] Navigating to {url} (headless={headless}, wait_until={wait_until_local})"
        )

        # Add timeout to prevent hanging (default 30 seconds)
        # Using wait_until parameter (default: 'networkidle', but can be 'domcontentloaded' or 'load')
        # networkidle can hang on pages with continuous background requests
        await page.goto(url, wait_until=wait_until_local, timeout=30000)
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
            "message": (
                f"Navigation complete: opened {url}. This only confirms the page "
                "loaded; it does not inspect content, validate visual rendering, "
                "or create an image export. Do not call browser_navigate again for "
                "this same URL unless navigation failed or the target URL changed. "
                "Use browser_screenshot for visual inspection/rendered output, "
                "or browser_extract_text for page text. "
                f"Session ID: '{session_id}'."
            ),
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
    session = await manager.get_or_create(session_id, locale=kwargs.get("locale"))
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
    session = await manager.get_or_create(session_id, locale=kwargs.get("locale"))
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

    session = await manager.get_or_create(session_id, locale=kwargs.get("locale"))
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
    session = await manager.get_or_create(session_id, locale=kwargs.get("locale"))
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
    session = await manager.get_or_create(session_id, locale=kwargs.get("locale"))
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
    session = await manager.get_or_create(session_id, locale=kwargs.get("locale"))
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
    session = await manager.get_or_create(session_id, locale=kwargs.get("locale"))
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


async def browser_close(session_id: str, **_kwargs: Any) -> Dict[str, Any]:
    """
    Explicitly close a browser session and free resources.

    Args:
        session_id: Session ID to close
        **_kwargs: Accepted and ignored. ``BrowserTaskSessionMixin._with_default_session``
            injects a ``locale`` kwarg into every tool call once a task locale is
            resolved, regardless of which tool happens to create the session first;
            every sibling browser_* function already tolerates that via **kwargs.

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
    session = await manager.get_or_create(session_id, locale=kwargs.get("locale"))
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
