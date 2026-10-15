"""
Browser automation tool with lazy initialization.

This module provides browser automation capabilities using Playwright.
Browser sessions are created on first use (lazy initialization) and
automatically cleaned up after a timeout period.
"""

import asyncio
import json
import logging
from collections.abc import Sequence
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from http.cookiejar import CookieJar, DefaultCookiePolicy
from http.cookies import SimpleCookie
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, cast
from urllib.parse import unquote, urlsplit, urlunsplit

import httpx

from ....config import (
    get_browser_allow_unsandboxed,
    get_browser_tool_default_locale,
    get_browser_tool_default_timezone,
)
from ...utils.security import PublicNetworkAsyncHTTPTransport, validate_public_http_url
from .web_content import get_trusted_proxy_url

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


_BROWSER_RESPONSE_MAX_BYTES = 16 * 1024 * 1024
_BROWSER_REQUEST_DEADLINE_SECONDS = 30.0
_BROWSER_REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}
logger = logging.getLogger(__name__)


class _NoStoreCookiePolicy(DefaultCookiePolicy):
    """Keep upstream cookies out of httpx; Chromium owns the cookie jar."""

    def set_ok(self, cookie: Any, request: Any) -> bool:
        return False


def _sanitize_browser_url(value: str) -> str:
    """Keep credentials and query tokens out of browser security logs."""
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname or ""
        if parsed.port:
            hostname = f"{hostname}:{parsed.port}"
        return urlunsplit((parsed.scheme, hostname, parsed.path, "", ""))
    except (TypeError, ValueError):
        return "<invalid-url>"


def _allowed_file_path(
    url: str,
    *,
    allowed_roots: Sequence[str] = (),
    allowed_files: Sequence[str] = (),
) -> Path | None:
    """Return an explicitly authorized local file, or ``None``."""
    try:
        parsed = urlsplit(url)
        if parsed.scheme != "file" or parsed.netloc:
            return None
        candidate = Path(unquote(parsed.path)).resolve()
        roots = tuple(Path(root).resolve() for root in allowed_roots)
        files = tuple(Path(path).resolve() for path in allowed_files)
    except (OSError, ValueError):
        return None
    authorized = candidate in files or any(
        candidate.is_relative_to(root) for root in roots
    )
    return candidate if authorized and candidate.is_file() else None


async def _validate_browser_target(
    url: str,
    *,
    allowed_file_roots: Sequence[str] = (),
    allowed_files: Sequence[str] = (),
) -> None:
    """Validate a browser target and every request made by its page."""
    try:
        parsed = urlsplit(url)
    except ValueError as exc:
        raise ValueError("Malformed browser URL") from exc

    scheme = parsed.scheme.lower()
    if scheme == "file":
        if (
            _allowed_file_path(
                url, allowed_roots=allowed_file_roots, allowed_files=allowed_files
            )
            is None
        ):
            raise ValueError("Only authorized workspace or uploaded files are allowed")
        return
    if scheme not in {"http", "https"}:
        raise ValueError("Only HTTP(S) URLs are allowed")
    await validate_public_http_url(url)


class BrowserSession:
    """Browser session with lazy initialization."""

    def __init__(
        self,
        session_id: str,
        headless: bool = True,
        locale: Optional[str] = None,
        allowed_file_roots: Sequence[str] = (),
        allowed_files: Sequence[str] = (),
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
        self.allowed_file_roots = tuple(allowed_file_roots)
        self.allowed_files = tuple(allowed_files)
        # No caller ever supplies a per-request timezone (no tool schema
        # exposes it, and _with_default_session only injects locale) -- this
        # is deployment-configured only, unlike locale.
        self.timezone_id = get_browser_tool_default_timezone()
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._playwright: Any = None
        self._http_client: httpx.AsyncClient | None = None
        # Bound aggregate buffered response memory for one browser session:
        # eight routes times _BROWSER_RESPONSE_MAX_BYTES at most.
        self._route_gate = asyncio.Semaphore(8)
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

            # Resolve the proxy before allocating Playwright resources. A
            # configured but untrusted ambient proxy must fail without leaving
            # a browser process behind for the next request.
            proxy_url = get_trusted_proxy_url()
            self._playwright = await async_playwright().start()

            # Launch browser with anti-detection settings
            browser_args = [
                # Disable WebDriver detection
                "--disable-blink-features=AutomationControlled",
                "--disable-infobars",
                "--window-size=1920,1080",
                # Prevent browser-managed UDP/WebRTC paths from bypassing the
                # context route and the pinned HTTP transport.
                "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
            ]
            try:
                allow_unsandboxed = get_browser_allow_unsandboxed()
                if allow_unsandboxed:
                    browser_args.append("--no-sandbox")
                self._browser = await self._playwright.chromium.launch(
                    headless=self.headless,
                    args=browser_args,
                    chromium_sandbox=not allow_unsandboxed,
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
                    service_workers="block",
                )
            except Exception:
                await self.close()
                raise

            try:
                self._http_client = httpx.AsyncClient(
                    transport=PublicNetworkAsyncHTTPTransport(proxy_url=proxy_url),
                    # Let Chromium follow redirects through the context route. If
                    # httpx follows them here, the final response is fulfilled for
                    # the original Playwright request and keeps the wrong origin.
                    follow_redirects=False,
                    timeout=httpx.Timeout(30.0),
                    cookies=CookieJar(policy=_NoStoreCookiePolicy()),
                )
            except Exception:
                await self.close()
                raise

            try:
                await self._context.route("**/*", self._guard_route)
                await self._context.route_web_socket(
                    "**/*", lambda websocket: websocket.close(code=1008)
                )

                self._page = await self._context.new_page()

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

                    // Shared workers and peer connections can open sockets that
                    // Playwright's page routing cannot observe. Browser tasks do
                    // not need either capability, so keep them unavailable.
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
                # Context-level scripts also cover popup pages created with
                # window.open(), not only the initial page.
                await self._context.add_init_script(init_script)
            except Exception:
                await self.close()
                raise

            self._initialized = True

    async def _guard_route(self, route: Any) -> None:
        """Proxy one browser request through the pinned, bounded HTTP client."""
        acquired = False
        try:
            loop = asyncio.get_running_loop()
            await asyncio.wait_for(
                self._route_gate.acquire(), _BROWSER_REQUEST_DEADLINE_SECONDS
            )
            acquired = True
            deadline = loop.time() + _BROWSER_REQUEST_DEADLINE_SECONDS
            request_url = route.request.url
            if urlsplit(request_url).scheme == "file":
                await _validate_browser_target(
                    request_url,
                    allowed_file_roots=self.allowed_file_roots,
                    allowed_files=self.allowed_files,
                )
                await route.continue_()
                return
            assert self._http_client is not None
            is_navigation_request = bool(route.request.is_navigation_request())
            request_headers = await route.request.all_headers()
            # Playwright may omit cookies from ``all_headers`` for a request
            # that was initiated by a worker or by a redirect.  Reconstruct
            # the browser's cookie header from the isolated context so the
            # upstream request has the same authenticated state as Chromium.
            if self._context is not None and not any(
                header.lower() == "cookie" for header in request_headers
            ):
                browser_cookies = await self._context.cookies([request_url])
                if browser_cookies:
                    request_headers["cookie"] = "; ".join(
                        f"{cookie['name']}={cookie['value']}"
                        for cookie in browser_cookies
                        if cookie.get("name") is not None
                        and cookie.get("value") is not None
                    )
            for header in (
                "accept-encoding",
                "connection",
                "content-length",
                "host",
                "proxy-authorization",
                "proxy-connection",
            ):
                request_headers.pop(header, None)
            # Keep encoded bytes out of Python's decoder. Chromium can still
            # decode a server that ignores this preference, while normal
            # responses arrive uncompressed and bounded.
            request_headers["accept-encoding"] = "identity"
            request = self._http_client.build_request(
                route.request.method,
                request_url,
                headers=request_headers,
                content=route.request.post_data_buffer,
            )
            redirect_history: list[tuple[str, list[str]]] = []
            response: httpx.Response
            async with asyncio.timeout(max(0.0, deadline - loop.time())):
                # Playwright invokes a route handler only for the first hop
                # in a redirect chain. Let HTTPX follow redirects for
                # subresources so its transport validates and pins every
                # hop. Navigation redirects are handled below with a fresh
                # browser navigation instead, preserving the final origin.
                current_request = request
                redirect_url: str | None = None
                for _ in range(21):
                    response = await self._http_client.send(
                        current_request,
                        stream=True,
                        follow_redirects=False,
                    )
                    if (
                        response.status_code not in _BROWSER_REDIRECT_STATUS_CODES
                        or not response.headers.get("location")
                    ):
                        break
                    next_request = response.next_request
                    if next_request is None:
                        await response.aclose()
                        raise ValueError("Browser redirect could not be replayed")
                    redirect_url = str(next_request.url)
                    try:
                        await _validate_browser_target(
                            redirect_url,
                            allowed_file_roots=self.allowed_file_roots,
                            allowed_files=self.allowed_files,
                        )
                    except Exception:
                        await response.aclose()
                        raise
                    if redirect_url == str(current_request.url):
                        await response.aclose()
                        raise ValueError("Browser redirect loop detected")
                    redirect_history.append(
                        (
                            str(current_request.url),
                            response.headers.get_list("set-cookie"),
                        )
                    )
                    if is_navigation_request:
                        if next_request.method != "GET":
                            await response.aclose()
                            raise ValueError(
                                "Redirected browser navigations must use GET"
                            )
                        break
                    await self._apply_response_cookies(
                        str(current_request.url),
                        response.headers.get_list("set-cookie"),
                    )
                    if self._context is not None:
                        next_request.headers.pop("cookie", None)
                        browser_cookies = await self._context.cookies([redirect_url])
                        if browser_cookies:
                            next_request.headers["cookie"] = "; ".join(
                                f"{cookie['name']}={cookie['value']}"
                                for cookie in browser_cookies
                                if cookie.get("name") is not None
                                and cookie.get("value") is not None
                            )
                    await response.aclose()
                    current_request = next_request
                else:
                    raise ValueError("Browser redirect limit exceeded")

                try:
                    chunks: list[bytes] = []
                    content_size = 0
                    async for chunk in response.aiter_raw():
                        content_size += len(chunk)
                        if content_size > _BROWSER_RESPONSE_MAX_BYTES:
                            raise ValueError("Browser response exceeded size limit")
                        chunks.append(chunk)
                finally:
                    await response.aclose()
            response_headers: dict[str, str] = {}
            for header, value in response.headers.multi_items():
                key = header.lower()
                if key in {
                    "connection",
                    "content-length",
                    "set-cookie",
                    "transfer-encoding",
                }:
                    continue
                if key in response_headers:
                    response_headers[key] = f"{response_headers[key]}, {value}"
                else:
                    response_headers[key] = value
            if is_navigation_request:
                for response_url, set_cookie_headers in redirect_history:
                    await self._apply_response_cookies(response_url, set_cookie_headers)
            final_response_url = str(
                getattr(getattr(response, "request", None), "url", "") or request_url
            )
            await self._apply_response_cookies(
                final_response_url, response.headers.get_list("set-cookie")
            )
            if is_navigation_request and redirect_url is not None:
                # A fulfilled 3xx is followed by Chromium without invoking
                # the route handler again. Return a same-origin document that
                # starts a fresh navigation, so the redirect target is a new
                # routed request and receives the same egress checks.
                redirect_target = (
                    json.dumps(redirect_url)
                    .replace("<", "\\u003c")
                    .replace(">", "\\u003e")
                    .replace("&", "\\u0026")
                )
                redirect_document = (
                    '<!doctype html><meta charset="utf-8">'
                    "<script>location.replace("
                    f"{redirect_target}"
                    ");</script>"
                )
                await route.fulfill(
                    status=200,
                    headers={"content-type": "text/html; charset=utf-8"},
                    body=redirect_document.encode(),
                )
                return
            # Chromium's Cookie header was forwarded explicitly above. Never
            # let httpx replay upstream Set-Cookie values on another request.
            self._http_client.cookies.clear()
            await route.fulfill(
                status=response.status_code,
                headers=response_headers,
                body=b"".join(chunks),
            )
        except Exception as exc:
            logger.warning(
                "Browser request blocked (%s): %s",
                type(exc).__name__,
                _sanitize_browser_url(getattr(route.request, "url", "")),
            )
            try:
                await route.abort()
            except Exception:
                logger.debug("Browser route was already closed", exc_info=True)
        finally:
            if acquired:
                self._route_gate.release()

    async def _apply_response_cookies(
        self, request_url: str, set_cookie_headers: Sequence[str]
    ) -> None:
        """Transfer every Set-Cookie line without collapsing duplicates."""
        if not set_cookie_headers or self._context is None:
            return
        request_host = (urlsplit(request_url).hostname or "").rstrip(".").lower()
        request_parts = urlsplit(request_url)
        request_path = request_parts.path or "/"
        default_path = request_path.rsplit("/", 1)[0] or "/"
        for header in set_cookie_headers:
            parsed = SimpleCookie()
            try:
                parsed.load(header)
            except Exception:
                continue
            for morsel in parsed.values():
                cookie: dict[str, Any] = {
                    "name": morsel.key,
                    "value": morsel.value,
                    "domain": request_host,
                    "path": morsel["path"] or default_path,
                }
                if morsel["domain"]:
                    cookie_domain = morsel["domain"].lstrip(".").rstrip(".").lower()
                    if request_host != cookie_domain and not request_host.endswith(
                        f".{cookie_domain}"
                    ):
                        continue
                    cookie["domain"] = morsel["domain"]
                if morsel["secure"]:
                    cookie["secure"] = True
                if morsel["httponly"]:
                    cookie["httpOnly"] = True
                same_site = morsel["samesite"].capitalize()
                if same_site in {"Strict", "Lax", "None"}:
                    cookie["sameSite"] = same_site
                if morsel["max-age"]:
                    try:
                        cookie["expires"] = datetime.now().timestamp() + int(
                            morsel["max-age"]
                        )
                    except ValueError:
                        pass
                elif morsel["expires"]:
                    try:
                        cookie["expires"] = parsedate_to_datetime(
                            morsel["expires"]
                        ).timestamp()
                    except (TypeError, ValueError):
                        pass
                try:
                    await self._context.add_cookies(cast(Any, [cookie]))
                except Exception:
                    # A malformed cookie must not discard an otherwise valid
                    # response. Chromium applies the same best-effort rule.
                    logger.debug("Ignoring invalid upstream cookie", exc_info=True)

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
            if self._http_client:
                await self._http_client.aclose()
                self._http_client = None
        except Exception:
            pass  # Ignore client close errors

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
        self,
        session_id: str,
        headless: bool = False,
        locale: Optional[str] = None,
        allowed_file_roots: Sequence[str] = (),
        allowed_files: Sequence[str] = (),
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
                # Keep grants for the lifetime of a session. Navigation is
                # often followed by click/screenshot/PDF calls that do not
                # carry the workspace arguments again; replacing them would
                # abort legitimate file subresources after the first tool.
                existing.allowed_file_roots = tuple(
                    dict.fromkeys((*existing.allowed_file_roots, *allowed_file_roots))
                )
                existing.allowed_files = tuple(
                    dict.fromkeys((*existing.allowed_files, *allowed_files))
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
                allowed_file_roots=allowed_file_roots,
                allowed_files=allowed_files,
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
    allowed_file_roots = kwargs.get("_allowed_file_roots")
    if not isinstance(allowed_file_roots, (list, tuple)) or not all(
        isinstance(root, str) for root in allowed_file_roots
    ):
        allowed_file_roots = ()
    allowed_files = kwargs.get("_allowed_files")
    if not isinstance(allowed_files, (list, tuple)) or not all(
        isinstance(path, str) for path in allowed_files
    ):
        allowed_files = ()
    try:
        await _validate_browser_target(
            str(url),
            allowed_file_roots=allowed_file_roots,
            allowed_files=allowed_files,
        )
    except (OSError, TypeError, ValueError) as exc:
        return {
            "success": False,
            "session_id": session_id,
            "url": url,
            "title": "",
            "message": "",
            "error": str(exc) or "Only public network URLs are allowed",
        }
    headless = kwargs.get("headless", False)
    wait_until = kwargs.get("wait_until", "networkidle")
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

    try:
        manager = get_browser_manager()
        session = await manager.get_or_create(
            session_id,
            headless,
            locale=locale,
            allowed_file_roots=allowed_file_roots,
            allowed_files=allowed_files,
        )
        page = await session.get_page()
    except Exception as exc:
        logger.warning(
            "[browser_navigate] Failed to initialize browser session (%s)",
            type(exc).__name__,
        )
        return {
            "success": False,
            "session_id": session_id,
            "url": url,
            "title": "",
            "message": "",
            "error": str(exc) or "Browser session initialization failed",
        }

    try:
        # Log navigation attempt for debugging
        logger.info(
            f"[browser_navigate] Navigating to {url} (headless={headless}, wait_until={wait_until})"
        )

        # Add timeout to prevent hanging (default 30 seconds)
        # Using wait_until parameter (default: 'networkidle', but can be 'domcontentloaded' or 'load')
        # networkidle can hang on pages with continuous background requests
        await page.goto(url, wait_until=wait_until, timeout=30000)
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
        import traceback

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
