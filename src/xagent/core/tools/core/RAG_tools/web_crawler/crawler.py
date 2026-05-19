"""Core web crawler implementation."""

import asyncio
import importlib
import logging
import time
from collections import deque
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Set, Tuple

import httpx

from ..core.schemas import CrawlResult, WebCrawlConfig
from .content_cleaner import ContentCleaner
from .link_extractor import LinkExtractor
from .url_filter import URLFilter

logger = logging.getLogger(__name__)

# Default fallback chain when WebCrawlConfig.tls_impersonate == "auto".
#
# Picks are based on a production-faithful matrix run on 2026-05-08
# from a single Aliyun outbound IP, against 9 sites (Cloudflare /
# Akamai / AWS-WAF / Imperva / Alibaba) with NO custom headers passed
# to curl_cffi (so the impersonate spec owns UA / Accept / Sec-Fetch
# headers consistently with its TLS fingerprint -- mirroring this
# crawler's actual call shape). Of the seven specs probed, four
# returned real content on 9/9 sites: chrome116, chrome110,
# safari17_0, safari15_5. chrome131/124/120 were challenge/blocked on
# detrack.com and medium.com; chrome120 in particular looked OK in an
# earlier run that mixed a Chrome 130 UA on top of curl_cffi's TLS,
# but failed once the test was made consistent with production.
#
# Why this specific triple:
#   - chrome116:   Chrome family, recent enough to look mainstream,
#                  not yet on Cloudflare's "frequently-abused" list.
#   - safari17_0:  Different TLS stack (WebKit) -- if Cloudflare adds
#                  Chrome-family blocks, Safari is unlikely to share
#                  the same JA3/JA4 signature.
#   - safari15_5:  Older Safari, same family as safari17_0 but with
#                  a distinct fingerprint; the third backup if both
#                  primary picks ever get added to a blocklist.
#
# Caveats: (1) results from a single IP/ASN do not generalize -- a
# different network may see different blocks. (2) WAF rules drift;
# if INFO logs start showing fallback hits dominated by chain[2],
# revisit this list.
_TLS_AUTO_CHAIN: Tuple[str, ...] = ("chrome116", "safari17_0", "safari15_5")

# HTTP statuses where retrying the request with a different TLS fingerprint
# has a chance of success. Statuses NOT in this set (e.g. 404, 401, 500)
# almost always indicate a content-side problem that won't change between
# fingerprints; retrying just wastes the rest of the chain and writes
# misleading "TLS fallback exhausted" log lines for ordinary errors.
_WAF_RETRY_STATUSES: frozenset = frozenset(
    {403, 429, 503, 520, 521, 522, 523, 524, 525, 526}
)

# Heuristic markers for a Cloudflare / similar JS challenge wrapper page.
# These pages return HTTP 200 but the body is a "Just a moment..." stub --
# accepting them as successful crawls would pollute the KB with garbage.
_CHALLENGE_PAGE_MARKERS: Tuple[str, ...] = (
    "checking your browser",
    "cf-challenge",
    "just a moment",
    "cf-please-wait",
    "needs to review the security",
)


def _looks_like_challenge_page(body: str) -> bool:
    """Heuristic: does this 200 response look like a JS challenge wrapper?"""
    if not body:
        return False
    lowered = body[:3000].lower()
    return any(marker in lowered for marker in _CHALLENGE_PAGE_MARKERS)


def _ensure_curl_cffi_available() -> None:
    """Validate optional curl_cffi dependency before starting async crawl work."""
    try:
        importlib.import_module("curl_cffi.requests")
    except ModuleNotFoundError as e:
        if e.name and e.name.startswith("curl_cffi"):
            raise ImportError(
                "WebCrawlConfig.tls_impersonate requires the optional "
                "curl_cffi dependency. Install it with `pip install "
                "'xagent[waf-crawl]'` or set tls_impersonate=None to use "
                "plain httpx."
            ) from e
        raise


# Default User-Agent used when tls_impersonate is None and the user has
# not overridden config.user_agent.
_DEFAULT_USER_AGENT: str = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0.0.0 Safari/537.36"
)

# Mapping from a curl_cffi impersonate spec to the User-Agent that spec
# actually puts on the wire. We need this for policy code (robots.txt)
# which has to reason about the identity we are presenting -- since
# curl_cffi sets the UA itself based on the impersonate spec, the
# crawler does not have a direct way to read it back. Specs not in this
# table fall back to "*" for robots.txt purposes (which matches catch-all
# rules and is the safe conservative choice).
_IMPERSONATE_TO_UA: Dict[str, str] = {
    "chrome110": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/110.0.0.0 Safari/537.36"
    ),
    "chrome116": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/116.0.0.0 Safari/537.36"
    ),
    "chrome120": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "chrome124": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "chrome131": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "safari15_5": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/15.5 Safari/605.1.15"
    ),
    "safari17_0": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/17.0 Safari/605.1.15"
    ),
}


class WebCrawler:
    """Asynchronous web crawler with configurable filtering and rate limiting."""

    def __init__(
        self,
        config: WebCrawlConfig,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        *,
        trace_id: Optional[str] = None,
    ):
        """Initialize web crawler.

        Args:
            config: Crawl configuration
            progress_callback: Optional callback for progress updates
                Args: (current_url, completed, total)
        """
        self.config = config
        self.progress_callback = progress_callback
        self.trace_id = trace_id

        # Initialize components
        self.url_filter = URLFilter(
            base_url=config.start_url,
            same_domain_only=config.same_domain_only,
            url_patterns=config.url_patterns,
            exclude_patterns=config.exclude_patterns,
            respect_robots_txt=config.respect_robots_txt,
        )
        self.content_cleaner = ContentCleaner(
            content_selector=config.content_selector,
            remove_selectors=config.remove_selectors,
            trace_id=trace_id,
        )
        self.link_extractor = LinkExtractor(config.start_url)

        # Crawl state
        self.visited_urls: Set[str] = set()
        self.pending_urls: deque = deque()
        self.crawl_results: List[CrawlResult] = []
        self.failed_urls: Dict[str, str] = {}

        # Statistics
        self.total_urls_found = 0
        self.start_time: Optional[float] = None

        # Resolve the TLS-fingerprint sequence to use for fetches.
        # Element type is Optional[str]: None means "use httpx, no impersonation".
        if config.tls_impersonate is None:
            self._tls_chain: Tuple[Optional[str], ...] = (None,)
        elif config.tls_impersonate == "auto":
            _ensure_curl_cffi_available()
            self._tls_chain = _TLS_AUTO_CHAIN
        else:
            _ensure_curl_cffi_available()
            self._tls_chain = (config.tls_impersonate,)

        # Effective User-Agent for policy decisions (robots.txt, etc.).
        if config.tls_impersonate is None:
            self._policy_user_agent: str = config.user_agent or _DEFAULT_USER_AGENT
        else:
            first_fp = next((fp for fp in self._tls_chain if fp is not None), None)
            self._policy_user_agent = (
                _IMPERSONATE_TO_UA.get(first_fp, "*") if first_fp else "*"
            )

        # Playwright runtime (optional, used when render_js=True)
        self._playwright: Any | None = None
        self._browser: Any | None = None
        self._browser_context: Any | None = None

    async def crawl(self) -> List[CrawlResult]:
        """Start crawling from the configured start URL.

        Returns:
            List of crawl results
        """
        self.start_time = time.time()

        # Add start URL to pending
        start_url_normalized = self.url_filter.normalize_url(self.config.start_url)
        if start_url_normalized:
            self.pending_urls.append((start_url_normalized, 0))  # (url, depth)
        else:
            self.failed_urls[self.config.start_url] = (
                "Invalid start_url: must start with http:// or https://"
            )
            logger.error(
                "Start URL normalization failed; crawl will not run",
                extra={
                    "trace_id": self.trace_id,
                    "start_url": self.config.start_url,
                    "base_domain": getattr(self.url_filter, "base_domain", None),
                    "same_domain_only": self.config.same_domain_only,
                },
            )
            # Fail fast: without a valid start URL, the crawl loop would do nothing.
            # We record the failure in failed_urls so upstream can surface an error.
            self.start_time = self.start_time or time.time()
            return []

        logger.info(
            "Starting crawl",
            extra={
                "trace_id": self.trace_id,
                "start_url": self.config.start_url,
                "start_url_normalized": start_url_normalized,
                "max_pages": self.config.max_pages,
                "max_depth": self.config.max_depth,
                "same_domain_only": self.config.same_domain_only,
                "url_patterns": self.config.url_patterns,
                "exclude_patterns": self.config.exclude_patterns,
                "respect_robots_txt": self.config.respect_robots_txt,
                "content_selector": self.config.content_selector,
                "remove_selectors": self.config.remove_selectors,
                "concurrent_requests": self.config.concurrent_requests,
                "request_delay": self.config.request_delay,
                "timeout": self.config.timeout,
                "user_agent": self.config.user_agent,
                "tls_impersonate": self.config.tls_impersonate,
                "render_js": self.config.render_js,
            },
        )

        try:
            if self.config.render_js:
                ua = self.config.user_agent or _DEFAULT_USER_AGENT
                await self._start_playwright(user_agent=ua)
                await self._crawl_loop({})
            else:
                httpx_headers: Optional[Dict[str, str]] = None
                if None in self._tls_chain:
                    user_agent = self.config.user_agent or _DEFAULT_USER_AGENT
                    httpx_headers = {
                        "User-Agent": user_agent,
                        "Accept": (
                            "text/html,application/xhtml+xml,application/xml;q=0.9,"
                            "image/avif,image/webp,*/*;q=0.8"
                        ),
                        "Accept-Language": "en-US,en;q=0.9",
                        "Accept-Encoding": "gzip, deflate, br",
                        "Sec-Fetch-Dest": "document",
                        "Sec-Fetch-Mode": "navigate",
                        "Sec-Fetch-Site": "none",
                        "Sec-Fetch-User": "?1",
                        "Upgrade-Insecure-Requests": "1",
                        "DNT": "1",
                    }

                async with self._open_sessions(httpx_headers) as sessions:
                    await self._crawl_loop(sessions)
        finally:
            if self.config.render_js:
                await self._stop_playwright()

        elapsed = time.time() - self.start_time
        logger.info(
            "Crawl completed",
            extra={
                "trace_id": self.trace_id,
                "start_url": self.config.start_url,
                "pages": len(self.crawl_results),
                "failed": len(self.failed_urls),
                "visited_urls": len(self.visited_urls),
                "pending_urls": len(self.pending_urls),
                "total_urls_found": self.total_urls_found,
                "elapsed_sec": round(elapsed, 3),
            },
        )

        return self.crawl_results

    @asynccontextmanager
    async def _open_sessions(
        self, httpx_headers: Optional[Dict[str, str]]
    ) -> AsyncIterator[Dict[Optional[str], Any]]:
        """Open one HTTP session per fingerprint in self._tls_chain.

        Pre-opening sessions matters because curl_cffi.AsyncSession setup
        isn't free (it initializes a TLS context that mimics a specific
        browser); we reuse the same session across all pages of the crawl.

        Yields a dict keyed by Optional[str]: the same key used in
        self._tls_chain. None -> httpx.AsyncClient; "chromeXX"/"safariXX"
        -> curl_cffi.requests.AsyncSession. Cleanup is delegated to
        AsyncExitStack so exception context propagates correctly.

        Args:
            httpx_headers: Header dict for the httpx client. Only used when
                self._tls_chain contains None. curl_cffi sessions get no
                explicit headers -- their impersonate spec sets them.
        """
        async with AsyncExitStack() as stack:
            sessions: Dict[Optional[str], Any] = {}
            cffi_session_cls: Any = None
            for fp in self._tls_chain:
                if fp is None:
                    sess: Any = httpx.AsyncClient(
                        headers=httpx_headers or {},
                        timeout=self.config.timeout,
                    )
                else:
                    if cffi_session_cls is None:
                        # Imported lazily so environments using the default
                        # httpx path do not need the optional waf-crawl extra.
                        from curl_cffi.requests import AsyncSession as CffiSession

                        cffi_session_cls = CffiSession
                    sess = cffi_session_cls(
                        impersonate=fp,
                        timeout=self.config.timeout,
                    )
                sessions[fp] = await stack.enter_async_context(sess)
            yield sessions

    async def _fetch_with_fallback(
        self,
        sessions: Dict[Optional[str], Any],
        url: str,
    ) -> Tuple[Optional[Any], Optional[str]]:
        """Fetch a URL by trying each TLS fingerprint in self._tls_chain.

        Behavior on each response status:

          * 2xx + body looks like real content
              -> return (response, fp_used).
          * 2xx + body looks like a JS challenge wrapper
              (Cloudflare "Just a moment...", etc.)
              -> treat as a WAF block, advance to the next fingerprint.
          * Status in _WAF_RETRY_STATUSES (403, 429, 503, 520-526)
              -> advance to next fingerprint.
          * Any other non-2xx (404, 401, 500, ...)
              -> return immediately. The error is content-side and
                 won't change with a different TLS fingerprint;
                 retrying just wastes the rest of the chain.

        Returns:
            (response, fingerprint_used) on first real 2xx success.
            (last_response, None) if the chain was exhausted or returned
                fast on a non-WAF non-2xx -- fingerprint_used=None signals
                "this response is a failure, not a success".
            (None, None) only if every attempt raised an exception.
        """
        last_response = None
        last_status: Optional[int] = None
        exception_log: List[str] = []

        for i, fp in enumerate(self._tls_chain):
            sess = sessions[fp]
            fp_label = fp or "httpx"
            try:
                # curl_cffi uses 'allow_redirects', httpx uses 'follow_redirects'.
                # Custom headers are NOT passed: httpx's are baked into the
                # session, curl_cffi's are owned by the impersonate spec.
                if fp is None:
                    response = await sess.get(url, follow_redirects=True)
                else:
                    response = await sess.get(url, allow_redirects=True)

                if 200 <= response.status_code < 300:
                    body_preview = response.text or ""
                    if _looks_like_challenge_page(body_preview):
                        last_response = response
                        last_status = response.status_code
                        logger.debug(
                            "TLS fp=%s got 200 challenge page for %s, trying next",
                            fp_label,
                            url,
                        )
                        continue
                    if i > 0:
                        logger.info(
                            "TLS fallback hit: url=%s fp=%s pos=%d/%d",
                            url,
                            fp_label,
                            i + 1,
                            len(self._tls_chain),
                        )
                    return response, fp_label

                if response.status_code in _WAF_RETRY_STATUSES:
                    last_response = response
                    last_status = response.status_code
                    logger.debug(
                        "TLS fp=%s returned %d (WAF-like) for %s, trying next",
                        fp_label,
                        response.status_code,
                        url,
                    )
                    continue

                # Non-WAF non-2xx (e.g. 404, 500): fail fast. Trying more
                # TLS fingerprints can't recover content-side errors.
                logger.debug(
                    "TLS fp=%s returned %d for %s (not WAF-like), failing fast",
                    fp_label,
                    response.status_code,
                    url,
                )
                return response, None
            except Exception as e:
                exception_log.append(f"{fp_label}:{type(e).__name__}: {e}")
                logger.debug(
                    "TLS fp=%s raised %s for %s, trying next",
                    fp_label,
                    type(e).__name__,
                    url,
                )

        if last_response is not None:
            logger.warning(
                "TLS fallback exhausted: url=%s last_status=%s chain=%s",
                url,
                last_status,
                [f or "httpx" for f in self._tls_chain],
            )
            return last_response, None

        if exception_log:
            logger.warning(
                "All TLS fingerprints failed for url=%s exceptions=%s",
                url,
                exception_log,
            )

        return None, None

    async def _crawl_loop(self, sessions: Dict[Optional[str], Any]) -> None:
        """Main crawl loop with concurrency control.

        Args:
            sessions: Dict mapping fingerprint -> open HTTP session.
                Keyed by Optional[str] same as self._tls_chain entries.
        """
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.concurrent_requests)

        # Process URLs until we reach max_pages or no more pending URLs
        while self.pending_urls and len(self.visited_urls) < self.config.max_pages:
            # Batch processing for concurrency
            tasks = []
            batch_size = min(
                self.config.concurrent_requests,
                len(self.pending_urls),
                self.config.max_pages - len(self.visited_urls),
            )

            for _ in range(batch_size):
                if not self.pending_urls:
                    break

                url, depth = self.pending_urls.popleft()

                # Skip if already visited
                if url in self.visited_urls:
                    continue

                # Skip if exceeds max depth
                if depth > self.config.max_depth:
                    logger.debug(
                        "Skipping %s: exceeds max depth %s", url, self.config.max_depth
                    )
                    continue

                self.visited_urls.add(url)

                # Create crawl task
                task = self._crawl_page(sessions, url, depth, semaphore)
                tasks.append(task)

            # Execute batch
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results and extract new links
                for result_tuple in results:
                    if isinstance(result_tuple, Exception):
                        logger.error("Crawl task failed: %s", result_tuple)
                        continue

                    # result_tuple should be a tuple (CrawlResult, Set[str])
                    if not isinstance(result_tuple, tuple) or len(result_tuple) != 2:
                        logger.error("Invalid result format: %s", result_tuple)
                        continue

                    result, links = result_tuple
                    if result and result.status == "success":
                        # Queue extracted links
                        await self._process_links(links, result.depth)

                # Rate limiting
                if self.pending_urls:
                    await asyncio.sleep(self.config.request_delay)

                # Progress callback
                if self.progress_callback:
                    self.progress_callback(
                        f"Crawled {len(self.visited_urls)} pages",
                        len(self.visited_urls),
                        self.config.max_pages,
                    )

    async def _crawl_page(
        self,
        sessions: Dict[Optional[str], Any],
        url: str,
        depth: int,
        semaphore: asyncio.Semaphore,
    ) -> tuple[Optional[CrawlResult], Set[str]]:
        """Crawl a single page.

        Args:
            sessions: Dict mapping fingerprint -> open HTTP session.
            url: URL to crawl
            depth: Current depth
            semaphore: Concurrency control semaphore

        Returns:
            Tuple of (CrawlResult or None, Set of extracted links)
        """
        async with semaphore:
            try:
                logger.debug("Crawling %s (depth: %s)", url, depth)

                # Fetch page — Playwright (JS rendered) or TLS fallback chain
                if self.config.render_js:
                    try:
                        html, _meta = await self._fetch_html_rendered(url)
                    except RuntimeError as e:
                        error_msg = str(e)
                        logger.error("Failed to crawl %s: %s", url, error_msg)
                        self.failed_urls[url] = error_msg
                        return None, set()

                    raw_status = _meta.get("status_code")
                    if raw_status is not None and isinstance(raw_status, int):
                        if not (200 <= raw_status < 300):
                            error_msg = f"HTTP {raw_status}"
                            logger.error("Failed to crawl %s: %s", url, error_msg)
                            self.failed_urls[url] = error_msg
                            return None, set()
                else:
                    response, fingerprint_used = await self._fetch_with_fallback(
                        sessions, url
                    )
                    if response is None:
                        error_msg = "All TLS fingerprints raised exceptions"
                        logger.error("Failed to crawl %s: %s", url, error_msg)
                        self.failed_urls[url] = error_msg
                        return None, set()

                    if fingerprint_used is None and 200 <= response.status_code < 300:
                        error_msg = "TLS fallback exhausted with challenge page"
                        logger.error("Failed to crawl %s: %s", url, error_msg)
                        self.failed_urls[url] = error_msg
                        return None, set()

                    if not 200 <= response.status_code < 300:
                        error_msg = f"HTTP {response.status_code}"
                        logger.error("Failed to crawl %s: %s", url, error_msg)
                        self.failed_urls[url] = error_msg
                        return None, set()

                    html = response.text

                # Clean and convert content
                cleaned = self.content_cleaner.clean_and_convert(html, url)

                # Validate content
                content = cleaned["content_markdown"]
                if not self.content_cleaner.is_valid_content(content, min_length=10):
                    logger.warning(
                        "Insufficient content at %s (content_length=%s, title=%s)",
                        url,
                        cleaned.get("content_length"),
                        cleaned.get("title"),
                        extra={"trace_id": self.trace_id},
                    )
                    if logger.isEnabledFor(logging.DEBUG):
                        snippet = (content or "").replace("\n", "\\n")
                        if len(snippet) > 400:
                            snippet = snippet[:400] + "...(truncated)"
                        logger.debug(
                            "Insufficient content snippet at %s (snippet=%s)",
                            url,
                            snippet,
                            extra={"trace_id": self.trace_id},
                        )
                    self.failed_urls[url] = "Insufficient content"
                    return None, set()

                # Extract links
                links = self.link_extractor.extract_links(html, url)

                # Filter links. Use the *effective* UA we send on the wire,
                # not raw config.user_agent, so robots.txt decisions are
                # consistent with what the WAF actually sees.
                valid_links = set()
                for link in links:
                    if self.url_filter.should_crawl(link, self._policy_user_agent):
                        valid_links.add(link)

                self.total_urls_found += len(links)

                # Create result
                result = CrawlResult(
                    url=url,
                    title=cleaned["title"],
                    content_markdown=cleaned["content_markdown"],
                    status="success",
                    depth=depth,
                    timestamp=datetime.now(timezone.utc),
                    content_length=cleaned["content_length"],
                    links_found=len(valid_links),
                )

                self.crawl_results.append(result)
                logger.info(
                    "Successfully crawled %s (%s chars, %s valid links)",
                    url,
                    cleaned["content_length"],
                    len(valid_links),
                )

                return result, valid_links

            except Exception as e:
                # Network/protocol errors are already swallowed inside
                # _fetch_with_fallback; this catches errors from text decode,
                # content cleaning, link extraction etc.
                error_msg = f"Unexpected error: {str(e)}"
                logger.error("Failed to crawl %s: %s", url, error_msg)
                self.failed_urls[url] = error_msg
                return None, set()

    async def _process_links(self, links: Set[str], current_depth: int) -> None:
        """Process and queue links from a crawled page.

        Args:
            links: Set of extracted links
            current_depth: Depth of the current page
        """
        next_depth = current_depth + 1

        # Skip if we've reached max depth
        if next_depth > self.config.max_depth:
            return

        # Add filtered links to pending queue
        pending_urls_set = {u for u, _ in self.pending_urls}
        for link in links:
            # Only add if not visited and not already pending
            if link not in self.visited_urls and link not in pending_urls_set:
                self.pending_urls.append((link, next_depth))

    async def _fetch_html_rendered(self, url: str) -> Tuple[str, Dict[str, object]]:
        # Lazy import to keep Playwright optional at runtime.
        try:
            from playwright.async_api import Error as PlaywrightError
            from playwright.async_api import TimeoutError as PlaywrightTimeoutError
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Playwright is not available. Install with: pip install playwright "
                "and then download browsers: playwright install chromium"
            ) from exc

        if self._browser_context is None:
            raise RuntimeError("Playwright browser context is not initialized")

        context = self._browser_context
        page = await context.new_page()
        try:
            resp = await page.goto(
                url,
                wait_until=self.config.render_wait_until,
                timeout=self.config.render_timeout_ms,
            )
            status_code: object = None
            if resp is not None:
                status_attr = getattr(resp, "status", None)
                # Playwright Response.status is an int property (not a method).
                status_code = status_attr() if callable(status_attr) else status_attr
            content_type = None
            try:
                headers = getattr(resp, "headers", None)
                if isinstance(headers, dict):
                    content_type = headers.get("content-type")
            except Exception:  # noqa: BLE001
                content_type = None

            html = await page.content()
            self._log_fetched_page(
                requested_url=url,
                final_url=page.url,
                status_code=status_code,
                content_type=content_type,
                html=html or "",
                mode="rendered",
            )
            return html or "", {
                "mode": "rendered",
                "final_url": getattr(page, "url", url),
                "status_code": status_code,
            }
        except (PlaywrightTimeoutError, PlaywrightError) as exc:
            raise RuntimeError(f"Playwright navigation failed: {exc}") from exc
        finally:
            await page.close()

    def _log_fetched_page(
        self,
        *,
        requested_url: str,
        final_url: str,
        status_code: object,
        content_type: object,
        html: str,
        mode: str,
    ) -> None:
        logger.info(
            "Fetched page [%s] (requested_url=%s, final_url=%s, status_code=%s, content_type=%s, html_length=%s)",
            mode,
            requested_url,
            final_url,
            status_code,
            content_type,
            len(html or ""),
            extra={"trace_id": self.trace_id},
        )
        if logger.isEnabledFor(logging.DEBUG):
            html_snippet = (html or "").replace("\n", "\\n")
            if len(html_snippet) > 400:
                html_snippet = html_snippet[:400] + "...(truncated)"
            logger.debug(
                "Fetched page snippet [%s] (requested_url=%s, html_snippet=%s)",
                mode,
                requested_url,
                html_snippet,
                extra={"trace_id": self.trace_id},
            )

    async def _start_playwright(self, *, user_agent: str) -> None:
        try:
            from playwright.async_api import Error as PlaywrightError
            from playwright.async_api import async_playwright
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Playwright is not installed. Install with: pip install playwright. "
                "Then download browsers: playwright install chromium"
            ) from exc

        self._playwright = await async_playwright().start()
        chromium = getattr(self._playwright, "chromium", None)
        if chromium is None:
            await self._playwright.stop()
            self._playwright = None
            raise RuntimeError("Playwright chromium is unavailable")
        try:
            self._browser = await chromium.launch(headless=True)
            self._browser_context = await self._browser.new_context(
                user_agent=user_agent
            )
        except PlaywrightError as exc:
            await self._stop_playwright()
            msg = str(exc)
            if "Executable doesn't exist" in msg or "playwright install" in msg:
                raise RuntimeError(
                    "Playwright browsers are not installed. "
                    "If running locally, run: playwright install chromium. "
                    "If running in Docker, update/rebuild the backend image (or run: docker compose exec backend playwright install chromium)."
                ) from exc
            raise RuntimeError(f"Playwright launch failed: {msg}") from exc

    async def _stop_playwright(self) -> None:
        try:
            if self._browser_context is not None:
                await self._browser_context.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._browser is not None:
                await self._browser.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._playwright is not None:
                await self._playwright.stop()
        except Exception:  # noqa: BLE001
            pass
        self._browser_context = None
        self._browser = None
        self._playwright = None

    def get_statistics(self) -> dict:
        """Get crawl statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_urls_found": self.total_urls_found,
            "visited_urls": len(self.visited_urls),
            "successful_pages": len(self.crawl_results),
            "failed_pages": len(self.failed_urls),
            "pending_urls": len(self.pending_urls),
        }


async def crawl_website(
    config: WebCrawlConfig,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> List[CrawlResult]:
    """Convenience function to crawl a website.

    Args:
        config: Crawl configuration
        progress_callback: Optional progress callback

    Returns:
        List of crawl results
    """
    crawler = WebCrawler(config, progress_callback)
    return await crawler.crawl()
