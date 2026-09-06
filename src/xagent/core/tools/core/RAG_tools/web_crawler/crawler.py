"""Core web crawler implementation."""

import asyncio
import importlib
import importlib.util
import logging
import time
from collections import Counter, deque
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Set, Tuple

import httpx

from ..core.schemas import CrawlResult, WebCrawlConfig
from .content_cleaner import ContentCleaner
from .link_extractor import LinkExtractor
from .url_filter import SITE_REJECTIONS, URLFilter

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

# Heuristic markers for Cloudflare / similar JS challenge wrapper pages.
# Strong markers are specific enough to retry/fail before accepting content.
# Weak markers are ordinary text fragments, so they only help explain failures
# after content extraction has already shown the page is unusable.
_STRONG_CHALLENGE_PAGE_MARKERS: Tuple[str, ...] = (
    "cf-challenge",
    "cf-mitigated",
    "cf-please-wait",
    "/cdn-cgi/challenge-platform/",
)
_WEAK_CHALLENGE_PAGE_MARKERS: Tuple[str, ...] = (
    "just a moment",
    "needs to review the security",
)

# Why the crawl loop ended. NO_LINKS and PAGE_CAP are the crawl doing what it
# was configured to do; NO_ELIGIBLE_LINKS means links existed but the site
# refused them, which is the case worth surfacing to the user. UNKNOWN is what
# get_statistics() reports when the loop did not reach its epilogue, which
# happens when an exception propagates out of it - crawl() has no try/finally.
# No caller handles it today; it exists so that state cannot be mistaken for
# one of the three real answers.
STOPPED_UNKNOWN = "unknown"
STOPPED_NO_LINKS = "no_more_links"
STOPPED_PAGE_CAP = "page_cap_reached"
STOPPED_NO_ELIGIBLE_LINKS = "no_eligible_links"

_MIN_MARKDOWN_TEXT_LENGTH = 10
_SHORT_TEXT_LENGTH = 200
_MIN_REPLACEMENT_CHAR_COUNT = 3
_MIN_CONTROL_CHAR_COUNT = 3
_MIN_UNREADABLE_CHAR_COUNT = 3
_MAX_REPLACEMENT_CHAR_RATIO = 0.01
_MAX_CONTROL_CHAR_RATIO = 0.005
_MIN_READABLE_CHAR_RATIO = 0.85
_QUALITY_OK = "ok"
_QUALITY_EMPTY = "empty"
_QUALITY_TOO_SHORT = "too_short"
_QUALITY_CHALLENGE = "challenge"
_QUALITY_NULL_BYTES = "null_bytes"
_QUALITY_REPLACEMENT_RATIO = "replacement_ratio"
_QUALITY_CONTROL_RATIO = "control_ratio"
_QUALITY_READABLE_RATIO = "readable_ratio"
_RAW_EMPTY_REASON = "raw content is empty"


@dataclass(frozen=True)
class _ChallengeDetection:
    confidence: str
    signals: Tuple[str, ...]


def _detect_challenge_page(body: str) -> _ChallengeDetection:
    """Detect WAF/challenge wrappers without treating generic text as enough."""
    if not body:
        return _ChallengeDetection("none", ())

    lowered = body[:3000].lower()
    signals: List[str] = []
    for marker in _STRONG_CHALLENGE_PAGE_MARKERS:
        if marker in lowered:
            signals.append(marker)

    if "<title>just a moment" in lowered and "checking your browser" in lowered:
        signals.append("cloudflare-title")

    if signals:
        return _ChallengeDetection("strong", tuple(signals))

    weak_signals = [
        marker for marker in _WEAK_CHALLENGE_PAGE_MARKERS if marker in lowered
    ]
    if weak_signals:
        return _ChallengeDetection("weak", tuple(weak_signals))

    return _ChallengeDetection("none", ())


def _challenge_reason(label: str, detection: _ChallengeDetection) -> str:
    signals = ",".join(detection.signals)
    if signals:
        return f"{label} challenge page detected: signals={signals}"
    return f"{label} challenge page detected"


@dataclass(frozen=True)
class _ContentQualityReport:
    ok: bool
    code: str
    reason: str
    text_length: int
    replacement_ratio: float
    control_ratio: float
    readable_ratio: float
    null_count: int
    replacement_count: int
    control_count: int
    unreadable_count: int


@dataclass(frozen=True)
class _FetchResult:
    response: Optional[Any]
    fingerprint_used: Optional[str]
    cleaned: Optional[Dict[str, Any]] = None
    raw_quality: Optional[_ContentQualityReport] = None
    markdown_quality: Optional[_ContentQualityReport] = None


def _format_ratio(value: float) -> str:
    return f"{value:.3f}"


def _assess_raw_crawl_response(body: str) -> _ContentQualityReport:
    """Detect raw responses that should not enter content cleaning."""
    text_length = len(body or "")
    if text_length == 0:
        return _ContentQualityReport(
            ok=False,
            code=_QUALITY_EMPTY,
            reason=_RAW_EMPTY_REASON,
            text_length=0,
            replacement_ratio=0.0,
            control_ratio=0.0,
            readable_ratio=0.0,
            null_count=0,
            replacement_count=0,
            control_count=0,
            unreadable_count=0,
        )

    challenge_detection = _detect_challenge_page(body)
    if challenge_detection.confidence == "strong":
        return _ContentQualityReport(
            ok=False,
            code=_QUALITY_CHALLENGE,
            reason=_challenge_reason("raw content", challenge_detection),
            text_length=text_length,
            replacement_ratio=0.0,
            control_ratio=0.0,
            readable_ratio=1.0,
            null_count=0,
            replacement_count=0,
            control_count=0,
            unreadable_count=0,
        )

    return _ContentQualityReport(
        ok=True,
        code=_QUALITY_OK,
        reason="",
        text_length=text_length,
        replacement_ratio=0.0,
        control_ratio=0.0,
        readable_ratio=1.0,
        null_count=0,
        replacement_count=0,
        control_count=0,
        unreadable_count=0,
    )


def _assess_text_quality(
    text: str,
    *,
    min_length: int,
    label: str,
) -> _ContentQualityReport:
    """Detect obviously unreadable crawl content before it enters the KB."""
    text_length = len(text or "")
    if text_length == 0:
        return _ContentQualityReport(
            ok=False,
            code=_QUALITY_EMPTY,
            reason=f"{label} is empty",
            text_length=0,
            replacement_ratio=0.0,
            control_ratio=0.0,
            readable_ratio=0.0,
            null_count=0,
            replacement_count=0,
            control_count=0,
            unreadable_count=0,
        )

    null_count = 0
    replacement_count = 0
    control_count = 0
    readable_count = 0
    for ch in text:
        if ch == "\x00":
            null_count += 1
        if ch == "\ufffd":
            replacement_count += 1

        is_allowed_whitespace = ch in "\n\r\t"
        if not is_allowed_whitespace and ord(ch) < 32:
            control_count += 1
        if is_allowed_whitespace or ch.isprintable():
            readable_count += 1

    replacement_ratio = replacement_count / text_length
    control_ratio = control_count / text_length
    readable_ratio = readable_count / text_length
    unreadable_count = text_length - readable_count

    code = _QUALITY_OK
    reason = ""
    if text_length < min_length:
        code = _QUALITY_TOO_SHORT
        reason = f"{label} too short: length={text_length}"
    else:
        challenge_detection = _detect_challenge_page(text)
        if challenge_detection.confidence == "strong":
            code = _QUALITY_CHALLENGE
            reason = _challenge_reason(label, challenge_detection)
        elif null_count:
            code = _QUALITY_NULL_BYTES
            reason = f"{label} contains null bytes: count={null_count}"
        elif text_length < _SHORT_TEXT_LENGTH:
            if replacement_count >= _MIN_REPLACEMENT_CHAR_COUNT:
                code = _QUALITY_REPLACEMENT_RATIO
                reason = (
                    f"{label} unreadable content: replacement_count={replacement_count}"
                )
            elif control_count >= _MIN_CONTROL_CHAR_COUNT:
                code = _QUALITY_CONTROL_RATIO
                reason = f"{label} unreadable content: control_count={control_count}"
            elif unreadable_count >= _MIN_UNREADABLE_CHAR_COUNT:
                code = _QUALITY_READABLE_RATIO
                reason = (
                    f"{label} unreadable content: unreadable_count={unreadable_count}"
                )
        elif (
            replacement_count >= _MIN_REPLACEMENT_CHAR_COUNT
            and replacement_ratio > _MAX_REPLACEMENT_CHAR_RATIO
        ):
            code = _QUALITY_REPLACEMENT_RATIO
            reason = (
                f"{label} unreadable content: "
                f"replacement_ratio={_format_ratio(replacement_ratio)}"
            )
        elif (
            control_count >= _MIN_CONTROL_CHAR_COUNT
            and control_ratio > _MAX_CONTROL_CHAR_RATIO
        ):
            code = _QUALITY_CONTROL_RATIO
            reason = (
                f"{label} unreadable content: "
                f"control_ratio={_format_ratio(control_ratio)}"
            )
        elif (
            unreadable_count >= _MIN_UNREADABLE_CHAR_COUNT
            and readable_ratio < _MIN_READABLE_CHAR_RATIO
        ):
            code = _QUALITY_READABLE_RATIO
            reason = (
                f"{label} unreadable content: "
                f"readable_ratio={_format_ratio(readable_ratio)}"
            )

    return _ContentQualityReport(
        ok=not reason,
        code=code,
        reason=reason,
        text_length=text_length,
        replacement_ratio=replacement_ratio,
        control_ratio=control_ratio,
        readable_ratio=readable_ratio,
        null_count=null_count,
        replacement_count=replacement_count,
        control_count=control_count,
        unreadable_count=unreadable_count,
    )


def _should_retry_cleaned_quality_failure(report: _ContentQualityReport) -> bool:
    """Return whether a cleaned-content failure might recover with another TLS fp."""
    if report.ok:
        return False
    return report.code in {
        _QUALITY_EMPTY,
        _QUALITY_CHALLENGE,
        _QUALITY_NULL_BYTES,
        _QUALITY_REPLACEMENT_RATIO,
        _QUALITY_CONTROL_RATIO,
        _QUALITY_READABLE_RATIO,
    }


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


def _get_httpx_accept_encoding() -> str:
    """Return encodings httpx can actually decode in this environment."""
    encodings = ["gzip", "deflate"]
    if importlib.util.find_spec("brotli") or importlib.util.find_spec("brotlicffi"):
        encodings.append("br")
    return ", ".join(encodings)


def _response_log_metadata(
    response: Any,
    report: Optional[_ContentQualityReport] = None,
) -> Dict[str, Any]:
    """Build safe response metadata for crawl quality logs."""
    headers = getattr(response, "headers", {}) or {}
    header_get = getattr(headers, "get", None)

    def _get_header(name: str) -> Optional[str]:
        if callable(header_get):
            value = header_get(name)
            return value if isinstance(value, str) else None
        return None

    metadata: Dict[str, Any] = {
        "status_code": getattr(response, "status_code", None),
        "content_encoding": _get_header("content-encoding"),
        "content_type": _get_header("content-type"),
    }
    if report is not None:
        metadata.update(
            {
                "quality_code": report.code,
                "text_length": report.text_length,
                "null_count": report.null_count,
                "replacement_count": report.replacement_count,
                "control_count": report.control_count,
                "unreadable_count": report.unreadable_count,
                "replacement_ratio": report.replacement_ratio,
                "control_ratio": report.control_ratio,
                "readable_ratio": report.readable_ratio,
            }
        )
    return metadata


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
    ):
        """Initialize web crawler.

        Args:
            config: Crawl configuration
            progress_callback: Optional callback for progress updates
                Args: (current_url, completed, total)
        """
        self.config = config
        self.progress_callback = progress_callback

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
        # When tls_impersonate is None we send config.user_agent on the
        # wire, so policy must reason about that string. When tls_impersonate
        # is set, curl_cffi controls the UA based on the impersonate spec --
        # config.user_agent is intentionally ignored in that path to keep
        # TLS fingerprint and HTTP UA consistent. For policy we use the
        # mapped UA of the *first* fingerprint in the chain, falling back
        # to "*" if the spec is unknown to us.
        if config.tls_impersonate is None:
            self._policy_user_agent: str = config.user_agent or _DEFAULT_USER_AGENT
        else:
            first_fp = next((fp for fp in self._tls_chain if fp is not None), None)
            self._policy_user_agent = (
                _IMPERSONATE_TO_UA.get(first_fp, "*") if first_fp else "*"
            )

        # Initialize components
        self.url_filter = URLFilter(
            base_url=config.start_url,
            same_domain_only=config.same_domain_only,
            url_patterns=config.url_patterns,
            exclude_patterns=config.exclude_patterns,
            respect_robots_txt=config.respect_robots_txt,
            user_agent=self._policy_user_agent,
        )
        self.content_cleaner = ContentCleaner(
            content_selector=config.content_selector,
            remove_selectors=config.remove_selectors,
        )
        self.link_extractor = LinkExtractor(config.start_url)

        # Crawl state
        self.visited_urls: Set[str] = set()
        self.pending_urls: deque = deque()
        self.crawl_results: List[CrawlResult] = []
        self.failed_urls: Dict[str, str] = {}

        # total_urls_found counts raw extractor output and is diagnostic. Only
        # total_links_queued tracks frontier progress - a link can survive every
        # filter and still queue nothing, having been seen already or sitting
        # past max_depth. It counts queue insertions, which are keyed on the raw
        # link, so /a#x and /a#y each count; deduplicating those would change
        # which pages get fetched, so it does not belong in this PR.
        self.total_urls_found = 0
        self.total_links_queued = 0
        self.rejected_urls: Set[str] = set()
        self.link_rejections: Counter[str] = Counter()
        self.stop_reason: str = STOPPED_UNKNOWN
        self.start_time: Optional[float] = None

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

        logger.info("Starting crawl from %s", self.config.start_url)

        # When tls_impersonate is None we use plain httpx; in that path we
        # need to manually compose a browser-like header set so the request
        # doesn't look obviously bot-shaped. When tls_impersonate is set,
        # curl_cffi owns headers (matching its TLS impersonation), and we
        # MUST NOT inject our own UA/headers -- doing so creates a TLS-vs-
        # HTTP-fingerprint mismatch which is exactly what WAFs catch.
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
                "Accept-Encoding": _get_httpx_accept_encoding(),
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Upgrade-Insecure-Requests": "1",
                "DNT": "1",
            }

        async with self._open_sessions(httpx_headers) as sessions:
            await self._crawl_loop(sessions)

        elapsed = time.time() - self.start_time
        logger.info(
            "Crawl completed: %s pages, %s failed, %.2fs",
            len(self.crawl_results),
            len(self.failed_urls),
            elapsed,
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
    ) -> _FetchResult:
        """Fetch a URL by trying each TLS fingerprint in self._tls_chain.

        Behavior on each response status:

          * 2xx + body looks like real content
              -> return (response, fp_used).
          * 2xx + empty body / JS challenge wrapper
              (Cloudflare "Just a moment...", etc.)
              -> treat as unusable raw content, advance to the next fingerprint.
          * Status in _WAF_RETRY_STATUSES (403, 429, 503, 520-526)
              -> advance to next fingerprint.
          * Any other non-2xx (404, 401, 500, ...)
              -> return immediately. The error is content-side and
                 won't change with a different TLS fingerprint;
                 retrying just wastes the rest of the chain.

        Returns:
            _FetchResult(response, fingerprint_used, ...) on first real 2xx success.
            _FetchResult(last_response, None, ...) if the chain was exhausted or returned
                fast on a non-WAF non-2xx -- fingerprint_used=None signals
                "this response is a failure, not a success".
            _FetchResult(None, None) only if every attempt raised an exception.
        """
        last_result = _FetchResult(response=None, fingerprint_used=None)
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
            except Exception as e:
                exception_log.append(f"{fp_label}:{type(e).__name__}: {e}")
                logger.debug(
                    "TLS fp=%s raised %s for %s, trying next",
                    fp_label,
                    type(e).__name__,
                    url,
                )
                continue

            if 200 <= response.status_code < 300:
                body_preview = response.text or ""
                raw_quality = _assess_raw_crawl_response(body_preview)
                if not raw_quality.ok:
                    last_result = _FetchResult(
                        response=response,
                        fingerprint_used=None,
                        raw_quality=raw_quality,
                    )
                    last_status = response.status_code
                    logger.debug(
                        "TLS fp=%s got 200 unusable raw content for %s "
                        "(%s), trying next",
                        fp_label,
                        url,
                        raw_quality.reason,
                    )
                    continue

                cleaned = self.content_cleaner.clean_and_convert(body_preview, url)
                markdown_quality = _assess_text_quality(
                    cleaned["content_markdown"],
                    min_length=_MIN_MARKDOWN_TEXT_LENGTH,
                    label="extracted content",
                )
                if (
                    self.config.tls_impersonate == "auto"
                    and _should_retry_cleaned_quality_failure(markdown_quality)
                ):
                    last_result = _FetchResult(
                        response=response,
                        fingerprint_used=None,
                        cleaned=cleaned,
                        raw_quality=raw_quality,
                        markdown_quality=markdown_quality,
                    )
                    last_status = response.status_code
                    logger.debug(
                        "TLS fp=%s got 200 unusable extracted content for %s "
                        "(%s), trying next",
                        fp_label,
                        url,
                        markdown_quality.reason,
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
                return _FetchResult(
                    response=response,
                    fingerprint_used=fp_label,
                    cleaned=cleaned,
                    raw_quality=raw_quality,
                    markdown_quality=markdown_quality,
                )

            if response.status_code in _WAF_RETRY_STATUSES:
                last_result = _FetchResult(
                    response=response,
                    fingerprint_used=None,
                )
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
            return _FetchResult(response=response, fingerprint_used=None)

        if last_result.response is not None:
            logger.warning(
                "TLS fallback exhausted: url=%s last_status=%s chain=%s",
                url,
                last_status,
                [f or "httpx" for f in self._tls_chain],
            )
            return last_result

        if exception_log:
            logger.warning(
                "All TLS fingerprints failed for url=%s exceptions=%s",
                url,
                exception_log,
            )

        return _FetchResult(response=None, fingerprint_used=None)

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

        if len(self.visited_urls) >= self.config.max_pages:
            self.stop_reason = STOPPED_PAGE_CAP
        elif self._site_refused_the_frontier():
            self.stop_reason = STOPPED_NO_ELIGIBLE_LINKS
        else:
            self.stop_reason = STOPPED_NO_LINKS

    def _site_refused_the_frontier(self) -> bool:
        """Whether the site's own policy is why there was nothing left to crawl.

        Asks about the frontier, not about rejection counts: a crawl that
        queued a page was making progress whatever else it turned away. Depth
        needs no separate guard - WebCrawlConfig.max_depth is Field(ge=1), so a
        page dropped for depth is always behind one that was queued. A crawl
        that could stop at depth 0 would need one.
        """
        if self.total_links_queued:
            return False
        return any(self.link_rejections[reason] for reason in SITE_REJECTIONS)

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

                # Fetch page (with TLS fingerprint fallback chain)
                fetch_result = await self._fetch_with_fallback(sessions, url)
                response = fetch_result.response
                fingerprint_used = fetch_result.fingerprint_used
                if response is None:
                    error_msg = "All TLS fingerprints raised exceptions"
                    logger.error("Failed to crawl %s: %s", url, error_msg)
                    self.failed_urls[url] = error_msg
                    return None, set()

                if fingerprint_used is None and 200 <= response.status_code < 300:
                    html = response.text or ""
                    raw_quality = (
                        fetch_result.raw_quality or _assess_raw_crawl_response(html)
                    )
                    if raw_quality.code == _QUALITY_CHALLENGE:
                        error_msg = "TLS fallback exhausted with challenge page"
                    elif (
                        fetch_result.markdown_quality
                        and not fetch_result.markdown_quality.ok
                    ):
                        error_msg = fetch_result.markdown_quality.reason
                    else:
                        error_msg = raw_quality.reason or "TLS fallback exhausted"
                    quality_report = raw_quality
                    if (
                        fetch_result.markdown_quality
                        and not fetch_result.markdown_quality.ok
                    ):
                        quality_report = fetch_result.markdown_quality
                    logger.error(
                        "Failed to crawl %s: %s",
                        url,
                        error_msg,
                        extra={
                            "crawl_url": url,
                            **_response_log_metadata(response, quality_report),
                        },
                    )
                    self.failed_urls[url] = error_msg
                    return None, set()

                # Explicit status check is library-agnostic (works for both
                # httpx and curl_cffi response objects, which raise different
                # exception types from raise_for_status()).
                if not 200 <= response.status_code < 300:
                    error_msg = f"HTTP {response.status_code}"
                    logger.error("Failed to crawl %s: %s", url, error_msg)
                    self.failed_urls[url] = error_msg
                    return None, set()

                html = response.text
                raw_quality = fetch_result.raw_quality or _assess_raw_crawl_response(
                    html
                )
                if not raw_quality.ok:
                    logger.warning(
                        "Rejected crawl content at %s: %s",
                        url,
                        raw_quality.reason,
                        extra={
                            "crawl_url": url,
                            **_response_log_metadata(response, raw_quality),
                        },
                    )
                    self.failed_urls[url] = raw_quality.reason
                    return None, set()

                # Clean and convert content
                cleaned = (
                    fetch_result.cleaned
                    or self.content_cleaner.clean_and_convert(html, url)
                )

                # Validate content
                content = cleaned["content_markdown"]
                markdown_quality = (
                    fetch_result.markdown_quality
                    or _assess_text_quality(
                        content,
                        min_length=_MIN_MARKDOWN_TEXT_LENGTH,
                        label="extracted content",
                    )
                )
                if not markdown_quality.ok:
                    logger.warning(
                        "Rejected extracted crawl content at %s: %s",
                        url,
                        markdown_quality.reason,
                        extra={
                            "crawl_url": url,
                            **_response_log_metadata(response, markdown_quality),
                        },
                    )
                    self.failed_urls[url] = markdown_quality.reason
                    return None, set()

                if not self.content_cleaner.is_valid_content(content, min_length=10):
                    logger.warning("Insufficient content at %s", url)
                    self.failed_urls[url] = "Insufficient content"
                    return None, set()

                # Extract links
                links = self.link_extractor.extract_links(html, url)

                # Filter links. Use the *effective* UA we send on the wire,
                # not raw config.user_agent, so robots.txt decisions are
                # consistent with what the WAF actually sees.
                valid_links = set()
                for link in links:
                    reason = self.url_filter.rejection_reason(
                        link, self._policy_user_agent
                    )
                    if reason is None:
                        valid_links.add(link)
                    else:
                        # Keyed on the normalized URL because that is what
                        # rejection_reason judged: /a#x, /a#y and //EXAMPLE.com/a
                        # are one refused link, as is a nav link on every page.
                        seen = self.url_filter.normalize_url(link) or link
                        if seen not in self.rejected_urls:
                            self.rejected_urls.add(seen)
                            self.link_rejections[reason] += 1

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
                self.total_links_queued += 1

    def get_statistics(self) -> dict:
        """Get crawl statistics.

        Returns:
            Dictionary with statistics. Beyond the page counts:
            total_urls_found (raw extractor output, before filtering),
            total_links_queued (links that actually entered the frontier),
            link_rejections (count per rejection reason, per distinct URL),
            and stop_reason (one of the STOPPED_* values). The page cap is
            checked first, so with max_pages=1 a crawl reports
            page_cap_reached even if the site also refused every link.
        """
        return {
            "total_urls_found": self.total_urls_found,
            "total_links_queued": self.total_links_queued,
            "link_rejections": dict(self.link_rejections),
            "stop_reason": self.stop_reason,
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
