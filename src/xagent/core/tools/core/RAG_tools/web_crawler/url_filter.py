"""URL filtering for web crawler.

Provides URL validation, filtering, and normalization functionality.
"""

import logging
import re
from typing import Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from ..core.web_url_utils import normalize_web_url

logger = logging.getLogger(__name__)


class URLFilter:
    """URL filtering and validation.

    Handles URL filtering based on domain, regex patterns, and robots.txt rules.
    """

    def __init__(
        self,
        base_url: str,
        *,
        same_domain_only: bool = True,
        url_patterns: Optional[list[str]] = None,
        exclude_patterns: Optional[list[str]] = None,
        respect_robots_txt: bool = True,
        user_agent: Optional[str] = None,
    ):
        """Initialize URL filter.

        Args:
            base_url: Base URL for crawling
            same_domain_only: Only allow URLs from same domain
            url_patterns: Regex patterns for allowed URLs
            exclude_patterns: Regex patterns for excluded URLs
            respect_robots_txt: Whether to check robots.txt
            user_agent: UA to send when fetching robots.txt. Must be the one
                the crawl itself sends: a WAF that 403s an unrecognised UA
                would otherwise be read as "this site has no rules".
        """
        normalized_base_url = normalize_web_url(base_url)
        self.base_domain = urlparse(normalized_base_url or base_url).netloc.lower()
        self.same_domain_only = same_domain_only
        self.url_patterns = [re.compile(p) for p in (url_patterns or [])]
        self.exclude_patterns = [re.compile(p) for p in (exclude_patterns or [])]
        self.respect_robots_txt = respect_robots_txt
        self.user_agent = user_agent

        # Initialize robots.txt parser
        self.robots_parser: Optional[RobotFileParser] = None
        self.robots_url: Optional[str] = None
        if respect_robots_txt:
            try:
                self.robots_url = f"{urlparse(normalized_base_url or base_url).scheme}://{self.base_domain}/robots.txt"
                self.robots_parser = RobotFileParser()
                self.robots_parser.set_url(self.robots_url)
                self._fetch_robots_txt()
            except Exception as e:
                logger.warning("Failed to fetch robots.txt: %s", e)
                self.robots_parser = None

    def _fetch_robots_txt(self) -> None:
        """Fetch robots.txt and apply the three-way RFC 9309 s2.3 policy: parse
        it on a 2xx, drop the parser on a 4xx (unavailable means unrestricted),
        and leave it unread on a 5xx or transport error (undefined means
        complete disallow)."""
        import httpx

        if not self.robots_parser or not self.robots_url:
            return

        try:
            # follow_redirects is required, not incidental: httpx defaults to
            # False, so a 3xx would reach the final branch below and deny the
            # whole site, with the Disallow rules left unread at the redirect
            # target. Client rather than httpx.get because only Client takes
            # max_redirects, and timeout is per-request: 20 default hops
            # stretch the ceiling to ~200s. Five is the floor RFC 9309
            # s2.3.1.2 asks crawlers to follow.
            headers = {"User-Agent": self.user_agent} if self.user_agent else None
            with httpx.Client(
                timeout=10, follow_redirects=True, max_redirects=5
            ) as client:
                response = client.get(self.robots_url, headers=headers)
            if 200 <= response.status_code < 300:
                self.robots_parser.parse(response.text.splitlines())
                logger.info("Loaded robots.txt from %s", self.robots_url)
            elif 400 <= response.status_code < 500 and response.status_code != 429:
                # RFC 9309 s2.3.1.3: unavailable means the crawler may access
                # anything. Dropping the parser is what expresses that - keeping
                # an unread RobotFileParser makes can_fetch() deny every URL,
                # which silently reduces a site with no robots.txt to its start
                # page alone. 5xx and transport errors deliberately fall through
                # to that deny-all state instead: s2.3.1.4 makes an undefined
                # robots.txt a complete disallow. s2.3.1.3 judges by meaning
                # ("the resource is unavailable") and only cites 400-499 as an
                # example, so 429 is excluded: a server asking us to slow down
                # has not said it has no rules.
                self.robots_parser = None
                logger.info(
                    "No robots.txt restrictions from %s (HTTP %s)",
                    self.robots_url,
                    response.status_code,
                )
            else:
                logger.warning(
                    "robots.txt undefined at %s (HTTP %s); only the start URL "
                    "will be fetched, discovered links will be skipped",
                    self.robots_url,
                    response.status_code,
                )
        except Exception as e:
            # Deliberately broad: narrowing to transport errors would let a bug
            # in here escape to __init__, which clears the parser and so fails
            # open on the whole site instead of closed.
            logger.warning("Could not fetch robots.txt from %s: %s", self.robots_url, e)

    def is_allowed(self, url: str, user_agent: str = "*") -> bool:
        """Check if URL is allowed by robots.txt.

        A parser of None is the upstream decision that the site has no rules
        (see _fetch_robots_txt), not a defensive fallback: returning True there
        is what makes a site without robots.txt crawlable.

        Args:
            url: URL to check
            user_agent: User agent string (default: "*")

        Returns:
            True if allowed, False otherwise
        """
        if not self.robots_parser:
            return True

        try:
            return self.robots_parser.can_fetch(user_agent, url)
        except Exception as e:
            logger.warning("Error checking robots.txt for %s: %s", url, e)
            return True

    def is_same_domain(self, url: str) -> bool:
        """Check if URL is from the same domain as base URL.

        Args:
            url: URL to check

        Returns:
            True if same domain, False otherwise
        """
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower() == self.base_domain
        except Exception as e:
            logger.debug("Error checking domain for %s: %s", url, e)
            return False

    def matches_patterns(self, url: str) -> bool:
        """Check if URL matches any of the allowed patterns.

        If no patterns are configured, all URLs are considered matching.

        Args:
            url: URL to check

        Returns:
            True if matches (or no patterns configured), False otherwise
        """
        if not self.url_patterns:
            return True

        return any(pattern.search(url) for pattern in self.url_patterns)

    def is_excluded(self, url: str) -> bool:
        """Check if URL matches any exclusion pattern.

        Args:
            url: URL to check

        Returns:
            True if excluded, False otherwise
        """
        return any(pattern.search(url) for pattern in self.exclude_patterns)

    def should_crawl(self, url: str, user_agent: str = "*") -> bool:
        """Check if URL should be crawled based on all rules.

        Args:
            url: URL to check
            user_agent: User agent string for robots.txt check

        Returns:
            True if URL should be crawled, False otherwise
        """
        # Normalize URL
        normalized = self.normalize_url(url)
        if not normalized:
            return False

        # Check domain
        if self.same_domain_only and not self.is_same_domain(normalized):
            logger.debug("Skipping %s: different domain", normalized)
            return False

        # Check robots.txt
        if self.respect_robots_txt and not self.is_allowed(normalized, user_agent):
            logger.debug("Skipping %s: disallowed by robots.txt", normalized)
            return False

        # Check exclusion patterns
        if self.is_excluded(normalized):
            logger.debug("Skipping %s: matches exclusion pattern", normalized)
            return False

        # Check inclusion patterns
        if not self.matches_patterns(normalized):
            logger.debug("Skipping %s: does not match inclusion pattern", normalized)
            return False

        return True

    def normalize_url(self, url: str, base_url: Optional[str] = None) -> Optional[str]:
        """Normalize URL by handling relative URLs and removing fragments.

        Args:
            url: URL to normalize
            base_url: Base URL for resolving relative URLs (defaults to start_url)

        Returns:
            Normalized absolute URL, or None if invalid
        """
        normalized = normalize_web_url(url, base_url=base_url)
        if normalized is None:
            logger.debug("Failed to normalize URL %s", url)
        return normalized
