"""Unit tests for URL filter."""

from functools import partial
from unittest.mock import MagicMock

import httpx
import pytest
from pydantic import ValidationError

from xagent.core.tools.core.RAG_tools.core.schemas import WebCrawlConfig
from xagent.core.tools.core.RAG_tools.web_crawler.url_filter import URLFilter


class TestURLFilter:
    """Test URL filtering functionality."""

    def test_same_domain_check(self):
        """Test same domain validation."""
        filter = URLFilter("https://example.com")

        assert filter.is_same_domain("https://example.com/page1") is True
        assert filter.is_same_domain("https://example.com/page1/sub") is True
        assert filter.is_same_domain("https://other.com/page") is False
        assert filter.is_same_domain("http://example.com/page") is True

    def test_normalize_url(self):
        """Test URL normalization."""
        filter = URLFilter("https://example.com")

        # Absolute URL
        assert (
            filter.normalize_url("https://example.com/test")
            == "https://example.com/test"
        )

        # Relative URL
        assert (
            filter.normalize_url("/page", "https://example.com/")
            == "https://example.com/page"
        )

        # Remove fragment
        assert (
            filter.normalize_url("https://example.com/test#section")
            == "https://example.com/test"
        )

        # Invalid scheme
        assert filter.normalize_url("ftp://example.com/file") is None

    def test_normalize_url_accepts_uppercase_scheme(self):
        """Normalization should be case-insensitive for HTTP(S) schemes."""
        filter = URLFilter("https://example.com")

        assert (
            filter.normalize_url("HTTP://Example.com/Test#section")
            == "http://example.com/Test"
        )

    def test_normalize_url_rejects_hostless_http_urls(self):
        """Hostless HTTP(S) inputs should fail before crawl-time errors."""
        filter = URLFilter("https://example.com")

        assert filter.normalize_url("http://@") is None
        assert filter.normalize_url("http://:80") is None

    def test_should_crawl_same_domain_only(self):
        """Test crawling with same domain restriction."""
        filter = URLFilter(
            "https://example.com", same_domain_only=True, respect_robots_txt=False
        )

        assert filter.should_crawl("https://example.com/page1") is True
        assert filter.should_crawl("https://other.com/page") is False

    def test_should_crawl_cross_domain(self):
        """Test crawling without same domain restriction."""
        filter = URLFilter(
            "https://example.com", same_domain_only=False, respect_robots_txt=False
        )

        assert filter.should_crawl("https://example.com/page1") is True
        assert filter.should_crawl("https://other.com/page") is True

    def test_url_patterns(self):
        """Test URL pattern matching."""
        filter = URLFilter(
            "https://example.com",
            url_patterns=[r"https://example.com/docs/.*"],
            respect_robots_txt=False,
        )

        assert filter.should_crawl("https://example.com/docs/page1") is True
        assert filter.should_crawl("https://example.com/blog/post1") is False

    def test_exclude_patterns(self):
        """Test URL exclusion patterns."""
        filter = URLFilter(
            "https://example.com",
            exclude_patterns=[r".*\.pdf$", r".*\.jpg$"],
            respect_robots_txt=False,
        )

        assert filter.should_crawl("https://example.com/page1") is True
        assert filter.should_crawl("https://example.com/doc.pdf") is False
        assert filter.should_crawl("https://example.com/image.jpg") is False

    def test_combined_patterns(self):
        """Test combined include and exclude patterns."""
        filter = URLFilter(
            "https://example.com",
            url_patterns=[r"https://example.com/.*"],
            exclude_patterns=[r".*/admin/.*", r".*\.pdf$"],
            respect_robots_txt=False,
        )

        assert filter.should_crawl("https://example.com/page1") is True
        assert filter.should_crawl("https://example.com/admin/settings") is False
        assert filter.should_crawl("https://example.com/doc.pdf") is False

    def test_invalid_protocols(self):
        """Test filtering of invalid protocols."""
        filter = URLFilter("https://example.com")

        # JavaScript, mailto, tel should be filtered out during normalization
        assert filter.normalize_url("javascript:void(0)") is None
        assert filter.normalize_url("mailto:test@example.com") is None
        assert filter.normalize_url("tel:+1234567890") is None

    def test_url_with_query_and_fragment(self):
        """Test URLs with query parameters and fragments."""
        filter = URLFilter("https://example.com")

        normalized = filter.normalize_url(
            "https://example.com/page?param=value#section"
        )
        assert normalized == "https://example.com/page?param=value"

    def test_empty_and_none_patterns(self):
        """Test with no patterns configured (allow all)."""
        filter = URLFilter(
            "https://example.com", url_patterns=None, respect_robots_txt=False
        )

        # Should allow all same-domain URLs when no patterns specified
        assert filter.should_crawl("https://example.com/any-page") is True
        assert filter.should_crawl("https://example.com/admin") is True


def test_web_crawl_config_normalizes_start_url():
    """WebCrawlConfig should normalize its shared crawl entrypoint."""
    config = WebCrawlConfig(start_url=" HTTP://Example.com/docs#intro ")

    assert config.start_url == "http://example.com/docs"


def test_web_crawl_config_preserves_ipv6_literal_brackets():
    """WebCrawlConfig should keep IPv6 literals bracketed when normalizing."""
    config = WebCrawlConfig(start_url="http://[::1]:8000/docs#frag")

    assert config.start_url == "http://[::1]:8000/docs"


def test_web_crawl_config_rejects_invalid_start_urls():
    """WebCrawlConfig should enforce the shared web URL validation boundary."""
    with pytest.raises(
        ValidationError,
        match="Invalid start_url: URL must start with http:// or https://",
    ):
        WebCrawlConfig(start_url="www.example.com")

    with pytest.raises(
        ValidationError,
        match="Invalid start_url: URL must include a hostname",
    ):
        WebCrawlConfig(start_url="http://@")


class _FakeResponse:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class TestRobotsTxtAvailability:
    """A missing robots.txt must not lock the crawler out of the whole site.

    RobotFileParser denies everything until it has been read, so any branch
    that leaves the parser untouched silently reduces a crawl to its start
    page: on a real documentation site this turned 42 reachable pages into 1.
    """

    def _filter_with_robots_response(
        self, monkeypatch, response=None, *, error=None
    ) -> URLFilter:
        self.client = MagicMock()
        self.get = self.client.return_value.__enter__.return_value.get
        self.get.side_effect = error or (lambda url, headers=None: response)
        monkeypatch.setattr(httpx, "Client", self.client)
        return URLFilter("https://example.com", respect_robots_txt=True)

    def test_robots_is_fetched_with_the_crawl_user_agent(self, monkeypatch):
        """A WAF that 403s an unrecognised UA would otherwise be read as "no
        rules" while the crawl itself proceeds under a browser UA."""
        self.client = MagicMock()
        self.get = self.client.return_value.__enter__.return_value.get
        self.get.side_effect = lambda url, headers=None: _FakeResponse(404)
        monkeypatch.setattr(httpx, "Client", self.client)
        URLFilter("https://example.com", user_agent="XagentBot/1.0")

        assert self.get.call_args.kwargs["headers"] == {"User-Agent": "XagentBot/1.0"}

    @pytest.mark.parametrize("status", [404, 403, 410])
    def test_4xx_robots_allows_every_path(self, monkeypatch, status):
        f = self._filter_with_robots_response(
            monkeypatch, _FakeResponse(status, "<html>Page Not Found</html>")
        )
        assert f.is_allowed("https://example.com/docs/intro") is True
        assert f.should_crawl("https://example.com/docs/intro") is True

    @pytest.mark.parametrize("status", [200, 204, 206])
    def test_any_2xx_robots_is_parsed(self, monkeypatch, status):
        """A CDN answering /robots.txt with 204 No Content has said the site
        has no rules; an exact match on 200 would drop it into the deny-all
        branch and invert that."""
        f = self._filter_with_robots_response(monkeypatch, _FakeResponse(status))
        assert f.should_crawl("https://example.com/docs/intro") is True

    @pytest.mark.parametrize("status", [500, 503])
    def test_5xx_robots_blocks_the_crawl(self, monkeypatch, status):
        """RFC 9309 s2.3.1.4: an unreachable robots.txt is undefined, and an
        undefined robots.txt is a complete disallow. Unlike 4xx, a server error
        says nothing about whether rules exist."""
        f = self._filter_with_robots_response(monkeypatch, _FakeResponse(status))
        assert f.should_crawl("https://example.com/docs/intro") is False

    @pytest.mark.parametrize(
        "error",
        [
            OSError("connection refused"),
            httpx.ConnectTimeout("timed out"),
            httpx.ConnectError("name resolution failed"),
            httpx.TooManyRedirects("redirect loop"),
        ],
    )
    def test_unreachable_robots_blocks_the_crawl(self, monkeypatch, error):
        """TooManyRedirects is included on purpose: s2.3.1.2 permits treating a
        too-long chain as unavailable (allow), and this takes the stricter of
        the two readings."""
        f = self._filter_with_robots_response(monkeypatch, error=error)
        assert f.should_crawl("https://example.com/docs/intro") is False

    def test_429_blocks_the_crawl(self, monkeypatch):
        """s2.3.1.3 judges by meaning, not by status class: a server asking us
        to slow down has not said it has no rules."""
        f = self._filter_with_robots_response(monkeypatch, _FakeResponse(429))
        assert f.should_crawl("https://example.com/docs/intro") is False

    def test_fetch_is_bounded_in_time(self, monkeypatch):
        """timeout is per-request, so it and the redirect limit together are
        what bound the total wait."""
        self._filter_with_robots_response(monkeypatch, _FakeResponse(404))

        assert self.client.call_args.kwargs["timeout"] == 10
        assert self.client.call_args.kwargs["max_redirects"] == 5

    def test_endless_redirects_block_the_crawl(self, monkeypatch):
        """The kwarg assertion above cannot show the sixth hop is refused.
        s2.3.1.2 permits treating an over-long chain as unavailable (allow);
        this takes the stricter reading, so it needs a behavioural anchor."""
        self._patch_client_with_handler(
            monkeypatch,
            lambda request: httpx.Response(
                301, headers={"Location": f"{request.url}/again"}
            ),
        )
        f = URLFilter("https://example.com", respect_robots_txt=True)

        assert f.should_crawl("https://example.com/docs/intro") is False

    def _patch_client_with_handler(self, monkeypatch, handler):
        monkeypatch.setattr(
            httpx,
            "Client",
            partial(httpx.Client, transport=httpx.MockTransport(handler)),
        )

    def test_disallow_rules_behind_a_redirect_are_honoured(self, monkeypatch):
        """http->https and www canonicalisation redirect /robots.txt routinely.
        httpx defaults follow_redirects to False, which would leave the rules
        unread at the redirect target and deny the whole site."""

        def handler(request):
            if request.url.host == "example.com":
                return httpx.Response(
                    301, headers={"Location": "https://www.example.com/robots.txt"}
                )
            return httpx.Response(200, text="User-agent: *\nDisallow: /private\n")

        self._patch_client_with_handler(monkeypatch, handler)
        f = URLFilter("https://example.com", respect_robots_txt=True)

        assert f.should_crawl("https://example.com/private/secret") is False
        assert f.should_crawl("https://example.com/public/page") is True

    def test_soft_404_html_body_still_allows_the_crawl(self, monkeypatch):
        """Static hosts often answer /robots.txt with HTTP 200 and an HTML
        404 page. Parsing that yields no rules, which must read as "no
        restrictions" rather than deny-all."""
        f = self._filter_with_robots_response(
            monkeypatch,
            _FakeResponse(200, "<html><body>Page Not Found</body></html>"),
        )

        assert f.should_crawl("https://example.com/docs/intro") is True

    def test_robots_is_fetched_from_the_site_root(self, monkeypatch):
        self._filter_with_robots_response(monkeypatch, _FakeResponse(404))

        assert [c.args[0] for c in self.get.call_args_list] == [
            "https://example.com/robots.txt"
        ]

    def test_real_robots_rules_are_still_enforced(self, monkeypatch):
        f = self._filter_with_robots_response(
            monkeypatch,
            _FakeResponse(200, "User-agent: *\nDisallow: /private\n"),
        )
        assert f.should_crawl("https://example.com/private/secret") is False
        assert f.should_crawl("https://example.com/public/page") is True
