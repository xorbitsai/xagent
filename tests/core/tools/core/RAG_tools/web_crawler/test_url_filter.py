"""Unit tests for URL filter."""

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
