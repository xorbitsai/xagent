"""Unit tests for web crawler."""

import logging
import subprocess
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import WebCrawlConfig
from xagent.core.tools.core.RAG_tools.web_crawler.crawler import (
    STOPPED_NO_ELIGIBLE_LINKS,
    STOPPED_NO_LINKS,
    STOPPED_PAGE_CAP,
    STOPPED_UNKNOWN,
    WebCrawler,
    _get_httpx_accept_encoding,
)
from xagent.core.tools.core.RAG_tools.web_crawler.url_filter import (
    REJECTED_EXCLUDED,
    REJECTED_OFF_DOMAIN,
    REJECTED_ROBOTS,
    REJECTED_UNPARSABLE,
)


class TestWebCrawler:
    """Test web crawler functionality."""

    @pytest.fixture
    def crawl_config(self):
        """Create a test crawl configuration.

        tls_impersonate=None makes existing tests run on the httpx path,
        so they keep mocking httpx.AsyncClient (TLS-impersonation-specific
        behavior is covered separately below).
        """
        return WebCrawlConfig(
            start_url="https://example.com",
            max_pages=5,
            max_depth=2,
            concurrent_requests=2,
            request_delay=0,
            tls_impersonate=None,
        )

    @pytest.fixture
    def sample_html(self):
        """Sample HTML content for testing."""
        return """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Main Heading</h1>
                <p>This is a test page with some content.</p>
                <a href="/page1">Page 1</a>
                <a href="/page2">Page 2</a>
                <a href="https://other.com/external">External</a>
            </body>
        </html>
        """

    @pytest.mark.asyncio
    async def test_crawler_initialization(self, crawl_config):
        """Test crawler initialization."""
        crawler = WebCrawler(crawl_config)

        assert crawler.config == crawl_config
        assert len(crawler.visited_urls) == 0
        assert len(crawler.pending_urls) == 0
        assert len(crawler.crawl_results) == 0

    def test_default_tls_impersonate_uses_httpx(self):
        """Unmodified configs should stay on the plain httpx path by default."""
        config = WebCrawlConfig(start_url="https://example.com")

        assert config.tls_impersonate is None

    def test_httpx_accept_encoding_excludes_brotli_without_decoder(self):
        """Do not advertise Brotli unless httpx can decode it."""
        with patch(
            "xagent.core.tools.core.RAG_tools.web_crawler.crawler."
            "importlib.util.find_spec",
            return_value=None,
        ):
            assert _get_httpx_accept_encoding() == "gzip, deflate"

    def test_httpx_accept_encoding_includes_brotli_with_decoder(self):
        """Advertise Brotli when a supported decoder package is installed."""

        def fake_find_spec(name):
            return object() if name == "brotlicffi" else None

        with patch(
            "xagent.core.tools.core.RAG_tools.web_crawler.crawler."
            "importlib.util.find_spec",
            side_effect=fake_find_spec,
        ):
            assert _get_httpx_accept_encoding() == "gzip, deflate, br"

    def test_httpx_accept_encoding_works_in_clean_interpreter(self):
        """Crawler should explicitly load importlib.util before using find_spec."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import importlib, sys; "
                    "sys.modules.pop('importlib.util', None); "
                    "delattr(importlib, 'util') if hasattr(importlib, 'util') else None; "
                    "from xagent.core.tools.core.RAG_tools.web_crawler.crawler "
                    "import _get_httpx_accept_encoding; "
                    "print(_get_httpx_accept_encoding())"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        assert result.stdout.strip() in {
            "gzip, deflate",
            "gzip, deflate, br",
        }

    @pytest.mark.asyncio
    async def test_crawl_single_page(self, crawl_config, sample_html):
        """Test crawling a single page."""
        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = sample_html

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            crawler = WebCrawler(crawl_config)
            results = await crawler.crawl()

        assert len(results) >= 1
        assert any(r.url == "https://example.com" for r in results)

    @pytest.mark.asyncio
    async def test_crawl_with_links(self, crawl_config, sample_html):
        """Test crawling and link discovery."""
        # Mock HTTP responses
        responses = {
            "https://example.com": sample_html,
            "https://example.com/page1": "<html><body><h1>Page 1</h1></body></html>",
            "https://example.com/page2": "<html><body><h1>Page 2</h1></body></html>",
        }

        def create_mock_response(url):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.text = responses.get(url, "")
            return mock_resp

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=lambda url, **kw: create_mock_response(url)
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            crawler = WebCrawler(crawl_config)
            results = await crawler.crawl()

        # Should have crawled start page and discovered links
        assert len(results) >= 1
        # Check that links were extracted
        stats = crawler.get_statistics()
        assert stats["total_urls_found"] > 0

    @pytest.mark.asyncio
    async def test_max_pages_limit(self, crawl_config, sample_html):
        """Test that max_pages limit is respected."""
        config = WebCrawlConfig(
            start_url="https://example.com",
            max_pages=2,  # Limit to 2 pages
            max_depth=3,
            concurrent_requests=1,
            request_delay=0,
            tls_impersonate=None,
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = sample_html

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            crawler = WebCrawler(config)
            await crawler.crawl()

        # Should not exceed max_pages
        assert len(crawler.visited_urls) <= 2

    @pytest.mark.asyncio
    async def test_max_depth_limit(self, sample_html):
        """Test that max_depth limit is respected."""
        config = WebCrawlConfig(
            start_url="https://example.com",
            max_pages=100,
            max_depth=1,  # Limit depth to 1
            concurrent_requests=1,
            request_delay=0,
            tls_impersonate=None,
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = sample_html

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            crawler = WebCrawler(config)
            results = await crawler.crawl()

        # All crawled pages should be at depth 0 or 1
        for result in results:
            assert result.depth <= 1

    @pytest.mark.asyncio
    async def test_http_error_handling(self, crawl_config):
        """Test handling of HTTP errors."""
        # Mock HTTP error response
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "<html><body>Not Found</body></html>"

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            crawler = WebCrawler(crawl_config)
            await crawler.crawl()

        # Should handle error gracefully
        assert len(crawler.failed_urls) > 0
        assert "https://example.com" in crawler.failed_urls

    @pytest.mark.asyncio
    async def test_network_error_handling(self, crawl_config):
        """Test handling of network errors."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.RequestError("Connection error"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            crawler = WebCrawler(crawl_config)
            await crawler.crawl()

        # Should handle error gracefully
        assert len(crawler.failed_urls) > 0

    @pytest.mark.asyncio
    async def test_insufficient_content_handling(self, crawl_config):
        """Test handling of pages with insufficient content."""
        # Mock response with very short content
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body>Hi</body></html>"

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            crawler = WebCrawler(crawl_config)
            results = await crawler.crawl()

        # Should skip pages with insufficient content
        assert len([r for r in results if r.status == "success"]) == 0

    @pytest.mark.asyncio
    async def test_rejects_unreadable_replacement_content(self, crawl_config):
        """2xx responses with heavy replacement characters must not enter KB."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body>" + ("\ufffd" * 240) + "</body></html>"

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            crawler = WebCrawler(crawl_config)
            results = await crawler.crawl()

        assert results == []
        assert "https://example.com" in crawler.failed_urls
        assert "replacement_ratio" in crawler.failed_urls["https://example.com"]

    @pytest.mark.asyncio
    async def test_rejects_null_byte_content(self, crawl_config):
        """2xx responses with null bytes are binary/undecodable enough to fail."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body>hello\x00world with more text</body></html>"

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            crawler = WebCrawler(crawl_config)
            results = await crawler.crawl()

        assert results == []
        assert "https://example.com" in crawler.failed_urls
        assert "null bytes" in crawler.failed_urls["https://example.com"]

    @pytest.mark.asyncio
    async def test_accepts_dirty_script_when_cleaned_content_is_valid(
        self, crawl_config
    ):
        """Raw script bytes should not fail if cleaned markdown is readable."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = (
            "<html><body><p>Hello world with enough text.</p>"
            "<script>const junk = '\x00\x01\ufffd';</script></body></html>"
        )

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            crawler = WebCrawler(crawl_config)
            results = await crawler.crawl()

        assert len(results) == 1
        assert results[0].content_markdown == "Hello world with enough text."
        assert "https://example.com" not in crawler.failed_urls

    @pytest.mark.asyncio
    async def test_rejects_high_control_character_content(self, crawl_config):
        """Control-character-heavy pages should be rejected as unreadable."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body>" + ("\x01" * 20) + ("readable text " * 20)

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            crawler = WebCrawler(crawl_config)
            results = await crawler.crawl()

        assert results == []
        assert "https://example.com" in crawler.failed_urls
        assert "control_ratio" in crawler.failed_urls["https://example.com"]

    @pytest.mark.asyncio
    async def test_accepts_short_readable_extracted_content(self, crawl_config):
        """Short but readable pages should keep the previous 10-char behavior."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = (
            "<html><body><h1>Contact</h1><p>Email support@example.com</p></body></html>"
        )

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            crawler = WebCrawler(crawl_config)
            results = await crawler.crawl()

        assert len(results) == 1
        assert results[0].content_markdown == "# Contact\n\nEmail support@example.com"
        assert "https://example.com" not in crawler.failed_urls

    @pytest.mark.asyncio
    async def test_accepts_short_raw_html_when_cleaned_content_is_valid(
        self, crawl_config
    ):
        """Raw HTML length should not reject concise readable pages."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<p>Hello world!</p>"

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            crawler = WebCrawler(crawl_config)
            results = await crawler.crawl()

        assert len(results) == 1
        assert results[0].content_markdown == "Hello world!"
        assert "https://example.com" not in crawler.failed_urls

    @pytest.mark.asyncio
    async def test_accepts_documentation_with_access_denied_phrase(self, crawl_config):
        """Generic security phrases can be normal documentation content."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = (
            "<html><head><title>How to fix Access denied errors</title></head>"
            "<body><h1>How to fix Access denied errors</h1>"
            "<p>This guide explains application authorization failures and "
            "how to resolve them.</p></body></html>"
        )

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            crawler = WebCrawler(crawl_config)
            results = await crawler.crawl()

        assert len(results) == 1
        assert "Access denied" in results[0].content_markdown
        assert "https://example.com" not in crawler.failed_urls

    @pytest.mark.asyncio
    async def test_accepts_readable_page_with_weak_challenge_phrase(self, crawl_config):
        """A weak marker alone should not reject readable content."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = (
            "<html><body><p>Just a moment while we explain the onboarding "
            "flow for new operators.</p></body></html>"
        )

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            crawler = WebCrawler(crawl_config)
            results = await crawler.crawl()

        assert len(results) == 1
        assert "Just a moment" in results[0].content_markdown
        assert "https://example.com" not in crawler.failed_urls

    @pytest.mark.asyncio
    async def test_accepts_short_content_with_single_decoding_artifacts(
        self, crawl_config
    ):
        """Short readable markdown should tolerate one-off artifact chars."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body>placeholder</body></html>"

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            crawler = WebCrawler(crawl_config)
            crawler.content_cleaner.clean_and_convert = MagicMock(
                return_value={
                    "title": "",
                    "content_markdown": "Short readable text with one � and one \x01.",
                    "content_length": 41,
                }
            )
            crawler.content_cleaner.is_valid_content = MagicMock(return_value=True)
            results = await crawler.crawl()

        assert len(results) == 1
        assert "https://example.com" not in crawler.failed_urls

    @pytest.mark.asyncio
    async def test_same_domain_filtering(self, sample_html):
        """Test same domain filtering."""
        config = WebCrawlConfig(
            start_url="https://example.com",
            max_pages=10,
            max_depth=2,
            same_domain_only=True,
            concurrent_requests=1,
            request_delay=0,
            tls_impersonate=None,
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = sample_html

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            crawler = WebCrawler(config)
            results = await crawler.crawl()

        # External links should not be crawled
        assert not any(r.url == "https://other.com/external" for r in results)

    @pytest.mark.asyncio
    async def test_get_statistics(self, crawl_config, sample_html):
        """Test statistics collection."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = sample_html

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            crawler = WebCrawler(crawl_config)
            await crawler.crawl()

        stats = crawler.get_statistics()
        assert "total_urls_found" in stats
        assert "visited_urls" in stats
        assert "successful_pages" in stats
        assert "failed_pages" in stats
        assert "pending_urls" in stats

    @pytest.mark.asyncio
    async def test_progress_callback(self, crawl_config, sample_html):
        """Test progress callback functionality."""
        progress_updates = []

        def progress_callback(message, completed, total):
            progress_updates.append((message, completed, total))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = sample_html

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            crawler = WebCrawler(crawl_config, progress_callback)
            await crawler.crawl()

        # Progress callback should have been called
        assert len(progress_updates) > 0

    @staticmethod
    def _make_cffi_session_factory(call_log, response_for):
        """Return a side_effect that builds a fresh AsyncMock per call.

        Args:
            call_log: list to append the impersonate spec on every .get()
            response_for: callable(impersonate) -> MagicMock response
        """

        def make_session(impersonate=None, **kwargs):
            sess = AsyncMock()
            sess.__aenter__ = AsyncMock(return_value=sess)
            sess.__aexit__ = AsyncMock(return_value=None)

            async def get(url, **kw):
                call_log.append(impersonate)
                return response_for(impersonate)

            sess.get = AsyncMock(side_effect=get)
            return sess

        return make_session

    @staticmethod
    def _install_fake_cffi(monkeypatch):
        """Install fake curl_cffi modules so optional-dependency tests stay hermetic."""
        cffi_module = types.ModuleType("curl_cffi")
        requests_module = types.ModuleType("curl_cffi.requests")
        requests_module.AsyncSession = MagicMock()
        cffi_module.requests = requests_module
        monkeypatch.setitem(sys.modules, "curl_cffi", cffi_module)
        monkeypatch.setitem(sys.modules, "curl_cffi.requests", requests_module)
        return requests_module

    @pytest.mark.asyncio
    async def test_tls_fallback_chain_advances_on_waf_block(
        self, sample_html, monkeypatch
    ):
        """When chain[0] returns 403 and chain[1] returns 200, the second
        fingerprint must be used and the page must succeed. httpx must
        not be touched at all on the auto path.
        """
        config = WebCrawlConfig(
            start_url="https://example.com",
            max_pages=1,
            request_delay=0,
            tls_impersonate="auto",
        )

        call_log = []
        cffi_requests = self._install_fake_cffi(monkeypatch)

        def response_for(impersonate):
            resp = MagicMock()
            if impersonate == "chrome116":
                resp.status_code = 403
                resp.text = "blocked"
            else:
                resp.status_code = 200
                resp.text = sample_html
            return resp

        with (
            patch.object(
                cffi_requests,
                "AsyncSession",
                side_effect=self._make_cffi_session_factory(call_log, response_for),
            ) as p_cffi,
            patch("httpx.AsyncClient") as p_httpx,
        ):
            crawler = WebCrawler(config)
            results = await crawler.crawl()

        # Three sessions opened (one per fingerprint), httpx not used at all
        assert p_cffi.call_count == 3
        p_httpx.assert_not_called()
        # First two fingerprints tried in order; chain[1] succeeded
        assert call_log[0] == "chrome116"
        assert call_log[1] == "safari17_0"
        assert len(results) == 1
        assert results[0].status == "success"

    @pytest.mark.asyncio
    async def test_tls_impersonate_none_uses_httpx_only(self, sample_html):
        """When tls_impersonate=None, curl_cffi must NEVER be touched."""
        config = WebCrawlConfig(
            start_url="https://example.com",
            max_pages=1,
            request_delay=0,
            tls_impersonate=None,
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = sample_html

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with (
            patch("httpx.AsyncClient", return_value=mock_client) as p_httpx,
            patch(
                "importlib.import_module",
                side_effect=AssertionError("curl_cffi should not be imported"),
            ) as p_import,
        ):
            crawler = WebCrawler(config)
            await crawler.crawl()

        p_httpx.assert_called()
        p_import.assert_not_called()

    def test_tls_impersonate_requires_waf_crawl_extra(self):
        """Opt-in TLS impersonation should fail early when curl_cffi is absent."""
        config = WebCrawlConfig(
            start_url="https://example.com",
            max_pages=1,
            request_delay=0,
            tls_impersonate="auto",
        )
        error = ModuleNotFoundError("No module named 'curl_cffi'")
        error.name = "curl_cffi"

        with (
            patch("importlib.import_module", side_effect=error),
            pytest.raises(ImportError, match="waf-crawl"),
        ):
            WebCrawler(config)

    @pytest.mark.asyncio
    async def test_404_does_not_trigger_fallback_chain(self, monkeypatch):
        """Ordinary HTTP errors (404, 401, 500) must fail fast.

        Only WAF-like statuses (403, 429, 503...) should advance the
        fallback chain. Otherwise we'd 3x the cost of every dead link
        and write misleading "TLS fallback exhausted" warnings for
        ordinary content errors.
        """
        config = WebCrawlConfig(
            start_url="https://example.com",
            max_pages=1,
            request_delay=0,
            tls_impersonate="auto",
        )

        call_log = []
        cffi_requests = self._install_fake_cffi(monkeypatch)

        def response_for(impersonate):
            resp = MagicMock()
            resp.status_code = 404
            resp.text = "<html><body>not found</body></html>"
            return resp

        with patch.object(
            cffi_requests,
            "AsyncSession",
            side_effect=self._make_cffi_session_factory(call_log, response_for),
        ):
            crawler = WebCrawler(config)
            await crawler.crawl()

        # Only the first fingerprint should have been tried
        assert call_log == ["chrome116"]
        # And it's recorded as failed
        assert "https://example.com" in crawler.failed_urls
        assert "404" in crawler.failed_urls["https://example.com"]

    @pytest.mark.asyncio
    async def test_challenge_page_advances_chain(self, sample_html, monkeypatch):
        """A 200 response that's actually a CF JS challenge wrapper must
        be treated like a WAF block (advance to next fingerprint), not
        accepted as content -- otherwise the KB gets polluted with
        "Just a moment..." stub pages.
        """
        config = WebCrawlConfig(
            start_url="https://example.com",
            max_pages=1,
            request_delay=0,
            tls_impersonate="auto",
        )

        challenge_body = (
            "<!DOCTYPE html><html><head><title>Just a moment...</title>"
            "</head><body>Checking your browser before accessing the "
            "site. cf-challenge in progress.</body></html>"
        )
        call_log = []
        cffi_requests = self._install_fake_cffi(monkeypatch)

        def response_for(impersonate):
            resp = MagicMock()
            resp.status_code = 200
            if impersonate == "chrome116":
                resp.text = challenge_body
            else:
                resp.text = sample_html
            return resp

        with patch.object(
            cffi_requests,
            "AsyncSession",
            side_effect=self._make_cffi_session_factory(call_log, response_for),
        ):
            crawler = WebCrawler(config)
            results = await crawler.crawl()

        # chain[0] returned a 200 challenge -> fallback to chain[1]
        assert call_log[0] == "chrome116"
        assert call_log[1] == "safari17_0"
        assert len(results) == 1
        assert results[0].status == "success"

    @pytest.mark.asyncio
    async def test_unreadable_200_advances_tls_auto_chain(
        self, sample_html, monkeypatch
    ):
        """Unreadable cleaned content should not short-circuit auto TLS."""
        config = WebCrawlConfig(
            start_url="https://example.com",
            max_pages=1,
            request_delay=0,
            tls_impersonate="auto",
        )
        call_log = []
        cffi_requests = self._install_fake_cffi(monkeypatch)

        def response_for(impersonate):
            resp = MagicMock()
            resp.status_code = 200
            if impersonate == "chrome116":
                resp.text = "<html><body>" + ("\ufffd" * 120) + "</body></html>"
            else:
                resp.text = sample_html
            return resp

        with patch.object(
            cffi_requests,
            "AsyncSession",
            side_effect=self._make_cffi_session_factory(call_log, response_for),
        ):
            crawler = WebCrawler(config)
            results = await crawler.crawl()

        assert call_log[0] == "chrome116"
        assert call_log[1] == "safari17_0"
        assert len(results) == 1
        assert results[0].status == "success"

    @pytest.mark.asyncio
    async def test_empty_extracted_content_advances_tls_auto_chain(
        self, sample_html, monkeypatch
    ):
        """A 200 JS shell that cleans empty should still try the next fp."""
        config = WebCrawlConfig(
            start_url="https://example.com",
            max_pages=1,
            request_delay=0,
            tls_impersonate="auto",
        )
        call_log = []
        cffi_requests = self._install_fake_cffi(monkeypatch)

        def response_for(impersonate):
            resp = MagicMock()
            resp.status_code = 200
            if impersonate == "chrome116":
                resp.text = (
                    "<html><body><script>location.href='/'</script></body></html>"
                )
            else:
                resp.text = sample_html
            return resp

        with patch.object(
            cffi_requests,
            "AsyncSession",
            side_effect=self._make_cffi_session_factory(call_log, response_for),
        ):
            crawler = WebCrawler(config)
            results = await crawler.crawl()

        assert call_log[0] == "chrome116"
        assert call_log[1] == "safari17_0"
        assert len(results) == 1
        assert results[0].status == "success"

    @pytest.mark.asyncio
    async def test_cleaner_exception_does_not_advance_tls_auto_chain(
        self, sample_html, monkeypatch
    ):
        """Cleaner/parser failures should not be labeled as TLS failures."""
        config = WebCrawlConfig(
            start_url="https://example.com",
            max_pages=1,
            request_delay=0,
            tls_impersonate="auto",
        )
        call_log = []
        cffi_requests = self._install_fake_cffi(monkeypatch)

        def response_for(impersonate):
            resp = MagicMock()
            resp.status_code = 200
            resp.text = sample_html
            return resp

        with patch.object(
            cffi_requests,
            "AsyncSession",
            side_effect=self._make_cffi_session_factory(call_log, response_for),
        ):
            crawler = WebCrawler(config)
            crawler.content_cleaner.clean_and_convert = MagicMock(
                side_effect=ValueError("cleaner boom")
            )
            results = await crawler.crawl()

        assert call_log == ["chrome116"]
        assert results == []
        assert "https://example.com" in crawler.failed_urls
        assert crawler.failed_urls["https://example.com"] == (
            "Unexpected error: cleaner boom"
        )

    @pytest.mark.asyncio
    async def test_exhausted_challenge_pages_fail_crawl(self, monkeypatch):
        """If every fingerprint returns a 200 challenge wrapper, fail the URL."""
        config = WebCrawlConfig(
            start_url="https://example.com",
            max_pages=1,
            request_delay=0,
            tls_impersonate="auto",
        )

        challenge_body = (
            "<!DOCTYPE html><html><head><title>Just a moment...</title>"
            "</head><body>Checking your browser before accessing the "
            "site. cf-challenge in progress.</body></html>"
        )
        call_log = []
        cffi_requests = self._install_fake_cffi(monkeypatch)

        def response_for(impersonate):
            resp = MagicMock()
            resp.status_code = 200
            resp.text = challenge_body
            return resp

        with patch.object(
            cffi_requests,
            "AsyncSession",
            side_effect=self._make_cffi_session_factory(call_log, response_for),
        ):
            crawler = WebCrawler(config)
            results = await crawler.crawl()

        assert call_log == ["chrome116", "safari17_0", "safari15_5"]
        assert results == []
        assert "https://example.com" in crawler.failed_urls
        assert crawler.failed_urls["https://example.com"] == (
            "TLS fallback exhausted with challenge page"
        )

    @pytest.mark.asyncio
    async def test_tls_exception_chain_logs_warning(self, monkeypatch, caplog):
        """If every fingerprint raises, operators should get a warning summary."""
        config = WebCrawlConfig(
            start_url="https://example.com",
            max_pages=1,
            request_delay=0,
            tls_impersonate="auto",
        )
        cffi_requests = self._install_fake_cffi(monkeypatch)

        def make_session(impersonate=None, **kwargs):
            sess = AsyncMock()
            sess.__aenter__ = AsyncMock(return_value=sess)
            sess.__aexit__ = AsyncMock(return_value=None)
            sess.get = AsyncMock(side_effect=TimeoutError(f"{impersonate} timed out"))
            return sess

        with (
            patch.object(cffi_requests, "AsyncSession", side_effect=make_session),
            caplog.at_level(logging.WARNING),
        ):
            crawler = WebCrawler(config)
            await crawler.crawl()

        assert "All TLS fingerprints failed" in caplog.text
        assert "chrome116:TimeoutError" in caplog.text
        assert "https://example.com" in crawler.failed_urls


class TestRobotsAtCrawlLevel:
    """The url_filter unit tests cannot catch this on their own: the start URL
    never passes through should_crawl, so a deny-all filter still yields one
    page and a non-zero total_urls_found. Only page count across a real crawl
    distinguishes "robots absent" from "robots denied everything"."""

    HTML = """
        <html><body><h1>Docs</h1>
        <p>Enough body text here to pass the content length check.</p>
        <a href="/a">A</a><a href="/b">B</a>
        </body></html>
    """

    def _client(self):
        response = MagicMock()
        response.status_code = 200
        response.text = self.HTML
        client = AsyncMock()
        client.get.return_value = response
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock()
        return client

    def _robots(self, monkeypatch, status, text=""):
        client = MagicMock()
        client.return_value.__enter__.return_value.get.return_value = MagicMock(
            status_code=status, text=text
        )
        monkeypatch.setattr(httpx, "Client", client)

    def test_robots_fetch_inherits_the_crawl_user_agent(self):
        """Probing robots.txt as python-httpx while crawling as a browser lets
        a WAF 403 read as "this site has no rules"."""
        config = WebCrawlConfig(
            start_url="https://example.com",
            user_agent="XagentBot/1.0",
            tls_impersonate=None,
            respect_robots_txt=False,
        )

        assert WebCrawler(config).url_filter.user_agent == "XagentBot/1.0"

    @pytest.mark.asyncio
    async def test_absent_robots_txt_does_not_stop_at_the_start_page(self, monkeypatch):
        self._robots(monkeypatch, 404)
        config = WebCrawlConfig(
            start_url="https://example.com",
            max_pages=10,
            max_depth=2,
            request_delay=0,
            tls_impersonate=None,
            respect_robots_txt=True,
        )
        with patch("httpx.AsyncClient", return_value=self._client()):
            results = await WebCrawler(config).crawl()

        assert len(results) > 1

    @pytest.mark.asyncio
    async def test_real_disallow_still_stops_the_crawl(self, monkeypatch):
        self._robots(monkeypatch, 200, "User-agent: *\nDisallow: /\n")
        config = WebCrawlConfig(
            start_url="https://example.com",
            max_pages=10,
            max_depth=2,
            request_delay=0,
            tls_impersonate=None,
            respect_robots_txt=True,
        )
        with patch("httpx.AsyncClient", return_value=self._client()):
            crawler = WebCrawler(config)
            results = await crawler.crawl()

        assert len(results) == 1
        # Every other stop_reason test stubs rejection_reason. This is the one
        # place the real URLFilter decides, so it is where the signal has to be
        # checked against an actual Disallow rule.
        assert crawler.stop_reason == STOPPED_NO_ELIGIBLE_LINKS


class TestStopReason:
    """The crawl loop records why it ended, so ingestion can tell a configured
    stop from a site that refused every link."""

    @staticmethod
    def _mock_client(html: str):
        response = MagicMock()
        response.status_code = 200
        response.text = html
        client = AsyncMock()
        client.get.return_value = response
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock()
        return client

    HTML_WITH_LINKS = """
        <html><body><h1>Docs</h1>
        <p>Enough body text here to pass the content length check.</p>
        <a href="/a">A</a><a href="/b">B</a>
        </body></html>
    """

    HTML_NO_LINKS = """
        <html><body><h1>Only page</h1>
        <p>Enough body text here to pass the content length check.</p>
        </body></html>
    """

    @pytest.mark.asyncio
    async def test_robots_refusing_every_link_is_flagged(self):
        config = WebCrawlConfig(
            start_url="https://example.com",
            max_pages=10,
            max_depth=2,
            request_delay=0,
            tls_impersonate=None,
        )
        with patch(
            "httpx.AsyncClient", return_value=self._mock_client(self.HTML_WITH_LINKS)
        ):
            crawler = WebCrawler(config)
            # Site allows the start page but refuses everything it links to.
            crawler.url_filter.rejection_reason = lambda url, ua="*": REJECTED_ROBOTS
            await crawler.crawl()

        assert crawler.stop_reason == STOPPED_NO_ELIGIBLE_LINKS
        assert crawler.total_links_queued == 0
        assert crawler.link_rejections[REJECTED_ROBOTS] == 2

    @pytest.mark.asyncio
    async def test_off_domain_links_only_is_not_flagged(self):
        """Deliberate filtering is the crawl doing what it was told. Uses the
        real filter: same_domain_only is what has to reject these."""
        config = WebCrawlConfig(
            start_url="https://example.com",
            max_pages=10,
            max_depth=2,
            request_delay=0,
            tls_impersonate=None,
            respect_robots_txt=False,
        )
        html = """
            <html><body><h1>Docs</h1>
            <p>Enough body text here to pass the content length check.</p>
            <a href="https://other.com/a">A</a><a href="https://other.com/b">B</a>
            </body></html>
        """
        with patch("httpx.AsyncClient", return_value=self._mock_client(html)):
            crawler = WebCrawler(config)
            await crawler.crawl()

        assert crawler.link_rejections[REJECTED_OFF_DOMAIN] == 2
        assert crawler.stop_reason == STOPPED_NO_LINKS

    @pytest.mark.asyncio
    async def test_one_in_scope_refusal_outweighs_many_out_of_scope_links(self):
        """Links that were never in scope cannot vouch for the site. A ratio
        over all rejections lets fifty off-domain links bury the one refusal
        that actually closed the crawl."""
        html = (
            "<html><body><h1>Docs</h1><p>Enough body text to pass the check.</p>"
            + "".join(f'<a href="https://other{i}.com/x">O</a>' for i in range(50))
            + '<a href="/blocked">B</a></body></html>'
        )
        config = WebCrawlConfig(
            start_url="https://example.com",
            max_pages=10,
            max_depth=2,
            request_delay=0,
            tls_impersonate=None,
        )
        with patch("httpx.AsyncClient", return_value=self._mock_client(html)):
            crawler = WebCrawler(config)
            crawler.url_filter.rejection_reason = (
                lambda url, ua="*": REJECTED_ROBOTS
                if url.startswith("https://example.com/")
                else REJECTED_OFF_DOMAIN
            )
            await crawler.crawl()

        assert crawler.stop_reason == STOPPED_NO_ELIGIBLE_LINKS

    @pytest.mark.asyncio
    async def test_unsupported_schemes_alone_are_not_a_refusal(self):
        """ftp:/data:/intent: anchors are this crawler's own limit. No site
        policy was consulted, so a one-page import stays a success."""
        html = """
            <html><body><h1>Only page</h1>
            <p>Enough body text here to pass the content length check.</p>
            <a href="ftp://example.com/f">F</a><a href="data:text/plain,x">D</a>
            </body></html>
        """
        config = WebCrawlConfig(
            start_url="https://example.com",
            max_pages=10,
            max_depth=2,
            request_delay=0,
            tls_impersonate=None,
            respect_robots_txt=False,
        )
        with patch("httpx.AsyncClient", return_value=self._mock_client(html)):
            crawler = WebCrawler(config)
            await crawler.crawl()

        assert crawler.link_rejections[REJECTED_UNPARSABLE] > 0
        assert crawler.stop_reason == STOPPED_NO_LINKS

    @pytest.mark.asyncio
    async def test_an_already_seen_link_is_not_frontier_progress(self):
        """A self link survives every filter and queues nothing. Counting it as
        progress hides that the only forward page was refused."""
        html = """
            <html><body><h1>Docs</h1>
            <p>Enough body text here to pass the content length check.</p>
            <a href="https://example.com">home</a><a href="/child">C</a>
            </body></html>
        """
        config = WebCrawlConfig(
            start_url="https://example.com",
            max_pages=10,
            max_depth=2,
            request_delay=0,
            tls_impersonate=None,
        )
        with patch("httpx.AsyncClient", return_value=self._mock_client(html)):
            crawler = WebCrawler(config)
            crawler.url_filter.rejection_reason = (
                lambda url, ua="*": REJECTED_ROBOTS if url.endswith("/child") else None
            )
            await crawler.crawl()

        assert crawler.total_links_queued == 0
        assert crawler.stop_reason == STOPPED_NO_ELIGIBLE_LINKS

    @pytest.mark.asyncio
    async def test_a_refusal_alongside_real_progress_is_not_a_blocked_crawl(self):
        """The common shape: robots forbids /private and the rest crawls fine.
        A refusal only closed the door if nothing was queued behind it."""
        config = WebCrawlConfig(
            start_url="https://example.com",
            max_pages=10,
            max_depth=2,
            request_delay=0,
            tls_impersonate=None,
        )
        with patch(
            "httpx.AsyncClient", return_value=self._mock_client(self.HTML_WITH_LINKS)
        ):
            crawler = WebCrawler(config)
            crawler.url_filter.rejection_reason = (
                lambda url, ua="*": REJECTED_ROBOTS if url.endswith("/b") else None
            )
            await crawler.crawl()

        assert crawler.link_rejections[REJECTED_ROBOTS] > 0
        assert crawler.total_links_queued > 0
        assert crawler.stop_reason == STOPPED_NO_LINKS

    @pytest.mark.asyncio
    async def test_a_shared_link_is_counted_once_across_pages(self):
        """A nav or footer link rejected on every page is one refused link, not
        one per page. The count reaches the user, so inflating it misstates how
        much the site turned away."""
        html = """
            <html><body><h1>Docs</h1>
            <p>Enough body text here to pass the content length check.</p>
            <a href="/next">next</a><a href="https://other.com/nav">nav</a>
            </body></html>
        """
        config = WebCrawlConfig(
            start_url="https://example.com",
            max_pages=10,
            max_depth=2,
            request_delay=0,
            tls_impersonate=None,
            respect_robots_txt=False,
        )
        with patch("httpx.AsyncClient", return_value=self._mock_client(html)):
            crawler = WebCrawler(config)
            await crawler.crawl()

        assert len(crawler.visited_urls) > 1
        assert crawler.link_rejections[REJECTED_OFF_DOMAIN] == 1

    @pytest.mark.asyncio
    async def test_the_same_link_written_differently_is_counted_once(self):
        """rejection_reason() judges the normalized URL, so the count has to be
        keyed on it too. Keying on the raw href counts one refused link three
        times, which is the number the user is shown."""
        html = """
            <html><body><h1>Docs</h1>
            <p>Enough body text here to pass the content length check.</p>
            <a href="/blog#top">a</a><a href="/blog#new">b</a>
            <a href="https://EXAMPLE.com/blog">c</a>
            </body></html>
        """
        config = WebCrawlConfig(
            start_url="https://example.com",
            max_pages=10,
            max_depth=2,
            request_delay=0,
            tls_impersonate=None,
            exclude_patterns=[r"/blog"],
            respect_robots_txt=False,
        )
        with patch("httpx.AsyncClient", return_value=self._mock_client(html)):
            crawler = WebCrawler(config)
            await crawler.crawl()

        assert crawler.link_rejections[REJECTED_EXCLUDED] == 1

    @pytest.mark.asyncio
    async def test_the_page_cap_outranks_a_refusal(self):
        """Documented in get_statistics(): at max_pages=1 the cap wins even
        when the site also refused every link. The crawl was told to fetch one
        page and did, so the refusal did not change the outcome."""
        config = WebCrawlConfig(
            start_url="https://example.com",
            max_pages=1,
            max_depth=2,
            request_delay=0,
            tls_impersonate=None,
        )
        with patch(
            "httpx.AsyncClient", return_value=self._mock_client(self.HTML_WITH_LINKS)
        ):
            crawler = WebCrawler(config)
            crawler.url_filter.rejection_reason = lambda url, ua="*": REJECTED_ROBOTS
            await crawler.crawl()

        assert crawler.link_rejections[REJECTED_ROBOTS] > 0
        assert crawler.total_links_queued == 0
        assert crawler.stop_reason == STOPPED_PAGE_CAP

    @pytest.mark.asyncio
    async def test_page_cap_is_recorded(self):
        config = WebCrawlConfig(
            start_url="https://example.com",
            max_pages=1,
            max_depth=2,
            request_delay=0,
            tls_impersonate=None,
        )
        with patch(
            "httpx.AsyncClient", return_value=self._mock_client(self.HTML_WITH_LINKS)
        ):
            crawler = WebCrawler(config)
            await crawler.crawl()

        assert crawler.stop_reason == STOPPED_PAGE_CAP

    @pytest.mark.asyncio
    async def test_an_out_of_scope_link_does_not_excuse_a_refusal(self):
        """Presence is not dominance: a scoped crawl that rejects many
        off-domain links and meets one robots-disallowed link was doing what it
        was configured to do."""
        config = WebCrawlConfig(
            start_url="https://example.com",
            max_pages=10,
            max_depth=2,
            request_delay=0,
            tls_impersonate=None,
        )
        with patch(
            "httpx.AsyncClient", return_value=self._mock_client(self.HTML_WITH_LINKS)
        ):
            crawler = WebCrawler(config)
            reasons = iter([REJECTED_OFF_DOMAIN, REJECTED_ROBOTS])
            crawler.url_filter.rejection_reason = lambda url, ua="*": next(
                reasons, REJECTED_OFF_DOMAIN
            )
            await crawler.crawl()

        assert crawler.link_rejections[REJECTED_OFF_DOMAIN] == 1
        assert crawler.link_rejections[REJECTED_ROBOTS] == 1
        # The off-domain link was never a candidate, so it cannot vouch for a
        # site that refused the only one that was.
        assert crawler.stop_reason == STOPPED_NO_ELIGIBLE_LINKS

    def test_stop_reason_defaults_to_unknown_before_the_loop_runs(self):
        """A success-implying default would misreport a crawl that raised
        before the loop epilogue assigned a real reason."""
        config = WebCrawlConfig(
            start_url="https://example.com", request_delay=0, tls_impersonate=None
        )
        assert WebCrawler(config).stop_reason == STOPPED_UNKNOWN

    @pytest.mark.asyncio
    async def test_site_with_no_links_is_not_flagged(self):
        config = WebCrawlConfig(
            start_url="https://example.com",
            max_pages=10,
            max_depth=2,
            request_delay=0,
            tls_impersonate=None,
        )
        with patch(
            "httpx.AsyncClient", return_value=self._mock_client(self.HTML_NO_LINKS)
        ):
            crawler = WebCrawler(config)
            await crawler.crawl()

        assert crawler.stop_reason == STOPPED_NO_LINKS
        assert crawler.link_rejections == {}
