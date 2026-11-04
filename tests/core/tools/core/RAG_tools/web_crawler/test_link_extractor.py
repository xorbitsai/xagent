"""Unit tests for link extractor."""

from xagent.core.tools.core.RAG_tools.web_crawler.link_extractor import LinkExtractor


class TestLinkExtractor:
    """Test link extraction functionality."""

    def test_extract_absolute_links(self):
        """Test extraction of absolute links."""
        extractor = LinkExtractor("https://example.com")

        html = """
        <html>
            <body>
                <a href="https://example.com/page1">Page 1</a>
                <a href="https://other.com/page2">Page 2</a>
            </body>
        </html>
        """

        links = extractor.extract_links(html, "https://example.com/")
        assert len(links) == 2
        assert "https://example.com/page1" in links
        assert "https://other.com/page2" in links

    def test_extract_relative_links(self):
        """Test extraction and conversion of relative links."""
        extractor = LinkExtractor("https://example.com")

        html = """
        <html>
            <body>
                <a href="/page1">Page 1</a>
                <a href="page2">Page 2</a>
                <a href="../page3">Page 3</a>
            </body>
        </html>
        """

        links = extractor.extract_links(html, "https://example.com/docs/")
        assert "https://example.com/page1" in links
        assert "https://example.com/docs/page2" in links
        assert "https://example.com/page3" in links

    def test_skip_empty_links(self):
        """Test skipping empty and invalid links."""
        extractor = LinkExtractor("https://example.com")

        html = """
        <html>
            <body>
                <a href="">Empty</a>
                <a>No href</a>
                <a href="   ">Whitespace</a>
                <a href="/valid">Valid</a>
            </body>
        </html>
        """

        links = extractor.extract_links(html, "https://example.com/")
        assert len(links) == 1
        assert "https://example.com/valid" in links

    def test_skip_javascript_and_special_protocols(self):
        """Test skipping JavaScript and special protocol links."""
        extractor = LinkExtractor("https://example.com")

        html = """
        <html>
            <body>
                <a href="javascript:void(0)">JS Link</a>
                <a href="javascript:alert('x')">JS Alert</a>
                <a href="mailto:test@example.com">Email</a>
                <a href="tel:+1234567890">Phone</a>
                <a href="/valid">Valid</a>
            </body>
        </html>
        """

        links = extractor.extract_links(html, "https://example.com/")
        assert len(links) == 1
        assert "https://example.com/valid" in links

    def test_skip_anchor_links(self):
        """Test skipping anchor-only links."""
        extractor = LinkExtractor("https://example.com")

        html = """
        <html>
            <body>
                <a href="#section1">Section 1</a>
                <a href="#top">Top</a>
                <a href="/page1">Page 1</a>
            </body>
        </html>
        """

        links = extractor.extract_links(html, "https://example.com/")
        assert len(links) == 1
        assert "https://example.com/page1" in links

    def test_duplicate_links(self):
        """Test that duplicate links are deduplicated."""
        extractor = LinkExtractor("https://example.com")

        html = """
        <html>
            <body>
                <a href="/page1">Link 1</a>
                <a href="/page1">Link 1 Again</a>
                <a href="/page2">Link 2</a>
                <a href="/page2">Link 2 Again</a>
            </body>
        </html>
        """

        links = extractor.extract_links(html, "https://example.com/")
        assert len(links) == 2
        assert "https://example.com/page1" in links
        assert "https://example.com/page2" in links

    def test_links_with_query_parameters(self):
        """Test links with query parameters."""
        extractor = LinkExtractor("https://example.com")

        html = """
        <html>
            <body>
                <a href="/page?param=value&foo=bar">Query Link</a>
            </body>
        </html>
        """

        links = extractor.extract_links(html, "https://example.com/")
        assert "https://example.com/page?param=value&foo=bar" in links

    def test_links_with_fragments(self):
        """Test that fragments are preserved in links."""
        extractor = LinkExtractor("https://example.com")

        html = """
        <html>
            <body>
                <a href="/page#section">Fragment Link</a>
            </body>
        </html>
        """

        links = extractor.extract_links(html, "https://example.com/")
        # Fragments should be preserved for link extraction (filtered later)
        assert "https://example.com/page#section" in links

    def test_empty_html(self):
        """Test extraction from empty HTML."""
        extractor = LinkExtractor("https://example.com")

        links = extractor.extract_links("", "https://example.com/")
        assert len(links) == 0

    def test_html_with_no_links(self):
        """Test HTML without any links."""
        extractor = LinkExtractor("https://example.com")

        html = """
        <html>
            <body>
                <p>No links here</p>
                <div>Just content</div>
            </body>
        </html>
        """

        links = extractor.extract_links(html, "https://example.com/")
        assert len(links) == 0

    def test_malformed_html(self):
        """Test handling of malformed HTML."""
        extractor = LinkExtractor("https://example.com")

        html = """
        <html>
            <body>
                <a href="/page1">Link 1
                <a href="/page2">Link 2</a>
                <a href="/page3">Link 3
            </body>
        """

        # Should not crash, just extract what it can
        links = extractor.extract_links(html, "https://example.com/")
        assert len(links) >= 1
