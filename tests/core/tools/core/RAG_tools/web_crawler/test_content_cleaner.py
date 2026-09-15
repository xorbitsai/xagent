"""Unit tests for content cleaner."""

from xagent.core.tools.core.RAG_tools.web_crawler.content_cleaner import ContentCleaner


class TestContentCleaner:
    """Test content cleaning functionality."""

    def test_extract_title_from_title_tag(self):
        """Test title extraction from <title> tag."""
        cleaner = ContentCleaner()

        html = """
        <html>
            <head><title>Test Page Title</title></head>
            <body>Content</body>
        </html>
        """

        result = cleaner.clean_and_convert(html, "https://example.com")
        assert result["title"] == "Test Page Title"

    def test_extract_title_from_h1(self):
        """Test title extraction from <h1> tag."""
        cleaner = ContentCleaner()

        html = """
        <html>
            <body>
                <h1>Main Heading</h1>
                <p>Content</p>
            </body>
        </html>
        """

        result = cleaner.clean_and_convert(html, "https://example.com")
        assert result["title"] == "Main Heading"

    def test_extract_title_from_meta(self):
        """Test title extraction from meta tag."""
        cleaner = ContentCleaner()

        html = """
        <html>
            <head>
                <meta property="og:title" content="Meta Title" />
            </head>
            <body>Content</body>
        </html>
        """

        result = cleaner.clean_and_convert(html, "https://example.com")
        assert result["title"] == "Meta Title"

    def test_remove_script_and_style(self):
        """Test removal of script and style elements."""
        cleaner = ContentCleaner()

        html = """
        <html>
            <body>
                <h1>Title</h1>
                <script>alert('test');</script>
                <style>body { color: red; }</style>
                <p>Content</p>
            </body>
        </html>
        """

        result = cleaner.clean_and_convert(html, "https://example.com")
        markdown = result["content_markdown"]
        assert "alert" not in markdown
        assert "color: red" not in markdown
        assert "Content" in markdown

    def test_html_to_markdown_conversion(self):
        """Test basic HTML to Markdown conversion."""
        cleaner = ContentCleaner()

        html = """
        <html>
            <body>
                <h1>Heading 1</h1>
                <h2>Heading 2</h2>
                <p>This is a paragraph.</p>
                <ul>
                    <li>Item 1</li>
                    <li>Item 2</li>
                </ul>
            </body>
        </html>
        """

        result = cleaner.clean_and_convert(html, "https://example.com")
        markdown = result["content_markdown"]

        assert "# Heading 1" in markdown
        assert "## Heading 2" in markdown
        assert "This is a paragraph" in markdown
        assert "* Item 1" in markdown or "- Item 1" in markdown

    def test_custom_remove_selectors(self):
        """Test custom element removal."""
        cleaner = ContentCleaner(remove_selectors=["nav", "footer", ".ad"])

        html = """
        <html>
            <body>
                <nav>Navigation</nav>
                <h1>Title</h1>
                <div class="ad">Advertisement</div>
                <p>Content</p>
                <footer>Footer</footer>
            </body>
        </html>
        """

        result = cleaner.clean_and_convert(html, "https://example.com")
        markdown = result["content_markdown"]

        assert "Navigation" not in markdown
        assert "Advertisement" not in markdown
        assert "Footer" not in markdown
        assert "Content" in markdown

    def test_content_selector_extraction(self):
        """Test extraction using CSS selector."""
        cleaner = ContentCleaner(content_selector="article")

        html = """
        <html>
            <body>
                <nav>Navigation</nav>
                <article>
                    <h1>Article Title</h1>
                    <p>Article content</p>
                </article>
                <footer>Footer</footer>
            </body>
        </html>
        """

        result = cleaner.clean_and_convert(html, "https://example.com")
        markdown = result["content_markdown"]

        assert "Article Title" in markdown
        assert "Article content" in markdown

    def test_content_selector_not_found(self):
        """Test behavior when content selector is not found."""
        cleaner = ContentCleaner(content_selector="article")

        html = """
        <html>
            <body>
                <h1>Title</h1>
                <p>Content</p>
            </body>
        </html>
        """

        # Should not crash, fallback to full content
        result = cleaner.clean_and_convert(html, "https://example.com")
        assert "Title" in result["content_markdown"]

    def test_is_valid_content(self):
        """Test content validation."""
        cleaner = ContentCleaner()

        # Valid content (longer than default min_length=100)
        long_content = "This is valid content with enough text. " * 5  # ~250 chars
        assert cleaner.is_valid_content(long_content) is True

        # Too short
        assert cleaner.is_valid_content("Short") is False

        # Empty
        assert cleaner.is_valid_content("") is False

        # Only whitespace
        assert cleaner.is_valid_content("   \n\n   ") is False

    def test_min_length_parameter(self):
        """Test custom minimum length."""
        cleaner = ContentCleaner()

        # With custom min_length
        assert cleaner.is_valid_content("a" * 50, min_length=50) is True
        assert cleaner.is_valid_content("a" * 50, min_length=100) is False

    def test_links_in_markdown(self):
        """Link text is preserved; the URL itself is not."""
        cleaner = ContentCleaner()

        html = """
        <html>
            <body>
                <a href="https://linked-site.test/page?q=1">Example Link</a>
            </body>
        </html>
        """

        result = cleaner.clean_and_convert(html, "https://example.com")
        markdown = result["content_markdown"]

        assert "Example Link" in markdown
        assert "linked-site.test" not in markdown

    def test_images_in_markdown(self):
        """Image alt text is preserved; the src URL is not."""
        cleaner = ContentCleaner()

        html = """
        <html>
            <body>
                <img src="image.jpg" alt="Test Image" />
            </body>
        </html>
        """

        result = cleaner.clean_and_convert(html, "https://example.com")
        markdown = result["content_markdown"]

        assert "image.jpg" not in markdown
        assert "Test Image" in markdown

    def test_code_blocks(self):
        """Test handling of code blocks."""
        cleaner = ContentCleaner()

        html = """
        <html>
            <body>
                <pre><code>def hello():
    print("Hello, World!")
</code></pre>
            </body>
        </html>
        """

        result = cleaner.clean_and_convert(html, "https://example.com")
        markdown = result["content_markdown"]

        assert "def hello():" in markdown

    def test_tables(self):
        """Test handling of HTML tables."""
        cleaner = ContentCleaner()

        html = """
        <html>
            <body>
                <table>
                    <tr><th>Header 1</th><th>Header 2</th></tr>
                    <tr><td>Data 1</td><td>Data 2</td></tr>
                </table>
            </body>
        </html>
        """

        result = cleaner.clean_and_convert(html, "https://example.com")
        markdown = result["content_markdown"]

        assert "Header 1" in markdown
        assert "Data 1" in markdown

    def test_image_and_link_urls_are_dropped_but_text_survives(self):
        """A signed CDN URL must not reach the chunk; its surrounding prose must."""
        cleaner = ContentCleaner()

        signed = (
            "https://downloads.intercomcdn.com/i/o/i31ha1vw/1916506582/"
            "15f517dbe751ff1f426f58f75e1d/Screenshot.png"
            "?expires=1787108400&signature=f6ffdfa27f7a4a4463637a36d7f3dae1"
        )
        html = f"""
        <html>
            <body>
                <p>The PDF export is downloaded as a ZIP file. Inside,
                   tickets are organised into folders by month.</p>
                <p><a href="{signed}"><img src="{signed}" alt="Print Settings"></a></p>
                <p>See <a href="https://help.example.com/en/articles/123">this guide</a>.</p>
            </body>
        </html>
        """

        content = cleaner.clean_and_convert(html, "https://example.com")[
            "content_markdown"
        ]

        assert "intercomcdn.com" not in content
        assert "signature=" not in content
        assert "help.example.com" not in content
        assert "tickets are organised into folders by month" in content
        assert "this guide" in content

    def test_url_shaped_alt_text_is_stripped(self):
        """images_to_alt keeps alt text -- but not when the alt is itself a URL."""
        cleaner = ContentCleaner()

        html = """
        <html>
            <body>
                <p>Before.</p>
                <img src="https://cdn.example.test/x.png?sig=abc"
                     alt="https://tracking.example.test/pixel?id=123">
                <p>After.</p>
            </body>
        </html>
        """

        content = cleaner.clean_and_convert(html, "https://example.com")[
            "content_markdown"
        ]

        assert "tracking.example.test" not in content
        assert "Before." in content and "After." in content

    def test_autolink_text_is_stripped(self):
        """ignore_links drops the href, so a URL used as its own anchor text needs stripping."""
        cleaner = ContentCleaner()

        html = """
        <html>
            <body>
                <p>See <a href="https://x.example.test/a?b=1">https://x.example.test/a?b=1</a>
                   and <a href="/rel">www.example.test/path</a> for details.</p>
            </body>
        </html>
        """

        content = cleaner.clean_and_convert(html, "https://example.com")[
            "content_markdown"
        ]

        assert "x.example.test" not in content
        assert "www.example.test" not in content
        assert "See and for details." in content

    def test_adjacent_inline_elements_do_not_fuse(self):
        """Removing link punctuation must not glue neighbouring anchors into one token."""
        cleaner = ContentCleaner()

        html = """
        <html>
            <body>
                <nav><a href="/a">One</a><a href="/b">Two</a><a href="/c">Three</a></nav>
                <p><img src="/a.png" alt="First"><img src="/b.png" alt="Second"></p>
            </body>
        </html>
        """

        content = cleaner.clean_and_convert(html, "https://example.com")[
            "content_markdown"
        ]

        assert "OneTwo" not in content
        assert "FirstSecond" not in content
        assert "One Two Three" in content
        assert "First Second" in content

    def test_prose_spacing_and_punctuation_are_preserved(self):
        """URL stripping must not leave gaps or orphan punctuation in the prose."""
        cleaner = ContentCleaner()

        html = """
        <html>
            <body>
                <p>Read <a href="/g">the guide</a>, then <a href="/h">the FAQ</a>.</p>
                <p>The export is a <b>ZIP file</b>. Inside, tickets sit in <i>folders</i> by month.</p>
                <p>Visit <a href="/x">https://x.example.test/a</a>, then stop.</p>
            </body>
        </html>
        """

        content = cleaner.clean_and_convert(html, "https://example.com")[
            "content_markdown"
        ]

        assert "Read the guide, then the FAQ." in content
        assert "Inside, tickets sit in _folders_ by month." in content
        assert "Visit, then stop." in content

    def test_image_without_alt_yields_nothing(self):
        """A bare image contributes no text and must not leak its src."""
        cleaner = ContentCleaner()

        html = """
        <html>
            <body>
                <p>Before.</p>
                <img src="https://cdn.example.test/x.png?sig=abc">
                <img src="https://cdn.example.test/y.png?sig=def" alt="">
                <p>After.</p>
            </body>
        </html>
        """

        content = cleaner.clean_and_convert(html, "https://example.com")[
            "content_markdown"
        ]

        assert "cdn.example.test" not in content
        assert "Before." in content and "After." in content
