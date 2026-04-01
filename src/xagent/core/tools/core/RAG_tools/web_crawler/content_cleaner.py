"""Content cleaning and markdown conversion for web crawler."""

import logging
from typing import Optional

import html2text
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class ContentCleaner:
    """Clean and convert HTML content to markdown."""

    def __init__(
        self,
        content_selector: Optional[str] = None,
        remove_selectors: Optional[list[str]] = None,
    ):
        """Initialize content cleaner.

        Args:
            content_selector: CSS selector for main content area
            remove_selectors: CSS selectors for elements to remove
        """
        self.content_selector = content_selector
        self.remove_selectors = remove_selectors or []

    def clean_and_convert(self, html: str, url: str) -> dict:
        """Clean HTML and convert to markdown.

        Args:
            html: Raw HTML content
            url: Page URL (for metadata)

        Returns:
            Dict with:
                - title: Page title
                - content_markdown: Markdown content
                - content_length: Length of markdown
        """
        try:
            soup = BeautifulSoup(html, "html.parser")

            # Extract title
            title = self._extract_title(soup)

            # Extract meta descriptions and keywords before removing unwanted elements
            meta_content = self._extract_meta_content(soup)

            # Remove unwanted elements
            self._remove_unwanted(soup)

            # Extract main content if selector specified
            if self.content_selector:
                content_element = soup.select_one(self.content_selector)
                if content_element:
                    # Replace soup with selected element
                    new_soup = BeautifulSoup(str(content_element), "html.parser")
                    soup = new_soup
                else:
                    logger.warning(
                        f"Content selector '{self.content_selector}' not found in {url}"
                    )

            # Convert to markdown
            markdown = self._to_markdown(soup)

            # Enhance markdown with meta content if it exists
            if meta_content:
                if markdown:
                    markdown = f"{meta_content}\n\n{markdown}"
                else:
                    markdown = meta_content

            return {
                "title": title,
                "content_markdown": markdown,
                "content_length": len(markdown),
            }

        except Exception as e:
            logger.error(f"Failed to clean content from {url}: {e}")
            return {
                "title": None,
                "content_markdown": "",
                "content_length": 0,
            }

    def _extract_meta_content(self, soup: BeautifulSoup) -> str:
        """Extract description and keywords from meta tags.

        Args:
            soup: BeautifulSoup object

        Returns:
            Formatted string of meta content or empty string
        """
        from bs4 import Tag

        meta_parts = []

        # Extract description
        desc_tag = soup.find("meta", attrs={"name": "description"})
        if not desc_tag:
            desc_tag = soup.find("meta", property="og:description")

        if isinstance(desc_tag, Tag):
            content = desc_tag.get("content")
            if isinstance(content, str) and content.strip():
                meta_parts.append(content.strip())

        # Extract keywords
        keywords_tag = soup.find("meta", attrs={"name": "keywords"})
        if isinstance(keywords_tag, Tag):
            content = keywords_tag.get("content")
            if isinstance(content, str) and content.strip():
                meta_parts.append(f"Keywords: {content.strip()}")

        return "\n\n".join(meta_parts)

    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract page title from HTML.

        Args:
            soup: BeautifulSoup object

        Returns:
            Title string or None
        """
        # Try <title> tag first
        from bs4 import Tag

        title_tag = soup.find("title")
        if isinstance(title_tag, Tag) and title_tag.string:
            return str(title_tag.string).strip()

        # Try <h1> tag
        h1_tag = soup.find("h1")
        if isinstance(h1_tag, Tag) and h1_tag.string:
            return str(h1_tag.string).strip()

        # Try meta title
        meta_title = soup.find("meta", property="og:title")
        if isinstance(meta_title, Tag):
            content = meta_title.get("content")
            if isinstance(content, str):
                return content.strip()

        return None

    def _remove_unwanted(self, soup: BeautifulSoup) -> None:
        """Remove unwanted elements from soup.

        Args:
            soup: BeautifulSoup object (modified in place)
        """
        # Default elements to remove
        default_selectors = [
            "script",
            "style",
            "noscript",
            "iframe",
            "svg",
        ]

        # Combine default with user-specified selectors
        all_selectors = default_selectors + self.remove_selectors

        for selector in all_selectors:
            for element in soup.select(selector):
                element.decompose()

        logger.debug(
            f"Removed unwanted elements matching {len(all_selectors)} selectors"
        )

    def _to_markdown(self, soup: BeautifulSoup) -> str:
        """Convert BeautifulSoup object to markdown.

        Args:
            soup: BeautifulSoup object

        Returns:
            Markdown string
        """
        h2t = html2text.HTML2Text()
        h2t.body_width = 0  # No line wrapping
        h2t.ignore_images = False
        h2t.ignore_emphasis = False
        h2t.ignore_links = False
        h2t.ignore_tables = False

        markdown = h2t.handle(str(soup))
        return markdown.strip()

    def is_valid_content(self, content: str, min_length: int = 100) -> bool:
        """Check if content is valid and substantial enough.

        Args:
            content: Content to check
            min_length: Minimum content length

        Returns:
            True if valid, False otherwise
        """
        if not content or len(content) < min_length:
            return False

        # Check if content has actual text (not just empty lines/spaces)
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        return len(lines) > 0
