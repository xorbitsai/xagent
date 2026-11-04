"""Link extraction from HTML for web crawler."""

import logging
from typing import Set
from urllib.parse import urljoin

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class LinkExtractor:
    """Extract links from HTML pages."""

    def __init__(self, base_url: str):
        """Initialize link extractor.

        Args:
            base_url: Base URL for resolving relative links
        """
        self.base_url = base_url

    def extract_links(self, html: str, current_url: str) -> Set[str]:
        """Extract all unique links from HTML.

        Args:
            html: HTML content
            current_url: URL of the current page (for resolving relative links)

        Returns:
            Set of unique absolute URLs
        """
        links: Set[str] = set()

        try:
            soup = BeautifulSoup(html, "html.parser")

            # Find all <a> tags with href attribute
            for anchor in soup.find_all("a", href=True):
                href = anchor["href"].strip()

                # Skip empty links, anchors, javascript/mailto/tel
                if (
                    not href
                    or href.startswith("#")
                    or href.startswith("javascript:")
                    or href.startswith("mailto:")
                    or href.startswith("tel:")
                ):
                    continue

                # Convert to absolute URL
                absolute_url = self._make_absolute(href, current_url)
                if absolute_url:
                    links.add(absolute_url)

            logger.debug("Extracted %s links from %s", len(links), current_url)
            return links

        except Exception as e:
            logger.error("Failed to extract links from %s: %s", current_url, e)
            return links

    def _make_absolute(self, url: str, base_url: str) -> str | None:
        """Convert relative URL to absolute URL.

        Args:
            url: URL (can be relative or absolute)
            base_url: Base URL for resolution

        Returns:
            Absolute URL, or None if invalid
        """
        try:
            absolute = urljoin(base_url, url)
            return absolute
        except Exception as e:
            logger.warning("Failed to make URL absolute: %s - %s", url, e)
            return None
