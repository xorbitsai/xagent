"""Shared webpage fetching and markdown extraction helpers."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

import html2text
import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
DEFAULT_MAX_CONTENT_BYTES = 10 * 1024 * 1024
HTML_CONTENT_TYPES = frozenset(
    {
        "",
        "text/html",
        "application/xhtml+xml",
        "application/xml",
        "text/xml",
    }
)
PLAIN_TEXT_CONTENT_TYPES = frozenset(
    {
        "application/json",
        "application/javascript",
        "application/x-javascript",
        "application/ld+json",
    }
)


@dataclass(frozen=True)
class WebContentFetchResult:
    """Structured result for fetching and extracting one webpage."""

    url: str
    content: str
    title: str = ""
    status_code: int | None = None
    content_type: str = ""
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None

    def as_search_content(self) -> str:
        """Return the legacy string form used by search result content fields."""

        if self.success:
            return self.content
        return f"Error fetching content: {self.error}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "status_code": self.status_code,
            "content_type": self.content_type,
            "error": self.error,
        }


def get_proxy_url() -> str | None:
    """Get proxy URL from environment variables."""

    https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
    http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
    return https_proxy or http_proxy


class WebContentFetcher:
    """Fetch webpages and convert readable HTML content to markdown."""

    def __init__(
        self,
        proxy_url: str | None = None,
        max_content_bytes: int = DEFAULT_MAX_CONTENT_BYTES,
    ) -> None:
        self._proxy_url = proxy_url
        self._max_content_bytes = max_content_bytes

    async def fetch(self, url: str) -> WebContentFetchResult:
        logger.info("Fetching webpage content from: %s", url)

        headers = {"User-Agent": DEFAULT_USER_AGENT}
        try:
            client_kwargs: dict[str, Any] = {}
            if self._proxy_url:
                client_kwargs["proxy"] = self._proxy_url
                logger.info("Using proxy for webpage fetch: %s", self._proxy_url)

            async with httpx.AsyncClient(**client_kwargs) as client:
                async with client.stream(
                    "GET",
                    url,
                    headers=headers,
                    timeout=10,
                    follow_redirects=True,
                ) as response:
                    response.raise_for_status()

                    content_type = response.headers.get("content-type", "")
                    final_url = str(response.url)
                    error = self._validate_content_length(
                        response.headers.get("content-length")
                    )
                    if error:
                        return WebContentFetchResult(
                            url=final_url,
                            content="",
                            status_code=response.status_code,
                            content_type=content_type,
                            error=error,
                        )

                    content, error = await self._read_limited_response(response)
                    if error:
                        return WebContentFetchResult(
                            url=final_url,
                            content="",
                            status_code=response.status_code,
                            content_type=content_type,
                            error=error,
                        )

                    if not self._is_html_content(content_type):
                        if self._is_plain_text_content(content_type):
                            return WebContentFetchResult(
                                url=final_url,
                                content=self._decode_text_response(response, content),
                                status_code=response.status_code,
                                content_type=content_type,
                            )

                        return WebContentFetchResult(
                            url=final_url,
                            content="",
                            status_code=response.status_code,
                            content_type=content_type,
                            error=f"Unsupported non-text content type: {content_type}",
                        )

                    soup = BeautifulSoup(content, "html.parser")
                    title = self._extract_title(soup)
                    markdown = self._soup_to_markdown(soup, final_url)

                    return WebContentFetchResult(
                        url=final_url,
                        title=title,
                        content=markdown,
                        status_code=response.status_code,
                        content_type=content_type,
                    )

        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            reason = e.response.reason_phrase
            error = f"HTTP {status_code} error for {url}: {reason}"
            logger.error("Webpage fetch failed: %s", error)
            return WebContentFetchResult(
                url=url,
                content="",
                status_code=status_code,
                error=error,
            )
        except httpx.RequestError as e:
            error = f"Network error for {url}: {str(e)}"
            logger.error("Webpage fetch failed: %s", error)
            return WebContentFetchResult(url=url, content="", error=error)
        except Exception as e:
            error = f"Unexpected error for {url}: {str(e)}"
            logger.error("Webpage fetch failed: %s", error)
            return WebContentFetchResult(url=url, content="", error=error)

    async def fetch_text(self, url: str) -> str:
        """Fetch webpage content in the legacy string form."""

        return (await self.fetch(url)).as_search_content()

    def _validate_content_length(self, content_length: str | None) -> str | None:
        if not content_length:
            return None
        try:
            size = int(content_length)
        except ValueError:
            return None
        if size > self._max_content_bytes:
            return (
                f"Response body size exceeds maximum of {self._max_content_bytes} bytes"
            )
        return None

    async def _read_limited_response(
        self, response: httpx.Response
    ) -> tuple[bytes, str | None]:
        chunks: list[bytes] = []
        downloaded = 0
        async for chunk in response.aiter_bytes():
            downloaded += len(chunk)
            if downloaded > self._max_content_bytes:
                return (
                    b"",
                    "Response body size exceeds maximum of "
                    f"{self._max_content_bytes} bytes",
                )
            chunks.append(chunk)
        return b"".join(chunks), None

    @staticmethod
    def _content_media_type(content_type: str) -> str:
        return content_type.split(";", 1)[0].strip().lower()

    @classmethod
    def _is_html_content(cls, content_type: str) -> bool:
        media_type = cls._content_media_type(content_type)
        return media_type in HTML_CONTENT_TYPES or media_type.endswith("+xml")

    @classmethod
    def _is_plain_text_content(cls, content_type: str) -> bool:
        media_type = cls._content_media_type(content_type)
        return (
            media_type.startswith("text/")
            or media_type in PLAIN_TEXT_CONTENT_TYPES
            or media_type.endswith("+json")
        )

    @staticmethod
    def _decode_text_response(response: httpx.Response, content: bytes) -> str:
        try:
            return content.decode(response.encoding or "utf-8", errors="replace")
        except LookupError:
            return content.decode("utf-8", errors="replace")

    @staticmethod
    def _extract_title(soup: BeautifulSoup) -> str:
        title = soup.find("title")
        if title and title.get_text(strip=True):
            return title.get_text(" ", strip=True)
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(" ", strip=True)
        return ""

    @staticmethod
    def _soup_to_markdown(soup: BeautifulSoup, base_url: str) -> str:
        for element in soup(["script", "style", "noscript", "svg"]):
            element.decompose()

        for tag in soup.find_all("a"):
            if not hasattr(tag, "get") or not hasattr(tag, "__setitem__"):
                continue
            if tag.get("href"):
                tag["href"] = urljoin(base_url, tag["href"])

        converter = html2text.HTML2Text()
        converter.body_width = 0
        converter.ignore_images = True
        converter.ignore_emphasis = False
        converter.ignore_links = False
        converter.ignore_tables = False

        markdown = converter.handle(str(soup)).strip()
        return markdown


async def fetch_web_content(url: str) -> WebContentFetchResult:
    """Fetch a webpage using the default proxy configuration."""

    return await WebContentFetcher(proxy_url=get_proxy_url()).fetch(url)


async def fetch_web_content_text(url: str) -> str:
    """Fetch a webpage and return only extracted text or a legacy error string."""

    return (await fetch_web_content(url)).as_search_content()
