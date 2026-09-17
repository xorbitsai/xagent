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

from ....config import get_trusted_egress_proxy_enabled
from ...utils.security import PrivateNetworkHostError, fetch_public_http_bytes

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
DEFAULT_MAX_CONTENT_BYTES = 10 * 1024 * 1024
MAX_DISCOVERED_ASSETS = 50
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
class WebAssetReference:
    """One static asset reference discovered from an official webpage."""

    url: str
    kind: str
    name: str = ""
    alt: str = ""
    source: str = "html"

    def as_dict(self) -> dict[str, str]:
        return {
            "url": self.url,
            "kind": self.kind,
            "name": self.name,
            "alt": self.alt,
            "source": self.source,
        }


@dataclass(frozen=True)
class WebContentFetchResult:
    """Structured result for fetching and extracting one webpage."""

    url: str
    content: str
    title: str = ""
    status_code: int | None = None
    content_type: str = ""
    error: str | None = None
    assets: tuple[WebAssetReference, ...] = ()

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
            "assets": [asset.as_dict() for asset in self.assets],
        }


def get_proxy_url() -> str | None:
    """Get proxy URL from environment variables."""

    https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
    http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
    return https_proxy or http_proxy


def get_trusted_proxy_url() -> str | None:
    """Return the ambient proxy URL, requiring an explicit trust opt-in.

    Fetches that go through ``fetch_public_http_bytes(..., via_proxy=True)``
    skip client-side IP pinning, since the proxy performs its own DNS
    resolution of the target host. That reopens the DNS-rebinding TOCTOU
    window this module's SSRF guarding is meant to close, unless the proxy
    itself is trusted to enforce private-range egress policy. Raise instead
    of silently trusting every ambient ``HTTP(S)_PROXY``.
    """

    proxy_url = get_proxy_url()
    if proxy_url and not get_trusted_egress_proxy_enabled():
        raise PrivateNetworkHostError(
            "An HTTP(S) proxy is configured but not marked as trusted for "
            "public-network egress; set XAGENT_TRUSTED_EGRESS_PROXY=1 only "
            "if the proxy itself enforces private-range egress policy."
        )
    return proxy_url


def build_isolated_httpx_client_kwargs(proxy_url: str | None) -> dict[str, Any]:
    """Return ``httpx.AsyncClient`` kwargs that never trust ambient proxy config.

    ``httpx.AsyncClient`` defaults to ``trust_env=True``, which (via
    ``httpx._utils.get_environment_proxies()``) falls back to
    ``urllib.request.getproxies()`` once no ``HTTP(S)_PROXY``/``ALL_PROXY``
    env var is set -- and that function itself falls back to the OS's own
    proxy configuration (``getproxies_macosx_sysconf()``/
    ``getproxies_registry()`` on macOS/Windows, confirmed against the
    installed httpx source). That reopens the DNS-rebinding TOCTOU window
    ``get_trusted_proxy_url()`` exists to close, even when it returns
    ``None`` because no proxy is explicitly trusted: the caller of this
    module would still silently route through whatever proxy the OS itself
    is configured with. ``trust_env=False`` disables that whole lookup, but
    an explicit ``proxy=`` kwarg (passed by the caller here, never derived
    from the client) is honored regardless of ``trust_env``.

    ``trust_env=False`` also stops httpx from honoring ``SSL_CERT_FILE``/
    ``SSL_CERT_DIR`` for the default CA bundle (both gated behind the same
    ``trust_env`` flag in ``httpx.create_ssl_context()``), which would break
    TLS verification for a host behind a private/internal CA. Build that SSL
    context explicitly with ``trust_env=True`` and pass it as ``verify=``:
    once ``verify`` is already a concrete ``ssl.SSLContext``,
    ``create_ssl_context()`` returns it unchanged, so the client's own
    ``trust_env=False`` no longer affects CA lookup at all.
    """

    client_kwargs: dict[str, Any] = {
        "trust_env": False,
        "verify": httpx.create_ssl_context(trust_env=True),
    }
    if proxy_url:
        client_kwargs["proxy"] = proxy_url
    return client_kwargs


class WebContentFetcher:
    """Fetch webpages and convert readable HTML content to markdown."""

    def __init__(
        self,
        proxy_url: str | None = None,
        max_content_bytes: int = DEFAULT_MAX_CONTENT_BYTES,
    ) -> None:
        self._proxy_url = proxy_url
        self._max_content_bytes = max_content_bytes

    async def fetch(
        self,
        url: str,
        *,
        include_assets: bool = False,
        asset_query: str | None = None,
    ) -> WebContentFetchResult:
        logger.info("Fetching webpage content from: %s", url)

        headers = {"User-Agent": DEFAULT_USER_AGENT}
        try:
            if self._proxy_url:
                logger.info("Using proxy for webpage fetch: %s", self._proxy_url)
            client_kwargs = build_isolated_httpx_client_kwargs(self._proxy_url)

            async with httpx.AsyncClient(**client_kwargs) as client:
                response = await fetch_public_http_bytes(
                    client,
                    url,
                    headers=headers,
                    timeout=10,
                    max_content_bytes=self._max_content_bytes,
                    via_proxy=bool(self._proxy_url),
                )
                content_type = response.content_type
                final_url = response.url
                content = response.content

                if not self._is_html_content(content_type):
                    if self._is_plain_text_content(content_type):
                        decoded = self._decode_text_response(response.encoding, content)
                        return WebContentFetchResult(
                            url=final_url,
                            content=decoded,
                            status_code=response.status_code,
                            content_type=content_type,
                            assets=(),
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
                assets: tuple[WebAssetReference, ...] = ()
                if include_assets:
                    discovered_assets = self._extract_html_assets(soup, final_url)
                    assets = self._filter_and_deduplicate_assets(
                        discovered_assets,
                        asset_query=asset_query,
                    )
                markdown = self._soup_to_markdown(soup, final_url)

                return WebContentFetchResult(
                    url=final_url,
                    title=title,
                    content=markdown,
                    status_code=response.status_code,
                    content_type=content_type,
                    assets=assets,
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

    @staticmethod
    def _extract_html_assets(
        soup: BeautifulSoup,
        base_url: str,
    ) -> list[WebAssetReference]:
        assets: list[WebAssetReference] = []

        def add(
            raw_url: str | None,
            *,
            kind: str,
            name: str = "",
            alt: str = "",
        ) -> None:
            if not raw_url:
                return
            normalized = str(raw_url).strip()
            if not normalized or normalized.startswith(("data:", "javascript:")):
                return
            assets.append(
                WebAssetReference(
                    url=urljoin(base_url, normalized),
                    kind=kind,
                    name=name,
                    alt=alt,
                )
            )

        for image in soup.find_all("img"):
            image_class = image.get("class")
            add(
                image.get("src") or image.get("data-src"),
                kind="image",
                name=str(
                    image.get("id")
                    or (
                        " ".join(image_class)
                        if isinstance(image_class, list)
                        else (image_class or "")
                    )
                ),
                alt=str(image.get("alt") or ""),
            )

        for source in soup.find_all("source"):
            srcset = str(source.get("srcset") or source.get("data-srcset") or "")
            for candidate in srcset.split(","):
                add(candidate.strip().split(" ", 1)[0], kind="image")

        for link in soup.find_all("link"):
            rel_values = {str(value).lower() for value in (link.get("rel") or [])}
            kind = "link"
            if "manifest" in rel_values:
                kind = "manifest"
            elif rel_values & {"icon", "shortcut", "apple-touch-icon"}:
                kind = "icon"
            elif "stylesheet" in rel_values:
                kind = "stylesheet"
            elif link.get("as") == "image":
                kind = "image"
            add(link.get("href"), kind=kind, name=" ".join(sorted(rel_values)))

        for script in soup.find_all("script"):
            add(script.get("src"), kind="script")

        for meta in soup.find_all("meta"):
            property_name = str(meta.get("property") or meta.get("name") or "").lower()
            if property_name in {"og:image", "twitter:image", "twitter:image:src"}:
                add(meta.get("content"), kind="image", name=property_name)

        return assets

    @staticmethod
    def _filter_and_deduplicate_assets(
        assets: list[WebAssetReference],
        *,
        asset_query: str | None,
    ) -> tuple[WebAssetReference, ...]:
        query = str(asset_query or "").strip().lower()
        result: list[WebAssetReference] = []
        seen: set[str] = set()
        for asset in assets:
            if asset.url in seen:
                continue
            if query and asset.kind != "manifest":
                haystack = f"{asset.url} {asset.name} {asset.alt}".lower()
                if query not in haystack:
                    continue
            seen.add(asset.url)
            result.append(asset)
            if len(result) >= MAX_DISCOVERED_ASSETS:
                break
        return tuple(result)

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
    def _decode_text_response(encoding: str | None, content: bytes) -> str:
        try:
            return content.decode(encoding or "utf-8", errors="replace")
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


async def fetch_web_content(
    url: str,
    *,
    include_assets: bool = False,
    asset_query: str | None = None,
) -> WebContentFetchResult:
    """Fetch a webpage using the default proxy configuration."""

    try:
        proxy_url = get_trusted_proxy_url()
    except PrivateNetworkHostError as exc:
        logger.error("Webpage fetch failed: %s", exc)
        return WebContentFetchResult(url=url, content="", error=str(exc))

    return await WebContentFetcher(proxy_url=proxy_url).fetch(
        url,
        include_assets=include_assets,
        asset_query=asset_query,
    )


async def fetch_web_content_text(url: str) -> str:
    """Fetch a webpage and return only extracted text or a legacy error string."""

    return (await fetch_web_content(url)).as_search_content()
