"""
Brave Web Search Tool
Standalone web search functionality using the Brave Search API.

Two endpoints are supported, chosen by ``XAGENT_BRAVE_SEARCH_ENDPOINT``:

* ``web``: ``GET /res/v1/web/search`` (titles, links and descriptions).
* ``llm_context``: ``POST /res/v1/llm/context`` (extracted page snippets).
  The query travels in the JSON body, not in the URL.

Privacy: the search query is never logged, and no error message is built from
the text of an ``httpx`` exception, because that text contains the request URL
(which carries the query on the ``web`` endpoint).
"""

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

from ....config import get_brave_search_endpoint
from .web_content import WebContentFetcher

logger = logging.getLogger(__name__)

BRAVE_WEB_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
BRAVE_LLM_CONTEXT_URL = "https://api.search.brave.com/res/v1/llm/context"

# Brave's documented limits for the ``q`` parameter.
MAX_QUERY_CHARS = 600
MAX_QUERY_WORDS = 75

MAX_RESULTS = 20
SNIPPET_MAX_CHARS = 500
CONTENT_MAX_CHARS = 8192
LLM_CONTEXT_MAX_TOKENS = 8192


class _DropBraveRequestLogs(logging.Filter):
    """Drop httpx's per-request INFO line for Brave calls.

    httpx logs ``HTTP Request: GET <full url>``; on the ``web`` endpoint the URL
    carries the search query.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            return "api.search.brave.com" not in record.getMessage()
        except Exception:
            return True


_httpx_filter_installed = False


def _install_httpx_log_filter() -> None:
    global _httpx_filter_installed
    if not _httpx_filter_installed:
        logging.getLogger("httpx").addFilter(_DropBraveRequestLogs())
        _httpx_filter_installed = True


class BraveWebSearchCore:
    """Pure Brave web search tool without framework dependencies."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key
        _install_httpx_log_filter()

    async def search(
        self,
        query: str,
        max_results: int = 5,
        include_content: bool = False,
    ) -> List[Dict[str, str]]:
        """
        Search the web using the Brave Search API.

        Args:
            query: The search query string (max 600 characters and 75 words)
            max_results: Number of results to return (max 20)
            include_content: Whether to include page content for each result

        Returns:
            List of search results with title, link, snippet and optionally content
        """
        self._validate_query(query)

        api_key = self._api_key or os.getenv("BRAVE_SEARCH_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing required environment variable BRAVE_SEARCH_API_KEY. "
                "Get one at https://brave.com/search/api/"
            )

        max_results = min(max(1, max_results), MAX_RESULTS)
        endpoint = get_brave_search_endpoint()
        logger.info(
            "Brave web search endpoint=%s max_results=%s include_content=%s",
            endpoint,
            max_results,
            include_content,
        )

        proxy_url = self._get_proxy_url()
        if proxy_url:
            # Do not log the URL: it can carry proxy credentials.
            logger.info("Using proxy for Brave web search")

        headers = {
            "X-Subscription-Token": api_key,
            "Accept": "application/json",
        }
        client_kwargs: Dict[str, Any] = {}
        if proxy_url:
            client_kwargs["proxy"] = proxy_url

        try:
            async with httpx.AsyncClient(**client_kwargs) as client:
                if endpoint == "llm_context":
                    response = await client.post(
                        BRAVE_LLM_CONTEXT_URL,
                        json={
                            "q": query,
                            "count": max_results,
                            "maximum_number_of_tokens": LLM_CONTEXT_MAX_TOKENS,
                        },
                        headers=headers,
                        timeout=30,
                    )
                else:
                    response = await client.get(
                        BRAVE_WEB_SEARCH_URL,
                        params={
                            "q": query,
                            "count": max_results,
                            "text_decorations": "false",
                        },
                        headers=headers,
                        timeout=30,
                    )
        except httpx.RequestError as e:
            # type(e).__name__ only: str(e) can contain the request URL.
            logger.error("Network error during Brave search: %s", type(e).__name__)
            raise ValueError(
                f"Network error during Brave search ({type(e).__name__})"
            ) from None

        self._raise_for_status(response.status_code)

        try:
            data = response.json()
        except ValueError:
            logger.error("Brave search returned a non-JSON body")
            raise ValueError("Brave search returned an invalid response body") from None

        if endpoint == "llm_context":
            results = self._normalize_llm_context(data, include_content)
        else:
            results = await self._normalize_web(data, include_content, proxy_url)
        logger.info("Brave search returned %d results", len(results))
        return results

    @staticmethod
    def _validate_query(query: str) -> None:
        if len(query) > MAX_QUERY_CHARS:
            raise ValueError(
                f"Brave search query is too long: the limit is "
                f"{MAX_QUERY_CHARS} characters."
            )
        if len(query.split()) > MAX_QUERY_WORDS:
            raise ValueError(
                f"Brave search query has too many words: the limit is "
                f"{MAX_QUERY_WORDS} words."
            )

    @staticmethod
    def _raise_for_status(status: int) -> None:
        if status in {401, 403}:
            raise ValueError(
                f"Brave Search API auth failed ({status}). "
                "Please check BRAVE_SEARCH_API_KEY."
            )
        if status == 429:
            raise ValueError(
                "Brave Search API rate limit or quota exceeded (429). "
                "Please retry later."
            )
        if status >= 400:
            logger.error("Brave Search API HTTP error: %s", status)
            raise ValueError(f"Brave Search API HTTP error ({status})")

    @staticmethod
    def _normalize_llm_context(
        data: Dict[str, Any], include_content: bool
    ) -> List[Dict[str, str]]:
        """Map ``grounding.generic[]`` to xagent's standard result format."""
        results: List[Dict[str, str]] = []
        grounding = data.get("grounding") if isinstance(data, dict) else None
        generic = grounding.get("generic") if isinstance(grounding, dict) else None
        for item in generic or []:
            snippets = [s for s in (item.get("snippets") or []) if isinstance(s, str)]
            result: Dict[str, str] = {
                "title": item.get("title") or "",
                "link": item.get("url") or "",
                "snippet": (snippets[0] if snippets else "")[:SNIPPET_MAX_CHARS],
            }
            if include_content:
                result["content"] = "\n\n".join(snippets)[:CONTENT_MAX_CHARS]
            results.append(result)
        return results

    async def _normalize_web(
        self,
        data: Dict[str, Any],
        include_content: bool,
        proxy_url: Optional[str],
    ) -> List[Dict[str, str]]:
        """Map ``web.results[]`` to xagent's standard result format."""
        results: List[Dict[str, str]] = []
        web = data.get("web") if isinstance(data, dict) else None
        items = web.get("results") if isinstance(web, dict) else None
        for item in items or []:
            result: Dict[str, str] = {
                "title": item.get("title") or "",
                "link": item.get("url") or "",
                "snippet": item.get("description") or "",
            }
            if include_content:
                result["content"] = await WebContentFetcher(
                    proxy_url=proxy_url
                ).fetch_text(result["link"])
            results.append(result)
        return results

    @staticmethod
    def _get_proxy_url() -> Optional[str]:
        """Get proxy URL from environment variables."""
        https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
        return https_proxy or http_proxy
