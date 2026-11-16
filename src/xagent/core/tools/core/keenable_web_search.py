"""
Keenable Web Search Tool
Standalone web search functionality using the Keenable Search API.

The API key is optional:

* With a key: ``POST /v1/search`` with an ``X-API-Key`` header.
* Without one: ``POST /v1/search/public``, which Keenable rate limits per IP
  address (1,000 requests per hour, 10 per second) and logs together with the
  query and IP address. See https://docs.keenable.ai/rate-limits.

Both paths take the query in the JSON body, never in the URL, and require an
``X-Keenable-Title`` header naming the application.

Privacy: the search query is never logged, and no error message is built from
the text of an ``httpx`` exception.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

from .media_usage import record_web_search_usage
from .web_content import WebContentFetcher, get_proxy_url
from .web_search_guard import run_guarded_search

logger = logging.getLogger(__name__)

KEENABLE_SEARCH_URL = "https://api.keenable.ai/v1/search"
KEENABLE_PUBLIC_SEARCH_URL = "https://api.keenable.ai/v1/search/public"
KEENABLE_APP_TITLE = "xagent"

# The API accepts 1 to 50 for ``max_results``.
MAX_RESULTS = 50
SNIPPET_MAX_CHARS = 500


class KeenableWebSearchCore:
    """Pure Keenable web search tool without framework dependencies."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key

    async def search(
        self,
        query: str,
        max_results: int = 5,
        include_content: bool = False,
    ) -> List[Dict[str, str]]:
        """
        Search the web using the Keenable Search API.

        Args:
            query: The search query string
            max_results: Number of results to return (max 50)
            include_content: Whether to include page content for each result

        Returns:
            List of search results with title, link, snippet and optionally content
        """
        return await run_guarded_search(
            "keenable",
            query,
            lambda redacted: self._search_redacted(
                redacted, max_results, include_content
            ),
            len,
        )

    async def _search_redacted(
        self,
        query: str,
        max_results: int = 5,
        include_content: bool = False,
    ) -> List[Dict[str, str]]:
        """Run the search for the query that is actually sent to Keenable."""
        api_key = self._api_key or os.getenv("KEENABLE_API_KEY")
        max_results = min(max(1, max_results), MAX_RESULTS)
        logger.info(
            "Keenable web search keyed=%s max_results=%s include_content=%s",
            bool(api_key),
            max_results,
            include_content,
        )

        headers = {
            "Accept": "application/json",
            "X-Keenable-Title": KEENABLE_APP_TITLE,
        }
        if api_key:
            headers["X-API-Key"] = api_key
        url = KEENABLE_SEARCH_URL if api_key else KEENABLE_PUBLIC_SEARCH_URL

        proxy_url = get_proxy_url()
        client_kwargs: Dict[str, Any] = {}
        if proxy_url:
            # Do not log the URL: it can carry proxy credentials.
            logger.info("Using proxy for Keenable web search")
            client_kwargs["proxy"] = proxy_url

        try:
            async with httpx.AsyncClient(**client_kwargs) as client:
                response = await client.post(
                    url,
                    json={"query": query, "max_results": max_results},
                    headers=headers,
                    timeout=30,
                )
        except httpx.RequestError as e:
            # type(e).__name__ only: str(e) can contain request details.
            logger.error("Network error during Keenable search: %s", type(e).__name__)
            raise ValueError(
                f"Network error during Keenable search ({type(e).__name__})"
            ) from None

        self._raise_for_status(response.status_code, bool(api_key))
        record_web_search_usage("keenable")

        try:
            data = response.json()
        except ValueError:
            logger.error("Keenable search returned a non-JSON body")
            raise ValueError(
                "Keenable search returned an invalid response body"
            ) from None

        results = await self._normalize(data, include_content, proxy_url)
        logger.info("Keenable search returned %d results", len(results))
        return results

    @staticmethod
    def _raise_for_status(status: int, keyed: bool) -> None:
        if status in {401, 403}:
            raise ValueError(
                f"Keenable Search API auth failed ({status}). "
                "Please check KEENABLE_API_KEY."
            )
        if status == 429:
            hint = (
                "Please retry later."
                if keyed
                else "The keyless tier allows 1,000 requests per hour per IP; "
                "set KEENABLE_API_KEY to remove the hourly cap."
            )
            raise ValueError(f"Keenable Search API rate limit exceeded (429). {hint}")
        if status >= 400:
            logger.error("Keenable Search API HTTP error: %s", status)
            raise ValueError(f"Keenable Search API HTTP error ({status})")

    @staticmethod
    async def _normalize(
        data: Dict[str, Any],
        include_content: bool,
        proxy_url: Optional[str],
    ) -> List[Dict[str, str]]:
        """Map ``results[]`` to xagent's standard result format."""
        results: List[Dict[str, str]] = []
        items = data.get("results") if isinstance(data, dict) else None
        for item in items or []:
            if not isinstance(item, dict) or not item.get("url"):
                continue
            snippet = item.get("snippet") or item.get("description") or ""
            result: Dict[str, str] = {
                "title": item.get("title") or "",
                "link": item["url"],
                "snippet": snippet[:SNIPPET_MAX_CHARS],
            }
            if include_content:
                result["content"] = await WebContentFetcher(
                    proxy_url=proxy_url
                ).fetch_text(result["link"])
            results.append(result)
        return results
