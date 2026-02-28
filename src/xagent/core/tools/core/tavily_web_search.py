"""
Tavily Web Search Tool
Standalone web search functionality using Tavily Search API.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class TavilyWebSearchCore:
    """Pure Tavily web search tool without framework dependencies."""

    async def search(
        self,
        query: str,
        max_results: int = 5,
        include_content: bool = True,
        search_depth: str = "basic",
    ) -> List[Dict[str, str]]:
        """
        Search the web using Tavily Search API.

        Args:
            query: The search query string
            max_results: Number of results to return (max 20)
            include_content: Whether to include raw page content
            search_depth: "basic" for fast results or "advanced" for deeper search

        Returns:
            List of search results with title, link, snippet and optionally content
        """
        logger.info(
            "🔍 Tavily web search query='%s' max_results=%s depth=%s",
            query,
            max_results,
            search_depth,
        )

        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing required environment variable TAVILY_API_KEY. "
                "Get one at https://tavily.com"
            )

        max_results = min(max(1, max_results), 20)

        payload: Dict[str, Any] = {
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_raw_content": include_content,
        }

        proxy_url = self._get_proxy_url()
        if proxy_url:
            logger.info("🌐 Using proxy: %s", proxy_url)

        try:
            client_kwargs: Dict[str, Any] = {}
            if proxy_url:
                client_kwargs["proxy"] = proxy_url

            async with httpx.AsyncClient(**client_kwargs) as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    json=payload,
                    timeout=30,
                )

                if response.status_code in {401, 403}:
                    raise ValueError(
                        f"Tavily API auth failed ({response.status_code}). "
                        "Please check TAVILY_API_KEY."
                    )

                response.raise_for_status()
                data = response.json()

                logger.info("✅ Tavily API request successful")
                return self._normalize_results(data, include_content)

        except httpx.RequestError as e:
            logger.error("❌ Network error during Tavily search: %s", str(e))
            raise ValueError(f"Network error during Tavily search: {str(e)}") from e
        except httpx.HTTPStatusError as e:
            logger.error("❌ Tavily API HTTP error: %s", str(e))
            raise ValueError(f"Tavily API HTTP error: {str(e)}") from e
        except ValueError:
            raise
        except Exception as e:
            logger.error("❌ Unexpected error during Tavily search: %s", str(e))
            raise ValueError(f"Unexpected error during Tavily search: {str(e)}") from e

    @staticmethod
    def _normalize_results(
        data: Dict[str, Any], include_content: bool
    ) -> List[Dict[str, str]]:
        """Normalize Tavily search results into xagent's standard format."""
        results: List[Dict[str, str]] = []
        for item in data.get("results", []):
            result: Dict[str, str] = {
                "title": item.get("title", ""),
                "link": item.get("url", ""),
                "snippet": item.get("content", ""),
            }
            if include_content:
                raw = item.get("raw_content") or item.get("content", "")
                result["content"] = raw[:8192] if raw else ""
            results.append(result)

        logger.info("🎯 Tavily search returned %d results", len(results))
        return results

    @staticmethod
    def _get_proxy_url() -> Optional[str]:
        """Get proxy URL from environment variables."""
        https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
        return https_proxy or http_proxy
