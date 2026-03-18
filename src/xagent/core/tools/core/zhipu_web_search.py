"""
Zhipu Web Search Tool
Standalone web search functionality using Zhipu Web Search API.
"""

import logging
import os
from typing import Any, Dict, List, Optional, cast

import httpx

logger = logging.getLogger(__name__)


class ZhipuWebSearchCore:
    """Pure Zhipu web search tool without framework dependencies."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self._api_key = api_key
        self._base_url = base_url

    async def search(
        self,
        query: str,
        search_engine: str = "search_pro_sogou",
        search_intent: bool = False,
        count: int = 10,
        search_domain_filter: Optional[str] = None,
        search_recency_filter: str = "noLimit",
        content_size: str = "medium",
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search the web using Zhipu Web Search API.

        Args:
            query: Search query string
            search_engine: Search engine code (e.g. search_std/search_pro)
            search_intent: Whether to return search intent analysis
            count: Number of results to return (1-50)
            search_domain_filter: Restrict results to a domain
            search_recency_filter: Recency filter (e.g. noLimit)
            content_size: Summary length (low/medium/high)
            request_id: Optional request id
            user_id: Optional user id

        Returns:
            Raw response JSON from Zhipu Web Search API
        """
        logger.info(
            "🔍 Zhipu web search query='%s' engine=%s count=%s intent=%s",
            query,
            search_engine,
            count,
            search_intent,
        )

        api_key = (
            self._api_key or os.getenv("ZHIPU_API_KEY") or os.getenv("BIGMODEL_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "Missing required environment variable. Please set ZHIPU_API_KEY."
            )

        count = min(max(1, count), 50)

        base_url = (
            self._base_url or os.getenv("ZHIPU_BASE_URL") or "https://open.bigmodel.cn"
        ).rstrip("/")
        url = f"{base_url}/api/paas/v4/web_search"

        payload: Dict[str, Any] = {
            "search_query": query,
            "search_engine": search_engine,
            "search_intent": search_intent,
            "count": count,
            "search_recency_filter": search_recency_filter,
            "content_size": content_size,
        }

        if search_domain_filter:
            payload["search_domain_filter"] = search_domain_filter
        if request_id:
            payload["request_id"] = request_id
        if user_id:
            payload["user_id"] = user_id

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
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
                    url, json=payload, headers=headers, timeout=15
                )

                if response.status_code in {401, 403}:
                    raise ValueError(
                        f"Zhipu API auth failed ({response.status_code}). "
                        "Please check ZHIPU_API_KEY."
                    )

                response.raise_for_status()
                return cast(Dict[str, Any], response.json())

        except httpx.RequestError as e:
            logger.error("❌ Network error during Zhipu web search: %s", str(e))
            raise ValueError(f"Network error during Zhipu web search: {str(e)}") from e
        except httpx.HTTPStatusError as e:
            logger.error("❌ Zhipu API HTTP error: %s", str(e))
            raise ValueError(f"Zhipu API HTTP error: {str(e)}") from e
        except Exception as e:
            logger.error("❌ Unexpected error during Zhipu web search: %s", str(e))
            raise ValueError(
                f"Unexpected error during Zhipu web search: {str(e)}"
            ) from e

    @staticmethod
    def normalize_results(response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Normalize Zhipu search results into a consistent format."""
        results: List[Dict[str, Any]] = []
        for item in response.get("search_result", []) or []:
            content = item.get("content", "")
            results.append(
                {
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": content,
                    "content": content,
                    "media": item.get("media", ""),
                    "icon": item.get("icon", ""),
                    "publish_date": item.get("publish_date", ""),
                    "refer": item.get("refer", ""),
                }
            )
        return results

    @staticmethod
    def _get_proxy_url() -> Optional[str]:
        """Get proxy URL from environment variables."""
        https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
        return https_proxy or http_proxy
