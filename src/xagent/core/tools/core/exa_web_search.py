"""
Exa Web Search Tool
Standalone web search functionality using Exa AI-powered search API.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ExaWebSearchCore:
    """Pure Exa web search tool without framework dependencies."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key

    async def search(
        self,
        query: str,
        num_results: int = 10,
        search_type: str = "auto",
        content_mode: str = "highlights",
        category: Optional[str] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        include_text: Optional[List[str]] = None,
        exclude_text: Optional[List[str]] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Search the web using Exa AI-powered search API.

        Args:
            query: The search query string
            num_results: Number of results to return (max 100)
            search_type: Search type - 'auto', 'neural', 'fast', or 'instant'
            content_mode: Content retrieval mode - 'highlights', 'text', 'summary', or 'none'
            category: Focus category - 'company', 'research paper', 'news', etc.
            include_domains: Restrict results to these domains
            exclude_domains: Exclude results from these domains
            include_text: Strings that must appear in webpage text
            exclude_text: Strings to exclude from results
            start_published_date: Filter by published date start (ISO 8601)
            end_published_date: Filter by published date end (ISO 8601)

        Returns:
            List of search results with title, link, snippet and optionally content
        """
        logger.info(
            "🔍 Exa web search query='%s' num_results=%s type=%s content=%s",
            query,
            num_results,
            search_type,
            content_mode,
        )

        api_key = self._api_key or os.getenv("EXA_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing required environment variable EXA_API_KEY. "
                "Get one at https://exa.ai"
            )

        num_results = min(max(1, num_results), 100)

        from exa_py import Exa

        client = Exa(api_key=api_key)
        client.headers["x-exa-integration"] = "xagent"

        # Build search kwargs
        search_kwargs: Dict[str, Any] = {
            "query": query,
            "num_results": num_results,
            "type": search_type,
        }

        if category:
            search_kwargs["category"] = category
        if include_domains:
            search_kwargs["include_domains"] = include_domains
        if exclude_domains:
            search_kwargs["exclude_domains"] = exclude_domains
        if include_text:
            search_kwargs["include_text"] = include_text
        if exclude_text:
            search_kwargs["exclude_text"] = exclude_text
        if start_published_date:
            search_kwargs["start_published_date"] = start_published_date
        if end_published_date:
            search_kwargs["end_published_date"] = end_published_date

        # Build contents parameter based on content_mode
        contents_param = self._build_contents_param(content_mode)

        try:
            if contents_param is not None:
                search_kwargs.update(contents_param)
                response = await asyncio.to_thread(
                    lambda: client.search_and_contents(**search_kwargs)
                )
            else:
                response = await asyncio.to_thread(
                    lambda: client.search(**search_kwargs)
                )
        except ValueError:
            raise
        except Exception as e:
            logger.error("❌ Error during Exa search: %s", str(e))
            raise ValueError(f"Error during Exa search: {str(e)}") from e

        logger.info("✅ Exa API request successful")
        return self._normalize_results(response, content_mode)

    @staticmethod
    def _build_contents_param(content_mode: str) -> Optional[Dict[str, Any]]:
        """Build the contents parameter for Exa API calls."""
        if content_mode == "highlights":
            return {"highlights": {"max_characters": 4000}}
        elif content_mode == "text":
            return {"text": {"max_characters": 10000}}
        elif content_mode == "summary":
            return {"summary": True}
        elif content_mode == "none":
            return None
        else:
            raise ValueError(
                f"Unknown content_mode '{content_mode}'. "
                "Must be one of: 'highlights', 'text', 'summary', 'none'"
            )

    @staticmethod
    def _normalize_results(response: Any, content_mode: str) -> List[Dict[str, str]]:
        """Normalize Exa search results into xagent's standard format."""
        results: List[Dict[str, str]] = []

        for item in response.results:
            title = getattr(item, "title", "") or ""
            url = getattr(item, "url", "") or ""

            # Extract content based on mode
            content = ""
            if content_mode == "highlights":
                highlights = getattr(item, "highlights", None)
                if highlights:
                    content = "\n".join(highlights)
            elif content_mode == "text":
                content = getattr(item, "text", "") or ""
            elif content_mode == "summary":
                content = getattr(item, "summary", "") or ""

            # Build snippet from content or highlights
            snippet = content[:500] if content else ""

            result: Dict[str, str] = {
                "title": title,
                "link": url,
                "snippet": snippet,
                "content": content[:8192] if content else "",
            }

            results.append(result)

        logger.info("🎯 Exa search returned %d results", len(results))
        return results
