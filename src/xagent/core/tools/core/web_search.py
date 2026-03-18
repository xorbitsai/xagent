"""
Pure Web Search Tool
Standalone web search functionality without framework dependencies
"""

import logging
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import html2text
import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class WebSearchCore:
    """Pure web search tool without framework dependencies"""

    def __init__(self, api_key: str | None = None, cse_id: str | None = None) -> None:
        self._api_key = api_key
        self._cse_id = cse_id

    async def search(
        self,
        query: str,
        num_results: int = 3,
        include_content: bool = True,
    ) -> List[Dict[str, str]]:
        """
        Search the web using Google Custom Search API.

        Args:
            query: The search query string
            num_results: Number of results to return (max 10)
            include_content: Include full webpage content

        Returns:
            List of search results with title, link, snippet and content
        """
        logger.info(
            f"🔍 Starting web search for query: '{query}' "
            f"(num_results={num_results}, include_content={include_content})"
        )

        api_key = self._api_key or os.getenv("GOOGLE_API_KEY")
        cse_id = self._cse_id or os.getenv("GOOGLE_CSE_ID")

        if not api_key or not cse_id:
            raise ValueError(
                "Missing required environment variables. Please set GOOGLE_API_KEY and GOOGLE_CSE_ID."
            )

        num_results = min(max(1, num_results), 10)

        # Setup proxy configuration
        proxy_url = self._get_proxy_url()
        if proxy_url:
            logger.info(f"🌐 Using proxy: {proxy_url}")

        params: Dict[str, Any] = {
            "key": api_key,
            "cx": cse_id,
            "q": query,
            "num": num_results,
            "hl": "en",
            "safe": "active",
        }

        try:
            client_kwargs: Dict[str, Any] = {}
            if proxy_url:
                client_kwargs["proxy"] = proxy_url

            logger.info("📡 Making request to Google Custom Search API...")
            async with httpx.AsyncClient(**client_kwargs) as client:
                response = await client.get(
                    "https://www.googleapis.com/customsearch/v1",
                    params=params,
                    timeout=10,
                )

                if response.status_code == 403:
                    self._handle_403_error(response)

                response.raise_for_status()
                data = response.json()

                logger.info("✅ Google API request successful")
                return await self._process_search_results(
                    data, include_content, proxy_url
                )

        except httpx.RequestError as e:
            logger.error(f"❌ Network error during search: {str(e)}")
            raise ValueError(f"Network error during search: {str(e)}") from e
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error during search: {str(e)}")
            raise ValueError(f"Unexpected error during search: {str(e)}") from e

    async def _process_search_results(
        self,
        data: Dict[str, Any],
        include_content: bool,
        proxy_url: Optional[str],
    ) -> List[Dict[str, str]]:
        """Process search results and optionally fetch page content"""
        results: List[Dict[str, str]] = []

        if "items" not in data:
            logger.warning("⚠️ No search results found in API response")
            return results

        logger.info(f"📋 Found {len(data['items'])} search results")

        for i, item in enumerate(data["items"], 1):
            result = {
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            }

            logger.info(f"🔗 Result {i}: {result['title']}")
            logger.info(f"   URL: {result['link']}")
            logger.info(f"   Snippet: {result['snippet'][:100]}...")

            if include_content:
                logger.info(f"📄 Fetching content for: {result['link']}")
                result["content"] = await self._fetch_page_content(
                    result["link"], proxy_url
                )
                content_length = len(result["content"])
                logger.info(f"   Content fetched: {content_length} characters")
                if content_length > 0:
                    content_preview = result["content"][:200].replace("\n", " ")
                    logger.info(f"   Content preview: {content_preview}...")

            results.append(result)

        logger.info(f"🎯 Search completed successfully with {len(results)} results")
        return results

    async def _fetch_page_content(
        self, url: str, proxy_url: Optional[str] = None
    ) -> str:
        """Fetch and convert webpage content to markdown"""
        logger.info(f"🌐 Fetching page content from: {url}")

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        try:
            client_kwargs: Dict[str, Any] = {}
            if proxy_url:
                client_kwargs["proxy"] = proxy_url
                logger.info(f"   Using proxy: {proxy_url}")

            async with httpx.AsyncClient(**client_kwargs) as client:
                logger.info("   Making HTTP request...")
                response = await client.get(url, headers=headers, timeout=10)
                response.raise_for_status()

                logger.info(
                    f"   Response received: {response.status_code} {response.reason_phrase}"
                )
                logger.info(
                    f"   Content-Type: {response.headers.get('content-type', 'unknown')}"
                )
                logger.info(f"   Content-Length: {len(response.text)} characters")

                soup = BeautifulSoup(response.text, "html.parser")
                logger.info("   HTML parsed successfully")

                # Remove script and style elements
                scripts_removed = len(soup(["script", "style"]))
                for script in soup(["script", "style"]):
                    script.decompose()
                logger.info(f"   Removed {scripts_removed} script/style elements")

                # Convert relative URLs to absolute
                links_processed = 0
                for tag in soup.find_all(["a", "img"]):
                    if not hasattr(tag, "get") or not hasattr(tag, "__setitem__"):
                        continue

                    if hasattr(tag, "get") and hasattr(tag, "__setitem__"):
                        if tag.get("href"):
                            tag["href"] = urljoin(url, tag["href"])
                            links_processed += 1
                        if tag.get("src"):
                            tag["src"] = urljoin(url, tag["src"])
                            links_processed += 1

                if links_processed > 0:
                    logger.info(f"   Processed {links_processed} relative URLs")

                h2t = html2text.HTML2Text()
                h2t.body_width = 0
                h2t.ignore_images = False
                h2t.ignore_emphasis = False
                h2t.ignore_links = False
                h2t.ignore_tables = False

                logger.info("   Converting HTML to markdown...")
                markdown = h2t.handle(str(soup))

                # Log a preview of the markdown content
                lines = markdown.strip().split("\n")
                non_empty_lines = [line.strip() for line in lines if line.strip()]
                if non_empty_lines:
                    preview_lines = non_empty_lines[:3]
                    logger.info("   Content preview (first 3 lines):")
                    for i, line in enumerate(preview_lines, 1):
                        preview = line[:100] + "..." if len(line) > 100 else line
                        logger.info(f"     {i}. {preview}")

                return markdown.strip()

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code} error for {url}: {e.response.reason_phrase}"
            logger.error(f"   ❌ {error_msg}")
            return f"Error fetching content: {error_msg}"
        except httpx.RequestError as e:
            error_msg = f"Network error for {url}: {str(e)}"
            logger.error(f"   ❌ {error_msg}")
            return f"Error fetching content: {error_msg}"
        except Exception as e:
            error_msg = f"Unexpected error for {url}: {str(e)}"
            logger.error(f"   ❌ {error_msg}")
            return f"Error fetching content: {error_msg}"

    def _get_proxy_url(self) -> Optional[str]:
        """Get proxy URL from environment variables"""
        https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
        return https_proxy or http_proxy

    def _handle_403_error(self, response: httpx.Response) -> None:
        """Handle 403 Forbidden errors from Google API"""
        try:
            error_data = response.json()
            error_message = error_data.get("error", {}).get("message", "Unknown error")
            error_reason = (
                error_data.get("error", {})
                .get("errors", [{}])[0]
                .get("reason", "Unknown")
            )
            logger.error(
                f"❌ Google API 403 Error: {error_message} (reason: {error_reason})"
            )
            raise ValueError(
                f"Google API 403 Error: {error_message}\n"
                f"Reason: {error_reason}\n"
                f"This usually means:\n"
                f"- API quota exceeded\n"
                f"- Invalid API key\n"
                f"- Custom Search Engine ID is incorrect\n"
                f"- Custom Search API is not enabled\n"
                f"Please check your Google Cloud Console settings."
            )
        except Exception:
            logger.error("❌ Google API 403 Forbidden error")
            raise ValueError(
                "Google API 403 Forbidden error. This usually means:\n"
                "- API quota exceeded\n"
                "- Invalid API key\n"
                "- Custom Search Engine ID is incorrect\n"
                "- Custom Search API is not enabled\n"
                "Please check your Google Cloud Console settings."
            )


# Convenience function for direct usage
async def search_web(
    query: str,
    num_results: int = 3,
    include_content: bool = True,
) -> List[Dict[str, str]]:
    """
    Search the web using Google Custom Search API.

    Args:
        query: The search query string
        num_results: Number of results to return (max 10)
        include_content: Include full webpage content

    Returns:
        List of search results with title, link, snippet and content
    """
    searcher = WebSearchCore()
    return await searcher.search(
        query=query,
        num_results=num_results,
        include_content=include_content,
    )
