"""Core web crawler implementation."""

import asyncio
import logging
import time
from collections import deque
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set

import httpx

from ..core.schemas import CrawlResult, WebCrawlConfig
from .content_cleaner import ContentCleaner
from .link_extractor import LinkExtractor
from .url_filter import URLFilter

logger = logging.getLogger(__name__)


class WebCrawler:
    """Asynchronous web crawler with configurable filtering and rate limiting."""

    def __init__(
        self,
        config: WebCrawlConfig,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ):
        """Initialize web crawler.

        Args:
            config: Crawl configuration
            progress_callback: Optional callback for progress updates
                Args: (current_url, completed, total)
        """
        self.config = config
        self.progress_callback = progress_callback

        # Initialize components
        self.url_filter = URLFilter(
            base_url=config.start_url,
            same_domain_only=config.same_domain_only,
            url_patterns=config.url_patterns,
            exclude_patterns=config.exclude_patterns,
            respect_robots_txt=config.respect_robots_txt,
        )
        self.content_cleaner = ContentCleaner(
            content_selector=config.content_selector,
            remove_selectors=config.remove_selectors,
        )
        self.link_extractor = LinkExtractor(config.start_url)

        # Crawl state
        self.visited_urls: Set[str] = set()
        self.pending_urls: deque = deque()
        self.crawl_results: List[CrawlResult] = []
        self.failed_urls: Dict[str, str] = {}

        # Statistics
        self.total_urls_found = 0
        self.start_time: Optional[float] = None

    async def crawl(self) -> List[CrawlResult]:
        """Start crawling from the configured start URL.

        Returns:
            List of crawl results
        """
        self.start_time = time.time()

        # Add start URL to pending
        start_url_normalized = self.url_filter.normalize_url(self.config.start_url)
        if start_url_normalized:
            self.pending_urls.append((start_url_normalized, 0))  # (url, depth)

        logger.info(f"Starting crawl from {self.config.start_url}")

        # Create HTTP client
        user_agent = self.config.user_agent or "Mozilla/5.0 (xagent WebCrawler/1.0)"
        headers = {"User-Agent": user_agent}
        async with httpx.AsyncClient(
            headers=headers,
            timeout=self.config.timeout,
        ) as client:
            await self._crawl_loop(client)

        elapsed = time.time() - self.start_time
        logger.info(
            f"Crawl completed: {len(self.crawl_results)} pages, "
            f"{len(self.failed_urls)} failed, {elapsed:.2f}s"
        )

        return self.crawl_results

    async def _crawl_loop(self, client: httpx.AsyncClient) -> None:
        """Main crawl loop with concurrency control.

        Args:
            client: HTTP client
        """
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.concurrent_requests)

        # Process URLs until we reach max_pages or no more pending URLs
        while self.pending_urls and len(self.visited_urls) < self.config.max_pages:
            # Batch processing for concurrency
            tasks = []
            batch_size = min(
                self.config.concurrent_requests,
                len(self.pending_urls),
                self.config.max_pages - len(self.visited_urls),
            )

            for _ in range(batch_size):
                if not self.pending_urls:
                    break

                url, depth = self.pending_urls.popleft()

                # Skip if already visited
                if url in self.visited_urls:
                    continue

                # Skip if exceeds max depth
                if depth > self.config.max_depth:
                    logger.debug(
                        f"Skipping {url}: exceeds max depth {self.config.max_depth}"
                    )
                    continue

                self.visited_urls.add(url)

                # Create crawl task
                task = self._crawl_page(client, url, depth, semaphore)
                tasks.append(task)

            # Execute batch
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results and extract new links
                for result_tuple in results:
                    if isinstance(result_tuple, Exception):
                        logger.error(f"Crawl task failed: {result_tuple}")
                        continue

                    # result_tuple should be a tuple (CrawlResult, Set[str])
                    if not isinstance(result_tuple, tuple) or len(result_tuple) != 2:
                        logger.error(f"Invalid result format: {result_tuple}")
                        continue

                    result, links = result_tuple
                    if result and result.status == "success":
                        # Queue extracted links
                        await self._process_links(links, result.depth)

                # Rate limiting
                if self.pending_urls:
                    await asyncio.sleep(self.config.request_delay)

                # Progress callback
                if self.progress_callback:
                    self.progress_callback(
                        f"Crawled {len(self.visited_urls)} pages",
                        len(self.visited_urls),
                        self.config.max_pages,
                    )

    async def _crawl_page(
        self,
        client: httpx.AsyncClient,
        url: str,
        depth: int,
        semaphore: asyncio.Semaphore,
    ) -> tuple[Optional[CrawlResult], Set[str]]:
        """Crawl a single page.

        Args:
            client: HTTP client
            url: URL to crawl
            depth: Current depth
            semaphore: Concurrency control semaphore

        Returns:
            Tuple of (CrawlResult or None, Set of extracted links)
        """
        async with semaphore:
            try:
                logger.debug(f"Crawling {url} (depth: {depth})")

                # Fetch page
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()

                html = response.text

                # Clean and convert content
                cleaned = self.content_cleaner.clean_and_convert(html, url)

                # Validate content
                content = cleaned["content_markdown"]
                if not self.content_cleaner.is_valid_content(content, min_length=10):
                    logger.warning(f"Insufficient content at {url}")
                    self.failed_urls[url] = "Insufficient content"
                    return None, set()

                # Extract links
                links = self.link_extractor.extract_links(html, url)

                # Filter links
                valid_links = set()
                user_agent = self.config.user_agent or "*"
                for link in links:
                    if self.url_filter.should_crawl(link, user_agent):
                        valid_links.add(link)

                self.total_urls_found += len(links)

                # Create result
                result = CrawlResult(
                    url=url,
                    title=cleaned["title"],
                    content_markdown=cleaned["content_markdown"],
                    status="success",
                    depth=depth,
                    timestamp=datetime.utcnow(),
                    content_length=cleaned["content_length"],
                    links_found=len(valid_links),
                )

                self.crawl_results.append(result)
                logger.info(
                    f"Successfully crawled {url} "
                    f"({cleaned['content_length']} chars, "
                    f"{len(valid_links)} valid links)"
                )

                return result, valid_links

            except httpx.HTTPStatusError as e:
                error_msg = f"HTTP {e.response.status_code}"
                logger.error(f"Failed to crawl {url}: {error_msg}")
                self.failed_urls[url] = error_msg
                return None, set()

            except httpx.RequestError as e:
                error_msg = f"Request error: {str(e)}"
                logger.error(f"Failed to crawl {url}: {error_msg}")
                self.failed_urls[url] = error_msg
                return None, set()

            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                logger.error(f"Failed to crawl {url}: {error_msg}")
                self.failed_urls[url] = error_msg
                return None, set()

    async def _process_links(self, links: Set[str], current_depth: int) -> None:
        """Process and queue links from a crawled page.

        Args:
            links: Set of extracted links
            current_depth: Depth of the current page
        """
        next_depth = current_depth + 1

        # Skip if we've reached max depth
        if next_depth > self.config.max_depth:
            return

        # Add filtered links to pending queue
        pending_urls_set = {u for u, _ in self.pending_urls}
        for link in links:
            # Only add if not visited and not already pending
            if link not in self.visited_urls and link not in pending_urls_set:
                self.pending_urls.append((link, next_depth))

    def get_statistics(self) -> dict:
        """Get crawl statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_urls_found": self.total_urls_found,
            "visited_urls": len(self.visited_urls),
            "successful_pages": len(self.crawl_results),
            "failed_pages": len(self.failed_urls),
            "pending_urls": len(self.pending_urls),
        }


async def crawl_website(
    config: WebCrawlConfig,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> List[CrawlResult]:
    """Convenience function to crawl a website.

    Args:
        config: Crawl configuration
        progress_callback: Optional progress callback

    Returns:
        List of crawl results
    """
    crawler = WebCrawler(config, progress_callback)
    return await crawler.crawl()
