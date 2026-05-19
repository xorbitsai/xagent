"""Website ingestion pipeline for knowledge base.

Crawls a website and imports all discovered pages into the knowledge base.
"""

import asyncio
import concurrent.futures
import logging
import tempfile
from contextvars import copy_context
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, TypedDict

from ..core.schemas import (
    CrawlResult,
    IngestionConfig,
    IngestionResult,
    WebCrawlConfig,
    WebIngestionResult,
)
from ..progress import get_progress_manager
from ..utils.config_utils import coerce_ingestion_config
from ..utils.string_utils import sanitize_for_doc_id
from ..utils.user_scope import resolve_user_scope
from ..web_crawler import WebCrawler
from .document_ingestion import run_document_ingestion

logger = logging.getLogger(__name__)


_CRAWLER_BLOCK_ERROR_MARKERS: tuple[str, ...] = (
    "http 403",
    "403 forbidden",
    "http 429",
    "429 too many requests",
    "checking your browser",
    "cf-challenge",
    "just a moment",
    "security review",
    "access denied",
    "blocked",
    "challenge page",
)

_CRAWLER_BLOCK_MESSAGE = (
    "Web ingestion failed. The target website is blocking access to "
    "automated crawlers. Please use a different method to create your KB."
)


class FileHandlerResult(TypedDict):
    """Return type for file_handler callback.

    Attributes:
        file_path: Path to the file for ingestion (persistent or temporary)
        file_id: Optional file_id for stable doc_id generation
    """

    file_path: str
    file_id: Optional[str]


def _looks_like_crawler_block(error: str) -> bool:
    """Heuristically detect WAF / anti-bot blocks from a crawl failure string."""
    normalized_error = error.lower()
    return any(marker in normalized_error for marker in _CRAWLER_BLOCK_ERROR_MARKERS)


async def run_web_ingestion(
    collection: str,
    crawl_config: WebCrawlConfig,
    *,
    ingestion_config: Optional[IngestionConfig] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    user_id: Optional[int] = None,
    is_admin: Optional[bool] = None,
    file_handler: Optional[Callable[[Path, str, str, str], FileHandlerResult]] = None,
    trace_id: Optional[str] = None,
) -> WebIngestionResult:
    """Crawl a website and ingest all pages into the knowledge base.

    This pipeline performs the following steps:
    1. Crawl the website according to the provided configuration
    2. For each crawled page, save content and call file_handler (if provided)
    3. Ingest each page using the returned file information
    4. Aggregate statistics and return comprehensive results

    Args:
        collection: Target collection name for ingestion
        crawl_config: Website crawling configuration
        ingestion_config: Optional document ingestion configuration
        progress_callback: Optional callback for progress updates
            Args: (message, completed, total)
        user_id: Optional user ID for ownership tracking
        is_admin: Optional admin override; when omitted, falls back to request scope
        file_handler: Optional callback to handle file persistence and UploadedFile
            record creation. Signature: (temp_file_path, title, collection, url)
            Returns FileHandlerResult with file_path and optional file_id.
            If not provided, temporary files will be used without UploadedFile records.

    Returns:
        WebIngestionResult: Comprehensive result with statistics

    Raises:
        ValueError: If configuration is invalid
        RuntimeError: If ingestion fails critically
    """
    scope = resolve_user_scope(user_id=user_id, is_admin=is_admin)
    user_id = scope.user_id
    is_admin = scope.is_admin

    start_time = datetime.now(timezone.utc)
    warnings: list[str] = []
    failed_urls: dict[str, str] = {}

    # Normalize ingestion config
    ing_cfg = coerce_ingestion_config(ingestion_config)

    logger.info(
        "Starting web ingestion",
        extra={
            "trace_id": trace_id,
            "collection": collection,
            "start_url": crawl_config.start_url,
            "user_id": user_id,
            "is_admin": is_admin,
            "max_pages": crawl_config.max_pages,
            "max_depth": crawl_config.max_depth,
            "same_domain_only": crawl_config.same_domain_only,
            "url_patterns": crawl_config.url_patterns,
            "exclude_patterns": crawl_config.exclude_patterns,
            "content_selector": crawl_config.content_selector,
            "remove_selectors": crawl_config.remove_selectors,
            "concurrent_requests": crawl_config.concurrent_requests,
            "request_delay": crawl_config.request_delay,
            "timeout": crawl_config.timeout,
            "respect_robots_txt": crawl_config.respect_robots_txt,
            "render_js": crawl_config.render_js,
            "render_wait_until": crawl_config.render_wait_until,
            "render_timeout_ms": crawl_config.render_timeout_ms,
            "ingestion_parse_method": str(ing_cfg.parse_method),
            "ingestion_chunk_strategy": str(ing_cfg.chunk_strategy),
            "ingestion_chunk_size": ing_cfg.chunk_size,
            "ingestion_chunk_overlap": ing_cfg.chunk_overlap,
            "ingestion_separators": ing_cfg.separators,
            "embedding_model_id": ing_cfg.embedding_model_id,
            "embedding_batch_size": ing_cfg.embedding_batch_size,
            "embedding_use_async": ing_cfg.embedding_use_async,
            "embedding_concurrent": ing_cfg.embedding_concurrent,
        },
    )

    # Step 1: Crawl the website
    logger.info("Step 1: Crawling website")
    crawler = WebCrawler(crawl_config, progress_callback, trace_id=trace_id)

    try:
        crawl_results: list[CrawlResult] = await crawler.crawl()
    except Exception as e:
        logger.exception("Website crawling failed")
        elapsed_ms = int(
            (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        )
        return WebIngestionResult(
            status="error",
            collection=collection,
            total_urls_found=0,
            pages_crawled=0,
            pages_failed=0,
            documents_created=0,
            chunks_created=0,
            embeddings_created=0,
            crawled_urls=[],
            failed_urls={},
            message=(
                "Website crawling failed: "
                + (
                    "Playwright browsers are not installed. Run: playwright install chromium"
                    if "Playwright browsers are not installed" in str(e)
                    else str(e)
                )
            ),
            warnings=[],
            elapsed_time_ms=elapsed_ms,
        )

    pages_crawled = len([r for r in crawl_results if r.status == "success"])

    # Collect failed URLs from crawler
    for url, error in crawler.failed_urls.items():
        failed_urls[url] = error

    # Calculate pages_failed (will be updated as ingestion failures are tracked)
    pages_failed = len(failed_urls)

    logger.info(
        "Crawling completed",
        extra={
            "trace_id": trace_id,
            "collection": collection,
            "start_url": crawl_config.start_url,
            "successful_pages": pages_crawled,
            "failed_pages": pages_failed,
            "total_urls_found": crawler.total_urls_found,
            "results_count": len(crawl_results),
        },
    )

    # Fail fast: crawling produced no successful pages but has concrete failure reasons.
    # This avoids continuing into ingestion and makes debugging much easier.
    if pages_crawled == 0 and pages_failed > 0:
        elapsed_ms = int(
            (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        )
        blocking_entry = next(
            (
                (url, err)
                for url, err in failed_urls.items()
                if _looks_like_crawler_block(err)
            ),
            None,
        )
        if blocking_entry is not None:
            fail_fast_message = _CRAWLER_BLOCK_MESSAGE
        else:
            reasons: list[str] = []
            for url, reason in list(failed_urls.items())[:3]:
                reasons.append(f"{url}: {reason}")
            reason_suffix = "; ".join(reasons)
            fail_fast_message = "Website crawling failed: no valid pages extracted." + (
                f" Reasons: {reason_suffix}" if reason_suffix else ""
            )
        return WebIngestionResult(
            status="error",
            collection=collection,
            total_urls_found=crawler.total_urls_found,
            pages_crawled=0,
            pages_failed=pages_failed,
            documents_created=0,
            chunks_created=0,
            embeddings_created=0,
            crawled_urls=[],
            failed_urls=failed_urls,
            message=fail_fast_message,
            warnings=warnings,
            elapsed_time_ms=elapsed_ms,
        )

    # Step 2: Ingest each crawled page
    logger.info("Step 2: Ingesting crawled pages")

    # Create temporary directory for markdown files
    with tempfile.TemporaryDirectory(prefix="xagent_web_ingest_") as temp_dir:
        documents_created = 0
        total_chunks = 0
        total_embeddings = 0

        # Copy context once before the loop to avoid repeated ContextVar copying.
        # The request-scoped user context remains constant throughout the request.
        # NOTE: This copies ALL ContextVars (tracing IDs, request IDs, etc.).
        loop = asyncio.get_event_loop()
        request_context = copy_context()

        for i, crawl_result in enumerate(crawl_results):
            if crawl_result.status != "success":
                continue

            # Progress callback
            if progress_callback:
                progress_callback(
                    f"Ingesting page {i + 1}/{len(crawl_results)}: {crawl_result.url}",
                    i + 1,
                    len(crawl_results),
                )

            try:
                logger.debug(
                    "Ingesting crawled page",
                    extra={
                        "trace_id": trace_id,
                        "collection": collection,
                        "url": crawl_result.url,
                        "title": crawl_result.title,
                        "depth": crawl_result.depth,
                        "content_length": crawl_result.content_length,
                        "index": i + 1,
                        "total": len(crawl_results),
                    },
                )
                # Save crawled content to temporary markdown file
                filename = sanitize_for_doc_id(crawl_result.title or f"page_{i + 1}")
                temp_file = Path(temp_dir) / f"{filename}.md"

                with open(temp_file, "w", encoding="utf-8") as f:
                    # Add metadata header
                    f.write(f"# {crawl_result.title or 'Untitled'}\n\n")
                    f.write(f"**Source:** {crawl_result.url}\n\n")
                    f.write(f"**Crawled:** {crawl_result.timestamp.isoformat()}\n\n")
                    f.write("---\n\n")
                    f.write(crawl_result.content_markdown)

                logger.debug("Saved %s to %s", crawl_result.url, temp_file)

                # Call file_handler if provided (for persistent storage and UploadedFile record)
                final_file_path = temp_file
                final_file_id = None
                copied_persistent_file = None

                if file_handler:
                    try:
                        file_info = file_handler(
                            temp_file,
                            crawl_result.title or f"page_{i + 1}",
                            collection,
                            crawl_result.url,
                        )
                        final_file_path = Path(file_info.get("file_path", temp_file))
                        final_file_id = file_info.get("file_id")

                        # Track if we successfully copied a persistent file for cleanup
                        if final_file_path != temp_file and final_file_path.exists():
                            copied_persistent_file = final_file_path

                        logger.debug(
                            "File handler returned: path=%s, file_id=%s",
                            final_file_path,
                            final_file_id,
                        )
                    except Exception as e:
                        logger.exception("File handler failed for %s", crawl_result.url)
                        failure_message = (
                            f"File persistence failed for {crawl_result.url}: {e}"
                        )
                        failed_urls[crawl_result.url] = failure_message
                        warnings.append(failure_message)
                        continue

                try:
                    progress_manager = get_progress_manager()

                    def _ingest_file() -> IngestionResult:
                        return run_document_ingestion(
                            collection=collection,
                            source_path=str(final_file_path),
                            file_id=final_file_id,
                            ingestion_config=ing_cfg,
                            progress_manager=progress_manager,
                            user_id=user_id,
                            is_admin=is_admin,
                            trace_id=trace_id,
                        )

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        ingest_result: IngestionResult = await loop.run_in_executor(
                            executor, lambda: request_context.run(_ingest_file)
                        )

                    if ingest_result.status == "success":
                        documents_created += 1
                        total_chunks += ingest_result.chunk_count
                        total_embeddings += ingest_result.embedding_count
                        logger.debug(
                            "Ingested crawled page successfully",
                            extra={
                                "trace_id": trace_id,
                                "collection": collection,
                                "url": crawl_result.url,
                                "doc_id": ingest_result.doc_id,
                                "parse_hash": ingest_result.parse_hash,
                                "chunk_count": ingest_result.chunk_count,
                                "embedding_count": ingest_result.embedding_count,
                                "vector_count": ingest_result.vector_count,
                            },
                        )
                        copied_persistent_file = None
                    else:
                        failed_urls[crawl_result.url] = ingest_result.message
                        msg = (
                            f"Partial ingestion for {crawl_result.url}: "
                            f"{ingest_result.message}"
                        )
                        warnings.append(msg)
                        logger.debug(
                            "Ingested crawled page failed",
                            extra={
                                "trace_id": trace_id,
                                "collection": collection,
                                "url": crawl_result.url,
                                "status": ingest_result.status,
                                "failed_step": ingest_result.failed_step,
                                "ingest_message": ingest_result.message,
                            },
                        )

                except Exception as e:
                    logger.exception("Failed to ingest %s", crawl_result.url)
                    failed_urls[crawl_result.url] = str(e)
                    warnings.append(f"Failed to ingest {crawl_result.url}: {str(e)}")

                    if copied_persistent_file and copied_persistent_file.exists():
                        try:
                            copied_persistent_file.unlink()
                            logger.info(
                                "Cleaned up persistent file due to ingestion failure: %s",
                                copied_persistent_file,
                            )
                        except Exception as cleanup_error:
                            logger.warning(
                                "Failed to clean up persistent file %s: %s",
                                copied_persistent_file,
                                cleanup_error,
                            )
                    copied_persistent_file = None

            except Exception as e:
                logger.exception("Failed to ingest %s", crawl_result.url)
                failed_urls[crawl_result.url] = str(e)
                warnings.append(f"Failed to ingest {crawl_result.url}: {str(e)}")

    # Step 3: Compile results
    elapsed_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)

    # Recalculate pages_failed to include ingestion failures
    # (pages_failed includes both crawl failures and ingestion failures)
    pages_failed = len(failed_urls)

    # Status determination:
    # - "error": No docs created AND there were actual failures
    # - "partial": Some docs created but some failures
    # - "success": No failures (empty results are successful)
    total_failures = pages_failed

    if documents_created == 0 and total_failures > 0:
        status = "error"
    elif total_failures > 0:
        status = "partial"
    else:
        status = "success"

    crawled_urls_list = [r.url for r in crawl_results if r.status == "success"]

    # Build a status-aware message. Previously this was unconditionally
    # "Web ingestion completed: ..." even on error, which produced the
    # "red error toast + green-toned 'completed' text" UX in the frontend
    # whenever every crawl attempt got blocked. On error/partial we now
    # check all failures for anti-bot/WAF signals and otherwise surface
    # the first failing URL and its reason so the user sees something
    # actionable.
    if (status == "error" or status == "partial") and failed_urls:
        first_url, first_err = next(iter(failed_urls.items()))
        blocking_entry = next(
            (
                (url, err)
                for url, err in crawler.failed_urls.items()
                if _looks_like_crawler_block(err)
            ),
            None,
        )

        if status == "error":
            if blocking_entry:
                message = _CRAWLER_BLOCK_MESSAGE
            else:
                message = f"Web ingestion failed: {first_url} returned {first_err}"
        else:
            if blocking_entry:
                blocking_url, _ = blocking_entry
                message = (
                    f"Web ingestion partial: {documents_created} documents from "
                    f"{pages_crawled} pages, {len(failed_urls)} failed. "
                    f"Some pages (e.g. {blocking_url}) are blocking access to "
                    "automated crawlers. Please use a different method to "
                    "create your KB for those pages."
                )
            else:
                message = (
                    f"Web ingestion partial: {documents_created} documents from "
                    f"{pages_crawled} pages, {len(failed_urls)} failed "
                    f"(first: {first_url} returned {first_err})"
                )
    else:
        message = (
            f"Web ingestion completed: {documents_created} documents, "
            f"{total_chunks} chunks, {total_embeddings} embeddings"
        )

    result = WebIngestionResult(
        status=status,
        collection=collection,
        total_urls_found=crawler.total_urls_found,
        pages_crawled=pages_crawled,
        pages_failed=pages_failed,
        documents_created=documents_created,
        chunks_created=total_chunks,
        embeddings_created=total_embeddings,
        crawled_urls=crawled_urls_list,
        failed_urls=failed_urls,
        message=message,
        warnings=warnings,
        elapsed_time_ms=elapsed_ms,
    )

    logger.info(
        "Web ingestion completed",
        extra={
            "trace_id": trace_id,
            "collection": collection,
            "start_url": crawl_config.start_url,
            "status": result.status,
            "total_urls_found": crawler.total_urls_found,
            "pages_crawled": pages_crawled,
            "pages_failed": pages_failed,
            "documents_created": documents_created,
            "chunks_created": total_chunks,
            "embeddings_created": total_embeddings,
            "failed_urls_count": len(failed_urls),
            "warnings_count": len(warnings),
            "elapsed_time_ms": elapsed_ms,
        },
    )

    return result
