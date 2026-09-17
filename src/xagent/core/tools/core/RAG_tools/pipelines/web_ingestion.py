"""Website ingestion pipeline for knowledge base.

Crawls a website and imports all discovered pages into the knowledge base.
"""

import asyncio
import inspect
import logging
import tempfile
from contextvars import copy_context
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    NotRequired,
    Optional,
    TypedDict,
    cast,
)

from ..core.schemas import (
    CrawlResult,
    IngestionConfig,
    IngestionResult,
    WebCrawlConfig,
    WebIngestionResult,
)
from ..kb.operation_compatibility import _close_awaitable_if_possible
from ..progress import get_progress_manager
from ..utils.config_utils import coerce_ingestion_config
from ..utils.string_utils import sanitize_for_doc_id
from ..utils.user_scope import resolve_user_scope
from ..web_crawler import WebCrawler
from ..web_crawler.crawler import STOPPED_NO_ELIGIBLE_LINKS
from ..web_crawler.url_filter import REJECTED_ROBOTS, SITE_REJECTIONS
from .document_ingestion import run_document_ingestion

if TYPE_CHECKING:
    from ..kb import KBPipelineCompatibilityFacade

logger = logging.getLogger(__name__)

# Reason keys are internal identifiers; this text reaches an end user. Only
# SITE_REJECTIONS reasons are ever rendered, so this covers exactly those -
# a reason added to that set needs a label here too.
_REJECTION_LABELS = {
    REJECTED_ROBOTS: "blocked by robots.txt rules",
}


def _blocked_crawl_explanation(
    crawl_stats: dict, documents_created: int, pages_crawled: int
) -> str:
    """Say that the site refused the crawl, counting only what the site did.

    Configured rejections are the operator's own scope rules; listing them in a
    sentence that blames the site would report that filtering as a refusal.
    """
    rejections = crawl_stats.get("link_rejections", {})
    refused = ", ".join(
        f"{rejections[reason]} {_REJECTION_LABELS.get(reason, reason)}"
        for reason in sorted(SITE_REJECTIONS)
        if rejections.get(reason)
    )
    # Unreachable today - the stop reason this message explains requires a site
    # rejection - but an empty parenthetical would read as a formatting bug.
    cause = f" ({refused})" if refused else ""
    return (
        f"Only {documents_created} document(s) from {pages_crawled} page(s) "
        f"could be imported: every link found was refused{cause}. The pages "
        f"that were reachable have been ingested."
    )


FileHandlerCallback = Callable[..., Any]


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
        rollback_on_failure: Optional callback to compensate file persistence
            when the subsequent document ingestion does not succeed.
        commit_on_success: Optional callback to finalize temporary rollback
            resources once the subsequent document ingestion succeeds.
        rollback_context: Optional operation-outcome metadata describing the
            web file side effect. This is internal and does not affect public
            web ingestion result schemas.
        file_compensation: Optional FILE-boundary compensation callback.
        document_compensation: Optional DOCUMENT-boundary compensation callback.
        status_compensation: Optional STATUS-boundary compensation callback.
        snapshot_compensation: Optional SNAPSHOT-boundary compensation callback.
    """

    file_path: str
    file_id: Optional[str]
    rollback_on_failure: NotRequired[FileHandlerCallback]
    commit_on_success: NotRequired[FileHandlerCallback]
    rollback_context: NotRequired[dict[str, object]]
    file_compensation: NotRequired[FileHandlerCallback]
    document_compensation: NotRequired[FileHandlerCallback]
    status_compensation: NotRequired[FileHandlerCallback]
    snapshot_compensation: NotRequired[FileHandlerCallback]


class _FileHandlerRollbackError(RuntimeError):
    def __init__(self, callback_name: str, url: str, reason: str) -> None:
        self.callback_name = callback_name
        self.url = url
        self.reason = reason
        super().__init__(f"File persistence {callback_name} failed for {url}: {reason}")


def _callback_accepts_ingestion_result(callback: FileHandlerCallback) -> bool:
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return False

    try:
        signature.bind(object())
    except TypeError:
        return False
    return True


def _run_sync_file_handler_callback(
    callback: FileHandlerCallback,
    *,
    callback_name: str,
    url: str,
    ingestion_result: Optional[IngestionResult],
) -> None:
    if _callback_accepts_ingestion_result(callback):
        result = callback(ingestion_result)
    else:
        result = callback()
    if inspect.isawaitable(result):
        _close_awaitable_if_possible(result)
        raise TypeError(f"Async {callback_name} callback is not supported for {url}")


def _run_file_handler_callback(
    file_info: Optional[FileHandlerResult],
    callback_name: str,
    *,
    url: str,
    warnings: list[str],
    ingestion_result: Optional[IngestionResult] = None,
) -> Optional[str]:
    if not file_info:
        return None

    callback = cast(Optional[FileHandlerCallback], file_info.get(callback_name))
    if callback is None:
        return None

    try:
        _run_sync_file_handler_callback(
            callback,
            callback_name=callback_name,
            url=url,
            ingestion_result=ingestion_result,
        )
    except Exception as cleanup_error:  # noqa: BLE001
        cleanup_reason = str(cleanup_error)
        message = f"File persistence {callback_name} failed for {url}: {cleanup_reason}"
        logger.warning(message, exc_info=True)
        warnings.append(message)
        return cleanup_reason
    return None


def _rollback_context_payload(
    file_info: Optional[FileHandlerResult],
) -> Optional[dict[str, object]]:
    if not file_info:
        return None
    rollback_context = file_info.get("rollback_context")
    return rollback_context if isinstance(rollback_context, dict) else None


def _has_per_boundary_compensation(file_info: FileHandlerResult) -> bool:
    return any(
        file_info.get(key)
        for key in (
            "file_compensation",
            "document_compensation",
            "status_compensation",
            "snapshot_compensation",
        )
    )


def _string_or_none(value: object) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def _ingestion_doc_id(ingestion_result: Optional[IngestionResult]) -> Optional[str]:
    if ingestion_result is None:
        return None
    return _string_or_none(ingestion_result.doc_id)


def _run_file_handler_compensation(
    *,
    pipeline_facade: "KBPipelineCompatibilityFacade",
    page_operation: Any,
    file_info: Optional[FileHandlerResult],
    collection: str,
    url: str,
    warnings: list[str],
    ingestion_result: Optional[IngestionResult] = None,
) -> Optional[str]:
    if not file_info:
        return None

    # Per-boundary compensation takes priority over legacy rollback_on_failure.
    has_per_boundary = _has_per_boundary_compensation(file_info)
    if has_per_boundary:
        return _run_per_boundary_compensation(
            pipeline_facade=pipeline_facade,
            page_operation=page_operation,
            file_info=file_info,
            collection=collection,
            url=url,
            warnings=warnings,
            ingestion_result=ingestion_result,
        )

    # Legacy monolithic rollback_on_failure callback (custom callbacks only)
    legacy_callback = cast(
        Optional[FileHandlerCallback], file_info.get("rollback_on_failure")
    )
    if legacy_callback is not None:
        return _run_legacy_rollback_compensation(
            pipeline_facade=pipeline_facade,
            page_operation=page_operation,
            file_info=file_info,
            collection=collection,
            url=url,
            warnings=warnings,
            legacy_callback=legacy_callback,
            ingestion_result=ingestion_result,
        )

    return None


def _run_per_boundary_compensation(
    *,
    pipeline_facade: "KBPipelineCompatibilityFacade",
    page_operation: Any,
    file_info: FileHandlerResult,
    collection: str,
    url: str,
    warnings: list[str],
    ingestion_result: Optional[IngestionResult] = None,
) -> Optional[str]:
    """Delegate per-boundary rollback orchestration to the coordinator (#515).

    Boundary ordering, the SNAPSHOT-after-FILE gate, and error folding are
    owned by KBCoordinator.rollback_failed_ingestion_sync. ``operation=None``
    (no active page operation) selects the coordinator's callback-only path,
    which absorbed the former _run_per_boundary_callbacks_without_operation
    duplicate.
    """
    from ..kb.models import RollbackFailedIngestionRequest

    rollback_context = dict(_rollback_context_payload(file_info) or {})
    rollback_context.setdefault("file_id", file_info.get("file_id"))
    rollback_context.setdefault("file_path", file_info.get("file_path"))

    request = RollbackFailedIngestionRequest(
        collection=collection,
        user_id=None,
        is_admin=False,
        operation=page_operation,
        ingestion_result=ingestion_result,
        doc_id=_ingestion_doc_id(ingestion_result),
        source=url,
        document_compensation=cast(
            Optional[FileHandlerCallback], file_info.get("document_compensation")
        ),
        file_compensation=cast(
            Optional[FileHandlerCallback], file_info.get("file_compensation")
        ),
        status_compensation=cast(
            Optional[FileHandlerCallback], file_info.get("status_compensation")
        ),
        snapshot_compensation=cast(
            Optional[FileHandlerCallback], file_info.get("snapshot_compensation")
        ),
        rollback_context=rollback_context,
    )
    result = pipeline_facade.rollback_failed_ingestion_sync(request)
    warnings.extend(result.warnings)
    return result.first_error


def _run_legacy_rollback_compensation(
    *,
    pipeline_facade: "KBPipelineCompatibilityFacade",
    page_operation: Any,
    file_info: FileHandlerResult,
    collection: str,
    url: str,
    warnings: list[str],
    legacy_callback: FileHandlerCallback,
    ingestion_result: Optional[IngestionResult] = None,
) -> Optional[str]:
    """Handle legacy monolithic rollback_on_failure (custom callbacks)."""
    rollback_context = _rollback_context_payload(file_info)

    def _compensate() -> None:
        try:
            _run_sync_file_handler_callback(
                legacy_callback,
                callback_name="rollback_on_failure",
                url=url,
                ingestion_result=ingestion_result,
            )
        except Exception as cleanup_error:  # noqa: BLE001
            raise _FileHandlerRollbackError(
                "rollback_on_failure",
                url,
                str(cleanup_error),
            ) from cleanup_error

    pipeline_facade.record_web_page_file_side_effect(
        page_operation,
        collection=collection,
        url=url,
        file_path=cast(Optional[str], file_info.get("file_path")),
        file_id=cast(Optional[str], file_info.get("file_id")),
        reason="rollback_on_failure",
        extra_payload=rollback_context,
        compensation=_compensate,
    )
    errors = pipeline_facade.compensate_web_page_file_side_effect(page_operation)
    if not errors:
        return None

    first_error = errors[0]
    cleanup_reason = (
        first_error.reason
        if isinstance(first_error, _FileHandlerRollbackError)
        else str(first_error)
    )
    message = f"File persistence rollback_on_failure failed for {url}: {cleanup_reason}"
    logger.warning(message)
    warnings.append(message)
    return cleanup_reason


def _run_legacy_persistent_file_compensation(
    *,
    pipeline_facade: "KBPipelineCompatibilityFacade",
    page_operation: Any,
    collection: str,
    url: str,
    copied_persistent_file: Optional[Path],
    file_info: Optional[FileHandlerResult],
    warnings: list[str],
) -> Optional[str]:
    if not copied_persistent_file or not copied_persistent_file.exists():
        return None
    if file_info and (
        "rollback_on_failure" in file_info or "file_compensation" in file_info
    ):
        return None

    def _compensate() -> None:
        copied_persistent_file.unlink()

    pipeline_facade.record_web_page_file_side_effect(
        page_operation,
        collection=collection,
        url=url,
        file_path=str(copied_persistent_file),
        file_id=cast(Optional[str], file_info.get("file_id")) if file_info else None,
        reason="legacy_persistent_file",
        extra_payload={"rollback_kind": "legacy_persistent_file"},
        compensation=_compensate,
    )
    errors = pipeline_facade.compensate_web_page_file_side_effect(page_operation)
    if not errors:
        logger.info(
            "Cleaned up persistent file due to ingestion failure: %s",
            copied_persistent_file,
        )
        return None

    cleanup_reason = str(errors[0])
    message = (
        f"Failed to clean up persistent file {copied_persistent_file}: {cleanup_reason}"
    )
    logger.warning(message)
    warnings.append(message)
    return cleanup_reason


def _looks_like_crawler_block(error: str) -> bool:
    """Heuristically detect WAF / anti-bot blocks from a crawl failure string."""
    normalized_error = error.lower()
    return any(marker in normalized_error for marker in _CRAWLER_BLOCK_ERROR_MARKERS)


def _get_pipeline_compatibility_facade() -> "KBPipelineCompatibilityFacade":
    """Return the coordinator-owned pipeline compatibility facade."""
    from ..kb import get_kb_coordinator

    return get_kb_coordinator().pipeline_compatibility


async def run_web_ingestion(
    collection: str,
    crawl_config: WebCrawlConfig,
    *,
    ingestion_config: Optional[IngestionConfig] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    user_id: Optional[int] = None,
    is_admin: Optional[bool] = None,
    file_handler: Optional[Callable[[Path, str, str, str], FileHandlerResult]] = None,
) -> WebIngestionResult:
    """Crawl a website and ingest all pages into the knowledge base."""
    return await _get_pipeline_compatibility_facade().run_web_ingestion(
        collection=collection,
        crawl_config=crawl_config,
        ingestion_config=ingestion_config,
        progress_callback=progress_callback,
        user_id=user_id,
        is_admin=is_admin,
        file_handler=file_handler,
    )


async def _run_web_ingestion_impl(
    collection: str,
    crawl_config: WebCrawlConfig,
    *,
    ingestion_config: Optional[IngestionConfig] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    user_id: Optional[int] = None,
    is_admin: Optional[bool] = None,
    file_handler: Optional[Callable[[Path, str, str, str], FileHandlerResult]] = None,
    pipeline_facade: Optional["KBPipelineCompatibilityFacade"] = None,
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
    rollback_failed_urls: dict[str, str] = {}

    # Normalize ingestion config
    ing_cfg = coerce_ingestion_config(ingestion_config)
    pipeline_facade = pipeline_facade or _get_pipeline_compatibility_facade()

    logger.info(
        "Starting web ingestion: collection=%s, start_url=%s",
        collection,
        crawl_config.start_url,
    )

    # Step 1: Crawl the website
    logger.info("Step 1: Crawling website")
    crawler = WebCrawler(crawl_config, progress_callback)

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
            message=f"Website crawling failed: {str(e)}",
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
        "Crawling completed: %s successful, %s failed", pages_crawled, pages_failed
    )

    # Step 2: Ingest each crawled page
    logger.info("Step 2: Ingesting crawled pages")

    # Create temporary directory for markdown files
    with tempfile.TemporaryDirectory(prefix="xagent_web_ingest_") as temp_dir:
        documents_created = 0
        successful_page_ingestions = 0
        total_chunks = 0
        total_embeddings = 0
        pages_started = 0

        loop = asyncio.get_event_loop()
        page_semaphore = asyncio.Semaphore(max(1, ing_cfg.page_ingest_concurrency))

        async def _process_page(i: int, crawl_result: CrawlResult) -> None:
            # Bound concurrency, then run the page. Split out so gather() can call
            # this directly (each page becomes its own semaphore-limited Task).
            async with page_semaphore:
                await _ingest_page(i, crawl_result)

        async def _ingest_page(i: int, crawl_result: CrawlResult) -> None:
            # Pages are independent; each runs its blocking ingestion in a worker
            # thread while sibling pages overlap (issue #912). The shared
            # accumulators / warnings / failed_urls below are only mutated between
            # awaits, so cooperative single-thread scheduling keeps them race-free
            # without locks; the cross-thread DB write hazard is guarded inside
            # collection_manager.
            nonlocal documents_created, successful_page_ingestions
            nonlocal total_chunks, total_embeddings, pages_started

            page_title = crawl_result.title or f"page_{i + 1}"
            with pipeline_facade.web_page_operation(
                collection=collection,
                url=crawl_result.url,
                title=page_title,
            ) as page_operation:
                # Progress callback. Pages complete out of order under concurrency,
                # so report a monotonic started-counter rather than the page index.
                # Safe without a lock: only mutated between awaits on the one loop.
                if progress_callback:
                    pages_started += 1
                    progress_callback(
                        f"Ingesting {pages_started}/{len(crawl_results)}: {crawl_result.url}",
                        pages_started,
                        len(crawl_results),
                    )

                try:
                    # Index-prefixed so concurrent same-titled pages don't collide
                    # on one temp path (torn read + shared doc_id when file_handler
                    # is None).
                    filename = sanitize_for_doc_id(page_title)
                    temp_file = Path(temp_dir) / f"{i}_{filename}.md"

                    with open(temp_file, "w", encoding="utf-8") as f:
                        # Add metadata header
                        f.write(f"# {page_title}\n\n")
                        f.write(f"**Source:** {crawl_result.url}\n\n")
                        f.write(
                            f"**Crawled:** {crawl_result.timestamp.isoformat()}\n\n"
                        )
                        f.write("---\n\n")
                        f.write(crawl_result.content_markdown)

                    logger.debug("Saved %s to %s", crawl_result.url, temp_file)

                    # Call file_handler if provided (for persistent storage and UploadedFile record)
                    final_file_path = temp_file
                    final_file_id = None
                    copied_persistent_file = None
                    file_info: Optional[FileHandlerResult] = None

                    if file_handler:
                        try:
                            file_info = file_handler(
                                temp_file,
                                page_title,
                                collection,
                                crawl_result.url,
                            )
                            if not file_info:
                                raise ValueError(
                                    "File handler returned no file information"
                                )
                            final_file_path = Path(
                                file_info.get("file_path") or temp_file
                            )
                            final_file_id = file_info.get("file_id")

                            has_per_boundary = _has_per_boundary_compensation(file_info)
                            file_compensation = cast(
                                Optional[FileHandlerCallback],
                                file_info.get("file_compensation"),
                            )
                            should_record_file_side_effect = (
                                file_compensation is not None
                                if has_per_boundary
                                else bool(final_file_path != temp_file or final_file_id)
                            )
                            if should_record_file_side_effect:
                                pipeline_facade.record_web_page_file_side_effect(
                                    page_operation,
                                    collection=collection,
                                    url=crawl_result.url,
                                    file_path=str(final_file_path),
                                    file_id=final_file_id,
                                    extra_payload=_rollback_context_payload(file_info),
                                    compensation=cast(
                                        Optional[Callable[[], None]],
                                        file_compensation,
                                    ),
                                )

                            # Track if we successfully copied a persistent file for cleanup
                            if (
                                final_file_path != temp_file
                                and final_file_path.exists()
                            ):
                                copied_persistent_file = final_file_path

                            logger.debug(
                                "File handler returned: path=%s, file_id=%s",
                                final_file_path,
                                final_file_id,
                            )
                        except Exception as e:
                            logger.exception(
                                "File handler failed for %s", crawl_result.url
                            )
                            failure_message = (
                                f"File persistence failed for {crawl_result.url}: {e}"
                            )
                            failed_urls[crawl_result.url] = failure_message
                            warnings.append(failure_message)
                            pipeline_facade.record_web_page_file_side_effect(
                                page_operation,
                                collection=collection,
                                url=crawl_result.url,
                                file_path=None,
                                file_id=None,
                                reason="file_handler_failed",
                            )
                            pipeline_facade.finish_web_page_operation(
                                page_operation,
                                status="error",
                                message=failure_message,
                            )
                            return

                    try:
                        # Ingest the file
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
                            )

                        # Copy the current ContextVars after the page child operation is active.
                        # This preserves user scope and lets document ingestion record into the same child.
                        request_context = copy_context()
                        ingest_result: IngestionResult = await loop.run_in_executor(
                            None, lambda: request_context.run(_ingest_file)
                        )

                        # Track statistics
                        if ingest_result.status == "success":
                            documents_created += 1
                            successful_page_ingestions += 1
                            total_chunks += ingest_result.chunk_count
                            total_embeddings += ingest_result.embedding_count
                            logger.info(
                                "Ingested %s: %s chunks, %s embeddings",
                                crawl_result.url,
                                ingest_result.chunk_count,
                                ingest_result.embedding_count,
                            )
                            pipeline_facade.finish_web_page_operation(
                                page_operation,
                                status="success",
                                message=ingest_result.message,
                            )
                            _run_file_handler_callback(
                                file_info,
                                "commit_on_success",
                                url=crawl_result.url,
                                warnings=warnings,
                                ingestion_result=ingest_result,
                            )
                            # Only clear temp file reference on success
                            copied_persistent_file = None
                        else:
                            failed_urls[crawl_result.url] = ingest_result.message
                            msg = (
                                f"Partial ingestion for {crawl_result.url}: "
                                f"{ingest_result.message}"
                            )
                            warnings.append(msg)
                            rollback_error = _run_file_handler_compensation(
                                pipeline_facade=pipeline_facade,
                                page_operation=page_operation,
                                file_info=file_info,
                                collection=collection,
                                url=crawl_result.url,
                                warnings=warnings,
                                ingestion_result=ingest_result,
                            )
                            legacy_cleanup_error = (
                                _run_legacy_persistent_file_compensation(
                                    pipeline_facade=pipeline_facade,
                                    page_operation=page_operation,
                                    collection=collection,
                                    url=crawl_result.url,
                                    copied_persistent_file=copied_persistent_file,
                                    file_info=file_info,
                                    warnings=warnings,
                                )
                            )
                            rollback_error = rollback_error or legacy_cleanup_error
                            if rollback_error:
                                rollback_failed_urls[crawl_result.url] = rollback_error
                            pipeline_facade.finish_web_page_operation(
                                page_operation,
                                status=ingest_result.status,
                                message=ingest_result.message,
                                side_effects_may_remain=bool(rollback_error),
                            )
                            copied_persistent_file = None

                    except Exception as e:
                        logger.exception("Failed to ingest %s", crawl_result.url)
                        failed_urls[crawl_result.url] = str(e)
                        failure_message = (
                            f"Failed to ingest {crawl_result.url}: {str(e)}"
                        )
                        warnings.append(failure_message)

                        rollback_error = _run_file_handler_compensation(
                            pipeline_facade=pipeline_facade,
                            page_operation=page_operation,
                            file_info=file_info,
                            collection=collection,
                            url=crawl_result.url,
                            warnings=warnings,
                        )
                        if rollback_error:
                            rollback_failed_urls[crawl_result.url] = rollback_error

                        # Legacy cleanup for handlers that only returned a file path.
                        legacy_cleanup_error = _run_legacy_persistent_file_compensation(
                            pipeline_facade=pipeline_facade,
                            page_operation=page_operation,
                            collection=collection,
                            url=crawl_result.url,
                            copied_persistent_file=copied_persistent_file,
                            file_info=file_info,
                            warnings=warnings,
                        )
                        rollback_error = rollback_error or legacy_cleanup_error
                        if rollback_error:
                            rollback_failed_urls[crawl_result.url] = rollback_error
                        copied_persistent_file = None
                        pipeline_facade.finish_web_page_operation(
                            page_operation,
                            status="error",
                            message=failure_message,
                            side_effects_may_remain=bool(rollback_error),
                        )

                except Exception as e:
                    logger.exception("Failed to ingest %s", crawl_result.url)
                    failed_urls[crawl_result.url] = str(e)
                    failure_message = f"Failed to ingest {crawl_result.url}: {str(e)}"
                    warnings.append(failure_message)
                    pipeline_facade.finish_web_page_operation(
                        page_operation,
                        status="error",
                        message=failure_message,
                    )

        pending_pages = [
            (i, crawl_result)
            for i, crawl_result in enumerate(crawl_results)
            if crawl_result.status == "success"
        ]
        # return_exceptions=True so one page's unexpected error can't abort the
        # others; per-page failures are already recorded inside _process_page.
        # gather() runs each page as its own Task, which is what gives per-page
        # ContextVar isolation — don't collapse this into shared-context awaits.
        page_outcomes = await asyncio.gather(
            *(_process_page(i, crawl_result) for i, crawl_result in pending_pages),
            return_exceptions=True,
        )
        for (_, crawl_result), outcome in zip(pending_pages, page_outcomes):
            if isinstance(outcome, Exception):
                logger.error(
                    "Unexpected error ingesting %s: %s",
                    crawl_result.url,
                    outcome,
                    exc_info=outcome,
                )
                failure_message = f"Failed to ingest {crawl_result.url}: {outcome}"
                failed_urls[crawl_result.url] = failure_message
                warnings.append(failure_message)

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

    has_successful_ingestion = successful_page_ingestions > 0
    if not has_successful_ingestion and total_failures > 0:
        status = "error"
    elif total_failures > 0:
        status = "partial"
    else:
        status = "success"
    if rollback_failed_urls:
        status = "error"

    # A crawl can end with zero failures and still not be what the user asked
    # for: every link the start page offered was refused by the site. Configured
    # stops (page cap, depth, deliberate filtering) are not this case, which is
    # why the reason comes from the crawler rather than from link counts.
    #
    # This deliberately does not touch `status`. The pages that landed were
    # ingested and published, which is what "success" means to every consumer
    # of that enum - the REST handler returns 200 and both frontend call sites
    # treat anything else as a hard failure. The blocked crawl is a different
    # fact about the same run, so it travels as its own flag plus a warning.
    crawl_stats = crawler.get_statistics()
    blocked_stop = (
        status == "success"
        and crawl_stats.get("stop_reason") == STOPPED_NO_ELIGIBLE_LINKS
    )

    crawled_urls_list = [r.url for r in crawl_results if r.status == "success"]

    # Build a status-aware message. Previously this was unconditionally
    # "Web ingestion completed: ..." even on error, which produced the
    # "red error toast + green-toned 'completed' text" UX in the frontend
    # whenever every crawl attempt got blocked. On error/partial we now
    # check all failures for anti-bot/WAF signals and otherwise surface
    # the first failing URL and its reason so the user sees something
    # actionable.
    if rollback_failed_urls:
        first_url, first_err = next(iter(rollback_failed_urls.items()))
        message = f"Web ingestion rollback failed for {first_url}: {first_err}"
    elif (status == "error" or status == "partial") and failed_urls:
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
    elif blocked_stop:
        message = _blocked_crawl_explanation(
            crawl_stats, documents_created, pages_crawled
        )
        warnings = [*warnings, message]
    else:
        message = (
            f"Web ingestion completed: {documents_created} documents, "
            f"{total_chunks} chunks, {total_embeddings} embeddings"
        )

    result = WebIngestionResult(
        status=status,
        crawl_blocked_by_site=blocked_stop,
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
        side_effects_may_remain=bool(rollback_failed_urls),
    )

    logger.info(
        "Web ingestion completed: %s, %s documents, %sms",
        result.status,
        documents_created,
        elapsed_ms,
    )

    return result
