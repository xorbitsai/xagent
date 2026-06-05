"""Unit tests for web ingestion pipeline."""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import (
    IngestionConfig,
    IngestionResult,
    WebCrawlConfig,
)
from xagent.core.tools.core.RAG_tools.pipelines.web_ingestion import (
    _callback_accepts_ingestion_result,
    run_web_ingestion,
)
from xagent.core.tools.core.RAG_tools.utils.string_utils import sanitize_for_doc_id
from xagent.core.tools.core.RAG_tools.utils.user_scope import user_scope_context


class TestWebIngestionPipeline:
    """Test web ingestion pipeline functionality."""

    @pytest.fixture
    def crawl_config(self):
        """Create a test crawl configuration."""
        return WebCrawlConfig(
            start_url="https://example.com",
            max_pages=3,
            max_depth=1,
            concurrent_requests=1,
            request_delay=0,
        )

    @pytest.fixture
    def ingestion_config(self):
        """Create a test ingestion configuration."""
        return IngestionConfig(
            chunk_size=500,
            chunk_overlap=100,
        )

    def test_callback_accepts_ingestion_result_requires_one_arg_callable(self):
        def no_args():
            return None

        def one_required(result):
            return result

        def one_optional(result=None):
            return result

        def two_required(result, extra):
            return result, extra

        def one_required_one_optional(result, extra=None):
            return result, extra

        def varargs(*args):
            return args

        def keyword_only(*, result):
            return result

        assert not _callback_accepts_ingestion_result(no_args)
        assert _callback_accepts_ingestion_result(one_required)
        assert _callback_accepts_ingestion_result(one_optional)
        assert not _callback_accepts_ingestion_result(two_required)
        assert _callback_accepts_ingestion_result(one_required_one_optional)
        assert _callback_accepts_ingestion_result(varargs)
        assert not _callback_accepts_ingestion_result(keyword_only)

    @pytest.mark.asyncio
    async def test_successful_web_ingestion(self, crawl_config, ingestion_config):
        """Test successful web ingestion."""
        # Mock crawler results
        mock_crawl_results = [
            MagicMock(
                url="https://example.com/page1",
                title="Page 1",
                content_markdown="# Page 1\n\nContent for page 1.",
                status="success",
                depth=0,
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                content_length=50,
            ),
            MagicMock(
                url="https://example.com/page2",
                title="Page 2",
                content_markdown="# Page 2\n\nContent for page 2.",
                status="success",
                depth=0,
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                content_length=50,
            ),
        ]

        # Mock ingestion results
        mock_ingestion_result = IngestionResult(
            status="success",
            doc_id="test_doc_id",
            parse_hash="test_hash",
            chunk_count=5,
            embedding_count=5,
            vector_count=5,
            completed_steps=[],
            failed_step=None,
            message="Success",
            warnings=[],
        )

        # Mock the crawler and document ingestion
        with patch(
            "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.WebCrawler"
        ) as mock_crawler_class:
            mock_crawler = MagicMock()
            mock_crawler.crawl = AsyncMock(return_value=mock_crawl_results)
            mock_crawler.total_urls_found = 2
            mock_crawler.failed_urls = {}
            mock_crawler_class.return_value = mock_crawler

            with patch(
                "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.run_document_ingestion",
                return_value=mock_ingestion_result,
            ):
                result = await run_web_ingestion(
                    collection="test_collection",
                    crawl_config=crawl_config,
                    ingestion_config=ingestion_config,
                )

        # Verify result
        assert result.status == "success"
        assert result.collection == "test_collection"
        assert result.pages_crawled == 2
        assert result.documents_created == 2
        assert result.chunks_created == 10  # 5 per page
        assert result.embeddings_created == 10

    @pytest.mark.asyncio
    async def test_crawl_failure(self, crawl_config, ingestion_config):
        """Test handling of crawl failure."""
        # Mock crawler exception
        with patch(
            "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.WebCrawler"
        ) as mock_crawler_class:
            mock_crawler = MagicMock()
            mock_crawler.crawl = AsyncMock(side_effect=Exception("Crawl failed"))
            mock_crawler_class.return_value = mock_crawler

            result = await run_web_ingestion(
                collection="test_collection",
                crawl_config=crawl_config,
                ingestion_config=ingestion_config,
            )

        # Should return error status
        assert result.status == "error"
        assert result.pages_crawled == 0
        assert result.documents_created == 0
        assert "Crawl failed" in result.message

    @pytest.mark.asyncio
    async def test_error_message_uses_first_failed_url(
        self, crawl_config, ingestion_config
    ):
        """When the crawler returns normally but every URL was blocked
        (e.g. all 403s), the result message must reflect the actual
        failure -- NOT the misleading 'Web ingestion completed' string
        that previously appeared with status=error.

        Regression test for the UX bug where the frontend showed a red
        error toast carrying "completed: 0 documents" text whenever a
        site was WAF-blocked.
        """
        with patch(
            "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.WebCrawler"
        ) as mock_crawler_class:
            # Crawler returns successfully but with empty results +
            # populated failed_urls (this is the path that previously
            # produced the misleading message)
            mock_crawler = MagicMock()
            mock_crawler.crawl = AsyncMock(return_value=[])
            mock_crawler.failed_urls = {
                "https://www.detrack.com": "HTTP 403",
                "https://www.detrack.com/about": "HTTP 403",
            }
            mock_crawler.total_urls_found = 0
            mock_crawler_class.return_value = mock_crawler

            result = await run_web_ingestion(
                collection="test_collection",
                crawl_config=crawl_config,
                ingestion_config=ingestion_config,
            )

        assert result.status == "error"
        assert result.documents_created == 0
        # The message must surface the actual failure, not "completed"
        assert "completed" not in result.message.lower()
        assert (
            result.message
            == "Web ingestion failed. The target website is blocking access "
            "to automated crawlers. Please use a different method to create "
            "your KB."
        )

    @pytest.mark.asyncio
    async def test_partial_ingestion_failure(self, crawl_config, ingestion_config):
        """Test handling of partial ingestion failures."""
        # Mock crawl results
        mock_crawl_results = [
            MagicMock(
                url="https://example.com/page1",
                title="Page 1",
                content_markdown="# Page 1\n\nContent",
                status="success",
                depth=0,
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                content_length=30,
            ),
            MagicMock(
                url="https://example.com/page2",
                title="Page 2",
                content_markdown="# Page 2\n\nContent",
                status="success",
                depth=0,
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                content_length=30,
            ),
        ]

        # Mock mixed ingestion results
        success_result = IngestionResult(
            status="success",
            doc_id="doc1",
            parse_hash="hash1",
            chunk_count=5,
            embedding_count=5,
            vector_count=5,
            completed_steps=[],
            failed_step=None,
            message="Success",
            warnings=[],
        )

        error_result = IngestionResult(
            status="error",
            doc_id="doc2",
            parse_hash="hash2",
            chunk_count=0,
            embedding_count=0,
            vector_count=0,
            completed_steps=[],
            failed_step="parse",
            message="Parse failed",
            warnings=[],
        )

        with patch(
            "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.WebCrawler"
        ) as mock_crawler_class:
            mock_crawler = MagicMock()
            mock_crawler.crawl = AsyncMock(return_value=mock_crawl_results)
            mock_crawler.total_urls_found = 2
            mock_crawler.failed_urls = {}
            mock_crawler_class.return_value = mock_crawler

            with patch(
                "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.run_document_ingestion",
                side_effect=[success_result, error_result],
            ):
                result = await run_web_ingestion(
                    collection="test_collection",
                    crawl_config=crawl_config,
                    ingestion_config=ingestion_config,
                )

        # Should return partial status
        assert result.status == "partial"
        assert result.pages_crawled == 2
        assert result.documents_created == 1
        assert result.pages_failed == 1
        assert "partial" in result.message.lower()
        assert "completed" not in result.message.lower()
        assert "https://example.com/page2" in result.message
        assert "Parse failed" in result.message
        assert len(result.failed_urls) == 1
        assert "https://example.com/page2" in result.failed_urls

    @pytest.mark.asyncio
    async def test_partial_message_is_human_friendly_for_crawler_blocks(
        self, crawl_config, ingestion_config
    ):
        """Partial results should avoid exposing raw 403 crawl errors."""
        mock_crawl_results = [
            MagicMock(
                url="https://example.com/page1",
                title="Page 1",
                content_markdown="# Page 1\n\nContent",
                status="success",
                depth=0,
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                content_length=30,
            )
        ]

        success_result = IngestionResult(
            status="success",
            doc_id="doc1",
            parse_hash="hash1",
            chunk_count=5,
            embedding_count=5,
            vector_count=5,
            completed_steps=[],
            failed_step=None,
            message="Success",
            warnings=[],
        )

        with patch(
            "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.WebCrawler"
        ) as mock_crawler_class:
            mock_crawler = MagicMock()
            mock_crawler.crawl = AsyncMock(return_value=mock_crawl_results)
            mock_crawler.total_urls_found = 2
            mock_crawler.failed_urls = {"https://example.com/blocked": "HTTP 403"}
            mock_crawler_class.return_value = mock_crawler

            with patch(
                "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.run_document_ingestion",
                return_value=success_result,
            ):
                result = await run_web_ingestion(
                    collection="test_collection",
                    crawl_config=crawl_config,
                    ingestion_config=ingestion_config,
                )

        assert result.status == "partial"
        assert "HTTP 403" not in result.message
        assert "automated crawlers" in result.message
        assert "different method to create your KB" in result.message
        assert "https://example.com/blocked" in result.message

    @pytest.mark.asyncio
    async def test_error_message_checks_all_failures_for_crawler_blocks(
        self, crawl_config, ingestion_config
    ):
        """Crawler block guidance should not depend on the first failure only."""
        with patch(
            "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.WebCrawler"
        ) as mock_crawler_class:
            mock_crawler = MagicMock()
            mock_crawler.crawl = AsyncMock(return_value=[])
            mock_crawler.failed_urls = {
                "https://example.com/missing": "HTTP 404",
                "https://example.com/blocked": "HTTP 403",
            }
            mock_crawler.total_urls_found = 0
            mock_crawler_class.return_value = mock_crawler

            result = await run_web_ingestion(
                collection="test_collection",
                crawl_config=crawl_config,
                ingestion_config=ingestion_config,
            )

        assert result.status == "error"
        assert (
            result.message
            == "Web ingestion failed. The target website is blocking access "
            "to automated crawlers. Please use a different method to create "
            "your KB."
        )

    @pytest.mark.asyncio
    async def test_partial_message_does_not_treat_ingestion_429_as_crawler_block(
        self, crawl_config, ingestion_config
    ):
        """Downstream ingestion failures must not be labeled as site anti-bot blocks."""
        mock_crawl_results = [
            MagicMock(
                url="https://example.com/page1",
                title="Page 1",
                content_markdown="# Page 1\n\nContent",
                status="success",
                depth=0,
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                content_length=30,
            ),
            MagicMock(
                url="https://example.com/page2",
                title="Page 2",
                content_markdown="# Page 2\n\nContent",
                status="success",
                depth=0,
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                content_length=30,
            ),
        ]

        success_result = IngestionResult(
            status="success",
            doc_id="doc1",
            parse_hash="hash1",
            chunk_count=5,
            embedding_count=5,
            vector_count=5,
            completed_steps=[],
            failed_step=None,
            message="Success",
            warnings=[],
        )

        error_result = IngestionResult(
            status="error",
            doc_id="doc2",
            parse_hash="hash2",
            chunk_count=0,
            embedding_count=0,
            vector_count=0,
            completed_steps=[],
            failed_step="embed",
            message="HTTP 429 from embedding API",
            warnings=[],
        )

        with patch(
            "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.WebCrawler"
        ) as mock_crawler_class:
            mock_crawler = MagicMock()
            mock_crawler.crawl = AsyncMock(return_value=mock_crawl_results)
            mock_crawler.total_urls_found = 2
            mock_crawler.failed_urls = {}
            mock_crawler_class.return_value = mock_crawler

            with patch(
                "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.run_document_ingestion",
                side_effect=[success_result, error_result],
            ):
                result = await run_web_ingestion(
                    collection="test_collection",
                    crawl_config=crawl_config,
                    ingestion_config=ingestion_config,
                )

        assert result.status == "partial"
        assert "automated crawlers" not in result.message
        assert "https://example.com/page2" in result.message
        assert "HTTP 429 from embedding API" in result.message

    @pytest.mark.asyncio
    async def test_empty_crawl_results(self, crawl_config, ingestion_config):
        """Test handling of empty crawl results."""
        with patch(
            "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.WebCrawler"
        ) as mock_crawler_class:
            mock_crawler = MagicMock()
            mock_crawler.crawl = AsyncMock(return_value=[])
            mock_crawler.total_urls_found = 0
            mock_crawler.failed_urls = {}
            mock_crawler_class.return_value = mock_crawler

            result = await run_web_ingestion(
                collection="test_collection",
                crawl_config=crawl_config,
                ingestion_config=ingestion_config,
            )

        # Should handle gracefully
        assert result.status == "success"
        assert result.pages_crawled == 0
        assert result.documents_created == 0

    @pytest.mark.asyncio
    async def test_ingestion_config_defaults(self, crawl_config):
        """Test that ingestion config defaults are applied."""
        # Mock successful crawl and ingestion
        mock_crawl_results = [
            MagicMock(
                url="https://example.com/page1",
                title="Page 1",
                content_markdown="# Page 1\n\nContent",
                status="success",
                depth=0,
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                content_length=30,
            )
        ]

        mock_ingestion_result = IngestionResult(
            status="success",
            doc_id="doc1",
            parse_hash="hash1",
            chunk_count=1,
            embedding_count=1,
            vector_count=1,
            completed_steps=[],
            failed_step=None,
            message="Success",
            warnings=[],
        )

        with patch(
            "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.WebCrawler"
        ) as mock_crawler_class:
            mock_crawler = MagicMock()
            mock_crawler.crawl = AsyncMock(return_value=mock_crawl_results)
            mock_crawler.total_urls_found = 1
            mock_crawler.failed_urls = {}
            mock_crawler_class.return_value = mock_crawler

            with patch(
                "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.run_document_ingestion",
                return_value=mock_ingestion_result,
            ) as mock_ingest:
                # Call without ingestion config
                await run_web_ingestion(
                    collection="test_collection",
                    crawl_config=crawl_config,
                )

                # Verify default config was used
                mock_ingest.assert_called_once()
                call_args = mock_ingest.call_args
                assert call_args[1]["ingestion_config"] is not None

    @pytest.mark.asyncio
    async def test_progress_callback(self, crawl_config, ingestion_config):
        """Test progress callback during ingestion."""
        progress_updates = []

        def progress_callback(message, completed, total):
            progress_updates.append((message, completed, total))

        mock_crawl_results = [
            MagicMock(
                url=f"https://example.com/page{i}",
                title=f"Page {i}",
                content_markdown=f"# Page {i}\n\nContent",
                status="success",
                depth=0,
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                content_length=30,
            )
            for i in range(3)
        ]

        mock_ingestion_result = IngestionResult(
            status="success",
            doc_id="doc1",
            parse_hash="hash1",
            chunk_count=1,
            embedding_count=1,
            vector_count=1,
            completed_steps=[],
            failed_step=None,
            message="Success",
            warnings=[],
        )

        with patch(
            "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.WebCrawler"
        ) as mock_crawler_class:
            mock_crawler = MagicMock()
            mock_crawler.crawl = AsyncMock(return_value=mock_crawl_results)
            mock_crawler.total_urls_found = 3
            mock_crawler.failed_urls = {}
            mock_crawler_class.return_value = mock_crawler

            with patch(
                "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.run_document_ingestion",
                return_value=mock_ingestion_result,
            ):
                await run_web_ingestion(
                    collection="test_collection",
                    crawl_config=crawl_config,
                    ingestion_config=ingestion_config,
                    progress_callback=progress_callback,
                )

        # Progress callback should have been called
        assert len(progress_updates) == 3
        assert all(len(update) == 3 for update in progress_updates)

    @pytest.mark.asyncio
    async def test_elapsed_time_tracking(self, crawl_config, ingestion_config):
        """Test that elapsed time is tracked."""
        mock_crawl_results = [
            MagicMock(
                url="https://example.com/page1",
                title="Page 1",
                content_markdown="# Page 1\n\nContent",
                status="success",
                depth=0,
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                content_length=30,
            )
        ]

        mock_ingestion_result = IngestionResult(
            status="success",
            doc_id="doc1",
            parse_hash="hash1",
            chunk_count=1,
            embedding_count=1,
            vector_count=1,
            completed_steps=[],
            failed_step=None,
            message="Success",
            warnings=[],
        )

        with patch(
            "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.WebCrawler"
        ) as mock_crawler_class:
            mock_crawler = MagicMock()
            mock_crawler.crawl = AsyncMock(return_value=mock_crawl_results)
            mock_crawler.total_urls_found = 1
            mock_crawler.failed_urls = {}
            mock_crawler_class.return_value = mock_crawler

            with patch(
                "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.run_document_ingestion",
                return_value=mock_ingestion_result,
            ):
                result = await run_web_ingestion(
                    collection="test_collection",
                    crawl_config=crawl_config,
                    ingestion_config=ingestion_config,
                )

        # Elapsed time should be tracked
        assert result.elapsed_time_ms >= 0


def test_sanitize_for_doc_id_behavior() -> None:
    """Test sanitize_for_doc_id behavior used by web ingestion."""
    # Replaces spaces and dots with underscores.
    assert sanitize_for_doc_id("report 2024.pdf") == "report_2024_pdf"

    # Path traversal-like input is normalized to safe token.
    assert sanitize_for_doc_id("../../etc/passwd") == "etc_passwd"

    # Non-allowed symbols collapse into underscores and trim boundaries.
    assert sanitize_for_doc_id("  .test.  ") == "test"

    # Empty input falls back to generated short identifier.
    fallback = sanitize_for_doc_id("")
    assert len(fallback) == 8
    assert fallback.isalnum()


class TestWebIngestionFileHandler:
    """Test file_handler callback functionality for persistent storage."""

    @pytest.fixture
    def crawl_config(self):
        """Create a test crawl configuration."""
        return WebCrawlConfig(
            start_url="https://example.com",
            max_pages=1,
            max_depth=1,
            concurrent_requests=1,
            request_delay=0,
        )

    @pytest.fixture
    def ingestion_config(self):
        """Create a test ingestion configuration."""
        return IngestionConfig(
            chunk_size=500,
            chunk_overlap=100,
        )

    @pytest.mark.asyncio
    async def test_file_handler_is_called(self, crawl_config, ingestion_config):
        """Test that file_handler callback is called for each crawled page."""
        mock_crawl_results = [
            MagicMock(
                url="https://example.com/page1",
                title="Test Page",
                content_markdown="# Test Page\n\nContent",
                status="success",
                depth=0,
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                content_length=30,
            )
        ]

        mock_ingestion_result = IngestionResult(
            status="success",
            doc_id="doc1",
            parse_hash="hash1",
            chunk_count=1,
            embedding_count=1,
            vector_count=1,
            completed_steps=[],
            failed_step=None,
            message="Success",
            warnings=[],
        )

        # Track file_handler calls
        file_handler_calls = []

        def mock_file_handler(
            temp_file_path: Path, title: str, collection: str, url: str
        ) -> dict[str, Any]:
            """Mock file handler that tracks calls and returns test data."""
            file_handler_calls.append(
                {
                    "temp_file_path": temp_file_path,
                    "title": title,
                    "collection": collection,
                    "url": url,
                }
            )
            return {
                "file_path": "/fake/persistent/path.md",
                "file_id": "test-file-id-123",
            }

        with patch(
            "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.WebCrawler"
        ) as mock_crawler_class:
            mock_crawler = MagicMock()
            mock_crawler.crawl = AsyncMock(return_value=mock_crawl_results)
            mock_crawler.total_urls_found = 1
            mock_crawler.failed_urls = {}
            mock_crawler_class.return_value = mock_crawler

            with patch(
                "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.run_document_ingestion",
                return_value=mock_ingestion_result,
            ) as mock_ingest:
                await run_web_ingestion(
                    collection="test_collection",
                    crawl_config=crawl_config,
                    ingestion_config=ingestion_config,
                    file_handler=mock_file_handler,
                )

                # Verify file_handler was called
                assert len(file_handler_calls) == 1
                assert file_handler_calls[0]["title"] == "Test Page"
                assert file_handler_calls[0]["collection"] == "test_collection"
                # Note: temp_file_path no longer exists at this point (temp dir cleaned up)
                assert "xagent_web_ingest" in str(
                    file_handler_calls[0]["temp_file_path"]
                )

                # Verify run_document_ingestion was called with file_id
                mock_ingest.assert_called_once()
                call_kwargs = mock_ingest.call_args[1]
                assert call_kwargs["file_id"] == "test-file-id-123"
                assert call_kwargs["source_path"] == "/fake/persistent/path.md"

    @pytest.mark.asyncio
    async def test_file_handler_failure_does_not_fallback_to_temp_file(
        self, crawl_config, ingestion_config
    ):
        """Test that file_handler failures do not ingest temporary files."""
        mock_crawl_results = [
            MagicMock(
                url="https://example.com/page1",
                title="Test Page",
                content_markdown="# Test Page\n\nContent",
                status="success",
                depth=0,
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                content_length=30,
            )
        ]

        mock_ingestion_result = IngestionResult(
            status="success",
            doc_id="doc1",
            parse_hash="hash1",
            chunk_count=1,
            embedding_count=1,
            vector_count=1,
            completed_steps=[],
            failed_step=None,
            message="Success",
            warnings=[],
        )

        # File handler that raises an exception
        def failing_file_handler(
            temp_file_path: Path, title: str, collection: str, url: str
        ) -> dict[str, Any]:
            raise Exception("File handler failed!")

        with patch(
            "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.WebCrawler"
        ) as mock_crawler_class:
            mock_crawler = MagicMock()
            mock_crawler.crawl = AsyncMock(return_value=mock_crawl_results)
            mock_crawler.total_urls_found = 1
            mock_crawler.failed_urls = {}
            mock_crawler_class.return_value = mock_crawler

            with patch(
                "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.run_document_ingestion",
                return_value=mock_ingestion_result,
            ) as mock_ingest:
                result = await run_web_ingestion(
                    collection="test_collection",
                    crawl_config=crawl_config,
                    ingestion_config=ingestion_config,
                    file_handler=failing_file_handler,
                )

                assert result.status == "error"
                assert result.documents_created == 0
                assert result.pages_failed == 1
                assert "https://example.com/page1" in result.failed_urls

                mock_ingest.assert_not_called()

    @pytest.mark.asyncio
    async def test_omitted_scope_falls_back_to_context(
        self, crawl_config, ingestion_config
    ):
        """Web ingestion should pass request-scoped user context to document ingestion."""
        mock_crawl_results = [
            MagicMock(
                url="https://example.com/page1",
                title="Page 1",
                content_markdown="# Page 1\n\nContent.",
                status="success",
                depth=0,
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                content_length=20,
            )
        ]
        mock_ingestion_result = IngestionResult(
            status="success",
            doc_id="test_doc_id",
            parse_hash="test_hash",
            chunk_count=1,
            embedding_count=1,
            vector_count=1,
            completed_steps=[],
            failed_step=None,
            message="Success",
            warnings=[],
        )

        with patch(
            "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.WebCrawler"
        ) as mock_crawler_class:
            mock_crawler = MagicMock()
            mock_crawler.crawl = AsyncMock(return_value=mock_crawl_results)
            mock_crawler.total_urls_found = 1
            mock_crawler.failed_urls = {}
            mock_crawler_class.return_value = mock_crawler

            with patch(
                "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion."
                "run_document_ingestion",
                return_value=mock_ingestion_result,
            ) as mock_ingest:
                with user_scope_context(user_id=47, is_admin=True):
                    result = await run_web_ingestion(
                        collection="test_collection",
                        crawl_config=crawl_config,
                        ingestion_config=ingestion_config,
                    )

        assert result.status == "success"
        call_kwargs = mock_ingest.call_args[1]
        assert call_kwargs["user_id"] == 47
        assert call_kwargs["is_admin"] is True

    @pytest.mark.asyncio
    async def test_no_file_handler_uses_temp_files(
        self, crawl_config, ingestion_config
    ):
        """Test that without file_handler, temporary files are used."""
        mock_crawl_results = [
            MagicMock(
                url="https://example.com/page1",
                title="Test Page",
                content_markdown="# Test Page\n\nContent",
                status="success",
                depth=0,
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                content_length=30,
            )
        ]

        mock_ingestion_result = IngestionResult(
            status="success",
            doc_id="doc1",
            parse_hash="hash1",
            chunk_count=1,
            embedding_count=1,
            vector_count=1,
            completed_steps=[],
            failed_step=None,
            message="Success",
            warnings=[],
        )

        with patch(
            "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.WebCrawler"
        ) as mock_crawler_class:
            mock_crawler = MagicMock()
            mock_crawler.crawl = AsyncMock(return_value=mock_crawl_results)
            mock_crawler.total_urls_found = 1
            mock_crawler.failed_urls = {}
            mock_crawler_class.return_value = mock_crawler

            with patch(
                "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.run_document_ingestion",
                return_value=mock_ingestion_result,
            ) as mock_ingest:
                # Call without file_handler
                result = await run_web_ingestion(
                    collection="test_collection",
                    crawl_config=crawl_config,
                    ingestion_config=ingestion_config,
                )

                # Verify ingestion succeeded
                assert result.status == "success"

                # Verify run_document_ingestion was called without file_id
                mock_ingest.assert_called_once()
                call_kwargs = mock_ingest.call_args[1]
                assert call_kwargs["file_id"] is None
                # source_path should be the temporary file path
                assert "xagent_web_ingest" in call_kwargs["source_path"]

    @pytest.mark.asyncio
    async def test_file_handler_rollback_runs_on_ingestion_error_result(
        self, crawl_config, ingestion_config
    ):
        """File persistence should be compensated when per-page ingestion fails."""
        events: list[tuple[str, Optional[IngestionResult]]] = []
        mock_crawl_results = [
            MagicMock(
                url="https://example.com/page1",
                title="Page 1",
                content_markdown="# Page 1\n\nContent.",
                status="success",
                depth=0,
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                content_length=20,
            )
        ]
        failed_ingestion = IngestionResult(
            status="error",
            doc_id="doc1",
            parse_hash="hash1",
            message="embedding failed",
        )

        def file_handler(
            temp_file_path: Path, title: str, collection: str, url: str
        ) -> dict[str, Any]:
            return {
                "file_path": str(temp_file_path),
                "file_id": "file-1",
                "rollback_on_failure": lambda result=None: events.append(
                    ("rollback", result)
                ),
                "commit_on_success": lambda result=None: events.append(
                    ("commit", result)
                ),
            }

        with patch(
            "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.WebCrawler"
        ) as mock_crawler_class:
            mock_crawler = MagicMock()
            mock_crawler.crawl = AsyncMock(return_value=mock_crawl_results)
            mock_crawler.total_urls_found = 1
            mock_crawler.failed_urls = {}
            mock_crawler_class.return_value = mock_crawler

            with patch(
                "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion."
                "run_document_ingestion",
                return_value=failed_ingestion,
            ):
                result = await run_web_ingestion(
                    collection="test_collection",
                    crawl_config=crawl_config,
                    ingestion_config=ingestion_config,
                    file_handler=file_handler,
                )

        assert result.status == "error"
        assert events == [("rollback", failed_ingestion)]

    @pytest.mark.asyncio
    async def test_file_handler_rollback_runs_on_ingestion_exception(
        self, crawl_config, ingestion_config
    ):
        """Raised per-page ingestion failures should also compensate persistence."""
        events: list[tuple[str, Optional[IngestionResult]]] = []
        mock_crawl_results = [
            MagicMock(
                url="https://example.com/page1",
                title="Page 1",
                content_markdown="# Page 1\n\nContent.",
                status="success",
                depth=0,
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                content_length=20,
            )
        ]

        def file_handler(
            temp_file_path: Path, title: str, collection: str, url: str
        ) -> dict[str, Any]:
            return {
                "file_path": str(temp_file_path),
                "file_id": "file-1",
                "rollback_on_failure": lambda result=None: events.append(
                    ("rollback", result)
                ),
                "commit_on_success": lambda result=None: events.append(
                    ("commit", result)
                ),
            }

        with patch(
            "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.WebCrawler"
        ) as mock_crawler_class:
            mock_crawler = MagicMock()
            mock_crawler.crawl = AsyncMock(return_value=mock_crawl_results)
            mock_crawler.total_urls_found = 1
            mock_crawler.failed_urls = {}
            mock_crawler_class.return_value = mock_crawler

            with patch(
                "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion."
                "run_document_ingestion",
                side_effect=RuntimeError("boom"),
            ):
                result = await run_web_ingestion(
                    collection="test_collection",
                    crawl_config=crawl_config,
                    ingestion_config=ingestion_config,
                    file_handler=file_handler,
                )

        assert result.status == "error"
        assert events == [("rollback", None)]

    @pytest.mark.asyncio
    async def test_file_handler_rollback_failure_forces_error_status(
        self, crawl_config, ingestion_config
    ):
        """A failed compensating rollback is surfaced as an overall error."""
        mock_crawl_results = [
            MagicMock(
                url="https://example.com/page1",
                title="Page 1",
                content_markdown="# Page 1\n\nContent.",
                status="success",
                depth=0,
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                content_length=20,
            ),
            MagicMock(
                url="https://example.com/page2",
                title="Page 2",
                content_markdown="# Page 2\n\nContent.",
                status="success",
                depth=0,
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                content_length=20,
            ),
        ]
        successful_ingestion = IngestionResult(
            status="success",
            doc_id="doc1",
            parse_hash="hash1",
            chunk_count=1,
            embedding_count=1,
            vector_count=1,
            message="ok",
        )
        failed_ingestion = IngestionResult(
            status="error",
            doc_id="doc2",
            parse_hash="hash2",
            message="embedding failed",
        )

        def rollback(_result=None):
            raise RuntimeError("rollback exploded")

        def file_handler(
            temp_file_path: Path, title: str, collection: str, url: str
        ) -> dict[str, Any]:
            return {
                "file_path": str(temp_file_path),
                "file_id": "file-1",
                "rollback_on_failure": rollback,
            }

        with patch(
            "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.WebCrawler"
        ) as mock_crawler_class:
            mock_crawler = MagicMock()
            mock_crawler.crawl = AsyncMock(return_value=mock_crawl_results)
            mock_crawler.total_urls_found = 2
            mock_crawler.failed_urls = {}
            mock_crawler_class.return_value = mock_crawler

            with patch(
                "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion."
                "run_document_ingestion",
                side_effect=[successful_ingestion, failed_ingestion],
            ):
                result = await run_web_ingestion(
                    collection="test_collection",
                    crawl_config=crawl_config,
                    ingestion_config=ingestion_config,
                    file_handler=file_handler,
                )

        assert result.status == "error"
        assert result.documents_created == 1
        assert result.side_effects_may_remain is True
        assert (
            result.message
            == "Web ingestion rollback failed for https://example.com/page2: "
            "rollback exploded"
        )
        assert any("rollback_on_failure failed" in item for item in result.warnings)

    @pytest.mark.asyncio
    async def test_file_handler_commit_runs_on_ingestion_success(
        self, crawl_config, ingestion_config
    ):
        """Rollback resources should be finalized after successful ingestion."""
        events: list[tuple[str, Optional[IngestionResult]]] = []
        mock_crawl_results = [
            MagicMock(
                url="https://example.com/page1",
                title="Page 1",
                content_markdown="# Page 1\n\nContent.",
                status="success",
                depth=0,
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                content_length=20,
            )
        ]
        successful_ingestion = IngestionResult(
            status="success",
            doc_id="doc1",
            parse_hash="hash1",
            chunk_count=1,
            embedding_count=1,
            vector_count=1,
            message="ok",
        )

        def file_handler(
            temp_file_path: Path, title: str, collection: str, url: str
        ) -> dict[str, Any]:
            return {
                "file_path": str(temp_file_path),
                "file_id": "file-1",
                "rollback_on_failure": lambda result=None: events.append(
                    ("rollback", result)
                ),
                "commit_on_success": lambda result=None: events.append(
                    ("commit", result)
                ),
            }

        with patch(
            "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.WebCrawler"
        ) as mock_crawler_class:
            mock_crawler = MagicMock()
            mock_crawler.crawl = AsyncMock(return_value=mock_crawl_results)
            mock_crawler.total_urls_found = 1
            mock_crawler.failed_urls = {}
            mock_crawler_class.return_value = mock_crawler

            with patch(
                "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion."
                "run_document_ingestion",
                return_value=successful_ingestion,
            ):
                result = await run_web_ingestion(
                    collection="test_collection",
                    crawl_config=crawl_config,
                    ingestion_config=ingestion_config,
                    file_handler=file_handler,
                )

        assert result.status == "success"
        assert events == [("commit", successful_ingestion)]
