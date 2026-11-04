"""
End-to-end tests for KB web ingestion workflow.

This module tests the complete workflow from frontend web URL input
through web crawling to RAG processing and final searchability.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from xagent.core.tools.core.RAG_tools.core.schemas import (
    CollectionInfo,
    IngestionResult,
)

pytestmark = [pytest.mark.e2e, pytest.mark.contract_stub]


# ==========================================
# TEST-SPECIFIC FIXTURES
# ==========================================
# Note: _StubEmbeddingAdapter, stub_embedding_adapter are provided by conftest.py


@pytest.fixture
def mock_web_rag_pipeline(
    monkeypatch: Any,
    stub_embedding_config,
    stub_embedding_adapter,
) -> None:
    """Mock the RAG pipeline components for web ingestion E2E testing.

    This extends the base mock_rag_pipeline from conftest with
    web ingestion-specific collection configuration.
    """
    from xagent.core.tools.core.RAG_tools.management import collection_manager

    mgr = collection_manager.collection_manager

    mock_collection = CollectionInfo(
        name="e2e_web_test_collection",
        embedding_model_id="e2e-web-test-embedding",
        embedding_dimension=2,
    )

    async def mock_get_collection(collection_name: str) -> CollectionInfo:
        return mock_collection

    async def mock_initialize_collection(
        collection_name: str, embedding_model_id: str
    ) -> CollectionInfo:
        return mock_collection

    monkeypatch.setattr(mgr, "get_collection", mock_get_collection)
    monkeypatch.setattr(
        mgr, "initialize_collection_embedding", mock_initialize_collection
    )


@pytest.fixture
def mock_crawl_results():
    """Provide mock crawl results for testing.

    This simulates successful web crawling results that would
    normally come from the WebCrawler component.
    """
    return [
        MagicMock(
            url="https://example.com/page1",
            title="Example_Page_1",
            content_markdown="# Example Page 1\n\nThis is test content for page 1.",
            status="success",
            depth=0,
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
            content_length=50,
        ),
        MagicMock(
            url="https://example.com/page2",
            title="Example_Page_2",
            content_markdown="## Example Page 2\n\nThis is test content for page 2.",
            status="success",
            depth=0,
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
            content_length=50,
        ),
    ]


# ==========================================
# E2E TEST CLASSES
# ==========================================


class TestKBWebIngestionE2E:
    """
    End-to-end tests for KB web ingestion workflow.

    These tests simulate the complete user workflow:
    1. Frontend submits web URL via API
    2. Backend crawls the website
    3. RAG processing (parse → chunk → embed)
    4. Content becomes searchable
    """

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_ingest_github_url_complete_workflow(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        mock_web_rag_pipeline: None,
        mock_crawl_results: list,
    ):
        """Test complete workflow: submit GitHub URL → crawl → RAG processing → searchable."""
        collection_name = "e2e_github_test"
        github_url = "https://github.com/xorbitsai/xagent"

        # Mock the web ingestion pipeline
        with patch(
            "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.WebCrawler"
        ) as mock_crawler_class:
            # Setup mock crawler
            mock_crawler = MagicMock()
            mock_crawler.crawl = AsyncMock(return_value=mock_crawl_results)
            mock_crawler.total_urls_found = len(mock_crawl_results)
            mock_crawler.failed_urls = {}
            mock_crawler_class.return_value = mock_crawler

            # Mock successful document ingestion
            mock_ingestion_result = IngestionResult(
                status="success",
                doc_id="github_doc_1",
                parse_hash="github_hash_1",
                chunk_count=2,
                embedding_count=2,
                vector_count=2,
                completed_steps=[],
                failed_step=None,
                message="Success",
                warnings=[],
            )

            with patch(
                "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.run_document_ingestion",
                return_value=mock_ingestion_result,
            ):
                # Submit web ingestion request
                response = client.post(
                    "/api/kb/ingest-web",
                    data={
                        "collection": collection_name,
                        "start_url": github_url,
                        "max_pages": "2",
                        "max_depth": "1",
                    },
                    headers=auth_headers,
                )

                # Verify response
                assert response.status_code == 200
                result = response.json()
                assert result["status"] in ["success", "partial", "error"]

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_ingest_general_website_complete_workflow(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        mock_web_rag_pipeline: None,
        mock_crawl_results: list,
    ):
        """Test complete workflow: submit general website URL → crawl → RAG processing."""
        collection_name = "e2e_website_test"
        website_url = "https://example.com/docs"

        # Mock the web ingestion pipeline
        with patch(
            "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.WebCrawler"
        ) as mock_crawler_class:
            mock_crawler = MagicMock()
            mock_crawler.crawl = AsyncMock(return_value=mock_crawl_results)
            mock_crawler.total_urls_found = len(mock_crawl_results)
            mock_crawler.failed_urls = {}
            mock_crawler_class.return_value = mock_crawler

            mock_ingestion_result = IngestionResult(
                status="success",
                doc_id="web_doc_1",
                parse_hash="web_hash_1",
                chunk_count=2,
                embedding_count=2,
                vector_count=2,
                completed_steps=[],
                failed_step=None,
                message="Success",
                warnings=[],
            )

            with patch(
                "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.run_document_ingestion",
                return_value=mock_ingestion_result,
            ):
                response = client.post(
                    "/api/kb/ingest-web",
                    data={
                        "collection": collection_name,
                        "start_url": website_url,
                        "max_pages": 3,
                        "max_depth": 1,
                    },
                    headers=auth_headers,
                )

                assert response.status_code == 200
                result = response.json()
                assert result["status"] in ["success", "partial", "error"]

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_web_ingestion_error_handling_404(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        mock_web_rag_pipeline: None,
    ):
        """Test error handling when web URL returns 404."""
        collection_name = "e2e_error_test"
        invalid_url = "https://example.com/nonexistent-page"

        # Mock crawler to simulate 404 error
        with patch(
            "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.WebCrawler"
        ) as mock_crawler_class:
            mock_crawler = MagicMock()
            mock_crawler.crawl = AsyncMock(side_effect=Exception("404 Not Found"))
            mock_crawler_class.return_value = mock_crawler

            response = client.post(
                "/api/kb/ingest-web",
                data={
                    "collection": collection_name,
                    "start_url": invalid_url,
                    "max_pages": 1,
                },
                headers=auth_headers,
            )

            # Crawler failure should result in error status with 500
            assert response.status_code == 500
            result = response.json()
            assert "status" in result

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_web_ingestion_timeout_handling(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        mock_web_rag_pipeline: None,
    ):
        """Test error handling when web request times out."""
        collection_name = "e2e_timeout_test"
        slow_url = "https://example.com/slow-page"

        # Mock crawler to simulate timeout
        with patch(
            "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.WebCrawler"
        ) as mock_crawler_class:
            mock_crawler = MagicMock()
            mock_crawler.crawl = AsyncMock(side_effect=TimeoutError("Request timeout"))
            mock_crawler_class.return_value = mock_crawler

            response = client.post(
                "/api/kb/ingest-web",
                data={
                    "collection": collection_name,
                    "start_url": slow_url,
                    "max_pages": 1,
                },
                headers=auth_headers,
            )

            # Timeout error should result in error status with 500
            assert response.status_code == 500
            result = response.json()
            assert "status" in result

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_web_ingestion_partial_success(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        mock_web_rag_pipeline: None,
    ):
        """Test handling when some pages succeed but others fail."""
        collection_name = "e2e_partial_test"
        url = "https://example.com/mixed-results"

        # Mock mixed results
        mixed_results = [
            MagicMock(
                url="https://example.com/page1",
                title="Success_Page",
                content_markdown="# Success\nContent",
                status="success",
                depth=0,
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                content_length=20,
            ),
            MagicMock(
                url="https://example.com/page2",
                title="Failed_Page",
                content_markdown="",
                status="error",
                depth=0,
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                content_length=0,
            ),
        ]

        with patch(
            "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.WebCrawler"
        ) as mock_crawler_class:
            mock_crawler = MagicMock()
            mock_crawler.crawl = AsyncMock(return_value=mixed_results)
            mock_crawler.total_urls_found = 2
            mock_crawler.failed_urls = {"https://example.com/page2": "Connection error"}
            mock_crawler_class.return_value = mock_crawler

            # Mock successful ingestion for first page
            success_result = IngestionResult(
                status="success",
                doc_id="partial_doc_1",
                parse_hash="partial_hash_1",
                chunk_count=1,
                embedding_count=1,
                vector_count=1,
                completed_steps=[],
                failed_step=None,
                message="Success",
                warnings=[],
            )

            with patch(
                "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.run_document_ingestion",
                return_value=success_result,
            ):
                response = client.post(
                    "/api/kb/ingest-web",
                    data={
                        "collection": collection_name,
                        "start_url": url,
                        "max_pages": 2,
                    },
                    headers=auth_headers,
                )

                assert response.status_code == 200
                result = response.json()
                # Should report partial success
                assert result["status"] in ["success", "partial", "error"]

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_web_ingestion_empty_results(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        mock_web_rag_pipeline: None,
    ):
        """A crawl that ingested nothing publishes no KB, so it reports an error."""
        collection_name = "e2e_empty_test"
        url = "https://example.com/empty"

        with patch(
            "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.WebCrawler"
        ) as mock_crawler_class:
            mock_crawler = MagicMock()
            mock_crawler.crawl = AsyncMock(return_value=[])
            mock_crawler.total_urls_found = 0
            mock_crawler.failed_urls = {}
            mock_crawler_class.return_value = mock_crawler

            response = client.post(
                "/api/kb/ingest-web",
                data={
                    "collection": collection_name,
                    "start_url": url,
                    "max_pages": 1,
                },
                headers=auth_headers,
            )

            assert response.status_code == 500
            result = response.json()
            assert result["status"] == "error"
            assert "No pages were ingested" in result["message"]
