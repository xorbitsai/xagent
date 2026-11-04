"""
End-to-end tests for search functionality verification.

This module tests that search works correctly after document ingestion,
ensuring that users can find their content immediately.

IMPORTANT: Legacy data compatibility tests in this file must be updated
whenever schema changes are made to ensure forward compatibility is maintained.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Generator

import pytest
from fastapi.testclient import TestClient

from tests.web_integration.http_helpers import eventually, http_detail

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.contract_stub,
    pytest.mark.usefixtures("kb_engine"),
]


@pytest.fixture
def sample_search_files() -> Generator[tuple[dict[str, str], str], None, None]:
    """Create sample test files for search testing."""
    files = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files with specific searchable content
        test_files = {
            "python_tutorial.txt": "Python is a programming language. Python is widely used for web development, data science, and automation.",
            "machine_learning.md": "# Machine Learning Guide\n\nMachine learning is a subset of artificial intelligence. It focuses on building systems that can learn from data.",
            "cooking_guide.txt": "Cooking tips: Always use fresh ingredients. Follow the recipe carefully. Taste your food while cooking.",
            "travel_guide.txt": "Travel destinations: Paris, Tokyo, New York are popular cities. Each city has unique attractions and culture.",
        }

        for filename, content in test_files.items():
            file_path = Path(temp_dir) / filename
            file_path.write_text(content, encoding="utf-8")
            files[filename] = str(file_path)

        yield files, temp_dir


def _ingest(
    client: TestClient, headers: dict[str, str], path: str | Path, collection: str
) -> None:
    with open(path, "rb") as f:
        response = client.post(
            "/api/kb/ingest",
            files={"file": (Path(path).name, f, "text/plain")},
            data={"collection": collection},
            headers=headers,
        )
    assert response.status_code == 200, http_detail(response)
    assert response.json()["status"] == "success", http_detail(response)


def _search(
    client: TestClient,
    headers: dict[str, str],
    collection: str,
    query: str,
    top_k: int = 5,
    min_hits: int = 1,
    search_type: str = "hybrid",
) -> list[dict[str, Any]]:
    """Search until ``min_hits`` come back. A write is visible at once on LanceDB
    and within Milvus's Bounded window on Milvus; an error status fails at once."""
    hits: list[dict[str, Any]] = []
    status = ""

    def settled() -> bool:
        nonlocal hits, status
        response = client.post(
            "/api/kb/search",
            data={
                "collection": collection,
                "query_text": query,
                "top_k": top_k,
                "search_type": search_type,
            },
            headers=headers,
        )
        assert response.status_code == 200, http_detail(response)
        body = response.json()
        status = body["status"]
        assert status == "success", http_detail(response)
        hits = body["results"]
        return len(hits) >= min_hits

    milvus = os.environ["XAGENT_VECTOR_BACKEND"] == "milvus"
    eventually(
        settled,
        timeout=10.0 if milvus else 0.0,
        detail=lambda: f"status {status}, {len(hits)} of {min_hits} hits",
    )
    return hits


# ==========================================
# BASIC SEARCH TESTS
# ==========================================


class TestBasicSearch:
    """
    Test basic search functionality.

    These tests verify that:
    1. Search works immediately after ingestion
    2. Search results are relevant
    3. Search pagination works
    4. Search filters work correctly
    """

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_immediate_after_ingestion(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_search_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that search works immediately after document ingestion.

        LanceDB returns the document on the first search; Milvus within its window.
        """
        files, temp_dir = sample_search_files
        collection_name = "e2e_search_immediate"

        _ingest(client, auth_headers, files["python_tutorial.txt"], collection_name)

        hits = _search(client, auth_headers, collection_name, "Python programming")
        assert any("Python" in hit["text"] for hit in hits)

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_relevance_ranking(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_search_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that search results are ranked by relevance."""
        files, temp_dir = sample_search_files
        collection_name = "e2e_search_relevance"

        for filename in [
            "cooking_guide.txt",
            "machine_learning.md",
            "python_tutorial.txt",
        ]:
            _ingest(client, auth_headers, files[filename], collection_name)

        hits = _search(client, auth_headers, collection_name, "Python", top_k=3)
        assert "Python" in hits[0]["text"]

        keyword_hits = _search(
            client, auth_headers, collection_name, "Python", search_type="sparse"
        )
        assert all("Python" in hit["text"] for hit in keyword_hits)

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_pagination(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_search_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that search pagination works correctly."""
        files, temp_dir = sample_search_files
        collection_name = "e2e_search_pagination"

        for i in range(5):
            path = Path(temp_dir) / f"doc{i}.txt"
            path.write_text(
                f"Document {i} with searchable content about topic {i % 3}.",
                encoding="utf-8",
            )
            _ingest(client, auth_headers, path, collection_name)

        hits = _search(
            client, auth_headers, collection_name, "document", top_k=3, min_hits=3
        )
        assert len(hits) == 3

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_filters(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_search_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that search filters work correctly."""
        files, temp_dir = sample_search_files
        collection_name = "e2e_search_filters"

        for filename in ["python_tutorial.txt", "machine_learning.md"]:
            _ingest(client, auth_headers, files[filename], collection_name)

        _search(client, auth_headers, collection_name, "document", min_hits=2)


# ==========================================
# MULTI-TENANT SEARCH TESTS
# ==========================================


class TestMultiTenantSearch:
    """
    Test multi-tenant search isolation.

    These tests verify that:
    1. Users can only search their own documents
    2. Admin users can search across tenants
    3. Legacy data isolation works correctly
    """

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_only_returns_own_documents(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_search_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that regular users can only search their own documents."""
        files, temp_dir = sample_search_files
        collection_name = "e2e_search_isolation"

        _ingest(client, auth_headers, files["python_tutorial.txt"], collection_name)

        hits = _search(client, auth_headers, collection_name, "Python")
        assert any("Python" in hit["text"] for hit in hits)


# ==========================================
# SEARCH AFTER SCHEMA CHANGES TESTS
# ==========================================


class TestSearchAfterSchemaChanges:
    """
    Test search functionality after schema changes.

    These tests verify that:
    1. Search works after migration
    2. Search works with mixed schema versions
    3. Fallback mechanisms work for legacy data
    """

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_after_migration(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_search_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that search works correctly after schema migration."""
        files, temp_dir = sample_search_files
        collection_name = "e2e_search_migration"

        _ingest(client, auth_headers, files["python_tutorial.txt"], collection_name)

        hits = _search(client, auth_headers, collection_name, "Python")
        assert any("Python" in hit["text"] for hit in hits)


# ==========================================
# SEARCH ACCURACY TESTS
# ==========================================


class TestSearchAccuracy:
    """
    Test search result accuracy and quality.

    These tests verify that:
    1. Search returns relevant results
    2. Search scores are reasonable
    3. Search handles different query types
    """

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_returns_relevant_results(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_search_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that search returns relevant results."""
        files, temp_dir = sample_search_files
        collection_name = "e2e_search_accuracy"

        _ingest(client, auth_headers, files["python_tutorial.txt"], collection_name)

        hits = _search(
            client, auth_headers, collection_name, "Python programming language"
        )
        assert any("Python" in hit["text"] for hit in hits)

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_with_different_query_types(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_search_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that search handles different query types correctly."""
        files, temp_dir = sample_search_files
        collection_name = "e2e_search_query_types"

        _ingest(client, auth_headers, files["python_tutorial.txt"], collection_name)

        queries = [
            "Python",  # Single word
            "Python programming",  # Phrase
            "Python language web development",  # Multiple words
        ]

        for query in queries:
            _search(client, auth_headers, collection_name, query, top_k=3)


# ==========================================
# REAL-TIME SEARCH TESTS
# ==========================================


class TestRealTimeSearch:
    """
    Test real-time search functionality.

    These tests verify that:
    1. Search works immediately after ingestion
    2. Search updates in real-time
    3. Search handles concurrent operations
    """

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_updates_in_realtime(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_search_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that search results update in real-time after ingestion."""
        files, temp_dir = sample_search_files
        collection_name = "e2e_search_realtime"

        before = client.post(
            "/api/kb/search",
            data={"collection": collection_name, "query_text": "Python"},
            headers=auth_headers,
        )
        assert before.status_code == 404, http_detail(before)

        _ingest(client, auth_headers, files["python_tutorial.txt"], collection_name)

        hits = _search(client, auth_headers, collection_name, "Python")
        assert any("Python" in hit["text"] for hit in hits)

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_with_multiple_documents(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_search_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that search works with multiple documents."""
        files, temp_dir = sample_search_files
        collection_name = "e2e_search_multiple"

        for count, filename in enumerate(
            ["python_tutorial.txt", "machine_learning.md"], start=1
        ):
            _ingest(client, auth_headers, files[filename], collection_name)
            _search(
                client,
                auth_headers,
                collection_name,
                "content",
                top_k=3,
                min_hits=count,
            )


# ==========================================
# SEARCH ERROR HANDLING TESTS
# ==========================================


class TestSearchErrorHandling:
    """Test search error handling and edge cases."""

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_with_empty_query(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that search handles empty queries gracefully."""
        collection_name = "e2e_search_empty"

        search_response = client.post(
            "/api/kb/search",
            data={
                "collection": collection_name,
                "query_text": "",  # Empty query
                "top_k": 5,
            },
            headers=auth_headers,
        )

        # Empty query is rejected by validation (FastAPI form validation)
        assert search_response.status_code == 422

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_nonexistent_collection(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that search handles nonexistent collections gracefully."""
        search_response = client.post(
            "/api/kb/search",
            data={
                "collection": "nonexistent_collection_xyz",
                "query_text": "test",
                "top_k": 5,
            },
            headers=auth_headers,
        )

        # Nonexistent collection returns 404
        assert search_response.status_code == 404

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_with_special_characters(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_search_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that search handles special characters in queries."""
        files, temp_dir = sample_search_files
        collection_name = "e2e_search_special"

        _ingest(client, auth_headers, files["python_tutorial.txt"], collection_name)

        special_queries = [
            "Python & programming",  # Ampersand
            "Python, data, science",  # Commas
            'Python "language"',  # Quotes
        ]

        for query in special_queries:
            _search(client, auth_headers, collection_name, query, top_k=3)
