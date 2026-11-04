"""
End-to-end tests for KB CRUD operations.

This module tests the complete CRUD operations from frontend API
through backend processing to database storage, ensuring all
operations work correctly together.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tests.web_integration.http_helpers import http_detail
from xagent.core.tools.core.RAG_tools.core.schemas import (
    IngestionResult,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.contract_stub,
    pytest.mark.usefixtures("kb_engine"),
]

# Note: _StubEmbeddingAdapter, stub_embedding_adapter, and mock_rag_pipeline
# are provided by conftest.py with autouse=True


@pytest.fixture
def sample_crud_files() -> Generator[tuple[dict[str, str], str], None, None]:
    """Create sample test files for CRUD testing."""
    files = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create diverse test files
        test_files = {
            "document1.txt": "First document with content for CRUD testing.",
            "document2.md": "# Second Document\n\nMarkdown content for CRUD testing.",
            "document3.json": '{"title": "Third Document", "content": "JSON data for CRUD"}',
        }

        for filename, content in test_files.items():
            file_path = Path(temp_dir) / filename
            file_path.write_text(content, encoding="utf-8")
            files[filename] = str(file_path)

        yield files, temp_dir


# ==========================================
# CREATE OPERATIONS TESTS
# ==========================================


class TestKBCreateOperations:
    """Test Create operations for KB resources."""

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_create_collection_with_single_file(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_crud_files: tuple[dict[str, str], str],
    ) -> None:
        """Test creating a collection by ingesting a single file."""
        files, temp_dir = sample_crud_files
        collection_name = "e2e_create_single"
        file_path = files["document1.txt"]

        # Create collection by ingesting a file
        with open(file_path, "rb") as f:
            response = client.post(
                "/api/kb/ingest",
                files={"file": ("document1.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        # Verify creation succeeded
        assert response.status_code == 200
        result = response.json()
        assert result["status"] in ["success", "partial"]

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_create_collection_with_multiple_files(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_crud_files: tuple[dict[str, str], str],
    ) -> None:
        """Test creating a collection by ingesting multiple files."""
        files, temp_dir = sample_crud_files
        collection_name = "e2e_create_multiple"

        created_count = 0
        for filename in ["document1.txt", "document2.md", "document3.json"]:
            file_path = files[filename]
            with open(file_path, "rb") as f:
                response = client.post(
                    "/api/kb/ingest",
                    files={"file": (filename, f, "text/plain")},
                    data={"collection": collection_name},
                    headers=auth_headers,
                )
                assert response.status_code == 200, http_detail(response)
                created_count += 1

        # Verify at least some files were created
        assert created_count >= 1

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_create_document_with_different_types(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_crud_files: tuple[dict[str, str], str],
    ) -> None:
        """Test creating documents with different file types."""
        files, temp_dir = sample_crud_files
        collection_name = "e2e_create_types"

        # Test different file types
        file_types = [
            ("document1.txt", "text/plain"),
            ("document2.md", "text/markdown"),
            ("document3.json", "application/json"),
        ]

        created_count = 0
        for filename, mime_type in file_types:
            file_path = files[filename]
            with open(file_path, "rb") as f:
                response = client.post(
                    "/api/kb/ingest",
                    files={"file": (filename, f, mime_type)},
                    data={"collection": collection_name},
                    headers=auth_headers,
                )
                assert response.status_code == 200, http_detail(response)
                created_count += 1

        # Verify different file types can be created
        assert created_count >= 1

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_create_document_with_custom_config(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_crud_files: tuple[dict[str, str], str],
    ) -> None:
        """Test creating documents with custom ingestion configuration."""
        files, temp_dir = sample_crud_files
        collection_name = "e2e_create_config"
        file_path = files["document1.txt"]

        # Create with custom chunking configuration
        with open(file_path, "rb") as f:
            response = client.post(
                "/api/kb/ingest",
                files={"file": ("document1.txt", f, "text/plain")},
                data={
                    "collection": collection_name,
                    "parse_method": "deepdoc",
                    "chunk_strategy": "fixed_size",
                    "chunk_size": "300",
                    "chunk_overlap": "100",
                },
                headers=auth_headers,
            )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] in ["success", "partial"]

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_create_collection_with_web_url(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test creating a collection by ingesting from a web URL."""
        collection_name = "e2e_create_web"
        target_url = "https://github.com/xorbitsai/xagent"

        # Mock web ingestion
        with patch(
            "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.WebCrawler"
        ) as mock_crawler_class:
            # Setup mock crawler results
            mock_crawl_results = [
                MagicMock(
                    url=target_url,
                    title="xagent",
                    content_markdown="# XAgent\n\nAI agent framework",
                    status="success",
                    depth=0,
                    content_length=50,
                )
            ]

            mock_crawler = MagicMock()
            mock_crawler.crawl = AsyncMock(return_value=mock_crawl_results)
            mock_crawler.total_urls_found = 1
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
                        "start_url": target_url,
                        "max_pages": "1",
                    },
                    headers=auth_headers,
                )

                # Verify web ingestion creates collection
                assert response.status_code == 200
                result = response.json()
                assert result["status"] in ["success", "partial", "error"]


# ==========================================
# READ OPERATIONS TESTS
# ==========================================


class TestKBReadOperations:
    """Test Read operations for KB resources."""

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_list_collections_empty(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test listing collections when none exist."""
        response = client.get("/api/kb/collections", headers=auth_headers)

        # Should succeed even if empty
        assert response.status_code == 200
        result = response.json()
        assert "collections" in result

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_list_collections_with_data(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_crud_files: tuple[dict[str, str], str],
    ) -> None:
        """Test listing collections after creating one."""
        files, temp_dir = sample_crud_files
        collection_name = "e2e_read_list"
        file_path = files["document1.txt"]

        # First create a collection
        with open(file_path, "rb") as f:
            create_response = client.post(
                "/api/kb/ingest",
                files={"file": ("document1.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        assert create_response.status_code == 200, http_detail(create_response)

        # Then list collections
        list_response = client.get("/api/kb/collections", headers=auth_headers)
        assert list_response.status_code == 200, http_detail(list_response)
        result = list_response.json()
        assert "collections" in result

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_get_collection_details(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_crud_files: tuple[dict[str, str], str],
    ) -> None:
        """Test getting detailed information about a collection."""
        files, temp_dir = sample_crud_files
        collection_name = "e2e_read_details"
        file_path = files["document1.txt"]

        # Create a collection first
        with open(file_path, "rb") as f:
            create_response = client.post(
                "/api/kb/ingest",
                files={"file": ("document1.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        assert create_response.status_code == 200, http_detail(create_response)

        list_response = client.get("/api/kb/collections", headers=auth_headers)
        assert list_response.status_code == 200, http_detail(list_response)
        names = {c.get("name") for c in list_response.json().get("collections", [])}
        assert collection_name in names

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_list_documents_in_collection(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_crud_files: tuple[dict[str, str], str],
    ) -> None:
        """Test listing documents within a collection."""
        files, temp_dir = sample_crud_files
        collection_name = "e2e_read_documents"
        file_path = files["document1.txt"]

        # Create a document first
        with open(file_path, "rb") as f:
            create_response = client.post(
                "/api/kb/ingest",
                files={"file": ("document1.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        assert create_response.status_code == 200, http_detail(create_response)

        list_response = client.post(
            f"/api/kb/collections/{collection_name}/documents/check",
            json={"filenames": ["document1.txt"]},
            headers=auth_headers,
        )
        assert list_response.status_code == 200, http_detail(list_response)
        assert "document1.txt" in list_response.json().get("existing_filenames", [])

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_get_document_stats(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_crud_files: tuple[dict[str, str], str],
    ) -> None:
        """Test getting statistics for a specific document."""
        files, temp_dir = sample_crud_files
        collection_name = "e2e_read_stats"
        file_path = files["document1.txt"]

        # Create a document first
        with open(file_path, "rb") as f:
            create_response = client.post(
                "/api/kb/ingest",
                files={"file": ("document1.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        assert create_response.status_code == 200, http_detail(create_response)
        body = create_response.json()
        assert "chunk_count" in body
        assert int(body["chunk_count"]) >= 0
        doc_id = body.get("doc_id")
        if doc_id:
            parse_response = client.get(
                f"/api/kb/collections/{collection_name}/parses/{doc_id}/parse_result",
                params={"page": 1, "page_size": 20},
                headers=auth_headers,
            )
            # With successful ingest and valid doc_id, parse result should be available
            assert parse_response.status_code == 200, http_detail(parse_response)


# ==========================================
# UPDATE OPERATIONS TESTS
# ==========================================


class TestKBUpdateOperations:
    """Test Update operations for KB resources."""

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_update_collection_config(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test updating collection configuration."""
        collection_name = "e2e_update_config"

        # Save collection config
        save_response = client.post(
            f"/api/kb/collections/{collection_name}/config",
            json={
                "parse_method": "deepdoc",
                "chunk_strategy": "fixed_size",
                "chunk_size": 500,
                "chunk_overlap": 50,
                "embedding_model_id": "text-embedding-v3",
            },
            headers=auth_headers,
        )

        # Config save endpoint should create/update config successfully
        assert save_response.status_code == 200

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_reingest_document(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_crud_files: tuple[dict[str, str], str],
    ) -> None:
        """Test re-ingesting a document to update its content."""
        files, temp_dir = sample_crud_files
        collection_name = "e2e_update_reingest"
        file_path = files["document1.txt"]

        # First ingestion
        with open(file_path, "rb") as f:
            first_response = client.post(
                "/api/kb/ingest",
                files={"file": ("document1.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        assert first_response.status_code == 200, http_detail(first_response)

        # Re-ingest the same document
        with open(file_path, "rb") as f:
            second_response = client.post(
                "/api/kb/ingest",
                files={"file": ("document1.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        # Re-ingestion should succeed
        assert second_response.status_code == 200, http_detail(second_response)

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_update_document_metadata(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_crud_files: tuple[dict[str, str], str],
    ) -> None:
        """Test updating document metadata."""
        files, temp_dir = sample_crud_files
        collection_name = "e2e_update_metadata"
        file_path = files["document1.txt"]

        # Create document first
        with open(file_path, "rb") as f:
            create_response = client.post(
                "/api/kb/ingest",
                files={"file": ("document1.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        assert create_response.status_code == 200, http_detail(create_response)

        # Try to update document metadata
        # Note: This endpoint does not exist yet - testing expected 405
        update_response = client.put(
            f"/api/kb/collections/{collection_name}/documents/document1.txt",
            json={"title": "Updated Title", "description": "Updated Description"},
            headers=auth_headers,
        )
        # Endpoint not implemented - should return 405 (Method Not Allowed)
        assert update_response.status_code == 405, http_detail(update_response)


# ==========================================
# DELETE OPERATIONS TESTS
# ==========================================


class TestKBDeleteOperations:
    """Test Delete operations for KB resources."""

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_delete_single_document_by_filename(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_crud_files: tuple[dict[str, str], str],
    ) -> None:
        """Test deleting a single document by filename (legacy method)."""
        files, temp_dir = sample_crud_files
        collection_name = "e2e_delete_single_filename"
        file_path = files["document1.txt"]

        # Create document first
        with open(file_path, "rb") as f:
            create_response = client.post(
                "/api/kb/ingest",
                files={"file": ("document1.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        assert create_response.status_code == 200, http_detail(create_response)

        # Delete the document by filename (legacy method)
        delete_response = client.delete(
            f"/api/kb/collections/{collection_name}/documents/document1.txt",
            headers=auth_headers,
        )
        # Deleting a just-created document should succeed
        assert delete_response.status_code == 200, http_detail(delete_response)

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_delete_single_document_by_file_id(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_crud_files: tuple[dict[str, str], str],
    ) -> None:
        """Test deleting a single document by file_id (recommended method)."""
        files, temp_dir = sample_crud_files
        collection_name = "e2e_delete_single_fileid"
        file_path = files["document1.txt"]

        # Create document first
        with open(file_path, "rb") as f:
            create_response = client.post(
                "/api/kb/ingest",
                files={"file": ("document1.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        assert create_response.status_code == 200, http_detail(create_response)

        # Try to get file_id from creation response
        create_result = create_response.json()
        file_id = create_result.get("file_id")

        # Delete the document by file_id if available
        if file_id:
            delete_response = client.delete(
                f"/api/kb/collections/{collection_name}/documents/document1.txt?file_id={file_id}",
                headers=auth_headers,
            )
            # Deleting a just-created document with file_id should succeed
            assert delete_response.status_code == 200, http_detail(delete_response)
        else:
            # Fallback to filename deletion if file_id not available
            delete_response = client.delete(
                f"/api/kb/collections/{collection_name}/documents/document1.txt",
                headers=auth_headers,
            )
            # Deleting a just-created document should succeed
            assert delete_response.status_code == 200, http_detail(delete_response)

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_delete_single_document_by_doc_id(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_crud_files: tuple[dict[str, str], str],
    ) -> None:
        """Test deleting a single document by doc_id."""
        files, temp_dir = sample_crud_files
        collection_name = "e2e_delete_single_docid"
        file_path = files["document1.txt"]

        # Create document first
        with open(file_path, "rb") as f:
            create_response = client.post(
                "/api/kb/ingest",
                files={"file": ("document1.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        assert create_response.status_code == 200, http_detail(create_response)

        # Try to get doc_id from creation response
        create_result = create_response.json()
        doc_id = create_result.get("doc_id")

        # Delete the document by doc_id if available
        if doc_id:
            delete_response = client.delete(
                f"/api/kb/collections/{collection_name}/documents/document1.txt?doc_id={doc_id}",
                headers=auth_headers,
            )
            # Deleting a just-created document with doc_id should succeed
            assert delete_response.status_code == 200, http_detail(delete_response)
        else:
            # Fallback to filename deletion if doc_id not available
            delete_response = client.delete(
                f"/api/kb/collections/{collection_name}/documents/document1.txt",
                headers=auth_headers,
            )
            # Deleting a just-created document should succeed
            assert delete_response.status_code == 200, http_detail(delete_response)

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_delete_prefer_file_id_over_filename(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_crud_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that file_id takes precedence over filename when both are provided."""
        files, temp_dir = sample_crud_files
        collection_name = "e2e_delete_prefer_fileid"
        file_path = files["document1.txt"]

        # Create document first
        with open(file_path, "rb") as f:
            create_response = client.post(
                "/api/kb/ingest",
                files={"file": ("document1.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        assert create_response.status_code == 200, http_detail(create_response)
        create_result = create_response.json()
        file_id = create_result.get("file_id")

        assert isinstance(file_id, str) and file_id, "Expected ingest to return file_id"

        # Delete with both file_id and filename - file_id should take precedence
        delete_response = client.delete(
            f"/api/kb/collections/{collection_name}/documents/some_different_name.txt?file_id={file_id}",
            headers=auth_headers,
        )
        # Deleting by file_id should succeed regardless of filename
        assert delete_response.status_code == 200, http_detail(delete_response)

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_delete_multiple_documents(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_crud_files: tuple[dict[str, str], str],
    ) -> None:
        """Test deleting multiple documents from a collection."""
        files, temp_dir = sample_crud_files
        collection_name = "e2e_delete_multiple"

        # Create multiple documents
        created_docs = []
        for filename in ["document1.txt", "document2.md"]:
            file_path = files[filename]
            with open(file_path, "rb") as f:
                response = client.post(
                    "/api/kb/ingest",
                    files={"file": (filename, f, "text/plain")},
                    data={"collection": collection_name},
                    headers=auth_headers,
                )
                assert response.status_code == 200, http_detail(response)
                created_docs.append(filename)

        # Delete documents one by one
        deleted_count = 0
        for doc_name in created_docs:
            delete_response = client.delete(
                f"/api/kb/collections/{collection_name}/documents/{doc_name}",
                headers=auth_headers,
            )
            assert delete_response.status_code == 200, http_detail(delete_response)
            deleted_count += 1

        # Verify at least some documents were created and deletion attempts were made
        assert len(created_docs) > 0

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_delete_collection_with_documents(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_crud_files: tuple[dict[str, str], str],
    ) -> None:
        """Test deleting an entire collection with documents."""
        files, temp_dir = sample_crud_files
        collection_name = "e2e_delete_collection"
        file_path = files["document1.txt"]

        # Create a document in the collection
        with open(file_path, "rb") as f:
            create_response = client.post(
                "/api/kb/ingest",
                files={"file": ("document1.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        assert create_response.status_code == 200, http_detail(create_response)

        # Delete the entire collection
        delete_response = client.delete(
            f"/api/kb/collections/{collection_name}",
            headers=auth_headers,
        )
        # Deleting a collection that exists should succeed
        assert delete_response.status_code == 200, http_detail(delete_response)

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_delete_collection_cleanup(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_crud_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that collection deletion properly cleans up resources."""
        files, temp_dir = sample_crud_files
        collection_name = "e2e_delete_cleanup"
        file_path = files["document1.txt"]

        # Create a document
        with open(file_path, "rb") as f:
            create_response = client.post(
                "/api/kb/ingest",
                files={"file": ("document1.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        assert create_response.status_code == 200, http_detail(create_response)

        # Delete collection
        delete_response = client.delete(
            f"/api/kb/collections/{collection_name}",
            headers=auth_headers,
        )
        assert delete_response.status_code == 200, http_detail(delete_response)

        list_after = client.get("/api/kb/collections", headers=auth_headers)
        assert list_after.status_code == 200, http_detail(list_after)
        names = {c.get("name") for c in list_after.json().get("collections", [])}
        assert collection_name not in names
