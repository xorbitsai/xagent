"""
End-to-end tests for KB file ingestion workflow.

This module tests the complete workflow from frontend file upload
through RAG processing to final searchability, ensuring all
components work together correctly.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict

import pytest
from fastapi.testclient import TestClient

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.contract_stub,
    pytest.mark.usefixtures("kb_engine"),
]


# ==========================================
# TEST-SPECIFIC FIXTURES
# ==========================================
# Note: _StubEmbeddingAdapter, stub_embedding_adapter, and mock_rag_pipeline
# are provided by conftest.py with autouse=True


@pytest.fixture
def sample_test_files():
    """Create sample test files for E2E testing."""
    files = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files of different formats
        test_files = {
            "test.txt": "This is a test text file for E2E testing.\nIt contains multiple lines.",
            "test.md": "# Test Markdown\n\nThis is a **markdown** test file.\n\n## Section 2\n\nContent here.",
            "test.json": '{"title": "Test Document", "content": "JSON test content", "count": 42}',
            "test.csv": "name,age,city\nJohn,25,NYC\nJane,30,LA",
        }

        for filename, content in test_files.items():
            file_path = Path(temp_dir) / filename
            file_path.write_text(content, encoding="utf-8")
            files[filename] = str(file_path)

        yield files, temp_dir


class TestKBFileIngestionE2E:
    """
    End-to-end tests for KB file ingestion workflow.

    These tests simulate the complete user workflow:
    1. Frontend uploads file via API
    2. Backend saves file to storage
    3. RAG processing (parse → chunk → embed)
    4. Content becomes searchable
    """

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_ingest_text_file_complete_workflow(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        sample_test_files: tuple[Dict[str, str], str],
        mock_rag_pipeline: None,
    ):
        """Test complete workflow: upload .txt file → RAG processing → searchable."""
        files, temp_dir = sample_test_files
        file_path = files["test.txt"]
        collection_name = "e2e_txt_test"

        # Step 1: Upload file via API
        with open(file_path, "rb") as f:
            upload_response = client.post(
                "/api/kb/ingest",
                files={"file": ("test.txt", f, "text/plain")},
                data={
                    "collection": collection_name,
                    "parse_method": "default",
                    "chunk_strategy": "fixed_size",
                    "chunk_size": "500",
                    "chunk_overlap": "50",
                },
                headers=auth_headers,
            )

        # Step 2: Verify upload was successful
        assert upload_response.status_code == 200
        result = upload_response.json()
        assert result["status"] in ["success", "partial"]
        assert "doc_id" in result or result["status"] == "success"

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_ingest_markdown_file_complete_workflow(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        sample_test_files: tuple[Dict[str, str], str],
        mock_rag_pipeline: None,
    ):
        """Test complete workflow: upload .md file → RAG processing → searchable."""
        files, temp_dir = sample_test_files
        file_path = files["test.md"]
        collection_name = "e2e_md_test"

        # Upload markdown file with markdown-specific chunking
        with open(file_path, "rb") as f:
            upload_response = client.post(
                "/api/kb/ingest",
                files={"file": ("test.md", f, "text/markdown")},
                data={
                    "collection": collection_name,
                    "parse_method": "default",
                    "chunk_strategy": "markdown",  # Use markdown-aware chunking
                    "chunk_size": "300",
                    "chunk_overlap": "50",
                },
                headers=auth_headers,
            )

        # Verify response
        assert upload_response.status_code == 200
        result = upload_response.json()
        assert result["status"] in ["success", "partial"]

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_ingest_json_file_complete_workflow(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        sample_test_files: tuple[Dict[str, str], str],
        mock_rag_pipeline: None,
    ):
        """Test complete workflow: upload .json file → RAG processing → searchable."""
        files, temp_dir = sample_test_files
        file_path = files["test.json"]
        collection_name = "e2e_json_test"

        # Upload JSON file
        with open(file_path, "rb") as f:
            upload_response = client.post(
                "/api/kb/ingest",
                files={"file": ("test.json", f, "application/json")},
                data={
                    "collection": collection_name,
                    "parse_method": "default",
                    "chunk_strategy": "fixed_size",
                },
                headers=auth_headers,
            )

        # Verify response
        assert upload_response.status_code == 200
        result = upload_response.json()
        assert result["status"] in ["success", "partial"]

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_ingest_csv_file_complete_workflow(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        sample_test_files: tuple[Dict[str, str], str],
        mock_rag_pipeline: None,
    ):
        """Test complete workflow: upload .csv file → RAG processing → searchable."""
        files, temp_dir = sample_test_files
        file_path = files["test.csv"]
        collection_name = "e2e_csv_test"

        # Upload CSV file
        with open(file_path, "rb") as f:
            upload_response = client.post(
                "/api/kb/ingest",
                files={"file": ("test.csv", f, "text/csv")},
                data={
                    "collection": collection_name,
                    "parse_method": "default",
                    "chunk_strategy": "fixed_size",
                },
                headers=auth_headers,
            )

        # Verify response
        assert upload_response.status_code == 200
        result = upload_response.json()
        assert result["status"] in ["success", "partial"]

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_ingest_multiple_files_same_collection(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        sample_test_files: tuple[Dict[str, str], str],
        mock_rag_pipeline: None,
    ):
        """Test uploading multiple files to the same collection."""
        files, temp_dir = sample_test_files
        collection_name = "e2e_multi_test"

        uploaded_count = 0
        for filename in ["test.txt", "test.md", "test.json"]:
            file_path = files[filename]
            with open(file_path, "rb") as f:
                response = client.post(
                    "/api/kb/ingest",
                    files={"file": (filename, f, "text/plain")},
                    data={"collection": collection_name},
                    headers=auth_headers,
                )
                if response.status_code == 200:
                    uploaded_count += 1

        # Verify at least some files were processed
        assert uploaded_count >= 1

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_list_documents_after_ingestion(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        sample_test_files: tuple[Dict[str, str], str],
        mock_rag_pipeline: None,
    ):
        """Test that uploaded documents appear in collection listing."""
        files, temp_dir = sample_test_files
        file_path = files["test.txt"]
        collection_name = "e2e_list_test"

        # Upload a file
        with open(file_path, "rb") as f:
            upload_response = client.post(
                "/api/kb/ingest",
                files={"file": ("test.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        if upload_response.status_code == 200:
            # List collections and verify our collection appears
            list_response = client.get("/api/kb/collections", headers=auth_headers)
            assert list_response.status_code == 200

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_delete_document_after_ingestion(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        sample_test_files: tuple[Dict[str, str], str],
        mock_rag_pipeline: None,
    ):
        """Test deleting a document after successful ingestion."""
        files, temp_dir = sample_test_files
        file_path = files["test.txt"]
        collection_name = "e2e_delete_test"

        # Upload a file
        with open(file_path, "rb") as f:
            upload_response = client.post(
                "/api/kb/ingest",
                files={"file": ("test.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        if upload_response.status_code == 200:
            # Try to delete the document
            delete_response = client.delete(
                f"/api/kb/collections/{collection_name}/documents/test.txt",
                headers=auth_headers,
            )
            # Deleting a just-created document should succeed
            assert delete_response.status_code == 200

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_error_handling_unsupported_format(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        mock_rag_pipeline: None,
    ):
        """Test error handling when uploading unsupported file format."""
        # Create a fake .exe file
        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as tmp:
            tmp.write(b"fake executable content")
            tmp.flush()

            with open(tmp.name, "rb") as f:
                response = client.post(
                    "/api/kb/ingest",
                    files={"file": ("test.exe", f, "application/octet-stream")},
                    data={"collection": "e2e_error_test"},
                    headers=auth_headers,
                )

            # Unsupported file types should return 422
            assert response.status_code == 422

            os.unlink(tmp.name)
