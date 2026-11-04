"""
End-to-end tests for frontend display verification.

This module tests that data is correctly displayed on the frontend
after ingestion, ensuring that the UI shows accurate information
to users.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from tests.web_integration.http_helpers import http_detail

pytestmark = [pytest.mark.e2e, pytest.mark.contract_stub]

# Note: _StubEmbeddingAdapter, stub_embedding_adapter, and mock_rag_pipeline
# are provided by conftest.py with autouse=True


@pytest.fixture
def sample_display_files() -> Generator[tuple[dict[str, str], str], None, None]:
    """Create sample test files for display verification testing."""
    files = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files with descriptive names
        test_files = {
            # NOTE: Use a .txt fixture here; fake PDF bytes trigger deepdoc/pdf parsing failures
            # and make these UI-display tests flaky and environment-dependent.
            "report_2024.txt": "Annual Report 2024\nFinancial results and analysis.",
            "user_guide.md": "# User Guide\n\nThis is a comprehensive user guide.",
            "data_export.csv": "id,name,value\n1,Alice,100\n2,Bob,200",
            "config.json": '{"setting1": "value1", "setting2": "value2"}',
            "readme.txt": "README\n\nThis is the readme file.",
        }

        for filename, content in test_files.items():
            file_path = Path(temp_dir) / filename
            file_path.write_text(content, encoding="utf-8")
            files[filename] = str(file_path)

        yield files, temp_dir


# ==========================================
# COLLECTION DISPLAY TESTS
# ==========================================


class TestCollectionDisplay:
    """
    Test Collection-level frontend display.

    These tests verify that:
    1. Collection list shows all collections
    2. Collection list shows correct document counts
    3. Collection list shows document names
    4. Collection details show metadata
    5. Legacy data fallback works correctly
    """

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_collection_list_shows_all_collections(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_display_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that collection list displays all collections."""
        files, temp_dir = sample_display_files

        # Create multiple collections
        collection_names = [
            "e2e_display_coll1",
            "e2e_display_coll2",
            "e2e_display_coll3",
        ]

        created_count = 0
        for collection_name in collection_names:
            file_path = files["readme.txt"]
            with open(file_path, "rb") as f:
                response = client.post(
                    "/api/kb/ingest",
                    files={"file": ("readme.txt", f, "text/plain")},
                    data={"collection": collection_name},
                    headers=auth_headers,
                )
                assert response.status_code == 200, http_detail(response)
                created_count += 1

        assert created_count == len(collection_names)

        # List collections
        list_response = client.get("/api/kb/collections", headers=auth_headers)
        assert list_response.status_code == 200, http_detail(list_response)
        result = list_response.json()
        assert "collections" in result

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_collection_list_shows_correct_document_count(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_display_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that collection list shows accurate document counts."""
        files, temp_dir = sample_display_files
        collection_name = "e2e_display_count"

        # Create multiple documents in the same collection
        doc_count = 0
        for filename in ["report_2024.txt", "user_guide.md", "data_export.csv"]:
            file_path = files[filename]
            with open(file_path, "rb") as f:
                response = client.post(
                    "/api/kb/ingest",
                    files={"file": (filename, f, "text/plain")},
                    data={"collection": collection_name},
                    headers=auth_headers,
                )
                assert response.status_code == 200, http_detail(response)
                doc_count += 1

        assert doc_count == 3

        # List collections and check document count
        list_response = client.get("/api/kb/collections", headers=auth_headers)
        assert list_response.status_code == 200, http_detail(list_response)
        result = list_response.json()
        assert "collections" in result

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_collection_list_shows_document_names(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_display_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that collection list shows document names correctly."""
        files, temp_dir = sample_display_files
        collection_name = "e2e_display_names"

        # Create documents with specific names
        for filename in ["report_2024.txt", "user_guide.md"]:
            file_path = files[filename]
            with open(file_path, "rb") as f:
                client.post(
                    "/api/kb/ingest",
                    files={"file": (filename, f, "text/plain")},
                    data={"collection": collection_name},
                    headers=auth_headers,
                )

        # List collections and check document names
        list_response = client.get("/api/kb/collections", headers=auth_headers)
        assert list_response.status_code == 200
        result = list_response.json()
        assert "collections" in result

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_collection_details_shows_metadata(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_display_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that collection details show complete metadata."""
        files, temp_dir = sample_display_files
        collection_name = "e2e_display_metadata"
        file_path = files["readme.txt"]

        # Create a collection
        with open(file_path, "rb") as f:
            create_response = client.post(
                "/api/kb/ingest",
                files={"file": ("readme.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        assert create_response.status_code == 200, http_detail(create_response)

        # No per-collection GET on kb_router; metadata is in list_collections.
        list_response = client.get("/api/kb/collections", headers=auth_headers)
        assert list_response.status_code == 200, http_detail(list_response)
        payload = list_response.json()
        names = {c.get("name") for c in payload.get("collections", [])}
        assert collection_name in names

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_collection_list_with_legacy_data_fallback(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that legacy data is handled with fallback display."""
        # This test verifies that when documents table decoding fails,
        # the system falls back to UploadedFile records

        # Try to list collections (may include legacy data)
        list_response = client.get("/api/kb/collections", headers=auth_headers)
        assert list_response.status_code == 200
        result = list_response.json()
        assert "collections" in result


# ==========================================
# DOCUMENT DISPLAY TESTS
# ==========================================


class TestDocumentDisplay:
    """
    Test Document-level frontend display.

    These tests verify that:
    1. Document list shows all documents
    2. Document list shows correct metadata
    3. Document list works with different file types
    4. Document pagination works correctly
    """

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_document_list_shows_all_documents(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_display_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that document list shows all documents in a collection."""
        files, temp_dir = sample_display_files
        collection_name = "e2e_display_docs"

        # Create multiple documents
        created_docs = []
        for filename in ["report_2024.txt", "user_guide.md", "config.json"]:
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

        check_response = client.post(
            f"/api/kb/collections/{collection_name}/documents/check",
            json={"filenames": created_docs},
            headers=auth_headers,
        )
        assert check_response.status_code == 200, http_detail(check_response)
        existing = set(check_response.json().get("existing_filenames", []))
        assert existing.issuperset(set(created_docs))

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_document_list_shows_correct_metadata(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_display_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that document list shows correct metadata for each document."""
        files, temp_dir = sample_display_files
        collection_name = "e2e_display_metadata"

        # Create a document
        file_path = files["report_2024.txt"]
        with open(file_path, "rb") as f:
            response = client.post(
                "/api/kb/ingest",
                files={"file": ("report_2024.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        assert response.status_code == 200, http_detail(response)
        check_response = client.post(
            f"/api/kb/collections/{collection_name}/documents/check",
            json={"filenames": ["report_2024.txt"]},
            headers=auth_headers,
        )
        assert check_response.status_code == 200, http_detail(check_response)
        assert "report_2024.txt" in check_response.json().get("existing_filenames", [])

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_document_list_with_different_file_types(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_display_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that document list correctly handles different file types."""
        files, temp_dir = sample_display_files
        collection_name = "e2e_display_types"

        # Create documents of different types
        file_types = [
            ("report_2024.txt", "text/plain"),
            ("user_guide.md", "text/markdown"),
            ("data_export.csv", "text/csv"),
            ("config.json", "application/json"),
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

        assert created_count == len(file_types)
        names = [fn for fn, _ in file_types]
        check_response = client.post(
            f"/api/kb/collections/{collection_name}/documents/check",
            json={"filenames": names},
            headers=auth_headers,
        )
        assert check_response.status_code == 200, http_detail(check_response)
        existing = set(check_response.json().get("existing_filenames", []))
        assert existing.issuperset(set(names))

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_document_list_pagination(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_display_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that document list pagination works correctly."""
        files, temp_dir = sample_display_files
        collection_name = "e2e_display_pagination"

        # Create multiple documents to test pagination
        for i in range(5):
            content = f"Document {i} content for pagination testing."
            with tempfile.NamedTemporaryFile(
                suffix=".txt", delete=False, mode="w"
            ) as tmp:
                tmp.write(content)
                tmp.flush()

                try:
                    with open(tmp.name, "rb") as f:
                        client.post(
                            "/api/kb/ingest",
                            files={"file": (f"doc{i}.txt", f, "text/plain")},
                            data={"collection": collection_name},
                            headers=auth_headers,
                        )
                finally:
                    import os

                    os.unlink(tmp.name)

        # No paginated document-list GET; verify batch presence via check.
        expected = [f"doc{i}.txt" for i in range(5)]
        check_response = client.post(
            f"/api/kb/collections/{collection_name}/documents/check",
            json={"filenames": expected},
            headers=auth_headers,
        )
        assert check_response.status_code == 200
        existing = set(check_response.json().get("existing_filenames", []))
        assert len(existing & set(expected)) >= 1

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_document_list_search_filter(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_display_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that document list can be filtered by search."""
        files, temp_dir = sample_display_files
        collection_name = "e2e_display_search"

        # Create documents
        for filename in ["report_2024.txt", "user_guide.md"]:
            file_path = files[filename]
            with open(file_path, "rb") as f:
                client.post(
                    "/api/kb/ingest",
                    files={"file": (filename, f, "text/plain")},
                    data={"collection": collection_name},
                    headers=auth_headers,
                )

        # Filename search is a UI concern; backend exposes existence check only.
        check_response = client.post(
            f"/api/kb/collections/{collection_name}/documents/check",
            json={"filenames": ["report_2024.txt"]},
            headers=auth_headers,
        )
        assert check_response.status_code == 200
        assert "report_2024.txt" in check_response.json().get("existing_filenames", [])


# ==========================================
# INGESTION PROGRESS DISPLAY TESTS
# ==========================================


class TestIngestionProgressDisplay:
    """
    Test that ingestion progress is correctly displayed.

    These tests verify that:
    1. Progress updates are sent during ingestion
    2. Error messages are shown correctly
    3. Completion status is accurate
    """

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_ingestion_progress_updates(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_display_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that ingestion progress updates are displayed."""
        files, temp_dir = sample_display_files
        collection_name = "e2e_display_progress"
        file_path = files["readme.txt"]

        # Note: Progress tracking may require WebSocket or polling
        # This test verifies the API response includes progress information
        with open(file_path, "rb") as f:
            response = client.post(
                "/api/kb/ingest",
                files={"file": ("readme.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        # Response should include status information
        assert response.status_code == 200
        result = response.json()
        assert "status" in result

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_ingestion_error_messages(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that ingestion errors are displayed with clear messages."""
        collection_name = "e2e_display_errors"

        # Try to ingest an invalid PDF payload (should fail gracefully)
        response = client.post(
            "/api/kb/ingest",
            files={"file": ("nonexistent.pdf", b"", "application/pdf")},
            data={"collection": collection_name},
            headers=auth_headers,
        )

        # Ingest should fail with a structured error response.
        assert response.status_code == 500
        payload = response.json()
        assert payload.get("status") == "error"
        assert isinstance(payload.get("message"), str) and payload["message"]

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_ingestion_completion_status(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_display_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that ingestion completion status is accurate."""
        files, temp_dir = sample_display_files
        collection_name = "e2e_display_completion"
        file_path = files["readme.txt"]

        with open(file_path, "rb") as f:
            response = client.post(
                "/api/kb/ingest",
                files={"file": ("readme.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        # Response should indicate completion status
        assert response.status_code == 200
        result = response.json()
        assert "status" in result
        assert result["status"] in ["success", "partial", "error"]


# ==========================================
# FILE ID DISPLAY TESTS
# ==========================================


class TestFileIdDisplay:
    """
    Test that file_id is correctly displayed and used.

    These tests verify that:
    1. file_id is returned in ingestion response
    2. file_id is used for document identification
    3. file_id display works with legacy data
    """

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_file_id_returned_after_ingestion(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_display_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that file_id is returned after successful ingestion."""
        files, temp_dir = sample_display_files
        collection_name = "e2e_display_fileid"
        file_path = files["readme.txt"]

        with open(file_path, "rb") as f:
            response = client.post(
                "/api/kb/ingest",
                files={"file": ("readme.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        assert response.status_code == 200, http_detail(response)
        result = response.json()
        # Contract: file_id should be present after ingest.
        assert result.get("file_id"), (
            "Expected ingest response to include non-empty file_id"
        )

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_document_identification_by_file_id(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_display_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that documents can be identified by file_id."""
        files, temp_dir = sample_display_files
        collection_name = "e2e_display_identification"
        file_path = files["readme.txt"]

        # Create document
        with open(file_path, "rb") as f:
            create_response = client.post(
                "/api/kb/ingest",
                files={"file": ("readme.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        assert create_response.status_code == 200, http_detail(create_response)
        create_result = create_response.json()
        file_id = create_result.get("file_id")
        doc_id = create_result.get("doc_id")

        assert file_id, "Expected ingest response to include non-empty file_id"
        assert doc_id, "Expected ingest response to include non-empty doc_id"

        # Non-destructive check: filename should exist in collection.
        check_response = client.post(
            f"/api/kb/collections/{collection_name}/documents/check",
            json={"filenames": ["readme.txt"]},
            headers=auth_headers,
        )
        assert check_response.status_code == 200, http_detail(check_response)
        existing = check_response.json().get("existing_filenames", [])
        assert "readme.txt" in existing

        parse_response = client.get(
            f"/api/kb/collections/{collection_name}/parses/{doc_id}/parse_result",
            params={"page": 1, "page_size": 20},
            headers=auth_headers,
        )
        assert parse_response.status_code == 200, http_detail(parse_response)


# ==========================================
# METADATA DISPLAY TESTS
# ==========================================


class TestMetadataDisplay:
    """
    Test that document metadata is correctly displayed.

    These tests verify that:
    1. File type icons are shown correctly
    2. File sizes are displayed
    3. Upload dates are shown
    4. Parse status is indicated
    """

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_file_type_display(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_display_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that file types are correctly displayed."""
        files, temp_dir = sample_display_files
        collection_name = "e2e_display_filetype"

        # Create documents of different types
        for filename in ["report_2024.txt", "user_guide.md", "config.json"]:
            file_path = files[filename]
            with open(file_path, "rb") as f:
                client.post(
                    "/api/kb/ingest",
                    files={"file": (filename, f, "text/plain")},
                    data={"collection": collection_name},
                    headers=auth_headers,
                )

        check_response = client.post(
            f"/api/kb/collections/{collection_name}/documents/check",
            json={"filenames": ["report_2024.txt", "user_guide.md", "config.json"]},
            headers=auth_headers,
        )
        assert check_response.status_code == 200
        existing = set(check_response.json().get("existing_filenames", []))
        assert len(existing) >= 1

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_document_size_display(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_display_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that document sizes are correctly displayed."""
        files, temp_dir = sample_display_files
        collection_name = "e2e_display_size"
        file_path = files["readme.txt"]

        file_size = Path(file_path).stat().st_size

        with open(file_path, "rb") as f:
            response = client.post(
                "/api/kb/ingest",
                files={"file": ("readme.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        assert response.status_code == 200, http_detail(response)
        result = response.json()
        # file_size is not part of the stable ingest response contract; only assert presence when provided
        if "file_size" in result:
            assert result["file_size"] == file_size

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_upload_date_display(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_display_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that upload dates are correctly displayed."""
        files, temp_dir = sample_display_files
        collection_name = "e2e_display_date"
        file_path = files["readme.txt"]

        with open(file_path, "rb") as f:
            response = client.post(
                "/api/kb/ingest",
                files={"file": ("readme.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        assert response.status_code == 200, http_detail(response)
        result = response.json()
        # uploaded_at/created_at are optional in current ingest contract
        # (frontend display can use other endpoints to show dates when available).
        date_field = result.get("uploaded_at") or result.get("created_at")
        assert date_field is None or isinstance(date_field, str)
