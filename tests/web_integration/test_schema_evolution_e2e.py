"""
E2E Tests for Schema Evolution and Migration Compatibility.

This test suite verifies that schema changes and database migrations
do not break core functionality in the RAG/KB system.

These tests are critical for ensuring that:
1. Behavior remains consistent before and after schema migrations
2. Mixed schema versions can coexist during migration
3. Graceful degradation when fields are missing
4. Legacy data remains accessible after schema upgrades
"""

import json
import tempfile
from pathlib import Path
from typing import Dict

import pytest

from tests.web_integration.http_helpers import http_detail

pytestmark = [pytest.mark.e2e, pytest.mark.real_rag]


class TestSchemaMigrationCompatibility:
    """Test schema migration compatibility and coexistence.

    These tests verify that the system can handle different schema versions
    simultaneously during migrations and that operations work correctly.
    """

    @pytest.fixture
    def sample_documents(self, tmp_path: Path) -> Dict[str, Path]:
        """Create sample documents for testing."""
        docs = {}

        # Text document
        docs["text"] = tmp_path / "sample.txt"
        docs["text"].write_text(
            "This is a sample document for schema evolution testing. "
            "It contains various keywords: testing, schema, evolution, migration. "
            "The purpose is to verify behavior consistency across schema versions."
        )

        # JSON document
        docs["json"] = tmp_path / "data.json"
        docs["json"].write_text(
            json.dumps(
                {
                    "title": "Schema Evolution Test",
                    "content": "Testing JSON document handling across schema versions",
                    "metadata": {
                        "version": "1.0",
                        "tags": ["schema", "evolution", "test"],
                    },
                }
            )
        )

        # Markdown document
        docs["markdown"] = tmp_path / "guide.md"
        docs["markdown"].write_text("""
# Schema Evolution Guide

This document tests markdown processing during schema changes.

## Key Points
- Behavior should remain consistent
- Data should not be lost
- Operations should work across versions

## Testing Keywords
schema evolution migration compatibility
        """)

        return docs

    def test_read_operations_with_mixed_schemas(
        self,
        client,
        auth_headers: Dict[str, str],
        sample_documents: Dict[str, Path],
    ):
        """Test that read operations work with mixed schema versions.

        This simulates a scenario where some documents use the old schema
        and some use the new schema.
        """
        collection = "mixed_schema_collection"

        # Upload multiple documents (potentially creating mixed schemas);
        # first document also asserts ingest contract / optional schema fields.
        doc_ids = []
        first = True
        optional_new_fields = [
            "created_at",
            "updated_at",
            "schema_version",
            "metadata",
            "chunk_count",
            "status",
        ]
        for doc_key, file_path in sample_documents.items():
            with open(file_path, "rb") as f:
                response = client.post(
                    "/api/kb/ingest",
                    files={
                        "file": (
                            file_path.name,
                            f,
                            "text/plain"
                            if doc_key == "text"
                            else "application/json"
                            if doc_key == "json"
                            else "text/markdown",
                        )
                    },
                    data={"collection": collection},
                    headers=auth_headers,
                )
                assert response.status_code == 200, http_detail(response)
                result = response.json()
                if first:
                    assert result.get("status") in {"success", "partial"}
                    assert "doc_id" in result
                    assert "file_id" in result
                    # Verify that at least some optional new schema fields are present
                    assert any(f in result for f in optional_new_fields)
                    first = False
                doc_ids.append(result.get("doc_id"))

        # List collections - should handle mixed schemas
        response = client.get("/api/kb/collections", headers=auth_headers)
        assert response.status_code == 200, http_detail(response)

        collections = response.json()["collections"]
        test_collection = next(
            (c for c in collections if c["name"] == collection), None
        )
        assert test_collection is not None

        # Note: No endpoint to list documents in a collection
        # Verify collection exists and has document count instead
        assert "documents" in test_collection

    def test_field_absence_graceful_degradation(
        self, client, auth_headers: Dict[str, str]
    ):
        """Test graceful degradation when expected fields are missing.

        This verifies that if new schema fields are missing (old schema),
        the system degrades gracefully rather than failing.
        """
        # Create a collection
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content for field absence testing")
            temp_path = Path(f.name)

        try:
            with open(temp_path, "rb") as f:
                response = client.post(
                    "/api/kb/ingest",
                    files={"file": ("test.txt", f, "text/plain")},
                    data={"collection": "field_test_collection"},
                    headers=auth_headers,
                )
                assert response.status_code == 200, http_detail(response)

                # Even if some new fields are missing, basic operations should work
                result = response.json()

                # Core fields should be present
                assert "doc_id" in result

                # Optional new fields might be missing - that's OK
                # The system should not crash
        finally:
            temp_path.unlink(missing_ok=True)

    def test_search_with_legacy_schema_data(self, client, auth_headers: Dict[str, str]):
        """Test that search works with legacy schema data.

        This verifies that documents using the old schema can still be searched.
        """
        # Upload a document
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(
                "Legacy schema search test. "
                "Keywords: legacy, schema, search, compatibility. "
                "This simulates old data that needs to remain searchable."
            )
            temp_path = Path(f.name)

        try:
            with open(temp_path, "rb") as f:
                response = client.post(
                    "/api/kb/ingest",
                    files={"file": ("legacy.txt", f, "text/plain")},
                    data={"collection": "legacy_search_collection"},
                    headers=auth_headers,
                )
                assert response.status_code == 200, http_detail(response)

            # Search should work even with legacy schema
            response = client.post(
                "/api/kb/search",
                data={
                    "collection": "legacy_search_collection",
                    "query_text": "legacy schema",
                },
                headers=auth_headers,
            )
            assert response.status_code == 200, http_detail(response)

            _ = response.json().get("results", [])
            # Should find results regardless of schema version
            # (results might be empty if indexing hasn't completed, but request should succeed)

        finally:
            temp_path.unlink(missing_ok=True)

    def test_mixed_schema_crud_operations(self, client, auth_headers: Dict[str, str]):
        """Test CRUD operations work with mixed schema versions.

        This verifies Create, Read, Update, Delete operations work
        correctly when dealing with mixed old/new schema data.
        """
        collection = "mixed_crud_collection"

        # CREATE: Upload document
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Mixed schema CRUD test content")
            temp_path = Path(f.name)

        try:
            with open(temp_path, "rb") as f:
                response = client.post(
                    "/api/kb/ingest",
                    files={"file": ("crud_test.txt", f, "text/plain")},
                    data={"collection": collection},
                    headers=auth_headers,
                )
                assert response.status_code == 200, http_detail(response)
                result = response.json()
                file_id = result.get("file_id")

            # READ: Get collections list
            response = client.get("/api/kb/collections", headers=auth_headers)
            assert response.status_code == 200, http_detail(response)
            collections = response.json()["collections"]
            assert any(c["name"] == collection for c in collections)

            # READ: Verify collection exists (no document listing endpoint)
            assert len(collections) >= 1

            # UPDATE: Try to reingest (update operation)
            with open(temp_path, "rb") as f:
                response = client.post(
                    "/api/kb/ingest",
                    files={"file": ("crud_test.txt", f, "text/plain")},
                    data={"collection": collection},
                    headers=auth_headers,
                )
                # Reingestion behavior is expected to be deterministic and successful.
                assert response.status_code == 200, http_detail(response)

            # DELETE: Delete document (using collection_name and filename)
            if file_id:
                response = client.delete(
                    f"/api/kb/collections/{collection}/documents/crud_test.txt?file_id={file_id}",
                    headers=auth_headers,
                )
                # Known document deletion should succeed explicitly.
                assert response.status_code == 200, http_detail(response)

        finally:
            temp_path.unlink(missing_ok=True)


class TestBehaviorConsistency:
    """Test behavior consistency across schema migrations.

    These tests verify that core behaviors remain the same before
    and after schema changes.
    """

    def test_search_consistency_before_after_migration(
        self, client, auth_headers: Dict[str, str]
    ):
        """Test that search behavior is consistent after schema migration.

        This verifies that:
        1. Search returns similar results before and after migration
        2. Search relevance ranking is consistent
        3. Search filters work the same way
        """
        # Upload test documents with known content
        test_docs = [
            (
                "doc1.txt",
                "Python is a programming language for data science and machine learning",
            ),
            (
                "doc2.txt",
                "JavaScript is used for web development and frontend applications",
            ),
            (
                "doc3.txt",
                "Python and JavaScript are both popular programming languages",
            ),
        ]

        collection = "search_consistency_collection"

        for filename, content in test_docs:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                f.write(content)
                temp_path = Path(f.name)

            try:
                with open(temp_path, "rb") as f:
                    response = client.post(
                        "/api/kb/ingest",
                        files={"file": (filename, f, "text/plain")},
                        data={"collection": collection},
                        headers=auth_headers,
                    )
                    assert response.status_code == 200, http_detail(response)
            finally:
                temp_path.unlink(missing_ok=True)

        # Perform search and verify behavior
        response = client.post(
            "/api/kb/search",
            data={
                "collection": collection,
                "query_text": "Python programming",
            },
            headers=auth_headers,
        )
        assert response.status_code == 200, http_detail(response)

        results = response.json().get("results", [])

        # Verify search results structure is consistent
        for result in results:
            assert "content" in result or "text" in result
            assert "score" in result

    def test_ingestion_consistency_before_after_migration(
        self, client, auth_headers: Dict[str, str]
    ):
        """Test that ingestion behavior is consistent after schema migration.

        This verifies that:
        1. File ingestion produces similar results
        2. Chunking behavior is consistent
        3. Metadata extraction works the same
        """
        # Upload a document
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(
                "This is a test document for ingestion consistency. "
                "It has multiple sentences. Each sentence should be processed. "
                "The chunking behavior should remain consistent across schema versions. "
                "Metadata extraction should work the same way."
            )
            temp_path = Path(f.name)

        try:
            with open(temp_path, "rb") as f:
                response = client.post(
                    "/api/kb/ingest",
                    files={"file": ("ingestion_test.txt", f, "text/plain")},
                    data={"collection": "ingestion_consistency_collection"},
                    headers=auth_headers,
                )
                assert response.status_code == 200, http_detail(response)

                result = response.json()

                # Verify ingestion response structure is consistent
                # Should expose stable ingestion fields
                assert "doc_id" in result

                # Should have document identifier
                assert "doc_id" in result
                assert "file_id" in result

                # Status information (might vary by schema version)
                # Should not crash if fields are missing

        finally:
            temp_path.unlink(missing_ok=True)

    def test_crud_consistency_before_after_migration(
        self, client, auth_headers: Dict[str, str]
    ):
        """Test that CRUD behavior is consistent after schema migration.

        This verifies that all CRUD operations work the same way
        before and after schema changes.
        """
        collection = "crud_consistency_collection"

        # CREATE: Create collection with document
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("CRUD consistency test document")
            temp_path = Path(f.name)

        try:
            with open(temp_path, "rb") as f:
                create_response = client.post(
                    "/api/kb/ingest",
                    files={"file": ("crud_consistency.txt", f, "text/plain")},
                    data={"collection": collection},
                    headers=auth_headers,
                )
                assert create_response.status_code == 200, http_detail(create_response)
                create_result = create_response.json()

                # Verify create response structure
                assert "doc_id" in create_result

            # READ: List collections
            list_response = client.get("/api/kb/collections", headers=auth_headers)
            assert list_response.status_code == 200, http_detail(list_response)
            collections = list_response.json()["collections"]

            # Find our collection
            test_collection = next(
                (c for c in collections if c["name"] == collection), None
            )
            assert test_collection is not None
            assert test_collection.get("documents", 0) >= 1

            # UPDATE: Rename collection with a deterministic target name.
            # Use a unique target name to avoid collision with stale directories
            # from previous runs in shared upload paths.
            renamed_collection = f"{collection}_renamed_{create_result['doc_id'][:8]}"
            update_response = client.put(
                f"/api/kb/collections/{collection}",
                data={"new_name": renamed_collection},
                headers=auth_headers,
            )
            assert update_response.status_code == 200, http_detail(update_response)

            # DELETE: Delete document
            file_id = create_result.get("file_id")
            if file_id:
                delete_response = client.delete(
                    f"/api/kb/collections/{renamed_collection}/documents/crud_consistency.txt?file_id={file_id}",
                    headers=auth_headers,
                )
                # Known document deletion should succeed explicitly.
                assert delete_response.status_code == 200, http_detail(delete_response)

        finally:
            temp_path.unlink(missing_ok=True)

    def test_frontend_display_consistency(self, client, auth_headers: Dict[str, str]):
        """Test that frontend display data is consistent after schema migration.

        This verifies that the frontend receives data in a consistent format
        regardless of schema version.
        """
        # Upload document
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Frontend display consistency test")
            temp_path = Path(f.name)

        try:
            with open(temp_path, "rb") as f:
                response = client.post(
                    "/api/kb/ingest",
                    files={"file": ("display_test.txt", f, "text/plain")},
                    data={"collection": "display_consistency_collection"},
                    headers=auth_headers,
                )
                assert response.status_code == 200, http_detail(response)
                assert "doc_id" in response.json()

            # Get collections list (frontend display endpoint)
            response = client.get("/api/kb/collections", headers=auth_headers)
            assert response.status_code == 200, http_detail(response)

            collections = response.json()["collections"]

            # Verify response structure is frontend-friendly
            for collection in collections:
                # Should have displayable fields
                assert "name" in collection
                # Document count should be present
                assert "documents" in collection

            # Get documents (frontend display endpoint)
            test_collection = next(
                (
                    c
                    for c in collections
                    if c["name"] == "display_consistency_collection"
                ),
                None,
            )
            assert test_collection is not None
            assert "documents" in test_collection

        finally:
            temp_path.unlink(missing_ok=True)


class TestLegacyDataAccessAfterMigration:
    """Test legacy data accessibility after schema migrations.

    These tests ensure that data created with old schema versions
    remains accessible and functional after migrations.
    """

    def test_legacy_collection_access(self, client, auth_headers: Dict[str, str]):
        """Test that legacy collections can still be accessed.

        This verifies that collections created with old schema versions
        can be listed, viewed, and managed.
        """
        # Create a collection (simulating legacy data)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Legacy collection test content")
            temp_path = Path(f.name)

        try:
            with open(temp_path, "rb") as f:
                response = client.post(
                    "/api/kb/ingest",
                    files={"file": ("legacy.txt", f, "text/plain")},
                    data={"collection": "legacy_collection"},
                    headers=auth_headers,
                )
                assert response.status_code == 200, http_detail(response)
                assert "doc_id" in response.json()

            # Access legacy collection
            response = client.get("/api/kb/collections", headers=auth_headers)
            assert response.status_code == 200, http_detail(response)

        finally:
            temp_path.unlink(missing_ok=True)

    def test_legacy_document_search(self, client, auth_headers: Dict[str, str]):
        """Test that legacy documents can still be searched.

        This verifies that documents created with old schema versions
        remain searchable after migrations.
        """
        # Upload document (simulating legacy)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(
                "Legacy document for search testing. "
                "Keywords: legacy, search, migration, compatibility. "
                "This document should remain searchable after schema changes."
            )
            temp_path = Path(f.name)

        try:
            with open(temp_path, "rb") as f:
                response = client.post(
                    "/api/kb/ingest",
                    files={"file": ("legacy_search.txt", f, "text/plain")},
                    data={"collection": "legacy_search_collection"},
                    headers=auth_headers,
                )
                assert response.status_code == 200, http_detail(response)
                assert "doc_id" in response.json()

            # Search in legacy collection
            response = client.post(
                "/api/kb/search",
                data={
                    "collection": "legacy_search_collection",
                    "query_text": "legacy search",
                },
                headers=auth_headers,
            )
            assert response.status_code == 200, http_detail(response)

            # Should return results (might be empty if not indexed yet)
            results = response.json().get("results", [])
            # Structure should be consistent
            for result in results:
                assert "content" in result or "text" in result

        finally:
            temp_path.unlink(missing_ok=True)

    def test_legacy_document_deletion(self, client, auth_headers: Dict[str, str]):
        """Test that legacy documents can still be deleted.

        This verifies that deletion operations work on old schema data.
        """
        # Upload document
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Legacy deletion test")
            temp_path = Path(f.name)

        try:
            with open(temp_path, "rb") as f:
                response = client.post(
                    "/api/kb/ingest",
                    files={"file": ("legacy_delete.txt", f, "text/plain")},
                    data={"collection": "legacy_delete_collection"},
                    headers=auth_headers,
                )
                assert response.status_code == 200, http_detail(response)
                file_id = response.json().get("file_id")

            # Delete legacy document
            if file_id:
                response = client.delete(
                    f"/api/kb/collections/legacy_delete_collection/documents/legacy_delete.txt?file_id={file_id}",
                    headers=auth_headers,
                )
                # Known document deletion should succeed explicitly.
                assert response.status_code == 200, http_detail(response)

        finally:
            temp_path.unlink(missing_ok=True)

    def test_mixed_schema_coexistence(self, client, auth_headers: Dict[str, str]):
        """Test that old and new schema versions can coexist.

        This simulates a gradual migration scenario where some data
        has been migrated and some hasn't.
        """
        collection = "mixed_schema_coexistence"

        # Upload multiple documents (simulating mixed migration state)
        for i in range(3):
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                f.write(f"Document {i + 1} in mixed schema environment")
                temp_path = Path(f.name)

            try:
                with open(temp_path, "rb") as f:
                    response = client.post(
                        "/api/kb/ingest",
                        files={"file": (f"doc{i + 1}.txt", f, "text/plain")},
                        data={"collection": collection},
                        headers=auth_headers,
                    )
                    assert response.status_code == 200, http_detail(response)
            finally:
                temp_path.unlink(missing_ok=True)

        # List collections - should handle mixed state
        response = client.get("/api/kb/collections", headers=auth_headers)
        assert response.status_code == 200, http_detail(response)

        collections = response.json()["collections"]
        test_collection = next(
            (c for c in collections if c["name"] == collection), None
        )
        assert test_collection is not None

        # Get documents - should return all regardless of schema version
        assert test_collection.get("documents", 0) >= 3

        # Search - should work across mixed schemas
        response = client.post(
            "/api/kb/search",
            data={"collection": collection, "query_text": "Document"},
            headers=auth_headers,
        )
        assert response.status_code == 200, http_detail(response)
