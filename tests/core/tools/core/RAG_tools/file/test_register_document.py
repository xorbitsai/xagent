"""Tests for register_document functionality."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import (
    ConfigurationError,
    DatabaseOperationError,
    DocumentValidationError,
    HashComputationError,
)
from xagent.core.tools.core.RAG_tools.file.register_document import (
    list_documents,
    register_document,
)
from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
    LanceDBVectorIndexStore,
)
from xagent.providers.vector_store.lancedb import get_connection_from_env


class TestRegisterDocument:
    """Test cases for register_document function."""

    def test_register_document_new_document(self, tmp_path: Path, monkeypatch) -> None:
        """Test registering a new document successfully."""
        # Setup environment variable
        db_dir = tmp_path / "lancedb"
        monkeypatch.setenv("LANCEDB_DIR", str(db_dir))

        # Setup test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        response = register_document(
            collection="test_collection",
            source_path=str(test_file),
        )

        assert response["doc_id"] is not None
        assert response["created"] is True
        assert response["content_hash"] is not None
        assert len(response["content_hash"]) == 64  # SHA256 hash length

    def test_register_document_persists_file_id(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """New registrations should persist file_id to the documents table."""
        db_dir = tmp_path / "lancedb"
        monkeypatch.setenv("LANCEDB_DIR", str(db_dir))

        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        response = register_document(
            collection="test_collection",
            source_path=str(test_file),
            file_id="file-123",
        )

        conn = get_connection_from_env()
        table = conn.open_table("documents")
        record = (
            table.search()
            .where(
                f"collection = 'test_collection' AND doc_id = '{response['doc_id']}'"
            )
            .to_pandas()
            .iloc[0]
        )
        assert record["file_id"] == "file-123"

    def test_register_document_preserves_none_file_id(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Registrations without file_id should persist None (not empty string)."""
        db_dir = tmp_path / "lancedb"
        monkeypatch.setenv("LANCEDB_DIR", str(db_dir))

        test_file = tmp_path / "test_no_file_id.txt"
        test_file.write_text("Test content without file_id")

        response = register_document(
            collection="test_collection",
            source_path=str(test_file),
            file_id=None,
        )

        conn = get_connection_from_env()
        table = conn.open_table("documents")
        record = (
            table.search()
            .where(
                f"collection = 'test_collection' AND doc_id = '{response['doc_id']}'"
            )
            .to_pandas()
            .iloc[0]
        )
        # Should be None, not empty string (new registrations preserve None)
        # Use pd.isna to handle both None and pandas NA values
        import pandas as pd

        assert record["file_id"] != "", "file_id must not be empty string for new rows"
        assert record["file_id"] is None or pd.isna(record["file_id"]), (
            f"file_id should be None or NA, got {record['file_id']!r} "
            f"(type: {type(record['file_id']).__name__})"
        )

    def test_register_document_auto_file_type_detection(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Test automatic file type detection from extension."""
        # Setup environment variable
        db_dir = tmp_path / "lancedb"
        monkeypatch.setenv("LANCEDB_DIR", str(db_dir))

        # Setup test file with .md extension
        test_file = tmp_path / "test.md"
        test_file.write_text("# Test Markdown")

        response = register_document(
            collection="test_collection",
            source_path=str(test_file),
        )

        assert response["doc_id"] is not None
        assert response["created"] is True
        assert response["content_hash"] is not None

    def test_register_document_idempotency(self, tmp_path: Path, monkeypatch) -> None:
        """Test idempotent registration with same doc_id."""
        # Setup environment variable
        db_dir = tmp_path / "lancedb"
        monkeypatch.setenv("LANCEDB_DIR", str(db_dir))

        # Setup test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        doc_id = "test-doc-123"

        # First registration
        response1 = register_document(
            collection="test_collection",
            source_path=str(test_file),
            doc_id=doc_id,
        )
        assert response1["doc_id"] == doc_id
        assert response1["created"] is True

        # Second registration with same doc_id
        response2 = register_document(
            collection="test_collection",
            source_path=str(test_file),
            doc_id=doc_id,
        )
        assert response2["doc_id"] == doc_id
        assert response2["created"] is False  # Should be update, not create

    def test_register_document_deterministic_doc_id_idempotency(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Without doc_id, same (collection, source_path) yields same doc_id and second is update."""
        db_dir = tmp_path / "lancedb"
        monkeypatch.setenv("LANCEDB_DIR", str(db_dir))

        test_file = tmp_path / "report.docx"
        test_file.write_text("content")

        response1 = register_document(
            collection="my_kb",
            source_path=str(test_file),
        )
        response2 = register_document(
            collection="my_kb",
            source_path=str(test_file),
        )

        assert response1["doc_id"] == response2["doc_id"]
        assert response1["created"] is True
        assert response2["created"] is False

    def test_register_document_custom_uploaded_at(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Test registration with custom uploaded_at timestamp."""
        # Setup environment variable
        db_dir = tmp_path / "lancedb"
        monkeypatch.setenv("LANCEDB_DIR", str(db_dir))

        # Setup test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        custom_time = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)

        response = register_document(
            collection="test_collection",
            source_path=str(test_file),
            uploaded_at=custom_time.isoformat(),
        )

        assert response["doc_id"] is not None
        assert response["created"] is True
        assert response["content_hash"] is not None

    def test_register_document_empty_file(self, tmp_path: Path, monkeypatch) -> None:
        """Test registering an empty file."""
        # Setup environment variable
        db_dir = tmp_path / "lancedb"
        monkeypatch.setenv("LANCEDB_DIR", str(db_dir))

        # Setup empty test file
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        response = register_document(
            collection="test_collection",
            source_path=str(test_file),
        )

        assert response["doc_id"] is not None
        assert response["created"] is True
        assert response["content_hash"] is not None
        # Empty file should have a hash (hash of empty string)
        assert response["content_hash"] == (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )

    def test_register_document_different_collection(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Test registration in different collection."""
        # Setup environment variable
        db_dir = tmp_path / "lancedb"
        monkeypatch.setenv("LANCEDB_DIR", str(db_dir))

        # Setup test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        response = register_document(
            collection="different_collection",
            source_path=str(test_file),
        )

        assert response["doc_id"] is not None
        assert response["created"] is True
        assert response["content_hash"] is not None

    def test_register_document_invalid_path(self) -> None:
        """Test registration with invalid file path."""
        with pytest.raises(DocumentValidationError, match="Source path does not exist"):
            register_document(
                collection="test_collection", source_path="/nonexistent/file.txt"
            )

    def test_register_document_empty_collection(self) -> None:
        """Test registration with empty collection name."""
        with pytest.raises(
            DocumentValidationError, match="Collection name cannot be empty"
        ):
            register_document(collection="", source_path="/tmp/test.txt")

    @patch("xagent.core.tools.core.RAG_tools.kb.collection_handle.compute_file_hash")
    def test_register_document_hash_computation_error(
        self, mock_hash, tmp_path: Path, monkeypatch
    ) -> None:
        """Test handling hash computation errors.

        The hash computation now lives in the collection handle; patch it there.
        """
        # Setup environment variable
        db_dir = tmp_path / "lancedb"
        monkeypatch.setenv("LANCEDB_DIR", str(db_dir))

        # Setup test file
        test_file = tmp_path / "hash_error_test.txt"
        test_file.write_text("Test content")

        # Mock hash computation to raise error
        mock_hash.side_effect = HashComputationError("Hash computation failed")

        # Should propagate HashComputationError
        with pytest.raises(HashComputationError):
            register_document(collection="test_collection", source_path=str(test_file))

    @patch.object(
        LanceDBVectorIndexStore,
        "count_rows_or_zero",
        side_effect=ConfigurationError("LANCEDB_DIR not configured"),
    )
    def test_register_document_configuration_error(
        self, _mock_count, tmp_path: Path, monkeypatch
    ) -> None:
        """Test handling configuration errors.

        The handle performs the existence check on the bound vector store; a
        ConfigurationError from the store must propagate unchanged.
        """
        db_dir = tmp_path / "lancedb"
        monkeypatch.setenv("LANCEDB_DIR", str(db_dir))

        test_file = tmp_path / "config_error_test.txt"
        test_file.write_text("Test content")

        # Should propagate ConfigurationError
        with pytest.raises(ConfigurationError):
            register_document(collection="test_collection", source_path=str(test_file))

    def test_register_document_unsupported_file_type(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Test registering document with unsupported file type."""
        # Setup environment variable
        db_dir = tmp_path / "lancedb"
        monkeypatch.setenv("LANCEDB_DIR", str(db_dir))

        # Create a file with unsupported extension
        unsupported_file = tmp_path / "test.unsupported"
        unsupported_file.write_text("Unsupported file content")

        collection = "test_collection"

        # Should raise DocumentValidationError for unsupported file type
        with pytest.raises(
            DocumentValidationError, match="Unsupported file type: \\.unsupported"
        ):
            register_document(collection=collection, source_path=str(unsupported_file))

    @patch.object(
        LanceDBVectorIndexStore,
        "count_rows_or_zero",
        side_effect=Exception("Table access failed"),
    )
    def test_register_document_database_operation_error(
        self, _mock_count, tmp_path: Path, monkeypatch
    ) -> None:
        """Test handling database operation errors.

        A generic store failure during the handle's existence check must be
        wrapped as DatabaseOperationError.
        """
        # Setup environment variable
        db_dir = tmp_path / "lancedb"
        monkeypatch.setenv("LANCEDB_DIR", str(db_dir))

        # Setup test file
        test_file = tmp_path / "db_error_test.txt"
        test_file.write_text("Test content")

        # Should propagate DatabaseOperationError
        with pytest.raises(DatabaseOperationError, match="Table access failed"):
            register_document(collection="test_collection", source_path=str(test_file))


class TestListDocuments:
    """Test list_documents function (collection filter for KB isolation)."""

    def test_list_documents_filters_by_collection(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """List_documents must return only documents from the requested collection (Issue #72)."""
        db_dir = tmp_path / "lancedb"
        db_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("LANCEDB_DIR", str(db_dir))

        # Register one doc in collection A
        file_a = tmp_path / "a.txt"
        file_a.write_text("Doc A")
        register_document(
            collection="coll_a",
            source_path=str(file_a),
            doc_id="doc-a",
        )

        # Register one doc in collection B
        file_b = tmp_path / "b.txt"
        file_b.write_text("Doc B")
        register_document(
            collection="coll_b",
            source_path=str(file_b),
            doc_id="doc-b",
        )

        # List only coll_a: must not include coll_b docs
        results_a = list_documents(str(db_dir), collection="coll_a", limit=100)
        assert len(results_a) == 1
        assert results_a[0]["collection"] == "coll_a"
        assert results_a[0]["doc_id"] == "doc-a"

        # List only coll_b: must not include coll_a docs
        results_b = list_documents(str(db_dir), collection="coll_b", limit=100)
        assert len(results_b) == 1
        assert results_b[0]["collection"] == "coll_b"
        assert results_b[0]["doc_id"] == "doc-b"

    def test_list_documents_empty_collection_returns_empty(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """List_documents for a collection with no docs returns empty list."""
        db_dir = tmp_path / "lancedb"
        db_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("LANCEDB_DIR", str(db_dir))

        results = list_documents(str(db_dir), collection="no_such_coll", limit=10)
        assert results == []
