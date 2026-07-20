"""Tests for parse_document functionality (core layer).

This module validates the parse pipeline contracts:
register_document -> parse_document
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import (
    DocumentNotFoundError,
    DocumentValidationError,
)
from xagent.core.tools.core.RAG_tools.core.schemas import ParseMethod
from xagent.core.tools.core.RAG_tools.file.register_document import register_document
from xagent.core.tools.core.RAG_tools.parse.parse_document import parse_document

RESOURCES_DIR = Path("tests/resources/test_files")


@pytest.fixture
def temp_lancedb_dir():
    """Create a temporary LanceDB directory for testing by pointing to a unique subdir.

    We intentionally use a per-test unique subdirectory under the project test lancedb root to avoid cross-test pollution.
    """
    # If the project defines a dev dir, still isolate per test
    base_dir = Path(os.environ.get("LANCEDB_DIR", "/tmp/.lancedb_test_root")).resolve()
    unique_dir = base_dir / f"pytest_{uuid.uuid4().hex[:8]}"
    unique_dir.mkdir(parents=True, exist_ok=True)
    old_dir = os.environ.get("LANCEDB_DIR")
    os.environ["LANCEDB_DIR"] = str(unique_dir)
    try:
        yield str(unique_dir)
    finally:
        # Restore original env to avoid side effects on other tests
        if old_dir is not None:
            os.environ["LANCEDB_DIR"] = old_dir
        else:
            os.environ.pop("LANCEDB_DIR", None)
        # Cleanup temp directory
        import shutil

        if unique_dir.exists():
            shutil.rmtree(unique_dir)


@pytest.fixture
def test_collection() -> str:
    return f"test_collection_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def test_doc_id() -> str:
    return str(uuid.uuid4())


class TestParseDocumentCore:
    """Core parse_document tests using files under tests/resources/test_files.

    These tests assume the sample files exist. If a specific sample is missing on the filesystem,
    the corresponding test will be skipped to keep CI green.
    """

    def _require_file(self, relative: str) -> Path:
        p = RESOURCES_DIR / relative
        if not p.exists():
            pytest.skip(f"Sample file not found: {p}")
        return p

    def test_parse_txt_default_happy_path(
        self, temp_lancedb_dir: str, test_collection: str, test_doc_id: str
    ) -> None:
        sample = self._require_file("test.txt")
        reg = register_document(
            collection=test_collection,
            source_path=str(sample),
            doc_id=test_doc_id,
            user_id=1,
        )
        assert reg["created"] is True
        out = parse_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_method=ParseMethod.DEEPDOC,
            user_id=1,
            is_admin=True,
        )
        assert out["written"] is True
        assert out["doc_id"] == test_doc_id
        assert isinstance(out.get("parse_hash"), str) and len(out["parse_hash"]) > 0
        assert isinstance(out.get("paragraphs"), list)
        # For non-empty txt, expect at least one paragraph
        if sample.read_text(encoding="utf-8").strip():
            assert len(out["paragraphs"]) >= 1
        # Metadata presence
        if out["paragraphs"]:
            meta = out["paragraphs"][0]["metadata"]
            assert meta.get("source")
            assert (
                meta.get("file_type") == "txt"
            )  # file_type from database is without dot
            assert meta.get("parse_method") == ParseMethod.DEEPDOC.value
            assert meta.get("parser")

    def test_idempotency_same_doc_same_params(
        self, temp_lancedb_dir: str, test_collection: str, test_doc_id: str
    ) -> None:
        sample = self._require_file("test.md")
        register_document(
            collection=test_collection,
            source_path=str(sample),
            doc_id=test_doc_id,
            user_id=1,
        )
        first = parse_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_method=ParseMethod.DEEPDOC,
            user_id=1,
            is_admin=True,
        )
        second = parse_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_method=ParseMethod.DEEPDOC,
            user_id=1,
            is_admin=True,
        )
        assert first["written"] is True
        assert second["written"] is False  # idempotent path
        # paragraphs should be materialized from DB on second call
        assert isinstance(second.get("paragraphs"), list)
        assert len(second["paragraphs"]) == len(first["paragraphs"])  # basic stability

    def test_collection_isolation(self, temp_lancedb_dir: str) -> None:
        sample = self._require_file("test.txt")
        doc_id = str(uuid.uuid4())
        c1 = f"c1_{uuid.uuid4().hex[:6]}"
        c2 = f"c2_{uuid.uuid4().hex[:6]}"
        # Register same file under two collections
        register_document(
            collection=c1, source_path=str(sample), doc_id=doc_id, user_id=1
        )
        register_document(
            collection=c2, source_path=str(sample), doc_id=doc_id, user_id=1
        )
        p1 = parse_document(
            collection=c1,
            doc_id=doc_id,
            parse_method=ParseMethod.DEEPDOC,
            user_id=1,
            is_admin=True,
        )
        p2 = parse_document(
            collection=c2,
            doc_id=doc_id,
            parse_method=ParseMethod.DEEPDOC,
            user_id=1,
            is_admin=True,
        )
        assert p1["written"] is True
        assert p2["written"] is True
        assert p1["parse_hash"] == p2["parse_hash"]  # same method/params
        # Both parses should return the same number of paragraphs
        assert len(p1["paragraphs"]) == len(p2["paragraphs"])

    def test_parse_pdf_pypdf(
        self, temp_lancedb_dir: str, test_collection: str, test_doc_id: str
    ) -> None:
        sample = self._require_file("test.pdf")
        register_document(
            collection=test_collection,
            source_path=str(sample),
            doc_id=test_doc_id,
            user_id=1,
        )
        out = parse_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_method=ParseMethod.PYPDF,
            user_id=1,
            is_admin=True,
        )
        assert out["doc_id"] == test_doc_id
        # Successful parse should write data
        assert out["written"] is True
        assert isinstance(out.get("paragraphs"), list)
        # Valid sample PDF should contain at least some paragraphs
        assert len(out["paragraphs"]) > 0

    def test_invalid_params_rejected(
        self, temp_lancedb_dir: str, test_collection: str, test_doc_id: str
    ) -> None:
        sample = self._require_file("test.txt")
        register_document(
            collection=test_collection,
            source_path=str(sample),
            doc_id=test_doc_id,
            user_id=1,
        )
        with pytest.raises(DocumentValidationError):
            parse_document(
                collection=test_collection,
                doc_id=test_doc_id,
                parse_method=ParseMethod.DEEPDOC,
                params={"unknown_flag": True},
                user_id=1,
                is_admin=True,
            )

    def test_document_not_found_raises(self, temp_lancedb_dir: str) -> None:
        with pytest.raises(DocumentNotFoundError):
            parse_document(
                collection=f"c_{uuid.uuid4().hex[:6]}",
                doc_id=str(uuid.uuid4()),
                parse_method=ParseMethod.DEEPDOC,
                user_id=1,
                is_admin=True,
            )

    def test_staged_parse_uses_physical_path_when_canonical_absent(
        self,
        tmp_path,
        temp_lancedb_dir: str,
        test_collection: str,
        test_doc_id: str,
    ) -> None:
        """Regression for GH #931: staged ingestion must parse the physical staged
        file while durable metadata keeps the canonical path, even though that
        canonical file is not published until after ingestion succeeds.
        """
        staged = tmp_path / "staged.txt"
        staged.write_text("staged marker content", encoding="utf-8")
        canonical = tmp_path / "canonical" / "doc.txt"
        assert not canonical.exists()

        register_document(
            collection=test_collection,
            source_path=str(staged),
            metadata_source_path=str(canonical),
            doc_id=test_doc_id,
            user_id=1,
        )

        out = parse_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_method=ParseMethod.DEEPDOC,
            user_id=1,
            is_admin=True,
            source_path_override=str(staged),
        )

        # Parse succeeded reading the staged file; canonical never had to exist.
        assert out["written"] is True
        assert not canonical.exists()
        joined = " ".join(p["text"] for p in out["paragraphs"])
        assert "staged marker content" in joined

        # Durable metadata keeps the canonical path (acceptance criterion #2):
        # both the document row and paragraph metadata store the canonical path,
        # not the transient staged path that gets removed after publish.
        from xagent.core.tools.core.RAG_tools.parse.parse_document import (
            _get_document_from_db,
        )

        document = _get_document_from_db(
            collection=test_collection, doc_id=test_doc_id, user_id=1, is_admin=True
        )
        assert document is not None
        assert document["source_path"] == str(canonical)
        assert out["paragraphs"][0]["metadata"]["source"] == str(canonical)

    def test_staged_parse_reads_new_staged_bytes_over_existing_canonical(
        self,
        tmp_path,
        temp_lancedb_dir: str,
        test_collection: str,
        test_doc_id: str,
    ) -> None:
        """Updating an existing target must parse the new staged bytes, not the
        previous canonical file contents (GH #931 second consequence)."""
        canonical = tmp_path / "canonical" / "doc.txt"
        canonical.parent.mkdir(parents=True, exist_ok=True)
        canonical.write_text("old canonical content", encoding="utf-8")
        staged = tmp_path / "staged.txt"
        staged.write_text("new staged content", encoding="utf-8")

        register_document(
            collection=test_collection,
            source_path=str(staged),
            metadata_source_path=str(canonical),
            doc_id=test_doc_id,
            user_id=1,
        )

        out = parse_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_method=ParseMethod.DEEPDOC,
            user_id=1,
            is_admin=True,
            source_path_override=str(staged),
        )

        assert out["written"] is True
        joined = " ".join(p["text"] for p in out["paragraphs"])
        assert "new staged content" in joined
        assert "old canonical content" not in joined

        # Durable metadata still stores the canonical path (acceptance criterion #2).
        from xagent.core.tools.core.RAG_tools.parse.parse_document import (
            _get_document_from_db,
        )

        document = _get_document_from_db(
            collection=test_collection, doc_id=test_doc_id, user_id=1, is_admin=True
        )
        assert document is not None
        assert document["source_path"] == str(canonical)


class TestParseDocumentFallback:
    """Test three-tier fallback logic for parse_document internal functions."""

    @pytest.fixture
    def temp_lancedb_dir(self):
        """Create a temporary LanceDB directory for testing."""
        base_dir = Path(
            os.environ.get("LANCEDB_DIR", "/tmp/.lancedb_test_root")
        ).resolve()
        unique_dir = base_dir / f"pytest_{uuid.uuid4().hex[:8]}"
        unique_dir.mkdir(parents=True, exist_ok=True)
        old_dir = os.environ.get("LANCEDB_DIR")
        os.environ["LANCEDB_DIR"] = str(unique_dir)
        try:
            yield str(unique_dir)
        finally:
            if old_dir is not None:
                os.environ["LANCEDB_DIR"] = old_dir
            else:
                os.environ.pop("LANCEDB_DIR", None)
            # Cleanup temp directory
            import shutil

            if unique_dir.exists():
                shutil.rmtree(unique_dir)

    @pytest.fixture
    def test_collection(self) -> str:
        return f"test_collection_{uuid.uuid4().hex[:8]}"

    def test_parse_document_arrow_fallback_chain(
        self, temp_lancedb_dir, test_collection
    ) -> None:
        """Test parse_document uses iter_batches with Arrow RecordBatch."""
        from unittest.mock import MagicMock, patch

        import pandas as pd

        from xagent.core.tools.core.RAG_tools.parse.parse_document import (
            _get_document_from_db,
        )

        # Mock the vector store
        mock_vector_store = MagicMock()

        # Create mock document data
        doc_data = {
            "collection": test_collection,
            "doc_id": "doc1",
            "source_path": "/path/to/file",
            "file_type": "txt",
            "content_hash": "hash1",
            "uploaded_at": pd.Timestamp.now(),
            "title": None,
            "language": None,
            "user_id": 1,
        }

        # Create mock batch
        mock_batch = MagicMock()
        mock_batch.num_rows = 1
        mock_batch.to_pandas.return_value = pd.DataFrame([doc_data])

        # Mock iter_batches to yield the mock batch
        mock_vector_store.iter_batches.return_value = iter([mock_batch])

        with patch(
            "xagent.core.tools.core.RAG_tools.parse.parse_document.get_vector_index_store",
            return_value=mock_vector_store,
        ):
            result = _get_document_from_db(
                collection=test_collection,
                doc_id="doc1",
                user_id=1,
            )

            assert result is not None
            assert result["doc_id"] == "doc1"
            # Verify iter_batches was called (no count_rows_or_zero)
            mock_vector_store.iter_batches.assert_called_once()

    def test_parse_document_fallback_to_list(
        self, temp_lancedb_dir, test_collection
    ) -> None:
        """Test parse_document handles batch data correctly."""
        from unittest.mock import MagicMock, patch

        import pandas as pd

        from xagent.core.tools.core.RAG_tools.parse.parse_document import (
            _get_document_from_db,
        )

        # Mock the vector store
        mock_vector_store = MagicMock()

        # Create mock document data
        doc_data = {
            "collection": test_collection,
            "doc_id": "doc1",
            "source_path": "/path/to/file",
            "file_type": "txt",
            "content_hash": "hash1",
            "uploaded_at": pd.Timestamp.now(),
            "title": None,
            "language": None,
            "user_id": 1,
        }

        # Create mock batch
        mock_batch = MagicMock()
        mock_batch.num_rows = 1
        mock_batch.to_pandas.return_value = pd.DataFrame([doc_data])

        # Mock iter_batches to yield the mock batch
        mock_vector_store.iter_batches.return_value = iter([mock_batch])

        with patch(
            "xagent.core.tools.core.RAG_tools.parse.parse_document.get_vector_index_store",
            return_value=mock_vector_store,
        ):
            result = _get_document_from_db(
                collection=test_collection,
                doc_id="doc1",
                user_id=1,
            )

            assert result is not None
            assert result["doc_id"] == "doc1"
            # Verify iter_batches was called (no count_rows_or_zero)
            mock_vector_store.iter_batches.assert_called_once()

    def test_parse_document_fallback_to_pandas_with_nan(
        self, temp_lancedb_dir, test_collection
    ) -> None:
        """Test parse_document handles batch data correctly via iter_batches."""
        from unittest.mock import MagicMock, patch

        import pandas as pd

        from xagent.core.tools.core.RAG_tools.parse.parse_document import (
            _get_document_from_db,
        )

        # Mock the vector store
        mock_vector_store = MagicMock()

        # Create mock document data (without NaN - use None directly)
        doc_data = {
            "collection": test_collection,
            "doc_id": "doc1",
            "source_path": "/path/to/file",
            "file_type": "txt",
            "content_hash": "hash1",
            "uploaded_at": pd.Timestamp.now(),
            "title": None,
            "language": None,
            "user_id": 1,
        }

        # Create mock batch
        mock_batch = MagicMock()
        mock_batch.num_rows = 1
        mock_batch.to_pandas.return_value = pd.DataFrame([doc_data])

        # Mock iter_batches to yield the mock batch
        mock_vector_store.iter_batches.return_value = iter([mock_batch])

        with patch(
            "xagent.core.tools.core.RAG_tools.parse.parse_document.get_vector_index_store",
            return_value=mock_vector_store,
        ):
            result = _get_document_from_db(
                collection=test_collection,
                doc_id="doc1",
                user_id=1,
            )

            assert result is not None
            assert result["doc_id"] == "doc1"
            # Verify None values are preserved
            assert result.get("title") is None
            assert result.get("language") is None
