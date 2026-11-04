"""Characterization tests locking current document-row lifecycle behavior.

These tests pin the *observable* behavior of the legacy document helpers
(``register_document`` / ``get_document`` / ``list_documents``) BEFORE the
#508 refactor moves that logic into ``KBCollectionHandle``. They act as the
equivalence oracle for the handle implementation and, in particular, lock:

* the exact ``documents`` table column set (lossless-mapping requirement), and
* the raw-dict shape returned by ``get_document`` / ``list_documents``.

Storage isolation + reset is provided by the autouse ``isolate_rag_storage``
fixture in ``tests/conftest.py``.
"""

from pathlib import Path

import pandas as pd

from xagent.core.tools.core.RAG_tools.file.register_document import (
    get_document,
    list_documents,
    register_document,
)
from xagent.providers.vector_store.lancedb import get_connection_from_env

# The full, current ``documents`` table schema (see LanceDB/schema_manager.py).
DOCUMENT_COLUMNS = {
    "collection",
    "doc_id",
    "file_id",
    "source_path",
    "file_type",
    "content_hash",
    "uploaded_at",
    "title",
    "language",
    "user_id",
}


def _raw_documents_row(collection: str, doc_id: str) -> dict:
    conn = get_connection_from_env()
    table = conn.open_table("documents")
    return (
        table.search()
        .where(f"collection = '{collection}' AND doc_id = '{doc_id}'")
        .to_pandas()
        .iloc[0]
        .to_dict()
    )


class TestRegisteredRowSchema:
    """Lock the persisted documents-row columns and field persistence."""

    def test_persisted_row_has_exact_columns_and_values(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setenv("LANCEDB_DIR", str(tmp_path / "lancedb"))
        src = tmp_path / "report.txt"
        src.write_text("hello world")

        response = register_document(
            collection="coll",
            source_path=str(src),
            doc_id="doc-1",
            user_id=7,
            file_id="file-9",
        )

        assert response["doc_id"] == "doc-1"
        assert response["created"] is True
        assert len(response["content_hash"]) == 64

        row = _raw_documents_row("coll", "doc-1")
        # Lossless-mapping oracle: exactly these columns, nothing more/less.
        assert set(row.keys()) == DOCUMENT_COLUMNS

        assert row["collection"] == "coll"
        assert row["doc_id"] == "doc-1"
        assert row["file_id"] == "file-9"
        assert row["source_path"] == str(src)
        assert row["file_type"] == "txt"  # auto-detected from extension
        assert row["content_hash"] == response["content_hash"]
        assert row["user_id"] == 7
        assert not pd.isna(row["uploaded_at"])
        # title / language are written as None and stay null.
        assert row["title"] is None or pd.isna(row["title"])
        assert row["language"] is None or pd.isna(row["language"])


class TestGetDocument:
    """Lock get_document's behavior under its (anonymous) default scope.

    The public ``get_document(db_dir, collection, doc_id)`` exposes no user
    scope, so it always runs as unauthenticated non-admin. Per the user-access
    rules (``UserPermissions.get_user_filter``) that scope can see *no* rows,
    so the call returns ``None`` even for an existing document. This helper has
    no production callers; the behavior is locked so the handle reproduces it.
    """

    def test_existing_document_under_anonymous_scope_returns_none(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setenv("LANCEDB_DIR", str(tmp_path / "lancedb"))
        src = tmp_path / "a.md"
        src.write_text("# title")

        register_document(collection="coll", source_path=str(src), doc_id="doc-a")

        # Row exists (admin-visible), but anonymous default scope sees nothing.
        assert get_document(str(tmp_path / "lancedb"), "coll", "doc-a") is None

    def test_missing_document_returns_none(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("LANCEDB_DIR", str(tmp_path / "lancedb"))
        src = tmp_path / "a.txt"
        src.write_text("x")
        register_document(collection="coll", source_path=str(src), doc_id="exists")

        assert get_document(str(tmp_path / "lancedb"), "coll", "nope") is None


class TestListDocuments:
    """Lock list_documents (admin scope) shape, collection filter, and limit."""

    def _register(self, tmp_path: Path, name: str, collection: str, doc_id: str):
        src = tmp_path / name
        src.write_text(f"content of {name}")
        register_document(collection=collection, source_path=str(src), doc_id=doc_id)

    def test_shape_filter_and_limit(self, tmp_path: Path, monkeypatch) -> None:
        db_dir = tmp_path / "lancedb"
        monkeypatch.setenv("LANCEDB_DIR", str(db_dir))

        self._register(tmp_path, "a1.txt", "coll_a", "a1")
        self._register(tmp_path, "a2.txt", "coll_a", "a2")
        self._register(tmp_path, "b1.txt", "coll_b", "b1")

        rows_a = list_documents(str(db_dir), collection="coll_a", limit=100)
        assert len(rows_a) == 2
        for row in rows_a:
            assert set(row.keys()) == DOCUMENT_COLUMNS
            assert row["collection"] == "coll_a"
        assert {row["doc_id"] for row in rows_a} == {"a1", "a2"}

        limited = list_documents(str(db_dir), collection="coll_a", limit=1)
        assert len(limited) == 1

    def test_empty_collection_returns_empty_list(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        db_dir = tmp_path / "lancedb"
        monkeypatch.setenv("LANCEDB_DIR", str(db_dir))
        assert list_documents(str(db_dir), collection="none", limit=10) == []
