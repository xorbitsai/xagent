"""The two vector-index reads behind the file-status paths (#2665)."""

from __future__ import annotations

from typing import Any

from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
    ensure_documents_table,
)
from xagent.core.tools.core.RAG_tools.storage import lancedb_stores
from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
    LanceDBVectorIndexStore,
)
from xagent.providers.vector_store.lancedb import get_connection_from_env


def _add_documents(*rows: dict[str, Any]) -> None:
    conn = get_connection_from_env()
    ensure_documents_table(conn)
    conn.open_table("documents").add(list(rows))


def _ref(collection: str, doc_id: str, user_id: int | None) -> dict[str, Any]:
    return {"collection": collection, "doc_id": doc_id, "user_id": user_id}


def test_document_rows_are_the_users_rows_as_stored():
    _add_documents(
        {
            "collection": "kb",
            "doc_id": "d-1",
            "file_id": "f-1",
            "source_path": " a.md ",
            "user_id": 7,
        },
        {"collection": "kb", "doc_id": "", "file_id": "f-2", "user_id": 7},
        {"collection": "kb", "doc_id": "d-3", "file_id": "f-3", "user_id": 8},
        {"collection": "kb", "doc_id": "d-4", "user_id": None},
    )
    store = LanceDBVectorIndexStore()

    own = store.list_document_rows(7, False)

    assert sorted(row["file_id"] for row in own) == ["f-1", "f-2"]
    row = next(row for row in own if row["file_id"] == "f-1")
    assert row["source_path"] == " a.md "
    assert set(row) == set(
        get_connection_from_env().open_table("documents").schema.names
    )
    assert sorted(row["doc_id"] for row in store.list_document_rows(1, True)) == [
        "",
        "d-1",
        "d-3",
        "d-4",
    ]
    assert store.list_document_rows(None, False) == []
    assert len(store.list_document_rows(1, True, max_results=2)) == 2


def test_document_rows_create_a_missing_documents_table():
    assert LanceDBVectorIndexStore().list_document_rows(7, False) == []


def test_indexed_refs_are_the_users_chunk_and_embedding_rows():
    conn = get_connection_from_env()
    conn.create_table(
        "chunks",
        data=[
            _ref("kb", "d-1", 7),
            _ref("kb", "d-3", 8),
            _ref("kb", "d'q", 7),
            _ref(" kb ", " d-9 ", 7),
        ],
    )
    conn.create_table(
        "embeddings_m", data=[_ref("kb", "d-2", 7), _ref("other", "d-1", None)]
    )
    refs = [
        ("kb", "d-1"),
        ("kb", "d-2"),
        ("kb", "d-3"),
        ("kb", "d-4"),
        ("other", "d-1"),
    ]
    store = LanceDBVectorIndexStore()

    assert store.list_indexed_doc_refs(refs, 7, False) == {("kb", "d-1"), ("kb", "d-2")}
    assert store.list_indexed_doc_refs(refs, 1, True) == {
        ("kb", "d-1"),
        ("kb", "d-2"),
        ("kb", "d-3"),
        ("other", "d-1"),
    }
    assert store.list_indexed_doc_refs(refs, None, False) == set()
    assert store.list_indexed_doc_refs([("kb", "d'q")], 7, False) == {("kb", "d'q")}
    assert store.list_indexed_doc_refs([(" kb ", " d-9 ")], 7, False) == {("kb", "d-9")}


def test_indexed_refs_skip_embeddings_once_chunks_cover_every_ref(monkeypatch):
    conn = get_connection_from_env()
    conn.create_table("chunks", data=[_ref("kb", "doc-legacy", 1)])
    monkeypatch.setattr(
        lancedb_stores,
        "list_embeddings_table_names",
        lambda _conn: ["embeddings_should_not_open"],
    )
    opened_tables: list[str] = []

    class RecordingConnection:
        def open_table(self, table_name: str):
            opened_tables.append(table_name)
            return conn.open_table(table_name)

    monkeypatch.setattr(
        LanceDBVectorIndexStore, "_get_connection", lambda self: RecordingConnection()
    )

    refs = LanceDBVectorIndexStore().list_indexed_doc_refs(
        [("kb", "doc-legacy")], 1, False
    )

    assert refs == {("kb", "doc-legacy")}
    assert opened_tables == ["chunks"]


def test_indexed_refs_skip_a_table_or_query_that_fails(monkeypatch):
    conn = get_connection_from_env()
    # No doc_id column, so the per-collection query fails.
    conn.create_table("chunks", data=[{"collection": "kb", "user_id": 7}])
    conn.create_table("embeddings_b", data=[_ref("kb", "d-1", 7), _ref("kb", "d-2", 7)])
    monkeypatch.setattr(
        lancedb_stores,
        "list_embeddings_table_names",
        lambda _conn: ["embeddings_missing", "embeddings_b"],
    )

    refs = LanceDBVectorIndexStore().list_indexed_doc_refs(
        [("kb", "d-1"), ("kb", "d-2")], 7, False
    )

    assert refs == {("kb", "d-1"), ("kb", "d-2")}


def test_indexed_refs_read_nothing_without_refs(monkeypatch):
    def _no_connection(self: Any) -> Any:
        raise AssertionError("connected without refs")

    monkeypatch.setattr(LanceDBVectorIndexStore, "_get_connection", _no_connection)

    assert LanceDBVectorIndexStore().list_indexed_doc_refs([], 7, False) == set()
