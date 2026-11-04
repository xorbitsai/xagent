"""Looking documents up by ``file_id`` through the vector-index contract (#2665)."""

from __future__ import annotations

from typing import Any

import pytest

from xagent.core.tools.core.RAG_tools.core.config import DEFAULT_VECTOR_STORE_SCAN_LIMIT
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
    ensure_documents_table,
)
from xagent.core.tools.core.RAG_tools.storage import lancedb_stores
from xagent.core.tools.core.RAG_tools.storage.contracts import DocumentRecord
from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
    LanceDBVectorIndexStore,
)
from xagent.providers.vector_store.lancedb import get_connection_from_env


def _doc(
    collection: Any, doc_id: Any, file_id: str, user_id: int | None
) -> dict[str, Any]:
    return {
        "collection": collection,
        "doc_id": doc_id,
        "file_id": file_id,
        "user_id": user_id,
    }


def _add(*rows: dict[str, Any]) -> None:
    conn = get_connection_from_env()
    ensure_documents_table(conn)
    conn.open_table("documents").add(list(rows))


def test_lookup_returns_candidates_rows_of_every_owner_across_batches(monkeypatch):
    monkeypatch.setattr(lancedb_stores, "_FILE_ID_LOOKUP_BATCH_SIZE", 2)
    _add(
        *(
            _doc("bulk", f"bulk-{i}", "f-1", 7)
            for i in range(DEFAULT_VECTOR_STORE_SCAN_LIMIT)
        )
    )
    _add(
        _doc("kb-a", "d-1", "f-1", 7),
        _doc("kb-b", "d-2", "f-2", 424242),
        _doc("kb-c", "d-3", "f-3", None),
        _doc("kb-x", "d-x", "f-x", 7),
    )

    records = LanceDBVectorIndexStore().list_document_records_by_file_ids(
        ["f-3", "f-1", "f-2", "f-4", "f-1", "f-1"]
    )

    assert len(records) == DEFAULT_VECTOR_STORE_SCAN_LIMIT + 3
    assert {r for r in records if r.collection != "bulk"} == {
        DocumentRecord(doc_id="d-1", file_id="f-1", user_id=7, collection="kb-a"),
        DocumentRecord(doc_id="d-2", file_id="f-2", user_id=424242, collection="kb-b"),
        DocumentRecord(doc_id="d-3", file_id="f-3", user_id=None, collection="kb-c"),
    }


def test_lookup_returns_values_as_stored():
    _add(_doc(None, None, "f-1", 7), _doc(" kb ", " d-1 ", "f-1", 7))

    records = LanceDBVectorIndexStore().list_document_records_by_file_ids(["f-1"])

    assert sorted((r.collection, r.doc_id) for r in records) == [
        ("", ""),
        (" kb ", " d-1 "),
    ]


def test_lookup_opens_the_store_only_for_candidates(monkeypatch):
    def _down() -> Any:
        raise RuntimeError("lancedb down")

    store = LanceDBVectorIndexStore()
    monkeypatch.setattr(store, "_get_connection", _down)

    assert store.list_document_records_by_file_ids([]) == []
    assert store.list_document_records_by_file_ids(["", ""]) == []
    with pytest.raises(RuntimeError, match="lancedb down"):
        store.list_document_records_by_file_ids(["f-1"])


def test_lookup_raises_when_the_query_fails(monkeypatch):
    _add(_doc("kb", "d-1", "f-1", 7))

    def _boom(_query: Any) -> Any:
        raise RuntimeError("query down")

    monkeypatch.setattr(lancedb_stores, "query_to_list", _boom)

    with pytest.raises(RuntimeError, match="query down"):
        LanceDBVectorIndexStore().list_document_records_by_file_ids(["f-1"])


def test_lookup_escapes_quotes():
    _add(_doc("kb", "d-1", "o'x", 7), _doc("kb", "d-2", "other", 7))
    store = LanceDBVectorIndexStore()

    assert [r.doc_id for r in store.list_document_records_by_file_ids(["o'x"])] == [
        "d-1"
    ]
    assert (
        store.list_document_records_by_file_ids(["o'x') OR (file_id IS NOT NULL"]) == []
    )
