"""Whole-document row snapshot and restore on the collection handle (#2665).

Storage isolation/reset is provided by the autouse ``isolate_rag_storage``
fixture in ``tests/conftest.py``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import DocumentValidationError
from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
    LanceDBCollectionHandle,
    _document_row_key_columns,
)
from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
    _restore_document_table_rows as _restore_rag_snapshot_rows,
)
from xagent.core.tools.core.RAG_tools.kb.models import (
    KBAccessMode,
    KBBackendCapabilities,
    KBCollectionContext,
    KBDocumentRowsSnapshot,
    KBStorageBackend,
    KBUserScope,
)
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
    ensure_main_pointers_table,
)
from xagent.core.tools.core.RAG_tools.storage.factory import (
    get_ingestion_status_store,
    get_main_pointer_store,
    get_metadata_store,
    get_vector_index_store,
)

FIXED_TABLES = frozenset(
    {"documents", "parses", "chunks", "main_pointers", "ingestion_runs"}
)


def make_handle(collection: str = "coll") -> LanceDBCollectionHandle:
    context = KBCollectionContext(
        collection=collection,
        user_scope=KBUserScope(user_id=None, is_admin=True),
        access_mode=KBAccessMode.WRITE,
        allow_create=True,
        hide_missing=True,
        metadata_store=get_metadata_store(),
        vector_index_store=get_vector_index_store(),
        ingestion_status_store=get_ingestion_status_store(),
        main_pointer_store=get_main_pointer_store(),
        backend=KBStorageBackend.LANCEDB,
        capabilities=KBBackendCapabilities.lancedb(),
        collection_info=None,
    )
    return LanceDBCollectionHandle(context)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _seed_document(
    collection: str, doc_id: str, *, source: str = "old", user_id: int = 1
) -> None:
    get_vector_index_store().upsert_documents(
        [
            {
                "collection": collection,
                "doc_id": doc_id,
                "file_id": "file-1",
                "source_path": source,
                "file_type": "md",
                "content_hash": "a" * 64,
                "uploaded_at": _now(),
                "title": None,
                "language": None,
                "user_id": user_id,
            }
        ]
    )


def _seed_parse(
    collection: str, doc_id: str, parse_hash: str, *, user_id: int = 1
) -> None:
    get_vector_index_store().upsert_parses(
        [
            {
                "collection": collection,
                "doc_id": doc_id,
                "parse_hash": parse_hash,
                "parser": "test",
                "created_at": _now(),
                "params_json": "{}",
                "parsed_content": "[]",
                "user_id": user_id,
            }
        ]
    )


def _chunk_row(
    collection: str,
    doc_id: str,
    chunk_id: str,
    *,
    parse_hash: str = "p1",
    text: str = "old",
    user_id: int = 1,
) -> dict:
    return {
        "collection": collection,
        "doc_id": doc_id,
        "parse_hash": parse_hash,
        "chunk_id": chunk_id,
        "index": 0,
        "text": text,
        "page_number": None,
        "section": None,
        "anchor": None,
        "json_path": None,
        "chunk_hash": f"h-{chunk_id}",
        "config_hash": "cfg",
        "created_at": _now(),
        "metadata": "{}",
        "user_id": user_id,
    }


def _seed_chunk(collection: str, doc_id: str, chunk_id: str, **fields) -> None:
    get_vector_index_store().upsert_chunks(
        [_chunk_row(collection, doc_id, chunk_id, **fields)]
    )


def _seed_embedding(
    collection: str,
    doc_id: str,
    chunk_id: str,
    *,
    parse_hash: str,
    model: str,
    user_id: int = 1,
) -> None:
    get_vector_index_store().upsert_embeddings(
        model,
        [
            {
                "collection": collection,
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "parse_hash": parse_hash,
                "model": model,
                "vector": [0.1, 0.2],
                "vector_dimension": 2,
                "text": chunk_id,
                "chunk_hash": f"h-{chunk_id}",
                "created_at": _now(),
                "metadata": "{}",
                "user_id": user_id,
            }
        ],
    )


def _set_pointer(collection: str, doc_id: str, step_type: str) -> None:
    conn = get_vector_index_store().get_raw_connection()
    ensure_main_pointers_table(conn)
    # main_pointers is timestamp[ms]; lancedb 0.29 rejects a microsecond value on add.
    now = datetime.now().replace(microsecond=0)
    conn.open_table("main_pointers").add(
        [
            {
                "collection": collection,
                "doc_id": doc_id,
                "step_type": step_type,
                # The store writes "" for no tag; merge_insert never matches a NULL key.
                "model_tag": "",
                "semantic_id": f"s-{step_type}",
                "technical_id": f"t-{step_type}",
                "created_at": now,
                "updated_at": now,
                "operator": "test",
            }
        ]
    )


def _conn():
    return get_vector_index_store().get_raw_connection()


def _table_names() -> list[str]:
    return sorted(_conn().table_names())


def _rows(table_name: str, *columns: str) -> list[tuple]:
    rows = _conn().open_table(table_name).to_arrow().to_pylist()
    return sorted(tuple(row[column] for column in columns) for row in rows)


def _embedding_tables() -> list[str]:
    return [name for name in _table_names() if name.startswith("embeddings_")]


def test_restore_puts_back_changed_rows_and_drops_rows_added_since() -> None:
    _seed_document("coll", "d1")
    _seed_parse("coll", "d1", "p1")
    _seed_chunk("coll", "d1", "k1")
    _seed_embedding("coll", "d1", "k1", parse_hash="p1", model="m1")
    _set_pointer("coll", "d1", "parse")
    handle = make_handle()
    snapshot = handle.capture_document_rows(["d1"], user_id=1, is_admin=True)

    _seed_document("coll", "d1", source="new")
    _seed_parse("coll", "d1", "p2")
    _seed_chunk("coll", "d1", "k1", text="new")
    _seed_chunk("coll", "d1", "k2", parse_hash="p2")
    _seed_embedding("coll", "d1", "k2", parse_hash="p2", model="m1")
    _seed_embedding("coll", "d1", "k1", parse_hash="p1", model="m2")
    _set_pointer("coll", "d1", "embed")
    get_ingestion_status_store().write_ingestion_status(
        "coll", "d1", status="processing"
    )
    assert len(_embedding_tables()) == 2

    assert handle.restore_document_rows(snapshot, user_id=1, is_admin=True) == []

    assert _rows("documents", "doc_id", "source_path") == [("d1", "old")]
    assert _rows("parses", "doc_id", "parse_hash") == [("d1", "p1")]
    assert _rows("chunks", "chunk_id", "parse_hash", "text") == [("k1", "p1", "old")]
    assert _rows("main_pointers", "doc_id", "step_type") == [("d1", "parse")]
    assert _rows("ingestion_runs", "doc_id") == []
    for name in _embedding_tables():
        assert _rows(name, "chunk_id", "parse_hash", "model") == (
            [("k1", "p1", "m1")] if name in snapshot.rows_by_table else []
        )


def test_capture_reads_the_fixed_tables_and_every_embedding_table_only() -> None:
    _seed_chunk("coll", "d1", "k1")
    _seed_embedding("coll", "d1", "k1", parse_hash="p1", model="m1")
    _conn().create_table("unrelated", data=[{"collection": "coll", "doc_id": "d1"}])
    handle = make_handle()

    snapshot = handle.capture_document_rows(["d1"], user_id=1, is_admin=True)
    _conn().open_table("unrelated").add([{"collection": "coll", "doc_id": "d1"}])
    handle.restore_document_rows(snapshot, user_id=1, is_admin=True)

    assert set(snapshot.rows_by_table) == FIXED_TABLES | set(_embedding_tables())
    assert len(_embedding_tables()) == 1
    assert len(_rows("unrelated", "doc_id")) == 2


def test_non_admin_rows_are_limited_to_the_caller_where_tables_have_user_id() -> None:
    _seed_chunk("coll", "d1", "k1", user_id=1)
    _seed_chunk("coll", "d1", "k-other", user_id=2)
    _seed_document("coll", "d1", user_id=2)
    _seed_parse("coll", "d1", "p-other", user_id=2)
    _seed_embedding("coll", "d1", "k-other", parse_hash="p1", model="m1", user_id=2)
    get_ingestion_status_store().write_ingestion_status(
        "coll", "d1", status="done", user_id=2
    )
    _set_pointer("coll", "d1", "parse")
    handle = make_handle()

    snapshot = handle.capture_document_rows(["d1"], user_id=1, is_admin=False)
    assert [row["chunk_id"] for row in snapshot.rows_by_table["chunks"]] == ["k1"]
    assert len(snapshot.rows_by_table["main_pointers"]) == 1
    assert [
        (name, row["user_id"])
        for name, rows in snapshot.rows_by_table.items()
        for row in rows
        if row.get("user_id") == 2
    ] == []

    _seed_chunk("coll", "d1", "k-new", user_id=1)
    _conn().open_table("chunks").add([_chunk_row("coll", "d1", "k-new", user_id=2)])
    _seed_chunk("coll", "d1", "k-other", text="changed", user_id=2)
    _seed_chunk("coll", "d1", "k-other-2", user_id=2)
    _set_pointer("coll", "d1", "embed")
    handle.restore_document_rows(snapshot, user_id=1, is_admin=False)

    assert _rows("chunks", "chunk_id", "text", "user_id") == [
        ("k-new", "old", 2),
        ("k-other", "changed", 2),
        ("k-other-2", "old", 2),
        ("k1", "old", 1),
    ]
    assert _rows("main_pointers", "step_type") == [("parse",)]
    owned_tables = ["documents", "parses", "ingestion_runs", *_embedding_tables()]
    assert len(owned_tables) == 4
    assert {name: _rows(name, "user_id") for name in owned_tables} == {
        name: [(2,)] for name in owned_tables
    }


def test_admin_capture_reads_every_owner() -> None:
    _seed_chunk("coll", "d1", "k1", user_id=1)
    _seed_chunk("coll", "d1", "k-other", user_id=2)

    snapshot = make_handle().capture_document_rows(["d1"], user_id=1, is_admin=True)

    assert sorted(row["chunk_id"] for row in snapshot.rows_by_table["chunks"]) == [
        "k-other",
        "k1",
    ]


def test_rows_of_other_collections_and_documents_are_left_alone() -> None:
    _seed_chunk("coll", "d1", "k1")
    _seed_chunk("coll", "d2", "k1")
    _seed_chunk("other", "d1", "k1")
    handle = make_handle()

    snapshot = handle.capture_document_rows(["d1"], user_id=1, is_admin=True)
    _seed_chunk("coll", "d2", "k2")
    _seed_chunk("other", "d1", "k2")
    handle.restore_document_rows(snapshot, user_id=1, is_admin=True)

    assert [
        (row["collection"], row["doc_id"]) for row in snapshot.rows_by_table["chunks"]
    ] == [("coll", "d1")]
    assert _rows("chunks", "collection", "doc_id", "chunk_id") == [
        ("coll", "d1", "k1"),
        ("coll", "d2", "k1"),
        ("coll", "d2", "k2"),
        ("other", "d1", "k1"),
        ("other", "d1", "k2"),
    ]


def test_no_documents_captures_nothing_and_restores_nothing() -> None:
    _seed_chunk("coll", "d1", "k1")
    handle = make_handle()

    snapshot = handle.capture_document_rows([], user_id=1, is_admin=True)
    _seed_chunk("coll", "d1", "k2")
    handle.restore_document_rows(snapshot, user_id=1, is_admin=True)

    assert snapshot.rows_by_table
    assert all(rows == [] for rows in snapshot.rows_by_table.values())
    assert _rows("chunks", "chunk_id") == [("k1",), ("k2",)]


def test_capture_takes_any_iterable_of_ids_but_not_a_str() -> None:
    _seed_chunk("coll", "d1", "k1")
    _seed_chunk("coll", "d2", "k1")
    handle = make_handle()

    snapshot = handle.capture_document_rows(
        (doc_id for doc_id in ["d1", "d2"]), user_id=1, is_admin=True
    )

    assert snapshot.doc_ids == ("d1", "d2")
    assert sorted(row["doc_id"] for row in snapshot.rows_by_table["chunks"]) == [
        "d1",
        "d2",
    ]
    with pytest.raises(DocumentValidationError):
        handle.capture_document_rows("d1", user_id=1, is_admin=True)


def test_restore_invalidates_the_table_cache_once_per_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_chunk("coll", "d1", "k1")
    handle = make_handle()
    snapshot = handle.capture_document_rows(["d1"], user_id=1, is_admin=True)
    store = get_vector_index_store()
    spy = MagicMock(wraps=store.invalidate_table_cache)
    monkeypatch.setattr(store, "invalidate_table_cache", spy)

    handle.restore_document_rows(snapshot, user_id=1, is_admin=True)
    spy.assert_called_once_with()
    handle.restore_document_rows(snapshot, user_id=1, is_admin=True)
    assert spy.call_count == 2


@pytest.mark.parametrize(
    "row",
    [_chunk_row("other", "d1", "k9"), _chunk_row("coll", "d2", "k9")],
    ids=["other-collection", "other-document"],
)
def test_restore_refuses_rows_outside_the_snapshot_documents(row) -> None:
    _seed_chunk("coll", "d1", "k1")
    snapshot = KBDocumentRowsSnapshot(
        collection="coll", doc_ids=("d1",), rows_by_table={"chunks": [row]}
    )

    with pytest.raises(DocumentValidationError):
        make_handle("coll").restore_document_rows(snapshot, user_id=1, is_admin=True)

    assert _rows("chunks", "collection", "doc_id", "chunk_id") == [("coll", "d1", "k1")]


def test_restore_refuses_another_collections_snapshot() -> None:
    snapshot = KBDocumentRowsSnapshot(
        collection="other", doc_ids=("d1",), rows_by_table={}
    )

    with pytest.raises(DocumentValidationError):
        make_handle("coll").restore_document_rows(snapshot, user_id=1, is_admin=True)


@pytest.mark.parametrize(
    ("table_name", "key_columns"),
    [
        ("documents", ("collection", "doc_id")),
        ("parses", ("collection", "doc_id", "parse_hash")),
        ("chunks", ("collection", "doc_id", "parse_hash", "chunk_id")),
        ("main_pointers", ("collection", "doc_id", "step_type", "model_tag")),
        ("ingestion_runs", ("collection", "doc_id")),
        (
            "embeddings_text_embedding_v4",
            ("collection", "doc_id", "chunk_id", "parse_hash", "model"),
        ),
        ("custom_table", None),
    ],
)
def test_row_key_columns_per_table(table_name, key_columns) -> None:
    assert _document_row_key_columns(table_name) == key_columns


class TestRestoreDocumentTableRows:
    """Moved verbatim from the web ingestion tests with the helper (#2665)."""

    def test_restore_rag_snapshot_rows_batches_unknown_table_delete(self) -> None:
        table = MagicMock()
        table.schema.names = []

        _restore_rag_snapshot_rows(
            table,
            table_name="custom_table",
            snapshot_rows=[{"collection": "c1", "doc_id": "doc-old"}],
            current_rows=[
                {"collection": "c1", "doc_id": "doc-1"},
                {"collection": "c1", "doc_id": "doc-2"},
            ],
            user_id=1,
            is_admin=False,
        )

        table.delete.assert_called_once()
        delete_filter = table.delete.call_args.args[0]
        assert "(collection = 'c1' and doc_id = 'doc-1')" in delete_filter
        assert "(collection = 'c1' and doc_id = 'doc-2')" in delete_filter
        assert " or " in delete_filter
        table.add.assert_called_once_with([{"collection": "c1", "doc_id": "doc-old"}])

    def test_restore_rag_snapshot_rows_batches_stale_row_delete(self) -> None:
        table = MagicMock()
        table.schema.names = []

        _restore_rag_snapshot_rows(
            table,
            table_name="chunks",
            snapshot_rows=[
                {
                    "collection": "c1",
                    "doc_id": "doc-1",
                    "parse_hash": "hash",
                    "chunk_id": "chunk-old",
                }
            ],
            current_rows=[
                {
                    "collection": "c1",
                    "doc_id": "doc-1",
                    "parse_hash": "hash",
                    "chunk_id": "chunk-old",
                },
                {
                    "collection": "c1",
                    "doc_id": "doc-1",
                    "parse_hash": "hash",
                    "chunk_id": "chunk-stale",
                },
                {
                    "collection": "c1",
                    "doc_id": "doc-1",
                    "parse_hash": "hash",
                    "chunk_id": "chunk-stale-2",
                },
            ],
            user_id=1,
            is_admin=False,
        )

        table.merge_insert.assert_called_once_with(
            ["collection", "doc_id", "parse_hash", "chunk_id"]
        )
        table.delete.assert_called_once()
        delete_filter = table.delete.call_args.args[0]
        assert "chunk_id = 'chunk-old'" not in delete_filter
        assert "(collection = 'c1'" in delete_filter
        assert "chunk_id = 'chunk-stale'" in delete_filter
        assert "chunk_id = 'chunk-stale-2'" in delete_filter
        assert " or " in delete_filter

    def test_restore_rag_snapshot_rows_keys_embeddings_by_parse_and_model(
        self,
    ) -> None:
        table = MagicMock()
        table.schema.names = []

        _restore_rag_snapshot_rows(
            table,
            table_name="embeddings_text_embedding_v4",
            snapshot_rows=[
                {
                    "collection": "c1",
                    "doc_id": "doc-1",
                    "chunk_id": "chunk-1",
                    "parse_hash": "parse-old",
                    "model": "model-a",
                }
            ],
            current_rows=[
                {
                    "collection": "c1",
                    "doc_id": "doc-1",
                    "chunk_id": "chunk-1",
                    "parse_hash": "parse-old",
                    "model": "model-a",
                },
                {
                    "collection": "c1",
                    "doc_id": "doc-1",
                    "chunk_id": "chunk-1",
                    "parse_hash": "parse-new",
                    "model": "model-a",
                },
                {
                    "collection": "c1",
                    "doc_id": "doc-1",
                    "chunk_id": "chunk-1",
                    "parse_hash": "parse-old",
                    "model": "model-b",
                },
            ],
            user_id=1,
            is_admin=False,
        )

        table.merge_insert.assert_called_once_with(
            ["collection", "doc_id", "chunk_id", "parse_hash", "model"]
        )
        table.delete.assert_called_once()
        delete_filter = table.delete.call_args.args[0]
        assert "parse_hash = 'parse-new'" in delete_filter
        assert "model = 'model-b'" in delete_filter
        assert "parse_hash = 'parse-old' and model = 'model-a'" not in delete_filter
        assert " or " in delete_filter
