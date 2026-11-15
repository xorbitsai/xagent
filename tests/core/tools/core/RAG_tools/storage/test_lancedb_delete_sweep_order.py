"""Deletes remove vectors, then chunk rows, then sweep the embedding tables in scope again."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pytest

from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import (
    written_embeddings_table_name,
)
from xagent.core.tools.core.RAG_tools.storage.factory import get_metadata_store
from xagent.core.tools.core.RAG_tools.utils.lancedb_query_utils import list_table_names

OTHER_MODEL = "new-model"
SOURCE_MODEL = "BAAI/old-model"


def _connection() -> Any:
    return get_metadata_store().get_raw_connection()


def _save_collection() -> None:
    from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo
    from xagent.core.tools.core.RAG_tools.management import collection_manager as cm

    asyncio.run(
        cm.collection_manager.save_collection(
            CollectionInfo(
                name="kb",
                embedding_model_id=SOURCE_MODEL,
                embedding_dimension=2,
                embeddings=4,
            )
        )
    )


_SCHEMA = pa.schema(
    [
        ("collection", pa.string()),
        ("doc_id", pa.string()),
        ("chunk_id", pa.string()),
        ("parse_hash", pa.string()),
        ("user_id", pa.int64()),
        ("text", pa.string()),
        ("chunk_hash", pa.string()),
        ("metadata", pa.string()),
        ("model", pa.string()),
        ("created_at", pa.timestamp("us")),
        ("vector", pa.list_(pa.float32(), 2)),
    ]
)


def _row(chunk_id: str, *, model: str = SOURCE_MODEL) -> Dict[str, Any]:
    return {
        "collection": "kb",
        "doc_id": "d1",
        "chunk_id": chunk_id,
        "parse_hash": "p",
        "user_id": 7,
        "text": f"text {chunk_id}",
        "chunk_hash": f"h-{chunk_id}",
        "metadata": None,
        "model": model,
        "created_at": datetime(2026, 1, 1),
        "vector": [0.5, 0.5],
    }


def _table(name: str, rows: List[Dict[str, Any]]) -> None:
    _connection().create_table(name, data=pa.Table.from_pylist(rows, schema=_SCHEMA))


def _rows_in(name: str) -> List[Dict[str, Any]]:
    conn = _connection()
    if name not in set(list_table_names(conn)):
        return []
    return conn.open_table(name).search().where("true").to_arrow().to_pylist()


def _source(rows: List[Dict[str, Any]]) -> None:
    _table(written_embeddings_table_name(SOURCE_MODEL), rows)


def test_the_documents_cascade_sweeps_a_table_a_writer_created_meanwhile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.LanceDB import schema_manager
    from xagent.core.tools.core.RAG_tools.storage import lancedb_stores

    for name in (
        "ensure_documents_table",
        "ensure_parses_table",
        "ensure_chunks_table",
        "ensure_main_pointers_table",
        "ensure_ingestion_runs_table",
    ):
        monkeypatch.setattr(schema_manager, name, lambda *args, **kwargs: None)

    conn = _connection()
    core = pa.schema(
        [
            ("collection", pa.string()),
            ("doc_id", pa.string()),
            ("chunk_id", pa.string()),
            ("user_id", pa.int64()),
        ]
    )
    documents = pa.Table.from_pylist(
        [{"collection": "kb", "doc_id": "d1", "chunk_id": "", "user_id": 7}],
        schema=core,
    )
    conn.create_table("documents", data=documents)
    conn.create_table("chunks", data=documents)

    late = written_embeddings_table_name(OTHER_MODEL)
    real_delete = lancedb_stores._vis_delete_rows_by_filters
    created: List[str] = []

    def _delete_then_writer_commits(table: Any, filters: List[str]) -> int:
        count = real_delete(table, filters)
        if count and not created:
            # A writer that read these chunk rows before they went commits
            # into another model's table while the cascade runs.
            conn.create_table(
                late,
                data=pa.Table.from_pylist(
                    [_row("c1", model=OTHER_MODEL)], schema=_SCHEMA
                ),
            )
            created.append(late)
        return count

    monkeypatch.setattr(
        lancedb_stores, "_vis_delete_rows_by_filters", _delete_then_writer_commits
    )

    lancedb_stores._vis_cascade_delete_documents(
        conn,
        collection="kb",
        doc_ids=["d1"],
        user_id=None,
        is_admin=True,
        preview_only=False,
        confirm=True,
    )

    assert created == [late]
    assert _rows_in(late) == []


def _handle(collection: str = "kb") -> Any:
    from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
        LanceDBCollectionHandle,
    )
    from xagent.core.tools.core.RAG_tools.kb.models import (
        KBAccessMode,
        KBBackendCapabilities,
        KBCollectionContext,
        KBStorageBackend,
        KBUserScope,
    )
    from xagent.core.tools.core.RAG_tools.storage.factory import (
        get_ingestion_status_store,
        get_main_pointer_store,
        get_metadata_store,
        get_vector_index_store,
    )

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


def _seed_retirable_core() -> None:
    """A document re-ingested as C2: chunk ``c1`` is stale, ``c2`` is current."""
    from xagent.core.tools.core.RAG_tools.storage.factory import (
        get_vector_index_store,
    )
    from xagent.core.tools.core.RAG_tools.utils.metadata_utils import (
        serialize_metadata,
    )

    store = get_vector_index_store()
    store.upsert_documents(
        [
            {
                "collection": "kb",
                "doc_id": "d1",
                "file_id": None,
                "source_path": "/x.txt",
                "file_type": "txt",
                "content_hash": "C2",
                "uploaded_at": datetime.now(timezone.utc),
                "title": None,
                "language": None,
                "user_id": None,
            }
        ]
    )
    store.upsert_parses(
        [
            {
                "collection": "kb",
                "doc_id": "d1",
                "parse_hash": "p",
                "parser": "local:default@v1.0.0",
                "created_at": datetime.now(timezone.utc),
                "params_json": json.dumps({"content_hash": "C2"}),
                "parsed_content": "[]",
                "user_id": None,
            }
        ]
    )
    store.upsert_chunks(
        [
            {
                "collection": "kb",
                "doc_id": "d1",
                "parse_hash": "p",
                "chunk_id": chunk_id,
                "index": 0,
                "text": chunk_id,
                "page_number": None,
                "section": None,
                "anchor": None,
                "json_path": None,
                "chunk_hash": "x",
                "config_hash": config_hash,
                "created_at": datetime.now(timezone.utc),
                "metadata": serialize_metadata({"content_hash": made_from}),
                "user_id": None,
            }
            for chunk_id, config_hash, made_from in (
                ("c1", "cfg-old", "C1"),
                ("c2", "cfg-new", "C2"),
            )
        ]
    )


def _store() -> Any:
    from xagent.core.tools.core.RAG_tools.storage.factory import (
        get_vector_index_store,
    )

    return get_vector_index_store()


def _emb(doc_id: str, chunk_id: str, model: str = "m1") -> Dict[str, Any]:
    return {
        "collection": "kb",
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "parse_hash": "p",
        "model": model,
        "vector": [0.1, 0.2],
        "vector_dimension": 2,
        "text": f"secret {chunk_id}",
        "chunk_hash": "h",
        "created_at": datetime.now(timezone.utc),
        "metadata": "{}",
        "user_id": None,
    }


def _emb_rows(doc_id: str, table: str = "embeddings_m1") -> List[str]:
    conn = _connection()
    return sorted(
        row["chunk_id"]
        for row in conn.open_table(table)
        .search()
        .where(f"doc_id = '{doc_id}'")
        .to_arrow()
        .to_pylist()
    )


def _doc_rows(doc_id: str) -> int:
    conn = _connection()
    return conn.open_table("documents").count_rows(f"doc_id = '{doc_id}'")


def _chunk_rows(doc_id: str) -> int:
    conn = _connection()
    return conn.open_table("chunks").count_rows(f"doc_id = '{doc_id}'")


def test_a_document_delete_that_fails_on_embeddings_can_be_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.management import collections as mc
    from xagent.core.tools.core.RAG_tools.storage import lancedb_stores

    _save_collection()
    _seed_retirable_core()
    store = _store()
    store.upsert_embeddings("m1", [_emb("d1", "c1"), _emb("d1", "c2")])
    real = lancedb_stores._vis_delete_rows_by_filters
    failed = {"once": False}

    def _fail_embeddings(table: Any, filters: List[str]) -> int:
        if getattr(table, "name", "").startswith("embeddings_") and not failed["once"]:
            failed["once"] = True
            raise OSError("injected embedding delete failure")
        return real(table, filters)

    monkeypatch.setattr(lancedb_stores, "_vis_delete_rows_by_filters", _fail_embeddings)

    with pytest.raises(Exception):
        store.delete_documents_data("kb", ["d1"], None, True)

    # Vectors go first: the failed attempt left every core row, document
    # included, so nothing a retry reads is gone.
    assert _doc_rows("d1") == 1
    assert _chunk_rows("d1") == 2
    assert _emb_rows("d1") == ["c1", "c2"]
    monkeypatch.setattr(lancedb_stores, "_vis_delete_rows_by_filters", real)

    result = mc.delete_document("kb", "d1", 1, is_admin=True)

    assert result.status == "success", result.message
    assert _doc_rows("d1") == 0
    assert _emb_rows("d1") == []


def test_a_retirement_that_fails_on_vectors_can_be_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
        LanceDBCollectionHandle,
    )

    _save_collection()
    _seed_retirable_core()
    _store().upsert_embeddings("m1", [_emb("d1", "c1"), _emb("d1", "c2")])
    real = LanceDBCollectionHandle.delete_embedding_records
    failed = {"once": False}

    def _fail_once(handle: Any, doc_id: str, **kwargs: Any) -> int:
        if not failed["once"]:
            failed["once"] = True
            raise OSError("injected")
        return real(handle, doc_id, **kwargs)

    monkeypatch.setattr(LanceDBCollectionHandle, "delete_embedding_records", _fail_once)

    with pytest.raises(OSError):
        _handle().retire_superseded_chunks("d1", "p")

    # The chunk rows are still there, so the retry can find the stale chunk.
    assert _handle().stale_chunks("d1", "p") == {("p", "cfg-old"): ["c1"]}
    assert _emb_rows("d1") == ["c1", "c2"]

    assert _handle().retire_superseded_chunks("d1", "p") == ["c1"]
    assert "c1" not in _emb_rows("d1")


def test_a_retirement_sweeps_a_table_a_writer_created_meanwhile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
        LanceDBCollectionHandle,
    )

    _save_collection()
    _seed_retirable_core()
    _source([_row("c1"), _row("c2")])
    late = written_embeddings_table_name(OTHER_MODEL)
    real_core = LanceDBCollectionHandle.delete_chunk_records
    created: List[str] = []

    def _writer_commits_then_core_goes(handle: Any, doc_id: str, **kwargs: Any) -> int:
        if not created:
            _table(late, [_row("c1", model=OTHER_MODEL)])
            created.append(late)
        return real_core(handle, doc_id, **kwargs)

    monkeypatch.setattr(
        LanceDBCollectionHandle, "delete_chunk_records", _writer_commits_then_core_goes
    )

    assert _handle().retire_superseded_chunks("d1", "p") == ["c1"]

    assert created == [late]
    assert _rows_in(late) == []
    assert {
        row["chunk_id"] for row in _rows_in(written_embeddings_table_name(SOURCE_MODEL))
    } == {"c2"}


def test_a_delete_reports_no_success_when_the_final_listing_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.management import collections as mc
    from xagent.core.tools.core.RAG_tools.storage import lancedb_cleanup_filters

    _save_collection()
    _seed_retirable_core()
    store = _store()
    store.upsert_embeddings("m1", [_emb("d1", "c1"), _emb("d1", "c2")])
    real = lancedb_cleanup_filters.select_embedding_tables
    calls = {"n": 0}

    def _flaky(conn: Any, model_tag: Optional[str] = None) -> List[str]:
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("listing failed")
        return real(conn, model_tag=model_tag)

    monkeypatch.setattr(lancedb_cleanup_filters, "select_embedding_tables", _flaky)

    with pytest.raises(Exception):
        store.delete_documents_data("kb", ["d1"], None, True)

    assert calls["n"] == 2
    monkeypatch.setattr(lancedb_cleanup_filters, "select_embedding_tables", real)

    # Not reported as a success, and still deletable through the document's row.
    assert _doc_rows("d1") == 1
    result = mc.delete_document("kb", "d1", 1, is_admin=True)
    assert result.status == "success", result.message
    assert _doc_rows("d1") == 0
    assert _emb_rows("d1") == []


def test_the_generic_cascade_sweeps_a_table_a_writer_created_meanwhile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.storage import lancedb_stores

    _save_collection()
    _seed_retirable_core()
    store = _store()
    store.upsert_embeddings("m1", [_emb("d1", "c1")])
    late = written_embeddings_table_name(OTHER_MODEL)
    real = lancedb_stores._vis_delete_rows_by_filters
    created: List[str] = []

    def _writer_commits_during_the_core_delete(table: Any, filters: List[str]) -> int:
        if not created and getattr(table, "name", "") == "chunks":
            # A writer that read the chunk rows before they went creates its
            # table while this delete runs.
            _connection().create_table(
                late,
                data=pa.Table.from_pylist(
                    [_row("c1", model=OTHER_MODEL)], schema=_SCHEMA
                ),
            )
            created.append(late)
        return real(table, filters)

    monkeypatch.setattr(
        lancedb_stores,
        "_vis_delete_rows_by_filters",
        _writer_commits_during_the_core_delete,
    )

    result = store.cascade_delete(
        target="document",
        collection="kb",
        doc_id="d1",
        is_admin=True,
        preview_only=False,
        confirm=True,
    )

    assert created == [late]
    assert _rows_in(late) == []
    assert result["documents"] == 1


@pytest.mark.parametrize("scope_name", ["chunk", "parse"])
def test_a_version_cleanup_sweeps_a_table_a_writer_created_meanwhile(
    monkeypatch: pytest.MonkeyPatch, scope_name: str
) -> None:
    from xagent.core.tools.core.RAG_tools.storage import lancedb_stores

    _save_collection()
    _seed_retirable_core()
    _source([_row("c1"), _row("c2")])
    late = written_embeddings_table_name(OTHER_MODEL)
    real = lancedb_stores._vis_delete_rows_by_filters
    created: List[str] = []

    def _writer_commits_during_the_core_delete(table: Any, filters: List[str]) -> int:
        if not created and getattr(table, "name", "") == "chunks":
            _connection().create_table(
                late,
                data=pa.Table.from_pylist(
                    [_row("c1", model=OTHER_MODEL)], schema=_SCHEMA
                ),
            )
            created.append(late)
        return real(table, filters)

    monkeypatch.setattr(
        lancedb_stores,
        "_vis_delete_rows_by_filters",
        _writer_commits_during_the_core_delete,
    )

    _store().cleanup_cascade_by_scope(
        "kb",
        "d1",
        scope_name,
        new_parse_hash="p-new",
        old_parse_hash="p",
        preview_only=False,
        confirm=True,
    )

    assert created == [late]
    assert _chunk_rows("d1") == 0
    assert _rows_in(late) == []


def test_a_cascade_whose_count_fails_deletes_nothing_and_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.utils import lancedb_query_utils

    _save_collection()
    _seed_retirable_core()
    _store().upsert_embeddings("m1", [_emb("d1", "c1"), _emb("d1", "c2")])
    real = lancedb_query_utils._safe_count_rows

    class _Unreadable:
        def count_rows(self, *args: Any) -> int:
            raise OSError("count failed")

    def _embeddings_count_fails(
        table: Any, filter_expr: Optional[str] = None, **kwargs: Any
    ) -> int:
        if str(getattr(table, "name", "")).startswith("embeddings_"):
            table = _Unreadable()
        return real(table, filter_expr, **kwargs)

    monkeypatch.setattr(
        lancedb_query_utils, "_safe_count_rows", _embeddings_count_fails
    )

    with pytest.raises(OSError):
        _store().cascade_delete(
            target="document",
            collection="kb",
            doc_id="d1",
            is_admin=True,
            preview_only=False,
            confirm=True,
        )

    assert _emb_rows("d1") == ["c1", "c2"]
    assert _chunk_rows("d1") == 2
    assert _doc_rows("d1") == 1


def test_a_failed_core_listing_fails_the_document_cascade_before_any_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.storage import lancedb_stores
    from xagent.core.tools.core.RAG_tools.utils import lancedb_query_utils

    _save_collection()
    _seed_retirable_core()
    store = _store()
    store.upsert_embeddings("m1", [_emb("d1", "c1"), _emb("d1", "c2")])
    real_listing = lancedb_query_utils.list_table_names
    real_cascade = lancedb_stores._vis_cascade_delete_documents
    armed = {"fail": False}

    def _listing(conn: Any) -> List[str]:
        if armed["fail"]:
            armed["fail"] = False
            raise OSError("listing failed")
        return real_listing(conn)

    def _cascade(*args: Any, **kwargs: Any) -> Dict[str, int]:
        armed["fail"] = True
        return real_cascade(*args, **kwargs)

    monkeypatch.setattr(lancedb_query_utils, "list_table_names", _listing)
    monkeypatch.setattr(lancedb_stores, "_vis_cascade_delete_documents", _cascade)

    with pytest.raises(Exception):
        store.delete_documents_data("kb", ["d1"], None, True)

    assert armed["fail"] is False
    assert _emb_rows("d1") == ["c1", "c2"]
    assert _chunk_rows("d1") == 2
    assert _doc_rows("d1") == 1


def _fail_listing_call(monkeypatch: pytest.MonkeyPatch, failing: int) -> Dict[str, int]:
    from xagent.core.tools.core.RAG_tools.utils import lancedb_query_utils

    real = lancedb_query_utils.list_table_names
    calls = {"n": 0}

    def _listing(conn: Any) -> List[str]:
        calls["n"] += 1
        if calls["n"] == failing:
            raise OSError("listing failed")
        return real(conn)

    monkeypatch.setattr(lancedb_query_utils, "list_table_names", _listing)
    return calls


@pytest.mark.parametrize(
    ("target", "failing"),
    [("collection", 1), ("document", 2)],
    ids=["collection-builder-listing", "document-executor-listing"],
)
def test_a_failed_listing_fails_the_cascade_before_any_delete(
    monkeypatch: pytest.MonkeyPatch, target: str, failing: int
) -> None:
    _save_collection()
    _seed_retirable_core()
    store = _store()
    store.upsert_embeddings("m1", [_emb("d1", "c1"), _emb("d1", "c2")])
    calls = _fail_listing_call(monkeypatch, failing)

    with pytest.raises(OSError):
        store.cascade_delete(
            target=target,
            collection="kb",
            doc_id="d1",
            is_admin=True,
            preview_only=False,
            confirm=True,
        )

    assert calls["n"] == failing
    assert _emb_rows("d1") == ["c1", "c2"]
    assert _chunk_rows("d1") == 2
    assert _doc_rows("d1") == 1


@pytest.mark.parametrize("delete", ["document-cascade", "parse-cleanup"])
def test_a_model_scoped_delete_keeps_the_other_models_vectors(delete: str) -> None:
    _save_collection()
    _seed_retirable_core()
    store = _store()
    store.upsert_embeddings("m1", [_emb("d1", "c1")])
    store.upsert_embeddings("m2", [_emb("d1", "c1", model="m2")])

    if delete == "document-cascade":
        store.cascade_delete(
            target="document",
            collection="kb",
            doc_id="d1",
            is_admin=True,
            model_tag="m1",
            preview_only=False,
            confirm=True,
        )
    else:
        store.cleanup_cascade_by_scope(
            "kb",
            "d1",
            "parse",
            new_parse_hash="p-new",
            old_parse_hash="p",
            model_tag="m1",
            preview_only=False,
            confirm=True,
        )

    assert _chunk_rows("d1") == 0
    assert _emb_rows("d1") == []
    assert _emb_rows("d1", "embeddings_m2") == ["c1"]
