"""A document without a status record is inferred from the table search reads."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence, Tuple

import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import DocumentProcessingStatus as S
from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import (
    embeddings_table_candidates,
    embeddings_table_name,
)
from xagent.core.tools.core.RAG_tools.management import collection_manager as cm
from xagent.core.tools.core.RAG_tools.management import (
    collections as collections_module,
)
from xagent.core.tools.core.RAG_tools.management import (
    get_document_stats,
    list_documents,
)
from xagent.core.tools.core.RAG_tools.utils import model_resolver

OPEN_MODEL_HUB = model_resolver._get_or_init_model_hub

KB = "kb"
# Writer table embeddings_vendor_new_model, once-tagged embeddings_VENDOR_new_model.
BOUND = "Vendor/new-model"
OTHER = "BAAI/old-model"
WRITER, ONCE = embeddings_table_candidates(BOUND)
OTHER_TABLE = embeddings_table_candidates(OTHER)[0]
# No candidate table of LEGACY; its rows sit in a provider-name table.
LEGACY = "acme-embed"
LEGACY_NAME = "acme-embed-v2"
LEGACY_TABLE = embeddings_table_name(LEGACY_NAME)
UNRESOLVED = "Could not resolve the searched table"


HUB_OPENED: List[str] = []


@pytest.fixture(autouse=True)
def store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """A temporary LanceDB directory and SQL file; opening the model hub is recorded."""
    from xagent.core.tools.core.RAG_tools.storage import reset_rag_storage_for_tests

    lancedb_dir = tmp_path / "status-lancedb"
    lancedb_dir.mkdir()
    monkeypatch.setenv("LANCEDB_DIR", str(lancedb_dir))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'hub.db'}")
    monkeypatch.delenv("XAGENT_VECTOR_BACKEND", raising=False)
    monkeypatch.delenv("VECTOR_STORE_BACKEND", raising=False)
    model_resolver._reset_model_hub_cache()
    HUB_OPENED.clear()

    def _record_hub() -> Any:
        HUB_OPENED.append("opened")
        raise RuntimeError("the model hub was opened")

    monkeypatch.setattr(model_resolver, "_get_or_init_model_hub", _record_hub)
    reset_rag_storage_for_tests()
    yield lancedb_dir
    reset_rag_storage_for_tests()
    model_resolver._reset_model_hub_cache()
    assert HUB_OPENED == []


@pytest.fixture
def both_spellings(store: Path) -> None:
    probe = store.parent / "case-probe"
    probe.mkdir()
    if (store.parent / "CASE-PROBE").exists():
        pytest.skip(
            "the writer and once-tagged table names differ only in letter case, "
            "and this filesystem does not tell them apart"
        )
    probe.rmdir()


def _store() -> Any:
    from xagent.core.tools.core.RAG_tools.storage.factory import (
        get_vector_index_store,
    )

    return get_vector_index_store()


def _save_collection(
    model_id: Optional[str] = BOUND, dimension: Optional[int] = 2
) -> None:
    from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo

    asyncio.run(
        cm.collection_manager.save_collection(
            CollectionInfo(
                name=KB, embedding_model_id=model_id, embedding_dimension=dimension
            )
        )
    )


def _seed_core(
    doc_ids: Sequence[str] = ("d1",),
    chunk_ids: Sequence[str] = ("c1", "c2"),
    owner: int = 7,
) -> None:
    now = datetime.now(timezone.utc)
    store = _store()
    store.upsert_documents(
        [
            {
                "collection": KB,
                "doc_id": doc_id,
                "file_id": None,
                "source_path": f"/{doc_id}.txt",
                "file_type": "txt",
                "content_hash": "C",
                "uploaded_at": now,
                "title": None,
                "language": None,
                "user_id": owner,
            }
            for doc_id in doc_ids
        ]
    )
    store.upsert_parses(
        [
            {
                "collection": KB,
                "doc_id": doc_id,
                "parse_hash": "p",
                "parser": "local:default@v1.0.0",
                "created_at": now,
                "params_json": json.dumps({"content_hash": "C"}),
                "parsed_content": "[]",
                "user_id": owner,
            }
            for doc_id in doc_ids
        ]
    )
    store.upsert_chunks(
        [
            {
                "collection": KB,
                "doc_id": doc_id,
                "parse_hash": "p",
                "chunk_id": chunk_id,
                "index": index,
                "text": chunk_id,
                "page_number": None,
                "section": None,
                "anchor": None,
                "json_path": None,
                "chunk_hash": "x",
                "config_hash": "cfg",
                "created_at": now,
                "metadata": None,
                "user_id": owner,
            }
            for doc_id in doc_ids
            for index, chunk_id in enumerate(chunk_ids)
        ]
    )


def _write(
    model: str, chunk_ids: Sequence[str], owner: int = 7, doc_id: str = "d1"
) -> None:
    """Rows written through the ingestion write path."""
    from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData
    from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
        write_vectors_to_db,
    )

    write_vectors_to_db(
        KB,
        [
            ChunkEmbeddingData(
                doc_id=doc_id,
                chunk_id=chunk_id,
                parse_hash="p",
                model=model,
                vector=[0.5, 0.5],
                text=chunk_id,
                chunk_hash=f"h-{chunk_id}",
                metadata=None,
            )
            for chunk_id in chunk_ids
        ],
        create_index=False,
        user_id=owner,
        verify_bound=False,
    )


def _create_table(name: str) -> None:
    from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
        ensure_embeddings_table,
    )

    conn = _store().get_raw_connection()
    ensure_embeddings_table(conn, name.removeprefix("embeddings_"), vector_dim=2)


def _move_writer_rows_to_once_tagged() -> None:
    conn = _store().get_raw_connection()
    rows = conn.open_table(WRITER).search().where("true").to_arrow()
    _create_table(ONCE)
    conn.open_table(ONCE).add(rows)
    conn.drop_table(WRITER)


def _project(
    user_id: Optional[int] = None,
    is_admin: bool = True,
    model_tag: Optional[str] = None,
) -> Tuple[Any, Any]:
    listed = list_documents(KB, user_id=user_id, is_admin=is_admin)
    assert listed.status == "success", listed.message
    assert HUB_OPENED == []
    detail = get_document_stats(
        KB, "d1", model_tag=model_tag, user_id=user_id, is_admin=is_admin
    )
    assert detail.status == "success", detail.message
    assert HUB_OPENED == []
    summary = {doc.doc_id: doc for doc in listed.documents}["d1"]
    return summary, detail.data


def _bound_c1_other_c1() -> None:
    _seed_core()
    _write(BOUND, ["c1"])
    _write(OTHER, ["c1"])


def test_other_model_rows_do_not_complete_a_partially_embedded_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_collection()
    _bound_c1_other_c1()
    closed: List[Optional[str]] = []
    real_close = collections_module._safe_close_table

    def _close(table: Any) -> None:
        closed.append(getattr(table, "name", None))
        real_close(table)

    monkeypatch.setattr(collections_module, "_safe_close_table", _close)

    summary, detail = _project()

    assert summary.status == S.PARTIALLY_EMBEDDED
    assert detail.status == S.PARTIALLY_EMBEDDED
    assert (summary.chunk_count, summary.embedding_count) == (2, 2)
    assert (detail.chunk_count, detail.embedding_count) == (2, 2)
    assert detail.embedding_breakdown == {WRITER: 1, OTHER_TABLE: 1}
    assert closed.count(WRITER) == 2


@pytest.mark.parametrize("other_rows", [False, True])
def test_complete_bound_rows_infer_success(other_rows: bool) -> None:
    _save_collection()
    _seed_core()
    _write(BOUND, ["c1", "c2"])
    if other_rows:
        _write(OTHER, ["c1", "c2"])

    summary, detail = _project()

    expected = 4 if other_rows else 2
    assert (summary.status, summary.embedding_count) == (S.SUCCESS, expected)
    assert (detail.status, detail.embedding_count) == (S.SUCCESS, expected)


def test_other_model_rows_only_with_the_searched_table_present() -> None:
    _save_collection()
    _seed_core()
    _create_table(WRITER)
    _write(OTHER, ["c1", "c2"])

    summary, detail = _project()

    assert (summary.status, summary.embedding_count) == (S.CHUNKED, 2)
    assert (detail.status, detail.embedding_count) == (S.CHUNKED, 2)


def test_no_table_of_the_bound_model_keeps_the_sum_over_every_table(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _save_collection()
    _seed_core()
    _write(OTHER, ["c1", "c2"])

    summary, detail = _project()

    assert (summary.status, summary.embedding_count) == (S.SUCCESS, 2)
    assert (detail.status, detail.embedding_count) == (S.SUCCESS, 2)
    assert UNRESOLVED not in caplog.text


def test_a_status_record_is_reported_as_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.management import collections

    _save_collection()
    _bound_c1_other_c1()
    # Both projections load status records with the no-access scope, so a
    # record written to the store is not returned to them.
    record = {"collection": KB, "doc_id": "d1", "status": S.FAILED.value}
    monkeypatch.setattr(
        collections, "_load_ingestion_status_impl", lambda **_kwargs: [record]
    )

    summary, detail = _project()

    assert (summary.status, summary.embedding_count) == (S.FAILED, 2)
    assert (detail.status, detail.embedding_count) == (S.FAILED, 2)


@pytest.mark.parametrize(
    "binding", [(BOUND, None), (None, None), ("none", 2), None], ids=str
)
def test_without_a_dimensioned_binding_the_sum_over_every_table_is_kept(
    binding: Optional[Tuple[Optional[str], Optional[int]]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    if binding is not None:
        _save_collection(*binding)
    _bound_c1_other_c1()
    if binding == ("none", 2):
        # The placeholder names no model, even with a table spelled after it.
        _create_table("embeddings_none")
    caplog.clear()

    summary, detail = _project()

    assert (summary.status, summary.embedding_count) == (S.SUCCESS, 2)
    assert (detail.status, detail.embedding_count) == (S.SUCCESS, 2)
    assert UNRESOLVED not in caplog.text
    assert "not found in hub" not in caplog.text


@pytest.mark.usefixtures("both_spellings")
def test_the_writer_table_is_counted_when_it_opens_even_when_empty() -> None:
    _save_collection()
    _seed_core()
    _write(BOUND, ["c1", "c2"])
    _move_writer_rows_to_once_tagged()
    _create_table(WRITER)

    summary, detail = _project()

    assert (summary.status, summary.embedding_count) == (S.CHUNKED, 2)
    assert (detail.status, detail.embedding_count) == (S.CHUNKED, 2)
    assert detail.embedding_breakdown == {ONCE: 2}


@pytest.mark.usefixtures("both_spellings")
def test_the_once_tagged_table_is_counted_when_the_writer_table_is_absent() -> None:
    _save_collection()
    _seed_core()
    _write(BOUND, ["c1"])
    _move_writer_rows_to_once_tagged()
    _write(OTHER, ["c2"])

    summary, detail = _project()

    assert (summary.status, summary.embedding_count) == (S.PARTIALLY_EMBEDDED, 2)
    assert (detail.status, detail.embedding_count) == (S.PARTIALLY_EMBEDDED, 2)
    assert detail.embedding_breakdown == {ONCE: 1, OTHER_TABLE: 1}


def test_the_list_counts_the_callers_rows_of_the_searched_table() -> None:
    _save_collection()
    _seed_core()
    _write(BOUND, ["c1"])
    _write(BOUND, ["c2"], owner=8)
    _write(OTHER, ["c2"])

    summary, detail = _project(user_id=7, is_admin=False)
    admin_summary, _ = _project()

    assert (summary.status, summary.embedding_count) == (S.PARTIALLY_EMBEDDED, 2)
    assert (admin_summary.status, admin_summary.embedding_count) == (S.SUCCESS, 3)
    # The detail infers from rows of every owner and reports the caller's breakdown.
    assert (detail.status, detail.embedding_count) == (S.SUCCESS, 3)
    assert detail.embedding_breakdown == {WRITER: 1, OTHER_TABLE: 1}


def test_an_explicit_model_tag_keeps_counting_that_models_tables() -> None:
    _save_collection()
    _seed_core()
    _write(BOUND, ["c1"])
    _write(OTHER, ["c1", "c2"])

    _, detail = _project(model_tag=OTHER)

    assert (detail.status, detail.embedding_count) == (S.SUCCESS, 2)
    assert detail.embedding_breakdown == {
        name: 2 if name == OTHER_TABLE else 0
        for name in embeddings_table_candidates(OTHER)
    }


def test_an_unreadable_binding_keeps_the_sum_over_every_table(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _save_collection()
    _seed_core(doc_ids=("d1", "d2"))
    _write(BOUND, ["c1"])
    _write(OTHER, ["c1"])

    def _fail(name: str) -> Any:
        raise RuntimeError("metadata unavailable")

    monkeypatch.setattr(cm, "get_collection_if_present_sync", _fail)

    summary, detail = _project()

    assert (summary.status, summary.embedding_count) == (S.SUCCESS, 2)
    assert (detail.status, detail.embedding_count) == (S.SUCCESS, 2)
    assert caplog.text.count(UNRESOLVED) == 2


def test_a_milvus_deployment_keeps_the_sum_over_every_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_collection()
    _seed_core()
    _create_table(WRITER)
    _write(OTHER, ["c1", "c2"])
    monkeypatch.setenv("XAGENT_VECTOR_BACKEND", "milvus")

    summary, detail = _project()

    assert (summary.status, summary.embedding_count) == (S.SUCCESS, 2)
    assert (detail.status, detail.embedding_count) == (S.SUCCESS, 2)


def test_the_list_reads_the_binding_once(monkeypatch: pytest.MonkeyPatch) -> None:
    _save_collection()
    _seed_core(doc_ids=("d1", "d2", "d3"))
    _write(BOUND, ["c1"])
    calls: List[str] = []
    real = cm.get_collection_if_present_sync

    def _spy(name: str) -> Any:
        calls.append(name)
        return real(name)

    monkeypatch.setattr(cm, "get_collection_if_present_sync", _spy)

    listed = list_documents(KB, is_admin=True)

    assert HUB_OPENED == []
    assert [doc.status for doc in listed.documents] == [
        S.PARTIALLY_EMBEDDED,
        S.CHUNKED,
        S.CHUNKED,
    ]
    assert calls == [KB]


@pytest.mark.usefixtures("both_spellings")
def test_a_writer_table_that_does_not_open_is_passed_over_as_search_does(
    store: Path,
) -> None:
    _save_collection()
    _seed_core()
    _write(BOUND, ["c1", "c2"])
    _move_writer_rows_to_once_tagged()
    (store / f"{WRITER}.lance").mkdir()
    assert WRITER in _store().list_table_names()
    assert _store().open_embeddings_table(BOUND)[1] == ONCE

    summary, detail = _project()

    assert (summary.status, summary.embedding_count) == (S.SUCCESS, 2)
    assert (detail.status, detail.embedding_count) == (S.SUCCESS, 2)


def test_rows_only_in_a_provider_name_table_keep_the_sum_over_every_table() -> None:
    _save_collection(LEGACY)
    _seed_core()
    _write(LEGACY_NAME, ["c1"])
    _write(OTHER, ["c1"])
    assert LEGACY_TABLE not in embeddings_table_candidates(LEGACY)

    summary, detail = _project()

    assert (summary.status, summary.embedding_count) == (S.SUCCESS, 2)
    assert (detail.status, detail.embedding_count) == (S.SUCCESS, 2)
    assert detail.embedding_breakdown == {LEGACY_TABLE: 1, OTHER_TABLE: 1}


def test_a_returned_name_the_listing_lacks_keeps_the_sum_over_every_table(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
        LanceDBVectorIndexStore,
    )

    _save_collection()
    _bound_c1_other_c1()
    real_open = LanceDBVectorIndexStore.open_embeddings_table

    def _other_spelling(self: Any, model_tag: str, **kwargs: Any) -> Any:
        table, _name = real_open(self, model_tag, **kwargs)
        return table, ONCE

    monkeypatch.setattr(
        LanceDBVectorIndexStore, "open_embeddings_table", _other_spelling
    )
    assert ONCE not in _store().list_table_names()

    summary, detail = _project()

    assert (summary.status, summary.embedding_count) == (S.SUCCESS, 2)
    assert (detail.status, detail.embedding_count) == (S.SUCCESS, 2)
    assert detail.embedding_breakdown == {WRITER: 1, OTHER_TABLE: 1}
    assert UNRESOLVED not in caplog.text


def test_a_status_read_leaves_a_fresh_sql_database_ready_for_init_db(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from sqlalchemy import create_engine, inspect

    from xagent.db.migration import get_alembic_revision
    from xagent.web.models import database

    url = f"sqlite:///{tmp_path / 'fresh.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setattr(model_resolver, "_get_or_init_model_hub", OPEN_MODEL_HUB)
    for name in (
        "_MODEL_HUB_ENGINE",
        "_MODEL_HUB_SESSION_LOCAL",
        "_MODEL_HUB_MODEL",
        "_MODEL_HUB_DB_URL",
    ):
        monkeypatch.setattr(model_resolver, name, None)
    previous_engine = database._engine
    monkeypatch.setattr(database, "_engine", previous_engine)
    monkeypatch.setattr(database, "_SessionLocal", database._SessionLocal)
    _save_collection()
    _seed_core()
    _write(OTHER, ["c1", "c2"])

    def _tables() -> List[str]:
        engine = create_engine(url)
        try:
            return inspect(engine).get_table_names()
        finally:
            engine.dispose()

    try:
        assert _tables() == []
        listed = list_documents(KB, is_admin=True)
        detail = get_document_stats(KB, "d1", is_admin=True)
        tables_after_reads = _tables()

        database.init_db(db_url=url)

        assert get_alembic_revision(database.get_engine())
        assert tables_after_reads == []
        assert [doc.status for doc in listed.documents] == [S.SUCCESS]
        assert detail.data is not None and detail.data.status == S.SUCCESS
    finally:
        if model_resolver._MODEL_HUB_ENGINE is not None:
            model_resolver._MODEL_HUB_ENGINE.dispose()
        if database._engine is not None and database._engine is not previous_engine:
            database._engine.dispose()
