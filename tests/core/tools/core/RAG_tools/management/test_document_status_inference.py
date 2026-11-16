"""A document without a status record is inferred from its fullest embeddings table."""

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
)
from xagent.core.tools.core.RAG_tools.management import collection_manager as cm
from xagent.core.tools.core.RAG_tools.management import (
    get_document_stats,
    list_documents,
)
from xagent.core.tools.core.RAG_tools.utils import model_resolver

OPEN_MODEL_HUB = model_resolver._get_or_init_model_hub

KB = "kb"
BOUND = "Vendor/new-model"
OTHER = "BAAI/old-model"
WRITER = embeddings_table_candidates(BOUND)[0]
OTHER_TABLE = embeddings_table_candidates(OTHER)[0]

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


def test_one_chunk_in_two_tables_is_partially_embedded() -> None:
    _save_collection()
    _bound_c1_other_c1()

    summary, detail = _project()

    assert (summary.status, summary.embedding_count) == (S.PARTIALLY_EMBEDDED, 2)
    assert (detail.status, detail.embedding_count) == (S.PARTIALLY_EMBEDDED, 2)
    assert summary.chunk_count == detail.chunk_count == 2
    assert detail.embedding_breakdown == {WRITER: 1, OTHER_TABLE: 1}


def test_chunks_split_across_two_tables_are_partially_embedded() -> None:
    _save_collection()
    _seed_core()
    _write(BOUND, ["c1"])
    _write(OTHER, ["c2"])

    summary, detail = _project()

    assert (summary.status, summary.embedding_count) == (S.PARTIALLY_EMBEDDED, 2)
    assert (detail.status, detail.embedding_count) == (S.PARTIALLY_EMBEDDED, 2)
    assert detail.embedding_breakdown == {WRITER: 1, OTHER_TABLE: 1}


@pytest.mark.parametrize("other_rows", [False, True])
def test_every_chunk_in_one_table_is_success(other_rows: bool) -> None:
    _save_collection()
    _seed_core()
    _write(BOUND, ["c1", "c2"])
    if other_rows:
        _write(OTHER, ["c1"])

    summary, detail = _project()

    expected = 3 if other_rows else 2
    assert (summary.status, summary.embedding_count) == (S.SUCCESS, expected)
    assert (detail.status, detail.embedding_count) == (S.SUCCESS, expected)


def test_every_chunk_in_a_table_of_a_model_not_bound_is_success() -> None:
    _save_collection()
    _seed_core()
    _create_table(WRITER)
    _write(OTHER, ["c1", "c2"])

    summary, detail = _project()

    assert (summary.status, summary.embedding_count) == (S.SUCCESS, 2)
    assert (detail.status, detail.embedding_count) == (S.SUCCESS, 2)


def test_chunks_without_embedding_rows_are_chunked() -> None:
    _save_collection()
    _seed_core()

    summary, detail = _project()

    assert (summary.status, summary.embedding_count) == (S.CHUNKED, 0)
    assert (detail.status, detail.embedding_count) == (S.CHUNKED, 0)


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


def test_an_explicit_model_tag_counts_that_models_tables() -> None:
    _save_collection()
    _seed_core()
    _write(BOUND, ["c1"])
    _write(OTHER, ["c1", "c2"])

    _, detail = _project(model_tag=BOUND)

    assert (detail.status, detail.embedding_count) == (S.PARTIALLY_EMBEDDED, 1)
    assert detail.embedding_breakdown == {
        name: 1 if name == WRITER else 0 for name in embeddings_table_candidates(BOUND)
    }


def test_the_list_infers_from_the_callers_rows_and_the_detail_from_all_owners() -> None:
    _save_collection()
    _seed_core()
    _write(BOUND, ["c1"])
    _write(BOUND, ["c2"], owner=8)
    _write(OTHER, ["c2"])

    summary, detail = _project(user_id=7, is_admin=False)
    admin_summary, _ = _project()

    assert (summary.status, summary.embedding_count) == (S.PARTIALLY_EMBEDDED, 2)
    assert (admin_summary.status, admin_summary.embedding_count) == (S.SUCCESS, 3)
    assert (detail.status, detail.embedding_count) == (S.SUCCESS, 3)
    assert detail.embedding_breakdown == {WRITER: 1, OTHER_TABLE: 1}


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


def test_an_injected_coordinator_does_not_infer_from_the_global_binding() -> None:
    from typing import cast
    from unittest.mock import AsyncMock, MagicMock

    from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo
    from xagent.core.tools.core.RAG_tools.kb import KBCoordinator
    from xagent.core.tools.core.RAG_tools.storage.factory import StorageFactory

    _save_collection(BOUND)
    _seed_core()
    _create_table(WRITER)
    _write(OTHER, ["c1", "c2"])
    vector_store = _store()

    class _StatusStore:
        def load_ingestion_status(self, **_kwargs: Any) -> List[Any]:
            return []

    class _InjectedFactory:
        def get_vector_index_store(self) -> Any:
            return vector_store

        def get_ingestion_status_store(self) -> Any:
            return _StatusStore()

        def get_metadata_store(self) -> Any:
            metadata = MagicMock()
            metadata.get_collection = AsyncMock(
                return_value=CollectionInfo(
                    name=KB, embedding_model_id=OTHER, embedding_dimension=2
                )
            )
            return metadata

        def get_main_pointer_store(self) -> Any:
            return MagicMock()

    injected = KBCoordinator(storage_factory=cast(StorageFactory, _InjectedFactory()))

    listed = injected.management.list_documents(KB, is_admin=True)
    detail = injected.management.get_document_stats(KB, "d1", is_admin=True)

    assert listed.status == "success", listed.message
    assert detail.status == "success", detail.message
    summary = {doc.doc_id: doc for doc in listed.documents}["d1"]
    assert (summary.status, summary.embedding_count) == (S.SUCCESS, 2)
    assert detail.data is not None
    assert (detail.data.status, detail.data.embedding_count) == (S.SUCCESS, 2)
    assert HUB_OPENED == []
