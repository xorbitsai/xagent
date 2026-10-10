"""Collection and document stats are read through the engine entry points.

LanceDB keeps the numbers it reported before (pinned as literal counts and
against the global aggregate for list stats); an engine whose vectors live
outside LanceDB must be able to supply its own counts.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

import lancedb
import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import ConfigurationError
from xagent.core.tools.core.RAG_tools.core.schemas import (
    CollectionInfo,
    DocumentProcessingStatus,
)
from xagent.core.tools.core.RAG_tools.kb import KBCoordinator, get_kb_coordinator
from xagent.core.tools.core.RAG_tools.kb.collection_handle import KBHandleProvider
from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import (
    embeddings_table_name,
)
from xagent.core.tools.core.RAG_tools.management import collections
from xagent.core.tools.core.RAG_tools.management.collection_manager import (
    rebuild_collection_metadata,
)
from xagent.core.tools.core.RAG_tools.storage.factory import (
    get_metadata_store,
    get_vector_index_store,
)

KB = "stats_kb"
TABLE_A = embeddings_table_name("model-a")
TABLE_B = embeddings_table_name("model-b")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _document_row(collection: str, doc_id: str, user_id: int) -> dict:
    return {
        "collection": collection,
        "doc_id": doc_id,
        "file_id": None,
        "source_path": f"/uploads/user_{user_id}/{doc_id}.txt",
        "file_type": "txt",
        "content_hash": "a" * 64,
        "uploaded_at": _now(),
        "title": None,
        "language": None,
        "user_id": user_id,
    }


def _parse_row(collection: str, doc_id: str, user_id: int) -> dict:
    return {
        "collection": collection,
        "doc_id": doc_id,
        "parse_hash": "ph",
        "parser": "p",
        "created_at": _now(),
        "params_json": "{}",
        "parsed_content": "full text",
        "user_id": user_id,
    }


def _seed_document(collection: str, doc_id: str, user_id: int) -> None:
    get_vector_index_store().upsert_documents(
        [_document_row(collection, doc_id, user_id)]
    )


def _seed_chunk(collection: str, doc_id: str, chunk_id: str, user_id: int) -> None:
    get_vector_index_store().upsert_chunks(
        [
            {
                "collection": collection,
                "doc_id": doc_id,
                "parse_hash": "ph",
                "chunk_id": chunk_id,
                "index": 0,
                "text": chunk_id,
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
        ]
    )


def _seed_embedding(
    collection: str, doc_id: str, chunk_id: str, model: str, user_id: int
) -> None:
    get_vector_index_store().upsert_embeddings(
        model,
        [
            {
                "collection": collection,
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "parse_hash": "ph",
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


def _seed_two_documents() -> None:
    """d1 (user 1): 2 chunks, 2 vectors in model A, 1 in model B; d2 (user 2): 1 chunk."""
    _seed_document(KB, "d1", 1)
    _seed_document(KB, "d2", 2)
    _seed_chunk(KB, "d1", "c1", 1)
    _seed_chunk(KB, "d1", "c2", 1)
    _seed_chunk(KB, "d2", "c3", 2)
    _seed_embedding(KB, "d1", "c1", "model-a", 1)
    _seed_embedding(KB, "d1", "c2", "model-a", 1)
    _seed_embedding(KB, "d1", "c1", "model-b", 1)
    _seed_chunk("other_kb", "d1", "c9", 1)
    _seed_embedding("other_kb", "d1", "c9", "model-a", 1)


def _count_rows_by_document(**kwargs):
    return get_kb_coordinator().count_rows_by_document_sync(KB, **kwargs)


def test_lancedb_counts_rows_per_document_and_table() -> None:
    _seed_two_documents()

    assert _count_rows_by_document(user_id=None, is_admin=True) == {
        "d1": {"chunks": 2, TABLE_A: 2, TABLE_B: 1},
        "d2": {"chunks": 1},
    }
    assert _count_rows_by_document(user_id=2, is_admin=False) == {"d2": {"chunks": 1}}
    assert _count_rows_by_document(user_id=1, is_admin=False, doc_id="d1") == {
        "d1": {"chunks": 2, TABLE_A: 2, TABLE_B: 1}
    }
    assert _count_rows_by_document(user_id=2, is_admin=False, doc_id="d1") == {}


def test_lancedb_document_list_and_detail_numbers() -> None:
    _seed_two_documents()
    store = get_vector_index_store()

    listed = collections.list_documents(KB, user_id=1, is_admin=False)
    by_doc = {summary.doc_id: summary for summary in listed.documents}
    old_chunks = store.aggregate_document_counts(
        table_name="chunks",
        doc_id_column="doc_id",
        collection_name=KB,
        user_id=1,
        is_admin=False,
    )
    assert set(by_doc) == {"d1"}
    assert by_doc["d1"].chunk_count == old_chunks["d1"] == 2
    assert by_doc["d1"].embedding_count == 3
    assert by_doc["d1"].status is DocumentProcessingStatus.SUCCESS

    # Totals ignore the caller scope, the per-table breakdown does not.
    stats = collections.get_document_stats(KB, "d1", user_id=2, is_admin=False)
    assert stats.data is not None
    assert stats.data.document_exists is True
    assert (stats.data.chunk_count, stats.data.embedding_count) == (2, 3)
    assert stats.data.embedding_breakdown == {}
    admin = collections.get_document_stats(KB, "d1", user_id=2, is_admin=True)
    assert admin.data is not None
    assert admin.data.embedding_breakdown == {TABLE_A: 2, TABLE_B: 1}

    owner = collections.get_document_stats(KB, "d1", user_id=1, is_admin=False)
    assert owner.data is not None
    assert owner.data.embedding_breakdown == {TABLE_A: 2, TABLE_B: 1}
    tagged = collections.get_document_stats(
        KB, "d1", model_tag="model-a", user_id=1, is_admin=False
    )
    assert tagged.data is not None
    assert tagged.data.embedding_count == 2
    assert tagged.data.embedding_breakdown == {TABLE_A: 2}
    other = collections.get_document_stats(
        KB, "d1", model_tag="model-a", user_id=2, is_admin=False
    )
    assert other.data is not None
    assert other.data.embedding_count == 0
    assert other.data.embedding_breakdown == {TABLE_A: 0}
    no_table = collections.get_document_stats(
        KB, "d1", model_tag="model-zzz", user_id=1, is_admin=False
    )
    assert no_table.status == "success"
    assert no_table.data is not None
    assert no_table.data.embedding_count == 0
    assert no_table.data.embedding_breakdown == {embeddings_table_name("model-zzz"): 0}


def test_lancedb_list_stats_are_one_global_aggregate() -> None:
    _seed_two_documents()
    store = get_vector_index_store()

    for user_id, is_admin in ((None, True), (1, False), (2, False)):
        assert get_kb_coordinator().aggregate_collection_stats_sync(
            user_id=user_id, is_admin=is_admin
        ) == store.aggregate_collection_stats(user_id=user_id, is_admin=is_admin)


def test_list_stats_reject_an_unimplemented_engine(monkeypatch) -> None:
    monkeypatch.setenv("XAGENT_VECTOR_BACKEND", "qdrant")

    with pytest.raises(ConfigurationError, match="not implemented"):
        KBHandleProvider().aggregate_collection_stats(user_id=None, is_admin=True)


def test_document_stats_see_rows_another_process_wrote() -> None:
    _seed_document(KB, "d1", 1)
    get_vector_index_store().upsert_parses([_parse_row(KB, "d1", 1)])
    before = collections.get_document_stats(KB, "d2", user_id=1, is_admin=False)
    assert before.data is not None
    assert (before.data.document_exists, before.data.parse_count) == (False, 0)

    other = lancedb.connect(os.environ["LANCEDB_DIR"])
    other.open_table("documents").add([_document_row(KB, "d2", 2)])
    other.open_table("parses").add([_parse_row(KB, "d2", 2)])

    after = collections.get_document_stats(KB, "d2", user_id=1, is_admin=False)
    assert after.data is not None
    assert (after.data.document_exists, after.data.parse_count) == (True, 1)


def test_list_documents_reports_a_handle_that_cannot_open() -> None:
    _seed_two_documents()
    asyncio.run(
        get_metadata_store().save_collection(
            CollectionInfo(name=KB, extra_metadata={"kb_storage": "bogus"})
        )
    )

    listed = collections.list_documents(KB, user_id=1, is_admin=False)

    assert (listed.status, listed.documents) == ("error", [])
    assert "bogus" in listed.message


class _ExternalVectorsCoordinator:
    """Engine whose vectors are not in LanceDB embeddings_* tables."""

    def __init__(self) -> None:
        self.batch_calls = 0

    def aggregate_collection_stats_sync(self, *, user_id, is_admin):
        self.batch_calls += 1
        return {KB: {"documents": 1, "parses": 1, "chunks": 2, "embeddings": 2}}

    def count_rows_by_document_sync(
        self, collection, *, user_id, is_admin, doc_id=None
    ):
        return {"d1": {"chunks": 2, "embeddings_external": 2, "parses": 1}}


def test_list_stats_come_from_the_engine_batch_entry() -> None:
    _seed_document(KB, "d1", 1)
    _seed_chunk(KB, "d1", "c1", 1)
    coordinator = _ExternalVectorsCoordinator()

    result = asyncio.run(
        collections._list_collections_impl(
            user_id=1, is_admin=False, coordinator=coordinator
        )
    )

    assert coordinator.batch_calls == 1
    [info] = [info for info in result.collections if info.name == KB]
    assert (info.chunks, info.embeddings) == (2, 2)


def test_document_counts_come_from_the_coordinator_entry() -> None:
    _seed_document(KB, "d1", 1)
    _seed_chunk(KB, "d1", "c1", 1)
    _seed_chunk(KB, "d1", "c2", 1)
    coordinator = _ExternalVectorsCoordinator()

    listed = collections._list_documents_impl(
        KB, user_id=1, is_admin=False, coordinator=coordinator
    )
    [summary] = listed.documents
    assert summary.embedding_count == 2
    assert summary.status is DocumentProcessingStatus.SUCCESS

    stats = collections._get_document_stats_impl(
        KB, "d1", user_id=1, is_admin=False, coordinator=coordinator
    )
    assert stats.data is not None
    assert stats.data.embedding_count == 2
    assert stats.data.embedding_breakdown == {"embeddings_external": 2}


def _rebuild_with_stored_model(monkeypatch, reported_vectors: int):
    """Rebuild a KB whose embeddings_* tables hold no rows for it."""
    _seed_document(KB, "d1", 1)
    _seed_chunk(KB, "d1", "c1", 1)
    asyncio.run(
        get_metadata_store().save_collection(
            CollectionInfo(name=KB, embedding_model_id="model-a", embedding_dimension=2)
        )
    )
    aggregate = KBCoordinator.aggregate_collection_stats_sync

    def _engine_stats(self, **kwargs):
        stats = aggregate(self, **kwargs)
        stats[KB]["embeddings"] = reported_vectors
        return stats

    monkeypatch.setattr(KBCoordinator, "aggregate_collection_stats_sync", _engine_stats)
    asyncio.run(rebuild_collection_metadata())
    return asyncio.run(get_metadata_store().get_collection(KB))


def test_rebuild_keeps_the_model_of_vectors_outside_embeddings_tables(
    monkeypatch,
) -> None:
    rebuilt = _rebuild_with_stored_model(monkeypatch, reported_vectors=3)

    assert (rebuilt.embedding_model_id, rebuilt.embedding_dimension) == ("model-a", 2)
    assert (rebuilt.documents, rebuilt.chunks, rebuilt.embeddings) == (1, 1, 3)


def test_rebuild_still_clears_the_model_of_a_kb_without_vectors(monkeypatch) -> None:
    rebuilt = _rebuild_with_stored_model(monkeypatch, reported_vectors=0)

    assert (rebuilt.embedding_model_id, rebuilt.embedding_dimension) == (None, None)


def test_rebuild_leaves_a_kb_the_engine_did_not_report_unchanged(monkeypatch) -> None:
    asyncio.run(
        get_metadata_store().save_collection(
            CollectionInfo(
                name=KB,
                embedding_model_id="model-a",
                embedding_dimension=8,
                documents=3,
                embeddings=9,
            )
        )
    )
    monkeypatch.setattr(
        KBCoordinator, "aggregate_collection_stats_sync", lambda self, **kwargs: {}
    )

    for _ in range(2):
        asyncio.run(rebuild_collection_metadata())
        kept = asyncio.run(get_metadata_store().get_collection(KB))
        assert (kept.embedding_model_id, kept.embedding_dimension) == ("model-a", 8)
        assert (kept.documents, kept.embeddings) == (3, 9)


def test_rebuild_unbinds_an_unreported_kb_whose_stored_row_has_no_vectors(
    monkeypatch,
) -> None:
    asyncio.run(
        get_metadata_store().save_collection(
            CollectionInfo(
                name=KB,
                embedding_model_id="model-a",
                embedding_dimension=8,
                documents=0,
                embeddings=0,
            )
        )
    )
    monkeypatch.setattr(
        KBCoordinator, "aggregate_collection_stats_sync", lambda self, **kwargs: {}
    )

    asyncio.run(rebuild_collection_metadata())

    rebuilt = asyncio.run(get_metadata_store().get_collection(KB))
    assert (rebuilt.embedding_model_id, rebuilt.embedding_dimension) == (None, None)
