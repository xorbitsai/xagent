"""Tests for the collection handle embedding lifecycle (#510).

The handle owns collection-scoped embedding storage mechanics: query-vector
validation, chunk-selection-for-embedding reads, vector writes/upserts, stale +
rollback cleanup. Embedding provider calls and batching stay in the pipeline.

Storage isolation/reset is provided by the autouse ``isolate_rag_storage``
fixture in ``tests/conftest.py``.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import VectorValidationError
from xagent.core.tools.core.RAG_tools.core.schemas import (
    ChunkEmbeddingData,
    EmbeddingReadResponse,
    EmbeddingWriteResponse,
)
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
from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import to_model_tag
from xagent.core.tools.core.RAG_tools.storage.factory import (
    get_metadata_store,
    get_vector_index_store,
)
from xagent.core.tools.core.RAG_tools.utils.metadata_utils import serialize_metadata


def make_handle(collection: str = "coll") -> LanceDBCollectionHandle:
    context = KBCollectionContext(
        collection=collection,
        user_scope=KBUserScope(user_id=None, is_admin=True),
        access_mode=KBAccessMode.WRITE,
        allow_create=True,
        hide_missing=True,
        metadata_store=get_metadata_store(),
        vector_index_store=get_vector_index_store(),
        backend=KBStorageBackend.LANCEDB,
        capabilities=KBBackendCapabilities.lancedb(),
        collection_info=None,
    )
    return LanceDBCollectionHandle(context)


def _seed_chunk(
    collection: str,
    doc_id: str,
    parse_hash: str,
    chunk_id: str,
    *,
    index: int = 0,
    page_number=None,
    section=None,
    anchor=None,
    json_path=None,
    metadata=None,
    user_id=None,
) -> None:
    get_vector_index_store().upsert_chunks(
        [
            {
                "collection": collection,
                "doc_id": doc_id,
                "parse_hash": parse_hash,
                "chunk_id": chunk_id,
                "index": index,
                "text": f"text-{chunk_id}",
                "page_number": page_number,
                "section": section,
                "anchor": anchor,
                "json_path": json_path,
                "chunk_hash": f"ch-{chunk_id}",
                "config_hash": "cfg1",
                "created_at": datetime.now(timezone.utc),
                "metadata": serialize_metadata(metadata or {"k": "v"}),
                "user_id": user_id,
            }
        ]
    )


def _seed_embedding(
    collection: str,
    doc_id: str,
    parse_hash: str,
    chunk_id: str,
    model: str,
    *,
    vector=None,
    user_id=None,
) -> None:
    vec = vector or [0.1, 0.2]
    get_vector_index_store().upsert_embeddings(
        model,
        [
            {
                "collection": collection,
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "parse_hash": parse_hash,
                "model": model,
                "vector": vec,
                "vector_dimension": len(vec),
                "text": f"text-{chunk_id}",
                "chunk_hash": f"ch-{chunk_id}",
                "created_at": datetime.now(timezone.utc),
                "metadata": "{}",
                "user_id": user_id,
            }
        ],
    )


class TestHandleValidateQueryVector:
    def test_accepts_valid_vector(self) -> None:
        make_handle().validate_query_vector([0.1, 0.2, 0.3])

    def test_accepts_numpy_scalars(self) -> None:
        make_handle().validate_query_vector([np.float64(0.1), np.int32(2)])

    def test_rejects_non_list(self) -> None:
        with pytest.raises(VectorValidationError, match="must be a list"):
            make_handle().validate_query_vector((0.1, 0.2))

    def test_rejects_empty(self) -> None:
        with pytest.raises(VectorValidationError, match="cannot be empty"):
            make_handle().validate_query_vector([])

    def test_rejects_non_numbers(self) -> None:
        with pytest.raises(VectorValidationError, match="only numbers"):
            make_handle().validate_query_vector([0.1, "x"])

    def test_rejects_nan_and_inf(self) -> None:
        handle = make_handle()
        with pytest.raises(VectorValidationError, match="NaN or infinity"):
            handle.validate_query_vector([0.1, float("nan")])
        with pytest.raises(VectorValidationError, match="NaN or infinity"):
            handle.validate_query_vector([0.1, float("inf")])


class TestHandleReadChunksNeedingEmbedding:
    def test_excludes_already_embedded_chunk_ids(self) -> None:
        handle = make_handle("coll")
        _seed_chunk("coll", "d1", "h1", "c0", index=0)
        _seed_chunk("coll", "d1", "h1", "c1", index=1)
        _seed_embedding("coll", "d1", "h1", "c0", "test_model")

        result = handle.read_chunks_needing_embedding(
            "d1", "h1", "test_model", is_admin=True
        )
        assert isinstance(result, EmbeddingReadResponse)
        assert result.total_count == 2
        assert result.pending_count == 1
        assert [c.chunk_id for c in result.chunks] == ["c1"]

    def test_empty_when_no_chunks(self) -> None:
        handle = make_handle("coll")
        result = handle.read_chunks_needing_embedding(
            "missing", "h1", "test_model", is_admin=True
        )
        assert result.total_count == 0
        assert result.pending_count == 0
        assert result.chunks == []

    def test_normalizes_optional_fields(self) -> None:
        handle = make_handle("coll")
        _seed_chunk(
            "coll",
            "d1",
            "h1",
            "c0",
            index=2,
            page_number=None,
            section=None,
            anchor=None,
            json_path=None,
            metadata={"a": 1},
        )
        result = handle.read_chunks_needing_embedding(
            "d1", "h1", "test_model", is_admin=True
        )
        assert result.pending_count == 1
        chunk = result.chunks[0]
        assert chunk.page_number is None
        assert chunk.section is None
        assert chunk.anchor is None
        assert chunk.json_path is None
        assert chunk.index == 2
        assert chunk.metadata == {"a": 1}

    def test_collection_scoped(self) -> None:
        handle = make_handle("coll")
        _seed_chunk("coll", "d1", "h1", "c0")
        _seed_chunk("other", "d1", "h1", "c1")
        result = handle.read_chunks_needing_embedding(
            "d1", "h1", "test_model", is_admin=True
        )
        assert result.total_count == 1
        assert [c.chunk_id for c in result.chunks] == ["c0"]

    def test_matches_model_tag_table(self) -> None:
        # Embeddings recorded under one model tag must not mask another model's
        # pending read (staleness is per-model-tag table).
        handle = make_handle("coll")
        _seed_chunk("coll", "d1", "h1", "c0")
        _seed_embedding("coll", "d1", "h1", "c0", "test_model")
        # A different model has no embeddings table -> chunk is still pending.
        result = handle.read_chunks_needing_embedding(
            "d1", "h1", "other_model", is_admin=True
        )
        assert result.pending_count == 1
        assert to_model_tag("other_model") == "other_model"

    def test_nan_page_number_normalized_to_none(self) -> None:
        # A missing page number can surface as a pandas/LanceDB NaN sentinel; it
        # must normalize to None, not be coerced to page 1.
        from unittest.mock import MagicMock

        handle, store = _mock_store_handle("coll")
        store.count_rows_or_zero.side_effect = [1, 0]  # 1 chunk, 0 embeddings
        batch = MagicMock()
        batch.to_pylist.return_value = [
            {
                "doc_id": "d1",
                "chunk_id": "c0",
                "parse_hash": "h1",
                "index": 0,
                "text": "t",
                "chunk_hash": "ch-c0",
                "page_number": float("nan"),
                "section": None,
                "anchor": None,
                "json_path": None,
                "metadata": None,
            }
        ]
        store.iter_batches.return_value = [batch]

        result = handle.read_chunks_needing_embedding(
            "d1", "h1", "test_model", is_admin=True
        )
        assert result.pending_count == 1
        assert result.chunks[0].page_number is None


def _embedding(
    doc_id: str,
    chunk_id: str,
    parse_hash: str,
    model: str,
    *,
    vector,
    text: str | None = None,
    metadata=None,
) -> ChunkEmbeddingData:
    return ChunkEmbeddingData(
        doc_id=doc_id,
        chunk_id=chunk_id,
        parse_hash=parse_hash,
        model=model,
        vector=list(vector),
        text=text or f"text-{chunk_id}",
        chunk_hash=f"ch-{chunk_id}",
        metadata=metadata,
    )


def _embedding_rows(model: str, collection: str, doc_id: str) -> list[dict]:
    store = get_vector_index_store()
    rows: list[dict] = []
    for batch in store.iter_batches(
        table_name=f"embeddings_{to_model_tag(model)}",
        filters={"collection": collection, "doc_id": doc_id},
        is_admin=True,
    ):
        rows.extend(batch.to_pylist())
    return rows


class TestHandleWriteEmbeddings:
    def test_persists_exact_row_and_counts(self) -> None:
        handle = make_handle("coll")
        result = handle.write_embeddings(
            [
                _embedding(
                    "d1",
                    "c0",
                    "h1",
                    "test_model",
                    vector=[0.1, 0.2, 0.3],
                    metadata={"a": 1},
                )
            ],
            create_index=False,
            user_id=7,
        )
        assert isinstance(result, EmbeddingWriteResponse)
        assert result.upsert_count == 1
        assert result.deleted_stale_count == 0
        assert result.index_status == "skipped"

        rows = _embedding_rows("test_model", "coll", "d1")
        assert len(rows) == 1
        row = rows[0]
        assert row["collection"] == "coll"
        assert row["chunk_id"] == "c0"
        assert row["model"] == "test_model"
        assert row["vector_dimension"] == 3
        assert row["user_id"] == 7

    def test_empty_short_circuits_without_table(self) -> None:
        handle = make_handle("coll")
        result = handle.write_embeddings([], create_index=True)
        assert result.upsert_count == 0
        assert result.deleted_stale_count == 0
        assert result.index_status == "skipped"
        assert (
            "embeddings_test_model" not in get_vector_index_store().list_table_names()
        )

    def test_rewrite_is_idempotent(self) -> None:
        handle = make_handle("coll")
        handle.write_embeddings(
            [_embedding("d1", "c0", "h1", "test_model", vector=[0.1, 0.2])],
            create_index=False,
        )
        handle.write_embeddings(
            [_embedding("d1", "c0", "h1", "test_model", vector=[0.3, 0.4], text="new")],
            create_index=False,
        )
        rows = _embedding_rows("test_model", "coll", "d1")
        assert len(rows) == 1
        assert rows[0]["text"] == "new"

    def test_multi_model_lands_in_per_model_tables(self) -> None:
        handle = make_handle("coll")
        result = handle.write_embeddings(
            [
                _embedding("d1", "c0", "h1", "m1", vector=[1.0, 2.0]),
                _embedding("d1", "c0", "h1", "m2", vector=[3.0, 4.0, 5.0]),
            ],
            create_index=False,
        )
        assert result.upsert_count == 2
        assert len(_embedding_rows("m1", "coll", "d1")) == 1
        assert len(_embedding_rows("m2", "coll", "d1")) == 1

    def test_intra_batch_dimension_mismatch_raises(self) -> None:
        handle = make_handle("coll")
        with pytest.raises(VectorValidationError):
            handle.write_embeddings(
                [
                    _embedding("d1", "c0", "h1", "test_model", vector=[1.0, 2.0]),
                    _embedding("d1", "c1", "h1", "test_model", vector=[1.0, 2.0, 3.0]),
                ],
                create_index=False,
            )

    def test_writes_into_context_collection(self) -> None:
        handle = make_handle("coll_a")
        handle.write_embeddings(
            [_embedding("d1", "c0", "h1", "test_model", vector=[1.0, 2.0])],
            create_index=False,
        )
        assert len(_embedding_rows("test_model", "coll_a", "d1")) == 1
        assert len(_embedding_rows("test_model", "coll_b", "d1")) == 0


class TestHandleDeleteEmbeddingRecords:
    def test_deletes_targeted_model_table_only(self) -> None:
        handle = make_handle("coll")
        _seed_embedding("coll", "d1", "h1", "c0", "m1")
        _seed_embedding("coll", "d1", "h1", "c0", "m2")

        deleted = handle.delete_embedding_records(
            "d1", parse_hash="h1", model_tag="m1", is_admin=True
        )
        assert deleted == 1
        store = get_vector_index_store()
        assert (
            store.count_rows("embeddings_m1", {"collection": "coll"}, is_admin=True)
            == 0
        )
        assert (
            store.count_rows("embeddings_m2", {"collection": "coll"}, is_admin=True)
            == 1
        )

    def test_model_tag_none_spans_all_tables(self) -> None:
        handle = make_handle("coll")
        _seed_embedding("coll", "d1", "h1", "c0", "m1")
        _seed_embedding("coll", "d1", "h1", "c0", "m2")
        assert handle.delete_embedding_records("d1", is_admin=True) == 2

    def test_narrows_by_chunk_ids_and_idempotent(self) -> None:
        handle = make_handle("coll")
        _seed_embedding("coll", "d1", "h1", "c0", "m1")
        _seed_embedding("coll", "d1", "h1", "c1", "m1")
        assert (
            handle.delete_embedding_records(
                "d1", chunk_ids=["c0"], model_tag="m1", is_admin=True
            )
            == 1
        )
        assert (
            handle.delete_embedding_records(
                "d1", chunk_ids=["c0"], model_tag="m1", is_admin=True
            )
            == 0
        )

    def test_collection_scoped(self) -> None:
        handle = make_handle("coll")
        _seed_embedding("coll", "d1", "h1", "c0", "m1")
        _seed_embedding("other", "d1", "h1", "c0", "m1")
        assert handle.delete_embedding_records("d1", model_tag="m1", is_admin=True) == 1
        store = get_vector_index_store()
        assert (
            store.count_rows("embeddings_m1", {"collection": "other"}, is_admin=True)
            == 1
        )


class TestHandleEmbeddingRollback:
    def test_snapshot_then_delete_then_restore_round_trip(self) -> None:
        handle = make_handle("coll")
        _seed_chunk("coll", "d1", "h1", "c0")
        _seed_chunk("coll", "d1", "h1", "c1")
        _seed_embedding("coll", "d1", "h1", "c0", "test_model", vector=[0.1, 0.2])
        _seed_embedding("coll", "d1", "h1", "c1", "test_model", vector=[0.3, 0.4])

        snapshot = handle.snapshot_embeddings("d1", "h1", is_admin=True)
        assert snapshot is not None
        assert {r.chunk_id for r in snapshot.records} == {"c0", "c1"}

        # Destroy, then restore brings every row back.
        assert handle.delete_created_embeddings("d1", "h1", is_admin=True) == 2
        store = get_vector_index_store()
        assert (
            store.count_rows(
                "embeddings_test_model", {"collection": "coll"}, is_admin=True
            )
            == 0
        )

        handle.restore_embeddings(snapshot)
        # read-for-embedding sees both chunks embedded again (none pending).
        result = handle.read_chunks_needing_embedding(
            "d1", "h1", "test_model", is_admin=True
        )
        assert result.total_count == 2
        assert result.pending_count == 0

    def test_snapshot_spans_multiple_model_tables(self) -> None:
        handle = make_handle("coll")
        _seed_embedding("coll", "d1", "h1", "c0", "m1", vector=[0.1, 0.2])
        _seed_embedding("coll", "d1", "h1", "c0", "m2", vector=[0.3, 0.4, 0.5])

        snapshot = handle.snapshot_embeddings("d1", "h1", is_admin=True)
        assert snapshot is not None
        grouped = snapshot.group_by_model_tag()
        assert set(grouped.keys()) == {"m1", "m2"}

    def test_snapshot_none_when_absent(self) -> None:
        handle = make_handle("coll")
        assert handle.snapshot_embeddings("d1", "h1", is_admin=True) is None

    def test_snapshot_narrows_by_model_tag(self) -> None:
        handle = make_handle("coll")
        _seed_embedding("coll", "d1", "h1", "c0", "m1")
        _seed_embedding("coll", "d1", "h1", "c0", "m2")
        snapshot = handle.snapshot_embeddings("d1", "h1", model_tag="m1", is_admin=True)
        assert snapshot is not None
        assert {r.model for r in snapshot.records} == {"m1"}

    def test_restore_rejects_snapshot_from_other_collection(self) -> None:
        from xagent.core.tools.core.RAG_tools.core.exceptions import (
            DocumentValidationError,
        )

        source = make_handle("coll_a")
        _seed_embedding("coll_a", "d1", "h1", "c0", "m1")
        snapshot = source.snapshot_embeddings("d1", "h1", is_admin=True)
        assert snapshot is not None

        other = make_handle("coll_b")
        with pytest.raises(DocumentValidationError, match="cannot restore"):
            other.restore_embeddings(snapshot)

    def test_delete_created_idempotent_multi_model(self) -> None:
        handle = make_handle("coll")
        _seed_embedding("coll", "d1", "h1", "c0", "m1")
        _seed_embedding("coll", "d1", "h1", "c0", "m2")
        assert handle.delete_created_embeddings("d1", "h1", is_admin=True) == 2
        assert handle.delete_created_embeddings("d1", "h1", is_admin=True) == 0

    def test_cleanup_failure_propagates_for_incomplete_rollback(self) -> None:
        # A store-side delete failure must surface (not be swallowed) so the
        # coordinator can report the rollback as incomplete. The live coordinator
        # wiring lands in #514; this pins the handle-level contract now.
        handle, store = _mock_store_handle()
        store.delete_embedding_records.side_effect = RuntimeError(
            "vector store unavailable"
        )
        with pytest.raises(RuntimeError, match="vector store unavailable"):
            handle.delete_created_embeddings("d1", "h1", is_admin=True)
        store.delete_embedding_records.assert_called_once()


def _mock_store_handle(collection: str = "coll"):
    """Build a handle whose vector store is a MagicMock (mechanics tests)."""
    from unittest.mock import MagicMock

    store = MagicMock()
    context = KBCollectionContext(
        collection=collection,
        user_scope=KBUserScope(user_id=None, is_admin=True),
        access_mode=KBAccessMode.WRITE,
        allow_create=True,
        hide_missing=True,
        metadata_store=MagicMock(),
        vector_index_store=store,
        backend=KBStorageBackend.LANCEDB,
        capabilities=KBBackendCapabilities.lancedb(),
        collection_info=None,
    )
    return LanceDBCollectionHandle(context), store


def _status(value: str):
    from unittest.mock import MagicMock

    result = MagicMock()
    result.status = value
    return result


class TestHandleWriteEmbeddingsMechanics:
    """Mechanics moved from the old vector_manager impl, tested at the handle."""

    def test_batch_size_from_env_splits_upserts(self, monkeypatch) -> None:
        monkeypatch.setenv("LANCEDB_BATCH_SIZE", "2")
        handle, store = _mock_store_handle()
        store.create_index.return_value = _status("skipped")
        result = handle.write_embeddings(
            [
                _embedding("d1", f"c{i}", "h1", "m1", vector=[0.1, 0.2])
                for i in range(3)
            ],
            create_index=False,
        )
        assert result.upsert_count == 3
        # 3 embeddings in batches of 2 -> two upsert calls.
        assert store.upsert_embeddings.call_count == 2

    def test_invalid_int_env_falls_back_to_defaults(self, monkeypatch) -> None:
        # A malformed integer override must fall back to the default instead of
        # crashing the write with a ValueError.
        monkeypatch.setenv("LANCEDB_BATCH_SIZE", "not-a-number")
        monkeypatch.setenv("LANCEDB_MAX_SPILL_RETRIES", "")
        handle, store = _mock_store_handle()
        result = handle.write_embeddings(
            [_embedding("d1", "c0", "h1", "m1", vector=[0.1, 0.2])],
            create_index=False,
        )
        assert result.upsert_count == 1
        store.upsert_embeddings.assert_called_once()

    def test_spill_error_reduces_batch_and_retries(self, monkeypatch) -> None:
        monkeypatch.setenv("LANCEDB_BATCH_SIZE", "100")
        handle, store = _mock_store_handle()
        calls = {"n": 0}

        def _upsert(model_tag, records):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("Spill has sent an error")

        store.upsert_embeddings.side_effect = _upsert
        result = handle.write_embeddings(
            [_embedding("d1", "c0", "h1", "m1", vector=[0.1, 0.2])],
            create_index=False,
        )
        # First batch spills, retried successfully -> 1 upserted, 2 calls.
        assert result.upsert_count == 1
        assert store.upsert_embeddings.call_count == 2

    def test_non_spill_error_propagates(self) -> None:
        handle, store = _mock_store_handle()
        store.upsert_embeddings.side_effect = RuntimeError("boom")
        with pytest.raises(Exception, match="boom"):
            handle.write_embeddings(
                [_embedding("d1", "c0", "h1", "m1", vector=[0.1, 0.2])],
                create_index=False,
            )

    @pytest.mark.parametrize(
        "statuses,expected",
        [
            (["index_building"], "created"),
            (["index_ready"], "ready"),
            (["failed"], "failed"),
            (["index_corrupted"], "failed"),
            (["below_threshold"], "skipped_threshold"),
            (["other"], "skipped"),
            (["index_ready", "index_building"], "created"),
        ],
    )
    def test_index_status_aggregation(self, statuses, expected) -> None:
        handle, store = _mock_store_handle()
        store.create_index.side_effect = [_status(s) for s in statuses]
        embeddings = [
            _embedding("d1", "c0", f"h{i}", f"m{i}", vector=[0.1, 0.2])
            for i in range(len(statuses))
        ]
        result = handle.write_embeddings(embeddings, create_index=True)
        assert result.index_status == expected

    def test_index_creation_failure_maps_to_failed(self) -> None:
        handle, store = _mock_store_handle()
        store.create_index.side_effect = RuntimeError("index boom")
        result = handle.write_embeddings(
            [_embedding("d1", "c0", "h1", "m1", vector=[0.1, 0.2])],
            create_index=True,
        )
        assert result.index_status == "failed"

    def test_empty_collection_name_raises(self) -> None:
        from xagent.core.tools.core.RAG_tools.core.exceptions import (
            DocumentValidationError,
        )

        handle, _store = _mock_store_handle(collection="")
        with pytest.raises(
            DocumentValidationError, match="Collection name is required"
        ):
            handle.write_embeddings(
                [_embedding("d1", "c0", "h1", "m1", vector=[0.1, 0.2])],
                create_index=False,
            )
