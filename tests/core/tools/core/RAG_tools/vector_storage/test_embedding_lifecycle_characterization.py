"""Characterization tests for the embedding lifecycle (#510 Cycle 0).

These golden-master tests pin the *current* embedding write/read/validate/
cleanup behavior against the real LanceDB store and the public vector-storage
boundary, before any mechanics move into the collection handle. They must stay
green through the refactor.

What they lock:
- the exact ``embeddings_{model_tag}`` column set written per row (lossless
  oracle for ``EmbeddingRecordDetail``);
- write idempotency (merge upsert on ``collection/doc_id/chunk_id`` does not
  duplicate rows);
- ``read_chunks_for_embedding`` excludes already-embedded ``chunk_id``s and its
  ``total/pending`` counts;
- multi-model writes land in one table per model tag;
- intra-batch dimension mismatch raises ``VectorValidationError``;
- ``deleted_stale_count`` is always 0 (stale deletion is a no-op);
- query-vector validation rules;
- rollback cleanup result contract (preview vs confirm vs skipped).

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
)
from xagent.core.tools.core.RAG_tools.kb import get_kb_coordinator
from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import to_model_tag
from xagent.core.tools.core.RAG_tools.storage.factory import get_vector_index_store
from xagent.core.tools.core.RAG_tools.utils.metadata_utils import serialize_metadata
from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
    read_chunks_for_embedding,
    validate_query_vector,
    write_vectors_to_db,
)

# The full physical schema of an ``embeddings_{model_tag}`` table. This is the
# lossless oracle the new ``EmbeddingRecordDetail`` must round-trip (Cycle 1).
EMBEDDING_COLUMNS = {
    "collection",
    "doc_id",
    "chunk_id",
    "parse_hash",
    "model",
    "vector",
    "vector_dimension",
    "text",
    "chunk_hash",
    "created_at",
    "metadata",
    "user_id",
}


def _seed_chunk(
    collection: str,
    doc_id: str,
    parse_hash: str,
    chunk_id: str,
    *,
    index: int = 0,
    text: str | None = None,
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
                "text": text or f"text-{chunk_id}",
                "page_number": None,
                "section": None,
                "anchor": None,
                "json_path": None,
                "chunk_hash": f"ch-{chunk_id}",
                "config_hash": "cfg1",
                "created_at": datetime.now(timezone.utc),
                "metadata": serialize_metadata(metadata or {"k": "v"}),
                "user_id": user_id,
            }
        ]
    )


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


def _read_embedding_rows(model: str, collection: str, doc_id: str) -> list[dict]:
    store = get_vector_index_store()
    table = f"embeddings_{to_model_tag(model)}"
    rows: list[dict] = []
    for batch in store.iter_batches(
        table_name=table,
        filters={"collection": collection, "doc_id": doc_id},
        is_admin=True,
    ):
        rows.extend(batch.to_pylist())
    return rows


class TestWriteColumnsAndIdempotency:
    def test_write_persists_exact_embedding_columns(self) -> None:
        _seed_chunk("coll", "d1", "h1", "c0")
        result = write_vectors_to_db(
            collection="coll",
            embeddings=[
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
        assert result.upsert_count == 1
        assert result.deleted_stale_count == 0
        assert result.index_status == "skipped"

        rows = _read_embedding_rows("test_model", "coll", "d1")
        assert len(rows) == 1
        row = rows[0]
        # Lossless oracle: the physical column set is exactly this.
        assert set(row.keys()) == EMBEDDING_COLUMNS
        assert row["collection"] == "coll"
        assert row["doc_id"] == "d1"
        assert row["chunk_id"] == "c0"
        assert row["parse_hash"] == "h1"
        assert row["model"] == "test_model"
        assert [round(v, 3) for v in row["vector"]] == [0.1, 0.2, 0.3]
        assert row["vector_dimension"] == 3
        assert row["text"] == "text-c0"
        assert row["chunk_hash"] == "ch-c0"
        assert row["user_id"] == 7
        assert isinstance(row["created_at"], datetime)
        # metadata is persisted as a serialized JSON string.
        assert isinstance(row["metadata"], str)

    def test_rewrite_same_keys_is_idempotent(self) -> None:
        _seed_chunk("coll", "d1", "h1", "c0")
        emb = _embedding("d1", "c0", "h1", "test_model", vector=[0.1, 0.2, 0.3])
        write_vectors_to_db(collection="coll", embeddings=[emb], create_index=False)
        # Re-write the same (collection, doc_id, chunk_id) -> overwrite, no dup.
        write_vectors_to_db(
            collection="coll",
            embeddings=[
                _embedding(
                    "d1", "c0", "h1", "test_model", vector=[0.4, 0.5, 0.6], text="new"
                )
            ],
            create_index=False,
        )
        rows = _read_embedding_rows("test_model", "coll", "d1")
        assert len(rows) == 1
        assert rows[0]["text"] == "new"

    def test_empty_write_short_circuits_without_table(self) -> None:
        result = write_vectors_to_db(
            collection="coll", embeddings=[], create_index=True
        )
        assert result.upsert_count == 0
        assert result.deleted_stale_count == 0
        assert result.index_status == "skipped"
        assert (
            "embeddings_test_model" not in get_vector_index_store().list_table_names()
        )


class TestReadChunksForEmbedding:
    def test_excludes_already_embedded_chunk_ids(self) -> None:
        _seed_chunk("coll", "d1", "h1", "c0", index=0)
        _seed_chunk("coll", "d1", "h1", "c1", index=1)
        # Embed only c0.
        write_vectors_to_db(
            collection="coll",
            embeddings=[_embedding("d1", "c0", "h1", "test_model", vector=[1.0, 2.0])],
            create_index=False,
        )

        result = read_chunks_for_embedding(
            collection="coll",
            doc_id="d1",
            parse_hash="h1",
            model="test_model",
            is_admin=True,
        )
        assert isinstance(result, EmbeddingReadResponse)
        assert result.total_count == 2
        assert result.pending_count == 1
        assert [c.chunk_id for c in result.chunks] == ["c1"]

    def test_no_chunks_returns_empty(self) -> None:
        result = read_chunks_for_embedding(
            collection="coll",
            doc_id="missing",
            parse_hash="h1",
            model="test_model",
            is_admin=True,
        )
        assert result.total_count == 0
        assert result.pending_count == 0
        assert result.chunks == []

    def test_all_embedded_yields_no_pending(self) -> None:
        _seed_chunk("coll", "d1", "h1", "c0")
        write_vectors_to_db(
            collection="coll",
            embeddings=[_embedding("d1", "c0", "h1", "test_model", vector=[1.0, 2.0])],
            create_index=False,
        )
        result = read_chunks_for_embedding(
            collection="coll",
            doc_id="d1",
            parse_hash="h1",
            model="test_model",
            is_admin=True,
        )
        assert result.total_count == 1
        assert result.pending_count == 0


class TestMultiModelAndDimensions:
    def test_multi_model_write_lands_in_per_model_tables(self) -> None:
        _seed_chunk("coll", "d1", "h1", "c0")
        write_vectors_to_db(
            collection="coll",
            embeddings=[
                _embedding("d1", "c0", "h1", "m1", vector=[1.0, 2.0]),
                _embedding("d1", "c0", "h1", "m2", vector=[3.0, 4.0, 5.0]),
            ],
            create_index=False,
        )
        tables = get_vector_index_store().list_table_names()
        assert "embeddings_m1" in tables
        assert "embeddings_m2" in tables
        assert len(_read_embedding_rows("m1", "coll", "d1")) == 1
        assert len(_read_embedding_rows("m2", "coll", "d1")) == 1

    def test_intra_batch_dimension_mismatch_raises(self) -> None:
        _seed_chunk("coll", "d1", "h1", "c0")
        _seed_chunk("coll", "d1", "h1", "c1")
        with pytest.raises(VectorValidationError):
            write_vectors_to_db(
                collection="coll",
                embeddings=[
                    _embedding("d1", "c0", "h1", "test_model", vector=[1.0, 2.0]),
                    _embedding("d1", "c1", "h1", "test_model", vector=[1.0, 2.0, 3.0]),
                ],
                create_index=False,
            )


class TestValidateQueryVector:
    def test_accepts_valid_vector(self) -> None:
        validate_query_vector([0.1, 0.2, 0.3])

    def test_accepts_numpy_scalars(self) -> None:
        validate_query_vector([np.float64(0.1), np.int32(2)])  # type: ignore[list-item]

    def test_rejects_non_list(self) -> None:
        with pytest.raises(VectorValidationError, match="must be a list"):
            validate_query_vector((0.1, 0.2))  # type: ignore[arg-type]

    def test_rejects_empty(self) -> None:
        with pytest.raises(VectorValidationError, match="cannot be empty"):
            validate_query_vector([])

    def test_rejects_non_numbers(self) -> None:
        with pytest.raises(VectorValidationError, match="only numbers"):
            validate_query_vector([0.1, "x"])  # type: ignore[list-item]

    def test_rejects_nan_and_inf(self) -> None:
        with pytest.raises(VectorValidationError, match="NaN or infinity"):
            validate_query_vector([0.1, float("nan")])
        with pytest.raises(VectorValidationError, match="NaN or infinity"):
            validate_query_vector([0.1, float("inf")])


class TestCleanupContract:
    def _facade(self):
        return get_kb_coordinator().vector_storage_compatibility

    def test_preview_then_confirm_delete(self) -> None:
        _seed_chunk("coll", "d1", "h1", "c0")
        write_vectors_to_db(
            collection="coll",
            embeddings=[_embedding("d1", "c0", "h1", "test_model", vector=[1.0, 2.0])],
            create_index=False,
        )

        preview = self._facade().cleanup_vectors_for_document(
            collection="coll",
            doc_id="d1",
            is_admin=True,
            preview_only=True,
            confirm=False,
        )
        assert preview.status == "planned"
        assert preview.deleted_count == 1
        assert preview.table_counts == {"embeddings_test_model": 1}
        # Preview does not delete.
        assert len(_read_embedding_rows("test_model", "coll", "d1")) == 1

        done = self._facade().cleanup_vectors_for_document(
            collection="coll",
            doc_id="d1",
            is_admin=True,
            preview_only=False,
            confirm=True,
        )
        assert done.status == "complete"
        assert done.deleted_count == 1
        assert _read_embedding_rows("test_model", "coll", "d1") == []

    def test_cleanup_skipped_when_no_embedding_tables(self) -> None:
        result = self._facade().cleanup_vectors_for_document(
            collection="coll",
            doc_id="d1",
            is_admin=True,
            preview_only=True,
            confirm=False,
        )
        assert result.status == "skipped"
        assert result.deleted_count == 0
        assert result.table_counts == {}
