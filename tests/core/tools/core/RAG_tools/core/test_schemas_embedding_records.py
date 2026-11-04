"""Tests for the lean handle-level embedding record schemas (#510).

``EmbeddingRecordDetail`` / ``EmbeddingRecordSnapshot`` are the semantic types
the collection handle uses for embedding-row snapshot and restore. They must map
losslessly back to the raw ``embeddings_{model_tag}`` table-row dict shape (the
column set is locked here against the LanceDB embeddings schema), and the
snapshot must expose per-model-tag grouping so restore can route each row to its
own embeddings table.
"""

from datetime import datetime, timezone

from xagent.core.tools.core.RAG_tools.core.schemas import (
    EmbeddingRecordDetail,
    EmbeddingRecordSnapshot,
)
from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import to_model_tag

EMBEDDING_ROW = {
    "collection": "coll",
    "doc_id": "doc-1",
    "chunk_id": "chunk-0",
    "parse_hash": "p" * 16,
    "model": "test_model",
    "vector": [0.1, 0.2, 0.3],
    "vector_dimension": 3,
    "text": "hello world",
    "chunk_hash": "c" * 16,
    "created_at": datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
    "metadata": '{"layout_type": "text"}',
    "user_id": 7,
}

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


class TestEmbeddingRecordDetail:
    def test_from_row_to_legacy_dict_round_trip(self) -> None:
        detail = EmbeddingRecordDetail.from_row(EMBEDDING_ROW)
        assert detail.to_legacy_dict() == EMBEDDING_ROW

    def test_column_set_matches_embeddings_schema(self) -> None:
        # Locks the full embeddings_{model_tag} column set (lossless oracle).
        assert (
            set(EmbeddingRecordDetail.from_row(EMBEDDING_ROW).to_legacy_dict().keys())
            == EMBEDDING_COLUMNS
        )

    def test_vector_preserved_as_float_list(self) -> None:
        detail = EmbeddingRecordDetail.from_row(EMBEDDING_ROW)
        assert detail.vector == [0.1, 0.2, 0.3]
        assert all(isinstance(v, float) for v in detail.vector)

    def test_from_row_normalizes_nan_and_coerces_ints(self) -> None:
        nan = float("nan")
        row = {
            **EMBEDDING_ROW,
            "vector_dimension": nan,
            "metadata": nan,
            "user_id": nan,
        }
        detail = EmbeddingRecordDetail.from_row(row)
        assert detail.vector_dimension is None
        assert detail.metadata is None
        assert detail.user_id is None

    def test_legacy_row_missing_user_id(self) -> None:
        row = {k: v for k, v in EMBEDDING_ROW.items() if k != "user_id"}
        detail = EmbeddingRecordDetail.from_row(row)
        assert detail.user_id is None

    def test_user_id_coerced_to_plain_int(self) -> None:
        detail = EmbeddingRecordDetail.from_row({**EMBEDDING_ROW, "user_id": 42})
        assert detail.user_id == 42
        assert isinstance(detail.user_id, int)


class TestEmbeddingRecordSnapshot:
    def test_round_trips_ordered_rows(self) -> None:
        rows = [
            {**EMBEDDING_ROW, "chunk_id": "chunk-0"},
            {**EMBEDDING_ROW, "chunk_id": "chunk-1"},
        ]
        snapshot = EmbeddingRecordSnapshot.from_rows(rows)
        assert [r.chunk_id for r in snapshot.records] == ["chunk-0", "chunk-1"]
        assert snapshot.to_legacy_dicts() == rows

    def test_empty_snapshot(self) -> None:
        snapshot = EmbeddingRecordSnapshot.from_rows([])
        assert snapshot.records == []
        assert snapshot.to_legacy_dicts() == []

    def test_group_by_model_tag_routes_rows(self) -> None:
        rows = [
            {**EMBEDDING_ROW, "chunk_id": "c0", "model": "openai/text-embedding-3"},
            {**EMBEDDING_ROW, "chunk_id": "c1", "model": "openai/text-embedding-3"},
            {**EMBEDDING_ROW, "chunk_id": "c2", "model": "bge-large"},
        ]
        snapshot = EmbeddingRecordSnapshot.from_rows(rows)
        grouped = snapshot.group_by_model_tag()

        assert set(grouped.keys()) == {
            to_model_tag("openai/text-embedding-3"),
            to_model_tag("bge-large"),
        }
        openai_rows = grouped[to_model_tag("openai/text-embedding-3")]
        assert [r["chunk_id"] for r in openai_rows] == ["c0", "c1"]
        # Each grouped value is the legacy raw-row dict shape ready for upsert.
        assert set(openai_rows[0].keys()) == EMBEDDING_COLUMNS
