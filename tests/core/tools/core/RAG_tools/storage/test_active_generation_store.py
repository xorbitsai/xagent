"""Tests for ActiveGenerationStore and generation schema columns (issue #438 PR1)."""

from __future__ import annotations

import pytest

from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import to_model_tag
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
    ensure_active_generations_table,
    ensure_chunks_table,
    ensure_embeddings_table,
)
from xagent.core.tools.core.RAG_tools.storage import get_vector_store_raw_connection
from xagent.core.tools.core.RAG_tools.storage.factory import (
    get_active_generation_store,
    reset_rag_storage_for_tests,
)
from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
    LanceDBActiveGenerationStore,
    LanceDBVectorIndexStore,
)


@pytest.fixture
def lancedb_conn(tmp_path, monkeypatch):
    """Isolated LanceDB connection for integration tests."""
    monkeypatch.setenv("LANCEDB_DIR", str(tmp_path / "lancedb"))
    reset_rag_storage_for_tests()
    conn = get_vector_store_raw_connection()
    yield conn
    reset_rag_storage_for_tests()


def test_get_active_generation_store_returns_lancedb_impl() -> None:
    store = get_active_generation_store()
    assert store.__class__.__name__ == "LanceDBActiveGenerationStore"


def test_active_generation_publish_and_get_roundtrip(lancedb_conn) -> None:
    ensure_active_generations_table(lancedb_conn)
    store = get_active_generation_store()

    store.publish_active_generation(
        collection="col_a",
        doc_id="doc_1",
        parse_hash="parse_abc",
        user_id=42,
        model_tag="text_embedding_v4",
        generation_id="gen_new",
        config_hash="cfg_1",
        operator="test",
    )

    pointer = store.get_active_generation(
        collection="col_a",
        doc_id="doc_1",
        parse_hash="parse_abc",
        user_id=42,
        model_tag="text_embedding_v4",
    )
    assert pointer is not None
    assert pointer["generation_id"] == "gen_new"
    assert pointer["config_hash"] == "cfg_1"
    assert pointer["user_id"] == 42


def test_active_generation_scoped_by_user_id(lancedb_conn) -> None:
    ensure_active_generations_table(lancedb_conn)
    store = get_active_generation_store()

    store.publish_active_generation(
        collection="col_a",
        doc_id="doc_1",
        parse_hash="parse_abc",
        user_id=1,
        model_tag="",
        generation_id="gen_user_1",
        config_hash="cfg_1",
    )
    store.publish_active_generation(
        collection="col_a",
        doc_id="doc_1",
        parse_hash="parse_abc",
        user_id=2,
        model_tag="",
        generation_id="gen_user_2",
        config_hash="cfg_2",
    )

    p1 = store.get_active_generation(
        collection="col_a",
        doc_id="doc_1",
        parse_hash="parse_abc",
        user_id=1,
        model_tag="",
    )
    p2 = store.get_active_generation(
        collection="col_a",
        doc_id="doc_1",
        parse_hash="parse_abc",
        user_id=2,
        model_tag="",
    )
    assert p1 is not None and p1["generation_id"] == "gen_user_1"
    assert p2 is not None and p2["generation_id"] == "gen_user_2"


def test_active_generation_legacy_none_user_scope_republishes_one_row(
    lancedb_conn,
) -> None:
    """Legacy user_id=None scope should upsert one pointer and remain queryable."""
    ensure_active_generations_table(lancedb_conn)
    store = get_active_generation_store()

    scope = dict(
        collection="col_a",
        doc_id="doc_1",
        parse_hash="parse_abc",
        user_id=None,
        model_tag="text_embedding_v4",
    )

    store.publish_active_generation(
        **scope,
        generation_id="gen_v1",
        config_hash="cfg_v1",
        operator="test",
    )
    store.publish_active_generation(
        **scope,
        generation_id="gen_v2",
        config_hash="cfg_v2",
        operator="test",
    )
    store.publish_active_generation(
        collection="col_a",
        doc_id="doc_1",
        parse_hash="parse_abc",
        user_id=42,
        model_tag="text_embedding_v4",
        generation_id="gen_user_42",
        config_hash="cfg_user_42",
        operator="test",
    )

    pointer = store.get_active_generation(**scope)
    assert pointer is not None
    assert pointer["user_id"] is None
    assert pointer["generation_id"] == "gen_v2"
    assert pointer["config_hash"] == "cfg_v2"

    rows = store.list_active_generations(collection="col_a", user_id=None)
    assert len(rows) == 1
    assert rows[0]["user_id"] is None
    assert rows[0]["generation_id"] == "gen_v2"


def test_active_generation_republish_preserves_created_at(lancedb_conn) -> None:
    """Republishing the same scope must keep created_at while bumping the rest."""
    ensure_active_generations_table(lancedb_conn)
    store = get_active_generation_store()

    scope = dict(
        collection="col_a",
        doc_id="doc_1",
        parse_hash="parse_abc",
        user_id=7,
        model_tag="text_embedding_v4",
    )

    store.publish_active_generation(
        **scope,
        generation_id="gen_v1",
        config_hash="cfg_v1",
        operator="test",
    )
    first = store.get_active_generation(**scope)
    assert first is not None
    original_created_at = first["created_at"]

    store.publish_active_generation(
        **scope,
        generation_id="gen_v2",
        config_hash="cfg_v2",
        operator="test",
    )
    second = store.get_active_generation(**scope)

    assert second is not None
    assert second["generation_id"] == "gen_v2"
    assert second["config_hash"] == "cfg_v2"
    # created_at must be carried over from the first publish even though
    # publish does not perform a prior get_active_generation() round-trip.
    assert second["created_at"] == original_created_at
    assert second["updated_at"] >= original_created_at
    assert second["published_at"] >= original_created_at


def test_active_generation_ensure_table_runs_once(monkeypatch, lancedb_conn) -> None:
    """Publish/get hot paths must reuse the cached ensure_table result."""
    ensure_active_generations_table(lancedb_conn)

    from xagent.core.tools.core.RAG_tools.LanceDB import schema_manager as sm

    call_count = {"n": 0}
    real_ensure = sm.ensure_active_generations_table

    def counting_ensure(conn):
        call_count["n"] += 1
        return real_ensure(conn)

    monkeypatch.setattr(sm, "ensure_active_generations_table", counting_ensure)

    store = LanceDBActiveGenerationStore()
    scope = dict(
        collection="col_a",
        doc_id="doc_1",
        parse_hash="parse_abc",
        user_id=11,
        model_tag="",
    )
    store.publish_active_generation(
        **scope,
        generation_id="gen_1",
        config_hash="cfg",
    )
    store.publish_active_generation(
        **scope,
        generation_id="gen_2",
        config_hash="cfg",
    )
    store.get_active_generation(**scope)
    store.list_active_generations(collection="col_a")

    assert call_count["n"] == 1


def test_chunks_table_persists_generation_id_column(lancedb_conn) -> None:
    """PR1: schema accepts generation_id; merge-key isolation is covered in PR3."""
    ensure_chunks_table(lancedb_conn)
    vector_store = LanceDBVectorIndexStore()
    base = {
        "collection": "col_a",
        "doc_id": "doc_1",
        "parse_hash": "parse_abc",
        "index": 0,
        "text": "hello",
        "config_hash": "cfg_1",
        "chunk_hash": "chash",
        "created_at": None,
        "metadata": None,
        "user_id": 7,
    }
    vector_store.upsert_chunks(
        [
            {**base, "chunk_id": "chunk_a", "generation_id": "gen_a"},
            {**base, "chunk_id": "chunk_b", "generation_id": "gen_b"},
        ]
    )
    table = lancedb_conn.open_table("chunks")
    try:
        rows = table.search().where("doc_id = 'doc_1'").to_arrow().to_pylist()
    finally:
        if hasattr(table, "close"):
            table.close()
    generation_ids = {row["generation_id"] for row in rows}
    assert generation_ids == {"gen_a", "gen_b"}


def test_embeddings_tables_accept_generation_id_and_config_hash(
    lancedb_conn,
) -> None:
    model_tag = to_model_tag("test-model")
    ensure_embeddings_table(lancedb_conn, model_tag, vector_dim=2)
    vector_store = LanceDBVectorIndexStore()
    vector_store.upsert_embeddings(
        "test-model",
        [
            {
                "collection": "col_a",
                "doc_id": "doc_1",
                "parse_hash": "parse_abc",
                "generation_id": "gen_1",
                "chunk_id": "chunk_1",
                "model": "test-model",
                "config_hash": "cfg_1",
                "vector": [0.1, 0.2],
                "vector_dimension": 2,
                "text": "hello",
                "chunk_hash": "chash",
                "created_at": None,
                "metadata": None,
                "user_id": 3,
            }
        ],
    )
    table_name = f"embeddings_{model_tag}"
    table = lancedb_conn.open_table(table_name)
    try:
        row = table.search().where("chunk_id = 'chunk_1'").to_arrow().to_pylist()[0]
    finally:
        if hasattr(table, "close"):
            table.close()
    assert row["generation_id"] == "gen_1"
    assert row["config_hash"] == "cfg_1"
