"""Schema tests for generation_id and active_generations (issue #438 PR1)."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import to_model_tag
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
    ensure_active_generations_table,
    ensure_chunks_table,
    ensure_embeddings_table,
)
from xagent.core.tools.core.RAG_tools.storage import get_vector_store_raw_connection


def _schema_field_names(conn, table_name: str) -> set[str]:
    table = conn.open_table(table_name)
    try:
        return set(table.schema.names)
    finally:
        if hasattr(table, "close"):
            table.close()


def test_chunks_table_includes_generation_id(tmp_path: Path, monkeypatch) -> None:
    """New and migrated chunks tables must expose generation_id."""
    monkeypatch.setenv("LANCEDB_DIR", str(tmp_path / "db"))
    conn = get_vector_store_raw_connection()
    ensure_chunks_table(conn)
    assert "generation_id" in _schema_field_names(conn, "chunks")


def test_embeddings_table_includes_generation_id_and_config_hash(
    tmp_path: Path, monkeypatch
) -> None:
    """Embeddings tables must expose generation_id and config_hash."""
    monkeypatch.setenv("LANCEDB_DIR", str(tmp_path / "db"))
    conn = get_vector_store_raw_connection()
    model_tag = to_model_tag("BAAI/bge-large-zh-v1.5")
    ensure_embeddings_table(conn, model_tag, vector_dim=4)
    table_name = f"embeddings_{model_tag}"
    names = _schema_field_names(conn, table_name)
    assert "generation_id" in names
    assert "config_hash" in names


def test_active_generations_table_schema(tmp_path: Path, monkeypatch) -> None:
    """active_generations table must include tenant-scoped pointer fields."""
    monkeypatch.setenv("LANCEDB_DIR", str(tmp_path / "db"))
    conn = get_vector_store_raw_connection()
    ensure_active_generations_table(conn)
    names = _schema_field_names(conn, "active_generations")
    assert {
        "collection",
        "doc_id",
        "parse_hash",
        "user_id",
        "model_tag",
        "generation_id",
        "config_hash",
        "created_at",
        "updated_at",
        "published_at",
        "operator",
    }.issubset(names)


def test_chunks_table_adds_generation_id_column_for_legacy_schema(
    tmp_path: Path, monkeypatch
) -> None:
    """Existing chunks tables without generation_id receive a nullable column."""
    monkeypatch.setenv("LANCEDB_DIR", str(tmp_path / "db"))
    conn = get_vector_store_raw_connection()
    legacy_schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("parse_hash", pa.string()),
            pa.field("chunk_id", pa.string()),
            pa.field("index", pa.int32()),
            pa.field("text", pa.large_string()),
            pa.field("page_number", pa.int32()),
            pa.field("section", pa.string()),
            pa.field("anchor", pa.string()),
            pa.field("json_path", pa.string()),
            pa.field("chunk_hash", pa.string()),
            pa.field("config_hash", pa.string()),
            pa.field("created_at", pa.timestamp("us")),
            pa.field("metadata", pa.string()),
            pa.field("user_id", pa.int64()),
        ]
    )
    conn.create_table("chunks", schema=legacy_schema)
    ensure_chunks_table(conn)
    assert "generation_id" in _schema_field_names(conn, "chunks")
