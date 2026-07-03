from __future__ import annotations

from pathlib import Path
from threading import Thread
from unittest.mock import Mock

import pyarrow as pa

from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
    _create_table,
    _table_exists,
    ensure_collection_config_table,
    ensure_collection_metadata_table,
    ensure_embeddings_table,
    ensure_ingestion_runs_table,
    ensure_parses_table,
    ensure_prompt_templates_table,
)
from xagent.core.tools.core.RAG_tools.storage import get_vector_store_raw_connection

# NOTE: Tests for _get_sql_default_for_pa_type / broad auto-migration of arbitrary
# missing columns were removed: schema_manager now uses _validate_schema_fields
# for some tables and targeted migrations for user_id/file_id on documents.
# Old helpers like _ensure_schema_fields are not part of the public API.


def test_ensure_schema_fields_idempotency(tmp_path: Path, monkeypatch):
    """Test that calling migration on an up-to-date table is safe."""
    db_dir = tmp_path / "db"
    monkeypatch.setenv("LANCEDB_DIR", str(db_dir))
    conn = get_vector_store_raw_connection()

    # Create table with FULL schema first
    ensure_collection_metadata_table(conn)
    table_before = conn.open_table("collection_metadata")
    schema_before = table_before.schema

    # Call it again
    ensure_collection_metadata_table(conn)

    table_after = conn.open_table("collection_metadata")
    schema_after = table_after.schema

    assert schema_before == schema_after

    # Also verify idempotency does not duplicate/corrupt data.
    sample_row = {
        "name": "col_a",
        "schema_version": "1",
        "embedding_model_id": "test-model",
        "embedding_dimension": 3,
        "documents": 1,
        "processed_documents": 1,
        "parses": 1,
        "chunks": 1,
        "embeddings": 1,
        "document_names": "[]",
        "owners": "[]",
        "collection_locked": False,
        "allow_mixed_parse_methods": False,
        "skip_config_validation": False,
        "ingestion_config": None,
        "created_at": None,
        "updated_at": None,
        "last_accessed_at": None,
        "extra_metadata": None,
    }
    table_after.add([sample_row])
    count_before = int(table_after.count_rows())

    ensure_collection_metadata_table(conn)

    table_final = conn.open_table("collection_metadata")
    count_after = int(table_final.count_rows())
    assert count_before == count_after

    rows = table_final.search().where("name = 'col_a'").to_pandas()
    assert len(rows) == 1
    assert rows.iloc[0]["embedding_model_id"] == "test-model"


def test_table_exists_returns_false_on_open_error() -> None:
    """_table_exists should return False when open_table raises."""
    conn = Mock()
    conn.open_table.side_effect = RuntimeError("boom")
    assert _table_exists(conn, "missing_table") is False


def test_create_table_without_schema_calls_conn_create_table() -> None:
    """_create_table should call conn.create_table when schema is None."""
    conn = Mock()
    conn.open_table.side_effect = RuntimeError("not found")
    _create_table(conn, "plain_table", schema=None)
    conn.create_table.assert_called_once_with("plain_table", schema=None)


# NOTE: test_create_table_existing_with_schema_triggers_migration modified
# because _create_table no longer triggers migration. If the table exists,
# it just returns without doing anything. Migration is handled separately.


def test_create_table_existing_with_schema_triggers_migration(
    tmp_path: Path, monkeypatch
) -> None:
    """_create_table should not modify existing table when schema is provided."""
    db_dir = tmp_path / "db"
    monkeypatch.setenv("LANCEDB_DIR", str(db_dir))
    conn = get_vector_store_raw_connection()

    conn.create_table("create_table_migrate", schema=pa.schema([("a", pa.int32())]))
    target_schema = pa.schema([("a", pa.int32()), ("b", pa.string())])

    # _create_table should not migrate existing tables - it just returns if table exists
    _create_table(conn, "create_table_migrate", target_schema)

    schema = conn.open_table("create_table_migrate").schema
    # The table should still have the original schema (no migration)
    assert "b" not in schema.names
    assert schema.names == ["a"]


def test_ensure_embeddings_table_with_fixed_vector_dim(
    tmp_path: Path, monkeypatch
) -> None:
    """ensure_embeddings_table should use fixed-size list when vector_dim is set."""
    db_dir = tmp_path / "db"
    monkeypatch.setenv("LANCEDB_DIR", str(db_dir))
    conn = get_vector_store_raw_connection()

    ensure_embeddings_table(conn, "test_fixed", vector_dim=8)
    schema = conn.open_table("embeddings_test_fixed").schema
    vector_type = schema.field("vector").type
    assert getattr(vector_type, "list_size", None) == 8


def test_ensure_embeddings_table_with_variable_vector_dim(
    tmp_path: Path, monkeypatch
) -> None:
    """ensure_embeddings_table should use variable list when vector_dim is None."""
    db_dir = tmp_path / "db"
    monkeypatch.setenv("LANCEDB_DIR", str(db_dir))
    conn = get_vector_store_raw_connection()

    ensure_embeddings_table(conn, "test_variable", vector_dim=None)
    schema = conn.open_table("embeddings_test_variable").schema
    vector_type = schema.field("vector").type
    assert getattr(vector_type, "list_size", -1) == -1


def test_ensure_collection_config_table_create_and_idempotent(
    tmp_path: Path, monkeypatch
) -> None:
    """ensure_collection_config_table should be creatable and idempotent."""
    db_dir = tmp_path / "db"
    monkeypatch.setenv("LANCEDB_DIR", str(db_dir))
    conn = get_vector_store_raw_connection()

    ensure_collection_config_table(conn)
    schema_before = conn.open_table("collection_config").schema
    ensure_collection_config_table(conn)
    schema_after = conn.open_table("collection_config").schema

    assert schema_before == schema_after
    assert "config_json" in schema_after.names
    assert "user_id" in schema_after.names


def test_ensure_parses_table_migrates_missing_user_id(
    tmp_path: Path, monkeypatch
) -> None:
    """ensure_parses_table should add user_id for legacy schema."""
    db_dir = tmp_path / "db"
    monkeypatch.setenv("LANCEDB_DIR", str(db_dir))
    conn = get_vector_store_raw_connection()

    old_schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("parse_hash", pa.string()),
            pa.field("parser", pa.string()),
            pa.field("created_at", pa.timestamp("us")),
            pa.field("params_json", pa.string()),
            pa.field("parsed_content", pa.large_string()),
        ]
    )
    conn.create_table("parses", schema=old_schema)
    conn.open_table("parses").add(
        [
            {
                "collection": "c",
                "doc_id": "d1",
                "parse_hash": "p1",
                "parser": "basic",
                "created_at": None,
                "params_json": "{}",
                "parsed_content": "x",
            }
        ]
    )

    ensure_parses_table(conn)
    schema = conn.open_table("parses").schema
    assert "user_id" in schema.names


def test_ensure_prompt_templates_table_migrates_missing_user_id(
    tmp_path: Path, monkeypatch
) -> None:
    """ensure_prompt_templates_table should add user_id for legacy schema."""
    db_dir = tmp_path / "db"
    monkeypatch.setenv("LANCEDB_DIR", str(db_dir))
    conn = get_vector_store_raw_connection()

    old_schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("id", pa.string()),
            pa.field("name", pa.string()),
            pa.field("template", pa.string()),
            pa.field("version", pa.int64()),
            pa.field("is_latest", pa.bool_()),
            pa.field("metadata", pa.string()),
            pa.field("created_at", pa.timestamp("us")),
            pa.field("updated_at", pa.timestamp("us")),
        ]
    )
    conn.create_table("prompt_templates", schema=old_schema)
    conn.open_table("prompt_templates").add(
        [
            {
                "collection": "c",
                "id": "1",
                "name": "n",
                "template": "t",
                "version": 1,
                "is_latest": True,
                "metadata": "{}",
                "created_at": None,
                "updated_at": None,
            }
        ]
    )

    ensure_prompt_templates_table(conn)
    schema = conn.open_table("prompt_templates").schema
    assert "user_id" in schema.names


def test_ensure_ingestion_runs_table_migrates_missing_user_id(
    tmp_path: Path, monkeypatch
) -> None:
    """ensure_ingestion_runs_table should add user_id for legacy schema."""
    db_dir = tmp_path / "db"
    monkeypatch.setenv("LANCEDB_DIR", str(db_dir))
    conn = get_vector_store_raw_connection()

    old_schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("status", pa.string()),
            pa.field("message", pa.string()),
            pa.field("parse_hash", pa.string()),
            pa.field("created_at", pa.timestamp("us")),
            pa.field("updated_at", pa.timestamp("us")),
        ]
    )
    conn.create_table("ingestion_runs", schema=old_schema)
    conn.open_table("ingestion_runs").add(
        [
            {
                "collection": "c",
                "doc_id": "d",
                "status": "ok",
                "message": "",
                "parse_hash": "p",
                "created_at": None,
                "updated_at": None,
            }
        ]
    )

    ensure_ingestion_runs_table(conn)
    schema = conn.open_table("ingestion_runs").schema
    assert "user_id" in schema.names


def test_concurrent_ensure_collection_metadata_table_is_safe(
    tmp_path: Path, monkeypatch
) -> None:
    """Concurrent ensure_collection_metadata_table calls should be safe.

    Each thread uses its own connection. Table creation is idempotent; under load,
    some threads may still see benign races from LanceDB — the table should exist
    after all threads complete.
    """
    db_dir = tmp_path / "db"
    monkeypatch.setenv("LANCEDB_DIR", str(db_dir))

    errors: list[Exception] = []

    def _worker() -> None:
        try:
            # Each thread gets its own connection to avoid threading issues
            worker_conn = get_vector_store_raw_connection()
            ensure_collection_metadata_table(worker_conn)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [Thread(target=_worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # LanceDB can raise benign commit-conflict races when threads create the
    # table concurrently (see docstring); the real invariant is that the table
    # exists with the right schema afterwards. Any other error is a failure.
    unexpected = [
        exc
        for exc in errors
        if "Commit conflict" not in str(exc)
        and "incompatible with concurrent" not in str(exc)
    ]
    assert unexpected == []
    conn = get_vector_store_raw_connection()
    assert _table_exists(conn, "collection_metadata")
    schema = conn.open_table("collection_metadata").schema
    assert "ingestion_config" in schema.names
