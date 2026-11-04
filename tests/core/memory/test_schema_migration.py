"""Unit tests for the safe memory schema-migration primitive and classifier (792-01)."""

from __future__ import annotations

import shutil
import tempfile

import lancedb  # type: ignore
import pyarrow as pa  # type: ignore
import pytest

from xagent.core.memory.schema_migration import (
    MemoryMismatchKind,
    classify_memory_schema_mismatch,
    migrate_table_swap,
)
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table


@pytest.fixture
def temp_db_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def conn(temp_db_dir):
    return lancedb.connect(temp_db_dir)


def _non_vector_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("metadata", pa.string()),
        ]
    )


def _vector_schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("metadata", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), list_size=dim)),
        ]
    )


# --------------------------------------------------------------------------
# Classifier
# --------------------------------------------------------------------------


def test_classify_no_mismatch_vectorless():
    """No vector column and no vector expected -> no mismatch."""
    result = classify_memory_schema_mismatch(_non_vector_schema(), expected_dim=None)
    assert result.kind is MemoryMismatchKind.NONE


def test_classify_no_mismatch_matching_dim():
    """Vector width matches expected dimension -> no mismatch."""
    result = classify_memory_schema_mismatch(_vector_schema(64), expected_dim=64)
    assert result.kind is MemoryMismatchKind.NONE
    assert result.current_dim == 64


def test_classify_missing_non_vector_column():
    """A missing id/text/metadata column with no vector mismatch -> backfill."""
    schema = pa.schema([pa.field("id", pa.string()), pa.field("text", pa.string())])
    result = classify_memory_schema_mismatch(schema, expected_dim=None)
    assert result.kind is MemoryMismatchKind.MISSING_NON_VECTOR_COLUMN
    assert result.missing_columns == ("metadata",)


def test_classify_vector_dimension_change():
    """Table at one width, store now produces another -> rebuild."""
    result = classify_memory_schema_mismatch(_vector_schema(64), expected_dim=128)
    assert result.kind is MemoryMismatchKind.VECTOR_REBUILD
    assert result.current_dim == 64


def test_classify_vector_presence_added():
    """Table has no vector but store now produces one -> rebuild."""
    result = classify_memory_schema_mismatch(_non_vector_schema(), expected_dim=64)
    assert result.kind is MemoryMismatchKind.VECTOR_REBUILD
    assert result.current_dim is None


def test_classify_vector_presence_removed():
    """Table has a vector but store no longer produces one -> rebuild (drop vector)."""
    result = classify_memory_schema_mismatch(_vector_schema(64), expected_dim=None)
    assert result.kind is MemoryMismatchKind.VECTOR_REBUILD
    assert result.current_dim == 64


def test_classify_vector_rebuild_takes_precedence_over_missing_column():
    """When both a column is missing and the vector mismatches, rebuild wins."""
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), list_size=64)),
        ]
    )
    result = classify_memory_schema_mismatch(schema, expected_dim=128)
    assert result.kind is MemoryMismatchKind.VECTOR_REBUILD
    assert result.missing_columns == ("metadata",)


# --------------------------------------------------------------------------
# Migration primitive
# --------------------------------------------------------------------------


def _seed_table(conn, name="mem"):
    data = pa.table(
        {
            "id": ["a", "b"],
            "text": ["hello", "world"],
            "metadata": ["{}", "{}"],
        }
    )
    tbl = conn.create_table(name, data=data)
    _safe_close_table(tbl)
    return name


def test_swap_replaces_table_contents(conn):
    """A successful transform swaps the table to the migrated data."""
    name = _seed_table(conn)

    def transform(existing: pa.Table) -> pa.Table:
        # Migrate by appending a new row and a new column.
        migrated = existing.append_column("extra", pa.array(["x"] * existing.num_rows))
        new_row = pa.table(
            {"id": ["c"], "text": ["new"], "metadata": ["{}"], "extra": ["x"]}
        )
        return pa.concat_tables([migrated, new_row])

    migrate_table_swap(conn, name, transform)

    tbl = conn.open_table(name)
    try:
        result = tbl.to_arrow()
    finally:
        _safe_close_table(tbl)
    assert result.num_rows == 3
    assert "extra" in result.schema.names
    assert set(result.column("id").to_pylist()) == {"a", "b", "c"}


def test_transform_failure_leaves_table_intact(conn):
    """If the transform raises, the original rows and schema are untouched."""
    name = _seed_table(conn)

    def failing_transform(existing: pa.Table) -> pa.Table:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        migrate_table_swap(conn, name, failing_transform)

    tbl = conn.open_table(name)
    try:
        result = tbl.to_arrow()
    finally:
        _safe_close_table(tbl)
    assert result.num_rows == 2
    assert result.schema.names == ["id", "text", "metadata"]
    assert set(result.column("id").to_pylist()) == {"a", "b"}


def test_transform_must_return_arrow_table(conn):
    """A transform returning a non-Arrow value aborts without touching the table."""
    name = _seed_table(conn)

    def bad_transform(existing: pa.Table):
        return [{"id": "c"}]

    with pytest.raises(TypeError):
        migrate_table_swap(conn, name, bad_transform)

    tbl = conn.open_table(name)
    try:
        result = tbl.to_arrow()
    finally:
        _safe_close_table(tbl)
    assert result.num_rows == 2


def test_swap_supports_empty_migrated_table(conn):
    """Swapping to an empty (but well-typed) table preserves schema, drops rows."""
    name = _seed_table(conn)

    def empty_transform(existing: pa.Table) -> pa.Table:
        return existing.schema.empty_table()

    migrate_table_swap(conn, name, empty_transform)

    tbl = conn.open_table(name)
    try:
        result = tbl.to_arrow()
    finally:
        _safe_close_table(tbl)
    assert result.num_rows == 0
    assert result.schema.names == ["id", "text", "metadata"]
