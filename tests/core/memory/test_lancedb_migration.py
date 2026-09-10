"""Regression tests for safe schema migration on the ``add()`` path (792-02)."""

from __future__ import annotations

import json
import shutil
import tempfile

import lancedb  # type: ignore
import pyarrow as pa  # type: ignore
import pytest

from xagent.core.memory.core import MemoryNote
from xagent.core.memory.lancedb import LanceDBMemoryStore
from xagent.core.model.embedding import BaseEmbedding
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table


class MockEmbedding(BaseEmbedding):
    """Deterministic embedding of a configurable dimension."""

    def __init__(self, dim: int = 64, value: float = 0.1):
        self._dimension = dim
        self._value = value

    def encode(self, text, dimension=None, instruct=None):
        if isinstance(text, str):
            return [self._value] * self._dimension
        return [[self._value] * self._dimension for _ in text]

    def get_dimension(self):
        return self._dimension

    @property
    def abilities(self):
        return ["embed"]


class BatchFailEmbedding(BaseEmbedding):
    """Encodes single strings fine but fails on batched (list) input.

    This lets a note be embedded on the write path (so the insert hits a real
    dimension mismatch) while the migration's batched re-embed fails, exercising
    the all-or-nothing abort.
    """

    def __init__(self, dim: int = 128):
        self._dimension = dim

    def encode(self, text, dimension=None, instruct=None):
        if isinstance(text, str):
            return [0.1] * self._dimension
        raise RuntimeError("batched embedding failed")

    def get_dimension(self):
        return self._dimension

    @property
    def abilities(self):
        return ["embed"]


@pytest.fixture
def temp_db_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def _store(temp_db_dir, embedding_model, name="mem"):
    return LanceDBMemoryStore(
        db_dir=temp_db_dir,
        collection_name=name,
        embedding_model=embedding_model,
    )


def test_add_preserves_rows_on_dimension_change(temp_db_dir):
    """Writing at dim A, switching to dim B, then add() keeps A-rows and stores B."""
    store_a = _store(temp_db_dir, MockEmbedding(64))
    added = store_a.add(MemoryNote(content="alpha"))
    assert added.success
    alpha_id = added.memory_id

    # New store over the same table with a different embedding dimension.
    store_b = _store(temp_db_dir, MockEmbedding(128))
    new = store_b.add(MemoryNote(content="beta"))
    assert new.success

    # The dimension-A row survived the migration...
    got_alpha = store_b.get(alpha_id)
    assert got_alpha.success
    assert got_alpha.content.content == "alpha"
    # ...and the new dimension-B row was stored.
    assert store_b.get(new.memory_id).success
    # Both are retrievable via search at the new dimension.
    contents = {n.content for n in store_b.search("beta", k=10)}
    assert {"alpha", "beta"} <= contents


def test_add_backfills_missing_non_vector_column(temp_db_dir):
    """A missing non-vector column is backfilled in place, without a rebuild."""
    store = _store(temp_db_dir, None)  # vector-less store
    assert store.add(MemoryNote(id="x", content="old")).success

    # Simulate a stale table that lost its metadata column.
    conn = store._vector_store.get_raw_connection()
    table = conn.open_table("mem")
    try:
        table.drop_columns(["metadata"])
    finally:
        _safe_close_table(table)

    # add() must backfill metadata in place (no rebuild) and preserve rows.
    assert store.add(MemoryNote(id="y", content="new")).success

    table = conn.open_table("mem")
    try:
        arrow = table.to_arrow()
    finally:
        _safe_close_table(table)
    # The metadata column was backfilled additively, and both rows survived.
    assert "metadata" in arrow.schema.names
    assert set(arrow.column("id").to_pylist()) == {"x", "y"}
    # The fully-formed new row round-trips through get().
    assert store.get("y").success


def test_add_migration_failure_leaves_table_intact(temp_db_dir):
    """If re-embedding fails mid-migration, no data is lost and add() fails."""
    store_a = _store(temp_db_dir, MockEmbedding(64))
    added = store_a.add(MemoryNote(content="alpha"))
    assert added.success
    alpha_id = added.memory_id

    # Switch to a model that fails the batched re-embed at a new dimension.
    store_fail = _store(temp_db_dir, BatchFailEmbedding(128))
    result = store_fail.add(MemoryNote(content="beta"))
    assert not result.success

    # The original row is untouched and still retrievable via the dim-64 store.
    got_alpha = store_a.get(alpha_id)
    assert got_alpha.success
    assert got_alpha.content.content == "alpha"
    # No partial state: only the original row exists.
    assert len(store_a.list_all()) == 1


def test_build_migrated_table_vectorless_when_no_model(temp_db_dir):
    """The rebuild transform produces a vector-less table (target_dim=None)."""
    store = _store(temp_db_dir, None)
    existing = pa.table(
        {
            "id": ["a", "b"],
            "text": ["alpha", "beta"],
            "metadata": ["{}", "{}"],
            "vector": pa.array([[0.1] * 64, [0.1] * 64], pa.list_(pa.float32(), 64)),
        }
    )

    migrated = store._build_migrated_table(existing, target_dim=None)

    assert "vector" not in migrated.schema.names
    assert migrated.num_rows == 2
    assert migrated.column("id").to_pylist() == ["a", "b"]
    assert migrated.column("text").to_pylist() == ["alpha", "beta"]


def test_build_migrated_table_reembeds_at_target_dim(temp_db_dir):
    """The rebuild transform re-embeds all rows at the new dimension."""
    store = _store(temp_db_dir, MockEmbedding(128))
    existing = pa.table(
        {
            "id": ["a", "b"],
            "text": ["alpha", "beta"],
            "metadata": ["{}", "{}"],
            "vector": pa.array([[0.1] * 64, [0.1] * 64], pa.list_(pa.float32(), 64)),
        }
    )

    migrated = store._build_migrated_table(existing, target_dim=128)

    assert migrated.column("vector").type.list_size == 128
    assert migrated.num_rows == 2


def _seed_table_missing_metadata(temp_db_dir, name="mem"):
    """Create a table with id/text/vector but no metadata column."""
    conn = lancedb.connect(temp_db_dir)
    table = conn.create_table(
        name,
        data=pa.table(
            {
                "id": ["a"],
                "text": ["alpha"],
                "vector": pa.array([[0.1] * 64], pa.list_(pa.float32(), 64)),
            }
        ),
    )
    _safe_close_table(table)


def test_init_backfills_missing_column_without_wipe(temp_db_dir):
    """Store init migrates a table missing a required column, preserving rows."""
    _seed_table_missing_metadata(temp_db_dir)

    # Constructing the store runs _ensure_table_schema, which must migrate.
    store = _store(temp_db_dir, MockEmbedding(64))

    conn = store._vector_store.get_raw_connection()
    table = conn.open_table("mem")
    try:
        arrow = table.to_arrow()
    finally:
        _safe_close_table(table)
    assert "metadata" in arrow.schema.names
    assert arrow.column("id").to_pylist() == ["a"]


def test_init_does_not_wipe_on_dimension_change(temp_db_dir):
    """Constructing a store over a different-dimension table preserves rows."""
    store_a = _store(temp_db_dir, MockEmbedding(64))
    added = store_a.add(MemoryNote(content="alpha"))
    assert added.success

    # A store at a different embedding dimension must not wipe on init; the
    # dimension mismatch is migrated lazily on the add() path instead.
    store_b = _store(temp_db_dir, MockEmbedding(128))
    conn = store_b._vector_store.get_raw_connection()
    table = conn.open_table("mem")
    try:
        ids = table.to_arrow().column("id").to_pylist()
    finally:
        _safe_close_table(table)
    assert added.memory_id in ids


def test_init_migration_failure_leaves_table_intact(temp_db_dir):
    """If init migration fails, the original table is left intact (no wipe)."""
    _seed_table_missing_metadata(temp_db_dir)

    # Missing metadata + a vector-dimension change whose batched re-embed fails
    # forces a rebuild that aborts; init must surface the error, not wipe.
    with pytest.raises(Exception):
        _store(temp_db_dir, BatchFailEmbedding(128))

    conn = lancedb.connect(temp_db_dir)
    table = conn.open_table("mem")
    try:
        arrow = table.to_arrow()
    finally:
        _safe_close_table(table)
    assert arrow.column("id").to_pylist() == ["a"]
    # The vector column was not rebuilt; the table is untouched.
    assert "vector" in arrow.schema.names
    assert "metadata" not in arrow.schema.names


def test_add_record_without_embedding_into_vector_table(temp_db_dir):
    """A note with no embedding stored into a vector table keeps a null vector.

    Exercises the `_insert_record` case where the record lacks a vector but the
    table has a vector column: LanceDB accepts a null vector, the row persists,
    and it stays retrievable (no migration, no data loss)."""
    store = _store(temp_db_dir, MockEmbedding(64))
    assert store.add(MemoryNote(id="withvec", content="alpha")).success

    # Whitespace-only content yields no embedding, so the record has no vector.
    assert store.add(MemoryNote(id="novec", content="   ")).success

    conn = store._vector_store.get_raw_connection()
    table = conn.open_table("mem")
    try:
        ids = set(table.to_arrow().column("id").to_pylist())
    finally:
        _safe_close_table(table)
    assert {"withvec", "novec"} <= ids
    # The vector-less row still round-trips through get().
    assert store.get("novec").success


def test_add_scope_columns_preserves_arrow_metadata(temp_db_dir):
    """_add_scope_columns keeps schema/field metadata and back-fills scope columns.

    Builds an Arrow table whose schema carries schema-level metadata and whose
    ``id`` field carries field metadata, then runs the additive migration
    directly. The old ``pa.table(columns)`` reconstruction dropped both, so
    the metadata assertions below fail on the pre-fix implementation.
    """
    store = _store(temp_db_dir, None)
    schema_metadata = {b"custom-key": b"custom-value"}
    id_field_metadata = {b"field-key": b"field-value"}
    fields = [
        pa.field("id", pa.string(), metadata=id_field_metadata),
        pa.field("text", pa.string()),
        pa.field("metadata", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 2)),
    ]
    schema = pa.schema(fields, metadata=schema_metadata)
    metadata_row = json.dumps({"user_id": 7, "execution_scope_d1": "x"})
    existing = pa.table(
        {
            "id": pa.array(["a", "b"], pa.string()),
            "text": pa.array(["alpha", "beta"], pa.string()),
            "metadata": pa.array([metadata_row, "{}"], pa.string()),
            "vector": pa.array(
                [[0.1, 0.2], [0.3, 0.4]], pa.list_(pa.float32(), 2)
            ),
        },
        schema=schema,
    )

    migrated = store._add_scope_columns(existing)

    # Schema- and field-level metadata survive the additive migration.
    assert migrated.schema.metadata == existing.schema.metadata
    assert migrated.schema.field("id").metadata == existing.schema.field(
        "id"
    ).metadata
    # Existing column data is unchanged; the vector column is preserved as-is.
    assert migrated.column("id").to_pylist() == ["a", "b"]
    assert migrated.column("text").to_pylist() == ["alpha", "beta"]
    assert migrated.column("metadata").to_pylist() == [metadata_row, "{}"]
    assert migrated.column("vector").to_pylist() == existing.column(
        "vector"
    ).to_pylist()
    # The derived scope columns are present with correct types and values.
    assert "user_id" in migrated.schema.names
    assert "scope_dims" in migrated.schema.names
    assert migrated.schema.field("user_id").type == pa.int64()
    assert migrated.schema.field("scope_dims").type == pa.list_(pa.string())
    assert migrated.column("user_id").to_pylist() == [7, None]
    assert migrated.column("scope_dims").to_pylist() == [["d1=x"], []]
    # Column order: existing columns in order, then user_id, then scope_dims.
    assert migrated.schema.names == [
        "id",
        "text",
        "metadata",
        "vector",
        "user_id",
        "scope_dims",
    ]
