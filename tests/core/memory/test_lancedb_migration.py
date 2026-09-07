"""Regression tests for safe schema migration on the ``add()`` path (792-02)."""

from __future__ import annotations

import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor

import lancedb  # type: ignore
import pyarrow as pa  # type: ignore
import pytest

from xagent.core.memory.core import MemoryNote
from xagent.core.memory.lancedb import LanceDBMemoryStore
from xagent.core.memory.schema_migration import (
    LEGACY_DASHSCOPE_IDENTITY,
    VectorSpaceCompatibility,
)
from xagent.core.model.embedding import BaseEmbedding
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table


class MockEmbedding(BaseEmbedding):
    """Deterministic embedding of a configurable dimension."""

    def __init__(self, dim: int = 64, value: float = 0.1):
        self._dimension = dim
        self._value = value
        self.inputs = []

    def encode(self, text, dimension=None, instruct=None):
        self.inputs.append(text)
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


def _store(temp_db_dir, embedding_model, name="mem", identity=None, maintain=True):
    return LanceDBMemoryStore(
        db_dir=temp_db_dir,
        collection_name=name,
        embedding_model=embedding_model,
        vector_space_identity=identity,
        run_schema_maintenance=maintain,
    )


def _identity(model="text-embedding-v4", dimension=64):
    return {**LEGACY_DASHSCOPE_IDENTITY, "model": model, "dimension": dimension}


def test_real_legacy_inspection_is_read_only_and_scope_columns_are_optional(
    temp_db_dir,
):
    _seed_table_missing_scope = lancedb.connect(temp_db_dir)
    table = _seed_table_missing_scope.create_table(
        "legacy",
        data=pa.table(
            {
                "id": ["old"],
                "text": ["alpha"],
                "metadata": ["{}"],
                "vector": pa.array([[0.1] * 64], pa.list_(pa.float32(), 64)),
            }
        ),
    )
    _safe_close_table(table)
    embedding = MockEmbedding(64)
    store = _store(temp_db_dir, embedding, "legacy", _identity(), False)

    with ThreadPoolExecutor(max_workers=2) as pool:
        inspections = pool.submit(
            lambda: [store.inspect_vector_space() for _ in range(8)]
        )
        write = pool.submit(store.add, MemoryNote(id="new", content="beta"))
        states, added = inspections.result(), write.result()

    assert states == [VectorSpaceCompatibility.LEGACY_COMPATIBLE] * 8
    assert added.success and embedding.inputs == ["beta"]
    arrow = _seed_table_missing_scope.open_table("legacy").to_arrow()
    assert set(arrow.column("id").to_pylist()) == {"old", "new"}
    assert {"user_id", "scope_dims"}.isdisjoint(arrow.schema.names)


def test_real_persisted_identity_match_and_mismatch(temp_db_dir):
    matching = _store(temp_db_dir, MockEmbedding(64), "identified", _identity(), False)
    mismatch = _store(
        temp_db_dir, MockEmbedding(64), "identified", _identity(model="other"), False
    )
    assert matching.inspect_vector_space() is VectorSpaceCompatibility.MATCHING
    assert mismatch.inspect_vector_space() is VectorSpaceCompatibility.MISMATCHING


def test_add_dimension_change_falls_back_without_replacing_vectors(temp_db_dir):
    """A new vector space writes text-only and leaves existing vectors intact."""
    store_a = _store(temp_db_dir, MockEmbedding(64))
    added = store_a.add(MemoryNote(content="alpha"))
    assert added.success
    alpha_id = added.memory_id

    # New store over the same table with a different embedding dimension.
    store_b = _store(temp_db_dir, MockEmbedding(128))
    new = store_b.add(MemoryNote(content="beta"))
    assert new.success

    # Both rows survive, while the authoritative vector schema remains at A.
    got_alpha = store_b.get(alpha_id)
    assert got_alpha.success
    assert got_alpha.content.content == "alpha"
    assert store_b.get(new.memory_id).success
    table = store_b._vector_store.get_raw_connection().open_table("mem")
    arrow = table.to_arrow()
    assert arrow.schema.field("vector").type.list_size == 64
    assert arrow.column("vector").to_pylist()[1] is None


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

    # Maintenance is explicit; request-time add never rewrites the table.
    store.maintain_schema()
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


def test_dimension_mismatch_never_attempts_batch_reembedding(temp_db_dir):
    """A mismatching requester cannot re-embed the shared table."""
    store_a = _store(temp_db_dir, MockEmbedding(64))
    added = store_a.add(MemoryNote(content="alpha"))
    assert added.success
    alpha_id = added.memory_id

    # Switch to a model that fails the batched re-embed at a new dimension.
    store_fail = _store(temp_db_dir, BatchFailEmbedding(128))
    result = store_fail.add(MemoryNote(content="beta"))
    assert result.success

    # The original row is untouched and still retrievable via the dim-64 store.
    got_alpha = store_a.get(alpha_id)
    assert got_alpha.success
    assert got_alpha.content.content == "alpha"
    assert len(store_a.list_all()) == 2


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
    store.maintain_schema()

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


def test_init_maintenance_never_rebuilds_mismatching_vectors(temp_db_dir):
    """Ordinary maintenance backfills columns without vector migration."""
    _seed_table_missing_metadata(temp_db_dir)

    store = _store(temp_db_dir, BatchFailEmbedding(128))
    store.maintain_schema()

    conn = lancedb.connect(temp_db_dir)
    table = conn.open_table("mem")
    try:
        arrow = table.to_arrow()
    finally:
        _safe_close_table(table)
    assert arrow.column("id").to_pylist() == ["a"]
    # The vector column was not rebuilt; only ordinary columns were maintained.
    assert "vector" in arrow.schema.names
    assert arrow.schema.field("vector").type.list_size == 64
    assert "metadata" in arrow.schema.names


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
