"""Regression tests for safe schema migration on the ``add()`` path (792-02)."""

from __future__ import annotations

import shutil
import tempfile

import lancedb  # type: ignore
import pyarrow as pa  # type: ignore
import pytest

import xagent.core.memory.lancedb as lancedb_memory
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
        self.batch_calls = 0

    def encode(self, text, dimension=None, instruct=None):
        if isinstance(text, str):
            return [0.1] * self._dimension
        self.batch_calls += 1
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


def test_add_rejects_dimension_change_without_rebuilding(temp_db_dir, monkeypatch):
    """An ordinary write never changes an admitted table's vector space."""
    store_a = _store(temp_db_dir, MockEmbedding(64))
    added = store_a.add(MemoryNote(content="alpha"))
    assert added.success
    alpha_id = added.memory_id

    committed = store_a.add(MemoryNote(content="committed"))
    assert committed.success

    # A provider returning a different width must fail the write rather than
    # re-embed and replace the live table from the request path.
    store_b = _store(temp_db_dir, MockEmbedding(128))
    swap_calls = 0

    def forbid_request_time_swap(*_args, **_kwargs):
        nonlocal swap_calls
        swap_calls += 1
        raise AssertionError("ordinary add reached migrate_table_swap")

    # Patch only after both constructors completed: an empty bootstrap table
    # may be shaped before publication, while this assertion is specifically
    # about add-error recovery on the populated persistent table.
    monkeypatch.setattr(lancedb_memory, "migrate_table_swap", forbid_request_time_swap)
    new = store_b.add(MemoryNote(content="beta"))
    assert not new.success
    assert swap_calls == 0

    schema, rows = _table_snapshot(store_a)
    assert schema.field("vector").type.list_size == 64
    assert {alpha_id, committed.memory_id} <= rows.keys()
    assert new.memory_id not in rows


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


def test_add_dimension_mismatch_never_attempts_reembedding(temp_db_dir):
    """Request-time mismatch handling cannot enter the batch re-embed path."""
    store_a = _store(temp_db_dir, MockEmbedding(64))
    added = store_a.add(MemoryNote(content="alpha"))
    assert added.success
    alpha_id = added.memory_id

    embedding = BatchFailEmbedding(128)
    store_fail = _store(temp_db_dir, embedding)
    result = store_fail.add(MemoryNote(content="beta"))
    assert not result.success
    assert embedding.batch_calls == 0

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


def _fail_next_table_add(monkeypatch, store, error):
    """Make the next ``table.add`` on the store's collection raise ``error``.

    Only the first call fails, so a (wrong) rebuild-then-retry would still be
    able to commit and the test observes what add() did to the table.
    """
    conn = store._vector_store.get_raw_connection()
    table = conn.open_table("mem")
    table_type = type(table)
    _safe_close_table(table)
    original_add = table_type.add
    calls = {"count": 0}

    def _flaky_add(self, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise error
        return original_add(self, *args, **kwargs)

    monkeypatch.setattr(table_type, "add", _flaky_add)
    return calls


def _table_snapshot(store):
    conn = store._vector_store.get_raw_connection()
    table = conn.open_table("mem")
    try:
        arrow = table.to_arrow()
    finally:
        _safe_close_table(table)
    rows = {
        row["id"]: row.get("vector")
        for row in arrow.select(
            [name for name in ("id", "vector") if name in arrow.schema.names]
        ).to_pylist()
    }
    return arrow.schema, rows


def test_text_only_write_failure_never_rebuilds_the_vector_table(
    temp_db_dir, monkeypatch
):
    """A no-adapter (TEXT_ONLY) store whose insert fails for a reason unrelated
    to schema must surface the failure and leave every row and vector intact —
    never rewrite the table without its vector column."""
    writer = _store(temp_db_dir, MockEmbedding(64))
    assert writer.add(MemoryNote(id="a", content="alpha")).success
    assert writer.add(MemoryNote(id="b", content="beta")).success
    schema_before, rows_before = _table_snapshot(writer)
    assert "vector" in schema_before.names

    text_only = _store(temp_db_dir, None)
    _fail_next_table_add(monkeypatch, text_only, OSError("No space left on device"))

    result = text_only.add(MemoryNote(id="c", content="gamma"))

    assert not result.success
    assert result.error
    schema_after, rows_after = _table_snapshot(text_only)
    assert schema_after == schema_before
    assert rows_after == rows_before


def test_text_only_write_failure_on_a_stale_vector_table_only_backfills(
    temp_db_dir, monkeypatch
):
    """Missing non-vector columns are still backfilled from the no-adapter add()
    path on a vector table, and the vectors survive the backfill."""
    writer = _store(temp_db_dir, MockEmbedding(64))
    assert writer.add(MemoryNote(id="a", content="alpha")).success
    # Open the TEXT_ONLY store first, so the stale column reaches add() rather
    # than the init path's own schema resolution.
    text_only = _store(temp_db_dir, None)
    conn = writer._vector_store.get_raw_connection()
    table = conn.open_table("mem")
    try:
        table.drop_columns(["text"])
    finally:
        _safe_close_table(table)
    _, rows_before = _table_snapshot(writer)

    assert text_only.add(MemoryNote(id="b", content="beta")).success

    schema_after, rows_after = _table_snapshot(text_only)
    assert "vector" in schema_after.names
    assert "text" in schema_after.names
    assert rows_after["a"] == rows_before["a"]
    assert rows_after["b"] is None


def test_adapter_store_write_failure_on_a_compatible_table_is_surfaced(
    temp_db_dir, monkeypatch
):
    """With an adapter and a matching schema, a non-schema insert failure is
    reported and nothing is migrated."""
    store = _store(temp_db_dir, MockEmbedding(64))
    assert store.add(MemoryNote(id="a", content="alpha")).success
    schema_before, rows_before = _table_snapshot(store)
    _fail_next_table_add(monkeypatch, store, OSError("transient"))

    assert not store.add(MemoryNote(id="b", content="beta")).success
    schema_after, rows_after = _table_snapshot(store)
    assert schema_after == schema_before
    assert rows_after == rows_before
