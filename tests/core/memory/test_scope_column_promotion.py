"""Store-level tests for scope-column promotion (#822, slice 001).

Covers that fresh tables are born with the derived columns, that the write path
populates them from note metadata, and that a legacy table lacking the columns
is migrated in place — back-filling from each row's metadata JSON, preserving all
existing rows and the authoritative metadata JSON byte-for-byte.
"""

from __future__ import annotations

import json
import shutil
import tempfile

import lancedb  # type: ignore
import pytest

from xagent.core.execution_scope import MEMORY_DIMENSION_METADATA_PREFIX
from xagent.core.memory.core import MemoryNote
from xagent.core.memory.lancedb import LanceDBMemoryStore
from xagent.core.memory.scope_columns import SCOPE_DIMS_COLUMN, USER_ID_COLUMN
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table

from .conftest import ConstantEmbedding

P = MEMORY_DIMENSION_METADATA_PREFIX


@pytest.fixture
def temp_db_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def _store(temp_db_dir, name="mem", embedding=None):
    return LanceDBMemoryStore(
        db_dir=temp_db_dir,
        collection_name=name,
        embedding_model=embedding if embedding is not None else ConstantEmbedding(64),
    )


def _arrow(store, name="mem"):
    conn = store._vector_store.get_raw_connection()
    table = conn.open_table(name)
    try:
        return table.to_arrow()
    finally:
        _safe_close_table(table)


def test_fresh_table_has_scope_columns(temp_db_dir):
    store = _store(temp_db_dir)
    names = set(_arrow(store).schema.names)
    assert {USER_ID_COLUMN, SCOPE_DIMS_COLUMN} <= names


def test_write_path_populates_scope_columns(temp_db_dir):
    store = _store(temp_db_dir)
    res = store.add(
        MemoryNote(
            id="n1",
            content="alpha",
            metadata={"user_id": 7, f"{P}tenant": "acme", f"{P}agent": "x"},
        )
    )
    assert res.success

    arrow = _arrow(store)
    row = {c: arrow.column(c).to_pylist() for c in arrow.schema.names}
    idx = row["id"].index("n1")
    assert row[USER_ID_COLUMN][idx] == 7
    assert row[SCOPE_DIMS_COLUMN][idx] == ["agent=x", "tenant=acme"]


def test_write_path_empty_dims_and_no_user(temp_db_dir):
    store = _store(temp_db_dir)
    assert store.add(MemoryNote(id="n2", content="beta")).success

    arrow = _arrow(store)
    row = {c: arrow.column(c).to_pylist() for c in arrow.schema.names}
    idx = row["id"].index("n2")
    assert row[USER_ID_COLUMN][idx] is None
    assert row[SCOPE_DIMS_COLUMN][idx] == []


def _create_legacy_table(temp_db_dir, name="mem"):
    """A pre-#822 table: id/text/metadata/vector, no derived scope columns."""
    db = lancedb.connect(temp_db_dir)
    ts = "2026-01-01T00:00:00"
    rows = [
        {
            "id": "1",
            "text": "alpha",
            "metadata": json.dumps(
                {
                    "content": "alpha",
                    "timestamp": ts,
                    "user_id": 7,
                    f"{P}tenant": "acme",
                }
            ),
            "vector": [0.1] * 64,
        },
        {
            "id": "2",
            "text": "beta",
            "metadata": json.dumps({"content": "beta", "timestamp": ts, "user_id": 9}),
            "vector": [0.2] * 64,
        },
    ]
    table = db.create_table(name, data=rows)
    _safe_close_table(table)


def test_legacy_table_is_migrated_and_backfilled(temp_db_dir):
    _create_legacy_table(temp_db_dir)

    # Constructing the store triggers _ensure_scope_columns during init.
    store = _store(temp_db_dir)

    arrow = _arrow(store)
    names = set(arrow.schema.names)
    assert {USER_ID_COLUMN, SCOPE_DIMS_COLUMN} <= names
    # The existing vector column is preserved (no re-embed / drop).
    assert "vector" in names

    by_id = {arrow.column("id").to_pylist()[i]: i for i in range(arrow.num_rows)}
    assert set(by_id) == {"1", "2"}, "all legacy rows preserved"

    uid = arrow.column(USER_ID_COLUMN).to_pylist()
    dims = arrow.column(SCOPE_DIMS_COLUMN).to_pylist()
    assert uid[by_id["1"]] == 7
    assert dims[by_id["1"]] == ["tenant=acme"]
    assert uid[by_id["2"]] == 9
    assert dims[by_id["2"]] == []

    # The authoritative metadata JSON is untouched by the promotion.
    meta1 = json.loads(arrow.column("metadata").to_pylist()[by_id["1"]])
    assert meta1["content"] == "alpha"
    assert meta1["user_id"] == 7
    assert meta1[f"{P}tenant"] == "acme"

    # Rows still round-trip through the store API.
    assert store.get("1").content.content == "alpha"
    assert store.get("2").content.content == "beta"


def test_promotion_is_idempotent(temp_db_dir):
    _create_legacy_table(temp_db_dir)
    _store(temp_db_dir)  # first construction migrates
    # Second construction must not error, wipe, or duplicate rows.
    store = _store(temp_db_dir)
    arrow = _arrow(store)
    assert arrow.num_rows == 2
    assert {USER_ID_COLUMN, SCOPE_DIMS_COLUMN} <= set(arrow.schema.names)
