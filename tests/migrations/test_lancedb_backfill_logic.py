"""Tests for LanceDB user_id backfill migration logic.

This module verifies that the backfill migration correctly populates user_id
fields in chunks and embeddings tables by joining with the documents table.
"""

from __future__ import annotations

import multiprocessing
import os
import tempfile
from typing import Any

import lancedb
import pyarrow as pa
import pytest

from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
    ensure_documents_table,
)
from xagent.migrations.lancedb import backfill_user_id
from xagent.migrations.lancedb.backfill_user_id import (
    ORPHANED_PERMANENT,
    backfill_all,
    backfill_chunks_table,
)


@pytest.fixture
def temp_conn():
    """Create a temporary LanceDB connection."""
    with tempfile.TemporaryDirectory() as temp_dir:
        conn = lancedb.connect(temp_dir)
        yield conn


def test_backfill_logic_success(temp_conn):
    """Test that backfill correctly updates user_id from documents to chunks/embeddings."""
    conn = temp_conn

    # 1. Setup: Create documents table with data (including user_id)
    ensure_documents_table(conn)
    docs_table = conn.open_table("documents")
    docs_data = [
        {"collection": "c1", "doc_id": "doc1", "user_id": 101, "source_path": "p1"},
        {"collection": "c1", "doc_id": "doc2", "user_id": 102, "source_path": "p2"},
    ]
    docs_table.add(docs_data)

    # 2. Setup: Create chunks table with OLD schema (no user_id)
    old_chunks_schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("chunk_id", pa.string()),
            pa.field("parse_hash", pa.string()),
            pa.field("text", pa.large_string()),
        ]
    )
    chunks_table = conn.create_table("chunks", schema=old_chunks_schema)
    chunks_data = [
        {
            "collection": "c1",
            "doc_id": "doc1",
            "chunk_id": "chk1",
            "parse_hash": "h1",
            "text": "t1",
        },
        {
            "collection": "c1",
            "doc_id": "doc2",
            "chunk_id": "chk2",
            "parse_hash": "h2",
            "text": "t2",
        },
        {
            "collection": "c1",
            "doc_id": "doc3",
            "chunk_id": "chk3",
            "parse_hash": "h3",
            "text": "t3",
        },  # doc3 doesn't exist in docs table
    ]
    chunks_table.add(chunks_data)

    # 3. Setup: Create embeddings table with OLD schema
    old_emb_schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("chunk_id", pa.string()),
            pa.field("parse_hash", pa.string()),
            pa.field("model", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 2)),
        ]
    )
    emb_table = conn.create_table("embeddings_test_model", schema=old_emb_schema)
    emb_data = [
        {
            "collection": "c1",
            "doc_id": "doc1",
            "chunk_id": "chk1",
            "parse_hash": "h1",
            "model": "m1",
            "vector": [0.1, 0.2],
        },
    ]
    emb_table.add(emb_data)

    # 4. Simulate the first step of migration: adding the column as NULL
    # This mimics what schema_manager.py does now
    chunks_table.add_columns({"user_id": "cast(null as bigint)"})
    emb_table.add_columns({"user_id": "cast(null as bigint)"})

    # Verify they are currently NULL
    assert (
        chunks_table.search().where("user_id IS NULL").to_list()[0]["user_id"] is None
    )

    # 5. Run the backfill migration
    result = backfill_all(dry_run=False, conn=conn)

    # Re-open table to ensure we are seeing persisted state
    chunks_table = conn.open_table("chunks")

    # 6. Verifications
    assert result["chunks"]["backfilled"] == 2
    assert result["chunks"]["skipped"] == 1  # doc3 skipped
    assert result["embeddings"]["backfilled"] == 1

    # Check actual data in chunks
    updated_chunks = chunks_table.search().to_list()
    chunk_map = {c["chunk_id"]: c["user_id"] for c in updated_chunks}
    assert chunk_map["chk1"] == 101
    assert chunk_map["chk2"] == 102
    assert chunk_map["chk3"] == ORPHANED_PERMANENT

    # Check actual data in embeddings
    emb_table = conn.open_table("embeddings_test_model")
    updated_emb = emb_table.search().to_list()
    assert updated_emb[0]["user_id"] == 101


def test_backfill_core_stops_on_zero_progress(monkeypatch: pytest.MonkeyPatch):
    """Backfill core aborts when an entire batch fails to update."""

    class _Query:
        def __init__(self, source: str):
            self.source = source

    class _Search:
        def __init__(self, source: str):
            self.source = source

        def where(self, _expr: str) -> "_Search":
            return self

        def limit(self, _n: int) -> _Query:
            return _Query(self.source)

    class _Table:
        def search(self) -> _Search:
            return _Search("chunks")

        def update(self, _update_filter: str, _values: dict[str, Any]) -> None:
            raise RuntimeError("simulated update failure")

    class _DocsTable:
        def search(self) -> _Search:
            return _Search("docs")

    chunks = _Table()
    docs = _DocsTable()
    batch_records = [
        {
            "doc_id": "doc_missing",
            "chunk_id": "chunk_1",
            "parse_hash": "hash_1",
        }
    ]

    def _fake_query_to_list(query: _Query) -> list[dict[str, Any]]:
        if query.source == "chunks":
            return batch_records
        return []

    monkeypatch.setattr(backfill_user_id, "query_to_list", _fake_query_to_list)

    result = backfill_user_id._backfill_table_core(
        table=chunks,
        docs_table=docs,
        query_filter="user_id IS NULL",
        filter_fields=["doc_id", "chunk_id", "parse_hash"],
        failure_user_id=backfill_user_id.ORPHANED_TEMPORARY,
        dry_run=False,
        log_prefix="test:",
    )

    assert result["failed"] == 1
    assert result["backfilled"] == 0


def test_backfill_all_returns_when_file_lock_held(
    temp_conn, monkeypatch: pytest.MonkeyPatch
):
    """Migration should exit early when file lock is already held by another process."""
    conn = temp_conn
    monkeypatch.setattr(backfill_user_id, "_acquire_file_lock", lambda: None)

    result = backfill_all(dry_run=True, conn=conn)
    assert result.get("error") == "Migration file lock already held"


def test_backfill_all_returns_when_thread_lock_held(temp_conn):
    """Migration should exit early when in-process lock is already held."""
    conn = temp_conn
    acquired = backfill_user_id._migration_lock.acquire(blocking=False)
    assert acquired is True
    try:
        result = backfill_all(dry_run=True, conn=conn)
        assert result.get("error") == "Migration lock already held"
    finally:
        if backfill_user_id._migration_lock.locked():
            backfill_user_id._migration_lock.release()


def test_backfill_all_releases_file_lock_on_exception(
    temp_conn, monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """File lock should be released even when migration raises."""
    conn = temp_conn
    lock_path = tmp_path / "migration.lock"
    monkeypatch.setenv("LANCEDB_MIGRATION_LOCK_FILE", str(lock_path))

    def _raise_chunks(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(backfill_user_id, "backfill_chunks_table", _raise_chunks)

    with pytest.raises(RuntimeError, match="boom"):
        backfill_all(dry_run=False, conn=conn)

    lock_file = backfill_user_id._acquire_file_lock()
    assert lock_file is not None
    backfill_user_id._release_file_lock(lock_file)


def _acquire_file_lock_in_subprocess(
    lock_path: str, queue: multiprocessing.Queue
) -> None:
    """Try acquiring migration file lock and report success to parent."""
    os.environ["LANCEDB_MIGRATION_LOCK_FILE"] = lock_path
    lock_file = backfill_user_id._acquire_file_lock()
    success = lock_file is not None
    if lock_file is not None:
        backfill_user_id._release_file_lock(lock_file)
    queue.put(success)


def test_file_lock_blocks_across_processes(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """File lock should prevent another process from acquiring concurrently."""
    lock_path = tmp_path / "migration.lock"
    monkeypatch.setenv("LANCEDB_MIGRATION_LOCK_FILE", str(lock_path))

    parent_lock = backfill_user_id._acquire_file_lock()
    assert parent_lock is not None
    try:
        queue: multiprocessing.Queue = multiprocessing.Queue()
        proc = multiprocessing.Process(
            target=_acquire_file_lock_in_subprocess,
            args=(str(lock_path), queue),
        )
        proc.start()
        proc.join(timeout=10)
        assert proc.exitcode == 0
        assert queue.get(timeout=5) is False
    finally:
        backfill_user_id._release_file_lock(parent_lock)

    queue2: multiprocessing.Queue = multiprocessing.Queue()
    proc2 = multiprocessing.Process(
        target=_acquire_file_lock_in_subprocess,
        args=(str(lock_path), queue2),
    )
    proc2.start()
    proc2.join(timeout=10)
    assert proc2.exitcode == 0
    assert queue2.get(timeout=5) is True


def test_phase1_misses_rows_when_user_id_default_is_non_null(temp_conn):
    """Guardrail: non-NULL defaults (e.g. 0) are invisible to phase1 IS NULL scan."""
    conn = temp_conn

    ensure_documents_table(conn)
    docs_table = conn.open_table("documents")
    docs_table.add(
        [
            {
                "collection": "c1",
                "doc_id": "doc1",
                "user_id": 101,
                "source_path": "p1",
            }
        ]
    )

    # Simulate an incorrect schema migration that adds user_id with non-NULL default.
    chunks_schema_with_user_id = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("chunk_id", pa.string()),
            pa.field("parse_hash", pa.string()),
            pa.field("text", pa.large_string()),
            pa.field("user_id", pa.int64()),
        ]
    )
    chunks_table = conn.create_table("chunks", schema=chunks_schema_with_user_id)
    chunks_table.add(
        [
            {
                "collection": "c1",
                "doc_id": "doc1",
                "chunk_id": "chk1",
                "parse_hash": "h1",
                "text": "t1",
                "user_id": 0,
            }
        ]
    )

    result = backfill_chunks_table(dry_run=False, conn=conn)
    assert result["backfilled"] == 0

    persisted = conn.open_table("chunks").search().to_list()
    assert persisted[0]["user_id"] == 0
