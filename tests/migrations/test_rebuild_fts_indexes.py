"""Tests for the embeddings FTS index rebuild migration."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import FtsRebuildOutcome
from xagent.migrations.lancedb import rebuild_fts_indexes as mod


class _FakeIndex:
    def __init__(self, name: str, columns: List[str] | None = None) -> None:
        self.name = name
        self.index_type = "FTS"
        self.columns = columns or ["text"]


class _FakeStats:
    num_indexed_rows = 10
    num_unindexed_rows = 0


class _FakeTable:
    def __init__(self, version: int = 7) -> None:
        self.version = version
        self.closed = False

    def list_indices(self) -> List[_FakeIndex]:
        return [_FakeIndex("text_idx")]

    def index_stats(self, name: str) -> _FakeStats:
        return _FakeStats()

    def close(self) -> None:
        self.closed = True


class _FakeConn:
    def __init__(self, names: List[str]) -> None:
        self._names = names
        self.versions = {name: 7 for name in names}

    def table_names(self) -> List[str]:
        return self._names

    def open_table(self, name: str) -> _FakeTable:
        if name not in self._names:
            raise ValueError(f"no table {name}")
        return _FakeTable(self.versions[name])


class _RecordingStore:
    """Stands in for LanceDBVectorIndexStore; ``outcomes`` overrides per table."""

    def __init__(self, outcomes: Dict[str, Any] | None = None) -> None:
        self.calls: List[str] = []
        self.outcomes = outcomes or {}

    def rebuild_text_fts_index(self, table_name: str) -> FtsRebuildOutcome:
        self.calls.append(table_name)
        outcome = self.outcomes.get(table_name, FtsRebuildOutcome.REBUILT)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _wire(monkeypatch: pytest.MonkeyPatch, **kw: Any) -> _RecordingStore:
    recorder = _RecordingStore(**kw)
    monkeypatch.setattr(mod, "LanceDBVectorIndexStore", lambda: recorder)
    return recorder


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> _RecordingStore:
    return _wire(monkeypatch)


def _without_text_fts(conn: _FakeConn, monkeypatch: pytest.MonkeyPatch, table: str):
    """Give ``table`` an FTS index on metadata instead of text."""

    class _MetadataOnly(_FakeTable):
        def list_indices(self) -> List[_FakeIndex]:
            return [_FakeIndex("metadata_idx", ["metadata"])]

    opened = conn.open_table
    monkeypatch.setattr(
        conn,
        "open_table",
        lambda name: (
            _MetadataOnly(conn.versions[name]) if name == table else opened(name)
        ),
    )


def test_lists_only_embeddings_tables() -> None:
    conn = _FakeConn(
        ["documents", "embeddings_b", "chunks", "embeddings_a", "parses"],
    )
    assert mod.list_embeddings_tables(conn) == ["embeddings_a", "embeddings_b"]


def test_dry_run_does_not_rebuild(store: _RecordingStore) -> None:
    conn = _FakeConn(["documents", "embeddings_a", "embeddings_b"])

    result = mod.rebuild_fts_indexes(dry_run=True, conn=conn)

    assert store.calls == []
    assert result["would_rebuild"] == ["embeddings_a", "embeddings_b"]
    assert result["succeeded"] == []
    assert result["failed"] == []
    assert result["dry_run"] is True


def test_rebuild_failure_is_reported_as_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising rebuild must not be reported as success."""
    conn = _FakeConn(["embeddings_a", "embeddings_b", "embeddings_c"])
    recorder = _wire(monkeypatch, outcomes={"embeddings_b": RuntimeError("boom")})

    result = mod.rebuild_fts_indexes(conn=conn)

    assert recorder.calls == ["embeddings_a", "embeddings_b", "embeddings_c"]
    assert result["succeeded"] == ["embeddings_a", "embeddings_c"]
    assert result["failed"] == ["embeddings_b"]


def test_rebuild_failure_sets_a_nonzero_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _FakeConn(["embeddings_a"])
    _wire(monkeypatch, outcomes={"embeddings_a": RuntimeError("boom")})
    monkeypatch.setattr(mod, "get_connection_from_env", lambda: conn)

    assert mod.main([]) == 1


def test_panicking_rebuild_is_reported_as_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pyo3 Rust panic is a BaseException, not an Exception."""

    class _Panic(BaseException):
        pass

    conn = _FakeConn(["embeddings_a"])
    _wire(monkeypatch, outcomes={"embeddings_a": _Panic("panicked")})

    result = mod.rebuild_fts_indexes(conn=conn)

    assert result["failed"] == ["embeddings_a"]
    assert result["succeeded"] == []


def test_table_without_text_fts_is_never_reported_as_rebuilt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An FTS index on metadata alone leaves the text index unbuilt."""
    conn = _FakeConn(["embeddings_a", "embeddings_b"])
    _without_text_fts(conn, monkeypatch, "embeddings_a")
    recorder = _wire(monkeypatch)

    result = mod.rebuild_fts_indexes(conn=conn)

    assert recorder.calls == ["embeddings_b"]
    assert result["skipped"] == ["embeddings_a"]
    assert result["succeeded"] == ["embeddings_b"]
    assert result["failed"] == []


def test_a_skipping_store_is_not_counted_as_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The plan said rebuild, so a lock held elsewhere is unfinished work."""
    conn = _FakeConn(["embeddings_a"])
    _wire(monkeypatch, outcomes={"embeddings_a": FtsRebuildOutcome.SKIPPED_LOCKED})

    result = mod.rebuild_fts_indexes(conn=conn)

    assert result["failed"] == ["embeddings_a"]
    assert result["succeeded"] == []


def test_dry_run_and_execute_classify_the_same_tables_alike(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _FakeConn(["embeddings_a", "embeddings_b", "embeddings_c"])
    _without_text_fts(conn, monkeypatch, "embeddings_b")
    _wire(monkeypatch)

    planned = mod.rebuild_fts_indexes(dry_run=True, conn=conn)
    executed = mod.rebuild_fts_indexes(conn=conn)

    assert planned["would_rebuild"] == executed["succeeded"]
    assert planned["skipped"] == executed["skipped"] == ["embeddings_b"]


def test_unreadable_table_is_a_failure_in_both_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _FakeConn(["embeddings_a", "embeddings_b"])
    opened = conn.open_table

    def open_table(name: str) -> _FakeTable:
        if name == "embeddings_a":
            raise RuntimeError("corrupt manifest")
        return opened(name)

    monkeypatch.setattr(conn, "open_table", open_table)
    recorder = _wire(monkeypatch)

    planned = mod.rebuild_fts_indexes(dry_run=True, conn=conn)
    executed = mod.rebuild_fts_indexes(conn=conn)

    assert planned["failed"] == executed["failed"] == ["embeddings_a"]
    assert recorder.calls == ["embeddings_b"]


def test_table_option_rejects_non_embeddings_table(store: _RecordingStore) -> None:
    conn = _FakeConn(["documents", "embeddings_a"])

    with pytest.raises(ValueError, match="documents"):
        mod.rebuild_fts_indexes(table="documents", conn=conn)
    assert store.calls == []


def test_table_option_rebuilds_only_that_table(store: _RecordingStore) -> None:
    conn = _FakeConn(["embeddings_a", "embeddings_b"])

    result = mod.rebuild_fts_indexes(table="embeddings_b", conn=conn)

    assert store.calls == ["embeddings_b"]
    assert result["tables"] == ["embeddings_b"]


def test_main_exit_code_reflects_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _FakeConn(["embeddings_a", "embeddings_b"])
    _wire(monkeypatch, outcomes={"embeddings_b": RuntimeError("boom")})
    monkeypatch.setattr(mod, "get_connection_from_env", lambda: conn)

    assert mod.main([]) == 1
    assert mod.main(["--dry-run"]) == 0
    assert mod.main(["--table", "embeddings_a"]) == 0
    assert mod.main(["--table", "documents"]) == 2


def test_describe_indexes_reports_version_and_row_counts() -> None:
    conn = _FakeConn(["embeddings_a"])

    summary = mod.describe_indexes(conn, "embeddings_a")

    assert summary["version"] == 7
    assert summary["indexes"]["text_idx"]["type"] == "FTS"
    assert summary["indexes"]["text_idx"]["indexed_rows"] == 10
