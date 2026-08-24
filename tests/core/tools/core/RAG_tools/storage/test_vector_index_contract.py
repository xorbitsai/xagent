"""Contract tests for the vector-index existence check (#1557).

These run against a real LanceDB table on purpose. The rest of the store suite
mocks ``lancedb.connect``, which is why an existence check that could never
match went unnoticed for months: LanceDB names the index after its column
(``vector_idx``), never ``vector``, and no mocked test could tell.
"""

from __future__ import annotations

import logging
import random

import lancedb
import pytest


def _rows(n: int, dim: int = 32) -> list[dict]:
    rnd = random.Random(1557)
    return [
        {
            "vector": [rnd.random() for _ in range(dim)],
            "text": f"document {i} lorem ipsum dolor",
        }
        for i in range(n)
    ]


@pytest.fixture
def indexed_table(tmp_path):
    """A real table carrying both a vector index and an FTS index."""
    db = lancedb.connect(str(tmp_path / "contract"))
    table = db.create_table("embeddings_probe", data=_rows(2000))
    table.create_index(metric="l2", index_type="IVF_HNSW_SQ", num_partitions=4)
    table.create_fts_index("text", with_position=True, replace=True)
    return table


def test_vector_index_is_discoverable_by_column(indexed_table):
    """The column-based check finds the index that was just created."""
    indexes = indexed_table.list_indices()

    assert any("vector" in idx.columns for idx in indexes)


def test_column_check_does_not_mistake_the_fts_index_for_a_vector_index(
    tmp_path,
):
    """An FTS index alone must not satisfy the vector-index check.

    Otherwise the fix would swap a never-matches check for an always-matches
    one, and the index would never be built at all.
    """
    db = lancedb.connect(str(tmp_path / "fts-only"))
    table = db.create_table("embeddings_fts_only", data=_rows(200))
    table.create_fts_index("text", with_position=True, replace=True)

    indexes = table.list_indices()

    assert indexes, "expected the FTS index to exist"
    assert not any("vector" in idx.columns for idx in indexes)


def _store_on(db):
    """A vector-index store bound to an already-open connection."""
    from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
        LanceDBVectorIndexStore,
    )

    store = LanceDBVectorIndexStore()
    store._conn = db
    return store


@pytest.fixture
def low_threshold_policy(monkeypatch):
    """Drop the 50k row gate so the index path runs on a small table."""
    from xagent.core.tools.core.RAG_tools.core import config as config_module

    original = config_module.IndexPolicy

    def _policy(*args, **kwargs):
        kwargs.setdefault("enable_threshold_rows", 100)
        kwargs.setdefault("hnsw_params", {"num_partitions": 4})
        return original(*args, **kwargs)

    monkeypatch.setattr(config_module, "IndexPolicy", _policy)


def test_create_index_does_not_rebuild_an_existing_index(
    tmp_path, low_threshold_policy
):
    """Two calls must build the index once.

    Every call rebuilt it in full before this fix -- ~114 MB and ~14 s per call
    on a 77k x 1024 table, several hundred times a day.
    """
    db = lancedb.connect(str(tmp_path / "rebuild"))
    db.create_table("embeddings_probe", data=_rows(500))
    store = _store_on(db)

    first = store.create_index("probe")
    version_after_first = db.open_table("embeddings_probe").version

    second = store.create_index("probe")
    version_after_second = db.open_table("embeddings_probe").version

    assert first.status == "index_building"
    assert second.status == "index_ready"
    assert version_after_second == version_after_first, (
        "the second call rebuilt the index: table version moved from "
        f"{version_after_first} to {version_after_second}"
    )


def _fts_index(table):
    return next(
        (i for i in table.list_indices() if i.index_type == "FTS"),
        None,
    )


def test_reindex_leaves_no_unindexed_fts_rows(tmp_path):
    """After maintenance the FTS index must cover every row.

    ``optimize()`` merges FTS incrementally, and that merge is where the
    upstream panic is reported (lance-format/lance#8310). Rebuilding first is
    what leaves the index complete; whether the merge then still panics is not
    something this suite can reproduce.
    """
    db = lancedb.connect(str(tmp_path / "fts-reindex"))
    table = db.create_table("embeddings_probe", data=_rows(500))
    table.create_fts_index("text", with_position=True, replace=True)
    table.add(_rows(50))
    assert _fts_index(table) is not None

    store = _store_on(db)
    assert store.trigger_reindex("embeddings_probe") is True

    table = db.open_table("embeddings_probe")
    stats = table.index_stats(_fts_index(table).name)
    assert stats.num_unindexed_rows == 0


def test_reindex_of_a_table_without_fts_builds_no_fts_index(tmp_path):
    """The guard matters: most tables routed here carry no FTS index.

    ``compact_tables`` sends documents, parses, chunks and ingestion_runs
    through this path; only the embeddings table has FTS.
    """
    db = lancedb.connect(str(tmp_path / "no-fts"))
    table = db.create_table("parses", data=_rows(200))
    assert _fts_index(table) is None

    store = _store_on(db)
    assert store.trigger_reindex("parses") is True

    assert _fts_index(db.open_table("parses")) is None


def _record_calls(table, calls):
    """Wrap the two index entry points, recording the order they fire in."""
    real_fts, real_optimize = table.create_fts_index, table.optimize

    def fts(*args, **kwargs):
        calls.append("create_fts_index")
        return real_fts(*args, **kwargs)

    def optimize(*args, **kwargs):
        calls.append("optimize")
        return real_optimize(*args, **kwargs)

    table.create_fts_index = fts
    table.optimize = optimize


def test_fts_is_rebuilt_before_optimize(tmp_path, monkeypatch):
    """The rebuild has to precede optimize, or the merge still panics.

    Rebuilding after optimize would leave the reported panic (lance#8310) in
    front of the rebuild, so the order is the part worth pinning.
    """
    db = lancedb.connect(str(tmp_path / "order"))
    table = db.create_table("embeddings_probe", data=_rows(400))
    table.create_fts_index("text", with_position=True, replace=True)

    calls: list[str] = []
    _record_calls(table, calls)
    store = _store_on(db)
    monkeypatch.setattr(store, "_get_connection", lambda: db)
    monkeypatch.setattr(db, "open_table", lambda name, **kw: table)

    assert store.trigger_reindex("embeddings_probe") is True

    assert calls == ["create_fts_index", "optimize"]


def test_optimize_of_a_table_without_fts_skips_the_rebuild(tmp_path, monkeypatch):
    db = lancedb.connect(str(tmp_path / "order-no-fts"))
    table = db.create_table("parses", data=_rows(200))

    calls: list[str] = []
    _record_calls(table, calls)
    store = _store_on(db)
    monkeypatch.setattr(store, "_get_connection", lambda: db)
    monkeypatch.setattr(db, "open_table", lambda name, **kw: table)

    assert store.trigger_reindex("parses") is True

    assert calls == ["optimize"]


def test_a_failed_fts_rebuild_still_lets_optimize_run(tmp_path, monkeypatch, caplog):
    """The rebuild must not take compaction down with it.

    Compaction and version pruning run before optimize's index step and are
    what reclaim the disk; upstream measured a panicking index step still
    releasing 85-90% of it. Letting a rebuild failure skip optimize entirely
    would be the same failure mode this change exists to remove.
    """
    from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
        _fragment_count,
    )

    db = lancedb.connect(str(tmp_path / "fts-boom"))
    table = db.create_table("embeddings_probe", data=_rows(100))
    for _ in range(5):
        table.add(_rows(20))
    table.create_fts_index("text", with_position=True, replace=True)
    fragments_before = _fragment_count(table)
    assert fragments_before > 1, "need a fragmented table to see compaction"

    calls: list[str] = []
    real_open_table = db.open_table
    real_optimize = table.optimize

    def exploding_fts(*args, **kwargs):
        calls.append("create_fts_index")
        raise RuntimeError("index out of bounds: the len is 10791")

    def optimize(*args, **kwargs):
        calls.append("optimize")
        return real_optimize(*args, **kwargs)

    table.create_fts_index = exploding_fts
    table.optimize = optimize
    store = _store_on(db)
    monkeypatch.setattr(store, "_get_connection", lambda: db)
    monkeypatch.setattr(db, "open_table", lambda name, **kw: table)

    with caplog.at_level(logging.WARNING):
        assert store.trigger_reindex("embeddings_probe") is True

    assert calls == ["create_fts_index", "optimize"]
    assert _fragment_count(real_open_table("embeddings_probe")) < fragments_before
    assert any(
        "FTS rebuild failed" in r.getMessage() and r.levelno == logging.WARNING
        for r in caplog.records
    ), "a silently swallowed rebuild failure is indistinguishable from success"


def test_a_panicking_fts_rebuild_still_lets_optimize_run(tmp_path, monkeypatch):
    """A Rust panic arrives as a BaseException, not an Exception.

    pyo3 raises PanicException off BaseException, so catching Exception alone
    would let it abort the compaction this isolation exists to protect.
    """

    class FakePanic(BaseException):
        pass

    db = lancedb.connect(str(tmp_path / "panic"))
    table = db.create_table("embeddings_probe", data=_rows(400))
    table.create_fts_index("text", with_position=True, replace=True)

    calls: list[str] = []
    real_optimize = table.optimize

    def panicking_fts(*args, **kwargs):
        calls.append("create_fts_index")
        raise FakePanic("index out of bounds: the len is 10791")

    def optimize(*args, **kwargs):
        calls.append("optimize")
        return real_optimize(*args, **kwargs)

    table.create_fts_index = panicking_fts
    table.optimize = optimize
    store = _store_on(db)
    monkeypatch.setattr(store, "_get_connection", lambda: db)
    monkeypatch.setattr(db, "open_table", lambda name, **kw: table)

    assert store.trigger_reindex("embeddings_probe") is True

    assert calls == ["create_fts_index", "optimize"]
