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


def test_retrain_rebuilds_an_existing_vector_index(indexed_table, tmp_path):
    """A retrain must actually rebuild: optimize() alone never retrains.

    ``optimize()``'s index step only assigns new rows to the partitions the
    index already has, so recall drifts as the corpus grows (#1557 change 4).
    """
    db = lancedb.connect(str(tmp_path / "contract"))
    store = _store_on(db)
    version_before = db.open_table("embeddings_probe").version

    assert store.retrain_vector_index("embeddings_probe") == "retrained"

    assert db.open_table("embeddings_probe").version > version_before


def test_retrain_skips_a_table_with_no_vector_index(tmp_path):
    """Most tables the sweep sees carry no vector index at all."""
    db = lancedb.connect(str(tmp_path / "retrain-no-index"))
    db.create_table("parses", data=_rows(200))
    store = _store_on(db)

    assert store.retrain_vector_index("parses") == "no_index"


def test_compaction_never_retrains_the_vector_index(indexed_table, tmp_path):
    """Change 4 has to be gated apart from change 3, not run every pass.

    A retrain on every maintenance pass is precisely the cost the fix for the
    never-matching existence check removed (~114 MB and ~14 s a time).
    """
    db = lancedb.connect(str(tmp_path / "contract"))
    indexed_table.add(_rows(100))
    store = _store_on(db)
    calls: list[str] = []
    real_create_index = indexed_table.create_index

    def create_index(*args, **kwargs):
        calls.append("create_index")
        return real_create_index(*args, **kwargs)

    indexed_table.create_index = create_index
    db_open = db.open_table
    db.open_table = lambda name, **kw: indexed_table

    try:
        assert store.trigger_reindex("embeddings_probe") is True
    finally:
        db.open_table = db_open

    assert calls == [], "compaction must not rebuild the vector index"


def _failures():
    from xagent.core.tools.core.RAG_tools.storage import lancedb_stores

    return lancedb_stores.failing_maintenance_keys()


@pytest.fixture(autouse=True)
def _reset_maintenance_counters():
    from xagent.core.tools.core.RAG_tools.storage import lancedb_stores

    lancedb_stores._maintenance_failures.clear()
    yield
    lancedb_stores._maintenance_failures.clear()


def test_repeated_maintenance_failures_become_visible(tmp_path, monkeypatch, caplog):
    """Consecutive failures must be distinguishable from never having run.

    Fourteen consecutive optimize failures produced fourteen identical
    warnings and no other signal, which is why both defects went unnoticed.
    """
    from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
        MAINTENANCE_FAILURE_ALERT_THRESHOLD,
    )

    db = lancedb.connect(str(tmp_path / "failing"))
    table = db.create_table("embeddings_probe", data=_rows(50))
    table.optimize = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("disk full"))
    store = _store_on(db)
    monkeypatch.setattr(store, "_get_connection", lambda: db)
    monkeypatch.setattr(db, "open_table", lambda name, **kw: table)

    with caplog.at_level(logging.ERROR):
        for _ in range(MAINTENANCE_FAILURE_ALERT_THRESHOLD - 1):
            assert store.trigger_reindex("embeddings_probe") is False
        assert not _failures(), "escalated before the threshold"
        assert not caplog.records
        assert store.trigger_reindex("embeddings_probe") is False

    assert _failures() == {"embeddings_probe": MAINTENANCE_FAILURE_ALERT_THRESHOLD}
    assert any(
        "consecutive times" in r.getMessage() and r.levelno == logging.ERROR
        for r in caplog.records
    )


def test_one_clean_pass_clears_the_failure_count(tmp_path, monkeypatch):
    """A recovered table must stop alerting, or the signal latches forever."""
    from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
        MAINTENANCE_FAILURE_ALERT_THRESHOLD,
    )

    db = lancedb.connect(str(tmp_path / "recovers"))
    table = db.create_table("embeddings_probe", data=_rows(50))
    real_optimize = table.optimize
    table.optimize = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("disk full"))
    store = _store_on(db)
    monkeypatch.setattr(store, "_get_connection", lambda: db)
    monkeypatch.setattr(db, "open_table", lambda name, **kw: table)

    for _ in range(MAINTENANCE_FAILURE_ALERT_THRESHOLD):
        store.trigger_reindex("embeddings_probe")
    assert _failures()

    table.optimize = real_optimize
    assert store.trigger_reindex("embeddings_probe") is True

    assert _failures() == {}


def test_a_swallowed_fts_rebuild_failure_still_counts_as_a_failed_pass(
    tmp_path, monkeypatch
):
    """Compaction succeeding is not the same as the index being current.

    ``trigger_reindex`` returns True after a swallowed FTS rebuild failure, so
    the return value alone cannot tell "current" from "silently stale".
    """
    from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
        MAINTENANCE_FAILURE_ALERT_THRESHOLD,
    )

    db = lancedb.connect(str(tmp_path / "fts-counts"))
    table = db.create_table("embeddings_probe", data=_rows(200))
    table.create_fts_index("text", with_position=True, replace=True)
    table.create_fts_index = lambda *a, **kw: (_ for _ in ()).throw(
        RuntimeError("index out of bounds: the len is 10791")
    )
    store = _store_on(db)
    monkeypatch.setattr(store, "_get_connection", lambda: db)
    monkeypatch.setattr(db, "open_table", lambda name, **kw: table)

    for _ in range(MAINTENANCE_FAILURE_ALERT_THRESHOLD):
        assert store.trigger_reindex("embeddings_probe") is True

    assert _failures() == {"embeddings_probe": MAINTENANCE_FAILURE_ALERT_THRESHOLD}


def test_the_alert_fires_once_per_streak_not_once_per_failure(tmp_path, caplog):
    """One incident is one page; every failure already logs its own WARNING."""
    from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
        MAINTENANCE_FAILURE_ALERT_THRESHOLD,
        _record_maintenance_outcome,
    )

    def _errors():
        return [r for r in caplog.records if r.levelno == logging.ERROR]

    with caplog.at_level(logging.ERROR):
        for _ in range(MAINTENANCE_FAILURE_ALERT_THRESHOLD):
            _record_maintenance_outcome("documents", False)
        assert len(_errors()) == 1

        for _ in range(5):
            _record_maintenance_outcome("documents", False)
        assert len(_errors()) == 1, "post-threshold failures must not re-page"

        # A recovered table starts a fresh streak, which must be able to alert.
        _record_maintenance_outcome("documents", True)
        for _ in range(MAINTENANCE_FAILURE_ALERT_THRESHOLD):
            _record_maintenance_outcome("documents", False)

    assert len(_errors()) == 2


def test_a_failed_fts_rebuild_keeps_the_table_eligible(tmp_path, monkeypatch):
    """Physical compaction success must not gate away an incomplete FTS pass.

    ``optimize()`` clears the fragment and version thresholds, so without an
    independent retry the stale FTS index is never rebuilt again.
    """
    from xagent.core.tools.core.RAG_tools.core.config import IndexPolicy

    db = lancedb.connect(str(tmp_path / "fts-eligibility"))
    table = db.create_table("embeddings_probe", data=_rows(100))
    for _ in range(12):
        table.add(_rows(10))
    table.create_fts_index("text", with_position=True, replace=True)

    attempts: list[str] = []

    def failing_fts(*args, **kwargs):
        attempts.append("create_fts_index")
        raise RuntimeError("index out of bounds: the len is 10791")

    table.create_fts_index = failing_fts
    store = _store_on(db)
    monkeypatch.setattr(store, "_get_connection", lambda: db)
    monkeypatch.setattr(db, "open_table", lambda name, **kw: table)
    policy = IndexPolicy(compact_fragment_threshold=10)

    assert store.compact_tables(["embeddings_probe"], policy) == ["embeddings_probe"]
    assert len(attempts) == 1
    # The premise: compaction really did clear the physical trigger.
    assert store.should_compact("embeddings_probe", policy) is False

    assert store.compact_tables(["embeddings_probe"], policy) == ["embeddings_probe"]
    assert len(attempts) == 2, "the stale FTS index was never retried"


def test_compaction_stops_at_the_next_table_when_asked(tmp_path, monkeypatch):
    """Shutdown must not have to wait out the whole listing."""
    import threading

    from xagent.core.tools.core.RAG_tools.core.config import IndexPolicy

    db = lancedb.connect(str(tmp_path / "stop"))
    for name in ("documents", "parses", "chunks"):
        table = db.create_table(name, data=_rows(20))
        for _ in range(12):
            table.add(_rows(5))

    stop = threading.Event()
    store = _store_on(db)
    real_trigger = store.trigger_reindex

    def trigger(name, **kwargs):
        stop.set()
        return real_trigger(name, **kwargs)

    monkeypatch.setattr(store, "trigger_reindex", trigger)

    compacted = store.compact_tables(
        ["documents", "parses", "chunks"],
        IndexPolicy(compact_fragment_threshold=10),
        stop_event=stop,
    )

    assert len(compacted) == 1, "the sweep ran on past the stop signal"


def test_a_retrain_waits_out_a_lock_it_lost(tmp_path, monkeypatch):
    """The weekly retrain has no cheap next attempt, so it waits.

    Hourly compaction and the weekly retrain share one per-table FileLock and
    their default intervals are exact multiples, so collision is routine.
    """
    import threading
    import time

    from xagent.core.tools.core.RAG_tools.storage import lancedb_stores

    db = lancedb.connect(str(tmp_path / "contended"))
    table = db.create_table("embeddings_probe", data=_rows(2000))
    table.create_index(metric="l2", index_type="IVF_HNSW_SQ", num_partitions=4)
    store = _store_on(db)
    monkeypatch.setattr(lancedb_stores, "RETRAIN_LOCK_WAIT_SECONDS", 10)

    holder_has_lock = threading.Event()
    release = threading.Event()

    def hold_the_lock():
        with lancedb_stores._compaction_lock(db, "embeddings_probe") as acquired:
            assert acquired
            holder_has_lock.set()
            release.wait(timeout=10)

    holder = threading.Thread(target=hold_the_lock)
    holder.start()
    holder_has_lock.wait(timeout=5)

    outcome: list[str] = []
    waiter = threading.Thread(
        target=lambda: outcome.append(store.retrain_vector_index("embeddings_probe"))
    )
    waiter.start()
    # Long enough that a non-blocking acquire would have given up by now.
    time.sleep(0.5)
    release.set()
    holder.join(timeout=10)
    waiter.join(timeout=30)

    assert outcome == ["retrained"], "the retrain gave up instead of waiting"


def test_a_retrain_that_gives_up_is_not_reported_as_nothing_to_do(
    tmp_path, monkeypatch
):
    """``contended`` and ``no_index`` must not collapse into one falsey value."""
    from xagent.core.tools.core.RAG_tools.storage import lancedb_stores

    db = lancedb.connect(str(tmp_path / "gives-up"))
    table = db.create_table("embeddings_probe", data=_rows(2000))
    table.create_index(metric="l2", index_type="IVF_HNSW_SQ", num_partitions=4)
    store = _store_on(db)
    monkeypatch.setattr(lancedb_stores, "RETRAIN_LOCK_WAIT_SECONDS", 0)

    with lancedb_stores._compaction_lock(db, "embeddings_probe") as acquired:
        assert acquired
        assert store.retrain_vector_index("embeddings_probe") == "contended"

    # And it is counted, so a run of them reaches the alert threshold.
    assert lancedb_stores._maintenance_failures["embeddings_probe:retrain"] == 1


def test_strict_listing_raises_where_the_lenient_one_returns_empty(tmp_path):
    """The two must differ, or the strict override is dead weight and gets cut.

    A patched-out strict method proves nothing: this drives the real
    implementation against a connection whose listing fails.
    """
    db = lancedb.connect(str(tmp_path / "listing"))
    db.create_table("documents", data=_rows(10))
    store = _store_on(db)

    def _boom():
        raise OSError("lancedb path unavailable")

    db.list_tables = _boom
    db.table_names = _boom

    assert store.list_table_names() == []
    with pytest.raises(OSError):
        store.list_table_names_strict()
