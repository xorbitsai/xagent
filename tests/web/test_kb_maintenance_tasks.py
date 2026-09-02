"""Scheduled KB index maintenance (#1557, changes 3-5).

Runs against a real LanceDB database, like the storage contract suite: the
point of moving maintenance onto a timer is that it works with no ingestion
context to scope it, and a mocked connection cannot show that.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import lancedb
import pyarrow as pa
import pytest

from xagent.core.tools.core.RAG_tools.core.config import IndexPolicy
from xagent.core.tools.core.RAG_tools.storage.factory import StorageFactory
from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
    LanceDBVectorIndexStore,
)
from xagent.web.jobs.celery_app import celery_app
from xagent.web.services import kb_maintenance


@pytest.fixture(autouse=True)
def _reset_maintenance_counters():
    from xagent.core.tools.core.RAG_tools.storage import lancedb_stores

    lancedb_stores._maintenance_failures.clear()
    yield
    lancedb_stores._maintenance_failures.clear()


def _fragmented_db(tmp_path: Any, *names: str) -> Any:
    db = lancedb.connect(str(tmp_path))
    schema = pa.schema([pa.field("name", pa.string())])
    for table_name in names:
        table = db.create_table(table_name, schema=schema)
        for i in range(12):
            table.add([{"name": f"c{i}"}])
    return db


def _bound_store(db: Any) -> LanceDBVectorIndexStore:
    store = LanceDBVectorIndexStore()
    store._conn = db
    return store


def test_beat_schedule_runs_maintenance_off_the_ingestion_path() -> None:
    """Maintenance has to have a scheduler entry, or it never runs at all.

    With ingestion idle the old per-document hook fired zero times, so neither
    version cleanup nor the FTS rebuild ever happened.
    """
    schedule = celery_app.conf.beat_schedule

    assert (
        schedule["compact-kb-storage"]["task"]
        == "xagent.web.jobs.kb_maintenance_tasks.compact_kb_storage"
    )
    assert (
        schedule["retrain-kb-vector-indexes"]["task"]
        == "xagent.web.jobs.kb_maintenance_tasks.retrain_kb_vector_indexes"
    )
    # The retrain is the expensive one (~14 s / +102 MB at 76k x 1024) and must
    # not ride the compaction cadence.
    assert (
        schedule["retrain-kb-vector-indexes"]["schedule"]
        > schedule["compact-kb-storage"]["schedule"]
    )
    # A scheduled name that no task registers is scheduled into nothing.
    for entry in ("compact-kb-storage", "retrain-kb-vector-indexes"):
        assert schedule[entry]["task"] in celery_app.tasks


def test_sweep_compacts_tables_no_ingestion_told_it_about(tmp_path: Any) -> None:
    """The scheduled sweep has no ingestion context, so it reads the listing.

    This also removes the model-tag spelling problem the old hook worked
    around: the listing carries the names the write path really created.
    """
    db = _fragmented_db(tmp_path, "documents", "embeddings_probe")
    store = _bound_store(db)

    with patch.object(StorageFactory, "get_vector_index_store", return_value=store):
        with patch(
            "xagent.core.tools.core.RAG_tools.storage.lancedb_stores."
            "DEFAULT_INDEX_POLICY",
            IndexPolicy(compact_fragment_threshold=10),
        ):
            result = kb_maintenance.sweep_kb_storage()

    assert set(result["compacted"]) == {"documents", "embeddings_probe"}
    assert db.open_table("documents").stats()["fragment_stats"]["num_fragments"] == 1


def test_sweep_reports_repeated_failures_instead_of_hiding_them(
    tmp_path: Any, caplog: Any
) -> None:
    """Repeated failures must become distinguishable from never having run.

    Fourteen consecutive optimize failures produced fourteen identical
    warnings and no other signal, which is why both defects went unnoticed.
    """
    import logging

    from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
        MAINTENANCE_FAILURE_ALERT_THRESHOLD,
    )

    db = _fragmented_db(tmp_path, "documents")
    store = _bound_store(db)
    broken = db.open_table("documents")
    real_optimize = broken.optimize
    broken.optimize = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("disk full"))

    with patch.object(StorageFactory, "get_vector_index_store", return_value=store):
        with patch(
            "xagent.core.tools.core.RAG_tools.storage.lancedb_stores."
            "DEFAULT_INDEX_POLICY",
            IndexPolicy(compact_fragment_threshold=10),
        ):
            with patch.object(db, "open_table", lambda name, **kw: broken):
                with caplog.at_level(logging.ERROR):
                    for _ in range(MAINTENANCE_FAILURE_ALERT_THRESHOLD):
                        failed = kb_maintenance.sweep_kb_storage()

                assert failed["compacted"] == []
                assert failed["failing"] == {
                    "documents": MAINTENANCE_FAILURE_ALERT_THRESHOLD
                }
                assert any(
                    "consecutive times" in r.getMessage() for r in caplog.records
                )

                # And a clean pass has to clear it, or the alert latches forever.
                broken.optimize = real_optimize
                result = kb_maintenance.sweep_kb_storage()

    assert result["compacted"] == ["documents"]
    assert result["failing"] == {}


def test_retrain_task_only_touches_embeddings_tables(tmp_path: Any) -> None:
    """Only embeddings tables carry a vector index; the rest are wasted opens."""
    db = _fragmented_db(tmp_path, "documents", "embeddings_probe")
    store = _bound_store(db)
    asked: list[str] = []

    def _retrain(table_name: str) -> str:
        asked.append(table_name)
        return "retrained"

    with patch.object(StorageFactory, "get_vector_index_store", return_value=store):
        with patch.object(store, "retrain_vector_index", _retrain):
            result = kb_maintenance.retrain_kb_vector_indexes()

    assert asked == ["embeddings_probe"]
    assert result["retrained"] == ["embeddings_probe"]


def test_an_empty_database_is_a_successful_sweep(tmp_path: Any) -> None:
    """A genuinely empty database is a valid boundary, not a failure."""
    db = lancedb.connect(str(tmp_path))
    store = _bound_store(db)

    with patch.object(StorageFactory, "get_vector_index_store", return_value=store):
        result = kb_maintenance.sweep_kb_storage()

    assert result["status"] == "ok"
    assert result["scanned"] == 0
    assert result["failing"] == {}


def test_a_listing_outage_is_not_reported_as_an_empty_database(tmp_path: Any) -> None:
    """``list_table_names`` returns [] on I/O failure, which read as healthy.

    A dead database and an empty one produced the same `status: ok, scanned: 0`
    while every table went unmaintained.
    """
    from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
        MAINTENANCE_FAILURE_ALERT_THRESHOLD,
    )
    from xagent.web.services.kb_maintenance import LISTING_FAILURE_KEY

    db = _fragmented_db(tmp_path, "documents")
    store = _bound_store(db)

    def _unreachable() -> list[str]:
        raise OSError("lancedb path unavailable")

    with patch.object(StorageFactory, "get_vector_index_store", return_value=store):
        with patch.object(store, "list_table_names_strict", _unreachable):
            for _ in range(MAINTENANCE_FAILURE_ALERT_THRESHOLD):
                result = kb_maintenance.sweep_kb_storage()
            retrain = kb_maintenance.retrain_kb_vector_indexes()

    assert result["status"] == "listing_failed"
    assert retrain["status"] == "listing_failed"
    assert result["failing"] == {
        LISTING_FAILURE_KEY: MAINTENANCE_FAILURE_ALERT_THRESHOLD
    }


def test_a_contended_retrain_is_not_reported_as_a_clean_pass(tmp_path: Any) -> None:
    """Losing the only weekly retrain must not read as `status: ok`."""
    db = _fragmented_db(tmp_path, "embeddings_probe")
    store = _bound_store(db)

    with patch.object(StorageFactory, "get_vector_index_store", return_value=store):
        with patch.object(store, "retrain_vector_index", lambda _n: "contended"):
            result = kb_maintenance.retrain_kb_vector_indexes()

    assert result["status"] == "contended"
    assert result["contended"] == ["embeddings_probe"]
    assert result["retrained"] == []
