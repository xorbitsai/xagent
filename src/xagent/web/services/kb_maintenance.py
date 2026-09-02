"""LanceDB index maintenance for the knowledge base (#1557).

Maintenance used to hang off the ingestion hot path, so its cost scaled with
document count and it stopped entirely whenever ingestion was idle -- exactly
when a stale index goes unnoticed. On a timer it is the reverse: the sweep has
no ingestion context to scope it, so it sweeps every table, but only once per
interval. The per-table gating (``should_compact``) and the per-table advisory
lock inside ``compact_tables`` are what keep a full sweep cheap.

Compaction runs in every supported deployment, like
``run_orphan_upload_gc_loop``: the in-process loop below always starts, and
Celery Beat schedules the same work wherever Beat is running. Both is safe --
the compaction lock is a ``FileLock`` on the shared LanceDB volume, so
whichever process loses simply skips that table. Enabling Celery is not
evidence that Beat exists (``scripts/dev_background_jobs.py --no-beat`` sets
one without the other), so gating the loop on it would leave that deployment
with no maintenance at all.

The retrain is Beat-only: reproducing a weekly cadence in-process would need a
restart-surviving last-run marker, and this loop keeps no state.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from ...core.tools.core.RAG_tools.storage.contracts import VectorIndexStore
from ...core.tools.core.RAG_tools.storage.factory import StorageFactory

logger = logging.getLogger(__name__)

#: Key the listing failure is counted under, so it shares the per-table alert
#: surface. Not a table name; the leading underscores keep it from colliding.
LISTING_FAILURE_KEY = "__listing__"


def _failing_keys() -> dict[str, int]:
    """Maintenance keys failing often enough to alert on, for the result.

    Per-process best-effort (see ``_maintenance_failures``): the escalated
    ERROR the store logs on crossing the threshold is the alert itself, since
    ops_signals/health is per-process and served by the backend, so it cannot
    see a sweep running in the Celery worker.
    """
    from ...core.tools.core.RAG_tools.storage.lancedb_stores import (
        failing_maintenance_keys,
    )

    return failing_maintenance_keys()


def _discover_tables(store: VectorIndexStore) -> list[str]:
    """Table listing that raises instead of reporting an outage as an empty DB.

    ``list_table_names`` swallows the error and returns ``[]``; a sweep that
    took that at face value reported a dead database as healthy.
    """
    from ...core.tools.core.RAG_tools.storage.lancedb_stores import (
        _record_maintenance_outcome,
    )

    try:
        tables = list(store.list_table_names_strict())
    except Exception:
        _record_maintenance_outcome(LISTING_FAILURE_KEY, False)
        raise
    _record_maintenance_outcome(LISTING_FAILURE_KEY, True)
    return tables


def sweep_kb_storage(stop_event: threading.Event | None = None) -> dict[str, Any]:
    """Compact every degraded KB table and refresh its FTS index."""
    store = StorageFactory.get_factory().get_vector_index_store()
    try:
        tables = _discover_tables(store)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not list KB tables, skipping maintenance: %s", exc)
        return {
            "status": "listing_failed",
            "scanned": 0,
            "compacted": [],
            "failing": _failing_keys(),
        }
    compacted = store.compact_tables(tables, stop_event=stop_event) if tables else []
    if compacted:
        logger.info("Compacted LanceDB tables: %s", ", ".join(compacted))
    return {
        "status": "ok",
        "scanned": len(tables),
        "compacted": compacted,
        "failing": _failing_keys(),
    }


def retrain_kb_vector_indexes() -> dict[str, Any]:
    """Rebuild every KB vector index from scratch.

    Gated apart from :func:`sweep_kb_storage`, deliberately: a retrain on every
    maintenance pass is what the fix for the never-matching existence check
    removed.
    """
    store = StorageFactory.get_factory().get_vector_index_store()
    try:
        tables = _discover_tables(store)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not list KB tables, skipping retrain: %s", exc)
        return {
            "status": "listing_failed",
            "retrained": [],
            "contended": [],
            "failing": _failing_keys(),
        }

    retrained: list[str] = []
    contended: list[str] = []
    for name in tables:
        if not name.startswith("embeddings_"):
            continue
        outcome = store.retrain_vector_index(name)
        if outcome == "retrained":
            retrained.append(name)
        elif outcome == "contended":
            contended.append(name)
    if retrained:
        logger.info("Retrained vector indices: %s", ", ".join(retrained))
    return {
        # A retrain lost to the lock is a whole week gone, so it must not read
        # as a clean pass.
        "status": "contended" if contended else "ok",
        "retrained": retrained,
        "contended": contended,
        "failing": _failing_keys(),
    }


async def run_kb_maintenance_loop(
    *,
    poll_interval_seconds: int,
    stop_event: threading.Event | None = None,
) -> None:
    """Run the compaction sweep on a timer inside the FastAPI process.

    Sweeps before sleeping, like ``run_orphan_upload_gc_loop``: a process
    recycled more often than the interval (gunicorn ``max_requests``) would
    otherwise never reach a single sweep. Cancelling this coroutine cannot stop
    the executor thread, so shutdown sets ``stop_event`` and the sweep unwinds
    at its next table boundary.
    """
    while True:
        if stop_event is not None and stop_event.is_set():
            return
        try:
            await asyncio.to_thread(sweep_kb_storage, stop_event)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("KB index maintenance sweep failed")
        await asyncio.sleep(poll_interval_seconds)
