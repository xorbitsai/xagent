"""Collection listing must not block the event loop on LanceDB scans.

``_list_collections_impl`` has an async signature, but its documents-table scan
and realtime aggregation are synchronous LanceDB work. Executed on the event loop
they stop every other coroutine in the process, which on a single-worker
deployment stalls the whole API, ``/health`` included, for the scan's duration.
"""

import asyncio
import time

import pyarrow as pa
import pytest

from xagent.core.tools.core.RAG_tools.management import (
    collections as collections_module,
)

BLOCKING_SECONDS = 0.5
HEARTBEAT_INTERVAL = 0.02


class _SlowVectorStore:
    """Blocks the calling thread the way a real LanceDB scan does."""

    def iter_batches(self, *, table_name, columns, user_id, is_admin):
        time.sleep(BLOCKING_SECONDS)
        yield pa.record_batch(
            [
                pa.array(["kb1"]),
                pa.array(["/tmp/user_1/kb1/a.pdf"]),
                pa.array(["doc-1"]),
                pa.array(["file-1"]),
                pa.array([1]),
            ],
            names=["collection", "source_path", "doc_id", "file_id", "user_id"],
        )

    def aggregate_collection_stats(self, *, user_id, is_admin):
        time.sleep(BLOCKING_SECONDS)
        return {"kb1": {"documents": 1, "parses": 1, "chunks": 1, "embeddings": 1}}


class _EmptyMetadataStore:
    async def list_collections(self, *, user_id, is_admin):
        return []

    async def save_collections(self, infos):
        return None


@pytest.mark.asyncio
async def test_list_collections_keeps_event_loop_responsive(monkeypatch):
    monkeypatch.setattr(
        collections_module, "get_vector_index_store", lambda: _SlowVectorStore()
    )
    monkeypatch.setattr(
        collections_module, "get_metadata_store", lambda: _EmptyMetadataStore()
    )

    ticks = 0

    async def heartbeat() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            ticks += 1

    beat = asyncio.create_task(heartbeat())
    try:
        result = await collections_module._list_collections_impl(
            user_id=1, is_admin=False
        )
    finally:
        beat.cancel()

    assert result.status == "success"
    assert [collection.name for collection in result.collections] == ["kb1"]
    # Two 0.5s blocking calls: a responsive loop ticks tens of times. Held on
    # the event loop thread it manages one or two.
    assert ticks >= 20, f"event loop was blocked, only {ticks} ticks"


@pytest.mark.asyncio
async def test_document_row_scan_runs_off_the_event_loop(monkeypatch):
    """The scan helper must be plain blocking code, dispatched to a thread."""
    scan_thread_ids: list[int] = []

    class _ThreadRecordingStore(_SlowVectorStore):
        def iter_batches(self, **kwargs):
            import threading

            scan_thread_ids.append(threading.get_ident())
            return super().iter_batches(**kwargs)

    monkeypatch.setattr(
        collections_module, "get_vector_index_store", lambda: _ThreadRecordingStore()
    )
    monkeypatch.setattr(
        collections_module, "get_metadata_store", lambda: _EmptyMetadataStore()
    )

    import threading

    loop_thread_id = threading.get_ident()
    await collections_module._list_collections_impl(user_id=1, is_admin=False)

    assert scan_thread_ids, "documents table was never scanned"
    assert loop_thread_id not in scan_thread_ids
