"""Stats rebuild must not block the event loop on LanceDB aggregation.

``_rebuild_collection_stats_impl`` has an async signature, but
``aggregate_collection_stats`` is synchronous LanceDB work. Executed on the event
loop it stops every other coroutine in the process, which on a single-worker
deployment stalls the whole API, ``/health`` included, for the scan's duration.
This is the same failure mode already fixed for collection listing.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo
from xagent.core.tools.core.RAG_tools.management import (
    collection_manager as collection_manager_module,
)

BLOCKING_SECONDS = 0.5
HEARTBEAT_INTERVAL = 0.02
COLLECTION_NAME = "stats_rebuild_event_loop"


class _SlowVectorStore:
    """Blocks the calling thread the way a real LanceDB aggregation does."""

    def __init__(self) -> None:
        self.thread_ids: list[int] = []

    def aggregate_collection_stats(self, *, user_id, is_admin):
        assert user_id is None
        assert is_admin is True
        self.thread_ids.append(threading.get_ident())
        time.sleep(BLOCKING_SECONDS)
        return {
            COLLECTION_NAME: {
                "documents": 2,
                "parses": 2,
                "chunks": 4,
                "embeddings": 4,
            }
        }


class _StubMetadataStore:
    async def get_collection(self, collection_name: str) -> CollectionInfo | None:
        return CollectionInfo(name=collection_name)


@pytest.fixture
def slow_vector_store(monkeypatch: pytest.MonkeyPatch) -> _SlowVectorStore:
    store = _SlowVectorStore()
    saved: list[CollectionInfo] = []

    async def _save_collection(collection: CollectionInfo) -> None:
        saved.append(collection)

    monkeypatch.setattr(
        collection_manager_module, "get_vector_index_store", lambda: store
    )
    monkeypatch.setattr(
        collection_manager_module, "get_metadata_store", lambda: _StubMetadataStore()
    )
    monkeypatch.setattr(
        collection_manager_module.collection_manager,
        "save_collection",
        _save_collection,
    )
    return store


async def test_rebuild_collection_stats_keeps_event_loop_responsive(
    slow_vector_store: _SlowVectorStore,
) -> None:
    """Given a slow aggregation, the stats rebuild leaves the loop responsive."""
    ticks = 0

    async def heartbeat() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            ticks += 1

    beat = asyncio.create_task(heartbeat())
    try:
        rebuilt = await collection_manager_module._rebuild_collection_stats_impl(
            COLLECTION_NAME
        )
    finally:
        beat.cancel()

    assert rebuilt is not None
    assert rebuilt.documents == 2
    assert rebuilt.chunks == 4
    # A 0.5s blocking call: a responsive loop ticks tens of times. Held on the
    # event loop thread it manages one or two.
    assert ticks >= 15, f"event loop was blocked, only {ticks} ticks"


async def test_rebuild_collection_stats_aggregates_off_the_event_loop(
    slow_vector_store: _SlowVectorStore,
) -> None:
    """The aggregation is plain blocking code and must run in a worker thread."""
    loop_thread_id = threading.get_ident()

    await collection_manager_module._rebuild_collection_stats_impl(COLLECTION_NAME)

    assert slow_vector_store.thread_ids, "stats were never aggregated"
    assert loop_thread_id not in slow_vector_store.thread_ids


def test_rebuild_collection_stats_sync_still_works(
    slow_vector_store: _SlowVectorStore,
) -> None:
    """The sync wrapper keeps working once aggregation moves to a thread."""
    rebuilt = collection_manager_module._rebuild_collection_stats_sync_impl(
        COLLECTION_NAME
    )

    assert rebuilt is not None
    assert rebuilt.documents == 2
    assert rebuilt.chunks == 4
