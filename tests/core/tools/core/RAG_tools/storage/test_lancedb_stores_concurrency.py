"""The store data plane must survive concurrent handle access.

The coordinator offloads blocking handle calls with ``asyncio.to_thread``. Once
those are awaited concurrently on a shared loop, several worker threads reach a
single process-wide store instance at the same time. These tests pin the two
pieces of *synchronous* shared state that made that unsafe: the table-handle
cache, and the per-instance sync connection cache that no longer exists.

Async connection init is covered separately, in
``tests/providers/vector_store/test_lancedb_async_pool.py``: it moved to a
process-wide pool once the per-instance ``asyncio.Lock`` guarding it turned
out to deadlock across event loops (#2200).

The table-cache assertion is a conservation law rather than a race detector:
every handle ``open_table`` returns must end up either in the cache or closed.
What actually trips these tests on an unguarded cache is the crash -- a
``move_to_end`` racing a concurrent ``clear``/``pop`` raises ``KeyError``.
The conservation law is what catches a dropped ``_safe_close_table``.

Both locks are load-bearing: dropping either one loses the conservation law,
because an unlocked insert can land inside the other's critical section and be
erased by its ``clear()`` without ever appearing in its stale snapshot.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from typing import Any, List
from unittest.mock import Mock, patch

import pytest

from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
    LanceDBIngestionStatusStore,
    LanceDBMainPointerStore,
    LanceDBMetadataStore,
    LanceDBPromptTemplateStore,
    LanceDBVectorIndexStore,
)


@pytest.fixture(autouse=True)
def _tight_switch_interval():
    """Force frequent GIL handoffs so the narrow race windows get hit.

    The insert-section window is a few bytecodes wide; at the default 5ms
    interval a missing lock there slips through most runs.
    """
    original = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        yield
    finally:
        sys.setswitchinterval(original)


class _FakeTable:
    def __init__(self, name: str, *, close_delay: float = 0.0) -> None:
        self.name = name
        self.close_count = 0
        self._close_delay = close_delay

    @property
    def closed(self) -> bool:
        return self.close_count > 0

    def close(self) -> None:
        # A slow close widens the window an unguarded invalidate leaves open
        # between snapshotting the cache and clearing it.
        time.sleep(self._close_delay)
        self.close_count += 1


class _RacyConnection:
    """Hands out a distinct handle per ``open_table``, slowly enough to overlap."""

    def __init__(self, *, delay: float = 0.005, close_delay: float = 0.0) -> None:
        self._delay = delay
        self._close_delay = close_delay
        self._lock = threading.Lock()
        self.opened: List[_FakeTable] = []

    def open_table(self, name: str) -> _FakeTable:
        # The delay is what makes the interleaving reproducible rather than rare.
        time.sleep(self._delay)
        table = _FakeTable(name, close_delay=self._close_delay)
        with self._lock:
            self.opened.append(table)
        return table


def _store_on(connection: _RacyConnection) -> LanceDBVectorIndexStore:
    store = LanceDBVectorIndexStore()
    store._get_connection = lambda: connection  # type: ignore[method-assign]
    return store


def _assert_every_handle_cached_or_closed(
    connection: _RacyConnection, store: LanceDBVectorIndexStore
) -> None:
    cached = set(id(table) for table in store._table_cache.values())
    leaked = [
        table
        for table in connection.opened
        if id(table) not in cached and not table.closed
    ]
    assert not leaked, (
        f"{len(leaked)} of {len(connection.opened)} opened handles are neither "
        f"cached nor closed"
    )
    twice = [table for table in connection.opened if table.close_count > 1]
    assert not twice, f"{len(twice)} handles were closed more than once"
    live = [table for table in connection.opened if id(table) in cached]
    assert not [t for t in live if t.closed], "a cached handle was closed"


def test_concurrent_open_of_one_table_leaks_no_handle() -> None:
    """Threads racing on the same table must not drop an opened handle."""
    connection = _RacyConnection()
    store = _store_on(connection)
    barrier = threading.Barrier(8)
    errors: List[BaseException] = []

    def worker() -> None:
        try:
            barrier.wait(timeout=30)
            store._get_table("documents")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, f"concurrent _get_table raised: {errors}"
    assert len(store._table_cache) == 1
    _assert_every_handle_cached_or_closed(connection, store)


def test_concurrent_get_and_invalidate_leaks_no_handle() -> None:
    """Interleaved cache fills and invalidations must stay consistent.

    Dropping ``invalidate_table_cache``'s lock fails this every run. Dropping
    the insert-section lock fails it most runs, but not all: that window is a
    few bytecodes wide, and what makes it reachable at all is the autouse
    ``sys.setswitchinterval`` fixture. Duration alone does nothing at the
    default interval; paired with the tight one it does, hence 1.5s here.
    """
    connection = _RacyConnection(delay=0.001, close_delay=0.002)
    store = _store_on(connection)
    names = [f"table_{index}" for index in range(48)]
    errors: List[BaseException] = []
    stop = threading.Event()

    def filler(name: str) -> None:
        try:
            while not stop.is_set():
                store._get_table(name)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def invalidator() -> None:
        try:
            while not stop.is_set():
                store.invalidate_table_cache()
                store.invalidate_table_cache("table_0")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=filler, args=(name,)) for name in names]
    threads += [threading.Thread(target=invalidator) for _ in range(4)]
    for thread in threads:
        thread.start()
    time.sleep(1.5)
    stop.set()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, f"concurrent cache access raised: {errors}"
    _assert_every_handle_cached_or_closed(connection, store)


def test_invalidation_during_open_does_not_cache_a_stale_handle() -> None:
    """A drop landing mid-open must not leave the dropped table cached.

    Deterministic rather than timing-based: the connection invalidates the
    cache from inside ``open_table``, which is exactly the interleaving a
    concurrent drop produces -- the cache was empty when the invalidation ran,
    so nothing there marks the in-flight handle as stale.
    """
    connection = _RacyConnection(delay=0.0)
    store = _store_on(connection)
    plain_open = connection.open_table
    fired: List[bool] = []

    def open_then_invalidate(name: str) -> _FakeTable:
        table = plain_open(name)
        if not fired:
            fired.append(True)
            store.invalidate_table_cache(name)
        return table

    connection.open_table = open_then_invalidate  # type: ignore[method-assign]

    handle = store._get_table("documents")

    assert len(connection.opened) == 2, "the raced handle must be re-opened"
    assert connection.opened[0].closed, "the raced handle must not be leaked"
    assert handle is connection.opened[1]
    assert store._table_cache["documents"] is connection.opened[1]
    _assert_every_handle_cached_or_closed(connection, store)


def test_cache_never_exceeds_maxsize_under_concurrency() -> None:
    """LRU eviction must hold when many threads fill the cache at once."""
    connection = _RacyConnection(delay=0.0)
    store = _store_on(connection)
    over = store._TABLE_CACHE_MAXSIZE * 2
    errors: List[BaseException] = []

    def worker(index: int) -> None:
        try:
            store._get_table(f"table_{index}")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(over)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, f"concurrent cache fill raised: {errors}"
    assert len(store._table_cache) <= store._TABLE_CACHE_MAXSIZE
    _assert_every_handle_cached_or_closed(connection, store)


@pytest.mark.asyncio
async def test_concurrent_to_thread_handle_calls_share_no_mutable_connection() -> None:
    """The shape #657 names: many to_thread storage calls on one shared loop."""
    connection = _RacyConnection(delay=0.002)
    store = _store_on(connection)

    results = await asyncio.gather(
        *(asyncio.to_thread(store._get_table, "documents") for _ in range(12))
    )

    assert all(table is results[0] for table in results)
    assert len(store._table_cache) == 1
    _assert_every_handle_cached_or_closed(connection, store)


@pytest.mark.parametrize(
    ("store_class", "getter_name", "cache_attr"),
    [
        (LanceDBMetadataStore, "get_raw_connection", "_conn"),
        (LanceDBVectorIndexStore, "_get_connection", "_conn"),
        (LanceDBIngestionStatusStore, "_get_sync_connection", "_sync_conn"),
        (LanceDBPromptTemplateStore, "_get_sync_connection", "_sync_conn"),
        (LanceDBMainPointerStore, "_get_sync_connection", "_sync_conn"),
    ],
)
@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_connection_is_not_cached_on_the_instance(
    mock_get_connection: Mock,
    store_class: type,
    getter_name: str,
    cache_attr: str,
) -> None:
    """Connections come from the process-wide pool, which holds its own lock.

    A per-instance cache would be unguarded shared state and would also outlive
    both the pool's TTL and ``clear_connection_cache()``. Every store that was
    carrying one is covered, so re-adding a cache to any of them fails here.
    """
    conn: Any = Mock()
    mock_get_connection.return_value = conn
    store = store_class()
    getter = getattr(store, getter_name)

    assert not hasattr(store, cache_attr)
    assert getter() is conn
    assert getter() is conn
    assert mock_get_connection.call_count == 2
