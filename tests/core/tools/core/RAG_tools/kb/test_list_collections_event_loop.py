"""Collection listing must not block the event loop on LanceDB scans.

``_list_collections_impl`` has an async signature, but every piece of LanceDB
work it drives is synchronous. Executed on the event loop it stops every other
coroutine in the process, which on a single-worker deployment stalls the whole
API, ``/health`` included, for the scan's duration.

The blocking calls live at two layers, and each is asserted where it belongs:

* The vector-store scans (``iter_batches`` behind ``_scan_document_rows``, and
  ``aggregate_collection_stats``) are plain sync methods, so
  ``_list_collections_impl`` dispatches them itself.
* The metadata-store calls (``list_collections``, ``save_collections``,
  ``get_collection_config``) are ``async def`` on the ``MetadataStore`` ABC, so
  the caller *cannot* wrap them: ``asyncio.to_thread`` on a coroutine function
  hands the worker thread a coroutine object and never runs it. Their dispatch
  lives inside ``LanceDBMetadataStore`` and is asserted against that class.

Every blocking call is scored **individually**. A single cumulative tick total
cannot tell a fully-fixed run from one that dropped a single dispatch, because
the calls that still yield donate far more than enough idle time to clear the
threshold on their own.
"""

import asyncio
import threading
import time

import pyarrow as pa
import pytest

from xagent.core.tools.core.RAG_tools.management import (
    collections as collections_module,
)
from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
    LanceDBMetadataStore,
)

BLOCKING_SECONDS = 0.3
HEARTBEAT_INTERVAL = 0.01
# A 0.3s call that yields the loop leaves room for ~30 ticks; one that holds the
# loop thread leaves room for zero or one.
MIN_TICKS_PER_CALL = 10


class _Heartbeat:
    """Tick counter that scores each blocking call over its own window."""

    def __init__(self) -> None:
        self.ticks = 0
        self.ticks_per_call: dict[str, int] = {}
        self.thread_per_call: dict[str, int] = {}
        self._task: asyncio.Task | None = None

    async def __aenter__(self) -> "_Heartbeat":
        async def beat() -> None:
            while True:
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                self.ticks += 1

        self._task = asyncio.create_task(beat())
        return self

    async def __aexit__(self, *_exc: object) -> None:
        assert self._task is not None
        self._task.cancel()

    def block(self, label: str) -> None:
        """Block the calling thread the way a real LanceDB call does."""
        before = self.ticks
        self.thread_per_call[label] = threading.get_ident()
        time.sleep(BLOCKING_SECONDS)
        self.ticks_per_call[label] = self.ticks - before

    def assert_yielded_the_loop(self, *labels: str) -> None:
        """Assert each named call ran off-loop and left the loop running.

        Must be awaited from the event loop thread so ``get_ident`` identifies
        the loop rather than a worker.
        """
        loop_thread = threading.get_ident()
        for label in labels:
            assert label in self.ticks_per_call, f"{label} was never called"
            assert self.thread_per_call[label] != loop_thread, (
                f"{label} ran on the event loop thread"
            )
            assert self.ticks_per_call[label] >= MIN_TICKS_PER_CALL, (
                f"{label} blocked the event loop: only "
                f"{self.ticks_per_call[label]} ticks elapsed during its call"
            )


# --------------------------------------------------------------------------
# Layer 1: scans dispatched by _list_collections_impl itself
# --------------------------------------------------------------------------


class _SlowVectorStore:
    """Blocks the calling thread the way a real LanceDB scan does."""

    def __init__(self, heartbeat: _Heartbeat) -> None:
        self._heartbeat = heartbeat

    def iter_batches(self, *, table_name, columns, user_id, is_admin):
        self._heartbeat.block("iter_batches")
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
        self._heartbeat.block("aggregate_collection_stats")
        return {"kb1": {"documents": 1, "parses": 1, "chunks": 1, "embeddings": 1}}


class _EmptyMetadataStore:
    """Instant metadata store, so only the vector-store calls are measured."""

    async def list_collections(self, *, user_id, is_admin):
        return []

    async def save_collections(self, infos):
        return None

    async def get_collection_config(self, collection, user_id, is_admin=False):
        return None


@pytest.mark.asyncio
async def test_each_vector_store_scan_runs_off_the_event_loop(monkeypatch):
    """Both scans must yield the loop, measured per call rather than in total.

    Scoring the calls separately is the point: with one cumulative assertion,
    dropping either dispatch still passes because the surviving one yields
    enough idle time to cover for it.
    """
    async with _Heartbeat() as heartbeat:
        monkeypatch.setattr(
            collections_module,
            "get_vector_index_store",
            lambda: _SlowVectorStore(heartbeat),
        )
        monkeypatch.setattr(
            collections_module, "get_metadata_store", lambda: _EmptyMetadataStore()
        )

        result = await collections_module._list_collections_impl(
            user_id=1, is_admin=False
        )

    assert result.status == "success"
    assert [collection.name for collection in result.collections] == ["kb1"]
    heartbeat.assert_yielded_the_loop("iter_batches", "aggregate_collection_stats")


@pytest.mark.asyncio
async def test_collection_configs_load_concurrently(monkeypatch):
    """Per-collection config lookups must not be serialised one at a time.

    ``_load_collection_ingestion_configs`` used to await inside a ``for`` loop,
    so its cost scaled linearly with the number of collections.
    """
    collection_count = 8
    per_call_delay = 0.05

    class _SlowConfigStore:
        async def get_collection_config(self, collection, user_id, is_admin=False):
            await asyncio.sleep(per_call_delay)
            return None

    monkeypatch.setattr(
        collections_module, "get_metadata_store", lambda: _SlowConfigStore()
    )

    keys = [f"kb{index}" for index in range(collection_count)]
    started = time.perf_counter()
    await collections_module._load_collection_ingestion_configs(keys, 1, False)
    elapsed = time.perf_counter() - started

    serial_cost = per_call_delay * collection_count
    assert elapsed < serial_cost / 2, (
        f"config lookups look serialised: {elapsed:.3f}s for {collection_count} "
        f"collections (serial would cost ~{serial_cost:.3f}s)"
    )


# --------------------------------------------------------------------------
# Layer 2: metadata-store calls, dispatched inside LanceDBMetadataStore
# --------------------------------------------------------------------------


class _FakeSearch:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def where(self, _clause: str) -> "_FakeSearch":
        return self

    def to_arrow(self):
        if not self._rows:
            return pa.table({})
        return pa.Table.from_pylist(self._rows)


class _FakeMergeInsert:
    def __init__(self, heartbeat: _Heartbeat, label: str) -> None:
        self._heartbeat = heartbeat
        self._label = label

    def when_matched_update_all(self) -> "_FakeMergeInsert":
        return self

    def when_not_matched_insert_all(self) -> "_FakeMergeInsert":
        return self

    def execute(self, _rows) -> None:
        self._heartbeat.block(self._label)


class _FakeTable:
    def __init__(self, heartbeat: _Heartbeat, label: str, rows: list[dict]) -> None:
        self._heartbeat = heartbeat
        self._label = label
        self._rows = rows
        # Present "owners" so the back-compat column patch is skipped.
        self.schema = pa.schema([("owners", pa.string())])

    def search(self) -> _FakeSearch:
        self._heartbeat.block(self._label)
        return _FakeSearch(self._rows)

    def merge_insert(self, _keys) -> _FakeMergeInsert:
        return _FakeMergeInsert(self._heartbeat, self._label)

    def close(self) -> None:
        return None


class _FakeConnection:
    """Minimal LanceDB connection whose table work blocks its caller."""

    def __init__(
        self, heartbeat: _Heartbeat, label: str, rows: list[dict] | None = None
    ) -> None:
        self._heartbeat = heartbeat
        self._label = label
        self._rows = list(rows or [])

    def list_tables(self) -> list[str]:
        # Both tables already exist, so ensure_* short-circuits before create.
        return ["collection_metadata", "collection_config"]

    def open_table(self, _name: str) -> _FakeTable:
        return _FakeTable(self._heartbeat, self._label, self._rows)


def _store_with(connection: _FakeConnection) -> LanceDBMetadataStore:
    store = LanceDBMetadataStore()
    store._conn = connection
    return store


@pytest.mark.asyncio
async def test_metadata_store_list_collections_runs_off_the_event_loop():
    async with _Heartbeat() as heartbeat:
        store = _store_with(_FakeConnection(heartbeat, "list_collections"))
        await store.list_collections(user_id=None, is_admin=True)

    heartbeat.assert_yielded_the_loop("list_collections")


@pytest.mark.asyncio
async def test_metadata_store_save_collections_runs_off_the_event_loop():
    from xagent.core.tools.core.RAG_tools.storage.contracts import CollectionInfo

    async with _Heartbeat() as heartbeat:
        store = _store_with(_FakeConnection(heartbeat, "save_collections"))
        await store.save_collections([CollectionInfo(name="kb1")])

    heartbeat.assert_yielded_the_loop("save_collections")


@pytest.mark.asyncio
async def test_metadata_store_get_collection_config_runs_off_the_event_loop():
    async with _Heartbeat() as heartbeat:
        store = _store_with(_FakeConnection(heartbeat, "get_collection_config"))
        await store.get_collection_config("kb1", 1, is_admin=False)

    heartbeat.assert_yielded_the_loop("get_collection_config")


@pytest.mark.asyncio
async def test_document_row_scan_runs_off_the_event_loop(monkeypatch):
    """The scan helper must be plain blocking code, dispatched to a thread."""
    scan_thread_ids: list[int] = []

    class _ThreadRecordingStore:
        def iter_batches(self, **kwargs):
            scan_thread_ids.append(threading.get_ident())
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
            return {"kb1": {"documents": 1, "parses": 1, "chunks": 1, "embeddings": 1}}

    monkeypatch.setattr(
        collections_module, "get_vector_index_store", lambda: _ThreadRecordingStore()
    )
    monkeypatch.setattr(
        collections_module, "get_metadata_store", lambda: _EmptyMetadataStore()
    )

    loop_thread_id = threading.get_ident()
    await collections_module._list_collections_impl(user_id=1, is_admin=False)

    assert scan_thread_ids, "documents table was never scanned"
    assert loop_thread_id not in scan_thread_ids
