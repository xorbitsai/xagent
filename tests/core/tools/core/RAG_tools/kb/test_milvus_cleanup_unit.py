"""Leftover Milvus row cleanup without a server: a fake client and a temp ledger.

The Milvus server tests cover the same paths against a real collection.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

import pytest

from xagent.core.tools.core.RAG_tools.core.config import (
    DEFAULT_VECTOR_STORE_DELETE_BATCH_SIZE,
)
from xagent.core.tools.core.RAG_tools.core.exceptions import DatabaseOperationError
from xagent.core.tools.core.RAG_tools.core.schemas import DocumentProcessingStatus
from xagent.core.tools.core.RAG_tools.kb import collection_handle, milvus_cleanup
from xagent.core.tools.core.RAG_tools.kb.collection_handle import milvus_collection_name
from xagent.core.tools.core.RAG_tools.kb.milvus_cleanup import (
    purge_leftover_milvus_rows,
)
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
    _safe_close_table,
    ensure_documents_table,
    ensure_ingestion_runs_table,
    ensure_kb_ids_table,
)
from xagent.core.tools.core.RAG_tools.storage.factory import get_vector_index_store
from xagent.providers.vector_store.milvus import MilvusConnectionManager

MODEL = "cleanup-unit"
COLLECTION = "kb"
NAME = milvus_collection_name(MODEL)


class _NotLoaded(Exception):
    code = 101


class _Iterator:
    def __init__(self, rows: list[dict[str, Any]], batch_size: int) -> None:
        self._rows = rows
        self._batch_size = batch_size
        self._pos = 0

    def next(self) -> list[dict[str, Any]]:
        if self._pos >= len(self._rows):
            return []
        batch = self._rows[self._pos : self._pos + self._batch_size]
        self._pos += self._batch_size
        return batch

    def close(self) -> None:
        return None


class _FakeLedgerTable:
    """A LanceDB-like handle whose row count is fixed at open time."""

    def __init__(self, count: int) -> None:
        self._count = count

    def count_rows(self) -> int:
        return self._count

    def close(self) -> None:
        return None


class _FakeLedgerBatch:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def to_pylist(self) -> list[dict[str, Any]]:
        return self._rows


class _FakeLedgerStore:
    """A ledger store whose first opened handle reports a stale count.

    The stale count would stick on a pinned handle, so the read can only
    finish if every attempt opens (and closes) its own table handles.
    """

    def __init__(self, *, true_count: int, stale_count: int) -> None:
        self._true = true_count
        self._stale = stale_count
        self.opens = 0

    def get_raw_connection(self) -> "_FakeLedgerStore":
        return self

    def list_tables(self) -> list[str]:
        return ["documents"]

    def open_table(self, _name: str) -> _FakeLedgerTable:
        self.opens += 1
        return _FakeLedgerTable(self._stale if self.opens == 1 else self._true)

    def iter_batches(self, **_kwargs: Any) -> Any:
        rows = [{"doc_id": f"d{i}"} for i in range(self._true)]
        return iter([_FakeLedgerBatch(rows)])


class FakeMilvus:
    """A tiny in-memory Milvus: query_iterator filters, delete removes matches."""

    def __init__(self, collections: dict[str, list[dict[str, Any]]]) -> None:
        self.collections = {name: list(rows) for name, rows in collections.items()}
        self.deletes: list[tuple[str, str, dict[str, Any]]] = []
        self.not_loaded: set[str] = set()
        self.delete_not_loaded: set[str] = set()
        self.fail_query: set[str] = set()
        self.before_delete = None

    def list_collections(self) -> list[str]:
        return list(self.collections)

    def describe_collection(self, name: str) -> dict[str, Any]:
        """Enough of a description for ``_with_loaded`` to load the collection."""
        return {
            "properties": {"xagent.model_id": MODEL},
            "fields": [{"name": "dense", "params": {"dim": 8}}],
        }

    def query_iterator(
        self,
        name: str,
        *,
        batch_size: int,
        filter: str,
        output_fields: list[str] | None = None,
        consistency_level: str | None = None,
    ) -> _Iterator:
        if name in self.fail_query:
            raise RuntimeError("boom")
        if name in self.not_loaded:
            raise _NotLoaded("collection is not loaded")
        cutoff = int(filter.split("<", 1)[1].strip())
        return _Iterator(
            [
                dict(row)
                for row in self.collections.get(name, [])
                if row["created_at"] < cutoff
            ],
            batch_size,
        )

    def delete(
        self, name: str, *, filter: str, filter_params: dict[str, Any]
    ) -> dict[str, int]:
        if name in self.delete_not_loaded:
            raise _NotLoaded("collection is not loaded")
        if self.before_delete is not None:
            self.before_delete(name)
        # Scope off the clauses in the expression, not the params, so dropping
        # a clause from the filter is visible to the fake.
        kb_id = filter_params.get("kb_id") if "kb_id ==" in filter else None
        doc_ids = filter_params.get("doc_ids") if "doc_id in" in filter else None
        cutoff = filter_params.get("cutoff") if "created_at <" in filter else None
        invisible_only = "visible == false" in filter
        kept: list[dict[str, Any]] = []
        deleted = 0
        for row in self.collections.get(name, []):
            match = True
            if kb_id is not None and row["kb_id"] != kb_id:
                match = False
            if cutoff is not None and not row["created_at"] < cutoff:
                match = False
            if doc_ids is not None and row["doc_id"] not in doc_ids:
                match = False
            if invisible_only and row["visible"] is not False:
                match = False
            if match:
                deleted += 1
            else:
                kept.append(row)
        self.collections[name] = kept
        self.deletes.append((name, filter, dict(filter_params)))
        return {"delete_count": deleted}


def _row(
    chunk_id: str,
    kb_id: str,
    doc_id: str,
    *,
    user_id: int | None = 1,
    visible: bool = False,
    created_at: int,
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "kb_id": kb_id,
        "doc_id": doc_id,
        "user_id": user_id,
        "visible": visible,
        "created_at": created_at,
    }


def _conn() -> Any:
    return get_vector_index_store().get_raw_connection()


def _seed_kb_id(kb_id: str, collection: str, user_id: int | None) -> None:
    conn = _conn()
    ensure_kb_ids_table(conn)
    table = conn.open_table("kb_ids")
    try:
        table.add(
            [
                {
                    "collection": collection,
                    "user_id": user_id,
                    "kb_id": kb_id,
                    "created_at": datetime.now(timezone.utc),
                }
            ]
        )
    finally:
        _safe_close_table(table)


def _seed_document(collection: str, doc_id: str, user_id: int | None) -> None:
    store = get_vector_index_store()
    conn = store.get_raw_connection()
    ensure_documents_table(conn)
    store.upsert_documents(
        [
            {
                "collection": collection,
                "doc_id": doc_id,
                "file_id": None,
                "source_path": None,
                "file_type": "txt",
                "content_hash": "hash",
                "uploaded_at": datetime.now(timezone.utc),
                "title": None,
                "language": None,
                "user_id": user_id,
            }
        ]
    )


def _run_row(
    collection: str,
    doc_id: str,
    user_id: int | None,
    *,
    status: str,
    updated_at: datetime,
) -> dict[str, Any]:
    return {
        "collection": collection,
        "doc_id": doc_id,
        "status": status,
        "message": "",
        "parse_hash": "",
        "created_at": updated_at,
        "updated_at": updated_at,
        "user_id": user_id,
    }


def _seed_runs(rows: list[dict[str, Any]]) -> None:
    # One add keeps the listed row order, so a last-wins reader sees the tail.
    conn = _conn()
    ensure_ingestion_runs_table(conn)
    table = conn.open_table("ingestion_runs")
    try:
        table.add(rows)
    finally:
        _safe_close_table(table)


def _seed_run(
    collection: str,
    doc_id: str,
    user_id: int | None,
    *,
    status: str,
    updated_at: datetime,
) -> None:
    _seed_runs(
        [_run_row(collection, doc_id, user_id, status=status, updated_at=updated_at)]
    )


@pytest.fixture(autouse=True)
def milvus_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAGENT_VECTOR_BACKEND", "milvus")


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeMilvus:
    client = FakeMilvus({NAME: []})
    monkeypatch.setattr(
        MilvusConnectionManager, "get_shared_client_from_env", lambda _self: client
    )
    return client


def _now() -> float:
    return time.time()


def test_lancedb_engine_returns_empty_before_touching_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_kb_id("kb-1", COLLECTION, 1)
    monkeypatch.setenv("XAGENT_VECTOR_BACKEND", "lancedb")
    touched: list[str] = []
    monkeypatch.setattr(
        MilvusConnectionManager,
        "get_shared_client_from_env",
        lambda _self: touched.append("client"),
    )

    assert purge_leftover_milvus_rows() == {}
    assert touched == []


@pytest.mark.parametrize(
    "status",
    [
        DocumentProcessingStatus.SUCCESS.value,
        DocumentProcessingStatus.FAILED.value,
        DocumentProcessingStatus.CANCELLED.value,
    ],
)
def test_an_invisible_row_with_an_old_terminal_run_is_deleted(
    fake: FakeMilvus, status: str
) -> None:
    now = _now()
    _seed_kb_id("kb-1", COLLECTION, 1)
    _seed_run(
        COLLECTION,
        "doc",
        1,
        status=status,
        updated_at=datetime.fromtimestamp(now - 3600, tz=timezone.utc),
    )
    fake.collections[NAME] = [_row("c1", "kb-1", "doc", created_at=int(now) - 3600)]

    assert purge_leftover_milvus_rows(grace_seconds=60, now=now) == {NAME: 1}
    assert fake.collections[NAME] == []


@pytest.mark.parametrize(
    "status",
    [
        DocumentProcessingStatus.RUNNING.value,
        DocumentProcessingStatus.PENDING.value,
        DocumentProcessingStatus.CHUNKED.value,
        DocumentProcessingStatus.PARTIALLY_EMBEDDED.value,
    ],
)
def test_an_invisible_row_with_a_non_terminal_run_is_kept(
    fake: FakeMilvus, status: str
) -> None:
    now = _now()
    _seed_kb_id("kb-1", COLLECTION, 1)
    _seed_run(
        COLLECTION,
        "doc",
        1,
        status=status,
        updated_at=datetime.fromtimestamp(now - 3600, tz=timezone.utc),
    )
    fake.collections[NAME] = [_row("c1", "kb-1", "doc", created_at=int(now) - 3600)]

    assert purge_leftover_milvus_rows(grace_seconds=60, now=now) == {}
    assert len(fake.collections[NAME]) == 1


@pytest.mark.parametrize(
    "updated_at", [None, "not-a-date"], ids=["missing", "unparsable"]
)
def test_a_run_row_without_a_usable_updated_at_is_not_purgeable(
    updated_at: Any,
) -> None:
    rows = [
        {
            "collection": COLLECTION,
            "doc_id": "doc",
            "status": DocumentProcessingStatus.SUCCESS.value,
            "updated_at": updated_at,
        }
    ]

    # The key stays out, with no exception, so the uncommitted check cannot fire.
    assert milvus_cleanup._purgeable_run_keys(rows, cutoff=10**12) == set()


def test_an_old_terminal_run_row_is_purgeable() -> None:
    rows = [
        {
            "collection": COLLECTION,
            "doc_id": "doc",
            "status": DocumentProcessingStatus.SUCCESS.value,
            "updated_at": 100,
        }
    ]

    assert milvus_cleanup._purgeable_run_keys(rows, cutoff=200) == {(COLLECTION, "doc")}


def test_an_invisible_row_with_a_terminal_run_inside_the_grace_is_kept(
    fake: FakeMilvus,
) -> None:
    now = _now()
    _seed_kb_id("kb-1", COLLECTION, 1)
    _seed_run(
        COLLECTION,
        "doc",
        1,
        status="failed",
        updated_at=datetime.fromtimestamp(now - 30, tz=timezone.utc),
    )
    fake.collections[NAME] = [_row("c1", "kb-1", "doc", created_at=int(now) - 3600)]

    assert purge_leftover_milvus_rows(grace_seconds=60, now=now) == {}
    assert len(fake.collections[NAME]) == 1


def test_an_invisible_row_with_no_run_row_but_a_document_is_kept(
    fake: FakeMilvus,
) -> None:
    now = _now()
    _seed_kb_id("kb-1", COLLECTION, 1)
    _seed_document(COLLECTION, "doc", 1)
    fake.collections[NAME] = [_row("c1", "kb-1", "doc", created_at=int(now) - 3600)]

    assert purge_leftover_milvus_rows(grace_seconds=60, now=now) == {}
    assert len(fake.collections[NAME]) == 1


@pytest.mark.parametrize("visible", [True, False])
def test_a_row_missing_from_documents_is_deleted(
    fake: FakeMilvus, visible: bool
) -> None:
    now = _now()
    _seed_kb_id("kb-1", COLLECTION, 1)
    _seed_document(COLLECTION, "other", 1)
    fake.collections[NAME] = [
        _row("c1", "kb-1", "gone", visible=visible, created_at=int(now) - 3600)
    ]

    assert purge_leftover_milvus_rows(grace_seconds=60, now=now) == {NAME: 1}
    assert fake.collections[NAME] == []


def test_a_missing_document_delete_spares_a_present_document_of_the_same_kb(
    fake: FakeMilvus,
) -> None:
    now = _now()
    _seed_kb_id("kb-1", COLLECTION, 1)
    _seed_document(COLLECTION, "kept", 1)
    fake.collections[NAME] = [
        _row("c1", "kb-1", "gone", created_at=int(now) - 3600),
        _row("c2", "kb-1", "kept", visible=True, created_at=int(now) - 3600),
    ]

    assert purge_leftover_milvus_rows(grace_seconds=60, now=now) == {NAME: 1}
    assert [row["chunk_id"] for row in fake.collections[NAME]] == ["c2"]


def test_a_missing_document_row_rewritten_before_the_delete_is_spared(
    fake: FakeMilvus,
) -> None:
    now = _now()
    _seed_kb_id("kb-1", COLLECTION, 1)
    _seed_document(COLLECTION, "kept", 1)
    fake.collections[NAME] = [_row("c1", "kb-1", "gone", created_at=int(now) - 3600)]

    def rewrite(_name: str) -> None:
        fake.collections[NAME][0]["created_at"] = int(now)
        fake.before_delete = None

    fake.before_delete = rewrite

    assert purge_leftover_milvus_rows(grace_seconds=60, now=now) == {}
    assert len(fake.collections[NAME]) == 1


def test_a_row_of_a_kb_id_absent_from_the_ledger_is_deleted(
    fake: FakeMilvus,
) -> None:
    now = _now()
    _seed_kb_id("kb-1", COLLECTION, 1)
    fake.collections[NAME] = [
        _row("c1", "kb-gone", "doc", visible=True, created_at=int(now) - 3600)
    ]

    assert purge_leftover_milvus_rows(grace_seconds=60, now=now) == {NAME: 1}
    assert fake.collections[NAME] == []


def test_an_orphan_delete_leaves_a_live_kb_id_with_the_same_doc_id(
    fake: FakeMilvus,
) -> None:
    now = _now()
    _seed_kb_id("kb-live", COLLECTION, 1)
    _seed_document(COLLECTION, "shared", 1)
    # Both kb_ids hold an old row of the same doc_id; only the orphan's row is
    # the cleanup's, so a delete that dropped its kb_id clause would take both.
    fake.collections[NAME] = [
        _row(
            "c-orphan", "kb-orphan", "shared", visible=True, created_at=int(now) - 3600
        ),
        _row("c-live", "kb-live", "shared", visible=True, created_at=int(now) - 3600),
    ]

    assert purge_leftover_milvus_rows(grace_seconds=60, now=now) == {NAME: 1}
    assert [row["chunk_id"] for row in fake.collections[NAME]] == ["c-live"]


def test_an_empty_kb_ids_ledger_deletes_nothing(
    fake: FakeMilvus, caplog: pytest.LogCaptureFixture
) -> None:
    now = _now()
    fake.collections[NAME] = [
        _row("c1", "kb-gone", "doc", visible=True, created_at=int(now) - 3600)
    ]

    with caplog.at_level("ERROR"):
        assert purge_leftover_milvus_rows(grace_seconds=60, now=now) == {}

    assert len(fake.collections[NAME]) == 1
    assert fake.deletes == []
    assert "kb_ids" in caplog.text


def test_a_scan_error_still_aborts_when_the_ledger_has_no_kb_ids(
    fake: FakeMilvus,
) -> None:
    now = _now()
    fake.collections[NAME] = [
        _row("c1", "kb-gone", "doc", visible=True, created_at=int(now) - 3600)
    ]
    fake.fail_query = {NAME}

    # The empty-kb_ids early return must not swallow the scan failure.
    with pytest.raises(RuntimeError):
        purge_leftover_milvus_rows(grace_seconds=60, now=now)

    assert len(fake.collections[NAME]) == 1
    assert fake.deletes == []


@pytest.mark.parametrize(
    "collections",
    [{}, {NAME: []}],
    ids=["no-collections", "empty-collection"],
)
def test_no_scanned_rows_returns_before_reading_the_ledger(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    collections: dict[str, list[dict[str, Any]]],
) -> None:
    now = _now()
    _seed_kb_id("kb-1", COLLECTION, 1)
    client = FakeMilvus(collections)
    monkeypatch.setattr(
        MilvusConnectionManager, "get_shared_client_from_env", lambda _self: client
    )
    reads: list[str] = []
    real_read = milvus_cleanup._read_rows

    def read(*args: Any, **kwargs: Any) -> Any:
        reads.append("ledger")
        return real_read(*args, **kwargs)

    monkeypatch.setattr(milvus_cleanup, "_read_rows", read)

    with caplog.at_level("ERROR"):
        assert purge_leftover_milvus_rows(grace_seconds=60, now=now) == {}

    assert reads == []
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_many_rows_of_one_document_are_one_candidate_and_one_delete(
    fake: FakeMilvus,
) -> None:
    now = _now()
    _seed_kb_id("kb-1", COLLECTION, 1)
    _seed_run(
        COLLECTION,
        "doc",
        1,
        status="success",
        updated_at=datetime.fromtimestamp(now - 3600, tz=timezone.utc),
    )
    rows = [_row(f"c{i}", "kb-1", "doc", created_at=int(now) - 3600) for i in range(49)]
    # A visible row last: a last-row-wins aggregation would drop the candidate,
    # and the delete must still spare this committed row.
    rows.append(
        _row("c-visible", "kb-1", "doc", visible=True, created_at=int(now) - 3600)
    )
    fake.collections[NAME] = rows

    assert purge_leftover_milvus_rows(grace_seconds=60, now=now) == {NAME: 49}
    assert [row["chunk_id"] for row in fake.collections[NAME]] == ["c-visible"]
    assert len(fake.deletes) == 1


@pytest.mark.parametrize("invisible_last", [True, False])
def test_an_invisible_row_in_a_later_batch_still_marks_its_document(
    fake: FakeMilvus, monkeypatch: pytest.MonkeyPatch, invisible_last: bool
) -> None:
    now = _now()
    _seed_kb_id("kb-1", COLLECTION, 1)
    _seed_run(
        COLLECTION,
        "doc",
        1,
        status="success",
        updated_at=datetime.fromtimestamp(now - 3600, tz=timezone.utc),
    )
    visible = _row("c-visible", "kb-1", "doc", visible=True, created_at=int(now) - 3600)
    invisible = _row("c-invisible", "kb-1", "doc", created_at=int(now) - 3600)
    fake.collections[NAME] = (
        [visible, invisible] if invisible_last else [invisible, visible]
    )
    # One row per batch: the aggregate must OR the batches, not keep the last.
    monkeypatch.setattr(milvus_cleanup, "_CLEANUP_ROW_BATCH", 1)

    assert purge_leftover_milvus_rows(grace_seconds=60, now=now) == {NAME: 1}
    assert [row["chunk_id"] for row in fake.collections[NAME]] == ["c-visible"]


def test_a_missing_documents_table_skips_only_the_document_check(
    fake: FakeMilvus,
) -> None:
    now = _now()
    old = datetime.fromtimestamp(now - 3600, tz=timezone.utc)
    _seed_kb_id("kb-1", COLLECTION, 1)
    _seed_run(COLLECTION, "doc", 1, status="success", updated_at=old)
    _seed_run(COLLECTION, "running", 1, status="running", updated_at=old)
    # If a missing table were read as empty, the running and run-less rows would
    # look missing and vanish; skipping the check must keep them.
    fake.collections[NAME] = [
        _row("c1", "kb-1", "doc", created_at=int(now) - 3600),
        _row("c2", "kb-1", "running", visible=True, created_at=int(now) - 3600),
        _row("c3", "kb-1", "unknown", visible=True, created_at=int(now) - 3600),
    ]

    assert purge_leftover_milvus_rows(grace_seconds=60, now=now) == {NAME: 1}
    assert [row["chunk_id"] for row in fake.collections[NAME]] == ["c2", "c3"]


def test_an_empty_documents_table_skips_only_the_document_check(
    fake: FakeMilvus, caplog: pytest.LogCaptureFixture
) -> None:
    now = _now()
    _seed_kb_id("kb-1", COLLECTION, 1)
    ensure_documents_table(_conn())
    _seed_run(
        COLLECTION,
        "doc",
        1,
        status="success",
        updated_at=datetime.fromtimestamp(now - 3600, tz=timezone.utc),
    )
    fake.collections[NAME] = [
        _row("c1", "kb-1", "doc", created_at=int(now) - 3600),
        _row("c2", "kb-1", "unknown", visible=True, created_at=int(now) - 3600),
    ]

    with caplog.at_level("WARNING"):
        assert purge_leftover_milvus_rows(grace_seconds=60, now=now) == {NAME: 1}

    assert [row["chunk_id"] for row in fake.collections[NAME]] == ["c2"]
    assert "empty" in caplog.text


def test_a_missing_ingestion_runs_table_skips_the_uncommitted_check(
    fake: FakeMilvus,
) -> None:
    now = _now()
    _seed_kb_id("kb-1", COLLECTION, 1)
    _seed_document(COLLECTION, "doc", 1)
    fake.collections[NAME] = [_row("c1", "kb-1", "doc", created_at=int(now) - 3600)]

    assert purge_leftover_milvus_rows(grace_seconds=60, now=now) == {}
    assert len(fake.collections[NAME]) == 1


def test_a_failed_ledger_read_aborts_before_any_delete(
    fake: FakeMilvus, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = _now()
    _seed_kb_id("kb-1", COLLECTION, 1)
    _seed_document(COLLECTION, "doc", 1)
    fake.collections[NAME] = [_row("c1", "kb-1", "doc", created_at=int(now) - 3600)]

    def explode(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("ledger is unreadable")

    monkeypatch.setattr(get_vector_index_store(), "iter_batches", explode)

    with pytest.raises(DatabaseOperationError):
        purge_leftover_milvus_rows(grace_seconds=60, now=now)
    assert len(fake.collections[NAME]) == 1
    assert fake.deletes == []


def test_a_short_ledger_read_aborts_before_any_delete(
    fake: FakeMilvus, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = _now()
    _seed_kb_id("kb-1", COLLECTION, 1)
    _seed_document(COLLECTION, "doc", 1)
    fake.collections[NAME] = [_row("c1", "kb-1", "doc", created_at=int(now) - 3600)]

    # The count sees the row but the iteration swallows and yields nothing.
    monkeypatch.setattr(get_vector_index_store(), "iter_batches", lambda **_: iter(()))

    with pytest.raises(DatabaseOperationError):
        purge_leftover_milvus_rows(grace_seconds=60, now=now)
    assert len(fake.collections[NAME]) == 1
    assert fake.deletes == []


@pytest.mark.parametrize("stale_count", [0, 5], ids=["stale-low", "stale-high"])
def test_a_stale_handle_count_is_retried_with_a_fresh_handle(
    stale_count: int,
) -> None:
    # The first handle's count disagrees with the three rows the iterator
    # yields; only reopening on the retry can see the consistent count.
    store = _FakeLedgerStore(true_count=3, stale_count=stale_count)

    rows = milvus_cleanup._read_table(store, "documents", ["doc_id"])

    assert rows == [{"doc_id": "d0"}, {"doc_id": "d1"}, {"doc_id": "d2"}]
    # One handle to count and iterate, one to re-count, per attempt.
    assert store.opens == 4


def test_the_cap_stops_the_run_and_leaves_the_rest(
    fake: FakeMilvus, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = _now()
    _seed_kb_id("kb-1", COLLECTION, 1)
    monkeypatch.setattr(milvus_cleanup, "MAX_DELETES_PER_RUN", 2)
    fake.collections[NAME] = [
        _row(f"c{i}", f"kb-gone-{i}", f"doc-{i}", created_at=int(now) - 3600)
        for i in range(5)
    ]

    assert purge_leftover_milvus_rows(grace_seconds=60, now=now) == {NAME: 2}
    assert len(fake.collections[NAME]) == 3


def test_another_users_same_doc_id_is_untouched(
    fake: FakeMilvus,
) -> None:
    now = _now()
    _seed_kb_id("kb-1", COLLECTION, 1)
    _seed_kb_id("kb-2", COLLECTION, 2)
    # The ledger keys documents by (collection, doc_id), so both owners share the
    # one row and neither owner's Milvus rows are the orphan under test.
    _seed_document(COLLECTION, "shared", 2)
    fake.collections[NAME] = [
        _row("c1", "kb-1", "shared", user_id=1, created_at=int(now) - 3600),
        _row("c2", "kb-2", "shared", user_id=2, created_at=int(now) - 3600),
    ]

    assert purge_leftover_milvus_rows(grace_seconds=60, now=now) == {}
    assert [row["kb_id"] for row in fake.collections[NAME]] == ["kb-1", "kb-2"]


def test_a_renaming_owners_rows_are_kept_when_only_the_other_owner_has_documents(
    fake: FakeMilvus,
) -> None:
    now = _now()
    # Owners 1 and 2 both use collection "kb"; only owner 2 has a documents row.
    # Owner 1 renamed the collection between the kb_ids and documents reads, so
    # its (collection, user_id) pair is absent and its rows must not look missing.
    _seed_kb_id("kb-1", COLLECTION, 1)
    _seed_kb_id("kb-2", COLLECTION, 2)
    _seed_document(COLLECTION, "doc-b", 2)
    fake.collections[NAME] = [
        _row(
            "c-a", "kb-1", "doc-a", user_id=1, visible=True, created_at=int(now) - 3600
        ),
        _row(
            "c-b", "kb-2", "doc-b", user_id=2, visible=True, created_at=int(now) - 3600
        ),
    ]

    # A guard keyed by collection name alone would delete owner 1's visible row.
    assert purge_leftover_milvus_rows(grace_seconds=60, now=now) == {}
    assert [row["chunk_id"] for row in fake.collections[NAME]] == ["c-a", "c-b"]


def test_a_collection_absent_from_documents_skips_its_missing_check(
    fake: FakeMilvus, caplog: pytest.LogCaptureFixture
) -> None:
    now = _now()
    # kb_ids says these kb_ids live in "renamed", but documents has no such row:
    # a stale/partial ledger must not make every scanned document look missing.
    # The guard is per owner pair, so each of the two owners warns once.
    _seed_kb_id("kb-1", "renamed", 1)
    _seed_kb_id("kb-2", "renamed", 2)
    _seed_document("other", "doc", 1)
    fake.collections[NAME] = [
        _row("c1", "kb-1", "doc-a", visible=True, created_at=int(now) - 3600),
        _row("c2", "kb-2", "doc-b", visible=True, created_at=int(now) - 3600),
    ]

    with caplog.at_level("WARNING"):
        assert purge_leftover_milvus_rows(grace_seconds=60, now=now) == {}

    assert len(fake.collections[NAME]) == 2
    assert caplog.text.count("renamed") == 2


def test_the_scan_finishes_before_the_first_ledger_read(
    fake: FakeMilvus, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = _now()
    _seed_kb_id("kb-1", COLLECTION, 1)
    fake.collections[NAME] = [_row("c1", "kb-1", "doc", created_at=int(now) - 3600)]
    events: list[str] = []
    real_scan = milvus_cleanup._scan_rows

    def scan(*args: Any, **kwargs: Any) -> Any:
        result = real_scan(*args, **kwargs)
        # Reached only after the real scan consumed and closed its iterator.
        events.append("scan")
        return result

    monkeypatch.setattr(milvus_cleanup, "_scan_rows", scan)
    real_read = milvus_cleanup._read_rows

    def read(*args: Any, **kwargs: Any) -> Any:
        events.append("ledger")
        return real_read(*args, **kwargs)

    monkeypatch.setattr(milvus_cleanup, "_read_rows", read)

    purge_leftover_milvus_rows(grace_seconds=60, now=now)

    assert events[0] == "scan"
    assert events.index("scan") < events.index("ledger")


def test_the_orphan_pass_batches_and_stops_at_the_soft_cap(
    fake: FakeMilvus, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = _now()
    _seed_kb_id("kb-1", COLLECTION, 1)
    monkeypatch.setattr(milvus_cleanup, "MAX_DELETES_PER_RUN", 150)
    size = DEFAULT_VECTOR_STORE_DELETE_BATCH_SIZE
    fake.collections[NAME] = [
        _row(f"c{i}", "kb-gone", f"doc-{i}", created_at=int(now) - 3600)
        for i in range(2 * size + 50)
    ]

    # The cap is soft: it is checked before each batch, so the run overshoots to
    # two batches (200) and the third, later batch of documents remains.
    assert purge_leftover_milvus_rows(grace_seconds=60, now=now) == {NAME: 2 * size}
    assert len(fake.collections[NAME]) == 50


def test_a_failing_delete_keeps_completed_batches_in_the_shared_run_state(
    fake: FakeMilvus,
) -> None:
    now = _now()
    size = DEFAULT_VECTOR_STORE_DELETE_BATCH_SIZE
    other = milvus_collection_name("cleanup-state")
    cutoff = int(now - 60)
    # Two collections of orphans; the first collection's second delete raises.
    fake.collections[NAME] = [
        _row(f"a{i}", "kb-gone-1", f"doc-a-{i}", created_at=int(now) - 3600)
        for i in range(size + 1)
    ]
    fake.collections[other] = [
        _row(f"b{i}", "kb-gone-2", f"doc-b-{i}", created_at=int(now) - 3600)
        for i in range(2 * size)
    ]
    calls = {"n": 0}

    def fail_second(_name: str) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("milvus delete failed")

    fake.before_delete = fail_second
    state = milvus_cleanup._RunState(budget_left=size + 50, result={})

    with pytest.raises(RuntimeError):
        milvus_cleanup._purge_collection(
            fake,
            NAME,
            milvus_cleanup._scan_rows(fake, NAME, cutoff),
            {},
            None,
            set(),
            None,
            cutoff,
            state,
            set(),
        )

    # The first batch is counted even though the collection raised, and its
    # budget is gone, so the next collection only sees what is left.
    assert state.result == {NAME: size}
    assert state.budget_left == 50

    fake.before_delete = None
    milvus_cleanup._purge_collection(
        fake,
        other,
        milvus_cleanup._scan_rows(fake, other, cutoff),
        {},
        None,
        set(),
        None,
        cutoff,
        state,
        set(),
    )
    assert state.result[other] == size
    assert state.budget_left == 50 - size
    assert len(fake.collections[other]) == size


def test_a_non_terminal_run_row_protects_the_document(
    fake: FakeMilvus,
) -> None:
    now = _now()
    old = datetime.fromtimestamp(now - 3600, tz=timezone.utc)
    _seed_kb_id("kb-1", COLLECTION, 1)
    # The non-terminal row is read first and the terminal row last; a last-wins
    # reader would purge on the terminal tail and break the every-row rule.
    _seed_runs(
        [
            _run_row(COLLECTION, "doc", 1, status="running", updated_at=old),
            _run_row(COLLECTION, "doc", 1, status="success", updated_at=old),
        ]
    )
    fake.collections[NAME] = [_row("c1", "kb-1", "doc", created_at=int(now) - 3600)]

    assert purge_leftover_milvus_rows(grace_seconds=60, now=now) == {}
    assert len(fake.collections[NAME]) == 1


def test_only_all_terminal_old_run_rows_purge_the_document(
    fake: FakeMilvus,
) -> None:
    now = _now()
    old = datetime.fromtimestamp(now - 3600, tz=timezone.utc)
    _seed_kb_id("kb-1", COLLECTION, 1)
    _seed_runs(
        [
            _run_row(COLLECTION, "doc", 1, status="success", updated_at=old),
            _run_row(COLLECTION, "doc", 1, status="failed", updated_at=old),
        ]
    )
    fake.collections[NAME] = [_row("c1", "kb-1", "doc", created_at=int(now) - 3600)]

    assert purge_leftover_milvus_rows(grace_seconds=60, now=now) == {NAME: 1}
    assert fake.collections[NAME] == []


def test_missing_document_does_not_shield_its_siblings_leftover(
    fake: FakeMilvus,
) -> None:
    now = _now()
    old = datetime.fromtimestamp(now - 3600, tz=timezone.utc)
    _seed_kb_id("kb-1", COLLECTION, 1)
    _seed_document(COLLECTION, "present", 1)
    _seed_run(COLLECTION, "present", 1, status="success", updated_at=old)
    fake.collections[NAME] = [
        _row("c-a", "kb-1", "gone-a", visible=True, created_at=int(now) - 3600),
        _row("c-b1", "kb-1", "present", created_at=int(now) - 3600),
        _row("c-b2", "kb-1", "present", visible=True, created_at=int(now) - 3600),
    ]

    # A is missing from documents, so all of its rows go; B's document is present,
    # so only its invisible leftover goes and its visible row stays.
    assert purge_leftover_milvus_rows(grace_seconds=60, now=now) == {NAME: 2}
    assert [row["chunk_id"] for row in fake.collections[NAME]] == ["c-b2"]


def test_a_row_rewritten_after_the_scan_is_spared_by_the_cutoff(
    fake: FakeMilvus,
) -> None:
    now = _now()
    _seed_kb_id("kb-1", COLLECTION, 1)
    fake.collections[NAME] = [_row("c1", "kb-gone", "doc", created_at=int(now) - 3600)]

    def rewrite(_name: str) -> None:
        fake.collections[NAME][0]["created_at"] = int(now)
        fake.before_delete = None

    fake.before_delete = rewrite

    assert purge_leftover_milvus_rows(grace_seconds=60, now=now) == {}
    assert len(fake.collections[NAME]) == 1


def test_a_failing_collection_still_lets_the_others_purge_then_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = _now()
    _seed_kb_id("kb-1", COLLECTION, 1)
    other = milvus_collection_name("cleanup-other")
    client = FakeMilvus(
        {
            NAME: [_row("c1", "kb-gone", "doc-a", created_at=int(now) - 3600)],
            other: [_row("c2", "kb-gone", "doc-b", created_at=int(now) - 3600)],
        }
    )
    monkeypatch.setattr(
        MilvusConnectionManager, "get_shared_client_from_env", lambda _self: client
    )
    # A plain error, not code 101, must fail just this collection.
    client.fail_query = {NAME}

    with pytest.raises(RuntimeError):
        purge_leftover_milvus_rows(grace_seconds=60, now=now)
    assert client.collections[other] == []


def test_each_delete_batch_logs_one_info_line(
    fake: FakeMilvus, caplog: pytest.LogCaptureFixture
) -> None:
    now = _now()
    old = datetime.fromtimestamp(now - 3600, tz=timezone.utc)
    _seed_kb_id("kb-1", COLLECTION, 1)
    _seed_document(COLLECTION, "doc", 1)
    _seed_run(COLLECTION, "doc", 1, status="success", updated_at=old)
    fake.collections[NAME] = [
        _row("c-orphan", "kb-gone", "d-orphan", created_at=int(now) - 3600),
        _row("c-missing", "kb-1", "gone", created_at=int(now) - 3600),
        _row("c-uncommitted", "kb-1", "doc", created_at=int(now) - 3600),
    ]

    with caplog.at_level("INFO"):
        assert purge_leftover_milvus_rows(grace_seconds=60, now=now) == {NAME: 3}

    lines = [
        record.getMessage()
        for record in caplog.records
        if "deleted batch" in record.getMessage()
    ]
    assert len(lines) == 3
    text = "\n".join(lines)
    assert f"collection={NAME}" in text
    assert "kb_id=kb-gone" in text
    assert "reason=orphan" in text
    assert "reason=missing document" in text
    assert "reason=uncommitted" in text
    assert "doc_ids=1" in text
    assert "deleted=1" in text


def test_a_raising_collection_logs_the_partial_result_before_reraising(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    now = _now()
    _seed_kb_id("kb-1", COLLECTION, 1)
    other = milvus_collection_name("cleanup-partial")
    client = FakeMilvus(
        {
            NAME: [_row("c1", "kb-gone", "doc-a", created_at=int(now) - 3600)],
            other: [_row("c2", "kb-gone", "doc-b", created_at=int(now) - 3600)],
        }
    )
    monkeypatch.setattr(
        MilvusConnectionManager, "get_shared_client_from_env", lambda _self: client
    )
    client.fail_query = {NAME}

    with caplog.at_level("WARNING"):
        with pytest.raises(RuntimeError):
            purge_leftover_milvus_rows(grace_seconds=60, now=now)

    partial = [
        record.getMessage()
        for record in caplog.records
        if "partial result" in record.getMessage()
    ]
    assert len(partial) == 1
    assert str({other: 1}) in partial[0]
    assert client.collections[other] == []


def test_a_not_loaded_collection_is_skipped_with_a_warning(
    fake: FakeMilvus, caplog: pytest.LogCaptureFixture
) -> None:
    now = _now()
    _seed_kb_id("kb-1", COLLECTION, 1)
    fake.collections[NAME] = [_row("c1", "kb-gone", "doc", created_at=int(now) - 3600)]
    fake.not_loaded = {NAME}

    with caplog.at_level("WARNING"):
        assert purge_leftover_milvus_rows(grace_seconds=60, now=now) == {}

    assert "not loaded" in caplog.text
    assert len(fake.collections[NAME]) == 1


def test_the_cap_accounts_deleted_rows_not_batch_document_ids(
    fake: FakeMilvus, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = _now()
    _seed_kb_id("kb-1", COLLECTION, 1)
    monkeypatch.setattr(milvus_cleanup, "MAX_DELETES_PER_RUN", 2)
    monkeypatch.setattr(milvus_cleanup, "DEFAULT_VECTOR_STORE_DELETE_BATCH_SIZE", 1)
    # Three single-doc batches of two rows each: charging the budget per deleted
    # row stops after one batch, while charging per doc_id would run two.
    fake.collections[NAME] = [
        _row(f"c-{doc}-{i}", "kb-gone", f"doc-{doc}", created_at=int(now) - 3600)
        for doc in range(3)
        for i in range(2)
    ]

    assert purge_leftover_milvus_rows(grace_seconds=60, now=now) == {NAME: 2}
    assert len(fake.deletes) == 1
    assert len(fake.collections[NAME]) == 4


def test_delete_rows_loads_an_unloaded_collection_before_deleting(
    fake: FakeMilvus, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = _now()
    _seed_kb_id("kb-1", COLLECTION, 1)
    fake.collections[NAME] = [_row("c1", "kb-gone", "doc", created_at=int(now) - 3600)]
    # The scan succeeds; only the first delete reports the collection not loaded.
    fake.delete_not_loaded = {NAME}
    loaded: list[tuple[Any, str, int]] = []

    def ensure(client: Any, model: str, dimension: int) -> None:
        loaded.append((client, model, dimension))
        fake.delete_not_loaded.discard(NAME)

    monkeypatch.setattr(collection_handle, "ensure_milvus_collection", ensure)

    assert purge_leftover_milvus_rows(grace_seconds=60, now=now) == {NAME: 1}
    assert fake.collections[NAME] == []
    assert len(fake.deletes) == 1
    assert loaded == [(fake, MODEL, 8)]
