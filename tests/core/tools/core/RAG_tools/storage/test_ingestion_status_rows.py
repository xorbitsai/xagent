"""Verbatim status-row snapshot and restore through the status contract (#2665)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
    ensure_ingestion_runs_table,
)
from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
    LanceDBIngestionStatusStore,
)
from xagent.providers.vector_store.lancedb import get_connection_from_env

_OLD = datetime(2020, 1, 2, 3, 4, 5, 678901)
_OLDER = datetime(2019, 6, 7, 8, 9, 10, 111213)


def _run(collection: str, doc_id: str, user_id: int | None) -> dict[str, Any]:
    return {
        "collection": collection,
        "doc_id": doc_id,
        "status": "success",
        "message": f"{collection}/{doc_id}/{user_id}",
        "parse_hash": "p1",
        "created_at": _OLDER,
        "updated_at": _OLD,
        "user_id": user_id,
    }


def _table() -> Any:
    conn = get_connection_from_env()
    ensure_ingestion_runs_table(conn)
    return conn.open_table("ingestion_runs")


def _all_rows() -> list[dict[str, Any]]:
    return _table().search().limit(-1).to_arrow().to_pylist()


def _key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (row["collection"], row["doc_id"], row["message"])


REFS = [("kb-a", "d-1"), ("kb-b", "d-2")]
TARGETS = [_run("kb-a", "d-1", 1), _run("kb-a", "d-1", None), _run("kb-b", "d-2", 2)]
# Same collection or same doc_id as a ref, but not a ref pair.
BYSTANDERS = [_run("kb-a", "d-2", 1), _run("kb-b", "d-1", 2), _run("kb-c", "d-1", 1)]


def test_load_rows_returns_every_owners_rows_of_the_ref_pairs_verbatim():
    _table().add(TARGETS + BYSTANDERS)

    rows = LanceDBIngestionStatusStore().load_ingestion_status_rows(REFS)

    assert sorted(rows, key=_key) == sorted(TARGETS, key=_key)


def test_load_rows_without_refs_reads_nothing():
    _table().add(TARGETS)

    assert LanceDBIngestionStatusStore().load_ingestion_status_rows([]) == []


def test_replace_rows_drops_every_owners_rows_and_keeps_given_timestamps():
    store = LanceDBIngestionStatusStore()
    _table().add(TARGETS + BYSTANDERS)
    snapshot = store.load_ingestion_status_rows(REFS)
    store.write_ingestion_status("kb-a", "d-1", status="processing", user_id=1)
    _table().add([_run("kb-b", "d-2", 424242), _run("kb-b", "d-2", None)])

    store.replace_ingestion_status_rows(REFS, snapshot)

    assert sorted(_all_rows(), key=_key) == sorted(TARGETS + BYSTANDERS, key=_key)


def test_replace_with_no_rows_only_deletes():
    _table().add(TARGETS + BYSTANDERS)

    LanceDBIngestionStatusStore().replace_ingestion_status_rows(REFS, [])

    assert sorted(_all_rows(), key=_key) == sorted(BYSTANDERS, key=_key)


def test_replace_rejects_rows_outside_the_refs_before_touching_the_table():
    _table().add(TARGETS + BYSTANDERS)

    with pytest.raises(ValueError, match="outside doc_refs"):
        LanceDBIngestionStatusStore().replace_ingestion_status_rows(
            REFS, TARGETS + BYSTANDERS[:1]
        )

    assert sorted(_all_rows(), key=_key) == sorted(TARGETS + BYSTANDERS, key=_key)


def test_empty_refs_read_and_change_nothing():
    store = LanceDBIngestionStatusStore()
    _table().add(TARGETS + BYSTANDERS)

    assert store.load_ingestion_status_rows([]) == []
    store.replace_ingestion_status_rows([], [])
    with pytest.raises(ValueError, match="outside doc_refs"):
        store.replace_ingestion_status_rows([], TARGETS)

    assert sorted(_all_rows(), key=_key) == sorted(TARGETS + BYSTANDERS, key=_key)
