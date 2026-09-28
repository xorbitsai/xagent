"""Web-file status snapshot, restore and reindex marker (#2665 item 3)."""

from __future__ import annotations

import logging
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pytest

from xagent.core.tools.core.RAG_tools.kb import get_kb_coordinator
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
    ensure_documents_table,
    ensure_ingestion_runs_table,
)
from xagent.core.tools.core.RAG_tools.storage.factory import (
    bind_storage_shim_for_current_context,
    get_ingestion_status_store,
)
from xagent.providers.vector_store.lancedb import get_connection_from_env
from xagent.web.api import kb as kb_module

_OLD = datetime(2020, 1, 2, 3, 4, 5, 678901)


class _VectorStore:
    def list_document_records_by_file_ids(self, file_ids: Any) -> list[Any]:
        assert list(file_ids) == ["f"]
        return [
            SimpleNamespace(doc_id="d-1", collection="kb-a"),
            SimpleNamespace(doc_id="d-2", collection="kb-b"),
        ]


class _StatusStore:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[Any, ...]] = []

    def _record(self, *call: Any) -> None:
        if self.fail:
            raise RuntimeError("status store down")
        self.calls.append(call)

    def load_ingestion_status_rows(self, doc_refs: Any) -> list[dict[str, Any]]:
        self._record("load", doc_refs)
        return [{"doc_id": "d-1"}]

    def replace_ingestion_status_rows(self, doc_refs: Any, rows: Any) -> None:
        self._record("replace", doc_refs, rows)

    def clear_ingestion_status(self, **kwargs: Any) -> None:
        self._record("clear", kwargs)


def _bind(monkeypatch: pytest.MonkeyPatch, status_store: _StatusStore) -> None:
    shim = get_kb_coordinator().storage_shim
    monkeypatch.setattr(shim, "get_vector_index_store", _VectorStore)
    monkeypatch.setattr(shim, "get_ingestion_status_store", lambda: status_store)


REFS = [("kb-a", "d-1"), ("kb-b", "d-2")]
_FAILING_AMBIENT = SimpleNamespace(
    get_ingestion_status_store=lambda: _StatusStore(fail=True),
    get_vector_index_store=_VectorStore,
)


def test_snapshot_and_restore_use_the_coordinators_status_store(monkeypatch):
    status_store = _StatusStore()
    _bind(monkeypatch, status_store)

    with bind_storage_shim_for_current_context(_FAILING_AMBIENT):
        snapshot = kb_module._snapshot_ingestion_runs_for_uploaded_file("f")
        assert snapshot is not None
        kb_module._restore_ingestion_runs_snapshot(snapshot)

    assert (snapshot.doc_refs, snapshot.rows) == (REFS, [{"doc_id": "d-1"}])
    assert status_store.calls == [
        ("load", REFS),
        ("replace", REFS, [{"doc_id": "d-1"}]),
    ]


def test_reindex_marker_clears_each_ref_for_every_owner(monkeypatch):
    status_store = _StatusStore()
    _bind(monkeypatch, status_store)

    with bind_storage_shim_for_current_context(_FAILING_AMBIENT):
        assert kb_module._mark_uploaded_file_for_reindex("f") is True

    assert status_store.calls == [
        ("clear", {"collection": c, "doc_id": d, "user_id": None, "is_admin": True})
        for c, d in REFS
    ]


def test_status_store_failures_keep_each_callers_contract(monkeypatch, caplog):
    _bind(monkeypatch, _StatusStore(fail=True))

    with caplog.at_level(logging.WARNING, logger=kb_module.logger.name):
        assert kb_module._snapshot_ingestion_runs_for_uploaded_file("f") is None
        assert kb_module._mark_uploaded_file_for_reindex("f") is False
    swallowed = [r.exc_info[1] for r in caplog.records if r.exc_info]
    assert [str(exc) for exc in swallowed] == ["status store down"] * 2
    with pytest.raises(RuntimeError, match="status store down"):
        kb_module._restore_ingestion_runs_snapshot(
            kb_module._IngestionRunsSnapshot(doc_refs=REFS, rows=[])
        )


def _run(collection: str, doc_id: str, user_id: int | None) -> dict[str, Any]:
    return {
        "collection": collection,
        "doc_id": doc_id,
        "status": "success",
        "message": f"{collection}/{doc_id}/{user_id}",
        "parse_hash": "p1",
        "created_at": _OLD,
        "updated_at": _OLD,
        "user_id": user_id,
    }


def _runs_table() -> Any:
    conn = get_connection_from_env()
    ensure_ingestion_runs_table(conn)
    return conn.open_table("ingestion_runs")


def _key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (row["collection"], row["doc_id"], row["message"])


def test_restore_puts_back_the_snapshot_rows_with_their_timestamps():
    conn = get_connection_from_env()
    ensure_documents_table(conn)
    conn.open_table("documents").add(
        [
            {"collection": "kb-a", "doc_id": "d-1", "file_id": "f", "user_id": 1},
            {"collection": "kb-b", "doc_id": "d-2", "file_id": "f", "user_id": 2},
            {"collection": "kb-a", "doc_id": "d-2", "file_id": "g", "user_id": 1},
        ]
    )
    targets = [
        _run("kb-a", "d-1", 1),
        _run("kb-a", "d-1", None),
        _run("kb-b", "d-2", 2),
    ]
    bystanders = [_run("kb-a", "d-2", 1), _run("kb-b", "d-1", 2)]
    _runs_table().add(targets + bystanders)

    snapshot = kb_module._snapshot_ingestion_runs_for_uploaded_file("f")
    assert snapshot is not None
    get_ingestion_status_store().write_ingestion_status(
        "kb-a", "d-1", status="processing", user_id=1
    )
    _runs_table().add([_run("kb-b", "d-2", 424242), _run("kb-b", "d-2", None)])
    kb_module._restore_ingestion_runs_snapshot(snapshot)

    assert sorted(snapshot.doc_refs) == REFS
    assert sorted(snapshot.rows, key=_key) == sorted(targets, key=_key)
    rows = _runs_table().search().limit(-1).to_arrow().to_pylist()
    assert sorted(rows, key=_key) == sorted(targets + bystanders, key=_key)
