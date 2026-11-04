"""Both web lookups by ``file_id`` read the coordinator's vector store (#2665)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterable

import pytest

from xagent.core.tools.core.RAG_tools.kb import KBCoordinator, get_kb_coordinator
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
    ensure_documents_table,
)
from xagent.core.tools.core.RAG_tools.storage.factory import (
    bind_storage_shim_for_current_context,
)
from xagent.providers.vector_store.lancedb import get_connection_from_env
from xagent.web.api import kb as kb_module
from xagent.web.services import kb_file_service


class _OtherBackendStore:
    def __init__(self, records: list[SimpleNamespace]) -> None:
        self.records = records
        self.calls: list[list[str]] = []

    def list_document_records_by_file_ids(
        self, file_ids: Iterable[str]
    ) -> list[SimpleNamespace]:
        self.calls.append(list(file_ids))
        return self.records


def test_document_refs_come_from_the_bound_vector_store(monkeypatch):
    store = _OtherBackendStore(
        [
            SimpleNamespace(doc_id="d-4", file_id="f", collection=" other "),
            SimpleNamespace(doc_id=" d-1 ", file_id="f", collection="kb"),
            SimpleNamespace(doc_id="", file_id="f", collection="kb"),
            SimpleNamespace(doc_id="d-2", file_id="f", collection=" "),
            SimpleNamespace(doc_id="d-3", file_id="f", collection=None),
        ]
    )
    monkeypatch.setattr(
        get_kb_coordinator().storage_shim, "get_vector_index_store", lambda: store
    )

    refs = kb_module._list_document_refs_for_uploaded_file("f")

    assert refs == [("other", "d-4"), ("kb", "d-1")]
    assert store.calls == [["f"]]


def test_referenced_file_ids_come_from_the_bound_vector_store(monkeypatch):
    store = _OtherBackendStore(
        [
            SimpleNamespace(doc_id="d-1", file_id="f-1", collection="kb"),
            SimpleNamespace(doc_id="", file_id="f-2", collection=""),
        ]
    )
    monkeypatch.setattr(
        get_kb_coordinator().storage_shim, "get_vector_index_store", lambda: store
    )

    referenced = kb_file_service.find_referenced_file_ids(["f-1", "f-2", "f-3"])

    assert referenced == {"f-1", "f-2"}
    assert store.calls == [["f-1", "f-2", "f-3"]]


def test_referenced_file_ids_read_their_coordinators_store():
    own = _OtherBackendStore(
        [SimpleNamespace(doc_id="d-1", file_id="f-1", collection="kb")]
    )
    ambient = _OtherBackendStore([])
    coordinator = KBCoordinator(
        storage_shim=SimpleNamespace(get_vector_index_store=lambda: own)
    )

    with bind_storage_shim_for_current_context(
        SimpleNamespace(get_vector_index_store=lambda: ambient)
    ):
        referenced = coordinator.file_compatibility.find_referenced_file_ids(["f-1"])

    assert referenced == {"f-1"}
    assert ambient.calls == []


def test_document_refs_lookup_raises_when_the_store_fails(monkeypatch):
    class _BrokenStore:
        def list_document_records_by_file_ids(self, file_ids: Iterable[str]) -> Any:
            raise RuntimeError("store down")

    monkeypatch.setattr(
        get_kb_coordinator().storage_shim, "get_vector_index_store", _BrokenStore
    )

    with pytest.raises(RuntimeError, match="store down"):
        kb_module._list_document_refs_for_uploaded_file("f")


def _doc(collection: str, doc_id: str, file_id: str, user_id: Any) -> dict[str, Any]:
    return {
        "collection": collection,
        "doc_id": doc_id,
        "file_id": file_id,
        "user_id": user_id,
    }


def test_document_refs_count_every_owner():
    conn = get_connection_from_env()
    ensure_documents_table(conn)
    conn.open_table("documents").add(
        [
            _doc("kb-a", "d-1", "f", 1),
            _doc("kb-b", "d-2", "f", 424242),
            _doc("kb-c", "d-3", "f", None),
            _doc("kb-d", "d-4", "g", 1),
        ]
    )

    refs = kb_module._list_document_refs_for_uploaded_file("f")

    assert sorted(refs) == [("kb-a", "d-1"), ("kb-b", "d-2"), ("kb-c", "d-3")]
