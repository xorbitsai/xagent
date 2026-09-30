"""Web-file RAG snapshot and restore, split per collection (#2665 item 2)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

import pytest

from xagent.core.tools.core.RAG_tools.kb.models import KBDocumentRowsSnapshot
from xagent.core.tools.core.RAG_tools.storage.factory import get_vector_index_store
from xagent.web.api.kb import (
    _RagDocumentSnapshot,
    _restore_rag_document_snapshot,
    _snapshot_rag_documents_for_uploaded_file,
)


def _doc(collection: str, doc_id: str) -> dict:
    return {
        "collection": collection,
        "doc_id": doc_id,
        "file_id": "file-1",
        "source_path": f"/uploads/{doc_id}.md",
        "file_type": "md",
        "content_hash": "a" * 64,
        "uploaded_at": datetime.now(timezone.utc),
        "title": None,
        "language": None,
        "user_id": 1,
    }


def _chunk(collection: str, doc_id: str, chunk_id: str, *, text="old", user_id=1):
    return {
        "collection": collection,
        "doc_id": doc_id,
        "parse_hash": "p1",
        "chunk_id": chunk_id,
        "index": 0,
        "text": text,
        "page_number": None,
        "section": None,
        "anchor": None,
        "json_path": None,
        "chunk_hash": f"h-{chunk_id}",
        "config_hash": "cfg",
        "created_at": datetime.now(timezone.utc),
        "metadata": "{}",
        "user_id": user_id,
    }


def _chunks() -> list[tuple]:
    conn = get_vector_index_store().get_raw_connection()
    return sorted(
        (row["collection"], row["doc_id"], row["chunk_id"], row["text"], row["user_id"])
        for row in conn.open_table("chunks").to_arrow().to_pylist()
    )


def _rows_snapshot(collection: str, doc_ids: list[str]) -> KBDocumentRowsSnapshot:
    return KBDocumentRowsSnapshot(
        collection=collection, doc_ids=tuple(doc_ids), rows_by_table={}
    )


def test_restore_puts_back_each_collection_and_drops_rows_added_since() -> None:
    store = get_vector_index_store()
    store.upsert_documents([_doc("c1", "d1"), _doc("c2", "d2"), _doc("c1", "d3")])
    store.upsert_chunks(
        [
            _chunk("c1", "d1", "k1"),
            _chunk("c2", "d2", "k1"),
            _chunk("c1", "d3", "k1"),
            _chunk("c1", "d1", "k-other", user_id=2),
        ]
    )
    snapshot = _snapshot_rag_documents_for_uploaded_file(
        "file-1", user_id=1, is_admin=False
    )
    assert snapshot is not None

    store.upsert_chunks(
        [
            _chunk("c1", "d1", "k1", text="new"),
            _chunk("c1", "d1", "k-new"),
            _chunk("c2", "d2", "k-new"),
            _chunk("c1", "d3", "k1", text="new"),
            _chunk("c1", "d3", "k-new"),
        ]
    )
    _restore_rag_document_snapshot(snapshot, user_id=1, is_admin=False)

    assert _chunks() == [
        ("c1", "d1", "k-other", "old", 2),
        ("c1", "d1", "k1", "old", 1),
        ("c1", "d3", "k1", "old", 1),
        ("c2", "d2", "k1", "old", 1),
    ]


def test_snapshot_and_restore_open_one_handle_per_collection() -> None:
    coordinator = MagicMock()
    coordinator.capture_document_rows_sync.side_effect = (
        lambda collection, doc_ids, **_kw: _rows_snapshot(collection, doc_ids)
    )
    refs = [("c2", "d2"), ("c1", "d1"), ("c2", "d3")]
    with (
        patch(
            "xagent.web.api.kb._list_document_refs_for_uploaded_file",
            return_value=refs,
        ),
        patch("xagent.web.api.kb.get_kb_coordinator", return_value=coordinator),
    ):
        snapshot = _snapshot_rag_documents_for_uploaded_file(
            "file-1", user_id=7, is_admin=False
        )
        assert snapshot is not None
        _restore_rag_document_snapshot(snapshot, user_id=7, is_admin=False)

    assert snapshot.doc_refs == refs
    assert coordinator.capture_document_rows_sync.call_args_list == [
        call("c2", ["d2", "d3"], user_id=7, is_admin=False),
        call("c1", ["d1"], user_id=7, is_admin=False),
    ]
    assert snapshot.collections == [
        _rows_snapshot("c2", ["d2", "d3"]),
        _rows_snapshot("c1", ["d1"]),
    ]
    assert coordinator.restore_document_rows_sync.call_args_list == [
        call(snapshot.collections[0], user_id=7, is_admin=False),
        call(snapshot.collections[1], user_id=7, is_admin=False),
    ]


def test_no_document_refs_opens_no_handle() -> None:
    coordinator = MagicMock()
    with (
        patch(
            "xagent.web.api.kb._list_document_refs_for_uploaded_file",
            return_value=[],
        ),
        patch("xagent.web.api.kb.get_kb_coordinator", return_value=coordinator),
    ):
        snapshot = _snapshot_rag_documents_for_uploaded_file(
            "file-1", user_id=7, is_admin=False
        )
        assert snapshot is not None
        _restore_rag_document_snapshot(snapshot, user_id=7, is_admin=False)

    assert snapshot.doc_refs == []
    assert snapshot.collections == []
    assert coordinator.method_calls == []


def test_snapshot_failure_returns_none(caplog: pytest.LogCaptureFixture) -> None:
    coordinator = MagicMock()
    coordinator.capture_document_rows_sync.side_effect = RuntimeError("boom")
    with (
        patch(
            "xagent.web.api.kb._list_document_refs_for_uploaded_file",
            return_value=[("c1", "d1")],
        ),
        patch("xagent.web.api.kb.get_kb_coordinator", return_value=coordinator),
        caplog.at_level(logging.WARNING, logger="xagent.web.api.kb"),
    ):
        snapshot = _snapshot_rag_documents_for_uploaded_file(
            "file-1", user_id=7, is_admin=False
        )

    assert snapshot is None
    assert "Failed to snapshot RAG document rows before web file refresh" in (
        caplog.text
    )


def test_restore_failure_propagates() -> None:
    coordinator = MagicMock()
    coordinator.restore_document_rows_sync.side_effect = RuntimeError("boom")
    snapshot = _RagDocumentSnapshot(
        doc_refs=[("c1", "d1")], collections=[_rows_snapshot("c1", ["d1"])]
    )
    with (
        patch("xagent.web.api.kb.get_kb_coordinator", return_value=coordinator),
        pytest.raises(RuntimeError, match="boom"),
    ):
        _restore_rag_document_snapshot(snapshot, user_id=7, is_admin=False)
