"""Failed-ingest compensation with the Milvus engine (#2869).

The unit cases need no server; the ``milvus`` cases run against ``MILVUS_URI``.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import DatabaseOperationError
from xagent.core.tools.core.RAG_tools.core.schemas import (
    DocumentProcessingStatus,
    IngestionResult,
    RegisterDocumentRequest,
)
from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
    KBCollectionHandle,
    KBHandleProvider,
    ensure_milvus_collection,
    milvus_collection_name,
)
from xagent.core.tools.core.RAG_tools.kb.coordinator import get_kb_coordinator
from xagent.core.tools.core.RAG_tools.kb.kb_ids import get_or_create_kb_id
from xagent.core.tools.core.RAG_tools.kb.models import (
    KBAccessMode,
    KBBackendCapabilities,
    KBCollectionContext,
    KBStorageBackend,
    KBUserScope,
    RollbackFailedIngestionRequest,
)
from xagent.core.tools.core.RAG_tools.storage.factory import (
    get_ingestion_status_store,
    get_main_pointer_store,
    get_metadata_store,
    get_vector_index_store,
)
from xagent.providers.vector_store.milvus import MilvusConnectionManager
from xagent.web.api import kb as kb_module
from xagent.web.api.kb import (
    _create_document_compensation,
    _create_status_compensation,
    _IngestionRunsSnapshot,
    _RagDocumentSnapshot,
    _restore_ingestion_runs_snapshot,
    _restore_rag_document_snapshot,
    _rollback_ingested_document,
    _snapshot_ingestion_runs_for_uploaded_file,
    _snapshot_rag_documents_for_uploaded_file,
)

COLLECTION = "kb"
PARSE, NEXT_PARSE = "ph-1", "ph-2"


def _existing(doc_id: str = "doc") -> IngestionResult:
    return IngestionResult(
        status="partial",
        doc_id=doc_id,
        completed_steps=[{"name": "register_document", "metadata": {"created": False}}],
        message="failed",
    )


def _rollback(result: IngestionResult, **kwargs: Any) -> None:
    _rollback_ingested_document(
        collection_name=COLLECTION,
        result=result,
        user_id=1,
        is_admin=False,
        label="test",
        **kwargs,
    )


@pytest.mark.parametrize(("lancedb", "discards"), [(True, False), (False, True)])
def test_only_a_milvus_deployment_discards_the_rows_of_an_existing_document(
    lancedb: bool, discards: bool
) -> None:
    coordinator = MagicMock()
    with (
        patch.object(kb_module, "ledger_holds_vectors", return_value=lancedb),
        patch.object(kb_module, "get_kb_coordinator", return_value=coordinator),
        patch.object(kb_module, "clear_ingestion_status") as clear,
    ):
        _rollback(_existing())

    clear.assert_called_once_with(COLLECTION, "doc", user_id=1, is_admin=False)
    assert coordinator.discard_uncommitted_embeddings_sync.call_args_list == (
        [call(COLLECTION, "doc", user_id=1)] if discards else []
    )


def test_a_new_document_and_a_restored_snapshot_are_not_discarded_separately() -> None:
    coordinator = MagicMock()
    created = IngestionResult(
        status="partial",
        doc_id="doc",
        completed_steps=[{"name": "register_document", "metadata": {"created": True}}],
        message="failed",
    )
    snapshot = _RagDocumentSnapshot(doc_refs=[], collections=[])
    with (
        patch.object(kb_module, "ledger_holds_vectors", return_value=False),
        patch.object(kb_module, "get_kb_coordinator", return_value=coordinator),
        patch.object(kb_module, "delete_document") as delete,
        patch.object(kb_module, "clear_ingestion_status"),
    ):
        delete.return_value = MagicMock(status="success")
        _rollback(created)
        _rollback(_existing(), rag_snapshot=snapshot)

    coordinator.discard_uncommitted_embeddings_sync.assert_not_called()


def test_the_restore_collects_the_documents_the_engine_marked() -> None:
    coordinator = MagicMock()
    coordinator.restore_document_rows_sync.side_effect = [["d1"], [], ["d3", "d4"]]
    snapshot = _RagDocumentSnapshot(
        doc_refs=[],
        collections=[
            MagicMock(collection="c1"),
            MagicMock(collection="c2"),
            MagicMock(collection="c3"),
        ],
    )

    with patch.object(kb_module, "get_kb_coordinator", return_value=coordinator):
        _restore_rag_document_snapshot(snapshot, user_id=1, is_admin=False)

    assert snapshot.incomplete == [("c1", "d1"), ("c3", "d3"), ("c3", "d4")]


@pytest.mark.parametrize(
    ("error", "reported"),
    [
        (DatabaseOperationError("boom", details={"marked": ["d2", "d3"]}), True),
        (RuntimeError("boom"), False),
    ],
)
def test_a_failed_restore_still_reports_the_documents_the_engine_marked(
    error: Exception, reported: bool
) -> None:
    coordinator = MagicMock()
    coordinator.restore_document_rows_sync.side_effect = [["d1"], error]
    snapshot = _RagDocumentSnapshot(
        doc_refs=[],
        collections=[
            MagicMock(collection="c1"),
            MagicMock(collection="c2"),
            MagicMock(collection="c3"),
        ],
    )

    with (
        patch.object(kb_module, "get_kb_coordinator", return_value=coordinator),
        pytest.raises(type(error), match="boom"),
    ):
        _restore_rag_document_snapshot(snapshot, user_id=1, is_admin=False)

    assert coordinator.restore_document_rows_sync.call_count == 2
    assert snapshot.incomplete == [("c1", "d1")] + (
        [("c2", "d2"), ("c2", "d3")] if reported else []
    )


def test_the_status_restore_skips_the_documents_in_keep() -> None:
    rows = [
        {"collection": "c", "doc_id": "d1", "status": "success"},
        {"collection": "c", "doc_id": "d2", "status": "success"},
    ]
    snapshot = _IngestionRunsSnapshot(doc_refs=[("c", "d1"), ("c", "d2")], rows=rows)
    facade = MagicMock()

    with patch.object(kb_module, "_get_api_compatibility_facade", return_value=facade):
        _restore_ingestion_runs_snapshot(snapshot, keep=[("c", "d1")])
        _restore_ingestion_runs_snapshot(snapshot)

    assert facade.replace_ingestion_status_rows.call_args_list == [
        (([("c", "d2")], rows[1:]),),
        ((snapshot.doc_refs, rows),),
    ]


# --- against Milvus ----------------------------------------------------------


@cache
def _client() -> Any:
    from pymilvus import MilvusClient

    return MilvusClient(uri=os.environ["MILVUS_URI"])


@pytest.fixture(scope="module")
def model() -> Iterator[str]:
    """One collection for the module: rows are told apart by each test's kb_id."""
    model = f"rollback-{uuid.uuid4().hex[:12]}"
    ensure_milvus_collection(_client(), model, 3)
    yield model
    _client().drop_collection(milvus_collection_name(model))


@pytest.fixture
def milvus(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("XAGENT_VECTOR_BACKEND", "milvus")
    return _client()


def _handle() -> KBCollectionHandle:
    return KBHandleProvider().open(
        KBCollectionContext(
            collection=COLLECTION,
            user_scope=KBUserScope(user_id=1, is_admin=False),
            access_mode=KBAccessMode.WRITE,
            allow_create=True,
            hide_missing=True,
            metadata_store=get_metadata_store(),
            vector_index_store=get_vector_index_store(),
            ingestion_status_store=get_ingestion_status_store(),
            main_pointer_store=get_main_pointer_store(),
            backend=KBStorageBackend.MILVUS,
            capabilities=KBBackendCapabilities.milvus(),
        )
    )


def _write(
    handle: KBCollectionHandle,
    model: str,
    ids: list[str],
    parse_hash: str,
    *,
    commit: bool,
    doc_id: str = "doc",
) -> None:
    """Write chunks to the ledger and rows to Milvus as an ingest leaves them.

    A commit also removes the document's other rows, the old batch.
    """
    now = datetime.now(timezone.utc)
    handle.write_chunks(
        doc_id,
        parse_hash,
        "cfg",
        {},
        [
            {"chunk_id": c, "index": i, "text": f"kiwi {c}", "created_at": now}
            for i, c in enumerate(ids)
        ],
        user_id=1,
    )
    name, kb_id = milvus_collection_name(model), _kb_id()
    if commit:
        _client().delete(
            name,
            filter="kb_id == {kb_id} and doc_id == {doc_id}",
            filter_params={"kb_id": kb_id, "doc_id": doc_id},
        )
    _client().upsert(
        name,
        [
            {
                "chunk_id": chunk_id,
                "kb_id": kb_id,
                "user_id": 1,
                "doc_id": doc_id,
                "parse_hash": parse_hash,
                "config_hash": "",
                "text": f"kiwi {chunk_id}",
                "dense": [1.0, 0.0, 0.0],
                "visible": commit,
                "created_at": 1,
                "metadata": {},
            }
            for chunk_id in ids
        ],
    )


def _kb_id() -> str:
    return get_or_create_kb_id(
        get_vector_index_store().get_raw_connection(), COLLECTION, 1
    )


def _register(handle: KBCollectionHandle, tmp_path: Path, doc_id: str) -> None:
    source = tmp_path / f"{doc_id}.txt"
    source.write_text("kiwi", encoding="utf-8")
    handle.register_document(
        RegisterDocumentRequest(
            collection=COLLECTION, source_path=str(source), doc_id=doc_id, user_id=1
        )
    )


def _rows(client: Any, model: str) -> dict[str, bool]:
    rows = client.query(
        milvus_collection_name(model),
        filter="kb_id == {kb_id}",
        filter_params={"kb_id": _kb_id()},
        output_fields=["visible"],
        consistency_level="Strong",
    )
    return {row["chunk_id"]: row["visible"] for row in rows}


@pytest.mark.milvus
def test_rolling_back_an_existing_document_drops_its_invisible_rows_only(
    milvus: Any, model: str, tmp_path: Path
) -> None:
    handle = _handle()
    for doc_id in ("doc", "other"):
        _register(handle, tmp_path, doc_id)
    _write(handle, model, ["a", "b"], PARSE, commit=True)
    _write(handle, model, ["c"], PARSE, commit=True, doc_id="other")
    _write(handle, model, ["x", "y"], NEXT_PARSE, commit=False)
    handle.write_ingestion_status("doc", status="failed", user_id=1)

    _rollback(_existing())

    assert _rows(milvus, model) == {"a": True, "b": True, "c": True}
    assert handle.load_ingestion_status(doc_id="doc", user_id=1) == []


def _refresh_rollback(file_id: str = "file-1") -> Any:
    refs = [(COLLECTION, "doc")]
    with patch.object(
        kb_module, "_list_document_refs_for_uploaded_file", return_value=refs
    ):
        runs = _snapshot_ingestion_runs_for_uploaded_file(file_id)
        rows = _snapshot_rag_documents_for_uploaded_file(
            file_id, user_id=1, is_admin=False
        )
    assert runs is not None and rows is not None

    def rollback() -> Any:
        return get_kb_coordinator().rollback_failed_ingestion_sync(
            RollbackFailedIngestionRequest(
                collection=COLLECTION,
                user_id=1,
                is_admin=False,
                ingestion_result=_existing(),
                doc_id="doc",
                source="file-1",
                document_compensation=_create_document_compensation(
                    collection_name=COLLECTION,
                    user_id=1,
                    is_admin=False,
                    file_record_id=file_id,
                    rag_document_snapshot=rows,
                ),
                status_compensation=_create_status_compensation(
                    collection_name=COLLECTION,
                    user_id=1,
                    is_admin=False,
                    ingestion_runs_snapshot=runs,
                    rag_document_snapshot=rows,
                ),
            )
        )

    return rollback


def _refresh_fixture(
    handle: KBCollectionHandle, model: str, tmp_path: Path, *, commit: bool
) -> Any:
    _register(handle, tmp_path, "doc")
    _write(handle, model, ["a", "b"], PARSE, commit=True)
    handle.write_ingestion_status("doc", status="success", parse_hash=PARSE, user_id=1)
    rollback = _refresh_rollback()
    handle.clear_ingestion_status("doc", user_id=1)
    _write(handle, model, ["x", "y"], NEXT_PARSE, commit=commit)
    return rollback


def _status(handle: KBCollectionHandle) -> dict[str, Any]:
    (row,) = handle.load_ingestion_status(doc_id="doc", user_id=1)
    return dict(row)


@pytest.mark.milvus
def test_a_refresh_that_stopped_before_its_commit_restores_the_old_document(
    milvus: Any, model: str, tmp_path: Path
) -> None:
    handle = _handle()
    rollback = _refresh_fixture(handle, model, tmp_path, commit=False)

    result = rollback()

    assert result.rollback_complete
    assert _rows(milvus, model) == {"a": True, "b": True}
    assert _status(handle)["status"] == DocumentProcessingStatus.SUCCESS.value


@pytest.mark.milvus
def test_a_refresh_that_committed_leaves_the_document_marked_for_reingest(
    milvus: Any, model: str, tmp_path: Path
) -> None:
    handle = _handle()
    rollback = _refresh_fixture(handle, model, tmp_path, commit=True)
    assert _rows(milvus, model) == {"x": True, "y": True}

    result = rollback()

    assert result.rollback_complete
    assert _rows(milvus, model) == {}
    status = _status(handle)
    assert status["status"] == DocumentProcessingStatus.PARTIALLY_EMBEDDED.value
    assert status["parse_hash"] == PARSE and "re-ingest" in status["message"]


class _FailingQuery:
    """Forwards every call to a Milvus client except ``query``, which raises."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def query(self, *_: Any, **__: Any) -> Any:
        raise RuntimeError("transient")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)


@pytest.mark.milvus
def test_a_restore_that_fails_while_aligning_milvus_still_leaves_the_document_marked(
    milvus: Any, model: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handle = _handle()
    rollback = _refresh_fixture(handle, model, tmp_path, commit=True)
    monkeypatch.setattr(
        MilvusConnectionManager,
        "get_shared_client_from_env",
        lambda _self: _FailingQuery(milvus),
    )

    result = rollback()

    assert not result.rollback_complete and "transient" in str(result.first_error)
    status = _status(handle)
    assert status["status"] == DocumentProcessingStatus.PARTIALLY_EMBEDDED.value
    assert status["parse_hash"] == PARSE and "re-ingest" in status["message"]
    assert _rows(milvus, model) == {"x": True, "y": True}
