"""``MilvusCollectionHandle`` skeleton: dispatch, ledger delegation, refusals,
capabilities, and the LanceDB-only paths a Milvus deployment skips.

Nothing here connects to Milvus; counts against a server are in
``test_milvus_storage.py``.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
from dataclasses import fields, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import lancedb
import pytest

from xagent.core.tools.core.RAG_tools import kb
from xagent.core.tools.core.RAG_tools.core.exceptions import (
    ConfigurationError,
    DatabaseOperationError,
    DocumentValidationError,
    VectorValidationError,
)
from xagent.core.tools.core.RAG_tools.core.schemas import (
    ChunkEmbeddingData,
    CollectionInfo,
    RegisterDocumentRequest,
)
from xagent.core.tools.core.RAG_tools.kb import collection_handle
from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
    KBCollectionHandle,
    KBHandleProvider,
    LanceDBCollectionHandle,
    MilvusCollectionHandle,
    ensure_milvus_collection,
    ledger_holds_vectors,
    milvus_collection_name,
)
from xagent.core.tools.core.RAG_tools.kb.coordinator import KBCoordinator
from xagent.core.tools.core.RAG_tools.kb.kb_ids import get_or_create_kb_id, read_kb_ids
from xagent.core.tools.core.RAG_tools.kb.models import (
    KBAccessMode,
    KBBackendCapabilities,
    KBCollectionContext,
    KBContextRequest,
    KBDocumentRowsSnapshot,
    KBStorageBackend,
    KBUserScope,
)
from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import (
    embeddings_table_name,
    to_model_tag,
)
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
    KB_IDS_TABLE,
    ensure_documents_table,
)
from xagent.core.tools.core.RAG_tools.management import (
    collection_manager,
    collections,
)
from xagent.core.tools.core.RAG_tools.pipelines.document_ingestion import (
    _INGEST_TABLES,
    _compact_storage_if_needed,
)
from xagent.core.tools.core.RAG_tools.storage.contracts import DocumentRecord
from xagent.core.tools.core.RAG_tools.storage.factory import (
    get_ingestion_status_store,
    get_main_pointer_store,
    get_metadata_store,
    get_vector_index_store,
)
from xagent.core.tools.core.RAG_tools.utils import migration_utils
from xagent.core.tools.core.RAG_tools.version_management import cascade_cleaner
from xagent.providers.vector_store.milvus import MilvusConnectionManager
from xagent.web.services import kb_file_service

ASYNC_SEARCH = "search_dense_async search_sparse_async search_hybrid_async"
CASCADES = """cleanup_cascade cleanup_document_cascade cleanup_parse_cascade
    cleanup_chunk_cascade cleanup_embed_cascade"""
VERSIONS = """list_candidates promote_version_main capture_candidate_cleanup_snapshot
    restore_candidate_cleanup_snapshot"""
FAMILIES = {
    "supports_documents": """register_document load_document list_documents
        delete_document_record snapshot_document restore_document
        delete_created_document""",
    "supports_parses": """parse_exists read_parse_paragraphs write_parse
        read_latest_parse_record read_parse_paragraph_dicts delete_parse_records
        snapshot_parse restore_parse delete_created_parse""",
    "supports_chunks": """chunk_exists read_existing_chunks write_chunks
        delete_chunk_records snapshot_chunks restore_chunks delete_created_chunks""",
    "supports_embeddings": """read_chunks_needing_embedding write_embeddings
        commit_embeddings discard_uncommitted_embeddings delete_embedding_records
        snapshot_embeddings restore_embeddings delete_created_embeddings
        cleanup_embeddings_for_operation""",
    "supports_search": "validate_query_vector search_dense search_sparse search_hybrid",
    "supports_versions": f"{VERSIONS} {CASCADES}",
    "supports_async_search": ASYNC_SEARCH,
}
LEDGER = set(
    f"""{FAMILIES["supports_documents"]} {FAMILIES["supports_parses"]}
    {FAMILIES["supports_chunks"]} capture_document_rows rename_collection_status
    rename_collection_metadata count_documents retire_superseded_chunks
    list_collection_documents write_ingestion_status load_ingestion_status
    clear_ingestion_status write_ingestion_status_async load_ingestion_status_async
    clear_ingestion_status_async get_main_pointer set_main_pointer list_main_pointers
    delete_main_pointer capture_status_snapshot restore_status_snapshot
    clear_status_snapshot capture_main_pointer_snapshot
    restore_main_pointer_snapshot""".split()
)
UNSUPPORTED = {
    name: family
    for family, names in (
        ("async search", ASYNC_SEARCH),
        ("cascade cleanup", CASCADES),
        ("version candidates and promotion", VERSIONS),
    )
    for name in names.split()
}
IMPLEMENTED = {
    "collection_stats",
    "count_rows_by_document",
    "read_chunks_needing_embedding",
    "write_embeddings",
    "commit_embeddings",
    "validate_query_vector",
    "search_dense",
    "search_sparse",
    "search_hybrid",
    "discard_uncommitted_embeddings",
    "delete_documents_data",
    "delete_collection_data",
    "cleanup_collection_data_after_rollback",
    "delete_collection_config",
    "restore_document_rows",
    "rename_collection_data",
}
PENDING = (
    set(f"""{FAMILIES["supports_embeddings"]} {FAMILIES["supports_search"]}""".split())
    - IMPLEMENTED
)


@pytest.fixture
def milvus_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAGENT_VECTOR_BACKEND", "milvus")


def _context(backend: KBStorageBackend, collection: str = "kb") -> KBCollectionContext:
    return KBCollectionContext(
        collection=collection,
        user_scope=KBUserScope(user_id=1, is_admin=False),
        access_mode=KBAccessMode.WRITE,
        allow_create=True,
        hide_missing=True,
        metadata_store=get_metadata_store(),
        vector_index_store=get_vector_index_store(),
        ingestion_status_store=get_ingestion_status_store(),
        main_pointer_store=get_main_pointer_store(),
        backend=backend,
        capabilities=KBCoordinator._capabilities_for_backend(backend),
    )


def _handle() -> tuple[MilvusCollectionHandle, MagicMock, MagicMock]:
    ledger = MagicMock(spec=KBCollectionHandle)
    connections = MagicMock(spec=MilvusConnectionManager)
    handle = MilvusCollectionHandle(
        _context(KBStorageBackend.MILVUS), ledger=ledger, connections=connections
    )
    return handle, ledger, connections


async def _call(handle: MilvusCollectionHandle, name: str) -> Any:
    result = getattr(handle, name)("arg", key="value")
    return await result if inspect.isawaitable(result) else result


def test_every_interface_method_is_delegated_refused_pending_or_implemented() -> None:
    interface = {
        name
        for name, member in vars(KBCollectionHandle).items()
        if inspect.isfunction(member)
    }

    assert (len(LEDGER), len(UNSUPPORTED), len(PENDING), len(IMPLEMENTED)) == (
        44,
        12,
        5,
        16,
    )
    assert LEDGER | set(UNSUPPORTED) | PENDING | IMPLEMENTED == interface
    for name in interface:
        routed = getattr(MilvusCollectionHandle, name)
        declared = getattr(KBCollectionHandle, name)
        assert inspect.iscoroutinefunction(routed) == (
            inspect.iscoroutinefunction(declared)
        ), name
        assert routed.__qualname__ == f"MilvusCollectionHandle.{name}"
        if name not in IMPLEMENTED:
            assert routed.__doc__ == declared.__doc__, name


@pytest.mark.parametrize("name", sorted(LEDGER))
async def test_ledger_methods_go_to_the_injected_ledger_handle(name: str) -> None:
    handle, ledger, connections = _handle()

    assert await _call(handle, name) is getattr(ledger, name).return_value
    getattr(ledger, name).assert_called_once_with("arg", key="value")
    assert connections.mock_calls == []


@pytest.mark.parametrize(
    ("name", "error", "message"),
    [
        *(
            (name, ConfigurationError, f"not support {family} \\({name}\\)")
            for name, family in sorted(UNSUPPORTED.items())
        ),
        *(
            (name, NotImplementedError, f"{name} needs Milvus rows and is not")
            for name in sorted(PENDING)
        ),
    ],
)
async def test_other_methods_raise_before_reaching_any_store(
    name: str, error: type[Exception], message: str
) -> None:
    handle, ledger, connections = _handle()

    with pytest.raises(error, match=message):
        await _call(handle, name)
    assert ledger.mock_calls == connections.mock_calls == []


def test_capabilities_report_exactly_what_the_handle_serves() -> None:
    capabilities = KBBackendCapabilities.milvus()

    assert {field.name for field in fields(capabilities)} == {
        *FAMILIES,
        "supports_raw_connection",
    }
    for flag, names in FAMILIES.items():
        served = set(names.split()) <= LEDGER | IMPLEMENTED
        assert getattr(capabilities, flag) == served, flag
    assert capabilities.supports_raw_connection is False
    assert KBCoordinator._capabilities_for_backend(KBStorageBackend.MILVUS) == (
        capabilities
    )


def _kb_names(user_id: int | None = 1, is_admin: bool = False) -> dict[str, list[str]]:
    conn = get_vector_index_store().get_raw_connection()
    return read_kb_ids(conn, user_id=user_id, is_admin=is_admin)


def _owners_of_kb(*owners: int) -> dict[int, str]:
    conn = get_vector_index_store().get_raw_connection()
    return {owner: get_or_create_kb_id(conn, "kb", owner) for owner in owners}


def test_a_rename_moves_the_kb_ids_before_the_ledger_data() -> None:
    handle, ledger, connections = _handle()
    ids = _owners_of_kb(1, 2)
    seen: list[dict[str, list[str]]] = []

    def rename(*_: Any) -> list[str]:
        seen.append(_kb_names())
        return []

    ledger.rename_collection_data.side_effect = rename

    warnings: list[str] = []
    assert handle.rename_collection_data("new", 1, False, warnings) == []

    assert seen == [{"new": [ids[1]]}]
    assert _kb_names(2) == {"kb": [ids[2]]}
    ledger.rename_collection_data.assert_called_once_with("new", 1, False, warnings)
    assert connections.mock_calls == []


@pytest.mark.parametrize("failure", ["raises", "warns"])
def test_a_failed_ledger_rename_moves_the_kb_ids_back(failure: str) -> None:
    handle, ledger, _ = _handle()
    ids = _owners_of_kb(1, 2)
    if failure == "raises":
        ledger.rename_collection_data.side_effect = RuntimeError("no disk")
    else:
        ledger.rename_collection_data.return_value = ["chunks: no disk"]

    if failure == "raises":
        with pytest.raises(RuntimeError, match="no disk"):
            handle.rename_collection_data("new", None, True)
    else:
        assert handle.rename_collection_data("new", None, True) == ["chunks: no disk"]

    assert sorted(_kb_names(None, True)["kb"]) == sorted(ids.values())
    assert list(_kb_names(None, True)) == ["kb"]


def test_an_admin_rename_moves_every_owners_kb_id_and_keeps_the_values() -> None:
    handle, ledger, _ = _handle()
    ids = _owners_of_kb(1, 2)
    ledger.rename_collection_data.return_value = []

    assert handle.rename_collection_data("new", None, True) == []

    assert list(_kb_names(None, True)) == ["new"]
    assert sorted(_kb_names(None, True)["new"]) == sorted(ids.values())


def _write_documents(*owners: int, collection: str = "kb", conn: Any = None) -> None:
    """Add one document per owner through ``conn`` (the store's by default)."""
    store_conn = get_vector_index_store().get_raw_connection()
    ensure_documents_table(store_conn)
    rows = [
        {
            "collection": collection,
            "doc_id": f"doc-{owner}",
            "file_id": None,
            "source_path": f"/uploads/{owner}.txt",
            "file_type": "txt",
            "content_hash": "a" * 64,
            "uploaded_at": datetime.now(timezone.utc),
            "title": None,
            "language": None,
            "user_id": owner,
        }
        for owner in owners
    ]
    (conn if conn is not None else store_conn).open_table("documents").add(rows)


class _FailingCount:
    def __init__(self, table: Any) -> None:
        self._table = table

    def count_rows(self, *_: Any) -> int:
        raise OSError("table is busy")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._table, name)


def _fail_the_document_count(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = get_vector_index_store().get_raw_connection()
    open_table = conn.open_table
    monkeypatch.setattr(
        conn,
        "open_table",
        lambda name: (
            _FailingCount(open_table(name)) if name == "documents" else open_table(name)
        ),
    )


def test_the_config_rows_and_the_kb_ids_are_deleted_together() -> None:
    handle, ledger, _ = _handle()
    ids = _owners_of_kb(1, 2)
    ledger.delete_collection_config = AsyncMock(return_value=3)

    assert asyncio.run(handle.delete_collection_config(tenant_only=True)) == 3
    assert _kb_names(None, True) == {"kb": [ids[2]]}
    ledger.delete_collection_config.assert_awaited_once_with(tenant_only=True)

    assert asyncio.run(handle.delete_collection_config()) == 3
    assert _kb_names(None, True) == {}
    ledger.count_documents.assert_not_called()


def test_deleting_every_owners_kb_ids_does_not_wait_for_documents() -> None:
    handle, ledger, _ = _handle()
    _owners_of_kb(1, 2)
    ledger.delete_collection_config = AsyncMock(return_value=1)
    _write_documents(1, 2)

    asyncio.run(handle.delete_collection_config())

    assert _kb_names(None, True) == {}


def test_a_caller_whose_documents_remain_keeps_the_kb_id() -> None:
    handle, ledger, _ = _handle()
    ids = _owners_of_kb(1, 2)
    ledger.delete_collection_config = AsyncMock(return_value=1)
    _write_documents(1, 2)

    assert asyncio.run(handle.delete_collection_config(tenant_only=True)) == 1

    assert sorted(_kb_names(None, True)["kb"]) == sorted(ids.values())


def test_other_owners_and_other_collections_documents_do_not_keep_the_kb_id() -> None:
    handle, ledger, _ = _handle()
    ids = _owners_of_kb(1, 2)
    ledger.delete_collection_config = AsyncMock(return_value=1)
    _write_documents(2)
    _write_documents(1, collection="other")

    asyncio.run(handle.delete_collection_config(tenant_only=True))

    assert _kb_names(None, True) == {"kb": [ids[2]]}


def test_a_document_another_connection_wrote_keeps_the_kb_id() -> None:
    handle, ledger, _ = _handle()
    ids = _owners_of_kb(1)
    ledger.delete_collection_config = AsyncMock(return_value=1)
    _write_documents(2)
    store = get_vector_index_store()
    assert store.count_rows("documents", {"collection": "kb"}, 1, False) == 0
    other = lancedb.connect(store.get_raw_connection().uri)

    _write_documents(1, conn=other)
    asyncio.run(handle.delete_collection_config(tenant_only=True))

    assert _kb_names(None, True) == {"kb": [ids[1]]}


def test_a_failed_document_count_keeps_the_kb_id_and_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle, ledger, _ = _handle()
    _owners_of_kb(1, 2)
    ledger.delete_collection_config = AsyncMock(return_value=1)
    _write_documents(2)
    _fail_the_document_count(monkeypatch)

    with pytest.raises(
        DatabaseOperationError, match="Cannot count the documents of kb.*table is busy"
    ):
        asyncio.run(handle.delete_collection_config(tenant_only=True))

    assert len(_kb_names(None, True)["kb"]) == 2


def test_an_admin_scope_deletes_only_its_own_kb_id_and_leaves_other_owners() -> None:
    ledger = MagicMock(spec=KBCollectionHandle)
    admin = replace(_context(KBStorageBackend.MILVUS), user_scope=KBUserScope(7, True))
    handle = MilvusCollectionHandle(
        admin, ledger=ledger, connections=MagicMock(spec=MilvusConnectionManager)
    )
    ids = _owners_of_kb(1, 2, 7)
    ledger.delete_collection_config = AsyncMock(return_value=0)
    _write_documents(1, 2)

    asyncio.run(handle.delete_collection_config(tenant_only=True))

    assert sorted(_kb_names(None, True)["kb"]) == sorted([ids[1], ids[2]])


def test_the_discard_reads_the_kb_id_and_creates_none() -> None:
    handle, _, connections = _handle()

    assert handle.discard_uncommitted_embeddings("doc", user_id=5) == 0

    assert _kb_names(None, True) == {}
    connections.get_shared_client_from_env.assert_not_called()


@pytest.mark.parametrize("failure", ["raises", "warns"])
def test_a_failed_move_back_does_not_replace_the_rename_failure(
    failure: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    handle, ledger, _ = _handle()
    ids = _owners_of_kb(1)

    def lock_timeout(*_: Any) -> None:
        raise OSError("lock timeout")

    monkeypatch.setattr(collection_handle, "set_kb_ids_collection", lock_timeout)
    if failure == "raises":
        ledger.rename_collection_data.side_effect = RuntimeError("no disk")
        with pytest.raises(RuntimeError, match="no disk"):
            handle.rename_collection_data("new", 1, False)
    else:
        ledger.rename_collection_data.return_value = ["chunks: no disk"]
        assert handle.rename_collection_data("new", 1, False) == ["chunks: no disk"]

    assert "Could not move the kb_ids back to kb: lock timeout" in caplog.text
    assert _kb_names() == {"new": [ids[1]]}


def test_two_kb_ids_for_one_owner_fail_a_delete_before_any_store_is_touched() -> None:
    handle, ledger, connections = _handle()
    _owners_of_kb(1)
    get_vector_index_store().get_raw_connection().open_table(KB_IDS_TABLE).add(
        [
            {
                "collection": "kb",
                "user_id": 1,
                "kb_id": "duplicate",
                "created_at": datetime.now(timezone.utc),
            }
        ]
    )

    with pytest.raises(DatabaseOperationError, match="several kb_ids"):
        handle.delete_documents_data(["doc"], user_id=1, is_admin=False)

    ledger.delete_documents_data.assert_not_called()
    connections.get_shared_client_from_env.assert_not_called()


def _restore_snapshot(statuses: dict[str, str]) -> KBDocumentRowsSnapshot:
    """A snapshot whose documents have the given statuses and one chunk each."""
    return KBDocumentRowsSnapshot(
        collection="kb",
        doc_ids=tuple(statuses),
        rows_by_table={
            "ingestion_runs": [
                {
                    "collection": "kb",
                    "doc_id": doc_id,
                    "status": status,
                    "message": f"{doc_id} message",
                    "parse_hash": "p1",
                    "user_id": 1,
                }
                for doc_id, status in statuses.items()
            ],
            "chunks": [
                {
                    "collection": "kb",
                    "doc_id": doc_id,
                    "parse_hash": "p1",
                    "chunk_id": f"{doc_id}-c",
                }
                for doc_id in statuses
            ],
        },
    )


def _marked(ledger: MagicMock) -> list[str]:
    return [call.args[0] for call in ledger.write_ingestion_status.call_args_list]


def test_a_restore_marks_only_a_success_or_partially_embedded_status() -> None:
    handle, ledger, connections = _handle()
    connections.get_shared_client_from_env.return_value.list_collections.return_value = []
    snapshot = _restore_snapshot(
        {
            "ok": "success",
            "part": "partially_embedded",
            "failed": "failed",
            "chunked": "chunked",
            "cancelled": "cancelled",
        }
    )

    restored = handle.restore_document_rows(snapshot, user_id=1, is_admin=False)

    assert restored == ["ok", "part"] and _marked(ledger) == ["ok", "part"]
    message = ledger.write_ingestion_status.call_args.kwargs["message"]
    assert "re-ingest" in message


def _failing_client(connections: MagicMock, *, fails_on: str) -> MagicMock:
    client = connections.get_shared_client_from_env.return_value
    client.list_collections.return_value = ["xagent_kb_m"]
    client.has_collection.return_value = True

    def query(name: str, **kwargs: Any) -> list[dict[str, Any]]:
        if kwargs["filter_params"]["doc_id"] == fails_on:
            raise RuntimeError("transient")
        return []

    client.query.side_effect = query
    return client


def test_a_failed_alignment_marks_the_documents_not_aligned_yet_and_reports_them() -> (
    None
):
    handle, ledger, connections = _handle()
    _owners_of_kb(1)
    _failing_client(connections, fails_on="d2")
    snapshot = _restore_snapshot(
        {"d1": "success", "d2": "success", "d3": "failed", "d4": "partially_embedded"}
    )

    with pytest.raises(
        DatabaseOperationError,
        match="^Could not align Milvus with the restored chunks: transient",
    ) as raised:
        handle.restore_document_rows(snapshot, user_id=1, is_admin=False)

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert raised.value.details == {"marked": ["d1", "d2", "d4"]}
    assert _marked(ledger) == ["d1", "d2", "d4"]


def test_a_failed_mark_is_logged_and_the_others_are_still_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    handle, ledger, connections = _handle()
    _owners_of_kb(1)
    _failing_client(connections, fails_on="d2")
    ledger.write_ingestion_status.side_effect = [None, OSError("disk full"), None]
    snapshot = _restore_snapshot({"d1": "success", "d2": "success", "d3": "success"})

    with pytest.raises(DatabaseOperationError, match="transient") as raised:
        handle.restore_document_rows(snapshot, user_id=1, is_admin=False)

    assert raised.value.details == {"marked": ["d1", "d3"]}
    assert "Could not mark d2 partially embedded: disk full" in caplog.text


def test_a_failed_mark_is_reported_as_a_mark_and_the_rest_are_still_marked() -> None:
    handle, ledger, connections = _handle()
    _owners_of_kb(1)
    _failing_client(connections, fails_on="none")
    ledger.write_ingestion_status.side_effect = [OSError("disk full"), None, None]
    snapshot = _restore_snapshot({"d1": "success", "d2": "success"})

    with pytest.raises(
        DatabaseOperationError,
        match="^Could not mark d1 partially embedded: disk full",
    ) as raised:
        handle.restore_document_rows(snapshot, user_id=1, is_admin=False)

    assert raised.value.details == {"marked": ["d1", "d2"]}


@pytest.mark.parametrize("method", ["documents", "collection"])
@pytest.mark.parametrize("failure", ["no model", "describe fails"])
def test_a_delete_resolves_the_count_keys_before_it_deletes(
    method: str, failure: str
) -> None:
    handle, ledger, connections = _handle()
    _owners_of_kb(1)
    client = connections.get_shared_client_from_env.return_value
    client.list_collections.return_value = ["xagent_kb_m"]
    if failure == "no model":
        client.describe_collection.return_value = {"properties": {}}
    else:
        client.describe_collection.side_effect = RuntimeError("describe failed")

    with pytest.raises(DatabaseOperationError, match="Cannot resolve the models"):
        if method == "documents":
            handle.delete_documents_data(["d"], user_id=1, is_admin=False)
        else:
            handle.delete_collection_data(user_id=1, is_admin=False)

    client.delete.assert_not_called()
    ledger.delete_documents_data.assert_not_called()
    ledger.delete_collection_data.assert_not_called()


def test_the_client_comes_from_the_connection_manager_on_first_use() -> None:
    handle, _ledger, connections = _handle()
    connections.get_shared_client_from_env.assert_not_called()

    assert handle.client is connections.get_shared_client_from_env.return_value
    assert handle.client is connections.get_shared_client_from_env.return_value
    connections.get_shared_client_from_env.assert_called_once_with()


def test_handles_opened_by_the_provider_share_one_client(
    milvus_deployment: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MilvusConnectionManager, "_shared_clients", {})
    monkeypatch.setenv("MILVUS_URI", "http://milvus.test:19530")
    monkeypatch.setattr(
        MilvusConnectionManager, "get_client", lambda self, **settings: object()
    )
    provider = KBHandleProvider()

    first = provider.open(_context(KBStorageBackend.MILVUS))
    second = provider.open(_context(KBStorageBackend.MILVUS, "other"))

    assert isinstance(first, MilvusCollectionHandle)
    assert isinstance(second, MilvusCollectionHandle)
    assert first.client is second.client


def test_the_provider_hands_its_connection_manager_to_the_handles(
    milvus_deployment: None,
) -> None:
    connections = MagicMock(spec=MilvusConnectionManager)

    handle = KBHandleProvider(connections=connections).open(
        _context(KBStorageBackend.MILVUS)
    )

    assert isinstance(handle, MilvusCollectionHandle)
    assert handle.connections is connections
    assert handle.client is connections.get_shared_client_from_env.return_value


def test_milvus_collection_names_follow_the_model_id() -> None:
    trio = ("BAAI/bge-m3", "baai-bge-m3", "baai_bge_m3")
    names = [milvus_collection_name(model) for model in trio]

    assert len(set(names)) == 3
    assert names == [
        "xagent_kb_BAAI_bge_m3_2022e1cf",
        "xagent_kb_baai_bge_m3_66328cf3",
        "xagent_kb_baai_bge_m3_48034a0a",
    ]
    assert milvus_collection_name(" BAAI/bge-m3 ") == names[0]
    assert milvus_collection_name(to_model_tag(trio[0])) != names[0]


@pytest.mark.parametrize(
    "model",
    [
        "bge-m3",
        "BAAI/bge-m3",
        "智谱/glm 4.5-embedding",
        "vendor/" + "long-model-name-" * 40,
        "",
        "///",
    ],
)
def test_milvus_collection_names_fit_the_milvus_name_rules(model: str) -> None:
    name = milvus_collection_name(model)

    assert re.fullmatch(r"xagent_kb_[A-Za-z0-9_]*_[0-9a-f]{8}", name), name
    assert len(name) < 255
    assert milvus_collection_name(model) == name


def test_milvus_collection_names_of_one_long_prefix_stay_apart() -> None:
    prefix = "vendor/" + "x" * 300

    assert milvus_collection_name(prefix + "a") != milvus_collection_name(prefix + "b")


MODEL = "BAAI/bge-m3"
PROPERTY = "xagent.model_id"


def _described(
    model: str | None = MODEL, dimension: int = 3, *, dense: bool = True
) -> dict[str, Any]:
    fields: list[dict[str, Any]] = [{"name": "chunk_id", "params": {}}]
    if dense:
        fields.append({"name": "dense", "params": {"dim": dimension}})
    return {"properties": {} if model is None else {PROPERTY: model}, "fields": fields}


def _milvus(
    *, exists: list[bool], described: dict[str, Any] | None = None, loaded: bool = False
) -> MagicMock:
    client = MagicMock()
    client.has_collection.side_effect = exists
    client.describe_collection.return_value = described or _described()
    state = SimpleNamespace(name="Loaded" if loaded else "NotLoad")
    client.get_load_state.return_value = {"state": state}
    return client


def _indexed_and_loaded(client: MagicMock) -> bool:
    return client.create_index.called and client.load_collection.called


def test_a_new_collection_records_its_model_then_is_indexed_and_loaded() -> None:
    client = _milvus(exists=[False])

    name = ensure_milvus_collection(client, f" {MODEL} ", 3)

    assert name == milvus_collection_name(MODEL)
    (create,) = client.create_collection.call_args_list
    assert create.args == (name,)
    assert create.kwargs["properties"] == {PROPERTY: MODEL}
    client.describe_collection.assert_called_once_with(name)
    client.load_collection.assert_called_once_with(name)
    assert _indexed_and_loaded(client)


def test_a_loaded_collection_of_the_model_is_left_alone() -> None:
    client = _milvus(exists=[True], loaded=True)

    assert ensure_milvus_collection(client, MODEL, 3) == milvus_collection_name(MODEL)
    client.create_collection.assert_not_called()
    assert not client.create_index.called and not client.load_collection.called


def test_a_collection_created_by_a_concurrent_caller_is_used_after_the_check() -> None:
    client = _milvus(exists=[False, True])
    client.create_collection.side_effect = RuntimeError("collection already exists")

    name = ensure_milvus_collection(client, MODEL, 3)

    client.describe_collection.assert_called_once_with(name)
    assert _indexed_and_loaded(client)


@pytest.mark.parametrize(
    ("described", "message"),
    [
        (_described(dimension=4), "4-dimensional vectors, but 'BAAI/bge-m3'.*3-dim"),
        (_described("other/model"), "model 'other/model', not 'BAAI/bge-m3'"),
        (_described(None), "model None, not 'BAAI/bge-m3'"),
        (_described(dense=False), "has no dense field"),
    ],
    ids=["dimension", "model", "no-model", "no-dense"],
)
@pytest.mark.parametrize("raced", [False, True], ids=["existing", "concurrent"])
def test_a_collection_that_does_not_fit_is_refused_before_index_and_load(
    described: dict[str, Any], message: str, raced: bool
) -> None:
    client = _milvus(exists=[False, True] if raced else [True], described=described)
    if raced:
        client.create_collection.side_effect = RuntimeError("already exists")
    name = milvus_collection_name(MODEL)

    with pytest.raises(VectorValidationError, match=f"{name}.*{message}"):
        ensure_milvus_collection(client, MODEL, 3)
    assert not client.create_index.called and not client.load_collection.called


def test_a_failed_create_with_no_collection_behind_it_is_raised() -> None:
    client = _milvus(exists=[False, False])
    client.create_collection.side_effect = RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        ensure_milvus_collection(client, MODEL, 3)
    client.describe_collection.assert_not_called()


def _write(
    handle: MilvusCollectionHandle, model: str, dimension: int, user: int
) -> None:
    handle.write_embeddings(
        [
            ChunkEmbeddingData(
                doc_id="doc",
                chunk_id="chunk",
                parse_hash="ph",
                model=model,
                vector=[1.0] * dimension,
                text="text",
                chunk_hash="h",
            )
        ],
        user_id=user,
    )


def test_an_ingest_scope_resolves_each_collection_and_kb_id_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensured: list[tuple[str, int]] = []
    owners: list[tuple[str, int | None]] = []
    monkeypatch.setattr(
        collection_handle,
        "ensure_milvus_collection",
        lambda client, model, dimension: ensured.append((model, dimension))
        or f"{model}-{dimension}",
    )
    monkeypatch.setattr(
        collection_handle,
        "get_or_create_kb_id",
        lambda conn, collection, user_id: owners.append((collection, user_id))
        or f"{collection}-{user_id}",
    )
    handle, _ledger, connections = _handle()
    other = MilvusCollectionHandle(
        _context(KBStorageBackend.MILVUS, "other"),
        ledger=MagicMock(spec=KBCollectionHandle),
        connections=connections,
    )

    with collection_handle.ingest_scope():
        for _ in range(3):
            _write(handle, "m1", 3, 1)
        _write(handle, "m2", 3, 1)
        _write(handle, "m1", 4, 1)
        _write(handle, "m1", 3, 2)
        _write(other, "m1", 3, 1)

    assert ensured == [("m1", 3), ("m2", 3), ("m1", 4)]
    assert owners == [("kb", 1), ("kb", 2), ("other", 1)]
    upserts = connections.get_shared_client_from_env.return_value.upsert.call_args_list
    assert [(call.args[0], call.args[1][0]["kb_id"]) for call in upserts] == [
        ("m1-3", "kb-1"),
        ("m1-3", "kb-1"),
        ("m1-3", "kb-1"),
        ("m2-3", "kb-1"),
        ("m1-4", "kb-1"),
        ("m1-3", "kb-2"),
        ("m1-3", "other-1"),
    ]


def test_each_ingest_scope_starts_empty_and_nothing_outlives_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensured: list[str] = []
    monkeypatch.setattr(
        collection_handle,
        "ensure_milvus_collection",
        lambda client, model, dimension: ensured.append(model) or model,
    )
    monkeypatch.setattr(collection_handle, "get_or_create_kb_id", lambda *args: "kb-id")
    handle, _ledger, _connections = _handle()

    _write(handle, "m1", 3, 1)
    with collection_handle.ingest_scope():
        _write(handle, "m1", 3, 1)
        _write(handle, "m1", 3, 1)
        with collection_handle.ingest_scope():
            _write(handle, "m1", 3, 1)
        _write(handle, "m1", 3, 1)
    with collection_handle.ingest_scope():
        _write(handle, "m1", 3, 1)
    _write(handle, "m1", 3, 1)

    assert ensured == ["m1"] * 5


@pytest.mark.parametrize(
    ("doc_id", "parse_hash", "model"),
    [("", "ph", "m"), ("doc", "", "m"), ("doc", "ph", "")],
)
def test_the_pending_read_requires_its_identifiers(
    doc_id: str, parse_hash: str, model: str
) -> None:
    handle, ledger, connections = _handle()

    with pytest.raises(DocumentValidationError, match="are required"):
        handle.read_chunks_needing_embedding(doc_id, parse_hash, model)
    assert ledger.mock_calls == connections.mock_calls == []


@pytest.mark.parametrize("model", ["BAAI/Bge-M3-test", "bge-m3", "Vendor/Model.V2"])
def test_per_document_keys_are_the_keys_lancedb_reports(model: str) -> None:
    lancedb = KBHandleProvider().open(_context(KBStorageBackend.LANCEDB))
    lancedb.write_embeddings(
        [
            ChunkEmbeddingData(
                doc_id="doc",
                chunk_id="chunk",
                parse_hash="ph",
                model=model,
                vector=[1.0, 0.0, 0.0],
                text="kiwi",
                chunk_hash="h",
            )
        ]
    )
    (lancedb_key,) = lancedb.count_rows_by_document(user_id=None, is_admin=True)["doc"]
    client = _milvus(exists=[True], described=_described(model))

    assert collection_handle._embeddings_key(client, "any") == lancedb_key
    assert lancedb_key == embeddings_table_name(to_model_tag(model))


def test_a_collection_that_records_no_model_has_no_per_document_key() -> None:
    client = _milvus(exists=[True], described=_described(None))

    with pytest.raises(VectorValidationError, match="xagent_kb_stray.*no xagent.model"):
        collection_handle._embeddings_key(client, "xagent_kb_stray")


def _iterator(*batches: list[dict[str, Any]]) -> MagicMock:
    iterator = MagicMock()
    iterator.next.side_effect = [*batches, []]
    return iterator


def test_models_that_tag_alike_add_up_under_one_per_document_key() -> None:
    handle, ledger, connections = _handle()
    ledger.count_rows_by_document.return_value = {"d": {"chunks": 3}}
    get_or_create_kb_id(get_vector_index_store().get_raw_connection(), "kb", 1)
    ids = ("baai-bge-m3", "baai_bge_m3")
    names = {milvus_collection_name(model): model for model in ids}
    client = connections.get_shared_client_from_env.return_value
    client.list_collections.return_value = [*names, "other"]
    client.describe_collection.side_effect = lambda name: _described(names[name])
    batches = dict(
        zip(names, (_iterator([{"doc_id": "d"}] * 2), _iterator([{"doc_id": "d"}])))
    )
    client.query_iterator.side_effect = lambda name, **_: batches[name]

    counts = handle.count_rows_by_document(user_id=1, is_admin=False)

    assert counts == {"d": {"chunks": 3, "embeddings_baai_bge_m3": 3}}


class _NotLoaded(Exception):
    code = 101


def test_a_collection_that_is_not_loaded_is_queried_and_warned_about_once_per_call(
    milvus_deployment: None, caplog: pytest.LogCaptureFixture
) -> None:
    conn = get_vector_index_store().get_raw_connection()
    chunk = {"chunk_id": "c", "index": 0, "text": "kiwi", "created_at": None}
    for collection in ("kb", "kb2"):
        KBHandleProvider().open(
            _context(KBStorageBackend.MILVUS, collection)
        ).write_chunks("doc", "ph", "cfg", {}, [chunk])
        get_or_create_kb_id(conn, collection, None)
    client = MagicMock()
    client.list_collections.return_value = ["xagent_kb_unloaded", "xagent_kb_ok", "x"]

    def query(name: str, **_: Any) -> list[dict[str, int]]:
        if name == "xagent_kb_unloaded":
            raise _NotLoaded("collection not loaded")
        return [{"count(*)": 2}]

    client.query.side_effect = query
    connections = MagicMock(spec=MilvusConnectionManager)
    connections.get_shared_client_from_env.return_value = client
    caplog.set_level(logging.WARNING)

    stats = KBHandleProvider(connections=connections).aggregate_collection_stats(
        user_id=None, is_admin=True
    )

    assert {name: row["embeddings"] for name, row in stats.items()} == {
        "kb": 2,
        "kb2": 2,
    }
    queried = [call.args[0] for call in client.query.call_args_list]
    assert queried.count("xagent_kb_unloaded") == 1
    assert caplog.text.count("xagent_kb_unloaded is not loaded") == 1


def test_the_provider_dispatches_on_the_engine(milvus_deployment: None) -> None:
    context = _context(KBStorageBackend.MILVUS)

    handle = KBHandleProvider().open(context)

    assert isinstance(handle, MilvusCollectionHandle)
    assert handle.context is context
    assert type(handle.ledger) is LanceDBCollectionHandle
    assert handle.ledger.context is context
    assert type(handle.connections) is MilvusConnectionManager
    lancedb = KBHandleProvider().open(_context(KBStorageBackend.LANCEDB))
    assert type(lancedb) is LanceDBCollectionHandle
    with pytest.raises(ValueError, match="'qdrant' is not supported"):
        KBHandleProvider().open(_context(KBStorageBackend.QDRANT))


def test_a_milvus_deployment_opens_milvus_handles(milvus_deployment: None) -> None:
    handle = kb.get_kb_coordinator().open_collection_sync(
        KBContextRequest(collection="new", user_id=1, hide_missing=True)
    )

    assert isinstance(handle, MilvusCollectionHandle)
    assert handle.context.capabilities == KBBackendCapabilities.milvus()
    handle.write_ingestion_status("doc", status="running", user_id=1)
    assert [row["doc_id"] for row in handle.load_ingestion_status(user_id=1)] == ["doc"]


def test_a_lancedb_deployment_refuses_a_milvus_binding(tmp_path: Path) -> None:
    binding = {"kb_storage": {"backend": "milvus"}}
    asyncio.run(
        get_metadata_store().save_collection(
            CollectionInfo(name="kb", extra_metadata=binding)
        )
    )
    source = tmp_path / "a.txt"
    source.write_text("kiwi", encoding="utf-8")
    coordinator = kb.get_kb_coordinator()

    mismatch = "bound to the milvus engine, but this deployment runs lancedb"
    with pytest.raises(ValueError, match=mismatch):
        coordinator.register_document_sync(
            RegisterDocumentRequest(
                collection="kb", source_path=str(source), doc_id="d", user_id=1
            )
        )
    with pytest.raises(ValueError, match=mismatch):
        coordinator.write_ingestion_status_sync("kb", "d", status="running", user_id=1)
    ledger = KBHandleProvider().open(_context(KBStorageBackend.LANCEDB))
    assert ledger.count_documents(None, True) == 0
    assert ledger.load_ingestion_status(is_admin=True) == []


def test_stats_without_kb_ids_never_reach_milvus(
    milvus_deployment: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    handle = KBHandleProvider().open(_context(KBStorageBackend.MILVUS))
    chunk = {"chunk_id": "c", "index": 0, "text": "kiwi", "created_at": None}
    handle.write_chunks("doc", "ph", "cfg", {}, [chunk])
    monkeypatch.setattr(
        MilvusConnectionManager,
        "get_shared_client_from_env",
        MagicMock(side_effect=AssertionError("Milvus was reached")),
    )

    stats = KBHandleProvider().aggregate_collection_stats(user_id=None, is_admin=True)
    assert stats["kb"]["chunks"] == 1 and stats["kb"]["embeddings"] == 0
    assert handle.collection_stats(None, True)["embeddings"] == 0
    assert handle.count_rows_by_document(user_id=None, is_admin=True) == {
        "doc": {"chunks": 1}
    }


def test_batched_stats_refuse_an_unsupported_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        collection_handle, "deployment_kb_backend", lambda: KBStorageBackend.QDRANT
    )
    with pytest.raises(ValueError, match="'qdrant' is not supported"):
        KBHandleProvider().aggregate_collection_stats(user_id=None, is_admin=True)


def test_lancedb_only_paths_follow_the_deployment_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert ledger_holds_vectors()
    monkeypatch.setenv("XAGENT_VECTOR_BACKEND", "milvus")
    assert not ledger_holds_vectors()
    monkeypatch.setenv("XAGENT_VECTOR_BACKEND", "qdrant")
    with pytest.raises(ConfigurationError, match="not implemented"):
        ledger_holds_vectors()


async def test_metadata_rebuild_keeps_the_model_without_lancedb_vectors(
    milvus_deployment: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    listed = CollectionInfo(
        name="kb", embeddings=3, embedding_model_id="m", embedding_dimension=3
    )
    monkeypatch.setattr(
        collections,
        "list_collections",
        AsyncMock(return_value=SimpleNamespace(status="success", collections=[listed])),
    )
    store, saved = MagicMock(), AsyncMock()
    monkeypatch.setattr(collection_manager, "get_vector_index_store", lambda: store)
    monkeypatch.setattr(collection_manager.collection_manager, "save_collection", saved)

    await collection_manager._rebuild_collection_metadata_impl()

    assert store.method_calls == []
    saved.assert_awaited_once_with(listed)


def test_file_statuses_skip_the_legacy_indexed_fallback(
    milvus_deployment: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = MagicMock()
    store.list_document_records_by_file_ids.return_value = [
        DocumentRecord(doc_id="d", file_id="f", user_id=1, collection="kb")
    ]
    monkeypatch.setattr(kb_file_service, "get_vector_index_store", lambda: store)
    monkeypatch.setattr(kb_file_service, "_load_ingestion_status_impl", lambda **_: [])

    assert kb_file_service._aggregate_uploaded_file_statuses_impl(
        file_ids=["f"], user_id=1, is_admin=False, use_cache=False
    ) == {"f": "UNKNOWN"}
    store.list_indexed_doc_refs.assert_not_called()


def test_stale_file_cleanup_deletes_through_the_handle(
    milvus_deployment: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = MagicMock()
    query = db.query.return_value.filter.return_value.order_by.return_value
    query.all.return_value = [SimpleNamespace(file_id="f", created_at=None)]
    store = MagicMock()
    store.list_document_records_by_file_ids.return_value = [
        DocumentRecord(doc_id="d", file_id="f", user_id=1, collection="kb")
    ]
    monkeypatch.setattr(kb_file_service, "get_vector_index_store", lambda: store)
    monkeypatch.setattr(
        kb_file_service,
        "_aggregate_uploaded_file_statuses_impl",
        lambda **_: {"f": "FAILED"},
    )
    coordinator = MagicMock()
    coordinator.delete_documents_data_sync.side_effect = NotImplementedError
    monkeypatch.setattr(kb, "get_kb_coordinator", lambda: coordinator)

    result = kb_file_service._reconcile_uploaded_files_impl(
        db, user_id=1, is_admin=False
    )

    coordinator.delete_documents_data_sync.assert_called_once_with(
        "kb", ["d"], user_id=1, is_admin=False
    )
    store.cascade_delete.assert_not_called()
    assert (result["deleted"], result["cleanup_errors"]) == (0, 1)


def test_version_cascade_delete_is_refused(milvus_deployment: None) -> None:
    with pytest.raises(ConfigurationError, match="does not support cascade cleanup"):
        cascade_cleaner.cascade_delete(
            target="collection", collection="kb", preview_only=False, confirm=True
        )


def test_search_time_model_inference_reads_no_lancedb_tables(
    milvus_deployment: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    connect = MagicMock()
    monkeypatch.setattr(migration_utils, "get_vector_store_raw_connection", connect)

    assert migration_utils._infer_embedding_config_from_collection("kb") == (
        None,
        None,
    )
    connect.assert_not_called()


def test_compaction_without_embeddings_tables_compacts_the_ledger(
    milvus_deployment: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = get_vector_index_store()
    handle = KBHandleProvider().open(_context(KBStorageBackend.MILVUS))
    handle.write_ingestion_status("doc", status="success", user_id=1)
    requested: list[list[str]] = []
    compact = store.compact_tables
    monkeypatch.setattr(
        store,
        "compact_tables",
        lambda names, policy=None: requested.append(names) or compact(names, policy),
    )

    _compact_storage_if_needed("model")

    assert requested == [[*_INGEST_TABLES, embeddings_table_name("model")]]
    assert not [name for name in store.list_table_names() if "embeddings_" in name]
    assert "compaction skipped" not in caplog.text


class _Unloaded(Exception):
    code = 101


def _described_client(properties: dict[str, str]) -> MagicMock:
    client = MagicMock()
    client.describe_collection.return_value = {
        "properties": properties,
        "fields": [{"name": "dense", "params": {"dim": "3"}}],
    }
    return client


def _attempts(*outcomes: Any) -> Any:
    remaining = iter(outcomes)

    def call() -> Any:
        outcome = next(remaining)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return call


def test_a_call_on_an_unloaded_collection_loads_it_once_and_runs_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensured: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        collection_handle,
        "ensure_milvus_collection",
        lambda *args: ensured.append(args),
    )
    client = _described_client({PROPERTY: MODEL})

    result = collection_handle._with_loaded(
        "xagent_kb_x", client, _attempts(_Unloaded(), "done")
    )

    assert result == "done" and ensured == [(client, MODEL, 3)]


def test_other_errors_and_a_second_unloaded_failure_are_not_retried_blindly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensured: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        collection_handle,
        "ensure_milvus_collection",
        lambda *args: ensured.append(args),
    )
    client = _described_client({PROPERTY: MODEL})

    with pytest.raises(RuntimeError, match="down"):
        collection_handle._with_loaded("n", client, _attempts(RuntimeError("down")))
    assert ensured == []
    with pytest.raises(_Unloaded):
        collection_handle._with_loaded("n", client, _attempts(_Unloaded(), _Unloaded()))
    assert len(ensured) == 1


@pytest.mark.parametrize("invisible_only", [False, True])
def test_a_delete_skips_an_unloaded_collection_only_for_invisible_rows(
    invisible_only: bool,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(collection_handle, "ensure_milvus_collection", lambda *_: None)
    client = _described_client({PROPERTY: MODEL})
    client.delete.side_effect = _Unloaded("not loaded")

    if invisible_only:
        counts = collection_handle._delete_milvus_rows(
            client, ["xagent_kb_x"], ["kb"], invisible_only=True
        )
        assert counts == {} and "xagent_kb_x is not loaded; skipped" in caplog.text
    else:
        with pytest.raises(DatabaseOperationError, match="Cannot delete from Milvus"):
            collection_handle._delete_milvus_rows(client, ["xagent_kb_x"], ["kb"])


def test_a_collection_without_a_recorded_model_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensured: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        collection_handle,
        "ensure_milvus_collection",
        lambda *args: ensured.append(args),
    )
    client = _described_client({})

    with pytest.raises(VectorValidationError, match=r"records no xagent\.model_id"):
        collection_handle._embeddings_key(client, "xagent_kb_x")
    with pytest.raises(VectorValidationError, match=r"records no xagent\.model_id"):
        collection_handle._with_loaded(
            "xagent_kb_x", client, _attempts(_Unloaded(), "never")
        )
    assert ensured == []
