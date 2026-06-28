"""Tests for RAG management utilities."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.xagent.core.tools.core.RAG_tools.core.exceptions import DatabaseOperationError
from src.xagent.core.tools.core.RAG_tools.core.schemas import DocumentProcessingStatus
from src.xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import (
    embeddings_table_name,
    to_model_tag,
)
from src.xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
    ensure_chunks_table,
    ensure_documents_table,
    ensure_embeddings_table,
    ensure_parses_table,
)
from src.xagent.core.tools.core.RAG_tools.management import (
    cancel_collection,
    cancel_document,
)
from src.xagent.core.tools.core.RAG_tools.management import (
    collections as collections_module,
)
from src.xagent.core.tools.core.RAG_tools.management import (
    delete_collection,
    delete_document,
    get_document_stats,
    list_collections,
    list_documents,
    retry_document,
)
from src.xagent.core.tools.core.RAG_tools.management.status import (
    load_ingestion_status,
    write_ingestion_status,
)
from src.xagent.core.tools.core.RAG_tools.storage import get_vector_index_store
from src.xagent.core.tools.core.RAG_tools.storage.factory import get_metadata_store
from src.xagent.providers.vector_store.lancedb import get_connection_from_env
from xagent.core.tools.core.RAG_tools.file.register_document import register_document


@pytest.fixture()
def temp_lancedb_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    """Isolate LANCE DB data directory per test."""

    original = os.environ.get("LANCEDB_DIR")
    monkeypatch.setenv("LANCEDB_DIR", str(tmp_path))
    from src.xagent.core.tools.core.RAG_tools.storage.factory import StorageFactory

    StorageFactory.get_factory().reset_all()
    yield str(tmp_path)
    if original is None:
        monkeypatch.delenv("LANCEDB_DIR", raising=False)
    else:
        monkeypatch.setenv("LANCEDB_DIR", original)


def _insert_documents(records: List[Dict[str, object]]) -> None:
    conn = get_vector_index_store().get_raw_connection()
    ensure_documents_table(conn)
    table = conn.open_table("documents")

    # Add user_id field to records if not present
    for r in records:
        if "user_id" not in r:
            r["user_id"] = None  # Legacy data

    table.add(records)

    # Sync with metadata table
    from xagent.core.tools.core.RAG_tools.management.collection_manager import (
        update_collection_stats_sync,
    )

    for r in records:
        update_collection_stats_sync(
            collection_name=str(r["collection"]),
            documents_delta=1,
            document_name=os.path.basename(str(r["source_path"])),
        )


def _insert_parses(records: List[Dict[str, object]]) -> None:
    conn = get_vector_index_store().get_raw_connection()
    ensure_parses_table(conn)
    table = conn.open_table("parses")
    table.add(records)

    # Sync with metadata table
    from xagent.core.tools.core.RAG_tools.management.collection_manager import (
        update_collection_stats_sync,
    )

    for r in records:
        update_collection_stats_sync(
            collection_name=str(r["collection"]),
            parses_delta=1,
        )


def _insert_chunks(records: List[Dict[str, object]]) -> None:
    conn = get_vector_index_store().get_raw_connection()
    ensure_chunks_table(conn)
    table = conn.open_table("chunks")
    table.add(records)


def _insert_embeddings(model_name: str, records: List[Dict[str, object]]) -> None:
    conn = get_vector_index_store().get_raw_connection()
    ensure_embeddings_table(conn, to_model_tag(model_name), vector_dim=3)
    table = conn.open_table(embeddings_table_name(model_name))
    table.add(records)

    # Sync with metadata table
    from xagent.core.tools.core.RAG_tools.management.collection_manager import (
        update_collection_stats_sync,
    )

    for r in records:
        update_collection_stats_sync(
            collection_name=str(r["collection"]),
            embeddings_delta=1,
        )


@pytest.mark.asyncio
async def test_list_collections_empty(temp_lancedb_dir: str) -> None:
    """When no data exists the result should be empty but successful."""

    result = await list_collections(user_id=None, is_admin=True)

    assert result.status == "success"
    assert result.total_count == 0
    assert result.collections == []
    assert result.warnings == []


@pytest.mark.asyncio
async def test_list_collections_with_data(temp_lancedb_dir: str) -> None:
    """Aggregate statistics should include counts per collection and document names."""

    collection = "demo_collection"
    doc_id = "doc-1"
    now = datetime.now(timezone.utc)

    _insert_documents(
        [
            {
                "collection": collection,
                "doc_id": doc_id,
                "source_path": "/path/sample.pdf",
                "file_type": "pdf",
                "content_hash": "hash-doc-1",
                "uploaded_at": now,
                "title": "Sample",
                "language": "zh",
            },
            {
                "collection": collection,
                "doc_id": "doc-2",
                "source_path": "/path/other.pdf",
                "file_type": "pdf",
                "content_hash": "hash-doc-2",
                "uploaded_at": now,
                "title": "Other",
                "language": "en",
            },
        ]
    )

    _insert_parses(
        [
            {
                "collection": collection,
                "doc_id": doc_id,
                "parse_hash": "parse-1",
                "parser": "deepdoc",
                "created_at": now,
                "params_json": "{}",
                "parsed_content": "content",
            }
        ]
    )

    _insert_embeddings(
        "text-embedding-v3",
        [
            {
                "collection": collection,
                "doc_id": doc_id,
                "chunk_id": "chunk-1",
                "parse_hash": "parse-1",
                "model": "text-embedding-v3",
                "vector": [0.1, 0.2, 0.3],
                "vector_dimension": 3,
                "text": "chunk text 1",
                "chunk_hash": "hash-chunk-1",
                "created_at": now,
            }
        ],
    )

    result = await list_collections(user_id=None, is_admin=True)

    assert result.status == "success"
    assert result.total_count == 1
    collection_map = {info.name: info for info in result.collections}
    assert collection in collection_map
    collection_info = collection_map[collection]
    assert collection_info.documents == 2
    assert collection_info.processed_documents == 1
    assert collection_info.embeddings == 1
    # document_names now contains source_path values
    assert sorted(collection_info.document_names) == sorted(["other.pdf", "sample.pdf"])
    assert result.warnings == []


@pytest.mark.asyncio
async def test_list_collections_admin_includes_config_from_other_user(
    temp_lancedb_dir: str,
) -> None:
    """Admin listing should attach ingestion_config stored under a tenant user_id."""

    import json

    from src.xagent.core.tools.core.RAG_tools.storage.factory import (
        get_metadata_store,
    )

    collection = "cfg_tenant_collection"
    doc_id = "doc-cfg"
    now = datetime.now(timezone.utc)

    _insert_documents(
        [
            {
                "collection": collection,
                "doc_id": doc_id,
                "source_path": "/path/x.pdf",
                "file_type": "pdf",
                "content_hash": "h1",
                "uploaded_at": now,
                "title": "T",
                "language": "zh",
            }
        ]
    )

    await get_metadata_store().save_collection_config(
        collection,
        json.dumps({}),
        user_id=99,
    )

    result = await list_collections(user_id=None, is_admin=True)

    assert result.status == "success"
    assert result.total_count == 1
    info = next(c for c in result.collections if c.name == collection)
    assert info.ingestion_config is not None


@pytest.mark.asyncio
async def test_list_collections_includes_empty_metadata_only_collection(
    temp_lancedb_dir: str,
) -> None:
    """Persisted collection metadata should keep empty collections visible."""

    from src.xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo

    await get_metadata_store().save_collection(CollectionInfo(name="empty_collection"))

    result = await list_collections(user_id=None, is_admin=True)

    assert result.status == "success"
    collection_map = {info.name: info for info in result.collections}
    assert "empty_collection" in collection_map
    collection_info = collection_map["empty_collection"]
    assert collection_info.documents == 0
    assert collection_info.document_names == []


@pytest.mark.asyncio
async def test_list_collections_non_admin_only_sees_owned_metadata_only_collections(
    temp_lancedb_dir: str,
) -> None:
    """Non-admin listing should not expose other users' metadata-only collections."""

    from src.xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo

    store = get_metadata_store()
    await store.save_collection(CollectionInfo(name="mine_only"))
    await store.save_collection(CollectionInfo(name="theirs_only"))
    await store.save_collection_config("mine_only", "{}", user_id=1)
    await store.save_collection_config("theirs_only", "{}", user_id=2)

    result = await list_collections(user_id=1, is_admin=False)

    assert result.status == "success"
    collection_names = {info.name for info in result.collections}
    assert "mine_only" in collection_names
    assert "theirs_only" not in collection_names


@pytest.mark.asyncio
async def test_delete_collection_removes_empty_collection_metadata(
    temp_lancedb_dir: str,
) -> None:
    """Deleting a collection should remove metadata-only empty collections too."""

    from src.xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo

    store = get_metadata_store()
    await store.save_collection(CollectionInfo(name="empty_collection"))
    await store.save_collection_config("empty_collection", "{}", user_id=1)

    result = delete_collection("empty_collection", user_id=1, is_admin=False)

    assert result.status == "success"

    listed = await list_collections(user_id=None, is_admin=True)
    collection_names = {info.name for info in listed.collections}
    assert "empty_collection" not in collection_names


@pytest.mark.asyncio
async def test_delete_collection_preserves_other_users_shared_collection_data(
    temp_lancedb_dir: str,
) -> None:
    """Tenant-scoped deletes should not wipe another user's shared collection rows."""

    from src.xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo

    collection = "shared_collection"
    now = datetime.now(timezone.utc)
    store = get_metadata_store()
    await store.save_collection(CollectionInfo(name=collection))
    await store.save_collection_config(collection, "{}", user_id=1)
    await store.save_collection_config(collection, "{}", user_id=2)

    _insert_documents(
        [
            {
                "collection": collection,
                "doc_id": "doc-user-1",
                "source_path": "/path/user-1.pdf",
                "file_type": "pdf",
                "content_hash": "hash-user-1",
                "uploaded_at": now,
                "title": "User 1",
                "language": "zh",
                "user_id": 1,
            },
            {
                "collection": collection,
                "doc_id": "doc-user-2",
                "source_path": "/path/user-2.pdf",
                "file_type": "pdf",
                "content_hash": "hash-user-2",
                "uploaded_at": now,
                "title": "User 2",
                "language": "en",
                "user_id": 2,
            },
        ]
    )

    result = delete_collection(collection, user_id=1, is_admin=False)

    assert result.status == "success"

    user_one_documents = list_documents(
        collection=collection, user_id=1, is_admin=False
    )
    user_two_documents = list_documents(
        collection=collection, user_id=2, is_admin=False
    )
    assert user_one_documents.documents == []
    assert {doc.doc_id for doc in user_two_documents.documents} == {"doc-user-2"}

    listed_for_admin = await list_collections(user_id=None, is_admin=True)
    assert collection in {info.name for info in listed_for_admin.collections}


def test_get_document_stats_missing_document(temp_lancedb_dir: str) -> None:
    """Missing documents should yield zero counts but succeed."""

    result = get_document_stats("demo", "missing-doc")

    assert result.status == "success"
    assert result.data is not None
    assert result.data.document_exists is False
    assert result.data.chunk_count == 0
    assert result.data.embedding_count == 0
    assert result.data.embedding_breakdown == {}
    assert result.warnings == []


def test_get_document_stats_with_embeddings(temp_lancedb_dir: str) -> None:
    """Document statistics should aggregate parse, chunk, and embedding counts."""

    collection = "demo_collection"
    doc_id = "doc-embed"
    now = datetime.now(timezone.utc)

    _insert_documents(
        [
            {
                "collection": collection,
                "doc_id": doc_id,
                "source_path": "/doc/embed.pdf",
                "file_type": "pdf",
                "content_hash": "hash",
                "uploaded_at": now,
                "title": "Embed",
                "language": "zh",
            }
        ]
    )

    result = get_document_stats(collection, doc_id)

    assert result.status == "success"
    assert result.data is not None
    assert result.data.document_exists is True
    assert result.warnings == []


def test_retry_and_cancel_document_update_status(temp_lancedb_dir: str) -> None:
    """retry_document and cancel_document should record status updates."""

    retry_result = retry_document("demo", "doc-9", user_id=1, is_admin=True)
    assert retry_result.status == "success"
    assert retry_result.new_status == DocumentProcessingStatus.PENDING

    cancel_result = cancel_document(
        "demo", "doc-9", user_id=1, is_admin=True, reason="User cancelled"
    )
    assert cancel_result.status == "success"
    assert cancel_result.new_status == DocumentProcessingStatus.FAILED

    status_entries = load_ingestion_status(
        collection="demo", doc_id="doc-9", user_id=1, is_admin=True
    )
    assert status_entries[-1]["status"] == DocumentProcessingStatus.FAILED.value
    assert status_entries[-1]["message"] == "User cancelled"


def test_cancel_collection_updates_all_documents(temp_lancedb_dir: str) -> None:
    """Collection-level cancel should update status for all discoverable documents."""

    collection = "demo"
    now = datetime.now(timezone.utc)

    _insert_documents(
        [
            {
                "collection": collection,
                "doc_id": "doc-1",
                "source_path": "/path/doc-1.pdf",
                "file_type": "pdf",
                "content_hash": "hash-1",
                "uploaded_at": now,
                "title": "First",
                "language": "zh",
            }
        ]
    )

    reason = "Manual stop"
    result = cancel_collection(collection, reason=reason, user_id=1, is_admin=True)

    assert result.status == "success"
    affected_ids = {detail.doc_id for detail in result.affected_documents}
    assert "doc-1" in affected_ids


def test_delete_collection_invokes_cleanup_all_documents(
    monkeypatch: pytest.MonkeyPatch, temp_lancedb_dir: str
) -> None:
    """Collection delete should cascade cleanup for each document variant."""

    collection = "demo"
    now = datetime.now(timezone.utc)

    _insert_documents(
        [
            {
                "collection": collection,
                "doc_id": "doc-1",
                "source_path": "/path/doc-1.pdf",
                "file_type": "pdf",
                "content_hash": "hash-1",
                "uploaded_at": now,
                "title": "First",
                "language": "zh",
            }
        ]
    )

    cleared_calls: List[tuple[str, str]] = []

    def _fake_clear(collection: str, doc_id: str, **_: object) -> None:
        cleared_calls.append((collection, doc_id))

    monkeypatch.setattr(
        collections_module,
        "_clear_ingestion_status_impl",
        _fake_clear,
    )

    result = delete_collection(collection, user_id=1, is_admin=True)

    assert result.status == "success"
    assert "documents" in result.deleted_counts


def test_delete_collection_preserves_partial_vector_cleanup(
    monkeypatch: pytest.MonkeyPatch, temp_lancedb_dir: str
) -> None:
    """Best-effort vector cleanup warnings (no exception) yield partial_success.

    The admin data-plane delete reports a per-table failure by appending to
    ``warnings_out`` rather than raising.  Routed through the coordinator/handle,
    such a warning alongside an actual deletion must surface as
    ``partial_success`` so callers are not told the delete fully succeeded.
    """
    warnings_from_store = "Failed to delete from 'parses': parse delete failed"

    store = get_vector_index_store()

    def _delete_collection_data(**kwargs: object) -> dict[str, int]:
        kwargs["warnings_out"].append(warnings_from_store)  # type: ignore[union-attr]
        return {"documents": 1}

    monkeypatch.setattr(store, "delete_collection_data", _delete_collection_data)

    result = delete_collection("demo", user_id=1, is_admin=True)

    assert result.status == "partial_success"
    assert result.deleted_counts == {"documents": 1}
    assert result.warnings == [warnings_from_store]


def test_delete_collection_non_admin_uses_batched_document_scoped_delete(
    monkeypatch: pytest.MonkeyPatch, temp_lancedb_dir: str
) -> None:
    """Non-admin collection delete batches the tenant's docs, never collection-wide."""

    now = datetime.now(timezone.utc)
    _insert_documents(
        [
            {
                "collection": "shared",
                "doc_id": "doc-1",
                "source_path": "/path/doc-1.pdf",
                "file_type": "pdf",
                "content_hash": "hash-1",
                "uploaded_at": now,
                "title": "First",
                "language": "zh",
                "user_id": 7,
            },
            {
                "collection": "shared",
                "doc_id": "doc-2",
                "source_path": "/path/doc-2.pdf",
                "file_type": "pdf",
                "content_hash": "hash-2",
                "uploaded_at": now,
                "title": "Second",
                "language": "zh",
                "user_id": 7,
            },
        ]
    )

    store = get_vector_index_store()
    batched_calls: list[dict[str, object]] = []
    original_delete_documents_data = store.delete_documents_data

    def _spy_delete_documents_data(*args: object, **kwargs: object) -> dict[str, int]:
        batched_calls.append(dict(kwargs))
        return original_delete_documents_data(*args, **kwargs)

    def _no_collection_wide(**_kwargs: object) -> dict[str, int]:
        raise AssertionError("non-admin delete must not use collection-wide cleanup")

    monkeypatch.setattr(store, "delete_documents_data", _spy_delete_documents_data)
    monkeypatch.setattr(store, "delete_collection_data", _no_collection_wide)

    result = delete_collection("shared", user_id=7, is_admin=False)

    assert result.status == "success"
    # A single batched call carrying both of the tenant's doc_ids, tenant-scoped.
    assert len(batched_calls) == 1
    assert batched_calls[0]["collection_name"] == "shared"
    assert batched_calls[0]["doc_ids"] == ["doc-1", "doc-2"]
    assert batched_calls[0]["user_id"] == 7
    assert batched_calls[0]["is_admin"] is False
    # End-to-end: the tenant's documents are gone afterwards.
    remaining = list_documents(collection="shared", user_id=7, is_admin=False)
    assert remaining.documents == []


def test_delete_collection_reports_partial_batched_delete_failure(
    monkeypatch: pytest.MonkeyPatch, temp_lancedb_dir: str
) -> None:
    """A failed batch surfaces prior progress as partial_success with per-doc status."""

    now = datetime.now(timezone.utc)
    _insert_documents(
        [
            {
                "collection": "shared",
                "doc_id": "doc-1",
                "source_path": "/path/doc-1.pdf",
                "file_type": "pdf",
                "content_hash": "hash-1",
                "uploaded_at": now,
                "title": "First",
                "language": "zh",
                "user_id": 7,
            },
            {
                "collection": "shared",
                "doc_id": "doc-2",
                "source_path": "/path/doc-2.pdf",
                "file_type": "pdf",
                "content_hash": "hash-2",
                "uploaded_at": now,
                "title": "Second",
                "language": "zh",
                "user_id": 7,
            },
        ]
    )

    store = get_vector_index_store()

    def _delete_documents_data(**kwargs: object) -> dict[str, int]:
        kwargs["warnings_out"].append(  # type: ignore[union-attr]
            "Failed to delete document batch 2: boom"
        )
        raise DatabaseOperationError(
            "Failed to delete document batch",
            details={
                "deleted_counts": {"documents": 1, "chunks": 2},
                "deleted_doc_ids": ["doc-1"],
            },
        )

    monkeypatch.setattr(store, "delete_documents_data", _delete_documents_data)

    result = delete_collection("shared", user_id=7, is_admin=False)

    assert result.status == "partial_success"
    assert result.deleted_counts == {"documents": 1, "chunks": 2}
    assert result.warnings == ["Failed to delete document batch 2: boom"]
    # The successfully deleted doc is SUCCESS; the rest are reported FAILED.
    status_by_doc = {d.doc_id: d.status for d in result.affected_documents}
    assert status_by_doc["doc-1"] == DocumentProcessingStatus.SUCCESS
    assert status_by_doc["doc-2"] == DocumentProcessingStatus.FAILED


def test_delete_collection_reports_success_when_only_orphan_artifacts_deleted(
    monkeypatch: pytest.MonkeyPatch, temp_lancedb_dir: str
) -> None:
    """Deleting orphan vector artifacts without documents is still success."""

    store = get_vector_index_store()

    def _delete_collection_data(**_kwargs: object) -> dict[str, int]:
        return {"documents": 0, "chunks": 2, "embeddings_m1": 3}

    monkeypatch.setattr(store, "delete_collection_data", _delete_collection_data)

    result = delete_collection("orphaned", user_id=None, is_admin=True)

    assert result.status == "success"
    # Zero-count tables are dropped from the summary (no "documents": 0 clutter).
    assert result.deleted_counts == {"chunks": 2, "embeddings_m1": 3}


def test_e2e_register_and_list_documents_with_legacy_empty_string_file_id(
    tmp_path: Path, temp_lancedb_dir: str
) -> None:
    """E2E: ingestion remains visible when legacy rows contain empty string file_id."""
    conn = get_connection_from_env()
    ensure_documents_table(conn)
    table = conn.open_table("documents")

    # Simulate legacy row created by previous PR's backfill (NULL -> "")
    table.add(
        [
            {
                "collection": "xagent",
                "doc_id": "legacy-doc",
                "file_id": "",  # Empty string from previous backfill
                "source_path": "/legacy/README.md",
                "file_type": "md",
                "content_hash": "legacy-hash",
                "uploaded_at": datetime.now(timezone.utc),
                "title": "legacy",
                "language": "en",
                "user_id": None,
            }
        ]
    )

    # Trigger schema ensure path again (startup/runtime behavior) to backfill.
    ensure_documents_table(conn)

    new_file = tmp_path / "README.md"
    new_file.write_text("# hello\n\nworld", encoding="utf-8")
    reg_result = register_document(
        collection="xagent",
        source_path=str(new_file),
        file_id=None,
        user_id=58,
    )
    assert reg_result["doc_id"]

    list_result = list_documents(collection="xagent", user_id=58, is_admin=False)
    assert list_result.status == "success"
    listed_ids = {doc.doc_id for doc in list_result.documents}
    assert reg_result["doc_id"] in listed_ids


# --- list_collections force_realtime Tests ---


@pytest.mark.asyncio
async def test_list_collections_force_realtime_bypasses_cache(
    temp_lancedb_dir: str,
) -> None:
    """force_realtime=True should skip metadata cache and use realtime aggregation."""
    now = datetime.now(timezone.utc)
    collection = "realtime_test"

    _insert_documents(
        [
            {
                "collection": collection,
                "doc_id": "doc-1",
                "source_path": "/path/doc.pdf",
                "file_type": "pdf",
                "content_hash": "hash-1",
                "uploaded_at": now,
                "title": "Doc",
                "language": "en",
            }
        ]
    )

    result = await list_collections(user_id=None, is_admin=True, force_realtime=True)

    assert result.status == "success"
    assert result.total_count == 1
    assert result.collections[0].name == collection
    assert result.collections[0].documents == 1


@pytest.mark.asyncio
async def test_list_collections_cache_filled_by_subsequent_call(
    temp_lancedb_dir: str,
) -> None:
    """After a force_realtime call fills the cache, subsequent call should use it."""
    now = datetime.now(timezone.utc)
    collection = "cache_fill_test"

    _insert_documents(
        [
            {
                "collection": collection,
                "doc_id": "doc-1",
                "source_path": "/path/doc.pdf",
                "file_type": "pdf",
                "content_hash": "hash-1",
                "uploaded_at": now,
                "title": "Doc",
                "language": "en",
            }
        ]
    )

    # First call with force_realtime fills the metadata cache
    result1 = await list_collections(user_id=None, is_admin=True, force_realtime=True)
    assert result1.status == "success"
    assert result1.total_count == 1

    # Second normal call should hit the cache
    result2 = await list_collections(user_id=None, is_admin=True)
    assert result2.status == "success"
    assert result2.total_count == 1
    assert result2.collections[0].name == collection
    assert result2.collections[0].documents == 1


@pytest.mark.asyncio
async def test_list_collections_preserves_cached_collection_timestamps(
    temp_lancedb_dir: str,
) -> None:
    """Cached collection timestamps should remain stable across list calls."""
    collection = "timestamp_cache_test"
    now = datetime.now(timezone.utc)

    _insert_documents(
        [
            {
                "collection": collection,
                "doc_id": "doc-1",
                "source_path": "/path/doc.pdf",
                "file_type": "pdf",
                "content_hash": "hash-1",
                "uploaded_at": now,
                "title": "Doc",
                "language": "en",
            }
        ]
    )

    first_result = await list_collections(
        user_id=None, is_admin=True, force_realtime=True
    )
    assert first_result.status == "success"
    first_info = next(c for c in first_result.collections if c.name == collection)

    second_result = await list_collections(user_id=None, is_admin=True)
    assert second_result.status == "success"
    second_info = next(c for c in second_result.collections if c.name == collection)

    assert second_info.created_at == first_info.created_at
    assert second_info.updated_at == first_info.updated_at
    assert second_info.last_accessed_at == first_info.last_accessed_at


@pytest.mark.asyncio
async def test_list_collections_cache_miss_uses_realtime(
    temp_lancedb_dir: str,
) -> None:
    """When metadata cache misses (no cached data), list_collections falls back to realtime aggregation."""
    collection = "miss_test"

    _insert_documents(
        [
            {
                "collection": collection,
                "doc_id": "doc-1",
                "source_path": "/path/doc.pdf",
                "file_type": "pdf",
                "content_hash": "hash-1",
                "uploaded_at": datetime.now(timezone.utc),
                "title": "Doc",
                "language": "en",
            }
        ]
    )

    # No prior cache population — should fallback to realtime successfully
    result = await list_collections(user_id=None, is_admin=True)
    assert result.status == "success"
    assert result.total_count >= 1

    info = next(c for c in result.collections if c.name == collection)
    assert info.created_at is not None
    assert info.updated_at is not None
    assert info.last_accessed_at is not None


@pytest.mark.asyncio
async def test_list_collections_non_admin_realtime_does_not_overwrite_global_metadata(
    temp_lancedb_dir: str,
) -> None:
    """Tenant-scoped realtime refresh should not write filtered stats to global metadata."""
    collection = "tenant_realtime_test"
    now = datetime.now(timezone.utc)

    _insert_documents(
        [
            {
                "collection": collection,
                "doc_id": "doc-1",
                "source_path": "/path/doc.pdf",
                "file_type": "pdf",
                "content_hash": "hash-1",
                "uploaded_at": now,
                "title": "Doc",
                "language": "en",
                "user_id": 1,
            }
        ]
    )

    store = get_metadata_store()
    original_save_collection = store.save_collection
    save_collection_mock = AsyncMock()
    store.save_collection = save_collection_mock
    try:
        result = await list_collections(user_id=1, is_admin=False, force_realtime=True)
    finally:
        store.save_collection = original_save_collection

    assert result.status == "success"
    assert any(info.name == collection for info in result.collections)
    save_collection_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_list_collections_non_admin_uses_tenant_stats_despite_metadata_cache(
    temp_lancedb_dir: str,
) -> None:
    """A visible shared-name collection should show only the caller's document count."""
    collection = "shared_name_cache_test"
    now = datetime.now(timezone.utc)

    _insert_documents(
        [
            {
                "collection": collection,
                "doc_id": "doc-user-1",
                "source_path": "/path/user1.pdf",
                "file_type": "pdf",
                "content_hash": "hash-1",
                "uploaded_at": now,
                "title": "User 1",
                "language": "en",
                "user_id": 1,
            },
            {
                "collection": collection,
                "doc_id": "doc-user-2",
                "source_path": "/path/user2.pdf",
                "file_type": "pdf",
                "content_hash": "hash-2",
                "uploaded_at": now,
                "title": "User 2",
                "language": "en",
                "user_id": 2,
            },
        ]
    )
    await get_metadata_store().save_collection_config(collection, "{}", user_id=1)

    result = await list_collections(user_id=1, is_admin=False)

    info = next(c for c in result.collections if c.name == collection)
    assert info.documents == 1
    assert info.owners == [1]
    assert info.document_names == ["user1.pdf"]


@pytest.mark.asyncio
async def test_list_collections_non_admin_stale_config_does_not_use_global_stats(
    temp_lancedb_dir: str,
) -> None:
    """A stale user config should not inherit another user's cached stats."""
    collection = "stale_config_shared_name_test"
    now = datetime.now(timezone.utc)

    _insert_documents(
        [
            {
                "collection": collection,
                "doc_id": "doc-user-2",
                "source_path": "/path/user2.pdf",
                "file_type": "pdf",
                "content_hash": "hash-2",
                "uploaded_at": now,
                "title": "User 2",
                "language": "en",
                "user_id": 2,
            }
        ]
    )
    await get_metadata_store().save_collection_config(collection, "{}", user_id=1)

    result = await list_collections(user_id=1, is_admin=False)

    info = next(c for c in result.collections if c.name == collection)
    assert info.documents == 0
    assert info.parses == 0
    assert info.chunks == 0
    assert info.embeddings == 0
    assert info.processed_documents == 0
    assert info.owners == []
    assert info.document_names == []


# --- delete_collection metadata cleanup Tests ---


@pytest.mark.asyncio
async def test_delete_collection_clears_metadata_cache(temp_lancedb_dir: str) -> None:
    """After deleting a collection, metadata cache should not return it."""
    now = datetime.now(timezone.utc)
    collection = "to_delete_test"

    _insert_documents(
        [
            {
                "collection": collection,
                "doc_id": "doc-1",
                "source_path": "/path/doc.pdf",
                "file_type": "pdf",
                "content_hash": "hash-1",
                "uploaded_at": now,
                "title": "Doc",
                "language": "en",
            }
        ]
    )

    # Populate metadata cache first
    await list_collections(user_id=None, is_admin=True, force_realtime=True)

    # Delete the collection
    del_result = delete_collection(collection, user_id=None, is_admin=True)
    assert del_result.status == "success"

    # Metadata cache should no longer include the deleted collection
    result = await list_collections(user_id=None, is_admin=True)
    remaining = [c.name for c in result.collections]
    assert collection not in remaining


def test_delete_document_authorizes_before_cascade() -> None:
    vector_store = MagicMock()
    vector_store.count_rows.return_value = 0
    vector_store.iter_batches.return_value = []

    with (
        patch.object(
            collections_module, "get_vector_index_store", return_value=vector_store
        ),
        patch.object(vector_store, "delete_document_data") as mock_delete_data,
        patch.object(collections_module, "_clear_ingestion_status_impl") as mock_clear,
    ):
        result = delete_document("demo", "doc-1", user_id=7, is_admin=False)

    assert result.status == "error"
    assert result.message == "Document not found or not accessible."
    vector_store.count_rows.assert_called_once_with(
        table_name="documents",
        filters={"collection": "demo", "doc_id": "doc-1"},
        user_id=7,
        is_admin=False,
    )
    mock_delete_data.assert_not_called()
    mock_clear.assert_not_called()


def test_delete_document_allows_legacy_owner_recovered_from_source_path() -> None:
    vector_store = MagicMock()
    vector_store.count_rows.return_value = 0

    legacy_batch = MagicMock()
    legacy_batch.num_rows = 1
    legacy_batch.to_pylist.return_value = [
        {
            "collection": "demo",
            "doc_id": "doc-legacy",
            "user_id": None,
            "source_path": "/uploads/user_7/demo/legacy.csv",
        }
    ]
    vector_store.iter_batches.return_value = [legacy_batch]

    with (
        patch.object(
            collections_module, "get_vector_index_store", return_value=vector_store
        ),
        patch.object(
            vector_store,
            "delete_document_data",
            return_value={"documents": 1, "main_pointers": 1},
        ) as mock_delete_data,
        patch.object(collections_module, "_clear_ingestion_status_impl") as mock_clear,
    ):
        result = delete_document("demo", "doc-legacy", user_id=7, is_admin=False)

    assert result.status == "success"
    mock_delete_data.assert_called_once_with(
        collection_name="demo",
        doc_id="doc-legacy",
        user_id=7,
        is_admin=False,
    )
    mock_clear.assert_called_once_with(
        "demo",
        "doc-legacy",
        user_id=None,
        is_admin=True,
    )
    vector_store.iter_batches.assert_called_once_with(
        table_name="documents",
        columns=["collection", "doc_id", "user_id", "source_path"],
        batch_size=1,
        filters={"collection": "demo", "doc_id": "doc-legacy"},
        user_id=None,
        is_admin=True,
    )


def test_delete_document_rejects_legacy_row_owned_by_another_user() -> None:
    vector_store = MagicMock()
    vector_store.count_rows.return_value = 0

    foreign_batch = MagicMock()
    foreign_batch.num_rows = 1
    foreign_batch.to_pylist.return_value = [
        {
            "collection": "demo",
            "doc_id": "doc-foreign",
            "user_id": None,
            "source_path": "/uploads/user_99/demo/foreign.csv",
        }
    ]
    vector_store.iter_batches.return_value = [foreign_batch]

    with (
        patch.object(
            collections_module, "get_vector_index_store", return_value=vector_store
        ),
        patch.object(vector_store, "delete_document_data") as mock_delete_data,
        patch.object(collections_module, "_clear_ingestion_status_impl") as mock_clear,
    ):
        result = delete_document("demo", "doc-foreign", user_id=7, is_admin=False)

    assert result.status == "error"
    assert result.message == "Document not found or not accessible."
    mock_delete_data.assert_not_called()
    mock_clear.assert_not_called()


def test_delete_document_clears_status_with_caller_scope() -> None:
    vector_store = MagicMock()
    vector_store.count_rows.return_value = 1

    with (
        patch.object(
            collections_module, "get_vector_index_store", return_value=vector_store
        ),
        patch.object(
            vector_store,
            "delete_document_data",
            return_value={"documents": 1, "main_pointers": 1},
        ) as mock_delete_data,
        patch.object(collections_module, "_clear_ingestion_status_impl") as mock_clear,
    ):
        result = delete_document("demo", "doc-1", user_id=9, is_admin=True)

    assert result.status == "success"
    assert result.details == {"documents": 1, "main_pointers": 1}
    mock_delete_data.assert_called_once_with(
        collection_name="demo",
        doc_id="doc-1",
        user_id=9,
        is_admin=True,
    )
    mock_clear.assert_called_once_with(
        "demo",
        "doc-1",
        user_id=9,
        is_admin=True,
    )


def test_delete_document_cleans_failed_ingest_status_by_doc_id(
    temp_lancedb_dir: str,
) -> None:
    """Failed ingest rollback cleanup should remove document data, vectors, and status."""

    collection = "failed_ingest_cleanup"
    doc_id = "failed-doc"
    model_name = "text-embedding-v3"
    now = datetime.now(timezone.utc)

    _insert_documents(
        [
            {
                "collection": collection,
                "doc_id": doc_id,
                "source_path": "/uploads/user_7/failed-doc.pdf",
                "file_type": "pdf",
                "content_hash": "failed-hash",
                "uploaded_at": now,
                "title": "Failed",
                "language": "zh",
                "user_id": 7,
            }
        ]
    )
    _insert_embeddings(
        model_name,
        [
            {
                "collection": collection,
                "doc_id": doc_id,
                "chunk_id": "chunk-1",
                "parse_hash": "parse-failed",
                "model": model_name,
                "vector": [0.1, 0.2, 0.3],
                "vector_dimension": 3,
                "text": "partial vector written before ingest failed",
                "chunk_hash": "failed-chunk-hash",
                "created_at": now,
                "metadata": "{}",
                "user_id": 7,
            }
        ],
    )
    write_ingestion_status(
        collection,
        doc_id,
        status=DocumentProcessingStatus.FAILED.value,
        message="Failed during ingest",
        user_id=7,
    )

    conn = get_vector_index_store().get_raw_connection()
    embedding_table = conn.open_table(embeddings_table_name(model_name))
    assert embedding_table.count_rows() == 1

    result = delete_document(collection, doc_id, user_id=7, is_admin=False)

    assert result.status == "success"
    assert result.details.get("documents") == 1
    assert result.details.get(embeddings_table_name(model_name)) == 1
    embedding_table = conn.open_table(embeddings_table_name(model_name))
    assert embedding_table.count_rows() == 0
    assert (
        load_ingestion_status(
            collection=collection,
            doc_id=doc_id,
            user_id=7,
            is_admin=False,
        )
        == []
    )


def test_delete_collection_removes_metadata_table_entry(temp_lancedb_dir: str) -> None:
    """Management delete should remove collection_metadata row directly."""
    now = datetime.now(timezone.utc)
    collection = "to_delete_metadata_row"

    _insert_documents(
        [
            {
                "collection": collection,
                "doc_id": "doc-1",
                "source_path": "/path/doc.pdf",
                "file_type": "pdf",
                "content_hash": "hash-1",
                "uploaded_at": now,
                "title": "Doc",
                "language": "en",
            }
        ]
    )

    # Prime metadata cache/table with current collection stats.
    import asyncio

    asyncio.run(list_collections(user_id=None, is_admin=True, force_realtime=True))

    conn = get_connection_from_env()
    before_table = conn.open_table("collection_metadata")
    before = before_table.search().where(f"name = '{collection}'").to_list()
    assert len(before) == 1

    del_result = delete_collection(collection, user_id=None, is_admin=True)
    assert del_result.status == "success"

    after_table = conn.open_table("collection_metadata")
    after = after_table.search().where(f"name = '{collection}'").to_list()
    assert after == []
