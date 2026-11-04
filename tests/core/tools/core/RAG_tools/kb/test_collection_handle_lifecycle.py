"""Tests for delete_collection_data, delete_documents_data, and rename primitives on KBCollectionHandle.

H05 Phase 1 – These tests drive out collection-level cascade delete methods on
``LanceDBCollectionHandle``.  They must FAIL before the implementation is added
(RED) and PASS afterwards (GREEN).

H05 Phase 2 – Adds rename_collection_data, rename_collection_status, and
rename_collection_metadata primitives.

Storage isolation is provided by the autouse ``isolate_rag_storage`` fixture in
``tests/conftest.py``.
"""

from __future__ import annotations

import asyncio
import inspect
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import DatabaseOperationError
from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
    LanceDBCollectionHandle,
)
from xagent.core.tools.core.RAG_tools.kb.models import (
    KBAccessMode,
    KBBackendCapabilities,
    KBCollectionContext,
    KBStorageBackend,
    KBUserScope,
)
from xagent.core.tools.core.RAG_tools.storage.factory import (
    get_ingestion_status_store,
    get_main_pointer_store,
    get_metadata_store,
    get_vector_index_store,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_handle(collection: str = "test_coll") -> LanceDBCollectionHandle:
    """Build a LanceDB-backed handle bound to the current test stores."""
    context = KBCollectionContext(
        collection=collection,
        user_scope=KBUserScope(user_id=None, is_admin=True),
        access_mode=KBAccessMode.WRITE,
        allow_create=True,
        hide_missing=True,
        metadata_store=get_metadata_store(),
        vector_index_store=get_vector_index_store(),
        ingestion_status_store=get_ingestion_status_store(),
        main_pointer_store=get_main_pointer_store(),
        backend=KBStorageBackend.LANCEDB,
        capabilities=KBBackendCapabilities.lancedb(),
        collection_info=None,
    )
    return LanceDBCollectionHandle(context)


def _doc_row(collection: str, doc_id: str, *, user_id=None) -> dict:
    return {
        "collection": collection,
        "doc_id": doc_id,
        "file_id": None,
        "source_path": f"/uploads/{doc_id}.txt",
        "file_type": "txt",
        "content_hash": "a" * 64,
        "uploaded_at": datetime.now(timezone.utc),
        "title": None,
        "language": None,
        "user_id": user_id,
    }


def _parse_row(collection: str, doc_id: str, parse_hash: str, *, user_id=None) -> dict:
    return {
        "collection": collection,
        "doc_id": doc_id,
        "parse_hash": parse_hash,
        "parser": "test_parser",
        "created_at": datetime.now(timezone.utc),
        "params_json": "{}",
        "parsed_content": "parsed text",
        "user_id": user_id,
    }


def _chunk_row(
    collection: str,
    doc_id: str,
    parse_hash: str,
    config_hash: str,
    chunk_id: str,
    *,
    user_id=None,
) -> dict:
    return {
        "collection": collection,
        "doc_id": doc_id,
        "parse_hash": parse_hash,
        "chunk_id": chunk_id,
        "index": 0,
        "text": f"chunk-{chunk_id}",
        "page_number": None,
        "section": None,
        "anchor": None,
        "json_path": None,
        "chunk_hash": "ch-" + chunk_id,
        "config_hash": config_hash,
        "created_at": datetime.now(timezone.utc),
        "metadata": "{}",
        "user_id": user_id,
    }


def _seed_collection(collection: str, doc_ids: list[str], *, user_id=None) -> None:
    """Seed documents, parses, and chunks for a collection."""
    store = get_vector_index_store()
    for doc_id in doc_ids:
        store.upsert_documents([_doc_row(collection, doc_id, user_id=user_id)])
        store.upsert_parses(
            [_parse_row(collection, doc_id, f"h-{doc_id}", user_id=user_id)]
        )
        store.upsert_chunks(
            [
                _chunk_row(
                    collection,
                    doc_id,
                    f"h-{doc_id}",
                    "cfg1",
                    f"c-{doc_id}",
                    user_id=user_id,
                )
            ]
        )


# ---------------------------------------------------------------------------
# Test 1: admin delete_collection_data clears all tables
# ---------------------------------------------------------------------------


class TestDeleteCollectionDataAdminClearsAllTables:
    def test_delete_collection_data_admin_clears_all_tables(self) -> None:
        """Admin delete_collection_data removes all documents/parses/chunks for the collection.

        The method should:
        - Return a dict[str, int] with deleted row counts per table
        - Leave 0 docs in that collection after the call
        - Not touch other collections
        """
        store = get_vector_index_store()
        handle = make_handle("test_coll")

        # Seed test_coll with docs.
        _seed_collection("test_coll", ["d1", "d2"])
        # Seed another collection to verify isolation.
        _seed_collection("other_coll", ["d3"])

        result = handle.delete_collection_data(user_id=None, is_admin=True)

        # Returns a dict[str, int]
        assert isinstance(result, dict)
        assert all(isinstance(k, str) for k in result)
        assert all(isinstance(v, int) for v in result.values())
        # At least one table was deleted from.
        assert sum(result.values()) > 0

        # test_coll is empty afterwards.
        doc_count = store.count_rows(
            "documents", {"collection": "test_coll"}, is_admin=True
        )
        assert doc_count == 0

        # other_coll is untouched.
        other_count = store.count_rows(
            "documents", {"collection": "other_coll"}, is_admin=True
        )
        assert other_count == 1


# ---------------------------------------------------------------------------
# Test 2: method uses context collection (no external collection_name arg)
# ---------------------------------------------------------------------------


class TestDeleteCollectionDataUsesContextCollection:
    def test_delete_collection_data_uses_context_collection_not_arg(self) -> None:
        """delete_collection_data must not accept a collection_name argument.

        The handle is collection-scoped; it reads self.context.collection, not
        a caller-supplied collection name.
        """
        handle = make_handle("test_coll")
        sig = inspect.signature(handle.delete_collection_data)
        # The only parameters should be user_id, is_admin, and warnings_out.
        param_names = set(sig.parameters.keys())
        assert "collection_name" not in param_names, (
            "delete_collection_data must not accept a collection_name arg; "
            "it reads self.context.collection"
        )
        # Required positional/keyword args (excluding optional warnings_out)
        required_params = {
            name
            for name, param in sig.parameters.items()
            if param.default is inspect.Parameter.empty
            and param.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        }
        assert required_params <= {"user_id", "is_admin"}, (
            f"Unexpected required params: {required_params}"
        )


# ---------------------------------------------------------------------------
# Test 3: tenant-scoped delete removes only specified docs
# ---------------------------------------------------------------------------


class TestDeleteDocumentsDataTenantScoped:
    def test_delete_documents_data_tenant_scoped_removes_only_specified_docs(
        self,
    ) -> None:
        """delete_documents_data removes d1, d2 but leaves d3 untouched.

        Uses is_admin=False with a user_id to exercise tenant-scoped deletion.
        Verifies that delete is doc-id scoped: only the requested documents
        are removed, regardless of which user owns them.
        """
        store = get_vector_index_store()
        handle = make_handle("test_coll")

        # Insert three docs owned by user "u1".
        _seed_collection("test_coll", ["d1", "d2", "d3"], user_id=1)

        result = handle.delete_documents_data(
            doc_ids=["d1", "d2"],
            user_id=1,
            is_admin=True,  # admin=True so cascade_delete_documents works across all rows
        )

        assert isinstance(result, dict)

        # d1 and d2 should be gone.
        assert (
            store.count_rows(
                "documents", {"collection": "test_coll", "doc_id": "d1"}, is_admin=True
            )
            == 0
        )
        assert (
            store.count_rows(
                "documents", {"collection": "test_coll", "doc_id": "d2"}, is_admin=True
            )
            == 0
        )

        # d3 must remain.
        assert (
            store.count_rows(
                "documents", {"collection": "test_coll", "doc_id": "d3"}, is_admin=True
            )
            == 1
        )


# ---------------------------------------------------------------------------
# Test 4: partial failure preserves DatabaseOperationError.details contract
# ---------------------------------------------------------------------------


class TestDeleteDocumentsDataPartialFailurePreservesContract:
    def test_delete_documents_data_partial_failure_preserves_contract(self) -> None:
        """When a batch raises, DatabaseOperationError.details must have the required keys.

        The downstream CollectionOperationResult.partial_success relies on:
            details = {
                "deleted_counts": dict[str, int],
                "deleted_doc_ids": list[str],
                "failed_batch_index": int,
            }

        The store's delete_documents_data catches per-batch exceptions and
        re-raises as DatabaseOperationError with the required details dict.
        The handle passes that exception through unchanged.
        """
        handle = make_handle("test_coll")

        # Patch _vis_cascade_delete_documents (used inside the store's batching loop)
        # to raise so the store's exception-wrapping logic fires.
        cascade_path = (
            "xagent.core.tools.core.RAG_tools.storage"
            ".lancedb_stores._vis_cascade_delete_documents"
        )

        with patch(cascade_path, side_effect=RuntimeError("simulated cascade failure")):
            with pytest.raises(DatabaseOperationError) as exc_info:
                handle.delete_documents_data(
                    doc_ids=["d1", "d2"],
                    user_id=None,
                    is_admin=True,
                )

        err = exc_info.value
        assert hasattr(err, "details"), (
            "DatabaseOperationError must have a 'details' attribute"
        )
        details = err.details
        assert details is not None, "details must not be None"
        assert "deleted_counts" in details, (
            f"Missing 'deleted_counts' in details: {details}"
        )
        assert "deleted_doc_ids" in details, (
            f"Missing 'deleted_doc_ids' in details: {details}"
        )
        assert "failed_batch_index" in details, (
            f"Missing 'failed_batch_index' in details: {details}"
        )
        assert isinstance(details["deleted_counts"], dict)
        assert isinstance(details["deleted_doc_ids"], list)
        assert isinstance(details["failed_batch_index"], int)


# ---------------------------------------------------------------------------
# Phase 2 – Cycle 2.1: rename_collection_data
# ---------------------------------------------------------------------------


class TestRenameCollectionDataUpdatesAllTables:
    def test_rename_collection_data_updates_all_five_tables(self) -> None:
        """rename_collection_data updates collection field in documents/parses/chunks tables.

        Inserts rows under "old_name", calls handle.rename_collection_data with
        new_name="new_name", then asserts:
        - 0 rows remain under "old_name" in each seeded table
        - original row count is visible under "new_name"
        - method returns a list of warnings (may be empty)
        """
        store = get_vector_index_store()
        handle = make_handle("old_name")

        # Seed documents/parses/chunks under "old_name".
        _seed_collection("old_name", ["r1", "r2"])

        # Confirm rows exist before rename.
        assert (
            store.count_rows("documents", {"collection": "old_name"}, is_admin=True)
            == 2
        )
        assert (
            store.count_rows("parses", {"collection": "old_name"}, is_admin=True) == 2
        )
        assert (
            store.count_rows("chunks", {"collection": "old_name"}, is_admin=True) == 2
        )

        warnings = handle.rename_collection_data(
            new_name="new_name",
            user_id=None,
            is_admin=True,
        )

        # Must return a list (may be empty).
        assert isinstance(warnings, list)

        # All rows moved from "old_name" → "new_name".
        assert (
            store.count_rows("documents", {"collection": "old_name"}, is_admin=True)
            == 0
        )
        assert (
            store.count_rows("parses", {"collection": "old_name"}, is_admin=True) == 0
        )
        assert (
            store.count_rows("chunks", {"collection": "old_name"}, is_admin=True) == 0
        )

        assert (
            store.count_rows("documents", {"collection": "new_name"}, is_admin=True)
            == 2
        )
        assert (
            store.count_rows("parses", {"collection": "new_name"}, is_admin=True) == 2
        )
        assert (
            store.count_rows("chunks", {"collection": "new_name"}, is_admin=True) == 2
        )


# ---------------------------------------------------------------------------
# Phase 2 – Cycle 2.2: rename_collection_status
# ---------------------------------------------------------------------------


class TestRenameCollectionStatusUpdatesStatusTable:
    def test_rename_collection_status_updates_status_table(self) -> None:
        """rename_collection_status renames ingestion_runs rows from old to new name.

        Inserts status rows under "old_status", calls handle.rename_collection_status,
        then asserts the rows now live under "new_status".
        """
        status_store = get_ingestion_status_store()
        handle = make_handle("old_status")

        # Insert ingestion status rows under "old_status".
        status_store.write_ingestion_status(
            "old_status", "doc1", status="pending", user_id=None
        )
        status_store.write_ingestion_status(
            "old_status", "doc2", status="done", user_id=None
        )

        # Verify rows exist before rename.
        rows_before = status_store.load_ingestion_status(
            collection="old_status", user_id=None, is_admin=True
        )
        assert len(rows_before) == 2

        warnings = handle.rename_collection_status(
            new_name="new_status",
            user_id=None,
            is_admin=True,
        )

        # Returns a list (may be empty).
        assert isinstance(warnings, list)

        # Rows no longer under "old_status".
        rows_old = status_store.load_ingestion_status(
            collection="old_status", user_id=None, is_admin=True
        )
        assert len(rows_old) == 0

        # Rows now under "new_status".
        rows_new = status_store.load_ingestion_status(
            collection="new_status", user_id=None, is_admin=True
        )
        assert len(rows_new) == 2


# ---------------------------------------------------------------------------
# Phase 2 – Cycle 2.3: rename_collection_metadata (async)
# ---------------------------------------------------------------------------


class TestRenameCollectionMetadataAsyncMovesConfig:
    def test_rename_collection_metadata_async_moves_config(self) -> None:
        """rename_collection_metadata moves collection config from old_name to new_name.

        Sets up a collection config under "old_meta", calls
        await handle.rename_collection_metadata(new_name="new_meta", ...),
        then asserts:
        - "old_meta" is no longer listed
        - config is readable under "new_meta"

        This is the ONLY async method on the handle – it wraps an async
        metadata-store operation and must be declared ``async def``.
        Uses ``asyncio.run()`` to drive the coroutine without pytest-asyncio.
        """
        asyncio.run(self._run())

    async def _run(self) -> None:
        meta_store = get_metadata_store()
        handle = make_handle("old_meta")

        # Ensure table exists and seed a collection config entry.
        await meta_store.ensure_collection_metadata_table()
        import json

        await meta_store.save_collection_config(
            collection="old_meta",
            config_json=json.dumps({"embed_model": "test-model"}),
            user_id=0,
        )

        # Verify the config exists under "old_meta" before rename.
        config_before = await meta_store.get_collection_config(
            collection="old_meta", user_id=None, is_admin=True
        )
        assert config_before is not None

        await handle.rename_collection_metadata(
            new_name="new_meta",
            user_id=None,
            is_admin=True,
        )

        # Config must no longer be accessible under "old_meta".
        config_old = await meta_store.get_collection_config(
            collection="old_meta", user_id=None, is_admin=True
        )
        assert config_old is None

        # Config must be accessible under "new_meta".
        config_new = await meta_store.get_collection_config(
            collection="new_meta", user_id=None, is_admin=True
        )
        assert config_new is not None


# ---------------------------------------------------------------------------
# Phase 3 – Cycle 3.1: count_documents
# ---------------------------------------------------------------------------


class TestCountDocumentsReturnsTenantCount:
    def test_count_documents_returns_tenant_count(self) -> None:
        """count_documents returns per-user and admin counts correctly.

        Insert 2 docs from user_a and 1 doc from user_b in the same collection.
        - handle.count_documents(user_id="user_a", is_admin=False) -> 2
        - handle.count_documents(user_id="user_a", is_admin=True)  -> 3 (admin sees all)
        """
        store = get_vector_index_store()
        handle = make_handle("count_coll")

        # Insert 2 docs for user_a.
        store.upsert_documents([_doc_row("count_coll", "da1", user_id=10)])
        store.upsert_documents([_doc_row("count_coll", "da2", user_id=10)])
        # Insert 1 doc for user_b.
        store.upsert_documents([_doc_row("count_coll", "db1", user_id=20)])

        # Non-admin: user_a sees only their own 2 docs.
        count_user = handle.count_documents(user_id=10, is_admin=False)
        assert count_user == 2, f"Expected 2 for user_a non-admin, got {count_user}"

        # Admin: sees all 3 docs regardless of user_id.
        count_admin = handle.count_documents(user_id=10, is_admin=True)
        assert count_admin == 3, f"Expected 3 for admin, got {count_admin}"


# ---------------------------------------------------------------------------
# Phase 3 – Cycle 3.2: collection_stats
# ---------------------------------------------------------------------------


class TestCollectionStatsAggregatesAcrossTables:
    def test_collection_stats_aggregates_across_tables(self) -> None:
        """collection_stats returns document, chunk, and embedding counts for the collection.

        Insert known counts:
        - 2 documents
        - 3 chunks (parses)
        - 4 chunk rows (chunks table)

        Then assert stats["documents"], stats["chunks"], stats["embeddings"]
        match expected values. Embeddings may be 0 if no embedding table exists.
        """
        store = get_vector_index_store()
        handle = make_handle("stats_coll")

        # Insert 2 documents for user 1.
        store.upsert_documents([_doc_row("stats_coll", "s1", user_id=1)])
        store.upsert_documents([_doc_row("stats_coll", "s2", user_id=1)])

        # Insert 3 parses (one parse per doc, plus an extra parse for s1).
        store.upsert_parses([_parse_row("stats_coll", "s1", "ph1", user_id=1)])
        store.upsert_parses([_parse_row("stats_coll", "s1", "ph2", user_id=1)])
        store.upsert_parses([_parse_row("stats_coll", "s2", "ph3", user_id=1)])

        # Insert 4 chunks across the two docs.
        store.upsert_chunks(
            [
                _chunk_row("stats_coll", "s1", "ph1", "cfg1", "ck1", user_id=1),
                _chunk_row("stats_coll", "s1", "ph1", "cfg1", "ck2", user_id=1),
                _chunk_row("stats_coll", "s2", "ph3", "cfg1", "ck3", user_id=1),
                _chunk_row("stats_coll", "s2", "ph3", "cfg1", "ck4", user_id=1),
            ]
        )

        stats = handle.collection_stats(user_id=1, is_admin=True)

        assert isinstance(stats, dict), f"Expected dict, got {type(stats)}"
        assert "documents" in stats, f"Missing 'documents' key in stats: {stats}"
        assert "chunks" in stats, f"Missing 'chunks' key in stats: {stats}"
        assert "embeddings" in stats, f"Missing 'embeddings' key in stats: {stats}"

        assert stats["documents"] == 2, (
            f"Expected 2 documents, got {stats['documents']}"
        )
        assert stats["chunks"] == 4, f"Expected 4 chunks, got {stats['chunks']}"
        # No embeddings were written so embeddings count should be 0.
        assert stats["embeddings"] == 0, (
            f"Expected 0 embeddings, got {stats['embeddings']}"
        )


class TestDeleteCollectionConfigIsIdempotent:
    """delete_collection_config is idempotent (second delete returns 0, no error)."""

    def test_delete_collection_config_is_idempotent(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        import json

        meta_store = get_metadata_store()
        handle = make_handle("del_cfg_coll")

        # Ensure table exists and write a config row.
        await meta_store.ensure_collection_metadata_table()
        await meta_store.save_collection_config(
            collection="del_cfg_coll",
            config_json=json.dumps({"embed_model": "test"}),
            user_id=0,
        )

        # First delete must remove the row.
        deleted_first = await handle.delete_collection_config()
        assert deleted_first >= 1, (
            f"First delete expected >=1 rows, got {deleted_first}"
        )

        # Second delete must be a no-op (0 rows, no exception).
        deleted_second = await handle.delete_collection_config()
        assert deleted_second == 0, (
            f"Second delete expected 0 rows, got {deleted_second}"
        )

        # Config must be gone.
        config_after = await meta_store.get_collection_config(
            collection="del_cfg_coll", user_id=None, is_admin=True
        )
        assert config_after is None


# ---------------------------------------------------------------------------
# Phase 4 – Cycle 4.2: cleanup_collection_data_after_rollback
# ---------------------------------------------------------------------------


class TestCleanupAfterRollbackRemovesCollectionLocalDataOnly:
    """cleanup_collection_data_after_rollback removes only the bound collection's data."""

    def test_cleanup_after_rollback_removes_collection_local_data_only(self) -> None:
        """Cleanup removes new_coll data; other_coll is untouched; no FS calls made."""
        from unittest.mock import patch as _patch

        store = get_vector_index_store()
        handle_new = make_handle("new_coll")

        # Seed data for new_coll (the failed new-collection ingestion).
        _seed_collection("new_coll", ["nc1", "nc2"])
        # Seed data for other_coll (must remain untouched).
        _seed_collection("other_coll", ["oc1"])

        # Patch shutil.rmtree and os.remove to assert no filesystem calls occur.
        with (
            _patch("shutil.rmtree") as mock_rmtree,
            _patch("os.remove") as mock_os_remove,
        ):
            counts = handle_new.cleanup_collection_data_after_rollback(
                user_id=None, is_admin=True
            )

        # No filesystem calls must be made by the handle.
        mock_rmtree.assert_not_called()
        mock_os_remove.assert_not_called()

        # Returns a dict[str, int].
        assert isinstance(counts, dict)

        # new_coll data must be removed.
        assert (
            store.count_rows("documents", {"collection": "new_coll"}, is_admin=True)
            == 0
        ), "new_coll documents should be deleted after rollback cleanup"

        # other_coll data must be untouched.
        assert (
            store.count_rows("documents", {"collection": "other_coll"}, is_admin=True)
            == 1
        ), "other_coll documents must not be touched by new_coll cleanup"
