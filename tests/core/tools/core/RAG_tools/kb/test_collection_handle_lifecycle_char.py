"""Characterization tests for collection lifecycle operations (H05 Phase 0).

These tests lock down the CURRENT behaviour of store/management-layer delete,
rename and count operations before any H05 refactoring moves them into
KBCollectionHandle.  All tests must remain green on unmodified code.

Storage isolation/reset is provided by the autouse ``isolate_rag_storage``
fixture in ``tests/conftest.py``.
"""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import DatabaseOperationError
from xagent.core.tools.core.RAG_tools.storage.factory import get_vector_index_store

# ---------------------------------------------------------------------------
# Helpers – row builders
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Test 1: admin delete_collection_data removes all rows for the collection
# ---------------------------------------------------------------------------


class TestAdminDeleteCollectionData:
    def test_admin_delete_collection_data_removes_all_tables(self) -> None:
        """store.delete_collection_data removes documents/parses/chunks for col_a.

        Characterizes the return value shape (dict[str, int]) and the observable
        post-delete state: zero rows for the target collection in every table.
        """
        store = get_vector_index_store()

        # Seed documents, parses, and chunks for col_a.
        store.upsert_documents(
            [
                _doc_row("col_a", "d1"),
                _doc_row("col_a", "d2"),
            ]
        )
        store.upsert_parses(
            [
                _parse_row("col_a", "d1", "h1"),
                _parse_row("col_a", "d2", "h2"),
            ]
        )
        store.upsert_chunks(
            [
                _chunk_row("col_a", "d1", "h1", "cfg1", "c0"),
                _chunk_row("col_a", "d1", "h1", "cfg1", "c1"),
            ]
        )

        # Seed a second collection to verify isolation.
        store.upsert_documents([_doc_row("col_b", "d3")])
        store.upsert_parses([_parse_row("col_b", "d3", "h3")])

        warnings_out: list[str] = []
        result = store.delete_collection_data(
            "col_a",
            user_id=None,
            is_admin=True,
            warnings_out=warnings_out,
        )

        # Return value must be a dict mapping table names to integer counts.
        assert isinstance(result, dict), "delete_collection_data must return a dict"
        for key, val in result.items():
            assert isinstance(key, str), f"key {key!r} must be str"
            assert isinstance(val, int), f"count for {key!r} must be int"

        # All col_a rows must be gone.
        assert (
            store.count_rows("documents", {"collection": "col_a"}, is_admin=True) == 0
        )
        assert store.count_rows("parses", {"collection": "col_a"}, is_admin=True) == 0
        assert store.count_rows("chunks", {"collection": "col_a"}, is_admin=True) == 0

        # col_b must be unaffected.
        assert (
            store.count_rows("documents", {"collection": "col_b"}, is_admin=True) == 1
        )
        assert store.count_rows("parses", {"collection": "col_b"}, is_admin=True) == 1


# ---------------------------------------------------------------------------
# Test 2: tenant delete_documents_data only removes the caller's documents
# ---------------------------------------------------------------------------


class TestTenantDeleteDocumentsData:
    def test_tenant_delete_documents_data_only_removes_own_docs(self) -> None:
        """delete_documents_data with is_admin=False is user-scoped.

        d1 (user u1) is deleted; d2 (user u2) must survive.
        """
        store = get_vector_index_store()

        store.upsert_documents(
            [
                _doc_row("coll", "d1", user_id=1),
                _doc_row("coll", "d2", user_id=2),
            ]
        )
        store.upsert_parses(
            [
                _parse_row("coll", "d1", "h1", user_id=1),
                _parse_row("coll", "d2", "h2", user_id=2),
            ]
        )
        store.upsert_chunks(
            [
                _chunk_row("coll", "d1", "h1", "cfg1", "c0", user_id=1),
                _chunk_row("coll", "d2", "h2", "cfg1", "c1", user_id=2),
            ]
        )

        warnings_out: list[str] = []
        store.delete_documents_data(
            "coll",
            ["d1"],
            user_id=1,
            is_admin=False,
            warnings_out=warnings_out,
        )

        # d1's rows should be gone.
        assert (
            store.count_rows(
                "documents", {"collection": "coll", "doc_id": "d1"}, is_admin=True
            )
            == 0
        )
        assert (
            store.count_rows(
                "parses", {"collection": "coll", "doc_id": "d1"}, is_admin=True
            )
            == 0
        )
        assert (
            store.count_rows(
                "chunks", {"collection": "coll", "doc_id": "d1"}, is_admin=True
            )
            == 0
        )

        # d2's rows must still be present.
        assert (
            store.count_rows(
                "documents", {"collection": "coll", "doc_id": "d2"}, is_admin=True
            )
            == 1
        )
        assert (
            store.count_rows(
                "parses", {"collection": "coll", "doc_id": "d2"}, is_admin=True
            )
            == 1
        )
        assert (
            store.count_rows(
                "chunks", {"collection": "coll", "doc_id": "d2"}, is_admin=True
            )
            == 1
        )


# ---------------------------------------------------------------------------
# Test 3: partial-failure contract for delete_documents_data
# ---------------------------------------------------------------------------


class TestDeleteDocumentsPartialFailure:
    def test_delete_documents_partial_failure_raises_with_contract_details(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When cascade_delete_documents raises on a batch, DatabaseOperationError
        is re-raised with a .details dict that contains:
          - deleted_counts  (dict of prior-batch counts)
          - deleted_doc_ids (list of successfully deleted doc ids)
          - failed_batch_index (1-based int indicating which batch failed)

        This contract is the downstream input for CollectionOperationResult.partial_success.
        """
        store = get_vector_index_store()

        # Insert two docs so we get at least two batches when batch_size=1.
        store.upsert_documents(
            [
                _doc_row("coll", "d1", user_id=None),
                _doc_row("coll", "d2", user_id=None),
            ]
        )

        call_count = 0

        def _fail_on_second(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise RuntimeError("simulated cascade failure on batch 2")
            # First call succeeds – return minimal counts dict.
            return {"documents": 1, "parses": 0, "chunks": 0}

        # _vis_cascade_delete_documents is the module-level function called inside
        # delete_documents_data's batching loop; patch it at its definition site in
        # lancedb_stores so the batching loop picks up the patched version.
        target = (
            "xagent.core.tools.core.RAG_tools.storage"
            ".lancedb_stores._vis_cascade_delete_documents"
        )
        with patch(target, side_effect=_fail_on_second):
            # Force batch_size to 1 so each doc is its own batch.
            with patch(
                "xagent.core.tools.core.RAG_tools.storage.lancedb_stores"
                ".DEFAULT_VECTOR_STORE_DELETE_BATCH_SIZE",
                1,
            ):
                with pytest.raises(DatabaseOperationError) as exc_info:
                    store.delete_documents_data(
                        "coll",
                        ["d1", "d2"],
                        user_id=None,
                        is_admin=True,
                        warnings_out=[],
                    )

        details = exc_info.value.details
        assert isinstance(details, dict), ".details must be a dict"

        # All three downstream-contract keys must be present.
        assert "deleted_counts" in details, "missing 'deleted_counts' in .details"
        assert "deleted_doc_ids" in details, "missing 'deleted_doc_ids' in .details"
        assert "failed_batch_index" in details, (
            "missing 'failed_batch_index' in .details"
        )

        # Type checks.
        assert isinstance(details["deleted_counts"], dict)
        assert isinstance(details["deleted_doc_ids"], list)
        assert isinstance(details["failed_batch_index"], int)

        # The first batch succeeded so failed_batch_index must be >= 2.
        assert details["failed_batch_index"] >= 2


# ---------------------------------------------------------------------------
# Test 4: rename_collection_data updates the collection field in all tables
# ---------------------------------------------------------------------------


class TestRenameCollectionData:
    def test_rename_collection_data_updates_collection_field_in_all_tables(
        self,
    ) -> None:
        """rename_collection_data rewrites the 'collection' field in every table.

        After rename:
          - old_name rows: 0 in documents, parses, chunks
          - new_name rows: original counts
        """
        store = get_vector_index_store()

        store.upsert_documents(
            [
                _doc_row("old_name", "d1"),
                _doc_row("old_name", "d2"),
            ]
        )
        store.upsert_parses(
            [
                _parse_row("old_name", "d1", "h1"),
                _parse_row("old_name", "d2", "h2"),
            ]
        )
        store.upsert_chunks(
            [
                _chunk_row("old_name", "d1", "h1", "cfg1", "c0"),
            ]
        )

        original_doc_count = store.count_rows(
            "documents", {"collection": "old_name"}, is_admin=True
        )
        original_parse_count = store.count_rows(
            "parses", {"collection": "old_name"}, is_admin=True
        )
        original_chunk_count = store.count_rows(
            "chunks", {"collection": "old_name"}, is_admin=True
        )

        assert original_doc_count == 2
        assert original_parse_count == 2
        assert original_chunk_count == 1

        warnings = store.rename_collection_data(
            "old_name",
            "new_name",
            user_id=None,
            is_admin=True,
        )

        # The call must return a list (of warning strings).
        assert isinstance(warnings, list)

        # rename_collection_data opens tables via conn.open_table() directly and
        # does not invalidate the in-process table cache. We must flush it before
        # counting so that count_rows() sees the on-disk state rather than a stale
        # cached handle. This is the current behavior contract – callers must
        # invalidate the cache after a rename.
        store.invalidate_table_cache()

        # old_name must have 0 rows in every table.
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

        # new_name must have the original row counts.
        assert (
            store.count_rows("documents", {"collection": "new_name"}, is_admin=True)
            == original_doc_count
        )
        assert (
            store.count_rows("parses", {"collection": "new_name"}, is_admin=True)
            == original_parse_count
        )
        assert (
            store.count_rows("chunks", {"collection": "new_name"}, is_admin=True)
            == original_chunk_count
        )


# ---------------------------------------------------------------------------
# Test 5: count_documents_grouped_by_collection returns per-collection counts
# ---------------------------------------------------------------------------


class TestCountDocumentsGroupedByCollection:
    def test_count_documents_grouped_by_collection_returns_per_collection_counts(
        self,
    ) -> None:
        """count_documents_grouped_by_collection returns a per-collection doc count.

        col_a: 3 docs, col_b: 2 docs.  The returned mapping must reflect these
        exact counts for the requested collection names.
        """
        store = get_vector_index_store()

        # Insert 3 docs into col_a.
        store.upsert_documents(
            [
                _doc_row("col_a", "a1"),
                _doc_row("col_a", "a2"),
                _doc_row("col_a", "a3"),
            ]
        )
        # Insert 2 docs into col_b.
        store.upsert_documents(
            [
                _doc_row("col_b", "b1"),
                _doc_row("col_b", "b2"),
            ]
        )

        result = store.count_documents_grouped_by_collection(
            collection_names=["col_a", "col_b"],
            user_id=None,
            is_admin=True,
        )

        assert isinstance(result, dict), "result must be a dict"
        assert result.get("col_a") == 3, f"col_a expected 3, got {result.get('col_a')}"
        assert result.get("col_b") == 2, f"col_b expected 2, got {result.get('col_b')}"
