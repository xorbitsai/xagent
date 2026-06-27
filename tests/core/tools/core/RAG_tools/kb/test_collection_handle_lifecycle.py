"""Tests for delete_collection_data and delete_documents_data on KBCollectionHandle.

H05 Phase 1 – These tests drive out collection-level cascade delete methods on
``LanceDBCollectionHandle``.  They must FAIL before the implementation is added
(RED) and PASS afterwards (GREEN).

Storage isolation is provided by the autouse ``isolate_rag_storage`` fixture in
``tests/conftest.py``.
"""

from __future__ import annotations

import inspect
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import DatabaseOperationError
from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
    KBCollectionHandle,
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
        store.upsert_parses([_parse_row(collection, doc_id, f"h-{doc_id}", user_id=user_id)])
        store.upsert_chunks(
            [_chunk_row(collection, doc_id, f"h-{doc_id}", "cfg1", f"c-{doc_id}", user_id=user_id)]
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
        doc_count = store.count_rows("documents", {"collection": "test_coll"}, is_admin=True)
        assert doc_count == 0

        # other_coll is untouched.
        other_count = store.count_rows("documents", {"collection": "other_coll"}, is_admin=True)
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
    def test_delete_documents_data_tenant_scoped_removes_only_specified_docs(self) -> None:
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
        assert store.count_rows("documents", {"collection": "test_coll", "doc_id": "d1"}, is_admin=True) == 0
        assert store.count_rows("documents", {"collection": "test_coll", "doc_id": "d2"}, is_admin=True) == 0

        # d3 must remain.
        assert store.count_rows("documents", {"collection": "test_coll", "doc_id": "d3"}, is_admin=True) == 1


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

        # Patch cascade_delete_documents (used inside the store's batching loop)
        # to raise so the store's exception-wrapping logic fires.
        cascade_path = (
            "xagent.core.tools.core.RAG_tools.version_management"
            ".cascade_cleaner.cascade_delete_documents"
        )

        with patch(cascade_path, side_effect=RuntimeError("simulated cascade failure")):
            with pytest.raises(DatabaseOperationError) as exc_info:
                handle.delete_documents_data(
                    doc_ids=["d1", "d2"],
                    user_id=None,
                    is_admin=True,
                )

        err = exc_info.value
        assert hasattr(err, "details"), "DatabaseOperationError must have a 'details' attribute"
        details = err.details
        assert details is not None, "details must not be None"
        assert "deleted_counts" in details, f"Missing 'deleted_counts' in details: {details}"
        assert "deleted_doc_ids" in details, f"Missing 'deleted_doc_ids' in details: {details}"
        assert "failed_batch_index" in details, (
            f"Missing 'failed_batch_index' in details: {details}"
        )
        assert isinstance(details["deleted_counts"], dict)
        assert isinstance(details["deleted_doc_ids"], list)
        assert isinstance(details["failed_batch_index"], int)
