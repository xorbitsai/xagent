"""Tests for the row-only ``delete_document_record`` store primitive (#508).

This primitive deletes ONLY the ``documents`` table row for a document. It must
not cascade into parse/chunk/embedding data (that stays with the cascade path),
must be idempotent, and must preserve document-scoped tenant safety.

Storage isolation/reset is provided by the autouse ``isolate_rag_storage``
fixture in ``tests/conftest.py``.
"""

from datetime import datetime, timezone

from xagent.core.tools.core.RAG_tools.storage.factory import get_vector_index_store


def _doc_row(collection: str, doc_id: str, user_id=None) -> dict:
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


def _parse_row(collection: str, doc_id: str) -> dict:
    return {
        "collection": collection,
        "doc_id": doc_id,
        "parse_hash": "parsehash1",
        "parser": "test_parser",
        "created_at": datetime.now(timezone.utc),
        "params_json": "{}",
        "parsed_content": "parsed text",
        "user_id": None,
    }


def test_deletes_only_documents_row_no_cascade() -> None:
    store = get_vector_index_store()
    store.upsert_documents([_doc_row("coll", "keep"), _doc_row("coll", "del")])
    store.upsert_parses([_parse_row("coll", "del")])

    # Sanity baseline.
    assert store.count_rows("documents", {"collection": "coll"}, is_admin=True) == 2
    assert (
        store.count_rows(
            "parses", {"collection": "coll", "doc_id": "del"}, is_admin=True
        )
        == 1
    )

    deleted = store.delete_document_record("coll", "del", user_id=None, is_admin=True)

    assert deleted == 1
    # documents row gone for the target, the other survives.
    assert (
        store.count_rows(
            "documents", {"collection": "coll", "doc_id": "del"}, is_admin=True
        )
        == 0
    )
    assert (
        store.count_rows(
            "documents", {"collection": "coll", "doc_id": "keep"}, is_admin=True
        )
        == 1
    )
    # parse row for the deleted doc is untouched (no cascade in this primitive).
    assert (
        store.count_rows(
            "parses", {"collection": "coll", "doc_id": "del"}, is_admin=True
        )
        == 1
    )


def test_idempotent_and_missing_returns_zero() -> None:
    store = get_vector_index_store()
    store.upsert_documents([_doc_row("coll", "d1")])

    assert store.delete_document_record("coll", "d1", user_id=None, is_admin=True) == 1
    # Second delete: row already gone -> 0, no error.
    assert store.delete_document_record("coll", "d1", user_id=None, is_admin=True) == 0
    # Never-existed doc -> 0.
    assert (
        store.delete_document_record("coll", "nope", user_id=None, is_admin=True) == 0
    )


def test_missing_documents_table_returns_zero() -> None:
    store = get_vector_index_store()
    # No documents table has been created in this fresh store.
    assert store.delete_document_record("coll", "d1", user_id=None, is_admin=True) == 0


def test_user_scoping_blocks_other_tenant() -> None:
    store = get_vector_index_store()
    store.upsert_documents([_doc_row("coll", "owned", user_id=5)])

    # A different non-admin user cannot delete another tenant's row.
    assert store.delete_document_record("coll", "owned", user_id=6, is_admin=False) == 0
    assert (
        store.count_rows(
            "documents", {"collection": "coll", "doc_id": "owned"}, is_admin=True
        )
        == 1
    )

    # The owner can delete their own row.
    assert store.delete_document_record("coll", "owned", user_id=5, is_admin=False) == 1
    assert (
        store.count_rows(
            "documents", {"collection": "coll", "doc_id": "owned"}, is_admin=True
        )
        == 0
    )
