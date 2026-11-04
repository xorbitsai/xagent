"""Tenant-scoped deletes must not remove other owners' main pointers (#2859).

``main_pointers`` has no ``user_id`` column; ownership comes from ``documents``.
Storage isolation is provided by the autouse ``isolate_rag_storage`` fixture.
"""

import logging
from datetime import datetime, timezone

import pytest

from xagent.core.tools.core.RAG_tools.storage.factory import (
    get_main_pointer_store,
    get_vector_index_store,
)

COLL = "shared"


def _doc(doc_id: str, user_id: int) -> dict:
    return {
        "collection": COLL,
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


@pytest.fixture
def seeded(caplog):
    caplog.set_level(logging.DEBUG)
    index = get_vector_index_store()
    pointers = get_main_pointer_store()
    index.upsert_documents([_doc("doc-1", 1), _doc("doc-2", 2)])
    for doc_id in ("doc-1", "doc-2"):
        pointers.set_main_pointer(COLL, doc_id, "embed", "v1", "h1", model_tag="tag-a")
    return index, pointers


def _no_filter_errors(caplog) -> None:
    # A rejected filter is swallowed by _safe_count_rows and only logged at debug.
    assert "count_rows failed" not in caplog.text


def _has(pointers, doc_id: str) -> bool:
    return pointers.get_main_pointer(COLL, doc_id, "embed", "tag-a") is not None


def test_delete_documents_data_keeps_other_owners_pointer(seeded, caplog) -> None:
    index, pointers = seeded
    index.delete_documents_data(COLL, ["doc-1"], user_id=2, is_admin=False)
    assert _has(pointers, "doc-1")
    assert _has(pointers, "doc-2")
    _no_filter_errors(caplog)


def test_delete_documents_data_removes_own_pointer_only(seeded) -> None:
    index, pointers = seeded
    result = index.delete_documents_data(
        COLL, ["doc-1", "doc-2"], user_id=2, is_admin=False
    )
    assert result["main_pointers"] == 1
    assert _has(pointers, "doc-1")
    assert not _has(pointers, "doc-2")


def test_cascade_delete_document_keeps_other_owners_pointer(seeded, caplog) -> None:
    index, pointers = seeded
    index.cascade_delete(
        target="document",
        collection=COLL,
        doc_id="doc-1",
        user_id=2,
        is_admin=False,
        preview_only=False,
        confirm=True,
    )
    assert _has(pointers, "doc-1")
    _no_filter_errors(caplog)


def test_cascade_delete_collection_keeps_other_owners_pointer(seeded, caplog) -> None:
    index, pointers = seeded
    index.cascade_delete(
        target="collection",
        collection=COLL,
        user_id=2,
        is_admin=False,
        preview_only=False,
        confirm=True,
    )
    assert _has(pointers, "doc-1")
    assert not _has(pointers, "doc-2")
    _no_filter_errors(caplog)


def test_admin_delete_still_removes_all_pointers(seeded) -> None:
    index, pointers = seeded
    index.delete_documents_data(COLL, ["doc-1", "doc-2"], user_id=None, is_admin=True)
    assert not _has(pointers, "doc-1")
    assert not _has(pointers, "doc-2")


def _break_owner_lookup(monkeypatch) -> None:
    from xagent.core.tools.core.RAG_tools.storage import lancedb_stores

    def boom(*args, **kwargs):
        raise RuntimeError("lookup failed")

    monkeypatch.setattr(lancedb_stores, "query_to_list", boom)


def _row_counts(index) -> dict:
    from xagent.core.tools.core.RAG_tools.storage.factory import (
        get_vector_store_raw_connection,
    )

    conn = get_vector_store_raw_connection()
    return {n: conn.open_table(n).count_rows() for n in ("documents", "main_pointers")}


def test_failed_owner_lookup_raises_and_deletes_nothing(seeded, monkeypatch) -> None:
    from xagent.core.tools.core.RAG_tools.core.exceptions import (
        DatabaseOperationError,
    )

    index, pointers = seeded
    before = _row_counts(index)
    _break_owner_lookup(monkeypatch)
    with pytest.raises(DatabaseOperationError, match="delete document batch"):
        index.delete_documents_data(COLL, ["doc-2"], user_id=2, is_admin=False)
    with pytest.raises(DatabaseOperationError, match="document owners"):
        index.cascade_delete(
            target="collection",
            collection=COLL,
            user_id=2,
            is_admin=False,
            preview_only=False,
            confirm=True,
        )
    assert _row_counts(index) == before
    assert _has(pointers, "doc-2")


def test_unreadable_documents_schema_fails_closed(seeded) -> None:
    from xagent.core.tools.core.RAG_tools.core.exceptions import (
        DatabaseOperationError,
    )
    from xagent.core.tools.core.RAG_tools.storage import lancedb_stores

    class _Unreadable:
        @property
        def schema(self):
            raise RuntimeError("schema unreadable")

    class _Conn:
        def __init__(self, real):
            self.real = real

        def open_table(self, name):
            return _Unreadable() if name == "documents" else self.real.open_table(name)

    index, _ = seeded
    with pytest.raises(DatabaseOperationError, match="document owners"):
        lancedb_stores._vis_main_pointers_filter(
            conn=_Conn(index._get_connection()),
            filter_expr=f"collection == '{COLL}'",
            user_id=2,
        )


def test_main_pointers_with_user_id_column_still_owner_scoped(seeded) -> None:
    import pyarrow as pa

    from xagent.core.tools.core.RAG_tools.storage.factory import (
        get_vector_store_raw_connection,
    )

    index, pointers = seeded
    conn = get_vector_store_raw_connection()
    data = conn.open_table("main_pointers").to_arrow()
    data = data.append_column("user_id", pa.nulls(data.num_rows, pa.int64()))
    conn.create_table("main_pointers", data=data, mode="overwrite")
    index.invalidate_table_cache()

    result = index.delete_documents_data(COLL, ["doc-2"], user_id=2, is_admin=False)
    assert result["main_pointers"] == 1
    assert not _has(pointers, "doc-2")
    assert _has(pointers, "doc-1")
    index.cascade_delete(
        target="document",
        collection=COLL,
        doc_id="doc-1",
        user_id=2,
        is_admin=False,
        preview_only=False,
        confirm=True,
    )
    assert _has(pointers, "doc-1")


def test_unauthenticated_delete_keeps_pointers_without_filter_errors(
    seeded, caplog
) -> None:
    index, pointers = seeded
    index.cascade_delete(
        target="collection",
        collection=COLL,
        user_id=None,
        is_admin=False,
        preview_only=False,
        confirm=True,
    )
    assert _has(pointers, "doc-1")
    assert _has(pointers, "doc-2")
    _no_filter_errors(caplog)


def test_preview_reports_only_owned_pointers(seeded) -> None:
    index, pointers = seeded
    non_owner = index.cleanup_cascade_by_scope(
        COLL, "doc-1", "pointers", user_id=2, is_admin=False, preview_only=True
    )
    assert non_owner.get("main_pointers", 0) == 0
    owner = index.cleanup_cascade_by_scope(
        COLL, "doc-2", "pointers", user_id=2, is_admin=False, preview_only=True
    )
    assert owner["main_pointers"] == 1
    assert _has(pointers, "doc-2")


def test_doc_id_with_quote_is_escaped_in_owner_filter(seeded) -> None:
    index, pointers = seeded
    quoted = "o'brien\" OR 1=1"
    index.upsert_documents([_doc(quoted, 2)])
    pointers.set_main_pointer(COLL, quoted, "embed", "v1", "h1", model_tag="tag-a")
    result = index.cascade_delete(
        target="collection",
        collection=COLL,
        user_id=2,
        is_admin=False,
        preview_only=False,
        confirm=True,
    )
    assert result["main_pointers"] == 2
    assert not _has(pointers, quoted)
    assert _has(pointers, "doc-1")


def test_owner_lookup_is_batched(seeded, monkeypatch) -> None:
    from xagent.core.tools.core.RAG_tools.storage import lancedb_stores

    index, pointers = seeded
    monkeypatch.setattr(lancedb_stores, "DEFAULT_VECTOR_STORE_DELETE_BATCH_SIZE", 1)
    index.cascade_delete(
        target="collection",
        collection=COLL,
        user_id=2,
        is_admin=False,
        preview_only=False,
        confirm=True,
    )
    assert _has(pointers, "doc-1")
    assert not _has(pointers, "doc-2")


def test_cascade_delete_document_removes_own_pointer(seeded) -> None:
    index, pointers = seeded
    index.cascade_delete(
        target="document",
        collection=COLL,
        doc_id="doc-2",
        user_id=2,
        is_admin=False,
        preview_only=False,
        confirm=True,
    )
    assert not _has(pointers, "doc-2")
    assert _has(pointers, "doc-1")


def test_pointers_scope_owner_removes_non_owner_does_not(seeded, caplog) -> None:
    index, pointers = seeded
    denied = index.cleanup_cascade_by_scope(
        COLL,
        "doc-1",
        "pointers",
        user_id=2,
        is_admin=False,
        preview_only=False,
        confirm=True,
    )
    assert denied.get("main_pointers", 0) == 0
    assert _has(pointers, "doc-1")

    removed = index.cleanup_cascade_by_scope(
        COLL,
        "doc-1",
        "pointers",
        user_id=1,
        is_admin=False,
        preview_only=False,
        confirm=True,
    )
    assert removed["main_pointers"] == 1
    assert not _has(pointers, "doc-1")
    assert _has(pointers, "doc-2")
    _no_filter_errors(caplog)


def test_legacy_null_owner_document_keeps_pointer_and_row(seeded, caplog) -> None:
    index, pointers = seeded
    index.upsert_documents([_doc("doc-legacy", None)])
    pointers.set_main_pointer(
        COLL, "doc-legacy", "embed", "v1", "h1", model_tag="tag-a"
    )
    result = index.delete_documents_data(
        COLL, ["doc-legacy"], user_id=2, is_admin=False
    )
    assert result.get("documents", 0) == 0
    assert result.get("main_pointers", 0) == 0
    assert _has(pointers, "doc-legacy")
    _no_filter_errors(caplog)


def test_no_match_filter_is_valid_on_main_pointers(seeded) -> None:
    from xagent.core.tools.core.RAG_tools.core.config import (
        MAIN_POINTERS_NO_MATCH_FILTER,
    )
    from xagent.core.tools.core.RAG_tools.storage.factory import (
        get_vector_store_raw_connection,
    )

    conn = get_vector_store_raw_connection()
    table = conn.open_table("main_pointers")
    expr = f"collection == '{COLL}' AND {MAIN_POINTERS_NO_MATCH_FILTER}"
    assert table.count_rows(expr) == 0
    assert len(table.search().where(expr).to_arrow()) == 0
    table.delete(expr)
    assert table.count_rows() == 2
