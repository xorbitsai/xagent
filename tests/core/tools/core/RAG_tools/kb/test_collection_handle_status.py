"""Tests for the ingestion-status lifecycle inline in LanceDBCollectionHandle (#513).

The handle owns collection-scoped status mechanics: write, load, clear (sync +
async).  The coordinator / facade thinning is exercised by these tests via the
real storage layer.

Storage isolation/reset is provided by the autouse ``isolate_rag_storage``
fixture in ``tests/conftest.py``.
"""

from __future__ import annotations

import pytest

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


def make_handle(collection: str = "coll") -> LanceDBCollectionHandle:
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


def test_write_then_load_status_row_fields() -> None:
    """write + load returns a row with all expected fields set correctly.

    Field expectations mirror management/test_status.py:36 byte-for-byte.
    """
    handle = make_handle("test_collection")
    handle.write_ingestion_status(
        "test_doc",
        status="failed",
        message="boom",
        parse_hash="hash-abc",
    )
    records = handle.load_ingestion_status(doc_id="test_doc", is_admin=True)
    assert len(records) == 1
    row = records[0]
    assert row["collection"] == "test_collection"
    assert row["doc_id"] == "test_doc"
    assert row["status"] == "failed"
    assert row["message"] == "boom"
    assert row["parse_hash"] == "hash-abc"


def test_clear_status() -> None:
    """clear_ingestion_status removes the status row for a document."""
    handle = make_handle("clear_coll")
    handle.write_ingestion_status("doc-1", status="success")
    rows_before = handle.load_ingestion_status(doc_id="doc-1", is_admin=True)
    assert len(rows_before) == 1

    handle.clear_ingestion_status("doc-1", is_admin=True)
    rows_after = handle.load_ingestion_status(doc_id="doc-1", is_admin=True)
    assert len(rows_after) == 0


@pytest.mark.asyncio
async def test_write_load_clear_async() -> None:
    """Async handle methods (write/load/clear) round-trip correctly."""
    handle = make_handle("async_coll")

    await handle.write_ingestion_status_async(
        "async-doc",
        status="pending",
        message="async-msg",
    )

    rows = await handle.load_ingestion_status_async(doc_id="async-doc", is_admin=True)
    assert len(rows) == 1
    assert rows[0]["status"] == "pending"
    assert rows[0]["message"] == "async-msg"

    await handle.clear_ingestion_status_async("async-doc", is_admin=True)
    rows_after = await handle.load_ingestion_status_async(
        doc_id="async-doc", is_admin=True
    )
    assert len(rows_after) == 0


def test_rename_collection_status_never_raises() -> None:
    """handle.rename_collection_status is best-effort: returns List[str] and never raises
    even when the underlying store raises.
    """
    from unittest.mock import patch

    handle = make_handle("rename_sync_coll")

    with patch.object(
        handle.ingestion_status_store,
        "rename_collection_status",
        side_effect=RuntimeError("store exploded"),
    ):
        result = handle.rename_collection_status("new_name", None, True)

    assert isinstance(result, list)
    assert len(result) == 1
    assert "store exploded" in result[0]


@pytest.mark.asyncio
async def test_rename_collection_status_async_never_raises() -> None:
    """handle.rename_collection_status_async is best-effort: returns List[str] and never
    raises even when the underlying async store call raises.
    """
    from unittest.mock import AsyncMock, patch

    handle = make_handle("rename_async_coll")

    with patch.object(
        handle.ingestion_status_store,
        "rename_collection_status_async",
        new=AsyncMock(side_effect=RuntimeError("async store exploded")),
    ):
        result = await handle.rename_collection_status_async("new_name", None, True)

    assert isinstance(result, list)
    assert len(result) == 1
    assert "async store exploded" in result[0]
