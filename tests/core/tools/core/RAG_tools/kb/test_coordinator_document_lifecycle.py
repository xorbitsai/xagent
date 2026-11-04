"""Tests for coordinator document-row delegation + sync wrappers (#508).

The coordinator opens a collection handle (Option A: via ``get_context``, which
resolves the backend from collection metadata) and delegates document-row
operations to it. Sync wrappers bridge the async coordinator for legacy callers.

Storage/coordinator reset is handled by the autouse ``isolate_rag_storage``
fixture in ``tests/conftest.py``.
"""

from pathlib import Path
from unittest.mock import patch

from xagent.core.tools.core.RAG_tools.core.schemas import RegisterDocumentRequest
from xagent.core.tools.core.RAG_tools.kb.coordinator import get_kb_coordinator
from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import LanceDBMetadataStore


def test_register_load_list_delete_sync(tmp_path: Path) -> None:
    coord = get_kb_coordinator()
    src = tmp_path / "a.txt"
    src.write_text("hello world")

    response = coord.register_document_sync(
        RegisterDocumentRequest(
            collection="coll", source_path=str(src), doc_id="doc-1", user_id=7
        )
    )
    assert response.created is True
    assert response.doc_id == "doc-1"

    detail = coord.load_document_sync("coll", "doc-1", is_admin=True)
    assert detail is not None
    assert detail.doc_id == "doc-1"
    assert detail.user_id == 7

    listing = coord.list_document_records_sync("coll", is_admin=True, limit=100)
    assert listing.total_count == 1
    assert listing.documents[0].doc_id == "doc-1"

    assert coord.delete_document_record_sync("coll", "doc-1", is_admin=True) == 1
    assert coord.load_document_sync("coll", "doc-1", is_admin=True) is None


def test_register_into_missing_collection_is_tolerated(tmp_path: Path) -> None:
    """hide_missing tolerance: registering into a never-created collection works."""
    coord = get_kb_coordinator()
    src = tmp_path / "a.txt"
    src.write_text("x")

    response = coord.register_document_sync(
        RegisterDocumentRequest(
            collection="never_created", source_path=str(src), doc_id="d1"
        )
    )
    assert response.created is True


def test_document_ops_route_through_get_context(tmp_path: Path) -> None:
    """Locks Option A: each op resolves context via metadata get_collection."""
    coord = get_kb_coordinator()
    src = tmp_path / "a.txt"
    src.write_text("x")

    calls: list[str] = []
    original = LanceDBMetadataStore.get_collection

    async def spy(self, collection_name):  # type: ignore[no-untyped-def]
        calls.append(collection_name)
        return await original(self, collection_name)

    with patch.object(LanceDBMetadataStore, "get_collection", spy):
        coord.register_document_sync(
            RegisterDocumentRequest(
                collection="coll", source_path=str(src), doc_id="d1"
            )
        )

    assert "coll" in calls
