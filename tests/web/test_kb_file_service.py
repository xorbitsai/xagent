"""Tests for KB uploaded-file/document bridge helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from xagent.web.services import kb_file_service


def test_list_documents_for_user_delegates_to_vector_store_without_raw_lancedb() -> (
    None
):
    """Document listing should go through the storage contract, not LanceDB directly."""
    fake_store = SimpleNamespace()
    fake_store.list_document_records = lambda **kwargs: [
        SimpleNamespace(
            collection="demo",
            doc_id="doc-1",
            file_id="file-1",
            source_path="/tmp/demo.pdf",
        )
    ]

    with patch.object(
        kb_file_service,
        "get_vector_index_store",
        return_value=fake_store,
    ) as mock_get_store:
        records = kb_file_service.list_documents_for_user(
            user_id=123,
            is_admin=False,
            collection_name="demo",
        )

    mock_get_store.assert_called_once_with()
    assert records == [
        {
            "collection": "demo",
            "doc_id": "doc-1",
            "file_id": "file-1",
            "source_path": "/tmp/demo.pdf",
        }
    ]


def test_aggregate_uploaded_file_statuses_chunks_file_ids_with_unbounded_limit() -> (
    None
):
    """Large file-id lists must be queried in batches with max_results=-1."""
    file_ids = [f"file-{index}" for index in range(250)]
    list_calls: list[dict[str, object]] = []

    def _fake_list_document_records(**kwargs: object) -> list[object]:
        list_calls.append(dict(kwargs))
        return []

    fake_store = SimpleNamespace(list_document_records=_fake_list_document_records)

    with (
        patch.object(
            kb_file_service, "get_vector_index_store", return_value=fake_store
        ),
        patch.object(kb_file_service, "load_ingestion_status", return_value=[]),
        patch.object(kb_file_service, "_load_indexed_doc_refs", return_value=set()),
    ):
        status_map = kb_file_service.aggregate_uploaded_file_statuses(
            file_ids=file_ids,
            user_id=7,
            is_admin=False,
            use_cache=False,
        )

    assert len(list_calls) == 2
    assert len(list_calls[0]["file_ids"]) == 200
    assert len(list_calls[1]["file_ids"]) == 50
    assert all(call["max_results"] == -1 for call in list_calls)
    assert all(call["user_id"] == 7 for call in list_calls)
    assert status_map == {file_id: "UNKNOWN" for file_id in file_ids}
