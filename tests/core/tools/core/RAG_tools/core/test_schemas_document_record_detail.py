"""Tests for the lean handle-level document record schemas (#508).

``DocumentRecordDetail`` / ``DocumentRecordListResult`` are the semantic types
the collection handle returns for document-row reads. They must map losslessly
back to the legacy file-level ``list[dict]`` shape (the full ``documents`` row).
"""

from datetime import datetime, timezone

from xagent.core.tools.core.RAG_tools.core.schemas import (
    DocumentRecordDetail,
    DocumentRecordListResult,
)

FULL_ROW = {
    "collection": "coll",
    "doc_id": "doc-1",
    "file_id": "file-9",
    "source_path": "/uploads/user_7/report.txt",
    "file_type": "txt",
    "content_hash": "a" * 64,
    "uploaded_at": datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
    "title": None,
    "language": None,
    "user_id": 7,
}


class TestDocumentRecordDetail:
    def test_from_row_to_legacy_dict_round_trip(self) -> None:
        detail = DocumentRecordDetail.from_row(FULL_ROW)
        assert detail.to_legacy_dict() == FULL_ROW

    def test_from_row_normalizes_nan_and_nat_to_none(self) -> None:
        nan = float("nan")
        row = {
            **FULL_ROW,
            "file_id": nan,  # NaN sentinel from pandas object column
            "title": nan,
            "language": None,
            "user_id": nan,  # int64 column upcast to float NaN when null
        }
        detail = DocumentRecordDetail.from_row(row)
        assert detail.file_id is None
        assert detail.title is None
        assert detail.language is None
        assert detail.user_id is None

    def test_user_id_coerced_to_python_int(self) -> None:
        detail = DocumentRecordDetail.from_row({**FULL_ROW, "user_id": 42})
        assert detail.user_id == 42
        assert isinstance(detail.user_id, int)


class TestDocumentRecordListResult:
    def test_to_legacy_dicts_preserves_order_and_count(self) -> None:
        d1 = DocumentRecordDetail.from_row({**FULL_ROW, "doc_id": "d1"})
        d2 = DocumentRecordDetail.from_row({**FULL_ROW, "doc_id": "d2"})
        result = DocumentRecordListResult(documents=[d1, d2], total_count=2)

        legacy = result.to_legacy_dicts()
        assert legacy == [d1.to_legacy_dict(), d2.to_legacy_dict()]
        assert [row["doc_id"] for row in legacy] == ["d1", "d2"]
        assert result.total_count == 2

    def test_empty_result(self) -> None:
        result = DocumentRecordListResult(documents=[], total_count=0)
        assert result.to_legacy_dicts() == []
        assert result.total_count == 0
