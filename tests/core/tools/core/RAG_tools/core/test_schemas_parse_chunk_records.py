"""Tests for the lean handle-level parse/chunk record schemas (#509).

``ParseRecordDetail`` / ``ChunkRecordDetail`` / ``ChunkRecordSnapshot`` are the
semantic types the collection handle uses for parse/chunk-row snapshot and
restore. They must map losslessly back to the raw table-row dict shape so a
restore re-upserts every column exactly (the column sets are locked here against
the LanceDB ``parses`` / ``chunks`` schemas).
"""

from datetime import datetime, timezone

from xagent.core.tools.core.RAG_tools.core.schemas import (
    ChunkRecordDetail,
    ChunkRecordSnapshot,
    ParseRecordDetail,
)

PARSE_ROW = {
    "collection": "coll",
    "doc_id": "doc-1",
    "parse_hash": "p" * 16,
    "parser": "local:default@v1.0.0",
    "created_at": datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
    "params_json": "{}",
    "parsed_content": '[{"text": "hello", "metadata": {}}]',
    "user_id": 7,
}

CHUNK_ROW = {
    "collection": "coll",
    "doc_id": "doc-1",
    "parse_hash": "p" * 16,
    "chunk_id": "chunk-0",
    "index": 0,
    "text": "hello world",
    "page_number": 1,
    "section": "Intro",
    "anchor": "a0",
    "json_path": "$.body[0]",
    "chunk_hash": "c" * 16,
    "config_hash": "cfg" * 4,
    "created_at": datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
    "metadata": '{"layout_type": "text"}',
    "user_id": 7,
}


class TestParseRecordDetail:
    def test_from_row_to_legacy_dict_round_trip(self) -> None:
        detail = ParseRecordDetail.from_row(PARSE_ROW)
        assert detail.to_legacy_dict() == PARSE_ROW

    def test_column_set_matches_parses_schema(self) -> None:
        # Locks the full parses column set (lossless oracle).
        assert set(ParseRecordDetail.from_row(PARSE_ROW).to_legacy_dict().keys()) == {
            "collection",
            "doc_id",
            "parse_hash",
            "parser",
            "created_at",
            "params_json",
            "parsed_content",
            "user_id",
        }

    def test_from_row_normalizes_nan_to_none_and_coerces_user_id(self) -> None:
        nan = float("nan")
        detail = ParseRecordDetail.from_row({**PARSE_ROW, "user_id": nan})
        assert detail.user_id is None

        detail2 = ParseRecordDetail.from_row({**PARSE_ROW, "user_id": 42})
        assert detail2.user_id == 42
        assert isinstance(detail2.user_id, int)

    def test_legacy_row_missing_user_id(self) -> None:
        row = {k: v for k, v in PARSE_ROW.items() if k != "user_id"}
        detail = ParseRecordDetail.from_row(row)
        assert detail.user_id is None


class TestChunkRecordDetail:
    def test_from_row_to_legacy_dict_round_trip(self) -> None:
        detail = ChunkRecordDetail.from_row(CHUNK_ROW)
        assert detail.to_legacy_dict() == CHUNK_ROW

    def test_column_set_matches_chunks_schema(self) -> None:
        assert set(ChunkRecordDetail.from_row(CHUNK_ROW).to_legacy_dict().keys()) == {
            "collection",
            "doc_id",
            "parse_hash",
            "chunk_id",
            "index",
            "text",
            "page_number",
            "section",
            "anchor",
            "json_path",
            "chunk_hash",
            "config_hash",
            "created_at",
            "metadata",
            "user_id",
        }

    def test_from_row_normalizes_nan_and_coerces_ints(self) -> None:
        nan = float("nan")
        row = {
            **CHUNK_ROW,
            "page_number": nan,
            "section": nan,
            "anchor": None,
            "json_path": nan,
            "user_id": nan,
        }
        detail = ChunkRecordDetail.from_row(row)
        assert detail.page_number is None
        assert detail.section is None
        assert detail.anchor is None
        assert detail.json_path is None
        assert detail.user_id is None
        # index is a real value -> coerced to a plain int
        assert detail.index == 0
        assert isinstance(detail.index, int)


class TestChunkRecordSnapshot:
    def test_round_trips_ordered_rows(self) -> None:
        rows = [
            {**CHUNK_ROW, "chunk_id": "chunk-0", "index": 0},
            {**CHUNK_ROW, "chunk_id": "chunk-1", "index": 1},
        ]
        snapshot = ChunkRecordSnapshot.from_rows(rows)
        assert [c.chunk_id for c in snapshot.chunks] == ["chunk-0", "chunk-1"]
        assert snapshot.to_legacy_dicts() == rows

    def test_empty_snapshot(self) -> None:
        snapshot = ChunkRecordSnapshot.from_rows([])
        assert snapshot.chunks == []
        assert snapshot.to_legacy_dicts() == []
