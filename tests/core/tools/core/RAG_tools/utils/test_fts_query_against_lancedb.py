"""Real-LanceDB checks that the FTS query builder is index-agnostic."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import lancedb
import pytest

from xagent.core.tools.core.RAG_tools.LanceDB.jieba_dictionary import (
    ensure_jieba_dictionary,
)
from xagent.core.tools.core.RAG_tools.utils.lancedb_query_utils import build_fts_query

DOCS = [
    {"id": 0, "text": "To print an incident report, open Incidents then Tickets."},
    {"id": 1, "text": "Timesheets are approved by the manager every Friday."},
    {"id": 2, "text": "Ticks and mites are common in rural areas."},
    {"id": 3, "text": "事故报告可以在工单页面下载 PDF 文件并打印。"},
    {"id": 4, "text": "Save the draft, then publish."},
    {"id": 5, "text": "Commas, only, here, nothing, else,"},
]
# Single words, so every tokenizer -- including the legacy ngram one -- agrees
# on them; multi-word queries are where the builder deliberately differs.
# No stop word here: the builder drops those, so the raw string is not its equal.
QUERIES = ["incident", "Tickets", "manager", "Timesheets", "nonexistent"]


@pytest.fixture(autouse=True)
def jieba_dictionary(monkeypatch, tmp_path: Path) -> None:
    """Point lance at a dictionary this test installs, not at the machine's."""
    monkeypatch.setenv("LANCE_LANGUAGE_MODEL_HOME", str(tmp_path / "lm"))
    assert ensure_jieba_dictionary() is True


def _table(tmp_path: Path, name: str, **index_params: Any) -> Any:
    table = lancedb.connect(str(tmp_path / name)).create_table(name, DOCS)
    table.create_fts_index("text", replace=True, with_position=True, **index_params)
    return table


def _rows(table: Any, query: Any) -> List[dict]:
    return table.search(query, query_type="fts").limit(9).to_list()


def _hits(table: Any, query: Any) -> List[int]:
    return sorted(row["id"] for row in _rows(table, query))


@pytest.fixture
def jieba_table(tmp_path: Path) -> Any:
    return _table(tmp_path, "jieba_table", base_tokenizer="jieba/default")


@pytest.fixture
def jieba_table_with_unindexed_row(tmp_path: Path) -> Any:
    """Rows added after the index is built are what make lance fail an empty clause."""
    table = _table(tmp_path, "jieba_tail", base_tokenizer="jieba/default")
    table.add([{"id": 6, "text": "Printers are listed under Settings."}])
    return table


@pytest.mark.integration
@pytest.mark.parametrize(
    "index_params",
    [
        {"base_tokenizer": "simple"},
        {"base_tokenizer": "jieba/default"},
        # The parameters every index built before this change still carries.
        {
            "base_tokenizer": "ngram",
            "ngram_min_length": 2,
            "prefix_only": True,
        },
    ],
    ids=["simple", "jieba", "legacy-ngram"],
)
def test_built_query_matches_the_raw_string_on_any_index(
    tmp_path: Path, index_params: dict
):
    """Word queries return the same rows whichever tokenizer the index was built with."""
    table = _table(tmp_path, "t", **index_params)

    for query_text in QUERIES:
        built = build_fts_query(query_text)
        assert built is not None
        assert _hits(table, built) == _hits(table, query_text), query_text


@pytest.mark.integration
def test_terms_are_combined_with_or(jieba_table: Any):
    assert _hits(jieba_table, build_fts_query("incident")) == [0]
    assert _hits(jieba_table, build_fts_query("manager")) == [1]
    assert _hits(jieba_table, build_fts_query("incident manager")) == [0, 1]


@pytest.mark.integration
def test_cjk_query_matches_the_chinese_row(jieba_table: Any):
    assert _hits(jieba_table, build_fts_query("工单")) == [3]


@pytest.mark.integration
def test_edge_punctuation_does_not_drag_in_comma_rows(jieba_table: Any):
    """Row 5 is all commas: the raw ``Save,`` matches it, the built query must not."""
    assert 5 in _hits(jieba_table, "Save,")
    assert _hits(jieba_table, build_fts_query("Save,")) == _hits(
        jieba_table, build_fts_query("Save")
    )
    assert 5 not in _hits(jieba_table, build_fts_query("Save,"))


@pytest.mark.integration
def test_repeated_term_does_not_change_the_score(jieba_table: Any):
    once = _rows(jieba_table, build_fts_query("save"))
    twice = _rows(jieba_table, build_fts_query("save SAVE"))
    assert [(r["id"], r["_score"]) for r in once] == [
        (r["id"], r["_score"]) for r in twice
    ]


@pytest.mark.integration
def test_punctuation_only_query_builds_nothing(jieba_table: Any):
    assert build_fts_query(" ,. ") is None


@pytest.mark.integration
def test_stop_words_do_not_break_the_search(jieba_table_with_unindexed_row: Any):
    """A stop word in the query no longer fails the whole lance query."""
    built = build_fts_query("print an incident report")
    assert _hits(jieba_table_with_unindexed_row, built) == [0]
