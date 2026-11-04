"""Tests for the LanceDB FTS query builder."""

from __future__ import annotations

import logging

import pytest

from xagent.core.tools.core.RAG_tools.utils.lancedb_query_utils import (
    _FTS_STOP_WORDS,
    build_fts_query,
)


def _terms(query_text: str, text_column: str = "text"):
    built = build_fts_query(query_text, text_column)
    assert built is not None
    return [(match.query, match.column) for _, match in built.queries]


def test_build_fts_query_splits_on_whitespace_and_cjk_punctuation():
    assert _terms("点击 Save 按钮，然后审批") == [
        ("点击", "text"),
        ("Save", "text"),
        ("按钮", "text"),
        ("然后审批", "text"),
    ]


def test_build_fts_query_splits_on_cjk_brackets_and_dashes():
    """Brackets, quotes and dashes are indexed tokens too, full-width ones folded."""
    assert _terms("点击「保存」按钮（右上角）即可。参见《手册》——权限……") == [
        ("点击", "text"),
        ("保存", "text"),
        ("按钮", "text"),
        ("右上角", "text"),
        ("即可", "text"),
        ("参见", "text"),
        ("手册", "text"),
        ("权限", "text"),
    ]


@pytest.mark.parametrize("query_text", ["COVID-19", "GPT-4o", "C++", "3.5", "it's"])
def test_build_fts_query_keeps_ascii_punctuation_inside_a_term(query_text: str):
    """These are single tokens in the index; splitting them makes the query miss."""
    assert _terms(query_text) == [(query_text, "text")]


@pytest.mark.parametrize(
    ("query_text", "expected"),
    [
        ("Save,", "Save"),
        ("approve?", "approve"),
        ('"quoted"', "quoted"),
        ("(right)", "right"),
        ("3.5.", "3.5"),
        ("...ellipsis...", "ellipsis"),
    ],
)
def test_build_fts_query_strips_punctuation_off_the_term_edges(
    query_text: str, expected: str
):
    """A trailing comma drags in every chunk that has one."""
    assert _terms(query_text) == [(expected, "text")]


def test_build_fts_query_drops_case_insensitive_duplicates():
    """The index lower-cases tokens, so these clauses would score the same row twice."""
    assert _terms("save save SAVE") == [("save", "text")]


def test_build_fts_query_caps_the_number_of_terms(caplog):
    """A pasted document must not turn into a thousand-clause query."""
    with caplog.at_level(logging.WARNING):
        built = build_fts_query(" ".join(f"term{i}" for i in range(200)))

    assert built is not None
    assert len(built.queries) == 64
    assert "truncated to 64 terms" in caplog.text


def test_build_fts_query_does_not_warn_when_nothing_is_dropped(caplog):
    with caplog.at_level(logging.WARNING):
        build_fts_query(" ".join(f"term{i}" for i in range(64)))

    assert "truncated" not in caplog.text


def test_build_fts_query_keeps_single_term_and_column():
    assert _terms("incident", "body") == [("incident", "body")]


@pytest.mark.parametrize("query_text", ["", " ", "   ", " ,. ", "，。", "😀"])
def test_build_fts_query_returns_none_without_terms(query_text: str):
    """No terms means no FTS query: the space token matches every chunk holding one."""
    assert build_fts_query(query_text) is None


def test_build_fts_query_drops_english_stop_words():
    assert _terms("how do I print an incident report") == [
        ("how", "text"),
        ("do", "text"),
        ("I", "text"),
        ("print", "text"),
        ("incident", "text"),
        ("report", "text"),
    ]


@pytest.mark.parametrize("query_text", ["the a an of to", "THE", "  to  "])
def test_build_fts_query_returns_none_for_stop_words_only(query_text: str):
    assert build_fts_query(query_text) is None


def test_stop_word_list_is_intact():
    """A missing space in the literal silently concatenates two words into one."""
    assert len(_FTS_STOP_WORDS) == 33


def test_build_fts_query_caps_the_terms_left_after_dropping_stop_words(caplog):
    """Stop words go before the cap, so they never eat a slot from a real term."""
    with caplog.at_level(logging.WARNING):
        built = build_fts_query(" ".join(f"the term{i}" for i in range(200)))

    assert built is not None
    assert [match.query for _, match in built.queries] == [
        f"term{i}" for i in range(64)
    ]
    assert "truncated to 64 terms" in caplog.text
