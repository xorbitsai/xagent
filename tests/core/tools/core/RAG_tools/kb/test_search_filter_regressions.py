"""Focused #671 regressions using the persisted embeddings schema."""

import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import lancedb
import pytest

from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
    LanceDBCollectionHandle,
)
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
    _safe_close_table,
    ensure_embeddings_table,
)
from xagent.core.tools.core.RAG_tools.storage.contracts import (
    FilterCondition,
    FilterOperator,
)
from xagent.core.tools.core.RAG_tools.storage.lancedb_filter_utils import (
    translate_filter_expression,
)


@pytest.fixture
def search_table(tmp_path):
    connection = lancedb.connect(str(tmp_path / "search"))
    ensure_embeddings_table(connection, "test", vector_dim=2)
    table = connection.open_table("embeddings_test")
    rows = []
    for doc_id, collection, user_id in [
        ("d1", "col1", 7),
        ("d2", "col1", 8),
        ("d3", "col1", 9),
        ("null", "col1", None),
        ("outside", "other", 8),
    ]:
        rows.append(
            dict(
                collection=collection,
                doc_id=doc_id,
                chunk_id=doc_id,
                parse_hash="parse",
                model="test",
                vector=[0.1, 0.2],
                vector_dimension=2,
                text="shared prefixneedlesuffix",
                chunk_hash=doc_id,
                created_at=datetime(2026, 1, 1),
                metadata=None if doc_id == "null" else json.dumps({"id": doc_id}),
                user_id=user_id,
            )
        )
    table.add(rows)
    table.create_fts_index("text")
    yield table
    _safe_close_table(table)


def make_handle(table):
    store = Mock()
    store.create_index.return_value = SimpleNamespace(
        status="index_ready", advice=None, fts_enabled=True
    )
    store.open_embeddings_table.return_value = (table, "embeddings_test")
    store.build_filter_expression.side_effect = (
        lambda filters, **kw: translate_filter_expression(filters)
    )

    def dense(**kw):
        return (
            table.search(kw["query_vector"])
            .where(translate_filter_expression(kw["filters"]))
            .limit(kw["top_k"])
            .to_list()
        )

    def sparse(**kw):
        return (
            table.search(kw["query_text"], query_type="fts")
            .where(translate_filter_expression(kw["filters"]))
            .limit(kw["top_k"])
            .to_list()
        )

    store.search_vectors_by_model.side_effect = dense
    store.search_vectors_by_model_async = AsyncMock(side_effect=dense)
    store.search_fts_by_model_async = AsyncMock(side_effect=sparse)
    handle = LanceDBCollectionHandle.__new__(LanceDBCollectionHandle)
    object.__setattr__(
        handle,
        "context",
        SimpleNamespace(
            collection="col1",
            vector_index_store=store,
            capabilities=SimpleNamespace(supports_search=True),
        ),
    )
    return handle


@pytest.mark.parametrize("mode", ["dense", "dense_async", "sparse", "sparse_async"])
@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        (FilterCondition("doc_id", FilterOperator.EQ, "d1"), {"d1"}),
        (
            (
                FilterCondition("user_id", FilterOperator.GTE, 8),
                FilterCondition("user_id", FilterOperator.LTE, 8),
            ),
            {"d2"},
        ),
        (
            [
                FilterCondition("user_id", FilterOperator.EQ, 7),
                FilterCondition("user_id", FilterOperator.EQ, 8),
            ],
            {"d1", "d2"},
        ),
        (
            (
                [
                    FilterCondition("user_id", FilterOperator.EQ, 7),
                    FilterCondition("user_id", FilterOperator.EQ, 8),
                ],
                FilterCondition("doc_id", FilterOperator.NE, "d1"),
            ),
            {"d2"},
        ),
        ({"doc_id": {"operator": "in", "value": ["d1", "d2"]}}, {"d1", "d2"}),
        (None, {"d1", "d2", "d3", "null"}),
        ({}, {"d1", "d2", "d3", "null"}),
        ((), {"d1", "d2", "d3", "null"}),
        ([], {"d1", "d2", "d3", "null"}),
    ],
)
async def test_search_preserves_filter_groups(search_table, mode, filters, expected):
    """The handle's predicate executes on LanceDB; async transport is stubbed."""
    handle = make_handle(search_table)
    query = [0.1, 0.2] if mode.startswith("dense") else "shared"
    response = getattr(handle, "search_" + mode)(
        "test", query, top_k=10, filters=filters, is_admin=True
    )
    if mode.endswith("async"):
        response = await response
    assert response.status == "success"
    assert {row.doc_id for row in response.results} == expected


@pytest.mark.parametrize(
    ("operator", "value", "expected"),
    [
        ("eq", 8, {"d2"}),
        ("ne", 8, {"d1", "d3"}),
        ("gt", 8, {"d3"}),
        ("gte", 8, {"d2", "d3"}),
        ("lt", 8, {"d1"}),
        ("lte", 8, {"d1", "d2"}),
        ("in", [7, 9], {"d1", "d3"}),
        ("eq", None, set()),
        ("ne", None, set()),
    ],
)
def test_fallback_legacy_operators(search_table, operator, value, expected):
    # A whole-token FTS query misses, but the literal substring exists.
    response = make_handle(search_table).search_sparse(
        "test",
        "needle",
        top_k=10,
        is_admin=True,
        filters={"user_id": {"operator": operator, "value": value}},
    )
    assert response.status == "success"
    assert {row.doc_id for row in response.results} == expected
    if expected:
        assert any(w.code == "FTS_FALLBACK" for w in response.warnings)


@pytest.mark.parametrize("sequence", [list, tuple, set])
def test_fallback_preserves_legacy_membership(search_table, sequence):
    warnings = []
    # Exercise the scan directly: backend sequence shorthand is a separate issue.
    results = make_handle(search_table)._substring_fallback(
        table=search_table,
        collection="col1",
        query_text="needle",
        model_tag="test",
        top_k=10,
        filters={"metadata": sequence([json.dumps({"id": "d1"}), None])},
        current_warnings=warnings,
    )
    assert {row.doc_id for row in results} == {"d1", "null"}


def test_fallback_loads_nested_filter_fields(search_table):
    filters = (
        [
            FilterCondition("user_id", FilterOperator.EQ, 7),
            FilterCondition("user_id", FilterOperator.EQ, 8),
        ],
        FilterCondition("doc_id", FilterOperator.NE, "d1"),
    )
    results = make_handle(search_table)._substring_fallback(
        table=search_table,
        collection="col1",
        query_text="needle",
        model_tag="test",
        top_k=10,
        filters=filters,
        current_warnings=[],
        batch_size=1,
    )
    assert [row.doc_id for row in results] == ["d2"]


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({"user_id": {"operator": "in", "value": []}}, []),
        ({"user_id": {"operator": "gte", "value": 8}}, ["d2"]),
        ({"doc_id": ["d1", "d2"], "user_id": {"operator": "gte", "value": 8}}, ["d2"]),
    ],
)
def test_fallback_filters_before_limit(search_table, filters, expected):
    results = make_handle(search_table)._substring_fallback(
        table=search_table,
        collection="col1",
        query_text="needle",
        model_tag="test",
        top_k=1,
        filters=filters,
        current_warnings=[],
        batch_size=1,
    )
    assert [row.doc_id for row in results] == expected
