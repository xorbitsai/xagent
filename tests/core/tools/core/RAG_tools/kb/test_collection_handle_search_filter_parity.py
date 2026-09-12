"""#671 - one filter input must mean one predicate on every search path.

Dense dropped every non-dict filter shape that sparse honoured, and both
substring fallbacks answered an operator filter with zero rows: the sync one
compared a column against the operator dict, the async one pushed the dict at a
store that rejects it and swallowed the error.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
    LanceDBCollectionHandle,
)
from xagent.core.tools.core.RAG_tools.storage.contracts import (
    FilterCondition,
    FilterOperator,
)


def _make_handle():
    handle = LanceDBCollectionHandle.__new__(LanceDBCollectionHandle)
    ctx = MagicMock()
    ctx.collection = "docs"
    object.__setattr__(handle, "context", ctx)
    store = MagicMock()
    ctx.vector_index_store = store
    caps = MagicMock()
    caps.supports_search = True
    ctx.capabilities = caps
    return handle, ctx, store


def _index_result():
    obj = MagicMock()
    obj.status = "index_ready"
    obj.advice = None
    obj.fts_enabled = True
    return obj


def _conditions(expr: Any) -> list[tuple[Any, Any, Any]]:
    if expr is None:
        return []
    if isinstance(expr, (tuple, list)):
        out: list[tuple[Any, Any, Any]] = []
        for item in expr:
            out.extend(_conditions(item))
        return out
    operator = getattr(expr, "operator", None)
    return [(expr.field, getattr(operator, "value", operator), expr.value)]


PAGE_GTE_2 = FilterCondition(field="page_number", operator=FilterOperator.GTE, value=2)

# Every shape the engines accept, and the triples it must become.
FILTER_SHAPES = [
    ("legacy_dict", {"page_number": {"operator": "gte", "value": 2}}),
    ("single_condition", PAGE_GTE_2),
    ("tuple_of_conditions", (PAGE_GTE_2,)),
    ("list_of_conditions", [PAGE_GTE_2]),
]


@pytest.mark.parametrize(
    ("shape", "filters"), FILTER_SHAPES, ids=[row[0] for row in FILTER_SHAPES]
)
def test_dense_honours_every_filter_shape(shape: str, filters: Any) -> None:
    handle, _ctx, store = _make_handle()
    store.create_index.return_value = _index_result()
    store.search_vectors_by_model.return_value = []

    handle.search_dense("model-a", [0.5], top_k=5, filters=filters)

    conditions = _conditions(store.search_vectors_by_model.call_args.kwargs["filters"])
    assert ("page_number", "gte", 2) in conditions, f"dense dropped the {shape} filter"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("shape", "filters"), FILTER_SHAPES, ids=[row[0] for row in FILTER_SHAPES]
)
async def test_dense_async_honours_every_filter_shape(shape: str, filters: Any) -> None:
    handle, _ctx, store = _make_handle()
    store.create_index.return_value = _index_result()
    store.search_vectors_by_model_async = AsyncMock(return_value=[])

    await handle.search_dense_async("model-a", [0.5], top_k=5, filters=filters)

    kwargs = store.search_vectors_by_model_async.call_args.kwargs
    assert ("page_number", "gte", 2) in _conditions(kwargs["filters"]), (
        f"async dense dropped the {shape} filter"
    )


@pytest.mark.parametrize(
    ("shape", "filters"), FILTER_SHAPES, ids=[row[0] for row in FILTER_SHAPES]
)
def test_sparse_agrees_with_dense_on_every_shape(shape: str, filters: Any) -> None:
    handle, _ctx, store = _make_handle()
    store.open_embeddings_table.return_value = (MagicMock(), "embeddings_model_a")
    store.create_index.return_value = _index_result()
    store.build_filter_expression.return_value = None
    fts_table = store.open_embeddings_table.return_value[0]
    fts_table.search.return_value.limit.return_value.to_pandas.return_value = (
        pd.DataFrame()
    )

    handle.search_sparse("model-a", "q", top_k=5, filters=filters)

    kwargs = store.build_filter_expression.call_args.kwargs
    assert ("page_number", "gte", 2) in _conditions(kwargs["filters"]), (
        f"sparse dropped the {shape} filter"
    )


def _fallback_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "collection": ["docs", "docs", "docs"],
            "doc_id": ["d1", "d2", "d3"],
            "chunk_id": ["c1", "c2", "c3"],
            "text": ["alpha hit", "alpha hit", "alpha hit"],
            "parse_hash": ["h", "h", "h"],
            "created_at": [1, 2, 3],
            "metadata": [None, None, None],
            "page_number": [1, 2, 3],
        }
    )


def _batch_for(frame: pd.DataFrame, columns: Any = None) -> MagicMock:
    """A batch that projects like the real one: repeated labels stay repeated."""
    batch = MagicMock()
    if columns is not None:
        frame = pd.concat([frame[name] for name in columns], axis=1)
    batch.to_pandas.return_value = frame
    return batch


def _projecting_table(frame: pd.DataFrame) -> MagicMock:
    table = MagicMock()
    table.schema.names = list(frame.columns)
    table.to_batches.side_effect = lambda columns, batch_size: [
        _batch_for(frame, columns)
    ]
    return table


def test_sync_substring_fallback_applies_an_operator_filter() -> None:
    """The operator form must select rows, not compare a column to a dict."""
    handle, _ctx, _store = _make_handle()
    table = _projecting_table(_fallback_frame())
    warnings: list[Any] = []

    results = handle._substring_fallback(
        table=table,
        collection="docs",
        query_text="alpha",
        model_tag="model-a",
        top_k=10,
        filters={"page_number": {"operator": "gte", "value": 2}},
        current_warnings=warnings,
    )

    assert [r.chunk_id for r in results] == ["c2", "c3"]


def test_sync_substring_fallback_still_applies_plain_equality() -> None:
    handle, _ctx, _store = _make_handle()
    table = _projecting_table(_fallback_frame())

    results = handle._substring_fallback(
        table=table,
        collection="docs",
        query_text="alpha",
        model_tag="model-a",
        top_k=10,
        filters={"doc_id": "d2"},
        current_warnings=[],
    )

    assert [r.chunk_id for r in results] == ["c2"]


def test_sync_substring_fallback_still_applies_a_list_as_membership() -> None:
    """A list value kept its IN semantics from before the parse was introduced."""
    handle, _ctx, _store = _make_handle()
    table = _projecting_table(_fallback_frame())

    results = handle._substring_fallback(
        table=table,
        collection="docs",
        query_text="alpha",
        model_tag="model-a",
        top_k=10,
        filters={"doc_id": ["d1", "d3"]},
        current_warnings=[],
    )

    assert [r.chunk_id for r in results] == ["c1", "c3"]


@pytest.mark.asyncio
async def test_async_substring_fallback_applies_an_operator_filter() -> None:
    """The async path must not hand the operator dict to the store."""
    handle, _ctx, store = _make_handle()
    store.open_embeddings_table.return_value = (MagicMock(), "embeddings_model_a")

    captured: dict[str, Any] = {}

    async def _iter_batches_async(**kwargs: Any):
        captured.update(kwargs)
        yield _batch_for(_fallback_frame(), kwargs["columns"])

    store.iter_batches_async = _iter_batches_async

    results = await handle._substring_fallback_async(
        model_tag="model-a",
        collection="docs",
        query_text="alpha",
        top_k=10,
        filters={"page_number": {"operator": "gte", "value": 2}},
        current_warnings=[],
    )

    assert [r.chunk_id for r in results] == ["c2", "c3"]
    # The operator condition must not be pushed down as a dict value.
    assert captured["filters"] == {"collection": "docs"}
    # ...and its column must be requested, or the scan would skip it silently.
    assert "page_number" in captured["columns"]


@pytest.mark.asyncio
async def test_async_substring_fallback_still_pushes_plain_equality_down() -> None:
    handle, _ctx, store = _make_handle()
    store.open_embeddings_table.return_value = (MagicMock(), "embeddings_model_a")

    captured: dict[str, Any] = {}

    async def _iter_batches_async(**kwargs: Any):
        captured.update(kwargs)
        yield _batch_for(_fallback_frame(), kwargs["columns"])

    store.iter_batches_async = _iter_batches_async

    await handle._substring_fallback_async(
        model_tag="model-a",
        collection="docs",
        query_text="alpha",
        top_k=10,
        filters={"doc_id": "d2"},
        current_warnings=[],
    )

    assert captured["filters"] == {"collection": "docs", "doc_id": "d2"}


CREATED_AT_GTE_2 = FilterCondition(
    field="created_at", operator=FilterOperator.GTE, value=2
)


@pytest.mark.asyncio
async def test_async_fallback_filters_on_a_column_it_already_projects() -> None:
    """A filter on a base column must not request that column twice.

    Duplicate labels make ``to_pandas`` hand back a DataFrame per label, and
    every mask built from it then raises into the swallowing except -- the same
    silent zero-result this change exists to remove.
    """
    handle, _ctx, store = _make_handle()
    store.open_embeddings_table.return_value = (MagicMock(), "embeddings_model_a")

    captured: dict[str, Any] = {}

    async def _iter_batches_async(**kwargs: Any):
        captured.update(kwargs)
        yield _batch_for(_fallback_frame(), kwargs["columns"])

    store.iter_batches_async = _iter_batches_async

    results = await handle._substring_fallback_async(
        model_tag="model-a",
        collection="docs",
        query_text="alpha",
        top_k=10,
        filters={"created_at": {"operator": "gte", "value": 2}},
        current_warnings=[],
    )

    assert len(captured["columns"]) == len(set(captured["columns"]))
    assert [r.chunk_id for r in results] == ["c2", "c3"]


@pytest.mark.parametrize(
    ("shape", "filters"),
    [
        # page_number is outside the fallback's base projection, so the columns
        # must come from the parsed conditions for these shapes to apply.
        ("single_condition", PAGE_GTE_2),
        ("tuple_of_conditions", (PAGE_GTE_2,)),
    ],
    ids=["single_condition", "tuple_of_conditions"],
)
def test_sync_fallback_projects_columns_for_non_dict_shapes(
    shape: str, filters: Any
) -> None:
    """The projection must follow the parsed conditions, not just dict keys."""
    handle, _ctx, _store = _make_handle()
    table = _projecting_table(_fallback_frame())

    results = handle._substring_fallback(
        table=table,
        collection="docs",
        query_text="alpha",
        model_tag="model-a",
        top_k=10,
        filters=filters,
        current_warnings=[],
    )

    assert [r.chunk_id for r in results] == ["c2", "c3"], (
        f"the {shape} filter was dropped by the scan"
    )


def test_a_top_level_list_stays_a_disjunction() -> None:
    """``contracts.FilterExpression`` reads a list as OR; it must not flatten."""
    handle, _ctx, store = _make_handle()
    store.create_index.return_value = _index_result()
    store.search_vectors_by_model.return_value = []
    or_group = [
        FilterCondition(field="doc_id", operator=FilterOperator.EQ, value="d1"),
        FilterCondition(field="doc_id", operator=FilterOperator.EQ, value="d3"),
    ]

    handle.search_dense("model-a", [0.5], top_k=5, filters=or_group)

    passed = store.search_vectors_by_model.call_args.kwargs["filters"]
    # collection AND (d1 OR d3): the disjunction survives as one element.
    assert isinstance(passed, tuple)
    assert any(isinstance(item, list) for item in passed), (
        f"the OR group was flattened into the AND list: {passed}"
    )


def _operator_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "collection": ["docs"] * 4,
            "doc_id": ["d1", "d2", "d3", "d4"],
            "chunk_id": ["c1", "c2", "c3", "c4"],
            "text": ["alpha"] * 4,
            "parse_hash": ["h"] * 4,
            "created_at": [1, 2, 3, 4],
            "metadata": [None] * 4,
            "label": ["a", None, "b%c", "bXc"],
        }
    )


# Every operator the mask claims to evaluate, and the chunks it must select.
OPERATOR_CASES = [
    (FilterOperator.EQ, "label", "a", ["c1"]),
    # SQL drops NULL rows on `!=`; so must the scan.
    (FilterOperator.NE, "label", "a", ["c3", "c4"]),
    (FilterOperator.GT, "created_at", 3, ["c4"]),
    (FilterOperator.GTE, "created_at", 3, ["c3", "c4"]),
    (FilterOperator.LT, "created_at", 2, ["c1"]),
    (FilterOperator.LTE, "created_at", 2, ["c1", "c2"]),
    (FilterOperator.IN, "doc_id", ["d1", "d4"], ["c1", "c4"]),
    (FilterOperator.IN, "doc_id", [], []),
    # A literal substring, not a LIKE pattern and not a regex.
    (FilterOperator.CONTAINS, "label", "b%c", ["c3"]),
    # '.' is a regex metacharacter: as a pattern this would also take c4.
    (FilterOperator.CONTAINS, "label", "b.c", []),
    (FilterOperator.IS_NULL, "label", None, ["c2"]),
    (FilterOperator.IS_NOT_NULL, "label", None, ["c1", "c3", "c4"]),
]


@pytest.mark.parametrize(
    ("operator", "field", "value", "expected"),
    OPERATOR_CASES,
    ids=[f"{row[0].value}-{row[1]}-{row[2]}" for row in OPERATOR_CASES],
)
def test_scan_evaluates_every_operator_it_accepts(
    operator: FilterOperator, field: str, value: Any, expected: list[str]
) -> None:
    handle, _ctx, _store = _make_handle()
    table = _projecting_table(_operator_frame())

    results = handle._substring_fallback(
        table=table,
        collection="docs",
        query_text="alpha",
        model_tag="model-a",
        top_k=10,
        filters=FilterCondition(field=field, operator=operator, value=value),
        current_warnings=[],
    )

    assert [r.chunk_id for r in results] == expected


def test_scan_refuses_a_filter_on_a_column_the_table_lacks() -> None:
    """Dropping an unapplicable filter would return rows the caller excluded.

    The embeddings schema carries no arbitrary metadata columns, so a filter on
    one cannot be evaluated by a scan; failing beats widening the result set.
    """
    handle, _ctx, _store = _make_handle()
    table = _projecting_table(_operator_frame())

    with pytest.raises(ValueError, match="page_number"):
        handle._substring_fallback(
            table=table,
            collection="docs",
            query_text="alpha",
            model_tag="model-a",
            top_k=10,
            filters={"page_number": {"operator": "gte", "value": 2}},
            current_warnings=[],
        )


def test_a_list_value_reaches_the_backend_as_membership() -> None:
    """``{"doc_id": [...]}`` is membership on both paths.

    As equality it translated to ``doc_id == '['d1', 'd3']'`` -- not valid SQL,
    while the scan path read the same input as ``isin``.
    """
    handle, _ctx, store = _make_handle()
    store.create_index.return_value = _index_result()
    store.search_vectors_by_model.return_value = []

    handle.search_dense("model-a", [0.5], top_k=5, filters={"doc_id": ["d1", "d3"]})

    conditions = _conditions(store.search_vectors_by_model.call_args.kwargs["filters"])
    assert ("doc_id", "in", ["d1", "d3"]) in conditions, conditions


def test_contains_does_not_match_the_rendering_of_a_null() -> None:
    """`astype(str)` renders NULL as "None"/"nan"; a short needle must not hit it."""
    handle, _ctx, _store = _make_handle()
    frame = _operator_frame()
    frame["label"] = ["alpha", None, "beta", "gamma"]
    table = _projecting_table(frame)

    for needle in ("on", "an", "No", "na"):
        results = handle._substring_fallback(
            table=table,
            collection="docs",
            query_text="alpha",
            model_tag="model-a",
            top_k=10,
            filters=FilterCondition(
                field="label", operator=FilterOperator.CONTAINS, value=needle
            ),
            current_warnings=[],
        )
        assert "c2" not in [r.chunk_id for r in results], (
            f"needle {needle!r} matched the NULL row"
        )


def test_an_empty_membership_filter_stays_valid_sql() -> None:
    """`IN ()` is a syntax error; an empty membership is simply false."""
    from xagent.core.tools.core.RAG_tools.storage.lancedb_filter_utils import (
        translate_filter_expression,
    )
    from xagent.core.tools.core.RAG_tools.utils.filter_utils import (
        parse_legacy_filters,
    )

    expression = parse_legacy_filters({"doc_id": []})

    assert translate_filter_expression(expression) == "FALSE"


def test_scan_reads_ne_against_null_the_way_sql_does() -> None:
    """`field != NULL` is NULL in SQL, so it selects nothing."""
    handle, _ctx, _store = _make_handle()
    frame = _operator_frame()
    frame["label"] = ["a", None, "b", "c"]
    table = _projecting_table(frame)

    results = handle._substring_fallback(
        table=table,
        collection="docs",
        query_text="alpha",
        model_tag="model-a",
        top_k=10,
        filters=FilterCondition(field="label", operator=FilterOperator.NE, value=None),
        current_warnings=[],
    )

    assert results == []


def test_scan_names_the_column_when_a_comparison_cannot_be_made() -> None:
    """A dtype mismatch fails on the backend too; say which column and value."""
    handle, _ctx, _store = _make_handle()
    frame = _operator_frame()
    frame["created_at"] = pd.to_datetime(
        ["2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01"]
    )
    table = _projecting_table(frame)

    with pytest.raises(ValueError, match="created_at"):
        handle._substring_fallback(
            table=table,
            collection="docs",
            query_text="alpha",
            model_tag="model-a",
            top_k=10,
            filters={"created_at": {"operator": "gte", "value": 2}},
            current_warnings=[],
        )
