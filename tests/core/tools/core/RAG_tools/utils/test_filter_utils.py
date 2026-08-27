"""Tests for backend-agnostic filter parsing and normalization."""

import pytest

from xagent.core.tools.core.RAG_tools.storage.contracts import (
    FilterCondition,
    FilterOperator,
)
from xagent.core.tools.core.RAG_tools.utils.filter_utils import (
    combine_filter_expressions,
    normalize_filter_input,
    parse_legacy_filters,
)


def test_normalize_filter_input_preserves_nested_boolean_semantics() -> None:
    collection_filter = FilterCondition(
        field="collection", operator=FilterOperator.EQ, value="docs"
    )
    page_filter = [
        FilterCondition(field="page_number", operator=FilterOperator.EQ, value=1),
        FilterCondition(field="page_number", operator=FilterOperator.EQ, value=2),
    ]

    normalized = normalize_filter_input(page_filter)
    combined = combine_filter_expressions(collection_filter, normalized)

    assert isinstance(combined, tuple)
    assert combined[0] == collection_filter
    assert isinstance(combined[1], list)
    assert combined[1] == page_filter


@pytest.mark.parametrize(
    ("operator", "expected"),
    [
        ("is_null", FilterOperator.IS_NULL),
        ("is_not_null", FilterOperator.IS_NOT_NULL),
    ],
)
def test_parse_legacy_filters_supports_unary_null_operators(
    operator: str,
    expected: FilterOperator,
) -> None:
    parsed = parse_legacy_filters({"metadata": {"operator": operator}})

    assert isinstance(parsed, FilterCondition)
    assert parsed.operator == expected
    assert parsed.value is None


@pytest.mark.parametrize("filters", [[], ()])
def test_normalize_filter_input_rejects_empty_expressions(filters: object) -> None:
    with pytest.raises(ValueError, match="filter expression cannot be empty"):
        normalize_filter_input(filters)  # type: ignore[arg-type]
