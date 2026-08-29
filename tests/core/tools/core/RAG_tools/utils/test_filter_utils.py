"""Tests for backend-agnostic filter parsing and normalization."""

import pytest

from xagent.core.tools.core.RAG_tools.storage.contracts import (
    FilterCondition,
    FilterOperator,
    build_filter_from_dict,
)
from xagent.core.tools.core.RAG_tools.storage.lancedb_filter_utils import (
    translate_condition,
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
    document_filter = [
        FilterCondition(field="doc_id", operator=FilterOperator.EQ, value="d1"),
        FilterCondition(field="doc_id", operator=FilterOperator.EQ, value="d2"),
    ]

    normalized = normalize_filter_input(document_filter)
    combined = combine_filter_expressions(collection_filter, normalized)

    assert isinstance(combined, tuple)
    assert combined[0] == collection_filter
    assert isinstance(combined[1], list)
    assert combined[1] == document_filter


@pytest.mark.parametrize("value", [["d1", "d2"], ("d1", "d2"), {"d1", "d2"}])
def test_parse_legacy_filters_preserves_sequence_membership(value: object) -> None:
    parsed = parse_legacy_filters({"doc_id": value})

    assert isinstance(parsed, FilterCondition)
    assert parsed.operator == FilterOperator.IN
    assert parsed.value == value


def test_build_filter_from_dict_preserves_sequence_membership() -> None:
    parsed = build_filter_from_dict({"doc_id": ["d1", "d2"]})

    assert isinstance(parsed, FilterCondition)
    assert parsed.operator == FilterOperator.IN


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


def test_filter_condition_rejects_empty_in() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        FilterCondition(field="doc_id", operator=FilterOperator.IN, value=[])


def test_normalize_filter_input_rejects_empty_legacy_in() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        normalize_filter_input({"doc_id": {"operator": "in", "value": []}})


def test_normalize_filter_input_validates_prebuilt_conditions() -> None:
    condition = FilterCondition(
        field="not_a_real_field", operator=FilterOperator.EQ, value="value"
    )

    with pytest.raises(ValueError, match="Invalid filter field"):
        normalize_filter_input(condition)


def test_normalize_filter_input_rejects_oversized_boolean_group() -> None:
    expression = [
        FilterCondition(field="doc_id", operator=FilterOperator.EQ, value=str(index))
        for index in range(129)
    ]

    with pytest.raises(ValueError, match="too many children"):
        normalize_filter_input(expression)


def test_normalize_filter_input_rejects_oversized_value_collection() -> None:
    condition = FilterCondition(
        field="doc_id",
        operator=FilterOperator.IN,
        value=[str(index) for index in range(1025)],
    )

    with pytest.raises(ValueError, match="collection exceeds"):
        normalize_filter_input(condition)


def test_normalize_filter_input_rejects_oversized_serialized_value() -> None:
    condition = FilterCondition(
        field="text",
        operator=FilterOperator.CONTAINS,
        value="x" * (64 * 1024 + 1),
    )

    with pytest.raises(ValueError, match="serialized size"):
        normalize_filter_input(condition)


def test_normalize_filter_input_rejects_depth_before_python_recursion_limit() -> None:
    expression: object = FilterCondition(
        field="doc_id", operator=FilterOperator.EQ, value="d1"
    )
    for _ in range(2000):
        expression = [expression]

    with pytest.raises(ValueError, match="depth exceeds"):
        normalize_filter_input(expression)  # type: ignore[arg-type]


def test_contains_translation_uses_literal_substring_function() -> None:
    condition = FilterCondition(
        field="text", operator=FilterOperator.CONTAINS, value="100%_real"
    )

    assert translate_condition(condition) == "strpos(text, '100%_real') > 0"
