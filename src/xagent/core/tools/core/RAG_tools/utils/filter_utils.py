"""Filter parsing utilities for backend-agnostic filter expressions.

This module provides utilities to convert API-facing filter dictionaries into
abstract filter expressions that can be translated to backend-specific syntax.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..storage.contracts import (
    FilterCondition,
    FilterExpression,
    FilterInput,
    FilterOperator,
    validate_field_name,
    validate_filter_value,
)

_DEFAULT_MAX_FILTER_NODES = 1024
_DEFAULT_MAX_FILTER_CHILDREN = 128
_DEFAULT_MAX_FILTER_VALUES = 1024
_DEFAULT_MAX_FILTER_VALUE_BYTES = 64 * 1024


def validate_filter_depth(
    expr: Optional[FilterExpression],
    max_depth: int = 10,
) -> None:
    """Validate filter expression depth to prevent deeply nested filters.

    This should be called on user-provided filter expressions before they
    are passed to build_filter_expression.

    Args:
        expr: Filter expression to validate.
        max_depth: Maximum allowed nesting depth (default: 10).

    Raises:
        ValueError: If filter expression exceeds max_depth.
    """
    if expr is None:
        return

    def _check_depth(e: FilterExpression, depth: int = 0) -> None:
        if depth > max_depth:
            raise ValueError(
                f"Filter expression depth exceeds maximum allowed depth of {max_depth}. "
                "This may indicate a malicious or malformed filter expression."
            )
        if isinstance(e, FilterCondition):
            return
        elif isinstance(e, tuple):
            for item in e:
                _check_depth(item, depth + 1)
        elif isinstance(e, list):
            for item in e:
                _check_depth(item, depth + 1)
        else:
            raise TypeError(f"Unsupported filter expression: {type(e).__name__}")

    _check_depth(expr)


def parse_legacy_filters(
    filters: Optional[Dict[str, Any]],
    max_depth: int = 10,
) -> Optional[FilterExpression]:
    """Convert Dict-based filters to an abstract FilterExpression.

    Supported input formats:
    - Simple equality:
      {"field": "value"}
    - Operator form:
      {"field": {"operator": "gte", "value": 5}}
    Multiple fields are combined as an AND expression (tuple convention).

    Args:
        filters: Filter dictionary from API layer.
        max_depth: Maximum allowed nesting depth (default: 10).

    Returns:
        Parsed FilterExpression, or None if filters is None/empty.

    Raises:
        ValueError: If an unsupported operator is provided or depth exceeds max_depth.
    """
    if not filters:
        return None

    op_map: Dict[str, FilterOperator] = {
        "eq": FilterOperator.EQ,
        "ne": FilterOperator.NE,
        "gt": FilterOperator.GT,
        "gte": FilterOperator.GTE,
        "lt": FilterOperator.LT,
        "lte": FilterOperator.LTE,
        "in": FilterOperator.IN,
        "contains": FilterOperator.CONTAINS,
        "is_null": FilterOperator.IS_NULL,
        "is_not_null": FilterOperator.IS_NOT_NULL,
    }

    conditions: list[FilterCondition] = []
    for field, spec in filters.items():
        if isinstance(spec, dict) and "operator" in spec:
            op_str = str(spec["operator"]).lower()
            if op_str not in op_map:
                raise ValueError(
                    f"Unknown filter operator: {op_str}. Supported operators: {sorted(op_map.keys())}"
                )
            operator = op_map[op_str]
            if operator in {FilterOperator.IS_NULL, FilterOperator.IS_NOT_NULL}:
                value = None
            elif "value" not in spec:
                raise ValueError(f"Filter operator '{op_str}' requires a value")
            else:
                value = spec["value"]
            conditions.append(
                FilterCondition(field=field, operator=operator, value=value)
            )
        elif isinstance(spec, (list, tuple, set)):
            conditions.append(
                FilterCondition(field=field, operator=FilterOperator.IN, value=spec)
            )
        else:
            conditions.append(
                FilterCondition(field=field, operator=FilterOperator.EQ, value=spec)
            )

    if len(conditions) == 1:
        parsed: FilterExpression = conditions[0]
    else:
        parsed = tuple(conditions)
    validate_filter_depth(parsed, max_depth=max_depth)
    return parsed


def normalize_filter_input(
    filters: Optional[FilterInput],
    max_depth: int = 10,
    max_nodes: int = _DEFAULT_MAX_FILTER_NODES,
    max_children: int = _DEFAULT_MAX_FILTER_CHILDREN,
    max_values: int = _DEFAULT_MAX_FILTER_VALUES,
    max_value_bytes: int = _DEFAULT_MAX_FILTER_VALUE_BYTES,
) -> Optional[FilterExpression]:
    """Normalize API-facing filters without changing boolean composition.

    Legacy dictionaries are parsed into ``FilterCondition`` objects. Existing
    expressions are copied through one bounded traversal so tuple/AND and list/OR
    semantics remain intact and malformed or oversized expressions fail before I/O.
    """
    if filters is None or filters == {}:
        return None

    node_count = 0

    def _validate_value_size(value: Any) -> int:
        if isinstance(value, (list, tuple, set)):
            if len(value) > max_values:
                raise ValueError(
                    "Filter value collection exceeds maximum allowed size of "
                    f"{max_values}"
                )
            size = 0
            for item in value:
                size += _validate_value_size(item)
                if size > max_value_bytes:
                    break
        elif isinstance(value, str):
            size = len(value.encode("utf-8"))
        else:
            size = len(str(value).encode("utf-8"))

        if size > max_value_bytes:
            raise ValueError(
                "Filter value exceeds maximum serialized size of "
                f"{max_value_bytes} bytes"
            )
        return size

    def _normalize(expr: FilterExpression, depth: int = 0) -> FilterExpression:
        nonlocal node_count
        if depth > max_depth:
            raise ValueError(
                f"Filter expression depth exceeds maximum allowed depth of {max_depth}. "
                "This may indicate a malicious or malformed filter expression."
            )

        node_count += 1
        if node_count > max_nodes:
            raise ValueError(
                "Filter expression contains too many nodes. "
                f"The maximum allowed is {max_nodes}."
            )

        if isinstance(expr, FilterCondition):
            if not isinstance(expr.operator, FilterOperator):
                raise TypeError(
                    "FilterCondition.operator must be a FilterOperator instance"
                )
            if not isinstance(expr.field, str):
                raise TypeError("FilterCondition.field must be a string")
            validate_field_name(expr.field)
            validate_filter_value(expr.value)
            _validate_value_size(expr.value)
            return expr
        if isinstance(expr, tuple):
            if not expr:
                raise ValueError("AND filter expression cannot be empty")
            if len(expr) > max_children:
                raise ValueError(
                    "AND filter expression contains too many children. "
                    f"The maximum allowed is {max_children}."
                )
            return tuple(_normalize(item, depth + 1) for item in expr)
        if isinstance(expr, list):
            if not expr:
                raise ValueError("OR filter expression cannot be empty")
            if len(expr) > max_children:
                raise ValueError(
                    "OR filter expression contains too many children. "
                    f"The maximum allowed is {max_children}."
                )
            return [_normalize(item, depth + 1) for item in expr]
        raise TypeError(f"Unsupported filter expression: {type(expr).__name__}")

    normalized = (
        parse_legacy_filters(filters, max_depth=max_depth)
        if isinstance(filters, dict)
        else _normalize(filters)
    )
    return _normalize(normalized) if normalized is not None else None


def combine_filter_expressions(
    *expressions: Optional[FilterExpression],
) -> Optional[FilterExpression]:
    """Combine complete filter expressions with AND without flattening them."""
    present = tuple(expression for expression in expressions if expression is not None)
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    return present
