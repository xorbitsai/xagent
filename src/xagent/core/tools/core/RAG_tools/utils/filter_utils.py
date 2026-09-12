"""Filter parsing utilities for backend-agnostic filter expressions.

This module provides utilities to convert API-facing filter dictionaries into
abstract filter expressions that can be translated to backend-specific syntax.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..storage.contracts import FilterCondition, FilterExpression, FilterOperator


def validate_filter_depth(
    expr: Optional[FilterExpression],
    max_depth: int = 10,
) -> None:
    """Validate filter expression depth to prevent DoS via deeply nested filters.

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
    }

    conditions: list[FilterCondition] = []
    for field, spec in filters.items():
        if isinstance(spec, dict) and "operator" in spec and "value" in spec:
            op_str = str(spec["operator"]).lower()
            if op_str not in op_map:
                raise ValueError(
                    f"Unknown filter operator: {op_str}. Supported operators: {sorted(op_map.keys())}"
                )
            conditions.append(
                FilterCondition(
                    field=field, operator=op_map[op_str], value=spec["value"]
                )
            )
        elif isinstance(spec, (list, tuple, set)):
            # Membership, not equality: EQ would translate to `field == '[...]'`.
            conditions.append(
                FilterCondition(
                    field=field, operator=FilterOperator.IN, value=list(spec)
                )
            )
        else:
            conditions.append(
                FilterCondition(field=field, operator=FilterOperator.EQ, value=spec)
            )

    if len(conditions) == 1:
        return conditions[0]
    return tuple(conditions)


def normalize_filter_conditions(
    filters: Any,
    max_depth: int = 10,
) -> list[FilterExpression]:
    """Return the conditions to AND together for any accepted filter shape.

    Accepts the legacy dict form, an already-built ``FilterExpression``, or a
    tuple/list of them. Search engines differed on which shapes they honoured
    (#671); routing every engine through this keeps one answer per input.

    A tuple is an AND group, so it flattens into the returned list; a list is an
    OR group (``contracts.FilterExpression``), so it stays one element and keeps
    its disjunction.
    """
    if not filters:
        return []

    if isinstance(filters, dict):
        parsed = parse_legacy_filters(filters, max_depth=max_depth)
        if parsed is None:
            return []
        return list(parsed) if isinstance(parsed, tuple) else [parsed]

    if isinstance(filters, tuple):
        return list(filters)

    return [filters]
