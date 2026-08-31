"""LanceDB filter expression utilities.

Shared functions for converting abstract filter expressions to LanceDB syntax.
"""

from typing import Any

from ..utils.string_utils import escape_lancedb_string
from .contracts import FilterCondition, FilterExpression, FilterOperator


def translate_condition(condition: FilterCondition) -> str:
    """Translate single FilterCondition to LanceDB syntax.

    Args:
        condition: FilterCondition to translate

    Returns:
        LanceDB filter string
    """
    field = condition.field
    op = condition.operator
    value = condition.value

    if op == FilterOperator.EQ:
        return f"{field} == {format_value(value)}"
    elif op == FilterOperator.NE:
        return f"{field} != {format_value(value)}"
    elif op == FilterOperator.GT:
        return f"{field} > {format_value(value)}"
    elif op == FilterOperator.GTE:
        return f"{field} >= {format_value(value)}"
    elif op == FilterOperator.LT:
        return f"{field} < {format_value(value)}"
    elif op == FilterOperator.LTE:
        return f"{field} <= {format_value(value)}"
    elif op == FilterOperator.IN:
        values = ", ".join(format_value(v) for v in value)
        return f"{field} IN ({values})"
    elif op == FilterOperator.CONTAINS:
        return f"{field} LIKE '%{escape_lancedb_string(value)}%'"
    elif op == FilterOperator.IS_NULL:
        return f"{field} IS NULL"
    elif op == FilterOperator.IS_NOT_NULL:
        return f"{field} IS NOT NULL"
    else:
        raise ValueError(f"Unsupported operator: {op}")


def format_value(value: Any) -> str:
    """Format value for LanceDB.

    Args:
        value: Value to format

    Returns:
        Formatted value string
    """
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    elif isinstance(value, (int, float)):
        return str(value)
    elif value is None:
        return "NULL"
    else:
        return f"'{escape_lancedb_string(value)}'"


def translate_filter_expression(expr: FilterExpression) -> str:
    """Translate FilterExpression to LanceDB syntax.

    Args:
        expr: FilterExpression (FilterCondition, tuple for AND, list for OR)

    Returns:
        LanceDB filter string
    """
    if isinstance(expr, FilterCondition):
        return translate_condition(expr)
    elif isinstance(expr, tuple):
        # AND combination
        return " AND ".join(f"({translate_filter_expression(e)})" for e in expr)
    elif isinstance(expr, list):
        # OR combination
        return " OR ".join(f"({translate_filter_expression(e)})" for e in expr)
    else:
        raise ValueError(f"Unsupported filter expression: {type(expr)}")
