"""Safe construction of LanceDB (DataFusion SQL) filter expressions.

LanceDB ``where``/``delete``/``count_rows``/``update`` predicates are plain SQL
strings parsed by DataFusion. Any caller-influenced value interpolated into one
must go through :func:`sql_string_literal`, so a value such as ``x' OR '1'='1``
stays a single string literal instead of rewriting the predicate.

DataFusion follows standard SQL string literals: a single quote is escaped by
doubling it, and a backslash is an ordinary character (no C-style escapes), so
doubling single quotes is the complete escape and the literal round-trips the
original value exactly.
"""

from __future__ import annotations

from typing import Iterable, Optional


def sql_string_literal(value: str) -> str:
    """Quote ``value`` as a DataFusion SQL string literal.

    Raises:
        TypeError: if ``value`` is not a ``str``. Callers must never rely on an
            implicit ``str()`` of an arbitrary object inside a predicate.
    """
    if not isinstance(value, str):
        raise TypeError(f"SQL string literal must be a str, got {type(value)!r}")
    return "'" + value.replace("'", "''") + "'"


def sql_string_in(column: str, values: Iterable[str]) -> str:
    """``column IN ('a', 'b', ...)`` with every value safely quoted.

    ``column`` is a trusted identifier supplied by code, never by a caller.
    An empty ``values`` yields ``false`` so the predicate matches nothing.
    """
    literals = [sql_string_literal(value) for value in values]
    if not literals:
        return "false"
    return f"{column} IN ({', '.join(literals)})"


def sql_and(terms: Iterable[Optional[str]]) -> Optional[str]:
    """AND together predicate terms, each wrapped in parentheses.

    Parenthesising every term keeps one term's internal ``OR`` from escaping
    and weakening another term (e.g. the owner condition). Empty/``None``
    terms are skipped; a single term is returned unchanged; no terms yields
    ``None``.
    """
    present = [term for term in terms if term]
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    return " AND ".join(f"({term})" for term in present)


__all__ = ["sql_and", "sql_string_in", "sql_string_literal"]
