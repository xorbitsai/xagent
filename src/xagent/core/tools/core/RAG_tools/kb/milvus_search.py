"""Milvus search pieces that need no server: caller filters as an expression,
route scores on LanceDB's scale, the substring fallback's pattern and result rows."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, cast

from ..core.exceptions import DocumentValidationError
from ..core.schemas import SearchResult
from ..storage.contracts import FilterCondition, FilterOperator
from ..utils.filter_utils import parse_legacy_filters

SEARCH_FIELDS = ["doc_id", "chunk_id", "text", "parse_hash", "created_at", "metadata"]
_TEXT_COLUMNS = {"doc_id", "chunk_id", "parse_hash", "text"}
_METADATA_PATH = re.compile(r"metadata(\.[\w-]+)+")
_LIKE_SPECIAL = re.compile(r"[%_\\]")
_COMPARISONS = {
    FilterOperator.EQ: "==",
    FilterOperator.NE: "!=",
    FilterOperator.GT: ">",
    FilterOperator.GTE: ">=",
    FilterOperator.LT: "<",
    FilterOperator.LTE: "<=",
}


def _clause(condition: FilterCondition, params: dict[str, Any]) -> str:
    field, operator, value = condition.field, condition.operator, condition.value
    if field in _TEXT_COLUMNS:
        target = field
    elif _METADATA_PATH.fullmatch(field):
        target = "metadata" + "".join(f'["{key}"]' for key in field.split(".")[1:])
    else:
        raise DocumentValidationError(
            f"Milvus search cannot filter on {field!r}; "
            f"use {sorted(_TEXT_COLUMNS)} or metadata.<key>"
        )
    values = list(value) if operator is FilterOperator.IN else [value]
    if not all(isinstance(item, (str, int, float)) for item in values):
        raise DocumentValidationError(
            f"Milvus search cannot filter {field!r} by {value!r}"
        )
    name = f"p{len(params)}"
    if operator is FilterOperator.IN:
        params[name] = values
        return f"{target} in {{{name}}}"
    if operator is FilterOperator.CONTAINS:
        if not isinstance(value, str) or _LIKE_SPECIAL.search(value):
            raise DocumentValidationError(
                f"Milvus search cannot filter {field!r} by contains {value!r}: "
                "the value must be text without %, _ or a backslash"
            )
        params[name] = f"%{value}%"
        return f"{target} like {{{name}}}"
    params[name] = value
    return f"{target} {_COMPARISONS[operator]} {{{name}}}"


def caller_filter(filters: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
    """Translate the public search filters to a Milvus expression and its parameters.

    A field, operator or value that cannot be translated raises
    ``DocumentValidationError``: ignoring it would widen the results.
    """
    if not filters:
        return "", {}
    if not isinstance(filters, dict):
        raise DocumentValidationError("Milvus search filters must be a dict")
    try:
        parsed = parse_legacy_filters(filters)
    except ValueError as error:
        raise DocumentValidationError(str(error)) from error
    conditions = cast(
        "tuple[FilterCondition, ...]",
        parsed if isinstance(parsed, tuple) else (parsed,),
    )
    params: dict[str, Any] = {}
    clauses = [_clause(condition, params) for condition in conditions]
    return " and ".join(f"({clause})" for clause in clauses), params


def dense_score(cosine: float) -> float:
    """LanceDB scores unit vectors 1/(1+d) with d = 2-2cos; Milvus COSINE returns cos."""
    return 1.0 / (3.0 - 2.0 * min(max(cosine, -1.0), 1.0))


def keyword_score(raw: float) -> float:
    """Squash a BM25 score the way LanceDB squashes its FTS score."""
    raw = max(raw, 0.0)
    return raw / (1.0 + raw)


def like_pattern(term: str) -> str:
    """Return a pattern matching every text with ``term``, maybe more: ``%``, ``_``
    and a backslash become ``_``, which Milvus can parse. Callers recheck."""
    return "%" + _LIKE_SPECIAL.sub("_", term) + "%"


def to_result(entity: dict[str, Any], score: float, model_tag: str) -> SearchResult:
    created_at = datetime.fromtimestamp(entity["created_at"], timezone.utc)
    return SearchResult(
        doc_id=entity["doc_id"],
        chunk_id=entity["chunk_id"],
        text=entity["text"],
        score=score,
        parse_hash=entity["parse_hash"],
        model_tag=model_tag,
        created_at=created_at.replace(tzinfo=None),
        metadata=entity["metadata"],
    )
