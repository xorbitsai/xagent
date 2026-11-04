"""LanceDB query utilities for RAG tools.

This module provides unified query functions for LanceDB operations,
implementing a three-tier fallback pattern for maximum compatibility.
"""

import logging
import re
from collections.abc import Callable, Iterable
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
import pyarrow as pa  # type: ignore
from lancedb.query import BooleanQuery, MatchQuery, Occur

logger = logging.getLogger(__name__)


def _safe_count_rows(
    table: Any,
    filter_expr: Optional[str] = None,
    *,
    on_error: Literal["zero", "raise"] = "zero",
) -> int:
    """Count rows safely, with selectable behavior when ``count_rows`` fails.

    Args:
        table: LanceDB table or query object that exposes ``count_rows``.
        filter_expr: Optional LanceDB filter expression.
        on_error: If ``\"zero\"`` (default), log at debug and return ``0`` (best-effort
            planning or non-security paths). If ``\"raise\"``, log at error and re-raise
            the exception (e.g. permission checks where treating failure as zero is unsafe).

    Returns:
        Row count as int. Returns ``0`` on failure only when ``on_error=\"zero\"``.

    Raises:
        Exception: Re-raises the underlying error when ``on_error=\"raise\"``.
    """
    try:
        if filter_expr is None:
            return int(table.count_rows())
        return int(table.count_rows(filter_expr))
    except Exception as exc:
        if on_error == "raise":
            logger.error("count_rows failed: %s", exc, exc_info=True)
            raise
        logger.debug("count_rows failed, fallback to 0: %s", exc)
        return 0


def query_to_list(
    query: Any,
    normalize_nan: bool = True,
) -> List[Dict[str, Any]]:
    """Convert LanceDB query result to List[Dict] with three-tier fallback.

    This function implements a performance-optimized fallback chain:
    1. to_arrow() (fastest, native format)
    2. to_list() (fast, direct List[Dict])
    3. to_pandas() (compatibility fallback)

    Args:
        query: LanceDB query object that supports to_arrow/to_list/to_pandas methods.
            Can be:
            - table.search().where(filter_expr)
            - search_query (pre-built query object)
            - table.head(n)
            - table.search().where().select()
        normalize_nan: Whether to convert pandas NaN values to None for consistent handling.

    Returns:
        List[Dict[str, Any]]: Query results as a list of dictionaries.
            Empty list indicates no results found.

    Example:
        >>> from ..utils.lancedb_query_utils import query_to_list
        >>> results = query_to_list(table.search().where(filter_expr))
        >>> if results:
        ...     record = results[0]
    """
    results: List[Dict[str, Any]] = []

    # Check if query is already a PyArrow Table (e.g., from table.head())
    if pa is not None and isinstance(query, pa.Table):
        # PyArrow Table can directly convert to list
        results = query.to_pylist()
        return results

    try:
        # First choice: Arrow (fastest, native format)
        arrow_table = query.to_arrow()
        results = arrow_table.to_pylist()
    except Exception as e:  # noqa: BLE001 - Catch all exceptions to ensure fallback works
        logger.debug("to_arrow() failed (will try fallback): %s", e)
        try:
            # Second choice: to_list() (fast, direct List[Dict])
            results = query.to_list()
        except Exception as e:  # noqa: BLE001 - Catch all exceptions to ensure fallback works
            # Last resort: pandas (compatibility fallback)
            logger.debug(
                "to_list() failed (will try pandas fallback): %s. Falling back to to_pandas()",
                e,
            )
            df = query.to_pandas()
            results = df.to_dict("records")

            # Normalize NaN to None for consistent handling (pandas may return NaN)
            # Only check scalar values to avoid errors with array/list fields (e.g., vector)
            if normalize_nan:
                for row in results:
                    for key, value in row.items():
                        # Only check NaN for scalar values, skip arrays/lists
                        if pd.api.types.is_scalar(value) and pd.isna(value):
                            row[key] = None

    return results


def list_table_names(conn: Any) -> list[str]:
    """List LanceDB table names with compatibility across API versions.

    LanceDB has exposed both `list_tables()` and `table_names()` across versions,
    and `list_tables()` may return either an iterable of names or an object with
    a `.tables` attribute.
    """
    list_tables_fn: Callable[[], Any] | None = getattr(conn, "list_tables", None)
    if list_tables_fn is None:
        list_tables_fn = getattr(conn, "table_names", None)
    if list_tables_fn is None:
        return []

    tables_res = list_tables_fn()
    if hasattr(tables_res, "tables"):
        raw = tables_res.tables
    else:
        raw = tables_res

    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, Iterable):
        return [str(t) for t in list(raw)]
    return []


def list_embeddings_table_names(conn: Any, prefix: str = "embeddings_") -> list[str]:
    """List embeddings table names (default prefix: `embeddings_`)."""
    return [name for name in list_table_names(conn) if str(name).startswith(prefix)]


# Whitespace and CJK punctuation: indexed tokens of their own, never part of one.
_FTS_SEPARATORS = r"\s，。！？；：、（）【】「」『』《》〈〉“”‘’—…～"
_FTS_TERM_CHUNKS = re.compile(rf"[^{_FTS_SEPARATORS}]+")
# ``+`` and ``#`` survive at a term edge so C++ and C# keep matching.
_FTS_TERM_EDGES = re.compile(r"^[^\w+#]+|[^\w+#]+$")
# Guards against a pasted document arriving as one query.
_FTS_MAX_TERMS = 64
# Tantivy strips these at index time, so a clause holding one matches nothing and
# lance fails the whole search on the empty token list.
_FTS_STOP_WORDS = frozenset(
    "a an and are as at be but by for if in into is it no not of on or such that"
    " the their then there these they this to was will with".split()
)


def build_fts_query(
    query_text: str, text_column: str = "text"
) -> Optional[BooleanQuery]:
    """Build an OR-of-terms FTS query, or ``None`` when the text has no terms.

    Separators and edge ASCII punctuation are indexed tokens of their own (lance
    folds ``，`` onto ASCII ``,``), so a term carrying them matches every chunk
    that contains one. Punctuation inside a term stays: ``COVID-19`` and ``3.5``
    are single tokens in the index. English stop words are dropped: tantivy
    removes them at index time, so a clause holding one matches nothing and
    makes lance fail the entire search. That removes the most common trigger,
    not the class -- any clause whose term tokenizes to nothing fails the same
    way, for instance a term longer than lance's ``max_token_length``.
    """
    terms: Dict[str, str] = {}
    truncated = False
    for chunk in _FTS_TERM_CHUNKS.finditer(query_text):
        term = _FTS_TERM_EDGES.sub("", chunk.group())
        folded = term.casefold()
        if not re.search(r"\w", term) or folded in terms or folded in _FTS_STOP_WORDS:
            continue
        if len(terms) == _FTS_MAX_TERMS:
            truncated = True
            break
        terms[folded] = term

    if truncated:
        logger.warning("FTS query truncated to %d terms", _FTS_MAX_TERMS)
    if not terms:
        return None
    return BooleanQuery(
        [(Occur.SHOULD, MatchQuery(term, text_column)) for term in terms.values()]
    )
