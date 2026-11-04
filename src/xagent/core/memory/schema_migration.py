"""Safe schema-migration primitives for the LanceDB memory store.

This module is the shared foundation both memory-store fix sites resolve through
(the ``add()`` write path and the ``_ensure_table_schema()`` init path). It
provides two things and no call-site wiring:

- :func:`migrate_table_swap` - a transform-then-swap migration primitive that
  fully materializes the migrated table before touching the live one, and never
  drops or empties a table as a fallback. On any failure the original table is
  left intact and the error propagates.
- :func:`classify_memory_schema_mismatch` - a mismatch classifier that reports
  which class of schema mismatch a table has against the fixed memory schema
  (``id`` / ``text`` / ``metadata`` / ``vector``) and the store's current
  embedding dimension.

The concrete transforms (batched re-embed rebuild, ``add_columns`` backfill) are
introduced by the consuming slices; this module only supplies the swap mechanic
and the classification.

Modeled on ``RAG_tools/LanceDB/schema_manager.py::_migrate_table_user_id_to_int64``,
which materializes converted data before swapping and raises rather than
discarding data on failure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

import pyarrow as pa  # type: ignore
from lancedb.db import DBConnection

from ..tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table

logger = logging.getLogger(__name__)

__all__ = [
    "MEMORY_NON_VECTOR_COLUMNS",
    "MEMORY_VECTOR_COLUMN",
    "MemoryMismatchKind",
    "MemorySchemaMismatch",
    "classify_memory_schema_mismatch",
    "migrate_table_swap",
]

# The memory schema is fixed: the only variable is the vector column's width.
MEMORY_NON_VECTOR_COLUMNS = ("id", "text", "metadata")
MEMORY_VECTOR_COLUMN = "vector"


class MemoryMismatchKind(str, Enum):
    """Class of schema mismatch, which determines how it is resolved."""

    NONE = "none"
    # A required non-vector column (id / text / metadata) is missing; resolvable
    # by an additive ``add_columns`` backfill.
    MISSING_NON_VECTOR_COLUMN = "missing_non_vector_column"
    # The vector column's width changed, or its presence changed (table has no
    # vector but records now do, or vice versa); requires a vector-column rebuild.
    VECTOR_REBUILD = "vector_rebuild"


@dataclass(frozen=True)
class MemorySchemaMismatch:
    """Result of classifying a memory table's schema against the current store state."""

    kind: MemoryMismatchKind
    # Missing non-vector columns, in fixed order (subset of MEMORY_NON_VECTOR_COLUMNS).
    missing_columns: tuple[str, ...] = ()
    # The table's current fixed vector width, or None if it has no fixed-width
    # vector column. Retained as the only test-observable output of the private
    # _vector_width helper.
    current_dim: Optional[int] = None


def _vector_width(schema: Any) -> Optional[int]:
    """Return the fixed width of the vector column, or None if it has none."""
    if MEMORY_VECTOR_COLUMN not in schema.names:
        return None
    try:
        vector_type = schema.field(MEMORY_VECTOR_COLUMN).type
    except Exception:
        return None
    if pa.types.is_fixed_size_list(vector_type):
        return int(vector_type.list_size)
    # A variable-length list has no fixed width to compare against.
    return None


def classify_memory_schema_mismatch(
    schema: Any, expected_dim: Optional[int]
) -> MemorySchemaMismatch:
    """Classify a memory table's schema against the store's current expectation.

    Args:
        schema: The existing table's Arrow schema (``table.schema``).
        expected_dim: The vector dimension the store now produces, or ``None``
            when no embedding model is available (no vector column expected).

    Returns:
        A :class:`MemorySchemaMismatch`. Precedence: a vector rebuild subsumes a
        missing-column backfill (the rebuild produces a fresh, complete schema),
        so ``VECTOR_REBUILD`` wins when both apply.
    """
    names = set(schema.names)
    has_vector = MEMORY_VECTOR_COLUMN in names
    current_dim = _vector_width(schema)
    missing_columns = tuple(c for c in MEMORY_NON_VECTOR_COLUMNS if c not in names)

    if expected_dim is None:
        # Store no longer produces vectors: only a mismatch if the table still
        # carries a vector column (presence change -> drop-vector rebuild).
        vector_mismatch = has_vector
    else:
        # Store produces vectors of expected_dim: mismatch if the table has no
        # vector column or its width differs.
        vector_mismatch = (not has_vector) or (current_dim != expected_dim)

    if vector_mismatch:
        kind = MemoryMismatchKind.VECTOR_REBUILD
    elif missing_columns:
        kind = MemoryMismatchKind.MISSING_NON_VECTOR_COLUMN
    else:
        kind = MemoryMismatchKind.NONE

    return MemorySchemaMismatch(
        kind=kind,
        missing_columns=missing_columns,
        current_dim=current_dim,
    )


def migrate_table_swap(
    conn: DBConnection,
    table_name: str,
    transform: Callable[[Any], Any],
) -> None:
    """Migrate a table by transform-then-swap, never dropping data on failure.

    Reads the table's full contents, hands them to ``transform`` to produce the
    migrated Arrow table (new schema and data), and only then replaces the live
    table. Because the migrated data is fully materialized before the live table
    is touched, any failure in reading or transforming leaves the original table
    byte-for-byte intact and propagates the error. This function never drops or
    empties a table as a fallback.

    Concurrency: the read snapshot (``to_arrow``) and the final overwrite are not
    covered by a shared lock or transaction, and this store has no locking
    elsewhere. A concurrent write that commits to the live table between the two
    would be replaced by the overwrite (computed from the earlier snapshot) and
    lost. The "never lose data" guarantee here is against migration *failure*,
    not against concurrent writers during a migration.

    Args:
        conn: A LanceDB connection (must expose ``open_table`` and ``create_table``).
        table_name: Name of the table to migrate.
        transform: Callable that receives the current table as a
            :class:`pyarrow.Table` and returns the fully-migrated table. It must
            materialize its result (not a lazy view) and may raise to abort the
            migration with no side effects.

    Raises:
        TypeError: If ``transform`` does not return a :class:`pyarrow.Table`.
        Exception: Any error raised while reading or transforming is propagated
            unchanged; the original table is left intact.
    """
    table = conn.open_table(table_name)
    try:
        existing = table.to_arrow()
    finally:
        _safe_close_table(table)

    # Fully materialize the migrated data BEFORE the live table is touched, so a
    # transform failure cannot leave the store in a partial state.
    migrated = transform(existing)
    if not isinstance(migrated, pa.Table):
        raise TypeError(
            f"migration transform must return a pyarrow.Table, got {type(migrated)!r}"
        )

    # Replace the table atomically. mode="overwrite" commits a new table version
    # in a single transaction: if it fails, the previous version and its rows
    # remain intact. We never fall back to drop_table on error.
    new_table = conn.create_table(table_name, data=migrated, mode="overwrite")
    _safe_close_table(new_table)
    logger.info(
        "Migrated memory table '%s' via transform-then-swap (%d rows)",
        table_name,
        migrated.num_rows,
    )
