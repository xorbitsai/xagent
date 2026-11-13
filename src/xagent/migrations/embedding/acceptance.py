"""Per-row acceptance before an embedding-model switch.

The check is set equality on what search can reach, per `(doc_id, chunk_id)`
row: every row the source model's table holds for a collection must have a
target row with the same owner, and none that the source does not have.
Ownerless and textless rows are excluded on both sides — the migration cannot
copy them, so rollback and switch agree — and the counts are reported. A
textless row that has a vector is still reachable by dense search; excluding it
is a stated coverage limit, not a claim that nothing searches it.
Retrieval-quality metrics are a separate phase and replace this module's
checks through the same callable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from xagent.migrations.embedding.embed import _validate_vector
from xagent.migrations.embedding.scope import (
    MigrationCliError,
    collection_filter,
    connection,
    iter_rows,
    schema_names,
    source_table,
)

RowKey = Tuple[str, str]


@dataclass
class AcceptanceResult:
    ok: bool
    checks: Dict[str, Any] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)


@dataclass
class _Rows:
    """What one model's table holds, split into comparable and excluded rows."""

    owners: Dict[RowKey, Optional[int]]
    excluded: Dict[str, int]
    bad_vectors: int = 0


def _read(
    collection: str, model_id: str, conn: Any, *, source: bool
) -> Optional[_Rows]:
    """The model's rows, or ``None`` when it has no table.

    Both sides read the one table the write path creates for the model, which
    is also where eligibility required the source to be.
    """
    from xagent.migrations.embedding.scope import table_names

    table = source_table(model_id)
    if table not in table_names(conn):
        return None
    if "text" not in schema_names(conn, table):
        raise MigrationCliError(f"embeddings table '{table}' has no text column")
    rows = _Rows(owners={}, excluded={"no-owner": 0, "no-text": 0})
    dimension: Optional[int] = None
    for row in iter_rows(
        conn,
        table,
        collection_filter(collection),
        ["doc_id", "chunk_id", "user_id", "text", "vector"],
    ):
        doc_id = str(row.get("doc_id") or "")
        chunk_id = str(row.get("chunk_id") or "")
        if not doc_id or not chunk_id:
            continue
        if row.get("user_id") is None:
            rows.excluded["no-owner"] += 1
            continue
        if not row.get("text"):
            rows.excluded["no-text"] += 1
            continue
        vector = row.get("vector")
        try:
            _validate_vector(vector, dimension)
            dimension = len(vector)
        except ValueError:
            rows.bad_vectors += 1
        rows.owners[(doc_id, chunk_id)] = row.get("user_id")
    return rows


def check_collection(
    collection: str,
    *,
    source_model_id: str,
    target_model_id: str,
    conn: Optional[Any] = None,
) -> AcceptanceResult:
    """Compare the source and target models' rows for one collection."""
    conn = conn if conn is not None else connection()
    source = _read(collection, source_model_id, conn, source=True)
    target = _read(collection, target_model_id, conn, source=False)
    result = AcceptanceResult(ok=False)
    if source is None:
        result.failures.append(
            f"the source model's table for '{source_model_id}' does not exist"
        )
        return result
    # No target table means nothing was written for it: an empty row set, not a
    # read failure, so an empty collection can still switch.
    target = target or _Rows(owners={}, excluded={"no-owner": 0, "no-text": 0})
    for key in sorted(source.excluded):
        result.checks[f"excluded_{key}"] = source.excluded[key]
        result.checks[f"target_excluded_{key}"] = target.excluded.get(key, 0)
    result.checks["source_rows"] = len(source.owners)
    result.checks["target_rows"] = len(target.owners)
    result.checks["target_bad_vectors"] = target.bad_vectors
    if target.bad_vectors:
        result.failures.append(
            f"{target.bad_vectors} target rows have a missing, non-finite, "
            "all-zero or wrong-dimension vector"
        )
    mismatches = sorted(
        f"{doc}:{chunk}"
        for (doc, chunk) in source.owners.keys() | target.owners.keys()
        if source.owners.get((doc, chunk)) != target.owners.get((doc, chunk))
    )
    result.checks["row_mismatches"] = mismatches[:20]
    result.checks["row_mismatch_count"] = len(mismatches)
    if mismatches:
        result.failures.append(f"{len(mismatches)} rows differ in owner or presence")
    result.ok = not result.failures
    return result


__all__ = ["AcceptanceResult", "check_collection"]
