"""LanceDB migration: rebuild FTS indexes on ``embeddings_*`` tables.

The FTS tokenizer is baked into the index at build time, so tables indexed
before the jieba switch keep the old tokenizer until the index is rebuilt.
``ensure_indexes`` only creates a missing index and the automatic rebuild in
``compact_tables`` needs fresh ingestion plus a fragment/version threshold, so a
quiescent knowledge base never picks the new tokenizer up on its own.

Rebuilds go through ``LanceDBVectorIndexStore.rebuild_text_fts_index``, which
replaces the index in place, reports what it did, and raises on failure. This
script deliberately does not compact: compaction is ``compact_tables``' job and
runs on ingestion.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Any, Dict, List, NamedTuple, Optional

from xagent.core.tools.core.RAG_tools.core.config import DEFAULT_INDEX_POLICY
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table
from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
    FtsRebuildOutcome,
    LanceDBVectorIndexStore,
)
from xagent.providers.vector_store.lancedb import get_connection_from_env

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

EMBEDDINGS_PREFIX = "embeddings_"

REBUILD = "rebuild"
SKIP = "skip"
ERROR = "error"


class Planned(NamedTuple):
    name: str
    decision: str
    snapshot: Dict[str, Any]


def list_embeddings_tables(conn: Any) -> List[str]:
    """Embeddings tables only: they are the sole carriers of an FTS index."""
    return sorted(n for n in conn.table_names() if n.startswith(EMBEDDINGS_PREFIX))


def describe_indexes(conn: Any, table_name: str) -> Dict[str, Any]:
    """Snapshot of the index state, for the eligibility plan and the log."""
    table = None
    try:
        table = conn.open_table(table_name)
        indexes = {}
        for idx in table.list_indices():
            entry: Dict[str, Any] = {
                "type": idx.index_type,
                "columns": list(idx.columns),
            }
            try:
                stats = table.index_stats(idx.name)
                entry["indexed_rows"] = stats.num_indexed_rows
                entry["unindexed_rows"] = stats.num_unindexed_rows
            except Exception as e:  # noqa: BLE001
                entry["stats_error"] = str(e)
            indexes[idx.name] = entry
        return {"version": table.version, "indexes": indexes}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    finally:
        _safe_close_table(table)


def _has_text_fts(snapshot: Dict[str, Any]) -> bool:
    """An FTS index on another column is not one the rebuild can replace."""
    indexes = snapshot.get("indexes") or {}
    return any(
        entry.get("type") == "FTS" and "text" in (entry.get("columns") or [])
        for entry in indexes.values()
    )


def build_plan(conn: Any, tables: List[str]) -> List[Planned]:
    """Classify each table once, so dry-run and execute cannot disagree."""
    plan = []
    for name in tables:
        snapshot = describe_indexes(conn, name)
        if "error" in snapshot:
            decision = ERROR
        elif _has_text_fts(snapshot):
            decision = REBUILD
        else:
            decision = SKIP
        plan.append(Planned(name, decision, snapshot))
    return plan


def rebuild_fts_indexes(
    table: Optional[str] = None,
    dry_run: bool = False,
    conn: Any = None,
) -> Dict[str, Any]:
    """Rebuild the FTS index of every ``embeddings_*`` table, or just one."""
    conn = conn or get_connection_from_env()
    tables = list_embeddings_tables(conn)

    if table is not None:
        if table not in tables:
            raise ValueError(
                f"{table!r} is not an embeddings table; found: {tables or 'none'}"
            )
        tables = [table]

    target_params = {"with_position": True, **(DEFAULT_INDEX_POLICY.fts_params or {})}
    plan = build_plan(conn, tables)
    logger.info("Tables examined (%d): %s", len(tables), tables or "none")
    logger.info("Target FTS params: %s", target_params)
    for item in plan:
        logger.info("%s: %s, before: %s", item.name, item.decision, item.snapshot)

    result: Dict[str, Any] = {
        "tables": tables,
        "succeeded": [],
        "skipped": [p.name for p in plan if p.decision == SKIP],
        "failed": [p.name for p in plan if p.decision == ERROR],
        "dry_run": dry_run,
    }

    if dry_run:
        result["would_rebuild"] = [p.name for p in plan if p.decision == REBUILD]
        logger.info("[dry-run] would rebuild: %s", result["would_rebuild"] or "none")
        logger.info("[dry-run] would skip: %s", result["skipped"] or "none")
        logger.info("[dry-run] unreadable: %s", result["failed"] or "none")
        return result

    store = LanceDBVectorIndexStore()
    for item in plan:
        if item.decision != REBUILD:
            continue
        started = time.monotonic()
        try:
            outcome = store.rebuild_text_fts_index(item.name)
        except BaseException as e:  # noqa: BLE001
            # BaseException: a pyo3 Rust panic is not an Exception, and a
            # panicking rebuild must not be reported as a success.
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            logger.error("%s: rebuild raised %s: %s", item.name, type(e).__name__, e)
            result["failed"].append(item.name)
            continue
        elapsed = time.monotonic() - started
        logger.info(
            "%s took %.2fs, outcome=%s, after: %s",
            item.name,
            elapsed,
            outcome.value,
            describe_indexes(conn, item.name),
        )
        if outcome is FtsRebuildOutcome.REBUILT:
            result["succeeded"].append(item.name)
        else:
            # The plan said this table needed a rebuild, so anything else --
            # a lock held elsewhere, an index that vanished -- is unfinished
            # work the operator has to re-run, not a clean skip.
            logger.error("%s was not rebuilt: %s", item.name, outcome.value)
            result["failed"].append(item.name)

    logger.info("Succeeded (%d): %s", len(result["succeeded"]), result["succeeded"])
    logger.info("Skipped (%d): %s", len(result["skipped"]), result["skipped"])
    logger.info("Failed (%d): %s", len(result["failed"]), result["failed"])
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild FTS indexes on LanceDB embeddings tables.\n\n"
        "The tokenizer is stored inside the index, so tables built before the "
        "jieba switch need an explicit rebuild. Safe to re-run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--table", help="Rebuild only this table (default: all)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be rebuilt and what would be skipped",
    )
    args = parser.parse_args(argv)

    try:
        result = rebuild_fts_indexes(table=args.table, dry_run=args.dry_run)
    except Exception as e:  # noqa: BLE001
        logger.error("Rebuild failed: %s", e, exc_info=True)
        return 2

    return 1 if result["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
