"""LanceDB migration: Backfill documents table file_id and user_id.

This migration script backfills missing values in the documents table:
- Phase 1: Convert legacy empty string file_id values to None
- Phase 2: Backfill user_id from source_path ownership hints

This migration ensures data consistency with the main branch semantics where
None is the standard representation for "no file_id" and user_id is properly
set for multi-tenancy support.

Uses similar patterns to backfill_user_id.py for concurrent safety.
"""

from __future__ import annotations

import argparse
import fcntl
import logging
import os
import sys
import tempfile
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lancedb.db import DBConnection

from xagent.core.tools.core.RAG_tools.core.config import (
    DEFAULT_BACKFILL_BATCH_SIZE,
    DEFAULT_BACKFILL_MAX_ITERATIONS,
)
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
    ensure_documents_table,
)
from xagent.core.tools.core.RAG_tools.utils.lancedb_query_utils import query_to_list
from xagent.core.tools.core.RAG_tools.utils.string_utils import escape_lancedb_string
from xagent.providers.vector_store.lancedb import get_connection_from_env

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global lock to prevent concurrent migrations
_migration_lock = threading.Lock()


def _get_migration_lock_file_path() -> str:
    """Resolve file lock path for cross-process migration coordination."""
    lock_file = os.environ.get("LANCEDB_MIGRATION_LOCK_FILE")
    if lock_file:
        return lock_file

    lancedb_dir = os.environ.get("LANCEDB_DIR")
    if lancedb_dir:
        return os.path.join(lancedb_dir, ".lancedb_documents_migration.lock")

    return os.path.join(
        tempfile.gettempdir(),
        "xagent_lancedb_documents_migration.lock",
    )


def _acquire_file_lock() -> Any | None:
    """Acquire non-blocking file lock shared by all local processes."""
    lock_path = _get_migration_lock_file_path()
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    lock_file = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(str(os.getpid()))
        lock_file.flush()
        return lock_file
    except BlockingIOError:
        lock_file.close()
        return None
    except Exception:
        lock_file.close()
        raise


def _release_file_lock(lock_file: Any) -> None:
    """Release file lock and close file handle safely."""
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    finally:
        lock_file.close()


def _extract_user_id_from_source_path(source_path: str) -> int | None:
    """Extract user_id from a storage path like ``.../user_58/...``."""
    import re

    match = re.search(r"/user_(\d+)(?:/|$)", source_path)
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def _chunked(seq: list[str], chunk_size: int) -> list[list[str]]:
    """Split a list into fixed-size chunks.

    Args:
        seq: Input list.
        chunk_size: Maximum chunk size (must be >= 1).

    Returns:
        A list of chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be >= 1")
    return [seq[i : i + chunk_size] for i in range(0, len(seq), chunk_size)]


def backfill_file_id_to_none(
    dry_run: bool = False, conn: DBConnection | None = None
) -> dict:
    """Convert legacy empty string file_id values to None for consistency.

    Previous versions normalized NULL file_id to empty strings. This function
    reverses that to maintain consistency with main branch, where None is the
    standard representation for "no file_id".

    Args:
        dry_run: If True, don't make actual changes
        conn: LanceDB connection (uses default if None)

    Returns:
        Dictionary with statistics
    """
    if conn is None:
        conn = get_connection_from_env()

    ensure_documents_table(conn)

    table_name = "documents"
    result = {"table": table_name, "updated": 0, "dry_run": dry_run}

    try:
        table = conn.open_table(table_name)
        if "file_id" not in table.schema.names:
            logger.info("Column 'file_id' not found in '%s'", table_name)
            return result

        # Check if there are any empty string file_id values to backfill
        empty_string_rows = query_to_list(table.search().where("file_id = ''").limit(1))
        if not empty_string_rows:
            logger.info("No empty string file_id values found in '%s'", table_name)
            return result

        # Count total rows to update
        total_rows = len(query_to_list(table.search().where("file_id = ''")))
        result["total_found"] = total_rows

        if dry_run:
            logger.info(
                "Dry-run: would update %d rows in '%s' from empty string to None",
                total_rows,
                table_name,
            )
            result["updated"] = total_rows
            return result

        # Convert empty strings to NULL
        table.update("file_id = ''", {"file_id": None})
        result["updated"] = total_rows

        logger.info(
            "Backfilled %d empty string file_id values to NULL in '%s'",
            total_rows,
            table_name,
        )
    except Exception as exc:
        logger.error("Failed to backfill file_id in '%s': %s", table_name, exc)
        result["error"] = str(exc)

    return result


def backfill_user_id_from_source_path(
    dry_run: bool = False, conn: DBConnection | None = None
) -> dict:
    """Backfill legacy documents.user_id from source_path ownership hints.

    Legacy rows may have user_id = NULL but still include stable upload
    paths like .../user_{id}/{collection}/file. Recovering user_id keeps
    multi-tenant filtering consistent and restores document visibility for the
    owning user without broadening access permissions.

    Args:
        dry_run: If True, don't make actual changes
        conn: LanceDB connection (uses default if None)

    Returns:
        Dictionary with statistics
    """
    if conn is None:
        conn = get_connection_from_env()

    ensure_documents_table(conn)

    table_name = "documents"
    result = {"table": table_name, "updated": 0, "dry_run": dry_run}

    try:
        table = conn.open_table(table_name)
        if (
            "user_id" not in table.schema.names
            or "source_path" not in table.schema.names
        ):
            logger.info(
                "Columns 'user_id' or 'source_path' not found in '%s'", table_name
            )
            return result

        total_updated = 0
        iteration = 0
        failed_doc_ids: set[tuple[str, str]] = (
            set()
        )  # Track (collection, doc_id) pairs that failed

        while iteration < DEFAULT_BACKFILL_MAX_ITERATIONS:
            iteration += 1

            # Build filter to exclude previously failed doc_ids
            base_filter = "user_id IS NULL"
            if failed_doc_ids:
                # Exclude failed (collection, doc_id) pairs to avoid retrying persistent failures
                exclusion_clauses = [
                    f"NOT (collection = '{escape_lancedb_string(c)}' AND doc_id = '{escape_lancedb_string(d)}')"
                    for c, d in failed_doc_ids
                ]
                base_filter = f"{base_filter} AND ({' AND '.join(exclusion_clauses)})"

            pending_rows = query_to_list(
                table.search().where(base_filter).limit(DEFAULT_BACKFILL_BATCH_SIZE)
            )
            if not pending_rows:
                # No more rows to process
                break

            if dry_run and iteration == 1:
                result["total_found"] = len(pending_rows)

            # Collect update candidates for this batch.
            # We avoid per-row updates by grouping updates that share
            # (collection, inferred_user_id) and updating doc_id via an IN (...) filter.
            # If a grouped update fails, we fall back to per-row updates for that group
            # to preserve accurate failure tracking.
            updates: list[dict[str, Any]] = []
            for row in pending_rows:
                source_path = row.get("source_path")
                if not isinstance(source_path, str) or not source_path:
                    continue
                inferred_user_id = _extract_user_id_from_source_path(source_path)
                if inferred_user_id is None:
                    continue
                doc_id = row.get("doc_id")
                collection = row.get("collection")
                if not isinstance(doc_id, str) or not isinstance(collection, str):
                    continue

                escaped_doc_id = escape_lancedb_string(doc_id)
                escaped_collection = escape_lancedb_string(collection)
                updates.append(
                    {
                        "filter": f"collection = '{escaped_collection}' and doc_id = '{escaped_doc_id}' and user_id IS NULL",
                        "values": {"user_id": inferred_user_id},
                        "collection": collection,  # Track for failure tracking
                        "doc_id": doc_id,  # Track for failure tracking
                        "escaped_collection": escaped_collection,
                        "escaped_doc_id": escaped_doc_id,
                        "user_id": inferred_user_id,
                    }
                )

            if not updates:
                # No valid updates in this batch, avoid infinite loop
                break

            if dry_run:
                # Dry-run: count potential updates without applying
                total_updated += len(updates)
                logger.info(
                    "Dry-run: would update %d rows in iteration %d",
                    len(updates),
                    iteration,
                )
                # Stop after first batch in dry-run mode
                break

            # Apply grouped batch updates
            updated_in_batch = 0
            batch_failures: list[tuple[str, str]] = []
            group_map: dict[tuple[str, int], list[dict[str, Any]]] = {}
            for update in updates:
                key = (str(update["escaped_collection"]), int(update["user_id"]))
                group_map.setdefault(key, []).append(update)

            # Keep IN clause size bounded to avoid huge filter strings.
            in_clause_chunk_size = 50

            for (escaped_collection, user_id_val), group_updates in group_map.items():
                escaped_doc_ids = [str(u["escaped_doc_id"]) for u in group_updates]
                for doc_id_chunk in _chunked(escaped_doc_ids, in_clause_chunk_size):
                    in_expr = ", ".join(f"'{d}'" for d in doc_id_chunk)
                    group_filter = (
                        f"user_id IS NULL AND collection = '{escaped_collection}' "
                        f"AND doc_id IN ({in_expr})"
                    )
                    try:
                        table.update(group_filter, {"user_id": user_id_val})
                        updated_in_batch += len(doc_id_chunk)
                    except Exception as exc:
                        logger.debug(
                            "Grouped update failed (collection=%s, user_id=%s, size=%d): %s",
                            escaped_collection,
                            user_id_val,
                            len(doc_id_chunk),
                            exc,
                        )
                        # Fall back to per-row updates for accurate failure tracking.
                        chunk_set = set(doc_id_chunk)
                        row_updates = [
                            u
                            for u in group_updates
                            if str(u["escaped_doc_id"]) in chunk_set
                        ]
                        for u in row_updates:
                            try:
                                table.update(u["filter"], u["values"])
                                updated_in_batch += 1
                            except Exception as row_exc:
                                batch_failures.append((u["collection"], u["doc_id"]))
                                logger.debug(
                                    "Failed to update (collection=%s, doc_id=%s): %s",
                                    u["collection"],
                                    u["doc_id"],
                                    row_exc,
                                )

            # Add failed doc_ids to exclusion set for next iteration
            failed_doc_ids.update(batch_failures)

            total_updated += updated_in_batch

            logger.info(
                "Iteration %d: updated %d rows (total: %d, failed: %d, excluded: %d)",
                iteration,
                updated_in_batch,
                total_updated,
                len(batch_failures),
                len(failed_doc_ids),
            )

            # Idempotency check: if no rows were updated, we're done
            if updated_in_batch == 0:
                # All updates in this batch failed - stop to avoid infinite retry
                if len(pending_rows) > 0:
                    logger.warning(
                        "All %d updates in batch failed. "
                        "These rows will be skipped in subsequent iterations. "
                        "Total excluded rows: %d",
                        len(pending_rows),
                        len(failed_doc_ids),
                    )
                break

        result["updated"] = total_updated
        result["iterations"] = iteration
        result["failed_excluded"] = len(failed_doc_ids)

        if total_updated > 0:
            logger.info(
                "Backfilled %d user_id values from source_path in '%s'",
                total_updated,
                table_name,
            )
        else:
            logger.info("No user_id backfill needed in '%s'", table_name)

    except Exception as exc:
        logger.error("Failed to backfill user_id in '%s': %s", table_name, exc)
        result["error"] = str(exc)

    return result


def backfill_all(dry_run: bool = False, conn: DBConnection | None = None) -> dict:
    """Run full backfill for documents table.

    Args:
        dry_run: If True, don't make actual changes
        conn: LanceDB connection (uses default if None)

    Returns:
        Dictionary with results from both backfill phases
    """
    if conn is None:
        conn = get_connection_from_env()

    if not _migration_lock.acquire(blocking=False):
        logger.warning("Another migration is already in progress")
        return {"error": "Migration lock already held"}

    file_lock = None
    try:
        file_lock = _acquire_file_lock()
        if file_lock is None:
            logger.warning("Another migration is running in a different process")
            return {"error": "Migration file lock already held"}

        logger.info("=" * 60)
        logger.info("LanceDB Documents Table Backfill Migration")
        logger.info("=" * 60)

        results: dict[str, Any] = {}

        # Phase 1: Convert empty string file_id to None
        logger.info("Phase 1: Converting empty string file_id to None...")
        file_id_result = backfill_file_id_to_none(dry_run=dry_run, conn=conn)
        results["file_id"] = file_id_result

        # Phase 2: Backfill user_id from source_path
        logger.info("Phase 2: Backfilling user_id from source_path...")
        user_id_result = backfill_user_id_from_source_path(dry_run=dry_run, conn=conn)
        results["user_id"] = user_id_result

        results["locked"] = True

        return results
    finally:
        if file_lock is not None:
            _release_file_lock(file_lock)
        _migration_lock.release()
        logger.info("Migration lock released")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill documents table for LanceDB.\n\n"
        "This migration performs two phases:\n"
        "  Phase 1: Convert empty string file_id values to None\n"
        "  Phase 2: Backfill user_id from source_path ownership hints\n\n"
        "This ensures consistency with main branch semantics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate migration without making actual changes",
    )
    parser.add_argument(
        "--file-id-only",
        action="store_true",
        help="Only backfill file_id (skip user_id)",
    )
    parser.add_argument(
        "--user-id-only",
        action="store_true",
        help="Only backfill user_id (skip file_id)",
    )
    args = parser.parse_args()

    try:
        if args.file_id_only:
            result = backfill_file_id_to_none(dry_run=args.dry_run)
        elif args.user_id_only:
            result = backfill_user_id_from_source_path(dry_run=args.dry_run)
        else:
            result = backfill_all(dry_run=args.dry_run)

        # Print results
        if "error" in result:
            logger.error("Migration failed: %s", result["error"])
            sys.exit(1)
        else:
            logger.info("Migration complete: %s", result)
            sys.exit(0)
    except Exception as e:
        logger.error("Migration failed: %s", e, exc_info=True)
        sys.exit(2)
