"""LanceDB migration: Backfill user_id for chunks and embeddings tables.

This migration script backfills the user_id field in chunks and embeddings tables
by joining with the documents table. This is necessary for multi-tenancy data isolation.

Uses two-phase migration:
- Phase 1: Normal backfill, mark orphaned records with reserved sentinel
- Phase 2: Retry orphaned records in case their parent documents were created concurrently
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

# Add parent directories to path for imports
# This must be done before importing project modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from xagent.core.tools.core.RAG_tools.core.config import MIN_INT64

# Import after path modification (required for standalone migration scripts)
# ruff: noqa: E402
from xagent.core.tools.core.RAG_tools.utils.lancedb_query_utils import (
    list_embeddings_table_names,
    query_to_list,
)
from xagent.core.tools.core.RAG_tools.utils.string_utils import escape_lancedb_string
from xagent.providers.vector_store.lancedb import get_connection_from_env

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Batch size for processing records to avoid memory issues
BATCH_SIZE = 10000

# Orphaned record markers
# Use int64 lower-bound sentinels reserved for internal migration states.
# This avoids collisions with user filtering semantics (e.g. unauthenticated access).
ORPHANED_TEMPORARY = (
    MIN_INT64  # Phase 1: Temporary orphan (may be due to concurrent document creation)
)
ORPHANED_PERMANENT = (
    MIN_INT64 + 1
)  # Phase 2: Permanent orphan (confirmed no matching document exists)

# Global lock to prevent concurrent migrations
_migration_lock = threading.Lock()


def _ensure_table_exists(conn: DBConnection, table_name: str) -> None:
    """Ensure a table exists, creating it with default schema if it doesn't.

    Unlike ensure_*_table functions, this doesn't validate schema, allowing
    migration code to work with old schema tables.
    """
    try:
        conn.open_table(table_name)
    except Exception:
        # Table doesn't exist, create it with default schema
        if table_name == "chunks":
            import pyarrow as pa  # type: ignore

            schema = pa.schema(
                [
                    pa.field("collection", pa.string()),
                    pa.field("doc_id", pa.string()),
                    pa.field("parse_hash", pa.string()),
                    pa.field("chunk_id", pa.string()),
                    pa.field("index", pa.int32()),
                    pa.field("text", pa.large_string()),
                    pa.field("page_number", pa.int32()),
                    pa.field("section", pa.string()),
                    pa.field("anchor", pa.string()),
                    pa.field("json_path", pa.string()),
                    pa.field("chunk_hash", pa.string()),
                    pa.field("config_hash", pa.string()),
                    pa.field("created_at", pa.timestamp("us")),
                    pa.field("metadata", pa.string()),
                    pa.field("user_id", pa.int64()),
                ]
            )
            conn.create_table(table_name, schema=schema)
        elif table_name == "documents":
            import pyarrow as pa

            schema = pa.schema(
                [
                    pa.field("collection", pa.string()),
                    pa.field("doc_id", pa.string()),
                    pa.field("file_id", pa.string()),
                    pa.field("source_path", pa.string()),
                    pa.field("file_type", pa.string()),
                    pa.field("content_hash", pa.string()),
                    pa.field("uploaded_at", pa.timestamp("us")),
                    pa.field("title", pa.string()),
                    pa.field("language", pa.string()),
                    pa.field("user_id", pa.int64()),
                ]
            )
            conn.create_table(table_name, schema=schema)


def _get_migration_lock_file_path() -> str:
    """Resolve file lock path for cross-process migration coordination."""
    lock_file = os.environ.get("LANCEDB_MIGRATION_LOCK_FILE")
    if lock_file:
        return lock_file

    lancedb_dir = os.environ.get("LANCEDB_DIR")
    if lancedb_dir:
        return os.path.join(lancedb_dir, ".lancedb_user_id_migration.lock")

    return os.path.join(
        tempfile.gettempdir(),
        "xagent_lancedb_user_id_migration.lock",
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


def _orphaned_temporary_filter() -> str:
    """Build a LanceDB-safe filter for ORPHANED_TEMPORARY int64 sentinel."""
    return f"user_id = cast({ORPHANED_TEMPORARY} as bigint)"


def _build_doc_id_in_filter(doc_ids: list[str]) -> str:
    """Build a safe LanceDB IN filter for doc_id values."""
    escaped_ids = [f"'{escape_lancedb_string(doc_id)}'" for doc_id in doc_ids]
    return f"doc_id IN ({', '.join(escaped_ids)})"


def _build_record_update_filter(
    record: dict[str, Any], filter_fields: list[str]
) -> str:
    """Build a safe AND filter targeting a single record by key fields."""
    filter_parts: list[str] = []
    for field_name in filter_fields:
        field_value = record.get(field_name)
        if field_value is None:
            filter_parts.append(f"{field_name} IS NULL")
            continue

        # Keep numeric comparisons unquoted; quote and escape all other scalar values.
        if isinstance(field_value, (int, float)) and not isinstance(field_value, bool):
            filter_parts.append(f"{field_name} = {field_value}")
        else:
            escaped_value = escape_lancedb_string(field_value)
            filter_parts.append(f"{field_name} = '{escaped_value}'")
    return " and ".join(filter_parts)


def _remap_legacy_orphaned_user_ids(conn: DBConnection, dry_run: bool = False) -> dict:
    """One-time compatibility remap for legacy orphan marker values.

    Historical migration runs used ``-1`` as temporary orphan marker, which now
    conflicts with unauthenticated read filtering semantics. This helper remaps
    legacy ``-1`` to the reserved int64 sentinel to keep Phase 2 retry behavior
    consistent after upgrading.
    """

    remapped_counts: dict[str, int] = {}
    target_tables = ["chunks", *_get_embeddings_tables(conn)]

    for table_name in target_tables:
        try:
            table = conn.open_table(table_name)
        except Exception as exc:
            logger.warning("Skip legacy remap for %s: %s", table_name, exc)
            continue

        try:
            legacy_rows = query_to_list(
                table.search().where("user_id = -1").limit(BATCH_SIZE)
            )
            if not legacy_rows:
                continue

            remapped_counts[table_name] = len(legacy_rows)
            if dry_run:
                logger.info(
                    "Dry-run legacy remap: %s rows in %s would be updated from -1 to %s",
                    len(legacy_rows),
                    table_name,
                    ORPHANED_TEMPORARY,
                )
                continue

            table.update("user_id = -1", {"user_id": ORPHANED_TEMPORARY})
            logger.info(
                "Legacy remap complete: %s rows in %s updated from -1 to %s",
                len(legacy_rows),
                table_name,
                ORPHANED_TEMPORARY,
            )
        except Exception as exc:
            logger.warning("Legacy remap failed for %s: %s", table_name, exc)

    return remapped_counts


def _backfill_table_core(
    table: Any,
    docs_table: Any,
    query_filter: str,
    filter_fields: list[str],
    failure_user_id: int,
    dry_run: bool,
    log_prefix: str = "",
) -> dict:
    """Core logic for backfilling a single table.

    Args:
        table: LanceDB table to backfill
        docs_table: Documents table for lookup
        query_filter: Filter to find records needing backfill (e.g., "user_id IS NULL")
        filter_fields: Fields used to identify a specific record for update
        failure_user_id: user_id to set if document lookup fails (e.g., -1 or -2)
        dry_run: If True, don't make actual changes
        log_prefix: Prefix for log messages

    Returns:
        Dictionary with statistics
    """
    total_backfilled = 0
    total_skipped = 0
    total_failed = 0
    batch_number = 0

    while True:
        # Get a batch of records matching the filter
        batch = query_to_list(table.search().where(query_filter).limit(BATCH_SIZE))

        if not batch:
            break

        batch_number += 1
        logger.info(
            f"{log_prefix} Processing batch #{batch_number}: {len(batch)} records..."
        )

        # Build doc_id -> user_id mapping from documents table
        doc_user_map = {}
        all_doc_ids = [
            doc_id for doc_id in set(r.get("doc_id") for r in batch) if doc_id
        ]

        if all_doc_ids:
            # Bulk lookup for documents
            docs = query_to_list(
                docs_table.search()
                .where(_build_doc_id_in_filter(all_doc_ids))
                .limit(len(all_doc_ids))
            )
            for doc in docs:
                if doc.get("user_id") is not None:
                    doc_user_map[doc.get("doc_id")] = doc.get("user_id")

        logger.info(
            f"{log_prefix} Batch #{batch_number}: Found user_id for {len(doc_user_map)} / {len(all_doc_ids)} documents"
        )

        # Update records
        skipped = 0
        updated_in_batch = 0
        for record in batch:
            doc_id = record.get("doc_id")

            if doc_id in doc_user_map:
                user_id = doc_user_map[doc_id]
                is_recovered = True
            else:
                user_id = failure_user_id
                is_recovered = False
                skipped += 1
                total_skipped += 1

            if not dry_run:
                try:
                    # Build update filter
                    update_filter = _build_record_update_filter(record, filter_fields)
                    table.update(update_filter, {"user_id": user_id})
                    updated_in_batch += 1

                    if is_recovered:
                        total_backfilled += 1
                except Exception as e:
                    total_failed += 1
                    logger.warning(f"{log_prefix} Failed to update record: {e}")
            else:
                if is_recovered:
                    total_backfilled += 1

        logger.info(
            f"{log_prefix} Batch #{batch_number}: {len(batch) - skipped} processed, {skipped} marked as failure_id ({failure_user_id})"
        )
        if dry_run:
            # Dry-run does not mutate records, so processing additional batches would
            # read the same records repeatedly and never converge.
            break
        if updated_in_batch == 0:
            logger.error(
                "%s Batch #%s made zero update progress; aborting to avoid infinite loop.",
                log_prefix,
                batch_number,
            )
            break

    return {
        "total": total_backfilled + total_skipped + total_failed,
        "backfilled": total_backfilled,
        "skipped": total_skipped,
        "failed": total_failed,
    }


def backfill_chunks_table(
    dry_run: bool = False, conn: DBConnection | None = None
) -> dict:
    """Backfill user_id for chunks table (Phase 1)."""
    if conn is None:
        conn = get_connection_from_env()

    # For migration, we need to work with existing tables even if they have old schema
    # Don't use ensure_chunks_table as it validates schema which fails for old tables
    _ensure_table_exists(conn, "chunks")
    _ensure_table_exists(conn, "documents")

    chunks_table = conn.open_table("chunks")
    docs_table = conn.open_table("documents")

    logger.info("Phase 1: Starting chunks table user_id backfill...")
    result = _backfill_table_core(
        table=chunks_table,
        docs_table=docs_table,
        query_filter="user_id IS NULL",
        filter_fields=["doc_id", "chunk_id", "parse_hash"],
        failure_user_id=ORPHANED_TEMPORARY,
        dry_run=dry_run,
        log_prefix="Chunks Phase 1:",
    )
    result["table"] = "chunks"
    return result


def backfill_orphaned_chunks(
    dry_run: bool = False, conn: DBConnection | None = None
) -> dict:
    """Retry backfill for orphaned chunks (Phase 2)."""
    if conn is None:
        conn = get_connection_from_env()

    # For migration, we need to work with existing tables even if they have old schema
    _ensure_table_exists(conn, "chunks")
    _ensure_table_exists(conn, "documents")

    chunks_table = conn.open_table("chunks")
    docs_table = conn.open_table("documents")

    logger.info("Phase 2: Retrying orphaned chunks (user_id = ORPHANED_TEMPORARY)...")
    result = _backfill_table_core(
        table=chunks_table,
        docs_table=docs_table,
        query_filter=_orphaned_temporary_filter(),
        filter_fields=["doc_id", "chunk_id", "parse_hash"],
        failure_user_id=ORPHANED_PERMANENT,
        dry_run=dry_run,
        log_prefix="Chunks Phase 2:",
    )
    result["table"] = "chunks"
    return result


def _get_embeddings_tables(conn: DBConnection) -> list[str]:
    """Helper to get all embeddings tables with API compatibility."""
    try:
        return list_embeddings_table_names(conn)
    except Exception as e:
        logger.warning(f"Failed to list LanceDB tables: {e}")
        return []


def backfill_embeddings_table(
    dry_run: bool = False, conn: DBConnection | None = None
) -> dict:
    """Backfill user_id for embeddings tables (Phase 1)."""
    if conn is None:
        conn = get_connection_from_env()

    _ensure_table_exists(conn, "documents")
    embeddings_tables = _get_embeddings_tables(conn)

    if not embeddings_tables:
        return {
            "table": "embeddings",
            "total": 0,
            "backfilled": 0,
            "skipped": 0,
            "failed": 0,
            "details": [],
        }

    docs_table = conn.open_table("documents")
    all_results = []

    for table_name in embeddings_tables:
        logger.info(f"Phase 1: Processing {table_name}...")
        res = _backfill_table_core(
            table=conn.open_table(table_name),
            docs_table=docs_table,
            query_filter="user_id IS NULL",
            filter_fields=["doc_id", "chunk_id", "parse_hash", "model"],
            failure_user_id=ORPHANED_TEMPORARY,
            dry_run=dry_run,
            log_prefix=f"Embeddings Phase 1 ({table_name}):",
        )
        res["table"] = table_name
        all_results.append(res)

    return {
        "table": "embeddings",
        "total": sum(r["total"] for r in all_results),
        "backfilled": sum(r["backfilled"] for r in all_results),
        "skipped": sum(r["skipped"] for r in all_results),
        "failed": sum(r["failed"] for r in all_results),
        "details": all_results,
    }


def backfill_orphaned_embeddings(
    dry_run: bool = False, conn: DBConnection | None = None
) -> dict:
    """Retry backfill for orphaned embeddings (Phase 2)."""
    if conn is None:
        conn = get_connection_from_env()

    _ensure_table_exists(conn, "documents")

    embeddings_tables = _get_embeddings_tables(conn)
    if not embeddings_tables:
        return {
            "table": "embeddings",
            "total": 0,
            "backfilled": 0,
            "skipped": 0,
            "failed": 0,
            "details": [],
        }

    docs_table = conn.open_table("documents")
    all_results = []

    for table_name in embeddings_tables:
        logger.info(f"Phase 2: Processing {table_name}...")
        res = _backfill_table_core(
            table=conn.open_table(table_name),
            docs_table=docs_table,
            query_filter=_orphaned_temporary_filter(),
            filter_fields=["doc_id", "chunk_id", "parse_hash", "model"],
            failure_user_id=ORPHANED_PERMANENT,
            dry_run=dry_run,
            log_prefix=f"Embeddings Phase 2 ({table_name}):",
        )
        res["table"] = table_name
        all_results.append(res)

    return {
        "table": "embeddings",
        "total": sum(r["total"] for r in all_results),
        "backfilled": sum(r["backfilled"] for r in all_results),
        "skipped": sum(r["skipped"] for r in all_results),
        "failed": sum(r["failed"] for r in all_results),
        "details": all_results,
    }


def backfill_all(dry_run: bool = False, conn: DBConnection | None = None) -> dict:
    """Run full two-phase backfill for all tables."""
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
        logger.info("LanceDB User ID Backfill Migration (Two-Phase)")
        logger.info("=" * 60)

        legacy_remap = _remap_legacy_orphaned_user_ids(conn=conn, dry_run=dry_run)
        if legacy_remap:
            logger.info("Legacy orphan remap summary: %s", legacy_remap)
        has_legacy_chunk_orphans = legacy_remap.get("chunks", 0) > 0
        has_legacy_embedding_orphans = any(
            table_name.startswith("embeddings_") and count > 0
            for table_name, count in legacy_remap.items()
        )

        # Phase 1
        chunks_res = backfill_chunks_table(dry_run=dry_run, conn=conn)
        embeddings_res = backfill_embeddings_table(dry_run=dry_run, conn=conn)

        # Phase 2
        chunks_retry = {"backfilled": 0, "skipped": chunks_res["skipped"]}
        embeddings_retry = {"backfilled": 0, "skipped": embeddings_res["skipped"]}

        if chunks_res["skipped"] > 0 or has_legacy_chunk_orphans:
            chunks_retry = backfill_orphaned_chunks(dry_run=dry_run, conn=conn)
            chunks_res["backfilled"] += chunks_retry["backfilled"]
            chunks_res["skipped"] = chunks_retry["skipped"]
            chunks_res["failed"] += chunks_retry["failed"]

        if embeddings_res["skipped"] > 0 or has_legacy_embedding_orphans:
            embeddings_retry = backfill_orphaned_embeddings(dry_run=dry_run, conn=conn)
            embeddings_res["backfilled"] += embeddings_retry["backfilled"]
            embeddings_res["skipped"] = embeddings_retry["skipped"]
            embeddings_res["failed"] += embeddings_retry["failed"]

        return {
            "chunks": chunks_res,
            "embeddings": embeddings_res,
            "locked": True,
        }
    finally:
        if file_lock is not None:
            _release_file_lock(file_lock)
        _migration_lock.release()
        logger.info("Migration lock released")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill user_id for LanceDB tables for multi-tenancy support.\n\n"
        "This script performs a two-phase migration:\n"
        f"  Phase 1: Backfill records, mark orphaned records with user_id = {ORPHANED_TEMPORARY}\n"
        f"  Phase 2: Retry orphaned records, mark permanent orphans with user_id = {ORPHANED_PERMANENT}\n\n"
        "Orphaned records occur when chunks/embeddings exist without matching documents,\n"
        "which can happen due to concurrent document creation during migration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate migration without making actual changes",
    )
    parser.add_argument(
        "--chunks-only",
        action="store_true",
        help="Only backfill chunks table (skip embeddings tables)",
    )
    parser.add_argument(
        "--embeddings-only",
        action="store_true",
        help="Only backfill embeddings tables (skip chunks table)",
    )
    args = parser.parse_args()

    try:
        if args.chunks_only:
            result = backfill_chunks_table(dry_run=args.dry_run)
        elif args.embeddings_only:
            result = backfill_embeddings_table(dry_run=args.dry_run)
        else:
            result = backfill_all(dry_run=args.dry_run)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        sys.exit(2)
