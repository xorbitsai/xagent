"""LanceDB migration: Fix file_id column to be nullable in documents table.

The on-disk documents table may have file_id as non-nullable, but the code
schema defines it as nullable (pa.field("file_id", pa.string())). The previous
migration (backfill_documents_file_id.py) wrote None values into this column,
creating rows with null values in a non-nullable field. This causes LanceDB
to crash on read with:

    RuntimeError: Found unmasked nulls for non-nullable StructArray field "file_id"

This script uses alter_columns to make file_id nullable, matching the code
schema definition.

Usage:
    # Dry-run (default)
    python -m xagent.migrations.lancedb.fix_file_id_nullable

    # Apply the fix
    python -m xagent.migrations.lancedb.fix_file_id_nullable --execute
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

from xagent.providers.vector_store.lancedb import get_connection_from_env

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global lock to prevent concurrent migrations
_migration_lock = threading.Lock()


def _get_lock_file_path() -> str:
    """Resolve file lock path for cross-process migration coordination."""
    lock_file = os.environ.get("LANCEDB_MIGRATION_LOCK_FILE")
    if lock_file:
        return lock_file

    lancedb_dir = os.environ.get("LANCEDB_DIR")
    if lancedb_dir:
        return os.path.join(lancedb_dir, ".lancedb_fix_file_id_nullable.lock")

    return os.path.join(
        tempfile.gettempdir(),
        "xagent_lancedb_fix_file_id_nullable.lock",
    )


def _acquire_file_lock() -> Any | None:
    """Acquire non-blocking file lock shared by all local processes."""
    lock_path = _get_lock_file_path()
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


def _is_file_id_nullable(table: Any) -> bool:
    """Check if file_id column is already nullable."""
    for field in table.schema:
        if field.name == "file_id":
            return bool(field.nullable)
    # Column doesn't exist — treat as needing no fix
    return True


def fix_file_id_nullable(
    dry_run: bool = True,
    conn: DBConnection | None = None,
) -> dict[str, Any]:
    """Make the file_id column nullable in the documents table.

    Args:
        dry_run: If True, only check and report without making changes.
        conn: LanceDB connection (uses default if None).

    Returns:
        Dictionary with migration status and statistics.
    """
    if conn is None:
        conn = get_connection_from_env()

    result: dict[str, Any] = {"table": "documents", "dry_run": dry_run}

    try:
        table = conn.open_table("documents")
    except Exception as exc:
        logger.error("Could not open 'documents' table: %s", exc)
        result["error"] = str(exc)
        return result

    # Check if file_id column exists
    if "file_id" not in table.schema.names:
        logger.info("Column 'file_id' not found in 'documents' table, nothing to fix")
        result["skipped"] = True
        result["reason"] = "column_not_found"
        return result

    # Check if already nullable
    if _is_file_id_nullable(table):
        logger.info("file_id is already nullable, no migration needed")
        result["skipped"] = True
        result["reason"] = "already_nullable"
        return result

    logger.info(
        "file_id is non-nullable in 'documents' table.%s",
        " Would fix with alter_columns." if dry_run else " Fixing with alter_columns.",
    )

    if dry_run:
        result["needs_fix"] = True
        return result

    # Apply the fix
    try:
        table.alter_columns({"path": "file_id", "nullable": True})
        logger.info("Successfully made 'file_id' nullable in 'documents' table")
        result["fixed"] = True
    except Exception as exc:
        logger.error("Failed to alter 'file_id' nullability: %s", exc)
        result["error"] = str(exc)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix file_id column to be nullable in LanceDB documents table.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the fix (default: dry-run)",
    )
    args = parser.parse_args()

    if not args.execute:
        logger.info("=" * 60)
        logger.info("DRY RUN MODE - No changes will be made")
        logger.info("Use --execute to actually perform the fix")
        logger.info("=" * 60)

    if not _migration_lock.acquire(blocking=False):
        logger.warning("Another migration is already in progress")
        sys.exit(1)

    file_lock = None
    try:
        file_lock = _acquire_file_lock()
        if file_lock is None:
            logger.warning("Another migration is running in a different process")
            sys.exit(1)

        result = fix_file_id_nullable(dry_run=not args.execute)

        if "error" in result:
            logger.error("Migration failed: %s", result["error"])
            sys.exit(1)
        elif result.get("skipped"):
            logger.info("Migration skipped: %s", result.get("reason"))
        elif result.get("fixed"):
            logger.info("Migration complete")
        elif result.get("needs_fix"):
            logger.info("Dry-run: file_id needs to be made nullable")

    finally:
        if file_lock is not None:
            _release_file_lock(file_lock)
        _migration_lock.release()


if __name__ == "__main__":
    main()
