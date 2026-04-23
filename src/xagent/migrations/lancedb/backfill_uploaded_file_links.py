"""LanceDB migration: backfill documents.file_id from UploadedFile records.

This migration links legacy ``documents`` rows without ``file_id`` to
``uploaded_files`` records using ``source_path`` as the primary join key.

Migration behavior:
- Reuse existing ``UploadedFile`` when ``storage_path == source_path``.
- Create missing ``UploadedFile`` when source file exists on disk.
- Mark unresolved rows as ``unbackfillable`` in migration report.

The script is idempotent and supports dry-run mode.
"""

from __future__ import annotations

import argparse
import fcntl
import logging
import mimetypes
import os
import re
import sys
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

if TYPE_CHECKING:
    from lancedb.db import DBConnection

from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
    ensure_documents_table,
)
from xagent.core.tools.core.RAG_tools.utils.lancedb_query_utils import query_to_list
from xagent.core.tools.core.RAG_tools.utils.string_utils import escape_lancedb_string
from xagent.providers.vector_store.lancedb import get_connection_from_env
from xagent.web.models.database import get_session_local
from xagent.web.models.uploaded_file import UploadedFile

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 500
_MAX_BACKFILL_ITERATIONS = 10000  # Safety cap to prevent pathological infinite loops
_migration_lock = threading.Lock()


def _get_migration_lock_file_path() -> str:
    lock_file = os.environ.get("LANCEDB_MIGRATION_LOCK_FILE")
    if lock_file:
        return lock_file

    lancedb_dir = os.environ.get("LANCEDB_DIR")
    if lancedb_dir:
        return os.path.join(lancedb_dir, ".lancedb_uploaded_file_links.lock")

    return os.path.join(
        tempfile.gettempdir(), "xagent_lancedb_uploaded_file_links.lock"
    )


def _acquire_file_lock() -> Any | None:
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
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    finally:
        lock_file.close()


def _extract_user_id_from_source_path(source_path: str) -> Optional[int]:
    match = re.search(r"/user_(\d+)(?:/|$)", source_path)
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def _build_update_filter(row: dict[str, Any]) -> str:
    collection = escape_lancedb_string(str(row.get("collection") or ""))
    doc_id = escape_lancedb_string(str(row.get("doc_id") or ""))
    return f"collection == '{collection}' and doc_id == '{doc_id}' and file_id IS NULL"


def _resolve_or_create_uploaded_file(
    db: Session, row: dict[str, Any], dry_run: bool
) -> tuple[Optional[str], str]:
    source_path = str(row.get("source_path") or "").strip()
    if not source_path:
        return None, "missing_source_path"

    user_id_raw = row.get("user_id")
    user_id: Optional[int] = None
    if user_id_raw is not None:
        try:
            user_id = int(user_id_raw)
        except (TypeError, ValueError):
            user_id = None
    if user_id is None:
        user_id = _extract_user_id_from_source_path(source_path)
    if user_id is None:
        return None, "missing_user_id"

    existing = (
        db.query(UploadedFile)
        .filter(
            UploadedFile.user_id == user_id,
            UploadedFile.storage_path == source_path,
        )
        .first()
    )
    if existing:
        return str(existing.file_id), "matched_existing"

    source_file = Path(source_path)
    if not source_file.exists() or not source_file.is_file():
        return None, "missing_file_on_disk"

    if dry_run:
        return "__DRY_RUN_NEW_FILE_ID__", "would_create_uploaded_file"

    record = UploadedFile(
        user_id=user_id,
        filename=source_file.name,
        storage_path=source_path,
        mime_type=mimetypes.guess_type(source_file.name)[0]
        or "application/octet-stream",
        file_size=source_file.stat().st_size,
    )
    db.add(record)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raced = (
            db.query(UploadedFile)
            .filter(
                UploadedFile.user_id == user_id,
                UploadedFile.storage_path == source_path,
            )
            .first()
        )
        if raced:
            return str(raced.file_id), "matched_existing_after_race"
        return None, "uploaded_file_integrity_error"

    db.refresh(record)
    return str(record.file_id), "created_uploaded_file"


def backfill_documents_file_links(
    *,
    dry_run: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    conn: DBConnection | None = None,
) -> dict[str, Any]:
    if conn is None:
        conn = get_connection_from_env()

    ensure_documents_table(conn)
    docs_table = conn.open_table("documents")

    SessionLocal = get_session_local()
    db = SessionLocal()

    stats: dict[str, Any] = {
        "dry_run": dry_run,
        "scanned": 0,
        "backfilled_by_match": 0,
        "backfilled_by_create": 0,
        "unbackfillable": 0,
        "failures": 0,
        "unbackfillable_samples": [],
        "iterations": 0,
    }

    try:
        while stats["iterations"] < _MAX_BACKFILL_ITERATIONS:
            stats["iterations"] += 1
            rows = query_to_list(
                docs_table.search().where("file_id IS NULL").limit(batch_size)
            )
            if not rows:
                break

            updated_in_batch = 0
            for row in rows:
                stats["scanned"] += 1
                file_id, reason = _resolve_or_create_uploaded_file(db, row, dry_run)
                if file_id is None:
                    stats["unbackfillable"] += 1
                    if len(stats["unbackfillable_samples"]) < 20:
                        stats["unbackfillable_samples"].append(
                            {
                                "collection": row.get("collection"),
                                "doc_id": row.get("doc_id"),
                                "source_path": row.get("source_path"),
                                "reason": reason,
                            }
                        )
                    continue

                if reason in {"matched_existing", "matched_existing_after_race"}:
                    stats["backfilled_by_match"] += 1
                elif reason in {"created_uploaded_file", "would_create_uploaded_file"}:
                    stats["backfilled_by_create"] += 1

                if dry_run:
                    continue

                try:
                    update_filter = _build_update_filter(row)
                    docs_table.update(update_filter, {"file_id": file_id})
                    updated_in_batch += 1
                except Exception as exc:  # noqa: BLE001
                    stats["failures"] += 1
                    logger.warning(
                        "Failed to update documents.file_id for %s: %s",
                        row.get("doc_id"),
                        exc,
                    )

            if dry_run:
                # Dry-run does not mutate table rows; avoid reprocessing same batch forever.
                break
            if updated_in_batch == 0:
                # No rows were updated in this batch, so additional loops will not make progress.
                break
        else:
            # Loop terminated due to reaching max iterations (pathological case)
            logger.warning(
                "Backfill reached maximum iteration limit (%d). "
                "This may indicate a pathological condition or alternating success/failure pattern.",
                _MAX_BACKFILL_ITERATIONS,
            )
            stats["hit_iteration_limit"] = True
    finally:
        db.close()

    return stats


def backfill_all(
    *, dry_run: bool = False, batch_size: int = DEFAULT_BATCH_SIZE
) -> dict[str, Any]:
    if not _migration_lock.acquire(blocking=False):
        logger.warning("Another migration is already in progress")
        return {"error": "Migration lock already held"}

    file_lock = None
    try:
        file_lock = _acquire_file_lock()
        if file_lock is None:
            logger.warning("Another migration is running in a different process")
            return {"error": "Migration file lock already held"}

        result = backfill_documents_file_links(dry_run=dry_run, batch_size=batch_size)
        logger.info("Backfill uploaded file links result: %s", result)
        return result
    finally:
        if file_lock is not None:
            _release_file_lock(file_lock)
        _migration_lock.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Backfill LanceDB documents.file_id using uploaded_files records.\n"
            "Supports dry-run mode and idempotent re-runs."
        )
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Simulate without writing data"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Rows processed per batch (default: {DEFAULT_BATCH_SIZE})",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        logger.error("--batch-size must be > 0")
        sys.exit(2)

    try:
        backfill_all(dry_run=args.dry_run, batch_size=args.batch_size)
        sys.exit(0)
    except Exception as exc:  # noqa: BLE001
        logger.error("Backfill migration failed: %s", exc, exc_info=True)
        sys.exit(2)
