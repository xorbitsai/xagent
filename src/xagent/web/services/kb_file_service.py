"""Helpers for bridging KB document metadata and uploaded file records."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from sqlalchemy.orm import Session

from ...config import get_uploads_dir
from ...core.tools.core.RAG_tools.LanceDB.schema_manager import ensure_documents_table
from ...core.tools.core.RAG_tools.management.status import load_ingestion_status
from ...core.tools.core.RAG_tools.storage.contracts import DocumentRecord
from ...core.tools.core.RAG_tools.utils.lancedb_query_utils import query_to_list
from ...core.tools.core.RAG_tools.utils.string_utils import (
    build_lancedb_filter_expression,
    escape_lancedb_string,
)
from ...core.tools.core.RAG_tools.utils.user_permissions import UserPermissions
from ...core.tools.core.RAG_tools.version_management.cascade_cleaner import (
    cascade_delete,
)
from ...providers.vector_store.lancedb import get_connection_from_env
from ..models.uploaded_file import UploadedFile

logger = logging.getLogger(__name__)

_FILE_STATUS_BATCH_SIZE = 200


class _FileStatusCache:
    """Simple TTL cache for file status aggregation results.

    Caches status maps keyed by (user_id, file_ids_tuple) to avoid
    repeated LanceDB queries for the same set of files within a short window.
    """

    def __init__(self, ttl_seconds: int = 5) -> None:
        self._cache: Dict[
            tuple[int, tuple[str, ...]], tuple[Dict[str, str], float]
        ] = {}
        self._ttl = ttl_seconds

    def get(self, user_id: int, file_ids: List[str]) -> Optional[Dict[str, str]]:
        key = (user_id, tuple(sorted(file_ids)))
        if key in self._cache:
            result, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                return result
            # Expired, remove
            del self._cache[key]
        return None

    def put(self, user_id: int, file_ids: List[str], result: Dict[str, str]) -> None:
        key = (user_id, tuple(sorted(file_ids)))
        self._cache[key] = (result, time.time())

    def invalidate_user(self, user_id: int) -> None:
        """Remove all cached entries for a specific user."""
        keys_to_delete = [k for k in self._cache if k[0] == user_id]
        for key in keys_to_delete:
            del self._cache[key]

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()


# Global cache instance
_file_status_cache = _FileStatusCache(ttl_seconds=5)


def upsert_uploaded_file_record(
    db: Session,
    *,
    user_id: int,
    filename: str,
    storage_path: Path,
    mime_type: Optional[str],
    file_size: int,
) -> UploadedFile:
    """Create or refresh an ``UploadedFile`` row for a stored file."""
    storage_path_str = str(storage_path)
    existing = (
        db.query(UploadedFile)
        .filter(UploadedFile.storage_path == storage_path_str)
        .first()
    )
    if existing:
        existing.filename = filename  # type: ignore[assignment]
        existing.file_size = int(file_size)  # type: ignore[assignment]
        if mime_type is not None:
            existing.mime_type = mime_type  # type: ignore[assignment]
        db.flush()
        file_record = existing
    else:
        file_record = UploadedFile(
            user_id=user_id,
            filename=filename,
            storage_path=storage_path_str,
            mime_type=mime_type,
            file_size=int(file_size),
        )
        db.add(file_record)
        db.flush()
    db.commit()
    db.refresh(file_record)

    # Invalidate cache for this user since file list may have changed
    _file_status_cache.invalidate_user(user_id)

    return file_record


def list_documents_for_user(
    *,
    user_id: int,
    is_admin: bool,
    collection_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load KB document metadata rows for a user."""
    conn = get_connection_from_env()
    ensure_documents_table(conn)
    table = conn.open_table("documents")

    base_filter = ""
    if collection_name:
        base_filter = build_lancedb_filter_expression({"collection": collection_name})
    user_filter = UserPermissions.get_user_filter(user_id, is_admin=is_admin)
    combined_filter = (
        f"({base_filter}) and ({user_filter})"
        if user_filter and base_filter
        else (user_filter or base_filter)
    )
    query = table.search()
    if combined_filter:
        query = query.where(combined_filter)
    return query_to_list(query.limit(10000))


def build_uploaded_filename_map(
    db: Session, *, user_id: int, file_ids: List[str]
) -> Dict[str, str]:
    """Resolve ``file_id`` values to current uploaded filenames."""
    normalized_file_ids = sorted({file_id for file_id in file_ids if file_id})
    if not normalized_file_ids:
        return {}
    records = (
        db.query(UploadedFile)
        .filter(
            UploadedFile.user_id == user_id,
            UploadedFile.file_id.in_(normalized_file_ids),
        )
        .all()
    )
    return {str(record.file_id): str(record.filename) for record in records}


def get_document_record_file_id(
    record: Union[Dict[str, Any], DocumentRecord],
) -> Optional[str]:
    """Extract a normalized ``file_id`` from a KB document record.

    Args:
        record: Either a Dict[str, Any] or DocumentRecord dataclass.

    Returns:
        Normalized file_id string or None.
    """
    # Handle both Dict and DocumentRecord types
    if isinstance(record, dict):
        raw_file_id = record.get("file_id")
    else:
        # Assume DocumentRecord dataclass with file_id attribute
        raw_file_id = getattr(record, "file_id", None)

    if raw_file_id is None:
        return None
    file_id = str(raw_file_id).strip()
    return file_id or None


def resolve_document_filename(
    record: Union[Dict[str, Any], DocumentRecord], filename_map: Dict[str, str]
) -> Optional[str]:
    """Resolve a user-facing filename from ``file_id`` first, then legacy path.

    Args:
        record: Either a Dict[str, Any] or DocumentRecord dataclass.
        filename_map: Mapping from file_id to filename.

    Returns:
        Resolved filename or None.
    """
    file_id = get_document_record_file_id(record)
    if file_id and filename_map.get(file_id):
        return filename_map[file_id]

    # Handle both Dict and DocumentRecord types for source_path
    if isinstance(record, dict):
        source_path = record.get("source_path")
    else:
        source_path = getattr(record, "source_path", None)

    if source_path:
        return os.path.basename(str(source_path))

    return None


def delete_uploaded_file_if_orphaned(
    db: Session,
    *,
    file_id: str,
    user_id: int,
    remaining_file_ids: set[str],
) -> bool:
    """Delete uploaded file row and local file when no documents still reference it.

    Args:
        db: Database session.
        file_id: The ID of the file to check.
        user_id: User ID for scoping.
        remaining_file_ids: A set of all file_id values still referenced by other documents.

    Returns:
        True if the file was deleted, False otherwise.
    """
    if not file_id or file_id in remaining_file_ids:
        return False

    file_record = (
        db.query(UploadedFile)
        .filter(
            UploadedFile.user_id == user_id,
            UploadedFile.file_id == file_id,
        )
        .first()
    )
    if file_record is None:
        return False

    uploads_root = get_uploads_dir().resolve()
    file_path = Path(str(file_record.storage_path))
    try:
        resolved_path = file_path.resolve()
        resolved_path.relative_to(uploads_root)
    except ValueError:
        logger.warning(
            "Skipping physical delete for file outside uploads root: %s",
            file_path,
        )
    else:
        if resolved_path.exists() and resolved_path.is_file():
            resolved_path.unlink()
            logger.info("Deleted orphaned physical file: %s", resolved_path)

    db.delete(file_record)
    db.flush()

    # Invalidate cache for this user since file list changed
    _file_status_cache.invalidate_user(user_id)

    return True


def _build_file_id_in_filter(file_ids: List[str]) -> str:
    escaped_ids = [f"'{escape_lancedb_string(file_id)}'" for file_id in file_ids]
    return f"file_id IN ({', '.join(escaped_ids)})"


def _combine_lancedb_filters(
    base_filter: Optional[str], user_filter: Optional[str]
) -> Optional[str]:
    if base_filter and user_filter:
        return f"({base_filter}) and ({user_filter})"
    return base_filter or user_filter


def aggregate_uploaded_file_statuses(
    *,
    file_ids: List[str],
    user_id: int,
    is_admin: bool,
    use_cache: bool = True,
) -> Dict[str, str]:
    """Aggregate file status by joining documents + ingestion status records.

    Args:
        file_ids: List of file IDs to get status for
        user_id: User ID for permission filtering
        is_admin: Whether user has admin privileges
        use_cache: Whether to use the in-memory cache (default: True)

    Returns:
        Dictionary mapping file_id to status (RUNNING, SUCCESS, FAILED, UNKNOWN)
    """
    normalized_file_ids = sorted({file_id for file_id in file_ids if file_id})
    if not normalized_file_ids:
        return {}

    # Check cache first
    if use_cache:
        cached_result = _file_status_cache.get(user_id, normalized_file_ids)
        if cached_result is not None:
            return cached_result

    # Cache miss - compute from database
    conn = get_connection_from_env()
    ensure_documents_table(conn)
    documents_table = conn.open_table("documents")
    user_filter = UserPermissions.get_user_filter(user_id, is_admin=is_admin)

    doc_refs_by_file_id: Dict[str, List[tuple[str, str]]] = {
        file_id: [] for file_id in normalized_file_ids
    }
    for offset in range(0, len(normalized_file_ids), _FILE_STATUS_BATCH_SIZE):
        batch = normalized_file_ids[offset : offset + _FILE_STATUS_BATCH_SIZE]
        base_filter = _build_file_id_in_filter(batch)
        combined_filter = _combine_lancedb_filters(base_filter, user_filter)

        query = documents_table.search()
        if combined_filter:
            query = query.where(combined_filter)
        rows = query_to_list(
            query.select(["file_id", "collection", "doc_id"]).limit(-1)
        )
        for row in rows:
            file_id = str(row.get("file_id") or "").strip()
            collection = str(row.get("collection") or "").strip()
            doc_id = str(row.get("doc_id") or "").strip()
            if file_id and collection and doc_id and file_id in doc_refs_by_file_id:
                doc_refs_by_file_id[file_id].append((collection, doc_id))

    collections = sorted(
        {
            collection
            for doc_refs in doc_refs_by_file_id.values()
            for collection, _ in doc_refs
        }
    )
    status_by_doc: Dict[tuple[str, str], str] = {}
    for collection in collections:
        for entry in load_ingestion_status(
            collection=collection,
            user_id=user_id,
            is_admin=is_admin,
        ):
            doc_id = str(entry.get("doc_id") or "").strip()
            status = str(entry.get("status") or "").strip().lower()
            if doc_id and status:
                status_by_doc[(collection, doc_id)] = status

    status_map: Dict[str, str] = {}
    for file_id, doc_refs in doc_refs_by_file_id.items():
        if not doc_refs:
            status_map[file_id] = "UNKNOWN"
            continue

        statuses = [
            status_by_doc.get((collection, doc_id), "")
            for collection, doc_id in doc_refs
        ]
        if any(status == "running" for status in statuses):
            status_map[file_id] = "RUNNING"
            continue

        has_failed = any(status == "failed" for status in statuses)
        has_success = any(status == "success" for status in statuses)
        if has_failed and not has_success:
            status_map[file_id] = "FAILED"
            continue
        if has_success:
            status_map[file_id] = "SUCCESS"
            continue
        status_map[file_id] = "UNKNOWN"

    # Store in cache for future requests
    if use_cache:
        _file_status_cache.put(user_id, normalized_file_ids, status_map)

    return status_map


def reconcile_uploaded_files(
    db: Session,
    *,
    user_id: int,
    is_admin: bool,
    stale_ttl_hours: int = 24 * 7,
    delete_stale: bool = True,
) -> Dict[str, int]:
    """Reconcile uploaded files with document + ingestion status state."""
    query = db.query(UploadedFile)
    if not is_admin:
        query = query.filter(UploadedFile.user_id == user_id)

    uploaded_files = query.order_by(UploadedFile.created_at.asc()).all()
    file_ids = [str(record.file_id) for record in uploaded_files if record.file_id]
    status_map = aggregate_uploaded_file_statuses(
        file_ids=file_ids,
        user_id=user_id,
        is_admin=is_admin,
    )

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max(stale_ttl_hours, 1))
    scanned = 0
    deleted = 0
    stale_candidates = 0
    cleanup_errors = 0
    conn = get_connection_from_env()
    ensure_documents_table(conn)
    documents_table = conn.open_table("documents")
    for record in uploaded_files:
        scanned += 1
        file_id = str(record.file_id)
        status = status_map.get(file_id, "UNKNOWN")
        if status not in {"FAILED", "UNKNOWN", "RUNNING"}:
            continue

        created_at = getattr(record, "created_at", None)
        if created_at is not None and getattr(created_at, "tzinfo", None) is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        if created_at is not None and created_at > cutoff:
            continue

        # Log warning for RUNNING files as they may indicate crashed ingestion
        if status == "RUNNING":
            logger.warning(
                "Found stale RUNNING file (possible crashed ingestion): file_id=%s, created_at=%s",
                file_id,
                created_at,
            )

        stale_candidates += 1
        if not delete_stale:
            continue

        safe_file_id = escape_lancedb_string(file_id)
        # Query documents table to get (collection, doc_id) pairs for cascade deletion
        try:
            doc_rows = query_to_list(
                documents_table.search()
                .where(f"file_id = '{safe_file_id}'")
                .select(["collection", "doc_id"])
                .limit(-1)
            )
        except Exception as exc:  # noqa: BLE001
            cleanup_errors += 1
            logger.error(
                "Failed to query documents for stale file_id=%s: %s",
                file_id,
                exc,
            )
            continue

        # Cascade delete all related data for each (collection, doc_id) pair
        # Note: We use cascade_delete for complete cleanup across all tables
        # (parses, chunks, embeddings_*, main_pointers, ingestion_runs, documents)
        cascade_deleted = 0
        cascade_error = False
        for row in doc_rows:
            collection = str(row.get("collection") or "").strip()
            doc_id = str(row.get("doc_id") or "").strip()
            if not collection or not doc_id:
                continue

            try:
                deleted_counts = cascade_delete(
                    target="document",
                    collection=collection,
                    doc_id=doc_id,
                    user_id=user_id,
                    is_admin=is_admin,
                    preview_only=False,
                    confirm=True,
                )
                cascade_deleted += sum(int(v) for v in deleted_counts.values())
                logger.info(
                    "Cascade deleted %d rows for stale document: collection=%s, doc_id=%s, file_id=%s",
                    sum(deleted_counts.values()),
                    collection,
                    doc_id,
                    file_id,
                )
            except Exception as exc:  # noqa: BLE001
                cascade_error = True
                cleanup_errors += 1
                logger.error(
                    "Failed to cascade delete for stale document: collection=%s, doc_id=%s, file_id=%s: %s",
                    collection,
                    doc_id,
                    file_id,
                    exc,
                )

        # If cascade delete failed, skip deleting the UploadedFile record
        # to maintain consistency (file record still references the documents)
        if cascade_error:
            logger.warning(
                "Skipping UploadedFile deletion due to cascade delete errors: file_id=%s",
                file_id,
            )
            continue

        # After relational/vector cleanup succeeds, delete physical file.
        file_path = Path(str(record.storage_path))
        uploads_root = get_uploads_dir().resolve()
        try:
            resolved_path = file_path.resolve()
            resolved_path.relative_to(uploads_root)
        except ValueError:
            logger.warning(
                "Skipping stale file cleanup outside uploads root: %s",
                file_path,
            )
        else:
            if resolved_path.exists() and resolved_path.is_file():
                try:
                    resolved_path.unlink()
                except OSError as exc:
                    cleanup_errors += 1
                    logger.error(
                        "Failed to delete stale file %s for file_id=%s: %s",
                        resolved_path,
                        file_id,
                        exc,
                    )
                    continue

        # Finally delete the UploadedFile record
        db.delete(record)
        deleted += 1
        logger.info(
            "Deleted stale UploadedFile record: file_id=%s (cascade deleted %d related rows)",
            file_id,
            cascade_deleted,
        )

    if delete_stale and deleted > 0:
        db.commit()

    return {
        "scanned": scanned,
        "stale_candidates": stale_candidates,
        "deleted": deleted,
        "cleanup_errors": cleanup_errors,
    }
