"""Helpers for bridging KB document metadata and uploaded file records."""

from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from sqlalchemy.orm import Session

from ...config import get_uploads_dir
from ...core.tools.core.RAG_tools.management.status import load_ingestion_status
from ...core.tools.core.RAG_tools.storage.contracts import (
    DocumentRecord,
    VectorIndexStore,
)
from ...core.tools.core.RAG_tools.storage.factory import get_vector_index_store
from ...core.tools.core.RAG_tools.utils.user_scope import resolve_user_scope
from ...core.tools.core.RAG_tools.version_management.cascade_cleaner import (
    cascade_delete,
)
from ..models.uploaded_file import UploadedFile

logger = logging.getLogger(__name__)

_FILE_STATUS_BATCH_SIZE = 200
_STALE_FILE_STATUSES = {"FAILED", "UNKNOWN", "RUNNING"}
_DEFAULT_DELETABLE_STALE_STATUSES = {"FAILED"}


class _FileStatusCache:
    """Simple TTL cache for file status aggregation results.

    Caches status maps keyed by (user_id, file_ids_tuple) to avoid
    repeated vector store queries for the same set of files within a short window.
    """

    def __init__(self, ttl_seconds: int = 5, maxsize: int = 1024) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        self._cache: OrderedDict[
            tuple[int, tuple[str, ...]], tuple[Dict[str, str], float]
        ] = OrderedDict()
        self._ttl = ttl_seconds
        self._maxsize = maxsize

    def get(self, user_id: int, file_ids: List[str]) -> Optional[Dict[str, str]]:
        key = (user_id, tuple(sorted(file_ids)))
        if key in self._cache:
            result, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                self._cache.move_to_end(key)
                return result
            # Expired, remove
            del self._cache[key]
        return None

    def put(self, user_id: int, file_ids: List[str], result: Dict[str, str]) -> None:
        key = (user_id, tuple(sorted(file_ids)))
        self._cache[key] = (result, time.time())
        self._cache.move_to_end(key)
        while len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

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


def _list_document_records_for_file_ids(
    store: VectorIndexStore,
    *,
    file_ids: Sequence[str],
    user_id: int,
    is_admin: bool,
) -> List[DocumentRecord]:
    """Load document rows for many file IDs without scan-limit truncation."""
    unique_ids = sorted({file_id for file_id in file_ids if file_id})
    if not unique_ids:
        return []

    records: List[DocumentRecord] = []
    for offset in range(0, len(unique_ids), _FILE_STATUS_BATCH_SIZE):
        batch = unique_ids[offset : offset + _FILE_STATUS_BATCH_SIZE]
        records.extend(
            store.list_document_records(
                collection_name=None,
                user_id=user_id,
                is_admin=is_admin,
                file_ids=batch,
                max_results=-1,
            )
        )
    return records


def upsert_uploaded_file_record(
    db: Session,
    *,
    user_id: Optional[int],
    filename: str,
    storage_path: Path,
    mime_type: Optional[str],
    file_size: int,
) -> UploadedFile:
    """Create or refresh an ``UploadedFile`` row for a stored file."""
    scope = resolve_user_scope(user_id=user_id, is_admin=False)
    if scope.user_id is None:
        raise ValueError("user_id is required for UploadedFile upsert")

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
            user_id=scope.user_id,
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
    _file_status_cache.invalidate_user(scope.user_id)

    return file_record


def list_documents_for_user(
    *,
    user_id: Optional[int] = None,
    is_admin: bool,
    collection_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load KB document metadata rows for a user."""
    records = get_vector_index_store().list_document_records(
        collection_name=collection_name,
        user_id=user_id,
        is_admin=is_admin,
        max_results=10000,
    )
    return [_document_record_to_dict(record) for record in records]


def _document_record_to_dict(
    record: Union[Dict[str, Any], DocumentRecord],
) -> Dict[str, Any]:
    """Convert a document record projection to the legacy dict shape."""
    if isinstance(record, dict):
        return dict(record)
    return {
        "collection": record.collection,
        "doc_id": record.doc_id,
        "file_id": record.file_id,
        "source_path": record.source_path,
    }


def build_uploaded_filename_map(
    db: Session, *, user_id: Optional[int], file_ids: List[str]
) -> Dict[str, str]:
    """Resolve ``file_id`` values to current uploaded filenames."""
    scope = resolve_user_scope(user_id=user_id, is_admin=False)
    if scope.user_id is None:
        return {}

    normalized_file_ids = sorted({file_id for file_id in file_ids if file_id})
    if not normalized_file_ids:
        return {}
    records = (
        db.query(UploadedFile)
        .filter(
            UploadedFile.user_id == scope.user_id,
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
    user_id: Optional[int],
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
    scope = resolve_user_scope(user_id=user_id, is_admin=False)
    if scope.user_id is None:
        return False

    if not file_id or file_id in remaining_file_ids:
        return False

    file_record = (
        db.query(UploadedFile)
        .filter(
            UploadedFile.user_id == scope.user_id,
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
    _file_status_cache.invalidate_user(scope.user_id)

    return True


def _load_indexed_doc_refs(
    store: VectorIndexStore,
    *,
    doc_ids_by_collection: Dict[str, set[str]],
    user_id: Optional[int],
    is_admin: bool,
) -> set[tuple[str, str]]:
    """Return document refs that have searchable artifacts in the vector store.

    Legacy deployments may not have ingestion status rows. If chunks or
    embeddings exist for a document, the file was already indexed enough to be
    user-visible and should be treated as successful for file-list status.
    """
    candidate_refs = {
        (collection, doc_id)
        for collection, doc_ids in doc_ids_by_collection.items()
        for doc_id in doc_ids
    }
    if not candidate_refs:
        return set()

    indexed_refs: set[tuple[str, str]] = set()
    candidate_tables = ["chunks"] + [
        t for t in store.list_table_names() if t.startswith("embeddings_")
    ]

    for table_name in candidate_tables:
        if indexed_refs.issuperset(candidate_refs):
            return indexed_refs
        for collection, doc_ids in doc_ids_by_collection.items():
            pending = doc_ids - {d for c, d in indexed_refs if c == collection}
            if not pending:
                continue
            try:
                counts = store.aggregate_document_counts(
                    table_name,
                    "doc_id",
                    collection,
                    user_id=user_id,
                    is_admin=is_admin,
                )
            except Exception:  # noqa: BLE001
                logger.debug(
                    "Skipping indexed status fallback table '%s' for collection '%s'",
                    table_name,
                    collection,
                )
                continue
            for doc_id, count in counts.items():
                if count > 0 and doc_id in pending:
                    indexed_refs.add((collection, doc_id))

    return indexed_refs


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

    # Cache miss - compute from database via abstraction layer
    store = get_vector_index_store()
    records = _list_document_records_for_file_ids(
        store,
        file_ids=normalized_file_ids,
        user_id=user_id,
        is_admin=is_admin,
    )

    doc_refs_by_file_id: Dict[str, List[tuple[str, str]]] = {
        file_id: [] for file_id in normalized_file_ids
    }
    for record in records:
        file_id = (record.file_id or "").strip()
        collection = (record.collection or "").strip()
        doc_id = record.doc_id.strip()
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

    doc_ids_by_collection: Dict[str, set[str]] = {}
    for doc_refs in doc_refs_by_file_id.values():
        for collection, doc_id in doc_refs:
            doc_ids_by_collection.setdefault(collection, set()).add(doc_id)

    indexed_doc_refs = _load_indexed_doc_refs(
        store,
        doc_ids_by_collection=doc_ids_by_collection,
        user_id=user_id,
        is_admin=is_admin,
    )

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
        if any(
            (collection, doc_id) in indexed_doc_refs for collection, doc_id in doc_refs
        ):
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
    deletable_statuses: Optional[set[str]] = None,
) -> Dict[str, int]:
    """Reconcile uploaded files with document + ingestion status state.

    Unknown and running statuses are intentionally report-only by default.
    Historical deployments may lack complete ``documents.file_id`` or
    ``ingestion_runs`` rows, so treating UNKNOWN/RUNNING as deletable would
    turn migration gaps into user data loss.

    The caller owns the SQL transaction boundary. This helper flushes its own
    UploadedFile deletes but does not commit the passed session.
    """
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
    effective_deletable_statuses = {
        status.upper()
        for status in (
            deletable_statuses
            if deletable_statuses is not None
            else _DEFAULT_DELETABLE_STALE_STATUSES
        )
    }
    store = get_vector_index_store()
    for record in uploaded_files:
        scanned += 1
        file_id = str(record.file_id)
        status = status_map.get(file_id, "UNKNOWN").upper()
        if status not in _STALE_FILE_STATUSES:
            continue

        created_at = getattr(record, "created_at", None)
        if created_at is not None and getattr(created_at, "tzinfo", None) is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        if created_at is not None and created_at > cutoff:
            continue

        if status == "RUNNING":
            logger.warning(
                "Found stale RUNNING file (possible crashed ingestion): file_id=%s, created_at=%s",
                file_id,
                created_at,
            )

        stale_candidates += 1
        if not delete_stale:
            continue
        if status not in effective_deletable_statuses:
            logger.warning(
                "Preserving stale UploadedFile with non-deletable status: "
                "file_id=%s, status=%s, created_at=%s",
                file_id,
                status,
                created_at,
            )
            continue

        try:
            doc_records = store.list_document_records(
                collection_name=None,
                user_id=user_id,
                is_admin=is_admin,
                file_ids=[file_id],
            )
        except Exception as exc:  # noqa: BLE001
            cleanup_errors += 1
            logger.error(
                "Failed to query documents for stale file_id=%s: %s",
                file_id,
                exc,
            )
            continue

        cascade_deleted = 0
        cascade_error = False
        for doc_rec in doc_records:
            collection = (doc_rec.collection or "").strip()
            doc_id = doc_rec.doc_id.strip()
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

    if deleted > 0:
        db.flush()

    return {
        "scanned": scanned,
        "stale_candidates": stale_candidates,
        "deleted": deleted,
        "cleanup_errors": cleanup_errors,
    }
