"""Helpers for bridging KB document metadata and uploaded file records."""

from __future__ import annotations

import logging
import os
import shutil
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Union

from sqlalchemy.orm import Session

from ...config import get_uploads_dir
from ...core.tools.core.RAG_tools.management.status import (
    _load_ingestion_status_impl,
)
from ...core.tools.core.RAG_tools.storage.contracts import DocumentRecord
from ...core.tools.core.RAG_tools.storage.factory import get_vector_index_store
from ...core.tools.core.RAG_tools.utils.user_permissions import UserPermissions
from ...core.tools.core.RAG_tools.utils.user_scope import resolve_user_scope
from ..models.uploaded_file import UploadedFile
from .uploaded_file_store import UploadedFileStore, UploadedFileVersionConflict

if TYPE_CHECKING:
    from ...core.tools.core.RAG_tools.kb import KBFileCompatibilityFacade

logger = logging.getLogger(__name__)

_STALE_FILE_STATUSES = {"FAILED", "UNKNOWN", "RUNNING"}
_DEFAULT_DELETABLE_STALE_STATUSES = {"FAILED"}
KB_RETAINED_DIR = ".kb-retained"


def _get_file_compatibility_facade() -> "KBFileCompatibilityFacade":
    from ...core.tools.core.RAG_tools.kb import get_kb_coordinator

    return get_kb_coordinator().file_compatibility


@dataclass(frozen=True)
class FileCompensationResult:
    """Result for idempotent file compensation helpers."""

    status: str
    side_effects_may_remain: bool = False
    errors: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return self.status == "complete" and not self.side_effects_may_remain


@dataclass(frozen=True)
class UploadedFileRefreshSnapshot:
    """Previous UploadedFile state captured before refreshing a durable file."""

    file_id: str
    user_id: Optional[int]
    row_fields: Dict[str, Any]
    previous_path: Path
    backup_path: Optional[Path]
    had_local_file: bool
    reindex_marker_applied: bool = False


class _FileStatusCache:
    """Simple TTL cache for file status aggregation results.

    Caches status maps keyed by (user_id, file_ids_tuple) to avoid
    repeated LanceDB queries for the same set of files within a short window.
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


def _upsert_uploaded_file_record_impl(
    db: Session,
    *,
    user_id: Optional[int],
    filename: str,
    storage_path: Path,
    mime_type: Optional[str],
    file_size: int,
    file_id: Optional[str] = None,
) -> UploadedFile:
    """Create or refresh an ``UploadedFile`` row for a stored file."""
    scope = resolve_user_scope(user_id=user_id, is_admin=False)
    if scope.user_id is None:
        raise ValueError("user_id is required for UploadedFile upsert")

    file_record = UploadedFileStore(db).upsert_by_storage_path(
        user_id=scope.user_id,
        filename=filename,
        storage_path=storage_path,
        mime_type=mime_type,
        file_size=file_size,
        file_id=file_id,
    )
    db.commit()
    db.refresh(file_record)

    # Invalidate cache for this user since file list may have changed
    _file_status_cache.invalidate_user(scope.user_id)

    return file_record


def _build_uploaded_filename_map_impl(
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


def _get_document_record_file_id_impl(
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


def _resolve_document_filename_impl(
    record: Union[Dict[str, Any], DocumentRecord], filename_map: Dict[str, str]
) -> Optional[str]:
    """Resolve a user-facing filename from ``file_id`` first, then legacy path.

    Args:
        record: Either a Dict[str, Any] or DocumentRecord dataclass.
        filename_map: Mapping from file_id to filename.

    Returns:
        Resolved filename or None.
    """
    file_id = _get_document_record_file_id_impl(record)
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


def _delete_uploaded_file_if_orphaned_impl(
    db: Session,
    *,
    file_id: str,
    user_id: Optional[int],
    remaining_file_ids: set[str],
    after_commit: Optional[List[tuple[str, Callable[[], None]]]] = None,
) -> bool:
    """Delete an uploaded file's row and bytes when no documents still reference it.

    Never commits. With ``after_commit``, the durable delete and local unlink are
    queued for the caller to run after it commits; without it, the bytes go at once.

    Args:
        db: Database session.
        file_id: The ID of the file to check.
        user_id: Requester; only their rows are freed, except rows collection
            delete retained under ``.kb-retained``, which go whoever owns them.
        remaining_file_ids: file_id values still referenced by other documents; must include every candidate that still is.

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
            UploadedFile.file_id == file_id,
            UploadedFile.storage_status != "compensating",
        )
        .first()
    )
    if file_record is None:
        return False
    retained_dir = Path(str(file_record.storage_path)).parent
    # Retained rows exist only for surviving references; any requester's last one frees them.
    if int(file_record.user_id) != scope.user_id and not (
        retained_dir.name == file_id and retained_dir.parent.name == KB_RETAINED_DIR
    ):
        return False

    uploads_root = get_uploads_dir().resolve()
    file_path = Path(str(file_record.storage_path))
    delete_local = False
    try:
        resolved_path = file_path.resolve()
        resolved_path.relative_to(uploads_root)
    except ValueError:
        logger.warning(
            "Skipping physical delete for file outside uploads root: %s",
            file_path,
        )
    else:
        delete_local = True

    try:
        UploadedFileStore(db).delete(
            file_record,
            delete_local=delete_local,
            local_root=uploads_root if delete_local else None,
            after_commit=after_commit,
        )
    except UploadedFileVersionConflict:
        # A cleanup claimant or replacement generation won after the KB
        # reference check. It owns the row and every resource from here.
        return False
    # Invalidate cache for this user since file list changed
    _file_status_cache.invalidate_user(int(file_record.user_id))
    return True


def _compensation_complete(*effects: str) -> FileCompensationResult:
    return FileCompensationResult(status="complete", effects=tuple(effects))


def _compensation_incomplete(
    error: BaseException | str,
    *,
    effects: tuple[str, ...] = (),
) -> FileCompensationResult:
    return FileCompensationResult(
        status="incomplete",
        side_effects_may_remain=True,
        errors=(str(error),),
        effects=effects,
    )


def _cleanup_local_copied_file_impl(
    *,
    file_path: Path,
    local_root: Optional[Path] = None,
) -> FileCompensationResult:
    """Idempotently remove a staged local file created for KB ingestion."""
    try:
        resolved_path = file_path.resolve()
        if local_root is not None:
            resolved_root = local_root.resolve()
            try:
                resolved_path.relative_to(resolved_root)
            except ValueError:
                return _compensation_incomplete(
                    f"refusing to delete file outside local_root: {file_path}"
                )
        if not resolved_path.exists():
            return _compensation_complete("already_removed")
        if not resolved_path.is_file():
            return _compensation_incomplete(f"not a file: {file_path}")
        resolved_path.unlink()
        return _compensation_complete("local_file_removed")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to clean up copied file %s: %s", file_path, exc)
        return _compensation_incomplete(exc)


_UPLOADED_FILE_REFRESH_FIELDS = (
    "filename",
    "storage_path",
    "mime_type",
    "file_size",
    "storage_backend",
    "storage_key",
    "storage_uri",
    "checksum",
    "etag",
    "storage_status",
    "workspace_relative_path",
    "workspace_category",
    "task_id",
)


def _capture_uploaded_file_refresh_snapshot_impl(
    file_record: UploadedFile,
    *,
    backup_path: Optional[Path] = None,
    reindex_marker_applied: bool = False,
) -> UploadedFileRefreshSnapshot:
    """Capture UploadedFile row and local-file state before a refresh."""
    row_fields = {
        field_name: getattr(file_record, field_name, None)
        for field_name in _UPLOADED_FILE_REFRESH_FIELDS
    }
    previous_path = Path(str(getattr(file_record, "storage_path")))
    raw_user_id = getattr(file_record, "user_id", None)
    user_id = int(raw_user_id) if raw_user_id is not None else None
    return UploadedFileRefreshSnapshot(
        file_id=str(getattr(file_record, "file_id")),
        user_id=user_id,
        row_fields=row_fields,
        previous_path=previous_path,
        backup_path=backup_path,
        had_local_file=previous_path.exists() and previous_path.is_file(),
        reindex_marker_applied=reindex_marker_applied,
    )


def _restore_uploaded_file_refresh_snapshot_impl(
    db: Session,
    snapshot: UploadedFileRefreshSnapshot,
) -> FileCompensationResult:
    """Restore UploadedFile row, local file, durable bytes, and cache state."""
    effects: list[str] = []
    errors: list[str] = []

    if snapshot.backup_path is not None:
        try:
            if snapshot.had_local_file:
                snapshot.previous_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(snapshot.backup_path, snapshot.previous_path)
                effects.append("local_file_restored")
            else:
                snapshot.previous_path.unlink(missing_ok=True)
                effects.append("local_file_removed")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to restore local UploadedFile backup for file_id=%s: %s",
                snapshot.file_id,
                exc,
            )
            errors.append(str(exc))

    try:
        db.rollback()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to roll back database session before UploadedFile refresh restore "
            "for file_id=%s: %s",
            snapshot.file_id,
            exc,
        )
        errors.append(str(exc))
        return FileCompensationResult(
            status="incomplete",
            side_effects_may_remain=True,
            errors=tuple(errors),
            effects=tuple(effects),
        )

    try:
        query = db.query(UploadedFile).filter(UploadedFile.file_id == snapshot.file_id)
        if snapshot.user_id is not None:
            query = query.filter(UploadedFile.user_id == snapshot.user_id)
        file_record = query.first()
        if file_record is None:
            errors.append(
                f"UploadedFile missing during refresh restore: {snapshot.file_id}"
            )
            return FileCompensationResult(
                status="incomplete",
                side_effects_may_remain=True,
                errors=tuple(errors),
                effects=tuple(effects),
            )

        changed_fields = tuple(
            field_name
            for field_name, value in snapshot.row_fields.items()
            if getattr(file_record, field_name, None) != value
        )
        if changed_fields:
            current_status = str(getattr(file_record, "storage_status", "unknown"))
            errors.append(
                "UploadedFile changed after the refresh snapshot; skipped unsafe "
                f"restore (status={current_status}, fields={','.join(changed_fields)})"
            )
            return FileCompensationResult(
                status="incomplete",
                side_effects_may_remain=True,
                errors=tuple(errors),
                effects=tuple(effects),
            )
        effects.append("uploaded_file_row_unchanged")

        if snapshot.user_id is not None:
            _file_status_cache.invalidate_user(snapshot.user_id)
            effects.append("file_status_cache_invalidated")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to restore UploadedFile refresh snapshot for file_id=%s: %s",
            snapshot.file_id,
            exc,
        )
        errors.append(str(exc))
        return FileCompensationResult(
            status="incomplete",
            side_effects_may_remain=True,
            errors=tuple(errors),
            effects=tuple(effects),
        )

    if snapshot.reindex_marker_applied:
        effects.append("reindex_marker_was_applied")
    if errors:
        return FileCompensationResult(
            status="incomplete",
            side_effects_may_remain=True,
            errors=tuple(errors),
            effects=tuple(effects),
        )
    return _compensation_complete(*effects)


def _aggregate_uploaded_file_statuses_impl(
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
    from ...core.tools.core.RAG_tools.kb.collection_handle import ledger_holds_vectors

    normalized_file_ids = sorted({file_id for file_id in file_ids if file_id})
    if not normalized_file_ids:
        return {}

    # Check cache first
    if use_cache:
        cached_result = _file_status_cache.get(user_id, normalized_file_ids)
        if cached_result is not None:
            return cached_result

    # Cache miss - compute from database
    vector_store = get_vector_index_store()
    doc_refs_by_file_id: Dict[str, List[tuple[str, str]]] = {
        file_id: [] for file_id in normalized_file_ids
    }
    for record in vector_store.list_document_records_by_file_ids(normalized_file_ids):
        if not UserPermissions.can_access_data(user_id, record.user_id, is_admin):
            continue
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
        for entry in _load_ingestion_status_impl(
            collection=collection,
            user_id=user_id,
            is_admin=is_admin,
        ):
            doc_id = str(entry.get("doc_id") or "").strip()
            status = str(entry.get("status") or "").strip().lower()
            if doc_id and status:
                status_by_doc[(collection, doc_id)] = status

    # Legacy deployments may lack status rows; indexed documents then count as SUCCESS.
    indexed_doc_refs: set[tuple[str, str]] = set()
    if ledger_holds_vectors():
        indexed_doc_refs = vector_store.list_indexed_doc_refs(
            [ref for doc_refs in doc_refs_by_file_id.values() for ref in doc_refs],
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


def _reconcile_uploaded_files_impl(
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
    from ...core.tools.core.RAG_tools.kb import get_kb_coordinator
    from ...core.tools.core.RAG_tools.kb.collection_handle import ledger_holds_vectors

    query = db.query(UploadedFile)
    if not is_admin:
        query = query.filter(UploadedFile.user_id == user_id)

    uploaded_files = query.order_by(UploadedFile.created_at.asc()).all()
    file_ids = [str(record.file_id) for record in uploaded_files if record.file_id]
    status_map = _aggregate_uploaded_file_statuses_impl(
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
            docs = get_vector_index_store().list_document_records_by_file_ids([file_id])
        except Exception as exc:  # noqa: BLE001
            cleanup_errors += 1
            logger.error(
                "Failed to query documents for stale file_id=%s: %s",
                file_id,
                exc,
            )
            continue

        # Delete each (collection, doc_id): LanceDB cascades over every table,
        # other engines go through the handle's delete_documents_data.
        cascade_deleted = 0
        cascade_error = False
        for doc in docs:
            collection = (doc.collection or "").strip()
            doc_id = doc.doc_id.strip()
            if not collection or not doc_id:
                continue

            try:
                if ledger_holds_vectors():
                    deleted_counts = get_vector_index_store().cascade_delete(
                        target="document",
                        collection=collection,
                        doc_id=doc_id,
                        user_id=user_id,
                        is_admin=is_admin,
                        preview_only=False,
                        confirm=True,
                    )
                else:
                    deleted_counts = get_kb_coordinator().delete_documents_data_sync(
                        collection, [doc_id], user_id=user_id, is_admin=is_admin
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

        try:
            UploadedFileStore(db).delete(
                record,
                delete_local=True,
                local_root=get_uploads_dir(),
            )
        except Exception as exc:  # noqa: BLE001
            cleanup_errors += 1
            logger.error(
                "Failed to delete stale UploadedFile storage for file_id=%s: %s",
                file_id,
                exc,
            )
            continue
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


def upsert_uploaded_file_record(
    db: Session,
    *,
    user_id: Optional[int],
    filename: str,
    storage_path: Path,
    mime_type: Optional[str],
    file_size: int,
    file_id: Optional[str] = None,
) -> UploadedFile:
    """Create or refresh an ``UploadedFile`` row for a stored file."""
    return _get_file_compatibility_facade().upsert_uploaded_file_record(
        db,
        user_id=user_id,
        filename=filename,
        storage_path=storage_path,
        mime_type=mime_type,
        file_size=file_size,
        file_id=file_id,
    )


def list_documents_for_user(
    *,
    user_id: Optional[int] = None,
    is_admin: bool,
) -> List[Dict[str, Any]]:
    """Load KB document metadata rows for a user."""
    return _get_file_compatibility_facade().list_documents_for_user(
        user_id=user_id,
        is_admin=is_admin,
    )


def find_referenced_file_ids(file_ids: Iterable[str]) -> set[str]:
    """Uncapped: which of ``file_ids`` any document references, whatever its owner."""
    return _get_file_compatibility_facade().find_referenced_file_ids(file_ids)


def build_uploaded_filename_map(
    db: Session, *, user_id: Optional[int], file_ids: List[str]
) -> Dict[str, str]:
    """Resolve ``file_id`` values to current uploaded filenames."""
    return _get_file_compatibility_facade().build_uploaded_filename_map(
        db,
        user_id=user_id,
        file_ids=file_ids,
    )


def get_document_record_file_id(
    record: Union[Dict[str, Any], DocumentRecord],
) -> Optional[str]:
    """Extract a normalized ``file_id`` from a KB document record."""
    return _get_file_compatibility_facade().get_document_record_file_id(record)


def resolve_document_filename(
    record: Union[Dict[str, Any], DocumentRecord], filename_map: Dict[str, str]
) -> Optional[str]:
    """Resolve a user-facing filename from ``file_id`` first, then legacy path."""
    return _get_file_compatibility_facade().resolve_document_filename(
        record,
        filename_map,
    )


def delete_uploaded_file_if_orphaned(
    db: Session,
    *,
    file_id: str,
    user_id: Optional[int],
    remaining_file_ids: set[str],
    after_commit: Optional[List[tuple[str, Callable[[], None]]]] = None,
) -> bool:
    """Delete an uploaded file's row and bytes when no documents still reference it.

    Never commits. With ``after_commit``, the durable delete and local unlink are
    queued for the caller to run after it commits; without it, the bytes go at once.
    """
    return _get_file_compatibility_facade().delete_uploaded_file_if_orphaned(
        db,
        file_id=file_id,
        user_id=user_id,
        remaining_file_ids=remaining_file_ids,
        after_commit=after_commit,
    )


def aggregate_uploaded_file_statuses(
    *,
    file_ids: List[str],
    user_id: int,
    is_admin: bool,
    use_cache: bool = True,
) -> Dict[str, str]:
    """Aggregate file status by joining documents + ingestion status records."""
    return _get_file_compatibility_facade().aggregate_uploaded_file_statuses(
        file_ids=file_ids,
        user_id=user_id,
        is_admin=is_admin,
        use_cache=use_cache,
    )


def reconcile_uploaded_files(
    db: Session,
    *,
    user_id: int,
    is_admin: bool,
    stale_ttl_hours: int = 24 * 7,
    delete_stale: bool = True,
    deletable_statuses: Optional[set[str]] = None,
) -> Dict[str, int]:
    """Reconcile uploaded files with document + ingestion status state."""
    return _get_file_compatibility_facade().reconcile_uploaded_files(
        db,
        user_id=user_id,
        is_admin=is_admin,
        stale_ttl_hours=stale_ttl_hours,
        delete_stale=delete_stale,
        deletable_statuses=deletable_statuses,
    )


def cleanup_local_copied_file(
    *,
    file_path: Path,
    local_root: Optional[Path] = None,
) -> FileCompensationResult:
    """Idempotently remove a staged local file created for KB ingestion."""
    return _get_file_compatibility_facade().cleanup_local_copied_file(
        file_path=file_path,
        local_root=local_root,
    )


def capture_uploaded_file_refresh_snapshot(
    file_record: UploadedFile,
    *,
    backup_path: Optional[Path] = None,
    reindex_marker_applied: bool = False,
) -> UploadedFileRefreshSnapshot:
    """Capture UploadedFile row and local-file state before a refresh."""
    return _get_file_compatibility_facade().capture_uploaded_file_refresh_snapshot(
        file_record,
        backup_path=backup_path,
        reindex_marker_applied=reindex_marker_applied,
    )


def restore_uploaded_file_refresh_snapshot(
    db: Session,
    snapshot: UploadedFileRefreshSnapshot,
) -> FileCompensationResult:
    """Restore UploadedFile row, local file, durable bytes, and cache state."""
    return _get_file_compatibility_facade().restore_uploaded_file_refresh_snapshot(
        db,
        snapshot,
    )
