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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from sqlalchemy.orm import Session

from ...config import get_uploads_dir
from ...core.tools.core.RAG_tools.LanceDB.schema_manager import (
    _safe_close_table,
    ensure_documents_table,
)
from ...core.tools.core.RAG_tools.management.status import (
    _load_ingestion_status_impl,
)
from ...core.tools.core.RAG_tools.storage.contracts import DocumentRecord
from ...core.tools.core.RAG_tools.utils.lancedb_query_utils import (
    list_embeddings_table_names,
    query_to_list,
)
from ...core.tools.core.RAG_tools.utils.string_utils import (
    build_lancedb_filter_expression,
    escape_lancedb_string,
)
from ...core.tools.core.RAG_tools.utils.user_permissions import UserPermissions
from ...core.tools.core.RAG_tools.utils.user_scope import resolve_user_scope
from ...core.tools.core.RAG_tools.version_management.cascade_cleaner import (
    cascade_delete,
)
from ...providers.vector_store.lancedb import get_connection_from_env
from ..models.uploaded_file import UploadedFile
from .uploaded_file_store import UploadedFileStore

if TYPE_CHECKING:
    from ...core.tools.core.RAG_tools.kb import KBFileCompatibilityFacade

logger = logging.getLogger(__name__)

_FILE_STATUS_BATCH_SIZE = 200
_STALE_FILE_STATUSES = {"FAILED", "UNKNOWN", "RUNNING"}
_DEFAULT_DELETABLE_STALE_STATUSES = {"FAILED"}


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


def _list_documents_for_user_impl(
    *,
    user_id: Optional[int] = None,
    is_admin: bool,
    collection_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load KB document metadata rows for a user."""
    conn = get_connection_from_env()
    ensure_documents_table(conn)
    table = None
    try:
        table = conn.open_table("documents")

        base_filter = ""
        if collection_name:
            base_filter = build_lancedb_filter_expression(
                {"collection": collection_name}
            )
        scope = resolve_user_scope(user_id=user_id, is_admin=is_admin)
        user_filter = UserPermissions.get_user_filter(
            scope.user_id, is_admin=scope.is_admin
        )
        combined_filter = (
            f"({base_filter}) and ({user_filter})"
            if user_filter and base_filter
            else (user_filter or base_filter)
        )
        query = table.search()
        if combined_filter:
            query = query.where(combined_filter)
        return query_to_list(query.limit(10000))
    finally:
        _safe_close_table(table)


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

    UploadedFileStore(db).delete(file_record, delete_local=False)
    # Invalidate cache for this user since file list changed
    _file_status_cache.invalidate_user(scope.user_id)
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


def _compensate_new_uploaded_file_impl(
    db: Session,
    *,
    file_id: str,
    user_id: Optional[int] = None,
    delete_local: bool = True,
    local_root: Optional[Path] = None,
) -> FileCompensationResult:
    """Idempotently remove a newly created UploadedFile row and artifacts.

    The caller owns commit/rollback timing. This helper flushes through
    ``UploadedFileStore.delete`` but intentionally does not commit.
    """
    normalized_file_id = str(file_id or "").strip()
    if not normalized_file_id:
        return _compensation_complete("missing_file_id")

    query = db.query(UploadedFile).filter(UploadedFile.file_id == normalized_file_id)
    if user_id is not None:
        scope = resolve_user_scope(user_id=user_id, is_admin=False)
        if scope.user_id is None:
            return _compensation_complete("missing_user")
        query = query.filter(UploadedFile.user_id == scope.user_id)
    file_record = query.first()
    if file_record is None:
        return _compensation_complete("already_removed")

    try:
        effective_local_root = local_root
        if delete_local and effective_local_root is None:
            effective_local_root = get_uploads_dir()
        UploadedFileStore(db).delete(
            file_record,
            delete_local=delete_local,
            local_root=effective_local_root,
        )
        record_user_id = getattr(file_record, "user_id", None)
        if record_user_id is not None:
            _file_status_cache.invalidate_user(int(record_user_id))
        return _compensation_complete("uploaded_file_removed")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to compensate UploadedFile creation: file_id=%s error=%s",
            normalized_file_id,
            exc,
        )
        return _compensation_incomplete(exc)


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


def _build_file_id_in_filter(file_ids: List[str]) -> str:
    escaped_ids = [f"'{escape_lancedb_string(file_id)}'" for file_id in file_ids]
    return f"file_id IN ({', '.join(escaped_ids)})"


def _build_doc_id_in_filter(doc_ids: List[str]) -> str:
    escaped_ids = [f"'{escape_lancedb_string(doc_id)}'" for doc_id in doc_ids]
    return f"doc_id IN ({', '.join(escaped_ids)})"


def _combine_lancedb_filters(
    base_filter: Optional[str], user_filter: Optional[str]
) -> Optional[str]:
    if base_filter and user_filter:
        return f"({base_filter}) and ({user_filter})"
    return base_filter or user_filter


def _load_indexed_doc_refs(
    conn: Any,
    *,
    collections: List[str],
    doc_refs_by_file_id: Dict[str, List[tuple[str, str]]],
    user_filter: Optional[str],
) -> set[tuple[str, str]]:
    """Return document refs that have searchable artifacts in LanceDB.

    Legacy deployments may not have ingestion status rows. If chunks or
    embeddings exist for a document, the file was already indexed enough to be
    user-visible and should be treated as successful for file-list status.
    """
    indexed_refs: set[tuple[str, str]] = set()
    doc_ids_by_collection: Dict[str, set[str]] = {
        collection: set() for collection in collections
    }
    for doc_refs in doc_refs_by_file_id.values():
        for collection, doc_id in doc_refs:
            if collection in doc_ids_by_collection:
                doc_ids_by_collection[collection].add(doc_id)

    candidate_refs = {
        (collection, doc_id)
        for collection, doc_ids in doc_ids_by_collection.items()
        for doc_id in doc_ids
    }
    if not candidate_refs:
        return indexed_refs

    candidate_tables = ["chunks", *list_embeddings_table_names(conn)]
    for table_name in candidate_tables:
        if indexed_refs.issuperset(candidate_refs):
            return indexed_refs
        table = None
        try:
            table = conn.open_table(table_name)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Skipping indexed status fallback table '%s': %s", table_name, exc
            )
            continue

        try:
            for collection, doc_ids in doc_ids_by_collection.items():
                pending_doc_ids = {
                    doc_id
                    for doc_id in doc_ids
                    if (collection, doc_id) not in indexed_refs
                }
                if not pending_doc_ids:
                    continue
                collection_filter = build_lancedb_filter_expression(
                    {"collection": collection},
                    skip_user_filter=True,
                )
                doc_filter = _build_doc_id_in_filter(sorted(pending_doc_ids))
                combined_filter = _combine_lancedb_filters(
                    f"({collection_filter}) and ({doc_filter})", user_filter
                )
                try:
                    query = table.search().where(combined_filter)
                    rows = query_to_list(
                        query.select(["collection", "doc_id"]).limit(-1)
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "Failed indexed status fallback query on '%s' for collection '%s': %s",
                        table_name,
                        collection,
                        exc,
                    )
                    continue

                for row in rows:
                    row_collection = str(row.get("collection") or "").strip()
                    row_doc_id = str(row.get("doc_id") or "").strip()
                    if row_collection and row_doc_id:
                        indexed_refs.add((row_collection, row_doc_id))
        finally:
            _safe_close_table(table)

    return indexed_refs


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
    user_filter = UserPermissions.get_user_filter(user_id, is_admin=is_admin)

    doc_refs_by_file_id: Dict[str, List[tuple[str, str]]] = {
        file_id: [] for file_id in normalized_file_ids
    }
    documents_table = None
    try:
        documents_table = conn.open_table("documents")
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
    finally:
        _safe_close_table(documents_table)

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

    indexed_doc_refs = _load_indexed_doc_refs(
        conn,
        collections=collections,
        doc_refs_by_file_id=doc_refs_by_file_id,
        user_filter=user_filter,
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
    conn = get_connection_from_env()
    ensure_documents_table(conn)
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

        safe_file_id = escape_lancedb_string(file_id)
        # Query documents table to get (collection, doc_id) pairs for cascade deletion
        documents_table = None
        try:
            documents_table = conn.open_table("documents")
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
        finally:
            _safe_close_table(documents_table)

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
    collection_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load KB document metadata rows for a user."""
    return _get_file_compatibility_facade().list_documents_for_user(
        user_id=user_id,
        is_admin=is_admin,
        collection_name=collection_name,
    )


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
) -> bool:
    """Delete uploaded file row and local file when no documents still reference it."""
    return _get_file_compatibility_facade().delete_uploaded_file_if_orphaned(
        db,
        file_id=file_id,
        user_id=user_id,
        remaining_file_ids=remaining_file_ids,
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


def compensate_new_uploaded_file(
    db: Session,
    *,
    file_id: str,
    user_id: Optional[int] = None,
    delete_local: bool = True,
    local_root: Optional[Path] = None,
) -> FileCompensationResult:
    """Idempotently remove a newly created UploadedFile row and artifacts."""
    return _get_file_compatibility_facade().compensate_new_uploaded_file(
        db,
        file_id=file_id,
        user_id=user_id,
        delete_local=delete_local,
        local_root=local_root,
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
