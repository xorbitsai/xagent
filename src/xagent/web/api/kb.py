"""Knowledge base API route handlers"""

import asyncio
import functools
import hashlib
import json
import logging
import mimetypes
import os
import re
import shutil
import threading
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypedDict, TypeVar, cast

from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import JSONResponse
from googleapiclient.discovery import build  # type: ignore
from googleapiclient.http import MediaIoBaseDownload  # type: ignore
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...core.tools.core.RAG_tools.core.config import DEFAULT_VECTOR_STORE_SCAN_LIMIT
from ...core.tools.core.RAG_tools.core.parser_registry import (
    get_supported_parsers,
    validate_parser_compatibility,
)
from ...core.tools.core.RAG_tools.core.schemas import (
    ChunkStrategy,
    CollectionDocumentMetadata,
    CollectionOperationResult,
    FusionConfig,
    IngestionConfig,
    IngestionResult,
    ListCollectionsResult,
    ParseMethod,
    ParseResultResponse,
    SearchConfig,
    SearchPipelineResult,
    SearchType,
    WebCrawlConfig,
    WebIngestionResult,
)
from ...core.tools.core.RAG_tools.management.collection_manager import (
    get_collection_sync,
)
from ...core.tools.core.RAG_tools.management.collections import (
    delete_collection,
    delete_document,
    list_collections,
    list_documents,
)
from ...core.tools.core.RAG_tools.management.status import clear_ingestion_status
from ...core.tools.core.RAG_tools.parse.parse_display import (
    paginate_parse_results,
    reconstruct_parse_result_from_db,
)
from ...core.tools.core.RAG_tools.pipelines.document_ingestion import (
    run_document_ingestion,
)
from ...core.tools.core.RAG_tools.pipelines.document_search import run_document_search
from ...core.tools.core.RAG_tools.pipelines.web_ingestion import (
    FileHandlerResult,
    run_web_ingestion,
)
from ...core.tools.core.RAG_tools.progress import get_progress_manager
from ...core.tools.core.RAG_tools.storage.contracts import DocumentRecord
from ...core.tools.core.RAG_tools.storage.factory import get_vector_index_store
from ...core.tools.core.RAG_tools.utils.string_utils import (
    generate_deterministic_doc_id,
)
from ...core.tools.core.RAG_tools.utils.user_scope import user_scope_context
from ..auth_dependencies import get_current_user
from ..config import (
    MAX_COLLECTION_NAME_LENGTH,
    MAX_FILE_SIZE,
    MAX_FILE_SIZE_LABEL,
    get_upload_path,
    is_allowed_file,
    sanitize_path_component,
)
from ..models.database import get_db, get_session_local
from ..models.uploaded_file import UploadedFile
from ..models.user import User
from ..services.kb_collection_service import (
    delete_collection_physical_dir,
    delete_collection_uploaded_files,
    rename_collection_storage,
)
from ..services.kb_file_service import (
    build_uploaded_filename_map as _build_uploaded_filename_map,
)
from ..services.kb_file_service import (
    delete_uploaded_file_if_orphaned as _delete_uploaded_file_if_orphaned,
)
from ..services.kb_file_service import (
    get_document_record_file_id as _get_document_record_file_id,
)
from ..services.kb_file_service import (
    list_documents_for_user as _list_documents_for_user,
)
from ..services.kb_file_service import (
    resolve_document_filename as _resolve_document_filename,
)
from ..services.kb_file_service import (
    upsert_uploaded_file_record as _upsert_uploaded_file_record,
)
from .cloud_storage import get_google_credentials

T = TypeVar("T", bound=Callable[..., Any])
logger = logging.getLogger(__name__)

_SQL_LIKE_ESCAPE = "\\"
_PDF_ONLY_PARSE_METHODS = {
    ParseMethod.PYPDF,
    ParseMethod.PDFPLUMBER,
    ParseMethod.PYMUPDF,
}
# lock_key -> (lock, active waiter/holder count)
_WEB_FILE_LOCKS: Dict[str, tuple[threading.Lock, int]] = {}
_WEB_FILE_LOCKS_GUARD = threading.Lock()
_WEB_FILENAME_HASH_LENGTH = 16
_WEB_FILENAME_SUFFIX = ".md"
_MAX_FILESYSTEM_FILENAME_BYTES = 255
_MAX_WEB_TITLE_FILENAME_BYTES = _MAX_FILESYSTEM_FILENAME_BYTES - len(
    f"{'0' * _WEB_FILENAME_HASH_LENGTH}_{_WEB_FILENAME_SUFFIX}".encode("utf-8")
)


def _like_contains_pattern(value: str) -> str:
    escaped = (
        value.replace(_SQL_LIKE_ESCAPE, _SQL_LIKE_ESCAPE * 2)
        .replace("%", f"{_SQL_LIKE_ESCAPE}%")
        .replace("_", f"{_SQL_LIKE_ESCAPE}_")
    )
    return f"%{escaped}%"


def _normalize_parse_method_for_filename(
    parse_method: Optional[ParseMethod], filename: str
) -> ParseMethod:
    normalized = parse_method if parse_method is not None else ParseMethod.DEFAULT
    if Path(filename).suffix.lower() == ".pdf":
        return normalized
    if normalized in _PDF_ONLY_PARSE_METHODS:
        logger.warning(
            "Falling back to default parser for non-PDF file %s (requested parser: %s)",
            filename,
            normalized.value,
        )
        return ParseMethod.DEFAULT
    return normalized


def _normalize_web_title_for_filename(title: str) -> str:
    """Convert arbitrary web page titles into filesystem-safe filename parts."""
    normalized = unicodedata.normalize("NFKC", title).strip()
    if not normalized:
        return "untitled"

    # Replace separators and punctuation-heavy runs with underscores so
    # ordinary article titles ("How to edit a completed job?") remain usable.
    normalized = normalized.replace("/", " ").replace("\\", " ")
    normalized = re.sub(r"[^\w.-]", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("._-")

    if not normalized:
        return "untitled"

    trimmed = normalized[:MAX_COLLECTION_NAME_LENGTH]
    while trimmed and len(trimmed.encode("utf-8")) > _MAX_WEB_TITLE_FILENAME_BYTES:
        trimmed = trimmed[:-1]
    trimmed = trimmed.rstrip("._-")
    return trimmed or "untitled"


def _validate_parser_for_file(
    filename: str,
    parse_method: Optional[ParseMethod],
    *,
    user_id: Any = None,
) -> None:
    """Fail-fast validation: reject files with no parser or incompatible parser.

    Raises:
        HTTPException(422): if no parser supports the extension or the
            requested parser is incompatible.
    """
    file_ext = Path(filename).suffix.lower()
    effective = _normalize_parse_method_for_filename(parse_method, filename)

    if effective == ParseMethod.DEFAULT:
        supported = get_supported_parsers(file_ext)
        if not supported:
            logger.warning(
                "KB ingest rejected: no parser supports extension=%s filename=%s user_id=%s",
                file_ext,
                filename,
                user_id,
            )
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Unsupported file type '{file_ext}' for ingestion. "
                    "No available parser supports this format."
                ),
            )
    else:
        if not validate_parser_compatibility(file_ext, str(effective)):
            supported = get_supported_parsers(file_ext)
            logger.warning(
                "KB ingest rejected: parser=%s not compatible with extension=%s filename=%s supported=%s",
                str(effective),
                file_ext,
                filename,
                supported,
            )
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Parser '{str(effective)}' is not compatible with "
                    f"file type '{file_ext}'. "
                    f"Supported parsers for this type: {supported}"
                ),
            )


def _get_completed_step_metadata(
    result: IngestionResult, step_name: str
) -> Optional[Dict[str, Any]]:
    for step in result.completed_steps:
        current_name = (
            step.get("name") if isinstance(step, dict) else getattr(step, "name", None)
        )
        if current_name != step_name:
            continue
        metadata = (
            step.get("metadata")
            if isinstance(step, dict)
            else getattr(step, "metadata", None)
        )
        return metadata if isinstance(metadata, dict) else None
    return None


def _restore_ingest_file_backup(
    *,
    file_path: Path,
    backup_path: Optional[Path],
    had_existing_file: bool,
) -> None:
    if backup_path is not None and backup_path.exists():
        if file_path.exists():
            file_path.unlink()
        backup_path.replace(file_path)
        logger.info("Restored pre-ingest backup for %s", file_path)
        return

    if had_existing_file:
        raise FileNotFoundError(f"Missing ingest backup for {file_path}")

    if file_path.exists():
        file_path.unlink()
        logger.info("Removed failed-ingest file %s", file_path)


def _ensure_cleanup_succeeded(operation_name: str, result_obj: Any) -> None:
    status = str(getattr(result_obj, "status", "")).strip().lower()
    if status in {"success", "partial_success"}:
        return
    message = str(getattr(result_obj, "message", "cleanup failed")).strip()
    raise RuntimeError(f"{operation_name} failed: {message}")


async def _cleanup_failed_new_collection_metadata(
    *,
    collection_name: str,
    user: User,
) -> None:
    """Remove config rows left behind when a brand-new collection ingest fails."""
    from ...core.tools.core.RAG_tools.storage.factory import get_metadata_store

    metadata_store = get_metadata_store()
    cleanup_result = await metadata_store.delete_collection_metadata(
        collection_name=collection_name,
        user_id=int(user.id),
        is_admin=bool(user.is_admin),
        delete_orphaned_metadata=True,
    )
    logger.info(
        "Cleaned failed-ingest collection metadata for %s: %s",
        collection_name,
        cleanup_result,
    )


async def _rollback_failed_ingestion(
    *,
    db: Session,
    user: User,
    collection_name: str,
    result: IngestionResult,
    file_path: Path,
    file_record: UploadedFile,
    collection_existed_before: bool,
    uploaded_file_existed_before: bool,
    file_backup_path: Optional[Path],
    had_existing_file: bool,
) -> None:
    user_id = int(user.id)
    vector_store = get_vector_index_store()
    register_metadata = _get_completed_step_metadata(result, "register_document") or {}
    register_created = bool(register_metadata.get("created"))
    doc_id = result.doc_id if isinstance(result.doc_id, str) and result.doc_id else None

    try:
        if not collection_existed_before:
            collection_records = vector_store.list_document_records(
                collection_name=collection_name,
                user_id=user_id,
                is_admin=bool(user.is_admin),
            )
            collection_file_ids = {
                file_id
                for file_id in (
                    _get_document_record_file_id(record)
                    for record in collection_records
                )
                if file_id
            }

            collection_delete_result = delete_collection(
                collection_name,
                user_id,
                bool(user.is_admin),
            )
            _ensure_cleanup_succeeded(
                f"delete collection '{collection_name}' during rollback",
                collection_delete_result,
            )

            physical_cleanup = delete_collection_physical_dir(
                user_id=user_id,
                collection_name=collection_name,
            )
            if physical_cleanup.status not in {"success", "not_found"}:
                error_detail = (
                    physical_cleanup.error or "unknown physical cleanup failure"
                )
                raise RuntimeError(
                    "delete collection physical directory during rollback failed: "
                    f"{error_detail}"
                )
            remaining_records = vector_store.list_document_records(
                collection_name=None,
                user_id=user_id,
                is_admin=bool(user.is_admin),
            )
            remaining_file_ids = {
                file_id
                for file_id in (
                    _get_document_record_file_id(record) for record in remaining_records
                )
                if file_id
            }
            delete_collection_uploaded_files(
                db,
                user_id=user_id,
                collection_file_ids=collection_file_ids,
                remaining_file_ids=remaining_file_ids,
                collection_dir=physical_cleanup.collection_dir,
            )
            if not uploaded_file_existed_before:
                refreshed_file_record = (
                    db.query(UploadedFile)
                    .filter(UploadedFile.file_id == file_record.file_id)
                    .first()
                )
                if refreshed_file_record is not None:
                    db.delete(refreshed_file_record)
            await _cleanup_failed_new_collection_metadata(
                collection_name=collection_name,
                user=user,
            )
            db.commit()
            _restore_ingest_file_backup(
                file_path=file_path,
                backup_path=file_backup_path,
                had_existing_file=had_existing_file,
            )
            return

        if register_created and doc_id:
            document_delete_result = delete_document(
                collection_name,
                doc_id,
                user_id,
                bool(user.is_admin),
            )
            _ensure_cleanup_succeeded(
                f"delete document '{doc_id}' during rollback",
                document_delete_result,
            )
            remaining_records = vector_store.list_document_records(
                collection_name=None,
                user_id=user_id,
                is_admin=bool(user.is_admin),
            )
            remaining_file_ids = {
                current_file_id
                for current_file_id in (
                    _get_document_record_file_id(record) for record in remaining_records
                )
                if current_file_id
            }
            _delete_uploaded_file_if_orphaned(
                db,
                file_id=str(file_record.file_id),
                user_id=user_id,
                remaining_file_ids=remaining_file_ids,
            )
            db.commit()
        else:
            if doc_id:
                clear_ingestion_status(
                    collection_name,
                    doc_id,
                    user_id=user_id,
                    is_admin=bool(user.is_admin),
                )
            if not uploaded_file_existed_before:
                db.delete(file_record)
                db.commit()

        _restore_ingest_file_backup(
            file_path=file_path,
            backup_path=file_backup_path,
            had_existing_file=had_existing_file,
        )
    except Exception as exc:
        db.rollback()
        restore_error: Optional[Exception] = None
        try:
            _restore_ingest_file_backup(
                file_path=file_path,
                backup_path=file_backup_path,
                had_existing_file=had_existing_file,
            )
        except Exception as restore_exc:  # noqa: BLE001
            restore_error = restore_exc
        logger.warning(
            "Failed to fully roll back ingest for %s/%s: %s",
            collection_name,
            file_path.name,
            exc,
        )
        message = f"Failed to fully roll back ingest for {collection_name}/{file_path.name}: {exc}"
        if restore_error is not None:
            message = f"{message}; backup restore also failed: {restore_error}"
        raise RollbackFailureError(message) from exc


async def _rollback_failed_cloud_ingestion(
    *,
    db: Session,
    user: User,
    collection_name: str,
    result: IngestionResult,
    file_path: Path,
    file_record: Optional[UploadedFile],
    collection_existed_before: bool,
    uploaded_file_existed_before: bool,
    file_backup_path: Optional[Path],
    had_existing_file: bool,
) -> None:
    user_id = int(user.id)
    vector_store = get_vector_index_store()
    register_metadata = _get_completed_step_metadata(result, "register_document") or {}
    register_created = bool(register_metadata.get("created"))
    doc_id = result.doc_id if isinstance(result.doc_id, str) and result.doc_id else None

    try:
        if register_created and doc_id:
            document_delete_result = delete_document(
                collection_name,
                doc_id,
                user_id,
                bool(user.is_admin),
            )
            _ensure_cleanup_succeeded(
                f"delete document '{doc_id}' during cloud rollback",
                document_delete_result,
            )
        elif doc_id:
            clear_ingestion_status(
                collection_name,
                doc_id,
                user_id=user_id,
                is_admin=bool(user.is_admin),
            )

        remaining_records = vector_store.list_document_records(
            collection_name=None,
            user_id=user_id,
            is_admin=bool(user.is_admin),
        )
        remaining_file_ids = {
            current_file_id
            for current_file_id in (
                _get_document_record_file_id(record) for record in remaining_records
            )
            if current_file_id
        }

        if file_record is not None:
            _delete_uploaded_file_if_orphaned(
                db,
                file_id=str(file_record.file_id),
                user_id=user_id,
                remaining_file_ids=remaining_file_ids,
            )

        collection_records = vector_store.list_document_records(
            collection_name=collection_name,
            user_id=user_id,
            is_admin=bool(user.is_admin),
            max_results=1,
        )
        removed_new_collection = False
        if not collection_existed_before and not collection_records:
            collection_delete_result = delete_collection(
                collection_name,
                user_id,
                bool(user.is_admin),
            )
            _ensure_cleanup_succeeded(
                f"delete collection '{collection_name}' during cloud rollback",
                collection_delete_result,
            )
            removed_new_collection = True

        if removed_new_collection:
            await _cleanup_failed_new_collection_metadata(
                collection_name=collection_name,
                user=user,
            )

        db.commit()
        _restore_ingest_file_backup(
            file_path=file_path,
            backup_path=file_backup_path,
            had_existing_file=had_existing_file,
        )
    except Exception as exc:
        db.rollback()
        restore_error: Optional[Exception] = None
        try:
            _restore_ingest_file_backup(
                file_path=file_path,
                backup_path=file_backup_path,
                had_existing_file=had_existing_file,
            )
        except Exception as restore_exc:  # noqa: BLE001
            restore_error = restore_exc
        logger.warning(
            "Failed to fully roll back cloud ingest for %s/%s: %s",
            collection_name,
            file_path.name,
            exc,
        )
        message = (
            "Failed to fully roll back cloud ingest for "
            f"{collection_name}/{file_path.name}: {exc}"
        )
        if restore_error is not None:
            message = f"{message}; backup restore also failed: {restore_error}"
        raise RollbackFailureError(message) from exc


def cleanup_orphaned_temp_files(upload_dir: Optional[Path] = None) -> int:
    """Clean up orphaned temporary files from interrupted atomic replacements.

    Removes files matching patterns like:
    - *.tmp-replace (old pattern)
    - .*.tmp (new NamedTemporaryFile pattern)

    Args:
        upload_dir: Base uploads directory to clean. If None, uses default uploads dir.

    Returns:
        Number of files cleaned up.
    """
    from ..config import get_uploads_dir

    base_dir = upload_dir or get_uploads_dir()
    if not base_dir.exists():
        return 0

    cleaned_count = 0
    now = time.time()

    # Walk through uploads directory and clean up temp files older than 1 hour
    # to avoid deleting files that might still be in use
    for root, dirs, files in os.walk(base_dir):
        for filename in files:
            file_path = Path(root) / filename

            # Check for old temp file pattern (*.tmp-replace)
            if filename.endswith(".tmp-replace"):
                file_age = now - file_path.stat().st_mtime
                if file_age > 3600:  # 1 hour
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                        logger.debug("Cleaned up orphaned temp file: %s", file_path)
                    except OSError as e:
                        logger.warning(
                            "Failed to clean up orphaned temp file %s: %s", file_path, e
                        )

            # Check for new temp file pattern (.*.tmp from NamedTemporaryFile)
            # Pattern: filename.XXXXXX.tmp where X is random hex
            if filename.endswith(".tmp") and "." in filename[:-4]:
                # Verify it looks like our temp pattern (has multiple extensions)
                parts = filename.split(".")
                if len(parts) >= 3 and parts[-1] == "tmp":
                    file_age = now - file_path.stat().st_mtime
                    if file_age > 3600:  # 1 hour
                        try:
                            file_path.unlink()
                            cleaned_count += 1
                            logger.debug("Cleaned up orphaned temp file: %s", file_path)
                        except OSError as e:
                            logger.warning(
                                "Failed to clean up orphaned temp file %s: %s",
                                file_path,
                                e,
                            )

    if cleaned_count > 0:
        logger.info("Cleaned up %d orphaned temporary file(s)", cleaned_count)

    return cleaned_count


def _get_file_sha256(file_path: Path) -> str:
    """Compute SHA256 hash for a local file."""
    hash_obj = hashlib.sha256()
    with file_path.open("rb") as file_obj:
        while True:
            chunk = file_obj.read(1024 * 1024)
            if not chunk:
                break
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def _atomic_replace_file(source_path: Path, target_path: Path) -> None:
    """Atomically replace target file with source file content.

    Uses a temporary file in the same directory as the target to ensure
    atomic replacement via os.replace(). The temp file is automatically
    cleaned up on success, and will be cleaned up by the OS on crash
    (on most systems) or on next startup via cleanup logic.
    """
    import tempfile

    # Ensure target directory exists
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a temp file in the same directory as target (required for atomic replace)
    # delete=False so we can use it for replace() and clean up manually
    with tempfile.NamedTemporaryFile(
        dir=target_path.parent,
        prefix=f"{target_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
        # Copy to temp file first
        shutil.copy2(source_path, tmp_path)

    # Atomic replace - this is atomic on POSIX systems
    tmp_path.replace(target_path)


def _mark_uploaded_file_for_reindex(file_id: str) -> bool:
    """Clear ingestion run markers so changed file can be re-indexed."""
    try:
        from ...core.tools.core.RAG_tools.LanceDB.schema_manager import (
            _safe_close_table,
            ensure_documents_table,
            ensure_ingestion_runs_table,
        )
        from ...core.tools.core.RAG_tools.utils.lancedb_query_utils import query_to_list
        from ...core.tools.core.RAG_tools.utils.string_utils import (
            escape_lancedb_string,
        )
        from ...providers.vector_store.lancedb import get_connection_from_env

        conn = get_connection_from_env()
        ensure_documents_table(conn)
        ensure_ingestion_runs_table(conn)
        documents_table = None
        ingestion_runs_table = None
        try:
            documents_table = conn.open_table("documents")
            ingestion_runs_table = conn.open_table("ingestion_runs")

            safe_file_id = escape_lancedb_string(file_id)
            rows = query_to_list(
                documents_table.search()
                .where(f"file_id = '{safe_file_id}'")
                .select(["collection", "doc_id"])
                .limit(-1)
            )
            for row in rows:
                collection = str(row.get("collection") or "").strip()
                doc_id = str(row.get("doc_id") or "").strip()
                if not collection or not doc_id:
                    continue
                safe_collection = escape_lancedb_string(collection)
                safe_doc_id = escape_lancedb_string(doc_id)
                ingestion_runs_table.delete(
                    f"collection = '{safe_collection}' and doc_id = '{safe_doc_id}'"
                )
        finally:
            _safe_close_table(documents_table)
            _safe_close_table(ingestion_runs_table)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to mark uploaded file for re-index: file_id=%s, error=%s",
            file_id,
            exc,
            exc_info=True,
        )
        return False


def _refresh_existing_file_if_changed(
    existing_record: Any,
    temp_file_path: Path,
    db_session: Session,
    user_id: int,
    url: str,
    filename: str,
    url_hash: str,
    processed_urls: Dict[str, str],
    context: str,
) -> Optional[FileHandlerResult]:
    """Refresh existing file if content has changed.

    This function:
    1. Compares file hashes to detect content changes
    2. If changed, marks for reindex FIRST (before any file modification)
    3. If mark succeeds, atomically replaces the file and updates DB record
    4. If mark fails, returns existing file without refresh (stale but consistent)

    Args:
        existing_record: The UploadedFile record from database
        temp_file_path: Path to the new temporary file
        db_session: Database session for updates
        user_id: User ID for record ownership
        url: Source URL (for logging)
        filename: Filename for the record
        url_hash: Hash key for processed_urls cache
        processed_urls: Cache dict to update with new file_id
        context: Context string for logging (e.g., "in-memory cache", "cross-session")

    Returns:
        FileHandlerResult when the existing file remains usable or was refreshed.
        Returns None only when the existing file path no longer exists and the
        caller should continue with normal new-file handling.
    """
    existing_path = Path(str(existing_record.storage_path))
    if not existing_path.exists():
        return None

    old_hash = _get_file_sha256(existing_path)
    new_hash = _get_file_sha256(temp_file_path)

    if old_hash == new_hash:
        # Content unchanged - return existing file
        return FileHandlerResult(
            file_path=str(existing_record.storage_path),
            file_id=str(existing_record.file_id),
        )

    # Content changed - first try to mark for reindex BEFORE modifying file
    if not _mark_uploaded_file_for_reindex(str(existing_record.file_id)):
        logger.warning(
            "Failed to mark file for reindex, skipping file refresh to avoid inconsistent state: "
            "url=%s, file_id=%s, context=%s",
            url,
            existing_record.file_id,
            context,
        )
        # Return existing file without refreshing (stale embeddings but consistent state)
        return FileHandlerResult(
            file_path=str(existing_record.storage_path),
            file_id=str(existing_record.file_id),
        )

    # Mark succeeded - now atomically replace the file
    _atomic_replace_file(temp_file_path, existing_path)
    file_record = _upsert_uploaded_file_record(
        db_session,
        user_id=user_id,
        filename=filename,
        storage_path=existing_path,
        mime_type="text/markdown",
        file_size=existing_path.stat().st_size,
    )
    processed_urls[url_hash] = str(file_record.file_id)

    logger.info(
        "Marked changed web file as PENDING_REINDEX and refreshed content: url=%s, file_id=%s, context=%s",
        url,
        file_record.file_id,
        context,
    )

    return FileHandlerResult(
        file_path=str(existing_record.storage_path),
        file_id=str(existing_record.file_id),
    )


class _WebFileLock:
    """Per-key in-process lock for web ingestion file operations."""

    def __init__(self, lock_key: str) -> None:
        self._lock_key = lock_key
        self._lock: Optional[threading.Lock] = None

    def __enter__(self) -> "_WebFileLock":
        with _WEB_FILE_LOCKS_GUARD:
            lock_entry = _WEB_FILE_LOCKS.get(self._lock_key)
            if lock_entry is None:
                lock = threading.Lock()
                _WEB_FILE_LOCKS[self._lock_key] = (lock, 1)
            else:
                lock, ref_count = lock_entry
                _WEB_FILE_LOCKS[self._lock_key] = (lock, ref_count + 1)
            self._lock = lock
        # Acquire the per-key lock outside the global guard to avoid
        # blocking other threads from accessing the registry for different keys.
        self._lock.acquire()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._lock is not None:
            self._lock.release()
        with _WEB_FILE_LOCKS_GUARD:
            lock_entry = _WEB_FILE_LOCKS.get(self._lock_key)
            if lock_entry is None:
                return
            lock, ref_count = lock_entry
            if ref_count <= 1:
                _WEB_FILE_LOCKS.pop(self._lock_key, None)
                return
            _WEB_FILE_LOCKS[self._lock_key] = (lock, ref_count - 1)


def handle_kb_exceptions(func: T) -> T:
    """Decorator to handle common exceptions in KB API routes."""

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            raise
        except (ValueError, KeyError, TypeError) as e:
            logger.error("Data format error in %s: %s", func.__name__, e)
            raise HTTPException(status_code=400, detail=f"数据格式错误: {str(e)}")
        except (PermissionError, OSError) as e:
            logger.error("File system error in %s: %s", func.__name__, e)
            raise HTTPException(status_code=403, detail=f"File system error: {str(e)}")
        except Exception as e:
            logger.exception("Unexpected error in %s: %s", func.__name__, e)
            raise HTTPException(
                status_code=500,
                detail=f"服务器内部错误: {str(e)}",
            )

    return cast(T, wrapper)


def with_kb_user_scope(func: T) -> T:
    """Wrap route handlers with request user scope context."""

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        user = kwargs.get("_user")

        if user is None:
            return await func(*args, **kwargs)

        with user_scope_context(
            user_id=int(getattr(user, "id")),
            is_admin=bool(getattr(user, "is_admin", False)),
        ):
            return await func(*args, **kwargs)

    return cast(T, wrapper)


# Create router
kb_router = APIRouter(prefix="/api/kb", tags=["kb"])


class CloudFile(BaseModel):
    provider: str
    fileId: str
    fileName: str


class CloudIngestRequest(BaseModel):
    files: List[CloudFile]
    collection: str
    parse_method: Optional[ParseMethod] = None
    chunk_strategy: Optional[ChunkStrategy] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    separators: Optional[List[str]] = None
    embedding_model_id: str = "text-embedding-v4"
    embedding_batch_size: Optional[int] = None
    max_retries: Optional[int] = None
    retry_delay: Optional[float] = None


class RollbackFailureError(RuntimeError):
    """Raised when best-effort ingest rollback cannot complete cleanly."""


def _build_cloud_storage_filename(original_filename: str, file_id: str) -> str:
    """Generate a collision-resistant local filename for cloud ingests."""
    original_path = Path(original_filename)
    suffix = original_path.suffix
    stem = original_path.stem or "cloud-file"
    digest = hashlib.sha256(file_id.encode("utf-8")).hexdigest()[:12]
    return f"{stem}__{digest}{suffix}"


def _raise_if_list_collections_failed(
    result: ListCollectionsResult, *, stage: str
) -> None:
    """Fail closed when collection listing cannot read storage (do not infer access)."""
    if result.status != "success":
        raise HTTPException(
            status_code=503,
            detail=(
                f"Knowledge base temporarily unavailable ({stage}): {result.message}"
            ),
        )


async def _list_collections_with_retry(
    *,
    user_id: Optional[int],
    is_admin: bool,
    stage: str,
) -> ListCollectionsResult:
    """Call ``list_collections`` with short retries on transient LanceDB/read errors."""
    delay_s = 0.05
    last: Optional[ListCollectionsResult] = None
    for attempt in range(3):
        last = await list_collections(user_id=user_id, is_admin=is_admin)
        if last.status == "success":
            return last
        if attempt < 2:
            logger.warning(
                "list_collections non-success (attempt %s/3, stage=%s, status=%r): %s",
                attempt + 1,
                stage,
                last.status,
                last.message,
            )
            await asyncio.sleep(delay_s)
            delay_s *= 2
    if last is None:
        raise HTTPException(
            status_code=503,
            detail=f"Knowledge base temporarily unavailable ({stage}): no result",
        )
    _raise_if_list_collections_failed(last, stage=stage)
    # _raise_if_list_collections_failed always raises on non-success.
    raise HTTPException(
        status_code=503,
        detail=f"Knowledge base temporarily unavailable ({stage}): unknown error",
    )


async def _ensure_collection_access(
    collection_name: str,
    user: User,
    *,
    hide_missing: bool = False,
    allow_create: bool = False,
) -> None:
    """Enforce collection-level access semantics for KB APIs.

    Rules:
    - Admin users always pass.
    - If collection exists but is not visible to current user: raise 403.
    - If collection does not exist globally: when ``allow_create`` is True, allow
      (first ingest / config for a new collection name); otherwise raise 404, or
      403 when ``hide_missing`` is True.
    - If ``list_collections`` returns ``status != "success"``, raise 503 (do not
      infer access from an empty list after a storage read failure).
    """
    if bool(user.is_admin):
        return

    current_user_id = int(user.id)
    visible = await _list_collections_with_retry(
        user_id=current_user_id,
        is_admin=False,
        stage="list_visible_collections_for_access_check",
    )
    if any(c.name == collection_name for c in visible.collections):
        return

    # hide_missing=True masks existence details as 403, so a global listing adds
    # no behavioral value on this path and only costs an extra storage call.
    if hide_missing:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied for collection: {collection_name}",
        )

    all_collections = await _list_collections_with_retry(
        user_id=None,
        is_admin=True,
        stage="list_all_collections_for_access_check",
    )
    if not any(c.name == collection_name for c in all_collections.collections):
        if allow_create:
            return
        raise HTTPException(
            status_code=404, detail=f"Collection not found: {collection_name}"
        )

    raise HTTPException(
        status_code=403,
        detail=f"Access denied for collection: {collection_name}",
    )


async def _ensure_collection_access_for_document_delete(
    collection_name: str,
    user: User,
) -> None:
    """Gate document deletes on collection visibility, with a vector-store fallback.

    ``list_collections`` can briefly disagree with LanceDB documents (e.g. control-plane
    rename lag). If we would return **403** only because the name is missing from the
    user's listing while it exists globally, still allow the request when the caller has
    at least one document row in that collection (same rule as ``delete_document``).

    Cross-tenant callers keep **403**: they have no scoped rows in the target collection.
    """
    if bool(user.is_admin):
        return

    try:
        await _ensure_collection_access(collection_name, user, hide_missing=True)
    except HTTPException as exc:
        if exc.status_code != 403:
            raise
        detail_text = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        if not detail_text.startswith("Access denied for collection:"):
            raise

        vector_store = get_vector_index_store()
        try:
            owned_records = vector_store.list_document_records(
                collection_name=collection_name,
                user_id=int(user.id),
                is_admin=False,
                max_results=1,
            )
        except Exception:
            owned_records = []

        if owned_records:
            return
        raise


def _parse_separators(separators: Optional[str]) -> Optional[List[str]]:
    """Parse optional custom separators (JSON array of strings) from form input.

    Returns None if input is missing/empty or invalid; returns a list of
    non-empty strings when valid (possibly empty list for input '[]').
    """
    if not separators or not separators.strip():
        return None
    try:
        raw = json.loads(separators)
        if isinstance(raw, list) and all(isinstance(x, str) for x in raw):
            return [s for s in raw if s]
        logger.warning("separators must be a list of strings; ignoring")
        return None
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("invalid separators JSON, using default: %s", e)
        return None


@kb_router.post(
    "/collections/{collection}/config",
    response_model=CollectionOperationResult,
)
@handle_kb_exceptions
async def save_collection_config(
    collection: str,
    config: IngestionConfig = Body(...),
    _user: User = Depends(get_current_user),
) -> CollectionOperationResult:
    """Save ingestion configuration for a specific collection."""
    from ...core.tools.core.RAG_tools.storage.factory import get_metadata_store

    try:
        safe_collection = sanitize_path_component(collection, "collection")
    except ValueError as e:
        raise HTTPException(
            status_code=422, detail=f"Invalid collection name: {str(e)}"
        ) from e

    await _ensure_collection_access(safe_collection, _user, allow_create=True)

    config_json = config.model_dump_json(exclude_unset=True)

    try:
        metadata_store = get_metadata_store()
        await metadata_store.save_collection_config(
            collection=safe_collection,
            config_json=config_json,
            user_id=int(_user.id),
        )

        return CollectionOperationResult(
            status="success",
            collection=safe_collection,
            operation="save_config",
            message=f"Configuration saved for collection '{safe_collection}'",
        )
    except Exception as e:
        logger.error("Failed to save collection config: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@kb_router.post(
    "/ingest",
    response_model=IngestionResult,
)
@with_kb_user_scope
@handle_kb_exceptions
async def ingest(
    collection: str = Form(None),
    file: UploadFile = File(...),
    *,
    # Ingestion configuration parameters
    parse_method: Optional[ParseMethod] = Form(
        None,
        description="Parser used during ingestion. Options: default, pypdf, pdfplumber, unstructured, pymupdf, deepdoc",
    ),
    chunk_strategy: Optional[ChunkStrategy] = Form(
        None,
        description="Chunking strategy. Options: recursive (default), fixed_size, markdown",
    ),
    chunk_size: Optional[int] = Form(
        None,
        gt=0,
        description="Chunk size in characters (default: 1000)",
    ),
    chunk_overlap: Optional[int] = Form(
        None,
        ge=0,
        description="Chunk overlap in characters (default: 200)",
    ),
    separators: Optional[str] = Form(
        None,
        description=(
            "Custom chunk separators as JSON array of strings, e.g. "
            '["\\n\\n", "\\n", "。"]. Only used when chunk_strategy is recursive. '
            "Omit or empty to use default separators."
        ),
    ),
    embedding_model_id: str = Form(
        "text-embedding-v4",
        description="Embedding model ID (default: text-embedding-v4)",
    ),
    embedding_batch_size: Optional[int] = Form(
        None,
        gt=0,
        description="Batch size for embedding (default: 10)",
    ),
    max_retries: Optional[int] = Form(
        None,
        ge=0,
        description="Maximum retries for embedding failures (default: 3)",
    ),
    retry_delay: Optional[float] = Form(
        None,
        ge=0.0,
        description="Delay between retries in seconds (default: 1.0)",
    ),
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> IngestionResult | JSONResponse:
    """Upload and ingest a document into the knowledge base.

    Args:
        collection: Target collection name. If not provided, uses file name.
        file: The document file to upload and process.
        parse_method: Parser used during ingestion.
        chunk_strategy: Strategy for chunking the document.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks.
        separators: Optional JSON array of custom chunk separators (recursive only).
        embedding_model_id: Embedding model ID from model hub.
        embedding_batch_size: Batch size for embedding operations.
        max_retries: Maximum retry attempts for failures.
        retry_delay: Delay between retry attempts in seconds.
    """
    if not file.filename or not file.filename.strip():
        raise HTTPException(status_code=422, detail="No filename provided")

    # SECURITY: Extract only basename to prevent path traversal attacks
    # e.g., "../../../etc/passwd.pdf" becomes "passwd.pdf"
    safe_filename = Path(file.filename).name

    if not is_allowed_file(safe_filename, "general"):
        raise HTTPException(
            status_code=422,
            detail=f"File type {Path(safe_filename).suffix.lower()} not supported",
        )

    _validate_parser_for_file(
        safe_filename, parse_method, user_id=getattr(_user, "id", None)
    )

    if not collection or not collection.strip():
        collection = Path(safe_filename).stem
        logger.info("Using file name as collection: %s", collection)

    try:
        # SECURITY: Validate collection name at API boundary
        safe_collection = sanitize_path_component(collection, "collection")
        collection = safe_collection

        file_path = Path(
            get_upload_path(
                safe_filename,
                user_id=int(_user.id),
                collection=safe_collection,
                collection_is_sanitized=True,
            )
        )
    except ValueError as e:
        logger.warning("Invalid collection name rejected: %s - %s", collection, e)
        raise HTTPException(
            status_code=422, detail=f"Invalid collection name: {str(e)}"
        ) from e

    await _ensure_collection_access(safe_collection, _user, allow_create=True)

    try:
        get_collection_sync(safe_collection)
        collection_existed_before = True
    except ValueError:
        collection_existed_before = False

    existing_file_record = (
        db.query(UploadedFile)
        .filter(UploadedFile.storage_path == str(file_path))
        .first()
    )
    uploaded_file_existed_before = existing_file_record is not None
    had_existing_file = file_path.exists()
    file_backup_path: Optional[Path] = None
    if had_existing_file:
        file_backup_path = file_path.with_name(
            f"{file_path.name}.rollback-{uuid.uuid4().hex}"
        )
        shutil.copy2(file_path, file_backup_path)

    try:
        total_size = 0
        # Must not shadow the Form parameter ``chunk_size`` (see issue #199).
        file_read_buffer_size = 1024 * 1024  # 1MB streaming read buffer only
        with open(file_path, "wb") as buffer:
            while True:
                chunk = await file.read(file_read_buffer_size)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File size exceeds maximum limit of {MAX_FILE_SIZE_LABEL}"
                        ),
                    )
                buffer.write(chunk)
        logger.info(
            "File uploaded: %s -> %s (user: %s, collection: %s)",
            safe_filename,
            file_path,
            _user.id,
            safe_collection,
        )
    except HTTPException:
        # Ensure partial file is removed on early abort (e.g., file too large)
        try:
            _restore_ingest_file_backup(
                file_path=file_path,
                backup_path=file_backup_path,
                had_existing_file=had_existing_file,
            )
        except Exception as restore_exc:  # noqa: BLE001
            raise RollbackFailureError(
                "Failed to restore ingest file after upload abort for "
                f"{collection}/{file_path.name}: {restore_exc}"
            ) from restore_exc
        raise
    except Exception as upload_exc:
        try:
            _restore_ingest_file_backup(
                file_path=file_path,
                backup_path=file_backup_path,
                had_existing_file=had_existing_file,
            )
        except Exception as restore_exc:  # noqa: BLE001
            raise RollbackFailureError(
                "Failed to restore ingest file after upload error for "
                f"{collection}/{file_path.name}: {restore_exc}"
            ) from restore_exc
        raise upload_exc

    # Register file in unified file management (file_id) for KB + file APIs.
    mime_type = (
        getattr(file, "content_type", None)
        or mimetypes.guess_type(safe_filename)[0]
        or "application/octet-stream"
    )
    file_record: Optional[UploadedFile] = None

    final_chunk_size = chunk_size if chunk_size is not None and chunk_size > 0 else 1000
    final_chunk_overlap = (
        chunk_overlap if chunk_overlap is not None and chunk_overlap >= 0 else 200
    )
    if final_chunk_overlap >= final_chunk_size:
        final_chunk_overlap = min(int(final_chunk_size * 0.2), final_chunk_size - 1)
        logger.warning(
            "Auto-adjusting chunk_overlap to %s to ensure it's less than chunk_size (%s)",
            final_chunk_overlap,
            final_chunk_size,
        )

    parsed_separators = _parse_separators(separators)
    final_strategy = (
        chunk_strategy if chunk_strategy is not None else ChunkStrategy.RECURSIVE
    )
    if separators and separators.strip() and final_strategy != ChunkStrategy.RECURSIVE:
        logger.warning(
            "separators are only used when chunk_strategy is recursive; "
            "current strategy is %s, ignoring separators",
            final_strategy.value,
        )

    normalized_parse_method = _normalize_parse_method_for_filename(
        parse_method, safe_filename
    )

    config = IngestionConfig(
        parse_method=normalized_parse_method,
        chunk_strategy=final_strategy,
        chunk_size=final_chunk_size,
        chunk_overlap=final_chunk_overlap,
        separators=parsed_separators,
        embedding_model_id=embedding_model_id,
        embedding_batch_size=embedding_batch_size
        if embedding_batch_size is not None and embedding_batch_size > 0
        else 10,
        max_retries=max_retries if max_retries is not None and max_retries >= 0 else 3,
        retry_delay=retry_delay
        if retry_delay is not None and retry_delay >= 0
        else 1.0,
    )

    progress_manager = get_progress_manager()

    try:
        from ...core.tools.core.RAG_tools.storage.factory import get_metadata_store

        metadata_store = get_metadata_store()
        await metadata_store.save_collection_config(
            collection=safe_collection,
            config_json=config.model_dump_json(exclude_unset=True),
            user_id=int(_user.id),
        )
    except Exception as e:
        logger.warning("Failed to save collection config during ingest: %s", e)

    try:
        file_record = _upsert_uploaded_file_record(
            db,
            user_id=int(_user.id),
            filename=safe_filename,
            storage_path=file_path,
            mime_type=mime_type,
            file_size=int(total_size),
        )

        def _run_ingestion() -> IngestionResult:
            return run_document_ingestion(
                collection=safe_collection,
                source_path=str(file_path),
                ingestion_config=config,
                progress_manager=progress_manager,
                user_id=int(_user.id),
                is_admin=bool(_user.is_admin),
                file_id=str(file_record.file_id),
            )

        loop = asyncio.get_running_loop()
        result: IngestionResult = await loop.run_in_executor(None, _run_ingestion)

        if result.status in {"error", "partial"}:
            await _rollback_failed_ingestion(
                db=db,
                user=_user,
                collection_name=collection,
                result=result,
                file_path=file_path,
                file_record=file_record,
                collection_existed_before=collection_existed_before,
                uploaded_file_existed_before=uploaded_file_existed_before,
                file_backup_path=file_backup_path,
                had_existing_file=had_existing_file,
            )

        if result.status == "error":
            return JSONResponse(
                status_code=500,
                content={**result.model_dump(), "status": "error"},
            )
        if result.status == "partial":
            logger.warning(
                "KB ingest partially completed (collection=%s, filename=%s, user_id=%s): %s",
                collection,
                safe_filename,
                _user.id,
                result.message,
            )
            return JSONResponse(
                status_code=500,
                content={**result.model_dump(), "status": "error"},
            )

        if file_backup_path is not None and file_backup_path.exists():
            try:
                file_backup_path.unlink()
            except OSError:
                logger.warning("Failed to remove ingest backup %s", file_backup_path)

        return JSONResponse(
            status_code=200,
            content={**result.model_dump(), "file_id": file_record.file_id},
        )
    except RollbackFailureError:
        raise
    except Exception:
        if file_record is not None:
            rollback_result = IngestionResult(
                status="error",
                doc_id=safe_filename,
                message="Ingestion setup failed before completion.",
            )
            await _rollback_failed_ingestion(
                db=db,
                user=_user,
                collection_name=collection,
                result=rollback_result,
                file_path=file_path,
                file_record=file_record,
                collection_existed_before=collection_existed_before,
                uploaded_file_existed_before=uploaded_file_existed_before,
                file_backup_path=file_backup_path,
                had_existing_file=had_existing_file,
            )
        elif not collection_existed_before:
            _restore_ingest_file_backup(
                file_path=file_path,
                backup_path=file_backup_path,
                had_existing_file=had_existing_file,
            )
            await _cleanup_failed_new_collection_metadata(
                collection_name=collection,
                user=_user,
            )
        else:
            _restore_ingest_file_backup(
                file_path=file_path,
                backup_path=file_backup_path,
                had_existing_file=had_existing_file,
            )
        raise


@kb_router.post("/ingest-cloud", response_model=List[IngestionResult])
@handle_kb_exceptions
async def ingest_cloud(
    request: CloudIngestRequest,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> List[IngestionResult]:
    """Ingest files from cloud storage."""
    try:
        safe_collection = sanitize_path_component(request.collection, "collection")
    except ValueError as e:
        raise HTTPException(
            status_code=422, detail=f"Invalid collection name: {str(e)}"
        ) from e

    results = []

    # Common configuration setup
    final_chunk_size = (
        request.chunk_size if request.chunk_size and request.chunk_size > 0 else 1000
    )
    final_chunk_overlap = (
        request.chunk_overlap
        if request.chunk_overlap and request.chunk_overlap >= 0
        else 200
    )
    if final_chunk_overlap >= final_chunk_size:
        final_chunk_overlap = min(int(final_chunk_size * 0.2), final_chunk_size - 1)

    config = IngestionConfig(
        parse_method=request.parse_method or ParseMethod.DEFAULT,
        chunk_strategy=request.chunk_strategy or ChunkStrategy.RECURSIVE,
        chunk_size=final_chunk_size,
        chunk_overlap=final_chunk_overlap,
        separators=request.separators,
        embedding_model_id=request.embedding_model_id,
        embedding_batch_size=request.embedding_batch_size or 10,
        max_retries=request.max_retries or 3,
        retry_delay=request.retry_delay or 1.0,
    )

    progress_manager = get_progress_manager()

    try:
        get_collection_sync(safe_collection)
        collection_existed_before = True
    except ValueError:
        collection_existed_before = False

    await _ensure_collection_access(safe_collection, _user, allow_create=True)

    try:
        from ...core.tools.core.RAG_tools.storage.factory import get_metadata_store

        metadata_store = get_metadata_store()
        await metadata_store.save_collection_config(
            collection=safe_collection,
            config_json=config.model_dump_json(exclude_unset=True),
            user_id=int(_user.id),
        )
    except Exception as e:
        logger.warning("Failed to save collection config during ingest_cloud: %s", e)

    # Concurrency limit for cloud ingestion to avoid overloading
    semaphore = asyncio.Semaphore(5)

    async def process_file(file_info: CloudFile) -> IngestionResult:
        async with semaphore:
            file_record: Optional[UploadedFile] = None
            file_backup_path: Optional[Path] = None
            had_existing_file = False
            uploaded_file_existed_before = False
            safe_filename = Path(file_info.fileName).name
            storage_filename = _build_cloud_storage_filename(
                safe_filename,
                file_info.fileId,
            )
            file_path = Path(get_upload_path(storage_filename, user_id=int(_user.id)))
            try:
                _validate_parser_for_file(
                    safe_filename,
                    request.parse_method,
                    user_id=int(_user.id),
                )
            except HTTPException as ve:
                return IngestionResult(
                    status="error",
                    message=ve.detail,
                    doc_id=file_info.fileName,
                )
            try:
                if file_info.provider == "google-drive":
                    # Get credentials (run in thread to avoid blocking)
                    try:
                        creds = await asyncio.to_thread(
                            get_google_credentials, int(_user.id), db
                        )
                    except HTTPException as e:
                        return IngestionResult(
                            status="error",
                            message=f"Authentication error: {e.detail}",
                            doc_id=file_info.fileName,
                        )

                    # Build service (blocking)
                    service = await asyncio.to_thread(
                        build, "drive", "v3", credentials=creds, cache_discovery=False
                    )

                    # Save to local path
                    had_existing_file = file_path.exists()
                    if had_existing_file:
                        file_backup_path = file_path.with_name(
                            f"{file_path.name}.rollback-{uuid.uuid4().hex}"
                        )
                        shutil.copy2(file_path, file_backup_path)

                    # Download file directly to disk
                    try:

                        def _download_file() -> None:
                            request_file = service.files().get_media(
                                fileId=file_info.fileId
                            )
                            with open(file_path, "wb") as fh:
                                downloader = MediaIoBaseDownload(fh, request_file)
                                done = False
                                while done is False:
                                    status, done = downloader.next_chunk()

                        await asyncio.to_thread(_download_file)

                    except Exception as e:
                        try:
                            _restore_ingest_file_backup(
                                file_path=file_path,
                                backup_path=file_backup_path,
                                had_existing_file=had_existing_file,
                            )
                        except Exception as restore_exc:  # noqa: BLE001
                            return IngestionResult(
                                status="error",
                                message=(
                                    "Failed to fully roll back cloud ingest for "
                                    f"{safe_collection}/{file_info.fileName}: {restore_exc}"
                                ),
                                doc_id=file_info.fileName,
                            )
                        return IngestionResult(
                            status="error",
                            message=f"Download failed: {str(e)}",
                            doc_id=file_info.fileName,
                        )

                    uploaded_file_existed_before = (
                        db.query(UploadedFile)
                        .filter(UploadedFile.storage_path == str(file_path))
                        .first()
                        is not None
                    )

                    file_record = _upsert_uploaded_file_record(
                        db,
                        user_id=int(_user.id),
                        filename=safe_filename,
                        storage_path=file_path,
                        mime_type=(
                            mimetypes.guess_type(safe_filename)[0]
                            or "application/octet-stream"
                        ),
                        file_size=int(file_path.stat().st_size),
                    )

                    # Run ingestion (blocking)
                    try:
                        normalized_parse_method = _normalize_parse_method_for_filename(
                            request.parse_method,
                            safe_filename,
                        )
                        file_config = config.model_copy(
                            update={"parse_method": normalized_parse_method}
                        )
                        result = await asyncio.to_thread(
                            run_document_ingestion,
                            collection=safe_collection,
                            source_path=str(file_path),
                            ingestion_config=file_config,
                            progress_manager=progress_manager,
                            user_id=int(_user.id),
                            is_admin=bool(_user.is_admin),
                            file_id=str(file_record.file_id),
                        )
                        if result.status in {"error", "partial"}:
                            await _rollback_failed_cloud_ingestion(
                                db=db,
                                user=_user,
                                collection_name=safe_collection,
                                result=result,
                                file_path=file_path,
                                file_record=file_record,
                                collection_existed_before=collection_existed_before,
                                uploaded_file_existed_before=uploaded_file_existed_before,
                                file_backup_path=file_backup_path,
                                had_existing_file=had_existing_file,
                            )
                        elif file_backup_path is not None:
                            try:
                                file_backup_path.unlink(missing_ok=True)
                            except OSError:
                                pass
                        return result
                    except RollbackFailureError as rollback_exc:
                        return IngestionResult(
                            status="error",
                            message=str(rollback_exc),
                            doc_id=file_info.fileName,
                        )
                    except Exception as e:
                        rollback_result = IngestionResult(
                            status="error",
                            doc_id=file_info.fileName,
                            message=f"Ingestion failed: {str(e)}",
                        )
                        await _rollback_failed_cloud_ingestion(
                            db=db,
                            user=_user,
                            collection_name=safe_collection,
                            result=rollback_result,
                            file_path=file_path,
                            file_record=file_record,
                            collection_existed_before=collection_existed_before,
                            uploaded_file_existed_before=uploaded_file_existed_before,
                            file_backup_path=file_backup_path,
                            had_existing_file=had_existing_file,
                        )
                        return IngestionResult(
                            status="error",
                            message=f"Ingestion failed: {str(e)}",
                            doc_id=file_info.fileName,
                        )

                else:
                    return IngestionResult(
                        status="error",
                        message=f"Unsupported provider: {file_info.provider}",
                        doc_id=file_info.fileName,
                    )

            except RollbackFailureError as e:
                logger.exception("Rollback failed for %s: %s", file_info.fileName, e)
                return IngestionResult(
                    status="error",
                    message=str(e),
                    doc_id=file_info.fileName,
                )
            except Exception as e:
                try:
                    _restore_ingest_file_backup(
                        file_path=file_path,
                        backup_path=file_backup_path,
                        had_existing_file=had_existing_file,
                    )
                except Exception as restore_exc:  # noqa: BLE001
                    logger.exception(
                        "Rollback failed for %s: %s", file_info.fileName, restore_exc
                    )
                    return IngestionResult(
                        status="error",
                        message=(
                            "Failed to fully roll back cloud ingest for "
                            f"{safe_collection}/{file_info.fileName}: {restore_exc}"
                        ),
                        doc_id=file_info.fileName,
                    )
                logger.exception(
                    "Unexpected error ingesting %s: %s", file_info.fileName, e
                )
                return IngestionResult(
                    status="error",
                    message=f"Unexpected error: {str(e)}",
                    doc_id=file_info.fileName,
                )

    # Run all file processings concurrently
    results = await asyncio.gather(*[process_file(f) for f in request.files])

    if not collection_existed_before and not any(
        result.status == "success" for result in results
    ):
        await _cleanup_failed_new_collection_metadata(
            collection_name=safe_collection,
            user=_user,
        )

    return results


@kb_router.get(
    "/collections",
    response_model=ListCollectionsResult,
)
@with_kb_user_scope
@handle_kb_exceptions
async def list_collections_api(
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ListCollectionsResult:
    """List all collections with their statistics."""
    kb_collections_timeout_seconds = 15

    try:
        result = await asyncio.wait_for(
            list_collections(user_id=int(_user.id), is_admin=bool(_user.is_admin)),
            timeout=kb_collections_timeout_seconds,
        )

        # Backward compatibility: some unit tests (and older callers) mock or return a
        # plain dict payload. In that case, skip post-processing and return it as-is.
        if isinstance(result, dict):
            return result

        # Fallback: when LanceDB documents table has legacy decode issues, collection
        # stats can still be built from chunks/parses but document_names may be empty.
        # In that case, fill names from UploadedFile rows under user_{id}/{collection}/.
        # Note: This is temporary compatibility code for legacy data. After running
        # the backfill migration (backfill_documents_file_id.py), this should no longer
        # be needed and can be removed.
        if result.collections:
            document_metadata_by_collection: Dict[
                str, List[CollectionDocumentMetadata]
            ] = {}
            document_metadata_seen: Dict[str, set[tuple[str, str, str]]] = {}
            fallback_names: Dict[str, set[str]] = {}

            def _collection_needs_document_scan(collection: Any) -> bool:
                if collection.document_metadata:
                    return False
                return (not collection.document_names) or (
                    collection.documents != len(collection.document_names)
                )

            collections_needing_scan = [
                collection
                for collection in result.collections
                if _collection_needs_document_scan(collection)
                and not document_metadata_by_collection.get(collection.name)
            ]
            scan_target_names = {c.name for c in collections_needing_scan}

            def _normalize_optional_identifier(value: Any) -> Optional[str]:
                if not isinstance(value, str):
                    return None
                normalized = value.strip()
                return normalized or None

            def _add_collection_document_metadata(
                collection_name: str,
                filename: Any,
                *,
                file_id: Optional[str] = None,
                doc_id: Optional[str] = None,
            ) -> None:
                if not isinstance(filename, str):
                    return
                normalized_filename = filename.strip()
                if not normalized_filename:
                    return

                normalized_file_id = _normalize_optional_identifier(file_id)
                normalized_doc_id = _normalize_optional_identifier(doc_id)
                dedupe_key = (
                    normalized_filename,
                    normalized_file_id or "",
                    normalized_doc_id or "",
                )
                seen_keys = document_metadata_seen.setdefault(collection_name, set())
                if dedupe_key in seen_keys:
                    return
                seen_keys.add(dedupe_key)
                document_metadata_by_collection.setdefault(collection_name, []).append(
                    CollectionDocumentMetadata(
                        filename=normalized_filename,
                        file_id=normalized_file_id,
                        doc_id=normalized_doc_id,
                    )
                )

            for collection in result.collections:
                for document_metadata in collection.document_metadata:
                    _add_collection_document_metadata(
                        collection.name,
                        document_metadata.filename,
                        file_id=document_metadata.file_id,
                        doc_id=document_metadata.doc_id,
                    )

            if collections_needing_scan:
                try:
                    doc_records = _list_documents_for_user(
                        user_id=int(_user.id),
                        is_admin=bool(_user.is_admin),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to list documents for metadata fallback: %s", exc
                    )
                    doc_records = []

                if doc_records:
                    filename_map = _build_uploaded_filename_map(
                        db,
                        user_id=int(_user.id),
                        file_ids=[
                            file_id
                            for file_id in (
                                _get_document_record_file_id(record)
                                for record in doc_records
                            )
                            if file_id
                        ],
                    )
                    for doc_rec in doc_records:
                        rec_collection = doc_rec.get("collection")
                        if not isinstance(rec_collection, str) or not rec_collection:
                            continue
                        if rec_collection not in scan_target_names:
                            continue
                        resolved_filename = _resolve_document_filename(
                            doc_rec, filename_map
                        )
                        resolved_doc_id = _normalize_optional_identifier(
                            doc_rec.get("doc_id")
                        )
                        _add_collection_document_metadata(
                            rec_collection,
                            resolved_filename or resolved_doc_id,
                            file_id=_get_document_record_file_id(doc_rec),
                            doc_id=resolved_doc_id,
                        )

                collections_needing_fallback = [
                    collection
                    for collection in collections_needing_scan
                    if not document_metadata_by_collection.get(collection.name)
                ]

                if collections_needing_fallback:
                    # Filter at SQL level to only load relevant uploaded files
                    collection_patterns = [
                        _like_contains_pattern(f"/user_{int(_user.id)}/{c.name}/")
                        for c in collections_needing_fallback
                    ]

                    uploaded_records = []
                    if len(collection_patterns) == 1:
                        uploaded_records = (
                            db.query(UploadedFile)
                            .filter(
                                UploadedFile.user_id == int(_user.id),
                                UploadedFile.storage_path.like(
                                    collection_patterns[0],
                                    escape=_SQL_LIKE_ESCAPE,
                                ),
                            )
                            .all()
                        )
                    else:
                        # Multiple collections: use OR logic
                        from sqlalchemy import or_

                        uploaded_records = (
                            db.query(UploadedFile)
                            .filter(
                                UploadedFile.user_id == int(_user.id),
                                or_(
                                    *[
                                        UploadedFile.storage_path.like(
                                            pattern,
                                            escape=_SQL_LIKE_ESCAPE,
                                        )
                                        for pattern in collection_patterns
                                    ]
                                ),
                            )
                            .all()
                        )

                    user_segment = f"user_{int(_user.id)}"
                    for rec in uploaded_records:
                        storage_path = Path(str(getattr(rec, "storage_path", "")))
                        parts = storage_path.parts
                        if user_segment not in parts:
                            continue
                        user_idx = parts.index(user_segment)
                        if user_idx + 2 >= len(parts):
                            continue
                        collection_name = parts[user_idx + 1]
                        if collection_name not in scan_target_names:
                            continue
                        fallback_filename = str(getattr(rec, "filename", "")).strip()
                        fallback_names.setdefault(collection_name, set()).add(
                            fallback_filename
                        )
                        fallback_file_id = _normalize_optional_identifier(
                            getattr(rec, "file_id", None)
                        )
                        fallback_doc_id = None
                        if str(getattr(rec, "storage_path", "")).strip():
                            fallback_doc_id = generate_deterministic_doc_id(
                                collection_name,
                                str(getattr(rec, "storage_path", "")).strip(),
                            )
                        _add_collection_document_metadata(
                            collection_name,
                            fallback_filename,
                            file_id=fallback_file_id,
                            doc_id=fallback_doc_id,
                        )

            for collection in result.collections:
                resolved_metadata = sorted(
                    document_metadata_by_collection.get(collection.name, []),
                    key=lambda item: (
                        item.filename,
                        item.file_id or "",
                        item.doc_id or "",
                    ),
                )
                collection.document_metadata = resolved_metadata
                if not collection.document_names and resolved_metadata:
                    collection.document_names = sorted(
                        {item.filename for item in resolved_metadata if item.filename}
                    )
                    if collection.documents == 0:
                        collection.documents = len(collection.document_names)
                    continue

                fallback = sorted(
                    name for name in fallback_names.get(collection.name, set()) if name
                )
                if fallback:
                    collection.document_names = fallback
                    if collection.documents == 0:
                        collection.documents = len(fallback)

        return result
    except asyncio.TimeoutError:
        logger.error(
            "Listing KB collections timed out after %s seconds",
            kb_collections_timeout_seconds,
        )
        raise HTTPException(
            status_code=503,
            detail="Knowledge base is temporarily unavailable. Please retry.",
        )


@kb_router.post(
    "/search",
    response_model=SearchPipelineResult,
)
@with_kb_user_scope
@handle_kb_exceptions
async def search(
    collection: str = Form(..., description="Target collection to search within"),
    query_text: str = Form(..., description="Query text to search for"),
    embedding_model_id: str = Form(
        "text-embedding-v4",
        description="Embedding model ID (default: text-embedding-v4)",
    ),
    *,
    # Search configuration parameters
    search_type: Optional[SearchType] = Form(
        None,
        description="Search strategy: dense, sparse, or hybrid (default: hybrid)",
    ),
    top_k: Optional[int] = Form(
        None,
        ge=1,
        le=100,
        description="Maximum number of results to return (default: 5)",
    ),
    filters: Optional[Dict[str, Any]] = Form(
        None,
        description="Optional filters to apply during search. "
        "Format: {field: value} for equality filters. "
        "For advanced filters, use {field: {operator: str, value: Any}} "
        "where operator can be: eq, ne, gt, gte, lt, lte, in, contains.",
    ),
    fusion_config: Optional[Dict[str, Any]] = Form(
        None,
        description="Optional fusion configuration for hybrid search",
    ),
    rerank_model_id: Optional[str] = Form(
        None,
        description="Optional rerank model ID for result reordering",
    ),
    rerank_top_k: Optional[int] = Form(
        None,
        description="Optional override for rerank result count",
    ),
    readonly: Optional[bool] = Form(
        None,
        description="Avoid index modifications (default: False)",
    ),
    nprobes: Optional[int] = Form(
        None,
        description="Number of partitions to probe for ANN search",
    ),
    refine_factor: Optional[int] = Form(
        None,
        description="Refine factor for ANN search re-ranking",
    ),
    fallback_to_sparse: Optional[bool] = Form(
        None,
        description="Allow hybrid search to fallback to sparse (default: True)",
    ),
    _user: User = Depends(get_current_user),
) -> SearchPipelineResult:
    """Search documents in the knowledge base.

    Args:
        collection: Target collection to search within.
        query_text: Query text to search for.
        embedding_model_id: Embedding model ID (required for dense/hybrid search).
        search_type: Search strategy (dense, sparse, or hybrid).
        top_k: Maximum number of results to return.
        filters: Optional filters for search.
        fusion_config: Optional fusion configuration for hybrid search.
        rerank_model_id: Optional rerank model for result reordering.
        rerank_top_k: Override for rerank result count.
        readonly: Whether to avoid index modifications.
        nprobes: Number of partitions to probe for ANN search.
        refine_factor: Refine factor for ANN search re-ranking.
        fallback_to_sparse: Allow hybrid search to fallback to sparse.
    """
    # CRITICAL: Handle empty strings from Swagger UI - convert to None BEFORE any processing
    if filters == "":
        filters = None
    if fusion_config == "":
        fusion_config = None

    if not collection or not query_text:
        raise HTTPException(status_code=422, detail="Missing required parameters")

    try:
        safe_collection = sanitize_path_component(collection, "collection")
    except ValueError as e:
        raise HTTPException(
            status_code=422, detail=f"Invalid collection name: {str(e)}"
        ) from e

    if not embedding_model_id:
        raise HTTPException(
            status_code=422,
            detail="embedding_model_id is required",
        )

    await _ensure_collection_access(safe_collection, _user, hide_missing=False)

    # Build configuration from individual parameters
    config = SearchConfig(
        search_type=search_type or SearchType.HYBRID,
        top_k=top_k or 5,
        filters=filters,
        fusion_config=FusionConfig.model_validate(fusion_config)
        if fusion_config
        else None,
        embedding_model_id=embedding_model_id,
        rerank_model_id=rerank_model_id,
        rerank_top_k=rerank_top_k,
        readonly=readonly or False,
        nprobes=nprobes,
        refine_factor=refine_factor,
        fallback_to_sparse=fallback_to_sparse
        if fallback_to_sparse is not None
        else True,
    )

    progress_manager = get_progress_manager()
    result = run_document_search(
        collection=safe_collection,
        query_text=query_text,
        config=config,
        progress_manager=progress_manager,
        user_id=int(_user.id),
        is_admin=bool(_user.is_admin),
    )

    return result


@kb_router.post(
    "/ingest-web",
    response_model=WebIngestionResult,
)
@with_kb_user_scope
@handle_kb_exceptions
async def ingest_web(
    collection: str = Form(..., description="Target collection name"),
    start_url: str = Form(..., description="Starting URL for crawling"),
    # WebCrawlConfig parameters
    max_pages: Optional[int] = Form(
        100,
        description="Maximum number of pages to crawl (default: 100)",
    ),
    max_depth: Optional[int] = Form(
        3,
        description="Maximum crawl depth (default: 3)",
    ),
    url_patterns: Optional[str] = Form(
        None,
        description="Comma-separated URL match patterns (regex)",
    ),
    exclude_patterns: Optional[str] = Form(
        None,
        description="Comma-separated exclusion patterns (regex)",
    ),
    same_domain_only: Optional[bool] = Form(
        True,
        description="Only crawl same domain (default: True)",
    ),
    content_selector: Optional[str] = Form(
        None,
        description="CSS selector for main content area",
    ),
    remove_selectors: Optional[str] = Form(
        None,
        description="Comma-separated CSS selectors to remove",
    ),
    concurrent_requests: Optional[int] = Form(
        3,
        ge=1,
        le=10,
        description="Concurrent requests (default: 3, max: 10)",
    ),
    request_delay: Optional[float] = Form(
        1.0,
        ge=0,
        description="Delay between requests in seconds (default: 1.0)",
    ),
    timeout: Optional[int] = Form(
        30,
        ge=1,
        description="Request timeout in seconds (default: 30)",
    ),
    respect_robots_txt: Optional[bool] = Form(
        True,
        description="Respect robots.txt (default: True)",
    ),
    # IngestionConfig parameters
    parse_method: Optional[ParseMethod] = Form(
        None,
        description="Parser used during ingestion",
    ),
    chunk_strategy: Optional[ChunkStrategy] = Form(
        None,
        description="Chunking strategy",
    ),
    chunk_size: Optional[int] = Form(
        None,
        gt=0,
        description="Chunk size in characters (default: 1000)",
    ),
    chunk_overlap: Optional[int] = Form(
        None,
        ge=0,
        description="Chunk overlap (default: 200)",
    ),
    separators: Optional[str] = Form(
        None,
        description=(
            "Custom chunk separators as JSON array of strings; "
            "only used when chunk_strategy is recursive."
        ),
    ),
    embedding_model_id: str = Form(
        "text-embedding-v4",
        description="Embedding model ID",
    ),
    embedding_batch_size: Optional[int] = Form(
        None,
        gt=0,
        description="Batch size for embedding (default: 10)",
    ),
    max_retries: Optional[int] = Form(
        None,
        ge=0,
        description="Maximum retries for embedding failures (default: 3)",
    ),
    retry_delay: Optional[float] = Form(
        None,
        ge=0.0,
        description="Delay between retries in seconds (default: 1.0)",
    ),
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> WebIngestionResult | JSONResponse:
    """Ingest website content into the knowledge base.

    Args:
        collection: Target collection name
        start_url: Starting URL for crawling
        max_pages: Maximum number of pages to crawl
        max_depth: Maximum crawl depth
        url_patterns: Comma-separated URL match patterns (regex)
        exclude_patterns: Comma-separated exclusion patterns (regex)
        same_domain_only: Only crawl same domain
        content_selector: CSS selector for main content area
        remove_selectors: Comma-separated CSS selectors to remove
        concurrent_requests: Number of concurrent requests
        request_delay: Delay between requests in seconds
        timeout: Request timeout in seconds
        respect_robots_txt: Respect robots.txt rules
        parse_method: Parser for document ingestion
        chunk_strategy: Chunking strategy
        chunk_size: Chunk size in characters
        chunk_overlap: Chunk overlap in characters
        embedding_model_id: Embedding model ID
        embedding_batch_size: Batch size for embedding
        max_retries: Maximum retry attempts
        retry_delay: Delay between retries
    """
    try:
        try:
            safe_collection = sanitize_path_component(collection, "collection")
        except ValueError as e:
            logger.warning("Invalid collection name rejected: %s - %s", collection, e)
            raise HTTPException(
                status_code=422, detail=f"Invalid collection name: {str(e)}"
            ) from e

        await _ensure_collection_access(safe_collection, _user, allow_create=True)

        url_patterns_list = (
            [p.strip() for p in url_patterns.split(",")] if url_patterns else None
        )
        exclude_patterns_list = (
            [p.strip() for p in exclude_patterns.split(",")]
            if exclude_patterns
            else None
        )
        remove_selectors_list = (
            [s.strip() for s in remove_selectors.split(",")]
            if remove_selectors
            else None
        )

        crawl_config = WebCrawlConfig(
            start_url=start_url,
            max_pages=max_pages or 100,
            max_depth=max_depth or 3,
            url_patterns=url_patterns_list,
            exclude_patterns=exclude_patterns_list,
            same_domain_only=(
                same_domain_only if same_domain_only is not None else True
            ),
            content_selector=content_selector,
            remove_selectors=remove_selectors_list,
            concurrent_requests=concurrent_requests or 3,
            request_delay=request_delay or 1.0,
            timeout=timeout or 30,
            respect_robots_txt=(
                respect_robots_txt if respect_robots_txt is not None else True
            ),
        )

        final_chunk_size = (
            chunk_size if chunk_size is not None and chunk_size > 0 else 1000
        )
        final_chunk_overlap = (
            chunk_overlap if chunk_overlap is not None and chunk_overlap >= 0 else 200
        )
        if final_chunk_overlap >= final_chunk_size:
            final_chunk_overlap = min(int(final_chunk_size * 0.2), final_chunk_size - 1)
            logger.warning(
                "Auto-adjusting chunk_overlap from %s to %s to ensure it's less than chunk_size (%s)",
                chunk_overlap,
                final_chunk_overlap,
                final_chunk_size,
            )

        web_parsed_separators = _parse_separators(separators)
        web_final_strategy = (
            chunk_strategy if chunk_strategy is not None else ChunkStrategy.RECURSIVE
        )
        if (
            separators
            and separators.strip()
            and web_final_strategy != ChunkStrategy.RECURSIVE
        ):
            logger.warning(
                "separators are only used when chunk_strategy is recursive; "
                "current strategy is %s, ignoring separators",
                web_final_strategy.value,
            )

        ingestion_config = IngestionConfig(
            parse_method=(
                parse_method if parse_method is not None else ParseMethod.DEFAULT
            ),
            chunk_strategy=web_final_strategy,
            chunk_size=final_chunk_size,
            chunk_overlap=final_chunk_overlap,
            separators=web_parsed_separators,
            embedding_model_id=embedding_model_id,
            embedding_batch_size=(
                embedding_batch_size
                if embedding_batch_size is not None and embedding_batch_size > 0
                else 10
            ),
            max_retries=(
                max_retries if max_retries is not None and max_retries >= 0 else 3
            ),
            retry_delay=(
                retry_delay if retry_delay is not None and retry_delay >= 0 else 1.0
            ),
        )

        try:
            get_collection_sync(safe_collection)
            collection_existed_before = True
        except ValueError:
            collection_existed_before = False

        try:
            from ...core.tools.core.RAG_tools.storage.factory import get_metadata_store

            metadata_store = get_metadata_store()
            await metadata_store.save_collection_config(
                collection=safe_collection,
                config_json=ingestion_config.model_dump_json(exclude_unset=True),
                user_id=int(_user.id),
            )
        except Exception as e:
            logger.warning("Failed to save collection config during ingest_web: %s", e)

        # Track processed URLs to prevent duplicate UploadedFile records
        # Key: URL hash, Value: file_id
        # Note: For large-scale web ingestion (>10000 pages), consider using
        # a bounded-size dict (e.g., with maxitems) to control memory usage.
        _processed_urls: Dict[str, str] = {}

        # Define file handler for persistent storage and UploadedFile record creation
        def _handle_web_file(
            temp_file_path: Path,
            title: str,
            collection_name: str,
            url: str,
            db_session: Session,
        ) -> FileHandlerResult:
            """Handle file persistence and UploadedFile record creation for web ingestion.

            This function:
            1. Checks if a file with this URL already exists (URL-based deduplication)
            2. If exists, reuses the existing file and UploadedFile record
            3. If not, copies the temporary file to the persistent uploads directory
            4. Creates an UploadedFile record in the database
            5. Returns the file_path and file_id for ingestion

            Args:
                temp_file_path: Path to the temporary markdown file
                title: Page title (used for display)
                collection_name: Collection name for organizing files
                url: Source URL (used for unique identification)

            Returns:
                FileHandlerResult with file_path and optional file_id
            """
            # Use URL hash for unique filename (true URL deduplication)
            # Using SHA256 for better collision resistance than MD5
            # Include collection to prevent cross-collection file sharing
            url_hash = hashlib.sha256(f"{collection_name}:{url}".encode()).hexdigest()[
                :16
            ]
            safe_title = _normalize_web_title_for_filename(title)
            filename = f"{url_hash}_{safe_title}.md"
            lock_key = f"{int(_user.id)}:{url_hash}"

            with _WebFileLock(lock_key):
                # Check if we've already processed this URL (in-memory cache)
                if url_hash in _processed_urls:
                    existing_file_id = _processed_urls[url_hash]
                    logger.info(
                        "Reusing existing UploadedFile record for web ingestion: url=%s, file_id=%s",
                        url,
                        existing_file_id,
                    )
                    existing_record = (
                        db_session.query(UploadedFile)
                        .filter(UploadedFile.file_id == existing_file_id)
                        .first()
                    )
                    if existing_record:
                        result = _refresh_existing_file_if_changed(
                            existing_record=existing_record,
                            temp_file_path=temp_file_path,
                            db_session=db_session,
                            user_id=int(_user.id),
                            url=url,
                            filename=filename,
                            url_hash=url_hash,
                            processed_urls=_processed_urls,
                            context="in-memory cache",
                        )
                        if result is not None:
                            return result
                    # Cached file_id was deleted from DB, fall through to recreate
                    logger.warning(
                        "Cached file_id %s not found in DB (record was deleted), will create new record for url=%s",
                        existing_file_id,
                        url,
                    )

                # Check database for existing file with same URL hash (cross-session deduplication)
                existing_record = (
                    db_session.query(UploadedFile)
                    .filter(
                        UploadedFile.user_id == int(_user.id),
                        UploadedFile.filename == filename,
                    )
                    .first()
                )

                if existing_record:
                    result = _refresh_existing_file_if_changed(
                        existing_record=existing_record,
                        temp_file_path=temp_file_path,
                        db_session=db_session,
                        user_id=int(_user.id),
                        url=url,
                        filename=filename,
                        url_hash=url_hash,
                        processed_urls=_processed_urls,
                        context="cross-session",
                    )
                    if result is not None:
                        # File existed and was handled (either unchanged or refreshed)
                        _processed_urls[url_hash] = str(existing_record.file_id)
                        logger.info(
                            "Found existing UploadedFile record from previous session: url=%s, file_id=%s",
                            url,
                            existing_record.file_id,
                        )
                        return result

                    # result is None means file doesn't exist - recreate it
                    existing_path = Path(str(existing_record.storage_path))
                    existing_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(temp_file_path, existing_path)
                    file_record = _upsert_uploaded_file_record(
                        db_session,
                        user_id=int(_user.id),
                        filename=filename,
                        storage_path=existing_path,
                        mime_type="text/markdown",
                        file_size=existing_path.stat().st_size,
                    )
                    _processed_urls[url_hash] = str(file_record.file_id)
                    logger.info(
                        "Recreated missing persistent file for existing UploadedFile record: url=%s, file_id=%s",
                        url,
                        file_record.file_id,
                    )
                    return FileHandlerResult(
                        file_path=str(existing_record.storage_path),
                        file_id=str(existing_record.file_id),
                    )

                persistent_file = get_upload_path(
                    filename,
                    user_id=int(_user.id),
                    collection=collection_name,
                    collection_is_sanitized=True,
                )
                persistent_file.parent.mkdir(parents=True, exist_ok=True)

                try:
                    shutil.copy2(temp_file_path, persistent_file)
                    logger.info(
                        "Copied web ingestion file from %s to %s",
                        temp_file_path,
                        persistent_file,
                    )

                    file_record = _upsert_uploaded_file_record(
                        db_session,
                        user_id=int(_user.id),
                        filename=filename,
                        storage_path=persistent_file,
                        mime_type="text/markdown",
                        file_size=persistent_file.stat().st_size,
                    )
                    logger.info(
                        "Created UploadedFile record for web ingestion: file_id=%s, filename=%s, url=%s",
                        file_record.file_id,
                        filename,
                        url,
                    )

                    _processed_urls[url_hash] = str(file_record.file_id)
                    return FileHandlerResult(
                        file_path=str(persistent_file),
                        file_id=str(file_record.file_id),
                    )
                except Exception:
                    if persistent_file.exists():
                        try:
                            persistent_file.unlink()
                            logger.warning(
                                "Cleaned up orphaned persistent file due to upsert failure: %s",
                                persistent_file,
                            )
                        except Exception as cleanup_error:
                            logger.warning(
                                "Failed to clean up orphaned persistent file %s: %s",
                                persistent_file,
                                cleanup_error,
                            )
                    raise

        # Create a wrapper that creates a dedicated DB session for the executor thread
        # This avoids sharing the request thread's session across thread boundaries,
        # which is fragile and could break with concurrent access.
        def _file_handler_with_db(
            temp_file_path: Path, title: str, collection_name: str, url: str
        ) -> FileHandlerResult:
            # Create a new session for this thread
            SessionLocal = get_session_local()
            db_session = SessionLocal()
            try:
                return _handle_web_file(
                    temp_file_path, title, collection_name, url, db_session
                )
            finally:
                db_session.close()

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: asyncio.run(
                run_web_ingestion(
                    collection=safe_collection,
                    crawl_config=crawl_config,
                    ingestion_config=ingestion_config,
                    user_id=int(_user.id),
                    is_admin=bool(_user.is_admin),
                    file_handler=_file_handler_with_db,
                )
            ),
        )

        if result.status == "error":
            if not collection_existed_before:
                await _cleanup_failed_new_collection_metadata(
                    collection_name=safe_collection,
                    user=_user,
                )
            return JSONResponse(status_code=500, content=result.model_dump())
        if result.status == "partial":
            logger.warning(
                "KB web ingest partially completed (collection=%s, start_url=%s, user_id=%s): %s",
                collection,
                start_url,
                _user.id,
                result.message,
            )

        return result

    except HTTPException:
        raise
    except (ValueError, KeyError, TypeError) as e:
        if "collection_existed_before" in locals() and not collection_existed_before:
            await _cleanup_failed_new_collection_metadata(
                collection_name=safe_collection,
                user=_user,
            )
        logger.error("Data format error in web ingestion: %s", e)
        raise HTTPException(
            status_code=400, detail=f"Data format error: {str(e)}"
        ) from e
    except Exception as e:
        if "collection_existed_before" in locals() and not collection_existed_before:
            await _cleanup_failed_new_collection_metadata(
                collection_name=safe_collection,
                user=_user,
            )
        logger.exception("Unexpected error in web ingestion: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Server internal error: {str(e)}",
        ) from e


class BatchDeleteCollectionsRequest(BaseModel):
    """Request body for batch delete collections."""

    collection_names: List[str] = Field(
        ...,
        min_length=1,
        max_length=200,
        description="List of collection names to delete",
    )


class BatchDeleteFailureItem(BaseModel):
    """One failed deletion in a batch."""

    name: str = Field(..., description="Collection name")
    error: str = Field(..., description="Error message")


class BatchDeleteCollectionsResponse(BaseModel):
    """Response for batch delete collections."""

    deleted: List[str] = Field(
        default_factory=list,
        description="Collection names that were deleted successfully",
    )
    failed: List[BatchDeleteFailureItem] = Field(
        default_factory=list,
        description="Collection names that failed to delete with reasons",
    )


class ResolvedDocumentMatch(TypedDict):
    """Resolved delete target enriched from records and UploadedFile metadata."""

    doc_id: str
    file_id: Optional[str]
    filename: str
    source_path: Optional[str]


def _http_detail_to_str(detail: Any) -> str:
    """Normalize FastAPI/Starlette ``HTTPException.detail`` to a string."""
    if isinstance(detail, str):
        return detail
    try:
        return json.dumps(detail)
    except (TypeError, ValueError):
        return str(detail)


def _check_can_delete_collection(
    collection_name: str,
    user_id: int,
    is_admin: bool,
) -> None:
    """Validate collection name and non-admin delete permission."""
    if not collection_name or not collection_name.strip():
        raise HTTPException(status_code=422, detail="Collection name cannot be empty")
    if is_admin:
        return
    try:
        vector_store = get_vector_index_store()
        total_count = int(
            vector_store.count_documents_grouped_by_collection(
                [collection_name], user_id=None, is_admin=True
            ).get(collection_name, 0)
        )
        own_count = int(
            vector_store.count_documents_grouped_by_collection(
                [collection_name], user_id=user_id, is_admin=False
            ).get(collection_name, 0)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to verify collection delete permission (documents table).",
        ) from exc

    if total_count > 0 and own_count < total_count:
        raise HTTPException(
            status_code=403,
            detail=(
                "Only admin users can delete collections containing documents "
                "from other users."
            ),
        )


def _preflight_batch_delete_permissions(
    unique_names: List[str],
    user_id: int,
    is_admin: bool,
) -> tuple[List[str], List[BatchDeleteFailureItem]]:
    """Preflight validation for batch delete permissions and empty names."""
    failed: List[BatchDeleteFailureItem] = []
    allowed: List[str] = []

    if is_admin:
        for name in unique_names:
            if not name or not name.strip():
                failed.append(
                    BatchDeleteFailureItem(
                        name=name or "",
                        error="Collection name cannot be empty",
                    )
                )
            else:
                allowed.append(name)
        return allowed, failed

    non_empty: List[str] = []
    for name in unique_names:
        if not name or not name.strip():
            failed.append(
                BatchDeleteFailureItem(
                    name=name or "",
                    error="Collection name cannot be empty",
                )
            )
        else:
            non_empty.append(name)

    if not non_empty:
        return [], failed

    vector_store = get_vector_index_store()
    try:
        totals = vector_store.count_documents_grouped_by_collection(
            non_empty, user_id=None, is_admin=True
        )
        owns = vector_store.count_documents_grouped_by_collection(
            non_empty, user_id=int(user_id), is_admin=False
        )
    except Exception as exc:
        logger.error(
            "Batch permission scan failed (vector store grouped counts): %s",
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to scan documents table for batch delete permission.",
        ) from exc

    forbidden_detail = (
        "Only admin users can delete collections containing documents from other users."
    )
    for name in unique_names:
        if not name or not name.strip():
            continue
        key = str(name).strip()
        total = int(totals.get(key, 0))
        own = int(owns.get(key, 0))
        if total > 0 and own < total:
            failed.append(BatchDeleteFailureItem(name=name, error=forbidden_detail))
        else:
            allowed.append(name)

    return allowed, failed


def _perform_kb_collection_delete(
    collection_name: str,
    user_id: int,
    is_admin: bool,
    db: Session,
) -> CollectionOperationResult:
    """Delete one KB collection (same pipeline as single-delete API)."""
    try:
        try:
            safe_collection = sanitize_path_component(collection_name, "collection")
        except ValueError as e:
            raise HTTPException(
                status_code=422, detail=f"Invalid collection name: {str(e)}"
            ) from e

        _check_can_delete_collection(safe_collection, user_id, is_admin)

        collection_dir = get_upload_path(
            "", user_id=user_id, collection=safe_collection
        )

        vector_store = get_vector_index_store()
        collection_records = vector_store.list_document_records(
            collection_name=safe_collection,
            user_id=user_id,
            is_admin=is_admin,
        )
        collection_file_ids = {
            file_id
            for file_id in (
                _get_document_record_file_id(record) for record in collection_records
            )
            if file_id
        }

        # Re-check right before vector deletion to reduce TOCTTOU window.
        _check_can_delete_collection(safe_collection, user_id, is_admin)
        result = delete_collection(safe_collection, user_id, is_admin)

        physical_cleanup = delete_collection_physical_dir(
            user_id=user_id,
            collection_name=safe_collection,
        )
        physical_cleanup_status = physical_cleanup.status
        physical_cleanup_error = physical_cleanup.error
        if physical_cleanup.collection_dir is not None:
            collection_dir = physical_cleanup.collection_dir

        if result.status == "error":
            cleanup_warnings = list(result.warnings) if result.warnings else []
            if physical_cleanup_status == "success":
                cleanup_warnings.append(
                    f"Physical directory moved to trash: {collection_dir} "
                    "(trash cleanup requires external scheduler/cron)"
                )
            elif physical_cleanup_status == "not_found":
                cleanup_warnings.append(
                    "Physical directory cleanup: No physical directory found (collection had no files)"
                )

            return CollectionOperationResult(
                status="error",
                collection=result.collection,
                message=result.message,
                warnings=cleanup_warnings,
                affected_documents=result.affected_documents,
                deleted_counts=result.deleted_counts,
            )

        remaining_records = vector_store.list_document_records(
            collection_name=None,
            user_id=user_id,
            is_admin=is_admin,
        )
        remaining_file_ids = {
            file_id
            for file_id in (
                _get_document_record_file_id(record) for record in remaining_records
            )
            if file_id
        }
        deleted_uploaded_files = 0
        if physical_cleanup_status in {"success", "not_found"}:
            deleted_uploaded_files = delete_collection_uploaded_files(
                db,
                user_id=user_id,
                collection_file_ids=collection_file_ids,
                remaining_file_ids=remaining_file_ids,
                collection_dir=collection_dir,
            )
        else:
            logger.warning(
                "Preserving UploadedFile records for collection %s because physical cleanup status is %s",
                safe_collection,
                physical_cleanup_status,
            )
        if deleted_uploaded_files:
            logger.info(
                "Deleted %s UploadedFile record(s) for collection %s",
                deleted_uploaded_files,
                safe_collection,
            )

        cleanup_warnings = list(result.warnings) if result.warnings else []
        cleanup_info_message = ""

        if physical_cleanup_status == "success":
            cleanup_info = (
                f"Physical directory moved to trash: {collection_dir} "
                "(trash cleanup requires external scheduler/cron)"
            )
            cleanup_warnings.append(cleanup_info)
            cleanup_info_message = f" {cleanup_info}."
        elif physical_cleanup_status == "not_found":
            cleanup_info = "Physical directory cleanup: No physical directory found (collection had no files)"
            cleanup_warnings.append(cleanup_info)
            cleanup_info_message = f" {cleanup_info}."
        elif physical_cleanup_status == "error" and physical_cleanup_error:
            cleanup_info = f"Physical directory cleanup: Warning - {physical_cleanup_error}. Database deletion proceeded, but physical file cleanup status is uncertain."
            cleanup_warnings.append(cleanup_info)
            cleanup_info_message = f" {cleanup_info}"
        elif physical_cleanup_status == "failed" and physical_cleanup_error:
            cleanup_info = (
                f"Physical directory cleanup: Failed - {physical_cleanup_error}"
            )
            cleanup_warnings.append(cleanup_info)
            cleanup_info_message = f" {cleanup_info}"

        final_status = result.status
        if result.status == "success" and physical_cleanup_status in (
            "error",
            "failed",
        ):
            final_status = "partial_success"
            if not cleanup_info_message:
                cleanup_info_message = " Database deletion succeeded, but physical file cleanup encountered issues."

        updated_message = result.message
        if cleanup_info_message:
            updated_message = f"{result.message}{cleanup_info_message}"

        updated_result = CollectionOperationResult(
            status=final_status,
            collection=result.collection,
            message=updated_message,
            warnings=cleanup_warnings,
            affected_documents=result.affected_documents,
            deleted_counts=result.deleted_counts,
        )

        return updated_result

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to delete collection '%s'", collection_name)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete collection: {exc}",
        ) from exc


@kb_router.delete(
    "/collections/{collection_name}",
)
@with_kb_user_scope
@handle_kb_exceptions
async def delete_collection_api(
    collection_name: str,
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CollectionOperationResult:
    """Delete a collection and all its data.

    This function ensures data consistency by attempting physical file deletion
    before database deletion. If physical deletion fails, the operation is
    aborted to prevent inconsistent state.

    Args:
        collection_name: Name of the collection to delete

    Returns:
        Deletion result with status, affected documents, and cleanup information

    Raises:
        HTTPException: If physical deletion fails (prevents database deletion)
    """
    result = _perform_kb_collection_delete(
        collection_name,
        int(_user.id),
        bool(_user.is_admin),
        db,
    )
    return result


@kb_router.post(
    "/collections/batch-delete",
    response_model=BatchDeleteCollectionsResponse,
)
@with_kb_user_scope
@handle_kb_exceptions
async def batch_delete_collections_api(
    body: BatchDeleteCollectionsRequest,
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> BatchDeleteCollectionsResponse:
    """Delete multiple collections in one request.

    For each name, runs the same pipeline as single delete (permissions, physical
    trash, LanceDB, ``UploadedFile`` cleanup). Per-item failures are collected in
    ``failed``; they do not roll back earlier successful deletions in the batch.
    LanceDB removal uses ``delete_collection`` with tenant-aware ``user_id`` and
    ``is_admin`` filtering. Returns ``deleted`` and ``failed`` name lists.
    """
    user_id = int(_user.id)
    is_admin = bool(_user.is_admin)
    deleted: List[str] = []

    # Deduplicate while keeping request order.
    seen: set[str] = set()
    unique_names: List[str] = []
    for raw_name in body.collection_names:
        key = str(raw_name)
        if key in seen:
            continue
        seen.add(key)
        unique_names.append(raw_name)

    allowed, failed = _preflight_batch_delete_permissions(
        unique_names, user_id, is_admin
    )

    try:
        for name in allowed:
            try:
                result = _perform_kb_collection_delete(name, user_id, is_admin, db)
                if result.status in ("success", "partial_success"):
                    deleted.append(name)
                else:
                    failed.append(
                        BatchDeleteFailureItem(
                            name=name,
                            error=result.message or "Unknown error",
                        )
                    )
            except HTTPException as e:
                # SQL-only rollback for this request; no vector/file rollback.
                db.rollback()
                failed.append(
                    BatchDeleteFailureItem(
                        name=name,
                        error=_http_detail_to_str(e.detail),
                    )
                )
                logger.warning(
                    "Batch delete aborted after HTTP error for %s; rolled back pending SQL.",
                    name,
                )
                break
            except Exception as e:
                db.rollback()
                logger.exception("Batch delete failed for collection %s: %s", name, e)
                failed.append(BatchDeleteFailureItem(name=name, error=str(e)))
                break
    except Exception:
        db.rollback()
        raise

    return BatchDeleteCollectionsResponse(deleted=deleted, failed=failed)


@kb_router.post(
    "/collections/{collection_name}/documents/check",
)
@handle_kb_exceptions
async def check_documents_exist_api(
    collection_name: str,
    body: Dict[str, Any] = Body(
        ..., description="JSON body with 'filenames': list of filename strings"
    ),
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Check which of the given filenames already exist in the collection.

    Used by the frontend to show "file already exists, re-upload?" before ingest.
    New records resolve names via `file_id -> UploadedFile.filename`; legacy records
    fall back to `source_path` basename.

    For duplicate check we always filter by current user's documents only (including
    for admins), so "already exists" matches what will be overwritten on re-upload.
    """
    try:
        filenames = body.get("filenames")
        if not isinstance(filenames, list):
            raise HTTPException(
                status_code=422,
                detail="Request body must contain 'filenames' as a list of strings",
            )
        if not all(isinstance(f, str) for f in filenames):
            raise HTTPException(
                status_code=422,
                detail="All 'filenames' elements must be strings",
            )
        requested = {f.strip() for f in filenames if f and f.strip()}
        if not requested:
            return {"existing_filenames": []}

        try:
            safe_collection = sanitize_path_component(collection_name, "collection")
        except ValueError as e:
            raise HTTPException(
                status_code=422, detail=f"Invalid collection name: {str(e)}"
            ) from e

        await _ensure_collection_access(safe_collection, _user, allow_create=True)

        # Use storage abstraction layer to fetch document records
        vector_store = get_vector_index_store()
        records = vector_store.list_document_records(
            collection_name=safe_collection,
            user_id=int(_user.id),
            is_admin=False,
            max_results=DEFAULT_VECTOR_STORE_SCAN_LIMIT,
        )

        # Build filename map from file_ids (for UploadedFile lookup)
        # This preserves main branch's file_id -> filename resolution
        filename_map = _build_uploaded_filename_map(
            db,
            user_id=int(_user.id),
            file_ids=[
                file_id
                for file_id in (
                    _get_document_record_file_id(record) for record in records
                )
                if file_id
            ],
        )

        existing_filenames = set()
        for record in records:
            # Resolve filename using file_id first, then fallback to source_path basename
            resolved_filename = _resolve_document_filename(record, filename_map)
            if resolved_filename:
                existing_filenames.add(resolved_filename)

        return {"existing_filenames": sorted(requested & existing_filenames)}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to check documents exist: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check documents: {str(e)}",
        ) from e


@kb_router.delete(
    "/collections/{collection_name}/documents/{filename}",
)
@handle_kb_exceptions
async def delete_document_api(
    collection_name: str,
    filename: str,
    file_id: Optional[str] = Query(
        None, description="Preferred UploadedFile file_id for document lookup"
    ),
    doc_id: Optional[str] = Query(
        None, description="Preferred doc_id for document lookup"
    ),
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """Delete a document and all its associated data.

    Args:
        collection_name: Name of the collection
        filename: Legacy filename lookup key for backward compatibility

    Returns:
        Deletion result with status, list of deleted doc_ids, and filename

    Note:
        This endpoint prefers `file_id` or `doc_id` when provided. The path
        `filename` is retained as a compatibility fallback for older clients.
    """
    # NOTE: Exceptions are normalized by @handle_kb_exceptions for consistent API responses.
    from ...core.tools.core.RAG_tools.management.collections import delete_document

    # Parameter validation
    try:
        safe_collection_name = sanitize_path_component(collection_name, "collection")
    except ValueError as e:
        raise HTTPException(
            status_code=422, detail=f"Invalid collection name: {str(e)}"
        ) from e

    # Collection-level gate + vector fallback (rename / metadata lag vs strict visibility).
    await _ensure_collection_access_for_document_delete(safe_collection_name, _user)

    # Use storage abstraction layer to fetch document records
    vector_store = get_vector_index_store()
    records: List[DocumentRecord] = []
    try:
        records = vector_store.list_document_records(
            collection_name=safe_collection_name,
            user_id=int(_user.id),
            is_admin=bool(_user.is_admin),
            max_results=DEFAULT_VECTOR_STORE_SCAN_LIMIT,
        )
    except Exception as exc:
        # Degrade gracefully when vector store cannot read records.
        logger.warning(
            "Failed to read documents for delete resolution (collection=%s): %s",
            safe_collection_name,
            exc,
        )

    def _collect_candidate_doc_ids(
        docs: list[ResolvedDocumentMatch],
    ) -> list[str]:
        candidate: set[str] = set()
        for item in docs:
            raw = item.get("doc_id")
            if isinstance(raw, str) and raw:
                candidate.add(raw)
        return sorted(candidate)

    def _resolve_doc_id_for_uploaded_file(
        *,
        file_id_str: str,
        storage_path: str,
    ) -> str:
        """Resolve the stored doc_id for an owned UploadedFile.

        Prefer the exact documents-table row keyed by `file_id`, then fall back
        to the same deterministic key ingestion uses for modern uploads.
        """
        vector_store = get_vector_index_store()
        exact_matches: list[tuple[str, str]] = []

        try:
            for batch in vector_store.iter_batches(
                table_name="documents",
                columns=["doc_id", "source_path"],
                batch_size=10,
                filters={
                    "collection": safe_collection_name,
                    "file_id": file_id_str,
                },
                user_id=None,
                is_admin=True,
            ):
                rows = batch.to_pylist()
                for row in rows:
                    raw_doc_id = str(row.get("doc_id") or "").strip()
                    if not raw_doc_id:
                        continue
                    raw_source_path = str(row.get("source_path") or "").strip()
                    exact_matches.append((raw_doc_id, raw_source_path))
        except Exception as exc:
            logger.warning(
                "Failed to resolve doc_id by file_id for delete fallback "
                "(collection=%s, file_id=%s): %s",
                safe_collection_name,
                file_id_str,
                exc,
            )
        else:
            if len(exact_matches) == 1:
                return exact_matches[0][0]

            if len(exact_matches) > 1:
                for raw_doc_id, raw_source_path in exact_matches:
                    if raw_source_path == storage_path:
                        return raw_doc_id
                logger.warning(
                    "Multiple documents matched file_id fallback "
                    "(collection=%s, file_id=%s); using deterministic fallback",
                    safe_collection_name,
                    file_id_str,
                )

        if file_id_str:
            return generate_deterministic_doc_id(safe_collection_name, file_id_str)

        return generate_deterministic_doc_id(safe_collection_name, storage_path)

    def _append_matching_uploaded_file_candidate(rec: UploadedFile) -> bool:
        file_id_str = str(getattr(rec, "file_id", "")).strip()
        if not file_id_str:
            return False
        storage_path = str(getattr(rec, "storage_path", "")).strip()
        if not storage_path:
            return False
        derived_doc_id = _resolve_doc_id_for_uploaded_file(
            file_id_str=file_id_str,
            storage_path=storage_path,
        )
        if doc_id and derived_doc_id != doc_id:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Provided `file_id` and `doc_id` do not reference the same document"
                ),
            )
        matching_docs.append(
            {
                "doc_id": derived_doc_id,
                "file_id": file_id_str,
                "filename": str(getattr(rec, "filename", "")).strip() or filename,
                "source_path": storage_path or None,
            }
        )
        return True

    def _resolve_cleanup_file_id(doc_info: ResolvedDocumentMatch) -> Optional[str]:
        current_file_id = str(doc_info.get("file_id") or "").strip()
        if current_file_id:
            return current_file_id

        source_path = str(doc_info.get("source_path") or "").strip()
        if source_path:
            exact_match = (
                db.query(UploadedFile)
                .filter(
                    UploadedFile.user_id == user_id_int,
                    UploadedFile.storage_path == source_path,
                )
                .first()
            )
            if exact_match is not None:
                exact_file_id = str(getattr(exact_match, "file_id", "")).strip()
                if exact_file_id:
                    return exact_file_id

        normalized_filename = str(doc_info.get("filename") or "").strip()
        normalized_doc_id = str(doc_info.get("doc_id") or "").strip()
        user_segment = f"/user_{user_id_int}/{safe_collection_name}/"
        uploaded_query = db.query(UploadedFile).filter(
            UploadedFile.user_id == user_id_int,
            UploadedFile.storage_path.like(
                _like_contains_pattern(user_segment),
                escape=_SQL_LIKE_ESCAPE,
            ),
        )
        if normalized_filename:
            uploaded_query = uploaded_query.filter(
                UploadedFile.filename == normalized_filename
            )

        matched_file_ids: set[str] = set()
        for rec in uploaded_query.all():
            candidate_file_id = str(getattr(rec, "file_id", "")).strip()
            if not candidate_file_id:
                continue
            if normalized_doc_id:
                candidate_storage_path = str(getattr(rec, "storage_path", "")).strip()
                if not candidate_storage_path:
                    continue
                derived_doc_id = generate_deterministic_doc_id(
                    safe_collection_name,
                    candidate_storage_path,
                )
                if derived_doc_id != normalized_doc_id:
                    continue
            matched_file_ids.add(candidate_file_id)

        if len(matched_file_ids) == 1:
            return next(iter(matched_file_ids))
        if len(matched_file_ids) > 1:
            logger.warning(
                "Multiple UploadedFile candidates matched cleanup resolution "
                "(collection=%s, filename=%s, doc_id=%s)",
                safe_collection_name,
                normalized_filename,
                normalized_doc_id,
            )

        return None

    def _build_resolved_document_match(
        summary_doc_id: str,
        summary_basename: Optional[str],
        normalized_source_path: str,
        *,
        matched_file_id: Optional[str],
        matched_filename: Optional[str],
    ) -> ResolvedDocumentMatch:
        return {
            "doc_id": summary_doc_id,
            "file_id": matched_file_id,
            "filename": matched_filename or summary_basename or filename,
            "source_path": normalized_source_path or None,
        }

    def _match_uploaded_file_summary(
        uploaded_file_record: UploadedFile,
        summary_doc_id: str,
        summary_basename: Optional[str],
        normalized_source_path: str,
    ) -> Optional[ResolvedDocumentMatch]:
        uploaded_storage_path = str(
            getattr(uploaded_file_record, "storage_path", "")
        ).strip()
        uploaded_filename = str(getattr(uploaded_file_record, "filename", "")).strip()

        if normalized_source_path == uploaded_storage_path:
            return _build_resolved_document_match(
                summary_doc_id,
                summary_basename,
                normalized_source_path,
                matched_file_id=file_id,
                matched_filename=uploaded_filename,
            )

        if not uploaded_storage_path:
            return None

        derived_doc_id = generate_deterministic_doc_id(
            safe_collection_name,
            uploaded_storage_path,
        )
        if derived_doc_id != summary_doc_id:
            return None

        return _build_resolved_document_match(
            summary_doc_id,
            summary_basename,
            normalized_source_path,
            matched_file_id=file_id,
            matched_filename=uploaded_filename,
        )

    def _resolve_list_documents_match() -> Optional[ResolvedDocumentMatch]:
        uploaded_file_record: Optional[UploadedFile] = None
        if file_id:
            uploaded_file_record = (
                db.query(UploadedFile)
                .filter(
                    UploadedFile.user_id == user_id_int,
                    UploadedFile.file_id == file_id,
                )
                .first()
            )

        doc_list = list_documents(
            collection=safe_collection_name,
            user_id=user_id_int,
            is_admin=bool(_user.is_admin),
        )
        for summary in doc_list.documents:
            summary_doc_id = getattr(summary, "doc_id", None)
            if not isinstance(summary_doc_id, str) or not summary_doc_id:
                continue

            summary_source_path = getattr(summary, "source_path", None)
            normalized_source_path = (
                str(summary_source_path).strip()
                if isinstance(summary_source_path, str)
                else ""
            )
            summary_basename = (
                Path(normalized_source_path).name if normalized_source_path else None
            )

            if doc_id and summary_doc_id != doc_id:
                continue

            if not file_id:
                return _build_resolved_document_match(
                    summary_doc_id,
                    summary_basename,
                    normalized_source_path,
                    matched_file_id=None,
                    matched_filename=None,
                )

            if uploaded_file_record is None:
                if doc_id:
                    return _build_resolved_document_match(
                        summary_doc_id,
                        summary_basename,
                        normalized_source_path,
                        matched_file_id=None,
                        matched_filename=None,
                    )
                continue

            uploaded_match = _match_uploaded_file_summary(
                uploaded_file_record,
                summary_doc_id,
                summary_basename,
                normalized_source_path,
            )
            if uploaded_match is not None:
                return uploaded_match

            if doc_id:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Provided `file_id` and `doc_id` do not reference the same document"
                    ),
                )

        return None

    # Build filename map from file_ids (for UploadedFile lookup)
    user_id_int = int(_user.id)
    filename_map = _build_uploaded_filename_map(
        db,
        user_id=user_id_int,
        file_ids=[
            file_id
            for file_id in (_get_document_record_file_id(record) for record in records)
            if file_id
        ],
    )

    # Find all matching documents (handle duplicates)
    matching_docs: list[ResolvedDocumentMatch] = []
    for record in records:
        current_doc_id = record.doc_id
        current_file_id = _get_document_record_file_id(record)
        resolved_filename = _resolve_document_filename(record, filename_map)

        # Support filtering by doc_id, file_id, or filename (main branch feature)
        if doc_id and current_doc_id != doc_id:
            continue
        if file_id and current_file_id != file_id:
            continue
        if not doc_id and not file_id and resolved_filename != filename:
            continue

        matching_docs.append(
            {
                "doc_id": current_doc_id,
                "file_id": current_file_id,
                "filename": resolved_filename or filename,
                "source_path": record.source_path,
            }
        )

    # Safety: refuse to delete by basename if it is ambiguous.
    # This endpoint keeps `filename` in the path for backward compatibility, but
    # deleting multiple documents with the same filename is dangerous and hard
    # for users to reason about. Require an explicit `file_id` or `doc_id` when
    # more than one candidate matches.
    if not doc_id and not file_id and len(matching_docs) > 1:
        candidate_doc_ids = _collect_candidate_doc_ids(matching_docs)
        raise HTTPException(
            status_code=409,
            detail=(
                f"Ambiguous document deletion for filename '{filename}'. "
                "Multiple documents match; please retry with query param "
                "`file_id` or `doc_id`. "
                f"Candidates: {candidate_doc_ids}"
            ),
        )

    if not matching_docs and file_id:
        user_segment = f"/user_{user_id_int}/{safe_collection_name}/"
        uploaded_candidates = (
            db.query(UploadedFile)
            .filter(
                UploadedFile.user_id == user_id_int,
                UploadedFile.file_id == file_id,
                UploadedFile.storage_path.like(
                    _like_contains_pattern(user_segment),
                    escape=_SQL_LIKE_ESCAPE,
                ),
            )
            .all()
        )
        for rec in uploaded_candidates:
            _append_matching_uploaded_file_candidate(rec)

    if not matching_docs and (doc_id or file_id):
        # Explicit identifiers: validate through other data sources before allowing deletion
        # to prevent accidental deletion of non-existent or wrong documents.
        try:
            resolved_match = _resolve_list_documents_match()
            if resolved_match is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Document not found in collection '{safe_collection_name}'",
                )

            matching_docs.append(resolved_match)
        except HTTPException:
            raise
        except Exception as exc:
            # If validation fails, err on the side of caution and refuse deletion
            logger.warning(
                "Failed to validate document existence for deletion (collection=%s): %s",
                safe_collection_name,
                exc,
            )
            raise HTTPException(
                status_code=503,
                detail="Unable to verify document existence. Deletion refused to prevent data loss.",
            )

    if not matching_docs:
        # Fallback 1: derive doc_id from UploadedFile linkage for uploaded docs.
        user_segment = f"/user_{user_id_int}/{safe_collection_name}/"
        uploaded_query = db.query(UploadedFile).filter(
            UploadedFile.user_id == user_id_int,
            UploadedFile.storage_path.like(
                _like_contains_pattern(user_segment),
                escape=_SQL_LIKE_ESCAPE,
            ),
        )
        if file_id:
            uploaded_query = uploaded_query.filter(UploadedFile.file_id == file_id)
        else:
            uploaded_query = uploaded_query.filter(UploadedFile.filename == filename)
        uploaded_candidates = uploaded_query.all()
        for rec in uploaded_candidates:
            _append_matching_uploaded_file_candidate(rec)

    if not doc_id and not file_id and len(matching_docs) > 1:
        candidate_doc_ids = _collect_candidate_doc_ids(matching_docs)
        raise HTTPException(
            status_code=409,
            detail=(
                f"Ambiguous document deletion for filename '{filename}'. "
                "Multiple documents match; please retry with query param "
                "`file_id` or `doc_id`. "
                f"Candidates: {candidate_doc_ids}"
            ),
        )

    if not matching_docs and not file_id and not doc_id:
        # Fallback 2: allow web-ingested docs to be deleted by doc_id-like filename.
        try:
            doc_list = list_documents(
                collection=safe_collection_name,
                user_id=user_id_int,
                is_admin=bool(_user.is_admin),
            )
            for summary in doc_list.documents:
                doc_id_value = getattr(summary, "doc_id", None)
                resolved_doc_id = (
                    str(doc_id_value).strip()
                    if isinstance(doc_id_value, str) and doc_id_value.strip()
                    else ""
                )
                source_path = getattr(summary, "source_path", None)
                fallback_basename: str | None = None
                if isinstance(source_path, str) and source_path.strip():
                    fallback_basename = Path(source_path).name
                if resolved_doc_id and (
                    resolved_doc_id == filename or fallback_basename == filename
                ):
                    matching_docs.append(
                        {
                            "doc_id": resolved_doc_id,
                            "file_id": None,
                            "filename": filename,
                            "source_path": source_path
                            if isinstance(source_path, str)
                            else None,
                        }
                    )
        except Exception as exc:
            logger.warning(
                "Fallback doc resolution via list_documents failed (collection=%s): %s",
                safe_collection_name,
                exc,
            )

    if not doc_id and not file_id and len(matching_docs) > 1:
        candidate_doc_ids = _collect_candidate_doc_ids(matching_docs)
        raise HTTPException(
            status_code=409,
            detail=(
                f"Ambiguous document deletion for filename '{filename}'. "
                "Multiple documents match; please retry with query param "
                "`file_id` or `doc_id`. "
                f"Candidates: {candidate_doc_ids}"
            ),
        )

    if not matching_docs:
        raise HTTPException(
            status_code=404,
            detail=f"Document not found: {filename}",
        )

    deleted_doc_ids = []
    deletion_errors = []
    cleanup_candidate_file_ids: set[str] = set()

    for doc_info in matching_docs:
        resolved_doc_id = doc_info["doc_id"]
        if not isinstance(resolved_doc_id, str) or not resolved_doc_id:
            error_msg = "Failed to delete document: resolved doc_id is missing"
            deletion_errors.append(error_msg)
            logger.error("%s", error_msg)
            continue
        try:
            delete_result = delete_document(
                safe_collection_name,
                resolved_doc_id,
                int(_user.id),
                bool(_user.is_admin),
            )
            delete_status = getattr(delete_result, "status", None)
            if delete_status != "success":
                error_msg = getattr(
                    delete_result,
                    "message",
                    f"Failed to delete doc_id {resolved_doc_id}",
                )
                deletion_errors.append(str(error_msg))
                logger.error(
                    "Delete operation returned non-success status for doc_id %s: %s",
                    resolved_doc_id,
                    error_msg,
                )
                continue

            deleted_doc_ids.append(resolved_doc_id)
            current_file_id = _resolve_cleanup_file_id(doc_info)
            if current_file_id:
                cleanup_candidate_file_ids.add(current_file_id)
            logger.info(
                "Deleted document '%s' (doc_id: %s) from collection '%s'",
                doc_info.get("filename", filename),
                resolved_doc_id,
                safe_collection_name,
            )
        except Exception as e:
            error_msg = f"Failed to delete doc_id {resolved_doc_id}: {str(e)}"
            deletion_errors.append(error_msg)
            logger.error("%s", error_msg)

    if cleanup_candidate_file_ids:
        try:
            remaining_records = _list_documents_for_user(
                user_id=user_id_int,
                is_admin=bool(_user.is_admin),
            )
            remaining_file_ids = {
                current_file_id
                for current_file_id in (
                    _get_document_record_file_id(record) for record in remaining_records
                )
                if current_file_id
            }
        except Exception as exc:
            logger.warning(
                "Failed to refresh remaining docs for orphan cleanup; skipping orphan cleanup for %s file(s): %s",
                len(cleanup_candidate_file_ids),
                exc,
            )
        else:
            for cleanup_file_id in cleanup_candidate_file_ids:
                _delete_uploaded_file_if_orphaned(
                    db,
                    file_id=cleanup_file_id,
                    user_id=user_id_int,
                    remaining_file_ids=remaining_file_ids,
                )

    # Commit all orphan file cleanups in a single batch after the loop
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        deletion_errors.append(f"Failed to persist orphan cleanup changes: {str(exc)}")
        logger.error(
            "Failed to commit orphan cleanup changes for collection %s: %s",
            safe_collection_name,
            exc,
        )

    if deletion_errors:
        return {
            "status": "partial_success" if deleted_doc_ids else "failed",
            "message": f"Deleted {len(deleted_doc_ids)} of {len(matching_docs)} documents",
            "collection": safe_collection_name,
            "filename": filename,
            "deleted_doc_ids": deleted_doc_ids,
            "errors": deletion_errors,
        }

    return {
        "status": "success",
        "message": f"Successfully deleted {len(deleted_doc_ids)} document(s)",
        "collection": safe_collection_name,
        "filename": filename,
        "deleted_doc_ids": deleted_doc_ids,
    }


@kb_router.put(
    "/collections/{collection_name}",
)
@handle_kb_exceptions
async def rename_collection_api(
    collection_name: str,
    new_name: str = Form(..., description="New collection name"),
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """Rename a collection.

    Args:
        collection_name: Current collection name
        new_name: New collection name

    Returns:
        Success message
    """
    from ...core.tools.core.RAG_tools.management.status import (
        clear_ingestion_status,
        load_ingestion_status,
        write_ingestion_status,
    )
    from ...core.tools.core.RAG_tools.storage.factory import (
        get_metadata_store,
        get_vector_index_store,
    )

    vector_store = get_vector_index_store()

    if not new_name or not new_name.strip():
        raise HTTPException(
            status_code=422,
            detail="New collection name cannot be empty",
        )

    warnings: list[str] = []

    # SECURITY: Validate both old and new collection names to prevent path traversal
    try:
        safe_old_collection = sanitize_path_component(collection_name, "collection")
        safe_new_collection = sanitize_path_component(new_name, "collection")
    except ValueError as e:
        raise HTTPException(
            status_code=422, detail=f"Invalid collection name: {str(e)}"
        ) from e

    # Quick return if name unchanged
    if safe_new_collection == safe_old_collection:
        return {"status": "success", "message": "Collection name unchanged"}

    # Access control check
    await _ensure_collection_access(safe_old_collection, _user, hide_missing=False)

    # Validate that target collection doesn't exist or user has access
    visible_for_user = await _list_collections_with_retry(
        user_id=int(_user.id),
        is_admin=False,
        stage="rename_list_visible_collections",
    )
    if any(c.name == safe_new_collection for c in visible_for_user.collections):
        raise HTTPException(
            status_code=409,
            detail=f"Target collection already exists: {safe_new_collection}",
        )
    if not any(c.name == safe_new_collection for c in visible_for_user.collections):
        all_named = await _list_collections_with_retry(
            user_id=None,
            is_admin=True,
            stage="rename_list_all_collections",
        )
        if any(c.name == safe_new_collection for c in all_named.collections):
            raise HTTPException(
                status_code=403,
                detail=f"Access denied for collection: {safe_new_collection}",
            )

    physical_rename_status = "not_found"
    physical_rename_error: Optional[str] = None
    old_collection_dir: Optional[Path] = None
    new_collection_dir: Optional[Path] = None
    collection_records = vector_store.list_document_records(
        collection_name=safe_old_collection,
        user_id=int(_user.id),
        is_admin=bool(_user.is_admin),
    )
    collection_file_ids = {
        file_id
        for file_id in (
            _get_document_record_file_id(record) for record in collection_records
        )
        if file_id
    }

    physical_rename = rename_collection_storage(
        db,
        user_id=int(_user.id),
        old_collection_name=safe_old_collection,
        new_collection_name=safe_new_collection,
        collection_file_ids=collection_file_ids,
    )
    physical_rename_status = physical_rename.status
    physical_rename_error = physical_rename.error
    old_collection_dir = physical_rename.old_collection_dir
    new_collection_dir = physical_rename.new_collection_dir
    if physical_rename_status == "failed":
        if (
            physical_rename_error
            == "Another operation is in progress; please try again later."
        ):
            raise HTTPException(status_code=409, detail=physical_rename_error)
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to rename collection: cannot rename physical directory. "
                f"Error: {physical_rename_error}. "
                "Please ensure the directory is not in use and you have proper permissions."
            ),
        )

    # Step 2: Update collection name in all tables (documents, parses, chunks, embeddings)
    # Use storage abstraction layer which handles all tables including embeddings
    vector_store = get_vector_index_store()
    warnings.extend(
        vector_store.rename_collection_data(
            collection_name=safe_old_collection,
            new_name=safe_new_collection,
        )
    )

    try:
        metadata_store = get_metadata_store()
        await metadata_store.rename_collection(
            old_name=safe_old_collection,
            new_name=safe_new_collection,
        )
    except Exception as e:
        logger.warning("Failed to rename metadata store keys: %s", e)
        warnings.append(f"Failed to rename collection metadata: {e}")

    # Migrate ingestion status from old collection name to new
    try:
        status_entries = load_ingestion_status(collection=safe_old_collection)
        for entry in status_entries:
            doc_id = entry.get("doc_id")
            if doc_id:
                write_ingestion_status(
                    safe_new_collection,
                    doc_id,
                    status=entry.get("status", "pending"),
                    message=entry.get("message", ""),
                    parse_hash=entry.get("parse_hash", ""),
                )
                clear_ingestion_status(safe_old_collection, doc_id)
    except Exception as e:
        logger.warning("Failed to update ingestion status: %s", e)
        warnings.append(f"Failed to update ingestion status: {e}")

    # Step 3: Add physical rename status to warnings and message for visibility
    rename_info_message = ""
    if (
        physical_rename_status == "success"
        and old_collection_dir is not None
        and new_collection_dir is not None
    ):
        rename_info = f"Physical directory renamed: {old_collection_dir.name} -> {new_collection_dir.name}"
        warnings.append(rename_info)
        rename_info_message = f" {rename_info}."
    elif physical_rename_status == "not_found":
        rename_info = "Physical directory rename: No physical directory found (collection had no files)"
        warnings.append(rename_info)
        rename_info_message = f" {rename_info}."
    elif physical_rename_status == "error" and physical_rename_error:
        rename_info = (
            f"Physical directory rename: Warning - {physical_rename_error}. "
            "Database rename proceeded, but physical directory rename status is uncertain."
        )
        warnings.append(rename_info)
        rename_info_message = f" {rename_info}"
    elif physical_rename_status == "failed" and physical_rename_error:
        rename_info = f"Physical directory rename: Failed - {physical_rename_error}"
        warnings.append(rename_info)
        rename_info_message = f" {rename_info}"

    # Step 4: Determine final status
    final_status = "success" if not warnings else "partial_success"
    if physical_rename_status in ("error", "failed"):
        final_status = "partial_success"
        if not rename_info_message:
            rename_info_message = " Database rename succeeded, but physical directory rename encountered issues."

    # Step 5: Build final message
    base_message = (
        f"Collection renamed from '{safe_old_collection}' to '{safe_new_collection}'"
    )
    if warnings and len(warnings) > (1 if physical_rename_status != "not_found" else 0):
        final_message = f"{base_message} with some warnings"
    else:
        final_message = base_message
    if rename_info_message:
        final_message = f"{final_message}{rename_info_message}"

    if warnings:
        return {
            "status": final_status,
            "message": final_message,
            "warnings": warnings,
        }

    return {
        "status": "success",
        "message": base_message,
    }


@kb_router.get(
    "/collections/{collection_name}/parses/{doc_id}/parse_result",
    response_model=ParseResultResponse,
)
@handle_kb_exceptions
async def get_parse_result_api(
    collection_name: str,
    doc_id: str,
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(20, ge=1, le=100, description="Number of elements per page"),
    parse_hash: Optional[str] = Query(
        None,
        description="Optional parse hash to filter. If None, uses the latest parse.",
    ),
    _user: User = Depends(get_current_user),
) -> ParseResultResponse:
    """Get parsed document results with pagination.

    Args:
        collection_name: Collection name
        doc_id: Document ID
        page: Page number (1-indexed, default: 1)
        page_size: Number of elements per page (default: 20)
        parse_hash: Optional parse hash to filter. If None, uses the latest parse.

    Returns:
        ParseResultResponse with paginated text segments, tables, and figures
    """
    from ...core.tools.core.RAG_tools.core.exceptions import DocumentNotFoundError
    from ...core.tools.core.RAG_tools.utils.string_utils import sanitize_for_doc_id

    safe_doc_id = sanitize_for_doc_id(doc_id)
    if safe_doc_id != doc_id:
        logger.warning("Invalid doc_id format detected: %s", doc_id)
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    if page < 1:
        raise HTTPException(status_code=422, detail="Page number must be >= 1")
    if page_size < 1 or page_size > 100:
        raise HTTPException(
            status_code=422, detail="Page size must be between 1 and 100"
        )

    try:
        safe_collection = sanitize_path_component(collection_name, "collection")
    except ValueError as e:
        raise HTTPException(
            status_code=422, detail=f"Invalid collection name: {str(e)}"
        ) from e

    await _ensure_collection_access(safe_collection, _user, hide_missing=False)

    try:
        elements, actual_parse_hash = reconstruct_parse_result_from_db(
            safe_collection,
            doc_id,
            parse_hash,
            user_id=int(_user.id),
            is_admin=bool(_user.is_admin),
        )
    except DocumentNotFoundError as e:
        logger.warning("Parse result not found: %s", e)
        raise HTTPException(status_code=404, detail=str(e))

    paginated_elements, pagination_info = paginate_parse_results(
        elements, page, page_size
    )

    return ParseResultResponse(
        doc_id=doc_id,
        parse_hash=actual_parse_hash or "",
        elements=paginated_elements,
        pagination=pagination_info,
    )
