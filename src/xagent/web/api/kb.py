"""Knowledge base API route handlers"""

import asyncio
import functools
import hashlib
import inspect
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
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    TypedDict,
    TypeVar,
    cast,
)

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
from pydantic import BaseModel, Field, ValidationError
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
    delete_collection_metadata_sync,
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
from ...core.tools.core.RAG_tools.storage.factory import (
    get_ingestion_status_store,
    get_metadata_store,
    get_vector_index_store,
)
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
from ..models.background_job import (
    BackgroundJob,
    BackgroundJobStatus,
    BackgroundJobType,
)
from ..models.database import get_db, get_session_local
from ..models.uploaded_file import UploadedFile
from ..models.user import User
from ..schemas.background_job import BackgroundJobResponse
from ..services.background_jobs import (
    QUEUE_DEFAULT,
    create_background_job,
    get_non_terminal_background_job_by_idempotency_key,
    is_background_job_enqueue_available,
    mark_job_failed,
)
from ..services.kb_collection_service import (
    delete_collection_physical_dir,
    delete_collection_uploaded_files,
    list_collection_uploaded_file_owner_ids,
    rename_collection_storage,
)
from ..services.kb_file_service import (
    build_uploaded_filename_map as _build_uploaded_filename_map,
)
from ..services.kb_file_service import (
    capture_uploaded_file_refresh_snapshot as _capture_uploaded_file_refresh_snapshot,
)
from ..services.kb_file_service import (
    compensate_new_uploaded_file as _compensate_new_uploaded_file,
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
    restore_uploaded_file_refresh_snapshot as _restore_uploaded_file_refresh_snapshot,
)
from ..services.kb_file_service import (
    upsert_uploaded_file_record as _upsert_uploaded_file_record,
)
from ..services.kb_ingest_targets import (
    admit_kb_ingest_target,
    release_kb_ingest_target_generation,
    tombstone_kb_ingest_target,
    tombstone_kb_ingest_targets_for_collection,
)
from ..services.managed_file_ref import (
    DurableObjectMissingError,
    ManagedFileRef,
    build_upload_storage_key,
)
from ..services.uploaded_file_store import UploadedFileStore
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
_BACKGROUND_INGEST_STAGING_DIR = ".background-ingest"


@dataclass(frozen=True)
class UploadCopyResult:
    total_size: int
    sha256: str


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


def _truncate_utf8_bytes(value: str, max_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _build_ingest_backup_path(file_path: Path) -> Path:
    suffix = f".rollback-{uuid.uuid4().hex}"
    max_name_bytes = _MAX_FILESYSTEM_FILENAME_BYTES - len(suffix.encode("utf-8"))
    truncated_name = _truncate_utf8_bytes(file_path.name, max_name_bytes).rstrip(" ._-")
    if not truncated_name:
        truncated_name = hashlib.sha256(file_path.name.encode("utf-8")).hexdigest()[:16]
    return file_path.with_name(f"{truncated_name}{suffix}")


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


def _delete_web_rag_side_effects_for_file_id(
    *,
    collection_name: str,
    file_id: str,
    user_id: int,
    is_admin: bool,
) -> None:
    """Best-effort RAG cleanup when document ingestion raised before returning."""
    from ...core.tools.core.RAG_tools.utils.string_utils import (
        generate_deterministic_doc_id,
    )

    doc_refs = [
        (collection, doc_id)
        for collection, doc_id in _list_document_refs_for_uploaded_file(file_id)
        if collection == collection_name
    ]

    if doc_refs:
        cleanup_errors: list[Exception] = []
        for collection, doc_id in doc_refs:
            try:
                document_delete_result = delete_document(
                    collection,
                    doc_id,
                    user_id,
                    is_admin,
                )
                _ensure_cleanup_succeeded(
                    f"delete document '{doc_id}' during web exception rollback",
                    document_delete_result,
                )
            except Exception as exc:  # noqa: BLE001
                cleanup_errors.append(exc)
                logger.warning(
                    "Failed to delete web RAG side effect during exception rollback: "
                    "collection=%s, doc_id=%s, file_id=%s, error=%s",
                    collection,
                    doc_id,
                    file_id,
                    exc,
                    exc_info=True,
                )
        if cleanup_errors:
            raise RuntimeError(
                "Best-effort RAG cleanup failed with "
                f"{len(cleanup_errors)} errors. First error: {cleanup_errors[0]}"
            ) from cleanup_errors[0]
        return

    doc_id = generate_deterministic_doc_id(collection_name, file_id)
    vector_store = get_vector_index_store()
    vector_store.delete_document_data(
        collection_name=collection_name,
        doc_id=doc_id,
        user_id=user_id,
        is_admin=is_admin,
    )
    clear_ingestion_status(
        collection_name,
        doc_id,
        user_id=user_id,
        is_admin=is_admin,
    )


def _rollback_failed_web_document_ingestion(
    *,
    collection_name: str,
    result: Optional[IngestionResult],
    user_id: int,
    is_admin: bool,
    rag_snapshot: Optional["_RagDocumentSnapshot"] = None,
    file_id: Optional[str] = None,
) -> None:
    """Remove RAG-side writes from a failed per-page web ingestion."""
    if result is None:
        if rag_snapshot is not None:
            _restore_rag_document_snapshot(
                rag_snapshot,
                user_id=user_id,
                is_admin=is_admin,
            )
        if file_id and (rag_snapshot is None or not rag_snapshot.doc_refs):
            _delete_web_rag_side_effects_for_file_id(
                collection_name=collection_name,
                file_id=file_id,
                user_id=user_id,
                is_admin=is_admin,
            )
        return

    register_metadata = _get_completed_step_metadata(result, "register_document") or {}
    register_created = bool(register_metadata.get("created"))
    doc_id = result.doc_id if isinstance(result.doc_id, str) and result.doc_id else None
    if not doc_id:
        if rag_snapshot is not None:
            _restore_rag_document_snapshot(
                rag_snapshot,
                user_id=user_id,
                is_admin=is_admin,
            )
        elif file_id:
            _delete_web_rag_side_effects_for_file_id(
                collection_name=collection_name,
                file_id=file_id,
                user_id=user_id,
                is_admin=is_admin,
            )
        return

    if register_created:
        document_delete_result = delete_document(
            collection_name,
            doc_id,
            user_id,
            is_admin,
        )
        _ensure_cleanup_succeeded(
            f"delete document '{doc_id}' during web rollback",
            document_delete_result,
        )
        if rag_snapshot is None:
            return

    if rag_snapshot is not None:
        _restore_rag_document_snapshot(
            rag_snapshot,
            user_id=user_id,
            is_admin=is_admin,
        )
        return

    clear_ingestion_status(
        collection_name,
        doc_id,
        user_id=user_id,
        is_admin=is_admin,
    )


def _extract_embedding_model_id_from_error(message: str) -> Optional[str]:
    match = re.search(r"Model '([^']+)' not found in hub", message)
    if match:
        return match.group(1).strip() or None
    return None


def _is_embedding_configuration_error(message: str) -> bool:
    normalized = message.lower()
    return (
        "no embedding model available" in normalized
        or (
            "not found in hub" in normalized
            and "environment configuration available for embedding" in normalized
        )
        or ("no environment configuration" in normalized and "embedding" in normalized)
    )


def _build_user_actionable_ingestion_message(
    message: str,
    *,
    embedding_model_id: Optional[str] = None,
) -> str:
    normalized = str(message).strip() or "Unknown ingestion failure"
    if "How to fix:" in normalized:
        return normalized
    if not _is_embedding_configuration_error(normalized):
        return normalized

    resolved_model_id = embedding_model_id or _extract_embedding_model_id_from_error(
        normalized
    )
    current_model_hint = (
        f" Current embedding_model_id: '{resolved_model_id}'."
        if resolved_model_id
        else ""
    )
    return (
        f"{normalized} Cause: knowledge-base ingestion requires a resolvable "
        f"embedding model, but the current model configuration could not be loaded."
        f"{current_model_hint} How to fix: configure a visible default embedding "
        "model in the model settings, or pass a valid embedding_model_id in the "
        "ingest request. If you rely on environment variables, set "
        "DASHSCOPE_EMBEDDING_MODEL and DASHSCOPE_EMBEDDING_API_KEY "
        "(or DASHSCOPE_API_KEY)."
    )


def _with_user_actionable_ingestion_message(
    result: IngestionResult,
    *,
    embedding_model_id: Optional[str] = None,
) -> IngestionResult:
    updated_message = _build_user_actionable_ingestion_message(
        result.message,
        embedding_model_id=embedding_model_id,
    )
    if updated_message == result.message:
        return result
    return result.model_copy(update={"message": updated_message})


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


def _collection_or_config_existed_before(
    collection_existed_before: bool,
    snapshot: Optional["_CollectionConfigSnapshot"],
) -> bool:
    """Treat config-only collections as pre-existing for rollback decisions."""
    return collection_existed_before or (
        snapshot is not None and snapshot.previous_config_json is not None
    )


async def _restore_or_cleanup_collection_config_after_failed_ingest(
    *,
    snapshot: Optional["_CollectionConfigSnapshot"],
    collection_existed_before: bool,
    collection_name: str,
    user: User,
    context: str,
    successful_documents: int = 0,
    side_effects_may_remain: bool = False,
) -> None:
    """Restore previous config or clean up only truly empty new collections."""
    if (
        snapshot is not None
        and not snapshot.saved
        and not snapshot.previous_config_known
    ):
        logger.warning(
            "Skipping failed-ingest collection metadata cleanup because previous "
            "config state is unknown: %s/user_%s",
            collection_name,
            int(user.id),
        )
        return

    effective_collection_existed_before = _collection_or_config_existed_before(
        collection_existed_before,
        snapshot,
    )
    config_only_existed_before = (
        not collection_existed_before
        and snapshot is not None
        and snapshot.previous_config_json is not None
    )
    if effective_collection_existed_before:
        await _restore_collection_config_after_failed_ingest(
            snapshot=snapshot,
            collection_existed_before=effective_collection_existed_before,
            context=context,
        )
        if (
            config_only_existed_before
            and successful_documents == 0
            and not side_effects_may_remain
            and snapshot is not None
        ):
            delete_metadata = getattr(
                snapshot.metadata_store, "delete_collection", None
            )
            if callable(delete_metadata):
                await _maybe_await(delete_metadata(collection_name))
                logger.info(
                    "Removed collection metadata created by failed %s while "
                    "preserving previous config: %s/user_%s",
                    context,
                    collection_name,
                    int(user.id),
                )
        return

    if successful_documents > 0 or side_effects_may_remain:
        if side_effects_may_remain:
            logger.warning(
                "Skipping failed-ingest collection metadata cleanup for %s/user_%s "
                "because rollback side effects may remain",
                collection_name,
                int(user.id),
            )
        return

    await _cleanup_failed_new_collection_metadata(
        collection_name=collection_name,
        user=user,
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
    embedding_model_id: Optional[str] = None,
) -> None:
    user_id = int(user.id)
    file_record_id = str(file_record.file_id)
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
                # The collection cleanup above may already delete+commit the UploadedFile
                # row, so reuse the stable file_id instead of touching a deleted ORM instance.
                refreshed_file_record = (
                    db.query(UploadedFile)
                    .filter(UploadedFile.file_id == file_record_id)
                    .first()
                )
                if refreshed_file_record is not None:
                    UploadedFileStore(db).delete(
                        refreshed_file_record, delete_local=False
                    )
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
                file_id=file_record_id,
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
                UploadedFileStore(db).delete(file_record, delete_local=False)
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
        original_error_message = _build_user_actionable_ingestion_message(
            result.message,
            embedding_model_id=embedding_model_id,
        )
        if original_error_message:
            message = f"{message}. Original ingestion error: {original_error_message}"
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
    embedding_model_id: Optional[str] = None,
) -> None:
    user_id = int(user.id)
    file_record_id = str(file_record.file_id) if file_record is not None else None
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

        if file_record_id is not None:
            _delete_uploaded_file_if_orphaned(
                db,
                file_id=file_record_id,
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
        original_error_message = _build_user_actionable_ingestion_message(
            result.message,
            embedding_model_id=embedding_model_id,
        )
        if original_error_message:
            message = f"{message}. Original ingestion error: {original_error_message}"
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


def _background_job_idempotency_key(namespace: str, payload: Dict[str, Any]) -> str:
    """Build a bounded idempotency key from stable request payload fields."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"{namespace}:{digest}"


def _copy_upload_file_to_path(
    file: UploadFile,
    file_path: Path,
    *,
    max_size: int | None = None,
) -> UploadCopyResult:
    effective_max_size = MAX_FILE_SIZE if max_size is None else max_size
    total_size = 0
    hash_obj = hashlib.sha256()
    file_read_buffer_size = 1024 * 1024
    file.file.seek(0)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as buffer:
        while True:
            chunk = file.file.read(file_read_buffer_size)
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > effective_max_size:
                raise HTTPException(
                    status_code=413,
                    detail=f"File size exceeds maximum limit of {MAX_FILE_SIZE_LABEL}",
                )
            hash_obj.update(chunk)
            buffer.write(chunk)
    return UploadCopyResult(total_size=total_size, sha256=hash_obj.hexdigest())


def _build_background_ingest_staging_path(*, user_id: int, filename: str) -> Path:
    user_root = get_upload_path(
        _BACKGROUND_INGEST_STAGING_DIR,
        user_id=user_id,
        create_if_not_exists=True,
    ).parent
    staging_dir = user_root / _BACKGROUND_INGEST_STAGING_DIR / uuid.uuid4().hex
    staging_dir.mkdir(parents=True, exist_ok=False)
    return staging_dir / Path(filename).name


def _background_ingest_file_id(*, user_id: int, storage_path: Path) -> str:
    stable_key = f"xagent-kb-ingest:{user_id}:{storage_path}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key))


def _cleanup_background_ingest_staging_file(staging_path: Path | str | None) -> None:
    if not staging_path:
        return
    path = Path(str(staging_path))
    try:
        if path.exists():
            path.unlink()
    except OSError:
        logger.warning("Failed to remove background ingest staging file %s", path)
        return
    try:
        path.parent.rmdir()
    except OSError:
        pass


async def _ensure_background_job_queue_available_async() -> None:
    if not await asyncio.to_thread(
        is_background_job_enqueue_available,
        check_worker=True,
    ):
        raise HTTPException(
            status_code=503,
            detail="Background job queue is unavailable",
        )


async def _enqueue_background_job_or_503_async(
    db: Session,
    job: BackgroundJob,
) -> BackgroundJob:
    if not await asyncio.to_thread(
        is_background_job_enqueue_available,
        check_worker=True,
    ):
        mark_job_failed(
            db,
            job,
            error_message="Background job queue is unavailable",
        )
        raise HTTPException(
            status_code=503,
            detail="Background job queue is unavailable",
        )

    try:
        from ..jobs.tasks import execute_background_job

        setattr(job, "status", BackgroundJobStatus.ENQUEUED.value)
        db.add(job)
        db.commit()
        db.refresh(job)

        async_result = await asyncio.to_thread(
            execute_background_job.apply_async,
            args=[job.id],
            queue=str(job.queue or QUEUE_DEFAULT),
        )
        db.refresh(job)
        setattr(job, "celery_task_id", async_result.id)
        db.add(job)
        db.commit()
        db.refresh(job)
        return job
    except Exception as exc:  # noqa: BLE001
        mark_job_failed(
            db,
            job,
            error_message=f"Background job queue is unavailable: {exc}",
        )
        raise HTTPException(
            status_code=503,
            detail=f"Background job queue is unavailable: {exc}",
        ) from exc


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


_INGESTION_RUN_COLUMNS = (
    "collection",
    "doc_id",
    "status",
    "message",
    "parse_hash",
    "created_at",
    "updated_at",
    "user_id",
)


@dataclass
class _IngestionRunsSnapshot:
    doc_refs: List[tuple[str, str]]
    rows: List[Dict[str, Any]]


@dataclass
class _RagDocumentSnapshot:
    doc_refs: List[tuple[str, str]]
    rows_by_table: Dict[str, List[Dict[str, Any]]]


def _ingestion_run_filter(collection: str, doc_id: str) -> str:
    from ...core.tools.core.RAG_tools.utils.string_utils import escape_lancedb_string

    safe_collection = escape_lancedb_string(collection)
    safe_doc_id = escape_lancedb_string(doc_id)
    return f"collection = '{safe_collection}' and doc_id = '{safe_doc_id}'"


def _table_has_user_id_column(table: Any) -> bool:
    schema = getattr(table, "schema", None)
    names = getattr(schema, "names", None) or []
    return "user_id" in names


def _rag_document_filter(
    table: Any,
    *,
    collection: str,
    doc_id: str,
    user_id: int,
    is_admin: bool,
) -> str:
    from ...core.tools.core.RAG_tools.utils.string_utils import escape_lancedb_string

    safe_collection = escape_lancedb_string(collection)
    safe_doc_id = escape_lancedb_string(doc_id)
    expr = f"collection = '{safe_collection}' and doc_id = '{safe_doc_id}'"
    if not is_admin and _table_has_user_id_column(table):
        expr = f"{expr} and user_id = {int(user_id)}"
    return expr


def _combine_lancedb_filters(filters: List[str]) -> str:
    return " or ".join(f"({filter_expr})" for filter_expr in filters)


def _rag_document_refs_filter(
    table: Any,
    doc_refs: List[tuple[str, str]],
    *,
    user_id: int,
    is_admin: bool,
) -> str:
    return _combine_lancedb_filters(
        [
            _rag_document_filter(
                table,
                collection=collection,
                doc_id=doc_id,
                user_id=user_id,
                is_admin=is_admin,
            )
            for collection, doc_id in doc_refs
        ]
    )


def _snapshot_rag_documents_for_uploaded_file(
    file_id: str,
    *,
    user_id: int,
    is_admin: bool,
) -> Optional[_RagDocumentSnapshot]:
    """Snapshot RAG rows for documents associated with an UploadedFile."""
    try:
        from ...core.tools.core.RAG_tools.LanceDB.schema_manager import (
            _safe_close_table,
            ensure_chunks_table,
            ensure_documents_table,
            ensure_ingestion_runs_table,
            ensure_main_pointers_table,
            ensure_parses_table,
        )
        from ...core.tools.core.RAG_tools.utils.lancedb_query_utils import (
            list_table_names,
            query_to_list,
        )
        from ...providers.vector_store.lancedb import get_connection_from_env

        doc_refs = _list_document_refs_for_uploaded_file(file_id)
        conn = get_connection_from_env()
        ensure_documents_table(conn)
        ensure_parses_table(conn)
        ensure_chunks_table(conn)
        ensure_main_pointers_table(conn)
        ensure_ingestion_runs_table(conn)

        table_names = set(list_table_names(conn))
        target_tables = [
            table_name
            for table_name in (
                "documents",
                "parses",
                "chunks",
                "main_pointers",
                "ingestion_runs",
            )
            if table_name in table_names
        ]
        target_tables.extend(
            sorted(name for name in table_names if name.startswith("embeddings_"))
        )

        rows_by_table: Dict[str, List[Dict[str, Any]]] = {}
        for table_name in target_tables:
            table = None
            try:
                table = conn.open_table(table_name)
                if doc_refs:
                    rows = query_to_list(
                        table.search()
                        .where(
                            _rag_document_refs_filter(
                                table,
                                doc_refs,
                                user_id=user_id,
                                is_admin=is_admin,
                            )
                        )
                        .limit(-1)
                    )
                else:
                    rows = []
                rows_by_table[table_name] = rows
            finally:
                _safe_close_table(table)
        return _RagDocumentSnapshot(doc_refs=doc_refs, rows_by_table=rows_by_table)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to snapshot RAG document rows before web file refresh: "
            "file_id=%s, error=%s",
            file_id,
            exc,
            exc_info=True,
        )
        return None


def _rag_snapshot_key_columns(table_name: str) -> Optional[tuple[str, ...]]:
    if table_name == "documents":
        return ("collection", "doc_id")
    if table_name == "parses":
        return ("collection", "doc_id", "parse_hash")
    if table_name == "chunks":
        return ("collection", "doc_id", "parse_hash", "chunk_id")
    if table_name == "main_pointers":
        return ("collection", "doc_id", "step_type", "model_tag")
    if table_name == "ingestion_runs":
        return ("collection", "doc_id")
    if table_name.startswith("embeddings_"):
        return ("collection", "doc_id", "chunk_id", "parse_hash", "model")
    return None


def _rag_snapshot_row_key(
    row: Dict[str, Any], key_columns: tuple[str, ...]
) -> tuple[Any, ...]:
    return tuple(row.get(column) for column in key_columns)


def _lancedb_literal(value: Any) -> str:
    from ...core.tools.core.RAG_tools.utils.string_utils import escape_lancedb_string

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return f"'{escape_lancedb_string(str(value))}'"


def _rag_snapshot_key_filter(
    table: Any,
    row: Dict[str, Any],
    key_columns: tuple[str, ...],
    *,
    user_id: int,
    is_admin: bool,
) -> str:
    clauses = []
    for column in key_columns:
        value = row.get(column)
        if value is None:
            clauses.append(f"{column} IS NULL")
        else:
            clauses.append(f"{column} = {_lancedb_literal(value)}")
    if not is_admin and _table_has_user_id_column(table):
        clauses.append(f"user_id = {int(user_id)}")
    return " and ".join(clauses)


def _restore_rag_snapshot_rows(
    table: Any,
    *,
    table_name: str,
    snapshot_rows: List[Dict[str, Any]],
    current_rows: List[Dict[str, Any]],
    user_id: int,
    is_admin: bool,
) -> None:
    """Upsert old rows before deleting stale rows introduced by a failed refresh."""
    key_columns = _rag_snapshot_key_columns(table_name)
    if key_columns is None:
        delete_filters = [
            _rag_snapshot_key_filter(
                table,
                row,
                ("collection", "doc_id"),
                user_id=user_id,
                is_admin=is_admin,
            )
            for row in current_rows
        ]
        if delete_filters:
            table.delete(_combine_lancedb_filters(delete_filters))
        if snapshot_rows:
            table.add(snapshot_rows)
        return

    if snapshot_rows:
        (
            table.merge_insert(list(key_columns))
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(snapshot_rows)
        )

    snapshot_keys = {_rag_snapshot_row_key(row, key_columns) for row in snapshot_rows}
    stale_rows = [
        row
        for row in current_rows
        if _rag_snapshot_row_key(row, key_columns) not in snapshot_keys
    ]
    delete_filters = [
        _rag_snapshot_key_filter(
            table,
            row,
            key_columns,
            user_id=user_id,
            is_admin=is_admin,
        )
        for row in stale_rows
    ]
    if delete_filters:
        table.delete(_combine_lancedb_filters(delete_filters))


def _restore_rag_document_snapshot(
    snapshot: _RagDocumentSnapshot,
    *,
    user_id: int,
    is_admin: bool,
) -> None:
    """Restore RAG document rows after a failed refresh of an existing web file."""
    from ...core.tools.core.RAG_tools.LanceDB.schema_manager import (
        _safe_close_table,
        ensure_chunks_table,
        ensure_documents_table,
        ensure_ingestion_runs_table,
        ensure_main_pointers_table,
        ensure_parses_table,
    )
    from ...core.tools.core.RAG_tools.utils.lancedb_query_utils import (
        list_table_names,
        query_to_list,
    )
    from ...providers.vector_store.lancedb import get_connection_from_env

    vector_store = get_vector_index_store()
    conn = get_connection_from_env()
    ensure_documents_table(conn)
    ensure_parses_table(conn)
    ensure_chunks_table(conn)
    ensure_main_pointers_table(conn)
    ensure_ingestion_runs_table(conn)

    current_table_names = set(list_table_names(conn))
    restore_table_names = [
        table_name
        for table_name in (
            "documents",
            "parses",
            "chunks",
            "main_pointers",
            "ingestion_runs",
        )
        if table_name in current_table_names or table_name in snapshot.rows_by_table
    ]
    for table_name in sorted(current_table_names):
        if table_name.startswith("embeddings_"):
            restore_table_names.append(table_name)
    for table_name in snapshot.rows_by_table:
        if (
            table_name.startswith("embeddings_")
            and table_name not in restore_table_names
        ):
            restore_table_names.append(table_name)

    for table_name in restore_table_names:
        snapshot_rows = snapshot.rows_by_table.get(table_name, [])
        table = None
        try:
            table = conn.open_table(table_name)
            if snapshot.doc_refs:
                current_rows = query_to_list(
                    table.search()
                    .where(
                        _rag_document_refs_filter(
                            table,
                            snapshot.doc_refs,
                            user_id=user_id,
                            is_admin=is_admin,
                        )
                    )
                    .limit(-1)
                )
            else:
                current_rows = []
            _restore_rag_snapshot_rows(
                table,
                table_name=table_name,
                snapshot_rows=snapshot_rows,
                current_rows=current_rows,
                user_id=user_id,
                is_admin=is_admin,
            )
        finally:
            _safe_close_table(table)

    invalidate_cache = getattr(vector_store, "invalidate_table_cache", None)
    if callable(invalidate_cache):
        invalidate_cache()


def _list_document_refs_for_uploaded_file(file_id: str) -> List[tuple[str, str]]:
    from ...core.tools.core.RAG_tools.LanceDB.schema_manager import (
        _safe_close_table,
        ensure_documents_table,
    )
    from ...core.tools.core.RAG_tools.utils.lancedb_query_utils import query_to_list
    from ...core.tools.core.RAG_tools.utils.string_utils import escape_lancedb_string
    from ...providers.vector_store.lancedb import get_connection_from_env

    conn = get_connection_from_env()
    ensure_documents_table(conn)
    documents_table = None
    try:
        documents_table = conn.open_table("documents")
        safe_file_id = escape_lancedb_string(file_id)
        rows = query_to_list(
            documents_table.search()
            .where(f"file_id = '{safe_file_id}'")
            .select(["collection", "doc_id"])
            .limit(-1)
        )
        doc_refs: List[tuple[str, str]] = []
        for row in rows:
            collection = str(row.get("collection") or "").strip()
            doc_id = str(row.get("doc_id") or "").strip()
            if collection and doc_id:
                doc_refs.append((collection, doc_id))
        return doc_refs
    finally:
        _safe_close_table(documents_table)


def _snapshot_ingestion_runs_for_uploaded_file(
    file_id: str,
) -> Optional[_IngestionRunsSnapshot]:
    """Snapshot current ingestion status rows before refreshing an existing file."""
    try:
        from ...core.tools.core.RAG_tools.LanceDB.schema_manager import (
            _safe_close_table,
            ensure_ingestion_runs_table,
        )
        from ...core.tools.core.RAG_tools.utils.lancedb_query_utils import query_to_list
        from ...providers.vector_store.lancedb import get_connection_from_env

        doc_refs = _list_document_refs_for_uploaded_file(file_id)
        conn = get_connection_from_env()
        ensure_ingestion_runs_table(conn)
        ingestion_runs_table = None
        try:
            ingestion_runs_table = conn.open_table("ingestion_runs")
            if doc_refs:
                combined_filter = _combine_lancedb_filters(
                    [
                        _ingestion_run_filter(collection, doc_id)
                        for collection, doc_id in doc_refs
                    ]
                )
                rows = query_to_list(
                    ingestion_runs_table.search()
                    .where(combined_filter)
                    .select(list(_INGESTION_RUN_COLUMNS))
                    .limit(-1)
                )
            else:
                rows = []
            return _IngestionRunsSnapshot(doc_refs=doc_refs, rows=rows)
        finally:
            _safe_close_table(ingestion_runs_table)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to snapshot ingestion runs before web file refresh: "
            "file_id=%s, error=%s",
            file_id,
            exc,
            exc_info=True,
        )
        return None


def _restore_ingestion_runs_snapshot(snapshot: _IngestionRunsSnapshot) -> None:
    """Restore ingestion status rows after a failed existing-file refresh."""
    from ...core.tools.core.RAG_tools.LanceDB.schema_manager import (
        _safe_close_table,
        ensure_ingestion_runs_table,
    )
    from ...providers.vector_store.lancedb import get_connection_from_env

    conn = get_connection_from_env()
    ensure_ingestion_runs_table(conn)
    ingestion_runs_table = None
    try:
        ingestion_runs_table = conn.open_table("ingestion_runs")
        if snapshot.doc_refs:
            combined_filter = _combine_lancedb_filters(
                [
                    _ingestion_run_filter(collection, doc_id)
                    for collection, doc_id in snapshot.doc_refs
                ]
            )
            ingestion_runs_table.delete(combined_filter)
        if snapshot.rows:
            ingestion_runs_table.add(snapshot.rows)
    finally:
        _safe_close_table(ingestion_runs_table)


_UPLOADED_FILE_ROLLBACK_FIELDS = (
    "filename",
    "storage_path",
    "storage_backend",
    "storage_key",
    "storage_uri",
    "checksum",
    "etag",
    "workspace_relative_path",
    "workspace_category",
    "storage_status",
    "mime_type",
    "file_size",
)


def _snapshot_uploaded_file_record(file_record: UploadedFile) -> Dict[str, Any]:
    return {
        field: getattr(file_record, field, None)
        for field in _UPLOADED_FILE_ROLLBACK_FIELDS
    }


def _restore_uploaded_file_record_snapshot(
    file_record: UploadedFile, snapshot: Dict[str, Any]
) -> None:
    for field, value in snapshot.items():
        setattr(file_record, field, value)


def _cleanup_failed_web_uploaded_file_setup(
    db_session: Session,
    *,
    file_id: str,
    user_id: int,
    filename: str,
    storage_path: Path,
    mime_type: str,
    storage_key: str,
) -> None:
    """Remove DB/durable side effects after web UploadedFile setup fails."""
    cleanup_errors: list[Exception] = []
    try:
        db_session.rollback()
    except Exception as exc:  # noqa: BLE001
        cleanup_errors.append(exc)
        logger.warning(
            "Failed to rollback DB session after web UploadedFile setup failure: %s",
            exc,
            exc_info=True,
        )

    try:
        persisted_record = (
            db_session.query(UploadedFile)
            .filter(
                (UploadedFile.file_id == file_id)
                | (UploadedFile.storage_path == str(storage_path))
            )
            .first()
        )
        if persisted_record is not None:
            UploadedFileStore(db_session).delete(persisted_record, delete_local=False)
            db_session.commit()
        else:
            placeholder = UploadedFile(
                file_id=file_id,
                user_id=user_id,
                filename=filename,
                storage_path=str(storage_path),
                mime_type=mime_type,
                file_size=storage_path.stat().st_size if storage_path.exists() else 0,
                storage_key=storage_key,
                storage_status="available",
            )
            ManagedFileRef(placeholder).delete_durable()
    except Exception as exc:  # noqa: BLE001
        db_session.rollback()
        cleanup_errors.append(exc)
        logger.warning(
            "Failed to clean web UploadedFile setup side effects: "
            "file_id=%s, storage_key=%s, error=%s",
            file_id,
            storage_key,
            exc,
            exc_info=True,
        )

    if cleanup_errors:
        raise RuntimeError(
            f"Failed to clean web UploadedFile setup side effects: {cleanup_errors[0]}"
        ) from cleanup_errors[0]


def _create_web_uploaded_file_record(
    db_session: Session,
    *,
    user_id: int,
    filename: str,
    storage_path: Path,
    mime_type: str,
) -> UploadedFile:
    """Create a web UploadedFile with a known durable key for compensation."""
    file_id = str(uuid.uuid4())
    storage_key = build_upload_storage_key(user_id, file_id, filename)
    try:
        file_record = UploadedFileStore(db_session).create_from_local_path(
            local_path=storage_path,
            user_id=user_id,
            filename=filename,
            file_id=file_id,
            mime_type=mime_type,
            storage_key=storage_key,
        )
        db_session.commit()
        db_session.refresh(file_record)
        return file_record
    except Exception:
        _cleanup_failed_web_uploaded_file_setup(
            db_session,
            file_id=file_id,
            user_id=user_id,
            filename=filename,
            storage_path=storage_path,
            mime_type=mime_type,
            storage_key=storage_key,
        )
        raise


def _existing_web_file_result_with_rollback(
    *,
    existing_record: Any,
    file_path: Path,
    collection_name: str,
    user_id: int,
    is_admin: bool,
    url: str,
    context: str,
) -> FileHandlerResult:
    """Return an existing web file only after preparing failure compensation."""
    file_record_id = str(existing_record.file_id)
    ingestion_runs_snapshot = _snapshot_ingestion_runs_for_uploaded_file(file_record_id)
    if ingestion_runs_snapshot is None:
        raise RuntimeError(
            "Failed to snapshot ingestion status before reusing existing web file"
        )

    rag_document_snapshot = _snapshot_rag_documents_for_uploaded_file(
        file_record_id,
        user_id=user_id,
        is_admin=is_admin,
    )
    if rag_document_snapshot is None:
        raise RuntimeError(
            "Failed to snapshot RAG document rows before reusing existing web file"
        )

    def _rollback_existing(
        ingestion_result: Optional[IngestionResult] = None,
    ) -> None:
        rollback_error: Optional[Exception] = None
        try:
            _rollback_failed_web_document_ingestion(
                collection_name=collection_name,
                result=ingestion_result,
                user_id=user_id,
                is_admin=is_admin,
                rag_snapshot=rag_document_snapshot,
                file_id=file_record_id,
            )
        except Exception as exc:  # noqa: BLE001
            rollback_error = exc
            logger.warning(
                "Failed to restore RAG rows during existing web file rollback: "
                "url=%s, file_id=%s, context=%s, error=%s",
                url,
                file_record_id,
                context,
                exc,
                exc_info=True,
            )

        try:
            _restore_ingestion_runs_snapshot(ingestion_runs_snapshot)
        except Exception as exc:  # noqa: BLE001
            if rollback_error is None:
                rollback_error = exc
            logger.warning(
                "Failed to restore ingestion runs during existing web file rollback: "
                "url=%s, file_id=%s, context=%s, error=%s",
                url,
                file_record_id,
                context,
                exc,
                exc_info=True,
            )

        if rollback_error is not None:
            raise rollback_error

    return FileHandlerResult(
        file_path=str(file_path),
        file_id=file_record_id,
        rollback_on_failure=_rollback_existing,
    )


def _create_new_web_file_handler_result(
    *,
    temp_file_path: Path,
    persistent_file: Path,
    db_session: Session,
    user_id: int,
    is_admin: bool,
    collection_name: str,
    filename: str,
    url: str,
    url_hash: str,
    processed_urls: Dict[str, str],
) -> FileHandlerResult:
    """Persist a new web-ingest file and attach failure compensation."""
    try:
        shutil.copy2(temp_file_path, persistent_file)
        logger.info(
            "Copied web ingestion file from %s to %s",
            temp_file_path,
            persistent_file,
        )

        file_record = _create_web_uploaded_file_record(
            db_session,
            user_id=user_id,
            filename=filename,
            storage_path=persistent_file,
            mime_type="text/markdown",
        )
        logger.info(
            "Created UploadedFile record for web ingestion: file_id=%s, filename=%s, url=%s",
            file_record.file_id,
            filename,
            url,
        )

        processed_urls[url_hash] = str(file_record.file_id)
        file_record_id = str(file_record.file_id)
        persistent_file_path = Path(persistent_file)

        def _rollback_new_web_file(
            ingestion_result: Optional[IngestionResult] = None,
        ) -> None:
            rollback_error: Optional[Exception] = None
            try:
                _rollback_failed_web_document_ingestion(
                    collection_name=collection_name,
                    result=ingestion_result,
                    user_id=user_id,
                    is_admin=is_admin,
                    file_id=file_record_id,
                )
            except Exception as exc:  # noqa: BLE001
                rollback_error = exc
                logger.warning(
                    "Failed to clean RAG rows during new web file rollback: "
                    "file_id=%s, error=%s",
                    file_record_id,
                    exc,
                    exc_info=True,
                )
            SessionLocal = get_session_local()
            rollback_db = SessionLocal()
            try:
                rollback_record = (
                    rollback_db.query(UploadedFile)
                    .filter(UploadedFile.file_id == file_record_id)
                    .first()
                )
                if rollback_record is not None:
                    UploadedFileStore(rollback_db).delete(
                        rollback_record,
                        delete_local=True,
                    )
                else:
                    persistent_file_path.unlink(missing_ok=True)
                rollback_db.commit()
            except Exception:
                rollback_db.rollback()
                raise
            finally:
                rollback_db.close()
            if rollback_error is not None:
                raise rollback_error

        return FileHandlerResult(
            file_path=str(persistent_file),
            file_id=file_record_id,
            rollback_on_failure=_rollback_new_web_file,
        )
    except Exception:
        if persistent_file.exists():
            try:
                persistent_file.unlink()
                logger.warning(
                    "Cleaned up orphaned persistent file due to web file setup failure: %s",
                    persistent_file,
                )
            except Exception as cleanup_error:  # noqa: BLE001
                logger.warning(
                    "Failed to clean up orphaned persistent file %s: %s",
                    persistent_file,
                    cleanup_error,
                )
        raise


def _refresh_existing_file_if_changed(
    existing_record: Any,
    temp_file_path: Path,
    db_session: Session,
    user_id: int,
    is_admin: bool,
    collection_name: str,
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
    record_snapshot = _snapshot_uploaded_file_record(existing_record)
    if not existing_path.exists():
        try:
            existing_path = ManagedFileRef(existing_record).ensure_local()
            record_snapshot["storage_path"] = str(existing_path)
        except DurableObjectMissingError:
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to restore durable web-ingestion file before refresh: "
                "url=%s, file_id=%s, context=%s, error=%s",
                url,
                existing_record.file_id,
                context,
                exc,
            )
            return None

    old_hash = _get_file_sha256(existing_path)
    new_hash = _get_file_sha256(temp_file_path)

    if old_hash == new_hash:
        return _existing_web_file_result_with_rollback(
            existing_record=existing_record,
            file_path=existing_path,
            collection_name=collection_name,
            user_id=user_id,
            is_admin=is_admin,
            url=url,
            context=context,
        )

    ingestion_runs_snapshot = _snapshot_ingestion_runs_for_uploaded_file(
        str(existing_record.file_id)
    )
    if ingestion_runs_snapshot is None:
        raise RuntimeError(
            "Failed to snapshot ingestion status before refreshing existing web file"
        )

    rag_document_snapshot = _snapshot_rag_documents_for_uploaded_file(
        str(existing_record.file_id),
        user_id=user_id,
        is_admin=is_admin,
    )
    if rag_document_snapshot is None:
        raise RuntimeError(
            "Failed to snapshot RAG document rows before refreshing existing web file"
        )

    # Content changed - first try to mark for reindex BEFORE modifying file
    if not _mark_uploaded_file_for_reindex(str(existing_record.file_id)):
        raise RuntimeError(
            "Failed to mark existing web file for reindex before refresh"
        )

    # Mark succeeded - now atomically replace the file
    backup_path = _build_ingest_backup_path(existing_path)
    try:
        shutil.copy2(existing_path, backup_path)
    except Exception:
        try:
            _restore_ingestion_runs_snapshot(ingestion_runs_snapshot)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to restore ingestion runs after web file backup failure: "
                "file_id=%s, error=%s",
                existing_record.file_id,
                exc,
                exc_info=True,
            )
            raise exc
        raise
    refresh_snapshot = _capture_uploaded_file_refresh_snapshot(
        existing_record,
        backup_path=backup_path,
        reindex_marker_applied=True,
    )
    try:
        _atomic_replace_file(temp_file_path, existing_path)
        file_record = _upsert_uploaded_file_record(
            db_session,
            user_id=user_id,
            filename=filename,
            storage_path=existing_path,
            mime_type="text/markdown",
            file_size=existing_path.stat().st_size,
        )
    except Exception as exc:
        restore_error: Optional[Exception] = None
        restore_result = _restore_uploaded_file_refresh_snapshot(
            db_session,
            refresh_snapshot,
        )
        if restore_result.side_effects_may_remain:
            restore_error = RollbackFailureError(
                "Failed to fully restore refreshed web file "
                f"for {url}: {'; '.join(restore_result.errors) or exc}"
            )
        try:
            _restore_ingestion_runs_snapshot(ingestion_runs_snapshot)
        except Exception as runs_exc:  # noqa: BLE001
            restore_error = runs_exc
            logger.warning(
                "Failed to restore ingestion runs after web file refresh setup "
                "failure: file_id=%s, error=%s",
                existing_record.file_id,
                runs_exc,
                exc_info=True,
            )
        if restore_error is not None:
            raise restore_error from exc
        backup_path.unlink(missing_ok=True)
        raise
    processed_urls[url_hash] = str(file_record.file_id)

    logger.info(
        "Marked changed web file as PENDING_REINDEX and refreshed content: url=%s, file_id=%s, context=%s",
        url,
        file_record.file_id,
        context,
    )

    file_record_id = str(file_record.file_id)

    def _rollback_refresh(
        ingestion_result: Optional[IngestionResult] = None,
    ) -> None:
        if not backup_path.exists():
            raise FileNotFoundError(
                f"Missing web ingest rollback backup: {backup_path}"
            )

        SessionLocal = get_session_local()
        rollback_db = SessionLocal()
        rollback_succeeded = False
        try:
            rollback_error: Optional[Exception] = None
            try:
                _rollback_failed_web_document_ingestion(
                    collection_name=collection_name,
                    result=ingestion_result,
                    user_id=user_id,
                    is_admin=is_admin,
                    rag_snapshot=rag_document_snapshot,
                    file_id=file_record_id,
                )
            except Exception as exc:  # noqa: BLE001
                rollback_error = exc
                logger.warning(
                    "Failed to restore RAG rows during web file refresh rollback: "
                    "file_id=%s, error=%s",
                    file_record_id,
                    exc,
                    exc_info=True,
                )

            try:
                refreshed_record = (
                    rollback_db.query(UploadedFile)
                    .filter(UploadedFile.file_id == file_record_id)
                    .first()
                )
                if refreshed_record is None:
                    _restore_ingest_file_backup(
                        file_path=existing_path,
                        backup_path=backup_path,
                        had_existing_file=True,
                    )
                    rollback_db.commit()
                else:
                    current_storage_key = str(
                        getattr(refreshed_record, "storage_key", "") or ""
                    )
                    previous_storage_key = str(record_snapshot.get("storage_key") or "")
                    if (
                        current_storage_key
                        and current_storage_key != previous_storage_key
                    ):
                        ManagedFileRef(refreshed_record).delete_durable()

                    _restore_ingest_file_backup(
                        file_path=existing_path,
                        backup_path=backup_path,
                        had_existing_file=True,
                    )
                    _restore_uploaded_file_record_snapshot(
                        refreshed_record, record_snapshot
                    )
                    if previous_storage_key:
                        UploadedFileStore(rollback_db).sync_existing(
                            refreshed_record,
                            storage_key=previous_storage_key,
                            mime_type=record_snapshot.get("mime_type"),
                        )
                    else:
                        rollback_db.flush()
                    rollback_db.commit()
            except Exception as exc:  # noqa: BLE001
                rollback_db.rollback()
                if rollback_error is None:
                    rollback_error = exc
                logger.warning(
                    "Failed to restore UploadedFile/local file during web file "
                    "refresh rollback: file_id=%s, error=%s",
                    file_record_id,
                    exc,
                    exc_info=True,
                )

            try:
                _restore_ingestion_runs_snapshot(ingestion_runs_snapshot)
            except Exception as exc:  # noqa: BLE001
                if rollback_error is None:
                    rollback_error = exc
                logger.warning(
                    "Failed to restore ingestion runs during web file refresh "
                    "rollback: file_id=%s, error=%s",
                    file_record_id,
                    exc,
                    exc_info=True,
                )

            if rollback_error is not None:
                raise rollback_error
            rollback_succeeded = True
        except Exception:
            raise
        finally:
            rollback_db.close()
            if rollback_succeeded:
                backup_path.unlink(missing_ok=True)
            elif backup_path.exists():
                logger.warning(
                    "Preserving web ingest rollback backup after failed rollback: %s",
                    backup_path,
                )

    def _commit_refresh() -> None:
        backup_path.unlink(missing_ok=True)

    return FileHandlerResult(
        file_path=str(existing_record.storage_path),
        file_id=str(existing_record.file_id),
        rollback_on_failure=_rollback_refresh,
        commit_on_success=_commit_refresh,
    )


def _recreate_missing_existing_file(
    *,
    existing_record: Any,
    temp_file_path: Path,
    db_session: Session,
    user_id: int,
    is_admin: bool,
    collection_name: str,
    filename: str,
    url_hash: str,
    processed_urls: Dict[str, str],
) -> FileHandlerResult:
    existing_path = Path(str(existing_record.storage_path))
    record_snapshot = _snapshot_uploaded_file_record(existing_record)
    ingestion_runs_snapshot = _snapshot_ingestion_runs_for_uploaded_file(
        str(existing_record.file_id)
    )
    if ingestion_runs_snapshot is None:
        raise RuntimeError(
            "Failed to snapshot ingestion status before recreating existing web file"
        )

    rag_document_snapshot = _snapshot_rag_documents_for_uploaded_file(
        str(existing_record.file_id),
        user_id=user_id,
        is_admin=is_admin,
    )
    if rag_document_snapshot is None:
        raise RuntimeError(
            "Failed to snapshot RAG document rows before recreating existing web file"
        )

    backup_path: Optional[Path] = None
    had_existing_file = existing_path.exists()
    if had_existing_file:
        backup_path = _build_ingest_backup_path(existing_path)
        shutil.copy2(existing_path, backup_path)

    try:
        _atomic_replace_file(temp_file_path, existing_path)
        file_record = _upsert_uploaded_file_record(
            db_session,
            user_id=user_id,
            filename=filename,
            storage_path=existing_path,
            mime_type="text/markdown",
            file_size=existing_path.stat().st_size,
        )
    except Exception as setup_exc:
        restore_error: Optional[Exception] = None
        try:
            db_session.rollback()
        except Exception as exc:  # noqa: BLE001
            restore_error = exc
            logger.warning(
                "Failed to rollback DB session after web file recreate setup "
                "failure: file_id=%s, error=%s",
                existing_record.file_id,
                exc,
                exc_info=True,
            )
        try:
            _restore_ingest_file_backup(
                file_path=existing_path,
                backup_path=backup_path,
                had_existing_file=had_existing_file,
            )
        except Exception as exc:  # noqa: BLE001
            restore_error = exc
            logger.warning(
                "Failed to restore local file after web file recreate setup "
                "failure: file_id=%s, path=%s, error=%s",
                existing_record.file_id,
                existing_path,
                exc,
                exc_info=True,
            )
        try:
            refreshed_record = (
                db_session.query(UploadedFile)
                .filter(UploadedFile.file_id == str(existing_record.file_id))
                .first()
            )
            if refreshed_record is not None:
                current_storage_key = str(
                    getattr(refreshed_record, "storage_key", "") or ""
                )
                previous_storage_key = str(record_snapshot.get("storage_key") or "")
                if current_storage_key and (
                    current_storage_key != previous_storage_key or not had_existing_file
                ):
                    ManagedFileRef(refreshed_record).delete_durable()
                _restore_uploaded_file_record_snapshot(
                    refreshed_record, record_snapshot
                )
                if previous_storage_key and existing_path.exists():
                    UploadedFileStore(db_session).sync_existing(
                        refreshed_record,
                        storage_key=previous_storage_key,
                        mime_type=record_snapshot.get("mime_type"),
                    )
                else:
                    db_session.flush()
                db_session.commit()
        except Exception as exc:  # noqa: BLE001
            db_session.rollback()
            restore_error = exc
            logger.warning(
                "Failed to restore UploadedFile record after web file recreate "
                "setup failure: file_id=%s, error=%s",
                existing_record.file_id,
                exc,
                exc_info=True,
            )
        try:
            _restore_ingestion_runs_snapshot(ingestion_runs_snapshot)
        except Exception as exc:  # noqa: BLE001
            restore_error = exc
            logger.warning(
                "Failed to restore ingestion runs after web file recreate setup "
                "failure: file_id=%s, error=%s",
                existing_record.file_id,
                exc,
                exc_info=True,
            )
        if restore_error is not None:
            raise restore_error from setup_exc
        raise

    processed_urls[url_hash] = str(file_record.file_id)

    file_record_id = str(file_record.file_id)
    backup_for_failure = backup_path

    def _rollback_recreate(
        ingestion_result: Optional[IngestionResult] = None,
    ) -> None:
        SessionLocal = get_session_local()
        rollback_db = SessionLocal()
        rollback_succeeded = False
        try:
            rollback_error: Optional[Exception] = None
            try:
                _rollback_failed_web_document_ingestion(
                    collection_name=collection_name,
                    result=ingestion_result,
                    user_id=user_id,
                    is_admin=is_admin,
                    rag_snapshot=rag_document_snapshot,
                    file_id=file_record_id,
                )
            except Exception as exc:  # noqa: BLE001
                rollback_error = exc
                logger.warning(
                    "Failed to clean RAG rows during recreated web file rollback: "
                    "file_id=%s, error=%s",
                    file_record_id,
                    exc,
                    exc_info=True,
                )

            try:
                refreshed_record = (
                    rollback_db.query(UploadedFile)
                    .filter(UploadedFile.file_id == file_record_id)
                    .first()
                )
                if refreshed_record is not None:
                    current_storage_key = str(
                        getattr(refreshed_record, "storage_key", "") or ""
                    )
                    previous_storage_key = str(record_snapshot.get("storage_key") or "")
                    if (
                        current_storage_key
                        and current_storage_key != previous_storage_key
                    ):
                        ManagedFileRef(refreshed_record).delete_durable()

                    _restore_uploaded_file_record_snapshot(
                        refreshed_record, record_snapshot
                    )
                    if previous_storage_key and backup_for_failure is not None:
                        _restore_ingest_file_backup(
                            file_path=existing_path,
                            backup_path=backup_for_failure,
                            had_existing_file=True,
                        )
                        UploadedFileStore(rollback_db).sync_existing(
                            refreshed_record,
                            storage_key=previous_storage_key,
                            mime_type=record_snapshot.get("mime_type"),
                        )
                    else:
                        _restore_ingest_file_backup(
                            file_path=existing_path,
                            backup_path=backup_for_failure,
                            had_existing_file=had_existing_file,
                        )
                        rollback_db.flush()
                else:
                    _restore_ingest_file_backup(
                        file_path=existing_path,
                        backup_path=backup_for_failure,
                        had_existing_file=had_existing_file,
                    )
                rollback_db.commit()
            except Exception as exc:  # noqa: BLE001
                rollback_db.rollback()
                if rollback_error is None:
                    rollback_error = exc
                logger.warning(
                    "Failed to restore UploadedFile/local file during recreated "
                    "web file rollback: file_id=%s, error=%s",
                    file_record_id,
                    exc,
                    exc_info=True,
                )

            try:
                _restore_ingestion_runs_snapshot(ingestion_runs_snapshot)
            except Exception as exc:  # noqa: BLE001
                if rollback_error is None:
                    rollback_error = exc
                logger.warning(
                    "Failed to restore ingestion runs during recreated web file "
                    "rollback: file_id=%s, error=%s",
                    file_record_id,
                    exc,
                    exc_info=True,
                )

            if rollback_error is not None:
                raise rollback_error
            rollback_succeeded = True
        except Exception:
            raise
        finally:
            rollback_db.close()
            if rollback_succeeded and backup_for_failure is not None:
                backup_for_failure.unlink(missing_ok=True)
            elif backup_for_failure is not None and backup_for_failure.exists():
                logger.warning(
                    "Preserving web ingest rollback backup after failed rollback: %s",
                    backup_for_failure,
                )

    def _commit_recreate() -> None:
        if backup_for_failure is not None:
            backup_for_failure.unlink(missing_ok=True)

    return FileHandlerResult(
        file_path=str(existing_record.storage_path),
        file_id=str(existing_record.file_id),
        rollback_on_failure=_rollback_recreate,
        commit_on_success=_commit_recreate,
    )


def _compensate_new_web_ingest_files(
    db: Session,
    *,
    file_ids: set[str],
    user_id: int,
) -> tuple[bool, list[str]]:
    cleanup_incomplete = False
    cleanup_errors: list[str] = []
    for file_id in sorted(file_ids):
        cleanup_result = _compensate_new_uploaded_file(
            db,
            file_id=file_id,
            user_id=user_id,
        )
        if cleanup_result.side_effects_may_remain:
            cleanup_incomplete = True
            cleanup_errors.extend(cleanup_result.errors)
            db.rollback()
            continue
        try:
            db.commit()
        except Exception as commit_exc:  # noqa: BLE001
            cleanup_incomplete = True
            cleanup_errors.append(
                f"Database commit failed for file {file_id}: {commit_exc}"
            )
            db.rollback()
    return cleanup_incomplete, cleanup_errors


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
        except RollbackFailureError as e:
            logger.error("KB rollback failure in %s: %s", func.__name__, e)
            raise HTTPException(status_code=500, detail=str(e))
        except (ValueError, KeyError, TypeError) as e:
            logger.error("Data format error in %s: %s", func.__name__, e)
            raise HTTPException(status_code=400, detail=f"Data format error: {str(e)}")
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


@dataclass
class _CollectionConfigSnapshot:
    metadata_store: Any
    collection: str
    user_id: int
    previous_config_json: Optional[str]
    previous_config_known: bool
    saved: bool


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _save_collection_config_with_snapshot(
    *,
    collection: str,
    config_json: str,
    user: User,
    context: str,
) -> _CollectionConfigSnapshot:
    """Save collection config while retaining the caller's previous config."""
    from ...core.tools.core.RAG_tools.storage.factory import get_metadata_store

    metadata_store = get_metadata_store()
    previous_config_json: Optional[str] = None
    previous_config_known = False

    try:
        get_config = getattr(metadata_store, "get_collection_config", None)
        if callable(get_config):
            loaded = get_config(
                collection=collection,
                user_id=int(user.id),
                is_admin=False,
            )
            loaded = await _maybe_await(loaded)
            if isinstance(loaded, str):
                previous_config_json = loaded
                previous_config_known = True
            elif loaded is None:
                previous_config_known = True
            else:
                logger.warning(
                    "Unexpected collection config snapshot type during %s: %s",
                    context,
                    type(loaded).__name__,
                )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to snapshot collection config during %s: %s",
            context,
            exc,
        )

    if not previous_config_known:
        logger.warning(
            "Skipping collection config save during %s because previous config "
            "state could not be read for %s/user_%s",
            context,
            collection,
            int(user.id),
        )
        return _CollectionConfigSnapshot(
            metadata_store=metadata_store,
            collection=collection,
            user_id=int(user.id),
            previous_config_json=previous_config_json,
            previous_config_known=previous_config_known,
            saved=False,
        )

    try:
        await metadata_store.save_collection_config(
            collection=collection,
            config_json=config_json,
            user_id=int(user.id),
        )
        return _CollectionConfigSnapshot(
            metadata_store=metadata_store,
            collection=collection,
            user_id=int(user.id),
            previous_config_json=previous_config_json,
            previous_config_known=previous_config_known,
            saved=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to save collection config during %s: %s", context, exc)
        return _CollectionConfigSnapshot(
            metadata_store=metadata_store,
            collection=collection,
            user_id=int(user.id),
            previous_config_json=previous_config_json,
            previous_config_known=previous_config_known,
            saved=False,
        )


async def _restore_collection_config_after_failed_ingest(
    *,
    snapshot: Optional[_CollectionConfigSnapshot],
    collection_existed_before: bool,
    context: str,
) -> None:
    """Undo the config save made before an ingest that ultimately failed."""
    if snapshot is None or not snapshot.saved:
        return

    try:
        if snapshot.previous_config_json is not None:
            await snapshot.metadata_store.save_collection_config(
                collection=snapshot.collection,
                config_json=snapshot.previous_config_json,
                user_id=snapshot.user_id,
            )
            logger.info(
                "Restored previous collection config after failed %s: %s/user_%s",
                context,
                snapshot.collection,
                snapshot.user_id,
            )
            return

        if not snapshot.previous_config_known:
            logger.warning(
                "Skipping collection config deletion after failed %s because "
                "previous config state is unknown: %s/user_%s",
                context,
                snapshot.collection,
                snapshot.user_id,
            )
            return

        delete_result = snapshot.metadata_store.delete_collection_metadata(
            collection_name=snapshot.collection,
            user_id=snapshot.user_id,
            is_admin=False,
            delete_orphaned_metadata=not collection_existed_before,
        )
        await _maybe_await(delete_result)
        logger.info(
            "Removed collection config created by failed %s: %s/user_%s",
            context,
            snapshot.collection,
            snapshot.user_id,
        )
    except Exception as exc:  # noqa: BLE001
        raise RollbackFailureError(
            "Failed to restore collection config after failed "
            f"{context} for {snapshot.collection}/user_{snapshot.user_id}: {exc}"
        ) from exc


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
        file_backup_path = _build_ingest_backup_path(file_path)
        await asyncio.to_thread(shutil.copy2, file_path, file_backup_path)

    try:
        copy_result = await asyncio.to_thread(
            _copy_upload_file_to_path, file, file_path
        )
        total_size = copy_result.total_size
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

    config_snapshot = await _save_collection_config_with_snapshot(
        collection=safe_collection,
        config_json=config.model_dump_json(exclude_unset=True),
        user=_user,
        context="ingest",
    )
    effective_collection_existed_before = _collection_or_config_existed_before(
        collection_existed_before,
        config_snapshot,
    )

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
        result = _with_user_actionable_ingestion_message(
            result,
            embedding_model_id=embedding_model_id,
        )

        if result.status in {"error", "partial"}:
            await _rollback_failed_ingestion(
                db=db,
                user=_user,
                collection_name=collection,
                result=result,
                file_path=file_path,
                file_record=file_record,
                collection_existed_before=effective_collection_existed_before,
                uploaded_file_existed_before=uploaded_file_existed_before,
                file_backup_path=file_backup_path,
                had_existing_file=had_existing_file,
                embedding_model_id=embedding_model_id,
            )
            if effective_collection_existed_before:
                await _restore_or_cleanup_collection_config_after_failed_ingest(
                    snapshot=config_snapshot,
                    collection_existed_before=collection_existed_before,
                    collection_name=collection,
                    user=_user,
                    context="ingest",
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
                collection_existed_before=effective_collection_existed_before,
                uploaded_file_existed_before=uploaded_file_existed_before,
                file_backup_path=file_backup_path,
                had_existing_file=had_existing_file,
                embedding_model_id=embedding_model_id,
            )
            if effective_collection_existed_before:
                await _restore_or_cleanup_collection_config_after_failed_ingest(
                    snapshot=config_snapshot,
                    collection_existed_before=collection_existed_before,
                    collection_name=collection,
                    user=_user,
                    context="ingest",
                )
        else:
            _restore_ingest_file_backup(
                file_path=file_path,
                backup_path=file_backup_path,
                had_existing_file=had_existing_file,
            )
            await _restore_or_cleanup_collection_config_after_failed_ingest(
                snapshot=config_snapshot,
                collection_existed_before=collection_existed_before,
                collection_name=collection,
                user=_user,
                context="ingest",
            )
        raise


@kb_router.post(
    "/ingest/jobs",
    response_model=BackgroundJobResponse,
    status_code=202,
)
@with_kb_user_scope
@handle_kb_exceptions
async def create_ingest_job(
    collection: str = Form(None),
    file: UploadFile = File(...),
    *,
    parse_method: Optional[ParseMethod] = Form(None),
    chunk_strategy: Optional[ChunkStrategy] = Form(None),
    chunk_size: Optional[int] = Form(None, gt=0),
    chunk_overlap: Optional[int] = Form(None, ge=0),
    separators: Optional[str] = Form(None),
    embedding_model_id: str = Form("text-embedding-v4"),
    embedding_batch_size: Optional[int] = Form(None, gt=0),
    max_retries: Optional[int] = Form(None, ge=0),
    retry_delay: Optional[float] = Form(None, ge=0.0),
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    """Upload a document and enqueue durable KB ingestion."""
    if not file.filename or not file.filename.strip():
        raise HTTPException(status_code=422, detail="No filename provided")

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

    try:
        safe_collection = sanitize_path_component(collection, "collection")
        file_path = Path(
            get_upload_path(
                safe_filename,
                user_id=int(_user.id),
                collection=safe_collection,
                collection_is_sanitized=True,
            )
        )
    except ValueError as e:
        raise HTTPException(
            status_code=422, detail=f"Invalid collection name: {str(e)}"
        ) from e

    await _ensure_collection_access(safe_collection, _user, allow_create=True)
    await _ensure_background_job_queue_available_async()

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
    file_id = (
        str(existing_file_record.file_id)
        if existing_file_record is not None
        else _background_ingest_file_id(user_id=int(_user.id), storage_path=file_path)
    )
    staged_file_path = _build_background_ingest_staging_path(
        user_id=int(_user.id),
        filename=safe_filename,
    )

    try:
        copy_result = await asyncio.to_thread(
            _copy_upload_file_to_path,
            file,
            staged_file_path,
        )
        total_size = copy_result.total_size
        file_sha256 = copy_result.sha256
    except HTTPException:
        _cleanup_background_ingest_staging_file(staged_file_path)
        raise
    except Exception as upload_exc:
        _cleanup_background_ingest_staging_file(staged_file_path)
        raise upload_exc

    mime_type = (
        getattr(file, "content_type", None)
        or mimetypes.guess_type(safe_filename)[0]
        or "application/octet-stream"
    )

    final_chunk_size = chunk_size if chunk_size is not None and chunk_size > 0 else 1000
    final_chunk_overlap = (
        chunk_overlap if chunk_overlap is not None and chunk_overlap >= 0 else 200
    )
    if final_chunk_overlap >= final_chunk_size:
        final_chunk_overlap = min(int(final_chunk_size * 0.2), final_chunk_size - 1)

    parsed_separators = _parse_separators(separators)
    final_strategy = (
        chunk_strategy if chunk_strategy is not None else ChunkStrategy.RECURSIVE
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

    idempotency_key = _background_job_idempotency_key(
        "kb.ingest.document",
        {
            "collection": safe_collection,
            "filename": safe_filename,
            "file_sha256": file_sha256,
            "ingestion_config": config.model_dump(mode="json"),
            "user_id": int(_user.id),
        },
    )
    existing_job = get_non_terminal_background_job_by_idempotency_key(
        db,
        idempotency_key,
    )
    if existing_job is not None:
        _cleanup_background_ingest_staging_file(staged_file_path)
        return existing_job

    generation_id = str(uuid.uuid4())

    job_payload = {
        "collection": safe_collection,
        "source_path": str(staged_file_path),
        "target_path": str(file_path),
        "file_id": file_id,
        "generation_id": generation_id,
        "file_sha256": file_sha256,
        "filename": safe_filename,
        "mime_type": mime_type,
        "file_size": int(total_size),
        "user_id": int(_user.id),
        "is_admin": bool(_user.is_admin),
        "ingestion_config": config.model_dump(mode="json"),
        "collection_existed_before": collection_existed_before,
    }

    try:
        job = create_background_job(
            db,
            user_id=int(_user.id),
            job_type=BackgroundJobType.KB_INGEST_DOCUMENT,
            payload=job_payload,
            idempotency_key=idempotency_key,
            reuse_terminal_idempotency_key=False,
        )
        if dict(job.payload or {}).get("source_path") != str(staged_file_path):
            _cleanup_background_ingest_staging_file(staged_file_path)
            return job
        admit_kb_ingest_target(
            db,
            user_id=int(_user.id),
            collection=safe_collection,
            target_path=str(file_path),
            file_id=file_id,
            generation_id=generation_id,
            job_id=str(job.id),
            file_sha256=file_sha256,
        )
    except Exception:
        db.rollback()
        _cleanup_background_ingest_staging_file(staged_file_path)
        raise
    try:
        return await _enqueue_background_job_or_503_async(db, job)
    except Exception:
        release_kb_ingest_target_generation(
            db,
            user_id=int(_user.id),
            collection=safe_collection,
            target_path=str(file_path),
            generation_id=generation_id,
        )
        _cleanup_background_ingest_staging_file(staged_file_path)
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

    config_snapshot = await _save_collection_config_with_snapshot(
        collection=safe_collection,
        config_json=config.model_dump_json(exclude_unset=True),
        user=_user,
        context="ingest_cloud",
    )
    effective_collection_existed_before = _collection_or_config_existed_before(
        collection_existed_before,
        config_snapshot,
    )

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
                        file_backup_path = _build_ingest_backup_path(file_path)
                        await asyncio.to_thread(
                            shutil.copy2, file_path, file_backup_path
                        )

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
                        result = _with_user_actionable_ingestion_message(
                            result,
                            embedding_model_id=request.embedding_model_id,
                        )
                        if result.status in {"error", "partial"}:
                            await _rollback_failed_cloud_ingestion(
                                db=db,
                                user=_user,
                                collection_name=safe_collection,
                                result=result,
                                file_path=file_path,
                                file_record=file_record,
                                collection_existed_before=effective_collection_existed_before,
                                uploaded_file_existed_before=uploaded_file_existed_before,
                                file_backup_path=file_backup_path,
                                had_existing_file=had_existing_file,
                                embedding_model_id=request.embedding_model_id,
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
                            collection_existed_before=effective_collection_existed_before,
                            uploaded_file_existed_before=uploaded_file_existed_before,
                            file_backup_path=file_backup_path,
                            had_existing_file=had_existing_file,
                            embedding_model_id=request.embedding_model_id,
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

    has_success = any(result.status == "success" for result in results)
    has_failure = any(result.status in {"error", "partial"} for result in results)

    if has_failure:
        await _restore_or_cleanup_collection_config_after_failed_ingest(
            snapshot=config_snapshot,
            collection_existed_before=collection_existed_before,
            collection_name=safe_collection,
            user=_user,
            context="ingest_cloud",
            successful_documents=1 if has_success else 0,
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

        try:
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
        except ValidationError as exc:
            errors = exc.errors()
            detail = errors[0]["msg"] if errors else "Invalid start_url"
            if isinstance(detail, str) and detail.startswith("Value error, "):
                detail = detail.removeprefix("Value error, ")
            raise HTTPException(status_code=422, detail=detail) from exc

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

        config_snapshot = await _save_collection_config_with_snapshot(
            collection=safe_collection,
            config_json=ingestion_config.model_dump_json(exclude_unset=True),
            user=_user,
            context="ingest_web",
        )
        # Track processed URLs to prevent duplicate UploadedFile records
        # Key: URL hash, Value: file_id
        # Note: For large-scale web ingestion (>10000 pages), consider using
        # a bounded-size dict (e.g., with maxitems) to control memory usage.
        _processed_urls: Dict[str, str] = {}
        _new_web_file_ids: set[str] = set()

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
                            is_admin=bool(_user.is_admin),
                            collection_name=collection_name,
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
                        is_admin=bool(_user.is_admin),
                        collection_name=collection_name,
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
                    result = _recreate_missing_existing_file(
                        existing_record=existing_record,
                        temp_file_path=temp_file_path,
                        db_session=db_session,
                        user_id=int(_user.id),
                        is_admin=bool(_user.is_admin),
                        collection_name=collection_name,
                        filename=filename,
                        url_hash=url_hash,
                        processed_urls=_processed_urls,
                    )
                    logger.info(
                        "Recreated missing persistent file for existing UploadedFile record: url=%s, file_id=%s",
                        url,
                        existing_record.file_id,
                    )
                    return result

                persistent_file = get_upload_path(
                    filename,
                    user_id=int(_user.id),
                    collection=collection_name,
                    collection_is_sanitized=True,
                )
                persistent_file.parent.mkdir(parents=True, exist_ok=True)

                return _create_new_web_file_handler_result(
                    temp_file_path=temp_file_path,
                    persistent_file=persistent_file,
                    db_session=db_session,
                    user_id=int(_user.id),
                    is_admin=bool(_user.is_admin),
                    collection_name=collection_name,
                    filename=filename,
                    url=url,
                    url_hash=url_hash,
                    processed_urls=_processed_urls,
                )

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
        web_updated_message = _build_user_actionable_ingestion_message(
            result.message,
            embedding_model_id=embedding_model_id,
        )
        if web_updated_message != result.message:
            result = result.model_copy(update={"message": web_updated_message})

        if result.status == "error":
            await _restore_or_cleanup_collection_config_after_failed_ingest(
                snapshot=config_snapshot,
                collection_existed_before=collection_existed_before,
                collection_name=safe_collection,
                user=_user,
                context="ingest_web",
                successful_documents=result.documents_created,
                side_effects_may_remain=result.side_effects_may_remain,
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
            await _restore_or_cleanup_collection_config_after_failed_ingest(
                snapshot=config_snapshot,
                collection_existed_before=collection_existed_before,
                collection_name=safe_collection,
                user=_user,
                context="ingest_web",
                successful_documents=result.documents_created,
                side_effects_may_remain=result.side_effects_may_remain,
            )

        return result

    except HTTPException:
        raise
    except (ValueError, KeyError, TypeError) as e:
        if "config_snapshot" in locals():
            await _restore_or_cleanup_collection_config_after_failed_ingest(
                snapshot=config_snapshot,
                collection_existed_before=collection_existed_before,
                collection_name=safe_collection,
                user=_user,
                context="ingest_web",
            )
        elif "collection_existed_before" in locals() and not collection_existed_before:
            await _cleanup_failed_new_collection_metadata(
                collection_name=safe_collection,
                user=_user,
            )
        logger.error("Data format error in web ingestion: %s", e)
        raise HTTPException(
            status_code=400, detail=f"Data format error: {str(e)}"
        ) from e
    except Exception as e:
        if "config_snapshot" in locals():
            await _restore_or_cleanup_collection_config_after_failed_ingest(
                snapshot=config_snapshot,
                collection_existed_before=collection_existed_before,
                collection_name=safe_collection,
                user=_user,
                context="ingest_web",
            )
        elif "collection_existed_before" in locals() and not collection_existed_before:
            await _cleanup_failed_new_collection_metadata(
                collection_name=safe_collection,
                user=_user,
            )
        logger.exception("Unexpected error in web ingestion: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Server internal error: {str(e)}",
        ) from e


@kb_router.post(
    "/ingest-web/jobs",
    response_model=BackgroundJobResponse,
    status_code=202,
)
@with_kb_user_scope
@handle_kb_exceptions
async def create_ingest_web_job(
    collection: str = Form(..., description="Target collection name"),
    start_url: str = Form(..., description="Starting URL for crawling"),
    max_pages: Optional[int] = Form(100),
    max_depth: Optional[int] = Form(3),
    url_patterns: Optional[str] = Form(None),
    exclude_patterns: Optional[str] = Form(None),
    same_domain_only: Optional[bool] = Form(True),
    content_selector: Optional[str] = Form(None),
    remove_selectors: Optional[str] = Form(None),
    concurrent_requests: Optional[int] = Form(3, ge=1, le=10),
    request_delay: Optional[float] = Form(1.0, ge=0),
    timeout: Optional[int] = Form(30, ge=1),
    respect_robots_txt: Optional[bool] = Form(True),
    parse_method: Optional[ParseMethod] = Form(None),
    chunk_strategy: Optional[ChunkStrategy] = Form(None),
    chunk_size: Optional[int] = Form(None, gt=0),
    chunk_overlap: Optional[int] = Form(None, ge=0),
    separators: Optional[str] = Form(None),
    embedding_model_id: str = Form("text-embedding-v4"),
    embedding_batch_size: Optional[int] = Form(None, gt=0),
    max_retries: Optional[int] = Form(None, ge=0),
    retry_delay: Optional[float] = Form(None, ge=0.0),
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    """Enqueue durable website ingestion into the knowledge base."""
    try:
        safe_collection = sanitize_path_component(collection, "collection")
    except ValueError as e:
        raise HTTPException(
            status_code=422, detail=f"Invalid collection name: {str(e)}"
        ) from e

    await _ensure_collection_access(safe_collection, _user, allow_create=True)

    url_patterns_list = (
        [p.strip() for p in url_patterns.split(",")] if url_patterns else None
    )
    exclude_patterns_list = (
        [p.strip() for p in exclude_patterns.split(",")] if exclude_patterns else None
    )
    remove_selectors_list = (
        [s.strip() for s in remove_selectors.split(",")] if remove_selectors else None
    )

    try:
        crawl_config = WebCrawlConfig(
            start_url=start_url,
            max_pages=max_pages or 100,
            max_depth=max_depth or 3,
            url_patterns=url_patterns_list,
            exclude_patterns=exclude_patterns_list,
            same_domain_only=same_domain_only if same_domain_only is not None else True,
            content_selector=content_selector,
            remove_selectors=remove_selectors_list,
            concurrent_requests=concurrent_requests or 3,
            request_delay=request_delay or 1.0,
            timeout=timeout or 30,
            respect_robots_txt=(
                respect_robots_txt if respect_robots_txt is not None else True
            ),
        )
    except ValidationError as exc:
        errors = exc.errors()
        detail = errors[0]["msg"] if errors else "Invalid start_url"
        if isinstance(detail, str) and detail.startswith("Value error, "):
            detail = detail.removeprefix("Value error, ")
        raise HTTPException(status_code=422, detail=detail) from exc
    await _ensure_background_job_queue_available_async()

    try:
        get_collection_sync(safe_collection)
        collection_existed_before = True
    except ValueError:
        collection_existed_before = False

    final_chunk_size = chunk_size if chunk_size is not None and chunk_size > 0 else 1000
    final_chunk_overlap = (
        chunk_overlap if chunk_overlap is not None and chunk_overlap >= 0 else 200
    )
    if final_chunk_overlap >= final_chunk_size:
        final_chunk_overlap = min(int(final_chunk_size * 0.2), final_chunk_size - 1)

    web_parsed_separators = _parse_separators(separators)
    web_final_strategy = (
        chunk_strategy if chunk_strategy is not None else ChunkStrategy.RECURSIVE
    )
    ingestion_config = IngestionConfig(
        parse_method=parse_method if parse_method is not None else ParseMethod.DEFAULT,
        chunk_strategy=web_final_strategy,
        chunk_size=final_chunk_size,
        chunk_overlap=final_chunk_overlap,
        separators=web_parsed_separators,
        embedding_model_id=embedding_model_id,
        embedding_batch_size=embedding_batch_size
        if embedding_batch_size is not None and embedding_batch_size > 0
        else 10,
        max_retries=max_retries if max_retries is not None and max_retries >= 0 else 3,
        retry_delay=retry_delay
        if retry_delay is not None and retry_delay >= 0
        else 1.0,
    )

    idempotency_key = _background_job_idempotency_key(
        "kb.ingest.web",
        {
            "collection": safe_collection,
            "crawl_config": crawl_config.model_dump(mode="json"),
            "ingestion_config": ingestion_config.model_dump(mode="json"),
            "user_id": int(_user.id),
        },
    )
    existing_job = get_non_terminal_background_job_by_idempotency_key(
        db,
        idempotency_key,
    )
    if existing_job is not None:
        return existing_job

    try:
        job = create_background_job(
            db,
            user_id=int(_user.id),
            job_type=BackgroundJobType.KB_INGEST_WEB,
            payload={
                "collection": safe_collection,
                "crawl_config": crawl_config.model_dump(mode="json"),
                "ingestion_config": ingestion_config.model_dump(mode="json"),
                "user_id": int(_user.id),
                "is_admin": bool(_user.is_admin),
                "collection_existed_before": collection_existed_before,
            },
            idempotency_key=idempotency_key,
            reuse_terminal_idempotency_key=False,
        )
        return await _enqueue_background_job_or_503_async(db, job)
    except Exception:
        raise


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


_CONFIG_ONLY_SENTINEL_KEY = "__config_only__"
_DeleteMode = Literal["full", "config_only"]
_CollectionMutationMode = Literal["tenant", "global"]


@dataclass
class CollectionMutationScope:
    """Resolved collection mutation boundary for tenant/global operations."""

    mode: _CollectionMutationMode
    collection_name: str
    requester_user_id: int
    is_admin: bool
    owner_user_ids: set[int]
    document_records: List[DocumentRecord]
    file_ids_by_owner: Dict[int, set[str]]


def _http_detail_to_str(detail: Any) -> str:
    """Normalize FastAPI/Starlette ``HTTPException.detail`` to a string."""
    if isinstance(detail, str):
        return detail
    try:
        return json.dumps(detail)
    except (TypeError, ValueError):
        return str(detail)


def _get_collection_document_counts(
    collection_name: str,
    user_id: int,
    is_admin: bool,
) -> tuple[int, int]:
    """Return total and caller-owned document counts for a collection."""
    if not collection_name or not collection_name.strip():
        raise HTTPException(status_code=422, detail="Collection name cannot be empty")
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

    return total_count, own_count


def _resolve_delete_mode_from_counts(
    total_count: int,
    own_count: int,
    is_admin: bool,
) -> _DeleteMode:
    """Decide full|config_only from precomputed total/own counts."""
    if is_admin:
        return "full"
    if total_count > 0 and own_count == 0:
        return "config_only"
    return "full"


def _get_collection_delete_mode(
    collection_name: str,
    user_id: int,
    is_admin: bool,
) -> _DeleteMode:
    """Return full|config_only for collection delete."""
    if not collection_name or not collection_name.strip():
        raise HTTPException(status_code=422, detail="Collection name cannot be empty")
    if is_admin:
        return "full"

    total_count, own_count = _get_collection_document_counts(
        collection_name, user_id, is_admin=False
    )
    return _resolve_delete_mode_from_counts(total_count, own_count, is_admin=False)


def _get_document_record_owner_id(
    record: DocumentRecord,
    *,
    fallback_user_id: int,
) -> int:
    """Return the owner for storage cleanup, falling back for legacy projections."""
    raw_user_id = getattr(record, "user_id", None)
    if raw_user_id is None:
        return fallback_user_id
    try:
        return int(raw_user_id)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid user_id on document record %s: %r; falling back to user_%s",
            getattr(record, "doc_id", "<unknown>"),
            raw_user_id,
            fallback_user_id,
        )
        return fallback_user_id


def _group_document_file_ids_by_owner(
    records: List[DocumentRecord],
    *,
    fallback_user_id: int,
) -> Dict[int, set[str]]:
    """Group uploaded file ids by document owner for tenant storage cleanup."""
    grouped: Dict[int, set[str]] = {}
    for record in records:
        owner_id = _get_document_record_owner_id(
            record, fallback_user_id=fallback_user_id
        )
        grouped.setdefault(owner_id, set())
        file_id = _get_document_record_file_id(record)
        if file_id:
            grouped[owner_id].add(file_id)
    return grouped


def _get_collection_storage_owner_ids(
    records: List[DocumentRecord],
    *,
    fallback_user_id: int,
) -> set[int]:
    """Return owners whose physical KB storage may need collection-level cleanup."""
    if not records:
        return {fallback_user_id}
    return {
        _get_document_record_owner_id(record, fallback_user_id=fallback_user_id)
        for record in records
    }


def _resolve_collection_mutation_scope(
    *,
    collection_name: str,
    requester_user_id: int,
    is_admin: bool,
    db: Session,
    vector_store: Any = None,
) -> CollectionMutationScope:
    """Resolve tenant/global mutation ownership once for delete and rename."""
    if vector_store is None:
        vector_store = get_vector_index_store()
    document_records = vector_store.list_document_records(
        collection_name=collection_name,
        user_id=requester_user_id,
        is_admin=is_admin,
    )
    file_ids_by_owner = _group_document_file_ids_by_owner(
        document_records,
        fallback_user_id=requester_user_id,
    )

    if not is_admin:
        owner_user_ids = {requester_user_id}
        file_ids_by_owner.setdefault(requester_user_id, set())
        return CollectionMutationScope(
            mode="tenant",
            collection_name=collection_name,
            requester_user_id=requester_user_id,
            is_admin=False,
            owner_user_ids=owner_user_ids,
            document_records=document_records,
            file_ids_by_owner=file_ids_by_owner,
        )

    owner_user_ids = (
        {
            _get_document_record_owner_id(record, fallback_user_id=requester_user_id)
            for record in document_records
        }
        if document_records
        else set()
    )
    owner_user_ids.update(
        get_metadata_store().list_collection_config_owner_ids(collection_name)
    )
    owner_user_ids.update(
        list_collection_uploaded_file_owner_ids(db, collection_name=collection_name)
    )
    if not owner_user_ids:
        owner_user_ids = {requester_user_id}
    for owner_id in owner_user_ids:
        file_ids_by_owner.setdefault(owner_id, set())

    return CollectionMutationScope(
        mode="global",
        collection_name=collection_name,
        requester_user_id=requester_user_id,
        is_admin=True,
        owner_user_ids=owner_user_ids,
        document_records=document_records,
        file_ids_by_owner=file_ids_by_owner,
    )


def _remove_user_collection_config(
    collection_name: str,
    user_id: int,
    *,
    delete_orphaned_metadata: bool = False,
) -> dict[str, int]:
    """Remove only the caller's collection config entry.

    Returns the number of rows removed so the caller can distinguish a real
    stale-list cleanup from a delete request for a collection the user never
    had in their KB list.
    """
    try:
        cleanup_counts = delete_collection_metadata_sync(
            collection_name=collection_name,
            user_id=user_id,
            is_admin=False,
            delete_orphaned_metadata=delete_orphaned_metadata,
        )
    except Exception as exc:
        logger.exception(
            "Failed to delete collection config for %s/user_%s",
            collection_name,
            user_id,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete collection configuration: {exc}",
        ) from exc

    return cleanup_counts


def _cleanup_collection_config_if_no_owned_documents(
    collection_name: str,
    user_id: int,
) -> dict[str, int]:
    """Clear user's config once they no longer own documents in the collection."""
    total_count, own_count = _get_collection_document_counts(
        collection_name, user_id, is_admin=False
    )
    if own_count > 0:
        return {}

    try:
        cleanup_counts = delete_collection_metadata_sync(
            collection_name=collection_name,
            user_id=user_id,
            is_admin=False,
            delete_orphaned_metadata=total_count == 0,
        )
    except Exception as exc:
        logger.exception(
            "Failed to delete collection config after document deletion for %s/user_%s",
            collection_name,
            user_id,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete collection configuration: {exc}",
        ) from exc
    if int(cleanup_counts.get("config_rows", 0)) <= 0:
        return {}

    logger.info(
        "Removed collection config for %s/user_%s after last owned document deletion: %s",
        collection_name,
        user_id,
        cleanup_counts,
    )
    return cleanup_counts


def _is_config_only_delete_result(result: CollectionOperationResult) -> bool:
    """Return True when collection delete only removed user visibility config."""
    return bool((result.deleted_counts or {}).get(_CONFIG_ONLY_SENTINEL_KEY))


def _strip_config_only_sentinel(
    result: CollectionOperationResult,
) -> CollectionOperationResult:
    """Return ``result`` without the internal config-only sentinel key."""
    deleted_counts = dict(result.deleted_counts or {})
    if _CONFIG_ONLY_SENTINEL_KEY not in deleted_counts:
        return result
    deleted_counts.pop(_CONFIG_ONLY_SENTINEL_KEY, None)
    return CollectionOperationResult(
        status=result.status,
        collection=result.collection,
        message=result.message,
        warnings=list(result.warnings or []),
        affected_documents=list(result.affected_documents or []),
        deleted_counts=deleted_counts,
    )


def _validate_and_prefetch_batch_delete_counts(
    unique_names: List[str],
    user_id: int,
    is_admin: bool,
) -> tuple[
    List[str],
    List[BatchDeleteFailureItem],
    Dict[str, tuple[int, int]],
]:
    """Validate batch delete names and prefetch count hints.

    Returns ``(valid_names, failed_items, counts_by_name)``. For non-admin
    callers ``counts_by_name`` maps the trimmed collection name to its
    ``(total, own)`` document counts so callers can reuse them when invoking
    ``_perform_kb_collection_delete`` (avoids redundant LanceDB scans).
    Admin callers receive an empty dict because counts are not needed.
    """
    failed: List[BatchDeleteFailureItem] = []
    allowed: List[str] = []
    counts_by_name: Dict[str, tuple[int, int]] = {}

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
        return allowed, failed, counts_by_name

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
        return [], failed, counts_by_name

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
            "Batch delete count prefetch failed (vector store grouped counts): %s",
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to scan documents table for batch delete.",
        ) from exc

    for name in non_empty:
        key = str(name).strip()
        total = int(totals.get(key, 0))
        own = int(owns.get(key, 0))
        allowed.append(name)
        counts_by_name[key] = (total, own)

    return allowed, failed, counts_by_name


def _build_config_only_delete_result(
    safe_collection: str,
    cleanup_counts: dict[str, int],
) -> CollectionOperationResult:
    """Construct a CollectionOperationResult for config-only delete paths."""
    removed_rows = int(cleanup_counts.get("config_rows", 0))
    if removed_rows > 0:
        message = (
            f"Removed collection '{safe_collection}' from your knowledge base list."
        )
    else:
        message = f"Collection '{safe_collection}' is not in your knowledge base list."
    return CollectionOperationResult(
        status="success",
        collection=safe_collection,
        message=message,
        deleted_counts={**cleanup_counts, _CONFIG_ONLY_SENTINEL_KEY: 1},
    )


def _perform_config_only_collection_delete(
    safe_collection: str,
    user_id: int,
) -> CollectionOperationResult:
    """Remove a stale user KB-list entry without touching collection documents."""
    cleanup_counts = _remove_user_collection_config(safe_collection, user_id)
    if int(cleanup_counts.get("config_rows", 0)) <= 0:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{safe_collection}' is not in your knowledge base list.",
        )
    return _build_config_only_delete_result(safe_collection, cleanup_counts)


def _perform_kb_collection_delete(
    collection_name: str,
    user_id: int,
    is_admin: bool,
    db: Session,
    *,
    preflight_counts: Optional[tuple[int, int]] = None,
) -> CollectionOperationResult:
    """Delete one KB collection (same pipeline as single-delete API).

    ``preflight_counts`` is an optional ``(total, own)`` pair already computed
    by an upstream batch preflight. It is used only as a stale-preflight hint;
    non-admin callers always get a live delete-mode recheck before mutation.
    """
    try:
        try:
            safe_collection = sanitize_path_component(collection_name, "collection")
        except ValueError as e:
            raise HTTPException(
                status_code=422, detail=f"Invalid collection name: {str(e)}"
            ) from e

        preflight_delete_mode: Optional[_DeleteMode] = None
        if preflight_counts is not None:
            total_count, own_count = preflight_counts
            preflight_delete_mode = _resolve_delete_mode_from_counts(
                int(total_count), int(own_count), is_admin
            )
        elif not is_admin:
            preflight_delete_mode = _get_collection_delete_mode(
                safe_collection, user_id, is_admin=False
            )

        if is_admin:
            delete_mode = "full"
        else:
            delete_mode = _get_collection_delete_mode(
                safe_collection, user_id, is_admin=False
            )
            if (
                preflight_delete_mode is not None
                and preflight_delete_mode != delete_mode
            ):
                logger.info(
                    "Collection delete mode changed after preflight for %s/user_%s: "
                    "preflight=%s live=%s",
                    safe_collection,
                    user_id,
                    preflight_delete_mode,
                    delete_mode,
                )

        if delete_mode == "config_only":
            return _perform_config_only_collection_delete(safe_collection, user_id)

        mutation_scope = _resolve_collection_mutation_scope(
            collection_name=safe_collection,
            requester_user_id=user_id,
            is_admin=is_admin,
            db=db,
        )
        for owner_id in sorted(mutation_scope.owner_user_ids):
            tombstone_kb_ingest_targets_for_collection(
                db,
                user_id=owner_id,
                collection=safe_collection,
            )

        result = delete_collection(safe_collection, user_id, is_admin)

        physical_cleanup_by_owner = {}
        for owner_id in sorted(mutation_scope.owner_user_ids):
            physical_cleanup_by_owner[owner_id] = delete_collection_physical_dir(
                user_id=owner_id,
                collection_name=safe_collection,
            )

        if result.status == "error":
            cleanup_warnings = list(result.warnings) if result.warnings else []
            for owner_id, physical_cleanup in physical_cleanup_by_owner.items():
                collection_dir = physical_cleanup.collection_dir or get_upload_path(
                    "", user_id=owner_id, collection=safe_collection
                )
                physical_cleanup_status = physical_cleanup.status
                if physical_cleanup_status == "success":
                    cleanup_warnings.append(
                        f"Physical directory moved to trash for user_{owner_id}: "
                        f"{collection_dir} "
                        "(trash cleanup requires external scheduler/cron)"
                    )
                elif physical_cleanup_status == "not_found":
                    cleanup_warnings.append(
                        f"Physical directory cleanup for user_{owner_id}: "
                        "No physical directory found (collection had no files)"
                    )
                elif physical_cleanup.error:
                    cleanup_warnings.append(
                        f"Physical directory cleanup for user_{owner_id}: "
                        f"{physical_cleanup.status} - {physical_cleanup.error}"
                    )

            return CollectionOperationResult(
                status="error",
                collection=safe_collection,
                message=result.message,
                warnings=cleanup_warnings,
                affected_documents=result.affected_documents,
                deleted_counts=result.deleted_counts,
            )

        remaining_records = get_vector_index_store().list_document_records(
            collection_name=None,
            user_id=user_id,
            is_admin=is_admin,
        )
        remaining_file_ids_by_owner = _group_document_file_ids_by_owner(
            remaining_records,
            fallback_user_id=user_id,
        )
        deleted_uploaded_files = 0
        for owner_id in sorted(mutation_scope.owner_user_ids):
            physical_cleanup = physical_cleanup_by_owner[owner_id]
            physical_cleanup_status = physical_cleanup.status
            collection_dir = physical_cleanup.collection_dir or get_upload_path(
                "", user_id=owner_id, collection=safe_collection
            )
            if physical_cleanup_status in {"success", "not_found"}:
                deleted_uploaded_files += delete_collection_uploaded_files(
                    db,
                    user_id=owner_id,
                    collection_file_ids=mutation_scope.file_ids_by_owner.get(
                        owner_id, set()
                    ),
                    remaining_file_ids=remaining_file_ids_by_owner.get(owner_id, set()),
                    collection_dir=collection_dir,
                )
            else:
                logger.warning(
                    "Preserving UploadedFile records for collection %s/user_%s "
                    "because physical cleanup status is %s",
                    safe_collection,
                    owner_id,
                    physical_cleanup_status,
                )
        if deleted_uploaded_files:
            logger.info(
                "Deleted %s UploadedFile record(s) for collection %s",
                deleted_uploaded_files,
                safe_collection,
            )

        cleanup_warnings = list(result.warnings) if result.warnings else []
        cleanup_info_messages: List[str] = []
        has_physical_cleanup_issue = False

        for owner_id, physical_cleanup in physical_cleanup_by_owner.items():
            physical_cleanup_status = physical_cleanup.status
            physical_cleanup_error = physical_cleanup.error
            collection_dir = physical_cleanup.collection_dir or get_upload_path(
                "", user_id=owner_id, collection=safe_collection
            )

            if physical_cleanup_status == "success":
                cleanup_info = (
                    f"Physical directory moved to trash for user_{owner_id}: "
                    f"{collection_dir} "
                    "(trash cleanup requires external scheduler/cron)"
                )
                cleanup_warnings.append(cleanup_info)
                cleanup_info_messages.append(cleanup_info)
            elif physical_cleanup_status == "not_found":
                cleanup_info = (
                    f"Physical directory cleanup for user_{owner_id}: "
                    "No physical directory found (collection had no files)"
                )
                cleanup_warnings.append(cleanup_info)
                cleanup_info_messages.append(cleanup_info)
            elif physical_cleanup_status == "error" and physical_cleanup_error:
                has_physical_cleanup_issue = True
                cleanup_info = (
                    f"Physical directory cleanup for user_{owner_id}: Warning - "
                    f"{physical_cleanup_error}. Database deletion proceeded, but "
                    "physical file cleanup status is uncertain."
                )
                cleanup_warnings.append(cleanup_info)
                cleanup_info_messages.append(cleanup_info)
            elif physical_cleanup_status == "failed" and physical_cleanup_error:
                has_physical_cleanup_issue = True
                cleanup_info = (
                    f"Physical directory cleanup for user_{owner_id}: Failed - "
                    f"{physical_cleanup_error}"
                )
                cleanup_warnings.append(cleanup_info)
                cleanup_info_messages.append(cleanup_info)

        cleanup_info_message = ""
        if cleanup_info_messages:
            cleanup_info_message = f" {'; '.join(cleanup_info_messages)}."

        final_status = result.status
        if result.status == "success" and has_physical_cleanup_issue:
            final_status = "partial_success"
            if not cleanup_info_message:
                cleanup_info_message = " Database deletion succeeded, but physical file cleanup encountered issues."

        updated_message = result.message
        if cleanup_info_message:
            updated_message = f"{result.message}{cleanup_info_message}"

        updated_result = CollectionOperationResult(
            status=final_status,
            collection=safe_collection,
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
    return _strip_config_only_sentinel(result)


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

    allowed, failed, counts_by_name = _validate_and_prefetch_batch_delete_counts(
        unique_names, user_id, is_admin
    )

    try:
        for name in allowed:
            try:
                preflight_counts = counts_by_name.get(str(name).strip())
                result = _perform_kb_collection_delete(
                    name,
                    user_id,
                    is_admin,
                    db,
                    preflight_counts=preflight_counts,
                )
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
    config_cleanup_counts: dict[str, int] = {}
    config_cleanup_error: Optional[str] = None

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
                if cleanup_file_id not in remaining_file_ids:
                    cleanup_record = (
                        db.query(UploadedFile)
                        .filter(
                            UploadedFile.user_id == user_id_int,
                            UploadedFile.file_id == cleanup_file_id,
                        )
                        .first()
                    )
                    if cleanup_record is not None:
                        tombstone_kb_ingest_target(
                            db,
                            user_id=user_id_int,
                            collection=safe_collection_name,
                            target_path=str(cleanup_record.storage_path),
                            file_id=str(cleanup_record.file_id),
                            commit=False,
                        )
                _delete_uploaded_file_if_orphaned(
                    db,
                    file_id=cleanup_file_id,
                    user_id=user_id_int,
                    remaining_file_ids=remaining_file_ids,
                )

    if deleted_doc_ids:
        try:
            config_cleanup_counts = _cleanup_collection_config_if_no_owned_documents(
                safe_collection_name,
                user_id_int,
            )
        except HTTPException as exc:
            config_cleanup_error = _http_detail_to_str(exc.detail)
            logger.warning(
                "Failed to clean collection config after deleting document(s) "
                "(collection=%s, user_id=%s): %s",
                safe_collection_name,
                user_id_int,
                exc.detail,
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
        partial_response: dict[str, Any] = {
            "status": "partial_success" if deleted_doc_ids else "failed",
            "message": f"Deleted {len(deleted_doc_ids)} of {len(matching_docs)} documents",
            "collection": safe_collection_name,
            "filename": filename,
            "deleted_doc_ids": deleted_doc_ids,
            "errors": deletion_errors,
        }
        if config_cleanup_counts:
            partial_response["collection_config_cleanup"] = config_cleanup_counts
        if config_cleanup_error:
            partial_response["collection_config_cleanup_error"] = config_cleanup_error
        return partial_response

    response: dict[str, Any] = {
        "status": "success",
        "message": f"Successfully deleted {len(deleted_doc_ids)} document(s)",
        "collection": safe_collection_name,
        "filename": filename,
        "deleted_doc_ids": deleted_doc_ids,
    }
    if config_cleanup_counts:
        response["collection_config_cleanup"] = config_cleanup_counts
    if config_cleanup_error:
        response["collection_config_cleanup_error"] = config_cleanup_error
    return response


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

    mutation_scope = _resolve_collection_mutation_scope(
        collection_name=safe_old_collection,
        requester_user_id=int(_user.id),
        is_admin=bool(_user.is_admin),
        db=db,
        vector_store=vector_store,
    )

    for owner_id in sorted(mutation_scope.owner_user_ids):
        old_dir = get_upload_path(
            "",
            user_id=owner_id,
            collection=safe_old_collection,
            create_if_not_exists=False,
            collection_is_sanitized=True,
        )
        new_dir = get_upload_path(
            "",
            user_id=owner_id,
            collection=safe_new_collection,
            create_if_not_exists=False,
            collection_is_sanitized=True,
        )
        if old_dir.exists() and old_dir.is_dir() and new_dir.exists():
            raise HTTPException(
                status_code=409,
                detail=(
                    "Failed to rename collection: target physical directory already "
                    f"exists for user_{owner_id}. A collection named "
                    f"'{safe_new_collection}' already has physical files."
                ),
            )

    physical_rename_results = {}
    renamed_owner_ids: list[int] = []
    for owner_id in sorted(mutation_scope.owner_user_ids):
        physical_rename = rename_collection_storage(
            db,
            user_id=owner_id,
            old_collection_name=safe_old_collection,
            new_collection_name=safe_new_collection,
            collection_file_ids=mutation_scope.file_ids_by_owner.get(owner_id, set()),
        )
        physical_rename_results[owner_id] = physical_rename
        if physical_rename.status == "success":
            renamed_owner_ids.append(owner_id)
        if physical_rename.status != "failed":
            continue

        for rollback_owner_id in reversed(renamed_owner_ids):
            rollback = rename_collection_storage(
                db,
                user_id=rollback_owner_id,
                old_collection_name=safe_new_collection,
                new_collection_name=safe_old_collection,
                collection_file_ids=mutation_scope.file_ids_by_owner.get(
                    rollback_owner_id, set()
                ),
            )
            if rollback.status != "success":
                logger.error(
                    "Failed to roll back physical collection rename for user_%s "
                    "%s -> %s: %s",
                    rollback_owner_id,
                    safe_new_collection,
                    safe_old_collection,
                    rollback.error,
                )

        physical_rename_error = physical_rename.error
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
    warnings.extend(
        vector_store.rename_collection_data(
            collection_name=safe_old_collection,
            new_name=safe_new_collection,
            user_id=int(_user.id),
            is_admin=bool(_user.is_admin),
        )
    )

    try:
        metadata_store = get_metadata_store()
        await metadata_store.rename_collection(
            old_name=safe_old_collection,
            new_name=safe_new_collection,
            user_id=int(_user.id),
            is_admin=bool(_user.is_admin),
        )
    except Exception as e:
        logger.warning("Failed to rename metadata store keys: %s", e)
        warnings.append(f"Failed to rename collection metadata: {e}")

    # Best-effort scoped ingestion status rename. The status store owns how
    # collection/user filters map to its backend tables.
    try:
        warnings.extend(
            get_ingestion_status_store().rename_collection_status(
                old_name=safe_old_collection,
                new_name=safe_new_collection,
                user_id=int(_user.id),
                is_admin=bool(_user.is_admin),
            )
        )
    except Exception as e:
        logger.warning("Failed to update ingestion status: %s", e)
        warnings.append(f"Failed to update ingestion status: {e}")

    # Step 3: Add physical rename status to warnings and message for visibility
    rename_info_messages: list[str] = []
    has_physical_rename_issue = False
    for owner_id, physical_rename in physical_rename_results.items():
        physical_rename_status = physical_rename.status
        physical_rename_error = physical_rename.error
        if (
            physical_rename_status == "success"
            and physical_rename.old_collection_dir is not None
            and physical_rename.new_collection_dir is not None
        ):
            rename_info = (
                f"Physical directory renamed for user_{owner_id}: "
                f"{physical_rename.old_collection_dir.name} -> "
                f"{physical_rename.new_collection_dir.name}"
            )
            warnings.append(rename_info)
            rename_info_messages.append(rename_info)
        elif physical_rename_status == "not_found":
            rename_info = (
                f"Physical directory rename for user_{owner_id}: "
                "No physical directory found (collection had no files)"
            )
            warnings.append(rename_info)
            rename_info_messages.append(rename_info)
        elif physical_rename_status == "error" and physical_rename_error:
            has_physical_rename_issue = True
            rename_info = (
                f"Physical directory rename for user_{owner_id}: Warning - "
                f"{physical_rename_error}. Database rename proceeded, but physical "
                "directory rename status is uncertain."
            )
            warnings.append(rename_info)
            rename_info_messages.append(rename_info)
        elif physical_rename_status == "failed" and physical_rename_error:
            has_physical_rename_issue = True
            rename_info = (
                f"Physical directory rename for user_{owner_id}: Failed - "
                f"{physical_rename_error}"
            )
            warnings.append(rename_info)
            rename_info_messages.append(rename_info)

    rename_info_message = ""
    if rename_info_messages:
        rename_info_message = f" {'; '.join(rename_info_messages)}."

    # Step 4: Determine final status
    final_status = "success" if not warnings else "partial_success"
    if has_physical_rename_issue:
        final_status = "partial_success"
        if not rename_info_message:
            rename_info_message = " Database rename succeeded, but physical directory rename encountered issues."

    # Step 5: Build final message
    base_message = (
        f"Collection renamed from '{safe_old_collection}' to '{safe_new_collection}'"
    )
    if warnings:
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
