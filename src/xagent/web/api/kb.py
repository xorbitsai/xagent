"""Knowledge base API route handlers"""

import asyncio
import concurrent.futures
import functools
import hashlib
import json
import logging
import mimetypes
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast

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
from ...core.tools.core.RAG_tools.management.collections import (
    delete_collection,
    list_collections,
    list_documents,
)
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
from ...core.tools.core.RAG_tools.storage.factory import get_vector_index_store
from ...core.tools.core.RAG_tools.utils.string_utils import (
    generate_deterministic_doc_id,
)
from ..auth_dependencies import get_current_user
from ..config import (
    MAX_FILE_SIZE,
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


def _like_contains_pattern(value: str) -> str:
    escaped = (
        value.replace(_SQL_LIKE_ESCAPE, _SQL_LIKE_ESCAPE * 2)
        .replace("%", f"{_SQL_LIKE_ESCAPE}%")
        .replace("_", f"{_SQL_LIKE_ESCAPE}_")
    )
    return f"%{escaped}%"


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
        logger.error(f"Failed to save collection config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@kb_router.post(
    "/ingest",
    response_model=IngestionResult,
)
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

    if not collection or not collection.strip():
        collection = Path(safe_filename).stem
        logger.info("Using file name as collection: %s", collection)

    try:
        # SECURITY: Validate collection name at API boundary
        safe_collection = sanitize_path_component(collection, "collection")
        collection = safe_collection

        file_path = get_upload_path(
            safe_filename,
            user_id=int(_user.id),
            collection=safe_collection,
            collection_is_sanitized=True,
        )
    except ValueError as e:
        logger.warning("Invalid collection name rejected: %s - %s", collection, e)
        raise HTTPException(
            status_code=422, detail=f"Invalid collection name: {str(e)}"
        ) from e

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
                        status_code=422,
                        detail=(
                            "File size exceeds maximum limit of "
                            f"{MAX_FILE_SIZE // (1024 * 1024)}MB"
                        ),
                    )
                buffer.write(chunk)
        logger.info(
            "File uploaded: %s -> %s (user: %s, collection: %s)",
            safe_filename,
            file_path,
            _user.id,
            collection,
        )
    except HTTPException:
        # Ensure partial file is removed on early abort (e.g., file too large)
        try:
            if file_path.exists():
                file_path.unlink()
        except OSError:
            pass
        raise

    # Register file in unified file management (file_id) for KB + file APIs.
    mime_type = (
        getattr(file, "content_type", None)
        or mimetypes.guess_type(safe_filename)[0]
        or "application/octet-stream"
    )
    file_record = _upsert_uploaded_file_record(
        db,
        user_id=int(_user.id),
        filename=safe_filename,
        storage_path=file_path,
        mime_type=mime_type,
        file_size=int(total_size),
    )

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

    config = IngestionConfig(
        parse_method=parse_method if parse_method is not None else ParseMethod.DEFAULT,
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

    def _run_ingestion() -> IngestionResult:
        return run_document_ingestion(
            collection=collection,
            source_path=str(file_path),
            ingestion_config=config,
            progress_manager=progress_manager,
            user_id=int(_user.id),
            is_admin=bool(_user.is_admin),
            file_id=str(file_record.file_id),
        )

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(_run_ingestion)
        result: IngestionResult = future.result()

    if result.status == "error":
        return JSONResponse(status_code=500, content=result.model_dump())
    if result.status == "partial":
        logger.warning(
            "KB ingest partially completed (collection=%s, filename=%s, user_id=%s): %s",
            collection,
            safe_filename,
            _user.id,
            result.message,
        )

    return JSONResponse(
        status_code=200,
        content={**result.model_dump(), "file_id": file_record.file_id},
    )


@kb_router.post("/ingest-cloud", response_model=List[IngestionResult])
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

    # Concurrency limit for cloud ingestion to avoid overloading
    semaphore = asyncio.Semaphore(5)

    async def process_file(file_info: CloudFile) -> IngestionResult:
        async with semaphore:
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
                    safe_filename = Path(file_info.fileName).name
                    file_path = get_upload_path(safe_filename, user_id=int(_user.id))

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
                        return IngestionResult(
                            status="error",
                            message=f"Download failed: {str(e)}",
                            doc_id=file_info.fileName,
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
                        result = await asyncio.to_thread(
                            run_document_ingestion,
                            collection=safe_collection,
                            source_path=str(file_path),
                            ingestion_config=config,
                            progress_manager=progress_manager,
                            user_id=int(_user.id),
                            is_admin=bool(_user.is_admin),
                            file_id=str(file_record.file_id),
                        )
                        return result
                    except Exception as e:
                        # Clean up the file record and physical file on failure
                        try:
                            db.delete(file_record)
                            db.commit()
                        except Exception:
                            db.rollback()
                        try:
                            if file_path.exists():
                                file_path.unlink()
                        except OSError:
                            pass
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

            except Exception as e:
                logger.exception(
                    f"Unexpected error ingesting {file_info.fileName}: {e}"
                )
                return IngestionResult(
                    status="error",
                    message=f"Unexpected error: {str(e)}",
                    doc_id=file_info.fileName,
                )

    # Run all file processings concurrently
    results = await asyncio.gather(*[process_file(f) for f in request.files])

    return results


@kb_router.get(
    "/collections",
    response_model=ListCollectionsResult,
)
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
            safe_title = (
                sanitize_path_component(title, "filename") if title else "untitled"
            )
            filename = f"{url_hash}_{safe_title}.md"

            # Check if we've already processed this URL (in-memory cache)
            if url_hash in _processed_urls:
                existing_file_id = _processed_urls[url_hash]
                logger.info(
                    f"Reusing existing UploadedFile record for web ingestion: "
                    f"url={url}, file_id={existing_file_id}"
                )
                # Query the database to get the storage path
                existing_record = (
                    db_session.query(UploadedFile)
                    .filter(UploadedFile.file_id == existing_file_id)
                    .first()
                )
                if existing_record:
                    return FileHandlerResult(
                        file_path=str(existing_record.storage_path),
                        file_id=str(existing_record.file_id),
                    )
                else:
                    # Cached file_id was deleted from DB, fall through to recreate
                    logger.warning(
                        f"Cached file_id {existing_file_id} not found in DB (record was deleted), "
                        f"will create new record for url={url}"
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
                logger.info(
                    f"Found existing UploadedFile record from previous session: "
                    f"url={url}, file_id={existing_record.file_id}"
                )
                _processed_urls[url_hash] = str(existing_record.file_id)
                return FileHandlerResult(
                    file_path=str(existing_record.storage_path),
                    file_id=str(existing_record.file_id),
                )

            # Generate persistent file path
            persistent_file = get_upload_path(
                filename,
                user_id=int(_user.id),
                collection=collection_name,
                collection_is_sanitized=True,
            )

            # Ensure directory exists
            persistent_file.parent.mkdir(parents=True, exist_ok=True)

            try:
                # Copy file to persistent location
                shutil.copy2(temp_file_path, persistent_file)
                logger.info(
                    f"Copied web ingestion file from {temp_file_path} to {persistent_file}"
                )

                # Create UploadedFile record
                file_record = _upsert_uploaded_file_record(
                    db_session,
                    user_id=int(_user.id),
                    filename=filename,
                    storage_path=persistent_file,
                    mime_type="text/markdown",
                    file_size=persistent_file.stat().st_size,
                )

                logger.info(
                    f"Created UploadedFile record for web ingestion: file_id={file_record.file_id}, "
                    f"filename={filename}, url={url}"
                )

                # Track this URL to prevent duplicates
                _processed_urls[url_hash] = str(file_record.file_id)

                return FileHandlerResult(
                    file_path=str(persistent_file),
                    file_id=str(file_record.file_id),
                )
            except Exception:
                # Clean up orphaned persistent file if upsert failed
                if persistent_file.exists():
                    try:
                        persistent_file.unlink()
                        logger.warning(
                            f"Cleaned up orphaned persistent file due to upsert failure: {persistent_file}"
                        )
                    except Exception as cleanup_error:
                        logger.warning(
                            f"Failed to clean up orphaned persistent file {persistent_file}: {cleanup_error}"
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
        logger.error("Data format error in web ingestion: %s", e)
        raise HTTPException(
            status_code=400, detail=f"Data format error: {str(e)}"
        ) from e
    except Exception as e:
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

        physical_cleanup = delete_collection_physical_dir(
            user_id=user_id,
            collection_name=safe_collection,
        )
        physical_cleanup_status = physical_cleanup.status
        physical_cleanup_error = physical_cleanup.error
        collection_dir = physical_cleanup.collection_dir
        if physical_cleanup_status == "failed":
            if (
                physical_cleanup_error
                == "Another operation is in progress; please try again later."
            ):
                raise HTTPException(status_code=409, detail=physical_cleanup_error)
            raise HTTPException(
                status_code=500,
                detail=(
                    "Failed to delete collection: cannot move physical files. "
                    f"Error: {physical_cleanup_error}. "
                    "Please ensure the directory is not in use and you have proper permissions."
                ),
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
        deleted_uploaded_files = delete_collection_uploaded_files(
            db,
            user_id=user_id,
            collection_file_ids=collection_file_ids,
            remaining_file_ids=remaining_file_ids,
            collection_dir=collection_dir,
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
    except Exception as e:
        logger.exception("Failed to delete collection '%s': %s", collection_name, e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete collection: {str(e)}",
        ) from e


@kb_router.delete(
    "/collections/{collection_name}",
)
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
    return _perform_kb_collection_delete(
        collection_name,
        int(_user.id),
        bool(_user.is_admin),
        db,
    )


@kb_router.post(
    "/collections/batch-delete",
    response_model=BatchDeleteCollectionsResponse,
)
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
        try:
            safe_collection_name = sanitize_path_component(
                collection_name, "collection"
            )
        except ValueError as e:
            raise HTTPException(
                status_code=422, detail=f"Invalid collection name: {str(e)}"
            ) from e

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

        # Use storage abstraction layer to fetch document records
        vector_store = get_vector_index_store()
        records = vector_store.list_document_records(
            collection_name=safe_collection_name,
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

    try:
        safe_collection_name = sanitize_path_component(collection_name, "collection")
    except ValueError as e:
        raise HTTPException(
            status_code=422, detail=f"Invalid collection name: {str(e)}"
        ) from e

    def _collect_candidate_doc_ids(
        docs: list[dict[str, Any]],
    ) -> list[str]:
        candidate: set[str] = set()
        for item in docs:
            raw = item.get("doc_id")
            if isinstance(raw, str) and raw:
                candidate.add(raw)
        return sorted(candidate)

    def _append_matching_uploaded_file_candidate(rec: UploadedFile) -> bool:
        file_id_str = str(getattr(rec, "file_id", "")).strip()
        if not file_id_str:
            return False
        storage_path = str(getattr(rec, "storage_path", "")).strip()
        if not storage_path:
            return False
        derived_doc_id = generate_deterministic_doc_id(
            safe_collection_name, storage_path
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
            }
        )
        return True

    def _resolve_cleanup_file_id(doc_info: dict[str, Any]) -> Optional[str]:
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
    ) -> dict[str, Any]:
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
    ) -> Optional[dict[str, Any]]:
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

    def _resolve_list_documents_match() -> Optional[dict[str, Any]]:
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

    user_id_int = int(_user.id)
    records: List[Dict[str, Any]] = []
    try:
        records = _list_documents_for_user(
            user_id=user_id_int,
            is_admin=bool(_user.is_admin),
            collection_name=safe_collection_name,
        )
    except Exception as exc:
        # Degrade gracefully when LanceDB cannot decode legacy rows.
        logger.warning(
            "Failed to read documents for delete resolution (collection=%s): %s",
            safe_collection_name,
            exc,
        )
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
    matching_docs = []
    for record in records:
        current_doc_id = record.get("doc_id")
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
                "source_path": record.get("source_path"),
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
                source_path = getattr(summary, "source_path", None)
                fallback_basename: str | None = None
                if isinstance(source_path, str) and source_path.strip():
                    fallback_basename = Path(source_path).name
                if doc_id_value == filename or fallback_basename == filename:
                    matching_docs.append(
                        {
                            "doc_id": doc_id_value,
                            "file_id": None,
                            "filename": filename,
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

    remaining_file_ids: set[str] = set()
    try:
        remaining_records = _list_documents_for_user(
            user_id=user_id_int,
            is_admin=bool(_user.is_admin),
            collection_name=safe_collection_name,
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
            "Failed to read remaining docs for orphan cleanup; fallback to UploadedFile set: %s",
            exc,
        )
        remaining_file_ids = {
            str(rec.file_id)
            for rec in db.query(UploadedFile)
            .filter(UploadedFile.user_id == user_id_int)
            .all()
            if getattr(rec, "file_id", None)
        }

    for doc_info in matching_docs:
        resolved_doc_id = doc_info["doc_id"]
        if not isinstance(resolved_doc_id, str) or not resolved_doc_id:
            error_msg = "Failed to delete document: resolved doc_id is missing"
            deletion_errors.append(error_msg)
            logger.error("%s", error_msg)
            continue
        try:
            delete_document(
                safe_collection_name,
                resolved_doc_id,
                int(_user.id),
                bool(_user.is_admin),
            )
            deleted_doc_ids.append(resolved_doc_id)
            current_file_id = _resolve_cleanup_file_id(doc_info)
            if current_file_id:
                remaining_file_ids.discard(current_file_id)
                if _delete_uploaded_file_if_orphaned(
                    db,
                    file_id=current_file_id,
                    user_id=user_id_int,
                    remaining_file_ids=remaining_file_ids,
                ):
                    pass
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
    from ...core.tools.core.RAG_tools.storage.factory import get_vector_index_store

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

    if safe_new_collection == safe_old_collection:
        return {"status": "success", "message": "Collection name unchanged"}

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
        "message": f"Collection renamed from '{collection_name}' to '{new_name}'",
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

    try:
        safe_collection_name = sanitize_path_component(collection_name, "collection")
    except ValueError as e:
        raise HTTPException(
            status_code=422, detail=f"Invalid collection name: {str(e)}"
        ) from e

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
        elements, actual_parse_hash = reconstruct_parse_result_from_db(
            safe_collection_name,
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
