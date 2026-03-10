"""Knowledge base API route handlers"""

import asyncio
import functools
import json
import logging
import os
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

from ...core.tools.core.RAG_tools.core.schemas import (
    ChunkStrategy,
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
)
from ...core.tools.core.RAG_tools.parse.parse_display import (
    paginate_parse_results,
    reconstruct_parse_result_from_db,
)
from ...core.tools.core.RAG_tools.pipelines.document_ingestion import (
    run_document_ingestion,
)
from ...core.tools.core.RAG_tools.pipelines.document_search import run_document_search
from ...core.tools.core.RAG_tools.pipelines.web_ingestion import run_web_ingestion
from ...core.tools.core.RAG_tools.progress import get_progress_manager
from ...providers.vector_store.lancedb import get_connection_from_env
from ..auth_dependencies import get_current_user
from ..config import MAX_FILE_SIZE, get_upload_path, is_allowed_file
from ..models.user import User

T = TypeVar("T", bound=Callable[..., Any])
logger = logging.getLogger(__name__)


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
            raise HTTPException(status_code=403, detail=f"文件系统错误: {str(e)}")
        except Exception as e:
            logger.exception("Unexpected error in %s: %s", func.__name__, e)
            raise HTTPException(
                status_code=500,
                detail=f"服务器内部错误: {str(e)}",
            )

    return cast(T, wrapper)


# Create router
kb_router = APIRouter(prefix="/api/kb", tags=["kb"])


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

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"File size exceeds maximum limit of {MAX_FILE_SIZE // (1024 * 1024)}MB",
        )

    file_path = get_upload_path(safe_filename, user_id=int(_user.id))

    try:
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        logger.info(
            "File uploaded: %s -> %s (user: %s)", safe_filename, file_path, _user.id
        )
    except (PermissionError, OSError) as e:
        logger.error("File system error saving file %s: %s", safe_filename, e)
        raise HTTPException(status_code=403, detail=f"文件系统错误: {str(e)}")

    final_chunk_size = chunk_size if chunk_size is not None and chunk_size > 0 else 1000
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
    result: IngestionResult = await asyncio.to_thread(
        run_document_ingestion,
        collection=collection,
        source_path=str(file_path),
        ingestion_config=config,
        progress_manager=progress_manager,
        user_id=int(_user.id),
        is_admin=bool(_user.is_admin),
    )

    if result.status == "error":
        logger.error(
            "KB ingest failed (collection=%s, filename=%s, file_path=%s, user_id=%s)",
            collection,
            safe_filename,
            file_path,
            _user.id,
        )
        return JSONResponse(status_code=500, content=result.model_dump())
    return result


@kb_router.get(
    "/collections",
    response_model=ListCollectionsResult,
)
@handle_kb_exceptions
async def list_collections_api(
    _user: User = Depends(get_current_user),
) -> ListCollectionsResult:
    """List all collections with their statistics."""
    kb_collections_timeout_seconds = 15

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(list_collections, int(_user.id), bool(_user.is_admin)),
            timeout=kb_collections_timeout_seconds,
        )
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
        description="Optional filters to apply during search (LanceDB format)",
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
        collection=collection,
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
    # Build WebCrawlConfig
    url_patterns_list = (
        [p.strip() for p in url_patterns.split(",")] if url_patterns else None
    )
    exclude_patterns_list = (
        [p.strip() for p in exclude_patterns.split(",")] if exclude_patterns else None
    )
    remove_selectors_list = (
        [s.strip() for s in remove_selectors.split(",")] if remove_selectors else None
    )

    crawl_config = WebCrawlConfig(
        start_url=start_url,
        max_pages=max_pages or 100,
        max_depth=max_depth or 3,
        url_patterns=url_patterns_list,
        exclude_patterns=exclude_patterns_list,
        same_domain_only=(same_domain_only if same_domain_only is not None else True),
        content_selector=content_selector,
        remove_selectors=remove_selectors_list,
        concurrent_requests=concurrent_requests or 3,
        request_delay=request_delay or 1.0,
        timeout=timeout or 30,
        respect_robots_txt=(
            respect_robots_txt if respect_robots_txt is not None else True
        ),
    )

    # Build IngestionConfig
    final_chunk_size = chunk_size if chunk_size is not None and chunk_size > 0 else 1000
    final_chunk_overlap = (
        chunk_overlap if chunk_overlap is not None and chunk_overlap >= 0 else 200
    )

    # Ensure overlap is always less than size
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

    # Run web ingestion
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: asyncio.run(
            run_web_ingestion(
                collection=collection,
                crawl_config=crawl_config,
                ingestion_config=ingestion_config,
                user_id=int(_user.id),
                is_admin=bool(_user.is_admin),
            )
        ),
    )

    if result.status == "error":
        logger.error(
            "KB web ingest failed (collection=%s, start_url=%s, user_id=%s)",
            collection,
            start_url,
            _user.id,
        )
        return JSONResponse(status_code=500, content=result.model_dump())
    return result


@kb_router.delete(
    "/collections/{collection_name}",
)
@handle_kb_exceptions
async def delete_collection_api(
    collection_name: str,
    _user: User = Depends(get_current_user),
) -> CollectionOperationResult:
    """Delete a collection and all its data.

    Args:
        collection_name: Name of the collection to delete

    Returns:
        Deletion result with status and affected documents
    """
    result = delete_collection(collection_name, int(_user.id), bool(_user.is_admin))
    return result


@kb_router.post(
    "/collections/{collection_name}/documents/check",
)
async def check_documents_exist_api(
    collection_name: str,
    body: Dict[str, Any] = Body(
        ..., description="JSON body with 'filenames': list of filename strings"
    ),
    _user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Check which of the given filenames already exist in the collection.

    Used by the frontend to show "file already exists, re-upload?" before ingest.
    Duplicate is determined by: same collection + document with same source_path basename.

    For duplicate check we always filter by current user's documents only (including
    for admins), so "already exists" matches what will be overwritten on re-upload.
    """
    try:
        from ...core.tools.core.RAG_tools.LanceDB.schema_manager import (
            ensure_documents_table,
        )
        from ...core.tools.core.RAG_tools.utils.lancedb_query_utils import query_to_list
        from ...core.tools.core.RAG_tools.utils.string_utils import (
            build_lancedb_filter_expression,
        )
        from ...core.tools.core.RAG_tools.utils.user_permissions import UserPermissions
        from ...providers.vector_store.lancedb import get_connection_from_env

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

        conn = get_connection_from_env()
        ensure_documents_table(conn)
        table = conn.open_table("documents")

        base_filter = build_lancedb_filter_expression({"collection": collection_name})
        # Use own-files-only filter even for admins so duplicate check matches re-upload behavior
        user_filter = UserPermissions.get_user_filter(int(_user.id), is_admin=False)
        combined_filter = (
            f"({base_filter}) and ({user_filter})"
            if user_filter and base_filter
            else (user_filter or base_filter)
        )
        MAX_SEARCH_RESULTS = 10000
        records = query_to_list(
            table.search().where(combined_filter).limit(MAX_SEARCH_RESULTS)
        )

        existing_basenames = set()
        for record in records:
            sp = record.get("source_path")
            if sp:
                existing_basenames.add(os.path.basename(str(sp)))

        existing_filenames = sorted(requested & existing_basenames)
        return {"existing_filenames": existing_filenames}

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
    _user: User = Depends(get_current_user),
) -> dict:
    """Delete a document and all its associated data.

    Args:
        collection_name: Name of the collection
        filename: Document filename (will be used to find doc_id)

    Returns:
        Deletion result with status, list of deleted doc_ids, and filename

    Note:
        This endpoint uses filename lookup which may have a race condition if
        the same filename is uploaded multiple times concurrently. For production
        use, consider using doc_id directly or adding a filename index column.
    """
    # NOTE: Exceptions are normalized by @handle_kb_exceptions for consistent API responses.
    from ...core.tools.core.RAG_tools.LanceDB.schema_manager import (
        ensure_documents_table,
    )
    from ...core.tools.core.RAG_tools.management.collections import delete_document
    from ...core.tools.core.RAG_tools.utils.lancedb_query_utils import query_to_list
    from ...core.tools.core.RAG_tools.utils.string_utils import (
        build_lancedb_filter_expression,
    )
    from ...core.tools.core.RAG_tools.utils.user_permissions import UserPermissions
    from ...providers.vector_store.lancedb import get_connection_from_env

    # Look up doc_id(s) by filename
    conn = get_connection_from_env()
    ensure_documents_table(conn)
    table = conn.open_table("documents")

    # Filter by collection first to reduce search space
    base_filter = build_lancedb_filter_expression({"collection": collection_name})

    user_filter = UserPermissions.get_user_filter(int(_user.id), bool(_user.is_admin))

    if user_filter and base_filter:
        combined_filter = f"({base_filter}) and ({user_filter})"
    elif user_filter:
        combined_filter = user_filter
    else:
        combined_filter = base_filter

    MAX_SEARCH_RESULTS = 10000
    records = query_to_list(
        table.search().where(combined_filter).limit(MAX_SEARCH_RESULTS)
    )

    matching_docs = []
    for record in records:
        source_path = record.get("source_path", "")
        if source_path and os.path.basename(str(source_path)) == filename:
            matching_docs.append(
                {
                    "doc_id": record.get("doc_id"),
                    "source_path": source_path,
                }
            )

    if not matching_docs:
        raise HTTPException(
            status_code=404,
            detail=f"Document not found: {filename}",
        )

    deleted_doc_ids = []
    deletion_errors = []

    for doc_info in matching_docs:
        doc_id = doc_info["doc_id"]
        try:
            delete_document(
                collection_name, doc_id, int(_user.id), bool(_user.is_admin)
            )
            deleted_doc_ids.append(doc_id)
            logger.info(
                "Deleted document '%s' (doc_id: %s) from collection '%s'",
                filename,
                doc_id,
                collection_name,
            )
        except Exception as e:
            error_msg = f"Failed to delete doc_id {doc_id}: {str(e)}"
            deletion_errors.append(error_msg)
            logger.error("%s", error_msg)

    if deletion_errors:
        return {
            "status": "partial_success" if deleted_doc_ids else "failed",
            "message": f"Deleted {len(deleted_doc_ids)} of {len(matching_docs)} documents",
            "collection": collection_name,
            "filename": filename,
            "deleted_doc_ids": deleted_doc_ids,
            "errors": deletion_errors,
        }

    return {
        "status": "success",
        "message": f"Successfully deleted {len(deleted_doc_ids)} document(s)",
        "collection": collection_name,
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
) -> dict:
    """Rename a collection.

    Args:
        collection_name: Current collection name
        new_name: New collection name

    Returns:
        Success message
    """
    from ...core.tools.core.RAG_tools.management.collections import (
        _list_table_names,
    )
    from ...core.tools.core.RAG_tools.management.status import (
        clear_ingestion_status,
        load_ingestion_status,
        write_ingestion_status,
    )
    from ...core.tools.core.RAG_tools.utils.string_utils import (
        escape_lancedb_string,
    )

    conn = get_connection_from_env()

    if not new_name or not new_name.strip():
        raise HTTPException(
            status_code=422,
            detail="New collection name cannot be empty",
        )

    new_name = new_name.strip()

    if new_name == collection_name:
        return {"status": "success", "message": "Collection name unchanged"}

    warnings: list[str] = []

    table_names = _list_table_names(conn, warnings)

    for table_name in ["documents", "parses", "chunks"]:
        if table_name in table_names:
            try:
                table = conn.open_table(table_name)
                table.update(
                    f"collection = '{escape_lancedb_string(collection_name)}'",
                    {"collection": new_name},
                )
            except Exception as e:
                logger.warning("Failed to update '%s': %s", table_name, e)
                warnings.append(f"Failed to update '{table_name}': {e}")

    for table_name in table_names:
        if not table_name.startswith("embeddings_"):
            continue
        try:
            table = conn.open_table(table_name)
            table.update(
                f"collection = '{escape_lancedb_string(collection_name)}'",
                {"collection": new_name},
            )
        except Exception as e:
            logger.warning("Failed to update embeddings table '%s': %s", table_name, e)
            warnings.append(f"Failed to update '{table_name}': {e}")

    try:
        status_entries = load_ingestion_status(collection=collection_name)
        for entry in status_entries:
            doc_id = entry.get("doc_id")
            if doc_id:
                write_ingestion_status(
                    new_name,
                    doc_id,
                    status=entry.get("status", "pending"),
                    message=entry.get("message", ""),
                    parse_hash=entry.get("parse_hash", ""),
                )
                clear_ingestion_status(collection_name, doc_id)
    except Exception as e:
        logger.warning("Failed to update ingestion status: %s", e)
        warnings.append(f"Failed to update ingestion status: {e}")

    if warnings:
        return {
            "status": "partial_success",
            "message": f"Collection renamed from '{collection_name}' to '{new_name}' with some warnings",
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
            collection_name,
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
