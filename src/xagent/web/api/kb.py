"""Knowledge base API route handlers"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ...core.tools.core.RAG_tools.core.schemas import (
    ChunkStrategy,
    CollectionOperationResult,
    FusionConfig,
    IngestionConfig,
    IngestionResult,
    ListCollectionsResult,
    ParseMethod,
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
from ...core.tools.core.RAG_tools.pipelines.document_ingestion import (
    run_document_ingestion,
)
from ...core.tools.core.RAG_tools.pipelines.document_search import run_document_search
from ...core.tools.core.RAG_tools.pipelines.web_ingestion import run_web_ingestion
from ...providers.vector_store.lancedb import get_connection_from_env
from ..auth_dependencies import get_current_user
from ..config import MAX_FILE_SIZE, get_upload_path, is_allowed_file
from ..models.user import User

logger = logging.getLogger(__name__)

# Create router
kb_router = APIRouter(prefix="/api/kb", tags=["kb"])


@kb_router.post(
    "/ingest",
    response_model=IngestionResult,
)
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
        embedding_model_id: Embedding model ID from model hub.
        embedding_batch_size: Batch size for embedding operations.
        max_retries: Maximum retry attempts for failures.
        retry_delay: Delay between retry attempts in seconds.
    """
    try:
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
            logger.info(f"Using file name as collection: {collection}")

        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=422,
                detail=f"File size exceeds maximum limit of {MAX_FILE_SIZE // (1024 * 1024)}MB",
            )

        # Get upload path with user isolation using unified path management
        file_path = get_upload_path(safe_filename, user_id=int(_user.id))

        # Save uploaded file with error handling
        try:
            with open(file_path, "wb") as buffer:
                buffer.write(content)
            logger.info(
                f"File uploaded: {safe_filename} -> {file_path} (user: {_user.id})"
            )
        except (PermissionError, OSError) as e:
            logger.error(f"File system error saving file {safe_filename}: {e}")
            raise HTTPException(status_code=403, detail=f"文件系统错误: {str(e)}")

        # Build configuration from individual parameters
        # Use defaults that match IngestionConfig defaults exactly
        # Validate user-provided values to prevent errors
        final_chunk_size = (
            chunk_size if chunk_size is not None and chunk_size > 0 else 1000
        )
        final_chunk_overlap = (
            chunk_overlap if chunk_overlap is not None and chunk_overlap >= 0 else 200
        )

        # Ensure overlap is always less than size
        if final_chunk_overlap >= final_chunk_size:
            final_chunk_overlap = min(int(final_chunk_size * 0.2), final_chunk_size - 1)
            logger.warning(
                f"Auto-adjusting chunk_overlap from {chunk_overlap} to {final_chunk_overlap} "
                f"to ensure it's less than chunk_size ({final_chunk_size})"
            )

        config = IngestionConfig(
            parse_method=parse_method
            if parse_method is not None
            else ParseMethod.DEFAULT,
            chunk_strategy=chunk_strategy
            if chunk_strategy is not None
            else ChunkStrategy.RECURSIVE,
            chunk_size=final_chunk_size,
            chunk_overlap=final_chunk_overlap,
            embedding_model_id=embedding_model_id,
            embedding_batch_size=embedding_batch_size
            if embedding_batch_size is not None and embedding_batch_size > 0
            else 10,
            max_retries=max_retries
            if max_retries is not None and max_retries >= 0
            else 3,
            retry_delay=retry_delay
            if retry_delay is not None and retry_delay >= 0
            else 1.0,
        )

        # Run document ingestion in a separate thread to avoid event loop conflict
        import concurrent.futures

        def _run_ingestion() -> IngestionResult:
            return run_document_ingestion(
                collection=collection,
                source_path=str(file_path),
                ingestion_config=config,
                user_id=int(_user.id),
            )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(_run_ingestion)
            result: IngestionResult = future.result()

        if result.status == "error":
            return JSONResponse(status_code=500, content=result.model_dump())

        return result

    except HTTPException:
        # Re-raise HTTP exceptions (like 422 validation errors)
        raise
    except (ValueError, KeyError, TypeError) as e:
        # 数据格式错误
        logger.error(f"Data format error in document ingestion: {e}")
        raise HTTPException(status_code=400, detail=f"数据格式错误: {str(e)}")
    except (PermissionError, OSError) as e:
        # 文件系统权限错误
        logger.error(f"File system error in document ingestion: {e}")
        raise HTTPException(status_code=403, detail=f"文件系统错误: {str(e)}")
    except Exception as e:
        # 其他错误
        logger.exception(f"Unexpected error in document ingestion: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"服务器内部错误: {str(e)}",
        )


@kb_router.get(
    "/collections",
    response_model=ListCollectionsResult,
)
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

    except Exception as e:
        logger.exception(f"Failed to list collections: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list collections: {str(e)}",
        )


@kb_router.post(
    "/search",
    response_model=SearchPipelineResult,
)
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
    try:
        # CRITICAL: Handle empty strings from Swagger UI - convert to None BEFORE any processing
        # This must be done at the very beginning to pass Pydantic validation for Dict fields
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

        result = run_document_search(
            collection=collection,
            query_text=query_text,
            search_config=config,
            user_id=int(_user.id),
            is_admin=bool(_user.is_admin),
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Document search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Document search failed: {str(e)}",
        )


@kb_router.post(
    "/ingest-web",
    response_model=WebIngestionResult,
)
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
    try:
        # Build WebCrawlConfig
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

        # Build IngestionConfig
        final_chunk_size = (
            chunk_size if chunk_size is not None and chunk_size > 0 else 1000
        )
        final_chunk_overlap = (
            chunk_overlap if chunk_overlap is not None and chunk_overlap >= 0 else 200
        )

        # Ensure overlap is always less than size
        if final_chunk_overlap >= final_chunk_size:
            final_chunk_overlap = min(int(final_chunk_size * 0.2), final_chunk_size - 1)
            logger.warning(
                f"Auto-adjusting chunk_overlap from {chunk_overlap} to {final_chunk_overlap} "
                f"to ensure it's less than chunk_size ({final_chunk_size})"
            )

        ingestion_config = IngestionConfig(
            parse_method=(
                parse_method if parse_method is not None else ParseMethod.DEFAULT
            ),
            chunk_strategy=(
                chunk_strategy
                if chunk_strategy is not None
                else ChunkStrategy.RECURSIVE
            ),
            chunk_size=final_chunk_size,
            chunk_overlap=final_chunk_overlap,
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
        import asyncio

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
            return JSONResponse(status_code=500, content=result.model_dump())

        return result

    except HTTPException:
        raise
    except (ValueError, KeyError, TypeError) as e:
        logger.error(f"Data format error in web ingestion: {e}")
        raise HTTPException(status_code=400, detail=f"Data format error: {str(e)}")
    except Exception as e:
        logger.exception(f"Unexpected error in web ingestion: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Server internal error: {str(e)}",
        )


@kb_router.delete(
    "/collections/{collection_name}",
)
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
    try:
        result = delete_collection(collection_name, int(_user.id), bool(_user.is_admin))
        return result

    except Exception as e:
        logger.exception(f"Failed to delete collection '{collection_name}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete collection: {str(e)}",
        )


@kb_router.delete(
    "/collections/{collection_name}/documents/{filename}",
)
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
    try:
        import os

        from ...core.tools.core.RAG_tools.LanceDB.schema_manager import (
            ensure_documents_table,
        )
        from ...core.tools.core.RAG_tools.management.collections import delete_document
        from ...core.tools.core.RAG_tools.utils.lancedb_query_utils import query_to_list
        from ...core.tools.core.RAG_tools.utils.string_utils import (
            build_lancedb_filter_expression,
        )
        from ...providers.vector_store.lancedb import get_connection_from_env

        # Look up doc_id(s) by filename
        conn = get_connection_from_env()
        ensure_documents_table(conn)
        table = conn.open_table("documents")

        # Filter by collection first to reduce search space
        base_filter = build_lancedb_filter_expression({"collection": collection_name})

        # Add user permission filter for multi-tenancy
        from ...core.tools.core.RAG_tools.utils.user_permissions import UserPermissions

        user_filter = UserPermissions.get_user_filter(
            int(_user.id), bool(_user.is_admin)
        )

        # Combine filters
        if user_filter and base_filter:
            combined_filter = f"({base_filter}) and ({user_filter})"
        elif user_filter:
            combined_filter = user_filter
        else:
            combined_filter = base_filter

        # Use limit() to avoid loading all records into memory
        # Assume reasonable max documents per collection for single file lookup
        MAX_SEARCH_RESULTS = 10000
        records = query_to_list(
            table.search().where(combined_filter).limit(MAX_SEARCH_RESULTS)
        )

        # Find all matching documents (handle duplicates)
        matching_docs = []
        for record in records:
            source_path = record.get("source_path", "")
            # Use basename for exact matching
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

        # Delete all matching documents
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
                    f"Deleted document '{filename}' (doc_id: {doc_id}) from collection '{collection_name}'"
                )
            except Exception as e:
                error_msg = f"Failed to delete doc_id {doc_id}: {str(e)}"
                deletion_errors.append(error_msg)
                logger.error(error_msg)

        # Return results
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

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            f"Failed to delete document '{filename}' from collection '{collection_name}': {e}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}",
        )


@kb_router.put(
    "/collections/{collection_name}",
)
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
    try:
        from ...core.tools.core.RAG_tools.management.collections import (
            _list_table_names,
        )
        from ...core.tools.core.RAG_tools.management.status import (
            load_ingestion_status,
            write_ingestion_status,
        )
        from ...core.tools.core.RAG_tools.utils.string_utils import (
            escape_lancedb_string,
        )

        conn = get_connection_from_env()

        # Validate new name
        if not new_name or not new_name.strip():
            raise HTTPException(
                status_code=422,
                detail="New collection name cannot be empty",
            )

        new_name = new_name.strip()

        if new_name == collection_name:
            return {"status": "success", "message": "Collection name unchanged"}

        warnings: list[str] = []

        # Update collection name in all tables
        table_names = _list_table_names(conn, warnings)

        # Core tables
        for table_name in ["documents", "parses", "chunks"]:
            if table_name in table_names:
                try:
                    table = conn.open_table(table_name)
                    # Update all rows for this collection
                    table.update(
                        f"collection = '{escape_lancedb_string(collection_name)}'",
                        {"collection": new_name},
                    )
                except Exception as e:
                    logger.warning(f"Failed to update '{table_name}': {e}")
                    warnings.append(f"Failed to update '{table_name}': {e}")

        # Embeddings tables
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
                logger.warning(f"Failed to update embeddings table '{table_name}': {e}")
                warnings.append(f"Failed to update '{table_name}': {e}")

        # Update ingestion status files
        try:
            status_entries = load_ingestion_status(collection=collection_name)
            for entry in status_entries:
                doc_id = entry.get("doc_id")
                if doc_id:
                    # Rewrite status with new collection name
                    write_ingestion_status(
                        new_name,
                        doc_id,
                        status=entry.get("status", "pending"),
                        message=entry.get("message", ""),
                        parse_hash=entry.get("parse_hash", ""),
                    )
                    # Clear old status
                    from ...core.tools.core.RAG_tools.management.status import (
                        clear_ingestion_status,
                    )

                    clear_ingestion_status(collection_name, doc_id)
        except Exception as e:
            logger.warning(f"Failed to update ingestion status: {e}")
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

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to rename collection '{collection_name}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to rename collection: {str(e)}",
        )
