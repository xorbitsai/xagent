"""Document ingestion pipeline orchestrating core RAG tools."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple, Union, cast

from xagent.core.model.embedding.base import BaseEmbedding
from xagent.core.model.model import EmbeddingModelConfig

from ..chunk.chunk_document import chunk_document
from ..core.exceptions import (
    DatabaseOperationError,
    DocumentValidationError,
    EmbeddingAdapterError,
    RagCoreException,
    VectorValidationError,
)
from ..core.schemas import (
    ChunkEmbeddingData,
    ChunkForEmbedding,
    DocumentProcessingStatus,
    EmbeddingReadResponse,
    EmbeddingWriteResponse,
    IngestionConfig,
    IngestionResult,
    IngestionStepResult,
    ParseDocumentResponse,
)
from ..file.register_document import register_document
from ..management.collection_manager import (
    initialize_collection_embedding_sync,
    update_collection_stats_sync,
    validate_document_processing_sync,
)
from ..management.status import write_ingestion_status
from ..parse.parse_document import parse_document
from ..utils.model_resolver import resolve_embedding_adapter
from ..vector_storage.vector_manager import (
    read_chunks_for_embedding,
    write_vectors_to_db,
)

logger = logging.getLogger(__name__)

IngestionConfigInput = Union[IngestionConfig, Mapping[str, Any]]


def _coerce_ingestion_config(config: Optional[IngestionConfigInput]) -> IngestionConfig:
    """Normalize user-provided ingestion configuration into ``IngestionConfig``."""

    if config is None:
        return IngestionConfig()
    if isinstance(config, IngestionConfig):
        return config
    if not isinstance(config, Mapping):
        raise TypeError(
            "ingestion_config must be an IngestionConfig instance or a mapping."
        )
    return IngestionConfig.model_validate(config)


def run_document_ingestion(
    collection: str,
    source_path: str,
    *,
    ingestion_config: Optional[IngestionConfigInput] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> IngestionResult:
    """Public entrypoint for LangGraph-compatible ingestion tooling.

    Accepts either a fully-specified :class:`IngestionConfig` instance or a
    mapping payload (e.g., parsed JSON) and normalises it before invoking
    :func:`process_document`.

    Args:
        collection: Target collection where the document should be ingested.
        source_path: Filesystem path to the document to ingest.
        ingestion_config: Optional configuration overrides or mapping supplied
            by external callers.
        user_id: Optional user ID for ownership tracking.
        is_admin: Whether the user has admin privileges for accessing any documents.

    Returns:
        IngestionResult: Same contract as :func:`process_document`.
    """
    cfg = _coerce_ingestion_config(ingestion_config)
    return process_document(
        collection, source_path, config=cfg, user_id=user_id, is_admin=is_admin
    )


@contextmanager
def _temp_environ(updates: Dict[str, Optional[str]]) -> Iterator[None]:
    """Temporarily set environment variables and restore afterward."""

    original: Dict[str, Optional[str]] = {}
    try:
        for key, value in updates.items():
            original[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _record_ingestion_status(
    collection: str,
    doc_id: Optional[str],
    *,
    status: DocumentProcessingStatus,
    message: str,
    parse_hash: Optional[str],
    user_id: Optional[int] = None,
) -> None:
    """Persist ingestion status without impacting pipeline flow."""
    if not doc_id:
        return
    try:
        write_ingestion_status(
            collection,
            doc_id,
            status=status.value,
            message=message,
            parse_hash=parse_hash or "",
            user_id=user_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Unable to record ingestion status for %s/%s: %s",
            collection,
            doc_id,
            exc,
        )


async def _compute_embeddings_async(
    chunks: List[ChunkForEmbedding],
    embedding_adapter: BaseEmbedding,
    embedding_config: EmbeddingModelConfig,
    max_concurrent: int,
    max_retries: int,
    retry_delay: float,
) -> List[ChunkEmbeddingData]:
    """Async concurrent computation of embedding vectors (for models that don't support batch processing, like text-embedding-v4).

    Since some models (e.g., DashScope text-embedding-v4) don't support batch processing,
    they can only handle individual requests. To improve efficiency, use asyncio for concurrent
    processing of multiple individual requests instead of serial processing.

    Args:
        chunks: List of chunks to embed
        embedding_adapter: Embedding adapter instance
        embedding_config: Embedding model configuration
        max_concurrent: Maximum concurrency
        max_retries: Maximum retry attempts
        retry_delay: Retry delay (seconds)

    Returns:
        List of embedding vector data
    """
    if not chunks:
        return []

    # Create semaphore to control concurrency
    semaphore = asyncio.Semaphore(max_concurrent)

    async def encode_single_with_retry(
        chunk: ChunkForEmbedding,
    ) -> Optional[ChunkEmbeddingData]:
        """Encode a single chunk with retry mechanism.

        Since some models (e.g., DashScope text-embedding-v4) don't support batch
        processing, they can only handle individual requests. Use asyncio.to_thread to
        execute synchronous encode calls in a thread pool, achieving async concurrent
        processing for improved efficiency.
        """
        async with semaphore:
            for retry_attempt in range(max_retries):
                try:
                    # Use asyncio.to_thread to execute synchronous encode call in thread pool
                    # Since v4 doesn't support batch processing, must process individually
                    raw_vector = await asyncio.to_thread(
                        embedding_adapter.encode, chunk.text
                    )

                    # Process return result (may be single vector or list)
                    vector: List[float]
                    if isinstance(raw_vector, list):
                        if raw_vector and isinstance(raw_vector[0], float):
                            # Single vector: List[float]
                            vector = cast(List[float], raw_vector)
                        else:
                            # List of vectors: List[List[float]], take the first one
                            if raw_vector and isinstance(raw_vector[0], list):
                                vector = cast(List[float], raw_vector[0])
                            else:
                                vector = []
                    else:
                        raise VectorValidationError(
                            f"Unexpected embedding response type: {type(raw_vector)}"
                        )

                    return ChunkEmbeddingData(
                        doc_id=chunk.doc_id,
                        chunk_id=chunk.chunk_id,
                        parse_hash=chunk.parse_hash,
                        model=embedding_config.model_name,
                        vector=vector,
                        text=chunk.text,
                        chunk_hash=chunk.chunk_hash,
                        metadata=chunk.metadata,
                    )
                except Exception as e:
                    if retry_attempt < max_retries - 1:
                        # Wait and retry (with exponential backoff)
                        await asyncio.sleep(retry_delay * (retry_attempt + 1))
                        continue
                    else:
                        # All retries failed
                        logger.warning(
                            "Chunk %s embedding computation failed after %d retries: %s",
                            chunk.chunk_id,
                            max_retries,
                            e,
                        )
                        return None
            return None

    # Execute encoding for all chunks concurrently
    tasks = [encode_single_with_retry(chunk) for chunk in chunks]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect successful results
    embeddings_data: List[ChunkEmbeddingData] = []
    failed_count = 0
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.warning(
                "Chunk %s processing exception: %s", chunks[i].chunk_id, result
            )
            failed_count += 1
        elif result is None:
            failed_count += 1
        elif isinstance(result, ChunkEmbeddingData):
            # Type guard: ensure result is ChunkEmbeddingData before appending
            embeddings_data.append(result)
        else:
            # Unexpected type (should not happen, but handle gracefully)
            logger.warning(
                "Unexpected result type for chunk %s: %s",
                chunks[i].chunk_id,
                type(result),
            )
            failed_count += 1

    if failed_count > 0:
        logger.warning(
            "%d out of %d chunks failed embedding computation",
            failed_count,
            len(chunks),
        )

    logger.info(
        "Successfully computed %d/%d embeddings using async concurrent processing",
        len(embeddings_data),
        len(chunks),
    )
    return embeddings_data


def _resolve_embedding_adapter(
    config: IngestionConfig,
) -> Tuple[EmbeddingModelConfig, BaseEmbedding]:
    """Resolve embedding adapter with priority: explicit model_id > hub > env fallback."""
    return resolve_embedding_adapter(
        config.embedding_model_id,
        api_key=config.embedding_api_key,
        base_url=config.embedding_base_url,
        timeout_sec=config.embedding_timeout_sec,
    )


def process_document(
    collection: str,
    source_path: str,
    *,
    config: Optional[IngestionConfig] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> IngestionResult:
    """Execute the full ingestion pipeline for a document.

    This orchestration step wires together document registration, parsing,
    chunking, embedding generation, and final vector-store updates. It is the
    primary entry point used by both CLI tooling and higher-level services when
    onboarding new knowledge into the RAG system.

    Args:
        collection: Logical collection name where the document and its chunks
            will be stored. Must already exist in the vector store.
        source_path: Absolute or workspace-relative path to the raw document on
            disk.
        config: Optional ingestion configuration override. When provided, any
            unspecified fields fall back to system defaults.
        user_id: Optional user ID for ownership tracking.
        is_admin: Whether the user has admin privileges.

    Returns:
        IngestionResult: A structured report describing the pipeline status,
        generated identifiers (document ID, parse hash), cumulative counts, and
        per-step metadata. The object is serialisable and intended for direct
        API responses.

    Raises:
        DocumentValidationError: If input arguments or configuration are
            invalid (e.g., missing file, chunk size constraints).
        RagCoreException: If any sub-step fails; the `failed_step` field within
            the result clarifies the exact stage.

    Notes:
        - The function aims to be idempotent: repeated runs with unchanged
          inputs will reuse existing records when possible.
        - Downstream API layers should surface `result.failed_step` and
          `result.warnings` to callers for better observability.
    """
    cfg = _coerce_ingestion_config(config)

    completed_steps: List[IngestionStepResult] = []
    warnings: List[str] = []
    doc_id: Optional[str] = None
    parse_hash: Optional[str] = None
    chunk_count = 0
    embedding_count = 0
    vector_count = 0
    current_step = "initialize_collection"
    embedding_config: Optional[EmbeddingModelConfig] = None
    embedding_adapter: Optional[BaseEmbedding] = None
    selected_model_id: Optional[str] = None

    try:
        # Step 0: Initialize/validate collection embedding configuration
        logger.info(
            "Step initialize_collection started",
            extra={"collection": collection, "source_path": source_path},
        )
        init_start = time.time()

        # Validate document processing config against collection settings
        validate_document_processing_sync(
            collection_name=collection,
            file_path=source_path,
            parsing_method=str(cfg.parse_method),
            chunking_method=str(cfg.chunk_method),
        )

        # Initialize collection embedding config if needed
        selected_model_id = cfg.embedding_model_id
        logger.info(
            f"Collection initialization: collection='{collection}', embedding_model_id='{selected_model_id}'"
        )
        if selected_model_id:
            initialize_collection_embedding_sync(
                collection_name=collection, embedding_model_id=selected_model_id
            )
        else:
            # Even without embedding_model_id, ensure basic metadata exists
            logger.info(
                f"No embedding_model_id provided for collection '{collection}', "
                "creating basic metadata without embedding configuration."
            )
            from ..management.collection_manager import get_collection_sync

            try:
                # Check if metadata already exists
                get_collection_sync(collection)
            except ValueError:
                # Metadata doesn't exist, create basic entry
                update_collection_stats_sync(collection_name=collection)
                logger.info(f"Created basic metadata for collection '{collection}'")

        init_elapsed = int((time.time() - init_start) * 1000)
        completed_steps.append(
            IngestionStepResult(
                name="initialize_collection",
                metadata={
                    "embedding_model_id": selected_model_id,
                    "elapsed_ms": init_elapsed,
                },
            )
        )
        logger.info(
            "Step initialize_collection completed",
            extra={
                "collection": collection,
                "embedding_model_id": selected_model_id,
                "elapsed_ms": init_elapsed,
            },
        )

        current_step = "resolve_embedding_adapter"
        # Step 0: Resolve embedding adapter
        # Note: Parameters passed to _resolve_embedding_adapter have priority over environment variables
        resolve_start = time.time()
        embedding_config, embedding_adapter = _resolve_embedding_adapter(cfg)
        selected_model_id = cfg.embedding_model_id or embedding_config.id

        provider = getattr(embedding_config, "model_provider", None)
        logger.info(
            "Using embedding model: id=%s, name=%s, provider=%s",
            selected_model_id,
            embedding_config.model_name,
            provider or "unknown",
        )
        resolve_elapsed = int((time.time() - resolve_start) * 1000)
        completed_steps.append(
            IngestionStepResult(
                name="resolve_embedding_adapter",
                metadata={
                    "model_id": selected_model_id,
                    "elapsed_ms": resolve_elapsed,
                },
            )
        )

        # Step 1: Register document
        current_step = "register_document"
        logger.info(
            "Step register_document started",
            extra={"collection": collection, "source_path": source_path},
        )
        register_start = time.time()
        register_result = register_document(
            collection=collection,
            source_path=source_path,
            user_id=user_id,
        )
        doc_id = register_result.get("doc_id")
        if not doc_id:
            raise DocumentValidationError(
                "register_document did not return doc_id",
                details={"collection": collection, "source_path": source_path},
            )
        _record_ingestion_status(
            collection,
            doc_id,
            status=DocumentProcessingStatus.RUNNING,
            message="Document ingestion started.",
            parse_hash=None,
            user_id=user_id,
        )
        register_elapsed = int((time.time() - register_start) * 1000)
        completed_steps.append(
            IngestionStepResult(
                name="register_document",
                metadata={
                    "doc_id": doc_id,
                    "created": register_result.get("created"),
                    "elapsed_ms": register_elapsed,
                },
            )
        )
        # Update total document count immediately after registration
        try:
            update_collection_stats_sync(
                collection_name=collection,
                documents_delta=1 if register_result.get("created") else 0,
            )
        except Exception as e:
            logger.warning(f"Failed to increment total_documents: {e}")

        logger.info(
            "Step register_document completed",
            extra={
                "doc_id": doc_id,
                "doc_created": register_result.get("created"),
                "elapsed_ms": register_elapsed,
            },
        )

        # Step 2: Parse document
        current_step = "parse_document"
        logger.info(
            "Step parse_document started",
            extra={
                "collection": collection,
                "doc_id": doc_id,
                "method": str(cfg.parse_method),
            },
        )
        parse_start = time.time()
        deepdoc_env: Dict[str, Optional[str]] = {}
        if cfg.deepdoc_processing_mode:
            deepdoc_env["DEEPDOC_PROCESSING_MODE"] = cfg.deepdoc_processing_mode
        if cfg.deepdoc_parallel_threads is not None:
            deepdoc_env["DEEPDOC_PARALLEL_THREADS"] = str(cfg.deepdoc_parallel_threads)
        if cfg.deepdoc_reserve_cpu is not None:
            deepdoc_env["DEEPDOC_RESERVE_CPU"] = str(cfg.deepdoc_reserve_cpu)
        if cfg.deepdoc_limiter_capacity is not None:
            deepdoc_env["DEEPDOC_LIMITER_CAPACITY"] = str(cfg.deepdoc_limiter_capacity)
        if cfg.deepdoc_pipeline_monitor is not None:
            deepdoc_env["DEEPDOC_PIPELINE_MONITOR"] = (
                "1" if cfg.deepdoc_pipeline_monitor else "0"
            )
        if cfg.deepdoc_pipeline_s1_workers is not None:
            deepdoc_env["DEEPDOC_PIPELINE_S1_WORKERS"] = str(
                cfg.deepdoc_pipeline_s1_workers
            )
        if cfg.deepdoc_gpu_sessions is not None:
            deepdoc_env["DEEPDOC_GPU_SESSIONS"] = str(cfg.deepdoc_gpu_sessions)

        with _temp_environ(deepdoc_env):
            parse_response = parse_document(
                collection=collection,
                doc_id=doc_id,
                parse_method=cfg.parse_method,
                params=None,
                user_id=user_id,
                is_admin=is_admin,
            )
        parse_model = (
            parse_response
            if isinstance(parse_response, ParseDocumentResponse)
            else ParseDocumentResponse.model_validate(parse_response)
        )
        parse_hash = parse_model.parse_hash
        paragraph_count = len(parse_model.paragraphs)

        if not parse_hash:
            raise DocumentValidationError(
                "parse_document did not return parse_hash",
                details={"collection": collection, "doc_id": doc_id},
            )
        parse_elapsed = int((time.time() - parse_start) * 1000)

        completed_steps.append(
            IngestionStepResult(
                name="parse_document",
                metadata={
                    "parse_hash": parse_hash,
                    "written": parse_model.written,
                    "paragraph_count": paragraph_count,
                    "elapsed_ms": parse_elapsed,
                },
            )
        )
        logger.info(
            "Step parse_document completed",
            extra={
                "collection": collection,
                "doc_id": doc_id,
                "parse_hash": parse_hash,
                "paragraph_count": paragraph_count,
                "elapsed_ms": parse_elapsed,
            },
        )

        # Step 3: Chunk document
        current_step = "chunk_document"
        logger.info(
            "Step chunk_document started",
            extra={
                "collection": collection,
                "doc_id": doc_id,
                "parse_hash": parse_hash,
                "strategy": str(cfg.chunk_strategy),
                "chunk_size": cfg.chunk_size,
                "chunk_overlap": cfg.chunk_overlap,
            },
        )
        chunk_start = time.time()
        chunk_response = chunk_document(
            collection=collection,
            doc_id=doc_id,
            parse_hash=parse_hash,
            chunk_strategy=cfg.chunk_strategy,
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            headers_to_split_on=getattr(cfg, "headers_to_split_on", None),
            separators=getattr(cfg, "separators", None),
            user_id=user_id,
        )
        chunk_count = int(chunk_response.get("chunk_count", 0))
        chunk_elapsed = int((time.time() - chunk_start) * 1000)
        completed_steps.append(
            IngestionStepResult(
                name="chunk_document",
                metadata={
                    "chunk_count": chunk_count,
                    "created": chunk_response.get("created"),
                    "elapsed_ms": chunk_elapsed,
                },
            )
        )
        logger.info(
            "Step chunk_document completed",
            extra={
                "collection": collection,
                "doc_id": doc_id,
                "parse_hash": parse_hash,
                "chunk_count": chunk_count,
                "elapsed_ms": chunk_elapsed,
            },
        )

        # Step 4: Read chunks for embedding
        current_step = "read_chunks_for_embedding"
        logger.info(
            "Step read_chunks_for_embedding started",
            extra={
                "collection": collection,
                "doc_id": doc_id,
                "parse_hash": parse_hash,
                "embedding_model": embedding_config.model_name,
            },
        )
        read_start = time.time()
        embedding_read_response = read_chunks_for_embedding(
            collection=collection,
            doc_id=doc_id,
            parse_hash=parse_hash,
            model=embedding_config.model_name,
            user_id=user_id,
            is_admin=is_admin,
        )
        read_model = (
            embedding_read_response
            if isinstance(embedding_read_response, EmbeddingReadResponse)
            else EmbeddingReadResponse.model_validate(embedding_read_response)
        )
        chunks: List[ChunkForEmbedding] = read_model.chunks
        pending_count = read_model.pending_count
        read_elapsed = int((time.time() - read_start) * 1000)

        completed_steps.append(
            IngestionStepResult(
                name="read_chunks_for_embedding",
                metadata={
                    "total_count": len(chunks),
                    "pending_count": pending_count,
                    "elapsed_ms": read_elapsed,
                },
            )
        )
        logger.info(
            "Step read_chunks_for_embedding completed",
            extra={
                "collection": collection,
                "doc_id": doc_id,
                "total_count": len(chunks),
                "pending_count": pending_count,
                "elapsed_ms": read_elapsed,
            },
        )

        if pending_count == 0:
            logger.info(
                "No pending chunks for embedding; returning early",
                extra={"collection": collection, "doc_id": doc_id},
            )
            _record_ingestion_status(
                collection,
                doc_id,
                status=DocumentProcessingStatus.SUCCESS,
                message="Document ingestion completed with no pending embeddings.",
                parse_hash=parse_hash,
                user_id=user_id,
            )
            return IngestionResult(
                status="success",
                doc_id=doc_id,
                parse_hash=parse_hash,
                chunk_count=chunk_count,
                embedding_count=0,
                vector_count=0,
                completed_steps=completed_steps,
                failed_step=None,
                message="Document ingestion completed with no pending embeddings",
                warnings=[],
            )

        # Step 5: Compute embeddings and write
        # Note: Some models (e.g., DashScope text-embedding-v4) do not support batch processing.
        # When embedding_use_async is True, we use async concurrent processing instead of batch API calls.
        # This wraps individual encode() calls with asyncio.to_thread for concurrent execution.
        current_step = "compute_embeddings"
        logger.info(
            "Step compute_embeddings started",
            extra={
                "collection": collection,
                "doc_id": doc_id,
                "pending_count": pending_count,
                "use_async": cfg.embedding_use_async,
                "batch_size": cfg.embedding_batch_size
                if not cfg.embedding_use_async
                else None,
                "concurrent": cfg.embedding_concurrent
                if cfg.embedding_use_async
                else None,
            },
        )
        embedding_start = time.time()
        total_embedding_count = 0
        total_vector_count = 0
        write_elapsed_total = 0.0
        last_write_response: Optional[EmbeddingWriteResponse] = None

        if cfg.embedding_use_async:
            # Async concurrent mode: Some models (e.g., v4) don't support batch processing,
            # so we use async concurrent single-item processing instead.
            logger.info(
                "Using async concurrent embedding computation (model does not support batch processing)"
            )
            embeddings_list = asyncio.run(
                _compute_embeddings_async(
                    chunks=chunks,
                    embedding_adapter=embedding_adapter,
                    embedding_config=embedding_config,
                    max_concurrent=cfg.embedding_concurrent,
                    max_retries=cfg.max_retries,
                    retry_delay=cfg.retry_delay,
                )
            )
            total_embedding_count = len(embeddings_list)

            # Write results in batches to database (maintain existing batch write logic)
            for batch_start in range(0, len(embeddings_list), cfg.embedding_batch_size):
                embeddings_batch_async = embeddings_list[
                    batch_start : batch_start + cfg.embedding_batch_size
                ]

                if not embeddings_batch_async:
                    continue

                write_batch_start = time.time()
                current_step = "write_vectors_to_db"
                try:
                    write_response = write_vectors_to_db(
                        collection=collection,
                        embeddings=embeddings_batch_async,
                        create_index=(
                            batch_start + cfg.embedding_batch_size
                            >= len(embeddings_list)
                        ),
                        user_id=user_id,
                    )
                    last_write_response = (
                        write_response
                        if isinstance(write_response, EmbeddingWriteResponse)
                        else EmbeddingWriteResponse.model_validate(write_response)
                    )
                    current_step = "compute_embeddings"
                except Exception as exc:  # noqa: BLE001
                    embedding_count = total_embedding_count
                    raise DatabaseOperationError(
                        "Failed to write embedding batch to vector store",
                        details={
                            "batch_start": batch_start,
                            "batch_size": len(embeddings_batch_async),
                            "error": str(exc),
                        },
                    ) from exc
                write_elapsed_total += time.time() - write_batch_start
                total_vector_count += last_write_response.upsert_count

        else:
            # Batch mode: Use original batch processing logic (for models that support batch processing)
            processed_batches = 0
            for batch_start in range(0, len(chunks), cfg.embedding_batch_size):
                batch_chunks = chunks[
                    batch_start : batch_start + cfg.embedding_batch_size
                ]
                batch_texts = [chunk.text for chunk in batch_chunks]
                raw_vectors = embedding_adapter.encode(batch_texts)
                vectors: List[List[float]]
                if not isinstance(raw_vectors, list):
                    raise VectorValidationError(
                        "Embedding adapter returned non-list response",
                        details={
                            "batch_index": processed_batches,
                            "response_type": type(raw_vectors).__name__,
                        },
                    )
                if not raw_vectors:
                    vectors = []
                elif isinstance(raw_vectors[0], float):
                    if len(batch_texts) != 1:
                        raise VectorValidationError(
                            "Embedding adapter returned single vector for multi-text batch",
                            details={
                                "batch_index": processed_batches,
                                "batch_size": len(batch_texts),
                            },
                        )
                    vectors = [raw_vectors]  # type: ignore[list-item]
                else:
                    vectors = cast(List[List[float]], raw_vectors)

                if len(vectors) != len(batch_chunks):
                    raise VectorValidationError(
                        "Embedding provider returned mismatched batch size",
                        details={
                            "batch_index": processed_batches,
                            "expected": len(batch_chunks),
                            "actual": len(vectors),
                        },
                    )

                embeddings_batch: List[ChunkEmbeddingData] = [
                    ChunkEmbeddingData(
                        doc_id=chunk.doc_id,
                        chunk_id=chunk.chunk_id,
                        parse_hash=chunk.parse_hash,
                        model=embedding_config.model_name,
                        vector=vector,
                        text=chunk.text,
                        chunk_hash=chunk.chunk_hash,
                        metadata=chunk.metadata,
                    )
                    for chunk, vector in zip(batch_chunks, vectors)
                ]
                total_embedding_count += len(embeddings_batch)
                processed_batches += 1

                if not embeddings_batch:
                    continue

                write_batch_start = time.time()
                current_step = "write_vectors_to_db"
                try:
                    write_response = write_vectors_to_db(
                        collection=collection,
                        embeddings=embeddings_batch,
                        create_index=(
                            batch_start + cfg.embedding_batch_size >= len(chunks)
                        ),
                        user_id=user_id,
                    )
                    last_write_response = (
                        write_response
                        if isinstance(write_response, EmbeddingWriteResponse)
                        else EmbeddingWriteResponse.model_validate(write_response)
                    )
                    current_step = "compute_embeddings"
                except Exception as exc:  # noqa: BLE001
                    embedding_count = total_embedding_count
                    raise DatabaseOperationError(
                        "Failed to write embedding batch to vector store",
                        details={
                            "batch_index": processed_batches - 1,
                            "batch_size": len(embeddings_batch),
                            "error": str(exc),
                        },
                    ) from exc
                write_elapsed_total += time.time() - write_batch_start
                total_vector_count += last_write_response.upsert_count

        embedding_count = total_embedding_count
        embedding_elapsed = int((time.time() - embedding_start) * 1000)

        # Check if embedding generation failed completely
        if chunk_count > 0 and embedding_count == 0:
            raise EmbeddingAdapterError(
                "Failed to generate any embeddings",
                details={
                    "chunk_count": chunk_count,
                    "use_async": cfg.embedding_use_async,
                    "embedding_model": embedding_config.model_name
                    if embedding_config
                    else None,
                },
            )

        completed_steps.append(
            IngestionStepResult(
                name="compute_embeddings",
                metadata={
                    "embedding_count": embedding_count,
                    "use_async": cfg.embedding_use_async,
                    "batch_size": cfg.embedding_batch_size
                    if not cfg.embedding_use_async
                    else None,
                    "concurrent": cfg.embedding_concurrent
                    if cfg.embedding_use_async
                    else None,
                    "elapsed_ms": embedding_elapsed,
                },
            )
        )
        logger.info(
            "Step compute_embeddings completed",
            extra={
                "collection": collection,
                "doc_id": doc_id,
                "embedding_count": embedding_count,
                "use_async": cfg.embedding_use_async,
                "elapsed_ms": embedding_elapsed,
            },
        )

        vector_count = total_vector_count
        write_elapsed_ms = int(write_elapsed_total * 1000)
        current_step = "write_vectors_to_db"
        completed_steps.append(
            IngestionStepResult(
                name="write_vectors_to_db",
                metadata={
                    "vector_count": vector_count,
                    "elapsed_ms": write_elapsed_ms,
                },
            )
        )
        logger.info(
            "Step write_vectors_to_db completed",
            extra={
                "collection": collection,
                "doc_id": doc_id,
                "vector_count": vector_count,
                "index_status": (
                    last_write_response.index_status
                    if last_write_response is not None
                    else "skipped"
                ),
                "elapsed_ms": write_elapsed_ms,
            },
        )

        # Update collection statistics
        try:
            import os

            document_name = os.path.basename(source_path)
            update_collection_stats_sync(
                collection_name=collection,
                documents_delta=1,  # Added one document
                processed_documents_delta=1,  # Success!
                parses_delta=1,  # One parse operation
                chunks_delta=chunk_count,
                embeddings_delta=vector_count,
                document_name=document_name,
            )
            logger.info(
                "Collection statistics updated",
                extra={
                    "collection": collection,
                    "document_name": document_name,
                    "parsing_method": str(cfg.parse_method),
                    "chunking_method": str(cfg.chunk_method),
                },
            )
        except Exception as stat_exc:
            logger.warning(
                "Failed to update collection statistics: %s",
                stat_exc,
                extra={"collection": collection, "doc_id": doc_id},
            )
            warnings.append(f"Collection statistics update failed: {stat_exc}")

        _record_ingestion_status(
            collection,
            doc_id,
            status=DocumentProcessingStatus.SUCCESS,
            message="Document ingestion completed successfully.",
            parse_hash=parse_hash,
            user_id=user_id,
        )
        return IngestionResult(
            status="success",
            doc_id=doc_id,
            parse_hash=parse_hash,
            chunk_count=chunk_count,
            embedding_count=embedding_count,
            vector_count=vector_count,
            completed_steps=completed_steps,
            failed_step=None,
            message="Document ingestion completed successfully",
            warnings=warnings,
        )

    except RagCoreException as exc:
        logger.exception("Document ingestion pipeline failed: %s", exc)
        status = "partial" if completed_steps else "error"
        _record_ingestion_status(
            collection,
            doc_id,
            status=DocumentProcessingStatus.FAILED,
            message=str(exc),
            parse_hash=parse_hash,
            user_id=user_id,
        )
        return IngestionResult(
            status=status,
            doc_id=doc_id,
            parse_hash=parse_hash,
            chunk_count=chunk_count if status == "partial" else 0,
            embedding_count=embedding_count if status == "partial" else 0,
            vector_count=vector_count if status == "partial" else 0,
            completed_steps=completed_steps,
            failed_step=current_step,
            message=str(exc),
            warnings=warnings,
        )
    except Exception as exc:
        logger.exception("Document ingestion pipeline failed: %s", exc)
        status = "partial" if completed_steps else "error"
        _record_ingestion_status(
            collection,
            doc_id,
            status=DocumentProcessingStatus.FAILED,
            message=str(exc),
            parse_hash=parse_hash,
            user_id=user_id,
        )
        return IngestionResult(
            status=status,
            doc_id=doc_id,
            parse_hash=parse_hash,
            chunk_count=chunk_count if status == "partial" else 0,
            embedding_count=embedding_count if status == "partial" else 0,
            vector_count=vector_count if status == "partial" else 0,
            completed_steps=completed_steps,
            failed_step=current_step,
            message=str(exc),
            warnings=warnings,
        )
