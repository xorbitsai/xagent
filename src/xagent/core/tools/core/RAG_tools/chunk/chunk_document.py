"""Main entry point for document chunking.

This module provides the main chunk_document function that orchestrates
document chunking using various chunking strategies.
"""

import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..core.config import (
    DEFAULT_IMAGE_CONTEXT_SIZE,
    DEFAULT_TABLE_CONTEXT_SIZE,
    DEFAULT_TIKTOKEN_ENCODING,
)
from ..core.exceptions import (
    DatabaseOperationError,
    DocumentNotFoundError,
    DocumentValidationError,
)
from ..core.schemas import ChunkStrategy
from ..storage.factory import get_vector_index_store
from ..utils.hash_utils import compute_chunk_hash
from ..utils.metadata_utils import deserialize_metadata, serialize_metadata
from .chunk_strategies import (
    _create_chunk_record,
    apply_fixed_size_strategy,
    apply_markdown_strategy,
    apply_recursive_strategy,
    attach_media_context,
)

logger = logging.getLogger(__name__)


def _should_use_spreadsheet_row_chunks(paragraphs: List[Dict[str, Any]]) -> bool:
    non_empty = [p for p in paragraphs if str(p.get("text", "")).strip()]
    if not non_empty:
        return False

    allowed_row_types = {"title", "header", "data"}
    for paragraph in non_empty:
        metadata = paragraph.get("metadata") or {}
        file_type = str(metadata.get("file_type") or "").lower()
        file_ext = str(metadata.get("file_ext") or "").lower()
        if file_type not in {"xlsx", ".xlsx"} and file_ext != ".xlsx":
            return False
        if metadata.get("row_type") not in allowed_row_types:
            return False
    return True


def _create_spreadsheet_row_chunks(
    paragraphs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for paragraph in paragraphs:
        text = str(paragraph.get("text", "")).strip()
        if not text:
            continue
        chunks.append(_create_chunk_record(text, paragraph))
    return chunks


def chunk_document(
    collection: str,
    doc_id: str,
    parse_hash: str,
    chunk_strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
    chunk_size: Optional[int] = 1000,
    chunk_overlap: int = 200,
    headers_to_split_on: Optional[List[Tuple[str, str]]] = None,
    separators: Optional[List[str]] = None,
    use_token_count: bool = False,
    tiktoken_encoding: str = DEFAULT_TIKTOKEN_ENCODING,
    enable_protected_content: bool = True,
    protected_patterns: Optional[List[str]] = None,
    table_context_size: int = DEFAULT_TABLE_CONTEXT_SIZE,
    image_context_size: int = DEFAULT_IMAGE_CONTEXT_SIZE,
    user_id: Optional[int] = None,
    is_admin: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Chunk parsed paragraphs and write to chunks table.

    Args:
        collection: Collection name for data isolation
        doc_id: Document ID whose parsed result to chunk
        parse_hash: Parse version hash to select parsed content
        chunk_strategy: Chunking strategy identifier
        chunk_size: Target chunk size in characters (or tokens when use_token_count=True). If None, semantic splitting is used without size limits
        chunk_overlap: Overlap between consecutive chunks (characters or tokens when use_token_count=True)
        headers_to_split_on: Markdown header rules for markdown strategy
        separators: Separators for recursive strategy
        use_token_count: If True, chunk_size and chunk_overlap are in tokens (tiktoken); only applies to RECURSIVE strategy
        tiktoken_encoding: tiktoken encoding name when use_token_count=True (e.g. "cl100k_base")
        enable_protected_content: If True (default), do not split inside code blocks, formulas, tables (P1)
        protected_patterns: Optional list of regex patterns for protected regions; None uses config default
        table_context_size: Chars from prev/next chunk to attach to table chunks; 0 = off (P2)
        image_context_size: Chars from prev/next chunk to attach to image chunks; 0 = off (P2)
        user_id: Optional user ID for multi-tenancy data isolation

    Returns:
        Dictionary containing chunk results

    Raises:
        DocumentValidationError: If parameters are invalid
        DocumentNotFoundError: If parsed content is not found
        DatabaseOperationError: If database operations fail
    """
    if not collection or not doc_id or not parse_hash:
        raise DocumentValidationError("collection/doc_id/parse_hash is required")

    params: Dict[str, Any] = {
        "chunk_strategy": str(
            chunk_strategy
        ),  # Convert enum to string for JSON serialization
        "chunk_size": chunk_size,  # Keep as Optional[int] or int
        "chunk_overlap": int(chunk_overlap),
        "headers_to_split_on": headers_to_split_on,
        "separators": separators,
        "use_token_count": use_token_count,
        "tiktoken_encoding": tiktoken_encoding,
        "enable_protected_content": enable_protected_content,
        "protected_patterns": protected_patterns,
        "table_context_size": table_context_size,
        "image_context_size": image_context_size,
    }

    logger.info(
        f"Starting document chunking: doc_id={doc_id}, strategy={chunk_strategy}"
    )

    # Validate chunk parameters
    _validate_chunk_params(chunk_strategy, params)

    # Compute configuration-level hash for this chunking run
    try:
        config_hash = compute_chunk_hash("", params)
    except Exception as e:
        raise DocumentValidationError(f"Failed to compute config_hash: {e}") from e

    logger.info(f"Computed chunk config hash: {config_hash}")

    # OPTIMIZATION: Check and get existing chunks in a single query
    # Instead of calling _chunks_exist() then _get_existing_chunks() (2 queries),
    # directly call _get_existing_chunks() and check if result is empty (1 query)
    existing_chunks = _get_existing_chunks(
        collection, doc_id, parse_hash, config_hash, user_id, is_admin
    )

    if existing_chunks:
        logger.info(
            f"Chunk record already exists for doc_id={doc_id}, parse_hash={parse_hash}, config_hash={config_hash}"
        )
        return {
            "doc_id": doc_id,
            "parse_hash": parse_hash,
            "chunk_count": len(existing_chunks),
            "stats": _compute_stats(existing_chunks),
            "created": False,
        }

    # Load parsed content from database
    paragraphs = _load_paragraphs(collection, doc_id, parse_hash, user_id, is_admin)
    if not paragraphs:
        raise DocumentNotFoundError(
            f"No parsed content found for doc_id={doc_id}, parse_hash={parse_hash}"
        )

    # Spreadsheet row data benefits from deterministic one-row-per-chunk output
    # for retrieval quality; bypass generic recursive merge when parse metadata
    # already marks row boundaries explicitly.
    if _should_use_spreadsheet_row_chunks(paragraphs):
        chunks = _create_spreadsheet_row_chunks(paragraphs)
    else:
        # Apply chunking strategy
        try:
            chunks = _apply_chunking_strategy(paragraphs, chunk_strategy, params)
        except Exception as e:
            logger.error(f"Document chunking failed: {e}")
            raise DocumentValidationError(f"Chunking failed: {e}") from e

    # P2: Attach surrounding context to table/image chunks
    if chunks and (
        params.get("table_context_size", 0) > 0
        or params.get("image_context_size", 0) > 0
    ):
        attach_media_context(
            chunks,
            table_context_size=int(params.get("table_context_size", 0)),
            image_context_size=int(params.get("image_context_size", 0)),
        )

    # Assign ids and indices
    indexed_chunks = []
    for idx, chunk in enumerate(chunks):
        indexed_chunks.append(
            {
                "chunk_id": chunk.get("chunk_id", str(uuid.uuid4())),
                "index": int(chunk.get("index", idx)),
                "text": chunk.get("text", ""),
                "page_number": chunk.get("page_number"),
                "section": chunk.get("section"),
                "anchor": chunk.get("anchor"),
                "json_path": chunk.get("json_path"),
                "created_at": chunk.get("created_at", pd.Timestamp.now(tz="UTC")),
                "metadata": chunk.get("metadata"),
            }
        )

    # Write to database
    try:
        written = _write_chunks_to_db(
            collection,
            doc_id,
            parse_hash,
            config_hash,
            params,
            indexed_chunks,
            user_id,
            is_admin,
        )
    except Exception as e:
        logger.error(f"Failed to write chunks to database: {e}")
        raise DatabaseOperationError(f"Database write failed: {e}") from e

    logger.info(
        f"Document chunking completed: doc_id={doc_id}, chunks={len(indexed_chunks)}"
    )
    return {
        "doc_id": doc_id,
        "parse_hash": parse_hash,
        "chunk_count": len(indexed_chunks),
        "stats": _compute_stats(indexed_chunks),
        "created": written,
    }


def _validate_chunk_params(
    chunk_strategy: ChunkStrategy, params: Dict[str, Any]
) -> None:
    """Validate chunking parameters."""
    # Enum validation is handled by type system, but keep runtime check for safety
    valid_strategies = {
        ChunkStrategy.RECURSIVE,
        ChunkStrategy.MARKDOWN,
        ChunkStrategy.FIXED_SIZE,
    }
    if chunk_strategy not in valid_strategies:
        raise DocumentValidationError(f"Unsupported chunk strategy: {chunk_strategy}")

    chunk_size = params.get("chunk_size", 1000)
    chunk_overlap = params.get("chunk_overlap", 200)
    use_token_count = bool(params.get("use_token_count"))

    if use_token_count and chunk_size is None:
        raise DocumentValidationError(
            "chunk_size is required when use_token_count is True"
        )
    if chunk_size is not None and chunk_size <= 0:
        raise DocumentValidationError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise DocumentValidationError("chunk_overlap must be non-negative")
    if chunk_size is not None and chunk_overlap >= chunk_size:
        raise DocumentValidationError("chunk_overlap must be less than chunk_size")


def _chunks_exist(
    collection: str, doc_id: str, parse_hash: str, config_hash: str
) -> bool:
    """Check if chunk records already exist."""
    try:
        vector_store = get_vector_index_store()

        # Build safe filter expression using utility function
        query_filters = {
            "collection": collection,
            "doc_id": doc_id,
            "parse_hash": parse_hash,
            "config_hash": config_hash,
        }
        return vector_store.count_rows_or_zero("chunks", filters=query_filters) > 0
    except Exception as e:
        logger.error(f"Failed to check chunk existence: {e}")
        raise DatabaseOperationError(f"Database query failed: {e}") from e


def _get_existing_chunks(
    collection: str,
    doc_id: str,
    parse_hash: str,
    config_hash: str,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> List[Dict[str, Any]]:
    """Get existing chunks from database.

    OPTIMIZATION: Uses count_rows() for memory-efficient existence check,
    then to_list() to avoid pandas overhead when loading data.

    Args:
        collection: Collection name
        doc_id: Document ID
        parse_hash: Parse hash
        config_hash: Configuration hash
        user_id: Optional user ID for multi-tenancy filtering
        is_admin: Whether user has admin privileges

    Returns:
        List of existing chunks accessible to the user
    """
    try:
        vector_store = get_vector_index_store()

        # Build safe filter expression using utility function
        query_filters = {
            "collection": collection,
            "doc_id": doc_id,
            "parse_hash": parse_hash,
            "config_hash": config_hash,
        }

        # OPTIMIZATION: Use count_rows_or_zero() for memory-efficient existence check
        if (
            vector_store.count_rows_or_zero(
                "chunks", filters=query_filters, user_id=user_id, is_admin=is_admin
            )
            == 0
        ):
            return []

        # Use iter_batches to load chunks
        chunks_data = []
        for batch in vector_store.iter_batches(
            table_name="chunks",
            filters=query_filters,
            user_id=user_id,
            is_admin=is_admin,
        ):
            # Convert batch to pandas for easier row-by-row processing
            batch_df = batch.to_pandas()
            for _, row in batch_df.iterrows():
                chunks_data.append(row.to_dict())

        # Convert to expected format with metadata deserialization
        # Arrow/to_list() returns None instead of NaN, so direct None check is sufficient
        chunks = []
        for row in chunks_data:
            # Deserialize metadata from JSON string to dictionary
            metadata = deserialize_metadata(row.get("metadata"))

            # Handle index with None check (NaN already normalized to None in pandas fallback)
            index_value = row.get("index")
            index = int(index_value) if index_value is not None else 0

            # Normalize optional fields: None check is sufficient
            page_number = (
                row.get("page_number") if row.get("page_number") is not None else None
            )
            section = row.get("section") if row.get("section") is not None else None
            anchor = row.get("anchor") if row.get("anchor") is not None else None
            json_path = (
                row.get("json_path") if row.get("json_path") is not None else None
            )

            chunks.append(
                {
                    "chunk_id": row["chunk_id"],
                    "index": index,
                    "text": row["text"],
                    "page_number": page_number,
                    "section": section,
                    "anchor": anchor,
                    "json_path": json_path,
                    "created_at": row["created_at"],
                    "metadata": metadata,
                }
            )
        return chunks
    except Exception as e:
        logger.error(f"Failed to get existing chunks: {e}")
        raise DatabaseOperationError(f"Database query failed: {e}") from e


def _load_paragraphs(
    collection: str,
    doc_id: str,
    parse_hash: str,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> List[Dict[str, Any]]:
    """Load parsed content from parses table."""
    try:
        vector_store = get_vector_index_store()

        # Build safe filter expression using utility function
        query_filters = {
            "collection": collection,
            "doc_id": doc_id,
            "parse_hash": parse_hash,
        }

        # First check if any parse exists using efficient count_rows_or_zero
        if (
            vector_store.count_rows_or_zero(
                "parses", filters=query_filters, user_id=user_id, is_admin=is_admin
            )
            == 0
        ):
            return []

        # Load data using iter_batches
        records = []
        for batch in vector_store.iter_batches(
            table_name="parses",
            filters=query_filters,
            user_id=user_id,
            is_admin=is_admin,
        ):
            # Convert batch to pandas for easier row-by-row processing
            batch_df = batch.to_pandas()
            for _, row in batch_df.iterrows():
                records.append(row.to_dict())

        if not records:
            return []
        record = records[0]

        parsed_content = record.get("parsed_content")
        if not parsed_content:
            return []

        data = json.loads(parsed_content)
        return [
            {"text": item.get("text", ""), "metadata": item.get("metadata", {})}
            for item in data
        ]
    except Exception as e:
        logger.error(f"Failed to read parses: {e}")
        raise DatabaseOperationError(f"Failed reading parses: {e}") from e


def _apply_chunking_strategy(
    paragraphs: List[Dict[str, Any]],
    chunk_strategy: ChunkStrategy,
    params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Apply the specified chunking strategy."""
    if chunk_strategy == ChunkStrategy.RECURSIVE:
        return apply_recursive_strategy(paragraphs, params)
    elif chunk_strategy == ChunkStrategy.MARKDOWN:
        return apply_markdown_strategy(paragraphs, params)
    elif chunk_strategy == ChunkStrategy.FIXED_SIZE:
        return apply_fixed_size_strategy(paragraphs, params)
    else:
        raise DocumentValidationError(f"Unsupported chunk strategy: {chunk_strategy}")


def _write_chunks_to_db(
    collection: str,
    doc_id: str,
    parse_hash: str,
    config_hash: str,
    params: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> bool:
    """Write chunk records to database using abstraction layer."""
    try:
        rows = []
        for chunk in chunks:
            text = chunk["text"]
            # Serialize metadata dictionary to JSON string for database storage
            metadata = chunk.get("metadata")
            row = {
                "collection": collection,
                "doc_id": doc_id,
                "parse_hash": parse_hash,
                "chunk_id": chunk["chunk_id"],
                "index": int(chunk["index"]),
                "text": text,
                "page_number": chunk.get("page_number"),
                "section": chunk.get("section"),
                "anchor": chunk.get("anchor"),
                "json_path": chunk.get("json_path"),
                "chunk_hash": compute_chunk_hash(text, params),
                "config_hash": config_hash,
                "created_at": chunk["created_at"],
                "metadata": serialize_metadata(metadata),
                "user_id": user_id,  # Add user_id for multi-tenancy
            }
            rows.append(row)

        if not rows:
            return False

        # Use abstraction layer for upsert
        vector_store = get_vector_index_store()
        vector_store.upsert_chunks(rows)

        logger.info(
            f"Chunk records written to database: doc_id={doc_id}, parse_hash={parse_hash}, config_hash={config_hash}"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to write chunk records: {e}")
        raise DatabaseOperationError(f"Database write failed: {e}") from e


def _compute_stats(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute statistics for chunks."""
    if not chunks:
        return {"total_chunks": 0, "avg_chunk_length": 0.0}

    total_length = sum(len(chunk.get("text", "")) for chunk in chunks)
    return {
        "total_chunks": len(chunks),
        "avg_chunk_length": float(total_length / len(chunks)),
    }


# Fine-grained chunking functions for LangGraph tools
# These functions provide specific chunking strategies while maintaining database integration


def chunk_recursive(
    collection: str,
    doc_id: str,
    parse_hash: str,
    chunk_size: Optional[int] = 1000,
    chunk_overlap: int = 200,
    separators: Optional[List[str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Chunk document using recursive character splitting strategy.

    Args:
        collection: Collection name for data isolation
        doc_id: Document ID whose parsed result to chunk
        parse_hash: Parse version hash to select parsed content
        chunk_size: Target chunk size in characters. If None, semantic splitting is used without size limits
        chunk_overlap: Overlap between consecutive chunks
        separators: Custom separators for splitting

    Returns:
        Dictionary containing chunk results and statistics
    """
    return chunk_document(
        collection=collection,
        doc_id=doc_id,
        parse_hash=parse_hash,
        chunk_strategy=ChunkStrategy.RECURSIVE,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        **kwargs,
    )


def chunk_markdown(
    collection: str,
    doc_id: str,
    parse_hash: str,
    chunk_size: Optional[int] = 1200,
    chunk_overlap: int = 200,
    headers_to_split_on: Optional[List[Tuple[str, str]]] = None,
    separators: Optional[List[str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Chunk document using markdown header-based strategy.

    Args:
        collection: Collection name for data isolation
        doc_id: Document ID whose parsed result to chunk
        parse_hash: Parse version hash to select parsed content
        chunk_size: Target chunk size in characters. If None, semantic splitting is used without size limits
        chunk_overlap: Overlap between consecutive chunks
        headers_to_split_on: Markdown header rules for splitting
        separators: Custom separators for splitting within sections

    Returns:
        Dictionary containing chunk results and statistics
    """
    return chunk_document(
        collection=collection,
        doc_id=doc_id,
        parse_hash=parse_hash,
        chunk_strategy=ChunkStrategy.MARKDOWN,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        headers_to_split_on=headers_to_split_on,
        separators=separators,
        **kwargs,
    )


def chunk_fixed_size(
    collection: str,
    doc_id: str,
    parse_hash: str,
    chunk_size: Optional[int] = 1000,
    chunk_overlap: int = 0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Chunk document using fixed size strategy.

    Args:
        collection: Collection name for data isolation
        doc_id: Document ID whose parsed result to chunk
        parse_hash: Parse version hash to select parsed content
        chunk_size: Target chunk size in characters. If None, returns whole document as one chunk
        chunk_overlap: Overlap between consecutive chunks

    Returns:
        Dictionary containing chunk results and statistics
    """
    return chunk_document(
        collection=collection,
        doc_id=doc_id,
        parse_hash=parse_hash,
        chunk_strategy=ChunkStrategy.FIXED_SIZE,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs,
    )
