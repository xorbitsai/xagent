"""Main entry point for document chunking.

This module provides the main chunk_document function that orchestrates
document chunking using various chunking strategies.
"""

import logging
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

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
from ..utils.hash_utils import compute_chunk_hash
from .chunk_strategies import (
    _create_chunk_record,
    apply_fixed_size_strategy,
    apply_markdown_strategy,
    apply_recursive_strategy,
    attach_media_context,
)

if TYPE_CHECKING:
    from ..kb import KBLegacyStepCompatibilityFacade
    from ..kb.collection_handle import LanceDBCollectionHandle

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


def _get_legacy_step_compatibility_facade() -> "KBLegacyStepCompatibilityFacade":
    """Return the coordinator-owned legacy step compatibility facade."""
    from ..kb import get_kb_coordinator

    return get_kb_coordinator().legacy_step_compatibility


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
    """Chunk parsed paragraphs and write to chunks table."""
    return _get_legacy_step_compatibility_facade().chunk_document(
        collection=collection,
        doc_id=doc_id,
        parse_hash=parse_hash,
        chunk_strategy=chunk_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        headers_to_split_on=headers_to_split_on,
        separators=separators,
        use_token_count=use_token_count,
        tiktoken_encoding=tiktoken_encoding,
        enable_protected_content=enable_protected_content,
        protected_patterns=protected_patterns,
        table_context_size=table_context_size,
        image_context_size=image_context_size,
        user_id=user_id,
        is_admin=is_admin,
        **kwargs,
    )


def _chunk_document_impl(
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
    *,
    handle: "LanceDBCollectionHandle",
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
        "[RAG][chunk] starting doc_id=%s strategy=%s chunk_size=%s chunk_overlap=%s "
        "use_token_count=%s enable_protected_content=%s",
        doc_id,
        chunk_strategy,
        chunk_size,
        chunk_overlap,
        use_token_count,
        enable_protected_content,
    )

    # Validate chunk parameters
    _validate_chunk_params(chunk_strategy, params)

    # Compute configuration-level hash for this chunking run
    try:
        config_hash = compute_chunk_hash("", params)
    except Exception as e:
        raise DocumentValidationError(f"Failed to compute config_hash: {e}") from e

    logger.info("Computed chunk config hash: %s", config_hash)

    # Combined existence + read in a single query: an empty result means no
    # reusable chunks exist for this (parse_hash, config_hash).
    existing_chunks = handle.read_existing_chunks(
        doc_id, parse_hash, config_hash, user_id=user_id, is_admin=is_admin
    )

    if existing_chunks:
        logger.info(
            "Chunk record already exists for doc_id=%s, parse_hash=%s, config_hash=%s",
            doc_id,
            parse_hash,
            config_hash,
        )
        return {
            "doc_id": doc_id,
            "parse_hash": parse_hash,
            "chunk_count": len(existing_chunks),
            "stats": _compute_stats(existing_chunks),
            "created": False,
        }

    # Load parsed content from database
    paragraphs = handle.read_parse_paragraph_dicts(
        doc_id, parse_hash, user_id=user_id, is_admin=is_admin
    )
    if not paragraphs:
        raise DocumentNotFoundError(
            f"No parsed content found for doc_id={doc_id}, parse_hash={parse_hash}"
        )
    _para_chars = sum(len(p.get("text") or "") for p in paragraphs)
    logger.info(
        "[RAG][chunk] loaded parsed paragraphs doc_id=%s paragraph_count=%s "
        "total_chars=%s",
        doc_id,
        len(paragraphs),
        _para_chars,
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
            logger.error("Document chunking failed: %s", e)
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
        written = handle.write_chunks(
            doc_id,
            parse_hash,
            config_hash,
            params,
            indexed_chunks,
            user_id=user_id,
        )
    except Exception as e:
        logger.error("Failed to write chunks to database: %s", e)
        raise DatabaseOperationError(f"Database write failed: {e}") from e

    _lengths = [len(c.get("text") or "") for c in indexed_chunks]
    if _lengths:
        logger.info(
            "[RAG][chunk] completed doc_id=%s chunk_count=%s char_len min=%s max=%s "
            "sum=%s avg=%.1f",
            doc_id,
            len(_lengths),
            min(_lengths),
            max(_lengths),
            sum(_lengths),
            sum(_lengths) / len(_lengths),
        )
    else:
        logger.info(
            "[RAG][chunk] completed doc_id=%s chunk_count=0 (no chunk texts)",
            doc_id,
        )
    logger.info(
        "Document chunking completed: doc_id=%s, chunks=%s", doc_id, len(indexed_chunks)
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
    """Chunk document using recursive character splitting strategy."""
    return _get_legacy_step_compatibility_facade().chunk_recursive(
        collection=collection,
        doc_id=doc_id,
        parse_hash=parse_hash,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        **kwargs,
    )


def _chunk_recursive_impl(
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
    return _chunk_document_impl(
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
    """Chunk document using markdown header-based strategy."""
    return _get_legacy_step_compatibility_facade().chunk_markdown(
        collection=collection,
        doc_id=doc_id,
        parse_hash=parse_hash,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        headers_to_split_on=headers_to_split_on,
        separators=separators,
        **kwargs,
    )


def _chunk_markdown_impl(
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
    return _chunk_document_impl(
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
    """Chunk document using fixed size strategy."""
    return _get_legacy_step_compatibility_facade().chunk_fixed_size(
        collection=collection,
        doc_id=doc_id,
        parse_hash=parse_hash,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs,
    )


def _chunk_fixed_size_impl(
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
    return _chunk_document_impl(
        collection=collection,
        doc_id=doc_id,
        parse_hash=parse_hash,
        chunk_strategy=ChunkStrategy.FIXED_SIZE,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs,
    )
