"""Main entry point for document parsing.

This module provides the main parse_document function that orchestrates
document parsing by calling the unified document parsing tool.
"""

import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ......core.tools.core.document_parser import (
    DocumentCapabilities,
    DocumentParseArgs,
)
from ......core.tools.core.document_parser import parse_document as core_parse_document
from ..core.exceptions import (
    ConfigurationError,
    DatabaseOperationError,
    DocumentNotFoundError,
    DocumentValidationError,
)
from ..core.schemas import (
    ParseDocumentRequest,
    ParseDocumentResponse,
    ParsedParagraph,
    ParseMethod,
)
from ..storage.factory import get_vector_index_store
from ..utils.hash_utils import compute_parse_hash, get_parse_params_whitelist

if TYPE_CHECKING:
    from ..kb import KBLegacyStepCompatibilityFacade
    from ..kb.collection_handle import LanceDBCollectionHandle

logger = logging.getLogger(__name__)


def _get_legacy_step_compatibility_facade() -> "KBLegacyStepCompatibilityFacade":
    """Return the coordinator-owned legacy step compatibility facade."""
    from ..kb import get_kb_coordinator

    return get_kb_coordinator().legacy_step_compatibility


def parse_document(
    collection: str,
    doc_id: str,
    parse_method: ParseMethod,
    params: Optional[Dict[str, Any]] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
    progress_callback: Optional[Any] = None,
    source_path_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Parse a document using the specified method."""
    return _get_legacy_step_compatibility_facade().parse_document(
        collection=collection,
        doc_id=doc_id,
        parse_method=parse_method,
        params=params,
        user_id=user_id,
        is_admin=is_admin,
        progress_callback=progress_callback,
        source_path_override=source_path_override,
    )


def _parse_document_impl(
    collection: str,
    doc_id: str,
    parse_method: ParseMethod,
    params: Optional[Dict[str, Any]] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
    progress_callback: Optional[Any] = None,
    source_path_override: Optional[str] = None,
    *,
    handle: "LanceDBCollectionHandle",
) -> Dict[str, Any]:
    """
    Parse a document using the specified method.

    Args:
        collection: Collection name for data isolation
        doc_id: Document ID to parse
        parse_method: Parsing method to use
        params: Optional parameters for parsing
        user_id: Optional user ID for ownership tracking
        is_admin: Whether the user has admin privileges
        progress_callback: Optional callback for progress updates

    Returns:
        Dictionary containing parse results
    """
    if params is None:
        params = {}

    request = ParseDocumentRequest(
        collection=collection,
        doc_id=doc_id,
        parse_method=parse_method,
        source_path_override=source_path_override,
        params=params,
        user_id=user_id,
        is_admin=is_admin,
    )

    response = asyncio.run(
        _parse_document_internal(request, progress_callback, handle=handle)
    )

    return response.model_dump()


async def _parse_document_internal(
    request: ParseDocumentRequest,
    progress_callback: Optional[Any] = None,
    *,
    handle: "LanceDBCollectionHandle",
) -> ParseDocumentResponse:
    """
    Internal document parsing logic.
    """
    # Enable detailed timing (controlled by environment variable)
    enable_timing = os.environ.get("PARSE_DETAILED_TIMING", "0").lower() in (
        "1",
        "true",
        "yes",
    )
    timing_data: Optional[Dict[str, float]] = {} if enable_timing else None

    if enable_timing:
        assert timing_data is not None  # Type guard for mypy
        timing_data["start"] = time.perf_counter()
        logger.debug("\n" + "=" * 60)
        logger.debug(
            "[PARSE TIMING] Starting document parsing: doc_id=%s", request.doc_id
        )
        logger.debug("=" * 60)

    collection = request.collection
    doc_id = request.doc_id
    parse_method = request.parse_method
    params = request.params or {}
    user_id = request.user_id
    is_admin = request.is_admin

    logger.info("Starting document parsing: doc_id=%s, method=%s", doc_id, parse_method)

    document = _get_document_from_db(collection, doc_id, user_id, is_admin)
    if not document:
        raise DocumentNotFoundError(f"Document not found: {doc_id}")

    source_path = request.source_path_override or document["source_path"]
    file_type = document["file_type"]
    logger.info("Found document: %s", source_path)

    _validate_parse_params(parse_method, params)

    parse_hash = compute_parse_hash(str(parse_method), params)
    logger.info("Computed parse hash: %s", parse_hash)

    if handle.parse_exists(doc_id, parse_hash, user_id=user_id, is_admin=is_admin):
        existing_paragraphs = handle.read_parse_paragraphs(
            doc_id, parse_hash, user_id=user_id, is_admin=is_admin
        )
        logger.info(
            "Parse record already exists for doc_id=%s, parse_hash=%s",
            doc_id,
            parse_hash,
        )
        return ParseDocumentResponse(
            doc_id=doc_id,
            parse_hash=parse_hash,
            paragraphs=existing_paragraphs,
            written=False,
        )

    # --- Refactored Parsing Logic ---
    try:
        # 1. Call the unified core document parser
        # If parse_method is DEFAULT, use None to let the parser auto-route based on file type
        if parse_method == ParseMethod.DEFAULT:
            parser_name = None  # Let auto-router decide based on file extension
        else:
            parser_name = str(parse_method)

        # Merge params with doc_id for parsers that need it (e.g., deepdoc for PDF images)
        parse_params = {**(params or {}), "doc_id": doc_id}
        tool_args = DocumentParseArgs(
            file_path=source_path,
            parser_name=parser_name,
            # This uses default capabilities, can be expanded to take from params
            capabilities=DocumentCapabilities(),
            parser_kwargs=parse_params,
        )

        if enable_timing:
            assert timing_data is not None  # Type guard for mypy
            timing_data["ocr_start"] = time.perf_counter()
            logger.debug("[PARSE TIMING] Starting OCR processing...")

        parse_result = await core_parse_document(tool_args, progress_callback)

        if enable_timing:
            assert timing_data is not None  # Type guard for mypy
            timing_data["ocr_end"] = time.perf_counter()
            ocr_time = timing_data["ocr_end"] - timing_data["ocr_start"]
            logger.debug(
                "[PARSE TIMING] OCR processing completed: %.3f seconds", ocr_time
            )

        # 2. Convert the rich ParseResult back to the RAG pipeline's ParsedParagraph list
        if enable_timing:
            assert timing_data is not None  # Type guard for mypy
            timing_data["convert_start"] = time.perf_counter()
            logger.debug(
                "[PARSE TIMING] Starting conversion of ParseResult to Paragraphs..."
            )

        paragraphs = _convert_parse_result_to_paragraphs(parse_result)

        if enable_timing:
            assert timing_data is not None  # Type guard for mypy
            timing_data["convert_end"] = time.perf_counter()
            convert_time = timing_data["convert_end"] - timing_data["convert_start"]
            logger.debug(
                "[PARSE TIMING] Conversion completed: %.3f seconds (paragraphs=%s)",
                convert_time,
                len(paragraphs),
            )

    except Exception as e:
        logger.error("Document parsing failed: %s", e)
        raise DocumentValidationError(f"Parsing failed: {e}") from e

    # --- End of Refactored Logic ---

    if enable_timing:
        assert timing_data is not None  # Type guard for mypy
        timing_data["enrich_start"] = time.perf_counter()
        logger.debug("[PARSE TIMING] Starting metadata enrichment...")

    enriched_paragraphs = []
    for paragraph in paragraphs:
        # Start with parser metadata, then override with authoritative values
        enriched_metadata = {
            **paragraph.metadata,
            # Persist the canonical path from the document row, not the physical
            # path actually read (which for staged ingestion is a transient file
            # removed after publish). Durable metadata must keep the canonical path.
            "source": document["source_path"],
            "file_type": file_type,  # Use file_type from database (without dot)
            "parse_method": str(parse_method),
            "parser": f"local:{parse_method}@v1.0.0",
        }
        enriched_paragraphs.append(
            ParsedParagraph(text=paragraph.text, metadata=enriched_metadata)
        )

    if enable_timing:
        assert timing_data is not None  # Type guard for mypy
        timing_data["enrich_end"] = time.perf_counter()
        enrich_time = timing_data["enrich_end"] - timing_data["enrich_start"]
        logger.debug(
            "[PARSE TIMING] Metadata enrichment completed: %.3f seconds (paragraphs=%s)",
            enrich_time,
            len(enriched_paragraphs),
        )

    if enable_timing:
        assert timing_data is not None  # Type guard for mypy
        timing_data["db_write_start"] = time.perf_counter()
        logger.debug("[PARSE TIMING] Starting database write...")

    # handle.write_parse already raises DatabaseOperationError on failure;
    # no outer wrap here to avoid double-wrapping the message.
    written = handle.write_parse(
        doc_id,
        parse_hash,
        str(parse_method),
        params,
        enriched_paragraphs,
        user_id=user_id,
    )

    if enable_timing:
        assert timing_data is not None  # Type guard for mypy
        timing_data["db_write_end"] = time.perf_counter()
        db_write_time = timing_data["db_write_end"] - timing_data["db_write_start"]
        logger.debug(
            "[PARSE TIMING] Database write completed: %.3f seconds", db_write_time
        )

    logger.info(
        "Document parsing completed: doc_id=%s, paragraphs=%s",
        doc_id,
        len(enriched_paragraphs),
    )

    if enable_timing:
        assert timing_data is not None  # Type guard for mypy
        timing_data["end"] = time.perf_counter()
        total_time = timing_data["end"] - timing_data["start"]

        # Calculate time spent in each stage
        ocr_time = timing_data.get("ocr_end", timing_data["end"]) - timing_data.get(
            "ocr_start", timing_data["start"]
        )
        convert_time = timing_data.get(
            "convert_end", timing_data.get("ocr_end", timing_data["end"])
        ) - timing_data.get(
            "convert_start", timing_data.get("ocr_end", timing_data["start"])
        )
        enrich_time = timing_data.get(
            "enrich_end", timing_data.get("convert_end", timing_data["end"])
        ) - timing_data.get(
            "enrich_start", timing_data.get("convert_end", timing_data["start"])
        )
        db_write_time = timing_data.get(
            "db_write_end", timing_data["end"]
        ) - timing_data.get(
            "db_write_start", timing_data.get("enrich_end", timing_data["end"])
        )

        logger.debug("\n" + "=" * 60)
        logger.debug("[PARSE TIMING] Document parsing time breakdown")
        logger.debug("=" * 60)
        logger.debug("  Total time: %.3f seconds", total_time)
        logger.debug(
            "  - OCR processing: %.3f seconds (%.1f%%)",
            ocr_time,
            ocr_time / total_time * 100,
        )
        logger.debug(
            "  - Data conversion: %.3f seconds (%.1f%%)",
            convert_time,
            convert_time / total_time * 100,
        )
        logger.debug(
            "  - Metadata enrichment: %.3f seconds (%.1f%%)",
            enrich_time,
            enrich_time / total_time * 100,
        )
        logger.debug(
            "  - Database write: %.3f seconds (%.1f%%)",
            db_write_time,
            db_write_time / total_time * 100,
        )
        logger.debug("=" * 60 + "\n")

    return ParseDocumentResponse(
        doc_id=doc_id,
        parse_hash=parse_hash,
        paragraphs=enriched_paragraphs,
        written=written,
    )


def _convert_parse_result_to_paragraphs(result: Any) -> List[ParsedParagraph]:
    """Converts a ParseResult object into a list of ParsedParagraphs."""
    paragraphs = []
    if result.text_segments:
        for seg in result.text_segments:
            paragraphs.append(ParsedParagraph(text=seg.text, metadata=seg.metadata))
    if result.tables:
        for tbl in result.tables:
            # Use HTML content as text for tables
            text = tbl.html or ""
            paragraphs.append(ParsedParagraph(text=text, metadata=tbl.metadata))
    if result.figures:
        for fig in result.figures:
            # Use caption as text for figures
            paragraphs.append(ParsedParagraph(text=fig.text, metadata=fig.metadata))
    return paragraphs


def _get_document_from_db(
    collection: str, doc_id: str, user_id: Optional[int] = None, is_admin: bool = False
) -> Optional[Any]:
    """Get document from database by doc_id using abstraction layer.

    Uses direct iter_batches lookup with retry to handle transient
    LanceDB read-after-write latency. Avoids count_rows_or_zero which
    silently swallows DatabaseOperationError, hiding the real failure.
    """
    vector_store = get_vector_index_store()
    query_filters = {"collection": collection, "doc_id": doc_id}

    max_retries = 3
    for attempt in range(max_retries):
        try:
            for batch in vector_store.iter_batches(
                table_name="documents",
                filters=query_filters,
                user_id=user_id,
                is_admin=is_admin,
            ):
                batch_df = batch.to_pandas()
                for _, row in batch_df.iterrows():
                    return row.to_dict()

            # No rows found — retry if attempts remain
            if attempt < max_retries - 1:
                logger.debug(
                    "Document %s not found in documents table, retrying (%d/%d)",
                    doc_id,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(0.1 * (attempt + 1))
                continue
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                logger.debug(
                    "Error looking up document %s, retrying (%d/%d): %s",
                    doc_id,
                    attempt + 1,
                    max_retries,
                    e,
                )
                time.sleep(0.1 * (attempt + 1))
                continue
            logger.error(
                "Failed to get document from database after %d retries: %s",
                max_retries,
                e,
            )
            raise DatabaseOperationError(f"Failed to get document: {e}") from e

    return None


def _validate_parse_params(parse_method: ParseMethod, params: Dict[str, Any]) -> None:
    """Validate parsing parameters against whitelist."""
    valid_methods = set(ParseMethod)
    if parse_method not in valid_methods:
        raise DocumentValidationError(f"Unsupported parse method: {parse_method}")
    try:
        whitelist = get_parse_params_whitelist(str(parse_method))
        for key in params:
            if key not in whitelist:
                raise DocumentValidationError(
                    f"Invalid parameter '{key}' for parse method '{parse_method}'"
                )
    except Exception as e:
        if isinstance(e, DocumentValidationError):
            raise
        raise ConfigurationError(f"Parameter validation failed: {e}") from e
