"""Main entry point for document parsing.

This module provides the main parse_document function that orchestrates
document parsing by calling the unified document parsing tool.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import pandas as pd

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
from ..utils.paragraph_page_utils import collect_pages_from_paragraphs

logger = logging.getLogger(__name__)

# Keys allowed in persisted ``params_json`` but never in user/API parse requests.
SYSTEM_ONLY_PARSE_PARAM_KEYS: frozenset[str] = frozenset({"_derived"})


def parse_document(
    collection: str,
    doc_id: str,
    parse_method: ParseMethod,
    params: Optional[Dict[str, Any]] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
    progress_callback: Optional[Any] = None,
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
        params=params,
        user_id=user_id,
        is_admin=is_admin,
    )

    response = asyncio.run(_parse_document_internal(request, progress_callback))

    return response.model_dump()


async def _parse_document_internal(
    request: ParseDocumentRequest,
    progress_callback: Optional[Any] = None,
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

    source_path = document["source_path"]
    file_type = document["file_type"]
    logger.info("Found document: %s", source_path)

    _validate_parse_params(parse_method, params)

    user_parse_params = strip_system_only_parse_params(params)
    parse_hash = compute_parse_hash(str(parse_method), user_parse_params)
    logger.info("Computed parse hash: %s", parse_hash)

    if _parse_exists(collection, doc_id, parse_hash, user_id, is_admin):
        existing_paragraphs = _get_existing_parse_content(
            collection, doc_id, parse_hash, user_id, is_admin
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

        parse_params = build_parser_kwargs(params, doc_id=doc_id)
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
            "source": source_path,
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

    # Derive basic page statistics for downstream consumers (e.g. UI).
    # Use the same union as chunk ``spanning_pages`` (page_number + DeepDoc bbox pages).
    page_dicts = [
        {"text": paragraph.text, "metadata": dict(paragraph.metadata)}
        for paragraph in enriched_paragraphs
    ]
    unique_pages = collect_pages_from_paragraphs(page_dicts)
    page_count = len(unique_pages)
    page_stats = {"page_count": page_count, "page_numbers": unique_pages}

    try:
        written = _write_parse_to_db(
            collection,
            doc_id,
            parse_hash,
            str(parse_method),
            user_parse_params,
            enriched_paragraphs,
            user_id,
            page_stats=page_stats,
        )
    except Exception as e:
        raise DatabaseOperationError(f"Database write failed: {e}") from e

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


def strip_system_only_parse_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return user/parser configuration keys only (excludes storage metadata)."""
    return {k: v for k, v in params.items() if k not in SYSTEM_ONLY_PARSE_PARAM_KEYS}


def build_parser_kwargs(params: Dict[str, Any], *, doc_id: str) -> Dict[str, Any]:
    """Build kwargs for the core document parser without system-only keys."""
    return {**strip_system_only_parse_params(params), "doc_id": doc_id}


def _validate_parse_params(parse_method: ParseMethod, params: Dict[str, Any]) -> None:
    """Validate user-supplied parsing parameters against the method whitelist."""
    valid_methods = set(ParseMethod)
    if parse_method not in valid_methods:
        raise DocumentValidationError(f"Unsupported parse method: {parse_method}")
    try:
        whitelist = get_parse_params_whitelist(str(parse_method))
        for key in params:
            if key in SYSTEM_ONLY_PARSE_PARAM_KEYS:
                raise DocumentValidationError(
                    f"System-only parameter '{key}' cannot be set in parse requests"
                )
            if key not in whitelist:
                raise DocumentValidationError(
                    f"Invalid parameter '{key}' for parse method '{parse_method}'"
                )
    except Exception as e:
        if isinstance(e, DocumentValidationError):
            raise
        raise ConfigurationError(f"Parameter validation failed: {e}") from e


def _validate_persisted_parse_params(
    parse_method: ParseMethod, params: Dict[str, Any]
) -> None:
    """Validate persisted ``params_json`` (user keys strict; ``_derived`` allowed)."""
    _validate_parse_params(parse_method, strip_system_only_parse_params(params))


def _parse_exists(
    collection: str,
    doc_id: str,
    parse_hash: str,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> bool:
    """Check if parse record already exists using abstraction layer.

    Args:
        collection: Collection name
        doc_id: Document ID
        parse_hash: Parse hash to check
        user_id: Optional user ID for filtering (for multi-tenancy)
        is_admin: Whether user has admin privileges

    Returns:
        True if parse record exists and is accessible to the user
    """
    try:
        vector_store = get_vector_index_store()
        query_filters = {
            "collection": collection,
            "doc_id": doc_id,
            "parse_hash": parse_hash,
        }
        return bool(
            vector_store.count_rows_or_zero(
                "parses", filters=query_filters, user_id=user_id, is_admin=is_admin
            )
            > 0
        )
    except Exception as e:
        raise DatabaseOperationError(f"Database query failed: {e}") from e


def _get_existing_parse_content(
    collection: str,
    doc_id: str,
    parse_hash: str,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> List[ParsedParagraph]:
    """Get existing parse content from database using abstraction layer.

    Args:
        collection: Collection name
        doc_id: Document ID
        parse_hash: Parse hash to retrieve
        user_id: Optional user ID for filtering (for multi-tenancy)
        is_admin: Whether user has admin privileges

    Returns:
        List of parsed paragraphs if found and accessible, empty list otherwise
    """
    try:
        vector_store = get_vector_index_store()
        query_filters = {
            "collection": collection,
            "doc_id": doc_id,
            "parse_hash": parse_hash,
        }

        if (
            vector_store.count_rows_or_zero(
                "parses", filters=query_filters, user_id=user_id, is_admin=is_admin
            )
            == 0
        ):
            return []

        # Use iter_batches to load the parse content
        for batch in vector_store.iter_batches(
            table_name="parses",
            filters=query_filters,
            user_id=user_id,
            is_admin=is_admin,
        ):
            batch_df = batch.to_pandas()
            for _, row in batch_df.iterrows():
                record = row.to_dict()
                parsed_content = record.get("parsed_content")
                if not parsed_content:
                    continue

                data = json.loads(parsed_content)
                paragraphs = []
                for item in data:
                    paragraphs.append(
                        ParsedParagraph(
                            text=item.get("text", ""),
                            metadata=item.get("metadata", {}),
                        )
                    )
                return paragraphs

        return []

    except Exception as e:
        logger.error("Failed to read parse content: %s", e)
        raise DatabaseOperationError(f"Failed reading parse content: {e}") from e


def _write_parse_to_db(
    collection: str,
    doc_id: str,
    parse_hash: str,
    parse_method: str,
    params: Dict[str, Any],
    paragraphs: List[ParsedParagraph],
    user_id: Optional[int] = None,
    page_stats: Optional[Dict[str, Any]] = None,
) -> bool:
    """Write parse record to database using abstraction layer."""
    enable_timing = os.environ.get("PARSE_DETAILED_TIMING", "0").lower() in (
        "1",
        "true",
        "yes",
    )

    try:
        vector_store = get_vector_index_store()

        if enable_timing:
            serialize_start = time.perf_counter()
            logger.debug(
                "[PARSE TIMING]    - Starting serialization of paragraphs (%s items)...",
                len(paragraphs),
            )

        paragraphs_data = [para.model_dump() for para in paragraphs]

        if enable_timing:
            serialize_end = time.perf_counter()
            serialize_time = serialize_end - serialize_start
            logger.debug(
                "[PARSE TIMING]    - Serialization completed: %.3f seconds",
                serialize_time,
            )
            json_start = time.perf_counter()
            logger.debug("[PARSE TIMING]    - Starting JSON serialization...")

        parsed_content = json.dumps(paragraphs_data, ensure_ascii=False)

        if enable_timing:
            json_end = time.perf_counter()
            json_time = json_end - json_start
            json_size_mb = len(parsed_content.encode("utf-8")) / (1024 * 1024)
            logger.debug(
                "[PARSE TIMING]    - JSON serialization completed: %.3f seconds (size: %.2f MB)",
                json_time,
                json_size_mb,
            )
            db_op_start = time.perf_counter()
            logger.debug(
                "[PARSE TIMING]    - Starting database operation (upsert_parses)..."
            )

        params_for_storage = dict(params)
        if page_stats:
            params_for_storage["_derived"] = {"page_stats": page_stats}

        parse_record = {
            "collection": collection,
            "doc_id": doc_id,
            "parse_hash": parse_hash,
            "parser": f"local:{parse_method}@v1.0.0",
            "created_at": pd.Timestamp.now(tz="UTC"),
            "params_json": json.dumps(params_for_storage, ensure_ascii=False),
            "parsed_content": parsed_content,
            "user_id": user_id,
        }

        # Use abstraction layer for upsert
        vector_store.upsert_parses([parse_record])

        if enable_timing:
            db_op_end = time.perf_counter()
            db_op_time = db_op_end - db_op_start
            logger.debug(
                "[PARSE TIMING]    - Database operation completed: %.3f seconds",
                db_op_time,
            )

        logger.info(
            "Parse record written to database: doc_id=%s, parse_hash=%s",
            doc_id,
            parse_hash,
        )
        return True
    except Exception as e:
        raise DatabaseOperationError(f"Database write failed: {e}") from e
