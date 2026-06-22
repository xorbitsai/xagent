"""Functions for displaying parse results with pagination support.

This module provides functions to retrieve and format parse results
from the database for display purposes.
"""

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ..core.exceptions import DatabaseOperationError, DocumentNotFoundError
from ..core.schemas import (
    ParsedElementDisplay,
    ParsedFigureDisplay,
    ParsedTableDisplay,
    ParsedTextSegmentDisplay,
)

if TYPE_CHECKING:
    from ..kb import KBParseDisplayCompatibilityFacade
    from ..kb.collection_handle import LanceDBCollectionHandle

logger = logging.getLogger(__name__)


def _get_parse_display_compatibility_facade() -> "KBParseDisplayCompatibilityFacade":
    """Return the coordinator-owned parse display compatibility facade."""
    from ..kb import get_kb_coordinator

    return get_kb_coordinator().parse_display_compatibility


def reconstruct_parse_result_from_db(
    collection: str,
    doc_id: str,
    parse_hash: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Reconstruct ParseResult-like structure from database using abstraction layer.

    Args:
        collection: Collection name
        doc_id: Document ID
        parse_hash: Optional parse hash to filter. If None, uses the latest parse
            (by created_at desc).
        user_id: Optional user ID for multi-tenancy filtering. If provided with
            is_admin=False, only parses owned by this user are visible.
        is_admin: If True, user_id filter is not applied (admin sees all).

    Returns:
        Tuple of (elements, parse_hash)
        elements is a list of dictionaries with 'type', 'text'/'html', and 'metadata' keys.
    """
    return _get_parse_display_compatibility_facade().reconstruct_parse_result_from_db(
        collection,
        doc_id,
        parse_hash=parse_hash,
        user_id=user_id,
        is_admin=is_admin,
    )


def _reconstruct_parse_result_from_db_impl(
    collection: str,
    doc_id: str,
    parse_hash: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
    *,
    handle: "LanceDBCollectionHandle",
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Implementation for reconstruct_parse_result_from_db.

    The handle owns the storage read and latest-parse selection; this helper
    keeps the legacy ``DocumentNotFoundError`` mapping, JSON-corruption
    handling, and the display element conversion.
    """
    try:
        # The handle owns the parse read + latest-by-created_at selection.
        record = handle.read_latest_parse_record(
            doc_id, parse_hash=parse_hash, user_id=user_id, is_admin=is_admin
        )
        if record is None:
            if parse_hash:
                raise DocumentNotFoundError(
                    f"Parse result not found: doc_id={doc_id}, parse_hash={parse_hash}"
                )
            raise DocumentNotFoundError(
                f"No parse results found for document: doc_id={doc_id}"
            )

        actual_parse_hash = record.parse_hash

        parsed_content = record.parsed_content
        if not parsed_content:
            logger.warning("Empty parsed_content for doc_id=%s", doc_id)
            return ([], actual_parse_hash)

        # Parse JSON string with error handling for data corruption
        try:
            data = json.loads(parsed_content)
        except json.JSONDecodeError as e:
            logger.error("Failed to decode parsed_content for doc_id=%s: %s", doc_id, e)
            raise DatabaseOperationError(
                f"Document parse data is corrupted for doc_id={doc_id}"
            )

        # Reconstruct unified elements list
        elements = []

        for item in data:
            text = item.get("text", "")
            metadata = item.get("metadata", {})
            layout_type = metadata.get("layout_type", "text")

            if layout_type == "text":
                elements.append({"type": "text", "text": text, "metadata": metadata})
            elif layout_type == "table":
                # Map text content to html field for tables
                elements.append({"type": "table", "html": text, "metadata": metadata})
            elif layout_type == "figure":
                elements.append({"type": "figure", "text": text, "metadata": metadata})
            else:
                # Unknown layout type, treat as text
                logger.debug("Unknown layout_type '%s', treating as text", layout_type)
                elements.append({"type": "text", "text": text, "metadata": metadata})

        logger.info("Reconstructed parse result: %s elements", len(elements))

        return (elements, actual_parse_hash)

    except DocumentNotFoundError:
        raise
    except Exception as e:
        logger.error("Failed to reconstruct parse result: %s", e)
        raise DatabaseOperationError(f"Failed to read parse result: {e}") from e


def paginate_parse_results(
    elements: List[Dict[str, Any]],
    page: int = 1,
    page_size: int = 20,
) -> Tuple[List[ParsedElementDisplay], Dict[str, Any]]:
    """Paginate parse results.

    Args:
        elements: List of unified element dicts
        page: Page number (1-indexed)
        page_size: Number of elements per page

    Returns:
        Tuple of (paginated_elements, pagination_info)
    """
    return _get_parse_display_compatibility_facade().paginate_parse_results(
        elements,
        page=page,
        page_size=page_size,
    )


def _paginate_parse_results_impl(
    elements: List[Dict[str, Any]],
    page: int = 1,
    page_size: int = 20,
) -> Tuple[List[ParsedElementDisplay], Dict[str, Any]]:
    """Implementation for paginate_parse_results."""
    # Validate inputs
    if page < 1:
        page = 1
    if page_size < 1:
        page_size = 20

    total_count = len(elements)
    total_pages = (total_count + page_size - 1) // page_size if total_count > 0 else 1

    # Calculate pagination
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    # Get paginated elements
    paginated_elements_dict = elements[start_idx:end_idx]

    # Convert dicts to Pydantic models
    paginated_elements: List[ParsedElementDisplay] = []
    for elem in paginated_elements_dict:
        elem_type = elem.get("type", "text")
        try:
            if elem_type == "text":
                paginated_elements.append(
                    ParsedTextSegmentDisplay(
                        type="text",
                        text=elem.get("text", ""),
                        metadata=elem.get("metadata", {}),
                    )
                )
            elif elem_type == "table":
                paginated_elements.append(
                    ParsedTableDisplay(
                        type="table",
                        html=elem.get("html", ""),
                        metadata=elem.get("metadata", {}),
                    )
                )
            elif elem_type == "figure":
                paginated_elements.append(
                    ParsedFigureDisplay(
                        type="figure",
                        text=elem.get("text", ""),
                        metadata=elem.get("metadata", {}),
                    )
                )
            else:
                # Unknown type, fallback to text
                logger.debug("Unknown element type '%s', treating as text", elem_type)
                paginated_elements.append(
                    ParsedTextSegmentDisplay(
                        type="text",
                        text=elem.get("text", ""),
                        metadata=elem.get("metadata", {}),
                    )
                )
        except Exception as e:
            logger.warning(
                "Failed to convert element to Pydantic model: %s, elem=%s", e, elem
            )
            # Fallback to text segment on conversion error
            paginated_elements.append(
                ParsedTextSegmentDisplay(
                    type="text",
                    text=elem.get("text", ""),
                    metadata=elem.get("metadata", {}),
                )
            )

    pagination_info = {
        "page": page,
        "page_size": page_size,
        "total_elements": total_count,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_previous": page > 1,
    }

    return (paginated_elements, pagination_info)
