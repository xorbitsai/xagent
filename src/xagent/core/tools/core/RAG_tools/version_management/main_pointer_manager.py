"""Main pointer management for version control.

This module provides functionality for managing main version pointers
across different processing stages (parse, chunk, embed).

Phase 1A Part 2: Refactored to use MainPointerStore abstraction layer.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..core.exceptions import MainPointerError
from ..storage.factory import get_main_pointer_store

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..kb import KBVersionCompatibilityFacade


def _get_version_compatibility_facade() -> "KBVersionCompatibilityFacade":
    from ..kb import get_kb_coordinator

    return get_kb_coordinator().version_compatibility


def _normalize_model_tag(model_tag: Optional[str]) -> str:
    """Normalize model_tag to empty string if None."""
    return model_tag if model_tag is not None else ""


def get_main_pointer(
    collection: str, doc_id: str, step_type: str, model_tag: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    return _get_version_compatibility_facade().get_main_pointer(
        collection=collection,
        doc_id=doc_id,
        step_type=step_type,
        model_tag=model_tag,
    )


def _get_main_pointer_impl(
    collection: str, doc_id: str, step_type: str, model_tag: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Get the main pointer for a specific document and stage.

    Args:
        collection: Collection name
        doc_id: Document ID
        step_type: Processing stage type (parse, chunk, embed)
        model_tag: Model tag for embed stage (optional)

    Returns:
        Main pointer data or None if not found

    Raises:
        MainPointerError: If there's an error retrieving the pointer
    """
    try:
        store = get_main_pointer_store()
        return store.get_main_pointer(
            collection=collection,
            doc_id=doc_id,
            step_type=step_type,
            model_tag=model_tag,
            user_id=None,
        )
    except Exception as e:
        raise MainPointerError(f"Failed to get main pointer: {e}") from e


def set_main_pointer(
    lancedb_dir: str,  # Kept for backward compatibility
    collection: str,
    doc_id: str,
    step_type: str,
    semantic_id: str,
    technical_id: str,
    model_tag: Optional[str] = None,
    operator: Optional[str] = None,
) -> None:
    _get_version_compatibility_facade().set_main_pointer(
        lancedb_dir=lancedb_dir,
        collection=collection,
        doc_id=doc_id,
        step_type=step_type,
        semantic_id=semantic_id,
        technical_id=technical_id,
        model_tag=model_tag,
        operator=operator,
    )


def _set_main_pointer_impl(
    lancedb_dir: str,  # Kept for backward compatibility
    collection: str,
    doc_id: str,
    step_type: str,
    semantic_id: str,
    technical_id: str,
    model_tag: Optional[str] = None,
    operator: Optional[str] = None,
) -> None:
    """Set or update the main pointer for a specific document and stage.

    Args:
        lancedb_dir: Directory for LanceDB (unused, kept for backward compatibility)
        collection: Collection name
        doc_id: Document ID
        step_type: Processing stage type (parse, chunk, embed)
        semantic_id: Semantic identifier for the version
        technical_id: Technical identifier (hash) for the version
        model_tag: Model tag for embed stage (optional)
        operator: Operator who made the change (optional)

    Raises:
        MainPointerError: If there's an error setting the pointer
    """
    try:
        store = get_main_pointer_store()
        store.set_main_pointer(
            collection=collection,
            doc_id=doc_id,
            step_type=step_type,
            semantic_id=semantic_id,
            technical_id=technical_id,
            model_tag=model_tag,
            operator=operator,
            user_id=None,
        )
        logger.info(
            "Set main pointer for %s/%s/%s to %s (semantic: %s)",
            collection,
            doc_id,
            step_type,
            technical_id,
            semantic_id,
        )
    except Exception as e:
        raise MainPointerError(f"Failed to set main pointer: {e}") from e


def list_main_pointers(
    collection: str, doc_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    return _get_version_compatibility_facade().list_main_pointers(
        collection=collection,
        doc_id=doc_id,
    )


def _list_main_pointers_impl(
    collection: str, doc_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """List main pointers for a collection and optionally a specific document.

    Args:
        collection: Collection name
        doc_id: Document ID (optional, if None lists all documents in collection)

    Returns:
        List of main pointer data

    Raises:
        MainPointerError: If there's an error listing pointers
    """
    try:
        store = get_main_pointer_store()
        return store.list_main_pointers(
            collection=collection,
            doc_id=doc_id,
            user_id=None,
            limit=100,
        )
    except Exception as e:
        raise MainPointerError(f"Failed to list main pointers: {e}") from e


def delete_main_pointer(
    collection: str, doc_id: str, step_type: str, model_tag: Optional[str] = None
) -> bool:
    return _get_version_compatibility_facade().delete_main_pointer(
        collection=collection,
        doc_id=doc_id,
        step_type=step_type,
        model_tag=model_tag,
    )


def _delete_main_pointer_impl(
    collection: str, doc_id: str, step_type: str, model_tag: Optional[str] = None
) -> bool:
    """Delete a main pointer.

    Behavior note (backward compatibility):
        When ``model_tag`` is ``None`` (normalized to empty string), this function deletes
        pointers whose ``model_tag`` is either ``''`` OR ``NULL``.

    Args:
        collection: Collection name
        doc_id: Document ID
        step_type: Processing stage type
        model_tag: Model tag for embed stage (optional)

    Returns:
        True if pointer was deleted, False if not found

    Raises:
        MainPointerError: If there's an error deleting the pointer
    """
    try:
        store = get_main_pointer_store()
        result = store.delete_main_pointer(
            collection=collection,
            doc_id=doc_id,
            step_type=step_type,
            model_tag=model_tag,
            user_id=None,
        )
        if result:
            logger.info(
                "Deleted main pointer for %s/%s/%s", collection, doc_id, step_type
            )
        return result
    except Exception as e:
        raise MainPointerError(f"Failed to delete main pointer: {e}") from e
