"""Helpers for tracking document ingestion status.

This module provides functions to track, load, and manage the ingestion status
of documents being processed in the RAG pipeline.

Phase 1A Part 2: Refactored to use IngestionStatusStore abstraction layer.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..storage.factory import get_ingestion_status_store

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..kb import KBCoreManagementCompatibilityFacade


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _get_management_facade() -> "KBCoreManagementCompatibilityFacade":
    from ..kb import get_kb_coordinator

    return get_kb_coordinator().management


def write_ingestion_status(
    collection: str,
    doc_id: str,
    *,
    status: str,
    message: Optional[str] = None,
    parse_hash: Optional[str] = None,
    user_id: Optional[int] = None,
) -> None:
    """Persist the latest ingestion status for a document.

    This function writes the current status of a document's ingestion process
    to the ingestion_runs table using the storage abstraction layer.

    Args:
        collection: Name of the collection
        doc_id: Unique identifier for the document
        status: Current status value (e.g., 'pending', 'processing', 'success', 'failed')
        message: Optional status message or error description
        parse_hash: Optional hash of the parsed document for change detection
        user_id: Optional user ID for multi-tenancy support

    Returns:
        None

    Raises:
        DatabaseOperationError: If write operation fails.
    """
    _get_management_facade().write_ingestion_status(
        collection=collection,
        doc_id=doc_id,
        status=status,
        message=message,
        parse_hash=parse_hash,
        user_id=user_id,
    )


def _write_ingestion_status_impl(
    collection: str,
    doc_id: str,
    *,
    status: str,
    message: Optional[str] = None,
    parse_hash: Optional[str] = None,
    user_id: Optional[int] = None,
) -> None:
    store = get_ingestion_status_store()
    store.write_ingestion_status(
        collection=collection,
        doc_id=doc_id,
        status=status,
        message=message,
        parse_hash=parse_hash,
        user_id=user_id,
    )


def load_ingestion_status(
    collection: Optional[str] = None,
    doc_id: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> List[Dict[str, Any]]:
    """Return ingestion status records filtered by collection/doc.

    This function retrieves ingestion status records from the ingestion_runs
    table using the storage abstraction layer, with optional filtering by
    collection and document.

    Args:
        collection: Optional collection name to filter by
        doc_id: Optional document ID to filter by
        user_id: Optional user ID for multi-tenancy filtering
        is_admin: Whether the user has admin privileges (bypasses filtering)

    Returns:
        List of dictionaries containing ingestion status records with keys:
        - collection: Collection name
        - doc_id: Document identifier
        - status: Current status
        - message: Status message if any
        - parse_hash: Parse hash if any
        - created_at: Creation timestamp
        - updated_at: Last update timestamp
        - user_id: User ID who owns the document

    Raises:
        DatabaseOperationError: If read operation fails.
    """
    return _get_management_facade().load_ingestion_status(
        collection=collection,
        doc_id=doc_id,
        user_id=user_id,
        is_admin=is_admin,
    )


def _load_ingestion_status_impl(
    collection: Optional[str] = None,
    doc_id: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> List[Dict[str, Any]]:
    store = get_ingestion_status_store()
    return store.load_ingestion_status(
        collection=collection,
        doc_id=doc_id,
        user_id=user_id,
        is_admin=is_admin,
    )


def clear_ingestion_status(
    collection: str, doc_id: str, user_id: Optional[int] = None, is_admin: bool = False
) -> None:
    """Remove stored ingestion status for a document.

    This function deletes the ingestion status record for a specific document
    from the ingestion_runs table using the storage abstraction layer.

    Args:
        collection: Name of the collection
        doc_id: Unique identifier for the document
        user_id: Optional user ID for multi-tenancy filtering
        is_admin: Whether the user has admin privileges (bypasses filtering)

    Returns:
        None

    Raises:
        DatabaseOperationError: If delete operation fails.
    """
    _get_management_facade().clear_ingestion_status(
        collection=collection,
        doc_id=doc_id,
        user_id=user_id,
        is_admin=is_admin,
    )


def _clear_ingestion_status_impl(
    collection: str, doc_id: str, user_id: Optional[int] = None, is_admin: bool = False
) -> None:
    store = get_ingestion_status_store()
    store.clear_ingestion_status(
        collection=collection,
        doc_id=doc_id,
        user_id=user_id,
        is_admin=is_admin,
    )


# ============================================================================
# Async variants (Phase 1A Option C: Hybrid approach)
# ============================================================================


async def write_ingestion_status_async(
    collection: str,
    doc_id: str,
    *,
    status: str,
    message: Optional[str] = None,
    parse_hash: Optional[str] = None,
    user_id: Optional[int] = None,
) -> None:
    """Async version of write_ingestion_status.

    Args:
        collection: Name of the collection
        doc_id: Unique identifier for the document
        status: Current status value
        message: Optional status message
        parse_hash: Optional parse hash
        user_id: Optional user ID

    Returns:
        None

    Raises:
        DatabaseOperationError: If write operation fails.
    """
    await _get_management_facade().write_ingestion_status_async(
        collection=collection,
        doc_id=doc_id,
        status=status,
        message=message,
        parse_hash=parse_hash,
        user_id=user_id,
    )


async def _write_ingestion_status_async_impl(
    collection: str,
    doc_id: str,
    *,
    status: str,
    message: Optional[str] = None,
    parse_hash: Optional[str] = None,
    user_id: Optional[int] = None,
) -> None:
    store = get_ingestion_status_store()
    await store.write_ingestion_status_async(
        collection=collection,
        doc_id=doc_id,
        status=status,
        message=message,
        parse_hash=parse_hash,
        user_id=user_id,
    )


async def load_ingestion_status_async(
    collection: Optional[str] = None,
    doc_id: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> List[Dict[str, Any]]:
    """Async version of load_ingestion_status.

    Args:
        collection: Optional collection name to filter by
        doc_id: Optional document ID to filter by
        user_id: Optional user ID for multi-tenancy filtering
        is_admin: Whether the user has admin privileges

    Returns:
        List of ingestion status records.

    Raises:
        DatabaseOperationError: If read operation fails.
    """
    return await _get_management_facade().load_ingestion_status_async(
        collection=collection,
        doc_id=doc_id,
        user_id=user_id,
        is_admin=is_admin,
    )


async def _load_ingestion_status_async_impl(
    collection: Optional[str] = None,
    doc_id: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> List[Dict[str, Any]]:
    store = get_ingestion_status_store()
    return await store.load_ingestion_status_async(
        collection=collection,
        doc_id=doc_id,
        user_id=user_id,
        is_admin=is_admin,
    )


async def clear_ingestion_status_async(
    collection: str, doc_id: str, user_id: Optional[int] = None, is_admin: bool = False
) -> None:
    """Async version of clear_ingestion_status.

    Args:
        collection: Name of the collection
        doc_id: Unique identifier for the document
        user_id: Optional user ID for multi-tenancy filtering
        is_admin: Whether the user has admin privileges

    Returns:
        None

    Raises:
        DatabaseOperationError: If delete operation fails.
    """
    await _get_management_facade().clear_ingestion_status_async(
        collection=collection,
        doc_id=doc_id,
        user_id=user_id,
        is_admin=is_admin,
    )


async def _clear_ingestion_status_async_impl(
    collection: str, doc_id: str, user_id: Optional[int] = None, is_admin: bool = False
) -> None:
    store = get_ingestion_status_store()
    await store.clear_ingestion_status_async(
        collection=collection,
        doc_id=doc_id,
        user_id=user_id,
        is_admin=is_admin,
    )


def _rename_collection_status_impl(
    old_name: str,
    new_name: str,
    *,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> List[str]:
    store = get_ingestion_status_store()
    try:
        return store.rename_collection_status(
            old_name=old_name,
            new_name=new_name,
            user_id=user_id,
            is_admin=is_admin,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "rename_collection_status(%r -> %r) failed: %s", old_name, new_name, exc
        )
        return [str(exc)]


async def _rename_collection_status_async_impl(
    old_name: str,
    new_name: str,
    *,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> List[str]:
    store = get_ingestion_status_store()
    try:
        return await store.rename_collection_status_async(
            old_name=old_name,
            new_name=new_name,
            user_id=user_id,
            is_admin=is_admin,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "rename_collection_status_async(%r -> %r) failed: %s",
            old_name,
            new_name,
            exc,
        )
        return [str(exc)]
