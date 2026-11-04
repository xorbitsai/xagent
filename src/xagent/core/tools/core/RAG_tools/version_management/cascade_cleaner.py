"""Cascade cleanup functions for version management.

Provide cascade cleanup utilities when promoting main versions,
ensuring data consistency across processing stages.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Optional

from typing_extensions import Literal

from ..core.exceptions import CascadeCleanupError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..kb import KBVersionCompatibilityFacade


def _get_version_compatibility_facade() -> "KBVersionCompatibilityFacade":
    from ..kb import get_kb_coordinator

    return get_kb_coordinator().version_compatibility


def cascade_delete(
    *,
    target: Literal["collection", "document"],
    collection: str,
    doc_id: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: Optional[bool] = None,
    model_tag: Optional[str] = None,
    preview_only: bool = True,
    confirm: bool = False,
) -> Dict[str, int]:
    return _get_version_compatibility_facade().cascade_delete(
        target=target,
        collection=collection,
        doc_id=doc_id,
        user_id=user_id,
        is_admin=is_admin,
        model_tag=model_tag,
        preview_only=preview_only,
        confirm=confirm,
    )


def cleanup_cascade(
    collection: str,
    doc_id: str,
    scope: str,
    new_parse_hash: Optional[str] = None,
    old_parse_hash: Optional[str] = None,
    model_tag: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: Optional[bool] = None,
    preview_only: bool = True,
    confirm: bool = False,
) -> Dict[str, int]:
    return _get_version_compatibility_facade().cleanup_cascade(
        collection=collection,
        doc_id=doc_id,
        scope=scope,
        new_parse_hash=new_parse_hash,
        old_parse_hash=old_parse_hash,
        model_tag=model_tag,
        user_id=user_id,
        is_admin=is_admin,
        preview_only=preview_only,
        confirm=confirm,
    )


def _cleanup_cascade_impl(
    collection: str,
    doc_id: str,
    scope: str,
    new_parse_hash: Optional[str] = None,
    old_parse_hash: Optional[str] = None,
    model_tag: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: Optional[bool] = None,
    preview_only: bool = True,
    confirm: bool = False,
) -> Dict[str, int]:
    """Unified cascade cleanup by scope with preview/confirm semantics.

    Args:
        collection: Collection name
        doc_id: Document ID
        scope: "document" | "parse" | "chunk" | "embeddings" | "pointers"
        new_parse_hash: New main parse hash for parse/chunk scopes
        old_parse_hash: Optional old main parse hash (auto-filled from pointers if None)
        model_tag: Optional embed model tag limiter
        user_id: Optional user ID for tenant scoping
        is_admin: Whether caller is admin (None to fallback to context, defaults to True
                   for system-level version promotion operations)
        preview_only: If True, only plan counts
        confirm: If True, execute deletions

    Returns:
        Deleted (or planned) counts per table scope
    """
    return _get_version_compatibility_facade().cleanup_cascade(
        collection=collection,
        doc_id=doc_id,
        scope=scope,
        new_parse_hash=new_parse_hash,
        old_parse_hash=old_parse_hash,
        model_tag=model_tag,
        user_id=user_id,
        is_admin=is_admin,
        preview_only=preview_only,
        confirm=confirm,
    )


def cleanup_document_cascade(
    collection: str,
    doc_id: str,
    model_tag: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: bool = True,
    preview_only: bool = True,
    confirm: bool = False,
) -> Dict[str, int]:
    return _get_version_compatibility_facade().cleanup_document_cascade(
        collection=collection,
        doc_id=doc_id,
        model_tag=model_tag,
        user_id=user_id,
        is_admin=is_admin,
        preview_only=preview_only,
        confirm=confirm,
    )


def _cleanup_document_cascade_impl(
    collection: str,
    doc_id: str,
    model_tag: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: bool = True,
    preview_only: bool = True,
    confirm: bool = False,
) -> Dict[str, int]:
    """Cascade delete all data for a document across all stages.

    Order: embeddings_* -> chunks -> parses -> main_pointers -> documents

    Args:
        collection: Collection name
        doc_id: Document ID
        model_tag: Optional model tag to limit embeddings deletion

    Returns:
        Deleted counts per scope
    """
    try:
        # Delegate to unified entry
        return _cleanup_cascade_impl(
            collection=collection,
            doc_id=doc_id,
            scope="document",
            model_tag=model_tag,
            user_id=user_id,
            is_admin=is_admin,
            preview_only=preview_only,
            confirm=confirm,
        )

    except Exception as e:
        raise CascadeCleanupError(f"Failed to cleanup document cascade: {e}")


def cleanup_parse_cascade(
    collection: str,
    doc_id: str,
    old_parse_hash: Optional[str] = None,
    new_parse_hash: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: bool = True,
    preview_only: bool = True,
    confirm: bool = False,
) -> Dict[str, int]:
    return _get_version_compatibility_facade().cleanup_parse_cascade(
        collection=collection,
        doc_id=doc_id,
        old_parse_hash=old_parse_hash,
        new_parse_hash=new_parse_hash,
        user_id=user_id,
        is_admin=is_admin,
        preview_only=preview_only,
        confirm=confirm,
    )


def _cleanup_parse_cascade_impl(
    collection: str,
    doc_id: str,
    old_parse_hash: Optional[str] = None,
    new_parse_hash: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: bool = True,
    preview_only: bool = True,
    confirm: bool = False,
) -> Dict[str, int]:
    """Clean up cascade when promoting a new parse version.

    This method:
    1. Deletes old parse's chunks and embeddings
    2. Deletes other parse candidates and their downstream data

    Args:
        collection: Collection name
        doc_id: Document ID
        old_parse_hash: Old main parse hash (optional)
        new_parse_hash: New main parse hash (optional)

    Returns:
        Dictionary with deletion counts

    Raises:
        CascadeCleanupError: If cleanup fails
    """
    try:
        return _cleanup_cascade_impl(
            collection=collection,
            doc_id=doc_id,
            scope="parse",
            new_parse_hash=new_parse_hash,
            old_parse_hash=old_parse_hash,
            user_id=user_id,
            is_admin=is_admin,
            preview_only=preview_only,
            confirm=confirm,
        )

    except Exception as e:
        raise CascadeCleanupError(f"Failed to cleanup parse cascade: {e}")


def cleanup_chunk_cascade(
    collection: str,
    doc_id: str,
    old_parse_hash: Optional[str] = None,
    new_parse_hash: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: bool = True,
    preview_only: bool = True,
    confirm: bool = False,
) -> Dict[str, int]:
    return _get_version_compatibility_facade().cleanup_chunk_cascade(
        collection=collection,
        doc_id=doc_id,
        old_parse_hash=old_parse_hash,
        new_parse_hash=new_parse_hash,
        user_id=user_id,
        is_admin=is_admin,
        preview_only=preview_only,
        confirm=confirm,
    )


def _cleanup_chunk_cascade_impl(
    collection: str,
    doc_id: str,
    old_parse_hash: Optional[str] = None,
    new_parse_hash: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: bool = True,
    preview_only: bool = True,
    confirm: bool = False,
) -> Dict[str, int]:
    """Clean up cascade when promoting a new chunk version.

    This method:
    1. Deletes old chunk's embeddings
    2. Deletes other chunk candidates

    Args:
        collection: Collection name
        doc_id: Document ID
        old_parse_hash: Old main parse hash (optional)
        new_parse_hash: New main parse hash (optional)

    Returns:
        Dictionary with deletion counts

    Raises:
        CascadeCleanupError: If cleanup fails
    """
    try:
        return _cleanup_cascade_impl(
            collection=collection,
            doc_id=doc_id,
            scope="chunk",
            new_parse_hash=new_parse_hash,
            old_parse_hash=old_parse_hash,
            user_id=user_id,
            is_admin=is_admin,
            preview_only=preview_only,
            confirm=confirm,
        )

    except Exception as e:
        raise CascadeCleanupError(f"Failed to cleanup chunk cascade: {e}")


def cleanup_embed_cascade(
    collection: str,
    doc_id: str,
    model_tag: Optional[str] = None,
    old_technical_id: Optional[str] = None,
    new_technical_id: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: bool = True,
    preview_only: bool = True,
    confirm: bool = False,
) -> Dict[str, int]:
    return _get_version_compatibility_facade().cleanup_embed_cascade(
        collection=collection,
        doc_id=doc_id,
        model_tag=model_tag,
        old_technical_id=old_technical_id,
        new_technical_id=new_technical_id,
        user_id=user_id,
        is_admin=is_admin,
        preview_only=preview_only,
        confirm=confirm,
    )


def _cleanup_embed_cascade_impl(
    collection: str,
    doc_id: str,
    model_tag: Optional[str] = None,
    old_technical_id: Optional[str] = None,
    new_technical_id: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: bool = True,
    preview_only: bool = True,
    confirm: bool = False,
) -> Dict[str, int]:
    """Clean up cascade when promoting a new embeddings version.

    This method:
    1. Deletes other embeddings candidates (optionally filtered by model_tag)

    Args:
        collection: Collection name
        doc_id: Document ID
        model_tag: Model tag filter (optional)
        old_technical_id: Old main technical ID (optional)
        new_technical_id: New main technical ID (optional)

    Returns:
        Dictionary with deletion counts

    Raises:
        CascadeCleanupError: If cleanup fails
    """
    try:
        # Delegate to unified entry; old/new technical ids are not used in current schema
        return _cleanup_cascade_impl(
            collection=collection,
            doc_id=doc_id,
            scope="embeddings",
            model_tag=model_tag,
            user_id=user_id,
            is_admin=is_admin,
            preview_only=preview_only,
            confirm=confirm,
        )

    except Exception as e:
        raise CascadeCleanupError(f"Failed to cleanup embed cascade: {e}")
