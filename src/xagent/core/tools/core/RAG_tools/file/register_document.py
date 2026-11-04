"""Public document-row helpers (compatibility wrappers).

These module-level functions preserve the historical import paths and call
signatures for document registration, loading, and listing. They delegate to
the coordinator-owned ``KBLegacyStepCompatibilityFacade``, which converts legacy
inputs into semantic requests and routes document-row lifecycle operations
through ``KBCoordinator`` and ``KBCollectionHandle`` (#508).
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from ..kb import KBLegacyStepCompatibilityFacade

logger = logging.getLogger(__name__)


def _get_legacy_step_compatibility_facade() -> "KBLegacyStepCompatibilityFacade":
    """Return the coordinator-owned legacy step compatibility facade."""
    from ..kb import get_kb_coordinator

    return get_kb_coordinator().legacy_step_compatibility


def register_document(
    collection: str,
    source_path: str,
    file_type: Optional[str] = None,
    doc_id: Optional[str] = None,
    uploaded_at: Optional[str] = None,
    user_id: Optional[int] = None,
    file_id: Optional[str] = None,
    metadata_source_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Register a document into the LanceDB system.

    Args:
        collection: LanceDB collection name (data isolation)
        source_path: Absolute path to the uploaded file
        file_type: Optional file type; auto-detected from extension if not provided
        doc_id: Optional document ID; deterministic/UUID generated if not provided
        uploaded_at: Optional ISO8601 timestamp string (supports trailing 'Z');
            defaults to now if not provided or parse fails
        user_id: Optional user ID for multi-tenancy ownership
        file_id: Optional UploadedFile file_id for stable file association
        metadata_source_path: Optional canonical source path to store in metadata
            when the file is read from a temporary immutable path

    Returns:
        A plain dict with ``doc_id``, ``created``, and ``content_hash``.
    """
    return _get_legacy_step_compatibility_facade().register_document(
        collection=collection,
        source_path=source_path,
        file_type=file_type,
        doc_id=doc_id,
        uploaded_at=uploaded_at,
        user_id=user_id,
        file_id=file_id,
        metadata_source_path=metadata_source_path,
    )


def get_document(db_dir: str, collection: str, doc_id: str) -> Optional[Any]:
    """Retrieve a document record from LanceDB (legacy raw-dict shape or None).

    ``db_dir`` is accepted for backward compatibility and ignored.
    """
    return _get_legacy_step_compatibility_facade().get_document(
        db_dir, collection, doc_id
    )


def list_documents(
    db_dir: str, collection: str, limit: int = 100
) -> list[Dict[str, Any]]:
    """List documents in the collection (legacy raw-dict list).

    ``db_dir`` is accepted for backward compatibility and ignored.
    """
    return _get_legacy_step_compatibility_facade().list_documents(
        db_dir, collection, limit
    )
