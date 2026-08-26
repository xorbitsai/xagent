from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from ..core.exceptions import DocumentValidationError
from ..core.schemas import (
    SparseSearchResponse,
)

if TYPE_CHECKING:
    from ..kb import KBLegacyStepCompatibilityFacade

logger = logging.getLogger(__name__)


def _get_legacy_step_compatibility_facade() -> "KBLegacyStepCompatibilityFacade":
    """Return the coordinator-owned legacy step compatibility facade."""
    from ..kb import get_kb_coordinator

    return get_kb_coordinator().legacy_step_compatibility


def _validate_sparse_inputs(
    collection: str,
    model_tag: str,
    top_k: int,
    query_text: str,
) -> None:
    """Validate sparse-search inputs at the public boundary."""

    if not collection or not isinstance(collection, str):
        raise DocumentValidationError("Collection must be a non-empty string")
    if not model_tag or not isinstance(model_tag, str):
        raise DocumentValidationError("model_tag must be a non-empty string")
    if not query_text or not isinstance(query_text, str):
        raise DocumentValidationError("query_text must be a non-empty string")
    if top_k <= 0 or top_k > 1000:
        raise DocumentValidationError("top_k must be between 1 and 1000")


def search_sparse(
    collection: str,
    model_tag: str,
    query_text: str,
    *,
    top_k: int,
    filters: Any = None,
    readonly: bool = False,
    nprobes: Optional[int] = None,
    refine_factor: Optional[int] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> SparseSearchResponse:
    """Performs sparse (Full-Text Search) retrieval on the specified collection."""
    _validate_sparse_inputs(collection, model_tag, top_k, query_text)
    return _get_legacy_step_compatibility_facade().search_sparse(
        collection=collection,
        model_tag=model_tag,
        query_text=query_text,
        top_k=top_k,
        filters=filters,
        readonly=readonly,
        nprobes=nprobes,
        refine_factor=refine_factor,
        user_id=user_id,
        is_admin=is_admin,
    )


# --- Async variant (Phase 1A Option C) ---


async def search_sparse_async(
    collection: str,
    model_tag: str,
    query_text: str,
    *,
    top_k: int,
    filters: Any = None,
    readonly: bool = False,
    nprobes: Optional[int] = None,
    refine_factor: Optional[int] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> SparseSearchResponse:
    """Perform sparse retrieval using async vector store abstraction."""
    _validate_sparse_inputs(collection, model_tag, top_k, query_text)
    return await _get_legacy_step_compatibility_facade().search_sparse_async(
        collection=collection,
        model_tag=model_tag,
        query_text=query_text,
        top_k=top_k,
        filters=filters,
        readonly=readonly,
        nprobes=nprobes,
        refine_factor=refine_factor,
        user_id=user_id,
        is_admin=is_admin,
    )
