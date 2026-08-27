from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from ..core.schemas import (
    SparseSearchResponse,
)
from ..storage.contracts import FilterInput
from ..utils.validation_utils import (
    validate_search_common_inputs,
    validate_search_query_text,
)

if TYPE_CHECKING:
    from ..kb import KBLegacyStepCompatibilityFacade

logger = logging.getLogger(__name__)


def _get_legacy_step_compatibility_facade() -> "KBLegacyStepCompatibilityFacade":
    """Return the coordinator-owned legacy step compatibility facade."""
    from ..kb import get_kb_coordinator

    return get_kb_coordinator().legacy_step_compatibility


def search_sparse(
    collection: str,
    model_tag: str,
    query_text: str,
    *,
    top_k: int,
    filters: Optional[FilterInput] = None,
    readonly: bool = False,
    nprobes: Optional[int] = None,
    refine_factor: Optional[int] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> SparseSearchResponse:
    """Performs sparse (Full-Text Search) retrieval on the specified collection."""
    validate_search_common_inputs(collection, model_tag, top_k)
    validate_search_query_text(query_text)
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
    filters: Optional[FilterInput] = None,
    readonly: bool = False,
    nprobes: Optional[int] = None,
    refine_factor: Optional[int] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> SparseSearchResponse:
    """Perform sparse retrieval using async vector store abstraction."""
    validate_search_common_inputs(collection, model_tag, top_k)
    validate_search_query_text(query_text)
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
