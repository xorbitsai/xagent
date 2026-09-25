"""
Dense vector search implementation for RAG retrieval.

This module provides the main entry point for dense vector search operations,
handling input validation and delegating to the KB coordinator.

Phase 1A Option C: Provides both sync and async search functions.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..core.schemas import DenseSearchResponse
from ..utils.validation_utils import validate_search_common_inputs
from ..vector_storage.vector_manager import validate_query_vector

if TYPE_CHECKING:
    from ..kb import KBCoordinator

logger = logging.getLogger(__name__)


def _get_coordinator() -> "KBCoordinator":
    """Return the process-wide KB coordinator that owns search routing."""
    from ..kb import get_kb_coordinator

    return get_kb_coordinator()


def _validate_dense_inputs(
    collection: str, model_tag: str, top_k: int, query_vector: List[float]
) -> None:
    """Validate dense-search inputs at the public boundary.

    Raises:
        DocumentValidationError: If collection/model_tag/top_k are invalid.
        VectorValidationError: If query-vector validation fails.
    """
    validate_search_common_inputs(collection, model_tag, top_k)
    validate_query_vector(query_vector)


def search_dense(
    collection: str,
    model_tag: str,
    query_vector: List[float],
    *,
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    readonly: bool = False,
    nprobes: Optional[int] = None,
    refine_factor: Optional[int] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> DenseSearchResponse:
    """Execute dense vector search for RAG retrieval.

    Raises:
        DocumentValidationError: If input validation fails.
        VectorValidationError: If query-vector validation fails.
    """
    # Input validation at the public boundary.
    _validate_dense_inputs(collection, model_tag, top_k, query_vector)

    return _get_coordinator().search_dense_sync(
        collection=collection,
        model_tag=model_tag,
        query_vector=query_vector,
        top_k=top_k,
        filters=filters,
        readonly=readonly,
        nprobes=nprobes,
        refine_factor=refine_factor,
        user_id=user_id,
        is_admin=is_admin,
    )


# --- Async variant (Phase 1A Option C) ---


async def search_dense_async(
    collection: str,
    model_tag: str,
    query_vector: List[float],
    *,
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    readonly: bool = False,
    nprobes: Optional[int] = None,
    refine_factor: Optional[int] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> DenseSearchResponse:
    """Execute dense vector search using async vector store abstraction.

    Raises:
        DocumentValidationError: If input validation fails.
        VectorValidationError: If query-vector validation fails.
    """
    # Input validation at the public boundary (shared with the sync path).
    _validate_dense_inputs(collection, model_tag, top_k, query_vector)
    return await _get_coordinator().search_dense(
        collection=collection,
        model_tag=model_tag,
        query_vector=query_vector,
        top_k=top_k,
        filters=filters,
        readonly=readonly,
        nprobes=nprobes,
        refine_factor=refine_factor,
        user_id=user_id,
        is_admin=is_admin,
    )
