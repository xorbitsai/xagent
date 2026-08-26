"""Hybrid search implementation combining dense and sparse retrieval.

This module provides fusion strategies for combining vector search and
full-text search results using RRF or linear weighted combination.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..core.exceptions import DocumentValidationError
from ..core.schemas import (
    FusionConfig,
    HybridSearchResponse,
    SearchResult,
)
from ..vector_storage.vector_manager import validate_query_vector

if TYPE_CHECKING:
    from ..kb import KBLegacyStepCompatibilityFacade

logger = logging.getLogger(__name__)


def _rrf_fusion(
    rank_lists: List[List[SearchResult]], k: int = 60
) -> List[SearchResult]:
    """Performs Reciprocal Rank Fusion (RRF) on multiple lists of SearchResult.

    Args:
        rank_lists: A list of ranked lists, where each inner list contains SearchResult objects.
        k: A constant that determines the impact of lower ranks. Higher k means smoother curve.

    Returns:
        A single list of SearchResult objects, fused and re-ranked by RRF score.
    """
    fused_scores: Dict[str, float] = {}
    # Track original search results by a unique identifier to preserve full data
    result_map: Dict[str, SearchResult] = {}

    for rank_list in rank_lists:
        for rank, result in enumerate(rank_list, start=1):
            # Create a unique key for each search result (doc_id + chunk_id + model_tag is a good candidate)
            unique_id = f"{result.doc_id}-{result.chunk_id}-{result.parse_hash}-{result.model_tag}"
            fused_scores[unique_id] = fused_scores.get(unique_id, 0.0) + (
                1.0 / (k + rank)
            )
            if unique_id not in result_map:
                result_map[unique_id] = result.model_copy()  # Store a copy

    # Sort results by fused RRF score in descending order
    sorted_unique_ids = sorted(
        fused_scores.keys(), key=lambda uid: fused_scores[uid], reverse=True
    )
    fused_results = []
    for uid in sorted_unique_ids:
        original_result = result_map[uid]
        new_score = fused_scores[uid]
        fused_results.append(
            original_result.model_copy(update={"score": new_score})
        )  # Create new instance with updated score

    return fused_results


def _linear_fusion(
    dense_results: List[SearchResult],
    sparse_results: List[SearchResult],
    dense_weight: float,
    sparse_weight: float,
    normalize_scores: bool,
) -> List[SearchResult]:
    """Performs linear weighted fusion on dense and sparse search results.

    Args:
        dense_results: List of SearchResult objects from dense search.
        sparse_results: List of SearchResult objects from sparse search.
        dense_weight: Weight for dense results in linear fusion (0-1).
        sparse_weight: Weight for sparse results in linear fusion (0-1).
        normalize_scores: Whether to normalize scores (Min-Max) before fusion.

    Returns:
        A single list of SearchResult objects, fused and re-ranked.
    """
    combined_results: Dict[str, SearchResult] = {}

    def normalize(scores: List[float]) -> List[float]:
        if not scores or not normalize_scores:
            return scores
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            logger.debug(
                "Score normalization skipped: all scores are equal (%f). Returning zeros.",
                max_score,
            )
            return [0.0 for _ in scores]
        return [(s - min_score) / (max_score - min_score) for s in scores]

    # Extract scores for normalization if needed
    dense_scores = [r.score for r in dense_results]
    sparse_scores = [r.score for r in sparse_results]

    normalized_dense_scores = normalize(dense_scores)
    normalized_sparse_scores = normalize(sparse_scores)

    # Map normalized scores back to results for processing
    for i, result in enumerate(dense_results):
        unique_id = (
            f"{result.doc_id}-{result.chunk_id}-{result.parse_hash}-{result.model_tag}"
        )
        current_score = normalized_dense_scores[i] * dense_weight
        if unique_id not in combined_results:
            combined_results[unique_id] = result.model_copy(
                update={"score": current_score}
            )
        else:
            # Create a new SearchResult with updated score
            existing_result = combined_results[unique_id]
            updated_score = existing_result.score + current_score
            combined_results[unique_id] = existing_result.model_copy(
                update={"score": updated_score}
            )

    for i, result in enumerate(sparse_results):
        unique_id = (
            f"{result.doc_id}-{result.chunk_id}-{result.parse_hash}-{result.model_tag}"
        )
        current_score = normalized_sparse_scores[i] * sparse_weight
        if unique_id not in combined_results:
            combined_results[unique_id] = result.model_copy(
                update={"score": current_score}
            )
        else:
            # Create a new SearchResult with updated score
            existing_result = combined_results[unique_id]
            updated_score = existing_result.score + current_score
            combined_results[unique_id] = existing_result.model_copy(
                update={"score": updated_score}
            )

    # Extract combined results
    fused_results = list(combined_results.values())

    # Normalize final scores to [0, 1] range to ensure consistency
    if fused_results:
        scores = [r.score for r in fused_results]
        min_score = min(scores)
        max_score = max(scores)

        if max_score > min_score:
            # Min-Max normalization to [0, 1] range
            fused_results = [
                r.model_copy(
                    update={"score": (r.score - min_score) / (max_score - min_score)}
                )
                for r in fused_results
            ]
        elif max_score == min_score and max_score > 0:
            # All scores are equal and non-zero, normalize to 1.0
            fused_results = [r.model_copy(update={"score": 1.0}) for r in fused_results]
        else:
            # All scores are zero or negative, set to 0.0
            fused_results = [r.model_copy(update={"score": 0.0}) for r in fused_results]

    # Sort results by normalized fused score in descending order
    fused_results.sort(key=lambda r: r.score, reverse=True)
    return fused_results


def _get_legacy_step_compatibility_facade() -> "KBLegacyStepCompatibilityFacade":
    """Return the coordinator-owned legacy step compatibility facade."""
    from ..kb import get_kb_coordinator

    return get_kb_coordinator().legacy_step_compatibility


def _validate_hybrid_inputs(
    collection: str,
    model_tag: str,
    top_k: int,
    query_text: str,
    query_vector: List[float],
) -> None:
    """Validate hybrid-search inputs at the public boundary."""

    if not collection or not isinstance(collection, str):
        raise DocumentValidationError("Collection must be a non-empty string")
    if not model_tag or not isinstance(model_tag, str):
        raise DocumentValidationError("model_tag must be a non-empty string")
    if not query_text or not isinstance(query_text, str):
        raise DocumentValidationError("query_text must be a non-empty string")
    if top_k <= 0 or top_k > 1000:
        raise DocumentValidationError("top_k must be between 1 and 1000")
    validate_query_vector(query_vector)


def search_hybrid(
    collection: str,
    model_tag: str,
    query_text: str,
    query_vector: List[float],
    *,
    top_k: int = 10,
    filters: Any = None,
    fusion_config: Optional[FusionConfig] = None,
    readonly: bool = False,
    nprobes: Optional[int] = None,
    refine_factor: Optional[int] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> HybridSearchResponse:
    """Performs hybrid search, combining dense and sparse retrieval."""
    _validate_hybrid_inputs(collection, model_tag, top_k, query_text, query_vector)
    return _get_legacy_step_compatibility_facade().search_hybrid(
        collection=collection,
        model_tag=model_tag,
        query_text=query_text,
        query_vector=query_vector,
        top_k=top_k,
        filters=filters,
        fusion_config=fusion_config,
        readonly=readonly,
        nprobes=nprobes,
        refine_factor=refine_factor,
        user_id=user_id,
        is_admin=is_admin,
    )
