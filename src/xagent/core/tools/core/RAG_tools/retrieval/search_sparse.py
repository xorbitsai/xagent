from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Set

import pandas as pd
import pyarrow as pa  # type: ignore
from pyarrow import Table as PyArrowTable

from ......providers.vector_store.lancedb import get_connection_from_env
from ..core.schemas import (
    SearchFallbackAction,
    SearchResult,
    SearchWarning,
    SparseSearchResponse,
)
from ..LanceDB.model_tag_utils import to_model_tag
from ..utils.metadata_utils import deserialize_metadata
from ..utils.string_utils import build_lancedb_filter_expression
from ..utils.user_permissions import UserPermissions
from ..vector_storage.index_manager import get_index_manager

logger = logging.getLogger(__name__)


def search_sparse(
    collection: str,
    model_tag: str,
    query_text: str,
    *,
    top_k: int,
    filters: Optional[Dict[str, Any]] = None,
    readonly: bool = False,
    nprobes: Optional[int] = None,
    refine_factor: Optional[int] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> SparseSearchResponse:
    """Performs sparse (Full-Text Search) retrieval on the specified collection."""

    table_name = f"embeddings_{to_model_tag(model_tag)}"
    _fts_enabled = False
    current_warnings: List[SearchWarning] = []

    if readonly:
        current_warnings.append(
            SearchWarning(
                code="READONLY_MODE",
                message=f"Readonly mode enabled for sparse search on {table_name}. No FTS index operations will be performed.",
                fallback_action=SearchFallbackAction.REBUILD_INDEX,
                affected_models=[model_tag],
            )
        )

    try:
        conn = get_connection_from_env()
        table = conn.open_table(table_name)

        index_manager = get_index_manager()
        _, _ = index_manager.check_and_create_index(table, table_name, readonly)
        _fts_enabled = index_manager.get_fts_index_status(table)

        if not _fts_enabled:
            current_warnings.append(
                SearchWarning(
                    code="FTS_INDEX_MISSING",
                    message=f"FTS index not found on 'text' column for {table_name}. Sparse search performance may be degraded.",
                    fallback_action=SearchFallbackAction.REBUILD_INDEX,
                    affected_models=[model_tag],
                )
            )

        search_query = table.search(query_text, query_type="fts").limit(top_k)

        # Build filter expression combining collection scope, user permissions and custom filters
        filter_clauses = []

        # Scope results to the requested collection (required for KB isolation)
        if collection:
            collection_filter = build_lancedb_filter_expression(
                {"collection": collection}
            )
            if collection_filter:
                filter_clauses.append(collection_filter)

        # Add user permission filter for multi-tenancy
        user_filter = UserPermissions.get_user_filter(user_id, is_admin)
        if user_filter:
            filter_clauses.append(user_filter)

        # Add custom filters if provided
        if filters:
            custom_filter = build_lancedb_filter_expression(filters)
            if custom_filter:
                filter_clauses.append(custom_filter)

        # Combine all filters with AND
        if filter_clauses:
            combined_filter = " and ".join(f"({clause})" for clause in filter_clauses)
            search_query = search_query.where(combined_filter)

        raw_results_df: pd.DataFrame = search_query.to_pandas()

        if not raw_results_df.empty:
            search_results: List[SearchResult] = []
            for _, row in raw_results_df.iterrows():
                # LanceDB FTS returns TF-IDF score (higher is better),
                # normalize to similarity score (0-1) similar to dense search
                # Using score/(1+score) formula to convert TF-IDF to normalized similarity
                raw_score_value = row.get("_score")
                raw_score = float(raw_score_value) if pd.notna(raw_score_value) else 0.0
                # Normalize TF-IDF score to [0, 1) range using x/(1+x) formula
                score = raw_score / (1.0 + raw_score)
                # Deserialize metadata from JSON string to dictionary
                metadata = deserialize_metadata(row.get("metadata"))
                search_results.append(
                    SearchResult(
                        doc_id=row["doc_id"],
                        chunk_id=row["chunk_id"],
                        text=row["text"],
                        score=score,
                        parse_hash=row["parse_hash"],
                        model_tag=model_tag,
                        created_at=row["created_at"],
                        metadata=metadata,
                    )
                )

            return _build_sparse_response(
                results=search_results,
                warnings=current_warnings,
                fts_enabled=_fts_enabled,
                query_text=query_text,
            )

        logger.warning(
            "FTS lookup returned no rows for query '%s'; falling back to substring match",
            query_text,
        )
        fallback_results = _substring_fallback(
            table=table,
            collection=collection,
            query_text=query_text,
            model_tag=model_tag,
            top_k=top_k,
            filters=filters,
            current_warnings=current_warnings,
        )

        return _build_sparse_response(
            results=fallback_results,
            warnings=current_warnings,
            fts_enabled=_fts_enabled,
            query_text=query_text,
        )

    except Exception as e:
        logger.error(
            f"Sparse search failed for {table_name} with query '{query_text}': {e}"
        )
        error_warnings = current_warnings + [
            SearchWarning(
                code="FTS_SEARCH_FAILED",
                message=f"An unexpected error occurred during sparse search: {str(e)}",
                fallback_action=SearchFallbackAction.PARTIAL_RESULTS,
                affected_models=[model_tag],
            )
        ]
        return _build_sparse_response(
            results=[],
            warnings=error_warnings,
            fts_enabled=_fts_enabled,
            query_text=query_text,
            status="failed",
        )


def _substring_fallback(
    *,
    table: Any,
    collection: str,
    query_text: str,
    model_tag: str,
    top_k: int,
    filters: Optional[Dict[str, Any]],
    current_warnings: List[SearchWarning],
    batch_size: int = 2048,
) -> List[SearchResult]:
    """Perform a memory-friendly substring scan across the table when FTS misses."""

    desired_columns: Set[str] = {
        "collection",
        "doc_id",
        "chunk_id",
        "text",
        "parse_hash",
        "created_at",
        "metadata",
    }
    if filters:
        desired_columns.update(filters.keys())

    results: List[SearchResult] = []

    try:
        if hasattr(table, "to_batches"):
            batch_iter: Iterable[Any] = table.to_batches(
                columns=list(desired_columns), batch_size=batch_size
            )
        else:
            if pa is None:  # pragma: no cover - Safety guard when pyarrow missing
                raise ImportError(
                    "pyarrow is required for substring fallback when LanceDB table does not expose to_batches()."
                )
            arrow_table: PyArrowTable = table.to_arrow()  # type: ignore
            arrow_table = arrow_table.select(list(desired_columns))
            batch_iter = arrow_table.to_batches(max_chunksize=batch_size)
    except Exception as exc:  # noqa: BLE001
        logger.error("Substring fallback failed to read batches: %s", exc)
        return results

    for batch in batch_iter:
        batch_df = batch.to_pandas()

        mask = batch_df["collection"] == collection
        if filters:
            for key, value in filters.items():
                if key not in batch_df.columns:
                    continue
                if isinstance(value, (list, tuple, set)):
                    mask &= batch_df[key].isin(list(value))
                else:
                    mask &= batch_df[key] == value

        if not mask.any():
            continue

        text_mask = (
            batch_df["text"].astype(str).str.contains(query_text, na=False, regex=False)
        )
        mask &= text_mask

        if not mask.any():
            continue

        for _, row in batch_df.loc[mask].iterrows():
            # Deserialize metadata from JSON string to dictionary
            metadata = deserialize_metadata(row.get("metadata"))
            results.append(
                SearchResult(
                    doc_id=row["doc_id"],
                    chunk_id=row["chunk_id"],
                    text=row["text"],
                    score=1.0,
                    parse_hash=row["parse_hash"],
                    model_tag=model_tag,
                    created_at=row["created_at"],
                    metadata=metadata,
                )
            )
            if len(results) >= top_k:
                break

        if len(results) >= top_k:
            break

    if results:
        current_warnings.append(
            SearchWarning(
                code="FTS_FALLBACK",
                message=(
                    "Full-text index returned no matches; used substring search fallback. "
                    "Check FTS tokenizer configuration or update LanceDB to ensure proper tokenisation for query language."
                ),
                fallback_action=SearchFallbackAction.BRUTE_FORCE,
                affected_models=[model_tag],
            )
        )

    return results


def _build_sparse_response(
    *,
    results: List[SearchResult],
    warnings: List[SearchWarning],
    fts_enabled: bool,
    query_text: str,
    status: str = "success",
) -> SparseSearchResponse:
    """Helper to assemble `SparseSearchResponse`. Allows fallback reuse."""

    return SparseSearchResponse(
        results=results,
        total_count=len(results),
        status=status,
        warnings=warnings,
        fts_enabled=fts_enabled,
        query_text=query_text,
    )
