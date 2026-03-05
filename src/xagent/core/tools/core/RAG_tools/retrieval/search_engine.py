"""
Core search engine implementation for dense vector retrieval.

This module provides the low-level search functionality that interacts
directly with LanceDB for performing ANN searches on embeddings tables.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from ......providers.vector_store.lancedb import get_connection_from_env
from ..core.schemas import SearchResult
from ..LanceDB.model_tag_utils import to_model_tag
from ..utils.lancedb_query_utils import query_to_list
from ..utils.metadata_utils import deserialize_metadata
from ..utils.string_utils import build_lancedb_filter_expression
from ..vector_storage.index_manager import get_index_manager

logger = logging.getLogger(__name__)


def search_dense_engine(
    collection: str,
    model_tag: str,
    query_vector: List[float],
    *,
    top_k: int,
    filters: Optional[Dict[str, Any]] = None,
    readonly: bool = False,
    nprobes: Optional[int] = None,
    refine_factor: Optional[int] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> Tuple[List[SearchResult], str, Optional[str]]:
    """
    Execute dense vector search against LanceDB embeddings table.

    Args:
        collection: Collection name for data isolation
        model_tag: Model tag to determine which embeddings table to search
        query_vector: Query vector for similarity search
        top_k: Number of top results to return
        filters: Optional filters to apply to the search
        readonly: If True, don't trigger index creation
        nprobes: Number of partitions to probe for ANN search (LanceDB specific).
        refine_factor: Refine factor for re-ranking results in memory (LanceDB specific).
        user_id: Optional user ID for multi-tenancy filtering.
        is_admin: Whether the user has admin privileges.

    Returns:
        Tuple of (search_results, index_status, index_advice)
    """
    try:
        # Get database connection
        conn = get_connection_from_env()

        # Build table name
        table_name = f"embeddings_{to_model_tag(model_tag)}"

        # Open table
        table = conn.open_table(table_name)

        # Check and create index if needed
        index_manager = get_index_manager()
        index_status, index_advice = index_manager.check_and_create_index(
            table, table_name, readonly
        )

        # Build LanceDB search query using query builder pattern
        search_query = table.search(
            query_vector,
            vector_column_name="vector",
        )

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
        from ..utils.user_permissions import UserPermissions

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

        # Limit results
        search_query = search_query.limit(top_k)

        # OPTIMIZATION: Use unified query_to_list() with three-tier fallback
        raw_results = query_to_list(search_query)

        # OPTIMIZATION: Use list comprehension instead of iterrows()
        # Convert raw results to SearchResult objects
        search_results = []
        for row in raw_results:
            # LanceDB returns Squared Euclidean Distance (L_2^{2} distance),
            # lower is better, convert to similarity score (higher is better)
            # Using 1/(1+distance) formula to convert distance to similarity
            # Arrow/to_list() returns None instead of NaN, so direct None check is sufficient
            distance_value = row.get("_distance")
            distance = float(distance_value) if distance_value is not None else 0.0
            score = 1.0 / (1.0 + distance)

            # Deserialize metadata from JSON string to dictionary
            metadata = deserialize_metadata(row.get("metadata"))

            search_result = SearchResult(
                doc_id=row["doc_id"],
                chunk_id=row["chunk_id"],
                text=row["text"],
                score=score,
                parse_hash=row["parse_hash"],
                model_tag=model_tag,
                created_at=row["created_at"],
                metadata=metadata,
            )
            search_results.append(search_result)

        return search_results, index_status, index_advice

    except Exception as e:
        logger.error(f"Failed to execute dense search: {str(e)}")
        raise
