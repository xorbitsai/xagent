"""Storage contracts for KB control-plane and vector-plane operations.

Phase 1A introduces these contracts to decouple API/business modules from
backend-specific database semantics.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

from ..core.config import DEFAULT_VECTOR_STORE_SCAN_LIMIT, IndexPolicy
from ..core.schemas import CollectionInfo, IndexResult

logger = logging.getLogger(__name__)

# Field name whitelist for filter validation
# Derived from all LanceDB table schemas in schema_manager.py
_VALID_FILTER_FIELDS = frozenset(
    {
        # documents table
        "collection",
        "doc_id",
        "source_path",
        "file_type",
        "content_hash",
        "uploaded_at",
        "title",
        "language",
        "user_id",
        # parses table
        "parse_hash",
        "parser",
        "created_at",
        "params_json",
        # chunks table
        "chunk_id",
        "index",
        "page_number",
        "section",
        "anchor",
        "json_path",
        "chunk_hash",
        "config_hash",
        "metadata",
        # embeddings table
        "model",
        "vector_dimension",
        "vector",
        # ingestion_runs table
        "status",
        "message",
        "updated_at",
        # main_pointers table
        "step_type",
        "model_tag",
        "semantic_id",
        "technical_id",
        "operator",
        # prompt_templates table
        "id",
        "name",
        "template",
        "version",
        "is_latest",
        # collection_metadata table
        "name",
        "schema_version",
        "embedding_model_id",
        "embedding_dimension",
        "documents",
        "processed_documents",
        "parses",
        "chunks",
        "embeddings",
        "document_names",
        "collection_locked",
        "allow_mixed_parse_methods",
        "skip_config_validation",
        "ingestion_config",
        "created_at",
        "updated_at",
        "last_accessed_at",
        "extra_metadata",
        # collection_config table
        "config_json",
    }
)


def validate_field_name(field: str) -> None:
    """Validate that a field name is in the allowed whitelist.

    Args:
        field: Field name to validate.

    Raises:
        ValueError: If field name is not in the whitelist.
    """
    if field not in _VALID_FILTER_FIELDS:
        raise ValueError(
            f"Invalid filter field '{field}'. "
            f"Field must be one of: {', '.join(sorted(_VALID_FILTER_FIELDS))}"
        )


def validate_filter_value(value: Any) -> None:
    """Validate that a filter value is an allowed type.

    Allowed types: str, int, float, bool, None, list, tuple, set.

    Args:
        value: Value to validate.

    Raises:
        ValueError: If value type is not allowed.
        TypeError: If value is a complex object (dict, custom class).
    """
    if value is None:
        return

    if isinstance(value, (str, int, float, bool)):
        return

    if isinstance(value, (list, tuple, set)):
        # Validate each element in the collection
        for item in value:
            if not isinstance(item, (str, int, float, bool, type(None))):
                raise TypeError(
                    f"Invalid filter value type in collection: {type(item).__name__}. "
                    f"Collection elements must be str, int, float, bool, or None."
                )
        return

    # Reject dict and complex objects
    raise TypeError(
        f"Invalid filter value type: {type(value).__name__}. "
        f"Allowed types: str, int, float, bool, None, list, tuple, set."
    )


def build_filter_from_dict(filters: Dict[str, Any]) -> Optional[FilterExpression]:
    """Convert a dictionary of filters to a FilterExpression with validation.

    This function provides a common entry point for building filter expressions
    from simple dictionary key-value pairs. All keys are validated against the
    field name whitelist, and all values are type-checked.

    Args:
        filters: Dictionary of field-name -> value mappings for equality filters.

    Returns:
        FilterExpression: Single FilterCondition for one filter,
                         tuple of conditions (AND) for multiple filters,
                         or None if filters is empty.

    Raises:
        ValueError: If a field name is not in the whitelist.
        TypeError: If a value type is not allowed.

    Example:
        >>> build_filter_from_dict({"collection": "my_collection", "doc_id": "doc123"})
        (FilterCondition(field='collection', operator=FilterOperator.EQ, value='my_collection'),
         FilterCondition(field='doc_id', operator=FilterOperator.EQ, value='doc123'))

        >>> build_filter_from_dict({"doc_id": "doc123"})
        FilterCondition(field='doc_id', operator=FilterOperator.EQ, value='doc123')
    """
    if not filters:
        return None

    conditions = []
    for field, value in filters.items():
        # Validate field name
        validate_field_name(field)

        # Validate value type
        validate_filter_value(value)

        # Create filter condition
        conditions.append(
            FilterCondition(field=field, operator=FilterOperator.EQ, value=value)
        )

    # Return single condition or tuple (AND combination)
    if len(conditions) == 1:
        return conditions[0]
    return tuple(conditions)


@runtime_checkable
class DatabaseConnection(Protocol):
    """Backend-agnostic database connection protocol.

    This protocol defines the minimal interface required for storage
    implementations to work with different database backends without
    importing concrete types like LanceDB's DBConnection.
    """

    def open_table(self, name: str) -> Any: ...

    def table_names(self) -> Sequence[str]: ...


@dataclass(frozen=True)
class DocumentRecord:
    """Lightweight document projection for metadata/control operations.

    Attributes:
        doc_id: Document identifier.
        file_id: Optional file identifier for uploaded file tracking.
        source_path: Original source path if available.
    """

    doc_id: str
    file_id: Optional[str] = None
    source_path: Optional[str] = None


class FilterOperator(str, Enum):
    """Comparison operators for filter expressions.

    These operators provide a backend-agnostic way to express filter conditions
    that can be translated to backend-specific query languages.
    """

    EQ = "eq"  # Equal
    NE = "ne"  # Not equal
    GT = "gt"  # Greater than
    GTE = "gte"  # Greater than or equal
    LT = "lt"  # Less than
    LTE = "lte"  # Less than or equal
    IN = "in"  # In list
    CONTAINS = "contains"  # String contains
    IS_NULL = "is_null"  # Is NULL
    IS_NOT_NULL = "is_not_null"  # Is not NULL


@dataclass(frozen=True)
class FilterCondition:
    """Single filter condition.

    Attributes:
        field: Field name to filter on.
        operator: Comparison operator.
        value: Value to compare against.

    Raises:
        ValueError: If operator requires list value but value is not a list.
    """

    field: str
    operator: FilterOperator
    value: Any

    def __post_init__(self) -> None:
        # Validate operator matches value type
        if self.operator in {FilterOperator.IN}:
            if not isinstance(self.value, (list, tuple, set)):
                raise ValueError(
                    f"IN operator requires list/tuple/set value, got {type(self.value)}"
                )


# Filter expression can be a single condition, AND combination (tuple), or OR combination (list)
# Use string annotation for recursive type definition
FilterExpression = Union[
    FilterCondition,  # Single condition
    "tuple[FilterExpression, ...]",  # AND combination
    "list[FilterExpression]",  # OR combination
]


class MetadataStore(ABC):
    """Control-plane metadata storage contract."""

    @abstractmethod
    async def get_collection(self, collection_name: str) -> CollectionInfo:
        """Read collection metadata.

        Args:
            collection_name: Target collection name.

        Returns:
            Collection metadata.

        Raises:
            ValueError: If collection is not found.
        """

    @abstractmethod
    async def save_collection(self, collection: CollectionInfo) -> None:
        """Create or update collection metadata."""

    async def save_collections(self, collections: Sequence[CollectionInfo]) -> None:
        """Create or update multiple collection metadata rows."""
        for collection in collections:
            await self.save_collection(collection)

    @abstractmethod
    async def list_collections(
        self,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> Sequence[CollectionInfo]:
        """List persisted collection metadata rows visible to the caller."""

    @abstractmethod
    async def delete_collection(self, collection_name: str) -> None:
        """Delete collection metadata entry."""

    @abstractmethod
    async def delete_collection_metadata(
        self,
        collection_name: str,
        user_id: Optional[int],
        is_admin: bool = False,
        delete_orphaned_metadata: bool = False,
    ) -> dict[str, int]:
        """Delete persisted metadata/config rows for a collection."""

    @abstractmethod
    async def rename_collection(self, old_name: str, new_name: str) -> None:
        """Rename persisted control-plane keys after a data-plane collection rename.

        Updates rows that gate :meth:`list_collections` visibility (for example
        per-tenant config rows and aggregate metadata) so they stay aligned with
        vector tables when the ``collection`` / ``name`` fields change.

        Args:
            old_name: Previous collection name (sanitized by the caller).
            new_name: Target collection name (sanitized by the caller).
        """

    @abstractmethod
    async def ensure_collection_metadata_table(self) -> None:
        """Ensure control-plane metadata table exists."""

    @abstractmethod
    async def save_collection_config(
        self,
        collection: str,
        config_json: str,
        user_id: int,
    ) -> None:
        """Save collection ingestion configuration.

        Args:
            collection: Collection name.
            config_json: JSON string of IngestionConfig.
            user_id: User ID for multi-tenancy.
        """

    @abstractmethod
    async def get_collection_config(
        self,
        collection: str,
        user_id: Optional[int],
        is_admin: bool = False,
    ) -> str | None:
        """Get collection ingestion configuration.

        Args:
            collection: Collection name.
            user_id: User ID for multi-tenancy. None is treated as 0 for non-admin,
                and as "load all configs" for admin mode.
            is_admin: Whether user has admin privileges (bypasses user_id filter).

        Returns:
            Config JSON string if found, None otherwise.
        """

    @abstractmethod
    def get_raw_connection(self) -> Any:
        """Return raw backend connection for legacy compatibility paths.

        The returned object conforms to the DatabaseConnection protocol but
        uses Any type to avoid importing backend-specific types.
        """


class VectorIndexStore(ABC):
    """Vector/data-plane storage contract.

    Phase 1A Option C: Hybrid sync/async methods for gradual migration.
    Sync methods provide backward compatibility; async methods enable
    non-blocking operations in async contexts (FastAPI, etc.).
    """

    @abstractmethod
    def list_document_records(
        self,
        collection_name: Optional[str],
        user_id: Optional[int],
        is_admin: bool,
        max_results: int = DEFAULT_VECTOR_STORE_SCAN_LIMIT,
    ) -> List[DocumentRecord]:
        """List document records from vector index side.

        Args:
            collection_name: Optional collection name filter. If None, lists records across all collections.
            user_id: User ID for multi-tenancy filtering.
            is_admin: Whether the user has admin privileges.
            max_results: Maximum records to return.
        """

    @abstractmethod
    def count_documents_grouped_by_collection(
        self,
        collection_names: Sequence[str],
        user_id: Optional[int],
        is_admin: bool,
    ) -> Dict[str, int]:
        """Count documents grouped by collection.

        Applies the same multi-tenancy filter semantics as other vector store
        reads: when ``is_admin`` is False, results are filtered by ``user_id``.

        Args:
            collection_names: Target collection names to include.
            user_id: User ID for multi-tenancy filtering.
            is_admin: Whether the caller has admin privileges.

        Returns:
            Mapping ``collection_name -> row_count`` for the requested names.
            Collections not present in results are treated as ``0``.
        """

    @abstractmethod
    def rename_collection_data(
        self,
        collection_name: str,
        new_name: str,
    ) -> List[str]:
        """Rename collection key across vector-side tables.

        Returns:
            Warning messages generated during best-effort updates.
        """

    @abstractmethod
    def delete_collection_data(
        self,
        collection_name: str,
        user_id: Optional[int],
        is_admin: bool,
        warnings_out: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """Delete all data for a collection from vector-side tables.

        Args:
            collection_name: Name of the collection to delete.
            user_id: User ID for tenant-scoped deletion.
            is_admin: Whether the caller can delete across tenants.
            warnings_out: Optional list to append best-effort deletion warnings to.

        Returns:
            Dictionary mapping table names to deleted row counts.
        """

    @abstractmethod
    def delete_document_data(
        self,
        collection_name: str,
        doc_id: str,
        user_id: Optional[int],
        is_admin: bool,
    ) -> Dict[str, int]:
        """Delete all vector-side data for a document.

        Args:
            collection_name: Name of the collection.
            doc_id: Document identifier.
            user_id: User ID for tenant-scoped deletion.
            is_admin: Whether the caller can delete across tenants.

        Returns:
            Dictionary mapping table names to deleted row counts.
        """

    @abstractmethod
    def aggregate_collection_stats(
        self,
        user_id: Optional[int],
        is_admin: bool,
    ) -> Dict[str, Dict[str, int]]:
        """Aggregate statistics for all collections.

        Returns:
            Dictionary mapping collection names to their stats:
            {
                "collection_name": {
                    "documents": int,
                    "parses": int,
                    "chunks": int,
                    "embeddings": int,
                }
            }
        """

    @abstractmethod
    def aggregate_document_stats(
        self,
        collection_name: str,
        doc_id: str,
        user_id: Optional[int],
        is_admin: bool,
    ) -> Dict[str, int]:
        """Aggregate statistics for a single document.

        Returns:
            Dictionary with counts:
            {
                "documents": int,
                "parses": int,
                "chunks": int,
                "embeddings": int,
            }
        """

    @abstractmethod
    def list_table_names(self) -> Sequence[str]:
        """List backend table names."""

    @abstractmethod
    def get_vector_dimension(self, table_name: str) -> Optional[int]:
        """Get the vector dimension from a table's schema.

        Reads the vector field's fixed_size dimension from the table schema.
        Returns None if the vector field is variable-length or dimension cannot
        be determined.

        Args:
            table_name: Name of the embeddings table to inspect.

        Returns:
            Vector dimension as int, or None if variable-length/unavailable.
        """

    @abstractmethod
    def open_embeddings_table(self, model_tag: str) -> Tuple[Any, str]:
        """Open embeddings table with legacy fallback support.

        Tries the primary Hub ID-based table name first, then falls back
        to legacy provider-based naming if the primary doesn't exist.

        This method encapsulates the legacy fallback logic for embeddings tables,
        providing a single source of truth for table name resolution.

        Args:
            model_tag: Model tag for the embeddings table.

        Returns:
            Tuple of (table_object, actual_table_name_used).

        Raises:
            DatabaseOperationError: If neither primary nor legacy table exists.
        """

    @abstractmethod
    def iter_batches(
        self,
        table_name: str,
        columns: Optional[Sequence[str]] = None,
        batch_size: int = 1000,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> Iterator[Any]:
        """Iterate over table data in batches (sync).

        Yields backend-specific batch objects (e.g., PyArrow RecordBatch).
        This method is designed for memory-efficient processing of large tables.

        Args:
            table_name: Name of table to iterate.
            columns: Optional columns to select. If None, selects all columns.
            batch_size: Rows per batch.
            filters: Optional filter criteria (key-value pairs for equality).
            user_id: Optional user filter for multi-tenancy.
            is_admin: Admin privilege flag.

        Yields:
            Backend-specific batch objects (e.g., PyArrow RecordBatch).
        """

    @abstractmethod
    def count_rows(
        self,
        table_name: str,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> int:
        """Count rows in a table with optional filters (sync).

        Args:
            table_name: Name of table to count.
            filters: Optional filter criteria (key-value pairs for equality).
            user_id: Optional user filter for multi-tenancy.
            is_admin: Admin privilege flag.

        Returns:
            Row count.

        Raises:
            DatabaseOperationError: If table cannot be opened or count fails.
        """

    def count_rows_or_zero(
        self,
        table_name: str,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> int:
        """Count rows in a table, returning 0 if table doesn't exist.

        This is a convenience method for existence checks where a missing table
        should be treated as "no data" rather than an error.

        Args:
            table_name: Name of table to count.
            filters: Optional filter criteria (key-value pairs for equality).
            user_id: Optional user filter for multi-tenancy.
            is_admin: Admin privilege flag.

        Returns:
            Row count, or 0 if table doesn't exist or count fails.
        """
        from ..core.exceptions import DatabaseOperationError

        try:
            return self.count_rows(table_name, filters, user_id, is_admin)
        except DatabaseOperationError as e:
            logger.debug(
                "count_rows_or_zero suppressed error for table '%s': %s",
                table_name,
                e,
            )
            return 0

    @abstractmethod
    def aggregate_document_counts(
        self,
        table_name: str,
        doc_id_column: str,
        collection_name: str,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> Dict[str, int]:
        """Aggregate records per document for a specific table.

        Args:
            table_name: Table to aggregate from.
            doc_id_column: Column containing document IDs.
            collection_name: Collection to scope to.
            user_id: Optional user filter.
            is_admin: Admin privilege flag.

        Returns:
            Dictionary mapping doc_id to count.
        """

    @abstractmethod
    def build_filter_expression(
        self,
        filters: Optional[FilterExpression],
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> Optional[str]:
        """Convert abstract filter expression to backend-specific syntax.

        Args:
            filters: Abstract filter expression.
            user_id: Optional user for multi-tenancy.
            is_admin: Admin privilege flag.

        Returns:
            Backend-specific filter string, or None if no filters.
        """

    @abstractmethod
    def upsert_documents(self, records: List[Dict[str, Any]]) -> None:
        """Upsert document records (sync).

        Args:
            records: List of document record dictionaries to upsert.
        """

    @abstractmethod
    def upsert_parses(self, records: List[Dict[str, Any]]) -> None:
        """Upsert parse records (sync).

        Args:
            records: List of parse record dictionaries to upsert.
        """

    @abstractmethod
    def upsert_chunks(self, records: List[Dict[str, Any]]) -> None:
        """Upsert chunk records (sync).

        Args:
            records: List of chunk record dictionaries to upsert.
        """

    @abstractmethod
    def upsert_embeddings(self, model_tag: str, records: List[Dict[str, Any]]) -> None:
        """Upsert embedding records (sync).

        Args:
            model_tag: Model tag for the embeddings table.
            records: List of embedding record dictionaries to upsert.
        """

    @abstractmethod
    def create_index(self, model_tag: str, readonly: bool = False) -> IndexResult:
        """Create or check vector index for embeddings table.

        Args:
            model_tag: Model tag for the embeddings table.
            readonly: If True, don't trigger index creation.

        Returns:
            IndexResult containing status, advice, and FTS enabled state.
        """

    @abstractmethod
    def search_vectors(
        self,
        table_name: str,
        query_vector: List[float],
        *,
        top_k: int,
        filters: Optional[FilterExpression] = None,
        vector_column_name: str = "vector",
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> List[Dict[str, Any]]:
        """Execute vector search (sync).

        Args:
            table_name: Name of embeddings table to search.
            query_vector: Query vector for similarity search.
            top_k: Number of top results to return.
            filters: Optional abstract filter expression.
            vector_column_name: Name of vector column (default "vector").
            user_id: Optional user ID for multi-tenancy filtering.
            is_admin: Whether the user has admin privileges.

        Returns:
            List of search result dictionaries with keys:
            - doc_id: Document ID
            - chunk_id: Chunk ID
            - text: Chunk text
            - _distance: Distance score (lower is better)
            - metadata: Additional metadata
        """

    def search_vectors_by_model(
        self,
        model_tag: str,
        query_vector: List[float],
        *,
        top_k: int,
        filters: Optional[FilterExpression] = None,
        vector_column_name: str = "vector",
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> List[Dict[str, Any]]:
        """Convenience method: search vectors by model_tag with automatic table resolution.

        This method combines open_embeddings_table() + search_vectors() for
        simpler API when searching by model_tag.

        Args:
            model_tag: Model tag for the embeddings table.
            query_vector: Query vector for similarity search.
            top_k: Number of top results to return.
            filters: Optional abstract filter expression.
            vector_column_name: Name of vector column (default "vector").
            user_id: Optional user ID for multi-tenancy filtering.
            is_admin: Whether the user has admin privileges.

        Returns:
            List of search result dictionaries with keys:
            - doc_id: Document ID
            - chunk_id: Chunk ID
            - text: Chunk text
            - _distance: Distance score (lower is better)
            - metadata: Additional metadata
        """
        _table, table_name = self.open_embeddings_table(model_tag)
        return self.search_vectors(
            table_name=table_name,
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
            vector_column_name=vector_column_name,
            user_id=user_id,
            is_admin=is_admin,
        )

    # --- Async variants (Phase 1A Option C: Hybrid approach) ---

    @abstractmethod
    async def search_vectors_async(
        self,
        table_name: str,
        query_vector: List[float],
        *,
        top_k: int,
        filters: Optional[FilterExpression] = None,
        vector_column_name: str = "vector",
    ) -> List[Dict[str, Any]]:
        """Execute vector search (async).

        Args:
            table_name: Name of embeddings table to search.
            query_vector: Query vector for similarity search.
            top_k: Number of top results to return.
            filters: Optional abstract filter expression.
            vector_column_name: Name of vector column (default "vector").

        Returns:
            List of search result dictionaries with keys:
            - doc_id: Document ID
            - chunk_id: Chunk ID
            - text: Chunk text
            - _distance: Distance score (lower is better)
            - metadata: Additional metadata
        """

    async def search_vectors_by_model_async(
        self,
        model_tag: str,
        query_vector: List[float],
        *,
        top_k: int,
        filters: Optional[FilterExpression] = None,
        vector_column_name: str = "vector",
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> List[Dict[str, Any]]:
        """Convenience method: search vectors by model_tag with automatic table resolution (async).

        This method combines open_embeddings_table() + search_vectors_async() for
        simpler API when searching by model_tag.

        Args:
            model_tag: Model tag for the embeddings table.
            query_vector: Query vector for similarity search.
            top_k: Number of top results to return.
            filters: Optional abstract filter expression.
            vector_column_name: Name of vector column (default "vector").
            user_id: Optional user ID for multi-tenancy filtering.
            is_admin: Whether the user has admin privileges.

        Returns:
            List of search result dictionaries with keys:
            - doc_id: Document ID
            - chunk_id: Chunk ID
            - text: Chunk text
            - _distance: Distance score (lower is better)
            - metadata: Additional metadata
        """
        _table, table_name = self.open_embeddings_table(model_tag)
        return await self.search_vectors_async(
            table_name=table_name,
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
            vector_column_name=vector_column_name,
        )

    @abstractmethod
    async def search_fts_async(
        self,
        table_name: str,
        query_text: str,
        *,
        top_k: int,
        filters: Optional[FilterExpression] = None,
        text_column_name: str = "text",
    ) -> List[Dict[str, Any]]:
        """Execute full-text search (async).

        Args:
            table_name: Name of embeddings/table to search (must have FTS index).
            query_text: Query text for full-text search.
            top_k: Number of top results to return.
            filters: Optional abstract filter expression.
            text_column_name: Name of text column with FTS index (default "text").

        Returns:
            List of search result dictionaries with keys:
            - doc_id: Document ID
            - chunk_id: Chunk ID
            - text: Chunk text
            - _score: TF-IDF score (higher is better)
            - metadata: Additional metadata

        Raises:
            DatabaseOperationError: If FTS index is not configured or search fails.
        """

    async def search_fts_by_model_async(
        self,
        model_tag: str,
        query_text: str,
        *,
        top_k: int,
        filters: Optional[FilterExpression] = None,
        text_column_name: str = "text",
    ) -> List[Dict[str, Any]]:
        """Convenience method: search FTS by model_tag with automatic table resolution.

        This method combines open_embeddings_table() + search_fts_async() for
        simpler API when searching by model_tag.

        Args:
            model_tag: Model tag for the embeddings table.
            query_text: Query text for full-text search.
            top_k: Number of top results to return.
            filters: Optional abstract filter expression.
            text_column_name: Name of text column with FTS index (default "text").

        Returns:
            List of search result dictionaries with keys:
            - doc_id: Document ID
            - chunk_id: Chunk ID
            - text: Chunk text
            - _score: TF-IDF score (higher is better)
            - metadata: Additional metadata

        Raises:
            DatabaseOperationError: If FTS index is not configured or search fails.
        """
        _table, table_name = self.open_embeddings_table(model_tag)
        return await self.search_fts_async(
            table_name=table_name,
            query_text=query_text,
            top_k=top_k,
            filters=filters,
            text_column_name=text_column_name,
        )

    @abstractmethod
    async def iter_batches_async(
        self,
        table_name: str,
        columns: Optional[Sequence[str]] = None,
        batch_size: int = 1000,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> Any:  # Returns AsyncIterator (async generator), but mypy has issues with async def + AsyncIterator return type
        """Iterate over table data in batches (async).

        This is an async generator that yields backend-specific batch objects
        (e.g., PyArrow RecordBatch). Use with: async for batch in iter_batches_async(...)

        Args:
            table_name: Name of table to iterate.
            columns: Optional columns to select.
            batch_size: Rows per batch.
            filters: Optional filter criteria.
            user_id: Optional user filter for multi-tenancy.
            is_admin: Admin privilege flag.

        Yields:
            Backend-specific batch objects (PyArrow RecordBatch).
        """

    @abstractmethod
    async def count_rows_async(
        self,
        table_name: str,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> int:
        """Count rows in a table with optional filters (async).

        Args:
            table_name: Name of table to count.
            filters: Optional filter criteria.
            user_id: Optional user filter.
            is_admin: Admin privilege flag.

        Returns:
            Row count (0 on error).
        """

    @abstractmethod
    async def get_vector_dimension_async(self, table_name: str) -> Optional[int]:
        """Get the vector dimension from a table's schema (async).

        Args:
            table_name: Name of the embeddings table to inspect.

        Returns:
            Vector dimension as int, or None if variable-length/unavailable.

        Note: Current implementation uses sync operations under the hood.
        True async I/O will be added in Phase 1B with RDB backend.
        """

    @abstractmethod
    async def upsert_documents_async(self, records: List[Dict[str, Any]]) -> None:
        """Upsert document records (async).

        Args:
            records: List of document record dictionaries to upsert.
        """

    @abstractmethod
    async def upsert_chunks_async(self, records: List[Dict[str, Any]]) -> None:
        """Upsert chunk records (async).

        Args:
            records: List of chunk record dictionaries to upsert.
        """

    @abstractmethod
    async def upsert_embeddings_async(
        self, model_tag: str, records: List[Dict[str, Any]]
    ) -> None:
        """Upsert embedding records (async).

        Args:
            model_tag: Model tag for the embeddings table.
            records: List of embedding record dictionaries to upsert.
        """

    # --- Index Management (Phase 1A Part 2) ---

    @abstractmethod
    def should_reindex(
        self,
        table_name: str,
        total_upserted: int,
        policy: IndexPolicy,
    ) -> bool:
        """Determine if reindex should be triggered.

        Args:
            table_name: Embeddings table name.
            total_upserted: Total upserted records since last index.
            policy: Index policy with reindex thresholds.

        Returns:
            True if reindex should be triggered.
        """

    @abstractmethod
    def trigger_reindex(self, table_name: str) -> bool:
        """Trigger index rebuild operation.

        Args:
            table_name: Embeddings table name.

        Returns:
            True if reindex was triggered successfully.
        """

    # --- Async index management variants ---

    @abstractmethod
    async def should_reindex_async(
        self,
        table_name: str,
        total_upserted: int,
        policy: IndexPolicy,
    ) -> bool:
        """Async version of should_reindex.

        Args:
            table_name: Embeddings table name.
            total_upserted: Total upserted records since last index.
            policy: Index policy with reindex thresholds.

        Returns:
            True if reindex should be triggered.

        Note: Current implementation uses sync operations under the hood.
        True async I/O will be added in Phase 1B with RDB backend.
        """

    @abstractmethod
    async def trigger_reindex_async(self, table_name: str) -> bool:
        """Async version of trigger_reindex.

        Args:
            table_name: Embeddings table name.

        Returns:
            True if reindex was triggered successfully.

        Note: Current implementation uses sync operations under the hood.
        True async I/O will be added in Phase 1B with RDB backend.
        """

    @abstractmethod
    def migrate_embeddings_table(
        self,
        model_id: str,
        batch_size: int = 1000,
    ) -> dict[str, Any]:
        """Migrate legacy embeddings table to Hub ID-based naming.

        This method copies data from a legacy table (embeddings_{model_name})
        to a new Hub ID-based table (embeddings_{hub_id}), rewriting the
        per-row ``model`` field to the Hub model ID.

        This is the proper location for migration logic, as it's part of
        the storage implementation. Migration should be run during maintenance
        windows, not during normal read operations.

        Args:
            model_id: Hub model ID to migrate (e.g., "text-embedding-ada-002").
            batch_size: Number of rows to copy per batch.

        Returns:
            Dictionary with migration results:
            {
                "success": bool,
                "source_table": str (legacy table name),
                "target_table": str (Hub ID table name),
                "rows_migrated": int,
                "error": str | None (if success=False)
            }

        Raises:
            VectorValidationError: If model_id is empty.
            DatabaseOperationError: If migration fails.

        Note:
            - This method uses file-based locking to prevent concurrent migrations.
            - The migration is idempotent and can be safely re-run.
            - Source table is preserved after migration.
        """
        pass

    @abstractmethod
    def get_raw_connection(self) -> Any:
        """Return raw backend connection for legacy compatibility paths.

        The returned object conforms to the DatabaseConnection protocol but
        uses Any type to avoid importing backend-specific types.

        DEPRECATED: Use specific upsert methods instead for write operations.
        """


class KBWriteCoordinator(ABC):
    """Contract for knowledge-base write/delete orchestration (Phase 1A shell).

    Phase 1A exposes only accessors to the configured metadata and vector
    stores; concrete implementations delegate without extra coordination.
    This type is a stable injection point for future write-path behavior such
    as distributed locking, write batching, and conflict resolution across
    metadata and vector backends.
    """

    @abstractmethod
    def metadata_store(self) -> MetadataStore:
        """Return configured metadata store."""

    @abstractmethod
    def vector_index_store(self) -> VectorIndexStore:
        """Return configured vector index store."""


# ============================================================================
# Phase 1A Part 2: Additional Store Contracts
# ============================================================================


class IngestionStatusStore(ABC):
    """Ingestion status tracking contract.

    Manages ingestion_runs table for tracking document processing status.

    Phase 1A Option C: Hybrid sync/async methods for gradual migration.
    Sync methods provide backward compatibility; async methods enable
    non-blocking operations in async contexts.
    """

    # --- Sync methods ---

    @abstractmethod
    def write_ingestion_status(
        self,
        collection: str,
        doc_id: str,
        *,
        status: str,
        message: Optional[str] = None,
        parse_hash: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> None:
        """Write ingestion status record (sync).

        Args:
            collection: Collection name.
            doc_id: Document ID.
            status: Status value (e.g., 'pending', 'processing', 'success', 'failed').
            message: Optional status message or error description.
            parse_hash: Optional hash of the parsed document for change detection.
            user_id: Optional user ID for multi-tenancy.

        Raises:
            DatabaseOperationError: If write operation fails.
        """

    @abstractmethod
    def load_ingestion_status(
        self,
        collection: Optional[str] = None,
        doc_id: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> List[Dict[str, Any]]:
        """Load ingestion status records (sync).

        Args:
            collection: Optional collection name to filter by.
            doc_id: Optional document ID to filter by.
            user_id: Optional user ID for multi-tenancy filtering.
            is_admin: Whether user has admin privileges (bypasses filtering).

        Returns:
            List of ingestion status records.

        Raises:
            DatabaseOperationError: If read operation fails.
        """

    @abstractmethod
    def clear_ingestion_status(
        self,
        collection: str,
        doc_id: str,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> None:
        """Remove ingestion status record (sync).

        Args:
            collection: Collection name.
            doc_id: Document ID.
            user_id: Optional user ID for multi-tenancy filtering.
            is_admin: Whether user has admin privileges (bypasses filtering).

        Raises:
            DatabaseOperationError: If delete operation fails.
        """

    # --- Async methods ---

    @abstractmethod
    async def write_ingestion_status_async(
        self,
        collection: str,
        doc_id: str,
        *,
        status: str,
        message: Optional[str] = None,
        parse_hash: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> None:
        """Write ingestion status record (async).

        Args:
            collection: Collection name.
            doc_id: Document ID.
            status: Status value.
            message: Optional status message.
            parse_hash: Optional parse hash.
            user_id: Optional user ID.

        Raises:
            DatabaseOperationError: If write operation fails.
        """

    @abstractmethod
    async def load_ingestion_status_async(
        self,
        collection: Optional[str] = None,
        doc_id: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> List[Dict[str, Any]]:
        """Load ingestion status records (async).

        Args:
            collection: Optional collection name to filter by.
            doc_id: Optional document ID to filter by.
            user_id: Optional user ID for multi-tenancy.
            is_admin: Whether user has admin privileges.

        Returns:
            List of ingestion status records.

        Raises:
            DatabaseOperationError: If read operation fails.
        """

    @abstractmethod
    async def clear_ingestion_status_async(
        self,
        collection: str,
        doc_id: str,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> None:
        """Remove ingestion status record (async).

        Args:
            collection: Collection name.
            doc_id: Document ID.
            user_id: Optional user ID for multi-tenancy.
            is_admin: Whether user has admin privileges.

        Raises:
            DatabaseOperationError: If delete operation fails.
        """


class PromptTemplateStore(ABC):
    """Prompt template management contract.

    Manages prompt_templates table for storing and retrieving prompt templates.

    Phase 1A Option C: Hybrid sync/async methods for gradual migration.
    """

    @abstractmethod
    def save_prompt_template(
        self,
        name: str,
        template: str,
        user_id: Optional[int] = None,
        metadata: Optional[str] = None,
    ) -> str:
        """Save or update a prompt template.

        Args:
            name: Template name (used for version grouping)
            template: Template content
            user_id: User ID for multi-tenancy
            metadata: Optional metadata as JSON string

        Returns:
            Template ID (UUID string)
        """

    @abstractmethod
    def get_prompt_template(
        self,
        template_id: str,
        user_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get a prompt template by ID.

        Args:
            template_id: Template UUID
            user_id: User ID for multi-tenancy

        Returns:
            Template data dict or None if not found
        """

    @abstractmethod
    def get_latest_prompt_template(
        self,
        name: str,
        user_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get the latest version of a prompt template by name.

        Args:
            name: Template name
            user_id: User ID for multi-tenancy

        Returns:
            Template data dict or None if not found
        """

    @abstractmethod
    def list_prompt_templates(
        self,
        name_filter: Optional[str] = None,
        latest_only: bool = False,
        user_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List prompt templates with optional filtering.

        Args:
            name_filter: Filter by template name (partial match)
            latest_only: Only return latest versions
            user_id: User ID for multi-tenancy
            limit: Maximum results to return

        Returns:
            List of template data dicts
        """

    @abstractmethod
    def delete_prompt_template(
        self,
        template_id: str,
        user_id: Optional[int] = None,
    ) -> bool:
        """Delete a prompt template by ID.

        Args:
            template_id: Template UUID
            user_id: User ID for multi-tenancy

        Returns:
            True if deleted, False if not found
        """

    @abstractmethod
    def update_metadata(
        self,
        template_id: str,
        metadata: Optional[str],
        user_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update metadata only, keeping same version and ID.

        Args:
            template_id: Template UUID
            metadata: New metadata as JSON string
            user_id: User ID for multi-tenancy

        Returns:
            Updated template data dict or None if not found
        """

    @abstractmethod
    def delete_by_name(
        self,
        name: str,
        version: Optional[int] = None,
        user_id: Optional[int] = None,
    ) -> int:
        """Delete template(s) by name.

        Handles is_latest flag updates for remaining versions.

        Args:
            name: Template name
            version: Specific version to delete (None = delete all versions)
            user_id: User ID for multi-tenancy

        Returns:
            Number of templates deleted

        Raises:
            DocumentNotFoundError: If template not found
        """

    @abstractmethod
    def get_versions_by_name(
        self,
        name: str,
        user_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get all versions of a template by name.

        Args:
            name: Template name
            user_id: User ID for multi-tenancy
            limit: Maximum results to return

        Returns:
            List of template data dicts
        """

    # --- Async methods (delegate to sync for Phase 1A) ---

    @abstractmethod
    async def save_prompt_template_async(
        self,
        name: str,
        template: str,
        user_id: Optional[int] = None,
        metadata: Optional[str] = None,
    ) -> str:
        """Async version of save_prompt_template."""

    @abstractmethod
    async def get_prompt_template_async(
        self,
        template_id: str,
        user_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Async version of get_prompt_template."""

    @abstractmethod
    async def get_latest_prompt_template_async(
        self,
        name: str,
        user_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Async version of get_latest_prompt_template."""

    @abstractmethod
    async def list_prompt_templates_async(
        self,
        name_filter: Optional[str] = None,
        latest_only: bool = False,
        user_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Async version of list_prompt_templates."""

    @abstractmethod
    async def delete_prompt_template_async(
        self,
        template_id: str,
        user_id: Optional[int] = None,
    ) -> bool:
        """Async version of delete_prompt_template."""

    @abstractmethod
    async def update_metadata_async(
        self,
        template_id: str,
        metadata: Optional[str],
        user_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Async version of update_metadata."""

    @abstractmethod
    async def delete_by_name_async(
        self,
        name: str,
        version: Optional[int] = None,
        user_id: Optional[int] = None,
    ) -> int:
        """Async version of delete_by_name."""

    @abstractmethod
    async def get_versions_by_name_async(
        self,
        name: str,
        user_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Async version of get_versions_by_name."""


class MainPointerStore(ABC):
    """Main pointer management contract for version control.

    Manages main_pointers table for tracking current versions across
    processing stages (parse, chunk, embed).

    Phase 1A Option C: Hybrid sync/async methods for gradual migration.

    NOTE: user_id parameter is included for API consistency but is not
    currently stored in the main_pointers table schema. A schema migration
    is required to add user_id support for multi-tenancy.
    """

    @abstractmethod
    def set_main_pointer(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        semantic_id: str,
        technical_id: str,
        model_tag: Optional[str] = None,
        operator: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> None:
        """Set or update a main pointer for a document.

        Args:
            collection: Collection name
            doc_id: Document ID
            step_type: Processing stage (parse, chunk, embed)
            semantic_id: Semantic identifier for the version (e.g., parse_id)
            technical_id: Technical identifier/hash for the version
            model_tag: Optional model tag for model-specific pointers
            operator: Optional operator who made the change
            user_id: Optional user ID (not stored, reserved for future use)
        """

    @abstractmethod
    def get_main_pointer(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get a main pointer for a document.

        Args:
            collection: Collection name
            doc_id: Document ID
            step_type: Processing stage (parse, chunk, embed)
            model_tag: Optional model tag for model-specific pointers
            user_id: Optional user ID (not used, reserved for future)

        Returns:
            Pointer data dict with keys: collection, doc_id, step_type,
            model_tag, semantic_id, technical_id, created_at, updated_at,
            operator. Returns None if not found.
        """

    @abstractmethod
    def list_main_pointers(
        self,
        collection: str,
        doc_id: Optional[str] = None,
        user_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List main pointers for a collection.

        Args:
            collection: Collection name
            doc_id: Optional document ID filter
            user_id: Optional user ID (not used, reserved for future)
            limit: Maximum results to return

        Returns:
            List of pointer data dicts
        """

    @abstractmethod
    def delete_main_pointer(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> bool:
        """Delete a main pointer.

        Args:
            collection: Collection name
            doc_id: Document ID
            step_type: Processing stage (parse, chunk, embed)
            model_tag: Optional model tag for model-specific pointers
            user_id: Optional user ID (not used, reserved for future)

        Returns:
            True if deleted, False if not found
        """

    # --- Async methods (delegate to sync for Phase 1A) ---

    @abstractmethod
    async def set_main_pointer_async(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        semantic_id: str,
        technical_id: str,
        model_tag: Optional[str] = None,
        operator: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> None:
        """Async version of set_main_pointer."""

    @abstractmethod
    async def get_main_pointer_async(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Async version of get_main_pointer."""

    @abstractmethod
    async def list_main_pointers_async(
        self,
        collection: str,
        doc_id: Optional[str] = None,
        user_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Async version of list_main_pointers."""

    @abstractmethod
    async def delete_main_pointer_async(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> bool:
        """Async version of delete_main_pointer."""
