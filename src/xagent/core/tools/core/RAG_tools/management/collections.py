"""Collection management utilities for RAG pipelines.

This module provides utilities for managing collections in a RAG (Retrieval-Augmented Generation)
system, including listing collections, managing documents, and handling deletion operations.
"""

from __future__ import annotations

import json
import logging
import warnings as py_warnings
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Set

import pyarrow as pa  # type: ignore
from lancedb.db import DBConnection

from ..core.config import (
    DEFAULT_LANCEDB_SCAN_BATCH_SIZE,
    DEFAULT_VECTOR_STORE_EXTENDED_SCAN_LIMIT,
)
from ..core.schemas import (
    CollectionDocumentMetadata,
    CollectionInfo,
    CollectionOperationDetail,
    CollectionOperationResult,
    DocumentListResult,
    DocumentOperationResult,
    DocumentProcessingStatus,
    DocumentStats,
    DocumentStatsResult,
    DocumentSummary,
    IngestionConfig,
    ListCollectionsResult,
)
from ..LanceDB.model_tag_utils import embeddings_table_name
from ..management.status import (
    clear_ingestion_status,
    load_ingestion_status,
    write_ingestion_status,
)
from ..storage.factory import get_metadata_store, get_vector_index_store
from ..utils.string_utils import build_lancedb_filter_expression, escape_lancedb_string
from ..utils.user_permissions import UserPermissions
from ..version_management.cascade_cleaner import cleanup_document_cascade

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = DEFAULT_LANCEDB_SCAN_BATCH_SIZE


def _iter_batches(
    conn: DBConnection,
    table_name: str,
    warnings: List[str],
    columns: Optional[Sequence[str]] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> Any:
    """Yield record batches from a LanceDB table while minimizing memory footprint.

    .. deprecated::
        This function is deprecated. Use VectorIndexStore.iter_batches() instead.
        This function will be removed in a future release.

    This generator function iterates through a LanceDB table in batches to
    minimize memory usage, with support for user filtering and column selection.

    Args:
        conn: LanceDB database connection
        table_name: Name of the table to iterate
        warnings: List to collect any warnings encountered
        columns: Optional list of column names to select
        batch_size: Number of rows per batch
        user_id: Optional user ID for multi-tenancy filtering
        is_admin: Whether user has admin privileges

    Yields:
        PyArrow RecordBatch objects containing the data
    """
    py_warnings.warn(
        "_iter_batches is deprecated, use VectorIndexStore.iter_batches() instead",
        DeprecationWarning,
        stacklevel=2,
    )

    try:
        table = conn.open_table(table_name)
    except Exception as exc:  # noqa: BLE001 - convert to warning
        message = f"Unable to open table '{table_name}': {exc}"
        logger.warning(message)
        warnings.append(message)
        return

    column_list = list(columns) if columns is not None else None

    # Apply user filter for multi-tenancy
    user_filter = UserPermissions.get_user_filter(user_id, is_admin)

    # Preferred path: streaming batches directly from LanceDB
    try:
        # Use filter if provided (for multi-tenancy)
        if user_filter is not None:
            for raw_batch in table.to_batches(
                filter=user_filter, batch_size=batch_size
            ):
                batch = raw_batch
                if column_list is not None:
                    arrays = []
                    names = []
                    for col_name in column_list:
                        idx = batch.schema.get_field_index(col_name)
                        if idx == -1:
                            continue
                        arrays.append(batch.column(idx))
                        names.append(col_name)
                    if not arrays:
                        continue
                    batch = pa.RecordBatch.from_arrays(arrays, names=names)
                if batch.num_rows > 0:
                    yield batch
            return

        for raw_batch in table.to_batches(batch_size=batch_size):
            batch = raw_batch
            if column_list is not None:
                arrays = []
                names = []
                for col_name in column_list:
                    idx = batch.schema.get_field_index(col_name)
                    if idx == -1:
                        continue
                    arrays.append(batch.column(idx))
                    names.append(col_name)
                if not arrays:
                    continue
                batch = pa.RecordBatch.from_arrays(arrays, names=names)
            if batch.num_rows > 0:
                yield batch
        return
    except Exception as exc:  # noqa: BLE001 - continue to Arrow fallback
        logger.debug("Batch streaming unavailable for table '%s': %s", table_name, exc)

    # Arrow fallback: materialize table as Arrow then iterate
    try:
        arrow_table = table.to_arrow()
    except Exception as exc:  # noqa: BLE001
        message = f"Unable to read table '{table_name}' via to_arrow(): {exc}"
        logger.warning(message)
        warnings.append(message)
        return

    # Apply user filter on Arrow table if needed
    if user_filter is not None and "user_id" in arrow_table.schema.names:
        try:
            # Parse filter and apply to Arrow table
            # Simple filter parsing for "user_id == X" or "user_id IS NULL"
            import re

            if UserPermissions.is_no_access_filter(user_filter):
                # Explicit unauthenticated no-access marker: return empty result directly.
                arrow_table = arrow_table.slice(0, 0)
            elif "user_id IS NULL" in user_filter:
                # Filter for NULL user_id
                mask = pa.compute.is_null(arrow_table["user_id"])
                arrow_table = arrow_table.filter(mask)
            elif "user_id ==" in user_filter:
                # Filter for specific user_id
                match = re.search(r"user_id == (-?\d+)", user_filter)
                if match:
                    user_val = int(match.group(1))
                    mask = pa.compute.equal(
                        arrow_table["user_id"], pa.scalar(user_val, type=pa.int64())
                    )
                    arrow_table = arrow_table.filter(mask)
        except Exception as filter_exc:
            logger.warning("Failed to apply user filter on Arrow table: %s", filter_exc)
            # Continue without filter if filtering fails

    if column_list is not None:
        try:
            arrow_table = arrow_table.select(column_list)
        except Exception as exc:  # noqa: BLE001
            message = (
                f"Table '{table_name}' missing expected columns {column_list}: {exc}"
            )
            logger.warning(message)
            warnings.append(message)
            return

    for batch in arrow_table.to_batches(max_chunksize=batch_size):
        if batch.num_rows > 0:
            yield batch


def _count_rows(
    conn: DBConnection,
    table_name: str,
    filters: Dict[str, str],
    warnings: List[str],
) -> int:
    """Count rows in a LanceDB table while handling failures gracefully.

    .. deprecated::
        This function is deprecated. Use VectorIndexStore.count_rows() instead.
        This function will be removed in a future release.

    This function counts rows in a LanceDB table with optional filters,
    returning 0 on any error and logging warnings.

    Args:
        conn: LanceDB database connection
        table_name: Name of the table to count rows in
        filters: Dictionary of field-value pairs to filter by
        warnings: List to collect any warnings encountered

    Returns:
        Number of rows matching the filter, or 0 on error
    """
    py_warnings.warn(
        "_count_rows is deprecated, use VectorIndexStore.count_rows() instead",
        DeprecationWarning,
        stacklevel=2,
    )

    try:
        table = conn.open_table(table_name)
    except Exception as exc:  # noqa: BLE001 - convert to warning
        message = f"Unable to open table '{table_name}': {exc}"
        logger.warning(message)
        warnings.append(message)
        return 0

    filter_expr = build_lancedb_filter_expression(filters)

    try:
        if filter_expr:
            return int(table.count_rows(filter_expr))
        return int(table.count_rows())
    except Exception as exc:  # noqa: BLE001 - convert to warning
        message = f"Failed to count rows in '{table_name}': {exc}"
        logger.warning(message)
        warnings.append(message)
        return 0


def _list_table_names(conn: DBConnection, warnings: List[str]) -> List[str]:
    """Return available LanceDB table names with graceful degradation.

    .. deprecated::
        This function is deprecated. Use VectorIndexStore.list_table_names() instead.
        This function will be removed in a future release.

    This function retrieves the list of table names from a LanceDB connection,
    handling errors gracefully by returning an empty list and logging warnings.

    Args:
        conn: LanceDB database connection
        warnings: List to collect any warnings encountered

    Returns:
        List of table names as strings, or empty list on error
    """
    py_warnings.warn(
        "_list_table_names is deprecated, use VectorIndexStore.list_table_names() instead",
        DeprecationWarning,
        stacklevel=2,
    )

    try:
        table_names_fn = getattr(conn, "table_names")
    except AttributeError as exc:
        message = f"LanceDB connection missing table_names(): {exc}"
        logger.warning(message)
        warnings.append(message)
        return []

    try:
        names = table_names_fn()
    except Exception as exc:  # noqa: BLE001 - convert to warning
        message = f"Failed to list LanceDB tables: {exc}"
        logger.warning(message)
        warnings.append(message)
        return []

    return [str(name) for name in names]


def _collect_doc_counts_for_collection(
    conn: DBConnection,
    table_name: str,
    doc_column: str,
    target_collection: str,
    warnings: List[str],
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> Dict[str, int]:
    """Aggregate per-document counts for the specified table within a collection.

    .. deprecated::
        This function is deprecated. Use VectorIndexStore.aggregate_document_counts() instead.
        This function will be removed in a future release.

    This function iterates through batches of a table and counts records
    per document for a specific collection.

    Args:
        conn: LanceDB database connection
        table_name: Name of the table to count from
        doc_column: Name of the column containing document IDs
        target_collection: Name of the collection to filter by
        warnings: List to collect any warnings encountered
        user_id: Optional user ID for multi-tenancy filtering
        is_admin: Whether user has admin privileges

    Returns:
        Dictionary mapping document IDs to their counts
    """
    py_warnings.warn(
        "_collect_doc_counts_for_collection is deprecated, use VectorIndexStore.aggregate_document_counts() instead",
        DeprecationWarning,
        stacklevel=2,
    )

    counts: Dict[str, int] = defaultdict(int)

    for batch in _iter_batches(
        conn,
        table_name,
        warnings,
        columns=["collection", doc_column],
        user_id=user_id,
        is_admin=is_admin,
    ):
        collection_idx = batch.schema.get_field_index("collection")
        doc_idx = batch.schema.get_field_index(doc_column)
        if collection_idx == -1 or doc_idx == -1:
            continue
        collection_array = batch.column(collection_idx)
        doc_array = batch.column(doc_idx)
        for idx in range(batch.num_rows):
            collection_raw = collection_array[idx].as_py()
            if not collection_raw or str(collection_raw) != target_collection:
                continue
            doc_raw = doc_array[idx].as_py()
            if not doc_raw:
                continue
            counts[str(doc_raw)] += 1

    return counts


def _collect_document_ids(
    conn: DBConnection,
    collection: str,
    warnings: List[str],
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> Set[str]:
    """Collect all known document identifiers for a collection.

    This function aggregates document IDs from multiple sources including
    the documents table, chunks table, embeddings tables, and status records.

    Args:
        conn: LanceDB database connection
        collection: Name of the collection to collect document IDs from
        warnings: List to collect any warnings encountered
        user_id: Optional user ID for multi-tenancy filtering
        is_admin: Whether user has admin privileges

    Returns:
        Set of unique document identifiers
    """

    doc_ids: Set[str] = set()

    for batch in _iter_batches(
        conn,
        "documents",
        warnings,
        columns=["collection", "doc_id"],
        user_id=user_id,
        is_admin=is_admin,
    ):
        collection_idx = batch.schema.get_field_index("collection")
        doc_idx = batch.schema.get_field_index("doc_id")
        if collection_idx == -1 or doc_idx == -1:
            continue
        collection_array = batch.column(collection_idx)
        doc_array = batch.column(doc_idx)
        for idx in range(batch.num_rows):
            collection_raw = collection_array[idx].as_py()
            doc_raw = doc_array[idx].as_py()
            if not collection_raw or not doc_raw:
                continue
            if str(collection_raw) == collection:
                doc_ids.add(str(doc_raw))

    chunk_docs = _collect_doc_counts_for_collection(
        conn, "chunks", "doc_id", collection, warnings, user_id, is_admin
    )
    doc_ids.update(chunk_docs.keys())

    for table_name in _list_table_names(conn, warnings):
        if not table_name.startswith("embeddings_"):
            continue
        embed_docs = _collect_doc_counts_for_collection(
            conn, table_name, "doc_id", collection, warnings, user_id, is_admin
        )
        doc_ids.update(embed_docs.keys())

    for status_entry in load_ingestion_status(collection=collection):
        raw_doc = status_entry.get("doc_id")
        if isinstance(raw_doc, str):
            doc_ids.add(raw_doc)

    return doc_ids


def _coerce_timestamp(value: Any) -> datetime | None:
    """Normalize timestamp-like values to aware datetime or None.

    This function attempts to convert various timestamp representations
    (datetime objects, Arrow timestamps, ISO format strings) to a
    standard Python datetime object.

    Args:
        value: The value to convert to datetime

    Returns:
        datetime object if conversion succeeds, None otherwise
    """

    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_pydatetime"):
        candidate = value.to_pydatetime()
        if isinstance(candidate, datetime):
            return candidate
        return None
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


async def _load_collection_ingestion_configs(
    collection_keys: List[str],
    user_id: Optional[int],
    is_admin: bool,
) -> Dict[str, IngestionConfig]:
    """Load ingestion configs for the given collections using metadata store rules.

    Args:
        collection_keys: Collection names returned by stats / document scan.
        user_id: Caller user id; None is treated as 0 for non-admin lookups.
        is_admin: When True, ``get_collection_config`` returns the latest config
            per collection across tenants.

    Returns:
        Map of collection name to parsed ingestion configuration.
    """
    metadata_store = get_metadata_store()
    collection_configs: Dict[str, IngestionConfig] = {}
    # Handle user_id=None explicitly: admin mode keeps None (load all configs),
    # non-admin mode converts to 0 (backward compatible)
    if is_admin and user_id is None:
        uid = None
    else:
        uid = 0 if user_id is None else user_id
    for collection in collection_keys:
        try:
            config_json = await metadata_store.get_collection_config(
                collection, uid, is_admin=is_admin
            )
            if not config_json:
                continue
            try:
                config_dict = json.loads(config_json)
                collection_configs[collection] = IngestionConfig(**config_dict)
            except Exception as e:
                logger.warning(
                    "Failed to parse config for collection %s: %s",
                    collection,
                    e,
                )
        except Exception as e:
            logger.debug("Could not load config for collection %s: %s", collection, e)
    return collection_configs


async def list_collections(
    user_id: Optional[int] = None, is_admin: bool = False
) -> ListCollectionsResult:
    """List all knowledge base collections along with aggregated statistics.

    This function returns a list of all available knowledge bases (collections)
    in the system, including document counts and metadata.

    Args:
        user_id: Optional user ID for filtering (for multi-tenancy).
        is_admin: Whether user has admin privileges.

    Returns:
        Aggregated collection metadata and status information for each
        knowledge base, including total documents, parses, chunks, and embeddings.
    """

    logger.info("Listing LanceDB collections")

    warnings: List[str] = []

    try:
        # Use storage abstraction for aggregation
        vector_store = get_vector_index_store()
        stats: Dict[str, Dict[str, int]] = vector_store.aggregate_collection_stats(
            user_id=user_id,
            is_admin=is_admin,
        )

        document_names: Dict[str, Set[str]] = defaultdict(set)
        document_metadata: Dict[str, List[CollectionDocumentMetadata]] = defaultdict(
            list
        )
        document_metadata_seen: Dict[str, Set[tuple[str, str, str]]] = defaultdict(set)

        def _normalize_optional_identifier(value: Any) -> Optional[str]:
            if not isinstance(value, str):
                return None
            normalized = value.strip()
            return normalized or None

        def _add_document_entry(
            collection_key: str,
            source_value: Any,
            doc_id_value: Any,
            file_id_value: Any,
        ) -> None:
            import os

            normalized_doc_id = _normalize_optional_identifier(doc_id_value)
            normalized_file_id = _normalize_optional_identifier(file_id_value)
            display_name = None
            if source_value:
                display_name = os.path.basename(str(source_value))
            display_name = (display_name or normalized_doc_id or "").strip()
            if not display_name:
                return

            document_names[collection_key].add(display_name)

            dedupe_key = (
                display_name,
                normalized_file_id or "",
                normalized_doc_id or "",
            )
            seen_keys = document_metadata_seen[collection_key]
            if dedupe_key in seen_keys:
                return
            seen_keys.add(dedupe_key)
            document_metadata[collection_key].append(
                CollectionDocumentMetadata(
                    filename=display_name,
                    file_id=normalized_file_id,
                    doc_id=normalized_doc_id,
                )
            )

        def _collect_document_names() -> None:
            for batch in vector_store.iter_batches(
                table_name="documents",
                columns=["collection", "source_path", "doc_id", "file_id"],
                user_id=user_id,
                is_admin=is_admin,
            ):
                collection_idx = batch.schema.get_field_index("collection")
                source_idx = batch.schema.get_field_index("source_path")
                doc_id_idx = batch.schema.get_field_index("doc_id")
                file_id_idx = batch.schema.get_field_index("file_id")
                if collection_idx == -1:
                    continue
                collection_array = batch.column(collection_idx)
                source_array = (
                    batch.column(source_idx)
                    if source_idx != -1
                    else pa.array([None] * batch.num_rows)
                )
                doc_id_array = (
                    batch.column(doc_id_idx)
                    if doc_id_idx != -1
                    else pa.array([None] * batch.num_rows)
                )
                file_id_array = (
                    batch.column(file_id_idx)
                    if file_id_idx != -1
                    else pa.array([None] * batch.num_rows)
                )
                for idx in range(batch.num_rows):
                    collection_raw = collection_array[idx].as_py()
                    if not collection_raw:
                        continue
                    collection_key = str(collection_raw)
                    _add_document_entry(
                        collection_key,
                        source_array[idx].as_py(),
                        doc_id_array[idx].as_py(),
                        file_id_array[idx].as_py(),
                    )

        _collect_document_names()

        collection_keys = sorted(stats.keys() | document_names.keys())

        # Load configs for collections (admin sees cross-tenant configs)
        collection_configs: Dict[str, IngestionConfig] = {}
        try:
            collection_configs = await _load_collection_ingestion_configs(
                collection_keys, user_id, is_admin
            )
        except Exception as e:
            logger.warning("Could not load collection configs: %s", e)

        # Ensure all collections have complete stats
        for collection in collection_keys:
            if collection not in stats:
                stats[collection] = {
                    "documents": 0,
                    "parses": 0,
                    "chunks": 0,
                    "embeddings": 0,
                }
            for key in ["documents", "parses", "chunks", "embeddings"]:
                if key not in stats[collection]:
                    stats[collection][key] = 0

        collections = [
            CollectionInfo(
                name=collection,
                documents=stats[collection]["documents"],
                parses=stats[collection]["parses"],
                chunks=stats[collection]["chunks"],
                embeddings=stats[collection]["embeddings"],
                processed_documents=stats[collection][
                    "parses"
                ],  # Use parses count as processed documents
                document_names=sorted(document_names[collection]),
                document_metadata=sorted(
                    document_metadata[collection],
                    key=lambda item: (
                        item.filename,
                        item.file_id or "",
                        item.doc_id or "",
                    ),
                ),
                ingestion_config=collection_configs.get(collection),
            )
            for collection in collection_keys
        ]

        message = f"Found {len(collections)} collections"
        logger.info(message)
        return ListCollectionsResult(
            status="success",
            collections=collections,
            total_count=len(collections),
            message=message,
            warnings=warnings,
        )

    except Exception as exc:  # noqa: BLE001 - convert to structured failure
        logger.error("Failed to list collections: %s", exc, exc_info=True)
        return ListCollectionsResult(
            status="error",
            collections=[],
            total_count=0,
            message=f"Failed to list collections: {exc}",
            warnings=warnings,
        )


def get_document_stats(
    collection: str,
    doc_id: str,
    model_tag: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> DocumentStatsResult:
    """Return statistics for a single document within a collection.

    Args:
        collection: Target collection name.
        doc_id: Document identifier inside the collection.
        model_tag: Optional embedding model name; when provided the
            statistics only include the corresponding embeddings table.
        user_id: Optional user ID for filtering (for multi-tenancy).
        is_admin: Whether user has admin privileges.

    Returns:
        DocumentStatsResult: Structured statistics and warnings.
    """

    warnings: List[str] = []

    try:
        # Use storage abstraction for basic aggregation
        vector_store = get_vector_index_store()
        raw_stats = vector_store.aggregate_document_stats(
            collection_name=collection,
            doc_id=doc_id,
            user_id=user_id,
            is_admin=is_admin,
        )

        document_count = raw_stats["documents"]
        document_exists = document_count > 0
        parse_count = raw_stats["parses"]
        chunk_count = raw_stats["chunks"]

        # Handle model_tag specific embeddings filtering
        embedding_breakdown: Dict[str, int] = {}

        if model_tag:
            # When model_tag is specified, only count embeddings for that specific table
            safe_collection = escape_lancedb_string(collection)
            safe_doc_id = escape_lancedb_string(doc_id)
            filters = {"collection": safe_collection, "doc_id": safe_doc_id}
            table_name = embeddings_table_name(model_tag)
            embedding_count = vector_store.count_rows(
                table_name=table_name,
                filters=filters,
                user_id=user_id,
                is_admin=is_admin,
            )
            embedding_breakdown[table_name] = embedding_count
        else:
            # Use the aggregated count from storage abstraction
            embedding_count = raw_stats["embeddings"]
            # Optionally include breakdown by table if needed
            safe_collection = escape_lancedb_string(collection)
            safe_doc_id = escape_lancedb_string(doc_id)
            filters = {"collection": safe_collection, "doc_id": safe_doc_id}

            try:
                table_names = vector_store.list_table_names()
            except Exception as exc:  # noqa: BLE001 - convert to warning
                message = f"Unable to enumerate embeddings tables: {exc}"
                logger.warning(message)
                warnings.append(message)
                table_names = []

            for table_name in table_names:
                if not table_name.startswith("embeddings_"):
                    continue
                count = vector_store.count_rows(
                    table_name=table_name,
                    filters=filters,
                    user_id=user_id,
                    is_admin=is_admin,
                )
                if count:
                    embedding_breakdown[table_name] = count

    except Exception as exc:  # noqa: BLE001 - convert to structured failure
        logger.error("Failed to get document stats: %s", exc, exc_info=True)
        return DocumentStatsResult(
            status="error",
            data=None,
            message=f"Failed to get document stats: {exc}",
            warnings=warnings,
        )

    # Load ingestion status
    status_record = None
    status_entries = load_ingestion_status(collection=collection, doc_id=doc_id)
    if status_entries:
        status_record = status_entries[-1]

    status_value = DocumentProcessingStatus.PENDING
    last_message = None
    updated_at_ts = None
    if status_record:
        raw_status = status_record.get("status")
        if isinstance(raw_status, str):
            try:
                status_value = DocumentProcessingStatus(raw_status)
            except ValueError:
                logger.warning(
                    "Unknown ingestion status '%s' for %s/%s",
                    raw_status,
                    collection,
                    doc_id,
                )
        last_message = status_record.get("message") or None
        updated_at_ts = _coerce_timestamp(status_record.get("updated_at"))
    else:
        # If no status record exists, infer status from chunk and embedding counts
        if chunk_count > 0:
            if embedding_count == 0:
                status_value = DocumentProcessingStatus.CHUNKED
            elif embedding_count < chunk_count:
                status_value = DocumentProcessingStatus.PARTIALLY_EMBEDDED
            else:  # embedding_count >= chunk_count
                status_value = DocumentProcessingStatus.SUCCESS
        else:
            status_value = DocumentProcessingStatus.PENDING

    stats = DocumentStats(
        collection=collection,
        doc_id=doc_id,
        document_exists=document_exists,
        parse_count=parse_count,
        chunk_count=chunk_count,
        embedding_count=embedding_count,
        embedding_breakdown=embedding_breakdown,
        status=status_value,
        last_message=last_message,
        updated_at=updated_at_ts,
    )

    if document_exists:
        message = "Document statistics collected successfully"
    else:
        message = (
            f"Document '{doc_id}' not found in collection '{collection}'."
            " Counts reflect zero state."
        )

    return DocumentStatsResult(
        status="success",
        data=stats,
        message=message,
        warnings=warnings,
    )


def list_documents(
    collection: str,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> DocumentListResult:
    """List documents for a collection including latest processing status.

    Uses batch processing to minimize memory footprint.

    Args:
        collection: Target collection name.
        user_id: Optional user ID for filtering (for multi-tenancy).
        is_admin: Whether user has admin privileges.

    Returns:
        DocumentListResult: List of documents with status information.
    """
    warnings: List[str] = []

    try:
        # Use storage abstraction for document records
        vector_store = get_vector_index_store()
        doc_records = vector_store.list_document_records(
            collection_name=collection,
            user_id=user_id,
            is_admin=is_admin,
            max_results=DEFAULT_VECTOR_STORE_EXTENDED_SCAN_LIMIT,  # Higher limit for listing
        )

        # Collect document info from records
        document_info: Dict[str, Dict[str, Any]] = {}
        for record in doc_records:
            document_info[record.doc_id] = {
                "source_path": record.source_path,
                "uploaded_at": None,  # Not available in DocumentRecord
            }

    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to list documents: %s", exc, exc_info=True)
        return DocumentListResult(
            status="error",
            documents=[],
            total_count=0,
            message=f"Failed to list documents: {exc}",
            warnings=warnings,
        )

    # Collect chunk counts using storage abstraction
    chunk_counts = vector_store.aggregate_document_counts(
        table_name="chunks",
        doc_id_column="doc_id",
        collection_name=collection,
        user_id=user_id,
        is_admin=is_admin,
    )

    # Collect embedding counts
    embedding_counts: Dict[str, int] = defaultdict(int)
    for table_name in vector_store.list_table_names():
        if not table_name.startswith("embeddings_"):
            continue
        table_counts = vector_store.aggregate_document_counts(
            table_name=table_name,
            doc_id_column="doc_id",
            collection_name=collection,
            user_id=user_id,
            is_admin=is_admin,
        )
        for doc_id, value in table_counts.items():
            embedding_counts[doc_id] += value

    # Load status records
    status_records = {
        entry["doc_id"]: entry for entry in load_ingestion_status(collection=collection)
    }

    # Combine all doc_ids from various sources
    doc_ids = (
        set(document_info.keys())
        | set(chunk_counts.keys())
        | set(embedding_counts.keys())
        | set(status_records.keys())
    )

    # Build summaries
    summaries: List[DocumentSummary] = []
    for doc_id in sorted(doc_ids):
        info = document_info.get(doc_id, {})
        status_entry = status_records.get(doc_id)
        chunk_count = chunk_counts.get(doc_id, 0)
        embedding_count = embedding_counts.get(doc_id, 0)

        if status_entry and isinstance(status_entry.get("status"), str):
            try:
                status_value = DocumentProcessingStatus(status_entry["status"])
            except ValueError:
                status_value = DocumentProcessingStatus.PENDING
        elif chunk_count > 0:
            if embedding_count == 0:
                status_value = DocumentProcessingStatus.CHUNKED
            elif embedding_count < chunk_count:
                status_value = DocumentProcessingStatus.PARTIALLY_EMBEDDED
            else:  # embedding_count >= chunk_count
                status_value = DocumentProcessingStatus.SUCCESS
        else:
            status_value = DocumentProcessingStatus.PENDING

        updated_at = _coerce_timestamp(
            status_entry.get("updated_at") if status_entry else None
        )
        message = status_entry.get("message") if status_entry else None

        summaries.append(
            DocumentSummary(
                collection=collection,
                doc_id=doc_id,
                source_path=info.get("source_path"),
                status=status_value,
                message=message or None,
                created_at=_coerce_timestamp(info.get("uploaded_at")),
                updated_at=updated_at or _coerce_timestamp(info.get("uploaded_at")),
                chunk_count=chunk_count,
                embedding_count=embedding_count,
            )
        )

    message = f"Retrieved {len(summaries)} documents for collection '{collection}'."
    return DocumentListResult(
        status="success",
        documents=summaries,
        total_count=len(summaries),
        message=message,
        warnings=warnings,
    )


def delete_collection(
    collection: str,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> CollectionOperationResult:
    """Delete all documents and associated artifacts for a collection.

    This function will delete:
    - All documents in the collection
    - All parsing data for the collection
    - All chunks for the collection
    - All embeddings for the collection

    Args:
        collection: Name of the collection to delete
        user_id: Optional user ID for permission check (for multi-tenancy).
        is_admin: Whether user has admin privileges.

    Returns:
        CollectionOperationResult indicating success or failure
    """

    warnings: List[str] = []

    try:
        # Use storage abstraction for deletion
        vector_store = get_vector_index_store()

        # Collect doc_ids before deletion for affected_documents
        # Use list_document_records which respects user filtering
        doc_records = vector_store.list_document_records(
            collection_name=collection,
            user_id=user_id,
            is_admin=is_admin,
            max_results=DEFAULT_VECTOR_STORE_EXTENDED_SCAN_LIMIT,  # Higher limit for collection deletion
        )
        doc_ids = sorted({r.doc_id for r in doc_records})

        # Delete all data using storage abstraction
        deleted_counts = vector_store.delete_collection_data(collection_name=collection)

        # Clear ingestion status for all documents
        for doc_id in doc_ids:
            try:
                clear_ingestion_status(collection, doc_id)
            except Exception as exc:  # noqa: BLE001
                warning = f"Failed to clear ingestion status for '{doc_id}': {exc}"
                logger.warning(warning)
                warnings.append(warning)

    except Exception as exc:  # noqa: BLE001 - convert to structured failure
        logger.error(
            "Failed to delete collection '%s': %s", collection, exc, exc_info=True
        )
        return CollectionOperationResult(
            status="error",
            collection=collection,
            message=f"Failed to delete collection: {exc}",
            warnings=warnings,
            affected_documents=[],
            deleted_counts={},
        )

    # Construct affected_documents list
    affected: List[CollectionOperationDetail] = [
        CollectionOperationDetail(
            doc_id=doc_id,
            status=DocumentProcessingStatus.FAILED,
            message="Document deleted successfully.",
        )
        for doc_id in doc_ids
    ]

    if not doc_ids and not deleted_counts:
        summary = f"No documents found in collection '{collection}'."
        return CollectionOperationResult(
            status="success",
            collection=collection,
            message=summary,
            warnings=warnings,
            affected_documents=[],
            deleted_counts={},
        )

    if affected and not warnings:
        status = "success"
    elif affected:
        status = "partial_success"
    else:
        status = "error"

    summary = f"Deleted {len(affected)} documents from collection '{collection}'."
    logger.info(
        f"Deleted collection '{collection}' - {sum(deleted_counts.values())} total rows across {len(deleted_counts)} tables"
    )
    return CollectionOperationResult(
        status=status,
        collection=collection,
        message=summary,
        warnings=warnings,
        affected_documents=affected,
        deleted_counts=dict(deleted_counts),
    )


def delete_document(
    collection: str, doc_id: str, user_id: int, is_admin: bool = False
) -> DocumentOperationResult:
    """Delete a document and all its associated data.

    This performs a cascade delete of the document's parses, chunks,
    embeddings, and related data.

    Args:
        collection: Collection name.
        doc_id: Document identifier.

    Returns:
        DocumentOperationResult: Operation result with deletion counts.
    """
    try:
        # Use cascade cleanup to delete all related data
        counts = cleanup_document_cascade(
            collection=collection,
            doc_id=doc_id,
            preview_only=False,
            confirm=True,
        )
        clear_ingestion_status(collection, doc_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to delete document %s/%s: %s", collection, doc_id, exc)
        return DocumentOperationResult(
            status="error",
            collection=collection,
            doc_id=doc_id,
            new_status=DocumentProcessingStatus.FAILED,
            message=f"Failed to delete document: {exc}",
            warnings=[],
            details={},
        )

    return DocumentOperationResult(
        status="success",
        collection=collection,
        doc_id=doc_id,
        new_status=DocumentProcessingStatus.FAILED,
        message="Document deleted successfully.",
        warnings=[],
        details=counts,
    )


def retry_document(
    collection: str, doc_id: str, user_id: int, is_admin: bool = False
) -> DocumentOperationResult:
    """Mark a document for retry by resetting its status to pending."""

    try:
        write_ingestion_status(
            collection,
            doc_id,
            status=DocumentProcessingStatus.PENDING.value,
            message="Retry requested.",
            parse_hash="",
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to mark document %s/%s for retry: %s", collection, doc_id, exc
        )
        return DocumentOperationResult(
            status="error",
            collection=collection,
            doc_id=doc_id,
            new_status=DocumentProcessingStatus.PENDING,
            message=f"Failed to mark document for retry: {exc}",
            warnings=[],
            details={},
        )
    return DocumentOperationResult(
        status="success",
        collection=collection,
        doc_id=doc_id,
        new_status=DocumentProcessingStatus.PENDING,
        message="Document marked for retry.",
        warnings=[],
        details={},
    )


def cancel_document(
    collection: str,
    doc_id: str,
    user_id: int,
    is_admin: bool = False,
    reason: Optional[str] = None,
) -> DocumentOperationResult:
    """Mark a document ingestion process as cancelled."""

    message = reason or "Cancelled by user."
    try:
        write_ingestion_status(
            collection,
            doc_id,
            status=DocumentProcessingStatus.FAILED.value,
            message=message,
            parse_hash="",
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to cancel document %s/%s: %s", collection, doc_id, exc)
        return DocumentOperationResult(
            status="error",
            collection=collection,
            doc_id=doc_id,
            new_status=DocumentProcessingStatus.FAILED,
            message=f"Failed to cancel document: {exc}",
            warnings=[],
            details={},
        )
    return DocumentOperationResult(
        status="success",
        collection=collection,
        doc_id=doc_id,
        new_status=DocumentProcessingStatus.FAILED,
        message="Document ingestion cancelled.",
        warnings=[],
        details={},
    )


def cancel_collection(
    collection: str,
    reason: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> CollectionOperationResult:
    """Mark all documents in a collection as cancelled."""

    warnings: List[str] = []

    try:
        # Use storage abstraction to get document IDs
        vector_store = get_vector_index_store()
        doc_records = vector_store.list_document_records(
            collection_name=collection,
            user_id=user_id,
            is_admin=is_admin,
            max_results=DEFAULT_VECTOR_STORE_EXTENDED_SCAN_LIMIT,  # Higher limit for collection operations
        )
        doc_ids = sorted({r.doc_id for r in doc_records})

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to get document IDs for cancel_collection: %s",
            exc,
            exc_info=True,
        )
        return CollectionOperationResult(
            status="error",
            collection=collection,
            message=f"Failed to get document IDs: {exc}",
            warnings=warnings,
            affected_documents=[],
            deleted_counts={},
        )

    cancellation_message = reason or "Cancelled by user."
    affected: List[CollectionOperationDetail] = []

    for doc_id in doc_ids:
        try:
            write_ingestion_status(
                collection=collection,
                doc_id=doc_id,
                status=DocumentProcessingStatus.FAILED.value,
                message=cancellation_message,
                parse_hash="",
            )
            affected.append(
                CollectionOperationDetail(
                    doc_id=doc_id,
                    status=DocumentProcessingStatus.FAILED,
                    message=cancellation_message,
                )
            )
        except Exception as exc:  # noqa: BLE001
            warning = f"Failed to cancel document '{doc_id}': {exc}"
            logger.error(warning, exc_info=True)
            warnings.append(warning)

    if not doc_ids:
        summary = f"No documents found in collection '{collection}'."
        return CollectionOperationResult(
            status="success",
            collection=collection,
            message=summary,
            warnings=warnings,
            affected_documents=[],
            deleted_counts={},
        )

    if affected and not warnings:
        status = "success"
    elif affected:
        status = "partial_success"
    else:
        status = "error"

    summary = f"Cancelled {len(affected)} documents in collection '{collection}'."
    return CollectionOperationResult(
        status=status,
        collection=collection,
        message=summary,
        warnings=warnings,
        affected_documents=affected,
        deleted_counts={},
    )


def get_document_status(collection: str, doc_id: str) -> Dict[str, Any]:
    """Get the current ingestion status for a document.

    Args:
        collection: Collection name.
        doc_id: Document identifier.

    Returns:
        Dictionary with status information, or empty dict if not found.
    """
    try:
        status_records = load_ingestion_status(collection=collection, doc_id=doc_id)
        if status_records:
            return status_records[0]
        return {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to get document status: %s", exc)
        return {}
