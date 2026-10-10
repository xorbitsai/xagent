"""Collection-scoped KB backend handle.

``KBCollectionHandle`` is the collection-scoped backend boundary that owns
backend-specific data-plane mechanics (the document-row lifecycle in #508 and
the parse/chunk lifecycle in #509). ``LanceDBCollectionHandle`` is the first
implementation and delegates to the current LanceDB tables via the bound
vector index store.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import inspect
import json
import logging
import numbers
import os
import re
import time
import uuid
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import timezone
from functools import cached_property, partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, NoReturn, Optional, TypeVar, cast

from ..storage.file_reference import guard_document_restore

if TYPE_CHECKING:
    from ......providers.vector_store.milvus import MilvusConnectionManager
    from .models import KBVectorStorageCleanupResult

import pandas as pd

from ..core.config import (
    DEFAULT_LANCEDB_BATCH_SIZE,
    DEFAULT_VECTOR_STORE_DELETE_BATCH_SIZE,
)
from ..core.exceptions import (
    ConfigurationError,
    DatabaseOperationError,
    DocumentValidationError,
    HashComputationError,
    MainPointerError,
    VectorValidationError,
)
from ..core.schemas import (
    ChunkEmbeddingData,
    ChunkForEmbedding,
    ChunkRecordSnapshot,
    DenseSearchResponse,
    DocumentProcessingStatus,
    DocumentRecordDetail,
    DocumentRecordListResult,
    EmbeddingReadResponse,
    EmbeddingRecordSnapshot,
    EmbeddingWriteResponse,
    FusionConfig,
    FusionStrategy,
    HybridSearchResponse,
    IndexOperation,
    IndexStatus,
    ParsedParagraph,
    ParseRecordDetail,
    RegisterDocumentRequest,
    RegisterDocumentResponse,
    SearchFallbackAction,
    SearchResult,
    SearchWarning,
    SparseSearchResponse,
    StepType,
)
from ..LanceDB.model_tag_utils import embeddings_table_name, to_model_tag
from ..LanceDB.schema_manager import (
    _safe_close_table,
    ensure_chunks_table,
    ensure_documents_table,
    ensure_ingestion_runs_table,
    ensure_main_pointers_table,
    ensure_parses_table,
)
from ..retrieval.search_hybrid import _linear_fusion, _rrf_fusion
from ..storage.contracts import (
    FilterCondition,
    FilterExpression,
    FilterOperator,
    IngestionStatusStore,
    MainPointerStore,
    MetadataStore,
    VectorIndexStore,
    build_filter_from_dict,
)
from ..storage.vector_backend import (
    get_configured_vector_backend,
    require_implemented_vector_backend,
)
from ..utils import check_file_type, compute_file_hash
from ..utils.filter_utils import parse_legacy_filters, validate_filter_depth
from ..utils.hash_utils import compute_chunk_hash
from ..utils.lancedb_query_utils import (
    _safe_count_rows,
    build_fts_query,
    list_table_names,
    query_to_list,
)
from ..utils.metadata_utils import deserialize_metadata, serialize_metadata
from ..utils.string_utils import escape_lancedb_string, generate_deterministic_doc_id
from .kb_ids import (
    delete_kb_ids,
    get_or_create_kb_id,
    read_kb_ids,
    rename_kb_ids,
    set_kb_ids_collection,
)
from .milvus_search import (
    SEARCH_FIELDS,
    caller_filter,
    dense_score,
    keyword_score,
    like_pattern,
    to_result,
)
from .models import (
    KBBackendCapabilities,
    KBCollectionContext,
    KBDocumentRowsSnapshot,
    KBStorageBackend,
)
from .storage_shim import KBStorageShimCompatibilityFacade
from .version_compatibility import (
    KBMainPointerSnapshot,
    KBVersionCandidateCleanupSnapshot,
    KBVersionCandidateRollbackResult,
)

logger = logging.getLogger(__name__)


def _safe_int_value(value: Any, default: int = 0) -> int:
    """Coerce a row value to ``int``, mapping ``None``/NaN to ``default``."""
    if value is None:
        return default
    try:
        if value != value:  # NaN is never equal to itself.  # noqa: PLR0124
            return default
    except Exception:  # noqa: BLE001 - non-comparable values fall through
        pass
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _int_env(name: str, default: int) -> int:
    """Read an integer environment variable, falling back to ``default``.

    A missing variable, or one set to a non-numeric value, yields ``default``
    instead of raising, so a malformed operator override cannot crash an
    embedding write.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (ValueError, TypeError):
        return default


def _safe_optional_str(value: Any) -> str | None:
    """Return the string value or ``None`` for ``None``/NaN sentinels."""
    if value is None:
        return None
    try:
        if value != value:  # NaN  # noqa: PLR0124
            return None
    except Exception:  # noqa: BLE001
        pass
    return str(value)


def _chunk_for_embedding(chunk_dict: dict[str, Any]) -> ChunkForEmbedding:
    """Build the embedding input of one ledger ``chunks`` row."""
    page_number_value = chunk_dict.get("page_number")
    # A missing page number can arrive as None or as a pandas/LanceDB
    # NaN sentinel (NaN != NaN); both mean "no page", not page 1.
    if page_number_value is not None and page_number_value == page_number_value:
        page_num = _safe_int_value(page_number_value, default=1)
        page_number = page_num if page_num > 0 else None
    else:
        page_number = None

    return ChunkForEmbedding(
        doc_id=chunk_dict["doc_id"],
        chunk_id=chunk_dict["chunk_id"],
        parse_hash=chunk_dict["parse_hash"],
        index=_safe_int_value(chunk_dict.get("index"), default=0),
        text=chunk_dict["text"],
        chunk_hash=chunk_dict["chunk_hash"],
        page_number=page_number,
        section=_safe_optional_str(chunk_dict.get("section")),
        anchor=_safe_optional_str(chunk_dict.get("anchor")),
        json_path=_safe_optional_str(chunk_dict.get("json_path")),
        metadata=deserialize_metadata(chunk_dict.get("metadata")),
    )


def validate_query_vector_format(query_vector: list[float]) -> None:
    """Validate a query vector's format and content (collection-independent).

    Pure check shared by the collection handle and the vector-storage facade
    (the facade's validate path has no collection to bind a handle). Raises
    ``VectorValidationError`` for non-list, empty, non-numeric, or NaN/inf
    vectors; numpy scalar types are admitted via ``numbers.Number``.
    """
    if not isinstance(query_vector, list):
        raise VectorValidationError("query_vector must be a list")

    if len(query_vector) == 0:
        raise VectorValidationError("query_vector cannot be empty")

    if not all(isinstance(x, numbers.Number) for x in query_vector):
        raise VectorValidationError("query_vector must contain only numbers")

    for x in query_vector:
        if not isinstance(x, numbers.Real):
            continue  # Skip non-real numbers (e.g. complex).
        float_val = float(x)
        if float_val != float_val or abs(float_val) == float("inf"):
            raise VectorValidationError(
                "query_vector contains invalid values (NaN or infinity)"
            )


_DOCUMENT_ROW_KEYS: dict[str, tuple[str, ...]] = {
    "documents": ("collection", "doc_id"),
    "parses": ("collection", "doc_id", "parse_hash"),
    "chunks": ("collection", "doc_id", "parse_hash", "chunk_id"),
    "main_pointers": ("collection", "doc_id", "step_type", "model_tag"),
    "ingestion_runs": ("collection", "doc_id"),
}
_EMBEDDING_ROW_KEY = ("collection", "doc_id", "chunk_id", "parse_hash", "model")


def _document_row_key_columns(table_name: str) -> tuple[str, ...] | None:
    if table_name.startswith("embeddings_"):
        return _EMBEDDING_ROW_KEY
    return _DOCUMENT_ROW_KEYS.get(table_name)


def _lancedb_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return f"'{escape_lancedb_string(str(value))}'"


def _any_of(filters: list[str]) -> str:
    return " or ".join(f"({filter_expr})" for filter_expr in filters)


def _owned_row_filter(
    table: Any, clauses: list[str], *, user_id: int, is_admin: bool
) -> str:
    if not is_admin and "user_id" in (
        getattr(getattr(table, "schema", None), "names", None) or []
    ):
        clauses = [*clauses, f"user_id = {int(user_id)}"]
    return " and ".join(clauses)


def _row_key_filter(
    table: Any,
    row: dict[str, Any],
    key_columns: tuple[str, ...],
    *,
    user_id: int,
    is_admin: bool,
) -> str:
    clauses = [
        f"{column} IS NULL"
        if row.get(column) is None
        else f"{column} = {_lancedb_literal(row.get(column))}"
        for column in key_columns
    ]
    return _owned_row_filter(table, clauses, user_id=user_id, is_admin=is_admin)


def _restore_document_table_rows(
    table: Any,
    *,
    table_name: str,
    snapshot_rows: list[dict[str, Any]],
    current_rows: list[dict[str, Any]],
    user_id: int,
    is_admin: bool,
) -> None:
    """Upsert old rows before deleting stale rows introduced by a failed refresh."""
    key_columns = _document_row_key_columns(table_name)
    if key_columns is None:
        delete_filters = [
            _row_key_filter(
                table, row, ("collection", "doc_id"), user_id=user_id, is_admin=is_admin
            )
            for row in current_rows
        ]
        if delete_filters:
            table.delete(_any_of(delete_filters))
        if snapshot_rows:
            table.add(snapshot_rows)
        return

    if snapshot_rows:
        (
            table.merge_insert(list(key_columns))
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(snapshot_rows)
        )

    def row_key(row: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(row.get(column) for column in key_columns)

    snapshot_keys = {row_key(row) for row in snapshot_rows}
    delete_filters = [
        _row_key_filter(table, row, key_columns, user_id=user_id, is_admin=is_admin)
        for row in current_rows
        if row_key(row) not in snapshot_keys
    ]
    if delete_filters:
        table.delete(_any_of(delete_filters))


def deployment_kb_backend() -> KBStorageBackend:
    """Return the KB engine of this deployment.

    Startup refuses a setting that differs from the recorded or detected engine;
    where neither can be determined (the LanceDB directory cannot be reached or
    listed), it only warns, and the setting is used as it is.
    Raises ``ConfigurationError`` for a known engine that is not implemented.
    """
    backend = get_configured_vector_backend()
    require_implemented_vector_backend(backend)
    return backend


def ledger_holds_vectors() -> bool:
    """Whether the LanceDB ledger also holds chunk vectors (``embeddings_*`` tables).

    True only in a LanceDB deployment; upper-layer code that reads those tables
    directly runs only when this holds.
    """
    return deployment_kb_backend() is KBStorageBackend.LANCEDB


def _fuse_hybrid(
    model_tag: str,
    query_text: str,
    dense_response: DenseSearchResponse,
    sparse_response: SparseSearchResponse,
    *,
    top_k: int,
    fusion_config: FusionConfig,
) -> HybridSearchResponse:
    """Fuse already-fetched dense/sparse responses into a hybrid response.

    Holds every step that runs after the dense/sparse calls in both the sync
    and async hybrid paths. It consumes already-fetched response objects, so
    it is purely synchronous and shared by ``search_hybrid`` and
    ``search_hybrid_async``.
    """
    all_warnings: List[SearchWarning] = []

    dense_results = dense_response.results
    all_warnings.extend(dense_response.warnings)

    sparse_results = sparse_response.results
    all_warnings.extend(sparse_response.warnings)

    # Get index status and advice from dense search (primary source for index info)
    index_status = dense_response.index_status
    index_advice = dense_response.index_advice

    # 3. Preserve original scores and ranks before fusion
    dense_rank_map: Dict[str, int] = {}
    sparse_rank_map: Dict[str, int] = {}
    dense_score_map: Dict[str, float] = {}
    sparse_score_map: Dict[str, float] = {}

    for rank, result in enumerate(dense_results, start=1):
        unique_id = (
            f"{result.doc_id}-{result.chunk_id}-{result.parse_hash}-{result.model_tag}"
        )
        dense_rank_map[unique_id] = rank
        dense_score_map[unique_id] = result.score

    for rank, result in enumerate(sparse_results, start=1):
        unique_id = (
            f"{result.doc_id}-{result.chunk_id}-{result.parse_hash}-{result.model_tag}"
        )
        sparse_rank_map[unique_id] = rank
        sparse_score_map[unique_id] = result.score

    # 4. Fuse Results
    logger.info("Fusing results using strategy: %s", fusion_config.strategy.value)
    fused_results: List[SearchResult] = []
    if fusion_config.strategy == FusionStrategy.RRF:
        fused_results = _rrf_fusion(
            [dense_results, sparse_results], k=fusion_config.rrf_k
        )
    elif fusion_config.strategy == FusionStrategy.LINEAR:
        fused_results = _linear_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            dense_weight=fusion_config.dense_weight,
            sparse_weight=fusion_config.sparse_weight,
            normalize_scores=fusion_config.normalize_scores,
        )
    else:
        logger.warning(
            "Unknown fusion strategy: %s. Defaulting to dense results.",
            fusion_config.strategy,
        )
        fused_results = dense_results

    # 5. Attach original scores and ranks to fused results
    updated_fused_results: List[SearchResult] = []
    for result in fused_results:
        unique_id = (
            f"{result.doc_id}-{result.chunk_id}-{result.parse_hash}-{result.model_tag}"
        )
        updated_fused_results.append(
            result.model_copy(
                update={
                    "vector_score": dense_score_map.get(unique_id),
                    "fts_score": sparse_score_map.get(unique_id),
                    "vector_rank": dense_rank_map.get(unique_id),
                    "fts_rank": sparse_rank_map.get(unique_id),
                }
            )
        )
    fused_results = updated_fused_results

    # Limit to top_k after fusion
    final_results = fused_results[:top_k]

    # 6. Build Response
    return HybridSearchResponse(
        results=final_results,
        total_count=len(final_results),
        status="success" if not all_warnings else "partial_success",
        warnings=all_warnings,
        fusion_config=fusion_config,
        dense_count=len(dense_results),
        sparse_count=len(sparse_results),
        index_status=index_status,
        index_advice=index_advice,
    )


class KBHandleProvider:
    """Open collection-scoped handles for resolved KB contexts."""

    def __init__(
        self,
        storage_shim: KBStorageShimCompatibilityFacade | None = None,
        connections: MilvusConnectionManager | None = None,
    ) -> None:
        self._storage_shim = storage_shim or KBStorageShimCompatibilityFacade()
        self._connections = connections

    def _milvus_connections(self) -> MilvusConnectionManager:
        from ......providers.vector_store.milvus import MilvusConnectionManager

        return self._connections or MilvusConnectionManager()

    def open(self, context: KBCollectionContext) -> KBCollectionHandle:
        """Return a backend-specific handle for the resolved collection context."""
        if context.backend is KBStorageBackend.LANCEDB:
            return LanceDBCollectionHandle(context)
        if context.backend is KBStorageBackend.MILVUS:
            deployment = deployment_kb_backend()
            if deployment is not KBStorageBackend.MILVUS:
                raise ValueError(
                    f"Collection {context.collection!r} is bound to the milvus "
                    f"engine, but this deployment runs {deployment.value}"
                )
            return MilvusCollectionHandle(
                context,
                ledger=LanceDBCollectionHandle(context),
                connections=self._milvus_connections(),
            )
        raise ValueError(
            f"KB storage backend {context.backend.value!r} is not supported by "
            "KBHandleProvider"
        )

    def aggregate_collection_stats(
        self, *, user_id: int | None, is_admin: bool
    ) -> dict[str, dict[str, int]]:
        """Return per-collection stats for every collection the caller can see.

        One batched ledger scan rather than one handle per collection; a Milvus
        deployment adds, per collection with a kb_id, one count query in each
        Milvus collection.
        """
        backend = deployment_kb_backend()
        if backend is KBStorageBackend.LANCEDB:
            store = self._storage_shim.get_vector_index_store()
            return store.aggregate_collection_stats(user_id=user_id, is_admin=is_admin)
        if backend is KBStorageBackend.MILVUS:
            store = self._storage_shim.get_vector_index_store()
            stats = store.aggregate_collection_stats(user_id=user_id, is_admin=is_admin)
            kb_ids = read_kb_ids(
                store.get_raw_connection(), user_id=user_id, is_admin=is_admin
            )
            client = (
                self._milvus_connections().get_shared_client_from_env()
                if kb_ids
                else None
            )
            names = _milvus_collections(client) if kb_ids else []
            not_loaded: set[str] = set()
            for collection, row in stats.items():
                owned = kb_ids.get(collection)
                row["embeddings"] = (
                    sum(_count_visible_rows(client, names, owned, not_loaded).values())
                    if owned
                    else 0
                )
            return stats
        raise ValueError(
            f"KB storage backend {backend.value!r} is not supported by KBHandleProvider"
        )

    def reset_for_tests(self) -> None:
        """Clear provider-owned caches for test reset.

        The provider holds the storage shim but keeps no caches of its own;
        the hook keeps the coordinator reset path ready for backend handle
        caches.
        """


class KBCollectionHandle(ABC):
    """Collection-scoped backend handle for KB data-plane operations.

    Phase 2 moves backend-specific, collection-local data-plane mechanics here.
    The first family (#508) is the document-row lifecycle. The coordinator owns
    context resolution, access policy, and orchestration; the handle owns the
    backend mechanics for a single collection.
    """

    @abstractmethod
    def register_document(
        self, request: RegisterDocumentRequest
    ) -> RegisterDocumentResponse:
        """Idempotently register (upsert) a document row for this collection.

        Preserves deterministic doc_id generation, content-hash calculation,
        file-type detection, and the exact persisted field set.
        """

    @abstractmethod
    def load_document(
        self, doc_id: str, *, user_id: int | None = None, is_admin: bool = False
    ) -> DocumentRecordDetail | None:
        """Load a single document row by id within the given scope.

        Returns ``None`` when the row is absent or not visible to the scope.
        """

    @abstractmethod
    def list_documents(
        self, *, user_id: int | None = None, is_admin: bool = False, limit: int = 100
    ) -> DocumentRecordListResult:
        """List document rows for this collection as a semantic result."""

    @abstractmethod
    def delete_document_record(
        self, doc_id: str, *, user_id: int | None = None, is_admin: bool = False
    ) -> int:
        """Delete only this document's row (no cascade).

        Idempotent; returns the number of rows deleted. Parse/chunk/embedding
        cleanup is intentionally out of scope here.
        """

    # --- Rollback compensation (document plane only) ---

    @abstractmethod
    def snapshot_document(
        self, doc_id: str, *, user_id: int | None = None, is_admin: bool = False
    ) -> DocumentRecordDetail | None:
        """Capture the current document row for later restore (None if absent)."""

    @abstractmethod
    def restore_document(self, snapshot: DocumentRecordDetail) -> None:
        """Restore a previously snapshotted document row, preserving all fields."""

    @abstractmethod
    def delete_created_document(
        self, doc_id: str, *, user_id: int | None = None, is_admin: bool = False
    ) -> int:
        """Idempotently delete a newly created document row (compensation)."""

    @abstractmethod
    def capture_document_rows(
        self, doc_ids: Sequence[str], *, user_id: int, is_admin: bool
    ) -> KBDocumentRowsSnapshot:
        """Capture these documents' rows across the document and embedding tables.

        Non-admin reads are limited to ``user_id`` on tables with that column.
        """

    @abstractmethod
    def restore_document_rows(
        self, snapshot: KBDocumentRowsSnapshot, *, user_id: int, is_admin: bool
    ) -> list[str]:
        """Upsert the snapshot's rows and delete its documents' rows it lacks.

        Reads and deletes are limited to ``user_id`` on tables with that column;
        upserts match by row key only. Returns the doc_ids whose vectors no longer
        match the restored chunks and were marked partially embedded.
        """

    # --- Parse data-plane (#509) ---

    @abstractmethod
    def parse_exists(
        self,
        doc_id: str,
        parse_hash: str,
        *,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> bool:
        """Return whether a parse row exists for ``(doc_id, parse_hash)``."""

    @abstractmethod
    def read_parse_paragraphs(
        self,
        doc_id: str,
        parse_hash: str,
        *,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> list[ParsedParagraph]:
        """Return the reuse-hit parsed paragraphs for ``(doc_id, parse_hash)``.

        Empty list when no visible parse row exists.
        """

    @abstractmethod
    def write_parse(
        self,
        doc_id: str,
        parse_hash: str,
        parse_method: Any,
        params: dict[str, Any],
        paragraphs: list[ParsedParagraph],
        *,
        user_id: int | None = None,
    ) -> bool:
        """Persist a parse row for this collection (idempotent upsert)."""

    @abstractmethod
    def read_latest_parse_record(
        self,
        doc_id: str,
        parse_hash: str | None = None,
        *,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> ParseRecordDetail | None:
        """Return the latest parse row (by ``created_at``) for display.

        When ``parse_hash`` is given only that version is considered. Returns
        ``None`` when no visible parse row exists; the display layer maps that
        to the appropriate ``DocumentNotFoundError``.
        """

    # --- Chunk data-plane (#509) ---

    @abstractmethod
    def chunk_exists(
        self,
        doc_id: str,
        parse_hash: str,
        config_hash: str,
        *,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> bool:
        """Return whether chunk rows exist for ``(doc_id, parse_hash, config_hash)``."""

    @abstractmethod
    def read_existing_chunks(
        self,
        doc_id: str,
        parse_hash: str,
        config_hash: str,
        *,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> list[dict[str, Any]]:
        """Return the reuse-hit chunk dicts (metadata deserialized).

        Empty list when no visible chunk rows exist.
        """

    @abstractmethod
    def read_parse_paragraph_dicts(
        self,
        doc_id: str,
        parse_hash: str,
        *,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> list[dict[str, Any]]:
        """Return parsed paragraphs as ``{text, metadata}`` dicts for chunking."""

    @abstractmethod
    def write_chunks(
        self,
        doc_id: str,
        parse_hash: str,
        config_hash: str,
        params: dict[str, Any],
        chunks: list[dict[str, Any]],
        *,
        user_id: int | None = None,
    ) -> bool:
        """Persist chunk rows for this collection (idempotent upsert).

        Returns ``False`` when there are no chunks to write.
        """

    # --- Embedding data-plane (#510) ---

    @abstractmethod
    def validate_query_vector(
        self,
        query_vector: list[float],
        *,
        model_tag: str | None = None,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> None:
        """Validate a query vector's format/content (no store access).

        Raises ``VectorValidationError`` for non-list, empty, non-numeric, or
        NaN/inf vectors. ``model_tag``/``user_id``/``is_admin`` are accepted for
        signature parity and logging only.
        """

    @abstractmethod
    def read_chunks_needing_embedding(
        self,
        doc_id: str,
        parse_hash: str,
        model: str,
        *,
        filters: dict[str, Any] | None = None,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> EmbeddingReadResponse:
        """Return chunks that still need an embedding for ``model``.

        Reads the chunks table for ``(doc_id, parse_hash)`` and excludes
        ``chunk_id``s already present in the ``embeddings_{model_tag}`` table.
        """

    @abstractmethod
    def write_embeddings(
        self,
        embeddings: list[ChunkEmbeddingData],
        *,
        create_index: bool = True,
        user_id: int | None = None,
    ) -> EmbeddingWriteResponse:
        """Write embedding vectors for this collection (idempotent upsert).

        Groups by model, validates per-model dimension consistency, routes each
        model to its ``embeddings_{model_tag}`` table, upserts in batches (with
        spill-retry), and optionally creates the index. Stale deletion is a
        no-op (``deleted_stale_count`` is always 0; merge handles overwrites).
        """

    @abstractmethod
    def commit_embeddings(
        self,
        doc_id: str,
        parse_hash: str,
        model: str,
        *,
        commit_gate: Callable[[], None] | None = None,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> None:
        """Make the vectors written for ``(doc_id, parse_hash)`` searchable.

        Called after the last batch and before the ledger status is updated, and
        also when a rerun finds nothing pending (a crash after the writes), so a
        call may follow no write at all: implementations must be idempotent. An
        engine that writes rows invisible makes them visible here; one that writes
        searchable rows does nothing.
        """

    @abstractmethod
    def discard_uncommitted_embeddings(
        self, doc_id: str, *, user_id: int | None = None
    ) -> int:
        """Delete the rows ``write_embeddings`` left invisible for ``doc_id``.

        Called when an ingest of an existing document failed. Idempotent; returns
        the number of rows deleted, 0 for an engine that writes searchable rows.
        """

    @abstractmethod
    def delete_embedding_records(
        self,
        doc_id: str,
        *,
        parse_hash: str | None = None,
        chunk_ids: list[str] | None = None,
        model_tag: str | None = None,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> int:
        """Delete embedding rows for a document across per-model tables.

        Row-only (no cascade); ``model_tag`` narrows to one model's table,
        ``None`` spans all. Idempotent; returns the total rows deleted.
        """

    # --- Embedding rollback compensation (methods only; wiring in #514) ---

    @abstractmethod
    def snapshot_embeddings(
        self,
        doc_id: str,
        parse_hash: str,
        *,
        chunk_ids: list[str] | None = None,
        model_tag: str | None = None,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> EmbeddingRecordSnapshot | None:
        """Capture embedding rows across matching model tables (None if absent)."""

    @abstractmethod
    def restore_embeddings(self, snapshot: EmbeddingRecordSnapshot) -> None:
        """Restore snapshotted embedding rows, grouped per model tag.

        Refuses rows from another collection (collection-guard); idempotent.
        """

    @abstractmethod
    def delete_created_embeddings(
        self,
        doc_id: str,
        parse_hash: str,
        *,
        chunk_ids: list[str] | None = None,
        model_tag: str | None = None,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> int:
        """Idempotently delete newly created embedding rows (compensation)."""

    # --- Search data-plane (#511) ---

    @abstractmethod
    def search_dense(
        self,
        model_tag: str,
        query_vector: list[float],
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        readonly: bool = False,
        nprobes: int | None = None,
        refine_factor: int | None = None,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> DenseSearchResponse:
        """Execute dense vector search for this collection.

        ``model_tag`` is the embedding model id; Milvus resolves its collection from it.
        Milvus raises, not a failed response, on a bad filter or unreadable kb_ids.
        """

    @abstractmethod
    async def search_dense_async(
        self,
        model_tag: str,
        query_vector: list[float],
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        readonly: bool = False,
        nprobes: int | None = None,
        refine_factor: int | None = None,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> DenseSearchResponse:
        """Async dense vector search for this collection.

        Milvus does not support async search.
        """

    @abstractmethod
    def search_sparse(
        self,
        model_tag: str,
        query_text: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
        readonly: bool = False,
        nprobes: int | None = None,
        refine_factor: int | None = None,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> SparseSearchResponse:
        """Execute sparse (FTS) search for this collection.

        ``model_tag`` is the embedding model id; Milvus resolves its collection from it.
        Milvus raises, not a failed response, on a bad filter or unreadable kb_ids.
        """

    @abstractmethod
    async def search_sparse_async(
        self,
        model_tag: str,
        query_text: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
        readonly: bool = False,
        nprobes: int | None = None,
        refine_factor: int | None = None,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> SparseSearchResponse:
        """Async sparse (FTS) search for this collection.

        Milvus does not support async search.
        """

    @abstractmethod
    def search_hybrid(
        self,
        model_tag: str,
        query_text: str,
        query_vector: list[float],
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        fusion_config: FusionConfig | None = None,
        readonly: bool = False,
        nprobes: int | None = None,
        refine_factor: int | None = None,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> HybridSearchResponse:
        """Execute hybrid (dense + sparse) search with fusion for this collection.

        ``model_tag`` is the embedding model id; Milvus resolves its collection from it.
        Milvus raises, not a failed response, on a bad filter or unreadable kb_ids.
        """

    @abstractmethod
    async def search_hybrid_async(
        self,
        model_tag: str,
        query_text: str,
        query_vector: list[float],
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        fusion_config: FusionConfig | None = None,
        readonly: bool = False,
        nprobes: int | None = None,
        refine_factor: int | None = None,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> HybridSearchResponse:
        """Async hybrid (dense + sparse) search with fusion for this collection.

        Milvus does not support async search.
        """

    # --- Parse/chunk cleanup (row only, collection scoped) (#509) ---

    @abstractmethod
    def delete_parse_records(
        self,
        doc_id: str,
        *,
        parse_hash: str | None = None,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> int:
        """Delete parse rows for a document (optionally one parse_hash).

        Row-only (no cascade into chunks/embeddings); idempotent.
        """

    @abstractmethod
    def delete_chunk_records(
        self,
        doc_id: str,
        *,
        parse_hash: str | None = None,
        config_hash: str | None = None,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> int:
        """Delete chunk rows for a document (optionally narrowed).

        Row-only (no cascade into embeddings); idempotent.
        """

    # --- Parse/chunk rollback compensation (methods only; wiring in #514) ---

    @abstractmethod
    def snapshot_parse(
        self,
        doc_id: str,
        parse_hash: str,
        *,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> ParseRecordDetail | None:
        """Capture a parse row for later restore (None if absent)."""

    @abstractmethod
    def restore_parse(self, snapshot: ParseRecordDetail) -> None:
        """Restore a snapshotted parse row, preserving every field."""

    @abstractmethod
    def delete_created_parse(
        self,
        doc_id: str,
        parse_hash: str,
        *,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> int:
        """Idempotently delete a newly created parse row (compensation)."""

    @abstractmethod
    def snapshot_chunks(
        self,
        doc_id: str,
        parse_hash: str,
        config_hash: str,
        *,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> ChunkRecordSnapshot | None:
        """Capture all chunk rows for a config for later restore (None if absent)."""

    @abstractmethod
    def restore_chunks(self, snapshot: ChunkRecordSnapshot) -> None:
        """Restore snapshotted chunk rows, preserving every field."""

    @abstractmethod
    def delete_created_chunks(
        self,
        doc_id: str,
        parse_hash: str,
        config_hash: str,
        *,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> int:
        """Idempotently delete newly created chunk rows (compensation)."""

    @abstractmethod
    def cleanup_cascade(
        self,
        doc_id: str,
        scope: str,
        *,
        new_parse_hash: Optional[str] = None,
        old_parse_hash: Optional[str] = None,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: Optional[bool] = None,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> Dict[str, int]:
        """Version cascade cleanup for a document by scope (policy layer)."""

    @abstractmethod
    def cleanup_document_cascade(
        self,
        doc_id: str,
        *,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = True,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> Dict[str, int]:
        """Cascade cleanup for all data of a document."""

    @abstractmethod
    def cleanup_parse_cascade(
        self,
        doc_id: str,
        *,
        old_parse_hash: Optional[str] = None,
        new_parse_hash: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = True,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> Dict[str, int]:
        """Cascade cleanup when promoting a parse version."""

    @abstractmethod
    def cleanup_chunk_cascade(
        self,
        doc_id: str,
        *,
        old_parse_hash: Optional[str] = None,
        new_parse_hash: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = True,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> Dict[str, int]:
        """Cascade cleanup when promoting a chunk version."""

    @abstractmethod
    def cleanup_embed_cascade(
        self,
        doc_id: str,
        *,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = True,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> Dict[str, int]:
        """Cascade cleanup when promoting an embeddings version."""

    # --- Collection-level rename primitives (#H05 Phase 2) ---

    @abstractmethod
    def rename_collection_data(
        self,
        new_name: str,
        user_id: int | None,
        is_admin: bool,
        warnings_out: list[str] | None = None,
    ) -> list[str]:
        """Rename the collection field across all vector-side data tables.

        Updates the ``collection`` column from ``self.context.collection`` to
        ``new_name`` in the documents, parses, chunks, and all embeddings_*
        tables.  Uses the same multi-tenancy filter semantics as other store
        writes.

        Args:
            new_name: Target collection name.
            user_id: User ID for tenant-scoped rename; ``None`` treated as 0
                for non-admin callers.
            is_admin: When ``True`` renames all matching rows regardless of
                ``user_id``.
            warnings_out: Optional list to accumulate per-table warning
                messages (best-effort updates).

        Returns:
            List of warning messages generated during best-effort updates
            (empty on full success).
        """

    @abstractmethod
    def rename_collection_status(
        self,
        new_name: str,
        user_id: int | None,
        is_admin: bool,
    ) -> list[str]:
        """Rename ingestion status rows from this collection's name to ``new_name``.

        Updates the ``collection`` column in the ``ingestion_runs`` table from
        ``self.context.collection`` to ``new_name``.

        Args:
            new_name: Target collection name.
            user_id: User ID for tenant-scoped rename.
            is_admin: When ``True`` renames all matching rows regardless of
                ``user_id``.

        Returns:
            List of warning messages on partial failure (empty on success).
        """

    @abstractmethod
    async def rename_collection_metadata(
        self,
        new_name: str,
        user_id: int | None,
        is_admin: bool,
    ) -> None:
        """Rename control-plane metadata from this collection's name to ``new_name``.

        Wraps ``await metadata_store.rename_collection(...)`` to update the
        ``collection_config`` and ``collection_metadata`` rows.

        Args:
            new_name: Target collection name.
            user_id: User ID for tenant-scoped rename.
            is_admin: When ``True`` renames across all tenants.
        """

    # --- Collection-level cascade delete (#H05) ---

    @abstractmethod
    def delete_collection_data(
        self,
        *,
        user_id: int | None,
        is_admin: bool,
        warnings_out: list[str] | None = None,
    ) -> dict[str, int]:
        """Delete all data for this collection (cascade across all vector-side tables).

        Uses ``self.context.collection`` as the collection name; no external
        ``collection_name`` argument is accepted (the handle is already scoped).

        Returns a ``dict[str, int]`` mapping table names to deleted row counts.
        Raises ``DatabaseOperationError`` on failure.
        """

    @abstractmethod
    def delete_documents_data(
        self,
        doc_ids: list[str],
        *,
        user_id: int | None,
        is_admin: bool,
        warnings_out: list[str] | None = None,
    ) -> dict[str, int]:
        """Delete vector-side data for specific document IDs in this collection.

        Batches deletes internally.  On partial failure raises
        ``DatabaseOperationError`` with ``details`` containing:
            ``{"deleted_counts": dict, "deleted_doc_ids": list, "failed_batch_index": int}``
        This exact shape is the downstream contract for
        ``CollectionOperationResult.partial_success``.

        Returns a ``dict[str, int]`` mapping table names to total deleted row
        counts across all successfully processed batches.
        """

    # --- Collection-level rollback config primitives (#H05 Phase 4) ---

    @abstractmethod
    async def delete_collection_config(self, *, tenant_only: bool = False) -> int:
        """Delete the collection_config row(s) for this collection.

        When ``tenant_only`` is ``False`` (default) all tenant rows for this
        collection are removed (admin scope – use only when the collection is
        completely empty across all tenants).  When ``tenant_only`` is ``True``
        only the row belonging to the handle's bound user scope is deleted,
        leaving other tenants' rows intact.

        Idempotent – returns the number of rows deleted (0 when no row
        existed, which is not an error).
        """

    @abstractmethod
    def cleanup_collection_data_after_rollback(
        self,
        *,
        user_id: int | None,
        is_admin: bool,
    ) -> dict[str, int]:
        """Remove all vector-side data for this collection (rollback compensation).

        Composes the Phase 1 :meth:`delete_collection_data` primitive to clean
        up a failed new-collection ingestion.  Does **not** touch the
        filesystem; physical file cleanup is the caller's responsibility.

        Returns a ``dict[str, int]`` mapping table names to deleted row counts.
        """

    @abstractmethod
    def cleanup_embeddings_for_operation(
        self,
        *,
        doc_id: Optional[str] = None,
        parse_hash: Optional[str] = None,
        chunk_ids: Optional[Sequence[str]] = None,
        model_tag: Optional[str] = None,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> "KBVectorStorageCleanupResult":
        """Delete or preview embedding rows created by a failed operation.

        The cleanup scope is bound to this handle's collection + user scope;
        rollback callers pass the operation's known document/parse/chunk/
        model-tag identity. Per-table failures never raise - they are folded
        into the result (``status="incomplete"``,
        ``side_effects_may_remain=True``). ``status="skipped"`` on an empty
        filter set; ``"planned"`` for previews; ``"complete"`` after deletes.
        """

    # --- Collection-level statistics (#H05 Phase 3) ---

    @abstractmethod
    def count_documents(self, user_id: int | None, is_admin: bool) -> int:
        """Count documents visible to the given user in this collection.

        When ``is_admin`` is ``True`` all rows are counted regardless of
        ``user_id``.  Otherwise only rows owned by ``user_id`` are counted.

        Returns:
            Number of document rows visible to the caller.
        """

    @abstractmethod
    def collection_stats(self, user_id: int | None, is_admin: bool) -> dict[str, int]:
        """Return aggregate statistics for this collection.

        Counts rows across the documents, chunks, and all embeddings_* tables
        that are visible to the caller under the given user/admin scope.

        Returns:
            A ``dict`` with at least these keys:
            - ``"documents"`` – count of document rows
            - ``"chunks"``    – count of chunk rows
            - ``"embeddings"``– total count of embedding rows across all model
              tables
        """

    @abstractmethod
    def count_rows_by_document(
        self,
        *,
        user_id: int | None,
        is_admin: bool,
        doc_id: str | None = None,
    ) -> dict[str, dict[str, int]]:
        """Count chunk and embedding rows per document in this collection.

        Maps each ``doc_id`` (only ``doc_id`` when given) to ``{"chunks": n}``
        plus one ``embeddings_<model_tag>`` entry per model that holds rows for
        it, counting only rows visible to the caller. Zero counts and documents
        without chunk or embedding rows are omitted.
        """

    @abstractmethod
    def list_collection_documents(
        self,
        user_id: int | None,
        is_admin: bool,
        max_results: int = 1_000_000,
    ) -> list[str]:
        """List document IDs visible to the given user in this collection.

        Returns a sorted list of unique doc_id strings for documents visible
        to the caller under the given user/admin scope.  Used by the coordinator
        before deletion to populate ``affected_documents`` and to collect
        tenant-owned doc_ids when the caller has not pre-computed them.

        Args:
            user_id: Owner filter; ``None`` treated as 0 for non-admin callers.
            is_admin: When ``True`` lists all documents regardless of user_id.
            max_results: Upper bound on the number of document IDs returned.

        Returns:
            Sorted list of unique doc_id strings.
        """

    # --- Ingestion-status data-plane (#513) ---

    @abstractmethod
    def write_ingestion_status(
        self,
        doc_id: str,
        *,
        status: str,
        message: str | None = None,
        parse_hash: str | None = None,
        user_id: int | None = None,
    ) -> None:
        """Write ingestion status for a document in this collection."""

    @abstractmethod
    def load_ingestion_status(
        self,
        *,
        doc_id: str | None = None,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> List[Dict[str, Any]]:
        """Load ingestion status rows for this collection."""

    @abstractmethod
    def clear_ingestion_status(
        self, doc_id: str, *, user_id: int | None = None, is_admin: bool = False
    ) -> None:
        """Remove the ingestion status row for a document in this collection."""

    @abstractmethod
    async def write_ingestion_status_async(
        self,
        doc_id: str,
        *,
        status: str,
        message: str | None = None,
        parse_hash: str | None = None,
        user_id: int | None = None,
    ) -> None:
        """Async :meth:`write_ingestion_status`."""

    @abstractmethod
    async def load_ingestion_status_async(
        self,
        *,
        doc_id: str | None = None,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> List[Dict[str, Any]]:
        """Async :meth:`load_ingestion_status`."""

    @abstractmethod
    async def clear_ingestion_status_async(
        self, doc_id: str, *, user_id: int | None = None, is_admin: bool = False
    ) -> None:
        """Async :meth:`clear_ingestion_status`."""

    # --- Main-pointer data-plane (#513) ---

    @abstractmethod
    def get_main_pointer(
        self, doc_id: str, step_type: str, model_tag: str | None = None
    ) -> Optional[Dict[str, Any]]:
        """Return the main pointer for a document stage, or ``None``.

        Known LanceDB deviation (#2858): with ``model_tag=None`` it never
        matches, since the ``IS NULL`` / ``= ''`` alternatives are passed as a tuple
        and combined with AND. New engines should not treat that as the reference.
        """

    @abstractmethod
    def set_main_pointer(
        self,
        doc_id: str,
        step_type: str,
        semantic_id: str,
        technical_id: str,
        model_tag: str | None = None,
        operator: str | None = None,
    ) -> None:
        """Set or update the main pointer for a document stage."""

    @abstractmethod
    def list_main_pointers(
        self, doc_id: str | None = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List main pointers for this collection.

        Known LanceDB deviation (#2858): on lancedb 0.33 it raises
        ``MainPointerError`` (the empty query builder has no ``count_rows``).
        """

    @abstractmethod
    def delete_main_pointer(
        self, doc_id: str, step_type: str, model_tag: str | None = None
    ) -> bool:
        """Delete the main pointer for a document stage; ``True`` if one existed.

        Known LanceDB deviation (#2858): with ``model_tag=None`` it never
        matches, for the same reason as :meth:`get_main_pointer`, so it deletes
        nothing.
        """

    # --- Version candidates and promotion (#513) ---

    @abstractmethod
    def list_candidates(
        self,
        doc_id: str,
        step_type: StepType | str,
        model_tag: Optional[str] = None,
        state: Optional[str] = None,
        limit: int = 50,
        order_by: str = "created_at desc",
    ) -> Dict[str, Any]:
        """List version candidates for a document stage in this collection.

        Returns:
            A ``dict`` with these keys:
            - ``"candidates"`` – candidate rows after the state filter, sort and limit
            - ``"total_count"`` – rows matching the state filter, before the limit
            - ``"returned_count"`` – number of rows in ``"candidates"``
            - ``"step_type"`` – the resolved step type value
            - ``"model_tag"`` – the requested model tag
            - ``"filters"`` – the requested ``state``, ``limit`` and ``order_by``
        """

    @abstractmethod
    def promote_version_main(
        self,
        doc_id: str,
        step_type: StepType | str,
        selected_id: str,
        operator: Optional[str] = None,
        preview_only: bool = False,
        confirm: bool = False,
        model_tag: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Promote a candidate version to main for a document stage.

        Returns:
            A ``dict`` with these keys:
            - ``"promoted"`` – ``True`` only when the promotion was executed
            - ``"preview"`` – ``True`` when only a preview was returned
            - ``"main_pointer"`` – ``step_type``, ``semantic_id``, ``technical_id``
              and ``model_tag`` of the selected candidate
            - ``"deleted_counts"`` – per-table counts, planned for a preview and
              actual after execution
            - ``"notes"`` – hints such as ``"Requires re-embed"``
            - ``"message"`` – preview only
            - ``"operator"`` – executed promotion only
        """

    # --- Rollback snapshot/restore primitives (#513) ---

    @abstractmethod
    def capture_status_snapshot(
        self, doc_id: str, *, user_id: int | None = None, is_admin: bool = True
    ) -> List[Dict[str, Any]]:
        """Capture ``doc_id``'s ingestion-status rows (empty list if absent)."""

    @abstractmethod
    def restore_status_snapshot(
        self, doc_id: str, snapshot: List[Dict[str, Any]], *, user_id: int | None = None
    ) -> None:
        """Rewrite the snapshot's status rows, or clear the row if it is empty."""

    @abstractmethod
    def clear_status_snapshot(
        self, doc_id: str, *, user_id: int | None = None, is_admin: bool = True
    ) -> None:
        """Clear ``doc_id``'s ingestion-status row after a rollback."""

    @abstractmethod
    def capture_main_pointer_snapshot(
        self, doc_id: str, step_type: str, model_tag: str | None = None
    ) -> KBMainPointerSnapshot:
        """Capture the current main pointer (``pointer=None`` if absent)."""

    @abstractmethod
    def restore_main_pointer_snapshot(
        self, snapshot: KBMainPointerSnapshot, *, operator: str | None = None
    ) -> bool:
        """Restore or delete the pointer; ``False`` if the snapshot is incomplete."""

    @abstractmethod
    def capture_candidate_cleanup_snapshot(
        self,
        doc_id: str,
        scope: str,
        *,
        new_parse_hash: str | None = None,
        old_parse_hash: str | None = None,
        model_tag: str | None = None,
        user_id: int | None = None,
        is_admin: bool | None = None,
    ) -> KBVersionCandidateCleanupSnapshot:
        """Preview what candidate cleanup would delete, without deleting."""

    def restore_candidate_cleanup_snapshot(
        self,
        snapshot: KBVersionCandidateCleanupSnapshot,
        *,
        cleanup_executed: bool = False,
    ) -> KBVersionCandidateRollbackResult:
        """Assess rollback feasibility for a candidate-cleanup snapshot.

        When ``cleanup_executed=True`` and the snapshot recorded side effects,
        returns a result with ``status="incomplete"``, ``restorable=False``,
        and ``side_effects_may_remain=True`` — the inspectable-incomplete
        invariant.  Never issues further deletes; the state is left inspectable.
        """
        cleanup_counts = dict(snapshot.cleanup_counts)
        has_candidate_side_effects = any(
            int(count) > 0 for count in cleanup_counts.values()
        )
        if not cleanup_executed or not has_candidate_side_effects:
            return KBVersionCandidateRollbackResult(
                collection=snapshot.collection,
                doc_id=snapshot.doc_id,
                status="not_needed",
                restorable=True,
                cleanup_counts=cleanup_counts,
            )

        return KBVersionCandidateRollbackResult(
            collection=snapshot.collection,
            doc_id=snapshot.doc_id,
            status="incomplete",
            skipped=True,
            restorable=False,
            reason="candidate_cleanup_not_restorable",
            cleanup_counts=cleanup_counts,
            warnings=(
                "Version candidate cleanup cannot be restored from the handle; "
                "preserve visible rollback state and report remaining side effects.",
            ),
            side_effects_may_remain=True,
        )


@dataclass(frozen=True)
class LanceDBCollectionHandle(KBCollectionHandle):
    """LanceDB-backed collection handle.

    The initial delegate is the current LanceDB documents-table implementation,
    reached through the bound vector index store.
    """

    context: KBCollectionContext

    @property
    def metadata_store(self) -> MetadataStore:
        """Return the metadata store bound to this collection context."""
        return self.context.metadata_store

    @property
    def vector_index_store(self) -> VectorIndexStore:
        """Return the vector index store bound to this collection context."""
        return self.context.vector_index_store

    @property
    def ingestion_status_store(self) -> IngestionStatusStore:
        """Return the ingestion status store bound to this collection context."""
        return self.context.ingestion_status_store

    @property
    def main_pointer_store(self) -> MainPointerStore:
        """Return the main pointer store bound to this collection context."""
        return self.context.main_pointer_store

    @property
    def backend(self) -> KBStorageBackend:
        """Return the collection storage backend."""
        return self.context.backend

    @property
    def capabilities(self) -> KBBackendCapabilities:
        """Return backend capabilities for this collection."""
        return self.context.capabilities

    def register_document(
        self, request: RegisterDocumentRequest
    ) -> RegisterDocumentResponse:
        """Register a document row in this collection's documents table.

        Behavior mirrors the legacy ``_register_document`` helper: input
        validation, file-type detection, deterministic doc_id (with UUID
        fallback), SHA256 content hash, an admin-scoped existence check for the
        ``created`` flag, and an idempotent upsert of the full row.
        """
        # The handle is collection-scoped: persist into the bound context
        # collection rather than trusting request.collection, so a reused handle
        # can never write outside its resolved collection. Through the
        # coordinator the two already match (context.collection is the
        # normalized form of request.collection).
        collection = self.context.collection
        file_id = request.file_id
        source_path = request.source_path
        metadata_source_path = request.metadata_source_path or source_path
        file_type = request.file_type
        doc_id = request.doc_id
        uploaded_at = request.uploaded_at

        if not collection:
            raise DocumentValidationError("Collection name cannot be empty")

        if not source_path or not Path(source_path).exists():
            raise DocumentValidationError(f"Source path does not exist: {source_path}")

        # Auto-detect file type if not provided.
        if not file_type:
            try:
                file_type = check_file_type(source_path)
            except DocumentValidationError as e:
                raise DocumentValidationError(f"File type detection failed: {e}") from e

        # Deterministic doc_id from (collection, file_id/source_path) for
        # idempotent registration; fall back to a UUID if generation fails.
        if not doc_id:
            try:
                stable_key = file_id or metadata_source_path
                doc_id = generate_deterministic_doc_id(collection, stable_key)
            except Exception as e:  # noqa: BLE001 - fallback keeps registration working
                logger.debug(
                    "Deterministic doc_id generation failed (%s), falling back to UUID",
                    e,
                )
                doc_id = str(uuid.uuid4())

        if not uploaded_at:
            uploaded_at = pd.Timestamp.now(tz="UTC")
        elif uploaded_at.tzinfo is None:
            uploaded_at = uploaded_at.replace(tzinfo=timezone.utc)

        try:
            content_hash = compute_file_hash(source_path)
        except Exception as e:
            raise HashComputationError(f"Failed to compute content hash: {e}") from e

        try:
            vector_store = self.vector_index_store

            # Existence check uses admin mode to see all records (incl. legacy).
            exists = (
                vector_store.count_rows_or_zero(
                    "documents",
                    filters={"collection": collection, "doc_id": doc_id},
                    user_id=request.user_id,
                    is_admin=True,
                )
                > 0
            )

            doc_record = {
                "collection": collection,
                "doc_id": doc_id,
                "file_id": file_id,
                "source_path": metadata_source_path,
                "file_type": file_type,
                "content_hash": content_hash,
                "uploaded_at": uploaded_at,
                "title": None,
                "language": None,
                "user_id": request.user_id,
            }

            vector_store.upsert_documents([doc_record])
            created = not exists
        except ConfigurationError:
            raise
        except Exception as e:
            raise DatabaseOperationError(
                f"Failed to register document in database: {e}"
            ) from e

        return RegisterDocumentResponse(
            doc_id=doc_id,
            created=created,
            content_hash=content_hash,
        )

    def load_document(
        self, doc_id: str, *, user_id: int | None = None, is_admin: bool = False
    ) -> DocumentRecordDetail | None:
        """Load a document row by id within this collection's scope.

        Streams the single matching row via ``iter_batches``. Returns ``None``
        when the row is absent or not visible to the given scope.
        """
        vector_store = self.vector_index_store
        query_filters = {"collection": self.context.collection, "doc_id": doc_id}
        try:
            # iter_batches yields only non-empty batches under the same scope
            # filter, so an absent or out-of-scope row yields nothing and we fall
            # through to None -- a separate existence count would be redundant.
            for batch in vector_store.iter_batches(
                table_name="documents",
                filters=query_filters,
                user_id=user_id,
                is_admin=is_admin,
            ):
                # to_pylist() converts the Arrow batch directly to native Python
                # objects in C++, avoiding Pandas' int->float upcasting on null
                # columns (the reason from_row still normalizes defensively).
                for row_dict in batch.to_pylist():
                    return DocumentRecordDetail.from_row(row_dict)
            return None
        except Exception as e:
            raise DatabaseOperationError(f"Failed to retrieve document: {e}") from e

    def list_documents(
        self, *, user_id: int | None = None, is_admin: bool = False, limit: int = 100
    ) -> DocumentRecordListResult:
        """List document rows for this collection.

        Mirrors the legacy file-level ``_list_documents_impl``: a batch scan of
        the documents table filtered by collection, honoring ``limit``.
        """
        vector_store = self.vector_index_store
        query_filters = {"collection": self.context.collection}
        records: list[DocumentRecordDetail] = []
        try:
            for batch in vector_store.iter_batches(
                table_name="documents",
                filters=query_filters,
                user_id=user_id,
                is_admin=is_admin,
            ):
                # to_pylist() bypasses Pandas (see load_document) when materializing
                # rows; from_row still normalizes any residual null sentinels.
                for row_dict in batch.to_pylist():
                    records.append(DocumentRecordDetail.from_row(row_dict))
                    if len(records) >= limit:
                        break
                if len(records) >= limit:
                    break
        except Exception as e:
            raise DatabaseOperationError(f"Failed to list documents: {e}") from e
        return DocumentRecordListResult(documents=records, total_count=len(records))

    def delete_document_record(
        self, doc_id: str, *, user_id: int | None = None, is_admin: bool = False
    ) -> int:
        """Delete only this document's row via the bound store (no cascade)."""
        return self.vector_index_store.delete_document_record(
            collection_name=self.context.collection,
            doc_id=doc_id,
            user_id=user_id,
            is_admin=is_admin,
        )

    def snapshot_document(
        self, doc_id: str, *, user_id: int | None = None, is_admin: bool = False
    ) -> DocumentRecordDetail | None:
        """Capture the current document row before a destructive operation.

        Returns ``None`` when there is no existing row to snapshot. Note that
        these compensation methods are added for #514 to wire into the live
        rollback path; #508 only provides the mechanics.
        """
        return self.load_document(doc_id, user_id=user_id, is_admin=is_admin)

    def restore_document(self, snapshot: DocumentRecordDetail) -> None:
        """Restore a snapshotted document row, preserving every field.

        Re-upserts the full row (keyed by collection + doc_id), so ``file_id``,
        ``user_id``, collection, metadata, content hash, and file type are all
        restored exactly. Refuses snapshots from another collection so the
        collection-scoped boundary holds even on direct handle reuse.
        """
        if snapshot.collection != self.context.collection:
            raise DocumentValidationError(
                f"Handle bound to collection {self.context.collection!r} "
                f"cannot restore a snapshot from {snapshot.collection!r}"
            )
        self.vector_index_store.upsert_documents([snapshot.to_legacy_dict()])

    def delete_created_document(
        self, doc_id: str, *, user_id: int | None = None, is_admin: bool = False
    ) -> int:
        """Idempotently delete a newly created document row (row-only)."""
        return self.delete_document_record(doc_id, user_id=user_id, is_admin=is_admin)

    def _document_row_connection(self) -> Any:
        conn = self.vector_index_store.get_raw_connection()
        ensure_documents_table(conn)
        ensure_parses_table(conn)
        ensure_chunks_table(conn)
        ensure_main_pointers_table(conn)
        ensure_ingestion_runs_table(conn)
        return conn

    def _read_document_rows(
        self, table: Any, doc_ids: Sequence[str], *, user_id: int, is_admin: bool
    ) -> list[dict[str, Any]]:
        if not doc_ids:
            return []
        safe_collection = escape_lancedb_string(self.context.collection)
        doc_filters = [
            _owned_row_filter(
                table,
                [
                    f"collection = '{safe_collection}'",
                    f"doc_id = '{escape_lancedb_string(doc_id)}'",
                ],
                user_id=user_id,
                is_admin=is_admin,
            )
            for doc_id in doc_ids
        ]
        return query_to_list(table.search().where(_any_of(doc_filters)).limit(-1))

    def capture_document_rows(
        self, doc_ids: Sequence[str], *, user_id: int, is_admin: bool
    ) -> KBDocumentRowsSnapshot:
        """Capture every row of ``doc_ids`` in the document and embedding tables.

        Non-admin reads are limited to ``user_id`` on tables with that column.
        """
        if isinstance(doc_ids, str):
            raise DocumentValidationError(
                "doc_ids must be a sequence of ids, not a str"
            )
        doc_ids = tuple(doc_ids)
        conn = self._document_row_connection()
        table_names = set(list_table_names(conn))
        target_tables = [name for name in _DOCUMENT_ROW_KEYS if name in table_names]
        target_tables.extend(
            sorted(name for name in table_names if name.startswith("embeddings_"))
        )
        rows_by_table: dict[str, list[dict[str, Any]]] = {}
        for table_name in target_tables:
            table = None
            try:
                table = conn.open_table(table_name)
                rows_by_table[table_name] = self._read_document_rows(
                    table, doc_ids, user_id=user_id, is_admin=is_admin
                )
            finally:
                _safe_close_table(table)
        return KBDocumentRowsSnapshot(
            collection=self.context.collection,
            doc_ids=doc_ids,
            rows_by_table=rows_by_table,
        )

    @guard_document_restore
    def restore_document_rows(
        self, snapshot: KBDocumentRowsSnapshot, *, user_id: int, is_admin: bool
    ) -> list[str]:
        """Restore a :meth:`capture_document_rows` snapshot table by table.

        Rows of the snapshot's documents that it does not hold are deleted, so
        rows written after the capture do not survive. Reads and deletes are
        limited to ``user_id`` on tables with that column; upserts match by row
        key only.
        """
        if snapshot.collection != self.context.collection:
            raise DocumentValidationError(
                f"Handle bound to collection {self.context.collection!r} "
                f"cannot restore a snapshot from {snapshot.collection!r}"
            )
        for rows in snapshot.rows_by_table.values():
            for row in rows:
                if (
                    row.get("collection") != self.context.collection
                    or row.get("doc_id") not in snapshot.doc_ids
                ):
                    raise DocumentValidationError(
                        f"Snapshot of {self.context.collection!r} holds a row of "
                        f"{row.get('collection')!r}/{row.get('doc_id')!r} "
                        "outside its documents"
                    )
        conn = self._document_row_connection()
        table_names = set(list_table_names(conn))
        restore_tables = [
            name
            for name in _DOCUMENT_ROW_KEYS
            if name in table_names or name in snapshot.rows_by_table
        ]
        restore_tables.extend(
            sorted(name for name in table_names if name.startswith("embeddings_"))
        )
        for name in snapshot.rows_by_table:
            if name.startswith("embeddings_") and name not in restore_tables:
                restore_tables.append(name)

        for table_name in restore_tables:
            table = None
            try:
                table = conn.open_table(table_name)
                _restore_document_table_rows(
                    table,
                    table_name=table_name,
                    snapshot_rows=snapshot.rows_by_table.get(table_name, []),
                    current_rows=self._read_document_rows(
                        table, snapshot.doc_ids, user_id=user_id, is_admin=is_admin
                    ),
                    user_id=user_id,
                    is_admin=is_admin,
                )
            finally:
                _safe_close_table(table)

        invalidate_cache = getattr(
            self.vector_index_store, "invalidate_table_cache", None
        )
        if callable(invalidate_cache):
            invalidate_cache()
        return []

    # --- Parse data-plane (#509) ---

    def parse_exists(
        self,
        doc_id: str,
        parse_hash: str,
        *,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> bool:
        """Return whether a parse row exists for ``(doc_id, parse_hash)``."""
        try:
            return bool(
                self.vector_index_store.count_rows_or_zero(
                    "parses",
                    filters={
                        "collection": self.context.collection,
                        "doc_id": doc_id,
                        "parse_hash": parse_hash,
                    },
                    user_id=user_id,
                    is_admin=is_admin,
                )
                > 0
            )
        except Exception as e:
            raise DatabaseOperationError(f"Database query failed: {e}") from e

    def read_parse_paragraphs(
        self,
        doc_id: str,
        parse_hash: str,
        *,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> list[ParsedParagraph]:
        """Return the reuse-hit parsed paragraphs for ``(doc_id, parse_hash)``."""
        vector_store = self.vector_index_store
        query_filters = {
            "collection": self.context.collection,
            "doc_id": doc_id,
            "parse_hash": parse_hash,
        }
        try:
            if (
                vector_store.count_rows_or_zero(
                    "parses", filters=query_filters, user_id=user_id, is_admin=is_admin
                )
                == 0
            ):
                return []
            for batch in vector_store.iter_batches(
                table_name="parses",
                filters=query_filters,
                user_id=user_id,
                is_admin=is_admin,
            ):
                for record in batch.to_pylist():
                    parsed_content = record.get("parsed_content")
                    if not parsed_content:
                        continue
                    data = json.loads(parsed_content)
                    return [
                        ParsedParagraph(
                            text=item.get("text", ""),
                            metadata=item.get("metadata", {}),
                        )
                        for item in data
                    ]
            return []
        except Exception as e:
            logger.error("Failed to read parse content: %s", e)
            raise DatabaseOperationError(f"Failed reading parse content: {e}") from e

    def write_parse(
        self,
        doc_id: str,
        parse_hash: str,
        parse_method: Any,
        params: dict[str, Any],
        paragraphs: list[ParsedParagraph],
        *,
        user_id: int | None = None,
    ) -> bool:
        """Persist a parse row into this collection (idempotent upsert)."""
        try:
            parsed_content = json.dumps(
                [para.model_dump() for para in paragraphs], ensure_ascii=False
            )
            parse_record = {
                "collection": self.context.collection,
                "doc_id": doc_id,
                "parse_hash": parse_hash,
                "parser": f"local:{parse_method}@v1.0.0",
                "created_at": pd.Timestamp.now(tz="UTC"),
                "params_json": json.dumps(params, ensure_ascii=False),
                "parsed_content": parsed_content,
                "user_id": user_id,
            }
            self.vector_index_store.upsert_parses([parse_record])
            return True
        except Exception as e:
            raise DatabaseOperationError(f"Database write failed: {e}") from e

    def read_latest_parse_record(
        self,
        doc_id: str,
        parse_hash: str | None = None,
        *,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> ParseRecordDetail | None:
        """Return the latest parse row (by ``created_at``) for display."""
        vector_store = self.vector_index_store
        query_filters: dict[str, Any] = {
            "collection": self.context.collection,
            "doc_id": doc_id,
        }
        if parse_hash:
            query_filters["parse_hash"] = parse_hash
        try:
            if (
                vector_store.count_rows_or_zero(
                    "parses", filters=query_filters, user_id=user_id, is_admin=is_admin
                )
                == 0
            ):
                return None
            records: list[dict[str, Any]] = []
            for batch in vector_store.iter_batches(
                table_name="parses",
                filters=query_filters,
                user_id=user_id,
                is_admin=is_admin,
            ):
                records.extend(batch.to_pylist())
            if not records:
                return None

            # Latest by created_at desc; (t is not None, t) sorts None rows last.
            def _created_at_key(record: dict[str, Any]) -> Any:
                created_at = record.get("created_at")
                return (created_at is not None, created_at)

            records_sorted = sorted(records, key=_created_at_key, reverse=True)
            return ParseRecordDetail.from_row(records_sorted[0])
        except Exception as e:
            logger.error("Failed to read latest parse record: %s", e)
            raise DatabaseOperationError(f"Failed to read parse result: {e}") from e

    # --- Chunk data-plane (#509) ---

    def chunk_exists(
        self,
        doc_id: str,
        parse_hash: str,
        config_hash: str,
        *,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> bool:
        """Return whether chunk rows exist for the given config."""
        try:
            return bool(
                self.vector_index_store.count_rows_or_zero(
                    "chunks",
                    filters={
                        "collection": self.context.collection,
                        "doc_id": doc_id,
                        "parse_hash": parse_hash,
                        "config_hash": config_hash,
                    },
                    user_id=user_id,
                    is_admin=is_admin,
                )
                > 0
            )
        except Exception as e:
            raise DatabaseOperationError(f"Database query failed: {e}") from e

    def read_existing_chunks(
        self,
        doc_id: str,
        parse_hash: str,
        config_hash: str,
        *,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> list[dict[str, Any]]:
        """Return the reuse-hit chunk dicts (metadata deserialized)."""
        vector_store = self.vector_index_store
        query_filters = {
            "collection": self.context.collection,
            "doc_id": doc_id,
            "parse_hash": parse_hash,
            "config_hash": config_hash,
        }
        try:
            if (
                vector_store.count_rows_or_zero(
                    "chunks", filters=query_filters, user_id=user_id, is_admin=is_admin
                )
                == 0
            ):
                return []

            chunks: list[dict[str, Any]] = []
            for batch in vector_store.iter_batches(
                table_name="chunks",
                filters=query_filters,
                user_id=user_id,
                is_admin=is_admin,
            ):
                for row in batch.to_pylist():
                    index_value = row.get("index")
                    chunks.append(
                        {
                            "chunk_id": row["chunk_id"],
                            "index": int(index_value) if index_value is not None else 0,
                            "text": row["text"],
                            "page_number": row.get("page_number"),
                            "section": row.get("section"),
                            "anchor": row.get("anchor"),
                            "json_path": row.get("json_path"),
                            "created_at": row["created_at"],
                            "metadata": deserialize_metadata(row.get("metadata")),
                        }
                    )
            # LanceDB does not guarantee scan order; sort by index so reused
            # chunks come back deterministically (mirrors snapshot_chunks).
            chunks.sort(key=lambda chunk: chunk["index"])
            return chunks
        except Exception as e:
            logger.error("Failed to get existing chunks: %s", e)
            raise DatabaseOperationError(f"Database query failed: {e}") from e

    def read_parse_paragraph_dicts(
        self,
        doc_id: str,
        parse_hash: str,
        *,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> list[dict[str, Any]]:
        """Return parsed paragraphs as ``{text, metadata}`` dicts for chunking."""
        vector_store = self.vector_index_store
        query_filters = {
            "collection": self.context.collection,
            "doc_id": doc_id,
            "parse_hash": parse_hash,
        }
        try:
            if (
                vector_store.count_rows_or_zero(
                    "parses", filters=query_filters, user_id=user_id, is_admin=is_admin
                )
                == 0
            ):
                return []

            records: list[dict[str, Any]] = []
            for batch in vector_store.iter_batches(
                table_name="parses",
                filters=query_filters,
                user_id=user_id,
                is_admin=is_admin,
            ):
                records.extend(batch.to_pylist())

            if not records:
                return []
            parsed_content = records[0].get("parsed_content")
            if not parsed_content:
                return []
            data = json.loads(parsed_content)
            return [
                {"text": item.get("text", ""), "metadata": item.get("metadata", {})}
                for item in data
            ]
        except Exception as e:
            logger.error("Failed to read parses: %s", e)
            raise DatabaseOperationError(f"Failed reading parses: {e}") from e

    def write_chunks(
        self,
        doc_id: str,
        parse_hash: str,
        config_hash: str,
        params: dict[str, Any],
        chunks: list[dict[str, Any]],
        *,
        user_id: int | None = None,
    ) -> bool:
        """Persist chunk rows into this collection (idempotent upsert)."""
        try:
            rows = []
            for chunk in chunks:
                text = chunk["text"]
                rows.append(
                    {
                        "collection": self.context.collection,
                        "doc_id": doc_id,
                        "parse_hash": parse_hash,
                        "chunk_id": chunk["chunk_id"],
                        "index": int(chunk["index"]),
                        "text": text,
                        "page_number": chunk.get("page_number"),
                        "section": chunk.get("section"),
                        "anchor": chunk.get("anchor"),
                        "json_path": chunk.get("json_path"),
                        "chunk_hash": compute_chunk_hash(text, params),
                        "config_hash": config_hash,
                        "created_at": chunk["created_at"],
                        "metadata": serialize_metadata(chunk.get("metadata")),
                        "user_id": user_id,
                    }
                )

            if not rows:
                return False

            self.vector_index_store.upsert_chunks(rows)
            return True
        except Exception as e:
            logger.error("Failed to write chunk records: %s", e)
            raise DatabaseOperationError(f"Database write failed: {e}") from e

    # --- Embedding data-plane (#510) ---

    def validate_query_vector(
        self,
        query_vector: list[float],
        *,
        model_tag: str | None = None,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> None:
        """Validate a query vector's format and content (no store access)."""
        validate_query_vector_format(query_vector)

    def read_chunks_needing_embedding(
        self,
        doc_id: str,
        parse_hash: str,
        model: str,
        *,
        filters: dict[str, Any] | None = None,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> EmbeddingReadResponse:
        """Return chunks that still need an embedding for ``model``."""
        collection = self.context.collection
        try:
            if not collection or not doc_id or not parse_hash or not model:
                raise DocumentValidationError(
                    "Collection, doc_id, parse_hash, and model are required"
                )

            vector_store = self.vector_index_store
            query_filters: dict[str, Any] = {
                "collection": collection,
                "doc_id": doc_id,
                "parse_hash": parse_hash,
            }
            if filters:
                query_filters.update(filters)

            total_count = vector_store.count_rows_or_zero(
                table_name="chunks",
                filters=query_filters,
                user_id=user_id,
                is_admin=is_admin,
            )
            if total_count == 0:
                return EmbeddingReadResponse(chunks=[], total_count=0, pending_count=0)

            chunks_data: list[dict[str, Any]] = []
            for batch in vector_store.iter_batches(
                table_name="chunks",
                columns=None,
                batch_size=1000,
                filters=query_filters,
                user_id=user_id,
                is_admin=is_admin,
            ):
                chunks_data.extend(batch.to_pylist())
                if len(chunks_data) >= total_count:
                    break

            # Chunks already embedded for this model tag are excluded by
            # chunk_id presence in the per-model embeddings table.
            embedded_chunk_ids: set[str] = set()
            model_tag = to_model_tag(model)
            embeddings_table_name = f"embeddings_{model_tag}"
            try:
                embedding_filters: dict[str, Any] = {
                    "collection": collection,
                    "doc_id": doc_id,
                    "parse_hash": parse_hash,
                }
                embedding_count = vector_store.count_rows_or_zero(
                    table_name=embeddings_table_name,
                    filters=embedding_filters,
                    user_id=user_id,
                    is_admin=is_admin,
                )
                if embedding_count > 0:
                    for batch in vector_store.iter_batches(
                        table_name=embeddings_table_name,
                        columns=["chunk_id"],
                        filters=embedding_filters,
                        user_id=user_id,
                        is_admin=is_admin,
                    ):
                        for row in batch.to_pylist():
                            chunk_id = row.get("chunk_id")
                            if chunk_id is not None:
                                embedded_chunk_ids.add(chunk_id)
            except Exception as e:  # noqa: BLE001 - missing/absent table = none embedded
                logger.warning(
                    "Failed to query existing embeddings for model %s "
                    "(assuming none exist): %s",
                    model,
                    e,
                )
                embedded_chunk_ids = set()

            pending_chunks: list[ChunkForEmbedding] = []
            for chunk_dict in chunks_data:
                chunk_id = chunk_dict["chunk_id"]
                if chunk_id in embedded_chunk_ids:
                    continue
                pending_chunks.append(_chunk_for_embedding(chunk_dict))

            return EmbeddingReadResponse(
                chunks=pending_chunks,
                total_count=total_count,
                pending_count=len(pending_chunks),
            )
        except Exception as e:
            if isinstance(
                e,
                (
                    DocumentValidationError,
                    DatabaseOperationError,
                    ConfigurationError,
                    VectorValidationError,
                ),
            ):
                raise
            logger.error("Failed to read chunks for embedding: %s", e)
            raise DatabaseOperationError(
                f"Failed to read chunks for embedding: {e}"
            ) from e

    def write_embeddings(
        self,
        embeddings: list[ChunkEmbeddingData],
        *,
        create_index: bool = True,
        user_id: int | None = None,
    ) -> EmbeddingWriteResponse:
        """Write embedding vectors for this collection (idempotent upsert)."""
        if not embeddings:
            return EmbeddingWriteResponse(
                upsert_count=0,
                deleted_stale_count=0,
                index_status=IndexOperation.SKIPPED.value,
            )

        collection = self.context.collection
        try:
            if not collection:
                raise DocumentValidationError("Collection name is required")

            embeddings_by_model: dict[str, list[ChunkEmbeddingData]] = {}
            for embedding in embeddings:
                embeddings_by_model.setdefault(embedding.model, []).append(embedding)

            total_upserted = 0
            index_statuses: list[str] = []
            for model, model_embeddings in embeddings_by_model.items():
                upserted, idx_status = self._process_model_embeddings(
                    model, model_embeddings, create_index, user_id
                )
                total_upserted += upserted
                index_statuses.append(idx_status)

            # Map create_index result strings onto IndexOperation.
            if "index_building" in index_statuses:
                overall = IndexOperation.CREATED
            elif "index_ready" in index_statuses:
                overall = IndexOperation.READY
            elif "failed" in index_statuses or "index_corrupted" in index_statuses:
                overall = IndexOperation.FAILED
            elif "below_threshold" in index_statuses:
                overall = IndexOperation.SKIPPED_THRESHOLD
            else:
                overall = IndexOperation.SKIPPED

            return EmbeddingWriteResponse(
                upsert_count=total_upserted,
                deleted_stale_count=0,  # merge_insert handles updates automatically
                index_status=overall.value,
            )
        except Exception as e:
            if isinstance(
                e,
                (
                    DocumentValidationError,
                    DatabaseOperationError,
                    ConfigurationError,
                    VectorValidationError,
                ),
            ):
                raise
            logger.error("Failed to write embeddings to database: %s", e)
            raise DatabaseOperationError(
                f"Failed to write embeddings to database: {e}"
            ) from e

    def commit_embeddings(
        self,
        doc_id: str,
        parse_hash: str,
        model: str,
        *,
        commit_gate: Callable[[], None] | None = None,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> None:
        """Do nothing: LanceDB writes searchable rows."""

    def discard_uncommitted_embeddings(
        self, doc_id: str, *, user_id: int | None = None
    ) -> int:
        """Do nothing: LanceDB has no invisible rows."""
        return 0

    def _process_model_embeddings(
        self,
        model: str,
        model_embeddings: list[ChunkEmbeddingData],
        create_index: bool,
        user_id: int | None,
    ) -> tuple[int, str]:
        """Upsert one model's embeddings via the bound store (batched)."""
        model_tag = to_model_tag(model)
        vector_store = self.vector_index_store

        first_dim = len(model_embeddings[0].vector)
        unique_dims = {len(item.vector) for item in model_embeddings}
        if len(unique_dims) > 1:
            raise VectorValidationError(
                f"Multiple vector dimensions found for model {model}: {unique_dims}"
            )

        original_batch_size = _int_env("LANCEDB_BATCH_SIZE", DEFAULT_LANCEDB_BATCH_SIZE)
        batch_size = original_batch_size
        batch_timestamp = pd.Timestamp.now(tz="UTC")
        max_spill_retries = _int_env("LANCEDB_MAX_SPILL_RETRIES", 3)
        spill_retry_count = 0

        upserted_count = 0
        current_idx = 0
        total_embeddings = len(model_embeddings)

        while current_idx < total_embeddings:
            end_idx = min(current_idx + batch_size, total_embeddings)
            batch_embeddings = model_embeddings[current_idx:end_idx]

            records_to_merge = [
                {
                    "collection": self.context.collection,
                    "doc_id": embedding.doc_id,
                    "chunk_id": embedding.chunk_id,
                    "parse_hash": embedding.parse_hash,
                    "model": model,
                    "vector": embedding.vector,
                    "text": embedding.text,
                    "chunk_hash": embedding.chunk_hash,
                    "created_at": batch_timestamp,
                    "vector_dimension": first_dim,
                    "metadata": serialize_metadata(embedding.metadata),
                    "user_id": user_id,
                }
                for embedding in batch_embeddings
            ]

            try:
                vector_store.upsert_embeddings(model_tag, records_to_merge)
                upserted_count += len(records_to_merge)
                current_idx = end_idx
                spill_retry_count = 0
            except Exception as batch_error:  # noqa: BLE001 - spill-retry then re-raise
                # TODO: brittle string match; replace with a typed lancedb spill
                # exception if/when one is exposed (cf. is_non_recoverable_merge_error).
                if "Spill has sent an error" in str(batch_error):
                    spill_retry_count += 1
                    if spill_retry_count <= max_spill_retries:
                        if batch_size > 50:
                            batch_size = max(50, batch_size // 2)
                            logger.info(
                                "Reducing batch size to %d and retrying "
                                "(spill retry %d/%d)",
                                batch_size,
                                spill_retry_count,
                                max_spill_retries,
                            )
                        continue
                raise

        index_status: str = IndexOperation.SKIPPED.value
        if create_index:
            try:
                index_status = vector_store.create_index(
                    model_tag, readonly=False
                ).status
            except Exception as index_error:  # noqa: BLE001 - index failure is non-fatal
                logger.warning(
                    "Failed to create index for embeddings_%s: %s",
                    model_tag,
                    index_error,
                )
                index_status = IndexOperation.FAILED.value

        return upserted_count, index_status

    def delete_embedding_records(
        self,
        doc_id: str,
        *,
        parse_hash: str | None = None,
        chunk_ids: list[str] | None = None,
        model_tag: str | None = None,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> int:
        """Delete embedding rows for a document via the bound store (no cascade)."""
        return self.vector_index_store.delete_embedding_records(
            self.context.collection,
            doc_id,
            parse_hash=parse_hash,
            chunk_ids=chunk_ids,
            model_tag=model_tag,
            user_id=user_id,
            is_admin=is_admin,
        )

    # --- Embedding rollback compensation (methods only; wiring in #514) ---

    def _embedding_table_names(self, model_tag: str | None) -> list[str]:
        """List bound embedding tables, optionally scoped to one model tag."""
        tables = [
            name
            for name in self.vector_index_store.list_table_names()
            if name.startswith("embeddings_")
        ]
        if model_tag is None:
            return tables
        candidates = {
            f"embeddings_{model_tag}",
            f"embeddings_{to_model_tag(model_tag)}",
        }
        return [name for name in tables if name in candidates]

    def snapshot_embeddings(
        self,
        doc_id: str,
        parse_hash: str,
        *,
        chunk_ids: list[str] | None = None,
        model_tag: str | None = None,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> EmbeddingRecordSnapshot | None:
        """Capture embedding rows across matching model tables (None if absent)."""
        vector_store = self.vector_index_store
        query_filters = {
            "collection": self.context.collection,
            "doc_id": doc_id,
            "parse_hash": parse_hash,
        }
        chunk_id_set = set(chunk_ids) if chunk_ids else None
        rows: list[dict[str, Any]] = []
        try:
            for table_name in self._embedding_table_names(model_tag):
                if (
                    vector_store.count_rows_or_zero(
                        table_name,
                        filters=query_filters,
                        user_id=user_id,
                        is_admin=is_admin,
                    )
                    == 0
                ):
                    continue
                for batch in vector_store.iter_batches(
                    table_name=table_name,
                    filters=query_filters,
                    user_id=user_id,
                    is_admin=is_admin,
                ):
                    for row in batch.to_pylist():
                        if chunk_id_set is not None and row.get("chunk_id") not in (
                            chunk_id_set
                        ):
                            continue
                        rows.append(row)
        except Exception as e:
            logger.error("Failed to snapshot embeddings: %s", e)
            raise DatabaseOperationError(f"Failed to snapshot embeddings: {e}") from e

        if not rows:
            return None
        # Deterministic order across tables for a faithful restore/round trip.
        rows.sort(key=lambda row: (row.get("model") or "", row.get("chunk_id") or ""))
        return EmbeddingRecordSnapshot.from_rows(rows)

    def restore_embeddings(self, snapshot: EmbeddingRecordSnapshot) -> None:
        """Restore snapshotted embedding rows, grouped per model tag."""
        if not snapshot.records:
            return
        for record in snapshot.records:
            if record.collection != self.context.collection:
                raise DocumentValidationError(
                    f"Handle bound to collection {self.context.collection!r} "
                    f"cannot restore an embedding snapshot from "
                    f"{record.collection!r}"
                )
        for model_tag, rows in snapshot.group_by_model_tag().items():
            self.vector_index_store.upsert_embeddings(model_tag, rows)

    def delete_created_embeddings(
        self,
        doc_id: str,
        parse_hash: str,
        *,
        chunk_ids: list[str] | None = None,
        model_tag: str | None = None,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> int:
        """Idempotently delete newly created embedding rows (compensation)."""
        return self.delete_embedding_records(
            doc_id,
            parse_hash=parse_hash,
            chunk_ids=chunk_ids,
            model_tag=model_tag,
            user_id=user_id,
            is_admin=is_admin,
        )

    # --- Search data-plane (#511) ---

    def _dense_unsupported(self, model_tag: str) -> DenseSearchResponse:
        return DenseSearchResponse(
            results=[],
            total_count=0,
            status="failed",
            warnings=[
                SearchWarning(
                    code="SEARCH_NOT_SUPPORTED",
                    message="This backend does not support search.",
                    fallback_action=SearchFallbackAction.PARTIAL_RESULTS,
                    affected_models=[model_tag],
                )
            ],
            index_status=IndexStatus.NO_INDEX,
            index_advice=None,
            idempotency_key=None,
            fallback_info=None,
            nprobes=None,
            refine_factor=None,
        )

    @staticmethod
    def _map_index_status(index_status: str) -> IndexStatus:
        return {
            "index_building": IndexStatus.INDEX_BUILDING,
            "no_index": IndexStatus.NO_INDEX,
            "index_corrupted": IndexStatus.INDEX_CORRUPTED,
            "readonly": IndexStatus.READONLY,
            "below_threshold": IndexStatus.BELOW_THRESHOLD,
        }.get(index_status, IndexStatus.INDEX_READY)

    def _dense_engine(
        self,
        collection: str,
        model_tag: str,
        query_vector: list[float],
        *,
        top_k: int,
        filters: dict | None,
        readonly: bool,
        nprobes: int | None,
        refine_factor: int | None,
        user_id: int | None,
        is_admin: bool,
    ) -> tuple[list[SearchResult], str, str | None]:
        try:
            vector_store = self.vector_index_store
            index_result_obj = vector_store.create_index(model_tag, readonly)
            index_status = index_result_obj.status
            index_advice = index_result_obj.advice
            filter_expr: FilterExpression | None = None
            if collection or filters:
                conditions: list[FilterExpression] = []
                if collection:
                    conditions.append(
                        FilterCondition(
                            field="collection",
                            operator=FilterOperator.EQ,
                            value=collection,
                        )
                    )
                if filters:
                    parsed = (
                        parse_legacy_filters(filters)
                        if isinstance(filters, dict)
                        else None
                    )
                    if parsed is not None:
                        if isinstance(parsed, tuple):
                            conditions.extend(parsed)
                        else:
                            conditions.append(parsed)
                if len(conditions) == 1:
                    filter_expr = conditions[0]
                elif len(conditions) > 1:
                    filter_expr = tuple(conditions)
            if filter_expr is not None:
                validate_filter_depth(filter_expr)
            raw_results = vector_store.search_vectors_by_model(
                model_tag=model_tag,
                query_vector=query_vector,
                top_k=top_k,
                filters=filter_expr,
                vector_column_name="vector",
                user_id=user_id,
                is_admin=is_admin,
            )
            search_results = []
            for row in raw_results:
                distance_value = row.get("_distance")
                distance = float(distance_value) if distance_value is not None else 0.0
                score = 1.0 / (1.0 + max(0.0, distance))
                metadata = deserialize_metadata(row.get("metadata"))
                search_results.append(
                    SearchResult(
                        doc_id=row["doc_id"],
                        chunk_id=row["chunk_id"],
                        text=row["text"],
                        score=score,
                        parse_hash=row.get("parse_hash"),
                        model_tag=model_tag,
                        created_at=row.get("created_at"),
                        metadata=metadata,
                    )
                )
            return search_results, index_status, index_advice
        except Exception as e:
            logger.error("Failed to execute dense search: %s", str(e))
            raise

    async def _dense_engine_async(
        self,
        collection: str,
        model_tag: str,
        query_vector: list[float],
        *,
        top_k: int,
        filters: dict | None,
        readonly: bool,
        nprobes: int | None,
        refine_factor: int | None,
        user_id: int | None,
        is_admin: bool,
    ) -> tuple[list[SearchResult], str, str | None]:
        try:
            vector_store = self.vector_index_store
            index_result_obj = vector_store.create_index(model_tag, readonly)
            index_status = index_result_obj.status
            index_advice = index_result_obj.advice
            filter_expr: FilterExpression | None = None
            if collection or filters:
                conditions: list[FilterExpression] = []
                if collection:
                    conditions.append(
                        FilterCondition(
                            field="collection",
                            operator=FilterOperator.EQ,
                            value=collection,
                        )
                    )
                if filters:
                    parsed = (
                        parse_legacy_filters(filters)
                        if isinstance(filters, dict)
                        else None
                    )
                    if parsed is not None:
                        if isinstance(parsed, tuple):
                            conditions.extend(parsed)
                        else:
                            conditions.append(parsed)
                if len(conditions) == 1:
                    filter_expr = conditions[0]
                elif len(conditions) > 1:
                    filter_expr = tuple(conditions)
            if filter_expr is not None:
                validate_filter_depth(filter_expr)
            raw_results = await vector_store.search_vectors_by_model_async(
                model_tag=model_tag,
                query_vector=query_vector,
                top_k=top_k,
                filters=filter_expr,
                vector_column_name="vector",
                user_id=user_id,
                is_admin=is_admin,
            )
            search_results = []
            for row in raw_results:
                distance_value = row.get("_distance")
                distance = float(distance_value) if distance_value is not None else 0.0
                score = 1.0 / (1.0 + max(0.0, distance))
                metadata = deserialize_metadata(row.get("metadata"))
                search_results.append(
                    SearchResult(
                        doc_id=row["doc_id"],
                        chunk_id=row["chunk_id"],
                        text=row["text"],
                        score=score,
                        parse_hash=row.get("parse_hash"),
                        model_tag=model_tag,
                        created_at=row.get("created_at"),
                        metadata=metadata,
                    )
                )
            return search_results, index_status, index_advice
        except Exception as e:
            logger.error("Failed to execute async dense search: %s", str(e))
            raise

    def search_dense(
        self,
        model_tag: str,
        query_vector: list[float],
        *,
        top_k: int = 10,
        filters: dict | None = None,
        readonly: bool = False,
        nprobes: int | None = None,
        refine_factor: int | None = None,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> DenseSearchResponse:
        if not self.capabilities.supports_search:
            return self._dense_unsupported(model_tag)
        collection = self.context.collection
        try:
            results, index_status, index_advice = self._dense_engine(
                collection,
                model_tag,
                query_vector,
                top_k=top_k,
                filters=filters,
                readonly=readonly,
                nprobes=nprobes,
                refine_factor=refine_factor,
                user_id=user_id,
                is_admin=is_admin,
            )
            return DenseSearchResponse(
                results=results,
                total_count=len(results),
                status="success",
                warnings=[],
                index_status=self._map_index_status(index_status),
                index_advice=index_advice,
                idempotency_key=None,
                fallback_info=None,
                nprobes=nprobes,
                refine_factor=refine_factor,
            )
        except Exception as e:  # noqa: BLE001 - search returns failed response, never raises
            logger.error(
                "Dense search failed for %s in collection '%s': %s",
                model_tag,
                collection,
                e,
            )
            return DenseSearchResponse(
                results=[],
                total_count=0,
                status="failed",
                warnings=[
                    SearchWarning(
                        code="DENSE_SEARCH_FAILED",
                        message=f"An unexpected error occurred during dense search: {e}",
                        fallback_action=SearchFallbackAction.PARTIAL_RESULTS,
                        affected_models=[model_tag],
                    )
                ],
                index_status=IndexStatus.NO_INDEX,
                index_advice=None,
                idempotency_key=None,
                fallback_info=None,
                nprobes=nprobes,
                refine_factor=refine_factor,
            )

    async def search_dense_async(
        self,
        model_tag: str,
        query_vector: list[float],
        *,
        top_k: int = 10,
        filters: dict | None = None,
        readonly: bool = False,
        nprobes: int | None = None,
        refine_factor: int | None = None,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> DenseSearchResponse:
        if not self.capabilities.supports_search:
            return self._dense_unsupported(model_tag)
        collection = self.context.collection
        try:
            results, index_status, index_advice = await self._dense_engine_async(
                collection,
                model_tag,
                query_vector,
                top_k=top_k,
                filters=filters,
                readonly=readonly,
                nprobes=nprobes,
                refine_factor=refine_factor,
                user_id=user_id,
                is_admin=is_admin,
            )
            return DenseSearchResponse(
                results=results,
                total_count=len(results),
                status="success",
                warnings=[],
                index_status=self._map_index_status(index_status),
                index_advice=index_advice,
                idempotency_key=None,
                fallback_info=None,
                nprobes=nprobes,
                refine_factor=refine_factor,
            )
        except Exception as e:  # noqa: BLE001 - search returns failed response, never raises
            logger.error(
                "Dense search failed (async) for %s in collection '%s': %s",
                model_tag,
                collection,
                e,
            )
            return DenseSearchResponse(
                results=[],
                total_count=0,
                status="failed",
                warnings=[
                    SearchWarning(
                        code="DENSE_SEARCH_FAILED",
                        message=f"An unexpected error occurred during dense search: {e}",
                        fallback_action=SearchFallbackAction.PARTIAL_RESULTS,
                        affected_models=[model_tag],
                    )
                ],
                index_status=IndexStatus.NO_INDEX,
                index_advice=None,
                idempotency_key=None,
                fallback_info=None,
                nprobes=nprobes,
                refine_factor=refine_factor,
            )

    def _sparse_unsupported(
        self, model_tag: str, query_text: str
    ) -> SparseSearchResponse:
        return SparseSearchResponse(
            results=[],
            total_count=0,
            status="failed",
            warnings=[
                SearchWarning(
                    code="SEARCH_NOT_SUPPORTED",
                    message="This backend does not support search.",
                    fallback_action=SearchFallbackAction.PARTIAL_RESULTS,
                    affected_models=[model_tag],
                )
            ],
            fts_enabled=False,
            query_text=query_text,
        )

    @staticmethod
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

    def _substring_fallback(
        self,
        *,
        table_name: str,
        collection: str,
        query_text: str,
        model_tag: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        current_warnings: List[SearchWarning],
        user_id: Optional[int] = None,
        is_admin: bool = False,
        batch_size: int = 2048,
    ) -> List[SearchResult]:
        """Perform a memory-friendly substring scan across the table when FTS misses."""

        query_filters: Dict[str, Any] = (
            dict(filters) if isinstance(filters, dict) else {}
        )
        # Caller filters AND the handle's collection, as in the FTS query.
        if query_filters.setdefault("collection", collection) != collection:
            return []

        results: List[SearchResult] = []

        try:
            for batch in self.vector_index_store.iter_batches(
                table_name=table_name,
                columns=[
                    "doc_id",
                    "chunk_id",
                    "text",
                    "parse_hash",
                    "created_at",
                    "metadata",
                ],
                batch_size=batch_size,
                filters=query_filters,
                user_id=user_id,
                is_admin=is_admin,
            ):
                batch_df = batch.to_pandas()
                text_mask = (
                    batch_df["text"]
                    .astype(str)
                    .str.contains(query_text, na=False, regex=False)
                )

                for _, row in batch_df.loc[text_mask].iterrows():
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
        except Exception as exc:  # noqa: BLE001
            logger.error("Substring fallback failed to read batches: %s", exc)
            return []

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

    async def _substring_fallback_async(
        self,
        *,
        model_tag: str,
        collection: str,
        query_text: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        current_warnings: List[SearchWarning],
        user_id: Optional[int] = None,
        is_admin: bool = False,
        batch_size: int = 2048,
    ) -> List[SearchResult]:
        """Perform async substring scan using iter_batches_async when FTS misses."""

        vector_store = self.vector_index_store
        results: List[SearchResult] = []

        # Build query filters
        query_filters: Dict[str, Any] = {"collection": collection}
        if filters and isinstance(filters, dict):
            query_filters.update(filters)

        _table = None
        try:
            # Open embeddings table with legacy fallback
            _table, table_name = vector_store.open_embeddings_table(model_tag)

            # Use async batch iteration for memory-efficient scanning
            # Specify only required columns to minimize memory usage
            async for batch in cast(
                AsyncIterator[Any],
                vector_store.iter_batches_async(
                    table_name=table_name,
                    columns=[
                        "doc_id",
                        "chunk_id",
                        "text",
                        "parse_hash",
                        "created_at",
                        "metadata",
                    ],
                    batch_size=batch_size,
                    filters=query_filters,
                    user_id=user_id,
                    is_admin=is_admin,
                ),
            ):
                batch_df = batch.to_pandas()

                # Apply substring filter
                text_mask = (
                    batch_df["text"]
                    .astype(str)
                    .str.contains(query_text, na=False, regex=False)
                )
                matching_rows = batch_df[text_mask]

                # Early exit: stop processing if we already have enough results
                if len(results) >= top_k:
                    break

                for _, row in matching_rows.iterrows():
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

                    # Early exit: stop as soon as we have enough results
                    if len(results) >= top_k:
                        break

                if len(results) >= top_k:
                    break

            if results:
                current_warnings.append(
                    SearchWarning(
                        code="FTS_FALLBACK",
                        message=(
                            "Full-text index returned no matches; used async substring search fallback. "
                            "Check FTS tokenizer configuration or update LanceDB to ensure proper tokenisation for query language."
                        ),
                        fallback_action=SearchFallbackAction.BRUTE_FORCE,
                        affected_models=[model_tag],
                    )
                )

        except Exception as exc:
            logger.error("Async substring fallback failed: %s", exc)
        finally:
            _safe_close_table(_table)

        return results

    def search_sparse(
        self,
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
        """Execute sparse (FTS) search for this collection."""
        if not self.capabilities.supports_search:
            return self._sparse_unsupported(model_tag, query_text)

        collection = self.context.collection
        _fts_enabled = False
        current_warnings: List[SearchWarning] = []

        if readonly:
            current_warnings.append(
                SearchWarning(
                    code="READONLY_MODE",
                    message=f"Readonly mode enabled for sparse search on {model_tag}. No FTS index operations will be performed.",
                    fallback_action=SearchFallbackAction.REBUILD_INDEX,
                    affected_models=[model_tag],
                )
            )

        table = None
        try:
            vector_store = self.vector_index_store

            # Open embeddings table with legacy fallback (handled by abstraction layer)
            # open_embeddings_table will handle adding the "embeddings_" prefix
            table, actual_table_name = vector_store.open_embeddings_table(model_tag)

            # Use storage abstraction for index management
            index_result_obj = vector_store.create_index(model_tag, readonly)

            # Use FTS enabled status from index result
            _fts_enabled = index_result_obj.fts_enabled

            if not _fts_enabled:
                current_warnings.append(
                    SearchWarning(
                        code="FTS_INDEX_MISSING",
                        message=f"FTS index not found on 'text' column for {model_tag}. Sparse search performance may be degraded.",
                        fallback_action=SearchFallbackAction.REBUILD_INDEX,
                        affected_models=[model_tag],
                    )
                )

            fts_query = build_fts_query(query_text)
            search_query = (
                None
                if fts_query is None
                else table.search(fts_query, query_type="fts").limit(top_k)
            )

            # Convert legacy dict format to FilterExpression if needed
            filter_expr: Optional[FilterExpression] = None
            if collection or filters:
                # Build filter conditions
                conditions: List[FilterExpression] = []

                # Add collection filter
                if collection:
                    conditions.append(
                        FilterCondition(
                            field="collection",
                            operator=FilterOperator.EQ,
                            value=collection,
                        )
                    )

                # Add custom filters
                if filters:
                    if isinstance(filters, dict):
                        # Legacy format: use parser
                        parsed_filters = parse_legacy_filters(filters)
                        # parsed_filters can be FilterCondition or tuple (AND combination)
                        if parsed_filters is not None:
                            if isinstance(parsed_filters, tuple):
                                # Type narrowing: tuple of FilterConditions
                                conditions.extend(parsed_filters)
                            else:
                                # Type narrowing: single FilterCondition
                                conditions.append(parsed_filters)
                    elif isinstance(filters, (tuple, list)):
                        # Already FilterExpression
                        conditions.extend(
                            filters if isinstance(filters, tuple) else list(filters)
                        )
                    else:
                        # Single FilterCondition
                        conditions.append(filters)

                # Combine conditions with AND
                if len(conditions) == 1:
                    filter_expr = conditions[0]
                elif len(conditions) > 1:
                    filter_expr = tuple(conditions)

            # Validate filter expression depth to prevent DoS
            if filter_expr is not None:
                validate_filter_depth(filter_expr)

            # Use abstract filter builder to get backend-specific syntax
            if filter_expr:
                backend_filter = vector_store.build_filter_expression(
                    filters=filter_expr,
                    user_id=user_id,
                    is_admin=is_admin,
                )
                if backend_filter and search_query is not None:
                    search_query = search_query.where(backend_filter)

            # LanceDB's search().to_pandas() returns Any due to missing type stubs
            raw_results_df = (
                pd.DataFrame()
                if search_query is None
                else pd.DataFrame(search_query.to_pandas())
            )

            if not raw_results_df.empty:
                search_results: List[SearchResult] = []
                for _, row in raw_results_df.iterrows():
                    # LanceDB FTS returns TF-IDF score (higher is better),
                    # normalize to similarity score (0-1) similar to dense search
                    # Using score/(1+score) formula to convert TF-IDF to normalized similarity
                    raw_score_value = row.get("_score")
                    raw_score = (
                        float(raw_score_value) if pd.notna(raw_score_value) else 0.0
                    )
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

                return self._build_sparse_response(
                    results=search_results,
                    warnings=current_warnings,
                    fts_enabled=_fts_enabled,
                    query_text=query_text,
                )

            logger.warning(
                "FTS lookup returned no rows for query '%s'; falling back to substring match",
                query_text,
            )
            fallback_results = self._substring_fallback(
                table_name=actual_table_name,
                collection=collection,
                query_text=query_text,
                model_tag=model_tag,
                top_k=top_k,
                filters=filters,
                current_warnings=current_warnings,
                user_id=user_id,
                is_admin=is_admin,
            )

            return self._build_sparse_response(
                results=fallback_results,
                warnings=current_warnings,
                fts_enabled=_fts_enabled,
                query_text=query_text,
            )

        except Exception as e:
            logger.error(
                "Sparse search failed for %s with query '%s': %s",
                model_tag,
                query_text,
                e,
            )
            error_warnings = current_warnings + [
                SearchWarning(
                    code="FTS_SEARCH_FAILED",
                    message=f"An unexpected error occurred during sparse search: {str(e)}",
                    fallback_action=SearchFallbackAction.PARTIAL_RESULTS,
                    affected_models=[model_tag],
                )
            ]
            return self._build_sparse_response(
                results=[],
                warnings=error_warnings,
                fts_enabled=_fts_enabled,
                query_text=query_text,
                status="failed",
            )
        finally:
            _safe_close_table(table)

    async def search_sparse_async(
        self,
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
        """Async sparse (FTS) search for this collection."""
        if not self.capabilities.supports_search:
            return self._sparse_unsupported(model_tag, query_text)

        collection = self.context.collection
        vector_store = self.vector_index_store

        _fts_enabled = False
        current_warnings: List[SearchWarning] = []

        if readonly:
            current_warnings.append(
                SearchWarning(
                    code="READONLY_MODE",
                    message=f"Readonly mode enabled for sparse search on {model_tag}. No FTS index operations will be performed.",
                    fallback_action=SearchFallbackAction.REBUILD_INDEX,
                    affected_models=[model_tag],
                )
            )

        try:
            # Check and create FTS index if needed (using storage abstraction layer)
            if not readonly:
                index_result_obj = vector_store.create_index(model_tag, readonly=False)
                _fts_enabled = index_result_obj.fts_enabled

            if not _fts_enabled:
                current_warnings.append(
                    SearchWarning(
                        code="FTS_INDEX_MISSING",
                        message=f"FTS index may not be enabled on 'text' column for {model_tag}. Sparse search performance may be degraded.",
                        fallback_action=SearchFallbackAction.REBUILD_INDEX,
                        affected_models=[model_tag],
                    )
                )

            # Convert API-facing dict filters into abstract FilterExpression
            filter_expr: Optional[FilterExpression] = None
            if collection or filters:
                conditions: List[FilterExpression] = []

                if collection:
                    conditions.append(
                        FilterCondition(
                            field="collection",
                            operator=FilterOperator.EQ,
                            value=collection,
                        )
                    )

                if filters:
                    if isinstance(filters, dict):
                        parsed_filters = parse_legacy_filters(filters)
                        if parsed_filters is not None:
                            if isinstance(parsed_filters, tuple):
                                conditions.extend(parsed_filters)
                            else:
                                conditions.append(parsed_filters)
                    elif isinstance(filters, (tuple, list)):
                        conditions.extend(
                            filters if isinstance(filters, tuple) else list(filters)
                        )
                    else:
                        conditions.append(filters)

                if len(conditions) == 1:
                    filter_expr = conditions[0]
                elif len(conditions) > 1:
                    filter_expr = tuple(conditions)

            # Validate filter expression depth to prevent DoS
            if filter_expr is not None:
                validate_filter_depth(filter_expr)

            # Execute async FTS search using abstraction layer (by model_tag)
            raw_results = await vector_store.search_fts_by_model_async(
                model_tag=model_tag,
                query_text=query_text,
                top_k=top_k,
                filters=filter_expr,
                text_column_name="text",
            )

            if not raw_results:
                logger.warning(
                    "FTS lookup returned no results for query '%s'; falling back to substring match",
                    query_text,
                )
                # Use async iter_batches for fallback
                fallback_results = await self._substring_fallback_async(
                    model_tag=model_tag,
                    collection=collection,
                    query_text=query_text,
                    top_k=top_k,
                    filters=filters,
                    current_warnings=current_warnings,
                    user_id=user_id,
                    is_admin=is_admin,
                )

                return self._build_sparse_response(
                    results=fallback_results,
                    warnings=current_warnings,
                    fts_enabled=_fts_enabled,
                    query_text=query_text,
                )

            # Convert raw results to SearchResult objects
            search_results: List[SearchResult] = []
            for row in raw_results:
                # LanceDB FTS returns TF-IDF score (higher is better)
                raw_score_value = row.get("_score")
                raw_score = (
                    float(raw_score_value) if raw_score_value is not None else 0.0
                )
                # Normalize TF-IDF score to [0, 1) range
                score = raw_score / (1.0 + raw_score)

                # Deserialize metadata
                metadata = deserialize_metadata(row.get("metadata"))

                search_results.append(
                    SearchResult(
                        doc_id=row["doc_id"],
                        chunk_id=row["chunk_id"],
                        text=row["text"],
                        score=score,
                        parse_hash=row.get("parse_hash"),
                        model_tag=model_tag,
                        created_at=row.get("created_at"),
                        metadata=metadata,
                    )
                )

            return self._build_sparse_response(
                results=search_results,
                warnings=current_warnings,
                fts_enabled=_fts_enabled,
                query_text=query_text,
            )

        except Exception as e:
            logger.error(
                "Async sparse search failed for %s with query '%s': %s",
                model_tag,
                query_text,
                e,
            )
            error_warnings = current_warnings + [
                SearchWarning(
                    code="FTS_SEARCH_FAILED",
                    message=f"An unexpected error occurred during sparse search: {str(e)}",
                    fallback_action=SearchFallbackAction.PARTIAL_RESULTS,
                    affected_models=[model_tag],
                )
            ]
            return self._build_sparse_response(
                results=[],
                warnings=error_warnings,
                fts_enabled=_fts_enabled,
                query_text=query_text,
                status="failed",
            )

    def _hybrid_unsupported(
        self, model_tag: str, fusion_config: FusionConfig | None
    ) -> HybridSearchResponse:
        return HybridSearchResponse(
            results=[],
            total_count=0,
            status="failed",
            warnings=[
                SearchWarning(
                    code="SEARCH_NOT_SUPPORTED",
                    message="This backend does not support search.",
                    fallback_action=SearchFallbackAction.PARTIAL_RESULTS,
                    affected_models=[model_tag],
                )
            ],
            fusion_config=fusion_config or FusionConfig(),
            dense_count=0,
            sparse_count=0,
            index_status=IndexStatus.NO_INDEX,
            index_advice=None,
        )

    def search_hybrid(
        self,
        model_tag: str,
        query_text: str,
        query_vector: list[float],
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        fusion_config: FusionConfig | None = None,
        readonly: bool = False,
        nprobes: int | None = None,
        refine_factor: int | None = None,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> HybridSearchResponse:
        """Execute hybrid (dense + sparse) search with fusion for this collection."""
        if not self.capabilities.supports_search:
            return self._hybrid_unsupported(model_tag, fusion_config)

        if fusion_config is None:
            fusion_config = FusionConfig()

        # 1. Execute Dense Search
        logger.info("Executing dense search for model %s...", model_tag)
        dense_response = self.search_dense(
            model_tag,
            query_vector,
            top_k=top_k * 2,
            filters=filters,
            readonly=readonly,
            nprobes=nprobes,
            refine_factor=refine_factor,
            user_id=user_id,
            is_admin=is_admin,
        )

        # 2. Execute Sparse Search
        logger.info("Executing sparse search for model %s...", model_tag)
        sparse_response = self.search_sparse(
            model_tag,
            query_text,
            top_k=top_k * 2,
            filters=filters,
            readonly=readonly,
            user_id=user_id,
            is_admin=is_admin,
        )

        # 3-6. Fuse and build the response (shared sync logic).
        return _fuse_hybrid(
            model_tag,
            query_text,
            dense_response,
            sparse_response,
            top_k=top_k,
            fusion_config=fusion_config,
        )

    async def search_hybrid_async(
        self,
        model_tag: str,
        query_text: str,
        query_vector: list[float],
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        fusion_config: FusionConfig | None = None,
        readonly: bool = False,
        nprobes: int | None = None,
        refine_factor: int | None = None,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> HybridSearchResponse:
        """Async hybrid (dense + sparse) search with fusion for this collection."""
        if not self.capabilities.supports_search:
            return self._hybrid_unsupported(model_tag, fusion_config)

        if fusion_config is None:
            fusion_config = FusionConfig()

        # 1. Execute Dense Search (async)
        logger.info("Executing async dense search for model %s...", model_tag)
        dense_response = await self.search_dense_async(
            model_tag,
            query_vector,
            top_k=top_k * 2,
            filters=filters,
            readonly=readonly,
            nprobes=nprobes,
            refine_factor=refine_factor,
            user_id=user_id,
            is_admin=is_admin,
        )

        # 2. Execute Sparse Search (async)
        logger.info("Executing async sparse search for model %s...", model_tag)
        sparse_response = await self.search_sparse_async(
            model_tag,
            query_text,
            top_k=top_k * 2,
            filters=filters,
            readonly=readonly,
            user_id=user_id,
            is_admin=is_admin,
        )

        # 3-6. Fuse and build the response (shared sync logic).
        return _fuse_hybrid(
            model_tag,
            query_text,
            dense_response,
            sparse_response,
            top_k=top_k,
            fusion_config=fusion_config,
        )

    # --- Parse/chunk cleanup (row only, collection scoped) (#509) ---

    def delete_parse_records(
        self,
        doc_id: str,
        *,
        parse_hash: str | None = None,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> int:
        """Delete parse rows for a document via the bound store (no cascade)."""
        return self.vector_index_store.delete_parse_records(
            collection_name=self.context.collection,
            doc_id=doc_id,
            parse_hash=parse_hash,
            user_id=user_id,
            is_admin=is_admin,
        )

    def delete_chunk_records(
        self,
        doc_id: str,
        *,
        parse_hash: str | None = None,
        config_hash: str | None = None,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> int:
        """Delete chunk rows for a document via the bound store (no cascade)."""
        return self.vector_index_store.delete_chunk_records(
            collection_name=self.context.collection,
            doc_id=doc_id,
            parse_hash=parse_hash,
            config_hash=config_hash,
            user_id=user_id,
            is_admin=is_admin,
        )

    # --- Parse/chunk rollback compensation (methods only; wiring in #514) ---

    def snapshot_parse(
        self,
        doc_id: str,
        parse_hash: str,
        *,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> ParseRecordDetail | None:
        """Capture a parse row before a destructive operation (None if absent)."""
        return self.read_latest_parse_record(
            doc_id, parse_hash=parse_hash, user_id=user_id, is_admin=is_admin
        )

    def restore_parse(self, snapshot: ParseRecordDetail) -> None:
        """Restore a snapshotted parse row, preserving every field.

        Refuses snapshots from another collection so the collection-scoped
        boundary holds even on direct handle reuse.
        """
        if snapshot.collection != self.context.collection:
            raise DocumentValidationError(
                f"Handle bound to collection {self.context.collection!r} "
                f"cannot restore a parse snapshot from {snapshot.collection!r}"
            )
        self.vector_index_store.upsert_parses([snapshot.to_legacy_dict()])

    def delete_created_parse(
        self,
        doc_id: str,
        parse_hash: str,
        *,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> int:
        """Idempotently delete a newly created parse row (compensation)."""
        return self.delete_parse_records(
            doc_id, parse_hash=parse_hash, user_id=user_id, is_admin=is_admin
        )

    def snapshot_chunks(
        self,
        doc_id: str,
        parse_hash: str,
        config_hash: str,
        *,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> ChunkRecordSnapshot | None:
        """Capture all chunk rows for a config (None if none exist)."""
        vector_store = self.vector_index_store
        query_filters = {
            "collection": self.context.collection,
            "doc_id": doc_id,
            "parse_hash": parse_hash,
            "config_hash": config_hash,
        }
        try:
            if (
                vector_store.count_rows_or_zero(
                    "chunks", filters=query_filters, user_id=user_id, is_admin=is_admin
                )
                == 0
            ):
                return None
            rows: list[dict[str, Any]] = []
            for batch in vector_store.iter_batches(
                table_name="chunks",
                filters=query_filters,
                user_id=user_id,
                is_admin=is_admin,
            ):
                rows.extend(batch.to_pylist())
            if not rows:
                return None
            # Preserve original chunk order for a faithful restore.
            rows.sort(key=lambda row: row.get("index") or 0)
            return ChunkRecordSnapshot.from_rows(rows)
        except Exception as e:
            logger.error("Failed to snapshot chunks: %s", e)
            raise DatabaseOperationError(f"Failed to snapshot chunks: {e}") from e

    def restore_chunks(self, snapshot: ChunkRecordSnapshot) -> None:
        """Restore snapshotted chunk rows, preserving every field.

        Refuses snapshots whose rows belong to another collection so the
        collection-scoped boundary holds even on direct handle reuse.
        """
        rows = snapshot.to_legacy_dicts()
        if not rows:
            return
        for chunk in snapshot.chunks:
            if chunk.collection != self.context.collection:
                raise DocumentValidationError(
                    f"Handle bound to collection {self.context.collection!r} "
                    f"cannot restore a chunk snapshot from {chunk.collection!r}"
                )
        self.vector_index_store.upsert_chunks(rows)

    def delete_created_chunks(
        self,
        doc_id: str,
        parse_hash: str,
        config_hash: str,
        *,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> int:
        """Idempotently delete newly created chunk rows (compensation)."""
        return self.delete_chunk_records(
            doc_id,
            parse_hash=parse_hash,
            config_hash=config_hash,
            user_id=user_id,
            is_admin=is_admin,
        )

    # --- Version cascade cleanup (Task 5) ---

    def cleanup_cascade(
        self,
        doc_id: str,
        scope: str,
        *,
        new_parse_hash: Optional[str] = None,
        old_parse_hash: Optional[str] = None,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: Optional[bool] = None,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> Dict[str, int]:
        """Policy layer: is_admin=None → True, then delegate to store."""
        from ..utils.user_scope import resolve_user_scope

        if is_admin is None:
            is_admin = True
        user_scope = resolve_user_scope(user_id=user_id, is_admin=is_admin)
        return self.vector_index_store.cleanup_cascade_by_scope(
            collection=self.context.collection,
            doc_id=doc_id,
            scope=scope,
            new_parse_hash=new_parse_hash,
            old_parse_hash=old_parse_hash,
            model_tag=model_tag,
            user_id=user_scope.user_id,
            is_admin=user_scope.is_admin,
            preview_only=preview_only,
            confirm=confirm,
        )

    def cleanup_document_cascade(
        self,
        doc_id: str,
        *,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = True,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> Dict[str, int]:
        from ..core.exceptions import CascadeCleanupError

        try:
            return self.cleanup_cascade(
                doc_id,
                "document",
                model_tag=model_tag,
                user_id=user_id,
                is_admin=is_admin,
                preview_only=preview_only,
                confirm=confirm,
            )
        except Exception as e:
            raise CascadeCleanupError(f"Failed to cleanup document cascade: {e}") from e

    def cleanup_parse_cascade(
        self,
        doc_id: str,
        *,
        old_parse_hash: Optional[str] = None,
        new_parse_hash: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = True,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> Dict[str, int]:
        from ..core.exceptions import CascadeCleanupError

        try:
            return self.cleanup_cascade(
                doc_id,
                "parse",
                old_parse_hash=old_parse_hash,
                new_parse_hash=new_parse_hash,
                user_id=user_id,
                is_admin=is_admin,
                preview_only=preview_only,
                confirm=confirm,
            )
        except Exception as e:
            raise CascadeCleanupError(f"Failed to cleanup parse cascade: {e}") from e

    def cleanup_chunk_cascade(
        self,
        doc_id: str,
        *,
        old_parse_hash: Optional[str] = None,
        new_parse_hash: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = True,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> Dict[str, int]:
        from ..core.exceptions import CascadeCleanupError

        try:
            return self.cleanup_cascade(
                doc_id,
                "chunk",
                old_parse_hash=old_parse_hash,
                new_parse_hash=new_parse_hash,
                user_id=user_id,
                is_admin=is_admin,
                preview_only=preview_only,
                confirm=confirm,
            )
        except Exception as e:
            raise CascadeCleanupError(f"Failed to cleanup chunk cascade: {e}") from e

    def cleanup_embed_cascade(
        self,
        doc_id: str,
        *,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = True,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> Dict[str, int]:
        from ..core.exceptions import CascadeCleanupError

        try:
            return self.cleanup_cascade(
                doc_id,
                "embeddings",
                model_tag=model_tag,
                user_id=user_id,
                is_admin=is_admin,
                preview_only=preview_only,
                confirm=confirm,
            )
        except Exception as e:
            raise CascadeCleanupError(f"Failed to cleanup embed cascade: {e}") from e

    # --- Ingestion-status data-plane (#513) ---

    def write_ingestion_status(
        self,
        doc_id: str,
        *,
        status: str,
        message: str | None = None,
        parse_hash: str | None = None,
        user_id: int | None = None,
    ) -> None:
        """Write ingestion status for a document in this collection (sync)."""
        self.ingestion_status_store.write_ingestion_status(
            collection=self.context.collection,
            doc_id=doc_id,
            status=status,
            message=message,
            parse_hash=parse_hash,
            user_id=user_id,
        )

    def load_ingestion_status(
        self,
        *,
        doc_id: str | None = None,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> List[Dict[str, Any]]:
        """Load ingestion status rows for this collection (sync)."""
        return self.ingestion_status_store.load_ingestion_status(
            collection=self.context.collection,
            doc_id=doc_id,
            user_id=user_id,
            is_admin=is_admin,
        )

    def clear_ingestion_status(
        self,
        doc_id: str,
        *,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> None:
        """Remove ingestion status row for a document in this collection (sync)."""
        self.ingestion_status_store.clear_ingestion_status(
            collection=self.context.collection,
            doc_id=doc_id,
            user_id=user_id,
            is_admin=is_admin,
        )

    # --- Collection-level cascade delete (#H05) ---

    def delete_collection_data(
        self,
        *,
        user_id: int | None,
        is_admin: bool,
        warnings_out: list[str] | None = None,
    ) -> dict[str, int]:
        """Delete all data for this collection from vector-side tables.

        Delegates to the bound vector index store's ``delete_collection_data``,
        passing ``self.context.collection`` so the handle boundary is respected.
        Subsequent reads will not observe stale cached table handles because the
        store invalidates its table cache internally.
        """
        return self.vector_index_store.delete_collection_data(
            collection_name=self.context.collection,
            user_id=user_id,
            is_admin=is_admin,
            warnings_out=warnings_out,
        )

    def delete_documents_data(
        self,
        doc_ids: list[str],
        *,
        user_id: int | None,
        is_admin: bool,
        warnings_out: list[str] | None = None,
    ) -> dict[str, int]:
        """Delete vector-side data for specific document IDs in this collection.

        Delegates to the bound vector index store's ``delete_documents_data``.
        On partial failure the store raises ``DatabaseOperationError`` with
        ``details={"deleted_counts": ..., "deleted_doc_ids": ..., "failed_batch_index": ...}``
        which is the downstream contract for ``CollectionOperationResult.partial_success``.
        That exception propagates unchanged so callers receive the exact details dict.
        """
        return self.vector_index_store.delete_documents_data(
            collection_name=self.context.collection,
            doc_ids=doc_ids,
            user_id=user_id,
            is_admin=is_admin,
            warnings_out=warnings_out,
        )

    # --- Collection-level rename primitives (#H05 Phase 2) ---

    def rename_collection_data(
        self,
        new_name: str,
        user_id: int | None,
        is_admin: bool,
        warnings_out: list[str] | None = None,
    ) -> list[str]:
        """Rename the collection field across all vector-side data tables.

        Delegates to the bound vector index store's ``rename_collection_data``,
        passing ``self.context.collection`` as the old name.  The coordinator
        should call this via ``asyncio.to_thread`` when running in an async
        context.

        The table cache is invalidated after the rename so that subsequent
        ``count_rows`` / ``iter_batches`` calls see the updated rows (matching
        the behaviour of ``delete_collection_data``).

        Returns:
            List of per-table warning messages (empty on full success).
        """
        store = self.vector_index_store
        warnings = store.rename_collection_data(
            collection_name=self.context.collection,
            new_name=new_name,
            user_id=user_id,
            is_admin=is_admin,
        )
        # Invalidate the table cache so subsequent reads observe the renamed rows.
        if hasattr(store, "invalidate_table_cache"):
            store.invalidate_table_cache()
        if warnings_out is not None:
            warnings_out.extend(warnings)
        return warnings

    def rename_collection_status(
        self,
        new_name: str,
        user_id: int | None,
        is_admin: bool,
    ) -> list[str]:
        """Rename ingestion status rows from this collection to ``new_name``.

        Best-effort: never raises; returns a list of warning strings on
        partial failures.
        """
        try:
            return self.ingestion_status_store.rename_collection_status(
                old_name=self.context.collection,
                new_name=new_name,
                user_id=user_id,
                is_admin=is_admin,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "rename_collection_status(%r -> %r) failed: %s",
                self.context.collection,
                new_name,
                exc,
            )
            return [str(exc)]

    async def rename_collection_status_async(
        self,
        new_name: str,
        user_id: int | None,
        is_admin: bool,
    ) -> List[str]:
        """Rename ingestion status rows from this collection to ``new_name`` (async).

        Best-effort: never raises; returns a list of warning strings on
        partial failures.
        """
        try:
            return await self.ingestion_status_store.rename_collection_status_async(
                old_name=self.context.collection,
                new_name=new_name,
                user_id=user_id,
                is_admin=is_admin,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "rename_collection_status_async(%r -> %r) failed: %s",
                self.context.collection,
                new_name,
                exc,
            )
            return [str(exc)]

    async def rename_collection_metadata(
        self,
        new_name: str,
        user_id: int | None,
        is_admin: bool,
    ) -> None:
        """Rename control-plane metadata rows to ``new_name``.

        It wraps ``await metadata_store.rename_collection(...)`` which updates the
        ``collection_config`` and ``collection_metadata`` rows.  The coordinator
        calls this directly with ``await`` (no ``asyncio.to_thread`` wrapper
        needed, unlike the two sync rename primitives above).

        Args:
            new_name: Target collection name.
            user_id: User ID for tenant-scoped rename.
            is_admin: When ``True`` renames across all tenants.
        """
        await self.metadata_store.rename_collection(
            old_name=self.context.collection,
            new_name=new_name,
            user_id=user_id,
            is_admin=is_admin,
        )

    # --- Collection-level statistics (#H05 Phase 3) ---

    def collection_stats(self, user_id: int | None, is_admin: bool) -> dict[str, int]:
        """Return aggregate statistics for this collection.

        Counts document rows, chunk rows, and all embedding rows (summed across
        all ``embeddings_*`` tables) that are visible to the caller under the
        given user/admin scope.

        Returns:
            A ``dict`` with keys ``"documents"``, ``"chunks"``, and
            ``"embeddings"``.
        """
        collection = self.context.collection
        store = self.vector_index_store

        documents = store.count_rows_or_zero(
            "documents",
            filters={"collection": collection},
            user_id=user_id,
            is_admin=is_admin,
        )
        chunks = store.count_rows_or_zero(
            "chunks",
            filters={"collection": collection},
            user_id=user_id,
            is_admin=is_admin,
        )
        embeddings = sum(
            store.count_rows_or_zero(
                table_name,
                filters={"collection": collection},
                user_id=user_id,
                is_admin=is_admin,
            )
            for table_name in store.list_table_names()
            if table_name.startswith("embeddings_")
        )
        return {
            "documents": documents,
            "chunks": chunks,
            "embeddings": embeddings,
        }

    def count_rows_by_document(
        self,
        *,
        user_id: int | None,
        is_admin: bool,
        doc_id: str | None = None,
    ) -> dict[str, dict[str, int]]:
        """Count chunk and embedding rows per document from the LanceDB tables.

        Tables are opened uncached so rows written by other processes are
        counted; a single document is counted without reading its rows.
        """
        store = self.vector_index_store
        collection = self.context.collection
        table_names = ["chunks"] + [
            name for name in store.list_table_names() if name.startswith("embeddings_")
        ]
        doc_filter = None
        if doc_id is not None:
            doc_filter = store.build_filter_expression(
                build_filter_from_dict({"collection": collection, "doc_id": doc_id}),
                user_id=user_id,
                is_admin=is_admin,
            )
        counts: dict[str, dict[str, int]] = {}
        for table_name in table_names:
            if doc_id is None:
                per_document = store.aggregate_document_counts(
                    table_name=table_name,
                    doc_id_column="doc_id",
                    collection_name=collection,
                    user_id=user_id,
                    is_admin=is_admin,
                )
            else:
                per_document = {doc_id: self._count_uncached(table_name, doc_filter)}
            for row_doc_id, count in per_document.items():
                if count:
                    counts.setdefault(row_doc_id, {})[table_name] = count
        return counts

    def _count_uncached(self, table_name: str, filter_expr: str | None) -> int:
        table = None
        try:
            table = self.vector_index_store.get_raw_connection().open_table(table_name)
            return _safe_count_rows(table, filter_expr)
        except Exception:  # noqa: BLE001 - a missing table holds no rows
            return 0
        finally:
            _safe_close_table(table)

    def count_documents(self, user_id: int | None, is_admin: bool) -> int:
        """Count documents visible to the given user in this collection.

        When ``is_admin`` is ``True`` all rows are counted regardless of
        ``user_id``.  Otherwise only rows owned by ``user_id`` are counted.
        """
        return self.vector_index_store.count_rows_or_zero(
            "documents",
            filters={"collection": self.context.collection},
            user_id=user_id,
            is_admin=is_admin,
        )

    def list_collection_documents(
        self,
        user_id: int | None,
        is_admin: bool,
        max_results: int = 1_000_000,
    ) -> list[str]:
        """List document IDs for this collection.

        Delegates to the bound vector index store's ``list_document_records`` and
        returns a sorted list of unique doc_id strings.  The coordinator uses this
        before deletion to collect tenant-owned doc_ids (when the caller has not
        pre-computed them) and to populate ``affected_documents`` in the result.
        """
        store = self.vector_index_store
        records = store.list_document_records(
            collection_name=self.context.collection,
            user_id=user_id,
            is_admin=is_admin,
            max_results=max_results,
        )
        return sorted({r.doc_id for r in records})

    # --- Collection-level rollback config primitives (#H05 Phase 4) ---

    async def delete_collection_config(self, *, tenant_only: bool = False) -> int:
        """Delete the collection_config row(s) for this collection (idempotent).

        When ``tenant_only`` is ``False`` (default) delegates with
        ``is_admin=True`` so all tenant rows for this collection are removed.
        When ``tenant_only`` is ``True`` uses the handle's bound user scope so
        only that tenant's config row is removed, leaving other tenants' rows
        intact.  Returns the number of config rows deleted (0 when none
        existed).

        ``delete_orphaned_metadata=True`` lets the metadata store drop the
        collection_metadata record once removing this tenant's config row
        leaves the collection with zero config rows (a true orphan).  This is
        scope-safe: it only ever deletes the current tenant's own config row
        and the shared metadata record when nothing remains — it never touches
        another tenant's config row.
        """
        if tenant_only:
            user_id = self.context.user_scope.user_id
            is_admin = self.context.user_scope.is_admin
        else:
            user_id = None
            is_admin = True
        result = await self.metadata_store.delete_collection_metadata(
            collection_name=self.context.collection,
            user_id=user_id,
            is_admin=is_admin,
            delete_orphaned_metadata=True,
        )
        return result.get("config_rows", 0)

    def cleanup_collection_data_after_rollback(
        self,
        *,
        user_id: int | None,
        is_admin: bool,
    ) -> dict[str, int]:
        """Remove all vector-side data for this collection (rollback compensation).

        Composes the Phase 1 :meth:`delete_collection_data` primitive.  Does
        **not** access the filesystem; physical file cleanup is the caller's
        responsibility.

        Returns a ``dict[str, int]`` mapping table names to deleted row counts.
        """
        return self.delete_collection_data(user_id=user_id, is_admin=is_admin)

    def cleanup_embeddings_for_operation(
        self,
        *,
        doc_id: Optional[str] = None,
        parse_hash: Optional[str] = None,
        chunk_ids: Optional[Sequence[str]] = None,
        model_tag: Optional[str] = None,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> "KBVectorStorageCleanupResult":
        """Delete or preview embedding rows for a failed operation (#515).

        Ports the former facade ``_cleanup_vectors_for_operation_impl``: builds
        per-table predicates via ``storage/lancedb_cleanup_filters``,
        counts/deletes through this handle's raw store connection, and derives
        status / side_effects_may_remain exactly as before.
        """
        from ..storage.lancedb_cleanup_filters import (
            build_embedding_cleanup_filters,
        )
        from ..utils.lancedb_query_utils import _safe_count_rows
        from .cleanup_filters import KBCleanupScope
        from .models import KBVectorStorageCleanupResult

        scope = KBCleanupScope(
            collection=self.context.collection,
            user_id=self.context.user_scope.user_id,
            is_admin=self.context.user_scope.is_admin,
            doc_id=doc_id,
            parse_hash=parse_hash,
            chunk_ids=tuple(chunk_ids or ()),
            model_tag=model_tag,
        )
        conn = self.vector_index_store.get_raw_connection()

        table_counts: dict[str, int] = {}
        warnings: list[str] = []
        side_effects_may_remain = False
        should_delete = bool(confirm and not preview_only)
        table_filters = build_embedding_cleanup_filters(conn, scope)

        if not table_filters:
            return KBVectorStorageCleanupResult(
                collection=scope.collection,
                status="skipped",
                deleted_count=0,
                table_counts={},
                model_tag=scope.model_tag,
                preview_only=preview_only,
            )

        for table_name, filter_exprs in table_filters.items():
            table = None
            try:
                table = conn.open_table(table_name)
                count = 0
                for filter_expr in filter_exprs:
                    matched = _safe_count_rows(table, filter_expr, on_error="raise")
                    if should_delete and matched > 0:
                        table.delete(filter_expr)
                    count += matched
                table_counts[table_name] = count
            except Exception as exc:  # noqa: BLE001 - report rollback cleanup state
                side_effects_may_remain = True
                message = f"{table_name}: {exc}"
                warnings.append(message)
                logger.warning("Vector cleanup failed for %s: %s", table_name, exc)
            finally:
                _safe_close_table(table)

        if should_delete:
            # Per-table failures are caught above, so some tables may be deleted.
            invalidate = getattr(
                self.vector_index_store, "invalidate_table_cache", None
            )
            if callable(invalidate):
                invalidate()

        deleted_count = sum(table_counts.values())
        if warnings:
            status = "incomplete"
        elif should_delete:
            status = "complete"
        else:
            status = "planned"

        return KBVectorStorageCleanupResult(
            collection=scope.collection,
            status=status,
            deleted_count=deleted_count,
            table_counts=table_counts,
            model_tag=scope.model_tag,
            preview_only=preview_only,
            warnings=tuple(warnings),
            side_effects_may_remain=side_effects_may_remain,
        )

    # --- Rollback snapshot/restore/clear primitives (#513 Task 7) ---

    def capture_status_snapshot(
        self,
        doc_id: str,
        *,
        user_id: int | None = None,
        is_admin: bool = True,
    ) -> List[Dict[str, Any]]:
        """Capture the current ingestion-status rows for ``doc_id`` (snapshot).

        Returns the list of status rows as returned by ``load_ingestion_status``.
        An empty list means no status row exists (snapshot of absence).
        """
        return self.load_ingestion_status(
            doc_id=doc_id, user_id=user_id, is_admin=is_admin
        )

    def restore_status_snapshot(
        self,
        doc_id: str,
        snapshot: List[Dict[str, Any]],
        *,
        user_id: int | None = None,
    ) -> None:
        """Restore an ingestion-status snapshot captured by ``capture_status_snapshot``.

        If ``snapshot`` is empty the status row is cleared (restoring absence).
        If ``snapshot`` contains a row it is re-written (restoring the prior state).
        """
        if not snapshot:
            self.clear_ingestion_status(doc_id, user_id=user_id, is_admin=True)
            return
        for row in snapshot:
            self.write_ingestion_status(
                doc_id,
                status=row.get("status", ""),
                message=row.get("message"),
                parse_hash=row.get("parse_hash"),
                user_id=row.get("user_id"),
            )

    def clear_status_snapshot(
        self,
        doc_id: str,
        *,
        user_id: int | None = None,
        is_admin: bool = True,
    ) -> None:
        """Clear the ingestion-status row for ``doc_id`` (post-rollback cleanup).

        Thin wrapper over ``clear_ingestion_status`` for the rollback path.
        """
        self.clear_ingestion_status(doc_id, user_id=user_id, is_admin=is_admin)

    def capture_main_pointer_snapshot(
        self,
        doc_id: str,
        step_type: str,
        model_tag: str | None = None,
    ) -> "KBMainPointerSnapshot":
        """Capture the current main-pointer row as a snapshot before mutation.

        Returns a :class:`KBMainPointerSnapshot`; ``pointer`` is ``None`` when
        no pointer exists (snapshot of absence).
        """
        return KBMainPointerSnapshot(
            collection=self.context.collection,
            doc_id=doc_id,
            step_type=step_type,
            model_tag=model_tag,
            pointer=self.get_main_pointer(doc_id, step_type, model_tag),
        )

    def restore_main_pointer_snapshot(
        self,
        snapshot: "KBMainPointerSnapshot",
        *,
        operator: str | None = None,
    ) -> bool:
        """Restore the main pointer to the state recorded in ``snapshot``.

        If ``snapshot.pointer`` is ``None`` the pointer is deleted (restoring
        absence).  Returns ``True`` on success, ``False`` when the snapshot
        pointer is incomplete (missing ``semantic_id`` or ``technical_id``).
        """
        if snapshot.pointer is None:
            self.delete_main_pointer(
                snapshot.doc_id, snapshot.step_type, snapshot.model_tag
            )
            return True

        semantic_id = snapshot.pointer.get("semantic_id")
        technical_id = snapshot.pointer.get("technical_id")
        if not semantic_id or not technical_id:
            logger.warning(
                "Failed to restore main pointer snapshot for %s/%s/%s: "
                "missing semantic_id or technical_id",
                snapshot.collection,
                snapshot.doc_id,
                snapshot.step_type,
            )
            return False

        self.set_main_pointer(
            snapshot.doc_id,
            snapshot.step_type,
            semantic_id,
            technical_id,
            model_tag=snapshot.model_tag,
            operator=operator,
        )
        return True

    def capture_candidate_cleanup_snapshot(
        self,
        doc_id: str,
        scope: str,
        *,
        new_parse_hash: str | None = None,
        old_parse_hash: str | None = None,
        model_tag: str | None = None,
        user_id: int | None = None,
        is_admin: bool | None = None,
    ) -> "KBVersionCandidateCleanupSnapshot":
        """Capture a preview of what candidate cleanup would delete (preview_only=True).

        The preview counts are stored in the snapshot; no rows are actually
        deleted.  The snapshot can later be passed to
        ``restore_candidate_cleanup_snapshot`` to determine rollback feasibility.
        """
        cleanup_counts = self.cleanup_cascade(
            doc_id,
            scope,
            new_parse_hash=new_parse_hash,
            old_parse_hash=old_parse_hash,
            model_tag=model_tag,
            user_id=user_id,
            is_admin=is_admin,
            preview_only=True,
            confirm=False,
        )
        return KBVersionCandidateCleanupSnapshot(
            collection=self.context.collection,
            doc_id=doc_id,
            scope=scope,
            cleanup_counts=cleanup_counts,
            new_parse_hash=new_parse_hash,
            old_parse_hash=old_parse_hash,
            model_tag=model_tag,
            user_id=user_id,
            is_admin=is_admin,
        )

    async def write_ingestion_status_async(
        self,
        doc_id: str,
        *,
        status: str,
        message: str | None = None,
        parse_hash: str | None = None,
        user_id: int | None = None,
    ) -> None:
        """Write ingestion status for a document in this collection (async)."""
        await self.ingestion_status_store.write_ingestion_status_async(
            collection=self.context.collection,
            doc_id=doc_id,
            status=status,
            message=message,
            parse_hash=parse_hash,
            user_id=user_id,
        )

    async def load_ingestion_status_async(
        self,
        *,
        doc_id: str | None = None,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> List[Dict[str, Any]]:
        """Load ingestion status rows for this collection (async)."""
        return await self.ingestion_status_store.load_ingestion_status_async(
            collection=self.context.collection,
            doc_id=doc_id,
            user_id=user_id,
            is_admin=is_admin,
        )

    async def clear_ingestion_status_async(
        self,
        doc_id: str,
        *,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> None:
        """Remove ingestion status row for a document in this collection (async)."""
        await self.ingestion_status_store.clear_ingestion_status_async(
            collection=self.context.collection,
            doc_id=doc_id,
            user_id=user_id,
            is_admin=is_admin,
        )

    # --- Main-pointer data-plane (#513) ---

    def get_main_pointer(
        self,
        doc_id: str,
        step_type: str,
        model_tag: str | None = None,
    ) -> Optional[Dict[str, Any]]:
        """Get the main pointer for a document stage in this collection (sync)."""
        try:
            return self.main_pointer_store.get_main_pointer(
                collection=self.context.collection,
                doc_id=doc_id,
                step_type=step_type,
                model_tag=model_tag,
                user_id=None,
            )
        except Exception as e:
            raise MainPointerError(f"Failed to get main pointer: {e}") from e

    def set_main_pointer(
        self,
        doc_id: str,
        step_type: str,
        semantic_id: str,
        technical_id: str,
        model_tag: str | None = None,
        operator: str | None = None,
    ) -> None:
        """Set or update the main pointer for a document stage in this collection (sync)."""
        try:
            self.main_pointer_store.set_main_pointer(
                collection=self.context.collection,
                doc_id=doc_id,
                step_type=step_type,
                semantic_id=semantic_id,
                technical_id=technical_id,
                model_tag=model_tag,
                operator=operator,
                user_id=None,
            )
        except Exception as e:
            raise MainPointerError(f"Failed to set main pointer: {e}") from e

    def list_main_pointers(
        self,
        doc_id: str | None = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List main pointers for this collection (sync)."""
        try:
            return self.main_pointer_store.list_main_pointers(
                collection=self.context.collection,
                doc_id=doc_id,
                user_id=None,
                limit=limit,
            )
        except Exception as e:
            raise MainPointerError(f"Failed to list main pointers: {e}") from e

    def delete_main_pointer(
        self,
        doc_id: str,
        step_type: str,
        model_tag: str | None = None,
    ) -> bool:
        """Delete the main pointer for a document stage in this collection (sync)."""
        try:
            return self.main_pointer_store.delete_main_pointer(
                collection=self.context.collection,
                doc_id=doc_id,
                step_type=step_type,
                model_tag=model_tag,
                user_id=None,
            )
        except Exception as e:
            raise MainPointerError(f"Failed to delete main pointer: {e}") from e

    async def get_main_pointer_async(
        self,
        doc_id: str,
        step_type: str,
        model_tag: str | None = None,
    ) -> Optional[Dict[str, Any]]:
        """Get the main pointer for a document stage in this collection (async)."""
        try:
            return await self.main_pointer_store.get_main_pointer_async(
                collection=self.context.collection,
                doc_id=doc_id,
                step_type=step_type,
                model_tag=model_tag,
                user_id=None,
            )
        except Exception as e:
            raise MainPointerError(f"Failed to get main pointer: {e}") from e

    async def set_main_pointer_async(
        self,
        doc_id: str,
        step_type: str,
        semantic_id: str,
        technical_id: str,
        model_tag: str | None = None,
        operator: str | None = None,
    ) -> None:
        """Set or update the main pointer for a document stage in this collection (async)."""
        try:
            await self.main_pointer_store.set_main_pointer_async(
                collection=self.context.collection,
                doc_id=doc_id,
                step_type=step_type,
                semantic_id=semantic_id,
                technical_id=technical_id,
                model_tag=model_tag,
                operator=operator,
                user_id=None,
            )
        except Exception as e:
            raise MainPointerError(f"Failed to set main pointer: {e}") from e

    async def list_main_pointers_async(
        self,
        doc_id: str | None = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List main pointers for this collection (async)."""
        try:
            return await self.main_pointer_store.list_main_pointers_async(
                collection=self.context.collection,
                doc_id=doc_id,
                user_id=None,
                limit=limit,
            )
        except Exception as e:
            raise MainPointerError(f"Failed to list main pointers: {e}") from e

    async def delete_main_pointer_async(
        self,
        doc_id: str,
        step_type: str,
        model_tag: str | None = None,
    ) -> bool:
        """Delete the main pointer for a document stage in this collection (async)."""
        try:
            return await self.main_pointer_store.delete_main_pointer_async(
                collection=self.context.collection,
                doc_id=doc_id,
                step_type=step_type,
                model_tag=model_tag,
                user_id=None,
            )
        except Exception as e:
            raise MainPointerError(f"Failed to delete main pointer: {e}") from e

    # --- Version candidate listing (#513 Task 4) ---

    def list_candidates(
        self,
        doc_id: str,
        step_type: "Any",
        model_tag: Optional[str] = None,
        state: Optional[str] = None,
        limit: int = 50,
        order_by: str = "created_at desc",
    ) -> Dict[str, Any]:
        """List version candidates for a document/stage within this collection.

        Resolves ``step_type``, calls the vector index store's semantic method,
        then applies state-filter / total-count capture / sort / limit /
        result-dict assembly (the orchestration from ``_list_candidates_impl``).

        Args:
            doc_id: Document ID.
            step_type: Processing stage as :class:`StepType` enum or string.
            model_tag: Required for ``embed`` step.
            state: Optional state filter (``"candidate"``, ``"main"``, etc.).
            limit: Maximum candidates to return (default 50).
            order_by: Sort order (``"created_at desc"`` or ``"created_at asc"``).

        Returns:
            Result dict with keys ``candidates``, ``total_count``,
            ``returned_count``, ``step_type``, ``model_tag``, ``filters``.
        """
        from ..core.exceptions import DatabaseOperationError, VersionManagementError
        from ..core.schemas import StepType as _StepType

        def _resolve(st: Any) -> "_StepType":
            if isinstance(st, _StepType):
                return st
            elif isinstance(st, str):
                try:
                    return _StepType(st)
                except ValueError:
                    raise VersionManagementError(
                        f"Invalid step_type string: '{st}'. Expected one of: "
                        + ", ".join(["'" + s.value + "'" for s in _StepType])
                    )
            else:
                raise VersionManagementError(
                    f"Unsupported step_type type: {type(st)}. Expected StepType or str."
                )

        try:
            resolved = _resolve(step_type)

            candidates = self.vector_index_store.list_version_candidate_rows(
                self.context.collection,
                doc_id,
                resolved.value,
                model_tag,
            )

            # Apply state filter if specified
            if state is not None:
                candidates = [c for c in candidates if c["state"] == state]

            # Record total count before limit (after state filter, matching _list_candidates_impl)
            total_count = len(candidates)

            # Sort by order_by (must happen before limit)
            if order_by == "created_at desc":
                candidates.sort(key=lambda x: x["created_at"], reverse=True)
            elif order_by == "created_at asc":
                candidates.sort(key=lambda x: x["created_at"], reverse=False)

            # Apply limit after sorting
            if limit > 0:
                candidates = candidates[:limit]

            return {
                "candidates": candidates,
                "total_count": total_count,
                "returned_count": len(candidates),
                "step_type": resolved.value,
                "model_tag": model_tag,
                "filters": {"state": state, "limit": limit, "order_by": order_by},
            }

        except (DatabaseOperationError, VersionManagementError):
            raise
        except Exception as e:
            raise VersionManagementError(f"Failed to list candidates: {e}") from e

    # --- Version promotion orchestration (#513 Task 6) ---

    def _call_cleanup_cascade_for_step(
        self,
        doc_id: str,
        step_type: Any,
        technical_id: str,
        old_technical_id: Optional[str] = None,
        model_tag: Optional[str] = None,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> Dict[str, int]:
        """Scope-mapping helper: route step_type → cleanup_cascade scope."""
        from ..core.exceptions import VersionManagementError
        from ..core.schemas import StepType as _StepType

        if step_type == _StepType.PARSE:
            return self.cleanup_cascade(
                doc_id,
                "parse",
                new_parse_hash=technical_id,
                old_parse_hash=old_technical_id,
                preview_only=preview_only,
                confirm=confirm,
            )
        elif step_type == _StepType.CHUNK:
            return self.cleanup_cascade(
                doc_id,
                "chunk",
                new_parse_hash=technical_id,
                old_parse_hash=old_technical_id,
                preview_only=preview_only,
                confirm=confirm,
            )
        elif step_type == _StepType.EMBED:
            if not model_tag:
                raise VersionManagementError("model_tag is required for embed step")
            return self.cleanup_cascade(
                doc_id,
                "embeddings",
                model_tag=model_tag,
                preview_only=preview_only,
                confirm=confirm,
            )
        else:
            step_type_str = (
                step_type.value if isinstance(step_type, _StepType) else str(step_type)
            )
            raise VersionManagementError(f"Invalid step_type: {step_type_str}")

    def _resolve_selected_id_from_candidates(
        self,
        doc_id: str,
        step_type: Any,
        selected_id: str,
        model_tag: Optional[str] = None,
    ) -> tuple:
        """Resolve selected_id → (technical_id, semantic_id) via list_candidates."""
        from ..core.exceptions import VersionManagementError

        try:
            candidates_result = self.list_candidates(
                doc_id,
                step_type,
                model_tag=model_tag,
            )
            candidates = candidates_result.get("candidates", [])

            if not candidates:
                raise VersionManagementError(f"No candidates found for {step_type}")

            for candidate in candidates:
                if candidate["technical_id"] == selected_id:
                    return candidate["technical_id"], candidate["semantic_id"]

            for candidate in candidates:
                if candidate["semantic_id"] == selected_id:
                    return candidate["technical_id"], candidate["semantic_id"]

            available_ids = [c["semantic_id"] for c in candidates]
            raise VersionManagementError(
                f"Selected ID '{selected_id}' not found. Available IDs: {available_ids}"
            )

        except Exception as e:
            if isinstance(e, VersionManagementError):
                raise
            raise VersionManagementError(f"Failed to resolve selected_id: {e}")

    def _calculate_cleanup_plan_for_step(
        self,
        doc_id: str,
        step_type: Any,
        technical_id: str,
        model_tag: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate cleanup plan for a version promotion (preview_only=True)."""
        from ..core.exceptions import VersionManagementError
        from ..core.schemas import StepType as _StepType

        try:
            current_pointer = self.get_main_pointer(doc_id, step_type.value, model_tag)
            old_technical_id = None
            if current_pointer:
                old_technical_id = current_pointer["technical_id"]

            deleted_counts = self._call_cleanup_cascade_for_step(
                doc_id,
                step_type,
                technical_id,
                old_technical_id=old_technical_id,
                model_tag=model_tag,
                preview_only=True,
                confirm=False,
            )

            notes = []
            if step_type == _StepType.PARSE and deleted_counts.get("chunks", 0) > 0:
                notes.append("Requires re-chunk/embed")
            elif (
                step_type == _StepType.CHUNK and deleted_counts.get("embeddings", 0) > 0
            ):
                notes.append("Requires re-embed")

            return {
                "deleted_counts": deleted_counts,
                "notes": notes,
                "current_pointer": current_pointer,
                "new_technical_id": technical_id,
            }

        except Exception as e:
            raise VersionManagementError(f"Failed to calculate cleanup plan: {e}")

    def promote_version_main(
        self,
        doc_id: str,
        step_type: "Any",
        selected_id: str,
        operator: Optional[str] = None,
        preview_only: bool = False,
        confirm: bool = False,
        model_tag: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Promote a candidate version to main for a document stage (pure orchestration).

        Relocates ``_promote_version_main_impl`` body to the handle, calling
        handle primitives (``list_candidates``, ``get_main_pointer``,
        ``cleanup_cascade``, ``set_main_pointer``) instead of module-level fns.
        The ``lancedb_dir`` resolution is dropped (no longer needed by the handle).

        Args:
            doc_id: Document ID.
            step_type: Processing step type (:class:`StepType` or string).
            selected_id: Technical or semantic ID of the candidate to promote.
            operator: Operator name (default: ``$USER`` env var or ``"unknown"``).
            preview_only: If True, only return preview without executing.
            confirm: If True, execute the promotion.
            model_tag: Required for embed step.

        Returns:
            Result dict with keys ``promoted``, ``preview``, ``main_pointer``,
            ``deleted_counts``, ``notes``, and optionally ``message``/``operator``.

        Raises:
            VersionManagementError: Any error during the promotion flow.
        """
        from ..core.exceptions import VersionManagementError
        from ..core.schemas import StepType as _StepType

        def _resolve(st: "Any") -> "_StepType":
            if isinstance(st, _StepType):
                return st
            elif isinstance(st, str):
                try:
                    return _StepType(st)
                except ValueError:
                    raise VersionManagementError(
                        f"Invalid step_type string: '{st}'. Expected one of: "
                        + ", ".join(["'" + s.value + "'" for s in _StepType])
                    )
            else:
                raise VersionManagementError(
                    f"Unsupported step_type type: {type(st)}. Expected StepType or str."
                )

        resolved_step_type = _resolve(step_type)

        # Validate and set operator
        if not operator:
            operator = os.environ.get("USER", "unknown")
        if len(operator) > 32:
            raise VersionManagementError("Operator name too long (max 32 characters)")

        try:
            # Resolve selected_id to technical_id and semantic_id
            technical_id, semantic_id = self._resolve_selected_id_from_candidates(
                doc_id, resolved_step_type, selected_id, model_tag
            )

            # Calculate cleanup plan
            cleanup_plan = self._calculate_cleanup_plan_for_step(
                doc_id, resolved_step_type, technical_id, model_tag
            )

            if preview_only or not confirm:
                message = (
                    "Preview of promotion to be applied."
                    if preview_only
                    else "Set confirm=True to execute the promotion."
                )
                return {
                    "promoted": False,
                    "preview": True,
                    "message": message,
                    "main_pointer": {
                        "step_type": resolved_step_type.value,
                        "semantic_id": semantic_id,
                        "technical_id": technical_id,
                        "model_tag": model_tag,
                    },
                    "deleted_counts": cleanup_plan["deleted_counts"],
                    "notes": cleanup_plan["notes"],
                }

            # Perform cascade cleanup
            old_technical_id = None
            if cleanup_plan["current_pointer"]:
                old_technical_id = cleanup_plan["current_pointer"]["technical_id"]

            deleted_counts = self._call_cleanup_cascade_for_step(
                doc_id,
                resolved_step_type,
                technical_id,
                old_technical_id=old_technical_id,
                model_tag=model_tag,
                preview_only=False,
                confirm=True,
            )

            if not deleted_counts:
                raise VersionManagementError(
                    f"[Promotion] No records deleted for "
                    f"{self.context.collection}/{doc_id}/{resolved_step_type.value}"
                )

            # Update main pointer
            self.set_main_pointer(
                doc_id,
                resolved_step_type.value,
                semantic_id,
                technical_id,
                model_tag,
                operator,
            )

            # Generate notes
            notes = []
            if (
                resolved_step_type == _StepType.PARSE
                and deleted_counts.get("chunks", 0) > 0
            ):
                notes.append("Requires re-chunk/embed")
            elif (
                resolved_step_type == _StepType.CHUNK
                and deleted_counts.get("embeddings", 0) > 0
            ):
                notes.append("Requires re-embed")

            logger.info(
                "Promoted version for %s/%s/%s to %s (operator: %s)",
                self.context.collection,
                doc_id,
                resolved_step_type.value,
                technical_id,
                operator,
            )

            return {
                "promoted": True,
                "preview": False,
                "main_pointer": {
                    "step_type": resolved_step_type.value,
                    "semantic_id": semantic_id,
                    "technical_id": technical_id,
                    "model_tag": model_tag,
                },
                "deleted_counts": deleted_counts,
                "notes": notes,
                "operator": operator,
            }

        except Exception as e:
            if isinstance(e, VersionManagementError):
                raise
            raise VersionManagementError(f"Failed to promote version main: {e}") from e


_Method = TypeVar("_Method", bound=Callable[..., Any])
_Resolved = TypeVar("_Resolved")


def _route(method: _Method, call: Callable[..., Any]) -> _Method:
    """Implement ``method`` with ``call(handle, ...)``, async when ``method`` is."""
    if inspect.iscoroutinefunction(method):

        async def call_async(handle: Any, *args: Any, **kwargs: Any) -> Any:
            return await call(handle, *args, **kwargs)

        routed = call_async
    else:
        routed = call
    routed.__name__ = method.__name__
    routed.__qualname__ = f"MilvusCollectionHandle.{method.__name__}"
    routed.__doc__ = method.__doc__
    return cast(_Method, routed)


def _ledger(method: _Method) -> _Method:
    """Serve ``method`` with the same method of the injected ledger handle."""
    name = method.__name__

    def forward(handle: MilvusCollectionHandle, *args: Any, **kwargs: Any) -> Any:
        return getattr(handle.ledger, name)(*args, **kwargs)

    return _route(method, forward)


def _unsupported(method: _Method, family: str) -> _Method:
    """Refuse ``method``: the Milvus engine does not support ``family``."""
    message = f"The milvus KB engine does not support {family} ({method.__name__})"

    def refuse(handle: MilvusCollectionHandle, *args: Any, **kwargs: Any) -> NoReturn:
        raise ConfigurationError(message)

    return _route(method, refuse)


def _pending(method: _Method) -> _Method:
    """Refuse ``method``, which needs Milvus rows, until it is implemented."""
    message = (
        f"MilvusCollectionHandle.{method.__name__} needs Milvus rows and is not "
        "implemented yet"
    )

    def refuse(handle: MilvusCollectionHandle, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError(message)

    return _route(method, refuse)


_ASYNC_SEARCH = "async search"
_CASCADE = "cascade cleanup"
_VERSIONS = "version candidates and promotion"

MILVUS_COLLECTION_PREFIX = "xagent_kb_"
_MARKABLE = {
    DocumentProcessingStatus.SUCCESS.value,
    DocumentProcessingStatus.PARTIALLY_EMBEDDED.value,
}
_MILVUS_MODEL_PROPERTY = "xagent.model_id"
_MILVUS_TAG_LENGTH = 64
_MILVUS_ID_LENGTH = 512
_MILVUS_TEXT_BYTES = 65_535
_MILVUS_QUERY_BATCH = 10_000
# Milvus rejects a query whose offset plus limit is above this.
_MILVUS_QUERY_WINDOW = 16_384
_MILVUS_FALLBACK_PAGE = 1_000
_MILVUS_NOT_LOADED = 101


def milvus_collection_name(model: str) -> str:
    """Return the Milvus collection of an embedding model id such as ``BAAI/bge-m3``.

    The hash of the id keeps apart ids whose tags coincide, and a tag does not
    recover it: pass the id the vectors were written with.
    """
    model_id = model.strip()
    tag = re.sub(r"[^A-Za-z0-9_]", "_", to_model_tag(model_id))[:_MILVUS_TAG_LENGTH]
    digest = hashlib.sha1(model_id.encode(), usedforsecurity=False).hexdigest()
    return f"{MILVUS_COLLECTION_PREFIX}{tag}_{digest[:8]}"


def ensure_milvus_collection(client: Any, model: str, dimension: int) -> str:
    """Create, index and load the Milvus collection of ``model``; return its name.

    ``kb_id``, ``text`` and ``dense`` are not nullable, so a partial update of a
    missing primary key fails instead of inserting a visible ghost row. The model id
    is recorded as a collection property; an existing collection of another model
    or ``dimension`` raises ``VectorValidationError``.
    """
    model_id = model.strip()
    name = milvus_collection_name(model_id)
    pymilvus = importlib.import_module("pymilvus")
    if not client.has_collection(name):
        types = pymilvus.DataType
        schema = client.create_schema(auto_id=False)
        schema.add_field(
            "chunk_id", types.VARCHAR, is_primary=True, max_length=_MILVUS_ID_LENGTH
        )
        schema.add_field(
            "kb_id", types.VARCHAR, max_length=_MILVUS_ID_LENGTH, is_partition_key=True
        )
        schema.add_field("user_id", types.INT64, nullable=True)
        for field in ("doc_id", "parse_hash", "config_hash"):
            schema.add_field(field, types.VARCHAR, max_length=_MILVUS_ID_LENGTH)
        schema.add_field(
            "text",
            types.VARCHAR,
            max_length=_MILVUS_TEXT_BYTES,
            enable_analyzer=True,
            analyzer_params={"type": "chinese"},
        )
        schema.add_field("sparse", types.SPARSE_FLOAT_VECTOR)
        schema.add_field("dense", types.FLOAT_VECTOR, dim=dimension)
        schema.add_field("visible", types.BOOL)
        schema.add_field("created_at", types.INT64)
        schema.add_field("metadata", types.JSON)
        schema.add_function(
            pymilvus.Function(
                name="text_bm25",
                function_type=pymilvus.FunctionType.BM25,
                input_field_names=["text"],
                output_field_names=["sparse"],
            )
        )
        try:
            client.create_collection(
                name, schema=schema, properties={_MILVUS_MODEL_PROPERTY: model_id}
            )
        except Exception:
            if not client.has_collection(name):
                raise
    description = client.describe_collection(name)
    recorded = (description.get("properties") or {}).get(_MILVUS_MODEL_PROPERTY)
    if recorded != model_id:
        raise VectorValidationError(
            f"Milvus collection {name} holds vectors of model {recorded!r}, "
            f"not {model_id!r}"
        )
    dense = next((f for f in description["fields"] if f["name"] == "dense"), None)
    if dense is None:
        raise VectorValidationError(f"Milvus collection {name} has no dense field")
    if int(dense["params"]["dim"]) != dimension:
        raise VectorValidationError(
            f"Milvus collection {name} holds {dense['params']['dim']}-dimensional "
            f"vectors, but {model_id!r} now produces {dimension}-dimensional ones"
        )
    if client.get_load_state(name)["state"].name == "Loaded":
        return name
    # A creator that stopped before loading leaves the collection unindexed;
    # creating the same indexes and loading again are both idempotent.
    indexes = client.prepare_index_params()
    indexes.add_index(
        "dense",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    indexes.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    for field in ("doc_id", "user_id"):
        indexes.add_index(field, index_type="INVERTED")
    client.create_index(name, indexes)
    client.load_collection(name)
    return name


def _milvus_collections(client: Any) -> list[str]:
    return [
        name
        for name in client.list_collections()
        if name.startswith(MILVUS_COLLECTION_PREFIX)
    ]


@contextmanager
def _unloaded_as_empty(name: str, not_loaded: set[str]) -> Iterator[None]:
    """Skip the body with one warning when ``name`` is not loaded (code 101).

    A new collection is unloaded until its creator loads it; one missing
    collection must not fail the counts of the others. ``not_loaded`` spans one
    call, so callers skip the names in it and the warning is not repeated.
    """
    try:
        yield
    except Exception as error:
        if getattr(error, "code", None) != _MILVUS_NOT_LOADED:
            raise
        not_loaded.add(name)
        logger.warning("Milvus collection %s is not loaded; treated as empty", name)


def _visible_filter(kb_ids: list[str]) -> str:
    # kb_ids are uuid hex strings read from the ledger, safe to inline.
    return f"kb_id in {json.dumps(kb_ids)} and visible == true"


def _count_visible_rows(
    client: Any,
    names: list[str],
    kb_ids: list[str],
    not_loaded: set[str],
    doc_id: str | None = None,
) -> dict[str, int]:
    """Count visible rows of ``kb_ids`` in each of ``names`` that holds some."""
    expr = _visible_filter(kb_ids)
    params: dict[str, Any] = {}
    if doc_id is not None:
        expr += " and doc_id == {doc_id}"
        params["doc_id"] = doc_id
    counts = {}
    for name in names:
        if name in not_loaded:
            continue
        with _unloaded_as_empty(name, not_loaded):
            rows = client.query(
                name, filter=expr, filter_params=params, output_fields=["count(*)"]
            )
            if rows[0]["count(*)"]:
                counts[name] = int(rows[0]["count(*)"])
    return counts


def _visible_rows_by_document(
    client: Any, name: str, kb_ids: list[str], not_loaded: set[str]
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    with _unloaded_as_empty(name, not_loaded):
        iterator = client.query_iterator(
            name,
            batch_size=_MILVUS_QUERY_BATCH,
            filter=_visible_filter(kb_ids),
            output_fields=["doc_id"],
        )
        try:
            while batch := iterator.next():
                counts.update(row["doc_id"] for row in batch)
        finally:
            iterator.close()
        return counts
    return Counter()


def _recorded_model(name: str, description: dict[str, Any]) -> str:
    model = (description.get("properties") or {}).get(_MILVUS_MODEL_PROPERTY)
    if model is None:
        raise VectorValidationError(
            f"Milvus collection {name} records no {_MILVUS_MODEL_PROPERTY}; "
            "it was not created by ensure_milvus_collection"
        )
    return str(model)


def _embeddings_key(client: Any, name: str) -> str:
    """Return the per-document count key of Milvus collection ``name``."""
    model = _recorded_model(name, client.describe_collection(name))
    # LanceDB tags a model id twice on write, so its tables carry the second pass.
    return embeddings_table_name(to_model_tag(model))


def _document_rows(
    client: Any, name: str, kb_id: str, doc_id: str, *, unloaded_as_empty: bool = False
) -> dict[str, bool]:
    """Read ``{chunk_id: visible}`` of one document under one kb_id, with Strong."""
    if not client.has_collection(name):
        return {}
    with _unloaded_as_empty(name, set()) if unloaded_as_empty else nullcontext():
        rows = client.query(
            name,
            filter="kb_id == {kb_id} and doc_id == {doc_id}",
            filter_params={"kb_id": kb_id, "doc_id": doc_id},
            output_fields=["chunk_id", "visible"],
            consistency_level="Strong",
        )
        return {row["chunk_id"]: row["visible"] for row in rows}
    return {}


def _with_loaded(name: str, client: Any, call: Callable[[], _Resolved]) -> _Resolved:
    """Run ``call``; if ``name`` is not loaded, load it as the writer does and retry.

    Deletes and restores must reach every row, so such a collection is not skipped.
    """
    try:
        return call()
    except Exception as error:
        if getattr(error, "code", None) != _MILVUS_NOT_LOADED:
            raise
    description = client.describe_collection(name)
    dense = next(f for f in description["fields"] if f["name"] == "dense")
    model = _recorded_model(name, description)
    ensure_milvus_collection(client, model, int(dense["params"]["dim"]))
    return call()


def _delete_milvus_rows(
    client: Any,
    names: list[str],
    kb_ids: list[str],
    *,
    doc_ids: list[str] | None = None,
    invisible_only: bool = False,
) -> Counter[str]:
    """Delete the rows of ``kb_ids`` from the collections ``names``; count them each.

    A collection that is not loaded is loaded first or the call fails; an
    ``invisible_only`` delete skips it, as those rows are never searchable.
    """
    expr = "kb_id in {kb_ids}"
    params: dict[str, Any] = {"kb_ids": kb_ids}
    if doc_ids is not None:
        expr += " and doc_id in {doc_ids}"
        params["doc_ids"] = doc_ids
    if invisible_only:
        expr += " and visible == false"
    counts: Counter[str] = Counter()
    for name in names:
        call = partial(client.delete, name, filter=expr, filter_params=params)
        try:
            deleted = call() if invisible_only else _with_loaded(name, client, call)
        except Exception as error:
            if invisible_only and getattr(error, "code", None) == _MILVUS_NOT_LOADED:
                logger.warning("Milvus collection %s is not loaded; skipped", name)
                continue
            raise DatabaseOperationError(
                f"Cannot delete from Milvus collection {name}: {error}"
            ) from None
        counts[name] += int(deleted.get("delete_count", 0))
    return counts


def _model_keys(client: Any, names: list[str]) -> dict[str, str]:
    """Resolve each collection's count key before anything is deleted."""
    try:
        return {name: _embeddings_key(client, name) for name in names}
    except Exception as error:
        raise DatabaseOperationError(
            f"Cannot resolve the models of the Milvus collections: {error}"
        ) from None


def _per_model(keys: dict[str, str], deleted: Counter[str]) -> Counter[str]:
    return Counter({keys[name]: count for name, count in deleted.items() if count})


_INGEST_MEMO: ContextVar[dict[tuple[Any, ...], Any] | None] = ContextVar(
    "xagent_kb_ingest_memo", default=None
)


@contextmanager
def ingest_scope() -> Iterator[None]:
    """Let one ingest's write batches share their collection and kb_id lookups.

    Handles are opened per call, so only a scope around the ingest spans the batches.
    """
    token = _INGEST_MEMO.set({})
    try:
        yield
    finally:
        _INGEST_MEMO.reset(token)


def _once_per_ingest(
    key: tuple[Any, ...], resolve: Callable[[], _Resolved]
) -> _Resolved:
    memo = _INGEST_MEMO.get()
    if memo is None:
        return resolve()
    if key not in memo:
        memo[key] = resolve()
    return cast(_Resolved, memo[key])


@dataclass(frozen=True)
class MilvusCollectionHandle(KBCollectionHandle):
    """Milvus-backed collection handle.

    The ledger (documents, parses, chunks, ingestion status, main pointers,
    config and metadata) stays in LanceDB and is served by the injected
    ``ledger`` handle; chunk copies and vectors for search live in Milvus.
    Async search, cascade cleanup, and version candidates and promotion raise
    ``ConfigurationError``. Stats, embedding writes, the commit, search, deletes and
    rollback use Milvus rows; the other methods that need them raise
    ``NotImplementedError`` until implemented.
    ``KBBackendCapabilities.milvus`` turns a family on only when every method of it
    is served here.
    """

    context: KBCollectionContext
    ledger: KBCollectionHandle
    connections: MilvusConnectionManager

    @cached_property
    def client(self) -> Any:
        """Milvus client for ``MILVUS_URI``, created on first use."""
        return self.connections.get_shared_client_from_env()

    def _kb_ids(self, user_id: int | None, is_admin: bool) -> list[str]:
        collection = self.context.collection
        return read_kb_ids(
            self.context.vector_index_store.get_raw_connection(),
            user_id=user_id,
            is_admin=is_admin,
            collection=collection,
        ).get(collection, [])

    def collection_stats(self, user_id: int | None, is_admin: bool) -> dict[str, int]:
        """Count documents and chunks in the ledger and visible rows in Milvus."""
        kb_ids = self._kb_ids(user_id, is_admin)
        embeddings = (
            _count_visible_rows(
                self.client, _milvus_collections(self.client), kb_ids, set()
            )
            if kb_ids
            else {}
        )
        stats = self.ledger.collection_stats(user_id, is_admin)
        return {**stats, "embeddings": sum(embeddings.values())}

    def count_rows_by_document(
        self,
        *,
        user_id: int | None,
        is_admin: bool,
        doc_id: str | None = None,
    ) -> dict[str, dict[str, int]]:
        """Count ledger chunks and visible Milvus rows per document.

        Milvus counts are keyed like the LanceDB tables, from the model id the
        Milvus collection records.
        """
        counts = self.ledger.count_rows_by_document(
            user_id=user_id, is_admin=is_admin, doc_id=doc_id
        )
        kb_ids = self._kb_ids(user_id, is_admin)
        if not kb_ids:
            return counts
        names = _milvus_collections(self.client)
        not_loaded: set[str] = set()
        if doc_id is None:
            per_model = {
                name: _visible_rows_by_document(self.client, name, kb_ids, not_loaded)
                for name in names
            }
        else:
            per_model = {
                name: {doc_id: count}
                for name, count in _count_visible_rows(
                    self.client, names, kb_ids, not_loaded, doc_id
                ).items()
            }
        for name, by_document in per_model.items():
            if not by_document:
                continue
            key = _embeddings_key(self.client, name)
            for row_doc_id, count in by_document.items():
                row = counts.setdefault(row_doc_id, {})
                row[key] = row.get(key, 0) + count
        return counts

    def _kb_id(self, user_id: int | None, *, fresh: bool = False) -> str:
        def resolve() -> str:
            return get_or_create_kb_id(
                self.context.vector_index_store.get_raw_connection(),
                self.context.collection,
                user_id,
            )

        if fresh:
            return resolve()
        return _once_per_ingest(("kb_id", self.context.collection, user_id), resolve)

    def _ledger_chunks(
        self,
        doc_id: str,
        parse_hash: str,
        filters: dict[str, Any] | None,
        user_id: int | None,
        is_admin: bool,
        columns: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Read the document's ledger chunk rows, failing on a short read.

        Ledger reads swallow errors, and a short read would delete the rest from
        Milvus. The count raises and opens the table anew: the cached one can lag.
        """
        store = self.context.vector_index_store
        query = {
            "collection": self.context.collection,
            "doc_id": doc_id,
            "parse_hash": parse_hash,
            **(filters or {}),
        }
        where = store.build_filter_expression(
            build_filter_from_dict(query), user_id=user_id, is_admin=is_admin
        )
        table = None
        try:
            conn = store.get_raw_connection()
            # write_chunks creates the table with the first non-empty chunk set.
            if "chunks" not in list_table_names(conn):
                return []
            table = conn.open_table("chunks")
            counted = _safe_count_rows(table, where, on_error="raise")
        except Exception as error:
            raise DatabaseOperationError(
                f"Cannot count the ledger chunks of {doc_id}: {error}"
            ) from error
        finally:
            _safe_close_table(table)
        if counted == 0:
            return []
        rows = [
            row
            for batch in store.iter_batches(
                table_name="chunks",
                columns=columns,
                filters=query,
                user_id=user_id,
                is_admin=is_admin,
            )
            for row in batch.to_pylist()
        ]
        if len(rows) < counted:
            raise DatabaseOperationError(
                f"Read {len(rows)} of {counted} ledger chunks of {doc_id}; "
                "retry the ingest"
            )
        return rows

    def read_chunks_needing_embedding(
        self,
        doc_id: str,
        parse_hash: str,
        model: str,
        *,
        filters: dict[str, Any] | None = None,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> EmbeddingReadResponse:
        """Return ledger chunks that Milvus holds no row for, visible or not.

        A collection that is not loaded is read as holding none; the write loads it.
        """
        if not self.context.collection or not doc_id or not parse_hash or not model:
            raise DocumentValidationError(
                "Collection, doc_id, parse_hash, and model are required"
            )
        chunks = [
            _chunk_for_embedding(row)
            for row in self._ledger_chunks(
                doc_id, parse_hash, filters, user_id, is_admin
            )
        ]
        held = _document_rows(
            self.client,
            milvus_collection_name(model),
            self._kb_id(user_id),
            doc_id,
            unloaded_as_empty=True,
        )
        pending = [chunk for chunk in chunks if chunk.chunk_id not in held]
        for chunk in pending:
            if len(chunk.text.encode()) > _MILVUS_TEXT_BYTES:
                raise DocumentValidationError(
                    f"Chunk {chunk.chunk_id} is over {_MILVUS_TEXT_BYTES} bytes, "
                    "the Milvus text limit; lower the chunk size"
                )
        return EmbeddingReadResponse(
            chunks=pending, total_count=len(chunks), pending_count=len(pending)
        )

    def write_embeddings(
        self,
        embeddings: list[ChunkEmbeddingData],
        *,
        create_index: bool = True,
        user_id: int | None = None,
    ) -> EmbeddingWriteResponse:
        """Upsert the vectors as invisible rows; ``commit_embeddings`` shows them."""
        by_model: dict[str, list[ChunkEmbeddingData]] = {}
        for embedding in embeddings:
            by_model.setdefault(embedding.model, []).append(embedding)
        created_at = int(time.time())
        for model, items in by_model.items():
            dimensions = {len(item.vector) for item in items}
            if len(dimensions) > 1:
                raise VectorValidationError(
                    f"Multiple vector dimensions found for model {model}: {dimensions}"
                )
            dimension = dimensions.pop()
            name = _once_per_ingest(
                ("collection", model, dimension),
                partial(ensure_milvus_collection, self.client, model, dimension),
            )
            kb_id = self._kb_id(user_id)
            self.client.upsert(
                name,
                [
                    {
                        "chunk_id": item.chunk_id,
                        "kb_id": kb_id,
                        "user_id": user_id,
                        "doc_id": item.doc_id,
                        "parse_hash": item.parse_hash,
                        "config_hash": "",
                        "text": item.text,
                        "dense": item.vector,
                        "visible": False,
                        "created_at": created_at,
                        "metadata": item.metadata or {},
                    }
                    for item in items
                ],
            )
        return EmbeddingWriteResponse(
            upsert_count=len(embeddings),
            deleted_stale_count=0,
            index_status=IndexOperation.SKIPPED.value,
        )

    def commit_embeddings(
        self,
        doc_id: str,
        parse_hash: str,
        model: str,
        *,
        commit_gate: Callable[[], None] | None = None,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> None:
        """Show the document's invisible rows, check the chunk set, drop the old batch.

        The gate runs before the flip and again before the delete, so a superseded
        ingest flips nothing and one superseded in between deletes nothing. The
        flip is one partial update per document: writes have a fixed 200 ms
        granularity, so batching would multiply the wait.
        """
        if commit_gate is not None:
            commit_gate()
        chunk_set = {
            row["chunk_id"]
            for row in self._ledger_chunks(
                doc_id, parse_hash, None, user_id, is_admin, columns=["chunk_id"]
            )
        }
        name = milvus_collection_name(model)
        # Fresh on purpose: a kb_id that changed since the ingest resolved it
        # fails the pre-flip check.
        kb_id = self._kb_id(user_id, fresh=True)

        def rows() -> dict[str, bool]:
            return _document_rows(self.client, name, kb_id, doc_id)

        def require(held: dict[str, bool], states: tuple[bool, ...], gap: str) -> None:
            missing = sorted(c for c in chunk_set if held.get(c) not in states)
            if missing:
                raise DatabaseOperationError(
                    f"{len(missing)} of {len(chunk_set)} chunks of {doc_id} {gap} "
                    f"(for example {missing[:5]}); retry the ingest"
                )

        def show(held: dict[str, bool]) -> None:
            hidden = [chunk_id for chunk_id in chunk_set if held.get(chunk_id) is False]
            if hidden:
                self.client.upsert(
                    name,
                    [{"chunk_id": chunk_id, "visible": True} for chunk_id in hidden],
                    partial_update=True,
                )

        held = rows()
        require(
            held,
            (False, True),
            "have no row under this kb_id, so nothing was committed",
        )
        flipped = sorted(c for c in chunk_set if held.get(c) is False)
        for attempt in (1, 2):
            try:
                show(held)
                break
            except Exception as error:  # noqa: BLE001 - the retry skips shown rows
                if attempt == 2:
                    raise DatabaseOperationError(
                        f"Could not show the chunks of {doc_id} in Milvus: {error}"
                    ) from error
                held = rows()
        held = rows()
        try:
            require(
                held, (True,), "are missing or invisible in Milvus after the commit"
            )
        except DatabaseOperationError:
            # Hide again what this commit showed, unless the gate raises (superseded or failed).
            try:
                if commit_gate is not None:
                    commit_gate()
            except Exception as error:  # noqa: BLE001 - the check error is reported
                logger.warning("Not hiding the chunks of %s again: %r", doc_id, error)
            else:
                self._hide(name, [c for c in flipped if held.get(c) is True])
            raise
        if commit_gate is not None:
            commit_gate()
        old = [chunk_id for chunk_id in held if chunk_id not in chunk_set]
        if old:
            self.client.delete(
                name,
                filter="kb_id == {kb_id} and doc_id == {doc_id} and chunk_id in {old}",
                filter_params={"kb_id": kb_id, "doc_id": doc_id, "old": old},
            )

    def validate_query_vector(
        self,
        query_vector: list[float],
        *,
        model_tag: str | None = None,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> None:
        """Check the vector's format; nothing is sent to Milvus."""
        validate_query_vector_format(query_vector)

    def _search_scope(
        self, filters: dict[str, Any] | None, user_id: int | None, is_admin: bool
    ) -> tuple[str, dict[str, Any]] | None:
        """Return the filter all three paths share and its parameters, or ``None``
        without a readable kb_id. Caller filters are translated first and may raise."""
        extra, params = caller_filter(filters)
        kb_ids = self._kb_ids(user_id, is_admin)
        if not kb_ids:
            return None
        return _visible_filter(kb_ids) + (f" and {extra}" if extra else ""), params

    def _read_model(
        self, model_tag: str, read: Callable[[str], list[Any]]
    ) -> list[Any]:
        """Run ``read`` on the collection of the model id; one not loaded holds nothing."""
        name = milvus_collection_name(model_tag)
        with _unloaded_as_empty(name, set()):
            return read(name)
        return []

    @staticmethod
    def _search_failure(
        code: str, route: str, model_tag: str, error: Exception
    ) -> SearchWarning:
        logger.error("%s search failed for %s: %s", route, model_tag, error)
        return SearchWarning(
            code=code,
            message=f"An unexpected error occurred during {route} search: {error}",
            fallback_action=SearchFallbackAction.PARTIAL_RESULTS,
            affected_models=[model_tag],
        )

    def _dense_response(
        self,
        model_tag: str,
        query_vector: list[float],
        top_k: int,
        scope: tuple[str, dict[str, Any]] | None,
        nprobes: int | None,
        refine_factor: int | None,
    ) -> DenseSearchResponse:
        try:
            hits: list[Any] = []
            if scope:
                expr, params = scope
                hits = self._read_model(
                    model_tag,
                    lambda name: self.client.search(
                        name,
                        data=[[float(x) for x in query_vector]],
                        anns_field="dense",
                        limit=top_k,
                        search_params={"metric_type": "COSINE"},
                        filter=expr,
                        filter_params=params,
                        output_fields=SEARCH_FIELDS,
                    )[0],
                )
            results = [
                to_result(hit["entity"], dense_score(hit["distance"]), model_tag)
                for hit in hits
            ]
        except Exception as error:
            warning = self._search_failure(
                "DENSE_SEARCH_FAILED", "dense", model_tag, error
            )
            return DenseSearchResponse(
                results=[],
                total_count=0,
                status="failed",
                warnings=[warning],
                index_status=IndexStatus.NO_INDEX,
                nprobes=nprobes,
                refine_factor=refine_factor,
            )
        return DenseSearchResponse(
            results=results,
            total_count=len(results),
            index_status=IndexStatus.INDEX_READY,
            nprobes=nprobes,
            refine_factor=refine_factor,
        )

    def _substring_results(
        self,
        model_tag: str,
        term: str,
        top_k: int,
        expr: str,
        params: dict[str, Any],
    ) -> list[SearchResult]:
        """Return up to ``top_k`` rows containing ``term``, paged because the pattern
        over-matches. Keyed by chunk_id: a write between pages can repeat a row."""
        expr += " and text like {pattern}"
        params = {**params, "pattern": like_pattern(term)}
        found: dict[str, SearchResult] = {}
        page_size = max(top_k, _MILVUS_FALLBACK_PAGE)
        for offset in range(0, _MILVUS_QUERY_WINDOW, page_size):
            size = min(page_size, _MILVUS_QUERY_WINDOW - offset)
            page = self._read_model(
                model_tag,
                lambda name: self.client.query(
                    name,
                    filter=expr,
                    filter_params=params,
                    output_fields=SEARCH_FIELDS,
                    limit=size,
                    offset=offset,
                ),
            )
            found.update(
                {
                    row["chunk_id"]: to_result(row, 1.0, model_tag)
                    for row in page
                    if term in row["text"]
                }
            )
            if len(found) >= top_k or len(page) < size:
                break
        return list(found.values())[:top_k]

    def _keyword_results(
        self,
        model_tag: str,
        query_text: str,
        top_k: int,
        expr: str,
        params: dict[str, Any],
    ) -> tuple[list[SearchResult], list[SearchWarning]]:
        hits = self._read_model(
            model_tag,
            lambda name: self.client.search(
                name,
                data=[query_text],
                anns_field="sparse",
                limit=top_k,
                search_params={"metric_type": "BM25"},
                filter=expr,
                filter_params=params,
                output_fields=SEARCH_FIELDS,
            )[0],
        )
        results = [
            to_result(hit["entity"], keyword_score(hit["distance"]), model_tag)
            for hit in hits
        ]
        if results:
            return results, []
        try:
            results = self._substring_results(
                model_tag, query_text, top_k, expr, params
            )
        except Exception as error:
            logger.error("Substring fallback failed: %s", error)
            return [], []
        fallback = SearchWarning(
            code="FTS_FALLBACK",
            message="Keyword search returned no matches; used substring search.",
            fallback_action=SearchFallbackAction.BRUTE_FORCE,
            affected_models=[model_tag],
        )
        return results, [fallback] if results else []

    def _sparse_response(
        self,
        model_tag: str,
        query_text: str,
        top_k: int,
        scope: tuple[str, dict[str, Any]] | None,
    ) -> SparseSearchResponse:
        try:
            results, warnings = (
                ([], [])
                if scope is None
                else self._keyword_results(model_tag, query_text, top_k, *scope)
            )
        except Exception as error:
            failure = self._search_failure(
                "FTS_SEARCH_FAILED", "sparse", model_tag, error
            )
            return SparseSearchResponse(
                results=[],
                total_count=0,
                status="failed",
                warnings=[failure],
                fts_enabled=True,
                query_text=query_text,
            )
        return SparseSearchResponse(
            results=results,
            total_count=len(results),
            warnings=warnings,
            fts_enabled=True,
            query_text=query_text,
        )

    def search_dense(
        self,
        model_tag: str,
        query_vector: list[float],
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        readonly: bool = False,
        nprobes: int | None = None,
        refine_factor: int | None = None,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> DenseSearchResponse:
        """Search the visible rows of the readable kb_ids by cosine similarity.

        ``model_tag`` is the model id the vectors were written with, as the search
        pipeline passes it. Scores follow LanceDB's L2 scale; ``readonly``,
        ``nprobes`` and ``refine_factor`` have no effect on the HNSW index.
        """
        scope = self._search_scope(filters, user_id, is_admin)
        return self._dense_response(
            model_tag, query_vector, top_k, scope, nprobes, refine_factor
        )

    def search_sparse(
        self,
        model_tag: str,
        query_text: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
        readonly: bool = False,
        nprobes: int | None = None,
        refine_factor: int | None = None,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> SparseSearchResponse:
        """BM25 search of the same rows; a query with no match falls back to ``LIKE``."""
        scope = self._search_scope(filters, user_id, is_admin)
        return self._sparse_response(model_tag, query_text, top_k, scope)

    def search_hybrid(
        self,
        model_tag: str,
        query_text: str,
        query_vector: list[float],
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        fusion_config: FusionConfig | None = None,
        readonly: bool = False,
        nprobes: int | None = None,
        refine_factor: int | None = None,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> HybridSearchResponse:
        """Run the dense and BM25 queries concurrently and fuse them in xagent."""
        scope = self._search_scope(filters, user_id, is_admin)
        with ThreadPoolExecutor(max_workers=1) as pool:
            dense = pool.submit(
                self._dense_response,
                model_tag,
                query_vector,
                top_k * 2,
                scope,
                nprobes,
                refine_factor,
            )
            sparse = self._sparse_response(model_tag, query_text, top_k * 2, scope)
        return _fuse_hybrid(
            model_tag,
            query_text,
            dense.result(),
            sparse,
            top_k=top_k,
            fusion_config=fusion_config or FusionConfig(),
        )

    def _hide(self, name: str, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        try:
            self.client.upsert(
                name,
                [{"chunk_id": chunk_id, "visible": False} for chunk_id in chunk_ids],
                partial_update=True,
            )
        except Exception as error:  # noqa: BLE001 - the commit error is the one to report
            logger.warning("Could not hide %s chunks again: %s", len(chunk_ids), error)

    def discard_uncommitted_embeddings(
        self, doc_id: str, *, user_id: int | None = None
    ) -> int:
        """Delete the document's invisible rows under the caller's kb_id, in every model.

        A failed commit hides again what it showed, so nothing else of the run is
        left. A collection that is not loaded is skipped: invisible rows are never searched.
        """
        kb_ids = self._kb_ids(user_id, False)
        if not kb_ids:
            return 0
        deleted = _delete_milvus_rows(
            self.client,
            _milvus_collections(self.client),
            kb_ids,
            doc_ids=[doc_id],
            invisible_only=True,
        )
        return sum(deleted.values())

    def delete_documents_data(
        self,
        doc_ids: list[str],
        *,
        user_id: int | None,
        is_admin: bool,
        warnings_out: list[str] | None = None,
    ) -> dict[str, int]:
        """Delete the documents' rows in Milvus first, then in the ledger.

        A failed Milvus delete raises before the ledger is touched, so the documents
        stay listed and a retry finishes the job.
        """
        ids = sorted({str(doc_id) for doc_id in doc_ids if doc_id})
        kb_ids = self._kb_ids(user_id, is_admin) if ids else []
        size = DEFAULT_VECTOR_STORE_DELETE_BATCH_SIZE
        batches = [ids[i : i + size] for i in range(0, len(ids), size)]
        names = _milvus_collections(self.client) if kb_ids else []
        keys = _model_keys(self.client, names)
        deleted: Counter[str] = Counter()
        for index, batch in enumerate(batches if kb_ids else [], start=1):
            try:
                deleted.update(
                    _delete_milvus_rows(self.client, names, kb_ids, doc_ids=batch)
                )
            except DatabaseOperationError as error:
                if warnings_out is not None:
                    warnings_out.append(str(error))
                raise DatabaseOperationError(
                    str(error),
                    details={
                        "deleted_counts": {},
                        "deleted_doc_ids": [],
                        "failed_batch_index": index,
                    },
                ) from None
        counts = _per_model(keys, deleted)
        counts.update(
            self.ledger.delete_documents_data(
                ids, user_id=user_id, is_admin=is_admin, warnings_out=warnings_out
            )
        )
        return dict(counts)

    def delete_collection_data(
        self,
        *,
        user_id: int | None,
        is_admin: bool,
        warnings_out: list[str] | None = None,
    ) -> dict[str, int]:
        """Delete the collection's rows in Milvus first, then in the ledger.

        Admins delete the rows of every owner of the name, others their own. A failed
        Milvus delete raises before the ledger and ``kb_ids`` are touched.
        """
        kb_ids = self._kb_ids(user_id, is_admin)
        names = _milvus_collections(self.client) if kb_ids else []
        keys = _model_keys(self.client, names)
        deleted = (
            _delete_milvus_rows(self.client, names, kb_ids) if kb_ids else Counter()
        )
        counts = _per_model(keys, deleted)
        counts.update(
            self.ledger.delete_collection_data(
                user_id=user_id, is_admin=is_admin, warnings_out=warnings_out
            )
        )
        return dict(counts)

    def cleanup_collection_data_after_rollback(
        self, *, user_id: int | None, is_admin: bool
    ) -> dict[str, int]:
        """Delete the collection's rows, as :meth:`delete_collection_data`."""
        return self.delete_collection_data(user_id=user_id, is_admin=is_admin)

    def _documents_of(self, user_id: int | None) -> int:
        """Count the caller's documents on a fresh table; raises when it cannot."""
        store = self.context.vector_index_store
        table = None
        try:
            conn = store.get_raw_connection()
            if "documents" not in list_table_names(conn):
                return 0
            table = conn.open_table("documents")
            where = store.build_filter_expression(
                build_filter_from_dict({"collection": self.context.collection}),
                user_id=user_id,
                is_admin=False,
            )
            return _safe_count_rows(table, where, on_error="raise")
        except Exception as error:
            raise DatabaseOperationError(
                f"Cannot count the documents of {self.context.collection}: {error}"
            ) from error
        finally:
            _safe_close_table(table)

    async def delete_collection_config(self, *, tenant_only: bool = False) -> int:
        """Delete the config rows, then kb_id rows: ``tenant_only`` only the caller's own, once a count shows it has no documents left."""
        deleted = await self.ledger.delete_collection_config(tenant_only=tenant_only)
        scope = self.context.user_scope
        if tenant_only:
            if await asyncio.to_thread(self._documents_of, scope.user_id):
                return deleted
        user_id, is_admin = (scope.user_id, False) if tenant_only else (None, True)
        await asyncio.to_thread(
            delete_kb_ids,
            self.context.vector_index_store.get_raw_connection(),
            self.context.collection,
            user_id=user_id,
            is_admin=is_admin,
        )
        return deleted

    def rename_collection_data(
        self,
        new_name: str,
        user_id: int | None,
        is_admin: bool,
        warnings_out: list[str] | None = None,
    ) -> list[str]:
        """Rename the kb_ids first, then the ledger data; an error or warning moves them back.

        The kb_id keeps its value, so Milvus rows follow the name without a write.
        """
        conn = self.context.vector_index_store.get_raw_connection()
        old = self.context.collection
        renamed = rename_kb_ids(conn, old, new_name, user_id=user_id, is_admin=is_admin)

        def undo() -> None:
            try:
                set_kb_ids_collection(conn, renamed, old)
            except Exception as error:  # noqa: BLE001 - the failed rename is reported
                logger.error("Could not move the kb_ids back to %s: %s", old, error)

        try:
            warnings = self.ledger.rename_collection_data(
                new_name, user_id, is_admin, warnings_out
            )
        except BaseException:
            undo()
            raise
        if warnings:
            undo()
        return warnings

    def restore_document_rows(
        self, snapshot: KBDocumentRowsSnapshot, *, user_id: int, is_admin: bool
    ) -> list[str]:
        """Restore the ledger from ``snapshot``, then align Milvus with its chunks.

        Rows whose chunk the restored ledger does not hold are deleted, whether the
        failed run stopped before its commit or after it. A document whose latest
        chunk set (the parse its restored ingestion status records) is then not fully
        visible is marked partially embedded and returned, if its restored status is
        success or partially embedded. When the alignment or a mark fails, the documents
        not aligned yet are marked the same way and the error carries every marked
        document as ``details["marked"]``.
        """
        self.ledger.restore_document_rows(snapshot, user_id=user_id, is_admin=is_admin)
        chunks: dict[tuple[str, str], set[str]] = {}
        for row in snapshot.rows_by_table.get("chunks", []):
            chunks.setdefault((row["doc_id"], row["parse_hash"]), set()).add(
                row["chunk_id"]
            )
        statuses = {
            row["doc_id"]: row
            for row in snapshot.rows_by_table.get("ingestion_runs", [])
        }
        marked: list[str] = []

        def mark(doc_id: str) -> None:
            status = statuses.get(doc_id)
            if status and status["status"] in _MARKABLE:
                self.ledger.write_ingestion_status(
                    doc_id,
                    status=DocumentProcessingStatus.PARTIALLY_EMBEDDED.value,
                    message="Vectors are incomplete after a failed refresh; "
                    "re-ingest the file.",
                    parse_hash=status["parse_hash"],
                    user_id=status.get("user_id"),
                )
                marked.append(doc_id)

        todo = list(snapshot.doc_ids)
        marking = None
        try:
            kb_ids = self._kb_ids(user_id, is_admin)
            names = _milvus_collections(self.client)
            while todo:
                doc_id = todo[0]
                keep = set().union(
                    *(ids for (doc, _), ids in chunks.items() if doc == doc_id)
                )
                visible: set[str] = set()
                for name in names:
                    for kb_id in kb_ids:
                        held = _with_loaded(
                            name,
                            self.client,
                            partial(_document_rows, self.client, name, kb_id, doc_id),
                        )
                        if stale := sorted(set(held) - keep):
                            self.client.delete(
                                name,
                                filter="kb_id == {kb_id} and doc_id == {doc_id} "
                                "and chunk_id in {stale}",
                                filter_params={
                                    "kb_id": kb_id,
                                    "doc_id": doc_id,
                                    "stale": stale,
                                },
                            )
                        visible.update(
                            c for c, shown in held.items() if shown and c in keep
                        )
                status = statuses.get(doc_id)
                latest = chunks.get((doc_id, status["parse_hash"])) if status else None
                if latest and not latest <= visible:
                    marking = doc_id
                    mark(doc_id)
                    marking = None
                todo.pop(0)
        except Exception as error:
            for doc_id in todo:
                try:
                    mark(doc_id)
                except Exception as mark_error:  # noqa: BLE001 - the alignment error is reported
                    logger.error(
                        "Could not mark %s partially embedded: %s", doc_id, mark_error
                    )
            step = (
                f"mark {marking} partially embedded"
                if marking
                else "align Milvus with the restored chunks"
            )
            raise DatabaseOperationError(
                f"Could not {step}: {error}", details={"marked": marked}
            ) from error
        return marked

    register_document = _ledger(KBCollectionHandle.register_document)
    load_document = _ledger(KBCollectionHandle.load_document)
    list_documents = _ledger(KBCollectionHandle.list_documents)
    delete_document_record = _ledger(KBCollectionHandle.delete_document_record)
    snapshot_document = _ledger(KBCollectionHandle.snapshot_document)
    restore_document = _ledger(KBCollectionHandle.restore_document)
    delete_created_document = _ledger(KBCollectionHandle.delete_created_document)
    parse_exists = _ledger(KBCollectionHandle.parse_exists)
    read_parse_paragraphs = _ledger(KBCollectionHandle.read_parse_paragraphs)
    write_parse = _ledger(KBCollectionHandle.write_parse)
    read_latest_parse_record = _ledger(KBCollectionHandle.read_latest_parse_record)
    chunk_exists = _ledger(KBCollectionHandle.chunk_exists)
    read_existing_chunks = _ledger(KBCollectionHandle.read_existing_chunks)
    read_parse_paragraph_dicts = _ledger(KBCollectionHandle.read_parse_paragraph_dicts)
    write_chunks = _ledger(KBCollectionHandle.write_chunks)
    delete_parse_records = _ledger(KBCollectionHandle.delete_parse_records)
    delete_chunk_records = _ledger(KBCollectionHandle.delete_chunk_records)
    snapshot_parse = _ledger(KBCollectionHandle.snapshot_parse)
    restore_parse = _ledger(KBCollectionHandle.restore_parse)
    delete_created_parse = _ledger(KBCollectionHandle.delete_created_parse)
    snapshot_chunks = _ledger(KBCollectionHandle.snapshot_chunks)
    restore_chunks = _ledger(KBCollectionHandle.restore_chunks)
    delete_created_chunks = _ledger(KBCollectionHandle.delete_created_chunks)
    rename_collection_status = _ledger(KBCollectionHandle.rename_collection_status)
    rename_collection_metadata = _ledger(KBCollectionHandle.rename_collection_metadata)
    count_documents = _ledger(KBCollectionHandle.count_documents)
    list_collection_documents = _ledger(KBCollectionHandle.list_collection_documents)
    write_ingestion_status = _ledger(KBCollectionHandle.write_ingestion_status)
    load_ingestion_status = _ledger(KBCollectionHandle.load_ingestion_status)
    clear_ingestion_status = _ledger(KBCollectionHandle.clear_ingestion_status)
    write_ingestion_status_async = _ledger(
        KBCollectionHandle.write_ingestion_status_async
    )
    load_ingestion_status_async = _ledger(
        KBCollectionHandle.load_ingestion_status_async
    )
    clear_ingestion_status_async = _ledger(
        KBCollectionHandle.clear_ingestion_status_async
    )
    get_main_pointer = _ledger(KBCollectionHandle.get_main_pointer)
    set_main_pointer = _ledger(KBCollectionHandle.set_main_pointer)
    list_main_pointers = _ledger(KBCollectionHandle.list_main_pointers)
    delete_main_pointer = _ledger(KBCollectionHandle.delete_main_pointer)
    capture_status_snapshot = _ledger(KBCollectionHandle.capture_status_snapshot)
    restore_status_snapshot = _ledger(KBCollectionHandle.restore_status_snapshot)
    clear_status_snapshot = _ledger(KBCollectionHandle.clear_status_snapshot)
    capture_main_pointer_snapshot = _ledger(
        KBCollectionHandle.capture_main_pointer_snapshot
    )
    restore_main_pointer_snapshot = _ledger(
        KBCollectionHandle.restore_main_pointer_snapshot
    )

    search_dense_async = _unsupported(
        KBCollectionHandle.search_dense_async, _ASYNC_SEARCH
    )
    search_sparse_async = _unsupported(
        KBCollectionHandle.search_sparse_async, _ASYNC_SEARCH
    )
    search_hybrid_async = _unsupported(
        KBCollectionHandle.search_hybrid_async, _ASYNC_SEARCH
    )
    cleanup_cascade = _unsupported(KBCollectionHandle.cleanup_cascade, _CASCADE)
    cleanup_document_cascade = _unsupported(
        KBCollectionHandle.cleanup_document_cascade, _CASCADE
    )
    cleanup_parse_cascade = _unsupported(
        KBCollectionHandle.cleanup_parse_cascade, _CASCADE
    )
    cleanup_chunk_cascade = _unsupported(
        KBCollectionHandle.cleanup_chunk_cascade, _CASCADE
    )
    cleanup_embed_cascade = _unsupported(
        KBCollectionHandle.cleanup_embed_cascade, _CASCADE
    )
    list_candidates = _unsupported(KBCollectionHandle.list_candidates, _VERSIONS)
    promote_version_main = _unsupported(
        KBCollectionHandle.promote_version_main, _VERSIONS
    )
    capture_candidate_cleanup_snapshot = _unsupported(
        KBCollectionHandle.capture_candidate_cleanup_snapshot, _VERSIONS
    )
    restore_candidate_cleanup_snapshot = _unsupported(
        KBCollectionHandle.restore_candidate_cleanup_snapshot, _VERSIONS
    )

    capture_document_rows = _ledger(KBCollectionHandle.capture_document_rows)
    delete_embedding_records = _pending(KBCollectionHandle.delete_embedding_records)
    snapshot_embeddings = _pending(KBCollectionHandle.snapshot_embeddings)
    restore_embeddings = _pending(KBCollectionHandle.restore_embeddings)
    delete_created_embeddings = _pending(KBCollectionHandle.delete_created_embeddings)
    cleanup_embeddings_for_operation = _pending(
        KBCollectionHandle.cleanup_embeddings_for_operation
    )
