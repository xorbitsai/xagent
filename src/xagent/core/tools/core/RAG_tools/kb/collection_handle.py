"""Collection-scoped KB backend handle.

``KBCollectionHandle`` is the collection-scoped backend boundary that owns
backend-specific data-plane mechanics (the document-row lifecycle in #508 and
the parse/chunk lifecycle in #509). ``LanceDBCollectionHandle`` is the first
implementation and delegates to the current LanceDB tables via the bound
vector index store.
"""

from __future__ import annotations

import json
import logging
import numbers
import os
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, cast

if TYPE_CHECKING:
    from .maintenance_compatibility import CollectionConfigSnapshot

import pandas as pd

try:
    import pyarrow as pa  # type: ignore
    from pyarrow import Table as PyArrowTable
except ImportError:  # pragma: no cover - pyarrow is an optional runtime dep
    pa = None
    PyArrowTable = Any

from ..core.config import (
    DEFAULT_LANCEDB_BATCH_SIZE,
)
from ..core.exceptions import (
    ConfigurationError,
    DatabaseOperationError,
    DocumentValidationError,
    HashComputationError,
    VectorValidationError,
)
from ..core.schemas import (
    ChunkEmbeddingData,
    ChunkForEmbedding,
    ChunkRecordSnapshot,
    DenseSearchResponse,
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
)
from ..LanceDB.model_tag_utils import to_model_tag
from ..LanceDB.schema_manager import _safe_close_table
from ..retrieval.search_hybrid import _linear_fusion, _rrf_fusion
from ..storage.contracts import (
    FilterCondition,
    FilterExpression,
    FilterOperator,
    MetadataStore,
    VectorIndexStore,
)
from ..utils import check_file_type, compute_file_hash
from ..utils.filter_utils import parse_legacy_filters, validate_filter_depth
from ..utils.hash_utils import compute_chunk_hash
from ..utils.metadata_utils import deserialize_metadata, serialize_metadata
from ..utils.string_utils import generate_deterministic_doc_id
from .models import KBBackendCapabilities, KBCollectionContext, KBStorageBackend

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


class KBHandleProvider:
    """Open collection-scoped handles for resolved KB contexts."""

    def open(self, context: KBCollectionContext) -> LanceDBCollectionHandle:
        """Return a backend-specific handle for the resolved collection context."""
        if context.backend is KBStorageBackend.LANCEDB:
            return LanceDBCollectionHandle(context)
        raise ValueError(
            f"KB storage backend {context.backend.value!r} is not supported by "
            "KBHandleProvider"
        )

    def reset_for_tests(self) -> None:
        """Clear provider-owned caches for test reset.

        The current provider is stateless, but the hook keeps the coordinator
        reset path ready for backend handle caches.
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
        """Execute dense vector search for this collection."""

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
        """Async dense vector search for this collection."""

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
        """Execute sparse (FTS) search for this collection."""

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
        """Async sparse (FTS) search for this collection."""

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
        """Execute hybrid (dense + sparse) search with fusion for this collection."""

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
        """Async hybrid (dense + sparse) search with fusion for this collection."""

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

        Async – this is the **only** async method on ``KBCollectionHandle``.
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
    async def capture_collection_config_snapshot(
        self,
    ) -> "CollectionConfigSnapshot":
        """Capture the collection_config row for this collection before mutation.

        Returns a :class:`CollectionConfigSnapshot` whose ``existed`` flag is
        ``True`` when a config row was present and ``False`` otherwise.  A
        snapshot with ``existed=False`` is safe to pass to
        :meth:`restore_collection_config_snapshot` – the restore is a no-op.
        """

    @abstractmethod
    async def restore_collection_config_snapshot(
        self,
        snapshot: "CollectionConfigSnapshot",
    ) -> None:
        """Restore or remove a collection_config row from a snapshot.

        When ``snapshot.existed`` is ``True`` the original config JSON is
        written back via :meth:`MetadataStore.save_collection_config`.  When
        ``snapshot.existed`` is ``False`` this is a no-op (the config row did
        not exist before the mutation so there is nothing to restore).

        The rollback-complete / side-effects-may-remain guard logic lives in
        the coordinator/policy layer, not here.
        """

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
                metadata = deserialize_metadata(chunk_dict.get("metadata"))
                index = _safe_int_value(chunk_dict.get("index"), default=0)

                page_number_value = chunk_dict.get("page_number")
                # A missing page number can arrive as None or as a pandas/LanceDB
                # NaN sentinel (NaN != NaN); both mean "no page", not page 1.
                if (
                    page_number_value is not None
                    and page_number_value == page_number_value
                ):
                    page_num = _safe_int_value(page_number_value, default=1)
                    page_number = page_num if page_num > 0 else None
                else:
                    page_number = None

                pending_chunks.append(
                    ChunkForEmbedding(
                        doc_id=chunk_dict["doc_id"],
                        chunk_id=chunk_id,
                        parse_hash=chunk_dict["parse_hash"],
                        index=index,
                        text=chunk_dict["text"],
                        chunk_hash=chunk_dict["chunk_hash"],
                        page_number=page_number,
                        section=_safe_optional_str(chunk_dict.get("section")),
                        anchor=_safe_optional_str(chunk_dict.get("anchor")),
                        json_path=_safe_optional_str(chunk_dict.get("json_path")),
                        metadata=metadata,
                    )
                )

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
        if filters and isinstance(filters, dict):
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
            if filters and isinstance(filters, dict):
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
                batch_df["text"]
                .astype(str)
                .str.contains(query_text, na=False, regex=False)
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

            search_query = table.search(query_text, query_type="fts").limit(top_k)

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
                if backend_filter:
                    search_query = search_query.where(backend_filter)

            # LanceDB's search().to_pandas() returns Any due to missing type stubs
            raw_results_df = pd.DataFrame(search_query.to_pandas())

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
                table=table,
                collection=collection,
                query_text=query_text,
                model_tag=model_tag,
                top_k=top_k,
                filters=filters,
                current_warnings=current_warnings,
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
        return self._fuse_hybrid(
            model_tag,
            query_text,
            dense_response,
            sparse_response,
            top_k=top_k,
            fusion_config=fusion_config,
        )

    def _fuse_hybrid(
        self,
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
            unique_id = f"{result.doc_id}-{result.chunk_id}-{result.parse_hash}-{result.model_tag}"
            dense_rank_map[unique_id] = rank
            dense_score_map[unique_id] = result.score

        for rank, result in enumerate(sparse_results, start=1):
            unique_id = f"{result.doc_id}-{result.chunk_id}-{result.parse_hash}-{result.model_tag}"
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
            unique_id = f"{result.doc_id}-{result.chunk_id}-{result.parse_hash}-{result.model_tag}"
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
        return self._fuse_hybrid(
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
        """Rename ingestion status rows in the ``ingestion_runs`` table.

        Delegates to the ingestion status store's ``rename_collection_status``,
        passing ``self.context.collection`` as the old name.  The coordinator
        should call this via ``asyncio.to_thread`` when running in an async
        context.

        Returns:
            List of warning messages on partial failure (empty on success).
        """
        from ..storage.factory import get_ingestion_status_store

        store = get_ingestion_status_store()
        return store.rename_collection_status(
            old_name=self.context.collection,
            new_name=new_name,
            user_id=user_id,
            is_admin=is_admin,
        )

    async def rename_collection_metadata(
        self,
        new_name: str,
        user_id: int | None,
        is_admin: bool,
    ) -> None:
        """Rename control-plane metadata rows to ``new_name``.

        This is the **only** async method on ``KBCollectionHandle``.  It wraps
        ``await metadata_store.rename_collection(...)`` which updates the
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

    async def capture_collection_config_snapshot(
        self,
    ) -> "CollectionConfigSnapshot":
        """Capture the collection_config row for this collection (metadata read only).

        Reads the config row via the metadata store and wraps it in a
        :class:`CollectionConfigSnapshot`.  ``config_user_id`` is normalized to
        0 when ``user_id`` is ``None``, matching legacy ownership convention.
        """
        from .maintenance_compatibility import CollectionConfigSnapshot

        collection = self.context.collection
        # Normalize: None user_id maps to 0 (legacy convention).
        user_id = self.context.user_scope.user_id
        config_user_id: int = 0 if user_id is None else int(user_id)

        config_json = await self.metadata_store.get_collection_config(
            collection,
            config_user_id,
            is_admin=False,
        )
        return CollectionConfigSnapshot(
            collection_name=collection,
            user_id=user_id,
            config_user_id=config_user_id,
            config_json=config_json,
            existed=config_json is not None,
        )

    async def restore_collection_config_snapshot(
        self,
        snapshot: "CollectionConfigSnapshot",
    ) -> None:
        """Restore a collection_config row from snapshot (metadata write only).

        When ``snapshot.existed`` is ``True`` the config JSON is written back
        via :meth:`MetadataStore.save_collection_config`.  When
        ``snapshot.existed`` is ``False`` this is a no-op.

        The rollback-complete / side-effects-may-remain guard lives in the
        coordinator/policy layer and is intentionally absent here.
        """
        if not snapshot.existed:
            return
        assert (
            snapshot.config_json is not None
        )  # invariant: existed ↔ config_json is not None
        await self.metadata_store.save_collection_config(
            snapshot.collection_name,
            snapshot.config_json,
            snapshot.config_user_id,
        )

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
