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
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..core.config import DEFAULT_LANCEDB_BATCH_SIZE
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
    DocumentRecordDetail,
    DocumentRecordListResult,
    EmbeddingReadResponse,
    EmbeddingRecordSnapshot,
    EmbeddingWriteResponse,
    IndexOperation,
    ParsedParagraph,
    ParseRecordDetail,
    RegisterDocumentRequest,
    RegisterDocumentResponse,
)
from ..LanceDB.model_tag_utils import to_model_tag
from ..storage.contracts import MetadataStore, VectorIndexStore
from ..utils import check_file_type, compute_file_hash
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
