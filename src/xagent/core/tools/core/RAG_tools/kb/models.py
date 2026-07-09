"""Semantic KB coordinator models.

These types describe collection-level KB context without moving existing
storage, API, pipeline, or tool behavior into the coordinator.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Optional

from ..core.schemas import CollectionInfo, IngestionResult
from ..storage.contracts import (
    IngestionStatusStore,
    MainPointerStore,
    MetadataStore,
    VectorIndexStore,
)
from .operation_compatibility import KBOperation, RollbackStatus


class KBAccessMode(StrEnum):
    """Semantic access mode requested by a KB caller."""

    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class KBStorageBackend(StrEnum):
    """Collection-level KB storage backend binding."""

    LANCEDB = "lancedb"


@dataclass(frozen=True)
class KBBackendCapabilities:
    """Capabilities exposed by a collection backend handle."""

    supports_documents: bool
    supports_parses: bool
    supports_chunks: bool
    supports_embeddings: bool
    supports_search: bool
    supports_versions: bool
    supports_raw_connection: bool

    @classmethod
    def lancedb(cls) -> KBBackendCapabilities:
        """Return current LanceDB-compatible KB capabilities."""
        return cls(
            supports_documents=True,
            supports_parses=True,
            supports_chunks=True,
            supports_embeddings=True,
            supports_search=True,
            supports_versions=True,
            supports_raw_connection=True,
        )

    @classmethod
    def unsupported(cls) -> KBBackendCapabilities:
        """Return capabilities for a known but unavailable backend."""
        return cls(
            supports_documents=False,
            supports_parses=False,
            supports_chunks=False,
            supports_embeddings=False,
            supports_search=False,
            supports_versions=False,
            supports_raw_connection=False,
        )


@dataclass(frozen=True)
class KBUserScope:
    """Resolved caller scope for KB context operations."""

    user_id: Optional[int]
    is_admin: bool


@dataclass(frozen=True)
class KBContextRequest:
    """Request for resolving collection-level KB context."""

    collection: str
    user_id: Optional[int] = None
    is_admin: Optional[bool] = None
    access_mode: KBAccessMode = KBAccessMode.READ
    allow_create: bool = False
    hide_missing: bool = False


@dataclass(frozen=True)
class KBCollectionContext:
    """Resolved collection-level context for coordinator and handle callers."""

    collection: str
    user_scope: KBUserScope
    access_mode: KBAccessMode
    allow_create: bool
    hide_missing: bool
    metadata_store: MetadataStore
    vector_index_store: VectorIndexStore
    ingestion_status_store: IngestionStatusStore
    main_pointer_store: MainPointerStore
    backend: KBStorageBackend
    capabilities: KBBackendCapabilities
    collection_info: Optional[CollectionInfo] = None


@dataclass(frozen=True)
class RollbackFailedIngestionRequest:
    """Request for coordinator-owned failed-ingest rollback orchestration (#515).

    ``document_compensation`` / ``status_compensation`` are factories invoked
    as ``factory(ingestion_result)`` returning the zero-arg callback;
    ``file_compensation`` / ``snapshot_compensation`` are plain zero-arg
    callables (mirrors the web pipeline's ``FileHandlerResult`` contract).

    ``rollback_context`` may carry web-file identity used for payload metadata
    and idempotency-key derivation. Reserved keys: ``rollback_kind``,
    ``backup_path``, ``file_id``, ``file_path``.

    ``operation=None`` selects the callback-only path (no saga engine).
    Additive-by-design: #795 may extend, but must not mutate, these fields.
    """

    collection: str
    user_id: Optional[int]
    is_admin: bool
    # Saga to compensate. None -> callback-only path.
    operation: Optional[KBOperation] = None
    ingestion_result: Optional[IngestionResult] = None
    doc_id: Optional[str] = None
    source: Optional[str] = None  # URL or source path, for warning text
    document_compensation: Optional[Callable[..., Any]] = None
    file_compensation: Optional[Callable[[], Any]] = None
    status_compensation: Optional[Callable[..., Any]] = None
    snapshot_compensation: Optional[Callable[[], Any]] = None
    rollback_context: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RollbackFailedIngestionResult:
    """Outcome of coordinator-owned failed-ingest rollback orchestration."""

    status: str  # "complete" | "incomplete" | "not_needed"
    rollback_status: RollbackStatus
    rollback_complete: bool
    side_effects_may_remain: bool
    first_error: Optional[str] = None
    boundary_errors: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class KBVectorStorageCleanupResult:
    """Outcome for vector-storage rollback cleanup actions."""

    collection: str
    status: str
    deleted_count: int = 0
    table_counts: dict[str, int] = field(default_factory=dict)
    model_tag: Optional[str] = None
    preview_only: bool = True
    warnings: tuple[str, ...] = ()
    side_effects_may_remain: bool = False
