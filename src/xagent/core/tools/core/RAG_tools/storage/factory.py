"""Unified factory for all KB storage contracts.

Phase 1A Part 2: StorageFactory manages singleton instances of all stores
with lazy initialization and thread-safe access.

Backward compatibility: Convenience functions (get_vector_index_store, etc.)
are provided for existing code.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Optional, cast

from .contracts import (
    ActiveGenerationStore,
    IngestionStatusStore,
    KBWriteCoordinator,
    MainPointerStore,
    MetadataStore,
    PromptTemplateStore,
    VectorIndexStore,
)
from .lancedb_stores import (
    LanceDBActiveGenerationStore,
    LanceDBIngestionStatusStore,
    LanceDBMainPointerStore,
    LanceDBMetadataStore,
    LanceDBPromptTemplateStore,
    LanceDBVectorIndexStore,
)
from .vector_backend import (
    VectorBackend,
    get_configured_vector_backend,
    require_implemented_vector_backend,
)

logger = logging.getLogger(__name__)
_storage_shim_override: ContextVar[Any] = ContextVar(
    "xagent_kb_storage_shim_override",
    default=None,
)

if TYPE_CHECKING:
    from ..kb.storage_shim import KBStorageShimCompatibilityFacade


def get_bound_storage_shim_for_current_context() -> (
    KBStorageShimCompatibilityFacade | None
):
    """Return the storage shim already bound to this execution context."""
    return cast(
        "KBStorageShimCompatibilityFacade | None",
        _storage_shim_override.get(),
    )


@contextmanager
def bind_storage_shim_for_current_context(
    storage_shim: KBStorageShimCompatibilityFacade,
) -> Iterator[None]:
    """Route legacy storage accessors through a coordinator-owned shim."""
    token = _storage_shim_override.set(storage_shim)
    try:
        yield
    finally:
        _storage_shim_override.reset(token)


class StorageFactory:
    """Unified factory for all storage contracts.

    Manages singleton instances of all stores with lazy initialization
    and thread-safe access using double-checked locking.

    Usage:
        factory = StorageFactory.get_factory()
        vector_store = factory.get_vector_index_store()
        metadata_store = factory.get_metadata_store()
    """

    _instance: Optional[StorageFactory] = None
    _lock = threading.RLock()  # RLock for reentrant locking

    def __init__(self) -> None:
        """Private constructor - use get_factory() instead."""
        if StorageFactory._instance is not None:
            raise RuntimeError("Use get_factory() to get StorageFactory instance")

        # Store instances (lazy initialization)
        self._vector_index_store: Optional[VectorIndexStore] = None
        self._vector_backend: Optional[VectorBackend] = None
        self._metadata_store: Optional[MetadataStore] = None
        self._ingestion_status_store: Optional[IngestionStatusStore] = None
        self._prompt_template_store: Optional[PromptTemplateStore] = None
        self._main_pointer_store: Optional[MainPointerStore] = None
        self._active_generation_store: Optional[ActiveGenerationStore] = None
        self._coordinator: Optional[KBWriteCoordinator] = None

    @classmethod
    def get_factory(cls) -> StorageFactory:
        """Get singleton factory instance.

        Uses double-checked locking for thread-safe lazy initialization.

        Returns:
            The singleton StorageFactory instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def reset_all(self) -> None:
        """Reset all store instances.

        Useful for tests/fixtures that need isolated storage.
        Thread-safe: uses factory lock to prevent race conditions.
        """
        with self._lock:
            self._vector_index_store = None
            self._vector_backend = None
            self._metadata_store = None
            self._ingestion_status_store = None
            self._prompt_template_store = None
            self._main_pointer_store = None
            self._active_generation_store = None
            self._coordinator = None

        # Keep semantic KB coordinator state aligned with factory-backed stores.
        from ..kb import reset_kb_coordinator_for_tests

        reset_kb_coordinator_for_tests()

    # --- VectorIndexStore ---

    def get_vector_index_store(self) -> VectorIndexStore:
        """Get or create vector index store.

        Backend is selected via :envvar:`XAGENT_VECTOR_BACKEND` (or legacy
        ``VECTOR_STORE_BACKEND``); see :mod:`.vector_backend`.

        Returns:
            Concrete :class:`~.contracts.VectorIndexStore` (currently
            :class:`~.lancedb_stores.LanceDBVectorIndexStore` when backend is
            ``lancedb``).

        Raises:
            ConfigurationError: Unknown backend name, or backend not implemented
                yet (e.g. ``milvus`` / ``qdrant`` without an adapter).
        """
        if self._vector_index_store is None:
            with self._lock:
                if self._vector_index_store is None:
                    backend = get_configured_vector_backend()
                    require_implemented_vector_backend(backend)
                    if backend is VectorBackend.LANCEDB:
                        self._vector_index_store = LanceDBVectorIndexStore()
                        self._vector_backend = backend
                    else:
                        raise AssertionError(
                            "require_implemented_vector_backend must prevent this branch"
                        )
        return self._vector_index_store

    def get_resolved_vector_backend(self) -> VectorBackend:
        """Return the backend bound to the current vector index store singleton.

        After the store is created, this reflects the backend used at creation
        time (cached). Before creation, returns :func:`.get_configured_vector_backend`
        without instantiating the store.
        """
        if self._vector_backend is not None:
            return self._vector_backend
        return get_configured_vector_backend()

    # --- MetadataStore ---

    def get_metadata_store(self) -> MetadataStore:
        """Get or create metadata store.

        Returns:
            LanceDBMetadataStore instance.
        """
        if self._metadata_store is None:
            with self._lock:
                if self._metadata_store is None:
                    self._metadata_store = LanceDBMetadataStore()
        return self._metadata_store

    # --- IngestionStatusStore ---

    def get_ingestion_status_store(self) -> IngestionStatusStore:
        """Get or create ingestion status store.

        Returns:
            LanceDBIngestionStatusStore instance.
        """
        if self._ingestion_status_store is None:
            with self._lock:
                if self._ingestion_status_store is None:
                    self._ingestion_status_store = LanceDBIngestionStatusStore()
        return self._ingestion_status_store

    # --- PromptTemplateStore ---

    def get_prompt_template_store(self) -> PromptTemplateStore:
        """Get or create prompt template store.

        Returns:
            LanceDBPromptTemplateStore instance.
        """
        if self._prompt_template_store is None:
            with self._lock:
                if self._prompt_template_store is None:
                    self._prompt_template_store = LanceDBPromptTemplateStore()
        return self._prompt_template_store

    # --- ActiveGenerationStore ---

    def get_active_generation_store(self) -> ActiveGenerationStore:
        """Get or create active generation pointer store."""
        if self._active_generation_store is None:
            with self._lock:
                if self._active_generation_store is None:
                    self._active_generation_store = LanceDBActiveGenerationStore()
        return self._active_generation_store

    # --- MainPointerStore ---

    def get_main_pointer_store(self) -> MainPointerStore:
        """Get or create main pointer store.

        Returns:
            LanceDBMainPointerStore instance.
        """
        if self._main_pointer_store is None:
            with self._lock:
                if self._main_pointer_store is None:
                    self._main_pointer_store = LanceDBMainPointerStore()
        return self._main_pointer_store

    # --- KBWriteCoordinator ---

    def get_kb_write_coordinator(self) -> KBWriteCoordinator:
        """Get or create KB write coordinator.

        Returns:
            DefaultKBWriteCoordinator: Phase 1A shell delegating to metadata
            and vector stores only; see that class for future coordination scope.
        """
        if self._coordinator is None:
            with self._lock:
                if self._coordinator is None:
                    self._coordinator = DefaultKBWriteCoordinator(
                        metadata=self.get_metadata_store(),
                        vector_index=self.get_vector_index_store(),
                    )
        return self._coordinator


# ============================================================================
# Backward Compatibility Functions
# ============================================================================


def _get_storage_shim() -> KBStorageShimCompatibilityFacade:
    """Return the coordinator-owned low-level storage compatibility facade."""
    override = get_bound_storage_shim_for_current_context()
    if override is not None:
        return override

    from ..kb import get_kb_coordinator

    return get_kb_coordinator().storage_shim


def reset_kb_write_coordinator() -> None:
    """Reset process-global coordinator (useful for tests/fixtures).

    Deprecated: Use StorageFactory.get_factory().reset_all() instead.
    """
    _get_storage_shim().reset_kb_write_coordinator()


def reset_rag_storage_for_tests() -> None:
    """Reset all process-global KB/RAG storage state for tests.

    This is the **single entry point** pytest fixtures should use instead of
    importing vector-database modules (e.g. LanceDB) directly. It:

    1. Clears backend-specific connection caches and related resources for the
       configured :class:`~.vector_backend.VectorBackend`.
    2. Resets :class:`StorageFactory` singletons via :func:`reset_kb_write_coordinator`.

    When adding Milvus, Qdrant, etc., extend the backend branch below so test
    isolation stays correct without changing every ``conftest.py``.

    Raises:
        ConfigurationError: If ``XAGENT_VECTOR_BACKEND`` is set to an unknown value
            (same as :func:`~.vector_backend.get_configured_vector_backend`).
    """
    _get_storage_shim().reset_rag_storage_for_tests()
    logger.debug("[TEST_RESET] Global collection locks reset via public helper")


def get_kb_write_coordinator() -> KBWriteCoordinator:
    """Return process-global KB write coordinator.

    Deprecated: Use StorageFactory.get_factory().get_kb_write_coordinator() instead.
    """
    return _get_storage_shim().get_kb_write_coordinator()


def get_metadata_store() -> MetadataStore:
    """Convenience accessor for metadata store.

    Deprecated: Use StorageFactory.get_factory().get_metadata_store() instead.
    """
    return _get_storage_shim().get_metadata_store()


def get_vector_index_store() -> VectorIndexStore:
    """Convenience accessor for vector index store.

    Deprecated: Use StorageFactory.get_factory().get_vector_index_store() instead.
    """
    return _get_storage_shim().get_vector_index_store()


def get_vector_store_raw_connection() -> Any:
    """Return the LanceDB handle exposed by the vector index store singleton.

    Central entry point for RAG code that still needs a raw connection during
    Phase 1A. Replaces duplicated per-module ``get_connection_from_env`` helpers
    that only delegated to ``get_vector_index_store().get_raw_connection()``.

    Returns:
        The object returned by :meth:`VectorIndexStore.get_raw_connection`.
    """
    return _get_storage_shim().get_vector_store_raw_connection()


def get_ingestion_status_store() -> IngestionStatusStore:
    """Get ingestion status store.

    Returns:
        LanceDBIngestionStatusStore instance.
    """
    return _get_storage_shim().get_ingestion_status_store()


def get_prompt_template_store() -> PromptTemplateStore:
    """Get prompt template store.

    Returns:
        LanceDBPromptTemplateStore instance.
    """
    return _get_storage_shim().get_prompt_template_store()


def get_main_pointer_store() -> MainPointerStore:
    """Get main pointer store.

    Returns:
        LanceDBMainPointerStore instance.
    """
    return _get_storage_shim().get_main_pointer_store()


def get_active_generation_store() -> ActiveGenerationStore:
    """Get active generation pointer store.

    Returns:
        LanceDBActiveGenerationStore instance.
    """
    return _get_storage_shim().get_active_generation_store()


# ============================================================================
# Default Coordinator Implementation
# ============================================================================


class DefaultKBWriteCoordinator(KBWriteCoordinator):
    """In-process KB write coordinator: Phase 1A placeholder implementation.

    Only :meth:`metadata_store` and :meth:`vector_index_store` are implemented;
    both delegate to the injected or default LanceDB-backed stores. This is
    sufficient as a shell while call sites converge on :class:`KBWriteCoordinator`.
    Future phases may add distributed locking, batched writes, and conflict
    resolution without changing the high-level factory entry point.
    """

    def __init__(
        self,
        metadata: MetadataStore | None = None,
        vector_index: VectorIndexStore | None = None,
    ) -> None:
        self._metadata = metadata or LanceDBMetadataStore()
        self._vector_index = vector_index or LanceDBVectorIndexStore()

    def metadata_store(self) -> MetadataStore:
        return self._metadata

    def vector_index_store(self) -> VectorIndexStore:
        return self._vector_index
