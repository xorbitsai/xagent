"""Low-level KB storage compatibility facade."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..storage.contracts import (
        ActiveGenerationStore,
        IngestionStatusStore,
        KBWriteCoordinator,
        MainPointerStore,
        MetadataStore,
        PromptTemplateStore,
        VectorIndexStore,
    )
    from ..storage.factory import StorageFactory


class KBStorageShimCompatibilityFacade:
    """Compatibility boundary for legacy storage singleton accessors.

    The facade keeps the current storage factory and object lifecycles intact
    while giving newer compatibility layers a stable path that is owned by the
    semantic KB coordinator.
    """

    def __init__(self, storage_factory: StorageFactory | None = None) -> None:
        if storage_factory is None:
            from ..storage.factory import StorageFactory

            storage_factory = StorageFactory.get_factory()
        self._storage_factory = storage_factory

    def get_kb_write_coordinator(self) -> KBWriteCoordinator:
        """Return the legacy write coordinator singleton."""
        return self._storage_factory.get_kb_write_coordinator()

    def get_metadata_store(self) -> MetadataStore:
        """Return the legacy metadata store singleton."""
        return self._storage_factory.get_metadata_store()

    def get_vector_index_store(self) -> VectorIndexStore:
        """Return the legacy vector index store singleton."""
        return self._storage_factory.get_vector_index_store()

    def get_vector_store_raw_connection(self) -> Any:
        """Return the raw vector-store connection escape hatch."""
        return self.get_vector_index_store().get_raw_connection()

    def get_ingestion_status_store(self) -> IngestionStatusStore:
        """Return the legacy ingestion status store singleton."""
        return self._storage_factory.get_ingestion_status_store()

    def get_prompt_template_store(self) -> PromptTemplateStore:
        """Return the legacy prompt template store singleton."""
        return self._storage_factory.get_prompt_template_store()

    def get_main_pointer_store(self) -> MainPointerStore:
        """Return the legacy main pointer store singleton."""
        return self._storage_factory.get_main_pointer_store()

    def get_active_generation_store(self) -> ActiveGenerationStore:
        """Return the legacy active generation store singleton."""
        return self._storage_factory.get_active_generation_store()

    def reset_kb_write_coordinator(self) -> None:
        """Clear legacy storage singletons and coordinator-backed shim state."""
        self._storage_factory.reset_all()
        self.reset_coordinator_caches()

    def reset_rag_storage_for_tests(self) -> None:
        """Reset all process-global KB/RAG storage state for tests."""
        from ..storage.vector_backend import (
            VectorBackend,
            get_configured_vector_backend,
        )

        backend = get_configured_vector_backend()
        if backend is VectorBackend.LANCEDB:
            from xagent.providers.vector_store.lancedb import clear_connection_cache

            clear_connection_cache()
        elif backend is VectorBackend.MILVUS:
            # Future: clear Milvus client pools / connection cache when implemented.
            pass
        elif backend is VectorBackend.QDRANT:
            # Future: clear Qdrant client singleton when implemented.
            pass

        self.reset_kb_write_coordinator()

        from ..management import collection_manager

        collection_manager.reset_locks_for_testing()

    def reset_coordinator_caches(self) -> None:
        """Clear facade-owned caches.

        The storage shim currently delegates directly to the storage factory and
        does not cache handles itself. This hook is intentionally kept as the
        reset point for future coordinator-side operation caches.
        """
