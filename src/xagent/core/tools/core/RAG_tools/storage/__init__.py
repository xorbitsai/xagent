"""Storage contracts and default implementations for KB.

Phase 1A Part 2: Extended with additional store contracts for complete decoupling.
"""

from .contracts import (
    IngestionStatusStore,
    KBWriteCoordinator,
    MainPointerStore,
    MetadataStore,
    PromptTemplateStore,
    VectorIndexStore,
)
from .factory import (
    StorageFactory,
    get_ingestion_status_store,
    get_kb_write_coordinator,
    get_main_pointer_store,
    get_metadata_store,
    get_prompt_template_store,
    get_vector_index_store,
    get_vector_store_raw_connection,
    reset_kb_write_coordinator,
    reset_rag_storage_for_tests,
)
from .vector_backend import (
    VECTOR_BACKEND_ENV,
    VECTOR_BACKEND_ENV_LEGACY,
    VectorBackend,
    get_configured_vector_backend,
)

__all__ = [
    # Contracts
    "KBWriteCoordinator",
    "MetadataStore",
    "VectorIndexStore",
    "IngestionStatusStore",
    "PromptTemplateStore",
    "MainPointerStore",
    # Factory
    "StorageFactory",
    "get_kb_write_coordinator",
    "get_metadata_store",
    "get_vector_index_store",
    "get_vector_store_raw_connection",
    "VectorBackend",
    "VECTOR_BACKEND_ENV",
    "VECTOR_BACKEND_ENV_LEGACY",
    "get_configured_vector_backend",
    "get_ingestion_status_store",
    "get_prompt_template_store",
    "get_main_pointer_store",
    "reset_kb_write_coordinator",
    "reset_rag_storage_for_tests",
]
