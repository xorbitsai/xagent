"""Tests for the KB storage shim compatibility facade."""

from __future__ import annotations

from inspect import signature


class _FakeVectorStore:
    def __init__(self) -> None:
        self.raw_connection = object()

    def get_raw_connection(self) -> object:
        return self.raw_connection


class _FakeStorageFactory:
    def __init__(self) -> None:
        self.coordinator = object()
        self.metadata_store = object()
        self.vector_index_store = _FakeVectorStore()
        self.ingestion_status_store = object()
        self.prompt_template_store = object()
        self.main_pointer_store = object()
        self.reset_count = 0

    def get_kb_write_coordinator(self) -> object:
        return self.coordinator

    def get_metadata_store(self) -> object:
        return self.metadata_store

    def get_vector_index_store(self) -> _FakeVectorStore:
        return self.vector_index_store

    def get_ingestion_status_store(self) -> object:
        return self.ingestion_status_store

    def get_prompt_template_store(self) -> object:
        return self.prompt_template_store

    def get_main_pointer_store(self) -> object:
        return self.main_pointer_store

    def reset_all(self) -> None:
        self.reset_count += 1


def test_storage_public_import_surface_retains_low_level_exports() -> None:
    """Given legacy callers, every retained storage export remains importable."""
    import xagent.core.tools.core.RAG_tools.storage as storage
    from xagent.core.tools.core.RAG_tools.storage import contracts, vector_backend
    from xagent.core.tools.core.RAG_tools.storage.factory import StorageFactory

    expected_symbols = [
        "KBWriteCoordinator",
        "MetadataStore",
        "VectorIndexStore",
        "IngestionStatusStore",
        "PromptTemplateStore",
        "MainPointerStore",
        "StorageFactory",
        "get_kb_write_coordinator",
        "get_metadata_store",
        "get_vector_index_store",
        "get_vector_store_raw_connection",
        "get_ingestion_status_store",
        "get_prompt_template_store",
        "get_main_pointer_store",
        "reset_kb_write_coordinator",
        "reset_rag_storage_for_tests",
        "VectorBackend",
        "VECTOR_BACKEND_ENV",
        "VECTOR_BACKEND_ENV_LEGACY",
        "get_configured_vector_backend",
    ]

    for symbol in expected_symbols:
        assert hasattr(storage, symbol)

    assert storage.KBWriteCoordinator is contracts.KBWriteCoordinator
    assert storage.MetadataStore is contracts.MetadataStore
    assert storage.VectorIndexStore is contracts.VectorIndexStore
    assert storage.IngestionStatusStore is contracts.IngestionStatusStore
    assert storage.PromptTemplateStore is contracts.PromptTemplateStore
    assert storage.MainPointerStore is contracts.MainPointerStore
    assert storage.StorageFactory is StorageFactory
    assert storage.VectorBackend is vector_backend.VectorBackend
    assert storage.VECTOR_BACKEND_ENV is vector_backend.VECTOR_BACKEND_ENV
    assert storage.VECTOR_BACKEND_ENV_LEGACY is vector_backend.VECTOR_BACKEND_ENV_LEGACY
    assert storage.get_configured_vector_backend is (
        vector_backend.get_configured_vector_backend
    )


def test_storage_shim_public_methods_match_legacy_function_signatures() -> None:
    """Given legacy storage functions, the facade exposes matching sync methods."""
    from xagent.core.tools.core.RAG_tools.kb import KBStorageShimCompatibilityFacade
    from xagent.core.tools.core.RAG_tools.storage import factory

    shim = KBStorageShimCompatibilityFacade(storage_factory=_FakeStorageFactory())  # type: ignore[arg-type]
    method_names = [
        "get_kb_write_coordinator",
        "get_metadata_store",
        "get_vector_index_store",
        "get_vector_store_raw_connection",
        "get_ingestion_status_store",
        "get_prompt_template_store",
        "get_main_pointer_store",
        "reset_kb_write_coordinator",
        "reset_rag_storage_for_tests",
    ]

    for name in method_names:
        assert signature(getattr(shim, name)) == signature(getattr(factory, name))


def test_storage_shim_delegates_to_injected_factory() -> None:
    """Given an injected factory, the shim preserves object identity."""
    from xagent.core.tools.core.RAG_tools.kb import KBStorageShimCompatibilityFacade

    fake_factory = _FakeStorageFactory()
    shim = KBStorageShimCompatibilityFacade(storage_factory=fake_factory)  # type: ignore[arg-type]

    assert shim.get_kb_write_coordinator() is fake_factory.coordinator
    assert shim.get_metadata_store() is fake_factory.metadata_store
    assert shim.get_vector_index_store() is fake_factory.vector_index_store
    assert (
        shim.get_vector_store_raw_connection()
        is fake_factory.vector_index_store.raw_connection
    )
    assert shim.get_ingestion_status_store() is fake_factory.ingestion_status_store
    assert shim.get_prompt_template_store() is fake_factory.prompt_template_store
    assert shim.get_main_pointer_store() is fake_factory.main_pointer_store


def test_storage_shim_reset_kb_write_coordinator_resets_factory_state() -> None:
    """Given shim reset, the legacy factory reset path is used."""
    from xagent.core.tools.core.RAG_tools.kb import KBStorageShimCompatibilityFacade

    fake_factory = _FakeStorageFactory()
    shim = KBStorageShimCompatibilityFacade(storage_factory=fake_factory)  # type: ignore[arg-type]

    shim.reset_kb_write_coordinator()

    assert fake_factory.reset_count == 1


def test_legacy_storage_functions_delegate_through_coordinator_shim(
    monkeypatch,
) -> None:
    """Given legacy module functions, calls flow through the coordinator shim."""
    from xagent.core.tools.core.RAG_tools.storage import factory

    fake_factory = _FakeStorageFactory()
    shim = factory._get_storage_shim()
    monkeypatch.setattr(factory, "_get_storage_shim", lambda: shim)
    monkeypatch.setattr(shim, "_storage_factory", fake_factory)

    assert factory.get_kb_write_coordinator() is fake_factory.coordinator
    assert factory.get_metadata_store() is fake_factory.metadata_store
    assert factory.get_vector_index_store() is fake_factory.vector_index_store
    assert (
        factory.get_vector_store_raw_connection()
        is fake_factory.vector_index_store.raw_connection
    )
    assert factory.get_ingestion_status_store() is fake_factory.ingestion_status_store
    assert factory.get_prompt_template_store() is fake_factory.prompt_template_store
    assert factory.get_main_pointer_store() is fake_factory.main_pointer_store


def test_reset_rag_storage_for_tests_resets_storage_and_coordinator_shim() -> None:
    """Given test reset, old stores and coordinator facade state are both cleared."""
    from xagent.core.tools.core.RAG_tools.kb import get_kb_coordinator
    from xagent.core.tools.core.RAG_tools.storage import (
        get_metadata_store,
        get_vector_index_store,
        reset_rag_storage_for_tests,
    )

    first_coordinator = get_kb_coordinator()
    first_shim = first_coordinator.storage_shim
    first_metadata_store = get_metadata_store()
    first_vector_index_store = get_vector_index_store()

    reset_rag_storage_for_tests()

    second_coordinator = get_kb_coordinator()
    second_shim = second_coordinator.storage_shim
    second_metadata_store = get_metadata_store()
    second_vector_index_store = get_vector_index_store()

    assert second_coordinator is not first_coordinator
    assert second_shim is not first_shim
    assert second_metadata_store is not first_metadata_store
    assert second_vector_index_store is not first_vector_index_store
    assert second_shim.get_metadata_store() is second_metadata_store
    assert second_shim.get_vector_index_store() is second_vector_index_store
