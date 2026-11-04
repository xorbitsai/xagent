"""Tests for storage factory and coordinator wiring."""

from xagent.core.tools.core.RAG_tools.storage import factory


def test_factory_is_singleton(monkeypatch) -> None:
    """Factory should return the same instance per process."""
    # Get existing factory and reset for test isolation
    try:
        f = factory.StorageFactory.get_factory()
        f.reset_all()
    except Exception:
        factory.StorageFactory._instance = None
        f = factory.StorageFactory.get_factory()

    first = factory.StorageFactory.get_factory()
    second = factory.StorageFactory.get_factory()

    assert first is second


def test_factory_reset_all(monkeypatch) -> None:
    """Factory reset_all should clear all store instances."""
    # Get existing factory and reset for test isolation
    try:
        f = factory.StorageFactory.get_factory()
        f.reset_all()
    except Exception:
        factory.StorageFactory._instance = None
        f = factory.StorageFactory.get_factory()

    # Create some stores
    f.get_vector_index_store()
    f.get_metadata_store()
    f.get_ingestion_status_store()

    # Reset
    f.reset_all()

    # Verify all stores are reset
    assert f._vector_index_store is None
    assert f._metadata_store is None
    assert f._ingestion_status_store is None


def test_convenience_functions_use_factory(monkeypatch) -> None:
    """Convenience functions should delegate to the singleton factory."""
    # Get existing factory and reset for test isolation
    try:
        f = factory.StorageFactory.get_factory()
        f.reset_all()
    except Exception:
        factory.StorageFactory._instance = None
        f = factory.StorageFactory.get_factory()

    first_vector = factory.get_vector_index_store()
    first_metadata = factory.get_metadata_store()

    # Get via factory directly
    second_vector = f.get_vector_index_store()
    second_metadata = f.get_metadata_store()

    assert first_vector is second_vector
    assert first_metadata is second_metadata


def test_coordinator_uses_factory_stores(monkeypatch) -> None:
    """Coordinator should use stores from the factory."""
    # Get existing factory or create new one
    try:
        f = factory.StorageFactory.get_factory()
        # Reset for test isolation
        f.reset_all()
    except Exception:
        # If factory is in bad state, reset singleton
        factory.StorageFactory._instance = None
        f = factory.StorageFactory.get_factory()

    coordinator = factory.get_kb_write_coordinator()

    assert coordinator.metadata_store() is f.get_metadata_store()
    assert coordinator.vector_index_store() is f.get_vector_index_store()


def test_reset_rag_storage_for_tests_is_idempotent(monkeypatch) -> None:
    """Unified test reset entry should run without error (default LanceDB backend)."""
    factory.reset_rag_storage_for_tests()
    factory.reset_rag_storage_for_tests()
