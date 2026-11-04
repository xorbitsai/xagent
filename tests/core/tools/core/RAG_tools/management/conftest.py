"""Pytest configuration and shared fixtures for collection management tests."""

import os
import tempfile
from typing import Any, Generator

import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo
from xagent.core.tools.core.RAG_tools.management.collection_manager import (
    CollectionManager,
)
from xagent.core.tools.core.RAG_tools.storage.factory import (
    get_vector_index_store,
    reset_rag_storage_for_tests,
)


@pytest.fixture
def temp_lancedb_dir() -> Generator[str, None, None]:
    """Create a temporary directory for LanceDB test data.

    The directory is cleaned up after the test.

    Yields:
        Path to temporary LanceDB directory
    """
    tmpdir = tempfile.mkdtemp()
    old_env = os.environ.get("LANCEDB_DIR")

    try:
        # Set environment variable for this test
        os.environ["LANCEDB_DIR"] = os.path.join(tmpdir, ".lancedb")

        # Reset storage (vector backend caches + factory) for clean state
        reset_rag_storage_for_tests()

        yield tmpdir
    finally:
        # Cleanup
        reset_rag_storage_for_tests()

        # Restore old environment
        if old_env is not None:
            os.environ["LANCEDB_DIR"] = old_env
        elif "LANCEDB_DIR" in os.environ:
            del os.environ["LANCEDB_DIR"]

        # Remove temp directory
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
async def real_store(temp_lancedb_dir: str) -> Any:
    """Create a real LanceDB metadata store for integration testing.

    This fixture provides an actual storage implementation rather than a mock,
    allowing tests to verify the complete data flow from CollectionManager
    through the storage layer.

    Args:
        temp_lancedb_dir: Temporary directory from temp_lancedb_dir fixture

    Yields:
        Real metadata store instance
    """
    from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
        LanceDBMetadataStore,
    )

    vector_store = get_vector_index_store()
    conn = vector_store.get_raw_connection()

    # Ensure metadata table exists
    try:
        conn.create_table(
            "collection_metadata",
            schema=LanceDBMetadataStore.get_schema(),
        )
    except Exception:
        # Table already exists
        pass

    store = LanceDBMetadataStore(conn=conn)
    yield store


@pytest.fixture
async def manager_with_real_store(real_store: Any) -> CollectionManager:
    """Create a CollectionManager with real storage backend.

    This fixture replaces the mock-based approach, allowing tests to verify
    actual data persistence and retrieval.

    Args:
        real_store: Real metadata store from real_store fixture

    Yields:
        CollectionManager instance with real storage
    """
    manager = CollectionManager()
    manager._metadata_store = real_store
    return manager


@pytest.fixture
def sample_collection() -> CollectionInfo:
    """Create a sample CollectionInfo for testing.

    Returns:
        CollectionInfo instance with test data
    """
    return CollectionInfo(
        name="test_collection",
        embedding_model_id="text-embedding-ada-002",
        embedding_dimension=1536,
        documents=5,
        processed_documents=3,
        document_names=["doc1.pdf", "doc2.md"],
    )
