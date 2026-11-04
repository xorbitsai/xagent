import shutil
import tempfile

import pytest

from xagent.core.memory.core import MemoryNote
from xagent.core.memory.lancedb import LanceDBMemoryStore
from xagent.core.model.embedding import DashScopeEmbedding
from xagent.core.model.model import EmbeddingModelConfig


class TestMemoryStoreWithModelConfig:
    """Test LanceDBMemoryStore with EmbeddingModel configuration."""

    @pytest.fixture
    def temp_db_dir(self):
        """Create a temporary directory for the database."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_memory_store_with_embedding_model_config(self, temp_db_dir):
        """Test that LanceDBMemoryStore can accept EmbeddingModel configuration."""
        # Create EmbeddingModel configuration
        model_config = EmbeddingModelConfig(
            id="test_embedding",
            model_name="text-embedding-v4",
            api_key="test_key",
            dimension=512,
            instruct="test instruction",
        )

        # Create memory store with config
        memory_store = LanceDBMemoryStore(
            db_dir=temp_db_dir,
            collection_name="test_memories",
            embedding_model=model_config,
        )

        # Verify the embedding model was created correctly
        assert hasattr(memory_store, "_embedding_model")
        assert hasattr(memory_store._embedding_model, "encode")
        assert hasattr(memory_store._embedding_model, "get_dimension")
        assert hasattr(memory_store._embedding_model, "abilities")

    def test_memory_store_with_base_embedding_instance(self, temp_db_dir):
        """Test that LanceDBMemoryStore can accept BaseEmbedding instance."""
        # Create BaseEmbedding instance
        embedding_instance = DashScopeEmbedding(api_key="test_key", dimension=256)

        # Create memory store with instance
        memory_store = LanceDBMemoryStore(
            db_dir=temp_db_dir,
            collection_name="test_memories",
            embedding_model=embedding_instance,
        )

        # Verify the embedding model was set correctly
        assert memory_store._embedding_model is embedding_instance
        assert memory_store._embedding_model.get_dimension() == 256

    def test_memory_store_with_no_embedding_model(self, temp_db_dir):
        """Test that LanceDBMemoryStore works with no embedding model (creates default)."""
        # Create memory store without embedding model
        memory_store = LanceDBMemoryStore(
            db_dir=temp_db_dir, collection_name="test_memories"
        )

        # Verify default embedding model was created
        assert hasattr(memory_store, "_embedding_model")
        # The embedding model might be None if no default is available
        if memory_store._embedding_model is not None:
            assert hasattr(memory_store._embedding_model, "encode")
            assert hasattr(memory_store._embedding_model, "get_dimension")

    def test_memory_store_with_invalid_embedding_model_provider(self, temp_db_dir):
        """Test that LanceDBMemoryStore raises error for invalid embedding model types."""
        # Create memory store with invalid embedding model type
        with pytest.raises(ValueError, match="Unsupported embedding model type"):
            LanceDBMemoryStore(
                db_dir=temp_db_dir,
                collection_name="test_memories",
                embedding_model="invalid_type",
            )

    def test_memory_store_functionality_with_config(self, temp_db_dir):
        """Test that memory store works correctly with EmbeddingModel configuration."""
        # Create EmbeddingModel configuration with mock embedding
        model_config = EmbeddingModelConfig(
            id="test_embedding",
            model_name="text-embedding-v4",
            api_key="test_key",
            dimension=64,
        )

        # Create memory store
        memory_store = LanceDBMemoryStore(
            db_dir=temp_db_dir,
            collection_name="test_memories",
            embedding_model=model_config,
        )

        # Test basic functionality
        note = MemoryNote(content="Test memory")
        response = memory_store.add(note)
        assert response.success
        assert response.memory_id is not None

        # Retrieve memory
        get_response = memory_store.get(response.memory_id)
        assert get_response.success
        assert get_response.content.content == "Test memory"
