from xagent.core.model.embedding import (
    BaseEmbedding,
    DashScopeEmbedding,
    OpenAIEmbedding,
)


class TestAbstractEmbeddingInterface:
    """Test the abstract embedding interface."""

    def test_dashscope_implements_base_embedding(self):
        """Test that DashScopeEmbedding implements BaseEmbedding."""
        embedding = DashScopeEmbedding(api_key="test_key")
        assert isinstance(embedding, BaseEmbedding)
        assert hasattr(embedding, "encode")
        assert hasattr(embedding, "get_dimension")
        assert hasattr(embedding, "abilities")

    def test_openai_implements_base_embedding(self):
        """Test that OpenAIEmbedding implements BaseEmbedding."""
        embedding = OpenAIEmbedding(api_key="test_key")
        assert isinstance(embedding, BaseEmbedding)
        assert hasattr(embedding, "encode")
        assert hasattr(embedding, "get_dimension")
        assert hasattr(embedding, "abilities")
