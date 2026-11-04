from typing import Any, Dict, List, Type

import pytest

from xagent.core.model.embedding import DashScopeEmbedding

from .test_embedding_base import BaseEmbeddingTest


class TestDashScopeEmbedding(BaseEmbeddingTest):
    """Test DashScopeEmbedding client."""

    def get_client_class(self) -> Type[DashScopeEmbedding]:
        return DashScopeEmbedding

    def get_default_model(self) -> str:
        return "text-embedding-v4"

    def get_api_key_error_message(self) -> str:
        return "DASHSCOPE_API_KEY is required"

    def get_embedding_error_message(self) -> str:
        return "DashScope embedding failed"

    def get_mock_response(self, embeddings: List[List[float]]) -> Dict[str, Any]:
        return {"output": {"embeddings": [{"embedding": emb} for emb in embeddings]}}

    def get_request_session_path(self) -> str:
        return "requests.Session.post"

    def get_init_kwargs(self) -> Dict[str, Any]:
        return {"instruct": "Test instruction"}

    def verify_request_payload(
        self, payload: Dict[str, Any], texts: List[str], **kwargs
    ):
        """Verify DashScope-specific request payload."""
        assert payload["model"] == self.get_default_model()
        assert payload["input"]["texts"] == texts
        assert payload["output_type"] == "dense"

        dimension = kwargs.get("dimension")
        if dimension:
            assert payload["parameters"]["dimension"] == dimension

    def test_instruction_parameter(self):
        """Test encoding with instruction parameter."""
        from unittest.mock import Mock, patch

        with patch("requests.Session.post") as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = self.get_mock_response([[0.1, 0.2]])
            mock_post.return_value = mock_response

            client = DashScopeEmbedding(
                api_key="test_key", dimension=2, instruct="Test instruction"
            )
            client.encode("Hello")

            # Verify instruction is included
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[1]["json"]["parameters"]["instruct"] == "Test instruction"

    def test_override_instruction(self):
        """Test encoding with overridden instruction."""
        from unittest.mock import Mock, patch

        with patch("requests.Session.post") as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = self.get_mock_response([[0.1, 0.2]])
            mock_post.return_value = mock_response

            client = DashScopeEmbedding(api_key="test_key", instruct="Default")
            client.encode("Hello", instruct="Override instruction")

            # Verify overridden instruction
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert (
                call_args[1]["json"]["parameters"]["instruct"] == "Override instruction"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
