from typing import Any, Dict, List, Type

import pytest
import requests

from xagent.core.model.embedding import OpenAIEmbedding

from .test_embedding_base import BaseEmbeddingTest


class TestOpenAIEmbedding(BaseEmbeddingTest):
    """Test OpenAIEmbedding client."""

    def get_client_class(self) -> Type[OpenAIEmbedding]:
        return OpenAIEmbedding

    def get_default_model(self) -> str:
        return "text-embedding-3-small"

    def get_api_key_error_message(self) -> str:
        return "OPENAI_API_KEY is required"

    def get_embedding_error_message(self) -> str:
        return "OpenAI embedding failed"

    def get_mock_response(self, embeddings: List[List[float]]) -> Dict[str, Any]:
        return {"data": [{"embedding": emb} for emb in embeddings]}

    def get_request_session_path(self) -> str:
        return "requests.Session.post"

    def get_init_kwargs(self) -> Dict[str, Any]:
        return {"base_url": "https://api.openai.com/v1/embeddings"}

    def verify_request_payload(
        self, payload: Dict[str, Any], texts: List[str], **kwargs
    ):
        """Verify OpenAI-specific request payload."""
        assert payload["model"] == self.get_default_model()
        assert payload["input"] == texts

        dimension = kwargs.get("dimension")
        if dimension:
            assert payload["dimensions"] == dimension
        else:
            assert "dimensions" not in payload

    def test_base_url_initialization(self):
        """Test base URL initialization."""
        client = OpenAIEmbedding(api_key="test_key", base_url="https://custom.api.com")
        assert client.base_url == "https://custom.api.com"

    def test_encode_surfaces_provider_error_detail(self, mocker):
        mock_response = mocker.Mock()
        mock_response.status_code = 500
        mock_response.text = '{"detail": "upstream exploded"}'
        mock_response.json.return_value = {
            "detail": "upstream exploded",
        }
        mock_response.raise_for_status.side_effect = requests.HTTPError(
            "500 Server Error", response=mock_response
        )

        mocker.patch("requests.Session.post", return_value=mock_response)

        client = OpenAIEmbedding(api_key="test_key")

        with pytest.raises(RuntimeError, match="upstream exploded"):
            client.encode("Hello")

    def test_encode_allows_custom_base_url_without_api_key(self, mocker):
        mock_response = mocker.Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2]}]}

        post_mock = mocker.patch("requests.Session.post", return_value=mock_response)

        client = OpenAIEmbedding(
            api_key=None,
            base_url="http://localhost:9997/v1",
            model="test-embedding-model",
        )

        embedding = client.encode("Hello")

        assert embedding == [0.1, 0.2]
        post_mock.assert_called_once()
        assert "Authorization" not in client._get_session().headers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
