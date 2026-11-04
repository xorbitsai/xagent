from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type
from unittest.mock import Mock, patch

import pytest

from xagent.core.model.embedding.base import BaseEmbedding


class BaseEmbeddingTest(ABC):
    """Base test class for embedding model implementations."""

    @abstractmethod
    def get_client_class(self) -> Type[BaseEmbedding]:
        """Return the embedding client class to test."""
        pass

    @abstractmethod
    def get_default_model(self) -> str:
        """Return the default model name."""
        pass

    @abstractmethod
    def get_api_key_error_message(self) -> str:
        """Return the expected error message when API key is missing."""
        pass

    @abstractmethod
    def get_embedding_error_message(self) -> str:
        """Return the expected error message when embedding fails."""
        pass

    @abstractmethod
    def get_mock_response(self, embeddings: List[List[float]]) -> Dict[str, Any]:
        """Return a mock API response with the given embeddings."""
        pass

    @abstractmethod
    def get_request_session_path(self) -> str:
        """Return the path to patch for requests.Session.post."""
        pass

    def get_init_kwargs(self) -> Dict[str, Any]:
        """Return additional initialization kwargs (override if needed)."""
        return {}

    def verify_request_payload(
        self, payload: Dict[str, Any], texts: List[str], **kwargs
    ):
        """Verify the request payload (override for custom verification)."""
        pass

    def test_initialization(self):
        """Test client initialization."""
        client_class = self.get_client_class()
        init_kwargs = self.get_init_kwargs()

        client = client_class(
            model=self.get_default_model(),
            api_key="test_key",
            dimension=1024,
            **init_kwargs,
        )

        assert client.model == self.get_default_model()
        assert client.api_key == "test_key"
        assert client.dimension == 1024

    def test_default_initialization(self):
        """Test client initialization with defaults."""
        client_class = self.get_client_class()
        client = client_class(api_key="test_key")

        assert client.model == self.get_default_model()
        assert client.dimension is None

    def test_get_dimension(self):
        """Test getting embedding dimension."""
        client_class = self.get_client_class()
        client = client_class(api_key="test_key", dimension=512)
        assert client.get_dimension() == 512

    def test_abilities(self):
        """Test getting model abilities."""
        client_class = self.get_client_class()
        client = client_class(api_key="test_key")
        assert client.abilities == ["embed"]

    def test_missing_api_key(self):
        """Test error when API key is missing."""
        client_class = self.get_client_class()
        client = client_class()

        with pytest.raises(RuntimeError, match=self.get_api_key_error_message()):
            client.encode("Hello")

    @patch("requests.Session.post")
    def test_encode_single_text_success(self, mock_post):
        """Test successful encoding of single text."""
        # Mock response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = self.get_mock_response([[0.1, 0.2, 0.3, 0.4]])
        mock_post.return_value = mock_response

        client_class = self.get_client_class()
        client = client_class(api_key="test_key", dimension=4)
        embedding = client.encode("Hello world")

        assert embedding == [0.1, 0.2, 0.3, 0.4]

        # Verify request
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.verify_request_payload(call_args[1]["json"], ["Hello world"], dimension=4)

    @patch("requests.Session.post")
    def test_encode_batch_texts_success(self, mock_post):
        """Test successful encoding of multiple texts."""
        # Mock response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = self.get_mock_response(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        )
        mock_post.return_value = mock_response

        client_class = self.get_client_class()
        client = client_class(api_key="test_key", dimension=2)
        embeddings = client.encode(["Hello", "World", "Test"])

        assert embeddings == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        # Verify request
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.verify_request_payload(
            call_args[1]["json"], ["Hello", "World", "Test"], dimension=2
        )

    @patch("requests.Session.post")
    def test_encode_without_dimension(self, mock_post):
        """Test encoding without dimension parameter."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = self.get_mock_response([[0.1, 0.2]])
        mock_post.return_value = mock_response

        client_class = self.get_client_class()
        client = client_class(api_key="test_key")
        client.encode("Hello")

        mock_post.assert_called_once()

    @patch("requests.Session.post")
    def test_encode_with_override_dimension(self, mock_post):
        """Test encoding with overridden dimension."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = self.get_mock_response([[0.1, 0.2]])
        mock_post.return_value = mock_response

        client_class = self.get_client_class()
        client = client_class(api_key="test_key", dimension=4)
        client.encode("Hello", dimension=2)

        # Verify overridden dimension
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.verify_request_payload(call_args[1]["json"], ["Hello"], dimension=2)

    @patch("requests.Session.post")
    def test_encode_api_error(self, mock_post):
        """Test handling of API errors."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_post.return_value = mock_response

        client_class = self.get_client_class()
        client = client_class(api_key="test_key")

        with pytest.raises(RuntimeError, match=self.get_embedding_error_message()):
            client.encode("Hello")

    @patch("requests.Session.post")
    def test_encode_invalid_response(self, mock_post):
        """Test handling of invalid response format."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"invalid": "response"}
        mock_post.return_value = mock_response

        client_class = self.get_client_class()
        client = client_class(api_key="test_key")

        with pytest.raises(RuntimeError, match=self.get_embedding_error_message()):
            client.encode("Hello")
