"""Xinference Embedding provider implementation."""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Union

from ..public_xinference import PublicXinferenceClient as XinferenceClient
from ..retrieval_metering import record_embedding_usage, reported_tokens
from .base import BaseEmbedding

logger = logging.getLogger(__name__)


class XinferenceEmbedding(BaseEmbedding):
    """
    Xinference embedding model client using the xinference-client SDK.
    Supports text embedding using Xinference's embedding models.
    """

    def __init__(
        self,
        model: str = "bge-base-en-v1.5",
        model_uid: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        dimension: Optional[int] = None,
        model_id: str = "",
    ):
        """
        Initialize Xinference embedding client.

        Args:
            model: Model name (e.g., "bge-base-en-v1.5")
            model_uid: Unique model UID in Xinference (if model is already launched)
            base_url: Xinference server base URL (e.g., "http://localhost:9997")
            api_key: Optional API key for authentication
            dimension: Optional embedding dimension
            model_id: Configured model id recorded on usage rows
        """
        self.model = model
        self.model_id = model_id
        self._model_uid = model_uid or model
        self.base_url = (base_url or "http://localhost:9997").rstrip("/")
        self.api_key = api_key
        self.dimension = dimension

        # Initialize the Xinference client (lazy initialization)
        self._client: Optional[XinferenceClient] = None
        self._model_handle: Optional[Any] = None

    def _get_session(self) -> XinferenceClient:
        """Get or create Xinference client."""
        if self._client is None:
            self._client = XinferenceClient(
                base_url=self.base_url, api_key=self.api_key
            )
        return self._client

    def _ensure_model_handle(self) -> Any:
        """Ensure the embedding model handle is initialized."""
        if self._model_handle is None:
            client = self._get_session()
            # Get the model handle (assumes model is already launched on the server)
            self._model_handle = client.get_model(self._model_uid)
        return self._model_handle

    def encode(
        self,
        text: Union[str, List[str]],
        dimension: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Union[List[float], List[List[float]]]:
        """
        Encode text into embedding vector(s).

        Args:
            text: Single text string or list of text strings
            dimension: Override default embedding dimension
            instruct: Instruction for embedding (not used by Xinference)

        Returns:
            Single embedding vector (list of floats) for single text,
            or list of embedding vectors for list of texts

        Raises:
            RuntimeError: If API call fails or returns invalid response
        """
        model_handle = self._ensure_model_handle()

        # Handle single text vs batch
        if isinstance(text, str):
            single_input = True
            texts = [text]
        else:
            single_input = False
            texts = text

        try:
            # Create embeddings
            embedding_result = model_handle.create_embedding(input=texts)
            # The SDK raises on a non-2xx reply, so a returned result is a
            # billed call; record it before its shape is checked below.
            record_embedding_usage(
                self,
                len(texts),
                reported_tokens(
                    embedding_result,
                    ("usage", "prompt_tokens"),
                    ("usage", "total_tokens"),
                ),
            )

            # Xinference returns an Embedding object with data field
            if hasattr(embedding_result, "data"):
                embeddings_data = embedding_result.data
            elif isinstance(embedding_result, dict):
                embeddings_data = embedding_result.get("data", [])
            else:
                embeddings_data = embedding_result

            if not embeddings_data:
                raise RuntimeError(f"Empty embedding response: {embedding_result}")

            # Extract embedding vectors
            if single_input:
                if hasattr(embeddings_data[0], "embedding"):
                    embedding: List[float] = embeddings_data[0].embedding
                else:
                    embedding = embeddings_data[0]
                return embedding
            else:
                if hasattr(embeddings_data[0], "embedding"):
                    embedding_list: List[List[float]] = [
                        emb.embedding for emb in embeddings_data
                    ]
                else:
                    embedding_list = embeddings_data
                return embedding_list

        except Exception as e:
            logger.error(f"Xinference embedding failed: {e}")
            raise RuntimeError(f"Xinference embedding failed: {str(e)}") from e

    def get_dimension(self) -> Optional[int]:
        """Get the embedding dimension."""
        return self.dimension

    @property
    def abilities(self) -> List[str]:
        """Get the list of abilities supported by this model."""
        return ["embed"]

    def close(self) -> None:
        """Close the Xinference client and cleanup resources."""
        if self._model_handle is not None:
            try:
                self._model_handle.close()
            except Exception:
                pass
            self._model_handle = None

        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def __enter__(self) -> "XinferenceEmbedding":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    @staticmethod
    def list_available_models(
        base_url: str, api_key: Optional[str] = None
    ) -> List[dict[str, Any]]:
        """Fetch available embedding models from Xinference server.

        Args:
            base_url: Xinference server base URL
            api_key: Optional API key for authentication

        Returns:
            List of available embedding models with their information

        Example:
            >>> models = XinferenceEmbedding.list_available_models(
            ...     base_url="http://localhost:9997"
            ... )
        """
        client = XinferenceClient(base_url=base_url, api_key=api_key)

        try:
            # Get list of running models
            # list_models returns Dict[str, Dict[str, Any]] where key is model_uid
            models_dict = client.list_models()

            # Filter for embedding models
            result = []
            for model_uid, model_info in models_dict.items():
                if model_info.get("model_type") == "embedding":
                    result.append(
                        {
                            "id": model_info.get("model_name", model_uid),
                            "model_uid": model_uid,
                            "model_type": model_info.get("model_type", ""),
                            "model_ability": model_info.get("model_ability", []),
                            "description": model_info.get("model_description", ""),
                        }
                    )

            return result

        except Exception as e:
            logger.error(f"Failed to fetch embedding models from Xinference: {e}")
            return []

        finally:
            client.close()
