from __future__ import annotations

from typing import Any, List, Optional, Union

import requests

from .base import BaseEmbedding


class OpenAIEmbedding(BaseEmbedding):
    """
    OpenAI text embedding model client.
    Supports text embedding using the OpenAI embeddings API.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dimension: Optional[int] = None,
    ):
        """
        Initialize OpenAI embedding client.

        Args:
            model: Model name (default: text-embedding-3-small)
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            base_url: API base URL
            dimension: Optional embedding dimension (for models that support it)
        """
        self.model = model
        self.api_key = api_key

        # Ensure base_url ends with /embeddings for OpenAI-compatible APIs
        if base_url:
            # First, strip trailing slashes for consistent checking
            clean_base_url = base_url.rstrip("/")

            # If base_url doesn't end with /embeddings, append it
            if not clean_base_url.endswith("/embeddings"):
                # Check if it ends with /v1 or similar
                if clean_base_url.endswith("/v1"):
                    self.base_url = clean_base_url + "/embeddings"
                else:
                    # For other cases, just use as-is (might be custom endpoint)
                    self.base_url = base_url
            else:
                # Already has /embeddings, use the cleaned version
                self.base_url = clean_base_url
        else:
            self.base_url = "https://api.openai.com/v1/embeddings"

        self.dimension = dimension
        self._session: Optional[requests.Session] = None

    def _get_session(self) -> requests.Session:
        """Get or create HTTP session."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
            )
        return self._session

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
            instruct: Unused for OpenAI embeddings

        Returns:
            Single embedding vector (list of floats) for single text,
            or list of embedding vectors for list of texts

        Raises:
            RuntimeError: If API call fails or returns invalid response
        """
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required")

        session = self._get_session()

        # Handle single text vs batch
        if isinstance(text, str):
            texts = [text]
            single_input = True
        else:
            texts = text
            single_input = False

        # Prepare request payload
        payload: dict[str, Any] = {
            "model": self.model,
            "input": texts,
        }

        # Add dimension if provided
        final_dimension = dimension or self.dimension
        if final_dimension:
            payload["dimensions"] = final_dimension

        try:
            response = session.post(self.base_url or "", json=payload)
            response.raise_for_status()

            data = response.json()

            if "data" not in data:
                raise ValueError(f"Unexpected response format: {data}")

            embeddings = data["data"]

            # Extract embedding vectors
            if single_input:
                embedding: list[float] = embeddings[0]["embedding"]
                return embedding
            else:
                embedding_list: list[list[float]] = [
                    emb["embedding"] for emb in embeddings
                ]
                return embedding_list

        except Exception as e:
            import traceback

            raise RuntimeError(
                f"OpenAI embedding failed: {str(e)}\n{traceback.format_exc()}"
            )

    def get_dimension(self) -> Optional[int]:
        """Get the embedding dimension."""
        return self.dimension

    @property
    def abilities(self) -> List[str]:
        """Get the list of abilities supported by this model."""
        return ["embed"]
