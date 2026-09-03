from __future__ import annotations

from typing import Any, List, Optional, Union

import requests

from .base import BaseEmbedding


class DashScopeEmbedding(BaseEmbedding):
    """
    DashScope text embedding model client.
    Supports text embedding using the DashScope text-embedding-v4 API.
    """

    def __init__(
        self,
        model: str = "text-embedding-v4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dimension: Optional[int] = None,
        instruct: Optional[str] = None,
    ):
        """
        Initialize DashScope embedding client.

        Args:
            model: Model name (default: text-embedding-v4)
            api_key: DashScope API key (or set DASHSCOPE_API_KEY env var)
            base_url: API base URL (defaults to DashScope embedding endpoint)
            dimension: Embedding dimension (default: None)
            instruct: Optional instruction for embedding context
        """
        self.model = model
        self.api_key = api_key
        self.base_url = (
            base_url
            or "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
        )
        self.dimension = dimension
        self.instruct = instruct
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

    # DashScope API batch size limit (API says "should not be larger than 10")
    MAX_BATCH_SIZE = 10

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
            instruct: Override default instruction

        Returns:
            Single embedding vector (list of floats) for single text,
            or list of embedding vectors for list of texts

        Raises:
            RuntimeError: If API call fails or returns invalid response
        """
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is required")

        session = self._get_session()

        # Handle single text vs batch
        if isinstance(text, str):
            texts = [text]
            single_input = True
        else:
            texts = text
            single_input = False

        # Split into batches if exceeding API limit
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.MAX_BATCH_SIZE):
            batch_texts = texts[i : i + self.MAX_BATCH_SIZE]

            # Prepare request payload for this batch
            payload: dict[str, Any] = {
                "model": self.model,
                "input": {"texts": batch_texts},
                "output_type": "dense",
                "parameters": {},
            }

            final_dimension = dimension or self.dimension
            if final_dimension is not None:
                payload["parameters"]["dimension"] = final_dimension

            # Add instruction if provided
            final_instruct = instruct or self.instruct
            if final_instruct:
                payload["parameters"]["instruct"] = final_instruct

            try:
                response = session.post(self.base_url, json=payload)
                response.raise_for_status()

                data = response.json()

                if "output" not in data or "embeddings" not in data["output"]:
                    raise ValueError(f"Unexpected response format: {data}")

                batch_embeddings = data["output"]["embeddings"]
                all_embeddings.extend([emb["embedding"] for emb in batch_embeddings])

            except Exception as e:
                raise RuntimeError(
                    f"DashScope embedding failed (batch {i // self.MAX_BATCH_SIZE + 1}): {str(e)}"
                )

        # Return single embedding or list based on input type
        if single_input:
            return all_embeddings[0]
        else:
            return all_embeddings

    def get_dimension(self) -> Optional[int]:
        """Get the embedding dimension."""
        return self.dimension or 1024

    @property
    def abilities(self) -> List[str]:
        """Get the list of abilities supported by this model."""
        return ["embed"]
