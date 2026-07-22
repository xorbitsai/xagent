from typing import List, Optional, Union

import requests

from ...retry import create_retry_wrapper
from ..model import EmbeddingModelConfig
from .base import BaseEmbedding
from .dashscope import DashScopeEmbedding
from .openai import OpenAIEmbedding
from .xinference import XinferenceEmbedding


def retry_on(e: Exception) -> bool:
    ERRORS = requests.exceptions.Timeout

    if isinstance(e, requests.exceptions.HTTPError):
        status_code = e.response.status_code
        return status_code == 429 or 500 <= status_code < 600  # 429 and 5xx
    return isinstance(e, ERRORS)


def create_embedding_adapter(model_config: EmbeddingModelConfig) -> BaseEmbedding:
    """
    Creates a custom BaseEmbedding instance from an EmbeddingModelConfig.
    """
    embedding = EmbeddingModelAdapter(model_config)

    return create_retry_wrapper(
        embedding,
        BaseEmbedding,  # type: ignore[type-abstract]
        retry_methods={"encode"},
        max_retries=model_config.max_retries,
        retry_on=retry_on,
    )


class EmbeddingModelAdapter(BaseEmbedding):
    """Adapter that makes the new embedding interface compatible with existing EmbeddingModel configs."""

    def __init__(self, model_config: EmbeddingModelConfig):
        self.model_config = model_config
        self._embedding_model = self._create_embedding_model()

    def _create_embedding_model(self) -> BaseEmbedding:
        """Create the actual embedding model from configuration."""
        # Normalize provider name: map variants to canonical names
        # Note: 'openai_embedding' is a legacy value that should be treated as 'openai'
        provider = self.model_config.model_provider.lower().strip()
        if provider in ("openai", "openai_embedding", "openai-compatible"):
            return OpenAIEmbedding(
                model=self.model_config.model_name,
                api_key=self.model_config.api_key,
                base_url=self.model_config.base_url,
                dimension=self.model_config.dimension,
            )
        elif provider == "dashscope":
            return DashScopeEmbedding(
                model=self.model_config.model_name,
                api_key=self.model_config.api_key,
                base_url=self.model_config.base_url,
                dimension=self.model_config.dimension,
                instruct=self.model_config.instruct,
            )
        elif provider == "xinference":
            return XinferenceEmbedding(
                model=self.model_config.model_name,
                base_url=self.model_config.base_url,
                api_key=self.model_config.api_key,
                dimension=self.model_config.dimension,
            )
        else:
            raise ValueError(
                f"Unsupported model provider: {self.model_config.model_provider}"
            )

    def encode(
        self,
        text: Union[str, List[str]],
        dimension: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Union[List[float], List[List[float]]]:
        """Encode text using the underlying embedding model."""
        return self._embedding_model.encode(text, dimension, instruct)

    def get_dimension(self) -> Optional[int]:
        """Get the embedding dimension."""
        return self._embedding_model.get_dimension()

    @property
    def abilities(self) -> List[str]:
        """Get the model abilities."""
        return self._embedding_model.abilities
