from abc import ABC, abstractmethod
from typing import List, Optional, Union


class BaseEmbedding(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
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
        """
        pass

    @abstractmethod
    def get_dimension(self) -> Optional[int]:
        """Get the embedding dimension."""
        pass

    @property
    @abstractmethod
    def abilities(self) -> List[str]:
        """Get the list of abilities supported by this model."""
        pass
