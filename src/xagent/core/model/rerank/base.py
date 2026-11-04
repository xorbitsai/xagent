from abc import ABC, abstractmethod
from collections.abc import Sequence


class BaseRerank(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def compress(
        self,
        documents: Sequence[str],
        query: str,
    ) -> Sequence[str]:
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
