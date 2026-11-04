"""Shared embedding stubs for the LanceDB memory-store tests (#822).

``ConstantEmbedding`` returns the same vector for every text, so vector distance
cannot rank results — isolating where-prefilter / top-k selection from similarity
ranking (used by the population-independence and column-promotion suites).

``VaryingEmbedding`` maps each registered text to a distinct vector, so cosine/L2
distance is meaningful and similarity ordering can be asserted *alongside* the
where-prefilter (used by the ranking-under-prefilter test).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from xagent.core.model.embedding import BaseEmbedding


class ConstantEmbedding(BaseEmbedding):
    """Every text embeds to the same vector (configurable dimension/value)."""

    def __init__(self, dim: int = 8, value: float = 0.1) -> None:
        self._dimension = dim
        self._value = value

    def encode(self, text: Any, dimension: Any = None, instruct: Any = None) -> Any:
        if isinstance(text, str):
            return [self._value] * self._dimension
        return [[self._value] * self._dimension for _ in text]

    def get_dimension(self) -> int:
        return self._dimension

    @property
    def abilities(self) -> list[str]:
        return ["embed"]


class VaryingEmbedding(BaseEmbedding):
    """Maps each registered text to a distinct vector so distance ranks results;
    an unregistered text falls back to a fixed zero vector."""

    def __init__(self, vectors: Mapping[str, Sequence[float]], dim: int) -> None:
        self._vectors = {key: list(value) for key, value in vectors.items()}
        self._dimension = dim
        self._default = [0.0] * dim

    def _one(self, text: str) -> list[float]:
        return list(self._vectors.get(text, self._default))

    def encode(self, text: Any, dimension: Any = None, instruct: Any = None) -> Any:
        if isinstance(text, str):
            return self._one(text)
        return [self._one(item) for item in text]

    def get_dimension(self) -> int:
        return self._dimension

    @property
    def abilities(self) -> list[str]:
        return ["embed"]
