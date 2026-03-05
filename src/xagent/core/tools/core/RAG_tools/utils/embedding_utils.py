"""Embedding output normalization for RAG pipelines.

Unifies provider-specific embedding responses (OpenAI, DashScope, Xinference, etc.)
into a single format so downstream code always receives List[float] or List[List[float]],
avoiding Pydantic validation errors (e.g. ChunkEmbeddingData.vector expecting list
when provider returns list of dicts with 'embedding' key).
"""

from __future__ import annotations

from typing import Any, List

from ..core.exceptions import VectorValidationError


def normalize_raw_embedding_to_vectors(raw: Any) -> List[List[float]]:
    """Normalize any embedding provider output to a list of embedding vectors.

    Accepted input shapes:
        - List[float]: single vector -> returned as [vector]
        - List[List[float]]: batch of vectors -> returned as-is (validated)
        - List[dict]: provider items with "embedding" key (e.g. OpenAI format
          {"index": 0, "object": "embedding", "embedding": [...]}) -> extract
          embedding from each item and return List[List[float]]

    Args:
        raw: Raw return value from BaseEmbedding.encode() or provider API.

    Returns:
        List of embedding vectors; each element is List[float]. Empty list if raw
        is empty or empty list.

    Raises:
        VectorValidationError: If raw is not a list, or list elements are not
            float, list of float, or dict with "embedding" key containing
            list of float. Details include response_type and first_item_type
            for debugging.
    """
    if not isinstance(raw, list):
        raise VectorValidationError(
            "Embedding response must be a list",
            details={
                "response_type": type(raw).__name__,
            },
        )

    if not raw:
        return []

    first = raw[0]

    # Single vector: List[float]
    if isinstance(first, (int, float)):
        if not all(isinstance(x, (int, float)) for x in raw):
            raise VectorValidationError(
                "Embedding response is a list but not all elements are numbers",
                details={
                    "response_type": "list",
                    "first_item_type": type(first).__name__,
                },
            )
        return [[float(x) for x in raw]]

    # Batch of vectors: List[List[float]]
    if isinstance(first, list):
        vectors: List[List[float]] = []
        for i, item in enumerate(raw):
            if not isinstance(item, list):
                raise VectorValidationError(
                    "Embedding batch must contain lists of numbers or dicts with 'embedding' key",
                    details={
                        "response_type": "list",
                        "item_index": i,
                        "item_type": type(item).__name__,
                    },
                )
            try:
                vectors.append([float(x) for x in item])
            except (TypeError, ValueError) as e:
                raise VectorValidationError(
                    "Embedding vector elements must be numbers",
                    details={
                        "item_index": i,
                        "error": str(e),
                    },
                ) from e
        return vectors

    # Provider dict format: e.g. {"index": 0, "object": "embedding", "embedding": [...]}
    if isinstance(first, dict):
        vectors = []
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                raise VectorValidationError(
                    "Embedding response mixes dict and non-dict items",
                    details={
                        "item_index": i,
                        "item_type": type(item).__name__,
                    },
                )
            emb = item.get("embedding")
            if emb is None:
                raise VectorValidationError(
                    "Embedding dict item missing 'embedding' key (e.g. OpenAI-style response)",
                    details={
                        "item_index": i,
                        "keys": list(item.keys()),
                    },
                )
            if not isinstance(emb, list):
                raise VectorValidationError(
                    "Embedding dict 'embedding' value must be a list of numbers",
                    details={
                        "item_index": i,
                        "embedding_type": type(emb).__name__,
                    },
                )
            try:
                vectors.append([float(x) for x in emb])
            except (TypeError, ValueError) as e:
                raise VectorValidationError(
                    "Embedding vector elements must be numbers",
                    details={
                        "item_index": i,
                        "error": str(e),
                    },
                ) from e
        return vectors

    raise VectorValidationError(
        "Embedding response list has unsupported element type",
        details={
            "response_type": "list",
            "first_item_type": type(first).__name__,
        },
    )


def normalize_single_embedding(raw: Any) -> List[float]:
    """Normalize a single-text embedding response to one vector.

    Convenience wrapper for async/single-chunk encode: ensures exactly one
    vector is returned.

    Args:
        raw: Raw return value from encode(single_text).

    Returns:
        Single embedding as List[float].

    Raises:
        VectorValidationError: If raw cannot be normalized or does not yield
            exactly one vector.
    """
    vectors = normalize_raw_embedding_to_vectors(raw)
    if len(vectors) != 1:
        raise VectorValidationError(
            "Single-text embedding must return exactly one vector",
            details={
                "expected_count": 1,
                "actual_count": len(vectors),
            },
        )
    return vectors[0]
