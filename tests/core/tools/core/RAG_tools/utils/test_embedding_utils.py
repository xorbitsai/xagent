"""Tests for embedding output normalization (embedding_utils)."""

from __future__ import annotations

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import VectorValidationError
from xagent.core.tools.core.RAG_tools.utils.embedding_utils import (
    normalize_raw_embedding_to_vectors,
    normalize_single_embedding,
)


class TestNormalizeRawEmbeddingToVectors:
    """Tests for normalize_raw_embedding_to_vectors."""

    def test_empty_list_returns_empty(self) -> None:
        assert normalize_raw_embedding_to_vectors([]) == []

    def test_single_vector_list_of_floats(self) -> None:
        raw = [0.1, -0.2, 0.3]
        got = normalize_raw_embedding_to_vectors(raw)
        assert got == [[0.1, -0.2, 0.3]]

    def test_batch_list_of_vectors(self) -> None:
        raw = [[0.1, 0.2], [0.3, 0.4]]
        got = normalize_raw_embedding_to_vectors(raw)
        assert got == [[0.1, 0.2], [0.3, 0.4]]

    def test_openai_style_list_of_dicts(self) -> None:
        """Provider returns list of dicts with 'embedding' key (e.g. OpenAI)."""
        raw = [
            {"index": 0, "object": "embedding", "embedding": [0.1, -0.2, 0.3]},
            {"index": 1, "object": "embedding", "embedding": [0.4, 0.5, 0.6]},
        ]
        got = normalize_raw_embedding_to_vectors(raw)
        assert got == [[0.1, -0.2, 0.3], [0.4, 0.5, 0.6]]

    def test_dict_with_only_embedding_key(self) -> None:
        raw = [{"embedding": [1.0, 2.0]}]
        got = normalize_raw_embedding_to_vectors(raw)
        assert got == [[1.0, 2.0]]

    def test_non_list_raises(self) -> None:
        with pytest.raises(VectorValidationError) as exc_info:
            normalize_raw_embedding_to_vectors({"data": []})
        assert "list" in exc_info.value.message
        assert exc_info.value.details.get("response_type") == "dict"

    def test_list_of_dict_missing_embedding_key_raises(self) -> None:
        with pytest.raises(VectorValidationError) as exc_info:
            normalize_raw_embedding_to_vectors([{"index": 0, "object": "embedding"}])
        assert "embedding" in exc_info.value.message
        assert "keys" in exc_info.value.details

    def test_list_of_dict_embedding_not_list_raises(self) -> None:
        with pytest.raises(VectorValidationError) as exc_info:
            normalize_raw_embedding_to_vectors([{"embedding": "not a list"}])
        assert "embedding" in exc_info.value.message or "list" in exc_info.value.message


class TestNormalizeSingleEmbedding:
    """Tests for normalize_single_embedding."""

    def test_single_vector_returns_one(self) -> None:
        raw = [0.1, 0.2, 0.3]
        got = normalize_single_embedding(raw)
        assert got == [0.1, 0.2, 0.3]

    def test_openai_style_single_dict(self) -> None:
        raw = {"index": 0, "object": "embedding", "embedding": [0.1, -0.046]}
        # normalize_raw_embedding_to_vectors expects list; single dict in list
        got = normalize_single_embedding([raw])
        assert got == [0.1, -0.046]

    def test_empty_list_raises(self) -> None:
        with pytest.raises(VectorValidationError) as exc_info:
            normalize_single_embedding([])
        assert "exactly one" in exc_info.value.message

    def test_two_vectors_raises(self) -> None:
        with pytest.raises(VectorValidationError) as exc_info:
            normalize_single_embedding([[0.1], [0.2]])
        assert exc_info.value.details.get("actual_count") == 2
