from datetime import datetime, timezone
from typing import List

import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import SearchResult
from xagent.core.tools.core.RAG_tools.retrieval.format_context import (
    format_search_results_for_llm,
)


@pytest.fixture
def sample_search_results() -> List[SearchResult]:
    """Provides a list of sample SearchResult objects for testing."""
    return [
        SearchResult(
            doc_id="doc1",
            chunk_id="chunk1",
            text="This is the first chunk of text.",
            score=0.95,
            parse_hash="hash1",
            model_tag="model-a",
            created_at=datetime.now(timezone.utc),
            metadata={"page": 1, "section": "intro", "source": "/path/to/file.pdf"},
        ),
        SearchResult(
            doc_id="doc2",
            chunk_id="chunk2",
            text="This is the second chunk, with important keywords.",
            score=0.88,
            parse_hash="hash2",
            model_tag="model-a",
            created_at=datetime.now(timezone.utc),
            metadata={"page": 2, "section": "body"},
        ),
        SearchResult(
            doc_id="doc3",
            chunk_id="chunk3",
            text="A third chunk for diverse testing scenarios.",
            score=0.75,
            parse_hash="hash3",
            model_tag="model-b",
            created_at=datetime.now(timezone.utc),
            metadata=None,  # Test case with no metadata
        ),
    ]


class TestFormatter:
    """Tests for the formatting utility functions."""

    def test_format_search_results_for_llm_basic(
        self,
        sample_search_results: List[SearchResult],
    ) -> None:
        """Test basic formatting without metadata and default top_k."""
        formatted_output = format_search_results_for_llm(sample_search_results)
        expected_output = (
            "[1]\nThis is the first chunk of text.\n---\n"
            "[2]\nThis is the second chunk, with important keywords.\n---\n"
            "[3]\nA third chunk for diverse testing scenarios."
        )
        assert formatted_output == expected_output

    def test_format_search_results_for_llm_with_metadata(
        self,
        sample_search_results: List[SearchResult],
    ) -> None:
        """Test formatting with metadata included."""
        formatted_output = format_search_results_for_llm(
            sample_search_results, include_metadata=True
        )
        # Note: Score will vary slightly due to float precision, so check presence and general format
        assert (
            "[1]\nDocument ID: doc1, Chunk ID: chunk1, Score: 0.9500, Metadata: {'page': 1, 'section': 'intro', 'source': '/path/to/file.pdf'}\nThis is the first chunk of text."
            in formatted_output
        )
        assert (
            "[2]\nDocument ID: doc2, Chunk ID: chunk2, Score: 0.8800, Metadata: {'page': 2, 'section': 'body'}\nThis is the second chunk, with important keywords."
            in formatted_output
        )
        # Result 3 has no metadata, should not include Metadata field
        assert (
            "[3]\nDocument ID: doc3, Chunk ID: chunk3, Score: 0.7500\nA third chunk for diverse testing scenarios."
            in formatted_output
        )
        assert formatted_output.count("\n---\n") == 2

    def test_format_search_results_for_llm_top_k(
        self,
        sample_search_results: List[SearchResult],
    ) -> None:
        """Test formatting with a specified top_k."""
        formatted_output = format_search_results_for_llm(sample_search_results, top_k=2)
        expected_output = (
            "[1]\nThis is the first chunk of text.\n---\n"
            "[2]\nThis is the second chunk, with important keywords."
        )
        assert formatted_output == expected_output

    def test_format_search_results_for_llm_empty_results(self) -> None:
        """Test formatting with an empty list of search results."""
        formatted_output = format_search_results_for_llm([])
        assert formatted_output == ""

    def test_format_search_results_for_llm_custom_separator(
        self,
        sample_search_results: List[SearchResult],
    ) -> None:
        """Test formatting with a custom separator."""
        formatted_output = format_search_results_for_llm(
            sample_search_results, separator="\n===\n"
        )
        expected_output = (
            "[1]\nThis is the first chunk of text.\n===\n"
            "[2]\nThis is the second chunk, with important keywords.\n===\n"
            "[3]\nA third chunk for diverse testing scenarios."
        )
        assert formatted_output == expected_output

    def test_format_search_results_for_llm_top_k_with_metadata(
        self,
        sample_search_results: List[SearchResult],
    ) -> None:
        """Test formatting with top_k and metadata included."""
        formatted_output = format_search_results_for_llm(
            sample_search_results, top_k=1, include_metadata=True
        )
        # With metadata, the output should include metadata information
        assert (
            "[1]\nDocument ID: doc1, Chunk ID: chunk1, Score: 0.9500, Metadata: {'page': 1, 'section': 'intro', 'source': '/path/to/file.pdf'}\nThis is the first chunk of text."
            in formatted_output
        )
        assert "---" not in formatted_output  # Only one result, no separator expected
