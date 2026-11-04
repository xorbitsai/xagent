"""Additional tests for metadata handling in format_context."""

from datetime import datetime, timezone

from xagent.core.tools.core.RAG_tools.core.schemas import SearchResult
from xagent.core.tools.core.RAG_tools.retrieval.format_context import (
    format_search_results_for_llm,
)


class TestFormatContextMetadata:
    """Test metadata handling in format_context."""

    def test_format_with_metadata_dict(self):
        """Test formatting with metadata dictionary."""
        results = [
            SearchResult(
                doc_id="doc1",
                chunk_id="chunk1",
                text="Test content",
                score=0.9,
                parse_hash="hash1",
                model_tag="model1",
                created_at=datetime.now(timezone.utc),
                metadata={"page": 1, "section": "intro", "source": "/path/to/file.pdf"},
            )
        ]

        formatted = format_search_results_for_llm(results, include_metadata=True)

        assert "Document ID: doc1" in formatted
        assert "Chunk ID: chunk1" in formatted
        assert "Score: 0.9000" in formatted
        assert "Metadata:" in formatted
        assert "page" in formatted or "1" in formatted

    def test_format_without_metadata_dict(self):
        """Test formatting when metadata is None."""
        results = [
            SearchResult(
                doc_id="doc1",
                chunk_id="chunk1",
                text="Test content",
                score=0.9,
                parse_hash="hash1",
                model_tag="model1",
                created_at=datetime.now(timezone.utc),
                metadata=None,
            )
        ]

        formatted = format_search_results_for_llm(results, include_metadata=True)

        assert "Document ID: doc1" in formatted
        assert "Chunk ID: chunk1" in formatted
        assert "Score: 0.9000" in formatted
        # Should not include "Metadata: None" when metadata is None
        assert formatted.count("Metadata:") == 0

    def test_format_with_empty_metadata_dict(self):
        """Test formatting with empty metadata dictionary."""
        results = [
            SearchResult(
                doc_id="doc1",
                chunk_id="chunk1",
                text="Test content",
                score=0.9,
                parse_hash="hash1",
                model_tag="model1",
                created_at=datetime.now(timezone.utc),
                metadata={},
            )
        ]

        formatted = format_search_results_for_llm(results, include_metadata=True)

        assert "Document ID: doc1" in formatted
        # Empty dict is falsy, so metadata won't be included (consistent with None handling)
        # The format_context checks `if result.metadata:` which is False for empty dict

    def test_format_with_complex_metadata(self):
        """Test formatting with complex nested metadata."""
        results = [
            SearchResult(
                doc_id="doc1",
                chunk_id="chunk1",
                text="Test content",
                score=0.9,
                parse_hash="hash1",
                model_tag="model1",
                created_at=datetime.now(timezone.utc),
                metadata={
                    "source": "/path/to/file.pdf",
                    "position": {"page": 1, "coordinates": {"x0": 0.1, "y0": 0.2}},
                    "parser": "deepdoc",
                    "parse_method": "default",
                },
            )
        ]

        formatted = format_search_results_for_llm(results, include_metadata=True)

        assert "Document ID: doc1" in formatted
        assert "Metadata:" in formatted
        # Should contain some metadata fields
        assert "source" in formatted or "parser" in formatted

    def test_format_mixed_metadata_results(self):
        """Test formatting with mixed metadata presence."""
        results = [
            SearchResult(
                doc_id="doc1",
                chunk_id="chunk1",
                text="First chunk",
                score=0.9,
                parse_hash="hash1",
                model_tag="model1",
                created_at=datetime.now(timezone.utc),
                metadata={"page": 1},
            ),
            SearchResult(
                doc_id="doc2",
                chunk_id="chunk2",
                text="Second chunk",
                score=0.8,
                parse_hash="hash2",
                model_tag="model1",
                created_at=datetime.now(timezone.utc),
                metadata=None,
            ),
        ]

        formatted = format_search_results_for_llm(results, include_metadata=True)

        # First result should have metadata
        assert "[1]" in formatted
        assert "Metadata:" in formatted.split("[2]")[0]

        # Second result should not have metadata
        assert "[2]" in formatted
        # Metadata: should not appear twice (only once for first result)
        assert formatted.count("Metadata:") == 1
