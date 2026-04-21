"""Unit tests for knowledge base search tool in Vibe adapters."""

from unittest.mock import MagicMock, patch

import pytest

from xagent.core.tools.adapters.vibe.document_search import (
    KnowledgeSearchTool,
    ListKnowledgeBasesTool,
    get_knowledge_search_tool,
    get_list_knowledge_bases_tool,
)
from xagent.core.tools.core.document_search import _format_search_results
from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo


class TestListKnowledgeBasesTool:
    """Test list knowledge bases tool."""

    def test_tool_initialization(self):
        """Test tool can be initialized."""
        tool = get_list_knowledge_bases_tool()
        assert isinstance(tool, ListKnowledgeBasesTool)
        assert tool.name == "list_knowledge_bases"
        assert "knowledge" in tool.tags

    def test_tool_initialization_with_allowed_collections(self):
        """Test tool can be initialized with allowed_collections."""
        tool = get_list_knowledge_bases_tool(allowed_collections=["kb1", "kb2"])
        assert isinstance(tool, ListKnowledgeBasesTool)
        assert tool.allowed_collections == ["kb1", "kb2"]

    def test_tool_has_description(self):
        """Test tool has proper description."""
        tool = get_list_knowledge_bases_tool()
        assert tool.description
        assert "knowledge bases" in tool.description.lower()

    @pytest.mark.asyncio
    async def test_list_empty_collections(self):
        """Test listing when no collections exist."""
        mock_result = MagicMock()
        mock_result.collections = []

        with patch(
            "xagent.core.tools.core.document_search.list_collections",
            return_value=mock_result,
        ):
            tool = get_list_knowledge_bases_tool()
            result = await tool.run_json_async({})

        assert result.knowledge_bases == []

    @pytest.mark.asyncio
    async def test_list_multiple_collections(self):
        """Test listing multiple collections."""
        mock_collections = [
            CollectionInfo(
                name="kb1",
                documents=10,
                embeddings=100,
                document_names=["doc1.pdf", "doc2.pdf"],
            ),
            CollectionInfo(
                name="kb2",
                documents=20,
                embeddings=200,
                document_names=["doc3.pdf"],
            ),
        ]

        mock_result = MagicMock()
        mock_result.collections = mock_collections

        with patch(
            "xagent.core.tools.core.document_search.list_collections",
            return_value=mock_result,
        ):
            tool = get_list_knowledge_bases_tool()
            result = await tool.run_json_async({})

        assert len(result.knowledge_bases) == 2
        assert result.knowledge_bases[0]["name"] == "kb1"
        assert result.knowledge_bases[0]["documents"] == 10
        assert result.knowledge_bases[1]["name"] == "kb2"
        assert result.knowledge_bases[1]["documents"] == 20

    @pytest.mark.asyncio
    async def test_list_with_allowed_collections_filter(self):
        """Test listing with allowed_collections filter."""
        mock_collections = [
            CollectionInfo(
                name="kb1",
                documents=10,
                embeddings=100,
                document_names=["doc1.pdf"],
            ),
            CollectionInfo(
                name="kb2",
                documents=20,
                embeddings=200,
                document_names=["doc2.pdf"],
            ),
            CollectionInfo(
                name="kb3",
                documents=30,
                embeddings=300,
                document_names=["doc3.pdf"],
            ),
        ]

        mock_result = MagicMock()
        mock_result.collections = mock_collections

        with patch(
            "xagent.core.tools.core.document_search.list_collections",
            return_value=mock_result,
        ):
            # Only allow kb1 and kb2
            tool = get_list_knowledge_bases_tool(allowed_collections=["kb1", "kb2"])
            result = await tool.run_json_async({})

        # Should only return kb1 and kb2, not kb3
        assert len(result.knowledge_bases) == 2
        names = {kb["name"] for kb in result.knowledge_bases}
        assert names == {"kb1", "kb2"}

    @pytest.mark.asyncio
    async def test_list_with_non_allowed_collection(self):
        """Test listing when allowed_collections contains non-existent collection."""
        mock_collections = [
            CollectionInfo(
                name="kb1",
                documents=10,
                embeddings=100,
                document_names=["doc1.pdf"],
            ),
        ]

        mock_result = MagicMock()
        mock_result.collections = mock_collections

        with patch(
            "xagent.core.tools.core.document_search.list_collections",
            return_value=mock_result,
        ):
            # Allow kb1 and kb2 (kb2 doesn't exist)
            tool = get_list_knowledge_bases_tool(allowed_collections=["kb1", "kb2"])
            result = await tool.run_json_async({})

        # Should only return kb1 (existing)
        assert len(result.knowledge_bases) == 1
        assert result.knowledge_bases[0]["name"] == "kb1"

    @pytest.mark.asyncio
    async def test_list_collections_error_handling(self):
        """Test error handling when list_collections fails."""
        with patch(
            "xagent.core.tools.core.document_search.list_collections",
            side_effect=Exception("Database error"),
        ):
            tool = get_list_knowledge_bases_tool()

            with pytest.raises(RuntimeError, match="Failed to list knowledge bases"):
                await tool.run_json_async({})


class TestKnowledgeSearchTool:
    """Test knowledge search tool."""

    def test_tool_initialization(self):
        """Test tool can be initialized."""
        tool = get_knowledge_search_tool()
        assert isinstance(tool, KnowledgeSearchTool)
        assert tool.name == "knowledge_search"
        assert "search" in tool.tags

    def test_tool_initialization_with_allowed_collections(self):
        """Test tool can be initialized with allowed_collections."""
        tool = get_knowledge_search_tool(allowed_collections=["kb1", "kb2"])
        assert isinstance(tool, KnowledgeSearchTool)
        assert tool.allowed_collections == ["kb1", "kb2"]

    def test_tool_has_description(self):
        """Test tool has proper description."""
        tool = get_knowledge_search_tool()
        assert tool.description
        assert "search" in tool.description.lower()

    @pytest.mark.asyncio
    async def test_search_no_collections_available(self):
        """Test searching when no collections exist."""
        mock_result = MagicMock()
        mock_result.collections = []

        with patch(
            "xagent.core.tools.core.document_search.list_collections",
            return_value=mock_result,
        ):
            tool = get_knowledge_search_tool()
            result = await tool.run_json_async(
                {
                    "query": "test query",
                    "collections": [],
                    "search_type": "hybrid",
                    "top_k": 5,
                    "min_score": 0.3,
                }
            )

        assert "No knowledge bases available" in result.summary

    @pytest.mark.asyncio
    async def test_search_specific_collections(self):
        """Test searching specific collections."""
        mock_collections = [
            CollectionInfo(
                name="kb1",
                total_documents=10,
                embeddings=100,
                document_names=["doc1.pdf"],
            ),
            CollectionInfo(
                name="kb2",
                total_documents=20,
                embeddings=200,
                document_names=["doc2.pdf"],
            ),
        ]

        mock_list_result = MagicMock()
        mock_list_result.collections = mock_collections

        # Mock search result
        mock_search_result = MagicMock()
        mock_search_result.results = [
            {
                "text": "Test content",
                "score": 0.85,
                "metadata": {"doc_id": "doc1", "chunk_id": "chunk1"},
            }
        ]

        with patch(
            "xagent.core.tools.core.document_search.list_collections",
            return_value=mock_list_result,
        ):
            with patch(
                "xagent.core.tools.core.document_search.run_document_search",
                return_value=mock_search_result,
            ):
                tool = get_knowledge_search_tool()
                result = await tool.run_json_async(
                    {
                        "query": "test query",
                        "collections": ["kb1"],  # Only search kb1
                        "search_type": "hybrid",
                        "top_k": 5,
                        "min_score": 0.3,
                    }
                )

        # Should search only kb1
        assert len(result.results) >= 1, "Expected at least one result"
        if result.results:
            assert result.results[0].text == "Test content"

    @pytest.mark.asyncio
    async def test_search_invalid_collection_name(self):
        """Test searching with invalid collection name."""
        mock_collections = [
            CollectionInfo(
                name="kb1",
                total_documents=10,
                embeddings=100,
                document_names=["doc1.pdf"],
            ),
        ]

        mock_result = MagicMock()
        mock_result.collections = mock_collections

        with patch(
            "xagent.core.tools.core.document_search.list_collections",
            return_value=mock_result,
        ):
            tool = get_knowledge_search_tool()
            result = await tool.run_json_async(
                {
                    "query": "test query",
                    "collections": ["invalid_kb"],  # Non-existent collection
                    "search_type": "hybrid",
                    "top_k": 5,
                    "min_score": 0.3,
                }
            )

        assert "Error:" in result.summary
        assert "do not exist" in result.summary

    @pytest.mark.asyncio
    async def test_search_all_collections_by_default(self):
        """Test that empty collections list searches all collections."""
        mock_collections = [
            CollectionInfo(
                name="kb1",
                total_documents=10,
                embeddings=100,
                document_names=["doc1.pdf"],
            ),
        ]

        mock_list_result = MagicMock()
        mock_list_result.collections = mock_collections

        mock_search_result = MagicMock()
        mock_search_result.results = []

        with patch(
            "xagent.core.tools.core.document_search.list_collections",
            return_value=mock_list_result,
        ):
            with patch(
                "xagent.core.tools.core.document_search.run_document_search",
                return_value=mock_search_result,
            ) as mock_search:
                tool = get_knowledge_search_tool()
                await tool.run_json_async(
                    {
                        "query": "test query",
                        "collections": [],  # Empty = search all
                        "search_type": "hybrid",
                        "top_k": 5,
                        "min_score": 0.3,
                    }
                )

            # Should have called run_document_search
            mock_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_skip_empty_collections(self):
        """Test that empty collections are skipped."""
        mock_collections = [
            CollectionInfo(
                name="empty_kb",
                total_documents=0,  # Empty
                embeddings=0,
                document_names=[],
            ),
            CollectionInfo(
                name="full_kb",
                total_documents=10,  # Has documents
                embeddings=100,
                document_names=["doc1.pdf"],
            ),
        ]

        mock_list_result = MagicMock()
        mock_list_result.collections = mock_collections

        mock_search_result = MagicMock()
        mock_search_result.results = []

        with patch(
            "xagent.core.tools.core.document_search.list_collections",
            return_value=mock_list_result,
        ):
            with patch(
                "xagent.core.tools.core.document_search.run_document_search",
                return_value=mock_search_result,
            ) as mock_search:
                tool = get_knowledge_search_tool()
                await tool.run_json_async(
                    {
                        "query": "test query",
                        "collections": [],
                        "search_type": "hybrid",
                        "top_k": 5,
                        "min_score": 0.3,
                    }
                )

                # Should only search full_kb, not empty_kb
                assert mock_search.call_count == 1
                call_args = mock_search.call_args
                assert call_args[1]["collection"] in ["full_kb", "empty_kb"]

    @pytest.mark.asyncio
    async def test_search_with_allowed_collections_default(self):
        """Test searching uses allowed_collections as default when collections is empty."""
        mock_collections = [
            CollectionInfo(
                name="kb1",
                total_documents=10,
                embeddings=100,
                document_names=["doc1.pdf"],
            ),
            CollectionInfo(
                name="kb2",
                total_documents=20,
                embeddings=200,
                document_names=["doc2.pdf"],
            ),
            CollectionInfo(
                name="kb3",
                total_documents=30,
                embeddings=300,
                document_names=["doc3.pdf"],
            ),
        ]

        mock_list_result = MagicMock()
        mock_list_result.collections = mock_collections

        mock_search_result = MagicMock()
        mock_search_result.results = []

        with patch(
            "xagent.core.tools.core.document_search.list_collections",
            return_value=mock_list_result,
        ):
            with patch(
                "xagent.core.tools.core.document_search.run_document_search",
                return_value=mock_search_result,
            ) as mock_search:
                # Create tool with allowed_collections=["kb1", "kb2"]
                tool = get_knowledge_search_tool(allowed_collections=["kb1", "kb2"])
                await tool.run_json_async(
                    {
                        "query": "test query",
                        "collections": [],  # Empty - should use allowed_collections
                        "search_type": "hybrid",
                        "top_k": 5,
                        "min_score": 0.3,
                    }
                )

                # Should only search kb1 and kb2, not kb3
                assert mock_search.call_count == 2
                searched_collections = {
                    call[1]["collection"] for call in mock_search.call_args_list
                }
                assert searched_collections == {"kb1", "kb2"}

    @pytest.mark.asyncio
    async def test_search_respects_allowed_collections_boundary(self):
        """Test that specified collections must be within allowed_collections."""
        mock_collections = [
            CollectionInfo(
                name="kb1",
                total_documents=10,
                embeddings=100,
                document_names=["doc1.pdf"],
            ),
            CollectionInfo(
                name="kb2",
                total_documents=20,
                embeddings=200,
                document_names=["doc2.pdf"],
            ),
        ]

        mock_result = MagicMock()
        mock_result.collections = mock_collections

        with patch(
            "xagent.core.tools.core.document_search.list_collections",
            return_value=mock_result,
        ):
            # Create tool with allowed_collections=["kb1"]
            tool = get_knowledge_search_tool(allowed_collections=["kb1"])
            result = await tool.run_json_async(
                {
                    "query": "test query",
                    "collections": ["kb1", "kb2"],  # kb2 is not allowed
                    "search_type": "hybrid",
                    "top_k": 5,
                    "min_score": 0.3,
                }
            )

        # Should return error about kb2 not being allowed
        assert "Error:" in result.summary
        assert "not allowed" in result.summary
        assert "kb2" in result.summary

    @pytest.mark.asyncio
    async def test_search_within_allowed_collections_succeeds(self):
        """Test searching within allowed_collections boundary succeeds."""
        mock_collections = [
            CollectionInfo(
                name="kb1",
                total_documents=10,
                embeddings=100,
                document_names=["doc1.pdf"],
            ),
            CollectionInfo(
                name="kb2",
                total_documents=20,
                embeddings=200,
                document_names=["doc2.pdf"],
            ),
        ]

        mock_list_result = MagicMock()
        mock_list_result.collections = mock_collections

        mock_search_result = MagicMock()
        mock_search_result.results = [{"text": "Test content", "score": 0.85}]

        with patch(
            "xagent.core.tools.core.document_search.list_collections",
            return_value=mock_list_result,
        ):
            with patch(
                "xagent.core.tools.core.document_search.run_document_search",
                return_value=mock_search_result,
            ) as mock_search:
                # Create tool with allowed_collections=["kb1", "kb2"]
                tool = get_knowledge_search_tool(allowed_collections=["kb1", "kb2"])
                await tool.run_json_async(
                    {
                        "query": "test query",
                        "collections": ["kb1"],  # Within allowed
                        "search_type": "hybrid",
                        "top_k": 5,
                        "min_score": 0.3,
                    }
                )

                # Should successfully search kb1
                mock_search.assert_called_once()
                assert mock_search.call_args[1]["collection"] == "kb1"

    @pytest.mark.asyncio
    async def test_search_without_allowed_collections_no_restriction(self):
        """Test that without allowed_collections, any collection can be searched."""
        mock_collections = [
            CollectionInfo(
                name="kb1",
                total_documents=10,
                embeddings=100,
                document_names=["doc1.pdf"],
            ),
            CollectionInfo(
                name="kb2",
                total_documents=20,
                embeddings=200,
                document_names=["doc2.pdf"],
            ),
        ]

        mock_list_result = MagicMock()
        mock_list_result.collections = mock_collections

        mock_search_result = MagicMock()
        mock_search_result.results = []

        with patch(
            "xagent.core.tools.core.document_search.list_collections",
            return_value=mock_list_result,
        ):
            with patch(
                "xagent.core.tools.core.document_search.run_document_search",
                return_value=mock_search_result,
            ) as mock_search:
                # Create tool WITHOUT allowed_collections
                tool = get_knowledge_search_tool()
                await tool.run_json_async(
                    {
                        "query": "test query",
                        "collections": ["kb1", "kb2"],  # Any collection
                        "search_type": "hybrid",
                        "top_k": 5,
                        "min_score": 0.3,
                    }
                )

                # Should search both kb1 and kb2
                assert mock_search.call_count == 2
                searched_collections = {
                    call[1]["collection"] for call in mock_search.call_args_list
                }
                assert searched_collections == {"kb1", "kb2"}


class TestFormatSearchResults:
    """Test search results formatting."""

    def test_format_empty_results(self):
        """Test formatting empty result list."""
        formatted_results, summary = _format_search_results([], "test query", 0)

        assert "test query" in summary
        assert "0 relevant results" in summary

    def test_format_single_result(self):
        """Test formatting single result."""
        results = [
            {
                "collection": "kb1",
                "score": 0.85,
                "text": "Test content here",
                "metadata": {"doc_id": "doc1", "chunk_id": "chunk1"},
            }
        ]

        formatted_results, summary = _format_search_results(results, "test query", 100)

        # Summary should only contain statistics, not full content
        assert "test query" in summary
        assert "1 relevant results" in summary
        assert "100 documents" in summary
        # Content should be in formatted_results, not summary
        assert "Test content here" not in summary
        # Check structured results
        assert len(formatted_results) == 1
        assert formatted_results[0]["collection"] == "kb1"
        assert formatted_results[0]["text"] == "Test content here"

    def test_format_multiple_results(self):
        """Test formatting multiple results."""
        results = [
            {
                "collection": "kb1",
                "score": 0.85,
                "text": "First result",
                "metadata": {},
            },
            {
                "collection": "kb2",
                "score": 0.75,
                "text": "Second result",
                "metadata": {},
            },
        ]

        formatted_results, summary = _format_search_results(results, "test query", 200)

        # Summary should only contain statistics, not full content
        assert "2 relevant results" in summary
        assert "200 documents" in summary
        # Content should be in formatted_results, not summary
        assert "First result" not in summary
        assert "Second result" not in summary
        assert "Result 1" not in summary
        assert "Result 2" not in summary

    def test_format_result_without_metadata(self):
        """Test formatting result without metadata."""
        results = [
            {
                "collection": "kb1",
                "score": 0.85,
                "text": "Test content",
                "metadata": {},  # Empty metadata
            }
        ]

        formatted_results, summary = _format_search_results(results, "test query", 100)

        # Summary should only contain statistics, not full content
        assert "Test content" not in summary
        assert "1 relevant results" in summary
        # Content should be in formatted_results
        assert formatted_results[0]["text"] == "Test content"
