"""
Tests for Exa Web Search tool
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from xagent.core.tools.adapters.vibe.exa_web_search import (
    ExaWebSearchArgs,
    ExaWebSearchResult,
    ExaWebSearchTool,
)


@pytest.fixture
def exa_search_tool():
    """Create ExaWebSearchTool instance for testing"""
    return ExaWebSearchTool()


def _make_mock_result(
    title="Test Article", url="https://example.com/article", **kwargs
):
    """Create a mock Exa result object."""
    result = MagicMock()
    result.title = title
    result.url = url
    result.highlights = kwargs.get("highlights", ["Key highlight from article"])
    result.text = kwargs.get("text", "Full text content of the article")
    result.summary = kwargs.get("summary", "Summary of the article")
    result.published_date = kwargs.get("published_date", None)
    result.author = kwargs.get("author", None)
    return result


def _make_mock_response(results=None):
    """Create a mock Exa search response."""
    response = MagicMock()
    if results is None:
        results = [
            _make_mock_result(
                title="Test Article 1",
                url="https://example.com/article1",
                highlights=["First highlight from article 1"],
            ),
            _make_mock_result(
                title="Test Article 2",
                url="https://example.com/article2",
                highlights=["Second highlight from article 2"],
            ),
        ]
    response.results = results
    return response


class TestExaWebSearchTool:
    """Test cases for ExaWebSearchTool"""

    def test_tool_properties(self, exa_search_tool):
        """Test basic tool properties"""
        assert exa_search_tool.name == "exa_web_search"
        assert "search" in exa_search_tool.tags
        assert "exa" in exa_search_tool.tags
        assert exa_search_tool.args_type() == ExaWebSearchArgs
        assert exa_search_tool.return_type() == ExaWebSearchResult

    def test_sync_not_implemented(self, exa_search_tool):
        """Test that sync execution raises NotImplementedError"""
        with pytest.raises(NotImplementedError):
            exa_search_tool.run_json_sync({"query": "test"})

    @pytest.mark.asyncio
    async def test_missing_api_key(self, exa_search_tool):
        """Test behavior when API key is missing"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError, match="Missing required environment variable EXA_API_KEY"
            ):
                await exa_search_tool.run_json_async({"query": "test search"})

    @pytest.mark.asyncio
    async def test_successful_search_highlights(self, exa_search_tool):
        """Test successful search with highlights content mode"""
        mock_response = _make_mock_response()
        mock_client = MagicMock()
        mock_client.search_and_contents.return_value = mock_response
        mock_client.headers = {}

        with patch.dict(os.environ, {"EXA_API_KEY": "test_key"}):
            with patch("exa_py.Exa", return_value=mock_client):
                result = await exa_search_tool.run_json_async(
                    {
                        "query": "test search",
                        "num_results": 2,
                        "content_mode": "highlights",
                    }
                )

                assert result["results"]
                assert len(result["results"]) == 2
                assert result["results"][0]["title"] == "Test Article 1"
                assert result["results"][0]["link"] == "https://example.com/article1"
                assert "First highlight" in result["results"][0]["content"]

    @pytest.mark.asyncio
    async def test_successful_search_text(self, exa_search_tool):
        """Test successful search with text content mode"""
        mock_response = _make_mock_response()
        mock_client = MagicMock()
        mock_client.search_and_contents.return_value = mock_response
        mock_client.headers = {}

        with patch.dict(os.environ, {"EXA_API_KEY": "test_key"}):
            with patch("exa_py.Exa", return_value=mock_client):
                result = await exa_search_tool.run_json_async(
                    {"query": "test search", "content_mode": "text"}
                )

                assert result["results"]
                assert "Full text content" in result["results"][0]["content"]

    @pytest.mark.asyncio
    async def test_successful_search_summary(self, exa_search_tool):
        """Test successful search with summary content mode"""
        mock_response = _make_mock_response()
        mock_client = MagicMock()
        mock_client.search_and_contents.return_value = mock_response
        mock_client.headers = {}

        with patch.dict(os.environ, {"EXA_API_KEY": "test_key"}):
            with patch("exa_py.Exa", return_value=mock_client):
                result = await exa_search_tool.run_json_async(
                    {"query": "test search", "content_mode": "summary"}
                )

                assert result["results"]
                assert "Summary of the article" in result["results"][0]["content"]

    @pytest.mark.asyncio
    async def test_successful_search_no_content(self, exa_search_tool):
        """Test successful search without content retrieval"""
        mock_response = _make_mock_response()
        mock_client = MagicMock()
        mock_client.search.return_value = mock_response
        mock_client.headers = {}

        with patch.dict(os.environ, {"EXA_API_KEY": "test_key"}):
            with patch("exa_py.Exa", return_value=mock_client):
                result = await exa_search_tool.run_json_async(
                    {"query": "test search", "content_mode": "none"}
                )

                assert result["results"]
                # search() should be called instead of search_and_contents()
                mock_client.search.assert_called_once()
                mock_client.search_and_contents.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_with_filters(self, exa_search_tool):
        """Test search with domain and category filters"""
        mock_response = _make_mock_response()
        mock_client = MagicMock()
        mock_client.search_and_contents.return_value = mock_response
        mock_client.headers = {}

        with patch.dict(os.environ, {"EXA_API_KEY": "test_key"}):
            with patch("exa_py.Exa", return_value=mock_client):
                await exa_search_tool.run_json_async(
                    {
                        "query": "AI startups",
                        "category": "company",
                        "include_domains": ["techcrunch.com"],
                        "exclude_domains": ["reddit.com"],
                        "include_text": ["funding"],
                        "start_published_date": "2024-01-01T00:00:00Z",
                    }
                )

                call_kwargs = mock_client.search_and_contents.call_args[1]
                assert call_kwargs["category"] == "company"
                assert call_kwargs["include_domains"] == ["techcrunch.com"]
                assert call_kwargs["exclude_domains"] == ["reddit.com"]
                assert call_kwargs["include_text"] == ["funding"]
                assert call_kwargs["start_published_date"] == "2024-01-01T00:00:00Z"

    @pytest.mark.asyncio
    async def test_integration_header_set(self, exa_search_tool):
        """Test that the x-exa-integration header is set correctly"""
        mock_response = _make_mock_response()
        mock_client = MagicMock()
        mock_client.search_and_contents.return_value = mock_response
        mock_client.headers = {}

        with patch.dict(os.environ, {"EXA_API_KEY": "test_key"}):
            with patch("exa_py.Exa", return_value=mock_client):
                await exa_search_tool.run_json_async({"query": "test"})

                assert mock_client.headers["x-exa-integration"] == "xagent"

    @pytest.mark.asyncio
    async def test_empty_search_results(self, exa_search_tool):
        """Test handling when no search results are found"""
        mock_response = _make_mock_response(results=[])
        mock_client = MagicMock()
        mock_client.search_and_contents.return_value = mock_response
        mock_client.headers = {}

        with patch.dict(os.environ, {"EXA_API_KEY": "test_key"}):
            with patch("exa_py.Exa", return_value=mock_client):
                result = await exa_search_tool.run_json_async(
                    {"query": "nonexistent query"}
                )

                assert result["results"] == []

    @pytest.mark.asyncio
    async def test_api_error_handling(self, exa_search_tool):
        """Test handling of Exa API errors"""
        mock_client = MagicMock()
        mock_client.search_and_contents.side_effect = Exception(
            "API rate limit exceeded"
        )
        mock_client.headers = {}

        with patch.dict(os.environ, {"EXA_API_KEY": "test_key"}):
            with patch("exa_py.Exa", return_value=mock_client):
                with pytest.raises(ValueError, match="Error during Exa search"):
                    await exa_search_tool.run_json_async({"query": "test"})

    @pytest.mark.asyncio
    async def test_num_results_limits(self, exa_search_tool):
        """Test that num_results is properly limited between 1 and 100"""
        mock_response = _make_mock_response()
        mock_client = MagicMock()
        mock_client.search_and_contents.return_value = mock_response
        mock_client.headers = {}

        with patch.dict(os.environ, {"EXA_API_KEY": "test_key"}):
            with patch("exa_py.Exa", return_value=mock_client):
                # Test with num_results > 100 (should be limited to 100)
                await exa_search_tool.run_json_async(
                    {"query": "test", "num_results": 150}
                )
                call_kwargs = mock_client.search_and_contents.call_args[1]
                assert call_kwargs["num_results"] == 100

                # Test with num_results < 1 (should be set to 1)
                await exa_search_tool.run_json_async(
                    {"query": "test", "num_results": 0}
                )
                call_kwargs = mock_client.search_and_contents.call_args[1]
                assert call_kwargs["num_results"] == 1

    def test_args_validation(self):
        """Test ExaWebSearchArgs validation"""
        # Valid args with defaults
        args = ExaWebSearchArgs(query="test search")
        assert args.query == "test search"
        assert args.num_results == 10
        assert args.search_type == "auto"
        assert args.content_mode == "highlights"
        assert args.category is None

        # Custom args
        args = ExaWebSearchArgs(
            query="AI research",
            num_results=20,
            search_type="neural",
            content_mode="text",
            category="research paper",
            include_domains=["arxiv.org"],
        )
        assert args.query == "AI research"
        assert args.num_results == 20
        assert args.search_type == "neural"
        assert args.content_mode == "text"
        assert args.category == "research paper"
        assert args.include_domains == ["arxiv.org"]

    def test_result_model(self):
        """Test ExaWebSearchResult model"""
        results = [
            {
                "title": "Test",
                "link": "https://example.com",
                "snippet": "Test snippet",
                "content": "Test content",
            }
        ]

        result = ExaWebSearchResult(results=results)
        assert result.results == results

        dumped = result.model_dump()
        assert "results" in dumped
        assert dumped["results"] == results
