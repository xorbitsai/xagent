"""
Tests for WebSearch tool
"""

import os
from unittest.mock import AsyncMock, Mock, call, patch

import httpx
import pytest

from xagent.core.tools.adapters.vibe.web_search import (
    WebSearchArgs,
    WebSearchResult,
    WebSearchTool,
)


@pytest.fixture
def web_search_tool():
    """Create WebSearchTool instance for testing"""
    return WebSearchTool()


@pytest.fixture
def mock_google_response():
    """Mock Google Custom Search API response"""
    return {
        "items": [
            {
                "title": "Test Article 1",
                "link": "https://example.com/article1",
                "snippet": "This is a test article about search functionality",
            },
            {
                "title": "Test Article 2",
                "link": "https://example.com/article2",
                "snippet": "Another test article with different content",
            },
        ]
    }


@pytest.fixture
def mock_webpage_content():
    """Mock webpage HTML content"""
    return """
    <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Test Article</h1>
            <p>This is test content for the webpage.</p>
            <script>console.log('should be removed');</script>
        </body>
    </html>
    """


def has_real_google_credentials():
    """Check if real Google API credentials are available"""
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    return bool(api_key and cse_id and api_key != "test_key" and cse_id != "test_cse")


class TestWebSearchTool:
    """Test cases for WebSearchTool"""

    def test_tool_properties(self, web_search_tool):
        """Test basic tool properties"""
        assert web_search_tool.name == "web_search"
        assert "search" in web_search_tool.tags
        assert web_search_tool.args_type() == WebSearchArgs
        assert web_search_tool.return_type() == WebSearchResult

    def test_sync_not_implemented(self, web_search_tool):
        """Test that sync execution raises NotImplementedError"""
        with pytest.raises(NotImplementedError):
            web_search_tool.run_json_sync({"query": "test"})

    @pytest.mark.asyncio
    async def test_missing_api_credentials(self, web_search_tool):
        """Test behavior when API credentials are missing"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError, match="Missing required environment variables"
            ):
                await web_search_tool.run_json_async({"query": "test search"})

    @pytest.mark.asyncio
    async def test_successful_search_without_content(
        self, web_search_tool, mock_google_response
    ):
        """Test successful search without fetching webpage content"""
        with patch.dict(
            os.environ, {"GOOGLE_API_KEY": "test_key", "GOOGLE_CSE_ID": "test_cse"}
        ):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_google_response
            mock_response.raise_for_status.return_value = None

            with patch("httpx.AsyncClient.get", return_value=mock_response) as mock_get:
                result = await web_search_tool.run_json_async(
                    {"query": "test search", "num_results": 2}
                )

                assert result["results"]
                assert len(result["results"]) == 2
                assert result["results"][0]["title"] == "Test Article 1"
                assert result["results"][0]["link"] == "https://example.com/article1"
                assert mock_get.call_count == 1
                assert (
                    "content" not in result["results"][0]
                    or result["results"][0]["content"] == ""
                )

    @pytest.mark.asyncio
    async def test_successful_search_with_content(
        self, web_search_tool, mock_google_response
    ):
        """Test successful search with webpage content fetching"""
        with patch.dict(
            os.environ, {"GOOGLE_API_KEY": "test_key", "GOOGLE_CSE_ID": "test_cse"}
        ):
            # Mock Google API response
            mock_api_response = Mock()
            mock_api_response.status_code = 200
            mock_api_response.json.return_value = mock_google_response
            mock_api_response.raise_for_status.return_value = None

            with (
                patch("httpx.AsyncClient.get", return_value=mock_api_response),
                patch(
                    "xagent.core.tools.core.web_search.WebContentFetcher.fetch_text",
                    new_callable=AsyncMock,
                ) as mock_fetch_text,
            ):
                mock_fetch_text.side_effect = [
                    "Extracted page content 1",
                    "Extracted page content 2",
                ]
                result = await web_search_tool.run_json_async(
                    {"query": "test search", "num_results": 2, "include_content": True}
                )

                assert result["results"]
                assert len(result["results"]) == 2
                assert "content" in result["results"][0]
                assert result["results"][0]["content"] == "Extracted page content 1"
                mock_fetch_text.assert_has_awaits(
                    [
                        call("https://example.com/article1"),
                        call("https://example.com/article2"),
                    ]
                )

    @pytest.mark.asyncio
    async def test_api_403_error(self, web_search_tool):
        """Test handling of Google API 403 error"""
        with patch.dict(
            os.environ, {"GOOGLE_API_KEY": "invalid_key", "GOOGLE_CSE_ID": "test_cse"}
        ):
            mock_response = Mock()
            mock_response.status_code = 403
            mock_response.json.return_value = {
                "error": {
                    "message": "API quota exceeded",
                    "errors": [{"reason": "quotaExceeded"}],
                }
            }

            with patch("httpx.AsyncClient.get", return_value=mock_response):
                with pytest.raises(ValueError, match="API quota exceeded"):
                    await web_search_tool.run_json_async({"query": "test"})

    @pytest.mark.asyncio
    async def test_network_error(self, web_search_tool):
        """Test handling of network errors"""
        with patch.dict(
            os.environ, {"GOOGLE_API_KEY": "test_key", "GOOGLE_CSE_ID": "test_cse"}
        ):
            with patch(
                "httpx.AsyncClient.get",
                side_effect=httpx.ConnectError("Connection failed"),
            ):
                with pytest.raises(ValueError, match="Network error during search"):
                    await web_search_tool.run_json_async({"query": "test"})

    @pytest.mark.asyncio
    async def test_empty_search_results(self, web_search_tool):
        """Test handling when no search results are found"""
        with patch.dict(
            os.environ, {"GOOGLE_API_KEY": "test_key", "GOOGLE_CSE_ID": "test_cse"}
        ):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {}  # No items
            mock_response.raise_for_status.return_value = None

            with patch("httpx.AsyncClient.get", return_value=mock_response):
                result = await web_search_tool.run_json_async(
                    {"query": "nonexistent query"}
                )

                assert result["results"] == []

    @pytest.mark.asyncio
    async def test_num_results_limits(self, web_search_tool, mock_google_response):
        """Test that num_results is properly limited between 1 and 10"""
        with patch.dict(
            os.environ, {"GOOGLE_API_KEY": "test_key", "GOOGLE_CSE_ID": "test_cse"}
        ):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_google_response
            mock_response.raise_for_status.return_value = None

            with patch("httpx.AsyncClient.get", return_value=mock_response) as mock_get:
                # Test with num_results > 10 (should be limited to 10)
                await web_search_tool.run_json_async(
                    {"query": "test", "num_results": 15, "include_content": False}
                )

                # Check that the API was called with num=10
                call_args = mock_get.call_args
                params = call_args[1]["params"]
                assert params["num"] == 10

                # Test with num_results < 1 (should be set to 1)
                await web_search_tool.run_json_async(
                    {"query": "test", "num_results": 0, "include_content": False}
                )

                call_args = mock_get.call_args
                params = call_args[1]["params"]
                assert params["num"] == 1

    @pytest.mark.asyncio
    async def test_proxy_configuration(self, web_search_tool, mock_google_response):
        """Test proxy configuration from environment variables"""
        with patch.dict(os.environ, {}, clear=True):  # Start with clean environment
            os.environ["GOOGLE_API_KEY"] = "test_key"
            os.environ["GOOGLE_CSE_ID"] = "test_cse"
            os.environ["HTTP_PROXY"] = "http://proxy:8080"

            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_google_response
            mock_response.raise_for_status.return_value = None

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.get.return_value = mock_response
                mock_client_class.return_value.__aenter__.return_value = mock_client

                await web_search_tool.run_json_async(
                    {"query": "test", "include_content": False}
                )

                # Check that proxy was passed to AsyncClient
                mock_client_class.assert_called_with(proxy="http://proxy:8080")

    @pytest.mark.asyncio
    async def test_webpage_fetch_error(self, web_search_tool, mock_google_response):
        """Test handling of webpage fetch errors"""
        with patch.dict(
            os.environ, {"GOOGLE_API_KEY": "test_key", "GOOGLE_CSE_ID": "test_cse"}
        ):
            # Mock successful API response
            mock_api_response = Mock()
            mock_api_response.status_code = 200
            mock_api_response.json.return_value = mock_google_response
            mock_api_response.raise_for_status.return_value = None

            with (
                patch("httpx.AsyncClient.get", return_value=mock_api_response),
                patch(
                    "xagent.core.tools.core.web_search.WebContentFetcher.fetch_text",
                    new_callable=AsyncMock,
                    return_value=(
                        "Error fetching content: HTTP 404 error for "
                        "https://example.com/article1: Not Found"
                    ),
                ),
            ):
                result = await web_search_tool.run_json_async(
                    {"query": "test", "include_content": True}
                )

                # Should still return results, but with error content
                assert result["results"]
                assert "Error fetching content" in result["results"][0]["content"]

    def test_args_validation(self):
        """Test WebSearchArgs validation"""
        # Valid args
        args = WebSearchArgs(query="test search")
        assert args.query == "test search"
        assert args.num_results == 3  # default
        assert args.include_content is False  # default

        # Custom args
        args = WebSearchArgs(
            query="custom search", num_results=5, include_content=False
        )
        assert args.query == "custom search"
        assert args.num_results == 5
        assert args.include_content is False

    def test_result_model(self):
        """Test WebSearchResult model"""
        results = [
            {
                "title": "Test",
                "link": "https://example.com",
                "snippet": "Test snippet",
                "content": "Test content",
            }
        ]

        result = WebSearchResult(results=results)
        assert result.results == results

        # Test model dump
        dumped = result.model_dump()
        assert "results" in dumped
        assert dumped["results"] == results


class TestWebSearchToolIntegration:
    """Integration tests with real Google API (requires valid credentials)"""

    @pytest.mark.skipif(
        not has_real_google_credentials(),
        reason="Real Google API credentials not available",
    )
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_real_google_search_european_leagues_top_scorers(
        self, web_search_tool
    ):
        """
        Real integration test: Search for top scorers in European leagues

        This test requires:
        - Valid GOOGLE_API_KEY in environment
        - Valid GOOGLE_CSE_ID in environment
        - Internet connection
        """
        try:
            # Search for current top scorers in European leagues
            result = await web_search_tool.run_json_async(
                {
                    "query": "top 5 European leagues current top scorers 2024 2025 season",
                    "num_results": 5,
                    "include_content": True,
                }
            )
        except ValueError as e:
            if "429 Too Many Requests" in str(e):
                pytest.skip("Google API rate limit exceeded (429 error)")
            elif "400 Bad Request" in str(e):
                pytest.skip(
                    "Google API credentials not configured or invalid (400 error)"
                )
            elif "Network error" in str(e) or "ConnectError" in str(e):
                pytest.skip("Network connection error - skipping integration test")
            else:
                raise
        except Exception as e:
            if "429" in str(e):
                pytest.skip("Google API rate limit exceeded (429 error)")
            elif "400" in str(e):
                pytest.skip(
                    "Google API credentials not configured or invalid (400 error)"
                )
            elif "ConnectError" in str(e) or "Network error" in str(e):
                pytest.skip("Network connection error - skipping integration test")
            else:
                raise

        # Verify basic structure
        assert "results" in result
        assert isinstance(result["results"], list)
        assert len(result["results"]) > 0

        # Verify each result has required fields
        for search_result in result["results"]:
            assert "title" in search_result
            assert "link" in search_result
            assert "snippet" in search_result
            assert "content" in search_result

            # Basic validation that results are not empty
            assert len(search_result["title"]) > 0
            assert search_result["link"].startswith(("http://", "https://"))
            assert len(search_result["snippet"]) > 0

        print(
            f"\n🔍 Found {len(result['results'])} search results for European top scorers"
        )

        # Print results for manual verification
        for i, search_result in enumerate(result["results"], 1):
            print(f"\n📄 Result {i}:")
            print(f"   Title: {search_result['title']}")
            print(f"   URL: {search_result['link']}")
            print(
                f"   Snippet: {search_result['snippet'][:150]}{'...' if len(search_result['snippet']) > 150 else ''}"
            )

            # Check if content contains football/soccer related keywords
            content_lower = search_result["content"].lower()
            football_keywords = [
                "goal",
                "scorer",
                "premier league",
                "la liga",
                "series a",
                "bundesliga",
                "ligue 1",
                "champions league",
                "football",
                "soccer",
                "striker",
                "forward",
            ]

            found_keywords = [
                keyword for keyword in football_keywords if keyword in content_lower
            ]
            print(
                f"   Football keywords found: {found_keywords[:5]}"
            )  # Show first 5 matches

        # Verify that at least one result contains football-related content
        all_content = " ".join(
            [
                r["content"] + " " + r["title"] + " " + r["snippet"]
                for r in result["results"]
            ]
        ).lower()

        football_indicators = [
            "goal",
            "scorer",
            "football",
            "soccer",
            "premier league",
            "la liga",
            "series a",
            "bundesliga",
            "ligue 1",
        ]

        has_football_content = any(
            indicator in all_content for indicator in football_indicators
        )
        assert has_football_content, (
            "Search results should contain football-related content"
        )

        print("\n✅ Integration test completed successfully!")
        print(f"   Total results: {len(result['results'])}")
        print("   All results have required fields")
        print("   Football-related content detected")

    @pytest.mark.skipif(
        not has_real_google_credentials(),
        reason="Real Google API credentials not available",
    )
    @pytest.mark.asyncio
    async def test_real_google_search_without_content(self, web_search_tool):
        """Test real Google search without fetching webpage content"""
        try:
            result = await web_search_tool.run_json_async(
                {
                    "query": "UEFA Champions League 2024",
                    "num_results": 3,
                    "include_content": False,
                }
            )
        except ValueError as e:
            if "429 Too Many Requests" in str(e):
                pytest.skip("Google API rate limit exceeded (429 error)")
            elif "400 Bad Request" in str(e):
                pytest.skip(
                    "Google API credentials not configured or invalid (400 error)"
                )
            elif "Network error" in str(e) or "ConnectError" in str(e):
                pytest.skip("Network connection error - skipping integration test")
            else:
                raise
        except Exception as e:
            if "429" in str(e):
                pytest.skip("Google API rate limit exceeded (429 error)")
            elif "400" in str(e):
                pytest.skip(
                    "Google API credentials not configured or invalid (400 error)"
                )
            elif "ConnectError" in str(e) or "Network error" in str(e):
                pytest.skip("Network connection error - skipping integration test")
            else:
                raise

        assert "results" in result
        assert len(result["results"]) <= 3

        for search_result in result["results"]:
            assert "title" in search_result
            assert "link" in search_result
            assert "snippet" in search_result
            # Content should be empty or not present when include_content=False
            assert search_result.get("content", "") == ""
