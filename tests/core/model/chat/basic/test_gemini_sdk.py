"""Test cases for Gemini LLM implementation using official SDK."""

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
import pytest_mock

from xagent.core.model.chat.basic.gemini import GeminiLLM


@pytest.fixture
def gemini_llm_config() -> Dict[str, Any]:
    """Gemini LLM configuration for testing."""
    return {
        "model_name": "gemini-2.0-flash-exp",
        "api_key": "test-api-key",
    }


class TestGeminiLLMSDK:
    """Test cases for Gemini LLM implementation using official SDK."""

    @pytest.fixture
    def llm(self, gemini_llm_config: Dict[str, str]) -> GeminiLLM:
        """Fixture providing Gemini LLM instance."""
        return GeminiLLM(**gemini_llm_config)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_basic_chat_completion_with_sdk(
        self, llm: GeminiLLM, mocker: pytest_mock.MockerFixture
    ) -> None:
        """Test basic chat completion functionality using SDK."""
        # Mock the SDK client at the genai module level
        mock_sdk_client = MagicMock()
        mock_response = MagicMock()
        mock_candidate = MagicMock()
        mock_content = MagicMock()
        mock_part = MagicMock()

        # Setup the mock response structure
        mock_part.text = "Hello World"
        mock_part.function_call = None
        mock_content.parts = [mock_part]
        mock_candidate.content = mock_content
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 5

        # Mock the Client constructor
        mocker.patch("google.genai.Client", return_value=mock_sdk_client)
        mock_sdk_client.models.generate_content.return_value = mock_response

        messages = [
            {
                "role": "user",
                "content": "Hello! Please respond with just 'Hello World'.",
            }
        ]

        response = await llm.chat(messages)

        # Verify response
        assert isinstance(response, str)
        assert response == "Hello World"
        print(f"Basic chat response: {response}")

    @pytest.mark.asyncio
    async def test_stream_chat_with_sdk(
        self, llm: GeminiLLM, mocker: pytest_mock.MockerFixture
    ) -> None:
        """Test streaming chat functionality using SDK."""
        # Mock the SDK client for streaming
        mock_client = MagicMock()

        # Create mock chunks
        mock_chunk1 = MagicMock()
        mock_chunk1.usage_metadata = MagicMock()
        mock_chunk1.usage_metadata.prompt_token_count = 8
        mock_chunk1.usage_metadata.candidates_token_count = 9
        mock_chunk1.candidates = None  # First chunk has usage only

        mock_chunk2 = MagicMock()
        mock_candidate2 = MagicMock()
        mock_content2 = MagicMock()
        mock_part2 = MagicMock()
        mock_part2.text = "1"
        mock_part2.function_call = None
        mock_content2.parts = [mock_part2]
        mock_candidate2.content = mock_content2
        mock_chunk2.candidates = [mock_candidate2]
        mock_chunk2.usage_metadata = None

        mock_chunk3 = MagicMock()
        mock_chunk3.candidates = None  # End chunk

        # Setup streaming response
        mock_client.models.generate_content_stream.return_value = [
            mock_chunk1,
            mock_chunk2,
            mock_chunk3,
        ]

        # Patch the _ensure_client to return our mock
        mocker.patch.object(llm, "_ensure_client")
        llm._client = mock_client

        messages = [{"role": "user", "content": "Count from 1 to 3."}]

        chunks = []
        usage_received = False
        async for chunk in llm.stream_chat(messages):
            chunks.append(chunk)
            if chunk.is_usage():
                usage_received = True

        # Verify we got the expected chunks
        assert len(chunks) >= 2  # At least usage + token + end
        assert usage_received, "Usage chunk should be received"
        print(f"Stream chat got {len(chunks)} chunks, usage received: {usage_received}")

    @pytest.mark.asyncio
    async def test_tool_calling_with_sdk(
        self, llm: GeminiLLM, mocker: pytest_mock.MockerFixture
    ) -> None:
        """Test tool calling functionality using SDK."""
        # Mock the SDK client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_candidate = MagicMock()
        mock_content = MagicMock()
        mock_part = MagicMock()
        mock_function_call = MagicMock()

        # Setup function call mock
        mock_function_call.name = "get_weather"
        mock_function_call.args = {"location": "Boston"}
        mock_part.function_call = mock_function_call
        mock_part.text = None
        mock_content.parts = [mock_part]
        mock_candidate.content = mock_content
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 15
        mock_response.usage_metadata.candidates_token_count = 10

        mock_client.models.generate_content.return_value = mock_response

        # Patch the _ensure_client to return our mock
        mocker.patch.object(llm, "_ensure_client")
        llm._client = mock_client

        messages = [{"role": "user", "content": "What's the weather like in Boston?"}]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state",
                            }
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        response = await llm.chat(messages, tools=tools)

        # Verify tool call response structure
        assert isinstance(response, dict)
        assert response.get("type") == "tool_call"
        assert "tool_calls" in response

        tool_calls = response["tool_calls"]
        assert len(tool_calls) > 0
        assert tool_calls[0]["function"]["name"] == "get_weather"
        print(f"Tool calling response: {response}")

    @pytest.mark.asyncio
    async def test_json_mode_with_sdk(
        self, llm: GeminiLLM, mocker: pytest_mock.MockerFixture
    ) -> None:
        """Test JSON mode (should_chat_directly scenario) using SDK."""
        # Mock the SDK client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_candidate = MagicMock()
        mock_content = MagicMock()
        mock_part = MagicMock()

        # Setup JSON response
        mock_part.text = '{"greeting": "Hello", "count": 1}'
        mock_part.function_call = None
        mock_content.parts = [mock_part]
        mock_candidate.content = mock_content
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 14
        mock_response.usage_metadata.candidates_token_count = 20

        mock_client.models.generate_content.return_value = mock_response

        # Patch the _ensure_client to return our mock
        mocker.patch.object(llm, "_ensure_client")
        llm._client = mock_client

        messages = [
            {
                "role": "user",
                "content": 'Respond with JSON with keys "greeting" and "count".',
            }
        ]

        response = await llm.chat(messages, response_format={"type": "json_object"})

        # Verify JSON response
        assert isinstance(response, str)
        assert "greeting" in response
        assert "count" in response
        print(f"JSON mode response: {response}")

    @pytest.mark.asyncio
    async def test_base_url_handling(self, gemini_llm_config: Dict[str, str]) -> None:
        """Test that base_url is correctly handled (version path removal)."""
        # Test 1: base_url without version path
        llm1 = GeminiLLM(base_url="https://proxy.com", **gemini_llm_config)
        llm1._ensure_client()
        # Should keep the URL as-is
        assert llm1._client is not None

        # Test 2: base_url with /v1beta
        llm2 = GeminiLLM(base_url="https://proxy.com/v1beta", **gemini_llm_config)
        llm2._ensure_client()
        # Should remove /v1beta from the URL (SDK adds it back)
        assert llm2._client is not None

        # Test 3: base_url with /v1
        llm3 = GeminiLLM(base_url="https://proxy.com/v1", **gemini_llm_config)
        llm3._ensure_client()
        # Should remove /v1 from the URL
        assert llm3._client is not None

        print("Base URL handling tests passed")

    @pytest.mark.asyncio
    async def test_context_manager(self, gemini_llm_config: Dict[str, str]) -> None:
        """Test async context manager functionality."""
        async with GeminiLLM(**gemini_llm_config) as llm:  # type: ignore[arg-type]
            # Verify client is initialized
            assert llm._client is not None
            print("Context manager test passed")

        # Verify the client was properly closed
        assert llm._client is None

    @pytest.mark.asyncio
    async def test_error_handling_missing_api_key(
        self, gemini_llm_config: Dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test error handling when API key is missing."""
        # Remove all API key environment variables
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        # Create LLM without API key
        config = gemini_llm_config.copy()
        config["api_key"] = None  # type: ignore[assignment]

        llm = GeminiLLM(**config)  # type: ignore[arg-type]
        messages = [{"role": "user", "content": "Hello"}]

        # Should raise a RuntimeError
        with pytest.raises(
            RuntimeError, match="GEMINI_API_KEY or GOOGLE_API_KEY must be set"
        ):
            await llm.chat(messages)

    @pytest.mark.asyncio
    async def test_cleanup(self, gemini_llm_config: Dict[str, str]) -> None:
        """Test that client cleanup works properly."""
        llm = GeminiLLM(**gemini_llm_config)  # type: ignore[arg-type]

        # Initialize client
        llm._ensure_client()

        # Verify client was created
        assert llm._client is not None

        # Close the client
        await llm.close()

        # Verify client is closed
        assert llm._client is None

    @pytest.mark.asyncio
    async def test_abilities_property(self, gemini_llm_config: Dict[str, str]) -> None:
        """Test that abilities are correctly set based on model name."""
        # Test vision model
        vision_llm = GeminiLLM(model_name="gemini-pro-vision", api_key="test-key")
        assert "vision" in vision_llm.abilities

        # Test non-vision model
        chat_llm = GeminiLLM(model_name="gemini-1.5-pro", api_key="test-key")
        assert "vision" not in chat_llm.abilities
        assert "chat" in chat_llm.abilities
        assert "tool_calling" in chat_llm.abilities

    @pytest.mark.asyncio
    async def test_supports_thinking_mode(self, llm: GeminiLLM) -> None:
        """Test that Gemini does not support thinking mode."""
        assert llm.supports_thinking_mode is False

    @pytest.mark.asyncio
    async def test_429_rate_limit_error_is_retryable(
        self, llm: GeminiLLM, mocker: pytest_mock.MockerFixture
    ) -> None:
        """Test that 429 rate limit errors are properly caught and converted to LLMRetryableError."""
        from google.genai import errors as genai_errors

        from xagent.core.model.chat.exceptions import LLMRetryableError

        # Create a mock 429 error
        mock_response = mocker.MagicMock()
        mock_response.status_code = 429
        mock_response.json.return_value = {
            "error": {
                "code": 429,
                "message": "RESOURCE_EXHAUSTED",
                "status": "PERMISSION_DENIED",
            }
        }

        # Mock the SDK to raise 429 error
        mock_client = mocker.MagicMock()
        mock_client.models.generate_content.side_effect = genai_errors.ClientError(
            code=429,
            response_json={"error": {"code": 429, "message": "RESOURCE_EXHAUSTED"}},
            response=mock_response,
        )

        # Patch the Client constructor
        mocker.patch("google.genai.Client", return_value=mock_client)

        messages = [{"role": "user", "content": "Test"}]

        # Should raise LLMRetryableError
        with pytest.raises(LLMRetryableError, match="code=429"):
            await llm.chat(messages)

        print("✅ 429 rate limit error correctly caught as retryable")

    @pytest.mark.asyncio
    async def test_500_server_error_is_retryable(
        self, llm: GeminiLLM, mocker: pytest_mock.MockerFixture
    ) -> None:
        """Test that 500 server errors are properly caught and converted to LLMRetryableError."""
        from google.genai import errors as genai_errors

        from xagent.core.model.chat.exceptions import LLMRetryableError

        # Create a mock 500 error
        mock_response = mocker.MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {
            "error": {
                "code": 500,
                "message": "Internal Server Error",
            }
        }

        # Mock the SDK to raise 500 error
        mock_client = mocker.MagicMock()
        mock_client.models.generate_content.side_effect = genai_errors.ClientError(
            code=500,
            response_json={"error": {"code": 500, "message": "Internal Server Error"}},
            response=mock_response,
        )

        # Patch the Client constructor
        mocker.patch("google.genai.Client", return_value=mock_client)

        messages = [{"role": "user", "content": "Test"}]

        # Should raise LLMRetryableError
        with pytest.raises(LLMRetryableError, match="code=500"):
            await llm.chat(messages)

        print("✅ 500 server error correctly caught as retryable")
