"""Test cases for Gemini LLM implementation using REST API."""

from typing import Any, Dict
from unittest.mock import AsyncMock

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


class TestGeminiLLM:
    """Test cases for Gemini LLM implementation using REST API."""

    @pytest.fixture
    def mock_response_text(self, mocker: pytest_mock.MockerFixture) -> Dict[str, Any]:
        """Mock the _call_gemini_rest_api method to return text response."""
        return {"candidates": [{"content": {"parts": [{"text": "Hello World"}]}}]}

    @pytest.fixture
    def mock_response_function_call(
        self, mocker: pytest_mock.MockerFixture
    ) -> Dict[str, Any]:
        """Mock the _call_gemini_rest_api method to return function call."""
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    "args": {"location": "Boston"},
                                }
                            }
                        ]
                    }
                }
            ]
        }

    @pytest.fixture
    def llm(self, gemini_llm_config: Dict[str, str]) -> GeminiLLM:
        """Fixture providing Gemini LLM instance."""
        return GeminiLLM(**gemini_llm_config)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_basic_chat_completion(
        self,
        llm: GeminiLLM,
        mock_response_text: Dict[str, Any],
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Test basic chat completion functionality."""
        # Mock the internal API call method
        mocker.patch.object(
            llm, "_call_gemini_rest_api", new=AsyncMock(return_value=mock_response_text)
        )

        messages = [
            {
                "role": "user",
                "content": "Hello! Please respond with just 'Hello World'.",
            },
        ]

        response = await llm.chat(messages)

        # Verify response is a non-empty string
        assert isinstance(response, str)
        assert response == "Hello World"
        print(f"Basic chat response: {response}")

    @pytest.mark.asyncio
    async def test_tool_calling(
        self,
        llm: GeminiLLM,
        mock_response_function_call: Dict[str, Any],
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Test tool calling functionality."""
        # Mock the internal API call method
        mocker.patch.object(
            llm,
            "_call_gemini_rest_api",
            new=AsyncMock(return_value=mock_response_function_call),
        )

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
                                "description": "The city and state, e.g. San Francisco, CA",
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
    async def test_custom_parameters(
        self,
        llm: GeminiLLM,
        mock_response_text: Dict[str, Any],
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Test custom parameters like temperature and max_tokens."""
        # Mock the internal API call method
        mocker.patch.object(
            llm, "_call_gemini_rest_api", new=AsyncMock(return_value=mock_response_text)
        )

        messages = [{"role": "user", "content": "Count from 1 to 3."}]

        # Test with custom temperature and max_tokens
        response = await llm.chat(
            messages,
            temperature=0.1,  # Low temperature for more deterministic output
            max_tokens=50,  # Limit response length
        )

        assert isinstance(response, str)
        assert response == "Hello World"
        print(f"Custom parameters response: {response}")

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
    async def test_vision_chat(
        self,
        gemini_llm_config: Dict[str, str],
        mock_response_text: Dict[str, Any],
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Test vision chat functionality."""
        # Test with vision-capable model
        vision_llm = GeminiLLM(model_name="gemini-pro-vision", api_key="test-key")

        # Mock the internal API call method
        mocker.patch.object(
            vision_llm,
            "_call_gemini_rest_api",
            new=AsyncMock(return_value=mock_response_text),
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
                        },
                    },
                ],
            }
        ]

        response = await vision_llm.vision_chat(messages)

        assert isinstance(response, str)
        assert response == "Hello World"
        print(f"Vision chat response: {response}")

    @pytest.mark.asyncio
    async def test_http_error_handling(
        self, llm: GeminiLLM, mocker: pytest_mock.MockerFixture
    ) -> None:
        """Test HTTP error handling."""
        # Mock httpx to raise HTTPStatusError
        import httpx

        mock_response = mocker.Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"

        mock_error = httpx.HTTPStatusError(
            "Bad Request", request=mocker.Mock(), response=mock_response
        )
        mock_response.raise_for_status.side_effect = mock_error

        # Mock the internal API call method to raise error
        mocker.patch.object(
            llm, "_call_gemini_rest_api", new=AsyncMock(side_effect=mock_error)
        )

        messages = [{"role": "user", "content": "Hello"}]

        # Should raise a RuntimeError
        with pytest.raises(RuntimeError):
            await llm.chat(messages)

    @pytest.mark.asyncio
    async def test_retryable_errors(self, llm, mocker):
        """Test that httpx errors are converted to LLMRetryableError."""
        import httpx

        from xagent.core.model.chat.exceptions import LLMRetryableError

        # Case 1: Timeout
        mocker.patch.object(
            llm, "_call_gemini_rest_api", side_effect=httpx.TimeoutException("Timeout")
        )
        with pytest.raises(LLMRetryableError):
            await llm.chat([{"role": "user", "content": "hi"}])

        # Case 2: Network Error
        mocker.patch.object(
            llm,
            "_call_gemini_rest_api",
            side_effect=httpx.NetworkError("Network Error"),
        )
        with pytest.raises(LLMRetryableError):
            await llm.chat([{"role": "user", "content": "hi"}])

        # Case 3: 429 Rate Limit
        mock_response = mocker.Mock()
        mock_response.status_code = 429
        error_429 = httpx.HTTPStatusError(
            "Rate Limit", request=None, response=mock_response
        )

        mocker.patch.object(llm, "_call_gemini_rest_api", side_effect=error_429)
        with pytest.raises(LLMRetryableError):
            await llm.chat([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_400_error_is_retryable(
        self, llm: GeminiLLM, mocker: pytest_mock.MockerFixture
    ):
        """Test that 400 Bad Request errors are retryable."""
        import httpx

        from xagent.core.model.chat.exceptions import LLMRetryableError

        mock_response = mocker.Mock()
        mock_response.status_code = 400
        error_400 = httpx.HTTPStatusError(
            "Bad Request", request=None, response=mock_response
        )

        mocker.patch.object(llm, "_call_gemini_rest_api", side_effect=error_400)
        with pytest.raises(LLMRetryableError):
            await llm.chat([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_stream_chat_network_error_is_retryable(
        self, llm: GeminiLLM, mocker: pytest_mock.MockerFixture
    ):
        """Test that network errors in stream_chat raise LLMRetryableError."""
        import httpx

        from xagent.core.model.chat.exceptions import LLMRetryableError

        # Mock _stream_gemini_rest_api to be an async generator that raises network error
        async def mock_stream_error(*args, **kwargs):
            raise httpx.NetworkError("Connection failed")
            yield  # Make it an async generator (never reached)

        mocker.patch.object(llm, "_stream_gemini_rest_api", mock_stream_error)

        with pytest.raises(LLMRetryableError):
            async for _ in llm.stream_chat([{"role": "user", "content": "hi"}]):
                pass

    @pytest.mark.asyncio
    async def test_stream_chat_400_error_is_retryable(
        self, llm: GeminiLLM, mocker: pytest_mock.MockerFixture
    ):
        """Test that 400 errors in stream_chat raise LLMRetryableError."""
        import httpx

        from xagent.core.model.chat.exceptions import LLMRetryableError

        mock_response = mocker.Mock()
        mock_response.status_code = 400
        error_400 = httpx.HTTPStatusError(
            "Bad Request", request=None, response=mock_response
        )

        # Mock _stream_gemini_rest_api to be an async generator that raises 400 error
        async def mock_stream_error(*args, **kwargs):
            raise error_400
            yield  # Make it an async generator (never reached)

        mocker.patch.object(llm, "_stream_gemini_rest_api", mock_stream_error)

        with pytest.raises(LLMRetryableError):
            async for _ in llm.stream_chat([{"role": "user", "content": "hi"}]):
                pass

    @pytest.mark.asyncio
    async def test_list_available_models_with_default_base_url(self, mocker):
        """Test listing available models using default base URL (official API)."""
        # Mock httpx response
        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {
                    "name": "models/gemini-2.0-flash-exp",
                    "displayName": "Gemini 2.0 Flash Experimental",
                    "description": "Fast and experimental",
                    "version": "v1beta",
                    "supportedGenerationMethods": [
                        "generateContent",
                        "streamGenerateContent",
                    ],
                },
                {
                    "name": "models/gemini-1.5-pro",
                    "displayName": "Gemini 1.5 Pro",
                    "description": "Capable model",
                    "version": "v1beta",
                    "supportedGenerationMethods": ["generateContent"],
                },
            ],
        }

        mock_async_client = mocker.AsyncMock()
        mock_async_client.get.return_value = mock_response
        mock_async_client.__aenter__.return_value = mock_async_client
        mock_async_client.__aexit__.return_value = None

        mocker.patch("httpx.AsyncClient", return_value=mock_async_client)

        # Call without base_url - should use official API
        models = await GeminiLLM.list_available_models("test-api-key")

        # Verify results
        assert len(models) == 2
        # Models are sorted by name alphabetically
        assert models[0]["id"] == "models/gemini-1.5-pro"
        assert models[1]["id"] == "models/gemini-2.0-flash-exp"

        # Verify the API was called with official base URL
        mock_async_client.get.assert_called_once()
        call_args = mock_async_client.get.call_args
        assert "googleapis.com/v1beta/models" in call_args[0][0]
        assert call_args[1]["headers"]["x-goog-api-key"] == "test-api-key"

    @pytest.mark.asyncio
    async def test_list_available_models_with_custom_base_url(self, mocker):
        """Test listing available models using custom base URL."""
        # Mock httpx response
        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {
                    "name": "models/custom-gemini",
                    "displayName": "Custom Gemini",
                    "description": "Custom model",
                    "version": "v1",
                    "supportedGenerationMethods": ["generateContent"],
                },
            ],
        }

        mock_async_client = mocker.AsyncMock()
        mock_async_client.get.return_value = mock_response
        mock_async_client.__aenter__.return_value = mock_async_client
        mock_async_client.__aexit__.return_value = None

        mocker.patch("httpx.AsyncClient", return_value=mock_async_client)

        # Call with custom base_url
        custom_base_url = "https://custom-proxy.com/v1beta"
        models = await GeminiLLM.list_available_models(
            "test-api-key", base_url=custom_base_url
        )

        # Verify results
        assert len(models) == 1
        assert models[0]["id"] == "models/custom-gemini"

        # Verify the API was called with custom base URL
        mock_async_client.get.assert_called_once()
        call_args = mock_async_client.get.call_args
        assert "custom-proxy.com/v1beta/models" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_list_available_models_unauthorized(self, mocker):
        """Test listing models with invalid API key."""
        import httpx

        # Mock httpx to raise 401 error
        mock_response = mocker.MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        error = httpx.HTTPStatusError(
            "Unauthorized", request=mocker.MagicMock(), response=mock_response
        )

        mock_async_client = mocker.AsyncMock()
        mock_async_client.get.side_effect = error
        mock_async_client.__aenter__.return_value = mock_async_client
        mock_async_client.__aexit__.return_value = None

        mocker.patch("httpx.AsyncClient", return_value=mock_async_client)

        # Should raise ValueError for invalid API key
        with pytest.raises(ValueError, match="Invalid Google API key"):
            await GeminiLLM.list_available_models("invalid-key")

    @pytest.mark.asyncio
    async def test_output_config_json_schema(
        self, llm: GeminiLLM, mocker: pytest_mock.MockerFixture
    ) -> None:
        """Test output_config with json_schema format for Gemini."""
        # Create a mock response with JSON schema content
        mock_response_data = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": '{"recipe_name": "Chocolate Chip Cookies", "ingredients": [{"name": "flour", "quantity": "2 and 1/4 cups"}], "instructions": ["Preheat oven to 375°F"]}'
                            }
                        ]
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 50,
            },
        }

        # Mock the internal API call method
        mock_call = AsyncMock(return_value=mock_response_data)
        mocker.patch.object(llm, "_call_gemini_rest_api", new=mock_call)

        messages = [{"role": "user", "content": "Extract the recipe from this text."}]

        # Test with output_config using json_schema format (Gemini 3.0+)
        output_config = {
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "recipe_name": {
                            "type": "string",
                            "description": "The name of the recipe.",
                        },
                        "ingredients": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "quantity": {"type": "string"},
                                },
                                "required": ["name", "quantity"],
                            },
                        },
                        "instructions": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["recipe_name", "ingredients", "instructions"],
                },
            }
        }

        response = await llm.chat(messages, output_config=output_config)

        assert isinstance(response, str)
        # Verify the response contains the expected JSON
        assert "recipe_name" in response
        assert "ingredients" in response

        # Verify the API was called with response_mime_type and response_json_schema
        mock_call.assert_called_once()
        call_args = mock_call.call_args
        # Check the generation_config parameter
        gen_config = call_args.kwargs.get("generation_config", {})
        assert gen_config.get("response_mime_type") == "application/json"
        assert "response_json_schema" in gen_config
