"""Test cases for OpenAI LLM implementation using OpenAI SDK."""

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from xagent.core.model.chat.basic.openai import OpenAILLM


class TestOpenAILLM:
    """Test cases for OpenAI LLM implementation."""

    @pytest.fixture
    def llm(self, openai_llm_config):
        """Fixture providing OpenAI LLM instance."""
        return OpenAILLM(**openai_llm_config)

    @pytest.mark.asyncio
    async def test_basic_chat_completion(self, llm, mock_chat_completion, mocker):
        """Test basic chat completion functionality."""
        # Setup mock
        mock_client = mocker.AsyncMock()
        mock_client.chat.completions.create.return_value = mock_chat_completion
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI",
            return_value=mock_client,
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Hello! Please respond with just 'Hello World'.",
            },
        ]

        response = await llm.chat(messages)

        # Verify response is a dict with text content
        assert isinstance(response, dict)
        assert response.get("type") == "text"
        assert response.get("content") == "Hello World"
        print(f"Basic chat response: {response}")

        # Verify the API was called with correct parameters
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "gpt-4o-mini"
        assert call_args.kwargs["messages"] == messages
        assert call_args.kwargs["temperature"] == 0.7
        # max_tokens should not be in the call if not explicitly provided
        assert "max_tokens" not in call_args.kwargs

    @pytest.mark.asyncio
    async def test_tool_calling(self, llm, mock_tool_call_completion, mocker):
        """Test tool calling functionality."""
        # Setup mock
        mock_client = mocker.AsyncMock()
        mock_client.chat.completions.create.return_value = mock_tool_call_completion
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI",
            return_value=mock_client,
        )

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant with access to tools.",
            },
            {
                "role": "user",
                "content": "What's the weather like in Boston? Please use the get_weather tool.",
            },
        ]

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
        assert tool_calls[0]["id"] == "call_test"
        print(f"Tool calling response: {response}")

        # Verify the API was called with tools
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert "tools" in call_args.kwargs
        assert call_args.kwargs["tools"] == tools

    @pytest.mark.asyncio
    async def test_json_mode(self, llm, mock_json_completion, mocker):
        """Test JSON mode functionality."""
        # Setup mock
        mock_client = mocker.AsyncMock()
        mock_client.chat.completions.create.return_value = mock_json_completion
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI",
            return_value=mock_client,
        )

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that responds in JSON format.",
            },
            {
                "role": "user",
                "content": "Please provide a simple JSON object with 'name' and 'age' fields.",
            },
        ]

        response = await llm.chat(messages, response_format={"type": "json_object"})

        # Verify JSON response
        assert isinstance(response, dict)
        assert response.get("type") == "text"

        # Try to parse as JSON
        content = response.get("content", "")
        parsed = json.loads(content)
        assert isinstance(parsed, dict)
        assert "name" in parsed
        assert "age" in parsed
        print(f"JSON mode response: {parsed}")

        # Verify response_format was passed
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_context_manager(
        self, openai_llm_config, mock_chat_completion, mocker
    ):
        """Test async context manager functionality."""
        # Setup mock
        mock_client = mocker.AsyncMock()
        mock_client.chat.completions.create.return_value = mock_chat_completion
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI",
            return_value=mock_client,
        )

        async with OpenAILLM(**openai_llm_config) as ctx_llm:
            messages = [{"role": "user", "content": "Say 'test'"}]
            response = await ctx_llm.chat(messages)

            assert isinstance(response, dict)
            assert response.get("type") == "text"
            assert response.get("content") == "Hello World"
            print(f"Context manager response: {response}")

        # Verify the client was properly closed
        assert ctx_llm._client is None

    @pytest.mark.asyncio
    async def test_error_handling_invalid_model(self, mocker):
        """Test error handling with invalid model name."""
        # Setup mock to raise an API error
        mock_client = mocker.AsyncMock()

        import httpx
        from openai import APIError

        # Create a mock request
        mock_request = httpx.Request(
            "POST", "https://api.openai.com/v1/chat/completions"
        )

        mock_client.chat.completions.create.side_effect = APIError(
            "Invalid model",
            request=mock_request,
            body={"error": {"message": "Model not found"}},
        )
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI",
            return_value=mock_client,
        )

        # Use a clearly invalid model name
        llm = OpenAILLM(
            model_name="invalid-model-name-that-does-not-exist-12345",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
        )

        messages = [{"role": "user", "content": "Hello"}]

        # Should raise a RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            await llm.chat(messages)

        # Verify error message contains API error information
        error_msg = str(exc_info.value)
        assert "OpenAI API error" in error_msg
        print(f"Error handling test passed: {error_msg}")

    @pytest.mark.asyncio
    async def test_custom_parameters(self, llm, mock_chat_completion, mocker):
        """Test custom parameters like temperature and max_tokens."""
        # Setup mock
        mock_client = mocker.AsyncMock()
        mock_client.chat.completions.create.return_value = mock_chat_completion
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI",
            return_value=mock_client,
        )

        messages = [{"role": "user", "content": "Count from 1 to 3."}]

        # Test with custom temperature and max_tokens
        response = await llm.chat(
            messages,
            temperature=0.1,  # Low temperature for more deterministic output
            max_tokens=50,  # Limit response length
        )

        assert isinstance(response, dict)
        assert response.get("type") == "text"
        assert response.get("content") == "Hello World"
        print(f"Custom parameters response: {response}")

        # Verify custom parameters were passed
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["temperature"] == 0.1
        assert call_args.kwargs["max_tokens"] == 50

    @pytest.mark.asyncio
    async def test_cleanup(self, openai_llm_config, mock_chat_completion, mocker):
        """Test that client cleanup works properly."""
        # Setup mock
        mock_client = mocker.AsyncMock()
        mock_client.chat.completions.create.return_value = mock_chat_completion
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI",
            return_value=mock_client,
        )

        llm = OpenAILLM(**openai_llm_config)

        # Make a request to ensure client is initialized
        messages = [{"role": "user", "content": "Hello"}]
        await llm.chat(messages)

        # Verify client was created
        assert llm._client is not None

        # Close the client
        await llm.close()

        # Verify client is closed (this is mainly to ensure no exceptions are raised)
        assert llm._client is None

    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(
        self, llm, mock_chat_completion, mocker
    ):
        """Test handling multiple concurrent requests."""
        # Setup mock
        mock_client = mocker.AsyncMock()
        mock_client.chat.completions.create.return_value = mock_chat_completion
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI",
            return_value=mock_client,
        )

        messages = [{"role": "user", "content": "Respond with just 'OK'"}]

        # Make multiple concurrent requests
        tasks = [llm.chat(messages) for _ in range(3)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # All requests should succeed
        for i, response in enumerate(responses):
            assert not isinstance(response, Exception), (
                f"Request {i} failed: {response}"
            )
            assert isinstance(response, dict)
            assert response.get("content") == "Hello World"

        print(f"Concurrent requests test passed with {len(responses)} responses")

        # Verify multiple calls were made
        assert mock_client.chat.completions.create.call_count == 3

    @pytest.mark.asyncio
    async def test_none_content_response(self, openai_llm_config, mocker):
        """Test handling of None content response."""
        # Setup mock with None content
        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = None
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = mocker.AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI",
            return_value=mock_client,
        )

        llm = OpenAILLM(**openai_llm_config)

        # Should raise RuntimeError when content is None and no tool calls
        with pytest.raises(
            RuntimeError, match="LLM returned None content and no tool calls"
        ):
            await llm.chat([{"role": "user", "content": "Hello"}])

    @pytest.mark.asyncio
    async def test_empty_content_response(self, openai_llm_config, mocker):
        """Test handling of empty string content response."""
        # Setup mock with empty string content
        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_message = MagicMock()
        mock_message.content = ""
        mock_message.tool_calls = None
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = mocker.AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI",
            return_value=mock_client,
        )

        llm = OpenAILLM(**openai_llm_config)

        # Should raise RuntimeError when content is empty and no tool calls
        with pytest.raises(
            RuntimeError, match="LLM returned empty content and no tool calls"
        ):
            await llm.chat([{"role": "user", "content": "Hello"}])

    @pytest.mark.asyncio
    async def test_empty_string_api_key(self, openai_llm_config, monkeypatch):
        """Test that empty string API key is allowed and does not fallback to environment variable."""
        # Set environment variable to ensure we can test that it's NOT used
        monkeypatch.setenv("OPENAI_API_KEY", "env-api-key-should-not-be-used")

        # Create LLM with empty string API key
        config = openai_llm_config.copy()
        config["api_key"] = ""  # Empty string

        llm = OpenAILLM(**config)

        # Verify that the API key is empty string, not the environment variable
        assert llm.api_key == ""
        print(
            f"Empty string API key test passed: API key is '{llm.api_key}' (not using env var)"
        )

    @pytest.mark.asyncio
    async def test_none_api_key_with_env_fallback(self, openai_llm_config, monkeypatch):
        """Test None API key with environment variable fallback."""
        # Set environment variable
        env_api_key = "env-api-key-for-fallback"
        monkeypatch.setenv("OPENAI_API_KEY", env_api_key)

        # Create LLM with None API key
        config = openai_llm_config.copy()
        config["api_key"] = None

        llm = OpenAILLM(**config)

        # Verify that the API key is from environment variable
        assert llm.api_key == env_api_key
        print(f"None API key with env fallback test passed: API key is '{llm.api_key}'")

    @pytest.mark.asyncio
    async def test_missing_api_key_initialization(self, openai_llm_config, monkeypatch):
        """Test LLM initialization when API key is completely missing."""
        # Remove environment variable
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        # Create LLM with None API key and no environment variable
        config = openai_llm_config.copy()
        config["api_key"] = None

        llm = OpenAILLM(**config)

        # The LLM should initialize with None API key
        # OpenAI SDK will handle the missing API key when making requests
        assert llm.api_key is None
        print(f"Missing API key test: LLM initialized with API key = {llm.api_key}")

    @pytest.mark.asyncio
    async def test_empty_string_api_key_request(
        self, openai_llm_config, monkeypatch, mocker
    ):
        """Test making a request with empty string API key."""
        # Remove environment variable to ensure we're testing empty string behavior
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        # Create LLM with empty string API key
        config = openai_llm_config.copy()
        config["api_key"] = ""  # Empty string

        # Mock AsyncOpenAI constructor to capture its arguments
        mock_async_openai = mocker.MagicMock()
        mock_client = mocker.AsyncMock()
        mock_async_openai.return_value = mock_client

        # Create a mock response
        mock_choice = mocker.Mock()
        mock_choice.finish_reason = "stop"
        mock_message = mocker.Mock()
        mock_message.content = "Test response"
        mock_message.tool_calls = None
        mock_choice.message = mock_message

        # Mock usage with integer values
        mock_usage = mocker.Mock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 5
        mock_response_usage = mocker.Mock()
        mock_response_usage.usage = mock_usage

        mock_response = mocker.Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_client.chat.completions.create.return_value = mock_response

        # Patch AsyncOpenAI before creating LLM
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI",
            mock_async_openai,
        ).start()

        # Create LLM and make a request
        llm = OpenAILLM(**config)

        # Verify LLM has empty string API key
        assert llm.api_key == ""

        # Make a request
        messages = [{"role": "user", "content": "Hello"}]
        response = await llm.chat(messages)

        # Verify the request was made successfully
        assert isinstance(response, dict)
        assert response.get("content") == "Test response"
        mock_client.chat.completions.create.assert_called_once()

        # Verify AsyncOpenAI was called with empty string API key
        mock_async_openai.assert_called_once()
        call_args = mock_async_openai.call_args

        # Check that api_key parameter is empty string
        # Note: The actual behavior depends on OpenAI SDK implementation
        # With empty string API key, the SDK might omit Authorization header
        assert call_args.kwargs.get("api_key") == ""
        print(
            f"Empty string API key request test passed: AsyncOpenAI called with API key = '{call_args.kwargs.get('api_key')}'"
        )

    @pytest.mark.asyncio
    async def test_list_available_models_with_default_base_url(self, mocker):
        """Test listing available models using default base URL (official API)."""
        # Mock httpx response
        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "object": "list",
            "data": [
                {
                    "id": "gpt-4o",
                    "created": 1234567890,
                    "owned_by": "openai",
                },
                {
                    "id": "gpt-4o-mini",
                    "created": 1234567891,
                    "owned_by": "openai",
                },
            ],
        }

        mock_async_client = mocker.AsyncMock()
        mock_async_client.get.return_value = mock_response
        mock_async_client.__aenter__.return_value = mock_async_client
        mock_async_client.__aexit__.return_value = None

        mocker.patch("httpx.AsyncClient", return_value=mock_async_client)

        # Call without base_url - should use official API
        models = await OpenAILLM.list_available_models("test-api-key")

        # Verify results
        assert len(models) == 2
        # Models are sorted by created date (newest first)
        # gpt-4o-mini: created=1234567891, gpt-4o: created=1234567890
        assert models[0]["id"] == "gpt-4o-mini"
        assert models[1]["id"] == "gpt-4o"

        # Verify the API was called with official base URL
        mock_async_client.get.assert_called_once()
        call_args = mock_async_client.get.call_args
        assert "api.openai.com/v1/models" in call_args[0][0]
        assert call_args[1]["headers"]["Authorization"] == "Bearer test-api-key"

    @pytest.mark.asyncio
    async def test_list_available_models_with_custom_base_url(self, mocker):
        """Test listing available models using custom base URL."""
        # Mock httpx response
        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "object": "list",
            "data": [
                {
                    "id": "custom-model-1",
                    "created": 1234567890,
                    "owned_by": "custom",
                },
            ],
        }

        mock_async_client = mocker.AsyncMock()
        mock_async_client.get.return_value = mock_response
        mock_async_client.__aenter__.return_value = mock_async_client
        mock_async_client.__aexit__.return_value = None

        mocker.patch("httpx.AsyncClient", return_value=mock_async_client)

        # Call with custom base_url
        custom_base_url = "https://custom-proxy.com/v1"
        models = await OpenAILLM.list_available_models(
            "test-api-key", base_url=custom_base_url
        )

        # Verify results
        assert len(models) == 1
        assert models[0]["id"] == "custom-model-1"

        # Verify the API was called with custom base URL
        mock_async_client.get.assert_called_once()
        call_args = mock_async_client.get.call_args
        assert "custom-proxy.com/v1/models" in call_args[0][0]

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
        with pytest.raises(ValueError, match="Invalid API key"):
            await OpenAILLM.list_available_models("invalid-key")

    @pytest.mark.asyncio
    async def test_output_config_json_schema(self, llm, mocker):
        """Test output_config with json_schema format for OpenAI."""
        from openai.types.chat import ChatCompletion
        from openai.types.chat.chat_completion import Choice
        from openai.types.chat.chat_completion_message import ChatCompletionMessage

        # Create mock completion with JSON schema response
        mock_completion = ChatCompletion(
            id="test-json-schema-id",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(
                        content='{"joke": "Why did the chicken cross the road?", "punchline": "To get to the other side!"}',
                        role="assistant",
                        tool_calls=None,
                    ),
                )
            ],
            created=1234567890,
            model="gpt-4o",
            object="chat.completion",
            usage=None,
        )

        # Setup mock
        mock_client = mocker.AsyncMock()
        mock_client.chat.completions.create.return_value = mock_completion
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI",
            return_value=mock_client,
        )

        messages = [{"role": "user", "content": "Tell me a short joke."}]

        # Test with output_config using json_schema format
        # For OpenAI, this should be converted to response_format
        output_config = {
            "format": {
                "type": "json_schema",
                "json_schema": {
                    "type": "object",
                    "properties": {
                        "joke": {
                            "type": "string",
                            "description": "The text of the joke.",
                        },
                        "punchline": {
                            "type": "string",
                            "description": "The punchline of the joke.",
                        },
                    },
                    "required": ["joke", "punchline"],
                    "additionalProperties": False,
                },
            }
        }

        response = await llm.chat(messages, output_config=output_config)

        assert isinstance(response, dict)
        assert response.get("type") == "text"
        # Verify the response contains the expected JSON
        assert "joke" in response.get("content", "")
        assert "punchline" in response.get("content", "")

        # Verify the API was called with response_format (OpenAI format)
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert "response_format" in call_args.kwargs
        assert call_args.kwargs["response_format"]["type"] == "json_schema"
        assert "json_schema" in call_args.kwargs["response_format"]
