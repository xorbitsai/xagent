"""Test cases for Claude LLM implementation using Anthropic SDK."""

import pytest

from xagent.core.model.chat.basic.claude import ClaudeLLM


class TestClaudeLLM:
    """Test cases for Claude LLM implementation."""

    @pytest.fixture
    def llm(self, claude_llm_config):
        """Fixture providing Claude LLM instance."""
        return ClaudeLLM(**claude_llm_config)

    @pytest.mark.asyncio
    async def test_basic_chat_completion(self, llm, mocker):
        """Test basic chat completion functionality."""
        # Setup mock
        mock_client = mocker.AsyncMock()

        # Create mock response
        mock_text_block = mocker.Mock()
        mock_text_block.type = "text"
        mock_text_block.text = "Hello World"

        # Mock usage with integer values
        mock_usage = mocker.Mock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 5

        mock_response = mocker.Mock()
        mock_response.stop_reason = "stop"
        mock_response.content = [mock_text_block]
        mock_response.usage = mock_usage
        mock_response.model_dump = mocker.Mock(return_value={"content": "Hello World"})

        mock_client.messages.create.return_value = mock_response
        mocker.patch(
            "xagent.core.model.chat.basic.claude.AsyncAnthropic",
            return_value=mock_client,
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

        # Verify the API was called with correct parameters
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args
        assert call_args.kwargs["model"] == "claude-3-5-sonnet-20241022"
        assert "temperature" in call_args.kwargs
        assert call_args.kwargs["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_tool_calling(self, llm, mocker):
        """Test tool calling functionality."""
        # Setup mock
        mock_client = mocker.AsyncMock()

        # Create mock response with tool use
        mock_tool_block = mocker.Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.id = "test_tool_id"
        mock_tool_block.name = "get_weather"
        mock_tool_block.input = {"location": "Boston"}

        # Mock usage with integer values
        mock_usage = mocker.Mock()
        mock_usage.input_tokens = 20
        mock_usage.output_tokens = 10

        mock_response = mocker.Mock()
        mock_response.stop_reason = "tool_use"
        mock_response.content = [mock_tool_block]
        mock_response.usage = mock_usage
        mock_response.model_dump = mocker.Mock(return_value={"tool_use": "get_weather"})

        mock_client.messages.create.return_value = mock_response
        mocker.patch(
            "xagent.core.model.chat.basic.claude.AsyncAnthropic",
            return_value=mock_client,
        )

        messages = [
            {"role": "user", "content": "What's the weather like in Boston?"},
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
        assert tool_calls[0]["id"] == "test_tool_id"
        print(f"Tool calling response: {response}")

        # Verify the API was called with tools
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args
        assert "tools" in call_args.kwargs

    @pytest.mark.asyncio
    async def test_system_message_handling(self, llm, mocker):
        """Test that system messages are properly handled."""
        # Setup mock
        mock_client = mocker.AsyncMock()

        mock_text_block = mocker.Mock()
        mock_text_block.type = "text"
        mock_text_block.text = "Response"

        # Mock usage with integer values
        mock_usage = mocker.Mock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 5

        mock_response = mocker.Mock()
        mock_response.stop_reason = "stop"
        mock_response.content = [mock_text_block]
        mock_response.usage = mock_usage

        mock_client.messages.create.return_value = mock_response
        mocker.patch(
            "xagent.core.model.chat.basic.claude.AsyncAnthropic",
            return_value=mock_client,
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]

        await llm.chat(messages)

        # Verify system message was passed separately
        call_args = mock_client.messages.create.call_args
        assert "system" in call_args.kwargs
        assert call_args.kwargs["system"] == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_context_manager(self, claude_llm_config, mocker):
        """Test async context manager functionality."""
        mock_client = mocker.AsyncMock()

        mock_text_block = mocker.Mock()
        mock_text_block.type = "text"
        mock_text_block.text = "test"

        # Mock usage with integer values
        mock_usage = mocker.Mock()
        mock_usage.input_tokens = 5
        mock_usage.output_tokens = 3

        mock_response = mocker.Mock()
        mock_response.stop_reason = "stop"
        mock_response.content = [mock_text_block]
        mock_response.usage = mock_usage

        mock_client.messages.create.return_value = mock_response
        mocker.patch(
            "xagent.core.model.chat.basic.claude.AsyncAnthropic",
            return_value=mock_client,
        )

        async with ClaudeLLM(**claude_llm_config) as ctx_llm:
            messages = [{"role": "user", "content": "Say 'test'"}]
            response = await ctx_llm.chat(messages)

            assert isinstance(response, str)
            assert response == "test"
            print(f"Context manager response: {response}")

        # Verify the client was properly closed
        assert ctx_llm._client is None

    @pytest.mark.asyncio
    async def test_error_handling_missing_sdk(self, claude_llm_config, mocker):
        """Test error handling when SDK is not installed."""
        # Mock AsyncAnthropic as None
        mocker.patch("xagent.core.model.chat.basic.claude.AsyncAnthropic", None)

        llm = ClaudeLLM(**claude_llm_config)
        messages = [{"role": "user", "content": "Hello"}]

        # Should raise a RuntimeError
        with pytest.raises(RuntimeError, match="anthropic SDK is not installed"):
            await llm.chat(messages)

    @pytest.mark.asyncio
    async def test_error_handling_missing_api_key(
        self, claude_llm_config, monkeypatch, mocker
    ):
        """Test error handling when API key is missing."""
        # Remove all API key environment variables
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("CLAUDE_API_KEY", raising=False)

        mocker.patch("xagent.core.model.chat.basic.claude.AsyncAnthropic")

        # Create LLM without API key
        config = claude_llm_config.copy()
        config["api_key"] = None

        llm = ClaudeLLM(**config)
        messages = [{"role": "user", "content": "Hello"}]

        # Should raise a RuntimeError
        with pytest.raises(
            RuntimeError, match="ANTHROPIC_API_KEY or CLAUDE_API_KEY must be set"
        ):
            await llm.chat(messages)

    @pytest.mark.asyncio
    async def test_custom_parameters(self, llm, mocker):
        """Test custom parameters like temperature and max_tokens."""
        # Setup mock
        mock_client = mocker.AsyncMock()

        mock_text_block = mocker.Mock()
        mock_text_block.type = "text"
        mock_text_block.text = "Test response"

        mock_response = mocker.Mock()
        mock_response.stop_reason = "stop"
        mock_response.content = [mock_text_block]
        # Mock usage with integer values
        mock_usage = mocker.Mock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 5

        mock_response.usage = mock_usage

        mock_client.messages.create.return_value = mock_response
        mocker.patch(
            "xagent.core.model.chat.basic.claude.AsyncAnthropic",
            return_value=mock_client,
        )

        messages = [{"role": "user", "content": "Count from 1 to 3."}]

        # Test with custom temperature and max_tokens
        response = await llm.chat(
            messages,
            temperature=0.1,  # Low temperature for more deterministic output
            max_tokens=50,  # Limit response length
        )

        assert isinstance(response, str)
        assert response == "Test response"
        print(f"Custom parameters response: {response}")

        # Verify custom parameters were passed
        call_args = mock_client.messages.create.call_args
        assert call_args.kwargs["temperature"] == 0.1
        assert call_args.kwargs["max_tokens"] == 50

    @pytest.mark.asyncio
    async def test_cleanup(self, claude_llm_config, mocker):
        """Test that client cleanup works properly."""
        mock_client = mocker.AsyncMock()

        mock_text_block = mocker.Mock()
        mock_text_block.type = "text"
        mock_text_block.text = "test"

        mock_response = mocker.Mock()
        mock_response.stop_reason = "stop"
        mock_response.content = [mock_text_block]
        # Mock usage with integer values
        mock_usage = mocker.Mock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 5

        mock_response.usage = mock_usage

        mock_client.messages.create.return_value = mock_response
        mocker.patch(
            "xagent.core.model.chat.basic.claude.AsyncAnthropic",
            return_value=mock_client,
        )

        llm = ClaudeLLM(**claude_llm_config)

        # Initialize client
        llm._ensure_client()

        # Verify client was created
        assert llm._client is not None

        # Close the client
        await llm.close()

        # Verify client is closed (this is mainly to ensure no exceptions are raised)
        assert llm._client is None

    @pytest.mark.asyncio
    async def test_abilities_property(self, claude_llm_config):
        """Test that abilities are correctly set based on model name."""
        # Test vision model (Claude 3 models support vision)
        vision_llm = ClaudeLLM(
            model_name="claude-3-5-sonnet-20241022", api_key="test-key"
        )
        assert "vision" in vision_llm.abilities

        # Test chat and tool_calling abilities
        assert "chat" in vision_llm.abilities
        assert "tool_calling" in vision_llm.abilities

    @pytest.mark.asyncio
    async def test_supports_thinking_mode(self, llm):
        """Test that Claude supports thinking mode."""
        assert llm.supports_thinking_mode is True

    @pytest.mark.asyncio
    async def test_thinking_mode_enabled(self, llm, mocker):
        """Test thinking mode configuration."""
        # Setup mock
        mock_client = mocker.AsyncMock()

        mock_text_block = mocker.Mock()
        mock_text_block.type = "text"
        mock_text_block.text = "Response with thinking"

        mock_response = mocker.Mock()
        mock_response.stop_reason = "stop"
        mock_response.content = [mock_text_block]
        # Mock usage with integer values
        mock_usage = mocker.Mock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 5

        mock_response.usage = mock_usage

        mock_client.messages.create.return_value = mock_response
        mocker.patch(
            "xagent.core.model.chat.basic.claude.AsyncAnthropic",
            return_value=mock_client,
        )

        messages = [{"role": "user", "content": "Think about this"}]

        # Test with thinking mode enabled
        response = await llm.chat(
            messages, thinking={"type": "enabled", "budget_tokens": 20480}
        )

        assert isinstance(response, str)

        # Verify thinking mode was passed
        call_args = mock_client.messages.create.call_args
        assert "thinking" in call_args.kwargs
        assert call_args.kwargs["thinking"]["type"] == "enabled"
        assert call_args.kwargs["thinking"]["budget_tokens"] == 20480

    @pytest.mark.asyncio
    async def test_thinking_mode_disabled(self, llm, mocker):
        """Test thinking mode disabled configuration."""
        # Setup mock
        mock_client = mocker.AsyncMock()

        mock_text_block = mocker.Mock()
        mock_text_block.type = "text"
        mock_text_block.text = "Response without thinking"

        mock_response = mocker.Mock()
        mock_response.stop_reason = "stop"
        mock_response.content = [mock_text_block]
        # Mock usage with integer values
        mock_usage = mocker.Mock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 5

        mock_response.usage = mock_usage

        mock_client.messages.create.return_value = mock_response
        mocker.patch(
            "xagent.core.model.chat.basic.claude.AsyncAnthropic",
            return_value=mock_client,
        )

        messages = [{"role": "user", "content": "Quick response"}]

        # Test with thinking mode explicitly disabled
        response = await llm.chat(messages, thinking={"type": "disabled"})

        assert isinstance(response, str)

        # Verify thinking mode was set to disabled
        call_args = mock_client.messages.create.call_args
        assert "thinking" in call_args.kwargs
        assert call_args.kwargs["thinking"]["type"] == "disabled"

    @pytest.mark.asyncio
    async def test_vision_chat(self, llm, mocker):
        """Test vision chat functionality."""
        # Setup mock
        mock_client = mocker.AsyncMock()

        mock_text_block = mocker.Mock()
        mock_text_block.type = "text"
        mock_text_block.text = "I see an image"

        mock_response = mocker.Mock()
        mock_response.stop_reason = "stop"
        mock_response.content = [mock_text_block]
        # Mock usage with integer values
        mock_usage = mocker.Mock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 5

        mock_response.usage = mock_usage

        mock_client.messages.create.return_value = mock_response
        mocker.patch(
            "xagent.core.model.chat.basic.claude.AsyncAnthropic",
            return_value=mock_client,
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

        response = await llm.vision_chat(messages)

        assert isinstance(response, str)
        assert response == "I see an image"
        print(f"Vision chat response: {response}")

    @pytest.mark.asyncio
    async def test_tool_choice_auto(self, llm, mocker):
        """Test tool_choice with 'auto' mode."""
        # Setup mock
        mock_client = mocker.AsyncMock()

        mock_text_block = mocker.Mock()
        mock_text_block.type = "text"
        mock_text_block.text = "Response"

        mock_response = mocker.Mock()
        mock_response.stop_reason = "stop"
        mock_response.content = [mock_text_block]
        # Mock usage with integer values
        mock_usage = mocker.Mock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 5

        mock_response.usage = mock_usage

        mock_client.messages.create.return_value = mock_response
        mocker.patch(
            "xagent.core.model.chat.basic.claude.AsyncAnthropic",
            return_value=mock_client,
        )

        messages = [{"role": "user", "content": "Hello"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "A test tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        await llm.chat(messages, tools=tools, tool_choice="auto")

        # Verify tool_choice was set to auto
        call_args = mock_client.messages.create.call_args
        assert "tool_choice" in call_args.kwargs
        assert call_args.kwargs["tool_choice"]["type"] == "auto"

    @pytest.mark.asyncio
    async def test_tool_choice_any(self, llm, mocker):
        """Test tool_choice with 'any' mode."""
        # Setup mock
        mock_client = mocker.AsyncMock()

        mock_text_block = mocker.Mock()
        mock_text_block.type = "text"
        mock_text_block.text = "Response"

        mock_response = mocker.Mock()
        mock_response.stop_reason = "stop"
        mock_response.content = [mock_text_block]
        # Mock usage with integer values
        mock_usage = mocker.Mock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 5

        mock_response.usage = mock_usage

        mock_client.messages.create.return_value = mock_response
        mocker.patch(
            "xagent.core.model.chat.basic.claude.AsyncAnthropic",
            return_value=mock_client,
        )

        messages = [{"role": "user", "content": "Hello"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "A test tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        await llm.chat(messages, tools=tools, tool_choice="any")

        # Verify tool_choice was set to any
        call_args = mock_client.messages.create.call_args
        assert "tool_choice" in call_args.kwargs
        assert call_args.kwargs["tool_choice"]["type"] == "any"

    @pytest.mark.asyncio
    async def test_default_max_tokens(self, claude_llm_config, mocker):
        """Test that default max_tokens is set when not provided."""
        # Setup mock
        mock_client = mocker.AsyncMock()

        mock_text_block = mocker.Mock()
        mock_text_block.type = "text"
        mock_text_block.text = "Response"

        mock_response = mocker.Mock()
        mock_response.stop_reason = "stop"
        mock_response.content = [mock_text_block]
        # Mock usage with integer values
        mock_usage = mocker.Mock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 5

        mock_response.usage = mock_usage

        mock_client.messages.create.return_value = mock_response
        mocker.patch(
            "xagent.core.model.chat.basic.claude.AsyncAnthropic",
            return_value=mock_client,
        )

        # Create LLM without default_max_tokens
        config = claude_llm_config.copy()
        config["default_max_tokens"] = None

        llm = ClaudeLLM(**config)

        messages = [{"role": "user", "content": "Hello"}]

        await llm.chat(messages)

        # Verify max_tokens was set to default (4096)
        call_args = mock_client.messages.create.call_args
        assert "max_tokens" in call_args.kwargs
        assert call_args.kwargs["max_tokens"] == 4096

    @pytest.mark.asyncio
    async def test_retryable_errors(self, llm, mocker):
        """Test that Anthropic errors are converted to LLMRetryableError."""
        from anthropic import APIStatusError, APITimeoutError, RateLimitError

        from xagent.core.model.chat.exceptions import LLMRetryableError

        # Mock the client
        mock_client = mocker.AsyncMock()
        mocker.patch(
            "xagent.core.model.chat.basic.claude.AsyncAnthropic",
            return_value=mock_client,
        )

        # Initialize client
        llm._ensure_client()
        llm._client = mock_client

        # Case 1: Timeout
        mock_client.messages.create.side_effect = APITimeoutError(request=mocker.Mock())
        with pytest.raises(LLMRetryableError):
            await llm.chat([{"role": "user", "content": "hi"}])

        # Case 2: Rate Limit
        mock_client.messages.create.side_effect = RateLimitError(
            message="Rate limit", response=mocker.Mock(), body={}
        )
        with pytest.raises(LLMRetryableError):
            await llm.chat([{"role": "user", "content": "hi"}])

        # Case 3: 500 Server Error
        mock_response = mocker.Mock()
        mock_response.status_code = 500
        # Mock usage with integer values
        mock_usage = mocker.Mock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 5

        mock_response.usage = mock_usage
        mock_client.messages.create.side_effect = APIStatusError(
            message="Server Error", response=mock_response, body={}
        )
        with pytest.raises(LLMRetryableError):
            await llm.chat([{"role": "user", "content": "hi"}])

        # Case 4: 400 Bad Request
        mock_response = mocker.Mock()
        mock_response.status_code = 400
        # Mock usage with integer values
        mock_usage = mocker.Mock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 5

        mock_response.usage = mock_usage
        mock_client.messages.create.side_effect = APIStatusError(
            message="Bad Request", response=mock_response, body={}
        )

        with pytest.raises(LLMRetryableError):
            await llm.chat([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_list_available_models_with_default_base_url(self, mocker):
        """Test listing available models using default base URL (official API)."""
        # Mock httpx response
        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "claude-3-5-sonnet-20241022",
                    "display_name": "Claude 3.5 Sonnet",
                    "created": 1234567890,
                    "type": "model",
                },
                {
                    "id": "claude-3-5-haiku-20241022",
                    "display_name": "Claude 3.5 Haiku",
                    "created": 1234567891,
                    "type": "model",
                },
            ],
        }

        mock_async_client = mocker.AsyncMock()
        mock_async_client.get.return_value = mock_response
        mock_async_client.__aenter__.return_value = mock_async_client
        mock_async_client.__aexit__.return_value = None

        mocker.patch("httpx.AsyncClient", return_value=mock_async_client)

        # Call without base_url - should use official API
        models = await ClaudeLLM.list_available_models("test-api-key")

        # Verify results
        assert len(models) == 2
        # Models are sorted by created date (newest first)
        # claude-3-5-haiku: created=1234567891, claude-3-5-sonnet: created=1234567890
        assert models[0]["id"] == "claude-3-5-haiku-20241022"
        assert models[1]["id"] == "claude-3-5-sonnet-20241022"

        # Verify the API was called with official base URL
        mock_async_client.get.assert_called_once()
        call_args = mock_async_client.get.call_args
        assert "api.anthropic.com/v1/models" in call_args[0][0]
        assert call_args[1]["headers"]["x-api-key"] == "test-api-key"

    @pytest.mark.asyncio
    async def test_list_available_models_with_custom_base_url(self, mocker):
        """Test listing available models using custom base URL."""
        # Mock httpx response
        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "custom-claude-model",
                    "display_name": "Custom Claude",
                    "created": 1234567890,
                    "type": "model",
                },
            ],
        }

        mock_async_client = mocker.AsyncMock()
        mock_async_client.get.return_value = mock_response
        mock_async_client.__aenter__.return_value = mock_async_client
        mock_async_client.__aexit__.return_value = None

        mocker.patch("httpx.AsyncClient", return_value=mock_async_client)

        # Call with custom base_url
        custom_base_url = "https://custom-proxy.com"
        models = await ClaudeLLM.list_available_models(
            "test-api-key", base_url=custom_base_url
        )

        # Verify results
        assert len(models) == 1
        assert models[0]["id"] == "custom-claude-model"

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
        with pytest.raises(ValueError, match="Invalid Anthropic API key"):
            await ClaudeLLM.list_available_models("invalid-key")

    @pytest.mark.asyncio
    async def test_output_config_json_schema(self, llm, mocker):
        """Test output_config with json_schema format."""
        # Setup mock
        mock_client = mocker.AsyncMock()

        mock_text_block = mocker.Mock()
        mock_text_block.type = "text"
        mock_text_block.text = '{"joke": "Why did the chicken cross the road?", "punchline": "To get to the other side!"}'

        mock_response = mocker.Mock()
        mock_response.stop_reason = "stop"
        mock_response.content = [mock_text_block]
        # Mock usage with integer values
        mock_usage = mocker.Mock()
        mock_usage.input_tokens = 20
        mock_usage.output_tokens = 15

        mock_response.usage = mock_usage

        mock_client.messages.create.return_value = mock_response
        mocker.patch(
            "xagent.core.model.chat.basic.claude.AsyncAnthropic",
            return_value=mock_client,
        )

        messages = [{"role": "user", "content": "Tell me a short joke."}]

        # Test with output_config using json_schema format
        output_config = {
            "format": {
                "type": "json_schema",
                "schema": {
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

        assert isinstance(response, str)
        # Verify the response contains the expected JSON
        assert "joke" in response
        assert "punchline" in response

        # Verify the API was called with output_config
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args
        assert "output_config" in call_args.kwargs
        assert call_args.kwargs["output_config"]["format"]["type"] == "json_schema"

    @pytest.mark.asyncio
    async def test_strict_tool_mode(self, llm, mocker):
        """Test strict mode for tool calling."""
        # Setup mock
        mock_client = mocker.AsyncMock()

        mock_tool_block = mocker.Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.id = "test_tool_id"
        mock_tool_block.name = "search_flights"
        mock_tool_block.input = {"destination": "Paris", "date": "2025-03-15"}

        mock_usage = mocker.Mock()
        mock_usage.input_tokens = 25
        mock_usage.output_tokens = 10

        mock_response = mocker.Mock()
        mock_response.stop_reason = "tool_use"
        mock_response.content = [mock_tool_block]
        mock_response.usage = mock_usage

        mock_client.messages.create.return_value = mock_response
        mocker.patch(
            "xagent.core.model.chat.basic.claude.AsyncAnthropic",
            return_value=mock_client,
        )

        messages = [
            {"role": "user", "content": "Search for flights to Paris next month"}
        ]

        # Test with strict tool mode
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_flights",
                    "description": "Search for flights",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "destination": {"type": "string"},
                            "date": {"type": "string", "format": "date"},
                        },
                        "required": ["destination", "date"],
                        "additionalProperties": False,
                    },
                },
                "strict": True,  # Enable strict mode
            }
        ]

        response = await llm.chat(messages, tools=tools)

        # Verify tool call response
        assert isinstance(response, dict)
        assert response.get("type") == "tool_call"

        # Verify the API was called with strict mode enabled
        call_args = mock_client.messages.create.call_args
        assert "tools" in call_args.kwargs
        anthropic_tools = call_args.kwargs["tools"]
        assert len(anthropic_tools) == 1
        assert anthropic_tools[0]["strict"] is True
        assert anthropic_tools[0]["name"] == "search_flights"
