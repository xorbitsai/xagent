"""Test cases for Zhipu LLM implementation."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from xagent.core.model.chat.basic.zhipu import ZhipuLLM
from xagent.core.model.chat.types import ChunkType


class TestZhipuLLM:
    """Test cases for ZhipuLLM class."""

    @pytest.fixture
    def mock_zhipu_client(self):
        """Create a mock Zhipu client."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock()
        return mock_client

    @pytest.fixture
    def zhipu_llm(self, mock_zhipu_client):
        """Create a ZhipuLLM instance with mocked client."""
        with patch(
            "xagent.core.model.chat.basic.zhipu.ZhipuAiClient",
            return_value=mock_zhipu_client,
        ):
            llm = ZhipuLLM(api_key="test_key")
            llm._client = mock_zhipu_client
            return llm

    @pytest.mark.asyncio
    async def test_normal_text_response(self, zhipu_llm, mock_zhipu_client):
        """Test normal text response handling."""
        # Mock response with text content
        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_message = MagicMock()
        mock_message.content = "Hello, world!"
        mock_message.tool_calls = None
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_zhipu_client.chat.completions.create.return_value = mock_response

        result = await zhipu_llm.chat([{"role": "user", "content": "Hello"}])

        assert result == "Hello, world!"

    @pytest.mark.asyncio
    async def test_none_content_response(self, zhipu_llm, mock_zhipu_client):
        """Test handling of None content response."""
        # Mock response with None content
        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = None
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_zhipu_client.chat.completions.create.return_value = mock_response

        # Should raise RuntimeError when content is None and no tool calls
        with pytest.raises(
            RuntimeError, match="LLM returned None content and no tool calls"
        ):
            await zhipu_llm.chat([{"role": "user", "content": "Hello"}])

    @pytest.mark.asyncio
    async def test_empty_content_response(self, zhipu_llm, mock_zhipu_client):
        """Test handling of empty string content response."""
        # Mock response with empty string content
        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_message = MagicMock()
        mock_message.content = ""
        mock_message.tool_calls = None
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_zhipu_client.chat.completions.create.return_value = mock_response

        # Should raise RuntimeError when content is empty and no tool calls
        with pytest.raises(
            RuntimeError, match="LLM returned empty content and no tool calls"
        ):
            await zhipu_llm.chat([{"role": "user", "content": "Hello"}])

    @pytest.mark.asyncio
    async def test_tool_call_response(self, zhipu_llm, mock_zhipu_client):
        """Test tool call response handling."""
        # Mock tool call
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "calculator"
        mock_tool_call.function.arguments = '{"expression": "2+2"}'

        mock_choice = MagicMock()
        mock_choice.finish_reason = "tool_calls"
        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_zhipu_client.chat.completions.create.return_value = mock_response

        result = await zhipu_llm.chat(
            [{"role": "user", "content": "Calculate 2+2"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "description": "Calculate mathematical expressions",
                        "parameters": {
                            "type": "object",
                            "properties": {"expression": {"type": "string"}},
                            "required": ["expression"],
                        },
                    },
                }
            ],
        )

        assert result["type"] == "tool_call"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "calculator"

    @pytest.mark.asyncio
    async def test_stream_chat_yields_tool_call_argument_deltas(
        self, zhipu_llm, mock_zhipu_client
    ):
        """Zhipu stream_chat should consume the producer queue while it streams."""
        mock_zhipu_client.chat.completions.create.return_value = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    id="call_1",
                                    index=0,
                                    function=SimpleNamespace(
                                        name="final_answer",
                                        arguments='{"answer":"Hel',
                                    ),
                                )
                            ],
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    id=None,
                                    index=0,
                                    function=SimpleNamespace(
                                        name=None,
                                        arguments='lo"}',
                                    ),
                                )
                            ],
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=None, tool_calls=None),
                        finish_reason="tool_calls",
                    )
                ],
                usage=None,
            ),
        ]

        chunks = [
            chunk
            async for chunk in zhipu_llm.stream_chat(
                [{"role": "user", "content": "answer"}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "final_answer",
                            "description": "Return final answer",
                            "parameters": {
                                "type": "object",
                                "properties": {"answer": {"type": "string"}},
                                "required": ["answer"],
                            },
                        },
                    }
                ],
            )
        ]

        tool_chunks = [chunk for chunk in chunks if chunk.type == ChunkType.TOOL_CALL]
        assert [
            chunk.tool_calls[0]["function"]["arguments"] for chunk in tool_chunks
        ] == ['{"answer":"Hel', '{"answer":"Hello"}', '{"answer":"Hello"}']
        assert tool_chunks[-1].finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_none_api_response(self, zhipu_llm, mock_zhipu_client):
        """Test handling of None API response."""
        mock_zhipu_client.chat.completions.create.return_value = None

        with pytest.raises(RuntimeError) as exc_info:
            await zhipu_llm.chat([{"role": "user", "content": "Hello"}])

        assert "Zhipu API returned None response" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_response_missing_choices(self, zhipu_llm, mock_zhipu_client):
        """Test handling of response missing choices."""
        mock_response = MagicMock()
        mock_response.choices = None

        mock_zhipu_client.chat.completions.create.return_value = mock_response

        with pytest.raises(RuntimeError) as exc_info:
            await zhipu_llm.chat([{"role": "user", "content": "Hello"}])

        assert "Zhipu API response missing choices" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_thinking_mode_disabled(self, zhipu_llm, mock_zhipu_client):
        """Test thinking mode configuration."""
        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_message = MagicMock()
        mock_message.content = "Response with thinking disabled"
        mock_message.tool_calls = None
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_zhipu_client.chat.completions.create.return_value = mock_response

        result = await zhipu_llm.chat(
            [{"role": "user", "content": "Hello"}], thinking={"type": "disabled"}
        )

        # Verify thinking mode was passed to API
        call_args = mock_zhipu_client.chat.completions.create.call_args
        assert "thinking" in call_args.kwargs
        assert call_args.kwargs["thinking"]["type"] == "disabled"

        assert result == "Response with thinking disabled"

    @pytest.mark.asyncio
    async def test_empty_string_api_key_fallback(self, monkeypatch):
        """Test that empty string API key falls back to environment variable (Zhipu uses 'or' operator)."""
        # Set environment variable
        env_api_key = "env-api-key-for-zhipu"
        monkeypatch.setenv("ZHIPU_API_KEY", env_api_key)

        # Create Zhipu LLM with empty string API key
        # Note: Zhipu uses 'or' operator, so empty string is falsy and should fallback
        with patch("xagent.core.model.chat.basic.zhipu.ZhipuAiClient"):
            llm = ZhipuLLM(api_key="")  # Empty string

        # Verify that the API key is from environment variable (not empty string)
        assert llm.api_key == env_api_key
        print(
            f"Zhipu empty string API key test: API key = '{llm.api_key}' (fell back to env var)"
        )

    @pytest.mark.asyncio
    async def test_none_api_key_with_env_fallback(self, monkeypatch):
        """Test None API key with environment variable fallback for Zhipu."""
        # Set environment variable
        env_api_key = "env-api-key-for-zhipu"
        monkeypatch.setenv("ZHIPU_API_KEY", env_api_key)

        # Create Zhipu LLM with None API key
        with patch("xagent.core.model.chat.basic.zhipu.ZhipuAiClient"):
            llm = ZhipuLLM(api_key=None)

        # Verify that the API key is from environment variable
        assert llm.api_key == env_api_key
        print(f"Zhipu None API key test: API key = '{llm.api_key}'")

    @pytest.mark.asyncio
    async def test_none_api_key_with_openai_env_fallback(self, monkeypatch):
        """Test None API key falls back to OPENAI_API_KEY when ZHIPU_API_KEY is not set."""
        # Set OPENAI_API_KEY environment variable (but not ZHIPU_API_KEY)
        openai_env_api_key = "openai-env-api-key"
        monkeypatch.setenv("OPENAI_API_KEY", openai_env_api_key)
        monkeypatch.delenv("ZHIPU_API_KEY", raising=False)

        # Create Zhipu LLM with None API key
        with patch("xagent.core.model.chat.basic.zhipu.ZhipuAiClient"):
            llm = ZhipuLLM(api_key=None)

        # Verify that the API key is from OPENAI_API_KEY environment variable
        assert llm.api_key == openai_env_api_key
        print(
            f"Zhipu None API key with OPENAI_API_KEY fallback test: API key = '{llm.api_key}'"
        )

    @pytest.mark.asyncio
    async def test_missing_api_key_initialization(self, monkeypatch):
        """Test Zhipu initialization when API key is completely missing."""
        # Remove all API key environment variables
        monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        # Create Zhipu LLM with None API key and no environment variable
        with patch("xagent.core.model.chat.basic.zhipu.ZhipuAiClient"):
            llm = ZhipuLLM(api_key=None)

        # Zhipu uses 'or' operator, so when api_key is None and no env vars, it should be None
        assert llm.api_key is None
        print(
            f"Zhipu missing API key test: LLM initialized with API key = {llm.api_key}"
        )

    @pytest.mark.asyncio
    async def test_explicit_api_key_not_overridden(self, monkeypatch):
        """Test that explicit API key is not overridden by environment variable."""
        # Set environment variable
        env_api_key = "env-api-key-should-not-be-used"
        monkeypatch.setenv("ZHIPU_API_KEY", env_api_key)

        # Create Zhipu LLM with explicit API key
        explicit_api_key = "explicit-api-key"
        with patch("xagent.core.model.chat.basic.zhipu.ZhipuAiClient"):
            llm = ZhipuLLM(api_key=explicit_api_key)

        # Verify that the explicit API key is used, not the environment variable
        assert llm.api_key == explicit_api_key
        assert llm.api_key != env_api_key
        print(
            f"Zhipu explicit API key test: API key = '{llm.api_key}' (not using env var)"
        )

    @pytest.mark.asyncio
    async def test_list_available_models_with_default_base_url(self, mocker):
        """Test listing available models using default base URL (official API)."""
        # The zai SDK method may not exist or work as expected, so we test
        # the HTTP fallback path by making the SDK method raise AttributeError
        mock_client = MagicMock()
        mock_client.models.list.side_effect = AttributeError("no such method")
        mocker.patch("zai.ZhipuAiClient", return_value=mock_client)

        # Mock httpx response for HTTP fallback
        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "object": "list",
            "data": [
                {
                    "id": "glm-4",
                    "created": 1234567890,
                    "owned_by": "zhipu",
                },
                {
                    "id": "glm-4-flash",
                    "created": 1234567891,
                    "owned_by": "zhipu",
                },
            ],
        }

        mock_async_client = mocker.AsyncMock()
        mock_async_client.get.return_value = mock_response
        mock_async_client.__aenter__.return_value = mock_async_client
        mock_async_client.__aexit__.return_value = None

        mocker.patch("httpx.AsyncClient", return_value=mock_async_client)

        # Call without base_url - should use official API
        models = await ZhipuLLM.list_available_models("test-api-key")

        # Verify results
        assert len(models) == 2
        # Models are sorted by created date (newest first)
        assert models[0]["id"] == "glm-4-flash"
        assert models[1]["id"] == "glm-4"

        # Verify the HTTP API was called (fallback from SDK)
        mock_async_client.get.assert_called_once()
        call_args = mock_async_client.get.call_args
        assert "open.bigmodel.cn/v1/models" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_list_available_models_with_custom_base_url(self, mocker):
        """Test listing available models using custom base URL."""
        # Mock httpx response for fallback
        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "object": "list",
            "data": [
                {
                    "id": "custom-glm-model",
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

        # Call with custom base_url - should fallback to HTTP
        custom_base_url = "https://custom-proxy.com/v1"
        models = await ZhipuLLM.list_available_models(
            "test-api-key", base_url=custom_base_url
        )

        # Verify results
        assert len(models) == 1
        assert models[0]["id"] == "custom-glm-model"

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

        # Mock zai to raise AttributeError (no models.list method)
        mock_client = MagicMock()
        mock_client.models.list.side_effect = AttributeError("no such method")
        mocker.patch("zai.ZhipuAiClient", return_value=mock_client)

        # Patch httpx module (top-level import, not in zhipu module)
        mocker.patch("httpx.AsyncClient", return_value=mock_async_client)

        # Should raise ValueError for invalid API key
        with pytest.raises(ValueError, match="Invalid API key"):
            await ZhipuLLM.list_available_models("invalid-key")


class TestZhipuPromptCacheUsage:
    """prompt_tokens_details.cached_tokens is recorded as cached input tokens."""

    @pytest.fixture
    def mock_zhipu_client(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock()
        return mock_client

    @pytest.fixture
    def zhipu_llm(self, mock_zhipu_client):
        with patch(
            "xagent.core.model.chat.basic.zhipu.ZhipuAiClient",
            return_value=mock_zhipu_client,
        ):
            llm = ZhipuLLM(api_key="test_key")
            llm._client = mock_zhipu_client
            return llm

    @pytest.mark.asyncio
    async def test_chat_records_cached_tokens(self, zhipu_llm, mock_zhipu_client):
        from xagent.core.model.chat.token_context import TokenContextManager

        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_message = MagicMock()
        mock_message.content = "ok"
        mock_message.tool_calls = None
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=5,
            prompt_tokens_details=SimpleNamespace(cached_tokens=60),
        )

        mock_zhipu_client.chat.completions.create.return_value = mock_response

        with TokenContextManager() as manager:
            await zhipu_llm.chat([{"role": "user", "content": "hi"}])
            usage = manager.get_usage()

        assert usage.input_tokens == 100
        inp = next(d for d in usage.details if d["type"] == "input")
        assert inp["cached_tokens"] == 60

    @pytest.mark.asyncio
    async def test_stream_chat_usage_chunk_carries_cached_tokens(
        self, zhipu_llm, mock_zhipu_client
    ):
        from xagent.core.model.chat.token_context import TokenContextManager

        mock_zhipu_client.chat.completions.create.return_value = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="ok", tool_calls=None),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=100,
                    completion_tokens=5,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=60),
                ),
            ),
        ]

        with TokenContextManager() as manager:
            chunks = [
                chunk
                async for chunk in zhipu_llm.stream_chat(
                    [{"role": "user", "content": "hi"}]
                )
            ]
            usage = manager.get_usage()

        inp = next(d for d in usage.details if d["type"] == "input")
        assert inp["cached_tokens"] == 60

        usage_payloads = [
            c.usage for c in chunks if c.type == ChunkType.USAGE and c.usage
        ]
        assert usage_payloads and usage_payloads[0]["cached_input_tokens"] == 60
