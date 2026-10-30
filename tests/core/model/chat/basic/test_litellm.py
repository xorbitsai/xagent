"""Test cases for LiteLLM chat model implementation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xagent.core.model.chat.basic.litellm import LiteLLM
from xagent.core.model.chat.exceptions import LLMRetryableError, LLMTimeoutError


def _mock_response(content="Hello", prompt_tokens=10, completion_tokens=5):
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    choice = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = None
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def _mock_tool_response(name="get_weather", arguments='{"city": "Paris"}'):
    tc = MagicMock()
    tc.id = "call_123"
    tc.function.name = name
    tc.function.arguments = arguments
    choice = MagicMock()
    choice.message.content = None
    choice.message.tool_calls = [tc]
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock(prompt_tokens=20, completion_tokens=10)
    return resp


class TestLiteLLMInit:
    def test_default_model(self):
        llm = LiteLLM()
        assert llm.model_name == "openai/gpt-4o-mini"

    def test_custom_model(self):
        llm = LiteLLM(model_name="anthropic/claude-sonnet-4-6")
        assert llm.model_name == "anthropic/claude-sonnet-4-6"

    def test_abilities_default(self):
        llm = LiteLLM()
        assert "chat" in llm.abilities
        assert "tool_calling" in llm.abilities

    def test_abilities_custom(self):
        llm = LiteLLM(abilities=["chat", "vision"])
        assert llm.abilities == ["chat", "vision"]

    def test_api_key_stored(self):
        llm = LiteLLM(api_key="sk-test")
        assert llm._api_key == "sk-test"

    def test_api_base_stored(self):
        llm = LiteLLM(api_base="http://localhost:4000")
        assert llm._api_base == "http://localhost:4000"

    def test_supports_thinking_mode_false(self):
        llm = LiteLLM()
        assert llm.supports_thinking_mode is False


class TestLiteLLMChat:
    @pytest.mark.asyncio
    async def test_basic_chat(self):
        llm = LiteLLM(model_name="openai/gpt-4o")
        resp = _mock_response("The answer is 4.")
        with patch(
            "litellm.acompletion", new_callable=AsyncMock, return_value=resp
        ) as mock:
            result = await llm.chat([{"role": "user", "content": "What is 2+2?"}])
            assert result == "The answer is 4."
            call_kwargs = mock.call_args.kwargs
            assert call_kwargs["model"] == "openai/gpt-4o"
            assert call_kwargs["drop_params"] is True

    @pytest.mark.asyncio
    async def test_api_key_forwarded(self):
        llm = LiteLLM(api_key="sk-test")
        with patch(
            "litellm.acompletion", new_callable=AsyncMock, return_value=_mock_response()
        ):
            await llm.chat([{"role": "user", "content": "test"}])
            from litellm import acompletion

            call_kwargs = acompletion.call_args.kwargs
            assert call_kwargs["api_key"] == "sk-test"

    @pytest.mark.asyncio
    async def test_api_key_omitted_when_none(self):
        llm = LiteLLM()
        with patch(
            "litellm.acompletion", new_callable=AsyncMock, return_value=_mock_response()
        ) as mock:
            await llm.chat([{"role": "user", "content": "test"}])
            assert "api_key" not in mock.call_args.kwargs

    @pytest.mark.asyncio
    async def test_api_base_forwarded(self):
        llm = LiteLLM(api_base="http://proxy:4000")
        with patch(
            "litellm.acompletion", new_callable=AsyncMock, return_value=_mock_response()
        ) as mock:
            await llm.chat([{"role": "user", "content": "test"}])
            assert mock.call_args.kwargs["api_base"] == "http://proxy:4000"

    @pytest.mark.asyncio
    async def test_temperature_forwarded(self):
        llm = LiteLLM()
        with patch(
            "litellm.acompletion", new_callable=AsyncMock, return_value=_mock_response()
        ) as mock:
            await llm.chat([{"role": "user", "content": "test"}], temperature=0.5)
            assert mock.call_args.kwargs["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_default_temperature_used(self):
        llm = LiteLLM(default_temperature=0.3)
        with patch(
            "litellm.acompletion", new_callable=AsyncMock, return_value=_mock_response()
        ) as mock:
            await llm.chat([{"role": "user", "content": "test"}])
            assert mock.call_args.kwargs["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_null_content_returns_empty(self):
        resp = _mock_response(content=None)
        llm = LiteLLM()
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp):
            result = await llm.chat([{"role": "user", "content": "test"}])
            assert result == ""


class TestLiteLLMToolCalling:
    @pytest.mark.asyncio
    async def test_tool_call_returned(self):
        llm = LiteLLM()
        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            return_value=_mock_tool_response(),
        ):
            result = await llm.chat(
                [{"role": "user", "content": "Weather?"}],
                tools=[{"type": "function", "function": {"name": "get_weather"}}],
            )
            assert result["type"] == "tool_call"
            assert result["tool_calls"][0]["function"]["name"] == "get_weather"


class TestLiteLLMErrors:
    @pytest.mark.asyncio
    async def test_timeout_raises_llm_timeout_error(self):
        import litellm as _litellm

        llm = LiteLLM()
        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=_litellm.Timeout(
                message="Request timed out", model="gpt-4o", llm_provider="openai"
            ),
        ):
            with pytest.raises(LLMTimeoutError):
                await llm.chat([{"role": "user", "content": "test"}])

    @pytest.mark.asyncio
    async def test_rate_limit_raises_retryable_error(self):
        import litellm as _litellm

        llm = LiteLLM()
        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=_litellm.RateLimitError(
                message="429", llm_provider="openai", model="gpt-4o"
            ),
        ):
            with pytest.raises(LLMRetryableError):
                await llm.chat([{"role": "user", "content": "test"}])

    @pytest.mark.asyncio
    async def test_connection_error_raises_retryable_error(self):
        import litellm as _litellm

        llm = LiteLLM()
        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=_litellm.APIConnectionError(
                message="Connection failed", llm_provider="openai", model="gpt-4o"
            ),
        ):
            with pytest.raises(LLMRetryableError):
                await llm.chat([{"role": "user", "content": "test"}])

    @pytest.mark.asyncio
    async def test_auth_error_propagates(self):
        import litellm as _litellm

        llm = LiteLLM()
        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=_litellm.AuthenticationError(
                message="Invalid key", llm_provider="openai", model="gpt-4o"
            ),
        ):
            with pytest.raises(_litellm.AuthenticationError):
                await llm.chat([{"role": "user", "content": "test"}])


class TestLiteLLMFactory:
    def test_adapter_creates_litellm(self):
        from xagent.core.model import ChatModelConfig
        from xagent.core.model.chat.basic.adapter import create_base_llm

        config = ChatModelConfig(
            id="test-litellm",
            model_name="anthropic/claude-sonnet-4-6",
            model_provider="litellm",
        )
        llm = create_base_llm(config)
        assert llm is not None
