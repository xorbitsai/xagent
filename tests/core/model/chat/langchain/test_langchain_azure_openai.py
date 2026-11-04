"""Tests for Azure OpenAI LangChain adapter."""

import pytest

from xagent.core.model import ChatModelConfig
from xagent.core.model.chat.langchain import (
    ChatModelRetryWrapper,
    create_base_chat_model,
    create_base_chat_model_with_retry,
)


class TestAzureOpenAILangChainAdapter:
    """Test suite for Azure OpenAI LangChain adapter."""

    def test_create_azure_chat_model(self, mocker, monkeypatch):
        """Test that AzureChatOpenAI is instantiated correctly."""
        monkeypatch.setenv("OPENAI_API_VERSION", "2024-08-01-preview")

        # Mock AzureChatOpenAI to avoid langchain compatibility issues
        mock_azure = mocker.patch("xagent.core.model.chat.langchain.AzureChatOpenAI")

        config = ChatModelConfig(
            id="test_azure_model",
            model_provider="azure_openai",
            model_name="gpt-4o",
            base_url="https://test.openai.azure.com",
            api_key="test-api-key",
            default_temperature=0.7,
            default_max_tokens=1024,
            timeout=30.0,
        )

        create_base_chat_model(config, None)

        # Verify AzureChatOpenAI was called with correct parameters
        mock_azure.assert_called_once()
        call_kwargs = mock_azure.call_args[1]
        assert call_kwargs["deployment_name"] == "gpt-4o"
        assert call_kwargs["azure_endpoint"] == "https://test.openai.azure.com"
        assert call_kwargs["api_key"] == "test-api-key"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 1024
        assert call_kwargs["timeout"] == 30.0

    def test_azure_chat_model_with_temperature_override(self, mocker, monkeypatch):
        """Test that temperature override works for Azure OpenAI."""
        monkeypatch.setenv("OPENAI_API_VERSION", "2024-08-01-preview")

        mock_azure = mocker.patch("xagent.core.model.chat.langchain.AzureChatOpenAI")

        config = ChatModelConfig(
            id="test_azure_model",
            model_provider="azure_openai",
            model_name="gpt-4o",
            base_url="https://test.openai.azure.com",
            api_key="test-api-key",
            default_temperature=0.5,
            default_max_tokens=1024,
            timeout=30.0,
        )

        create_base_chat_model(config, 0.9)

        call_kwargs = mock_azure.call_args[1]
        assert call_kwargs["temperature"] == 0.9

    def test_azure_chat_model_uses_default_temperature(self, mocker, monkeypatch):
        """Test that default temperature from config is used when not overridden."""
        monkeypatch.setenv("OPENAI_API_VERSION", "2024-08-01-preview")

        mock_azure = mocker.patch("xagent.core.model.chat.langchain.AzureChatOpenAI")

        config = ChatModelConfig(
            id="test_azure_model",
            model_provider="azure_openai",
            model_name="gpt-4o",
            base_url="https://test.openai.azure.com",
            api_key="test-api-key",
            default_temperature=0.7,
            default_max_tokens=1024,
            timeout=30.0,
        )

        create_base_chat_model(config, None)

        call_kwargs = mock_azure.call_args[1]
        assert call_kwargs["temperature"] == 0.7

    def test_azure_api_version_from_env(self, mocker, monkeypatch):
        """Test that api_version is correctly sourced from environment variable."""
        monkeypatch.setenv("OPENAI_API_VERSION", "2025-04-01-preview")

        mock_azure = mocker.patch("xagent.core.model.chat.langchain.AzureChatOpenAI")

        config = ChatModelConfig(
            id="test_azure_model",
            model_provider="azure_openai",
            model_name="gpt-4o",
            base_url="https://test.openai.azure.com",
            api_key="test-api-key",
        )

        create_base_chat_model(config, None)

        call_kwargs = mock_azure.call_args[1]
        assert call_kwargs["api_version"] == "2025-04-01-preview"

    def test_azure_api_version_default(self, mocker, monkeypatch):
        """Test that default api_version is used when env var is not set."""
        # Ensure the env var is not set
        monkeypatch.delenv("OPENAI_API_VERSION", raising=False)

        mock_azure = mocker.patch("xagent.core.model.chat.langchain.AzureChatOpenAI")

        config = ChatModelConfig(
            id="test_azure_model",
            model_provider="azure_openai",
            model_name="gpt-4o",
            base_url="https://test.openai.azure.com",
            api_key="test-api-key",
        )

        create_base_chat_model(config, None)

        call_kwargs = mock_azure.call_args[1]
        assert call_kwargs["api_version"] == "2024-08-01-preview"

    def test_unsupported_provider_raises_error(self):
        """Test that unsupported model provider raises TypeError."""
        config = ChatModelConfig(
            id="test_model",
            model_provider="unsupported_provider",
            model_name="gpt-4o",
            api_key="test-api-key",
        )

        with pytest.raises(TypeError, match="Unsupported LLM model provider"):
            create_base_chat_model(config, None)

    def test_invalid_config_type_raises_error(self):
        """Test that non-ChatModelConfig raises TypeError."""
        from xagent.core.model import EmbeddingModelConfig

        config = EmbeddingModelConfig(
            id="test_model",
            model_provider="openai",
            model_name="text-embedding-ada-002",
            api_key="test-api-key",
        )

        with pytest.raises(TypeError, match="Unsupported Chat model type"):
            create_base_chat_model(config, None)

    def test_azure_openai_preserves_none_values(self, mocker, monkeypatch):
        """Test that None values for temperature and max_tokens are handled correctly."""
        monkeypatch.setenv("OPENAI_API_VERSION", "2024-08-01-preview")

        mock_azure = mocker.patch("xagent.core.model.chat.langchain.AzureChatOpenAI")

        config = ChatModelConfig(
            id="test_azure_model",
            model_provider="azure_openai",
            model_name="gpt-4o",
            base_url="https://test.openai.azure.com",
            api_key="test-api-key",
            # No default_temperature or default_max_tokens set
        )

        create_base_chat_model(config, None)

        call_kwargs = mock_azure.call_args[1]
        # When None is passed, AzureChatOpenAI will use its own defaults
        assert "temperature" in call_kwargs
        assert "max_tokens" in call_kwargs

    def test_create_deepseek_chat_model(self, mocker):
        """Test that DeepSeek uses ChatOpenAI with the DeepSeek default base URL."""
        mock_chat_openai = mocker.patch("xagent.core.model.chat.langchain.ChatOpenAI")

        config = ChatModelConfig(
            id="test_deepseek_model",
            model_provider="deepseek",
            model_name="deepseek-v4-flash",
            api_key="test-api-key",
            timeout=20.0,
        )

        create_base_chat_model(config, 0.3)

        call_kwargs = mock_chat_openai.call_args[1]
        assert call_kwargs["model"] == "deepseek-v4-flash"
        assert call_kwargs["api_key"] == "test-api-key"
        assert call_kwargs["base_url"] == "https://api.deepseek.com"
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["timeout"] == 20.0

    def test_create_deepseek_chat_model_uses_env_key_fallback(
        self, mocker, monkeypatch
    ):
        """DeepSeek LangChain adapter should share DeepSeek key fallback logic."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "env-deepseek-key")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-compatible-key")
        mock_chat_openai = mocker.patch("xagent.core.model.chat.langchain.ChatOpenAI")

        config = ChatModelConfig(
            id="test_deepseek_model",
            model_provider="deepseek",
            model_name="deepseek-v4-flash",
            api_key="",
        )

        create_base_chat_model(config, None)

        call_kwargs = mock_chat_openai.call_args[1]
        assert call_kwargs["api_key"] == "env-deepseek-key"

    def test_deepseek_retry_wrapper_disables_thinking_for_langchain_helpers(
        self, mocker
    ):
        """DeepSeek LangChain helper paths should disable thinking mode."""
        mock_chat_openai = mocker.patch("xagent.core.model.chat.langchain.ChatOpenAI")
        mock_model = mocker.Mock()
        mock_bound_model = mocker.Mock()
        mock_tool_runnable = mocker.Mock()
        mock_structured_runnable = mocker.Mock()
        mock_chat_openai.return_value = mock_model
        mock_model.bind.return_value = mock_bound_model
        mock_bound_model.bind_tools.return_value = mock_tool_runnable
        mock_bound_model.with_structured_output.return_value = mock_structured_runnable
        mocker.patch(
            "xagent.core.model.chat.langchain.create_retry_wrapper",
            side_effect=lambda model, *_args, **_kwargs: model,
        )

        config = ChatModelConfig(
            id="test_deepseek_model",
            model_provider="deepseek",
            model_name="deepseek-v4-flash",
            api_key="test-api-key",
        )

        wrapper = create_base_chat_model_with_retry(config, None)

        assert isinstance(wrapper, ChatModelRetryWrapper)
        wrapper.bind_tools([{"type": "function"}], tool_choice="auto")
        mock_model.bind.assert_called_with(
            extra_body={"thinking": {"type": "disabled"}}
        )
        mock_bound_model.bind_tools.assert_called_once_with(
            [{"type": "function"}], tool_choice="auto"
        )

        wrapper.with_structured_output({"type": "object"}, include_raw=True)
        mock_bound_model.with_structured_output.assert_called_once_with(
            {"type": "object"}, include_raw=True
        )
