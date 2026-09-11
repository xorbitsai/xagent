"""Test cases for create_default_llm function with strict separation."""

from unittest.mock import patch

import pytest

from xagent.web.services.agent_service_manager import create_default_llm


@pytest.fixture(autouse=True)
def clear_deepseek_env(monkeypatch):
    """Keep DeepSeek env vars from example.env from affecting non-DeepSeek tests."""
    for key in ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL_NAME"):
        monkeypatch.delenv(key, raising=False)


class TestCreateDefaultLLM:
    """Test cases for create_default_llm function with strict separation."""

    def test_openai_with_empty_string_api_key(self, monkeypatch):
        """Test OpenAI LLM creation with empty string API key."""
        # Set environment variables
        monkeypatch.setenv("OPENAI_API_KEY", "")  # Empty string
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
        monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
        monkeypatch.delenv("ZHIPU_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("ZHIPU_MODEL_NAME", raising=False)

        # Mock OpenAILLM constructor to capture arguments
        with patch(
            "xagent.web.services.agent_service_manager.OpenAILLM"
        ) as mock_openai_llm:
            mock_openai_llm.return_value = None  # Return None for simplicity

            result = create_default_llm()

            # Verify OpenAILLM was called with empty string API key
            mock_openai_llm.assert_called_once()
            call_args = mock_openai_llm.call_args

            # Check API key is empty string
            assert call_args.kwargs["api_key"] == ""
            assert call_args.kwargs["model_name"] == "gpt-4o-mini"
            assert call_args.kwargs["base_url"] is None

            # Result should be None because we mocked the return value
            assert result is None

    def test_openai_with_none_api_key_returns_none(self, monkeypatch):
        """Test OpenAI LLM creation with None API key returns None."""
        # Set environment variables
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # None
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
        monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
        monkeypatch.delenv("ZHIPU_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("ZHIPU_MODEL_NAME", raising=False)

        # Mock OpenAILLM constructor
        with patch(
            "xagent.web.services.agent_service_manager.OpenAILLM"
        ) as mock_openai_llm:
            # OpenAILLM should not be called because api_key is None
            result = create_default_llm()

            # OpenAILLM should not be called
            mock_openai_llm.assert_not_called()

            # Result should be None because openai_api_key is None
            assert result is None

    def test_zhipu_with_empty_string_api_key_returns_none(self, monkeypatch):
        """Test Zhipu LLM creation with empty string API key returns None."""
        # Set environment variables for Zhipu
        monkeypatch.setenv("ZHIPU_API_KEY", "")  # Empty string
        monkeypatch.setenv("ZHIPU_MODEL_NAME", "glm-4.7")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("ZHIPU_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_MODEL", raising=False)

        # Mock ZhipuLLM constructor
        with patch(
            "xagent.web.services.agent_service_manager.ZhipuLLM"
        ) as mock_zhipu_llm:
            # ZhipuLLM should not be called because api_key is empty string
            result = create_default_llm()

            # ZhipuLLM should not be called
            mock_zhipu_llm.assert_not_called()

            # Result should be None because empty string API key is not allowed for Zhipu
            assert result is None

    def test_zhipu_with_valid_api_key(self, monkeypatch):
        """Test Zhipu LLM creation with valid API key."""
        # Set environment variables for Zhipu
        zhipu_api_key = "valid-zhipu-api-key"
        monkeypatch.setenv("ZHIPU_API_KEY", zhipu_api_key)
        monkeypatch.setenv("ZHIPU_MODEL_NAME", "glm-4.7")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("ZHIPU_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_MODEL", raising=False)

        with patch(
            "xagent.web.services.agent_service_manager.ZhipuLLM"
        ) as mock_zhipu_llm:
            mock_zhipu_llm.return_value = None

            result = create_default_llm()

            mock_zhipu_llm.assert_called_once()
            call_args = mock_zhipu_llm.call_args

            # Check Zhipu parameters are passed correctly
            assert call_args.kwargs["api_key"] == zhipu_api_key
            assert call_args.kwargs["model_name"] == "glm-4.7"
            assert call_args.kwargs["base_url"] is None
            assert result is None

    def test_zhipu_detection_by_model_name(self, monkeypatch):
        """Test Zhipu detection based on model name."""
        # Set environment variables for Zhipu
        zhipu_api_key = "zhipu-api-key"
        monkeypatch.setenv("ZHIPU_API_KEY", zhipu_api_key)
        monkeypatch.setenv("ZHIPU_MODEL_NAME", "glm-4.7-flash")  # Zhipu model
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("ZHIPU_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_MODEL", raising=False)

        with patch(
            "xagent.web.services.agent_service_manager.ZhipuLLM"
        ) as mock_zhipu_llm:
            mock_zhipu_llm.return_value = None

            result = create_default_llm()

            # ZhipuLLM should be called
            mock_zhipu_llm.assert_called_once()
            call_args = mock_zhipu_llm.call_args

            # Check Zhipu model is detected and used
            assert call_args.kwargs["api_key"] == zhipu_api_key
            assert call_args.kwargs["model_name"] == "glm-4.7-flash"
            assert result is None

    def test_zhipu_detection_by_base_url(self, monkeypatch):
        """Test Zhipu detection based on base URL."""
        # Set environment variables
        zhipu_api_key = "zhipu-api-key"
        zhipu_base_url = "https://open.bigmodel.cn/api/paas/v4"
        monkeypatch.setenv("ZHIPU_API_KEY", zhipu_api_key)
        monkeypatch.setenv("ZHIPU_BASE_URL", zhipu_base_url)
        monkeypatch.setenv("ZHIPU_MODEL_NAME", "gpt-4o-mini")  # Not a Zhipu model name
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_MODEL", raising=False)

        with patch(
            "xagent.web.services.agent_service_manager.ZhipuLLM"
        ) as mock_zhipu_llm:
            mock_zhipu_llm.return_value = None

            result = create_default_llm()

            # ZhipuLLM should be called because base URL indicates Zhipu
            mock_zhipu_llm.assert_called_once()
            call_args = mock_zhipu_llm.call_args

            # Check Zhipu is used based on base URL
            assert call_args.kwargs["api_key"] == zhipu_api_key
            assert call_args.kwargs["base_url"] == zhipu_base_url
            assert call_args.kwargs["model_name"] == "gpt-4o-mini"
            assert result is None

    def test_openai_with_valid_api_key(self, monkeypatch):
        """Test OpenAI LLM creation with valid API key."""
        # Set environment variables
        openai_api_key = "valid-openai-api-key"
        monkeypatch.setenv("OPENAI_API_KEY", openai_api_key)
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
        monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
        monkeypatch.delenv("ZHIPU_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("ZHIPU_MODEL_NAME", raising=False)

        with patch(
            "xagent.web.services.agent_service_manager.OpenAILLM"
        ) as mock_openai_llm:
            mock_openai_llm.return_value = None

            result = create_default_llm()

            mock_openai_llm.assert_called_once()
            call_args = mock_openai_llm.call_args

            # Check OpenAI parameters are passed correctly
            assert call_args.kwargs["api_key"] == openai_api_key
            assert call_args.kwargs["model_name"] == "gpt-4o"
            assert call_args.kwargs["base_url"] is None
            assert result is None

    def test_openai_with_base_url(self, monkeypatch):
        """Test OpenAI LLM creation with base URL."""
        # Set environment variables
        openai_api_key = "openai-api-key"
        openai_base_url = "https://api.openai.com/v1"
        monkeypatch.setenv("OPENAI_API_KEY", openai_api_key)
        monkeypatch.setenv("OPENAI_BASE_URL", openai_base_url)
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
        monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
        monkeypatch.delenv("ZHIPU_BASE_URL", raising=False)
        monkeypatch.delenv("ZHIPU_MODEL_NAME", raising=False)

        with patch(
            "xagent.web.services.agent_service_manager.OpenAILLM"
        ) as mock_openai_llm:
            mock_openai_llm.return_value = None

            result = create_default_llm()

            mock_openai_llm.assert_called_once()
            call_args = mock_openai_llm.call_args

            # Check base_url is passed correctly
            assert call_args.kwargs["api_key"] == openai_api_key
            assert call_args.kwargs["base_url"] == openai_base_url
            assert call_args.kwargs["model_name"] == "gpt-4o-mini"
            assert result is None

    def test_openai_with_empty_string_base_url(self, monkeypatch):
        """Test OpenAI LLM creation with empty string base URL."""
        # Set environment variables
        openai_api_key = "openai-api-key"
        monkeypatch.setenv("OPENAI_API_KEY", openai_api_key)
        monkeypatch.setenv("OPENAI_BASE_URL", "")  # Empty string
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
        monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
        monkeypatch.delenv("ZHIPU_BASE_URL", raising=False)
        monkeypatch.delenv("ZHIPU_MODEL_NAME", raising=False)

        with patch(
            "xagent.web.services.agent_service_manager.OpenAILLM"
        ) as mock_openai_llm:
            mock_openai_llm.return_value = None

            result = create_default_llm()

            mock_openai_llm.assert_called_once()
            call_args = mock_openai_llm.call_args

            # Check base_url is empty string
            assert call_args.kwargs["api_key"] == openai_api_key
            assert call_args.kwargs["base_url"] == ""
            assert call_args.kwargs["model_name"] == "gpt-4o-mini"
            assert result is None

    def test_zhipu_with_empty_string_base_url(self, monkeypatch):
        """Test Zhipu LLM creation with empty string base URL."""
        # Set environment variables
        zhipu_api_key = "zhipu-api-key"
        monkeypatch.setenv("ZHIPU_API_KEY", zhipu_api_key)
        monkeypatch.setenv("ZHIPU_BASE_URL", "")  # Empty string
        monkeypatch.setenv("ZHIPU_MODEL_NAME", "glm-4.7")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_MODEL", raising=False)

        with patch(
            "xagent.web.services.agent_service_manager.ZhipuLLM"
        ) as mock_zhipu_llm:
            mock_zhipu_llm.return_value = None

            result = create_default_llm()

            mock_zhipu_llm.assert_called_once()
            call_args = mock_zhipu_llm.call_args

            # Check base_url is empty string
            assert call_args.kwargs["api_key"] == zhipu_api_key
            assert call_args.kwargs["base_url"] == ""
            assert call_args.kwargs["model_name"] == "glm-4.7"
            assert result is None

    def test_model_name_defaults(self, monkeypatch):
        """Test model name defaults."""
        # Set environment variables for OpenAI
        openai_api_key = "openai-api-key"
        monkeypatch.setenv("OPENAI_API_KEY", openai_api_key)
        monkeypatch.delenv("OPENAI_MODEL", raising=False)  # None, should use default
        monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
        monkeypatch.delenv("ZHIPU_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("ZHIPU_MODEL_NAME", raising=False)

        with patch(
            "xagent.web.services.agent_service_manager.OpenAILLM"
        ) as mock_openai_llm:
            mock_openai_llm.return_value = None

            result = create_default_llm()

            mock_openai_llm.assert_called_once()
            call_args = mock_openai_llm.call_args

            # Check model_name is default "gpt-4o-mini"
            assert call_args.kwargs["model_name"] == "gpt-4o-mini"
            assert result is None

    def test_zhipu_model_name_default(self, monkeypatch):
        """Test Zhipu model name default."""
        # Set environment variables for Zhipu
        zhipu_api_key = "zhipu-api-key"
        monkeypatch.setenv("ZHIPU_API_KEY", zhipu_api_key)
        monkeypatch.setenv(
            "ZHIPU_MODEL_NAME", "glm-4.7"
        )  # Set a Zhipu model name to trigger detection
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("ZHIPU_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_MODEL", raising=False)

        with patch(
            "xagent.web.services.agent_service_manager.ZhipuLLM"
        ) as mock_zhipu_llm:
            mock_zhipu_llm.return_value = None

            result = create_default_llm()

            mock_zhipu_llm.assert_called_once()
            call_args = mock_zhipu_llm.call_args

            # Check model_name is "glm-4.7" (the one we set)
            assert call_args.kwargs["model_name"] == "glm-4.7"
            assert result is None

    def test_thinking_mode_configuration_for_zhipu(self, monkeypatch):
        """Test thinking mode configuration for Zhipu LLM."""
        # Set environment variables
        zhipu_api_key = "zhipu-api-key"
        monkeypatch.setenv("ZHIPU_API_KEY", zhipu_api_key)
        monkeypatch.setenv("ZHIPU_MODEL_NAME", "glm-4.7")
        monkeypatch.setenv("ZHIPU_THINKING_MODE", "true")  # Enable thinking mode
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("ZHIPU_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_MODEL", raising=False)

        with patch(
            "xagent.web.services.agent_service_manager.ZhipuLLM"
        ) as mock_zhipu_llm:
            mock_zhipu_llm.return_value = None

            result = create_default_llm()

            mock_zhipu_llm.assert_called_once()
            call_args = mock_zhipu_llm.call_args

            # Check thinking mode is enabled
            assert call_args.kwargs["api_key"] == zhipu_api_key
            assert call_args.kwargs["thinking_mode"] is True
            assert result is None

    def test_thinking_mode_auto_for_zhipu(self, monkeypatch):
        """Test thinking mode 'auto' configuration for Zhipu LLM."""
        # Set environment variables
        zhipu_api_key = "zhipu-api-key"
        monkeypatch.setenv("ZHIPU_API_KEY", zhipu_api_key)
        monkeypatch.setenv("ZHIPU_MODEL_NAME", "glm-4.7")
        monkeypatch.setenv("ZHIPU_THINKING_MODE", "auto")  # Auto thinking mode
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("ZHIPU_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_MODEL", raising=False)

        with patch(
            "xagent.web.services.agent_service_manager.ZhipuLLM"
        ) as mock_zhipu_llm:
            mock_zhipu_llm.return_value = None

            result = create_default_llm()

            mock_zhipu_llm.assert_called_once()
            call_args = mock_zhipu_llm.call_args

            # Check thinking mode is None for 'auto'
            assert call_args.kwargs["api_key"] == zhipu_api_key
            assert call_args.kwargs["thinking_mode"] is None
            assert result is None

    def test_no_api_key_returns_none(self, monkeypatch):
        """Test that None is returned when no API key is available."""
        # Remove all API key environment variables
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("ZHIPU_BASE_URL", raising=False)
        monkeypatch.delenv("ZHIPU_MODEL_NAME", raising=False)
        monkeypatch.delenv("OPENAI_MODEL", raising=False)

        result = create_default_llm()

        # Should return None when no API key is available
        assert result is None

    def test_openai_and_zhipu_both_exist_zhipu_used(self, monkeypatch):
        """Test when both OpenAI and Zhipu API keys exist, Zhipu is used (Zhipu detection priority)."""
        # Set both environment variables
        openai_api_key = "openai-api-key"
        zhipu_api_key = "zhipu-api-key"
        monkeypatch.setenv("OPENAI_API_KEY", openai_api_key)
        monkeypatch.setenv("ZHIPU_API_KEY", zhipu_api_key)
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
        monkeypatch.setenv(
            "ZHIPU_MODEL_NAME", "glm-4.7"
        )  # Zhipu model triggers detection
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("ZHIPU_BASE_URL", raising=False)

        with patch(
            "xagent.web.services.agent_service_manager.ZhipuLLM"
        ) as mock_zhipu_llm:
            mock_zhipu_llm.return_value = None

            result = create_default_llm()

            # ZhipuLLM should be called (Zhipu detection priority)
            mock_zhipu_llm.assert_called_once()
            call_args = mock_zhipu_llm.call_args

            # Check Zhipu parameters
            assert call_args.kwargs["api_key"] == zhipu_api_key
            assert call_args.kwargs["model_name"] == "glm-4.7"

            with patch(
                "xagent.web.services.agent_service_manager.OpenAILLM"
            ) as mock_openai_llm:
                # OpenAILLM should not be called
                mock_openai_llm.assert_not_called()

            assert result is None

    def test_openai_empty_string_and_zhipu_valid_zhipu_used(self, monkeypatch):
        """Test when OpenAI API key is empty string and Zhipu is valid, Zhipu is used (Zhipu detection priority)."""
        # Set environment variables
        monkeypatch.setenv("OPENAI_API_KEY", "")  # Empty string
        monkeypatch.setenv("ZHIPU_API_KEY", "valid-zhipu-api-key")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
        monkeypatch.setenv(
            "ZHIPU_MODEL_NAME", "glm-4.7"
        )  # Zhipu model triggers detection
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("ZHIPU_BASE_URL", raising=False)

        with patch(
            "xagent.web.services.agent_service_manager.ZhipuLLM"
        ) as mock_zhipu_llm:
            mock_zhipu_llm.return_value = None

            result = create_default_llm()

            # ZhipuLLM should be called (Zhipu detection priority)
            mock_zhipu_llm.assert_called_once()
            call_args = mock_zhipu_llm.call_args

            # Check Zhipu parameters
            assert call_args.kwargs["api_key"] == "valid-zhipu-api-key"
            assert call_args.kwargs["model_name"] == "glm-4.7"

            with patch(
                "xagent.web.services.agent_service_manager.OpenAILLM"
            ) as mock_openai_llm:
                # OpenAILLM should not be called
                mock_openai_llm.assert_not_called()

            assert result is None

    def test_deepseek_with_valid_api_key(self, monkeypatch):
        """Test DeepSeek LLM creation with valid API key."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "valid-deepseek-api-key")
        monkeypatch.setenv("DEEPSEEK_MODEL_NAME", "deepseek-v4-pro")
        monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
        monkeypatch.delenv("ZHIPU_BASE_URL", raising=False)
        monkeypatch.delenv("ZHIPU_MODEL_NAME", raising=False)

        with patch(
            "xagent.web.services.agent_service_manager.DeepSeekLLM"
        ) as mock_deepseek_llm:
            mock_deepseek_llm.return_value = None

            result = create_default_llm()

            mock_deepseek_llm.assert_called_once_with(
                model_name="deepseek-v4-pro",
                base_url="https://api.deepseek.com",
                api_key="valid-deepseek-api-key",
            )
            assert result is None

    def test_deepseek_placeholder_api_key_is_ignored(self, monkeypatch):
        """DeepSeek example.env placeholders should not enable env fallback."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "your-deepseek-api-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
        monkeypatch.delenv("ZHIPU_BASE_URL", raising=False)
        monkeypatch.delenv("ZHIPU_MODEL_NAME", raising=False)

        with patch(
            "xagent.web.services.agent_service_manager.DeepSeekLLM"
        ) as mock_deepseek_llm:
            result = create_default_llm()

        mock_deepseek_llm.assert_not_called()
        assert result is None

    def test_openai_placeholder_does_not_block_deepseek(self, monkeypatch):
        """OpenAI example.env placeholder should not prevent DeepSeek fallback."""
        monkeypatch.setenv("OPENAI_API_KEY", "your-openai-api-key")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "valid-deepseek-api-key")
        monkeypatch.setenv("DEEPSEEK_MODEL_NAME", "deepseek-v4-flash")
        monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
        monkeypatch.delenv("ZHIPU_BASE_URL", raising=False)
        monkeypatch.delenv("ZHIPU_MODEL_NAME", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)

        with (
            patch(
                "xagent.web.services.agent_service_manager.OpenAILLM"
            ) as mock_openai_llm,
            patch(
                "xagent.web.services.agent_service_manager.DeepSeekLLM"
            ) as mock_deepseek_llm,
        ):
            mock_openai_llm.return_value = None
            mock_deepseek_llm.return_value = None

            result = create_default_llm()

        mock_openai_llm.assert_not_called()
        mock_deepseek_llm.assert_called_once_with(
            model_name="deepseek-v4-flash",
            base_url=None,
            api_key="valid-deepseek-api-key",
        )
        assert result is None

    def test_openai_and_deepseek_both_exist_openai_used(self, monkeypatch):
        """When both keys exist and Zhipu is not selected, preserve OpenAI priority."""
        monkeypatch.setenv("OPENAI_API_KEY", "openai-api-key")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-api-key")
        monkeypatch.setenv("DEEPSEEK_MODEL_NAME", "deepseek-v4-flash")
        monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
        monkeypatch.delenv("ZHIPU_BASE_URL", raising=False)
        monkeypatch.delenv("ZHIPU_MODEL_NAME", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)

        with (
            patch(
                "xagent.web.services.agent_service_manager.DeepSeekLLM"
            ) as mock_deepseek_llm,
            patch(
                "xagent.web.services.agent_service_manager.OpenAILLM"
            ) as mock_openai_llm,
        ):
            mock_openai_llm.return_value = None
            mock_deepseek_llm.return_value = None

            result = create_default_llm()

            mock_openai_llm.assert_called_once()
            mock_deepseek_llm.assert_not_called()
            assert result is None


if __name__ == "__main__":
    pytest.main([__file__])
