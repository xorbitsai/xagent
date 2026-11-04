"""Test cases for Azure OpenAI LLM implementation using OpenAI SDK."""

import pytest

from xagent.core.model.chat.basic.azure_openai import AzureOpenAILLM


class TestAzureOpenAILLM:
    """Test cases for Azure OpenAI LLM implementation."""

    @pytest.fixture
    def llm(self, azure_openai_llm_config):
        """Fixture providing Azure OpenAI LLM instance."""
        return AzureOpenAILLM(**azure_openai_llm_config)

    @pytest.mark.asyncio
    async def test_basic_chat_completion(self, llm, mock_chat_completion, mocker):
        """Test basic chat completion functionality."""
        # Setup mock
        mock_client = mocker.AsyncMock()
        mock_client.chat.completions.create.return_value = mock_chat_completion
        mocker.patch(
            "xagent.core.model.chat.basic.azure_openai.AsyncAzureOpenAI",
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
        assert call_args.kwargs["model"] == "gpt-4o"
        assert call_args.kwargs["messages"] == messages
        assert call_args.kwargs["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_azure_client_initialization(self, azure_openai_llm_config, mocker):
        """Test that Azure OpenAI client is initialized with correct parameters."""
        mock_client = mocker.AsyncMock()
        mockAzureOpenAI = mocker.patch(
            "xagent.core.model.chat.basic.azure_openai.AsyncAzureOpenAI",
            return_value=mock_client,
        )

        # Create LLM instance
        llm = AzureOpenAILLM(**azure_openai_llm_config)

        # Trigger client initialization
        llm._ensure_client()

        # Verify AsyncAzureOpenAI was called with correct parameters
        mockAzureOpenAI.assert_called_once()
        call_args = mockAzureOpenAI.call_args
        assert call_args.kwargs["azure_endpoint"] == "https://test.openai.azure.com/"
        assert call_args.kwargs["api_version"] == "2024-08-01-preview"
        assert call_args.kwargs["api_key"] == "test-api-key"
        assert call_args.kwargs["timeout"] == 30.0

    @pytest.mark.asyncio
    async def test_azure_endpoint_validation(self):
        """Test that azure_endpoint validation works correctly."""
        # Test missing azure_endpoint raises ValueError
        with pytest.raises(ValueError, match="azure_endpoint must be provided"):
            AzureOpenAILLM(
                model_name="gpt-4o",
                api_key="test-key",
            )

    @pytest.mark.asyncio
    async def test_inheritance_from_openai(self, llm):
        """Test that AzureOpenAILM properly inherits from OpenAILLM."""
        # Verify that AzureOpenAILM has all the same methods as OpenAILLM
        from xagent.core.model.chat.basic.openai import OpenAILLM

        assert isinstance(llm, OpenAILLM)
        assert hasattr(llm, "chat")
        assert hasattr(llm, "vision_chat")
        assert hasattr(llm, "abilities")
        assert hasattr(llm, "model_name")

    @pytest.mark.asyncio
    async def test_tool_calling(self, llm, mock_tool_call_completion, mocker):
        """Test tool calling functionality."""
        mock_client = mocker.AsyncMock()
        mock_client.chat.completions.create.return_value = mock_tool_call_completion
        mocker.patch(
            "xagent.core.model.chat.basic.azure_openai.AsyncAzureOpenAI",
            return_value=mock_client,
        )

        messages = [{"role": "user", "content": "What's the weather like?"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather",
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

        # Verify tool call response
        assert isinstance(response, dict)
        assert response["type"] == "tool_call"
        assert "tool_calls" in response
        assert len(response["tool_calls"]) > 0
        print(f"Tool call response: {response}")

        # Verify the API was called
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert "tools" in call_args.kwargs

    @pytest.mark.asyncio
    async def test_custom_api_version(self, mocker):
        """Test custom API version can be set."""
        mock_client = mocker.AsyncMock()
        mockAzureOpenAI = mocker.patch(
            "xagent.core.model.chat.basic.azure_openai.AsyncAzureOpenAI",
            return_value=mock_client,
        )

        llm = AzureOpenAILLM(
            model_name="gpt-4o",
            azure_endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            api_version="2025-04-01-preview",  # Custom API version
        )

        llm._ensure_client()

        # Verify custom API version was used
        call_args = mockAzureOpenAI.call_args
        assert call_args.kwargs["api_version"] == "2025-04-01-preview"

    @pytest.mark.asyncio
    async def test_environment_variable_fallback(self, mocker, monkeypatch):
        """Test that environment variables are used as fallback."""
        # Set environment variables
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://env.openai.azure.com/")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-api-key")
        monkeypatch.setenv("OPENAI_API_VERSION", "2024-02-01-preview")

        mock_client = mocker.AsyncMock()
        mockAzureOpenAI = mocker.patch(
            "xagent.core.model.chat.basic.azure_openai.AsyncAzureOpenAI",
            return_value=mock_client,
        )

        llm = AzureOpenAILLM(model_name="gpt-4o")
        llm._ensure_client()

        # Verify environment variables were used
        call_args = mockAzureOpenAI.call_args
        assert call_args.kwargs["azure_endpoint"] == "https://env.openai.azure.com/"
        assert call_args.kwargs["api_key"] == "env-api-key"
        assert call_args.kwargs["api_version"] == "2024-02-01-preview"

    @pytest.mark.asyncio
    async def test_empty_string_api_key(self, azure_openai_llm_config, monkeypatch):
        """Test that empty string API key is allowed and does not fallback to environment variable."""
        # Set environment variable to ensure we can test that it's NOT used
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-api-key-should-not-be-used")

        # Create Azure OpenAI LLM with empty string API key
        config = azure_openai_llm_config.copy()
        config["api_key"] = ""  # Empty string

        llm = AzureOpenAILLM(**config)

        # Verify that the API key is empty string, not the environment variable
        assert llm.api_key == ""
        print(
            f"Azure OpenAI empty string API key test passed: API key is '{llm.api_key}' (not using env var)"
        )

    @pytest.mark.asyncio
    async def test_none_api_key_with_env_fallback(
        self, azure_openai_llm_config, monkeypatch
    ):
        """Test None API key with environment variable fallback for Azure OpenAI."""
        # Set environment variable
        env_api_key = "env-api-key-for-azure"
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", env_api_key)

        # Create Azure OpenAI LLM with None API key
        config = azure_openai_llm_config.copy()
        config["api_key"] = None

        llm = AzureOpenAILLM(**config)

        # Verify that the API key is from environment variable
        assert llm.api_key == env_api_key
        print(f"Azure OpenAI None API key test passed: API key is '{llm.api_key}'")

    @pytest.mark.asyncio
    async def test_none_api_key_with_openai_env_fallback(
        self, azure_openai_llm_config, monkeypatch
    ):
        """Test None API key falls back to OPENAI_API_KEY when AZURE_OPENAI_API_KEY is not set."""
        # Set OPENAI_API_KEY environment variable (but not AZURE_OPENAI_API_KEY)
        openai_env_api_key = "openai-env-api-key"
        monkeypatch.setenv("OPENAI_API_KEY", openai_env_api_key)
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

        # Create Azure OpenAI LLM with None API key
        config = azure_openai_llm_config.copy()
        config["api_key"] = None

        llm = AzureOpenAILLM(**config)

        # Verify that the API key is from OPENAI_API_KEY environment variable
        assert llm.api_key == openai_env_api_key
        print(
            f"Azure OpenAI None API key with OPENAI_API_KEY fallback test passed: API key is '{llm.api_key}'"
        )

    @pytest.mark.asyncio
    async def test_missing_api_key_initialization(
        self, azure_openai_llm_config, monkeypatch
    ):
        """Test Azure OpenAI initialization when API key is completely missing."""
        # Remove all API key environment variables
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        # Create Azure OpenAI LLM with None API key and no environment variable
        config = azure_openai_llm_config.copy()
        config["api_key"] = None

        llm = AzureOpenAILLM(**config)

        # The LLM should initialize with None API key
        assert llm.api_key is None
        print(
            f"Azure OpenAI missing API key test: LLM initialized with API key = {llm.api_key}"
        )
