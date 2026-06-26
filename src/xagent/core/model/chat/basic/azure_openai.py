import os
from typing import List, Optional

from openai import AsyncAzureOpenAI

from ..timeout_config import TimeoutConfig
from .openai import OpenAILLM


class AzureOpenAILLM(OpenAILLM):
    """
    Azure OpenAI LLM client using the official OpenAI SDK with Azure-specific configuration.

    This class extends OpenAILLM to support Azure OpenAI Service, which requires:
    - azure_endpoint: The Azure OpenAI endpoint URL
    - api_version: The API version (e.g., "2025-04-01-preview")
    - api_key: The Azure OpenAI API key

    All other functionality (chat, tool_calling, thinking_mode, vision) is inherited from OpenAILLM.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        default_temperature: Optional[float] = None,
        default_max_tokens: Optional[int] = None,
        timeout: float = 180.0,
        abilities: Optional[List[str]] = None,
        timeout_config: Optional[TimeoutConfig] = None,
    ):
        """
        Initialize the Azure OpenAI LLM client.

        Args:
            model_name: The Azure deployment name (e.g., "gpt-4o", "gpt-4o-mini")
            azure_endpoint: The Azure OpenAI endpoint URL (e.g., "https://your-resource.openai.azure.com/")
            api_key: The Azure OpenAI API key
            api_version: The API version (e.g., "2025-04-01-preview", "2024-08-01-preview")
            default_temperature: Default sampling temperature
            default_max_tokens: Default maximum tokens to generate
            timeout: Request timeout in seconds
            abilities: List of abilities supported by this model
        """
        # Store Azure-specific configuration
        self.azure_endpoint = (
            azure_endpoint
            or os.getenv("AZURE_OPENAI_ENDPOINT")
            or os.getenv("OPENAI_API_BASE")
        )
        self.api_version = api_version or os.getenv(
            "OPENAI_API_VERSION", "2024-08-01-preview"
        )

        # Validate Azure endpoint
        if not self.azure_endpoint:
            raise ValueError(
                "azure_endpoint must be provided either as a parameter or "
                "via AZURE_OPENAI_ENDPOINT or OPENAI_API_BASE environment variable"
            )

        # Call parent constructor with base_url=None (Azure doesn't use base_url)
        # We pass the api_key which will be used by the parent class
        # Use the same logic as OpenAILLM to allow empty string API key
        api_key_value = (
            api_key
            if api_key is not None
            else (os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"))
        )
        super().__init__(
            model_name=model_name,
            base_url=None,  # Azure uses azure_endpoint instead
            api_key=api_key_value,
            default_temperature=default_temperature,
            default_max_tokens=default_max_tokens,
            timeout=timeout,
            abilities=abilities,
            timeout_config=timeout_config,
        )

    def _ensure_client(self) -> None:
        """
        Ensure the Azure OpenAI client is initialized.

        Overrides the parent method to create an AsyncAzureOpenAI client instead of AsyncOpenAI.
        """
        if self._client is None:
            assert self.azure_endpoint is not None, "azure_endpoint must be set"
            self._client = AsyncAzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_version=self.api_version,
                api_key=self.api_key,
                timeout=self.timeout,
            )
