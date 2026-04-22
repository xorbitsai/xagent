import os

from ....model import ChatModelConfig, ModelConfig
from ....retry import create_retry_wrapper
from ...providers import provider_compatibility_for_provider
from ..error import retry_on
from .azure_openai import AzureOpenAILLM
from .base import BaseLLM
from .claude import ClaudeLLM
from .gemini import GeminiLLM
from .openai import OpenAILLM
from .xinference import XinferenceLLM
from .zhipu import ZhipuLLM


def create_base_llm(model: ModelConfig) -> BaseLLM:
    """
    Creates a custom BaseLLM instance from a ModelConfig.
    """
    if not isinstance(model, ChatModelConfig):
        raise TypeError(f"Invalid model type: {type(model).__name__}")

    compatibility = provider_compatibility_for_provider(model.model_provider)

    if model.model_provider == "openai" or compatibility == "openai_compatible":
        llm: BaseLLM = OpenAILLM(
            model_name=model.model_name,
            api_key=model.api_key,
            base_url=model.base_url,
            default_temperature=model.default_temperature,
            default_max_tokens=model.default_max_tokens,
            timeout=model.timeout,
            abilities=model.abilities,
        )
    elif model.model_provider == "claude" or compatibility == "claude_compatible":
        llm = ClaudeLLM(
            model_name=model.model_name,
            api_key=model.api_key,
            base_url=model.base_url,
            default_temperature=model.default_temperature,
            default_max_tokens=model.default_max_tokens,
            timeout=model.timeout,
            abilities=model.abilities,
        )
    elif model.model_provider == "azure_openai":
        llm = AzureOpenAILLM(
            model_name=model.model_name,
            azure_endpoint=model.base_url,  # Reuse base_url as azure_endpoint
            api_key=model.api_key,
            api_version=os.getenv("OPENAI_API_VERSION", "2024-08-01-preview"),
            default_temperature=model.default_temperature,
            default_max_tokens=model.default_max_tokens,
            timeout=model.timeout,
            abilities=model.abilities,
        )
    elif model.model_provider == "zhipu":
        llm = ZhipuLLM(
            model_name=model.model_name,
            api_key=model.api_key,
            base_url=model.base_url,
            default_temperature=model.default_temperature,
            default_max_tokens=model.default_max_tokens,
            timeout=model.timeout,
            abilities=model.abilities,
        )
    elif model.model_provider == "gemini":
        llm = GeminiLLM(
            model_name=model.model_name,
            api_key=model.api_key,
            base_url=model.base_url,
            default_temperature=model.default_temperature,
            default_max_tokens=model.default_max_tokens,
            timeout=model.timeout,
            abilities=model.abilities,
        )
    elif model.model_provider == "xinference":
        llm = XinferenceLLM(
            model_name=model.model_name,
            base_url=model.base_url,
            api_key=model.api_key,
            default_temperature=model.default_temperature,
            default_max_tokens=model.default_max_tokens,
            timeout=model.timeout,
            abilities=model.abilities,
        )
    else:
        raise TypeError(f"Unsupported LLM model type: {model.model_provider}")

    return create_retry_wrapper(
        llm,
        BaseLLM,  # type: ignore[type-abstract]
        retry_methods={"chat", "vision_chat", "stream_chat"},
        retry_on=retry_on,
    )
