"""Service to fetch available models from various providers using their SDKs."""

import logging
from typing import Any, Dict, List, Optional

from ...core.utils.security import redact_sensitive_text

logger = logging.getLogger(__name__)


async def fetch_openai_models(
    api_key: str, base_url: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Fetch available models from OpenAI using OpenAILLM.list_available_models().

    Args:
        api_key: OpenAI API key
        base_url: Custom base URL (optional)

    Returns:
        List of available models with their information
    """
    from ...core.model.chat.basic.openai import OpenAILLM

    return await OpenAILLM.list_available_models(api_key, base_url)


async def fetch_zhipu_models(
    api_key: str, base_url: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Fetch available models from Zhipu AI using ZhipuLLM.list_available_models().

    Args:
        api_key: Zhipu API key
        base_url: Custom base URL (optional)

    Returns:
        List of available Zhipu models
    """
    from ...core.model.chat.basic.zhipu import ZhipuLLM

    return await ZhipuLLM.list_available_models(api_key, base_url)


async def fetch_claude_models(
    api_key: str, base_url: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Fetch available models from Anthropic Claude using ClaudeLLM.list_available_models().

    Args:
        api_key: Anthropic API key
        base_url: Custom base URL (optional)

    Returns:
        List of available Claude models
    """
    from ...core.model.chat.basic.claude import ClaudeLLM

    return await ClaudeLLM.list_available_models(api_key, base_url)


async def fetch_gemini_models(
    api_key: str, base_url: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Fetch available models from Google Gemini using GeminiLLM.list_available_models().

    Args:
        api_key: Google API key
        base_url: Custom base URL (optional)

    Returns:
        List of available Gemini models
    """
    from ...core.model.chat.basic.gemini import GeminiLLM

    return await GeminiLLM.list_available_models(api_key, base_url)


async def fetch_xinference_models(
    api_key: str, base_url: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Fetch available models from Xinference using XinferenceLLM.list_available_models().

    Args:
        api_key: Xinference API key (optional)
        base_url: Xinference server base URL (required)

    Returns:
        List of available Xinference models
    """
    if not base_url:
        raise ValueError("base_url is required for Xinference")

    from ...core.model.chat.basic.xinference import XinferenceLLM

    return await XinferenceLLM.list_available_models(base_url=base_url, api_key=api_key)


# Provider registry mapping provider names to their fetch functions
PROVIDER_FETCHERS: Dict[str, Any] = {
    "openai": fetch_openai_models,
    "zhipu": fetch_zhipu_models,
    "claude": fetch_claude_models,
    "anthropic": fetch_claude_models,
    "gemini": fetch_gemini_models,
    "google": fetch_gemini_models,
    "xinference": fetch_xinference_models,
}


async def fetch_models_from_provider(
    provider: str,
    api_key: str,
    base_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch available models from a specific provider.

    Args:
        provider: Provider name (openai, zhipu, claude, etc.)
        api_key: API key for the provider
        base_url: Custom base URL (optional)

    Returns:
        List of available models
    """
    fetcher = PROVIDER_FETCHERS.get(provider.lower())

    if not fetcher:
        logger.warning(f"Unknown provider: {provider}")
        return []

    try:
        result: List[Dict[str, Any]] = await fetcher(api_key, base_url)
        return result
    except Exception as e:
        logger.error(
            "Error fetching models from %s: %s",
            provider,
            redact_sensitive_text(str(e)),
        )
        raise


def get_supported_providers() -> List[Dict[str, Any]]:
    """Get list of supported providers.

    Returns:
        List of provider information
    """
    return [
        {
            "id": "openai",
            "name": "OpenAI",
            "description": "OpenAI API compatible models",
            "requires_base_url": False,
        },
        {
            "id": "claude",
            "name": "Anthropic Claude",
            "description": "Anthropic's Claude models",
            "requires_base_url": False,
        },
        {
            "id": "gemini",
            "name": "Google Gemini",
            "description": "Google's Gemini models",
            "requires_base_url": False,
        },
        {
            "id": "xinference",
            "name": "Xinference",
            "description": "Xinference models for local inference",
            "requires_base_url": True,
        },
        {
            "id": "zhipu",
            "name": "Zhipu AI",
            "description": "Zhipu AI models (GLM series) using zai SDK",
            "requires_base_url": False,
        },
        {
            "id": "dashscope",
            "name": "DashScope",
            "description": "Alibaba Cloud's DashScope models",
            "requires_base_url": False,
            "default_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        },
    ]
