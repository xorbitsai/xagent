"""Service to fetch available models from various providers using their SDKs."""

import logging
from typing import Any, Dict, List, Optional

import aiohttp

from ...core.model.providers import (
    canonical_provider_name,
    curated_models_for_provider,
    default_base_url_for_provider,
    get_supported_provider_metadata,
)
from ...core.utils.security import redact_sensitive_text

logger = logging.getLogger(__name__)


def _static_model_list(models: tuple[str, ...], owned_by: str) -> List[Dict[str, Any]]:
    return [{"id": model_id, "created": 0, "owned_by": owned_by} for model_id in models]


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


async def fetch_deepseek_models(
    api_key: str, base_url: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Return DeepSeek v4 curated model list."""
    from ...core.model.chat.basic.deepseek import DeepSeekLLM

    return await DeepSeekLLM.list_available_models(api_key, base_url)


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


async def fetch_xinference_rerank_models(
    api_key: str, base_url: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Fetch available rerank models from a Xinference server.

    Args:
        api_key: Xinference API key (optional)
        base_url: Xinference server base URL (required)

    Returns:
        List of available rerank models on the server, shaped as
        ``{"id", "model_uid", ...}`` (see ``XinferenceRerank.list_available_models``).
    """
    if not base_url:
        raise ValueError("base_url is required for Xinference rerank")

    from ...core.model.rerank.xinference import XinferenceRerank

    return XinferenceRerank.list_available_models(base_url=base_url, api_key=api_key)


async def fetch_xinference_video_models(
    api_key: str, base_url: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Fetch available video models from a Xinference server."""
    if not base_url:
        raise ValueError("base_url is required for Xinference video")

    from ...core.model.chat.basic.xinference import _normalize_model_list_response

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    timeout = aiohttp.ClientTimeout(total=30.0)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(
            f"{base_url.rstrip('/')}/v1/models", headers=headers
        ) as response:
            if response.status != 200:
                try:
                    detail = await response.json()
                except Exception:
                    detail = await response.text()
                raise RuntimeError(
                    f"Failed to list Xinference video models: HTTP {response.status}, detail: {detail}"
                )
            response_data = await response.json()
    data_list = response_data.get("data", []) if isinstance(response_data, dict) else []
    models_list = {
        str(item["id"]): item
        for item in data_list
        if isinstance(item, dict) and item.get("id")
    }
    normalized_models = _normalize_model_list_response(models_list)

    result: List[Dict[str, Any]] = []
    for model_uid, model_info in normalized_models:
        abilities = [str(item) for item in model_info.get("model_ability", [])]
        model_type = str(model_info.get("model_type", ""))
        is_video = model_type == "video" or any(
            "video" in ability.lower() for ability in abilities
        )
        if not is_video:
            continue
        result.append(
            {
                "id": model_info.get("model_name", model_uid),
                "model_uid": model_uid,
                "model_type": model_type,
                "category": "video",
                "model_ability": ["generate"],
                "abilities": ["generate"],
                "description": model_info.get("model_description", ""),
            }
        )
    return result


async def fetch_elevenlabs_models(
    api_key: str, base_url: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Fetch available ElevenLabs TTS models using the official SDK."""

    from ...core.model.tts.elevenlabs import ElevenLabsTTS

    return await ElevenLabsTTS.async_list_available_models(
        api_key=api_key, base_url=base_url
    )


async def fetch_dashscope_rerank_models(
    api_key: str, base_url: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Return curated DashScope rerank models.

    DashScope's OpenAI-compatible model list endpoint does NOT expose rerank
    models, so we cannot reuse ``fetch_openai_models`` here. The rerank
    families currently supported by ``DashscopeRerank`` are documented in
    ``NEW_FORMAT_MODELS`` / ``OLD_FORMAT_MODELS`` in
    ``xagent.core.model.rerank.dashscope``; we expose them as a static curated
    list so the UI can preselect a known-good model.
    """
    _ = api_key, base_url

    from ...core.model.rerank.dashscope import NEW_FORMAT_MODELS, OLD_FORMAT_MODELS

    models: List[Dict[str, Any]] = []
    for model_id in sorted(NEW_FORMAT_MODELS | OLD_FORMAT_MODELS):
        models.append(
            {
                "id": model_id,
                "object": "model",
                "owned_by": "dashscope",
            }
        )
    return models


async def fetch_alibaba_coding_plan_models(
    api_key: str, base_url: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Return curated Alibaba Bailian coding plan models."""
    _ = api_key, base_url
    return _static_model_list(
        curated_models_for_provider("alibaba-coding-plan"),
        owned_by="alibaba-coding-plan",
    )


async def fetch_alibaba_coding_plan_cn_models(
    api_key: str, base_url: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Return curated Alibaba Bailian coding plan models (China)."""
    _ = api_key, base_url
    return _static_model_list(
        curated_models_for_provider("alibaba-coding-plan-cn"),
        owned_by="alibaba-coding-plan-cn",
    )


async def fetch_minimax_coding_plan_models(
    api_key: str, base_url: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Return curated MiniMax coding plan models (minimax.io)."""
    _ = api_key, base_url
    return _static_model_list(
        curated_models_for_provider("minimax-coding-plan"),
        owned_by="minimax-coding-plan",
    )


async def fetch_minimax_cn_coding_plan_models(
    api_key: str, base_url: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Return curated MiniMax coding plan models (minimaxi.com)."""
    _ = api_key, base_url
    return _static_model_list(
        curated_models_for_provider("minimax-cn-coding-plan"),
        owned_by="minimax-cn-coding-plan",
    )


async def fetch_kimi_for_coding_models(
    api_key: str, base_url: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Fetch available Kimi For Coding models via the Claude-compatible API."""
    return await fetch_claude_models(api_key, base_url)


async def _fetch_openai_compatible_video_models(
    api_key: str,
    base_url: Optional[str],
    *,
    owned_by: str,
    default_base_url: str,
) -> List[Dict[str, Any]]:
    if not api_key or not base_url:
        return []

    provider_models = await fetch_openai_models(api_key, base_url)
    dynamic_models: List[Dict[str, Any]] = []
    for model in provider_models:
        model_id = str(model.get("id") or "")
        if not model_id or "seedance" not in model_id.lower():
            continue
        dynamic_models.append(
            {
                **model,
                "owned_by": model.get("owned_by") or owned_by,
                "category": "video",
                "abilities": ["generate"],
                "model_ability": ["generate"],
                "base_url": base_url,
                "default_base_url": default_base_url,
            }
        )
    return dynamic_models


async def fetch_volcengine_ark_video_models(
    api_key: str, base_url: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Fetch Volcengine Ark video models from the live model list endpoint."""
    from xagent.core.model.video.ark import ARK_DOMESTIC_BASE_URL

    resolved_base_url = base_url or ARK_DOMESTIC_BASE_URL
    return await _fetch_openai_compatible_video_models(
        api_key,
        resolved_base_url,
        owned_by="volcengine-ark",
        default_base_url=ARK_DOMESTIC_BASE_URL,
    )


async def fetch_byteplus_ark_video_models(
    api_key: str, base_url: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Fetch BytePlus Ark video models from the live model list endpoint."""
    from xagent.core.model.video.ark import ARK_BYTEPLUS_BASE_URL

    resolved_base_url = base_url or ARK_BYTEPLUS_BASE_URL
    return await _fetch_openai_compatible_video_models(
        api_key,
        resolved_base_url,
        owned_by="byteplus-ark",
        default_base_url=ARK_BYTEPLUS_BASE_URL,
    )


# Provider registry mapping provider names to their fetch functions
PROVIDER_FETCHERS: Dict[str, Any] = {
    "openai": fetch_openai_models,
    "openrouter": fetch_openai_models,
    "deepseek": fetch_deepseek_models,
    "dashscope": fetch_openai_models,
    "zhipu": fetch_zhipu_models,
    "claude": fetch_claude_models,
    "anthropic": fetch_claude_models,
    "gemini": fetch_gemini_models,
    "google": fetch_gemini_models,
    "xinference": fetch_xinference_models,
    "xinference-rerank": fetch_xinference_rerank_models,
    "xinference-video": fetch_xinference_video_models,
    "elevenlabs": fetch_elevenlabs_models,
    "dashscope-rerank": fetch_dashscope_rerank_models,
    "zai-coding-plan": fetch_openai_models,
    "zhipuai-coding-plan": fetch_openai_models,
    "alibaba-coding-plan": fetch_alibaba_coding_plan_models,
    "alibaba-coding-plan-cn": fetch_alibaba_coding_plan_cn_models,
    "minimax-coding-plan": fetch_minimax_coding_plan_models,
    "minimax-cn-coding-plan": fetch_minimax_cn_coding_plan_models,
    "kimi-for-coding": fetch_kimi_for_coding_models,
    "volcengine-ark": fetch_volcengine_ark_video_models,
    "byteplus-ark": fetch_byteplus_ark_video_models,
    "ark": fetch_volcengine_ark_video_models,
    "ark-video": fetch_volcengine_ark_video_models,
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
    provider_id = canonical_provider_name(provider)
    fetcher = PROVIDER_FETCHERS.get(provider_id)

    if not fetcher:
        logger.warning(f"Unknown provider: {provider}")
        return []

    try:
        resolved_base_url = base_url or default_base_url_for_provider(provider_id)
        result: List[Dict[str, Any]] = await fetcher(api_key, resolved_base_url)
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
    return get_supported_provider_metadata()
