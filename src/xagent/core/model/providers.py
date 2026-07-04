import os
from typing import Any, Optional

# When an OpenRouter model carries this name, route the prompt through
# xrouter-llm (in-process) instead of calling OpenRouter directly with it.
AUTO_MODEL_NAME = "auto"

_PROVIDER_ALIASES: dict[str, str] = {
    "ark": "volcengine-ark",
    "ark-video": "volcengine-ark",
    "modelark": "volcengine-ark",
    "model_ark": "volcengine-ark",
    "volcengine": "volcengine-ark",
    "volcengine_ark": "volcengine-ark",
    "byteplus": "byteplus-ark",
    "byteplus_ark": "byteplus-ark",
    "zai_coding_plan": "zai-coding-plan",
    "zhipuai_coding_plan": "zhipuai-coding-plan",
    "alibaba_coding_plan": "alibaba-coding-plan",
    "alibaba_coding_plan_cn": "alibaba-coding-plan-cn",
    "minimax_coding_plan": "minimax-coding-plan",
    "minimax_cn_coding_plan": "minimax-cn-coding-plan",
    "kimi_for_coding": "kimi-for-coding",
}

# Provider default base URLs used when callers omit an explicit base URL.
_DEFAULT_BASE_URL_BY_PROVIDER: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://api.deepseek.com",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "volcengine-ark": "https://ark.cn-beijing.volces.com/api/v3",
    "byteplus-ark": "https://ark.ap-southeast.bytepluses.com/api/v3",
    "elevenlabs": "https://api.elevenlabs.io",
    # Opencode / models.dev naming
    "zai-coding-plan": "https://api.z.ai/api/coding/paas/v4",
    "zhipuai-coding-plan": "https://open.bigmodel.cn/api/coding/paas/v4",
    # Alibaba Bailian (Model Studio) coding plan
    "alibaba-coding-plan": "https://coding-intl.dashscope.aliyuncs.com/v1",
    "alibaba-coding-plan-cn": "https://coding.dashscope.aliyuncs.com/v1",
    "minimax-coding-plan": "https://api.minimax.io/anthropic",
    "minimax-cn-coding-plan": "https://api.minimaxi.com/anthropic",
    "kimi-for-coding": "https://api.kimi.com/coding",
}

# Provider-scoped base URL environment overrides. Keep these narrow so shared
# client implementations do not redirect sibling providers to the wrong API.
_BASE_URL_ENV_BY_PROVIDER: dict[str, str] = {
    "deepseek": "DEEPSEEK_BASE_URL",
    "dashscope": "DASHSCOPE_BASE_URL",
    "volcengine-ark": "ARK_BASE_URL",
    "byteplus-ark": "BYTEPLUS_ARK_BASE_URL",
}

_CURATED_MODELS_BY_PROVIDER: dict[str, tuple[str, ...]] = {
    # "auto" routes via xrouter-llm; the other entries are dispatched to
    # OpenRouter directly. Users may also type any OpenRouter slug.
    "openrouter": (AUTO_MODEL_NAME,),
    "deepseek": (
        "deepseek-v4-flash",
        "deepseek-v4-pro",
    ),
    "alibaba-coding-plan": (
        "glm-4.7",
        "glm-5",
        "qwen3-coder-next",
        "qwen3-coder-plus",
        "qwen3-max-2026-01-23",
        "qwen3.5-plus",
    ),
    "alibaba-coding-plan-cn": (
        "glm-4.7",
        "glm-5",
        "qwen3-coder-next",
        "qwen3-coder-plus",
        "qwen3-max-2026-01-23",
        "qwen3.5-plus",
    ),
    "minimax-coding-plan": (
        "MiniMax-M2",
        "MiniMax-M2.1",
        "MiniMax-M2.5",
    ),
    "minimax-cn-coding-plan": (
        "MiniMax-M2",
        "MiniMax-M2.1",
        "MiniMax-M2.5",
    ),
}

_SUPPORTED_PROVIDER_METADATA: tuple[dict[str, Any], ...] = (
    {
        "id": "openai",
        "name": "OpenAI",
        "description": "OpenAI API compatible models",
        "requires_base_url": False,
        "compatibility": "openai_compatible",
    },
    {
        "id": "claude",
        "name": "Anthropic Claude",
        "description": "Anthropic's Claude models",
        "requires_base_url": False,
        "compatibility": "claude_compatible",
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
        "id": "elevenlabs",
        "name": "ElevenLabs",
        "description": "ElevenLabs speech models for text-to-speech",
        "requires_base_url": False,
    },
    {
        "id": "deepseek",
        "name": "DeepSeek",
        "description": "DeepSeek v4 models with tool calling and thinking mode",
        "requires_base_url": False,
    },
    {
        "id": "openrouter",
        "name": "OpenRouter",
        "description": (
            "OpenRouter aggregator: reach Claude, Gemini, GPT, DeepSeek, GLM, "
            "and more through one OpenAI-compatible key. Use model 'auto' to let "
            "xrouter-llm pick the cheapest capable model per prompt."
        ),
        "requires_base_url": False,
        "compatibility": "openai_compatible",
    },
    {
        "id": "dashscope",
        "name": "DashScope",
        "description": "Alibaba Cloud's DashScope models",
        "requires_base_url": False,
        "compatibility": "openai_compatible",
    },
    {
        "id": "volcengine-ark",
        "name": "Volcengine Ark",
        "description": "Volcengine ModelArk provider for Seedance video generation",
        "requires_base_url": False,
        "default_base_url": "https://ark.cn-beijing.volces.com/api/v3",
    },
    {
        "id": "byteplus-ark",
        "name": "BytePlus Ark",
        "description": "BytePlus ModelArk provider for Seedance video generation",
        "requires_base_url": False,
        "default_base_url": "https://ark.ap-southeast.bytepluses.com/api/v3",
    },
    {
        "id": "alibaba-coding-plan",
        "name": "Alibaba Coding Plan",
        "description": "Alibaba Bailian (Model Studio) coding plan",
        "requires_base_url": False,
        "compatibility": "openai_compatible",
    },
    {
        "id": "alibaba-coding-plan-cn",
        "name": "Alibaba Coding Plan (China)",
        "description": "Alibaba Bailian (Model Studio) coding plan (China)",
        "requires_base_url": False,
        "compatibility": "openai_compatible",
    },
    {
        "id": "zhipu",
        "name": "Zhipu AI",
        "description": "Zhipu AI models (GLM series) using zai SDK",
        "requires_base_url": False,
    },
    {
        "id": "zai-coding-plan",
        "name": "Z.AI Coding Plan",
        "description": "GLM coding plan via Z.AI",
        "requires_base_url": False,
        "compatibility": "openai_compatible",
    },
    {
        "id": "zhipuai-coding-plan",
        "name": "Zhipu AI Coding Plan",
        "description": "GLM coding plan via Zhipu AI",
        "requires_base_url": False,
        "compatibility": "openai_compatible",
    },
    {
        "id": "minimax-coding-plan",
        "name": "MiniMax Coding Plan (International)",
        "description": "MiniMax coding plan via api.minimax.io",
        "requires_base_url": False,
        "compatibility": "claude_compatible",
        "default_base_url": "https://api.minimax.io/anthropic",
    },
    {
        "id": "minimax-cn-coding-plan",
        "name": "MiniMax Coding Plan (China)",
        "description": "MiniMax coding plan via api.minimaxi.com",
        "requires_base_url": False,
        "compatibility": "claude_compatible",
        "default_base_url": "https://api.minimaxi.com/anthropic",
    },
    {
        "id": "kimi-for-coding",
        "name": "Kimi For Coding",
        "description": "Kimi coding endpoint",
        "requires_base_url": False,
        "compatibility": "claude_compatible",
        "default_base_url": "https://api.kimi.com/coding",
    },
)


def _normalize_provider(provider: str) -> str:
    return provider.lower().strip()


def canonical_provider_name(provider: str) -> str:
    normalized = _normalize_provider(provider)
    return _PROVIDER_ALIASES.get(normalized, normalized)


def is_placeholder_api_key(api_key: Optional[str]) -> bool:
    if api_key is None:
        return True

    normalized = api_key.strip().strip("\"'")
    if not normalized:
        return True

    return normalized.startswith("your-") and normalized.endswith("-key")


def default_base_url_for_provider(provider: str) -> Optional[str]:
    return _DEFAULT_BASE_URL_BY_PROVIDER.get(canonical_provider_name(provider))


def resolve_base_url_for_provider(
    provider: str, explicit_base_url: Optional[str] = None
) -> Optional[str]:
    """Resolve a provider base URL using explicit value, scoped env, then default.

    Environment overrides are keyed by canonical provider, not by client class.
    This prevents providers that share a transport implementation from
    inheriting unrelated endpoint overrides.
    """
    if explicit_base_url:
        return explicit_base_url

    canonical = canonical_provider_name(provider)
    env_name = _BASE_URL_ENV_BY_PROVIDER.get(canonical)
    if env_name and (env_value := os.getenv(env_name)):
        return env_value

    return _DEFAULT_BASE_URL_BY_PROVIDER.get(canonical)


def curated_models_for_provider(provider: str) -> tuple[str, ...]:
    return _CURATED_MODELS_BY_PROVIDER.get(canonical_provider_name(provider), ())


def is_auto_router_model(provider: str, model_name: Optional[str]) -> bool:
    """True when an OpenRouter model is the virtual ``auto`` router.

    Such a model is served by xrouter-llm (in-process selection) instead of
    being sent to OpenRouter directly.
    """
    return (
        canonical_provider_name(provider) == "openrouter"
        and (model_name or "").strip().lower() == AUTO_MODEL_NAME
    )


def provider_compatibility_for_provider(provider: str) -> Optional[str]:
    provider_id = canonical_provider_name(provider)
    for provider_info in _SUPPORTED_PROVIDER_METADATA:
        if provider_info["id"] == provider_id:
            compatibility = provider_info.get("compatibility")
            return str(compatibility) if compatibility is not None else None
    return None


def get_supported_provider_metadata() -> list[dict[str, Any]]:
    providers: list[dict[str, Any]] = []
    for provider in _SUPPORTED_PROVIDER_METADATA:
        provider_info = dict(provider)
        default_base_url = default_base_url_for_provider(provider_info["id"])
        if default_base_url is not None:
            provider_info["default_base_url"] = default_base_url
        providers.append(provider_info)
    return providers
