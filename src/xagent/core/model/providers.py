import copy
import os
import re
from urllib.parse import urlsplit
from typing import Any, Mapping, Optional

# When an OpenRouter or configured router model carries this name, route the
# prompt through xrouter-llm (in-process) instead of calling a provider directly.
AUTO_MODEL_NAME = "auto"
ROUTER_PROVIDER = "router"

# Endpoint kinds for providers' official endpoints. ``official`` resolves from
# the registry defaults (and narrow env overrides); ``azure_resource`` is
# constructed from a validated Azure resource name -- still an official
# Microsoft endpoint, with no URL ever accepted as configuration input.
ENDPOINT_KIND_OFFICIAL = "official"
ENDPOINT_KIND_AZURE_RESOURCE = "azure_resource"

# Azure resource names: 2-64 lowercase alphanumerics and hyphens, not
# starting or ending with a hyphen (Microsoft naming rules for the resource).
_AZURE_RESOURCE_NAME_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])$")
AZURE_OPENAI_ENDPOINT_SUFFIX = ".openai.azure.com"

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
    "elevenlabs-sound_effect": "https://api.elevenlabs.io",
    "elevenlabs-music": "https://api.elevenlabs.io",
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
        "deepseek-flash",
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

# Model ids a provider documents as its own server-side multi-model entry
# points, so none of them names one fixed model: routers pick a (possibly
# different) underlying model for every request, and Fusion may answer through
# a panel of models plus an analyst model (whenever the outer model decides the
# task warrants deliberation). Source: OpenRouter's "Routers" documentation --
# Auto Router ``openrouter/auto`` and its ``openrouter/auto-beta`` track, the
# Free Models Router, the Pareto Router, and Fusion ``openrouter/fusion`` with
# its ``openrouter/fusion-flash`` preset -- checked 2026-09-25.
_ROUTING_MODEL_IDS_BY_PROVIDER: dict[str, frozenset[str]] = {
    "openrouter": frozenset(
        {
            "openrouter/auto",
            "openrouter/auto-beta",
            "openrouter/free",
            "openrouter/fusion",
            "openrouter/fusion-flash",
            "openrouter/pareto-code",
        }
    ),
}

_SUPPORTED_PROVIDER_METADATA: tuple[dict[str, Any], ...] = (
    {
        "id": "openai",
        "name": "OpenAI",
        "description": "OpenAI models",
        "requires_base_url": False,
        "compatibility": "openai_compatible",
        "category": ["llm", "embedding"],
    },
    {
        "id": "claude",
        "name": "Anthropic Claude",
        "description": "Anthropic's Claude models",
        "requires_base_url": False,
        "compatibility": "claude_compatible",
        "category": ["llm"],
    },
    {
        "id": "bedrock",
        "name": "Amazon Bedrock",
        "description": "Amazon Bedrock models through Converse and ConverseStream",
        "requires_base_url": False,
        "supports_model_listing": False,
        "category": ["llm"],
        "credential_fields": [
            {
                "name": "bedrock_region",
                "label": "AWS region",
                "kind": "plain",
                "required": True,
            },
            {
                "name": "bedrock_auth_mode",
                "label": "Authentication mode",
                "kind": "plain",
                "required": True,
            },
            {
                "name": "api_key",
                "label": "Amazon Bedrock API key",
                "kind": "secret",
                "required": False,
            },
        ],
    },
    {
        "id": "gemini",
        "name": "Google Gemini",
        "description": "Google's Gemini models",
        "requires_base_url": False,
        "category": ["llm", "image"],
    },
    {
        "id": "xinference",
        "name": "Xinference",
        "description": "Xinference models for local inference",
        "requires_base_url": True,
        "category": ["llm", "embedding", "image", "video", "speech", "rerank"],
        # The server checks a key only when its authentication is enabled.
        "credential_fields": [
            {"name": "api_key", "label": "API key", "kind": "secret", "required": False}
        ],
    },
    {
        "id": "elevenlabs",
        "name": "ElevenLabs",
        "description": "ElevenLabs models for speech, sound effects, and music",
        "requires_base_url": False,
        "category": ["speech", "sound_effect", "music"],
    },
    {
        "id": "deepseek",
        "name": "DeepSeek",
        "description": "DeepSeek v4 models with tool calling and thinking mode",
        "requires_base_url": False,
        "category": ["llm"],
    },
    {
        "id": "openai-compatible",
        "name": "OpenAI-Compatible",
        "description": "OpenAI-compatible models",
        "requires_base_url": True,
        "compatibility": "openai_compatible",
        "category": ["llm", "embedding"],
        # Keeps the required default key: the OpenAI adapter fills a missing
        # key from OPENAI_API_KEY, and recent openai SDKs refuse an empty one.
    },
    {
        "id": "azure_openai",
        "name": "Azure OpenAI",
        "description": (
            "Azure OpenAI Service: the endpoint is built from your resource "
            "name on the official Microsoft domain"
        ),
        # The endpoint is the user's own official Azure resource endpoint,
        # constructed from a validated resource identifier -- never a URL the
        # caller types in, so ``requires_base_url`` stays False. Hosts still
        # store it in ``ModelConfig.base_url``, which the chat adapters pass
        # on as ``azure_endpoint`` (see official_endpoint_for_provider).
        "requires_base_url": False,
        "category": ["llm"],
        "endpoint_kind": ENDPOINT_KIND_AZURE_RESOURCE,
        "credential_fields": [
            {
                "name": "resource_name",
                "label": "Resource name",
                "kind": "plain",
                "required": True,
            },
            {"name": "api_key", "label": "API key", "kind": "secret", "required": True},
        ],
    },
    {
        "id": "openrouter",
        "name": "OpenRouter",
        "description": (
            "OpenRouter aggregator: reach Claude, Gemini, GPT, DeepSeek, GLM, "
            "and more through one OpenAI-compatible key."
        ),
        # How to select a routing model (see is_routing_model), kept apart from
        # the description so hosts that accept only single models can omit it.
        "routing_hint": (
            "Use model 'auto' to let xrouter-llm pick the cheapest capable "
            "model per prompt."
        ),
        "requires_base_url": False,
        "compatibility": "openai_compatible",
        "category": ["llm"],
    },
    {
        "id": "dashscope",
        "name": "DashScope",
        "description": "Alibaba Cloud's DashScope models",
        "requires_base_url": False,
        "compatibility": "openai_compatible",
        "category": ["llm", "embedding", "image", "rerank"],
    },
    {
        "id": "volcengine-ark",
        "name": "Volcengine Ark",
        "description": "Volcengine ModelArk provider for Seedance video generation",
        "requires_base_url": False,
        "default_base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "category": ["video"],
    },
    {
        "id": "byteplus-ark",
        "name": "BytePlus Ark",
        "description": "BytePlus ModelArk provider for Seedance video generation",
        "requires_base_url": False,
        "default_base_url": "https://ark.ap-southeast.bytepluses.com/api/v3",
        "category": ["video"],
    },
    {
        "id": "alibaba-coding-plan",
        "name": "Alibaba Coding Plan",
        "description": "Alibaba Bailian (Model Studio) coding plan",
        "requires_base_url": False,
        "compatibility": "openai_compatible",
        "category": ["llm"],
    },
    {
        "id": "alibaba-coding-plan-cn",
        "name": "Alibaba Coding Plan (China)",
        "description": "Alibaba Bailian (Model Studio) coding plan (China)",
        "requires_base_url": False,
        "compatibility": "openai_compatible",
        "category": ["llm"],
    },
    {
        "id": "zhipu",
        "name": "Zhipu AI",
        "description": "Zhipu AI models (GLM series) using zai SDK",
        "requires_base_url": False,
        "category": ["llm"],
    },
    {
        "id": "zai-coding-plan",
        "name": "Z.AI Coding Plan",
        "description": "GLM coding plan via Z.AI",
        "requires_base_url": False,
        "compatibility": "openai_compatible",
        "category": ["llm"],
    },
    {
        "id": "zhipuai-coding-plan",
        "name": "Zhipu AI Coding Plan",
        "description": "GLM coding plan via Zhipu AI",
        "requires_base_url": False,
        "compatibility": "openai_compatible",
        "category": ["llm"],
    },
    {
        "id": "minimax-coding-plan",
        "name": "MiniMax Coding Plan (International)",
        "description": "MiniMax coding plan via api.minimax.io",
        "requires_base_url": False,
        "compatibility": "claude_compatible",
        "default_base_url": "https://api.minimax.io/anthropic",
        "category": ["llm"],
    },
    {
        "id": "minimax-cn-coding-plan",
        "name": "MiniMax Coding Plan (China)",
        "description": "MiniMax coding plan via api.minimaxi.com",
        "requires_base_url": False,
        "compatibility": "claude_compatible",
        "default_base_url": "https://api.minimaxi.com/anthropic",
        "category": ["llm"],
    },
    {
        "id": "kimi-for-coding",
        "name": "Kimi For Coding",
        "description": "Kimi coding endpoint",
        "requires_base_url": False,
        "compatibility": "claude_compatible",
        "default_base_url": "https://api.kimi.com/coding",
        "category": ["llm"],
    },
)


def _normalize_provider(provider: str) -> str:
    return provider.lower().strip()


def canonical_provider_name(provider: str) -> str:
    normalized = _normalize_provider(provider)
    return _PROVIDER_ALIASES.get(normalized, normalized)


_AWS_REGION_PATTERN = re.compile(r"^[a-z]{2}(?:-[a-z0-9]+)+-\d+$")


def validate_bedrock_settings(
    *,
    region: Optional[str],
    auth_mode: str,
    api_key: Optional[str],
    endpoint_url: Optional[str],
) -> None:
    """Validate the persisted settings for the native Bedrock runtime."""

    normalized_region = (region or "").strip()
    if not normalized_region or not _AWS_REGION_PATTERN.fullmatch(normalized_region):
        raise ValueError("A valid AWS region is required for Amazon Bedrock")
    if auth_mode not in {"api_key", "credentials_chain"}:
        raise ValueError("Unsupported Amazon Bedrock authentication mode")
    if auth_mode == "api_key" and is_placeholder_api_key(api_key):
        raise ValueError(
            "An Amazon Bedrock API key is required for api_key authentication"
        )
    if auth_mode == "credentials_chain" and not is_placeholder_api_key(api_key):
        raise ValueError(
            "Amazon Bedrock credentials_chain authentication cannot include an API key"
        )

    if not endpoint_url:
        return
    parsed = urlsplit(endpoint_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("The Amazon Bedrock runtime endpoint must be an HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "The Amazon Bedrock runtime endpoint contains unsupported URL components"
        )
    if parsed.hostname.lower().startswith("bedrock-mantle."):
        raise ValueError(
            "bedrock-mantle endpoints do not support Converse; use a bedrock-runtime endpoint"
        )


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
    """True when a model is the virtual ``auto`` router.

    ``openrouter/auto`` is the legacy single-credential form. ``router/auto``
    resolves every selected routing profile to a concrete saved model config.
    """
    return (
        canonical_provider_name(provider) in {"openrouter", ROUTER_PROVIDER}
        and (model_name or "").strip().lower() == AUTO_MODEL_NAME
    )


def is_routing_model(provider: str, model_name: Optional[str]) -> bool:
    """True when a model id selects among or combines models instead of naming one.

    Covers the in-process virtual router (:func:`is_auto_router_model`) and
    the server-side multi-model entry points a provider documents (routers
    and Fusion). A variant suffix (``openrouter/auto:online``,
    ``openrouter/fusion:free``) still names the same entry point. Ordinary
    ids, including a provider's stable aliases for one model, are not
    routing models.

    The server-side ids come from a static list, so any id not on it is
    treated as a single model -- including a router the provider adds after
    the list was last checked. Callers that must refuse routers rely on the
    list being kept current.
    """
    if is_auto_router_model(provider, model_name):
        return True
    routers = _ROUTING_MODEL_IDS_BY_PROVIDER.get(canonical_provider_name(provider))
    if not routers:
        return False
    base_name = (model_name or "").strip().lower().split(":", 1)[0]
    return base_name in routers


def provider_compatibility_for_provider(provider: str) -> Optional[str]:
    provider_id = canonical_provider_name(provider)
    for provider_info in _SUPPORTED_PROVIDER_METADATA:
        if provider_info["id"] == provider_id:
            compatibility = provider_info.get("compatibility")
            return str(compatibility) if compatibility is not None else None
    return None


def provider_requires_base_url(provider: str) -> bool:
    """True when the provider's metadata marks base_url as mandatory.

    Unregistered providers (including provider+category combos like
    "xinference-rerank") default to False rather than raising, since
    callers already validate the provider exists via PROVIDER_FETCHERS.
    """
    provider_id = canonical_provider_name(provider)
    for provider_info in _SUPPORTED_PROVIDER_METADATA:
        if provider_info["id"] == provider_id:
            return bool(provider_info.get("requires_base_url", False))
    return False


def get_supported_provider_metadata() -> list[dict[str, Any]]:
    """Every registered provider, with its registry default base URL added
    when it has one.

    Each entry is a deep copy, nested lists and dicts included, so callers
    may edit what they get back (for example to localize labels) without
    changing the registry.
    """
    providers: list[dict[str, Any]] = []
    for provider in _SUPPORTED_PROVIDER_METADATA:
        provider_info = copy.deepcopy(provider)
        default_base_url = default_base_url_for_provider(provider_info["id"])
        if default_base_url is not None:
            provider_info["default_base_url"] = default_base_url
        providers.append(provider_info)
    return providers


def azure_resource_endpoint_for_resource_name(resource_name: str) -> str:
    """Construct a resource's official Azure OpenAI endpoint from its name.

    Callers supply only the resource identifier; the official Microsoft
    domain is fixed here, so no URL (and no other host) can enter provider
    configuration through this path.
    """
    if not isinstance(resource_name, str):
        raise ValueError("resource_name must be a string")
    normalized = resource_name.strip().lower()
    if _AZURE_RESOURCE_NAME_PATTERN.fullmatch(normalized) is None:
        raise ValueError(
            "resource_name must be 2-64 lowercase letters, digits, or hyphens, "
            "not starting or ending with a hyphen"
        )
    return f"https://{normalized}{AZURE_OPENAI_ENDPOINT_SUFFIX}"


def provider_endpoint_kind(provider: str) -> str:
    """The endpoint kind declared for a provider, defaulting to ``official``."""
    provider_id = canonical_provider_name(provider)
    for provider_info in _SUPPORTED_PROVIDER_METADATA:
        if provider_info["id"] == provider_id:
            kind = provider_info.get("endpoint_kind")
            return str(kind) if kind is not None else ENDPOINT_KIND_OFFICIAL
    return ENDPOINT_KIND_OFFICIAL


_DEFAULT_CREDENTIAL_FIELDS: list[dict[str, Any]] = [
    {"name": "api_key", "label": "API key", "kind": "secret", "required": True}
]


def provider_credential_fields(provider: str) -> list[dict[str, Any]]:
    """The credential fields a provider declares, with the single-key default.

    Each field is ``{"name", "label", "kind" ("plain" | "secret"),
    "required"}``. Providers with more than a bare API key (for example a
    resource identifier), or whose key is optional, declare the full list on
    their metadata entry; every other provider, and any unregistered id,
    gets one required API key rather than being forced through a
    per-provider branch.

    ``required: False`` means the provider's server can run without the
    value, not that a missing value stays missing: an adapter may fill it
    from the environment (the Xinference rerank adapter reads
    ``XINFERENCE_API_KEY`` when the key is missing or empty).
    """
    provider_id = canonical_provider_name(provider)
    for provider_info in _SUPPORTED_PROVIDER_METADATA:
        if provider_info["id"] == provider_id:
            fields = provider_info.get("credential_fields")
            if fields:
                return [dict(field) for field in fields]
            return [dict(field) for field in _DEFAULT_CREDENTIAL_FIELDS]
    return [dict(field) for field in _DEFAULT_CREDENTIAL_FIELDS]


def official_endpoint_for_provider(
    provider: str, credentials: Optional[Mapping[str, str]] = None
) -> Optional[str]:
    """Resolve a provider's official endpoint for model construction.

    Registry-defaulted endpoints come from :func:`resolve_base_url_for_provider`;
    identifier-derived official endpoints (Azure resources) are constructed
    and validated here. Raises ValueError when a declared identifier field is
    missing or malformed, so a bad value fails before any network call.

    Callers store the result in ``ModelConfig.base_url``: model construction
    takes the endpoint from there. For Azure this is not optional even though
    ``requires_base_url`` is False (no URL is typed in, but one is still
    needed): the chat adapters pass ``base_url`` on as ``azure_endpoint``,
    and when it is unset they fall back to ``AZURE_OPENAI_ENDPOINT``
    (``AzureOpenAILLM`` also to ``OPENAI_API_BASE``) or fail to construct.

    ``None`` means the registry has no endpoint to give, in one of three
    cases a caller tells apart from the registry itself:

    - a registered provider without a default (``claude``, ``gemini``): its
      SDK resolves the endpoint, from the SDK's own base-URL environment
      variable (such as ``ANTHROPIC_BASE_URL`` or ``GOOGLE_GEMINI_BASE_URL``)
      when set, else its default;
    - a provider whose metadata sets ``requires_base_url``
      (``openai-compatible``, ``xinference``): there is no official
      endpoint, and the base URL has to come from the caller;
    - an unregistered id.

    Providers with a scoped ``*_BASE_URL`` environment override (see
    :func:`resolve_base_url_for_provider`) return the override when it is
    set, so the result is the deployment's configured endpoint, not always
    the official one. Deployments that must reach only official endpoints
    leave those variables, and the SDKs' own base-URL variables, unset.
    """
    if provider_endpoint_kind(provider) == ENDPOINT_KIND_AZURE_RESOURCE:
        resource_name = (credentials or {}).get("resource_name")
        if resource_name is None:
            raise ValueError("resource_name is required for this provider")
        return azure_resource_endpoint_for_resource_name(resource_name)
    return resolve_base_url_for_provider(provider)
