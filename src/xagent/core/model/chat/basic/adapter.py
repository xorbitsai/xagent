import os
from typing import Callable, Optional, overload

from ....model import ChatModelConfig, ModelConfig
from ....retry import chat_retry_budget, create_retry_wrapper
from ...providers import (
    AUTO_MODEL_NAME,
    ROUTER_PROVIDER,
    canonical_provider_name,
    is_auto_router_model,
    provider_compatibility_for_provider,
    resolve_base_url_for_provider,
)
from ..error import retry_on
from .azure_openai import AzureOpenAILLM
from .base import BaseLLM
from .claude import ClaudeLLM
from .dashscope import DashScopeLLM
from .deepseek import DeepSeekLLM
from .gemini import GeminiLLM
from .litellm import LiteLLM
from .openai import OpenAILLM
from .openrouter import OpenRouterLLM
from .router import RouterLLM
from .xinference import XinferenceLLM
from .zhipu import ZhipuLLM


@overload
def attach_chat_retry_wrapper(
    llm: BaseLLM, max_retries: Optional[int] = ...
) -> BaseLLM: ...


@overload
def attach_chat_retry_wrapper(llm: None, max_retries: Optional[int] = ...) -> None: ...


def attach_chat_retry_wrapper(
    llm: Optional[BaseLLM], max_retries: Optional[int] = None
) -> Optional[BaseLLM]:
    """Install the shared retry layer on a chat model.

    ``create_base_llm`` is not the only way a chat model gets built: the
    environment and default fallbacks construct one directly. Those paths used
    to lean on the provider SDK's own retry budget, which is now zero so that
    retry policy lives in one layer -- this is that layer, and every
    construction path has to reach it or it has no retries at all.

    ``max_retries`` defaults to what a model row without an explicit value
    would carry, read off the config model so the two cannot drift.
    """
    # The env and default factories are all ``Optional``-returning, and this
    # sits on their ``return`` statements: pass a missing model through rather
    # than handing callers a retry proxy wrapped around nothing.
    if llm is None:
        return None
    if max_retries is None:
        max_retries = int(ModelConfig.model_fields["max_retries"].default)
    return create_retry_wrapper(
        llm,
        BaseLLM,  # type: ignore[type-abstract]
        retry_methods={"chat", "vision_chat", "stream_chat"},
        max_retries=max_retries,
        retry_on=retry_on,
        # ``max_retries`` alone cannot bound how long one call holds an
        # execution slot, because each attempt may consume a full request
        # timeout. The budget adds the wall-clock ceiling and the short
        # capacity-refusal budget on top of it.
        budget=chat_retry_budget(),
    )


def create_base_llm(
    model: ModelConfig,
    downstream_resolver: Optional[Callable[[str], BaseLLM]] = None,
) -> BaseLLM:
    """
    Creates a custom BaseLLM instance from a ModelConfig.

    ``downstream_resolver`` is used by virtual ``auto`` models: given a chosen
    routing profile it returns the concrete configured LLM that runs it.
    """
    if not isinstance(model, ChatModelConfig):
        raise TypeError(f"Invalid model type: {type(model).__name__}")

    provider = canonical_provider_name(model.model_provider)
    compatibility = provider_compatibility_for_provider(provider)
    llm: BaseLLM

    if is_auto_router_model(provider, model.model_name):
        # Pick a concrete model via xrouter-llm. Legacy OpenRouter Auto clones
        # its own provider config; configured Auto injects saved model bindings.
        router = RouterLLM(
            model_name=model.router_config_name or AUTO_MODEL_NAME,
            api_key=model.api_key,
            base_url=model.base_url,
            default_temperature=model.default_temperature,
            default_max_tokens=model.default_max_tokens,
            timeout=model.timeout,
            abilities=model.abilities,
            downstream_resolver=downstream_resolver,
            candidate_models=model.router_candidate_models,
            fallback_model=model.router_fallback_model,
            use_environment_fallback=provider != ROUTER_PROVIDER,
        )
        router.context_window = model.context_window
        # No _model_id stamp here: RouterLLM delegates every call to a resolved
        # downstream LLM (which carries its own id), so a value set here would
        # never reach token-usage details.
        return router
    elif provider == "deepseek":
        llm = DeepSeekLLM(
            model_name=model.model_name,
            api_key=model.api_key,
            base_url=resolve_base_url_for_provider(provider, model.base_url),
            default_temperature=model.default_temperature,
            default_max_tokens=model.default_max_tokens,
            timeout=model.timeout,
            abilities=model.abilities,
        )
    elif provider == "openrouter":
        llm = OpenRouterLLM(
            model_name=model.model_name,
            api_key=model.api_key,
            base_url=model.base_url,
            default_temperature=model.default_temperature,
            default_max_tokens=model.default_max_tokens,
            timeout=model.timeout,
            abilities=model.abilities,
        )
    elif provider in {
        "dashscope",
        "alibaba-coding-plan",
        "alibaba-coding-plan-cn",
    }:
        llm = DashScopeLLM(
            model_name=model.model_name,
            api_key=model.api_key,
            base_url=resolve_base_url_for_provider(provider, model.base_url),
            default_temperature=model.default_temperature,
            default_max_tokens=model.default_max_tokens,
            timeout=model.timeout,
            abilities=model.abilities,
        )
    elif provider == "openai" or compatibility == "openai_compatible":
        llm = OpenAILLM(
            model_name=model.model_name,
            api_key=model.api_key,
            base_url=model.base_url,
            default_temperature=model.default_temperature,
            default_max_tokens=model.default_max_tokens,
            timeout=model.timeout,
            abilities=model.abilities,
        )
    elif provider == "claude" or compatibility == "claude_compatible":
        llm = ClaudeLLM(
            model_name=model.model_name,
            api_key=model.api_key,
            base_url=model.base_url,
            default_temperature=model.default_temperature,
            default_max_tokens=model.default_max_tokens,
            timeout=model.timeout,
            abilities=model.abilities,
        )
    elif provider == "azure_openai":
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
    elif provider == "zhipu":
        llm = ZhipuLLM(
            model_name=model.model_name,
            api_key=model.api_key,
            base_url=model.base_url,
            default_temperature=model.default_temperature,
            default_max_tokens=model.default_max_tokens,
            timeout=model.timeout,
            abilities=model.abilities,
        )
    elif provider == "gemini":
        llm = GeminiLLM(
            model_name=model.model_name,
            api_key=model.api_key,
            base_url=model.base_url,
            default_temperature=model.default_temperature,
            default_max_tokens=model.default_max_tokens,
            timeout=model.timeout,
            abilities=model.abilities,
        )
    elif provider == "litellm":
        llm = LiteLLM(
            model_name=model.model_name,
            api_key=model.api_key,
            api_base=model.base_url,
            default_temperature=model.default_temperature,
            default_max_tokens=model.default_max_tokens,
            timeout=model.timeout,
            abilities=model.abilities,
        )
    elif provider == "xinference":
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

    llm.context_window = model.context_window
    # Stamp the unique model id so token-usage details can disambiguate models
    # that share a model_name (e.g. a platform model vs a user's own).
    llm._model_id = model.id
    return attach_chat_retry_wrapper(llm, model.max_retries)
