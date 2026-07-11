from typing import Any, AsyncIterator, Dict, List, Optional

from .....config import get_openrouter_official_providers_only
from ..timeout_config import TimeoutConfig
from ..types import StreamChunk
from .deepseek_tool_protocol import (
    adapt_deepseek_stream,
    normalize_deepseek_response,
)
from .openai import OpenAILLM

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_OPENROUTER_OFFICIAL_PROVIDERS_BY_AUTHOR: dict[str, tuple[str, ...]] = {
    "anthropic": ("anthropic",),
    "deepseek": ("deepseek",),
    "google": ("google-ai-studio", "google-vertex"),
    "minimax": ("minimax",),
    "openai": ("openai",),
    "z-ai": ("z-ai",),
}


def _openrouter_model_author(model_name: str) -> str:
    model_slug = model_name.strip().split(":", 1)[0]
    parts = [part for part in model_slug.split("/") if part]
    if len(parts) >= 3 and parts[0].lower() == "openrouter":
        return parts[1].lower()
    if len(parts) >= 2:
        return parts[0].lower()
    return ""


class OpenRouterLLM(OpenAILLM):
    """OpenRouter client using the OpenAI SDK with OpenRouter-specific options."""

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        default_temperature: Optional[float] = None,
        default_max_tokens: Optional[int] = None,
        timeout: float = 180.0,
        abilities: Optional[List[str]] = None,
        timeout_config: Optional[TimeoutConfig] = None,
    ):
        super().__init__(
            model_name=model_name,
            base_url=base_url or OPENROUTER_BASE_URL,
            api_key=api_key,
            default_temperature=default_temperature,
            default_max_tokens=default_max_tokens,
            timeout=timeout,
            abilities=abilities,
            timeout_config=timeout_config,
        )

    @property
    def _uses_deepseek_tool_protocol(self) -> bool:
        return _openrouter_model_author(self._model_name) == "deepseek"

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str | Dict[str, Any]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        thinking: Optional[Dict[str, Any]] = None,
        output_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        response = await super().chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            thinking=thinking,
            output_config=output_config,
            **kwargs,
        )
        if not self._uses_deepseek_tool_protocol:
            return response
        return normalize_deepseek_response(response, tools=tools)

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str | Dict[str, Any]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        thinking: Optional[Dict[str, Any]] = None,
        output_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        stream = super().stream_chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            thinking=thinking,
            output_config=output_config,
            **kwargs,
        )
        if not self._uses_deepseek_tool_protocol:
            async for chunk in stream:
                yield chunk
            return
        async for chunk in adapt_deepseek_stream(stream, tools=tools):
            yield chunk

    def _is_official_openrouter_client(self) -> bool:
        return self.base_url.rstrip("/") == OPENROUTER_BASE_URL

    def _prepare_extra_body(self, extra_body: Dict[str, Any]) -> Dict[str, Any]:
        if (
            not get_openrouter_official_providers_only()
            or not self._is_official_openrouter_client()
            or "provider" in extra_body
        ):
            return extra_body

        author = _openrouter_model_author(self._model_name)
        official_providers = _OPENROUTER_OFFICIAL_PROVIDERS_BY_AUTHOR.get(author)
        if not official_providers:
            return extra_body

        return {
            **extra_body,
            "provider": {
                "only": list(official_providers),
                "allow_fallbacks": False,
                "require_parameters": True,
            },
        }

    def _prepare_provider_reasoning_extra_body(
        self,
        *,
        extra_body: Dict[str, Any],
        thinking: Optional[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        response_format: Optional[Dict[str, Any]],
        output_config: Optional[Dict[str, Any]],
        is_streaming: bool,
    ) -> Dict[str, Any]:
        _ = tools, output_config
        updated_extra_body = dict(extra_body)
        should_disable = False

        if thinking is not None:
            should_enable = thinking.get("type") == "enabled" or thinking.get(
                "enable", False
            )
            should_disable = not should_enable and (
                thinking.get("type") == "disabled" or not thinking.get("enable", False)
            )
        elif is_streaming and response_format:
            should_disable = self.supports_thinking_mode
            should_enable = False
        else:
            should_enable = False

        if should_disable:
            updated_extra_body["reasoning"] = {"enabled": False}
            updated_extra_body["thinking"] = {"type": "disabled"}
        elif should_enable:
            updated_extra_body["reasoning"] = {"enabled": True}
            updated_extra_body["thinking"] = {"type": "enabled"}

        updated_extra_body.pop("enable_thinking", None)
        return updated_extra_body
