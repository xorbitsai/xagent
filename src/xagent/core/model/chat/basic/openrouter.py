from typing import Any, Dict, List, Optional

from .....config import get_openrouter_official_providers_only
from ..timeout_config import TimeoutConfig
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

    def _disable_thinking_extra_body(
        self, extra_body: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        updated_extra_body = dict(extra_body or {})
        updated_extra_body["reasoning"] = {"enabled": False}
        updated_extra_body["thinking"] = {"type": "disabled"}
        updated_extra_body.pop("enable_thinking", None)
        return updated_extra_body
