"""Router LLM: a virtual model that delegates to xrouter-llm for selection.

On every call it asks the xrouter-llm decision library (imported in-process, no
external service) to pick ONE concrete model for the prompt, then dispatches the
actual completion through a single OpenAI-compatible backend pointed at
OpenRouter. Every provider (Claude, DeepSeek, Gemini, GLM, GPT, ...) is reached
via OpenRouter, so xagent needs only ONE credential pair: `OPENAI_API_KEY` (an
OpenRouter key) and `OPENAI_BASE_URL` (https://openrouter.ai/api/v1).

xrouter-llm ships a trained router, the model-profile registry, and the named
router configs as package data, so the decision runs entirely in-process. The
registry returns ids that are already canonical OpenRouter slugs (e.g.
`anthropic/claude-opus-4.8`, `openai/gpt-5.5`), so the chosen id is passed
straight through as the downstream model name.

Every decision (prompt, candidate models with their predicted completion and
cost, and the chosen slug) is logged to a SQLite call history via xrouter-llm's
CallStore, defaulting to ``<storage_root>/xrouter/calls.db``.

Env overrides (all optional; default to the bundled package data):
  XAGENT_XROUTER_MODEL          path to a trained predictor .joblib
  XAGENT_XROUTER_MODELS_DIR     model-profile registry dir/file
  XAGENT_XROUTER_ROUTERS_DIR    router configs dir/file
  XAGENT_XROUTER_DB             routing-decision SQLite history path
  XAGENT_ROUTER_FALLBACK_MODEL  slug to use if routing fails
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any, AsyncIterator, Callable, List, Optional, cast

from ....model import ChatModelConfig
from ...providers import default_base_url_for_provider
from ..types import StreamChunk
from .base import BaseLLM

logger = logging.getLogger(__name__)

_DEFAULT_ROUTER_ABILITIES = ["chat", "tool_calling"]
_UNROUTED_ROUTER_ABILITIES = {"vision", "thinking_mode"}
_DISABLE_DOWNSTREAM_THINKING = {"type": "disabled", "enable": False}


def _should_retry_without_thinking(
    exc: Exception,
    *,
    thinking: dict[str, Any] | None,
    tool_choice: str | dict[str, Any] | None,
) -> bool:
    # Deliberate OpenRouter/DeepSeek compatibility bridge: the provider returns
    # an OpenAI-compatible 400 without a typed error for this thinking/tool_choice
    # conflict. Replace this with provider-owned typed exceptions once the
    # follow-up tracking issue lands.
    exc_msg = str(exc).lower()
    return (
        (thinking is None or isinstance(thinking, dict))
        and tool_choice is not None
        and "thinking" in exc_msg
        and "tool_choice" in exc_msg
    )


class _NullStore:
    """Duck-typed CallStore that drops the decision log (degradation fallback)."""

    def record(self, **_kwargs: Any) -> int:
        return 0


def _store_path() -> str:
    """SQLite path for the routing-decision history."""
    override = os.getenv("XAGENT_XROUTER_DB")
    if override:
        return override
    try:
        from xagent.config import get_storage_root

        return str(get_storage_root() / "xrouter" / "calls.db")
    except Exception:  # pragma: no cover - config unavailable
        return "xrouter_calls.db"


# A RoutingService loads a trained predictor plus a multilingual embedding model,
# which is expensive, so build it once per (model, registry, configs) tuple and
# share it across all RouterLLM instances.
_SERVICE_LOCK = threading.Lock()
_SERVICE_CACHE: dict[tuple[str, str, str], Any] = {}


def _build_service(model_path: str, models_dir: str, routers_dir: str) -> Any:
    try:
        import joblib
        from xrouter_llm import load_benchmark_profiles
        from xrouter_llm.serving import RoutingService, load_router_configs
        from xrouter_llm.store import CallStore
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise RuntimeError(
            "The 'router' (auto) provider needs the xrouter-llm package. "
            "Install it with `pip install 'xagent[router]'` (or `pip install xrouter-llm`)."
        ) from exc

    predictor = joblib.load(model_path)
    if not hasattr(predictor, "predict"):
        raise TypeError(f"{model_path} is not a fitted xrouter-llm predictor")
    profiles = load_benchmark_profiles(models_dir)
    configs = load_router_configs(routers_dir)
    try:
        store: Any = CallStore(_store_path())
    except Exception as exc:  # noqa: BLE001 - history must not break routing
        logger.warning("xrouter call history disabled (%s)", exc)
        store = _NullStore()
    return RoutingService(predictor, profiles=profiles, configs=configs, store=store)


def _get_service() -> Any:
    """Lazily build and cache the in-process routing service."""
    from xrouter_llm import (
        default_model_path,
        default_models_dir,
        default_routers_dir,
    )

    model_path = os.getenv("XAGENT_XROUTER_MODEL") or default_model_path()
    models_dir = os.getenv("XAGENT_XROUTER_MODELS_DIR") or default_models_dir()
    routers_dir = os.getenv("XAGENT_XROUTER_ROUTERS_DIR") or default_routers_dir()
    key = (model_path, models_dir, routers_dir)

    service = _SERVICE_CACHE.get(key)
    if service is not None:
        return service
    with _SERVICE_LOCK:
        service = _SERVICE_CACHE.get(key)
        if service is None:
            service = _build_service(*key)
            _SERVICE_CACHE[key] = service
        return service


class RouterLLM(BaseLLM):
    def __init__(
        self,
        model_name: str = "auto",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_temperature: Optional[float] = None,
        default_max_tokens: Optional[int] = None,
        timeout: float = 180.0,
        abilities: Optional[List[str]] = None,
        downstream_resolver: Optional[Callable[[str], BaseLLM]] = None,
    ) -> None:
        # model_name doubles as the xrouter-llm router config name (e.g. "auto").
        self._config_name = model_name or "auto"
        # Given a chosen OpenRouter slug, build the LLM that runs it. Injected by
        # the model store so "auto" reuses the user-configured OpenRouter model
        # (credentials + base_url) instead of any environment variable.
        self._downstream_resolver = downstream_resolver
        # The auto model's own OpenRouter credentials. Routing is in-process (not
        # an HTTP call), but these are used by the fallback resolver below when no
        # downstream OpenRouter model is injected (e.g. test-connection paths), so
        # the chosen slug still runs against the user's configured key/base_url.
        self._api_key = api_key
        self._base_url = base_url
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.timeout = timeout
        # xrouter-llm currently routes from text only and does not filter
        # candidates by multimodal or reasoning support.
        self._abilities = [
            ability
            for ability in (abilities or _DEFAULT_ROUTER_ABILITIES)
            if ability not in _UNROUTED_ROUTER_ABILITIES
        ]
        self._fallback_model = os.getenv("XAGENT_ROUTER_FALLBACK_MODEL") or None

    # ---- BaseLLM interface --------------------------------------------------
    @property
    def abilities(self) -> List[str]:
        return self._abilities

    @property
    def model_name(self) -> str:
        return self._config_name

    @property
    def supports_thinking_mode(self) -> bool:
        return "thinking_mode" in self._abilities

    async def _run_non_streaming_with_thinking_retry(
        self,
        method: Callable[..., Any],
        messages: list[dict[str, Any]],
        *,
        temperature: float | None,
        max_tokens: int | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        response_format: dict[str, Any] | None,
        thinking: dict[str, Any] | None,
        output_config: dict[str, Any] | None,
        kwargs: dict[str, Any],
    ) -> str | dict[str, Any]:
        try:
            result = await method(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                thinking=thinking,
                output_config=output_config,
                **kwargs,
            )
            return cast(str | dict[str, Any], result)
        except Exception as exc:  # noqa: BLE001 - inspect a provider compatibility error.
            if not _should_retry_without_thinking(
                exc, thinking=thinking, tool_choice=tool_choice
            ):
                raise
            logger.info(
                "selected model rejected thinking with tool_choice; retrying without thinking"
            )
            # Auto routing dispatches selected slugs through OpenRouterLLM
            # (injected resolver or fallback), whose disabled-thinking hook emits
            # the provider-specific reasoning payload.
            result = await method(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                thinking=_DISABLE_DOWNSTREAM_THINKING,
                output_config=output_config,
                **kwargs,
            )
            return cast(str | dict[str, Any], result)

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        thinking: dict[str, Any] | None = None,
        output_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str | dict[str, Any]:
        llm = await self._resolve(messages)
        return await self._run_non_streaming_with_thinking_retry(
            llm.chat,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            thinking=thinking,
            output_config=output_config,
            kwargs=kwargs,
        )

    async def vision_chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        thinking: dict[str, Any] | None = None,
        output_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str | dict[str, Any]:
        llm = await self._resolve(messages)
        return await self._run_non_streaming_with_thinking_retry(
            llm.vision_chat,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            thinking=thinking,
            output_config=output_config,
            kwargs=kwargs,
        )

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        thinking: dict[str, Any] | None = None,
        output_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        llm = await self._resolve(messages)
        has_yielded = False
        try:
            async for chunk in llm.stream_chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                thinking=thinking,
                output_config=output_config,
                **kwargs,
            ):
                has_yielded = True
                yield chunk
        except Exception as exc:  # noqa: BLE001 - inspect a provider compatibility error.
            if has_yielded or not _should_retry_without_thinking(
                exc, thinking=thinking, tool_choice=tool_choice
            ):
                raise
            logger.info(
                "selected model rejected thinking with tool_choice; retrying stream without thinking"
            )
            # See _run_non_streaming_with_thinking_retry: OpenRouterLLM owns the
            # provider-specific disabled-reasoning payload.
            async for chunk in llm.stream_chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                thinking=_DISABLE_DOWNSTREAM_THINKING,
                output_config=output_config,
                **kwargs,
            ):
                yield chunk

    # ---- Routing ------------------------------------------------------------
    async def _resolve(self, messages: list[dict[str, Any]]) -> BaseLLM:
        # Route on the agent's current goal (the user's request, or a DAG step's
        # objective) rather than the scaffolded sub-prompt this particular LLM
        # call happens to carry.
        from ...intent import current_goal

        prompt = current_goal() or self._extract_prompt(messages)
        model_id = await self._select_model(prompt)
        logger.info("xrouter selected %s -> openrouter", model_id)
        if self._downstream_resolver is not None:
            # Reuse the user-configured OpenRouter model (credentials + base_url).
            return self._downstream_resolver(model_id)
        # Fallback when no downstream resolver was injected: an OpenAI-compatible
        # client using this model's own OpenRouter credentials (or the ambient
        # OPENAI_BASE_URL / OPENAI_API_KEY env when those are unset).
        # Lazy import avoids a circular import (adapter imports this module).
        from .adapter import create_base_llm

        config = ChatModelConfig(
            id=f"router:{model_id}",
            model_name=model_id,
            model_provider="openrouter",
            base_url=self._base_url or default_base_url_for_provider("openrouter"),
            api_key=self._api_key,
            default_temperature=self.default_temperature,
            default_max_tokens=self.default_max_tokens,
            timeout=self.timeout,
            abilities=self._abilities,
        )
        return create_base_llm(config)

    async def _select_model(self, prompt: str) -> str:
        # The decision loads/embeds in-process and is CPU-bound, so run it in a
        # worker thread to avoid blocking the event loop.
        try:
            selected = await asyncio.to_thread(self._route_sync, prompt)
        except Exception as exc:  # noqa: BLE001 - routing must not crash the agent
            if self._fallback_model:
                logger.warning(
                    "xrouter route failed (%s); using fallback %s",
                    exc,
                    self._fallback_model,
                )
                return self._fallback_model
            raise RuntimeError(
                f"xrouter-llm routing failed: {exc}. "
                "Set XAGENT_ROUTER_FALLBACK_MODEL to degrade gracefully."
            ) from exc
        if not selected:
            if self._fallback_model:
                return self._fallback_model
            raise RuntimeError("xrouter-llm returned no selected model")
        return str(selected[0])

    def _route_sync(self, prompt: str) -> list[str]:
        service = _get_service()
        result = service.route(prompt, config_name=self._config_name)
        return list(result.get("selected") or [])

    @staticmethod
    def _extract_prompt(messages: list[dict[str, Any]]) -> str:
        """Use the latest user message as the routing prompt."""
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = [
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                if texts:
                    return "\n".join(texts)
        # Fallback: concatenate any string content.
        return "\n".join(
            m["content"] for m in messages if isinstance(m.get("content"), str)
        )
