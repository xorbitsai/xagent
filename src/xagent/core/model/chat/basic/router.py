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
import inspect
import logging
import os
import threading
from typing import Any, AsyncIterator, Callable, List, Optional, cast

from ....context_ref import CONTEXT_REFS_KEY, normalize_context_references
from ....model import ChatModelConfig
from ....task_runtime import normalize_input_modalities
from ...providers import default_base_url_for_provider
from ..types import StreamChunk
from .base import BaseLLM

logger = logging.getLogger(__name__)

_DEFAULT_ROUTER_ABILITIES = ["chat", "tool_calling"]
_UNROUTED_ROUTER_ABILITIES = {"vision", "thinking_mode"}
_DISABLE_DOWNSTREAM_THINKING = {"type": "disabled", "enable": False}
_CONTENT_PART_MODALITIES = {
    "audio": "audio",
    "audio_url": "audio",
    "file": "file",
    "file_url": "file",
    "image": "image",
    "image_url": "image",
    "input_audio": "audio",
    "input_file": "file",
    "input_image": "image",
    "input_video": "video",
    "video": "video",
    "video_url": "video",
}
_MODALITY_ABILITIES = {
    "audio": "audio",
    "image": "vision",
    "video": "video",
}


class RouterModalityRoutingError(RuntimeError):
    """The installed router cannot enforce required input modalities."""


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


def _should_retry_with_relaxed_tool_choice(
    exc: Exception,
    *,
    tools: list[dict[str, Any]] | None,
    tool_choice: str | dict[str, Any] | None,
) -> bool:
    if not tools or tool_choice in (None, "auto", "none"):
        return False

    # Deliberate OpenRouter compatibility bridge: official provider routing can
    # reject strict tool_choice values before selecting an endpoint. This
    # degrades forced tool use to "auto" instead of failing the whole agent run.
    # Replace string matching with typed provider errors when available.
    exc_msg = str(exc).lower()
    return "no endpoints found" in exc_msg and "tool_choice" in exc_msg


def _next_retry_state(
    exc: Exception,
    *,
    tools: list[dict[str, Any]] | None,
    thinking: dict[str, Any] | None,
    tool_choice: str | dict[str, Any] | None,
) -> tuple[str | dict[str, Any] | None, dict[str, Any] | None, str, str] | None:
    if _should_retry_without_thinking(exc, thinking=thinking, tool_choice=tool_choice):
        return (
            tool_choice,
            _DISABLE_DOWNSTREAM_THINKING,
            "selected model rejected thinking with tool_choice; retrying without thinking",
            "disable_thinking",
        )

    if _should_retry_with_relaxed_tool_choice(
        exc, tools=tools, tool_choice=tool_choice
    ):
        return (
            "auto",
            thinking,
            "selected OpenRouter endpoint rejected tool_choice; retrying with tool_choice=auto",
            "relax_tool_choice",
        )

    return None


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
        # A virtual router cannot advertise one candidate's dynamic abilities.
        # The resolved per-call wrapper derives those from the selected model's
        # profile after xrouter applies any input-modality preferences.
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

    async def _run_non_streaming_with_provider_retry(
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
        current_tool_choice = tool_choice
        current_thinking = thinking
        attempted_retry_actions: set[str] = set()

        while True:
            try:
                result = await method(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    tool_choice=current_tool_choice,
                    response_format=response_format,
                    thinking=current_thinking,
                    output_config=output_config,
                    **kwargs,
                )
                return cast(str | dict[str, Any], result)
            except Exception as exc:  # noqa: BLE001 - inspect a provider compatibility error.
                retry_state = _next_retry_state(
                    exc,
                    tools=tools,
                    thinking=current_thinking,
                    tool_choice=current_tool_choice,
                )
                if retry_state is None:
                    raise

                next_tool_choice, next_thinking, log_message, action_key = retry_state
                if action_key in attempted_retry_actions:
                    raise

                attempted_retry_actions.add(action_key)
                logger.info(log_message)
                current_tool_choice = next_tool_choice
                current_thinking = next_thinking

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
        prepared = await self.prepare_for_call(messages)
        return await prepared.chat(
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
        prepared = await self.prepare_for_call(messages)
        return await prepared.vision_chat(
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
        prepared = await self.prepare_for_call(messages)
        async for chunk in prepared.stream_chat(
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
            yield chunk

    async def _run_streaming_with_provider_retry(
        self,
        llm: BaseLLM,
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
    ) -> AsyncIterator[StreamChunk]:
        has_yielded = False
        current_tool_choice = tool_choice
        current_thinking = thinking
        attempted_retry_actions: set[str] = set()

        while True:
            try:
                async for chunk in llm.stream_chat(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    tool_choice=current_tool_choice,
                    response_format=response_format,
                    thinking=current_thinking,
                    output_config=output_config,
                    **kwargs,
                ):
                    has_yielded = True
                    yield chunk
                return
            except Exception as exc:  # noqa: BLE001 - inspect a provider compatibility error.
                if has_yielded:
                    raise

                retry_state = _next_retry_state(
                    exc,
                    tools=tools,
                    thinking=current_thinking,
                    tool_choice=current_tool_choice,
                )
                if retry_state is None:
                    raise

                next_tool_choice, next_thinking, log_message, action_key = retry_state
                if action_key in attempted_retry_actions:
                    raise

                attempted_retry_actions.add(action_key)
                logger.info(log_message)
                current_tool_choice = next_tool_choice
                current_thinking = next_thinking

    # ---- Routing ------------------------------------------------------------
    async def prepare_for_call(
        self,
        messages: list[dict[str, Any]],
        *,
        preferred_input_modalities: tuple[str, ...] = (),
    ) -> BaseLLM:
        """Resolve one xrouter decision into a reusable per-call LLM.

        The returned wrapper keeps RouterLLM's compatibility retries without
        routing a second time, and carries the selected model's context window
        from xrouter's model profile catalog.

        ``preferred_input_modalities`` supplied by the caller (task runtime
        extensions) is *advisory*: a router that cannot honour it degrades by
        routing without it. Modalities derived from the messages themselves are
        hard requirements, because the conversation genuinely cannot be sent to
        a model that does not accept them.
        """
        required_input_modalities = self._preferred_input_modalities(messages)
        advisory_input_modalities = tuple(
            modality
            for modality in dict.fromkeys(
                normalize_input_modalities(preferred_input_modalities)
            )
            if modality not in required_input_modalities
        )
        route_input_modalities = (
            *advisory_input_modalities,
            *required_input_modalities,
        )
        model_id, downstream = await self._resolve_route(
            messages,
            preferred_input_modalities=required_input_modalities,
            advisory_input_modalities=advisory_input_modalities,
        )
        context_window = getattr(self, "context_window", None)
        if not context_window:
            context_window = await asyncio.to_thread(
                self._profile_context_window, model_id
            )
        if not context_window:
            context_window = getattr(downstream, "context_window", None)
        input_modalities = (
            self._profile_input_modalities(model_id) if route_input_modalities else ()
        )
        return _ResolvedRouterLLM(
            router=self,
            downstream=downstream,
            selected_model=model_id,
            context_window=context_window,
            input_modalities=input_modalities,
        )

    async def _resolve_route(
        self,
        messages: list[dict[str, Any]],
        *,
        preferred_input_modalities: tuple[str, ...] = (),
        advisory_input_modalities: tuple[str, ...] = (),
    ) -> tuple[str, BaseLLM]:
        # Route on the agent's current goal (the user's request, or a DAG step's
        # objective) rather than the scaffolded sub-prompt this particular LLM
        # call happens to carry.
        from ...intent import current_goal

        prompt = current_goal() or self._extract_prompt(messages)
        select_kwargs: dict[str, Any] = {}
        if preferred_input_modalities:
            select_kwargs["preferred_input_modalities"] = preferred_input_modalities
        if advisory_input_modalities:
            select_kwargs["advisory_input_modalities"] = advisory_input_modalities
        model_id = await self._select_model(prompt, **select_kwargs)
        logger.info("xrouter selected %s -> openrouter", model_id)
        if self._downstream_resolver is not None:
            # Reuse the user-configured OpenRouter model (credentials + base_url).
            return model_id, self._downstream_resolver(model_id)
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
        return model_id, create_base_llm(config)

    @staticmethod
    def _profile_context_window(model_id: str) -> int | None:
        """Read context length from xrouter's already-loaded model catalog."""
        try:
            profile = _get_service().profiles.get(model_id)
            value = getattr(profile, "context_length", None)
        except Exception as exc:  # noqa: BLE001 - metadata is best effort
            logger.warning(
                "Could not resolve xrouter context window for %s: %s",
                model_id,
                exc,
            )
            return None
        return value if isinstance(value, int) and value > 0 else None

    @staticmethod
    def _profile_input_modalities(model_id: str) -> tuple[str, ...]:
        try:
            profile = _get_service().profiles.get(model_id)
            values = getattr(profile, "input_modalities", ())
        except Exception as exc:  # noqa: BLE001 - metadata is best effort
            logger.warning(
                "Could not resolve xrouter input modalities for %s: %s",
                model_id,
                exc,
            )
            return ()
        if not isinstance(values, (list, tuple, set, frozenset)):
            return ()
        return tuple(
            dict.fromkeys(
                str(value).strip().lower() for value in values if str(value).strip()
            )
        )

    async def _select_model(
        self,
        prompt: str,
        *,
        preferred_input_modalities: tuple[str, ...] = (),
        advisory_input_modalities: tuple[str, ...] = (),
    ) -> str:
        # The decision loads/embeds in-process and is CPU-bound, so run it in a
        # worker thread to avoid blocking the event loop.
        try:
            selected = await asyncio.to_thread(
                self._route_sync,
                prompt,
                preferred_input_modalities,
                advisory_input_modalities,
            )
        except RouterModalityRoutingError:
            raise
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

    def _route_sync(
        self,
        prompt: str,
        preferred_input_modalities: tuple[str, ...] = (),
        advisory_input_modalities: tuple[str, ...] = (),
    ) -> list[str]:
        """Route once.

        ``preferred_input_modalities`` are hard requirements derived from the
        conversation's own content; ``advisory_input_modalities`` are
        preferences declared by a task runtime extension. When the installed
        router cannot express modality preferences at all, the hard
        requirements raise while the advisory ones are simply dropped.
        """
        service = _get_service()
        route_kwargs: dict[str, Any] = {"config_name": self._config_name}
        try:
            route_parameters = dict(inspect.signature(service.route).parameters)
        except (TypeError, ValueError):
            route_parameters = {}
        supports_modality_preferences = (
            "preferred_input_modalities" in route_parameters
            or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in route_parameters.values()
            )
        )
        requested_modalities = tuple(
            dict.fromkeys((*advisory_input_modalities, *preferred_input_modalities))
        )
        if requested_modalities and supports_modality_preferences:
            route_kwargs["preferred_input_modalities"] = requested_modalities
        elif preferred_input_modalities:
            requested = ", ".join(preferred_input_modalities)
            raise RouterModalityRoutingError(
                "The installed xrouter-llm RoutingService cannot enforce input "
                f"modalities ({requested}). Choose an explicit compatible model or "
                "install an xrouter-llm build whose route() API accepts "
                "preferred_input_modalities."
            )
        elif advisory_input_modalities:
            # Advisory only: routing without the preference is a valid
            # degradation, unlike a conversation that actually carries the
            # unsupported modality.
            logger.info(
                "The installed xrouter-llm RoutingService cannot express input "
                "modality preferences (%s); routing without them.",
                ", ".join(advisory_input_modalities),
            )
        result = service.route(prompt, **route_kwargs)
        return list(result.get("selected") or [])

    @staticmethod
    def _preferred_input_modalities(
        messages: list[dict[str, Any]],
    ) -> tuple[str, ...]:
        modalities: set[str] = set()
        for message in messages:
            try:
                references = normalize_context_references(message.get(CONTEXT_REFS_KEY))
            except (TypeError, ValueError):
                references = ()
            modalities.update(reference.type for reference in references)

            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                modality = _CONTENT_PART_MODALITIES.get(
                    str(part.get("type") or "").lower()
                )
                if modality is not None:
                    modalities.add(modality)
        return tuple(sorted(modalities))

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


class _ResolvedRouterLLM(BaseLLM):
    """A concrete xrouter selection reused for one logical LLM call."""

    def __init__(
        self,
        *,
        router: RouterLLM,
        downstream: BaseLLM,
        selected_model: str,
        context_window: int | None,
        input_modalities: tuple[str, ...],
    ) -> None:
        self._router = router
        self._downstream = downstream
        self._selected_model = selected_model
        self.context_window = context_window
        abilities = list(router.abilities)
        for modality in input_modalities:
            ability = _MODALITY_ABILITIES.get(modality)
            if ability is not None and ability not in abilities:
                abilities.append(ability)
        self._abilities = abilities

    @property
    def model_id(self) -> str:
        return self._downstream.model_id

    @property
    def timeout(self) -> float:
        return self._router.timeout

    @property
    def abilities(self) -> List[str]:
        return self._abilities

    @property
    def model_name(self) -> str:
        return self._selected_model

    @property
    def supports_thinking_mode(self) -> bool:
        return self._router.supports_thinking_mode

    @property
    def supports_json_schema_response_format(self) -> bool:
        return self._downstream.supports_json_schema_response_format

    @property
    def supports_json_object_response_format(self) -> bool:
        return self._downstream.supports_json_object_response_format

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
        return await self._router._run_non_streaming_with_provider_retry(
            self._downstream.chat,
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
        return await self._router._run_non_streaming_with_provider_retry(
            self._downstream.vision_chat,
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
        async for chunk in self._router._run_streaming_with_provider_retry(
            self._downstream,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            thinking=thinking,
            output_config=output_config,
            kwargs=kwargs,
        ):
            yield chunk
