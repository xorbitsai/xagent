"""Router LLM: a virtual model that delegates to xrouter-llm for selection.

On every call it asks the xrouter-llm decision library (imported in-process;
only the prompt embedding may go to a remote service, see
XROUTER_EMBEDDING_BACKEND below) to pick ONE routing profile for the prompt,
then dispatches the actual completion through the concrete saved model bound to
that profile.
The legacy OpenRouter ``auto`` model remains supported and resolves every
profile through its own OpenRouter credential pair.

xrouter-llm ships a trained router, the model-profile registry, and the named
router configs as package data, so the decision runs in-process except for the
prompt embedding when a remote backend is selected. The registry returns
canonical ids (e.g. `anthropic/claude-opus-4.8`,
`openai/gpt-5.5`) which configured Auto maps to concrete model records.

When the Auto model has a fallback, a call whose selected model fails with an
availability failure (provider down, timeout, rate limit, quota, rejected
credential, missing model) is sent once more to the fallback, and -- except
after a rate limit -- that configured candidate is skipped in favour of the
fallback for a cooldown (``XAGENT_AUTO_CANDIDATE_COOLDOWN_SECONDS``, per
process). A stream fails over
only before its first chunk. A failure the request itself caused (context
length, an invalid request, a tool-protocol violation, a refusal that names
the output budget) never fails over.

A configured fallback is held in reserve: while the Auto model has other
candidates, routing chooses only between those, and the fallback serves a call
only when routing fails, the selected candidate is unavailable, or no other
candidate left after ``XAGENT_XROUTER_EXCLUDED_MODELS`` accepts the request's
required input modalities. Routing itself is bounded to the candidates that
accept those modalities; advisory modalities remain a preference within them.

Every decision (prompt, candidate models with their predicted completion and
cost, and the chosen slug) is logged to a SQLite call history via xrouter-llm's
CallStore, defaulting to ``<storage_root>/xrouter/calls.db``. A configured Auto
model with a single eligible candidate besides its fallback makes no decision
and logs nothing.

Env overrides (all optional; default to the bundled package data):
  XAGENT_XROUTER_MODEL          path to a trained predictor .joblib
  XAGENT_XROUTER_MODELS_DIR     model-profile registry dir/file
  XAGENT_XROUTER_ROUTERS_DIR    router configs dir/file
  XAGENT_XROUTER_DB             routing-decision SQLite history path
  XAGENT_XROUTER_EXCLUDED_MODELS comma-separated candidate model slugs to omit
  XAGENT_ROUTER_FALLBACK_MODEL  slug to use if routing fails

Embedding backend for the predictor's prompt encoder (all optional; the last
three are read only when XROUTER_EMBEDDING_BACKEND is set):
  XROUTER_EMBEDDING_BACKEND     unset keeps the in-process backend serialized in
                                the artifact; "xinference" sends prompt
                                embeddings to a remote OpenAI-compatible
                                endpoint; "sentence-transformers" loads the
                                model in process, rebuilt with upstream
                                defaults (model from XROUTER_EMBEDDING_MODEL,
                                max sequence length 512) instead of keeping the
                                serialized backend. Case-sensitive; any other
                                value fails the service build with ValueError
  XROUTER_EMBEDDING_MODEL       embedding model name (default
                                Qwen/Qwen3-Embedding-0.6B); must be the model
                                the artifact was trained with
  XINFERENCE_BASE_URL           endpoint base URL ending in /v1, "/embeddings"
                                is appended (default http://127.0.0.1:9997/v1)
  XINFERENCE_API_KEY            bearer key for that endpoint; the same variable
                                is read by the Xinference rerank client and as
                                the DeepDoc Xinference fallback key
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import threading
import time
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    List,
    Mapping,
    Optional,
    Sequence,
)

from .....config import (
    get_auto_candidate_cooldown_seconds,
    get_xrouter_excluded_models,
)
from ....context_ref import CONTEXT_REFS_KEY, normalize_context_references
from ....model import ChatModelConfig
from ....task_runtime import normalize_input_modalities
from ...providers import default_base_url_for_provider
from ..types import ChunkType, StreamChunk
from .base import BaseLLM
from .call_boundary import (
    CREDENTIAL_REJECTED,
    MODEL_NOT_AVAILABLE,
    PROVIDER_QUOTA,
    PROVIDER_UNAVAILABLE,
    RATE_LIMITED,
    TIMEOUT,
    ProviderCallError,
    classify_provider_failure,
    names_output_budget,
)

logger = logging.getLogger(__name__)

_DEFAULT_ROUTER_ABILITIES = ["chat", "tool_calling"]
_UNROUTED_ROUTER_ABILITIES = {"vision", "thinking_mode"}
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


# Downstream failures that say the selected model cannot serve requests right
# now -- its provider is down, slow, out of capacity or quota, or no longer
# accepts the credential or the model name -- rather than that this request
# is wrong. Only these move a call to the Auto fallback: a request the
# selected model rejected on its merits (context length, an invalid request,
# a tool-protocol violation) would fail on the fallback too. So would a
# refusal that names the output budget, even under one of these codes (an
# OpenRouter 402 "fewer max_tokens" is ``provider_quota``): the caller can
# retry it with a smaller budget, which a fallback's own failure would hide.
_FAILOVER_FAILURE_CODES = frozenset(
    {
        CREDENTIAL_REJECTED,
        MODEL_NOT_AVAILABLE,
        PROVIDER_QUOTA,
        PROVIDER_UNAVAILABLE,
        RATE_LIMITED,
        TIMEOUT,
    }
)
# A rate limit clears within seconds and, on a shared model, one caller's 429
# says little about the next caller's request, so it moves only the call that
# hit it; the rest also send the candidate's later calls to the fallback.
_COOLDOWN_FAILURE_CODES = _FAILOVER_FAILURE_CODES - {RATE_LIMITED}


def _failover_code(error: BaseException) -> str | None:
    """The failure code of a downstream failure that moves the call, if any."""
    if not isinstance(error, Exception):
        return None
    if isinstance(error, ProviderCallError):
        # An inner call boundary already classified it and dropped the cause.
        code = error.code
    else:
        try:
            code = classify_provider_failure(error)
        except Exception:  # noqa: BLE001 - classification must not mask the failure
            return None
    if code not in _FAILOVER_FAILURE_CODES or names_output_budget(error):
        return None
    return code


def is_auto_failover_failure(error: BaseException) -> bool:
    """Whether a configured Auto call would move to its fallback after ``error``."""
    return _failover_code(error) is not None


def _now() -> float:
    """The cooldown clock; tests replace this, not the process-wide clock."""
    return time.monotonic()


class _CandidateCooldowns:
    """Configured Auto candidates that recently failed, keyed by saved model.

    Per process: each replica learns about an outage from its own first
    failure. An entry expires on its own, after which the candidate is tried
    again.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._until: dict[str, float] = {}

    def start(self, key: str, seconds: float) -> None:
        with self._lock:
            self._until[key] = _now() + seconds

    def active(self, key: str) -> bool:
        with self._lock:
            until = self._until.get(key)
            if until is None:
                return False
            if _now() < until:
                return True
            del self._until[key]
            return False

    def clear(self) -> None:
        with self._lock:
            self._until.clear()


_CANDIDATE_COOLDOWNS = _CandidateCooldowns()


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


# A RoutingService holds a trained predictor whose prompt encoder runs a
# multilingual embedding model (in process unless XROUTER_EMBEDDING_BACKEND
# selects a remote backend), which is expensive, so build it once per
# (model, registry, configs) tuple and share it across all RouterLLM instances.
# The XROUTER_EMBEDDING_* / XINFERENCE_* variables are not part of that key:
# they take effect when a process first builds the service.
_SERVICE_LOCK = threading.Lock()
_SERVICE_CACHE: dict[tuple[str, str, str], Any] = {}


def _build_service(model_path: str, models_dir: str, routers_dir: str) -> Any:
    try:
        import joblib
        from xrouter_llm import embedding_backend_from_env, load_benchmark_profiles
        from xrouter_llm.serving import RoutingService, load_router_configs
        from xrouter_llm.store import CallStore
    except ImportError as exc:  # pragma: no cover - dependency missing or too old
        raise RuntimeError(
            "The 'router' (auto) provider needs xrouter-llm>=0.3.6; it is not "
            "installed or the installed version is too old. Install it with "
            "`pip install 'xagent[router]'` (or `pip install 'xrouter-llm>=0.3.6'`)."
        ) from exc

    # Resolve the backend selected by XROUTER_EMBEDDING_BACKEND (see the module
    # docstring) before loading anything. An unknown name raises ValueError here,
    # and _get_service caches only successful builds, so a misconfigured process
    # retries on every Auto call; failing first keeps each retry from reloading
    # the artifact and rerunning the call-history migrations. None keeps the
    # in-process backend serialized in the artifact.
    embedding_backend = embedding_backend_from_env()

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
    if embedding_backend is not None:
        logger.info("xrouter embedding backend: %s", embedding_backend.name)
    return RoutingService(
        predictor,
        profiles=profiles,
        configs=configs,
        store=store,
        embedding_backend=embedding_backend,
    )


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
        downstream_resolver: Optional[Callable[..., BaseLLM]] = None,
        mode: str = "routing",
        candidate_models: Optional[Sequence[str]] = None,
        candidate_profiles: Optional[Mapping[str, str]] = None,
        candidate_input_modalities: Optional[Mapping[str, Sequence[str]]] = None,
        fallback_model: Optional[str] = None,
        use_environment_fallback: bool = True,
    ) -> None:
        # model_name doubles as the xrouter-llm router config name (e.g. "auto").
        self._config_name = model_name or "auto"
        # Given a chosen routing profile, build the concrete LLM that runs it.
        # Configured Auto injects bindings to saved models; legacy OpenRouter
        # Auto injects a resolver that reuses its own credentials and base URL.
        # Configured Auto's resolver also takes ``limit_attempts=True`` for a
        # candidate whose call can still move to the fallback.
        self._downstream_resolver = downstream_resolver
        if mode not in {"routing", "sequential_failover"}:
            raise ValueError(f"Unsupported Auto mode {mode!r}")
        self._mode = mode
        self._candidate_models = (
            tuple(dict.fromkeys(str(model) for model in candidate_models))
            if candidate_models is not None
            else None
        )
        self._candidate_profiles = {
            str(candidate): str(profile)
            for candidate, profile in (candidate_profiles or {}).items()
        }
        self._candidate_input_modalities = {
            str(candidate): tuple(str(modality) for modality in modalities)
            for candidate, modalities in (candidate_input_modalities or {}).items()
        }
        if self._candidate_models == ():
            raise ValueError("A configured Auto model needs at least one candidate")
        if self._mode == "sequential_failover" and self._candidate_models is None:
            raise ValueError("Sequential Auto needs configured candidates")
        if self._candidate_profiles and (
            self._candidate_models is None
            or set(self._candidate_profiles) != set(self._candidate_models)
        ):
            raise ValueError("Auto candidate profile bindings must match candidates")
        if self._candidate_input_modalities and (
            self._candidate_models is None
            or set(self._candidate_input_modalities) != set(self._candidate_models)
        ):
            raise ValueError("Auto candidate modality bindings must match candidates")
        # The auto model's own OpenRouter credentials. Routing itself never calls
        # OpenRouter (only the prompt embedding may call a remote embedding
        # service), but these are used by the fallback resolver below when no
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
        configured_abilities = (
            list(abilities) if abilities else list(_DEFAULT_ROUTER_ABILITIES)
        )
        self._abilities = [
            ability
            for ability in configured_abilities
            if ability not in _UNROUTED_ROUTER_ABILITIES
        ]
        # Kept unfiltered (unlike ``self._abilities`` above) for the
        # ``_resolve_route`` fallback branch: that branch builds a real
        # downstream ``OpenRouterLLM`` from a ``ChatModelConfig``, and that
        # client's own ``vision``/``thinking_mode`` support must reflect what
        # the user actually configured, not this virtual router's
        # deliberately-narrowed advertised abilities.
        self._raw_abilities = configured_abilities
        self._fallback_model = fallback_model
        self._use_environment_fallback = use_environment_fallback
        if self._fallback_model is None and use_environment_fallback:
            self._fallback_model = os.getenv("XAGENT_ROUTER_FALLBACK_MODEL") or None
        if (
            self._candidate_models is not None
            and self._fallback_model is not None
            and self._fallback_model not in self._candidate_models
        ):
            raise ValueError("The Auto fallback model must be one of its candidates")
        if self._mode == "sequential_failover" and self._fallback_model is not None:
            raise ValueError("Sequential Auto uses its ordered candidates as fallbacks")
        # The candidates routing chooses between. The fallback sits out while
        # there is anything else to route to, so it only takes over a call.
        self._routed_models = (
            tuple(
                model
                for model in self._candidate_models
                if model != self._fallback_model
            )
            or self._candidate_models
            if self._candidate_models is not None
            else None
        )

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

    @property
    def uses_configured_candidates(self) -> bool:
        """Whether this router is backed by an explicit Auto candidate set."""

        return self._candidate_models is not None

    @property
    def uses_sequential_failover(self) -> bool:
        return self._mode == "sequential_failover"

    def _profile_id(self, candidate_id: str) -> str:
        """Return the routing profile behind a concrete candidate binding."""

        return self._candidate_profiles.get(candidate_id, candidate_id)

    def _input_modalities(self, candidate_id: str) -> tuple[str, ...]:
        configured = self._candidate_input_modalities.get(candidate_id)
        if configured is not None:
            return configured
        return self._profile_input_modalities(self._profile_id(candidate_id))

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
            tool_choice=prepared.compatible_tool_choice(tool_choice),
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
            tool_choice=prepared.compatible_tool_choice(tool_choice),
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
            tool_choice=prepared.compatible_tool_choice(tool_choice),
            response_format=response_format,
            thinking=thinking,
            output_config=output_config,
            **kwargs,
        ):
            yield chunk

    # ---- Routing ------------------------------------------------------------
    async def prepare_for_call(
        self,
        messages: list[dict[str, Any]],
        *,
        preferred_input_modalities: tuple[str, ...] = (),
    ) -> BaseLLM:
        """Resolve one xrouter decision into a reusable per-call LLM.

        The returned wrapper dispatches directly to the selected downstream
        model without routing a second time, and carries the selected model's
        context window from xrouter's model profile catalog. Both resolver
        branches in ``_resolve_route`` currently build an ``OpenRouterLLM``
        client for the chosen slug (the injected ``_downstream_resolver`` in
        practice, and the ``create_base_llm`` fallback always, since it is
        given ``model_provider="openrouter"``), so provider-compat retries (a
        rejected tool_choice, a thinking/tool_choice conflict, a model that
        mandates reasoning) are that client's own responsibility today, not
        this router's. ``_downstream_resolver`` is typed as
        ``Callable[..., BaseLLM]`` (it takes the profile id, and configured
        Auto's also ``limit_attempts``), not ``OpenRouterLLM``, so this is a
        statement about current call sites, not a type-level guarantee.

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
        if self.uses_sequential_failover:
            return await self._prepare_sequential_failover(
                required_input_modalities,
                route_input_modalities=route_input_modalities,
            )
        model_id, downstream = await self._resolve_route(
            messages,
            preferred_input_modalities=required_input_modalities,
            advisory_input_modalities=advisory_input_modalities,
        )
        fallback_model = self._failover_model(model_id, required_input_modalities)
        if fallback_model is not None and self._cooling_down(downstream):
            logger.warning(
                "Auto candidate %s failed recently; using fallback %s",
                model_id,
                fallback_model,
            )
            try:
                downstream = self._resolve_downstream(fallback_model)
            except Exception:  # noqa: BLE001 - the candidate may still answer
                logger.exception("Could not build Auto fallback %s", fallback_model)
                # No fallback to move to after all: the candidate gets its
                # full retry budget back.
                downstream = self._resolve_downstream(model_id)
            else:
                model_id = fallback_model
            fallback_model = None
        context_window = await self._context_window_for(model_id, downstream)
        input_modalities = (
            await asyncio.to_thread(self._profile_input_modalities, model_id)
            if route_input_modalities
            else ()
        )
        return _ResolvedRouterLLM(
            router=self,
            downstream=downstream,
            selected_model=model_id,
            context_window=context_window,
            input_modalities=input_modalities,
            fallback_model=fallback_model,
        )

    async def _prepare_sequential_failover(
        self,
        required_input_modalities: tuple[str, ...],
        *,
        route_input_modalities: tuple[str, ...],
    ) -> BaseLLM:
        """Bind the first usable candidate and retain the ordered remainder."""

        candidates = self._eligible_candidates()
        assert candidates is not None
        if required_input_modalities:
            candidates = await self._candidates_accepting(
                candidates, required_input_modalities
            )
        if not candidates:
            requested = ", ".join(required_input_modalities) or "this request"
            raise RouterModalityRoutingError(
                f"No sequential Auto candidate supports {requested}"
            )

        selected_model = candidates[0]
        downstream = self._resolve_downstream(
            selected_model, failover_available=len(candidates) > 1
        )
        remaining = candidates[1:]
        while remaining and self._cooling_down(downstream):
            logger.warning(
                "Auto candidate %s failed recently; trying next sequential candidate",
                selected_model,
            )
            selected_model = remaining[0]
            remaining = remaining[1:]
            downstream = self._resolve_downstream(
                selected_model, failover_available=bool(remaining)
            )

        context_window = await self._context_window_for(selected_model, downstream)
        input_modalities = (
            await asyncio.to_thread(self._input_modalities, selected_model)
            if route_input_modalities
            else ()
        )
        return _ResolvedRouterLLM(
            router=self,
            downstream=downstream,
            selected_model=selected_model,
            context_window=context_window,
            input_modalities=input_modalities,
            fallback_models=remaining,
        )

    async def _context_window_for(
        self, model_id: str, downstream: BaseLLM
    ) -> int | None:
        """The context window of the model that will serve a call."""
        # A saved target model describes the endpoint that will actually serve
        # the request, so its explicit context window wins over catalog data.
        context_window = getattr(downstream, "context_window", None)
        if not context_window:
            context_window = getattr(self, "context_window", None)
        if not context_window:
            context_window = await asyncio.to_thread(
                self._profile_context_window, self._profile_id(model_id)
            )
        return context_window if isinstance(context_window, int) else None

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
        # Decide once which configured candidates may serve this call, and use
        # that one decision both to hand the call to a held-back fallback and
        # to bound routing. xrouter treats modalities only as a preference, so
        # narrowing to candidates that accept the content's required
        # modalities is what keeps that content off an incompatible model;
        # advisory modalities stay a preference within the narrowed pool.
        model_id: str | None = None
        candidates = self._eligible_candidates()
        if candidates is not None and preferred_input_modalities:
            accepting = await self._candidates_accepting(
                candidates, preferred_input_modalities
            )
            if accepting:
                candidates = accepting
            else:
                model_id = self._usable_fallback(preferred_input_modalities)
                if model_id is not None:
                    logger.info(
                        "No eligible Auto candidate accepts %s; using fallback %s",
                        ", ".join(preferred_input_modalities),
                        model_id,
                    )
        elif candidates == ():
            model_id = self._usable_fallback(preferred_input_modalities)
        if (
            model_id is None
            and candidates is not None
            and len(candidates) == 1
            and not preferred_input_modalities
        ):
            # Nothing to route between: skip just the routing decision.
            model_id = candidates[0]
        if model_id is None:
            if candidates is not None:
                select_kwargs["candidates"] = candidates
            model_id = await self._select_model(prompt, **select_kwargs)
            logger.info("xrouter selected routing profile %s", model_id)
        downstream = self._resolve_downstream(model_id)
        if await self._can_fail_over(model_id, downstream, preferred_input_modalities):
            downstream = self._resolve_downstream(model_id, failover_available=True)
        return model_id, downstream

    async def _can_fail_over(
        self,
        model_id: str,
        downstream: BaseLLM,
        required_input_modalities: tuple[str, ...],
    ) -> bool:
        """Whether a call on ``model_id`` could move to the fallback.

        Mirrors the in-call checks in ``_ResolvedRouterLLM._fail_over``: the
        fallback must exist, differ from the selection, accept the request's
        required modalities, build, and not have a known window smaller than
        the selection's. Only then may the candidate give up early.
        """
        fallback_model = self._failover_model(model_id, required_input_modalities)
        if fallback_model is None:
            return False
        try:
            fallback = self._resolve_downstream(fallback_model)
        except Exception:  # noqa: BLE001 - no fallback means no early give-up
            return False
        fallback_window = await self._context_window_for(fallback_model, fallback)
        window = await self._context_window_for(model_id, downstream)
        return not (
            fallback_window is not None
            and window is not None
            and fallback_window < window
        )

    def _eligible_candidates(self) -> tuple[str, ...] | None:
        """Routed candidates left after ``XAGENT_XROUTER_EXCLUDED_MODELS``.

        ``None`` for legacy Auto, whose pool is xrouter's preset config.
        """
        if self._routed_models is None:
            return None
        excluded = frozenset(get_xrouter_excluded_models())
        return tuple(
            model
            for model in self._routed_models
            if self._profile_id(model) not in excluded
        )

    async def _candidates_accepting(
        self,
        candidates: tuple[str, ...],
        required_input_modalities: tuple[str, ...],
    ) -> tuple[str, ...]:
        """The candidates whose profiles accept every required modality."""
        required = set(required_input_modalities)
        accepting = []
        for model in candidates:
            supported = await asyncio.to_thread(self._input_modalities, model)
            if required <= set(supported):
                accepting.append(model)
        return tuple(accepting)

    def _usable_fallback(
        self, required_input_modalities: tuple[str, ...]
    ) -> str | None:
        """The fallback, when it accepts the call's required modalities."""
        try:
            return self._compatible_fallback(required_input_modalities)
        except RouterModalityRoutingError:
            return None

    def _resolve_downstream(
        self, model_id: str, *, failover_available: bool = False
    ) -> BaseLLM:
        """Build the concrete LLM that serves one routing profile.

        ``failover_available`` says this call can move to the fallback, so a
        configured resolver may give the candidate a shorter retry budget.
        """
        if self._downstream_resolver is not None:
            if failover_available and self.uses_configured_candidates:
                return self._downstream_resolver(model_id, limit_attempts=True)
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
            # Unfiltered abilities (see ``self._raw_abilities``): this builds
            # a real downstream client, not the virtual router itself, so it
            # must not inherit the router's own vision/thinking_mode
            # exclusion.
            abilities=self._raw_abilities,
        )
        return create_base_llm(config)

    def _failover_model(
        self, selected_model: str, required_input_modalities: tuple[str, ...]
    ) -> str | None:
        """The fallback a call routed to ``selected_model`` may move to."""
        if self._fallback_model is None or self._fallback_model == selected_model:
            return None
        return self._usable_fallback(required_input_modalities)

    def _cooldown_key(self, downstream: BaseLLM) -> str | None:
        # Only a configured candidate names a saved model -- one endpoint and
        # one credential. Legacy OpenRouter Auto derives every child from the
        # caller's own key, so its ids are shared across unrelated callers.
        if not self.uses_configured_candidates:
            return None
        key = getattr(downstream, "model_id", "")
        return str(key) if key else None

    def _start_cooldown(self, downstream: BaseLLM, failure_code: str) -> None:
        key = self._cooldown_key(downstream)
        if key is not None and failure_code in _COOLDOWN_FAILURE_CODES:
            _CANDIDATE_COOLDOWNS.start(key, get_auto_candidate_cooldown_seconds())

    def _cooling_down(self, downstream: BaseLLM) -> bool:
        key = self._cooldown_key(downstream)
        return key is not None and _CANDIDATE_COOLDOWNS.active(key)

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

    def _compatible_fallback(
        self, required_input_modalities: tuple[str, ...]
    ) -> str | None:
        if self._fallback_model is None:
            return None
        if not required_input_modalities:
            return self._fallback_model
        supported = set(self._profile_input_modalities(self._fallback_model))
        missing = sorted(set(required_input_modalities) - supported)
        if missing:
            raise RouterModalityRoutingError(
                f"Auto fallback model {self._fallback_model!r} does not support "
                f"required input modalities: {', '.join(missing)}"
            )
        return self._fallback_model

    async def _select_model(
        self,
        prompt: str,
        *,
        candidates: Optional[Sequence[str]] = None,
        preferred_input_modalities: tuple[str, ...] = (),
        advisory_input_modalities: tuple[str, ...] = (),
    ) -> str:
        # The decision either embeds in-process (CPU-bound) or calls a remote
        # embedding endpoint (blocking I/O), so run it in a worker thread to avoid
        # blocking the event loop.
        try:
            selected = await asyncio.to_thread(
                self._route_sync,
                prompt,
                preferred_input_modalities,
                advisory_input_modalities,
                candidates,
            )
        except RouterModalityRoutingError:
            raise
        except Exception as exc:  # noqa: BLE001 - routing must not crash the agent
            fallback_model = self._compatible_fallback(preferred_input_modalities)
            if fallback_model:
                logger.warning(
                    "xrouter route failed (%s); using fallback %s",
                    exc,
                    fallback_model,
                )
                return fallback_model
            raise RuntimeError(
                f"xrouter-llm routing failed: {exc}. "
                + (
                    "Set XAGENT_ROUTER_FALLBACK_MODEL to degrade gracefully."
                    if self._use_environment_fallback
                    else "Configure an Auto fallback model to degrade gracefully."
                )
            ) from exc
        if not selected:
            fallback_model = self._compatible_fallback(preferred_input_modalities)
            if fallback_model:
                return fallback_model
            raise RuntimeError("xrouter-llm returned no selected model")
        return str(selected[0])

    def _route_sync(
        self,
        prompt: str,
        preferred_input_modalities: tuple[str, ...] = (),
        advisory_input_modalities: tuple[str, ...] = (),
        candidates: Optional[Sequence[str]] = None,
    ) -> list[str]:
        """Route once.

        ``candidates`` bounds a configured Auto call to the pool already
        decided for it; without it the router's own pool is used.

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
        supports_keyword_arguments = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in route_parameters.values()
        )
        supports_modality_preferences = (
            "preferred_input_modalities" in route_parameters
            or supports_keyword_arguments
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

        excluded_models = frozenset(get_xrouter_excluded_models())
        configured_models = (
            tuple(candidates)
            if candidates is not None
            else tuple(self._candidate_pool(service))
        )
        eligible_models = [
            model for model in configured_models if model not in excluded_models
        ]
        if self._candidate_models is not None or (
            excluded_models and len(eligible_models) != len(configured_models)
        ):
            if not eligible_models:
                raise RuntimeError(
                    f"{self._config_name!r} has no candidates after applying "
                    "XAGENT_XROUTER_EXCLUDED_MODELS"
                )
            if "models" not in route_parameters and not supports_keyword_arguments:
                raise RuntimeError(
                    "The installed xrouter-llm RoutingService cannot apply "
                    "XAGENT_XROUTER_EXCLUDED_MODELS; upgrade xrouter-llm."
                )
            route_kwargs["models"] = eligible_models
        result = service.route(prompt, **route_kwargs)
        return list(result.get("selected") or [])

    def _candidate_pool(self, service: Any) -> list[str]:
        """Return this Auto model's explicit or preset candidate pool."""
        if self._routed_models is not None:
            return list(self._routed_models)
        config = getattr(service, "configs", {}).get(self._config_name)
        return list(getattr(config, "models", ()) or ())

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
    """A concrete xrouter selection reused for one logical LLM call.

    With a ``fallback_model``, the first call that fails with an availability
    failure moves this selection to the fallback for good: the call is sent
    again there, and later calls on this object go there directly.
    """

    def __init__(
        self,
        *,
        router: RouterLLM,
        downstream: BaseLLM,
        selected_model: str,
        context_window: int | None,
        input_modalities: tuple[str, ...],
        fallback_model: str | None = None,
        fallback_models: tuple[str, ...] = (),
    ) -> None:
        self._router = router
        self._input_modalities = input_modalities
        self._fallback_models = (
            (fallback_model,) if fallback_model is not None else fallback_models
        )
        self._attempted_models = [selected_model]
        self.context_window = context_window
        self._forced_tool_choice_supported = True
        self._bind(downstream, selected_model)

    def _bind(self, downstream: BaseLLM, selected_model: str) -> None:
        router = self._router
        self._downstream = downstream
        self._selected_model = selected_model
        # Callers adapt ``tool_choice`` to the model bound when they build the
        # request and read this again after the call. A forced choice only
        # survived if every model this call was bound to supports it, so a
        # move to a fallback can lower this but never raise it.
        self._forced_tool_choice_supported = (
            self._forced_tool_choice_supported
            and bool(getattr(downstream, "supports_forced_tool_choice", True))
        )
        ability_source = downstream if router.uses_configured_candidates else router
        abilities = list(getattr(ability_source, "abilities", router.abilities))
        if not router.uses_configured_candidates:
            for modality in self._input_modalities:
                ability = _MODALITY_ABILITIES.get(modality)
                if ability is not None and ability not in abilities:
                    abilities.append(ability)
        self._abilities = abilities

    def _record_failure(self, error: BaseException) -> str | None:
        """Start the candidate's cooldown after an availability failure.

        Independent of whether this call moves: a failure after output has
        reached the caller cannot be retried elsewhere, but the next call
        should still go to the fallback. Returns the failure code, if any.
        """
        if not self._fallback_models:
            return None
        failure_code = _failover_code(error)
        if failure_code is not None:
            self._router._start_cooldown(self._downstream, failure_code)
        return failure_code

    async def _fail_over(self, error: BaseException) -> bool:
        """Move to the fallback after ``error``; whether the call may go again.

        A fallback with a known window smaller than the one this call was
        sized for is not used for it: the transcript may not fit, and a
        context-length failure ends the task where the original availability
        failure can be resumed. The cooldown still starts, so the next call
        goes to the fallback and is compacted for its window first.
        """
        failure_code = self._record_failure(error)
        if failure_code is None:
            return False
        failed_model = self._selected_model
        while self._fallback_models:
            fallback_model = self._fallback_models[0]
            self._fallback_models = self._fallback_models[1:]
            try:
                fallback = self._router._resolve_downstream(
                    fallback_model,
                    failover_available=bool(self._fallback_models),
                )
            except Exception:  # noqa: BLE001 - report the provider failure
                logger.exception(
                    "Could not build Auto fallback %s after %s failed",
                    fallback_model,
                    failed_model,
                )
                continue
            fallback_window = await self._router._context_window_for(
                fallback_model, fallback
            )
            if (
                fallback_window is not None
                and self.context_window is not None
                and fallback_window < self.context_window
            ):
                logger.warning(
                    "Auto candidate %s failed (%s); fallback %s has a smaller "
                    "context window, so this call skips it",
                    failed_model,
                    failure_code,
                    fallback_model,
                )
                continue
            logger.warning(
                "Auto candidate %s failed (%s); sending the call to fallback %s",
                failed_model,
                failure_code,
                fallback_model,
            )
            self._attempted_models.append(fallback_model)
            self._bind(fallback, fallback_model)
            if fallback_window is not None:
                self.context_window = fallback_window
            return True
        return False

    def _add_exhaustion_note(self, error: BaseException) -> None:
        if len(self._attempted_models) < 2:
            return
        add_note = getattr(error, "add_note", None)
        if add_note is not None:
            add_note(
                "Auto sequential failover exhausted: "
                + " -> ".join(self._attempted_models)
            )

    @property
    def model_id(self) -> str:
        return self._downstream.model_id

    @property
    def timeout(self) -> float:
        return getattr(self._downstream, "timeout", self._router.timeout)

    @property
    def abilities(self) -> List[str]:
        return self._abilities

    @property
    def model_name(self) -> str:
        return self._selected_model

    @property
    def supports_thinking_mode(self) -> bool:
        if not self._router.uses_configured_candidates:
            return self._router.supports_thinking_mode
        return getattr(
            self._downstream,
            "supports_thinking_mode",
            self._router.supports_thinking_mode,
        )

    @property
    def supports_json_schema_response_format(self) -> bool:
        return self._downstream.supports_json_schema_response_format

    @property
    def supports_json_object_response_format(self) -> bool:
        return self._downstream.supports_json_object_response_format

    @property
    def supports_forced_tool_choice(self) -> bool:
        return self._forced_tool_choice_supported

    def _method(self, name: str) -> Callable[..., Awaitable[str | dict[str, Any]]]:
        if name == "vision_chat":
            return self._downstream.vision_chat
        return self._downstream.chat

    async def _call(
        self,
        method: str,
        messages: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] | None,
        kwargs: dict[str, Any],
    ) -> str | dict[str, Any]:
        while True:
            try:
                return await self._method(method)(
                    messages,
                    tool_choice=self.compatible_tool_choice(tool_choice),
                    **kwargs,
                )
            except Exception as exc:
                if await self._fail_over(exc):
                    continue
                self._add_exhaustion_note(exc)
                raise

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
        return await self._call(
            "chat",
            messages,
            tool_choice,
            dict(
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                response_format=response_format,
                thinking=thinking,
                output_config=output_config,
                **kwargs,
            ),
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
        return await self._call(
            "vision_chat",
            messages,
            tool_choice,
            dict(
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                response_format=response_format,
                thinking=thinking,
                output_config=output_config,
                **kwargs,
            ),
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
        call_kwargs = dict(
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            response_format=response_format,
            thinking=thinking,
            output_config=output_config,
            **kwargs,
        )
        while True:
            stream = self._downstream.stream_chat(
                messages,
                tool_choice=self.compatible_tool_choice(tool_choice),
                **call_kwargs,
            )
            failed_over = False
            try:
                started = False
                try:
                    async for chunk in stream:
                        raw = getattr(chunk, "raw", None)
                        if getattr(
                            chunk, "type", None
                        ) == ChunkType.ERROR and isinstance(raw, BaseException):
                            if not started and await self._fail_over(raw):
                                failed_over = True
                                break
                            if started:
                                self._record_failure(raw)
                        started = True
                        yield chunk
                except Exception as exc:
                    if started:
                        self._record_failure(exc)
                        raise
                    if not await self._fail_over(exc):
                        self._add_exhaustion_note(exc)
                        raise
                    failed_over = True
            finally:
                aclose = getattr(stream, "aclose", None)
                if aclose is not None:
                    try:
                        await aclose()
                    except Exception:  # noqa: BLE001 - must not replace the outcome
                        logger.warning(
                            "Could not close the Auto candidate stream", exc_info=True
                        )
            if not failed_over:
                return
