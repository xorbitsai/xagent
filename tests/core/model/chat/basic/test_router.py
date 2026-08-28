"""Tests for RouterLLM's xrouter routing and per-call dispatch.

Provider-compatibility retries used to live here too; they are now owned by
OpenRouterLLM itself (see test_openrouter.py) since RouterLLM no longer
retries the downstream call on its own.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, AsyncIterator

import httpx
import openai
import pytest

from xagent.core.context_ref import CONTEXT_REFS_KEY, ContextReference
from xagent.core.model.chat.basic.base import BaseLLM
from xagent.core.model.chat.basic.openrouter import OpenRouterLLM
from xagent.core.model.chat.basic.router import (
    RouterLLM,
    RouterModalityRoutingError,
)
from xagent.core.model.chat.error import retry_on
from xagent.core.model.chat.exceptions import LLMToolProtocolError
from xagent.core.model.chat.types import (
    PROVIDER_STATE_METADATA_KEY,
    ChunkType,
    StreamChunk,
)
from xagent.core.retry.strategy import FixedDelay
from xagent.core.retry.wrapper import create_retry_wrapper


class _ScriptedChatLLM(BaseLLM):
    def __init__(self, errors: list[str]) -> None:
        self.errors = list(errors)
        self.tool_choices: list[str | dict[str, Any] | None] = []
        self.thinking_values: list[dict[str, Any] | None] = []

    @property
    def abilities(self) -> list[str]:
        return ["chat", "tool_calling"]

    @property
    def model_name(self) -> str:
        return "z-ai/glm-5.2"

    @property
    def supports_thinking_mode(self) -> bool:
        return False

    async def chat(
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
        del messages, temperature, max_tokens, tools, response_format
        del output_config, kwargs
        self.tool_choices.append(tool_choice)
        self.thinking_values.append(thinking)
        if self.errors:
            raise RuntimeError(self.errors.pop(0))
        return "ok"

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
        del messages, temperature, max_tokens, tools, response_format
        del output_config, kwargs
        self.tool_choices.append(tool_choice)
        self.thinking_values.append(thinking)
        if self.errors:
            raise RuntimeError(self.errors.pop(0))
        yield StreamChunk(type=ChunkType.TOKEN, content="ok", delta="ok")


@pytest.mark.asyncio
async def test_prepare_for_call_reuses_route_and_exposes_profile_context_window(
    monkeypatch,
):
    downstream = _ScriptedChatLLM([])
    downstream._model_id = "configured-openrouter-model"
    router = RouterLLM(
        timeout=42.0,
        downstream_resolver=lambda _model_id: downstream,
    )
    selected: list[str] = []

    async def select_model(prompt: str) -> str:
        selected.append(prompt)
        return "deepseek/deepseek-v4-flash"

    monkeypatch.setattr(router, "_select_model", select_model)
    monkeypatch.setattr(
        router,
        "_profile_context_window",
        lambda _model_id: 1_048_576,
    )

    prepared = await router.prepare_for_call(
        [{"role": "user", "content": "make a podcast"}]
    )

    assert prepared.model_name == "deepseek/deepseek-v4-flash"
    assert prepared.model_id == "configured-openrouter-model"
    assert prepared.timeout == 42.0
    assert prepared.context_window == 1_048_576
    assert await prepared.chat([{"role": "user", "content": "continue"}]) == "ok"
    assert selected == ["make a podcast"]


@pytest.mark.asyncio
async def test_prepare_for_call_prefers_and_exposes_context_ref_modality(
    monkeypatch,
):
    downstream = _ScriptedChatLLM([])
    router = RouterLLM(downstream_resolver=lambda _model_id: downstream)
    selected: list[tuple[str, tuple[str, ...]]] = []

    async def select_model(
        prompt: str,
        *,
        preferred_input_modalities: tuple[str, ...] = (),
    ) -> str:
        selected.append((prompt, preferred_input_modalities))
        return "openai/gpt-5.5"

    reference = ContextReference(
        file_ref={
            "file_id": "image-1",
            "filename": "image.png",
            "mime_type": "image/png",
        }
    )
    monkeypatch.setattr(router, "_select_model", select_model)
    monkeypatch.setattr(router, "_profile_context_window", lambda _model_id: 128_000)
    monkeypatch.setattr(
        router,
        "_profile_input_modalities",
        lambda _model_id: ("text", "image"),
    )

    prepared = await router.prepare_for_call(
        [
            {
                "role": "user",
                "content": "inspect",
                CONTEXT_REFS_KEY: [reference.durable_dict()],
            }
        ]
    )

    assert selected == [("inspect", ("image",))]
    assert prepared.has_ability("vision")


@pytest.mark.asyncio
async def test_prepare_for_call_merges_runtime_and_message_modalities(monkeypatch):
    downstream = _ScriptedChatLLM([])
    router = RouterLLM(downstream_resolver=lambda _model_id: downstream)
    selected: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []

    async def select_model(
        prompt: str,
        *,
        preferred_input_modalities: tuple[str, ...] = (),
        advisory_input_modalities: tuple[str, ...] = (),
    ) -> str:
        selected.append((prompt, preferred_input_modalities, advisory_input_modalities))
        return "openai/gpt-5.5"

    monkeypatch.setattr(router, "_select_model", select_model)
    monkeypatch.setattr(router, "_profile_context_window", lambda _model_id: 128_000)
    monkeypatch.setattr(
        router,
        "_profile_input_modalities",
        lambda _model_id: ("text", "image"),
    )

    prepared = await router.prepare_for_call(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "inspect"},
                    {"type": "input_audio", "input_audio": {}},
                ],
            }
        ],
        preferred_input_modalities=(None, "IMAGE"),  # type: ignore[arg-type]
    )

    # Message-derived audio is a hard requirement; the extension-declared image
    # preference stays advisory and is kept separately addressable.
    assert selected == [("inspect", ("audio",), ("image",))]
    assert prepared.has_ability("vision")


def test_router_detects_modalities_from_refs_and_content_parts() -> None:
    reference = ContextReference(
        file_ref={
            "file_id": "image-1",
            "filename": "image.png",
            "mime_type": "image/png",
        }
    )

    modalities = RouterLLM._preferred_input_modalities(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "inspect"},
                    {"type": "input_audio", "input_audio": {}},
                ],
                CONTEXT_REFS_KEY: [reference.durable_dict()],
            }
        ]
    )

    assert modalities == ("audio", "image")


def test_route_sync_forwards_modalities_when_router_supports_them(
    monkeypatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class Service:
        def route(
            self,
            prompt: str,
            *,
            config_name: str,
            preferred_input_modalities: tuple[str, ...] = (),
        ) -> dict[str, Any]:
            calls.append(
                {
                    "prompt": prompt,
                    "config_name": config_name,
                    "preferred_input_modalities": preferred_input_modalities,
                }
            )
            return {"selected": ["openai/gpt-5.5"]}

    monkeypatch.setattr(
        "xagent.core.model.chat.basic.router._get_service",
        lambda: Service(),
    )

    selected = RouterLLM()._route_sync("inspect", ("image",))

    assert selected == ["openai/gpt-5.5"]
    assert calls == [
        {
            "prompt": "inspect",
            "config_name": "auto",
            "preferred_input_modalities": ("image",),
        }
    ]


def test_route_sync_forwards_advisory_modalities_when_supported(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    class Service:
        def route(
            self,
            prompt: str,
            *,
            config_name: str,
            preferred_input_modalities: tuple[str, ...] = (),
        ) -> dict[str, Any]:
            del prompt, config_name
            calls.append(preferred_input_modalities)
            return {"selected": ["openai/gpt-5.5"]}

    monkeypatch.setattr(
        "xagent.core.model.chat.basic.router._get_service",
        lambda: Service(),
    )

    selected = RouterLLM()._route_sync("inspect", ("audio",), ("image",))

    assert selected == ["openai/gpt-5.5"]
    assert calls == [("image", "audio")]


def test_route_sync_rejects_older_router_api_for_modality_requests(
    monkeypatch,
) -> None:
    class Service:
        def route(self, prompt: str, *, config_name: str) -> dict[str, Any]:
            return {"selected": ["text/model"]}

    monkeypatch.setattr(
        "xagent.core.model.chat.basic.router._get_service",
        lambda: Service(),
    )

    with pytest.raises(RouterModalityRoutingError, match="image"):
        RouterLLM()._route_sync("inspect", ("image",))


@pytest.mark.asyncio
async def test_modality_support_error_is_not_hidden_by_generic_fallback(
    monkeypatch,
) -> None:
    class Service:
        def route(self, prompt: str, *, config_name: str) -> dict[str, Any]:
            return {"selected": ["text/model"]}

    monkeypatch.setenv("XAGENT_ROUTER_FALLBACK_MODEL", "fallback/model")
    monkeypatch.setattr(
        "xagent.core.model.chat.basic.router._get_service",
        lambda: Service(),
    )
    router = RouterLLM()

    with pytest.raises(RouterModalityRoutingError, match="explicit compatible model"):
        await router._select_model(
            "inspect",
            preferred_input_modalities=("image",),
        )


class _LegacyModalityUnawareService:
    """An installed xrouter-llm whose route() predates modality preferences."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def route(self, prompt: str, *, config_name: str) -> dict[str, Any]:
        self.calls.append({"prompt": prompt, "config_name": config_name})
        return {"selected": ["text/model"]}


def _legacy_modality_unaware_router(monkeypatch) -> tuple[RouterLLM, Any]:
    service = _LegacyModalityUnawareService()
    monkeypatch.setenv("XAGENT_ROUTER_FALLBACK_MODEL", "fallback/model")
    monkeypatch.setattr(
        "xagent.core.model.chat.basic.router._get_service",
        lambda: service,
    )
    downstream = _ScriptedChatLLM([])
    router = RouterLLM(downstream_resolver=lambda _model_id: downstream)
    monkeypatch.setattr(router, "_profile_context_window", lambda _model_id: 128_000)
    monkeypatch.setattr(
        router,
        "_profile_input_modalities",
        lambda _model_id: ("text",),
    )
    return router, service


@pytest.mark.asyncio
async def test_extension_modalities_degrade_when_router_cannot_honor_them(
    monkeypatch,
) -> None:
    """Extension-declared modalities are advisory: degrade, never hard-fail."""

    router, service = _legacy_modality_unaware_router(monkeypatch)

    prepared = await router.prepare_for_call(
        [{"role": "user", "content": "plain text only"}],
        preferred_input_modalities=("image",),
    )

    assert prepared.model_name == "text/model"
    assert service.calls == [{"prompt": "plain text only", "config_name": "auto"}]


@pytest.mark.asyncio
async def test_message_derived_modalities_still_hard_fail(monkeypatch) -> None:
    """A modality the conversation actually carries stays a hard requirement."""

    router, _service = _legacy_modality_unaware_router(monkeypatch)

    with pytest.raises(RouterModalityRoutingError, match="image"):
        await router.prepare_for_call(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "inspect"},
                        {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
                    ],
                }
            ]
        )


@pytest.mark.asyncio
async def test_mixed_modalities_hard_fail_on_message_derived_requirement(
    monkeypatch,
) -> None:
    router, _service = _legacy_modality_unaware_router(monkeypatch)

    with pytest.raises(RouterModalityRoutingError, match="image"):
        await router.prepare_for_call(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "inspect"},
                        {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
                    ],
                }
            ],
            preferred_input_modalities=("audio",),
        )


@pytest.mark.asyncio
async def test_prepare_for_call_handles_missing_router_context_window(monkeypatch):
    downstream = _ScriptedChatLLM([])
    router = RouterLLM(downstream_resolver=lambda _model_id: downstream)

    async def select_model(_prompt: str) -> str:
        return "deepseek/deepseek-v4-flash"

    monkeypatch.delattr(BaseLLM, "context_window")
    monkeypatch.setattr(router, "_select_model", select_model)
    monkeypatch.setattr(
        router,
        "_profile_context_window",
        lambda _model_id: 1_048_576,
    )

    prepared = await router.prepare_for_call(
        [{"role": "user", "content": "make a podcast"}]
    )

    assert prepared.context_window == 1_048_576


def test_profile_context_window_returns_none_when_catalog_lookup_fails(
    monkeypatch, caplog
):
    def fail_service_lookup():
        raise RuntimeError("profile catalog unavailable")

    monkeypatch.setattr(
        "xagent.core.model.chat.basic.router._get_service",
        fail_service_lookup,
    )

    assert RouterLLM._profile_context_window("test/model") is None
    assert "Could not resolve xrouter context window for test/model" in caplog.text


async def _select_glm(_prompt: str) -> str:
    return "z-ai/glm-5.2"


def _tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "answer",
            "description": "Answer the user",
            "parameters": {"type": "object", "properties": {}},
        },
    }


# ==========================================================================
# Three-layer sandwich: RouterLLM -> create_retry_wrapper -> OpenRouterLLM.
#
# RouterLLM itself no longer retries (see router.py); OpenRouterLLM owns the
# provider-compat retry, and create_retry_wrapper (adapter.py's usual wrapping
# of a downstream model) owns the outer retry-on-retryable-error budget. This
# pins the combined upper bound on upstream calls so the two layers cannot
# compound into an unbounded retry storm.
# ==========================================================================

_PURE_TOOL_CHOICE_404 = (
    "No endpoints found that support the provided 'tool_choice' value."
)
_THINKING_ONLY_CONFLICT = "Thinking mode does not support this tool_choice"
_MANDATORY_REASONING_ONLY = (
    "Reasoning is mandatory for this endpoint and cannot be disabled."
)


def _sandwiched_openrouter_llm(
    mocker,
    *,
    side_effect,
    max_retries: int,
    model_name: str = "z-ai/glm-5.2",
):
    """RouterLLM wired to create_retry_wrapper(OpenRouterLLM), matching how
    adapter.py wraps a resolved downstream model in production.

    The OpenAI client itself is mocked (not ``_chat_with_prefix_retry``), so
    ``mock_client.chat.completions.create.await_count`` reflects the real
    number of upstream requests OpenRouterLLM actually issues.
    """
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = side_effect
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    inner = OpenRouterLLM(model_name=model_name, api_key="test-key")
    wrapped = create_retry_wrapper(
        inner,
        BaseLLM,  # type: ignore[type-abstract]
        retry_methods={"chat"},
        strategy=FixedDelay(delay_ms=0),
        max_retries=max_retries,
        retry_on=retry_on,
    )
    router = RouterLLM(downstream_resolver=lambda _model_id: wrapped)
    return router, mock_client


@pytest.mark.asyncio
async def test_sandwiched_pure_404_bounds_total_upstream_calls(monkeypatch, mocker):
    """A persistent tool_choice 404 never compounds past the documented bound.

    OpenRouterLLM's own rule 3 relaxes tool_choice to "auto" once; a second
    identical 404 then fails rule 3's own precondition (tool_choice is no
    longer a strict value), so the compat retry gives up with a plain
    RuntimeError that create_retry_wrapper's retry_on() does not retry.
    """
    router, mock_client = _sandwiched_openrouter_llm(
        mocker,
        side_effect=RuntimeError(_PURE_TOOL_CHOICE_404),
        max_retries=2,
    )
    monkeypatch.setattr(router, "_select_model", _select_glm)

    with pytest.raises(RuntimeError, match="No endpoints found"):
        await router.chat(
            [{"role": "user", "content": "score?"}],
            tools=[_tool_schema()],
            tool_choice="required",
        )

    # 1 original call + 1 tool_choice relaxation: the second identical 404
    # then fails rule 3's own precondition (tool_choice is already "auto"),
    # so no further adjustment is possible and the exact count is pinned.
    assert mock_client.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_sandwiched_alternating_errors_bounds_total_upstream_calls(
    monkeypatch, mocker
):
    """A worst-case chain (all 3 compat rules fire, then a wrapper-retryable
    error) repeats at most once per wrapper attempt, bounding total upstream
    calls at wrapper_budget * 4 (1 original + 3 adjustments per attempt).
    """
    calls = 0

    def fake_create(*_args, **kwargs):
        nonlocal calls
        del kwargs
        calls += 1
        position = (calls - 1) % 4
        if position == 0:
            raise RuntimeError(_PURE_TOOL_CHOICE_404)
        if position == 1:
            raise RuntimeError(_THINKING_ONLY_CONFLICT)
        if position == 2:
            raise RuntimeError(_MANDATORY_REASONING_ONLY)
        # The 4th call in each attempt: every compat rule has already fired
        # once this call, so this retryable error is what escapes to the
        # outer wrapper and decides whether to spend another whole attempt.
        raise LLMToolProtocolError(
            provider="deepseek",
            code="model_output_boundary",
            message="retryable boundary condition",
        )

    wrapper_budget = 2
    router, mock_client = _sandwiched_openrouter_llm(
        mocker, side_effect=fake_create, max_retries=wrapper_budget
    )
    monkeypatch.setattr(router, "_select_model", _select_glm)

    with pytest.raises(LLMToolProtocolError, match="retryable boundary"):
        await router.chat(
            [{"role": "user", "content": "score?"}],
            tools=[_tool_schema()],
            tool_choice="required",
            thinking={"type": "enabled", "enable": True},
        )

    assert mock_client.chat.completions.create.await_count <= wrapper_budget * 4


def _deepseek_prefix_error() -> openai.BadRequestError:
    return openai.BadRequestError(
        "Error code: 400 - {'error': {'message': 'Provider returned error'}}",
        response=httpx.Response(
            400,
            request=httpx.Request(
                "POST", "https://openrouter.ai/api/v1/chat/completions"
            ),
        ),
        body={
            "error": {
                "message": "Provider returned error",
                "code": 400,
                "metadata": {
                    "provider_name": "DeepSeek",
                    "raw": (
                        '{"error":{"message":'
                        '"Function call should not be used with prefix"}}'
                    ),
                },
            }
        },
    )


def _deepseek_tool_call_history() -> list[dict]:
    return [
        {"role": "user", "content": "Generate music"},
        {
            "role": "assistant",
            "content": "I will generate the music first.",
            "tool_calls": [
                {
                    "id": "call_music",
                    "type": "function",
                    "function": {
                        "name": "generate_music",
                        "arguments": '{"prompt":"intro"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_music", "content": '{"success":true}'},
    ]


async def _select_deepseek(_prompt: str) -> str:
    return "deepseek/deepseek-v4-flash"


@pytest.mark.asyncio
async def test_sandwiched_deepseek_prefix_sanitizes_messages_once(monkeypatch, mocker):
    """A DeepSeek prefix rejection sanitizes messages once per call, not once
    per compat iteration.

    Before the out-param fix, ``_run_chat_with_compat_retry`` replayed the
    same unsanitized messages on every compat iteration, so a DeepSeek model
    chaining a prefix rejection with a compat-adjustable error re-triggered
    the prefix rejection (and its sanitize-and-retry) on every iteration
    too, doubling the real upstream call count. With the fix, the sanitized
    messages from the first iteration carry over, so the prefix rejection
    only ever fires once and the total call count stays bounded.
    """
    calls = 0
    prefix_error_count = 0
    post_sanitize_calls = 0
    success_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="ok", tool_calls=None, reasoning_content=None
                )
            )
        ],
        usage=None,
        model_dump=lambda: {"id": "openrouter-sandwiched-deepseek"},
    )

    def fake_create(*_args, **kwargs):
        nonlocal calls, prefix_error_count, post_sanitize_calls
        calls += 1
        messages = kwargs["messages"]
        assistant_message = next(
            message for message in messages if message.get("tool_calls")
        )
        if assistant_message.get("content"):
            prefix_error_count += 1
            raise _deepseek_prefix_error()
        post_sanitize_calls += 1
        if post_sanitize_calls == 1:
            raise RuntimeError(_MANDATORY_REASONING_ONLY)
        return success_response

    router, mock_client = _sandwiched_openrouter_llm(
        mocker,
        side_effect=fake_create,
        max_retries=2,
        model_name="deepseek/deepseek-v4-flash",
    )
    monkeypatch.setattr(router, "_select_model", _select_deepseek)

    result = await router.chat(_deepseek_tool_call_history())

    assert result["content"] == "ok"
    assert prefix_error_count == 1
    # 1 prefix rejection + 1 sanitized-but-mandatory-reasoning failure + 1
    # sanitized success: the sanitized messages carry over across compat
    # iterations, so the prefix rejection never fires a second time.
    assert mock_client.chat.completions.create.await_count == 3


# ---------------------------------------------------------------------------
# ``_resolve_route``'s fallback branch (no injected ``downstream_resolver``):
# the abilities passed to the ``ChatModelConfig`` it builds must be the
# caller's original configuration, not ``RouterLLM._abilities`` (which always
# excludes vision/thinking_mode -- see ``_UNROUTED_ROUTER_ABILITIES`` -- since
# a virtual router cannot claim a candidate's dynamic abilities before
# routing picks one). Before this fix, a user-declared
# ``thinking_mode``/``vision`` ability was silently dropped from the
# actually-constructed downstream client on this branch.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_router_fallback_downstream_supports_thinking_mode_matches_configured_abilities(
    monkeypatch,
):
    router = RouterLLM(
        api_key="test-key",
        abilities=["chat", "tool_calling", "thinking_mode"],
    )
    monkeypatch.setattr(router, "_select_model", _select_deepseek)

    model_id, downstream = await router._resolve_route(
        [{"role": "user", "content": "hi"}]
    )

    assert model_id == "deepseek/deepseek-v4-flash"
    # The router's own advertised abilities never include thinking_mode...
    assert router.supports_thinking_mode is False
    # ...but the downstream client the fallback branch actually built must
    # still see it, because that is what the user configured.
    assert downstream.supports_thinking_mode is True


@pytest.mark.asyncio
async def test_router_fallback_downstream_has_ability_vision_matches_configured_abilities(
    monkeypatch,
):
    router = RouterLLM(
        api_key="test-key",
        abilities=["chat", "tool_calling", "vision"],
    )
    monkeypatch.setattr(router, "_select_model", _select_deepseek)

    _model_id, downstream = await router._resolve_route(
        [{"role": "user", "content": "hi"}]
    )

    assert router.has_ability("vision") is False
    assert downstream.has_ability("vision") is True


@pytest.mark.asyncio
async def test_router_auto_fallback_vision_deepseek_round_trips_reasoning(
    monkeypatch, mocker
):
    """Auto routing to a deepseek model on a vision call still captures reasoning.

    This is where two separate behaviors meet: ``vision_chat`` inheriting
    reasoning capture, and the fallback branch building its downstream
    client from the caller's configured abilities. The combination is a real
    user situation -- an image in the conversation, no explicit model
    chosen, routed by "auto" to a deepseek slug, on the code path with no
    injected ``downstream_resolver`` -- so it must run end-to-end with
    capture firing.
    """
    tool_call = SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name="search", arguments="{}"),
    )
    message = SimpleNamespace(
        content=None,
        tool_calls=[tool_call],
        reasoning_content="Looking at the picture first.",
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=None,
        model_dump=lambda: {"id": "auto-fallback-vision-deepseek"},
    )
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = response
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )

    async def select_deepseek(_prompt: str, **_kwargs: Any) -> str:
        return "deepseek/deepseek-v4-flash"

    router = RouterLLM(
        api_key="test-key",
        abilities=["chat", "tool_calling", "vision", "thinking_mode"],
    )
    monkeypatch.setattr(router, "_select_model", select_deepseek)
    monkeypatch.setattr(router, "_profile_context_window", lambda _model_id: None)
    monkeypatch.setattr(
        RouterLLM,
        "_profile_input_modalities",
        staticmethod(lambda _model_id: ("image",)),
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
            ],
        }
    ]

    prepared = await router.prepare_for_call(messages)
    # xrouter's profile reports this model supports the image modality, so
    # the resolved wrapper's own abilities pick up "vision" too (separate
    # from the fallback-construction fix above, but both must hold for this
    # combined scenario to actually reach vision_chat successfully).
    assert prepared.has_ability("vision") is True

    result = await prepared.vision_chat(
        messages,
        tools=[_tool_schema()],
        thinking={"type": "enabled"},
    )

    assert result[PROVIDER_STATE_METADATA_KEY] == {
        "deepseek": {"reasoning_content": "Looking at the picture first."}
    }
