"""Tests for Auto failing a call over to its fallback model."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

import httpx
import openai
import pytest

from xagent.core.model.chat.basic import router as router_module
from xagent.core.model.chat.basic.adapter import attach_chat_retry_wrapper
from xagent.core.model.chat.basic.base import BaseLLM
from xagent.core.model.chat.basic.call_boundary import ProviderCallError
from xagent.core.model.chat.basic.router import RouterLLM
from xagent.core.model.chat.exceptions import (
    LLMEmptyContentError,
    LLMToolProtocolError,
)
from xagent.core.model.chat.types import ChunkType, StreamChunk

PRIMARY = "deepseek/deepseek-v4-flash"
FALLBACK = "openai/gpt-5.5"
MESSAGES = [{"role": "user", "content": "summarize the report"}]

_REQUEST = httpx.Request("POST", "https://provider.example/v1/chat/completions")


def _status_error(cls: type[openai.APIStatusError], status: int) -> Exception:
    return cls(
        "provider said no",
        response=httpx.Response(status, request=_REQUEST),
        body=None,
    )


class _FakeLLM(BaseLLM):
    def __init__(
        self,
        model_id: str,
        *,
        error: BaseException | None = None,
        chunks: list[StreamChunk] | None = None,
        error_after_chunks: bool = False,
        context_window: int | None = None,
        forced_tool_choice: bool = True,
    ) -> None:
        self._model_id = model_id
        self.forced_tool_choice = forced_tool_choice
        self.tool_choices: list[Any] = []
        self.context_window = context_window
        self.stream_closed = False
        self.error = error
        self.chunks = chunks or [
            StreamChunk(type=ChunkType.TOKEN, content=model_id, delta=model_id)
        ]
        self.error_after_chunks = error_after_chunks
        self.calls = 0

    @property
    def abilities(self) -> list[str]:
        return ["chat", "tool_calling"]

    @property
    def model_name(self) -> str:
        return self._model_id

    @property
    def supports_thinking_mode(self) -> bool:
        return False

    @property
    def supports_forced_tool_choice(self) -> bool:
        return self.forced_tool_choice

    async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        del messages
        self.tool_choices.append(kwargs.get("tool_choice"))
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self._model_id

    async def vision_chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        return await self.chat(messages, **kwargs)

    async def stream_chat(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[StreamChunk]:
        del messages, kwargs
        self.calls += 1
        if self.error is not None and not self.error_after_chunks:
            raise self.error
        try:
            for chunk in self.chunks:
                yield chunk
            if self.error is not None:
                raise self.error
        finally:
            self.stream_closed = True


@pytest.fixture(autouse=True)
def _fresh_cooldowns():
    router_module._CANDIDATE_COOLDOWNS.clear()
    yield
    router_module._CANDIDATE_COOLDOWNS.clear()


def _router(
    monkeypatch: pytest.MonkeyPatch,
    models: dict[str, _FakeLLM],
    *,
    selected: str = PRIMARY,
    fallback: str | None = FALLBACK,
    configured: bool = True,
) -> RouterLLM:
    resolutions: list[tuple[str, bool]] = []
    selections: list[str] = []

    def resolve(profile: str, *, limit_attempts: bool = False) -> BaseLLM:
        resolutions.append((profile, limit_attempts))
        return models[profile]

    router = RouterLLM(
        downstream_resolver=resolve,
        candidate_models=list(models) if configured else None,
        fallback_model=fallback,
        use_environment_fallback=False,
    )

    async def select_model(prompt: str, **_kwargs: Any) -> str:
        selections.append(prompt)
        return selected

    monkeypatch.setattr(router, "_select_model", select_model)
    monkeypatch.setattr(router, "_profile_context_window", lambda _model: None)
    router.resolutions = resolutions  # type: ignore[attr-defined]
    router.selections = selections  # type: ignore[attr-defined]
    return router


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        openai.APIConnectionError(request=_REQUEST),
        openai.APITimeoutError(request=_REQUEST),
        _status_error(openai.InternalServerError, 503),
        _status_error(openai.RateLimitError, 429),
        _status_error(openai.AuthenticationError, 401),
        _status_error(openai.NotFoundError, 404),
        _status_error(openai.PermissionDeniedError, 403),
        _status_error(openai.APIStatusError, 402),
        RuntimeError("DeepSeek request failed"),
        ProviderCallError("provider_unavailable"),
    ],
    ids=[
        "connection",
        "timeout",
        "503",
        "429",
        "401",
        "404",
        "403",
        "402",
        "wrapped-connection",
        "guarded-unavailable",
    ],
)
async def test_availability_failure_fails_over_to_fallback(monkeypatch, error):
    if type(error) is RuntimeError:
        # Adapters re-raise provider errors wrapped in a RuntimeError.
        error.__cause__ = openai.APIConnectionError(request=_REQUEST)
    primary = _FakeLLM("saved-deepseek", error=error)
    fallback = _FakeLLM("saved-gpt")
    router = _router(monkeypatch, {PRIMARY: primary, FALLBACK: fallback})

    prepared = await router.prepare_for_call(MESSAGES)

    assert await prepared.chat(MESSAGES) == "saved-gpt"
    assert (primary.calls, fallback.calls) == (1, 1)
    # Usage and later calls on this selection belong to the fallback now.
    assert prepared.model_name == FALLBACK
    assert prepared.model_id == "saved-gpt"
    assert await prepared.chat(MESSAGES) == "saved-gpt"
    assert (primary.calls, fallback.calls) == (1, 2)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("This model's maximum context length is 128000 tokens"),
        _status_error(openai.BadRequestError, 400),
        LLMToolProtocolError(
            provider="deepseek",
            code="malformed_tool_arguments",
            message="bad tool arguments",
        ),
        RuntimeError("unexpected response shape"),
    ],
    ids=["context-length", "400", "tool-protocol", "unrecognized"],
)
async def test_request_failure_does_not_fail_over(monkeypatch, error):
    primary = _FakeLLM("saved-deepseek", error=error)
    fallback = _FakeLLM("saved-gpt")
    router = _router(monkeypatch, {PRIMARY: primary, FALLBACK: fallback})

    prepared = await router.prepare_for_call(MESSAGES)

    with pytest.raises(type(error)) as raised:
        await prepared.chat(MESSAGES)
    assert raised.value is error
    assert fallback.calls == 0
    assert not router_module._CANDIDATE_COOLDOWNS.active("saved-deepseek")


@pytest.mark.asyncio
async def test_failed_candidate_is_skipped_until_its_cooldown_ends(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(router_module, "_now", lambda: now[0])
    monkeypatch.setenv("XAGENT_AUTO_CANDIDATE_COOLDOWN_SECONDS", "60")
    primary = _FakeLLM(
        "saved-deepseek", error=openai.APIConnectionError(request=_REQUEST)
    )
    fallback = _FakeLLM("saved-gpt")
    router = _router(monkeypatch, {PRIMARY: primary, FALLBACK: fallback})

    first = await router.prepare_for_call(MESSAGES)
    assert await first.chat(MESSAGES) == "saved-gpt"
    assert primary.calls == 1

    now[0] += 59
    during = await router.prepare_for_call(MESSAGES)
    assert during.model_name == FALLBACK
    assert await during.chat(MESSAGES) == "saved-gpt"
    assert primary.calls == 1

    now[0] += 2
    primary.error = None
    after = await router.prepare_for_call(MESSAGES)
    assert after.model_name == PRIMARY
    assert await after.chat(MESSAGES) == "saved-deepseek"
    assert primary.calls == 2


@pytest.mark.asyncio
async def test_legacy_auto_fails_over_without_a_shared_cooldown(monkeypatch):
    primary = _FakeLLM(
        "router:deepseek", error=openai.APIConnectionError(request=_REQUEST)
    )
    fallback = _FakeLLM("router:gpt")
    router = _router(
        monkeypatch, {PRIMARY: primary, FALLBACK: fallback}, configured=False
    )

    assert await (await router.prepare_for_call(MESSAGES)).chat(MESSAGES) == (
        "router:gpt"
    )
    # Legacy children share ids across callers with different keys, so one
    # caller's failure must not reroute anyone else.
    second = await router.prepare_for_call(MESSAGES)
    assert second.model_name == PRIMARY
    assert primary.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("selected", "fallback"),
    [(PRIMARY, None), (FALLBACK, FALLBACK)],
    ids=["no-fallback", "selected-is-fallback"],
)
async def test_nothing_to_fail_over_to_raises(monkeypatch, selected, fallback):
    error = openai.APIConnectionError(request=_REQUEST)
    models = {
        PRIMARY: _FakeLLM("saved-deepseek", error=error),
        FALLBACK: _FakeLLM("saved-gpt", error=error),
    }
    router = _router(monkeypatch, models, selected=selected, fallback=fallback)

    with pytest.raises(openai.APIConnectionError):
        await (await router.prepare_for_call(MESSAGES)).chat(MESSAGES)
    assert sum(model.calls for model in models.values()) == 1


@pytest.mark.asyncio
async def test_fallback_without_a_required_modality_is_not_used(monkeypatch):
    error = openai.APIConnectionError(request=_REQUEST)
    primary = _FakeLLM("saved-deepseek", error=error)
    fallback = _FakeLLM("saved-gpt")
    router = _router(monkeypatch, {PRIMARY: primary, FALLBACK: fallback})
    monkeypatch.setattr(
        router,
        "_profile_input_modalities",
        lambda model: ("text", "image") if model == PRIMARY else ("text",),
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
            ],
        }
    ]

    with pytest.raises(openai.APIConnectionError):
        await (await router.prepare_for_call(messages)).chat(messages)
    assert fallback.calls == 0


@pytest.mark.asyncio
async def test_fallback_failure_is_reported_without_the_first_error(monkeypatch):
    primary = _FakeLLM(
        "saved-deepseek", error=openai.APIConnectionError(request=_REQUEST)
    )
    fallback_error = _status_error(openai.InternalServerError, 503)
    fallback = _FakeLLM("saved-gpt", error=fallback_error)
    router = _router(monkeypatch, {PRIMARY: primary, FALLBACK: fallback})

    with pytest.raises(openai.InternalServerError) as raised:
        await (await router.prepare_for_call(MESSAGES)).chat(MESSAGES)
    assert raised.value is fallback_error
    assert raised.value.__context__ is None


@pytest.mark.asyncio
async def test_fallback_that_cannot_be_built_reports_the_first_error(monkeypatch):
    error = openai.APIConnectionError(request=_REQUEST)
    primary = _FakeLLM("saved-deepseek", error=error)

    def resolve(profile: str, **_kwargs: Any) -> BaseLLM:
        if profile == FALLBACK:
            raise RuntimeError("fallback endpoint is misconfigured")
        return primary

    router = _router(monkeypatch, {PRIMARY: primary, FALLBACK: primary})
    router._downstream_resolver = resolve

    with pytest.raises(openai.APIConnectionError) as raised:
        await (await router.prepare_for_call(MESSAGES)).chat(MESSAGES)
    assert raised.value is error


async def _collect(stream: AsyncIterator[StreamChunk]) -> list[str]:
    return [chunk.content async for chunk in stream]


@pytest.mark.asyncio
async def test_stream_fails_over_before_its_first_chunk(monkeypatch):
    primary = _FakeLLM(
        "saved-deepseek", error=_status_error(openai.InternalServerError, 502)
    )
    fallback = _FakeLLM("saved-gpt")
    router = _router(monkeypatch, {PRIMARY: primary, FALLBACK: fallback})

    prepared = await router.prepare_for_call(MESSAGES)

    assert await _collect(prepared.stream_chat(MESSAGES)) == ["saved-gpt"]
    assert prepared.model_id == "saved-gpt"


@pytest.mark.asyncio
async def test_stream_fails_over_on_a_leading_error_chunk(monkeypatch):
    error = openai.APIConnectionError(request=_REQUEST)
    primary = _FakeLLM(
        "saved-claude",
        chunks=[StreamChunk(type=ChunkType.ERROR, content="failed", raw=error)],
    )
    fallback = _FakeLLM("saved-gpt")
    router = _router(monkeypatch, {PRIMARY: primary, FALLBACK: fallback})

    prepared = await router.prepare_for_call(MESSAGES)

    assert await _collect(prepared.stream_chat(MESSAGES)) == ["saved-gpt"]


@pytest.mark.asyncio
async def test_stream_does_not_fail_over_after_its_first_chunk(monkeypatch):
    primary = _FakeLLM(
        "saved-deepseek",
        error=openai.APIConnectionError(request=_REQUEST),
        error_after_chunks=True,
    )
    fallback = _FakeLLM("saved-gpt")
    router = _router(monkeypatch, {PRIMARY: primary, FALLBACK: fallback})

    received: list[str] = []
    with pytest.raises(openai.APIConnectionError):
        async for chunk in (await router.prepare_for_call(MESSAGES)).stream_chat(
            MESSAGES
        ):
            received.append(chunk.content)
    assert received == ["saved-deepseek"]
    assert fallback.calls == 0


@pytest.mark.asyncio
async def test_rate_limit_fails_over_without_a_cooldown(monkeypatch):
    primary = _FakeLLM(
        "saved-deepseek", error=_status_error(openai.RateLimitError, 429)
    )
    fallback = _FakeLLM("saved-gpt")
    router = _router(monkeypatch, {PRIMARY: primary, FALLBACK: fallback})

    assert await (await router.prepare_for_call(MESSAGES)).chat(MESSAGES) == (
        "saved-gpt"
    )
    assert (await router.prepare_for_call(MESSAGES)).model_name == PRIMARY


@pytest.mark.asyncio
async def test_cooldown_keeps_the_candidate_when_the_fallback_cannot_be_built(
    monkeypatch,
):
    primary = _FakeLLM("saved-deepseek")
    router = _router(monkeypatch, {PRIMARY: primary, FALLBACK: primary})
    fallback_builds = [0]

    def resolve(profile: str, **_kwargs: Any) -> BaseLLM:
        if profile == FALLBACK:
            fallback_builds[0] += 1
            # Builds for the early give-up check, then fails when the
            # cooldown actually needs it.
            if fallback_builds[0] > 1:
                raise RuntimeError("fallback endpoint is misconfigured")
        return primary

    resolutions: list[tuple[str, bool]] = []

    def recording(profile: str, *, limit_attempts: bool = False) -> BaseLLM:
        resolutions.append((profile, limit_attempts))
        return resolve(profile)

    router._downstream_resolver = recording
    router_module._CANDIDATE_COOLDOWNS.start("saved-deepseek", 60)

    prepared = await router.prepare_for_call(MESSAGES)

    assert prepared.model_name == PRIMARY
    assert await prepared.chat(MESSAGES) == "saved-deepseek"
    # With no fallback to move to after all, the candidate is rebuilt with its
    # full retry budget.
    assert (PRIMARY, True) in resolutions
    assert resolutions[-1] == (PRIMARY, False)


IMAGE_MESSAGES = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
        ],
    }
]


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback", [None, PRIMARY])
async def test_sole_candidate_skips_routing_but_keeps_its_context_window(
    monkeypatch, fallback
):
    only = _FakeLLM("saved-deepseek")
    router = _router(monkeypatch, {PRIMARY: only}, fallback=fallback)
    # The saved model has no window, so the profile catalog supplies it, as it
    # does for a routed call.
    monkeypatch.setattr(router, "_profile_context_window", lambda _model: 64_000)

    prepared = await router.prepare_for_call(MESSAGES)

    assert router.selections == []
    assert prepared.model_name == PRIMARY
    assert prepared.context_window == 64_000
    assert await prepared.chat(MESSAGES) == "saved-deepseek"
    assert router.resolutions == [(PRIMARY, False)]


@pytest.mark.asyncio
async def test_excluded_sole_candidate_still_goes_through_routing(monkeypatch):
    monkeypatch.setenv("XAGENT_XROUTER_EXCLUDED_MODELS", PRIMARY)
    router = _router(monkeypatch, {PRIMARY: _FakeLLM("saved-deepseek")}, fallback=None)

    await router.prepare_for_call(MESSAGES)

    # Routing is where the exclusion is enforced (or the fallback applied).
    assert router.selections == ["summarize the report"]


@pytest.mark.asyncio
async def test_sole_candidate_with_required_modalities_goes_through_routing(
    monkeypatch,
):
    router = _router(monkeypatch, {PRIMARY: _FakeLLM("saved-deepseek")}, fallback=None)
    monkeypatch.setattr(router, "_profile_input_modalities", lambda _model: ())

    await router.prepare_for_call(IMAGE_MESSAGES)

    assert len(router.selections) == 1


@pytest.mark.asyncio
async def test_candidate_retry_budget_is_limited_only_when_the_call_can_fail_over(
    monkeypatch,
):
    primary = _FakeLLM(
        "saved-deepseek", error=openai.APIConnectionError(request=_REQUEST)
    )
    fallback = _FakeLLM("saved-gpt")
    router = _router(monkeypatch, {PRIMARY: primary, FALLBACK: fallback})
    monkeypatch.setattr(
        router,
        "_profile_input_modalities",
        lambda model: ("text", "image") if model == PRIMARY else ("text",),
    )

    # A text call can move to the fallback: the candidate may give up early,
    # and the fallback it moves to keeps its full budget.
    assert await (await router.prepare_for_call(MESSAGES)).chat(MESSAGES) == (
        "saved-gpt"
    )
    assert (PRIMARY, True) in router.resolutions
    assert router.resolutions[-1] == (FALLBACK, False)

    # The text-only fallback cannot serve an image call, so the candidate
    # keeps every saved attempt.
    router_module._CANDIDATE_COOLDOWNS.clear()
    router.resolutions.clear()
    await router.prepare_for_call(IMAGE_MESSAGES)
    assert (PRIMARY, True) not in router.resolutions


@pytest.mark.asyncio
async def test_candidate_without_a_fallback_keeps_its_retry_budget(monkeypatch):
    models = {PRIMARY: _FakeLLM("saved-deepseek"), FALLBACK: _FakeLLM("saved-gpt")}
    router = _router(monkeypatch, models, fallback=None)

    await router.prepare_for_call(MESSAGES)

    assert router.resolutions == [(PRIMARY, False)]


@pytest.mark.asyncio
async def test_vision_call_fails_over_to_fallback(monkeypatch):
    primary = _FakeLLM(
        "saved-deepseek", error=_status_error(openai.InternalServerError, 503)
    )
    fallback = _FakeLLM("saved-gpt")
    router = _router(monkeypatch, {PRIMARY: primary, FALLBACK: fallback})

    prepared = await router.prepare_for_call(MESSAGES)

    assert await prepared.vision_chat(MESSAGES) == "saved-gpt"


@pytest.mark.asyncio
async def test_cancellation_is_not_failed_over(monkeypatch):
    primary = _FakeLLM("saved-deepseek", error=asyncio.CancelledError())
    fallback = _FakeLLM("saved-gpt")
    router = _router(monkeypatch, {PRIMARY: primary, FALLBACK: fallback})

    with pytest.raises(asyncio.CancelledError):
        await (await router.prepare_for_call(MESSAGES)).chat(MESSAGES)
    assert fallback.calls == 0
    assert not router_module._CANDIDATE_COOLDOWNS.active("saved-deepseek")


@pytest.mark.asyncio
async def test_failover_adopts_the_fallback_context_window(monkeypatch):
    primary = _FakeLLM(
        "saved-deepseek",
        error=openai.APIConnectionError(request=_REQUEST),
        context_window=128_000,
    )
    fallback = _FakeLLM("saved-gpt", context_window=400_000)
    router = _router(monkeypatch, {PRIMARY: primary, FALLBACK: fallback})

    prepared = await router.prepare_for_call(MESSAGES)
    assert prepared.context_window == 128_000

    assert await prepared.chat(MESSAGES) == "saved-gpt"
    assert prepared.context_window == 400_000


@pytest.mark.asyncio
async def test_smaller_fallback_is_not_used_for_the_call_but_takes_the_next(
    monkeypatch,
):
    error = openai.APIConnectionError(request=_REQUEST)
    primary = _FakeLLM("saved-deepseek", error=error, context_window=128_000)
    fallback = _FakeLLM("saved-gpt", context_window=32_000)
    router = _router(monkeypatch, {PRIMARY: primary, FALLBACK: fallback})

    prepared = await router.prepare_for_call(MESSAGES)

    # The transcript was sized for 128k; a 32k fallback could only end the
    # task with a context-length failure, so the resumable error stands.
    with pytest.raises(openai.APIConnectionError) as raised:
        await prepared.chat(MESSAGES)
    assert raised.value is error
    assert fallback.calls == 0
    assert prepared.model_id == "saved-deepseek"

    # The cooldown still sends the next call to the fallback, where it is
    # prepared (and compacted) for the fallback's own window.
    following = await router.prepare_for_call(MESSAGES)
    assert following.model_name == FALLBACK
    assert following.context_window == 32_000


@pytest.mark.asyncio
async def test_failed_candidate_stream_is_closed_before_failing_over(monkeypatch):
    error = openai.APIConnectionError(request=_REQUEST)
    primary = _FakeLLM(
        "saved-claude",
        chunks=[StreamChunk(type=ChunkType.ERROR, content="failed", raw=error)],
    )
    fallback = _FakeLLM("saved-gpt")
    router = _router(monkeypatch, {PRIMARY: primary, FALLBACK: fallback})

    prepared = await router.prepare_for_call(MESSAGES)

    assert await _collect(prepared.stream_chat(MESSAGES)) == ["saved-gpt"]
    assert primary.stream_closed


class _UnclosableStream:
    def __init__(self, chunks: list[StreamChunk]) -> None:
        self._chunks = iter(chunks)

    def __aiter__(self) -> "_UnclosableStream":
        return self

    async def __anext__(self) -> StreamChunk:
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None

    async def aclose(self) -> None:
        raise RuntimeError("connection already torn down")


@pytest.mark.asyncio
async def test_stream_close_failure_does_not_abort_the_failover(monkeypatch):
    error = openai.APIConnectionError(request=_REQUEST)
    primary = _FakeLLM("saved-claude")
    monkeypatch.setattr(
        primary,
        "stream_chat",
        lambda *_args, **_kwargs: _UnclosableStream(
            [StreamChunk(type=ChunkType.ERROR, content="failed", raw=error)]
        ),
    )
    fallback = _FakeLLM("saved-gpt")
    router = _router(monkeypatch, {PRIMARY: primary, FALLBACK: fallback})

    prepared = await router.prepare_for_call(MESSAGES)

    assert await _collect(prepared.stream_chat(MESSAGES)) == ["saved-gpt"]


@pytest.mark.asyncio
async def test_candidate_retries_only_failover_failures_briefly(monkeypatch):
    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("xagent.core.retry.wrapper.asyncio.sleep", no_sleep)

    class _Flaky(_FakeLLM):
        def __init__(self, failures: list[BaseException]) -> None:
            super().__init__("saved-deepseek")
            self.failures = list(failures)

        async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
            self.calls += 1
            if self.failures:
                raise self.failures.pop(0)
            return "recovered"

    # Failures the fallback would not take over keep the saved budget...
    flaky = _Flaky([LLMEmptyContentError("empty")] * 3)
    llm = attach_chat_retry_wrapper(flaky, 5, failover_max_attempts=2)
    assert await llm.chat(MESSAGES) == "recovered"
    assert flaky.calls == 4

    # ...while an outage stops after two attempts so the call can move on.
    down = _Flaky([openai.APIConnectionError(request=_REQUEST)] * 3)
    llm = attach_chat_retry_wrapper(down, 5, failover_max_attempts=2)
    with pytest.raises(openai.APIConnectionError):
        await llm.chat(MESSAGES)
    assert down.calls == 2


@pytest.mark.asyncio
async def test_candidate_keeps_its_retry_budget_when_the_fallback_is_smaller(
    monkeypatch,
):
    models = {
        PRIMARY: _FakeLLM("saved-deepseek", context_window=128_000),
        FALLBACK: _FakeLLM("saved-gpt", context_window=32_000),
    }
    router = _router(monkeypatch, models)

    await router.prepare_for_call(MESSAGES)

    # The call would not move to a smaller fallback, so it must not give up
    # early either.
    assert (PRIMARY, True) not in router.resolutions


@pytest.mark.asyncio
async def test_failure_after_stream_output_still_starts_the_cooldown(monkeypatch):
    error = openai.APIConnectionError(request=_REQUEST)
    primary = _FakeLLM("saved-deepseek", error=error, error_after_chunks=True)
    fallback = _FakeLLM("saved-gpt")
    router = _router(monkeypatch, {PRIMARY: primary, FALLBACK: fallback})

    received: list[str] = []
    with pytest.raises(openai.APIConnectionError) as raised:
        async for chunk in (await router.prepare_for_call(MESSAGES)).stream_chat(
            MESSAGES
        ):
            received.append(chunk.content)

    # The partial answer is not replayed and the error stands...
    assert raised.value is error
    assert received == ["saved-deepseek"]
    assert fallback.calls == 0
    # ...but the next independent call no longer lands on the failing candidate.
    following = await router.prepare_for_call(MESSAGES)
    assert following.model_name == FALLBACK


FORCED = {"type": "function", "function": {"name": "final_answer"}}


@pytest.mark.asyncio
async def test_forced_choice_support_reflects_every_model_the_call_used(
    monkeypatch,
):
    # The candidate cannot force a tool, so callers send "auto"; after moving
    # to a fallback that could, the call still did not force one.
    primary = _FakeLLM(
        "saved-deepseek",
        error=openai.APIConnectionError(request=_REQUEST),
        forced_tool_choice=False,
    )
    fallback = _FakeLLM("saved-gpt", forced_tool_choice=True)
    router = _router(monkeypatch, {PRIMARY: primary, FALLBACK: fallback})
    prepared = await router.prepare_for_call(MESSAGES)
    assert not prepared.supports_forced_tool_choice

    await prepared.chat(MESSAGES, tool_choice="auto")

    assert fallback.tool_choices == ["auto"]
    assert not prepared.supports_forced_tool_choice


@pytest.mark.asyncio
async def test_forced_choice_is_adapted_for_a_fallback_that_cannot_force(
    monkeypatch,
):
    primary = _FakeLLM(
        "saved-deepseek",
        error=openai.APIConnectionError(request=_REQUEST),
        forced_tool_choice=True,
    )
    fallback = _FakeLLM("saved-gpt", forced_tool_choice=False)
    router = _router(monkeypatch, {PRIMARY: primary, FALLBACK: fallback})
    prepared = await router.prepare_for_call(MESSAGES)
    assert prepared.supports_forced_tool_choice

    await prepared.chat(MESSAGES, tool_choice=FORCED)

    assert primary.tool_choices == [FORCED]
    assert fallback.tool_choices == ["auto"]
    assert not prepared.supports_forced_tool_choice


def _affordability_refusal() -> Exception:
    # OpenRouter's 402 for a request whose output budget the account cannot
    # cover; the runtime's compaction ladder recovers by asking for less.
    return openai.APIStatusError(
        "This request requires more credits, or fewer max_tokens. "
        "You requested up to 8000 tokens, but can only afford 1500.",
        response=httpx.Response(402, request=_REQUEST),
        body=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "refusal",
    [
        _affordability_refusal(),
        ProviderCallError("provider_quota", budget_sensitive=True),
    ],
    ids=["raw", "guarded"],
)
async def test_output_budget_refusal_is_not_hidden_by_an_unavailable_fallback(
    monkeypatch, refusal
):
    primary = _FakeLLM("saved-deepseek", error=refusal)
    fallback = _FakeLLM("saved-gpt", error=openai.APIConnectionError(request=_REQUEST))
    router = _router(monkeypatch, {PRIMARY: primary, FALLBACK: fallback})

    with pytest.raises(type(refusal)) as raised:
        await (await router.prepare_for_call(MESSAGES)).chat(MESSAGES, max_tokens=8000)

    # The caller still sees the budget refusal, so it can retry with less.
    assert raised.value is refusal
    assert fallback.calls == 0
    assert not router_module._CANDIDATE_COOLDOWNS.active("saved-deepseek")
    assert not router_module.is_auto_failover_failure(refusal)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exhausted",
    [
        openai.APIStatusError(
            "insufficient_quota: you have run out of credits",
            response=httpx.Response(402, request=_REQUEST),
            body=None,
        ),
        ProviderCallError("provider_quota"),
    ],
    ids=["raw", "guarded"],
)
async def test_exhausted_quota_still_fails_over(monkeypatch, exhausted):
    primary = _FakeLLM("saved-deepseek", error=exhausted)
    fallback = _FakeLLM("saved-gpt")
    router = _router(monkeypatch, {PRIMARY: primary, FALLBACK: fallback})

    assert await (await router.prepare_for_call(MESSAGES)).chat(MESSAGES) == (
        "saved-gpt"
    )
    assert router_module._CANDIDATE_COOLDOWNS.active("saved-deepseek")


@pytest.mark.asyncio
async def test_error_chunk_after_output_still_starts_the_cooldown(monkeypatch):
    # Claude yields a transport failure that arrives mid-response as an ERROR
    # chunk carrying the raw exception.
    error = httpx.RemoteProtocolError("peer closed connection")
    primary = _FakeLLM(
        "saved-claude",
        chunks=[
            StreamChunk(type=ChunkType.TOKEN, content="partial", delta="partial"),
            StreamChunk(type=ChunkType.ERROR, content="failed", raw=error),
        ],
    )
    fallback = _FakeLLM("saved-gpt")
    router = _router(monkeypatch, {PRIMARY: primary, FALLBACK: fallback})

    chunks = [
        chunk
        async for chunk in (await router.prepare_for_call(MESSAGES)).stream_chat(
            MESSAGES
        )
    ]

    # The current response is not replayed: the caller sees what the
    # candidate produced, error included...
    assert [chunk.type for chunk in chunks] == [ChunkType.TOKEN, ChunkType.ERROR]
    assert fallback.calls == 0
    # ...and the next preparation goes to the fallback.
    following = await router.prepare_for_call(MESSAGES)
    assert following.model_name == FALLBACK
