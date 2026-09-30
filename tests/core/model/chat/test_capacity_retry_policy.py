"""Retry policy for provider capacity refusals lives in exactly one layer.

Covers the three acceptance criteria of #2605: a capacity refusal fails within
a bounded wall clock, no provider SDK adds a retry budget inside ours, and
existing transient-fault retries keep working.
"""

import time
from unittest.mock import AsyncMock

import httpx
import openai
import pytest

from xagent.core.model import ChatModelConfig
from xagent.core.model.chat.basic.adapter import create_base_llm
from xagent.core.model.chat.basic.azure_openai import AzureOpenAILLM
from xagent.core.model.chat.basic.claude import ClaudeLLM
from xagent.core.model.chat.basic.openai import OpenAICompatibleLLM
from xagent.core.model.chat.basic.zhipu import ZhipuLLM
from xagent.core.model.chat.error import retry_on
from xagent.core.retry.policy import RetryBudget
from xagent.core.retry.strategy import FixedDelay
from xagent.core.retry.wrapper import RetryWrapper

REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
CAPACITY_BODY = (
    "Exceeded on-demand capacity. Ensure your workload does not double "
    "faster than once per 30 minutes."
)


def _sdk_error(cls, status: int, message: str) -> openai.APIStatusError:
    return cls(message, response=httpx.Response(status, request=REQUEST), body=None)


def _as_adapter_raises(error: BaseException, prefix: str) -> RuntimeError:
    """Mirror how ``OpenAICompatibleLLM.chat`` re-raises a provider failure."""
    wrapped = RuntimeError(f"{prefix}: {error}")
    wrapped.__cause__ = error
    return wrapped


class TestNoNestedRetryBudget:
    """One provider request per our attempt: the SDK must not retry inside us."""

    def _captured_kwargs(self, monkeypatch, target: str) -> dict:
        captured: dict = {}

        def construct(**kwargs):
            captured.update(kwargs)
            return AsyncMock()

        monkeypatch.setattr(target, construct)
        return captured

    def test_openai_compatible_client_disables_sdk_retries(self, monkeypatch):
        captured = self._captured_kwargs(
            monkeypatch, "xagent.core.model.chat.basic.openai.AsyncOpenAI"
        )
        llm = OpenAICompatibleLLM("m", base_url=None, api_key="k")

        llm._ensure_client()

        assert captured["max_retries"] == 0

    def test_azure_client_disables_sdk_retries(self, monkeypatch):
        captured = self._captured_kwargs(
            monkeypatch, "xagent.core.model.chat.basic.azure_openai.AsyncAzureOpenAI"
        )
        llm = AzureOpenAILLM(
            "m",
            azure_endpoint="https://example.openai.azure.com",
            api_key="k",
        )

        llm._ensure_client()

        assert captured["max_retries"] == 0

    def test_claude_client_disables_sdk_retries(self, monkeypatch):
        captured = self._captured_kwargs(
            monkeypatch, "xagent.core.model.chat.basic.claude.AsyncAnthropic"
        )
        llm = ClaudeLLM("m", base_url=None, api_key="k")

        llm._ensure_client()

        assert captured["max_retries"] == 0

    def test_zhipu_client_disables_sdk_retries(self):
        """The zai SDK defaults to three retries, not two."""
        pytest.importorskip("zai")
        captured = {}

        def construct(**kwargs):
            captured.update(kwargs)
            return AsyncMock()

        import xagent.core.model.chat.basic.zhipu as zhipu_module

        original = zhipu_module.ZhipuAiClient
        zhipu_module.ZhipuAiClient = construct
        try:
            ZhipuLLM("m", base_url=None, api_key="k")._ensure_client()
        finally:
            zhipu_module.ZhipuAiClient = original

        assert captured["max_retries"] == 0

    async def test_one_shot_model_listing_keeps_its_own_retries(self, monkeypatch):
        """``list_available_models`` is not inside our wrapper.

        It runs one request with no outer retry layer, so the SDK's budget is
        the only resilience it has and must be left alone. Asserted on the
        kwargs the constructor actually receives, not on the source text.
        """
        captured = self._captured_kwargs(
            monkeypatch, "xagent.core.model.chat.basic.openai.AsyncOpenAI"
        )

        await OpenAICompatibleLLM.list_available_models("sk-test")

        assert captured, "the SDK client was never constructed"
        assert "max_retries" not in captured


class TestTransientFaultsUnchanged:
    """Removing the SDK's retries must not remove retries for real blips."""

    @pytest.mark.parametrize(
        "error",
        [
            openai.APIConnectionError(request=REQUEST),
            openai.APITimeoutError(request=REQUEST),
            httpx.ConnectError("refused"),
            httpx.ReadTimeout("slow"),
        ],
    )
    def test_network_faults_stay_retryable(self, error):
        assert retry_on(error) is True

    def test_rate_limit_stays_retryable(self):
        error = _sdk_error(openai.RateLimitError, 429, "Rate limit reached")

        assert retry_on(error) is True
        assert retry_on(_as_adapter_raises(error, "OpenAI rate limit exceeded")) is True

    @pytest.mark.parametrize("status", [500, 502, 503, 504])
    def test_provider_5xx_stays_retryable(self, status):
        """The SDK used to be the only layer retrying these; now we are.

        Before #2605 ``retry_on`` returned ``False`` for an openai-SDK 5xx --
        it matched only ``httpx.HTTPStatusError``, which the adapters never let
        through. Setting ``max_retries=0`` on the SDK client would otherwise
        have deleted 5xx retries outright.
        """
        error = _sdk_error(openai.InternalServerError, status, "Bad gateway")

        assert retry_on(error) is True
        assert retry_on(_as_adapter_raises(error, "OpenAI API error")) is True

    @pytest.mark.parametrize(
        "error",
        [
            _sdk_error(openai.BadRequestError, 400, "bad input"),
            _sdk_error(openai.AuthenticationError, 401, "bad key"),
            _sdk_error(openai.PermissionDeniedError, 403, "no access"),
            _sdk_error(openai.NotFoundError, 404, "no model"),
            _sdk_error(openai.UnprocessableEntityError, 422, "nope"),
        ],
    )
    def test_permanent_failures_stay_non_retryable(self, error):
        assert retry_on(error) is False
        assert retry_on(_as_adapter_raises(error, "OpenAI API error")) is False

    def test_an_unusable_status_does_not_crash_the_predicate(self, mocker):
        """A status we cannot read must fall through, never raise.

        ``retry_on`` runs inside the ``except`` arm of every LLM call, so an
        exception escaping it replaces a retryable provider error with an
        unrelated crash.
        """
        error = openai.RateLimitError("rate limited", response=mocker.Mock(), body=None)

        assert retry_on(error) is True

    def test_context_length_stays_non_retryable(self):
        error = _sdk_error(
            openai.BadRequestError, 400, "This model's maximum context length is 8192"
        )

        assert retry_on(error) is False


class TestCapacityRefusalIsBounded:
    def test_capacity_refusal_is_retryable_but_not_generously(self):
        """A capacity refusal arrives on a retryable path and stays on it.

        Classification only lowers the budget; it must not start retrying a
        refusal shape that ``retry_on`` rejects today.
        """
        error = _sdk_error(openai.RateLimitError, 429, CAPACITY_BODY)
        wrapped = _as_adapter_raises(error, "OpenAI rate limit exceeded")

        assert retry_on(wrapped) is True
        assert RetryBudget(capacity_max_attempts=2).attempt_limit(wrapped) == 2

    def test_a_capacity_refusal_on_a_permanent_status_is_still_permanent(self):
        error = _sdk_error(openai.BadRequestError, 400, CAPACITY_BODY)

        assert retry_on(_as_adapter_raises(error, "OpenAI bad request")) is False

    async def test_capacity_refusal_fails_the_turn_within_the_deadline(self):
        """The attempt half of the bound, measured on the real clock.

        What stops this loop is ``capacity_max_attempts``, not the deadline --
        two fast attempts never reach 5s. The deadline's own proof is the
        test below, which gives the refusal no attempt limit to hit.
        """
        calls = 0

        class OverCapacity:
            async def ainvoke(self, *args, **kwargs):
                nonlocal calls
                calls += 1
                raise _as_adapter_raises(
                    _sdk_error(openai.RateLimitError, 429, CAPACITY_BODY),
                    "OpenAI rate limit exceeded",
                )

        wrapper = RetryWrapper(
            OverCapacity(),
            strategy=FixedDelay(delay_ms=10),
            max_retries=10,
            retry_on=retry_on,
            budget=RetryBudget(deadline_seconds=5.0, capacity_max_attempts=2),
        )

        started = time.monotonic()
        with pytest.raises(RuntimeError, match="on-demand capacity"):
            await wrapper.ainvoke()
        elapsed = time.monotonic() - started

        assert calls == 2
        assert elapsed < 1.0

    async def test_an_unclassified_stall_is_still_bounded_by_the_deadline(self):
        """The deadline is the safety net for markers we do not recognize."""
        calls = 0

        class SlowFailure:
            async def ainvoke(self, *args, **kwargs):
                nonlocal calls
                calls += 1
                raise openai.APIConnectionError(request=REQUEST)

        wrapper = RetryWrapper(
            SlowFailure(),
            strategy=FixedDelay(delay_ms=60),
            max_retries=1000,
            retry_on=retry_on,
            budget=RetryBudget(deadline_seconds=0.3),
        )

        started = time.monotonic()
        with pytest.raises(openai.APIConnectionError):
            await wrapper.ainvoke()
        elapsed = time.monotonic() - started

        assert elapsed < 2.0
        assert calls < 1000


class TestAdapterWiring:
    def test_chat_models_are_built_with_a_bounded_budget(self, monkeypatch):
        monkeypatch.delenv("XAGENT_LLM_RETRY_DEADLINE_SECONDS", raising=False)
        monkeypatch.delenv("XAGENT_LLM_CAPACITY_MAX_ATTEMPTS", raising=False)
        config = ChatModelConfig(
            id="m1",
            model_name="gpt-4o",
            model_provider="openai",
            api_key="k",
            base_url=None,
        )

        llm = create_base_llm(config)

        budget = llm._retry_wrapper.budget
        assert budget is not None
        assert budget.deadline_seconds is not None
        assert budget.capacity_max_attempts == 2

    def test_the_deployment_can_retune_the_deadline(self, monkeypatch):
        monkeypatch.setenv("XAGENT_LLM_RETRY_DEADLINE_SECONDS", "90")
        config = ChatModelConfig(
            id="m1",
            model_name="gpt-4o",
            model_provider="openai",
            api_key="k",
            base_url=None,
        )

        llm = create_base_llm(config)

        assert llm._retry_wrapper.budget.deadline_seconds == pytest.approx(90.0)


class TestStreamingTransientFaultsStillReachTheWrapper:
    """Acceptance 3 on the streaming paths, where the SDK was the only layer.

    ``stream_chat`` on claude and zhipu used to swallow provider failures into
    an ERROR chunk, so the shared RetryWrapper never saw one and each SDK's own
    budget was the only cover a streaming call had. Setting ``max_retries=0``
    removes that budget, so a transient streaming failure has to reach our
    layer or it loses its retries outright.
    """

    async def _drain(self, llm, messages):
        chunks = []
        async for chunk in llm.stream_chat(messages):
            chunks.append(chunk)
        return chunks

    async def test_claude_streaming_5xx_raises_for_the_wrapper(self, mocker):
        """A 529 Overloaded is exactly the capacity shape #2605 is about."""
        from anthropic import InternalServerError

        from xagent.core.model.chat.exceptions import LLMRetryableError

        llm = ClaudeLLM("claude-sonnet-4", base_url=None, api_key="k")
        client = mocker.AsyncMock()
        client.messages.create.side_effect = InternalServerError(
            "Overloaded",
            response=httpx.Response(529, request=REQUEST),
            body=None,
        )
        llm._client = client

        with pytest.raises(LLMRetryableError, match="Overloaded"):
            await self._drain(llm, [{"role": "user", "content": "hi"}])

    async def test_claude_streaming_permanent_status_keeps_the_error_chunk(
        self, mocker
    ):
        """A 400 cannot be helped by replaying it; do not start retrying it."""
        from anthropic import BadRequestError

        from xagent.core.model.chat.types import ChunkType

        llm = ClaudeLLM("claude-sonnet-4", base_url=None, api_key="k")
        client = mocker.AsyncMock()
        client.messages.create.side_effect = BadRequestError(
            "bad input",
            response=httpx.Response(400, request=REQUEST),
            body=None,
        )
        llm._client = client

        chunks = await self._drain(llm, [{"role": "user", "content": "hi"}])

        assert [c.type for c in chunks] == [ChunkType.ERROR]

    async def test_zhipu_streaming_5xx_raises_for_the_wrapper(self, mocker):
        pytest.importorskip("zai")
        from zai.core._errors import APIStatusError as ZaiAPIStatusError

        from xagent.core.model.chat.basic.zhipu import ZhipuLLM
        from xagent.core.model.chat.exceptions import LLMRetryableError

        llm = ZhipuLLM("glm-4", base_url=None, api_key="k")
        client = mocker.MagicMock()
        client.chat.completions.create.side_effect = ZaiAPIStatusError(
            "service unavailable",
            response=httpx.Response(503, request=REQUEST),
        )
        llm._client = client

        with pytest.raises(LLMRetryableError):
            await self._drain(llm, [{"role": "user", "content": "hi"}])


class TestProviderRetryVetoAndTransientStatuses:
    """What the SDK budget carried besides an attempt count.

    Removing the SDK's retries removed three things with it: its
    ``Retry-After`` date handling, its ``x-should-retry`` header precedence,
    and its classification of 408/409. The first lives in
    ``core/retry/policy.py``; the other two are here.
    """

    def _status(self, cls, status: int, headers=None) -> openai.APIStatusError:
        return cls(
            "boom",
            response=httpx.Response(status, request=REQUEST, headers=headers or {}),
            body=None,
        )

    @pytest.mark.parametrize("status", [408, 409])
    def test_transient_request_statuses_are_retryable(self, status):
        """The locked SDKs retried these; with their budget gone we must."""
        error = self._status(openai.APIStatusError, status, None)

        assert retry_on(error) is True
        assert retry_on(_as_adapter_raises(error, "OpenAI API error")) is True

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_other_4xx_stay_permanent(self, status):
        error = self._status(openai.APIStatusError, status, None)

        assert retry_on(error) is False

    @pytest.mark.parametrize("status", [500, 503, 408, 409])
    def test_a_provider_veto_is_honoured(self, status):
        """``x-should-retry: false`` is the provider telling us not to."""
        error = self._status(openai.APIStatusError, status, {"x-should-retry": "false"})

        assert retry_on(error) is False

    def test_the_veto_does_not_change_the_pre_existing_429_path(self):
        """A vetoed 429 retried before this PR, through the ERRORS tuple.

        Narrowing that is a separate behaviour change, not a repair of what
        this PR introduced, so the veto must not reach it.
        """
        error = self._status(openai.RateLimitError, 429, {"x-should-retry": "false"})

        assert retry_on(error) is True

    def test_a_veto_on_a_permanent_status_stays_permanent(self):
        error = self._status(openai.APIStatusError, 400, {"x-should-retry": "false"})

        assert retry_on(error) is False

    @pytest.mark.parametrize("status", [408, 409])
    async def test_claude_streaming_transient_statuses_reach_the_wrapper(
        self, mocker, status
    ):
        """The streaming guard must agree with the predicate on every status."""
        from anthropic import APIStatusError as AnthropicStatusError

        from xagent.core.model.chat.exceptions import LLMRetryableError

        llm = ClaudeLLM("claude-sonnet-4", base_url=None, api_key="k")
        client = mocker.AsyncMock()
        client.messages.create.side_effect = AnthropicStatusError(
            "request timeout",
            response=httpx.Response(status, request=REQUEST),
            body=None,
        )
        llm._client = client

        with pytest.raises(LLMRetryableError):
            async for _ in llm.stream_chat([{"role": "user", "content": "hi"}]):
                pass
