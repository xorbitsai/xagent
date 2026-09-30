"""The safe error boundary around an LLM's provider calls.

Keys are synthetic. The real-SDK tests replace only the bottom httpx
transport, with a fake provider that echoes the received credential in its
error bodies -- the case the boundary exists for.
"""

import asyncio
import contextvars
import json
import logging
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, AsyncIterator, List

import httpx
import pytest
from google.genai import errors as genai_errors

from xagent.core.model.chat.basic import call_boundary as call_boundary_module
from xagent.core.model.chat.basic.adapter import create_base_llm
from xagent.core.model.chat.basic.base import BaseLLM
from xagent.core.model.chat.basic.call_boundary import (
    CALL_SCOPE_UNAVAILABLE,
    CONTEXT_LENGTH,
    CREDENTIAL_REJECTED,
    INVALID_REQUEST,
    MODEL_NOT_AVAILABLE,
    PROVIDER_ERROR,
    PROVIDER_QUOTA,
    PROVIDER_UNAVAILABLE,
    RATE_LIMITED,
    SAFE_PROVIDER_ERRORS,
    TIMEOUT,
    VISION_UNAVAILABLE,
    ProviderCallError,
    ProviderContextLengthError,
    UnavailableVisionModel,
    classify_provider_failure,
    guard_llm_calls,
)
from xagent.core.model.chat.error import is_context_length_error
from xagent.core.model.chat.exceptions import (
    LLMContextLengthError,
    LLMRetryableError,
    LLMToolProtocolError,
)
from xagent.core.model.chat.types import ChunkType, StreamChunk
from xagent.core.model.model import ChatModelConfig

SECRET = "sk-boundary-CALLER-SECRET-0001"
MESSAGES = [{"role": "user", "content": "hi"}]

in_scope: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "boundary_test_scope", default=False
)


class ScopeProbe:
    """A call scope that records how often it was entered and exited."""

    def __init__(self) -> None:
        self.entered = 0
        self.exited = 0

    @contextmanager
    def __call__(self):
        self.entered += 1
        token = in_scope.set(True)
        try:
            yield
        finally:
            in_scope.reset(token)
            self.exited += 1


class FailingScope:
    """A call scope that fails to enter, or fails while it is left, on the
    given (1-based) entry or exit numbers."""

    def __init__(
        self, *, enter_fails: set[int] = frozenset(), exit_fails: set[int] = frozenset()
    ) -> None:
        self.enter_fails = set(enter_fails)
        self.exit_fails = set(exit_fails)
        self.entered = 0
        self.exited = 0

    @contextmanager
    def __call__(self):
        self.entered += 1
        if self.entered in self.enter_fails:
            raise RuntimeError(f"scope could not open {SECRET}")
        token = in_scope.set(True)
        try:
            yield
        finally:
            in_scope.reset(token)
            self.exited += 1
            if self.exited in self.exit_fails:
                raise RuntimeError(f"scope could not close {SECRET}")


class ScopeRecorder(logging.Handler):
    """Captures each record with whether it was created inside the scope."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[tuple[logging.LogRecord, bool]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append((record, in_scope.get()))


@pytest.fixture
def recorder():
    handler = ScopeRecorder()
    root = logging.getLogger()
    saved = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        root.removeHandler(handler)
        root.setLevel(saved)


class FakeLLM(BaseLLM):
    """A scripted model: ``behaviour`` decides what each call does."""

    def __init__(self, behaviour: Any = "ok", abilities: list[str] | None = None):
        self.behaviour = behaviour
        self._abilities = abilities or ["chat", "tool_calling"]
        self.scope_during_calls: list[bool] = []
        self.closed = False
        self.close_in_scope: bool | None = None
        self.api_key = SECRET
        self.context_window = 12345
        self._model_id = "caller-model-id"

    @property
    def abilities(self) -> List[str]:
        return self._abilities

    @property
    def model_name(self) -> str:
        return "fake-model"

    @property
    def supports_thinking_mode(self) -> bool:
        return False

    async def chat(self, messages, **kwargs):  # type: ignore[override]
        self.scope_during_calls.append(in_scope.get())
        await asyncio.sleep(0)
        self.scope_during_calls.append(in_scope.get())
        if isinstance(self.behaviour, BaseException):
            raise self.behaviour
        return {"type": "text", "content": "answer"}

    async def stream_chat(self, messages, **kwargs) -> AsyncIterator[StreamChunk]:  # type: ignore[override]
        try:
            for item in self.behaviour:
                self.scope_during_calls.append(in_scope.get())
                await asyncio.sleep(0)
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            self.closed = True
            self.close_in_scope = in_scope.get()


class HttpError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class AuthenticationError(Exception):
    pass


class CodedError(Exception):
    """An SDK-style error with structured attributes and an optional status."""

    def __init__(self, message: str, *, status_code: Any = None, **attrs: Any):
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code
        for name, value in attrs.items():
            setattr(self, name, value)


def _named(name: str, message: str = "provider failure") -> Exception:
    """An error whose only signal is its SDK class name."""
    return type(name, (Exception,), {})(message)


def _chained(*errors: BaseException) -> BaseException:
    """``errors[0]`` caused by ``errors[1]`` caused by ... (explicit links)."""
    for outer, inner in zip(errors, errors[1:]):
        outer.__cause__ = inner
    return errors[0]


def _gemini(code: int, payload: dict[str, Any]) -> BaseException:
    """A real genai ClientError, wrapped the way the Gemini adapter wraps it."""
    try:
        try:
            raise genai_errors.ClientError(code, payload)
        except genai_errors.ClientError as exc:
            if code == 429:
                raise LLMRetryableError(
                    f"Gemini API error (code={exc.code}): {exc}"
                ) from exc
            raise RuntimeError(f"Gemini SDK API error: {exc}") from exc
    except Exception as outer:  # noqa: BLE001
        return outer


_GEMINI_PER_MINUTE_429 = {
    "error": {
        "code": 429,
        "message": "You exceeded your current quota, please check your plan and "
        "billing details.",
        "status": "RESOURCE_EXHAUSTED",
        "details": [
            {
                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [
                    {"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}
                ],
            }
        ],
    }
}
_GEMINI_BAD_KEY_400 = {
    "error": {
        "code": 400,
        "message": "API key not valid. Please pass a valid API key.",
        "status": "INVALID_ARGUMENT",
        "details": [
            {
                "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                "reason": "API_KEY_INVALID",
            }
        ],
    }
}


def _echo(status: int) -> HttpError:
    return HttpError(f"provider says: bad key {SECRET}", status)


def _assert_safe(exc: BaseException) -> None:
    assert SECRET not in str(exc)
    assert SECRET not in repr(exc)
    assert exc.__cause__ is None
    assert exc.__context__ is None


class TestClassification:
    @pytest.mark.parametrize(
        ("error", "code"),
        [
            (_echo(401), CREDENTIAL_REJECTED),
            (_echo(403), CREDENTIAL_REJECTED),
            (AuthenticationError("nope"), CREDENTIAL_REJECTED),
            (_echo(402), PROVIDER_QUOTA),
            # For a rate limit only a structured code means quota: Gemini's
            # ordinary per-minute 429 says "exceeded your current quota".
            (HttpError("You exceeded your current quota", 429), RATE_LIMITED),
            (_gemini(429, _GEMINI_PER_MINUTE_429), RATE_LIMITED),
            (
                CodedError("quota", status_code=429, code="insufficient_quota"),
                PROVIDER_QUOTA,
            ),
            (
                CodedError("quota", status_code=429, type="insufficient_quota"),
                PROVIDER_QUOTA,
            ),
            (CodedError("no status", code="insufficient_balance"), PROVIDER_QUOTA),
            # Off a rate limit the text markers still count.
            (HttpError("Insufficient Balance", 400), PROVIDER_QUOTA),
            # An unambiguous balance marker counts on a 429 as well.
            (
                HttpError(
                    "Insufficient balance or no resource package. Please recharge.",
                    429,
                ),
                PROVIDER_QUOTA,
            ),
            (_gemini(400, _GEMINI_BAD_KEY_400), CREDENTIAL_REJECTED),
            (_echo(429), RATE_LIMITED),
            (_echo(404), MODEL_NOT_AVAILABLE),
            (_echo(408), TIMEOUT),
            (asyncio.TimeoutError(), TIMEOUT),
            (httpx.ReadTimeout("slow"), TIMEOUT),
            (_echo(503), PROVIDER_UNAVAILABLE),
            (httpx.ConnectError("refused"), PROVIDER_UNAVAILABLE),
            (_echo(400), INVALID_REQUEST),
            (RuntimeError("maximum context length is 8192 tokens"), CONTEXT_LENGTH),
            (ValueError(f"odd {SECRET}"), PROVIDER_ERROR),
            # Class names alone.
            (_named("RateLimitError"), RATE_LIMITED),
            (_named("NotFoundError"), MODEL_NOT_AVAILABLE),
            (_named("APITimeoutError"), TIMEOUT),
            (_named("APIConnectionError"), PROVIDER_UNAVAILABLE),
            (_named("PermissionDeniedError"), CREDENTIAL_REJECTED),
            # A status found only on the response.
            (
                CodedError("gone", response=SimpleNamespace(status_code=404)),
                MODEL_NOT_AVAILABLE,
            ),
            # A bool is not a status.
            (CodedError("flags", status_code=True, code=True), PROVIDER_ERROR),
            # Precedence: a rejected credential wins over quota text, a
            # structured quota code wins over the 429.
            (HttpError("insufficient_quota on this key", 401), CREDENTIAL_REJECTED),
            (
                CodedError("slow down", status_code=429, code="insufficient_quota"),
                PROVIDER_QUOTA,
            ),
        ],
    )
    def test_codes(self, error, code):
        assert classify_provider_failure(error) == code

    def test_a_cyclic_chain_is_read_once(self):
        first, second = RuntimeError("adapter failed"), _echo(404)
        first.__cause__, second.__cause__ = second, first
        assert classify_provider_failure(first) == MODEL_NOT_AVAILABLE

    def test_the_chain_is_read_eight_links_deep(self):
        def chain_with_401_at(depth: int) -> BaseException:
            links = [RuntimeError(f"wrapper {i}") for i in range(depth)]
            return _chained(*links, _echo(401))

        # The 401 is the 8th link: read. As the 9th: beyond the cap.
        assert classify_provider_failure(chain_with_401_at(7)) == CREDENTIAL_REJECTED
        assert classify_provider_failure(chain_with_401_at(8)) == PROVIDER_ERROR

    def test_a_wrapped_cause_is_classified(self):
        try:
            try:
                raise _echo(401)
            except HttpError as inner:
                raise RuntimeError(f"adapter failed: {inner}") from inner
        except RuntimeError as outer:
            assert classify_provider_failure(outer) == CREDENTIAL_REJECTED


class TestChat:
    async def test_an_ordinary_exception_group_is_replaced(self):
        inner = FakeLLM(behaviour=ExceptionGroup("provider failures", [_echo(401)]))
        llm = guard_llm_calls(inner, call_scope=ScopeProbe())
        with pytest.raises(ProviderCallError) as caught:
            await llm.chat(MESSAGES)
        _assert_safe(caught.value)
        assert caught.value.code == PROVIDER_ERROR

    async def test_success_passes_through_inside_the_scope(self):
        probe = ScopeProbe()
        inner = FakeLLM()
        llm = guard_llm_calls(inner, call_scope=probe)
        assert await llm.chat(MESSAGES) == {"type": "text", "content": "answer"}
        assert inner.scope_during_calls == [True, True]
        assert (probe.entered, probe.exited) == (1, 1)
        assert not in_scope.get()

    @pytest.mark.parametrize("operation", ["chat", "vision_chat"])
    async def test_provider_error_is_replaced_inside_the_scope(
        self, recorder, operation
    ):
        failures: list[str] = []
        inner = FakeLLM(behaviour=_echo(401), abilities=["chat", "vision"])
        llm = guard_llm_calls(
            inner, call_scope=ScopeProbe(), on_failure=failures.append
        )
        with pytest.raises(ProviderCallError) as caught:
            await getattr(llm, operation)(MESSAGES)
        _assert_safe(caught.value)
        assert caught.value.code == CREDENTIAL_REJECTED
        assert failures == [CREDENTIAL_REJECTED]
        boundary_logs = [
            (record, scoped)
            for record, scoped in recorder.records
            if record.name.endswith("call_boundary")
        ]
        assert boundary_logs and all(scoped for _record, scoped in boundary_logs)

    async def test_context_length_keeps_its_runtime_meaning(self):
        llm = guard_llm_calls(
            FakeLLM(behaviour=RuntimeError(f"context_length_exceeded for {SECRET}")),
            call_scope=ScopeProbe(),
        )
        with pytest.raises(ProviderContextLengthError) as caught:
            await llm.chat(MESSAGES)
        _assert_safe(caught.value)
        assert isinstance(caught.value, LLMContextLengthError)
        assert is_context_length_error(caught.value)

    async def test_tool_protocol_error_is_copied_without_its_chain(self):
        protocol = LLMToolProtocolError(
            provider="openrouter",
            code="malformed_tool_arguments",
            message="arguments were not JSON",
            details={"tool": "search"},
        )
        protocol.__context__ = _echo(400)
        llm = guard_llm_calls(FakeLLM(behaviour=protocol), call_scope=ScopeProbe())
        with pytest.raises(LLMToolProtocolError) as caught:
            await llm.chat(MESSAGES)
        assert caught.value is not protocol
        assert caught.value.code == "malformed_tool_arguments"
        assert caught.value.details == {"tool": "search"}
        assert caught.value.__context__ is None and caught.value.__cause__ is None

    async def test_cancellation_is_not_a_failure(self):
        failures: list[str] = []

        class Hanging(FakeLLM):
            async def chat(self, messages, **kwargs):  # type: ignore[override]
                await asyncio.Event().wait()

        probe = ScopeProbe()
        llm = guard_llm_calls(Hanging(), call_scope=probe, on_failure=failures.append)
        task = asyncio.ensure_future(llm.chat(MESSAGES))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert failures == []
        assert probe.entered == probe.exited == 1

    async def test_a_scope_that_cannot_open_fails_safely_and_sends_nothing(self):
        inner = FakeLLM()

        @contextmanager
        def broken():
            try:
                json.loads(f"{{{SECRET}")
            except json.JSONDecodeError as exc:
                raise RuntimeError("credential unreadable") from exc
            yield  # pragma: no cover

        failures: list[str] = []
        llm = guard_llm_calls(inner, call_scope=broken, on_failure=failures.append)
        with pytest.raises(ProviderCallError) as caught:
            await llm.chat(MESSAGES)
        _assert_safe(caught.value)
        assert caught.value.code == CALL_SCOPE_UNAVAILABLE
        assert inner.scope_during_calls == []
        assert failures == [CALL_SCOPE_UNAVAILABLE]

    async def test_a_raising_failure_callback_does_not_replace_the_error(self):
        def explode(_code: str) -> None:
            raise RuntimeError("callback bug")

        llm = guard_llm_calls(
            FakeLLM(behaviour=_echo(429)), call_scope=ScopeProbe(), on_failure=explode
        )
        with pytest.raises(ProviderCallError) as caught:
            await llm.chat(MESSAGES)
        assert caught.value.code == RATE_LIMITED


class TestTheBoundaryCannotLeakOnItsOwnFailure:
    async def test_a_failing_classifier_falls_back_to_provider_error(self, monkeypatch):
        def broken(_exc):
            raise RuntimeError("classifier bug")

        monkeypatch.setattr(call_boundary_module, "classify_provider_failure", broken)
        failures: list[str] = []
        llm = guard_llm_calls(
            FakeLLM(behaviour=_echo(401)),
            call_scope=ScopeProbe(),
            on_failure=failures.append,
        )
        with pytest.raises(ProviderCallError) as caught:
            await llm.chat(MESSAGES)
        _assert_safe(caught.value)
        assert caught.value.code == PROVIDER_ERROR
        assert failures == [PROVIDER_ERROR]

    async def test_a_failing_classifier_on_an_error_chunk_falls_back(self, monkeypatch):
        def broken(_exc):
            raise RuntimeError("classifier bug")

        monkeypatch.setattr(call_boundary_module, "classify_provider_failure", broken)
        chunks = [StreamChunk(type=ChunkType.ERROR, content=f"LLM failed {SECRET}")]
        llm = guard_llm_calls(FakeLLM(behaviour=chunks), call_scope=ScopeProbe())
        with pytest.raises(ProviderCallError) as caught:
            async for _chunk in llm.stream_chat(messages=MESSAGES):
                pass
        _assert_safe(caught.value)
        assert caught.value.code == PROVIDER_ERROR

    async def test_a_failing_classifier_while_closing_does_not_escape(
        self, monkeypatch
    ):
        def broken(_exc):
            raise RuntimeError("classifier bug")

        class BadClose:
            def __init__(self) -> None:
                self.done = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.done:
                    raise StopAsyncIteration
                self.done = True
                return StreamChunk(type=ChunkType.TOKEN, content="x", delta="x")

            async def aclose(self):
                raise HttpError(f"close echoed {SECRET}", 500)

        class Inner(FakeLLM):
            def stream_chat(self, messages, **kwargs):  # type: ignore[override]
                return BadClose()

        monkeypatch.setattr(call_boundary_module, "classify_provider_failure", broken)
        llm = guard_llm_calls(Inner(), call_scope=ScopeProbe())
        received = [chunk async for chunk in llm.stream_chat(messages=MESSAGES)]
        assert [chunk.delta for chunk in received] == ["x"]

    async def test_an_unprintable_error_is_still_replaced(self):
        class Unprintable(Exception):
            def __str__(self) -> str:
                raise RuntimeError(f"cannot render {SECRET}")

        llm = guard_llm_calls(FakeLLM(behaviour=Unprintable()), call_scope=ScopeProbe())
        with pytest.raises(ProviderCallError) as caught:
            await llm.chat(MESSAGES)
        _assert_safe(caught.value)
        assert caught.value.code == PROVIDER_ERROR

    async def test_a_raising_callback_and_log_filter_together_leak_nothing(self):
        class Explode(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                raise RuntimeError("filter bug")

        def explode(_code: str) -> None:
            raise ValueError("callback bug")

        boundary_logger = logging.getLogger(call_boundary_module.__name__)
        broken_filter = Explode()
        boundary_logger.addFilter(broken_filter)
        try:
            llm = guard_llm_calls(
                FakeLLM(behaviour=_echo(401)),
                call_scope=ScopeProbe(),
                on_failure=explode,
            )
            with pytest.raises(ProviderCallError) as caught:
                await llm.chat(MESSAGES)
        finally:
            boundary_logger.removeFilter(broken_filter)
        _assert_safe(caught.value)
        assert caught.value.code == CREDENTIAL_REJECTED

    @pytest.mark.parametrize("scope_fault", ["enter", "exit"])
    async def test_a_raising_log_filter_does_not_replace_a_scope_failure(
        self, scope_fault
    ):
        class Explode(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                raise RuntimeError("filter bug")

        scope = (
            FailingScope(enter_fails={1})
            if scope_fault == "enter"
            else FailingScope(exit_fails={1})
        )
        boundary_logger = logging.getLogger(call_boundary_module.__name__)
        broken_filter = Explode()
        boundary_logger.addFilter(broken_filter)
        try:
            llm = guard_llm_calls(FakeLLM(), call_scope=scope)
            with pytest.raises(ProviderCallError) as caught:
                await llm.chat(MESSAGES)
        finally:
            boundary_logger.removeFilter(broken_filter)
        _assert_safe(caught.value)
        assert caught.value.code == CALL_SCOPE_UNAVAILABLE

    async def test_a_raising_log_filter_does_not_replace_the_error(self):
        class Explode(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                raise RuntimeError("filter bug")

        boundary_logger = logging.getLogger(call_boundary_module.__name__)
        explode = Explode()
        boundary_logger.addFilter(explode)
        try:
            llm = guard_llm_calls(
                FakeLLM(behaviour=_echo(401)), call_scope=ScopeProbe()
            )
            with pytest.raises(ProviderCallError) as caught:
                await llm.chat(MESSAGES)
        finally:
            boundary_logger.removeFilter(explode)
        _assert_safe(caught.value)
        assert caught.value.code == CREDENTIAL_REJECTED


class TestTransient:
    @pytest.mark.parametrize(
        ("error", "transient"),
        [
            # Retried by the retry layer: its retries are spent.
            (LLMRetryableError(f"provider overloaded {SECRET}"), True),
            # Retried, but the provider named the output budget: a smaller
            # budget may fix it, so it is not marked.
            (LLMRetryableError("max_tokens: 8000 is too large"), False),
            (LLMRetryableError("max_output_tokens exceeds the limit"), False),
            # Not something the retry layer retries.
            (_echo(400), False),
        ],
    )
    async def test_transient_records_what_the_retry_layer_decided(
        self, error, transient
    ):
        llm = guard_llm_calls(FakeLLM(behaviour=error), call_scope=ScopeProbe())
        with pytest.raises(ProviderCallError) as caught:
            await llm.chat(MESSAGES)
        _assert_safe(caught.value)
        assert caught.value.transient is transient

    def test_the_flag_is_off_by_default(self):
        assert ProviderCallError(RATE_LIMITED).transient is False


class TestNestedBoundaries:
    async def test_an_inner_boundary_keeps_its_code_and_both_scopes_run(self):
        outer_probe, inner_probe = ScopeProbe(), ScopeProbe()
        outer_failures: list[str] = []
        inner_failures: list[str] = []
        inner = guard_llm_calls(
            FakeLLM(behaviour=_echo(429)),
            call_scope=inner_probe,
            on_failure=inner_failures.append,
        )
        llm = guard_llm_calls(
            inner, call_scope=outer_probe, on_failure=outer_failures.append
        )
        with pytest.raises(ProviderCallError) as caught:
            await llm.chat(MESSAGES)
        _assert_safe(caught.value)
        assert caught.value.code == RATE_LIMITED
        assert inner_failures == outer_failures == [RATE_LIMITED]
        assert (outer_probe.entered, inner_probe.entered) == (1, 1)

    async def test_the_transient_flag_survives_an_outer_boundary(self):
        inner = guard_llm_calls(
            FakeLLM(behaviour=LLMRetryableError("overloaded")), call_scope=ScopeProbe()
        )
        llm = guard_llm_calls(inner, call_scope=ScopeProbe())
        with pytest.raises(ProviderCallError) as caught:
            await llm.chat(MESSAGES)
        assert caught.value.code == PROVIDER_ERROR
        assert caught.value.transient is True

    async def test_an_inner_context_length_error_stays_one(self):
        inner = guard_llm_calls(
            FakeLLM(behaviour=RuntimeError("context_length_exceeded")),
            call_scope=ScopeProbe(),
        )
        llm = guard_llm_calls(inner, call_scope=ScopeProbe())
        with pytest.raises(ProviderContextLengthError) as caught:
            await llm.chat(MESSAGES)
        _assert_safe(caught.value)

    async def test_an_inner_stream_error_keeps_its_code(self):
        chunks: list[Any] = [
            StreamChunk(type=ChunkType.TOKEN, content="a", delta="a"),
            _echo(401),
        ]
        inner = guard_llm_calls(FakeLLM(behaviour=chunks), call_scope=ScopeProbe())
        llm = guard_llm_calls(inner, call_scope=ScopeProbe())
        with pytest.raises(ProviderCallError) as caught:
            async for _chunk in llm.stream_chat(messages=MESSAGES):
                pass
        _assert_safe(caught.value)
        assert caught.value.code == CREDENTIAL_REJECTED

    async def test_an_error_chunk_carrying_a_safe_error_keeps_its_code(self):
        chunks = [
            StreamChunk(
                type=ChunkType.ERROR,
                content="vision refused",
                raw=ProviderCallError(VISION_UNAVAILABLE),
            )
        ]
        llm = guard_llm_calls(FakeLLM(behaviour=chunks), call_scope=ScopeProbe())
        with pytest.raises(ProviderCallError) as caught:
            async for _chunk in llm.stream_chat(messages=MESSAGES):
                pass
        assert caught.value.code == VISION_UNAVAILABLE

    def test_an_unavailable_vision_model_is_returned_as_it_is(self):
        model = UnavailableVisionModel()
        assert guard_llm_calls(model, call_scope=ScopeProbe()) is model

    async def test_both_safe_error_types_are_caught_by_one_tuple(self):
        for behaviour in (_echo(401), RuntimeError("context_length_exceeded")):
            llm = guard_llm_calls(FakeLLM(behaviour=behaviour), call_scope=ScopeProbe())
            try:
                await llm.chat(MESSAGES)
            except SAFE_PROVIDER_ERRORS as exc:
                assert exc.code in {CREDENTIAL_REJECTED, CONTEXT_LENGTH}
            else:  # pragma: no cover
                pytest.fail("no safe error raised")


class TestCancellationPassesWithoutItsChain:
    """A cancellation passes through, but not the provider error behind it."""

    async def test_a_cancellation_during_a_retry_backoff_drops_the_provider_error(
        self,
    ):
        class RetryingInBackoff(FakeLLM):
            async def chat(self, messages, **kwargs):  # type: ignore[override]
                try:
                    raise LLMRetryableError(f"invalid x-api-key {SECRET}")
                except LLMRetryableError:
                    # A retry layer sleeping in its handler before the next try.
                    await asyncio.Event().wait()

        llm = guard_llm_calls(RetryingInBackoff(), call_scope=ScopeProbe())
        task = asyncio.ensure_future(llm.chat(MESSAGES))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert caught.value.__context__ is None
        assert caught.value.__cause__ is None

    async def test_the_same_holds_for_a_stream_step(self):
        class Inner(FakeLLM):
            async def stream_chat(self, messages, **kwargs):  # type: ignore[override]
                yield StreamChunk(type=ChunkType.TOKEN, content="a", delta="a")
                try:
                    raise LLMRetryableError(f"overloaded {SECRET}")
                except LLMRetryableError:
                    await asyncio.Event().wait()

        llm = guard_llm_calls(Inner(), call_scope=ScopeProbe())
        errors: list[BaseException] = []

        async def consume() -> None:
            try:
                async for _chunk in llm.stream_chat(messages=MESSAGES):
                    pass
            except BaseException as exc:
                errors.append(exc)
                raise

        task = asyncio.ensure_future(consume())
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert isinstance(errors[0], asyncio.CancelledError)
        assert errors[0].__context__ is None and errors[0].__cause__ is None

    async def test_a_scope_that_would_suppress_it_does_not(self):
        class Swallowing:
            def __enter__(self):
                return self

            def __exit__(self, *exc_info: Any) -> bool:
                return True

        class Hanging(FakeLLM):
            async def chat(self, messages, **kwargs):  # type: ignore[override]
                await asyncio.Event().wait()

        llm = guard_llm_calls(Hanging(), call_scope=Swallowing)
        task = asyncio.ensure_future(llm.chat(MESSAGES))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


class TestAScopeThatFailsWhileItIsLeft:
    async def test_it_fails_an_otherwise_successful_call(self, recorder):
        reports: list[tuple[str, bool]] = []
        inner = FakeLLM()
        scope = FailingScope(exit_fails={1})
        llm = guard_llm_calls(
            inner,
            call_scope=scope,
            on_failure=lambda code: reports.append((code, in_scope.get())),
        )
        with pytest.raises(ProviderCallError) as caught:
            await llm.chat(MESSAGES)
        _assert_safe(caught.value)
        assert caught.value.code == CALL_SCOPE_UNAVAILABLE
        assert inner.scope_during_calls == [True, True]
        # Reported once the scope is gone; its error is logged by type only.
        assert reports == [(CALL_SCOPE_UNAVAILABLE, False)]
        assert not any(
            SECRET in record.getMessage() or record.exc_info
            for record, _scoped in recorder.records
            if record.name.endswith("call_boundary")
        )
        assert not in_scope.get()

    async def test_a_failed_call_keeps_its_own_code(self):
        failures: list[str] = []
        llm = guard_llm_calls(
            FakeLLM(behaviour=_echo(401)),
            call_scope=FailingScope(exit_fails={1}),
            on_failure=failures.append,
        )
        with pytest.raises(ProviderCallError) as caught:
            await llm.chat(MESSAGES)
        _assert_safe(caught.value)
        assert caught.value.code == CREDENTIAL_REJECTED
        assert failures == [CREDENTIAL_REJECTED, CALL_SCOPE_UNAVAILABLE]

    async def test_a_cancellation_in_flight_is_not_replaced(self):
        class Hanging(FakeLLM):
            async def chat(self, messages, **kwargs):  # type: ignore[override]
                await asyncio.Event().wait()

        failures: list[str] = []
        llm = guard_llm_calls(
            Hanging(),
            call_scope=FailingScope(exit_fails={1}),
            on_failure=failures.append,
        )
        task = asyncio.ensure_future(llm.chat(MESSAGES))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert failures == [CALL_SCOPE_UNAVAILABLE]

    async def test_it_ends_a_stream_at_that_step(self):
        chunks = [
            StreamChunk(type=ChunkType.TOKEN, content="a", delta="a"),
            StreamChunk(type=ChunkType.TOKEN, content="ab", delta="b"),
        ]
        inner = FakeLLM(behaviour=chunks)
        failures: list[str] = []
        # Exits: 1 creation, 2 first chunk, 3 second chunk.
        llm = guard_llm_calls(
            inner, call_scope=FailingScope(exit_fails={3}), on_failure=failures.append
        )
        received: list[StreamChunk] = []
        with pytest.raises(ProviderCallError) as caught:
            async for chunk in llm.stream_chat(messages=MESSAGES):
                received.append(chunk)
        _assert_safe(caught.value)
        assert caught.value.code == CALL_SCOPE_UNAVAILABLE
        assert [chunk.delta for chunk in received] == ["a"]
        assert failures == [CALL_SCOPE_UNAVAILABLE]
        assert inner.closed

    async def test_while_closing_a_stream_it_is_reported_not_raised(self):
        chunks = [StreamChunk(type=ChunkType.TOKEN, content="a", delta="a")]
        failures: list[str] = []
        # Exits: 1 creation, 2 chunk, 3 end of stream, 4 close.
        llm = guard_llm_calls(
            FakeLLM(behaviour=chunks),
            call_scope=FailingScope(exit_fails={4}),
            on_failure=failures.append,
        )
        received = [chunk async for chunk in llm.stream_chat(messages=MESSAGES)]
        assert [chunk.delta for chunk in received] == ["a"]
        assert failures == [CALL_SCOPE_UNAVAILABLE]


class TestStream:
    async def test_every_step_and_the_close_run_inside_the_scope(self):
        chunks = [
            StreamChunk(type=ChunkType.TOKEN, content="he", delta="he"),
            StreamChunk(type=ChunkType.TOKEN, content="hello", delta="llo"),
        ]
        probe = ScopeProbe()
        inner = FakeLLM(behaviour=chunks)
        llm = guard_llm_calls(inner, call_scope=probe)
        received = [chunk async for chunk in llm.stream_chat(messages=MESSAGES)]
        assert [chunk.delta for chunk in received] == ["he", "llo"]
        assert inner.scope_during_calls == [True, True]
        assert inner.closed and inner.close_in_scope is True
        assert probe.entered == probe.exited
        assert not in_scope.get()

    async def test_a_mid_stream_error_ends_the_stream_safely(self):
        chunks: list[Any] = [
            StreamChunk(type=ChunkType.TOKEN, content="partial", delta="partial"),
            HttpError(f"upstream rejected key {SECRET}", 500),
        ]
        failures: list[str] = []
        inner = FakeLLM(behaviour=chunks)
        llm = guard_llm_calls(
            inner, call_scope=ScopeProbe(), on_failure=failures.append
        )
        received: list[StreamChunk] = []
        with pytest.raises(ProviderCallError) as caught:
            async for chunk in llm.stream_chat(messages=MESSAGES):
                received.append(chunk)
        assert [chunk.delta for chunk in received] == ["partial"]
        _assert_safe(caught.value)
        assert caught.value.code == PROVIDER_UNAVAILABLE
        assert failures == [PROVIDER_UNAVAILABLE]
        assert inner.closed

    async def test_an_error_chunk_is_treated_as_the_error_it_reports(self):
        echoed = _echo(401)
        chunks = [
            StreamChunk(
                type=ChunkType.ERROR,
                content=f"Zhipu streaming API error: {echoed}",
                raw=echoed,
            )
        ]
        llm = guard_llm_calls(FakeLLM(behaviour=chunks), call_scope=ScopeProbe())
        received: list[StreamChunk] = []
        with pytest.raises(ProviderCallError) as caught:
            async for chunk in llm.stream_chat(messages=MESSAGES):
                received.append(chunk)
        assert received == []
        _assert_safe(caught.value)
        assert caught.value.code == CREDENTIAL_REJECTED

    async def test_an_error_chunk_without_an_exception_is_classified_by_text(self):
        chunks = [StreamChunk(type=ChunkType.ERROR, content=f"LLM failed {SECRET}")]
        llm = guard_llm_calls(FakeLLM(behaviour=chunks), call_scope=ScopeProbe())
        with pytest.raises(ProviderCallError) as caught:
            async for _chunk in llm.stream_chat(messages=MESSAGES):
                pass
        _assert_safe(caught.value)
        assert caught.value.code == PROVIDER_ERROR

    async def test_protocol_error_chunks_pass_through(self):
        chunks = [
            StreamChunk(
                type=ChunkType.PROTOCOL_ERROR,
                protocol_error={"code": "unavailable_tool_call"},
            )
        ]
        llm = guard_llm_calls(FakeLLM(behaviour=chunks), call_scope=ScopeProbe())
        received = [chunk async for chunk in llm.stream_chat(messages=MESSAGES)]
        assert [chunk.type for chunk in received] == [ChunkType.PROTOCOL_ERROR]

    async def test_early_close_by_the_consumer_closes_the_provider_stream(self):
        chunks = [
            StreamChunk(type=ChunkType.TOKEN, content="a", delta="a"),
            StreamChunk(type=ChunkType.TOKEN, content="ab", delta="b"),
        ]
        probe = ScopeProbe()
        inner = FakeLLM(behaviour=chunks)
        llm = guard_llm_calls(inner, call_scope=probe)
        stream = llm.stream_chat(messages=MESSAGES)
        first = await anext(stream)
        assert first.delta == "a"
        await stream.aclose()
        assert inner.closed and inner.close_in_scope is True
        assert probe.entered == probe.exited

    async def test_a_failing_close_does_not_replace_the_outcome(self, recorder):
        class BadClose:
            def __init__(self) -> None:
                self.done = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.done:
                    raise StopAsyncIteration
                self.done = True
                return StreamChunk(type=ChunkType.TOKEN, content="x", delta="x")

            async def aclose(self):
                raise HttpError(f"close echoed {SECRET}", 500)

        class Inner(FakeLLM):
            def stream_chat(self, messages, **kwargs):  # type: ignore[override]
                return BadClose()

        llm = guard_llm_calls(Inner(), call_scope=ScopeProbe())
        received = [chunk async for chunk in llm.stream_chat(messages=MESSAGES)]
        assert [chunk.delta for chunk in received] == ["x"]
        close_logs = [
            (record, scoped)
            for record, scoped in recorder.records
            if record.getMessage().startswith("Closing a model provider stream failed")
        ]
        assert close_logs and all(scoped for _record, scoped in close_logs)

    async def test_cancellation_mid_stream_propagates_and_closes(self):
        release = asyncio.Event()

        class Slow:
            """An iterator that does not close itself: only ``aclose`` does."""

            def __init__(self) -> None:
                self.sent = 0
                self.aclose_calls: list[bool] = []

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.sent == 0:
                    self.sent = 1
                    return StreamChunk(type=ChunkType.TOKEN, content="a", delta="a")
                await release.wait()
                raise StopAsyncIteration  # pragma: no cover

            async def aclose(self) -> None:
                self.aclose_calls.append(in_scope.get())

        slow = Slow()

        class Inner(FakeLLM):
            def stream_chat(self, messages, **kwargs):  # type: ignore[override]
                return slow

        failures: list[str] = []
        probe = ScopeProbe()
        llm = guard_llm_calls(Inner(), call_scope=probe, on_failure=failures.append)

        async def consume() -> None:
            async for _chunk in llm.stream_chat(messages=MESSAGES):
                pass

        task = asyncio.ensure_future(consume())
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert failures == []
        # The boundary closed the provider stream, inside the scope.
        assert slow.aclose_calls == [True]
        assert probe.entered == probe.exited

    async def test_a_scope_that_cannot_reopen_mid_stream_ends_it_safely(self):
        chunks = [
            StreamChunk(type=ChunkType.TOKEN, content="a", delta="a"),
            StreamChunk(type=ChunkType.TOKEN, content="ab", delta="b"),
        ]
        inner = FakeLLM(behaviour=chunks)
        failures: list[str] = []
        # Entries: 1 creation, 2 first chunk, 3 second chunk (fails), 4 close.
        llm = guard_llm_calls(
            inner, call_scope=FailingScope(enter_fails={3}), on_failure=failures.append
        )
        received: list[StreamChunk] = []
        with pytest.raises(ProviderCallError) as caught:
            async for chunk in llm.stream_chat(messages=MESSAGES):
                received.append(chunk)
        _assert_safe(caught.value)
        assert caught.value.code == CALL_SCOPE_UNAVAILABLE
        assert [chunk.delta for chunk in received] == ["a"]
        assert failures == [CALL_SCOPE_UNAVAILABLE]
        assert inner.closed and inner.close_in_scope is True

    async def test_a_scope_that_cannot_reopen_for_the_close_reports_twice(self):
        chunks = [
            StreamChunk(type=ChunkType.TOKEN, content="a", delta="a"),
            StreamChunk(type=ChunkType.TOKEN, content="ab", delta="b"),
        ]
        failures: list[str] = []
        llm = guard_llm_calls(
            FakeLLM(behaviour=chunks),
            call_scope=FailingScope(enter_fails={3, 4}),
            on_failure=failures.append,
        )
        with pytest.raises(ProviderCallError) as caught:
            async for _chunk in llm.stream_chat(messages=MESSAGES):
                pass
        assert caught.value.code == CALL_SCOPE_UNAVAILABLE
        assert failures == [CALL_SCOPE_UNAVAILABLE, CALL_SCOPE_UNAVAILABLE]

    async def test_a_close_scope_failure_after_success_is_reported(self):
        chunks = [StreamChunk(type=ChunkType.TOKEN, content="a", delta="a")]
        failures: list[str] = []
        # Entries: 1 creation, 2 chunk, 3 end of stream, 4 close (fails).
        llm = guard_llm_calls(
            FakeLLM(behaviour=chunks),
            call_scope=FailingScope(enter_fails={4}),
            on_failure=failures.append,
        )
        received = [chunk async for chunk in llm.stream_chat(messages=MESSAGES)]
        assert [chunk.delta for chunk in received] == ["a"]
        assert failures == [CALL_SCOPE_UNAVAILABLE]

    async def test_a_stream_that_fails_to_start_is_replaced(self):
        class Inner(FakeLLM):
            def stream_chat(self, messages, **kwargs):  # type: ignore[override]
                raise _echo(401)

        failures: list[str] = []
        llm = guard_llm_calls(
            Inner(), call_scope=ScopeProbe(), on_failure=failures.append
        )
        with pytest.raises(ProviderCallError) as caught:
            async for _chunk in llm.stream_chat(messages=MESSAGES):
                pass
        _assert_safe(caught.value)
        assert caught.value.code == CREDENTIAL_REJECTED
        assert failures == [CREDENTIAL_REJECTED]

    async def test_a_protocol_error_mid_stream_is_copied_without_its_chain(self):
        protocol = LLMToolProtocolError(
            provider="openrouter",
            code="malformed_tool_arguments",
            message="arguments were not JSON",
            details={"tool": "search"},
        )
        protocol.__context__ = _echo(400)
        chunks: list[Any] = [
            StreamChunk(type=ChunkType.TOKEN, content="a", delta="a"),
            protocol,
        ]
        inner = FakeLLM(behaviour=chunks)
        llm = guard_llm_calls(inner, call_scope=ScopeProbe())
        received: list[StreamChunk] = []
        with pytest.raises(LLMToolProtocolError) as caught:
            async for chunk in llm.stream_chat(messages=MESSAGES):
                received.append(chunk)
        assert caught.value is not protocol
        assert caught.value.code == "malformed_tool_arguments"
        assert caught.value.__context__ is None and caught.value.__cause__ is None
        assert [chunk.delta for chunk in received] == ["a"]
        assert inner.closed


class TestSurface:
    def test_model_facts_are_forwarded(self):
        inner = FakeLLM(abilities=["chat", "tool_calling", "vision"])
        llm = guard_llm_calls(inner, call_scope=ScopeProbe())
        assert llm.abilities == ["chat", "tool_calling", "vision"]
        assert llm.has_ability("vision")
        assert llm.model_name == "fake-model"
        assert llm.model_id == "caller-model-id"
        assert llm.context_window == 12345

    def test_capabilities_and_video_content_are_forwarded(self):
        class Capable(FakeLLM):
            supports_thinking_mode = True  # type: ignore[assignment]
            supports_json_schema_response_format = True
            supports_json_object_response_format = True
            supports_native_video_input = True
            supports_native_video_with_images = True
            supports_native_video_time_range = True

            def build_native_video_content(
                self, video_url, *, start_time=None, end_time=None
            ):
                return {"video": video_url, "range": (start_time, end_time)}

        llm = guard_llm_calls(Capable(), call_scope=ScopeProbe())
        assert llm.supports_thinking_mode is True
        assert llm.supports_json_schema_response_format is True
        assert llm.supports_json_object_response_format is True
        assert llm.supports_native_video_input is True
        assert llm.supports_native_video_with_images is True
        assert llm.supports_native_video_time_range is True
        assert llm.build_native_video_content(
            "v.mp4", start_time=1.0, end_time=2.0
        ) == {
            "video": "v.mp4",
            "range": (1.0, 2.0),
        }

    def test_no_attribute_reaches_the_wrapped_model(self):
        llm = guard_llm_calls(FakeLLM(), call_scope=ScopeProbe())
        with pytest.raises(AttributeError):
            llm.api_key  # noqa: B018
        with pytest.raises(AttributeError):
            llm.behaviour  # noqa: B018


class TestUnavailableVisionModel:
    @pytest.mark.parametrize("operation", ["chat", "vision_chat"])
    async def test_every_call_is_refused(self, operation):
        model = UnavailableVisionModel()
        assert not model.has_ability("vision")
        with pytest.raises(ProviderCallError) as caught:
            await getattr(model, operation)(MESSAGES)
        assert caught.value.code == VISION_UNAVAILABLE


class TestRealOpenAIPath:
    """create_base_llm -> RetryWrapper -> OpenAILLM -> openai SDK -> httpx."""

    @pytest.fixture
    def echo_provider(self, monkeypatch):
        state: dict[str, Any] = {"mode": "401", "requests": []}

        async def handle_async_request(self, request):
            await request.aread()
            auth = request.headers.get("authorization", "")
            state["requests"].append(auth)
            key = auth[7:] if auth.lower().startswith("bearer ") else auth
            if state["mode"] == "401":
                response = httpx.Response(
                    401,
                    json={"error": {"message": f"Incorrect API key provided: {key}"}},
                )
            elif state["mode"] == "sse-midway":
                first = {
                    "id": "c1",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "gpt-4o",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "he"},
                            "finish_reason": None,
                        }
                    ],
                }
                error = {"error": {"message": f"upstream rejected key {key}"}}
                body = f"data: {json.dumps(first)}\n\ndata: {json.dumps(error)}\n\n"
                response = httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=body.encode(),
                )
            else:  # pragma: no cover - test bug
                raise AssertionError(state["mode"])
            response.request = request
            return response

        monkeypatch.setattr(
            httpx.AsyncHTTPTransport, "handle_async_request", handle_async_request
        )
        for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE"):
            monkeypatch.delenv(name, raising=False)
        return state

    def _caller_llm(self) -> BaseLLM:
        return create_base_llm(
            ChatModelConfig(
                id="gpt-4o",
                model_provider="openai",
                model_name="gpt-4o",
                api_key=SECRET,
                max_retries=1,
            )
        )

    async def test_an_echoed_key_never_leaves_a_chat_failure(
        self, echo_provider, recorder
    ):
        llm = guard_llm_calls(self._caller_llm(), call_scope=ScopeProbe())
        with pytest.raises(ProviderCallError) as caught:
            await llm.chat(MESSAGES)
        _assert_safe(caught.value)
        assert caught.value.code == CREDENTIAL_REJECTED
        assert echo_provider["requests"] == [f"Bearer {SECRET}"]
        # Every log record that could carry the echo was made inside the scope.
        leaking = [
            record
            for record, scoped in recorder.records
            if not scoped
            and SECRET in (record.getMessage() + str(record.exc_info or ""))
        ]
        assert leaking == []

    async def test_a_mid_stream_sse_error_never_leaves_the_stream(
        self, echo_provider, recorder
    ):
        echo_provider["mode"] = "sse-midway"
        llm = guard_llm_calls(self._caller_llm(), call_scope=ScopeProbe())
        received: list[StreamChunk] = []
        with pytest.raises(ProviderCallError) as caught:
            async for chunk in llm.stream_chat(messages=MESSAGES):
                received.append(chunk)
        _assert_safe(caught.value)
        leaking = [
            record
            for record, scoped in recorder.records
            if not scoped
            and SECRET in (record.getMessage() + str(record.exc_info or ""))
        ]
        assert leaking == []
