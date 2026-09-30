"""A safe error boundary around every provider call one LLM makes.

Provider and SDK errors carry the provider's response text, and some
providers echo the request's credential back in it. That text travels a long
way once an exception leaves the model: retry warnings, trace events stored
in the database, outbound stream frames, runner logs, and whatever the
calling product renders to its users. A log filter cannot help with most of
those, because they are data, not log records.

:func:`guard_llm_calls` replaces provider exceptions from the ``BaseLLM``
call and streaming protocol with fixed errors.
Every provider-touching operation -- one ``chat``/``vision_chat`` await, each
step of a ``stream_chat`` iteration, and the stream's ``aclose`` -- runs
inside a context manager the caller supplies (a log-redaction scope, say),
and ordinary provider exceptions are caught inside that context and replaced by a
:class:`ProviderCallError` that carries only a fixed failure code, a fixed
message and a text-free ``transient`` flag, without the provider's cause or
context chain. Stream ``ERROR`` chunks, whose
``content``/``raw`` hold the same provider text, are treated as the error
they report. The caller learns each failure code through ``on_failure``.
A scope that cannot be entered, or that fails while it is left, fails the
call with ``call_scope_unavailable`` rather than letting its own exception
out. :data:`SAFE_PROVIDER_ERRORS` names the two provider failure types.

What stays as it was:

* ``asyncio.CancelledError`` and other ``BaseException`` shutdown signals
  pass through; a cancelled call is not a failed call. Only their exception
  chain is cleared: one raised while a retry layer was handling a provider
  error (during a backoff sleep, say) would otherwise carry that error.
  This does not sanitize a shutdown signal's own payload or recursively
  sanitize members of a mixed ``BaseExceptionGroup``.
* A context-window rejection becomes :class:`ProviderContextLengthError`,
  a :class:`LLMContextLengthError`, so the agent runtime's compaction
  recovery still recognizes it.
* :class:`LLMToolProtocolError` and ``PROTOCOL_ERROR`` chunks are derived
  from the model's own output, not from a provider error body, and the ReAct
  pattern repairs them from their code and details; they are re-raised as a
  fresh, chain-free copy and passed through respectively.
* A safe error raised by an inner boundary (an already-guarded model, or an
  :class:`UnavailableVisionModel`) keeps its code; it is re-raised as a
  fresh copy, not re-classified.
* Successful responses and ordinary chunks are returned unchanged.

Streaming adapters must yield the ``StreamChunk`` values required by
``BaseLLM.stream_chat``. Arbitrary extension objects with raising properties
are outside that protocol. Traceback frame locals are not scrubbed; hosts
must not capture them as a substitute for the fixed failure code.

The wrapper exposes the ``BaseLLM`` surface only. There is deliberately no
attribute fallthrough to the wrapped object, so no caller can reach an
unguarded client method by accident.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractContextManager
from typing import Any, List, NamedTuple

import httpx

from ..error import is_context_length_error, retry_on
from ..exceptions import LLMContextLengthError, LLMToolProtocolError
from ..types import ChunkType, StreamChunk
from .base import BaseLLM

logger = logging.getLogger(__name__)

# Fixed failure vocabulary. Callers map these to their own user-facing text.
CONTEXT_LENGTH = "context_length"
CREDENTIAL_REJECTED = "credential_rejected"
PROVIDER_QUOTA = "provider_quota"
RATE_LIMITED = "rate_limited"
TIMEOUT = "timeout"
PROVIDER_UNAVAILABLE = "provider_unavailable"
MODEL_NOT_AVAILABLE = "model_not_available"
INVALID_REQUEST = "invalid_request"
PROVIDER_ERROR = "provider_error"
# The caller's call scope itself could not be entered (for example the
# credential it needs could not be read); no provider was contacted.
CALL_SCOPE_UNAVAILABLE = "call_scope_unavailable"
# A model that was deliberately configured without vision was asked to see.
VISION_UNAVAILABLE = "vision_unavailable"

PROVIDER_CALL_FAILURE_CODES = frozenset(
    {
        CONTEXT_LENGTH,
        CREDENTIAL_REJECTED,
        PROVIDER_QUOTA,
        RATE_LIMITED,
        TIMEOUT,
        PROVIDER_UNAVAILABLE,
        MODEL_NOT_AVAILABLE,
        INVALID_REQUEST,
        PROVIDER_ERROR,
        CALL_SCOPE_UNAVAILABLE,
        VISION_UNAVAILABLE,
    }
)

# Failures that asking again with a smaller output budget cannot fix: the
# cause is the provider's availability, the credential, the model or the call
# scope, and the wrapped model's own retries have already run. A caller that
# steps its budget down after a failure (the runtime's compaction-summary
# ladder) stops on these. Left out on purpose, so they keep stepping down:
# ``invalid_request`` and ``provider_error`` (a rejected ``max_tokens``
# arrives as one of them) and ``provider_quota`` (some providers refuse an
# over-budget request with 402 "fewer max_tokens"). ``context_length`` has
# its own handling.
BUDGET_INSENSITIVE_FAILURE_CODES = frozenset(
    {
        CREDENTIAL_REJECTED,
        RATE_LIMITED,
        TIMEOUT,
        PROVIDER_UNAVAILABLE,
        MODEL_NOT_AVAILABLE,
        CALL_SCOPE_UNAVAILABLE,
        VISION_UNAVAILABLE,
    }
)

_SAFE_MESSAGES = {
    # Keeps a "context length exceeded" marker so is_context_length_error
    # recognizes the message as well as the type.
    CONTEXT_LENGTH: "Model provider call failed: context length exceeded.",
    CREDENTIAL_REJECTED: "Model provider call failed: the credential was rejected.",
    PROVIDER_QUOTA: "Model provider call failed: the provider account has no quota left.",
    RATE_LIMITED: "Model provider call failed: rate limited by the provider.",
    TIMEOUT: "Model provider call failed: the request timed out.",
    PROVIDER_UNAVAILABLE: "Model provider call failed: the provider is unavailable.",
    MODEL_NOT_AVAILABLE: "Model provider call failed: the model is not available.",
    INVALID_REQUEST: "Model provider call failed: the provider rejected the request.",
    PROVIDER_ERROR: "Model provider call failed.",
    CALL_SCOPE_UNAVAILABLE: "Model call could not be prepared.",
    VISION_UNAVAILABLE: "This model is not configured to read images.",
}

_CREDENTIAL_ERROR_NAMES = frozenset({"AuthenticationError", "PermissionDeniedError"})
_RATE_LIMIT_ERROR_NAMES = frozenset({"RateLimitError"})
_NOT_FOUND_ERROR_NAMES = frozenset({"NotFoundError"})
_TIMEOUT_ERROR_NAMES = frozenset({"APITimeoutError", "LLMTimeoutError"})
_UNAVAILABLE_ERROR_NAMES = frozenset(
    {
        "APIConnectionError",
        "InternalServerError",
        "OverloadedError",
        "ServiceUnavailableError",
    }
)
# Structured quota codes (an SDK's ``code``/``type`` attribute).
_QUOTA_CODES = frozenset({"insufficient_quota", "insufficient_balance"})
_QUOTA_MARKERS = (
    "insufficient_quota",
    "insufficient quota",
    "insufficient balance",
    "insufficient_balance",
    "billing hard limit",
)
# Also said by an ordinary per-minute 429 (Gemini's RESOURCE_EXHAUSTED), so it
# means quota only when the failure is not a rate limit.
_AMBIGUOUS_QUOTA_MARKERS = ("exceeded your current quota",)
# Gemini reports a wrong or expired key as a 400 whose reason is
# API_KEY_INVALID.
_CREDENTIAL_MARKERS = ("api_key_invalid",)
# A refusal that names the output budget: a smaller budget may fix it even
# when the wrapped model's retry layer treated it as transient.
_BUDGET_MARKERS = ("max_tokens", "max_output_tokens", "max_completion_tokens")


class ProviderCallError(RuntimeError):
    """A provider call failed; only a fixed code and message are kept.

    ``transient`` is a text-free flag: the retry predicate (``retry_on``)
    treats the failure as transient -- behind a retry layer, its retries are
    already spent -- and the provider did not name the output budget as the
    problem. A caller that steps its budget down after a failure (the
    runtime's compaction-summary ladder) stops on it.
    """

    def __init__(self, code: str, *, transient: bool = False) -> None:
        self.code = code if code in PROVIDER_CALL_FAILURE_CODES else PROVIDER_ERROR
        self.transient = bool(transient)
        super().__init__(_SAFE_MESSAGES[self.code])


class ProviderContextLengthError(LLMContextLengthError):
    """A context-window rejection with the provider's text removed."""

    def __init__(self) -> None:
        self.code = CONTEXT_LENGTH
        super().__init__(_SAFE_MESSAGES[CONTEXT_LENGTH])


# Every error type a guarded call can raise for a provider failure; both
# carry a fixed ``.code``. ``ProviderContextLengthError`` is an
# ``LLMContextLengthError`` rather than a ``ProviderCallError`` so compaction
# recovery keeps recognizing it; catch this tuple to handle both.
SAFE_PROVIDER_ERRORS: tuple[type[Exception], ...] = (
    ProviderCallError,
    ProviderContextLengthError,
)


def _chain(exc: BaseException) -> list[BaseException]:
    seen: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and len(seen) < 8 and current not in seen:
        seen.append(current)
        current = current.__cause__ or current.__context__
    return seen


def _http_status(exc: BaseException) -> int | None:
    candidates = [
        getattr(exc, name, None) for name in ("status_code", "code", "status")
    ]
    candidates.append(getattr(getattr(exc, "response", None), "status_code", None))
    for value in candidates:
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and 100 <= value <= 599
        ):
            return value
    return None


def _has_quota_code(exc: BaseException) -> bool:
    for name in ("code", "type"):
        value = getattr(exc, name, None)
        if isinstance(value, str) and value.lower() in _QUOTA_CODES:
            return True
    return False


def _mentions(exc: BaseException, markers: tuple[str, ...]) -> bool:
    # Read, never emitted: only the fixed code leaves this module.
    try:
        text = str(exc).lower()
    except Exception:  # noqa: BLE001 - an unprintable error has no text to match
        return False
    return any(marker in text for marker in markers)


def classify_provider_failure(exc: BaseException) -> str:
    """The fixed failure code for an error raised by a provider call.

    Reads type names, HTTP statuses, structured quota codes, and a few
    well-known credential and quota markers anywhere in the exception chain;
    the provider's text itself is never returned. "Exceeded your current
    quota" alone does not make a rate limit ``provider_quota``: an ordinary
    per-minute 429 can say it too. Anything unrecognized is
    ``provider_error``.
    """
    if is_context_length_error(exc):
        return CONTEXT_LENGTH
    causes = _chain(exc)
    names = {type(cause).__name__ for cause in causes}
    statuses = {
        status for cause in causes if (status := _http_status(cause)) is not None
    }
    if (
        names & _CREDENTIAL_ERROR_NAMES
        or statuses & {401, 403}
        or any(_mentions(cause, _CREDENTIAL_MARKERS) for cause in causes)
    ):
        return CREDENTIAL_REJECTED
    rate_limited = bool(names & _RATE_LIMIT_ERROR_NAMES) or 429 in statuses
    if (
        402 in statuses
        or any(_has_quota_code(cause) for cause in causes)
        or any(_mentions(cause, _QUOTA_MARKERS) for cause in causes)
        or (
            not rate_limited
            and any(_mentions(cause, _AMBIGUOUS_QUOTA_MARKERS) for cause in causes)
        )
    ):
        return PROVIDER_QUOTA
    if rate_limited:
        return RATE_LIMITED
    if names & _NOT_FOUND_ERROR_NAMES or 404 in statuses:
        return MODEL_NOT_AVAILABLE
    if (
        names & _TIMEOUT_ERROR_NAMES
        or 408 in statuses
        or any(
            isinstance(cause, (TimeoutError, httpx.TimeoutException))
            for cause in causes
        )
    ):
        return TIMEOUT
    if (
        names & _UNAVAILABLE_ERROR_NAMES
        or any(status >= 500 for status in statuses)
        or any(
            isinstance(cause, (ConnectionError, httpx.TransportError))
            for cause in causes
        )
    ):
        return PROVIDER_UNAVAILABLE
    if any(400 <= status < 500 for status in statuses):
        return INVALID_REQUEST
    return PROVIDER_ERROR


def _classify_safely(exc: BaseException) -> str:
    """``classify_provider_failure`` that cannot itself fail."""
    try:
        return classify_provider_failure(exc)
    except Exception:  # noqa: BLE001 - the boundary must not leak on its own failure
        return PROVIDER_ERROR


def _retried_as_transient(exc: BaseException) -> bool:
    """Whether the retry predicate treats ``exc`` as transient.

    A refusal that names the output budget is excluded: a smaller budget may
    still fix it.
    """
    if not isinstance(exc, Exception):
        return False
    try:
        return retry_on(exc) and not any(
            _mentions(cause, _BUDGET_MARKERS) for cause in _chain(exc)
        )
    except Exception:  # noqa: BLE001 - the boundary must not leak on its own failure
        return False


class _Failure(NamedTuple):
    code: str
    transient: bool = False


def _safe_error(failure: _Failure) -> Exception:
    if failure.code == CONTEXT_LENGTH:
        return ProviderContextLengthError()
    return ProviderCallError(failure.code, transient=failure.transient)


def _detached_protocol_error(exc: LLMToolProtocolError) -> LLMToolProtocolError:
    """A copy of a model-output protocol error without its exception chain."""
    return LLMToolProtocolError(
        provider=exc.provider,
        code=exc.code,
        message=exc.protocol_message,
        details=exc.details,
    )


class BoundaryLLM(BaseLLM):
    """An LLM whose provider calls all run inside one safe error boundary.

    Build it with :func:`guard_llm_calls`.
    """

    def __init__(
        self,
        inner: BaseLLM,
        *,
        call_scope: Callable[[], AbstractContextManager[Any]],
        on_failure: Callable[[str], None] | None = None,
    ) -> None:
        self._inner = inner
        self._call_scope = call_scope
        self._on_failure = on_failure
        # Fixed at construction: the wrapped model is immutable for its run.
        self.context_window = inner.context_window
        self._model_id = inner.model_id or None

    # -- BaseLLM surface ----------------------------------------------------

    @property
    def abilities(self) -> List[str]:
        return list(self._inner.abilities)

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    @property
    def supports_thinking_mode(self) -> bool:
        return self._inner.supports_thinking_mode

    @property
    def supports_json_schema_response_format(self) -> bool:
        return self._inner.supports_json_schema_response_format

    @property
    def supports_json_object_response_format(self) -> bool:
        return self._inner.supports_json_object_response_format

    @property
    def supports_native_video_input(self) -> bool:
        return self._inner.supports_native_video_input

    @property
    def supports_native_video_with_images(self) -> bool:
        return self._inner.supports_native_video_with_images

    @property
    def supports_native_video_time_range(self) -> bool:
        return self._inner.supports_native_video_time_range

    def build_native_video_content(
        self,
        video_url: str,
        *,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> dict[str, Any]:
        return self._inner.build_native_video_content(
            video_url, start_time=start_time, end_time=end_time
        )

    # -- the boundary --------------------------------------------------------

    def _open_scope(self) -> _GuardedScope:
        """Enter the caller's scope, or fail with a safe, chain-free error."""
        stack = contextlib.ExitStack()
        failed_type: str | None = None
        try:
            stack.enter_context(self._call_scope())
        except Exception as exc:  # noqa: BLE001 - replaced below, outside the handler
            failed_type = type(exc).__name__
        if failed_type is not None:
            # Entry did not complete, so log only the type name.
            try:
                logger.warning(
                    "Model call scope could not be entered (%s)", failed_type
                )
            except Exception:  # noqa: BLE001 - a failing log filter changes nothing here
                pass
            self._report(CALL_SCOPE_UNAVAILABLE)
            raise ProviderCallError(CALL_SCOPE_UNAVAILABLE)
        return _GuardedScope(self, stack)

    def _report(self, code: str) -> None:
        if self._on_failure is None:
            return
        failed_type: str | None = None
        try:
            self._on_failure(code)
        except Exception as exc:  # noqa: BLE001 - reporting must not replace the failure
            failed_type = type(exc).__name__
        if failed_type is not None:
            try:
                logger.warning("Model failure callback raised (%s)", failed_type)
            except Exception:  # noqa: BLE001 - a failing log filter changes nothing here
                pass

    def _record_failure(self, exc: BaseException, operation: str) -> _Failure:
        """Classify and log one provider error. Call inside the caller's scope."""
        failure = _Failure(PROVIDER_ERROR)
        try:
            if isinstance(exc, ProviderCallError):
                # An inner boundary already replaced the provider's error: keep
                # its code rather than re-classify its fixed message.
                code = getattr(exc, "code", PROVIDER_ERROR)
                failure = _Failure(
                    code if code in PROVIDER_CALL_FAILURE_CODES else PROVIDER_ERROR,
                    getattr(exc, "transient", False) is True,
                )
            elif isinstance(exc, ProviderContextLengthError):
                failure = _Failure(CONTEXT_LENGTH)
            else:
                failure = _Failure(_classify_safely(exc), _retried_as_transient(exc))
        except Exception:  # noqa: BLE001 - the boundary must not leak on its own failure
            pass
        try:
            # Inside the caller's scope: a redacting scope scrubs the known
            # secret values from this record's message and traceback.
            logger.warning(
                "Model provider %s failed: %s (%s)",
                operation,
                failure.code,
                type(exc).__name__,
                exc_info=exc,
            )
        except Exception:  # noqa: BLE001 - a failing log filter must not replace the failure
            pass
        self._report(failure.code)
        return failure

    async def _guarded_call(self, operation: str, **kwargs: Any) -> Any:
        failure: _Failure | None = None
        protocol_error: LLMToolProtocolError | None = None
        raw: Exception | None = None
        result: Any = None
        scope = self._open_scope()
        with scope:
            try:
                result = await getattr(self._inner, operation)(**kwargs)
            except LLMToolProtocolError as exc:
                protocol_error = _detached_protocol_error(exc)
            except Exception as exc:  # noqa: BLE001 - recorded below, outside the handler
                raw = exc
            if raw is not None:
                # Outside the handler, still inside the scope: nothing raised
                # while classifying, logging or reporting can carry the
                # provider's error as its context.
                failure = self._record_failure(raw, operation)
                raw = None
        # Raised outside the handlers, so neither error carries a context chain.
        if protocol_error is not None:
            raise protocol_error
        if failure is not None:
            raise _safe_error(failure)
        if scope.exit_failed:
            # A scope that failed while it was left fails the call: its
            # redaction may not have been undone, so do not hand back a
            # result as if the call had run as the caller set it up.
            raise ProviderCallError(CALL_SCOPE_UNAVAILABLE)
        return result

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
        return await self._guarded_call(  # type: ignore[no-any-return]
            "chat",
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
        return await self._guarded_call(  # type: ignore[no-any-return]
            "vision_chat",
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
        # The scope is entered for each step rather than held across ``yield``:
        # what the consumer does between chunks is not a provider call, and a
        # context variable set in one step must be reset in the same context.
        failure: _Failure | None = None
        protocol_error: LLMToolProtocolError | None = None
        raw: Exception | None = None
        stream: Any = None
        try:
            scope = self._open_scope()
            with scope:
                try:
                    stream = aiter(
                        self._inner.stream_chat(
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
                    )
                except Exception as exc:  # noqa: BLE001 - recorded below
                    raw = exc
                if raw is not None:
                    failure = self._record_failure(raw, "stream_chat")
                    raw = None
            if failure is None and scope.exit_failed:
                failure = _Failure(CALL_SCOPE_UNAVAILABLE)
            while failure is None and protocol_error is None:
                chunk: StreamChunk | None = None
                finished = False
                scope = self._open_scope()
                with scope:
                    try:
                        chunk = await anext(stream)
                    except StopAsyncIteration:
                        finished = True
                    except LLMToolProtocolError as exc:
                        protocol_error = _detached_protocol_error(exc)
                    except Exception as exc:  # noqa: BLE001 - recorded below
                        raw = exc
                    if raw is not None:
                        failure = self._record_failure(raw, "stream_chat")
                        raw = None
                    if (
                        chunk is not None
                        and getattr(chunk, "type", None) == ChunkType.ERROR
                    ):
                        failure = self._record_error_chunk(chunk)
                        chunk = None
                if scope.exit_failed and failure is None and protocol_error is None:
                    # Same rule as a single call: the step's outcome is not
                    # handed on when its scope failed while it was left.
                    failure = _Failure(CALL_SCOPE_UNAVAILABLE)
                    chunk = None
                if finished or chunk is None:
                    break
                yield chunk
        finally:
            await self._close_stream(stream)
        if protocol_error is not None:
            raise protocol_error
        if failure is not None:
            raise _safe_error(failure)

    def _record_error_chunk(self, chunk: StreamChunk) -> _Failure:
        """Classify an ``ERROR`` chunk as the error it reports. Inside scope."""
        raw = chunk.raw
        if isinstance(raw, BaseException):
            return self._record_failure(raw, "stream_chat")
        code = _classify_safely(RuntimeError(chunk.content or ""))
        try:
            logger.warning(
                "Model provider stream_chat reported an error chunk: %s", code
            )
        except Exception:  # noqa: BLE001 - a failing log filter must not replace the failure
            pass
        self._report(code)
        return _Failure(code)

    async def _close_stream(self, stream: Any) -> None:
        aclose = getattr(stream, "aclose", None)
        if not callable(aclose):
            return
        try:
            scope = self._open_scope()
        except ProviderCallError:
            # The scope could not be entered for cleanup (already reported);
            # the stream is left to garbage collection rather than closed
            # outside the scope.
            return
        # A failure while leaving this scope is reported by the scope itself
        # and, like every cleanup failure here, does not replace the outcome.
        close_error: Exception | None = None
        with scope:
            try:
                await aclose()
            except Exception as exc:  # noqa: BLE001 - cleanup must not replace the outcome
                close_error = exc
            if close_error is not None:
                code = _classify_safely(close_error)
                try:
                    logger.warning(
                        "Closing a model provider stream failed: %s (%s)",
                        code,
                        type(close_error).__name__,
                        exc_info=close_error,
                    )
                except Exception:  # noqa: BLE001 - a failing log filter changes nothing here
                    pass
                close_error = None


class _GuardedScope:
    """One entered caller scope whose own exit failure cannot escape.

    Used as the ``with`` target around one provider-touching step. When the
    caller's scope raises while it is left, the error is logged by type only
    and reported as ``call_scope_unavailable`` after the caller's exit attempt
    and :attr:`exit_failed` tells the step to fail. An exception already in
    flight (a cancellation, say) keeps propagating, with its exception chain
    cleared, even if the caller's scope would suppress it.
    """

    def __init__(self, owner: BoundaryLLM, stack: contextlib.ExitStack) -> None:
        self._owner = owner
        self._stack = stack
        self.exit_failed = False

    def __enter__(self) -> _GuardedScope:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc is not None and not isinstance(exc, Exception):
            # Only a cancellation or shutdown signal is in flight here (every
            # Exception is caught inside the step). One raised while a retry
            # layer was handling a provider error -- during its backoff sleep,
            # say -- would carry that error, text and all, as its context.
            exc.__context__ = None
            exc.__cause__ = None
        failed_type: str | None = None
        try:
            self._stack.__exit__(exc_type, exc, tb)
        except Exception as scope_exc:  # noqa: BLE001 - replaced by the step's outcome
            failed_type = type(scope_exc).__name__
        if failed_type is not None:
            self.exit_failed = True
            try:
                logger.warning(
                    "Model call scope could not be left cleanly (%s)", failed_type
                )
            except Exception:  # noqa: BLE001 - a failing log filter changes nothing here
                pass
            self._owner._report(CALL_SCOPE_UNAVAILABLE)
        # Never suppress (returns None): what is in flight around a provider
        # call keeps propagating even if the caller's scope would swallow it.


def guard_llm_calls(
    llm: BaseLLM,
    *,
    call_scope: Callable[[], AbstractContextManager[Any]],
    on_failure: Callable[[str], None] | None = None,
) -> BaseLLM:
    """Wrap ``llm`` so every provider call runs inside ``call_scope`` and fails safely.

    ``call_scope`` is entered afresh around each provider-touching operation
    and must be cheap and re-entrant. If it cannot be entered, or raises
    while it is left, the operation fails with ``call_scope_unavailable``: an
    otherwise successful result is not returned, an error the operation
    already raised keeps its own code, and closing a stream is only reported.

    ``on_failure`` is told the fixed code of each provider failure the
    wrapper replaces, from inside the scope, and ``call_scope_unavailable``
    when the scope itself could not be entered or left -- that one is
    reported after the failed entry or exit attempt. The caller is responsible
    for cleaning up any context state its scope changed before failing.
    One call can report more than once: a
    failed call whose scope then fails while it is left also reports
    ``call_scope_unavailable``, and so does a ``stream_chat`` whose scope
    cannot be re-entered to close the stream, after a successful stream or
    after a failed one; the last code reported can then differ from the
    raised error's. A failure while closing the stream is logged, not
    reported. Do not count calls to it or treat the last code as the call's
    outcome; the raised error's ``.code`` is authoritative. ``on_failure``
    must not raise; an exception from it is logged by type only.

    Wrapping an already-guarded model again is safe: both scopes are
    entered, and the inner boundary's error keeps its code (and
    ``transient`` flag). An :class:`UnavailableVisionModel` contacts no
    provider and is returned as it is.
    """
    if isinstance(llm, UnavailableVisionModel):
        return llm
    return BoundaryLLM(llm, call_scope=call_scope, on_failure=on_failure)


class UnavailableVisionModel(BaseLLM):
    """Stands in for a vision model that was deliberately left out.

    Every call raises ``ProviderCallError("vision_unavailable")`` and sends
    nothing anywhere. It is deliberately truthy: the tool configuration falls
    back to the owner's configured vision model whenever its explicit vision
    model is missing or falsy, and this stand-in is what keeps that fallback
    shut. Vision tools built on it refuse through their own
    ``has_ability("vision")`` check, with their own message, before calling
    it; ``apply_task_model_override`` therefore also drops the ``vision``
    tool category from an explicit selection, so those tools are not offered.
    """

    @property
    def abilities(self) -> List[str]:
        return []

    @property
    def model_name(self) -> str:
        return "vision-unavailable"

    @property
    def supports_thinking_mode(self) -> bool:
        return False

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
        raise ProviderCallError(VISION_UNAVAILABLE)

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
        raise ProviderCallError(VISION_UNAVAILABLE)
