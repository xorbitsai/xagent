import httpx
import openai

from .exceptions import (
    LLMContextLengthError,
    LLMRetryableError,
    LLMToolProtocolError,
)

try:
    from zai.core._errors import APIStatusError as ZaiAPIStatusError  # type: ignore
except ImportError:
    ZaiAPIStatusError = None


_CONTEXT_LENGTH_ERROR_MARKERS = (
    "context_length_exceeded",
    "context length exceeded",
    "maximum context length",
    "exceeds the context window",
    "exceeds the maximum number of tokens allowed",
    "input is too long",
    "prompt is too long",
    "too many input tokens",
)


def is_context_length_error(error: BaseException) -> bool:
    """Recognize provider context-window failures through wrapper exceptions."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, LLMContextLengthError):
            return True
        message = str(current).lower()
        if any(marker in message for marker in _CONTEXT_LENGTH_ERROR_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False


# Transient HTTP statuses, in one place so every consumer compares the same
# set. 408 and 409 are here because the provider SDKs retried them and this
# layer replaced those budgets (see ``OpenAICompatibleLLM._ensure_client``);
# omitting them would fail a request the SDK would have recovered.
_TRANSIENT_HTTP_STATUSES = frozenset({408, 409, 429})


def is_retryable_http_status(status: int) -> bool:
    """Whether an HTTP status is worth replaying the identical request for."""
    return status in _TRANSIENT_HTTP_STATUSES or 500 <= status < 600


def _status_code_of(error: BaseException) -> int | None:
    """Read a provider status error's HTTP status, or ``None`` if unusable."""
    status = getattr(error, "status_code", None)
    return status if isinstance(status, int) else None


def _provider_vetoed_retry(error: BaseException) -> bool:
    """Whether the provider told us not to replay this response.

    ``x-should-retry: false`` is an explicit veto that the SDK honoured ahead
    of its own status classification. Disabling the SDK's retries took that
    precedence with it, so this layer has to apply it.
    """
    response = getattr(error, "response", None)
    getter = getattr(getattr(response, "headers", None), "get", None)
    if not callable(getter):
        return False
    return str(getter("x-should-retry") or "").strip().lower() == "false"


def retry_on(e: Exception) -> bool:
    if is_context_length_error(e):
        return False

    ERRORS = (
        httpx.TimeoutException,
        httpx.NetworkError,
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.APITimeoutError,
    )

    def _is_retryable(exc: BaseException) -> bool:
        # These failures need a changed agent-level decision or repair prompt.
        # Blindly replaying the identical provider request only burns the retry
        # budget and hides the structured error from the execution pattern.
        if isinstance(exc, LLMToolProtocolError):
            return exc.code not in {
                "malformed_tool_arguments",
                "unavailable_tool_call",
            }

        # Handle LLM-specific retryable errors
        # These are explicitly marked as retryable by the LLM implementation
        if isinstance(exc, LLMRetryableError):
            return True

        # Handle httpx errors
        if isinstance(exc, httpx.HTTPStatusError):
            return (
                exc.response.status_code == 429 or 500 <= exc.response.status_code < 600
            )

        # Handle openai SDK status errors. The status lives on the exception,
        # not on an httpx response the branch above can reach, so a provider
        # 5xx arriving through the openai SDK used to fall through to the
        # tuple test below and be treated as permanent. The SDK's own retry
        # budget hid that; now that the clients are built with
        # ``max_retries=0`` (see ``OpenAICompatibleLLM._ensure_client``) this
        # is the only layer left to recognize them, which is also what
        # ``OpenRouterLLM._chat_with_compat_retry``'s contract already assumed.
        #
        # Deliberately additive: these branches can only turn a "no" into a
        # "yes" and never the reverse, so no failure shape that retries today
        # stops retrying. An unusable status falls through to the tuple test
        # rather than raising out of the predicate, which would turn a
        # retryable provider error into an unrelated crash.
        #
        # The veto is checked here rather than around the whole predicate for
        # the same reason: a vetoed 429 reaches the tuple test below and keeps
        # retrying exactly as it did before this layer existed. Narrowing that
        # is a behaviour change, not a repair of what this layer introduced.
        if isinstance(exc, openai.APIStatusError) and not _provider_vetoed_retry(exc):
            status = _status_code_of(exc)
            if status is not None and is_retryable_http_status(status):
                return True

        # Handle Zai/Zhipu SDK errors
        if (
            ZaiAPIStatusError
            and isinstance(exc, ZaiAPIStatusError)
            and not _provider_vetoed_retry(exc)
        ):
            status = _status_code_of(exc)
            if status is not None and is_retryable_http_status(status):
                return True

        return isinstance(exc, ERRORS)

    if _is_retryable(e):
        return True

    # Check the underlying cause (fix for RuntimeError wrapping)
    if e.__cause__ and _is_retryable(e.__cause__):
        return True

    return False
