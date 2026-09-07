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

        # Handle Zai/Zhipu SDK errors
        if ZaiAPIStatusError and isinstance(exc, ZaiAPIStatusError):
            return bool(exc.status_code == 429 or 500 <= exc.status_code < 600)

        return isinstance(exc, ERRORS)

    if _is_retryable(e):
        return True

    # Check the underlying cause (fix for RuntimeError wrapping)
    if e.__cause__ and _is_retryable(e.__cause__):
        return True

    return False
