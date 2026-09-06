"""LLM-specific exceptions for retry logic."""

from typing import Any


class LLMRetryableError(RuntimeError):
    """Base exception for LLM errors that should trigger retry.

    This exception is used for transient LLM errors that may succeed on retry,
    such as:
    - Empty content responses
    - Invalid API responses
    - Timeout errors
    - Rate limit errors (429)
    - Server errors (5xx)

    Subclass this exception for specific retryable error types.
    """

    pass


class LLMToolProtocolError(LLMRetryableError):
    """Structured provider tool-protocol failure.

    Most protocol failures can benefit from replaying the same request because
    the model may emit a valid structured call on the next sample. Failures that
    require changed agent context, including ``unavailable_tool_call`` and
    ``malformed_tool_arguments``, are excluded by the retry filter so the agent
    layer can retry with an explicit correction instead.
    """

    def __init__(
        self,
        *,
        provider: str,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.provider = str(provider or "unknown")
        self.code = str(code or "invalid_tool_protocol")
        self.protocol_message = str(message or "Invalid tool protocol response.")
        self.details = dict(details or {})
        super().__init__(
            f"{self.provider} tool protocol error ({self.code}): "
            f"{self.protocol_message}"
        )


class LLMEmptyContentError(LLMRetryableError):
    """Raised when LLM returns empty content with no tool calls.

    This is a transient error that may occur due to:
    - API temporary issues
    - Rate limiting
    - Network glitches
    - Model-specific behavior

    The request should be retried.
    """

    pass


class LLMInvalidResponseError(LLMRetryableError):
    """Raised when LLM response cannot be parsed or is invalid.

    This includes:
    - Malformed JSON responses
    - Missing required fields
    - Unexpected response structure
    - Cannot decode response

    The request should be retried.
    """

    pass


class LLMNoTextContentError(LLMInvalidResponseError):
    """Raised when a chat response carries no usable text content.

    Distinct from ``LLMEmptyContentError`` (the provider answered with an
    empty string): this means the response has a non-text shape -- a
    tool_call envelope or an unrecognized payload -- where the caller
    required text. Stringifying such a response would leak an internal
    dict repr into compacted context or API responses as if it were model
    output (#1714), so consumers must fail explicitly instead.
    """

    pass


class LLMTimeoutError(LLMRetryableError):
    """Raised when LLM request times out.

    This includes:
    - First token timeout (no response within configured time)
    - Token interval timeout (gap between tokens exceeds configured time)
    - Network timeout

    The request should be retried.
    """

    pass
