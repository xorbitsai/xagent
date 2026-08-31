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

    Attributes:
        usage_attempts: Billed provider usage payloads of attempts made
            before this error was raised, oldest first. Adapters attach
            them when usage was already booked (``add_token_usage``) before
            the failure; the retry wrapper merges them into the eventual
            response envelope's ``usage_attempts`` (or onto the final
            exception when every attempt fails) so billed tokens are never
            lost from the execution context and trace. ``None`` means no
            billed attempts were recorded. Always ``None`` at construction;
            assigned by the adapter/wrapper afterwards.
    """

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self.usage_attempts: list[Any] | None = None


def attach_usage_attempts(error: BaseException, attempts: list[Any]) -> None:
    """Attach ordered billed-usage payloads to a surfaced exception.

    This is the accounting carrier for a call that fails after the adapter
    already booked tokens: the retry wrapper merges it into an eventual
    success envelope, and ``PatternRuntime.on_llm_error`` books it when every
    attempt fails. Assignment goes through ``setattr`` because only
    ``LLMRetryableError`` declares the attribute statically.
    """
    if attempts:
        setattr(error, "usage_attempts", list(attempts))


def merge_usage_attempts_into_result(history: list[Any], result: Any) -> None:
    """Fold billed attempts from superseded/failed retries into a success envelope.

    ``usage`` stays the final attempt (the context-freshness baseline);
    ``usage_attempts`` becomes the ordered list of every known billed
    attempt, final included. Written whenever ``history`` is non-empty: a
    known billed attempt is never suppressed just because the final success
    is unmetered or because a single attempt ended up known. A call with no
    prior billed history keeps the single-attempt shape (no key).
    """
    if not history or not isinstance(result, dict):
        return
    final_attempts = result.get("usage_attempts")
    if final_attempts is None:
        final_usage = result.get("usage")
        final_attempts = [final_usage] if final_usage is not None else []
    merged = list(history) + list(final_attempts)
    if merged:
        result["usage_attempts"] = merged


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
