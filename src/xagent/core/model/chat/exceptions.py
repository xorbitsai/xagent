"""LLM-specific exceptions for retry logic."""

from typing import Any


class LLMContextLengthError(RuntimeError):
    """The provider rejected an input because it exceeded the model window."""

    pass


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


class LLMTimeoutError(LLMRetryableError):
    """Raised when LLM request times out.

    This includes:
    - First token timeout (no response within configured time)
    - Token interval timeout (gap between tokens exceeds configured time)
    - Network timeout

    The request should be retried.
    """

    pass


MODEL_PROVIDER_FAILURE_KINDS = frozenset(
    {
        "timeout",  # openai.APITimeoutError, or HTTP 408
        "connection_failed",  # openai.APIConnectionError that is not a timeout
        "bad_request",  # HTTP 400
        "authentication_failed",  # HTTP 401
        "access_denied",  # HTTP 403
        "not_found",  # HTTP 404
        "rate_limited",  # HTTP 429
        "server_error",  # HTTP 5xx
        "rejected",  # any other 4xx (402, 409, 422, ...)
        "unknown",  # no HTTP status and not a transport failure
    }
)


def format_model_provider_error(
    prefix: str,
    status_code: object | None,
    message: str,
    details: list[str],
) -> str:
    """The text of a provider failure.

    ``"<prefix> (<status_code>): <message>"``, or ``"<prefix>: <message>"``
    when ``status_code`` is ``None``, followed by ``" | "`` and the ``details``
    joined with ``" | "`` when ``details`` is not empty.
    """
    text = prefix
    if status_code is not None:
        text = f"{text} ({status_code})"
    text = f"{text}: {message}"
    if details:
        text = f"{text} | " + " | ".join(details)
    return text


class ModelProviderError(RuntimeError):
    """The model provider rejected or failed a request; the response survives as fields.

    Fields: ``prefix``, ``kind`` (one of ``MODEL_PROVIDER_FAILURE_KINDS``),
    ``status_code`` (``int | None``), ``provider_code`` (``str | None``),
    ``provider_message`` (``str | None``, the error body's own ``message``,
    capped), ``sdk_message`` (``str``, the SDK exception's message, capped)
    and ``details`` (``list[str]``, the diagnostic suffixes the adapter
    builds).

    ``str()`` is :func:`format_model_provider_error` applied to ``prefix``,
    ``status_code``, ``sdk_message`` and ``details``.

    Not retryable by class: the retry predicate reads ``__cause__``, which
    every raise site sets. Use :class:`ModelProviderRetryableError` where the
    failure is retryable by class.
    """

    def __init__(
        self,
        *,
        prefix: str,
        kind: str,
        status_code: int | None,
        provider_code: str | None,
        provider_message: str | None,
        sdk_message: str,
        details: list[str],
    ) -> None:
        self.prefix = prefix
        self.kind = kind
        self.status_code = status_code
        self.provider_code = provider_code
        self.provider_message = provider_message
        self.sdk_message = sdk_message
        self.details = list(details)
        super().__init__(
            format_model_provider_error(prefix, status_code, sdk_message, self.details)
        )

    def structured_fields(self) -> dict[str, Any]:
        """``kind``, ``status_code``, ``provider_code`` and ``message`` as a new dict.

        ``message`` is ``provider_message``: the provider's own text as
        received (capped), not rewritten for any audience. It is not
        client-safe by itself, so every surface shown to an end user or to
        another party must pass it through the web-layer projection first.
        ``prefix``, ``sdk_message`` and ``details`` are deliberately left out.
        """
        return {
            "kind": self.kind,
            "status_code": self.status_code,
            "provider_code": self.provider_code,
            "message": self.provider_message,
        }


class ModelProviderRetryableError(ModelProviderError, LLMRetryableError):
    """Same fields and text as :class:`ModelProviderError`, retryable by class.

    Raised only where the adapter already raised ``LLMRetryableError`` (stream
    timeouts, rate limits and connection failures), so
    ``isinstance(x, LLMRetryableError)`` is unchanged for those failures.
    """
