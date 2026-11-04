from __future__ import annotations

import httpx
import openai
import pytest
from sqlalchemy import exc as sa_exc

from xagent.core.agent.checkpoint import (
    CheckpointPersistenceError,
    ExecutionEventPersistenceError,
)
from xagent.core.agent.exceptions import MaxIterationsError
from xagent.core.agent.interruption import (
    InterruptionReason,
    classify_run_failure,
    classify_run_result,
    interruption_reason_value,
    is_database_unavailable,
)
from xagent.core.model.chat.exceptions import (
    LLMContextLengthError,
    LLMEmptyContentError,
    LLMInvalidResponseError,
    LLMTimeoutError,
    LLMToolProtocolError,
    ModelProviderError,
)
from xagent.core.tools.adapters.vibe.connector_runtime import ConnectorRuntimeError
from xagent.web.services.llm_utils import AutoModelUnavailableError

REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _status_error(status: int, message: str = "provider error") -> Exception:
    response = httpx.Response(status, request=REQUEST)
    return openai.APIStatusError(message, response=response, body=None)


def _rate_limit_error() -> Exception:
    response = httpx.Response(429, request=REQUEST)
    return openai.RateLimitError("slow down", response=response, body=None)


def _chained(outer: BaseException, cause: BaseException) -> BaseException:
    outer.__cause__ = cause
    return outer


def _provider_error_from(cause: BaseException, *, status: int | None) -> Exception:
    error = ModelProviderError(
        prefix="Provider API error",
        kind="unknown",
        status_code=status,
        provider_code=None,
        provider_message=None,
        sdk_message=str(cause),
        details=[],
    )
    error.__cause__ = cause
    return error


def _timeout_error() -> Exception:
    return openai.APITimeoutError(request=REQUEST)


def _tool_protocol_error(code: str) -> LLMToolProtocolError:
    return LLMToolProtocolError(provider="openai", code=code, message="bad call")


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        # Persistence.
        (
            ExecutionEventPersistenceError("event write failed"),
            InterruptionReason.PERSISTENCE_FAILURE,
        ),
        (
            CheckpointPersistenceError("checkpoint write failed"),
            InterruptionReason.PERSISTENCE_FAILURE,
        ),
        (
            sa_exc.OperationalError("SELECT 1", {}, Exception("server closed")),
            InterruptionReason.PERSISTENCE_FAILURE,
        ),
        (
            sa_exc.InterfaceError("SELECT 1", {}, Exception("connection closed")),
            InterruptionReason.PERSISTENCE_FAILURE,
        ),
        (
            sa_exc.DBAPIError(
                "SELECT 1", {}, Exception("reset"), connection_invalidated=True
            ),
            InterruptionReason.PERSISTENCE_FAILURE,
        ),
        (
            sa_exc.TimeoutError("QueuePool limit reached"),
            InterruptionReason.PERSISTENCE_FAILURE,
        ),
        # Unusable model output, including the retryable protocol codes that
        # ``retry_on`` would otherwise accept.
        (
            _tool_protocol_error("invalid_tool_protocol"),
            InterruptionReason.MODEL_OUTPUT_INVALID,
        ),
        (
            _tool_protocol_error("malformed_tool_arguments"),
            InterruptionReason.MODEL_OUTPUT_INVALID,
        ),
        (
            LLMEmptyContentError("empty content"),
            InterruptionReason.MODEL_OUTPUT_INVALID,
        ),
        (
            LLMInvalidResponseError("unparsable"),
            InterruptionReason.MODEL_OUTPUT_INVALID,
        ),
        # Provider unavailable.
        (_rate_limit_error(), InterruptionReason.LLM_UNAVAILABLE),
        (_status_error(503), InterruptionReason.LLM_UNAVAILABLE),
        (
            httpx.ReadTimeout("read timed out", request=REQUEST),
            InterruptionReason.LLM_UNAVAILABLE,
        ),
        (LLMTimeoutError("first token timeout"), InterruptionReason.LLM_UNAVAILABLE),
        (
            RuntimeError("Provider is overloaded, try later"),
            InterruptionReason.LLM_UNAVAILABLE,
        ),
        # Terminal.
        (_status_error(401, "invalid api key"), None),
        (_status_error(400, "bad request"), None),
        (AutoModelUnavailableError("no usable model"), None),
        (ConnectorRuntimeError("connector_missing", "connector missing"), None),
        (MaxIterationsError("ReActPattern", 10), None),
        (ValueError("tool exploded"), None),
        (
            sa_exc.DBAPIError("SELECT 1", {}, Exception("constraint")),
            None,
        ),
    ],
)
def test_classify_run_failure_branches(
    error: BaseException, expected: InterruptionReason | None
) -> None:
    assert classify_run_failure(error) is expected


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            _provider_error_from(_rate_limit_error(), status=429),
            InterruptionReason.LLM_UNAVAILABLE,
        ),
        (
            _provider_error_from(_status_error(503), status=503),
            InterruptionReason.LLM_UNAVAILABLE,
        ),
        (
            _provider_error_from(_timeout_error(), status=None),
            InterruptionReason.LLM_UNAVAILABLE,
        ),
        (_provider_error_from(_status_error(401), status=401), None),
        (_provider_error_from(_status_error(403), status=403), None),
        (_provider_error_from(_status_error(404), status=404), None),
        (_provider_error_from(_status_error(400), status=400), None),
    ],
    ids=["429", "503", "timeout", "401", "403", "404", "400"],
)
def test_classify_run_failure_for_model_provider_errors(
    error: BaseException, expected: InterruptionReason | None
) -> None:
    assert classify_run_failure(error) is expected


@pytest.mark.parametrize(
    "error",
    [
        LLMContextLengthError("window exceeded"),
        RuntimeError("maximum context length is 8192 tokens"),
        _chained(RuntimeError("llm call failed"), LLMContextLengthError("too long")),
        # Context length wins even over a transient-looking or persistence
        # wrapper: the task cannot fit, however often it is retried.
        _chained(_rate_limit_error(), LLMContextLengthError("too long")),
        _chained(
            CheckpointPersistenceError("write failed"),
            RuntimeError("prompt is too long"),
        ),
    ],
)
def test_context_length_failures_are_terminal(error: BaseException) -> None:
    assert classify_run_failure(error) is None


def _insufficient_quota_error() -> Exception:
    response = httpx.Response(429, request=REQUEST)
    return openai.RateLimitError(
        "You exceeded your current quota, please check your plan and billing",
        response=response,
        body={"code": "insufficient_quota", "type": "insufficient_quota"},
    )


@pytest.mark.parametrize(
    "error",
    [
        _insufficient_quota_error(),
        # How openai.py re-raises an SDK rate-limit error.
        _chained(
            RuntimeError("OpenAI rate limit exceeded: quota"),
            _insufficient_quota_error(),
        ),
        _chained(RuntimeError("chat failed"), _status_error(402, "pay up")),
        _chained(RuntimeError("chat failed"), _status_error(401, "bad key")),
    ],
    ids=["quota", "quota_wrapped", "payment_required", "credential_wrapped"],
)
def test_quota_and_credential_refusals_are_terminal(error: BaseException) -> None:
    assert classify_run_failure(error) is None


def test_plain_rate_limit_stays_resumable() -> None:
    assert classify_run_failure(_rate_limit_error()) is (
        InterruptionReason.LLM_UNAVAILABLE
    )
    wrapped = _chained(
        RuntimeError("OpenAI rate limit exceeded: slow down"), _rate_limit_error()
    )
    assert classify_run_failure(wrapped) is InterruptionReason.LLM_UNAVAILABLE


def test_classifier_walks_explicit_cause_chain() -> None:
    db_down = sa_exc.OperationalError("SELECT 1", {}, Exception("server closed"))
    wrapped = _chained(
        RuntimeError("pattern failed"), _chained(RuntimeError("step failed"), db_down)
    )

    assert classify_run_failure(wrapped) is InterruptionReason.PERSISTENCE_FAILURE

    provider = _chained(
        RuntimeError("chat failed"),
        _chained(RuntimeError("adapter failed"), _status_error(502)),
    )
    assert classify_run_failure(provider) is InterruptionReason.LLM_UNAVAILABLE


def test_model_output_invalid_precedes_retry_on_through_wrappers() -> None:
    # ``retry_on`` on the wrapper looks one ``__cause__`` deep and accepts the
    # ``LLMRetryableError`` subclass there; that must not turn an invalid
    # model output into an unavailable provider.
    wrapped = _chained(RuntimeError("llm failed"), LLMEmptyContentError("empty"))
    assert classify_run_failure(wrapped) is InterruptionReason.MODEL_OUTPUT_INVALID

    protocol_over_rate_limit = _chained(
        _tool_protocol_error("invalid_tool_protocol"), _rate_limit_error()
    )
    assert (
        classify_run_failure(protocol_over_rate_limit)
        is InterruptionReason.MODEL_OUTPUT_INVALID
    )


def test_persistence_precedes_provider_failures() -> None:
    error = _chained(ExecutionEventPersistenceError("event lost"), _status_error(503))
    assert classify_run_failure(error) is InterruptionReason.PERSISTENCE_FAILURE


def test_implicit_context_is_not_followed() -> None:
    def raise_during_cleanup(context: BaseException) -> BaseException:
        try:
            raise context
        except BaseException:
            try:
                raise ValueError("tool exploded")
            except ValueError as error:
                return error

    persistence_context = raise_during_cleanup(
        CheckpointPersistenceError("unrelated cleanup")
    )
    assert persistence_context.__context__ is not None
    assert classify_run_failure(persistence_context) is None

    provider_context = raise_during_cleanup(_status_error(503))
    assert classify_run_failure(provider_context) is None

    # A context-length error found only through ``__context__`` does not
    # mask a real persistence failure either.
    try:
        raise LLMContextLengthError("too long")
    except LLMContextLengthError:
        try:
            raise CheckpointPersistenceError("write failed")
        except CheckpointPersistenceError as error:
            persistence = error
    assert classify_run_failure(persistence) is InterruptionReason.PERSISTENCE_FAILURE


def test_cyclic_cause_chain_terminates() -> None:
    first = RuntimeError("first")
    second = RuntimeError("second")
    first.__cause__ = second
    second.__cause__ = first

    assert classify_run_failure(first) is None


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (
            {"success": False, "status": "invalid_tool_protocol"},
            InterruptionReason.MODEL_OUTPUT_INVALID,
        ),
        (
            {"success": False, "interruption_reason": "llm_unavailable"},
            InterruptionReason.LLM_UNAVAILABLE,
        ),
        (
            {
                "success": False,
                "interruption_reason": InterruptionReason.PERSISTENCE_FAILURE,
            },
            InterruptionReason.PERSISTENCE_FAILURE,
        ),
        ({"success": False, "interruption_reason": "not-a-reason"}, None),
        ({"success": False, "interruption_reason": None}, None),
        ({"success": False, "interruption_reason": 3}, None),
        ({"success": False, "status": "failed", "error": "boom"}, None),
        ({"success": True, "output": "done"}, None),
        (None, None),
        ("failed", None),
    ],
)
def test_classify_run_result(
    result: object, expected: InterruptionReason | None
) -> None:
    assert classify_run_result(result) is expected


def test_reason_values_are_stable_strings() -> None:
    assert [reason.value for reason in InterruptionReason] == [
        "lease_expired",
        "persistence_failure",
        "llm_unavailable",
        "model_output_invalid",
        "shutdown",
        "user_pause",
        "input_outcome_unknown",
        "unknown_tool_effect",
        "not_recoverable",
    ]
    assert interruption_reason_value(InterruptionReason.LLM_UNAVAILABLE) == (
        "llm_unavailable"
    )
    assert interruption_reason_value(None) is None


def test_is_database_unavailable_follows_the_cause_chain():
    from sqlalchemy.exc import OperationalError
    from sqlalchemy.exc import TimeoutError as PoolTimeoutError

    reset = OperationalError("SELECT", {}, Exception("connection reset"))
    wrapped = RuntimeError("read failed")
    wrapped.__cause__ = reset

    assert is_database_unavailable(reset)
    assert is_database_unavailable(wrapped)
    assert is_database_unavailable(PoolTimeoutError("pool exhausted"))
    assert not is_database_unavailable(TypeError("resolver bug"))
    context_only = RuntimeError("cleanup")
    context_only.__context__ = reset
    assert not is_database_unavailable(context_only)


@pytest.mark.parametrize(
    ("code", "transient", "expected"),
    [
        ("provider_quota", True, None),
        ("credential_rejected", True, None),
        ("rate_limited", False, InterruptionReason.LLM_UNAVAILABLE),
        ("timeout", False, InterruptionReason.LLM_UNAVAILABLE),
        ("provider_unavailable", False, InterruptionReason.LLM_UNAVAILABLE),
        ("provider_error", True, InterruptionReason.LLM_UNAVAILABLE),
        ("provider_error", False, None),
    ],
)
def test_guarded_provider_failure_is_classified_by_its_code(code, transient, expected):
    """A guard_llm_calls model raises a cause-less ProviderCallError, which
    retry_on and the text markers cannot read; its code decides instead."""
    from xagent.core.model.chat.basic.call_boundary import ProviderCallError

    error = ProviderCallError(code, transient=transient)
    assert error.__cause__ is None and error.__context__ is None
    assert classify_run_failure(error) is expected
    wrapped = RuntimeError("pattern failed")
    wrapped.__cause__ = error
    assert classify_run_failure(wrapped) is expected
