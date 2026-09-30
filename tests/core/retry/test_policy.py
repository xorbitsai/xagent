"""Bounds that attempt counting cannot express: deadlines and capacity budgets."""

import email.utils
import time

import httpx
import openai
import pytest

from xagent.config import (
    get_llm_capacity_max_attempts,
    get_llm_retry_deadline_seconds,
)
from xagent.core.retry.policy import (
    MAX_HONORED_RETRY_AFTER_SECONDS,
    RetryBudget,
    chat_retry_budget,
    is_capacity_error,
    retry_after_seconds,
)

REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _status_error(status: int, message: str, headers=None) -> openai.APIStatusError:
    response = httpx.Response(status, request=REQUEST, headers=headers or {})
    return openai.APIStatusError(message, response=response, body=None)


class TestIsCapacityError:
    @pytest.mark.parametrize(
        "message",
        [
            "Exceeded on-demand capacity. Ensure your workload does not double "
            "faster than once per 30 minutes.",
            "EXCEEDED ON-DEMAND CAPACITY.",
            "The engine is currently overloaded, please try again later",
            "insufficient capacity for this model",
        ],
    )
    def test_recognizes_provider_capacity_refusals(self, message):
        assert is_capacity_error(RuntimeError(message)) is True

    @pytest.mark.parametrize(
        "message",
        [
            "Rate limit reached for gpt-4o in organization org-x",
            "Internal server error",
            "context_length_exceeded",
            "Connection error.",
            "",
        ],
    )
    def test_leaves_other_failures_unclassified(self, message):
        assert is_capacity_error(RuntimeError(message)) is False

    def test_walks_the_cause_chain_through_runtimeerror_wrapping(self):
        """``OpenAILLM.chat`` re-raises every provider error as ``RuntimeError``."""
        cause = _status_error(503, "Exceeded on-demand capacity.")
        try:
            raise RuntimeError("OpenAI API error: server overloaded") from cause
        except RuntimeError as wrapped:
            assert is_capacity_error(wrapped) is True

    def test_survives_a_self_referential_cause_chain(self):
        first = RuntimeError("first")
        second = RuntimeError("second")
        first.__cause__ = second
        second.__cause__ = first

        assert is_capacity_error(first) is False


class TestRetryAfterSeconds:
    def test_reads_retry_after_seconds_header(self):
        error = _status_error(429, "slow down", headers={"retry-after": "12"})

        assert retry_after_seconds(error) == pytest.approx(12.0)

    def test_reads_retry_after_ms_header(self):
        error = _status_error(429, "slow down", headers={"retry-after-ms": "2500"})

        assert retry_after_seconds(error) == pytest.approx(2.5)

    def test_prefers_the_millisecond_header_when_both_are_present(self):
        error = _status_error(
            429, "slow down", headers={"retry-after": "30", "retry-after-ms": "500"}
        )

        assert retry_after_seconds(error) == pytest.approx(0.5)

    def test_ignores_a_hint_longer_than_we_are_willing_to_hold_a_slot(self):
        error = _status_error(429, "slow down", headers={"retry-after": "600"})

        assert retry_after_seconds(error) is None

    def test_ignores_a_non_positive_hint(self):
        error = _status_error(429, "slow down", headers={"retry-after": "0"})

        assert retry_after_seconds(error) is None

    def test_ignores_an_unparseable_hint(self):
        error = _status_error(
            429, "slow down", headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}
        )

        assert retry_after_seconds(error) is None

    def test_walks_the_cause_chain(self):
        cause = _status_error(429, "slow down", headers={"retry-after": "7"})
        try:
            raise RuntimeError("OpenAI rate limit exceeded: slow down") from cause
        except RuntimeError as wrapped:
            assert retry_after_seconds(wrapped) == pytest.approx(7.0)

    def test_returns_none_without_a_response(self):
        assert retry_after_seconds(RuntimeError("boom")) is None

    def test_honored_ceiling_matches_the_sdk_we_stop_relying_on(self):
        assert MAX_HONORED_RETRY_AFTER_SECONDS == 60.0


class TestRetryBudget:
    def test_an_empty_budget_imposes_nothing(self):
        budget = RetryBudget()

        assert budget.deadline_seconds is None
        assert budget.attempt_limit(RuntimeError("over capacity")) is None

    def test_capacity_refusals_get_the_short_budget(self):
        budget = RetryBudget(deadline_seconds=300.0, capacity_max_attempts=2)

        assert budget.attempt_limit(RuntimeError("Exceeded on-demand capacity.")) == 2

    def test_transient_faults_keep_the_configured_attempt_count(self):
        budget = RetryBudget(deadline_seconds=300.0, capacity_max_attempts=2)

        assert budget.attempt_limit(RuntimeError("Connection error.")) is None


class TestChatRetryBudget:
    def test_reflects_the_configured_bounds(self, monkeypatch):
        """The factory is wiring only; parsing is covered in test_config.py."""
        monkeypatch.delenv("XAGENT_LLM_RETRY_DEADLINE_SECONDS", raising=False)
        monkeypatch.delenv("XAGENT_LLM_CAPACITY_MAX_ATTEMPTS", raising=False)

        budget = chat_retry_budget()

        assert budget.deadline_seconds == get_llm_retry_deadline_seconds()
        assert budget.capacity_max_attempts == get_llm_capacity_max_attempts()

    def test_reads_the_configuration_at_call_time(self, monkeypatch):
        """A deployment override must reach models built after it is set."""
        monkeypatch.setenv("XAGENT_LLM_RETRY_DEADLINE_SECONDS", "45")
        monkeypatch.setenv("XAGENT_LLM_CAPACITY_MAX_ATTEMPTS", "1")

        budget = chat_retry_budget()

        assert budget.deadline_seconds == pytest.approx(45.0)
        assert budget.capacity_max_attempts == 1

    def test_unusable_configuration_still_yields_a_bound(self, monkeypatch):
        """Bad config must not silently restore the unbounded loop.

        Asserted against the accessors rather than against literals, so the
        defaults have one home (``xagent.config``) and cannot drift.
        """
        monkeypatch.delenv("XAGENT_LLM_RETRY_DEADLINE_SECONDS", raising=False)
        monkeypatch.delenv("XAGENT_LLM_CAPACITY_MAX_ATTEMPTS", raising=False)
        default_deadline = get_llm_retry_deadline_seconds()
        default_attempts = get_llm_capacity_max_attempts()

        monkeypatch.setenv("XAGENT_LLM_RETRY_DEADLINE_SECONDS", "0")
        monkeypatch.setenv("XAGENT_LLM_CAPACITY_MAX_ATTEMPTS", "-1")

        budget = chat_retry_budget()

        assert budget.deadline_seconds == default_deadline
        assert budget.capacity_max_attempts == default_attempts
        assert budget.deadline_seconds is not None


class TestChainWalkIsExplicitOnly:
    """The walk follows ``__cause__`` only, and a test has to hold that.

    ``__context__`` is set implicitly whenever an exception is raised while
    another is being handled, so a marker found there can belong to an
    unrelated cleanup failure. Misclassifying a transient fault as a capacity
    refusal cuts its budget from the model's full ``max_retries`` to two, so
    the fallback costs resilience and must not come back.
    """

    def test_a_context_only_link_is_not_followed(self):
        capacity = RuntimeError("Exceeded on-demand capacity.")
        try:
            try:
                raise capacity
            except RuntimeError:
                # No ``from``, so only __context__ links the two.
                raise RuntimeError("connection reset by peer")
        except RuntimeError as unrelated:
            assert unrelated.__context__ is capacity
            assert unrelated.__cause__ is None
            assert is_capacity_error(unrelated) is False

    def test_an_explicit_cause_is_followed(self):
        capacity = RuntimeError("Exceeded on-demand capacity.")
        try:
            raise RuntimeError("OpenAI API error") from capacity
        except RuntimeError as wrapped:
            assert is_capacity_error(wrapped) is True

    def test_retry_after_also_ignores_a_context_only_link(self):
        hinted = _status_error(429, "slow down", headers={"retry-after": "9"})
        try:
            try:
                raise hinted
            except openai.APIStatusError:
                raise RuntimeError("unrelated cleanup failure")
        except RuntimeError as unrelated:
            assert retry_after_seconds(unrelated) is None


class TestRetryAfterHttpDate:
    """A date is a legal ``Retry-After``, and the SDKs we replaced honoured it.

    ``openai`` 2.24.0 and ``anthropic`` 0.84.0 both fall through to
    ``email.utils.parsedate_tz`` and return ``date - time.time()``, subject to
    the same 60s ceiling. Dropping a date meant a capacity refusal spent its
    single retry on the ~200ms local backoff against a provider that had named
    a later time.
    """

    def _dated(self, offset_seconds: float) -> openai.APIStatusError:
        value = email.utils.formatdate(time.time() + offset_seconds, usegmt=True)
        return _status_error(429, "slow down", headers={"retry-after": value})

    def test_a_future_date_is_honoured(self):
        assert retry_after_seconds(self._dated(30)) == pytest.approx(30, abs=2)

    def test_a_past_date_is_ignored(self):
        assert retry_after_seconds(self._dated(-30)) is None

    def test_a_date_beyond_the_ceiling_is_ignored(self):
        """Same bound as a numeric hint: we will not hold a slot that long."""
        assert retry_after_seconds(self._dated(600)) is None

    def test_a_numeric_hint_still_wins_over_a_date(self):
        error = _status_error(
            429,
            "slow down",
            headers={
                "retry-after": "5",
                "retry-after-ms": "1500",
            },
        )

        assert retry_after_seconds(error) == pytest.approx(1.5)

    def test_unparseable_text_is_still_ignored(self):
        error = _status_error(429, "slow down", headers={"retry-after": "soon"})

        assert retry_after_seconds(error) is None
