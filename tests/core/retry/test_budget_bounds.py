"""``RetryWrapper`` honours a :class:`RetryBudget` on top of its attempt count."""

import logging

import httpx
import openai
import pytest

from xagent.core.retry import wrapper as wrapper_module
from xagent.core.retry.policy import RetryBudget
from xagent.core.retry.strategy import FixedDelay
from xagent.core.retry.wrapper import RetryWrapper, create_retry_wrapper

CAPACITY_MESSAGE = (
    "Exceeded on-demand capacity. Ensure your workload does not double "
    "faster than once per 30 minutes."
)
REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


class FakeTime:
    """A simulated clock for the wrapper's ``time`` module.

    Reading the clock does not advance it, so a test's expectations follow
    from the simulated durations rather than from how many times the
    implementation happens to call ``monotonic()``.
    """

    def __init__(self, overshoot: float = 0.0) -> None:
        self.now = 0.0
        self.slept: list[float] = []
        # A real sleep can wake late: a busy scheduler, a blocked event loop.
        self.overshoot = overshoot

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds + self.overshoot


class FakeAsyncio:
    """Stands in for the wrapper's ``asyncio`` module.

    The wrapper's two async loops sleep through ``asyncio.sleep``, so a
    ``time.sleep`` stub alone never observes them.
    """

    def __init__(self, clock: FakeTime) -> None:
        self.clock = clock

    async def sleep(self, seconds: float) -> None:
        self.clock.sleep(seconds)


class AlwaysFails:
    def __init__(self, error: BaseException, clock=None, duration: float = 0.0):
        self.error = error
        self.clock = clock
        self.duration = duration
        self.call_count = 0

    def _run(self):
        self.call_count += 1
        # Simulate an attempt that consumes wall clock, which is what makes a
        # deadline bite before the attempt count runs out.
        if self.clock is not None:
            self.clock.now += self.duration
        raise self.error

    def invoke(self, *args, **kwargs):
        self._run()

    async def ainvoke(self, *args, **kwargs):
        self._run()


@pytest.fixture
def clock(monkeypatch):
    """Install the simulated clock on both sleep paths the wrapper uses."""
    fake = FakeTime()
    monkeypatch.setattr(wrapper_module, "time", fake)
    monkeypatch.setattr(wrapper_module, "asyncio", FakeAsyncio(fake))
    return fake


@pytest.fixture
def late_clock(monkeypatch):
    """A clock whose sleeps wake 20 simulated seconds late."""
    fake = FakeTime(overshoot=20.0)
    monkeypatch.setattr(wrapper_module, "time", fake)
    monkeypatch.setattr(wrapper_module, "asyncio", FakeAsyncio(fake))
    return fake


def _retry_after_error(seconds: str) -> RuntimeError:
    """The shape a provider throttle reaches the wrapper in."""
    response = httpx.Response(429, request=REQUEST, headers={"retry-after": seconds})
    cause = openai.RateLimitError("slow down", response=response, body=None)
    error = RuntimeError("OpenAI rate limit exceeded: slow down")
    error.__cause__ = cause
    return error


def _wrapper(target, **kwargs) -> RetryWrapper:
    kwargs.setdefault("strategy", FixedDelay(delay_ms=1))
    kwargs.setdefault("max_retries", 10)
    return RetryWrapper(target, **kwargs)


class TestNoBudget:
    def test_absent_budget_leaves_the_attempt_count_in_charge(self, clock):
        target = AlwaysFails(RuntimeError(CAPACITY_MESSAGE))
        wrapper = _wrapper(target, max_retries=4)

        with pytest.raises(RuntimeError):
            wrapper.invoke()

        assert target.call_count == 4

    def test_empty_budget_leaves_the_attempt_count_in_charge(self, clock):
        target = AlwaysFails(RuntimeError(CAPACITY_MESSAGE))
        wrapper = _wrapper(target, max_retries=4, budget=RetryBudget())

        with pytest.raises(RuntimeError):
            wrapper.invoke()

        assert target.call_count == 4


class TestCapacityAttemptLimit:
    def test_capacity_refusal_stops_at_the_short_budget(self, clock):
        target = AlwaysFails(RuntimeError(CAPACITY_MESSAGE))
        wrapper = _wrapper(target, budget=RetryBudget(capacity_max_attempts=2))

        with pytest.raises(RuntimeError, match="on-demand capacity"):
            wrapper.invoke()

        assert target.call_count == 2

    def test_transient_fault_keeps_the_full_budget(self, clock):
        target = AlwaysFails(RuntimeError("Connection error."))
        wrapper = _wrapper(
            target, max_retries=5, budget=RetryBudget(capacity_max_attempts=2)
        )

        with pytest.raises(RuntimeError):
            wrapper.invoke()

        assert target.call_count == 5

    def test_a_limit_never_grants_more_attempts_than_max_retries(self, clock):
        """The budget may only lower the ceiling, never raise it."""
        target = AlwaysFails(RuntimeError(CAPACITY_MESSAGE))
        wrapper = _wrapper(
            target, max_retries=1, budget=RetryBudget(capacity_max_attempts=9)
        )

        with pytest.raises(RuntimeError):
            wrapper.invoke()

        assert target.call_count == 1

    def test_a_limit_never_overrides_a_refusing_predicate(self, clock):
        target = AlwaysFails(RuntimeError(CAPACITY_MESSAGE))
        wrapper = _wrapper(
            target,
            budget=RetryBudget(capacity_max_attempts=9),
            retry_on=lambda _: False,
        )

        with pytest.raises(RuntimeError):
            wrapper.invoke()

        assert target.call_count == 1

    @pytest.mark.asyncio
    async def test_capacity_refusal_stops_at_the_short_budget_async(self, clock):
        target = AlwaysFails(RuntimeError(CAPACITY_MESSAGE))
        wrapper = _wrapper(target, budget=RetryBudget(capacity_max_attempts=2))

        with pytest.raises(RuntimeError):
            await wrapper.ainvoke()

        assert target.call_count == 2


class TestWallClockDeadline:
    """Each attempt below burns 2 simulated seconds and each retry sleeps 1,
    so a 10s deadline admits four attempts out of a budget of ten. The
    expectations follow from those durations, not from how many times the
    implementation reads the clock."""

    def _slow_failure(self, clock, error) -> AlwaysFails:
        return AlwaysFails(error, clock=clock, duration=2.0)

    def test_deadline_stops_the_loop_before_the_attempts_run_out(self, clock):
        target = self._slow_failure(clock, RuntimeError("Connection error."))
        wrapper = _wrapper(
            target,
            max_retries=10,
            strategy=FixedDelay(delay_ms=1000),
            budget=RetryBudget(deadline_seconds=10.0),
        )

        with pytest.raises(RuntimeError):
            wrapper.invoke()

        assert target.call_count == 4
        assert clock.slept == [1.0, 1.0, 1.0]

    def test_deadline_applies_to_capacity_refusals_too(self, clock):
        target = self._slow_failure(clock, RuntimeError(CAPACITY_MESSAGE))
        wrapper = _wrapper(
            target,
            strategy=FixedDelay(delay_ms=1000),
            budget=RetryBudget(deadline_seconds=10.0, capacity_max_attempts=9),
        )

        with pytest.raises(RuntimeError):
            wrapper.invoke()

        assert target.call_count == 4

    def test_the_attempt_limit_wins_when_it_is_tighter(self, clock):
        target = self._slow_failure(clock, RuntimeError(CAPACITY_MESSAGE))
        wrapper = _wrapper(
            target,
            strategy=FixedDelay(delay_ms=1000),
            budget=RetryBudget(deadline_seconds=1000.0, capacity_max_attempts=2),
        )

        with pytest.raises(RuntimeError):
            wrapper.invoke()

        assert target.call_count == 2

    @pytest.mark.asyncio
    async def test_deadline_stops_the_async_loop(self, clock):
        """The async loop is the production path; it sleeps via asyncio."""
        target = self._slow_failure(clock, RuntimeError("Connection error."))
        wrapper = _wrapper(
            target,
            max_retries=10,
            strategy=FixedDelay(delay_ms=1000),
            budget=RetryBudget(deadline_seconds=10.0),
        )

        with pytest.raises(RuntimeError):
            await wrapper.ainvoke()

        assert target.call_count == 4
        assert clock.slept == [1.0, 1.0, 1.0]

    def test_the_raised_error_is_still_the_provider_failure(self, clock):
        target = self._slow_failure(clock, RuntimeError("Connection error."))
        wrapper = _wrapper(
            target,
            strategy=FixedDelay(delay_ms=1000),
            budget=RetryBudget(deadline_seconds=10.0),
        )

        with pytest.raises(RuntimeError, match="Connection error."):
            wrapper.invoke()


class TestProviderRetryHint:
    def test_a_retry_after_hint_replaces_our_backoff(self, clock):
        target = AlwaysFails(_retry_after_error("7"))
        wrapper = _wrapper(target, max_retries=2, budget=RetryBudget())

        with pytest.raises(RuntimeError):
            wrapper.invoke()

        assert clock.slept == [pytest.approx(7.0)]

    def test_our_backoff_stands_without_a_hint(self, clock):
        target = AlwaysFails(RuntimeError("Connection error."))
        wrapper = _wrapper(
            target,
            max_retries=2,
            strategy=FixedDelay(delay_ms=250),
            budget=RetryBudget(),
        )

        with pytest.raises(RuntimeError):
            wrapper.invoke()

        assert clock.slept == [pytest.approx(0.25)]

    def test_no_budget_means_no_hint_handling(self, clock):
        """Callers that pass no budget keep exactly their strategy's backoff."""
        target = AlwaysFails(_retry_after_error("7"))
        wrapper = _wrapper(target, max_retries=2, strategy=FixedDelay(delay_ms=250))

        with pytest.raises(RuntimeError):
            wrapper.invoke()

        assert clock.slept == [pytest.approx(0.25)]

    @pytest.mark.asyncio
    async def test_a_retry_after_hint_replaces_our_backoff_async(self, clock):
        """The async loop sleeps through ``asyncio.sleep``, not ``time``."""
        target = AlwaysFails(_retry_after_error("7"))
        wrapper = _wrapper(target, max_retries=2, budget=RetryBudget())

        with pytest.raises(RuntimeError):
            await wrapper.ainvoke()

        assert clock.slept == [pytest.approx(7.0)]

    def test_a_hint_is_not_slept_through_past_the_deadline(self, clock):
        target = AlwaysFails(_retry_after_error("30"), clock=clock, duration=2.0)
        wrapper = _wrapper(target, budget=RetryBudget(deadline_seconds=10.0))

        with pytest.raises(RuntimeError):
            wrapper.invoke()

        # 2s spent plus a 30s hint overshoots the 10s deadline, so we give up
        # rather than hold the slot idle for a retry we would not be allowed.
        assert clock.slept == []
        assert target.call_count == 1


class TestGiveUpLogging:
    def test_capacity_exhaustion_is_logged_as_its_own_reason(self, clock, caplog):
        target = AlwaysFails(RuntimeError(CAPACITY_MESSAGE))
        wrapper = _wrapper(target, budget=RetryBudget(capacity_max_attempts=2))

        with caplog.at_level(logging.WARNING, logger="xagent.core.retry.wrapper"):
            with pytest.raises(RuntimeError):
                wrapper.invoke()

        assert any(
            "llm_capacity_refusal" in record.getMessage() for record in caplog.records
        )

    def test_deadline_exhaustion_is_logged_as_its_own_reason(self, clock, caplog):
        target = AlwaysFails(
            RuntimeError("Connection error."), clock=clock, duration=20.0
        )
        wrapper = _wrapper(target, budget=RetryBudget(deadline_seconds=10.0))

        with caplog.at_level(logging.WARNING, logger="xagent.core.retry.wrapper"):
            with pytest.raises(RuntimeError):
                wrapper.invoke()

        assert any(
            "retry_deadline_exceeded" in record.getMessage()
            for record in caplog.records
        )


class TestFactoryPassthrough:
    def test_create_retry_wrapper_forwards_the_budget(self, clock):
        class Base:
            def run(self) -> str:
                raise NotImplementedError

        class Inner(Base):
            def __init__(self) -> None:
                self.call_count = 0

            def run(self) -> str:
                self.call_count += 1
                raise RuntimeError(CAPACITY_MESSAGE)

        inner = Inner()
        wrapped = create_retry_wrapper(
            inner,
            Base,
            retry_methods={"run"},
            strategy=FixedDelay(delay_ms=1),
            max_retries=10,
            budget=RetryBudget(capacity_max_attempts=2),
        )

        with pytest.raises(RuntimeError):
            wrapped.run()

        assert inner.call_count == 2


class TestDeadlineIsRecheckedAtAttemptStart:
    """A late wake must not buy a whole extra provider request.

    ``_plan_retry`` clears a retry against the *nominal* delay. If the sleep
    overshoots -- a busy scheduler, a blocked event loop -- the loop would
    otherwise start another attempt past the deadline, and that attempt can
    then consume a full per-attempt timeout. Distinct from the documented
    overrun of an attempt already in flight when the deadline passes.
    """

    def test_a_late_wake_does_not_start_another_attempt(self, late_clock):
        target = AlwaysFails(RuntimeError("Connection error."))
        wrapper = _wrapper(
            target,
            max_retries=10,
            strategy=FixedDelay(delay_ms=1000),
            budget=RetryBudget(deadline_seconds=10.0),
        )

        with pytest.raises(RuntimeError, match="Connection error."):
            wrapper.invoke()

        # The nominal 1s delay cleared the 10s deadline, but waking 21s later
        # puts attempt 2 past it.
        assert late_clock.slept == [1.0]
        assert target.call_count == 1

    @pytest.mark.asyncio
    async def test_a_late_wake_does_not_start_another_async_attempt(self, late_clock):
        target = AlwaysFails(RuntimeError("Connection error."))
        wrapper = _wrapper(
            target,
            max_retries=10,
            strategy=FixedDelay(delay_ms=1000),
            budget=RetryBudget(deadline_seconds=10.0),
        )

        with pytest.raises(RuntimeError):
            await wrapper.ainvoke()

        assert target.call_count == 1

    def test_a_punctual_wake_still_retries(self, clock):
        """The recheck must not cost a retry that fits inside the deadline."""
        target = AlwaysFails(RuntimeError("Connection error."))
        wrapper = _wrapper(
            target,
            max_retries=3,
            strategy=FixedDelay(delay_ms=1000),
            budget=RetryBudget(deadline_seconds=100.0),
        )

        with pytest.raises(RuntimeError):
            wrapper.invoke()

        assert target.call_count == 3

    def test_no_budget_means_no_recheck(self, late_clock):
        """Callers that pass no budget keep their exact attempt count."""
        target = AlwaysFails(RuntimeError("Connection error."))
        wrapper = _wrapper(target, max_retries=3, strategy=FixedDelay(delay_ms=1))

        with pytest.raises(RuntimeError):
            wrapper.invoke()

        assert target.call_count == 3
