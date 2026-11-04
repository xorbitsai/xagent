import pytest

from xagent.core.retry.strategy import ExponentialBackoff, FixedDelay, LinearBackoff
from xagent.core.retry.wrapper import Retryable, RetryWrapper


class MockInvokable:
    def __init__(self, fail_times=0):
        self.call_count = 0
        self.fail_times = fail_times

    def invoke(self, *args, **kwargs):
        self.call_count += 1
        if self.call_count <= self.fail_times:
            raise ValueError("Mock failure")
        return "success"

    async def ainvoke(self, *args, **kwargs):
        self.call_count += 1
        if self.call_count <= self.fail_times:
            raise ValueError("Mock failure")
        return "success"


def test_success_no_retry():
    target = MockInvokable(fail_times=0)
    wrapper = RetryWrapper(target, max_retries=3)

    result = wrapper.invoke()

    assert result == "success"
    assert target.call_count == 1


def test_success_after_retries():
    target = MockInvokable(fail_times=2)
    wrapper = RetryWrapper(target, max_retries=3)

    result = wrapper.invoke()

    assert result == "success"
    assert target.call_count == 3


def test_max_retries_exceeded():
    target = MockInvokable(fail_times=5)
    wrapper = RetryWrapper(target, max_retries=3)

    with pytest.raises(ValueError, match="Mock failure"):
        wrapper.invoke()

    assert target.call_count == 3


def test_retry_on_specific_error_types():
    class RetryableError(Exception):
        pass

    class NonRetryableError(Exception):
        pass

    class SelectiveFailure(Retryable):
        def __init__(self):
            self.call_count = 0

        def invoke(self, *args, **kwargs):
            self.call_count += 1
            if self.call_count == 1:
                raise RetryableError("Should retry")
            if self.call_count == 2:
                raise NonRetryableError("Should not retry")
            return "success"

    target = SelectiveFailure()
    wrapper = RetryWrapper(
        target, max_retries=5, retry_on=lambda e: isinstance(e, RetryableError)
    )

    with pytest.raises(NonRetryableError, match="Should not retry"):
        wrapper.invoke()

    assert target.call_count == 2  # First call + one retry, then fails


@pytest.mark.asyncio
async def test_async_success_after_retries():
    target = MockInvokable(fail_times=2)
    wrapper = RetryWrapper(target, max_retries=3)

    result = await wrapper.ainvoke()

    assert result == "success"
    assert target.call_count == 3


def test_linear_backoff():
    strategy = LinearBackoff(base_delay_ms=100)

    assert strategy.get_delay(0) == 100
    assert strategy.get_delay(1) == 200
    assert strategy.get_delay(2) == 300


def test_exponential_backoff():
    strategy = ExponentialBackoff(base_delay_ms=100, multiplier=2.0, max_delay_ms=1000)

    delay_0 = strategy.get_delay(0)
    delay_1 = strategy.get_delay(1)

    assert 100 <= delay_0 <= 120  # base + jitter
    assert 200 <= delay_1 <= 240  # base*2 + jitter
    assert strategy.get_delay(10) <= 1000  # max cap


def test_fixed_delay():
    strategy = FixedDelay(delay_ms=500)

    assert strategy.get_delay(0) == 500
    assert strategy.get_delay(5) == 500


def test_callable_interface():
    target = MockInvokable(fail_times=0)
    wrapper = RetryWrapper(target)

    result = wrapper()

    assert result == "success"
