import logging
import random
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class RetryStrategy(Protocol):
    def get_delay(self, attempt: int) -> int:
        """Returns sleep time in milliseconds for the given attempt index (0-based)."""
        ...


class LinearBackoff(RetryStrategy):
    def __init__(self, base_delay_ms: int = 500):
        self.base_delay_ms = base_delay_ms

    def get_delay(self, attempt: int) -> int:
        return self.base_delay_ms * (attempt + 1)


class ExponentialBackoff(RetryStrategy):
    def __init__(
        self,
        base_delay_ms: int = 200,
        multiplier: float = 2.0,
        max_delay_ms: float = 60000.0,
    ):
        self.base_delay_ms = base_delay_ms
        self.multiplier = multiplier
        self.max_delay_ms = max_delay_ms

    def get_delay(self, attempt: int) -> int:
        delay = self.base_delay_ms * (self.multiplier**attempt)
        # Add jitter to prevent thundering herd
        jitter = random.uniform(0, 0.1 * delay)
        return int(min(delay + jitter, self.max_delay_ms))


class FixedDelay(RetryStrategy):
    def __init__(self, delay_ms: int = 1000):
        self.delay_ms = delay_ms

    def get_delay(self, attempt: int) -> int:
        return self.delay_ms
