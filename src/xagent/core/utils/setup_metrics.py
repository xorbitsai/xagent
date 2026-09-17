"""Bounded in-memory setup counters; the SaaS monitor exports window summaries."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field


@dataclass
class SetupMetrics:
    active: int = 0
    peak: int = 0
    started: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    slow: int = 0
    max_ms: float = 0.0
    # MCP setup can also run in a helper thread's event loop.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def start(self) -> float:
        with self._lock:
            self.active += 1
            self.started += 1
            self.peak = max(self.peak, self.active)
        return time.monotonic()

    def finish(
        self, started: float, *, failed: bool = False, cancelled: bool = False
    ) -> None:
        elapsed = time.monotonic() - started
        with self._lock:
            self.active -= 1
            self.completed += 1
            self.failed += failed
            self.cancelled += cancelled
            self.slow += elapsed >= 1.0
            self.max_ms = max(self.max_ms, elapsed * 1000)

    @asynccontextmanager
    async def measure(self) -> AsyncIterator[None]:
        started = self.start()
        failed = cancelled = False
        try:
            yield
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception:
            failed = True
            raise
        finally:
            self.finish(started, failed=failed, cancelled=cancelled)

    def drain(self) -> dict[str, int | float]:
        with self._lock:
            result = {
                "active": self.active,
                "peak": self.peak,
                "started": self.started,
                "completed": self.completed,
                "failed": self.failed,
                "cancelled": self.cancelled,
                "slow": self.slow,
                "max_ms": round(self.max_ms, 1),
            }
            self.peak = self.active
            self.started = self.completed = self.failed = self.cancelled = self.slow = 0
            self.max_ms = 0.0
            return result


agent_setup = SetupMetrics()
mcp_setup = SetupMetrics()
trigger_execution = SetupMetrics()
