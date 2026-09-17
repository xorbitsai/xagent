import asyncio
from types import SimpleNamespace

import pytest

from xagent.core.utils import setup_metrics
from xagent.core.utils.setup_metrics import SetupMetrics


async def test_overlapping_setup_windows_preserve_active_and_failure(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(
        setup_metrics, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    metrics = SetupMetrics()
    with pytest.raises(ValueError):
        async with metrics.measure():
            assert metrics.drain()["active"] == 1
            async with metrics.measure():
                clock[0] = 2.0
            raise ValueError("secret")
    summary = metrics.drain()
    assert summary == {
        "active": 0,
        "peak": 2,
        "started": 1,
        "completed": 2,
        "failed": 1,
        "cancelled": 0,
        "slow": 2,
        "max_ms": 2000.0,
    }
    assert metrics.drain()["failed"] == 0


async def test_decorated_setup_preserves_return_and_cancellation():
    metrics = SetupMetrics()

    @metrics.measure()
    async def run(cancel):
        if cancel:
            raise asyncio.CancelledError
        return 42

    assert await run(False) == 42
    with pytest.raises(asyncio.CancelledError):
        await run(True)
    result = metrics.drain()
    assert result["active"] == 0
    assert result["completed"] == 2
    assert result["cancelled"] == 1
    assert result["failed"] == 0
