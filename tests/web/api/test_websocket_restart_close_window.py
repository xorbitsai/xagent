"""``close_for_restart`` spreads 1012 closes over a bounded window."""

import asyncio
import time

import pytest

from xagent.web.api import websocket as ws_module
from xagent.web.api.websocket import ConnectionManager, _command_origins


class TimedSocket:
    def __init__(self, t0: float, *, fail: bool = False) -> None:
        self.t0 = t0
        self.fail = fail
        self.closed_at: float | None = None
        self.codes: list[tuple[int, str | None]] = []

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        if self.fail:
            raise RuntimeError("peer gone")
        self.closed_at = time.monotonic() - self.t0
        self.codes.append((code, reason))


@pytest.fixture
def manager():
    mgr = ConnectionManager()
    yield mgr
    _command_origins._origins.clear()


def _register(manager, n, **kw):
    t0 = time.monotonic()
    socks = [TimedSocket(t0, **kw) for _ in range(n)]
    for i, ws in enumerate(socks):
        manager.register_connection(ws, i + 1)
    return socks


@pytest.mark.asyncio
async def test_closes_spread_across_the_window(manager):
    socks = _register(manager, 40)
    t0 = time.monotonic()
    handled = await manager.close_for_restart(close_window_seconds=1.0)
    elapsed = time.monotonic() - t0

    assert handled == 40
    times = sorted(ws.closed_at for ws in socks)
    offsets = [t - (t0 - socks[0].t0) for t in times]
    assert all(ws.codes == [(1012, "Service Restart")] for ws in socks)
    assert offsets[-1] - offsets[0] > 0.5  # spread, not one instant
    assert max(offsets) <= 1.0 + ws_module.RESTART_CLOSE_WINDOW_GRACE_SECONDS
    assert elapsed >= max(offsets) - 0.05  # returns after the last close
    assert manager.connection_count() == 0


@pytest.mark.asyncio
async def test_window_zero_closes_at_once(manager):
    socks = _register(manager, 20)
    t0 = time.monotonic()
    await manager.close_for_restart(close_window_seconds=0)
    assert time.monotonic() - t0 < 0.3
    assert all(ws.codes for ws in socks)


@pytest.mark.asyncio
async def test_env_window_is_used_and_invalid_falls_back(manager, monkeypatch):
    monkeypatch.setenv(ws_module.RESTART_CLOSE_WINDOW_ENV, "0")
    assert ws_module.restart_close_window_seconds() == 0
    monkeypatch.setenv(ws_module.RESTART_CLOSE_WINDOW_ENV, "abc")
    assert (
        ws_module.restart_close_window_seconds()
        == ws_module.RESTART_CLOSE_WINDOW_DEFAULT_SECONDS
    )
    monkeypatch.setenv(ws_module.RESTART_CLOSE_WINDOW_ENV, "-3")
    assert (
        ws_module.restart_close_window_seconds()
        == ws_module.RESTART_CLOSE_WINDOW_DEFAULT_SECONDS
    )
    monkeypatch.delenv(ws_module.RESTART_CLOSE_WINDOW_ENV)
    assert (
        ws_module.restart_close_window_seconds()
        == ws_module.RESTART_CLOSE_WINDOW_DEFAULT_SECONDS
    )


@pytest.mark.asyncio
async def test_erroring_socket_does_not_stop_the_others(manager):
    t0 = time.monotonic()
    bad = TimedSocket(t0, fail=True)
    good = [TimedSocket(t0) for _ in range(5)]
    for i, ws in enumerate([bad, *good]):
        manager.register_connection(ws, i + 1)

    handled = await manager.close_for_restart(close_window_seconds=0.5)

    assert handled == 6
    assert all(ws.codes for ws in good)
    assert manager.connection_count() == 0


@pytest.mark.asyncio
async def test_cancel_mid_window_closes_the_rest_at_once(manager):
    socks = _register(manager, 10)
    closing = asyncio.create_task(manager.close_for_restart(close_window_seconds=30))
    await asyncio.sleep(0.1)
    assert sum(1 for ws in socks if ws.codes) < 10
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing

    assert all(ws.codes == [(1012, "Service Restart")] for ws in socks)
    assert manager.connection_count() == 0


@pytest.mark.asyncio
async def test_abort_during_window_leaves_the_rest_open(manager):
    socks = _register(manager, 10)
    closing = asyncio.create_task(manager.close_for_restart(close_window_seconds=30))
    await asyncio.sleep(0.1)
    manager.end_restart_drain()
    await asyncio.wait_for(closing, 3)
    await asyncio.sleep(0.05)
    assert sum(1 for ws in socks if ws.codes) < 10


@pytest.mark.asyncio
async def test_call_returns_only_after_the_last_close(manager):
    socks = _register(manager, 6)
    await manager.close_for_restart(close_window_seconds=0.6)
    assert all(ws.closed_at is not None for ws in socks)
    assert manager.connection_count() == 0
