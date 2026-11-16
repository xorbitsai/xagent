"""``close_for_restart`` spreads 1012 closes over a bounded window."""

import asyncio
import time

import pytest

from xagent.web.api import websocket as ws_module
from xagent.web.api.websocket import ConnectionManager, _command_origins

ENV = "XAGENT_WS_RESTART_CLOSE_WINDOW_SECONDS"


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


@pytest.fixture(autouse=True)
def _default_env(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)


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
    return socks, t0


@pytest.mark.asyncio
async def test_closes_spread_across_the_window(manager):
    socks, t0 = _register(manager, 40)
    handled = await manager.close_for_restart(close_window_seconds=1.0)
    ended = time.monotonic() - t0

    offsets = [ws.closed_at for ws in socks]
    assert handled == 40
    assert all(ws.codes == [(1012, "Service Restart")] for ws in socks)
    # 1.0 s window = 4 slots of 0.25 s: closes land in several distinct slots.
    assert len({int(o / 0.25) for o in offsets}) >= 3
    assert max(offsets) <= 1.0 + ws_module.RESTART_CLOSE_WINDOW_GRACE_SECONDS
    assert ended >= max(offsets)  # returns only after the last close
    assert manager.connection_count() == 0


@pytest.mark.asyncio
async def test_uneven_batches_still_fill_the_whole_window(manager):
    # 9 sockets, 4 slots: the old ceil(n/size) split left the last slot empty.
    socks, _ = _register(manager, 9)
    await manager.close_for_restart(close_window_seconds=1.0)

    assert max(ws.closed_at for ws in socks) >= 0.75 - 0.02


@pytest.mark.asyncio
async def test_window_zero_and_default_close_at_once(manager):
    socks, t0 = _register(manager, 20)
    await manager.close_for_restart(close_window_seconds=0)
    assert time.monotonic() - t0 < 0.3
    assert all(ws.codes for ws in socks)

    manager2 = ConnectionManager()
    socks2, t1 = _register(manager2, 5)
    await manager2.close_for_restart()  # env unset: default is immediate
    assert time.monotonic() - t1 < 0.3
    assert all(ws.codes for ws in socks2)


@pytest.mark.asyncio
async def test_env_window_is_used_when_no_argument(manager, monkeypatch):
    monkeypatch.setenv(ENV, "0.6")
    socks, _ = _register(manager, 12)
    await manager.close_for_restart()
    assert max(ws.closed_at for ws in socks) > 0.3


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [float("inf"), float("nan"), -5.0, 1e400])
async def test_invalid_window_argument_falls_back_and_still_closes(manager, bad):
    socks, _ = _register(manager, 4)
    assert await manager.close_for_restart(close_window_seconds=bad) == 4
    assert all(ws.codes for ws in socks)


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
    socks, _ = _register(manager, 10)
    closing = asyncio.create_task(manager.close_for_restart(close_window_seconds=30))
    await asyncio.sleep(0.1)
    assert sum(1 for ws in socks if ws.codes) < 10
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing

    assert all(ws.codes == [(1012, "Service Restart")] for ws in socks)
    assert manager.connection_count() == 0


@pytest.mark.asyncio
async def test_abort_during_window_leaves_the_rest_open_and_counts_closed(manager):
    socks, _ = _register(manager, 10)
    closing = asyncio.create_task(manager.close_for_restart(close_window_seconds=30))
    await asyncio.sleep(0.1)
    manager.end_restart_drain()
    handled = await asyncio.wait_for(closing, 3)
    await asyncio.sleep(0.05)

    closed = sum(1 for ws in socks if ws.codes)
    assert closed < 10
    assert manager.connection_count() == 10 - closed
    assert handled == closed


@pytest.mark.asyncio
async def test_command_accepted_mid_window_gets_its_reply_before_1012(manager):
    socks, _ = _register(manager, 4)
    busy = socks[0]
    released = asyncio.Event()
    release_at: list[float] = []

    async def command():
        async with manager.ingress(busy):
            await released.wait()
            release_at.append(time.monotonic())

    closing = asyncio.create_task(
        manager.close_for_restart(close_window_seconds=1.0, reply_deadline_seconds=5)
    )
    await asyncio.sleep(0.05)  # inside the window, before any slot fires
    handler = asyncio.create_task(command())
    await asyncio.sleep(1.3)  # window over: the others are closed
    assert all(ws.codes for ws in socks[1:])
    assert busy.codes == []  # held for its reply
    released.set()
    await handler
    await asyncio.wait_for(closing, 3)

    assert busy.codes == [(1012, "Service Restart")]
    assert busy.t0 + busy.closed_at >= release_at[0]


@pytest.mark.asyncio
async def test_slow_close_is_bounded_by_the_hard_stop(manager):
    class Hang(TimedSocket):
        async def close(self, code=1000, reason=None):
            await asyncio.Event().wait()

    t0 = time.monotonic()
    hung = Hang(t0)
    ok = TimedSocket(t0)
    manager.register_connection(hung, 1)
    manager.register_connection(ok, 2)
    started = time.monotonic()
    await asyncio.wait_for(
        manager.close_for_restart(close_window_seconds=0.5, close_timeout_seconds=0.3),
        timeout=5,
    )

    assert ok.codes
    assert time.monotonic() - started < 0.5 + 2.0 + 0.5
    assert manager.connection_count() == 0


@pytest.mark.asyncio
async def test_hard_stop_waits_out_a_slow_close_instead_of_reissuing_it(manager):
    calls: list[int] = []

    class Slow(TimedSocket):
        async def close(self, code=1000, reason=None):
            calls.append(1)
            await asyncio.sleep(2.5)
            await super().close(code, reason)

    slow = Slow(time.monotonic())
    manager.register_connection(slow, 1)
    await manager.close_for_restart(
        close_window_seconds=0.2,
        close_timeout_seconds=3.0,
        reply_deadline_seconds=0,
    )

    # grace (2 s) alone would cancel the close at 2.2 s and issue it again.
    assert len(calls) == 1
    assert slow.codes == [(1012, "Service Restart")]
