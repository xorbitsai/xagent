"""Regression tests for issue #889: a stalled MCP server must not stall
agent setup (or pin resources) indefinitely."""

import asyncio
import time

import pytest

from xagent.config import MCP_TOOL_INIT_TIMEOUT_SECONDS
from xagent.core.tools.adapters.vibe import mcp_adapter as mcp_adapter_module
from xagent.core.tools.adapters.vibe.mcp_adapter import (
    _load_server_tools_bounded,
    load_mcp_tools_as_agent_tools,
)


@pytest.mark.asyncio
async def test_stalled_server_times_out_and_other_servers_still_load(monkeypatch):
    """A server whose initialize/list-tools stalls is skipped at the timeout;
    the remaining servers still load."""
    monkeypatch.setenv(MCP_TOOL_INIT_TIMEOUT_SECONDS, "1")

    healthy_tool = object()

    async def fake_load_direct(server_name, connection, **kwargs):
        if server_name == "stalled":
            await asyncio.Event().wait()  # never completes
        return [healthy_tool]

    monkeypatch.setattr(mcp_adapter_module, "_load_direct_mcp_tools", fake_load_direct)

    started = time.monotonic()
    tools = await load_mcp_tools_as_agent_tools(
        {
            "stalled": {"transport": "streamable_http", "url": "http://x"},
            "healthy": {"transport": "streamable_http", "url": "http://y"},
        }
    )
    elapsed = time.monotonic() - started

    assert tools == [healthy_tool]
    # 1s timeout for the stalled server plus fast healthy load; the old
    # behavior blocked forever.
    assert elapsed < 5


@pytest.mark.asyncio
async def test_bounded_load_returns_even_when_cleanup_hangs():
    """The bound must hold even if the load task ignores cancellation (e.g. a
    hung streamable-HTTP session blocking in __aexit__)."""

    cleanup_entered = asyncio.Event()
    # Set by the test AFTER the bounded call returns, so a cleanup exit
    # before it proves the caller didn't wait. Also lets the abandoned task
    # finish so pytest-asyncio's loop teardown doesn't hang on it.
    release_cleanup = asyncio.Event()

    async def uncancellable_load():
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_entered.set()
            # Simulate hung cleanup: swallow cancellation until released.
            while not release_cleanup.is_set():
                try:
                    await asyncio.sleep(0.05)
                except asyncio.CancelledError:
                    continue
        return []  # pragma: no cover

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        await _load_server_tools_bounded("hung", uncancellable_load(), 1)
    elapsed = time.monotonic() - started

    assert elapsed < 5
    # The load was cancelled (cleanup began) but the caller did not wait on
    # it: the bounded call returned while cleanup was still blocked.
    await asyncio.wait_for(cleanup_entered.wait(), timeout=5)
    release_cleanup.set()
    await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_burst_larger_than_gate_does_not_fan_out(monkeypatch):
    """A burst of concurrent loads for the same hung server must not create
    more underlying load tasks (transports/sockets) than the per-server cap:
    callers beyond the cap fail fast at the gate without starting a load."""
    monkeypatch.setattr(mcp_adapter_module, "_MAX_INFLIGHT_LOADS_PER_SERVER", 2)

    started_loads = 0
    release_cleanup = asyncio.Event()

    async def uncancellable_load():
        nonlocal started_loads
        started_loads += 1
        try:
            await asyncio.Event().wait()
        finally:
            while not release_cleanup.is_set():
                try:
                    await asyncio.sleep(0.05)
                except asyncio.CancelledError:
                    continue
        return []  # pragma: no cover

    async def one_caller():
        with pytest.raises(TimeoutError):
            await _load_server_tools_bounded("burst-server", uncancellable_load(), 1)

    began = time.monotonic()
    await asyncio.gather(*(one_caller() for _ in range(6)))
    elapsed = time.monotonic() - began

    # Every caller returned within its own bound...
    assert elapsed < 5
    # ...but only cap-many loads (transports) ever started; the abandoned
    # ones keep holding their slots so the other four callers failed fast
    # at the gate.
    assert started_loads == 2

    # A follow-up caller while both slots are still held by abandoned loads
    # also fails fast without starting a load.
    with pytest.raises(TimeoutError):
        await _load_server_tools_bounded("burst-server", uncancellable_load(), 1)
    assert started_loads == 2

    # Let the abandoned tasks finish so loop teardown doesn't hang.
    release_cleanup.set()
    await asyncio.sleep(0.2)


@pytest.mark.asyncio
async def test_caller_cancellation_cancels_child_and_frees_slot(monkeypatch):
    """Cancelling the caller must propagate to the owned load task —
    asyncio.wait doesn't do it — or cancelled requests would strand live
    loads that hold gate slots and transports forever."""
    monkeypatch.setattr(mcp_adapter_module, "_MAX_INFLIGHT_LOADS_PER_SERVER", 1)

    load_started = asyncio.Event()
    child_cancelled = asyncio.Event()

    async def hung_but_cancellable_load():
        load_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            child_cancelled.set()
            raise
        return []  # pragma: no cover

    caller = asyncio.create_task(
        _load_server_tools_bounded("cancel-server", hung_but_cancellable_load(), 30)
    )
    await asyncio.wait_for(load_started.wait(), timeout=5)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller

    # The child observed the cancellation (it would previously run forever)...
    await asyncio.wait_for(child_cancelled.wait(), timeout=5)

    # ...and released its slot: with a cap of 1, a fresh load on the same
    # server can start and complete.
    async def quick_load():
        return ["tool"]

    assert await _load_server_tools_bounded("cancel-server", quick_load(), 5) == [
        "tool"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_seconds", [0, 30])
async def test_cancel_while_queued_at_gate_closes_unstarted_coro(
    monkeypatch, timeout_seconds
):
    """Cancelling a caller that is still waiting for a gate slot must close
    the never-started load coroutine (no 'was never awaited' at GC), in both
    the timed and the timeout-disabled branches."""
    import inspect as inspect_mod

    monkeypatch.setattr(mcp_adapter_module, "_MAX_INFLIGHT_LOADS_PER_SERVER", 1)
    server = f"queued-cancel-{timeout_seconds}"

    release_holder = asyncio.Event()

    async def slot_holder_load():
        await release_holder.wait()
        return []

    holder = asyncio.create_task(
        _load_server_tools_bounded(server, slot_holder_load(), 30)
    )
    await asyncio.sleep(0.05)  # holder occupies the single slot

    queued_coro = slot_holder_load()
    queued = asyncio.create_task(
        _load_server_tools_bounded(server, queued_coro, timeout_seconds)
    )
    await asyncio.sleep(0.05)  # queued caller is waiting at the gate
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued

    assert inspect_mod.getcoroutinestate(queued_coro) == "CORO_CLOSED"

    release_holder.set()
    assert await holder == []


@pytest.mark.asyncio
async def test_bounded_load_disabled_with_zero_timeout():
    async def quick_load():
        return ["tool"]

    assert await _load_server_tools_bounded("s", quick_load(), 0) == ["tool"]


@pytest.mark.asyncio
async def test_bounded_load_passes_result_through():
    async def quick_load():
        return ["tool"]

    assert await _load_server_tools_bounded("s", quick_load(), 30) == ["tool"]


@pytest.mark.asyncio
async def test_create_mcp_tools_releases_db_before_network_init(monkeypatch):
    """The tool config's DB connection is released before the MCP network
    phase begins, so the handshake never runs inside an open transaction."""
    from xagent.core.tools.adapters.vibe.factory import ToolFactory
    from xagent.core.tools.adapters.vibe.mcp_tools import create_mcp_tools

    calls: list[str] = []

    class FakeConfig:
        def get_tool_selection_spec(self):
            return None

        async def get_mcp_server_configs(self):
            calls.append("load_configs")
            return [
                {
                    "name": "srv",
                    "transport": "streamable_http",
                    "config": {"url": "http://x"},
                }
            ]

        def release_db_connection(self):
            calls.append("release_db")

        def get_sandbox(self):
            return None

    async def fake_create(mcp_configs, sandbox=None):
        calls.append("network_init")
        return []

    monkeypatch.setattr(
        ToolFactory,
        "_create_mcp_tools_from_configs",
        staticmethod(fake_create),
    )

    await create_mcp_tools(FakeConfig())

    assert calls == ["load_configs", "release_db", "network_init"]
