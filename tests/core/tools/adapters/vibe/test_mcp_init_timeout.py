"""Regression tests for issue #889: a stalled MCP server must not stall
agent setup (or pin resources) indefinitely.

Also covers the bounded reclaim of abandoned MCP initializations: once a load
is abandoned (timed out, or its caller cancelled), its recorder is sealed so
the retry loop cannot start another attempt, and its HTTP transports (sse,
streamable_http) are force-closed after a grace period if it has not unwound
on its own.
"""

import asyncio
import contextlib
import inspect
import logging
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest

from xagent.config import (
    MCP_TOOL_INIT_TIMEOUT_SECONDS,
    get_mcp_tool_init_timeout_seconds,
)
from xagent.core.tools.adapters.vibe import mcp_adapter as mcp_adapter_module
from xagent.core.tools.adapters.vibe.config import MCPFailurePolicy
from xagent.core.tools.adapters.vibe.mcp_adapter import (
    MCPFailurePhase,
    MCPLoadResult,
    _load_direct_mcp_tools,
    _load_server_tools_bounded,
    _TransportReclaimedError,
    _TransportRecorder,
    load_mcp_tools_as_agent_tools,
)

_LOGGER_NAME = mcp_adapter_module.__name__


class _FakeHttpxClient:
    """Stand-in for httpx.AsyncClient: does no real I/O. Tracks how many
    times aclose() was called and exposes an asyncio.Event a caller can wait
    on to notice when that first happens."""

    def __init__(self) -> None:
        self.aclose_calls = 0
        self.closed = asyncio.Event()

    async def aclose(self) -> None:
        self.aclose_calls += 1
        self.closed.set()


def _counting_http_connection(transport: str) -> tuple[dict, dict]:
    """An sse/streamable_http connection whose httpx client factory builds
    ``_FakeHttpxClient`` instances and counts how many it has built."""
    counts = {"built": 0}

    def factory(headers=None, timeout=None, auth=None) -> _FakeHttpxClient:
        counts["built"] += 1
        return _FakeHttpxClient()

    connection = {
        "transport": transport,
        "url": "http://x",
        "httpx_client_factory": factory,
    }
    return connection, counts


def _uncancel_current_task() -> None:
    task = asyncio.current_task()
    if task is not None:
        while task.cancelling():
            task.uncancel()


async def _swallow_cancels_for(seconds: float) -> None:
    """Swallow every cancellation received while waiting out ``seconds``."""
    remaining = seconds
    loop = asyncio.get_event_loop()
    while remaining > 0:
        start = loop.time()
        try:
            await asyncio.sleep(remaining)
            return
        except asyncio.CancelledError:
            _uncancel_current_task()
            remaining -= loop.time() - start


async def _swallow_cancels_until(
    event: asyncio.Event, give_up_after: float = 5.0
) -> None:
    """Swallow every cancellation received while waiting for ``event``, but
    only for ``give_up_after`` seconds: if the code under test never sets the
    event, the test must fail rather than leave a task no cancel can end."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + give_up_after
    while not event.is_set():
        remaining = deadline - loop.time()
        if remaining <= 0:
            return
        try:
            await asyncio.wait_for(event.wait(), min(0.05, remaining))
        except asyncio.CancelledError:
            _uncancel_current_task()
        except TimeoutError:
            continue


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_stalled_server_times_out_and_other_servers_still_load(monkeypatch):
    """A server whose initialize/list-tools stalls is skipped at the timeout;
    the remaining servers still load."""
    monkeypatch.setenv(MCP_TOOL_INIT_TIMEOUT_SECONDS, "1")

    healthy_tool = object()

    async def fake_load_direct(server_name, connection, **kwargs):
        if server_name == "stalled":
            await asyncio.Event().wait()  # never completes
        return MCPLoadResult(
            tools=(healthy_tool,),
            loaded_servers=(server_name,),
            failures=(),
        )

    monkeypatch.setattr(mcp_adapter_module, "_load_direct_mcp_tools", fake_load_direct)

    result = await load_mcp_tools_as_agent_tools(
        {
            "stalled": {"transport": "streamable_http", "url": "http://x"},
            "healthy": {"transport": "streamable_http", "url": "http://y"},
        }
    )

    assert len(result.tools) == 1
    assert result.tools[0].target is healthy_tool
    assert result.loaded_servers == ("healthy",)
    assert len(result.failures) == 1
    assert result.failures[0].server_name == "stalled"
    assert result.failures[0].phase is MCPFailurePhase.INITIALIZE
    assert result.failures[0].error_type == "TimeoutError"


@pytest.mark.asyncio
@pytest.mark.timeout(30)
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

    with pytest.raises(TimeoutError):
        await _load_server_tools_bounded("hung", uncancellable_load(), 1)

    # The load was cancelled (cleanup began) but the caller did not wait on
    # it: the bounded call returned while cleanup was still blocked.
    await asyncio.wait_for(cleanup_entered.wait(), timeout=5)
    release_cleanup.set()
    await asyncio.sleep(0.1)


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_concurrent_loads_for_one_server_are_not_capped():
    """Concurrent callers for the same server each start their own load and
    are bounded only by their own timeout; none waits on another."""
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

    await asyncio.gather(*(one_caller() for _ in range(6)))

    assert started_loads == 6

    # Let the abandoned tasks finish so loop teardown doesn't hang.
    release_cleanup.set()
    await asyncio.sleep(0.2)


@pytest.mark.asyncio
async def test_caller_cancellation_cancels_child():
    """Cancelling the caller must propagate to the owned load task --
    asyncio.wait doesn't do it -- or cancelled requests would strand live
    loads that hold transports forever."""
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

    # The child observed the cancellation (it would previously run forever).
    await asyncio.wait_for(child_cancelled.wait(), timeout=5)


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

        def get_mcp_failure_policy(self):
            return MCPFailurePolicy.BEST_EFFORT

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


# ---------------------------------------------------------------------------
# Bounded reclaim of abandoned MCP initializations.
# ---------------------------------------------------------------------------


def _reclaim_warnings(records) -> list[str]:
    return [
        r.getMessage()
        for r in records
        if r.name == _LOGGER_NAME
        and r.levelno == logging.WARNING
        and "did not unwind within" in r.getMessage()
    ]


def _still_alive_warnings(records) -> list[str]:
    return [
        r.getMessage()
        for r in records
        if r.name == _LOGGER_NAME
        and r.levelno == logging.WARNING
        and "still alive" in r.getMessage()
    ]


def _make_stalling_create_session(*, k: int, ending: str, has_factory: bool):
    """Build a ``create_session`` stand-in for a load whose k-th attempt (1
    based) is the one that gets abandoned.

    Attempts before k build a client when the transport has one (the same as
    a real HTTP transport would, right before failing) and then fail at once
    with ``OSError``, like a refused connection. Attempt k blocks until the
    caller abandons it, then swallows every cancellation it receives: the
    ``by_itself`` ending raises a plain exception 0.2s later (cleanup that
    ends on its own without honouring the cancel); the ``by_force_close``
    ending (HTTP transports only, since only they have a client) stays
    blocked until its own client's ``aclose()`` runs. Any attempt after k
    (only reachable if the per-attempt seal check is broken) is recorded and
    fails the same way as the attempts before k.

    Returns ``(stand_in, calls)`` where ``calls["n"]`` is the number of times
    ``create_session`` was entered.
    """
    calls = {"n": 0}

    @asynccontextmanager
    async def stand_in(connection):
        calls["n"] += 1
        n = calls["n"]
        client = None
        if has_factory:
            client = connection["httpx_client_factory"]()
        if n != k:
            raise OSError("refused")
            yield None  # pragma: no cover
        try:
            await asyncio.Event().wait()  # never set: this is the stuck attempt
            yield None  # pragma: no cover
        except asyncio.CancelledError:
            _uncancel_current_task()
            if ending == "by_itself":
                await _swallow_cancels_for(0.2)
                raise RuntimeError("stuck attempt ended on its own") from None
            assert client is not None
            await _swallow_cancels_until(client.closed)
            raise RuntimeError("client already closed") from None

    return stand_in, calls


@pytest.mark.asyncio
@pytest.mark.timeout(30)
@pytest.mark.parametrize("transport", ["sse", "streamable_http"])
@pytest.mark.parametrize("cleanup", ["swallows_cancel", "blocked_on_transport"])
async def test_abandoned_http_load_is_force_closed_after_grace_bare_coroutine(
    monkeypatch, caplog, transport, cleanup
):
    """Variant a: a bare coroutine (no retry loop) that gets stuck while the
    test holds one client built through the instrumented connection's
    factory. Whichever shape its stuck cleanup takes, the reaper must
    force-close its client after the grace period and let the task end.

    ``swallows_cancel``: swallows only the abandonment's cancel, then blocks
    on an unrelated event -- it is the reaper's *second* cancel (issued right
    after force-closing) that actually ends it.
    ``blocked_on_transport``: swallows every cancel it receives and only ends
    once its own client is force-closed.
    """
    monkeypatch.setattr(mcp_adapter_module, "_HANDSHAKE_RECLAIM_GRACE_SECONDS", 0.2)
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)

    connection, _counts = _counting_http_connection(transport)
    recorder = _TransportRecorder("s")
    instrumented = recorder.instrument(connection)
    client = instrumented["httpx_client_factory"]()

    async def bare_load():
        if cleanup == "swallows_cancel":
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                _uncancel_current_task()
            await asyncio.Event().wait()  # ended only by the reaper's 2nd cancel
        else:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                _uncancel_current_task()
                await _swallow_cancels_until(client.closed)
        return []  # pragma: no cover

    before = set(mcp_adapter_module._active_load_tasks)
    reapers_before = set(mcp_adapter_module._RECLAIM_TASKS)
    with pytest.raises(TimeoutError):
        await _load_server_tools_bounded("s", bare_load(), 0.05, recorder=recorder)
    abandoned_at = asyncio.get_event_loop().time()

    task = _new_active_load_task(before)
    # Whichever ending: "swallows_cancel" is ended by the reaper's second
    # cancel (task.cancelled() is True); "blocked_on_transport" ends
    # normally once its own client is force-closed.
    await _assert_task_done_within(task, timeout=5)
    ended_after = asyncio.get_event_loop().time() - abandoned_at

    assert client.aclose_calls == 1
    assert ended_after < 0.2 + 0.5
    await asyncio.sleep(0.05)
    assert _new_reapers(reapers_before) == set()
    assert len(_reclaim_warnings(caplog.records)) == 1


@pytest.mark.asyncio
@pytest.mark.timeout(30)
@pytest.mark.parametrize("transport", ["sse", "streamable_http"])
async def test_abandoned_http_load_is_force_closed_after_grace_real_retry_loop(
    monkeypatch, caplog, transport
):
    """Variant b: the real retry loop and real bounded call. The load's first
    (and only) attempt gets stuck; after the grace period the reaper
    force-closes it. Because the retry loop checks the seal before every
    attempt, the load starts no second attempt at all -- it ends with
    ``_TransportReclaimedError``."""
    monkeypatch.setattr(mcp_adapter_module, "_HANDSHAKE_RECLAIM_GRACE_SECONDS", 1.5)
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)

    connection, counts = _counting_http_connection(transport)
    stand_in, calls = _make_stalling_create_session(
        k=1, ending="by_force_close", has_factory=True
    )
    monkeypatch.setattr(mcp_adapter_module, "create_session", stand_in)

    recorder = _TransportRecorder("s")
    before = set(mcp_adapter_module._active_load_tasks)
    reapers_before = set(mcp_adapter_module._RECLAIM_TASKS)
    with pytest.raises(TimeoutError):
        await _load_server_tools_bounded(
            "s",
            _load_direct_mcp_tools(
                "s",
                connection,
                name_prefix="p_",
                visibility=None,
                allow_users=None,
                recorder=recorder,
            ),
            0.5,
            recorder=recorder,
        )
    abandoned_at = asyncio.get_event_loop().time()

    built_at_abandon = counts["built"]
    task = _new_active_load_task(before)
    await _assert_task_done_within(task, timeout=5)
    assert isinstance(task.exception(), _TransportReclaimedError)
    ended_after = asyncio.get_event_loop().time() - abandoned_at

    assert counts["built"] == built_at_abandon  # zero new clients after abandonment
    assert calls["n"] == 1  # zero new attempts
    assert ended_after < 1.5 + 1 + 0.5
    await asyncio.sleep(0.05)
    assert _new_reapers(reapers_before) == set()
    assert len(_reclaim_warnings(caplog.records)) == 1
    assert _still_alive_warnings(caplog.records) == []


@pytest.mark.asyncio
@pytest.mark.timeout(30)
@pytest.mark.parametrize("unwind_delay_factor", [0.0, 0.5, 0.9])
async def test_reclaim_is_silent_when_abandoned_load_unwinds(
    monkeypatch, caplog, unwind_delay_factor
):
    """A load that unwinds by itself within the grace period is never
    force-closed, no reclaim WARNING is logged, and the reaper exits as soon
    as the load ends -- it does not hold the recorder for the full grace."""
    grace = 1.0
    monkeypatch.setattr(mcp_adapter_module, "_HANDSHAKE_RECLAIM_GRACE_SECONDS", grace)
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)

    connection, _counts = _counting_http_connection("streamable_http")
    recorder = _TransportRecorder("s")
    instrumented = recorder.instrument(connection)
    client = instrumented["httpx_client_factory"]()
    unwind_delay = grace * unwind_delay_factor

    async def bare_load():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            _uncancel_current_task()
            await _swallow_cancels_for(unwind_delay)
        return []  # pragma: no cover

    before = set(mcp_adapter_module._active_load_tasks)
    reapers_before = set(mcp_adapter_module._RECLAIM_TASKS)
    with pytest.raises(TimeoutError):
        await _load_server_tools_bounded("s", bare_load(), 0.05, recorder=recorder)

    task = _new_active_load_task(before)
    await _assert_task_done_within(task, timeout=5)

    assert client.aclose_calls == 0
    assert _reclaim_warnings(caplog.records) == []
    # The reaper's own asyncio.wait({task}, ...) returns as soon as the task
    # ends, so it does not linger for the rest of the grace period.
    await _wait_until_no_new_reapers(reapers_before, timeout=0.5)


def _new_active_load_task(before: set) -> "asyncio.Task[Any]":
    """Return the one task added to ``_active_load_tasks`` since ``before``
    was snapshotted.

    That set is module-level and shared across every test in this file, so
    picking any element of it (rather than the one this test's own call just
    added) could silently grab a task a previous test left stranded.
    """
    new_tasks = mcp_adapter_module._active_load_tasks - before
    assert len(new_tasks) == 1, (
        f"expected exactly one new load task, got {len(new_tasks)}"
    )
    return next(iter(new_tasks))


def _new_reapers(before: set) -> "set[asyncio.Task[Any]]":
    """Reapers added to ``_RECLAIM_TASKS`` since ``before`` was snapshotted.

    Like ``_active_load_tasks``, that set is module-level and shared across
    every test in this file, so a reaper another test left behind must not
    count here.
    """
    return mcp_adapter_module._RECLAIM_TASKS - before


async def _wait_until_no_new_reapers(before: set, timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while _new_reapers(before):
        assert loop.time() < deadline, f"reapers still running after {timeout}s"
        await asyncio.sleep(0.02)


async def _assert_task_done_within(task: "asyncio.Task[Any]", timeout: float) -> None:
    """Wait for ``task`` to finish within ``timeout``, and fail explicitly if
    it doesn't.

    ``asyncio.wait_for`` would cancel ``task`` at the deadline and then keep
    awaiting it, so a task that swallows every cancellation it receives would
    hang ``wait_for`` itself forever. ``asyncio.wait`` only watches; it
    returns at the deadline whether or not the task has finished, and never
    raises the task's own outcome (a return value, an exception, or having
    been cancelled) into the caller.
    """
    done, _pending = await asyncio.wait({task}, timeout=timeout)
    assert task in done, f"task did not finish within {timeout}s"


@pytest.mark.asyncio
@pytest.mark.timeout(30)
@pytest.mark.parametrize("caller_context", ["plain_task", "inside_task_group"])
async def test_caller_cancellation_schedules_reclaim(monkeypatch, caller_context):
    """Cancelling the caller (not a timeout) must go through the same
    recorder-sealing + reaper path, and the reaper must be able to start even
    from a caller that is itself unwinding a cancellation -- including one
    running inside a TaskGroup that is aborting, which is exactly why
    _schedule_reclaim uses asyncio.ensure_future and not the group's own
    create_task (that would raise RuntimeError once the group starts
    aborting)."""
    grace = 0.3
    monkeypatch.setattr(mcp_adapter_module, "_HANDSHAKE_RECLAIM_GRACE_SECONDS", grace)

    connection, _counts = _counting_http_connection("streamable_http")
    recorder = _TransportRecorder("s")
    instrumented = recorder.instrument(connection)
    client = instrumented["httpx_client_factory"]()
    load_started = asyncio.Event()

    async def bare_load():
        load_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            _uncancel_current_task()
            await _swallow_cancels_until(client.closed)
        return []  # pragma: no cover

    async def call_bounded():
        await _load_server_tools_bounded("s", bare_load(), 30, recorder=recorder)

    before = set(mcp_adapter_module._active_load_tasks)
    if caller_context == "plain_task":
        caller = asyncio.create_task(call_bounded())
        await asyncio.wait_for(load_started.wait(), timeout=5)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
    else:
        holder: dict[str, Any] = {}
        real_schedule_reclaim = mcp_adapter_module._schedule_reclaim

        async def _never_runs() -> None:
            pass  # pragma: no cover

        def probing_schedule_reclaim(server_name, task, recorder):
            # At the exact moment the caller's cancellation reaches
            # _schedule_reclaim, the enclosing group must already be
            # shutting down: trying to add a task to it here must be
            # refused the same way it would be for any other caller. This
            # checks that through the group's own public create_task, not
            # by reaching into its private state.
            group = holder["group"]
            probe_coro = _never_runs()
            try:
                with pytest.raises(RuntimeError):
                    group.create_task(probe_coro)
            finally:
                probe_coro.close()
            return real_schedule_reclaim(server_name, task, recorder)

        monkeypatch.setattr(
            mcp_adapter_module, "_schedule_reclaim", probing_schedule_reclaim
        )

        async def group_body():
            async with asyncio.TaskGroup() as tg:
                holder["group"] = tg
                holder["caller"] = tg.create_task(call_bounded())
                await asyncio.wait_for(load_started.wait(), timeout=5)
                # Cancel the task running this function (the group's own
                # parent task), not just the child: a child that is merely
                # cancelled makes the group return without aborting
                # (asyncio/taskgroups.py's _on_task_done: `if
                # task.cancelled(): return`, never calling `_abort()`).
                # Only a cancellation received while still inside the
                # `async with` block makes __aexit__ call `_abort()` --
                # the actual "the group is shutting down" state
                # `_schedule_reclaim` must survive.
                asyncio.current_task().cancel()

        group_task = asyncio.create_task(group_body())
        try:
            await group_task
        except (asyncio.CancelledError, BaseExceptionGroup):
            pass

    # The reaper must already be running (scheduled via ensure_future, not
    # refused by an aborting TaskGroup) by the time the caller has unwound.
    assert recorder._sealed is True
    load_task = _new_active_load_task(before)
    assert not load_task.done() or load_task.cancelled()
    await _assert_task_done_within(load_task, timeout=5)
    assert client.aclose_calls == 1


@pytest.mark.asyncio
@pytest.mark.timeout(30)
@pytest.mark.parametrize("tool_count", [0, 1, 3])
async def test_successful_load_never_reclaims(monkeypatch, tool_count):
    """A load that succeeds must never be handed to a reaper, whatever the
    number of tools it returns."""
    reapers_before = set(mcp_adapter_module._RECLAIM_TASKS)
    connection, _counts = _counting_http_connection("streamable_http")
    recorder = _TransportRecorder("s")
    instrumented = recorder.instrument(connection)
    client = instrumented["httpx_client_factory"]()

    async def quick_load():
        return MCPLoadResult(
            tools=tuple(object() for _ in range(tool_count)),
            loaded_servers=("s",) if tool_count else (),
            failures=(),
        )

    result = await _load_server_tools_bounded("s", quick_load(), 5, recorder=recorder)

    assert len(result.tools) == tool_count
    assert _new_reapers(reapers_before) == set()
    assert client.aclose_calls == 0
    assert recorder._sealed is False


@pytest.mark.asyncio
@pytest.mark.timeout(30)
@pytest.mark.parametrize("raised", [RuntimeError, OSError, httpx.HTTPError])
async def test_reclaim_failure_is_logged_and_contained(monkeypatch, caplog, raised):
    """A force-close failure (a client's aclose() itself raising) is logged
    and does not stop the reaper from finishing, or escape as an unretrieved
    task exception."""
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    grace = 0.1
    monkeypatch.setattr(mcp_adapter_module, "_HANDSHAKE_RECLAIM_GRACE_SECONDS", grace)

    recorder = _TransportRecorder("s")

    class _BrokenClient:
        def __init__(self) -> None:
            self.aclose_calls = 0

        async def aclose(self) -> None:
            self.aclose_calls += 1
            raise raised("boom")

    broken = _BrokenClient()
    recorder._clients.append(broken)

    async def bare_load():
        try:
            await asyncio.Event().wait()
        finally:
            _uncancel_current_task()
            await asyncio.Event().wait()  # stay stuck forever

    before = set(mcp_adapter_module._active_load_tasks)
    reapers_before = set(mcp_adapter_module._RECLAIM_TASKS)
    with pytest.raises(TimeoutError):
        await _load_server_tools_bounded("s", bare_load(), 0.05, recorder=recorder)

    task = _new_active_load_task(before)
    await _wait_until_no_new_reapers(reapers_before)

    assert broken.aclose_calls == 1
    close_failures = [
        r.getMessage()
        for r in caplog.records
        if r.name == _LOGGER_NAME
        and "closing an abandoned transport failed" in r.getMessage()
    ]
    assert len(close_failures) == 1
    assert raised.__name__ in close_failures[0]
    # The stuck load task itself is left running forever by this test; make
    # it done so pytest-asyncio's loop teardown does not hang on it.
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_reaper_own_failure_is_logged_and_contained(monkeypatch, caplog):
    """A failure inside the reaper itself (here: force_close raising, which
    it never does today) is logged once, and the reaper still ends normally
    rather than carrying an exception nobody retrieves."""
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    monkeypatch.setattr(mcp_adapter_module, "_HANDSHAKE_RECLAIM_GRACE_SECONDS", 0.1)
    recorder = _TransportRecorder("s")

    async def broken_force_close():
        raise RuntimeError("reaper boom")

    monkeypatch.setattr(recorder, "force_close", broken_force_close)
    release = asyncio.Event()

    async def stuck_load():
        await _swallow_cancels_until(release)

    before = set(mcp_adapter_module._active_load_tasks)
    reapers_before = set(mcp_adapter_module._RECLAIM_TASKS)
    with pytest.raises(TimeoutError):
        await _load_server_tools_bounded("s", stuck_load(), 0.05, recorder=recorder)
    task = _new_active_load_task(before)
    reapers = _new_reapers(reapers_before)
    assert len(reapers) == 1
    (reaper,) = reapers
    await _assert_task_done_within(reaper, timeout=2)
    failures = [
        r.getMessage()
        for r in caplog.records
        if r.name == _LOGGER_NAME
        and "reclaiming an abandoned initialization failed" in r.getMessage()
    ]
    release.set()
    await _assert_task_done_within(task, timeout=5)

    assert reaper.exception() is None
    assert failures == [
        "MCP server s: reclaiming an abandoned initialization failed (RuntimeError)"
    ]


@pytest.mark.asyncio
@pytest.mark.timeout(30)
@pytest.mark.parametrize("abandoned_by", ["timeout", "caller_cancel"])
async def test_abandoned_load_late_exception_is_consumed(caplog, abandoned_by):
    """An abandoned load that later ends with an ordinary exception has that
    exception retrieved exactly once by the loader (one DEBUG line), whether
    it was abandoned by the timeout or by its caller being cancelled."""
    caplog.set_level(logging.DEBUG, logger=_LOGGER_NAME)
    started = asyncio.Event()

    async def late_failing_load():
        started.set()
        await _swallow_cancels_for(0.1)
        raise RuntimeError("late failure")

    before = set(mcp_adapter_module._active_load_tasks)
    if abandoned_by == "timeout":
        with pytest.raises(TimeoutError):
            await _load_server_tools_bounded("s", late_failing_load(), 0.05)
    else:
        caller = asyncio.ensure_future(
            _load_server_tools_bounded("s", late_failing_load(), 30)
        )
        await started.wait()
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
    task = _new_active_load_task(before)
    await _assert_task_done_within(task, timeout=2)
    await asyncio.sleep(0)

    consumed = [
        r.getMessage()
        for r in caplog.records
        if r.name == _LOGGER_NAME
        and r.levelno == logging.DEBUG
        and "finished with: late failure" in r.getMessage()
    ]
    assert len(consumed) == 1


@pytest.mark.parametrize(
    "transport",
    ["sse", "streamable_http", "stdio", "websocket", None, "carrier-pigeon"],
)
@pytest.mark.parametrize("custom_factory", [False, True])
def test_transport_recorder_instrument_domain(monkeypatch, transport, custom_factory):
    """Only sse/streamable_http connections are instrumented; every other
    connection (including one with no transport key or an unknown one) comes
    back unchanged. Instrumenting delegates to the connection's own factory
    when it has one, else to the module's default. The module default is
    replaced by a stand-in, so no real httpx client is built."""
    recorder = _TransportRecorder("s")
    connection: dict = {"url": "http://x"}
    if transport is not None:
        connection["transport"] = transport
    inner_calls = {"n": 0}
    if custom_factory:

        def inner_factory(headers=None, timeout=None, auth=None):
            inner_calls["n"] += 1
            return _FakeHttpxClient()

        connection["httpx_client_factory"] = inner_factory

    default_calls = {"n": 0}

    def fake_default(headers=None, timeout=None, auth=None):
        default_calls["n"] += 1
        return _FakeHttpxClient()

    monkeypatch.setattr(mcp_adapter_module, "create_mcp_http_client", fake_default)

    out = recorder.instrument(connection)

    if transport in ("sse", "streamable_http"):
        assert out is not connection
        assert out["httpx_client_factory"] is not connection.get("httpx_client_factory")
        client = out["httpx_client_factory"]()
        assert isinstance(client, _FakeHttpxClient)
        assert inner_calls["n"] == (1 if custom_factory else 0)
        assert default_calls["n"] == (0 if custom_factory else 1)
        assert recorder._clients == [client]
    else:
        assert out is connection


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_instrumented_connection_never_leaves_session_creation(monkeypatch):
    """The connection handed to the tool adapter (and so, downstream, to
    gate_mcp_tools) must be the caller's original connection, never the
    instrumented copy create_session was given -- or every later tool call
    would add a client to a recorder nobody closes."""
    connection, _counts = _counting_http_connection("streamable_http")
    recorder = _TransportRecorder("s")
    seen_connections: list[object] = []

    @asynccontextmanager
    async def fake_create_session(conn):
        class FakeSession:
            async def initialize(self) -> None:
                return None

        yield FakeSession()

    async def fake_load_mcp_tools(session):
        from mcp.types import Tool as MCPTool

        return [MCPTool(name="t", description="d", inputSchema={"type": "object"})]

    def fake_build_adapter(server_name, conn, mcp_tool, **kwargs):
        seen_connections.append(conn)
        return object()

    monkeypatch.setattr(mcp_adapter_module, "create_session", fake_create_session)
    monkeypatch.setattr(mcp_adapter_module, "load_mcp_tools", fake_load_mcp_tools)
    monkeypatch.setattr(
        mcp_adapter_module, "_build_mcp_tool_adapter", fake_build_adapter
    )

    result = await _load_direct_mcp_tools(
        "s",
        connection,
        name_prefix="p_",
        visibility=None,
        allow_users=None,
        recorder=recorder,
    )

    assert len(result.tools) == 1
    assert len(seen_connections) == 1
    assert seen_connections[0] is connection
    assert seen_connections[0].get("httpx_client_factory") is connection.get(
        "httpx_client_factory"
    )


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_zero_timeout_never_reclaims(monkeypatch):
    """Timeout 0 disables both the deadline and the reclaim: the load coro
    is awaited directly, the recorder is never touched at all. The load
    suspends once, so a timeout of 0 treated as a deadline would expire
    before it returns."""
    reapers_before = set(mcp_adapter_module._RECLAIM_TASKS)
    connection, _counts = _counting_http_connection("streamable_http")
    recorder = _TransportRecorder("s")
    instrumented = recorder.instrument(connection)
    client = instrumented["httpx_client_factory"]()

    async def suspending_load():
        await asyncio.sleep(0.05)
        return MCPLoadResult(tools=(), loaded_servers=(), failures=())

    result = await _load_server_tools_bounded(
        "s", suspending_load(), 0, recorder=recorder
    )

    assert result.tools == ()
    assert _new_reapers(reapers_before) == set()
    assert client.aclose_calls == 0
    assert recorder._sealed is False


@pytest.mark.asyncio
@pytest.mark.parametrize("client_count", [0, 1, 3])
async def test_force_close_drops_client_references_group1(client_count):
    """force_close() closes every recorded client exactly once, drops the
    references first (a closed client still carries auth headers), and a
    second call closes nothing new (httpx treats a repeat aclose as a
    no-op, so clients from earlier retry attempts cost nothing)."""
    recorder = _TransportRecorder("s")
    clients = [_FakeHttpxClient() for _ in range(client_count)]
    recorder._clients.extend(clients)

    await recorder.force_close()

    assert recorder._clients == []
    for client in clients:
        assert client.aclose_calls == 1

    await recorder.force_close()

    for client in clients:
        assert client.aclose_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["seal", "force_close"])
async def test_force_close_drops_client_references_group2(action):
    """On an unsealed recorder, either seal() or force_close() makes the
    client factory and raise_if_sealed() both refuse from then on; seal()
    alone closes nothing, including a client built before it was called."""
    recorder = _TransportRecorder("s")
    connection, _counts = _counting_http_connection("streamable_http")
    instrumented = recorder.instrument(connection)
    pre_existing_client = instrumented["httpx_client_factory"]()

    assert recorder._sealed is False
    if action == "seal":
        recorder.seal()
    else:
        await recorder.force_close()

    with pytest.raises(_TransportReclaimedError):
        instrumented["httpx_client_factory"]()
    with pytest.raises(_TransportReclaimedError):
        recorder.raise_if_sealed()

    if action == "seal":
        assert recorder._clients == [pre_existing_client]
        assert pre_existing_client.aclose_calls == 0


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_sandboxed_load_does_not_use_a_recorder(monkeypatch):
    """The sandbox branch must never pass a recorder: the handshake runs
    inside the sandbox, so this process holds no MCP transport to close."""
    from xagent.core.tools.adapters.vibe import mcp_adapter as ma

    calls: list[dict] = []
    real_bounded = ma._load_server_tools_bounded

    async def spying_bounded(server_name, load_coro, timeout_seconds, *, recorder=None):
        calls.append({"server_name": server_name, "recorder": recorder})
        return await real_bounded(
            server_name, load_coro, timeout_seconds, recorder=recorder
        )

    def fake_should_sandbox(connection):
        return True

    class FakeSandbox:
        pass

    async def fake_load_sandboxed(connection, sandbox, tool_builder):
        from xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_mcp_tool_helper import (
            SandboxedMCPLoadResult,
        )

        return SandboxedMCPLoadResult(
            tools=(), adapter_error_types=(), wrap_error_types=()
        )

    monkeypatch.setattr(ma, "_load_server_tools_bounded", spying_bounded)
    monkeypatch.setattr(ma, "should_sandbox_mcp_connection", fake_should_sandbox)
    monkeypatch.setattr(ma, "load_sandboxed_mcp_tools", fake_load_sandboxed)

    await load_mcp_tools_as_agent_tools(
        {"s": {"transport": "streamable_http", "url": "http://x"}},
        sandbox=FakeSandbox(),
    )

    assert len(calls) == 1
    assert calls[0]["recorder"] is None


def test_instrumentable_transports_match_session_creators():
    """`_INSTRUMENTABLE_TRANSPORTS` must name exactly the transports whose
    session creator in sessions.py accepts an httpx client factory -- the
    only transports for which adding that key to the connection dict does
    not raise TypeError before a byte is sent."""
    from xagent.core.tools.core.mcp import sessions as sessions_module

    creators = {
        "stdio": sessions_module._create_stdio_session,
        "sse": sessions_module._create_sse_session,
        "streamable_http": sessions_module._create_streamable_http_session,
        "websocket": sessions_module._create_websocket_session,
    }
    accepts_factory = {
        transport
        for transport, fn in creators.items()
        if "httpx_client_factory" in inspect.signature(fn).parameters
    }
    does_not_accept = set(creators) - accepts_factory

    assert accepts_factory == mcp_adapter_module._INSTRUMENTABLE_TRANSPORTS
    assert does_not_accept == {"stdio", "websocket"}

    # The creator table above is written by hand: check it names every
    # transport create_session dispatches on, so a new transport cannot
    # bypass this check.
    import ast

    tree = ast.parse(inspect.getsource(sessions_module.create_session))
    dispatched = {
        node.comparators[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "transport"
        and isinstance(node.comparators[0], ast.Constant)
    }
    assert dispatched == set(creators)


_I17_CELLS = [
    (transport, k, ending)
    for transport in ("sse", "streamable_http")
    for k in (1, 2, 3)
    for ending in ("by_itself", "by_force_close")
] + [(transport, 1, "by_itself") for transport in ("stdio", "websocket")]


@pytest.mark.asyncio
@pytest.mark.timeout(30)
@pytest.mark.parametrize("transport,k,ending", _I17_CELLS)
async def test_abandoned_load_starts_no_further_attempt(
    monkeypatch, caplog, transport, k, ending
):
    """After abandonment, no transport starts another attempt: the retry
    loop checks the seal before every attempt, whatever the transport.
    A load whose k-th (of 3) attempt is the one abandoned ends with
    _TransportReclaimedError when k < 3 (the next attempt's check fails it);
    when k == 3 there is no next attempt, so it returns normally with a
    failure record instead of raising -- attempts == 3, error_type is the
    stand-in's own exception class name."""
    grace = 1.5
    monkeypatch.setattr(mcp_adapter_module, "_HANDSHAKE_RECLAIM_GRACE_SECONDS", grace)
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)

    has_factory = transport in ("sse", "streamable_http")
    if has_factory:
        connection, counts = _counting_http_connection(transport)
    else:
        connection, counts = {"transport": transport}, {"built": 0}

    stand_in, calls = _make_stalling_create_session(
        k=k, ending=ending, has_factory=has_factory
    )
    monkeypatch.setattr(mcp_adapter_module, "create_session", stand_in)

    recorder = _TransportRecorder("s")
    caller_timeout = (k - 1) + 0.5
    before = set(mcp_adapter_module._active_load_tasks)
    reapers_before = set(mcp_adapter_module._RECLAIM_TASKS)
    with pytest.raises(TimeoutError):
        await _load_server_tools_bounded(
            "s",
            _load_direct_mcp_tools(
                "s",
                connection,
                name_prefix="p_",
                visibility=None,
                allow_users=None,
                recorder=recorder,
            ),
            caller_timeout,
            recorder=recorder,
        )
    abandoned_at = asyncio.get_event_loop().time()

    task = _new_active_load_task(before)
    await _assert_task_done_within(task, timeout=10)
    if k < 3:
        assert isinstance(task.exception(), _TransportReclaimedError)
    else:
        result = task.result()
        assert len(result.failures) == 1
        failure = result.failures[0]
        assert failure.attempts == 3
        assert failure.error_type == "RuntimeError"
        assert failure.phase is MCPFailurePhase.SESSION_START
    ended_after = asyncio.get_event_loop().time() - abandoned_at

    assert calls["n"] == k  # zero new attempts after abandonment
    if has_factory:
        assert counts["built"] == k  # zero new clients after abandonment

    backoff = 1 if k < 3 else 0
    if ending == "by_itself":
        assert ended_after < 0.2 + backoff + 0.5
    else:
        assert ended_after < grace + backoff + 0.5

    await _wait_until_no_new_reapers(reapers_before)
    assert _still_alive_warnings(caplog.records) == []


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_second_reclaim_warning_reports_live_load_count(monkeypatch, caplog):
    """When an abandoned load is still alive after its transports were
    force-closed, the second WARNING carries how many MCP load tasks are
    still alive in this process -- not some other count that merely happens
    to match it -- and the reaper still removes itself from _RECLAIM_TASKS
    once it has logged it."""
    grace = 0.1
    monkeypatch.setattr(mcp_adapter_module, "_HANDSHAKE_RECLAIM_GRACE_SECONDS", grace)
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)

    connection, _counts = _counting_http_connection("streamable_http")
    recorder = _TransportRecorder("s")
    instrumented = recorder.instrument(connection)
    instrumented["httpx_client_factory"]()

    # Stays alive through both cancels the reaper delivers (abandonment, then
    # after force-close), whatever happens in the body below: it is released
    # only from this test's own `finally`, and (see _swallow_cancels_until)
    # gives up on its own after 5s regardless, so it can never hang the loop
    # teardown even if an assertion above fails first.
    release = asyncio.Event()

    async def bare_load():
        await _swallow_cancels_until(release)
        return []  # pragma: no cover

    # A second, unrelated pending task, added directly to the module-level
    # set the WARNING's count is read from. This makes that count (2)
    # diverge from _RECLAIM_TASKS's count (1), so a WARNING that reports the
    # wrong set shows up as the wrong number, not as a coincidentally correct
    # one (both sets hold exactly one task otherwise).
    dummy_task = asyncio.ensure_future(asyncio.Event().wait())
    mcp_adapter_module._active_load_tasks.add(dummy_task)

    task = None
    try:
        before = set(mcp_adapter_module._active_load_tasks)
        reapers_before = set(mcp_adapter_module._RECLAIM_TASKS)
        with pytest.raises(TimeoutError):
            await _load_server_tools_bounded("s", bare_load(), 0.05, recorder=recorder)

        task = _new_active_load_task(before)
        await _wait_until_no_new_reapers(reapers_before)

        still_alive = [
            r
            for r in caplog.records
            if r.name == _LOGGER_NAME and "still alive" in r.getMessage()
        ]
        assert len(still_alive) == 1
        # len(_active_load_tasks) at that time: before's snapshot (which
        # already includes the dummy task) plus the one real load task.
        assert still_alive[0].args[-1] == len(before) + 1
        assert "MCP load tasks alive in this process" in still_alive[0].getMessage()
    finally:
        release.set()
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        dummy_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await dummy_task
        mcp_adapter_module._active_load_tasks.discard(dummy_task)


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_reclaim_grace_exceeds_library_cleanup_bounds(monkeypatch):
    """The grace period must exceed the cleanup bound of each transport it
    cannot force-close (websocket's close_timeout default, stdio's
    termination timeout) and the retry loop's backoff, and grace plus that
    backoff must fit within the default initialization timeout. The backoff
    is measured as the real retry loop exhausting every attempt, which is at
    least what can remain after abandonment, not taken from the literal
    1s/3-attempts constants, so a change to either is caught here."""
    import mcp.client.stdio as stdio_module
    import websockets.asyncio.client as ws_client_module

    close_timeout_default = (
        inspect.signature(ws_client_module.connect.__init__)
        .parameters["close_timeout"]
        .default
    )
    assert close_timeout_default < mcp_adapter_module._HANDSHAKE_RECLAIM_GRACE_SECONDS
    assert (
        2 * stdio_module.PROCESS_TERMINATION_TIMEOUT
        < mcp_adapter_module._HANDSHAKE_RECLAIM_GRACE_SECONDS
    )

    @asynccontextmanager
    async def always_refuses(connection):
        raise OSError("refused")
        yield None  # pragma: no cover

    monkeypatch.setattr(mcp_adapter_module, "create_session", always_refuses)

    start = asyncio.get_event_loop().time()
    result = await _load_direct_mcp_tools(
        "s",
        {"transport": "stdio", "command": "true"},
        name_prefix="p_",
        visibility=None,
        allow_users=None,
    )
    elapsed = asyncio.get_event_loop().time() - start

    assert len(result.failures) == 1
    assert result.failures[0].attempts == 3
    assert elapsed < mcp_adapter_module._HANDSHAKE_RECLAIM_GRACE_SECONDS

    monkeypatch.delenv(MCP_TOOL_INIT_TIMEOUT_SECONDS, raising=False)
    assert (
        mcp_adapter_module._HANDSHAKE_RECLAIM_GRACE_SECONDS + elapsed
        <= get_mcp_tool_init_timeout_seconds()
    )
