import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from xagent.core.agent import PatternRuntime
from xagent.core.model.chat.cache_lifecycle import agent_cache_session


@pytest.mark.asyncio
async def test_parallel_tools_wait_until_last_tool_finishes_then_resume() -> None:
    runtime = PatternRuntime()
    events: list[dict] = []
    waiting = asyncio.Event()

    async def notify(body: dict) -> None:
        events.append(body)
        if body["phase"] == "waiting":
            waiting.set()

    async def chat(**kwargs: object) -> dict:
        session = agent_cache_session.get()
        assert session is not None
        metadata = session.register(("server", "model"), notify)
        assert metadata is not None
        return metadata

    model = SimpleNamespace(chat=chat)
    first = await runtime.run_llm_call(model)
    releases = [asyncio.Event(), asyncio.Event()]
    started = [asyncio.Event(), asyncio.Event()]

    async def tool(index: int) -> str:
        started[index].set()
        await releases[index].wait()
        return "ok"

    tasks = [
        asyncio.create_task(runtime.run_tool_call(lambda i=i: tool(i)))
        for i in range(2)
    ]
    await asyncio.wait_for(waiting.wait(), 1)
    await asyncio.gather(*(event.wait() for event in started))
    assert [event["phase"] for event in events] == ["waiting"]
    releases[0].set()
    await tasks[0]
    assert [event["phase"] for event in events] == ["waiting"]
    releases[1].set()
    assert await tasks[1] == "ok"
    second = await runtime.run_llm_call(model)
    assert first["id"] == second["id"]
    assert second["revision"] == 2
    await runtime.finish_cache_session()
    if runtime._cache_session._publisher is not None:
        await runtime._cache_session._publisher
    assert [event["phase"] for event in events] == ["waiting", "active", "finished"]
    assert [event["revision"] for event in events] == [1, 2, 3]
    assert not runtime._cache_session._providers
    assert runtime._cache_session.register(("server", "late"), notify) is None
    assert agent_cache_session.get() is None


@pytest.mark.asyncio
async def test_overlapping_llm_and_tool_keep_session_active() -> None:
    runtime = PatternRuntime()
    notify = AsyncMock()
    runtime._cache_session.register(("server", "model"), notify)
    release = asyncio.Event()
    entered = asyncio.Event()

    async def chat(**kwargs: object) -> str:
        entered.set()
        await release.wait()
        return "ok"

    llm_task = asyncio.create_task(runtime.run_llm_call(SimpleNamespace(chat=chat)))
    await entered.wait()
    assert (
        await runtime.run_tool_call(lambda: asyncio.sleep(0, result="tool")) == "tool"
    )
    notify.assert_not_awaited()
    release.set()
    assert await llm_task == "ok"


@pytest.mark.asyncio
async def test_cancelled_tool_propagates_and_session_can_finish() -> None:
    runtime = PatternRuntime()
    waiting = asyncio.Event()
    events = []

    async def notify(body: dict) -> None:
        events.append(body["phase"])
        waiting.set()

    runtime._cache_session.register(("server", "model"), notify)
    task = asyncio.create_task(runtime.run_tool_call(lambda: asyncio.Event().wait()))
    await waiting.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await runtime.finish_cache_session()
    if runtime._cache_session._publisher is not None:
        await runtime._cache_session._publisher
    assert events[0] == "waiting" and events[-1] == "finished"
    assert events.count("finished") == 1
    assert not runtime._active_tool_tasks


@pytest.mark.asyncio
async def test_hint_failure_does_not_fail_tools_or_finish() -> None:
    runtime = PatternRuntime()
    notify = AsyncMock(side_effect=RuntimeError("cache server unavailable"))
    runtime._cache_session.register(("server", "model"), notify)
    assert (
        await runtime.run_tool_call(lambda: asyncio.sleep(0, result="tool result"))
        == "tool result"
    )
    await runtime.finish_cache_session()
    if runtime._cache_session._publisher is not None:
        await runtime._cache_session._publisher
    assert runtime._cache_session.phase == "finished"
    assert 1 <= notify.await_count <= 3


@pytest.mark.asyncio
async def test_concurrent_executions_keep_separate_cache_sessions() -> None:
    runtimes = [PatternRuntime(), PatternRuntime()]
    ids = []

    async def chat(**kwargs: object) -> str:
        session = agent_cache_session.get()
        assert session is not None
        ids.append(session.id)
        await asyncio.sleep(0)
        assert agent_cache_session.get() is session
        return "ok"

    await asyncio.gather(
        *(runtime.run_llm_call(SimpleNamespace(chat=chat)) for runtime in runtimes)
    )
    assert len(set(ids)) == 2
    assert agent_cache_session.get() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["llm", "tool"])
async def test_cancellation_during_hint_drains_started_work(kind: str) -> None:
    runtime = PatternRuntime()
    hint_entered = asyncio.Event()
    cleanup_entered = asyncio.Event()
    cleanup_release = asyncio.Event()
    blocked_phase = "active" if kind == "llm" else "waiting"
    runtime._cache_session.phase = "waiting" if kind == "llm" else "active"

    async def notify(body: dict) -> None:
        if body["phase"] == blocked_phase:
            hint_entered.set()
            await asyncio.Event().wait()

    async def work(**kwargs: object) -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_entered.set()
            await cleanup_release.wait()

    runtime._cache_session.register(("server", "model"), notify)
    call = (
        runtime.run_llm_call(SimpleNamespace(chat=work))
        if kind == "llm"
        else runtime.run_tool_call(work)
    )
    task = asyncio.create_task(call)
    await hint_entered.wait()
    task.cancel()
    try:
        await asyncio.wait_for(cleanup_entered.wait(), 1)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(task), 0.03)
    finally:
        cleanup_release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not runtime._active_llm_tasks and not runtime._active_tool_tasks


@pytest.mark.asyncio
async def test_slow_control_never_blocks_completed_tool_results(monkeypatch) -> None:
    from xagent.core.model.chat import cache_lifecycle as hints

    monkeypatch.setattr(hints, "_HINT_TIMEOUT", 0.5)
    monkeypatch.setattr(hints, "_FINISH_RETRY_DELAY", 0)
    runtime = PatternRuntime()
    entered, release_tool = asyncio.Event(), asyncio.Event()

    async def notify(body):
        entered.set()
        await asyncio.Event().wait()

    runtime._cache_session.register(("slow-control", "model"), notify)
    task = asyncio.create_task(runtime.run_tool_call(release_tool.wait))
    await asyncio.wait_for(entered.wait(), 1)
    release_tool.set()
    assert await asyncio.wait_for(task, 0.1) is True
    # The HTTP operation is still in flight after the tool result is delivered.
    assert runtime._cache_session._publisher is not None
    await runtime.finish_cache_session()
    publisher = runtime._cache_session._publisher
    if publisher is not None:
        await asyncio.wait_for(publisher, 2)
    assert not runtime._cache_session._finished_providers


@pytest.mark.asyncio
async def test_waiting_heartbeat_refreshes_same_revision_and_stops_at_finish(
    monkeypatch,
) -> None:
    from xagent.core.model.chat import cache_lifecycle as hints

    monkeypatch.setattr(hints, "_HEARTBEAT_INTERVAL", 0.01)
    session = hints.AgentCacheSession()
    refreshed = asyncio.Event()
    events = []

    async def notify(payload):
        events.append(payload)
        if len(events) == 2:
            refreshed.set()

    session.register(("heartbeat", "model"), notify)
    await session.transition("waiting")
    await asyncio.wait_for(refreshed.wait(), 1)
    assert (
        events[0] == events[1] == {"id": session.id, "revision": 1, "phase": "waiting"}
    )
    await session.transition("finished")
    if session._publisher is not None:
        await session._publisher
    assert events[-1]["phase"] == "finished"
    assert events[-1]["revision"] == 2
    assert not session._providers and not session._finished_providers
    assert session._publisher is None


@pytest.mark.asyncio
async def test_failed_provider_cools_down_and_warns_once_without_secrets(
    monkeypatch, caplog
) -> None:
    from collections import OrderedDict

    import aiohttp

    from xagent.core.model.chat import cache_lifecycle as hints

    monkeypatch.setattr(hints, "_warned_providers", OrderedDict())
    session = hints.AgentCacheSession()
    sent = asyncio.Event()

    async def fail(payload):
        sent.set()
        raise aiohttp.ClientResponseError(
            request_info=SimpleNamespace(real_url="https://secret.invalid/token"),
            history=(),
            status=401,
            message="api-secret",
        )

    notify = AsyncMock(side_effect=fail)
    session.register(("secret-provider", "model"), notify)
    await session.transition("waiting")
    await asyncio.wait_for(sent.wait(), 1)
    await session.transition("active")
    if session._publisher is not None:
        await session._publisher
    assert notify.await_count == 1
    await session.transition("finished")
    if session._publisher is not None:
        await session._publisher
    assert notify.await_count == 2  # Final state still gets a best-effort attempt.
    warnings = [r.getMessage() for r in caplog.records if r.name == hints.__name__]
    assert len(warnings) == 1 and "401" in warnings[0]
    assert "api-secret" not in warnings[0] and "secret.invalid" not in warnings[0]


@pytest.mark.asyncio
async def test_finished_retries_once_after_transient_failure(monkeypatch) -> None:
    from xagent.core.model.chat import cache_lifecycle as hints

    monkeypatch.setattr(hints, "_FINISH_RETRY_DELAY", 0)
    session = hints.AgentCacheSession()
    notify = AsyncMock(side_effect=[ConnectionError("offline"), None])
    session.register(("retry", "model"), notify)
    await session.transition("finished")
    if session._publisher is not None:
        await session._publisher
    assert notify.await_count == 2
    assert notify.await_args_list[0] == notify.await_args_list[1]
    assert not session._finished_providers


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["llm", "tool", "stream"])
async def test_repeated_cancellation_still_removes_work_from_runtime(
    monkeypatch, kind
) -> None:
    runtime = PatternRuntime()
    cleanup, blocking_sync = asyncio.Event(), asyncio.Event()
    original_sync = runtime._sync_cache_phase
    first = True

    async def sync():
        nonlocal first
        if first:
            first = False
            blocking_sync.set()
            await asyncio.Event().wait()
        await original_sync()

    async def work(**kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cleanup.set()
            await asyncio.Event().wait()

    class Model:
        chat = staticmethod(work)

        async def stream_chat(self, **kwargs):
            await work()
            yield "unreachable"

    monkeypatch.setattr(runtime, "_sync_cache_phase", sync)
    if kind == "tool":
        call = runtime.run_tool_call(work)
    elif kind == "llm":
        call = runtime.run_llm_call(Model())
    else:
        call = runtime.run_streaming_llm_call(Model())
    task = asyncio.create_task(call)
    await asyncio.wait_for(blocking_sync.wait(), 1)
    # Give the owned work task a turn before cancelling during phase setup.
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.wait_for(cleanup.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not runtime._active_tool_tasks and not runtime._active_llm_tasks
    assert runtime._cache_session.phase == "active"


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["end", "error"])
async def test_pattern_hooks_count_nested_patterns_and_renew_finished_session(
    ending,
) -> None:
    from xagent.core.agent import ExecutionContext

    runtime = PatternRuntime()
    context = ExecutionContext(execution_id="cache-hooks")
    pattern = SimpleNamespace()
    events = []

    async def notify(payload):
        events.append(payload)

    original_id = runtime._cache_session.id
    await runtime.on_pattern_start(context=context, pattern=pattern)
    await runtime.on_pattern_start(context=context, pattern=pattern)
    runtime._cache_session.register(("hooks", "model"), notify)
    await runtime.on_pattern_end(
        context=context, pattern=pattern, result={"status": "completed"}
    )
    assert runtime._cache_pattern_count == 1 and events == []
    if ending == "error":
        await runtime.on_pattern_error(
            context=context, pattern=pattern, error=RuntimeError("failed")
        )
        assert runtime.pattern_error_reported
    else:
        await runtime.on_pattern_end(
            context=context, pattern=pattern, result={"status": "completed"}
        )
    if runtime._cache_session._publisher is not None:
        await runtime._cache_session._publisher
    assert [e["phase"] for e in events] == ["finished"]
    assert runtime._cache_pattern_count == 0
    await runtime.on_pattern_start(context=context, pattern=pattern)
    assert runtime._cache_session.id != original_id
    assert runtime._cache_session.phase == "active"
    assert runtime._cache_pattern_count == 1
    await runtime.finish_cache_session()


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["end", "error"])
async def test_terminal_trace_precedes_cancelled_cache_cleanup(
    monkeypatch, ending
) -> None:
    from xagent.core.agent import ExecutionContext

    runtime = PatternRuntime()
    emitted = []

    async def emit(*args, **kwargs):
        emitted.append("trace")

    async def finish():
        emitted.append("cache")
        raise asyncio.CancelledError()

    monkeypatch.setattr(runtime, "_emit_pattern_trace", emit)
    monkeypatch.setattr(runtime, "_emit_trace_event", emit)
    monkeypatch.setattr(runtime, "_cache_pattern_finished", finish)
    options = {"context": ExecutionContext(), "pattern": SimpleNamespace()}
    with pytest.raises(asyncio.CancelledError):
        if ending == "end":
            await runtime.on_pattern_end(**options, result={"status": "completed"})
        else:
            await runtime.on_pattern_error(**options, error=RuntimeError("failed"))
    assert emitted == ["trace", "cache"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 404, 422])
async def test_finished_does_not_retry_permanent_http_errors(status) -> None:
    import aiohttp

    from xagent.core.model.chat.cache_lifecycle import AgentCacheSession

    session = AgentCacheSession()
    notify = AsyncMock(
        side_effect=aiohttp.ClientResponseError(
            request_info=SimpleNamespace(real_url="https://provider.invalid"),
            history=(),
            status=status,
        )
    )
    session.register(("permanent-error", str(status)), notify)
    await session.transition("finished")
    if session._publisher is not None:
        await session._publisher
    notify.assert_awaited_once()
    assert not session._finished_providers
