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
    assert events == ["waiting", "active", "finished"]
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
    assert runtime._cache_session.phase == "finished"
    assert notify.await_count == 3


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
