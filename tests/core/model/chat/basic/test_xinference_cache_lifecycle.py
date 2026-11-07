import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from xagent.core.model.chat.basic.xinference import XinferenceLLM
from xagent.core.model.chat.cache_lifecycle import AgentCacheSession


@pytest.mark.asyncio
@pytest.mark.parametrize("supported", [True, False])
@pytest.mark.parametrize("mode", ["chat", "stream", "runtime-stream"])
async def test_cache_lifecycle_is_automatic_only_for_discovered_support(
    supported: bool, mode: str
) -> None:
    streaming = mode != "chat"

    async def chunks():
        yield {"choices": [{"delta": {"content": "OK"}, "finish_reason": "stop"}]}

    handle = SimpleNamespace(
        chat=AsyncMock(return_value=chunks() if streaming else {"content": "OK"}),
        close=AsyncMock(),
    )
    sent = asyncio.Event()

    async def notify(*args):
        sent.set()

    client = SimpleNamespace(
        get_chat_model=AsyncMock(return_value=(handle, False)),
        supports_agent_cache_lifecycle=lambda _: supported,
        update_cache_session=AsyncMock(side_effect=notify),
        close=AsyncMock(),
    )
    llm = XinferenceLLM(model_name="model")
    llm._client = client
    session = AgentCacheSession()

    async def invoke() -> None:
        if mode == "runtime-stream":
            from xagent.core.agent import PatternRuntime

            runtime = PatternRuntime()
            runtime._cache_session = session
            assert (
                await runtime.run_streaming_llm_call(
                    llm, messages=[{"role": "user", "content": "OK"}]
                )
                == "OK"
            )
        elif streaming:
            async for _ in llm.stream_chat(
                messages=[{"role": "user", "content": "OK"}]
            ):
                pass
        else:
            await llm.chat(messages=[{"role": "user", "content": "OK"}])

    try:
        await session.create_task(invoke())
        config = handle.chat.call_args.kwargs["generate_config"]
        if supported:
            assert config["cache_session"] == {"id": session.id, "revision": 0}
            await session.transition("waiting")
            await asyncio.wait_for(sent.wait(), 1)
            client.update_cache_session.assert_awaited_once_with(
                "model", {"id": session.id, "revision": 1, "phase": "waiting"}
            )
        else:
            assert "cache_session" not in config
            await session.transition("waiting")
            client.update_cache_session.assert_not_awaited()
    finally:
        await session.transition("finished")
        if session._publisher is not None:
            await session._publisher
        await llm.close()
