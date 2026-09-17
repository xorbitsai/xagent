"""Client setup must not stall unrelated tasks on the agent event loop."""

import asyncio
import threading
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from xagent.core.model.chat.basic.openai import OpenAICompatibleLLM


@pytest_asyncio.fixture
async def pending_client(monkeypatch):
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()
    calls = []
    client = AsyncMock()

    def construct(**kwargs):
        calls.append(threading.get_ident())
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5), "Test did not release client construction"
        return client

    monkeypatch.setattr("xagent.core.model.chat.basic.openai.AsyncOpenAI", construct)
    llm = OpenAICompatibleLLM(
        "test", base_url=None, api_key="test", abilities=["chat", "vision"]
    )
    yield llm, client, entered, release, calls
    release.set()


@pytest.mark.parametrize("method", ["chat", "vision_chat", "stream_chat"])
async def test_request_setup_yields_and_drains_repeated_cancellation(
    pending_client, method
):
    llm, client, entered, release, calls = pending_client

    async def request():
        if method == "stream_chat":
            async for _ in llm.stream_chat([]):
                pass
        else:
            await getattr(llm, method)([])

    caller = asyncio.create_task(request())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert len(calls) == 1 and calls[0] != threading.get_ident()
        # Getting here while the constructor is blocked proves loop progress.
        for _ in range(2):
            caller.cancel()
            await asyncio.sleep(0)
            assert not caller.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert llm._client is client
        assert llm._client_init_task is None
        client.chat.completions.create.assert_not_called()
        await llm.close()
        client.close.assert_awaited_once()
        assert llm._client is None
    finally:
        release.set()
        await asyncio.gather(caller, return_exceptions=True)


async def test_concurrent_callers_share_initialization(pending_client):
    llm, client, entered, release, calls = pending_client
    first = asyncio.create_task(llm._ensure_client_async())
    second = asyncio.create_task(llm._ensure_client_async())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        first.cancel()
        await asyncio.sleep(0)
        assert not first.done() and not second.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        await second
        await llm._ensure_client_async()
        assert len(calls) == 1
        assert llm._client is client
        await llm.close()
    finally:
        release.set()
        await asyncio.gather(first, second, return_exceptions=True)


async def test_close_waits_for_inflight_initialization(pending_client):
    llm, client, entered, release, calls = pending_client
    initializing = asyncio.create_task(llm._ensure_client_async())
    closing = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        closing = asyncio.create_task(llm.close())
        await asyncio.sleep(0)
        assert not closing.done()
        client.close.assert_not_called()
        closing.cancel()
        await asyncio.sleep(0)
        assert not closing.done()
        release.set()
        await initializing
        with pytest.raises(asyncio.CancelledError):
            await closing
        client.close.assert_awaited_once()
        assert llm._client is None
        assert len(calls) == 1
    finally:
        release.set()
        await asyncio.gather(
            initializing, *([closing] if closing else []), return_exceptions=True
        )


async def test_failed_initialization_can_retry_and_preserves_provider_hook():
    client = AsyncMock()
    calls = []

    class Provider(OpenAICompatibleLLM):
        def _ensure_client(self):
            calls.append(threading.get_ident())
            if len(calls) == 1:
                raise ValueError("invalid client configuration")
            self._client = client

    llm = Provider("test", base_url=None, api_key="test")
    with pytest.raises(ValueError, match="invalid client configuration"):
        await llm._ensure_client_async()
    assert llm._client_init_task is None
    await llm._ensure_client_async()
    assert len(calls) == 2
    assert all(thread_id != threading.get_ident() for thread_id in calls)
    assert llm._client is client
    await llm.close()


async def test_close_logs_initialization_failure_without_sensitive_details(caplog):
    llm = OpenAICompatibleLLM("test", base_url=None, api_key="test")

    async def fail():
        raise ValueError("private-api-key at https://private-endpoint.invalid")

    llm._client_init_task = asyncio.create_task(fail())
    logger_name = "xagent.core.model.chat.basic.openai"
    with caplog.at_level("DEBUG", logger=logger_name):
        await llm.close()

    records = [record for record in caplog.records if record.name == logger_name]
    assert len(records) == 1
    assert records[0].getMessage() == (
        "Client initialization failed during close (ValueError)"
    )
    assert records[0].exc_info is None
    assert "private" not in caplog.text
    assert llm._client is None
    assert llm._client_init_task is None
