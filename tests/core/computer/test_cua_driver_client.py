from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager
from typing import Any

import pytest
from mcp.types import CallToolResult, ImageContent, TextContent

from xagent.core.computer import cua_driver
from xagent.core.computer.cua_driver import CuaDriverError, CuaDriverMCPClient


class FakeSession:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.initialized = False
        self.closed = False

    async def __aenter__(self) -> "FakeSession":
        return self

    async def __aexit__(self, *_args: Any) -> None:
        self.closed = True

    async def initialize(self) -> None:
        self.initialized = True

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> CallToolResult:
        self.calls.append((name, arguments))
        if name == "failure":
            return CallToolResult(
                isError=True,
                content=[TextContent(type="text", text="permission denied")],
            )
        return CallToolResult(
            content=[
                TextContent(type="text", text=f"{name} done"),
                ImageContent(
                    type="image",
                    data=base64.b64encode(b"png").decode(),
                    mimeType="image/png",
                ),
            ],
            structuredContent={"name": name},
        )


class BlockingSession(FakeSession):
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        super().__init__(*_args, **_kwargs)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> CallToolResult:
        self.calls.append((name, arguments))
        if name == "blocking":
            self.started.set()
            await self.release.wait()
        return CallToolResult(
            content=[TextContent(type="text", text=f"{name} done")],
            structuredContent={"name": name},
        )


class BlockingInitializeSession(FakeSession):
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        super().__init__(*_args, **_kwargs)
        self.initialize_started = asyncio.Event()

    async def initialize(self) -> None:
        self.initialize_started.set()
        await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_cua_driver_client_serializes_calls_on_owned_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions: list[FakeSession] = []

    @asynccontextmanager
    async def fake_stdio(*_args: Any, **kwargs: Any):
        assert kwargs["errlog"].fileno() >= 0
        yield object(), object()

    def fake_session(*args: Any, **kwargs: Any) -> FakeSession:
        session = FakeSession(*args, **kwargs)
        sessions.append(session)
        return session

    monkeypatch.setattr(cua_driver, "stdio_client", fake_stdio)
    monkeypatch.setattr(cua_driver, "ClientSession", fake_session)
    client = CuaDriverMCPClient(command="fake-driver", timeout_seconds=1)

    first, second = await asyncio.gather(
        client.call_tool("one", {"value": 1}),
        client.call_tool("two", {"value": 2}),
    )
    await client.close()

    assert first.structured == {"name": "one"}
    assert first.image_bytes == b"png"
    assert second.text == "two done"
    assert len(sessions) == 1
    assert sessions[0].initialized is True
    assert sessions[0].calls == [
        ("one", {"value": 1}),
        ("two", {"value": 2}),
    ]
    assert sessions[0].closed is True


@pytest.mark.asyncio
async def test_cua_driver_client_surfaces_mcp_tool_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @asynccontextmanager
    async def fake_stdio(*_args: Any, **_kwargs: Any):
        yield object(), object()

    monkeypatch.setattr(cua_driver, "stdio_client", fake_stdio)
    monkeypatch.setattr(cua_driver, "ClientSession", FakeSession)
    client = CuaDriverMCPClient(command="fake-driver", timeout_seconds=1)

    with pytest.raises(CuaDriverError, match="permission denied"):
        await client.call_tool("failure")

    await client.close()


@pytest.mark.asyncio
async def test_cua_driver_client_restores_sessions_before_using_restarted_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions: list[FakeSession] = []

    @asynccontextmanager
    async def fake_stdio(*_args: Any, **_kwargs: Any):
        yield object(), object()

    def fake_session(*args: Any, **kwargs: Any) -> FakeSession:
        session = FakeSession(*args, **kwargs)
        sessions.append(session)
        return session

    monkeypatch.setattr(cua_driver, "stdio_client", fake_stdio)
    monkeypatch.setattr(cua_driver, "ClientSession", fake_session)
    client = CuaDriverMCPClient(command="fake-driver", timeout_seconds=1)

    await client.call_tool(
        "start_session",
        {"session": "task-1", "capture_scope": "window"},
    )
    assert client._queue is not None
    assert client._worker is not None
    await client._queue.put(None)
    await client._worker

    await client.call_tool("capture", {"session": "task-1"})
    await client.close()

    assert len(sessions) == 2
    assert sessions[1].calls == [
        (
            "start_session",
            {"session": "task-1", "capture_scope": "window"},
        ),
        ("capture", {"session": "task-1"}),
    ]


@pytest.mark.asyncio
async def test_cua_driver_client_cleans_up_timed_out_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = BlockingInitializeSession()

    @asynccontextmanager
    async def fake_stdio(*_args: Any, **_kwargs: Any):
        yield object(), object()

    monkeypatch.setattr(cua_driver, "stdio_client", fake_stdio)
    monkeypatch.setattr(cua_driver, "ClientSession", lambda *_args, **_kwargs: session)
    client = CuaDriverMCPClient(command="fake-driver", timeout_seconds=0.01)

    with pytest.raises(CuaDriverError, match="initialization timed out"):
        await client.call_tool("one")

    assert session.initialize_started.is_set()
    assert session.closed is True
    assert client._worker is None
    assert client._queue is None
    assert client._ready is None


@pytest.mark.asyncio
async def test_cua_driver_client_cleans_up_cancelled_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = BlockingInitializeSession()

    @asynccontextmanager
    async def fake_stdio(*_args: Any, **_kwargs: Any):
        yield object(), object()

    monkeypatch.setattr(cua_driver, "stdio_client", fake_stdio)
    monkeypatch.setattr(cua_driver, "ClientSession", lambda *_args, **_kwargs: session)
    client = CuaDriverMCPClient(command="fake-driver", timeout_seconds=0.01)
    call = asyncio.create_task(client.call_tool("one"))
    await session.initialize_started.wait()

    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call

    assert session.closed is True
    assert client._worker is None
    assert client._queue is None
    assert client._ready is None


@pytest.mark.asyncio
async def test_cua_driver_client_skips_cancelled_queued_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = BlockingSession()

    @asynccontextmanager
    async def fake_stdio(*_args: Any, **_kwargs: Any):
        yield object(), object()

    monkeypatch.setattr(cua_driver, "stdio_client", fake_stdio)
    monkeypatch.setattr(cua_driver, "ClientSession", lambda *_args, **_kwargs: session)
    client = CuaDriverMCPClient(command="fake-driver", timeout_seconds=1)

    blocking = asyncio.create_task(client.call_tool("blocking"))
    await session.started.wait()
    stale = asyncio.create_task(client.call_tool("stale"))
    while client._queue is None or client._queue.qsize() < 1:
        await asyncio.sleep(0)

    stale.cancel()
    with pytest.raises(asyncio.CancelledError):
        await stale
    session.release.set()

    assert (await blocking).structured == {"name": "blocking"}
    await client.close()
    assert session.calls == [("blocking", {})]
