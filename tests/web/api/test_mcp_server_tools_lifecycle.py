"""Admission covers the real adapter's detached transport cleanup."""

import asyncio
import gc
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from xagent.core.tools.adapters.vibe import mcp_adapter as adapter
from xagent.web.api import mcp
from xagent.web.services import mcp_connector_session as session


@pytest.fixture(autouse=True)
async def no_abandoned_listing_owners():
    from xagent.core.tools.adapters.vibe.mcp_load_lifetime import _active_lifetimes

    before = set(_active_lifetimes)
    yield
    for _ in range(20):
        if _active_lifetimes == before:
            break
        await asyncio.sleep(0)
    assert _active_lifetimes == before


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["cancelled", "timed_out", "failed"])
@pytest.mark.parametrize("through_service", [False, True])
async def test_slot_covers_real_bounded_loader_cleanup(
    monkeypatch, ending, through_service
):
    started = asyncio.Event()
    cleanup_started = asyncio.Event()
    cleanup_allowed = asyncio.Event()
    active = 0

    async def transport(*args, **kwargs):
        nonlocal active
        active += 1
        started.set()
        try:
            if ending == "failed":
                raise RuntimeError("offline failure")
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            while not cleanup_allowed.is_set():
                try:
                    await cleanup_allowed.wait()
                except asyncio.CancelledError:
                    # Model a transport whose close cannot finish yet, even
                    # when its initialization deadline expires during close.
                    continue
            active -= 1

    async def listing(db, *, user, server_id, name_prefix):
        if through_service:
            return await session._start_and_list(
                SimpleNamespace(release_db_connection=lambda: None),
                {
                    "name": "offline",
                    "transport": "stdio",
                    "config": {"command": "unused"},
                },
                user_id=user.id,
                server_name="offline",
                name_prefix=name_prefix,
                refuse_sandboxed=False,
                result=lambda outcome, **kw: session.ConnectorToolsResult(
                    outcome, server_id, "offline", **kw
                ),
            )
        result = await adapter.load_mcp_tools_as_agent_tools(
            {"offline": {"transport": "stdio", "command": "unused"}},
            name_prefix=name_prefix,
        )
        return session.ConnectorToolsResult(
            session.ConnectorToolsOutcome.OK, server_id, "offline", load_result=result
        )

    monkeypatch.setattr(mcp, "_MCP_SERVER_TOOLS_MAX_INFLIGHT", 1)
    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_tools._apply_stdio_output_limit_env",
        lambda configs, _: configs,
    )
    monkeypatch.setattr(session, "list_connector_tools", listing)
    monkeypatch.setattr(adapter, "_load_direct_mcp_tools", transport)
    monkeypatch.setattr(
        adapter._root_config, "get_mcp_tool_init_timeout_seconds", lambda: 0.1
    )
    user = SimpleNamespace(id=1, is_admin=False)
    first = asyncio.create_task(mcp.get_mcp_server_tools(1, user, object()))
    try:
        await asyncio.wait_for(started.wait(), 2)
        if ending == "cancelled":
            first.cancel()
        await asyncio.wait_for(asyncio.gather(first, return_exceptions=True), 2)
        await asyncio.wait_for(cleanup_started.wait(), 2)
        assert active == 1
        with pytest.raises(HTTPException) as exc:
            await asyncio.wait_for(mcp.get_mcp_server_tools(1, user, object()), 1)
        assert exc.value.status_code == 503
        assert active == 1
    finally:
        cleanup_allowed.set()
        await asyncio.gather(first, return_exceptions=True)
        await asyncio.gather(*list(adapter._active_load_tasks), return_exceptions=True)
        await asyncio.gather(*list(adapter._RECLAIM_TASKS), return_exceptions=True)
    for _ in range(20):
        if not mcp._mcp_server_tools_gate().locked():
            break
        await asyncio.sleep(0)
    assert active == 0
    assert not mcp._mcp_server_tools_gate().locked()
    # A completed transport makes admission usable again, not just unlocked.
    with pytest.raises(HTTPException) as exc:
        await asyncio.wait_for(mcp.get_mcp_server_tools(1, user, object()), 2)
    assert exc.value.status_code == 502
    await asyncio.gather(*list(adapter._active_load_tasks), return_exceptions=True)
    await asyncio.gather(*list(adapter._RECLAIM_TASKS), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("pause_at", ["acquire", "attach", "release"])
async def test_cancelled_listing_owns_sandbox_until_release(monkeypatch, pause_at):
    from xagent.web import sandbox_manager

    paused = asyncio.Event()
    proceed = asyncio.Event()
    released = asyncio.Event()

    async def step(name):
        if name == pause_at:
            paused.set()
            await proceed.wait()

    class Manager:
        async def get_or_create_lease_provider(self, *args):
            await step("acquire")
            return object()

        async def attach_provider(self, *args):
            await step("attach")
            return True

        async def release(self, *args):
            await step("release")
            released.set()

    async def loader(*args, **kwargs):
        return adapter.MCPLoadResult(tools=(), loaded_servers=(), failures=())

    async def listing(db, *, user, server_id, name_prefix):
        return await session._start_and_list(
            SimpleNamespace(release_db_connection=lambda: None),
            {
                "name": "offline",
                "transport": "stdio",
                "config": {"command": "npx", "args": ["unused"]},
            },
            user_id=user.id,
            server_name="offline",
            name_prefix=name_prefix,
            refuse_sandboxed=False,
            result=lambda outcome, **kw: session.ConnectorToolsResult(
                outcome, server_id, "offline", **kw
            ),
        )

    monkeypatch.setattr(mcp, "_MCP_SERVER_TOOLS_MAX_INFLIGHT", 1)
    monkeypatch.setattr(session, "list_connector_tools", listing)
    monkeypatch.setattr(sandbox_manager, "get_sandbox_manager", Manager)
    monkeypatch.setattr(adapter, "load_mcp_tools_as_agent_tools", loader)
    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_tools._apply_stdio_output_limit_env",
        lambda configs, _: configs,
    )
    user = SimpleNamespace(id=1, is_admin=False)
    first = asyncio.create_task(mcp.get_mcp_server_tools(1, user, object()))
    try:
        await asyncio.wait_for(paused.wait(), 2)
        first.cancel()
        await asyncio.wait_for(asyncio.gather(first, return_exceptions=True), 1)
        assert first.cancelled()
        first.cancel()  # Repeated request cancellation must not interrupt release.
        gc.collect()  # Pending owners must have roots beyond their own callbacks.
        with pytest.raises(HTTPException) as exc:
            await asyncio.wait_for(mcp.get_mcp_server_tools(1, user, object()), 0.5)
        assert exc.value.status_code == 503
        assert not released.is_set()
    finally:
        proceed.set()
        await asyncio.gather(first, return_exceptions=True)
    await asyncio.wait_for(released.wait(), 2)
    for _ in range(20):
        if not mcp._mcp_server_tools_gate().locked():
            break
        await asyncio.sleep(0)
    assert not mcp._mcp_server_tools_gate().locked()
