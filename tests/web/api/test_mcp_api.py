"""
Test MCP API endpoints and functions
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

import xagent.web.api.mcp as mcp_module
from xagent.core.tools.adapters.vibe.mcp_adapter import (
    MCPFailurePhase,
    MCPLoadResult,
    MCPServerLoadFailure,
)
from xagent.web.api.mcp import (
    _MCP_CONNECTION_TEST_MAX_INFLIGHT,
    _MCP_CONNECTION_TEST_SERVER_NAME,
    MCPConnectionTest,
    MCPConnectionTestResponse,
    MCPServerCreate,
    MCPServerUpdate,
    _auth_metadata_tampered,
    _build_server_config,
    _check_mcp_permission,
    _db_server_to_response,
    _global_config_tampered,
    _load_mcp_connection_test_tools,
    _mask_env,
    _mcp_connection_test_loads,
    _mcp_connection_test_timeout_result,
    _merge_masked_env,
    _project_mcp_tool_load_result,
    get_mcp_servers,
    get_supported_transports,
)
from xagent.web.api.mcp import test_mcp_connection as run_mcp_connection_test
from xagent.web.models.custom_api import CustomApi, UserCustomApi
from xagent.web.models.mcp import MCPServer
from xagent.web.models.user import User


async def _wait_until(predicate: Callable[[], bool], *, timeout: float = 1.0) -> None:
    """Poll ``predicate`` until it is true, without assuming how long that
    takes: each iteration only yields to the loop, so this converges as
    fast as the loop can schedule the tasks ``predicate`` is waiting on,
    rather than betting on a fixed sleep being long enough under load."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError(f"condition not met within {timeout}s")
        await asyncio.sleep(0)


async def _wait_until_connection_test_loads_empty(timeout: float = 1.0) -> None:
    """Poll until no in-flight load task tied to the current event loop
    remains in the module-level set.

    That set is module-level and outlives any one test's event loop, so a
    task a previous, unrelated test left stranded (tied to a loop that is
    already closed) must not make this test fail too: only tasks on the
    loop running right now count.
    """
    loop = asyncio.get_running_loop()

    def _pending_here() -> set:
        return {t for t in _mcp_connection_test_loads if t.get_loop() is loop}

    deadline = time.monotonic() + timeout
    while _pending_here() and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    assert not _pending_here()


@dataclass
class _FirstCallHangsStub:
    """A fake ``load_mcp_tools_as_agent_tools`` whose first call hangs and
    whose later calls succeed immediately, shared by every test below that
    needs to watch a slot get taken, then freed, then taken again."""

    call: Callable[..., Any]
    first_call_started: asyncio.Event
    first_call_cancelled: asyncio.Event
    release_first_call: asyncio.Event
    second_call_started: asyncio.Event


def _make_first_call_hangs_stub(
    *, swallow_cancellation: bool, raise_after_release: BaseException | None = None
) -> _FirstCallHangsStub:
    """Build a ``_FirstCallHangsStub``.

    The first call always sets ``first_call_started`` before it blocks.
    When ``swallow_cancellation`` is False, it then blocks on an event
    nobody sets, so the only way out is being cancelled — when that
    happens it sets ``first_call_cancelled`` and re-raises. When it is
    True, it instead swallows every cancellation it receives and only
    unblocks once the caller sets ``release_first_call``, at which point
    it raises ``raise_after_release`` if one was given, or returns an
    empty result. Every call after the first sets ``second_call_started``
    and returns an empty result immediately.
    """
    call_count = 0
    first_call_started = asyncio.Event()
    first_call_cancelled = asyncio.Event()
    release_first_call = asyncio.Event()
    second_call_started = asyncio.Event()

    async def stub(connections: dict, name_prefix: str) -> MCPLoadResult:
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            second_call_started.set()
            return MCPLoadResult(tools=(), loaded_servers=(), failures=())

        first_call_started.set()
        if swallow_cancellation:
            while True:
                try:
                    await release_first_call.wait()
                except asyncio.CancelledError:
                    continue
                break
        else:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                first_call_cancelled.set()
                raise

        if raise_after_release is not None:
            raise raise_after_release
        return MCPLoadResult(tools=(), loaded_servers=(), failures=())

    return _FirstCallHangsStub(
        call=stub,
        first_call_started=first_call_started,
        first_call_cancelled=first_call_cancelled,
        release_first_call=release_first_call,
        second_call_started=second_call_started,
    )


@pytest.mark.asyncio
async def test_connection_test_reports_successful_structured_load(monkeypatch):
    tool = MagicMock()

    async def load_tools(*args, **kwargs):
        return MCPLoadResult(tools=(tool,), loaded_servers=("test",), failures=())

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools",
        load_tools,
    )

    response = await run_mcp_connection_test(
        MCPConnectionTest(name="mail", transport="stdio", config={"command": "python"}),
        MagicMock(),
    )

    assert response.success is True
    assert response.details == {"tool_count": 1}


@pytest.mark.asyncio
async def test_connection_test_keeps_partial_tools_and_reports_safe_failures(
    monkeypatch,
):
    tool = MagicMock()

    async def load_tools(*args, **kwargs):
        return MCPLoadResult(
            tools=(tool,),
            loaded_servers=("test",),
            failures=(
                MCPServerLoadFailure(
                    server_name="test",
                    phase=MCPFailurePhase.ADAPTER_CONSTRUCTION,
                    error_type="BearerSecretError",
                ),
            ),
        )

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools",
        load_tools,
    )

    response = await run_mcp_connection_test(
        MCPConnectionTest(name="mail", transport="stdio", config={"command": "python"}),
        MagicMock(),
    )

    assert response.success is True
    assert response.details == {
        "tool_count": 1,
        "failures": [
            {
                "server_name": "test",
                "phase": "adapter_construction",
                "attempts": 1,
            }
        ],
    }
    assert "BearerSecretError" not in repr(response)


@pytest.mark.asyncio
async def test_connection_test_reports_structured_load_failure_without_exception_text(
    monkeypatch,
):
    async def load_tools(*args, **kwargs):
        return MCPLoadResult(
            tools=(),
            loaded_servers=(),
            failures=(
                MCPServerLoadFailure(
                    server_name="test",
                    phase=MCPFailurePhase.INITIALIZE,
                    error_type="BearerSecretError",
                    attempts=3,
                ),
            ),
        )

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools",
        load_tools,
    )

    response = await run_mcp_connection_test(
        MCPConnectionTest(name="mail", transport="stdio", config={"command": "python"}),
        MagicMock(),
    )

    assert response.success is False
    assert response.message == "MCP server initialization failed."
    assert response.details == {
        "tool_count": 0,
        "failures": [{"server_name": "test", "phase": "initialize", "attempts": 3}],
    }
    assert "BearerSecretError" not in repr(response)


@pytest.mark.asyncio
async def test_connection_test_reports_no_tools_as_failure(monkeypatch):
    async def load_tools(*args, **kwargs):
        return MCPLoadResult(
            tools=(),
            loaded_servers=(),
            failures=(
                MCPServerLoadFailure(
                    server_name="test",
                    phase=MCPFailurePhase.NO_TOOLS_RETURNED,
                    error_type=None,
                ),
            ),
        )

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools",
        load_tools,
    )

    response = await run_mcp_connection_test(
        MCPConnectionTest(name="mail", transport="stdio", config={"command": "python"}),
        MagicMock(),
    )

    assert response.success is False
    assert response.message == "MCP server returned no available tools."
    assert response.details["tool_count"] == 0


@pytest.mark.asyncio
@pytest.mark.timeout(10)
@pytest.mark.parametrize("timeout", [0, 5])
async def test_connection_test_caps_concurrent_loads(monkeypatch, timeout):
    """At most ``_MCP_CONNECTION_TEST_MAX_INFLIGHT`` test loads run at once;
    a freed slot lets exactly one more queued load start. Covers both the
    branch with no deadline (``timeout=0``) and the branch that shares the
    deadline with the load (``timeout=5``, long enough that it never fires
    during the test) — the cap comes from the same gate either way, but
    only exercising one branch would leave the other unguarded."""
    monkeypatch.setattr(
        mcp_module, "get_mcp_tool_init_timeout_seconds", lambda: timeout
    )

    started = 0
    release_events: list[asyncio.Event] = []

    async def fake_load(connections, name_prefix):
        nonlocal started
        started += 1
        release_event = asyncio.Event()
        release_events.append(release_event)
        await release_event.wait()
        return MCPLoadResult(tools=(), loaded_servers=(), failures=())

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools",
        fake_load,
    )

    total_requests = _MCP_CONNECTION_TEST_MAX_INFLIGHT + 2
    tasks = [
        asyncio.create_task(_load_mcp_connection_test_tools({"transport": "stdio"}))
        for _ in range(total_requests)
    ]
    try:
        await _wait_until(lambda: started >= _MCP_CONNECTION_TEST_MAX_INFLIGHT)
        assert started == _MCP_CONNECTION_TEST_MAX_INFLIGHT

        release_events[0].set()
        await _wait_until(lambda: started >= _MCP_CONNECTION_TEST_MAX_INFLIGHT + 1)
        assert started == _MCP_CONNECTION_TEST_MAX_INFLIGHT + 1
    finally:
        # Cancel and drain regardless of the outcome above: an assertion
        # failure here must not strand tasks stuck on an event nobody will
        # ever set, which would hang the test runner's loop teardown.
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await _wait_until_connection_test_loads_empty()


@pytest.mark.asyncio
@pytest.mark.timeout(10)
@pytest.mark.parametrize("where", ["queued", "loading"])
async def test_connection_test_deadline_matches_loader_timeout(monkeypatch, where):
    """A request that hits the deadline, whether still queued for a slot or
    already loading, gets exactly the response the real adapter loader's own
    per-server timeout would produce for that same server, within the
    deadline plus a margin, and that response is built from a load result
    whose un-projected ``failures``/``tools`` also match the loader's own —
    the interface response's projection drops ``error_type``, so comparing
    only the response would miss a mismatch there."""
    # Computed by running the real adapter loader, with only the per-server
    # direct load stubbed to never answer, so the loader's own per-server
    # timeout mechanism is what produces the failure — rather than
    # hand-typing a response that could drift from what that real path
    # produces.
    from xagent.config import MCP_TOOL_INIT_TIMEOUT_SECONDS
    from xagent.core.tools.adapters.vibe import mcp_adapter as mcp_adapter_module

    async def never_answers(server_name, connection, **kwargs):
        await asyncio.Event().wait()

    with pytest.MonkeyPatch.context() as real_timeout_patch:
        real_timeout_patch.setenv(MCP_TOOL_INIT_TIMEOUT_SECONDS, "1")
        real_timeout_patch.setattr(
            mcp_adapter_module, "_load_direct_mcp_tools", never_answers
        )
        real_timeout_result = await mcp_adapter_module.load_mcp_tools_as_agent_tools(
            {
                _MCP_CONNECTION_TEST_SERVER_NAME: {
                    "transport": "streamable_http",
                    "url": "http://x",
                }
            }
        )

    expected_projection = _project_mcp_tool_load_result(real_timeout_result)
    expected_details = {"tool_count": len(expected_projection.tools)}
    if expected_projection.failures:
        expected_details["failures"] = list(expected_projection.failures)
    expected = MCPConnectionTestResponse(
        success=False,
        message=expected_projection.failure_message,
        details=expected_details,
    )

    # A deadline of 1 second, comfortably above scheduling jitter under a
    # loaded CI runner, with a 1.3-second upper bound on the request's own
    # elapsed time.
    monkeypatch.setattr(mcp_module, "get_mcp_tool_init_timeout_seconds", lambda: 1.0)

    release_event = asyncio.Event()
    starts = 0

    async def hung_load(connections, name_prefix):
        nonlocal starts
        starts += 1
        while True:
            try:
                await release_event.wait()
            except asyncio.CancelledError:
                continue
            return MCPLoadResult(tools=(), loaded_servers=(), failures=())

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools",
        hung_load,
    )

    # Spy on the projection call the endpoint makes, so the assertions below
    # can compare the un-projected load result too, not just the response
    # built from it.
    seen_load_results: list[Any] = []
    real_project = mcp_module._project_mcp_tool_load_result

    def _spying_project(load_result: Any) -> Any:
        seen_load_results.append(load_result)
        return real_project(load_result)

    monkeypatch.setattr(mcp_module, "_project_mcp_tool_load_result", _spying_project)

    blockers: list[asyncio.Task] = []
    if where == "queued":
        # Fill every slot with a load that never returns on its own, so the
        # request under test never gets a slot before its own deadline.
        blockers = [
            asyncio.create_task(_load_mcp_connection_test_tools({"transport": "stdio"}))
            for _ in range(_MCP_CONNECTION_TEST_MAX_INFLIGHT)
        ]
        await _wait_until(lambda: starts >= _MCP_CONNECTION_TEST_MAX_INFLIGHT)

    try:
        start = time.monotonic()
        response = await run_mcp_connection_test(
            MCPConnectionTest(
                name="mail", transport="stdio", config={"command": "python"}
            ),
            MagicMock(),
        )
        elapsed = time.monotonic() - start

        assert response.model_dump() == expected.model_dump()
        assert elapsed < 1.3

        assert len(seen_load_results) == 1
        assert seen_load_results[0].failures == real_timeout_result.failures
        assert seen_load_results[0].tools == real_timeout_result.tools == ()
    finally:
        # Release every hung load regardless of the outcome above: a
        # mid-swallow load left un-released would hang the test runner's
        # loop teardown.
        release_event.set()
        await _wait_until_connection_test_loads_empty()
        await asyncio.gather(*blockers, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.timeout(10)
@pytest.mark.parametrize("trigger", ["deadline", "request_cancelled"])
async def test_connection_test_interrupts_load_and_returns_slot(monkeypatch, trigger):
    """When a request reaches its deadline or is itself cancelled, its load
    is cancelled too, and the freed slot lets the next queued load start."""
    monkeypatch.setattr(mcp_module, "_MCP_CONNECTION_TEST_MAX_INFLIGHT", 1)

    stub = _make_first_call_hangs_stub(swallow_cancellation=False)
    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools",
        stub.call,
    )

    if trigger == "deadline":
        # request A gets a short deadline so it times out on its own;
        # request B gets a long one so its own wait for a slot never expires
        # before A's slot is freed.
        timeouts = iter([0.2, 5.0])
        monkeypatch.setattr(
            mcp_module, "get_mcp_tool_init_timeout_seconds", lambda: next(timeouts)
        )

        task_a = asyncio.create_task(
            _load_mcp_connection_test_tools({"transport": "stdio"})
        )
        await asyncio.wait_for(stub.first_call_started.wait(), timeout=1)
        await task_a  # returns the loader's timeout result; does not raise
    else:
        monkeypatch.setattr(
            mcp_module, "get_mcp_tool_init_timeout_seconds", lambda: 5.0
        )

        task_a = asyncio.create_task(
            _load_mcp_connection_test_tools({"transport": "stdio"})
        )
        await asyncio.wait_for(stub.first_call_started.wait(), timeout=1)
        task_a.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task_a

    assert stub.first_call_cancelled.is_set()

    task_b = asyncio.create_task(
        _load_mcp_connection_test_tools({"transport": "stdio"})
    )
    await asyncio.wait_for(stub.second_call_started.wait(), timeout=0.1)

    task_b.cancel()
    await asyncio.gather(task_b, return_exceptions=True)
    await _wait_until_connection_test_loads_empty()


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_connection_test_zero_timeout_waits_and_propagates_cancel(monkeypatch):
    """With the deadline disabled, cancelling the request cancels the load
    directly and the slot returns via the ``finally`` so the next request
    can use it."""
    monkeypatch.setattr(mcp_module, "get_mcp_tool_init_timeout_seconds", lambda: 0)
    monkeypatch.setattr(mcp_module, "_MCP_CONNECTION_TEST_MAX_INFLIGHT", 1)

    stub = _make_first_call_hangs_stub(swallow_cancellation=False)
    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools",
        stub.call,
    )

    task_a = asyncio.create_task(
        _load_mcp_connection_test_tools({"transport": "stdio"})
    )
    await asyncio.wait_for(stub.first_call_started.wait(), timeout=1)

    task_a.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task_a

    assert stub.first_call_cancelled.is_set()

    result = await asyncio.wait_for(
        _load_mcp_connection_test_tools({"transport": "stdio"}), timeout=1
    )
    assert stub.second_call_started.is_set()
    assert result.tools == ()


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_connection_test_late_load_exception_is_consumed(monkeypatch, caplog):
    """A load that only raises after its request already returned at the
    deadline must not surface as an unretrieved task exception, and its
    slot must still come back."""
    monkeypatch.setattr(mcp_module, "get_mcp_tool_init_timeout_seconds", lambda: 0.2)
    monkeypatch.setattr(mcp_module, "_MCP_CONNECTION_TEST_MAX_INFLIGHT", 1)

    stub = _make_first_call_hangs_stub(
        swallow_cancellation=True, raise_after_release=RuntimeError("late failure")
    )
    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools",
        stub.call,
    )

    task_a = asyncio.create_task(
        _load_mcp_connection_test_tools({"transport": "stdio"})
    )
    result = await task_a  # returns at the deadline; the load is still running
    assert result == _mcp_connection_test_timeout_result()

    loop = asyncio.get_running_loop()
    handler_calls: list[dict] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: handler_calls.append(context))

    try:
        with caplog.at_level("DEBUG", logger="xagent.web.api.mcp"):
            stub.release_first_call.set()
            await _wait_until_connection_test_loads_empty()
    finally:
        loop.set_exception_handler(previous_handler)

    assert handler_calls == []
    debug_records = [
        r
        for r in caplog.records
        if r.levelname == "DEBUG" and "RuntimeError" in r.getMessage()
    ]
    assert len(debug_records) == 1

    second_result = await asyncio.wait_for(
        _load_mcp_connection_test_tools({"transport": "stdio"}), timeout=1
    )
    assert stub.second_call_started.is_set()
    assert second_result.tools == ()


class TestMCPServerModel:
    """Test MCPServer database model."""

    def test_to_connection_dict_stdio(self):
        """Test to_connection_dict method for STDIO transport."""
        server = MCPServer(
            name="test_server",
            transport="stdio",
            managed="external",
            command="python",
            args=["server.py"],
            env={"API_KEY": "secret"},
            cwd="/tmp",
        )

        connection_dict = server.to_connection_dict()

        assert connection_dict["name"] == "test_server"
        assert connection_dict["transport"] == "stdio"
        assert connection_dict["command"] == "python"
        assert connection_dict["args"] == ["server.py"]
        assert connection_dict["env"] == {"API_KEY": "secret"}
        assert connection_dict["cwd"] == "/tmp"

    def test_to_connection_dict_includes_mcp_concurrency_metadata(self):
        """MCP loader receives scheduling metadata alongside connection fields."""
        server = MCPServer(
            name="test_server",
            transport="stdio",
            managed="external",
            command="python",
            args=["server.py"],
            concurrency_safe=True,
            concurrent_tools=["list_messages"],
        )

        connection_dict = server.to_connection_dict()

        assert connection_dict["concurrency_safe"] is True
        assert connection_dict["concurrent_tools"] == ["list_messages"]

    def test_to_connection_dict_websocket(self):
        """Test to_connection_dict method for WebSocket transport."""
        server = MCPServer(
            name="test_websocket_server",
            transport="websocket",
            managed="external",
            url="ws://localhost:8080/ws",
            headers={"Authorization": "Bearer token"},
        )

        connection_dict = server.to_connection_dict()

        assert connection_dict["name"] == "test_websocket_server"
        assert connection_dict["transport"] == "websocket"
        assert connection_dict["url"] == "ws://localhost:8080/ws"
        assert connection_dict["headers"] == {"Authorization": "Bearer token"}

    def test_mcp_oauth_to_connection_dict_strips_static_authorization(self):
        """MCP OAuth runtime credentials must not fall back to stored headers."""
        server = MCPServer(
            name="test_oauth_server",
            transport="streamable_http",
            managed="external",
            url="https://mcp.example.com/mcp",
            headers={
                "Authorization": "Bearer static-token",
                "X-Request-Source": "xagent",
            },
            auth={"type": "mcp_oauth"},
        )

        connection_dict = server.to_connection_dict()

        assert connection_dict["headers"] == {"X-Request-Source": "xagent"}

    def test_to_config_dict_external(self):
        """Test to_config_dict method for external server."""
        server = MCPServer(
            name="external_server",
            transport="stdio",
            managed="external",
            description="Test external server",
            command="python",
            args=["server.py"],
            env={"KEY": "value"},
            cwd="/app",
        )

        config = server.to_config_dict()

        assert config["name"] == "external_server"
        assert config["description"] == "Test external server"
        assert config["managed"] == "external"
        assert config["transport"] == "stdio"
        assert config["command"] == "python"
        assert config["args"] == ["server.py"]
        assert config["env"] == {"KEY": "value"}
        assert config["cwd"] == "/app"
        assert config["concurrency_safe"] is False
        assert config["concurrent_tools"] == []
        # Internal-only fields should not be present
        assert "docker_url" not in config
        assert "docker_image" not in config

    def test_to_config_dict_includes_mcp_concurrency_config(self):
        """MCP config responses include explicit concurrency opt-in settings."""
        server = MCPServer(
            name="external_server",
            transport="stdio",
            managed="external",
            command="python",
            concurrency_safe=True,
            concurrent_tools=["list_messages"],
        )

        config = server.to_config_dict()

        assert config["concurrency_safe"] is True
        assert config["concurrent_tools"] == ["list_messages"]

    def test_to_config_dict_internal(self):
        """Test to_config_dict method for internal server."""
        server = MCPServer(
            name="internal_server",
            transport="stdio",
            managed="internal",
            description="Test internal server",
            docker_url="unix:///var/run/docker.sock",
            docker_image="mcp-server:latest",
            docker_environment={"ENV_VAR": "value"},
            docker_working_dir="/app",
            volumes=["/host:/container"],
            bind_ports={"8080": 8080},
            restart_policy="unless-stopped",
            auto_start=True,
        )

        config = server.to_config_dict()

        assert config["name"] == "internal_server"
        assert config["managed"] == "internal"
        assert config["docker_url"] == "unix:///var/run/docker.sock"
        assert config["docker_image"] == "mcp-server:latest"
        assert config["docker_environment"] == {"ENV_VAR": "value"}
        assert config["docker_working_dir"] == "/app"
        assert config["volumes"] == ["/host:/container"]
        assert config["bind_ports"] == {"8080": 8080}
        assert config["restart_policy"] == "unless-stopped"
        assert config["auto_start"] is True

    def test_from_config_external(self):
        """Test from_config class method for external server."""
        config = {
            "name": "test_server",
            "description": "Test server",
            "managed": "external",
            "transport": "stdio",
            "command": "python",
            "args": ["server.py"],
            "env": {"KEY": "value"},
            "cwd": "/app",
        }

        server = MCPServer.from_config(config)

        assert server.name == "test_server"
        assert server.description == "Test server"
        assert server.managed == "external"
        assert server.transport == "stdio"
        assert server.command == "python"
        assert server.args == ["server.py"]
        # env is encrypted at rest but decrypts back for consumption
        assert server.env["KEY"].startswith("gAAAAAB")
        assert server.to_connection_dict()["env"] == {"KEY": "value"}
        assert server.cwd == "/app"
        assert server.concurrency_safe is False
        assert server.concurrent_tools == []

    def test_from_config_persists_mcp_concurrency_config(self):
        """MCP concurrency opt-in survives database model construction."""
        config = {
            "name": "test_server",
            "description": "Test server",
            "managed": "external",
            "transport": "stdio",
            "command": "python",
            "concurrency_safe": True,
            "concurrent_tools": ["list_messages"],
        }

        server = MCPServer.from_config(config)

        assert server.concurrency_safe is True
        assert server.concurrent_tools == ["list_messages"]

    def test_from_config_internal(self):
        """Test from_config class method for internal server."""
        config = {
            "name": "internal_server",
            "managed": "internal",
            "transport": "stdio",
            "docker_url": "unix:///var/run/docker.sock",
            "docker_image": "mcp-server:latest",
            "docker_environment": {"ENV": "value"},
            "restart_policy": "always",
            "auto_start": True,
        }

        server = MCPServer.from_config(config)

        assert server.name == "internal_server"
        assert server.managed == "internal"
        assert server.docker_url == "unix:///var/run/docker.sock"
        assert server.docker_image == "mcp-server:latest"
        assert server.docker_environment == {"ENV": "value"}
        assert server.restart_policy == "always"
        assert server.auto_start is True

    def test_from_config_encrypts_oauth_access_token(self):
        """Test from_config encrypts OAuth access tokens at rest."""
        server = MCPServer.from_config(
            {
                "name": "oauth_server",
                "managed": "external",
                "transport": "streamable_http",
                "url": "https://example.com/mcp",
                "auth": {
                    "type": "oauth2",
                    "access_token": "plain-access-token",
                    "token_type": "Bearer",
                },
            }
        )

        assert server.auth["access_token"] != "plain-access-token"
        assert server.to_config_dict()["auth"]["access_token"] == "plain-access-token"

    def test_from_config_rejects_masked_auth_secret_without_existing_value(self):
        """Masked placeholders are response values, not creatable secrets."""
        with pytest.raises(ValueError, match="Masked auth value"):
            MCPServer.from_config(
                {
                    "name": "masked_server",
                    "managed": "external",
                    "transport": "streamable_http",
                    "url": "https://example.com/mcp",
                    "auth": {
                        "type": "mcp_oauth",
                        "client_id": "client-123",
                        "client_secret": "********",
                    },
                }
            )

    def test_transport_display_property(self):
        """Test transport_display property for different transports."""
        stdio_server = MCPServer(
            name="stdio_test", transport="stdio", managed="external"
        )
        websocket_server = MCPServer(
            name="ws_test", transport="websocket", managed="external"
        )
        sse_server = MCPServer(name="sse_test", transport="sse", managed="external")
        streamable_server = MCPServer(
            name="http_test", transport="streamable_http", managed="external"
        )
        unknown_server = MCPServer(
            name="unknown_test", transport="unknown", managed="external"
        )

        assert stdio_server.transport_display == "STDIO"
        assert websocket_server.transport_display == "WebSocket"
        assert sse_server.transport_display == "Server-Sent Events"
        assert streamable_server.transport_display == "Streamable HTTP"
        assert unknown_server.transport_display == "UNKNOWN"

    def test_repr_method(self):
        """Test __repr__ method."""
        server = MCPServer(
            id=1, name="test_server", transport="stdio", managed="external"
        )

        repr_str = repr(server)
        assert "MCPServer" in repr_str
        assert "id=1" in repr_str
        assert "name='test_server'" in repr_str
        assert "transport='stdio'" in repr_str
        assert "managed='external'" in repr_str


class TestMCPApiFunctions:
    """Test MCP API utility functions."""

    def test_get_supported_transports_data(self):
        """Test get_supported_transports_data function."""
        transports_data = get_supported_transports()

        assert "transports" in transports_data
        assert isinstance(transports_data["transports"], list)
        assert len(transports_data["transports"]) > 0

        # Check required transports
        transport_ids = [t["id"] for t in transports_data["transports"]]
        assert "stdio" in transport_ids
        assert "websocket" in transport_ids
        assert "sse" in transport_ids
        assert "streamable_http" in transport_ids

        # Check stdio transport structure
        stdio_transport = next(
            t for t in transports_data["transports"] if t["id"] == "stdio"
        )
        assert stdio_transport["name"] == "STDIO"
        assert "Standard input/output transport" in stdio_transport["description"]
        assert "config_fields" in stdio_transport
        assert isinstance(stdio_transport["config_fields"], list)

        # Check required stdio config fields
        config_fields = {f["name"]: f for f in stdio_transport["config_fields"]}
        assert "command" in config_fields
        assert config_fields["command"]["required"] is True
        assert "args" in config_fields
        assert config_fields["args"]["required"] is False
        assert "env" in config_fields
        assert "cwd" in config_fields

    def test_db_server_to_response_masks_oauth_access_token(self):
        """Test API responses mask OAuth access tokens like other auth secrets."""
        server = MCPServer.from_config(
            {
                "name": "oauth_server",
                "managed": "external",
                "transport": "streamable_http",
                "url": "https://example.com/mcp",
                "auth": {
                    "type": "oauth2",
                    "access_token": "plain-access-token",
                    "token_type": "Bearer",
                },
            }
        )
        server.id = 1

        user_mcp = MagicMock()
        user_mcp.user_id = 1
        user_mcp.is_active = True
        user_mcp.is_default = False
        user_mcp.is_owner = True
        user_mcp.env = None
        user_mcp.env_source = None

        response = _db_server_to_response(
            server=server,
            user_mcp=user_mcp,
            manager=MagicMock(),
        )

        assert response.config["auth"]["type"] == "oauth2"
        assert response.config["auth"]["token_type"] == "Bearer"
        assert response.config["auth"]["access_token"] == "********"

    def test_db_server_to_response_masks_env_and_returns_user_env(self):
        """Env values are masked (keys kept); per-user env is returned masked too."""
        server = MCPServer.from_config(
            {
                "name": "envy",
                "managed": "external",
                "transport": "stdio",
                "command": "python",
                "env": {"API_KEY": "global-secret", "REGION": "us"},
            }
        )
        server.id = 1

        user_mcp = MagicMock()
        user_mcp.user_id = 1
        user_mcp.is_active = True
        user_mcp.is_default = False
        user_mcp.is_owner = False
        user_mcp.env = {"API_KEY": "my-secret"}
        user_mcp.env_source = None

        response = _db_server_to_response(
            server=server, user_mcp=user_mcp, manager=MagicMock()
        )

        # Keys visible, values masked
        assert set(response.config["env"]) == {"API_KEY", "REGION"}
        assert response.config["env"]["API_KEY"] == "********"
        assert response.user_env == {"API_KEY": "********"}
        # Non-owner cannot edit the global fallback
        assert response.can_edit_global is False

    def test_get_mcp_servers_projects_custom_api_runtime_configuration(self):
        """The aggregate connector list must not lose Custom API runtime fields."""
        runtime_input_schema = {
            "context": {"account_id": {"type": "string", "required": True}}
        }
        runtime_bindings = [
            {
                "source": {"input_type": "context", "key": "account_id"},
                "target": {"target_type": "headers", "key": "X-Account-ID"},
            }
        ]
        api = CustomApi(
            id=12,
            name="accounts",
            description="Account lookup",
            url="https://api.example.com/accounts",
            method="GET",
            env={"API_KEY": "encrypted"},
            runtime_input_schema=runtime_input_schema,
            runtime_bindings=runtime_bindings,
            allow_delegated_authorization=True,
            created_at=None,
            updated_at=None,
        )
        user_api = UserCustomApi(
            user_id=7,
            custom_api_id=12,
            is_active=True,
            is_default=False,
        )

        def query_result(rows):
            query = MagicMock()
            query.join.return_value = query
            query.filter.return_value = query
            query.order_by.return_value = query
            query.all.return_value = rows
            return query

        db = MagicMock()
        db.query.side_effect = [
            query_result([]),
            query_result([]),
            query_result([(user_api, api)]),
        ]

        responses = get_mcp_servers(current_user=User(id=7), db=db)

        assert len(responses) == 1
        response = responses[0]
        assert response.runtime_input_schema == runtime_input_schema
        assert response.runtime_bindings == runtime_bindings
        assert response.allow_delegated_authorization is True
        assert response.config["env"] == {"API_KEY": "********"}

    def test_merge_masked_env_preserves_stored_secrets(self):
        """Masked values keep the stored secret; real values overwrite; new keys added."""
        merged = _merge_masked_env(
            {"API_KEY": "********", "TOKEN": "new", "EXTRA": "x"},
            {"API_KEY": "old-secret", "TOKEN": "old"},
        )
        assert merged == {"API_KEY": "old-secret", "TOKEN": "new", "EXTRA": "x"}

    def test_merge_masked_env_rejects_masked_key_without_stored_value(self):
        """A mask cannot be moved to a different key identity."""
        with pytest.raises(ValueError, match="NEW"):
            _merge_masked_env({"NEW": "********"}, {"OLD": "secret"})

    def test_check_mcp_permission(self):
        """Owner gates edit; owner/can_delete gates delete; admin bypasses."""
        owner = MagicMock(is_owner=True, can_delete=True)
        guest = MagicMock(is_owner=False, can_delete=False)
        assert _check_mcp_permission(owner, is_admin=False, require="edit") is True
        assert _check_mcp_permission(owner, is_admin=False, require="delete") is True
        assert _check_mcp_permission(guest, is_admin=False, require="edit") is False
        assert _check_mcp_permission(guest, is_admin=False, require="delete") is False
        # Admin bypasses the per-row flags
        assert _check_mcp_permission(guest, is_admin=True, require="delete") is True
        # Regression: an owner whose can_delete was never set (OAuth provisioning,
        # migration-skipped rows) must still be able to delete their own server.
        legacy_owner = MagicMock(is_owner=True, can_delete=False)
        assert _check_mcp_permission(legacy_owner, is_admin=False, require="delete")
        # A non-owner explicitly granted can_delete may delete.
        grantee = MagicMock(is_owner=False, can_delete=True)
        assert _check_mcp_permission(grantee, is_admin=False, require="delete")

    def test_global_config_tampered(self):
        """Non-secret global fields are diffed; unchanged payloads pass."""
        server = MCPServer.from_config(
            {
                "name": "svc",
                "managed": "external",
                "transport": "stdio",
                "command": "python",
                "args": ["-m", "svc"],
            }
        )
        # Same values (what a non-owner's disabled-but-submitted form sends) -> ok
        unchanged = MCPServerUpdate(config={"command": "python", "args": ["-m", "svc"]})
        assert _global_config_tampered(unchanged, server) is False
        # Changed command -> tampered
        assert _global_config_tampered(
            MCPServerUpdate(config={"command": "sh"}), server
        )
        # Changed top-level name -> tampered
        assert _global_config_tampered(MCPServerUpdate(name="other"), server)

    def test_auth_metadata_tampered(self):
        """Non-secret auth metadata is diffed; secrets/masked values are ignored."""
        current = {"type": "oauth2", "client_id": "abc", "client_secret": "enc"}
        # Unchanged metadata (masked secret) -> not tampered
        assert not _auth_metadata_tampered(
            {"type": "oauth2", "client_id": "abc", "client_secret": "********"},
            current,
        )
        # Changed non-secret metadata -> tampered
        assert _auth_metadata_tampered({"client_id": "hijacked"}, current)
        assert _auth_metadata_tampered({"issuer": "https://evil"}, current)
        # Only a (masked) secret changed -> secrets can't be diffed, not tampered
        assert not _auth_metadata_tampered({"client_secret": "********"}, current)
        assert not _auth_metadata_tampered(None, current)

    def test_mask_env_keeps_keys(self):
        assert _mask_env({"A": "1", "B": ""}) == {"A": "********", "B": ""}

    def test_env_dict_encryption_roundtrip_and_no_double_encrypt(self):
        """env values encrypt at rest, decrypt back, and never double-encrypt."""
        from xagent.core.utils.encryption import decrypt_env_dict, encrypt_env_dict

        enc = encrypt_env_dict({"API_KEY": "secret", "EMPTY": ""})
        assert enc["API_KEY"].startswith("gAAAAAB")
        assert enc["EMPTY"] == ""  # empty values are not secrets
        # Re-encrypting an already-encrypted value is a no-op
        assert encrypt_env_dict(enc)["API_KEY"] == enc["API_KEY"]
        assert decrypt_env_dict(enc) == {"API_KEY": "secret", "EMPTY": ""}

    def test_build_server_config_parses_mcp_concurrency_config(self):
        """API request config accepts the explicit MCP concurrency opt-in."""
        server_data = MCPServerCreate(
            name="mail",
            transport="stdio",
            description="Mail MCP",
            config={
                "command": "python",
                "concurrency_safe": "true",
                "concurrent_tools": "list_messages, search_messages",
            },
        )

        config = _build_server_config(server_data)

        assert config.concurrency_safe is True
        assert config.concurrent_tools == ["list_messages", "search_messages"]


class TestMCPApiModels:
    """Test MCP API Pydantic models."""

    def test_mcp_server_create_model(self):
        """Test MCPServerCreate model validation."""
        from xagent.web.api.mcp import MCPServerCreate

        # Valid data
        valid_data = {
            "name": "test_server",
            "transport": "stdio",
            "description": "Test server",
            "config": {"command": "echo", "args": ["hello"]},
        }

        server = MCPServerCreate(**valid_data)
        assert server.name == "test_server"
        assert server.transport == "stdio"
        assert server.config == {"command": "echo", "args": ["hello"]}

        # Test required fields
        invalid_data = {"transport": "stdio", "config": {}}

        with pytest.raises(ValueError):
            MCPServerCreate(**invalid_data)

    def test_mcp_server_update_model(self):
        """Test MCPServerUpdate model validation."""
        from xagent.web.api.mcp import MCPServerUpdate

        # Partial update data
        partial_data = {"name": "updated_server", "description": "Updated"}

        server = MCPServerUpdate(**partial_data)
        assert server.name == "updated_server"
        assert server.description == "Updated"

        # Empty update data
        empty_data = {}
        server = MCPServerUpdate(**empty_data)
        assert server.name is None
        assert server.transport is None

    def test_mcp_server_response_model(self):
        """Test MCPServerResponse model."""
        from xagent.web.api.mcp import MCPServerResponse

        response_data = {
            "id": 1,
            "user_id": 1,
            "is_default": True,
            "name": "test_server",
            "transport": "stdio",
            "description": "Test server",
            "is_active": True,
            "config": {
                "command": "python",
                "args": ["server.py"],
            },
            "user_env": None,
            "runtime_input_schema": None,
            "runtime_bindings": None,
            "allow_delegated_authorization": False,
            "can_edit_global": True,
            "transport_display": "STDIO",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        }

        response = MCPServerResponse(**response_data)
        assert response.id == 1
        assert response.name == "test_server"
        assert response.transport == "stdio"
        assert response.is_active is True
        assert response.config["command"] == "python"
        assert response.config["args"] == ["server.py"]

    def test_mcp_server_response_requires_runtime_projection_fields(self):
        """Mapper drift must fail validation instead of silently using defaults."""
        response_data = {
            "id": 1,
            "user_id": 1,
            "is_default": False,
            "name": "test_server",
            "transport": "stdio",
            "description": None,
            "is_active": True,
            "config": {},
            "transport_display": "STDIO",
            "created_at": None,
            "updated_at": None,
        }

        from xagent.web.api.mcp import MCPServerResponse

        with pytest.raises(ValidationError) as exc_info:
            MCPServerResponse(**response_data)

        missing_fields = {error["loc"] for error in exc_info.value.errors()}
        assert missing_fields == {
            ("user_env",),
            ("runtime_input_schema",),
            ("runtime_bindings",),
            ("allow_delegated_authorization",),
            ("can_edit_global",),
        }

    def test_mcp_connection_test_models(self):
        """Test MCP connection test models."""
        from xagent.web.api.mcp import MCPConnectionTest, MCPConnectionTestResponse

        # Test request model
        test_data = {
            "name": "test_connection",
            "transport": "stdio",
            "config": {
                "command": "echo",
            },
        }

        test_request = MCPConnectionTest(**test_data)
        assert test_request.name == "test_connection"
        assert test_request.transport == "stdio"
        assert test_request.config["command"] == "echo"

        # Test response model
        response_data = {
            "success": True,
            "message": "Connection successful",
            "details": {"tool_count": 5},
        }

        response = MCPConnectionTestResponse(**response_data)
        assert response.success is True
        assert response.message == "Connection successful"
        assert response.details == {"tool_count": 5}

        # Test response without details
        minimal_response = {"success": False, "message": "Connection failed"}

        response = MCPConnectionTestResponse(**minimal_response)
        assert response.success is False
        assert response.details is None
