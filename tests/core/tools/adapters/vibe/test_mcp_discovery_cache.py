"""Cross-build cache of stdio MCP tool discovery (#3926).

A stdio server's tool list is learned by spawning its process and running
``initialize`` + ``tools/list``; tool calls never reuse that process. The
cache lets a later agent build skip the spawn when the server is launched the
same way, without ever sharing a listing across credentials.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pytest
from mcp.types import Tool as MCPTool

from xagent.config import MCP_TOOL_DISCOVERY_CACHE_SECONDS
from xagent.core.tools.adapters.vibe import mcp_adapter as mcp_adapter_module
from xagent.core.tools.adapters.vibe import mcp_discovery_cache as cache_module
from xagent.core.tools.adapters.vibe.mcp_adapter import (
    MCPFailurePhase,
    _load_direct_mcp_tools,
    load_mcp_tools_as_agent_tools,
)
from xagent.core.tools.adapters.vibe.mcp_discovery_cache import (
    cached_discovery,
    discovery_key,
    remember_discovery,
)
from xagent.core.tools.core.mcp import sessions as sessions_module
from xagent.core.tools.core.mcp.tools import (
    attach_raw_annotations,
    raw_annotations_for,
)


def _stdio(token: str = "tok-a", **overrides: Any) -> dict[str, Any]:
    connection: dict[str, Any] = {
        "transport": "stdio",
        "command": "python",
        "args": ["-m", "example.server"],
        "env": {
            "EXAMPLE_ACCESS_TOKEN": token,
            "XAGENT_GMAIL_FILE_ALLOWED_DIRS": '["/tasks/1/input"]',
            "XAGENT_GOOGLE_DRIVE_OUTPUT_DIR": "/tasks/1/output",
        },
    }
    connection.update(overrides)
    return connection


def _key(connection: dict[str, Any]) -> str:
    key = discovery_key(connection)
    assert key is not None
    return key


def _tool(name: str = "search", raw: dict[str, Any] | None = None) -> MCPTool:
    tool = MCPTool(
        name=name,
        description=f"{name} things",
        inputSchema={"type": "object", "properties": {"q": {"type": "string"}}},
    )
    return attach_raw_annotations(tool, raw)


def test_only_stdio_connections_have_a_key() -> None:
    assert discovery_key(_stdio()) is not None
    assert discovery_key({"transport": "streamable_http", "url": "https://x"}) is None
    assert discovery_key({"transport": "sse", "url": "https://x"}) is None
    assert discovery_key({"transport": "stdio"}) is None  # no command


def test_the_same_launch_gives_the_same_key() -> None:
    assert discovery_key(_stdio()) == discovery_key(_stdio())


def test_credentials_are_part_of_the_key() -> None:
    # A listing may depend on the account; it is never shared across tokens.
    assert discovery_key(_stdio("tok-a")) != discovery_key(_stdio("tok-b"))


@pytest.mark.parametrize(
    "overrides",
    [
        {"command": "python3"},
        {"args": ["-m", "example.other"]},
        {"cwd": "/srv/other"},
        {"encoding": "latin-1"},
    ],
)
def test_every_launch_field_is_part_of_the_key(overrides: dict[str, Any]) -> None:
    assert discovery_key(_stdio(**overrides)) != discovery_key(_stdio())


def test_an_env_without_path_keys_like_one_naming_the_process_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The launcher adds the process PATH when the env names none, so the child
    # gets the same env either way.
    monkeypatch.setenv("PATH", "/fixed/bin")
    bare = {"EXAMPLE_ACCESS_TOKEN": "tok-a"}
    assert _key(_stdio(env=bare)) == _key(_stdio(env={**bare, "PATH": "/fixed/bin"}))


def test_an_empty_or_missing_env_keys_like_the_process_path_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "/fixed/bin")
    named = _key(_stdio(env={"PATH": "/fixed/bin"}))
    assert _key(_stdio(env={})) == named
    assert _key(_stdio(env=None)) == named


def test_an_explicit_empty_path_keys_differently_from_an_absent_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "/fixed/bin")
    bare = {"EXAMPLE_ACCESS_TOKEN": "tok-a"}
    assert _key(_stdio(env={**bare, "PATH": ""})) != _key(_stdio(env=bare))


def test_the_process_path_is_part_of_the_key_of_an_env_without_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bare = {"EXAMPLE_ACCESS_TOKEN": "tok-a"}
    monkeypatch.setenv("PATH", "/fixed/bin")
    first = _key(_stdio(env=bare))
    monkeypatch.setenv("PATH", "/other/bin")
    assert _key(_stdio(env=bare)) != first


def test_computing_the_key_leaves_the_callers_env_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "/fixed/bin")
    env = {"EXAMPLE_ACCESS_TOKEN": "tok-a"}
    _key(_stdio(env=env))
    assert env == {"EXAMPLE_ACCESS_TOKEN": "tok-a"}


def test_a_non_task_env_value_is_part_of_the_key() -> None:
    changed = _stdio()
    changed["env"] = {**changed["env"], "EXAMPLE_WRITE_ENABLED": "true"}
    assert discovery_key(changed) != discovery_key(_stdio())


def test_task_file_directory_values_do_not_change_the_key() -> None:
    other_task = _stdio()
    other_task["env"] = {
        **other_task["env"],
        "XAGENT_GMAIL_FILE_ALLOWED_DIRS": '["/tasks/2/input"]',
        "XAGENT_GOOGLE_DRIVE_OUTPUT_DIR": "/tasks/2/output",
    }
    assert discovery_key(other_task) == discovery_key(_stdio())


def test_the_presence_of_a_task_file_directory_is_part_of_the_key() -> None:
    without = _stdio()
    without["env"] = {
        key: value
        for key, value in without["env"].items()
        if key != "XAGENT_GOOGLE_DRIVE_OUTPUT_DIR"
    }
    assert discovery_key(without) != discovery_key(_stdio())


def test_an_unlisted_directory_name_keys_by_value() -> None:
    # Only the names the host writes per task are normalised; a look-alike
    # name a server operator set keeps its value in the key.
    first = _stdio()
    first["env"] = {**first["env"], "XAGENT_CUSTOM_ALLOWED_DIRS": "/srv/a"}
    second = _stdio()
    second["env"] = {**second["env"], "XAGENT_CUSTOM_ALLOWED_DIRS": "/srv/b"}
    assert discovery_key(first) != discovery_key(second)


def test_every_task_directory_the_host_writes_is_listed() -> None:
    # The host's per-task env names, read from where they are written. A name
    # missing from the cache's list would make every task miss the cache.
    import re
    from pathlib import Path

    import xagent.config

    source = (
        Path(xagent.config.__file__).parent / "web" / "tools" / "config.py"
    ).read_text()
    written = set(
        re.findall(r'env\["(XAGENT_[A-Z0-9_]+_(?:ALLOWED_DIRS|OUTPUT_DIR))"\]', source)
    )

    assert written
    assert written <= cache_module._TASK_FILE_DIRECTORY_ENV


def test_a_launch_that_cannot_be_digested_has_no_key() -> None:
    assert discovery_key(_stdio(args=[object()])) is None


def test_session_kwargs_leave_the_launch_without_a_key() -> None:
    # They can change the client's own ``initialize``, which the key does not see.
    assert discovery_key(_stdio(session_kwargs={"read_timeout_seconds": 5})) is None


def test_fields_the_session_does_not_launch_with_do_not_change_the_key() -> None:
    # Read-only allowlists, connector runtime context and the like are applied
    # to the listing on every build, after the cache.
    decorated = _stdio(read_only_tools=["search"], concurrency_safe=True)
    assert discovery_key(decorated) == discovery_key(_stdio())


def test_a_remembered_listing_comes_back_as_independent_copies() -> None:
    key = discovery_key(_stdio())
    assert key is not None
    original = [_tool("search", {"readOnlyHint": True}), _tool("create")]

    remember_discovery(key, original)
    first = cached_discovery(key)
    assert first is not None
    first[0].inputSchema["properties"]["q"]["type"] = "integer"
    original[1].description = "changed after it was remembered"
    second = cached_discovery(key)

    assert second is not None
    assert [tool.name for tool in second] == ["search", "create"]
    assert second[0].inputSchema["properties"]["q"]["type"] == "string"
    assert second[1].description == "create things"
    assert second[0] is not original[0]


def test_wire_annotations_survive_the_cache() -> None:
    # The approval gate and the read-only classification read these.
    key = discovery_key(_stdio())
    assert key is not None
    remember_discovery(
        key, [_tool("search", {"readOnlyHint": True}), _tool("create", None)]
    )

    tools = cached_discovery(key)

    assert tools is not None
    assert raw_annotations_for(tools[0]) == {"readOnlyHint": True}
    assert raw_annotations_for(tools[1]) is None


def test_an_empty_listing_is_never_remembered() -> None:
    key = discovery_key(_stdio())
    assert key is not None
    remember_discovery(key, [])
    assert cached_discovery(key) is None


def test_an_entry_expires_after_its_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MCP_TOOL_DISCOVERY_CACHE_SECONDS, "10")
    now = [1000.0]
    monkeypatch.setattr(cache_module, "_clock", lambda: now[0])
    key = discovery_key(_stdio())
    assert key is not None
    remember_discovery(key, [_tool()])

    now[0] += 10
    assert cached_discovery(key) is not None
    now[0] += 0.5
    assert cached_discovery(key) is None


def test_zero_disables_the_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    key = _key(_stdio())
    remember_discovery(key, [_tool("search")])

    monkeypatch.setenv(MCP_TOOL_DISCOVERY_CACHE_SECONDS, "0")
    assert cached_discovery(key) is None
    remember_discovery(key, [_tool("other")])

    monkeypatch.delenv(MCP_TOOL_DISCOVERY_CACHE_SECONDS)
    # Nothing was stored while it was off.
    tools = cached_discovery(key)
    assert tools is not None
    assert [tool.name for tool in tools] == ["search"]


def test_remembering_drops_expired_listings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MCP_TOOL_DISCOVERY_CACHE_SECONDS, "10")
    now = [1000.0]
    monkeypatch.setattr(cache_module, "_clock", lambda: now[0])
    remember_discovery(_key(_stdio("tok-old")), [_tool()])

    now[0] += 11
    remember_discovery(_key(_stdio("tok-new")), [_tool()])

    assert list(cache_module._entries) == [_key(_stdio("tok-new"))]


def test_a_broken_cache_never_fails_the_load(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("broken")

    monkeypatch.setattr(cache_module, "_copy_with_annotations", broken)
    key = _key(_stdio())

    remember_discovery(key, [_tool()])
    assert cached_discovery(key) is None
    monkeypatch.setattr(cache_module, "_launch_digest", broken)
    assert discovery_key(_stdio()) is None


def test_the_cache_keeps_a_bounded_number_of_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cache_module, "_MAX_ENTRIES", 2)
    keys = [_key(_stdio(f"tok-{i}")) for i in range(3)]
    for key in keys:
        remember_discovery(key, [_tool()])

    assert cached_discovery(keys[0]) is None
    assert cached_discovery(keys[1]) is not None
    assert cached_discovery(keys[2]) is not None


class _Sessions:
    """Fake ``create_session``: counts the sessions opened and lists ``tools``."""

    def __init__(self, tools: list[MCPTool] | None = None, fail: bool = False):
        self.opened: list[dict[str, Any]] = []
        self.tools = tools if tools is not None else [_tool("search")]
        self.fail = fail

    def __call__(self, connection: dict[str, Any]) -> Any:
        sessions = self

        @asynccontextmanager
        async def session():
            sessions.opened.append(connection)
            if sessions.fail:
                raise ConnectionError("server down")

            class _Session:
                async def initialize(self) -> None:
                    return None

            yield _Session()

        return session()

    async def list_tools(self, session: Any) -> list[MCPTool]:
        return [tool.model_copy(deep=True) for tool in self.tools]


@pytest.fixture
def sessions(monkeypatch: pytest.MonkeyPatch) -> _Sessions:
    fake = _Sessions()
    monkeypatch.setattr(mcp_adapter_module, "create_session", fake)
    monkeypatch.setattr(mcp_adapter_module, "load_mcp_tools", fake.list_tools)
    return fake


async def _load(connection: dict[str, Any], *, discovery_cache: bool) -> Any:
    return await _load_direct_mcp_tools(
        "example",
        connection,
        name_prefix="mcp_",
        visibility=None,
        allow_users=None,
        discovery_cache=discovery_cache,
    )


@pytest.mark.asyncio
async def test_a_second_build_reuses_the_listing_without_spawning(
    sessions: _Sessions,
) -> None:
    first = await _load(_stdio(), discovery_cache=True)
    second = await _load(_stdio(), discovery_cache=True)

    assert len(sessions.opened) == 1
    assert [tool.name for tool in first.tools] == [tool.name for tool in second.tools]
    assert second.loaded_servers == first.loaded_servers
    assert second.failures == ()


@pytest.mark.asyncio
async def test_a_build_for_another_task_reuses_the_listing(sessions: _Sessions) -> None:
    other_task = _stdio()
    other_task["env"] = {
        **other_task["env"],
        "XAGENT_GOOGLE_DRIVE_OUTPUT_DIR": "/tasks/2/output",
    }

    await _load(_stdio(), discovery_cache=True)
    await _load(other_task, discovery_cache=True)

    assert len(sessions.opened) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changed",
    [
        _stdio("tok-b"),
        _stdio(command="python3"),
        _stdio(args=["-m", "example.other"]),
        _stdio(env={"EXAMPLE_ACCESS_TOKEN": "tok-a", "EXAMPLE_WRITE_ENABLED": "1"}),
    ],
    ids=["credentials", "command", "args", "env"],
)
async def test_another_launch_discovers_again(
    sessions: _Sessions, changed: dict[str, Any]
) -> None:
    await _load(_stdio(), discovery_cache=True)
    await _load(changed, discovery_cache=True)

    assert len(sessions.opened) == 2


@pytest.mark.asyncio
async def test_without_the_flag_every_build_discovers(sessions: _Sessions) -> None:
    # Connection tests and connector checks must reach the live server.
    await _load(_stdio(), discovery_cache=False)
    await _load(_stdio(), discovery_cache=False)
    await _load(_stdio(), discovery_cache=True)

    assert len(sessions.opened) == 3


@pytest.mark.asyncio
async def test_remote_servers_are_never_cached(sessions: _Sessions) -> None:
    remote = {"transport": "streamable_http", "url": "https://mcp.example.com/mcp"}

    await _load(remote, discovery_cache=True)
    await _load(remote, discovery_cache=True)

    assert len(sessions.opened) == 2


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_a_failed_discovery_is_not_cached(sessions: _Sessions) -> None:
    sessions.fail = True
    failed = await _load(_stdio(), discovery_cache=True)
    assert failed.failures[0].phase is MCPFailurePhase.SESSION_START
    opened_by_failure = len(sessions.opened)

    sessions.fail = False
    recovered = await _load(_stdio(), discovery_cache=True)

    assert len(sessions.opened) == opened_by_failure + 1
    assert [tool.name for tool in recovered.tools] == ["mcp_example_search"]


@pytest.mark.asyncio
async def test_an_empty_discovery_is_not_cached(sessions: _Sessions) -> None:
    sessions.tools = []
    empty = await _load(_stdio(), discovery_cache=True)
    assert empty.failures[0].phase is MCPFailurePhase.NO_TOOLS_RETURNED

    sessions.tools = [_tool("search")]
    await _load(_stdio(), discovery_cache=True)

    assert len(sessions.opened) == 2


@pytest.mark.asyncio
async def test_the_read_only_allowlist_still_applies_to_a_cached_listing(
    sessions: _Sessions,
) -> None:
    sessions.tools = [_tool("search"), _tool("delete")]
    await _load(_stdio(), discovery_cache=True)

    restricted = await _load(_stdio(read_only_tools=["search"]), discovery_cache=True)

    assert len(sessions.opened) == 1
    assert [tool.name for tool in restricted.tools] == ["mcp_example_search"]


@pytest.mark.asyncio
async def test_the_public_loader_does_not_cache_unless_asked(
    sessions: _Sessions,
) -> None:
    await load_mcp_tools_as_agent_tools({"example": _stdio()})
    await load_mcp_tools_as_agent_tools({"example": _stdio()})
    assert len(sessions.opened) == 2

    await load_mcp_tools_as_agent_tools({"example": _stdio()}, discovery_cache=True)
    await load_mcp_tools_as_agent_tools({"example": _stdio()}, discovery_cache=True)
    assert len(sessions.opened) == 3


class _FakeClientSession:
    def __init__(self, read: Any, write: Any, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> _FakeClientSession:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        return None

    async def initialize(self) -> None:
        return None


@pytest.fixture
def launched(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, str]]:
    """The real stdio launcher, minus the process: the envs it hands the SDK."""
    envs: list[dict[str, str]] = []

    @asynccontextmanager
    async def fake_stdio_client(server_params: Any):
        envs.append(dict(server_params.env or {}))
        yield None, None

    async def fake_load_mcp_tools(session: Any) -> list[MCPTool]:
        return [_tool("search")]

    monkeypatch.setattr(sessions_module, "stdio_client", fake_stdio_client)
    monkeypatch.setattr(sessions_module, "ClientSession", _FakeClientSession)
    monkeypatch.setattr(mcp_adapter_module, "load_mcp_tools", fake_load_mcp_tools)
    return envs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "env",
    [
        None,
        {},
        {"EXAMPLE_ACCESS_TOKEN": "tok-a"},
        {"EXAMPLE_ACCESS_TOKEN": "tok-a", "PATH": ""},
        {"EXAMPLE_ACCESS_TOKEN": "tok-a", "PATH": "/elsewhere/bin"},
    ],
)
async def test_the_key_is_that_of_the_env_the_launcher_sends(
    monkeypatch: pytest.MonkeyPatch,
    launched: list[dict[str, str]],
    env: dict[str, str] | None,
) -> None:
    # The cache repeats the launcher's PATH default; this keeps the two in step.
    monkeypatch.setenv("PATH", "/fixed/bin")
    key = _key(_stdio(env=env))

    # A copy: the launcher may write into the mapping it is given.
    async with sessions_module.create_session(
        _stdio(env=dict(env) if env is not None else None)
    ):
        pass

    assert _key(_stdio(env=launched[0])) == key


@pytest.mark.asyncio
async def test_the_launchers_path_write_does_not_change_the_key(
    launched: list[dict[str, str]],
) -> None:
    # The real launcher writes PATH into the caller's env after the key was
    # computed; a second build with the same mapping must still hit the cache.
    connection = _stdio()
    assert "PATH" not in connection["env"]

    await load_mcp_tools_as_agent_tools({"example": connection}, discovery_cache=True)
    # The launcher's write into the caller's mapping: the case under test.
    assert "PATH" in connection["env"]
    await load_mcp_tools_as_agent_tools({"example": connection}, discovery_cache=True)

    assert len(launched) == 1


@pytest.mark.asyncio
async def test_sandboxed_stdio_servers_are_never_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandboxed_loads: list[dict[str, Any]] = []

    async def fake_ensure(sandbox: Any) -> None:
        return None

    async def fake_load_sandboxed(
        connection: Any, sandbox: Any, tool_builder: Any
    ) -> Any:
        sandboxed_loads.append(connection)
        return mcp_adapter_module.SandboxedMCPLoadResult(
            tools=(object(),), adapter_error_types=(), wrap_error_types=()
        )

    monkeypatch.setattr(
        mcp_adapter_module, "ensure_sandbox_mcp_dependencies", fake_ensure
    )
    monkeypatch.setattr(
        mcp_adapter_module, "load_sandboxed_mcp_tools", fake_load_sandboxed
    )
    monkeypatch.setattr(
        mcp_adapter_module, "should_sandbox_mcp_connection", lambda c: True
    )

    for _ in range(2):
        await load_mcp_tools_as_agent_tools(
            {"example": _stdio()}, sandbox=object(), discovery_cache=True
        )

    assert len(sandboxed_loads) == 2


@pytest.mark.asyncio
async def test_agent_builds_ask_the_loader_for_the_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.adapters.vibe.factory import ToolFactory

    asked: list[dict[str, Any]] = []

    async def fake_load(connections: Any, **kwargs: Any) -> Any:
        asked.append(kwargs)
        return mcp_adapter_module.MCPLoadResult(
            tools=(), loaded_servers=(), failures=()
        )

    monkeypatch.setattr(mcp_adapter_module, "load_mcp_tools_as_agent_tools", fake_load)

    await ToolFactory._create_mcp_tools_from_configs(
        [
            {
                "name": "example",
                "transport": "stdio",
                "config": {"command": "python", "args": ["-m", "example"]},
            }
        ]
    )

    assert asked and asked[0]["discovery_cache"] is True
