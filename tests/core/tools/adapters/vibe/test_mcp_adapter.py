import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from xagent.core.tools.adapters.vibe.factory import ToolFactory
from xagent.core.tools.adapters.vibe.mcp_adapter import (
    MCPToolAdapter,
    _build_mcp_tool_adapter,
    load_mcp_tools_as_agent_tools,
)
from xagent.core.tools.core.mcp.oauth.errors import MCPReauthorizationRequired


def test_build_mcp_tool_adapter_stamps_normalized_source_server():
    """``_build_mcp_tool_adapter`` carries the originating server identity
    onto ``metadata.source_server``, normalized once via the shared SSOT,
    while the LLM-visible name keeps its original casing. Server-scoped
    selection matches on the structured field, not the tool name."""
    mcp_tool = SimpleNamespace(
        name="send_message",
        description="Send a message",
        inputSchema={"type": "object", "properties": {}},
    )
    adapter = _build_mcp_tool_adapter(
        "Google Drive",
        {"transport": "stdio", "command": "python", "args": []},
        mcp_tool,
    )

    assert adapter.source_server == "google_drive"
    assert adapter.metadata.source_server == "google_drive"
    # LLM-visible name keeps original casing / spacing folded to underscores.
    assert adapter.name == "mcp_Google Drive_send_message".replace(" ", "_")


@asynccontextmanager
async def _reauth_session(_connection):
    raise MCPReauthorizationRequired("notion", 7)
    yield


@pytest.mark.asyncio
async def test_load_mcp_tools_collects_reauthorization_instead_of_raising(monkeypatch):
    """A server whose token can't be used/refreshed must not raise out of
    ``load_mcp_tools_as_agent_tools`` — it is collected in the returned
    ``reauth_failures`` list so other, healthy servers in the same batch
    still get their tools returned (see the two-server test below)."""
    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.create_session",
        _reauth_session,
    )

    tools, reauth_failures = await load_mcp_tools_as_agent_tools(
        {"notion": {"transport": "streamable_http", "url": "https://x"}}
    )

    assert tools == []
    assert len(reauth_failures) == 1
    assert reauth_failures[0].server_name == "notion"
    assert reauth_failures[0].mcpserver_id == 7


@pytest.mark.asyncio
async def test_load_mcp_tools_isolates_reauth_failure_from_healthy_server(monkeypatch):
    """One server needing reauth must not wipe out tools from OTHER, healthy
    servers in the same batch (Finding 2 regression test)."""
    healthy_tool = SimpleNamespace(
        name="search",
        description="search",
        inputSchema={"type": "object", "properties": {}},
    )

    @asynccontextmanager
    async def fake_create_session(connection):
        if connection.get("url") == "https://dead":
            raise MCPReauthorizationRequired("dead_server", 42)

        class _Session:
            async def initialize(self):
                return None

        yield _Session()

    async def fake_load_mcp_tools(_session):
        return [healthy_tool]

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.create_session",
        fake_create_session,
    )
    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools",
        fake_load_mcp_tools,
    )

    tools, reauth_failures = await load_mcp_tools_as_agent_tools(
        {
            "healthy_server": {"transport": "streamable_http", "url": "https://ok"},
            "dead_server": {"transport": "streamable_http", "url": "https://dead"},
        }
    )

    assert len(tools) == 1
    assert tools[0].mcp_tool.name == "search"
    assert len(reauth_failures) == 1
    assert reauth_failures[0].server_name == "dead_server"
    assert reauth_failures[0].mcpserver_id == 42


@pytest.mark.asyncio
async def test_factory_mcp_config_loader_calls_reauth_hook_and_returns_tools(
    monkeypatch,
):
    """The batch caller (``_create_mcp_tools_from_configs``) must not raise
    when a server needs reauth — it should invoke the owning config's
    ``on_mcp_reauthorization_required`` hook and still return whatever tools
    it did load."""
    healthy_tool = SimpleNamespace(name="ok_tool")

    async def fake_load_mcp_tools_as_agent_tools(*args, **kwargs):
        return [healthy_tool], [MCPReauthorizationRequired("notion", 7)]

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools",
        fake_load_mcp_tools_as_agent_tools,
    )

    hook_calls = []

    class Config:
        async def on_mcp_reauthorization_required(self, mcpserver_id):
            hook_calls.append(mcpserver_id)

    result = await ToolFactory._create_mcp_tools_from_configs(
        [
            {
                "name": "notion",
                "transport": "streamable_http",
                "config": {"url": "https://x"},
            }
        ],
        tool_config=Config(),
    )

    assert result == [healthy_tool]
    assert hook_calls == [7]


@pytest.mark.asyncio
async def test_registered_mcp_creator_propagates_reauthorization(monkeypatch):
    from xagent.core.tools.adapters.vibe import mcp_tools

    class Config:
        def get_tool_selection_spec(self):
            return None

        async def get_mcp_server_configs(self):
            return [{"name": "notion", "transport": "streamable_http", "config": {}}]

        def get_sandbox(self):
            return None

    async def fake_create(*args, **kwargs):
        raise MCPReauthorizationRequired("notion", 7)

    monkeypatch.setattr(
        ToolFactory,
        "_create_mcp_tools_from_configs",
        staticmethod(fake_create),
    )

    with pytest.raises(MCPReauthorizationRequired):
        await mcp_tools.create_mcp_tools(Config())


def test_factory_sync_wrapper_propagates_reauthorization(monkeypatch):
    async def fake_create_mcp_tools(cls, db, user_id=None):
        raise MCPReauthorizationRequired("notion", 7)

    monkeypatch.setattr(
        ToolFactory,
        "create_mcp_tools",
        classmethod(fake_create_mcp_tools),
    )

    # ``_create_mcp_tools`` is a sync wrapper that needs a current event loop
    # in this thread. Whether one exists depends on what earlier tests in
    # this pytest-asyncio process left behind, so set up a fresh loop
    # explicitly rather than relying on ambient state (avoids order-dependent
    # "no current event loop" flakiness).
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        with pytest.raises(MCPReauthorizationRequired):
            ToolFactory._create_mcp_tools(None, 1)
    finally:
        asyncio.set_event_loop(None)
        loop.close()


@pytest.mark.asyncio
async def test_mcp_tool_run_propagates_reauthorization(monkeypatch):
    monkeypatch.setenv("XAGENT_USER_ID", "1")
    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.create_session",
        _reauth_session,
    )
    mcp_tool = SimpleNamespace(
        name="ping",
        description="ping",
        inputSchema={"type": "object", "properties": {}},
    )
    adapter = MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection={"transport": "streamable_http", "url": "https://x"},
    )

    with pytest.raises(MCPReauthorizationRequired):
        await adapter.run_json_async({})


def test_mcp_tool_adapter_source_server_defaults_none():
    """A directly constructed adapter with no server origin reports
    ``source_server`` as ``None`` (no scoped-selection match)."""
    mcp_tool = SimpleNamespace(
        name="ping",
        description="ping",
        inputSchema={"type": "object", "properties": {}},
    )
    adapter = MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection={"transport": "stdio", "command": "python", "args": []},
    )
    assert adapter.source_server is None
    assert adapter.metadata.source_server is None


def test_mcp_tool_adapter_defaults_to_not_concurrency_safe():
    mcp_tool = SimpleNamespace(
        name="list_messages",
        description="List messages",
        inputSchema={"type": "object", "properties": {}},
    )
    adapter = MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection={"transport": "stdio", "command": "python", "args": []},
    )

    assert adapter.metadata.concurrency_safe is False


def test_build_mcp_tool_adapter_marks_all_tools_safe_when_server_opts_in():
    mcp_tool = SimpleNamespace(
        name="list_messages",
        description="List messages",
        inputSchema={"type": "object", "properties": {}},
    )

    adapter = _build_mcp_tool_adapter(
        "mail",
        {"transport": "stdio", "command": "python", "args": []},
        mcp_tool,
        concurrency_safe=True,
    )

    assert adapter.metadata.concurrency_safe is True


def test_build_mcp_tool_adapter_honors_concurrent_tool_allowlist():
    safe_tool = SimpleNamespace(
        name="list_messages",
        description="List messages",
        inputSchema={"type": "object", "properties": {}},
    )
    unsafe_tool = SimpleNamespace(
        name="delete_message",
        description="Delete a message",
        inputSchema={"type": "object", "properties": {}},
    )

    safe_adapter = _build_mcp_tool_adapter(
        "mail",
        {"transport": "stdio", "command": "python", "args": []},
        safe_tool,
        concurrency_safe=True,
        concurrent_tools=["list_messages"],
    )
    unsafe_adapter = _build_mcp_tool_adapter(
        "mail",
        {"transport": "stdio", "command": "python", "args": []},
        unsafe_tool,
        concurrency_safe=True,
        concurrent_tools=["list_messages"],
    )

    assert safe_adapter.metadata.concurrency_safe is True
    assert unsafe_adapter.metadata.concurrency_safe is False


def test_build_args_model_handles_optional_array_schema():
    mcp_tool = SimpleNamespace(
        name="gmail_manage_labels",
        description="Manage Gmail labels",
        inputSchema={
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "add_label_ids": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "string"}},
                        {"type": "null"},
                    ],
                    "default": None,
                },
            },
            "required": ["action"],
        },
    )
    adapter = MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection={"transport": "stdio", "command": "python", "args": []},
    )

    args_model = adapter.args_type()
    parsed = args_model(action="modify_message", add_label_ids=["TRASH"])

    assert parsed.add_label_ids == ["TRASH"]


def test_normalize_args_by_schema_wraps_scalar_for_array_only_field():
    mcp_tool = SimpleNamespace(
        name="gmail_manage_labels",
        description="Manage Gmail labels",
        inputSchema={
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "add_label_ids": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "string"}},
                        {"type": "null"},
                    ],
                    "default": None,
                },
            },
            "required": ["action"],
        },
    )
    adapter = MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection={"transport": "stdio", "command": "python", "args": []},
    )

    normalized = adapter._normalize_args_by_schema(
        {"action": "modify_message", "add_label_ids": "TRASH"}
    )

    assert normalized["add_label_ids"] == ["TRASH"]


def test_normalize_args_by_schema_keeps_scalar_for_union_scalar_or_array_field():
    mcp_tool = SimpleNamespace(
        name="multi_shape_tool",
        description="Accept string or string array input",
        inputSchema={
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ]
                }
            },
            "required": ["value"],
        },
    )
    adapter = MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection={"transport": "stdio", "command": "python", "args": []},
    )

    normalized = adapter._normalize_args_by_schema({"value": "abc"})

    assert normalized["value"] == "abc"


def test_build_args_model_handles_anyof_multi_type_schema():
    mcp_tool = SimpleNamespace(
        name="multi_type_tool",
        description="Accept string or integer input",
        inputSchema={
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "integer"},
                    ]
                }
            },
            "required": ["value"],
        },
    )
    adapter = MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection={"transport": "stdio", "command": "python", "args": []},
    )

    args_model = adapter.args_type()

    assert args_model(value="abc").value == "abc"
    assert args_model(value=123).value == 123


def test_build_args_model_handles_multi_value_type_list():
    mcp_tool = SimpleNamespace(
        name="multi_value_type_tool",
        description="Accept string or integer input",
        inputSchema={
            "type": "object",
            "properties": {"value": {"type": ["string", "integer", "null"]}},
            "required": ["value"],
        },
    )
    adapter = MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection={"transport": "stdio", "command": "python", "args": []},
    )

    args_model = adapter.args_type()

    assert args_model(value="abc").value == "abc"
    assert args_model(value=123).value == 123
