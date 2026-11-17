"""``read_only_tools``: optional per-server allowlist of raw MCP tool names."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from xagent.core.tools.adapters.vibe import mcp_adapter as mcp_adapter_module
from xagent.core.tools.adapters.vibe.mcp_adapter import (
    load_mcp_tools_as_agent_tools,
)
from xagent.core.tools.adapters.vibe.mcp_tool_allowlist import (
    READ_ONLY_TOOLS_KEY,
    filter_tools_by_read_only_allowlist,
    read_only_tool_names,
)
from xagent.core.tools.adapters.vibe.sandboxed_tool import (
    sandboxed_mcp_tool_helper as sandbox_helper,
)


def _tool(name: str):
    return SimpleNamespace(name=name)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (["a", " b ", "", 3], frozenset({"a", "b"})),
        ("a, b,,", frozenset({"a", "b"})),
        ([], frozenset()),
        ({"a": 1}, frozenset()),  # malformed -> fail closed
        (None, frozenset()),  # present null -> fail closed
        (7, frozenset()),
    ],
)
def test_read_only_tool_names(value, expected):
    assert read_only_tool_names({READ_ONLY_TOOLS_KEY: value}) == expected


def test_absent_key_is_unrestricted_but_explicit_null_fails_closed():
    assert read_only_tool_names({}) is None
    assert read_only_tool_names({READ_ONLY_TOOLS_KEY: None}) == frozenset()


def test_absent_key_keeps_every_tool():
    tools = [_tool("read"), _tool("write")]
    assert filter_tools_by_read_only_allowlist(tools, {"transport": "stdio"}) == tools


def test_listed_tools_only_and_empty_list_loads_nothing():
    tools = [_tool("read"), _tool("write")]
    kept = filter_tools_by_read_only_allowlist(tools, {READ_ONLY_TOOLS_KEY: ["read"]})
    assert [t.name for t in kept] == ["read"]
    assert filter_tools_by_read_only_allowlist(tools, {READ_ONLY_TOOLS_KEY: []}) == []


def _direct_tool(name: str):
    return SimpleNamespace(
        name=name,
        description="",
        inputSchema={"type": "object", "properties": {}},
        annotations=None,
        meta=None,
    )


async def _load_direct(monkeypatch, connection):
    class FakeSession:
        async def initialize(self):
            return None

    @asynccontextmanager
    async def fake_create_session(_connection):
        yield FakeSession()

    monkeypatch.setattr(mcp_adapter_module, "create_session", fake_create_session)
    monkeypatch.setattr(
        mcp_adapter_module,
        "load_mcp_tools",
        AsyncMock(return_value=[_direct_tool("query"), _direct_tool("delete_item")]),
    )
    return await load_mcp_tools_as_agent_tools({"db": connection})


@pytest.mark.asyncio
async def test_direct_loader_applies_allowlist(monkeypatch):
    result = await _load_direct(
        monkeypatch,
        {
            "transport": "stdio",
            "command": "python",
            "args": [],
            READ_ONLY_TOOLS_KEY: ["query"],
        },
    )
    assert len(result.tools) == 1
    assert result.tools[0].name.endswith("query")
    assert result.loaded_servers == ("db",)


@pytest.mark.asyncio
async def test_direct_loader_unchanged_without_allowlist(monkeypatch):
    result = await _load_direct(
        monkeypatch, {"transport": "stdio", "command": "python", "args": []}
    )
    assert len(result.tools) == 2


@pytest.mark.asyncio
async def test_sandboxed_loader_applies_allowlist(monkeypatch):
    monkeypatch.setattr(
        sandbox_helper,
        "list_tools_in_sandbox",
        AsyncMock(return_value=[_tool("query"), _tool("delete_item")]),
    )
    monkeypatch.setattr(
        sandbox_helper, "create_sandboxed_tool", AsyncMock(side_effect=lambda t, s: t)
    )
    monkeypatch.setattr(sandbox_helper, "set_instance_sandbox_config", MagicMock())
    built: list[str] = []

    def builder(mcp_tool):
        built.append(mcp_tool.name)
        return mcp_tool

    result = await sandbox_helper.load_sandboxed_mcp_tools(
        {"transport": "stdio", "command": "npx", READ_ONLY_TOOLS_KEY: ["query"]},
        MagicMock(),
        builder,
    )
    assert built == ["query"]  # no adapter is even built for a dropped tool
    assert [t.name for t in result.tools] == ["query"]


@pytest.fixture()
def catalog_db():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from xagent.web.models.database import Base
    from xagent.web.models.public_mcp import PublicMCPApp

    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    db.add(
        PublicMCPApp(
            app_id="cosmos",
            name="Azure Cosmos DB",
            description="d",
            transport="stdio",
            launch_config={
                "command": "npx",
                "args": ["-y", "cosmos-mcp"],
                "required_env": ["COSMOS_KEY"],
                "read_only_tools": ["query_items"],
            },
        )
    )
    db.add(
        PublicMCPApp(
            app_id="open-app",
            name="Open App",
            description="d",
            transport="stdio",
            launch_config={"command": "npx", "args": [], "required_env": []},
        )
    )
    db.commit()
    yield db
    db.close()
    engine.dispose()


def _lookup(db, server):
    from xagent.web.tools.config import WebToolConfig

    return WebToolConfig._catalog_read_only_tools(SimpleNamespace(db=db), server)


def test_api_key_catalog_row_made_by_connect_gets_its_allowlist(catalog_db):
    """The row is built by catalog connect itself, which stamps no app_id."""
    from xagent.web.api.mcp import _ensure_catalog_app_server

    server, _ = _ensure_catalog_app_server(catalog_db, "cosmos")

    assert not (server.auth or {}).get("app_id")
    assert _lookup(catalog_db, server) == ["query_items"]


def test_connected_app_without_declared_allowlist_is_unfiltered(catalog_db):
    from xagent.web.api.mcp import _ensure_catalog_app_server

    server, _ = _ensure_catalog_app_server(catalog_db, "open-app")
    assert _lookup(catalog_db, server) is None


def test_plain_custom_server_stays_unfiltered(catalog_db):
    server = SimpleNamespace(name="my-own-server", auth=None)
    assert _lookup(catalog_db, server) is None
    assert _lookup(catalog_db, SimpleNamespace(name="x", auth={"other": 1})) is None


@pytest.mark.parametrize(
    "auth",
    [
        {"app_id": "deleted-app"},
        {"app_id": ""},
        {"app_id": 5},
        {"builtin_provenance": {"registry": "xagent", "app_id": "deleted-app"}},
        {"builtin_provenance": "garbage"},
    ],
)
def test_row_claiming_an_unresolvable_catalog_identity_fails_closed(catalog_db, auth):
    server = SimpleNamespace(name="cosmos", auth=auth)  # name would match a real app
    assert _lookup(catalog_db, server) == []


def test_stamped_row_resolves_by_id_not_name(catalog_db):
    server = SimpleNamespace(name="open-app", auth={"app_id": "cosmos"})
    assert _lookup(catalog_db, server) == ["query_items"]


def test_malformed_declared_allowlist_fails_closed(catalog_db):
    from xagent.web.models.public_mcp import PublicMCPApp

    row = catalog_db.query(PublicMCPApp).filter_by(app_id="cosmos").one()
    row.launch_config = {**row.launch_config, "read_only_tools": "query_items"}
    catalog_db.commit()
    assert _lookup(catalog_db, SimpleNamespace(name="cosmos", auth=None)) == []


def test_catalog_is_queried_once_per_config_instance(catalog_db):
    from sqlalchemy import event

    from xagent.web.tools.config import WebToolConfig

    selects: list[str] = []
    event.listen(
        catalog_db.get_bind(),
        "before_cursor_execute",
        lambda conn, cur, stmt, *a: selects.append(stmt),
    )
    stub = SimpleNamespace(db=catalog_db)
    for name in ("a", "b", "cosmos", "open-app", "c"):
        WebToolConfig._catalog_read_only_tools(
            stub, SimpleNamespace(name=name, auth=None)
        )
    assert len(selects) == 1


def test_provenance_marker_from_another_registry_is_not_a_claim(catalog_db):
    server = SimpleNamespace(
        name="my-own-server",
        auth={"builtin_provenance": {"registry": "other", "app_id": "gone"}},
    )
    assert _lookup(catalog_db, server) is None


def test_explicit_null_in_launch_config_fails_closed(catalog_db):
    from xagent.web.models.public_mcp import PublicMCPApp

    row = catalog_db.query(PublicMCPApp).filter_by(app_id="cosmos").one()
    row.launch_config = {**row.launch_config, "read_only_tools": None}
    catalog_db.commit()
    assert _lookup(catalog_db, SimpleNamespace(name="cosmos", auth=None)) == []


def test_cached_snapshot_survives_session_close_and_rollback(catalog_db):
    """The snapshot is cached on the config instance; the session behind it can
    be rolled back and closed between builds, which detaches live ORM rows."""
    from xagent.web.tools.config import WebToolConfig

    stub = SimpleNamespace(db=catalog_db)
    custom = SimpleNamespace(name="my-own-server", auth=None)
    connected = SimpleNamespace(name="cosmos", auth=None)
    assert WebToolConfig._catalog_read_only_tools(stub, connected) == ["query_items"]

    catalog_db.rollback()
    catalog_db.close()  # detaches every row loaded through the session

    # A name lookup scans every cached app, so a retained ORM row would raise.
    assert WebToolConfig._catalog_read_only_tools(stub, custom) is None
    assert WebToolConfig._catalog_read_only_tools(stub, connected) == ["query_items"]
