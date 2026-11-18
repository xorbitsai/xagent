"""``read_only_tools``: optional per-app allowlist of raw MCP tool names."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from xagent.core.tools.adapters.vibe import mcp_adapter as mcp_adapter_module
from xagent.core.tools.adapters.vibe.mcp_adapter import load_mcp_tools_as_agent_tools
from xagent.core.tools.adapters.vibe.mcp_tool_allowlist import (
    READ_ONLY_TOOLS_KEY,
    filter_tools_by_read_only_allowlist,
    is_valid_read_only_tools,
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
        (["a", "b"], frozenset({"a", "b"})),
        ([], frozenset()),
        (["a", ""], frozenset()),  # blank item: malformed, fails closed
        (["a", None], frozenset()),  # [null] must not allow a tool named "None"
        (["a", 3], frozenset()),
        ("a,b", frozenset()),  # no comma-string form
        (("a",), frozenset()),
        ({"a": 1}, frozenset()),
        (None, frozenset()),  # present null fails closed
        (7, frozenset()),
    ],
)
def test_read_only_tool_names(value, expected):
    assert read_only_tool_names({READ_ONLY_TOOLS_KEY: value}) == expected
    assert is_valid_read_only_tools(value) is (value == ["a", "b"] or value == [])


def test_absent_key_is_unrestricted():
    assert read_only_tool_names({}) is None


def test_filter_keeps_listed_tools_only():
    tools = [_tool("read"), _tool("write")]
    assert filter_tools_by_read_only_allowlist(tools, {}) == tools
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


def _patch_direct(monkeypatch, names=("query", "delete_item")):
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
        AsyncMock(return_value=[_direct_tool(n) for n in names]),
    )


async def _load_direct(connection):
    return await load_mcp_tools_as_agent_tools({"db": connection})


_STDIO = {"transport": "stdio", "command": "python", "args": []}


@pytest.mark.asyncio
async def test_direct_loader_applies_allowlist(monkeypatch):
    _patch_direct(monkeypatch)
    result = await _load_direct({**_STDIO, READ_ONLY_TOOLS_KEY: ["query"]})
    assert len(result.tools) == 1 and result.tools[0].name.endswith("query")
    assert result.loaded_servers == ("db",)


@pytest.mark.asyncio
async def test_direct_loader_unchanged_without_allowlist(monkeypatch):
    _patch_direct(monkeypatch)
    assert len((await _load_direct(dict(_STDIO))).tools) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("declared", [[], None, "query", ["query", None]])
async def test_both_loaders_load_nothing_for_empty_or_malformed(monkeypatch, declared):
    _patch_direct(monkeypatch)
    direct = await _load_direct({**_STDIO, READ_ONLY_TOOLS_KEY: declared})
    assert direct.tools == ()

    monkeypatch.setattr(
        sandbox_helper,
        "list_tools_in_sandbox",
        AsyncMock(return_value=[_tool("query"), _tool("delete_item")]),
    )
    sandboxed = await sandbox_helper.load_sandboxed_mcp_tools(
        {"transport": "stdio", "command": "npx", READ_ONLY_TOOLS_KEY: declared},
        MagicMock(),
        lambda t: t,
    )
    assert sandboxed.tools == ()


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


@pytest.mark.asyncio
async def test_execution_scoped_chrome_loader_applies_allowlist(monkeypatch):
    pool = SimpleNamespace(
        get_or_create=AsyncMock(return_value=SimpleNamespace(sandbox=object())),
        close_shielded=AsyncMock(),
    )
    monkeypatch.setattr(
        "xagent.web.services.chrome_mcp_runtime.get_chrome_execution_session_pool",
        lambda: pool,
    )
    monkeypatch.setattr(
        mcp_adapter_module.ChromeDaemonLaunchSpec,
        "from_connection",
        staticmethod(lambda connection: object()),
    )
    monkeypatch.setattr(
        mcp_adapter_module, "chrome_metadata_connection", lambda connection: connection
    )
    monkeypatch.setattr(
        mcp_adapter_module,
        "list_tools_in_sandbox",
        AsyncMock(return_value=[_tool("snapshot"), _tool("evaluate_script")]),
    )
    built: list[str] = []

    def fake_adapter(server_name, connection, mcp_tool, **kwargs):
        built.append(mcp_tool.name)
        return mcp_tool

    monkeypatch.setattr(
        mcp_adapter_module, "_build_execution_scoped_chrome_tool_adapter", fake_adapter
    )

    result = await mcp_adapter_module.load_execution_scoped_chrome_tools(
        "chrome",
        {"transport": "stdio", "command": "x", READ_ONLY_TOOLS_KEY: ["snapshot"]},
        scope=object(),  # type: ignore[arg-type]
    )
    assert built == ["snapshot"] and len(result.tools) == 1


@pytest.mark.asyncio
async def test_binding_is_dropped_when_its_issuer_is_not_allowlisted(
    monkeypatch, caplog
):
    from tests.core.tools.adapters.vibe.test_signed_upload_settings import (
        CONNECTION_URL,
        _entry,
    )

    _patch_direct(monkeypatch, names=("attach_document", "request_upload", "other"))
    seen: dict[str, Any] = {}

    def fake_build(server_name, connection, mcp_tool, **kwargs):
        seen[mcp_tool.name] = kwargs.get("signed_upload")
        return SimpleNamespace(name=f"mcp_{mcp_tool.name}")

    monkeypatch.setattr(mcp_adapter_module, "_build_mcp_tool_adapter", fake_build)
    connection = {
        "transport": "streamable_http",
        "url": CONNECTION_URL,
        "signed_uploads": [_entry()],
    }
    kept = await _load_direct({**connection, READ_ONLY_TOOLS_KEY: ["attach_document"]})
    assert [t.name for t in kept.tools] == ["mcp_attach_document"]
    assert seen == {"attach_document": None}  # issuer not allowlisted: no binding
    assert "does not list the issuing tool" not in caplog.text  # it is listed

    seen.clear()
    both = await _load_direct(
        {**connection, READ_ONLY_TOOLS_KEY: ["attach_document", "request_upload"]}
    )
    assert seen["attach_document"] is not None  # issuer allowlisted: binding kept
    assert len(both.tools) == 1  # the issuer itself is never shown to the model


# ---- catalog lookup: rows are built the way catalog connect builds them ----


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
    for app_id, name, extra in [
        ("cosmos", "Azure Cosmos DB", {"read_only_tools": ["query_items"]}),
        ("open-app", "Open App", {}),
    ]:
        db.add(
            PublicMCPApp(
                app_id=app_id,
                name=name,
                description="d",
                transport="stdio",
                launch_config={
                    "command": "npx",
                    "args": ["-y", "pkg"],
                    "required_env": ["KEY"],
                    **extra,
                },
            )
        )
    db.commit()
    yield db
    db.close()
    engine.dispose()


def _cfg(db):
    from xagent.web.tools.config import WebToolConfig

    return WebToolConfig(db=db, request=None, user_id=1)


def _lookup(db, server, cfg=None, resolved_app=None):
    return (cfg or _cfg(db))._catalog_read_only_tools(server, resolved_app)


def _unresolved():
    from xagent.web.tools.config import _CatalogIdentityUnresolved

    return pytest.raises(_CatalogIdentityUnresolved)


def test_api_key_catalog_row_made_by_connect_gets_its_allowlist(catalog_db):
    from xagent.web.api.mcp import _ensure_catalog_app_server

    server, _ = _ensure_catalog_app_server(catalog_db, "cosmos")

    assert not (server.auth or {}).get("app_id")  # connect stamps nothing here
    assert _lookup(catalog_db, server) == {"read_only_tools": ["query_items"]}


def test_connected_app_without_declared_allowlist_is_unfiltered(catalog_db):
    from xagent.web.api.mcp import _ensure_catalog_app_server

    server, _ = _ensure_catalog_app_server(catalog_db, "open-app")
    assert _lookup(catalog_db, server) == {}


def test_allowlist_reaches_the_connection_through_the_builder(catalog_db):
    """The only catalog-to-loader wiring: deleting it must fail a test."""
    from xagent.web.api.mcp import _ensure_catalog_app_server

    def build(app_id):
        server, _ = _ensure_catalog_app_server(catalog_db, app_id)
        return asyncio.run(
            _cfg(catalog_db)._build_mcp_server_config(
                server=server,
                user_env_by_id={},
                shared_env_by_id={},
                env_source_by_id={},
            )
        )["config"]

    assert build("cosmos")["read_only_tools"] == ["query_items"]
    assert "read_only_tools" not in build("open-app")


def test_declared_empty_and_null_reach_the_connection_as_declared(catalog_db):
    from xagent.web.models.public_mcp import PublicMCPApp

    row = catalog_db.query(PublicMCPApp).filter_by(app_id="cosmos").one()
    for declared in ([], None, "query_items"):
        row.launch_config = {**row.launch_config, "read_only_tools": declared}
        catalog_db.commit()
        got = _lookup(catalog_db, SimpleNamespace(name="cosmos", auth=None))
        assert got == {"read_only_tools": declared}
        assert read_only_tool_names(got) == frozenset()  # the loader fails closed


def test_builtin_launch_config_comes_from_the_registry(catalog_db, monkeypatch):
    from xagent.web import mcp_apps

    registry_launch = {
        "command": "npx",
        "args": [],
        "required_env": [],
        "read_only_tools": ["registry_tool"],
    }
    monkeypatch.setattr(
        mcp_apps,
        "get_builtin_execution_fields_and_optional_scopes",
        lambda app_id: (
            {
                "name": "Registry",
                "transport": "stdio",
                "provider_name": None,
                "oauth_scopes": [],
                "launch_config": registry_launch,
            }
            if app_id == "open-app"
            else None,
            [],
        ),
    )
    monkeypatch.setattr(
        mcp_apps, "_persisted_builtin_provenance_matches", lambda *a: True
    )
    # The stored row declares nothing; the registry's declaration governs.
    assert _lookup(catalog_db, SimpleNamespace(name="open-app", auth=None)) == {
        "read_only_tools": ["registry_tool"]
    }


def test_plain_custom_server_stays_unfiltered(catalog_db):
    assert _lookup(catalog_db, SimpleNamespace(name="mine", auth=None)) == {}
    assert _lookup(catalog_db, SimpleNamespace(name="x", auth={"other": 1})) == {}


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
def test_claimed_but_unresolved_identity_is_unresolved(catalog_db, auth):
    with _unresolved():
        _lookup(catalog_db, SimpleNamespace(name="cosmos", auth=auth))


def test_unresolved_claim_is_reported_as_an_unavailable_config(catalog_db):
    from xagent.web.models.mcp import MCPServer

    server = MCPServer(
        id=9,
        name="cosmos",
        description="d",
        managed="external",
        transport="stdio",
        command="npx",
        args=[],
        auth={"app_id": "gone"},
    )
    built = asyncio.run(
        _cfg(catalog_db)._build_mcp_server_config(
            server=server, user_env_by_id={}, shared_env_by_id={}, env_source_by_id={}
        )
    )
    assert built["transport"] == "unavailable"
    assert built["config"]["reason"] == "catalog_app_not_found"


def test_provenance_marker_from_another_registry_is_not_a_claim(catalog_db):
    server = SimpleNamespace(
        name="mine", auth={"builtin_provenance": {"registry": "other", "app_id": "x"}}
    )
    assert _lookup(catalog_db, server) == {}


def test_stamped_row_resolves_by_id_not_name(catalog_db):
    server = SimpleNamespace(name="open-app", auth={"app_id": "cosmos"})
    assert _lookup(catalog_db, server) == {"read_only_tools": ["query_items"]}


def test_resolved_app_from_the_builder_is_used_as_is(catalog_db):
    app = {"launch_config": {"read_only_tools": ["only_this"]}}
    server = SimpleNamespace(name="open-app", auth=None)
    assert _lookup(catalog_db, server, resolved_app=app) == {
        "read_only_tools": ["only_this"]
    }


def test_ambiguous_name_fails_closed_when_a_candidate_declares_an_allowlist(
    catalog_db,
):
    from xagent.web.models.public_mcp import PublicMCPApp

    # App B is named after app A's app_id, so the name "cosmos" owns two apps.
    catalog_db.add(
        PublicMCPApp(
            app_id="other",
            name="cosmos",
            description="d",
            transport="stdio",
            launch_config={"command": "npx", "required_env": ["K"]},
        )
    )
    catalog_db.commit()
    with _unresolved():
        _lookup(catalog_db, SimpleNamespace(name="cosmos", auth=None))


def test_ambiguous_name_without_any_declaration_stays_unfiltered(catalog_db):
    from xagent.web.models.public_mcp import PublicMCPApp

    catalog_db.add(
        PublicMCPApp(
            app_id="other",
            name="open-app",
            description="d",
            transport="stdio",
            launch_config={"command": "npx", "required_env": ["K"]},
        )
    )
    catalog_db.commit()
    assert _lookup(catalog_db, SimpleNamespace(name="open-app", auth=None)) == {}


def test_snapshot_survives_session_close_and_rollback(catalog_db):
    cfg = _cfg(catalog_db)
    connected = SimpleNamespace(name="cosmos", auth=None)
    assert _lookup(catalog_db, connected, cfg) == {"read_only_tools": ["query_items"]}

    catalog_db.rollback()
    catalog_db.close()  # detaches every ORM row loaded through the session

    # A name lookup scans every cached app, so a retained ORM row would raise.
    assert _lookup(catalog_db, SimpleNamespace(name="mine", auth=None), cfg) == {}
    assert _lookup(catalog_db, connected, cfg) == {"read_only_tools": ["query_items"]}


def test_changed_allowlist_is_seen_on_the_next_load_pass(catalog_db):
    from xagent.web.models.public_mcp import PublicMCPApp

    cfg = _cfg(catalog_db)
    server = SimpleNamespace(name="cosmos", auth=None)
    assert _lookup(catalog_db, server, cfg) == {"read_only_tools": ["query_items"]}

    row = catalog_db.query(PublicMCPApp).filter_by(app_id="cosmos").one()
    row.launch_config = {**row.launch_config, "read_only_tools": []}
    catalog_db.commit()
    # Within a pass the projection is reused (one query per pass)...
    assert _lookup(catalog_db, server, cfg) == {"read_only_tools": ["query_items"]}
    # ...and every load pass starts by resetting it.
    cfg._reset_mcp_config_load_cache_state()
    assert _lookup(catalog_db, server, cfg) == {"read_only_tools": []}

    catalog_db.delete(row)
    catalog_db.commit()
    cfg._reset_mcp_config_load_cache_state()
    with _unresolved():
        _lookup(catalog_db, SimpleNamespace(name="x", auth={"app_id": "cosmos"}), cfg)


def test_catalog_is_queried_once_per_pass(catalog_db):
    from sqlalchemy import event

    selects: list[str] = []
    event.listen(
        catalog_db.get_bind(),
        "before_cursor_execute",
        lambda conn, cur, stmt, *a: selects.append(stmt),
    )
    cfg = _cfg(catalog_db)
    for name in ("a", "b", "cosmos", "open-app", "c"):
        _lookup(catalog_db, SimpleNamespace(name=name, auth=None), cfg)
    assert len(selects) == 1


def test_without_a_database_there_is_no_catalog_to_consult():
    from xagent.web.tools.config import WebToolConfig

    cfg = WebToolConfig(db=None, request=None, user_id=1)
    assert cfg._catalog_read_only_tools(SimpleNamespace(name="x", auth=None)) == {}


# ---- the actor stdio config carries the declaration -------------------------


@pytest.mark.parametrize("declared", [None, ["snapshot"]], ids=["absent", "declared"])
def test_actor_stdio_config_carries_the_declaration(monkeypatch, declared):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from tests.web.services import test_actor_mcp_runtime as fixtures
    from xagent.web.models.database import Base
    from xagent.web.services import actor_mcp_runtime

    monkeypatch.setenv("XAGENT_TOBY_PERSONAL_STDIO_ENABLED", "true")
    original = actor_mcp_runtime._canonical_stdio_execution

    def with_declaration(*args, **kwargs):
        execution, fields = original(*args, **kwargs)
        if declared is not None:
            launch = {**execution["launch_config"], "read_only_tools": declared}
            execution = {**execution, "launch_config": launch}
        return execution, fields

    monkeypatch.setattr(
        actor_mcp_runtime, "_canonical_stdio_execution", with_declaration
    )
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        app = fixtures._seed_app(db)
        adapter = fixtures._FakeAdapter(
            fixtures._identity(app), fixtures._credentials()
        )
        result = actor_mcp_runtime.resolve_actor_mcp_stdio_configs(
            db,
            user_id=fixtures.USER_ID,
            policy=fixtures._policy(),
            adapter=adapter,
            visible_servers=(),
        )
    engine.dispose()
    [config] = result.configs
    if declared is None:
        assert READ_ONLY_TOOLS_KEY not in config["config"]
    else:
        assert config["config"][READ_ONLY_TOOLS_KEY] == declared


# ---- admin write models ------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"read_only_tools": "q"},
        {"read_only_tools": ["q", ""]},
        {"read_only_tools": None},
    ],
)
def test_admin_models_reject_a_malformed_declaration(bad):
    from xagent.web.api.admin_mcp import PublicMCPAppCreate, PublicMCPAppUpdate

    base = {"app_id": "a", "name": "A", "transport": "stdio"}
    launch = {"command": "npx", "required_env": ["K"], **bad}
    with pytest.raises(ValidationError):
        PublicMCPAppCreate(**base, launch_config=launch)
    with pytest.raises(ValidationError):
        PublicMCPAppUpdate(launch_config=launch)


def test_admin_models_accept_a_list_or_no_declaration():
    from xagent.web.api.admin_mcp import PublicMCPAppCreate, PublicMCPAppUpdate

    base = {"app_id": "a", "name": "A", "transport": "stdio"}
    launch = {"command": "npx", "required_env": ["K"]}
    PublicMCPAppCreate(**base, launch_config=launch)
    PublicMCPAppCreate(**base, launch_config={**launch, "read_only_tools": []})
    PublicMCPAppUpdate(launch_config={**launch, "read_only_tools": ["q"]})
    PublicMCPAppUpdate(launch_config=None)
