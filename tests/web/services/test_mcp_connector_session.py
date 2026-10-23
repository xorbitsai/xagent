"""Starting one connected MCP server the way an agent task does, outside a task.

The parity tests run the real task tool creator (``create_mcp_tools``) and the
connector session module against the same database, and compare the
connection each hands to the MCP loader.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from tests.web.api.test_mcp_oauth_flow import _add_mcp_oauth_server, _add_oauth_client
from xagent.builtin_identity import owned_catalog_marker
from xagent.core.tools.adapters.vibe.connector_runtime import ConnectorRef
from xagent.core.tools.adapters.vibe.mcp_adapter import (
    MCPFailurePhase,
    MCPLoadResult,
    MCPServerLoadFailure,
)
from xagent.core.tools.adapters.vibe.mcp_tools import create_mcp_tools
from xagent.core.tools.adapters.vibe.signed_upload import signed_upload_settings_error
from xagent.core.utils.encryption import encrypt_env_dict, encrypt_value
from xagent.web import mcp_apps
from xagent.web.api.mcp import _ensure_catalog_app_server, get_mcp_server_tools
from xagent.web.builtin_mcp_registry import seed_builtin_oauth_and_public_mcp_apps
from xagent.web.models import MCPOAuthGrant
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.services.mcp_connector_session import (
    ConnectorToolsOutcome,
    list_connector_tools,
)
from xagent.web.services.mcp_oauth import MCPOAuthRuntimeError
from xagent.web.services.mcp_runtime import (
    CALLER_ID_ENV_VAR,
    mcp_env_hooks_snapshot,
    set_mcp_shared_env_hook,
)
from xagent.web.tools.config import WebToolConfig, set_oauth_token_resolver_hook

LOADER = "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools"
SANDBOX_MANAGER = "xagent.web.sandbox_manager.get_sandbox_manager"
BUILD_RUNTIME_CONNECTION = (
    "xagent.web.services.mcp_runtime.build_mcp_runtime_connection"
)

SIGNED_UPLOADS = [
    {
        "tool_name": "attach_document",
        "file_argument": "document_file_id",
        "issuing_tool_name": "request_upload",
        "allowed_upload_hosts": ["uploads.example.test"],
        "upload_path_prefix": "/staged/",
        "content_types": {".pdf": "application/pdf"},
    }
]


@pytest.fixture()
def db_users(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'connector-session.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        seed_builtin_oauth_and_public_mcp_apps(connection)
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    user = User(username="monitor", password_hash="x", is_admin=False)
    other_user = User(username="other", password_hash="x", is_admin=False)
    db.add_all([user, other_user])
    db.commit()
    db.refresh(user)
    db.refresh(other_user)
    yield db, user, other_user
    db.close()
    engine.dispose()


@pytest.fixture(autouse=True)
def _no_sandbox_manager(monkeypatch):
    monkeypatch.setattr(SANDBOX_MANAGER, lambda: None)


class _Loader:
    """Stand-in for the MCP loader that records what it was handed.

    It keeps the narrow signature the existing route tests use, so the module
    must not pass ``sandbox`` or ``workspace`` unless it has one.
    """

    def __init__(self, result: MCPLoadResult | None = None) -> None:
        self.calls: list[tuple[dict[str, Any], dict[str, Any]]] = []
        self.result = result

    async def __call__(self, connections, **kwargs):
        self.calls.append((connections, kwargs))
        if self.result is not None:
            return self.result
        return MCPLoadResult(
            tools=tuple(
                SimpleNamespace(name=f"{name}_tool", description="A tool")
                for name in connections
            ),
            loaded_servers=tuple(connections),
            failures=(),
        )


def _link(
    db, user: User, server: MCPServer, *, is_owner: bool = True, **fields: Any
) -> None:
    db.add(
        UserMCPServer(
            user_id=user.id,
            mcpserver_id=server.id,
            is_owner=is_owner,
            is_active=True,
            **fields,
        )
    )
    db.commit()


def _connect_github(db, user: User, *, token: str | None = "gh-token") -> MCPServer:
    app = mcp_apps.get_app_by_id(db, "github")
    assert app is not None
    server = MCPServer(
        name=app["name"],
        description=app["description"],
        managed="external",
        transport="oauth",
        # The metadata the OAuth callback stamps (api/auth.py).
        auth={
            "app_id": "github",
            "provider": "github",
            "builtin_provenance": owned_catalog_marker(app),
        },
    )
    db.add(server)
    db.commit()
    _link(db, user, server)
    if token is not None:
        db.add(
            UserOAuth(
                user_id=user.id,
                provider="github",
                access_token=token,
                refresh_token="gh-refresh",
                provider_user_id="1",
                expires_at=datetime.now(timezone.utc) + timedelta(days=1),
            )
        )
        db.commit()
    return server


def _connect_stripe(db, user: User, *, env_source: str | None = "own") -> MCPServer:
    server, _ = _ensure_catalog_app_server(db, "stripe")
    server.env = encrypt_env_dict({"STRIPE_API_KEY": "platform-key"})
    db.commit()
    _link(
        db,
        user,
        server,
        is_owner=False,
        env=encrypt_env_dict({"STRIPE_API_KEY": "user-key"}),
        env_source=env_source,
    )
    return server


def _connect_remote_oauth(db, user: User) -> MCPServer:
    server = _add_mcp_oauth_server(db, user)
    server.headers = {"X-Request-Source": "xagent"}
    client = _add_oauth_client(db, server)
    db.add(
        MCPOAuthGrant(
            mcp_server_id=server.id,
            user_id=user.id,
            mcp_oauth_client_id=client.id,
            resource_owner_key=f"xagent:user:{user.id}",
            issuer="https://auth.example.com",
            resource="https://mcp.example.com/mcp",
            scope="records.read",
            access_token=encrypt_value("runtime-token"),
            status="active",
        )
    )
    db.commit()
    return server


def _connect_plain_stdio(db, user: User) -> MCPServer:
    server = MCPServer.from_config(
        {
            "name": "custom",
            "managed": "external",
            "transport": "stdio",
            "command": "python",
            "args": ["-m", "custom_server", "--flag"],
            "env": {"CUSTOM_SETTING": "on"},
        }
    )
    db.add(server)
    db.commit()
    _link(db, user, server)
    return server


def _connect_chrome(db, user: User) -> MCPServer:
    server = MCPServer.from_config(
        {
            "name": "chrome-devtools",
            "managed": "external",
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "chrome-devtools-mcp@1.6.0"],
        }
    )
    db.add(server)
    db.commit()
    _link(db, user, server)
    return server


def _comparable(connection: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in connection.items() if not callable(value)}


async def _task_connection(
    db, user: User, server: MCPServer, monkeypatch
) -> tuple[dict[str, Any], dict[str, Any]]:
    loader = _Loader(MCPLoadResult(tools=(), loaded_servers=(), failures=()))
    monkeypatch.setattr(LOADER, loader)
    await create_mcp_tools(
        WebToolConfig(db=db, request=None, user=user, user_id=user.id)
    )
    (connections, kwargs) = loader.calls[0]
    return connections[server.name], kwargs["connector_refs"]


async def _session_connection(
    db, user: User, server: MCPServer, monkeypatch
) -> tuple[dict[str, Any], dict[str, Any]]:
    loader = _Loader()
    monkeypatch.setattr(LOADER, loader)
    monkeypatch.setattr(SANDBOX_MANAGER, lambda: _SandboxManager())
    result = await list_connector_tools(
        db, user=user, server_id=server.id, name_prefix="mcp_"
    )
    assert result.outcome is ConnectorToolsOutcome.OK
    ((connections, kwargs),) = loader.calls
    assert set(connections) == {server.name}
    return connections[server.name], kwargs["connector_refs"]


async def _assert_task_parity(db, user: User, server: MCPServer, monkeypatch):
    task_connection, task_refs = await _task_connection(db, user, server, monkeypatch)
    connection, refs = await _session_connection(db, user, server, monkeypatch)
    assert _comparable(connection) == _comparable(task_connection)
    assert refs == task_refs == {server.name: ConnectorRef("mcp", server.id)}
    assert "id" not in connection
    return connection


@pytest.mark.asyncio
async def test_builtin_oauth_connector_starts_with_the_users_token(
    db_users, monkeypatch
):
    db, user, _ = db_users
    server = _connect_github(db, user)

    connection = await _assert_task_parity(db, user, server, monkeypatch)

    assert connection["transport"] == "stdio"
    assert connection["command"] == "python"
    assert connection["args"] == ["-m", "xagent.web.tools.mcp.github"]
    assert connection["env"]["GITHUB_ACCESS_TOKEN"] == "gh-token"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("env_source", "expected_key"),
    [
        ("own", "user-key"),
        ("shared", "shared-key"),
        ("platform", "platform-key"),
        (None, "user-key"),
    ],
)
async def test_api_key_connector_gets_the_env_layer_the_user_picked(
    db_users, monkeypatch, env_source, expected_key
):
    db, user, _ = db_users
    server = _connect_stripe(db, user, env_source=env_source)

    with mcp_env_hooks_snapshot():
        set_mcp_shared_env_hook(
            lambda _db, _user_id: {server.id: {"STRIPE_API_KEY": "shared-key"}}
        )
        connection = await _assert_task_parity(db, user, server, monkeypatch)

    assert connection["env"]["STRIPE_API_KEY"] == expected_key
    assert connection["env"][CALLER_ID_ENV_VAR] == str(user.id)


@pytest.mark.asyncio
async def test_builtin_oauth_token_is_refreshed_before_the_start(db_users, monkeypatch):
    db, user, _ = db_users
    server = _connect_github(db, user)
    db.query(UserOAuth).update(
        {"expires_at": datetime.now(timezone.utc) + timedelta(minutes=1)}
    )
    db.commit()
    refreshed: list[str] = []

    async def refresh(oauth_db, oauth_account, provider_name):
        # Stands in for the provider round trip of the task's own refresh
        # step (web/tools/config.py), which decides on the 5-minute skew.
        refreshed.append(provider_name)
        oauth_account.access_token = "gh-refreshed"
        oauth_account.expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        return True

    monkeypatch.setattr(
        "xagent.web.tools.config.refresh_oauth_token_if_needed", refresh
    )
    loader = _Loader()
    monkeypatch.setattr(LOADER, loader)

    result = await list_connector_tools(
        db, user=user, server_id=server.id, name_prefix="mcp_"
    )

    assert result.outcome is ConnectorToolsOutcome.OK
    assert refreshed == ["github"]
    connection = loader.calls[0][0][server.name]
    assert connection["env"]["GITHUB_ACCESS_TOKEN"] == "gh-refreshed"
    db.expire_all()
    assert db.query(UserOAuth).one().access_token == "gh-refreshed"


@pytest.mark.asyncio
async def test_remote_mcp_oauth_connector_gets_its_grant(db_users, monkeypatch):
    db, user, _ = db_users
    server = _connect_remote_oauth(db, user)

    connection = await _assert_task_parity(db, user, server, monkeypatch)

    assert connection["headers"]["Authorization"] == "Bearer runtime-token"
    assert connection["headers"]["X-Request-Source"] == "xagent"


@pytest.mark.asyncio
async def test_plain_stdio_server_matches_the_task(db_users, monkeypatch):
    db, user, _ = db_users
    server = _connect_plain_stdio(db, user)

    connection = await _assert_task_parity(db, user, server, monkeypatch)

    assert connection["args"] == ["-m", "custom_server", "--flag"]
    assert connection["env"]["CUSTOM_SETTING"] == "on"


@pytest.mark.asyncio
async def test_ownerless_custom_stdio_is_still_sandbox_required(db_users, monkeypatch):
    db, user, _ = db_users
    server = _connect_plain_stdio(db, user)
    association = (
        db.query(UserMCPServer)
        .filter(
            UserMCPServer.user_id == user.id,
            UserMCPServer.mcpserver_id == server.id,
        )
        .one()
    )
    association.is_owner = False
    db.commit()

    connection, _refs = await _task_connection(db, user, server, monkeypatch)

    assert connection["sandbox_required"] is True


@pytest.mark.asyncio
async def test_custom_stdio_cannot_spoof_catalog_app_id(db_users, monkeypatch):
    db, user, _ = db_users
    server = _connect_plain_stdio(db, user)
    app = mcp_apps.get_app_by_id(db, "google-maps")
    assert app is not None
    launch = app["launch_config"]
    association = (
        db.query(UserMCPServer)
        .filter(
            UserMCPServer.user_id == user.id,
            UserMCPServer.mcpserver_id == server.id,
        )
        .one()
    )
    association.is_owner = False
    server.name = "google-maps"
    server.command = launch["command"]
    server.args = launch["args"]
    server.auth = {"app_id": "google-maps"}
    server.cwd = "/tmp"
    server.env = {"PYTHONPATH": "/attacker"}
    db.commit()

    connection, _refs = await _task_connection(db, user, server, monkeypatch)

    assert connection["command"] == launch["command"]
    assert connection["sandbox_required"] is True


@pytest.mark.asyncio
async def test_only_the_requested_server_is_resolved(db_users):
    db, user, _ = db_users
    github = _connect_github(db, user)
    stripe = _connect_stripe(db, user)
    config = WebToolConfig(db=db, request=None, user=user, user_id=user.id)

    everything = await config._load_mcp_server_configs()
    only_stripe = await config._load_mcp_server_configs(only_server_id=stripe.id)

    assert {entry["id"] for entry in everything} == {github.id, stripe.id}
    assert [entry["id"] for entry in only_stripe] == [stripe.id]


@pytest.mark.asyncio
async def test_a_config_without_a_user_resolves_nothing(db_users):
    db, user, _ = db_users
    server = _connect_github(db, user)
    config = WebToolConfig(db=db, request=None, user_id=None)

    assert await config.load_mcp_server_config(server.id) is None


@pytest.mark.asyncio
async def test_a_server_without_the_users_active_link_is_not_found(
    db_users, monkeypatch
):
    db, user, other_user = db_users
    foreign = _connect_plain_stdio(db, other_user)
    inactive = _connect_github(db, user)
    db.query(UserMCPServer).filter(UserMCPServer.mcpserver_id == inactive.id).update(
        {"is_active": False}
    )
    db.commit()
    loader = _Loader()
    monkeypatch.setattr(LOADER, loader)

    for server_id in (foreign.id, inactive.id, 999_999):
        result = await list_connector_tools(
            db, user=user, server_id=server_id, name_prefix="mcp_"
        )
        assert result.outcome is ConnectorToolsOutcome.NOT_FOUND

    assert loader.calls == []


@pytest.mark.asyncio
async def test_builtin_oauth_connector_without_a_grant_is_not_connected(
    db_users, monkeypatch
):
    db, user, _ = db_users
    server = _connect_github(db, user, token=None)
    loader = _Loader()
    monkeypatch.setattr(LOADER, loader)

    result = await list_connector_tools(
        db, user=user, server_id=server.id, name_prefix="mcp_"
    )

    assert result.outcome is ConnectorToolsOutcome.NOT_CONNECTED
    assert result.reason == "oauth_token_required"
    assert loader.calls == []


@pytest.mark.asyncio
async def test_remote_mcp_oauth_without_a_grant_carries_its_diagnostic(
    db_users, monkeypatch
):
    db, user, _ = db_users
    server = _add_mcp_oauth_server(db, user)
    loader = _Loader()
    monkeypatch.setattr(LOADER, loader)

    result = await list_connector_tools(
        db, user=user, server_id=server.id, name_prefix="mcp_"
    )

    assert result.outcome is ConnectorToolsOutcome.NOT_CONNECTED
    assert result.diagnostic is not None
    assert result.diagnostic["code"] == "authorization_required"
    assert loader.calls == []


@pytest.mark.asyncio
async def test_execution_scoped_connector_is_unsupported(db_users, monkeypatch):
    db, user, _ = db_users
    server = _connect_chrome(db, user)
    loader = _Loader()
    monkeypatch.setattr(LOADER, loader)

    result = await list_connector_tools(
        db, user=user, server_id=server.id, name_prefix="mcp_"
    )

    assert result.outcome is ConnectorToolsOutcome.UNSUPPORTED
    assert loader.calls == []


def _failed(phase: MCPFailurePhase, error_type: str = "RuntimeError") -> MCPLoadResult:
    return MCPLoadResult(
        tools=(),
        loaded_servers=(),
        failures=(
            MCPServerLoadFailure(
                server_name="custom", phase=phase, error_type=error_type
            ),
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("load_result", "outcome"),
    [
        (_failed(MCPFailurePhase.INITIALIZE, "TimeoutError"), "timed_out"),
        (_failed(MCPFailurePhase.NO_TOOLS_RETURNED), "no_tools"),
        (_failed(MCPFailurePhase.AUTH_REJECTED), "not_connected"),
        (_failed(MCPFailurePhase.SESSION_START), "failed_to_start"),
        (_failed(MCPFailurePhase.LIST_TOOLS), "failed_to_start"),
        (
            MCPLoadResult(
                tools=(SimpleNamespace(name="t", description="d"),),
                loaded_servers=("custom",),
                failures=_failed(MCPFailurePhase.ADAPTER_CONSTRUCTION).failures,
            ),
            "ok",
        ),
    ],
)
async def test_load_results_are_classified(db_users, monkeypatch, load_result, outcome):
    db, user, _ = db_users
    server = _connect_plain_stdio(db, user)
    monkeypatch.setattr(LOADER, _Loader(load_result))
    monkeypatch.setattr(SANDBOX_MANAGER, lambda: _SandboxManager())

    result = await list_connector_tools(
        db, user=user, server_id=server.id, name_prefix="mcp_"
    )

    assert result.outcome.value == outcome
    assert result.load_result is load_result


@pytest.mark.asyncio
async def test_result_does_not_show_the_credential(db_users, monkeypatch):
    db, user, _ = db_users
    server = _connect_github(db, user)
    monkeypatch.setattr(LOADER, _Loader())

    result = await list_connector_tools(
        db, user=user, server_id=server.id, name_prefix="mcp_"
    )

    assert result.outcome is ConnectorToolsOutcome.OK
    assert "gh-token" not in repr(result)


class _SandboxManager:
    def __init__(self, *, attach: bool = True, create_error: bool = False) -> None:
        self.lease = object()
        self.attach = attach
        self.create_error = create_error
        self.released: list[tuple[str, str]] = []

    async def get_or_create_lease_provider(self, lifecycle: str, user_id: str):
        if self.create_error:
            raise RuntimeError("no capacity")
        return self.lease

    async def attach_provider(self, lifecycle: str, user_id: str, provider) -> bool:
        return self.attach

    async def release(self, lifecycle: str, user_id: str) -> None:
        self.released.append((lifecycle, user_id))


def _connect_google_maps(db, user: User) -> MCPServer:
    server, _ = _ensure_catalog_app_server(db, "google-maps")
    _link(db, user, server, is_owner=False)
    return server


@pytest.mark.asyncio
async def test_npx_connector_runs_in_the_users_sandbox(db_users, monkeypatch):
    db, user, _ = db_users
    server = _connect_google_maps(db, user)
    manager = _SandboxManager()
    monkeypatch.setattr(SANDBOX_MANAGER, lambda: manager)
    loader = _Loader()
    monkeypatch.setattr(LOADER, loader)

    result = await list_connector_tools(
        db, user=user, server_id=server.id, name_prefix="mcp_"
    )

    assert result.outcome is ConnectorToolsOutcome.OK
    assert "sandbox_required" not in loader.calls[0][0]["google-maps"]
    assert loader.calls[0][1]["sandbox"] is manager.lease
    assert manager.released == [("tools", str(user.id))]


@pytest.mark.asyncio
async def test_refuse_sandboxed_declines_before_leasing_or_loading(
    db_users, monkeypatch
):
    db, user, _ = db_users
    server = _connect_google_maps(db, user)
    manager = _SandboxManager()
    monkeypatch.setattr(SANDBOX_MANAGER, lambda: manager)
    loader = _Loader()
    monkeypatch.setattr(LOADER, loader)

    result = await list_connector_tools(
        db, user=user, server_id=server.id, name_prefix="mcp_", refuse_sandboxed=True
    )

    assert result.outcome is ConnectorToolsOutcome.UNSUPPORTED
    assert result.reason == "sandboxed"
    assert loader.calls == []
    assert manager.released == []


@pytest.mark.asyncio
@pytest.mark.parametrize("manager_kwargs", [{"attach": False}, {"create_error": True}])
async def test_npx_connector_fails_closed_without_a_sandbox_lease(
    db_users, monkeypatch, manager_kwargs
):
    db, user, _ = db_users
    server = _connect_google_maps(db, user)
    manager = _SandboxManager(**manager_kwargs)
    monkeypatch.setattr(SANDBOX_MANAGER, lambda: manager)
    loader = _Loader()
    monkeypatch.setattr(LOADER, loader)

    result = await list_connector_tools(
        db, user=user, server_id=server.id, name_prefix="mcp_"
    )

    assert result.outcome is ConnectorToolsOutcome.FAILED_TO_START
    assert result.reason == "sandbox_unavailable"
    assert loader.calls == []
    assert manager.released == []


@pytest.mark.asyncio
async def test_catalog_npx_connector_keeps_host_fallback_without_a_sandbox_manager(
    db_users, monkeypatch
):
    db, user, _ = db_users
    server = _connect_google_maps(db, user)
    loader = _Loader()
    monkeypatch.setattr(LOADER, loader)

    result = await list_connector_tools(
        db, user=user, server_id=server.id, name_prefix="mcp_"
    )

    assert result.outcome is ConnectorToolsOutcome.OK
    assert "sandbox" not in loader.calls[0][1]


@pytest.mark.asyncio
async def test_python_connector_never_leases_a_sandbox(db_users, monkeypatch):
    db, user, _ = db_users
    server = _connect_stripe(db, user)
    manager = _SandboxManager(create_error=True)
    monkeypatch.setattr(SANDBOX_MANAGER, lambda: manager)
    loader = _Loader()
    monkeypatch.setattr(LOADER, loader)

    result = await list_connector_tools(
        db, user=user, server_id=server.id, name_prefix="mcp_"
    )

    assert result.outcome is ConnectorToolsOutcome.OK
    assert "sandbox" not in loader.calls[0][1]


# GET /api/mcp/servers/{id}/tools goes through the module. Its remote MCP OAuth
# behavior is covered by tests/web/api/test_mcp_oauth_flow.py.


@pytest.mark.asyncio
async def test_tools_route_lists_a_builtin_oauth_connector(db_users, monkeypatch):
    db, user, _ = db_users
    server = _connect_github(db, user)
    loader = _Loader()
    monkeypatch.setattr(LOADER, loader)

    response = await get_mcp_server_tools(server.id, user, db)

    assert response["tool_count"] == 1
    ((connections, kwargs),) = loader.calls
    # The route used to hand the bare oauth row to the loader, which cannot
    # open a session for transport "oauth".
    assert connections[server.name]["transport"] == "stdio"
    assert connections[server.name]["env"]["GITHUB_ACCESS_TOKEN"] == "gh-token"
    assert kwargs["name_prefix"] == f"server_{server.id}_"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("connect", "status_code", "code"),
    [
        (
            lambda db, user: _connect_github(db, user, token=None),
            400,
            "oauth_token_required",
        ),
        (_connect_chrome, 400, "unsupported"),
    ],
)
async def test_tools_route_reports_why_a_connector_cannot_start(
    db_users, monkeypatch, connect, status_code, code
):
    db, user, _ = db_users
    server = connect(db, user)
    monkeypatch.setattr(LOADER, _Loader())

    with pytest.raises(HTTPException) as exc_info:
        await get_mcp_server_tools(server.id, user, db)

    assert exc_info.value.status_code == status_code
    assert exc_info.value.detail["code"] == code


@pytest.mark.asyncio
async def test_tools_route_returns_any_remote_oauth_diagnostic_as_400(
    db_users, monkeypatch
):
    db, user, _ = db_users
    server = _connect_remote_oauth(db, user)

    async def invalid_configuration(*args, **kwargs):
        raise MCPOAuthRuntimeError("invalid_configuration", "Scope is invalid")

    monkeypatch.setattr(
        "xagent.web.services.mcp_runtime.resolve_mcp_oauth_runtime_auth",
        invalid_configuration,
    )
    monkeypatch.setattr(LOADER, _Loader())

    with pytest.raises(HTTPException) as exc_info:
        await get_mcp_server_tools(server.id, user, db)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["code"] == "invalid_configuration"
    assert exc_info.value.detail["message"] == "Scope is invalid"


@pytest.mark.asyncio
async def test_tools_route_reports_a_connector_that_cannot_be_started(
    db_users, monkeypatch
):
    db, user, _ = db_users
    server = _connect_google_maps(db, user)
    monkeypatch.setattr(SANDBOX_MANAGER, lambda: _SandboxManager(attach=False))

    with pytest.raises(HTTPException) as exc_info:
        await get_mcp_server_tools(server.id, user, db)

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == {
        "code": "mcp_tools_unavailable",
        "message": "MCP server could not be started.",
        "reason": "sandbox_unavailable",
        "failures": [],
    }


def _connect_plain_remote(db, user: User, *, transport: str, url: str) -> MCPServer:
    server = MCPServer.from_config(
        {
            "name": f"uploads-{transport}",
            "managed": "external",
            "transport": transport,
            "url": url,
        }
    )
    db.add(server)
    db.commit()
    _link(db, user, server)
    return server


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transport", "url"),
    [
        ("streamable_http", "https://mcp.example.test/mcp"),
        ("sse", "https://mcp.example.test/sse"),
    ],
)
async def test_signed_uploads_reach_connection(db_users, monkeypatch, transport, url):
    db, user, _ = db_users
    server = _connect_plain_remote(db, user, transport=transport, url=url)
    server.signed_uploads = SIGNED_UPLOADS
    db.commit()

    connection = await _assert_task_parity(db, user, server, monkeypatch)

    assert connection["signed_uploads"] == SIGNED_UPLOADS


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["websocket", "stdio"])
async def test_signed_uploads_stay_off_other_transports(
    db_users, monkeypatch, transport
):
    db, user, _ = db_users
    if transport == "websocket":
        server = _connect_plain_remote(
            db, user, transport="websocket", url="wss://mcp.example.test/ws"
        )
    else:
        server = _connect_plain_stdio(db, user)
    server.signed_uploads = SIGNED_UPLOADS
    db.commit()

    connection = await _assert_task_parity(db, user, server, monkeypatch)

    assert "signed_uploads" not in connection


MARKER = "planted_marker_tool"


def _invalid_entry(**changes: Any) -> dict[str, Any]:
    entry = {**SIGNED_UPLOADS[0], "tool_name": MARKER}
    entry.update(changes)
    return entry


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stored", "url"),
    [
        pytest.param({"tool_name": MARKER}, None, id="not-a-list"),
        pytest.param({}, None, id="empty-object"),
        pytest.param(
            [
                {
                    key: value
                    for key, value in _invalid_entry().items()
                    if key != "content_types"
                }
            ],
            None,
            id="missing-key",
        ),
        pytest.param(
            [_invalid_entry(allowed_upload_hosts=["10.0.0.1"])],
            None,
            id="ip-upload-host",
        ),
        pytest.param([_invalid_entry()], "https:///mcp", id="url-without-host"),
    ],
)
async def test_invalid_settings_fail_closed_in_web_runtime(
    db_users, monkeypatch, caplog, stored, url
):
    caplog.set_level("DEBUG")
    db, user, _ = db_users
    server = _connect_remote_oauth(db, user)
    server.signed_uploads = stored
    if url is not None:
        server.url = url
    db.commit()

    runtime_calls: list[tuple[Any, ...]] = []

    async def _forbidden_runtime_connection(*args: Any, **kwargs: Any):
        runtime_calls.append(args)
        raise AssertionError("the connection must not be built for invalid settings")

    monkeypatch.setattr(BUILD_RUNTIME_CONNECTION, _forbidden_runtime_connection)

    # Records and supplies no token, so a resolution that ran before the check
    # would leave the outcome unchanged and show up only in this list.
    resolver_calls: list[Any] = []

    def _recording_resolver(request: Any) -> None:
        resolver_calls.append(request)
        return None

    set_oauth_token_resolver_hook(_recording_resolver)
    try:
        configs = await WebToolConfig(
            db=db, request=None, user=user, user_id=user.id
        ).get_mcp_server_configs()
    finally:
        set_oauth_token_resolver_hook(None)

    (config,) = [item for item in configs if item["name"] == server.name]
    assert config["transport"] == "unavailable"
    assert config["config"]["unavailable"] is True
    assert config["config"]["reason"] == "invalid_configuration"
    assert config["config"]["message"] == "MCP server upload settings are invalid."
    assert runtime_calls == []
    assert resolver_calls == []
    assert MARKER not in caplog.text
    error = signed_upload_settings_error(
        stored, transport=server.transport, url=server.url
    )
    assert isinstance(error, str) and error
    assert error not in caplog.text
    assert any(
        record.levelname == "WARNING" and server.name in record.getMessage()
        for record in caplog.records
    )
