"""The connector session and probe under an actor policy (#3288).

Each parity test builds the same connector twice against one database: once
through the task tool creator (``create_mcp_tools``) with a ``WebToolConfig``
carrying the actor policy, and once through the connector session with the
same policy passed through. The connection each hands to the MCP loader must
match, per auth shape: builtin OAuth, remote ``mcp_oauth``, actor stdio, and
a team-owned server.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from tests.web.services.test_actor_mcp_runtime import (
    _credentials,
    _credentials_for,
    _FakeAdapter,
)
from tests.web.services.test_connector_probe import _Adapter
from tests.web.services.test_connector_probe import _Loader as _ProbeLoader
from tests.web.services.test_connector_probe import _text_result
from tests.web.services.test_mcp_connector_session import (
    LOADER,
    SANDBOX_MANAGER,
    _comparable,
    _Loader,
    _SandboxManager,
)
from tests.web.tools.test_mcp_actor_owner_runtime import (
    _add_remote_grant,
    _add_remote_server,
)
from xagent.config import TOBY_PERSONAL_STDIO_ENABLED
from xagent.core.tools.adapters.vibe.config import ACTOR_STDIO_SHADOWED_REASON
from xagent.core.tools.adapters.vibe.connector_runtime import (
    ConnectorRef,
    ConnectorRuntimeError,
)
from xagent.core.tools.adapters.vibe.mcp_adapter import MCPLoadResult
from xagent.core.tools.adapters.vibe.mcp_tools import create_mcp_tools
from xagent.web.api.mcp import _ensure_catalog_app_server
from xagent.web.builtin_mcp_registry import seed_builtin_oauth_and_public_mcp_apps
from xagent.web.mcp_apps import ensure_builtin_oauth_server_definition
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.services import (
    actor_mcp_runtime,
    connector_probe,
    connector_team_scope,
)
from xagent.web.services.actor_mcp_runtime import ActorMCPStdioConnectionIdentity
from xagent.web.services.connector_probe import (
    ConnectorProbeBusy,
    ConnectorProbeOutcome,
    run_actor_stdio_connector_probe,
    run_connector_probe,
)
from xagent.web.services.mcp_connector_session import (
    ConnectorToolsOutcome,
    ConnectorToolsResult,
    list_actor_stdio_connector_tools,
    list_connector_tools,
)
from xagent.web.services.mcp_runtime import (
    MCPActorAuthorizationPolicy,
    MCPActorExecutionIdentity,
)
from xagent.web.tools.config import WebToolConfig

OWNER = "toby:slack:T1:U1"
TEAM_ID = 41
STDIO_APP_ID = "posthog"


def _policy(*, stdio: bool = False) -> MCPActorAuthorizationPolicy:
    return MCPActorAuthorizationPolicy(
        resource_owner_key=OWNER, allow_builtin_stdio=stdio
    )


@pytest.fixture()
def db_users(tmp_path):
    """The seeded catalog plus one plain user, as in the session tests."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'connector-actor.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        seed_builtin_oauth_and_public_mcp_apps(connection)
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    user = User(username="monitor", password_hash="x", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    yield db, user
    db.close()
    engine.dispose()


@pytest.fixture(autouse=True)
def _no_sandbox_manager(monkeypatch):
    monkeypatch.setattr(SANDBOX_MANAGER, lambda: None)


@pytest.fixture(autouse=True)
def _reset_team_hooks():
    yield
    connector_team_scope.set_connector_team_hooks()


@pytest.fixture(autouse=True)
def _not_busy():
    connector_probe._probe_running = False
    yield
    connector_probe._probe_running = False


async def _task_load(
    db, user, monkeypatch, **actor: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The connections and connector refs a task hands to the loader."""
    loader = _Loader(MCPLoadResult(tools=(), loaded_servers=(), failures=()))
    monkeypatch.setattr(LOADER, loader)
    await create_mcp_tools(
        WebToolConfig(db=db, request=None, user=user, user_id=user.id, **actor)
    )
    if not loader.calls:
        return {}, {}
    ((connections, kwargs),) = loader.calls
    return connections, kwargs["connector_refs"]


def _session_load(loader: _Loader, name: str) -> tuple[dict[str, Any], Any]:
    ((connections, kwargs),) = loader.calls
    assert set(connections) == {name}
    return connections[name], kwargs["connector_refs"]


# --- builtin OAuth -----------------------------------------------------------


def _connect_github(db, user) -> MCPServer:
    """The canonical github row the actor classification accepts, linked the
    way the connect routes link a catalog row (no ownership)."""
    server = ensure_builtin_oauth_server_definition(db, app_id="github")
    db.commit()
    db.add(
        UserMCPServer(
            user_id=user.id, mcpserver_id=server.id, is_owner=False, is_active=True
        )
    )
    db.commit()
    return server


def _add_github_token(db, user, *, owner: str | None, token: str) -> None:
    db.add(
        UserOAuth(
            user_id=user.id,
            provider="github",
            resource_owner_key=owner,
            access_token=token,
            refresh_token="gh-refresh",
            provider_user_id="1",
            expires_at=datetime.now(timezone.utc) + timedelta(days=1),
        )
    )
    db.commit()


@pytest.mark.asyncio
async def test_builtin_oauth_lists_with_the_actor_row_like_the_task(
    db_users, monkeypatch
):
    db, user = db_users
    server = _connect_github(db, user)
    _add_github_token(db, user, owner=OWNER, token="actor-token")
    task_connections, task_refs = await _task_load(
        db, user, monkeypatch, mcp_runtime_authorization_policy=_policy()
    )

    loader = _Loader()
    monkeypatch.setattr(LOADER, loader)
    result = await list_connector_tools(
        db,
        user=user,
        server_id=server.id,
        name_prefix="mcp_",
        mcp_runtime_authorization_policy=_policy(),
    )

    assert result.outcome is ConnectorToolsOutcome.OK
    connection, refs = _session_load(loader, server.name)
    assert _comparable(connection) == _comparable(task_connections[server.name])
    assert refs == task_refs == {server.name: ConnectorRef("mcp", server.id)}
    assert "actor-token" in connection["env"].values()


@pytest.mark.asyncio
async def test_builtin_oauth_actor_row_is_ignored_without_the_policy(db_users):
    db, user = db_users
    server = _connect_github(db, user)
    _add_github_token(db, user, owner=OWNER, token="actor-token")

    result = await list_connector_tools(
        db, user=user, server_id=server.id, name_prefix="mcp_"
    )

    assert result.outcome is ConnectorToolsOutcome.NOT_CONNECTED
    assert result.reason == "oauth_token_required"


@pytest.mark.asyncio
async def test_builtin_oauth_account_row_is_not_used_under_the_policy(db_users):
    db, user = db_users
    server = _connect_github(db, user)
    _add_github_token(db, user, owner=None, token="account-token")

    result = await list_connector_tools(
        db,
        user=user,
        server_id=server.id,
        name_prefix="mcp_",
        mcp_runtime_authorization_policy=_policy(),
    )

    assert result.outcome is ConnectorToolsOutcome.NOT_CONNECTED
    assert result.reason == "oauth_token_required"


@pytest.mark.parametrize("policy", [True, False])
@pytest.mark.asyncio
async def test_probe_reads_the_actor_row_only_under_the_policy(
    db_users, monkeypatch, policy
):
    db, user = db_users
    server = _connect_github(db, user)
    _add_github_token(db, user, owner=OWNER, token="actor-token")
    adapter = _Adapter(
        "github_get_current_user",
        _text_result(json.dumps({"status": "success", "user": {"login": "a"}})),
    )
    monkeypatch.setattr(LOADER, _ProbeLoader(adapter))

    result = await run_connector_probe(
        db,
        user=user,
        server_id=server.id,
        mcp_runtime_authorization_policy=_policy() if policy else None,
    )

    if policy:
        assert result.outcome is ConnectorProbeOutcome.OK
        assert adapter.calls == [{}]
    else:
        assert result.outcome is ConnectorProbeOutcome.NOT_CONNECTED
        assert result.reason == "oauth_token_required"


# --- remote mcp_oauth --------------------------------------------------------


@pytest.mark.parametrize(
    ("actor_grant", "expected"),
    [(True, "Bearer actor-token"), (False, "Bearer workspace-token")],
)
@pytest.mark.asyncio
async def test_remote_oauth_prefers_the_actor_grant_like_the_task(
    db_users, monkeypatch, actor_grant, expected
):
    db, user = db_users
    server = _add_remote_server(db, user)
    _add_remote_grant(
        db, user, server, owner=f"xagent:user:{user.id}", token="workspace-token"
    )
    if actor_grant:
        _add_remote_grant(db, user, server, owner=OWNER, token="actor-token")
    task_connections, task_refs = await _task_load(
        db, user, monkeypatch, mcp_runtime_authorization_policy=_policy()
    )

    loader = _Loader()
    monkeypatch.setattr(LOADER, loader)
    result = await list_connector_tools(
        db,
        user=user,
        server_id=server.id,
        name_prefix="mcp_",
        mcp_runtime_authorization_policy=_policy(),
    )

    assert result.outcome is ConnectorToolsOutcome.OK
    connection, refs = _session_load(loader, server.name)
    assert _comparable(connection) == _comparable(task_connections[server.name])
    assert refs == task_refs
    assert connection["headers"]["Authorization"] == expected


# --- team-owned servers ------------------------------------------------------


def _team_owns(*servers: MCPServer, owns_definitions: bool = False) -> None:
    ids = {int(server.id) for server in servers}

    def team_visibility(_db, *, team_id):
        mine = set(ids) if team_id == TEAM_ID else set()
        answer: dict[str, set[int]] = {"mcp": mine, "custom_api": set()}
        if owns_definitions:
            answer[connector_team_scope.TEAM_OWNED_MCP_DEFINITIONS_KEY] = set(mine)
        return answer

    connector_team_scope.set_connector_team_hooks(team_visibility=team_visibility)


def _add_unlinked_stdio(db) -> MCPServer:
    server = MCPServer.from_config(
        {
            "name": "team-tool",
            "managed": "external",
            "transport": "stdio",
            "command": "python",
            "args": ["-m", "team_server"],
        }
    )
    db.add(server)
    db.commit()
    return server


@pytest.mark.asyncio
async def test_team_owned_server_lists_with_the_team_like_the_task(
    db_users, monkeypatch
):
    db, user = db_users
    server = _add_unlinked_stdio(db)
    _team_owns(server)
    task_connections, task_refs = await _task_load(
        db, user, monkeypatch, connector_team_id=TEAM_ID
    )

    loader = _Loader()
    monkeypatch.setattr(LOADER, loader)
    monkeypatch.setattr(SANDBOX_MANAGER, lambda: _SandboxManager())
    result = await list_connector_tools(
        db,
        user=user,
        server_id=server.id,
        name_prefix="mcp_",
        connector_team_id=TEAM_ID,
    )

    assert result.outcome is ConnectorToolsOutcome.OK
    connection, refs = _session_load(loader, server.name)
    assert _comparable(connection) == _comparable(task_connections[server.name])
    assert refs == task_refs == {server.name: ConnectorRef("mcp", server.id)}


@pytest.mark.asyncio
async def test_team_only_remote_oauth_lists_with_the_actor_grant_like_the_task(
    db_users, monkeypatch
):
    db, user = db_users
    server = _add_remote_server(db, user)
    # Team-only: drop the user's personal link, so only the team hook shows it.
    db.query(UserMCPServer).filter(UserMCPServer.mcpserver_id == server.id).delete()
    db.commit()
    _team_owns(server, owns_definitions=True)
    _add_remote_grant(db, user, server, owner=OWNER, token="actor-token")
    task_connections, task_refs = await _task_load(
        db,
        user,
        monkeypatch,
        mcp_runtime_authorization_policy=_policy(),
        connector_team_id=TEAM_ID,
    )

    loader = _Loader()
    monkeypatch.setattr(LOADER, loader)
    result = await list_connector_tools(
        db,
        user=user,
        server_id=server.id,
        name_prefix="mcp_",
        mcp_runtime_authorization_policy=_policy(),
        connector_team_id=TEAM_ID,
    )

    assert result.outcome is ConnectorToolsOutcome.OK
    connection, refs = _session_load(loader, server.name)
    assert _comparable(connection) == _comparable(task_connections[server.name])
    assert refs == task_refs == {server.name: ConnectorRef("mcp", server.id)}
    assert connection["headers"]["Authorization"] == "Bearer actor-token"


@pytest.mark.parametrize("team_id", [None, TEAM_ID + 1])
@pytest.mark.asyncio
async def test_team_owned_server_is_not_found_outside_its_team(db_users, team_id):
    db, user = db_users
    server = _add_unlinked_stdio(db)
    _team_owns(server)

    listing = await list_connector_tools(
        db,
        user=user,
        server_id=server.id,
        name_prefix="mcp_",
        connector_team_id=team_id,
    )
    probe = await run_connector_probe(
        db, user=user, server_id=server.id, connector_team_id=team_id
    )

    assert listing.outcome is ConnectorToolsOutcome.NOT_FOUND
    assert probe.outcome is ConnectorProbeOutcome.NOT_FOUND


@pytest.mark.asyncio
async def test_unresolvable_team_scope_raises_the_typed_error(db_users, stdio_adapter):
    db, user = db_users
    server = _add_unlinked_stdio(db)

    def broken(_db, *, team_id):
        raise RuntimeError("team service down")

    connector_team_scope.set_connector_team_hooks(team_visibility=broken)

    with pytest.raises(ConnectorRuntimeError):
        await list_connector_tools(
            db,
            user=user,
            server_id=server.id,
            name_prefix="mcp_",
            connector_team_id=TEAM_ID,
        )
    with pytest.raises(ConnectorRuntimeError):
        await run_connector_probe(
            db, user=user, server_id=server.id, connector_team_id=TEAM_ID
        )
    assert connector_probe._probe_running is False
    with pytest.raises(ConnectorRuntimeError):
        await list_actor_stdio_connector_tools(
            db,
            user=user,
            app_id=STDIO_APP_ID,
            name_prefix="mcp_",
            connector_team_id=TEAM_ID,
            **_stdio_actor(stdio_adapter),
        )
    with pytest.raises(ConnectorRuntimeError):
        await run_actor_stdio_connector_probe(
            db,
            user=user,
            app_id=STDIO_APP_ID,
            connector_team_id=TEAM_ID,
            **_stdio_actor(stdio_adapter),
        )
    assert connector_probe._probe_running is False


@pytest.mark.asyncio
async def test_team_owned_catalog_row_is_probed_with_the_team(db_users, monkeypatch):
    db, user = db_users
    server, _ = _ensure_catalog_app_server(db, "stripe")
    db.commit()
    _team_owns(server)
    adapter = _Adapter(
        "stripe_list_customers",
        _text_result(json.dumps({"status": "success", "customers": []})),
    )
    monkeypatch.setattr(LOADER, _ProbeLoader(adapter))

    result = await run_connector_probe(
        db, user=user, server_id=server.id, connector_team_id=TEAM_ID
    )

    assert result.outcome is ConnectorProbeOutcome.OK
    assert adapter.calls == [{"limit": 1}]


# --- actor stdio -------------------------------------------------------------


@pytest.fixture()
def stdio_adapter(db_users, monkeypatch) -> _FakeAdapter:
    return _connect_actor_stdio(*db_users, monkeypatch, STDIO_APP_ID, _credentials())


def _connect_actor_stdio(
    db, user, monkeypatch, app_id: str, credentials: dict[str, str]
) -> _FakeAdapter:
    """An adapter holding the actor's one connection to catalog ``app_id``."""
    monkeypatch.setenv(TOBY_PERSONAL_STDIO_ENABLED, "true")
    app = db.query(PublicMCPApp).filter(PublicMCPApp.app_id == app_id).one()
    identity = ActorMCPStdioConnectionIdentity(
        user_id=int(user.id),
        resource_owner_key=OWNER,
        app_id=app_id,
        catalog_app_generation=app.generation,
        lifecycle_generation=uuid.uuid4(),
    )
    return _FakeAdapter(identity, credentials)


def _stdio_actor(adapter: _FakeAdapter, *, stdio: bool = True) -> dict[str, Any]:
    return {
        "mcp_runtime_authorization_policy": _policy(stdio=stdio),
        "mcp_actor_stdio_connection_adapter": adapter,
    }


@pytest.mark.asyncio
async def test_actor_stdio_lists_by_app_id_like_the_task(
    db_users, monkeypatch, stdio_adapter
):
    db, user = db_users
    task_connections, task_refs = await _task_load(
        db, user, monkeypatch, **_stdio_actor(stdio_adapter)
    )

    loader = _Loader()
    monkeypatch.setattr(LOADER, loader)
    result = await list_actor_stdio_connector_tools(
        db,
        user=user,
        app_id=STDIO_APP_ID,
        name_prefix="mcp_",
        **_stdio_actor(stdio_adapter),
    )

    assert result.outcome is ConnectorToolsOutcome.OK
    assert result.server_id is None
    assert result.server_name == STDIO_APP_ID
    connection, refs = _session_load(loader, STDIO_APP_ID)
    assert _comparable(connection) == _comparable(task_connections[STDIO_APP_ID])
    assert refs == {}
    assert STDIO_APP_ID not in task_refs
    assert connection["env"]["POSTHOG_API_KEY"] == "actor-api-key"


@pytest.mark.asyncio
async def test_actor_stdio_app_the_actor_has_not_connected_is_not_found(
    db_users, stdio_adapter
):
    db, user = db_users

    result = await list_actor_stdio_connector_tools(
        db,
        user=user,
        app_id="linear",
        name_prefix="mcp_",
        **_stdio_actor(stdio_adapter),
    )

    assert result.outcome is ConnectorToolsOutcome.NOT_FOUND


@pytest.mark.asyncio
async def test_actor_stdio_is_not_found_when_the_policy_disallows_stdio(
    db_users, stdio_adapter
):
    db, user = db_users

    result = await list_actor_stdio_connector_tools(
        db,
        user=user,
        app_id=STDIO_APP_ID,
        name_prefix="mcp_",
        **_stdio_actor(stdio_adapter, stdio=False),
    )

    assert result.outcome is ConnectorToolsOutcome.NOT_FOUND
    assert stdio_adapter.list_calls == []


@pytest.mark.asyncio
async def test_actor_stdio_is_not_found_when_the_deployment_flag_is_off(
    db_users, monkeypatch, stdio_adapter
):
    db, user = db_users
    monkeypatch.setenv(TOBY_PERSONAL_STDIO_ENABLED, "false")

    result = await list_actor_stdio_connector_tools(
        db,
        user=user,
        app_id=STDIO_APP_ID,
        name_prefix="mcp_",
        **_stdio_actor(stdio_adapter),
    )

    assert result.outcome is ConnectorToolsOutcome.NOT_FOUND
    assert stdio_adapter.list_calls == []


@pytest.mark.asyncio
async def test_actor_stdio_shadowed_by_a_visible_server_is_not_found_like_the_task(
    db_users, monkeypatch, stdio_adapter
):
    db, user = db_users
    shadow = MCPServer.from_config(
        {
            "name": STDIO_APP_ID,
            "managed": "external",
            "transport": "stdio",
            "command": "python",
            "args": ["-m", "lookalike"],
        }
    )
    db.add(shadow)
    db.commit()
    _team_owns(shadow)
    actor = {**_stdio_actor(stdio_adapter), "connector_team_id": TEAM_ID}
    # _task_load returns {} when the loader is never called, which would make
    # an absence check vacuous: read the configs the task itself builds.
    task_configs = await WebToolConfig(
        db=db, request=None, user=user, user_id=user.id, **actor
    ).get_mcp_server_configs()

    result = await list_actor_stdio_connector_tools(
        db, user=user, app_id=STDIO_APP_ID, name_prefix="mcp_", **actor
    )

    (task_config,) = [c for c in task_configs if c.get("name") == STDIO_APP_ID]
    assert task_config["transport"] == "unavailable"
    assert task_config["config"]["unavailable"] is True
    assert task_config["config"]["reason"] == ACTOR_STDIO_SHADOWED_REASON
    assert task_config["config"]["server_id"] == shadow.id
    assert result.outcome is ConnectorToolsOutcome.NOT_FOUND


@pytest.mark.asyncio
async def test_execution_scoped_actor_stdio_is_unsupported(db_users, stdio_adapter):
    db, user = db_users

    result = await list_actor_stdio_connector_tools(
        db,
        user=user,
        app_id="chrome-devtools",
        name_prefix="mcp_",
        **_stdio_actor(stdio_adapter),
    )

    assert result.outcome is ConnectorToolsOutcome.UNSUPPORTED
    assert result.reason == "execution_scoped"
    assert stdio_adapter.list_calls == []


@pytest.mark.asyncio
async def test_server_id_call_never_starts_the_actor_stdio_app(
    db_users, monkeypatch, stdio_adapter
):
    """The task's config appends the actor stdio configs to every load, so a
    server-id call must start only the row it names."""
    db, user = db_users
    server = _add_unlinked_stdio(db)
    db.add(
        UserMCPServer(
            user_id=user.id, mcpserver_id=server.id, is_owner=True, is_active=True
        )
    )
    db.commit()
    loader = _Loader()
    monkeypatch.setattr(LOADER, loader)
    monkeypatch.setattr(SANDBOX_MANAGER, lambda: _SandboxManager())

    result = await list_connector_tools(
        db,
        user=user,
        server_id=server.id,
        name_prefix="mcp_",
        **_stdio_actor(stdio_adapter),
    )

    assert result.outcome is ConnectorToolsOutcome.OK
    _session_load(loader, server.name)


@pytest.mark.asyncio
async def test_npx_actor_stdio_is_refused_before_the_loader(db_users, monkeypatch):
    db, user = db_users
    app_id = "google-maps"
    adapter = _connect_actor_stdio(
        db, user, monkeypatch, app_id, _credentials_for(app_id)
    )
    loader = _Loader()
    monkeypatch.setattr(LOADER, loader)

    result = await list_actor_stdio_connector_tools(
        db,
        user=user,
        app_id=app_id,
        name_prefix="mcp_",
        refuse_sandboxed=True,
        **_stdio_actor(adapter),
    )

    assert result.outcome is ConnectorToolsOutcome.UNSUPPORTED
    assert result.reason == "sandboxed"
    assert loader.calls == []


@pytest.mark.asyncio
async def test_actor_stdio_scan_failure_is_a_config_load_failure(
    db_users, monkeypatch, stdio_adapter
):
    db, user = db_users

    def broken(*_args: Any, **_kwargs: Any):
        raise RuntimeError("database is gone")

    monkeypatch.setattr(actor_mcp_runtime, "resolve_actor_mcp_stdio_configs", broken)

    result = await list_actor_stdio_connector_tools(
        db,
        user=user,
        app_id=STDIO_APP_ID,
        name_prefix="mcp_",
        **_stdio_actor(stdio_adapter),
    )

    assert result.outcome is ConnectorToolsOutcome.FAILED_TO_START
    assert result.reason == "config_load_failed"


@pytest.mark.asyncio
async def test_actor_stdio_probe_runs_the_declared_tool(
    db_users, monkeypatch, stdio_adapter
):
    db, user = db_users
    adapter = _Adapter(
        "posthog_list_feature_flags",
        _text_result(json.dumps({"status": "success", "feature_flags": []})),
    )
    loader = _ProbeLoader(adapter)
    monkeypatch.setattr(LOADER, loader)

    result = await run_actor_stdio_connector_probe(
        db, user=user, app_id=STDIO_APP_ID, **_stdio_actor(stdio_adapter)
    )

    assert result.outcome is ConnectorProbeOutcome.OK
    assert result.server_id is None
    assert result.app_id == STDIO_APP_ID
    assert result.tool == "posthog_list_feature_flags"
    ((connections, kwargs),) = loader.calls
    assert set(connections) == {STDIO_APP_ID}
    assert kwargs["connector_refs"] == {}
    assert result.payload()["server_id"] is None


@pytest.mark.asyncio
async def test_actor_stdio_probe_of_an_unconnected_app_is_not_found(
    db_users, stdio_adapter
):
    db, user = db_users

    # stripe declares a probe, so only the missing connection can stop it.
    result = await run_actor_stdio_connector_probe(
        db, user=user, app_id="stripe", **_stdio_actor(stdio_adapter)
    )

    assert result.outcome is ConnectorProbeOutcome.NOT_FOUND


@pytest.mark.asyncio
async def test_actor_stdio_probe_shares_the_one_probe_slot(db_users, stdio_adapter):
    db, user = db_users
    connector_probe._probe_running = True

    with pytest.raises(ConnectorProbeBusy):
        await run_actor_stdio_connector_probe(
            db, user=user, app_id=STDIO_APP_ID, **_stdio_actor(stdio_adapter)
        )


@pytest.mark.parametrize("entry", ["server", "actor_stdio"])
@pytest.mark.asyncio
async def test_probe_passes_every_actor_keyword_through(
    db_users, monkeypatch, stdio_adapter, entry
):
    db, user = db_users
    server, _ = _ensure_catalog_app_server(db, "stripe")
    db.add(
        UserMCPServer(
            user_id=user.id, mcpserver_id=server.id, is_owner=False, is_active=True
        )
    )
    db.commit()
    actor = {
        **_stdio_actor(stdio_adapter),
        "mcp_actor_execution_identity": MCPActorExecutionIdentity(
            task_id=7, run_id="run", turn_id="turn", lease_attempt_id="lease"
        ),
        "connector_team_id": TEAM_ID,
    }
    seen: list[dict[str, Any]] = []
    running_during_listing: list[bool] = []

    async def listing(_db, **kwargs: Any) -> ConnectorToolsResult:
        seen.append(kwargs)
        running_during_listing.append(connector_probe._probe_running)
        return ConnectorToolsResult(ConnectorToolsOutcome.NOT_CONNECTED, None)

    if entry == "server":
        monkeypatch.setattr(connector_probe, "list_connector_tools", listing)
        await run_connector_probe(db, user=user, server_id=server.id, **actor)
    else:
        monkeypatch.setattr(
            connector_probe, "list_actor_stdio_connector_tools", listing
        )
        await run_actor_stdio_connector_probe(
            db, user=user, app_id=STDIO_APP_ID, **actor
        )

    (kwargs,) = seen
    assert {key: kwargs[key] for key in actor} == actor
    # The probe holds the one probe slot while it lists, and frees it after.
    assert running_during_listing == [True]
    assert connector_probe._probe_running is False
