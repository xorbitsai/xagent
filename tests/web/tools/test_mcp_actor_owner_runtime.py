from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError, fields
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from xagent.core.utils.encryption import encrypt_value
from xagent.web import mcp_apps
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.mcp_oauth import MCPOAuthClient, MCPOAuthGrant
from xagent.web.models.oauth_provider import OAuthProvider
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.services import connector_team_scope
from xagent.web.services.mcp_runtime import MCPBuiltinOAuthActorPolicy
from xagent.web.tools import config as web_tools_config
from xagent.web.tools.config import WebToolConfig

OWNER_A = "toby:slack:team:actor-a"
OWNER_B = "toby:slack:team:actor-b"
APP_ID = "actor-drive"
APP_EXECUTION = {
    "name": "Actor Drive",
    "transport": "oauth",
    "provider_name": "google",
    "oauth_scopes": [],
    "launch_config": {
        "command": "python",
        "args": ["-m", "actor_drive"],
        "env_mapping": {"ACTOR_ACCESS_TOKEN": "access_token"},
    },
}


@pytest.fixture()
def db_session(tmp_path, monkeypatch):
    registry_lookup = mcp_apps.get_builtin_execution_fields_and_optional_scopes

    def test_registry(app_id: str):
        if app_id == APP_ID:
            return APP_EXECUTION, []
        return registry_lookup(app_id)

    monkeypatch.setattr(
        mcp_apps, "get_builtin_execution_fields_and_optional_scopes", test_registry
    )

    engine = create_engine(
        f"sqlite:///{tmp_path / 'actor-mcp.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = session_factory()
    user = User(username="actor-runtime-user", password_hash="x", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    try:
        yield SimpleNamespace(db=db, user=user, session_factory=session_factory)
    finally:
        connector_team_scope.set_connector_team_hooks()
        db.close()
        engine.dispose()


def _policy(owner: str = OWNER_A) -> MCPBuiltinOAuthActorPolicy:
    return MCPBuiltinOAuthActorPolicy(resource_owner_key=owner)


def _add_builtin_server(
    db: Session,
    user: User,
    *,
    personal: bool = True,
) -> MCPServer:
    app = PublicMCPApp(
        app_id=APP_ID,
        name="Actor Drive",
        description="Canonical actor drive app",
        transport="oauth",
        provider_name="google",
        launch_config={
            "command": "python",
            "args": ["-m", "actor_drive"],
            "env_mapping": {"ACTOR_ACCESS_TOKEN": "access_token"},
        },
        is_visible_in_connector=True,
    )
    server = MCPServer(
        name="Actor Drive",
        description="Canonical actor drive server",
        managed="external",
        transport="oauth",
        auth={"app_id": APP_ID, "provider": "google"},
    )
    db.add_all([app, server])
    db.flush()
    if personal:
        db.add(
            UserMCPServer(
                user_id=int(user.id),
                mcpserver_id=int(server.id),
                is_owner=False,
                is_active=True,
            )
        )
    db.commit()
    return server


def _add_actor_credential(
    db: Session,
    user: User,
    *,
    owner: str | None,
    token: str,
) -> UserOAuth:
    account = UserOAuth(
        user_id=int(user.id),
        provider=APP_ID,
        resource_owner_key=owner,
        access_token=token,
        provider_user_id="provider-user",
    )
    db.add(account)
    db.commit()
    return account


def _config(
    seeded,
    *,
    policy: MCPBuiltinOAuthActorPolicy | None,
    connector_team_id: int | None = None,
    mcp_auth_context: dict | None = None,
) -> WebToolConfig:
    return WebToolConfig(
        db=seeded.db,
        db_factory=seeded.session_factory,
        request=None,
        user=seeded.user,
        user_id=int(seeded.user.id),
        workspace_config={"base_dir": "/tmp", "task_id": "1"},
        connector_team_id=connector_team_id,
        mcp_auth_context=mcp_auth_context,
        mcp_runtime_authorization_policy=policy,
    )


def _token(config: dict) -> str:
    return config["config"]["env"]["ACTOR_ACCESS_TOKEN"]


def test_actor_policy_is_frozen_normalized_and_owner_only() -> None:
    policy = MCPBuiltinOAuthActorPolicy(resource_owner_key=f"  {OWNER_A}  ")

    assert policy.resource_owner_key == OWNER_A
    assert [field.name for field in fields(policy)] == ["resource_owner_key"]
    assert OWNER_A not in repr(policy)
    with pytest.raises(FrozenInstanceError):
        policy.resource_owner_key = OWNER_B  # type: ignore[misc]


@pytest.mark.parametrize("value", [None, 7, True, "", "   "])
def test_actor_policy_rejects_invalid_owner(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        MCPBuiltinOAuthActorPolicy(resource_owner_key=value)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_actor_builtin_uses_only_exact_owner_namespace(db_session) -> None:
    server = _add_builtin_server(db_session.db, db_session.user)
    _add_actor_credential(
        db_session.db, db_session.user, owner=None, token="ordinary-token"
    )
    _add_actor_credential(
        db_session.db, db_session.user, owner=OWNER_A, token="actor-a-token"
    )
    _add_actor_credential(
        db_session.db, db_session.user, owner=OWNER_B, token="actor-b-token"
    )

    configs = await _config(db_session, policy=_policy()).get_mcp_server_configs()

    assert [config["id"] for config in configs] == [int(server.id)]
    assert _token(configs[0]) == "actor-a-token"
    assert "ordinary-token" not in str(configs)
    assert "actor-b-token" not in str(configs)


@pytest.mark.asyncio
async def test_actor_builtin_refresh_remains_in_exact_owner_namespace(
    db_session, monkeypatch
) -> None:
    _add_builtin_server(db_session.db, db_session.user)
    ordinary = _add_actor_credential(
        db_session.db, db_session.user, owner=None, token="ordinary-token"
    )
    actor_a = _add_actor_credential(
        db_session.db, db_session.user, owner=OWNER_A, token="expired-actor-a-token"
    )
    actor_b = _add_actor_credential(
        db_session.db, db_session.user, owner=OWNER_B, token="actor-b-token"
    )

    async def refresh_exact_actor(_db, account, provider_name):
        assert account.id == actor_a.id
        assert account.resource_owner_key == OWNER_A
        assert provider_name == "google"
        account.access_token = "refreshed-actor-a-token"
        return True

    monkeypatch.setattr(
        web_tools_config,
        "refresh_oauth_token_if_needed",
        refresh_exact_actor,
    )

    configs = await _config(db_session, policy=_policy()).get_mcp_server_configs()

    assert _token(configs[0]) == "refreshed-actor-a-token"
    db_session.db.expire_all()
    assert db_session.db.get(UserOAuth, ordinary.id).access_token == "ordinary-token"
    assert db_session.db.get(UserOAuth, actor_b.id).access_token == "actor-b-token"


@pytest.mark.asyncio
async def test_concurrent_actor_refresh_keeps_rotated_credential(
    db_session,
    monkeypatch,
) -> None:
    _add_builtin_server(db_session.db, db_session.user)
    db_session.db.add(
        OAuthProvider(
            provider_name="google",
            name="Google",
            client_id="client-id",
            client_secret="client-secret",
            auth_url="https://accounts.example/authorize",
            token_url="https://accounts.example/token",
        )
    )
    account = _add_actor_credential(
        db_session.db,
        db_session.user,
        owner=OWNER_A,
        token="expired-token",
    )
    account.refresh_token = "rotating-refresh-token"
    account.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    db_session.db.commit()

    class Response:
        def __init__(self, status_code: int, payload: dict | None = None) -> None:
            self.status_code = status_code
            self._payload = payload or {}

        def json(self) -> dict:
            return self._payload

    post_calls = 0

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, *_args, **_kwargs):
            nonlocal post_calls
            post_calls += 1
            if post_calls == 1:
                await asyncio.sleep(0.05)
                return Response(
                    200,
                    {
                        "access_token": "winner-token",
                        "refresh_token": "winner-refresh-token",
                        "expires_in": 3600,
                    },
                )

            for _ in range(100):
                with db_session.session_factory() as check_db:
                    current = check_db.get(UserOAuth, int(account.id))
                    if current is not None and current.access_token == "winner-token":
                        break
                await asyncio.sleep(0.01)
            return Response(400)

    monkeypatch.setattr(web_tools_config.httpx, "AsyncClient", Client)

    first, second = await asyncio.gather(
        _config(db_session, policy=_policy()).get_mcp_server_configs(),
        _config(db_session, policy=_policy()).get_mcp_server_configs(),
    )

    assert post_calls == 1
    assert _token(first[0]) == "winner-token"
    assert _token(second[0]) == "winner-token"
    db_session.db.expire_all()
    persisted = db_session.db.get(UserOAuth, int(account.id))
    assert persisted is not None
    assert persisted.access_token == "winner-token"
    assert persisted.refresh_token == "winner-refresh-token"


@pytest.mark.asyncio
async def test_actor_builtin_missing_exact_credential_does_not_fallback(
    db_session,
) -> None:
    _add_builtin_server(db_session.db, db_session.user)
    _add_actor_credential(
        db_session.db, db_session.user, owner=None, token="ordinary-token"
    )
    _add_actor_credential(
        db_session.db, db_session.user, owner=OWNER_B, token="actor-b-token"
    )
    tool_config = _config(db_session, policy=_policy())

    configs = await tool_config.get_mcp_server_configs()

    assert configs[0]["transport"] == "unavailable"
    assert configs[0]["config"]["reason"] == "oauth_token_required"
    assert "ordinary-token" not in str(configs)
    assert "actor-b-token" not in str(configs)
    assert OWNER_A not in str(configs)
    assert OWNER_A not in str(tool_config.get_mcp_oauth_diagnostics())


@pytest.mark.asyncio
async def test_team_visible_canonical_builtin_uses_exact_actor_credential(
    db_session,
) -> None:
    server = _add_builtin_server(db_session.db, db_session.user, personal=False)
    _add_actor_credential(
        db_session.db, db_session.user, owner=OWNER_A, token="team-actor-token"
    )
    connector_team_scope.set_connector_team_hooks(
        team_visibility=lambda _db, *, team_id: {
            "mcp": {int(server.id)} if team_id == 41 else set(),
            "custom_api": set(),
        }
    )

    configs = await _config(
        db_session,
        policy=_policy(),
        connector_team_id=41,
    ).get_mcp_server_configs()

    assert [config["id"] for config in configs] == [int(server.id)]
    assert _token(configs[0]) == "team-actor-token"


@pytest.mark.asyncio
async def test_drifted_reserved_builtin_cannot_fall_through_to_native_stdio(
    db_session,
) -> None:
    server = _add_builtin_server(db_session.db, db_session.user)
    server.transport = "stdio"
    server.command = "/bin/unsafe-native"
    server.args = ["--secret", "task-token"]
    db_session.db.commit()
    tool_config = _config(db_session, policy=_policy())

    configs = await tool_config.get_mcp_server_configs()

    assert configs[0]["transport"] == "unavailable"
    assert configs[0]["config"]["reason"] == "config_load_failed"
    assert "/bin/unsafe-native" not in str(configs)
    assert "task-token" not in str(configs)


@pytest.mark.asyncio
async def test_actor_builtin_rejects_selectors_and_ignores_task_credentials(
    db_session,
    monkeypatch,
) -> None:
    server = _add_builtin_server(db_session.db, db_session.user)
    _add_actor_credential(
        db_session.db, db_session.user, owner=OWNER_A, token="actor-token"
    )
    monkeypatch.setattr(
        "xagent.web.services.connector_runtime.load_connector_runtime_view",
        lambda **_kwargs: {
            f"mcp:{server.id}": {
                "context": {"secret-context": "context-value"},
                "secrets": {"access_token": "task-secret-token"},
                "auth_selector": {"resource_owner_key": OWNER_B},
            }
        },
    )
    tool_config = _config(
        db_session,
        policy=_policy(),
        mcp_auth_context={
            str(server.id): {
                "resource_owner_key": OWNER_B,
                "access_token": "selector-token",
            }
        },
    )

    configs = await tool_config.get_mcp_server_configs()

    assert configs[0]["transport"] == "unavailable"
    assert configs[0]["config"]["reason"] == "config_load_failed"
    rendered = str((configs, tool_config.get_mcp_oauth_diagnostics()))
    for secret in (
        OWNER_A,
        OWNER_B,
        "task-secret-token",
        "selector-token",
        "context-value",
        "actor-token",
    ):
        assert secret not in rendered
    assert "connector_runtime" not in configs[0]
    assert "runtime_bindings" not in configs[0]

    ignored_runtime_config = _config(db_session, policy=_policy())
    ignored_runtime_configs = await ignored_runtime_config.get_mcp_server_configs()
    assert _token(ignored_runtime_configs[0]) == "actor-token"
    ignored_rendered = str(
        (
            ignored_runtime_configs,
            ignored_runtime_config.get_mcp_oauth_diagnostics(),
        )
    )
    for secret in (OWNER_A, OWNER_B, "task-secret-token", "context-value"):
        assert secret not in ignored_rendered
    assert "connector_runtime" not in ignored_runtime_configs[0]
    assert "runtime_bindings" not in ignored_runtime_configs[0]


@pytest.mark.asyncio
async def test_normalized_reserved_builtin_alias_cannot_run_as_native(
    db_session,
) -> None:
    server = _add_builtin_server(db_session.db, db_session.user)
    server.auth = None
    server.name = "  ACTOR   DRIVE  "
    server.transport = "stdio"
    server.command = "/bin/unsafe-alias"
    db_session.db.commit()

    configs = await _config(db_session, policy=_policy()).get_mcp_server_configs()

    assert configs[0]["transport"] == "unavailable"
    assert configs[0]["config"]["reason"] == "config_load_failed"
    assert "/bin/unsafe-alias" not in str(configs)


@pytest.mark.asyncio
async def test_normalized_reserved_auth_alias_cannot_run_as_native(
    db_session,
) -> None:
    server = _add_builtin_server(db_session.db, db_session.user)
    server.name = "Foreign native server"
    server.auth = {"app_id": "  ACTOR   DRIVE  ", "provider": "google"}
    server.transport = "stdio"
    server.command = "/bin/unsafe-auth-alias"
    db_session.db.commit()

    configs = await _config(db_session, policy=_policy()).get_mcp_server_configs()

    assert configs[0]["transport"] == "unavailable"
    assert configs[0]["config"]["reason"] == "config_load_failed"
    assert "/bin/unsafe-auth-alias" not in str(configs)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["stdio", "sse", "websocket", "streamable_http"])
async def test_actor_policy_preserves_native_mcp_transports(
    db_session, transport
) -> None:
    kwargs = (
        {"command": "echo", "args": ["native"], "env": {"NATIVE": "value"}}
        if transport == "stdio"
        else {"url": "https://native.example/mcp", "headers": {"X-Native": "value"}}
    )
    server = MCPServer(
        name=f"native-{transport}",
        description="native server",
        managed="external",
        transport=transport,
        **kwargs,
    )
    db_session.db.add(server)
    db_session.db.flush()
    db_session.db.add(
        UserMCPServer(
            user_id=int(db_session.user.id),
            mcpserver_id=int(server.id),
            is_owner=True,
            is_active=True,
        )
    )
    db_session.db.commit()

    ordinary = await _config(db_session, policy=None).get_mcp_server_configs()
    actor = await _config(db_session, policy=_policy()).get_mcp_server_configs()

    assert actor == ordinary
    assert actor[0]["transport"] == transport


@pytest.mark.asyncio
async def test_actor_policy_preserves_native_mcp_oauth(db_session) -> None:
    server = MCPServer(
        name="native-mcp-oauth",
        description="native OAuth server",
        managed="external",
        transport="streamable_http",
        url="https://native.example/mcp",
        auth={
            "type": "mcp_oauth",
            "resource": "https://native.example/mcp",
            "issuer": "https://issuer.example",
            "client_id": "native-client",
            "scope": "read",
        },
    )
    db_session.db.add(server)
    db_session.db.flush()
    db_session.db.add(
        UserMCPServer(
            user_id=int(db_session.user.id),
            mcpserver_id=int(server.id),
            is_owner=True,
            is_active=True,
        )
    )
    client = MCPOAuthClient(
        mcp_server_id=int(server.id),
        issuer="https://issuer.example",
        authorization_endpoint="https://issuer.example/authorize",
        token_endpoint="https://issuer.example/token",
        client_id="native-client",
        token_endpoint_auth_method="none",
        redirect_uri="https://xagent.example/callback",
    )
    db_session.db.add(client)
    db_session.db.flush()
    db_session.db.add(
        MCPOAuthGrant(
            mcp_server_id=int(server.id),
            user_id=int(db_session.user.id),
            mcp_oauth_client_id=int(client.id),
            resource_owner_key=f"xagent:user:{db_session.user.id}",
            issuer="https://issuer.example",
            resource="https://native.example/mcp",
            scope="read",
            access_token=encrypt_value("native-oauth-token"),
            expires_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
            status="active",
        )
    )
    db_session.db.commit()

    configs = await _config(db_session, policy=_policy()).get_mcp_server_configs()

    assert configs[0]["transport"] == "streamable_http"
    assert configs[0]["config"]["headers"]["Authorization"] == (
        "Bearer native-oauth-token"
    )


@pytest.mark.asyncio
async def test_policy_none_preserves_ordinary_builtin_oauth(db_session) -> None:
    _add_builtin_server(db_session.db, db_session.user)
    _add_actor_credential(
        db_session.db, db_session.user, owner=None, token="ordinary-token"
    )
    _add_actor_credential(
        db_session.db, db_session.user, owner=OWNER_A, token="actor-token"
    )

    configs = await _config(db_session, policy=None).get_mcp_server_configs()

    assert _token(configs[0]) == "ordinary-token"
    assert "actor-token" not in str(configs)
