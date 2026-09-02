from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.agent.runtime import PatternRuntime
from xagent.core.tools.adapters.vibe.connector_runtime import ConnectorRuntimeError
from xagent.core.tools.adapters.vibe.mcp_adapter import MCPToolAdapter
from xagent.core.utils.encryption import encrypt_value
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.oauth_provider import OAuthProvider
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.services.mcp_oauth import MCPAuthorizationChallenge
from xagent.web.tools import config as web_tools_config
from xagent.web.tools.config import (
    ResolvedToken,
    TokenRequest,
    WebToolConfig,
    set_oauth_token_resolver_hook,
)


def test_token_request_refresh_defaults_to_none():
    request = TokenRequest(provider="google", user_id=1)

    assert request.refresh is None


def test_resolver_contract_repr_hides_token_and_generation():
    refresh = web_tools_config.OAuthRefreshContext(
        reason="invalid_token",
        resource_metadata_url=None,
        challenge_scope=None,
        failed_generation="failed-generation-secret",
    )
    token = ResolvedToken(
        access_token="access-token-secret",
        generation="current-generation-secret",
    )

    rendered = repr((refresh, token))

    assert "access-token-secret" not in rendered
    assert "failed-generation-secret" not in rendered
    assert "current-generation-secret" not in rendered


def test_oauth_token_generation_max_length_is_1024():
    assert web_tools_config.OAUTH_TOKEN_GENERATION_MAX_LENGTH == 1024


@pytest.fixture(autouse=True)
def clear_oauth_token_resolver_hook():
    set_oauth_token_resolver_hook(None)
    yield
    set_oauth_token_resolver_hook(None)


@pytest.fixture()
def db_session(tmp_path):
    db_path = tmp_path / "oauth-token-resolver.db"
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = SessionLocal()

    user = User(username="alice", password_hash="x", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)

    yield db, user
    db.close()
    engine.dispose()


def _launch_config(
    env_key: str = "GOOGLE_ACCESS_TOKEN", resource: str | None = None
) -> dict:
    launch_config = {
        "command": "npx",
        "args": ["-y", "@mcp-servers/google-drive"],
        "env_mapping": {env_key: "access_token", "IGNORED": "refresh_token"},
    }
    if resource is not None:
        launch_config["resource"] = resource
    return launch_config


def _add_oauth_server(
    db,
    user: User,
    *,
    name: str = "Google Drive",
    app_id: str | None = "resolver-google-drive",
    provider: str | None = "google",
    launch_config: object | None = None,
    register_app: bool = True,
) -> MCPServer:
    server = MCPServer(
        name=name,
        description=f"{name} server",
        managed="external",
        transport="oauth",
    )
    db.add(server)
    db.commit()
    db.refresh(server)
    db.add(
        UserMCPServer(
            user_id=user.id,
            mcpserver_id=server.id,
            is_owner=True,
            is_active=True,
        )
    )
    if register_app:
        db.add(
            PublicMCPApp(
                app_id=app_id or f"{name.lower()}-app",
                name=name,
                description=f"{name} app",
                transport="oauth",
                provider_name=provider,
                launch_config=launch_config if launch_config is not None else {},
            )
        )
    db.commit()
    return server


def _add_stdio_server(db, user: User, *, name: str) -> MCPServer:
    server = MCPServer(
        name=name,
        description=f"{name} server",
        managed="external",
        transport="stdio",
        command="echo",
        args=["ok"],
    )
    db.add(server)
    db.commit()
    db.refresh(server)
    db.add(
        UserMCPServer(
            user_id=user.id,
            mcpserver_id=server.id,
            is_owner=True,
            is_active=True,
        )
    )
    db.commit()
    return server


def _add_remote_server(
    db,
    user: User,
    *,
    name: str = "Remote Records",
    app_id: str = "remote-records",
    provider: str = "records",
    auth: object | None = None,
    headers: dict | None = None,
    runtime_bindings: list[dict] | None = None,
    allow_delegated_authorization: bool = False,
) -> MCPServer:
    server = MCPServer(
        name=name,
        description=f"{name} server",
        managed="external",
        transport="streamable_http",
        url="https://mcp.example/api",
        auth=auth,
        headers=headers,
        runtime_bindings=runtime_bindings,
        allow_delegated_authorization=allow_delegated_authorization,
    )
    db.add(server)
    db.commit()
    db.refresh(server)
    db.add(
        UserMCPServer(
            user_id=user.id,
            mcpserver_id=server.id,
            is_owner=True,
            is_active=True,
        )
    )
    db.add(
        PublicMCPApp(
            app_id=app_id,
            name=name,
            description=f"{name} app",
            transport="streamable_http",
            provider_name=provider,
            launch_config={},
        )
    )
    db.commit()
    return server


def _add_user_oauth(
    db,
    user: User,
    *,
    provider: str = "google",
    access_token: str = "user-token",
    instance_url: str | None = None,
) -> UserOAuth:
    account = UserOAuth(
        user_id=user.id,
        provider=provider,
        access_token=access_token,
        provider_user_id=f"{provider}-user",
        instance_url=instance_url,
    )
    db.add(account)
    db.commit()
    db.refresh(account)
    return account


def _tool_config(db, user: User, **kwargs) -> WebToolConfig:
    return WebToolConfig(
        db=db,
        request=None,
        user=user,
        user_id=user.id,
        workspace_config={"base_dir": "/tmp", "task_id": "task-1"},
        **kwargs,
    )


def _access_token_env(config: dict, key: str = "GOOGLE_ACCESS_TOKEN") -> str:
    return config["config"]["env"][key]


def _assert_same_oauth_config_except_token(
    hook_config: dict,
    user_config: dict,
    *,
    token_key: str,
    hook_token: str,
    user_token: str,
) -> None:
    hook_env = dict(hook_config["config"]["env"])
    user_env = dict(user_config["config"]["env"])
    assert hook_env.pop(token_key) == hook_token
    assert user_env.pop(token_key) == user_token
    assert hook_config["transport"] == user_config["transport"] == "stdio"
    assert hook_config["config"]["transport"] == user_config["config"]["transport"]
    assert hook_config["config"]["command"] == user_config["config"]["command"]
    assert hook_config["config"]["args"] == user_config["config"]["args"]
    assert hook_env == user_env


def _assert_unavailable_mcp_config(
    config: dict,
    server: MCPServer,
    *,
    reason: str,
    oauth_token_required: bool = False,
) -> None:
    assert config["name"] == server.name
    assert config["transport"] == "unavailable"
    assert config["description"] == server.description
    assert config["config"]["unavailable"] is True
    assert config["config"]["reason"] == reason
    assert config["config"]["server_id"] == server.id
    if oauth_token_required:
        assert config["config"]["failure_code"] == "oauth_token_required"
    else:
        assert "failure_code" not in config["config"]
    expected_user_id = str(server.user_mcpservers[0].user_id)
    assert config["user_id"] == expected_user_id
    assert config["allow_users"] == [expected_user_id]
    assert "runtime_input_schema" not in config
    assert "runtime_bindings" not in config
    assert "allow_delegated_authorization" not in config
    assert "connector_runtime" not in config


@pytest.mark.asyncio
async def test_builtin_oauth_server_uses_stable_app_id_and_canonical_runtime_config(
    db_session,
):
    db, user = db_session
    server = _add_oauth_server(
        db,
        user,
        name="Renamed Gmail Server",
        app_id="custom-same-name",
        provider="wrong-provider",
        launch_config={
            "command": "uv",
            "args": ["run", "wrong-server.py"],
            "env_mapping": {"WRONG_TOKEN": "access_token"},
        },
    )
    server.auth = {"app_id": "gmail"}
    db.add(
        PublicMCPApp(
            app_id="gmail",
            name="Stale Gmail Catalog Name",
            transport="oauth",
            provider_name="wrong-provider",
            oauth_scopes=["wrong-scope"],
            launch_config={
                "command": "uv",
                "args": ["run", "python", "-m", "xagent.web.tools.mcp.gmail"],
                "env_mapping": {"WRONG_TOKEN": "access_token"},
            },
        )
    )
    db.commit()

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        return ResolvedToken(
            access_token="hook-token",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )

    set_oauth_token_resolver_hook(resolver)

    configs = await _tool_config(db, user).get_mcp_server_configs()

    assert configs[0]["transport"] == "stdio"
    assert configs[0]["config"]["command"] == "python"
    assert configs[0]["config"]["args"] == [
        "-m",
        "xagent.web.tools.mcp.gmail",
    ]
    assert configs[0]["config"]["env"]["GOOGLE_ACCESS_TOKEN"] == "hook-token"
    assert "WRONG_TOKEN" not in configs[0]["config"]["env"]


@pytest.mark.asyncio
async def test_hook_receives_provider_candidates_in_order_and_first_hit_wins(
    db_session,
):
    db, user = db_session
    _add_oauth_server(db, user, launch_config=_launch_config())
    seen: list[str] = []

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        seen.append(request.provider)
        if request.provider == "resolver-google-drive":
            return ResolvedToken(
                access_token="hook-token",
                expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            )
        return None

    set_oauth_token_resolver_hook(resolver)

    configs = await _tool_config(db, user).get_mcp_server_configs()

    assert seen == ["google", "resolver-google-drive"]
    assert len(configs) == 1
    assert configs[0]["transport"] == "stdio"
    assert _access_token_env(configs[0]) == "hook-token"


@pytest.mark.asyncio
async def test_hook_accepts_sync_resolver(db_session):
    db, user = db_session
    _add_oauth_server(db, user, launch_config=_launch_config())
    seen: list[str] = []

    def resolver(request: TokenRequest) -> ResolvedToken | None:
        seen.append(request.provider)
        return ResolvedToken(
            access_token="sync-hook-token",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )

    set_oauth_token_resolver_hook(resolver)

    configs = await _tool_config(db, user).get_mcp_server_configs()

    assert seen == ["google"]
    assert configs[0]["transport"] == "stdio"
    assert _access_token_env(configs[0]) == "sync-hook-token"


@pytest.mark.asyncio
async def test_hook_dedupes_provider_candidates(db_session):
    db, user = db_session
    _add_oauth_server(
        db,
        user,
        app_id="google",
        provider="google",
        launch_config=_launch_config(),
    )
    seen: list[str] = []

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        seen.append(request.provider)
        return None

    set_oauth_token_resolver_hook(resolver)
    await _tool_config(db, user).get_mcp_server_configs()

    assert seen == ["google"]


@pytest.mark.asyncio
async def test_hook_request_receives_provider_resource_and_scope_verbatim(db_session):
    db, user = db_session
    scope = object()
    resource = "https://MCP.EXAMPLE.com:443/mcp/%7Euser/?Q=1#Fragment"
    _add_oauth_server(db, user, launch_config=_launch_config(resource=resource))
    seen: list[tuple[str, str | None, object | None]] = []

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        seen.append((request.provider, request.resource, request.scope))
        if request.provider == "resolver-google-drive":
            return ResolvedToken(
                access_token="hook-token",
                expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            )
        return None

    set_oauth_token_resolver_hook(resolver)

    configs = await _tool_config(
        db, user, execution_scope=scope
    ).get_mcp_server_configs()

    assert seen == [
        ("google", resource, scope),
        ("resolver-google-drive", resource, scope),
    ]
    assert _access_token_env(configs[0]) == "hook-token"


@pytest.mark.asyncio
async def test_hook_decline_falls_back_to_user_oauth(db_session):
    db, user = db_session
    _add_oauth_server(db, user, launch_config=_launch_config())
    _add_user_oauth(db, user, provider="google", access_token="user-token")

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        return None

    set_oauth_token_resolver_hook(resolver)

    configs = await _tool_config(db, user).get_mcp_server_configs()

    assert configs[0]["transport"] == "stdio"
    assert _access_token_env(configs[0]) == "user-token"


@pytest.mark.asyncio
async def test_hook_supply_fallback_npx_env_matches_user_oauth_shape(db_session):
    db, user = db_session
    _add_oauth_server(db, user, name="Legacy App", app_id="legacy-app")

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        return ResolvedToken(
            access_token="hook-token",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )

    set_oauth_token_resolver_hook(resolver)

    hook_config = (await _tool_config(db, user).get_mcp_server_configs())[0]
    set_oauth_token_resolver_hook(None)
    _add_user_oauth(db, user, provider="google", access_token="user-token")
    user_config = (await _tool_config(db, user).get_mcp_server_configs())[0]

    _assert_same_oauth_config_except_token(
        hook_config,
        user_config,
        token_key="LEGACY_APP_ACCESS_TOKEN",
        hook_token="hook-token",
        user_token="user-token",
    )


@pytest.mark.asyncio
async def test_hook_supply_launch_config_env_matches_user_oauth_shape(db_session):
    db, user = db_session
    _add_oauth_server(
        db,
        user,
        launch_config=_launch_config(env_key="CUSTOM_ACCESS_TOKEN"),
    )

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        return ResolvedToken(
            access_token="hook-token",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )

    set_oauth_token_resolver_hook(resolver)

    hook_config = (await _tool_config(db, user).get_mcp_server_configs())[0]
    set_oauth_token_resolver_hook(None)
    _add_user_oauth(db, user, provider="google", access_token="user-token")
    user_config = (await _tool_config(db, user).get_mcp_server_configs())[0]

    _assert_same_oauth_config_except_token(
        hook_config,
        user_config,
        token_key="CUSTOM_ACCESS_TOKEN",
        hook_token="hook-token",
        user_token="user-token",
    )


@pytest.mark.asyncio
async def test_hook_supply_instance_url_env_matches_user_oauth_shape(db_session):
    launch_config = {
        "command": "python",
        "args": ["-m", "xagent.web.tools.mcp.salesforce"],
        "env_mapping": {
            "SALESFORCE_ACCESS_TOKEN": "access_token",
            "SALESFORCE_INSTANCE_URL": "instance_url",
        },
    }
    db, user = db_session
    _add_oauth_server(db, user, provider="salesforce", launch_config=launch_config)

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        return ResolvedToken(
            access_token="hook-token",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            instance_url="https://acme.my.salesforce.com",
        )

    set_oauth_token_resolver_hook(resolver)

    hook_config = (await _tool_config(db, user).get_mcp_server_configs())[0]
    set_oauth_token_resolver_hook(None)
    _add_user_oauth(
        db,
        user,
        provider="salesforce",
        access_token="user-token",
        instance_url="https://acme.my.salesforce.com",
    )
    user_config = (await _tool_config(db, user).get_mcp_server_configs())[0]

    assert (
        hook_config["config"]["env"]["SALESFORCE_INSTANCE_URL"]
        == "https://acme.my.salesforce.com"
    )
    _assert_same_oauth_config_except_token(
        hook_config,
        user_config,
        token_key="SALESFORCE_ACCESS_TOKEN",
        hook_token="hook-token",
        user_token="user-token",
    )


@pytest.mark.asyncio
async def test_hook_missing_instance_url_retains_unavailable_server(db_session):
    launch_config = {
        "command": "python",
        "args": ["-m", "xagent.web.tools.mcp.salesforce"],
        "env_mapping": {
            "SALESFORCE_ACCESS_TOKEN": "access_token",
            "SALESFORCE_INSTANCE_URL": "instance_url",
        },
    }
    db, user = db_session
    server = _add_oauth_server(
        db, user, provider="salesforce", launch_config=launch_config
    )

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        return ResolvedToken(
            access_token="hook-token",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )

    set_oauth_token_resolver_hook(resolver)

    config = (await _tool_config(db, user).get_mcp_server_configs())[0]

    _assert_unavailable_mcp_config(
        config, server, reason="oauth_token_required", oauth_token_required=True
    )


@pytest.mark.asyncio
async def test_legacy_missing_instance_url_retains_unavailable_server(db_session):
    """Same as test_hook_missing_instance_url_retains_unavailable_server but
    via the legacy UserOAuth DB path (no resolver hook installed) -- the two
    call sites of _build_oauth_mcp_stdio_transport_config each have their
    own except _OAuthInstanceUrlRequired handler, and only the hook path had
    a regression test for it."""
    launch_config = {
        "command": "python",
        "args": ["-m", "xagent.web.tools.mcp.salesforce"],
        "env_mapping": {
            "SALESFORCE_ACCESS_TOKEN": "access_token",
            "SALESFORCE_INSTANCE_URL": "instance_url",
        },
    }
    db, user = db_session
    server = _add_oauth_server(
        db, user, provider="salesforce", launch_config=launch_config
    )
    _add_user_oauth(db, user, provider="salesforce", access_token="user-token")

    config = (await _tool_config(db, user).get_mcp_server_configs())[0]

    _assert_unavailable_mcp_config(
        config, server, reason="oauth_token_required", oauth_token_required=True
    )


@pytest.mark.asyncio
async def test_launch_config_args_none_matches_user_oauth_shape(db_session):
    db, user = db_session
    launch_config = _launch_config()
    launch_config["args"] = None
    _add_oauth_server(db, user, launch_config=launch_config)

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        return ResolvedToken(
            access_token="hook-token",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )

    set_oauth_token_resolver_hook(resolver)

    hook_config = (await _tool_config(db, user).get_mcp_server_configs())[0]
    set_oauth_token_resolver_hook(None)
    _add_user_oauth(db, user, provider="google", access_token="user-token")
    user_config = (await _tool_config(db, user).get_mcp_server_configs())[0]

    assert hook_config["config"]["args"] == []
    _assert_same_oauth_config_except_token(
        hook_config,
        user_config,
        token_key="GOOGLE_ACCESS_TOKEN",
        hook_token="hook-token",
        user_token="user-token",
    )


@pytest.mark.asyncio
async def test_launch_config_args_string_matches_user_oauth_shape(db_session):
    db, user = db_session
    launch_config = _launch_config()
    launch_config["args"] = '--flag "two words"'
    _add_oauth_server(db, user, launch_config=launch_config)

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        return ResolvedToken(
            access_token="hook-token",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )

    set_oauth_token_resolver_hook(resolver)

    hook_config = (await _tool_config(db, user).get_mcp_server_configs())[0]
    set_oauth_token_resolver_hook(None)
    _add_user_oauth(db, user, provider="google", access_token="user-token")
    user_config = (await _tool_config(db, user).get_mcp_server_configs())[0]

    assert hook_config["config"]["args"] == ["--flag", "two words"]
    _assert_same_oauth_config_except_token(
        hook_config,
        user_config,
        token_key="GOOGLE_ACCESS_TOKEN",
        hook_token="hook-token",
        user_token="user-token",
    )


@pytest.mark.asyncio
async def test_launch_config_args_unsupported_type_warning_matches_contract(
    db_session,
    caplog,
):
    db, user = db_session
    caplog.set_level(logging.WARNING)
    launch_config = _launch_config()
    launch_config["args"] = {"not": "supported"}
    _add_oauth_server(db, user, launch_config=launch_config)

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        return ResolvedToken(
            access_token="hook-token",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )

    set_oauth_token_resolver_hook(resolver)

    config = (await _tool_config(db, user).get_mcp_server_configs())[0]

    assert config["config"]["args"] == []
    assert (
        "Ignoring OAuth MCP launch config args because args must be a list or a string"
        in caplog.text
    )


@pytest.mark.asyncio
async def test_launch_config_env_mapping_none_matches_user_oauth_shape(db_session):
    db, user = db_session
    launch_config = _launch_config()
    launch_config["env_mapping"] = None
    _add_oauth_server(db, user, launch_config=launch_config)

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        return ResolvedToken(
            access_token="hook-token",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )

    set_oauth_token_resolver_hook(resolver)

    hook_config = (await _tool_config(db, user).get_mcp_server_configs())[0]
    set_oauth_token_resolver_hook(None)
    _add_user_oauth(db, user, provider="google", access_token="user-token")
    user_config = (await _tool_config(db, user).get_mcp_server_configs())[0]

    assert "GOOGLE_ACCESS_TOKEN" not in hook_config["config"]["env"]
    assert hook_config == user_config


@pytest.mark.asyncio
async def test_launch_config_missing_command_retains_unavailable_server_for_hook(
    db_session,
    caplog,
):
    db, user = db_session
    caplog.set_level(logging.WARNING)
    _add_stdio_server(db, user, name="before")
    launch_config = _launch_config()
    launch_config.pop("command")
    oauth_server = _add_oauth_server(db, user, launch_config=launch_config)
    _add_stdio_server(db, user, name="after")

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        return ResolvedToken(
            access_token="hook-token",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )

    set_oauth_token_resolver_hook(resolver)

    configs = await _tool_config(db, user).get_mcp_server_configs()

    assert [config["name"] for config in configs] == [
        "before",
        "Google Drive",
        "after",
    ]
    _assert_unavailable_mcp_config(
        configs[1], oauth_server, reason="invalid_launch_config"
    )
    assert "hook-token" not in repr(configs[1])
    assert "launch_config.command is invalid" in caplog.text


@pytest.mark.asyncio
async def test_launch_config_missing_command_retains_unavailable_server_for_user_oauth(
    db_session,
    caplog,
):
    db, user = db_session
    caplog.set_level(logging.WARNING)
    _add_stdio_server(db, user, name="before")
    launch_config = _launch_config()
    launch_config.pop("command")
    oauth_server = _add_oauth_server(db, user, launch_config=launch_config)
    _add_user_oauth(db, user, provider="google", access_token="user-token")
    _add_stdio_server(db, user, name="after")

    configs = await _tool_config(db, user).get_mcp_server_configs()

    assert [config["name"] for config in configs] == [
        "before",
        "Google Drive",
        "after",
    ]
    _assert_unavailable_mcp_config(
        configs[1], oauth_server, reason="invalid_launch_config"
    )
    assert "user-token" not in repr(configs[1])
    assert "launch_config.command is invalid" in caplog.text


@pytest.mark.asyncio
async def test_launch_config_non_mapping_retains_unavailable_server_for_hook(
    db_session,
    caplog,
):
    db, user = db_session
    caplog.set_level(logging.WARNING)
    _add_stdio_server(db, user, name="before")
    oauth_server = _add_oauth_server(db, user, launch_config="not-a-mapping")
    _add_stdio_server(db, user, name="after")

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        return ResolvedToken(
            access_token="hook-token",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )

    set_oauth_token_resolver_hook(resolver)

    configs = await _tool_config(db, user).get_mcp_server_configs()

    assert [config["name"] for config in configs] == [
        "before",
        "Google Drive",
        "after",
    ]
    _assert_unavailable_mcp_config(
        configs[1], oauth_server, reason="invalid_launch_config"
    )
    assert "hook-token" not in repr(configs[1])
    assert "launch_config.type is invalid" in caplog.text


@pytest.mark.asyncio
async def test_launch_config_non_mapping_retains_unavailable_server_for_user_oauth(
    db_session,
    caplog,
):
    db, user = db_session
    caplog.set_level(logging.WARNING)
    _add_stdio_server(db, user, name="before")
    oauth_server = _add_oauth_server(db, user, launch_config=["not", "a", "mapping"])
    _add_user_oauth(db, user, provider="google", access_token="user-token")
    _add_stdio_server(db, user, name="after")

    configs = await _tool_config(db, user).get_mcp_server_configs()

    assert [config["name"] for config in configs] == [
        "before",
        "Google Drive",
        "after",
    ]
    _assert_unavailable_mcp_config(
        configs[1], oauth_server, reason="invalid_launch_config"
    )
    assert "user-token" not in repr(configs[1])
    assert "launch_config.type is invalid" in caplog.text


@pytest.mark.asyncio
async def test_missing_catalog_app_retains_unavailable_oauth_server(db_session):
    db, user = db_session
    server = _add_oauth_server(db, user, name="Unregistered OAuth", register_app=False)
    seen: list[str] = []

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        seen.append(request.provider)
        raise AssertionError("resolver should not be called")

    set_oauth_token_resolver_hook(resolver)

    configs = await _tool_config(db, user).get_mcp_server_configs()

    assert seen == []
    assert len(configs) == 1
    _assert_unavailable_mcp_config(configs[0], server, reason="catalog_app_not_found")


@pytest.mark.asyncio
async def test_invalid_stable_app_id_does_not_fall_back_to_catalog_name(db_session):
    db, user = db_session
    server = _add_oauth_server(db, user, launch_config=_launch_config())
    server.auth = {"app_id": "missing-stable-app"}
    db.commit()

    configs = await _tool_config(db, user).get_mcp_server_configs()

    assert len(configs) == 1
    _assert_unavailable_mcp_config(configs[0], server, reason="catalog_app_not_found")


@pytest.mark.asyncio
async def test_missing_user_oauth_token_retains_unavailable_server_and_later_servers(
    db_session,
):
    db, user = db_session
    oauth_server = _add_oauth_server(db, user, launch_config=_launch_config())
    _add_stdio_server(db, user, name="after")

    configs = await _tool_config(db, user).get_mcp_server_configs()

    assert [config["name"] for config in configs] == ["Google Drive", "after"]
    _assert_unavailable_mcp_config(
        configs[0],
        oauth_server,
        reason="oauth_token_required",
        oauth_token_required=True,
    )


@pytest.mark.asyncio
async def test_unexpected_server_config_failure_retains_failure_and_later_server(
    db_session,
    monkeypatch,
    caplog,
):
    db, user = db_session
    failed_server = _add_stdio_server(db, user, name="before")
    failed_server.env = {"FAIL": "encrypted-secret"}
    _add_stdio_server(db, user, name="after")
    db.commit()

    def fail_first_server(env):
        if env == {"FAIL": "encrypted-secret"}:
            raise RuntimeError("decrypt-secret")
        return {}

    monkeypatch.setattr(
        "xagent.core.utils.encryption.decrypt_env_dict",
        fail_first_server,
    )

    with caplog.at_level(logging.WARNING):
        configs = await _tool_config(db, user).get_mcp_server_configs()

    assert [config["name"] for config in configs] == ["before", "after"]
    _assert_unavailable_mcp_config(
        configs[0],
        failed_server,
        reason="config_load_failed",
    )
    assert configs[1]["config"]["command"] == "echo"
    public_output = repr(configs[0]) + caplog.text
    assert "encrypted-secret" not in public_output
    assert "decrypt-secret" not in public_output
    assert "RuntimeError" in caplog.text


@pytest.mark.asyncio
async def test_user_oauth_refresh_transient_failure_retains_unavailable_without_deleting_record(
    db_session,
    monkeypatch,
):
    """A refresh failure that doesn't confirm the refresh token is dead --
    e.g. a network error, timeout, or provider outage -- must not delete the
    connection: it may well succeed the next time it's used. See
    _OAuthRefreshPermanentlyInvalid.
    """
    db, user = db_session
    oauth_server = _add_oauth_server(db, user, launch_config=_launch_config())
    oauth_account = _add_user_oauth(
        db, user, provider="google", access_token="expired-secret-token"
    )
    account_id = oauth_account.id
    _add_stdio_server(db, user, name="after")
    pending_app = PublicMCPApp(
        app_id="pending-unrelated-app",
        name="Pending unrelated app",
        transport="stdio",
    )
    db.add(pending_app)
    isolated_session_factory = sessionmaker(
        bind=db.get_bind(), autoflush=False, autocommit=False
    )

    async def fail_refresh(*args, **kwargs):
        return False

    monkeypatch.setattr(web_tools_config, "refresh_oauth_token_if_needed", fail_refresh)

    configs = await _tool_config(
        db, user, db_factory=isolated_session_factory
    ).get_mcp_server_configs()

    assert [config["name"] for config in configs] == ["Google Drive", "after"]
    _assert_unavailable_mcp_config(
        configs[0],
        oauth_server,
        reason="oauth_token_refresh_failed",
        oauth_token_required=True,
    )
    assert "expired-secret-token" not in repr(configs[0])
    assert pending_app in db.new
    with isolated_session_factory() as verification_db:
        assert verification_db.get(UserOAuth, account_id) is not None
        assert (
            verification_db.query(PublicMCPApp)
            .filter(PublicMCPApp.app_id == pending_app.app_id)
            .first()
            is None
        )


@pytest.mark.asyncio
async def test_user_oauth_refresh_permanently_invalid_deletes_record(
    db_session,
    monkeypatch,
):
    """Only a confirmed-dead refresh token (_OAuthRefreshPermanentlyInvalid)
    should cost the user their stored connection.
    """
    db, user = db_session
    oauth_server = _add_oauth_server(db, user, launch_config=_launch_config())
    oauth_account = _add_user_oauth(
        db, user, provider="google", access_token="expired-secret-token"
    )
    account_id = oauth_account.id
    isolated_session_factory = sessionmaker(
        bind=db.get_bind(), autoflush=False, autocommit=False
    )

    async def fail_refresh(*args, **kwargs):
        raise web_tools_config._OAuthRefreshPermanentlyInvalid()

    monkeypatch.setattr(web_tools_config, "refresh_oauth_token_if_needed", fail_refresh)

    configs = await _tool_config(
        db, user, db_factory=isolated_session_factory
    ).get_mcp_server_configs()

    assert [config["name"] for config in configs] == ["Google Drive"]
    _assert_unavailable_mcp_config(
        configs[0],
        oauth_server,
        reason="oauth_token_refresh_failed",
        oauth_token_required=True,
    )
    with isolated_session_factory() as verification_db:
        assert verification_db.get(UserOAuth, account_id) is None


@pytest.mark.asyncio
async def test_user_oauth_refresh_generic_invalid_grant_retains_row(
    db_session, monkeypatch
):
    """RFC 6749 section 5.2's invalid_grant also covers a refresh token
    "issued to another client" -- reachable if a shared OAuthProvider
    row's client_id/secret get rotated while an existing UserOAuth row
    still holds a refresh_token issued under the old ones. A generic
    (non-github/slack/meta) invalid_grant must not be trusted as proof
    THIS account's grant is dead end-to-end through the real resolver, or
    one credential rotation would mass-delete every user's connection to
    that provider."""
    db, user = db_session
    db.add(
        OAuthProvider(
            provider_name="google",
            name="Google",
            client_id=encrypt_value("rotated-client-id"),
            client_secret=encrypt_value("rotated-client-secret"),
            auth_url="https://accounts.google.com/o/oauth2/v2/auth",
            token_url="https://oauth2.googleapis.com/token",
        )
    )
    oauth_server = _add_oauth_server(db, user, launch_config=_launch_config())
    oauth_account = _add_user_oauth(
        db, user, provider="google", access_token="old-token"
    )
    oauth_account.refresh_token = "refresh-token-issued-under-old-client"
    oauth_account.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    account_id = oauth_account.id
    db.commit()

    isolated_session_factory = sessionmaker(
        bind=db.get_bind(), autoflush=False, autocommit=False
    )

    class RotatedCredentialAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def post(self, *args, **kwargs):
            return httpx.Response(400, json={"error": "invalid_grant"})

    monkeypatch.setattr(
        web_tools_config.httpx, "AsyncClient", RotatedCredentialAsyncClient
    )

    configs = await _tool_config(
        db, user, db_factory=isolated_session_factory
    ).get_mcp_server_configs()

    assert [config["name"] for config in configs] == ["Google Drive"]
    _assert_unavailable_mcp_config(
        configs[0],
        oauth_server,
        reason="oauth_token_refresh_failed",
        oauth_token_required=True,
    )
    with isolated_session_factory() as verification_db:
        assert verification_db.get(UserOAuth, account_id) is not None


@pytest.mark.asyncio
async def test_user_oauth_refresh_does_not_commit_unrelated_caller_changes(
    db_session,
    monkeypatch,
):
    db, user = db_session
    db.add(
        OAuthProvider(
            provider_name="meta",
            name="Meta",
            client_id=encrypt_value("client-id-secret"),
            client_secret=encrypt_value("client-secret-value"),
            auth_url="https://auth.example/authorize",
            token_url="https://auth.example/token",
        )
    )
    _add_oauth_server(
        db,
        user,
        name="Meta Connector",
        provider="meta",
        launch_config=_launch_config(env_key="META_ACCESS_TOKEN"),
    )
    oauth_account = _add_user_oauth(
        db, user, provider="meta", access_token="expired-access-secret"
    )
    account_id = oauth_account.id
    oauth_account.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    db.commit()
    pending_app = PublicMCPApp(
        app_id="pending-refresh-app",
        name="Pending refresh app",
        transport="stdio",
    )
    db.add(pending_app)
    isolated_session_factory = sessionmaker(
        bind=db.get_bind(), autoflush=False, autocommit=False
    )

    class SuccessfulAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def get(self, *args, **kwargs):
            return httpx.Response(
                200,
                json={"access_token": "refreshed-token", "expires_in": 3600},
            )

    monkeypatch.setattr(
        web_tools_config.httpx,
        "AsyncClient",
        SuccessfulAsyncClient,
    )

    configs = await _tool_config(
        db, user, db_factory=isolated_session_factory
    ).get_mcp_server_configs()

    assert _access_token_env(configs[0], "META_ACCESS_TOKEN") == "refreshed-token"
    assert pending_app in db.new
    with isolated_session_factory() as verification_db:
        refreshed_account = verification_db.get(UserOAuth, account_id)
        assert refreshed_account is not None
        assert refreshed_account.access_token == "refreshed-token"
        assert (
            verification_db.query(PublicMCPApp)
            .filter(PublicMCPApp.app_id == pending_app.app_id)
            .first()
            is None
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ["response", "exception"])
async def test_user_oauth_refresh_failure_logs_only_safe_metadata(
    db_session,
    monkeypatch,
    caplog,
    failure_kind,
):
    db, user = db_session
    response_secret = "response-body-secret-token"
    exception_secret = "transport-exception-secret-token"
    db.add(
        OAuthProvider(
            provider_name="github",
            name="GitHub",
            client_id=encrypt_value("client-id-secret"),
            client_secret=encrypt_value("client-secret-value"),
            auth_url="https://auth.example/authorize",
            token_url="https://auth.example/token",
        )
    )
    oauth_account = _add_user_oauth(
        db, user, provider="github", access_token="expired-access-secret"
    )
    oauth_account.refresh_token = "refresh-secret-value"
    oauth_account.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    db.commit()

    class FailingAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def post(self, *args, **kwargs):
            if failure_kind == "exception":
                raise RuntimeError(exception_secret)
            return httpx.Response(
                400,
                json={
                    "error": "bad_refresh_token",
                    "error_description": response_secret,
                    "access_token": "leaked-response-access-token",
                },
            )

    monkeypatch.setattr(
        web_tools_config.httpx,
        "AsyncClient",
        FailingAsyncClient,
    )

    with caplog.at_level(logging.ERROR):
        if failure_kind == "response":
            # GitHub's own confirmed-dead bad_refresh_token is a permanent
            # failure (see _OAuthRefreshPermanentlyInvalid) and is
            # signalled by raising, not by returning False, so the caller
            # knows to drop the connection instead of retrying it later.
            with pytest.raises(web_tools_config._OAuthRefreshPermanentlyInvalid):
                await web_tools_config.refresh_oauth_token_if_needed(
                    db, oauth_account, "github"
                )
        else:
            is_valid = await web_tools_config.refresh_oauth_token_if_needed(
                db, oauth_account, "github"
            )
            assert is_valid is False

    assert response_secret not in caplog.text
    assert exception_secret not in caplog.text
    assert "leaked-response-access-token" not in caplog.text
    assert "client-secret-value" not in caplog.text
    assert "refresh-secret-value" not in caplog.text


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(400, json={"error": ""}),
        httpx.Response(400, json={}),
        httpx.Response(400, json=[]),
        httpx.Response(400, json="not a mapping"),
        httpx.Response(400, content=b"not json"),
    ],
)
def test_oauth_refresh_error_code_rejects_malformed_bodies(response):
    """An empty/absent `error`, a non-dict JSON body (list/string), or a
    non-JSON body must all resolve to "no error code extracted" rather than
    a falsy-but-truthy value (e.g. "") that would defeat the `error_code or
    "unknown"` logging fallback or accidentally satisfy an `in` check
    against _PROVIDER_DEAD_REFRESH_TOKEN_ERROR_CODES."""
    assert web_tools_config._oauth_refresh_error_code(response, "google") is None


def test_oauth_refresh_error_code_gates_meta_shape_on_provider():
    """Meta's nested-error-object normalization must only apply when the
    caller is actually refreshing a Meta connection -- an unrelated
    provider whose error body happens to carry the same {"type":
    "OAuthException", "code": 190} shape must not be misread as Meta's
    "access token is invalid/expired" signal."""
    response = httpx.Response(
        400, json={"error": {"type": "OAuthException", "code": 190}}
    )
    assert web_tools_config._oauth_refresh_error_code(response, "meta") == (
        "invalid_grant"
    )
    assert web_tools_config._oauth_refresh_error_code(response, "google") is None


@pytest.mark.asyncio
async def test_refresh_unknown_provider_is_transient(db_session):
    """No OAuthProvider row for this provider name is an admin-fixable
    config problem (the row was never created, or was deleted), not
    evidence this account's refresh token is dead -- must not raise
    _OAuthRefreshPermanentlyInvalid, unlike a confirmed-dead token."""
    db, user = db_session
    oauth_account = UserOAuth(
        user_id=user.id,
        provider="not-a-configured-provider",
        access_token="old-token",
        refresh_token="old-refresh",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        provider_user_id="42",
    )
    db.add(oauth_account)
    db.commit()

    assert (
        await web_tools_config.refresh_oauth_token_if_needed(
            db, oauth_account, "not-a-configured-provider"
        )
        is False
    )


@pytest.mark.asyncio
async def test_refresh_missing_provider_credentials_is_transient(
    db_session, monkeypatch
):
    """A provider row with no client_id/client_secret (and no env-var
    fallback) is the same admin-fixable config problem as an unknown
    provider -- must not raise _OAuthRefreshPermanentlyInvalid.

    Deletes the actual fallback env vars (_resolve_oauth_secret /
    _oauth_env_name resolve "google" + "CLIENT_ID" to plain
    GOOGLE_CLIENT_ID, not an XAGENT_-prefixed name) and asserts via an
    exploding AsyncClient that the missing-credentials early return is
    genuinely what's under test -- a real GOOGLE_CLIENT_ID/SECRET present
    in a developer or Docker environment would otherwise resolve real
    credentials and build an unmocked HTTP client, making this pass or
    fail for ambient reasons instead of exercising the code path at all.
    """
    db, user = db_session
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_CLIENT_SECRET", raising=False)
    db.add(
        OAuthProvider(
            provider_name="google",
            name="Google",
            client_id="",
            client_secret="",
            auth_url="https://accounts.google.com/o/oauth2/v2/auth",
            token_url="https://oauth2.googleapis.com/token",
            redirect_uri="https://app.example.com/api/auth/google/callback",
            userinfo_url="https://openidconnect.googleapis.com/v1/userinfo",
            user_id_path="sub",
            email_path="email",
            default_scopes=["email"],
        )
    )
    oauth_account = UserOAuth(
        user_id=user.id,
        provider="google",
        access_token="old-token",
        refresh_token="old-refresh",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        provider_user_id="42",
    )
    db.add(oauth_account)
    db.commit()

    class ExplodingAsyncClient:
        def __call__(self, *args, **kwargs):
            raise AssertionError(
                "refresh_oauth_token_if_needed must not open an HTTP client "
                "when provider credentials are missing"
            )

    monkeypatch.setattr(web_tools_config.httpx, "AsyncClient", ExplodingAsyncClient())

    assert (
        await web_tools_config.refresh_oauth_token_if_needed(
            db, oauth_account, "google"
        )
        is False
    )


@pytest.mark.asyncio
async def test_refresh_5xx_with_oauth_shaped_body_is_transient(db_session, monkeypatch):
    """A 5xx is the provider's own signal that something went wrong on its
    end -- e.g. a proxy/gateway outage, or a misbehaving custom/admin-
    configured token endpoint -- never proof that this specific grant is
    dead. It must stay transient even if its body happens to carry an
    otherwise-recognized permanent error code."""
    db, user = db_session
    db.add(
        OAuthProvider(
            provider_name="google",
            name="Google",
            client_id=encrypt_value("client-id-secret"),
            client_secret=encrypt_value("client-secret-value"),
            auth_url="https://auth.example/authorize",
            token_url="https://auth.example/token",
        )
    )
    oauth_account = _add_user_oauth(
        db, user, provider="google", access_token="expired-access-secret"
    )
    oauth_account.refresh_token = "refresh-secret-value"
    oauth_account.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    db.commit()

    class FiveHundredAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def post(self, *args, **kwargs):
            return httpx.Response(502, json={"error": "invalid_grant"})

    monkeypatch.setattr(
        web_tools_config.httpx,
        "AsyncClient",
        FiveHundredAsyncClient,
    )

    assert (
        await web_tools_config.refresh_oauth_token_if_needed(
            db, oauth_account, "google"
        )
        is False
    )


@pytest.mark.asyncio
async def test_refresh_slack_style_200_invalid_refresh_token_is_permanent(
    db_session, monkeypatch
):
    """Slack's Web API convention -- HTTP 200 with {"ok": false, "error":
    ...} -- applies to its oauth.v2.access refresh grant too (for
    workspaces with token rotation enabled). invalid_refresh_token is
    Slack's non-standard equivalent of invalid_grant/bad_refresh_token and
    must still raise _OAuthRefreshPermanentlyInvalid despite the 200
    status, the same way GitHub's bad_refresh_token does."""
    db, user = db_session
    db.add(
        OAuthProvider(
            provider_name="slack",
            name="Slack",
            client_id=encrypt_value("slack-client-id"),
            client_secret=encrypt_value("slack-client-secret"),
            auth_url="https://slack.com/oauth/v2/authorize",
            token_url="https://slack.com/api/oauth.v2.access",
        )
    )
    oauth_account = UserOAuth(
        user_id=user.id,
        provider="slack",
        access_token="old-token",
        refresh_token="old-refresh",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        provider_user_id="42",
    )
    db.add(oauth_account)
    db.commit()

    class SlackAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def post(self, *args, **kwargs):
            return httpx.Response(
                200, json={"ok": False, "error": "invalid_refresh_token"}
            )

    monkeypatch.setattr(
        web_tools_config.httpx,
        "AsyncClient",
        SlackAsyncClient,
    )

    with pytest.raises(web_tools_config._OAuthRefreshPermanentlyInvalid):
        await web_tools_config.refresh_oauth_token_if_needed(db, oauth_account, "slack")


@pytest.mark.asyncio
async def test_refresh_provider_specific_dead_token_codes_are_gated_by_provider(
    db_session, monkeypatch
):
    """bad_refresh_token/invalid_refresh_token are GitHub's/Slack's own
    non-standard vocabulary, not a generic OAuth2 code -- an unrelated
    provider that happens to return one of these strings for a different,
    non-fatal reason must not be misread as a dead-token signal, the same
    way Meta's differently-shaped error object is gated on provider_name
    rather than trusted for any provider."""
    db, user = db_session
    db.add(
        OAuthProvider(
            provider_name="google",
            name="Google",
            client_id=encrypt_value("client-id-secret"),
            client_secret=encrypt_value("client-secret-value"),
            auth_url="https://auth.example/authorize",
            token_url="https://auth.example/token",
        )
    )
    oauth_account = UserOAuth(
        user_id=user.id,
        provider="google",
        access_token="old-token",
        refresh_token="old-refresh",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        provider_user_id="42",
    )
    db.add(oauth_account)
    db.commit()

    class GoogleAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def post(self, *args, **kwargs):
            return httpx.Response(400, json={"error": "bad_refresh_token"})

    monkeypatch.setattr(
        web_tools_config.httpx,
        "AsyncClient",
        GoogleAsyncClient,
    )

    assert (
        await web_tools_config.refresh_oauth_token_if_needed(
            db, oauth_account, "google"
        )
        is False
    )


@pytest.mark.asyncio
async def test_remote_hook_without_app_info_can_claim_authorization(db_session):
    db, user = db_session
    server = _add_remote_server(db, user, name="Unregistered Remote")
    db.query(PublicMCPApp).filter(PublicMCPApp.name == server.name).delete()
    db.commit()
    seen: list[TokenRequest] = []

    async def resolver(request: TokenRequest) -> ResolvedToken:
        seen.append(request)
        return ResolvedToken(access_token="resolver-token", generation="generation-1")

    set_oauth_token_resolver_hook(resolver)

    configs = await _tool_config(db, user).get_mcp_server_configs()

    assert [request.provider for request in seen] == ["Unregistered Remote"]
    assert configs[0]["config"]["headers"]["Authorization"] == ("Bearer resolver-token")


@pytest.mark.asyncio
async def test_hook_failure_does_not_fallback_and_later_servers_still_build(
    db_session,
    caplog,
):
    db, user = db_session
    caplog.set_level(logging.WARNING)
    _add_stdio_server(db, user, name="before")
    oauth_server = _add_oauth_server(db, user, launch_config=_launch_config())
    _add_user_oauth(db, user, provider="google", access_token="user-token")
    _add_stdio_server(db, user, name="after")

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        raise RuntimeError("secret-token-in-exception")

    set_oauth_token_resolver_hook(resolver)

    cfg = _tool_config(db, user)
    configs = await cfg.get_mcp_server_configs()

    assert [config["name"] for config in configs] == [
        "before",
        "Google Drive",
        "after",
    ]
    unavailable = configs[1]
    assert unavailable["transport"] == "unavailable"
    assert unavailable["config"]["reason"] == "oauth_token_resolver_failed"
    assert unavailable["config"]["server_id"] == oauth_server.id
    assert "secret-token-in-exception" not in str(unavailable)
    assert "secret-token-in-exception" not in caplog.text
    assert "runtime_input_schema" not in unavailable
    assert "runtime_bindings" not in unavailable
    assert "allow_delegated_authorization" not in unavailable
    assert "connector_runtime" not in unavailable
    diagnostics = cfg.get_mcp_oauth_diagnostics()
    assert "actor_id" not in diagnostics[0]
    assert diagnostics == [
        {
            "code": "oauth_token_resolver_failed",
            "message": "OAuth token resolver failed",
            "server_id": oauth_server.id,
            "server_name": "Google Drive",
            "resource_owner_key": None,
            "resource": None,
            "scope": "",
            "issuer": None,
            "providers": ["google", "resolver-google-drive"],
            "exception_type": "RuntimeError",
        }
    ]


@pytest.mark.asyncio
async def test_hook_connector_runtime_error_propagates(db_session):
    db, user = db_session
    _add_oauth_server(db, user, launch_config=_launch_config())
    _add_user_oauth(db, user, provider="google", access_token="user-token")

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        raise ConnectorRuntimeError(
            "connector_runtime_unavailable",
            "Connector runtime context is unavailable.",
            status_code=503,
        )

    set_oauth_token_resolver_hook(resolver)

    cfg = _tool_config(db, user)
    with pytest.raises(ConnectorRuntimeError):
        await cfg.get_mcp_server_configs()

    assert cfg.get_mcp_oauth_diagnostics() == []


@pytest.mark.asyncio
async def test_hook_failure_diagnostic_includes_bounded_resource(db_session):
    db, user = db_session
    resource = "https://mcp.example.com/" + "r" * 160
    _add_oauth_server(db, user, launch_config=_launch_config(resource=resource))

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        raise RuntimeError("resolver failed")

    set_oauth_token_resolver_hook(resolver)

    cfg = _tool_config(db, user)
    configs = await cfg.get_mcp_server_configs()

    assert configs[0]["transport"] == "unavailable"
    diagnostic = cfg.get_mcp_oauth_diagnostics()[0]
    assert diagnostic["resource"] == f"{resource[:125]}..."
    assert len(diagnostic["resource"]) == 128


@pytest.mark.asyncio
async def test_hook_failure_diagnostic_includes_bounded_actor_id_without_secrets(
    db_session,
    caplog,
):
    db, user = db_session
    caplog.set_level(logging.WARNING)
    resource = "https://mcp.example.com/oauth/resource"
    _add_oauth_server(db, user, launch_config=_launch_config(resource=resource))
    actor_id = "actor-" + "x" * 200

    class DelegatedRefreshFailure(RuntimeError):
        oauth_token_resolver_diagnostic_actor_id = actor_id

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        raise DelegatedRefreshFailure("secret-token-in-exception")

    set_oauth_token_resolver_hook(resolver)

    cfg = _tool_config(db, user)
    configs = await cfg.get_mcp_server_configs()

    unavailable = configs[0]
    diagnostic = cfg.get_mcp_oauth_diagnostics()[0]
    assert unavailable["transport"] == "unavailable"
    assert diagnostic["resource"] == resource
    assert diagnostic["actor_id"] == f"{actor_id[:125]}..."
    assert len(diagnostic["actor_id"]) == 128
    assert "secret-token-in-exception" not in str(unavailable)
    assert "secret-token-in-exception" not in diagnostic["message"]
    assert "secret-token-in-exception" not in caplog.text


@pytest.mark.asyncio
async def test_hook_failure_with_raising_actor_attribute_stays_sanitized(
    db_session,
    caplog,
):
    db, user = db_session
    caplog.set_level(logging.WARNING)
    _add_stdio_server(db, user, name="before")
    oauth_server = _add_oauth_server(db, user, launch_config=_launch_config())
    _add_stdio_server(db, user, name="after")

    class AttributeReadFailure(RuntimeError):
        @property
        def oauth_token_resolver_diagnostic_actor_id(self) -> str:
            raise RuntimeError("secret-token-from-diagnostic-property")

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        raise AttributeReadFailure("resolver failed")

    set_oauth_token_resolver_hook(resolver)

    cfg = _tool_config(db, user)
    configs = await cfg.get_mcp_server_configs()

    assert [config["name"] for config in configs] == [
        "before",
        "Google Drive",
        "after",
    ]
    unavailable = configs[1]
    diagnostic = cfg.get_mcp_oauth_diagnostics()[0]
    assert unavailable["transport"] == "unavailable"
    assert unavailable["config"]["server_id"] == oauth_server.id
    assert diagnostic["exception_type"] == "AttributeReadFailure"
    assert "actor_id" not in diagnostic
    assert "secret-token-from-diagnostic-property" not in str(unavailable)
    assert "secret-token-from-diagnostic-property" not in caplog.text


@pytest.mark.asyncio
async def test_hook_failure_with_raising_actor_string_subclass_stays_sanitized(
    db_session,
    caplog,
):
    db, user = db_session
    caplog.set_level(logging.WARNING)
    _add_stdio_server(db, user, name="before")
    oauth_server = _add_oauth_server(db, user, launch_config=_launch_config())
    _add_stdio_server(db, user, name="after")

    class RaisingStr(str):
        def __str__(self) -> str:
            raise RuntimeError("secret-from-string-conversion")

    class ResolverFailure(RuntimeError):
        oauth_token_resolver_diagnostic_actor_id = RaisingStr("actor-1")

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        raise ResolverFailure("resolver failed")

    set_oauth_token_resolver_hook(resolver)

    cfg = _tool_config(db, user)
    configs = await cfg.get_mcp_server_configs()

    assert [config["name"] for config in configs] == [
        "before",
        "Google Drive",
        "after",
    ]
    unavailable = configs[1]
    diagnostic = cfg.get_mcp_oauth_diagnostics()[0]
    assert unavailable["transport"] == "unavailable"
    assert unavailable["config"]["server_id"] == oauth_server.id
    assert diagnostic["exception_type"] == "ResolverFailure"
    assert "actor_id" not in diagnostic
    assert "secret-from-string-conversion" not in str(unavailable)
    assert "secret-from-string-conversion" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("raw_actor_id", [123, 1.5, True, ""])
async def test_hook_failure_drops_non_string_or_empty_actor_id(
    db_session,
    raw_actor_id,
):
    db, user = db_session
    _add_oauth_server(db, user, launch_config=_launch_config())

    class ResolverFailure(RuntimeError):
        oauth_token_resolver_diagnostic_actor_id = raw_actor_id

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        raise ResolverFailure("resolver failed")

    set_oauth_token_resolver_hook(resolver)

    cfg = _tool_config(db, user)
    configs = await cfg.get_mcp_server_configs()

    diagnostic = cfg.get_mcp_oauth_diagnostics()[0]
    assert configs[0]["transport"] == "unavailable"
    assert diagnostic["exception_type"] == "ResolverFailure"
    assert "actor_id" not in diagnostic


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("resolved", "exception_type"),
    [
        (object(), "object"),
        (ResolvedToken(access_token="", expires_at=None), "InvalidAccessToken"),
        (ResolvedToken(access_token=123, expires_at=None), "InvalidAccessToken"),
        (
            ResolvedToken(access_token="hook-token", expires_at="soon"),
            "InvalidExpiresAt",
        ),
        (
            ResolvedToken(access_token="hook-token", instance_url=123),
            "InvalidInstanceUrl",
        ),
        (
            ResolvedToken(access_token="hook-token", instance_url=""),
            "InvalidInstanceUrl",
        ),
    ],
)
async def test_hook_malformed_token_creates_unavailable_config(
    db_session,
    resolved,
    exception_type,
):
    db, user = db_session
    server = _add_oauth_server(db, user, launch_config=_launch_config())

    async def resolver(request: TokenRequest):
        return resolved

    set_oauth_token_resolver_hook(resolver)

    cfg = _tool_config(db, user)
    configs = await cfg.get_mcp_server_configs()

    assert configs[0]["transport"] == "unavailable"
    assert configs[0]["config"]["server_id"] == server.id
    assert cfg.get_mcp_oauth_diagnostics()[0]["exception_type"] == exception_type


@pytest.mark.asyncio
@pytest.mark.parametrize("generation", [None, " generation-v1 "])
async def test_hook_preserves_valid_generation_during_normalization(
    db_session,
    generation,
):
    db, user = db_session

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        return ResolvedToken(
            access_token="access-token-secret",
            expires_at=None,
            generation=generation,
        )

    set_oauth_token_resolver_hook(resolver)

    resolved = await _tool_config(db, user)._resolve_oauth_token_from_hook(
        providers=["google"],
        resource=None,
    )

    assert resolved is not None
    assert resolved.provider == "google"
    assert resolved.access_token == "access-token-secret"
    assert resolved.generation == generation
    assert "access-token-secret" not in repr(resolved)
    if generation is not None:
        assert generation not in repr(resolved)


class _SecretStringGeneration(str):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("generation", "secret_marker"),
    [
        ("", None),
        (
            _SecretStringGeneration("wrong-type-generation-secret"),
            "wrong-type-generation-secret",
        ),
        ("oversized-generation-secret" + "x" * 1024, "oversized-generation-secret"),
    ],
    ids=["empty", "wrong-type", "oversized"],
)
async def test_hook_invalid_generation_creates_sanitized_unavailable_config(
    db_session,
    caplog,
    generation,
    secret_marker,
):
    db, user = db_session
    caplog.set_level(logging.WARNING)
    server = _add_oauth_server(db, user, launch_config=_launch_config())

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        return ResolvedToken(
            access_token="hook-token",
            expires_at=None,
            generation=generation,
        )

    set_oauth_token_resolver_hook(resolver)

    cfg = _tool_config(db, user)
    configs = await cfg.get_mcp_server_configs()

    assert configs[0]["transport"] == "unavailable"
    assert configs[0]["config"]["server_id"] == server.id
    assert cfg.get_mcp_oauth_diagnostics()[0]["exception_type"] == ("InvalidGeneration")
    if secret_marker is not None:
        assert secret_marker not in repr(configs)
        assert secret_marker not in repr(cfg.get_mcp_oauth_diagnostics())
        assert secret_marker not in caplog.text


@pytest.mark.asyncio
async def test_hook_is_skipped_when_user_id_is_none(db_session):
    db, user = db_session
    seen: list[TokenRequest] = []

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        seen.append(request)
        return ResolvedToken(access_token="hook-token", expires_at=None)

    set_oauth_token_resolver_hook(resolver)

    cfg = _tool_config(db, user)
    cfg._user_id = None
    resolved = await cfg._resolve_oauth_token_from_hook(
        providers=["google"],
        resource=None,
    )

    assert resolved is None
    assert seen == []


@pytest.mark.asyncio
async def test_hook_ignores_bare_meta_token_for_app_scoped_facebook(db_session):
    """A resolver-hook token keyed to the bare "meta" provider must not be
    injected for the Facebook connector, mirroring the legacy UserOAuth
    scoping in APPS_REQUIRING_APP_SCOPED_OAUTH_GRANT: the bare provider flow
    never requested Facebook's app-specific oauth_scopes (e.g.
    pages_read_user_content), so it can't stand in for an app-scoped grant.
    """
    db, user = db_session
    server = _add_oauth_server(
        db,
        user,
        name="Facebook Pages",
        app_id="facebook",
        provider="meta",
        launch_config=_launch_config(env_key="META_ACCESS_TOKEN"),
    )
    seen_providers: list[str] = []

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        seen_providers.append(request.provider)
        if request.provider == "meta":
            return ResolvedToken(access_token="bare-meta-hook-token", expires_at=None)
        return None

    set_oauth_token_resolver_hook(resolver)

    cfg = _tool_config(db, user)
    configs = await cfg.get_mcp_server_configs()

    assert seen_providers == ["facebook"]
    _assert_unavailable_mcp_config(
        configs[0], server, reason="oauth_token_required", oauth_token_required=True
    )


@pytest.mark.asyncio
async def test_hook_still_accepts_bare_meta_token_for_instagram(db_session):
    """Negative counterpart to the Facebook filtering test above: Instagram is
    deliberately absent from APPS_REQUIRING_APP_SCOPED_OAUTH_GRANT (its
    required scopes haven't changed), so a resolver-hook token keyed to the
    bare "meta" provider must still be usable. Guards against an
    over-filtering regression in _oauth_token_provider_candidates that only
    the Facebook-side test above wouldn't catch.
    """
    db, user = db_session
    _add_oauth_server(
        db,
        user,
        name="Instagram",
        app_id="instagram",
        provider="meta",
        launch_config=_launch_config(env_key="META_ACCESS_TOKEN"),
    )
    seen_providers: list[str] = []

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        seen_providers.append(request.provider)
        if request.provider == "meta":
            return ResolvedToken(access_token="bare-meta-hook-token", expires_at=None)
        return None

    set_oauth_token_resolver_hook(resolver)

    cfg = _tool_config(db, user)
    configs = await cfg.get_mcp_server_configs()

    assert seen_providers == ["meta"]
    assert _access_token_env(configs[0], key="META_ACCESS_TOKEN") == (
        "bare-meta-hook-token"
    )


@pytest.mark.asyncio
async def test_invalid_launch_config_with_uncacheable_hook_token_is_not_cached(
    db_session,
):
    db, user = db_session
    server = _add_oauth_server(db, user, launch_config="not-a-mapping")
    calls = 0

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        nonlocal calls
        calls += 1
        return ResolvedToken(access_token=f"hook-token-{calls}", expires_at=None)

    set_oauth_token_resolver_hook(resolver)
    cfg = _tool_config(db, user)

    first = await cfg.get_mcp_server_configs()
    app = db.query(PublicMCPApp).filter(PublicMCPApp.name == "Google Drive").one()
    app.launch_config = _launch_config()
    db.commit()
    second = await cfg.get_mcp_server_configs()

    _assert_unavailable_mcp_config(first[0], server, reason="invalid_launch_config")
    assert "hook-token-1" not in repr(first[0])
    assert calls == 2
    assert _access_token_env(second[0]) == "hook-token-2"


@pytest.mark.asyncio
async def test_hook_expires_at_none_is_used_but_not_cached(db_session):
    db, user = db_session
    _add_oauth_server(db, user, launch_config=_launch_config())
    calls = 0

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        nonlocal calls
        calls += 1
        return ResolvedToken(access_token=f"hook-token-{calls}", expires_at=None)

    set_oauth_token_resolver_hook(resolver)
    cfg = _tool_config(db, user)

    first = await cfg.get_mcp_server_configs()
    second = await cfg.get_mcp_server_configs()

    assert calls == 2
    assert _access_token_env(first[0]) == "hook-token-1"
    assert _access_token_env(second[0]) == "hook-token-2"


@pytest.mark.asyncio
async def test_hook_future_expiry_is_cached_until_generation_changes(db_session):
    db, user = db_session
    _add_oauth_server(db, user, launch_config=_launch_config())
    calls = 0

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        nonlocal calls
        calls += 1
        return ResolvedToken(
            access_token=f"hook-token-{calls}",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )

    set_oauth_token_resolver_hook(resolver)
    cfg = _tool_config(db, user)

    first = await cfg.get_mcp_server_configs()
    second = await cfg.get_mcp_server_configs()
    set_oauth_token_resolver_hook(resolver)
    third = await cfg.get_mcp_server_configs()

    assert calls == 2
    assert _access_token_env(first[0]) == "hook-token-1"
    assert _access_token_env(second[0]) == "hook-token-1"
    assert _access_token_env(third[0]) == "hook-token-2"


@pytest.mark.asyncio
async def test_hook_naive_expiry_is_interpreted_as_utc(db_session):
    db, user = db_session
    _add_oauth_server(db, user, launch_config=_launch_config())

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        return ResolvedToken(
            access_token="hook-token",
            expires_at=datetime.now(timezone.utc).replace(tzinfo=None)
            + timedelta(hours=1),
        )

    set_oauth_token_resolver_hook(resolver)

    configs = await _tool_config(db, user).get_mcp_server_configs()

    assert _access_token_env(configs[0]) == "hook-token"


@pytest.mark.asyncio
async def test_hook_near_expiry_token_is_used_but_not_cached(db_session):
    db, user = db_session
    _add_oauth_server(db, user, launch_config=_launch_config())
    calls = 0

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        nonlocal calls
        calls += 1
        return ResolvedToken(
            access_token=f"hook-token-{calls}",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=1),
        )

    set_oauth_token_resolver_hook(resolver)
    cfg = _tool_config(db, user)

    first = await cfg.get_mcp_server_configs()
    second = await cfg.get_mcp_server_configs()

    assert calls == 2
    assert _access_token_env(first[0]) == "hook-token-1"
    assert _access_token_env(second[0]) == "hook-token-2"


@pytest.mark.asyncio
async def test_hook_expired_token_creates_unavailable_config(db_session):
    db, user = db_session
    server = _add_oauth_server(db, user, launch_config=_launch_config())

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        return ResolvedToken(
            access_token="hook-token",
            expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        )

    set_oauth_token_resolver_hook(resolver)

    cfg = _tool_config(db, user)
    configs = await cfg.get_mcp_server_configs()

    assert configs[0]["transport"] == "unavailable"
    assert configs[0]["config"]["server_id"] == server.id
    assert cfg.get_mcp_oauth_diagnostics()[0]["exception_type"] == (
        "ExpiredAccessToken"
    )


@pytest.mark.asyncio
async def test_hook_request_receives_execution_scope(db_session):
    db, user = db_session
    scope = object()
    _add_oauth_server(db, user, launch_config=_launch_config())
    seen_scope: list[object] = []

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        seen_scope.append(request.scope)
        return ResolvedToken(
            access_token="hook-token",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )

    set_oauth_token_resolver_hook(resolver)

    configs = await _tool_config(
        db, user, execution_scope=scope
    ).get_mcp_server_configs()

    assert seen_scope == [scope]
    assert _access_token_env(configs[0]) == "hook-token"


def _remote_runtime_bindings() -> list[dict]:
    return [
        {
            "source": {"input_type": "secrets", "key": "runtime_header"},
            "target": {
                "target_type": "transport_headers",
                "key": "X-Runtime",
            },
        },
        {
            "source": {"input_type": "secrets", "key": "authorization"},
            "target": {
                "target_type": "transport_headers",
                "key": "Authorization",
            },
        },
    ]


def _challenge() -> MCPAuthorizationChallenge:
    return MCPAuthorizationChallenge(
        resource_metadata_url=(
            "https://mcp.example/.well-known/oauth-protected-resource"
        ),
        scope="records.read",
        params={},
    )


@pytest.mark.asyncio
async def test_remote_without_hook_skips_resolver_candidate_work(
    db_session,
    monkeypatch,
):
    db, user = db_session
    _add_remote_server(
        db,
        user,
        headers={"X-Static": "static", "Authorization": "Bearer static-token"},
    )

    def unexpected_resolver_work(*args, **kwargs):
        pytest.fail("resolver-specific work ran without a registered hook")

    monkeypatch.setattr(
        "xagent.web.mcp_apps.get_app_for_mcp_server", unexpected_resolver_work
    )
    monkeypatch.setattr(
        "xagent.web.services.mcp_runtime.effective_mcp_oauth_resource",
        unexpected_resolver_work,
    )

    configs = await _tool_config(db, user).get_mcp_server_configs()

    assert configs[0]["config"]["headers"] == {
        "X-Static": "static",
        "Authorization": "Bearer static-token",
    }


@pytest.mark.asyncio
async def test_remote_hook_future_expiry_is_cached_until_registration_changes(
    db_session,
):
    db, user = db_session
    _add_remote_server(db, user)
    calls = 0

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        nonlocal calls
        calls += 1
        return ResolvedToken(
            access_token=f"remote-hook-token-{calls}",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            generation=f"remote-generation-{calls}",
        )

    set_oauth_token_resolver_hook(resolver)
    cfg = _tool_config(db, user)

    first = await cfg.get_mcp_server_configs()
    second = await cfg.get_mcp_server_configs()
    set_oauth_token_resolver_hook(resolver)
    third = await cfg.get_mcp_server_configs()

    assert calls == 2
    assert first[0]["config"]["headers"]["Authorization"] == (
        "Bearer remote-hook-token-1"
    )
    assert second[0]["config"]["headers"]["Authorization"] == (
        "Bearer remote-hook-token-1"
    )
    assert third[0]["config"]["headers"]["Authorization"] == (
        "Bearer remote-hook-token-2"
    )


@pytest.mark.asyncio
async def test_remote_hook_near_expiry_token_is_used_but_not_cached(db_session):
    db, user = db_session
    _add_remote_server(db, user)
    calls = 0

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        nonlocal calls
        calls += 1
        return ResolvedToken(
            access_token=f"remote-hook-token-{calls}",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=1),
            generation=f"remote-generation-{calls}",
        )

    set_oauth_token_resolver_hook(resolver)
    cfg = _tool_config(db, user)

    first = await cfg.get_mcp_server_configs()
    second = await cfg.get_mcp_server_configs()

    assert calls == 2
    assert first[0]["config"]["headers"]["Authorization"] == (
        "Bearer remote-hook-token-1"
    )
    assert second[0]["config"]["headers"]["Authorization"] == (
        "Bearer remote-hook-token-2"
    )


@pytest.mark.asyncio
async def test_remote_hook_owns_connection_and_preserves_non_auth_snapshot(
    db_session,
    caplog,
):
    db, user = db_session
    scope = object()
    server = _add_remote_server(
        db,
        user,
        auth={"type": "mcp_oauth", "resource": "https://auth.example/resource"},
        headers={"X-Static": "static", "authorization": "Bearer static-token"},
        runtime_bindings=_remote_runtime_bindings(),
        allow_delegated_authorization=True,
    )
    requests: list[TokenRequest] = []

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        requests.append(request)
        return ResolvedToken(
            access_token="resolver-token",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            generation="generation-1",
        )

    set_oauth_token_resolver_hook(resolver)
    cfg = _tool_config(
        db,
        user,
        execution_scope=scope,
        mcp_auth_context={
            str(server.id): {"resource": " https://selector.example/resource "}
        },
    )
    cfg._connector_runtime_view = {
        f"mcp:{server.id}": {
            "context": {"account_id": "account-1"},
            "secrets": {
                "runtime_header": "runtime",
                "authorization": "Bearer delegated-token",
            },
            "auth_selector": {},
        }
    }

    with caplog.at_level("WARNING"):
        configs = await cfg.get_mcp_server_configs()

    assert [
        (request.provider, request.resource, request.scope) for request in requests
    ] == [("records", " https://selector.example/resource ", scope)]
    assert configs[0]["config"]["headers"] == {
        "X-Static": "static",
        "X-Runtime": "runtime",
        "Authorization": "Bearer resolver-token",
    }
    assert callable(configs[0]["config"]["_oauth_token_resolver_refresh"])
    assert "_connector_runtime_refresh" not in configs[0]["config"]
    assert configs[0]["connector_runtime"] == {
        "context": {"account_id": "account-1"},
        "secrets": {},
        "auth_selector": {},
    }
    assert configs[0]["runtime_bindings"] == _remote_runtime_bindings()
    assert "delegated authorization is disabled" not in caplog.text


@pytest.mark.asyncio
async def test_remote_hook_checks_all_candidates_before_existing_plain_path(
    db_session,
):
    db, user = db_session
    _add_remote_server(
        db,
        user,
        headers={"X-Static": "static", "Authorization": "Bearer static-token"},
    )
    seen: list[str] = []

    async def resolver(request: TokenRequest) -> None:
        seen.append(request.provider)
        return None

    set_oauth_token_resolver_hook(resolver)

    configs = await _tool_config(db, user).get_mcp_server_configs()

    assert seen == ["records", "remote-records"]
    assert configs[0]["config"]["headers"] == {
        "X-Static": "static",
        "Authorization": "Bearer static-token",
    }
    assert "_oauth_token_resolver_refresh" not in configs[0]["config"]
    assert "_connector_runtime_refresh" not in configs[0]["config"]


@pytest.mark.asyncio
async def test_remote_hook_decline_preserves_delegated_runtime_path(db_session):
    db, user = db_session
    server = _add_remote_server(
        db,
        user,
        headers={"X-Static": "static"},
        runtime_bindings=_remote_runtime_bindings(),
        allow_delegated_authorization=True,
    )
    seen: list[str] = []

    async def resolver(request: TokenRequest) -> None:
        seen.append(request.provider)
        return None

    set_oauth_token_resolver_hook(resolver)
    cfg = _tool_config(db, user)
    cfg._connector_runtime_view = {
        f"mcp:{server.id}": {
            "context": {},
            "secrets": {
                "runtime_header": "runtime",
                "authorization": "Bearer delegated-token",
            },
            "auth_selector": {},
        }
    }

    configs = await cfg.get_mcp_server_configs()

    assert seen == ["records", "remote-records"]
    assert configs[0]["config"]["headers"]["Authorization"] == (
        "Bearer delegated-token"
    )
    assert callable(configs[0]["config"]["_connector_runtime_refresh"])
    assert "_oauth_token_resolver_refresh" not in configs[0]["config"]


@pytest.mark.asyncio
async def test_remote_hook_decline_preserves_standard_oauth_path(db_session):
    db, user = db_session
    server = _add_remote_server(
        db,
        user,
        auth={"type": "mcp_oauth", "resource": "https://auth.example/resource"},
    )
    seen: list[str] = []

    async def resolver(request: TokenRequest) -> None:
        seen.append(request.provider)
        return None

    set_oauth_token_resolver_hook(resolver)
    cfg = _tool_config(db, user)

    configs = await cfg.get_mcp_server_configs()

    assert seen == ["records", "remote-records"]
    _assert_unavailable_mcp_config(
        configs[0],
        server,
        reason="authorization_required",
        oauth_token_required=True,
    )
    assert cfg.get_mcp_oauth_diagnostics()[0]["code"] == "authorization_required"


@pytest.mark.asyncio
async def test_remote_runtime_connection_exception_retains_safe_unavailable_config(
    db_session,
    monkeypatch,
    caplog,
):
    db, user = db_session
    _add_stdio_server(db, user, name="before")
    remote_server = _add_remote_server(
        db,
        user,
        auth={"type": "mcp_oauth", "resource": "https://auth.example/resource"},
        headers={"Authorization": "Bearer static-secret"},
    )
    _add_stdio_server(db, user, name="after")

    async def fail_runtime_connection(*args, **kwargs):
        raise RuntimeError("runtime-connection-secret")

    monkeypatch.setattr(
        "xagent.web.services.mcp_runtime.build_mcp_runtime_connection",
        fail_runtime_connection,
    )

    with caplog.at_level(logging.WARNING):
        configs = await _tool_config(db, user).get_mcp_server_configs()

    assert [config["name"] for config in configs] == [
        "before",
        "Remote Records",
        "after",
    ]
    _assert_unavailable_mcp_config(
        configs[1], remote_server, reason="runtime_connection_failed"
    )
    public_output = repr(configs[1]) + caplog.text
    assert "runtime-connection-secret" not in public_output
    assert "static-secret" not in public_output


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("resolved", "exception_type"),
    [
        (object(), "object"),
        (
            ResolvedToken(access_token="", generation="generation-1"),
            "InvalidAccessToken",
        ),
    ],
)
async def test_remote_hook_malformed_initial_result_fails_closed(
    db_session,
    resolved,
    exception_type,
):
    db, user = db_session
    server = _add_remote_server(
        db,
        user,
        headers={"Authorization": "Bearer static-token"},
        auth={"type": "bearer", "bearer_token": "fallback-token"},
    )

    async def resolver(request: TokenRequest):
        return resolved

    set_oauth_token_resolver_hook(resolver)
    cfg = _tool_config(db, user)

    configs = await cfg.get_mcp_server_configs()

    assert configs[0]["transport"] == "unavailable"
    assert configs[0]["config"]["server_id"] == server.id
    assert cfg.get_mcp_oauth_diagnostics()[0]["exception_type"] == exception_type
    assert "fallback-token" not in repr(configs)


@pytest.mark.asyncio
async def test_remote_hook_initial_raise_fails_closed(db_session):
    db, user = db_session
    _add_remote_server(
        db,
        user,
        headers={"Authorization": "Bearer static-token"},
    )

    async def resolver(request: TokenRequest):
        raise RuntimeError("resolver-secret")

    set_oauth_token_resolver_hook(resolver)
    cfg = _tool_config(db, user)

    configs = await cfg.get_mcp_server_configs()

    assert configs[0]["transport"] == "unavailable"
    assert "resolver-secret" not in repr(configs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw_failure_code", "expected_failure_code"),
    [
        ("oauth_token_required", "oauth_token_required"),
        ("other_valid_code", None),
        (" oauth_token_required", None),
        (123, None),
    ],
)
async def test_remote_hook_preserves_only_allowlisted_resolver_failure_code(
    db_session,
    raw_failure_code,
    expected_failure_code,
):
    db, user = db_session
    _add_remote_server(db, user)

    class ResolverFailure(RuntimeError):
        oauth_token_resolver_failure_code = raw_failure_code

    async def resolver(request: TokenRequest):
        raise ResolverFailure("resolver-internal-secret")

    set_oauth_token_resolver_hook(resolver)
    cfg = _tool_config(db, user)

    configs = await cfg.get_mcp_server_configs()

    unavailable_config = configs[0]["config"]
    if expected_failure_code is None:
        assert "failure_code" not in unavailable_config
    else:
        assert unavailable_config["failure_code"] == expected_failure_code
    assert "failure_code" not in cfg.get_mcp_oauth_diagnostics()[0]
    assert "resolver-internal-secret" not in repr(configs)


@pytest.mark.asyncio
async def test_remote_hook_failure_code_property_error_is_sanitized(db_session):
    db, user = db_session
    _add_remote_server(db, user)

    class ResolverFailure(RuntimeError):
        @property
        def oauth_token_resolver_failure_code(self):
            raise RuntimeError("failure-code-property-secret")

    async def resolver(request: TokenRequest):
        raise ResolverFailure("resolver-internal-secret")

    set_oauth_token_resolver_hook(resolver)
    cfg = _tool_config(db, user)

    configs = await cfg.get_mcp_server_configs()

    assert configs[0]["transport"] == "unavailable"
    assert "failure_code" not in configs[0]["config"]
    public_output = repr(configs) + repr(cfg.get_mcp_oauth_diagnostics())
    assert "failure-code-property-secret" not in public_output
    assert "resolver-internal-secret" not in public_output


@pytest.mark.asyncio
async def test_remote_hook_consecutive_refreshes_advance_failed_generation(db_session):
    db, user = db_session
    server = _add_remote_server(
        db,
        user,
        headers={"X-Static": "static", "Authorization": "Bearer static-token"},
    )
    requests: list[TokenRequest] = []

    async def resolver(request: TokenRequest) -> ResolvedToken:
        requests.append(request)
        if request.refresh is None:
            return ResolvedToken(
                access_token="initial-token", generation="generation-1"
            )
        return ResolvedToken(access_token="refreshed-token", generation="generation-2")

    set_oauth_token_resolver_hook(resolver)
    configs = await _tool_config(db, user).get_mcp_server_configs()
    refresh = configs[0]["config"]["_oauth_token_resolver_refresh"]

    refreshed = await refresh(_challenge())

    assert requests[1].provider == requests[0].provider == "records"
    assert requests[1].resource == requests[0].resource == server.url
    assert requests[1].refresh == web_tools_config.OAuthRefreshContext(
        reason="invalid_token",
        resource_metadata_url=(
            "https://mcp.example/.well-known/oauth-protected-resource"
        ),
        challenge_scope="records.read",
        failed_generation="generation-1",
    )
    assert refreshed["headers"] == {
        "X-Static": "static",
        "Authorization": "Bearer refreshed-token",
    }
    assert "auth" not in refreshed
    next_refresh = refreshed["_oauth_token_resolver_refresh"]
    assert next_refresh is not refresh
    assert "_connector_runtime_refresh" not in refreshed

    assert await next_refresh(_challenge()) is None
    assert requests[2].refresh == web_tools_config.OAuthRefreshContext(
        reason="invalid_token",
        resource_metadata_url=(
            "https://mcp.example/.well-known/oauth-protected-resource"
        ),
        challenge_scope="records.read",
        failed_generation="generation-2",
    )


@pytest.mark.asyncio
async def test_delegated_remote_evaluates_bindings_once_without_false_warning(
    db_session,
    monkeypatch,
    caplog,
):
    db, user = db_session
    server = _add_remote_server(
        db,
        user,
        runtime_bindings=_remote_runtime_bindings(),
        allow_delegated_authorization=True,
    )
    evaluations = 0
    original = web_tools_config.runtime_bindings_from_config

    def counted_runtime_bindings_from_config(config):
        nonlocal evaluations
        evaluations += 1
        return original(config)

    monkeypatch.setattr(
        web_tools_config,
        "runtime_bindings_from_config",
        counted_runtime_bindings_from_config,
    )
    cfg = _tool_config(db, user)
    cfg._connector_runtime_view = {
        f"mcp:{server.id}": {
            "context": {},
            "secrets": {
                "runtime_header": "runtime",
                "authorization": "Bearer delegated-token",
            },
            "auth_selector": {},
        }
    }

    with caplog.at_level("WARNING"):
        configs = await cfg.get_mcp_server_configs()

    assert evaluations == 1
    assert configs[0]["config"]["headers"] == {
        "X-Runtime": "runtime",
        "Authorization": "Bearer delegated-token",
    }
    assert "delegated authorization is disabled" not in caplog.text


@pytest.mark.asyncio
async def test_remote_hook_refresh_reuses_captured_non_auth_runtime_snapshot(
    db_session,
):
    db, user = db_session
    server = _add_remote_server(
        db,
        user,
        headers={"X-Static": "static"},
        runtime_bindings=_remote_runtime_bindings(),
    )

    async def resolver(request: TokenRequest) -> ResolvedToken:
        if request.refresh is None:
            return ResolvedToken(
                access_token="initial-token", generation="generation-1"
            )
        return ResolvedToken(access_token="refreshed-token", generation="generation-2")

    set_oauth_token_resolver_hook(resolver)
    cfg = _tool_config(db, user)
    cfg._connector_runtime_view = {
        f"mcp:{server.id}": {
            "context": {},
            "secrets": {"runtime_header": "captured-runtime"},
            "auth_selector": {},
        }
    }
    configs = await cfg.get_mcp_server_configs()
    refresh = configs[0]["config"]["_oauth_token_resolver_refresh"]
    cfg._connector_runtime_view = {
        f"mcp:{server.id}": {
            "context": {},
            "secrets": {"runtime_header": "later-runtime"},
            "auth_selector": {},
        }
    }

    refreshed = await refresh(_challenge())

    assert refreshed["headers"]["X-Runtime"] == "captured-runtime"


@pytest.mark.asyncio
@pytest.mark.parametrize("registration_action", ["same", "replace", "clear"])
async def test_remote_hook_refresh_rejects_changed_registration(
    db_session,
    registration_action,
):
    db, user = db_session
    _add_remote_server(db, user)
    calls = 0

    async def resolver(request: TokenRequest) -> ResolvedToken:
        nonlocal calls
        calls += 1
        return ResolvedToken(
            access_token=f"token-{calls}", generation=f"generation-{calls}"
        )

    set_oauth_token_resolver_hook(resolver)
    configs = await _tool_config(db, user).get_mcp_server_configs()
    refresh = configs[0]["config"]["_oauth_token_resolver_refresh"]

    if registration_action == "same":
        set_oauth_token_resolver_hook(resolver)
    elif registration_action == "replace":
        set_oauth_token_resolver_hook(lambda request: None)
    else:
        set_oauth_token_resolver_hook(None)

    assert await refresh(_challenge()) is None
    assert calls == 1


@pytest.mark.asyncio
async def test_remote_hook_refresh_rejects_registration_change_during_await(
    db_session,
):
    db, user = db_session
    _add_remote_server(db, user)
    refresh_started = asyncio.Event()
    release_refresh = asyncio.Event()

    async def resolver(request: TokenRequest) -> ResolvedToken:
        if request.refresh is None:
            return ResolvedToken(
                access_token="initial-token", generation="generation-1"
            )
        refresh_started.set()
        await release_refresh.wait()
        return ResolvedToken(access_token="refreshed-token", generation="generation-2")

    set_oauth_token_resolver_hook(resolver)
    configs = await _tool_config(db, user).get_mcp_server_configs()
    refresh = configs[0]["config"]["_oauth_token_resolver_refresh"]

    pending = asyncio.create_task(refresh(_challenge()))
    await refresh_started.wait()
    set_oauth_token_resolver_hook(resolver)
    release_refresh.set()

    assert await pending is None


@pytest.mark.asyncio
async def test_remote_hook_refresh_classification_rejects_registration_change_during_await(
    db_session,
):
    db, user = db_session
    _add_remote_server(db, user)
    refresh_started = asyncio.Event()
    release_refresh = asyncio.Event()

    class RefreshFailure(RuntimeError):
        oauth_token_resolver_failure_code = "oauth_token_required"

    async def resolver(request: TokenRequest):
        if request.refresh is None:
            return ResolvedToken(
                access_token="initial-token", generation="generation-1"
            )
        refresh_started.set()
        await release_refresh.wait()
        raise RefreshFailure("refresh-private-secret")

    set_oauth_token_resolver_hook(resolver)
    configs = await _tool_config(db, user).get_mcp_server_configs()
    refresh = configs[0]["config"]["_oauth_token_resolver_refresh"]

    pending = asyncio.create_task(refresh(_challenge()))
    await refresh_started.wait()
    set_oauth_token_resolver_hook(resolver)
    release_refresh.set()

    assert await pending is None


@pytest.mark.asyncio
async def test_remote_hook_refresh_unknown_failure_code_remains_unclassified(
    db_session,
):
    db, user = db_session
    _add_remote_server(db, user)

    class RefreshFailure(RuntimeError):
        oauth_token_resolver_failure_code = "other_valid_code"

    async def resolver(request: TokenRequest):
        if request.refresh is None:
            return ResolvedToken(
                access_token="initial-token", generation="generation-1"
            )
        raise RefreshFailure("refresh-private-secret")

    set_oauth_token_resolver_hook(resolver)
    configs = await _tool_config(db, user).get_mcp_server_configs()
    refresh = configs[0]["config"]["_oauth_token_resolver_refresh"]

    assert await refresh(_challenge()) is None


@pytest.mark.asyncio
async def test_refresh_failure_ignores_non_oauth_runtime_failure_code(
    db_session,
):
    """A runtime-allowlisted code that isn't OAuth-specific must not raise.

    ``ClassifiedToolFailure`` now validates against the OAuth-only set, not
    the wider runtime allowlist. A resolver that raises with a non-OAuth
    runtime code (allowlisted for other consumers) must fall through to
    ``None`` here rather than let a ``ValueError`` escape the handler.
    """
    db, user = db_session
    _add_remote_server(db, user)

    class RefreshFailure(RuntimeError):
        oauth_token_resolver_failure_code = "unsupported_nested_interaction"

    async def resolver(request: TokenRequest):
        if request.refresh is None:
            return ResolvedToken(
                access_token="initial-token", generation="generation-1"
            )
        raise RefreshFailure("refresh-private-secret")

    set_oauth_token_resolver_hook(resolver)
    configs = await _tool_config(db, user).get_mcp_server_configs()
    refresh = configs[0]["config"]["_oauth_token_resolver_refresh"]

    assert await refresh(_challenge()) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "refresh_result",
    [
        None,
        object(),
        ResolvedToken(access_token="refreshed", generation=None),
        ResolvedToken(access_token="refreshed", generation="generation-1"),
        ResolvedToken(access_token="refreshed", generation=""),
        ResolvedToken(
            access_token="refreshed",
            generation="generation-2",
            expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        ),
    ],
    ids=[
        "none",
        "invalid",
        "missing-generation",
        "unchanged",
        "invalid-generation",
        "expired",
    ],
)
async def test_remote_hook_refresh_invalid_results_fail_closed(
    db_session,
    refresh_result,
):
    db, user = db_session
    _add_remote_server(
        db,
        user,
        headers={"Authorization": "Bearer static-token"},
    )

    async def resolver(request: TokenRequest):
        if request.refresh is None:
            return ResolvedToken(
                access_token="initial-token", generation="generation-1"
            )
        return refresh_result

    set_oauth_token_resolver_hook(resolver)
    configs = await _tool_config(db, user).get_mcp_server_configs()
    refresh = configs[0]["config"]["_oauth_token_resolver_refresh"]

    assert await refresh(_challenge()) is None


@pytest.mark.asyncio
async def test_remote_hook_missing_initial_generation_does_not_call_refresh_resolver(
    db_session,
):
    db, user = db_session
    _add_remote_server(db, user)
    refresh_calls = 0

    async def resolver(request: TokenRequest) -> ResolvedToken:
        nonlocal refresh_calls
        if request.refresh is None:
            return ResolvedToken(access_token="initial-token")
        refresh_calls += 1
        return ResolvedToken(access_token="refreshed-token", generation="generation-2")

    set_oauth_token_resolver_hook(resolver)
    configs = await _tool_config(db, user).get_mcp_server_configs()
    refresh = configs[0]["config"]["_oauth_token_resolver_refresh"]

    assert await refresh(_challenge()) is None
    assert refresh_calls == 0


@pytest.mark.asyncio
async def test_remote_hook_refresh_resolver_raise_has_no_authorization_fallback(
    db_session,
):
    db, user = db_session
    _add_remote_server(
        db,
        user,
        headers={"Authorization": "Bearer static-token"},
    )

    async def resolver(request: TokenRequest):
        if request.refresh is None:
            return ResolvedToken(
                access_token="initial-token", generation="generation-1"
            )
        raise RuntimeError("refresh-secret")

    set_oauth_token_resolver_hook(resolver)
    configs = await _tool_config(db, user).get_mcp_server_configs()
    refresh = configs[0]["config"]["_oauth_token_resolver_refresh"]

    assert await refresh(_challenge()) is None


@pytest.mark.asyncio
async def test_remote_hook_refresh_classification_reaches_tool_failure_trace(
    db_session,
    monkeypatch,
    caplog,
):
    db, user = db_session
    _add_remote_server(db, user)
    private_exception_text = "refresh-private-exception-secret"

    class RefreshFailure(RuntimeError):
        oauth_token_resolver_failure_code = "oauth_token_required"

    async def resolver(request: TokenRequest):
        if request.refresh is None:
            return ResolvedToken(
                access_token="initial-token", generation="generation-1"
            )
        raise RefreshFailure(private_exception_text)

    set_oauth_token_resolver_hook(resolver)
    configs = await _tool_config(db, user).get_mcp_server_configs()
    connection = {
        "transport": configs[0]["transport"],
        **configs[0]["config"],
    }
    adapter = MCPToolAdapter(
        mcp_tool=SimpleNamespace(
            name="list_records",
            description="List records",
            inputSchema={"type": "object", "properties": {}},
        ),
        connection=connection,
        allow_users=configs[0]["allow_users"],
    )
    request = httpx.Request("POST", "https://mcp.example/api")
    response = httpx.Response(
        401,
        headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
        request=request,
    )
    execution_error = httpx.HTTPStatusError(
        "http-private-exception-secret",
        request=request,
        response=response,
    )

    async def execute(connection, tool_args, tool_meta):
        raise execution_error

    monkeypatch.setattr(adapter, "_execute_mcp_call", execute)
    monkeypatch.setenv("XAGENT_USER_ID", str(user.id))
    caplog.set_level("ERROR")

    result = await adapter.run_json_async({})

    assert result["is_error"] is True
    assert result["failure_code"] == "oauth_token_required"
    assert "delegated_authorization_failed" in result["content"][0]["text"]

    class CapturingTracer:
        def __init__(self) -> None:
            self.events: list[dict[str, Any]] = []

        async def trace_event(self, event_type: Any, **kwargs: Any) -> None:
            self.events.append(
                {
                    "type": getattr(event_type, "value", str(event_type)),
                    "data": kwargs.get("data") or {},
                }
            )

    tracer = CapturingTracer()
    runtime = PatternRuntime(tracer=tracer, execution_id="task-refresh-required")
    await runtime.on_tool_end(
        tool_call={"name": adapter.name, "id": "call-1"},
        result=result,
    )

    assert tracer.events[0]["type"] == "action_error_tool"
    assert tracer.events[0]["data"]["failure_code"] == "oauth_token_required"
    public_output = repr(result) + repr(tracer.events) + caplog.text
    assert private_exception_text not in public_output
    assert "http-private-exception-secret" not in public_output
