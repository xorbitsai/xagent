from __future__ import annotations

from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Query, sessionmaker

from xagent.core.utils.encryption import decrypt_value, encrypt_value
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.mcp_oauth import MCPOAuthClient, MCPOAuthGrant
from xagent.web.models.user import User
from xagent.web.services import mcp_oauth as mcp_oauth_service
from xagent.web.services.mcp_runtime import (
    connection_with_bearer_authorization,
    connection_without_authorization,
    effective_mcp_oauth_resource,
)
from xagent.web.tools.config import WebToolConfig


@pytest.fixture()
def db_session(tmp_path):
    db_path = tmp_path / "mcp-oauth-runtime.db"
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = SessionLocal()

    user = User(username="alice", password_hash="x", is_admin=False)
    other_user = User(username="bob", password_hash="x", is_admin=False)
    db.add_all([user, other_user])
    db.commit()
    db.refresh(user)
    db.refresh(other_user)

    yield db, user, other_user
    db.close()
    engine.dispose()


def _add_mcp_oauth_server(
    db,
    user: User,
    *,
    include_scope: bool = True,
    resource: str = "https://mcp.example.com/mcp",
    issuer: str = "https://auth.example.com",
) -> MCPServer:
    auth_config = {
        "type": "mcp_oauth",
        "resource": resource,
        "issuer": issuer,
        "client_id": "client-123",
    }
    if include_scope:
        auth_config["scope"] = "records.read"
    server = MCPServer.from_config(
        {
            "name": "records-runtime",
            "managed": "external",
            "transport": "streamable_http",
            "url": "https://mcp.example.com/mcp",
            "headers": {
                "X-Request-Source": "xagent",
                "Authorization": "Bearer static-token",
            },
            "auth": auth_config,
        }
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


def _add_grant(
    db,
    *,
    server: MCPServer,
    user: User,
    resource_owner_key: str,
    access_token: str,
    refresh_token: str | None = None,
    expires_at: datetime | None = None,
    scope: str = "records.read",
    status: str = "active",
) -> MCPOAuthGrant:
    oauth_client = (
        db.query(MCPOAuthClient)
        .filter(
            MCPOAuthClient.mcp_server_id == server.id,
            MCPOAuthClient.issuer == "https://auth.example.com",
            MCPOAuthClient.client_id == "client-123",
        )
        .first()
    )
    if oauth_client is None:
        oauth_client = MCPOAuthClient(
            mcp_server_id=server.id,
            issuer="https://auth.example.com",
            authorization_endpoint="https://auth.example.com/authorize",
            token_endpoint="https://auth.example.com/token",
            client_id="client-123",
            token_endpoint_auth_method="none",
            redirect_uri="https://xagent.example.com/api/mcp/oauth/callback",
        )
        db.add(oauth_client)
        db.flush()

    grant = MCPOAuthGrant(
        mcp_server_id=server.id,
        user_id=user.id,
        mcp_oauth_client_id=oauth_client.id,
        resource_owner_key=resource_owner_key,
        issuer="https://auth.example.com",
        resource="https://mcp.example.com/mcp",
        scope=scope,
        access_token=encrypt_value(access_token),
        refresh_token=encrypt_value(refresh_token) if refresh_token else None,
        expires_at=expires_at,
        status=status,
    )
    db.add(grant)
    db.commit()
    db.refresh(grant)
    return grant


async def _load_configs(db, user: User, **kwargs):
    cfg = WebToolConfig(
        db=db,
        request=None,
        user=user,
        user_id=user.id,
        workspace_config={"base_dir": "/tmp", "task_id": "test"},
        **kwargs,
    )
    return await cfg.get_mcp_server_configs(), cfg


def _assert_unavailable_runtime_config(
    config: dict,
    server: MCPServer,
    diagnostic: dict,
) -> None:
    assert config["name"] == server.name
    assert config["transport"] == "unavailable"
    assert config["description"] == server.description
    assert config["config"]["unavailable"] is True
    assert config["config"]["reason"] == diagnostic["code"]
    assert config["config"]["server_id"] == server.id
    assert config["config"]["diagnostic"] == diagnostic
    if diagnostic["code"] == "authorization_required":
        assert config["config"]["failure_code"] == "oauth_token_required"
    else:
        # insufficient_scope and token_refresh_failed mean a grant already
        # exists and the failure is transient/narrower than a missing grant -
        # pausing with a connect_apps card for those would show a false
        # "Connected" badge and a Continue that loops forever (see config.py's
        # runtime_build.connection is None branch), so failure_code is
        # omitted entirely rather than set, and _run_unavailable takes the
        # plain error path instead of pausing.
        assert "failure_code" not in config["config"]
    assert config["user_id"] == str(server.user_mcpservers[0].user_id)
    assert config["allow_users"] == [str(server.user_mcpservers[0].user_id)]
    assert "runtime_input_schema" not in config
    assert "runtime_bindings" not in config
    assert "allow_delegated_authorization" not in config
    assert "connector_runtime" not in config


def test_effective_mcp_oauth_resource_prefers_selector_verbatim(db_session):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)

    assert (
        effective_mcp_oauth_resource(
            server,
            mcp_auth_context={
                str(server.id): {"resource": " https://selector.example/mcp "}
            },
        )
        == " https://selector.example/mcp "
    )


def test_effective_mcp_oauth_resource_falls_back_to_auth_verbatim(db_session):
    db, user, _ = db_session
    configured_server = _add_mcp_oauth_server(
        db,
        user,
        resource=" https://configured.example/mcp ",
    )

    assert (
        effective_mcp_oauth_resource(
            configured_server,
            mcp_auth_context={
                str(configured_server.id): {"resource": ""},
            },
        )
        == " https://configured.example/mcp "
    )


def test_effective_mcp_oauth_resource_falls_back_to_server_url(db_session):
    db, user, _ = db_session
    transport_server = _add_mcp_oauth_server(db, user, resource="")

    assert (
        effective_mcp_oauth_resource(
            transport_server,
            mcp_auth_context={
                str(transport_server.id): {"resource": object()},
            },
        )
        == "https://mcp.example.com/mcp"
    )


def test_connection_without_authorization_copies_and_filters_headers():
    connection = {
        "transport": "streamable_http",
        "url": "https://mcp.example.com/mcp",
        "headers": {
            "Authorization": "Bearer uppercase",
            "authorization": "Bearer lowercase",
            "AUTHORIZATION": "Bearer shouting",
            "X-Request-Source": "xagent",
        },
        "auth": {"type": "mcp_oauth"},
    }

    result = connection_without_authorization(connection)

    assert result == {
        "transport": "streamable_http",
        "url": "https://mcp.example.com/mcp",
        "headers": {"X-Request-Source": "xagent"},
        "auth": {"type": "mcp_oauth"},
    }
    assert result is not connection
    assert result["headers"] is not connection["headers"]
    assert connection["headers"] == {
        "Authorization": "Bearer uppercase",
        "authorization": "Bearer lowercase",
        "AUTHORIZATION": "Bearer shouting",
        "X-Request-Source": "xagent",
    }


def test_connection_with_bearer_authorization_replaces_static_auth_without_mutation():
    connection = {
        "url": "https://mcp.example.com/mcp",
        "headers": {
            "authorization": "Basic static-credential",
            "X-Request-Source": "xagent",
        },
    }

    result = connection_with_bearer_authorization(connection, "runtime-token")

    assert result == {
        "url": "https://mcp.example.com/mcp",
        "headers": {
            "X-Request-Source": "xagent",
            "Authorization": "Bearer runtime-token",
        },
    }
    assert result is not connection
    assert result["headers"] is not connection["headers"]
    assert connection["headers"] == {
        "authorization": "Basic static-credential",
        "X-Request-Source": "xagent",
    }


@pytest.mark.asyncio
async def test_mcp_oauth_missing_grant_does_not_fall_back_to_static_headers(
    db_session,
):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)

    configs, cfg = await _load_configs(db, user)

    diagnostics = cfg.get_mcp_oauth_diagnostics()
    assert diagnostics == [
        {
            "code": "authorization_required",
            "message": "No active MCP OAuth grant exists for the selected resource owner",
            "server_id": server.id,
            "server_name": "records-runtime",
            "resource_owner_key": f"xagent:user:{user.id}",
            "resource": "https://mcp.example.com/mcp",
            "scope": "records.read",
            "issuer": "https://auth.example.com",
        }
    ]
    assert len(configs) == 1
    _assert_unavailable_runtime_config(configs[0], server, diagnostics[0])


@pytest.mark.asyncio
async def test_mcp_oauth_runtime_selects_resource_owner_grant(db_session):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    _add_grant(
        db,
        server=server,
        user=user,
        resource_owner_key="resource-owner-a",
        access_token="access-token-a",
    )
    _add_grant(
        db,
        server=server,
        user=user,
        resource_owner_key="resource-owner-b",
        access_token="access-token-b",
    )

    configs, cfg = await _load_configs(
        db,
        user,
        mcp_auth_context={str(server.id): {"resource_owner_key": "resource-owner-b"}},
    )

    assert cfg.get_mcp_oauth_diagnostics() == []
    assert len(configs) == 1
    headers = configs[0]["config"]["headers"]
    assert headers == {
        "X-Request-Source": "xagent",
        "Authorization": "Bearer access-token-b",
    }
    assert "auth" not in configs[0]["config"]


@pytest.mark.asyncio
async def test_mcp_oauth_runtime_allows_discovered_scope_when_not_configured(
    db_session,
):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user, include_scope=False)
    _add_grant(
        db,
        server=server,
        user=user,
        resource_owner_key=f"xagent:user:{user.id}",
        access_token="discovered-scope-token",
    )

    configs, cfg = await _load_configs(db, user)

    assert cfg.get_mcp_oauth_diagnostics() == []
    assert configs[0]["config"]["headers"]["Authorization"] == (
        "Bearer discovered-scope-token"
    )


@pytest.mark.asyncio
async def test_mcp_oauth_runtime_uses_discovered_grant_without_configured_selectors(
    db_session,
):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    server.auth = {"type": "mcp_oauth", "scope": "records.read"}
    db.commit()
    _add_grant(
        db,
        server=server,
        user=user,
        resource_owner_key=f"xagent:user:{user.id}",
        access_token="discovered-resource-token",
    )

    configs, cfg = await _load_configs(db, user)

    assert cfg.get_mcp_oauth_diagnostics() == []
    assert configs[0]["config"]["headers"]["Authorization"] == (
        "Bearer discovered-resource-token"
    )


@pytest.mark.asyncio
async def test_mcp_oauth_runtime_canonicalizes_configured_resource_for_lookup(
    db_session,
):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(
        db,
        user,
        resource="https://MCP.EXAMPLE.com:443/mcp/",
        issuer="https://AUTH.EXAMPLE.com:443/",
    )
    _add_grant(
        db,
        server=server,
        user=user,
        resource_owner_key=f"xagent:user:{user.id}",
        access_token="canonical-resource-token",
    )

    configs, cfg = await _load_configs(db, user)

    assert cfg.get_mcp_oauth_diagnostics() == []
    assert configs[0]["config"]["headers"]["Authorization"] == (
        "Bearer canonical-resource-token"
    )


@pytest.mark.asyncio
async def test_mcp_oauth_runtime_accepts_grant_scope_superset_regardless_of_order(
    db_session,
):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    _add_grant(
        db,
        server=server,
        user=user,
        resource_owner_key=f"xagent:user:{user.id}",
        access_token="superset-scope-token",
        scope="records.write records.read records.delete",
    )

    configs, cfg = await _load_configs(
        db,
        user,
        mcp_auth_context={str(server.id): {"scope": "records.read records.write"}},
    )

    assert cfg.get_mcp_oauth_diagnostics() == []
    assert configs[0]["config"]["headers"]["Authorization"] == (
        "Bearer superset-scope-token"
    )


@pytest.mark.asyncio
async def test_mcp_oauth_runtime_reports_insufficient_scope_without_static_fallback(
    db_session,
):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    _add_grant(
        db,
        server=server,
        user=user,
        resource_owner_key=f"xagent:user:{user.id}",
        access_token="read-only-token",
        scope="records.read",
    )

    configs, cfg = await _load_configs(
        db,
        user,
        mcp_auth_context={str(server.id): {"scope": "records.write"}},
    )

    diagnostics = cfg.get_mcp_oauth_diagnostics()
    assert diagnostics[0]["code"] == "insufficient_scope"
    assert diagnostics[0]["scope"] == "records.write"
    assert len(configs) == 1
    _assert_unavailable_runtime_config(configs[0], server, diagnostics[0])


@pytest.mark.asyncio
async def test_mcp_oauth_runtime_ignores_revoked_grant_without_static_fallback(
    db_session,
):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    _add_grant(
        db,
        server=server,
        user=user,
        resource_owner_key=f"xagent:user:{user.id}",
        access_token="revoked-token",
        status="revoked",
    )

    configs, cfg = await _load_configs(db, user)

    diagnostics = cfg.get_mcp_oauth_diagnostics()
    assert diagnostics[0]["code"] == "authorization_required"
    assert len(configs) == 1
    _assert_unavailable_runtime_config(configs[0], server, diagnostics[0])


@pytest.mark.asyncio
async def test_mcp_oauth_runtime_prefers_valid_grant_over_expired_without_refresh(
    db_session,
):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    _add_grant(
        db,
        server=server,
        user=user,
        resource_owner_key=f"xagent:user:{user.id}",
        access_token="valid-access-token",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    _add_grant(
        db,
        server=server,
        user=user,
        resource_owner_key=f"xagent:user:{user.id}",
        access_token="expired-access-token",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        scope="records.read records.extra",
    )

    configs, cfg = await _load_configs(db, user)

    assert cfg.get_mcp_oauth_diagnostics() == []
    assert configs[0]["config"]["headers"]["Authorization"] == (
        "Bearer valid-access-token"
    )


@pytest.mark.asyncio
async def test_mcp_oauth_runtime_expired_grant_without_refresh_requires_reauth(
    db_session,
):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    _add_grant(
        db,
        server=server,
        user=user,
        resource_owner_key=f"xagent:user:{user.id}",
        access_token="expired-access-token",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
    )

    configs, cfg = await _load_configs(db, user)

    diagnostics = cfg.get_mcp_oauth_diagnostics()
    assert diagnostics[0]["code"] == "authorization_required"
    assert "reauthorization" in diagnostics[0]["message"]
    assert len(configs) == 1
    _assert_unavailable_runtime_config(configs[0], server, diagnostics[0])


@pytest.mark.asyncio
async def test_mcp_oauth_runtime_does_not_use_other_users_grant(db_session):
    db, user, other_user = db_session
    server = _add_mcp_oauth_server(db, user)
    _add_grant(
        db,
        server=server,
        user=other_user,
        resource_owner_key="resource-owner-b",
        access_token="other-user-token",
    )

    configs, cfg = await _load_configs(
        db,
        user,
        mcp_auth_context={str(server.id): {"resource_owner_key": "resource-owner-b"}},
    )

    diagnostic = cfg.get_mcp_oauth_diagnostics()[0]
    assert diagnostic["code"] == "authorization_required"
    assert len(configs) == 1
    _assert_unavailable_runtime_config(configs[0], server, diagnostic)


@pytest.mark.asyncio
async def test_mcp_oauth_runtime_refreshes_expired_grant(db_session, monkeypatch):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    grant = _add_grant(
        db,
        server=server,
        user=user,
        resource_owner_key=f"xagent:user:{user.id}",
        access_token="expired-access-token",
        refresh_token="refresh-token-123",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
    )
    grant.oauth_client.client_secret = encrypt_value("client-secret")
    grant.oauth_client.token_endpoint_auth_method = "client_secret_post"
    db.add(
        MCPOAuthClient(
            mcp_server_id=server.id,
            issuer="https://auth.example.com",
            authorization_endpoint="https://auth.example.com/authorize",
            token_endpoint="https://auth.example.com/stale-token",
            client_id="stale-client",
            token_endpoint_auth_method="none",
            redirect_uri="https://xagent.example.com/api/mcp/oauth/callback",
        )
    )
    db.commit()

    real_async_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        form = parse_qs(request.content.decode())
        assert form["grant_type"] == ["refresh_token"]
        assert form["refresh_token"] == ["refresh-token-123"]
        assert form["resource"] == ["https://mcp.example.com/mcp"]
        assert form["client_id"] == ["client-123"]
        assert form["client_secret"] == ["client-secret"]
        return httpx.Response(
            200,
            json={
                "access_token": "fresh-access-token",
                "refresh_token": "fresh-refresh-token",
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": "records.read",
            },
        )

    def async_client_factory(*args, **kwargs):
        return real_async_client(transport=httpx.MockTransport(handler))

    async def skip_url_policy(*args, **kwargs):
        return None

    monkeypatch.setattr(mcp_oauth_service, "validate_oauth_http_url", skip_url_policy)
    monkeypatch.setattr(mcp_oauth_service.httpx, "AsyncClient", async_client_factory)

    pending_user = User(username="pending-refresh-user", password_hash="x")
    db.add(pending_user)

    configs, cfg = await _load_configs(db, user)

    assert cfg.get_mcp_oauth_diagnostics() == []
    assert configs[0]["config"]["headers"]["Authorization"] == (
        "Bearer fresh-access-token"
    )
    assert pending_user in db.new
    assert pending_user.id is None
    assert decrypt_value(grant.access_token) == "fresh-access-token"
    assert decrypt_value(grant.refresh_token) == "fresh-refresh-token"


@pytest.mark.asyncio
async def test_mcp_oauth_runtime_refreshes_discovered_grant_without_resource_selector(
    db_session, monkeypatch
):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    server.auth = {"type": "mcp_oauth", "scope": "records.read"}
    db.commit()
    grant = _add_grant(
        db,
        server=server,
        user=user,
        resource_owner_key=f"xagent:user:{user.id}",
        access_token="expired-access-token",
        refresh_token="refresh-token-123",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
    )

    async def fake_refresh_grant(oauth_client, *, refresh_token, resource, client):
        assert oauth_client.client_id == "client-123"
        assert refresh_token == "refresh-token-123"
        assert resource == "https://mcp.example.com/mcp"
        assert client is None
        return {
            "access_token": "fresh-access-token",
            "refresh_token": "fresh-refresh-token",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "records.read",
        }

    monkeypatch.setattr(
        mcp_oauth_service, "_refresh_mcp_oauth_grant", fake_refresh_grant
    )

    configs, cfg = await _load_configs(db, user)

    assert cfg.get_mcp_oauth_diagnostics() == []
    assert configs[0]["config"]["headers"]["Authorization"] == (
        "Bearer fresh-access-token"
    )
    db.refresh(grant)
    assert decrypt_value(grant.access_token) == "fresh-access-token"
    assert decrypt_value(grant.refresh_token) == "fresh-refresh-token"


@pytest.mark.asyncio
async def test_mcp_oauth_runtime_refresh_calls_token_endpoint_before_row_lock(
    db_session, monkeypatch
):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    _add_grant(
        db,
        server=server,
        user=user,
        resource_owner_key=f"xagent:user:{user.id}",
        access_token="expired-access-token",
        refresh_token="refresh-token-123",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
    )
    events: list[str] = []
    original_with_for_update = Query.with_for_update

    def recording_with_for_update(self, *args, **kwargs):
        events.append("lock")
        return original_with_for_update(self, *args, **kwargs)

    async def fake_refresh_grant(*args, **kwargs):
        events.append("network")
        return {
            "access_token": "fresh-access-token",
            "refresh_token": "fresh-refresh-token",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "records.read",
        }

    monkeypatch.setattr(Query, "with_for_update", recording_with_for_update)
    monkeypatch.setattr(
        mcp_oauth_service, "_refresh_mcp_oauth_grant", fake_refresh_grant
    )

    configs, cfg = await _load_configs(db, user)

    assert cfg.get_mcp_oauth_diagnostics() == []
    assert configs[0]["config"]["headers"]["Authorization"] == (
        "Bearer fresh-access-token"
    )
    assert events == ["network", "lock"]


@pytest.mark.asyncio
async def test_mcp_oauth_runtime_refresh_without_expires_in_clears_stale_expiry(
    db_session, monkeypatch
):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    grant = _add_grant(
        db,
        server=server,
        user=user,
        resource_owner_key=f"xagent:user:{user.id}",
        access_token="expired-access-token",
        refresh_token="refresh-token-123",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
    )
    db.commit()

    real_async_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "access_token": "fresh-access-token",
                "token_type": "Bearer",
                "scope": "records.read",
            },
        )

    def async_client_factory(*args, **kwargs):
        return real_async_client(transport=httpx.MockTransport(handler))

    async def skip_url_policy(*args, **kwargs):
        return None

    monkeypatch.setattr(mcp_oauth_service, "validate_oauth_http_url", skip_url_policy)
    monkeypatch.setattr(mcp_oauth_service.httpx, "AsyncClient", async_client_factory)

    configs, cfg = await _load_configs(db, user)

    assert cfg.get_mcp_oauth_diagnostics() == []
    assert configs[0]["config"]["headers"]["Authorization"] == (
        "Bearer fresh-access-token"
    )
    db.refresh(grant)
    assert decrypt_value(grant.access_token) == "fresh-access-token"
    assert grant.expires_at is None


@pytest.mark.asyncio
async def test_mcp_oauth_runtime_refresh_failure_retains_unavailable_without_static_fallback(
    db_session,
    monkeypatch,
):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    grant = _add_grant(
        db,
        server=server,
        user=user,
        resource_owner_key=f"xagent:user:{user.id}",
        access_token="expired-access-token",
        refresh_token="refresh-token-123",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
    )
    db.commit()

    real_async_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": "invalid_grant",
                "error_description": "refresh token is invalid",
                "access_token": "leaked-access-token",
                "refresh_token": "leaked-refresh-token",
            },
        )

    def async_client_factory(*args, **kwargs):
        return real_async_client(transport=httpx.MockTransport(handler))

    async def skip_url_policy(*args, **kwargs):
        return None

    monkeypatch.setattr(mcp_oauth_service, "validate_oauth_http_url", skip_url_policy)
    monkeypatch.setattr(mcp_oauth_service.httpx, "AsyncClient", async_client_factory)

    configs, cfg = await _load_configs(db, user)

    diagnostics = cfg.get_mcp_oauth_diagnostics()
    assert diagnostics[0]["code"] == "token_refresh_failed"
    assert diagnostics[0]["message"] == "refresh token is invalid"
    assert "leaked-access-token" not in str(diagnostics[0])
    assert "leaked-refresh-token" not in str(diagnostics[0])
    assert len(configs) == 1
    _assert_unavailable_runtime_config(configs[0], server, diagnostics[0])
    assert "leaked-access-token" not in str(configs[0])
    assert "leaked-refresh-token" not in str(configs[0])
    db.refresh(grant)
    assert decrypt_value(grant.access_token) == "expired-access-token"


@pytest.mark.asyncio
async def test_mcp_oauth_runtime_refresh_sanitizes_transport_exception(
    db_session,
    monkeypatch,
):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    grant = _add_grant(
        db,
        server=server,
        user=user,
        resource_owner_key=f"xagent:user:{user.id}",
        access_token="expired-access-token",
        refresh_token="refresh-token-123",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
    )
    real_async_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("raw refresh transport detail with secret-token")

    def async_client_factory(*args, **kwargs):
        return real_async_client(transport=httpx.MockTransport(handler))

    async def skip_url_policy(*args, **kwargs):
        return None

    monkeypatch.setattr(mcp_oauth_service, "validate_oauth_http_url", skip_url_policy)
    monkeypatch.setattr(mcp_oauth_service.httpx, "AsyncClient", async_client_factory)

    configs, cfg = await _load_configs(db, user)

    diagnostics = cfg.get_mcp_oauth_diagnostics()
    assert diagnostics[0]["code"] == "token_refresh_failed"
    assert diagnostics[0]["message"] == "OAuth request failed"
    assert "secret-token" not in str(diagnostics[0])
    assert len(configs) == 1
    _assert_unavailable_runtime_config(configs[0], server, diagnostics[0])
    assert "secret-token" not in str(configs[0])
    db.refresh(grant)
    assert decrypt_value(grant.access_token) == "expired-access-token"


@pytest.mark.asyncio
async def test_mcp_oauth_runtime_refresh_rejects_narrowed_scope_without_commit(
    db_session,
    monkeypatch,
):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    grant = _add_grant(
        db,
        server=server,
        user=user,
        resource_owner_key=f"xagent:user:{user.id}",
        access_token="expired-access-token",
        refresh_token="refresh-token-123",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        scope="records.read records.write",
    )

    real_async_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "access_token": "narrowed-access-token",
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": "records.read",
            },
        )

    def async_client_factory(*args, **kwargs):
        return real_async_client(transport=httpx.MockTransport(handler))

    async def skip_url_policy(*args, **kwargs):
        return None

    monkeypatch.setattr(mcp_oauth_service, "validate_oauth_http_url", skip_url_policy)
    monkeypatch.setattr(mcp_oauth_service.httpx, "AsyncClient", async_client_factory)

    configs, cfg = await _load_configs(
        db,
        user,
        mcp_auth_context={str(server.id): {"scope": "records.write"}},
    )

    diagnostics = cfg.get_mcp_oauth_diagnostics()
    assert diagnostics[0]["code"] == "insufficient_scope"
    assert len(configs) == 1
    _assert_unavailable_runtime_config(configs[0], server, diagnostics[0])
    db.refresh(grant)
    assert decrypt_value(grant.access_token) == "expired-access-token"
    assert grant.scope == "records.read records.write"


@pytest.mark.asyncio
async def test_mcp_oauth_runtime_refresh_rejects_oversized_scope_without_commit(
    db_session,
    monkeypatch,
):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    grant = _add_grant(
        db,
        server=server,
        user=user,
        resource_owner_key=f"xagent:user:{user.id}",
        access_token="expired-access-token",
        refresh_token="refresh-token-123",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        scope="records.read",
    )
    oversized_scope = "scope-" + "x" * mcp_oauth_service.MCP_OAUTH_SCOPE_MAX_LENGTH

    real_async_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "access_token": "oversized-scope-token",
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": oversized_scope,
            },
        )

    def async_client_factory(*args, **kwargs):
        return real_async_client(transport=httpx.MockTransport(handler))

    async def skip_url_policy(*args, **kwargs):
        return None

    monkeypatch.setattr(mcp_oauth_service, "validate_oauth_http_url", skip_url_policy)
    monkeypatch.setattr(mcp_oauth_service.httpx, "AsyncClient", async_client_factory)

    configs, cfg = await _load_configs(db, user)

    diagnostics = cfg.get_mcp_oauth_diagnostics()
    assert diagnostics[0]["code"] == "token_refresh_failed"
    assert "at most" in diagnostics[0]["message"]
    assert len(configs) == 1
    _assert_unavailable_runtime_config(configs[0], server, diagnostics[0])
    db.refresh(grant)
    assert decrypt_value(grant.access_token) == "expired-access-token"
    assert grant.scope == "records.read"


@pytest.mark.asyncio
async def test_mcp_oauth_runtime_refresh_rejects_oversized_token_type_without_commit(
    db_session,
    monkeypatch,
):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    grant = _add_grant(
        db,
        server=server,
        user=user,
        resource_owner_key=f"xagent:user:{user.id}",
        access_token="expired-access-token",
        refresh_token="refresh-token-123",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        scope="records.read",
    )
    oversized_token_type = (
        "Bearer" + "x" * mcp_oauth_service.MCP_OAUTH_TOKEN_TYPE_MAX_LENGTH
    )

    real_async_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "access_token": "oversized-token-type-token",
                "token_type": oversized_token_type,
                "expires_in": 3600,
                "scope": "records.read",
            },
        )

    def async_client_factory(*args, **kwargs):
        return real_async_client(transport=httpx.MockTransport(handler))

    async def skip_url_policy(*args, **kwargs):
        return None

    monkeypatch.setattr(mcp_oauth_service, "validate_oauth_http_url", skip_url_policy)
    monkeypatch.setattr(mcp_oauth_service.httpx, "AsyncClient", async_client_factory)

    configs, cfg = await _load_configs(db, user)

    diagnostics = cfg.get_mcp_oauth_diagnostics()
    assert diagnostics[0]["code"] == "token_refresh_failed"
    assert "token_type" in diagnostics[0]["message"]
    assert len(configs) == 1
    _assert_unavailable_runtime_config(configs[0], server, diagnostics[0])
    db.refresh(grant)
    assert decrypt_value(grant.access_token) == "expired-access-token"
    assert grant.scope == "records.read"
    assert grant.token_type == "Bearer"
