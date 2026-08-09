from __future__ import annotations

import json
from datetime import timedelta
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from starlette.requests import Request

from xagent.core.tools.adapters.vibe.mcp_adapter import (
    MCPFailurePhase,
    MCPLoadResult,
    MCPServerLoadFailure,
)
from xagent.core.utils.encryption import decrypt_value, encrypt_value
from xagent.web.api import mcp as mcp_api
from xagent.web.api.mcp import (
    MCPOAuthConnectRequest,
    MCPOAuthDiscoverRequest,
    MCPOAuthStatusResponse,
    MCPServerUpdate,
    connect_mcp_oauth,
    connect_mcp_oauth_app,
    delete_mcp_oauth_grant,
    delete_mcp_server,
    discover_mcp_oauth,
    get_mcp_oauth_status,
    get_mcp_server_tools,
    list_mcp_apps,
    mcp_oauth_callback,
    update_mcp_server,
)
from xagent.web.models import MCPOAuthClient, MCPOAuthFlowState, MCPOAuthGrant
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.mcp_oauth import mcp_oauth_client_registration_lookup_hash
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.services import mcp_oauth as mcp_oauth_service
from xagent.web.services.mcp_oauth import (
    MCP_OAUTH_PERSISTED_VALUE_MAX_LENGTH,
    MCP_OAUTH_RESOURCE_OWNER_KEY_MAX_LENGTH,
    MCP_OAUTH_SCOPE_MAX_LENGTH,
    MCP_OAUTH_TOKEN_ENDPOINT_AUTH_METHOD_MAX_LENGTH,
    MCP_OAUTH_TOKEN_TYPE_MAX_LENGTH,
)


@pytest.fixture()
def db_session(tmp_path):
    db_path = tmp_path / "mcp-oauth.db"
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


def _request(
    path: str,
    headers: list[tuple[bytes, bytes]] | None = None,
    *,
    bind_oauth_state_cookie: bool = True,
) -> Request:
    parsed = urlparse(path)
    request_headers = list(headers or [])
    query = parse_qs(parsed.query)
    state = query.get("state", [None])[0]
    if bind_oauth_state_cookie and parsed.path == "/api/mcp/oauth/callback" and state:
        request_headers.append(
            (
                b"cookie",
                (
                    f"{mcp_api.MCP_OAUTH_STATE_COOKIE}="
                    f"{mcp_api._mcp_oauth_state_cookie_value(state)}"
                ).encode(),
            )
        )
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": parsed.path,
            "query_string": parsed.query.encode(),
            "headers": request_headers,
        }
    )


def _redirect_query(response):
    return parse_qs(urlparse(response.headers["location"]).query)


def _discovery() -> SimpleNamespace:
    return SimpleNamespace(
        resource="https://mcp.example.com/mcp",
        scopes=("records.read",),
        protected_resource=SimpleNamespace(
            authorization_servers=("https://auth.example.com",),
        ),
        authorization_server=SimpleNamespace(
            issuer="https://auth.example.com",
            authorization_endpoint="https://auth.example.com/authorize",
            token_endpoint="https://auth.example.com/token",
            registration_endpoint="https://auth.example.com/register",
            client_id_metadata_document_supported=True,
            raw={"issuer": "https://auth.example.com"},
        ),
    )


def _add_mcp_oauth_server(
    db,
    user: User,
    *,
    scope: str = "records.read",
    transport: str = "streamable_http",
    client_id: str = "client-123",
    client_secret: str | None = "client-secret",
    redirect_uri: str | None = "https://xagent.example.com/api/mcp/oauth/callback",
    token_endpoint_auth_method: str = "client_secret_post",
) -> MCPServer:
    server = MCPServer.from_config(
        {
            "name": "records",
            "managed": "external",
            "transport": transport,
            "url": "https://mcp.example.com/mcp",
            "auth": {
                "type": "mcp_oauth",
                "resource": "https://mcp.example.com/mcp",
                "issuer": "https://auth.example.com",
                "scope": scope,
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": redirect_uri,
                "token_endpoint_auth_method": token_endpoint_auth_method,
            },
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


def _add_oauth_client(
    db,
    server: MCPServer,
    *,
    client_id: str = "client-123",
    client_secret: str | None = None,
    token_endpoint_auth_method: str = "none",
    metadata_json: dict | None = None,
) -> MCPOAuthClient:
    client = MCPOAuthClient(
        mcp_server_id=server.id,
        issuer="https://auth.example.com",
        authorization_endpoint="https://auth.example.com/authorize",
        token_endpoint="https://auth.example.com/token",
        client_id=client_id,
        client_secret=client_secret,
        token_endpoint_auth_method=token_endpoint_auth_method,
        redirect_uri="https://xagent.example.com/api/mcp/oauth/callback",
        metadata_json=metadata_json,
    )
    db.add(client)
    db.flush()
    return client


def _add_callback_client_and_state(
    db,
    user: User,
    *,
    state: str,
    metadata_json: dict | None = None,
    redirect_after: str = "/mcp",
) -> tuple[MCPServer, MCPOAuthClient, MCPOAuthFlowState]:
    server = _add_mcp_oauth_server(db, user)
    client = _add_oauth_client(db, server, metadata_json=metadata_json)
    flow_state = MCPOAuthFlowState(
        state=state,
        mcp_server_id=server.id,
        user_id=user.id,
        mcp_oauth_client_id=client.id,
        resource_owner_key="resource-owner-a",
        issuer="https://auth.example.com",
        resource="https://mcp.example.com/mcp",
        scope="records.read",
        code_verifier=mcp_api.encrypt_value("verifier-123"),
        redirect_after=redirect_after,
        expires_at=mcp_api._utc_now() + timedelta(minutes=10),
    )
    db.add(flow_state)
    db.commit()
    return server, client, flow_state


def _set_user_mcp_active(db, user: User, server: MCPServer, is_active: bool) -> None:
    user_mcp = (
        db.query(UserMCPServer)
        .filter(
            UserMCPServer.user_id == user.id,
            UserMCPServer.mcpserver_id == server.id,
        )
        .one()
    )
    user_mcp.is_active = is_active
    db.commit()


@pytest.mark.asyncio
async def test_get_mcp_server_tools_requires_oauth_grant_without_static_fallback(
    db_session,
    monkeypatch,
):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    server.headers = {
        "Authorization": "Bearer static-token",
        "X-Request-Source": "xagent",
    }
    db.commit()

    async def fail_load_tools(*args, **kwargs):
        pytest.fail("MCP tools loader should not be called without an OAuth grant")

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools",
        fail_load_tools,
    )

    with pytest.raises(HTTPException) as exc_info:
        await get_mcp_server_tools(server.id, user, db)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["code"] == "authorization_required"
    assert (
        exc_info.value.detail["message"]
        == "No active MCP OAuth grant exists for the selected resource owner"
    )


@pytest.mark.asyncio
async def test_get_mcp_server_tools_injects_runtime_oauth_grant(
    db_session,
    monkeypatch,
):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    server.headers = {
        "Authorization": "Bearer static-token",
        "X-Request-Source": "xagent",
    }
    client = _add_oauth_client(db, server, client_id="client-123")
    grant = MCPOAuthGrant(
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
    db.add(grant)
    db.commit()
    captured_connections = []

    async def fake_load_tools(connections, name_prefix):
        captured_connections.append(connections)
        return MCPLoadResult(
            tools=(
                SimpleNamespace(name="search_records", description="Search records"),
            ),
            loaded_servers=("records",),
            failures=(),
        )

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools",
        fake_load_tools,
    )

    response = await get_mcp_server_tools(server.id, user, db)

    assert response["tool_count"] == 1
    connection = captured_connections[0]["records"]
    assert connection["headers"]["Authorization"] == "Bearer runtime-token"
    assert connection["headers"]["X-Request-Source"] == "xagent"


@pytest.mark.asyncio
async def test_get_mcp_server_tools_keeps_partial_tools_and_reports_safe_failures(
    db_session,
    monkeypatch,
):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    client = _add_oauth_client(db, server, client_id="client-123")
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

    async def partially_failed_load(*args, **kwargs):
        return MCPLoadResult(
            tools=(
                SimpleNamespace(name="search_records", description="Search records"),
            ),
            loaded_servers=("records",),
            failures=(
                MCPServerLoadFailure(
                    server_name="records",
                    phase=MCPFailurePhase.ADAPTER_CONSTRUCTION,
                    error_type="BearerSecretError",
                ),
            ),
        )

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools",
        partially_failed_load,
    )

    response = await get_mcp_server_tools(server.id, user, db)

    assert response == {
        "server_name": "records",
        "tool_count": 1,
        "tools": [
            {"name": "search_records", "description": "Search records"},
        ],
        "failures": [
            {
                "server_name": "records",
                "phase": "adapter_construction",
                "attempts": 1,
            }
        ],
    }
    assert "BearerSecretError" not in repr(response)


@pytest.mark.asyncio
async def test_get_mcp_server_tools_reports_structured_runtime_failure(
    db_session,
    monkeypatch,
):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    client = _add_oauth_client(db, server, client_id="client-123")
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

    async def failed_load(*args, **kwargs):
        return MCPLoadResult(
            tools=(),
            loaded_servers=(),
            failures=(
                MCPServerLoadFailure(
                    server_name="records",
                    phase=MCPFailurePhase.LIST_TOOLS,
                    error_type="BearerSecretError",
                ),
            ),
        )

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools",
        failed_load,
    )

    with pytest.raises(HTTPException) as exc_info:
        await get_mcp_server_tools(server.id, user, db)

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == {
        "code": "mcp_tools_unavailable",
        "message": "MCP server tools could not be loaded.",
        "failures": [{"server_name": "records", "phase": "list_tools", "attempts": 1}],
    }
    assert "BearerSecretError" not in repr(exc_info.value.detail)


@pytest.mark.asyncio
async def test_get_mcp_server_tools_rejects_inactive_user_server(db_session):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    _set_user_mcp_active(db, user, server, False)

    with pytest.raises(HTTPException) as exc_info:
        await get_mcp_server_tools(server.id, user, db)

    assert exc_info.value.status_code == 404


def test_update_mcp_server_can_reactivate_inactive_user_server(db_session):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    server.runtime_input_schema = {"context": {"account_id": {"type": "string"}}}
    server.runtime_bindings = [
        {
            "source": {"input_type": "context", "key": "account_id"},
            "target": {"target_type": "mcp_meta", "key": "account_id"},
        }
    ]
    server.allow_delegated_authorization = True
    db.commit()
    _set_user_mcp_active(db, user, server, False)

    response = update_mcp_server(
        server.id,
        MCPServerUpdate(description="Updated while inactive", is_active=True),
        user,
        db,
    )

    assert response.description == "Updated while inactive"
    assert response.is_active is True
    assert response.runtime_input_schema == {
        "context": {"account_id": {"type": "string"}}
    }
    assert response.runtime_bindings == [
        {
            "source": {"input_type": "context", "key": "account_id"},
            "target": {"target_type": "mcp_meta", "key": "account_id"},
        }
    ]
    assert response.allow_delegated_authorization is True


def test_update_mcp_server_persists_runtime_config(db_session):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)

    response = update_mcp_server(
        server.id,
        MCPServerUpdate(
            runtime_input_schema={"secrets": {"authorization": {"type": "string"}}},
            runtime_bindings=[
                {
                    "source": {"input_type": "secrets", "key": "authorization"},
                    "target": {
                        "target_type": "transport_headers",
                        "key": "Authorization",
                    },
                }
            ],
            allow_delegated_authorization=True,
        ),
        user,
        db,
    )

    assert response.runtime_input_schema == {
        "secrets": {"authorization": {"type": "string"}}
    }
    assert response.runtime_bindings == [
        {
            "source": {"input_type": "secrets", "key": "authorization"},
            "target": {
                "target_type": "transport_headers",
                "key": "Authorization",
            },
        }
    ]
    assert response.allow_delegated_authorization is True
    db.refresh(server)
    assert server.runtime_input_schema == response.runtime_input_schema
    assert server.runtime_bindings == response.runtime_bindings
    assert server.allow_delegated_authorization is True


def test_update_mcp_server_explicit_null_clears_runtime_config(db_session):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    server.runtime_input_schema = {"context": {"account_id": {"type": "string"}}}
    server.runtime_bindings = [
        {
            "source": {"input_type": "context", "key": "account_id"},
            "target": {"target_type": "mcp_meta", "key": "account_id"},
        }
    ]
    server.allow_delegated_authorization = True
    db.commit()

    response = update_mcp_server(
        server.id,
        MCPServerUpdate(
            runtime_input_schema=None,
            runtime_bindings=None,
            allow_delegated_authorization=False,
        ),
        user,
        db,
    )

    assert response.runtime_input_schema is None
    assert response.runtime_bindings is None
    assert response.allow_delegated_authorization is False
    db.refresh(server)
    assert server.runtime_input_schema is None
    assert server.runtime_bindings is None
    assert server.allow_delegated_authorization is False


def test_update_mcp_server_rejects_renamed_masked_global_env_key(db_session):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    encrypted_secret = encrypt_value("global-secret")
    server.env = {"TOKEN": encrypted_secret}
    db.commit()

    with pytest.raises(HTTPException) as exc_info:
        update_mcp_server(
            server.id,
            MCPServerUpdate(config={"env": {"RENAMED_TOKEN": "********"}}),
            user,
            db,
        )

    assert exc_info.value.status_code == 400
    db.refresh(server)
    assert server.env == {"TOKEN": encrypted_secret}


def test_update_mcp_server_rejects_renamed_masked_user_env_key(db_session):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    user_mcp = (
        db.query(UserMCPServer)
        .filter(
            UserMCPServer.user_id == user.id,
            UserMCPServer.mcpserver_id == server.id,
        )
        .one()
    )
    encrypted_secret = encrypt_value("user-secret")
    user_mcp.env = {"TOKEN": encrypted_secret}
    db.commit()

    with pytest.raises(HTTPException) as exc_info:
        update_mcp_server(
            server.id,
            MCPServerUpdate(user_env={"RENAMED_TOKEN": "********"}),
            user,
            db,
        )

    assert exc_info.value.status_code == 400
    db.refresh(user_mcp)
    assert user_mcp.env == {"TOKEN": encrypted_secret}


@pytest.mark.asyncio
async def test_connect_creates_pkce_state_and_redirects(db_session, monkeypatch):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    server.headers = {"Authorization": "Bearer static-token"}
    db.commit()

    async def fake_discover(*args, **kwargs):
        assert kwargs["headers"] is None
        return _discovery()

    monkeypatch.setattr(mcp_api, "discover_mcp_oauth_metadata", fake_discover)

    response = await connect_mcp_oauth(
        server.id,
        MCPOAuthConnectRequest(redirect_after="/settings/mcp"),
        user,
        db,
    )

    assert response.status_code == 303
    location = response.headers["location"]
    parsed = urlparse(location)
    query = parse_qs(parsed.query)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == (
        "https://auth.example.com/authorize"
    )
    assert query["response_type"] == ["code"]
    assert query["client_id"] == ["client-123"]
    assert query["redirect_uri"] == [
        "https://xagent.example.com/api/mcp/oauth/callback"
    ]
    assert query["resource"] == ["https://mcp.example.com/mcp"]
    assert query["scope"] == ["records.read"]
    assert query["code_challenge_method"] == ["S256"]
    assert "code_challenge" in query
    assert "code_verifier" not in query

    flow_state = db.query(MCPOAuthFlowState).one()
    assert flow_state.state == query["state"][0]
    assert flow_state.resource_owner_key == f"xagent:user:{user.id}"
    assert flow_state.redirect_after == "/settings/mcp"
    assert decrypt_value(flow_state.code_verifier) != flow_state.code_verifier

    client = db.query(MCPOAuthClient).one()
    assert client.client_id == "client-123"
    assert client.client_secret != "client-secret"
    assert decrypt_value(client.client_secret) == "client-secret"
    assert flow_state.mcp_oauth_client_id == client.id


@pytest.mark.asyncio
async def test_connect_dynamically_registers_public_client_when_client_id_is_empty(
    db_session, monkeypatch
):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(
        db,
        user,
        client_id="",
        client_secret=None,
        redirect_uri=None,
        token_endpoint_auth_method="none",
    )
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.xagent.test/")

    async def fake_discover(*args, **kwargs):
        return _discovery()

    registration_requests: list[httpx.Request] = []

    def registration_handler(request: httpx.Request) -> httpx.Response:
        registration_requests.append(request)
        return httpx.Response(
            201,
            json={
                "client_id": "dynamic-client-123",
                "token_endpoint_auth_method": "none",
            },
        )

    registration_client = httpx.AsyncClient(
        transport=httpx.MockTransport(registration_handler)
    )
    monkeypatch.setattr(mcp_api, "discover_mcp_oauth_metadata", fake_discover)
    monkeypatch.setattr(
        mcp_oauth_service,
        "create_mcp_oauth_http_client",
        lambda **kwargs: registration_client,
    )

    response = await connect_mcp_oauth(
        server.id,
        MCPOAuthConnectRequest(redirect_after="/settings/mcp"),
        user,
        db,
    )

    query = parse_qs(urlparse(response.headers["location"]).query)
    assert query["client_id"] == ["dynamic-client-123"]
    assert query["redirect_uri"] == ["https://api.xagent.test/api/mcp/oauth/callback"]
    assert len(registration_requests) == 1
    assert str(registration_requests[0].url) == "https://auth.example.com/register"
    assert json.loads(registration_requests[0].content) == {
        "application_type": "web",
        "client_name": "Xagent",
        "grant_types": ["authorization_code", "refresh_token"],
        "redirect_uris": ["https://api.xagent.test/api/mcp/oauth/callback"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    }
    client = db.query(MCPOAuthClient).one()
    assert client.client_id == "dynamic-client-123"
    assert client.client_secret is None
    assert client.token_endpoint_auth_method == "none"

    second_response = await connect_mcp_oauth(
        server.id,
        MCPOAuthConnectRequest(redirect_after="/settings/mcp"),
        user,
        db,
    )
    second_query = parse_qs(urlparse(second_response.headers["location"]).query)
    assert second_query["client_id"] == ["dynamic-client-123"]
    assert len(registration_requests) == 1
    assert db.query(MCPOAuthClient).count() == 1


def test_default_mcp_oauth_redirect_uri_prefers_public_api_base(monkeypatch):
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.xagent.test/base/")
    monkeypatch.setenv("XAGENT_APP_BASE_URL", "https://frontend.xagent.test/")

    assert mcp_api._default_mcp_oauth_redirect_uri() == (
        "https://api.xagent.test/base/api/mcp/oauth/callback"
    )


@pytest.mark.asyncio
async def test_connect_without_client_id_or_registration_endpoint_requires_preregistration(
    db_session, monkeypatch
):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(
        db,
        user,
        client_id="",
        client_secret=None,
        token_endpoint_auth_method="none",
    )

    async def fake_discover(*args, **kwargs):
        discovery = _discovery()
        discovery.authorization_server.registration_endpoint = None
        return discovery

    monkeypatch.setattr(mcp_api, "discover_mcp_oauth_metadata", fake_discover)

    with pytest.raises(HTTPException) as exc:
        await connect_mcp_oauth(
            server.id,
            MCPOAuthConnectRequest(redirect_after="/settings/mcp"),
            user,
            db,
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == {
        "code": "client_registration_unavailable",
        "message": (
            "Authorization server does not support dynamic client registration; "
            "configure a pre-registered MCP OAuth client_id"
        ),
    }


def test_upsert_oauth_client_preserves_existing_masked_client_secret(db_session):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    existing = _add_oauth_client(
        db,
        server,
        client_secret=encrypt_value("client-secret"),
        token_endpoint_auth_method="client_secret_post",
    )
    db.commit()

    client = mcp_api._upsert_mcp_oauth_client(
        db,
        server_id=server.id,
        discovery=_discovery(),
        client_id="client-123",
        client_secret=mcp_api.MASKED_SECRET_VALUE,
        token_endpoint_auth_method="client_secret_basic",
        redirect_uri="https://xagent.example.com/api/mcp/oauth/callback",
    )

    assert client.id == existing.id
    assert decrypt_value(client.client_secret) == "client-secret"
    assert client.token_endpoint_auth_method == "client_secret_basic"


def test_upsert_oauth_client_recovers_from_concurrent_insert(db_session, monkeypatch):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    SessionLocal = sessionmaker(bind=db.get_bind(), autoflush=False, autocommit=False)
    real_flush = db.flush
    flush_calls = 0

    def fail_first_flush_after_concurrent_insert(*args, **kwargs):
        nonlocal flush_calls
        if flush_calls == 0:
            flush_calls += 1
            concurrent_db = SessionLocal()
            try:
                concurrent_client = MCPOAuthClient(
                    mcp_server_id=server.id,
                    issuer="https://auth.example.com",
                    authorization_endpoint="https://auth.example.com/authorize-old",
                    token_endpoint="https://auth.example.com/token-old",
                    client_id="client-123",
                    token_endpoint_auth_method="none",
                    redirect_uri="https://xagent.example.com/api/mcp/oauth/callback",
                )
                concurrent_db.add(concurrent_client)
                concurrent_db.commit()
            finally:
                concurrent_db.close()
            raise IntegrityError("insert", {}, Exception("duplicate lookup_hash"))
        return real_flush(*args, **kwargs)

    monkeypatch.setattr(db, "flush", fail_first_flush_after_concurrent_insert)

    client = mcp_api._upsert_mcp_oauth_client(
        db,
        server_id=server.id,
        discovery=_discovery(),
        client_id="client-123",
        client_secret="client-secret",
        token_endpoint_auth_method="client_secret_post",
        redirect_uri="https://xagent.example.com/api/mcp/oauth/callback",
    )

    assert client.id is not None
    assert client.authorization_endpoint == "https://auth.example.com/authorize"
    assert client.token_endpoint == "https://auth.example.com/token"
    assert client.token_endpoint_auth_method == "client_secret_post"
    assert decrypt_value(client.client_secret) == "client-secret"


def test_dynamic_client_conflict_adopts_registered_winner(db_session, monkeypatch):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user, client_id="", client_secret=None)
    SessionLocal = sessionmaker(bind=db.get_bind(), autoflush=False, autocommit=False)
    real_flush = db.flush
    registration_lookup_hash = mcp_oauth_client_registration_lookup_hash(
        server.id,
        "https://auth.example.com",
        "https://xagent.example.com/api/mcp/oauth/callback",
    )
    flush_calls = 0

    def insert_winner_before_first_flush(*args, **kwargs):
        nonlocal flush_calls
        if flush_calls == 0:
            flush_calls += 1
            concurrent_db = SessionLocal()
            try:
                concurrent_db.add(
                    MCPOAuthClient(
                        mcp_server_id=server.id,
                        registration_lookup_hash=registration_lookup_hash,
                        issuer="https://auth.example.com",
                        authorization_endpoint="https://auth.example.com/authorize",
                        token_endpoint="https://auth.example.com/token",
                        client_id="winner-client",
                        token_endpoint_auth_method="none",
                        redirect_uri=(
                            "https://xagent.example.com/api/mcp/oauth/callback"
                        ),
                    )
                )
                concurrent_db.commit()
            finally:
                concurrent_db.close()
            raise IntegrityError("insert", {}, Exception("duplicate registration"))
        return real_flush(*args, **kwargs)

    monkeypatch.setattr(db, "flush", insert_winner_before_first_flush)

    client = mcp_api._upsert_mcp_oauth_client(
        db,
        server_id=server.id,
        discovery=_discovery(),
        client_id="loser-client",
        client_secret=None,
        token_endpoint_auth_method="none",
        redirect_uri="https://xagent.example.com/api/mcp/oauth/callback",
        registration_lookup_hash=registration_lookup_hash,
    )

    assert client.client_id == "winner-client"
    assert db.query(MCPOAuthClient).count() == 1


def test_upsert_oauth_client_rejects_masked_client_secret_without_existing_value(
    db_session,
):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)

    with pytest.raises(HTTPException) as exc:
        mcp_api._upsert_mcp_oauth_client(
            db,
            server_id=server.id,
            discovery=_discovery(),
            client_id="client-123",
            client_secret=mcp_api.MASKED_SECRET_VALUE,
            token_endpoint_auth_method="client_secret_post",
            redirect_uri="https://xagent.example.com/api/mcp/oauth/callback",
        )

    assert exc.value.detail["code"] == "invalid_resource"


def test_oauth_api_length_constants_match_schema():
    assert MCP_OAUTH_TOKEN_ENDPOINT_AUTH_METHOD_MAX_LENGTH == 100
    assert MCP_OAUTH_RESOURCE_OWNER_KEY_MAX_LENGTH == 512


@pytest.mark.asyncio
async def test_connect_canonicalizes_scope_before_persisting_flow_state(
    db_session, monkeypatch
):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(
        db,
        user,
        scope="records.write records.read records.write",
    )

    async def fake_discover(*args, **kwargs):
        return _discovery()

    monkeypatch.setattr(mcp_api, "discover_mcp_oauth_metadata", fake_discover)

    response = await connect_mcp_oauth(
        server.id,
        MCPOAuthConnectRequest(redirect_after="/settings/mcp"),
        user,
        db,
    )

    query = parse_qs(urlparse(response.headers["location"]).query)
    assert query["scope"] == ["records.read records.write"]
    assert db.query(MCPOAuthFlowState).one().scope == "records.read records.write"


@pytest.mark.asyncio
async def test_connect_rejects_scope_that_cannot_fit_grant_lookup_key(
    db_session, monkeypatch
):
    db, user, _ = db_session
    oversized_scope = "scope-" + "x" * MCP_OAUTH_SCOPE_MAX_LENGTH
    server = _add_mcp_oauth_server(db, user, scope=oversized_scope)

    async def fake_discover(*args, **kwargs):
        return _discovery()

    monkeypatch.setattr(mcp_api, "discover_mcp_oauth_metadata", fake_discover)

    with pytest.raises(HTTPException) as exc:
        await connect_mcp_oauth(
            server.id,
            MCPOAuthConnectRequest(redirect_after="/settings/mcp"),
            user,
            db,
        )

    assert exc.value.status_code == 400
    assert exc.value.detail["code"] == "invalid_scope"
    assert db.query(MCPOAuthFlowState).count() == 0


@pytest.mark.asyncio
async def test_connect_rejects_client_id_that_cannot_fit_persistence(
    db_session, monkeypatch
):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(
        db,
        user,
        client_id="client-" + "x" * MCP_OAUTH_PERSISTED_VALUE_MAX_LENGTH,
    )

    async def fake_discover(*args, **kwargs):
        return _discovery()

    monkeypatch.setattr(mcp_api, "discover_mcp_oauth_metadata", fake_discover)

    with pytest.raises(HTTPException) as exc:
        await connect_mcp_oauth(
            server.id,
            MCPOAuthConnectRequest(redirect_after="/settings/mcp"),
            user,
            db,
        )

    assert exc.value.status_code == 400
    assert exc.value.detail["code"] == "invalid_resource"
    assert db.query(MCPOAuthClient).count() == 0
    assert db.query(MCPOAuthFlowState).count() == 0


@pytest.mark.asyncio
async def test_connect_sanitizes_backslash_redirect_after(db_session, monkeypatch):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)

    async def fake_discover(*args, **kwargs):
        return _discovery()

    monkeypatch.setattr(mcp_api, "discover_mcp_oauth_metadata", fake_discover)

    await connect_mcp_oauth(
        server.id,
        MCPOAuthConnectRequest(redirect_after="/\\evil.example.com"),
        user,
        db,
    )

    flow_state = db.query(MCPOAuthFlowState).one()
    assert flow_state.redirect_after == "/tools"


@pytest.mark.asyncio
async def test_connect_sanitizes_oversized_redirect_after(db_session, monkeypatch):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)

    async def fake_discover(*args, **kwargs):
        return _discovery()

    monkeypatch.setattr(mcp_api, "discover_mcp_oauth_metadata", fake_discover)

    await connect_mcp_oauth(
        server.id,
        MCPOAuthConnectRequest(
            redirect_after="/" + "x" * MCP_OAUTH_PERSISTED_VALUE_MAX_LENGTH
        ),
        user,
        db,
    )

    assert db.query(MCPOAuthFlowState).one().redirect_after == "/tools"


@pytest.mark.asyncio
async def test_connect_merges_authorization_endpoint_query_and_preserves_fragment(
    db_session, monkeypatch
):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)

    async def fake_discover(*args, **kwargs):
        discovery = _discovery()
        discovery.authorization_server.authorization_endpoint = (
            "https://auth.example.com/authorize?prompt=consent#login"
        )
        return discovery

    monkeypatch.setattr(mcp_api, "discover_mcp_oauth_metadata", fake_discover)

    response = await connect_mcp_oauth(
        server.id,
        MCPOAuthConnectRequest(redirect_after="/settings/mcp"),
        user,
        db,
        accept="application/json",
    )

    payload = json.loads(response.body)
    parsed = urlparse(payload["authorization_url"])
    query = parse_qs(parsed.query)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == (
        "https://auth.example.com/authorize"
    )
    assert query["prompt"] == ["consent"]
    assert query["client_id"] == ["client-123"]
    assert query["resource"] == ["https://mcp.example.com/mcp"]
    assert parsed.fragment == "login"


def test_connect_request_rejects_public_resource_owner_key():
    with pytest.raises(ValueError):
        MCPOAuthConnectRequest.model_validate(
            {
                "redirect_after": "/settings/mcp",
                "resource_owner_key": "external:public-request",
            }
        )


def test_oauth_request_models_reject_public_config_overrides():
    public_overrides = {
        "resource": "https://other-resource.example.com/mcp",
        "issuer": "https://other-auth.example.com",
        "scope": "records.admin",
        "resource_metadata_url": "https://other-resource.example.com/.well-known/oauth-protected-resource",
    }

    with pytest.raises(ValueError):
        MCPOAuthDiscoverRequest.model_validate(public_overrides)
    with pytest.raises(ValueError):
        MCPOAuthConnectRequest.model_validate(
            {
                **public_overrides,
                "redirect_after": "/settings/mcp",
            }
        )


@pytest.mark.asyncio
async def test_connect_can_return_authorization_url_json(db_session, monkeypatch):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)

    async def fake_discover(*args, **kwargs):
        return _discovery()

    monkeypatch.setattr(mcp_api, "discover_mcp_oauth_metadata", fake_discover)

    response = await connect_mcp_oauth(
        server.id,
        MCPOAuthConnectRequest(redirect_after="/settings/mcp"),
        user,
        db,
        accept="application/json",
    )

    payload = json.loads(response.body)
    authorization_url = payload["authorization_url"]
    parsed = urlparse(authorization_url)
    query = parse_qs(parsed.query)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == (
        "https://auth.example.com/authorize"
    )
    assert query["client_id"] == ["client-123"]
    assert query["resource"] == ["https://mcp.example.com/mcp"]
    assert db.query(MCPOAuthFlowState).count() == 1


@pytest.mark.asyncio
async def test_oauth_routes_allow_websocket_transport(db_session, monkeypatch):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user, transport="websocket")

    async def fake_discover(*args, **kwargs):
        return _discovery()

    monkeypatch.setattr(mcp_api, "discover_mcp_oauth_metadata", fake_discover)

    discovery_response = await discover_mcp_oauth(
        server.id,
        MCPOAuthDiscoverRequest(),
        user,
        db,
    )
    assert discovery_response.issuer == "https://auth.example.com"

    connect_response = await connect_mcp_oauth(
        server.id,
        MCPOAuthConnectRequest(redirect_after="/settings/mcp"),
        user,
        db,
    )

    assert connect_response.status_code == 303
    assert db.query(MCPOAuthFlowState).count() == 1


@pytest.mark.asyncio
async def test_oauth_routes_reject_inactive_user_mcp_server(db_session, monkeypatch):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    _set_user_mcp_active(db, user, server, False)

    async def fail_discover(*args, **kwargs):
        pytest.fail("inactive MCP server must not run OAuth discovery")

    monkeypatch.setattr(mcp_api, "discover_mcp_oauth_metadata", fail_discover)

    with pytest.raises(mcp_api.HTTPException) as exc:
        await discover_mcp_oauth(
            server.id,
            MCPOAuthDiscoverRequest(),
            user,
            db,
        )
    assert exc.value.status_code == 404

    with pytest.raises(mcp_api.HTTPException) as exc:
        await connect_mcp_oauth(
            server.id,
            MCPOAuthConnectRequest(redirect_after="/settings/mcp"),
            user,
            db,
        )
    assert exc.value.status_code == 404

    with pytest.raises(mcp_api.HTTPException) as exc:
        await get_mcp_oauth_status(server.id, user, db)
    assert exc.value.status_code == 404

    client = _add_oauth_client(db, server)
    grant = MCPOAuthGrant(
        mcp_server_id=server.id,
        user_id=user.id,
        mcp_oauth_client_id=client.id,
        resource_owner_key=f"xagent:user:{user.id}",
        issuer="https://auth.example.com",
        resource="https://mcp.example.com/mcp",
        scope="records.read",
        access_token=mcp_api.encrypt_value("own-access-token"),
    )
    db.add(grant)
    db.commit()
    db.refresh(grant)

    with pytest.raises(mcp_api.HTTPException) as exc:
        await delete_mcp_oauth_grant(server.id, grant.id, user, db)
    assert exc.value.status_code == 404


def _add_remote_oauth_catalog_app(db, *, app_id: str = "remote-notes") -> None:
    """A built-in catalog row shaped like a real remote-MCP-OAuth connector:
    only a URL and auth.type — no static client_id, matching a DCR-only
    provider (e.g. Granola) that never hands out pre-registered credentials.

    The synthetic app_id must NOT match a real builtin registry entry: the
    builtin execution overlay (get_builtin_execution_fields) replaces a DB
    row's execution fields with the canonical registry values for matching
    app_ids, which would silently override this fixture's url/auth."""
    db.add(
        PublicMCPApp(
            app_id=app_id,
            name=app_id.title(),
            transport="streamable_http",
            launch_config={
                "url": "https://mcp.example.com/mcp",
                "auth": {"type": "mcp_oauth"},
            },
        )
    )
    db.commit()


@pytest.mark.asyncio
async def test_connect_app_creates_server_and_association_then_starts_dcr_flow(
    db_session, monkeypatch
):
    db, user, _ = db_session
    _add_remote_oauth_catalog_app(db)
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.xagent.test/")

    async def fake_discover(*args, **kwargs):
        return _discovery()

    registration_requests: list[httpx.Request] = []

    def registration_handler(request: httpx.Request) -> httpx.Response:
        registration_requests.append(request)
        return httpx.Response(
            201,
            json={
                "client_id": "dynamic-client-123",
                "token_endpoint_auth_method": "none",
            },
        )

    registration_client = httpx.AsyncClient(
        transport=httpx.MockTransport(registration_handler)
    )
    monkeypatch.setattr(mcp_api, "discover_mcp_oauth_metadata", fake_discover)
    monkeypatch.setattr(
        mcp_oauth_service,
        "create_mcp_oauth_http_client",
        lambda **kwargs: registration_client,
    )

    response = await connect_mcp_oauth_app(
        "remote-notes",
        MCPOAuthConnectRequest(redirect_after="/settings/mcp"),
        user,
        db,
    )

    assert response.status_code == 303
    query = parse_qs(urlparse(response.headers["location"]).query)
    assert query["client_id"] == ["dynamic-client-123"]
    assert len(registration_requests) == 1

    server = db.query(MCPServer).filter(MCPServer.name == "remote-notes").one()
    assert server.transport == "streamable_http"
    assert server.url == "https://mcp.example.com/mcp"
    assert server.auth["type"] == "mcp_oauth"

    assoc = (
        db.query(UserMCPServer)
        .filter(
            UserMCPServer.user_id == user.id, UserMCPServer.mcpserver_id == server.id
        )
        .one()
    )
    assert assoc.is_active is True
    assert assoc.is_owner is False


@pytest.mark.asyncio
async def test_connect_app_is_idempotent_across_repeated_connects(
    db_session, monkeypatch
):
    db, user, _ = db_session
    _add_remote_oauth_catalog_app(db)
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.xagent.test/")

    async def fake_discover(*args, **kwargs):
        return _discovery()

    def registration_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            201,
            json={
                "client_id": "dynamic-client-123",
                "token_endpoint_auth_method": "none",
            },
        )

    registration_client = httpx.AsyncClient(
        transport=httpx.MockTransport(registration_handler)
    )
    monkeypatch.setattr(mcp_api, "discover_mcp_oauth_metadata", fake_discover)
    monkeypatch.setattr(
        mcp_oauth_service,
        "create_mcp_oauth_http_client",
        lambda **kwargs: registration_client,
    )

    await connect_mcp_oauth_app(
        "remote-notes", MCPOAuthConnectRequest(redirect_after="/mcp"), user, db
    )
    await connect_mcp_oauth_app(
        "remote-notes", MCPOAuthConnectRequest(redirect_after="/mcp"), user, db
    )

    assert db.query(MCPServer).filter(MCPServer.name == "remote-notes").count() == 1
    assert (
        db.query(UserMCPServer)
        .join(MCPServer, UserMCPServer.mcpserver_id == MCPServer.id)
        .filter(MCPServer.name == "remote-notes", UserMCPServer.user_id == user.id)
        .count()
        == 1
    )


@pytest.mark.asyncio
async def test_connect_app_reactivates_a_previously_disconnected_association(
    db_session, monkeypatch
):
    db, user, _ = db_session
    _add_remote_oauth_catalog_app(db)

    async def fake_discover(*args, **kwargs):
        return _discovery()

    monkeypatch.setattr(mcp_api, "discover_mcp_oauth_metadata", fake_discover)
    monkeypatch.setattr(
        mcp_oauth_service,
        "create_mcp_oauth_http_client",
        lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(
                    201,
                    json={
                        "client_id": "dynamic-client-123",
                        "token_endpoint_auth_method": "none",
                    },
                )
            )
        ),
    )

    await connect_mcp_oauth_app(
        "remote-notes", MCPOAuthConnectRequest(redirect_after="/mcp"), user, db
    )
    server = db.query(MCPServer).filter(MCPServer.name == "remote-notes").one()
    _set_user_mcp_active(db, user, server, False)

    await connect_mcp_oauth_app(
        "remote-notes", MCPOAuthConnectRequest(redirect_after="/mcp"), user, db
    )

    assoc = (
        db.query(UserMCPServer)
        .filter(
            UserMCPServer.user_id == user.id, UserMCPServer.mcpserver_id == server.id
        )
        .one()
    )
    assert assoc.is_active is True


@pytest.mark.asyncio
async def test_connect_app_syncs_auth_when_catalog_auth_changes(
    db_session, monkeypatch
):
    """The catalog is the source of truth for the shared row's auth config:
    a registry change (e.g. adding a scope hint) must propagate to the
    already-provisioned server row on the next connect, not persist stale."""
    db, user, _ = db_session
    _add_remote_oauth_catalog_app(db)

    async def fake_discover(*args, **kwargs):
        return _discovery()

    monkeypatch.setattr(mcp_api, "discover_mcp_oauth_metadata", fake_discover)
    monkeypatch.setattr(
        mcp_oauth_service,
        "create_mcp_oauth_http_client",
        lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(
                    201,
                    json={
                        "client_id": "dynamic-client-123",
                        "token_endpoint_auth_method": "none",
                    },
                )
            )
        ),
    )

    await connect_mcp_oauth_app(
        "remote-notes", MCPOAuthConnectRequest(redirect_after="/mcp"), user, db
    )

    app_row = db.query(PublicMCPApp).filter(PublicMCPApp.app_id == "remote-notes").one()
    app_row.launch_config = {
        "url": "https://mcp.example.com/mcp",
        "auth": {"type": "mcp_oauth", "scope": "meetings.read"},
    }
    db.commit()

    await connect_mcp_oauth_app(
        "remote-notes", MCPOAuthConnectRequest(redirect_after="/mcp"), user, db
    )

    server = db.query(MCPServer).filter(MCPServer.name == "remote-notes").one()
    assert server.auth == {"type": "mcp_oauth", "scope": "meetings.read"}


@pytest.mark.asyncio
async def test_connect_app_auth_sync_tolerates_a_malformed_sensitive_field(
    db_session, monkeypatch
):
    """F13: encrypt_value() calls .encode() unconditionally, so a mis-authored
    non-string sensitive field (e.g. a nested object where client_secret
    should be a string) must not crash this user-facing connect request —
    it's an admin authoring bug to catch at write time, not here."""
    db, user, _ = db_session
    _add_remote_oauth_catalog_app(db)

    async def fake_discover(*args, **kwargs):
        return _discovery()

    monkeypatch.setattr(mcp_api, "discover_mcp_oauth_metadata", fake_discover)
    monkeypatch.setattr(
        mcp_oauth_service,
        "create_mcp_oauth_http_client",
        lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(
                    201,
                    json={
                        "client_id": "dynamic-client-123",
                        "token_endpoint_auth_method": "none",
                    },
                )
            )
        ),
    )

    await connect_mcp_oauth_app(
        "remote-notes", MCPOAuthConnectRequest(redirect_after="/mcp"), user, db
    )

    app_row = db.query(PublicMCPApp).filter(PublicMCPApp.app_id == "remote-notes").one()
    app_row.launch_config = {
        "url": "https://mcp.example.com/mcp",
        "auth": {"type": "mcp_oauth", "client_secret": {"nested": "not-a-string"}},
    }
    db.commit()

    # Must not raise — the malformed field is left as-is rather than crashing.
    await connect_mcp_oauth_app(
        "remote-notes", MCPOAuthConnectRequest(redirect_after="/mcp"), user, db
    )

    server = db.query(MCPServer).filter(MCPServer.name == "remote-notes").one()
    assert server.auth["client_secret"] == {"nested": "not-a-string"}


@pytest.mark.asyncio
async def test_connect_app_does_not_rewrite_unchanged_auth_with_secret(
    db_session, monkeypatch
):
    """Auth drift is detected on the DECRYPTED stored value: sensitive auth
    fields are encrypted at rest, so a raw stored-vs-catalog comparison would
    spuriously differ on every connect and rewrite the row each time. Fernet
    ciphertext changes on re-encryption, so an unchanged ciphertext across
    two connects proves no rewrite happened."""
    db, user, _ = db_session
    db.add(
        PublicMCPApp(
            app_id="remote-notes",
            name="Remote Notes",
            transport="streamable_http",
            launch_config={
                "url": "https://mcp.example.com/mcp",
                "auth": {
                    "type": "mcp_oauth",
                    "client_id": "static-client",
                    "client_secret": "static-secret",
                },
            },
        )
    )
    db.commit()

    async def fake_discover(*args, **kwargs):
        return _discovery()

    monkeypatch.setattr(mcp_api, "discover_mcp_oauth_metadata", fake_discover)

    await connect_mcp_oauth_app(
        "remote-notes", MCPOAuthConnectRequest(redirect_after="/mcp"), user, db
    )
    server = db.query(MCPServer).filter(MCPServer.name == "remote-notes").one()
    stored_secret_ciphertext = server.auth["client_secret"]
    assert stored_secret_ciphertext != "static-secret"  # encrypted at rest
    assert decrypt_value(stored_secret_ciphertext) == "static-secret"

    await connect_mcp_oauth_app(
        "remote-notes", MCPOAuthConnectRequest(redirect_after="/mcp"), user, db
    )

    db.refresh(server)
    assert server.auth["client_secret"] == stored_secret_ciphertext


@pytest.mark.asyncio
async def test_connect_app_rejects_non_mcp_oauth_catalog_app(db_session):
    db, user, _ = db_session
    db.add(
        PublicMCPApp(
            app_id="google-maps",
            name="Google Maps",
            transport="stdio",
            launch_config={"command": "npx", "required_env": ["GOOGLE_MAPS_API_KEY"]},
        )
    )
    db.commit()

    with pytest.raises(mcp_api.HTTPException) as exc:
        await connect_mcp_oauth_app(
            "google-maps", MCPOAuthConnectRequest(redirect_after="/mcp"), user, db
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_connect_app_rejects_hidden_mcp_oauth_app(db_session):
    """Round-5 m8: the hidden-app gate is wired into this path
    (_ensure_catalog_mcp_oauth_server -> _reject_hidden_catalog_app) but had
    no coverage here — this call site could be deleted and the suite would
    stay green. Mirrors test_connect_rejects_hidden_app in
    test_mcp_apps_connect.py, for the mcp_oauth connect path instead of the
    api_key/keyless one."""
    db, user, _ = db_session
    db.add(
        PublicMCPApp(
            app_id="hidden-remote-notes",
            name="Hidden Remote Notes",
            transport="streamable_http",
            is_visible_in_connector=False,
            launch_config={
                "url": "https://mcp.example.com/mcp",
                "auth": {"type": "mcp_oauth"},
            },
        )
    )
    db.commit()

    with pytest.raises(mcp_api.HTTPException) as exc:
        await connect_mcp_oauth_app(
            "hidden-remote-notes",
            MCPOAuthConnectRequest(redirect_after="/mcp"),
            user,
            db,
        )
    assert exc.value.status_code == 404

    # The gate fired before provisioning: no shared server row was created.
    assert (
        db.query(MCPServer).filter(MCPServer.name == "hidden-remote-notes").first()
        is None
    )


@pytest.mark.asyncio
async def test_connect_app_rejects_unknown_app_id(db_session):
    db, user, _ = db_session

    with pytest.raises(mcp_api.HTTPException) as exc:
        await connect_mcp_oauth_app(
            "no-such-app", MCPOAuthConnectRequest(redirect_after="/mcp"), user, db
        )
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_connect_app_rejects_hijacked_server_with_foreign_url(db_session):
    """A pre-existing row under the catalog id with a different remote URL must
    not be reused — otherwise a victim's DCR/PKCE flow talks to an attacker's
    MCP server. Mirrors the stdio hijack guard test in test_mcp_apps_connect.py;
    this shape was previously untested (D1)."""
    db, user, _ = db_session
    _add_remote_oauth_catalog_app(db)
    db.add(
        MCPServer(
            name="remote-notes",
            managed="external",
            transport="streamable_http",
            url="https://evil.example.com/mcp",
        )
    )
    db.commit()

    with pytest.raises(mcp_api.HTTPException) as exc:
        await connect_mcp_oauth_app(
            "remote-notes", MCPOAuthConnectRequest(redirect_after="/mcp"), user, db
        )
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_connect_app_rejects_user_owned_server_even_with_matching_config(
    db_session,
):
    """A row under the catalog id that a user OWNS is a custom server squatting
    the id. Even with a config that matches the official launch, it must not be
    adopted as the shared row (D1's guard, previously untested for mcp_oauth)."""
    db, user, other_user = db_session
    _add_remote_oauth_catalog_app(db)
    server = MCPServer(
        name="remote-notes",
        managed="external",
        transport="streamable_http",
        url="https://mcp.example.com/mcp",
        auth={"type": "mcp_oauth"},
    )
    db.add(server)
    db.commit()
    db.add(
        UserMCPServer(
            user_id=other_user.id, mcpserver_id=server.id, is_owner=True, can_edit=True
        )
    )
    db.commit()

    with pytest.raises(mcp_api.HTTPException) as exc:
        await connect_mcp_oauth_app(
            "remote-notes", MCPOAuthConnectRequest(redirect_after="/mcp"), user, db
        )
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_connect_app_json_accept_returns_authorization_url_in_body(
    db_session, monkeypatch
):
    """The Accept: application/json branch is what the actual frontend popup
    flow uses; every other test here exercises the 303-redirect branch instead
    (accept=None)."""
    db, user, _ = db_session
    _add_remote_oauth_catalog_app(db)

    async def fake_discover(*args, **kwargs):
        return _discovery()

    monkeypatch.setattr(mcp_api, "discover_mcp_oauth_metadata", fake_discover)
    monkeypatch.setattr(
        mcp_oauth_service,
        "create_mcp_oauth_http_client",
        lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(
                    201,
                    json={
                        "client_id": "dynamic-client-123",
                        "token_endpoint_auth_method": "none",
                    },
                )
            )
        ),
    )

    response = await connect_mcp_oauth_app(
        "remote-notes",
        MCPOAuthConnectRequest(redirect_after="/mcp"),
        user,
        db,
        accept="application/json, text/plain, */*",
    )

    assert response.status_code == 200
    body = json.loads(response.body)
    assert "authorization_url" in body
    query = parse_qs(urlparse(body["authorization_url"]).query)
    assert query["client_id"] == ["dynamic-client-123"]


@pytest.mark.asyncio
async def test_mcp_oauth_app_not_connected_until_grant_completes(
    db_session, monkeypatch
):
    """M1: the UserMCPServer association is created before the user ever
    reaches the consent screen, so it alone must not mean "connected" — an
    abandoned/denied/failed authorization must not render as Connected."""
    db, user, _ = db_session
    _add_remote_oauth_catalog_app(db)

    async def fake_discover(*args, **kwargs):
        return _discovery()

    monkeypatch.setattr(mcp_api, "discover_mcp_oauth_metadata", fake_discover)
    monkeypatch.setattr(
        mcp_oauth_service,
        "create_mcp_oauth_http_client",
        lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(
                    201,
                    json={
                        "client_id": "dynamic-client-123",
                        "token_endpoint_auth_method": "none",
                    },
                )
            )
        ),
    )

    await connect_mcp_oauth_app(
        "remote-notes", MCPOAuthConnectRequest(redirect_after="/mcp"), user, db
    )
    server = db.query(MCPServer).filter(MCPServer.name == "remote-notes").one()

    apps_before_grant = list_mcp_apps(current_user=user, db=db)
    remote_notes_before = next(
        a for a in apps_before_grant if a["id"] == "remote-notes"
    )
    assert remote_notes_before["is_connected"] is False

    # The DCR flow already registered a client during connect_mcp_oauth_app
    # above; reuse it rather than registering a second one under the same
    # (server, issuer) pair, which would trip the unique lookup_hash.
    client = (
        db.query(MCPOAuthClient).filter(MCPOAuthClient.mcp_server_id == server.id).one()
    )
    db.add(
        MCPOAuthGrant(
            mcp_server_id=server.id,
            user_id=user.id,
            mcp_oauth_client_id=client.id,
            resource_owner_key=f"xagent:user:{user.id}",
            issuer="https://auth.example.com",
            resource="https://mcp.example.com/mcp",
            scope="",
            access_token=encrypt_value("remote-notes-access-token"),
            status="active",
        )
    )
    db.commit()

    apps_after_grant = list_mcp_apps(current_user=user, db=db)
    remote_notes_after = next(a for a in apps_after_grant if a["id"] == "remote-notes")
    assert remote_notes_after["is_connected"] is True
    assert remote_notes_after["server_id"] == server.id


@pytest.mark.asyncio
async def test_mcp_oauth_local_listing_also_requires_a_grant(db_session):
    """F1: the location=local/all branch computed connection state via its
    own name-based membership check and never consulted the active-grant
    gate at all — a custom (non-catalog) mcp_oauth server the user abandoned
    mid-consent rendered as connected there regardless of M1's fix to the
    default/remote branch."""
    db, user, _ = db_session
    # _add_mcp_oauth_server creates a server named "records" with no matching
    # catalog PublicMCPApp, so it falls into the local/all branch rather than
    # being excluded as a known catalog app.
    server = _add_mcp_oauth_server(db, user)

    local_apps_before_grant = list_mcp_apps(location="local", current_user=user, db=db)
    records_before = next(a for a in local_apps_before_grant if a["id"] == "records")
    assert records_before["is_connected"] is False

    client = _add_oauth_client(db, server)
    db.add(
        MCPOAuthGrant(
            mcp_server_id=server.id,
            user_id=user.id,
            mcp_oauth_client_id=client.id,
            resource_owner_key=f"xagent:user:{user.id}",
            issuer="https://auth.example.com",
            resource="https://mcp.example.com/mcp",
            scope="",
            access_token=encrypt_value("records-access-token"),
            status="active",
        )
    )
    db.commit()

    local_apps_after_grant = list_mcp_apps(location="local", current_user=user, db=db)
    records_after = next(a for a in local_apps_after_grant if a["id"] == "records")
    assert records_after["is_connected"] is True


@pytest.mark.asyncio
async def test_delete_mcp_server_revokes_only_the_disconnecting_users_grant(
    db_session, monkeypatch
):
    """M3: disconnecting a shared mcp_oauth catalog server must revoke the
    disconnecting user's own grant immediately, not merely wait for the row to
    cascade away once the last associated user disconnects — and must leave a
    still-connected sibling user's grant untouched."""
    db, user, other_user = db_session
    _add_remote_oauth_catalog_app(db)

    async def fake_discover(*args, **kwargs):
        return _discovery()

    monkeypatch.setattr(mcp_api, "discover_mcp_oauth_metadata", fake_discover)
    monkeypatch.setattr(
        mcp_oauth_service,
        "create_mcp_oauth_http_client",
        lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(
                    201,
                    json={
                        "client_id": "dynamic-client-123",
                        "token_endpoint_auth_method": "none",
                    },
                )
            )
        ),
    )

    await connect_mcp_oauth_app(
        "remote-notes", MCPOAuthConnectRequest(redirect_after="/mcp"), user, db
    )
    await connect_mcp_oauth_app(
        "remote-notes", MCPOAuthConnectRequest(redirect_after="/mcp"), other_user, db
    )
    server = db.query(MCPServer).filter(MCPServer.name == "remote-notes").one()
    # Both connects register against the same (server, issuer) pair, so DCR
    # reuses a single client row (see register_mcp_oauth_public_client).
    client = (
        db.query(MCPOAuthClient).filter(MCPOAuthClient.mcp_server_id == server.id).one()
    )

    own_grant = MCPOAuthGrant(
        mcp_server_id=server.id,
        user_id=user.id,
        mcp_oauth_client_id=client.id,
        resource_owner_key=f"xagent:user:{user.id}",
        issuer="https://auth.example.com",
        resource="https://mcp.example.com/mcp",
        scope="",
        access_token=encrypt_value("own-access-token"),
        status="active",
    )
    sibling_grant = MCPOAuthGrant(
        mcp_server_id=server.id,
        user_id=other_user.id,
        mcp_oauth_client_id=client.id,
        resource_owner_key=f"xagent:user:{other_user.id}",
        issuer="https://auth.example.com",
        resource="https://mcp.example.com/mcp",
        scope="",
        access_token=encrypt_value("sibling-access-token"),
        status="active",
    )
    db.add_all([own_grant, sibling_grant])
    db.commit()
    db.refresh(own_grant)
    db.refresh(sibling_grant)

    own_grant_id = own_grant.id
    await delete_mcp_server(server.id, current_user=user, db=db)

    # F10: a disconnected grant must not just flip to "revoked" and linger —
    # the row itself (still holding the encrypted access token) is purged.
    assert (
        db.query(MCPOAuthGrant).filter(MCPOAuthGrant.id == own_grant_id).one_or_none()
        is None
    )
    db.refresh(sibling_grant)
    assert sibling_grant.status == "active"

    # The shared row survives (other_user is still associated); only the
    # disconnecting user's association and grant are gone.
    assert (
        db.query(MCPServer).filter(MCPServer.id == server.id).one_or_none() is not None
    )
    assert (
        db.query(UserMCPServer)
        .filter(
            UserMCPServer.user_id == user.id, UserMCPServer.mcpserver_id == server.id
        )
        .one_or_none()
        is None
    )


@pytest.mark.asyncio
async def test_callback_exchanges_code_and_stores_encrypted_grant(
    db_session, monkeypatch
):
    monkeypatch.setenv("XAGENT_APP_BASE_URL", "https://app.example.com/")
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    client = _add_oauth_client(
        db,
        server,
        client_secret=mcp_api.encrypt_value("client-secret"),
        token_endpoint_auth_method="client_secret_post",
    )
    flow_state = MCPOAuthFlowState(
        state="state-123",
        mcp_server_id=server.id,
        user_id=user.id,
        mcp_oauth_client_id=client.id,
        resource_owner_key="resource-owner-a",
        issuer="https://auth.example.com",
        resource="https://mcp.example.com/mcp",
        scope="records.read",
        code_verifier=mcp_api.encrypt_value("verifier-123"),
        redirect_after="/mcp",
        expires_at=mcp_api._utc_now() + timedelta(minutes=10),
    )
    db.add(flow_state)
    db.commit()

    real_async_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        form = dict(item.split("=") for item in request.content.decode().split("&"))
        assert form["grant_type"] == "authorization_code"
        assert form["code"] == "auth-code"
        assert form["code_verifier"] == "verifier-123"
        assert form["resource"] == "https%3A%2F%2Fmcp.example.com%2Fmcp"
        assert form["client_secret"] == "client-secret"
        return httpx.Response(
            200,
            json={
                "access_token": "plain-access-token",
                "refresh_token": "plain-refresh-token",
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": "records.read",
            },
        )

    def async_client_factory(*args, **kwargs):
        return real_async_client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(mcp_api, "create_mcp_oauth_http_client", async_client_factory)

    response = await mcp_oauth_callback(
        _request("/api/mcp/oauth/callback?code=auth-code&state=state-123"),
        db,
    )

    assert response.status_code == 307
    # mcp_oauth_success=1 is the explicit positive signal the connect popup's
    # self-close effect keys on (N5) instead of inferring success from the
    # absence of error params.
    assert (
        response.headers["location"]
        == "https://app.example.com/mcp?mcp_oauth_success=1"
    )
    grant = db.query(MCPOAuthGrant).one()
    assert grant.resource_owner_key == "resource-owner-a"
    assert grant.access_token != "plain-access-token"
    assert decrypt_value(grant.access_token) == "plain-access-token"
    assert decrypt_value(grant.refresh_token) == "plain-refresh-token"
    assert db.query(MCPOAuthFlowState).one().consumed_at is not None


@pytest.mark.asyncio
async def test_exchange_code_sanitizes_transport_exception(db_session, monkeypatch):
    db, user, _ = db_session
    _, client, _ = _add_callback_client_and_state(
        db,
        user,
        state="transport-error-state",
    )
    real_async_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("raw transport detail with secret-token")

    def async_client_factory(*args, **kwargs):
        return real_async_client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(mcp_api, "create_mcp_oauth_http_client", async_client_factory)

    with pytest.raises(HTTPException) as exc:
        await mcp_api._exchange_mcp_oauth_code(
            client=client,
            code="auth-code",
            code_verifier="verifier-123",
            resource="https://mcp.example.com/mcp",
        )

    assert exc.value.detail["code"] == "token_exchange_failed"
    assert exc.value.detail["message"] == "OAuth request failed"
    assert "secret-token" not in str(exc.value.detail)


@pytest.mark.asyncio
async def test_callback_rejects_token_type_that_cannot_fit_persistence(
    db_session, monkeypatch
):
    db, user, _ = db_session
    _add_callback_client_and_state(
        db,
        user,
        state="oversized-token-type-state",
        redirect_after="/tools?tab=mcp",
    )

    async def fake_exchange(**kwargs):
        return {
            "access_token": "plain-access-token",
            "token_type": "Bearer" + "x" * MCP_OAUTH_TOKEN_TYPE_MAX_LENGTH,
            "scope": "records.read",
        }

    monkeypatch.setattr(mcp_api, "_exchange_mcp_oauth_code", fake_exchange)

    response = await mcp_oauth_callback(
        _request(
            "/api/mcp/oauth/callback?code=auth-code&state=oversized-token-type-state"
        ),
        db,
    )

    assert response.status_code == 307
    assert response.headers["location"].startswith("/tools?tab=mcp&")
    query = _redirect_query(response)
    assert query["mcp_oauth_error"] == ["invalid_resource"]
    assert "token_type" in query["mcp_oauth_error_message"][0]
    assert db.query(MCPOAuthGrant).count() == 0


@pytest.mark.asyncio
async def test_callback_accepts_matching_issuer_when_supported(db_session, monkeypatch):
    db, user, _ = db_session
    _add_callback_client_and_state(
        db,
        user,
        state="issuer-match-state",
        metadata_json={"authorization_response_iss_parameter_supported": True},
    )

    async def fake_exchange(**kwargs):
        return {
            "access_token": "plain-access-token",
            "token_type": "Bearer",
            "scope": "records.read",
        }

    monkeypatch.setattr(mcp_api, "_exchange_mcp_oauth_code", fake_exchange)

    response = await mcp_oauth_callback(
        _request(
            "/api/mcp/oauth/callback?code=auth-code&state=issuer-match-state"
            "&iss=https%3A%2F%2Fauth.example.com%2F"
        ),
        db,
    )

    assert response.status_code == 307
    assert db.query(MCPOAuthGrant).count() == 1


@pytest.mark.asyncio
async def test_callback_rejects_state_without_browser_session_cookie(
    db_session, monkeypatch
):
    db, user, _ = db_session
    _add_callback_client_and_state(db, user, state="missing-cookie-state")

    async def fail_exchange(**kwargs):
        pytest.fail("token exchange must not run without browser-bound state cookie")

    monkeypatch.setattr(mcp_api, "_exchange_mcp_oauth_code", fail_exchange)

    response = await mcp_oauth_callback(
        _request(
            "/api/mcp/oauth/callback?code=auth-code&state=missing-cookie-state",
            bind_oauth_state_cookie=False,
        ),
        db,
    )

    assert response.status_code == 307
    assert _redirect_query(response)["mcp_oauth_error"] == ["invalid_state"]
    assert db.query(MCPOAuthGrant).count() == 0


@pytest.mark.asyncio
async def test_callback_uses_flow_bound_client_when_same_issuer_has_multiple_clients(
    db_session, monkeypatch
):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    _add_oauth_client(db, server, client_id="stale-client")
    bound_client = _add_oauth_client(db, server, client_id="bound-client")
    flow_state = MCPOAuthFlowState(
        state="client-bound-state",
        mcp_server_id=server.id,
        user_id=user.id,
        mcp_oauth_client_id=bound_client.id,
        resource_owner_key="resource-owner-a",
        issuer="https://auth.example.com",
        resource="https://mcp.example.com/mcp",
        scope="records.read",
        code_verifier=mcp_api.encrypt_value("verifier-123"),
        redirect_after="/mcp",
        expires_at=mcp_api._utc_now() + timedelta(minutes=10),
    )
    db.add(flow_state)
    db.commit()

    async def fake_exchange(**kwargs):
        assert kwargs["client"].client_id == "bound-client"
        return {
            "access_token": "plain-access-token",
            "token_type": "Bearer",
            "scope": "records.read",
        }

    monkeypatch.setattr(mcp_api, "_exchange_mcp_oauth_code", fake_exchange)

    response = await mcp_oauth_callback(
        _request("/api/mcp/oauth/callback?code=auth-code&state=client-bound-state"),
        db,
    )

    assert response.status_code == 307
    grant = db.query(MCPOAuthGrant).one()
    assert grant.mcp_oauth_client_id == bound_client.id


@pytest.mark.asyncio
async def test_callback_canonicalizes_scope_before_persisting_grant(
    db_session, monkeypatch
):
    db, user, _ = db_session
    server, client, flow_state = _add_callback_client_and_state(
        db,
        user,
        state="scope-canonical-state",
    )
    flow_state.scope = "records.write records.read records.write"
    db.commit()

    async def fake_exchange(**kwargs):
        return {
            "access_token": "plain-access-token",
            "token_type": "Bearer",
            "scope": "records.write records.read records.write",
        }

    monkeypatch.setattr(mcp_api, "_exchange_mcp_oauth_code", fake_exchange)

    response = await mcp_oauth_callback(
        _request("/api/mcp/oauth/callback?code=auth-code&state=scope-canonical-state"),
        db,
    )

    assert response.status_code == 307
    grant = db.query(MCPOAuthGrant).one()
    assert grant.mcp_server_id == server.id
    assert grant.mcp_oauth_client_id == client.id
    assert grant.scope == "records.read records.write"


@pytest.mark.asyncio
async def test_callback_rejects_scope_that_cannot_fit_grant_lookup_key(
    db_session, monkeypatch
):
    db, user, _ = db_session
    _server, _client, _flow_state = _add_callback_client_and_state(
        db,
        user,
        state="oversized-scope-state",
        redirect_after="/tools",
    )
    oversized_scope = "scope-" + "x" * MCP_OAUTH_SCOPE_MAX_LENGTH

    async def fake_exchange(**kwargs):
        return {
            "access_token": "plain-access-token",
            "token_type": "Bearer",
            "scope": oversized_scope,
        }

    monkeypatch.setattr(mcp_api, "_exchange_mcp_oauth_code", fake_exchange)

    response = await mcp_oauth_callback(
        _request("/api/mcp/oauth/callback?code=auth-code&state=oversized-scope-state"),
        db,
    )

    assert response.status_code == 307
    query = _redirect_query(response)
    assert query["mcp_oauth_error"] == ["invalid_scope"]
    assert db.query(MCPOAuthGrant).count() == 0


@pytest.mark.asyncio
async def test_callback_without_expires_in_clears_existing_grant_expiry(
    db_session, monkeypatch
):
    db, user, _ = db_session
    server, client, _ = _add_callback_client_and_state(
        db,
        user,
        state="no-expiry-state",
    )
    existing_grant = MCPOAuthGrant(
        mcp_server_id=server.id,
        user_id=user.id,
        mcp_oauth_client_id=client.id,
        resource_owner_key="resource-owner-a",
        issuer="https://auth.example.com",
        resource="https://mcp.example.com/mcp",
        scope="records.read",
        access_token=mcp_api.encrypt_value("old-access-token"),
        expires_at=mcp_api._utc_now() - timedelta(minutes=1),
    )
    db.add(existing_grant)
    db.commit()

    async def fake_exchange(**kwargs):
        return {
            "access_token": "plain-access-token",
            "token_type": "Bearer",
            "scope": "records.read",
        }

    monkeypatch.setattr(mcp_api, "_exchange_mcp_oauth_code", fake_exchange)

    response = await mcp_oauth_callback(
        _request("/api/mcp/oauth/callback?code=auth-code&state=no-expiry-state"),
        db,
    )

    assert response.status_code == 307
    db.refresh(existing_grant)
    assert decrypt_value(existing_grant.access_token) == "plain-access-token"
    assert existing_grant.expires_at is None


@pytest.mark.asyncio
async def test_callback_rejects_missing_required_issuer_before_token_exchange(
    db_session, monkeypatch
):
    db, user, _ = db_session
    _add_callback_client_and_state(
        db,
        user,
        state="issuer-required-state",
        metadata_json={"authorization_response_iss_parameter_supported": True},
    )

    async def fail_exchange(**kwargs):
        pytest.fail("token exchange must not run when callback issuer is required")

    monkeypatch.setattr(mcp_api, "_exchange_mcp_oauth_code", fail_exchange)

    response = await mcp_oauth_callback(
        _request("/api/mcp/oauth/callback?code=auth-code&state=issuer-required-state"),
        db,
    )

    assert response.status_code == 307
    assert _redirect_query(response)["mcp_oauth_error"] == ["issuer_mismatch"]
    assert db.query(MCPOAuthGrant).count() == 0


@pytest.mark.asyncio
async def test_callback_rejects_mismatched_issuer_before_token_exchange(
    db_session, monkeypatch
):
    db, user, _ = db_session
    _add_callback_client_and_state(
        db,
        user,
        state="issuer-mismatch-state",
        metadata_json={"authorization_response_iss_parameter_supported": False},
    )

    async def fail_exchange(**kwargs):
        pytest.fail("token exchange must not run when callback issuer mismatches")

    monkeypatch.setattr(mcp_api, "_exchange_mcp_oauth_code", fail_exchange)

    response = await mcp_oauth_callback(
        _request(
            "/api/mcp/oauth/callback?code=auth-code&state=issuer-mismatch-state"
            "&iss=https%3A%2F%2Fevil.example.com"
        ),
        db,
    )

    assert response.status_code == 307
    assert _redirect_query(response)["mcp_oauth_error"] == ["issuer_mismatch"]
    assert db.query(MCPOAuthGrant).count() == 0


@pytest.mark.asyncio
async def test_callback_rejects_error_response_mismatched_issuer(
    db_session, monkeypatch
):
    db, user, _ = db_session
    _add_callback_client_and_state(
        db,
        user,
        state="issuer-error-state",
        metadata_json={"authorization_response_iss_parameter_supported": True},
    )

    async def fail_exchange(**kwargs):
        pytest.fail("token exchange must not run for authorization error callbacks")

    monkeypatch.setattr(mcp_api, "_exchange_mcp_oauth_code", fail_exchange)

    response = await mcp_oauth_callback(
        _request(
            "/api/mcp/oauth/callback?error=access_denied&state=issuer-error-state"
            "&iss=https%3A%2F%2Fevil.example.com"
        ),
        db,
    )

    assert response.status_code == 307
    assert _redirect_query(response)["mcp_oauth_error"] == ["issuer_mismatch"]
    assert db.query(MCPOAuthGrant).count() == 0


@pytest.mark.asyncio
async def test_callback_sanitizes_authorization_error_response(db_session, monkeypatch):
    db, user, _ = db_session
    _add_callback_client_and_state(
        db,
        user,
        state="authorization-error-state",
        redirect_after="/tools",
    )

    async def fail_exchange(**kwargs):
        pytest.fail("token exchange must not run for authorization error callbacks")

    monkeypatch.setattr(mcp_api, "_exchange_mcp_oauth_code", fail_exchange)
    oversized_error = "access_denied_" + "x" * 1000

    response = await mcp_oauth_callback(
        _request(
            "/api/mcp/oauth/callback?"
            f"error={oversized_error}&state=authorization-error-state"
        ),
        db,
    )

    assert response.status_code == 307
    query = _redirect_query(response)
    assert query["mcp_oauth_error"] == ["token_exchange_failed"]
    assert len(query["mcp_oauth_error_message"][0]) <= 500
    assert query["mcp_oauth_error_message"][0].endswith("...")
    assert db.query(MCPOAuthGrant).count() == 0


@pytest.mark.asyncio
async def test_callback_accepts_absent_issuer_when_not_supported(
    db_session, monkeypatch
):
    db, user, _ = db_session
    _add_callback_client_and_state(
        db,
        user,
        state="issuer-unsupported-state",
        metadata_json={"authorization_response_iss_parameter_supported": False},
    )

    async def fake_exchange(**kwargs):
        return {
            "access_token": "plain-access-token",
            "token_type": "Bearer",
            "scope": "records.read",
        }

    monkeypatch.setattr(mcp_api, "_exchange_mcp_oauth_code", fake_exchange)

    response = await mcp_oauth_callback(
        _request(
            "/api/mcp/oauth/callback?code=auth-code&state=issuer-unsupported-state"
        ),
        db,
    )

    assert response.status_code == 307
    assert db.query(MCPOAuthGrant).count() == 1


@pytest.mark.asyncio
async def test_callback_rejects_state_replay(db_session):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    client = _add_oauth_client(db, server)
    db.add(
        MCPOAuthFlowState(
            state="used-state",
            mcp_server_id=server.id,
            user_id=user.id,
            mcp_oauth_client_id=client.id,
            resource_owner_key="resource-owner-a",
            issuer="https://auth.example.com",
            resource="https://mcp.example.com/mcp",
            scope="records.read",
            code_verifier=mcp_api.encrypt_value("verifier-123"),
            redirect_after="/mcp",
            expires_at=mcp_api._utc_now() + timedelta(minutes=10),
            consumed_at=mcp_api._utc_now(),
        )
    )
    db.commit()

    response = await mcp_oauth_callback(
        _request("/api/mcp/oauth/callback?code=auth-code&state=used-state"),
        db,
    )

    assert response.status_code == 307
    assert _redirect_query(response)["mcp_oauth_error"] == ["state_already_consumed"]


@pytest.mark.asyncio
async def test_callback_rejects_expired_state(db_session):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    client = _add_oauth_client(db, server)
    db.add(
        MCPOAuthFlowState(
            state="expired-state",
            mcp_server_id=server.id,
            user_id=user.id,
            mcp_oauth_client_id=client.id,
            resource_owner_key="resource-owner-a",
            issuer="https://auth.example.com",
            resource="https://mcp.example.com/mcp",
            scope="records.read",
            code_verifier=mcp_api.encrypt_value("verifier-123"),
            redirect_after="/mcp",
            expires_at=mcp_api._utc_now() - timedelta(minutes=1),
        )
    )
    db.commit()

    response = await mcp_oauth_callback(
        _request("/api/mcp/oauth/callback?code=auth-code&state=expired-state"),
        db,
    )

    assert response.status_code == 307
    assert _redirect_query(response)["mcp_oauth_error"] == ["expired_state"]


@pytest.mark.asyncio
async def test_callback_rejects_state_after_user_loses_mcp_access(db_session):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    client = _add_oauth_client(db, server)
    db.add(
        MCPOAuthFlowState(
            state="orphaned-access-state",
            mcp_server_id=server.id,
            user_id=user.id,
            mcp_oauth_client_id=client.id,
            resource_owner_key="resource-owner-a",
            issuer="https://auth.example.com",
            resource="https://mcp.example.com/mcp",
            scope="records.read",
            code_verifier=mcp_api.encrypt_value("verifier-123"),
            redirect_after="/tools",
            expires_at=mcp_api._utc_now() + timedelta(minutes=10),
        )
    )
    db.query(UserMCPServer).filter(
        UserMCPServer.user_id == user.id,
        UserMCPServer.mcpserver_id == server.id,
    ).delete()
    db.commit()

    response = await mcp_oauth_callback(
        _request("/api/mcp/oauth/callback?code=auth-code&state=orphaned-access-state"),
        db,
    )

    assert response.status_code == 307
    assert _redirect_query(response)["mcp_oauth_error"] == ["invalid_state"]
    assert db.query(MCPOAuthGrant).count() == 0


@pytest.mark.asyncio
async def test_callback_rejects_state_after_user_mcp_server_is_deactivated(
    db_session, monkeypatch
):
    db, user, _ = db_session
    server, _, _ = _add_callback_client_and_state(
        db,
        user,
        state="inactive-access-state",
    )
    _set_user_mcp_active(db, user, server, False)

    async def fail_exchange(**kwargs):
        pytest.fail("token exchange must not run for inactive MCP server access")

    monkeypatch.setattr(mcp_api, "_exchange_mcp_oauth_code", fail_exchange)

    response = await mcp_oauth_callback(
        _request("/api/mcp/oauth/callback?code=auth-code&state=inactive-access-state"),
        db,
    )

    assert response.status_code == 307
    assert _redirect_query(response)["mcp_oauth_error"] == ["invalid_state"]
    assert db.query(MCPOAuthGrant).count() == 0


@pytest.mark.asyncio
async def test_callback_reports_token_exchange_failure(db_session, monkeypatch):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    client = _add_oauth_client(db, server)
    db.add(
        MCPOAuthFlowState(
            state="bad-token-state",
            mcp_server_id=server.id,
            user_id=user.id,
            mcp_oauth_client_id=client.id,
            resource_owner_key="resource-owner-a",
            issuer="https://auth.example.com",
            resource="https://mcp.example.com/mcp",
            scope="records.read",
            code_verifier=mcp_api.encrypt_value("verifier-123"),
            redirect_after="/mcp",
            expires_at=mcp_api._utc_now() + timedelta(minutes=10),
        )
    )
    db.commit()

    real_async_client = httpx.AsyncClient

    def async_client_factory(*args, **kwargs):
        return real_async_client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    400,
                    json={
                        "error": "invalid_grant",
                        "error_description": "authorization code is invalid",
                        "access_token": "leaked-access-token",
                        "refresh_token": "leaked-refresh-token",
                    },
                )
            )
        )

    monkeypatch.setattr(mcp_api, "create_mcp_oauth_http_client", async_client_factory)

    response = await mcp_oauth_callback(
        _request("/api/mcp/oauth/callback?code=auth-code&state=bad-token-state"),
        db,
    )

    assert response.status_code == 307
    query = _redirect_query(response)
    assert query["mcp_oauth_error"] == ["token_exchange_failed"]
    assert query["mcp_oauth_error_message"] == ["authorization code is invalid"]
    assert "leaked-access-token" not in response.headers["location"]
    assert "leaked-refresh-token" not in response.headers["location"]
    assert db.query(MCPOAuthGrant).count() == 0
    flow_state = db.query(MCPOAuthFlowState).filter_by(state="bad-token-state").one()
    assert flow_state.consumed_at is not None

    async def fail_exchange(**kwargs):
        pytest.fail("terminal failed state must not be exchanged again")

    monkeypatch.setattr(mcp_api, "_exchange_mcp_oauth_code", fail_exchange)
    response = await mcp_oauth_callback(
        _request("/api/mcp/oauth/callback?code=auth-code&state=bad-token-state"),
        db,
    )
    assert response.status_code == 307
    assert _redirect_query(response)["mcp_oauth_error"] == ["state_already_consumed"]


@pytest.mark.asyncio
async def test_status_and_delete_are_scoped_to_current_user(db_session):
    db, user, other_user = db_session
    server = _add_mcp_oauth_server(db, user)
    client = _add_oauth_client(db, server)
    own_grant = MCPOAuthGrant(
        mcp_server_id=server.id,
        user_id=user.id,
        mcp_oauth_client_id=client.id,
        resource_owner_key="resource-owner-a",
        issuer="https://auth.example.com",
        resource="https://mcp.example.com/mcp",
        scope="records.read",
        access_token=mcp_api.encrypt_value("own-access-token"),
    )
    other_grant = MCPOAuthGrant(
        mcp_server_id=server.id,
        user_id=other_user.id,
        mcp_oauth_client_id=client.id,
        resource_owner_key="resource-owner-b",
        issuer="https://auth.example.com",
        resource="https://mcp.example.com/mcp",
        scope="records.read",
        access_token=mcp_api.encrypt_value("other-access-token"),
    )
    db.add_all([own_grant, other_grant])
    db.commit()
    db.refresh(own_grant)
    db.refresh(other_grant)

    status_response = await get_mcp_oauth_status(server.id, user, db)

    assert isinstance(status_response, MCPOAuthStatusResponse)
    assert [grant.id for grant in status_response.grants] == [own_grant.id]

    with pytest.raises(mcp_api.HTTPException) as exc:
        await delete_mcp_oauth_grant(server.id, other_grant.id, user, db)
    assert exc.value.status_code == 404

    await delete_mcp_oauth_grant(server.id, own_grant.id, user, db)
    db.refresh(own_grant)
    assert own_grant.status == "revoked"
    assert own_grant.revoked_at is not None

    status_response = await get_mcp_oauth_status(server.id, user, db)
    assert status_response.grants == []


@pytest.mark.asyncio
async def test_delete_grant_revokes_external_tokens_when_endpoint_is_advertised(
    db_session, monkeypatch
):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    client = _add_oauth_client(
        db,
        server,
        client_secret=encrypt_value("client-secret"),
        token_endpoint_auth_method="client_secret_post",
        metadata_json={"revocation_endpoint": "https://auth.example.com/revoke"},
    )
    grant = MCPOAuthGrant(
        mcp_server_id=server.id,
        user_id=user.id,
        mcp_oauth_client_id=client.id,
        resource_owner_key=f"xagent:user:{user.id}",
        issuer="https://auth.example.com",
        resource="https://mcp.example.com/mcp",
        scope="records.read",
        access_token=encrypt_value("access-token"),
        refresh_token=encrypt_value("refresh-token"),
    )
    db.add(grant)
    db.commit()
    db.refresh(grant)
    requests: list[dict[str, list[str]]] = []
    real_async_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(parse_qs(request.content.decode()))
        return httpx.Response(200)

    def async_client_factory(*args, **kwargs):
        return real_async_client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(mcp_api, "create_mcp_oauth_http_client", async_client_factory)

    await delete_mcp_oauth_grant(server.id, grant.id, user, db)

    assert [request["token"] for request in requests] == [
        ["access-token"],
        ["refresh-token"],
    ]
    assert [request["token_type_hint"] for request in requests] == [
        ["access_token"],
        ["refresh_token"],
    ]
    assert all(request["client_secret"] == ["client-secret"] for request in requests)
    db.refresh(grant)
    assert grant.status == "revoked"
    assert grant.revoked_at is not None


@pytest.mark.asyncio
async def test_delete_mcp_server_revokes_external_tokens_when_endpoint_is_advertised(
    db_session, monkeypatch
):
    """M3's remaining half: disconnecting a server must actually reach the
    provider's revocation_endpoint, not just flip local status (which was
    already fixed in an earlier round but never called the external API)."""
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    client = _add_oauth_client(
        db,
        server,
        client_secret=encrypt_value("client-secret"),
        token_endpoint_auth_method="client_secret_post",
        metadata_json={"revocation_endpoint": "https://auth.example.com/revoke"},
    )
    grant = MCPOAuthGrant(
        mcp_server_id=server.id,
        user_id=user.id,
        mcp_oauth_client_id=client.id,
        resource_owner_key=f"xagent:user:{user.id}",
        issuer="https://auth.example.com",
        resource="https://mcp.example.com/mcp",
        scope="records.read",
        access_token=encrypt_value("access-token"),
        refresh_token=encrypt_value("refresh-token"),
    )
    db.add(grant)
    db.commit()
    db.refresh(grant)
    grant_id = grant.id
    requests: list[dict[str, list[str]]] = []
    real_async_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(parse_qs(request.content.decode()))
        return httpx.Response(200)

    def async_client_factory(*args, **kwargs):
        return real_async_client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(mcp_api, "create_mcp_oauth_http_client", async_client_factory)

    await delete_mcp_server(server.id, current_user=user, db=db)

    assert [request["token"] for request in requests] == [
        ["access-token"],
        ["refresh-token"],
    ]
    # Purged outright rather than left as an inert "revoked" row (F10).
    assert (
        db.query(MCPOAuthGrant).filter(MCPOAuthGrant.id == grant_id).one_or_none()
        is None
    )


@pytest.mark.asyncio
async def test_delete_mcp_server_purges_the_users_own_flow_state_rows(db_session):
    """A leftover MCPOAuthFlowState (e.g. an abandoned/expired connect
    attempt) carries a per-user code_verifier secret that nothing else
    sweeps — disconnect must not leave it behind (F10)."""
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    client = _add_oauth_client(db, server)
    flow_state = MCPOAuthFlowState(
        state="leftover-state",
        mcp_server_id=server.id,
        user_id=user.id,
        mcp_oauth_client_id=client.id,
        resource_owner_key=f"xagent:user:{user.id}",
        issuer="https://auth.example.com",
        resource="https://mcp.example.com/mcp",
        scope="records.read",
        code_verifier=encrypt_value("verifier-123"),
        expires_at=mcp_api._utc_now() + timedelta(minutes=10),
    )
    db.add(flow_state)
    db.commit()
    flow_state_id = flow_state.id

    await delete_mcp_server(server.id, current_user=user, db=db)

    assert (
        db.query(MCPOAuthFlowState)
        .filter(MCPOAuthFlowState.id == flow_state_id)
        .one_or_none()
        is None
    )


@pytest.mark.asyncio
async def test_delete_grant_continues_local_revoke_when_token_decryption_fails(
    db_session, monkeypatch
):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    client = _add_oauth_client(
        db,
        server,
        token_endpoint_auth_method="none",
        metadata_json={"revocation_endpoint": "https://auth.example.com/revoke"},
    )
    grant = MCPOAuthGrant(
        mcp_server_id=server.id,
        user_id=user.id,
        mcp_oauth_client_id=client.id,
        resource_owner_key=f"xagent:user:{user.id}",
        issuer="https://auth.example.com",
        resource="https://mcp.example.com/mcp",
        scope="records.read",
        access_token="not-encrypted-token",
    )
    db.add(grant)
    db.commit()
    db.refresh(grant)
    real_async_client = httpx.AsyncClient
    real_decrypt_value = mcp_api.decrypt_value

    def fail_target_token_decrypt(value: str) -> str:
        if value == "not-encrypted-token":
            raise ValueError("cannot decrypt token")
        return real_decrypt_value(value)

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("External revocation should be skipped when token decrypt fails")

    def async_client_factory(*args, **kwargs):
        return real_async_client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(mcp_api, "decrypt_value", fail_target_token_decrypt)
    monkeypatch.setattr(mcp_api, "create_mcp_oauth_http_client", async_client_factory)

    await delete_mcp_oauth_grant(server.id, grant.id, user, db)

    db.refresh(grant)
    assert grant.status == "revoked"
    assert grant.revoked_at is not None


@pytest.mark.asyncio
async def test_status_only_reports_grants_matching_current_oauth_config(db_session):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    stale_client = _add_oauth_client(db, server, client_id="stale-client")
    current_client = _add_oauth_client(db, server, client_id="client-123")
    stale_grant = MCPOAuthGrant(
        mcp_server_id=server.id,
        user_id=user.id,
        mcp_oauth_client_id=stale_client.id,
        resource_owner_key="resource-owner-a",
        issuer="https://auth.example.com",
        resource="https://mcp.example.com/mcp",
        scope="records.read",
        access_token=mcp_api.encrypt_value("stale-access-token"),
    )
    current_grant = MCPOAuthGrant(
        mcp_server_id=server.id,
        user_id=user.id,
        mcp_oauth_client_id=current_client.id,
        resource_owner_key="resource-owner-b",
        issuer="https://auth.example.com",
        resource="https://mcp.example.com/mcp",
        scope="records.write records.read",
        access_token=mcp_api.encrypt_value("current-access-token"),
    )
    db.add_all([stale_grant, current_grant])
    db.commit()
    db.refresh(stale_grant)
    db.refresh(current_grant)

    status_response = await get_mcp_oauth_status(server.id, user, db)

    assert [grant.id for grant in status_response.grants] == [current_grant.id]


@pytest.mark.asyncio
async def test_status_reports_discovered_grant_without_configured_selectors(db_session):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    server.auth = {"type": "mcp_oauth", "scope": "records.read"}
    client = _add_oauth_client(db, server, client_id="dynamically-registered-client")
    grant = MCPOAuthGrant(
        mcp_server_id=server.id,
        user_id=user.id,
        mcp_oauth_client_id=client.id,
        resource_owner_key=f"xagent:user:{user.id}",
        issuer="https://auth.example.com",
        resource="https://mcp.example.com/mcp",
        scope="records.read",
        access_token=mcp_api.encrypt_value("access-token"),
    )
    db.add(grant)
    db.commit()
    db.refresh(grant)

    status_response = await get_mcp_oauth_status(server.id, user, db)

    assert [item.id for item in status_response.grants] == [grant.id]
