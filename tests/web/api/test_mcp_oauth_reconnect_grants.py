"""A successful MCP OAuth callback replaces the owner's earlier grants.

The grant lookup hash includes the granted scope, so reconnecting with a
different granted scope stores a new row. Once that grant is one grant
selection would use and it can be refreshed or does not expire, the callback
deletes the owner's other active grants for the same server, client, issuer
and resource, so one usable grant is left.
Grants of other owners, users, clients, issuers, resources and servers are
not touched, a callback that fails keeps the earlier grant, and the runtime
refresh path keeps working with the grant that is left.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette.requests import Request

from xagent.core.utils.encryption import decrypt_value
from xagent.web.api import mcp as mcp_api
from xagent.web.api.mcp import mcp_oauth_callback
from xagent.web.models import MCPOAuthClient, MCPOAuthFlowState, MCPOAuthGrant
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.user import User
from xagent.web.services import mcp_oauth as mcp_oauth_service
from xagent.web.services.mcp_oauth import (
    resolve_mcp_oauth_runtime_auth,
    select_mcp_oauth_grants,
)

MCP_URL = "https://mcp.example.com/mcp"
ISSUER = "https://auth.example.com"
OWNER = "actor:workspace:alice"
FULL = "files.read records.read records.write"
NARROW = "records.read records.write"
REQUIRED = "records.read"
AUTH = {"type": "mcp_oauth", "scope": FULL, "required_scope": REQUIRED}
ISSUED_TOKENS = {
    "access_token": "issued-access-token",
    "refresh_token": "issued-refresh-token",
    "token_type": "Bearer",
    "expires_in": 3600,
}


@pytest.fixture()
def engine(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'reconnect-grants.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture()
def db_session(engine):
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    user = User(username="alice", password_hash="x", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    yield db, user
    db.close()


@pytest.fixture()
def provider(monkeypatch):
    """Record every provider-side revocation instead of sending it."""
    revoked: dict[str, list] = {"issued": [], "grants": []}

    async def record_issued(snapshot):
        revoked["issued"].append(snapshot)

    async def record_grant(snapshot):
        revoked["grants"].append(snapshot)

    monkeypatch.setattr(
        mcp_api, "_revoke_mcp_oauth_issued_token_externally", record_issued
    )
    monkeypatch.setattr(
        mcp_api, "_revoke_mcp_oauth_grant_snapshot_externally", record_grant
    )
    return revoked


def _token_response(monkeypatch, token_data: dict) -> None:
    async def fake_exchange(**_kwargs):
        return dict(token_data)

    monkeypatch.setattr(mcp_api, "_exchange_mcp_oauth_code", fake_exchange)


def _add_server(db, user: User, auth: dict, *, name: str = "records") -> MCPServer:
    server = MCPServer.from_config(
        {
            "name": name,
            "managed": "external",
            "transport": "streamable_http",
            "url": MCP_URL,
            "auth": auth,
        }
    )
    db.add(server)
    db.commit()
    db.refresh(server)
    db.add(
        UserMCPServer(
            user_id=user.id,
            mcpserver_id=server.id,
            is_owner=False,
            is_active=True,
        )
    )
    db.commit()
    return server


def _add_client(db, server: MCPServer, *, client_id: str = "client-1"):
    client = MCPOAuthClient(
        mcp_server_id=server.id,
        issuer=ISSUER,
        authorization_endpoint=f"{ISSUER}/authorize",
        token_endpoint=f"{ISSUER}/token",
        client_id=client_id,
        token_endpoint_auth_method="none",
        redirect_uri="https://xagent.example.com/api/mcp/oauth/callback",
        metadata_json={"revocation_endpoint": f"{ISSUER}/revoke"},
    )
    db.add(client)
    db.commit()
    db.refresh(client)
    return client


def _add_grant(
    db,
    *,
    server: MCPServer,
    user: User,
    client: MCPOAuthClient,
    scope: str,
    owner: str = OWNER,
    issuer: str = ISSUER,
    resource: str = MCP_URL,
    status: str = "active",
    access_token: str = "earlier-access-token",
) -> int:
    grant = MCPOAuthGrant(
        mcp_server_id=server.id,
        user_id=user.id,
        mcp_oauth_client_id=client.id,
        resource_owner_key=owner,
        issuer=issuer,
        resource=resource,
        scope=scope,
        access_token=mcp_api.encrypt_value(access_token),
        refresh_token=mcp_api.encrypt_value("earlier-refresh-token"),
        token_type="Bearer",
        status=status,
        expires_at=mcp_api._utc_now() + timedelta(hours=1),
    )
    db.add(grant)
    db.commit()
    return int(grant.id)


def _add_bystander_grant(db, *, server: MCPServer, user: User, client) -> None:
    """Another owner's grant, added last so that SQLite does not hand a
    deleted grant's id to the grant that replaces it."""
    _add_grant(
        db,
        server=server,
        user=user,
        client=client,
        scope=FULL,
        owner="actor:workspace:carol",
    )


def _add_flow(db, *, server: MCPServer, user: User, client, state: str) -> None:
    generation = (
        db.query(UserMCPServer.lifecycle_generation)
        .filter(
            UserMCPServer.user_id == user.id,
            UserMCPServer.mcpserver_id == server.id,
        )
        .scalar()
    )
    db.add(
        MCPOAuthFlowState(
            state=state,
            mcp_server_id=server.id,
            user_id=user.id,
            association_lifecycle_generation=generation,
            mcp_oauth_client_id=client.id,
            resource_owner_key=OWNER,
            issuer=ISSUER,
            resource=MCP_URL,
            scope=FULL,
            code_verifier=mcp_api.encrypt_value("verifier-123"),
            redirect_after="/tools",
            expires_at=mcp_api._utc_now() + timedelta(minutes=10),
        )
    )
    db.commit()


def _callback_request(state: str, query: str = "code=auth-code") -> Request:
    cookie = (
        f"{mcp_api.MCP_OAUTH_STATE_COOKIE}="
        f"{mcp_api._mcp_oauth_state_cookie_value(state)}"
    )
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/mcp/oauth/callback",
            "query_string": f"{query}&state={state}".encode(),
            "headers": [(b"cookie", cookie.encode())],
        }
    )


def _redirect_query(response) -> dict[str, list[str]]:
    return parse_qs(urlparse(response.headers["location"]).query)


def _owner_grants(db, server: MCPServer) -> list[MCPOAuthGrant]:
    db.expire_all()
    return (
        db.query(MCPOAuthGrant)
        .filter(
            MCPOAuthGrant.mcp_server_id == server.id,
            MCPOAuthGrant.resource_owner_key == OWNER,
        )
        .order_by(MCPOAuthGrant.id)
        .all()
    )


def _selected_ids(db, server: MCPServer, user: User, auth: dict) -> list[int]:
    return [
        int(grant.id)
        for grant in select_mcp_oauth_grants(
            db,
            server_id=server.id,
            user_id=user.id,
            auth_config=auth,
            resource_owner_key=OWNER,
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("auth", "earlier_scopes", "granted_scope"),
    (
        pytest.param(AUTH, (FULL,), NARROW, id="narrower-grant"),
        pytest.param(AUTH, (NARROW,), FULL, id="wider-grant"),
        pytest.param(AUTH, (FULL, REQUIRED), NARROW, id="two-earlier-grants"),
        pytest.param({"type": "mcp_oauth"}, (FULL,), NARROW, id="no-configured-scope"),
    ),
)
async def test_reconnect_leaves_one_usable_grant(
    db_session, monkeypatch, provider, caplog, auth, earlier_scopes, granted_scope
):
    db, user = db_session
    server = _add_server(db, user, auth)
    client = _add_client(db, server)
    earlier_ids = [
        _add_grant(db, server=server, user=user, client=client, scope=scope)
        for scope in earlier_scopes
    ]
    _add_bystander_grant(db, server=server, user=user, client=client)
    assert sorted(_selected_ids(db, server, user, auth)) == earlier_ids
    _add_flow(db, server=server, user=user, client=client, state="state-1")
    _token_response(monkeypatch, {**ISSUED_TOKENS, "scope": granted_scope})

    with caplog.at_level(logging.INFO, logger=mcp_api.logger.name):
        response = await mcp_oauth_callback(_callback_request("state-1"), db)

    assert _redirect_query(response) == {"mcp_oauth_success": ["1"]}
    grants = _owner_grants(db, server)
    assert [(grant.scope, grant.status) for grant in grants] == [
        (granted_scope, "active")
    ]
    assert int(grants[0].id) not in earlier_ids
    assert decrypt_value(str(grants[0].access_token)) == "issued-access-token"
    assert _selected_ids(db, server, user, auth) == [int(grants[0].id)]
    # The replaced grants share the new grant's client and account at the
    # provider, so nothing is revoked there.
    assert provider == {"issued": [], "grants": []}
    assert f"count={len(earlier_ids)}" in caplog.text
    assert "earlier-access-token" not in caplog.text
    assert "issued-access-token" not in caplog.text


@pytest.mark.asyncio
async def test_reconnect_with_the_same_scope_updates_the_grant_in_place(
    db_session, monkeypatch, provider
):
    db, user = db_session
    server = _add_server(db, user, AUTH)
    client = _add_client(db, server)
    grant_id = _add_grant(db, server=server, user=user, client=client, scope=NARROW)
    _add_flow(db, server=server, user=user, client=client, state="state-same")
    _token_response(
        monkeypatch,
        {
            "access_token": "issued-access-token",
            "token_type": "Bearer",
            "scope": NARROW,
        },
    )

    response = await mcp_oauth_callback(_callback_request("state-same"), db)

    assert _redirect_query(response) == {"mcp_oauth_success": ["1"]}
    grants = _owner_grants(db, server)
    assert [int(grant.id) for grant in grants] == [grant_id]
    assert decrypt_value(str(grants[0].access_token)) == "issued-access-token"
    # Updated in place: a response without a refresh token keeps the old one.
    assert decrypt_value(str(grants[0].refresh_token)) == "earlier-refresh-token"
    assert provider == {"issued": [], "grants": []}


@pytest.mark.asyncio
async def test_reconnect_keeps_every_grant_outside_the_replaced_set(
    db_session, monkeypatch, provider
):
    db, user = db_session
    other_user = User(username="bob", password_hash="x", is_admin=False)
    db.add(other_user)
    db.commit()
    server = _add_server(db, user, AUTH)
    other_server = _add_server(db, user, AUTH, name="records-copy")
    client = _add_client(db, server)
    other_client = _add_client(db, server, client_id="client-2")
    replaced_id = _add_grant(db, server=server, user=user, client=client, scope=FULL)
    kept = {
        "other owner": _add_grant(
            db,
            server=server,
            user=user,
            client=client,
            scope=FULL,
            owner="actor:workspace:bob",
        ),
        "other user": _add_grant(
            db, server=server, user=other_user, client=client, scope=FULL
        ),
        "other client": _add_grant(
            db, server=server, user=user, client=other_client, scope=FULL
        ),
        "other issuer": _add_grant(
            db,
            server=server,
            user=user,
            client=client,
            scope=FULL,
            issuer="https://other-auth.example.com",
        ),
        "other resource": _add_grant(
            db,
            server=server,
            user=user,
            client=client,
            scope=FULL,
            resource="https://mcp.example.com/other",
        ),
        "other server": _add_grant(
            db, server=other_server, user=user, client=client, scope=FULL
        ),
        "revoked": _add_grant(
            db,
            server=server,
            user=user,
            client=client,
            scope=REQUIRED,
            status="revoked",
        ),
    }
    _add_flow(db, server=server, user=user, client=client, state="state-others")
    _token_response(monkeypatch, {**ISSUED_TOKENS, "scope": NARROW})

    response = await mcp_oauth_callback(_callback_request("state-others"), db)

    assert _redirect_query(response) == {"mcp_oauth_success": ["1"]}
    db.expire_all()
    assert db.get(MCPOAuthGrant, replaced_id) is None
    survivors = {
        label: db.get(MCPOAuthGrant, grant_id) for label, grant_id in kept.items()
    }
    assert {
        label: (grant.status, decrypt_value(str(grant.access_token)))
        for label, grant in survivors.items()
        if grant is not None
    } == {
        label: ("revoked" if label == "revoked" else "active", "earlier-access-token")
        for label in kept
    }
    new_grants = [grant for grant in _owner_grants(db, server) if grant.scope == NARROW]
    assert len(new_grants) == 1
    assert provider == {"issued": [], "grants": []}


@pytest.mark.asyncio
async def test_new_grant_that_selection_would_not_use_replaces_nothing(
    db_session, monkeypatch, provider
):
    """Without ``required_scope`` a narrower grant is stored but not usable, so
    the usable grant it would otherwise replace stays."""
    db, user = db_session
    auth = {"type": "mcp_oauth", "scope": FULL}
    server = _add_server(db, user, auth)
    client = _add_client(db, server)
    usable_id = _add_grant(db, server=server, user=user, client=client, scope=FULL)
    _add_flow(db, server=server, user=user, client=client, state="state-narrow")
    _token_response(monkeypatch, {**ISSUED_TOKENS, "scope": NARROW})

    response = await mcp_oauth_callback(_callback_request("state-narrow"), db)

    assert _redirect_query(response) == {"mcp_oauth_success": ["1"]}
    assert [(grant.scope, grant.status) for grant in _owner_grants(db, server)] == [
        (FULL, "active"),
        (NARROW, "active"),
    ]
    assert _selected_ids(db, server, user, auth) == [usable_id]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "expires_in", (120, 3600), ids=("expires-within-minutes", "expires-in-an-hour")
)
async def test_new_grant_that_cannot_be_refreshed_replaces_nothing(
    db_session, monkeypatch, provider, expires_in
):
    """A new grant without a refresh token stops working when its access token
    expires, so the earlier grant that can be refreshed stays, and runtime
    uses it once the new access token is due for refresh."""
    db, user = db_session
    server = _add_server(db, user, AUTH)
    client = _add_client(db, server)
    earlier_id = _add_grant(db, server=server, user=user, client=client, scope=FULL)
    _add_flow(db, server=server, user=user, client=client, state="state-no-refresh")
    _token_response(
        monkeypatch,
        {
            "access_token": "issued-access-token",
            "token_type": "Bearer",
            "scope": NARROW,
            "expires_in": expires_in,
        },
    )

    response = await mcp_oauth_callback(_callback_request("state-no-refresh"), db)

    assert _redirect_query(response) == {"mcp_oauth_success": ["1"]}
    earlier, new = _owner_grants(db, server)
    assert (int(earlier.id), earlier.scope, earlier.status) == (
        earlier_id,
        FULL,
        "active",
    )
    assert decrypt_value(str(earlier.refresh_token)) == "earlier-refresh-token"
    assert (new.scope, new.status, new.refresh_token) == (NARROW, "active", None)
    new.expires_at = mcp_api._utc_now() + timedelta(minutes=2)
    db.commit()

    runtime_auth = await resolve_mcp_oauth_runtime_auth(
        db,
        server_id=server.id,
        user_id=user.id,
        auth_config=AUTH,
        resource_owner_key=OWNER,
    )

    assert runtime_auth.grant_id == earlier_id
    assert runtime_auth.access_token == "earlier-access-token"
    assert provider == {"issued": [], "grants": []}


@pytest.mark.asyncio
async def test_new_grant_whose_access_token_does_not_expire_replaces_others(
    db_session, monkeypatch, provider
):
    """Without ``expires_in`` the new access token does not expire, so the new
    grant needs no refresh token to replace the earlier grant."""
    db, user = db_session
    server = _add_server(db, user, AUTH)
    client = _add_client(db, server)
    _add_grant(db, server=server, user=user, client=client, scope=FULL)
    _add_flow(db, server=server, user=user, client=client, state="state-no-expiry")
    _token_response(
        monkeypatch,
        {
            "access_token": "issued-access-token",
            "token_type": "Bearer",
            "scope": NARROW,
        },
    )

    response = await mcp_oauth_callback(_callback_request("state-no-expiry"), db)

    assert _redirect_query(response) == {"mcp_oauth_success": ["1"]}
    [grant] = _owner_grants(db, server)
    assert (grant.scope, grant.refresh_token, grant.expires_at) == (NARROW, None, None)
    assert _selected_ids(db, server, user, AUTH) == [int(grant.id)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "token_data", "error_code"),
    (
        pytest.param(
            "code=auth-code",
            {**ISSUED_TOKENS, "scope": "files.read"},
            "insufficient_scope",
            id="insufficient-scope",
        ),
        pytest.param(
            "error=access_denied",
            ISSUED_TOKENS,
            "provider_error",
            id="provider-refusal",
        ),
        pytest.param(
            "code=auth-code",
            {**ISSUED_TOKENS, "scope": NARROW, "token_type": "x" * 51},
            "invalid_resource",
            id="failure-after-the-replacement",
        ),
    ),
)
async def test_failed_callback_keeps_the_earlier_grant(
    db_session, monkeypatch, provider, query, token_data, error_code
):
    db, user = db_session
    server = _add_server(db, user, AUTH)
    client = _add_client(db, server)
    earlier_id = _add_grant(db, server=server, user=user, client=client, scope=FULL)
    _add_flow(db, server=server, user=user, client=client, state="state-fail")
    _token_response(monkeypatch, token_data)

    response = await mcp_oauth_callback(_callback_request("state-fail", query), db)

    assert _redirect_query(response)["mcp_oauth_error"] == [error_code]
    grants = _owner_grants(db, server)
    assert [(int(grant.id), grant.status) for grant in grants] == [
        (earlier_id, "active")
    ]
    assert decrypt_value(str(grants[0].access_token)) == "earlier-access-token"
    assert provider["grants"] == []


@pytest.mark.asyncio
async def test_failed_commit_keeps_the_earlier_grant(db_session, monkeypatch, provider):
    db, user = db_session
    server = _add_server(db, user, AUTH)
    client = _add_client(db, server)
    earlier_id = _add_grant(db, server=server, user=user, client=client, scope=FULL)
    _add_flow(db, server=server, user=user, client=client, state="state-commit")
    _token_response(monkeypatch, {**ISSUED_TOKENS, "scope": NARROW})

    real_commit = db.commit

    def fail_to_commit_a_new_grant():
        if any(isinstance(pending, MCPOAuthGrant) for pending in db.new):
            raise RuntimeError("commit failed")
        real_commit()

    monkeypatch.setattr(db, "commit", fail_to_commit_a_new_grant)

    response = await mcp_oauth_callback(_callback_request("state-commit"), db)

    monkeypatch.undo()
    assert _redirect_query(response)["mcp_oauth_error"] == ["token_exchange_failed"]
    assert [(int(grant.id), grant.status) for grant in _owner_grants(db, server)] == [
        (earlier_id, "active")
    ]


def _expire(db, grant: MCPOAuthGrant) -> None:
    grant.expires_at = mcp_api._utc_now() - timedelta(minutes=1)
    db.commit()


@pytest.mark.asyncio
async def test_refresh_to_the_replaced_scope_does_not_collide(
    db_session, monkeypatch, provider
):
    """The replaced row is gone, so a refresh that returns its scope can take
    over that lookup hash."""
    db, user = db_session
    server = _add_server(db, user, AUTH)
    client = _add_client(db, server)
    _add_grant(db, server=server, user=user, client=client, scope=FULL)
    _add_flow(db, server=server, user=user, client=client, state="state-refresh")
    _token_response(monkeypatch, {**ISSUED_TOKENS, "scope": NARROW})
    response = await mcp_oauth_callback(_callback_request("state-refresh"), db)
    assert _redirect_query(response) == {"mcp_oauth_success": ["1"]}
    [grant] = _owner_grants(db, server)
    _expire(db, grant)
    refreshed_tokens: list[str] = []

    async def fake_refresh(_client, *, refresh_token, resource, client=None):
        refreshed_tokens.append(refresh_token)
        return {"access_token": "refreshed-access-token", "scope": FULL}

    monkeypatch.setattr(mcp_oauth_service, "_refresh_mcp_oauth_grant", fake_refresh)

    runtime_auth = await resolve_mcp_oauth_runtime_auth(
        db,
        server_id=server.id,
        user_id=user.id,
        auth_config=AUTH,
        resource_owner_key=OWNER,
    )

    assert refreshed_tokens == ["issued-refresh-token"]
    assert runtime_auth.refreshed is True
    assert runtime_auth.grant_id == int(grant.id)
    assert runtime_auth.scope == FULL
    assert [(int(row.id), row.scope) for row in _owner_grants(db, server)] == [
        (int(grant.id), FULL)
    ]


@pytest.mark.asyncio
async def test_runtime_refreshing_a_replaced_grant_moves_to_the_new_grant(
    engine, db_session, monkeypatch, provider
):
    """A refresh of the earlier grant that is in flight while the callback
    replaces it selects the new grant instead of failing."""
    db, user = db_session
    server = _add_server(db, user, AUTH)
    client = _add_client(db, server)
    earlier_id = _add_grant(db, server=server, user=user, client=client, scope=FULL)
    _expire(db, db.get(MCPOAuthGrant, earlier_id))
    _add_bystander_grant(db, server=server, user=user, client=client)
    _add_flow(db, server=server, user=user, client=client, state="state-inflight")
    _token_response(monkeypatch, {**ISSUED_TOKENS, "scope": NARROW})
    callback_db = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    refreshed_tokens: list[str] = []

    async def reconnect_during_refresh(
        _client, *, refresh_token, resource, client=None
    ):
        refreshed_tokens.append(refresh_token)
        response = await mcp_oauth_callback(
            _callback_request("state-inflight"), callback_db
        )
        assert _redirect_query(response) == {"mcp_oauth_success": ["1"]}
        return {"access_token": "refreshed-earlier-access-token", "scope": FULL}

    monkeypatch.setattr(
        mcp_oauth_service, "_refresh_mcp_oauth_grant", reconnect_during_refresh
    )

    try:
        runtime_auth = await resolve_mcp_oauth_runtime_auth(
            db,
            server_id=server.id,
            user_id=user.id,
            auth_config=AUTH,
            resource_owner_key=OWNER,
        )
    finally:
        callback_db.close()

    assert refreshed_tokens == ["earlier-refresh-token"]
    [grant] = _owner_grants(db, server)
    assert grant.scope == NARROW
    assert runtime_auth.grant_id == int(grant.id) != earlier_id
    assert runtime_auth.access_token == "issued-access-token"
    assert runtime_auth.refreshed is False


@pytest.mark.parametrize(
    ("auth", "selected_scope"),
    (
        pytest.param(AUTH, REQUIRED, id="required-scope"),
        pytest.param(
            {"type": "mcp_oauth", "scope": "records.write records.read"},
            NARROW,
            id="whole-scope",
        ),
        pytest.param(
            {"type": "mcp_oauth", "scope": ["records.write", "records.read"]},
            NARROW,
            id="whole-scope-list",
        ),
        pytest.param({"type": "mcp_oauth"}, "", id="no-scope"),
        pytest.param({"type": "oauth", "scope": FULL}, "", id="another-auth-type"),
        pytest.param(None, "", id="no-auth"),
    ),
)
def test_the_scope_a_new_grant_must_cover_to_replace_others(auth, selected_scope):
    server = SimpleNamespace(auth=auth)

    assert mcp_api._mcp_oauth_selected_scope(server) == selected_scope
