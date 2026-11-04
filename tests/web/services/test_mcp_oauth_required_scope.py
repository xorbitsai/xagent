"""An MCP OAuth configuration can require less than it requests.

``auth.scope`` is what an authorization requests. The optional
``auth.required_scope`` is the part of it a grant must include to count as
usable: when the connector is listed as connected, at runtime and after a
token refresh. Without it the whole requested scope stays required.

The remote Atlassian connector requests a product subset but requires only the
basic identity scopes, because Atlassian grants product scopes only for the
products the consenting account's site has.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.utils.encryption import decrypt_value, encrypt_value
from xagent.web.builtin_mcp_registry import (
    ATLASSIAN_MCP_OAUTH_REQUIRED_SCOPE,
    ATLASSIAN_MCP_OAUTH_SCOPE,
    get_builtin_public_mcp_app,
)
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.mcp_oauth import MCPOAuthClient, MCPOAuthGrant
from xagent.web.models.user import User
from xagent.web.services.mcp_oauth import (
    MCPOAuthDiscoveryError,
    MCPOAuthRuntimeError,
    mcp_oauth_required_scope,
    mcp_oauth_required_scope_is_set,
    resolve_mcp_oauth_runtime_auth,
    select_mcp_oauth_grants,
    select_mcp_oauth_owner,
)
from xagent.web.services.mcp_runtime import build_mcp_runtime_connection

MCP_URL = "https://mcp.atlassian.com/v2/mcp"
ISSUER = "https://auth.atlassian.com/tenant"
TOKEN_ENDPOINT = "https://auth.atlassian.com/oauth/token"
ACTOR_OWNER = "actor:workspace:alice"
BASIC_IDENTITY_SCOPES = ("email", "offline_access", "read:account", "read:me")
BITBUCKET_SCOPES = ("read:bitbucket:agent-interface", "write:bitbucket:agent-interface")


def _scope_without(*left_out: str) -> str:
    return " ".join(
        scope for scope in ATLASSIAN_MCP_OAUTH_SCOPE.split() if scope not in left_out
    )


# What a site with Jira and Confluence but no Bitbucket was granted for the
# catalog request: everything except the two Bitbucket scopes (11 of 13).
GRANTED_WITHOUT_BITBUCKET = _scope_without(*BITBUCKET_SCOPES)


def _atlassian_auth() -> dict:
    app = get_builtin_public_mcp_app("atlassian")
    assert app is not None
    return dict(app["launch_config"]["auth"])


@pytest.fixture()
def db_session(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'required-scope.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    user = User(username="alice", password_hash="x", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    yield db, user
    db.close()
    engine.dispose()


def _add_server(db, user: User, auth: dict) -> MCPServer:
    server = MCPServer.from_config(
        {
            "name": "atlassian",
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


def _add_grant(
    db,
    *,
    server: MCPServer,
    user: User,
    scope: str,
    access_token: str = "granted-access-token",
    refresh_token: str | None = None,
    expires_at: datetime | None = None,
) -> MCPOAuthGrant:
    oauth_client = (
        db.query(MCPOAuthClient)
        .filter(MCPOAuthClient.mcp_server_id == server.id)
        .first()
    )
    if oauth_client is None:
        oauth_client = MCPOAuthClient(
            mcp_server_id=server.id,
            issuer=ISSUER,
            authorization_endpoint="https://auth.atlassian.com/authorize",
            token_endpoint=TOKEN_ENDPOINT,
            client_id="atlassian-dynamic-client",
            token_endpoint_auth_method="none",
            redirect_uri="https://api.xagent.test/api/mcp/oauth/callback",
        )
        db.add(oauth_client)
        db.flush()
    grant = MCPOAuthGrant(
        mcp_server_id=server.id,
        user_id=user.id,
        mcp_oauth_client_id=oauth_client.id,
        resource_owner_key=ACTOR_OWNER,
        issuer=ISSUER,
        resource=MCP_URL,
        scope=scope,
        access_token=encrypt_value(access_token),
        refresh_token=encrypt_value(refresh_token) if refresh_token else None,
        expires_at=expires_at,
        status="active",
    )
    db.add(grant)
    db.commit()
    db.refresh(grant)
    return grant


def _select(db, server: MCPServer, user: User, auth: dict) -> list:
    return select_mcp_oauth_grants(
        db,
        server_id=server.id,
        user_id=user.id,
        auth_config=auth,
        resource_owner_key=ACTOR_OWNER,
    )


async def _resolve(db, server: MCPServer, user: User, auth: dict, **kwargs):
    return await resolve_mcp_oauth_runtime_auth(
        db,
        server_id=server.id,
        user_id=user.id,
        auth_config=auth,
        resource_owner_key=ACTOR_OWNER,
        **kwargs,
    )


def _token_endpoint(requests: list[dict], scope: str) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == TOKEN_ENDPOINT
        requests.append(parse_qs(request.content.decode()))
        return httpx.Response(
            200,
            json={
                "access_token": "refreshed-access-token",
                "refresh_token": "rotated-refresh-token",
                "token_type": "Bearer",
                "expires_in": 28800,
                "scope": scope,
            },
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# --- the configuration -------------------------------------------------------


@pytest.mark.parametrize(
    "auth",
    [
        {"type": "mcp_oauth", "scope": "records.write records.read"},
        {"type": "mcp_oauth", "scope": ["records.write", "records.read"]},
        {"type": "mcp_oauth", "scope": "records.read", "required_scope": None},
        {"type": "mcp_oauth", "scope": ""},
        {"type": "mcp_oauth"},
    ],
    ids=("string", "list", "explicit-null", "empty-scope", "no-scope"),
)
def test_without_required_scope_the_requested_scope_is_required_as_configured(
    auth,
):
    """Behavior of every configuration without required_scope is unchanged:
    the requirement is the configured scope, returned untouched so each
    caller normalizes it exactly as before."""
    assert mcp_oauth_required_scope(auth) is auth.get("scope")


@pytest.mark.parametrize(
    "required_scope",
    ["records.read  admin records.read", ["admin", "records.read"]],
    ids=("string", "list"),
)
def test_required_scope_is_normalized_like_scope(required_scope):
    auth = {
        "type": "mcp_oauth",
        "scope": "records.write admin records.read",
        "required_scope": required_scope,
    }

    assert mcp_oauth_required_scope(auth) == "admin records.read"


@pytest.mark.parametrize(
    "auth",
    [
        {"scope": "records.read", "required_scope": "records.read records.write"},
        {"scope": "records.read", "required_scope": "admin"},
        {"required_scope": "records.read"},
        {"scope": "", "required_scope": "records.read"},
        {"scope": "records.read", "required_scope": 7},
        {"scope": "records.read", "required_scope": {"records.read": True}},
    ],
    ids=(
        "not-a-subset",
        "unrequested-scope",
        "no-scope",
        "empty-scope",
        "number",
        "object",
    ),
)
def test_required_scope_outside_the_requested_scope_is_a_configuration_error(auth):
    with pytest.raises(MCPOAuthDiscoveryError) as exc_info:
        mcp_oauth_required_scope({"type": "mcp_oauth", **auth})

    assert exc_info.value.code == "invalid_configuration"


def test_atlassian_requests_the_subset_but_requires_only_basic_identity():
    auth = _atlassian_auth()

    assert auth["scope"] == ATLASSIAN_MCP_OAUTH_SCOPE
    assert len(auth["scope"].split()) == 13
    assert auth["required_scope"] == "email offline_access read:account read:me"
    assert auth["required_scope"] == ATLASSIAN_MCP_OAUTH_REQUIRED_SCOPE
    assert mcp_oauth_required_scope(auth) == auth["required_scope"]
    # Granting the whole request less Bitbucket still covers the requirement.
    assert set(BITBUCKET_SCOPES) <= set(auth["scope"].split())
    assert set(BASIC_IDENTITY_SCOPES) <= set(GRANTED_WITHOUT_BITBUCKET.split())


# --- listing a grant as usable -----------------------------------------------


def test_atlassian_grant_without_bitbucket_scopes_is_selected(db_session):
    db, user = db_session
    auth = _atlassian_auth()
    server = _add_server(db, user, auth)
    grant = _add_grant(db, server=server, user=user, scope=GRANTED_WITHOUT_BITBUCKET)

    assert [selected.id for selected in _select(db, server, user, auth)] == [grant.id]
    # The actor's own grant is usable, so the actor (not the workspace owner)
    # is the resource owner its runtime uses.
    assert (
        select_mcp_oauth_owner(
            db,
            server_id=server.id,
            user_id=user.id,
            actor_owner_key=ACTOR_OWNER,
            auth_config=auth,
        )
        == ACTOR_OWNER
    )


@pytest.mark.parametrize("left_out", BASIC_IDENTITY_SCOPES)
def test_atlassian_grant_without_a_basic_identity_scope_is_not_selected(
    db_session, left_out
):
    db, user = db_session
    auth = _atlassian_auth()
    server = _add_server(db, user, auth)
    _add_grant(db, server=server, user=user, scope=_scope_without(left_out))

    assert _select(db, server, user, auth) == []


def test_without_required_scope_a_grant_must_still_cover_the_whole_request(
    db_session,
):
    db, user = db_session
    auth = {"type": "mcp_oauth", "scope": ATLASSIAN_MCP_OAUTH_SCOPE}
    server = _add_server(db, user, auth)
    _add_grant(db, server=server, user=user, scope=GRANTED_WITHOUT_BITBUCKET)

    assert _select(db, server, user, auth) == []

    full_grant = _add_grant(
        db,
        server=server,
        user=user,
        scope=ATLASSIAN_MCP_OAUTH_SCOPE,
        access_token="full-access-token",
    )
    assert [selected.id for selected in _select(db, server, user, auth)] == [
        full_grant.id
    ]


def test_an_invalid_required_scope_makes_no_grant_usable(db_session):
    db, user = db_session
    auth = {**_atlassian_auth(), "required_scope": "read:me manage:jira-configuration"}
    server = _add_server(db, user, auth)
    _add_grant(db, server=server, user=user, scope=ATLASSIAN_MCP_OAUTH_SCOPE)

    assert _select(db, server, user, auth) == []


# --- runtime -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_runtime_uses_an_atlassian_grant_without_bitbucket_scopes(db_session):
    db, user = db_session
    auth = _atlassian_auth()
    server = _add_server(db, user, auth)
    grant = _add_grant(
        db,
        server=server,
        user=user,
        scope=GRANTED_WITHOUT_BITBUCKET,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )

    runtime_auth = await _resolve(db, server, user, auth)

    assert runtime_auth.access_token == "granted-access-token"
    assert runtime_auth.grant_id == grant.id
    assert runtime_auth.scope == GRANTED_WITHOUT_BITBUCKET
    assert runtime_auth.refreshed is False


@pytest.mark.asyncio
async def test_atlassian_grant_without_bitbucket_scopes_stays_usable_after_refresh(
    db_session,
):
    db, user = db_session
    auth = _atlassian_auth()
    server = _add_server(db, user, auth)
    grant = _add_grant(
        db,
        server=server,
        user=user,
        scope=GRANTED_WITHOUT_BITBUCKET,
        access_token="expired-access-token",
        refresh_token="refresh-token",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
    )
    requests: list[dict] = []

    async with _token_endpoint(requests, GRANTED_WITHOUT_BITBUCKET) as client:
        refreshed = await _resolve(db, server, user, auth, client=client)

    assert [request["grant_type"] for request in requests] == [["refresh_token"]]
    assert requests[0]["refresh_token"] == ["refresh-token"]
    assert refreshed.refreshed is True
    assert refreshed.access_token == "refreshed-access-token"
    assert refreshed.scope == GRANTED_WITHOUT_BITBUCKET
    db.refresh(grant)
    assert decrypt_value(grant.access_token) == "refreshed-access-token"
    assert decrypt_value(grant.refresh_token) == "rotated-refresh-token"
    assert grant.scope == GRANTED_WITHOUT_BITBUCKET

    # The refreshed grant is still listed as usable and used without another
    # refresh.
    assert [selected.id for selected in _select(db, server, user, auth)] == [grant.id]
    again = await _resolve(db, server, user, auth)
    assert again.refreshed is False
    assert again.access_token == "refreshed-access-token"


@pytest.mark.asyncio
async def test_refresh_that_drops_a_basic_identity_scope_is_refused(db_session):
    db, user = db_session
    auth = _atlassian_auth()
    server = _add_server(db, user, auth)
    grant = _add_grant(
        db,
        server=server,
        user=user,
        scope=GRANTED_WITHOUT_BITBUCKET,
        access_token="expired-access-token",
        refresh_token="refresh-token",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
    )
    requests: list[dict] = []

    async with _token_endpoint(
        requests, _scope_without(*BITBUCKET_SCOPES, "read:me")
    ) as client:
        with pytest.raises(MCPOAuthRuntimeError) as exc_info:
            await _resolve(db, server, user, auth, client=client)

    assert exc_info.value.code == "insufficient_scope"
    assert len(requests) == 1
    db.refresh(grant)
    assert decrypt_value(grant.access_token) == "expired-access-token"
    assert grant.scope == GRANTED_WITHOUT_BITBUCKET


@pytest.mark.asyncio
@pytest.mark.parametrize("left_out", BASIC_IDENTITY_SCOPES)
async def test_runtime_refuses_a_grant_without_a_basic_identity_scope(
    db_session, left_out
):
    db, user = db_session
    auth = _atlassian_auth()
    server = _add_server(db, user, auth)
    _add_grant(db, server=server, user=user, scope=_scope_without(left_out))

    with pytest.raises(MCPOAuthRuntimeError) as exc_info:
        await _resolve(db, server, user, auth)

    assert exc_info.value.code == "insufficient_scope"


@pytest.mark.asyncio
async def test_runtime_without_required_scope_still_requires_the_whole_request(
    db_session,
):
    db, user = db_session
    auth = {"type": "mcp_oauth", "scope": ATLASSIAN_MCP_OAUTH_SCOPE}
    server = _add_server(db, user, auth)
    _add_grant(db, server=server, user=user, scope=GRANTED_WITHOUT_BITBUCKET)

    with pytest.raises(MCPOAuthRuntimeError) as exc_info:
        await _resolve(db, server, user, auth)

    assert exc_info.value.code == "insufficient_scope"


@pytest.mark.asyncio
async def test_runtime_reports_an_invalid_required_scope_as_a_configuration_error(
    db_session,
):
    db, user = db_session
    auth = {**_atlassian_auth(), "required_scope": "read:me manage:jira-configuration"}
    server = _add_server(db, user, auth)
    _add_grant(db, server=server, user=user, scope=ATLASSIAN_MCP_OAUTH_SCOPE)

    with pytest.raises(MCPOAuthRuntimeError) as exc_info:
        await _resolve(db, server, user, auth)

    assert exc_info.value.code == "invalid_configuration"


@pytest.mark.asyncio
async def test_runtime_connection_carries_the_atlassian_grant(db_session):
    db, user = db_session
    auth = _atlassian_auth()
    server = _add_server(db, user, auth)
    _add_grant(db, server=server, user=user, scope=GRANTED_WITHOUT_BITBUCKET)

    build = await build_mcp_runtime_connection(
        db,
        server,
        user_id=user.id,
        mcp_auth_context={str(server.id): {"resource_owner_key": ACTOR_OWNER}},
    )

    assert build.diagnostic is None
    assert build.connection is not None
    assert build.connection["headers"]["Authorization"] == (
        "Bearer granted-access-token"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("auth", "expected_scope"),
    [
        (_atlassian_auth(), "email offline_access read:account read:me"),
        (
            {"type": "mcp_oauth", "scope": ATLASSIAN_MCP_OAUTH_SCOPE},
            ATLASSIAN_MCP_OAUTH_SCOPE,
        ),
    ],
    ids=("required-scope", "no-required-scope"),
)
async def test_runtime_diagnostic_names_the_scope_a_grant_had_to_cover(
    db_session, auth, expected_scope
):
    db, user = db_session
    server = _add_server(db, user, auth)
    _add_grant(
        db, server=server, user=user, scope=_scope_without(*BITBUCKET_SCOPES, "email")
    )

    build = await build_mcp_runtime_connection(
        db,
        server,
        user_id=user.id,
        mcp_auth_context={str(server.id): {"resource_owner_key": ACTOR_OWNER}},
    )

    assert build.connection is None
    assert build.diagnostic is not None
    assert build.diagnostic["code"] == "insufficient_scope"
    assert build.diagnostic["scope"] == expected_scope


# --- why a required_scope is invalid -----------------------------------------


@pytest.mark.parametrize(
    ("auth", "reason"),
    [
        (
            {"scope": "records.read", "required_scope": 7},
            "must be a scope string or a list of scopes",
        ),
        (
            {"scope": "records.read", "required_scope": True},
            "must be a scope string or a list of scopes",
        ),
        (
            {"scope": "records.read", "required_scope": {"records.read": True}},
            "must be a scope string or a list of scopes",
        ),
        ({"required_scope": "records.read"}, "needs an explicit scope"),
        ({"scope": "", "required_scope": "records.read"}, "needs an explicit scope"),
        ({"scope": [], "required_scope": "records.read"}, "needs an explicit scope"),
        (
            {"scope": "records.read", "required_scope": "records.read records.write"},
            "must only name scopes that scope requests",
        ),
        (
            {"scope": "records.read", "required_scope": "admin"},
            "must only name scopes that scope requests",
        ),
    ],
    ids=(
        "number",
        "boolean",
        "object",
        "no-scope",
        "empty-scope",
        "empty-scope-list",
        "not-a-subset",
        "unrequested-scope",
    ),
)
def test_an_invalid_required_scope_reports_why(auth, reason):
    """Each check reports its own reason, so an administrator can tell which
    part of the configuration to fix."""
    with pytest.raises(MCPOAuthDiscoveryError) as exc_info:
        mcp_oauth_required_scope({"type": "mcp_oauth", **auth})

    assert exc_info.value.code == "invalid_configuration"
    assert exc_info.value.message == f"MCP OAuth required_scope {reason}"


# --- a scope the caller names ------------------------------------------------


def _unexpired() -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=1)


def test_selection_holds_an_explicit_scope_instead_of_required_scope(db_session):
    db, user = db_session
    auth = _atlassian_auth()
    server = _add_server(db, user, auth)
    grant = _add_grant(db, server=server, user=user, scope=_scope_without("email"))

    def select(scope: str) -> list:
        return select_mcp_oauth_grants(
            db,
            server_id=server.id,
            user_id=user.id,
            auth_config=auth,
            resource_owner_key=ACTOR_OWNER,
            scope=scope,
        )

    # The grant lacks a basic identity scope, so required_scope rejects it ...
    assert _select(db, server, user, auth) == []
    # ... but a caller naming a scope is held to that scope only, both when it
    # asks for less than required_scope and when it asks for more.
    assert [selected.id for selected in select("read:me")] == [grant.id]
    assert select(ATLASSIAN_MCP_OAUTH_SCOPE) == []


@pytest.mark.asyncio
async def test_runtime_holds_an_explicit_scope_instead_of_required_scope(db_session):
    """A caller that names the scope it needs is held to that scope, not to
    the configured required_scope, as before required_scope existed."""
    db, user = db_session
    auth = _atlassian_auth()
    server = _add_server(db, user, auth)
    grant = _add_grant(
        db,
        server=server,
        user=user,
        scope=GRANTED_WITHOUT_BITBUCKET,
        expires_at=_unexpired(),
    )

    # The grant covers required_scope ...
    assert (await _resolve(db, server, user, auth)).grant_id == grant.id
    # ... but not a caller's explicit request for the whole subset.
    with pytest.raises(MCPOAuthRuntimeError) as exc_info:
        await _resolve(db, server, user, auth, scope=ATLASSIAN_MCP_OAUTH_SCOPE)
    assert exc_info.value.code == "insufficient_scope"


@pytest.mark.asyncio
async def test_runtime_accepts_an_explicit_scope_narrower_than_required_scope(
    db_session,
):
    db, user = db_session
    auth = _atlassian_auth()
    server = _add_server(db, user, auth)
    grant = _add_grant(
        db,
        server=server,
        user=user,
        scope=_scope_without("email"),
        expires_at=_unexpired(),
    )

    with pytest.raises(MCPOAuthRuntimeError) as exc_info:
        await _resolve(db, server, user, auth)
    assert exc_info.value.code == "insufficient_scope"

    runtime_auth = await _resolve(db, server, user, auth, scope="read:me")
    assert runtime_auth.grant_id == grant.id
    assert runtime_auth.access_token == "granted-access-token"


@pytest.mark.asyncio
async def test_runtime_connection_holds_a_selection_to_its_own_scope(db_session):
    db, user = db_session
    auth = _atlassian_auth()
    server = _add_server(db, user, auth)
    _add_grant(db, server=server, user=user, scope=GRANTED_WITHOUT_BITBUCKET)

    build = await build_mcp_runtime_connection(
        db,
        server,
        user_id=user.id,
        mcp_auth_context={
            str(server.id): {
                "resource_owner_key": ACTOR_OWNER,
                "scope": ATLASSIAN_MCP_OAUTH_SCOPE,
            }
        },
    )

    assert build.connection is None
    assert build.diagnostic is not None
    assert build.diagnostic["code"] == "insufficient_scope"
    assert build.diagnostic["scope"] == ATLASSIAN_MCP_OAUTH_SCOPE


@pytest.mark.asyncio
async def test_runtime_diagnostic_names_the_requested_scope_when_required_scope_is_invalid(
    db_session,
):
    db, user = db_session
    auth = {**_atlassian_auth(), "required_scope": "read:me manage:jira-configuration"}
    server = _add_server(db, user, auth)
    _add_grant(db, server=server, user=user, scope=ATLASSIAN_MCP_OAUTH_SCOPE)

    build = await build_mcp_runtime_connection(
        db,
        server,
        user_id=user.id,
        mcp_auth_context={str(server.id): {"resource_owner_key": ACTOR_OWNER}},
    )

    assert build.connection is None
    assert build.diagnostic is not None
    assert build.diagnostic["code"] == "invalid_configuration"
    assert build.diagnostic["message"] == (
        "MCP OAuth required_scope must only name scopes that scope requests"
    )
    # The required scope cannot be worked out, so the diagnostic names the
    # scope the configuration requests.
    assert build.diagnostic["scope"] == ATLASSIAN_MCP_OAUTH_SCOPE


# --- blank entries in required_scope -----------------------------------------

# Whitespace and empty entries name no scope, in a list as in a string. Next to
# real scopes they are dropped; a required_scope made only of them is unset
# (see below).


@pytest.mark.parametrize(
    ("required_scope", "expected"),
    [
        (["read:me", " "], "read:me"),
        (["", "read:me"], "read:me"),
        ([" read:me ", "\temail"], "email read:me"),
        ((" read:me", "email "), "email read:me"),
        (" read:me \t email ", "email read:me"),
    ],
    ids=(
        "blank-entry-after",
        "empty-entry-before",
        "padded-entries",
        "padded-tuple-entries",
        "padded-string",
    ),
)
def test_blank_entries_next_to_a_scope_are_ignored(required_scope, expected):
    """A blank entry beside real scopes is dropped and the real scopes stay
    required, as whitespace around the scopes of a string always was."""
    auth = {**_atlassian_auth(), "required_scope": required_scope}

    assert mcp_oauth_required_scope(auth) == expected


def test_a_blank_entry_next_to_a_scope_still_requires_that_scope(db_session):
    db, user = db_session
    auth = {**_atlassian_auth(), "required_scope": ["read:me", " "]}
    server = _add_server(db, user, auth)
    _add_grant(db, server=server, user=user, scope="email read:account")

    assert _select(db, server, user, auth) == []

    grant = _add_grant(
        db,
        server=server,
        user=user,
        scope="read:me",
        access_token="read-me-access-token",
    )
    assert [selected.id for selected in _select(db, server, user, auth)] == [grant.id]


@pytest.mark.parametrize(
    ("required_scope", "expected"),
    [
        ("read:me", "read:me"),
        ("read:me email", "email read:me"),
        (["read:me"], "read:me"),
        (["read:me", "email"], "email read:me"),
        (["read:me", "read:me"], "read:me"),
        (("read:me", "email"), "email read:me"),
        ({"read:me", "email"}, "email read:me"),
        (ATLASSIAN_MCP_OAUTH_REQUIRED_SCOPE, ATLASSIAN_MCP_OAUTH_REQUIRED_SCOPE),
    ],
    ids=(
        "one-scope",
        "string",
        "one-entry",
        "list",
        "repeated-entry",
        "tuple",
        "set",
        "atlassian",
    ),
)
def test_a_valid_required_scope_is_returned_as_before(required_scope, expected):
    """Ignoring blank entries changes nothing for a required_scope without
    them: the same sorted, space-separated scopes come back."""
    auth = {**_atlassian_auth(), "required_scope": required_scope}

    assert mcp_oauth_required_scope(auth) == expected


# --- a blank required_scope is unset -----------------------------------------

# A required_scope that names no scope is unset, as a blank scope is: the
# whole requested scope stays required, exactly as without the key. This is
# stricter than any required_scope that names a scope, never looser.
UNSET_REQUIRED_SCOPES = (
    pytest.param(None, id="null"),
    pytest.param("", id="empty-string"),
    pytest.param("  ", id="blank-string"),
    pytest.param([], id="empty-list"),
    pytest.param([" "], id="blank-entry"),
    pytest.param(["", " "], id="empty-and-blank-entries"),
    pytest.param(["\t", " \n "], id="whitespace-entries"),
    pytest.param((" ",), id="blank-tuple-entry"),
)


@pytest.mark.parametrize("required_scope", UNSET_REQUIRED_SCOPES)
def test_a_blank_required_scope_is_unset(required_scope):
    auth = {**_atlassian_auth(), "required_scope": required_scope}
    without_required_scope = {"type": "mcp_oauth", "scope": auth["scope"]}

    assert mcp_oauth_required_scope_is_set(auth) is False
    # The requested scope comes back untouched, as without required_scope.
    assert mcp_oauth_required_scope(auth) is auth["scope"]
    assert mcp_oauth_required_scope(auth) == mcp_oauth_required_scope(
        without_required_scope
    )


@pytest.mark.parametrize("required_scope", UNSET_REQUIRED_SCOPES)
def test_a_blank_required_scope_needs_no_explicit_scope(required_scope):
    """Unset means no new requirement at all, so the "needs an explicit
    scope" rule for a set required_scope does not apply either."""
    for auth in (
        {"type": "mcp_oauth", "required_scope": required_scope},
        {"type": "mcp_oauth", "scope": "", "required_scope": required_scope},
    ):
        assert mcp_oauth_required_scope(auth) == auth.get("scope")


@pytest.mark.parametrize(
    "required_scope",
    ["read:me", ["read:me", " "], 7, {"read:me": True}, False],
    ids=("string", "list-with-a-blank-entry", "number", "object", "boolean"),
)
def test_a_required_scope_that_is_not_blank_is_set(required_scope):
    """Only a value that names no scope is unset. A value of another type
    stays set, so it is still reported as an invalid configuration."""
    auth = {**_atlassian_auth(), "required_scope": required_scope}

    assert mcp_oauth_required_scope_is_set(auth) is True


@pytest.mark.parametrize("required_scope", UNSET_REQUIRED_SCOPES)
def test_with_a_blank_required_scope_a_grant_must_cover_the_whole_scope(
    db_session, required_scope
):
    db, user = db_session
    auth = {**_atlassian_auth(), "required_scope": required_scope}
    server = _add_server(db, user, auth)
    _add_grant(db, server=server, user=user, scope=GRANTED_WITHOUT_BITBUCKET)

    # Without Bitbucket the grant misses part of the request, so it is not
    # usable; with the whole request it is.
    assert _select(db, server, user, auth) == []
    whole = _add_grant(
        db,
        server=server,
        user=user,
        scope=ATLASSIAN_MCP_OAUTH_SCOPE,
        access_token="whole-scope-access-token",
    )
    assert [selected.id for selected in _select(db, server, user, auth)] == [whole.id]


@pytest.mark.asyncio
@pytest.mark.parametrize("required_scope", UNSET_REQUIRED_SCOPES)
async def test_runtime_holds_a_blank_required_scope_to_the_whole_scope(
    db_session, required_scope
):
    db, user = db_session
    auth = {**_atlassian_auth(), "required_scope": required_scope}
    server = _add_server(db, user, auth)
    _add_grant(
        db,
        server=server,
        user=user,
        scope=GRANTED_WITHOUT_BITBUCKET,
        expires_at=_unexpired(),
    )

    with pytest.raises(MCPOAuthRuntimeError) as exc_info:
        await _resolve(db, server, user, auth)

    assert exc_info.value.code == "insufficient_scope"
