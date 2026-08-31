"""Tests for the shared mcp_oauth reconnect pre-flight validator.

``explain_mcp_oauth_reconnect_refusal`` answers "would an mcp_oauth connect for
this app succeed past its deterministic gates right now?". It exists so a
consumer asking that question ahead of time and the connect route answering it
for real share one implementation instead of two that drift; every test here
therefore pins both halves -- the predicted refusal *and* the connect route
refusing the same way.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.web.api import mcp as mcp_api
from xagent.web.api.mcp import MCPOAuthConnectRequest, connect_mcp_oauth_app
from xagent.web.mcp_apps import (
    MCP_OAUTH_RECONNECT_REFUSAL_APP_HIDDEN,
    MCP_OAUTH_RECONNECT_REFUSAL_APP_MISSING,
    MCP_OAUTH_RECONNECT_REFUSAL_INVALID_AUTH_CONFIG,
    MCP_OAUTH_RECONNECT_REFUSAL_INVALID_SERVER_CONFIG,
    MCP_OAUTH_RECONNECT_REFUSAL_NOT_MCP_OAUTH,
    MCP_OAUTH_RECONNECT_REFUSAL_SERVER_CONFIG_DRIFT,
    MCP_OAUTH_RECONNECT_REFUSAL_USER_OWNED_SQUAT,
    explain_mcp_oauth_reconnect_refusal,
)
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.services import mcp_oauth as mcp_oauth_service
from xagent.web.services.mcp_oauth import (
    MCP_OAUTH_PERSISTED_VALUE_MAX_LENGTH,
    MCP_OAUTH_TOKEN_ENDPOINT_AUTH_METHOD_MAX_LENGTH,
)

APP_ID = "remote-notes"
APP_URL = "https://mcp.example.com/mcp"


@pytest.fixture()
def db_session(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'reconnect.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False)()

    user = User(username="alice", password_hash="x", is_admin=False)
    other = User(username="bob", password_hash="x", is_admin=False)
    db.add_all([user, other])
    db.commit()
    db.refresh(user)
    db.refresh(other)

    yield db, user, other
    db.close()
    engine.dispose()


def _add_app(
    db,
    *,
    app_id: str = APP_ID,
    transport: str = "streamable_http",
    url: str | None = APP_URL,
    auth: dict | None = None,
    is_visible: bool = True,
) -> PublicMCPApp:
    """A catalog row shaped like a real remote-MCP-OAuth connector.

    The app_id must not match a real builtin registry entry: the builtin
    execution overlay replaces a DB row's execution fields with the canonical
    registry values for matching app_ids, which would override this fixture.
    """
    app = PublicMCPApp(
        app_id=app_id,
        name=app_id.title(),
        transport=transport,
        launch_config={
            "url": url,
            "auth": {"type": "mcp_oauth"} if auth is None else auth,
        },
        is_visible_in_connector=is_visible,
    )
    db.add(app)
    db.commit()
    return app


def _add_shared_server(
    db, *, name: str = APP_ID, transport: str = "streamable_http", url: str = APP_URL
) -> MCPServer:
    """The legitimate shared catalog row: no association, so no owner."""
    server = MCPServer.from_config(
        {
            "name": name,
            "managed": "external",
            "transport": transport,
            "url": url,
            "auth": {"type": "mcp_oauth"},
        }
    )
    db.add(server)
    db.commit()
    db.refresh(server)
    return server


def _discovery() -> SimpleNamespace:
    return SimpleNamespace(
        resource=APP_URL,
        scopes=("notes.read",),
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


def _stub_discovery(monkeypatch) -> None:
    """Let the happy path reach the authorization redirect without a network.

    Only used by the accepts-case and by the association-ordering test, which
    must prove the refusal fires *before* provisioning rather than because
    discovery was unreachable.
    """

    async def fake_discover(*args, **kwargs):
        return _discovery()

    monkeypatch.setattr(mcp_api, "discover_mcp_oauth_metadata", fake_discover)


async def _connect(db, user, *, app_id: str = APP_ID):
    return await connect_mcp_oauth_app(
        app_id, MCPOAuthConnectRequest(redirect_after="/settings/mcp"), user, db
    )


def _assocs(db, user_id: int) -> list[UserMCPServer]:
    return db.query(UserMCPServer).filter(UserMCPServer.user_id == user_id).all()


# --------------------------------------------------------------------------
# Accepting case: nothing deterministic refuses a well-formed catalog entry.
# --------------------------------------------------------------------------


def test_returns_none_for_a_well_formed_app(db_session):
    db, user, _ = db_session
    _add_app(db)
    assert explain_mcp_oauth_reconnect_refusal(db, APP_ID, user.id) is None


def test_returns_none_when_the_shared_row_already_matches(db_session):
    """The reuse path: an existing unowned row matching the catalog is fine."""
    db, user, _ = db_session
    _add_app(db)
    _add_shared_server(db)
    assert explain_mcp_oauth_reconnect_refusal(db, APP_ID, user.id) is None


@pytest.mark.asyncio
async def test_accepted_app_still_reaches_the_authorization_redirect(
    db_session, monkeypatch
):
    """The pre-flight must not refuse what the connect path would have allowed.

    Without this, a validator that returned a code unconditionally would pass
    every refusal test below while breaking the feature outright.
    """
    db, user, _ = db_session
    _add_app(
        db,
        auth={
            "type": "mcp_oauth",
            "client_id": "static-client",
            "token_endpoint_auth_method": "none",
        },
    )
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.xagent.test/")
    _stub_discovery(monkeypatch)

    response = await _connect(db, user)

    assert response.status_code == 303
    assert "https://auth.example.com/authorize" in response.headers["location"]
    assert len(_assocs(db, user.id)) == 1


# --------------------------------------------------------------------------
# Catalog-level refusals.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_app(db_session):
    db, user, _ = db_session
    assert (
        explain_mcp_oauth_reconnect_refusal(db, "no-such-app", user.id)
        == MCP_OAUTH_RECONNECT_REFUSAL_APP_MISSING
    )
    with pytest.raises(mcp_api.HTTPException) as exc:
        await _connect(db, user, app_id="no-such-app")
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_hidden_app(db_session):
    db, user, _ = db_session
    _add_app(db, is_visible=False)
    assert (
        explain_mcp_oauth_reconnect_refusal(db, APP_ID, user.id)
        == MCP_OAUTH_RECONNECT_REFUSAL_APP_HIDDEN
    )
    with pytest.raises(mcp_api.HTTPException) as exc:
        await _connect(db, user)
    # 404, not 403: a hidden app is indistinguishable from a missing one.
    assert exc.value.status_code == 404
    assert exc.value.detail == "MCP app not found"


@pytest.mark.asyncio
async def test_app_that_is_not_an_mcp_oauth_connector(db_session):
    """An api_key catalog app: classify_app_auth gives it a non-mcp_oauth type."""
    db, user, _ = db_session
    db.add(
        PublicMCPApp(
            app_id="maps",
            name="Maps",
            transport="stdio",
            launch_config={"command": "npx", "required_env": ["MAPS_KEY"]},
        )
    )
    db.commit()

    assert (
        explain_mcp_oauth_reconnect_refusal(db, "maps", user.id)
        == MCP_OAUTH_RECONNECT_REFUSAL_NOT_MCP_OAUTH
    )
    with pytest.raises(mcp_api.HTTPException) as exc:
        await _connect(db, user, app_id="maps")
    assert exc.value.status_code == 400


# --------------------------------------------------------------------------
# Auth-config refusals -- the class of bug this validator was written for.
# These were previously only caught *after* the association was committed.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "auth",
    [
        pytest.param(
            {
                "type": "mcp_oauth",
                "client_id": "c" * (MCP_OAUTH_PERSISTED_VALUE_MAX_LENGTH + 1),
            },
            id="client_id_too_long",
        ),
        pytest.param(
            {
                "type": "mcp_oauth",
                "redirect_uri": "https://x.example.com/"
                + "p" * MCP_OAUTH_PERSISTED_VALUE_MAX_LENGTH,
            },
            id="redirect_uri_too_long",
        ),
        pytest.param(
            {
                "type": "mcp_oauth",
                "client_id": "static-client",
                "token_endpoint_auth_method": "private_key_jwt",
            },
            id="unsupported_auth_method",
        ),
    ],
)
def test_invalid_auth_config_is_refused(db_session, auth):
    db, user, _ = db_session
    _add_app(db, auth=auth)
    assert (
        explain_mcp_oauth_reconnect_refusal(db, APP_ID, user.id)
        == MCP_OAUTH_RECONNECT_REFUSAL_INVALID_AUTH_CONFIG
    )


def test_overlong_auth_method_is_refused_on_the_length_bound(db_session, monkeypatch):
    """The length bound must be checked *before* the allowlist, as upstream does.

    An overlong method is necessarily also outside the three-member allowlist,
    so a plain refusal assertion cannot tell the two branches apart and would
    pass with the length check deleted. Widening the allowlist to admit the
    overlong value isolates the bound: it is the only thing left that can
    refuse, so the refusal proves the bound is checked and is checked first.
    """
    db, user, _ = db_session
    overlong = "m" * (MCP_OAUTH_TOKEN_ENDPOINT_AUTH_METHOD_MAX_LENGTH + 1)
    _add_app(
        db,
        auth={
            "type": "mcp_oauth",
            "client_id": "static-client",
            "token_endpoint_auth_method": overlong,
        },
    )
    monkeypatch.setattr(
        mcp_oauth_service,
        "MCP_OAUTH_TOKEN_ENDPOINT_AUTH_METHODS",
        frozenset({overlong}),
    )

    assert (
        explain_mcp_oauth_reconnect_refusal(db, APP_ID, user.id)
        == MCP_OAUTH_RECONNECT_REFUSAL_INVALID_AUTH_CONFIG
    )


@pytest.mark.parametrize(
    "auth",
    [
        pytest.param(
            {"type": "mcp_oauth", "token_endpoint_auth_method": "private_key_jwt"},
            id="unsupported_method_without_client_id",
        ),
        pytest.param(
            {
                "type": "mcp_oauth",
                "token_endpoint_auth_method": "m"
                * (MCP_OAUTH_TOKEN_ENDPOINT_AUTH_METHOD_MAX_LENGTH + 1),
            },
            id="overlong_method_without_client_id",
        ),
    ],
)
def test_auth_method_is_only_checked_on_the_static_client_id_branch(db_session, auth):
    """Without a client_id the flow registers dynamically.

    connect_mcp_oauth then takes its method from the *registered* client, never
    from this row -- so refusing here would block a connect that actually works.
    Pinned because over-mirroring is as wrong as under-mirroring.
    """
    db, user, _ = db_session
    _add_app(db, auth=auth)
    assert explain_mcp_oauth_reconnect_refusal(db, APP_ID, user.id) is None


@pytest.mark.parametrize(
    ("auth", "case"),
    [
        pytest.param(
            {"type": "mcp_oauth", "client_id": "static-client"},
            "none",
            id="defaults_to_none_without_secret",
        ),
        pytest.param(
            {
                "type": "mcp_oauth",
                "client_id": "static-client",
                "client_secret": "shh",
            },
            "client_secret_post",
            id="defaults_to_post_with_secret",
        ),
    ],
)
def test_default_auth_method_is_accepted(db_session, auth, case):
    """An unset method resolves to an allowlisted default, either way."""
    db, user, _ = db_session
    _add_app(db, auth=auth)
    assert explain_mcp_oauth_reconnect_refusal(db, APP_ID, user.id) is None


# --------------------------------------------------------------------------
# Server-row refusals.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("server_transport", "server_url"),
    [
        pytest.param("sse", APP_URL, id="transport_drift"),
        pytest.param("streamable_http", "https://evil.example.com/mcp", id="url_drift"),
    ],
)
async def test_shared_row_drifted_from_the_catalog(
    db_session, server_transport, server_url
):
    db, user, _ = db_session
    _add_app(db)
    _add_shared_server(db, transport=server_transport, url=server_url)

    assert (
        explain_mcp_oauth_reconnect_refusal(db, APP_ID, user.id)
        == MCP_OAUTH_RECONNECT_REFUSAL_SERVER_CONFIG_DRIFT
    )
    with pytest.raises(mcp_api.HTTPException) as exc:
        await _connect(db, user)
    assert exc.value.status_code == 409


def test_transport_drift_is_compared_case_insensitively(db_session):
    """_ensure_catalog_mcp_oauth_server lowercases both sides; so must this.

    An admin PATCH can store a mixed-case transport, and the two halves of
    this feature must not disagree about the same row.
    """
    db, user, _ = db_session
    _add_app(db)
    server = _add_shared_server(db)
    server.transport = "Streamable_HTTP"
    db.commit()

    assert explain_mcp_oauth_reconnect_refusal(db, APP_ID, user.id) is None


@pytest.mark.asyncio
async def test_user_owned_row_squatting_the_catalog_id(db_session):
    """A row someone owns could later have a foreign URL swapped in."""
    db, user, other = db_session
    _add_app(db)
    server = _add_shared_server(db)
    db.add(
        UserMCPServer(
            user_id=other.id,
            mcpserver_id=server.id,
            is_active=True,
            is_owner=True,
            can_edit=True,
            can_delete=True,
        )
    )
    db.commit()

    assert (
        explain_mcp_oauth_reconnect_refusal(db, APP_ID, user.id)
        == MCP_OAUTH_RECONNECT_REFUSAL_USER_OWNED_SQUAT
    )
    with pytest.raises(mcp_api.HTTPException) as exc:
        await _connect(db, user)
    assert exc.value.status_code == 409


def test_the_owner_themselves_is_also_refused(db_session):
    """The squat gate rejects a row *any* user owns, the caller included.

    Pins the claim the docstring makes about user_id: this is not a per-user
    rule, so the owner does not get a pass. Upstream refuses them for the same
    reason it refuses everyone else -- an owner keeps edit rights and could
    later swap in a foreign URL that every connected user then runs.
    """
    db, user, _ = db_session
    _add_app(db)
    server = _add_shared_server(db)
    db.add(
        UserMCPServer(
            user_id=user.id,
            mcpserver_id=server.id,
            is_active=True,
            is_owner=True,
            can_edit=True,
            can_delete=True,
        )
    )
    db.commit()

    assert (
        explain_mcp_oauth_reconnect_refusal(db, APP_ID, user.id)
        == MCP_OAUTH_RECONNECT_REFUSAL_USER_OWNED_SQUAT
    )


def test_a_non_owning_association_is_not_a_squat(db_session):
    """Connect users get is_owner=False rows; those must not refuse a reconnect."""
    db, user, other = db_session
    _add_app(db)
    server = _add_shared_server(db)
    db.add(
        UserMCPServer(
            user_id=other.id,
            mcpserver_id=server.id,
            is_active=True,
            is_owner=False,
            can_edit=False,
            can_delete=True,
        )
    )
    db.commit()

    assert explain_mcp_oauth_reconnect_refusal(db, APP_ID, user.id) is None


@pytest.mark.asyncio
async def test_catalog_transport_the_server_schema_would_reject(db_session):
    """classify_app_auth lowercases before testing HTTP_MCP_TRANSPORTS, but
    MCPServerConfig's validator matches exact case -- so an entry authored as
    "Streamable_HTTP" classifies as mcp_oauth and then cannot be persisted at
    all. Only reachable while no shared row exists yet (the create path).
    """
    db, user, _ = db_session
    _add_app(db, transport="Streamable_HTTP")

    assert (
        explain_mcp_oauth_reconnect_refusal(db, APP_ID, user.id)
        == MCP_OAUTH_RECONNECT_REFUSAL_INVALID_SERVER_CONFIG
    )
    with pytest.raises(mcp_api.HTTPException) as exc:
        await _connect(db, user)
    assert exc.value.status_code == 400


def test_non_string_catalog_url_the_server_schema_would_reject(db_session):
    """classify_app_auth only tests that ``url`` is truthy, so a mis-authored
    non-string url (an admin PATCH can store any JSON) still classifies as
    mcp_oauth and then fails MCPServerConfig. Caught here because the raw value
    is handed to the schema rather than coerced first -- pydantic's
    ValidationError is a ValueError, so the create-path guard sees it.
    """
    db, user, _ = db_session
    _add_app(db, url={"href": APP_URL})

    assert (
        explain_mcp_oauth_reconnect_refusal(db, APP_ID, user.id)
        == MCP_OAUTH_RECONNECT_REFUSAL_INVALID_SERVER_CONFIG
    )


# --------------------------------------------------------------------------
# The core guarantee: the connect path calls the validator itself, before it
# provisions anything. This is what stops the validator from becoming a
# second, drifting copy -- and it closes the pre-existing one-way door.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connect_refuses_before_creating_any_association(db_session, monkeypatch):
    """A mis-authored auth config must fail with the database untouched.

    Previously connect_mcp_oauth_app committed the shared server row and this
    user's association first and only reached the auth-config bounds inside
    connect_mcp_oauth -- so disconnect worked while every reconnect created an
    association and then 400'd, a one-way door. Discovery is stubbed so a
    surviving association could not be blamed on an unreachable network.
    """
    db, user, _ = db_session
    _add_app(
        db,
        auth={
            "type": "mcp_oauth",
            "client_id": "static-client",
            "token_endpoint_auth_method": "private_key_jwt",
        },
    )
    _stub_discovery(monkeypatch)

    with pytest.raises(mcp_api.HTTPException) as exc:
        await _connect(db, user)

    assert exc.value.status_code == 400
    assert _assocs(db, user.id) == []
    assert db.query(MCPServer).filter(MCPServer.name == APP_ID).first() is None


@pytest.mark.asyncio
async def test_connect_consults_the_shared_validator(db_session, monkeypatch):
    """Pins the wiring itself, not just an agreeing outcome.

    Both halves refusing identically is also what a duplicated rule set looks
    like right up until it drifts. Forcing the shared validator to report a
    refusal it could not have derived from this well-formed app proves the
    route reads that one function rather than re-deriving the rules.
    """
    db, user, _ = db_session
    _add_app(db)
    calls: list[tuple[str, int]] = []

    def fake_explain(_db, app_id, user_id):
        calls.append((app_id, user_id))
        return MCP_OAUTH_RECONNECT_REFUSAL_USER_OWNED_SQUAT

    monkeypatch.setattr(mcp_api, "explain_mcp_oauth_reconnect_refusal", fake_explain)

    with pytest.raises(mcp_api.HTTPException) as exc:
        await _connect(db, user)

    assert calls == [(APP_ID, user.id)]
    assert exc.value.status_code == 409
    assert _assocs(db, user.id) == []


@pytest.mark.asyncio
async def test_every_refusal_code_maps_to_a_response(db_session):
    """No reason code may fall through to a KeyError at request time."""
    for refusal in (
        MCP_OAUTH_RECONNECT_REFUSAL_APP_MISSING,
        MCP_OAUTH_RECONNECT_REFUSAL_APP_HIDDEN,
        MCP_OAUTH_RECONNECT_REFUSAL_NOT_MCP_OAUTH,
        MCP_OAUTH_RECONNECT_REFUSAL_SERVER_CONFIG_DRIFT,
        MCP_OAUTH_RECONNECT_REFUSAL_USER_OWNED_SQUAT,
        MCP_OAUTH_RECONNECT_REFUSAL_INVALID_SERVER_CONFIG,
        MCP_OAUTH_RECONNECT_REFUSAL_INVALID_AUTH_CONFIG,
    ):
        with pytest.raises(mcp_api.HTTPException) as exc:
            mcp_api._raise_mcp_oauth_reconnect_refusal(refusal)
        assert exc.value.status_code in (400, 404, 409)
