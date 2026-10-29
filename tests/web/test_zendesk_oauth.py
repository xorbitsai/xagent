"""Zendesk per-account OAuth (#2930): the registered account host, the
callback identity it nests, the token lifetimes requested on every token
request, and how a dead or superseded refresh token is classified.
"""

import asyncio
import socket
import sqlite3
from datetime import datetime, timedelta, timezone
from http.cookies import SimpleCookie
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.parse import parse_qs, urlparse

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from tests.shared.oauth_browser_fixture import create_access_token
from xagent.core.utils.encryption import encrypt_value
from xagent.web import mcp_apps, oauth_account_hosts
from xagent.web.api import auth as auth_api
from xagent.web.api.auth import generic_oauth_login
from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.oauth_provider import OAuthProvider
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.oauth_account_hosts import (
    get_oauth_account_host,
    resolve_oauth_account_endpoints,
)
from xagent.web.tools import config as tool_config
from xagent.web.tools.config import WebToolConfig

TOKEN_LIFETIMES = {"expires_in": "172800", "refresh_token_expires_in": "7776000"}
PUBLIC_ADDRESS = "93.184.216.34"


def _resolve_to(address: str):
    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]

    return fake_getaddrinfo


class _ProviderResponse:
    def __init__(self, data: dict, status_code: int = 200) -> None:
        self._data = data
        self.status_code = status_code
        self.text = ""

    def json(self) -> dict:
        return self._data


def _token_for(user: User) -> str:
    return create_access_token(
        data={
            "sub": user.username,
            "user_id": user.id,
            "session_epoch": user.session_epoch,
            "type": "access",
        },
        expires_delta=timedelta(minutes=5),
    )


@pytest.fixture(autouse=True)
def _public_dns(monkeypatch):
    monkeypatch.setattr(
        oauth_account_hosts.socket, "getaddrinfo", _resolve_to(PUBLIC_ADDRESS)
    )


MARKETPLACE_ENV = {
    "ZENDESK_MARKETPLACE_NAME": "Xagent",
    "ZENDESK_MARKETPLACE_ORGANIZATION_ID": "12345",
    "ZENDESK_MARKETPLACE_APP_ID": "67890",
}
MARKETPLACE_HEADERS = {
    "X-Zendesk-Marketplace-Name": "Xagent",
    "X-Zendesk-Marketplace-Organization-Id": "12345",
    "X-Zendesk-Marketplace-App-Id": "67890",
}


@pytest.fixture(autouse=True)
def _marketplace_unset(monkeypatch):
    for name in MARKETPLACE_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def marketplace_configured(monkeypatch):
    for name, value in MARKETPLACE_ENV.items():
        monkeypatch.setenv(name, value)


@pytest.fixture()
def session_factory(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'zendesk-oauth.db'}")
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine, autoflush=False, autocommit=False)
    engine.dispose()


@pytest.fixture()
def zendesk_db(session_factory):
    row = get_builtin_public_mcp_app("zendesk")
    assert row is not None
    with session_factory() as db:
        user = User(username="alice", password_hash="hash")
        db.add(user)
        db.add(
            PublicMCPApp(
                app_id=row["app_id"],
                name=row["name"],
                description=row["description"],
                transport=row["transport"],
                provider_name=row["provider_name"],
                oauth_scopes=row["oauth_scopes"],
                launch_config=row["launch_config"],
                # Shown so the login's hidden-app gate lets the flow start.
                is_visible_in_connector=True,
            )
        )
        db.add(
            OAuthProvider(
                provider_name="zendesk",
                name="Zendesk",
                client_id=encrypt_value("client-id"),
                client_secret=encrypt_value("client-secret"),
                auth_url="",
                token_url="",
                redirect_uri="https://xagent.example/api/auth/zendesk/callback",
                userinfo_url="",
                user_id_path="id",
                email_path="email",
                default_scopes=["read"],
            )
        )
        db.commit()
        db.refresh(user)
        yield db, user


def _db_provider(db):
    return db.query(OAuthProvider).filter_by(provider_name="zendesk").one()


# ---------- registry and catalog ---------------------------------------------


def test_zendesk_endpoints_live_on_the_account_subdomain():
    host = get_oauth_account_host("zendesk")
    assert host is not None

    endpoints = resolve_oauth_account_endpoints(host, "Acme")

    assert endpoints.account == "acme"
    assert endpoints.authorize_url == (
        "https://acme.zendesk.com/oauth/authorizations/new"
    )
    assert endpoints.token_url == "https://acme.zendesk.com/oauth/tokens"
    assert endpoints.userinfo_url == "https://acme.zendesk.com/api/v2/users/me.json"


def test_zendesk_requests_the_maximum_token_lifetimes():
    host = get_oauth_account_host("zendesk")

    assert host is not None
    assert dict(host.token_params) == TOKEN_LIFETIMES


def test_zendesk_catalog_app_asks_for_the_subdomain(zendesk_db):
    db, _user = zendesk_db

    app = mcp_apps.get_app_by_id(db, "zendesk")

    assert app is not None
    assert app["oauth_account_input"]["suffix"] == ".zendesk.com"
    assert app["oauth_account_input"]["label"] == "Zendesk subdomain"


# ---------- login and callback -----------------------------------------------


def _login(db, user, account: str):
    return generic_oauth_login(
        "zendesk",
        token=_token_for(user),
        app_id="zendesk",
        redirect=None,
        db=db,
        db_provider=_db_provider(db),
        account=account,
    )


def _login_state(db, user, account: str = "acme") -> tuple[str, dict[str, str]]:
    response = _login(db, user, account)
    location = response.headers["location"]
    cookie = SimpleCookie()
    cookie.load(response.headers["set-cookie"])
    return (
        parse_qs(urlparse(location).query)["state"][0],
        {name: morsel.value for name, morsel in cookie.items()},
    )


def _mock_provider_http(monkeypatch, userinfo: dict):
    post = Mock(
        return_value=_ProviderResponse(
            {
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "expires_in": 172800,
                "refresh_token_expires_in": 7776000,
                "token_type": "bearer",
                "scope": "read tickets:write",
            }
        )
    )
    get = Mock(return_value=_ProviderResponse(userinfo))
    monkeypatch.setattr(auth_api.requests, "post", post)
    monkeypatch.setattr(auth_api.requests, "get", get)
    return post, get


def _callback(db, state: str, cookies: dict[str, str] | None = None):
    request = SimpleNamespace(
        query_params={"state": state, "code": "code"},
        cookies=cookies or {},
    )
    return auth_api.generic_oauth_callback(
        "zendesk", request, db=db, db_provider=_db_provider(db)
    )


def _grant(db, user) -> UserOAuth | None:
    return (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "zendesk")
        .one_or_none()
    )


def test_login_sends_the_user_to_the_accounts_consent_page(zendesk_db):
    db, user = zendesk_db

    response = _login(db, user, "Acme")

    location = urlparse(response.headers["location"])
    assert (location.scheme, location.netloc, location.path) == (
        "https",
        "acme.zendesk.com",
        "/oauth/authorizations/new",
    )
    assert parse_qs(location.query)["scope"] == ["read tickets:write"]


def test_blank_provider_credentials_come_from_env_at_request_time(
    zendesk_db, monkeypatch
):
    """The seeded provider row leaves the client credentials blank so a
    switch from the local to the global client is an env change."""
    db, user = zendesk_db
    provider = _db_provider(db)
    provider.client_id = ""
    provider.client_secret = ""
    db.commit()
    monkeypatch.setenv("ZENDESK_CLIENT_ID", "global-client")
    monkeypatch.setenv("ZENDESK_CLIENT_SECRET", "global-secret")

    response = _login(db, user, "acme")

    query = parse_qs(urlparse(response.headers["location"]).query)
    assert query["client_id"] == ["global-client"]


def test_callback_persists_the_identity_nested_under_user(zendesk_db, monkeypatch):
    db, user = zendesk_db
    post, get = _mock_provider_http(
        monkeypatch, {"user": {"id": 123456, "email": "agent@acme.example"}}
    )

    state, cookies = _login_state(db, user)
    response = _callback(db, state, cookies)

    assert response.status_code == 200
    assert post.call_args.args[0] == "https://acme.zendesk.com/oauth/tokens"
    assert get.call_args.args[0] == "https://acme.zendesk.com/api/v2/users/me.json"
    assert get.call_args.kwargs["allow_redirects"] is False
    grant = _grant(db, user)
    assert grant is not None
    assert grant.provider_user_id == "123456"
    assert grant.email == "agent@acme.example"
    assert grant.instance_url == "acme"
    assert grant.access_token == "new-access"
    assert grant.refresh_token == "new-refresh"


def test_code_exchange_requests_the_maximum_token_lifetimes(zendesk_db, monkeypatch):
    db, user = zendesk_db
    post, _get = _mock_provider_http(
        monkeypatch, {"user": {"id": 1, "email": "agent@acme.example"}}
    )

    state, cookies = _login_state(db, user)
    _callback(db, state, cookies)

    sent = post.call_args.kwargs["data"]
    assert sent["grant_type"] == "authorization_code"
    for name, value in TOKEN_LIFETIMES.items():
        assert sent[name] == value


def test_callback_sends_the_marketplace_headers_to_the_account(
    zendesk_db, monkeypatch, marketplace_configured
):
    db, user = zendesk_db
    post, get = _mock_provider_http(
        monkeypatch, {"user": {"id": 1, "email": "agent@acme.example"}}
    )

    state, cookies = _login_state(db, user)
    assert _callback(db, state, cookies).status_code == 200

    exchange_headers = post.call_args.kwargs["headers"]
    for name, value in MARKETPLACE_HEADERS.items():
        assert exchange_headers[name] == value
    assert get.call_args.kwargs["headers"] == {
        "Authorization": "Bearer new-access",
        **MARKETPLACE_HEADERS,
    }


def test_callback_without_marketplace_settings_sends_no_new_headers(
    zendesk_db, monkeypatch
):
    db, user = zendesk_db
    post, get = _mock_provider_http(
        monkeypatch, {"user": {"id": 1, "email": "agent@acme.example"}}
    )

    state, cookies = _login_state(db, user)
    _callback(db, state, cookies)

    assert post.call_args.kwargs["headers"] == {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    assert get.call_args.kwargs["headers"] == {"Authorization": "Bearer new-access"}


@pytest.mark.parametrize(
    "userinfo",
    [
        {"id": 1, "email": "top-level@acme.example"},
        {"user": None},
        {"user": {"email": "no-id@acme.example"}},
    ],
)
def test_callback_without_a_nested_user_id_persists_no_grant(
    zendesk_db, monkeypatch, userinfo
):
    db, user = zendesk_db
    _mock_provider_http(monkeypatch, userinfo)

    state, cookies = _login_state(db, user)
    response = _callback(db, state, cookies)

    assert response.status_code == 400
    assert _grant(db, user) is None


# ---------- refresh ----------------------------------------------------------


def _expired_grant(db, user, refresh_token: str = "old-refresh") -> UserOAuth:
    grant = UserOAuth(
        user_id=user.id,
        provider="zendesk",
        access_token="old-access",
        refresh_token=refresh_token,
        instance_url="acme",
        provider_user_id="123456",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
    )
    db.add(grant)
    db.commit()
    return grant


def _fake_token_endpoint(monkeypatch, data: dict, status_code: int = 200) -> list:
    posted: list = []

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, **kwargs):
            posted.append((url, kwargs))
            return _ProviderResponse(data, status_code)

    monkeypatch.setattr(tool_config, "create_public_async_client", FakeAsyncClient)
    return posted


@pytest.mark.asyncio
async def test_refresh_requests_the_maximum_token_lifetimes_again(
    zendesk_db, monkeypatch
):
    db, user = zendesk_db
    grant = _expired_grant(db, user)
    posted = _fake_token_endpoint(
        monkeypatch,
        {"access_token": "rotated-access", "refresh_token": "rotated-refresh"},
    )

    assert await tool_config.refresh_oauth_token_if_needed(db, grant, "zendesk")

    url, kwargs = posted[0]
    assert url == "https://acme.zendesk.com/oauth/tokens"
    assert kwargs["data"]["refresh_token"] == "old-refresh"
    for name, value in TOKEN_LIFETIMES.items():
        assert kwargs["data"][name] == value
    assert grant.access_token == "rotated-access"
    assert grant.refresh_token == "rotated-refresh"


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", [True, False])
async def test_refresh_sends_the_marketplace_headers_only_when_configured(
    zendesk_db, monkeypatch, configured
):
    if configured:
        for name, value in MARKETPLACE_ENV.items():
            monkeypatch.setenv(name, value)
    db, user = zendesk_db
    grant = _expired_grant(db, user)
    posted = _fake_token_endpoint(
        monkeypatch,
        {"access_token": "rotated-access", "refresh_token": "rotated-refresh"},
    )

    assert await tool_config.refresh_oauth_token_if_needed(db, grant, "zendesk")

    _url, kwargs = posted[0]
    assert kwargs["headers"] == (MARKETPLACE_HEADERS if configured else {})


@pytest.mark.asyncio
async def test_refresh_rejected_as_invalid_grant_is_permanent(zendesk_db, monkeypatch):
    db, user = zendesk_db
    grant = _expired_grant(db, user)
    _fake_token_endpoint(
        monkeypatch,
        {"error": "invalid_grant", "error_description": "expired"},
        status_code=400,
    )

    with pytest.raises(tool_config._OAuthRefreshPermanentlyInvalid):
        await tool_config.refresh_oauth_token_if_needed(db, grant, "zendesk")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status_code, data",
    [
        (401, {"error": "invalid_client"}),
        (400, {"error": "invalid_request"}),
        (500, {"error": "invalid_grant"}),
        (400, {}),
    ],
)
async def test_other_refresh_failures_stay_transient(
    zendesk_db, monkeypatch, status_code, data
):
    db, user = zendesk_db
    grant = _expired_grant(db, user)
    _fake_token_endpoint(monkeypatch, data, status_code=status_code)

    assert (
        await tool_config.refresh_oauth_token_if_needed(db, grant, "zendesk") is False
    )


# ---------- concurrent rotation on the personal path -------------------------


def _resolve(session_factory, user_id: int):
    config = WebToolConfig(
        db=None, request=None, db_factory=session_factory, user_id=user_id
    )
    return asyncio.run(
        config._resolve_legacy_oauth_access_token(
            provider_name="zendesk", app_id="zendesk"
        )
    )


def _stored(session_factory, grant_id: int) -> UserOAuth:
    with session_factory() as db:
        grant = db.get(UserOAuth, grant_id)
        assert grant is not None
        db.expunge(grant)
        return grant


def test_dead_refresh_token_tombstones_the_grant(
    zendesk_db, session_factory, monkeypatch
):
    db, user = zendesk_db
    grant_id = _expired_grant(db, user).id

    async def dead_refresh(*args, **kwargs):
        raise tool_config._OAuthRefreshPermanentlyInvalid()

    monkeypatch.setattr(tool_config, "refresh_oauth_token_if_needed", dead_refresh)

    resolution = _resolve(session_factory, int(user.id))

    assert resolution.access_token is None
    assert resolution.refresh_failed is True
    stored = _stored(session_factory, grant_id)
    assert stored.access_token == ""
    assert stored.refresh_token is None
    assert stored.provider_user_id == "123456"


def test_losing_a_concurrent_rotation_keeps_the_winners_grant(
    zendesk_db, session_factory, monkeypatch
):
    """Zendesk refresh tokens are single-use: when two refreshes send the
    same token, the loser gets invalid_grant for a token the winner already
    rotated. The loser must not tombstone the winner's fresh grant."""
    db, user = zendesk_db
    grant_id = _expired_grant(db, user).id

    async def lose_the_race(*args, **kwargs):
        with session_factory() as winner:
            row = winner.get(UserOAuth, grant_id)
            row.access_token = "winner-access"
            row.refresh_token = "winner-refresh"
            row.expires_at = datetime.now(timezone.utc) + timedelta(days=2)
            winner.commit()
        raise tool_config._OAuthRefreshPermanentlyInvalid()

    monkeypatch.setattr(tool_config, "refresh_oauth_token_if_needed", lose_the_race)

    resolution = _resolve(session_factory, int(user.id))

    assert resolution.access_token is None
    assert resolution.refresh_failed is True
    assert resolution.credential_present is True
    stored = _stored(session_factory, grant_id)
    assert stored.access_token == "winner-access"
    assert stored.refresh_token == "winner-refresh"
    assert stored.expires_at is not None


def test_tombstone_rereads_the_grant_under_a_write_lock(
    zendesk_db, session_factory, monkeypatch
):
    """Nothing serializes a personal refresh, so the check-then-clear must
    hold the row's write lock from the re-read until the commit; otherwise a
    concurrent rotation can land between the check and the clear."""
    from sqlalchemy import event

    db, user = zendesk_db
    _expired_grant(db, user)
    engine = session_factory.kw["bind"]
    database = engine.url.database
    armed: list[bool] = []
    probes: list[str] = []

    async def dead_refresh(*args, **kwargs):
        armed.append(True)
        raise tool_config._OAuthRefreshPermanentlyInvalid()

    def probe_at_reread(conn, cursor, statement, parameters, context, many):
        if not armed or probes:
            return
        if not statement.lstrip().upper().startswith("SELECT"):
            return
        if "user_oauth" not in statement:
            return
        # BEGIN IMMEDIATE asks only for the write (RESERVED) lock, which the
        # re-read's own open read cursor does not hold.
        other_writer = sqlite3.connect(database, timeout=0, isolation_level=None)
        try:
            other_writer.execute("BEGIN IMMEDIATE")
            other_writer.execute("ROLLBACK")
            probes.append("write lock free")
        except sqlite3.OperationalError as exc:
            probes.append(str(exc))
        finally:
            other_writer.close()

    monkeypatch.setattr(tool_config, "refresh_oauth_token_if_needed", dead_refresh)
    event.listen(engine, "after_cursor_execute", probe_at_reread)
    try:
        _resolve(session_factory, int(user.id))
    finally:
        event.remove(engine, "after_cursor_execute", probe_at_reread)

    assert probes == ["database is locked"]


def test_gmail_tombstone_takes_no_database_write_lock_before_its_own_lock(
    zendesk_db, session_factory, monkeypatch
):
    """On SQLite, Gmail's tombstone enters an in-process lock that every other
    Gmail writer takes before writing. Holding the database write lock while
    waiting for it would block those writers until the busy timeout."""
    from xagent.web.services import gmail_provisioning

    db, user = zendesk_db
    grant = UserOAuth(
        user_id=user.id,
        provider="gmail",
        access_token="old-access",
        refresh_token=None,
        provider_user_id="gmail-user",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
    )
    db.add(grant)
    db.commit()
    database = session_factory.kw["bind"].url.database
    probes: list[str] = []

    async def dead_refresh(*args, **kwargs):
        raise tool_config._OAuthRefreshPermanentlyInvalid()

    def probe_then_clear(oauth_db, *, oauth_account, expected_refresh_token=None):
        _ = expected_refresh_token
        # Stands in for another Gmail writer that already holds the
        # transition lock and now writes.
        other_writer = sqlite3.connect(database, timeout=0)
        try:
            other_writer.execute("UPDATE users SET id = id WHERE id = ?", (user.id,))
            other_writer.commit()
            probes.append("written")
        except sqlite3.OperationalError as exc:
            probes.append(str(exc))
        finally:
            other_writer.close()
        return 0

    monkeypatch.setattr(tool_config, "refresh_oauth_token_if_needed", dead_refresh)
    monkeypatch.setattr(
        gmail_provisioning, "mark_gmail_oauth_reconnect_required", probe_then_clear
    )

    config = WebToolConfig(
        db=None, request=None, db_factory=session_factory, user_id=int(user.id)
    )
    resolution = asyncio.run(
        config._resolve_legacy_oauth_access_token(
            provider_name="google", app_id="gmail"
        )
    )

    assert resolution.refresh_failed is True
    assert probes == ["written"]


# ---------- connector launch -------------------------------------------------


async def _connector_env(db, user) -> dict[str, str]:
    app = mcp_apps.get_app_by_id(db, "zendesk")
    assert app is not None
    server = MCPServer(
        name="Zendesk",
        description="Zendesk",
        managed="external",
        transport="oauth",
        auth={
            "app_id": "zendesk",
            "provider": "zendesk",
            "builtin_provenance": app["launch_config"]["builtin_provenance"],
        },
    )
    db.add(server)
    db.commit()
    db.add(
        UserMCPServer(
            user_id=user.id, mcpserver_id=server.id, is_owner=True, is_active=True
        )
    )
    db.add(
        UserOAuth(
            user_id=user.id,
            provider="zendesk",
            access_token="live-access",
            refresh_token="live-refresh",
            instance_url="acme",
            provider_user_id="123456",
            expires_at=datetime.now(timezone.utc) + timedelta(days=1),
        )
    )
    db.commit()

    configs = await WebToolConfig(
        db=db,
        request=None,
        user=user,
        user_id=user.id,
        workspace_config={"base_dir": "/tmp", "task_id": "task-1"},
    ).get_mcp_server_configs()

    zendesk = next(config for config in configs if config["name"] == "Zendesk")
    assert zendesk["config"]["args"] == ["-m", "xagent.web.tools.mcp.zendesk"]
    return zendesk["config"]["env"]


@pytest.mark.asyncio
async def test_connector_gets_the_token_and_the_grants_subdomain(zendesk_db):
    env = await _connector_env(*zendesk_db)

    assert env["ZENDESK_ACCESS_TOKEN"] == "live-access"
    assert env["ZENDESK_SUBDOMAIN"] == "acme"
    assert "ZENDESK_API_TOKEN" not in env
    assert not set(MARKETPLACE_ENV).intersection(env)


@pytest.mark.asyncio
async def test_connector_gets_the_marketplace_settings(
    zendesk_db, marketplace_configured
):
    env = await _connector_env(*zendesk_db)

    for name, value in MARKETPLACE_ENV.items():
        assert env[name] == value
