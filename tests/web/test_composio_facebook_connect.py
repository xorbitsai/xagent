"""Facebook connects through Composio-hosted auth (#3532).

With ``XAGENT_FACEBOOK_AUTH_BACKEND=composio`` the start keeps xagent-saas's
contract (a redirect carrying ``state`` plus exactly one actor proof cookie),
the browser then begins at our own callback path, Composio's identity
verification comes back to the same path with only a ``session_uri``, and the
stored grant is the one #3531 executes. Composio is faked here; no request
leaves the process.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from http.cookies import SimpleCookie
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from xagent.core.utils.encryption import encrypt_value
from xagent.web import composio_auth
from xagent.web.api import auth as auth_api
from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app
from xagent.web.mcp_apps import ensure_builtin_oauth_server_definition
from xagent.web.models.actor_oauth_flow import ActorOAuthFlowState
from xagent.web.models.database import Base
from xagent.web.models.mcp import UserMCPServer
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.services.user_oauth import delete_scoped_user_oauth_accounts

OWNER = "toby:workspace-7:member-alice"
OTHER_OWNER = "toby:workspace-7:member-bob"
API_KEY = "test-composio-api-key-value"
AUTH_CONFIG_ID = "ac_facebook_test"
USER_ID_SECRET = "test-user-id-secret"
CALLBACK_URL = "https://app.example.com/api/auth/meta/callback"
CALLBACK_PATH = "/api/auth/meta/callback"
COMPOSIO_REDIRECT = "https://connect.composio.dev/link/lk_test"
SESSION_URI = "https://backend.composio.dev/verify/sess_test_value_123"
FACEBOOK_USER_ID = "102030405060"
# xagent-saas's own checks on the start response (region.py,
# toby_onboarding.py), reproduced so a change here that breaks them fails.
SAAS_CALLBACK_PATH = re.compile(
    r"^/api/auth/(?!\.+/)[A-Za-z0-9._-]+/callback$|^/api/mcp/oauth/callback$"
)


class _Response:
    def __init__(self, status_code: int, payload: Any = None) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


class FakeComposio:
    """Composio's API as #3532 uses it, recording every call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Any]] = []
        self.created: list[str] = []
        self.complete_status = 200
        self.complete_account: str | None = None
        self.account_overrides: dict[str, Any] = {}
        self.me_response: tuple[int, Any] | None = None
        self.raise_on: str | None = None

    def paths(self, method: str | None = None) -> list[str]:
        return [path for verb, path, _ in self.calls if method in (None, verb)]

    def request(self, *, method, url, headers, json, timeout):  # noqa: A002
        assert url.startswith(composio_auth.COMPOSIO_API_BASE_URL)
        assert headers["x-api-key"] == API_KEY
        assert timeout
        path = url[len(composio_auth.COMPOSIO_API_BASE_URL) :]
        self.calls.append((method, path, json))
        if self.raise_on == path:
            raise ConnectionError(f"cannot reach {url}")
        if method == "POST" and path == "/connected_accounts/link":
            account_id = f"ca_created_{len(self.created) + 1}"
            self.created.append(account_id)
            return _Response(
                201,
                {
                    "redirect_url": COMPOSIO_REDIRECT,
                    "connected_account_id": account_id,
                    "link_token": "lk_test",
                    "expires_at": "2026-11-04T00:00:00Z",
                },
            )
        if method == "POST" and path == "/connected_accounts/complete_auth":
            if self.complete_status != 200:
                return _Response(
                    self.complete_status, {"error": {"message": "refused"}}
                )
            return _Response(
                200,
                {
                    "connected_account_id": self.complete_account or self.created[-1],
                    "toolkit_slug": "facebook",
                },
            )
        if method == "GET" and path.startswith("/connected_accounts/"):
            account_id = path.rsplit("/", 1)[1]
            account = {
                "id": account_id,
                "status": "ACTIVE",
                "user_id": json_user_id(self.calls),
                "toolkit": {"slug": "facebook"},
                "auth_config": {"id": AUTH_CONFIG_ID},
            }
            account.update(self.account_overrides)
            return _Response(200, account)
        if method == "POST" and path == "/tools/execute/proxy":
            if self.me_response is not None:
                return _Response(*self.me_response)
            return _Response(200, {"data": {"id": FACEBOOK_USER_ID}, "status": 200})
        if method == "DELETE" and path.startswith("/connected_accounts/"):
            return _Response(200, {"success": True})
        raise AssertionError(f"unexpected Composio call {method} {path}")


def json_user_id(calls: list[tuple[str, str, Any]]) -> str | None:
    """The user id the flow completed with, as Composio would report it."""
    for verb, path, body in reversed(calls):
        if verb == "POST" and path == "/connected_accounts/complete_auth":
            return body["user_id"]
    return None


@pytest.fixture
def composio_env(monkeypatch):
    monkeypatch.setenv(composio_auth.FACEBOOK_AUTH_BACKEND_ENV, "composio")
    monkeypatch.setenv(composio_auth.COMPOSIO_API_KEY_ENV, API_KEY)
    monkeypatch.setenv(
        composio_auth.COMPOSIO_FACEBOOK_AUTH_CONFIG_ID_ENV, AUTH_CONFIG_ID
    )
    monkeypatch.setenv(composio_auth.COMPOSIO_USER_ID_SECRET_ENV, USER_ID_SECRET)
    monkeypatch.delenv(composio_auth.CONNECTOR_OAUTH_ROUTE_ENV, raising=False)


@pytest.fixture
def composio(monkeypatch) -> FakeComposio:
    fake = FakeComposio()
    monkeypatch.setattr(composio_auth.requests, "request", fake.request)

    def no_meta_call(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a Composio flow must never call Meta directly")

    monkeypatch.setattr(auth_api.requests, "post", no_meta_call)
    monkeypatch.setattr(auth_api.requests, "get", no_meta_call)
    return fake


@pytest.fixture
def db_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'composio.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with factory() as db:
        user = User(username="workspace-account", password_hash="hash")
        db.add(user)
        app = get_builtin_public_mcp_app("facebook")
        assert app is not None
        db.add(PublicMCPApp(**app))
        db.commit()
        server = ensure_builtin_oauth_server_definition(db, app_id="facebook")
        db.add(
            UserMCPServer(
                user_id=int(user.id),
                mcpserver_id=int(server.id),
                is_owner=True,
                is_active=True,
            )
        )
        db.commit()
        db.refresh(user)
        yield db, user
    engine.dispose()


def _meta_provider() -> SimpleNamespace:
    return SimpleNamespace(
        provider_name="meta",
        client_id=encrypt_value("meta-client-id"),
        client_secret=encrypt_value("meta-client-secret"),
        auth_url="https://www.facebook.com/v25.0/dialog/oauth",
        token_url="https://graph.facebook.com/v25.0/oauth/access_token",
        redirect_uri=CALLBACK_URL,
        userinfo_url="https://graph.facebook.com/v25.0/me?fields=id,email",
        user_id_path="id",
        email_path="email",
        default_scopes=["public_profile"],
    )


class Browser:
    """One browser's cookie jar, scoped by path like a real one."""

    def __init__(self) -> None:
        self.cookies: dict[str, tuple[str, str]] = {}

    def receive(self, response: Any) -> None:
        for header in response.headers.getlist("set-cookie"):
            jar = SimpleCookie()
            jar.load(header)
            for name, morsel in jar.items():
                if morsel["max-age"] == "0" or not morsel.value:
                    self.cookies.pop(name, None)
                else:
                    self.cookies[name] = (morsel.value, morsel["path"] or "/")

    def for_path(self, path: str) -> dict[str, str]:
        return {
            name: value
            for name, (value, scope) in self.cookies.items()
            if path.startswith(scope)
        }

    def get(self, db: Session, url: str) -> Any:
        parts = urlsplit(url)
        assert parts.path == CALLBACK_PATH
        query = {name: values[0] for name, values in parse_qs(parts.query).items()}
        request = SimpleNamespace(query_params=query, cookies=self.for_path(parts.path))
        response = auth_api.generic_oauth_callback(
            "meta", request, db, _meta_provider()
        )
        self.receive(response)
        return response


def _start(db: Session, user: User, browser: Browser, owner: str = OWNER) -> Any:
    response = auth_api.start_builtin_oauth_for_resource_owner(
        provider="meta",
        app_id="facebook",
        user=user,
        resource_owner_key=owner,
        redirect="https://toby.example.com/settings",
        db=db,
        db_provider=_meta_provider(),
    )
    db.commit()
    browser.receive(response)
    return response


def _connect_until_verification(db: Session, user: User, owner: str = OWNER):
    browser = Browser()
    start = _start(db, user, browser, owner)
    begin = browser.get(db, start.headers["location"])
    assert begin.status_code == 302, begin.body
    return browser, start


def _verify_url(**extra: str) -> str:
    query = {"session_uri": SESSION_URI, **extra}
    return f"{CALLBACK_URL}?" + "&".join(f"{k}={v}" for k, v in query.items())


def _grants(db: Session) -> list[UserOAuth]:
    db.expire_all()
    return db.query(UserOAuth).order_by(UserOAuth.id).all()


def _expected_composio_user(user: User, owner: str = OWNER) -> str:
    config = composio_auth.load_config()
    assert config is not None
    return composio_auth.composio_user_id(config, user_id=int(user.id), owner_key=owner)


# --- Switch ---------------------------------------------------------------------


def test_only_facebook_on_meta_follows_the_switch(composio_env, monkeypatch):
    assert composio_auth.facebook_uses_composio("meta", "facebook")
    assert composio_auth.facebook_uses_composio("META", "Facebook")
    for app_id in ("instagram", "meta-ads", "whatsapp", None):
        assert not composio_auth.facebook_uses_composio("meta", app_id)
    assert not composio_auth.facebook_uses_composio("google", "facebook")

    monkeypatch.setenv(composio_auth.FACEBOOK_AUTH_BACKEND_ENV, "native")
    assert not composio_auth.facebook_uses_composio("meta", "facebook")
    monkeypatch.delenv(composio_auth.FACEBOOK_AUTH_BACKEND_ENV)
    assert not composio_auth.facebook_uses_composio("meta", "facebook")


# --- Start ----------------------------------------------------------------------


def test_start_meets_the_xagent_saas_contract(db_session, composio_env, composio):
    db, user = db_session

    response = _start(db, user, Browser())

    assert response.status_code in (302, 307)
    location = urlsplit(response.headers["location"])
    assert f"{location.scheme}://{location.netloc}{location.path}" == CALLBACK_URL
    query = parse_qs(location.query)
    assert set(query) == {"state"}
    state = auth_api.verify_token(query["state"][0])
    assert state["app_id"] == "facebook"
    assert state[composio_auth.AUTH_BACKEND_CLAIM] == "composio"
    # _require_oauth_proof_cookie: exactly one Set-Cookie, an actor proof cookie.
    cookies = response.headers.getlist("set-cookie")
    assert len(cookies) == 1
    assert auth_api.is_actor_oauth_cookie_header(cookies[0])
    jar = SimpleCookie()
    jar.load(cookies[0])
    (morsel,) = jar.values()
    assert SAAS_CALLBACK_PATH.match(morsel["path"])
    assert morsel["path"] == CALLBACK_PATH
    # Nothing is sent to Composio until the browser itself arrives.
    assert composio.calls == []
    assert db.query(ActorOAuthFlowState).count() == 1


def test_start_refuses_while_connector_oauth_routing_is_on(
    db_session, composio_env, composio, monkeypatch
):
    monkeypatch.setenv(composio_auth.CONNECTOR_OAUTH_ROUTE_ENV, " True ")
    db, user = db_session

    response = _start(db, user, Browser())

    assert response.status_code == 503
    assert "set-cookie" not in response.headers
    assert db.query(ActorOAuthFlowState).count() == 0
    assert composio.calls == []


def test_start_refuses_without_composio_configuration(
    db_session, composio_env, composio, monkeypatch
):
    missing = composio_auth.COMPOSIO_USER_ID_SECRET_ENV
    monkeypatch.setenv(missing, "  ")
    db, user = db_session

    response = _start(db, user, Browser())

    assert response.status_code == 500
    assert missing in response.body.decode()
    assert API_KEY not in response.body.decode()
    assert db.query(ActorOAuthFlowState).count() == 0


# --- Begin ----------------------------------------------------------------------


def test_begin_creates_the_link_and_keeps_the_flow(db_session, composio_env, composio):
    db, user = db_session
    browser = Browser()
    start = _start(db, user, browser)

    begin = browser.get(db, start.headers["location"])

    assert begin.status_code == 302
    assert begin.headers["location"] == COMPOSIO_REDIRECT
    assert begin.headers["referrer-policy"] == "no-referrer"
    assert begin.headers["cache-control"] == "no-store"
    ((method, path, body),) = composio.calls
    assert (method, path) == ("POST", "/connected_accounts/link")
    assert body == {
        "auth_config_id": AUTH_CONFIG_ID,
        "user_id": _expected_composio_user(user),
        "callback_url": CALLBACK_URL,
    }
    # The flow is still open for the verification trip.
    assert db.query(ActorOAuthFlowState).count() == 1
    jar = SimpleCookie()
    jar.load(begin.headers["set-cookie"])
    flow_cookie = jar[composio_auth.FLOW_COOKIE_NAME]
    assert flow_cookie["path"] == CALLBACK_PATH
    assert flow_cookie["httponly"]
    assert flow_cookie["samesite"].lower() == "lax"
    assert flow_cookie["secure"]
    assert composio.created[0] not in flow_cookie.value


def test_composio_user_id_is_per_actor_and_opaque(composio_env):
    config = composio_auth.load_config()
    assert config is not None
    alice = composio_auth.composio_user_id(config, user_id=3, owner_key=OWNER)
    bob = composio_auth.composio_user_id(config, user_id=3, owner_key=OTHER_OWNER)
    assert alice != bob
    assert alice == composio_auth.composio_user_id(config, user_id=3, owner_key=OWNER)
    assert "alice" not in alice and "workspace" not in alice
    assert re.fullmatch(r"xagent-[0-9a-f]{64}", alice)


def test_begin_without_the_proof_cookie_creates_no_link(
    db_session, composio_env, composio
):
    db, user = db_session
    browser = Browser()
    start = _start(db, user, browser)
    browser.cookies = {}

    response = browser.get(db, start.headers["location"])

    assert response.status_code == 400
    assert "same browser" in response.body.decode()
    assert composio.calls == []
    assert db.query(ActorOAuthFlowState).count() == 1


def test_begin_with_a_foreign_proof_cookie_creates_no_link(
    db_session, composio_env, composio
):
    db, user = db_session
    browser = Browser()
    start = _start(db, user, browser)
    ((name, (_, path)),) = browser.cookies.items()
    browser.cookies = {name: ("someone-elses-cookie", path)}

    response = browser.get(db, start.headers["location"])

    assert response.status_code == 400
    assert composio.calls == []


def test_begin_refuses_a_native_state(db_session, composio_env, composio, monkeypatch):
    db, user = db_session
    monkeypatch.setenv(composio_auth.FACEBOOK_AUTH_BACKEND_ENV, "native")
    browser = Browser()
    native = _start(db, user, browser)
    native_state = parse_qs(urlsplit(native.headers["location"]).query)["state"][0]

    response = browser.get(db, composio_auth.begin_location(CALLBACK_URL, native_state))

    assert response.status_code == 400
    assert composio.calls == []
    assert db.query(ActorOAuthFlowState).count() == 1


def test_begin_carrying_a_session_uri_is_not_a_begin(
    db_session, composio_env, composio
):
    db, user = db_session
    browser = Browser()
    start = _start(db, user, browser)

    response = browser.get(
        db, start.headers["location"] + f"&session_uri={SESSION_URI}"
    )

    assert response.status_code == 400
    assert composio.calls == []
    assert db.query(ActorOAuthFlowState).count() == 1


def test_begin_after_the_flow_expired_creates_no_link(
    db_session, composio_env, composio
):
    db, user = db_session
    browser = Browser()
    start = _start(db, user, browser)
    db.query(ActorOAuthFlowState).update(
        {
            ActorOAuthFlowState.expires_at: datetime.now(timezone.utc)
            - timedelta(seconds=1)
        }
    )
    db.commit()

    response = browser.get(db, start.headers["location"])

    assert response.status_code == 400
    assert "already been used" in response.body.decode()
    assert composio.calls == []


# --- Verification and grant -------------------------------------------------------


def test_verification_stores_the_composio_grant(db_session, composio_env, composio):
    db, user = db_session
    browser, _ = _connect_until_verification(db, user)

    response = browser.get(db, _verify_url())

    assert response.status_code == 200, response.body
    body = response.body.decode()
    assert "oauth-success" in body and '"facebook"' in body
    composio_user = _expected_composio_user(user)
    complete = [
        c for c in composio.calls if c[1] == "/connected_accounts/complete_auth"
    ]
    assert complete == [
        (
            "POST",
            "/connected_accounts/complete_auth",
            {"session_uri": SESSION_URI, "user_id": composio_user},
        )
    ]
    account_id = composio.created[0]
    assert ("GET", f"/connected_accounts/{account_id}") in [
        (verb, path) for verb, path, _ in composio.calls
    ]
    proxy = [body for _, path, body in composio.calls if path == "/tools/execute/proxy"]
    assert proxy[0]["connected_account_id"] == account_id
    assert proxy[0]["user_id"] == composio_user
    assert proxy[0]["endpoint"].endswith("/me")

    (grant,) = _grants(db)
    assert grant.provider == "facebook"
    assert grant.resource_owner_key == OWNER
    assert grant.token_type == "composio"
    assert grant.access_token == account_id
    assert grant.instance_url == composio_user
    assert grant.provider_user_id == FACEBOOK_USER_ID
    assert grant.refresh_token is None
    assert grant.expires_at is None
    assert db.query(ActorOAuthFlowState).count() == 0
    assert composio_auth.FLOW_COOKIE_NAME not in browser.cookies
    assert composio.paths("DELETE") == []


def test_verification_with_a_state_is_not_accepted(db_session, composio_env, composio):
    db, user = db_session
    browser, start = _connect_until_verification(db, user)
    state = parse_qs(urlsplit(start.headers["location"]).query)["state"][0]

    response = browser.get(db, _verify_url(state=state))

    assert response.status_code == 400
    assert "/connected_accounts/complete_auth" not in composio.paths()
    assert _grants(db) == []
    assert db.query(ActorOAuthFlowState).count() == 1


def test_verification_without_the_proof_cookie_does_not_complete(
    db_session, composio_env, composio
):
    db, user = db_session
    browser, _ = _connect_until_verification(db, user)
    browser.cookies = {
        name: value
        for name, value in browser.cookies.items()
        if name == composio_auth.FLOW_COOKIE_NAME
    }

    response = browser.get(db, _verify_url())

    assert response.status_code == 400
    assert "/connected_accounts/complete_auth" not in composio.paths()
    assert _grants(db) == []


def test_verification_with_a_tampered_flow_cookie_does_not_complete(
    db_session, composio_env, composio
):
    db, user = db_session
    browser, _ = _connect_until_verification(db, user)
    value, path = browser.cookies[composio_auth.FLOW_COOKIE_NAME]
    browser.cookies[composio_auth.FLOW_COOKIE_NAME] = (value[:-4] + "AAAA", path)

    response = browser.get(db, _verify_url())

    assert response.status_code == 400
    assert "/connected_accounts/complete_auth" not in composio.paths()


def test_a_reused_session_uri_is_not_completed_twice(
    db_session, composio_env, composio
):
    db, user = db_session
    browser, _ = _connect_until_verification(db, user)
    saved = dict(browser.cookies)
    assert browser.get(db, _verify_url()).status_code == 200
    browser.cookies = saved

    response = browser.get(db, _verify_url())

    assert response.status_code == 400
    assert "start" in response.body.decode().lower() or "Connect again" in (
        response.body.decode()
    )
    assert composio.paths().count("/connected_accounts/complete_auth") == 1
    assert len(_grants(db)) == 1


def test_a_user_id_composio_refuses_stores_nothing(db_session, composio_env, composio):
    db, user = db_session
    browser, _ = _connect_until_verification(db, user)
    composio.complete_status = 400

    response = browser.get(db, _verify_url())

    assert response.status_code == 400
    assert _grants(db) == []
    assert "/tools/execute/proxy" not in composio.paths()
    # The flow is spent either way: the member starts again.
    assert db.query(ActorOAuthFlowState).count() == 0


@pytest.mark.parametrize(
    "override",
    [
        {"status": "INITIATED"},
        {"user_id": "xagent-someone-else"},
        {"auth_config": {"id": "ac_another_app"}},
        {"toolkit": {"slug": "instagram"}},
    ],
    ids=["not-active", "user-id-mismatch", "auth-config-mismatch", "not-facebook"],
)
def test_an_account_that_fails_verification_is_not_stored(
    db_session, composio_env, composio, override
):
    db, user = db_session
    browser, _ = _connect_until_verification(db, user)
    composio.account_overrides = override

    response = browser.get(db, _verify_url())

    assert response.status_code == 400
    assert _grants(db) == []
    assert "/tools/execute/proxy" not in composio.paths()
    # It is this flow's own account, so it is not left behind unbound.
    assert composio.paths("DELETE") == [f"/connected_accounts/{composio.created[0]}"]


def test_an_account_without_the_deprecated_user_id_is_accepted(
    db_session, composio_env, composio
):
    db, user = db_session
    browser, _ = _connect_until_verification(db, user)
    composio.account_overrides = {"user_id": None}

    assert browser.get(db, _verify_url()).status_code == 200
    assert len(_grants(db)) == 1


def test_an_account_this_flow_did_not_create_is_not_stored_or_deleted(
    db_session, composio_env, composio
):
    db, user = db_session
    browser, _ = _connect_until_verification(db, user)
    composio.complete_account = "ca_belongs_to_someone_else"

    response = browser.get(db, _verify_url())

    assert response.status_code == 400
    assert _grants(db) == []
    assert "/connected_accounts/ca_belongs_to_someone_else" not in composio.paths()


@pytest.mark.parametrize(
    "me_response",
    [
        # Composio answers 200 and carries Graph's own error status (#3531).
        (200, {"data": {"error": {"message": "x"}}, "status": 400}),
        (500, {"error": {"message": "x"}}),
    ],
    ids=["graph-error", "composio-error"],
)
def test_no_facebook_identity_stores_nothing(
    db_session, composio_env, composio, me_response
):
    db, user = db_session
    browser, _ = _connect_until_verification(db, user)
    composio.me_response = me_response

    response = browser.get(db, _verify_url())

    assert response.status_code in (400, 503)
    assert _grants(db) == []
    assert composio.paths("DELETE") == [f"/connected_accounts/{composio.created[0]}"]


def test_a_link_turned_off_during_sign_in_stores_nothing(
    db_session, composio_env, composio, monkeypatch
):
    db, user = db_session
    browser, _ = _connect_until_verification(db, user)
    original = composio.request

    def disconnect_during_lookup(**kwargs):
        if kwargs["url"].endswith("/tools/execute/proxy"):
            db.query(UserMCPServer).update({UserMCPServer.is_active: False})
            db.commit()
        return original(**kwargs)

    monkeypatch.setattr(composio_auth.requests, "request", disconnect_during_lookup)

    response = browser.get(db, _verify_url())

    assert response.status_code == 400
    assert _grants(db) == []
    assert composio.paths("DELETE") == [f"/connected_accounts/{composio.created[0]}"]


def test_reconnecting_replaces_the_grant_and_deletes_the_old_account(
    db_session, composio_env, composio
):
    db, user = db_session
    browser, _ = _connect_until_verification(db, user)
    assert browser.get(db, _verify_url()).status_code == 200
    (old,) = _grants(db)
    old_account = old.access_token

    browser, _ = _connect_until_verification(db, user)
    composio.me_response = (200, {"data": {"id": "999888777"}, "status": 200})
    assert browser.get(db, _verify_url()).status_code == 200

    (new,) = _grants(db)
    assert new.access_token == composio.created[1] != old_account
    assert new.provider_user_id == "999888777"
    assert composio.paths("DELETE") == [f"/connected_accounts/{old_account}"]


def test_a_native_callback_never_spends_a_composio_state(
    db_session, composio_env, composio
):
    db, user = db_session
    browser = Browser()
    start = _start(db, user, browser)

    response = browser.get(db, start.headers["location"] + "&code=meta-code")

    assert response.status_code == 400
    assert composio.calls == []
    assert _grants(db) == []
    assert db.query(ActorOAuthFlowState).count() == 1


def test_no_secret_reaches_logs_or_responses(
    db_session, composio_env, composio, caplog
):
    caplog.set_level(logging.DEBUG)
    db, user = db_session
    responses = []
    browser = Browser()
    responses.append(_start(db, user, browser))
    responses.append(browser.get(db, responses[0].headers["location"]))
    responses.append(browser.get(db, _verify_url()))
    # And a failing run, whose logs are the likelier place for a leak.
    browser, _ = _connect_until_verification(db, user)
    composio.account_overrides = {"status": "FAILED"}
    responses.append(browser.get(db, _verify_url()))
    composio.raise_on = f"/connected_accounts/{composio.created[1]}"
    responses.append(browser.get(db, _verify_url()))

    secrets = [
        API_KEY,
        USER_ID_SECRET,
        SESSION_URI,
        "sess_test_value_123",
        _expected_composio_user(user),
        *composio.created,
    ]
    texts = [record.getMessage() for record in caplog.records]
    texts += [str(record.exc_info) for record in caplog.records if record.exc_info]
    for response in responses:
        texts.append(response.body.decode())
        texts.extend(f"{k}: {v}" for k, v in response.headers.items())
    for secret in secrets:
        assert not any(secret in text for text in texts), secret


# --- Disconnect -------------------------------------------------------------------


def _store_composio_grant(
    db: Session,
    user: User,
    *,
    account_id: str,
    owner: str = OWNER,
    provider: str = "facebook",
    facebook_id: str = FACEBOOK_USER_ID,
) -> None:
    db.add(
        UserOAuth(
            user_id=int(user.id),
            provider=provider,
            resource_owner_key=owner,
            provider_user_id=facebook_id,
            access_token=account_id,
            token_type="composio",
            instance_url="xagent-user",
        )
    )
    db.commit()


def _disconnect(db: Session, user: User, owner: str = OWNER) -> int:
    return delete_scoped_user_oauth_accounts(
        db, user_id=int(user.id), resource_owner_key=owner, providers=["facebook"]
    )


def test_disconnect_deletes_the_composio_account_after_commit(
    db_session, composio_env, composio
):
    db, user = db_session
    _store_composio_grant(db, user, account_id="ca_to_delete")

    assert _disconnect(db, user) == 1
    assert composio.calls == []  # nothing before the local commit
    db.commit()

    assert composio.paths("DELETE") == ["/connected_accounts/ca_to_delete"]
    assert _grants(db) == []


def test_a_rolled_back_disconnect_keeps_the_composio_account(
    db_session, composio_env, composio
):
    db, user = db_session
    _store_composio_grant(db, user, account_id="ca_kept")

    _disconnect(db, user)
    db.rollback()
    db.commit()  # a later, unrelated commit must not delete it either

    assert composio.calls == []
    assert len(_grants(db)) == 1


def test_disconnecting_a_native_grant_calls_composio_never(
    db_session, composio_env, composio
):
    db, user = db_session
    db.add(
        UserOAuth(
            user_id=int(user.id),
            provider="facebook",
            resource_owner_key=OWNER,
            access_token="meta-user-token",
            token_type="bearer",
        )
    )
    db.commit()

    _disconnect(db, user)
    db.commit()

    assert composio.calls == []


def test_a_composio_failure_does_not_fail_the_disconnect(
    db_session, composio_env, composio, caplog
):
    db, user = db_session
    _store_composio_grant(db, user, account_id="ca_flaky")
    composio.raise_on = "/connected_accounts/ca_flaky"

    _disconnect(db, user)
    db.commit()

    assert _grants(db) == []
    assert composio.paths("DELETE") == ["/connected_accounts/ca_flaky"]
    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "deletion" in messages
    assert "ca_flaky" not in messages and API_KEY not in messages


def test_an_account_still_referenced_is_not_deleted(db_session, composio_env, composio):
    db, user = db_session
    _store_composio_grant(db, user, account_id="ca_shared")
    _store_composio_grant(db, user, account_id="ca_shared", owner=OTHER_OWNER)

    _disconnect(db, user)
    db.commit()

    assert composio.calls == []
    assert len(_grants(db)) == 1
