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
from sqlalchemy import create_engine, event
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
INSTAGRAM_AUTH_CONFIG_ID = "ac_instagram_test"
INSTAGRAM_USER_ID = "17841400000000001"
USER_ID_SECRET = "test-user-id-secret"
CALLBACK_URL = "https://app.example.com/api/auth/meta/callback"
CALLBACK_PATH = "/api/auth/meta/callback"
COMPOSIO_REDIRECT = "https://connect.composio.dev/link/lk_test"
SESSION_PREFIX = "https://backend.composio.dev/verify/sess_"
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
        # Composio binds each link to the user id it was created for, and a
        # session can be completed once, only with that user id.
        self.owners: dict[str, str] = {}
        # The auth config each link was created with decides its app.
        self.auth_configs: dict[str, str] = {}
        self.completed: set[str] = set()
        self.complete_status = 200
        # Composio activates the account, then the answer never arrives.
        self.lose_completion = False
        # Composio answers 2xx but with no usable connected_account_id.
        self.complete_without_id = False
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
            self.owners[account_id] = json["user_id"]
            self.auth_configs[account_id] = json["auth_config_id"]
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
            account_id = json["session_uri"].removeprefix(SESSION_PREFIX)
            if account_id not in self.owners or account_id in self.completed:
                return _Response(404, {"error": {"message": "not found"}})
            if json["user_id"] != self.owners[account_id]:
                return _Response(400, {"error": {"message": "user mismatch"}})
            self.completed.add(account_id)
            if self.lose_completion:
                raise TimeoutError(f"read timed out for {url}")
            if self.complete_without_id:
                return _Response(200, {"toolkit_slug": "facebook"})
            return _Response(
                200,
                {
                    "connected_account_id": self.complete_account or account_id,
                    "toolkit_slug": "facebook",
                },
            )
        if method == "GET" and path.startswith("/connected_accounts/"):
            account_id = path.rsplit("/", 1)[1]
            auth_config_id = self.auth_configs.get(account_id, AUTH_CONFIG_ID)
            toolkit = (
                "instagram"
                if auth_config_id == INSTAGRAM_AUTH_CONFIG_ID
                else "facebook"
            )
            account = {
                "id": account_id,
                "status": "ACTIVE" if account_id in self.completed else "INITIALIZING",
                "user_id": self.owners.get(account_id),
                "toolkit": {"slug": toolkit},
                "auth_config": {"id": auth_config_id},
            }
            account.update(self.account_overrides)
            return _Response(200, account)
        if method == "POST" and path == "/tools/execute/proxy":
            if self.me_response is not None:
                return _Response(*self.me_response)
            if json["endpoint"].startswith("https://graph.instagram.com/"):
                return _Response(
                    200, {"data": {"user_id": INSTAGRAM_USER_ID}, "status": 200}
                )
            return _Response(200, {"data": {"id": FACEBOOK_USER_ID}, "status": 200})
        if method == "DELETE" and path.startswith("/connected_accounts/"):
            return _Response(200, {"success": True})
        raise AssertionError(f"unexpected Composio call {method} {path}")


@pytest.fixture
def composio_env(monkeypatch):
    monkeypatch.setenv(composio_auth.FACEBOOK_AUTH_BACKEND_ENV, "composio")
    monkeypatch.setenv(composio_auth.COMPOSIO_API_KEY_ENV, API_KEY)
    monkeypatch.setenv(
        composio_auth.COMPOSIO_FACEBOOK_AUTH_CONFIG_ID_ENV, AUTH_CONFIG_ID
    )
    monkeypatch.setenv(composio_auth.COMPOSIO_USER_ID_SECRET_ENV, USER_ID_SECRET)
    monkeypatch.delenv(composio_auth.CONNECTOR_OAUTH_ROUTE_ENV, raising=False)
    # Instagram stays off unless a test switches it on.
    monkeypatch.delenv("XAGENT_INSTAGRAM_AUTH_BACKEND", raising=False)
    monkeypatch.delenv("COMPOSIO_INSTAGRAM_AUTH_CONFIG_ID", raising=False)


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

    # pysqlite does not emit BEGIN itself, so SAVEPOINT and ROLLBACK would not
    # undo anything; SQLAlchemy's documented recipe makes them real.
    @event.listens_for(engine, "connect")
    def _no_driver_transactions(dbapi_connection, _record):
        dbapi_connection.isolation_level = None

    @event.listens_for(engine, "begin")
    def _emit_begin(connection):
        connection.exec_driver_sql("BEGIN")

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


def _start(
    db: Session,
    user: User,
    browser: Browser,
    owner: str = OWNER,
    app_id: str = "facebook",
) -> Any:
    response = auth_api.start_builtin_oauth_for_resource_owner(
        provider="meta",
        app_id=app_id,
        user=user,
        resource_owner_key=owner,
        redirect="https://toby.example.com/settings",
        db=db,
        db_provider=_meta_provider(),
    )
    db.commit()
    browser.receive(response)
    return response


def _connect_until_verification(
    db: Session, user: User, owner: str = OWNER, app_id: str = "facebook"
):
    browser = Browser()
    start = _start(db, user, browser, owner, app_id)
    begin = browser.get(db, start.headers["location"])
    assert begin.status_code == 302, begin.body
    return browser, start


def _session_for(account_id: str) -> str:
    return f"{SESSION_PREFIX}{account_id}"


def _flow_cookies(browser: Browser) -> list[str]:
    return [
        name
        for name in browser.cookies
        if name.startswith(composio_auth.FLOW_COOKIE_PREFIX)
    ]


def _verify_url(
    composio: FakeComposio, account_id: str | None = None, **extra: str
) -> str:
    """The verification trip for ``account_id``'s session (default: the latest)."""
    session = _session_for(account_id or composio.created[-1])
    query = {"session_uri": session, **extra}
    return f"{CALLBACK_URL}?" + "&".join(f"{k}={v}" for k, v in query.items())


def _grants(db: Session) -> list[UserOAuth]:
    db.expire_all()
    return db.query(UserOAuth).order_by(UserOAuth.id).all()


def _expected_composio_user(user: User, owner: str = OWNER) -> str:
    config = composio_auth.load_config(composio_auth.FACEBOOK_APP_ID)
    assert config is not None
    return composio_auth.composio_user_id(config, user_id=int(user.id), owner_key=owner)


# --- Switch ---------------------------------------------------------------------


def _via_composio(provider: str, app_id: str | None) -> str | None:
    app = composio_auth.composio_app(provider, app_id)
    return app.app_id if app else None


def test_each_app_follows_only_its_own_switch(composio_env, monkeypatch):
    """Facebook and Instagram each have their own switch (#3533); Meta Ads
    and WhatsApp share the meta provider and have none."""
    assert _via_composio("meta", "facebook") == "facebook"
    assert _via_composio("META", "Facebook") == "facebook"
    for app_id in ("instagram", "meta-ads", "whatsapp", None):
        assert _via_composio("meta", app_id) is None
    assert _via_composio("google", "facebook") is None

    monkeypatch.setenv("XAGENT_INSTAGRAM_AUTH_BACKEND", "composio")
    monkeypatch.setenv(composio_auth.FACEBOOK_AUTH_BACKEND_ENV, "native")
    assert _via_composio("meta", "instagram") == "instagram"
    assert _via_composio("meta", "facebook") is None
    monkeypatch.delenv(composio_auth.FACEBOOK_AUTH_BACKEND_ENV)
    assert _via_composio("meta", "facebook") is None


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
    (flow_cookie,) = [
        morsel
        for name, morsel in jar.items()
        if name.startswith(composio_auth.FLOW_COOKIE_PREFIX)
    ]
    assert flow_cookie["path"] == CALLBACK_PATH
    assert flow_cookie["httponly"]
    assert flow_cookie["samesite"].lower() == "lax"
    assert flow_cookie["secure"]
    assert composio.created[0] not in flow_cookie.value


def test_composio_user_id_is_per_actor_and_opaque(composio_env):
    config = composio_auth.load_config(composio_auth.FACEBOOK_APP_ID)
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
        db, start.headers["location"] + f"&session_uri={_session_for('ca_x')}"
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

    response = browser.get(db, _verify_url(composio))

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
            {
                "session_uri": _session_for(composio.created[0]),
                "user_id": composio_user,
            },
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
    assert _flow_cookies(browser) == []
    assert composio.paths("DELETE") == []


def test_verification_with_a_state_is_not_accepted(db_session, composio_env, composio):
    db, user = db_session
    browser, start = _connect_until_verification(db, user)
    state = parse_qs(urlsplit(start.headers["location"]).query)["state"][0]

    response = browser.get(db, _verify_url(composio, state=state))

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
        if name.startswith(composio_auth.FLOW_COOKIE_PREFIX)
    }

    response = browser.get(db, _verify_url(composio))

    assert response.status_code == 400
    assert "/connected_accounts/complete_auth" not in composio.paths()
    assert _grants(db) == []


def test_verification_with_a_tampered_flow_cookie_does_not_complete(
    db_session, composio_env, composio
):
    db, user = db_session
    browser, _ = _connect_until_verification(db, user)
    (name,) = _flow_cookies(browser)
    value, path = browser.cookies[name]
    browser.cookies[name] = (value[:-4] + "AAAA", path)

    response = browser.get(db, _verify_url(composio))

    assert response.status_code == 400
    assert "/connected_accounts/complete_auth" not in composio.paths()


def test_a_reused_session_uri_is_not_completed_twice(
    db_session, composio_env, composio
):
    db, user = db_session
    browser, _ = _connect_until_verification(db, user)
    saved = dict(browser.cookies)
    assert browser.get(db, _verify_url(composio)).status_code == 200
    browser.cookies = saved

    response = browser.get(db, _verify_url(composio))

    assert response.status_code == 400
    assert "start" in response.body.decode().lower() or "Connect again" in (
        response.body.decode()
    )
    assert composio.paths().count("/connected_accounts/complete_auth") == 1
    assert len(_grants(db)) == 1


def _begin(
    db: Session,
    user: User,
    browser: Browser,
    owner: str = OWNER,
    app_id: str = "facebook",
) -> None:
    start = _start(db, user, browser, owner, app_id)
    assert browser.get(db, start.headers["location"]).status_code == 302


def test_two_connects_in_one_browser_each_complete_their_own_flow(
    db_session, composio_env, composio
):
    """Begin A, begin B, then A's session returns first (R1)."""
    db, user = db_session
    browser = Browser()
    _begin(db, user, browser)
    _begin(db, user, browser)
    first, second = composio.created
    assert len(_flow_cookies(browser)) == 2

    assert browser.get(db, _verify_url(composio, first)).status_code == 200

    (grant,) = _grants(db)
    assert grant.access_token == first
    # B's flow and its cookie are untouched, and it still completes.
    assert db.query(ActorOAuthFlowState).count() == 1
    assert len(_flow_cookies(browser)) == 1
    assert browser.get(db, _verify_url(composio, second)).status_code == 200
    (grant,) = _grants(db)
    assert grant.access_token == second
    # The reconnect replaced A's grant, so A's account is deleted.
    assert composio.paths("DELETE") == [f"/connected_accounts/{first}"]


def test_another_actors_begin_drops_the_first_actors_flow(
    db_session, composio_env, composio
):
    """A browser keeps one actor's flows: the earlier actor's session then
    fails Composio's user check and is never activated."""
    db, user = db_session
    browser = Browser()
    _begin(db, user, browser, OWNER)
    _begin(db, user, browser, OTHER_OWNER)
    alice, bob = composio.created
    assert len(_flow_cookies(browser)) == 1

    assert browser.get(db, _verify_url(composio, alice)).status_code == 400

    assert alice not in composio.completed
    assert _grants(db) == []
    # Bob's flow is not spent by the refused session.
    assert browser.get(db, _verify_url(composio, bob)).status_code == 200
    (grant,) = _grants(db)
    assert (grant.resource_owner_key, grant.access_token) == (OTHER_OWNER, bob)


def test_a_session_whose_flow_cookie_expired_is_not_left_active(
    db_session, composio_env, composio
):
    """A's cookie expired while the member was on Facebook; B is still open.
    Composio activates A (same actor, same user id), which no open flow
    created: it is deleted once Composio confirms it is this actor's."""
    db, user = db_session
    browser = Browser()
    _begin(db, user, browser)
    first_cookie = _flow_cookies(browser)[0]
    _begin(db, user, browser)
    first, second = composio.created
    browser.cookies.pop(first_cookie)

    response = browser.get(db, _verify_url(composio, first))

    assert response.status_code == 400
    assert _grants(db) == []
    assert composio.paths("DELETE") == [f"/connected_accounts/{first}"]
    assert browser.get(db, _verify_url(composio, second)).status_code == 200


def test_a_begin_past_the_open_flow_limit_creates_no_link(
    db_session, composio_env, composio
):
    db, user = db_session
    browser = Browser()
    for _ in range(3):
        _begin(db, user, browser)
    start = _start(db, user, browser)

    response = browser.get(db, start.headers["location"])

    assert response.status_code == 400
    assert "Too many" in response.body.decode()
    assert composio.paths().count("/connected_accounts/link") == 3


def test_a_reloaded_begin_deletes_nothing_and_its_old_link_is_cleaned_if_completed(
    db_session, composio_env, composio
):
    """The original window may be completing the replaced link while the
    reload runs, so the reload deletes nothing. If that old link's session
    comes back later, no open flow claims it and it is removed there."""
    db, user = db_session
    browser = Browser()
    start = _start(db, user, browser)
    assert browser.get(db, start.headers["location"]).status_code == 302
    assert browser.get(db, start.headers["location"]).status_code == 302
    replaced, current = composio.created
    assert composio.paths("DELETE") == []

    assert browser.get(db, _verify_url(composio, replaced)).status_code == 400
    assert composio.paths("DELETE") == [f"/connected_accounts/{replaced}"]
    assert browser.get(db, _verify_url(composio, current)).status_code == 200
    assert _grants(db)[0].access_token == current


def test_a_reload_racing_a_completion_never_deletes_the_bound_account(
    db_session, composio_env, composio, monkeypatch
):
    """The original window completes while the reload waits on Composio."""
    db, user = db_session
    browser = Browser()
    start = _start(db, user, browser)
    assert browser.get(db, start.headers["location"]).status_code == 302
    first = composio.created[0]
    original = composio.request
    raced = []

    def complete_during_link(**kwargs):
        if kwargs["url"].endswith("/connected_accounts/link") and not raced:
            raced.append(True)
            other_window = Browser()
            other_window.cookies = dict(browser.cookies)
            assert other_window.get(db, _verify_url(composio, first)).status_code == 200
        return original(**kwargs)

    monkeypatch.setattr(composio_auth.requests, "request", complete_during_link)
    browser.get(db, start.headers["location"])

    (grant,) = _grants(db)
    assert grant.access_token == first
    assert f"/connected_accounts/{first}" not in composio.paths("DELETE")


@pytest.mark.parametrize("lost_answer", [False, True], ids=["reply", "lost-reply"])
@pytest.mark.parametrize("flow_end", ["expired", "spent-by-another-request"])
def test_completion_crossing_the_deadline_cleans_only_an_unspent_flow(
    db_session, composio_env, composio, monkeypatch, lost_answer, flow_end
):
    db, user = db_session
    browser, _ = _connect_until_verification(db, user)
    account = composio.created[0]
    composio.lose_completion = lost_answer
    complete = composio_auth.complete_auth

    def complete_as_flow_ends(*args, **kwargs):
        try:
            return complete(*args, **kwargs)
        finally:
            if flow_end == "expired":
                # The request began in time, but its network call crossed the
                # deadline. The server-side flow still exists, unclaimed.
                db.query(ActorOAuthFlowState).update(
                    {
                        ActorOAuthFlowState.expires_at: datetime.now(timezone.utc)
                        - timedelta(seconds=1)
                    }
                )
                db.commit()
            else:
                # Another request owns the flow now and has bound its account.
                db.query(ActorOAuthFlowState).delete()
                _store_composio_grant(db, user, account_id=account)

    monkeypatch.setattr(composio_auth, "complete_auth", complete_as_flow_ends)
    response = browser.get(db, _verify_url(composio, account))

    assert response.status_code == (503 if lost_answer else 400)
    assert account in composio.completed
    assert db.query(ActorOAuthFlowState).count() == 0
    if flow_end == "expired":
        assert _grants(db) == []
        assert composio.paths("DELETE") == [f"/connected_accounts/{account}"]
    else:
        assert _grants(db)[0].access_token == account
        assert composio.paths("DELETE") == []


def test_a_lost_completion_ends_the_flow_composio_activated(
    db_session, composio_env, composio
):
    """Composio activated the account, then the answer was lost (R2)."""
    db, user = db_session
    browser, _ = _connect_until_verification(db, user)
    composio.lose_completion = True

    response = browser.get(db, _verify_url(composio))

    assert response.status_code == 503
    assert _grants(db) == []
    assert db.query(ActorOAuthFlowState).count() == 0
    assert _flow_cookies(browser) == []
    assert composio.paths("DELETE") == [f"/connected_accounts/{composio.created[0]}"]


def test_a_lost_completion_among_several_flows_ends_only_the_activated_one(
    db_session, composio_env, composio
):
    db, user = db_session
    browser = Browser()
    _begin(db, user, browser)
    _begin(db, user, browser)
    first, second = composio.created
    composio.lose_completion = True

    response = browser.get(db, _verify_url(composio, second))

    assert response.status_code == 503
    assert composio.paths("DELETE") == [f"/connected_accounts/{second}"]
    # The other flow is untouched and still completes.
    assert db.query(ActorOAuthFlowState).count() == 1
    composio.lose_completion = False
    assert browser.get(db, _verify_url(composio, first)).status_code == 200


def test_a_lost_answer_before_activation_deletes_nothing(
    db_session, composio_env, composio
):
    db, user = db_session
    browser, _ = _connect_until_verification(db, user)
    composio.raise_on = "/connected_accounts/complete_auth"

    response = browser.get(db, _verify_url(composio))

    assert response.status_code == 503
    assert composio.paths("DELETE") == []
    assert db.query(ActorOAuthFlowState).count() == 1


def test_a_lost_completion_of_an_expired_flow_does_not_end_another_flow(
    db_session, composio_env, composio
):
    """A's cookie expired and B is open; A's lost completion must not be
    pinned on B just because B is the only open flow."""
    db, user = db_session
    browser = Browser()
    _begin(db, user, browser)
    first_cookie = _flow_cookies(browser)[0]
    _begin(db, user, browser)
    first, second = composio.created
    browser.cookies.pop(first_cookie)
    composio.lose_completion = True

    response = browser.get(db, _verify_url(composio, first))

    assert response.status_code == 503
    assert composio.paths("DELETE") == []
    composio.lose_completion = False
    assert browser.get(db, _verify_url(composio, second)).status_code == 200


def test_a_completion_without_an_id_is_handled_as_a_lost_answer(
    db_session, composio_env, composio
):
    db, user = db_session
    browser, _ = _connect_until_verification(db, user)
    composio.complete_without_id = True

    response = browser.get(db, _verify_url(composio))

    assert response.status_code == 503
    assert composio.paths("DELETE") == [f"/connected_accounts/{composio.created[0]}"]


def _denied_url(account_id: str) -> str:
    # What Composio sends back when the member cancels on Facebook (seen live).
    return (
        f"{CALLBACK_URL}?status=failed&error=access_denied"
        f"&error_description=Permissions+error&connected_account_id={account_id}"
    )


def test_a_denied_authorization_says_what_to_do_and_ends_its_flow(
    db_session, composio_env, composio
):
    db, user = db_session
    browser, _ = _connect_until_verification(db, user)

    response = browser.get(db, _denied_url(composio.created[0]))

    assert response.status_code == 400
    body = response.body.decode()
    assert "Permission was not granted" in body
    assert "access_denied" in body
    assert composio.created[0] not in body
    assert _grants(db) == []
    assert db.query(ActorOAuthFlowState).count() == 0
    assert _flow_cookies(browser) == []
    assert composio.paths("DELETE") == [f"/connected_accounts/{composio.created[0]}"]


def test_a_denied_return_for_an_account_no_flow_created_deletes_nothing(
    db_session, composio_env, composio
):
    db, user = db_session
    browser, _ = _connect_until_verification(db, user)

    response = browser.get(db, _denied_url("ca_someone_elses"))

    assert response.status_code == 400
    assert "Permission was not granted" in response.body.decode()
    assert composio.paths("DELETE") == []
    assert db.query(ActorOAuthFlowState).count() == 1
    assert len(_flow_cookies(browser)) == 1


def test_a_return_without_verification_tells_operators_and_deletes_the_account(
    db_session, composio_env, composio, caplog
):
    """A Composio project without OAuth user verification activates the
    connection itself and sends status=success; it is never bound here."""
    db, user = db_session
    browser, _ = _connect_until_verification(db, user)
    account = composio.created[0]

    response = browser.get(
        db, f"{CALLBACK_URL}?status=success&connected_account_id={account}"
    )

    assert response.status_code == 503
    assert "Permission was not granted" not in response.body.decode()
    assert "verification_not_configured" in caplog.text
    assert _grants(db) == []
    assert db.query(ActorOAuthFlowState).count() == 0
    assert composio.paths("DELETE") == [f"/connected_accounts/{account}"]


def test_a_bare_success_return_is_not_logged_as_an_operator_error(
    db_session, composio_env, composio, caplog
):
    db, _ = db_session

    response = Browser().get(
        db, f"{CALLBACK_URL}?status=success&connected_account_id=ca_stray"
    )

    assert response.status_code == 503
    assert composio.paths("DELETE") == []
    records = [r for r in caplog.records if "without verification" in r.getMessage()]
    assert [r.levelno for r in records] == [logging.WARNING]


def test_a_user_id_composio_refuses_stores_nothing(db_session, composio_env, composio):
    db, user = db_session
    browser, _ = _connect_until_verification(db, user)
    composio.complete_status = 400

    response = browser.get(db, _verify_url(composio))

    assert response.status_code == 400
    assert _grants(db) == []
    assert "/tools/execute/proxy" not in composio.paths()
    # Nothing was activated, so the flow is not spent by a refused session.
    assert db.query(ActorOAuthFlowState).count() == 1


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

    response = browser.get(db, _verify_url(composio))

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

    assert browser.get(db, _verify_url(composio)).status_code == 200
    assert len(_grants(db)) == 1


def test_an_account_this_flow_did_not_create_is_not_stored_or_deleted(
    db_session, composio_env, composio
):
    db, user = db_session
    browser, _ = _connect_until_verification(db, user)
    composio.complete_account = "ca_belongs_to_someone_else"

    response = browser.get(db, _verify_url(composio))

    assert response.status_code == 400
    assert _grants(db) == []
    # Looked up, found not to be this actor's, and left alone.
    assert "/connected_accounts/ca_belongs_to_someone_else" in composio.paths("GET")
    assert composio.paths("DELETE") == []


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

    response = browser.get(db, _verify_url(composio))

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

    response = browser.get(db, _verify_url(composio))

    assert response.status_code == 400
    assert _grants(db) == []
    assert composio.paths("DELETE") == [f"/connected_accounts/{composio.created[0]}"]


def test_reconnecting_replaces_the_grant_and_deletes_the_old_account(
    db_session, composio_env, composio
):
    db, user = db_session
    browser, _ = _connect_until_verification(db, user)
    assert browser.get(db, _verify_url(composio)).status_code == 200
    (old,) = _grants(db)
    old_account = old.access_token

    browser, _ = _connect_until_verification(db, user)
    composio.me_response = (200, {"data": {"id": "999888777"}, "status": 200})
    assert browser.get(db, _verify_url(composio)).status_code == 200

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
    responses.append(browser.get(db, _verify_url(composio)))
    # And two failing runs, whose logs are the likelier place for a leak.
    browser, _ = _connect_until_verification(db, user)
    composio.account_overrides = {"status": "FAILED"}
    responses.append(browser.get(db, _verify_url(composio)))
    composio.account_overrides = {}
    # A fresh flow, so the request reaches the account lookup that raises.
    browser, _ = _connect_until_verification(db, user)
    lookup = f"/connected_accounts/{composio.created[2]}"
    composio.raise_on = lookup
    responses.append(browser.get(db, _verify_url(composio)))
    assert ("GET", lookup) in [(verb, path) for verb, path, _ in composio.calls]
    assert responses[-1].status_code == 503

    secrets = [
        API_KEY,
        USER_ID_SECRET,
        _expected_composio_user(user),
        *composio.created,
        *(_session_for(account_id) for account_id in composio.created),
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


def test_a_savepoint_committed_then_rolled_back_outside_keeps_the_account(
    db_session, composio_env, composio
):
    db, user = db_session
    _store_composio_grant(db, user, account_id="ca_savepoint")

    savepoint = db.begin_nested()
    _disconnect(db, user)
    savepoint.commit()  # SQLAlchemy fires after_commit for this too
    assert composio.calls == []
    db.rollback()
    db.commit()

    assert composio.calls == []
    assert len(_grants(db)) == 1


def test_a_savepoint_rolled_back_then_committed_outside_keeps_the_account(
    db_session, composio_env, composio
):
    db, user = db_session
    _store_composio_grant(db, user, account_id="ca_savepoint")

    savepoint = db.begin_nested()
    _disconnect(db, user)
    savepoint.rollback()
    db.commit()

    assert composio.calls == []
    assert len(_grants(db)) == 1


def test_a_savepoint_committed_with_its_transaction_deletes_the_account(
    db_session, composio_env, composio
):
    db, user = db_session
    _store_composio_grant(db, user, account_id="ca_savepoint")

    savepoint = db.begin_nested()
    _disconnect(db, user)
    savepoint.commit()
    assert composio.calls == []
    db.commit()

    assert composio.paths("DELETE") == ["/connected_accounts/ca_savepoint"]
    assert _grants(db) == []


def test_an_account_still_referenced_is_not_deleted(db_session, composio_env, composio):
    db, user = db_session
    _store_composio_grant(db, user, account_id="ca_shared")
    _store_composio_grant(db, user, account_id="ca_shared", owner=OTHER_OWNER)

    _disconnect(db, user)
    db.commit()

    assert composio.calls == []
    assert len(_grants(db)) == 1


# --- Instagram (#3533) -----------------------------------------------------------


@pytest.fixture
def instagram_connect(db_session, composio_env, composio, monkeypatch):
    """Instagram switched to Composio, its connector installed, and Composio
    answering for an Instagram account."""
    monkeypatch.setenv("XAGENT_INSTAGRAM_AUTH_BACKEND", "composio")
    monkeypatch.setenv("COMPOSIO_INSTAGRAM_AUTH_CONFIG_ID", INSTAGRAM_AUTH_CONFIG_ID)
    db, user = db_session
    app = get_builtin_public_mcp_app("instagram")
    assert app is not None
    db.add(PublicMCPApp(**app))
    db.commit()
    server = ensure_builtin_oauth_server_definition(db, app_id="instagram")
    db.add(
        UserMCPServer(
            user_id=int(user.id),
            mcpserver_id=int(server.id),
            is_owner=True,
            is_active=True,
        )
    )
    db.commit()
    return db, user, composio


@pytest.fixture
def instagram_only(instagram_connect, monkeypatch):
    """A deployment that turned only Instagram over to Composio."""
    monkeypatch.delenv(composio_auth.FACEBOOK_AUTH_BACKEND_ENV)
    monkeypatch.delenv(composio_auth.COMPOSIO_FACEBOOK_AUTH_CONFIG_ID_ENV)
    return instagram_connect


def test_an_instagram_only_deployment_connects_instagram(instagram_only):
    """Nothing in start, begin or verify needs the Facebook config."""
    db, user, composio = instagram_only
    browser, _ = _connect_until_verification(db, user, app_id="instagram")

    assert browser.get(db, _verify_url(composio)).status_code == 200

    (grant,) = _grants(db)
    assert (grant.provider, grant.provider_user_id) == ("instagram", INSTAGRAM_USER_ID)
    assert composio.paths("DELETE") == []


def test_an_instagram_only_deployment_deletes_an_unclaimed_account(instagram_only):
    """The first flow's cookie is gone when its session returns; the account
    is recognised by the Instagram auth config and deleted."""
    db, user, composio = instagram_only
    browser = Browser()
    _begin(db, user, browser, app_id="instagram")
    (lost,) = composio.created
    for name in _flow_cookies(browser):
        browser.cookies.pop(name)
    _begin(db, user, browser, app_id="instagram")

    response = browser.get(db, _verify_url(composio, lost))

    assert response.status_code == 400
    assert _grants(db) == []
    assert composio.paths("DELETE") == [f"/connected_accounts/{lost}"]


def test_instagram_start_refuses_without_its_auth_config(
    instagram_connect, monkeypatch
):
    """Instagram's own auth config is required, even with Facebook's set."""
    db, user, composio = instagram_connect
    monkeypatch.delenv("COMPOSIO_INSTAGRAM_AUTH_CONFIG_ID")

    response = _start(db, user, Browser(), app_id="instagram")

    assert response.status_code == 500
    assert "COMPOSIO_INSTAGRAM_AUTH_CONFIG_ID" in response.body.decode()
    assert db.query(ActorOAuthFlowState).count() == 0
    assert composio.calls == []


def test_the_open_flow_limit_counts_facebook_and_instagram_together(
    instagram_connect,
):
    db, user, composio = instagram_connect
    browser = Browser()
    _begin(db, user, browser)
    _begin(db, user, browser)
    _begin(db, user, browser, app_id="instagram")
    start = _start(db, user, browser, app_id="instagram")

    response = browser.get(db, start.headers["location"])

    assert response.status_code == 400
    assert "Too many" in response.body.decode()
    assert composio.paths().count("/connected_accounts/link") == 3


@pytest.mark.parametrize(
    "config_removed", [False, True], ids=["configured", "config-removed"]
)
def test_a_denied_instagram_flow_ends_and_deletes_its_account(
    instagram_connect, monkeypatch, config_removed
):
    """The account is deleted with the shared API key, so it is cleaned up
    even when Instagram's auth config went away after the flow began."""
    db, user, composio = instagram_connect
    browser, _ = _connect_until_verification(db, user, app_id="instagram")
    if config_removed:
        monkeypatch.delenv("COMPOSIO_INSTAGRAM_AUTH_CONFIG_ID")

    response = browser.get(db, _denied_url(composio.created[0]))

    assert response.status_code == 400
    assert "Permission was not granted" in response.body.decode()
    assert _grants(db) == []
    assert _flow_cookies(browser) == []
    assert composio.paths("DELETE") == [f"/connected_accounts/{composio.created[0]}"]


def test_instagram_connect_stores_an_instagram_grant(instagram_connect):
    """The flow is the same as Facebook's; the app's own auth config, toolkit
    and Instagram Login identity are used, and the grant is keyed
    ``instagram`` for the Instagram connector."""
    db, user, composio = instagram_connect
    browser, _ = _connect_until_verification(db, user, app_id="instagram")
    link = [body for _, path, body in composio.calls if path.endswith("/link")]
    assert link[0]["auth_config_id"] == INSTAGRAM_AUTH_CONFIG_ID

    response = browser.get(db, _verify_url(composio))

    assert response.status_code == 200, response.body
    assert '"instagram"' in response.body.decode()
    proxy = [body for _, path, body in composio.calls if path == "/tools/execute/proxy"]
    assert proxy[0]["endpoint"] == "https://graph.instagram.com/v25.0/me"
    assert proxy[0]["parameters"] == [
        {"name": "fields", "value": "user_id", "type": "query"}
    ]
    (grant,) = _grants(db)
    assert grant.provider == "instagram"
    assert grant.resource_owner_key == OWNER
    assert grant.token_type == "composio"
    assert grant.access_token == composio.created[0]
    assert grant.provider_user_id == INSTAGRAM_USER_ID
    assert grant.expires_at is None


def test_facebook_and_instagram_connects_in_one_browser_each_complete(
    instagram_connect,
):
    """Begin Facebook, then Instagram, in one browser; the Instagram session
    returns while the Facebook flow is still open. Each flow is checked
    against its own app's auth config and toolkit and stored under its own
    app, and the other flow stays usable."""
    db, user, composio = instagram_connect
    browser = Browser()
    _begin(db, user, browser)
    _begin(db, user, browser, app_id="instagram")
    facebook, instagram = composio.created
    assert len(_flow_cookies(browser)) == 2

    assert browser.get(db, _verify_url(composio, instagram)).status_code == 200
    assert len(_flow_cookies(browser)) == 1
    assert browser.get(db, _verify_url(composio, facebook)).status_code == 200

    grants = {grant.provider: grant for grant in _grants(db)}
    assert set(grants) == {"facebook", "instagram"}
    assert (grants["facebook"].access_token, grants["facebook"].provider_user_id) == (
        facebook,
        FACEBOOK_USER_ID,
    )
    assert (
        grants["instagram"].access_token,
        grants["instagram"].provider_user_id,
    ) == (instagram, INSTAGRAM_USER_ID)
    proxy = [body for _, path, body in composio.calls if path == "/tools/execute/proxy"]
    assert [body["endpoint"] for body in proxy] == [
        "https://graph.instagram.com/v25.0/me",
        "https://graph.facebook.com/v25.0/me",
    ]
    assert composio.paths("DELETE") == []


def test_one_apps_missing_config_does_not_block_the_others_verification(
    instagram_connect, monkeypatch
):
    """Instagram's auth config is removed while both flows are open: the
    Facebook connect still completes, and the Instagram one is refused with
    its just-activated account deleted rather than left unbound."""
    db, user, composio = instagram_connect
    browser = Browser()
    _begin(db, user, browser)
    _begin(db, user, browser, app_id="instagram")
    facebook, instagram = composio.created
    monkeypatch.delenv("COMPOSIO_INSTAGRAM_AUTH_CONFIG_ID")

    assert browser.get(db, _verify_url(composio, facebook)).status_code == 200
    (grant,) = _grants(db)
    assert (grant.provider, grant.access_token) == ("facebook", facebook)

    response = browser.get(db, _verify_url(composio, instagram))
    assert response.status_code == 503
    assert "Instagram" in response.body.decode()
    assert [grant.provider for grant in _grants(db)] == ["facebook"]
    assert composio.paths("DELETE") == [f"/connected_accounts/{instagram}"]
    assert _flow_cookies(browser) == []


def test_an_unclaimed_account_of_the_other_app_is_deleted(instagram_connect):
    """The Instagram flow's cookie is gone and only a Facebook flow is open
    when the Instagram session returns: the activated Instagram account is
    still recognised as this actor's and deleted."""
    db, user, composio = instagram_connect
    browser = Browser()
    _begin(db, user, browser, app_id="instagram")
    (instagram,) = composio.created
    for name in _flow_cookies(browser):
        browser.cookies.pop(name)
    _begin(db, user, browser)

    response = browser.get(db, _verify_url(composio, instagram))

    assert response.status_code == 400
    assert _grants(db) == []
    assert composio.paths("DELETE") == [f"/connected_accounts/{instagram}"]


@pytest.mark.parametrize(
    "override",
    [
        {"toolkit": {"slug": "facebook"}},
        {"auth_config": {"id": AUTH_CONFIG_ID}},
    ],
    ids=["facebook-toolkit", "facebook-auth-config"],
)
def test_an_instagram_flow_refuses_a_facebook_account(instagram_connect, override):
    """Each app checks its own toolkit and auth config, so a Facebook account
    is never bound as an Instagram grant."""
    db, user, composio = instagram_connect
    browser, _ = _connect_until_verification(db, user, app_id="instagram")
    composio.account_overrides = {**composio.account_overrides, **override}

    response = browser.get(db, _verify_url(composio))

    assert response.status_code == 400
    assert _grants(db) == []
    assert composio.paths("DELETE") == [f"/connected_accounts/{composio.created[0]}"]
