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
import threading
from datetime import datetime, timedelta, timezone
from http.cookies import SimpleCookie
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from xagent.core.utils.encryption import encrypt_value
from xagent.web import composio_apps, composio_auth
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
        # A single-account lookup's own answer, by connected account id.
        self.account_lookup: dict[str, tuple[int, Any]] = {}
        # Runs before a single-account lookup answers, to change the row first.
        self.on_lookup: Any = None
        self.me_response: tuple[int, Any] | None = None
        self.raise_on: str | None = None
        # Pages GET /connected_accounts returns, by cursor ("" is the first).
        # cursor ("" for the first page; "<auth config>:<cursor>" for an auth
        # config other than Facebook's) -> page, or an int status to fail it
        self.listing: dict[str, dict[str, Any] | int] = {}
        self.listing_status = 200

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
        if method == "GET" and path.startswith("/connected_accounts?"):
            query = {k: v[0] for k, v in parse_qs(urlsplit(path).query).items()}
            config, cursor = query["auth_config_ids"], query.get("cursor", "")
            key = cursor if config == AUTH_CONFIG_ID else f"{config}:{cursor}"
            page = self.listing.get(key, {"items": []})
            status = self.listing_status if isinstance(page, dict) else page
            if status != 200:
                return _Response(status, {"error": {"message": "x"}})
            return _Response(200, page)
        if method == "GET" and path.startswith("/connected_accounts/"):
            account_id = path.rsplit("/", 1)[1]
            if self.on_lookup is not None:
                self.on_lookup(account_id)
            if account_id in self.account_lookup:
                status, payload = self.account_lookup[account_id]
                return _Response(status, payload)
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
    # The Meta callback, unless a test points the shared callback elsewhere.
    monkeypatch.delenv("XAGENT_COMPOSIO_CALLBACK_URL", raising=False)
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


def test_start_proceeds_while_connector_oauth_routing_is_on(
    db_session, composio_env, composio, monkeypatch
):
    """The start no longer refuses while xagent-saas's switch is on: routing
    Composio's trips per connection is left to xagent-saas and the edge
    (xagent-saas#2461), and a trip that reaches the wrong region binds
    nothing (test_composio_cross_region.py)."""
    monkeypatch.setenv("XAGENT_TOBY_CONNECTOR_OAUTH_ROUTE_ENABLED", "true")
    db, user = db_session

    response = _start(db, user, Browser())

    assert response.status_code in (302, 307)
    assert len(response.headers.getlist("set-cookie")) == 1
    assert db.query(ActorOAuthFlowState).count() == 1


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
    assert "already been used" in response.body.decode()
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


def test_a_session_whose_flow_cookie_expired_is_left_to_the_reconciler(
    db_session, composio_env, composio
):
    """A's cookie expired while the member was on Facebook; B is still open.
    Composio activates A (same actor, same user id), which no open flow
    claims: nothing is bound or deleted, and B still completes."""
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
    assert composio.paths("DELETE") == []
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


def test_a_reloaded_begin_deletes_nothing_and_the_current_link_still_completes(
    db_session, composio_env, composio
):
    """The original window may be completing the replaced link while the
    reload runs, so the reload deletes nothing; a late completion of the old
    link is not bound and is left to the reconciler."""
    db, user = db_session
    browser = Browser()
    start = _start(db, user, browser)
    assert browser.get(db, start.headers["location"]).status_code == 302
    assert browser.get(db, start.headers["location"]).status_code == 302
    replaced, current = composio.created

    assert browser.get(db, _verify_url(composio, replaced)).status_code == 400
    assert composio.paths("DELETE") == []
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


@pytest.mark.parametrize("flow_end", ["expired", "spent-by-another-request"])
def test_completion_crossing_the_deadline_never_binds_or_deletes(
    db_session, composio_env, composio, monkeypatch, flow_end
):
    """The flow ends while complete_auth is in flight: either it expires (no
    grant may be stored after that) or another request spends and binds it
    (its account must survive)."""
    db, user = db_session
    browser, _ = _connect_until_verification(db, user)
    account = composio.created[0]
    complete = composio_auth.complete_auth

    def complete_as_flow_ends(*args, **kwargs):
        try:
            return complete(*args, **kwargs)
        finally:
            if flow_end == "expired":
                db.query(ActorOAuthFlowState).update(
                    {
                        ActorOAuthFlowState.expires_at: datetime.now(timezone.utc)
                        - timedelta(seconds=1)
                    }
                )
                db.commit()
            else:
                db.query(ActorOAuthFlowState).delete()
                _store_composio_grant(db, user, account_id=account)

    monkeypatch.setattr(composio_auth, "complete_auth", complete_as_flow_ends)
    response = browser.get(db, _verify_url(composio, account))

    assert response.status_code == 400
    assert composio.paths("DELETE") == []
    if flow_end == "expired":
        assert _grants(db) == []
    else:
        assert _grants(db)[0].access_token == account


@pytest.mark.parametrize("loss", ["timeout", "no-id"])
def test_a_lost_answer_spends_and_deletes_nothing(
    db_session, composio_env, composio, loss
):
    """Composio may have activated the account; which one cannot be told, so
    nothing is spent or deleted and the reconciler handles an unbound one."""
    db, user = db_session
    browser, _ = _connect_until_verification(db, user)
    if loss == "timeout":
        composio.lose_completion = True
    else:
        composio.complete_without_id = True

    response = browser.get(db, _verify_url(composio))

    assert response.status_code == 503
    assert _grants(db) == []
    assert composio.paths("DELETE") == []
    assert db.query(ActorOAuthFlowState).count() == 1


def test_a_lost_answer_never_touches_a_flow_another_window_just_completed(
    db_session, composio_env, composio, monkeypatch
):
    """Window A has completed flow 1 but not spent it yet when window B's
    complete_auth for flow 2 loses its answer. A still binds flow 1."""
    db, user = db_session
    browser = Browser()
    _begin(db, user, browser)
    _begin(db, user, browser)
    first, second = composio.created
    composio.completed.add(first)  # A's complete_auth already activated it
    composio.lose_completion = True

    assert browser.get(db, _verify_url(composio, second)).status_code == 503

    assert composio.paths("DELETE") == []
    assert db.query(ActorOAuthFlowState).count() == 2
    composio.lose_completion = False
    composio.completed.discard(first)
    assert browser.get(db, _verify_url(composio, first)).status_code == 200
    assert _grants(db)[0].access_token == first


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
    # The unfinished account is left to the reconciler.
    assert composio.paths("DELETE") == []


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


def test_a_return_without_verification_tells_operators_and_ends_the_flow(
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
    assert composio.paths("DELETE") == []


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

    assert response.status_code == 503
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
    refresh_token: str | None = None,
    expires_at: datetime | None = None,
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
            refresh_token=refresh_token,
            expires_at=expires_at,
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


def test_an_instagram_only_deployment_leaves_an_unclaimed_account_to_the_reconciler(
    instagram_only,
):
    """The first flow's cookie is gone when its session returns: the account
    is not bound, and nothing is deleted."""
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
    assert composio.paths("DELETE") == []


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
def test_a_denied_instagram_flow_ends_even_without_its_config(
    instagram_connect, monkeypatch, config_removed
):
    """The flow ends even when Instagram's auth config went away after it
    began, and nothing is deleted here. The reconciler lists only auth
    configs that are still set, so with the config removed the unfinished
    account stays in Composio (see the retirement steps in example.env)."""
    db, user, composio = instagram_connect
    browser, _ = _connect_until_verification(db, user, app_id="instagram")
    if config_removed:
        monkeypatch.delenv("COMPOSIO_INSTAGRAM_AUTH_CONFIG_ID")

    response = browser.get(db, _denied_url(composio.created[0]))

    assert response.status_code == 400
    assert "Permission was not granted" in response.body.decode()
    assert _grants(db) == []
    assert _flow_cookies(browser) == []
    assert composio.paths("DELETE") == []


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


def test_an_unclaimed_account_of_the_other_app_is_not_bound(instagram_connect):
    """The Instagram flow's cookie is gone and only a Facebook flow is open
    when the Instagram session returns: the Facebook flow does not take the
    Instagram account, and nothing is deleted."""
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
    assert composio.paths("DELETE") == []


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


# --- Native callbacks and configuration -----------------------------------------


def test_a_native_state_only_meta_callback_keeps_its_native_answer(
    db_session, composio_env, composio, monkeypatch
):
    """Only a state carrying the Composio marker is a begin."""
    db, user = db_session
    monkeypatch.setenv(composio_auth.FACEBOOK_AUTH_BACKEND_ENV, "native")
    browser = Browser()
    native = _start(db, user, browser)
    state = parse_qs(urlsplit(native.headers["location"]).query)["state"][0]

    response = browser.get(db, composio_auth.begin_location(CALLBACK_URL, state))

    assert response.status_code == 400
    assert "Missing code or state" in response.body.decode()
    assert composio.calls == []


def test_a_denial_shaped_callback_is_native_while_the_switch_is_off(
    db_session, composio_env, composio, monkeypatch
):
    db, _ = db_session
    monkeypatch.setenv(composio_auth.FACEBOOK_AUTH_BACKEND_ENV, "native")

    response = Browser().get(
        db, f"{CALLBACK_URL}?status=failed&connected_account_id=ca_x"
    )

    assert response.status_code == 400
    assert "Permission was not granted" not in response.body.decode()
    assert composio.calls == []


@pytest.mark.parametrize("app_id", ["facebook", "instagram"])
def test_an_unknown_backend_value_warns_and_stays_native(monkeypatch, caplog, app_id):
    app = composio_auth.COMPOSIO_APPS[app_id]
    monkeypatch.setenv(app.backend_env, "composio-x")
    monkeypatch.setattr(composio_apps, "_WARNED_BACKENDS", set())

    assert composio_auth.composio_app("meta", app_id) is None
    assert composio_auth.composio_app("meta", app_id) is None

    warnings = [r for r in caplog.records if "composio-x" in r.getMessage()]
    assert len(warnings) == 1
    assert app.backend_env in warnings[0].getMessage()


def test_a_grant_that_cannot_be_stored_deletes_its_account(
    db_session, composio_env, composio, monkeypatch
):
    """This request spent the flow, so the account is its own to delete."""
    db, user = db_session
    browser, _ = _connect_until_verification(db, user)

    def fail_to_store(*args, **kwargs):
        raise RuntimeError("database is gone")

    monkeypatch.setattr(composio_auth, "_store_grant", fail_to_store)

    response = browser.get(db, _verify_url(composio))

    assert response.status_code == 500
    assert _grants(db) == []
    assert composio.paths("DELETE") == [f"/connected_accounts/{composio.created[0]}"]


# --- Reconciler ---------------------------------------------------------------------

OLD = "2026-01-01T00:00:00.000Z"
NOW = datetime(2026, 1, 1, 2, tzinfo=timezone.utc)  # two hours later


def _listed(
    account_id: str,
    *,
    user_id: str | None = "xagent-" + "a" * 64,
    created_at: str = OLD,
    auth_config_id: str = AUTH_CONFIG_ID,
) -> dict[str, Any]:
    return {
        "id": account_id,
        "user_id": user_id,
        "status": "ACTIVE",
        "created_at": created_at,
        "auth_config": {"id": auth_config_id},
        "toolkit": {"slug": "facebook"},
    }


def test_the_reconciler_deletes_only_old_unbound_xagent_accounts(
    db_session, composio_env, composio
):
    db, user = db_session
    _store_composio_grant(db, user, account_id="ca_bound")
    composio.listing[""] = {
        "items": [
            _listed("ca_orphan"),
            _listed("ca_bound"),
            _listed("ca_young", created_at="2026-01-01T01:50:00Z"),
            _listed("ca_dashboard", user_id="pg-test-user"),
            _listed("ca_no_user_id", user_id=None),
            _listed("ca_other_config", auth_config_id="ac_other"),
        ]
    }

    assert composio_auth.reconcile_orphan_accounts(db, now=NOW) == 1

    assert composio.paths("DELETE") == ["/connected_accounts/ca_orphan"]


@pytest.mark.parametrize("failure", ["status", "exception"])
def test_a_failed_listing_deletes_nothing(db_session, composio_env, composio, failure):
    db, _ = db_session
    composio.listing[""] = {"items": [_listed("ca_orphan")]}
    if failure == "status":
        composio.listing_status = 500
    else:
        composio.raise_on = (
            "/connected_accounts?auth_config_ids=ac_facebook_test&limit=100"
        )

    assert composio_auth.reconcile_orphan_accounts(db, now=NOW) == 0

    assert composio.paths("DELETE") == []


def test_a_listing_failing_on_a_later_page_deletes_what_was_collected(
    db_session, composio_env, composio
):
    db, _ = db_session
    composio.listing[""] = {"items": [_listed("ca_page_1")], "next_cursor": "c2"}
    composio.listing["c2"] = 500

    assert composio_auth.reconcile_orphan_accounts(db, now=NOW) == 1

    assert composio.paths("DELETE") == ["/connected_accounts/ca_page_1"]


def test_the_reconciler_has_no_page_limit_and_stops_on_a_repeated_cursor(
    db_session, composio_env, composio
):
    db, _ = db_session
    pages = 60
    for page in range(pages):
        composio.listing["" if page == 0 else f"c{page}"] = {
            "items": [_listed(f"ca_page_{page}")],
            "next_cursor": f"c{page + 1}" if page + 1 < pages else "c1",
        }

    assert composio_auth.reconcile_orphan_accounts(db, now=NOW) == pages

    listed = [p for p in composio.paths("GET") if p.startswith("/connected_accounts?")]
    assert len(listed) == pages


def test_an_undecryptable_grant_stops_the_pass(db_session, composio_env, composio):
    db, user = db_session
    _store_composio_grant(db, user, account_id="ca_bound")
    # A token encrypted under another key, as after a key rotation.
    foreign = Fernet(Fernet.generate_key()).encrypt(b"ca_bound").decode()
    db.execute(text("UPDATE user_oauth SET access_token = :t"), {"t": foreign})
    db.commit()
    composio.listing[""] = {"items": [_listed("ca_bound"), _listed("ca_orphan")]}

    assert composio_auth.reconcile_orphan_accounts(db, now=NOW) == 0

    assert composio.paths("DELETE") == []


def test_a_stop_signal_ends_the_pass_between_deletes(
    db_session, composio_env, composio, monkeypatch
):
    db, _ = db_session
    composio.listing[""] = {"items": [_listed("ca_first"), _listed("ca_second")]}
    stop = threading.Event()
    delete = composio_auth.delete_account

    def delete_then_stop(api_key: str, account_id: str) -> bool:
        stop.set()
        return delete(api_key, account_id)

    monkeypatch.setattr(composio_auth, "delete_account", delete_then_stop)

    assert composio_auth.reconcile_orphan_accounts(db, now=NOW, stop_event=stop) == 1

    assert composio.paths("DELETE") == ["/connected_accounts/ca_first"]


def test_a_stop_signal_before_the_pass_lists_nothing(
    db_session, composio_env, composio
):
    db, _ = db_session
    stop = threading.Event()
    stop.set()

    assert composio_auth.reconcile_orphan_accounts(db, now=NOW, stop_event=stop) == 0

    assert composio.calls == []


# --- Reconciler: grants whose connected account Composio no longer has ---------------

MISSING_ACCOUNT = {
    "error": {
        "slug": "ConnectedAccount_ResourceNotFound",
        "message": "Connected account not found",
    }
}


def _lookups(composio: FakeComposio) -> list[str]:
    """The single-account lookups, not the auth-config listings."""
    return [
        path
        for path in composio.paths("GET")
        if path.startswith("/connected_accounts/")
    ]


def test_a_grant_whose_account_is_gone_is_tombstoned(
    db_session, composio_env, composio, monkeypatch
):
    from xagent.web.api import mcp as mcp_api

    db, user = db_session
    monkeypatch.setattr(composio_auth, "_missing_account_cursor", 0)
    _store_composio_grant(
        db,
        user,
        account_id="ca_gone",
        refresh_token="refresh",
        expires_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
    )
    row = _grants(db)[0]
    assert mcp_api._oauth_account_can_connect(row) is True
    assert mcp_api._build_oauth_account_lookup([row]) == {"facebook": row}
    composio.account_lookup["ca_gone"] = (404, MISSING_ACCOUNT)

    assert composio_auth.tombstone_missing_composio_accounts(db) == 1

    row = _grants(db)[0]
    assert row.id is not None
    assert row.resource_owner_key == OWNER
    assert row.access_token == ""
    assert row.refresh_token is None
    assert row.expires_at is None
    assert mcp_api._oauth_account_can_connect(row) is False
    assert mcp_api._build_oauth_account_lookup([row]) == {}


def test_a_reconnect_between_lookup_and_write_is_kept(
    db_session, composio_env, composio, monkeypatch
):
    db, user = db_session
    monkeypatch.setattr(composio_auth, "_missing_account_cursor", 0)
    _store_composio_grant(db, user, account_id="ca_old")
    row_id = _grants(db)[0].id
    composio.account_lookup["ca_old"] = (404, MISSING_ACCOUNT)

    def reconnect(account_id: str) -> None:
        db.execute(
            text("UPDATE user_oauth SET access_token = :t WHERE id = :i"),
            {"t": encrypt_value("ca_new"), "i": row_id},
        )
        db.commit()

    composio.on_lookup = reconnect

    assert composio_auth.tombstone_missing_composio_accounts(db) == 0

    assert _grants(db)[0].access_token == "ca_new"


@pytest.mark.parametrize(
    "answer",
    [
        (404, {"error": {"slug": "ConnectedAccount_NotFound"}}),
        (404, None),
        (401, {"error": {"slug": "ConnectedAccount_ResourceNotFound"}}),
        (500, {"error": {"slug": "ConnectedAccount_ResourceNotFound"}}),
        (200, {"id": "ca_bound", "status": "EXPIRED"}),
    ],
)
def test_any_other_lookup_answer_leaves_the_grant_alone(
    db_session, composio_env, composio, monkeypatch, answer
):
    db, user = db_session
    monkeypatch.setattr(composio_auth, "_missing_account_cursor", 0)
    _store_composio_grant(db, user, account_id="ca_bound")
    composio.account_lookup["ca_bound"] = answer

    assert composio_auth.tombstone_missing_composio_accounts(db) == 0

    assert _lookups(composio) == ["/connected_accounts/ca_bound"]
    assert _grants(db)[0].access_token == "ca_bound"


def test_an_unreachable_composio_ends_the_pass(
    db_session, composio_env, composio, monkeypatch
):
    db, user = db_session
    monkeypatch.setattr(composio_auth, "_missing_account_cursor", 0)
    _store_composio_grant(db, user, account_id="ca_first")
    _store_composio_grant(db, user, account_id="ca_second", facebook_id="102030405061")
    composio.raise_on = "/connected_accounts/ca_first"

    assert composio_auth.tombstone_missing_composio_accounts(db) == 0

    assert _lookups(composio) == ["/connected_accounts/ca_first"]
    assert [grant.access_token for grant in _grants(db)] == [
        "ca_first",
        "ca_second",
    ]


def test_an_undecryptable_grant_is_skipped_and_the_pass_continues(
    db_session, composio_env, composio, monkeypatch
):
    db, user = db_session
    monkeypatch.setattr(composio_auth, "_missing_account_cursor", 0)
    _store_composio_grant(db, user, account_id="ca_foreign")
    _store_composio_grant(db, user, account_id="ca_gone", facebook_id="102030405061")
    foreign_id = _grants(db)[0].id
    foreign = Fernet(Fernet.generate_key()).encrypt(b"ca_foreign").decode()
    db.execute(
        text("UPDATE user_oauth SET access_token = :t WHERE id = :i"),
        {"t": foreign, "i": foreign_id},
    )
    db.commit()
    composio.account_lookup["ca_gone"] = (404, MISSING_ACCOUNT)

    assert composio_auth.tombstone_missing_composio_accounts(db) == 1

    assert _lookups(composio) == ["/connected_accounts/ca_gone"]
    assert (
        db.execute(
            text("SELECT access_token FROM user_oauth WHERE id = :i"),
            {"i": foreign_id},
        ).scalar()
        == foreign
    )


def test_a_stop_signal_ends_the_missing_account_pass_between_lookups(
    db_session, composio_env, composio, monkeypatch
):
    db, user = db_session
    monkeypatch.setattr(composio_auth, "_missing_account_cursor", 0)
    _store_composio_grant(db, user, account_id="ca_first")
    _store_composio_grant(db, user, account_id="ca_second", facebook_id="102030405061")
    stop = threading.Event()
    composio.on_lookup = lambda account_id: stop.set()

    assert composio_auth.tombstone_missing_composio_accounts(db, stop_event=stop) == 0

    assert _lookups(composio) == ["/connected_accounts/ca_first"]


def test_a_stop_signal_before_the_missing_account_pass_looks_nothing_up(
    db_session, composio_env, composio, monkeypatch
):
    db, user = db_session
    monkeypatch.setattr(composio_auth, "_missing_account_cursor", 0)
    _store_composio_grant(db, user, account_id="ca_gone")
    stop = threading.Event()
    stop.set()

    assert composio_auth.tombstone_missing_composio_accounts(db, stop_event=stop) == 0

    assert _lookups(composio) == []


def test_one_pass_is_capped_and_the_next_pass_continues(
    db_session, composio_env, composio, monkeypatch
):
    db, user = db_session
    monkeypatch.setattr(composio_auth, "_missing_account_cursor", 0)
    monkeypatch.setattr(composio_auth, "_MISSING_ACCOUNT_LOOKUP_LIMIT", 1)
    _store_composio_grant(db, user, account_id="ca_first")
    _store_composio_grant(db, user, account_id="ca_second", facebook_id="102030405061")

    assert composio_auth.tombstone_missing_composio_accounts(db) == 0

    assert _lookups(composio) == ["/connected_accounts/ca_first"]

    assert composio_auth.tombstone_missing_composio_accounts(db) == 0

    assert _lookups(composio) == [
        "/connected_accounts/ca_first",
        "/connected_accounts/ca_second",
    ]


def test_the_cursor_wraps_after_the_last_row(
    db_session, composio_env, composio, monkeypatch
):
    db, user = db_session
    monkeypatch.setattr(composio_auth, "_missing_account_cursor", 0)
    monkeypatch.setattr(composio_auth, "_MISSING_ACCOUNT_LOOKUP_LIMIT", 1)
    _store_composio_grant(db, user, account_id="ca_first")
    _store_composio_grant(db, user, account_id="ca_second", facebook_id="102030405061")

    for _ in range(3):
        assert composio_auth.tombstone_missing_composio_accounts(db) == 0

    assert _lookups(composio) == [
        "/connected_accounts/ca_first",
        "/connected_accounts/ca_second",
        "/connected_accounts/ca_first",
    ]


def test_rows_without_a_usable_lookup_do_not_count_toward_the_cap(
    db_session, composio_env, composio, monkeypatch
):
    db, user = db_session
    monkeypatch.setattr(composio_auth, "_missing_account_cursor", 0)
    monkeypatch.setattr(composio_auth, "_MISSING_ACCOUNT_LOOKUP_LIMIT", 1)
    _store_composio_grant(db, user, account_id="ca bad/id")
    _store_composio_grant(
        db,
        user,
        account_id="ca_other_app",
        provider="instagram",
        facebook_id="102030405061",
    )
    _store_composio_grant(db, user, account_id="ca_gone", facebook_id="102030405062")
    composio.account_lookup["ca_gone"] = (404, MISSING_ACCOUNT)

    assert composio_auth.tombstone_missing_composio_accounts(db) == 1

    assert _lookups(composio) == ["/connected_accounts/ca_gone"]
    assert [grant.access_token for grant in _grants(db)] == [
        "ca bad/id",
        "ca_other_app",
        "",
    ]


# --- Missing-account circuit breaker --------------------------------------------------


def _warning_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == composio_auth.__name__ and record.levelno == logging.WARNING
    ]


def test_most_lookups_answering_missing_write_nothing(
    db_session, composio_env, composio, monkeypatch, caplog
):
    db, user = db_session
    monkeypatch.setattr(composio_auth, "_missing_account_cursor", 0)
    for index, account_id in enumerate(("ca_1", "ca_2", "ca_3")):
        _store_composio_grant(
            db, user, account_id=account_id, facebook_id=f"10203040506{index}"
        )
        composio.account_lookup[account_id] = (404, MISSING_ACCOUNT)
    _store_composio_grant(db, user, account_id="ca_4", facebook_id="102030405063")
    composio.account_lookup["ca_4"] = (200, {"id": "ca_4"})

    with caplog.at_level(logging.WARNING, logger=composio_auth.__name__):
        assert composio_auth.tombstone_missing_composio_accounts(db) == 0

    assert [grant.access_token for grant in _grants(db)] == [
        "ca_1",
        "ca_2",
        "ca_3",
        "ca_4",
    ]
    warnings = _warning_messages(caplog)
    assert len(warnings) == 1
    assert "3 of 4" in warnings[0]


def test_below_the_breaker_minimum_missing_rows_are_tombstoned(
    db_session, composio_env, composio, monkeypatch
):
    db, user = db_session
    monkeypatch.setattr(composio_auth, "_missing_account_cursor", 0)
    for index, account_id in enumerate(("ca_1", "ca_2")):
        _store_composio_grant(
            db, user, account_id=account_id, facebook_id=f"10203040506{index}"
        )
        composio.account_lookup[account_id] = (404, MISSING_ACCOUNT)

    assert composio_auth.tombstone_missing_composio_accounts(db) == 2

    assert [grant.access_token for grant in _grants(db)] == ["", ""]


def test_a_single_missing_answer_among_three_is_tombstoned(
    db_session, composio_env, composio, monkeypatch
):
    db, user = db_session
    monkeypatch.setattr(composio_auth, "_missing_account_cursor", 0)
    for index, account_id in enumerate(("ca_1", "ca_2", "ca_3")):
        _store_composio_grant(
            db, user, account_id=account_id, facebook_id=f"10203040506{index}"
        )
    composio.account_lookup["ca_1"] = (404, MISSING_ACCOUNT)

    assert composio_auth.tombstone_missing_composio_accounts(db) == 1

    assert [grant.access_token for grant in _grants(db)] == ["", "ca_2", "ca_3"]


def test_another_404_slug_leaves_the_grant_and_logs_the_slug(
    db_session, composio_env, composio, monkeypatch, caplog
):
    db, user = db_session
    monkeypatch.setattr(composio_auth, "_missing_account_cursor", 0)
    _store_composio_grant(db, user, account_id="ca_keep")
    composio.account_lookup["ca_keep"] = (
        404,
        {"error": {"slug": "ConnectedAccount_NotFound", "message": "gone"}},
    )

    with caplog.at_level(logging.WARNING, logger=composio_auth.__name__):
        assert composio_auth.tombstone_missing_composio_accounts(db) == 0

    assert _grants(db)[0].access_token == "ca_keep"
    messages = _warning_messages(caplog)
    assert any("ConnectedAccount_NotFound" in message for message in messages)
    assert all("ca_keep" not in message for message in messages)


@pytest.mark.parametrize(
    ("status", "second_pass_first"),
    [(401, "ca_first"), (403, "ca_first"), (429, "ca_first"), (503, "ca_second")],
)
def test_a_refused_or_unavailable_lookup_ends_the_pass_before_writing(
    db_session, composio_env, composio, monkeypatch, status, second_pass_first
):
    db, user = db_session
    monkeypatch.setattr(composio_auth, "_missing_account_cursor", 0)
    _store_composio_grant(db, user, account_id="ca_first")
    _store_composio_grant(db, user, account_id="ca_second", facebook_id="102030405061")
    composio.account_lookup["ca_first"] = (status, {"error": {"message": "x"}})

    assert composio_auth.tombstone_missing_composio_accounts(db) == 0

    assert _lookups(composio) == ["/connected_accounts/ca_first"]
    assert [grant.access_token for grant in _grants(db)] == [
        "ca_first",
        "ca_second",
    ]

    # 5xx is skipped next pass; a refused key retries the same row.
    composio.account_lookup["ca_first"] = (200, {"id": "ca_first"})
    assert composio_auth.tombstone_missing_composio_accounts(db) == 0
    assert _lookups(composio)[1] == f"/connected_accounts/{second_pass_first}"


def test_a_row_deleted_between_lookup_and_write_is_not_an_error(
    db_session, composio_env, composio, monkeypatch
):
    db, user = db_session
    monkeypatch.setattr(composio_auth, "_missing_account_cursor", 0)
    _store_composio_grant(db, user, account_id="ca_old")
    row_id = _grants(db)[0].id
    composio.account_lookup["ca_old"] = (404, MISSING_ACCOUNT)

    def disconnect(account_id: str) -> None:
        db.execute(text("DELETE FROM user_oauth WHERE id = :i"), {"i": row_id})
        db.commit()

    composio.on_lookup = disconnect

    assert composio_auth.tombstone_missing_composio_accounts(db) == 0

    assert _grants(db) == []


def test_a_reconnect_replacing_the_row_between_lookup_and_write_is_kept(
    db_session, composio_env, composio, monkeypatch
):
    db, user = db_session
    monkeypatch.setattr(composio_auth, "_missing_account_cursor", 0)
    _store_composio_grant(db, user, account_id="ca_old")
    row_id = _grants(db)[0].id
    composio.account_lookup["ca_old"] = (404, MISSING_ACCOUNT)

    def reconnect(account_id: str) -> None:
        db.execute(text("DELETE FROM user_oauth WHERE id = :i"), {"i": row_id})
        db.commit()
        _store_composio_grant(db, user, account_id="ca_new")

    composio.on_lookup = reconnect

    assert composio_auth.tombstone_missing_composio_accounts(db) == 0

    grants = _grants(db)
    assert [grant.access_token for grant in grants] == ["ca_new"]


def test_the_orphan_pass_leaves_missing_accounts_to_their_own_loop(
    db_session, composio_env, composio, monkeypatch
):
    db, user = db_session
    monkeypatch.setattr(composio_auth, "_missing_account_cursor", 0)
    _store_composio_grant(db, user, account_id="ca_gone")
    composio.account_lookup["ca_gone"] = (404, MISSING_ACCOUNT)
    composio.listing[""] = {"items": []}

    composio_auth.reconcile_orphan_accounts_isolated(session_factory=lambda: db)

    assert _lookups(composio) == []
    assert _grants(db)[0].access_token == "ca_gone"


def test_the_isolated_missing_account_pass_tombstones_missing_accounts(
    db_session, composio_env, composio, monkeypatch
):
    db, user = db_session
    monkeypatch.setattr(composio_auth, "_missing_account_cursor", 0)
    _store_composio_grant(db, user, account_id="ca_gone")
    composio.account_lookup["ca_gone"] = (404, MISSING_ACCOUNT)

    tombstoned = composio_auth.tombstone_missing_composio_accounts_isolated(
        session_factory=lambda: db
    )

    assert tombstoned == 1
    assert _grants(db)[0].access_token == ""


def test_the_missing_account_loop_needs_a_configured_app(monkeypatch):
    from xagent.web.app import start_composio_missing_account_task

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv(composio_auth.COMPOSIO_API_KEY_ENV, raising=False)
    monkeypatch.delenv(
        "XAGENT_COMPOSIO_MISSING_ACCOUNT_INTERVAL_SECONDS", raising=False
    )
    app = SimpleNamespace(state=SimpleNamespace())

    assert start_composio_missing_account_task(app) is None


def test_the_missing_account_loop_is_off_at_zero(composio_env, monkeypatch):
    from xagent.web.app import start_composio_missing_account_task

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("XAGENT_COMPOSIO_MISSING_ACCOUNT_INTERVAL_SECONDS", "0")
    app = SimpleNamespace(state=SimpleNamespace())

    assert start_composio_missing_account_task(app) is None


async def test_the_missing_account_loop_starts_with_its_variable_unset(
    composio_env, monkeypatch
):
    import asyncio

    from xagent.web.app import (
        start_composio_missing_account_task,
        stop_composio_missing_account_task,
    )

    started = threading.Event()

    def pass_once(*, stop_event: threading.Event) -> int:
        started.set()
        return 0

    monkeypatch.setattr(
        composio_auth, "tombstone_missing_composio_accounts_isolated", pass_once
    )
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv(
        "XAGENT_COMPOSIO_MISSING_ACCOUNT_INTERVAL_SECONDS", raising=False
    )
    app = SimpleNamespace(state=SimpleNamespace())

    task = start_composio_missing_account_task(app)
    assert task is not None
    assert await asyncio.to_thread(started.wait, 5)
    await stop_composio_missing_account_task(app)

    assert task.done()


def test_the_reconcile_loop_is_off_unless_its_interval_is_set(
    composio_env, monkeypatch
):
    from xagent.web.app import start_composio_reconcile_task

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("XAGENT_COMPOSIO_RECONCILE_INTERVAL_SECONDS", raising=False)
    app = SimpleNamespace(state=SimpleNamespace())

    assert start_composio_reconcile_task(app) is None


async def test_stopping_the_reconcile_loop_reaches_the_running_pass(
    composio_env, monkeypatch
):
    import asyncio

    from xagent.web.app import (
        start_composio_reconcile_task,
        stop_composio_reconcile_task,
    )

    started = threading.Event()
    finished = threading.Event()

    def long_pass(*, stop_event: threading.Event) -> int:
        started.set()
        stop_event.wait(timeout=5)
        finished.set()
        return 0

    monkeypatch.setattr(composio_auth, "reconcile_orphan_accounts_isolated", long_pass)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("XAGENT_COMPOSIO_RECONCILE_INTERVAL_SECONDS", "300")
    app = SimpleNamespace(state=SimpleNamespace())

    task = start_composio_reconcile_task(app)
    assert task is not None
    await asyncio.to_thread(started.wait, 5)
    await stop_composio_reconcile_task(app)

    assert task.done()
    assert await asyncio.to_thread(finished.wait, 1)


def test_the_reconciler_covers_every_configured_app(
    db_session, composio_env, composio, monkeypatch
):
    """Each app's auth config is listed; a grant of either app is a reference."""
    db, user = db_session
    monkeypatch.setenv("COMPOSIO_INSTAGRAM_AUTH_CONFIG_ID", INSTAGRAM_AUTH_CONFIG_ID)
    _store_composio_grant(db, user, account_id="ca_ig_bound", provider="instagram")
    composio.listing[""] = {"items": [_listed("ca_fb_orphan")]}
    composio.listing[f"{INSTAGRAM_AUTH_CONFIG_ID}:"] = {
        "items": [
            _listed("ca_ig_orphan", auth_config_id=INSTAGRAM_AUTH_CONFIG_ID),
            _listed("ca_ig_bound", auth_config_id=INSTAGRAM_AUTH_CONFIG_ID),
            # Listed under Instagram's config but naming Facebook's: skipped.
            _listed("ca_mislabelled"),
        ]
    }

    assert composio_auth.reconcile_orphan_accounts(db, now=NOW) == 2

    assert sorted(composio.paths("DELETE")) == [
        "/connected_accounts/ca_fb_orphan",
        "/connected_accounts/ca_ig_orphan",
    ]


def test_the_reconciler_does_nothing_without_composio_configuration(
    db_session, composio_env, composio, monkeypatch
):
    db, _ = db_session
    monkeypatch.delenv(composio_auth.COMPOSIO_API_KEY_ENV)

    assert composio_auth.reconcile_orphan_accounts(db, now=NOW) == 0

    assert composio.calls == []
