"""Gmail and Google Drive connect through Composio-hosted auth (#3649).

Composio's Google app already holds Google's restricted-scope approval, so a
Gmail or Drive connect started through it gets full access while our own
review gate (``XAGENT_GOOGLE_RESTRICTED_SCOPES_ENABLED``) stays closed. Native
Google connects keep obeying that gate. Composio is faked here; no request
leaves the process.
"""

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from sqlalchemy.orm import Session

from xagent.core.utils.encryption import encrypt_value
from xagent.web import composio_auth, mcp_apps
from xagent.web.api import auth as auth_api
from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app
from xagent.web.mcp_apps import ensure_builtin_oauth_server_definition
from xagent.web.models.actor_oauth_flow import ActorOAuthFlowState
from xagent.web.models.database import (
    _initialize_database_schema,
    get_engine,
    init_db,
)
from xagent.web.models.mcp import UserMCPServer
from xagent.web.models.oauth_provider import OAuthProvider
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.services.google_oauth_grants import (
    resolve_google_shared_oauth_account,
    write_google_shared_oauth_callback,
)
from xagent.web.services.user_oauth import delete_scoped_user_oauth_accounts

from . import test_composio_facebook_connect as facebook_connect
from .test_composio_facebook_connect import (
    OWNER,
    Browser,
    FakeComposio,
    _flow_cookies,
    _grants,
    _meta_provider,
    _Response,
    _verify_url,
)

composio_env = facebook_connect.composio_env
db_session = facebook_connect.db_session

RESTRICTED_FLAG = "XAGENT_GOOGLE_RESTRICTED_SCOPES_ENABLED"
USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
GOOGLE_SUBJECT = "110248495921238986420"
AUTH_CONFIGS = {"gmail": "ac_gmail_test", "google-drive": "ac_drive_test"}
TOOLKITS = {"gmail": "gmail", "google-drive": "googledrive"}
SWITCHES = {
    "gmail": ("XAGENT_GMAIL_AUTH_BACKEND", "COMPOSIO_GMAIL_AUTH_CONFIG_ID"),
    "google-drive": (
        "XAGENT_GOOGLE_DRIVE_AUTH_BACKEND",
        "COMPOSIO_GOOGLE_DRIVE_AUTH_CONFIG_ID",
    ),
}
GOOGLE_APPS = tuple(AUTH_CONFIGS)


class GoogleComposio(FakeComposio):
    """Composio answering for Gmail and Drive accounts too."""

    def __init__(self) -> None:
        super().__init__()
        self.google_identity: Any = {"sub": GOOGLE_SUBJECT, "email": "a@example.com"}

    def request(self, *, method, url, headers, json, timeout):  # noqa: A002
        path = url[len(composio_auth.COMPOSIO_API_BASE_URL) :]
        if (
            method == "POST"
            and path == "/tools/execute/proxy"
            and json["endpoint"] == USERINFO_URL
        ):
            self.calls.append((method, path, json))
            return _Response(200, {"data": self.google_identity, "status": 200})
        response = super().request(
            method=method, url=url, headers=headers, json=json, timeout=timeout
        )
        if method == "GET" and path.startswith("/connected_accounts/"):
            config = self.auth_configs.get(path.rsplit("/", 1)[1])
            for app_id, google_config in AUTH_CONFIGS.items():
                if config == google_config:
                    response._payload["toolkit"] = {"slug": TOOLKITS[app_id]}
        return response


@pytest.fixture
def composio(monkeypatch) -> GoogleComposio:
    fake = GoogleComposio()
    monkeypatch.setattr(composio_auth.requests, "request", fake.request)

    def no_provider_call(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a Composio flow must never call Google directly")

    monkeypatch.setattr(auth_api.requests, "post", no_provider_call)
    monkeypatch.setattr(auth_api.requests, "get", no_provider_call)
    return fake


def _switch_on(monkeypatch, app_id: str) -> None:
    backend_env, auth_config_env = SWITCHES[app_id]
    monkeypatch.setenv(backend_env, "composio")
    monkeypatch.setenv(auth_config_env, AUTH_CONFIGS[app_id])


@pytest.fixture
def google(db_session, composio_env, composio, monkeypatch):
    """Both switches and the review gate off; nothing installed yet."""
    for backend_env, auth_config_env in SWITCHES.values():
        monkeypatch.delenv(backend_env, raising=False)
        monkeypatch.delenv(auth_config_env, raising=False)
    monkeypatch.delenv(RESTRICTED_FLAG, raising=False)
    db, user = db_session
    meta = _meta_provider()
    db.add(
        OAuthProvider(
            **{
                name: getattr(meta, name)
                for name in (
                    "provider_name",
                    "client_id",
                    "client_secret",
                    "auth_url",
                    "token_url",
                    "redirect_uri",
                    "userinfo_url",
                    "user_id_path",
                    "email_path",
                    "default_scopes",
                )
            },
            name="Meta",
        )
    )
    db.commit()
    return db, user, composio


def _install(db: Session, user: User, *app_ids: str) -> None:
    """Install the apps' catalog rows as the current environment seeds them,
    and link them to ``user``. Drive is visible, as in production: while the
    gate is closed the startup sync leaves its visibility alone, and the Picker
    migration turned it on."""
    for app_id in app_ids:
        row = get_builtin_public_mcp_app(app_id)
        assert row is not None
        if app_id == "google-drive":
            row = {**row, "is_visible_in_connector": True}
        db.add(PublicMCPApp(**row))
    db.commit()
    for app_id in app_ids:
        server = ensure_builtin_oauth_server_definition(db, app_id=app_id)
        db.add(
            UserMCPServer(
                user_id=int(user.id),
                mcpserver_id=int(server.id),
                is_owner=True,
                is_active=True,
            )
        )
    db.commit()


def _google_provider() -> SimpleNamespace:
    return SimpleNamespace(
        provider_name="google",
        client_id=encrypt_value("google-client-id"),
        client_secret=encrypt_value("google-client-secret"),
        auth_url="https://accounts.google.com/o/oauth2/auth",
        token_url="https://oauth2.googleapis.com/token",
        redirect_uri="https://app.example.com/api/auth/google/callback",
        userinfo_url="https://www.googleapis.com/oauth2/v2/userinfo",
        user_id_path="id",
        email_path="email",
        default_scopes=["openid", "email", "profile"],
    )


def _start_google(db: Session, user: User, browser: Browser, app_id: str) -> Any:
    response = auth_api.start_builtin_oauth_for_resource_owner(
        provider="google",
        app_id=app_id,
        user=user,
        resource_owner_key=OWNER,
        redirect="https://toby.example.com/settings",
        db=db,
        db_provider=_google_provider(),
    )
    db.commit()
    browser.receive(response)
    return response


def _connect(db: Session, user: User, composio: GoogleComposio, app_id: str) -> Any:
    browser = Browser()
    start = _start_google(db, user, browser, app_id)
    assert start.status_code in (302, 307), start.body
    begin = browser.get(db, start.headers["location"])
    assert begin.status_code == 302, begin.body
    return browser.get(db, _verify_url(composio))


# --- Registration ---------------------------------------------------------------


@pytest.mark.parametrize("app_id", GOOGLE_APPS)
def test_each_google_app_follows_only_its_own_switch(google, monkeypatch, app_id):
    other = next(other for other in GOOGLE_APPS if other != app_id)
    assert composio_auth.composio_app("google", app_id) is None

    _switch_on(monkeypatch, app_id)

    app = composio_auth.composio_app("google", app_id)
    assert app is not None and app.app_id == app_id
    assert app.toolkit_slug == TOOLKITS[app_id]
    assert composio_auth.composio_app("google", other) is None
    # Other Google apps (Calendar, Docs...) have no Composio switch.
    assert composio_auth.composio_app("google", "google-calendar") is None
    assert composio_auth.composio_app("meta", app_id) is None


@pytest.mark.parametrize("app_id", GOOGLE_APPS)
def test_a_google_app_connects_and_stores_its_google_subject(
    google, monkeypatch, app_id
):
    db, user, composio = google
    # The review gate is open here; closing it is covered below.
    monkeypatch.setenv(RESTRICTED_FLAG, "true")
    _switch_on(monkeypatch, app_id)
    _install(db, user, app_id)

    response = _connect(db, user, composio, app_id)

    assert response.status_code == 200, response.body
    (link,) = [body for _, path, body in composio.calls if path.endswith("/link")]
    assert link["auth_config_id"] == AUTH_CONFIGS[app_id]
    (identity,) = [
        body for _, path, body in composio.calls if path == "/tools/execute/proxy"
    ]
    assert identity["endpoint"] == USERINFO_URL
    assert identity["parameters"] == []
    (grant,) = _grants(db)
    assert (grant.provider, grant.provider_user_id) == (app_id, GOOGLE_SUBJECT)
    assert grant.token_type == "composio"
    assert grant.resource_owner_key == OWNER
    assert db.query(ActorOAuthFlowState).count() == 0


@pytest.mark.parametrize(
    "identity",
    [{"sub": "not-a-google-subject"}, {"sub": 110248495921238986420}, {}],
    ids=["text", "number", "missing"],
)
def test_a_google_identity_must_be_a_google_subject(google, monkeypatch, identity):
    db, user, composio = google
    monkeypatch.setenv(RESTRICTED_FLAG, "true")
    _switch_on(monkeypatch, "gmail")
    _install(db, user, "gmail")
    composio.google_identity = identity

    response = _connect(db, user, composio, "gmail")

    assert response.status_code == 503
    assert _grants(db) == []


def test_a_google_flow_uses_the_shared_callback_not_googles(google, monkeypatch):
    db, user, composio = google
    monkeypatch.setenv(RESTRICTED_FLAG, "true")
    _switch_on(monkeypatch, "google-drive")
    _install(db, user, "google-drive")
    browser = Browser()

    start = _start_google(db, user, browser, "google-drive")

    assert start.headers["location"].startswith(
        "https://app.example.com/api/auth/meta/callback?state="
    )
    assert composio.calls == []
    assert _flow_cookies(browser) == []


# --- The review gate binds native starts only -------------------------------------


@pytest.mark.parametrize("app_id", GOOGLE_APPS)
def test_a_composio_google_app_connects_while_the_gate_is_closed(
    google, monkeypatch, app_id
):
    db, user, composio = google
    _switch_on(monkeypatch, app_id)
    _install(db, user, app_id)

    response = _connect(db, user, composio, app_id)

    assert response.status_code == 200, response.body
    (grant,) = _grants(db)
    assert (grant.provider, grant.token_type) == (app_id, "composio")


def test_gmail_stays_refused_while_only_drive_goes_through_composio(
    google, monkeypatch
):
    db, user, composio = google
    _switch_on(monkeypatch, "google-drive")

    response = _start_google(db, user, Browser(), "gmail")

    assert response.status_code == 404
    assert "location" not in response.headers
    assert composio.calls == []


def test_a_native_gmail_start_is_refused_while_the_composio_switch_is_on(
    google, monkeypatch
):
    db, user, composio = google
    _switch_on(monkeypatch, "gmail")
    _install(db, user, "gmail")

    # The ordinary (non-actor) start never goes through Composio.
    response = auth_api.generic_oauth_login(
        "google",
        token=auth_api.create_access_token(
            data={
                "sub": user.username,
                "user_id": user.id,
                "session_epoch": user.session_epoch,
            }
        ),
        app_id="gmail",
        db=db,
        db_provider=_google_provider(),
    )

    assert response.status_code == 404
    assert "location" not in response.headers


def test_a_native_gmail_callback_is_refused_while_the_composio_switch_is_on(
    google, monkeypatch
):
    """A native Gmail state that reaches Google's callback, however it was
    minted, still meets the gate there."""
    from tests.shared.oauth_browser_fixture import (
        create_access_token,
        generic_oauth_callback,
    )

    db, user, _ = google
    _switch_on(monkeypatch, "gmail")
    _install(db, user, "gmail")
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "google",
            "app_id": "gmail",
        },
        expires_delta=timedelta(minutes=10),
    )
    request = SimpleNamespace(query_params={"code": "code", "state": state})

    response = generic_oauth_callback("google", request, db, _google_provider())

    assert response.status_code == 404
    assert _grants(db) == []


def test_a_web_drive_connect_stays_native_while_the_composio_switch_is_on(
    google, monkeypatch
):
    """Drive on the web is a working drive.file connect with the Picker; the
    switch moves only Toby's connect to Composio."""
    db, user, composio = google
    _switch_on(monkeypatch, "google-drive")
    _install(db, user, "google-drive")

    response = auth_api.generic_oauth_login(
        "google",
        token=auth_api.create_access_token(
            data={
                "sub": user.username,
                "user_id": user.id,
                "session_epoch": user.session_epoch,
            }
        ),
        app_id="google-drive",
        db=db,
        db_provider=_google_provider(),
    )

    assert response.status_code in (302, 307)
    location = urlsplit(response.headers["location"])
    assert location.netloc == "accounts.google.com"
    scopes = parse_qs(location.query)["scope"][0].split()
    assert "https://www.googleapis.com/auth/drive.file" in scopes
    assert "https://www.googleapis.com/auth/drive" not in scopes
    assert composio.calls == []


# --- The catalog -------------------------------------------------------------------


def _catalog(app_id: str) -> dict[str, Any]:
    row = get_builtin_public_mcp_app(app_id)
    assert row is not None
    return row


def test_the_drive_switch_changes_only_drives_description(monkeypatch):
    for backend_env, _ in SWITCHES.values():
        monkeypatch.delenv(backend_env, raising=False)
    monkeypatch.delenv(RESTRICTED_FLAG, raising=False)
    native_gmail, native_drive = _catalog("gmail"), _catalog("google-drive")
    assert native_gmail["is_visible_in_connector"] is False
    assert "file picker" in native_drive["description"]

    _switch_on(monkeypatch, "gmail")
    _switch_on(monkeypatch, "google-drive")

    gmail, drive = _catalog("gmail"), _catalog("google-drive")
    assert gmail["is_visible_in_connector"] is True
    # Web Drive stays a native drive.file connect with the Picker, so the
    # switch leaves the row alone except for a description true of both.
    assert {k: v for k, v in drive.items() if k != "description"} == {
        k: v for k, v in native_drive.items() if k != "description"
    }
    assert drive["description"] != native_drive["description"]
    # True of a full-Drive Composio grant, a Picker-limited web one and an
    # existing native drive.file grant alike.
    assert "picker" not in drive["description"].lower()
    assert "allowed to use" in drive["description"]
    # Scopes are the native ones: a Composio start never sends them.
    assert gmail["oauth_scopes"] == native_gmail["oauth_scopes"] == []


def test_a_stale_gmail_row_is_shown_only_while_gmail_goes_through_composio(
    google, monkeypatch
):
    db, _user, _ = google
    db.add(PublicMCPApp(**{**_catalog("gmail"), "is_visible_in_connector": True}))
    db.commit()

    assert mcp_apps.get_app_by_id(db, "gmail")["is_visible_in_connector"] is False
    _switch_on(monkeypatch, "gmail")
    assert mcp_apps.get_app_by_id(db, "gmail")["is_visible_in_connector"] is True


def test_a_gmail_row_an_administrator_hid_stays_hidden(google, monkeypatch):
    """The switch lifts the gate's override; it does not override the row."""
    db, _user, _ = google
    db.add(PublicMCPApp(**{**_catalog("gmail"), "is_visible_in_connector": False}))
    db.commit()
    _switch_on(monkeypatch, "gmail")

    assert mcp_apps.get_app_by_id(db, "gmail")["is_visible_in_connector"] is False


@pytest.mark.parametrize("drive_visible", [True, False])
def test_startup_converges_the_google_rows_with_the_switches(
    tmp_path, monkeypatch, drive_visible
):
    for backend_env, _ in SWITCHES.values():
        monkeypatch.delenv(backend_env, raising=False)
    monkeypatch.delenv(RESTRICTED_FLAG, raising=False)
    init_db(db_url=f"sqlite:///{tmp_path / 'startup.db'}")
    engine = get_engine()
    with Session(engine) as db:
        db.query(PublicMCPApp).filter_by(
            app_id="google-drive"
        ).one().is_visible_in_connector = drive_visible
        db.commit()

    def rows() -> tuple[PublicMCPApp, PublicMCPApp]:
        with Session(engine) as db:
            return (
                db.query(PublicMCPApp).filter_by(app_id="gmail").one(),
                db.query(PublicMCPApp).filter_by(app_id="google-drive").one(),
            )

    gmail, drive = rows()
    assert gmail.is_visible_in_connector is False
    picker_description, drive_scopes = drive.description, drive.oauth_scopes

    _switch_on(monkeypatch, "gmail")
    _switch_on(monkeypatch, "google-drive")
    assert _initialize_database_schema(engine) == []
    gmail, drive = rows()
    assert gmail.is_visible_in_connector is True
    assert drive.description == _catalog("google-drive")["description"]
    assert drive.description != picker_description
    # Drive's visibility and scopes are the web's, and the switch leaves them.
    assert drive.is_visible_in_connector is drive_visible
    assert drive.oauth_scopes == drive_scopes

    for backend_env, _ in SWITCHES.values():
        monkeypatch.delenv(backend_env)
    assert _initialize_database_schema(engine) == []
    gmail, drive = rows()
    assert gmail.is_visible_in_connector is False
    assert drive.description == picker_description
    assert drive.is_visible_in_connector is drive_visible
    assert drive.oauth_scopes == drive_scopes


def test_startup_keeps_an_administrators_drive_description(tmp_path, monkeypatch):
    for backend_env, _ in SWITCHES.values():
        monkeypatch.delenv(backend_env, raising=False)
    monkeypatch.delenv(RESTRICTED_FLAG, raising=False)
    init_db(db_url=f"sqlite:///{tmp_path / 'startup.db'}")
    engine = get_engine()
    with Session(engine) as db:
        drive = db.query(PublicMCPApp).filter_by(app_id="google-drive").one()
        drive.description = "Our own words"
        db.commit()

    _switch_on(monkeypatch, "google-drive")
    assert _initialize_database_schema(engine) == []

    with Session(engine) as db:
        drive = db.query(PublicMCPApp).filter_by(app_id="google-drive").one()
        assert drive.description == "Our own words"


# --- Native shared Google grants ---------------------------------------------------

SHARED_FLAG = "XAGENT_GOOGLE_SHARED_OAUTH_GRANTS_ENABLED"


def _native_shared_grant(db: Session, user: User, *app_ids: str) -> UserOAuth:
    """A native shared Google grant of the same Google account, owned by the
    same actor and associated with ``app_ids``."""
    shared = None
    for app_id in app_ids:
        legacy = UserOAuth(
            user_id=int(user.id),
            provider=app_id,
            resource_owner_key=OWNER,
            provider_user_id=GOOGLE_SUBJECT,
            access_token=f"native-{app_id}",
        )
        db.add(legacy)
        db.flush()
        shared, _, _ = write_google_shared_oauth_callback(
            db,
            user_id=int(user.id),
            resource_owner_key=OWNER,
            oauth_client_id="google-client-id",
            app_id=app_id,
            google_subject=GOOGLE_SUBJECT,
            access_token="native-access",
            refresh_token="native-refresh",
            granted_scopes="openid email profile",
            requested_scopes=["openid"],
            email="a@example.com",
            token_type="Bearer",
            expires_at=None,
            legacy_oauth_account=legacy,
        )
    db.commit()
    assert shared is not None
    return shared


def _shared_for(db: Session, user: User, app_id: str) -> UserOAuth | None:
    db.expire_all()
    return resolve_google_shared_oauth_account(
        db,
        user_id=int(user.id),
        resource_owner_key=OWNER,
        oauth_client_id="google-client-id",
        app_id=app_id,
    )


def test_a_native_shared_grant_never_stands_in_for_the_composio_grant(
    google, monkeypatch
):
    db, user, composio = google
    monkeypatch.setenv(SHARED_FLAG, "true")
    _switch_on(monkeypatch, "google-drive")
    _install(db, user, "google-drive")
    shared = _native_shared_grant(db, user, "google-drive")
    shared_id = int(shared.id)
    assert _shared_for(db, user, "google-drive") is not None

    response = _connect(db, user, composio, "google-drive")

    assert response.status_code == 200, response.body
    # The runtime prefers an associated shared grant over the app's own row.
    # The association named the native Drive row the Composio grant replaced,
    # so it no longer resolves and the native drive.file token cannot stand
    # in for the Composio grant.
    assert _shared_for(db, user, "google-drive") is None
    (grant,) = [row for row in _grants(db) if row.provider == "google-drive"]
    assert grant.token_type == "composio"
    # Connecting through Composio deletes no native credential.
    assert db.get(UserOAuth, shared_id).access_token == "native-access"


def test_a_composio_grant_leaves_the_shared_grant_other_apps_use(google, monkeypatch):
    db, user, composio = google
    monkeypatch.setenv(SHARED_FLAG, "true")
    _switch_on(monkeypatch, "google-drive")
    _install(db, user, "google-drive")
    shared = _native_shared_grant(db, user, "google-drive", "google-docs")
    shared_id = int(shared.id)

    response = _connect(db, user, composio, "google-drive")

    assert response.status_code == 200, response.body
    assert _shared_for(db, user, "google-drive") is None
    docs = _shared_for(db, user, "google-docs")
    assert docs is not None and int(docs.id) == shared_id
    assert db.get(UserOAuth, shared_id).access_token == "native-access"


def test_disconnecting_the_composio_grant_leaves_the_native_shared_grant(
    google, monkeypatch
):
    db, user, composio = google
    monkeypatch.setenv(SHARED_FLAG, "true")
    _switch_on(monkeypatch, "google-drive")
    _install(db, user, "google-drive")
    shared = _native_shared_grant(db, user, "google-drive", "google-docs")
    shared_id = int(shared.id)
    _connect(db, user, composio, "google-drive")

    delete_scoped_user_oauth_accounts(
        db,
        user_id=int(user.id),
        resource_owner_key=OWNER,
        providers=["google-drive"],
    )
    db.commit()

    assert [row.provider for row in _grants(db) if row.token_type == "composio"] == []
    docs = _shared_for(db, user, "google-docs")
    assert docs is not None and int(docs.id) == shared_id


@pytest.mark.parametrize("app_id", GOOGLE_APPS)
def test_a_composio_start_never_enters_shared_grant_mode(google, monkeypatch, app_id):
    db, user, composio = google
    monkeypatch.setenv(SHARED_FLAG, "true")
    # Composio needs no Google client of ours; shared-grant mode keys on it.
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
    _switch_on(monkeypatch, app_id)
    _install(db, user, app_id)
    provider = _google_provider()
    provider.client_id = None
    browser = Browser()

    start = auth_api.start_builtin_oauth_for_resource_owner(
        provider="google",
        app_id=app_id,
        user=user,
        resource_owner_key=OWNER,
        redirect="https://toby.example.com/settings",
        db=db,
        db_provider=provider,
    )
    db.commit()
    browser.receive(start)

    assert start.status_code in (302, 307), start.body
    state = auth_api.verify_token(
        parse_qs(urlsplit(start.headers["location"]).query)["state"][0]
    )
    assert "google_shared_mode" not in state
    assert browser.get(db, start.headers["location"]).status_code == 302
    assert browser.get(db, _verify_url(composio)).status_code == 200


def test_a_native_shared_connect_leaves_the_composio_gmail_grant(google, monkeypatch):
    db, user, composio = google
    monkeypatch.setenv(SHARED_FLAG, "true")
    _switch_on(monkeypatch, "gmail")
    _install(db, user, "gmail")
    assert _connect(db, user, composio, "gmail").status_code == 200
    (before,) = _grants(db)
    account_id = before.access_token

    # A native shared Google connect of the same Google account for another
    # app, by the same actor.
    _native_shared_grant(db, user, "google-docs")

    (gmail,) = [row for row in _grants(db) if row.provider == "gmail"]
    assert (gmail.token_type, gmail.access_token) == ("composio", account_id)
    assert gmail.refresh_token is None
    assert _shared_for(db, user, "gmail") is None
