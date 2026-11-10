"""Gmail and Google Drive connect through Composio-hosted auth (#3649).

Composio's Google app already holds Google's restricted-scope approval, so a
Gmail or Drive connect started through it gets full access while our own
review gate (``XAGENT_GOOGLE_RESTRICTED_SCOPES_ENABLED``) stays closed. Native
Google connects keep obeying that gate. Composio is faked here; no request
leaves the process.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy.orm import Session

from xagent.core.utils.encryption import encrypt_value
from xagent.web import composio_auth
from xagent.web.api import auth as auth_api
from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app
from xagent.web.mcp_apps import ensure_builtin_oauth_server_definition
from xagent.web.models.actor_oauth_flow import ActorOAuthFlowState
from xagent.web.models.mcp import UserMCPServer
from xagent.web.models.oauth_provider import OAuthProvider
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User

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
    and link them to ``user``."""
    for app_id in app_ids:
        row = get_builtin_public_mcp_app(app_id)
        assert row is not None
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
