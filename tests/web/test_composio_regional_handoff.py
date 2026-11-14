"""A regional web launch starts a Composio flow in the region that redeems it.

The handoff ticket carries the provider and launch options; the target region
makes the same switch decision as the web entries, so a switched-on app needs
no native provider row there either.
"""

from __future__ import annotations

import html
import json
import re
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest
from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from starlette.requests import Request

from xagent.web import composio_auth
from xagent.web.api import auth
from xagent.web.models.actor_oauth_flow import ActorOAuthFlowState
from xagent.web.models.database import Base
from xagent.web.models.oauth_provider import OAuthProvider
from xagent.web.models.user import User
from xagent.web.oauth_handoff import OAuthHandoffContext, redeem_handoff

from . import test_composio_google_connect as google_connect
from . import test_composio_salesforce_connect as salesforce_connect
from .test_composio_facebook_connect import (
    API_KEY,
    CALLBACK_URL,
    USER_ID_SECRET,
    Browser,
    _grants,
    _verify_url,
)

OLD = "https://region.example"
NEW = "https://public.example"

# app_id -> (provider, fake Composio, switch env, auth-config env, config id,
# native provider row factory, native authorize host)
APPS = {
    "salesforce": (
        "salesforce",
        salesforce_connect.SalesforceComposio,
        "XAGENT_SALESFORCE_AUTH_BACKEND",
        "COMPOSIO_SALESFORCE_AUTH_CONFIG_ID",
        salesforce_connect.SALESFORCE_AUTH_CONFIG_ID,
        salesforce_connect._salesforce_provider,
        "login.salesforce.com",
    ),
    "google-drive": (
        "google",
        google_connect.GoogleComposio,
        "XAGENT_GOOGLE_DRIVE_AUTH_BACKEND",
        "COMPOSIO_GOOGLE_DRIVE_AUTH_CONFIG_ID",
        google_connect.AUTH_CONFIGS["google-drive"],
        google_connect._google_provider,
        "accounts.google.com",
    ),
}


class Adapter:
    def context(self, request):
        return OAuthHandoffContext("product", "test", "au", OLD)

    def set_route_cookie(self, response, *, state, kind, provider):
        response.set_cookie("scoped-" + kind, "au")


@pytest.fixture
def regional(tmp_path, monkeypatch, app_id):
    provider, fake_cls, backend_env, config_env, config_id, provider_row, _ = APPS[
        app_id
    ]
    fake = fake_cls()
    monkeypatch.setattr(composio_auth.requests, "request", fake.request)
    monkeypatch.setenv("XAGENT_APP_BASE_URL", OLD)
    monkeypatch.setenv("XAGENT_DUAL_ENTRY_OAUTH_ENABLED", "true")
    monkeypatch.setenv(
        "XAGENT_PUBLIC_ENTRIES_JSON",
        json.dumps(
            [
                {
                    "origin": origin,
                    "region_cookie_domain": None,
                    "google_oidc_redirect_uri": origin
                    + "/api/auth/oidc/google/callback",
                }
                for origin in [OLD, NEW]
            ]
        ),
    )
    monkeypatch.setenv("XAGENT_COMPOSIO_CALLBACK_URL", CALLBACK_URL)
    monkeypatch.setenv(composio_auth.COMPOSIO_API_KEY_ENV, API_KEY)
    monkeypatch.setenv(composio_auth.COMPOSIO_USER_ID_SECRET_ENV, USER_ID_SECRET)
    for name in (
        "XAGENT_FACEBOOK_AUTH_BACKEND",
        "XAGENT_INSTAGRAM_AUTH_BACKEND",
        "XAGENT_GMAIL_AUTH_BACKEND",
        "XAGENT_GOOGLE_DRIVE_AUTH_BACKEND",
        "XAGENT_SALESFORCE_AUTH_BACKEND",
    ):
        monkeypatch.delenv(name, raising=False)
    engine = create_engine(
        f"sqlite:///{tmp_path / 'regional.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    db = Session(engine)
    user = User(username="regional", password_hash="unused", session_epoch="epoch")
    db.add(user)
    db.commit()
    app = FastAPI()
    app.state.oauth_handoff_adapter = Adapter()
    try:
        yield (
            db,
            user,
            fake,
            app,
            provider,
            provider_row,
            backend_env,
            config_env,
            config_id,
        )
    finally:
        db.close()
        engine.dispose()


def _request(app: FastAPI, query: dict[str, str], *, cookie: str | None = None):
    headers = [(b"host", urlsplit(OLD).netloc.encode())]
    if cookie:
        headers.append((b"cookie", cookie.encode()))
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/oauth/launch/au",
            "query_string": urlencode(query).encode(),
            "headers": headers,
            "app": app,
        }
    )


def _regional_launch(db, user, app, provider, app_id) -> tuple[str, str, str]:
    """Start at the public entry, then open the target region's confirmation."""
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/",
            "query_string": b"",
            "headers": [
                (b"host", urlsplit(NEW).netloc.encode()),
                (b"x-xagent-oauth-launch-version", b"2"),
            ],
            "app": app,
        }
    )
    launch = auth.start_oauth_authorization(
        provider,
        request,
        auth.OAuthLaunchRequest(app_id=app_id, redirect=NEW),
        db,
        user,
    )
    assert launch.status_code == 200, launch.body
    body = json.loads(launch.body)
    callback = urlsplit(CALLBACK_URL)
    assert body["callback_origin"] == f"{callback.scheme}://{callback.netloc}"
    ticket = parse_qs(urlsplit(body["authorization_url"]).query)["ticket"][0]
    page = redeem_handoff("au", _request(app, {"ticket": ticket}), db)
    assert page.status_code == 200, page.body
    href = html.unescape(re.search(r'href="([^"]+)"', page.body.decode())[1])
    confirm = parse_qs(urlsplit(href).query)["confirm"][0]
    cookie = page.headers["set-cookie"].split(";", 1)[0]
    return ticket, confirm, cookie


def _redeem(db, app, ticket: str, confirm: str, cookie: str):
    return redeem_handoff(
        "au", _request(app, {"ticket": ticket, "confirm": confirm}, cookie=cookie), db
    )


@pytest.mark.parametrize("app_id", ["salesforce"])
def test_a_regional_web_launch_starts_a_composio_flow(regional, monkeypatch, app_id):
    db, user, fake, app, provider, _row, backend_env, config_env, config_id = regional
    monkeypatch.setenv(backend_env, "composio")
    monkeypatch.setenv(config_env, config_id)
    assert (
        db.query(OAuthProvider).filter(OAuthProvider.provider_name == provider).first()
        is None
    )
    ticket, confirm, cookie = _regional_launch(db, user, app, provider, app_id)
    page = redeem_handoff("au", _request(app, {"ticket": ticket}), db)
    assert f"Connect {composio_auth.COMPOSIO_APPS[app_id].label}?" in page.body.decode()

    response = _redeem(db, app, ticket, confirm, cookie)

    assert response.status_code == 303, response.body
    location = response.headers["location"]
    assert location.startswith(f"{CALLBACK_URL}?state=")
    state = auth.verify_token(parse_qs(urlsplit(location).query)["state"][0])
    assert state[composio_auth.AUTH_BACKEND_CLAIM] == composio_auth.AUTH_BACKEND
    assert state["app_id"] == app_id
    assert state["session_epoch"] == "epoch"
    assert state["actor_flow_nonce"] == state["browser_nonce_hash"]
    # The proof cookie follows the shared callback resolved in this region.
    cookie_name = auth._ordinary_oauth_cookie_name(state["browser_nonce_hash"])
    browser = Browser()
    browser.receive(response)
    assert browser.cookies[cookie_name][1] == "/api/auth/meta/callback"

    assert browser.get(db, location).status_code == 302
    verify = browser.get(db, _verify_url(fake))
    assert verify.status_code == 200, verify.body

    db.expire_all()
    (grant,) = _grants(db)
    assert (grant.provider, grant.token_type, grant.access_token) == (
        app_id,
        "composio",
        fake.created[-1],
    )
    assert grant.resource_owner_key is None
    assert db.query(ActorOAuthFlowState).count() == 0


INVALID_CALLBACK_URL = f"{CALLBACK_URL}?x=1"


@pytest.mark.parametrize("app_id", ["salesforce"])
def test_a_regional_composio_launch_refuses_an_unusable_callback_url(
    regional, monkeypatch, app_id
):
    db, user, fake, app, provider, _row, backend_env, config_env, config_id = regional
    monkeypatch.setenv(backend_env, "composio")
    monkeypatch.setenv(config_env, config_id)
    ticket, confirm, cookie = _regional_launch(db, user, app, provider, app_id)
    monkeypatch.setenv("XAGENT_COMPOSIO_CALLBACK_URL", INVALID_CALLBACK_URL)

    response = _redeem(db, app, ticket, confirm, cookie)

    assert response.status_code == 503
    body = response.body.decode()
    assert f"{composio_auth.COMPOSIO_APPS[app_id].label} can't be connected" in body
    assert "location" not in response.headers
    assert db.query(ActorOAuthFlowState).count() == 0
    assert fake.calls == []
    # Spent before the refusal, as redeeming a launch ticket is everywhere else.
    with pytest.raises(HTTPException) as replay:
        _redeem(db, app, ticket, confirm, cookie)
    assert replay.value.status_code == 401


@pytest.mark.parametrize("app_id", ["salesforce"])
def test_a_regional_composio_launch_refuses_missing_composio_config(
    regional, monkeypatch, app_id
):
    db, user, fake, app, provider, _row, backend_env, config_env, config_id = regional
    monkeypatch.setenv(backend_env, "composio")
    monkeypatch.setenv(config_env, config_id)
    ticket, confirm, cookie = _regional_launch(db, user, app, provider, app_id)
    monkeypatch.delenv(composio_auth.COMPOSIO_API_KEY_ENV)

    response = _redeem(db, app, ticket, confirm, cookie)

    assert response.status_code == 500
    body = response.body.decode()
    assert "OAuth provider not configured" in body
    assert f"Missing {composio_auth.COMPOSIO_API_KEY_ENV}" in body
    assert "location" not in response.headers
    assert db.query(ActorOAuthFlowState).count() == 0
    assert fake.calls == []


@pytest.mark.parametrize("app_id", ["google-drive"])
def test_an_unusable_composio_setting_never_refuses_a_native_regional_launch(
    regional, monkeypatch, app_id
):
    db, user, fake, app, provider, provider_row, backend_env, config_env, config_id = (
        regional
    )
    monkeypatch.setenv(backend_env, "composio")
    monkeypatch.setenv(config_env, config_id)
    _add_native_provider(db, provider, provider_row)
    ticket, confirm, cookie = _regional_launch(db, user, app, provider, app_id)
    monkeypatch.setenv("XAGENT_COMPOSIO_CALLBACK_URL", INVALID_CALLBACK_URL)

    response = _redeem(db, app, ticket, confirm, cookie)

    assert response.status_code == 303
    assert urlsplit(response.headers["location"]).netloc == APPS[app_id][6]
    assert db.query(ActorOAuthFlowState).count() == 0
    assert fake.calls == []


def _add_native_provider(db, provider, provider_row) -> None:
    native = provider_row()
    db.add(
        OAuthProvider(
            provider_name=provider,
            name=provider.title(),
            client_id=native.client_id,
            client_secret=native.client_secret,
            auth_url=native.auth_url,
            token_url=native.token_url,
            redirect_uri=native.redirect_uri,
            userinfo_url=native.userinfo_url,
            user_id_path=native.user_id_path,
            email_path=native.email_path,
            default_scopes=native.default_scopes,
        )
    )
    db.commit()


@pytest.mark.parametrize("app_id", ["google-drive"])
def test_a_regional_web_launch_stays_native_with_the_switch_on(
    regional, monkeypatch, app_id
):
    db, user, fake, app, provider, provider_row, backend_env, config_env, config_id = (
        regional
    )
    monkeypatch.setenv(backend_env, "composio")
    monkeypatch.setenv(config_env, config_id)
    _add_native_provider(db, provider, provider_row)
    ticket, confirm, cookie = _regional_launch(db, user, app, provider, app_id)

    response = _redeem(db, app, ticket, confirm, cookie)

    assert response.status_code == 303
    location = urlsplit(response.headers["location"])
    assert location.netloc == APPS[app_id][6]
    state = auth.verify_token(parse_qs(location.query)["state"][0])
    assert composio_auth.AUTH_BACKEND_CLAIM not in state
    assert "actor_flow_nonce" not in state
    assert db.query(ActorOAuthFlowState).count() == 0
    assert fake.calls == []
    assert _grants(db) == []


@pytest.mark.parametrize("app_id", sorted(APPS))
def test_a_regional_web_launch_stays_native_with_the_switch_off(regional, app_id):
    db, user, fake, app, provider, provider_row, _backend_env, _config_env, _ = regional
    _add_native_provider(db, provider, provider_row)
    ticket, confirm, cookie = _regional_launch(db, user, app, provider, app_id)

    response = _redeem(db, app, ticket, confirm, cookie)

    assert response.status_code == 303
    location = urlsplit(response.headers["location"])
    assert location.netloc == APPS[app_id][6]
    state = auth.verify_token(parse_qs(location.query)["state"][0])
    assert composio_auth.AUTH_BACKEND_CLAIM not in state
    assert "actor_flow_nonce" not in state
    assert db.query(ActorOAuthFlowState).count() == 0
    assert fake.calls == []
    # The native proof cookie keeps its browser-wide default path.
    cookie_name = auth._ordinary_oauth_cookie_name(state["browser_nonce_hash"])
    browser = Browser()
    browser.receive(response)
    assert browser.cookies[cookie_name][1] == "/"
    assert _grants(db) == []
