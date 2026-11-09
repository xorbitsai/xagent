"""The Composio connect flow serves apps of any provider (#3645).

Facebook and Instagram were the only Composio apps, both under the ``meta``
provider. Here a Linear app is registered the way the Salesforce and Google
apps will be (#3646, #3649): another provider, its own toolkit, its own
identity read and id format. Composio's OAuth user verification is one URL
per project, so every app's trips land on the one shared callback path,
while each flow keeps its own provider for its state, link and log checks.
"""

from __future__ import annotations

import logging
import re
import time
from http.cookies import SimpleCookie
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlsplit

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
    CALLBACK_PATH,
    CALLBACK_URL,
    COMPOSIO_REDIRECT,
    FACEBOOK_USER_ID,
    OWNER,
    Browser,
    FakeComposio,
    _flow_cookies,
    _grants,
    _meta_provider,
    _Response,
    _start,
    _verify_url,
)

# Facebook's fixtures, shared: the environment and a database with the
# Facebook connector installed.
composio_env = facebook_connect.composio_env
db_session = facebook_connect.db_session

LINEAR_AUTH_CONFIG_ID = "ac_linear_test"
LINEAR_USER_ID = "8f2c1d3e-4b5a-4c6d-9e8f-0a1b2c3d4e5f"
LINEAR_IDENTITY_URL = "https://api.linear.app/oauth/userinfo"
LINEAR_APP = composio_auth.ComposioApp(
    app_id="linear",
    provider="linear",
    label="Linear",
    backend_env="XAGENT_LINEAR_AUTH_BACKEND",
    auth_config_env="COMPOSIO_LINEAR_AUTH_CONFIG_ID",
    toolkit_slug="linear",
    identity_url=LINEAR_IDENTITY_URL,
    identity_field="id",
    identity_params=(),
    identity_id_pattern=re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}"),
)


class LinearComposio(FakeComposio):
    """Composio answering for Linear accounts too."""

    def __init__(self) -> None:
        super().__init__()
        self.linear_identity: Any = LINEAR_USER_ID

    def request(self, *, method, url, headers, json, timeout):  # noqa: A002
        path = url[len(composio_auth.COMPOSIO_API_BASE_URL) :]
        if method == "POST" and path == "/tools/execute/proxy":
            if json["endpoint"] == LINEAR_IDENTITY_URL:
                self.calls.append((method, path, json))
                return _Response(
                    200, {"data": {"id": self.linear_identity}, "status": 200}
                )
        response = super().request(
            method=method, url=url, headers=headers, json=json, timeout=timeout
        )
        if (
            method == "GET"
            and path.startswith("/connected_accounts/")
            and self.auth_configs.get(path.rsplit("/", 1)[1]) == LINEAR_AUTH_CONFIG_ID
        ):
            response._payload["toolkit"] = {"slug": "linear"}
        return response


@pytest.fixture
def composio(monkeypatch) -> LinearComposio:
    fake = LinearComposio()
    monkeypatch.setattr(composio_auth.requests, "request", fake.request)

    def no_provider_call(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a Composio flow must never call the provider directly")

    monkeypatch.setattr(auth_api.requests, "post", no_provider_call)
    monkeypatch.setattr(auth_api.requests, "get", no_provider_call)
    return fake


@pytest.fixture
def linear(db_session, composio_env, composio, monkeypatch):
    """Linear registered as a Composio app, switched on, its connector installed."""
    monkeypatch.setitem(composio_auth.COMPOSIO_APPS, "linear", LINEAR_APP)
    monkeypatch.setenv(LINEAR_APP.backend_env, "composio")
    monkeypatch.setenv(LINEAR_APP.auth_config_env, LINEAR_AUTH_CONFIG_ID)
    db, user = db_session
    # The migration-seeded Meta row whose callback URL every Composio app uses.
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
    app = get_builtin_public_mcp_app("linear")
    assert app is not None
    db.add(PublicMCPApp(**app))
    db.commit()
    server = ensure_builtin_oauth_server_definition(db, app_id="linear")
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


def _linear_provider() -> SimpleNamespace:
    return SimpleNamespace(
        provider_name="linear",
        client_id=encrypt_value("linear-client-id"),
        client_secret=encrypt_value("linear-client-secret"),
        auth_url="https://linear.app/oauth/authorize",
        token_url="https://api.linear.app/oauth/token",
        redirect_uri="https://app.example.com/api/auth/linear/callback",
        userinfo_url=None,
        user_id_path="id",
        email_path="email",
        default_scopes=["read"],
    )


def _start_linear(db: Session, user: User, browser: Browser) -> Any:
    response = auth_api.start_builtin_oauth_for_resource_owner(
        provider="linear",
        app_id="linear",
        user=user,
        resource_owner_key=OWNER,
        redirect="https://toby.example.com/settings",
        db=db,
        db_provider=_linear_provider(),
    )
    db.commit()
    browser.receive(response)
    return response


def _state_of(response: Any) -> str:
    return parse_qs(urlsplit(response.headers["location"]).query)["state"][0]


def _resigned(state: str, **claims: Any) -> str:
    """The same genuinely signed state with some claims changed."""
    payload = auth_api.jwt.decode(
        state,
        auth_api.JWT_SECRET_KEY,
        algorithms=[auth_api.JWT_ALGORITHM],
        options={"verify_exp": False},
    )
    payload.update(claims)
    return auth_api.jwt.encode(
        payload, auth_api.JWT_SECRET_KEY, algorithm=auth_api.JWT_ALGORITHM
    )


def _begin_url(state: str) -> str:
    return f"{CALLBACK_URL}?state={state}"


# --- Which apps go through Composio --------------------------------------------


def test_an_app_goes_through_composio_only_under_its_own_provider(linear):
    assert composio_auth.composio_app("linear", "linear") is LINEAR_APP
    assert composio_auth.composio_app("Linear", "linear") is LINEAR_APP
    assert composio_auth.composio_app("meta", "linear") is None
    assert composio_auth.composio_app("linear", "facebook") is None
    assert composio_auth.composio_app("meta", "facebook").app_id == "facebook"


# --- A non-Meta app end to end -----------------------------------------------------


def test_a_non_meta_app_starts_on_the_shared_callback_path(linear):
    db, user, composio = linear

    response = _start_linear(db, user, Browser())

    assert response.status_code in (302, 307)
    location = urlsplit(response.headers["location"])
    assert f"{location.scheme}://{location.netloc}{location.path}" == CALLBACK_URL
    state = auth_api.verify_token(_state_of(response))
    assert (state["provider"], state["app_id"]) == ("linear", "linear")
    (cookie,) = response.headers.getlist("set-cookie")
    assert auth_api.is_actor_oauth_cookie_header(cookie)
    jar = SimpleCookie()
    jar.load(cookie)
    (morsel,) = jar.values()
    assert morsel["path"] == CALLBACK_PATH
    assert composio.calls == []


def test_a_non_meta_app_connects_and_stores_its_own_grant(linear):
    db, user, composio = linear
    browser = Browser()
    start = _start_linear(db, user, browser)

    begin = browser.get(db, start.headers["location"])
    assert begin.status_code == 302, begin.body
    assert begin.headers["location"] == COMPOSIO_REDIRECT
    (link,) = [body for _, path, body in composio.calls if path.endswith("/link")]
    assert link["auth_config_id"] == LINEAR_AUTH_CONFIG_ID
    assert link["callback_url"] == CALLBACK_URL

    response = browser.get(db, _verify_url(composio))

    assert response.status_code == 200, response.body
    (identity,) = [
        body for _, path, body in composio.calls if path == "/tools/execute/proxy"
    ]
    assert identity["endpoint"] == LINEAR_IDENTITY_URL
    # Graph's ``fields`` is a Meta parameter; Linear's read takes none.
    assert identity["parameters"] == []
    (grant,) = _grants(db)
    assert (grant.provider, grant.provider_user_id) == ("linear", LINEAR_USER_ID)
    assert grant.token_type == "composio"
    assert db.query(ActorOAuthFlowState).count() == 0
    assert _flow_cookies(browser) == []


@pytest.mark.parametrize(
    "identity", ["102030405060", "not-a-uuid", None], ids=["numeric", "text", "none"]
)
def test_a_non_meta_identity_must_match_its_apps_own_format(linear, identity):
    db, user, composio = linear
    composio.linear_identity = identity
    browser = Browser()
    browser.get(db, _start_linear(db, user, browser).headers["location"])

    response = browser.get(db, _verify_url(composio))

    assert response.status_code == 503
    assert _grants(db) == []


def test_a_meta_identity_keeps_its_numeric_format(db_session, composio_env, composio):
    db, user = db_session
    composio.me_response = (200, {"data": {"id": LINEAR_USER_ID}, "status": 200})
    browser = Browser()
    browser.get(db, _start(db, user, browser).headers["location"])

    response = browser.get(db, _verify_url(composio))

    assert response.status_code == 503
    assert _grants(db) == []


def test_a_denied_non_meta_flow_ends_and_is_logged_under_its_provider(linear, caplog):
    db, user, composio = linear
    browser = Browser()
    browser.get(db, _start_linear(db, user, browser).headers["location"])
    caplog.set_level(logging.WARNING)

    response = browser.get(
        db,
        f"{CALLBACK_URL}?status=failed&error=access_denied"
        f"&connected_account_id={composio.created[0]}",
    )

    assert response.status_code == 400
    assert "Permission was not granted" in response.body.decode()
    assert db.query(ActorOAuthFlowState).count() == 0
    assert _flow_cookies(browser) == []
    assert "provider=linear app_id=linear" in caplog.text


def test_too_many_windows_names_no_other_app(linear, caplog):
    db, user, composio = linear
    browser = Browser()
    for _ in range(3):
        browser.get(db, _start_linear(db, user, browser).headers["location"])
    caplog.set_level(logging.WARNING)

    response = browser.get(db, _start_linear(db, user, browser).headers["location"])

    assert response.status_code == 400
    body = response.body.decode()
    assert "Linear was not connected" in body
    assert "Facebook" not in body and "Instagram" not in body
    assert "provider=linear app_id=linear actor_flow=true" in caplog.text
    assert "reason=too_many_open_flows" in caplog.text


def test_the_open_flow_limit_counts_every_apps_windows_together(linear):
    db, user, composio = linear
    browser = Browser()
    browser.get(db, _start(db, user, browser).headers["location"])
    browser.get(db, _start(db, user, browser).headers["location"])
    browser.get(db, _start_linear(db, user, browser).headers["location"])

    response = browser.get(db, _start_linear(db, user, browser).headers["location"])

    assert response.status_code == 400
    assert "Too many" in response.body.decode()
    assert composio.paths().count("/connected_accounts/link") == 3


def test_facebook_and_linear_connects_in_one_browser_each_complete(linear):
    """Each session completes its own flow and stores its own app's grant."""
    db, user, composio = linear
    browser = Browser()
    browser.get(db, _start(db, user, browser).headers["location"])
    facebook_account = composio.created[-1]
    browser.get(db, _start_linear(db, user, browser).headers["location"])
    linear_account = composio.created[-1]

    assert browser.get(db, _verify_url(composio, linear_account)).status_code == 200
    assert browser.get(db, _verify_url(composio, facebook_account)).status_code == 200

    stored = {
        (grant.provider, grant.provider_user_id, grant.access_token)
        for grant in _grants(db)
    }
    assert stored == {
        ("linear", LINEAR_USER_ID, linear_account),
        ("facebook", FACEBOOK_USER_ID, facebook_account),
    }
    assert db.query(ActorOAuthFlowState).count() == 0


# --- A state is spent only on its own provider's flow ---------------------------


@pytest.mark.parametrize(
    "app_id,provider",
    [("linear", "meta"), ("facebook", "linear")],
    ids=["linear-state-as-meta", "facebook-state-as-linear"],
)
def test_a_state_naming_another_provider_creates_no_link(
    linear, monkeypatch, app_id, provider
):
    db, user, composio = linear
    browser = Browser()
    if app_id == "linear":
        start = _start_linear(db, user, browser)
    else:
        start = _start(db, user, browser)

    response = browser.get(
        db, _begin_url(_resigned(_state_of(start), provider=provider))
    )

    assert response.status_code == 400
    assert composio.calls == []
    assert db.query(ActorOAuthFlowState).count() == 1


def test_an_expired_non_meta_state_says_the_window_expired(linear):
    db, user, composio = linear
    browser = Browser()
    start = _start_linear(db, user, browser)

    response = browser.get(
        db, _begin_url(_resigned(_state_of(start), exp=int(time.time()) - 60))
    )

    assert response.status_code == 400
    assert "already been used" in response.body.decode()
    assert composio.calls == []


def test_a_return_no_flow_claims_names_no_app(linear):
    """Anyone can send a bare success return; the page cannot know the app."""
    db, _user, composio = linear

    response = Browser().get(
        db, f"{CALLBACK_URL}?status=success&connected_account_id=ca_unknown"
    )

    assert response.status_code == 503
    body = response.body.decode()
    assert "This app can't be connected right now" in body
    assert "The connection service could not" in body
    assert "Facebook" not in body and "Linear" not in body
    assert composio.calls == []


# --- The dedicated Composio callback (XAGENT_COMPOSIO_CALLBACK_URL) --------------

DEDICATED_URL = "https://app.example.com/api/auth/composio/callback"
DEDICATED_PATH = "/api/auth/composio/callback"


class RoutedBrowser(Browser):
    """A browser whose requests reach whichever callback route they name."""

    def get(self, db: Session, url: str) -> Any:
        parts = urlsplit(url)
        if parts.path == CALLBACK_PATH:
            return super().get(db, url)
        assert parts.path == DEDICATED_PATH, parts.path
        query = {name: values[0] for name, values in parse_qs(parts.query).items()}
        request = SimpleNamespace(query_params=query, cookies=self.for_path(parts.path))
        response = auth_api.composio_oauth_callback(request, db)
        self.receive(response)
        return response


@pytest.fixture
def dedicated(linear, monkeypatch):
    monkeypatch.setenv("XAGENT_COMPOSIO_CALLBACK_URL", DEDICATED_URL)
    return linear


def _cookie_path(response: Any, prefix: str) -> str:
    jar = SimpleCookie()
    for header in response.headers.getlist("set-cookie"):
        jar.load(header)
    (path,) = [m["path"] for name, m in jar.items() if name.startswith(prefix)]
    return path


@pytest.mark.parametrize("app_id", ["facebook", "linear"])
def test_with_the_setting_every_app_uses_the_dedicated_callback(dedicated, app_id):
    db, user, composio = dedicated
    browser = RoutedBrowser()
    start = (_start if app_id == "facebook" else _start_linear)(db, user, browser)

    assert start.headers["location"].startswith(f"{DEDICATED_URL}?state=")
    assert _cookie_path(start, "xagent_actor_oauth") == DEDICATED_PATH

    begin = browser.get(db, start.headers["location"])
    assert begin.status_code == 302, begin.body
    (link,) = [body for _, path, body in composio.calls if path.endswith("/link")]
    assert link["callback_url"] == DEDICATED_URL
    assert _cookie_path(begin, composio_auth.FLOW_COOKIE_PREFIX) == DEDICATED_PATH

    verify = browser.get(
        db, f"{DEDICATED_URL}?session_uri={_verify_url(composio).split('=', 1)[1]}"
    )

    assert verify.status_code == 200, verify.body
    (grant,) = _grants(db)
    assert grant.provider == app_id
    # Cleared on the path it was set on, or a real browser would keep it.
    assert _cookie_path(verify, composio_auth.FLOW_COOKIE_PREFIX) == DEDICATED_PATH
    assert _flow_cookies(browser) == []


def test_with_the_setting_no_meta_provider_row_is_needed(dedicated):
    db, user, composio = dedicated
    db.query(OAuthProvider).filter(OAuthProvider.provider_name == "meta").delete()
    db.commit()
    browser = RoutedBrowser()
    start = _start_linear(db, user, browser)
    browser.get(db, start.headers["location"])

    verify = browser.get(
        db, f"{DEDICATED_URL}?session_uri={_verify_url(composio).split('=', 1)[1]}"
    )

    assert verify.status_code == 200, verify.body
    assert [grant.provider for grant in _grants(db)] == ["linear"]


def test_a_denied_return_on_the_dedicated_callback_ends_its_flow(dedicated):
    db, user, composio = dedicated
    browser = RoutedBrowser()
    browser.get(db, _start_linear(db, user, browser).headers["location"])

    response = browser.get(
        db,
        f"{DEDICATED_URL}?status=failed&error=access_denied"
        f"&connected_account_id={composio.created[0]}",
    )

    assert response.status_code == 400
    assert "Permission was not granted" in response.body.decode()
    assert db.query(ActorOAuthFlowState).count() == 0
    assert _flow_cookies(browser) == []


@pytest.mark.parametrize(
    "query", ["code=abc&state=xyz", "error=access_denied", ""], ids=str
)
def test_the_dedicated_callback_takes_only_composio_trips(dedicated, query, caplog):
    db, _user, composio = dedicated
    caplog.set_level(logging.WARNING)

    response = RoutedBrowser().get(db, f"{DEDICATED_URL}?{query}")

    assert response.status_code == 400
    assert "reason=not_a_composio_trip" in caplog.text
    assert composio.calls == []


def test_the_dedicated_route_matches_before_the_provider_route(db_session):
    """Without its own route, /composio/callback would be an unknown provider."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from xagent.web.models.database import get_db

    db, _user = db_session
    app = FastAPI()
    app.include_router(auth_api.auth_router)
    app.dependency_overrides[get_db] = lambda: db

    response = TestClient(app).get(f"{DEDICATED_PATH}?code=abc&state=xyz")

    assert response.status_code == 400
    assert "Unsupported provider" not in response.text


@pytest.mark.parametrize(
    "value",
    [
        "not a url",
        "ftp://app.example.com/api/auth/composio/callback",
        "https://app.example.com/api/auth/composio/callback?x=1",
        "https://user:pass@app.example.com/api/auth/composio/callback",
        "https://app.example.com/api/auth/linear/callback",
    ],
    ids=["text", "ftp", "query", "credentials", "not-a-composio-route"],
)
def test_an_invalid_setting_refuses_the_start_and_the_trips(linear, monkeypatch, value):
    db, user, composio = linear
    monkeypatch.setenv("XAGENT_COMPOSIO_CALLBACK_URL", value)

    start = _start_linear(db, user, Browser())

    assert start.status_code == 503
    assert "set-cookie" not in start.headers
    assert db.query(ActorOAuthFlowState).count() == 0

    trip = Browser().get(db, _verify_url(composio, "ca_any"))
    assert trip.status_code == 503
    assert composio.calls == []


# --- Our own client id is not needed for a Composio start -------------------------


def _linear_provider_without_client_id() -> SimpleNamespace:
    row = _linear_provider()
    row.client_id = ""
    return row


def test_a_composio_start_needs_no_client_id_of_ours(linear, monkeypatch):
    db, user, composio = linear
    monkeypatch.delenv("LINEAR_CLIENT_ID", raising=False)

    response = auth_api.start_builtin_oauth_for_resource_owner(
        provider="linear",
        app_id="linear",
        user=user,
        resource_owner_key=OWNER,
        db=db,
        db_provider=_linear_provider_without_client_id(),
    )

    assert response.status_code in (302, 307)
    assert urlsplit(response.headers["location"]).path == CALLBACK_PATH


def test_a_native_start_still_needs_our_client_id(linear, monkeypatch):
    db, user, composio = linear
    monkeypatch.delenv("LINEAR_CLIENT_ID", raising=False)
    monkeypatch.setenv(LINEAR_APP.backend_env, "native")

    response = auth_api.start_builtin_oauth_for_resource_owner(
        provider="linear",
        app_id="linear",
        user=user,
        resource_owner_key=OWNER,
        db=db,
        db_provider=_linear_provider_without_client_id(),
    )

    assert response.status_code == 500
    assert "LINEAR_CLIENT_ID" in response.body.decode()


def test_another_providers_own_callback_is_never_a_composio_trip(linear):
    """Only the Meta callback and the dedicated one take Composio trips."""
    db, user, composio = linear
    browser = Browser()
    browser.get(db, _start_linear(db, user, browser).headers["location"])

    response = auth_api.generic_oauth_callback(
        "linear",
        SimpleNamespace(
            query_params={"session_uri": "https://backend.composio.dev/verify/x"},
            cookies={},
        ),
        db,
        _linear_provider(),
    )

    # The provider's own native handling answers, not the Composio flow.
    assert response.status_code == 400
    assert "Missing code or state" in response.body.decode()
    assert not [path for _, path, _ in composio.calls if "complete_auth" in path]


def test_the_setting_may_name_the_meta_callback_behind_a_prefix(linear, monkeypatch):
    """A reverse proxy may mount the backend under a prefix."""
    db, user, composio = linear
    monkeypatch.setenv(
        "XAGENT_COMPOSIO_CALLBACK_URL",
        "https://app.example.com/x/api/auth/meta/callback",
    )

    start = _start_linear(db, user, Browser())

    assert start.headers["location"].startswith(
        "https://app.example.com/x/api/auth/meta/callback?state="
    )
