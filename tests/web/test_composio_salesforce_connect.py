"""Salesforce connects through Composio-hosted auth (#3646).

With ``XAGENT_SALESFORCE_AUTH_BACKEND=composio`` a Toby Salesforce connect
goes through Composio's ``salesforce`` toolkit. Composio's hosted page finds
the member's org at sign-in, so we never ask for a My Domain; the org's host
is only known from the connected account's ``base_url``. The grant records
the Salesforce user id read from that org's own OIDC userinfo, which (unlike
login.salesforce.com's) also serves sandbox orgs. Composio is faked here; no
request leaves the process.
"""

from __future__ import annotations

import logging
from datetime import timedelta
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
from xagent.web.models.user_oauth import UserOAuth

from . import test_composio_facebook_connect as facebook_connect
from .test_composio_facebook_connect import (
    CALLBACK_URL,
    COMPOSIO_REDIRECT,
    OTHER_OWNER,
    OWNER,
    Browser,
    FakeComposio,
    _flow_cookies,
    _grants,
    _meta_provider,
    _Response,
    _verify_url,
)

# Facebook's fixtures, shared: the environment and a database with the
# Facebook connector installed.
composio_env = facebook_connect.composio_env
db_session = facebook_connect.db_session

SALESFORCE_AUTH_CONFIG_ID = "ac_salesforce_test"
ORG_HOST = "acme.my.salesforce.com"
BASE_URL = f"https://{ORG_HOST}/services/data/v67.0"
SALESFORCE_USER_ID = "005Hs00000AbCdEFGH"
ORGANIZATION_ID = "00DHs00000XyZaBCDE"


class SalesforceComposio(FakeComposio):
    """Composio answering for Salesforce accounts too."""

    def __init__(self) -> None:
        super().__init__()
        # The connected account's ``data`` for a Salesforce account, as the
        # live test read it; a test may replace it.
        self.salesforce_data: Any = {"base_url": BASE_URL, "subdomain": "login"}
        self.userinfo: Any = {
            "user_id": SALESFORCE_USER_ID,
            "organization_id": ORGANIZATION_ID,
            # Its host is login.salesforce.com whatever the org: never read.
            "sub": (
                "https://login.salesforce.com/id/"
                f"{ORGANIZATION_ID}/{SALESFORCE_USER_ID}"
            ),
        }

    def request(self, *, method, url, headers, json, timeout):  # noqa: A002
        path = url[len(composio_auth.COMPOSIO_API_BASE_URL) :]
        if method == "POST" and path == "/tools/execute/proxy":
            if urlsplit(json["endpoint"]).path == "/services/oauth2/userinfo":
                self.calls.append((method, path, json))
                return _Response(200, {"data": self.userinfo, "status": 200})
        response = super().request(
            method=method, url=url, headers=headers, json=json, timeout=timeout
        )
        if (
            method == "GET"
            and path.startswith("/connected_accounts/")
            and self.auth_configs.get(path.rsplit("/", 1)[1])
            == SALESFORCE_AUTH_CONFIG_ID
        ):
            response._payload["toolkit"] = {"slug": "salesforce"}
            if self.salesforce_data is not None:
                response._payload["data"] = self.salesforce_data
        return response


@pytest.fixture
def composio(monkeypatch) -> SalesforceComposio:
    fake = SalesforceComposio()
    monkeypatch.setattr(composio_auth.requests, "request", fake.request)

    def no_provider_call(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a Composio flow must never call Salesforce directly")

    monkeypatch.setattr(auth_api.requests, "post", no_provider_call)
    monkeypatch.setattr(auth_api.requests, "get", no_provider_call)
    return fake


@pytest.fixture
def salesforce_env(composio_env, monkeypatch):
    monkeypatch.setenv("XAGENT_SALESFORCE_AUTH_BACKEND", "composio")
    monkeypatch.setenv("COMPOSIO_SALESFORCE_AUTH_CONFIG_ID", SALESFORCE_AUTH_CONFIG_ID)


@pytest.fixture
def salesforce(db_session, salesforce_env, composio):
    """Salesforce switched on, its connector installed for the user."""
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
    app = get_builtin_public_mcp_app("salesforce")
    assert app is not None
    db.add(PublicMCPApp(**app))
    db.commit()
    server = ensure_builtin_oauth_server_definition(db, app_id="salesforce")
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


def _salesforce_provider() -> SimpleNamespace:
    return SimpleNamespace(
        provider_name="salesforce",
        client_id=encrypt_value("salesforce-client-id"),
        client_secret=encrypt_value("salesforce-client-secret"),
        auth_url="https://login.salesforce.com/services/oauth2/authorize",
        token_url="https://login.salesforce.com/services/oauth2/token",
        redirect_uri="https://app.example.com/api/auth/salesforce/callback",
        userinfo_url="",
        user_id_path="user_id",
        email_path="email",
        default_scopes=["api", "refresh_token", "openid"],
    )


def _start_salesforce(db: Session, user: User, browser: Browser) -> Any:
    response = auth_api.start_builtin_oauth_for_resource_owner(
        provider="salesforce",
        app_id="salesforce",
        user=user,
        resource_owner_key=OWNER,
        redirect="https://toby.example.com/settings",
        db=db,
        db_provider=_salesforce_provider(),
    )
    db.commit()
    browser.receive(response)
    return response


def _connect(db: Session, user: User, composio: SalesforceComposio) -> Any:
    browser = Browser()
    begin = browser.get(db, _start_salesforce(db, user, browser).headers["location"])
    assert begin.status_code == 302, begin.body
    return browser.get(db, _verify_url(composio))


def _identity_reads(composio: SalesforceComposio) -> list[Any]:
    return [body for _, path, body in composio.calls if path == "/tools/execute/proxy"]


# --- Registration and switch ----------------------------------------------------


def _via_composio(provider: str, app_id: str | None) -> str | None:
    app = composio_auth.composio_app(provider, app_id)
    return app.app_id if app else None


def test_salesforce_is_registered_on_composios_salesforce_toolkit():
    app = composio_auth.COMPOSIO_APPS["salesforce"]

    assert app.provider == "salesforce"
    assert app.toolkit_slug == "salesforce"
    assert app.backend_env == "XAGENT_SALESFORCE_AUTH_BACKEND"
    assert app.auth_config_env == "COMPOSIO_SALESFORCE_AUTH_CONFIG_ID"


def test_salesforce_follows_only_its_own_switch(composio_env, monkeypatch):
    monkeypatch.delenv("XAGENT_SALESFORCE_AUTH_BACKEND", raising=False)
    assert _via_composio("salesforce", "salesforce") is None
    # Facebook's switch is on in this environment; it moves nothing else.
    assert _via_composio("meta", "facebook") == "facebook"

    monkeypatch.setenv("XAGENT_SALESFORCE_AUTH_BACKEND", "composio")
    monkeypatch.setenv(composio_auth.FACEBOOK_AUTH_BACKEND_ENV, "native")
    assert _via_composio("salesforce", "salesforce") == "salesforce"
    assert _via_composio("Salesforce", "Salesforce") == "salesforce"
    assert _via_composio("meta", "facebook") is None
    # Only under its own provider.
    assert _via_composio("meta", "salesforce") is None
    assert _via_composio("salesforce", "facebook") is None

    monkeypatch.setenv("XAGENT_SALESFORCE_AUTH_BACKEND", "native")
    assert _via_composio("salesforce", "salesforce") is None


def test_with_the_switch_off_a_salesforce_start_stays_native(
    db_session, composio_env, composio, monkeypatch
):
    monkeypatch.delenv("XAGENT_SALESFORCE_AUTH_BACKEND", raising=False)
    db, user = db_session
    app = get_builtin_public_mcp_app("salesforce")
    assert app is not None
    db.add(PublicMCPApp(**app))
    db.commit()
    server = ensure_builtin_oauth_server_definition(db, app_id="salesforce")
    db.add(
        UserMCPServer(
            user_id=int(user.id),
            mcpserver_id=int(server.id),
            is_owner=True,
            is_active=True,
        )
    )
    db.commit()

    response = _start_salesforce(db, user, Browser())

    assert response.status_code in (302, 307)
    location = urlsplit(response.headers["location"])
    assert (location.netloc, location.path) == (
        "login.salesforce.com",
        "/services/oauth2/authorize",
    )
    query = parse_qs(location.query)
    assert query["client_id"] == ["salesforce-client-id"]
    assert query["scope"] == ["api refresh_token openid"]
    state = auth_api.verify_token(query["state"][0])
    assert composio_auth.AUTH_BACKEND_CLAIM not in state
    assert composio.calls == []


def test_with_the_switch_on_a_web_salesforce_connect_stays_native(salesforce):
    """Only actor (Toby) connects go through Composio: an ordinary web connect,
    which has no resource owner key, still uses our own connected app."""
    db, user, composio = salesforce
    token = auth_api.create_access_token(
        data={"sub": user.username, "user_id": user.id, "type": "access"},
        expires_delta=timedelta(minutes=5),
    )

    response = auth_api.generic_oauth_login(
        provider="salesforce",
        token=token,
        app_id="salesforce",
        db=db,
        db_provider=_salesforce_provider(),
    )

    assert response.status_code in (302, 307)
    location = urlsplit(response.headers["location"])
    assert (location.netloc, location.path) == (
        "login.salesforce.com",
        "/services/oauth2/authorize",
    )
    query = parse_qs(location.query)
    assert query["client_id"] == ["salesforce-client-id"]
    state = auth_api.verify_token(query["state"][0])
    assert state.get("resource_owner_key") is None
    assert composio_auth.AUTH_BACKEND_CLAIM not in state
    assert composio.calls == []


# --- Connect end to end ---------------------------------------------------------


def test_a_salesforce_connect_stores_the_orgs_own_user_id(salesforce):
    db, user, composio = salesforce
    browser = Browser()
    start = _start_salesforce(db, user, browser)

    begin = browser.get(db, start.headers["location"])
    assert begin.status_code == 302, begin.body
    assert begin.headers["location"] == COMPOSIO_REDIRECT
    (link,) = [body for _, path, body in composio.calls if path.endswith("/link")]
    assert link["auth_config_id"] == SALESFORCE_AUTH_CONFIG_ID
    assert link["callback_url"] == CALLBACK_URL
    # No My Domain is collected or sent: Composio's page finds the org.
    assert "connection_data" not in link

    response = browser.get(db, _verify_url(composio))

    assert response.status_code == 200, response.body
    (identity,) = _identity_reads(composio)
    assert identity["endpoint"] == f"https://{ORG_HOST}/services/oauth2/userinfo"
    assert identity["method"] == "GET"
    assert identity["parameters"] == []
    # The host came from the account verification already read.
    account_reads = [
        path
        for verb, path, _ in composio.calls
        if verb == "GET" and path.startswith("/connected_accounts/")
    ]
    assert account_reads == [f"/connected_accounts/{composio.created[-1]}"]
    (grant,) = _grants(db)
    assert (grant.provider, grant.provider_user_id) == (
        "salesforce",
        SALESFORCE_USER_ID,
    )
    assert grant.token_type == "composio"
    assert grant.access_token == composio.created[-1]
    assert db.query(ActorOAuthFlowState).count() == 0
    assert _flow_cookies(browser) == []


def _native_salesforce_grant(user: User, owner: str | None, token: str) -> UserOAuth:
    return UserOAuth(
        user_id=int(user.id),
        provider="salesforce",
        resource_owner_key=owner,
        provider_user_id="005Hs00000OldOLDOLD",
        access_token=token,
        refresh_token="native-refresh-token",
        token_type="bearer",
        instance_url="https://acme.my.salesforce.com",
    )


def test_a_salesforce_connect_replaces_the_actors_native_grant(salesforce):
    """Switching the backend to Composio: the actor's next connect replaces
    its native grant. The native row is only deleted here (no Composio
    account to delete, nothing sent to Salesforce); a web grant and another
    actor's grant are left alone."""
    db, user, composio = salesforce
    db.add_all(
        [
            _native_salesforce_grant(user, OWNER, "native-actor-token"),
            _native_salesforce_grant(user, None, "native-web-token"),
            _native_salesforce_grant(user, OTHER_OWNER, "native-other-token"),
        ]
    )
    db.commit()

    response = _connect(db, user, composio)

    assert response.status_code == 200, response.body
    rows = _grants(db)
    grants = {grant.resource_owner_key: grant for grant in rows}
    assert len(rows) == 3
    assert set(grants) == {OWNER, None, OTHER_OWNER}
    new = grants[OWNER]
    assert (new.token_type, new.access_token, new.provider_user_id) == (
        "composio",
        composio.created[-1],
        SALESFORCE_USER_ID,
    )
    assert new.refresh_token is None
    assert grants[None].access_token == "native-web-token"
    assert grants[OTHER_OWNER].access_token == "native-other-token"
    # The replaced grant was native: no Composio account to delete. (The
    # fixture fails any direct call to Salesforce, so none was revoked.)
    assert composio.paths("DELETE") == []


@pytest.mark.parametrize(
    "host",
    [
        "acme--uat.sandbox.my.salesforce.com",
        "mindful-raccoon-9pmrsm-dev-ed.trailblaze.my.salesforce.com",
        "ACME.My.Salesforce.com",
        "acme.my.salesforce.com.",
    ],
    ids=["sandbox", "playground", "mixed-case", "trailing-dot"],
)
def test_the_identity_is_read_on_whichever_org_host_the_account_names(salesforce, host):
    db, user, composio = salesforce
    composio.salesforce_data = {"base_url": f"https://{host}/services/data/v67.0"}

    response = _connect(db, user, composio)

    assert response.status_code == 200, response.body
    (identity,) = _identity_reads(composio)
    expected_host = host.lower().rstrip(".")
    assert identity["endpoint"] == f"https://{expected_host}/services/oauth2/userinfo"


@pytest.mark.parametrize(
    "data",
    [
        {"base_url": "http://acme.my.salesforce.com/services/data/v67.0"},
        {"base_url": "https://sneaky:pw@acme.my.salesforce.com/services/data"},
        {"base_url": "https://acme.my.salesforce.com:443/services/data/v67.0"},
        {"base_url": "https://acme.my.salesforce.com:bad/services/data"},
        {"base_url": "https://acme.example.com/services/data/v67.0"},
        {"base_url": "https://acme.salesforce.com.example.com/services/data"},
        {"base_url": "https://salesforce.com/services/data/v67.0"},
        {"base_url": "https://evilsalesforce.com/services/data/v67.0"},
        {"base_url": "https://acme my.salesforce.com/services/data"},
        {"base_url": "https://[acme.my.salesforce.com/services/data"},
        {"base_url": ""},
        {"base_url": 42},
        {"subdomain": "login"},
        None,
    ],
    ids=[
        "http",
        "userinfo",
        "port",
        "bad-port",
        "foreign",
        "look-alike",
        "bare-domain",
        "suffix-only",
        "space",
        "unparsable",
        "empty",
        "not-a-string",
        "missing",
        "no-data",
    ],
)
def test_an_unusable_org_host_is_refused_without_echoing_it(salesforce, data, caplog):
    db, user, composio = salesforce
    composio.salesforce_data = data
    caplog.set_level(logging.DEBUG)

    response = _connect(db, user, composio)

    assert response.status_code == 400
    assert _identity_reads(composio) == []
    assert composio.paths("DELETE") == [f"/connected_accounts/{composio.created[-1]}"]
    assert _grants(db) == []
    text = response.body.decode()
    raw = data.get("base_url") if isinstance(data, dict) else None
    if isinstance(raw, str) and raw:
        # The host part, even where the URL itself can't be parsed, and the
        # bare hostname without credentials, port or brackets.
        host = raw.split("://", 1)[-1].split("/", 1)[0]
        hostname = host.rsplit("@", 1)[-1].split(":", 1)[0].strip("[")
        for shown in (raw, raw.strip(), host, hostname, "sneaky"):
            assert shown not in caplog.text
            assert shown not in text
    assert "no usable Salesforce host" in caplog.text
    assert "reason=identity_host_invalid" in caplog.text
    _assert_org_host_help(text)


def _assert_org_host_help(text: str) -> None:
    """The page for an account on no usable org host. Connecting again the
    same way can't fix it, so it says what to change on Composio's page."""
    assert "Salesforce wasn't connected" in text
    assert (
        "select Connect again, and on Composio's page change the Salesforce "
        "subdomain to your My Domain prefix (for example acme.my, or "
        "acme.develop.my for a Developer Edition org)."
    ) in text
    assert (
        "For a sandbox org, change the subdomain on Composio's page to test, "
        "or to your sandbox's My Domain prefix (for example "
        "acme--uat.sandbox.my)."
    ) in text
    # Not the generic page for a sign-in that could not be verified.
    assert "could not be verified" not in text


def test_the_host_is_not_read_from_params_or_sub(salesforce):
    """The runtime (#3669) reads ``data.base_url`` only, so the connect does
    too: a grant it could not run is not stored."""
    db, user, composio = salesforce
    composio.salesforce_data = {"subdomain": "login"}
    composio.account_overrides = {"params": {"base_url": BASE_URL}}

    response = _connect(db, user, composio)

    assert response.status_code == 400
    assert _identity_reads(composio) == []
    assert _grants(db) == []
    _assert_org_host_help(response.body.decode())


@pytest.mark.parametrize(
    "user_id",
    [
        "005Hs00000AbCdE",  # the 15-character form
        "001Hs00000AbCdEFGH",  # an Account id, not a user
        "005Hs00000AbCdEFGHX",
        "005Hs00000AbCd-FGH",
        "",
        12345,
        None,
    ],
    ids=["15-char", "not-a-user", "too-long", "symbol", "empty", "number", "none"],
)
def test_a_malformed_salesforce_user_id_is_refused(salesforce, user_id):
    db, user, composio = salesforce
    composio.userinfo = {**composio.userinfo, "user_id": user_id}

    response = _connect(db, user, composio)

    assert response.status_code == 503
    assert composio.paths("DELETE") == [f"/connected_accounts/{composio.created[-1]}"]
    assert _grants(db) == []


def test_the_identity_is_never_taken_from_sub(salesforce):
    db, user, composio = salesforce
    composio.userinfo = {"sub": composio.userinfo["sub"]}

    response = _connect(db, user, composio)

    assert response.status_code == 503
    assert _grants(db) == []


# --- Connect guidance -----------------------------------------------------------
#
# An org admin has to install Composio's connected app before anyone else in
# the org can connect, and nothing comes back when they haven't: Salesforce
# stops on its own error page. So Toby explains it before the popup opens and
# again when it closes without a connection (#3646).


def _catalog_entry(db: Session, app_id: str) -> dict[str, Any]:
    from xagent.web.mcp_apps import get_app_by_id

    entry = get_app_by_id(db, app_id)
    assert entry is not None
    return entry


def test_the_catalog_carries_connect_guidance_while_salesforce_is_on_composio(
    salesforce,
):
    db, _user, _composio = salesforce

    guidance = _catalog_entry(db, "salesforce")["connect_guidance"]

    assert set(guidance) == {"before_connect", "not_connected"}
    for text in guidance.values():
        assert "Connected Apps OAuth Usage" in text
        assert "All users may self-authorize" in text
        assert "acme.my" in text
        assert "Use Custom Domain" in text
        # A sandbox signs in through test.salesforce.com, not "login".
        assert (
            "For a sandbox org, change the subdomain on Composio's page to "
            "test, or to your sandbox's My Domain prefix (for example "
            "acme--uat.sandbox.my)."
        ) in text
    assert "admins" in guidance["before_connect"]
    # Only after a failed attempt: the two ways it fails without saying why.
    assert "mistyped" in guidance["not_connected"]
    assert "Please check your username and password" in guidance["not_connected"]


@pytest.mark.parametrize("backend", [None, "native", "other"])
def test_no_connect_guidance_while_salesforce_is_native(
    salesforce, monkeypatch, backend
):
    db, _user, _composio = salesforce
    if backend is None:
        monkeypatch.delenv("XAGENT_SALESFORCE_AUTH_BACKEND")
    else:
        monkeypatch.setenv("XAGENT_SALESFORCE_AUTH_BACKEND", backend)

    assert "connect_guidance" not in _catalog_entry(db, "salesforce")


def test_other_apps_on_composio_carry_no_connect_guidance(salesforce):
    """Facebook's switch is on here too; its entry is unchanged."""
    db, _user, _composio = salesforce
    from xagent.web.mcp_apps import get_all_mcp_apps

    entries = {entry["id"]: entry for entry in get_all_mcp_apps(db)}

    assert "connect_guidance" in entries["salesforce"]
    assert [
        app_id for app_id, entry in entries.items() if "connect_guidance" in entry
    ] == ["salesforce"]
