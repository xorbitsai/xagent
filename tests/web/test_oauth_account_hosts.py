"""Per-account OAuth endpoints (#2928): providers whose authorize and token
endpoints live on the customer's own account host, e.g.
``{subdomain}.zendesk.com``.

The registry ships empty -- each provider registers itself in its own
change -- so every test here registers a fake provider entry.
"""

import base64
import dataclasses
import json
import socket
from datetime import datetime, timedelta, timezone
from http.cookies import SimpleCookie
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.parse import parse_qs, urlparse

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from tests.shared.oauth_browser_fixture import create_access_token
from xagent.core.utils.encryption import encrypt_value
from xagent.web import mcp_apps, oauth_account_hosts
from xagent.web.api import auth as auth_api
from xagent.web.api.auth import generic_oauth_login, verify_token
from xagent.web.models.actor_oauth_flow import ActorOAuthFlowState
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.oauth_provider import OAuthProvider
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.oauth_account_hosts import (
    OAuthAccountError,
    OAuthAccountHost,
    get_oauth_account_host,
    resolve_oauth_account_endpoints,
)
from xagent.web.tools import config as tool_config

FAKE_PROVIDER = "acmedesk"
FAKE_HOST = OAuthAccountHost(
    host_suffix="acmedesk.example",
    authorize_path="/oauth/authorize",
    token_path="/oauth/token",
    userinfo_path="/api/me",
    input_label="Acmedesk subdomain",
    input_placeholder="your-company",
    input_help="The part before .acmedesk.example in your account URL.",
)
PUBLIC_ADDRESS = "93.184.216.34"


def _resolve_to(address: str):
    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]

    return fake_getaddrinfo


@pytest.fixture()
def fake_provider(monkeypatch):
    monkeypatch.setitem(
        oauth_account_hosts.OAUTH_ACCOUNT_HOSTS, FAKE_PROVIDER, FAKE_HOST
    )
    monkeypatch.setattr(
        oauth_account_hosts.socket, "getaddrinfo", _resolve_to(PUBLIC_ADDRESS)
    )
    return FAKE_HOST


def test_endpoints_are_built_on_the_account_host(fake_provider):
    endpoints = resolve_oauth_account_endpoints(fake_provider, "acme")

    assert endpoints.account == "acme"
    assert endpoints.authorize_url == "https://acme.acmedesk.example/oauth/authorize"
    assert endpoints.token_url == "https://acme.acmedesk.example/oauth/token"
    assert endpoints.userinfo_url == "https://acme.acmedesk.example/api/me"


def test_account_is_trimmed_and_lowercased(fake_provider):
    endpoints = resolve_oauth_account_endpoints(fake_provider, "  Acme-Support\n")

    assert endpoints.account == "acme-support"
    assert endpoints.token_url == "https://acme-support.acmedesk.example/oauth/token"


@pytest.mark.parametrize(
    "account",
    [
        None,
        "",
        "   ",
        123,
        "acme.evil.example",
        "https://acme.acmedesk.example",
        "acme/path",
        "acme:443",
        "user@acme",
        "-acme",
        "acme-",
        "ac_me",
        "a" * 64,
    ],
)
def test_account_that_is_not_a_single_dns_label_is_rejected(fake_provider, account):
    with pytest.raises(OAuthAccountError):
        resolve_oauth_account_endpoints(fake_provider, account)


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.5", "169.254.169.254", "::1"])
def test_account_host_resolving_to_a_private_address_is_rejected(
    fake_provider, monkeypatch, address
):
    monkeypatch.setattr(oauth_account_hosts.socket, "getaddrinfo", _resolve_to(address))

    with pytest.raises(OAuthAccountError):
        resolve_oauth_account_endpoints(fake_provider, "acme")


def test_account_host_is_resolved_on_the_composed_hostname(fake_provider, monkeypatch):
    looked_up = []

    def recording_getaddrinfo(host, port, *args, **kwargs):
        looked_up.append((host, port))
        return _resolve_to(PUBLIC_ADDRESS)(host, port)

    monkeypatch.setattr(
        oauth_account_hosts.socket, "getaddrinfo", recording_getaddrinfo
    )

    resolve_oauth_account_endpoints(fake_provider, "acme")

    assert looked_up == [("acme.acmedesk.example", 443)]


def test_account_host_that_does_not_resolve_is_rejected(fake_provider, monkeypatch):
    def failing_getaddrinfo(host, port, *args, **kwargs):
        raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")

    monkeypatch.setattr(oauth_account_hosts.socket, "getaddrinfo", failing_getaddrinfo)

    with pytest.raises(OAuthAccountError):
        resolve_oauth_account_endpoints(fake_provider, "acme")


def test_account_host_with_no_resolved_addresses_is_rejected(
    fake_provider, monkeypatch
):
    monkeypatch.setattr(
        oauth_account_hosts.socket, "getaddrinfo", lambda *args, **kwargs: []
    )

    with pytest.raises(OAuthAccountError):
        resolve_oauth_account_endpoints(fake_provider, "acme")


@pytest.mark.parametrize(
    "key",
    [
        "grant_type",
        "code",
        "redirect_uri",
        "code_verifier",
        "refresh_token",
        "client_id",
        "client_secret",
        "scope",
    ],
)
def test_token_params_cannot_override_standard_token_request_fields(key):
    # The code exchange and refresh merge token_params at different points,
    # so letting it name a standard field would make them disagree.
    with pytest.raises(ValueError, match=key):
        dataclasses.replace(FAKE_HOST, token_params={key: "x"})


def test_lookup_is_case_insensitive_and_unknown_providers_have_no_host(fake_provider):
    assert get_oauth_account_host("AcmeDesk") is fake_provider
    assert get_oauth_account_host("github") is None
    assert get_oauth_account_host(None) is None


def test_variant_rows_match_the_longest_registered_provider_name(
    fake_provider, monkeypatch
):
    regional = dataclasses.replace(fake_provider, host_suffix="eu.acmedesk.example")
    monkeypatch.setitem(
        oauth_account_hosts.OAUTH_ACCOUNT_HOSTS, f"{FAKE_PROVIDER}-eu", regional
    )

    assert get_oauth_account_host("acmedesk-eu-sandbox") is regional
    assert get_oauth_account_host("acmedesk-sandbox") is fake_provider


def test_dash_anchored_variant_rows_share_the_base_providers_host(fake_provider):
    assert get_oauth_account_host("acmedesk-sandbox") is fake_provider
    assert get_oauth_account_host("ACMEDESK-Sandbox") is fake_provider
    assert get_oauth_account_host("acmedesklite") is None


# ---------- login entry ------------------------------------------------------

FAKE_APP_ID = "acmedesk-app"
FAKE_APP_EXECUTION = {
    "name": "Acmedesk",
    "transport": "oauth",
    "provider_name": FAKE_PROVIDER,
    "oauth_scopes": ["tickets:read"],
    "launch_config": {"command": "acmedesk"},
}
# A catalog app on a provider that has no account host.
PLAIN_APP_ID = "plain-app"
PLAIN_APP_EXECUTION = {
    "name": "Plain",
    "transport": "oauth",
    "provider_name": "custom",
    "oauth_scopes": ["tickets:read"],
    "launch_config": {"command": "plain"},
}
ACTOR_OWNER = "toby:slack:41:UALICE"


@pytest.fixture()
def oauth_db(tmp_path, monkeypatch, fake_provider):
    registry_lookup = mcp_apps.get_builtin_execution_fields_and_optional_scopes

    def test_registry(app_id: str):
        if app_id == FAKE_APP_ID:
            return FAKE_APP_EXECUTION, []
        if app_id == PLAIN_APP_ID:
            return PLAIN_APP_EXECUTION, []
        return registry_lookup(app_id)

    monkeypatch.setattr(
        mcp_apps, "get_builtin_execution_fields_and_optional_scopes", test_registry
    )
    engine = create_engine(f"sqlite:///{tmp_path / 'account-oauth.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with factory() as db:
        user = User(username="alice", password_hash="hash")
        db.add(user)
        db.add(
            PublicMCPApp(
                app_id=FAKE_APP_ID,
                name="Acmedesk",
                description="Acmedesk",
                transport="oauth",
                provider_name=FAKE_PROVIDER,
                oauth_scopes=["tickets:read"],
                launch_config={"command": "acmedesk"},
                is_visible_in_connector=True,
            )
        )
        db.add(
            PublicMCPApp(
                app_id=PLAIN_APP_ID,
                name="Plain",
                description="Plain",
                transport="oauth",
                provider_name="custom",
                oauth_scopes=["tickets:read"],
                launch_config={"command": "plain"},
                is_visible_in_connector=True,
            )
        )
        db.commit()
        db.refresh(user)
        yield db, user
    engine.dispose()


def _db_provider(provider_name: str = FAKE_PROVIDER) -> SimpleNamespace:
    """Static endpoints an account provider must never use."""
    return SimpleNamespace(
        provider_name=provider_name,
        client_id=encrypt_value("client-id"),
        client_secret=encrypt_value("client-secret"),
        auth_url="https://static.example/authorize",
        token_url="https://static.example/token",
        userinfo_url="https://static.example/me",
        redirect_uri=f"https://xagent.example/api/auth/{provider_name}/callback",
        default_scopes=[],
        user_id_path="id",
        email_path="email",
    )


def _token_for(user: User) -> str:
    return create_access_token(
        data={"sub": user.username, "type": "access"},
        expires_delta=timedelta(minutes=5),
    )


def _login(db: Session, user: User, *, provider: str = FAKE_PROVIDER, **kwargs):
    return generic_oauth_login(
        provider,
        token=_token_for(user),
        app_id=kwargs.pop("app_id", FAKE_APP_ID),
        redirect=None,
        db=db,
        db_provider=_db_provider(provider),
        **kwargs,
    )


def _state_payload(response) -> dict:
    state = parse_qs(urlparse(response.headers["location"]).query)["state"][0]
    payload = verify_token(state)
    assert payload is not None
    return payload


def test_login_redirects_to_the_account_authorize_endpoint(oauth_db):
    db, user = oauth_db

    response = _login(db, user, account="Acme")

    assert response.status_code == 307
    location = urlparse(response.headers["location"])
    assert f"{location.scheme}://{location.netloc}{location.path}" == (
        "https://acme.acmedesk.example/oauth/authorize"
    )
    params = parse_qs(location.query)
    assert params["client_id"] == ["client-id"]
    assert params["scope"] == ["tickets:read"]
    assert _state_payload(response)["oauth_account"] == "acme"


@pytest.mark.parametrize("account", [None, "", "acme.evil.example"])
def test_login_without_a_valid_account_is_rejected_before_any_redirect(
    oauth_db, account
):
    db, user = oauth_db

    response = _login(db, user, account=account)

    assert response.status_code == 400
    assert "location" not in response.headers


def test_login_rejects_an_account_resolving_to_a_private_address(oauth_db, monkeypatch):
    db, user = oauth_db
    monkeypatch.setattr(
        oauth_account_hosts.socket, "getaddrinfo", _resolve_to("10.0.0.5")
    )

    response = _login(db, user, account="acme")

    assert response.status_code == 400
    assert "location" not in response.headers


def test_account_error_page_escapes_its_message():
    response = auth_api._oauth_account_error_response("<script>alert(1)</script>")

    assert response.status_code == 400
    assert b"&lt;script&gt;" in response.body
    assert b"<script>" not in response.body


def test_login_rejects_an_account_for_a_provider_without_account_hosts(oauth_db):
    db, user = oauth_db

    response = _login(db, user, provider="custom", app_id=None, account="acme")

    assert response.status_code == 400
    assert "location" not in response.headers


def test_login_for_a_provider_without_account_hosts_is_unchanged(oauth_db):
    db, user = oauth_db

    response = _login(db, user, provider="custom", app_id=None)

    assert response.status_code == 307
    location = urlparse(response.headers["location"])
    assert f"{location.scheme}://{location.netloc}{location.path}" == (
        "https://static.example/authorize"
    )
    assert "oauth_account" not in _state_payload(response)


def test_login_route_forwards_the_account_query_parameter(oauth_db, monkeypatch):
    db, user = oauth_db
    captured = {}

    def fake_generic_oauth_login(
        provider, token, app_id, redirect, db, db_provider, account=None
    ):
        captured["account"] = account
        return "redirect"

    monkeypatch.setattr(auth_api, "generic_oauth_login", fake_generic_oauth_login)
    db.add(
        OAuthProvider(
            provider_name=FAKE_PROVIDER,
            name="Acmedesk",
            client_id="client-id",
            client_secret="client-secret",
            auth_url="",
            token_url="",
        )
    )
    db.commit()

    result = auth_api.oauth_login(
        FAKE_PROVIDER,
        token="t",
        app_id=FAKE_APP_ID,
        redirect=None,
        account="acme",
        db=db,
    )

    assert result == "redirect"
    assert captured["account"] == "acme"


def _actor_link(
    db: Session,
    user: User,
    *,
    provider: str = FAKE_PROVIDER,
    app_id: str = FAKE_APP_ID,
) -> None:
    server = MCPServer(
        name="Acmedesk" if app_id == FAKE_APP_ID else "Plain",
        description="Acmedesk" if app_id == FAKE_APP_ID else "Plain",
        managed="external",
        transport="oauth",
        auth={"app_id": app_id, "provider": provider},
    )
    db.add(server)
    db.flush()
    db.add(
        UserMCPServer(
            user_id=int(user.id),
            mcpserver_id=int(server.id),
            is_owner=False,
            is_active=True,
        )
    )
    db.commit()


def test_actor_login_binds_the_account_too(oauth_db):
    db, user = oauth_db
    _actor_link(db, user)

    response = auth_api.start_builtin_oauth_for_resource_owner(
        provider=FAKE_PROVIDER,
        app_id=FAKE_APP_ID,
        user=user,
        resource_owner_key=ACTOR_OWNER,
        db=db,
        db_provider=_db_provider(),
        account="acme",
    )

    assert response.status_code == 307
    assert response.headers["location"].startswith(
        "https://acme.acmedesk.example/oauth/authorize?"
    )
    assert _state_payload(response)["oauth_account"] == "acme"


@pytest.mark.parametrize("account", [None, "acme.evil.example"])
def test_actor_login_without_a_valid_account_starts_no_flow(oauth_db, account):
    db, user = oauth_db
    _actor_link(db, user)

    response = auth_api.start_builtin_oauth_for_resource_owner(
        provider=FAKE_PROVIDER,
        app_id=FAKE_APP_ID,
        user=user,
        resource_owner_key=ACTOR_OWNER,
        db=db,
        db_provider=_db_provider(),
        account=account,
    )
    db.commit()

    assert response.status_code == 400
    assert "set-cookie" not in response.headers
    assert db.query(ActorOAuthFlowState).count() == 0


def test_actor_login_with_an_account_for_a_provider_without_account_hosts_starts_no_flow(
    oauth_db,
):
    db, user = oauth_db
    _actor_link(db, user, provider="custom", app_id=PLAIN_APP_ID)

    response = auth_api.start_builtin_oauth_for_resource_owner(
        provider="custom",
        app_id=PLAIN_APP_ID,
        user=user,
        resource_owner_key=ACTOR_OWNER,
        db=db,
        db_provider=_db_provider("custom"),
        account="acme",
    )
    db.commit()

    assert response.status_code == 400
    assert "set-cookie" not in response.headers
    assert db.query(ActorOAuthFlowState).count() == 0


def test_actor_login_rejects_an_account_resolving_to_a_private_address(
    oauth_db, monkeypatch
):
    db, user = oauth_db
    _actor_link(db, user)
    monkeypatch.setattr(
        oauth_account_hosts.socket, "getaddrinfo", _resolve_to("10.0.0.5")
    )

    response = auth_api.start_builtin_oauth_for_resource_owner(
        provider=FAKE_PROVIDER,
        app_id=FAKE_APP_ID,
        user=user,
        resource_owner_key=ACTOR_OWNER,
        db=db,
        db_provider=_db_provider(),
        account="acme",
    )
    db.commit()

    assert response.status_code == 400
    assert "set-cookie" not in response.headers
    assert db.query(ActorOAuthFlowState).count() == 0


# ---------- callback ---------------------------------------------------------


class _ProviderResponse:
    def __init__(self, data: dict, status_code: int = 200) -> None:
        self._data = data
        self.status_code = status_code
        self.text = ""

    def json(self) -> dict:
        return self._data


def _mock_provider_http(monkeypatch, token_data: dict | None = None):
    post = Mock(
        return_value=_ProviderResponse(
            token_data
            or {
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "expires_in": 1800,
                "scope": "tickets:read",
            }
        )
    )
    get = Mock(return_value=_ProviderResponse({"id": 42, "email": "a@example.com"}))
    monkeypatch.setattr(auth_api.requests, "post", post)
    monkeypatch.setattr(auth_api.requests, "get", get)
    return post, get


def _callback(
    db: Session,
    state: str,
    provider: str = FAKE_PROVIDER,
    cookies: dict[str, str] | None = None,
):
    request = SimpleNamespace(
        query_params={"state": state, "code": "code"},
        cookies=cookies or {},
    )
    from tests.shared.oauth_browser_fixture import generic_oauth_callback

    return generic_oauth_callback(
        provider, request, db=db, db_provider=_db_provider(provider)
    )


def _state_for(user: User, provider: str = FAKE_PROVIDER, **extra) -> str:
    return create_access_token(
        data={
            "type": "oauth_state",
            "user_id": int(user.id),
            "provider": provider,
            "app_id": FAKE_APP_ID if provider == FAKE_PROVIDER else None,
            "redirect": None,
            **extra,
        },
        expires_delta=timedelta(minutes=10),
    )


def _grant(db: Session, user: User, provider: str = FAKE_APP_ID) -> UserOAuth | None:
    return (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == provider)
        .one_or_none()
    )


def test_callback_exchanges_the_code_on_the_account_bound_at_login(
    oauth_db, monkeypatch
):
    db, user = oauth_db
    post, get = _mock_provider_http(monkeypatch)
    start = _login(db, user, account="acme")
    state = parse_qs(urlparse(start.headers["location"]).query)["state"][0]
    cookie = SimpleCookie()
    cookie.load(start.headers["set-cookie"])
    cookies = {name: morsel.value for name, morsel in cookie.items()}

    response = _callback(db, state, cookies=cookies)

    assert response.status_code == 200
    assert post.call_args.args[0] == "https://acme.acmedesk.example/oauth/token"
    assert post.call_args.kwargs["allow_redirects"] is False
    assert get.call_args.args[0] == "https://acme.acmedesk.example/api/me"
    assert get.call_args.kwargs["allow_redirects"] is False
    grant = _grant(db, user)
    assert grant is not None
    assert grant.access_token == "new-access"
    assert grant.instance_url == "acme"
    replay = _callback(db, state + "=", cookies=cookies)
    assert replay.status_code == 400
    assert post.call_count == 1


def test_callback_keeps_the_bound_account_over_a_token_response_instance_url(
    oauth_db, monkeypatch
):
    db, user = oauth_db
    _mock_provider_http(
        monkeypatch,
        {"access_token": "new-access", "instance_url": "https://evil.example"},
    )

    response = _callback(db, _state_for(user, oauth_account="acme"))

    assert response.status_code == 200
    assert _grant(db, user).instance_url == "acme"


@pytest.mark.parametrize("extra", [{}, {"oauth_account": "acme.evil.example"}])
def test_callback_without_a_valid_bound_account_is_rejected_without_network(
    oauth_db, monkeypatch, extra
):
    db, user = oauth_db
    post, get = _mock_provider_http(monkeypatch)

    response = _callback(db, _state_for(user, **extra))

    assert response.status_code == 400
    post.assert_not_called()
    get.assert_not_called()
    assert _grant(db, user) is None


def test_callback_rejects_a_bound_account_that_now_resolves_privately(
    oauth_db, monkeypatch
):
    db, user = oauth_db
    post, _get = _mock_provider_http(monkeypatch)
    monkeypatch.setattr(
        oauth_account_hosts.socket, "getaddrinfo", _resolve_to("10.0.0.5")
    )

    response = _callback(db, _state_for(user, oauth_account="acme"))

    assert response.status_code == 400
    post.assert_not_called()


def test_callback_rejects_an_account_bound_for_a_provider_without_account_hosts(
    oauth_db, monkeypatch
):
    db, user = oauth_db
    post, _get = _mock_provider_http(monkeypatch)

    response = _callback(db, _state_for(user, "custom", oauth_account="acme"), "custom")

    assert response.status_code == 400
    post.assert_not_called()


def test_callback_without_a_userinfo_endpoint_persists_no_grant(oauth_db, monkeypatch):
    db, user = oauth_db
    monkeypatch.setitem(
        oauth_account_hosts.OAUTH_ACCOUNT_HOSTS,
        FAKE_PROVIDER,
        dataclasses.replace(FAKE_HOST, userinfo_path=None),
    )
    _post, get = _mock_provider_http(monkeypatch)

    response = _callback(db, _state_for(user, oauth_account="acme"))

    assert response.status_code == 400
    get.assert_not_called()
    assert _grant(db, user) is None


def test_callback_with_a_nested_userinfo_identity_persists_no_grant(
    oauth_db, monkeypatch
):
    db, user = oauth_db
    _post, get = _mock_provider_http(monkeypatch)
    get.return_value = _ProviderResponse({"user": {"id": 1}})

    response = _callback(db, _state_for(user, oauth_account="acme"))

    assert response.status_code == 400
    assert _grant(db, user) is None


def _login_state(db: Session, user: User, account: str) -> tuple[str, dict[str, str]]:
    response = _login(db, user, account=account)
    location = response.headers["location"]
    cookie = SimpleCookie()
    cookie.load(response.headers["set-cookie"])
    return (
        parse_qs(urlparse(location).query)["state"][0],
        {name: morsel.value for name, morsel in cookie.items()},
    )


def test_callback_ignores_an_account_in_the_callback_query_string(
    oauth_db, monkeypatch
):
    db, user = oauth_db
    post, _get = _mock_provider_http(monkeypatch)
    state, cookies = _login_state(db, user, "acme")
    request = SimpleNamespace(
        query_params={
            "state": state,
            "code": "code",
            "oauth_account": "evil",
            "account": "evil",
        },
        cookies=cookies,
    )

    response = auth_api.generic_oauth_callback(
        FAKE_PROVIDER, request, db=db, db_provider=_db_provider()
    )

    assert response.status_code == 200
    assert post.call_args.args[0] == "https://acme.acmedesk.example/oauth/token"
    assert _grant(db, user).instance_url == "acme"


def test_callback_rejects_a_state_whose_account_was_tampered_with(
    oauth_db, monkeypatch
):
    db, user = oauth_db
    post, _get = _mock_provider_http(monkeypatch)
    state, cookies = _login_state(db, user, "acme")
    header, body, signature = state.split(".")
    claims = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    claims["oauth_account"] = "evil"
    forged_body = (
        base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    )

    response = _callback(db, f"{header}.{forged_body}.{signature}", cookies=cookies)

    assert response.status_code == 400
    post.assert_not_called()
    assert _grant(db, user) is None


def test_reconnecting_to_another_account_replaces_account_and_tokens_together(
    oauth_db, monkeypatch
):
    db, user = oauth_db
    _mock_provider_http(monkeypatch)
    state, cookies = _login_state(db, user, "acme")
    assert _callback(db, state, cookies=cookies).status_code == 200
    _mock_provider_http(
        monkeypatch, {"access_token": "beta-access", "refresh_token": "beta-refresh"}
    )

    state, cookies = _login_state(db, user, "beta")
    assert _callback(db, state, cookies=cookies).status_code == 200

    grants = db.query(UserOAuth).filter(UserOAuth.user_id == user.id).all()
    assert [(g.instance_url, g.access_token, g.refresh_token) for g in grants] == [
        ("beta", "beta-access", "beta-refresh")
    ]


def test_actor_callback_persists_the_account_bound_at_actor_login(
    oauth_db, monkeypatch
):
    db, user = oauth_db
    _actor_link(db, user)
    post, _get = _mock_provider_http(monkeypatch)
    start = auth_api.start_builtin_oauth_for_resource_owner(
        provider=FAKE_PROVIDER,
        app_id=FAKE_APP_ID,
        user=user,
        resource_owner_key=ACTOR_OWNER,
        db=db,
        db_provider=_db_provider(),
        account="acme",
    )
    db.commit()
    cookies = SimpleCookie()
    cookies.load(start.headers["set-cookie"])
    request = SimpleNamespace(
        query_params={
            "state": parse_qs(urlparse(start.headers["location"]).query)["state"][0],
            "code": "code",
        },
        cookies={name: morsel.value for name, morsel in cookies.items()},
    )

    response = auth_api.generic_oauth_callback(
        FAKE_PROVIDER, request, db=db, db_provider=_db_provider()
    )

    assert response.status_code == 200
    assert post.call_args.args[0] == "https://acme.acmedesk.example/oauth/token"
    grant = (
        db.query(UserOAuth).filter(UserOAuth.resource_owner_key == ACTOR_OWNER).one()
    )
    assert grant.instance_url == "acme"


# ---------- refresh ----------------------------------------------------------


def _expired_grant(db: Session, user: User, instance_url: str | None) -> UserOAuth:
    db.add(
        OAuthProvider(
            provider_name=FAKE_PROVIDER,
            name="Acmedesk",
            client_id=encrypt_value("client-id"),
            client_secret=encrypt_value("client-secret"),
            auth_url="https://static.example/authorize",
            token_url="https://static.example/token",
            redirect_uri="https://xagent.example/api/auth/acmedesk/callback",
            default_scopes=[],
        )
    )
    grant = UserOAuth(
        user_id=user.id,
        provider=FAKE_APP_ID,
        access_token="old-access",
        refresh_token="old-refresh",
        instance_url=instance_url,
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
    )
    db.add(grant)
    db.commit()
    return grant


def _fake_async_client(monkeypatch, response_data: dict | None = None) -> list:
    posted: list = []

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, **kwargs):
            posted.append((url, kwargs))
            return _ProviderResponse(
                response_data
                or {"access_token": "new-access", "refresh_token": "new-refresh"}
            )

    monkeypatch.setattr(tool_config.httpx, "AsyncClient", FakeAsyncClient)
    return posted


@pytest.mark.asyncio
async def test_refresh_posts_to_the_stored_account_token_endpoint(
    oauth_db, monkeypatch
):
    db, user = oauth_db
    grant = _expired_grant(db, user, "acme")
    posted = _fake_async_client(monkeypatch)

    assert await tool_config.refresh_oauth_token_if_needed(db, grant, FAKE_PROVIDER)

    assert [url for url, _ in posted] == ["https://acme.acmedesk.example/oauth/token"]
    assert posted[0][1]["data"]["refresh_token"] == "old-refresh"
    assert posted[0][1]["follow_redirects"] is False
    assert grant.access_token == "new-access"
    assert grant.refresh_token == "new-refresh"


@pytest.mark.asyncio
async def test_refresh_keeps_the_bound_account_over_a_response_instance_url(
    oauth_db, monkeypatch
):
    db, user = oauth_db
    grant = _expired_grant(db, user, "acme")
    _fake_async_client(
        monkeypatch,
        {"access_token": "new-access", "instance_url": "https://evil.example"},
    )

    assert await tool_config.refresh_oauth_token_if_needed(db, grant, FAKE_PROVIDER)

    assert grant.instance_url == "acme"


@pytest.mark.asyncio
@pytest.mark.parametrize("instance_url", [None, "", "https://evil.example"])
async def test_refresh_without_a_valid_stored_account_is_permanent_and_offline(
    oauth_db, monkeypatch, instance_url
):
    db, user = oauth_db
    grant = _expired_grant(db, user, instance_url)
    posted = _fake_async_client(monkeypatch)

    with pytest.raises(tool_config._OAuthRefreshPermanentlyInvalid):
        await tool_config.refresh_oauth_token_if_needed(db, grant, FAKE_PROVIDER)

    assert posted == []
    assert grant.access_token == "old-access"


@pytest.mark.asyncio
async def test_refresh_is_transient_when_the_account_host_resolves_privately(
    oauth_db, monkeypatch
):
    db, user = oauth_db
    grant = _expired_grant(db, user, "acme")
    posted = _fake_async_client(monkeypatch)
    monkeypatch.setattr(
        oauth_account_hosts.socket, "getaddrinfo", _resolve_to("10.0.0.5")
    )

    assert (
        await tool_config.refresh_oauth_token_if_needed(db, grant, FAKE_PROVIDER)
        is False
    )

    assert posted == []
    assert grant.access_token == "old-access"
    assert grant.instance_url == "acme"


# ---------- catalog ----------------------------------------------------------


def test_catalog_app_declares_the_account_input_it_needs(oauth_db):
    db, _user = oauth_db

    app = mcp_apps.get_app_by_id(db, FAKE_APP_ID)

    assert app is not None
    assert app["oauth_account_input"] == {
        "label": "Acmedesk subdomain",
        "placeholder": "your-company",
        "help": "The part before .acmedesk.example in your account URL.",
        "suffix": ".acmedesk.example",
    }


def test_catalog_app_without_account_hosts_has_no_account_input(oauth_db):
    db, _user = oauth_db
    db.add(
        PublicMCPApp(
            app_id="plain-oauth",
            name="Plain",
            description="Plain",
            transport="oauth",
            provider_name="custom",
            launch_config={"command": "plain"},
            is_visible_in_connector=True,
        )
    )
    db.add(
        PublicMCPApp(
            app_id="acmedesk-key",
            name="Acmedesk key",
            description="Key-based app under the same provider name",
            transport="stdio",
            provider_name=FAKE_PROVIDER,
            launch_config={"command": "acmedesk", "required_env": ["ACME_KEY"]},
            is_visible_in_connector=True,
        )
    )
    db.commit()

    assert "oauth_account_input" not in mcp_apps.get_app_by_id(db, "plain-oauth")
    assert "oauth_account_input" not in mcp_apps.get_app_by_id(db, "acmedesk-key")
