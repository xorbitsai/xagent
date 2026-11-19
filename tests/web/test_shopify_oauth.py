"""Shopify per-account OAuth (#2810): the registered store host, the signed
callback checks made before the code exchange, the granted-scope check, the
shop identity, and refresh on the stored store label.
"""

import hashlib
import hmac
import logging
import socket
from datetime import datetime, timedelta, timezone
from http.cookies import SimpleCookie
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette.datastructures import QueryParams

from tests.shared.oauth_browser_fixture import create_access_token
from xagent.core.utils.encryption import encrypt_value
from xagent.web import oauth_account_hosts, shopify_oauth
from xagent.web.api import auth as auth_api
from xagent.web.api.auth import generic_oauth_login
from xagent.web.models.database import Base
from xagent.web.models.oauth_provider import OAuthProvider
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.oauth_account_hosts import (
    get_oauth_account_host,
    resolve_oauth_account_endpoints,
)
from xagent.web.oauth_provider_quirks import requires_json_accept_header
from xagent.web.shopify_oauth import (
    ShopifyOAuthError,
    fetch_shopify_shop_identity,
    require_shopify_scopes,
    verify_shopify_callback,
)
from xagent.web.tools import config as tool_config
from xagent.web.tools.mcp import shopify as shopify_connector

CLIENT_SECRET = "client-secret"
PUBLIC_ADDRESS = "23.227.38.65"
TEST_APP_ID = "shopify"
SHOP_GID = "gid://shopify/Shop/548380009"
REQUIRED = ["read_customers", "write_orders", "write_products"]
TOKEN_URL = "https://acme.myshopify.com/admin/oauth/access_token"
GRAPHQL_URL = "https://acme.myshopify.com/admin/api/2026-07/graphql.json"


def _resolve_to(address: str):
    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]

    return fake_getaddrinfo


def _sign(params: dict[str, str], secret: str = CLIENT_SECRET) -> str:
    message = "&".join(f"{key}={value}" for key, value in sorted(params.items()))
    return hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()


def _signed_query(params: dict[str, str], secret: str = CLIENT_SECRET) -> QueryParams:
    return QueryParams({**params, "hmac": _sign(params, secret)})


class _Response:
    def __init__(self, data: object, status_code: int = 200) -> None:
        self._data = data
        self.status_code = status_code
        self.text = ""

    def json(self) -> object:
        if isinstance(self._data, Exception):
            raise self._data
        return self._data


@pytest.fixture(autouse=True)
def _public_dns(monkeypatch):
    monkeypatch.setattr(
        oauth_account_hosts.socket, "getaddrinfo", _resolve_to(PUBLIC_ADDRESS)
    )


# ---------- registry -----------------------------------------------------------


def test_shopify_endpoints_live_on_the_store_host():
    host = get_oauth_account_host("shopify")
    assert host is not None

    endpoints = resolve_oauth_account_endpoints(host, "Acme")

    assert endpoints.account == "acme"
    assert endpoints.authorize_url == "https://acme.myshopify.com/admin/oauth/authorize"
    assert endpoints.token_url == TOKEN_URL
    # Shopify has no userinfo endpoint; the callback's Shopify branch fetches
    # the shop identity instead.
    assert endpoints.userinfo_url is None
    assert host.input_metadata()["suffix"] == ".myshopify.com"


def test_shopify_requests_expiring_offline_tokens():
    host = get_oauth_account_host("shopify")

    assert host is not None
    assert dict(host.token_params) == {"expiring": "1"}


def test_variant_rows_ask_shopify_for_json_too():
    """A "-"-anchored variant row shares the account host, so it shares every
    other Shopify step of the flow as well."""
    assert requires_json_accept_header("shopify")
    assert requires_json_accept_header("Shopify-Sandbox")
    assert not requires_json_accept_header("shopifylite")


def test_required_scopes_and_api_version_match_the_connector():
    assert shopify_oauth.SHOPIFY_REQUIRED_SCOPES == (
        shopify_connector.SHOPIFY_REQUIRED_ADMIN_SCOPES
    )
    assert shopify_oauth.SHOPIFY_API_VERSION == shopify_connector.SHOPIFY_API_VERSION


# ---------- callback signature -------------------------------------------------


def test_known_shopify_hmac_vector_verifies():
    """The worked example from Shopify's OAuth docs (client secret "hush")."""
    query = QueryParams(
        "code=0907a61c0c8d55e99db179b68161bc00"
        "&hmac=700e2dadb827fcc8609e9d5ce208b2e9cdaab9df07390d2cbca10d7c328fc4bf"
        "&shop=some-shop.myshopify.com&state=0.6784241404160823"
        "&timestamp=1337178173"
    )

    verify_shopify_callback(
        query.multi_items(), client_secret="hush", bound_account="some-shop"
    )


def _escaped_signature(params: dict[str, str]) -> str:
    """Sign as Shopify's Python library does: escape % and & in keys and
    values, and = in keys, before joining (shopify.Session.calculate_hmac)."""

    def escape(text: str) -> str:
        return text.replace("%", "%25").replace("&", "%26")

    message = "&".join(
        sorted(
            f"{escape(k).replace('=', '%3D')}={escape(v)}" for k, v in params.items()
        )
    )
    return hmac.new(
        CLIENT_SECRET.encode(), message.encode(), hashlib.sha256
    ).hexdigest()


def test_delimiters_in_signed_parameters_are_escaped_as_shopify_does():
    params = {**_callback_params(), "note": "50%&more", "a=b": "c"}

    verify_shopify_callback(
        [*params.items(), ("hmac", _escaped_signature(params))],
        client_secret=CLIENT_SECRET,
        bound_account="acme",
    )
    with pytest.raises(ShopifyOAuthError):
        verify_shopify_callback(
            [*params.items(), ("hmac", _sign(params))],
            client_secret=CLIENT_SECRET,
            bound_account="acme",
        )


def _callback_params(**overrides: str) -> dict[str, str]:
    return {
        "code": "auth-code",
        "host": "YWRtaW4uc2hvcGlmeS5jb20vc3RvcmUvYWNtZQ",
        "shop": "acme.myshopify.com",
        "state": "signed.state.token",
        "timestamp": "1760000000",
        **overrides,
    }


def test_signed_callback_for_the_bound_store_verifies():
    query = _signed_query(_callback_params())

    verify_shopify_callback(
        query.multi_items(), client_secret=CLIENT_SECRET, bound_account="acme"
    )


def test_shop_is_compared_case_insensitively():
    query = _signed_query(_callback_params(shop="Acme.MyShopify.com"))

    verify_shopify_callback(
        query.multi_items(), client_secret=CLIENT_SECRET, bound_account="acme"
    )


@pytest.mark.parametrize(
    ("items", "reason"),
    [
        pytest.param(
            list(_callback_params().items()),
            "hmac_missing",
            id="missing-hmac",
        ),
        pytest.param(
            [*_callback_params().items(), ("hmac", "")],
            "hmac_missing",
            id="blank-hmac",
        ),
        pytest.param(
            [*_callback_params().items(), ("hmac", _sign(_callback_params(), "x"))],
            "hmac_invalid",
            id="other-secret",
        ),
        pytest.param(
            [
                *_callback_params(code="swapped").items(),
                ("hmac", _sign(_callback_params())),
            ],
            "hmac_invalid",
            id="tampered-param",
        ),
        pytest.param(
            [*_callback_params().items(), ("hmac", "é" * 64)],
            "hmac_invalid",
            id="non-ascii-hmac",
        ),
        pytest.param(
            [
                *_callback_params().items(),
                # The same value again, so the signature itself still matches
                # and only the repeated-parameter rule can reject it.
                ("code", "auth-code"),
                ("hmac", _sign(_callback_params())),
            ],
            "hmac_invalid",
            id="repeated-param",
        ),
        pytest.param(
            # host folded into code: "code=auth-code&host=..." would join to
            # the same unescaped string as the two signed parameters.
            [
                *(
                    (key, value)
                    for key, value in _callback_params(
                        code=f"auth-code&host={_callback_params()['host']}"
                    ).items()
                    if key != "host"
                ),
                ("hmac", _sign(_callback_params())),
            ],
            "hmac_invalid",
            id="delimiter-in-value",
        ),
        pytest.param(
            list(_signed_query(_callback_params(shop="other.myshopify.com")).items()),
            "shop_mismatch",
            id="other-store",
        ),
        pytest.param(
            list(
                _signed_query(
                    _callback_params(shop="acme.myshopify.com.attacker.example")
                ).items()
            ),
            "shop_mismatch",
            id="suffixed-store",
        ),
        pytest.param(
            list(
                _signed_query(
                    {k: v for k, v in _callback_params().items() if k != "shop"}
                ).items()
            ),
            "shop_mismatch",
            id="missing-shop",
        ),
    ],
)
def test_unverifiable_callbacks_are_rejected(items, reason):
    with pytest.raises(ShopifyOAuthError) as caught:
        verify_shopify_callback(
            items, client_secret=CLIENT_SECRET, bound_account="acme"
        )

    assert caught.value.reason == reason


# ---------- granted scopes -----------------------------------------------------


@pytest.mark.parametrize(
    "scope",
    [
        pytest.param("write_products, write_orders,read_customers", id="comma-list"),
        pytest.param(["write_products", " write_orders", "read_customers"], id="list"),
        pytest.param(
            "write_products,write_orders,write_customers", id="write-for-read"
        ),
    ],
)
def test_required_scopes_are_granted(scope):
    granted = require_shopify_scopes(scope)

    # Stored space-separated, the format the scope column's readers split.
    assert "," not in granted
    assert {"write_products", "write_orders"} <= set(granted.split(" "))


@pytest.mark.parametrize(
    ("scope", "missing"),
    [
        pytest.param(
            "read_products,write_orders",
            "read_customers, write_products",
            id="read-for-write",
        ),
        pytest.param(None, "read_customers, write_orders, write_products", id="none"),
    ],
)
def test_missing_scopes_are_listed(scope, missing):
    with pytest.raises(ShopifyOAuthError) as caught:
        require_shopify_scopes(scope)

    assert caught.value.reason == "scope_missing"
    assert missing in caught.value.message


# ---------- shop identity ------------------------------------------------------


def _mock_post(monkeypatch, response: _Response) -> list:
    calls: list = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return response

    monkeypatch.setattr(shopify_oauth.requests, "post", fake_post)
    return calls


def _shop_payload(**shop: object) -> dict:
    return {
        "data": {
            "shop": {"id": SHOP_GID, "myshopifyDomain": "acme.myshopify.com", **shop}
        }
    }


def test_identity_is_the_shop_gid_read_with_the_admin_api_header(monkeypatch):
    calls = _mock_post(monkeypatch, _Response(_shop_payload()))

    assert fetch_shopify_shop_identity("acme", "shpat-token") == SHOP_GID

    url, kwargs = calls[0]
    assert url == GRAPHQL_URL
    assert kwargs["headers"]["X-Shopify-Access-Token"] == "shpat-token"
    assert "Authorization" not in kwargs["headers"]
    assert kwargs["allow_redirects"] is False


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        pytest.param(_Response({}, status_code=401), "identity_failed", id="401"),
        pytest.param(_Response(ValueError("not json")), "identity_failed", id="html"),
        pytest.param(_Response(["not", "an", "object"]), "identity_failed", id="list"),
        pytest.param(
            _Response({"errors": [{"message": "Access denied"}]}),
            "identity_failed",
            id="graphql-errors",
        ),
        pytest.param(
            _Response(_shop_payload(id="gid://shopify/Customer/1")),
            "identity_failed",
            id="not-a-shop-gid",
        ),
        pytest.param(
            _Response(_shop_payload(myshopifyDomain="other.myshopify.com")),
            "shop_mismatch",
            id="other-store",
        ),
    ],
)
def test_unusable_identity_responses_are_rejected(monkeypatch, response, reason):
    _mock_post(monkeypatch, response)

    with pytest.raises(ShopifyOAuthError) as caught:
        fetch_shopify_shop_identity("acme", "shpat-token")

    assert caught.value.reason == reason
    assert "shpat-token" not in caught.value.message


@pytest.mark.parametrize(
    ("response", "logged"),
    [
        pytest.param(_Response({}, status_code=401), "status 401", id="401"),
        pytest.param(
            _Response(
                {
                    "errors": [
                        {
                            "message": "Access denied for shpat-token",
                            "extensions": {"code": "ACCESS_DENIED"},
                        },
                        {"message": "no code"},
                    ]
                }
            ),
            "codes=ACCESS_DENIED",
            id="graphql-errors",
        ),
    ],
)
def test_identity_failures_log_the_status_and_error_codes_only(
    monkeypatch, caplog, response, logged
):
    """The live store check needs to know why Shopify refused, without the
    response body or the token ever reaching the log."""
    _mock_post(monkeypatch, response)

    with caplog.at_level(logging.WARNING, logger=shopify_oauth.__name__):
        with pytest.raises(ShopifyOAuthError):
            fetch_shopify_shop_identity("acme", "shpat-token")

    assert logged in caplog.text
    assert "shpat-token" not in caplog.text
    assert "Access denied" not in caplog.text


def test_unreachable_identity_endpoint_does_not_echo_the_error(monkeypatch):
    def failing_post(url, **kwargs):
        raise shopify_oauth.requests.ConnectionError(f"boom {kwargs['headers']}")

    monkeypatch.setattr(shopify_oauth.requests, "post", failing_post)

    with pytest.raises(ShopifyOAuthError) as caught:
        fetch_shopify_shop_identity("acme", "shpat-token")

    assert caught.value.reason == "identity_unreachable"
    assert "shpat-token" not in caught.value.message


# ---------- login and callback -------------------------------------------------


def seed_shopify_oauth(db) -> User:
    """Add the user, the Shopify OAuth catalog row and the provider row."""
    user = User(username="alice", password_hash="hash")
    db.add(user)
    # The catalog row ships in a later change; this stands in for it.
    db.add(
        PublicMCPApp(
            app_id=TEST_APP_ID,
            name="Shopify",
            description="Shopify test app",
            transport="oauth",
            provider_name="shopify",
            oauth_scopes=REQUIRED,
            launch_config={
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.shopify"],
                "env_mapping": {
                    "SHOPIFY_ACCESS_TOKEN": "access_token",
                    "SHOPIFY_STORE_DOMAIN": "instance_url",
                },
            },
            is_visible_in_connector=True,
        )
    )
    db.add(
        OAuthProvider(
            provider_name="shopify",
            name="Shopify",
            client_id=encrypt_value("client-id"),
            client_secret=encrypt_value(CLIENT_SECRET),
            auth_url="",
            token_url="",
            redirect_uri="https://xagent.example/api/auth/shopify/callback",
            userinfo_url="",
            default_scopes=[],
        )
    )
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture()
def shopify_db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'shopify-oauth.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with factory() as db:
        yield db, seed_shopify_oauth(db)
    engine.dispose()


def _db_provider(db, provider: str = "shopify") -> OAuthProvider:
    return db.query(OAuthProvider).filter_by(provider_name=provider).one()


def _login(
    db,
    user,
    account: str = "acme",
    provider: str = "shopify",
    app_id: str = TEST_APP_ID,
):
    token = create_access_token(
        data={
            "sub": user.username,
            "user_id": user.id,
            "session_epoch": user.session_epoch,
            "type": "access",
        },
        expires_delta=timedelta(minutes=5),
    )
    return generic_oauth_login(
        provider,
        token=token,
        app_id=app_id,
        redirect=None,
        db=db,
        db_provider=_db_provider(db, provider),
        account=account,
    )


def _login_state(
    db, user, account: str = "acme", **login: str
) -> tuple[str, dict[str, str]]:
    """The state and the browser cookie the login binds it to."""
    response = _login(db, user, account, **login)
    cookie = SimpleCookie()
    cookie.load(response.headers["set-cookie"])
    return (
        parse_qs(urlparse(response.headers["location"]).query)["state"][0],
        {name: morsel.value for name, morsel in cookie.items()},
    )


def _token_response(**overrides: object) -> dict:
    return {
        "access_token": "shpat-access",
        "scope": "write_products,write_orders,read_customers",
        "expires_in": 3600,
        "refresh_token": "shprt-refresh",
        "refresh_token_expires_in": 7776000,
        **overrides,
    }


def _mock_shopify_http(
    monkeypatch, token: dict | None = None, identity: dict | None = None
) -> list:
    """Route the code exchange and the identity query, recording both."""
    calls: list = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        if url == TOKEN_URL:
            return _Response(token if token is not None else _token_response())
        if url == GRAPHQL_URL:
            return _Response(identity if identity is not None else _shop_payload())
        raise AssertionError(f"unexpected POST {url}")

    # auth.py and shopify_oauth.py share the one requests module.
    monkeypatch.setattr(auth_api.requests, "post", fake_post)
    return calls


def _callback(db, flow: tuple[str, dict[str, str]], **overrides: str):
    state, cookies = flow
    params = _callback_params(state=state, **overrides)
    request = SimpleNamespace(query_params=_signed_query(params), cookies=cookies)
    return auth_api.generic_oauth_callback(
        "shopify", request, db=db, db_provider=_db_provider(db)
    )


def _grant(db, user) -> UserOAuth | None:
    return (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "shopify")
        .one_or_none()
    )


def test_login_sends_the_merchant_to_the_stores_consent_page(shopify_db):
    db, user = shopify_db

    response = _login(db, user, "Acme")

    location = urlparse(response.headers["location"])
    assert (location.scheme, location.netloc, location.path) == (
        "https",
        "acme.myshopify.com",
        "/admin/oauth/authorize",
    )
    # Shopify's authorize endpoint takes a comma-separated scope list.
    assert parse_qs(location.query)["scope"] == [",".join(REQUIRED)]


def test_callback_stores_the_store_bound_grant(shopify_db, monkeypatch):
    db, user = shopify_db
    calls = _mock_shopify_http(monkeypatch)

    response = _callback(db, _login_state(db, user))

    assert response.status_code == 200
    (token_url, exchange), (identity_url, _identity) = calls
    assert token_url == TOKEN_URL
    assert exchange["data"]["code"] == "auth-code"
    assert exchange["data"]["expiring"] == "1"
    assert exchange["headers"]["Accept"] == "application/json"
    assert exchange["allow_redirects"] is False
    assert identity_url == GRAPHQL_URL
    grant = _grant(db, user)
    assert grant is not None
    assert grant.provider_user_id == SHOP_GID
    assert grant.instance_url == "acme"
    assert grant.access_token == "shpat-access"
    assert grant.refresh_token == "shprt-refresh"
    assert grant.expires_at is not None
    assert grant.scope == "read_customers write_orders write_products"


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"shop": "other.myshopify.com"}, id="other-store"),
        pytest.param({"code": "forged"}, id="unsigned-change"),
    ],
)
def test_unverifiable_callback_never_reaches_the_token_endpoint(
    shopify_db, monkeypatch, overrides
):
    db, user = shopify_db
    calls = _mock_shopify_http(monkeypatch)
    # The browser's own cookie is sent, so only the signature can fail.
    state, cookies = _login_state(db, user)
    params = _callback_params(state=state)
    query = _signed_query(params)
    if "code" in overrides:
        # Re-sign nothing: the changed code no longer matches the signature.
        query = QueryParams({**dict(query), **overrides})
    else:
        query = _signed_query({**params, **overrides})
    request = SimpleNamespace(query_params=query, cookies=cookies)

    response = auth_api.generic_oauth_callback(
        "shopify", request, db=db, db_provider=_db_provider(db)
    )

    assert response.status_code == 400
    # Shopify's own refusal, not an earlier state, cookie or account one.
    assert b"Cannot connect Shopify" in response.body
    assert calls == []
    assert _grant(db, user) is None


def test_a_replayed_callback_never_reaches_the_token_endpoint(shopify_db, monkeypatch):
    """#2810 acceptance: a replayed state fails. The first callback spends it,
    so the same signed callback from the same browser is refused before any
    second code exchange."""
    db, user = shopify_db
    calls = _mock_shopify_http(monkeypatch)
    flow = _login_state(db, user)
    assert _callback(db, flow).status_code == 200
    exchanges = [url for url, _kwargs in calls].count(TOKEN_URL)

    response = _callback(db, flow)

    assert response.status_code == 400
    assert [url for url, _kwargs in calls].count(TOKEN_URL) == exchanges


def test_callback_missing_a_required_scope_stores_nothing(shopify_db, monkeypatch):
    db, user = shopify_db
    calls = _mock_shopify_http(
        monkeypatch, token=_token_response(scope="write_products,read_orders")
    )

    response = _callback(db, _login_state(db, user))

    assert response.status_code == 400
    assert b"read_customers" in response.body
    assert b"write_orders" in response.body
    assert b"shpat-access" not in response.body
    assert [url for url, _kwargs in calls] == [TOKEN_URL]
    assert _grant(db, user) is None


def test_callback_for_a_token_of_another_store_stores_nothing(shopify_db, monkeypatch):
    db, user = shopify_db
    calls = _mock_shopify_http(
        monkeypatch, identity=_shop_payload(myshopifyDomain="other.myshopify.com")
    )

    response = _callback(db, _login_state(db, user))

    assert response.status_code == 400
    # Refused by the identity check, after the exchange and the shop query.
    assert b"Error verifying the connected account" in response.body
    assert [url for url, _kwargs in calls] == [TOKEN_URL, GRAPHQL_URL]
    assert _grant(db, user) is None


@pytest.mark.parametrize("failure", ["missing-scope", "other-store"])
def test_a_rejected_reconnect_leaves_the_existing_grant_intact(
    shopify_db, monkeypatch, failure
):
    """The scope and identity checks run before the delete-then-recreate
    write, so a refused reconnect never costs the working grant."""
    db, user = shopify_db
    db.add(
        UserOAuth(
            user_id=user.id,
            provider="shopify",
            access_token="kept-access",
            refresh_token="kept-refresh",
            instance_url="acme",
            provider_user_id=SHOP_GID,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
    )
    db.commit()
    if failure == "missing-scope":
        _mock_shopify_http(monkeypatch, token=_token_response(scope="write_products"))
    else:
        _mock_shopify_http(
            monkeypatch, identity=_shop_payload(myshopifyDomain="other.myshopify.com")
        )

    response = _callback(db, _login_state(db, user))

    assert response.status_code == 400
    db.expire_all()
    grant = _grant(db, user)
    assert grant is not None
    assert (grant.access_token, grant.refresh_token) == ("kept-access", "kept-refresh")


def test_a_shopify_variant_provider_row_gets_the_same_signature_check(
    shopify_db, monkeypatch
):
    """An admin-created "shopify-*" provider takes the store-host path, so it
    cannot skip the callback signature check."""
    db, user = shopify_db
    db.add(
        OAuthProvider(
            provider_name="shopify-acme",
            name="Shopify Acme",
            client_id=encrypt_value("client-id"),
            client_secret=encrypt_value(CLIENT_SECRET),
            auth_url="",
            token_url="",
            redirect_uri="https://xagent.example/api/auth/shopify-acme/callback",
            userinfo_url="",
            default_scopes=[],
        )
    )
    db.add(
        PublicMCPApp(
            app_id="shopify-acme",
            name="Shopify Acme",
            description="Shopify variant test app",
            transport="oauth",
            provider_name="shopify-acme",
            oauth_scopes=REQUIRED,
            launch_config={
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.shopify"],
                "env_mapping": {
                    "SHOPIFY_ACCESS_TOKEN": "access_token",
                    "SHOPIFY_STORE_DOMAIN": "instance_url",
                },
            },
            is_visible_in_connector=True,
        )
    )
    db.commit()
    calls = _mock_shopify_http(monkeypatch)
    state, cookies = _login_state(
        db, user, provider="shopify-acme", app_id="shopify-acme"
    )
    signed = _signed_query(_callback_params(state=state))
    forged = QueryParams({**dict(signed), "code": "forged"})

    response = auth_api.generic_oauth_callback(
        "shopify-acme",
        SimpleNamespace(query_params=forged, cookies=cookies),
        db=db,
        db_provider=_db_provider(db, "shopify-acme"),
    )

    assert response.status_code == 400
    assert b"Cannot connect Shopify" in response.body
    assert calls == []


# ---------- refresh ------------------------------------------------------------


@pytest.mark.asyncio
async def test_refresh_rotates_on_the_stored_store(shopify_db, monkeypatch):
    db, user = shopify_db
    grant = UserOAuth(
        user_id=user.id,
        provider="shopify",
        access_token="old-access",
        refresh_token="old-refresh",
        instance_url="acme",
        provider_user_id=SHOP_GID,
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
    )
    db.add(grant)
    db.commit()
    posted: list = []

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, **kwargs):
            posted.append((url, kwargs))
            return _Response(
                {
                    "access_token": "rotated-access",
                    "refresh_token": "rotated-refresh",
                    "expires_in": 3600,
                }
            )

    monkeypatch.setattr(tool_config, "create_public_async_client", FakeAsyncClient)

    assert await tool_config.refresh_oauth_token_if_needed(db, grant, "shopify")

    url, kwargs = posted[0]
    assert url == TOKEN_URL
    assert kwargs["data"]["grant_type"] == "refresh_token"
    assert kwargs["data"]["refresh_token"] == "old-refresh"
    # Refresh asks for an expiring token again, like the code exchange.
    assert kwargs["data"]["expiring"] == "1"
    assert kwargs["follow_redirects"] is False
    assert kwargs["headers"]["Accept"] == "application/json"
    assert grant.access_token == "rotated-access"
    assert grant.refresh_token == "rotated-refresh"
    assert grant.instance_url == "acme"


# ---------- one grant per store ------------------------------------------------
#
# Shopify keeps one current offline token per app and store: a new grant
# retires the store's other refresh tokens, and a grant and a refresh running
# together retire each other's result. So a store connected by one owner
# refuses everyone else before the code exchange, and the exchange and every
# refresh for a store take the same per-store lock.


def _other_user(db, username: str = "bob") -> User:
    other = User(username=username, password_hash="hash")
    db.add(other)
    db.commit()
    db.refresh(other)
    return other


def _held_grant(
    db,
    user,
    *,
    store: str = "acme",
    oauth_provider: str | None = "shopify",
    access_token: str = "held-access",
    resource_owner_key: str | None = None,
    expires_in: timedelta | None = timedelta(hours=1),
    connector: str = TEST_APP_ID,
) -> UserOAuth:
    grant = UserOAuth(
        user_id=user.id,
        provider=connector,
        oauth_provider=oauth_provider,
        resource_owner_key=resource_owner_key,
        access_token=access_token,
        refresh_token="held-refresh",
        instance_url=store,
        provider_user_id=SHOP_GID,
        expires_at=None
        if expires_in is None
        else datetime.now(timezone.utc) + expires_in,
    )
    db.add(grant)
    db.commit()
    return grant


def test_a_store_connected_by_another_user_is_refused_before_the_exchange(
    shopify_db, monkeypatch
):
    db, user = shopify_db
    bob = _other_user(db)
    _held_grant(db, bob)
    calls = _mock_shopify_http(monkeypatch)

    response = _callback(db, _login_state(db, user))

    assert response.status_code == 409
    assert b"already connected by another user" in response.body
    assert b"bob" not in response.body
    # Refused before the exchange: no new grant retires bob's token.
    assert calls == []
    assert _grant(db, user) is None
    db.expire_all()
    held = _grant(db, bob)
    assert held is not None
    assert held.access_token == "held-access"
    # The refusal releases the store at once.
    assert _probe_store_lock(db) is True


def test_the_owner_can_reconnect_their_store(shopify_db, monkeypatch):
    db, user = shopify_db
    _held_grant(db, user, access_token="old-access")
    _mock_shopify_http(monkeypatch)

    response = _callback(db, _login_state(db, user))

    assert response.status_code == 200
    db.expire_all()
    grant = _grant(db, user)
    assert grant is not None
    assert grant.access_token == "shpat-access"


@pytest.mark.parametrize(
    "held",
    [
        # Disconnecting deletes the grant; a dead refresh token clears it.
        pytest.param({"access_token": ""}, id="cleared-after-failed-refresh"),
        pytest.param({"store": "other"}, id="another-store"),
        # A variant provider row is another Shopify app, with its own token.
        pytest.param({"oauth_provider": "shopify-acme"}, id="another-app"),
        # A Composio-held grant records no issuer; its token is not ours.
        pytest.param({"oauth_provider": None}, id="no-issuer"),
        # A refresh token lasts at most 90 days from the last token issued:
        # an owner who stopped using the store cannot refresh any more.
        pytest.param(
            {"expires_in": -timedelta(days=90, minutes=1)},
            id="past-the-refresh-window",
        ),
    ],
)
def test_a_store_not_held_by_a_live_grant_of_this_app_can_be_connected(
    shopify_db, monkeypatch, held
):
    db, user = shopify_db
    _held_grant(db, _other_user(db), **held)
    _mock_shopify_http(monkeypatch)

    response = _callback(db, _login_state(db, user))

    assert response.status_code == 200
    assert _grant(db, user) is not None


@pytest.mark.parametrize(
    "expires_in",
    [
        pytest.param(-timedelta(days=89), id="within-the-refresh-window"),
        # A token without an expiry never lapses.
        pytest.param(None, id="non-expiring"),
    ],
)
def test_an_expired_access_token_still_holds_the_store_within_the_refresh_window(
    shopify_db, monkeypatch, expires_in
):
    db, user = shopify_db
    _held_grant(db, _other_user(db), expires_in=expires_in)
    calls = _mock_shopify_http(monkeypatch)

    response = _callback(db, _login_state(db, user))

    assert response.status_code == 409
    assert calls == []


def _actor_login_state(db, user, owner: str = "toby:w1:u1"):
    """Start the store's OAuth for an actor, as Toby does."""
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app
    from xagent.web.mcp_apps import ensure_builtin_oauth_server_definition
    from xagent.web.models.mcp import UserMCPServer

    # An actor flow starts only from the registry's own catalog row.
    registry_row = get_builtin_public_mcp_app(TEST_APP_ID)
    assert registry_row is not None
    app = db.query(PublicMCPApp).filter_by(app_id=TEST_APP_ID).one()
    for field, value in registry_row.items():
        setattr(app, field, value)
    app.is_visible_in_connector = True
    db.commit()
    server = ensure_builtin_oauth_server_definition(db, app_id=TEST_APP_ID)
    db.add(
        UserMCPServer(
            user_id=int(user.id),
            mcpserver_id=int(server.id),
            is_owner=False,
            is_active=True,
        )
    )
    db.commit()
    response = auth_api.start_builtin_oauth_for_resource_owner(
        provider="shopify",
        app_id=TEST_APP_ID,
        user=user,
        resource_owner_key=owner,
        redirect="https://app.example/connectors",
        db=db,
        db_provider=_db_provider(db),
        account="acme",
    )
    db.commit()
    cookie = SimpleCookie()
    cookie.load(response.headers["set-cookie"])
    return (
        parse_qs(urlparse(response.headers["location"]).query)["state"][0],
        {name: morsel.value for name, morsel in cookie.items()},
    )


def test_an_actor_connecting_a_store_held_by_someone_else_is_refused(
    shopify_db, monkeypatch
):
    db, user = shopify_db
    _held_grant(db, _other_user(db))
    calls = _mock_shopify_http(monkeypatch)

    response = _callback(db, _actor_login_state(db, user))

    assert response.status_code == 409
    assert b"already connected by another user" in response.body
    assert calls == []
    assert (
        db.query(UserOAuth).filter(UserOAuth.resource_owner_key == "toby:w1:u1").count()
        == 0
    )
    assert _probe_store_lock(db) is True


def test_a_store_lock_timeout_asks_to_try_again(shopify_db, monkeypatch):
    """Another callback or refresh holds the store past the database's lock
    or busy timeout: nothing was exchanged, so the merchant can retry."""
    from sqlalchemy.exc import OperationalError

    db, user = shopify_db
    calls = _mock_shopify_http(monkeypatch)

    def busy(session, **kwargs):
        raise OperationalError("lock", {}, Exception("database is locked"))

    monkeypatch.setattr(auth_api, "lock_shopify_store", busy)

    response = _callback(db, _login_state(db, user))

    assert response.status_code == 503
    assert b"start connecting the store again" in response.body
    assert response.headers["cache-control"] == "no-store"
    assert calls == []
    assert _grant(db, user) is None


@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [
        pytest.param({"token": {"error": "invalid_request"}}, 400, id="exchange-error"),
        pytest.param(
            {"token": _token_response(scope="write_products")},
            400,
            id="missing-scope",
        ),
    ],
)
def test_a_callback_that_fails_after_locking_releases_the_store(
    shopify_db, monkeypatch, failure, expected_status
):
    db, user = shopify_db
    _mock_shopify_http(monkeypatch, **failure)

    response = _callback(db, _login_state(db, user))

    assert response.status_code == expected_status
    assert _grant(db, user) is None
    assert _probe_store_lock(db) is True


def test_another_owner_of_the_same_account_holds_the_store_too(shopify_db):
    """An actor (e.g. a Toby member) and the account's own connection are
    separate owners with separate tokens, so either one holds the store
    against the other."""
    from xagent.web.services.shopify_store_locks import shopify_store_holder

    db, user = shopify_db
    _held_grant(db, user, resource_owner_key="toby:w1:u1")

    def holder_against(resource_owner_key, connector_key=TEST_APP_ID):
        return shopify_store_holder(
            db,
            provider="shopify",
            store="acme",
            user_id=user.id,
            resource_owner_key=resource_owner_key,
            connector_key=connector_key,
        )

    assert holder_against(None) == "another_owner"
    assert holder_against("toby:w1:u2") == "another_owner"
    assert holder_against("toby:w1:u1") is None
    # The same owner through another connector of this Shopify app would hold
    # a second token that retires the first.
    assert holder_against("toby:w1:u1", "shopify-catalog-copy") == (
        "another_connection"
    )


def test_a_store_held_by_the_same_owner_through_another_connector_is_refused(
    shopify_db, monkeypatch
):
    db, user = shopify_db
    _held_grant(db, user, connector="shopify-catalog-copy")
    calls = _mock_shopify_http(monkeypatch)

    response = _callback(db, _login_state(db, user))

    assert response.status_code == 409
    assert b"another connection" in response.body
    assert b"another user" not in response.body
    assert calls == []


def _probe_store_lock(db) -> bool:
    """Whether another connection could take the store lock right now."""
    from sqlalchemy.exc import OperationalError

    from xagent.web.services.shopify_store_locks import lock_shopify_store

    engine = create_engine(db.get_bind().url, connect_args={"timeout": 0.1})
    try:
        with sessionmaker(bind=engine)() as other:
            try:
                lock_shopify_store(other, provider="shopify", store="acme")
            except OperationalError:
                return False
            other.rollback()
            return True
    finally:
        engine.dispose()


def test_the_store_stays_locked_through_the_code_exchange(shopify_db, monkeypatch):
    """A second callback (or a refresh) for the store waits until this one has
    stored its grant, so it then sees the store as held."""
    db, user = shopify_db
    calls = _mock_shopify_http(monkeypatch)
    fake_post = auth_api.requests.post
    lock_free_during_exchange: list[bool] = []

    def exchange_while_probing(url, **kwargs):
        if url == TOKEN_URL:
            lock_free_during_exchange.append(_probe_store_lock(db))
        return fake_post(url, **kwargs)

    monkeypatch.setattr(auth_api.requests, "post", exchange_while_probing)

    response = _callback(db, _login_state(db, user))

    assert response.status_code == 200
    assert [url for url, _kwargs in calls][0] == TOKEN_URL
    assert lock_free_during_exchange == [False]
    # Released with the callback's commit.
    assert _probe_store_lock(db) is True


def _resolve_shopify(db, user_id: int):
    import asyncio

    factory = sessionmaker(bind=db.get_bind(), autoflush=False, autocommit=False)
    config = tool_config.WebToolConfig(
        db=None, request=None, db_factory=factory, user_id=user_id
    )
    return asyncio.run(
        config._resolve_legacy_oauth_access_token(
            provider_name="shopify", app_id=TEST_APP_ID
        )
    )


def _fake_refresh_client(monkeypatch, on_post=None) -> list:
    posted: list = []

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, **kwargs):
            posted.append((url, kwargs))
            if on_post is not None:
                on_post()
            return _Response(
                {
                    "access_token": "rotated-access",
                    "refresh_token": "rotated-refresh",
                    "expires_in": 3600,
                }
            )

    monkeypatch.setattr(tool_config, "create_public_async_client", FakeAsyncClient)
    return posted


def test_a_refresh_holds_the_store_lock(shopify_db, monkeypatch):
    db, user = shopify_db
    _held_grant(db, user, expires_in=-timedelta(minutes=1))
    lock_free_during_refresh: list[bool] = []
    posted = _fake_refresh_client(
        monkeypatch,
        on_post=lambda: lock_free_during_refresh.append(_probe_store_lock(db)),
    )

    resolution = _resolve_shopify(db, user.id)

    assert resolution.access_token == "rotated-access"
    assert len(posted) == 1
    assert lock_free_during_refresh == [False]


def test_a_token_that_needs_no_refresh_does_not_wait_for_the_store(
    shopify_db, monkeypatch
):
    """Only a refresh takes the store lock: a valid token is used while a
    callback or refresh for the store holds it."""
    from xagent.web.services import shopify_store_locks

    db, user = shopify_db
    _held_grant(db, user)
    posted = _fake_refresh_client(monkeypatch)
    locked: list = []
    monkeypatch.setattr(
        shopify_store_locks,
        "lock_shopify_store",
        lambda session, **kwargs: locked.append(kwargs),
    )

    resolution = _resolve_shopify(db, user.id)

    assert resolution.access_token == "held-access"
    assert locked == []
    assert posted == []


def test_a_valid_token_does_not_queue_behind_the_users_refresh(shopify_db, monkeypatch):
    """Another request of the same user holds the in-process refresh lock
    (it waits for the store): a token that needs no refresh is used at once."""
    import asyncio

    db, user = shopify_db
    _held_grant(db, user)
    posted = _fake_refresh_client(monkeypatch)
    factory = sessionmaker(bind=db.get_bind(), autoflush=False, autocommit=False)

    async def resolve_while_locked():
        async with tool_config._shopify_oauth_refresh_lock(user.id, "shopify"):
            config = tool_config.WebToolConfig(
                db=None, request=None, db_factory=factory, user_id=user.id
            )
            return await asyncio.wait_for(
                config._resolve_legacy_oauth_access_token(
                    provider_name="shopify", app_id=TEST_APP_ID
                ),
                timeout=2,
            )

    resolution = asyncio.run(resolve_while_locked())

    assert resolution.access_token == "held-access"
    assert posted == []


def test_a_token_about_to_become_due_takes_the_store_lock(shopify_db, monkeypatch):
    """A token expiring just beyond the refresh skew is not refreshed now, but
    it takes the lock: the refresh reads the clock again and may find it due."""
    from xagent.web.services import shopify_store_locks

    db, user = shopify_db
    _held_grant(db, user, expires_in=timedelta(minutes=5, seconds=30))
    posted = _fake_refresh_client(monkeypatch)
    real_lock = shopify_store_locks.lock_shopify_store
    locked: list = []

    def recording_lock(session, **kwargs):
        locked.append(kwargs["store"])
        return real_lock(session, **kwargs)

    monkeypatch.setattr(shopify_store_locks, "lock_shopify_store", recording_lock)

    resolution = _resolve_shopify(db, user.id)

    assert resolution.access_token == "held-access"
    assert locked == ["acme"]
    assert posted == []


@pytest.mark.parametrize("failure", ["transient", "permanent"])
def test_a_failed_refresh_releases_the_store(shopify_db, monkeypatch, failure):
    db, user = shopify_db
    grant = _held_grant(db, user, expires_in=-timedelta(minutes=1))
    if failure == "permanent":
        # No refresh token: the grant can never be refreshed.
        grant.refresh_token = None
        db.commit()
    posted: list = []

    class FailingAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, **kwargs):
            posted.append(url)
            return _Response({"errors": "unavailable"}, status_code=503)

    monkeypatch.setattr(tool_config, "create_public_async_client", FailingAsyncClient)

    resolution = _resolve_shopify(db, user.id)

    assert resolution.access_token is None
    assert resolution.credential_present is True
    # A transient failure is retried within the refresh; a missing refresh
    # token never reaches the provider.
    assert bool(posted) == (failure == "transient")
    assert _probe_store_lock(db) is True
    db.expire_all()
    if failure == "permanent":
        # The dead grant is cleared, so the store is free for someone else.
        assert not _grant(db, user).access_token
        flow = _login_state(db, _other_user(db))
        _mock_shopify_http(monkeypatch)
        assert _callback(db, flow).status_code == 200
    else:
        # A transient failure keeps the grant, and with it the store.
        assert _grant(db, user).access_token == "held-access"


def test_a_refresh_that_waited_for_the_store_uses_the_newer_token(
    shopify_db, monkeypatch
):
    """Another request refreshed (or the owner reconnected) while this one
    waited for the store lock: the stored token is current, so refreshing
    again would only retire it."""
    from xagent.web.services import shopify_store_locks

    db, user = shopify_db
    grant = _held_grant(db, user, expires_in=-timedelta(minutes=1))
    grant_id = grant.id
    posted = _fake_refresh_client(monkeypatch)
    real_lock = shopify_store_locks.lock_shopify_store

    def lock_after_another_refresh(session, **kwargs):
        factory = sessionmaker(bind=db.get_bind())
        with factory() as other:
            row = other.get(UserOAuth, grant_id)
            row.access_token = "newer-access"
            row.refresh_token = "newer-refresh"
            row.expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
            other.commit()
        return real_lock(session, **kwargs)

    monkeypatch.setattr(
        shopify_store_locks, "lock_shopify_store", lock_after_another_refresh
    )

    resolution = _resolve_shopify(db, user.id)

    assert resolution.access_token == "newer-access"
    assert posted == []


def test_an_actor_refresh_locks_the_store_before_its_grant(shopify_db, monkeypatch):
    """The callback locks the store, then replaces the actor's grant row; an
    actor refresh that locked the row first and then waited for the store
    would deadlock with it on PostgreSQL. The SQL order shows the refresh
    takes them the same way round."""
    import asyncio

    from sqlalchemy import event

    db, user = shopify_db
    _held_grant(
        db, user, resource_owner_key="toby:w1:u1", expires_in=-timedelta(minutes=1)
    )
    posted = _fake_refresh_client(monkeypatch)
    statements: list[str] = []

    def record(conn, cursor, statement, *args):
        statements.append(" ".join(statement.split()).lower())

    engine = db.get_bind()
    event.listen(engine, "before_cursor_execute", record)
    try:
        factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
        config = tool_config.WebToolConfig(
            db=None, request=None, db_factory=factory, user_id=user.id
        )
        resolution = asyncio.run(
            config._resolve_legacy_oauth_access_token(
                provider_name="shopify",
                app_id=TEST_APP_ID,
                resource_owner_key="toby:w1:u1",
            )
        )
    finally:
        event.remove(engine, "before_cursor_execute", record)

    assert resolution.access_token == "rotated-access"
    assert len(posted) == 1
    store_lock = next(
        i for i, s in enumerate(statements) if "into shopify_store_locks" in s
    )
    row_lock = next(
        i
        for i, s in enumerate(statements)
        if s.startswith("update user_oauth set id = id")
    )
    assert store_lock < row_lock


def refresh_two_owners_at_once(db, user, monkeypatch, *, timeout: float):
    """Two owners of one store resolve their expired grants in one event loop.

    Different owners take different in-process locks, so only the database
    store lock makes them take turns. Returns the resolutions, the most
    refreshes in flight at once, and how often the loop ticked meanwhile.
    """
    import asyncio

    bob = _other_user(db)
    _held_grant(db, user, expires_in=-timedelta(minutes=1))
    _held_grant(db, bob, expires_in=-timedelta(minutes=1))
    in_flight: list[int] = [0, 0]

    class SlowAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, **kwargs):
            in_flight[0] += 1
            in_flight[1] = max(in_flight[1], in_flight[0])
            try:
                await asyncio.sleep(0.2)
            finally:
                in_flight[0] -= 1
            return _Response(
                {
                    "access_token": "rotated-access",
                    "refresh_token": "rotated-refresh",
                    "expires_in": 3600,
                }
            )

    monkeypatch.setattr(tool_config, "create_public_async_client", SlowAsyncClient)
    factory = sessionmaker(bind=db.get_bind(), autoflush=False, autocommit=False)

    async def resolve_both():
        ticks = 0
        done = False

        async def tick():
            nonlocal ticks
            while not done:
                ticks += 1
                await asyncio.sleep(0.01)

        ticker = asyncio.create_task(tick())
        try:
            resolutions = await asyncio.wait_for(
                asyncio.gather(
                    *(
                        tool_config.WebToolConfig(
                            db=None, request=None, db_factory=factory, user_id=uid
                        )._resolve_legacy_oauth_access_token(
                            provider_name="shopify", app_id=TEST_APP_ID
                        )
                        for uid in (user.id, bob.id)
                    )
                ),
                timeout=timeout,
            )
        finally:
            done = True
            await ticker
        return resolutions, ticks

    resolutions, ticks = asyncio.run(resolve_both())
    return resolutions, in_flight[1], ticks


def test_two_owners_refreshing_one_store_take_turns_off_the_event_loop(
    shopify_db, monkeypatch
):
    """The store lock is waited for off the event loop: waiting on it there
    would stop the loop, and with it the refresh that holds the lock."""
    db, user = shopify_db

    resolutions, most_in_flight, ticks = refresh_two_owners_at_once(
        db, user, monkeypatch, timeout=4
    )

    assert [r.access_token for r in resolutions] == ["rotated-access"] * 2
    assert most_in_flight == 1
    # Two 0.2 s refreshes in turn, with the loop free throughout.
    assert ticks >= 20


@pytest.mark.parametrize(
    ("store", "expected_token", "expected_refreshes"),
    [
        pytest.param("acme", "reconnected-access", 0, id="same-store"),
        # The new grant is not under the lock this refresh holds: it releases
        # the old store, locks the new one and refreshes that grant.
        pytest.param("beta", "rotated-access", 1, id="another-store"),
    ],
)
def test_a_refresh_that_waited_through_a_reconnect_uses_the_new_grant(
    shopify_db, monkeypatch, store, expected_token, expected_refreshes
):
    """A reconnect replaces the grant row (delete, then insert under a new
    id) while a refresh waits for the store; the refresh then uses the new
    grant of that store rather than reporting no connection."""
    from xagent.web.services import shopify_store_locks

    db, user = shopify_db
    grant = _held_grant(db, user, expires_in=-timedelta(minutes=1))
    grant_id = grant.id
    # Keeps SQLite from handing the replacement row the deleted row's id.
    _held_grant(db, _other_user(db), store="other")
    posted = _fake_refresh_client(monkeypatch)
    real_lock = shopify_store_locks.lock_shopify_store

    reconnected: list[bool] = []

    def lock_after_a_reconnect(session, **kwargs):
        if reconnected:
            return real_lock(session, **kwargs)
        reconnected.append(True)
        with sessionmaker(bind=db.get_bind())() as other:
            other.delete(other.get(UserOAuth, grant_id))
            other.flush()
            other.add(
                UserOAuth(
                    user_id=user.id,
                    provider=TEST_APP_ID,
                    oauth_provider="shopify",
                    access_token="reconnected-access",
                    refresh_token="reconnected-refresh",
                    instance_url=store,
                    provider_user_id=SHOP_GID,
                    # Expired, so only a refresh could make it usable.
                    expires_at=datetime.now(timezone.utc)
                    + timedelta(hours=1 if store == "acme" else -1),
                )
            )
            other.commit()
        return real_lock(session, **kwargs)

    monkeypatch.setattr(
        shopify_store_locks, "lock_shopify_store", lock_after_a_reconnect
    )
    events, remove = _record_lock_order(db.get_bind())
    try:
        resolution = _resolve_shopify(db, user.id)
    finally:
        remove()

    # One store lock at a time: the old store is released before the new one
    # is locked, so two moving owners never wait on each other.
    _assert_one_store_per_transaction(events)
    assert resolution.access_token == expected_token
    assert len(posted) == expected_refreshes
    if expected_refreshes:
        assert f"{store}.myshopify.com" in posted[0][0]


@pytest.mark.parametrize(
    "new_owner",
    [
        pytest.param("another-user", id="another-user"),
        pytest.param("actor", id="same-account-actor"),
    ],
)
def test_a_refresh_never_takes_another_owners_grant_that_reused_its_id(
    shopify_db, monkeypatch, new_owner
):
    """SQLite hands a new row the id of a deleted row with the highest id.
    The owner disconnects while their refresh waits for the store and
    someone else connects the store; the waiting refresh must not take the
    new owner's grant by its reused id."""
    from xagent.web.services import shopify_store_locks

    db, user = shopify_db
    grant = _held_grant(db, user, expires_in=-timedelta(minutes=1))
    grant_id = grant.id
    if new_owner == "another-user":
        new_user_id, new_resource_owner_key = _other_user(db).id, None
    else:
        new_user_id, new_resource_owner_key = user.id, "toby:w1:u1"
    reused_ids: list[int] = []
    posted = _fake_refresh_client(monkeypatch)
    real_lock = shopify_store_locks.lock_shopify_store

    def lock_after_another_owner_connects(session, **kwargs):
        with sessionmaker(bind=db.get_bind())() as other:
            other.delete(other.get(UserOAuth, grant_id))
            other.flush()
            new_grant = UserOAuth(
                user_id=new_user_id,
                provider=TEST_APP_ID,
                oauth_provider="shopify",
                resource_owner_key=new_resource_owner_key,
                access_token="new-owner-access",
                refresh_token="new-owner-refresh",
                instance_url="acme",
                provider_user_id=SHOP_GID,
                expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            )
            other.add(new_grant)
            other.commit()
            reused_ids.append(new_grant.id)
        return real_lock(session, **kwargs)

    monkeypatch.setattr(
        shopify_store_locks, "lock_shopify_store", lock_after_another_owner_connects
    )

    resolution = _resolve_shopify(db, user.id)

    # The case under test: the new owner's grant took the old grant's id.
    assert reused_ids == [grant_id]
    assert resolution.access_token is None
    assert posted == []


def _record_lock_order(engine) -> tuple[list, object]:
    """Record, per connection, store locks, actor grant-row locks and the
    ends of transactions, in the order SQL runs them."""
    from sqlalchemy import event

    events: list = []

    def on_execute(conn, cursor, statement, parameters, *args):
        sql = " ".join(statement.split()).lower()
        key = id(conn.connection.dbapi_connection)
        if sql.startswith("insert into shopify_store_locks"):
            values = parameters.values() if isinstance(parameters, dict) else parameters
            events.append((key, "store", tuple(values)))
        elif sql.startswith("update user_oauth set id = id") or (
            # PostgreSQL locks the actor's grant row with SELECT ... FOR UPDATE.
            "from user_oauth" in sql and sql.endswith("for update")
        ):
            events.append((key, "row", None))

    def on_end(conn):
        events.append((id(conn.connection.dbapi_connection), "end", None))

    listeners = [
        ("before_cursor_execute", on_execute),
        ("commit", on_end),
        ("rollback", on_end),
    ]
    for name, fn in listeners:
        event.listen(engine, name, fn)

    def remove():
        for name, fn in listeners:
            event.remove(engine, name, fn)

    return events, remove


def _assert_no_new_store_lock_after_a_row_lock(events) -> None:
    """Within a transaction, every store lock comes before the grant-row
    lock; locking a store already held again is harmless."""
    state: dict = {}
    for key, kind, store in events:
        held, row_locked = state.setdefault(key, (set(), False))
        if kind == "end":
            state[key] = (set(), False)
        elif kind == "row":
            state[key] = (held, True)
        else:
            assert not row_locked or store in held, (
                f"store {store} locked after the grant row"
            )
            state[key] = (held | {store}, row_locked)


def _resolve_actor_shopify(db, user_id: int):
    import asyncio

    factory = sessionmaker(bind=db.get_bind(), autoflush=False, autocommit=False)
    config = tool_config.WebToolConfig(
        db=None, request=None, db_factory=factory, user_id=user_id
    )
    return asyncio.run(
        config._resolve_legacy_oauth_access_token(
            provider_name="shopify",
            app_id=TEST_APP_ID,
            resource_owner_key="toby:w1:u1",
        )
    )


def test_an_actor_refresh_whose_grant_moved_store_never_locks_a_store_after_it(
    shopify_db, monkeypatch
):
    """The actor reconnects to another store between the refresh reading
    the grant's store and locking the grant. Locking the new store after the
    grant would invert the callback's order (store, then grant) and can
    deadlock on PostgreSQL; the refresh starts over instead."""
    from xagent.web.services import shopify_store_locks

    db, user = shopify_db
    grant = _held_grant(
        db, user, resource_owner_key="toby:w1:u1", expires_in=-timedelta(minutes=1)
    )
    grant_id = grant.id
    posted = _fake_refresh_client(monkeypatch)
    real_lock = shopify_store_locks.lock_shopify_store
    moved: list[bool] = []

    def lock_after_a_move(session, **kwargs):
        if not moved:
            moved.append(True)
            with sessionmaker(bind=db.get_bind())() as other:
                other.get(UserOAuth, grant_id).instance_url = "beta"
                other.commit()
        return real_lock(session, **kwargs)

    monkeypatch.setattr(shopify_store_locks, "lock_shopify_store", lock_after_a_move)
    events, remove = _record_lock_order(db.get_bind())
    try:
        resolution = _resolve_actor_shopify(db, user.id)
    finally:
        remove()

    _assert_no_new_store_lock_after_a_row_lock(events)
    assert resolution.access_token == "rotated-access"
    assert len(posted) == 1
    assert "beta.myshopify.com" in posted[0][0]


def test_an_actor_grant_that_appears_after_the_store_read_is_locked_store_first(
    shopify_db, monkeypatch
):
    """The actor had no grant when the refresh read the store, and connects
    before the grant is locked: the refresh starts over and locks the new
    grant's store first."""
    from sqlalchemy import event

    db, user = shopify_db
    posted = _fake_refresh_client(monkeypatch)
    engine = db.get_bind()
    connected: list[bool] = []

    def connect_before_the_row_lock(conn, cursor, statement, *args):
        sql = " ".join(statement.split()).lower()
        if not connected and sql.startswith("update user_oauth set id = id"):
            connected.append(True)
            with sessionmaker(bind=engine)() as other:
                _held_grant(
                    other,
                    user,
                    resource_owner_key="toby:w1:u1",
                    expires_in=-timedelta(minutes=1),
                )

    event.listen(engine, "before_cursor_execute", connect_before_the_row_lock)
    events, remove = _record_lock_order(engine)
    try:
        resolution = _resolve_actor_shopify(db, user.id)
    finally:
        remove()
        event.remove(engine, "before_cursor_execute", connect_before_the_row_lock)

    assert connected == [True]
    _assert_no_new_store_lock_after_a_row_lock(events)
    assert resolution.access_token == "rotated-access"
    assert len(posted) == 1


def test_an_actor_refresh_gives_up_on_a_grant_that_keeps_changing_store(
    shopify_db, monkeypatch
):
    from xagent.web.services import shopify_store_locks

    db, user = shopify_db
    grant = _held_grant(
        db, user, resource_owner_key="toby:w1:u1", expires_in=-timedelta(minutes=1)
    )
    grant_id = grant.id
    posted = _fake_refresh_client(monkeypatch)
    real_lock = shopify_store_locks.lock_shopify_store
    peeked_stores: list[str] = []

    def lock_after_a_move(session, *, store, **kwargs):
        peeked_stores.append(store)
        with sessionmaker(bind=db.get_bind())() as other:
            row = other.get(UserOAuth, grant_id)
            row.instance_url = "beta" if row.instance_url == "acme" else "acme"
            other.commit()
        return real_lock(session, store=store, **kwargs)

    monkeypatch.setattr(shopify_store_locks, "lock_shopify_store", lock_after_a_move)

    with pytest.raises(RuntimeError, match="kept changing store"):
        _resolve_actor_shopify(db, user.id)

    assert peeked_stores == ["acme", "beta", "acme"]
    assert posted == []


def _assert_one_store_per_transaction(events) -> None:
    held: dict = {}
    for key, kind, store in events:
        if kind == "end":
            held.pop(key, None)
        elif kind == "store":
            assert held.setdefault(key, store) == store, (
                f"{store} locked while {held[key]} is held"
            )
