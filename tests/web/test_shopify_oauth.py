"""Shopify per-account OAuth (#2810): the registered store host, the signed
callback checks made before the code exchange, the granted-scope check, the
shop identity, and refresh on the stored store label.
"""

import hashlib
import hmac
import socket
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette.datastructures import QueryParams

from xagent.core.utils.encryption import encrypt_value
from xagent.web import oauth_account_hosts, shopify_oauth
from xagent.web.api import auth as auth_api
from xagent.web.api.auth import create_access_token, generic_oauth_login
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
                ("code", "second"),
                ("hmac", _sign(_callback_params())),
            ],
            "hmac_invalid",
            id="repeated-param",
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


def test_unreachable_identity_endpoint_does_not_echo_the_error(monkeypatch):
    def failing_post(url, **kwargs):
        raise shopify_oauth.requests.ConnectionError(f"boom {kwargs['headers']}")

    monkeypatch.setattr(shopify_oauth.requests, "post", failing_post)

    with pytest.raises(ShopifyOAuthError) as caught:
        fetch_shopify_shop_identity("acme", "shpat-token")

    assert caught.value.reason == "identity_unreachable"
    assert "shpat-token" not in caught.value.message


# ---------- login and callback -------------------------------------------------


@pytest.fixture()
def shopify_db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'shopify-oauth.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with factory() as db:
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
        yield db, user
    engine.dispose()


def _db_provider(db) -> OAuthProvider:
    return db.query(OAuthProvider).filter_by(provider_name="shopify").one()


def _login(db, user, account: str = "acme"):
    token = create_access_token(
        data={"sub": user.username, "type": "access"},
        expires_delta=timedelta(minutes=5),
    )
    return generic_oauth_login(
        "shopify",
        token=token,
        app_id=TEST_APP_ID,
        redirect=None,
        db=db,
        db_provider=_db_provider(db),
        account=account,
    )


def _login_state(db, user, account: str = "acme") -> str:
    location = _login(db, user, account).headers["location"]
    return parse_qs(urlparse(location).query)["state"][0]


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


def _callback(db, state: str, **overrides: str):
    params = _callback_params(state=state, **overrides)
    request = SimpleNamespace(query_params=_signed_query(params), cookies={})
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
    state = _login_state(db, user)
    params = _callback_params(state=state)
    query = _signed_query(params)
    if "code" in overrides:
        # Re-sign nothing: the changed code no longer matches the signature.
        query = QueryParams({**dict(query), **overrides})
    else:
        query = _signed_query({**params, **overrides})
    request = SimpleNamespace(query_params=query, cookies={})

    response = auth_api.generic_oauth_callback(
        "shopify", request, db=db, db_provider=_db_provider(db)
    )

    assert response.status_code == 400
    assert calls == []
    assert _grant(db, user) is None


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
    _mock_shopify_http(
        monkeypatch, identity=_shop_payload(myshopifyDomain="other.myshopify.com")
    )

    response = _callback(db, _login_state(db, user))

    assert response.status_code == 400
    assert _grant(db, user) is None


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

    monkeypatch.setattr(tool_config.httpx, "AsyncClient", FakeAsyncClient)

    assert await tool_config.refresh_oauth_token_if_needed(db, grant, "shopify")

    url, kwargs = posted[0]
    assert url == TOKEN_URL
    assert kwargs["data"]["grant_type"] == "refresh_token"
    assert kwargs["data"]["refresh_token"] == "old-refresh"
    assert kwargs["follow_redirects"] is False
    assert kwargs["headers"]["Accept"] == "application/json"
    assert grant.access_token == "rotated-access"
    assert grant.refresh_token == "rotated-refresh"
    assert grant.instance_url == "acme"
