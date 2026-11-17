"""The request Shopify sends to the app URL after an install (#2810).

Shopify opens the app URL with ``shop``, ``timestamp``, ``host`` and an
``hmac`` over them, and with no ``code`` or ``state`` of ours. The store label
the install flow starts OAuth on is taken from that signed ``shop``, so the
check has to cover the signature, the store host, and how old the request is.
"""

import hashlib
import hmac

import pytest
from starlette.datastructures import QueryParams

from xagent.web.shopify_oauth import (
    SHOPIFY_LAUNCH_MAX_AGE_SECONDS,
    ShopifyOAuthError,
    verify_shopify_launch,
)

CLIENT_SECRET = "client-secret"
NOW = 1_760_000_000


def _sign(params: dict[str, str], secret: str = CLIENT_SECRET) -> str:
    message = "&".join(f"{key}={value}" for key, value in sorted(params.items()))
    return hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()


def _launch_params(**overrides: str) -> dict[str, str]:
    return {
        "host": "YWRtaW4uc2hvcGlmeS5jb20vc3RvcmUvYWNtZQ",
        "shop": "acme.myshopify.com",
        "timestamp": str(NOW),
        **overrides,
    }


def _signed(params: dict[str, str], secret: str = CLIENT_SECRET):
    return QueryParams({**params, "hmac": _sign(params, secret)}).multi_items()


def _verify(items, now: float = NOW) -> str:
    return verify_shopify_launch(items, client_secret=CLIENT_SECRET, now=now)


def test_a_signed_launch_returns_the_store_label():
    assert _verify(_signed(_launch_params())) == "acme"


def test_the_store_host_is_read_case_insensitively():
    assert _verify(_signed(_launch_params(shop="Acme.MyShopify.com"))) == "acme"


def test_a_launch_slightly_ahead_of_our_clock_verifies():
    # Shopify's clock may run a little ahead of ours.
    assert _verify(_signed(_launch_params(timestamp=str(NOW + 60)))) == "acme"


def test_a_launch_within_the_allowed_age_verifies():
    assert (
        _verify(_signed(_launch_params()), now=NOW + SHOPIFY_LAUNCH_MAX_AGE_SECONDS)
        == "acme"
    )


@pytest.mark.parametrize(
    ("items", "reason"),
    [
        pytest.param(list(_launch_params().items()), "hmac_missing", id="unsigned"),
        pytest.param(
            _signed(_launch_params(), "other-secret"), "hmac_invalid", id="other-secret"
        ),
        pytest.param(
            [
                *_launch_params(shop="evil.myshopify.com").items(),
                ("hmac", _sign(_launch_params())),
            ],
            "hmac_invalid",
            id="shop-swapped-after-signing",
        ),
        pytest.param(
            [*_signed(_launch_params()), ("shop", "acme.myshopify.com")],
            "hmac_invalid",
            id="repeated-param",
        ),
        pytest.param(
            _signed({k: v for k, v in _launch_params().items() if k != "shop"}),
            "shop_invalid",
            id="missing-shop",
        ),
        pytest.param(
            _signed(_launch_params(shop="acme.myshopify.com.attacker.example")),
            "shop_invalid",
            id="suffixed-host",
        ),
        pytest.param(
            _signed(_launch_params(shop="a.b.myshopify.com")),
            "shop_invalid",
            id="nested-label",
        ),
        pytest.param(
            _signed(_launch_params(shop="ac_me.myshopify.com")),
            "shop_invalid",
            id="not-a-dns-label",
        ),
        pytest.param(
            _signed(_launch_params(shop="acme.example.com")),
            "shop_invalid",
            id="other-domain",
        ),
        pytest.param(
            _signed({k: v for k, v in _launch_params().items() if k != "timestamp"}),
            "timestamp_invalid",
            id="missing-timestamp",
        ),
        pytest.param(
            _signed(_launch_params(timestamp="soon")),
            "timestamp_invalid",
            id="non-numeric-timestamp",
        ),
        pytest.param(
            _signed(
                _launch_params(timestamp=str(NOW - SHOPIFY_LAUNCH_MAX_AGE_SECONDS - 1))
            ),
            "timestamp_stale",
            id="too-old",
        ),
        pytest.param(
            _signed(_launch_params(timestamp=str(NOW + 120))),
            "timestamp_stale",
            id="from-the-future",
        ),
    ],
)
def test_unverifiable_launches_are_rejected(items, reason):
    with pytest.raises(ShopifyOAuthError) as caught:
        _verify(items)

    assert caught.value.reason == reason
