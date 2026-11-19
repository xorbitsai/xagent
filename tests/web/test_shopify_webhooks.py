"""Shopify's app webhooks (#2810): ``app/uninstalled`` and the three mandatory
compliance topics, received at ``POST /api/shopify/webhooks``.

Shopify signs each delivery with the app's client secret over the raw body
(``X-Shopify-Hmac-Sha256``). The topic and shop headers are not signed, so the
store comes from the signed body and a destructive topic must carry its own
body shape.
"""

import base64
import hashlib
import hmac
import json
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.web.api import shopify_webhooks
from xagent.web.builtin_mcp_registry import get_builtin_oauth_provider_rows
from xagent.web.models.database import Base, get_db
from xagent.web.models.oauth_provider import OAuthProvider
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.shopify_oauth import verify_shopify_webhook

CLIENT_SECRET = "shpss_test_secret"
PATH = "/api/shopify/webhooks"

UNINSTALLED = {
    "id": 548380009,
    "name": "Acme",
    "domain": "shop.acme.example",
    "myshopify_domain": "acme.myshopify.com",
}
SHOP_REDACT = {"shop_id": 548380009, "shop_domain": "acme.myshopify.com"}
CUSTOMERS_REDACT = {
    "shop_id": 548380009,
    "shop_domain": "acme.myshopify.com",
    "customer": {
        "id": 207119551,
        "email": "jane@customer.example",
        "phone": "555-625-1199",
    },
    "orders_to_redact": [299938, 280263],
}
CUSTOMERS_DATA_REQUEST = {
    "shop_id": 548380009,
    "shop_domain": "acme.myshopify.com",
    "orders_requested": [299938],
    "customer": {
        "id": 207119551,
        "email": "jane@customer.example",
        "phone": "555-625-1199",
    },
    "data_request": {"id": 9999},
}


def _signature(body: bytes, secret: str = CLIENT_SECRET) -> str:
    digest = hmac.new(secret.encode(), body, hashlib.sha256).digest()
    return base64.b64encode(digest).decode()


@pytest.fixture()
def shopify_db(tmp_path, monkeypatch):
    monkeypatch.delenv("SHOPIFY_CLIENT_SECRET", raising=False)
    engine = create_engine(f"sqlite:///{tmp_path / 'shopify-webhooks.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with factory() as db:
        user = User(username="alice", password_hash="hash")
        other = User(username="bob", password_hash="hash")
        db.add_all([user, other])
        # The registry's own row: client id and secret blank, read from env.
        db.add(
            OAuthProvider(
                **next(
                    row
                    for row in get_builtin_oauth_provider_rows()
                    if row["provider_name"] == "shopify"
                )
            )
        )
        db.commit()
        yield factory, db, user, other
    engine.dispose()


@pytest.fixture()
def client(shopify_db, monkeypatch):
    factory, *_ = shopify_db
    monkeypatch.setenv("SHOPIFY_CLIENT_SECRET", CLIENT_SECRET)
    app = FastAPI()
    app.include_router(shopify_webhooks.router)

    def _db():
        with factory() as db:
            yield db

    app.dependency_overrides[get_db] = _db
    return TestClient(app)


def _grant(
    user, *, store="acme", oauth_provider="shopify", actor=None, gid="1", app="shopify"
):
    return UserOAuth(
        user_id=user.id,
        provider=app,
        oauth_provider=oauth_provider,
        resource_owner_key=actor,
        access_token="access",
        refresh_token="refresh",
        instance_url=store,
        provider_user_id=f"gid://shopify/Shop/{gid}",
    )


def _post(client, topic, payload, *, secret=CLIENT_SECRET, body=None, signature=None):
    raw = body if body is not None else json.dumps(payload).encode()
    headers = {
        "Content-Type": "application/json",
        "X-Shopify-Topic": topic,
        "X-Shopify-Shop-Domain": "acme.myshopify.com",
        "X-Shopify-Webhook-Id": "b54557e4-bdd9-4b37-8a5f-bf7d70bcd043",
    }
    if signature is not False:
        headers["X-Shopify-Hmac-Sha256"] = signature or _signature(raw, secret)
    return client.post(PATH, content=raw, headers=headers)


def _stores(db):
    db.expire_all()
    return sorted(
        (grant.oauth_provider, grant.instance_url, grant.resource_owner_key)
        for grant in db.query(UserOAuth).all()
    )


# ---------- signature ---------------------------------------------------------


def test_known_webhook_signature_vector_verifies():
    body = b'{"shop_id":1,"shop_domain":"acme.myshopify.com"}'
    expected = base64.b64encode(
        hmac.new(b"secret", body, hashlib.sha256).digest()
    ).decode()
    assert verify_shopify_webhook(body, expected, client_secret="secret")
    assert not verify_shopify_webhook(body, expected, client_secret="other")
    assert not verify_shopify_webhook(body + b" ", expected, client_secret="secret")
    assert not verify_shopify_webhook(body, None, client_secret="secret")
    assert not verify_shopify_webhook(body, "", client_secret="secret")
    assert not verify_shopify_webhook(body, expected, client_secret="")


@pytest.mark.parametrize(
    "case", ["wrong-secret", "missing", "tampered-body", "not-base64"]
)
def test_an_unverified_delivery_gets_401_and_changes_nothing(client, shopify_db, case):
    _, db, user, _ = shopify_db
    db.add(_grant(user))
    db.commit()
    raw = json.dumps(UNINSTALLED).encode()
    if case == "wrong-secret":
        response = _post(client, "app/uninstalled", UNINSTALLED, secret="other")
    elif case == "missing":
        response = _post(client, "app/uninstalled", UNINSTALLED, signature=False)
    elif case == "tampered-body":
        response = _post(
            client,
            "app/uninstalled",
            None,
            body=raw.replace(b"548380009", b"548380010"),
            signature=_signature(raw),
        )
    else:
        response = _post(client, "app/uninstalled", UNINSTALLED, signature="%%%")
    assert response.status_code == 401
    assert _stores(db) == [("shopify", "acme", None)]


def test_an_unconfigured_secret_is_a_server_error_not_a_pass(
    client, shopify_db, monkeypatch
):
    """Without a secret nothing can be verified; 503 makes Shopify retry once
    the deployment is fixed instead of dropping the delivery."""
    _, db, user, _ = shopify_db
    db.add(_grant(user))
    db.commit()
    monkeypatch.delenv("SHOPIFY_CLIENT_SECRET")
    response = _post(client, "app/uninstalled", UNINSTALLED)
    assert response.status_code == 503
    assert _stores(db) == [("shopify", "acme", None)]


# ---------- app/uninstalled and shop/redact -----------------------------------


@pytest.mark.parametrize(
    "topic,payload", [("app/uninstalled", UNINSTALLED), ("shop/redact", SHOP_REDACT)]
)
def test_uninstall_and_shop_redact_delete_every_grant_for_the_store(
    client, shopify_db, topic, payload
):
    _, db, user, other = shopify_db
    db.add_all(
        [
            _grant(user),
            _grant(user, actor="actor:toby:1"),
            _grant(other, store="ACME"),
            _grant(user, store="beta", gid="2"),
            # Another issuer's grant that happens to hold the same label.
            _grant(other, oauth_provider="deputy", app="deputy"),
        ]
    )
    db.commit()

    response = _post(client, topic, payload)

    assert response.status_code == 200
    assert _stores(db) == [("deputy", "acme", None), ("shopify", "beta", None)]


def test_a_repeated_delivery_is_still_accepted(client, shopify_db):
    _, db, user, _ = shopify_db
    db.add(_grant(user))
    db.commit()
    assert _post(client, "app/uninstalled", UNINSTALLED).status_code == 200
    assert _post(client, "app/uninstalled", UNINSTALLED).status_code == 200
    assert _post(client, "shop/redact", SHOP_REDACT).status_code == 200
    assert _stores(db) == []


@pytest.mark.parametrize(
    "topic,payload",
    [
        # The topic header is not signed: a signed customer body relabelled
        # as shop/redact, or one topic's body relabelled as the other's, must
        # not delete anything.
        ("shop/redact", CUSTOMERS_REDACT),
        ("shop/redact", CUSTOMERS_DATA_REQUEST),
        ("shop/redact", UNINSTALLED),
        ("app/uninstalled", SHOP_REDACT),
        ("app/uninstalled", CUSTOMERS_REDACT),
        # Only a store directly under myshopify.com is ours to look up.
        ("app/uninstalled", {**UNINSTALLED, "myshopify_domain": "acme.example.com"}),
        ("app/uninstalled", {**UNINSTALLED, "myshopify_domain": "a.b.myshopify.com"}),
        ("shop/redact", {**SHOP_REDACT, "shop_domain": ""}),
        ("shop/redact", {**SHOP_REDACT, "shop_domain": 5}),
    ],
)
def test_a_body_that_does_not_fit_its_topic_is_refused(
    client, shopify_db, topic, payload
):
    _, db, user, _ = shopify_db
    db.add(_grant(user))
    db.commit()
    response = _post(client, topic, payload)
    assert response.status_code == 400
    assert _stores(db) == [("shopify", "acme", None)]


@pytest.mark.parametrize("body", [b"not json", b"[]", b"null"])
def test_a_signed_body_that_is_not_a_json_object_is_refused(client, body):
    response = _post(client, "shop/redact", None, body=body)
    assert response.status_code == 400


# ---------- customer topics ---------------------------------------------------


@pytest.mark.parametrize(
    "topic,payload",
    [
        ("customers/redact", CUSTOMERS_REDACT),
        ("customers/data_request", CUSTOMERS_DATA_REQUEST),
    ],
)
def test_customer_topics_are_acknowledged_without_touching_grants(
    client, shopify_db, caplog, topic, payload
):
    """We keep no per-customer records, so these change nothing; the log line
    names the store and topic but never the customer."""
    _, db, user, _ = shopify_db
    db.add(_grant(user))
    db.commit()
    with caplog.at_level(logging.INFO):
        response = _post(client, topic, payload)
    assert response.status_code == 200
    assert _stores(db) == [("shopify", "acme", None)]
    assert any(topic in record.getMessage() for record in caplog.records)
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "jane@customer.example" not in logged
    assert "555-625-1199" not in logged
    assert "207119551" not in logged


def test_an_unknown_topic_is_acknowledged_and_ignored(client, shopify_db):
    _, db, user, _ = shopify_db
    db.add(_grant(user))
    db.commit()
    response = _post(client, "orders/create", {"id": 1})
    assert response.status_code == 200
    assert _stores(db) == [("shopify", "acme", None)]


# ---------- reuse ------------------------------------------------------------


def test_the_provider_row_secret_wins_over_the_environment(shopify_db, monkeypatch):
    """Resolved like the connector's own OAuth client: row first, then env."""
    from xagent.core.utils.encryption import encrypt_value

    _, db, _, _ = shopify_db
    monkeypatch.setenv("SHOPIFY_CLIENT_SECRET", "env-secret")
    assert shopify_webhooks.shopify_webhook_secret(db) == "env-secret"
    row = db.query(OAuthProvider).filter_by(provider_name="shopify").one()
    row.client_secret = encrypt_value("row-secret")
    db.commit()
    assert shopify_webhooks.shopify_webhook_secret(db) == "row-secret"


def test_the_route_is_registered_on_the_web_app():
    from xagent.web.app import app

    assert "post" in app.openapi()["paths"][PATH]
