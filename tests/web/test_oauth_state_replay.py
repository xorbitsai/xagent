import hashlib
from datetime import timedelta
from unittest.mock import Mock
from urllib.parse import urlencode

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker
from starlette.requests import Request

from xagent.web.api import auth
from xagent.web.models.database import Base
from xagent.web.models.oauth_provider import OAuthProvider
from xagent.web.models.oidc_consumed_token import OidcConsumedToken
from xagent.web.models.user import User


@pytest.fixture
def db_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'callbacks.db'}")
    Base.metadata.create_all(engine)
    try:
        with sessionmaker(bind=engine)() as db:
            user = User(username="alice", password_hash="hash", is_admin=False)
            db.add(user)
            db.commit()
            yield db, user
    finally:
        engine.dispose()


@pytest.fixture
def callback_client(db_session):
    db, _ = db_session
    db.add(
        OAuthProvider(
            provider_name="custom",
            name="Custom",
            client_id="client",
            client_secret="secret",
            auth_url="https://provider.example/authorize",
            token_url="https://provider.example/token",
        )
    )
    db.commit()
    app = FastAPI()
    app.include_router(auth.auth_router)
    app.dependency_overrides[auth.get_db] = lambda: db
    with TestClient(app) as client:
        yield client


def _callback(state, nonce="browser-secret", *, error=True):
    digest = hashlib.sha256(nonce.encode()).hexdigest()
    params = {
        "state": state,
        **({"error": "access_denied"} if error else {"code": "code"}),
    }
    return Request(
        {
            "type": "http",
            "query_string": urlencode(params).encode(),
            "headers": [
                (
                    b"cookie",
                    f"{auth._ordinary_oauth_cookie_name(digest)}={nonce}".encode(),
                )
            ],
        }
    )


def _state():
    return auth.create_access_token(
        {
            "type": "oauth_state",
            "provider": "custom",
            "user_id": 1,
            "browser_nonce_hash": hashlib.sha256(b"browser-secret").hexdigest(),
        },
        expires_delta=timedelta(minutes=10),
    )


def test_denied_provider_flow_is_consumed_and_cannot_be_replayed(
    db_session, monkeypatch
):
    db, _ = db_session
    post = Mock()
    monkeypatch.setattr(auth.requests, "post", post)
    state = _state()
    first = auth.generic_oauth_callback("custom", _callback(state), db, None)
    assert first.status_code == 400
    assert db.query(OidcConsumedToken).count() == 1
    # Closing this Session models a return to another API worker.
    db.close()
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    alternate_bits = state[:-1] + alphabet[alphabet.index(state[-1]) ^ 1]
    for variant in (state, state + "=", alternate_bits):
        assert auth.verify_token(variant) is not None
        replay = auth.generic_oauth_callback(
            "custom", _callback(variant, error=False), db, None
        )
        assert replay.status_code == 400
        assert b"Invalid or expired state" in replay.body
    post.assert_not_called()


def test_wrong_browser_does_not_burn_valid_state(db_session, monkeypatch):
    db, _ = db_session
    claim = Mock()
    monkeypatch.setattr(auth, "claim_oauth_state", claim)
    response = auth.generic_oauth_callback(
        "custom", _callback(_state(), "other-browser"), db, None
    )
    assert response.status_code == 400
    claim.assert_not_called()
    assert db.query(OidcConsumedToken).count() == 0


def test_claim_store_outage_returns_sanitized_503_before_exchange(
    db_session, monkeypatch
):
    db, _ = db_session
    monkeypatch.setattr(
        auth,
        "claim_oauth_state",
        Mock(
            side_effect=OperationalError(
                "private-sql", {}, Exception("private-db-host")
            )
        ),
    )
    post = Mock()
    monkeypatch.setattr(auth.requests, "post", post)
    response = auth.generic_oauth_callback(
        "custom", _callback(_state(), error=False), db, None
    )
    assert response.status_code == 503
    assert b"private" not in response.body
    post.assert_not_called()


@pytest.mark.parametrize("binding", [None, "", "not-a-digest", "g" * 64, True])
def test_public_callback_rejects_unbound_or_malformed_legacy_states(
    callback_client, monkeypatch, binding
):
    state = auth.create_access_token(
        {
            "type": "oauth_state",
            "provider": "custom",
            "user_id": 1,
            "browser_nonce_hash": binding,
        },
        expires_delta=timedelta(minutes=10),
    )
    response = callback_client.get(
        "/api/auth/custom/callback", params={"state": state, "code": "code"}
    )
    assert response.status_code == 400
    assert "Invalid or expired state" in response.text
    direct = auth.generic_oauth_callback(
        "custom",
        _callback(state, error=False),
        callback_client.app.dependency_overrides[auth.get_db](),
        None,
    )
    assert direct.status_code == 400


def test_public_callback_checks_cookie_and_consumes_durable_state(
    callback_client, db_session, monkeypatch
):
    post = Mock()
    monkeypatch.setattr(auth.requests, "post", post)
    state = _state()
    request = _callback(state)
    headers = {"Cookie": request.headers["cookie"]}
    path = "/api/auth/custom/callback?" + request.scope["query_string"].decode()
    assert callback_client.get(path, headers=headers).status_code == 400
    assert db_session[0].query(OidcConsumedToken).count() == 1
    replay = callback_client.get(path, headers=headers)
    assert replay.status_code == 400
    assert "Invalid or expired state" in replay.text
    post.assert_not_called()


def test_unknown_provider_error_escapes_request_input(db_session):
    response = auth.oauth_callback(
        '<img src=x onerror="alert(1)">', _callback("invalid"), db_session[0]
    )
    assert response.status_code == 400
    assert b"<img" not in response.body
    assert b"&lt;img" in response.body
