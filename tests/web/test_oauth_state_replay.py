import hashlib
import json
import re
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
from xagent.web.models.user_oauth import UserOAuth


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
def configured_provider(db_session):
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


@pytest.fixture
def callback_client(db_session, configured_provider):
    db, _ = db_session
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


def _state(session_epoch=None, **extra_claims):
    return auth.create_access_token(
        {
            "type": "oauth_state",
            "provider": "custom",
            "user_id": 1,
            "session_epoch": session_epoch,
            "browser_nonce_hash": hashlib.sha256(b"browser-secret").hexdigest(),
            **extra_claims,
        },
        expires_delta=timedelta(minutes=10),
    )


@pytest.mark.parametrize(
    "email",
    [
        "alice@example.com",
        "</script><script>window.injected=true</script>",
        '</ScRiPt><img src=x onerror="alert(1)">',
        "<!--<script>\u2028\u2029中文&",
    ],
)
@pytest.mark.parametrize("field", ["email", "app_id"])
def test_success_callback_keeps_provider_identity_inert(
    configured_provider, db_session, monkeypatch, email, field
):
    db, _ = db_session
    provider = db.query(OAuthProvider).filter_by(provider_name="custom").one()
    provider.userinfo_url = "https://provider.example/me"
    db.commit()
    identity_email = email if field == "email" else "alice@example.com"
    claims = {"app_id": email} if field == "app_id" else {}
    # Exercise the third JSON expression (postMessage target origin), too.
    claims["redirect"] = "https://ui<&>.example/path"
    monkeypatch.setattr(
        auth.requests,
        "post",
        Mock(return_value=Mock(status_code=200, json=lambda: {"access_token": "test"})),
    )
    monkeypatch.setattr(
        auth.requests,
        "get",
        Mock(
            return_value=Mock(
                status_code=200, json=lambda: {"id": "1", "email": identity_email}
            )
        ),
    )
    monkeypatch.setattr(auth, "_run_post_commit_oauth_side_effects", Mock())
    response = auth.generic_oauth_callback(
        "custom", _callback(_state(**claims), error=False), db, provider
    )
    assert response.status_code == 200
    body = response.body.decode()
    assert body.lower().count("<script>") == 1
    assert body.lower().count("</script>") == 1
    assert "<img" not in body
    encoded_email = re.search(r"email: (.*),\n", body)
    assert encoded_email is not None
    assert json.loads(encoded_email.group(1)) == identity_email
    encoded_provider = re.search(r"provider: (.*)\n", body)
    assert encoded_provider is not None
    assert json.loads(encoded_provider.group(1)) == (
        email if field == "app_id" else "custom"
    )
    assert '"https://ui\\u003c\\u0026\\u003e.example"' in body
    assert db.query(UserOAuth).one().email == identity_email


@pytest.mark.parametrize("initial_epoch", [None, "current-generation"])
def test_password_change_revokes_pending_connector_state_before_provider_exchange(
    db_session, monkeypatch, initial_epoch
):
    db, user = db_session
    user.session_epoch = initial_epoch
    db.commit()
    state = _state(initial_epoch)
    user.session_epoch = "revoked-generation"
    db.commit()
    post = Mock()
    monkeypatch.setattr(auth.requests, "post", post)
    rejection = Mock()
    monkeypatch.setattr(auth, "_log_oauth_callback_rejection", rejection)
    response = auth.generic_oauth_callback(
        "custom", _callback(state, error=False), db, None
    )
    assert response.status_code == 400
    assert b"Invalid or expired state" in response.body
    assert db.query(UserOAuth).count() == 0
    post.assert_not_called()
    rejection.assert_called_once_with(
        "custom",
        "state_session_mismatch",
        app_id=None,
        actor_flow=False,
        provider_error=None,
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
    # PyJWT accepts canonical signatures with optional padding, but rejects
    # nonzero unused base64url bits before the durable replay check.
    for variant, valid_encoding in (
        (state, True),
        (state + "=", True),
        (alternate_bits, False),
    ):
        assert (auth.verify_token(variant) is not None) is valid_encoding
        replay = auth.generic_oauth_callback(
            "custom", _callback(variant, error=False), db, None
        )
        assert replay.status_code == 400
        assert b"Invalid or expired state" in replay.body
    assert db.query(OidcConsumedToken).count() == 1
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
