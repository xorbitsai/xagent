from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette.middleware.sessions import SessionMiddleware

from xagent.web.api.auth import auth_router, set_registration_enabled
from xagent.web.models.database import Base, get_db
from xagent.web.models.user import User
from xagent.web.models.user_identity import UserIdentity


def test_exchange_cleanup_preserves_connector_proof_namespaces(oidc_client):
    from xagent.web.api import oidc_google
    from xagent.web.models.oidc_consumed_token import OidcConsumedToken

    _, SessionLocal = oidc_client
    with SessionLocal() as db:
        old = datetime.now(timezone.utc) - timedelta(minutes=5)
        db.add_all(
            [
                OidcConsumedToken(token_id=key, expires_at=old)
                for key in (
                    "oauth-launch:old",
                    "oauth-state:old",
                    "expired-exchange",
                )
            ]
        )
        db.commit()
        code = oidc_google._create_exchange_code(1, None)
        assert oidc_google._consume_exchange_code(db, code) is not None
        keys = {row.token_id for row in db.query(OidcConsumedToken)}
        assert "oauth-launch:old" in keys
        assert "oauth-state:old" in keys
        assert "expired-exchange" not in keys


@pytest.fixture()
def oidc_client(tmp_path, monkeypatch):
    db_path = tmp_path / "oidc.db"
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    monkeypatch.setenv("XAGENT_GOOGLE_OIDC_CLIENT_ID", "google-client-id")
    monkeypatch.setenv("XAGENT_GOOGLE_OIDC_CLIENT_SECRET", "google-client-secret")
    monkeypatch.setenv(
        "XAGENT_GOOGLE_OIDC_REDIRECT_URI",
        "http://testserver/api/auth/oidc/google/callback",
    )
    monkeypatch.setenv("XAGENT_FRONTEND_URL", "http://frontend.local")

    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key="test-session-secret")
    app.include_router(auth_router)
    app.dependency_overrides[get_db] = override_get_db

    yield TestClient(app), SessionLocal

    engine.dispose()


def _create_admin(SessionLocal) -> User:
    from xagent.web.api.auth import hash_password
    from xagent.web.models.system_setting import SystemSetting

    db = SessionLocal()
    try:
        user = User(
            username="admin",
            password_hash=hash_password("admin123"),
            is_admin=True,
        )
        db.add(user)
        db.add(SystemSetting(key="setup_completed", value="true"))
        db.commit()
        db.refresh(user)
        return user
    finally:
        db.close()


def _install_google_fakes(monkeypatch, claims: dict[str, object]) -> None:
    async def fake_complete(request, *, redirect_uri: str):
        code = request.query_params.get("code")
        assert code == "google-code"
        assert redirect_uri == "http://testserver/api/auth/oidc/google/callback"
        return claims

    monkeypatch.setattr(
        "xagent.web.api.oidc_google.complete_google_oidc_authorization",
        fake_complete,
    )


class FakeAuthlibGoogleClient:
    def __init__(self, expires_in: int = 3600) -> None:
        self.expires_in = expires_in

    async def authorize_redirect(self, request, redirect_uri, **authorize_params):
        from authlib.common.security import generate_token
        from authlib.oauth2.rfc7636 import create_s256_code_challenge
        from starlette.responses import RedirectResponse

        state = generate_token(48)
        nonce = generate_token(20)
        code_verifier = generate_token(48)
        code_challenge = create_s256_code_challenge(code_verifier)
        request.session[f"_state_google_{state}"] = {
            "data": {
                "redirect_uri": redirect_uri,
                "nonce": nonce,
                "code_verifier": code_verifier,
            },
            "exp": time.time() + self.expires_in,
        }
        url = (
            "https://accounts.google.com/o/oauth2/v2/auth"
            f"?response_type=code"
            f"&client_id=google-client-id"
            f"&redirect_uri={redirect_uri}"
            f"&scope=openid+email+profile"
            f"&state={state}"
            f"&nonce={nonce}"
            f"&code_challenge={code_challenge}"
            f"&code_challenge_method=S256"
        )
        if prompt := authorize_params.get("prompt"):
            url = f"{url}&prompt={prompt}"
        return RedirectResponse(url)


def _install_authlib_login_fake(monkeypatch) -> None:
    monkeypatch.setattr(
        "xagent.web.api.oidc_google._google_oauth_client",
        lambda: FakeAuthlibGoogleClient(),
    )


def _start_google_login(client: TestClient) -> str:
    response = client.get("/api/auth/oidc/google/login", follow_redirects=False)
    if response.status_code != 307:
        raise AssertionError(response.text)
    location = response.headers["location"]
    query = parse_qs(urlparse(location).query)
    assert query["client_id"] == ["google-client-id"]
    assert query["scope"] == ["openid email profile"]
    assert query["response_type"] == ["code"]
    assert query["code_challenge_method"] == ["S256"]
    assert query["prompt"] == ["select_account"]
    assert "code_challenge" in query
    assert "state" in query
    assert "nonce" in query
    return query["state"][0]


def _complete_google_callback(client: TestClient, state: str) -> str:
    response = client.get(
        f"/api/auth/oidc/google/callback?code=google-code&state={state}",
        follow_redirects=False,
    )
    assert response.status_code == 307
    redirect = response.headers["location"]
    parsed = urlparse(redirect)
    query = parse_qs(parsed.query)
    assert parsed.scheme == "http"
    assert parsed.netloc == "frontend.local"
    assert parsed.path == "/auth/oidc/callback"
    assert query["provider"] == ["google"]
    assert "code" in query
    return query["code"][0]


def test_google_oidc_login_creates_user_when_registration_enabled(
    oidc_client, monkeypatch
):
    client, SessionLocal = oidc_client
    _create_admin(SessionLocal)
    _install_authlib_login_fake(monkeypatch)

    claims = {
        "sub": "google-sub-123",
        "email": "person@example.com",
        "email_verified": True,
        "name": "Person Example",
        "picture": "https://example.com/avatar.png",
    }
    _install_google_fakes(monkeypatch, claims)

    state = _start_google_login(client)
    exchange_code = _complete_google_callback(client, state)

    response = client.post(
        "/api/auth/oidc/google/exchange", json={"code": exchange_code}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["user"]["username"] == "person@example.com"
    assert data["user"]["email"] == "person@example.com"
    assert data["user"]["is_admin"] is False
    assert data["user"]["has_password"] is False
    assert data["access_token"]
    assert data["refresh_token"]

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == "person@example.com").one()
        identity = db.query(UserIdentity).filter(UserIdentity.user_id == user.id).one()
        assert identity.provider == "google"
        assert identity.provider_subject == "google-sub-123"
        assert identity.email == "person@example.com"
        assert identity.email_verified is True
    finally:
        db.close()


def test_google_oidc_status_reports_configuration(oidc_client):
    client, _ = oidc_client

    response = client.get("/api/auth/oidc/google/status")

    assert response.status_code == 200
    assert response.json() == {"configured": True, "provider": "google"}


def test_google_oidc_rejects_new_user_when_registration_disabled(
    oidc_client, monkeypatch
):
    client, SessionLocal = oidc_client
    _create_admin(SessionLocal)
    db = SessionLocal()
    try:
        set_registration_enabled(db, False)
    finally:
        db.close()
    _install_authlib_login_fake(monkeypatch)

    _install_google_fakes(
        monkeypatch,
        {
            "sub": "google-sub-456",
            "email": "blocked@example.com",
            "email_verified": True,
        },
    )

    state = _start_google_login(client)
    response = client.get(
        f"/api/auth/oidc/google/callback?code=google-code&state={state}",
        follow_redirects=False,
    )

    assert response.status_code == 307
    parsed = urlparse(response.headers["location"])
    query = parse_qs(parsed.query)
    assert parsed.path == "/login"
    assert query["oidc_error"] == ["registration_disabled"]

    db = SessionLocal()
    try:
        assert (
            db.query(User).filter(User.username == "blocked@example.com").first()
            is None
        )
    finally:
        db.close()


def test_google_oidc_rejects_unverified_email(oidc_client, monkeypatch):
    client, SessionLocal = oidc_client
    _create_admin(SessionLocal)
    _install_authlib_login_fake(monkeypatch)
    _install_google_fakes(
        monkeypatch,
        {
            "sub": "google-sub-789",
            "email": "unverified@example.com",
            "email_verified": False,
        },
    )

    state = _start_google_login(client)
    response = client.get(
        f"/api/auth/oidc/google/callback?code=google-code&state={state}",
        follow_redirects=False,
    )

    assert response.status_code == 307
    parsed = urlparse(response.headers["location"])
    query = parse_qs(parsed.query)
    assert parsed.path == "/login"
    assert query["oidc_error"] == ["email_unverified"]


def test_google_oidc_exchange_code_is_stateless_across_workers(
    oidc_client, monkeypatch
):
    client, SessionLocal = oidc_client
    _create_admin(SessionLocal)
    _install_authlib_login_fake(monkeypatch)
    _install_google_fakes(
        monkeypatch,
        {
            "sub": "google-sub-single-use",
            "email": "single@example.com",
            "email_verified": True,
        },
    )

    state = _start_google_login(client)
    exchange_code = _complete_google_callback(client, state)

    from xagent.web.api.oidc_google import _decode_exchange_code

    transaction = _decode_exchange_code(exchange_code)

    assert transaction is not None
    assert isinstance(transaction.user_id, int)
    assert transaction.token_id


def test_google_oidc_exchange_code_rejects_replay(oidc_client, monkeypatch):
    client, SessionLocal = oidc_client
    _create_admin(SessionLocal)
    _install_authlib_login_fake(monkeypatch)
    _install_google_fakes(
        monkeypatch,
        {
            "sub": "google-sub-replay",
            "email": "replay@example.com",
            "email_verified": True,
        },
    )

    state = _start_google_login(client)
    exchange_code = _complete_google_callback(client, state)

    first_response = client.post(
        "/api/auth/oidc/google/exchange", json={"code": exchange_code}
    )
    second_response = client.post(
        "/api/auth/oidc/google/exchange", json={"code": exchange_code}
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 400
    assert second_response.json()["detail"] == "Invalid or expired OIDC exchange code"


def test_google_oidc_exchange_code_rejects_tampering(oidc_client, monkeypatch):
    client, SessionLocal = oidc_client
    _create_admin(SessionLocal)
    _install_authlib_login_fake(monkeypatch)
    _install_google_fakes(
        monkeypatch,
        {
            "sub": "google-sub-tamper",
            "email": "tamper@example.com",
            "email_verified": True,
        },
    )

    state = _start_google_login(client)
    exchange_code = _complete_google_callback(client, state)
    tampered_code = f"{exchange_code}x"

    response = client.post(
        "/api/auth/oidc/google/exchange", json={"code": tampered_code}
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid or expired OIDC exchange code"


def test_existing_google_identity_logs_in_when_registration_disabled(
    oidc_client, monkeypatch
):
    client, SessionLocal = oidc_client
    _create_admin(SessionLocal)
    _install_authlib_login_fake(monkeypatch)
    db = SessionLocal()
    try:
        user = User(
            username="linked@example.com",
            password_hash="!oidc",
            is_admin=False,
        )
        db.add(user)
        db.flush()
        db.add(
            UserIdentity(
                user_id=user.id,
                provider="google",
                provider_subject="linked-google-sub",
                email="linked@example.com",
                email_verified=True,
            )
        )
        set_registration_enabled(db, False)
        db.commit()
    finally:
        db.close()

    _install_google_fakes(
        monkeypatch,
        {
            "sub": "linked-google-sub",
            "email": "linked@example.com",
            "email_verified": True,
        },
    )

    state = _start_google_login(client)
    exchange_code = _complete_google_callback(client, state)

    response = client.post(
        "/api/auth/oidc/google/exchange", json={"code": exchange_code}
    )
    assert response.status_code == 200
    assert response.json()["user"]["username"] == "linked@example.com"


def test_oidc_state_expires(oidc_client, monkeypatch):
    client, SessionLocal = oidc_client
    _create_admin(SessionLocal)
    _install_authlib_login_fake(monkeypatch)

    state = _start_google_login(client)

    async def fake_invalid_state(request, *, redirect_uri: str):
        raise ValueError("invalid_state")

    monkeypatch.setattr(
        "xagent.web.api.oidc_google.complete_google_oidc_authorization",
        fake_invalid_state,
    )
    response = client.get(
        f"/api/auth/oidc/google/callback?code=google-code&state={state}",
        follow_redirects=False,
    )

    assert response.status_code == 307
    query = parse_qs(urlparse(response.headers["location"]).query)
    assert query["oidc_error"] == ["invalid_state"]


def test_google_oidc_login_uses_configured_state_ttl(oidc_client, monkeypatch):
    monkeypatch.setenv("XAGENT_OIDC_LOGIN_TTL_SECONDS", "7")

    from xagent.web.api.oidc_google import _google_oauth_client

    class RequestStub:
        session: dict[str, object] = {}

    started_at = time.time()
    client = _google_oauth_client()
    request = RequestStub()
    asyncio.run(
        client.save_authorize_data(
            request,
            state="short-state",
            redirect_uri="http://testserver/api/auth/oidc/google/callback",
        )
    )

    assert client.framework.expires_in == 7
    assert request.session["_state_google_short-state"]["exp"] == pytest.approx(
        started_at + 7,
        abs=2,
    )
