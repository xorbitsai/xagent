from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.parse import parse_qs, urlparse

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.utils.encryption import encrypt_value
from xagent.web.api import auth as auth_api
from xagent.web.api.auth import (
    create_access_token,
    generic_oauth_callback,
    generic_oauth_login,
)
from xagent.web.models.database import Base
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth


class MockResponse:
    def __init__(self, json_data=None, status_code: int = 200, text: str = ""):
        self._json_data = json_data or {}
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._json_data


@pytest.mark.parametrize(
    "raw_business_id,expected",
    [
        (
            "11111111-2222-3333-4444-555555555555",
            "11111111-2222-3333-4444-555555555555",
        ),
        # Case-insensitivity: MYOB's own GUIDs are typically uppercase.
        (
            "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE",
            "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE",
        ),
        # Surrounding whitespace is stripped before validating, unlike
        # require_clean_identifier's stricter rejection elsewhere -- this
        # value comes straight off the raw callback query string, not a
        # caller-typed id, so tolerating incidental whitespace here is safe.
        (
            "  11111111-2222-3333-4444-555555555555  ",
            "11111111-2222-3333-4444-555555555555",
        ),
        ("not-a-guid", None),
        ("11111111-2222-3333-4444-55555555555", None),  # one hex digit short
        ("", None),
        ("   ", None),
        (None, None),
        (123, None),
    ],
)
def test_normalize_myob_business_id(raw_business_id, expected):
    assert auth_api._normalize_myob_business_id(raw_business_id) == expected


@pytest.fixture()
def db_session(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = SessionLocal()

    user = User(username="alice", password_hash="x", is_admin=False)
    db.add(user)
    db.add(
        PublicMCPApp(
            app_id="myob",
            name="MYOB",
            description="Connect to MYOB AccountRight.",
            transport="oauth",
            provider_name="myob",
            category="Operations",
            oauth_scopes=["sme-company-file", "sme-contacts-customer"],
            is_visible_in_connector=True,
            launch_config={
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.myob"],
                "env_mapping": {
                    "MYOB_ACCESS_TOKEN": "access_token",
                    "MYOB_BUSINESS_ID": "instance_url",
                },
                "static_env": {"MYOB_API_KEY": "MYOB_CLIENT_ID"},
            },
        )
    )
    db.commit()
    db.refresh(user)

    yield db, user
    db.close()
    engine.dispose()


def _myob_provider() -> SimpleNamespace:
    return SimpleNamespace(
        provider_name="myob",
        client_id=encrypt_value("myob-client-id"),
        client_secret=encrypt_value("myob-client-secret"),
        auth_url="https://secure.myob.com/oauth2/account/authorize/",
        token_url="https://secure.myob.com/oauth2/v1/authorize/",
        redirect_uri="https://app.example.com/api/auth/myob/callback",
        userinfo_url="",
        user_id_path="",
        email_path="",
        default_scopes=[],
    )


def _token_for(user: User) -> str:
    return create_access_token(
        data={"sub": user.username, "type": "access"},
        expires_delta=timedelta(minutes=5),
    )


def _callback_request(
    db, user, provider: str = "myob", *, business_id: str | None = None
):
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": provider,
            "app_id": provider,
        },
        expires_delta=timedelta(minutes=10),
    )
    query_params = {"code": "myob-code", "state": state}
    if business_id is not None:
        query_params["businessId"] = business_id
    return SimpleNamespace(query_params=query_params)


BUSINESS_ID = "11111111-2222-3333-4444-555555555555"


def test_login_sets_prompt_consent_for_myob(db_session):
    """Without prompt=consent, MYOB never appends businessId to the
    authorization redirect at all -- the callback's businessId guard would
    then always trigger, so this must hold for every MYOB login, not just
    the resource-owner flow."""
    db, user = db_session
    token = _token_for(user)

    provider = SimpleNamespace(
        client_id=encrypt_value("myob-client-id"),
        auth_url="https://secure.myob.com/oauth2/account/authorize/",
        redirect_uri="https://app.example.com/api/auth/myob/callback",
        default_scopes=[],
    )

    resp = generic_oauth_login(
        provider="myob",
        token=token,
        app_id=None,
        redirect=None,
        db=db,
        db_provider=provider,
    )
    url = resp.headers["location"]
    qs = parse_qs(urlparse(url).query)

    assert qs.get("prompt") == ["consent"], f"myob prompt missing: {url}"


def test_callback_persists_business_id_and_identity(db_session, monkeypatch):
    db, user = db_session
    mock_post = Mock(
        return_value=MockResponse(
            {
                "access_token": "myob-token",
                "refresh_token": "myob-refresh",
                "token_type": "bearer",
                "expires_in": 3600,
                "user": {"uid": "42", "username": "alice@acme.example"},
            }
        )
    )
    monkeypatch.setattr(auth_api.requests, "post", mock_post)

    response = generic_oauth_callback(
        "myob",
        _callback_request(db, user, business_id=BUSINESS_ID),
        db,
        _myob_provider(),
    )

    assert response.status_code == 200
    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "myob")
        .one()
    )
    assert oauth_account.access_token == "myob-token"
    assert oauth_account.refresh_token == "myob-refresh"
    assert oauth_account.instance_url == BUSINESS_ID
    assert oauth_account.provider_user_id == "42"
    assert oauth_account.email == "alice@acme.example"


def test_callback_rejects_missing_business_id_without_touching_prior_grant(
    db_session, monkeypatch
):
    """A callback with no businessId (prompt=consent missing/ignored) must
    be rejected before the delete-then-recreate persistence step -- letting
    it through would destroy a prior *working* grant while still reporting
    success, since instance_url is required for the connector to launch at
    all (launch_config.env_mapping)."""
    db, user = db_session
    existing = UserOAuth(
        user_id=user.id,
        provider="myob",
        access_token="old-working-token",
        instance_url="99999999-8888-7777-6666-555555555555",
    )
    db.add(existing)
    db.commit()

    mock_post = Mock(
        return_value=MockResponse(
            {
                "access_token": "myob-token",
                "user": {"uid": "42", "username": "alice@acme.example"},
            }
        )
    )
    monkeypatch.setattr(auth_api.requests, "post", mock_post)

    response = generic_oauth_callback(
        "myob", _callback_request(db, user, business_id=None), db, _myob_provider()
    )

    assert response.status_code == 400
    assert "businessId" in response.body.decode()
    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "myob")
        .one()
    )
    assert oauth_account.access_token == "old-working-token"
    assert oauth_account.instance_url == "99999999-8888-7777-6666-555555555555"


def test_callback_rejects_malformed_business_id(db_session, monkeypatch):
    db, user = db_session
    mock_post = Mock(return_value=MockResponse({"access_token": "myob-token"}))
    monkeypatch.setattr(auth_api.requests, "post", mock_post)

    response = generic_oauth_callback(
        "myob",
        _callback_request(db, user, business_id="not-a-guid"),
        db,
        _myob_provider(),
    )

    assert response.status_code == 400
    assert (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "myob")
        .first()
        is None
    )


def test_callback_reports_error_in_token_data_for_myob(db_session, monkeypatch):
    db, user = db_session
    mock_post = Mock(
        return_value=MockResponse(
            {"error": "invalid_grant", "error_description": "bad code"}
        )
    )
    monkeypatch.setattr(auth_api.requests, "post", mock_post)

    response = generic_oauth_callback(
        "myob",
        _callback_request(db, user, business_id=BUSINESS_ID),
        db,
        _myob_provider(),
    )

    assert response.status_code == 400
    assert (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "myob")
        .first()
        is None
    )


def test_callback_rejects_non_2xx_status_for_myob(db_session, monkeypatch):
    db, user = db_session
    mock_post = Mock(
        return_value=MockResponse(
            {"access_token": "myob-token"},
            status_code=502,
        )
    )
    monkeypatch.setattr(auth_api.requests, "post", mock_post)

    response = generic_oauth_callback(
        "myob",
        _callback_request(db, user, business_id=BUSINESS_ID),
        db,
        _myob_provider(),
    )

    assert response.status_code == 400
    assert (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "myob")
        .first()
        is None
    )


def test_callback_succeeds_when_identity_fields_are_absent(db_session, monkeypatch):
    """A deliberately lenient design, matching every other identity branch
    in this file: a token response with no recognizable `user` shape must
    still succeed the connect, leaving provider_user_id/email as None
    rather than failing the whole connect."""
    db, user = db_session
    mock_post = Mock(return_value=MockResponse({"access_token": "myob-token"}))
    monkeypatch.setattr(auth_api.requests, "post", mock_post)

    response = generic_oauth_callback(
        "myob",
        _callback_request(db, user, business_id=BUSINESS_ID),
        db,
        _myob_provider(),
    )

    assert response.status_code == 200
    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "myob")
        .one()
    )
    assert oauth_account.provider_user_id is None
    assert oauth_account.email is None
