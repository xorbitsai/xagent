"""The callback of a provider that needs a refresh token (Xero) does not save
an authorization that came back without one.

Without a refresh token the grant stops working when its access token expires,
yet the connection shows as connected when it is made; and the callback's
delete-then-recreate step would also drop a prior working grant. Providers that
issue non-expiring tokens and no refresh token must keep saving.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from tests.shared.oauth_browser_fixture import (
    create_access_token,
    generic_oauth_callback,
)
from xagent.core.utils.encryption import encrypt_value
from xagent.web.api import auth as auth_api
from xagent.web.models.database import Base
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth


class MockResponse:
    def __init__(self, json_data=None, status_code: int = 200):
        self._json_data = json_data or {}
        self.status_code = status_code
        self.text = ""

    def json(self):
        return self._json_data


@pytest.fixture()
def db_session(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    user = User(username="alice", password_hash="x", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    yield db, user
    db.close()
    engine.dispose()


def _provider(name: str, userinfo_url: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        provider_name=name,
        client_id=encrypt_value("client-id"),
        client_secret=encrypt_value("client-secret"),
        auth_url=f"https://auth.example.com/{name}/authorize",
        token_url=f"https://auth.example.com/{name}/token",
        redirect_uri=f"https://app.example.com/api/auth/{name}/callback",
        userinfo_url=userinfo_url,
        user_id_path="sub",
        email_path="email",
        default_scopes=["openid", "offline_access"],
    )


def _callback_request(user, provider: str) -> SimpleNamespace:
    state = create_access_token(
        data={"type": "oauth_state", "user_id": user.id, "provider": provider},
        expires_delta=timedelta(minutes=10),
    )
    return SimpleNamespace(query_params={"code": "the-code", "state": state})


def _connect(db, user, monkeypatch, provider: str, token_response: dict):
    monkeypatch.setattr(
        auth_api.requests, "post", Mock(return_value=MockResponse(token_response))
    )
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(return_value=MockResponse({"sub": "u-1", "email": "alice@acme.example"})),
    )
    userinfo_url = (
        "https://identity.example.com/userinfo" if provider.startswith("xero") else ""
    )
    return generic_oauth_callback(
        provider,
        _callback_request(user, provider),
        db,
        _provider(provider, userinfo_url),
    )


def _accounts(db, user, provider: str) -> list[UserOAuth]:
    return (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == provider)
        .all()
    )


def test_xero_grant_with_a_refresh_token_is_saved(db_session, monkeypatch):
    db, user = db_session

    response = _connect(
        db,
        user,
        monkeypatch,
        "xero",
        {
            "access_token": "new-access",
            "refresh_token": "new-refresh",
            "expires_in": 1800,
            "scope": "openid offline_access",
        },
    )

    assert response.status_code == 200
    (account,) = _accounts(db, user, "xero")
    assert account.access_token == "new-access"
    assert account.refresh_token == "new-refresh"


@pytest.mark.parametrize(
    ("missing", "reason"),
    [
        ({}, "refresh_token_absent"),
        ({"refresh_token": None}, "refresh_token_null"),
        ({"refresh_token": ""}, "refresh_token_empty"),
        ({"refresh_token": 12345}, "refresh_token_not_a_string"),
    ],
)
def test_xero_grant_without_a_usable_refresh_token_is_refused_and_logged(
    db_session, monkeypatch, caplog, missing, reason
):
    db, user = db_session

    with caplog.at_level(logging.WARNING, logger=auth_api.logger.name):
        response = _connect(
            db,
            user,
            monkeypatch,
            "xero",
            {
                "access_token": "short-lived-access",
                "expires_in": 1800,
                "scope": "openid offline_access",
                **missing,
            },
        )

    assert response.status_code == 400
    body = response.body.decode()
    assert "Cannot connect xero" in body
    assert "Nothing was changed" in body
    assert _accounts(db, user, "xero") == []
    # The reason is a code for the shape only, and no token reaches the log.
    assert f"reason={reason}" in caplog.text
    assert "short-lived-access" not in caplog.text


def test_refused_xero_grant_keeps_the_prior_working_grant(db_session, monkeypatch):
    db, user = db_session
    db.add(
        UserOAuth(
            user_id=user.id,
            provider="xero",
            access_token="old-access",
            refresh_token="old-refresh",
            provider_user_id="u-1",
        )
    )
    db.commit()

    response = _connect(
        db,
        user,
        monkeypatch,
        "xero",
        {"access_token": "new-access", "expires_in": 1800},
    )

    assert response.status_code == 400
    (account,) = _accounts(db, user, "xero")
    assert account.access_token == "old-access"
    assert account.refresh_token == "old-refresh"


def test_xero_variant_row_is_held_to_the_same_rule(db_session, monkeypatch):
    db, user = db_session

    response = _connect(
        db, user, monkeypatch, "xero-sandbox", {"access_token": "new-access"}
    )

    assert response.status_code == 400
    assert _accounts(db, user, "xero-sandbox") == []


def test_provider_outside_the_family_still_saves_a_grant_without_a_refresh_token(
    db_session, monkeypatch
):
    # A provider whose tokens do not expire issues no refresh token at all.
    db, user = db_session

    response = _connect(
        db, user, monkeypatch, "acme-tokens-never-expire", {"access_token": "token"}
    )

    assert response.status_code == 200
    (account,) = _accounts(db, user, "acme-tokens-never-expire")
    assert account.access_token == "token"
    assert account.refresh_token is None
