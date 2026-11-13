"""A successful refresh never replaces the stored refresh token with nothing.

A provider that rotates refresh tokens sends the new one with each refresh. A
response whose ``refresh_token`` is null or empty is not a replacement: writing
it would leave a fresh access token and nothing to renew it with.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import xagent.web.api.auth  # noqa: F401  (see test_oauth_refresh_retry.py)
from xagent.core.utils.encryption import encrypt_value
from xagent.web.builtin_mcp_registry import get_builtin_oauth_provider_rows
from xagent.web.models.database import Base
from xagent.web.models.oauth_provider import OAuthProvider
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.tools import config as tool_config

STORED_REFRESH_TOKEN = "stored-refresh-token"


@pytest.fixture()
def db(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'refresh.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False)()
    (row,) = [
        row
        for row in get_builtin_oauth_provider_rows()
        if row["provider_name"] == "xero"
    ]
    session.add(
        OAuthProvider(
            **{
                **row,
                "client_id": encrypt_value("client-id"),
                "client_secret": encrypt_value("client-secret"),
                "redirect_uri": "https://app.example.com/api/auth/xero/callback",
            }
        )
    )
    session.add(User(username="alice", password_hash="x", is_admin=False))
    session.commit()
    yield session
    session.close()
    engine.dispose()


def _refresh(db, monkeypatch, response_body: dict) -> UserOAuth:
    monkeypatch.setattr(
        tool_config,
        "_oauth_refresh_http_client",
        lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=response_body)
            )
        ),
    )
    user = db.query(User).one()
    grant = UserOAuth(
        user_id=user.id,
        provider="xero",
        access_token="old-access",
        refresh_token=STORED_REFRESH_TOKEN,
        provider_user_id="1",
        expires_at=datetime.now(UTC) - timedelta(minutes=1),
    )
    db.add(grant)
    db.commit()
    assert asyncio.run(tool_config.refresh_oauth_token_if_needed(db, grant, "xero"))
    db.refresh(grant)
    return grant


def test_a_rotated_refresh_token_replaces_the_stored_one(db, monkeypatch):
    grant = _refresh(
        db,
        monkeypatch,
        {"access_token": "new-access", "refresh_token": "rotated", "expires_in": 1800},
    )

    assert grant.access_token == "new-access"
    assert grant.refresh_token == "rotated"


def test_a_response_without_a_refresh_token_keeps_the_stored_one(db, monkeypatch):
    grant = _refresh(
        db, monkeypatch, {"access_token": "new-access", "expires_in": 1800}
    )

    assert grant.access_token == "new-access"
    assert grant.refresh_token == STORED_REFRESH_TOKEN


@pytest.mark.parametrize(
    ("value", "shape"),
    [(None, "null"), ("", "empty"), (12345, "not a string")],
)
def test_an_unusable_refresh_token_in_the_response_keeps_the_stored_one(
    db, monkeypatch, caplog, value, shape
):
    with caplog.at_level(logging.WARNING, logger=tool_config.logger.name):
        grant = _refresh(
            db,
            monkeypatch,
            {"access_token": "new-access", "refresh_token": value, "expires_in": 1800},
        )

    # The access token still renews, so the grant keeps working.
    assert grant.access_token == "new-access"
    assert grant.refresh_token == STORED_REFRESH_TOKEN
    assert f"refresh_token that is {shape}" in caplog.text
    assert STORED_REFRESH_TOKEN not in caplog.text
