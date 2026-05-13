"""Regression tests for generic_oauth_login URL construction.

Covers two bugs the PR fixed:
  1. When auth_url already contains a query string, params must be appended
     with '&' (no second '?').
  2. Zoom provider must include `prompt=login` in the redirect URL.
"""

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.utils.encryption import encrypt_value
from xagent.web.api.auth import create_access_token, generic_oauth_login
from xagent.web.models.database import Base
from xagent.web.models.user import User

# ---------- helpers ---------------------------------------------------------


@pytest.fixture()
def db_session(tmp_path):
    """Fresh SQLite DB + a single user for each test."""
    db_path = tmp_path / "test.db"
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = SessionLocal()

    user = User(username="alice", password_hash="x", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)

    yield db, user
    db.close()
    engine.dispose()


def _token_for(user: User) -> str:
    return create_access_token(
        data={"sub": user.username, "type": "access"},
        expires_delta=timedelta(minutes=5),
    )


def _provider(auth_url: str, default_scopes=None, redirect_uri=None):
    """A duck-typed stand-in for the OAuthProvider ORM row."""
    return SimpleNamespace(
        client_id=encrypt_value("test-client-id"),
        auth_url=auth_url,
        redirect_uri=redirect_uri,
        default_scopes=default_scopes or [],
    )


def _location(response) -> str:
    # RedirectResponse stores the target in the Location header.
    return response.headers["location"]


# ---------- the actual regression checks ------------------------------------


def test_auth_url_with_query_uses_ampersand_separator(db_session):
    """If db_provider.auth_url already has '?', params must be appended with '&'."""
    db, user = db_session
    token = _token_for(user)

    provider = _provider(
        auth_url="https://example.com/oauth/authorize?tenant=acme",
        default_scopes=["openid", "profile"],
        redirect_uri="https://app.example.com/cb",
    )

    resp = generic_oauth_login(
        provider="custom",
        token=token,
        app_id=None,
        redirect=None,
        db=db,
        db_provider=provider,
    )
    url = _location(resp)

    # Only one '?' allowed in the whole URL — this is the regression.
    assert url.count("?") == 1, f"second '?' leaked into URL: {url}"

    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    qs = parse_qs(parsed.query)

    assert base == "https://example.com/oauth/authorize"
    assert qs["tenant"] == ["acme"], "pre-existing query param dropped"
    assert qs["client_id"] == ["test-client-id"]
    assert qs["redirect_uri"] == ["https://app.example.com/cb"]
    assert qs["response_type"] == ["code"]
    assert "state" in qs


def test_auth_url_without_query_uses_question_mark(db_session):
    db, user = db_session
    token = _token_for(user)

    provider = _provider(
        auth_url="https://example.com/oauth/authorize",
        default_scopes=["openid"],
        redirect_uri="https://app.example.com/cb",
    )

    resp = generic_oauth_login(
        provider="custom",
        token=token,
        app_id=None,
        redirect=None,
        db=db,
        db_provider=provider,
    )
    url = _location(resp)

    assert url.count("?") == 1
    assert url.startswith("https://example.com/oauth/authorize?")


def test_zoom_provider_sets_prompt_login(db_session):
    db, user = db_session
    token = _token_for(user)

    provider = _provider(
        auth_url="https://zoom.us/oauth/authorize",
        default_scopes=["user:read"],
        redirect_uri="https://app.example.com/cb",
    )

    resp = generic_oauth_login(
        provider="zoom",
        token=token,
        app_id=None,
        redirect=None,
        db=db,
        db_provider=provider,
    )
    url = _location(resp)
    qs = parse_qs(urlparse(url).query)

    assert qs.get("prompt") == ["login"], f"zoom prompt missing: {url}"


def test_non_zoom_provider_does_not_set_prompt_login(db_session):
    """Sanity: only Zoom gets prompt=login (Google gets prompt=consent, others none)."""
    db, user = db_session
    token = _token_for(user)

    provider = _provider(
        auth_url="https://example.com/oauth/authorize",
        default_scopes=["openid"],
        redirect_uri="https://app.example.com/cb",
    )

    resp = generic_oauth_login(
        provider="custom",
        token=token,
        app_id=None,
        redirect=None,
        db=db,
        db_provider=provider,
    )
    qs = parse_qs(urlparse(_location(resp)).query)
    assert "prompt" not in qs
