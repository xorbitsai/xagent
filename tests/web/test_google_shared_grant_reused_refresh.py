"""A shared Google grant may reuse its stored refresh token only if Google
confirms the token covers the scopes the new authorization asked for.

When the user connects another Google app, Google can answer the code exchange
without a new refresh token. The callback used to keep the stored one and write
the new access token's scopes next to it. A refresh then returned the older,
narrower scopes, while the row kept claiming the new ones.
"""

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock
from urllib.parse import parse_qs, urlparse

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
from xagent.web.models.google_oauth_grant import GoogleOAuthAppAssociation
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.services.google_oauth_grants import (
    GoogleOAuthRefreshTokenRequired,
    stored_google_refresh_token_for_callback,
    write_google_shared_oauth_callback,
)

SUBJECT = "google-subject"
CLIENT_ID = "google-client-id"
DRIVE = "test-google-drive"
DOCS = "test-google-docs"


class _Response:
    def __init__(self, payload: dict[str, Any], status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = ""
        self.headers: dict[str, str] = {}

    def json(self) -> dict[str, Any]:
        return self._payload


@pytest.fixture()
def db_session(tmp_path, monkeypatch):
    monkeypatch.setenv("XAGENT_GOOGLE_SHARED_OAUTH_GRANTS_ENABLED", "true")
    monkeypatch.delenv("XAGENT_GOOGLE_RESTRICTED_SCOPES_ENABLED", raising=False)
    engine = create_engine(
        f"sqlite:///{tmp_path / 'google-shared.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    user = User(username="alice", password_hash="x", is_admin=False)
    session.add(user)
    session.add_all(
        [
            PublicMCPApp(
                app_id=app_id,
                name=app_id,
                transport="oauth",
                provider_name="google",
                oauth_scopes=[scope],
                is_visible_in_connector=True,
                launch_config={"command": "python", "args": ["-m", app_id]},
            )
            for app_id, scope in ((DRIVE, "scope.drive"), (DOCS, "scope.docs"))
        ]
    )
    session.commit()
    session.refresh(user)
    yield session, user
    session.close()
    engine.dispose()


def _provider() -> SimpleNamespace:
    return SimpleNamespace(
        provider_name="google",
        client_id=encrypt_value(CLIENT_ID),
        client_secret=encrypt_value("google-client-secret"),
        auth_url="https://accounts.google.com/o/oauth2/auth",
        token_url="https://oauth2.googleapis.com/token",
        redirect_uri="https://app.example.com/api/auth/google/callback",
        userinfo_url="https://www.googleapis.com/oauth2/v2/userinfo",
        user_id_path="id",
        email_path="email",
        default_scopes=["openid"],
    )


def _connect_drive(db, user) -> UserOAuth:
    """Drive is connected first: one shared credential with its own refresh token."""
    legacy = UserOAuth(
        user_id=int(user.id),
        provider=DRIVE,
        provider_user_id=SUBJECT,
        access_token="drive-access",
    )
    db.add(legacy)
    db.flush()
    shared, _, _ = write_google_shared_oauth_callback(
        db,
        user_id=int(user.id),
        resource_owner_key=None,
        oauth_client_id=CLIENT_ID,
        app_id=DRIVE,
        google_subject=SUBJECT,
        access_token="drive-access",
        refresh_token="stored-refresh",
        granted_scopes="openid scope.drive",
        requested_scopes=["openid", "scope.drive"],
        email="alice@example.com",
        token_type="Bearer",
        expires_at=None,
        legacy_oauth_account=legacy,
    )
    db.commit()
    return shared


def _docs_callback(db, user, monkeypatch, *, refresh_response: _Response):
    """Run the callback of a docs authorization whose code exchange carries no
    refresh token. The second token-endpoint call is the refresh check."""
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "google",
            "app_id": DOCS,
            "google_shared_mode": True,
            "google_shared_scopes": ["openid", "scope.drive", "scope.docs"],
        },
        expires_delta=timedelta(minutes=10),
    )
    post = Mock(
        side_effect=[
            _Response(
                {
                    "access_token": "docs-access",
                    "token_type": "Bearer",
                    "scope": "openid scope.drive scope.docs",
                    "expires_in": 3600,
                }
            ),
            refresh_response,
        ]
    )
    monkeypatch.setattr(auth_api.requests, "post", post)
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(return_value=_Response({"id": SUBJECT, "email": "alice@example.com"})),
    )
    request = SimpleNamespace(query_params={"code": "code", "state": state})
    response = generic_oauth_callback("google", request, db, _provider())
    return response, post


def _snapshot(db, shared_id: int) -> tuple[str, str]:
    db.expire_all()
    shared = db.get(UserOAuth, shared_id)
    return str(shared.refresh_token), str(shared.scope)


def test_a_reused_refresh_token_that_does_not_cover_the_new_scopes_asks_for_consent(
    db_session, monkeypatch
):
    db, user = db_session
    shared = _connect_drive(db, user)
    shared_id = int(shared.id)

    response, post = _docs_callback(
        db,
        user,
        monkeypatch,
        # Google answers the stored token's refresh with the old scopes only.
        refresh_response=_Response(
            {
                "access_token": "refreshed",
                "token_type": "Bearer",
                "scope": "openid scope.drive",
                "expires_in": 3600,
            }
        ),
    )

    assert response.status_code in (302, 307), response.body
    params = parse_qs(urlparse(response.headers["location"]).query)
    assert params["prompt"] == ["consent"]
    assert post.call_count == 2
    refresh_call = post.call_args_list[1]
    assert refresh_call.kwargs["data"]["grant_type"] == "refresh_token"
    assert refresh_call.kwargs["data"]["refresh_token"] == "stored-refresh"
    # Nothing was written: the row still says what Google confirmed.
    assert _snapshot(db, shared_id) == ("stored-refresh", "openid scope.drive")
    assert db.query(GoogleOAuthAppAssociation).count() == 1


def test_a_reused_refresh_token_that_covers_the_new_scopes_is_kept(
    db_session, monkeypatch
):
    db, user = db_session
    shared = _connect_drive(db, user)
    shared_id = int(shared.id)

    response, _post = _docs_callback(
        db,
        user,
        monkeypatch,
        refresh_response=_Response(
            {
                "access_token": "refreshed",
                "token_type": "Bearer",
                "scope": "openid scope.drive scope.docs",
                "expires_in": 3600,
            }
        ),
    )

    assert response.status_code == 200, response.body
    refresh_token, scope = _snapshot(db, shared_id)
    assert refresh_token == "stored-refresh"
    assert set(scope.split()) == {"openid", "scope.drive", "scope.docs"}
    assert db.query(GoogleOAuthAppAssociation).count() == 2


@pytest.mark.parametrize(
    "refresh_response",
    [
        _Response({"error": "invalid_grant"}, 400),
        _Response({"error": "backend_error"}, 503),
        _Response({"access_token": "refreshed", "token_type": "Bearer"}),
    ],
    ids=["revoked", "unavailable", "no-scope"],
)
def test_a_stored_refresh_token_google_cannot_vouch_for_is_not_reused(
    db_session, monkeypatch, refresh_response
):
    db, user = db_session
    shared = _connect_drive(db, user)
    shared_id = int(shared.id)

    response, _post = _docs_callback(
        db, user, monkeypatch, refresh_response=refresh_response
    )

    assert response.status_code in (302, 307), response.body
    assert parse_qs(urlparse(response.headers["location"]).query)["prompt"] == [
        "consent"
    ]
    assert _snapshot(db, shared_id) == ("stored-refresh", "openid scope.drive")


def test_the_writer_refuses_to_reuse_a_refresh_token_nobody_verified(db_session):
    db, user = db_session
    shared = _connect_drive(db, user)
    shared_id = int(shared.id)
    docs = UserOAuth(
        user_id=int(user.id),
        provider=DOCS,
        provider_user_id=SUBJECT,
        access_token="docs-access",
    )
    db.add(docs)
    db.flush()
    arguments: dict[str, Any] = dict(
        user_id=int(user.id),
        resource_owner_key=None,
        oauth_client_id=CLIENT_ID,
        app_id=DOCS,
        google_subject=SUBJECT,
        access_token="docs-access",
        refresh_token=None,
        granted_scopes="openid scope.drive scope.docs",
        requested_scopes=["openid", "scope.drive", "scope.docs"],
        email="alice@example.com",
        token_type="Bearer",
        expires_at=None,
        legacy_oauth_account=docs,
    )

    with pytest.raises(GoogleOAuthRefreshTokenRequired):
        write_google_shared_oauth_callback(db, **arguments)
    db.rollback()
    assert _snapshot(db, shared_id) == ("stored-refresh", "openid scope.drive")

    updated, _legacy, _association = write_google_shared_oauth_callback(
        db, stored_refresh_scopes_verified=True, **arguments
    )
    assert updated.refresh_token == "stored-refresh"


def test_a_new_refresh_token_needs_no_verification(db_session):
    db, user = db_session
    _connect_drive(db, user)
    docs = UserOAuth(
        user_id=int(user.id),
        provider=DOCS,
        provider_user_id=SUBJECT,
        access_token="docs-access",
    )
    db.add(docs)
    db.flush()

    updated, _legacy, _association = write_google_shared_oauth_callback(
        db,
        user_id=int(user.id),
        resource_owner_key=None,
        oauth_client_id=CLIENT_ID,
        app_id=DOCS,
        google_subject=SUBJECT,
        access_token="docs-access",
        refresh_token="new-refresh",
        granted_scopes="openid scope.drive scope.docs",
        requested_scopes=["openid", "scope.drive", "scope.docs"],
        email="alice@example.com",
        token_type="Bearer",
        expires_at=None,
        legacy_oauth_account=docs,
    )

    assert updated.refresh_token == "new-refresh"


def test_the_stored_refresh_token_is_the_one_the_writer_would_reuse(db_session):
    db, user = db_session
    legacy = UserOAuth(
        user_id=int(user.id),
        provider=DOCS,
        provider_user_id=SUBJECT,
        access_token="docs-access",
        refresh_token="legacy-refresh",
    )
    db.add(legacy)
    db.commit()

    def stored(client_id: str = CLIENT_ID) -> str | None:
        return stored_google_refresh_token_for_callback(
            db,
            user_id=int(user.id),
            resource_owner_key=None,
            oauth_client_id=client_id,
            google_subject=SUBJECT,
            legacy_oauth_account=legacy,
        )

    # No shared grant yet: the same-subject legacy row is what a callback reuses.
    assert stored() == "legacy-refresh"

    _connect_drive(db, user)
    # The shared credential wins over the legacy row.
    assert stored() == "stored-refresh"
    # A rotated OAuth client needs a fresh token, so nothing is reusable.
    assert stored("another-client") is None
