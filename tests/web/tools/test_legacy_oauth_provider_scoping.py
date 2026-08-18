"""Regression tests: a bare provider-level Meta grant (UserOAuth.provider ==
"meta", created by the app_id-less connect flow) must not satisfy the
Facebook Pages connector, since that flow never requested Facebook's
app-specific oauth_scopes (e.g. pages_read_user_content). Instagram's shared
"meta" grant must keep working, since its required scopes haven't changed.

The same policy covers a bare provider-level Google grant
(UserOAuth.provider == "google") and the Calendar connector: that bare grant
never requested Calendar's own oauth_scopes either, and Google's
include_granted_scopes=true means it could otherwise accumulate the old,
broad calendar scope from a separate authorization and satisfy Calendar
without ever going through Calendar's own connect flow. See
20260817_narrow_google_calendar_scope.py's module docstring for why the data
migration deliberately does not delete that bare row itself.
"""

import asyncio

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.web.api.mcp import _oauth_keys_for_app
from xagent.web.mcp_apps import (
    requires_app_scoped_oauth_grant,
    restrict_to_app_scoped_oauth_grant,
)
from xagent.web.models.database import Base
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.tools.config import WebToolConfig


def test_oauth_keys_for_facebook_excludes_bare_meta_provider():
    app = {"id": "facebook", "provider": "meta"}
    assert _oauth_keys_for_app(app) == ["facebook"]


def test_oauth_keys_for_instagram_still_includes_bare_meta_provider():
    app = {"id": "instagram", "provider": "meta"}
    keys = _oauth_keys_for_app(app)
    assert "instagram" in keys
    assert "meta" in keys


def test_restrict_to_app_scoped_oauth_grant_narrows_facebook():
    assert restrict_to_app_scoped_oauth_grant("facebook", ["meta", "facebook"]) == [
        "facebook"
    ]


def test_restrict_to_app_scoped_oauth_grant_passes_through_instagram():
    assert restrict_to_app_scoped_oauth_grant("instagram", ["meta", "instagram"]) == [
        "meta",
        "instagram",
    ]


def test_restrict_to_app_scoped_oauth_grant_normalizes_admin_created_app_id():
    """An admin-created PublicMCPApp.app_id is free-form (POST /admin/mcp/apps
    has no format validation). A differently-cased/whitespace-padded id must
    still match the policy — this is the exact drift MAJOR-4 flagged: two of
    three call sites used to compare raw strings and silently never filtered.
    """
    assert restrict_to_app_scoped_oauth_grant(" Facebook ", ["meta", "facebook"]) == [
        "facebook"
    ]
    assert requires_app_scoped_oauth_grant("Facebook") is True
    assert requires_app_scoped_oauth_grant("FACEBOOK") is True


def test_restrict_to_app_scoped_oauth_grant_dedupes_and_preserves_order():
    assert restrict_to_app_scoped_oauth_grant(
        "instagram", ["meta", "meta", "instagram", None, ""]
    ) == ["meta", "instagram"]


def test_oauth_keys_for_google_calendar_excludes_bare_google_provider():
    app = {"id": "google-calendar", "provider": "google"}
    assert _oauth_keys_for_app(app) == ["google-calendar"]


def test_oauth_keys_for_gmail_still_includes_bare_google_provider():
    app = {"id": "gmail", "provider": "google"}
    keys = _oauth_keys_for_app(app)
    assert "gmail" in keys
    assert "google" in keys


def test_restrict_to_app_scoped_oauth_grant_narrows_google_calendar():
    assert restrict_to_app_scoped_oauth_grant(
        "google-calendar", ["google", "google-calendar"]
    ) == ["google-calendar"]
    assert requires_app_scoped_oauth_grant("google-calendar") is True


def test_restrict_to_app_scoped_oauth_grant_passes_through_gmail():
    assert restrict_to_app_scoped_oauth_grant("gmail", ["google", "gmail"]) == [
        "google",
        "gmail",
    ]


@pytest.fixture()
def db_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = session_local()
    yield session
    session.close()
    engine.dispose()


def test_legacy_token_resolution_ignores_bare_meta_grant_for_facebook(db_session):
    db_session.add(
        UserOAuth(
            user_id=1,
            provider="meta",
            access_token="bare-meta-token",
        )
    )
    db_session.commit()

    cfg = WebToolConfig(db=None, request=None, db_factory=lambda: db_session, user_id=1)

    resolution = asyncio.run(
        cfg._resolve_legacy_oauth_access_token(provider_name="meta", app_id="facebook")
    )

    assert resolution.access_token is None


def test_legacy_token_resolution_still_uses_bare_meta_grant_for_instagram(db_session):
    db_session.add(
        UserOAuth(
            user_id=1,
            provider="meta",
            access_token="bare-meta-token",
        )
    )
    db_session.commit()

    cfg = WebToolConfig(db=None, request=None, db_factory=lambda: db_session, user_id=1)

    resolution = asyncio.run(
        cfg._resolve_legacy_oauth_access_token(provider_name="meta", app_id="instagram")
    )

    assert resolution.access_token == "bare-meta-token"


def test_legacy_token_resolution_uses_app_scoped_facebook_grant(db_session):
    db_session.add(
        UserOAuth(
            user_id=1,
            provider="facebook",
            access_token="app-scoped-facebook-token",
        )
    )
    db_session.commit()

    cfg = WebToolConfig(db=None, request=None, db_factory=lambda: db_session, user_id=1)

    resolution = asyncio.run(
        cfg._resolve_legacy_oauth_access_token(provider_name="meta", app_id="facebook")
    )

    assert resolution.access_token == "app-scoped-facebook-token"


def test_legacy_token_resolution_ignores_bare_google_grant_for_calendar(db_session):
    """The runtime half of the C1 fix: even if a bare "google" row somehow
    still carries the old calendar scope (e.g. it predates this migration,
    or a future reconnect recreates a combined grant), it must never be
    accepted as a Calendar credential."""
    db_session.add(
        UserOAuth(
            user_id=1,
            provider="google",
            access_token="bare-google-token",
            scope="https://www.googleapis.com/auth/calendar",
        )
    )
    db_session.commit()

    cfg = WebToolConfig(db=None, request=None, db_factory=lambda: db_session, user_id=1)

    resolution = asyncio.run(
        cfg._resolve_legacy_oauth_access_token(
            provider_name="google", app_id="google-calendar"
        )
    )

    assert resolution.access_token is None


def test_legacy_token_resolution_still_uses_bare_google_grant_for_gmail(db_session):
    db_session.add(
        UserOAuth(
            user_id=1,
            provider="google",
            access_token="bare-google-token",
        )
    )
    db_session.commit()

    cfg = WebToolConfig(db=None, request=None, db_factory=lambda: db_session, user_id=1)

    resolution = asyncio.run(
        cfg._resolve_legacy_oauth_access_token(provider_name="google", app_id="gmail")
    )

    assert resolution.access_token == "bare-google-token"


def test_legacy_token_resolution_uses_app_scoped_google_calendar_grant(db_session):
    db_session.add(
        UserOAuth(
            user_id=1,
            provider="google-calendar",
            access_token="app-scoped-calendar-token",
        )
    )
    db_session.commit()

    cfg = WebToolConfig(db=None, request=None, db_factory=lambda: db_session, user_id=1)

    resolution = asyncio.run(
        cfg._resolve_legacy_oauth_access_token(
            provider_name="google", app_id="google-calendar"
        )
    )

    assert resolution.access_token == "app-scoped-calendar-token"
