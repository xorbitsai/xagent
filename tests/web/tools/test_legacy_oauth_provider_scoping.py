"""Regression tests: a bare provider-level Meta grant (UserOAuth.provider ==
"meta", created by the app_id-less connect flow) must not satisfy the
Facebook Pages connector, since that flow never requested Facebook's
app-specific oauth_scopes (e.g. pages_read_user_content). Instagram's shared
"meta" grant must keep working, since its required scopes haven't changed.
"""

import asyncio

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.web.api.mcp import _oauth_keys_for_app
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
