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


def test_requires_app_scoped_oauth_grant_github_addition_leaves_meta_unaffected():
    """github was added to APPS_REQUIRING_APP_SCOPED_OAUTH_GRANT alongside
    facebook -- pin that this didn't also flip instagram or the bare
    "meta" provider string, which must stay unaffected."""
    assert requires_app_scoped_oauth_grant("github") is True
    assert requires_app_scoped_oauth_grant("GitHub") is True
    assert requires_app_scoped_oauth_grant("instagram") is False
    assert requires_app_scoped_oauth_grant("meta") is False


def test_requires_app_scoped_oauth_grant_covers_myob():
    """MYOB's oauth_providers row seeds an empty default_scopes (there is no
    shared identity scope; every functional sme-* scope lives solely on the
    app row) -- an even more extreme version of github's situation, since a
    bare grant here would request zero scopes, not just an under-scoped
    identity-only set. Pin membership the same way as github's own
    regression test above, and confirm it didn't flip anything unrelated."""
    assert requires_app_scoped_oauth_grant("myob") is True
    assert requires_app_scoped_oauth_grant("MyOB") is True
    assert requires_app_scoped_oauth_grant("instagram") is False
    assert requires_app_scoped_oauth_grant("meta") is False


def test_restrict_to_app_scoped_oauth_grant_dedupes_and_preserves_order():
    assert restrict_to_app_scoped_oauth_grant(
        "instagram", ["meta", "meta", "instagram", None, ""]
    ) == ["meta", "instagram"]


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


def test_ordinary_token_resolution_ignores_actor_owned_grant(db_session):
    db_session.add(
        UserOAuth(
            user_id=1,
            provider="instagram",
            resource_owner_key="toby:slack:41:UALICE",
            access_token="actor-token",
        )
    )
    db_session.commit()

    cfg = WebToolConfig(db=None, request=None, db_factory=lambda: db_session, user_id=1)

    resolution = asyncio.run(
        cfg._resolve_legacy_oauth_access_token(
            provider_name="meta",
            app_id="instagram",
        )
    )

    assert resolution.access_token is None


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


def test_app_scoped_token_resolution_picks_newest_row_on_tie(db_session):
    """More than one UserOAuth row for the same (user, app-scoped provider
    set) shouldn't normally exist, but a provider whose identity backfill
    can't always derive a non-NULL provider_user_id (e.g. Employment Hero
    with zero accessible organisations) can leave more than one row behind
    after a race. Resolution must deterministically prefer the
    newest-created row rather than whatever order the backend happens to
    return, so a stale token left behind by a lost race isn't the one
    silently used."""
    db_session.add(
        UserOAuth(
            user_id=1,
            provider="facebook",
            access_token="stale-facebook-token",
        )
    )
    db_session.commit()
    db_session.add(
        UserOAuth(
            user_id=1,
            provider="facebook",
            access_token="fresh-facebook-token",
        )
    )
    db_session.commit()

    cfg = WebToolConfig(db=None, request=None, db_factory=lambda: db_session, user_id=1)

    resolution = asyncio.run(
        cfg._resolve_legacy_oauth_access_token(provider_name="meta", app_id="facebook")
    )

    assert resolution.access_token == "fresh-facebook-token"


def test_bare_provider_token_resolution_picks_newest_row_on_tie(db_session):
    """Same tie-break as the app-scoped case above, but for the bare
    (app_id-less) provider-lookup branch."""
    db_session.add(
        UserOAuth(
            user_id=1,
            provider="meta",
            access_token="stale-meta-token",
        )
    )
    db_session.commit()
    db_session.add(
        UserOAuth(
            user_id=1,
            provider="meta",
            access_token="fresh-meta-token",
        )
    )
    db_session.commit()

    cfg = WebToolConfig(db=None, request=None, db_factory=lambda: db_session, user_id=1)

    resolution = asyncio.run(
        cfg._resolve_legacy_oauth_access_token(provider_name="meta", app_id=None)
    )

    assert resolution.access_token == "fresh-meta-token"
