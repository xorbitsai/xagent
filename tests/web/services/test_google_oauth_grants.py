from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.web.models.database import Base
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.services.google_oauth_grants import (
    GoogleOAuthGrantConflict,
    create_google_oauth_grant_association,
    google_oauth_client_fingerprint,
    google_oauth_grant_covers_scopes,
    resolve_google_shared_oauth_account,
)


@pytest.fixture()
def db_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'google-grants.db'}")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False)()
    user = User(username="alice", password_hash="x", is_admin=False)
    session.add(user)
    session.commit()
    session.refresh(user)
    yield session, user
    session.close()
    engine.dispose()


def _account(
    db,
    user: User,
    *,
    owner: str | None = None,
    subject: str = "google-subject",
    provider: str = "google",
    token: str = "shared-access",
    scopes: str = "openid scope.drive scope.sheets",
) -> UserOAuth:
    account = UserOAuth(
        user_id=int(user.id),
        provider=provider,
        resource_owner_key=owner,
        provider_user_id=subject,
        access_token=token,
        refresh_token="refresh",
        scope=scopes,
    )
    db.add(account)
    db.commit()
    db.refresh(account)
    return account


def test_shared_grant_resolution_requires_app_client_and_scopes(db_session):
    db, user = db_session
    account = _account(db, user)
    association = create_google_oauth_grant_association(
        db,
        oauth_account=account,
        oauth_client_id="client-a",
        app_id="google-sheets",
    )
    db.commit()

    assert association.grant.oauth_client_fingerprint == (
        google_oauth_client_fingerprint("client-a")
    )
    assert (
        resolve_google_shared_oauth_account(
            db,
            user_id=int(user.id),
            resource_owner_key=None,
            oauth_client_id="client-a",
            app_id="google-sheets",
            required_scopes=["scope.sheets", "openid"],
        )
        == account
    )
    assert (
        resolve_google_shared_oauth_account(
            db,
            user_id=int(user.id),
            resource_owner_key=None,
            oauth_client_id="client-b",
            app_id="google-sheets",
        )
        is None
    )
    assert (
        resolve_google_shared_oauth_account(
            db,
            user_id=int(user.id),
            resource_owner_key=None,
            oauth_client_id="client-a",
            app_id="google-drive",
        )
        is None
    )
    assert (
        resolve_google_shared_oauth_account(
            db,
            user_id=int(user.id),
            resource_owner_key=None,
            oauth_client_id="client-a",
            app_id="google-sheets",
            required_scopes=["scope.not-granted"],
        )
        is None
    )


def test_newest_association_fails_closed_instead_of_using_older_subject(db_session):
    db, user = db_session
    older = _account(db, user, subject="subject-a", token="older-token")
    create_google_oauth_grant_association(
        db,
        oauth_account=older,
        oauth_client_id="client-a",
        app_id="google-drive",
    )
    newer = _account(
        db,
        user,
        subject="subject-b",
        token="newer-token",
        scopes="openid",
    )
    create_google_oauth_grant_association(
        db,
        oauth_account=newer,
        oauth_client_id="client-a",
        app_id="google-drive",
    )
    db.commit()

    assert (
        resolve_google_shared_oauth_account(
            db,
            user_id=int(user.id),
            resource_owner_key=None,
            oauth_client_id="client-a",
            app_id="google-drive",
            required_scopes=["scope.drive"],
        )
        is None
    )


def test_newest_tombstoned_association_does_not_use_older_subject(db_session):
    db, user = db_session
    older = _account(db, user, subject="subject-a", token="older-token")
    create_google_oauth_grant_association(
        db,
        oauth_account=older,
        oauth_client_id="client-a",
        app_id="google-drive",
    )
    newer = _account(db, user, subject="subject-b", token="newer-token")
    create_google_oauth_grant_association(
        db,
        oauth_account=newer,
        oauth_client_id="client-a",
        app_id="google-drive",
    )
    newer.access_token = ""
    newer.refresh_token = None
    db.commit()

    assert (
        resolve_google_shared_oauth_account(
            db,
            user_id=int(user.id),
            resource_owner_key=None,
            oauth_client_id="client-a",
            app_id="google-drive",
        )
        is None
    )


def test_shared_grants_are_isolated_by_resource_owner(db_session):
    db, user = db_session
    ordinary = _account(db, user, token="ordinary-token")
    actor = _account(
        db,
        user,
        owner="toby:workspace:actor",
        token="actor-token",
    )
    for account in (ordinary, actor):
        create_google_oauth_grant_association(
            db,
            oauth_account=account,
            oauth_client_id="client-a",
            app_id="google-drive",
        )
    db.commit()

    resolved_ordinary = resolve_google_shared_oauth_account(
        db,
        user_id=int(user.id),
        resource_owner_key=None,
        oauth_client_id="client-a",
        app_id="google-drive",
    )
    resolved_actor = resolve_google_shared_oauth_account(
        db,
        user_id=int(user.id),
        resource_owner_key="toby:workspace:actor",
        oauth_client_id="client-a",
        app_id="google-drive",
    )

    assert resolved_ordinary is not None
    assert resolved_ordinary.access_token == "ordinary-token"
    assert resolved_actor is not None
    assert resolved_actor.access_token == "actor-token"


def test_association_rejects_cross_owner_legacy_account(db_session):
    db, user = db_session
    shared = _account(db, user)
    actor_legacy = _account(
        db,
        user,
        owner="toby:workspace:actor",
        provider="google-drive",
    )

    with pytest.raises(GoogleOAuthGrantConflict, match="crosses"):
        create_google_oauth_grant_association(
            db,
            oauth_account=shared,
            oauth_client_id="client-a",
            app_id="google-drive",
            legacy_oauth_account=actor_legacy,
        )


def test_association_rejects_legacy_account_without_exact_subject(db_session):
    db, user = db_session
    shared = _account(db, user)
    legacy = _account(
        db,
        user,
        subject="",
        provider="google-drive",
    )

    with pytest.raises(GoogleOAuthGrantConflict, match="crosses"):
        create_google_oauth_grant_association(
            db,
            oauth_account=shared,
            oauth_client_id="client-a",
            app_id="google-drive",
            legacy_oauth_account=legacy,
        )


def test_association_rejects_shared_account_as_legacy_reference(db_session):
    db, user = db_session
    shared = _account(db, user)

    with pytest.raises(GoogleOAuthGrantConflict, match="own legacy"):
        create_google_oauth_grant_association(
            db,
            oauth_account=shared,
            oauth_client_id="client-a",
            app_id="google",
            legacy_oauth_account=shared,
        )


def test_association_is_idempotent_and_preserves_legacy_reference(db_session):
    db, user = db_session
    shared = _account(db, user)
    legacy = _account(db, user, provider="google-drive")

    first = create_google_oauth_grant_association(
        db,
        oauth_account=shared,
        oauth_client_id="client-a",
        app_id="google-drive",
        legacy_oauth_account=legacy,
    )
    db.commit()
    second = create_google_oauth_grant_association(
        db,
        oauth_account=shared,
        oauth_client_id="client-a",
        app_id="google-drive",
        legacy_oauth_account=legacy,
    )

    assert second.id == first.id
    assert second.legacy_oauth_account_id == legacy.id
    assert google_oauth_grant_covers_scopes(shared, ["scope.drive"])

    with pytest.raises(GoogleOAuthGrantConflict, match="another Google grant"):
        create_google_oauth_grant_association(
            db,
            oauth_account=shared,
            oauth_client_id="client-b",
            app_id="google-drive",
        )
