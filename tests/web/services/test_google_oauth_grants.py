from __future__ import annotations

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Query, sessionmaker

from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app
from xagent.web.models.database import Base
from xagent.web.models.google_oauth_grant import (
    GoogleOAuthAppAssociation,
    GoogleOAuthGrant,
)
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.services import google_oauth_grants as google_oauth_grants_service
from xagent.web.services.google_oauth_grants import (
    GoogleOAuthGrantConflict,
    GoogleOAuthRefreshTokenRequired,
    create_google_oauth_grant_association,
    detach_google_shared_oauth_app,
    detach_google_shared_oauth_legacy_account,
    google_oauth_authorization_context,
    google_oauth_client_fingerprint,
    google_oauth_grant_covers_scopes,
    mirror_shared_google_oauth_credentials,
    resolve_google_shared_oauth_account,
    write_google_shared_oauth_callback,
)
from xagent.web.tools.config import WebToolConfig


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
    legacy = _account(db, user, provider="google-sheets")
    association = create_google_oauth_grant_association(
        db,
        oauth_account=account,
        oauth_client_id="client-a",
        app_id="google-sheets",
        legacy_oauth_account=legacy,
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
        legacy = _account(
            db,
            user,
            owner=account.resource_owner_key,
            provider="google-drive",
        )
        create_google_oauth_grant_association(
            db,
            oauth_account=account,
            oauth_client_id="client-a",
            app_id="google-drive",
            legacy_oauth_account=legacy,
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


def test_authorization_context_rebuilds_scopes_from_current_catalog_links(db_session):
    db, user = db_session
    db.add(
        PublicMCPApp(
            app_id="test-google-mail",
            name="Test Google Mail",
            transport="oauth",
            provider_name="google",
            oauth_scopes=["https://www.googleapis.com/auth/gmail.modify"],
        )
    )
    db.commit()
    _account(
        db,
        user,
        provider="google-drive",
        scopes="openid scope.drive",
    )
    _account(
        db,
        user,
        provider="google-calendar",
        scopes="scope.calendar",
    )
    shared = _account(
        db,
        user,
        subject="other-google-subject",
        scopes="openid https://www.googleapis.com/auth/gmail.modify",
    )
    gmail = _account(
        db,
        user,
        subject="other-google-subject",
        provider="test-google-mail",
        scopes="https://www.googleapis.com/auth/gmail.modify",
    )
    create_google_oauth_grant_association(
        db,
        oauth_account=shared,
        oauth_client_id="client-a",
        app_id="test-google-mail",
        legacy_oauth_account=gmail,
    )
    db.commit()

    context = google_oauth_authorization_context(
        db,
        user_id=int(user.id),
        resource_owner_key=None,
        oauth_client_id="client-a",
        app_id="google-drive",
    )

    assert context.has_refresh_token is True
    assert "scope.drive" not in context.associated_scopes
    assert "scope.calendar" not in context.associated_scopes
    assert "https://www.googleapis.com/auth/gmail.modify" in context.associated_scopes


def test_sequential_callbacks_request_and_resolve_one_shared_scope_union(db_session):
    db, user = db_session
    db.add_all(
        [
            PublicMCPApp(
                app_id="test-google-drive",
                name="Test Google Drive",
                transport="oauth",
                provider_name="google",
                oauth_scopes=["scope.drive"],
            ),
            PublicMCPApp(
                app_id="test-google-docs",
                name="Test Google Docs",
                transport="oauth",
                provider_name="google",
                oauth_scopes=["scope.docs"],
            ),
        ]
    )
    db.commit()
    drive = _account(
        db,
        user,
        provider="test-google-drive",
        token="drive-access",
        scopes="openid scope.drive",
    )
    shared, _drive_mirror, _drive_association = write_google_shared_oauth_callback(
        db,
        user_id=int(user.id),
        resource_owner_key=None,
        oauth_client_id="client-a",
        app_id="test-google-drive",
        google_subject="google-subject",
        access_token="drive-access",
        refresh_token="shared-refresh",
        granted_scopes="openid scope.drive",
        requested_scopes=["openid", "scope.drive"],
        email="alice@example.com",
        token_type="Bearer",
        expires_at=None,
        legacy_oauth_account=drive,
    )
    db.commit()

    context = google_oauth_authorization_context(
        db,
        user_id=int(user.id),
        resource_owner_key=None,
        oauth_client_id="client-a",
        app_id="test-google-docs",
    )
    assert context.associated_scopes == frozenset({"scope.drive"})

    docs = _account(
        db,
        user,
        provider="test-google-docs",
        token="docs-access",
        scopes="openid scope.drive scope.docs",
    )
    updated, _docs_mirror, _docs_association = write_google_shared_oauth_callback(
        db,
        user_id=int(user.id),
        resource_owner_key=None,
        oauth_client_id="client-a",
        app_id="test-google-docs",
        google_subject="google-subject",
        access_token="docs-access",
        refresh_token=None,
        granted_scopes="openid scope.drive scope.docs",
        requested_scopes=["openid", "scope.drive", "scope.docs"],
        email="alice@example.com",
        token_type="Bearer",
        expires_at=None,
        legacy_oauth_account=docs,
    )
    db.commit()

    assert updated.id == shared.id
    assert updated.refresh_token == "shared-refresh"
    assert set(str(updated.scope).split()) == {"openid", "scope.drive", "scope.docs"}
    for app_id, required_scope in (
        ("test-google-drive", "scope.drive"),
        ("test-google-docs", "scope.docs"),
    ):
        resolved = resolve_google_shared_oauth_account(
            db,
            user_id=int(user.id),
            resource_owner_key=None,
            oauth_client_id="client-a",
            app_id=app_id,
            required_scopes=[required_scope],
        )
        assert resolved is not None
        assert resolved.id == shared.id


def test_callback_writer_preserves_refresh_and_shared_primary_key(db_session):
    db, user = db_session
    legacy = _account(
        db,
        user,
        provider="google-drive",
        token="legacy-access",
        scopes="openid scope.drive",
    )

    shared, mirrored, association = write_google_shared_oauth_callback(
        db,
        user_id=int(user.id),
        resource_owner_key=None,
        oauth_client_id="client-a",
        app_id="google-drive",
        google_subject="google-subject",
        access_token="shared-access-1",
        refresh_token=None,
        granted_scopes="openid scope.drive",
        requested_scopes=["openid", "scope.drive"],
        email="alice@example.com",
        token_type="Bearer",
        expires_at=None,
        legacy_oauth_account=legacy,
    )
    db.commit()
    shared_id = int(shared.id)

    assert mirrored.id == legacy.id
    assert shared.refresh_token == "refresh"
    assert mirrored.refresh_token == "refresh"
    assert association.legacy_oauth_account_id == legacy.id
    assert db.query(GoogleOAuthGrant).count() == 1
    assert db.query(GoogleOAuthAppAssociation).count() == 1

    updated, updated_legacy, updated_association = write_google_shared_oauth_callback(
        db,
        user_id=int(user.id),
        resource_owner_key=None,
        oauth_client_id="client-a",
        app_id="google-drive",
        google_subject="google-subject",
        access_token="shared-access-2",
        refresh_token=None,
        granted_scopes="scope.sheets",
        requested_scopes=["scope.sheets"],
        email="alice@example.com",
        token_type="Bearer",
        expires_at=None,
        legacy_oauth_account=legacy,
    )
    db.commit()

    assert int(updated.id) == shared_id
    assert updated.access_token == "shared-access-2"
    assert updated.refresh_token == "refresh"
    assert set(str(updated.scope).split()) == {"scope.sheets"}
    assert updated_legacy.access_token == "shared-access-2"
    assert updated_association.id == association.id

    rotated, _rotated_legacy, _rotated_association = write_google_shared_oauth_callback(
        db,
        user_id=int(user.id),
        resource_owner_key=None,
        oauth_client_id="client-b",
        app_id="google-drive",
        google_subject="google-subject",
        access_token="shared-access-3",
        refresh_token="rotated-refresh",
        granted_scopes="openid scope.drive scope.sheets",
        requested_scopes=["scope.drive"],
        email="alice@example.com",
        token_type="Bearer",
        expires_at=None,
        legacy_oauth_account=legacy,
    )
    db.commit()

    assert int(rotated.id) == shared_id
    assert db.query(GoogleOAuthGrant).one().oauth_client_fingerprint == (
        google_oauth_client_fingerprint("client-b")
    )
    assert set(str(rotated.scope).split()) == {
        "openid",
        "scope.drive",
        "scope.sheets",
    }


def test_client_rotation_requires_a_fresh_refresh_token(db_session):
    db, user = db_session
    legacy = _account(db, user, provider="google-drive")
    write_google_shared_oauth_callback(
        db,
        user_id=int(user.id),
        resource_owner_key=None,
        oauth_client_id="client-a",
        app_id="google-drive",
        google_subject="google-subject",
        access_token="shared-access",
        refresh_token="client-a-refresh",
        granted_scopes="openid scope.drive",
        requested_scopes=["openid", "scope.drive"],
        email="alice@example.com",
        token_type="Bearer",
        expires_at=None,
        legacy_oauth_account=legacy,
    )
    db.commit()

    with pytest.raises(GoogleOAuthRefreshTokenRequired):
        write_google_shared_oauth_callback(
            db,
            user_id=int(user.id),
            resource_owner_key=None,
            oauth_client_id="client-b",
            app_id="google-drive",
            google_subject="google-subject",
            access_token="client-b-access",
            refresh_token=None,
            granted_scopes="openid scope.drive",
            requested_scopes=["openid", "scope.drive"],
            email="alice@example.com",
            token_type="Bearer",
            expires_at=None,
            legacy_oauth_account=legacy,
        )
    db.rollback()
    assert db.query(GoogleOAuthGrant).one().oauth_client_fingerprint == (
        google_oauth_client_fingerprint("client-a")
    )


def test_callback_writer_adopts_existing_unowned_bare_google_account(db_session):
    db, user = db_session
    bare = _account(
        db,
        user,
        subject="google-subject",
        token="old-bare-access",
        scopes="openid",
    )
    legacy = _account(
        db,
        user,
        provider="google-drive",
        token="legacy-access",
        scopes="openid scope.drive",
    )

    shared, mirrored, association = write_google_shared_oauth_callback(
        db,
        user_id=int(user.id),
        resource_owner_key=None,
        oauth_client_id="client-a",
        app_id="google-drive",
        google_subject="google-subject",
        access_token="new-shared-access",
        refresh_token="new-shared-refresh",
        granted_scopes="openid scope.drive",
        requested_scopes=["openid", "scope.drive"],
        email="alice@example.com",
        token_type="Bearer",
        expires_at=None,
        legacy_oauth_account=legacy,
    )
    db.commit()

    assert shared.id == bare.id
    assert mirrored.id == legacy.id
    assert association.legacy_oauth_account_id == legacy.id
    assert db.query(UserOAuth).filter_by(provider="google").count() == 1
    assert shared.access_token == "new-shared-access"


def test_resolver_rejects_association_after_legacy_link_is_deleted(db_session):
    db, user = db_session
    shared = _account(db, user)
    legacy = _account(db, user, provider="google-drive")
    create_google_oauth_grant_association(
        db,
        oauth_account=shared,
        oauth_client_id="client-a",
        app_id="google-drive",
        legacy_oauth_account=legacy,
    )
    db.commit()

    db.delete(legacy)
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


def test_callback_removes_orphaned_association_after_subject_switch(db_session):
    db, user = db_session
    old_shared = _account(db, user, subject="subject-a", token="old-shared")
    old_legacy = _account(
        db,
        user,
        subject="subject-a",
        provider="google-drive",
        token="old-legacy",
    )
    create_google_oauth_grant_association(
        db,
        oauth_account=old_shared,
        oauth_client_id="client-a",
        app_id="google-drive",
        legacy_oauth_account=old_legacy,
    )
    db.commit()
    old_shared_id = int(old_shared.id)
    db.delete(old_legacy)
    db.commit()

    new_legacy = _account(
        db,
        user,
        subject="subject-b",
        provider="google-drive",
        token="new-legacy",
    )
    write_google_shared_oauth_callback(
        db,
        user_id=int(user.id),
        resource_owner_key=None,
        oauth_client_id="client-a",
        app_id="google-drive",
        google_subject="subject-b",
        access_token="new-shared",
        refresh_token="new-refresh",
        granted_scopes="openid scope.drive",
        requested_scopes=["openid", "scope.drive"],
        email="b@example.com",
        token_type="Bearer",
        expires_at=None,
        legacy_oauth_account=new_legacy,
    )
    db.commit()

    assert db.get(UserOAuth, old_shared_id) is None
    assert db.query(GoogleOAuthGrant).one().google_subject == "subject-b"
    association = db.query(GoogleOAuthAppAssociation).one()
    assert association.legacy_oauth_account_id == int(new_legacy.id)


def test_detach_keeps_shared_grant_until_last_app_is_removed(db_session):
    db, user = db_session
    shared = _account(db, user)
    drive = _account(db, user, provider="google-drive")
    sheets = _account(db, user, provider="google-sheets")
    create_google_oauth_grant_association(
        db,
        oauth_account=shared,
        oauth_client_id="client-a",
        app_id="google-drive",
        legacy_oauth_account=drive,
    )
    create_google_oauth_grant_association(
        db,
        oauth_account=shared,
        oauth_client_id="client-a",
        app_id="google-sheets",
        legacy_oauth_account=sheets,
    )
    db.commit()

    assert (
        detach_google_shared_oauth_app(
            db,
            user_id=int(user.id),
            app_id="google-drive",
        )
        == 1
    )
    db.commit()
    assert db.query(UserOAuth).filter_by(provider="google").one().id == shared.id
    assert db.query(GoogleOAuthGrant).count() == 1
    assert db.query(GoogleOAuthAppAssociation).count() == 1

    assert (
        detach_google_shared_oauth_app(
            db,
            user_id=int(user.id),
            app_id="google-sheets",
        )
        == 1
    )
    db.commit()
    assert db.query(UserOAuth).filter_by(provider="google").count() == 0
    assert db.query(GoogleOAuthGrant).count() == 0
    assert db.query(GoogleOAuthAppAssociation).count() == 0


def test_mirror_preserves_legacy_refresh_when_shared_token_is_missing(db_session):
    db, user = db_session
    db.add(
        PublicMCPApp(
            app_id="test-google-drive",
            name="Test Google Drive",
            transport="oauth",
            provider_name="google",
            oauth_scopes=["scope.drive"],
        )
    )
    db.commit()
    shared = _account(db, user)
    legacy = _account(db, user, provider="test-google-drive")
    legacy.refresh_token = "rollback-refresh"
    create_google_oauth_grant_association(
        db,
        oauth_account=shared,
        oauth_client_id="client-a",
        app_id="test-google-drive",
        legacy_oauth_account=legacy,
    )
    shared.access_token = "new-shared-access"
    shared.refresh_token = None

    assert mirror_shared_google_oauth_credentials(db, oauth_account=shared) == 1

    assert legacy.access_token == "new-shared-access"
    assert legacy.refresh_token == "rollback-refresh"


def test_mirror_uses_canonical_builtin_scope_policy(db_session):
    db, user = db_session
    db.add(
        PublicMCPApp(
            app_id="google-drive",
            name="Google Drive",
            transport="oauth",
            provider_name="google",
            # Deliberately stale. The trusted builtin registry requires a real
            # Drive scope and must override this persisted value.
            oauth_scopes=["scope.drive"],
        )
    )
    db.commit()
    shared = _account(db, user, token="new-shared", scopes="scope.drive")
    legacy = _account(
        db,
        user,
        provider="google-drive",
        token="existing-drive",
        scopes="scope.drive",
    )
    create_google_oauth_grant_association(
        db,
        oauth_account=shared,
        oauth_client_id="client-a",
        app_id="google-drive",
        legacy_oauth_account=legacy,
    )
    db.commit()

    assert mirror_shared_google_oauth_credentials(db, oauth_account=shared) == 0
    assert legacy.access_token == "existing-drive"


def test_runtime_mirror_does_not_revive_a_gmail_reconnect_tombstone(db_session):
    db, user = db_session
    gmail_app = get_builtin_public_mcp_app("gmail")
    assert gmail_app is not None
    scopes = " ".join(gmail_app["oauth_scopes"])
    shared = _account(db, user, token="live-shared", scopes=scopes)
    gmail = _account(db, user, provider="gmail", token="live-gmail", scopes=scopes)
    gmail.refresh_token = "live-refresh"
    create_google_oauth_grant_association(
        db,
        oauth_account=shared,
        oauth_client_id="client-a",
        app_id="gmail",
        legacy_oauth_account=gmail,
    )
    db.commit()
    gmail_id = int(gmail.id)
    # Keep the live object in this session's identity map while the mailbox
    # transition commits a tombstone through another transaction.
    assert gmail.access_token == "live-gmail"
    isolated_session_factory = sessionmaker(bind=db.get_bind(), autoflush=False)
    with isolated_session_factory() as transition_db:
        transitioned = transition_db.get(UserOAuth, gmail_id)
        assert transitioned is not None
        transitioned.access_token = ""
        transitioned.refresh_token = None
        transitioned.expires_at = None
        transition_db.commit()

    assert (
        mirror_shared_google_oauth_credentials(
            db,
            oauth_account=shared,
            skip_gmail_tombstones=True,
        )
        == 0
    )
    db.refresh(gmail)
    assert gmail.access_token == ""
    assert gmail.refresh_token is None


def test_callback_immediately_repairs_scope_valid_sibling_mirror(db_session):
    db, user = db_session
    db.add_all(
        [
            PublicMCPApp(
                app_id="test-google-drive",
                name="Test Google Drive",
                transport="oauth",
                provider_name="google",
                oauth_scopes=["scope.drive"],
            ),
            PublicMCPApp(
                app_id="test-google-sheets",
                name="Test Google Sheets",
                transport="oauth",
                provider_name="google",
                oauth_scopes=["scope.sheets"],
            ),
            PublicMCPApp(
                app_id="test-google-calendar",
                name="Test Google Calendar",
                transport="oauth",
                provider_name="google",
                oauth_scopes=["scope.calendar"],
            ),
        ]
    )
    db.commit()
    shared = _account(
        db,
        user,
        token="revoked-shared-access",
        scopes="openid scope.drive scope.sheets",
    )
    shared.refresh_token = "revoked-shared-refresh"
    drive = _account(
        db,
        user,
        provider="test-google-drive",
        token="revoked-drive-access",
        scopes="openid scope.drive",
    )
    drive.refresh_token = "revoked-drive-refresh"
    sheets = _account(
        db,
        user,
        provider="test-google-sheets",
        token="revoked-sheets-access",
        scopes="openid scope.sheets",
    )
    sheets.refresh_token = "revoked-sheets-refresh"
    calendar = _account(
        db,
        user,
        provider="test-google-calendar",
        token="existing-calendar-access",
        scopes="openid scope.calendar",
    )
    calendar.refresh_token = "existing-calendar-refresh"
    create_google_oauth_grant_association(
        db,
        oauth_account=shared,
        oauth_client_id="client-a",
        app_id="test-google-drive",
        legacy_oauth_account=drive,
    )
    create_google_oauth_grant_association(
        db,
        oauth_account=shared,
        oauth_client_id="client-a",
        app_id="test-google-sheets",
        legacy_oauth_account=sheets,
    )
    create_google_oauth_grant_association(
        db,
        oauth_account=shared,
        oauth_client_id="client-a",
        app_id="test-google-calendar",
        legacy_oauth_account=calendar,
    )
    db.commit()

    write_google_shared_oauth_callback(
        db,
        user_id=int(user.id),
        resource_owner_key=None,
        oauth_client_id="client-a",
        app_id="test-google-sheets",
        google_subject="google-subject",
        access_token="reconnected-access",
        refresh_token="reconnected-refresh",
        granted_scopes="openid scope.drive scope.sheets",
        requested_scopes=["openid", "scope.drive", "scope.sheets"],
        email="alice@example.com",
        token_type="Bearer",
        expires_at=None,
        legacy_oauth_account=sheets,
    )
    db.commit()

    db.refresh(drive)
    db.refresh(calendar)
    assert drive.access_token == "reconnected-access"
    assert drive.refresh_token == "reconnected-refresh"
    assert calendar.access_token == "existing-calendar-access"
    assert calendar.refresh_token == "existing-calendar-refresh"


@pytest.mark.asyncio
async def test_adopting_bare_google_row_preserves_covered_connected_sibling(
    db_session, monkeypatch
):
    db, user = db_session
    bare = _account(
        db,
        user,
        token="bare-access",
        scopes="openid scope.drive scope.sheets scope.slides",
    )
    apps = [
        PublicMCPApp(
            app_id="test-google-drive",
            name="Test Google Drive",
            transport="oauth",
            provider_name="google",
            oauth_scopes=["scope.drive"],
        ),
        PublicMCPApp(
            app_id="test-google-sheets",
            name="Test Google Sheets",
            transport="oauth",
            provider_name="google",
            oauth_scopes=["scope.sheets"],
        ),
        PublicMCPApp(
            app_id="test-google-calendar",
            name="Test Google Calendar",
            transport="oauth",
            provider_name="google",
            oauth_scopes=["scope.calendar"],
        ),
        PublicMCPApp(
            app_id="test-google-slides",
            name="Test Google Slides",
            transport="oauth",
            provider_name="google",
            oauth_scopes=["scope.slides"],
        ),
    ]
    db.add_all(apps)
    db.flush()
    for app in apps[1:]:
        server = MCPServer(
            name=app.name,
            managed="external",
            transport="oauth",
            auth={"app_id": app.app_id, "provider": "google"},
        )
        db.add(server)
        db.flush()
        db.add(
            UserMCPServer(
                user_id=int(user.id),
                mcpserver_id=int(server.id),
                is_owner=True,
                is_active=True,
            )
        )
    legacy = UserOAuth(
        user_id=int(user.id),
        provider="test-google-drive",
        provider_user_id="google-subject",
        access_token="drive-legacy-access",
        refresh_token="drive-legacy-refresh",
        scope="openid scope.drive",
    )
    db.add(legacy)
    db.commit()

    shared, _mirrored, _association = write_google_shared_oauth_callback(
        db,
        user_id=int(user.id),
        resource_owner_key=None,
        oauth_client_id="client-a",
        app_id="test-google-drive",
        google_subject="google-subject",
        access_token="adopted-access",
        refresh_token="adopted-refresh",
        granted_scopes="openid scope.drive scope.sheets",
        requested_scopes=["openid", "scope.drive", "scope.sheets"],
        email="alice@example.com",
        token_type="Bearer",
        expires_at=None,
        legacy_oauth_account=legacy,
    )
    db.commit()

    sheets_legacy = db.query(UserOAuth).filter_by(provider="test-google-sheets").one()
    slides_tombstone = (
        db.query(UserOAuth).filter_by(provider="test-google-slides").one()
    )
    assert shared.id == bare.id
    assert sheets_legacy.access_token == "adopted-access"
    assert db.query(UserOAuth).filter_by(provider="test-google-calendar").count() == 0
    assert slides_tombstone.access_token == ""
    assert slides_tombstone.refresh_token is None
    assert slides_tombstone.provider_user_id == "google-subject"
    assert {row.app_id for row in db.query(GoogleOAuthAppAssociation).all()} == {
        "test-google-drive",
        "test-google-sheets",
        "test-google-slides",
    }
    assert (
        resolve_google_shared_oauth_account(
            db,
            user_id=int(user.id),
            resource_owner_key=None,
            oauth_client_id="client-a",
            app_id="test-google-sheets",
            required_scopes=["scope.sheets"],
        )
        == shared
    )
    monkeypatch.setenv("XAGENT_GOOGLE_SHARED_OAUTH_GRANTS_ENABLED", "false")
    isolated_session_factory = sessionmaker(
        bind=db.get_bind(), autoflush=False, autocommit=False
    )
    config = WebToolConfig(
        db=db,
        request=None,
        user=user,
        user_id=int(user.id),
        workspace_config={"base_dir": "/tmp", "task_id": "task-1"},
        db_factory=isolated_session_factory,
    )
    resolved = await config._resolve_legacy_oauth_access_token(
        provider_name="google",
        app_id="test-google-sheets",
        app_info={"oauth_scopes": ["scope.sheets"]},
    )
    assert resolved.access_token == "adopted-access"


def test_detach_exact_legacy_account_preserves_then_removes_shared_grant(db_session):
    db, user = db_session
    shared = _account(db, user)
    drive = _account(db, user, provider="google-drive")
    sheets = _account(db, user, provider="google-sheets")
    create_google_oauth_grant_association(
        db,
        oauth_account=shared,
        oauth_client_id="client-a",
        app_id="google-drive",
        legacy_oauth_account=drive,
    )
    create_google_oauth_grant_association(
        db,
        oauth_account=shared,
        oauth_client_id="client-a",
        app_id="google-sheets",
        legacy_oauth_account=sheets,
    )
    db.commit()

    assert (
        detach_google_shared_oauth_legacy_account(
            db,
            user_id=int(user.id),
            legacy_oauth_account_id=int(drive.id),
        )
        == 1
    )
    db.commit()
    assert db.get(UserOAuth, int(shared.id)) is not None
    assert db.query(GoogleOAuthAppAssociation).one().app_id == "google-sheets"

    assert (
        detach_google_shared_oauth_legacy_account(
            db,
            user_id=int(user.id),
            legacy_oauth_account_id=int(sheets.id),
        )
        == 1
    )
    db.commit()
    assert db.get(UserOAuth, int(shared.id)) is None
    assert db.query(GoogleOAuthGrant).count() == 0


def test_detach_locks_shared_account_before_deleting_association(db_session):
    db, user = db_session
    shared = _account(db, user)
    legacy = _account(db, user, provider="test-google-drive")
    create_google_oauth_grant_association(
        db,
        oauth_account=shared,
        oauth_client_id="client-a",
        app_id="test-google-drive",
        legacy_oauth_account=legacy,
    )
    db.commit()
    statements: list[str] = []

    def observe_sql(_conn, _cursor, statement, _parameters, _context, _many):
        statements.append("".join(statement.lower().split()))

    event.listen(db.get_bind(), "before_cursor_execute", observe_sql)
    try:
        detach_google_shared_oauth_app(
            db, user_id=int(user.id), app_id="test-google-drive"
        )
        db.flush()
    finally:
        event.remove(db.get_bind(), "before_cursor_execute", observe_sql)

    account_lock = next(
        index
        for index, statement in enumerate(statements)
        if statement.startswith("updateuser_oauthsetid=")
    )
    association_delete = next(
        index
        for index, statement in enumerate(statements)
        if statement.startswith("deletefromgoogle_oauth_app_associations")
    )
    assert account_lock < association_delete


def test_callback_locks_shared_account_before_grant(db_session, monkeypatch):
    db, user = db_session
    shared = _account(db, user)
    legacy = _account(db, user, provider="test-google-drive")
    create_google_oauth_grant_association(
        db,
        oauth_account=shared,
        oauth_client_id="client-a",
        app_id="test-google-drive",
        legacy_oauth_account=legacy,
    )
    db.commit()
    lock_events: list[str] = []
    original_account_lock = google_oauth_grants_service._lock_google_oauth_accounts

    def observe_account_lock(lock_db, account_ids):
        lock_events.append("account")
        return original_account_lock(lock_db, account_ids)

    def observe_orm_execute(execute_state):
        statement = execute_state.statement
        if getattr(
            statement, "_for_update_arg", None
        ) is not None and "FROM google_oauth_grants" in str(statement):
            lock_events.append("grant")

    monkeypatch.setattr(
        google_oauth_grants_service,
        "_lock_google_oauth_accounts",
        observe_account_lock,
    )
    event.listen(db, "do_orm_execute", observe_orm_execute)
    try:
        write_google_shared_oauth_callback(
            db,
            user_id=int(user.id),
            resource_owner_key=None,
            oauth_client_id="client-a",
            app_id="test-google-drive",
            google_subject="google-subject",
            access_token="reconnected-access",
            refresh_token="reconnected-refresh",
            granted_scopes="openid scope.drive",
            requested_scopes=["openid", "scope.drive"],
            email="alice@example.com",
            token_type="Bearer",
            expires_at=None,
            legacy_oauth_account=legacy,
        )
        db.flush()
    finally:
        event.remove(db, "do_orm_execute", observe_orm_execute)

    def account_lock_precedes_grant(events: list[str]) -> bool:
        return events.index("account") < events.index("grant")

    assert account_lock_precedes_grant(lock_events)
    assert not account_lock_precedes_grant(["grant", "account"])


def test_postgresql_shared_lock_queries_compile_for_update_of(db_session, monkeypatch):
    db, user = db_session
    shared = _account(db, user)
    legacy = _account(db, user, provider="test-google-drive")
    create_google_oauth_grant_association(
        db,
        oauth_account=shared,
        oauth_client_id="client-a",
        app_id="test-google-drive",
        legacy_oauth_account=legacy,
    )
    db.commit()

    lock_statements = []
    original_with_for_update = Query.with_for_update

    def capture_for_update(query, *args, **kwargs):
        locked_query = original_with_for_update(query, *args, **kwargs)
        lock_statements.append(locked_query.statement)
        return locked_query

    monkeypatch.setattr(Query, "with_for_update", capture_for_update)
    monkeypatch.setattr(db.get_bind().dialect, "name", "postgresql")

    resolved = resolve_google_shared_oauth_account(
        db,
        user_id=int(user.id),
        resource_owner_key=None,
        oauth_client_id="client-a",
        app_id="test-google-drive",
        required_scopes=["scope.drive"],
        lock=True,
    )
    assert resolved is not None

    legacy.provider_user_id = "different-google-subject"
    db.flush()
    google_oauth_grants_service._remove_stale_legacy_associations(
        db,
        user_id=int(user.id),
        resource_owner_key=None,
        app_id="test-google-drive",
    )

    compiled_locks = [
        " ".join(
            str(
                statement.compile(
                    dialect=postgresql.dialect(),
                    compile_kwargs={"literal_binds": True},
                )
            ).split()
        ).lower()
        for statement in lock_statements
    ]
    assert any("for update of user_oauth" in sql for sql in compiled_locks)
    assert any(
        "for update of google_oauth_app_associations" in sql for sql in compiled_locks
    )
