"""Revoking a Google OAuth grant when a Google-backed connection is
disconnected.

Google grants are stored under the connecting app's id (``gmail``,
``google-drive``, ...) or the bare ``google`` key, and every one of them was
issued to the same Google OAuth client -- so the revoke resolves the
provider from the row's recorded issuer (``UserOAuth.oauth_provider``),
revokes through the refresh token, and skips while any sibling row for the
same Google account is still connected. Deleting a row enqueues its
revocation in the credential revocation outbox; these tests drain the outbox
to see the provider call.
"""

from __future__ import annotations

import asyncio
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from tests.shared.credential_revocations import (
    credential_revocation_statuses,
    drain_credential_revocations,
)
from xagent.web.api import auth as auth_api
from xagent.web.models.credential_revocation import CredentialRevocation
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.oauth_revocation_fence import OAuthRevocationFence
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_oauth import COMPOSIO_TOKEN_TYPE, UserOAuth
from xagent.web.services.credential_revocations import ProviderRevocationError
from xagent.web.services.google_oauth_grants import (
    create_google_oauth_grant_association,
)

GOOGLE_REVOKE_URL = "https://oauth2.googleapis.com/revoke"


class MockResponse:
    def __init__(self, status_code: int = 200, text: str = ""):
        self.status_code = status_code
        self.text = text


@pytest.fixture()
def db():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    with Session(engine) as session:
        for app_id in ("gmail", "google-drive"):
            session.add(
                PublicMCPApp(
                    app_id=app_id,
                    name=app_id,
                    transport="oauth",
                    provider_name="google",
                    launch_config={},
                )
            )
        session.commit()
        yield session
    engine.dispose()


@pytest.fixture()
def post(monkeypatch) -> Mock:
    mock = Mock(return_value=MockResponse(200))
    monkeypatch.setattr(auth_api.requests, "post", mock)
    return mock


def _user(db: Session, username: str = "alice") -> User:
    user = User(username=username, password_hash="h", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _revoke_deleted(
    db: Session,
    *,
    provider_user_id: str | None = "sub-1",
    refresh_token: str | None = None,
) -> CredentialRevocation:
    """Delete one Gmail row of a user of its own, drain the outbox, and
    return the row's record."""
    user = _user(db, "deleting-user")
    row = UserOAuth(
        user_id=int(user.id),
        provider="gmail",
        oauth_provider="google",
        provider_user_id=provider_user_id,
        access_token="access",
        refresh_token=refresh_token,
    )
    db.add(row)
    db.commit()
    db.delete(row)
    asyncio.run(drain_credential_revocations(db))
    db.expire_all()
    return db.query(CredentialRevocation).one()


def _assert_revoked_with(post: Mock, token: str) -> None:
    post.assert_called_once()
    args, kwargs = post.call_args
    assert args[0] == GOOGLE_REVOKE_URL
    # In the form body, never the URL.
    assert kwargs["data"] == {"token": token}


def test_resolve_maps_a_google_issuer_to_a_google_revocation(db):
    resolved = auth_api.resolve_builtin_oauth_revocation(
        db,
        oauth_provider="google",
        access_token="access",
        provider_user_id="sub-1",
        refresh_token="refresh",
    )

    assert resolved == auth_api.BuiltinOAuthRevocation(
        provider="google",
        access_token="access",
        client_id="",
        client_secret="",
        provider_user_id="sub-1",
        refresh_token="refresh",
    )


def test_resolve_skips_a_composio_backed_row(db):
    """Its access_token is a Composio connected-account id, and its grant
    belongs to Composio's OAuth app; Composio account deletion handles it."""
    assert (
        auth_api.resolve_builtin_oauth_revocation(
            db,
            oauth_provider=None,
            access_token="ca_composio-account",
            provider_user_id="sub-1",
            token_type=COMPOSIO_TOKEN_TYPE,
        )
        is None
    )


def test_revocation_repr_redacts_the_refresh_token():
    revocation = auth_api.BuiltinOAuthRevocation(
        provider="google",
        access_token="the-access-token",
        client_id="",
        client_secret="",
        provider_user_id="sub-1",
        refresh_token="the-refresh-token",
    )

    assert "the-refresh-token" not in repr(revocation)
    assert "the-access-token" not in repr(revocation)


def test_revoke_prefers_the_refresh_token(db, post):
    """An access token may have expired, and Google answers an expired one
    with invalid_token without revoking the grant behind it."""
    _revoke_deleted(db, provider_user_id="sub-1", refresh_token="refresh")

    _assert_revoked_with(post, "refresh")


def test_revoke_falls_back_to_the_access_token(db, post):
    _revoke_deleted(db, provider_user_id="sub-1", refresh_token=None)

    _assert_revoked_with(post, "access")


def test_revoke_skips_while_a_sibling_google_app_is_connected(db, post):
    """Google revokes the whole grant for the client, so disconnecting Gmail
    must not cut off the same account's still-connected Drive."""
    user = _user(db)
    db.add(
        UserOAuth(
            user_id=int(user.id),
            provider="google-drive",
            oauth_provider="google",
            provider_user_id="sub-1",
            access_token="drive-access",
        )
    )
    db.commit()

    _revoke_deleted(db, provider_user_id="sub-1", refresh_token="refresh")

    post.assert_not_called()


def test_a_composio_sibling_does_not_block_the_revoke(db, post):
    user = _user(db)
    db.add(
        UserOAuth(
            user_id=int(user.id),
            provider="google-drive",
            provider_user_id="sub-1",
            access_token="ca_composio-account",
            token_type=COMPOSIO_TOKEN_TYPE,
        )
    )
    db.commit()

    _revoke_deleted(db, provider_user_id="sub-1", refresh_token="refresh")

    _assert_revoked_with(post, "refresh")


def test_a_google_row_without_an_upstream_id_is_never_revoked(db, post):
    """Disconnecting a legacy row that recorded no Google account id can't
    prove no other connection shares its grant, so it must not revoke."""
    user = _user(db)
    db.add(
        UserOAuth(
            user_id=int(user.id),
            provider="google-drive",
            oauth_provider="google",
            provider_user_id="sub-1",
            access_token="drive-access",
        )
    )
    db.commit()

    _revoke_deleted(db, provider_user_id=None, refresh_token="refresh")

    post.assert_not_called()


def test_a_same_user_google_row_without_an_upstream_id_blocks_the_revoke(db, post):
    """The legacy Gmail row may hold the very grant that disconnecting Drive
    would revoke, so it counts as a live reference."""
    user = _user(db)
    db.add(
        UserOAuth(
            user_id=int(user.id),
            provider="gmail",
            oauth_provider="google",
            provider_user_id=None,
            access_token="legacy-gmail-access",
        )
    )
    db.commit()

    _revoke_deleted(db, provider_user_id="sub-1", refresh_token="refresh")

    post.assert_not_called()


def test_another_users_google_row_without_an_upstream_id_blocks_the_revoke(db, post):
    """Bob's legacy Gmail row may be a connection to the same Google account
    Alice is disconnecting Drive from; nothing proves it isn't, so Alice's
    disconnect must not revoke the grant Bob may still use."""
    _user(db)
    other = _user(db, "bob")
    db.add(
        UserOAuth(
            user_id=int(other.id),
            provider="gmail",
            oauth_provider="google",
            provider_user_id=None,
            access_token="bobs-legacy-access",
        )
    )
    db.commit()

    _revoke_deleted(db, provider_user_id="sub-1", refresh_token="refresh")

    post.assert_not_called()


def test_a_tokenless_or_composio_google_row_without_an_upstream_id_does_not_block(
    db, post
):
    """Only a row that could still use our grant counts: a disconnected
    tombstone holds no token, and a Composio row's grant is Composio's."""
    user = _user(db)
    db.add_all(
        [
            UserOAuth(
                user_id=int(user.id),
                provider="gmail",
                oauth_provider="google",
                provider_user_id=None,
                access_token="",
            ),
            UserOAuth(
                user_id=int(user.id),
                provider="google",
                provider_user_id=None,
                access_token="ca_composio-account",
                token_type=COMPOSIO_TOKEN_TYPE,
            ),
        ]
    )
    db.commit()

    _revoke_deleted(db, provider_user_id="sub-1", refresh_token="refresh")

    _assert_revoked_with(post, "refresh")


def test_the_skip_names_an_unidentified_legacy_row(db, post):
    """Once such a row exists every Google revoke is skipped, so the record
    must tell it apart from an ordinary sibling reference."""
    user = _user(db)
    db.add(
        UserOAuth(
            user_id=int(user.id),
            provider="gmail",
            oauth_provider="google",
            provider_user_id=None,
            access_token="legacy-gmail-access",
        )
    )
    db.commit()

    record = _revoke_deleted(db, refresh_token="refresh")

    post.assert_not_called()
    assert record.status == "skipped"
    assert "no recorded account id" in record.last_error
    assert "another local connection" not in record.last_error


def test_the_skip_names_a_sibling_row_recording_the_same_account(db, post):
    user = _user(db)
    db.add(
        UserOAuth(
            user_id=int(user.id),
            provider="google-drive",
            oauth_provider="google",
            provider_user_id="sub-1",
            access_token="drive-access",
        )
    )
    db.commit()

    record = _revoke_deleted(db, refresh_token="refresh")

    post.assert_not_called()
    assert record.status == "skipped"
    assert "another local connection" in record.last_error
    assert "no recorded account id" not in record.last_error


def test_a_rejected_access_token_is_reported_as_not_revoked(monkeypatch):
    """Google answers an expired access token with invalid_token without
    revoking the grant behind it: that must not read as a revocation, and
    retrying would not help either."""
    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(return_value=MockResponse(400, '{"error": "invalid_token"}')),
    )

    detail = auth_api._revoke_google_oauth_token("access", kind="access_token")

    assert "cannot be revoked" in detail


def test_an_unknown_provider_record_is_refused_not_sent_to_github(monkeypatch):
    delete = Mock()
    post = Mock()
    monkeypatch.setattr(auth_api.requests, "delete", delete)
    monkeypatch.setattr(auth_api.requests, "post", post)

    with pytest.raises(ValueError, match="no revoke implementation"):
        auth_api.revoke_resolved_builtin_oauth_grant(
            auth_api.BuiltinOAuthRevocation(
                provider="slack",
                access_token="tok",
                client_id="cid",
                client_secret="secret",
                provider_user_id="42",
            )
        )

    delete.assert_not_called()
    post.assert_not_called()


@pytest.mark.parametrize(
    ("response", "settled"),
    [
        (MockResponse(200), True),
        # Already revoked or expired: also a settled outcome.
        (MockResponse(400, '{"error": "invalid_token"}'), True),
        (MockResponse(400, '{"error": "unsupported_token_type"}'), False),
        (MockResponse(503), False),
    ],
)
def test_revoke_google_token_raises_only_on_genuine_failures(
    monkeypatch, response, settled
):
    monkeypatch.setattr(auth_api.requests, "post", Mock(return_value=response))

    if settled:
        auth_api._revoke_google_oauth_token("tok", kind="refresh_token")
    else:
        with pytest.raises(ProviderRevocationError, match=str(response.status_code)):
            auth_api._revoke_google_oauth_token("tok", kind="refresh_token")


def test_revoke_google_token_failure_does_not_leak_the_token(monkeypatch):
    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(side_effect=RuntimeError("boom secret-refresh-token")),
    )

    with pytest.raises(ProviderRevocationError) as raised:
        auth_api._revoke_google_oauth_token(
            "secret-refresh-token", kind="refresh_token"
        )

    assert "RuntimeError" in str(raised.value)
    assert "secret-refresh-token" not in str(raised.value)
    assert raised.value.__cause__ is None


@pytest.mark.asyncio
async def test_delete_mcp_server_revokes_the_google_grant(db, post):
    """The disconnect's bulk delete enqueues the row as stored, and the
    outbox revokes through its refresh token."""
    from xagent.web.api.mcp import delete_mcp_server

    user = _user(db)
    server = MCPServer(
        name="Gmail", transport="oauth", managed=False, auth={"app_id": "gmail"}
    )
    db.add(server)
    db.flush()
    db.add(
        UserMCPServer(
            user_id=int(user.id), mcpserver_id=server.id, is_active=True, is_owner=True
        )
    )
    db.add(
        UserOAuth(
            user_id=int(user.id),
            provider="gmail",
            oauth_provider="google",
            provider_user_id="sub-1",
            access_token="access",
            refresh_token="refresh",
        )
    )
    db.commit()

    await delete_mcp_server(server_id=int(server.id), current_user=user, db=db)
    post.assert_not_called()
    await drain_credential_revocations(db)

    assert db.query(UserOAuth).filter(UserOAuth.user_id == user.id).count() == 0
    _assert_revoked_with(post, "refresh")


def _shared_grant_with_tombstoned_last_link(db: Session, user: User) -> UserOAuth:
    """A shared ``google`` credential whose only app link points at a legacy
    Drive row that holds no token any more (a disconnected tombstone).
    Detaching that link deletes the shared credential as an orphan. The
    tombstone's dropped copy of the shared tokens is recorded too, and it is
    the same Google grant."""
    shared = UserOAuth(
        user_id=int(user.id),
        provider="google",
        oauth_provider="google",
        provider_user_id="sub-1",
        access_token="shared-access",
        refresh_token="shared-refresh",
        scope="openid scope.drive",
    )
    legacy = UserOAuth(
        user_id=int(user.id),
        provider="google-drive",
        oauth_provider="google",
        provider_user_id="sub-1",
        access_token="shared-access",
        refresh_token="shared-refresh",
        scope="openid scope.drive",
    )
    db.add_all([shared, legacy])
    db.flush()
    create_google_oauth_grant_association(
        db,
        oauth_account=shared,
        oauth_client_id="google-client-id",
        app_id="google-drive",
        legacy_oauth_account=legacy,
    )
    legacy.access_token = ""
    legacy.refresh_token = None
    db.commit()
    return legacy


@pytest.mark.asyncio
async def test_mcp_disconnect_revokes_the_shared_grant_its_last_link_deletes(db, post):
    from xagent.web.api.mcp import delete_mcp_server

    user = _user(db)
    _shared_grant_with_tombstoned_last_link(db, user)
    server = MCPServer(
        name="google-drive",
        transport="oauth",
        managed=False,
        auth={"app_id": "google-drive"},
    )
    db.add(server)
    db.flush()
    db.add(
        UserMCPServer(
            user_id=int(user.id), mcpserver_id=server.id, is_active=True, is_owner=True
        )
    )
    db.commit()

    await delete_mcp_server(server_id=int(server.id), current_user=user, db=db)
    post.assert_not_called()
    await drain_credential_revocations(db)

    assert db.query(UserOAuth).filter(UserOAuth.user_id == user.id).count() == 0
    _assert_revoked_with(post, "shared-refresh")


@pytest.mark.asyncio
async def test_cloud_account_delete_revokes_the_shared_grant_its_last_link_deletes(
    db, post
):
    from xagent.web.api.cloud_storage import delete_connected_account

    user = _user(db)
    legacy = _shared_grant_with_tombstoned_last_link(db, user)

    await delete_connected_account(account_id=int(legacy.id), db=db, user=user)
    await drain_credential_revocations(db)

    assert db.query(UserOAuth).filter(UserOAuth.user_id == user.id).count() == 0
    _assert_revoked_with(post, "shared-refresh")


@pytest.mark.asyncio
async def test_a_shared_grant_a_sibling_app_still_links_is_not_revoked(db, post):
    """The shared credential survives the detach, so it is still there for
    the reference check of the deleted legacy row's record."""
    from xagent.web.api.cloud_storage import delete_connected_account

    user = _user(db)
    legacy = _shared_grant_with_tombstoned_last_link(db, user)
    shared = db.query(UserOAuth).filter(UserOAuth.provider == "google").one()
    calendar = UserOAuth(
        user_id=int(user.id),
        provider="google-calendar",
        oauth_provider="google",
        provider_user_id="sub-1",
        access_token="shared-access",
        refresh_token="shared-refresh",
    )
    db.add(calendar)
    db.flush()
    create_google_oauth_grant_association(
        db,
        oauth_account=shared,
        oauth_client_id="google-client-id",
        app_id="google-calendar",
        legacy_oauth_account=calendar,
    )
    db.commit()

    await delete_connected_account(account_id=int(legacy.id), db=db, user=user)
    await drain_credential_revocations(db)

    assert db.query(UserOAuth).filter(UserOAuth.provider == "google").count() == 1
    post.assert_not_called()


@pytest.mark.asyncio
async def test_deleting_a_non_google_account_leaves_the_google_fences_alone(db, post):
    """A Dropbox disconnect cannot detach a shared Google link, and Dropbox
    has no provider-side revocation, so its record is skipped without a
    fence insert, lock or reference query."""
    from xagent.web.api.cloud_storage import delete_connected_account

    user = _user(db)
    _shared_grant_with_tombstoned_last_link(db, user)
    # Tombstoning the legacy row dropped its copy of the shared tokens; that
    # record is skipped while the shared credential is live. Settle it first
    # so only the Dropbox disconnect is measured.
    await drain_credential_revocations(db)
    assert credential_revocation_statuses(db) == ["skipped"]
    fences = db.query(OAuthRevocationFence).count()
    dropbox = UserOAuth(
        user_id=int(user.id),
        provider="dropbox",
        oauth_provider="dropbox",
        provider_user_id="dropbox-1",
        access_token="dropbox-access",
    )
    db.add(dropbox)
    db.commit()

    await delete_connected_account(account_id=int(dropbox.id), db=db, user=user)
    await drain_credential_revocations(db)

    assert credential_revocation_statuses(db) == ["skipped", "skipped"]
    assert db.query(UserOAuth).filter(UserOAuth.provider == "dropbox").count() == 0
    assert db.query(UserOAuth).filter(UserOAuth.provider == "google").count() == 1
    assert db.query(OAuthRevocationFence).count() == fences
    post.assert_not_called()
