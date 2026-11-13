"""Revoking a Google OAuth grant when a Google-backed connection is
disconnected.

Google grants are stored under the connecting app's id (``gmail``,
``google-drive``, ...) or the bare ``google`` key, and every one of them was
issued to the same Google OAuth client -- so the revoke resolves the
provider through the catalog, revokes through the refresh token, and skips
while any sibling row for the same Google account is still connected.
"""

from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from xagent.web.api import auth as auth_api
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_oauth import COMPOSIO_TOKEN_TYPE, UserOAuth

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


def _assert_revoked_with(post: Mock, token: str) -> None:
    post.assert_called_once()
    args, kwargs = post.call_args
    assert args[0] == GOOGLE_REVOKE_URL
    # In the form body, never the URL.
    assert kwargs["data"] == {"token": token}


def test_resolve_maps_an_app_key_to_a_google_revocation(db):
    resolved = auth_api.resolve_builtin_oauth_revocation(
        db,
        provider="gmail",
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
            provider="gmail",
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
    auth_api.revoke_builtin_oauth_grant(
        db,
        provider="google-drive",
        access_token="access",
        provider_user_id="sub-1",
        refresh_token="refresh",
    )

    _assert_revoked_with(post, "refresh")


def test_revoke_falls_back_to_the_access_token(db, post):
    auth_api.revoke_builtin_oauth_grant(
        db,
        provider="google",
        access_token="access",
        provider_user_id="sub-1",
    )

    _assert_revoked_with(post, "access")


def test_revoke_skips_while_a_sibling_google_app_is_connected(db, post):
    """Google revokes the whole grant for the client, so disconnecting Gmail
    must not cut off the same account's still-connected Drive."""
    user = _user(db)
    db.add(
        UserOAuth(
            user_id=int(user.id),
            provider="google-drive",
            provider_user_id="sub-1",
            access_token="drive-access",
        )
    )
    db.commit()

    auth_api.revoke_builtin_oauth_grant(
        db,
        provider="gmail",
        access_token="access",
        provider_user_id="sub-1",
        refresh_token="refresh",
    )

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

    auth_api.revoke_builtin_oauth_grant(
        db,
        provider="gmail",
        access_token="access",
        provider_user_id="sub-1",
        refresh_token="refresh",
    )

    _assert_revoked_with(post, "refresh")


def test_a_google_row_without_an_upstream_id_is_never_revoked(db, post):
    """Disconnecting a legacy row that recorded no Google account id can't
    prove no other connection shares its grant, so it must not revoke."""
    user = _user(db)
    db.add(
        UserOAuth(
            user_id=int(user.id),
            provider="google-drive",
            provider_user_id="sub-1",
            access_token="drive-access",
        )
    )
    db.commit()

    auth_api.revoke_builtin_oauth_grant(
        db,
        provider="gmail",
        access_token="access",
        provider_user_id=None,
        refresh_token="refresh",
    )

    post.assert_not_called()


def test_a_same_user_google_row_without_an_upstream_id_blocks_the_revoke(db, post):
    """The legacy Gmail row may hold the very grant that disconnecting Drive
    would revoke, so it counts as a live reference."""
    user = _user(db)
    db.add(
        UserOAuth(
            user_id=int(user.id),
            provider="gmail",
            provider_user_id=None,
            access_token="legacy-gmail-access",
        )
    )
    db.commit()

    auth_api.revoke_builtin_oauth_grant(
        db,
        provider="google-drive",
        access_token="access",
        provider_user_id="sub-1",
        refresh_token="refresh",
    )

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
            provider_user_id=None,
            access_token="bobs-legacy-access",
        )
    )
    db.commit()

    auth_api.revoke_builtin_oauth_grant(
        db,
        provider="google-drive",
        access_token="access",
        provider_user_id="sub-1",
        refresh_token="refresh",
    )

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

    auth_api.revoke_builtin_oauth_grant(
        db,
        provider="google-drive",
        access_token="access",
        provider_user_id="sub-1",
        refresh_token="refresh",
    )

    _assert_revoked_with(post, "refresh")


def test_a_rejected_access_token_is_logged_as_not_revoked(monkeypatch, caplog):
    """Google answers an expired access token with invalid_token without
    revoking the grant behind it: that must not read as success."""
    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(return_value=MockResponse(400, '{"error": "invalid_token"}')),
    )

    with caplog.at_level(logging.INFO, logger=auth_api.logger.name):
        auth_api._revoke_google_oauth_token("access", kind="access_token")

    assert "not revoked" in caplog.text


def test_an_unknown_provider_record_is_skipped_not_sent_to_github(monkeypatch, caplog):
    delete = Mock()
    post = Mock()
    monkeypatch.setattr(auth_api.requests, "delete", delete)
    monkeypatch.setattr(auth_api.requests, "post", post)

    with caplog.at_level(logging.WARNING, logger=auth_api.logger.name):
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
    assert "no revoke implementation" in caplog.text


@pytest.mark.parametrize(
    ("response", "warned"),
    [
        (MockResponse(200), False),
        # Already revoked or expired: also a successful outcome.
        (MockResponse(400, '{"error": "invalid_token"}'), False),
        (MockResponse(400, '{"error": "unsupported_token_type"}'), True),
        (MockResponse(503), True),
    ],
)
def test_revoke_google_token_logs_only_genuine_failures(
    monkeypatch, caplog, response, warned
):
    monkeypatch.setattr(auth_api.requests, "post", Mock(return_value=response))

    with caplog.at_level(logging.WARNING, logger=auth_api.logger.name):
        auth_api._revoke_google_oauth_token("tok", kind="refresh_token")

    assert bool(caplog.records) is warned


def test_revoke_google_token_swallows_a_failure_without_leaking_it(monkeypatch, caplog):
    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(side_effect=RuntimeError("boom secret-refresh-token")),
    )

    with caplog.at_level(logging.WARNING, logger=auth_api.logger.name):
        # Must not raise.
        auth_api._revoke_google_oauth_token(
            "secret-refresh-token", kind="refresh_token"
        )

    assert "RuntimeError" in caplog.text
    assert "secret-refresh-token" not in caplog.text


@pytest.mark.asyncio
async def test_delete_mcp_server_revokes_the_google_grant(db, post):
    """The disconnect endpoints snapshot the refresh token before the bulk
    delete and revoke it after the commit."""
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
            provider_user_id="sub-1",
            access_token="access",
            refresh_token="refresh",
        )
    )
    db.commit()

    await delete_mcp_server(server_id=int(server.id), current_user=user, db=db)

    assert db.query(UserOAuth).filter(UserOAuth.user_id == user.id).count() == 0
    _assert_revoked_with(post, "refresh")
