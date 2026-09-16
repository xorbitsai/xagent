"""Revoking a builtin OAuth grant when a GitHub-backed MCP server is
disconnected through the generic connector endpoints.

Disconnecting only deletes the local ``UserOAuth`` row; GitHub's own grant
stays alive unless something also calls its revoke API, which is exactly
what lets a reconnect silently skip GitHub's consent screen (see
auth.py's ``resolve_builtin_oauth_revocation`` /
``revoke_resolved_github_oauth_grant``). This pins that *both* generic
disconnect endpoints call it -- ``delete_mcp_server`` and the app-scoped
``teardown_mcp_app_server`` -- not just Toby's own personal-connector
disconnect path (covered separately in ``xagent-saas``'s own test suite).
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from xagent.core.utils.encryption import encrypt_value
from xagent.web.api import auth as auth_api
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.oauth_provider import OAuthProvider
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth

pytestmark = pytest.mark.asyncio


class MockResponse:
    def __init__(self, status_code: int = 204):
        self.status_code = status_code


def _seed_github_catalog_and_provider(db: Session) -> None:
    db.add(
        PublicMCPApp(
            app_id="github",
            name="GitHub",
            transport="oauth",
            provider_name="github",
            category="Development",
            oauth_scopes=["repo", "user:email"],
            is_visible_in_connector=True,
            launch_config={},
        )
    )
    db.add(
        OAuthProvider(
            provider_name="github",
            name="GitHub",
            client_id=encrypt_value("github-client-id"),
            client_secret=encrypt_value("github-client-secret"),
            auth_url="https://github.com/login/oauth/authorize",
            token_url="https://github.com/login/oauth/access_token",
            redirect_uri="https://app.example.com/api/auth/github/callback",
            userinfo_url="https://api.github.com/user",
            user_id_path="id",
            email_path="login",
            default_scopes=["read:user"],
        )
    )


def _connected_github_server(db: Session) -> tuple[User, MCPServer, UserMCPServer]:
    """One user with a connected GitHub MCP server and a live access token,
    stamped with its own ``auth.app_id`` (the current, non-legacy
    provisioning convention -- see mcp_apps.get_app_for_mcp_server)."""
    user = User(username="octocat", password_hash="h", is_admin=False)
    db.add(user)
    _seed_github_catalog_and_provider(db)
    db.commit()
    db.refresh(user)

    server = MCPServer(
        name="GitHub",
        transport="oauth",
        managed=False,
        auth={"app_id": "github"},
    )
    db.add(server)
    db.flush()
    user_mcp = UserMCPServer(
        user_id=int(user.id),
        mcpserver_id=server.id,
        is_active=True,
        is_owner=True,
    )
    db.add(user_mcp)
    db.add(
        UserOAuth(
            user_id=int(user.id),
            provider="github",
            access_token="live-access-token",
            provider_user_id="42",
        )
    )
    db.commit()
    db.refresh(server)
    db.refresh(user_mcp)
    return user, server, user_mcp


def _assert_github_grant_was_revoked(delete: Mock) -> None:
    delete.assert_called_once()
    args, kwargs = delete.call_args
    assert args[0] == "https://api.github.com/applications/github-client-id/grant"
    assert kwargs["auth"] == ("github-client-id", "github-client-secret")
    assert kwargs["json"] == {"access_token": "live-access-token"}


@pytest.fixture()
def db():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


async def test_delete_mcp_server_revokes_the_github_grant(db, monkeypatch):
    from xagent.web.api.mcp import delete_mcp_server

    user, server, _user_mcp = _connected_github_server(db)
    delete = Mock(return_value=MockResponse())
    monkeypatch.setattr(auth_api.requests, "delete", delete)

    await delete_mcp_server(server_id=int(server.id), current_user=user, db=db)

    assert db.query(UserOAuth).filter(UserOAuth.user_id == user.id).count() == 0
    _assert_github_grant_was_revoked(delete)


async def test_delete_mcp_server_skips_revocation_with_a_live_sibling(db, monkeypatch):
    """The end-to-end regression for the sibling-reference finding: a
    second XAgent user connected to the same upstream GitHub account keeps
    working after the first user disconnects. Exercised through the real
    disconnect endpoint and the real _snapshot_builtin_oauth_revocations ->
    has_other_builtin_oauth_reference path, not a hand-built
    BuiltinOAuthRevocation -- a wiring bug that drops or miswires
    provider_user_id anywhere along that path would go undetected by a
    test built the other way."""
    from xagent.web.api.mcp import delete_mcp_server

    user, server, _user_mcp = _connected_github_server(db)
    other_user = User(username="bob", password_hash="h", is_admin=False)
    db.add(other_user)
    db.flush()
    db.add(
        UserOAuth(
            user_id=other_user.id,
            provider="github",
            provider_user_id="42",
            access_token="bobs-still-live-token",
        )
    )
    db.commit()

    delete = Mock()
    monkeypatch.setattr(auth_api.requests, "delete", delete)

    await delete_mcp_server(server_id=int(server.id), current_user=user, db=db)

    # The disconnecting user's own row is gone -- disconnect still worked --
    # but GitHub was never called, so bob's still-live token survives.
    assert db.query(UserOAuth).filter(UserOAuth.user_id == user.id).count() == 0
    assert db.query(UserOAuth).filter(UserOAuth.user_id == other_user.id).count() == 1
    delete.assert_not_called()


async def test_delete_mcp_server_disconnect_survives_a_revoke_failure(db, monkeypatch):
    """A dead network to GitHub must not turn an otherwise-successful
    disconnect into a failed request."""
    from xagent.web.api.mcp import delete_mcp_server

    user, server, _user_mcp = _connected_github_server(db)
    monkeypatch.setattr(
        auth_api.requests,
        "delete",
        Mock(side_effect=auth_api.requests.ConnectionError("network down")),
    )

    await delete_mcp_server(server_id=int(server.id), current_user=user, db=db)

    assert db.query(UserOAuth).filter(UserOAuth.user_id == user.id).count() == 0


@pytest.fixture()
def teardown_db(tmp_path):
    # A real file, not ``:memory:``: _teardown_mcp_app_server_locally runs
    # under ``asyncio.to_thread`` (a genuine OS thread), and this is the same
    # setup the existing off-event-loop seam test uses for that reason.
    db_path = tmp_path / "builtin-oauth-teardown.db"
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = SessionLocal()
    yield db
    db.close()
    engine.dispose()


async def test_teardown_mcp_app_server_revokes_the_github_grant(
    teardown_db, monkeypatch
):
    from xagent.web.api.mcp import teardown_mcp_app_server

    db = teardown_db
    user, server, user_mcp = _connected_github_server(db)
    app = db.query(PublicMCPApp).filter(PublicMCPApp.app_id == "github").one()
    delete = Mock(return_value=MockResponse())
    monkeypatch.setattr(auth_api.requests, "delete", delete)

    await teardown_mcp_app_server(
        int(server.id),
        app_id="github",
        expected_provider_name="github",
        expected_catalog_generation=app.generation,
        expected_association_generation=user_mcp.lifecycle_generation,
        current_user=user,
        db=db,
    )

    assert db.query(UserOAuth).filter(UserOAuth.user_id == user.id).count() == 0
    _assert_github_grant_was_revoked(delete)


async def test_teardown_mcp_app_server_skips_revocation_with_a_live_sibling(
    teardown_db, monkeypatch
):
    from xagent.web.api.mcp import teardown_mcp_app_server

    db = teardown_db
    user, server, user_mcp = _connected_github_server(db)
    other_user = User(username="bob", password_hash="h", is_admin=False)
    db.add(other_user)
    db.flush()
    db.add(
        UserOAuth(
            user_id=other_user.id,
            provider="github",
            provider_user_id="42",
            access_token="bobs-still-live-token",
        )
    )
    db.commit()
    app = db.query(PublicMCPApp).filter(PublicMCPApp.app_id == "github").one()

    delete = Mock()
    monkeypatch.setattr(auth_api.requests, "delete", delete)

    await teardown_mcp_app_server(
        int(server.id),
        app_id="github",
        expected_provider_name="github",
        expected_catalog_generation=app.generation,
        expected_association_generation=user_mcp.lifecycle_generation,
        current_user=user,
        db=db,
    )

    assert db.query(UserOAuth).filter(UserOAuth.user_id == user.id).count() == 0
    assert db.query(UserOAuth).filter(UserOAuth.user_id == other_user.id).count() == 1
    delete.assert_not_called()
