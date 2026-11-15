"""The issuing OAuth provider recorded on each native credential.

``UserOAuth.provider`` is the connecting app's key, which for a custom app
says nothing about who issued the tokens. The callback records the issuer in
``UserOAuth.oauth_provider``, and provider-side revocation reads only that:
it keeps targeting the real issuer after an admin deletes, retargets or
re-creates the app, and refuses to guess when the issuer is unknown.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from tests.shared.credential_revocations import drain_credential_revocations
from tests.shared.oauth_browser_fixture import (
    create_access_token,
    generic_oauth_callback,
)
from xagent.core.utils.encryption import decrypt_value_strict, encrypt_value
from xagent.web.api import auth as auth_api
from xagent.web.models.credential_revocation import CredentialRevocation
from xagent.web.models.database import Base
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_oauth import COMPOSIO_TOKEN_TYPE, UserOAuth

CUSTOM_APP = "acme-mail"


class MockResponse:
    def __init__(self, json_data: dict | None = None, status_code: int = 200):
        self._json_data = json_data or {}
        self.status_code = status_code
        self.text = ""

    def json(self) -> dict:
        return self._json_data


@pytest.fixture()
def db():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


@pytest.fixture()
def user(db: Session) -> User:
    user = User(username="alice", password_hash="h", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _custom_app(db: Session, provider_name: str) -> PublicMCPApp:
    app = PublicMCPApp(
        app_id=CUSTOM_APP,
        name="Acme Mail",
        transport="oauth",
        provider_name=provider_name,
        launch_config={},
    )
    db.add(app)
    db.commit()
    return app


def _callback(
    db: Session, user: User, monkeypatch, *, provider: str, app_id: str, userinfo
):
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": provider,
            "app_id": app_id,
        },
        expires_delta=timedelta(minutes=10),
    )
    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {"access_token": f"{provider}-token", "token_type": "Bearer"}
            )
        ),
    )
    monkeypatch.setattr(
        auth_api.requests, "get", Mock(return_value=MockResponse(userinfo))
    )
    db_provider = SimpleNamespace(
        provider_name=provider,
        client_id=encrypt_value(f"{provider}-client-id"),
        client_secret=encrypt_value(f"{provider}-client-secret"),
        token_url=f"https://{provider}.example.com/token",
        redirect_uri=f"https://app.example.com/api/auth/{provider}/callback",
        userinfo_url=f"https://{provider}.example.com/userinfo",
        user_id_path="sub" if provider == "google" else "id",
        email_path="email" if provider == "google" else "login",
        default_scopes=[],
    )
    request = SimpleNamespace(query_params={"code": "code", "state": state})
    return generic_oauth_callback(provider, request, db, db_provider)


def _row(db: Session, key: str) -> UserOAuth:
    return db.query(UserOAuth).filter(UserOAuth.provider == key).one()


def test_the_callback_records_a_builtin_provider_as_the_issuer(db, user, monkeypatch):
    response = _callback(
        db,
        user,
        monkeypatch,
        provider="github",
        app_id="github",
        userinfo={"id": 42, "login": "octocat"},
    )

    assert response.status_code == 200
    assert _row(db, "github").oauth_provider == "github"


def test_the_callback_records_google_for_a_custom_app_backed_by_google(
    db, user, monkeypatch
):
    _custom_app(db, "google")

    response = _callback(
        db,
        user,
        monkeypatch,
        provider="google",
        app_id=CUSTOM_APP,
        userinfo={"sub": "google-sub", "email": "alice@example.com"},
    )

    assert response.status_code == 200
    row = _row(db, CUSTOM_APP)
    assert row.provider == CUSTOM_APP
    assert row.oauth_provider == "google"


def _google_row(db: Session, user: User, *, subject: str = "google-sub") -> UserOAuth:
    row = UserOAuth(
        user_id=int(user.id),
        provider=CUSTOM_APP,
        oauth_provider="google",
        provider_user_id=subject,
        access_token="google-access",
        refresh_token="google-refresh",
    )
    db.add(row)
    db.commit()
    return row


def _retarget(db: Session) -> None:
    db.query(PublicMCPApp).filter_by(app_id=CUSTOM_APP).one().provider_name = "github"
    db.commit()


def _delete(db: Session) -> None:
    db.delete(db.query(PublicMCPApp).filter_by(app_id=CUSTOM_APP).one())
    db.commit()


def _delete_and_recreate(db: Session) -> None:
    _delete(db)
    _custom_app(db, "github")


@pytest.mark.parametrize(
    "admin_change", [_retarget, _delete, _delete_and_recreate], ids=lambda f: f.__name__
)
def test_revocation_targets_the_recorded_issuer_after_a_catalog_change(
    db, user, admin_change
):
    _custom_app(db, "google")
    row = _google_row(db, user)
    admin_change(db)

    db.delete(row)
    db.commit()

    # The deletion's outbox record carries the recorded issuer, whatever the
    # catalog says now; the worker revokes against it.
    [record] = db.query(CredentialRevocation).all()
    assert record.issuer == "google"
    assert decrypt_value_strict(record.refresh_token) == "google-refresh"
    assert record.provider_user_id == "google-sub"


@pytest.mark.parametrize(
    "admin_change", [_retarget, _delete, _delete_and_recreate], ids=lambda f: f.__name__
)
def test_a_sibling_still_references_google_after_a_catalog_change(
    db, user, admin_change
):
    """The reference check matches the recorded issuer too: a retargeted
    custom app's Google row keeps another Google connection's grant alive."""
    _custom_app(db, "google")
    _google_row(db, user)
    admin_change(db)

    assert auth_api.has_other_builtin_oauth_reference(
        db, provider="google", provider_user_id="google-sub"
    )
    assert not auth_api.has_other_builtin_oauth_reference(
        db, provider="github", provider_user_id="google-sub"
    )


def test_an_unknown_issuer_resolves_to_no_revocation(db, caplog):
    """Never a guess from the catalog, even when it names a provider."""
    _custom_app(db, "google")

    with caplog.at_level(logging.WARNING, logger=auth_api.logger.name):
        resolved = auth_api.resolve_builtin_oauth_revocation(
            db,
            oauth_provider=None,
            access_token="access",
            provider_user_id="google-sub",
            refresh_token="refresh",
        )

    assert resolved is None
    assert "recorded no issuing provider" in caplog.text


def test_an_unknown_issuer_row_is_recorded_and_failed_without_a_guess(db, user):
    _custom_app(db, "google")
    row = UserOAuth(
        user_id=int(user.id),
        provider=CUSTOM_APP,
        oauth_provider=None,
        provider_user_id="google-sub",
        access_token="access",
        refresh_token="refresh",
    )
    db.add(row)
    db.commit()
    db.delete(row)
    asyncio.run(drain_credential_revocations(db))

    [record] = db.query(CredentialRevocation).all()
    assert record.issuer is None
    assert record.status == "failed"
    assert "unknown issuer" in record.last_error


def test_an_unknown_issuer_row_blocks_every_providers_revoke(db, user):
    """Nothing proves which grant it holds, so it may share any of them."""
    db.add(
        UserOAuth(
            user_id=int(user.id),
            provider=CUSTOM_APP,
            oauth_provider=None,
            provider_user_id="shared-id",
            access_token="access",
        )
    )
    db.commit()

    for provider in ("google", "github"):
        assert auth_api.has_other_builtin_oauth_reference(
            db, provider=provider, provider_user_id="shared-id"
        )


def test_composio_rows_have_no_issuer_and_are_never_revoked(db, user):
    row = UserOAuth(
        user_id=int(user.id),
        provider="gmail",
        provider_user_id="google-sub",
        access_token="ca_composio-account",
        token_type=COMPOSIO_TOKEN_TYPE,
    )
    db.add(row)
    db.commit()

    assert row.oauth_provider is None
    assert not auth_api.has_other_builtin_oauth_reference(
        db, provider="google", provider_user_id="google-sub"
    )
    db.delete(row)
    db.commit()
    assert db.query(CredentialRevocation).count() == 0


def _issuer_migration() -> ModuleType:
    import xagent.migrations as migrations_pkg

    path = (
        Path(next(iter(migrations_pkg.__path__)))
        / "versions"
        / "20261115_user_oauth_issuer.py"
    )
    spec = importlib.util.spec_from_file_location("user_oauth_issuer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_pre_audit_retarget_cannot_expose_a_live_google_row_to_revocation(
    db, user, monkeypatch
):
    """``acme-mail`` issued a live Google credential, then was retargeted to
    GitHub before catalog auditing existed, so the upgrade sees only
    ``github``. The backfill must leave the row unknown rather than label it
    ``github``: once the app is restored to Google, another user's
    disconnect of the same Google account must still find it and skip the
    project-wide revoke."""
    _custom_app(db, "github")
    survivor = UserOAuth(
        user_id=int(user.id),
        provider=CUSTOM_APP,
        provider_user_id="google-sub",
        access_token="alice-live-google-access",
        refresh_token="alice-live-google-refresh",
    )
    db.add(survivor)
    db.commit()

    migration = _issuer_migration()
    with patch.object(
        migration, "op", Operations(MigrationContext.configure(db.connection()))
    ):
        migration.upgrade()
    db.commit()
    db.refresh(survivor)
    assert survivor.oauth_provider is None

    db.query(PublicMCPApp).filter_by(app_id=CUSTOM_APP).one().provider_name = "google"
    bob = User(username="bob", password_hash="h", is_admin=False)
    db.add(bob)
    db.flush()
    disconnected = UserOAuth(
        user_id=int(bob.id),
        provider="gmail",
        oauth_provider="google",
        provider_user_id="google-sub",
        access_token="bob-access",
        refresh_token="bob-refresh",
    )
    db.add(disconnected)
    db.commit()

    post = Mock()
    monkeypatch.setattr(auth_api.requests, "post", post)
    db.delete(disconnected)
    asyncio.run(drain_credential_revocations(db))

    post.assert_not_called()
    [record] = db.query(CredentialRevocation).all()
    assert record.status == "skipped"
    assert db.get(UserOAuth, int(survivor.id)).access_token == (
        "alice-live-google-access"
    )
