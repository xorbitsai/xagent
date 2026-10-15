from cryptography.fernet import Fernet
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from xagent.core.utils.encryption import get_cipher
from xagent.web.models.database import Base
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth


@pytest.fixture(autouse=True)
def _clear_cipher_cache():
    get_cipher.cache_clear()
    yield
    get_cipher.cache_clear()


def test_user_oauth_tokens_round_trip_encrypted_and_presence_filters(monkeypatch):
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("ENCRYPTION_KEY", key)
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine, tables=[User.__table__, UserOAuth.__table__])

    with Session(engine) as session:
        user = User(
            username="oauth-encryption",
            email="oauth-encryption@example.com",
            password_hash="not-a-real-password-hash",
        )
        session.add(user)
        session.flush()
        session.add_all(
            [
                UserOAuth(
                    user_id=user.id,
                    provider="test",
                    provider_user_id="encrypted",
                    access_token="access-token",
                    refresh_token="refresh-token",
                ),
                UserOAuth(
                    user_id=user.id,
                    provider="test-empty",
                    provider_user_id="empty",
                    access_token="",
                    refresh_token=None,
                ),
            ]
        )
        session.commit()

        encrypted = session.execute(
            text(
                "SELECT access_token, refresh_token FROM user_oauth WHERE provider = 'test'"
            )
        ).one()
        assert encrypted.access_token != "access-token"
        assert encrypted.refresh_token != "refresh-token"

        row = session.query(UserOAuth).filter_by(provider="test").one()
        assert row.access_token == "access-token"
        assert row.refresh_token == "refresh-token"
        assert (
            session.query(UserOAuth).filter(UserOAuth.access_token == "").one().provider
            == "test-empty"
        )
        assert (
            session.query(UserOAuth)
            .filter(UserOAuth.refresh_token.is_(None))
            .one()
            .provider
            == "test-empty"
        )


def test_wrong_key_marks_tokens_for_reauthorization_without_rewriting(
    monkeypatch,
):
    original_key = Fernet.generate_key().decode()
    monkeypatch.setenv("ENCRYPTION_KEY", original_key)
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine, tables=[User.__table__, UserOAuth.__table__])

    with Session(engine) as session:
        user = User(
            username="oauth-wrong-key",
            email="oauth-wrong-key@example.com",
            password_hash="not-a-real-password-hash",
        )
        session.add(user)
        session.flush()
        session.add(
            UserOAuth(
                user_id=user.id,
                provider="test",
                provider_user_id="encrypted",
                access_token="access-token",
                refresh_token="refresh-token",
            )
        )
        session.commit()
        raw_before = tuple(
            session.execute(
                text(
                    "SELECT access_token, refresh_token "
                    "FROM user_oauth WHERE provider = 'test'"
                )
            ).one()
        )

    monkeypatch.setenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
    get_cipher.cache_clear()
    with Session(engine) as session:
        row = session.query(UserOAuth).filter_by(provider="test").one()
        assert row.access_token == ""
        assert row.refresh_token == ""
        raw_after = tuple(
            session.execute(
                text(
                    "SELECT access_token, refresh_token "
                    "FROM user_oauth WHERE provider = 'test'"
                )
            ).one()
        )

    assert raw_after == raw_before
