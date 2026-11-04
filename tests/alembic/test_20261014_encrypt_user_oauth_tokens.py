import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from cryptography.fernet import Fernet
from sqlalchemy import create_engine, inspect, text

from xagent.core.utils.encryption import get_cipher

VERSIONS = Path(__file__).parent.parent.parent / "src/xagent/migrations/versions"


def _load_migration():
    spec = importlib.util.spec_from_file_location(
        "encrypt_user_oauth_tokens_migration",
        VERSIONS / "20261014_encrypt_user_oauth_tokens.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection):
    return Operations(MigrationContext.configure(connection))


@pytest.fixture(autouse=True)
def _clear_cipher_cache():
    get_cipher.cache_clear()
    yield
    get_cipher.cache_clear()


def test_upgrade_encrypts_legacy_tokens_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
    engine = create_engine(f"sqlite:///{tmp_path / 'oauth.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE user_oauth ("
                "id INTEGER PRIMARY KEY, access_token VARCHAR NOT NULL, "
                "refresh_token VARCHAR)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO user_oauth (id, access_token, refresh_token) VALUES "
                "(1, 'access-token', 'refresh-token'), "
                "(2, '', NULL)"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        first = connection.execute(
            text("SELECT access_token, refresh_token FROM user_oauth ORDER BY id")
        ).all()
        cipher = get_cipher()
        assert first[0].access_token != "access-token"
        assert first[0].refresh_token != "refresh-token"
        assert cipher.decrypt(first[0].access_token.encode()).decode() == "access-token"
        assert (
            cipher.decrypt(first[0].refresh_token.encode()).decode() == "refresh-token"
        )
        assert first[1] == ("", None)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        second = connection.execute(
            text("SELECT access_token, refresh_token FROM user_oauth ORDER BY id")
        ).all()
        assert second == first


def test_upgrade_is_safe_when_user_oauth_schema_is_missing(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'oauth.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert "user_oauth" not in inspect(connection).get_table_names()


def test_downgrade_decrypts_tokens_for_the_legacy_orm(tmp_path, monkeypatch):
    monkeypatch.setenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
    engine = create_engine(f"sqlite:///{tmp_path / 'oauth.db'}")
    migration = _load_migration()
    cipher = get_cipher()
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE user_oauth ("
                "id INTEGER PRIMARY KEY, access_token VARCHAR NOT NULL, "
                "refresh_token VARCHAR)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO user_oauth (id, access_token, refresh_token) "
                "VALUES (1, :access_token, :refresh_token), (2, '', NULL)"
            ),
            {
                "access_token": cipher.encrypt(b"access-token").decode(),
                "refresh_token": cipher.encrypt(b"refresh-token").decode(),
            },
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()
        rows = connection.execute(
            text("SELECT access_token, refresh_token FROM user_oauth ORDER BY id")
        ).all()
        assert rows == [("access-token", "refresh-token"), ("", None)]


def test_upgrade_preserves_ciphertext_from_a_different_key(tmp_path, monkeypatch):
    old_key = Fernet.generate_key().decode()
    monkeypatch.setenv("ENCRYPTION_KEY", old_key)
    old_cipher = get_cipher()
    access_token = old_cipher.encrypt(b"access-token").decode()
    refresh_token = old_cipher.encrypt(b"refresh-token").decode()

    monkeypatch.setenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
    get_cipher.cache_clear()
    engine = create_engine(f"sqlite:///{tmp_path / 'oauth.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE user_oauth ("
                "id INTEGER PRIMARY KEY, access_token VARCHAR NOT NULL, "
                "refresh_token VARCHAR)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO user_oauth (id, access_token, refresh_token) "
                "VALUES (1, :access_token, :refresh_token)"
            ),
            {"access_token": access_token, "refresh_token": refresh_token},
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert connection.execute(
            text("SELECT access_token, refresh_token FROM user_oauth")
        ).one() == (access_token, refresh_token)
