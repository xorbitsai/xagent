import importlib.util
from pathlib import Path
from unittest.mock import patch

from cryptography.fernet import Fernet
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text

from xagent.core.utils.encryption import decrypt_value, get_cipher


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


def test_upgrade_encrypts_legacy_tokens_and_is_idempotent(tmp_path, monkeypatch):
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
        assert decrypt_value(first[0].access_token) == "access-token"
        assert decrypt_value(first[0].refresh_token) == "refresh-token"
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
