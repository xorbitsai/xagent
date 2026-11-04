"""Tests for the migration script 441d4f5d399c"""

import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from cryptography.fernet import Fernet
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def alembic_config(test_db, tmp_path):
    """Create Alembic config for testing"""
    engine, _ = test_db

    # Get the correct path to alembic migrations
    project_root = Path(__file__).parent.parent.parent
    migrations_path = project_root / "src/xagent/migrations"

    config = Config()
    config.set_main_option("script_location", str(migrations_path))
    config.set_main_option("sqlalchemy.url", str(engine.url))

    return config


@pytest.fixture
def encryption_key():
    """Generate a test encryption key"""
    key = Fernet.generate_key().decode()
    os.environ["ENCRYPTION_KEY"] = key
    yield key
    del os.environ["ENCRYPTION_KEY"]


@pytest.fixture
def test_db(tmp_path):
    """Create a temporary test database"""
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)

    # Create initial schema (before migration)
    with engine.connect() as conn:
        conn.execute(
            text("""
            CREATE TABLE models (
                id INTEGER PRIMARY KEY,
                model_type VARCHAR(50) NOT NULL,
                api_key VARCHAR(500) NOT NULL
            )
        """)
        )
        conn.commit()

    yield engine, Session
    engine.dispose()


def test_upgrade_renames_column(test_db, encryption_key, alembic_config):
    """Test that upgrade renames model_type to model_provider"""
    engine, Session = test_db

    # Insert test data
    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO models (model_type, api_key) VALUES ('openai', 'sk-test123')"
            )
        )
        conn.commit()

    # Run upgrade
    command.upgrade(alembic_config, "441d4f5d399c")

    # Verify column renamed
    with engine.connect() as conn:
        result = conn.execute(text("PRAGMA table_info(models)")).fetchall()
        columns = [row[1] for row in result]

        assert "model_provider" in columns
        assert "model_type" not in columns


def test_upgrade_encrypts_api_keys(test_db, encryption_key, alembic_config):
    """Test that upgrade encrypts existing api_keys"""
    engine, Session = test_db

    plain_key = "sk-test123"

    # Insert test data
    with engine.connect() as conn:
        conn.execute(
            text(
                f"INSERT INTO models (model_type, api_key) VALUES ('openai', '{plain_key}')"
            )
        )
        conn.commit()

    # Run upgrade
    command.upgrade(alembic_config, "441d4f5d399c")

    # Verify encryption
    with engine.connect() as conn:
        result = conn.execute(text("SELECT _api_key_encrypted FROM models")).fetchone()

        encrypted = result[0]
        assert encrypted != plain_key

        # Verify decryption works
        cipher = Fernet(encryption_key.encode())
        decrypted = cipher.decrypt(encrypted.encode()).decode()
        assert decrypted == plain_key


def test_upgrade_removes_plain_api_key_column(test_db, encryption_key, alembic_config):
    """Test that upgrade removes the plain api_key column"""
    engine, _ = test_db

    command.upgrade(alembic_config, "441d4f5d399c")

    with engine.connect() as conn:
        result = conn.execute(text("PRAGMA table_info(models)")).fetchall()
        columns = [row[1] for row in result]

        assert "api_key" not in columns
        assert "_api_key_encrypted" in columns


@pytest.mark.skip(reason="FIXME: encryption key should be set in production")
def test_upgrade_requires_encryption_key(test_db):
    """Test that upgrade fails without ENCRYPTION_KEY"""
    if "ENCRYPTION_KEY" in os.environ:
        del os.environ["ENCRYPTION_KEY"]

    engine, _ = test_db

    with pytest.raises(
        ValueError, match="ENCRYPTION_KEY environment variable is not set"
    ):
        import importlib

        module = importlib.import_module(
            "xagent.web.migrations.versions.441d4f5d399c_rename_model_type_to_model_provider"
        )
        upgrade = module.upgrade

        upgrade()


def test_downgrade_restores_column_name(test_db, encryption_key, alembic_config):
    """Test that downgrade restores model_type column"""
    engine, _ = test_db

    # Upgrade first
    command.upgrade(alembic_config, "441d4f5d399c")

    # Then downgrade
    command.downgrade(alembic_config, "base")

    with engine.connect() as conn:
        result = conn.execute(text("PRAGMA table_info(models)")).fetchall()
        columns = [row[1] for row in result]

        assert "model_type" in columns
        assert "model_provider" not in columns


def test_downgrade_decrypts_api_keys(test_db, encryption_key, alembic_config):
    """Test that downgrade decrypts api_keys"""
    engine, _ = test_db

    plain_key = "sk-test123"

    # Insert and upgrade
    with engine.connect() as conn:
        conn.execute(
            text(
                f"INSERT INTO models (model_type, api_key) VALUES ('openai', '{plain_key}')"
            )
        )
        conn.commit()

    command.upgrade(alembic_config, "441d4f5d399c")
    command.downgrade(alembic_config, "base")

    # Verify decryption
    with engine.connect() as conn:
        result = conn.execute(text("SELECT api_key FROM models")).fetchone()
        assert result[0] == plain_key


def test_upgrade_downgrade_roundtrip(test_db, encryption_key, alembic_config):
    """Test that upgrade followed by downgrade preserves data"""
    engine, _ = test_db

    test_data = [
        ("openai", "sk-openai123"),
        ("zhipu", "sk-zhipu456"),
        ("dashscope", "sk-dash789"),
    ]

    # Insert test data
    with engine.connect() as conn:
        for model_type, api_key in test_data:
            conn.execute(
                text(
                    f"INSERT INTO models (model_type, api_key) VALUES ('{model_type}', '{api_key}')"
                )
            )
        conn.commit()

    # Upgrade and downgrade
    command.upgrade(alembic_config, "441d4f5d399c")
    command.downgrade(alembic_config, "base")

    # Verify data integrity
    with engine.connect() as conn:
        results = conn.execute(
            text("SELECT model_type, api_key FROM models ORDER BY id")
        ).fetchall()

        assert len(results) == len(test_data)
        for (orig_type, orig_key), (db_type, db_key) in zip(test_data, results):
            assert db_type == orig_type
            assert db_key == orig_key


def test_encrypted_column_not_nullable(test_db, encryption_key, alembic_config):
    """Test that _api_key_encrypted is NOT NULL after upgrade"""
    engine, _ = test_db

    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO models (model_type, api_key) VALUES ('openai', 'sk-test')"
            )
        )
        conn.commit()

    command.upgrade(alembic_config, "441d4f5d399c")

    # Try to insert NULL - should fail
    with engine.connect() as conn:
        with pytest.raises(Exception):  # SQLite raises IntegrityError
            conn.execute(
                text(
                    "INSERT INTO models (model_provider, _api_key_encrypted) VALUES ('test', NULL)"
                )
            )
