"""Tests for the backfill_mcp_server_ownership_flags migration."""

import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from cryptography.fernet import Fernet
from sqlalchemy import create_engine, text

TARGET_REVISION = "5bb3df522a7d"  # down_revision of the migration under test
MIGRATION_REVISION = "ac3599eb3f1c"


@pytest.fixture
def alembic_config(test_db):
    """Create Alembic config for testing"""
    engine = test_db

    project_root = Path(__file__).parent.parent.parent
    migrations_path = project_root / "src/xagent/migrations"

    config = Config()
    config.set_main_option("script_location", str(migrations_path))
    config.set_main_option("sqlalchemy.url", str(engine.url))

    return config


@pytest.fixture
def encryption_key():
    """Generate a test encryption key (required by earlier migrations in the chain)."""
    key = Fernet.generate_key().decode()
    os.environ["ENCRYPTION_KEY"] = key
    yield key
    del os.environ["ENCRYPTION_KEY"]


@pytest.fixture
def test_db(tmp_path):
    """Create a temporary sqlite database for the full migration chain."""
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    yield engine
    engine.dispose()


def _seed_user_and_server(engine, *, user_id: int, server_id: int) -> None:
    """Insert a users row (the `users` table itself is created by application
    SQLAlchemy metadata, not by the alembic chain, so the alembic-only test
    database needs it created here) plus an mcp_servers row.
    """
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    username VARCHAR(50) NOT NULL,
                    email VARCHAR(100) NOT NULL
                )
                """
            )
        )
        conn.execute(
            text(
                "INSERT INTO users (id, username, email) VALUES (:id, :username, :email)"
            ),
            {
                "id": user_id,
                "username": f"user{user_id}",
                "email": f"user{user_id}@example.com",
            },
        )
        conn.execute(
            text(
                "INSERT INTO mcp_servers (id, name, managed, transport) "
                "VALUES (:id, :name, 'external', 'stdio')"
            ),
            {"id": server_id, "name": f"server-{server_id}"},
        )


def test_upgrade_backfills_all_false_ownership_flags(
    test_db, encryption_key, alembic_config
):
    """A pre-existing row with every flag false/unenforced gets fully backfilled."""
    engine = test_db

    # Run the chain up to (but not including) the migration under test.
    command.upgrade(alembic_config, TARGET_REVISION)

    _seed_user_and_server(engine, user_id=1, server_id=1)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO user_mcpservers
                    (user_id, mcpserver_id, is_owner, can_edit, can_delete,
                     is_shared, is_active, is_default)
                VALUES
                    (1, 1, 0, 0, 0, 0, 1, 0)
                """
            )
        )

    # Now run the migration under test.
    command.upgrade(alembic_config, MIGRATION_REVISION)

    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT is_owner, can_edit, can_delete FROM user_mcpservers "
                "WHERE user_id = 1 AND mcpserver_id = 1"
            )
        ).fetchone()

    assert row is not None
    assert tuple(row) == (1, 1, 1)


def test_upgrade_does_not_clobber_rows_with_a_real_flag(
    test_db, encryption_key, alembic_config
):
    """A row that already has at least one flag set (e.g. is_owner=True) is untouched."""
    engine = test_db

    command.upgrade(alembic_config, TARGET_REVISION)

    _seed_user_and_server(engine, user_id=2, server_id=2)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO user_mcpservers
                    (user_id, mcpserver_id, is_owner, can_edit, can_delete,
                     is_shared, is_active, is_default)
                VALUES
                    (2, 2, 1, 0, 0, 0, 1, 0)
                """
            )
        )

    command.upgrade(alembic_config, MIGRATION_REVISION)

    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT is_owner, can_edit, can_delete FROM user_mcpservers "
                "WHERE user_id = 2 AND mcpserver_id = 2"
            )
        ).fetchone()

    assert row is not None
    assert tuple(row) == (1, 0, 0)
