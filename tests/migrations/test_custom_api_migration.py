"""
Tests for custom_apis and user_custom_apis tables migration.

Tests for:
- Migration 654adb358ecd_add_custom_apis_and_user_custom_apis_.py
- Table creation and schema
"""

from pathlib import Path
from typing import Any

import pytest
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

# The previous migration version that this migration depends on
PREVIOUS_MIGRATION_VERSION = "20260410_add_filename_index"


@pytest.fixture
def alembic_config(tmp_path: Path) -> tuple[Config, str]:
    """Create Alembic configuration for testing migrations."""
    # Use in-memory SQLite for testing
    db_path = tmp_path / "test.db"
    db_url = f"sqlite:///{db_path}"

    # Create minimal alembic config
    config = Config()
    config.set_main_option("sqlalchemy.url", db_url)
    config.set_main_option("script_location", "src/xagent/migrations")

    return config, db_url


@pytest.fixture
def engine_with_migration(alembic_config: tuple[Config, str]) -> Any:
    """Create engine with migration applied."""
    from alembic import command

    config, db_url = alembic_config
    engine = create_engine(db_url)

    # Stamp to previous version, then upgrade to current
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS alembic_version (version_num VARCHAR(32) NOT NULL)"
            )
        )
        # Create users table since user_custom_apis references it
        conn.execute(
            text(
                """CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    hashed_password VARCHAR(255) NOT NULL,
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    is_superuser BOOLEAN NOT NULL DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )"""
            )
        )
        conn.execute(
            text(
                f"INSERT INTO alembic_version (version_num) VALUES ('{PREVIOUS_MIGRATION_VERSION}')"
            )
        )

    # Run the migration using alembic command
    command.upgrade(config, "654adb358ecd")

    # Dispose and recreate engine so inspector can see the new indexes
    engine.dispose()
    engine = create_engine(db_url)

    yield engine

    # Cleanup
    engine.dispose()


class TestCustomApiMigrationCreatesTables:
    """Tests that migration creates the custom_apis and user_custom_apis tables."""

    def test_tables_exist(self, engine_with_migration: Any) -> None:
        """Test that tables are created."""
        inspector = inspect(engine_with_migration)
        tables = inspector.get_table_names()
        assert "custom_apis" in tables
        assert "user_custom_apis" in tables

    def test_custom_apis_columns(self, engine_with_migration: Any) -> None:
        """Test that custom_apis has all expected columns."""
        inspector = inspect(engine_with_migration)
        columns = {col["name"]: col for col in inspector.get_columns("custom_apis")}

        expected_columns = {
            "id",
            "name",
            "description",
            "url",
            "method",
            "headers",
            "env",
            "created_at",
            "updated_at",
        }

        assert set(columns.keys()) == expected_columns

    def test_user_custom_apis_columns(self, engine_with_migration: Any) -> None:
        """Test that user_custom_apis has all expected columns."""
        inspector = inspect(engine_with_migration)
        columns = {
            col["name"]: col for col in inspector.get_columns("user_custom_apis")
        }

        expected_columns = {
            "id",
            "user_id",
            "custom_api_id",
            "is_owner",
            "can_edit",
            "can_delete",
            "is_active",
            "is_default",
            "is_shared",
            "created_at",
            "updated_at",
        }

        assert set(columns.keys()) == expected_columns

    def test_foreign_keys(self, engine_with_migration: Any) -> None:
        """Test foreign key constraints on user_custom_apis."""
        inspector = inspect(engine_with_migration)
        fks = inspector.get_foreign_keys("user_custom_apis")

        referred_tables = {fk["referred_table"] for fk in fks}
        assert "users" in referred_tables
        assert "custom_apis" in referred_tables


class TestMigrationFunctionality:
    """Tests for migration functionality."""

    def test_can_insert_records(self, engine_with_migration: Any) -> None:
        """Test that records can be inserted after migration."""
        with engine_with_migration.begin() as conn:
            # Insert a user first
            conn.execute(
                text(
                    "INSERT INTO users (id, email, hashed_password) VALUES (1, 'test@test.com', 'hash')"
                )
            )

            # Insert custom api
            result1 = conn.execute(
                text("""
                INSERT INTO custom_apis (id, name, description, url)
                VALUES (1, 'my_api', 'Test API', 'https://api.test.com')
            """)
            )
            assert result1.rowcount == 1

            # Insert user custom api
            result2 = conn.execute(
                text("""
                INSERT INTO user_custom_apis (user_id, custom_api_id, is_active, is_owner, can_edit, can_delete, is_default, is_shared)
                VALUES (1, 1, 1, 1, 1, 1, 0, 0)
            """)
            )
            assert result2.rowcount == 1
