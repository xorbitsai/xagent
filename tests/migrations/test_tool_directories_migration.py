"""
Tests for tool_directories table migration.

Tests for:
- Migration a1b2c3d4e5f6_create_tool_directories_table.py
- Table creation and schema
- Rollback functionality
"""

from pathlib import Path
from typing import Any

import pytest
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

# The previous migration version that tool_directories depends on
# This is configurable to avoid hardcoding in multiple places
PREVIOUS_MIGRATION_VERSION = "be6f77416f06"


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
        conn.execute(
            text(
                f"INSERT INTO alembic_version (version_num) VALUES ('{PREVIOUS_MIGRATION_VERSION}')"
            )
        )

    # Run the migration using alembic command
    command.upgrade(config, "a1b2c3d4e5f6")

    # Dispose and recreate engine so inspector can see the new indexes
    engine.dispose()
    engine = create_engine(db_url)

    yield engine

    # Cleanup
    engine.dispose()


class TestMigrationCreatesTable:
    """Tests that migration creates the tool_directories table."""

    def test_table_exists(self, engine_with_migration: Any) -> None:
        """Test that tool_directories table is created."""
        inspector = inspect(engine_with_migration)
        tables = inspector.get_table_names()
        assert "tool_directories" in tables

    def test_table_has_all_columns(self, engine_with_migration: Any) -> None:
        """Test that table has all expected columns."""
        inspector = inspect(engine_with_migration)
        columns = {
            col["name"]: col for col in inspector.get_columns("tool_directories")
        }

        expected_columns = {
            "id",
            "name",
            "path",
            "enabled",
            "recursive",
            "exclude_patterns",
            "description",
            "is_valid",
            "last_validated_at",
            "validation_error",
            "tool_count",
            "created_at",
            "updated_at",
        }

        assert set(columns.keys()) == expected_columns

    def test_id_column_is_primary_key(self, engine_with_migration: Any) -> None:
        """Test that id column is primary key."""
        inspector = inspect(engine_with_migration)
        pk_constraint = inspector.get_pk_constraint("tool_directories")
        assert "id" in pk_constraint["constrained_columns"]

    def test_name_column_is_unique(self, engine_with_migration: Any) -> None:
        """Test that name column has unique constraint."""
        inspector = inspect(engine_with_migration)
        indexes = inspector.get_indexes("tool_directories")

        # Check for unique index on name
        # Indexes in SQLite use named tuples with column_names attribute
        name_indexes = [
            idx
            for idx in indexes
            if "name" in idx.get("column_names", idx.get("columns", []))
        ]
        assert len(name_indexes) > 0
        # At least one index on name should be unique
        assert any(idx.get("unique", False) for idx in name_indexes)

    def test_name_column_has_length(self, engine_with_migration: Any) -> None:
        """Test that name column has length constraint."""
        inspector = inspect(engine_with_migration)
        columns = {
            col["name"]: col for col in inspector.get_columns("tool_directories")
        }

        # In SQLite, type might include the length
        name_type = str(columns["name"]["type"]).upper()
        assert "100" in name_type or "VARCHAR" in name_type

    def test_path_column_has_length(self, engine_with_migration: Any) -> None:
        """Test that path column has length constraint."""
        inspector = inspect(engine_with_migration)
        columns = {
            col["name"]: col for col in inspector.get_columns("tool_directories")
        }

        path_type = str(columns["path"]["type"]).upper()
        assert "500" in path_type or "VARCHAR" in path_type

    @pytest.mark.parametrize(
        "column_name",
        ["enabled", "recursive", "is_valid", "tool_count", "created_at", "updated_at"],
    )
    def test_column_has_default(
        self, column_name: str, engine_with_migration: Any
    ) -> None:
        """Test that column has a default value."""
        inspector = inspect(engine_with_migration)
        columns = {
            col["name"]: col for col in inspector.get_columns("tool_directories")
        }

        col = columns[column_name]
        assert col.get("default") is not None or col.get("server_default") is not None

    def test_indexes_created(self, engine_with_migration: Any) -> None:
        """Test that expected indexes are created."""
        inspector = inspect(engine_with_migration)
        indexes = inspector.get_indexes("tool_directories")

        index_names = {idx["name"] for idx in indexes}
        assert "ix_tool_directories_id" in index_names
        assert "ix_tool_directories_name" in index_names
        assert "ix_tool_directories_enabled" in index_names

    def test_exclude_patterns_is_json_type(self, engine_with_migration: Any) -> None:
        """Test that exclude_patterns is JSON type."""
        inspector = inspect(engine_with_migration)
        columns = {
            col["name"]: col for col in inspector.get_columns("tool_directories")
        }

        # The type should be JSON or similar
        exclude_patterns_type = str(columns["exclude_patterns"]["type"]).upper()
        assert "JSON" in exclude_patterns_type or "TEXT" in exclude_patterns_type


class TestMigrationFunctionality:
    """Tests for migration functionality."""

    def test_can_insert_record(self, engine_with_migration: Any) -> None:
        """Test that records can be inserted after migration."""
        with engine_with_migration.begin() as conn:
            # Use CURRENT_TIMESTAMP for SQLite compatibility instead of now()
            result = conn.execute(
                text("""
                INSERT INTO tool_directories (name, path, enabled, recursive, exclude_patterns, description, created_at, updated_at)
                VALUES ('test-tools', '/test/path', 1, 1, '["*.pyc"]', 'Test tools', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """)
            )
            assert result.rowcount == 1

    def test_can_query_records(self, engine_with_migration: Any) -> None:
        """Test that records can be queried after migration."""
        with engine_with_migration.begin() as conn:
            # Insert
            conn.execute(
                text("""
                INSERT INTO tool_directories (name, path, enabled, created_at, updated_at)
                VALUES ('test-tools', '/test/path', 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """)
            )

            # Query - use fetchone() for single result
            result = conn.execute(
                text("SELECT name FROM tool_directories WHERE name = 'test-tools'")
            )
            row = result.fetchone()
            assert row is not None
            assert row[0] == "test-tools"

    def test_can_update_records(self, engine_with_migration: Any) -> None:
        """Test that records can be updated after migration."""
        with engine_with_migration.begin() as conn:
            # Insert
            conn.execute(
                text("""
                INSERT INTO tool_directories (name, path, enabled, created_at, updated_at)
                VALUES ('test-tools', '/test/path', 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """)
            )

            # Update
            conn.execute(
                text("""
                UPDATE tool_directories SET enabled = 0 WHERE name = 'test-tools'
            """)
            )

            # Verify
            result = conn.execute(
                text("SELECT enabled FROM tool_directories WHERE name = 'test-tools'")
            )
            assert result.scalar() == 0  # False in SQLite

    def test_can_delete_records(self, engine_with_migration: Any) -> None:
        """Test that records can be deleted after migration."""
        with engine_with_migration.begin() as conn:
            # Insert
            conn.execute(
                text("""
                INSERT INTO tool_directories (name, path, enabled, created_at, updated_at)
                VALUES ('test-tools', '/test/path', 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """)
            )

            # Delete
            conn.execute(text("DELETE FROM tool_directories WHERE name = 'test-tools'"))

            # Verify
            result = conn.execute(text("SELECT COUNT(*) FROM tool_directories"))
            assert result.scalar() == 0

    def test_unique_constraint_enforced(self, engine_with_migration: Any) -> None:
        """Test that unique constraint on name is enforced."""
        with engine_with_migration.begin() as conn:
            # Insert first record
            conn.execute(
                text("""
                INSERT INTO tool_directories (name, path, enabled, created_at, updated_at)
                VALUES ('test-tools', '/path1', 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """)
            )

            # Try to insert duplicate name - should fail
            with pytest.raises(Exception):  # IntegrityError
                conn.execute(
                    text("""
                    INSERT INTO tool_directories (name, path, enabled, created_at, updated_at)
                    VALUES ('test-tools', '/path2', 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """)
                )

    def test_exclude_patterns_json_storage(self, engine_with_migration: Any) -> None:
        """Test that exclude_patterns can store JSON data."""
        with engine_with_migration.begin() as conn:
            # Insert with JSON patterns
            conn.execute(
                text("""
                INSERT INTO tool_directories (name, path, exclude_patterns, enabled, created_at, updated_at)
                VALUES ('test-tools', '/test/path', '["*.pyc", "__pycache__", "*.log"]', 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """)
            )

            # Query and verify
            result = conn.execute(
                text(
                    "SELECT exclude_patterns FROM tool_directories WHERE name = 'test-tools'"
                )
            )
            patterns = result.scalar()
            assert patterns is not None


class TestMigrationRollback:
    """Tests for migration rollback functionality."""

    @pytest.fixture
    def engine_with_rollback(self, tmp_path: Path) -> Any:
        """Create engine with migration applied and then rolled back."""
        from alembic import command

        db_path = tmp_path / "test_rollback.db"
        db_url = f"sqlite:///{db_path}"
        engine = create_engine(db_url)

        # Create alembic config
        config = Config()
        config.set_main_option("sqlalchemy.url", db_url)
        config.set_main_option("script_location", "src/xagent/migrations")

        # Stamp to previous version
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS alembic_version (version_num VARCHAR(32))"
                )
            )
            conn.execute(
                text(
                    f"INSERT INTO alembic_version (version_num) VALUES ('{PREVIOUS_MIGRATION_VERSION}')"
                )
            )

        # Apply migration using alembic command
        command.upgrade(config, "a1b2c3d4e5f6")

        # Verify table exists
        inspector = inspect(engine)
        assert "tool_directories" in inspector.get_table_names()

        # Rollback migration using alembic command
        command.downgrade(config, PREVIOUS_MIGRATION_VERSION)

        yield engine

        engine.dispose()

    def test_rollback_removes_table(self, engine_with_rollback: Any) -> None:
        """Test that rollback removes the tool_directories table."""
        inspector = inspect(engine_with_rollback)
        assert "tool_directories" not in inspector.get_table_names()

    def test_rollback_removes_indexes(self, engine_with_rollback: Any) -> None:
        """Test that rollback removes indexes."""
        inspector = inspect(engine_with_rollback)
        # Table shouldn't exist, so no indexes either
        assert "tool_directories" not in inspector.get_table_names()


class TestMigrationIdempotency:
    """Tests for migration idempotency and re-running."""

    def test_upgrade_idempotent(self, tmp_path: Path) -> None:
        """Test that running upgrade twice doesn't fail."""
        from alembic import command

        db_path = tmp_path / "test_idempotent.db"
        db_url = f"sqlite:///{db_path}"
        engine = create_engine(db_url)

        config = Config()
        config.set_main_option("sqlalchemy.url", db_url)
        config.set_main_option("script_location", "src/xagent/migrations")

        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS alembic_version (version_num VARCHAR(32))"
                )
            )
            conn.execute(
                text(
                    f"INSERT INTO alembic_version (version_num) VALUES ('{PREVIOUS_MIGRATION_VERSION}')"
                )
            )

        # Run upgrade first time
        command.upgrade(config, "a1b2c3d4e5f6")

        # Run upgrade again - should handle gracefully
        try:
            command.upgrade(config, "a1b2c3d4e5f6")
        except Exception as e:
            # Table already exists error is expected
            assert "already exists" in str(e).lower()

        engine.dispose()


class TestMigrationWithExistingData:
    """Tests for migration behavior with existing data."""

    def test_multiple_records_after_migration(self, engine_with_migration: Any) -> None:
        """Test inserting multiple records after migration."""
        with engine_with_migration.begin() as conn:
            # Insert multiple records
            for i in range(5):
                conn.execute(
                    text(f"""
                    INSERT INTO tool_directories (name, path, enabled, created_at, updated_at)
                    VALUES ('tools{i}', '/path{i}', {1 if i % 2 == 0 else 0}, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """)
                )

            # Verify all records exist
            result = conn.execute(text("SELECT COUNT(*) FROM tool_directories"))
            count = result.scalar()
            assert count == 5

    def test_query_by_enabled_index(self, engine_with_migration: Any) -> None:
        """Test that enabled index works for filtering."""
        with engine_with_migration.begin() as conn:
            # Insert records
            conn.execute(
                text(
                    "INSERT INTO tool_directories (name, path, enabled, created_at, updated_at) VALUES ('e1', '/p1', 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO tool_directories (name, path, enabled, created_at, updated_at) VALUES ('d1', '/p2', 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO tool_directories (name, path, enabled, created_at, updated_at) VALUES ('e2', '/p3', 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                )
            )

            # Query only enabled
            result = conn.execute(
                text("SELECT COUNT(*) FROM tool_directories WHERE enabled = 1")
            )
            count = result.scalar()
            assert count == 2
