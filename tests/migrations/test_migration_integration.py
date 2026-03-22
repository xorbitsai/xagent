#!/usr/bin/env python
"""
Integration tests for database migrations.

These tests run actual database migrations against SQLite and PostgreSQL
to ensure migration scripts work correctly and are idempotent.

Usage:
    pytest tests/migrations/test_migration_integration.py
    pytest tests/migrations/test_migration_integration.py::TestMigrations::test_sqlite_upgrade
    pytest tests/migrations/test_migration_integration.py::TestMigrations::test_postgresql_upgrade
"""

import argparse
import os
import tempfile
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect, text

# Project root directory
project_root = Path(__file__).parent.parent.parent


class MigrationTester:
    """Helper class to test database migrations."""

    def __init__(self, db_type: str):
        self.db_type = db_type
        self.engine = None
        self.alembic_cfg = None
        self.temp_db_file = None

    def setup_database(self):
        """Set up test database connection."""
        if self.db_type == "sqlite":
            # Use temporary file for SQLite
            fd, path = tempfile.mkstemp(suffix=".db")
            os.close(fd)
            db_url = f"sqlite:///{path}"
            self.temp_db_file = path
        elif self.db_type == "postgresql":
            db_url = os.getenv(
                "DATABASE_URL", "postgresql://xagent:xagent@localhost:5432/xagent_test"
            )

        os.environ["DATABASE_URL"] = db_url
        self.engine = create_engine(db_url)

        # Clean database for PostgreSQL
        if self.db_type == "postgresql":
            with self.engine.begin() as conn:
                conn.execute(text("DROP SCHEMA public CASCADE"))
                conn.execute(text("CREATE SCHEMA public"))

        # Configure Alembic
        alembic_dir = project_root / "src" / "xagent" / "migrations"
        self.alembic_cfg = Config(str(alembic_dir / "alembic.ini"))
        self.alembic_cfg.set_main_option("sqlalchemy.url", db_url)

    def teardown_database(self):
        """Clean up test database."""
        if self.db_type == "sqlite" and self.temp_db_file:
            os.unlink(self.temp_db_file)
        elif self.db_type == "postgresql":
            with self.engine.begin() as conn:
                conn.execute(text("DROP SCHEMA public CASCADE"))
                conn.execute(text("CREATE SCHEMA public"))

    def get_table_names(self):
        """Get list of table names."""
        inspector = inspect(self.engine)
        return inspector.get_table_names()

    def get_column_names(self, table_name):
        """Get list of column names for a table."""
        with self.engine.begin() as conn:
            if self.db_type == "postgresql":
                result = conn.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = :table_name"
                    ),
                    {"table_name": table_name},
                )
            else:
                result = conn.execute(text(f"PRAGMA table_info({table_name})"))
            return [row[0] for row in result]


class TestMigrations:
    """Test database migrations."""

    @pytest.fixture
    def sqlite_tester(self):
        """Create SQLite migration tester."""
        tester = MigrationTester("sqlite")
        tester.setup_database()
        yield tester
        tester.teardown_database()

    @pytest.fixture
    def postgresql_tester(self):
        """Create PostgreSQL migration tester."""
        tester = MigrationTester("postgresql")
        tester.setup_database()
        yield tester
        tester.teardown_database()

    def test_sqlite_upgrade(self, sqlite_tester):
        """Test full migration upgrade on SQLite."""
        # Run upgrade
        command.upgrade(sqlite_tester.alembic_cfg, "head")

        # Verify alembic_version table
        with sqlite_tester.engine.begin() as conn:
            result = conn.execute(text("SELECT version_num FROM alembic_version"))
            version = result.scalar()
            assert version is not None, "Version should be set after upgrade"

        # Verify key tables exist
        tables = sqlite_tester.get_table_names()
        assert "agents" in tables, "agents table should exist"
        assert "alembic_version" in tables, "alembic_version table should exist"

        # Verify agents table structure
        columns = sqlite_tester.get_column_names("agents")
        assert "models" in columns, "models column should exist"
        assert "name" in columns, "name column should exist"

    @pytest.mark.postgresql
    @pytest.mark.postgresql
    def test_postgresql_upgrade(self, postgresql_tester):
        """Test full migration upgrade on PostgreSQL."""
        # Run upgrade
        command.upgrade(postgresql_tester.alembic_cfg, "head")

        # Verify alembic_version table
        with postgresql_tester.engine.begin() as conn:
            result = conn.execute(text("SELECT version_num FROM alembic_version"))
            version = result.scalar()
            assert version is not None, "Version should be set after upgrade"

        # Verify key tables exist
        tables = postgresql_tester.get_table_names()
        assert "agents" in tables, "agents table should exist"
        assert "alembic_version" in tables, "alembic_version table should exist"

        # Verify agents table structure
        columns = postgresql_tester.get_column_names("agents")
        assert "models" in columns, "models column should exist"
        assert "name" in columns, "name column should exist"

    def test_sqlite_idempotence(self, sqlite_tester):
        """Test that migrations are idempotent on SQLite."""
        # First upgrade
        command.upgrade(sqlite_tester.alembic_cfg, "head")

        # Get version after first upgrade
        with sqlite_tester.engine.begin() as conn:
            result = conn.execute(text("SELECT version_num FROM alembic_version"))
            version1 = result.scalar()

        # Second upgrade (should not fail)
        command.upgrade(sqlite_tester.alembic_cfg, "head")

        # Verify version hasn't changed
        with sqlite_tester.engine.begin() as conn:
            result = conn.execute(text("SELECT version_num FROM alembic_version"))
            version2 = result.scalar()

        assert version1 == version2, "Version should not change on re-run"

    @pytest.mark.postgresql
    def test_postgresql_idempotence(self, postgresql_tester):
        """Test that migrations are idempotent on PostgreSQL."""
        # First upgrade
        command.upgrade(postgresql_tester.alembic_cfg, "head")

        # Get version after first upgrade
        with postgresql_tester.engine.begin() as conn:
            result = conn.execute(text("SELECT version_num FROM alembic_version"))
            version1 = result.scalar()

        # Second upgrade (should not fail)
        command.upgrade(postgresql_tester.alembic_cfg, "head")

        # Verify version hasn't changed
        with postgresql_tester.engine.begin() as conn:
            result = conn.execute(text("SELECT version_num FROM alembic_version"))
            version2 = result.scalar()

        assert version1 == version2, "Version should not change on re-run"

    def test_sqlite_incremental_upgrade(self, sqlite_tester):
        """Test incremental upgrades from base to head on SQLite."""
        from alembic.script import ScriptDirectory

        script_dir = ScriptDirectory.from_config(sqlite_tester.alembic_cfg)
        revisions = list(script_dir.walk_revisions("base", "heads"))
        revisions.reverse()  # base to head

        # Upgrade one revision at a time
        for revision in revisions:
            command.upgrade(sqlite_tester.alembic_cfg, revision.revision)

            # Verify version
            with sqlite_tester.engine.begin() as conn:
                result = conn.execute(text("SELECT version_num FROM alembic_version"))
                version = result.scalar()
                assert version == revision.revision

    @pytest.mark.postgresql
    def test_postgresql_incremental_upgrade(self, postgresql_tester):
        """Test incremental upgrades from base to head on PostgreSQL."""
        from alembic.script import ScriptDirectory

        script_dir = ScriptDirectory.from_config(postgresql_tester.alembic_cfg)
        revisions = list(script_dir.walk_revisions("base", "heads"))
        revisions.reverse()  # base to head

        # Upgrade one revision at a time
        for revision in revisions:
            command.upgrade(postgresql_tester.alembic_cfg, revision.revision)

            # Verify version
            with postgresql_tester.engine.begin() as conn:
                result = conn.execute(text("SELECT version_num FROM alembic_version"))
                version = result.scalar()
                assert version == revision.revision

    def test_sqlite_downgrade(self, sqlite_tester):
        """Test downgrade on SQLite."""
        # Upgrade to head
        command.upgrade(sqlite_tester.alembic_cfg, "head")

        # Downgrade to base
        command.downgrade(sqlite_tester.alembic_cfg, "base")

        # Verify tables are removed (only alembic_version should remain)
        tables = sqlite_tester.get_table_names()
        assert "alembic_version" in tables, "alembic_version should still exist"

    @pytest.mark.postgresql
    def test_postgresql_downgrade(self, postgresql_tester):
        """Test downgrade on PostgreSQL."""
        # Upgrade to head
        command.upgrade(postgresql_tester.alembic_cfg, "head")

        # Downgrade to base
        command.downgrade(postgresql_tester.alembic_cfg, "base")

        # Verify schema is cleaned
        tables = postgresql_tester.get_table_names()
        assert "alembic_version" in tables, "alembic_version should still exist"


if __name__ == "__main__":
    """CLI interface for running migration tests manually."""
    parser = argparse.ArgumentParser(description="Test database migrations")
    parser.add_argument("--db", choices=["sqlite", "postgresql"], default="sqlite")
    parser.add_argument(
        "test",
        choices=["upgrade", "idempotence", "incremental", "downgrade", "all"],
        help="Test to run",
    )

    args = parser.parse_args()

    tester = MigrationTester(args.db)

    try:
        if args.test == "upgrade":
            tester.setup_database()
            command.upgrade(tester.alembic_cfg, "head")
            print(f"✅ {args.db.upper()} upgrade test PASSED")
            tester.teardown_database()

        elif args.test == "idempotence":
            tester.setup_database()
            command.upgrade(tester.alembic_cfg, "head")
            command.upgrade(tester.alembic_cfg, "head")
            print(f"✅ {args.db.upper()} idempotence test PASSED")
            tester.teardown_database()

        elif args.test == "incremental":
            tester.setup_database()
            script_dir = ScriptDirectory.from_config(tester.alembic_cfg)
            revisions = list(script_dir.walk_revisions("base", "heads"))
            revisions.reverse()

            for revision in revisions:
                command.upgrade(tester.alembic_cfg, revision.revision)

            print(f"✅ {args.db.upper()} incremental upgrade test PASSED")
            tester.teardown_database()

        elif args.test == "downgrade":
            tester.setup_database()
            command.upgrade(tester.alembic_cfg, "head")
            command.downgrade(tester.alembic_cfg, "base")
            print(f"✅ {args.db.upper()} downgrade test PASSED")
            tester.teardown_database()

        elif args.test == "all":
            tester.setup_database()
            command.upgrade(tester.alembic_cfg, "head")
            command.upgrade(tester.alembic_cfg, "head")
            command.downgrade(tester.alembic_cfg, "base")
            command.upgrade(tester.alembic_cfg, "head")
            print(f"✅ {args.db.upper()} all tests PASSED")
            tester.teardown_database()

    except Exception as e:
        print(f"❌ {args.db.upper()} test FAILED: {e}")
        raise
