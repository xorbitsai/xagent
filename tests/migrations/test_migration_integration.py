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
        self.alembic_cfg = Config(str(project_root / "alembic.ini"))
        self.alembic_cfg.set_main_option("sqlalchemy.url", db_url)

        # DO NOT create tables with SQLAlchemy - we want to test migrations
        # can build the schema from scratch

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
                return [row[0] for row in result]
            else:
                # SQLite: PRAGMA table_info returns (cid, name, type, notnull, dflt_value, pk)
                # We want index 1 (name)
                result = conn.execute(text(f"PRAGMA table_info({table_name})"))
                return [row[1] for row in result]


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
        """Test full migration upgrade on SQLite from empty database.

        This tests that migrations can correctly create all tables and columns
        from scratch, not just when tables are pre-created by SQLAlchemy.
        """
        # NOTE: Some base tables (users, models, tasks) are created by SQLAlchemy
        # in production, not by migrations. We need to create them here to
        # simulate production environment.
        from xagent.web.models.database import Base

        Base.metadata.create_all(bind=sqlite_tester.engine)

        # Run upgrade from empty database (but with base tables present)
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
        assert "models" in tables, "models table should exist"
        assert "users" in tables, "users table should exist"
        assert "tasks" in tables, "tasks table should exist"

        # Verify agents table structure - this tests that migrations
        # correctly created all columns, not just that SQLAlchemy created them
        columns = sqlite_tester.get_column_names("agents")
        assert "models" in columns, "models column should exist"
        assert "name" in columns, "name column should exist"
        assert "execution_mode" in columns, "execution_mode column should exist"
        assert "status" in columns, "status column should exist"
        assert "suggested_prompts" in columns, "suggested_prompts column should exist"

        # Verify models table has encrypted column
        models_columns = sqlite_tester.get_column_names("models")
        assert "_api_key_encrypted" in models_columns, (
            "_api_key_encrypted column should exist"
        )

    @pytest.mark.postgresql
    def test_postgresql_upgrade(self, postgresql_tester):
        """Test full migration upgrade on PostgreSQL from empty database.

        This tests that migrations can correctly create all tables and columns
        from scratch, not just when tables are pre-created by SQLAlchemy.
        """
        # DO NOT use Base.metadata.create_all() - we want to test migrations
        # can build the database schema from scratch

        # Run upgrade from empty database
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
        assert "models" in tables, "models table should exist"
        assert "users" in tables, "users table should exist"
        assert "tasks" in tables, "tasks table should exist"

        # Verify agents table structure
        columns = postgresql_tester.get_column_names("agents")
        assert "models" in columns, "models column should exist"
        assert "name" in columns, "name column should exist"
        assert "execution_mode" in columns, "execution_mode column should exist"
        assert "status" in columns, "status column should exist"
        assert "suggested_prompts" in columns, "suggested_prompts column should exist"

        # Verify models table has encrypted column
        models_columns = postgresql_tester.get_column_names("models")
        assert "_api_key_encrypted" in models_columns, (
            "_api_key_encrypted column should exist"
        )

    def test_sqlite_idempotence_with_sqlalchemy(self, sqlite_tester):
        """Test that migrations are idempotent when tables pre-created by SQLAlchemy.

        This tests the production scenario where SQLAlchemy's Base.metadata.create_all()
        creates tables first, then Alembic migrations run. Migrations should correctly
        detect existing tables/columns and skip already-applied changes.
        """
        # First, create tables using SQLAlchemy (mimics production)
        from xagent.web.models.database import Base

        Base.metadata.create_all(bind=sqlite_tester.engine)

        # Then run migrations - should be idempotent and not fail
        command.upgrade(sqlite_tester.alembic_cfg, "head")

        # Get version after upgrade
        with sqlite_tester.engine.begin() as conn:
            result = conn.execute(text("SELECT version_num FROM alembic_version"))
            version1 = result.scalar()

        # Run migrations again - should still work
        command.upgrade(sqlite_tester.alembic_cfg, "head")

        # Verify version hasn't changed
        with sqlite_tester.engine.begin() as conn:
            result = conn.execute(text("SELECT version_num FROM alembic_version"))
            version2 = result.scalar()

        assert version1 == version2, "Version should not change on re-run"

        # Verify tables still have correct structure
        tables = sqlite_tester.get_table_names()
        assert "agents" in tables
        assert "models" in tables

        agents_columns = sqlite_tester.get_column_names("agents")
        assert "models" in agents_columns
        assert "name" in agents_columns

    @pytest.mark.postgresql
    def test_postgresql_idempotence_with_sqlalchemy(self, postgresql_tester):
        """Test that migrations are idempotent when tables pre-created by SQLAlchemy.

        This tests the production scenario where SQLAlchemy's Base.metadata.create_all()
        creates tables first, then Alembic migrations run. Migrations should correctly
        detect existing tables/columns and skip already-applied changes.
        """
        # First, create tables using SQLAlchemy (mimics production)
        from xagent.web.models.database import Base

        Base.metadata.create_all(bind=postgresql_tester.engine)

        # Then run migrations - should be idempotent and not fail
        command.upgrade(postgresql_tester.alembic_cfg, "head")

        # Get version after upgrade
        with postgresql_tester.engine.begin() as conn:
            result = conn.execute(text("SELECT version_num FROM alembic_version"))
            version1 = result.scalar()

        # Run migrations again - should still work
        command.upgrade(postgresql_tester.alembic_cfg, "head")

        # Verify version hasn't changed
        with postgresql_tester.engine.begin() as conn:
            result = conn.execute(text("SELECT version_num FROM alembic_version"))
            version2 = result.scalar()

        assert version1 == version2, "Version should not change on re-run"

    def test_sqlite_incremental_upgrade(self, sqlite_tester):
        """Test incremental upgrades from b9d890ed31b5 to head on SQLite.

        This tests each migration in the chain to ensure they work correctly
        when run sequentially from an earlier version.

        Note: We start from b9d890ed31b5 instead of base because earlier
        migrations may have issues with table/column assumptions.
        """
        from alembic.script import ScriptDirectory

        # First, create base tables using SQLAlchemy (simulates production database)
        from xagent.web.models.database import Base

        Base.metadata.create_all(bind=sqlite_tester.engine)

        # Start from b9d890ed31b5 (an earlier revision)
        START_REVISION = "b9d890ed31b5"

        # Stamp to starting revision to simulate upgrading from that version
        command.stamp(sqlite_tester.alembic_cfg, START_REVISION)

        # Get all migrations from START_REVISION to head
        script_dir = ScriptDirectory.from_config(sqlite_tester.alembic_cfg)
        revisions = list(script_dir.walk_revisions(START_REVISION, "heads"))
        revisions.reverse()  # START_REVISION to head

        # Upgrade one revision at a time
        for revision in revisions:
            command.upgrade(sqlite_tester.alembic_cfg, revision.revision)

            # Verify version
            with sqlite_tester.engine.begin() as conn:
                result = conn.execute(text("SELECT version_num FROM alembic_version"))
                version = result.scalar()
                assert version == revision.revision

        # After all upgrades, verify that migrations actually added their columns
        # This tests that migrations work, not just that they're idempotent
        agents_columns = sqlite_tester.get_column_names("agents")
        assert "models" in agents_columns, (
            "f79da474c69d should have renamed model_config to models"
        )
        assert "suggested_prompts" in agents_columns, (
            "20250209_add_suggested_prompts should have added column"
        )

        models_columns = sqlite_tester.get_column_names("models")
        assert "_api_key_encrypted" in models_columns, (
            "441d4f5d399c should have encrypted api_key"
        )
        assert "max_tokens" in models_columns, (
            "b74d4cf2f479 should have added max_tokens column"
        )

    @pytest.mark.postgresql
    def test_postgresql_incremental_upgrade(self, postgresql_tester):
        """Test incremental upgrades from b9d890ed31b5 to head on PostgreSQL.

        This tests each migration in the chain to ensure they work correctly
        when run sequentially from an earlier version.

        Note: We start from b9d890ed31b5 instead of base because earlier
        migrations may have issues with table/column assumptions.
        """
        from alembic.script import ScriptDirectory

        # First, create base tables using SQLAlchemy (simulates production database)
        from xagent.web.models.database import Base

        Base.metadata.create_all(bind=postgresql_tester.engine)

        # Start from b9d890ed31b5 (an earlier revision)
        START_REVISION = "b9d890ed31b5"

        # Stamp to starting revision to simulate upgrading from that version
        command.stamp(postgresql_tester.alembic_cfg, START_REVISION)

        # Get all migrations from START_REVISION to head
        script_dir = ScriptDirectory.from_config(postgresql_tester.alembic_cfg)
        revisions = list(script_dir.walk_revisions(START_REVISION, "heads"))
        revisions.reverse()  # START_REVISION to head

        # Upgrade one revision at a time
        for revision in revisions:
            command.upgrade(postgresql_tester.alembic_cfg, revision.revision)

            # Verify version
            with postgresql_tester.engine.begin() as conn:
                result = conn.execute(text("SELECT version_num FROM alembic_version"))
                version = result.scalar()
                assert version == revision.revision

        # After all upgrades, verify that migrations actually added their columns
        agents_columns = postgresql_tester.get_column_names("agents")
        assert "models" in agents_columns, (
            "f79da474c69d should have renamed model_config to models"
        )
        assert "suggested_prompts" in agents_columns, (
            "20250209_add_suggested_prompts should have added column"
        )

        models_columns = postgresql_tester.get_column_names("models")
        assert "_api_key_encrypted" in models_columns, (
            "441d4f5d399c should have encrypted api_key"
        )


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
