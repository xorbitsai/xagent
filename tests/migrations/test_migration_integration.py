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
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import sessionmaker

# Project root directory
project_root = Path(__file__).parent.parent.parent


def _revision_reached(
    script_dir: ScriptDirectory, target_revision: str, current_revisions: set[str]
) -> bool:
    """Return whether target_revision is present in or behind current heads."""
    pending = list(current_revisions)
    visited: set[str] = set()

    while pending:
        revision_id = pending.pop()
        if revision_id == target_revision:
            return True
        if revision_id in visited:
            continue

        visited.add(revision_id)
        revision = script_dir.get_revision(revision_id)

        down_revisions = revision.down_revision
        if down_revisions is None:
            continue
        if isinstance(down_revisions, str):
            pending.append(down_revisions)
        else:
            pending.extend(down_revisions)

    return False


class MigrationTester:
    """Helper class to test database migrations."""

    def __init__(self, db_type: str):
        self.db_type = db_type
        self.engine = None
        self.alembic_cfg = None
        self.temp_db_file = None
        self._old_database_url: str | None = None

    def setup_database(self):
        """Set up test database connection."""
        self._old_database_url = os.environ.get("DATABASE_URL")
        if self.db_type == "sqlite":
            # Use temporary file for SQLite
            fd, path = tempfile.mkstemp(suffix=".db")
            os.close(fd)
            db_url = f"sqlite:///{path}"
            self.temp_db_file = path
        elif self.db_type == "postgresql":
            configured_postgres_url = os.getenv("POSTGRES_TEST_DATABASE_URL")
            if not configured_postgres_url:
                database_url = os.getenv("DATABASE_URL")
                if database_url and database_url.startswith("postgresql"):
                    configured_postgres_url = database_url

            db_url = configured_postgres_url or os.getenv(
                "XAGENT_TEST_POSTGRES_URL",
                "postgresql://xagent:xagent@localhost:5432/xagent_test",
            )

        os.environ["DATABASE_URL"] = db_url
        try:
            self.engine = create_engine(db_url)

            # Clean database for PostgreSQL
            if self.db_type == "postgresql":
                with self.engine.begin() as conn:
                    conn.execute(text("DROP SCHEMA public CASCADE"))
                    conn.execute(text("CREATE SCHEMA public"))
        except SQLAlchemyError as exc:
            if self.db_type == "postgresql":
                self._restore_database_url()
                pytest.skip(f"PostgreSQL test database unavailable: {exc}")
            raise

        # Configure Alembic
        self.alembic_cfg = Config(str(project_root / "alembic.ini"))
        self.alembic_cfg.set_main_option("sqlalchemy.url", db_url)

        # DO NOT create tables with SQLAlchemy - we want to test migrations
        # can build the schema from scratch

    def _restore_database_url(self) -> None:
        """Restore ``DATABASE_URL`` to its pre-test value."""
        if self._old_database_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = self._old_database_url

    def teardown_database(self):
        """Clean up test database."""
        try:
            if self.db_type == "sqlite" and self.temp_db_file:
                os.unlink(self.temp_db_file)
            elif self.db_type == "postgresql" and self.engine is not None:
                with self.engine.begin() as conn:
                    conn.execute(text("DROP SCHEMA public CASCADE"))
                    conn.execute(text("CREATE SCHEMA public"))
        finally:
            self._restore_database_url()

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

    def get_alembic_versions(self) -> set[str]:
        """Get current Alembic version rows."""
        with self.engine.begin() as conn:
            result = conn.execute(text("SELECT version_num FROM alembic_version"))
            return {row[0] for row in result}


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

    def test_sqlite_stamp_head_creates_wide_alembic_version_table(self, sqlite_tester):
        """Stamping a fresh database must support long revision IDs."""
        command.stamp(sqlite_tester.alembic_cfg, "head")

        columns = inspect(sqlite_tester.engine).get_columns("alembic_version")
        version_num = next(col for col in columns if col["name"] == "version_num")

        assert version_num["type"].length == 255

    @pytest.mark.postgresql
    def test_postgresql_upgrade(self, postgresql_tester):
        """Test a full PostgreSQL migration upgrade from an empty database.

        Historical migrations intentionally leave core metadata-owned tables,
        such as ``models`` and ``users``, to SQLAlchemy. This test therefore
        verifies only tables and columns that the migration chain owns.
        """
        # Run the complete migration chain without pre-creating ORM metadata.
        command.upgrade(postgresql_tester.alembic_cfg, "head")

        # Verify alembic_version table
        with postgresql_tester.engine.begin() as conn:
            result = conn.execute(text("SELECT version_num FROM alembic_version"))
            version = result.scalar()
            assert version is not None, "Version should be set after upgrade"

        # Verify migration-owned tables exist. Core tables created by current
        # SQLAlchemy metadata are intentionally outside this test's contract.
        tables = postgresql_tester.get_table_names()
        assert "agents" in tables, "agents table should exist"
        assert "alembic_version" in tables, "alembic_version table should exist"
        assert "public_mcp_apps" in tables, "public_mcp_apps table should exist"
        assert "user_oauth" in tables, "user_oauth table should exist"

        # Verify agents table structure.
        columns = postgresql_tester.get_column_names("agents")
        assert "models" in columns, "models column should exist"
        assert "name" in columns, "name column should exist"
        assert "execution_mode" in columns, "execution_mode column should exist"
        assert "status" in columns, "status column should exist"
        assert "suggested_prompts" in columns, "suggested_prompts column should exist"

        user_oauth_columns = postgresql_tester.get_column_names("user_oauth")
        assert "resource_owner_key" in user_oauth_columns, (
            "owner-aware OAuth column should exist"
        )

    @pytest.mark.postgresql
    def test_postgresql_owner_index_collision_allows_retry_after_remediation(
        self, postgresql_tester
    ):
        """After operator remediation, retry starts from the intact old schema."""
        down_revision = "20260819_merge_jira_and_linear_heads"
        ordinary_index = "uq_user_oauth_ordinary_account"
        actor_index = "uq_user_oauth_actor_account"
        old_constraint = "uq_user_provider_account"

        command.upgrade(postgresql_tester.alembic_cfg, down_revision)
        with postgresql_tester.engine.begin() as conn:
            conn.execute(text(f"CREATE INDEX {actor_index} ON user_oauth (user_id)"))

        with pytest.raises(SQLAlchemyError):
            command.upgrade(postgresql_tester.alembic_cfg, "head")

        inspector = inspect(postgresql_tester.engine)
        assert "resource_owner_key" not in {
            column["name"] for column in inspector.get_columns("user_oauth")
        }
        assert old_constraint in {
            constraint["name"]
            for constraint in inspector.get_unique_constraints("user_oauth")
        }
        indexes = {index["name"] for index in inspector.get_indexes("user_oauth")}
        assert actor_index in indexes
        assert ordinary_index not in indexes
        script_dir = ScriptDirectory.from_config(postgresql_tester.alembic_cfg)
        current_revisions = postgresql_tester.get_alembic_versions()
        assert _revision_reached(script_dir, down_revision, current_revisions)
        assert not _revision_reached(
            script_dir,
            "20260818_user_oauth_resource_owner",
            current_revisions,
        )

        with postgresql_tester.engine.begin() as conn:
            conn.execute(text(f"DROP INDEX {actor_index}"))
        command.upgrade(postgresql_tester.alembic_cfg, "head")

        assert "resource_owner_key" in postgresql_tester.get_column_names("user_oauth")

    @pytest.mark.postgresql
    def test_postgresql_owner_indexes_enforce_namespace_boundaries(
        self, postgresql_tester
    ):
        """The migrated PostgreSQL predicates separate ordinary and actor rows."""
        parent = "20260819_merge_jira_and_linear_heads"
        owner_revision = "20260818_user_oauth_resource_owner"
        command.upgrade(postgresql_tester.alembic_cfg, parent)
        with postgresql_tester.engine.begin() as conn:
            conn.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY)"))
            conn.execute(text("INSERT INTO users (id) VALUES (7101)"))

        command.upgrade(postgresql_tester.alembic_cfg, owner_revision)
        with postgresql_tester.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO user_oauth "
                    "(user_id, provider, provider_user_id, access_token, "
                    "resource_owner_key) VALUES "
                    "(7101, 'gmail', 'shared', 'ordinary', NULL), "
                    "(7101, 'gmail', 'shared', 'actor', 'actor:alice')"
                )
            )

        with pytest.raises(IntegrityError):
            with postgresql_tester.engine.begin() as conn:
                conn.execute(
                    text(
                        "INSERT INTO user_oauth "
                        "(user_id, provider, provider_user_id, access_token) "
                        "VALUES (7101, 'gmail', 'shared', 'duplicate-ordinary')"
                    )
                )

        with pytest.raises(IntegrityError):
            with postgresql_tester.engine.begin() as conn:
                conn.execute(
                    text(
                        "INSERT INTO user_oauth "
                        "(user_id, provider, provider_user_id, access_token, "
                        "resource_owner_key) VALUES "
                        "(7101, 'gmail', 'shared', 'duplicate-actor', 'actor:alice')"
                    )
                )

    @pytest.mark.postgresql
    def test_postgresql_user_delete_cascades_actor_owned_oauth_rows(
        self, postgresql_tester
    ):
        """Filtered ordinary relationship leaves actor cleanup to the FK."""
        from xagent.web.models.database import Base
        from xagent.web.models.user import User
        from xagent.web.models.user_oauth import UserOAuth

        Base.metadata.create_all(bind=postgresql_tester.engine)
        db = sessionmaker(
            bind=postgresql_tester.engine,
            autoflush=False,
            autocommit=False,
        )()
        try:
            user = User(username="oauth-cascade-owner", password_hash="hash")
            db.add(user)
            db.commit()
            db.refresh(user)
            db.add_all(
                [
                    UserOAuth(
                        user_id=int(user.id),
                        provider="gmail",
                        provider_user_id="ordinary",
                        access_token="ordinary-token",
                    ),
                    UserOAuth(
                        user_id=int(user.id),
                        provider="gmail",
                        resource_owner_key="actor:alice",
                        provider_user_id="alice",
                        access_token="actor-token",
                    ),
                ]
            )
            db.commit()
            db.expire(user, ["oauth_accounts"])

            assert [row.resource_owner_key for row in user.oauth_accounts] == [None]

            db.delete(user)
            db.commit()

            assert db.query(UserOAuth).count() == 0
        finally:
            db.close()

    @pytest.mark.postgresql
    def test_postgresql_owner_migration_downgrade_restores_legacy_schema(
        self, postgresql_tester
    ):
        """The supported PostgreSQL downgrade restores the old identity shape."""
        parent = "20260819_merge_jira_and_linear_heads"
        owner_revision = "20260818_user_oauth_resource_owner"
        command.upgrade(postgresql_tester.alembic_cfg, parent)
        with postgresql_tester.engine.begin() as conn:
            # The historical migration chain creates user_oauth from scratch
            # but assumes the core users table already exists. Complete that
            # legacy shape explicitly before exercising the owner revision.
            conn.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY)"))
            conn.execute(
                text(
                    "ALTER TABLE user_oauth ADD CONSTRAINT "
                    "fk_user_oauth_user_id_users FOREIGN KEY (user_id) "
                    "REFERENCES users(id) ON DELETE CASCADE"
                )
            )
            conn.execute(text("INSERT INTO users (id) VALUES (7001)"))
            conn.execute(
                text(
                    "INSERT INTO user_oauth "
                    "(id, user_id, provider, access_token, provider_user_id) "
                    "VALUES (9001, 7001, 'gmail', 'token', 'provider-account')"
                )
            )

        command.upgrade(postgresql_tester.alembic_cfg, owner_revision)
        command.downgrade(postgresql_tester.alembic_cfg, parent)

        inspector = inspect(postgresql_tester.engine)
        assert "resource_owner_key" not in {
            column["name"] for column in inspector.get_columns("user_oauth")
        }
        assert "uq_user_provider_account" in {
            constraint["name"]
            for constraint in inspector.get_unique_constraints("user_oauth")
        }
        assert not {
            "uq_user_oauth_ordinary_account",
            "uq_user_oauth_actor_account",
        } & {index["name"] for index in inspector.get_indexes("user_oauth")}
        with postgresql_tester.engine.begin() as conn:
            assert conn.execute(
                text("SELECT provider, access_token FROM user_oauth WHERE id = 9001")
            ).one() == ("gmail", "token")

    def test_sqlite_owner_downgrade_restores_one_known_head(self, sqlite_tester):
        """Rollback removes owner storage and retains the existing merge head."""
        parent = "20260819_merge_jira_and_linear_heads"

        command.upgrade(sqlite_tester.alembic_cfg, "head")
        command.downgrade(sqlite_tester.alembic_cfg, parent)

        assert sqlite_tester.get_alembic_versions() == {parent}
        assert "resource_owner_key" not in sqlite_tester.get_column_names("user_oauth")

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
    def test_postgresql_owner_migration_accepts_current_metadata(
        self, postgresql_tester
    ):
        """The owner revision accepts its schema from current metadata."""
        from xagent.web.models.database import Base

        Base.metadata.create_all(bind=postgresql_tester.engine)
        # Older migrations are not all compatible with current pre-created
        # foreign keys. Stamp the immediate parent so this test isolates the
        # owner migration's fresh-schema path.
        command.stamp(
            postgresql_tester.alembic_cfg,
            "20260819_merge_jira_and_linear_heads",
        )

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
        assert revisions, "Expected migrations from START_REVISION to heads"
        revisions.reverse()  # START_REVISION to head

        # Upgrade one revision at a time
        for revision in revisions:
            command.upgrade(sqlite_tester.alembic_cfg, revision.revision)

            # Verify version
            current_revisions = sqlite_tester.get_alembic_versions()
            assert _revision_reached(
                script_dir, revision.revision, current_revisions
            ), (
                f"{revision.revision} should be present in or behind current "
                f"Alembic heads {sorted(current_revisions)}"
            )

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
        assert revisions, "Expected migrations from START_REVISION to heads"
        revisions.reverse()  # START_REVISION to head

        # Upgrade one revision at a time
        for revision in revisions:
            command.upgrade(postgresql_tester.alembic_cfg, revision.revision)

            # Verify version
            current_revisions = postgresql_tester.get_alembic_versions()
            assert _revision_reached(
                script_dir, revision.revision, current_revisions
            ), (
                f"{revision.revision} should be present in or behind current "
                f"Alembic heads {sorted(current_revisions)}"
            )

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
    except pytest.skip.Exception as e:
        print(f"{args.db.upper()} test SKIPPED: {e.msg}")
