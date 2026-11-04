import importlib.util
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory

from xagent.db.config import create_alembic_config


def _load_migration():
    path = (
        Path(__file__).parents[2]
        / "src/xagent/migrations/versions/20261016_add_email_confirmation.py"
    )
    spec = importlib.util.spec_from_file_location("email_confirmation_migration", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    return migration


def test_email_confirmation_migration_keeps_recovery_address():
    migration = _load_migration()
    engine = sa.create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT)"
        )
        connection.exec_driver_sql(
            "INSERT INTO users VALUES (1, 'original@example.com')"
        )
        migration.op = Operations(MigrationContext.configure(connection))
        migration.upgrade()
        migration.upgrade()
        assert connection.exec_driver_sql(
            "SELECT email, pending_email FROM users"
        ).one() == ("original@example.com", None)
        migration.downgrade()
        assert (
            connection.exec_driver_sql("SELECT email FROM users").scalar_one()
            == "original@example.com"
        )
        migration.downgrade()


def test_email_confirmation_migration_skips_schema_without_users():
    migration = _load_migration()
    engine = sa.create_engine("sqlite:///:memory:")
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "CREATE TABLE unrelated (id INTEGER PRIMARY KEY)"
            )
            migration.op = Operations(MigrationContext.configure(connection))
            migration.upgrade()
            migration.downgrade()
            assert sa.inspect(connection).get_table_names() == ["unrelated"]
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    "previous", ["20261016_email_confirmation", "20261016_session_planner_merge"]
)
def test_email_merge_upgrades_either_existing_branch(previous):
    engine = sa.create_engine("sqlite:///:memory:")
    try:
        config = create_alembic_config(engine)
        script = ScriptDirectory.from_config(config)
        assert len(script.get_heads()) == 1
        merge = script.get_revision("20261016_email_session_merge")
        assert set(merge.down_revision) == {
            "20261016_email_confirmation",
            "20261016_session_planner_merge",
        }
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT, session_epoch TEXT)"
            )
            connection.exec_driver_sql(
                "INSERT INTO users VALUES (1, 'original@example.com', 'revoked')"
            )
            expected_pending = None
            if previous == "20261016_email_confirmation":
                for column in (
                    "pending_email TEXT",
                    "email_confirmation_hash TEXT",
                    "email_confirmation_expires_at DATETIME",
                ):
                    connection.exec_driver_sql(f"ALTER TABLE users ADD COLUMN {column}")
                connection.exec_driver_sql(
                    "UPDATE users SET pending_email = 'pending@example.com'"
                )
                expected_pending = "pending@example.com"
            config.attributes["connection"] = connection
            command.stamp(config, previous)
            command.upgrade(config, "head")
            command.upgrade(config, "head")
            assert connection.exec_driver_sql(
                "SELECT email, session_epoch, pending_email FROM users"
            ).one() == ("original@example.com", "revoked", expected_pending)
    finally:
        engine.dispose()
