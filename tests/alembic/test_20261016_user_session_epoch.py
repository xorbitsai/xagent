import importlib.util
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory

from xagent.db.config import create_alembic_config


def test_session_generation_migration_preserves_existing_users():
    path = (
        Path(__file__).parents[2]
        / "src/xagent/migrations/versions/20261016_add_user_session_epoch.py"
    )
    spec = importlib.util.spec_from_file_location("session_epoch_migration", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    engine = sa.create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT)"
        )
        connection.exec_driver_sql("INSERT INTO users VALUES (1, 'existing-user')")
        migration.op = Operations(MigrationContext.configure(connection))
        migration.upgrade()
        migration.upgrade()
        assert connection.exec_driver_sql(
            "SELECT username, session_epoch FROM users"
        ).one() == ("existing-user", None)
        migration.downgrade()
        assert (
            connection.exec_driver_sql("SELECT username FROM users").scalar_one()
            == "existing-user"
        )
        migration.downgrade()


def test_session_generation_migration_skips_schema_without_users():
    path = (
        Path(__file__).parents[2]
        / "src/xagent/migrations/versions/20261016_add_user_session_epoch.py"
    )
    spec = importlib.util.spec_from_file_location("session_epoch_migration", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
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
    "previous", ["20261016_user_session_epoch", "20261015_planner_team_read_scope"]
)
def test_session_merge_upgrades_either_existing_branch(previous):
    engine = sa.create_engine("sqlite:///:memory:")
    try:
        config = create_alembic_config(engine)
        script = ScriptDirectory.from_config(config)
        assert len(script.get_heads()) == 1
        merge = script.get_revision("20261016_session_planner_merge")
        assert set(merge.down_revision) == {
            "20261016_user_session_epoch",
            "20261015_planner_team_read_scope",
        }
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT)"
            )
            connection.exec_driver_sql("INSERT INTO users VALUES (1, 'existing')")
            expected_epoch = None
            if previous == "20261016_user_session_epoch":
                connection.exec_driver_sql(
                    "ALTER TABLE users ADD COLUMN session_epoch TEXT"
                )
                connection.exec_driver_sql(
                    "UPDATE users SET session_epoch = 'already-revoked'"
                )
                expected_epoch = "already-revoked"
            config.attributes["connection"] = connection
            command.stamp(config, previous)
            command.upgrade(config, "head")
            command.upgrade(config, "head")
            assert connection.exec_driver_sql(
                "SELECT username, session_epoch FROM users"
            ).one() == ("existing", expected_epoch)
    finally:
        engine.dispose()
