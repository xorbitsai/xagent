import importlib.util
from pathlib import Path

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations


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
