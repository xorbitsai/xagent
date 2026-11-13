import importlib.util
from pathlib import Path

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations


def _migration():
    path = (
        Path(__file__).parents[2]
        / "src/xagent/migrations/versions/20261113_browser_idle_session.py"
    )
    spec = importlib.util.spec_from_file_location("browser_idle_migration", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    return migration


def test_browser_session_migration_is_idempotent_and_keeps_users():
    migration = _migration()
    engine = sa.create_engine("sqlite:///:memory:")
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT, "
                "refresh_token TEXT)"
            )
            connection.exec_driver_sql(
                "INSERT INTO users VALUES (1, 'existing-user', 'digest')"
            )
            migration.op = Operations(MigrationContext.configure(connection))
            migration.upgrade()
            migration.upgrade()
            inspector = sa.inspect(connection)
            assert "browser_sessions" in inspector.get_table_names()
            # Existing refresh tokens start unattached: the readiness query
            # counts them until a refresh moves them onto a browser session.
            assert connection.exec_driver_sql(
                "SELECT username, refresh_token, browser_session_id FROM users"
            ).one() == ("existing-user", "digest", None)

            migration.downgrade()
            migration.downgrade()
            inspector = sa.inspect(connection)
            assert "browser_sessions" not in inspector.get_table_names()
            assert "browser_session_id" not in {
                c["name"] for c in inspector.get_columns("users")
            }
            assert (
                connection.exec_driver_sql("SELECT username FROM users").scalar_one()
                == "existing-user"
            )
    finally:
        engine.dispose()


def test_browser_session_migration_skips_schema_without_users():
    migration = _migration()
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
