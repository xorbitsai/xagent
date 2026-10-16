import importlib.util
from pathlib import Path

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations


def test_email_confirmation_migration_keeps_recovery_address():
    path = (
        Path(__file__).parents[2]
        / "src/xagent/migrations/versions/20261016_add_email_confirmation.py"
    )
    spec = importlib.util.spec_from_file_location("email_confirmation_migration", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
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
    path = (
        Path(__file__).parents[2]
        / "src/xagent/migrations/versions/20261016_add_email_confirmation.py"
    )
    spec = importlib.util.spec_from_file_location("email_confirmation_migration", path)
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
