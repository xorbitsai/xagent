"""Shared input and reply-route migration is independently reversible."""

from pathlib import Path

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text

from tests.shared.postgres_disposable import (
    disposable_database_factory,
    load_migration_module,
)


@pytest.fixture(
    params=["sqlite", pytest.param("postgresql", marks=pytest.mark.postgresql)]
)
def migration_engine(request, tmp_path):
    if request.param == "postgresql":
        with disposable_database_factory("shared_foundations") as make_database:
            yield make_database("migration")
    else:
        engine = create_engine(f"sqlite:///{tmp_path / 'migration.db'}")
        try:
            yield engine
        finally:
            engine.dispose()


def test_shared_execution_migration_round_trip(migration_engine):
    migration = load_migration_module(
        Path("src/xagent/migrations/versions/20260912_shared_task_execution.py")
    )
    with migration_engine.begin() as connection:
        connection.execute(text("CREATE TABLE tasks (id INTEGER PRIMARY KEY)"))
        connection.execute(
            text("CREATE TABLE task_execution_commands (id INTEGER PRIMARY KEY)")
        )
        connection.execute(text("INSERT INTO task_execution_commands (id) VALUES (1)"))
        with Operations.context(MigrationContext.configure(connection)):
            migration.upgrade()
            migration.upgrade()
        inspector = inspect(connection)
        columns = {
            column["name"]
            for column in inspector.get_columns("task_execution_commands")
        }
        assert {"reply_host_id", "reply_origin"} <= columns
        assert connection.execute(
            text(
                "SELECT reply_host_id, reply_origin FROM task_execution_commands WHERE id = 1"
            )
        ).one() == (None, None)
        assert (
            inspector.get_foreign_keys("task_runtime_secrets")[0]["options"]["ondelete"]
            == "CASCADE"
        )
        assert any(
            constraint["column_names"] == ["task_id", "turn_id"]
            for constraint in inspector.get_unique_constraints("task_runtime_secrets")
        )
        with Operations.context(MigrationContext.configure(connection)):
            migration.downgrade()
            migration.downgrade()
        assert not inspect(connection).has_table("task_runtime_secrets")
        assert {
            column["name"]
            for column in inspect(connection).get_columns("task_execution_commands")
        } == {"id"}
        assert (
            connection.execute(
                text("SELECT id FROM task_execution_commands")
            ).scalar_one()
            == 1
        )


@pytest.mark.parametrize("parent_table", [None, "tasks", "task_execution_commands"])
def test_migration_tolerates_absent_metadata_tables(migration_engine, parent_table):
    migration = load_migration_module(
        Path("src/xagent/migrations/versions/20260912_shared_task_execution.py")
    )
    with migration_engine.begin() as connection:
        if parent_table == "tasks":
            connection.execute(text("CREATE TABLE tasks (id INTEGER PRIMARY KEY)"))
        elif parent_table == "task_execution_commands":
            connection.execute(
                text("CREATE TABLE task_execution_commands (id INTEGER PRIMARY KEY)")
            )
        original_tables = set(inspect(connection).get_table_names())
        with Operations.context(MigrationContext.configure(connection)):
            migration.upgrade()
            migration.upgrade()
        assert inspect(connection).has_table("task_runtime_secrets") == (
            parent_table == "tasks"
        )
        if parent_table == "task_execution_commands":
            assert {
                col["name"] for col in inspect(connection).get_columns(parent_table)
            } == {"id", "reply_host_id", "reply_origin"}
        with Operations.context(MigrationContext.configure(connection)):
            migration.downgrade()
            migration.downgrade()
        assert set(inspect(connection).get_table_names()) == original_tables
