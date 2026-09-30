import importlib

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

from tests.web.services.task_database_shared import engine as engine_fixture

engine = engine_fixture


def test_nullable_outcome_migration_preserves_legacy_rows(engine):
    migration = importlib.import_module(
        "xagent.migrations.versions.20260930_task_completion_outcome"
    )
    with (
        engine.begin() as connection,
        Operations.context(MigrationContext.configure(connection)),
    ):
        connection.execute(sa.text("CREATE TABLE tasks (id INTEGER PRIMARY KEY)"))
        connection.execute(sa.text("INSERT INTO tasks (id) VALUES (1)"))
        migration.upgrade()
        migration.upgrade()
        column = next(
            c
            for c in sa.inspect(connection).get_columns("tasks")
            if c["name"] == "completion_outcome"
        )
        assert column["nullable"] is True
        assert column["default"] is None
        assert (
            connection.execute(sa.text("SELECT completion_outcome FROM tasks")).scalar()
            is None
        )
        migration.downgrade()
        migration.downgrade()
        assert [c["name"] for c in sa.inspect(connection).get_columns("tasks")] == [
            "id"
        ]
