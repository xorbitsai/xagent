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


def test_outcome_downgrade_preserves_task_dependents_and_inline_checks(engine):
    migration = importlib.import_module(
        "xagent.migrations.versions.20260930_task_completion_outcome"
    )
    with (
        engine.begin() as connection,
        Operations.context(MigrationContext.configure(connection)),
    ):
        connection.execute(
            sa.text(
                "CREATE TABLE tasks (id INTEGER PRIMARY KEY, "
                "conversation_storage_version INTEGER DEFAULT 1 NOT NULL "
                "CONSTRAINT ck_tasks_conversation_storage_version "
                "CHECK (conversation_storage_version IN (1, 2)))"
            )
        )
        connection.execute(
            sa.text(
                "CREATE TABLE task_dependents (id INTEGER PRIMARY KEY, "
                "task_id INTEGER REFERENCES tasks(id) ON DELETE CASCADE)"
            )
        )
        migration.upgrade()
        connection.execute(
            sa.text("INSERT INTO tasks (id, completion_outcome) VALUES (1, 'partial')")
        )
        connection.execute(
            sa.text("INSERT INTO task_dependents (id, task_id) VALUES (2, 1)")
        )

        migration.downgrade()

        assert connection.execute(sa.text("SELECT id FROM tasks")).scalars().all() == [
            1
        ]
        assert connection.execute(
            sa.text("SELECT id, task_id FROM task_dependents")
        ).all() == [(2, 1)]
        # Older migrations drop the version column together with its inline CHECK.
        # Rebuilding tasks would turn that CHECK into a table-level constraint.
        connection.execute(
            sa.text("ALTER TABLE tasks DROP COLUMN conversation_storage_version")
        )
