from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

from tests.migrations.test_20260905_add_task_execution_events import (
    MIGRATION as STORAGE,
)
from tests.migrations.test_20260905_add_task_execution_events import (
    engine as engine_fixture,
)
from tests.shared.postgres_disposable import load_migration_module

engine = engine_fixture

WRITERS = load_migration_module(
    Path(__file__).parents[2]
    / "src/xagent/migrations/versions/20260905_enable_task_execution_event_writers.py"
)


def run(connection, migration, operation):
    with Operations.context(MigrationContext.configure(connection)):
        getattr(migration, operation)()


def test_upgrade_preserves_tasks_and_inbound_cascade_rows(engine):
    with engine.begin() as db:
        db.execute(sa.text("CREATE TABLE tasks (id INTEGER PRIMARY KEY)"))
        db.execute(
            sa.text(
                "CREATE TABLE task_chat_messages (id INTEGER PRIMARY KEY, task_id INTEGER REFERENCES tasks(id) ON DELETE CASCADE)"
            )
        )
        db.execute(sa.text("INSERT INTO tasks VALUES (1)"))
        db.execute(sa.text("INSERT INTO task_chat_messages VALUES (1, 1)"))
        run(db, STORAGE, "upgrade")
        run(db, WRITERS, "upgrade")
        assert db.scalar(sa.text("SELECT count(*) FROM task_chat_messages")) == 1
        assert db.scalar(sa.text("SELECT conversation_storage_version FROM tasks")) == 1
        db.execute(
            sa.text("INSERT INTO tasks(id, conversation_storage_version) VALUES (2, 2)")
        )
        # create_all parity / idempotent re-entry must not reset canonical tasks.
        run(db, WRITERS, "upgrade")
        assert (
            db.scalar(
                sa.text("SELECT conversation_storage_version FROM tasks WHERE id=2")
            )
            == 2
        )
        with pytest.raises(RuntimeError, match="event-backed"):
            run(db, WRITERS, "downgrade")
        db.execute(sa.text("DELETE FROM tasks WHERE id=2"))
        run(db, WRITERS, "downgrade")
        assert db.scalar(sa.text("SELECT count(*) FROM task_chat_messages")) == 1
        run(db, WRITERS, "upgrade")
        db.execute(sa.text("INSERT INTO tasks(id) VALUES (3)"))
        assert (
            db.scalar(
                sa.text("SELECT conversation_storage_version FROM tasks WHERE id=3")
            )
            == 1
        )


@pytest.mark.parametrize("with_chat_table", [False, True])
def test_upgrade_downgrade_without_metadata_owned_tasks(engine, with_chat_table):
    with engine.begin() as db:
        if with_chat_table:
            db.execute(
                sa.text("CREATE TABLE task_chat_messages (id INTEGER PRIMARY KEY)")
            )
        run(db, STORAGE, "upgrade")
        run(db, WRITERS, "upgrade")
        if with_chat_table:
            assert "execution_event_id" in {
                column["name"]
                for column in sa.inspect(db).get_columns("task_chat_messages")
            }
        run(db, WRITERS, "downgrade")
        run(db, STORAGE, "downgrade")
        assert not sa.inspect(db).has_table("tasks")
        if with_chat_table:
            assert [
                column["name"]
                for column in sa.inspect(db).get_columns("task_chat_messages")
            ] == ["id"]
