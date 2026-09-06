from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.migration import MigrationContext
from alembic.operations import Operations

from tests.shared.postgres_disposable import (
    disposable_database_factory,
    load_migration_module,
)
from xagent.db.config import create_alembic_config
from xagent.web.models.database import Base

REVISION = "20260905_task_execution_events"
PREVIOUS = "20260902_seed_magento_mcp_app"
MIGRATION = load_migration_module(
    Path(__file__).parents[2]
    / "src/xagent/migrations/versions/20260905_add_task_execution_events.py"
)


@pytest.fixture(
    params=["sqlite", pytest.param("postgresql", marks=pytest.mark.postgresql)]
)
def engine(request, tmp_path):
    if request.param == "postgresql":
        with disposable_database_factory("event_migration") as make:
            yield make("schema")
    else:
        result = sa.create_engine(f"sqlite:///{tmp_path / 'migration.db'}")

        @sa.event.listens_for(result, "connect")
        def foreign_keys(connection, _record):
            connection.execute("PRAGMA foreign_keys=ON")

        try:
            yield result
        finally:
            result.dispose()


def _assert_schema(connection):
    inspector = sa.inspect(connection)
    columns = {c["name"]: c for c in inspector.get_columns("tasks")}
    assert columns["conversation_storage_version"]["nullable"] is False
    assert columns["conversation_event_sequence"]["nullable"] is False
    assert {
        "ck_tasks_conversation_storage_version",
        "ck_tasks_conversation_event_sequence",
    } <= {c["name"] for c in inspector.get_check_constraints("tasks")}
    event_table = Base.metadata.tables["task_execution_events"]
    assert {c["name"] for c in inspector.get_columns(event_table.name)} == set(
        event_table.columns.keys()
    )
    assert {
        c.name for c in event_table.constraints if isinstance(c, sa.UniqueConstraint)
    } == {c["name"] for c in inspector.get_unique_constraints(event_table.name)}
    assert {
        c.name for c in event_table.constraints if isinstance(c, sa.CheckConstraint)
    } == {c["name"] for c in inspector.get_check_constraints(event_table.name)}
    assert {i.name for i in event_table.indexes} == {
        i["name"]
        for i in inspector.get_indexes(event_table.name)
        if not i.get("duplicates_constraint")
    }
    assert (
        inspector.get_foreign_keys(event_table.name)[0]["options"]["ondelete"]
        == "CASCADE"
    )
    if connection.dialect.name == "postgresql":
        payload = next(
            c for c in inspector.get_columns(event_table.name) if c["name"] == "payload"
        )
        assert isinstance(payload["type"], sa.dialects.postgresql.JSONB)


def test_upgrade_preserves_legacy_rows_and_round_trips(engine):
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "CREATE TABLE tasks (id INTEGER PRIMARY KEY, title VARCHAR(200) NOT NULL)"
            )
        )
        connection.execute(
            sa.text(
                "CREATE TABLE task_chat_messages (id INTEGER PRIMARY KEY, "
                "task_id INTEGER REFERENCES tasks(id), content TEXT NOT NULL)"
            )
        )
        connection.execute(sa.text("INSERT INTO tasks VALUES (1, 'existing')"))
        connection.execute(
            sa.text("INSERT INTO task_chat_messages VALUES (1, 1, 'hello')")
        )
    config = create_alembic_config(engine)
    with engine.connect() as connection:
        config.attributes["connection"] = connection
        command.stamp(config, PREVIOUS)
        connection.commit()
        command.upgrade(config, REVISION)
        connection.commit()
        _assert_schema(connection)
        assert connection.execute(
            sa.text(
                "SELECT id, title, conversation_storage_version, conversation_event_sequence FROM tasks"
            )
        ).all() == [(1, "existing", 1, 0)]
        assert connection.execute(
            sa.text("SELECT * FROM task_chat_messages")
        ).all() == [(1, 1, "hello")]
        connection.execute(
            sa.text("INSERT INTO tasks (id, title) VALUES (2, 'old writer')")
        )
        connection.commit()
        assert connection.execute(
            sa.text(
                "SELECT conversation_storage_version, conversation_event_sequence FROM tasks WHERE id = 2"
            )
        ).one() == (1, 0)
        connection.rollback()
        for statement in (
            "UPDATE tasks SET conversation_storage_version = 2 WHERE id = 1",
            "UPDATE tasks SET conversation_event_sequence = -1 WHERE id = 1",
        ):
            with pytest.raises(sa.exc.IntegrityError):
                connection.execute(sa.text(statement))
            connection.rollback()
        command.downgrade(config, PREVIOUS)
        connection.commit()
        assert not sa.inspect(connection).has_table("task_execution_events")
        assert connection.execute(sa.text("SELECT * FROM tasks ORDER BY id")).all() == [
            (1, "existing"),
            (2, "old writer"),
        ]
        assert connection.execute(
            sa.text("SELECT * FROM task_chat_messages")
        ).all() == [(1, 1, "hello")]
        connection.rollback()
        command.upgrade(config, REVISION)
        connection.commit()
        _assert_schema(connection)


def test_create_all_schema_and_upgrade_are_compatible(engine):
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        with Operations.context(MigrationContext.configure(connection)):
            MIGRATION.upgrade()
            MIGRATION.upgrade()
        _assert_schema(connection)


@pytest.mark.parametrize("dialect", ["sqlite", "postgresql"])
def test_offline_ddl_keeps_legacy_default_without_tasks_rebuild(dialect):
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name=dialect, opts={"as_sql": True, "output_buffer": output}
    )
    with Operations.context(context):
        MIGRATION.upgrade()
    sql = output.getvalue()
    assert "conversation_storage_version INTEGER DEFAULT 1 NOT NULL" in sql
    assert "CHECK (conversation_storage_version = 1)" in sql
    assert "CREATE TABLE task_execution_events" in sql
    assert "DROP TABLE tasks" not in sql
    if dialect == "sqlite":
        with sa.create_engine("sqlite://").connect() as connection:
            connection.exec_driver_sql("CREATE TABLE tasks (id INTEGER PRIMARY KEY)")
            connection.connection.driver_connection.executescript(sql)
            _assert_schema(connection)
