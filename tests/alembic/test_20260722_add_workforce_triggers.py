"""Tests for the agent_triggers workforce-support migration (#950)."""

import importlib.util
from pathlib import Path

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20260722_add_workforce_triggers.py"
)
TABLE = "agent_triggers"
COLUMN = "workforce_id"


def _load_migration_module():
    spec = importlib.util.spec_from_file_location(
        "add_workforce_triggers_migration", MIGRATION_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection) -> Operations:
    return Operations(MigrationContext.configure(connection))


def _create_legacy_schema(connection) -> None:
    connection.exec_driver_sql(
        "CREATE TABLE agents (id INTEGER PRIMARY KEY, origin VARCHAR(50))"
    )
    connection.exec_driver_sql("CREATE TABLE workforces (id INTEGER PRIMARY KEY)")
    connection.exec_driver_sql(
        f"""
        CREATE TABLE {TABLE} (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            agent_id INTEGER NOT NULL REFERENCES agents(id),
            type VARCHAR(32) NOT NULL,
            name VARCHAR(200) NOT NULL,
            enabled BOOLEAN NOT NULL DEFAULT 1,
            config JSON NOT NULL
        )
        """
    )
    connection.execute(sa.text("INSERT INTO agents (id, origin) VALUES (1, 'user')"))
    connection.execute(
        sa.text(
            "INSERT INTO agents (id, origin) VALUES (2, 'workforce_generated_manager')"
        )
    )
    connection.execute(
        sa.text(
            f"INSERT INTO {TABLE} (id, user_id, agent_id, type, name, enabled, config)"
            " VALUES (1, 1, 1, 'webhook', 'Normal trigger', 1, '{}')"
        )
    )
    connection.execute(
        sa.text(
            f"INSERT INTO {TABLE} (id, user_id, agent_id, type, name, enabled, config)"
            " VALUES (2, 1, 2, 'webhook', 'Orphaned manager trigger', 1, '{}')"
        )
    )


def _columns(connection) -> dict[str, dict]:
    return {
        column["name"]: column for column in sa.inspect(connection).get_columns(TABLE)
    }


def test_upgrade_adds_column_relaxes_agent_id_and_disables_manager_triggers() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        _create_legacy_schema(connection)
        with Operations.context(_operations(connection).get_context()):
            migration.upgrade()

        columns = _columns(connection)
        assert COLUMN in columns
        assert columns[COLUMN]["nullable"] is True
        assert columns["agent_id"]["nullable"] is True

        index_names = {
            index["name"] for index in sa.inspect(connection).get_indexes(TABLE)
        }
        assert "ix_agent_triggers_workforce_id" in index_names

        # The exactly-one-owner CHECK constraint rejects both-null / both-set.
        import pytest

        with pytest.raises(Exception):
            connection.execute(
                sa.text(
                    f"INSERT INTO {TABLE} "
                    "(id, user_id, agent_id, workforce_id, type, name, enabled, config)"
                    " VALUES (90, 1, NULL, NULL, 'webhook', 'no owner', 1, '{}')"
                )
            )
        with pytest.raises(Exception):
            connection.execute(
                sa.text(
                    f"INSERT INTO {TABLE} "
                    "(id, user_id, agent_id, workforce_id, type, name, enabled, config)"
                    " VALUES (91, 1, 1, 1, 'webhook', 'two owners', 1, '{}')"
                )
            )

        rows = connection.execute(
            sa.text(f"SELECT id, enabled FROM {TABLE} ORDER BY id")
        ).fetchall()
        # Trigger 1 (normal agent) stays enabled; trigger 2 (bound to a
        # workforce-generated manager agent) is disabled by the cleanup.
        assert [(row[0], bool(row[1])) for row in rows] == [(1, True), (2, False)]


def test_upgrade_is_idempotent() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        _create_legacy_schema(connection)
        with Operations.context(_operations(connection).get_context()):
            migration.upgrade()
            migration.upgrade()

        columns = _columns(connection)
        assert COLUMN in columns
        assert columns["agent_id"]["nullable"] is True


def test_downgrade_drops_workforce_rows_and_restores_agent_id_not_null() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        _create_legacy_schema(connection)
        operations = _operations(connection)
        with Operations.context(operations.get_context()):
            migration.upgrade()

        connection.execute(sa.text("INSERT INTO workforces (id) VALUES (7)"))
        connection.execute(
            sa.text(
                f"INSERT INTO {TABLE} "
                "(id, user_id, agent_id, workforce_id, type, name, enabled, config) "
                "VALUES (3, 1, NULL, 7, 'webhook', 'Workforce trigger', 1, '{}')"
            )
        )

        with Operations.context(operations.get_context()):
            migration.downgrade()

        columns = _columns(connection)
        assert COLUMN not in columns
        assert columns["agent_id"]["nullable"] is False
        remaining = connection.execute(
            sa.text(f"SELECT id FROM {TABLE} ORDER BY id")
        ).fetchall()
        assert [row[0] for row in remaining] == [1, 2]


def test_upgrade_and_downgrade_skip_when_table_missing() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        with Operations.context(_operations(connection).get_context()):
            migration.upgrade()
            migration.downgrade()

        assert TABLE not in sa.inspect(connection).get_table_names()
