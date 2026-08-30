import re
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text

from tests.shared.postgres_disposable import (
    disposable_database_factory,
    load_migration_module,
)
from xagent.db.config import create_alembic_config
from xagent.web.models.task_command_terminal_event import TaskCommandTerminalEvent

REVISION = "20260828_terminal_cmd_events"
DOWN_REVISION = "20260821_actor_oauth_flow_states"
TABLE = "task_command_terminal_events"
COMMAND_TABLE = "task_execution_commands"
MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20260828_add_task_command_terminal_events.py"
)
UNIQUE_CONSTRAINT_NAMES = {
    constraint.name
    for constraint in TaskCommandTerminalEvent.__table__.constraints
    if isinstance(constraint, sa.UniqueConstraint)
}
INDEX_NAMES = {index.name for index in TaskCommandTerminalEvent.__table__.indexes}
EXPECTED_FOREIGN_KEYS = {
    (
        element.parent.name,
        element.target_fullname.split(".", 1)[0],
        element.ondelete,
    )
    for constraint in TaskCommandTerminalEvent.__table__.foreign_key_constraints
    for element in constraint.elements
}
NON_NULL_COLUMNS = {
    column.name
    for column in TaskCommandTerminalEvent.__table__.columns
    if not column.nullable
}


def _offline_sql(migration, dialect_name: str, operation: str) -> str:
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name=dialect_name,
        opts={"as_sql": True, "output_buffer": output},
    )
    with Operations.context(context):
        getattr(migration, operation)()
    return output.getvalue()


@pytest.fixture
def postgresql_engine_factory():
    with disposable_database_factory("xagent_terminal_events") as make:
        yield make


def test_upgrade_adds_terminal_task_command_event_log() -> None:
    engine = create_engine("sqlite:///:memory:")
    config = create_alembic_config(engine)

    with engine.begin() as connection:
        connection.execute(
            text("CREATE TABLE alembic_version (version_num VARCHAR(255) NOT NULL)")
        )
        connection.execute(
            text("INSERT INTO alembic_version (version_num) VALUES (:revision)"),
            {"revision": DOWN_REVISION},
        )
        connection.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY)"))
        connection.execute(
            text(
                "CREATE TABLE tasks ("
                "id INTEGER PRIMARY KEY, state_version INTEGER NOT NULL DEFAULT 0)"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE task_execution_commands ("
                "id INTEGER PRIMARY KEY, task_id INTEGER NOT NULL, "
                "target_run_id VARCHAR(64))"
            )
        )
        connection.execute(
            text(
                "INSERT INTO task_execution_commands "
                "(id, task_id, target_run_id) VALUES (1, 1, 'legacy-run')"
            )
        )
        config.attributes["connection"] = connection

        command.upgrade(config, REVISION)

        inspector = inspect(connection)
        column_details = {
            column["name"]: column for column in inspector.get_columns(TABLE)
        }
        columns = set(column_details)
        indexes = {index["name"] for index in inspector.get_indexes(TABLE)}
        assert {
            "event_id",
            "task_command_id",
            "task_id",
            "task_run_id",
            "task_state_version",
            "command_id",
            "command_kind",
            "actor_user_id",
            "task_owner_user_id",
            "outcome_version",
            "outcome",
            "message_code",
            "resend_safe",
            "include_command_identity",
            "created_at",
        } <= columns
        assert indexes == INDEX_NAMES
        assert {
            constraint["name"] for constraint in inspector.get_unique_constraints(TABLE)
        } == UNIQUE_CONSTRAINT_NAMES
        assert {
            (
                foreign_key["constrained_columns"][0],
                foreign_key["referred_table"],
                foreign_key["options"].get("ondelete"),
            )
            for foreign_key in inspector.get_foreign_keys(TABLE)
        } == EXPECTED_FOREIGN_KEYS
        assert {
            name for name, details in column_details.items() if not details["nullable"]
        } == NON_NULL_COLUMNS
        for column_name in (
            "resend_safe",
            "include_command_identity",
            "created_at",
        ):
            assert column_details[column_name]["default"] is not None
        command_columns = {
            column["name"]
            for column in inspect(connection).get_columns("task_execution_commands")
        }
        assert "target_state_version" in command_columns
        legacy_version = connection.execute(
            text(
                "SELECT target_state_version FROM task_execution_commands WHERE id = 1"
            )
        ).scalar_one()
        assert legacy_version is None

        command.downgrade(config, DOWN_REVISION)
        assert (
            "task_command_terminal_events" not in inspect(connection).get_table_names()
        )
        command_columns = {
            column["name"]
            for column in inspect(connection).get_columns("task_execution_commands")
        }
        assert "target_state_version" not in command_columns


def test_upgrade_skips_without_command_table() -> None:
    engine = create_engine("sqlite:///:memory:")
    config = create_alembic_config(engine)

    with engine.begin() as connection:
        connection.execute(
            text("CREATE TABLE alembic_version (version_num VARCHAR(255) NOT NULL)")
        )
        connection.execute(
            text("INSERT INTO alembic_version (version_num) VALUES (:revision)"),
            {"revision": DOWN_REVISION},
        )
        connection.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY)"))
        connection.execute(text("CREATE TABLE tasks (id INTEGER PRIMARY KEY)"))
        config.attributes["connection"] = connection

        command.upgrade(config, REVISION)

        tables = set(inspect(connection).get_table_names())
        assert TABLE not in tables
        assert COMMAND_TABLE not in tables
        assert (
            connection.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalar_one()
            == REVISION
        )


@pytest.mark.parametrize("dialect_name", ["sqlite", "postgresql"])
def test_offline_upgrade_emits_terminal_event_schema(dialect_name: str) -> None:
    migration = load_migration_module(
        MIGRATION_PATH, f"terminal_command_events_offline_upgrade_{dialect_name}"
    )

    with patch.object(
        migration.sa,
        "inspect",
        side_effect=AssertionError("offline branch must not reflect"),
    ):
        sql = _offline_sql(migration, dialect_name, "upgrade")

    assert "ALTER TABLE task_execution_commands ADD COLUMN target_state_version" in sql
    assert f"CREATE TABLE {TABLE}" in sql
    for name in UNIQUE_CONSTRAINT_NAMES:
        assert f"CONSTRAINT {name} UNIQUE" in sql
    assert sql.count("FOREIGN KEY(") == len(EXPECTED_FOREIGN_KEYS)
    assert sql.count("ON DELETE CASCADE") == 3
    assert sql.count("ON DELETE SET NULL") == 1
    for name in NON_NULL_COLUMNS - {"id"}:
        assert re.search(rf"^\s*{re.escape(name)}\s+.*NOT NULL", sql, re.MULTILINE)
    for name in ("resend_safe", "include_command_identity", "created_at"):
        assert re.search(
            rf"^\s*{name}\s+.*DEFAULT.*NOT NULL",
            sql,
            re.MULTILINE,
        )
    for name in INDEX_NAMES:
        assert f"CREATE INDEX {name} ON {TABLE}" in sql
    assert "%(" not in sql


@pytest.mark.parametrize("dialect_name", ["sqlite", "postgresql"])
def test_offline_downgrade_emits_terminal_event_cleanup(dialect_name: str) -> None:
    migration = load_migration_module(
        MIGRATION_PATH, f"terminal_command_events_offline_downgrade_{dialect_name}"
    )

    with patch.object(
        migration.sa,
        "inspect",
        side_effect=AssertionError("offline branch must not reflect"),
    ):
        sql = _offline_sql(migration, dialect_name, "downgrade")

    assert "DROP TABLE task_command_terminal_events" in sql
    assert "ALTER TABLE task_execution_commands DROP COLUMN target_state_version" in sql
    assert "%(" not in sql


@pytest.mark.postgresql
def test_postgresql_upgrade_targets_visible_schema_and_preserves_legacy_version(
    postgresql_engine_factory,
) -> None:
    migration = load_migration_module(
        MIGRATION_PATH, "terminal_command_events_migration"
    )
    engine = postgresql_engine_factory("upgrade")

    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY)"))
        connection.execute(
            text(
                "CREATE TABLE tasks ("
                "id INTEGER PRIMARY KEY, state_version INTEGER NOT NULL DEFAULT 0)"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE task_execution_commands ("
                "id INTEGER PRIMARY KEY, task_id INTEGER NOT NULL, "
                "target_run_id VARCHAR(64))"
            )
        )
        connection.execute(
            text(
                "INSERT INTO task_execution_commands "
                "(id, task_id, target_run_id) VALUES (1, 1, 'legacy-run')"
            )
        )
        connection.execute(text("CREATE SCHEMA app"))
        for parent in ("users", "tasks", "task_execution_commands"):
            connection.execute(
                text(f"CREATE TABLE app.{parent} (LIKE public.{parent} INCLUDING ALL)")
            )
        connection.execute(
            text(
                "INSERT INTO app.task_execution_commands "
                "(id, task_id, target_run_id) VALUES (1, 1, 'legacy-run')"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE public.task_command_terminal_events "
                "(decoy_marker INTEGER PRIMARY KEY)"
            )
        )
        connection.execute(text("SET search_path TO app, public"))
        context = MigrationContext.configure(connection)
        with Operations.context(context):
            migration.upgrade()

        inspector = sa.inspect(connection)
        assert "task_command_terminal_events" in inspector.get_table_names(schema="app")
        assert {
            column["name"]
            for column in inspector.get_columns(
                "task_command_terminal_events", schema="public"
            )
        } == {"decoy_marker"}
        legacy_version = connection.execute(
            text(
                "SELECT target_state_version FROM task_execution_commands WHERE id = 1"
            )
        ).scalar_one()
        assert legacy_version is None

        with Operations.context(context):
            migration.downgrade()
        assert "task_command_terminal_events" not in sa.inspect(
            connection
        ).get_table_names(schema="app")
