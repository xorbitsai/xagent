from alembic import command
from sqlalchemy import create_engine, inspect, text

from xagent.db.config import create_alembic_config

REVISION = "20260711_task_commands"
DOWN_REVISION = "20260711_add_trace_events_task_idx"


def test_upgrade_adds_durable_task_command_inbox() -> None:
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

        columns = {
            column["name"]
            for column in inspect(connection).get_columns("task_execution_commands")
        }
        indexes = {
            index["name"]
            for index in inspect(connection).get_indexes("task_execution_commands")
        }
        assert {
            "command_id",
            "kind",
            "payload",
            "target_run_id",
            "target_runner_id",
            "status",
            "claimed_by",
            "claim_expires_at",
            "attempt_count",
            "failure_count",
            "defer_count",
            "result",
            "error",
            "completed_at",
        } <= columns
        assert "ix_task_commands_status_created" in indexes
        assert "ix_task_commands_task_order" in indexes
        assert "ix_task_execution_commands_task_id" not in indexes

        command.downgrade(config, DOWN_REVISION)
        assert "task_execution_commands" not in inspect(connection).get_table_names()


def test_upgrade_skips_when_sqlalchemy_base_tables_do_not_exist() -> None:
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
        config.attributes["connection"] = connection

        command.upgrade(config, REVISION)

        assert "task_execution_commands" not in inspect(connection).get_table_names()
