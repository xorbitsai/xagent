from alembic import command
from sqlalchemy import create_engine, inspect, text

from xagent.db.config import create_alembic_config

REVISION = "20260711_task_control_state"
DOWN_REVISION = "20260710_add_chat_delivery_state"


def test_upgrade_adds_and_backfills_task_control_state() -> None:
    engine = create_engine("sqlite:///:memory:")
    config = create_alembic_config(engine)

    with engine.begin() as connection:
        connection.execute(
            text("CREATE TABLE alembic_version (version_num VARCHAR(255) NOT NULL)")
        )
        connection.execute(
            text("INSERT INTO alembic_version (version_num) VALUES (:down_revision)"),
            {"down_revision": DOWN_REVISION},
        )
        connection.execute(
            text(
                "CREATE TABLE tasks ("
                "id INTEGER PRIMARY KEY, status VARCHAR(32) NOT NULL)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO tasks (id, status) VALUES "
                "(1, 'RUNNING'), (2, 'paused'), (3, 'completed'), (4, 'pending')"
            )
        )

        config.attributes["connection"] = connection
        command.upgrade(config, REVISION)

        columns = {
            column["name"] for column in inspect(connection).get_columns("tasks")
        }
        indexes = {index["name"] for index in inspect(connection).get_indexes("tasks")}
        rows = connection.execute(
            text(
                "SELECT id, run_id, state_version, control_state FROM tasks ORDER BY id"
            )
        ).all()

        assert {"run_id", "state_version", "control_state"} <= columns
        assert "ix_tasks_run_id" in indexes
        assert rows == [
            (1, None, 0, "running"),
            (2, None, 0, "paused"),
            (3, None, 0, "completed"),
            (4, None, 0, "idle"),
        ]

        command.downgrade(config, DOWN_REVISION)
        downgraded_columns = {
            column["name"] for column in inspect(connection).get_columns("tasks")
        }
        assert "run_id" not in downgraded_columns
        assert "state_version" not in downgraded_columns
        assert "control_state" not in downgraded_columns


def test_upgrade_defaults_control_state_when_legacy_tasks_lack_status() -> None:
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
        connection.execute(text("CREATE TABLE tasks (id INTEGER PRIMARY KEY)"))
        connection.execute(text("INSERT INTO tasks (id) VALUES (1)"))

        config.attributes["connection"] = connection
        command.upgrade(config, REVISION)

        row = connection.execute(
            text("SELECT run_id, state_version, control_state FROM tasks")
        ).one()

        assert row == (None, 0, "idle")
