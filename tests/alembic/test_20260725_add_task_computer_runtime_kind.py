from alembic import command
from sqlalchemy import create_engine, inspect, text

from xagent.db.config import create_alembic_config

REVISION = "20260725_add_task_computer_runtime_kind"
DOWN_REVISION = "20260724_seed_google_ads_mcp_app"


def test_upgrade_adds_nullable_task_computer_runtime_kind() -> None:
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

        columns = {
            column["name"] for column in inspect(connection).get_columns("tasks")
        }
        assert "computer_runtime_kind" in columns
        assert (
            connection.execute(
                text("SELECT computer_runtime_kind FROM tasks WHERE id = 1")
            ).scalar_one()
            is None
        )

        command.downgrade(config, DOWN_REVISION)
        downgraded_columns = {
            column["name"] for column in inspect(connection).get_columns("tasks")
        }
        assert "computer_runtime_kind" not in downgraded_columns
