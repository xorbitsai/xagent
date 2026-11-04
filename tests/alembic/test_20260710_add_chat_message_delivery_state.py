from alembic import command
from sqlalchemy import create_engine, inspect, text

from xagent.db.config import create_alembic_config


def test_upgrade_adds_delivery_state_and_deduplicates_turn_ids() -> None:
    engine = create_engine("sqlite:///:memory:")
    config = create_alembic_config(engine)

    with engine.begin() as connection:
        connection.execute(
            text("CREATE TABLE alembic_version (version_num VARCHAR(255) NOT NULL)")
        )
        connection.execute(
            text("INSERT INTO alembic_version (version_num) VALUES ('1c2ae61b5a6d')")
        )
        connection.execute(
            text(
                "CREATE TABLE task_chat_messages ("
                "id INTEGER PRIMARY KEY, task_id INTEGER NOT NULL, "
                "role VARCHAR(32) NOT NULL, turn_id VARCHAR(64))"
            )
        )
        connection.execute(
            text(
                "INSERT INTO task_chat_messages (id, task_id, role, turn_id) "
                "VALUES (1, 7, 'user', 'same-turn'), "
                "(2, 7, 'user', 'same-turn'), "
                "(3, 7, 'assistant', 'same-turn')"
            )
        )

        config.attributes["connection"] = connection
        command.upgrade(config, "head")

        columns = {
            column["name"]
            for column in inspect(connection).get_columns("task_chat_messages")
        }
        indexes = {
            index["name"]: index
            for index in inspect(connection).get_indexes("task_chat_messages")
        }
        rows = connection.execute(
            text("SELECT id, turn_id FROM task_chat_messages ORDER BY id")
        ).all()

    assert "delivery_status" in columns
    assert indexes["uq_task_chat_messages_task_role_turn_id"]["unique"] == 1
    assert rows == [(1, "same-turn"), (2, None), (3, "same-turn")]
