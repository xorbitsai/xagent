import pytest
from alembic import command
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

from xagent.db.config import create_alembic_config


def test_upgrade_allows_multiple_assistant_rows_but_keeps_user_idempotency() -> None:
    engine = create_engine("sqlite:///:memory:")
    config = create_alembic_config(engine)

    with engine.begin() as connection:
        connection.execute(
            text("CREATE TABLE alembic_version (version_num VARCHAR(255) NOT NULL)")
        )
        connection.execute(
            text(
                "INSERT INTO alembic_version (version_num) "
                "VALUES ('20261027_wrap_legacy_password_hashes')"
            )
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
                "CREATE UNIQUE INDEX uq_task_chat_messages_task_role_turn_id "
                "ON task_chat_messages (task_id, role, turn_id)"
            )
        )
        config.attributes["connection"] = connection
        command.upgrade(config, "head")

        connection.execute(
            text(
                "INSERT INTO task_chat_messages (id, task_id, role, turn_id) "
                "VALUES (1, 7, 'assistant', 'turn'), "
                "(2, 7, 'assistant', 'turn'), (3, 7, 'user', 'turn')"
            )
        )
        index = {
            item["name"]: item
            for item in inspect(connection).get_indexes("task_chat_messages")
        }["uq_task_chat_messages_task_role_turn_id"]
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO task_chat_messages "
                    "(id, task_id, role, turn_id) VALUES (4, 7, 'user', 'turn')"
                )
            )

    assert index["unique"] == 1
    assert index["dialect_options"]["sqlite_where"] is not None
