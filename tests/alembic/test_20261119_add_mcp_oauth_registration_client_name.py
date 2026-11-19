import importlib
from io import StringIO

from alembic import command
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text

from xagent.db.config import create_alembic_config

REVISION = "20261119_mcp_oauth_client_name"
DOWN_REVISION = "20261117_chat_message_metadata"


def test_upgrade_adds_nullable_registration_client_name() -> None:
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
        connection.execute(
            text(
                "CREATE TABLE mcp_oauth_clients ("
                "id INTEGER PRIMARY KEY, "
                "client_id VARCHAR(1000) NOT NULL"
                ")"
            )
        )
        connection.execute(
            text(
                "INSERT INTO mcp_oauth_clients (id, client_id) "
                "VALUES (1, 'manual-client')"
            )
        )
        config.attributes["connection"] = connection

        command.upgrade(config, REVISION)

        columns = {
            column["name"]: column
            for column in inspect(connection).get_columns("mcp_oauth_clients")
        }
        assert columns["registration_client_name"]["nullable"] is True
        assert columns["registration_client_name"]["type"].length == 255
        assert (
            connection.execute(
                text(
                    "SELECT registration_client_name FROM mcp_oauth_clients "
                    "WHERE id = 1"
                )
            ).scalar_one()
            is None
        )

        command.downgrade(config, DOWN_REVISION)
        assert "registration_client_name" not in {
            column["name"]
            for column in inspect(connection).get_columns("mcp_oauth_clients")
        }


def test_upgrade_is_idempotent_when_the_column_already_exists() -> None:
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
        connection.execute(
            text(
                "CREATE TABLE mcp_oauth_clients ("
                "id INTEGER PRIMARY KEY, "
                "client_id VARCHAR(1000) NOT NULL"
                ")"
            )
        )
        config.attributes["connection"] = connection

        command.upgrade(config, REVISION)
        # Rewind the stamp so the same revision runs again against a table
        # that already has the column.
        connection.execute(
            text("UPDATE alembic_version SET version_num = :revision"),
            {"revision": DOWN_REVISION},
        )
        command.upgrade(config, REVISION)

        columns = {
            column["name"]
            for column in inspect(connection).get_columns("mcp_oauth_clients")
        }
        assert "registration_client_name" in columns


def test_upgrade_skips_when_mcp_oauth_clients_does_not_exist() -> None:
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

        assert "mcp_oauth_clients" not in inspect(connection).get_table_names()


def test_offline_upgrade_emits_portable_column() -> None:
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="postgresql",
        opts={"as_sql": True, "output_buffer": output},
    )
    migration = importlib.import_module(
        "xagent.migrations.versions.20261119_add_mcp_oauth_registration_client_name"
    )

    with Operations.context(context):
        migration.upgrade()

    assert (
        "ALTER TABLE mcp_oauth_clients ADD COLUMN registration_client_name"
        in output.getvalue()
    )
