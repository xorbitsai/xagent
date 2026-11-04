import importlib
from io import StringIO

from alembic import command
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

from xagent.db.config import create_alembic_config

REVISION = "20260713_mcp_oauth_dcr"
DOWN_REVISION = "20260711_task_commands"


def test_upgrade_adds_nullable_unique_registration_lookup_hash() -> None:
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
        assert columns["registration_lookup_hash"]["nullable"] is True
        assert columns["registration_lookup_hash"]["type"].length == 64
        indexes = {
            index["name"]: index
            for index in inspect(connection).get_indexes("mcp_oauth_clients")
        }
        registration_index = indexes["ux_mcp_oauth_clients_registration_lookup_hash"]
        assert registration_index["column_names"] == ["registration_lookup_hash"]
        assert registration_index["unique"] == 1
        assert (
            connection.execute(
                text(
                    "SELECT registration_lookup_hash FROM mcp_oauth_clients WHERE id = 1"
                )
            ).scalar_one()
            is None
        )

        connection.execute(
            text(
                "INSERT INTO mcp_oauth_clients "
                "(id, client_id, registration_lookup_hash) "
                "VALUES (2, 'dynamic-client', 'same-hash')"
            )
        )
        try:
            connection.execute(
                text(
                    "INSERT INTO mcp_oauth_clients "
                    "(id, client_id, registration_lookup_hash) "
                    "VALUES (3, 'other-dynamic-client', 'same-hash')"
                )
            )
        except IntegrityError:
            pass
        else:
            raise AssertionError("registration lookup hash must be unique")

        command.downgrade(config, DOWN_REVISION)
        assert "registration_lookup_hash" not in {
            column["name"]
            for column in inspect(connection).get_columns("mcp_oauth_clients")
        }


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


def test_offline_upgrade_emits_portable_column_and_unique_index() -> None:
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="postgresql",
        opts={"as_sql": True, "output_buffer": output},
    )
    migration = importlib.import_module(
        "xagent.migrations.versions.20260713_add_mcp_oauth_client_registration"
    )

    with Operations.context(context):
        migration.upgrade()

    sql = output.getvalue()
    assert "ALTER TABLE mcp_oauth_clients ADD COLUMN registration_lookup_hash" in sql
    assert "CREATE UNIQUE INDEX ux_mcp_oauth_clients_registration_lookup_hash" in sql
