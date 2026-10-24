"""Tests for the MCP server signed upload settings migration."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session


def _load_migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20261023_add_mcp_signed_uploads.py"
    )
    spec = importlib.util.spec_from_file_location(
        "mcp_signed_uploads_migration", migration_file
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection):
    context = MigrationContext.configure(connection)
    return Operations(context)


def _engine(tmp_path):
    return create_engine(f"sqlite:///{tmp_path / 'test.db'}")


def _create_mcp_servers(connection) -> None:
    connection.execute(
        text(
            """
            CREATE TABLE mcp_servers (
                id INTEGER PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                managed VARCHAR(20) NOT NULL,
                transport VARCHAR(50) NOT NULL
            )
            """
        )
    )


def _signed_upload_columns(connection) -> list[dict]:
    return [
        column
        for column in inspect(connection).get_columns("mcp_servers")
        if column["name"] == "signed_uploads"
    ]


def test_upgrade_adds_nullable_signed_uploads_column(tmp_path) -> None:
    engine = _engine(tmp_path)
    migration = _load_migration_module()

    with engine.begin() as connection:
        _create_mcp_servers(connection)

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()

        columns = _signed_upload_columns(connection)
        assert len(columns) == 1
        assert columns[0]["nullable"] is True


def test_upgrade_and_downgrade_skip_a_missing_table(tmp_path) -> None:
    engine = _engine(tmp_path)
    migration = _load_migration_module()

    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()

        assert "mcp_servers" not in inspect(connection).get_table_names()


def test_downgrade_removes_signed_uploads_column(tmp_path) -> None:
    engine = _engine(tmp_path)
    migration = _load_migration_module()

    with engine.begin() as connection:
        _create_mcp_servers(connection)

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
            migration.downgrade()

        names = {
            column["name"] for column in inspect(connection).get_columns("mcp_servers")
        }
        assert "signed_uploads" not in names
        assert {"id", "name", "managed", "transport"} <= names


def test_upgrade_after_create_all_keeps_one_column(tmp_path) -> None:
    from xagent.web import models as _models  # noqa: F401
    from xagent.web.models.database import Base

    engine = _engine(tmp_path)
    migration = _load_migration_module()

    with engine.begin() as connection:
        Base.metadata.create_all(connection)

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        assert len(_signed_upload_columns(connection)) == 1


def test_model_stores_none_as_sql_null(tmp_path) -> None:
    from xagent.web import models as _models  # noqa: F401
    from xagent.web.models.database import Base
    from xagent.web.models.mcp import MCPServer

    engine = _engine(tmp_path)
    Base.metadata.create_all(engine)

    def _server(name: str, signed_uploads) -> MCPServer:
        server = MCPServer.from_config(
            {
                "name": name,
                "managed": "external",
                "transport": "streamable_http",
                "url": "https://mcp.example.test/mcp",
            }
        )
        server.signed_uploads = signed_uploads
        return server

    with Session(engine) as session:
        session.add(_server("explicit-none", None))
        session.add(_server("empty-list", []))
        session.commit()

    with engine.connect() as connection:
        rows = {
            row[0]: (row[1], row[2])
            for row in connection.execute(
                text(
                    "SELECT name, signed_uploads IS NULL, signed_uploads "
                    "FROM mcp_servers"
                )
            )
        }

    assert rows["explicit-none"][0] == 1
    assert rows["empty-list"] == (0, "[]")
