"""Tests for the Zendesk side_conversations:write scope migration."""

import importlib.util
import json
import sqlite3
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20261102_zendesk_side_conversations_write_scope.py"
)


def _load_migration_module():
    spec = importlib.util.spec_from_file_location(
        "zendesk_side_conversations_write_scope_migration", MIGRATION_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection) -> Operations:
    return Operations(MigrationContext.configure(connection))


def _public_mcp_apps(metadata: sa.MetaData) -> sa.Table:
    return sa.Table(
        "public_mcp_apps",
        metadata,
        sa.Column("app_id", sa.String(100), primary_key=True),
        sa.Column("provider_name", sa.String(50)),
        sa.Column("oauth_scopes", sa.JSON),
    )


def _rows(connection, table: sa.Table) -> dict[str, dict[str, object]]:
    return {
        row["app_id"]: dict(row)
        for row in connection.execute(sa.select(table)).mappings()
    }


def test_upgrade_and_downgrade_values(tmp_path) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    table = _public_mcp_apps(metadata)

    with engine.begin() as connection:
        metadata.create_all(connection)
        connection.execute(
            sa.insert(table),
            [
                {
                    "app_id": migration.APP_ID,
                    "provider_name": migration.PROVIDER_NAME,
                    "oauth_scopes": list(migration.PREVIOUS_SCOPES),
                },
                {
                    "app_id": "custom-zendesk",
                    "provider_name": migration.PROVIDER_NAME,
                    "oauth_scopes": ["read"],
                },
            ],
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()
        rows = _rows(connection, table)
        assert rows[migration.APP_ID]["oauth_scopes"] == [
            "read",
            "tickets:write",
            "side_conversations:write",
        ]
        assert rows["custom-zendesk"]["oauth_scopes"] == ["read"]

        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()
        row = _rows(connection, table)[migration.APP_ID]
        assert row["oauth_scopes"] == ["read", "tickets:write"]


def test_upgrade_leaves_provider_collision_unchanged(tmp_path) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    table = _public_mcp_apps(metadata)

    with engine.begin() as connection:
        metadata.create_all(connection)
        connection.execute(
            sa.insert(table),
            {
                "app_id": migration.APP_ID,
                "provider_name": "custom-provider",
                "oauth_scopes": ["custom:scope"],
            },
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        row = _rows(connection, table)[migration.APP_ID]
        assert row["oauth_scopes"] == ["custom:scope"]


def test_upgrade_is_noop_without_expected_table_shape(tmp_path) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")

    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert "public_mcp_apps" not in sa.inspect(connection).get_table_names()


def test_upgrade_does_not_disconnect_existing_grants(tmp_path) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    table = _public_mcp_apps(metadata)
    grants = sa.Table(
        "user_oauth",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("provider", sa.String),
        sa.Column("access_token", sa.String),
        sa.Column("refresh_token", sa.String),
    )

    with engine.begin() as connection:
        metadata.create_all(connection)
        connection.execute(
            sa.insert(table),
            {
                "app_id": migration.APP_ID,
                "provider_name": migration.PROVIDER_NAME,
                "oauth_scopes": list(migration.PREVIOUS_SCOPES),
            },
        )
        connection.execute(
            sa.insert(grants),
            {
                "id": 1,
                "provider": migration.PROVIDER_NAME,
                "access_token": "existing-access",
                "refresh_token": "existing-refresh",
            },
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        grant = connection.execute(sa.select(grants)).mappings().one()

    assert grant["access_token"] == "existing-access"
    assert grant["refresh_token"] == "existing-refresh"


def test_offline_sqlite_upgrade_and_downgrade_round_trip() -> None:
    migration = _load_migration_module()
    upgrade_output = StringIO()
    upgrade_context = MigrationContext.configure(
        dialect_name="sqlite",
        opts={"as_sql": True, "output_buffer": upgrade_output},
    )
    with Operations.context(upgrade_context):
        migration.upgrade()

    downgrade_output = StringIO()
    downgrade_context = MigrationContext.configure(
        dialect_name="sqlite",
        opts={"as_sql": True, "output_buffer": downgrade_output},
    )
    with Operations.context(downgrade_context):
        migration.downgrade()

    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE public_mcp_apps ("
        "app_id TEXT PRIMARY KEY, provider_name TEXT, oauth_scopes JSON)"
    )
    connection.execute(
        "INSERT INTO public_mcp_apps VALUES (?, ?, ?)",
        (
            migration.APP_ID,
            migration.PROVIDER_NAME,
            json.dumps(migration.PREVIOUS_SCOPES),
        ),
    )
    connection.executescript(upgrade_output.getvalue())
    (stored_scopes,) = connection.execute(
        "SELECT oauth_scopes FROM public_mcp_apps WHERE app_id='zendesk'"
    ).fetchone()
    assert json.loads(stored_scopes) == list(migration.CURRENT_SCOPES)

    connection.executescript(downgrade_output.getvalue())
    (stored_scopes,) = connection.execute(
        "SELECT oauth_scopes FROM public_mcp_apps WHERE app_id='zendesk'"
    ).fetchone()
    connection.close()
    assert json.loads(stored_scopes) == list(migration.PREVIOUS_SCOPES)


def test_offline_postgresql_upgrade_uses_literal_json_cast() -> None:
    migration = _load_migration_module()
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="postgresql", opts={"as_sql": True, "output_buffer": output}
    )

    with Operations.context(context):
        migration.upgrade()

    sql = output.getvalue()
    assert "UPDATE public_mcp_apps SET" in sql
    assert "side_conversations:write" in sql
    assert "provider_name = 'zendesk'" in sql
    assert "CAST(" in sql
    assert "%(" not in sql


def test_registry_and_migration_values_match() -> None:
    migration = _load_migration_module()
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app

    row = get_builtin_public_mcp_app(migration.APP_ID)

    assert row is not None
    assert row["provider_name"] == migration.PROVIDER_NAME
    assert row["oauth_scopes"] == list(migration.CURRENT_SCOPES)


def test_revision_metadata() -> None:
    migration = _load_migration_module()

    assert migration.revision == "20261102_zendesk_side_conv"
    assert migration.down_revision == "20261031_speech_settings"
