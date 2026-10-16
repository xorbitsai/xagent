"""Tests for adding offline_access to the builtin Microsoft connectors."""

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
    / "src/xagent/migrations/versions/20261008_microsoft_offline_access.py"
)

OLD_SCOPES = {
    "teams": [
        "Team.ReadBasic.All",
        "Channel.ReadBasic.All",
        "TeamMember.Read.All",
        "ChannelMessage.Read.All",
        "ChannelMessage.Send",
        "Chat.ReadWrite",
    ],
    "outlook": ["Mail.Read", "Mail.Send", "Calendars.ReadWrite", "Contacts.Read"],
    "onedrive": ["Files.ReadWrite"],
    "planner": ["Tasks.ReadWrite"],
    "powerpoint": ["Files.ReadWrite"],
    "sharepoint": ["Sites.ReadWrite.All"],
    "word": ["Files.ReadWrite.All"],
}
NEW_SCOPES = {
    app_id: [*scopes, "offline_access"] for app_id, scopes in OLD_SCOPES.items()
}
EXCEL_SCOPES = ["Files.ReadWrite", "offline_access"]


def _load_migration_module():
    spec = importlib.util.spec_from_file_location(
        "microsoft_offline_access_migration", MIGRATION_PATH
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
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("app_id", sa.String(100), nullable=False, unique=True),
        sa.Column("provider_name", sa.String(50)),
        sa.Column("oauth_scopes", sa.JSON),
    )


def _seed(connection, table: sa.Table) -> None:
    rows = [
        {"app_id": app_id, "provider_name": "microsoft", "oauth_scopes": scopes}
        for app_id, scopes in OLD_SCOPES.items()
    ]
    rows.append(
        {"app_id": "excel", "provider_name": "microsoft", "oauth_scopes": EXCEL_SCOPES}
    )
    connection.execute(sa.insert(table), rows)


def _scopes_by_app_id(connection, table: sa.Table) -> dict[str, object]:
    return {
        row["app_id"]: row["oauth_scopes"]
        for row in connection.execute(sa.select(table)).mappings()
    }


def test_upgrade_adds_offline_access_to_every_microsoft_connector(tmp_path) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    table = _public_mcp_apps(metadata)

    with engine.begin() as connection:
        metadata.create_all(connection)
        _seed(connection, table)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        scopes = _scopes_by_app_id(connection, table)

    assert scopes == {**NEW_SCOPES, "excel": EXCEL_SCOPES}


def test_upgrade_is_idempotent(tmp_path) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    table = _public_mcp_apps(metadata)

    with engine.begin() as connection:
        metadata.create_all(connection)
        _seed(connection, table)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()
        scopes = _scopes_by_app_id(connection, table)

    assert scopes == {**NEW_SCOPES, "excel": EXCEL_SCOPES}


def test_downgrade_restores_previous_scopes(tmp_path) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    table = _public_mcp_apps(metadata)

    with engine.begin() as connection:
        metadata.create_all(connection)
        _seed(connection, table)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        scopes = _scopes_by_app_id(connection, table)

    assert scopes == {**OLD_SCOPES, "excel": EXCEL_SCOPES}


def test_upgrade_leaves_non_microsoft_row_with_same_app_id_alone(tmp_path) -> None:
    """A custom row squatting a builtin app_id under another provider is the
    operator's own app; offline_access may not even be a valid scope there."""
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    table = _public_mcp_apps(metadata)

    with engine.begin() as connection:
        metadata.create_all(connection)
        connection.execute(
            sa.insert(table),
            {
                "app_id": "sharepoint",
                "provider_name": "custom-idp",
                "oauth_scopes": ["custom-scope"],
            },
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        scopes = _scopes_by_app_id(connection, table)

    assert scopes == {"sharepoint": ["custom-scope"]}


def test_upgrade_and_downgrade_no_op_without_table(tmp_path) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")

    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        table_names = set(sa.inspect(connection).get_table_names())

    assert "public_mcp_apps" not in table_names


def test_offline_sqlite_upgrade_round_trips_scopes() -> None:
    migration = _load_migration_module()
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="sqlite",
        opts={"as_sql": True, "output_buffer": output},
    )

    with Operations.context(context):
        migration.upgrade()

    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE public_mcp_apps ("
        "app_id TEXT PRIMARY KEY, provider_name TEXT, oauth_scopes JSON)"
    )
    connection.executemany(
        "INSERT INTO public_mcp_apps VALUES (?, 'microsoft', ?)",
        [
            (app_id, json.dumps(scopes))
            for app_id, scopes in OLD_SCOPES.items()
            if app_id != "sharepoint"
        ],
    )
    # app_id is unique, so a custom row squatting a builtin app_id takes the
    # place of that builtin row rather than sitting beside it.
    connection.execute(
        "INSERT INTO public_mcp_apps VALUES "
        "('sharepoint', 'custom-idp', '[\"custom-scope\"]')"
    )
    connection.executescript(output.getvalue())
    scopes = {
        app_id: json.loads(value)
        for app_id, value in connection.execute(
            "SELECT app_id, oauth_scopes FROM public_mcp_apps"
        )
    }
    connection.close()

    assert scopes == {**NEW_SCOPES, "sharepoint": ["custom-scope"]}


def test_offline_postgresql_upgrade_contains_only_literal_updates() -> None:
    migration = _load_migration_module()
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="postgresql",
        opts={"as_sql": True, "output_buffer": output},
    )

    with Operations.context(context):
        migration.upgrade()

    sql = output.getvalue()
    assert sql.count("UPDATE public_mcp_apps SET") == len(OLD_SCOPES)
    assert sql.count("offline_access") == len(OLD_SCOPES)
    assert sql.count("provider_name = 'microsoft'") == len(OLD_SCOPES)
    assert sql.count("CAST(") == len(OLD_SCOPES)
    assert "%(" not in sql


def test_registry_and_migration_values_match() -> None:
    migration = _load_migration_module()
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    rows = {row["app_id"]: row for row in get_builtin_public_mcp_app_rows()}

    for app_id, scopes in migration.CURRENT_SCOPES.items():
        assert rows[app_id]["provider_name"] == "microsoft"
        registry_scopes = rows[app_id]["oauth_scopes"]
        if app_id == "planner":
            # Team.ReadBasic.All was added after this frozen migration snapshot.
            registry_scopes = [
                scope for scope in registry_scopes if scope != "Team.ReadBasic.All"
            ]
        assert registry_scopes == list(scopes)
    assert {
        app_id: list(scopes) for app_id, scopes in migration.PREVIOUS_SCOPES.items()
    } == OLD_SCOPES


def test_revision_metadata() -> None:
    migration = _load_migration_module()

    assert migration.revision == "20261008_microsoft_offline_access"
    assert migration.down_revision == "20261008_task_auto_recovery"
