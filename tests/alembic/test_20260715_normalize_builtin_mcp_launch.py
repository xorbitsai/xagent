"""Tests for normalizing built-in MCP execution fields."""

import importlib.util
import json
import sqlite3
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20260715_normalize_builtin_mcp_launch.py"
)

EXECUTION_FIELD_NAMES = (
    "name",
    "transport",
    "provider_name",
    "oauth_scopes",
    "launch_config",
)

EXPECTED_EXECUTION_FIELDS = {
    "linkedin": {
        "name": "LinkedIn",
        "transport": "oauth",
        "provider_name": "linkedin",
        "oauth_scopes": ["openid", "profile", "email", "w_member_social"],
        "launch_config": {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.linkedin"],
            "env_mapping": {"LINKEDIN_ACCESS_TOKEN": "access_token"},
        },
    },
    "gmail": {
        "name": "Gmail",
        "transport": "oauth",
        "provider_name": "google",
        "oauth_scopes": ["https://www.googleapis.com/auth/gmail.modify"],
        "launch_config": {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.gmail"],
            "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"},
        },
    },
    "google-drive": {
        "name": "Google Drive",
        "transport": "oauth",
        "provider_name": "google",
        "oauth_scopes": ["https://www.googleapis.com/auth/drive"],
        "launch_config": {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.google_drive"],
            "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"},
        },
    },
    "google-calendar": {
        "name": "Google Calendar",
        "transport": "oauth",
        "provider_name": "google",
        "oauth_scopes": ["https://www.googleapis.com/auth/calendar"],
        "launch_config": {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.calendar"],
            "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"},
        },
    },
    "teams": {
        "name": "Teams",
        "transport": "oauth",
        "provider_name": "microsoft",
        "oauth_scopes": [
            "Team.ReadBasic.All",
            "Channel.ReadBasic.All",
            "TeamMember.Read.All",
            "ChannelMessage.Read.All",
            "ChannelMessage.Send",
            "Chat.ReadWrite",
        ],
        "launch_config": {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.teams"],
            "env_mapping": {"AUTH_TOKEN": "access_token"},
        },
    },
    "outlook": {
        "name": "Outlook",
        "transport": "oauth",
        "provider_name": "microsoft",
        "oauth_scopes": [
            "Mail.Read",
            "Mail.Send",
            "Calendars.ReadWrite",
            "Contacts.Read",
        ],
        "launch_config": {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.outlook"],
            "env_mapping": {"AUTH_TOKEN": "access_token"},
        },
    },
    "onedrive": {
        "name": "OneDrive",
        "transport": "oauth",
        "provider_name": "microsoft",
        "oauth_scopes": ["Files.ReadWrite"],
        "launch_config": {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.onedrive"],
            "env_mapping": {"AUTH_TOKEN": "access_token"},
        },
    },
    "facebook": {
        "name": "Facebook Pages",
        "transport": "oauth",
        "provider_name": "meta",
        "oauth_scopes": [
            "pages_show_list",
            "pages_read_engagement",
            "pages_manage_posts",
        ],
        "launch_config": {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.facebook"],
            "env_mapping": {"META_ACCESS_TOKEN": "access_token"},
        },
    },
    "instagram": {
        "name": "Instagram",
        "transport": "oauth",
        "provider_name": "meta",
        "oauth_scopes": [
            "pages_show_list",
            "pages_read_engagement",
            "instagram_basic",
            "instagram_content_publish",
        ],
        "launch_config": {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.instagram"],
            "env_mapping": {"META_ACCESS_TOKEN": "access_token"},
        },
    },
    "google-maps": {
        "name": "Google Maps",
        "transport": "stdio",
        "provider_name": None,
        "oauth_scopes": None,
        "launch_config": {
            "command": "npx",
            "args": ["-y", "@cablate/mcp-google-map", "--stdio"],
            "required_env": ["GOOGLE_MAPS_API_KEY"],
        },
    },
}

CUSTOM_EXECUTION_FIELDS = {
    "name": "Custom uv MCP",
    "transport": "stdio",
    "provider_name": "custom-provider",
    "oauth_scopes": ["custom-scope"],
    "launch_config": {
        "command": "uv",
        "args": ["run", "custom_server.py"],
        "env": {"CUSTOM_SETTING": "preserve-me"},
    },
}


def _load_migration_module():
    spec = importlib.util.spec_from_file_location(
        "normalize_builtin_mcp_launch_migration", MIGRATION_PATH
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
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("transport", sa.String(50), nullable=False),
        sa.Column("provider_name", sa.String(50)),
        sa.Column("oauth_scopes", sa.JSON),
        sa.Column("launch_config", sa.JSON),
    )


def _uv_config(canonical: dict[str, object]) -> dict[str, object]:
    return {
        **canonical,
        "command": "uv",
        "args": ["run", "python", *canonical["args"]],
    }


def _drifted_execution_fields(
    canonical: dict[str, object],
) -> dict[str, object]:
    return {
        "name": f"Stale {canonical['name']}",
        "transport": "stdio" if canonical["transport"] == "oauth" else "oauth",
        "provider_name": "stale-provider",
        "oauth_scopes": ["stale-scope"],
        "launch_config": _uv_config(canonical["launch_config"]),
    }


def _seed_environment(
    connection,
    table: sa.Table,
    *,
    drifted_app_ids: set[str],
    missing_app_ids: set[str] | None = None,
) -> None:
    missing_app_ids = missing_app_ids or set()
    rows = [
        {
            "app_id": app_id,
            **(
                _drifted_execution_fields(execution_fields)
                if app_id in drifted_app_ids
                else execution_fields
            ),
        }
        for app_id, execution_fields in EXPECTED_EXECUTION_FIELDS.items()
        if app_id not in missing_app_ids
    ]
    rows.append({"app_id": "custom-uv", **CUSTOM_EXECUTION_FIELDS})
    connection.execute(sa.insert(table), rows)


def _execution_fields_by_app_id(
    connection, table: sa.Table
) -> dict[str, dict[str, object]]:
    rows = connection.execute(
        sa.select(
            table.c.app_id,
            *(table.c[field] for field in EXECUTION_FIELD_NAMES),
        ).order_by(table.c.app_id)
    )
    return {
        row.app_id: {field: getattr(row, field) for field in EXECUTION_FIELD_NAMES}
        for row in rows
    }


@pytest.mark.parametrize(
    "drifted_app_ids",
    [
        set(EXPECTED_EXECUTION_FIELDS),
        {"instagram", "google-maps"},
    ],
    ids=["au-shaped-data", "sg-shaped-data"],
)
def test_upgrade_converges_environment_data_without_touching_other_apps(
    tmp_path, drifted_app_ids
) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    table = _public_mcp_apps(metadata)
    metadata.create_all(engine)

    with engine.begin() as connection:
        _seed_environment(connection, table, drifted_app_ids=drifted_app_ids)

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        execution_fields = _execution_fields_by_app_id(connection, table)

    for app_id, expected_fields in EXPECTED_EXECUTION_FIELDS.items():
        assert execution_fields[app_id] == expected_fields
    assert execution_fields["custom-uv"] == CUSTOM_EXECUTION_FIELDS


def test_upgrade_is_update_only_and_idempotent(tmp_path) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    table = _public_mcp_apps(metadata)
    metadata.create_all(engine)

    with engine.begin() as connection:
        _seed_environment(
            connection,
            table,
            drifted_app_ids=set(EXPECTED_EXECUTION_FIELDS),
            missing_app_ids={"facebook"},
        )

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            first_upgrade = _execution_fields_by_app_id(connection, table)
            migration.upgrade()
            second_upgrade = _execution_fields_by_app_id(connection, table)

    assert "facebook" not in first_upgrade
    assert first_upgrade == second_upgrade
    assert first_upgrade["google-maps"] == EXPECTED_EXECUTION_FIELDS["google-maps"]
    assert first_upgrade["custom-uv"] == CUSTOM_EXECUTION_FIELDS


def test_downgrade_does_not_restore_invalid_execution_fields(tmp_path) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    table = _public_mcp_apps(metadata)
    metadata.create_all(engine)

    with engine.begin() as connection:
        _seed_environment(
            connection,
            table,
            drifted_app_ids=set(EXPECTED_EXECUTION_FIELDS),
        )

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()

        execution_fields = _execution_fields_by_app_id(connection, table)

    for app_id, expected_fields in EXPECTED_EXECUTION_FIELDS.items():
        assert execution_fields[app_id] == expected_fields


def test_upgrade_skips_when_catalog_table_is_absent() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()


def test_upgrade_skips_when_required_catalog_columns_are_absent() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")
    metadata = sa.MetaData()
    table = sa.Table(
        "public_mcp_apps",
        metadata,
        sa.Column("app_id", sa.String(100), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("launch_config", sa.JSON),
    )
    metadata.create_all(engine)
    stale_launch_config = _uv_config(
        EXPECTED_EXECUTION_FIELDS["gmail"]["launch_config"]
    )

    with engine.begin() as connection:
        connection.execute(
            sa.insert(table),
            {
                "app_id": "gmail",
                "name": "Stale Gmail",
                "launch_config": stale_launch_config,
            },
        )

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        stored = connection.execute(sa.select(table)).mappings().one()

    assert stored["name"] == "Stale Gmail"
    assert stored["launch_config"] == stale_launch_config


def test_offline_postgresql_upgrade_emits_literal_update_only_sql() -> None:
    migration = _load_migration_module()
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="postgresql",
        opts={"as_sql": True, "output_buffer": output},
    )

    with Operations.context(context):
        migration.upgrade()

    sql = output.getvalue()
    assert sql.count("UPDATE public_mcp_apps SET") == len(EXPECTED_EXECUTION_FIELDS)
    assert sql.count("SET name=") == len(EXPECTED_EXECUTION_FIELDS)
    assert sql.count("transport=") == len(EXPECTED_EXECUTION_FIELDS)
    assert sql.count("provider_name=") == len(EXPECTED_EXECUTION_FIELDS)
    assert sql.count("oauth_scopes=") == len(EXPECTED_EXECUTION_FIELDS)
    assert sql.count("launch_config=") == len(EXPECTED_EXECUTION_FIELDS)
    assert "INSERT INTO public_mcp_apps" not in sql
    assert "DELETE FROM public_mcp_apps" not in sql
    assert "%(" not in sql
    assert '"command": "python"' in sql
    for app_id, expected_fields in EXPECTED_EXECUTION_FIELDS.items():
        assert f"public_mcp_apps.app_id = '{app_id}'" in sql
        assert expected_fields["name"] in sql
        assert expected_fields["launch_config"]["args"][1] in sql


def test_offline_sqlite_upgrade_round_trips_json_catalog_values() -> None:
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
        "app_id TEXT PRIMARY KEY, name TEXT NOT NULL, transport TEXT NOT NULL, "
        "provider_name TEXT, oauth_scopes JSON, launch_config JSON)"
    )
    drifted = _drifted_execution_fields(EXPECTED_EXECUTION_FIELDS["gmail"])
    connection.execute(
        "INSERT INTO public_mcp_apps "
        "(app_id, name, transport, provider_name, oauth_scopes, launch_config) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            "gmail",
            drifted["name"],
            drifted["transport"],
            drifted["provider_name"],
            json.dumps(drifted["oauth_scopes"]),
            json.dumps(drifted["launch_config"]),
        ),
    )
    connection.executescript(output.getvalue())

    stored = connection.execute(
        "SELECT name, transport, provider_name, oauth_scopes, launch_config "
        "FROM public_mcp_apps WHERE app_id = 'gmail'"
    ).fetchone()

    assert stored is not None
    expected = EXPECTED_EXECUTION_FIELDS["gmail"]
    assert stored[:3] == (
        expected["name"],
        expected["transport"],
        expected["provider_name"],
    )
    assert json.loads(stored[3]) == expected["oauth_scopes"]
    assert json.loads(stored[4]) == expected["launch_config"]


def test_online_and_offline_upgrades_store_json_null_consistently(tmp_path) -> None:
    migration = _load_migration_module()
    online_engine = sa.create_engine(f"sqlite:///{tmp_path / 'online.db'}")
    metadata = sa.MetaData()
    table = _public_mcp_apps(metadata)
    metadata.create_all(online_engine)

    with online_engine.begin() as connection:
        _seed_environment(
            connection,
            table,
            drifted_app_ids={"google-maps"},
            missing_app_ids=set(EXPECTED_EXECUTION_FIELDS) - {"google-maps"},
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        online_value = connection.exec_driver_sql(
            "SELECT oauth_scopes, oauth_scopes IS NULL "
            "FROM public_mcp_apps WHERE app_id = 'google-maps'"
        ).one()

    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="sqlite",
        opts={"as_sql": True, "output_buffer": output},
    )
    with Operations.context(context):
        migration.upgrade()

    offline_connection = sqlite3.connect(":memory:")
    offline_connection.execute(
        "CREATE TABLE public_mcp_apps ("
        "app_id TEXT PRIMARY KEY, name TEXT NOT NULL, transport TEXT NOT NULL, "
        "provider_name TEXT, oauth_scopes JSON, launch_config JSON)"
    )
    drifted = _drifted_execution_fields(EXPECTED_EXECUTION_FIELDS["google-maps"])
    offline_connection.execute(
        "INSERT INTO public_mcp_apps "
        "(app_id, name, transport, provider_name, oauth_scopes, launch_config) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            "google-maps",
            drifted["name"],
            drifted["transport"],
            drifted["provider_name"],
            json.dumps(drifted["oauth_scopes"]),
            json.dumps(drifted["launch_config"]),
        ),
    )
    offline_connection.executescript(output.getvalue())
    offline_value = offline_connection.execute(
        "SELECT oauth_scopes, oauth_scopes IS NULL "
        "FROM public_mcp_apps WHERE app_id = 'google-maps'"
    ).fetchone()
    offline_connection.close()

    assert online_value == offline_value == ("null", 0)


def test_offline_mysql_upgrade_emits_json_text_literals_without_bind_parameters() -> (
    None
):
    migration = _load_migration_module()
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="mysql",
        opts={"as_sql": True, "output_buffer": output},
    )

    with Operations.context(context):
        migration.upgrade()

    sql = output.getvalue()
    assert sql.count("UPDATE public_mcp_apps") == len(EXPECTED_EXECUTION_FIELDS)
    assert sql.count("SET name=") == len(EXPECTED_EXECUTION_FIELDS)
    assert sql.count("transport=") == len(EXPECTED_EXECUTION_FIELDS)
    assert sql.count("provider_name=") == len(EXPECTED_EXECUTION_FIELDS)
    assert sql.count("oauth_scopes=") == len(EXPECTED_EXECUTION_FIELDS)
    assert sql.count("launch_config=") == len(EXPECTED_EXECUTION_FIELDS)
    assert "CAST(" not in sql
    assert ":app_id" not in sql
    for field in EXECUTION_FIELD_NAMES:
        assert f":{field}" not in sql
    for app_id, expected_fields in EXPECTED_EXECUTION_FIELDS.items():
        assert f"app_id = '{app_id}'" in sql
        assert expected_fields["name"] in sql
        assert json.dumps(expected_fields["oauth_scopes"], sort_keys=True) in sql
        assert json.dumps(expected_fields["launch_config"], sort_keys=True) in sql


def test_offline_postgresql_downgrade_emits_no_sql() -> None:
    migration = _load_migration_module()
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="postgresql",
        opts={"as_sql": True, "output_buffer": output},
    )

    with Operations.context(context):
        migration.downgrade()

    assert output.getvalue() == ""


def test_revision_metadata() -> None:
    migration = _load_migration_module()

    assert migration.revision == "20260715_normalize_builtin_mcp_launch"
    assert migration.down_revision == "20260713_add_agent_visibility"
