"""Tests for the WhatsApp Business MCP connector seed migration."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text


def _load_migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20260909_seed_whatsapp_mcp_app.py"
    )
    spec = importlib.util.spec_from_file_location(
        "seed_whatsapp_migration", migration_file
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection):
    return Operations(MigrationContext.configure(connection))


def _create_apps_table(connection):
    connection.execute(
        text(
            """
            CREATE TABLE public_mcp_apps (
                id INTEGER PRIMARY KEY,
                app_id VARCHAR(100) NOT NULL UNIQUE,
                name VARCHAR(200) NOT NULL,
                description TEXT,
                icon VARCHAR(1000),
                transport VARCHAR(50) NOT NULL DEFAULT 'oauth',
                provider_name VARCHAR(50),
                category VARCHAR(100),
                oauth_scopes JSON,
                is_visible_in_connector BOOLEAN NOT NULL DEFAULT 1,
                launch_config JSON
            )
            """
        )
    )


def _app_ids(connection):
    return set(connection.execute(text("SELECT app_id FROM public_mcp_apps")).scalars())


def _run(connection, migration, *steps):
    with patch.object(migration, "op", _operations(connection)):
        for step in steps:
            getattr(migration, step)()


def test_upgrade_inserts_whatsapp(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_apps_table(connection)
        _run(connection, migration, "upgrade")
        assert "whatsapp" in _app_ids(connection)
        row = connection.execute(
            text(
                "SELECT transport, provider_name, oauth_scopes, launch_config, "
                "is_visible_in_connector, category FROM public_mcp_apps"
                " WHERE app_id='whatsapp'"
            )
        ).first()
        assert row[0] == "oauth"
        assert row[1] == "meta"
        for scope in (
            "business_management",
            "whatsapp_business_management",
            "whatsapp_business_messaging",
        ):
            assert scope in str(row[2])
        assert "xagent.web.tools.mcp.whatsapp" in str(row[3])
        assert "META_ACCESS_TOKEN" in str(row[3])
        assert row[4] == 1
        assert row[5] == "Communication"


def test_upgrade_is_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_apps_table(connection)
        _run(connection, migration, "upgrade", "upgrade")
        rows = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id='whatsapp'")
        ).scalar()
        assert rows == 1


def test_upgrade_skips_columns_missing_from_a_reduced_schema(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE public_mcp_apps (
                    id INTEGER PRIMARY KEY,
                    app_id VARCHAR(100) NOT NULL UNIQUE,
                    name VARCHAR(200) NOT NULL,
                    transport VARCHAR(50) NOT NULL DEFAULT 'oauth',
                    launch_config JSON
                )
                """
            )
        )
        _run(connection, migration, "upgrade")
        assert "whatsapp" in _app_ids(connection)


def test_upgrade_preserves_a_preexisting_row_with_the_same_app_id(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_apps_table(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, description, transport, is_visible_in_connector) "
                "VALUES ('whatsapp', 'Operator WhatsApp', 'hand-made', 'oauth', 0)"
            )
        )
        _run(connection, migration, "upgrade")
        row = connection.execute(
            text(
                "SELECT COUNT(*), MIN(name), MIN(is_visible_in_connector) "
                "FROM public_mcp_apps WHERE app_id='whatsapp'"
            )
        ).first()
        assert row == (1, "Operator WhatsApp", 0)


def test_seed_row_matches_registry():
    """The migration snapshot and the runtime registry must define the same
    whatsapp row (the migration is a frozen copy; this catches drift)."""
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    migration = _load_migration_module()
    registry_row = next(
        r for r in get_builtin_public_mcp_app_rows() if r["app_id"] == "whatsapp"
    )
    assert migration.ROW == registry_row
    assert registry_row["is_visible_in_connector"] is True


def test_downgrade_removes_whatsapp(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_apps_table(connection)
        _run(connection, migration, "upgrade", "downgrade")
        assert "whatsapp" not in _app_ids(connection)


def test_downgrade_preserves_a_preexisting_operator_row(tmp_path):
    """upgrade() skips a row that already occupies the app_id, so downgrade()
    must not delete it either -- it is the operator's, not this migration's."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_apps_table(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, description, transport, provider_name) "
                "VALUES ('whatsapp', 'WhatsApp Business', 'hand-made', "
                "'oauth', 'meta')"
            )
        )
        _run(connection, migration, "upgrade", "downgrade")
        row = connection.execute(
            text(
                "SELECT name, description FROM public_mcp_apps WHERE app_id='whatsapp'"
            )
        ).first()
        assert row == ("WhatsApp Business", "hand-made")


def test_downgrade_preserves_row_admin_edited_beyond_structural_fields(tmp_path):
    """description and is_visible_in_connector are admin-PATCHable on builtin
    rows; a migration-created row an admin then hid must survive downgrade
    rather than be treated as still 'ours' on a name/transport-only match."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_apps_table(connection)
        _run(connection, migration, "upgrade")
        connection.execute(
            text(
                "UPDATE public_mcp_apps SET is_visible_in_connector=0 "
                "WHERE app_id='whatsapp'"
            )
        )
        _run(connection, migration, "downgrade")
        assert "whatsapp" in _app_ids(connection)


def test_downgrade_is_a_noop_when_a_guard_column_is_missing(tmp_path):
    """A reduced-schema table missing a guard column must not make the guard
    SELECT reference a nonexistent column and raise, and must not fall back
    to a weaker match on the remaining columns either."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE public_mcp_apps (
                    id INTEGER PRIMARY KEY,
                    app_id VARCHAR(100) NOT NULL UNIQUE,
                    name VARCHAR(200) NOT NULL,
                    transport VARCHAR(50) NOT NULL DEFAULT 'oauth',
                    launch_config JSON
                )
                """
            )
        )
        _run(connection, migration, "upgrade", "downgrade")
        assert "whatsapp" in _app_ids(connection)


def test_upgrade_and_downgrade_no_op_without_table(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _run(connection, migration, "upgrade", "downgrade")
        table_names = set(
            connection.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            ).scalars()
        )
        assert "public_mcp_apps" not in table_names
