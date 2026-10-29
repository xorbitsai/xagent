"""Tests for updating the ChartMogul connector description."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text


def _load_module(name: str, filename: str):
    migration_file = Path(__file__).parent.parent.parent / filename
    spec = importlib.util.spec_from_file_location(name, migration_file)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_migration_module():
    return _load_module(
        "update_chartmogul_description_migration",
        "src/xagent/migrations/versions/20261029_update_chartmogul_description.py",
    )


def _operations(connection):
    return Operations(MigrationContext.configure(connection))


def _create_table(connection, app_id: str, description: str | None):
    connection.execute(
        text(
            """
            CREATE TABLE public_mcp_apps (
                id INTEGER PRIMARY KEY,
                app_id VARCHAR(100) NOT NULL UNIQUE,
                description TEXT
            )
            """
        )
    )
    connection.execute(
        text(
            "INSERT INTO public_mcp_apps (app_id, description) "
            "VALUES (:app_id, :description)"
        ),
        {"app_id": app_id, "description": description},
    )


def _description(connection, app_id: str = "chartmogul"):
    return connection.execute(
        text("SELECT description FROM public_mcp_apps WHERE app_id = :app_id"),
        {"app_id": app_id},
    ).scalar()


def test_upgrade_updates_description(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, migration.APP_ID, migration.PREVIOUS_DESCRIPTION)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert _description(connection) == migration.CURRENT_DESCRIPTION


def test_upgrade_is_idempotent_and_downgrade_restores_previous(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, migration.APP_ID, migration.PREVIOUS_DESCRIPTION)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()
            migration.downgrade()
        assert _description(connection) == migration.PREVIOUS_DESCRIPTION


def test_upgrade_preserves_admin_customized_description(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, migration.APP_ID, "Our internal metrics connector")
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert _description(connection) == "Our internal metrics connector"


def test_downgrade_preserves_admin_customized_description(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, migration.APP_ID, migration.CURRENT_DESCRIPTION)
        connection.execute(
            text(
                "UPDATE public_mcp_apps SET description = :description "
                "WHERE app_id = :app_id"
            ),
            {"app_id": migration.APP_ID, "description": "Custom description"},
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()
        assert _description(connection) == "Custom description"


def test_upgrade_without_required_schema_is_a_noop(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        connection.execute(
            text("CREATE TABLE public_mcp_apps (id INTEGER PRIMARY KEY, app_id TEXT)")
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        columns = {
            column["name"]
            for column in sa.inspect(connection).get_columns("public_mcp_apps")
        }
        assert columns == {"id", "app_id"}


def test_upgrade_without_matching_row_is_a_noop(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, "other-app", migration.PREVIOUS_DESCRIPTION)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert _description(connection, "other-app") == migration.PREVIOUS_DESCRIPTION


def test_current_description_matches_registry_and_seed():
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    migration = _load_migration_module()
    seed_migration = _load_module(
        "seed_chartmogul_migration",
        "src/xagent/migrations/versions/20260901_seed_chartmogul_mcp_app.py",
    )
    registry_row = next(
        row
        for row in get_builtin_public_mcp_app_rows()
        if row["app_id"] == migration.APP_ID
    )
    assert registry_row["description"] == migration.CURRENT_DESCRIPTION
    assert seed_migration.ROW["description"] == migration.CURRENT_DESCRIPTION


def test_down_revision_matches_current_head():
    migration = _load_migration_module()

    assert migration.down_revision == "20261029_google_oauth_grants"
