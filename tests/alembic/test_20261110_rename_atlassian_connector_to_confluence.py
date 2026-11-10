"""Tests for renaming the Atlassian connector's display name to Confluence."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text

MIGRATION_FILE = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions"
    / "20261110_rename_atlassian_connector_to_confluence.py"
)


def _load_migration_module():
    spec = importlib.util.spec_from_file_location(
        "rename_atlassian_to_confluence_migration", MIGRATION_FILE
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run(connection, migration, *steps):
    operations = Operations(MigrationContext.configure(connection))
    with patch.object(migration, "op", operations):
        for step in steps:
            getattr(migration, step)()


def _create_table(connection):
    connection.execute(
        text(
            "CREATE TABLE public_mcp_apps "
            "(id INTEGER PRIMARY KEY, app_id TEXT, name TEXT)"
        )
    )


def _insert(connection, app_id: str, name: str):
    connection.execute(
        text("INSERT INTO public_mcp_apps (app_id, name) VALUES (:a, :n)"),
        {"a": app_id, "n": name},
    )


def _name(connection, app_id: str = "atlassian"):
    return connection.execute(
        text("SELECT name FROM public_mcp_apps WHERE app_id = :a"), {"a": app_id}
    ).scalar_one()


def test_upgrade_renames_and_downgrade_restores(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        _insert(connection, migration.APP_ID, migration.PREVIOUS_NAME)

        _run(connection, migration, "upgrade", "upgrade")
        assert _name(connection) == "Confluence"

        _run(connection, migration, "downgrade")
        assert _name(connection) == migration.PREVIOUS_NAME


def test_upgrade_leaves_unexpected_name_and_other_apps_alone(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        _insert(connection, migration.APP_ID, "Our Atlassian")
        _insert(connection, "other-app", migration.PREVIOUS_NAME)

        _run(connection, migration, "upgrade")
        assert _name(connection) == "Our Atlassian"
        assert _name(connection, "other-app") == migration.PREVIOUS_NAME


def test_upgrade_without_required_schema_is_a_noop(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _run(connection, migration, "upgrade", "downgrade")
        connection.execute(
            text("CREATE TABLE public_mcp_apps (id INTEGER PRIMARY KEY)")
        )
        _run(connection, migration, "upgrade", "downgrade")


def test_previous_name_matches_prior_migration_and_current_matches_registry():
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    migration = _load_migration_module()
    prior_path = MIGRATION_FILE.with_name(
        "20261105_update_atlassian_name_and_description.py"
    )
    spec = importlib.util.spec_from_file_location("prior_atlassian", prior_path)
    prior = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(prior)
    assert migration.PREVIOUS_NAME == prior.CURRENT_NAME

    registry_row = next(
        row
        for row in get_builtin_public_mcp_app_rows()
        if row["app_id"] == migration.APP_ID
    )
    assert registry_row["name"] == migration.CURRENT_NAME


def test_down_revision_matches_previous_head():
    migration = _load_migration_module()
    assert migration.down_revision == "20261109_seed_jira_ticket_mcp_app"
