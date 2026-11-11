"""Tests for renaming the Atlassian connector away from Jira."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

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
        "update_atlassian_name_and_description_migration",
        "src/xagent/migrations/versions/"
        "20261105_update_atlassian_name_and_description.py",
    )


def _load_seed_migration_module():
    return _load_module(
        "seed_atlassian_migration",
        "src/xagent/migrations/versions/20260920_seed_atlassian_mcp_app.py",
    )


def _operations(connection):
    return Operations(MigrationContext.configure(connection))


def _run(connection, migration, *steps):
    with patch.object(migration, "op", _operations(connection)):
        for step in steps:
            getattr(migration, step)()


def _create_table(connection):
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


def _insert(connection, app_id: str, name: str, description: str | None):
    connection.execute(
        text(
            "INSERT INTO public_mcp_apps (app_id, name, description) "
            "VALUES (:app_id, :name, :description)"
        ),
        {"app_id": app_id, "name": name, "description": description},
    )


def _name_and_description(connection, app_id: str = "atlassian"):
    return tuple(
        connection.execute(
            text(
                "SELECT name, description FROM public_mcp_apps WHERE app_id = :app_id"
            ),
            {"app_id": app_id},
        ).one()
    )


def test_upgrade_updates_name_and_description(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        _insert(
            connection,
            migration.APP_ID,
            migration.PREVIOUS_NAME,
            migration.PREVIOUS_DESCRIPTION,
        )
        _run(connection, migration, "upgrade")
        assert _name_and_description(connection) == (
            migration.CURRENT_NAME,
            migration.CURRENT_DESCRIPTION,
        )


def test_upgrade_is_idempotent_and_downgrade_restores_previous(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        _insert(
            connection,
            migration.APP_ID,
            migration.PREVIOUS_NAME,
            migration.PREVIOUS_DESCRIPTION,
        )
        _run(connection, migration, "upgrade", "upgrade", "downgrade")
        assert _name_and_description(connection) == (
            migration.PREVIOUS_NAME,
            migration.PREVIOUS_DESCRIPTION,
        )


def test_upgrade_renames_even_when_admin_customized_description(tmp_path):
    """name is code-owned (admins cannot PATCH it), so a customized
    description must not hold the name back: a stale name would be reported
    as built-in catalog drift on every startup."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        _insert(
            connection,
            migration.APP_ID,
            migration.PREVIOUS_NAME,
            "Our Atlassian connector",
        )
        _run(connection, migration, "upgrade")
        assert _name_and_description(connection) == (
            migration.CURRENT_NAME,
            "Our Atlassian connector",
        )

        _run(connection, migration, "downgrade")
        assert _name_and_description(connection) == (
            migration.PREVIOUS_NAME,
            "Our Atlassian connector",
        )


def test_upgrade_leaves_an_unexpected_name_alone(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        _insert(
            connection,
            migration.APP_ID,
            "Atlassian Cloud",
            migration.PREVIOUS_DESCRIPTION,
        )
        _run(connection, migration, "upgrade")
        assert _name_and_description(connection) == (
            "Atlassian Cloud",
            migration.CURRENT_DESCRIPTION,
        )


def test_downgrade_preserves_admin_customized_description(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        _insert(
            connection,
            migration.APP_ID,
            migration.CURRENT_NAME,
            "Custom description",
        )
        _run(connection, migration, "downgrade")
        assert _name_and_description(connection) == (
            migration.PREVIOUS_NAME,
            "Custom description",
        )


def test_upgrade_without_matching_row_is_a_noop(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        _insert(
            connection,
            "other-app",
            migration.PREVIOUS_NAME,
            migration.PREVIOUS_DESCRIPTION,
        )
        _run(connection, migration, "upgrade")
        assert _name_and_description(connection, "other-app") == (
            migration.PREVIOUS_NAME,
            migration.PREVIOUS_DESCRIPTION,
        )


def test_upgrade_without_required_schema_is_a_noop(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _run(connection, migration, "upgrade", "downgrade")

        connection.execute(
            text("CREATE TABLE public_mcp_apps (id INTEGER PRIMARY KEY, app_id TEXT)")
        )
        _run(connection, migration, "upgrade", "downgrade")


def test_name_is_renamed_when_the_description_column_is_absent(tmp_path):
    """Each column is guarded on its own, so a schema without description
    must still have its name renamed and restored."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE public_mcp_apps "
                "(id INTEGER PRIMARY KEY, app_id TEXT, name TEXT)"
            )
        )
        connection.execute(
            text("INSERT INTO public_mcp_apps (app_id, name) VALUES (:app_id, :name)"),
            {"app_id": migration.APP_ID, "name": migration.PREVIOUS_NAME},
        )

        def name():
            return connection.execute(
                text("SELECT name FROM public_mcp_apps WHERE app_id = :app_id"),
                {"app_id": migration.APP_ID},
            ).scalar_one()

        _run(connection, migration, "upgrade")
        assert name() == migration.CURRENT_NAME

        _run(connection, migration, "downgrade")
        assert name() == migration.PREVIOUS_NAME


def test_full_downgrade_through_the_seed_removes_the_row(tmp_path):
    """The seed's downgrade deletes the row only while it still matches the
    frozen seed snapshot, so this migration's downgrade must restore both
    columns exactly or the row is orphaned (the deputy bug, PR #2449)."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    seed_migration = _load_seed_migration_module()
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        _run(connection, seed_migration, "upgrade")
        _run(connection, migration, "upgrade")
        assert _name_and_description(connection) == (
            migration.CURRENT_NAME,
            migration.CURRENT_DESCRIPTION,
        )

        _run(connection, migration, "downgrade")
        _run(connection, seed_migration, "downgrade")
        remaining = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps")
        ).scalar_one()
        assert remaining == 0


def test_previous_values_match_seed_and_current_values_match_registry():
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    migration = _load_migration_module()
    seed_migration = _load_seed_migration_module()
    assert seed_migration.ROW["name"] == migration.PREVIOUS_NAME
    assert seed_migration.ROW["description"] == migration.PREVIOUS_DESCRIPTION

    registry_row = next(
        row
        for row in get_builtin_public_mcp_app_rows()
        if row["app_id"] == migration.APP_ID
    )
    # The name has since been superseded by 20261110_rename_atlassian_to_confluence
    # (see its own test); the description is still the one set here.
    assert registry_row["description"] == migration.CURRENT_DESCRIPTION
    assert "Jira" not in migration.CURRENT_NAME


def test_down_revision_matches_current_head():
    migration = _load_migration_module()

    assert migration.down_revision == "20261031_task_sdk_credential_kind"
