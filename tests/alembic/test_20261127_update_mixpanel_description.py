"""Tests for stating the Mixpanel profile-query role in its description."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text


def _load_migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20261127_update_mixpanel_description.py"
    )
    spec = importlib.util.spec_from_file_location(
        "update_mixpanel_description_migration", migration_file
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run(connection, migration, *steps):
    with patch.object(
        migration, "op", Operations(MigrationContext.configure(connection))
    ):
        for step in steps:
            getattr(migration, step)()


def _seed(connection, rows):
    connection.execute(
        text(
            "CREATE TABLE public_mcp_apps ("
            "id INTEGER PRIMARY KEY, app_id VARCHAR(100) NOT NULL UNIQUE, "
            "description TEXT)"
        )
    )
    for app_id, description in rows:
        connection.execute(
            text("INSERT INTO public_mcp_apps (app_id, description) VALUES (:a, :d)"),
            {"a": app_id, "d": description},
        )


def _description(connection, app_id="mixpanel"):
    return connection.execute(
        text("SELECT description FROM public_mcp_apps WHERE app_id = :a"),
        {"a": app_id},
    ).scalar()


def test_upgrade_and_downgrade_swap_the_builtin_description(tmp_path):
    migration = _load_migration_module()
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    with engine.begin() as connection:
        _seed(
            connection,
            [("mixpanel", migration.PREVIOUS_DESCRIPTION), ("posthog", "other")],
        )
        _run(connection, migration, "upgrade", "upgrade")
        assert _description(connection) == migration.CURRENT_DESCRIPTION
        assert _description(connection, "posthog") == "other"
        _run(connection, migration, "downgrade")
        assert _description(connection) == migration.PREVIOUS_DESCRIPTION


def test_admin_edited_description_is_kept(tmp_path):
    migration = _load_migration_module()
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    with engine.begin() as connection:
        _seed(connection, [("mixpanel", "Our analytics")])
        _run(connection, migration, "upgrade")
        assert _description(connection) == "Our analytics"
        _run(connection, migration, "downgrade")
        assert _description(connection) == "Our analytics"


def test_missing_table_is_a_noop(tmp_path):
    migration = _load_migration_module()
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    with engine.begin() as connection:
        _run(connection, migration, "upgrade", "downgrade")


def test_migration_matches_registry():
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    migration = _load_migration_module()
    registry_row = next(
        r for r in get_builtin_public_mcp_app_rows() if r["app_id"] == "mixpanel"
    )
    assert registry_row["description"] == migration.CURRENT_DESCRIPTION
