"""Tests for updating already-seeded Word catalog rows."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text


def _load_migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20260921_update_word_contract.py"
    )
    spec = importlib.util.spec_from_file_location(
        "update_word_contract_migration", migration_file
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_seed_migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20260917_seed_word_mcp_app.py"
    )
    spec = importlib.util.spec_from_file_location(
        "seed_word_migration_for_contract_test", migration_file
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection):
    return Operations(MigrationContext.configure(connection))


def _create_table(connection):
    connection.execute(
        text(
            """
            CREATE TABLE public_mcp_apps (
                app_id VARCHAR(100) PRIMARY KEY,
                description TEXT,
                launch_config JSON
            )
            """
        )
    )


def test_upgrade_updates_only_owned_word_row(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        launch_config = {
            "command": "python",
            "builtin_provenance": migration.BUILTIN_PROVENANCE,
        }
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, description, launch_config) "
                "VALUES ('word', :description, :launch_config)"
            ),
            {
                "description": migration.OLD_DESCRIPTION,
                "launch_config": json.dumps(launch_config),
            },
        )

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        row = connection.execute(
            text(
                "SELECT description, launch_config FROM public_mcp_apps "
                "WHERE app_id='word'"
            )
        ).first()
        assert row[0] == migration.NEW_DESCRIPTION
        assert migration.OUTPUT_LIMIT_ENV in str(row[1])


def test_upgrade_preserves_unowned_word_row(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, description, launch_config) "
                "VALUES ('word', 'Operator description', '{\"command\": \"custom\"}')"
            )
        )

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        row = connection.execute(
            text(
                "SELECT description, launch_config FROM public_mcp_apps "
                "WHERE app_id='word'"
            )
        ).first()
        assert row[0] == "Operator description"
        assert migration.OUTPUT_LIMIT_ENV not in str(row[1])


def test_downgrade_restores_owned_word_contract(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        launch_config = {
            "command": "python",
            "static_env": {migration.OUTPUT_LIMIT_ENV: migration.OUTPUT_LIMIT_ENV},
            "builtin_provenance": migration.BUILTIN_PROVENANCE,
        }
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, description, launch_config) "
                "VALUES ('word', :description, :launch_config)"
            ),
            {
                "description": migration.NEW_DESCRIPTION,
                "launch_config": json.dumps(launch_config),
            },
        )

        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()

        row = connection.execute(
            text(
                "SELECT description, launch_config FROM public_mcp_apps "
                "WHERE app_id='word'"
            )
        ).first()
        assert row[0] == migration.OLD_DESCRIPTION
        assert migration.OUTPUT_LIMIT_ENV not in str(row[1])


def test_upgrade_preserves_customized_description_on_owned_row(tmp_path):
    """A provenance-owned row's launch_config contract must still upgrade
    even when an admin has customized its description (description is not a
    protected builtin field, so it can diverge from either canonical value)
    -- and that customization must survive, not be silently overwritten."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        launch_config = {
            "command": "python",
            "builtin_provenance": migration.BUILTIN_PROVENANCE,
        }
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, description, launch_config) "
                "VALUES ('word', :description, :launch_config)"
            ),
            {
                "description": "An operator's own custom description",
                "launch_config": json.dumps(launch_config),
            },
        )

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        row = connection.execute(
            text(
                "SELECT description, launch_config FROM public_mcp_apps "
                "WHERE app_id='word'"
            )
        ).first()
        assert row[0] == "An operator's own custom description"
        assert migration.OUTPUT_LIMIT_ENV in str(row[1])


def test_downgrade_preserves_customized_description_on_owned_row(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        launch_config = {
            "command": "python",
            "static_env": {migration.OUTPUT_LIMIT_ENV: migration.OUTPUT_LIMIT_ENV},
            "builtin_provenance": migration.BUILTIN_PROVENANCE,
        }
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, description, launch_config) "
                "VALUES ('word', :description, :launch_config)"
            ),
            {
                "description": "An operator's own custom description",
                "launch_config": json.dumps(launch_config),
            },
        )

        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()

        row = connection.execute(
            text(
                "SELECT description, launch_config FROM public_mcp_apps "
                "WHERE app_id='word'"
            )
        ).first()
        assert row[0] == "An operator's own custom description"
        assert migration.OUTPUT_LIMIT_ENV not in str(row[1])


def test_fresh_upgrade_then_downgrade_restores_down_revision_exactly(tmp_path):
    """The follow-up revision must downgrade to the frozen seed contract.

    This catches the subtle case where both migrations seed the newest values:
    upgrading a fresh database looks correct, but downgrading one revision then
    invents state that never existed at its down_revision.
    """
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    seed = _load_seed_migration_module()
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(seed, "op", _operations(connection)):
            seed.upgrade()
        seeded = connection.execute(
            text(
                "SELECT description, launch_config FROM public_mcp_apps "
                "WHERE app_id='word'"
            )
        ).first()

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            upgraded = connection.execute(
                text(
                    "SELECT description, launch_config FROM public_mcp_apps "
                    "WHERE app_id='word'"
                )
            ).first()
            assert upgraded[0] == migration.NEW_DESCRIPTION
            assert migration.OUTPUT_LIMIT_ENV in str(upgraded[1])
            migration.downgrade()

        downgraded = connection.execute(
            text(
                "SELECT description, launch_config FROM public_mcp_apps "
                "WHERE app_id='word'"
            )
        ).first()
        assert downgraded == seeded
