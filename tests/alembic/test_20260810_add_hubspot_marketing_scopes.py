"""Tests for adding HubSpot Marketing Hub OAuth scopes and description."""

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
        / "src/xagent/migrations/versions/20260810_add_hubspot_marketing_scopes.py"
    )
    spec = importlib.util.spec_from_file_location(
        "add_hubspot_marketing_scopes_migration", migration_file
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
                id INTEGER PRIMARY KEY,
                app_id VARCHAR(100) NOT NULL UNIQUE,
                description TEXT,
                oauth_scopes JSON
            )
            """
        )
    )
    connection.execute(
        text(
            "INSERT INTO public_mcp_apps (app_id, description, oauth_scopes) "
            "VALUES ('hubspot', 'old description', "
            '\'["crm.objects.contacts.read", "crm.objects.contacts.write", '
            '"crm.objects.companies.read", "crm.objects.companies.write", '
            '"crm.objects.deals.read"]\')'
        )
    )


def _row(connection):
    row = connection.execute(
        text(
            "SELECT description, oauth_scopes FROM public_mcp_apps "
            "WHERE app_id='hubspot'"
        )
    ).first()
    description, scopes = row[0], row[1]
    return description, json.loads(scopes) if isinstance(scopes, str) else scopes


def test_upgrade_adds_marketing_scopes_and_description(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        description, scopes = _row(connection)
        assert description == migration.CURRENT_DESCRIPTION
        assert scopes == migration.CURRENT_SCOPES


def test_upgrade_is_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()
        _, scopes = _row(connection)
        assert scopes == migration.CURRENT_SCOPES


def test_downgrade_restores_previous_scopes_and_description(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        description, scopes = _row(connection)
        assert description == migration.PREVIOUS_DESCRIPTION
        assert scopes == migration.PREVIOUS_SCOPES


def test_upgrade_without_table_is_a_noop(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()  # must not raise when the table doesn't exist


def test_migration_fields_match_registry():
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    migration = _load_migration_module()
    registry_row = next(
        r for r in get_builtin_public_mcp_app_rows() if r["app_id"] == "hubspot"
    )
    assert migration.CURRENT_SCOPES == registry_row["oauth_scopes"]
    assert migration.CURRENT_DESCRIPTION == registry_row["description"]
