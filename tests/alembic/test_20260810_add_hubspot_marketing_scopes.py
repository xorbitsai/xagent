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


def _create_table(connection, description: str, with_description_column: bool = True):
    description_column = "description TEXT," if with_description_column else ""
    connection.execute(
        text(
            f"""
            CREATE TABLE public_mcp_apps (
                id INTEGER PRIMARY KEY,
                app_id VARCHAR(100) NOT NULL UNIQUE,
                {description_column}
                oauth_scopes JSON
            )
            """
        )
    )
    description_col = ", description" if with_description_column else ""
    description_val = ", :description" if with_description_column else ""
    connection.execute(
        text(
            f"INSERT INTO public_mcp_apps (app_id, oauth_scopes{description_col}) "
            f"VALUES ('hubspot', :scopes{description_val})"
        ),
        {
            "scopes": json.dumps(
                [
                    "crm.objects.contacts.read",
                    "crm.objects.contacts.write",
                    "crm.objects.companies.read",
                    "crm.objects.companies.write",
                    "crm.objects.deals.read",
                ]
            ),
            "description": description,
        },
    )


def _create_user_oauth_table(connection, with_refresh_token_column: bool = True):
    refresh_column = "refresh_token VARCHAR," if with_refresh_token_column else ""
    connection.execute(
        text(
            f"""
            CREATE TABLE user_oauth (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                provider VARCHAR(50) NOT NULL,
                {refresh_column}
                access_token VARCHAR NOT NULL
            )
            """
        )
    )
    refresh_col = ", refresh_token" if with_refresh_token_column else ""
    refresh_hubspot = ", 'old-hubspot-refresh'" if with_refresh_token_column else ""
    refresh_other = ", 'old-salesforce-refresh'" if with_refresh_token_column else ""
    connection.execute(
        text(
            f"INSERT INTO user_oauth (user_id, provider, access_token{refresh_col}) "
            f"VALUES (1, 'hubspot', 'old-hubspot-token'{refresh_hubspot}), "
            f"(1, 'salesforce', 'old-salesforce-token'{refresh_other})"
        )
    )


def _access_tokens(connection):
    return dict(
        connection.execute(text("SELECT provider, access_token FROM user_oauth")).all()
    )


def _refresh_tokens(connection):
    return dict(
        connection.execute(text("SELECT provider, refresh_token FROM user_oauth")).all()
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
        _create_table(connection, description=migration.PREVIOUS_DESCRIPTION)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        description, scopes = _row(connection)
        assert description == migration.CURRENT_DESCRIPTION
        assert scopes == migration.CURRENT_SCOPES


def test_upgrade_is_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, description=migration.PREVIOUS_DESCRIPTION)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()
        description, scopes = _row(connection)
        assert description == migration.CURRENT_DESCRIPTION
        assert scopes == migration.CURRENT_SCOPES


def test_downgrade_restores_previous_scopes_and_description(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, description=migration.PREVIOUS_DESCRIPTION)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        description, scopes = _row(connection)
        assert description == migration.PREVIOUS_DESCRIPTION
        assert scopes == migration.PREVIOUS_SCOPES


def test_upgrade_downgrade_upgrade_round_trip(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, description=migration.PREVIOUS_DESCRIPTION)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
            migration.upgrade()
        description, scopes = _row(connection)
        assert description == migration.CURRENT_DESCRIPTION
        assert scopes == migration.CURRENT_SCOPES


def test_upgrade_preserves_customized_description_already_at_current_scopes(tmp_path):
    """A row already migrated once (CURRENT_SCOPES) with a since-customized
    description must not have that customization overwritten on a repeat
    upgrade (e.g. a second alembic run, or upgrade-downgrade-upgrade landing
    back on CURRENT_SCOPES with the customization still in place)."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, description=migration.PREVIOUS_DESCRIPTION)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            connection.execute(
                text(
                    "UPDATE public_mcp_apps SET description = :d "
                    "WHERE app_id = 'hubspot'"
                ),
                {"d": "Our internal HubSpot connector"},
            )
            migration.upgrade()
        description, scopes = _row(connection)
        assert description == "Our internal HubSpot connector"
        assert scopes == migration.CURRENT_SCOPES


def test_upgrade_preserves_admin_customized_description(tmp_path):
    """description is not in _BUILTIN_PROTECTED_FIELDS (admin_mcp.py), so an
    operator can have edited it via the admin PATCH endpoint. The migration
    must not clobber a value that no longer matches the last-known default.
    """
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, description="Our internal HubSpot connector")
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        description, scopes = _row(connection)
        assert description == "Our internal HubSpot connector"
        assert scopes == migration.CURRENT_SCOPES


def test_downgrade_preserves_admin_customized_description(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, description=migration.PREVIOUS_DESCRIPTION)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            connection.execute(
                text(
                    "UPDATE public_mcp_apps SET description = :d "
                    "WHERE app_id = 'hubspot'"
                ),
                {"d": "Our internal HubSpot connector"},
            )
            migration.downgrade()
        description, scopes = _row(connection)
        assert description == "Our internal HubSpot connector"
        assert scopes == migration.PREVIOUS_SCOPES


def test_upgrade_without_description_column_still_updates_scopes(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(
            connection,
            description=migration.PREVIOUS_DESCRIPTION,
            with_description_column=False,
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()  # must not raise when description is missing
        scopes = connection.execute(
            text("SELECT oauth_scopes FROM public_mcp_apps WHERE app_id='hubspot'")
        ).scalar()
        assert json.loads(scopes) == migration.CURRENT_SCOPES


def test_upgrade_without_table_is_a_noop(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()  # must not raise when the table doesn't exist


def test_upgrade_invalidates_existing_hubspot_grant_only(tmp_path):
    """Both tokens must be cleared: refresh_oauth_token_if_needed re-mints an
    access token off refresh_token alone, so a surviving refresh_token would
    silently resurrect the old-scoped grant on the next tool call.
    """
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, description=migration.PREVIOUS_DESCRIPTION)
        _create_user_oauth_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        tokens = _access_tokens(connection)
        assert tokens["hubspot"] == ""
        assert tokens["salesforce"] == "old-salesforce-token"
        refresh_tokens = _refresh_tokens(connection)
        assert refresh_tokens["hubspot"] is None
        assert refresh_tokens["salesforce"] == "old-salesforce-refresh"


def test_upgrade_clears_access_token_without_refresh_token_column(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, description=migration.PREVIOUS_DESCRIPTION)
        _create_user_oauth_table(connection, with_refresh_token_column=False)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()  # must not raise when refresh_token is missing
        tokens = _access_tokens(connection)
        assert tokens["hubspot"] == ""


def test_upgrade_without_user_oauth_table_is_a_noop(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, description=migration.PREVIOUS_DESCRIPTION)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()  # must not raise when user_oauth doesn't exist


def test_downgrade_does_not_touch_user_oauth(tmp_path):
    """Cleared access tokens are gone for good; downgrade only reverts
    public_mcp_apps, mirroring the Facebook scope migration's approach."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, description=migration.PREVIOUS_DESCRIPTION)
        _create_user_oauth_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        tokens = _access_tokens(connection)
        assert tokens["hubspot"] == ""


def test_migration_fields_match_registry():
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    migration = _load_migration_module()
    registry_row = next(
        r for r in get_builtin_public_mcp_app_rows() if r["app_id"] == "hubspot"
    )
    assert migration.CURRENT_SCOPES == registry_row["oauth_scopes"]
    assert migration.CURRENT_DESCRIPTION == registry_row["description"]
