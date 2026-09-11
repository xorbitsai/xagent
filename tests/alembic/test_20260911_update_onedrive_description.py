"""Tests for updating the OneDrive connector's description."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text


def _load_migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20260911_update_onedrive_description.py"
    )
    spec = importlib.util.spec_from_file_location(
        "update_onedrive_description_migration", migration_file
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
            f"INSERT INTO public_mcp_apps (app_id{description_col}) "
            f"VALUES ('onedrive'{description_val})"
        ),
        {"description": description},
    )


def _description(connection):
    return connection.execute(
        text("SELECT description FROM public_mcp_apps WHERE app_id='onedrive'")
    ).scalar()


def test_upgrade_updates_description(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, description=migration.PREVIOUS_DESCRIPTION)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert _description(connection) == migration.CURRENT_DESCRIPTION


def test_upgrade_is_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, description=migration.PREVIOUS_DESCRIPTION)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()
        assert _description(connection) == migration.CURRENT_DESCRIPTION


def test_downgrade_restores_previous_description(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, description=migration.PREVIOUS_DESCRIPTION)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        assert _description(connection) == migration.PREVIOUS_DESCRIPTION


def test_upgrade_downgrade_upgrade_round_trip(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, description=migration.PREVIOUS_DESCRIPTION)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
            migration.upgrade()
        assert _description(connection) == migration.CURRENT_DESCRIPTION


def test_upgrade_preserves_admin_customized_description(tmp_path):
    """description is not in _BUILTIN_PROTECTED_FIELDS (admin_mcp.py), so an
    operator can have edited it via the admin PATCH endpoint. The migration
    must not clobber a value that no longer matches the last-known default.
    """
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, description="Our internal storage connector")
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert _description(connection) == "Our internal storage connector"


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
                    "WHERE app_id = 'onedrive'"
                ),
                {"d": "Our internal storage connector"},
            )
            migration.downgrade()
        assert _description(connection) == "Our internal storage connector"


def test_upgrade_without_description_column_is_a_noop(tmp_path):
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
        # The column genuinely doesn't exist -- confirms the no-op above
        # actually exercised the missing-column guard, not a coincidence.
        columns = {
            c["name"] for c in sa.inspect(connection).get_columns("public_mcp_apps")
        }
        assert "description" not in columns


def test_upgrade_without_app_id_column_is_a_noop(tmp_path):
    """_columns_present requires BOTH app_id and description; only the
    description-missing half was covered above."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE public_mcp_apps (
                    id INTEGER PRIMARY KEY,
                    description TEXT
                )
                """
            )
        )
        connection.execute(
            text("INSERT INTO public_mcp_apps (description) VALUES (:d)"),
            {"d": migration.PREVIOUS_DESCRIPTION},
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()  # must not raise when app_id is missing
        description = connection.execute(
            text("SELECT description FROM public_mcp_apps")
        ).scalar()
        assert description == migration.PREVIOUS_DESCRIPTION


def test_upgrade_without_table_is_a_noop(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()  # must not raise when the table doesn't exist
        tables = set(sa.inspect(connection).get_table_names())
        assert "public_mcp_apps" not in tables


def test_upgrade_without_matching_row_is_a_noop(tmp_path):
    """A different app_id in the table must be untouched, and no matching
    row is not a customization to protect -- it's simply nothing to do."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
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
                "INSERT INTO public_mcp_apps (app_id, description) "
                "VALUES ('google-drive', 'unrelated')"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        description = connection.execute(
            text("SELECT description FROM public_mcp_apps WHERE app_id='google-drive'")
        ).scalar()
        assert description == "unrelated"


def test_migration_fields_match_registry():
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    migration = _load_migration_module()
    registry_row = next(
        r for r in get_builtin_public_mcp_app_rows() if r["app_id"] == "onedrive"
    )
    assert registry_row["description"] == migration.CURRENT_DESCRIPTION
