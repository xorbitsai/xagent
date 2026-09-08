"""Tests for updating the Google Drive connector's description and category."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text


def _load_migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20260908_update_google_drive_description_category.py"
    )
    spec = importlib.util.spec_from_file_location(
        "update_google_drive_description_category_migration", migration_file
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection):
    return Operations(MigrationContext.configure(connection))


def _create_table(
    connection,
    description: str,
    category: str,
    with_description_column: bool = True,
    with_category_column: bool = True,
):
    description_column = "description TEXT," if with_description_column else ""
    category_column = "category VARCHAR(100)," if with_category_column else ""
    connection.execute(
        text(
            f"""
            CREATE TABLE public_mcp_apps (
                id INTEGER PRIMARY KEY,
                app_id VARCHAR(100) NOT NULL UNIQUE,
                {description_column}
                {category_column}
                oauth_scopes JSON
            )
            """
        )
    )
    columns = ["app_id"]
    values = {"app_id": "google-drive"}
    params = [":app_id"]
    if with_description_column:
        columns.append("description")
        params.append(":description")
        values["description"] = description
    if with_category_column:
        columns.append("category")
        params.append(":category")
        values["category"] = category
    connection.execute(
        text(
            f"INSERT INTO public_mcp_apps ({', '.join(columns)}) "
            f"VALUES ({', '.join(params)})"
        ),
        values,
    )


def _row(connection):
    row = connection.execute(
        text(
            "SELECT description, category FROM public_mcp_apps "
            "WHERE app_id='google-drive'"
        )
    ).first()
    return row[0], row[1]


def test_upgrade_updates_description_and_category(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(
            connection,
            description=migration.PREVIOUS_DESCRIPTION,
            category=migration.PREVIOUS_CATEGORY,
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        description, category = _row(connection)
        assert description == migration.CURRENT_DESCRIPTION
        assert category == migration.CURRENT_CATEGORY


def test_upgrade_is_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(
            connection,
            description=migration.PREVIOUS_DESCRIPTION,
            category=migration.PREVIOUS_CATEGORY,
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()
        description, category = _row(connection)
        assert description == migration.CURRENT_DESCRIPTION
        assert category == migration.CURRENT_CATEGORY


def test_downgrade_restores_previous_description_and_category(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(
            connection,
            description=migration.PREVIOUS_DESCRIPTION,
            category=migration.PREVIOUS_CATEGORY,
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        description, category = _row(connection)
        assert description == migration.PREVIOUS_DESCRIPTION
        assert category == migration.PREVIOUS_CATEGORY


def test_upgrade_downgrade_upgrade_round_trip(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(
            connection,
            description=migration.PREVIOUS_DESCRIPTION,
            category=migration.PREVIOUS_CATEGORY,
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
            migration.upgrade()
        description, category = _row(connection)
        assert description == migration.CURRENT_DESCRIPTION
        assert category == migration.CURRENT_CATEGORY


def test_upgrade_preserves_admin_customized_description(tmp_path):
    """description is not in _BUILTIN_PROTECTED_FIELDS (admin_mcp.py), so an
    operator can have edited it via the admin PATCH endpoint. The migration
    must not clobber a value that no longer matches the last-known default.
    """
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(
            connection,
            description="Our internal Drive connector",
            category=migration.PREVIOUS_CATEGORY,
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        description, category = _row(connection)
        assert description == "Our internal Drive connector"
        assert category == migration.CURRENT_CATEGORY


def test_upgrade_preserves_admin_customized_category(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(
            connection,
            description=migration.PREVIOUS_DESCRIPTION,
            category="Productivity",
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        description, category = _row(connection)
        assert description == migration.CURRENT_DESCRIPTION
        assert category == "Productivity"


def test_downgrade_preserves_admin_customized_description(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(
            connection,
            description=migration.PREVIOUS_DESCRIPTION,
            category=migration.PREVIOUS_CATEGORY,
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            connection.execute(
                text(
                    "UPDATE public_mcp_apps SET description = :d "
                    "WHERE app_id = 'google-drive'"
                ),
                {"d": "Our internal Drive connector"},
            )
            migration.downgrade()
        description, category = _row(connection)
        assert description == "Our internal Drive connector"
        assert category == migration.PREVIOUS_CATEGORY


def test_downgrade_preserves_admin_customized_category(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(
            connection,
            description=migration.PREVIOUS_DESCRIPTION,
            category=migration.PREVIOUS_CATEGORY,
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            connection.execute(
                text(
                    "UPDATE public_mcp_apps SET category = :c "
                    "WHERE app_id = 'google-drive'"
                ),
                {"c": "Productivity"},
            )
            migration.downgrade()
        description, category = _row(connection)
        assert description == migration.PREVIOUS_DESCRIPTION
        assert category == "Productivity"


def test_upgrade_without_description_column_still_updates_category(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(
            connection,
            description=migration.PREVIOUS_DESCRIPTION,
            category=migration.PREVIOUS_CATEGORY,
            with_description_column=False,
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()  # must not raise when description is missing
        category = connection.execute(
            text("SELECT category FROM public_mcp_apps WHERE app_id='google-drive'")
        ).scalar()
        assert category == migration.CURRENT_CATEGORY


def test_upgrade_without_category_column_still_updates_description(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(
            connection,
            description=migration.PREVIOUS_DESCRIPTION,
            category=migration.PREVIOUS_CATEGORY,
            with_category_column=False,
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()  # must not raise when category is missing
        description = connection.execute(
            text("SELECT description FROM public_mcp_apps WHERE app_id='google-drive'")
        ).scalar()
        assert description == migration.CURRENT_DESCRIPTION


def test_upgrade_without_table_is_a_noop(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()  # must not raise when the table doesn't exist


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
                    category VARCHAR(100),
                    oauth_scopes JSON
                )
                """
            )
        )
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps (app_id, description, category) "
                "VALUES ('onedrive', 'unrelated', 'Storage')"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        row = connection.execute(
            text(
                "SELECT description, category FROM public_mcp_apps "
                "WHERE app_id='onedrive'"
            )
        ).first()
        assert row[0] == "unrelated"
        assert row[1] == "Storage"


def test_migration_fields_match_registry():
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    migration = _load_migration_module()
    registry_row = next(
        r for r in get_builtin_public_mcp_app_rows() if r["app_id"] == "google-drive"
    )
    assert registry_row["description"] == migration.CURRENT_DESCRIPTION
    assert registry_row["category"] == migration.CURRENT_CATEGORY
