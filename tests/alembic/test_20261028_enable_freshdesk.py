"""Tests for showing the Freshdesk connector in the catalog."""

import importlib.util
import sqlite3
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20261028_enable_freshdesk.py"
)
SEED_MIGRATION_PATH = MIGRATION_PATH.with_name("20260923_seed_freshdesk_mcp_app.py")


def _load_migration_module(path: Path = MIGRATION_PATH):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection) -> Operations:
    return Operations(MigrationContext.configure(connection))


def _public_mcp_apps(metadata: sa.MetaData) -> sa.Table:
    return sa.Table(
        "public_mcp_apps",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("app_id", sa.String(100), nullable=False, unique=True),
        sa.Column("is_visible_in_connector", sa.Boolean, nullable=False),
    )


def test_upgrade_enables_freshdesk_idempotently(tmp_path) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    table = _public_mcp_apps(metadata)
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(
            sa.insert(table),
            [
                {"app_id": "freshdesk", "is_visible_in_connector": False},
                {"app_id": "other", "is_visible_in_connector": False},
            ],
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()
        rows = {
            row["app_id"]: dict(row)
            for row in connection.execute(sa.select(table)).mappings()
        }

    assert rows["freshdesk"]["is_visible_in_connector"] is True
    assert rows["other"]["is_visible_in_connector"] is False


def test_full_downgrade_keeps_a_row_an_operator_enabled(tmp_path) -> None:
    """Hiding the row on downgrade would make it match the seed snapshot
    again, so the seed migration's downgrade would then delete a row an
    operator had made visible before this migration ran."""
    seed = _load_migration_module(SEED_MIGRATION_PATH)
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    table = sa.Table(
        "public_mcp_apps",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        *(
            sa.Column(column.name, column.type)
            for column in seed.PUBLIC_MCP_APPS_TABLE.c
        ),
    )
    metadata.create_all(engine)
    select_rows = sa.select(table.c.app_id, table.c.is_visible_in_connector)

    with engine.begin() as connection:
        with patch.object(seed, "op", _operations(connection)):
            seed.upgrade()
        connection.execute(
            sa.update(table)
            .where(table.c.app_id == "freshdesk")
            .values(is_visible_in_connector=True)
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        assert connection.execute(select_rows).all() == [("freshdesk", True)]

        with patch.object(seed, "op", _operations(connection)):
            seed.downgrade()
        assert connection.execute(select_rows).all() == [("freshdesk", True)]


def test_upgrade_is_noop_when_catalog_columns_are_missing() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")
    metadata = sa.MetaData()
    table = sa.Table(
        "public_mcp_apps",
        metadata,
        sa.Column("app_id", sa.String(100), primary_key=True),
    )
    metadata.create_all(engine)

    with (
        engine.begin() as connection,
        patch.object(migration, "op", _operations(connection)),
    ):
        connection.execute(sa.insert(table), {"app_id": "freshdesk"})
        migration.upgrade()
        assert connection.execute(sa.select(table)).all() == [("freshdesk",)]


def test_upgrade_is_noop_without_catalog_table() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with (
        engine.begin() as connection,
        patch.object(migration, "op", _operations(connection)),
    ):
        migration.upgrade()
        assert "public_mcp_apps" not in sa.inspect(connection).get_table_names()


def test_offline_sqlite_upgrade_round_trips_visibility() -> None:
    migration = _load_migration_module()
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="sqlite",
        opts={"as_sql": True, "output_buffer": output},
    )
    with Operations.context(context):
        migration.upgrade()

    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE public_mcp_apps (app_id TEXT PRIMARY KEY, is_visible_in_connector BOOLEAN)"
    )
    connection.execute("INSERT INTO public_mcp_apps VALUES (?, ?)", ("freshdesk", 0))
    connection.executescript(output.getvalue())
    row = connection.execute(
        "SELECT is_visible_in_connector FROM public_mcp_apps WHERE app_id = 'freshdesk'"
    ).fetchone()
    connection.close()

    assert row == (1,)


def test_offline_postgresql_upgrade_renders_true_literal() -> None:
    migration = _load_migration_module()
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="postgresql", opts={"as_sql": True, "output_buffer": output}
    )
    with Operations.context(context):
        migration.upgrade()

    sql = output.getvalue()
    assert "UPDATE public_mcp_apps SET is_visible_in_connector=true" in sql
    assert "app_id = 'freshdesk'" in sql
    assert "%(" not in sql


def test_offline_downgrade_emits_no_sql() -> None:
    migration = _load_migration_module()
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="sqlite",
        opts={"as_sql": True, "output_buffer": output},
    )
    with Operations.context(context):
        migration.downgrade()

    assert output.getvalue() == ""


def test_revision_metadata() -> None:
    migration = _load_migration_module()

    assert migration.revision == "20261028_enable_freshdesk"
    assert migration.down_revision == "20261029_update_chartmogul_description"


def test_registry_default_is_visible() -> None:
    """A fresh install seeds from the registry, an existing one is flipped by
    this migration; both must end up visible."""
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app

    app = get_builtin_public_mcp_app("freshdesk")
    assert app is not None
    assert app["is_visible_in_connector"] is True
