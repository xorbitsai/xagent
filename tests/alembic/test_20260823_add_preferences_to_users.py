"""Tests for the users.preferences column migration."""

import importlib.util
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20260823_add_preferences_to_users.py"
)
TABLE = "users"
COLUMN = "preferences"


def _migration_module():
    spec = importlib.util.spec_from_file_location(
        "user_preferences_migration", MIGRATION_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _operations(connection: sa.Connection) -> Operations:
    return Operations(MigrationContext.configure(connection))


def _offline_sql(migration, dialect_name: str, operation: str) -> str:
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name=dialect_name,
        opts={"as_sql": True, "output_buffer": output},
    )
    with Operations.context(context):
        getattr(migration, operation)()
    return output.getvalue()


def _legacy_users_table(engine) -> None:
    """Migration-only schema: just the columns this migration reads."""
    metadata = sa.MetaData()
    sa.Table(
        TABLE,
        metadata,
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("username", sa.String(50), nullable=False),
    )
    metadata.create_all(engine)


def _column_names(connection) -> set[str]:
    return {column["name"] for column in sa.inspect(connection).get_columns(TABLE)}


def test_online_upgrade_adds_a_nullable_column_and_downgrade_removes_it() -> None:
    migration = _migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        _legacy_users_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

            columns = {c["name"]: c for c in sa.inspect(connection).get_columns(TABLE)}
            assert COLUMN in columns
            assert columns[COLUMN]["nullable"] is True

            migration.downgrade()
            assert COLUMN not in _column_names(connection)


def test_online_upgrade_is_idempotent() -> None:
    migration = _migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        _legacy_users_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()

            columns = [c["name"] for c in sa.inspect(connection).get_columns(TABLE)]
            assert columns.count(COLUMN) == 1


def test_online_downgrade_is_idempotent_when_column_is_absent() -> None:
    migration = _migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        _legacy_users_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()

            assert COLUMN not in _column_names(connection)


def test_online_upgrade_and_downgrade_noop_without_the_users_table() -> None:
    migration = _migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()

        assert TABLE not in sa.inspect(connection).get_table_names()


@pytest.mark.parametrize("dialect_name", ["sqlite", "postgresql"])
def test_offline_upgrade_emits_plain_add_column_on_both_dialects(dialect_name) -> None:
    migration = _migration_module()

    with patch.object(
        migration.sa,
        "inspect",
        side_effect=AssertionError("offline branch must not reflect"),
    ):
        upgrade_sql = _offline_sql(migration, dialect_name, "upgrade")

    assert f"ALTER TABLE {TABLE} ADD COLUMN {COLUMN}" in upgrade_sql
    assert "CONCURRENTLY" not in upgrade_sql


@pytest.mark.parametrize("dialect_name", ["sqlite", "postgresql"])
def test_offline_downgrade_emits_plain_drop_column_on_both_dialects(
    dialect_name,
) -> None:
    migration = _migration_module()

    with patch.object(
        migration.sa,
        "inspect",
        side_effect=AssertionError("offline branch must not reflect"),
    ):
        downgrade_sql = _offline_sql(migration, dialect_name, "downgrade")

    assert f"ALTER TABLE {TABLE} DROP COLUMN {COLUMN}" in downgrade_sql
    assert "CONCURRENTLY" not in downgrade_sql


@pytest.mark.parametrize("dialect_name", ["sqlite", "postgresql"])
def test_offline_sql_carries_no_bind_parameters(dialect_name) -> None:
    migration = _migration_module()

    upgrade_sql = _offline_sql(migration, dialect_name, "upgrade")
    downgrade_sql = _offline_sql(migration, dialect_name, "downgrade")

    for sql in (upgrade_sql, downgrade_sql):
        assert "%(" not in sql
        assert ":table_name" not in sql


def test_target_schema_resolves_the_visible_users_relation() -> None:
    """version_table_schema names only the Alembic version table and
    current_schema() is merely the first search_path entry, so neither
    identifies the relation the unqualified DDL resolves to."""

    migration = _migration_module()
    sql = str(migration.POSTGRES_VISIBLE_TABLE_SCHEMA_SQL)

    assert "to_regclass" in sql
    assert "pg_catalog.pg_namespace" in sql

    # Non-PostgreSQL falls back to version_table_schema, and None keeps every
    # operation on plain unqualified behaviour.
    engine = sa.create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            assert migration._target_schema() is None


def test_target_schema_uses_the_catalog_answer_on_postgresql() -> None:
    """On PostgreSQL, ``_target_schema`` trusts the catalog lookup over
    ``version_table_schema`` whenever the catalog resolves the relation, and
    only falls back to the version table's configured schema when the
    catalog has nothing to say (e.g. the relation is not visible yet)."""

    migration = _migration_module()

    resolving_op = MagicMock()
    resolving_op.get_bind.return_value.dialect.name = "postgresql"
    resolving_op.get_bind.return_value.execute.return_value.scalar.return_value = (
        "tenant_x"
    )

    with patch.object(migration, "op", resolving_op):
        assert migration._target_schema() == "tenant_x"

    resolving_op.get_bind.return_value.execute.assert_called_once_with(
        migration.POSTGRES_VISIBLE_TABLE_SCHEMA_SQL, {"table_name": TABLE}
    )

    empty_catalog_op = MagicMock()
    empty_catalog_op.get_bind.return_value.dialect.name = "postgresql"
    empty_catalog_op.get_bind.return_value.execute.return_value.scalar.return_value = (
        None
    )
    empty_catalog_op.get_context.return_value.version_table_schema = "fallback_schema"

    with patch.object(migration, "op", empty_catalog_op):
        assert migration._target_schema() == "fallback_schema"
