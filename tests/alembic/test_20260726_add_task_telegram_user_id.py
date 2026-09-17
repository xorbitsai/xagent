"""Tests for the Telegram sender ownership migration."""

import importlib.util
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

MIGRATION_PATH = (
    Path(__file__).parents[2]
    / "src/xagent/migrations/versions/20260726_add_task_telegram_user_id.py"
)
TABLE = "tasks"
COLUMN = "telegram_user_id"
INDEX = "ix_tasks_telegram_user_id"


def _migration_module():
    spec = importlib.util.spec_from_file_location(
        "telegram_task_owner_migration", MIGRATION_PATH
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


def _transactional_offline_sql(migration, dialect_name: str, operation: str) -> str:
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name=dialect_name,
        opts={"as_sql": True, "output_buffer": output},
    )
    with Operations.context(context), context.begin_transaction():
        getattr(migration, operation)()
    return output.getvalue()


def test_migration_adds_and_removes_telegram_user_id() -> None:
    migration = _migration_module()
    engine = sa.create_engine("sqlite:///:memory:")
    metadata = sa.MetaData()
    sa.Table(
        TABLE,
        metadata,
        sa.Column("id", sa.Integer(), primary_key=True),
    )
    metadata.create_all(engine)

    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()

            inspector = sa.inspect(connection)
            assert COLUMN in {column["name"] for column in inspector.get_columns(TABLE)}
            assert INDEX in {index["name"] for index in inspector.get_indexes(TABLE)}

            migration.downgrade()
            inspector = sa.inspect(connection)
            assert COLUMN not in {
                column["name"] for column in inspector.get_columns(TABLE)
            }


def test_migration_noops_without_tasks_table() -> None:
    migration = _migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()

        assert TABLE not in sa.inspect(connection).get_table_names()


def test_sqlite_offline_upgrade_and_downgrade_emit_plain_ddl() -> None:
    migration = _migration_module()

    upgrade_sql = _offline_sql(migration, "sqlite", "upgrade")
    downgrade_sql = _offline_sql(migration, "sqlite", "downgrade")

    assert f"ALTER TABLE {TABLE} ADD COLUMN {COLUMN} VARCHAR(32)" in upgrade_sql
    assert f"CREATE INDEX {INDEX} ON {TABLE} ({COLUMN})" in upgrade_sql
    assert "CONCURRENTLY" not in upgrade_sql
    assert f"DROP INDEX {INDEX}" in downgrade_sql
    assert "CONCURRENTLY" not in downgrade_sql


def test_postgresql_offline_upgrade_and_downgrade_emit_concurrent_index_sql() -> None:
    migration = _migration_module()

    upgrade_sql = _offline_sql(migration, "postgresql", "upgrade")
    downgrade_sql = _offline_sql(migration, "postgresql", "downgrade")

    assert f"ALTER TABLE {TABLE} ADD COLUMN {COLUMN} VARCHAR(32)" in upgrade_sql
    assert f"CREATE INDEX CONCURRENTLY {INDEX} ON {TABLE} ({COLUMN})" in upgrade_sql
    assert f"DROP INDEX CONCURRENTLY {INDEX}" in downgrade_sql
    assert f"ALTER TABLE {TABLE} DROP COLUMN {COLUMN}" in downgrade_sql


def test_postgresql_offline_concurrent_ddl_escapes_outer_transaction() -> None:
    migration = _migration_module()

    for operation in ("upgrade", "downgrade"):
        sql = _transactional_offline_sql(migration, "postgresql", operation)
        # CREATE/DROP INDEX CONCURRENTLY cannot run inside a transaction block,
        # so the generated script must COMMIT before emitting it.
        concurrent_at = sql.index("CONCURRENTLY")
        assert "COMMIT;" in sql[:concurrent_at]


def test_index_definition_matches_only_a_plain_full_index() -> None:
    """Key columns alone are insufficient: UNIQUE blocks a sender's second task
    and a partial index does not cover every row."""

    migration = _migration_module()
    matches = migration._index_definition_matches

    assert matches({"column_names": [COLUMN], "unique": 0}) is True
    assert matches(None) is False
    assert matches({"column_names": ["status"], "unique": 0}) is False
    assert matches({"column_names": [COLUMN], "unique": 1}) is False
    assert (
        matches(
            {
                "column_names": [COLUMN],
                "unique": 0,
                "dialect_options": {"postgresql_where": "x IS NOT NULL"},
            }
        )
        is False
    )
    assert (
        matches(
            {
                "column_names": [COLUMN],
                "unique": 0,
                "dialect_options": {"sqlite_where": "x IS NOT NULL"},
            }
        )
        is False
    )


def test_postgresql_index_validity_sql_is_scoped_to_the_target_table() -> None:
    """to_regclass() follows search_path, so the lookup must be constrained by
    the table and schema instead."""

    migration = _migration_module()
    sql = str(migration.POSTGRES_INDEX_VALIDITY_SQL)

    assert "to_regclass" not in sql
    assert "tbl.relname = :table_name" in sql
    assert "ns.nspname = :schema_name" in sql


def test_postgresql_online_upgrade_rebuilds_unusable_or_drifted_indexes() -> None:
    """A failed CREATE INDEX CONCURRENTLY leaves the index invalid, and a
    same-name index may have different semantics. IF NOT EXISTS would skip both
    rebuilds, so the retry must drop first. Only a valid, plain, full index over
    the right column is accepted as-is."""

    migration = _migration_module()
    context = MigrationContext.configure(dialect_name="postgresql")

    class _NoopAutocommit:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_exc: object) -> bool:
            return False

    plain = {"column_names": [COLUMN], "unique": 0}
    cases = (
        # (validity, definition, expect_drop, expect_create)
        (False, plain, True, True),  # present but invalid
        (None, None, False, True),  # absent
        (True, plain, False, False),  # valid and correct
        (True, {"column_names": ["status"], "unique": 0}, True, True),  # drifted
        (True, {"column_names": [COLUMN], "unique": 1}, True, True),  # UNIQUE
        (
            True,
            {
                "column_names": [COLUMN],
                "unique": 0,
                "dialect_options": {"postgresql_where": "x IS NOT NULL"},
            },
            True,
            True,
        ),  # partial
    )
    for validity, definition, expect_drop, expect_create in cases:
        created: list[dict] = []
        dropped: list[dict] = []
        with Operations.context(context):
            with (
                patch.object(context, "autocommit_block", _NoopAutocommit),
                patch.object(migration.op, "get_context", return_value=context),
                patch.object(migration, "_target_schema", return_value="public"),
                patch.object(migration, "_online_table_exists", return_value=True),
                patch.object(migration, "_online_columns", return_value={COLUMN}),
                patch.object(
                    migration, "_postgres_index_validity", return_value=validity
                ),
                patch.object(
                    migration,
                    "_online_index_definition",
                    return_value=definition,
                ),
                patch.object(
                    migration.op,
                    "create_index",
                    side_effect=lambda *a, **kw: created.append(kw),
                ),
                patch.object(
                    migration.op,
                    "drop_index",
                    side_effect=lambda *a, **kw: dropped.append(kw),
                ),
            ):
                migration.upgrade()

        label = (validity, definition)
        assert bool(dropped) is expect_drop, label
        if expect_create:
            # Every DDL call is schema-qualified, so reflection, the catalog
            # lookup, and the rebuild always address the same relation.
            assert created == [
                {
                    "schema": "public",
                    "if_not_exists": True,
                    "postgresql_concurrently": True,
                }
            ], label
        else:
            assert created == [], label


@pytest.mark.parametrize(
    "existing_ddl",
    [
        f"CREATE INDEX {INDEX} ON {TABLE} (status)",
        f"CREATE UNIQUE INDEX {INDEX} ON {TABLE} ({COLUMN})",
        f"CREATE INDEX {INDEX} ON {TABLE} ({COLUMN}) WHERE {COLUMN} IS NOT NULL",
    ],
)
def test_sqlite_online_upgrade_rebuilds_semantically_wrong_indexes(
    existing_ddl: str,
) -> None:
    """A same-name index that is drifted, UNIQUE, or partial must be rebuilt."""

    migration = _migration_module()
    engine = sa.create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "CREATE TABLE tasks ("
                "id INTEGER PRIMARY KEY, "
                "status VARCHAR(32), "
                f"{COLUMN} VARCHAR(32))"
            )
        )
        connection.execute(sa.text(existing_ddl))

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        rebuilt = next(
            item
            for item in sa.inspect(connection).get_indexes(TABLE)
            if item["name"] == INDEX
        )
        assert tuple(rebuilt["column_names"]) == (COLUMN,)
        assert not rebuilt["unique"]
        assert not any(
            key.endswith("_where") for key in (rebuilt.get("dialect_options") or {})
        )


def test_target_schema_resolves_the_visible_tasks_relation() -> None:
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
