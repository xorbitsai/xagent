"""Tests for the tasks.interaction_protocol_version migration."""

from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

from tests.shared.postgres_disposable import (
    disposable_database_factory,
    load_migration_module,
)
from tests.web.services.checkpoint_anchor_shared import (
    reset_checkpoint_anchor_fk_create_rule,
)
from xagent.web.models.database import Base

MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20260810_add_task_interaction_protocol_version.py"
)
TABLE = "tasks"
COLUMN = "interaction_protocol_version"
CONSTRAINT_NAME = "ck_tasks_interaction_protocol_version"


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


def _legacy_tasks_table(engine) -> None:
    """Migration-only schema: just the columns this migration reads."""
    metadata = sa.MetaData()
    sa.Table(
        TABLE,
        metadata,
        sa.Column("id", sa.Integer(), primary_key=True),
    )
    metadata.create_all(engine)


def _column_names(connection) -> set[str]:
    return {column["name"] for column in sa.inspect(connection).get_columns(TABLE)}


def _check_constraint_names(connection) -> set[str]:
    return {
        item["name"]
        for item in sa.inspect(connection).get_check_constraints(TABLE)
        if item.get("name")
    }


# ---- T-M-1a / T-M-1b: offline upgrade content ----


def test_offline_upgrade_postgresql_emits_add_column_and_check() -> None:
    """A rendering check: it proves the offline SQL emits the expected ADD
    COLUMN and CHECK statements, not that they execute against a live
    database -- the execution half is covered by the online migration
    tests."""
    migration = load_migration_module(MIGRATION_PATH)

    with patch.object(
        migration.sa,
        "inspect",
        side_effect=AssertionError("offline branch must not reflect"),
    ):
        sql = _offline_sql(migration, "postgresql", "upgrade")

    assert f"ALTER TABLE {TABLE} ADD COLUMN {COLUMN} INTEGER" in sql
    assert (
        f"ALTER TABLE {TABLE} ADD CONSTRAINT {CONSTRAINT_NAME} CHECK "
        f"({migration.CONSTRAINT_CONDITION})"
    ) in sql


def test_offline_upgrade_sqlite_emits_add_column_only() -> None:
    migration = load_migration_module(MIGRATION_PATH)

    with patch.object(
        migration.sa,
        "inspect",
        side_effect=AssertionError("offline branch must not reflect"),
    ):
        sql = _offline_sql(migration, "sqlite", "upgrade")

    assert f"ALTER TABLE {TABLE} ADD COLUMN {COLUMN} INTEGER" in sql
    assert "ADD CONSTRAINT" not in sql
    assert "CHECK" not in sql


# ---- T-M-1c / T-M-1d: offline downgrade content and ordering ----


def test_offline_downgrade_postgresql_drops_constraint_before_column() -> None:
    migration = load_migration_module(MIGRATION_PATH)

    with patch.object(
        migration.sa,
        "inspect",
        side_effect=AssertionError("offline branch must not reflect"),
    ):
        sql = _offline_sql(migration, "postgresql", "downgrade")

    drop_constraint_at = sql.index(
        f"ALTER TABLE {TABLE} DROP CONSTRAINT {CONSTRAINT_NAME}"
    )
    drop_column_at = sql.index(f"ALTER TABLE {TABLE} DROP COLUMN {COLUMN}")
    assert drop_constraint_at < drop_column_at


def test_offline_downgrade_sqlite_only_drops_column() -> None:
    """Pins the plain DROP COLUMN as the offline SQLite downgrade's
    rendering. That rendering is not executable against a create_all-built
    SQLite database -- see the migration's downgrade() comment for why that
    is accepted rather than fixed."""
    migration = load_migration_module(MIGRATION_PATH)

    with patch.object(
        migration.sa,
        "inspect",
        side_effect=AssertionError("offline branch must not reflect"),
    ):
        sql = _offline_sql(migration, "sqlite", "downgrade")

    assert f"ALTER TABLE {TABLE} DROP COLUMN {COLUMN}" in sql
    assert "DROP CONSTRAINT" not in sql


# ---- T-M-1e: no bind parameters on either dialect or direction ----


@pytest.mark.parametrize("dialect_name", ["sqlite", "postgresql"])
def test_offline_sql_carries_no_bind_parameters(dialect_name) -> None:
    migration = load_migration_module(MIGRATION_PATH)

    upgrade_sql = _offline_sql(migration, dialect_name, "upgrade")
    downgrade_sql = _offline_sql(migration, dialect_name, "downgrade")

    for sql in (upgrade_sql, downgrade_sql):
        assert "%(" not in sql
        assert ":table_name" not in sql


# ---- T-M-1f: the offline branch never reflects ----


@pytest.mark.parametrize("dialect_name", ["sqlite", "postgresql"])
@pytest.mark.parametrize("operation", ["upgrade", "downgrade"])
def test_offline_branch_does_not_reflect(dialect_name, operation) -> None:
    migration = load_migration_module(MIGRATION_PATH)

    with patch.object(
        migration.sa,
        "inspect",
        side_effect=AssertionError("offline branch must not reflect"),
    ):
        sql = _offline_sql(migration, dialect_name, operation)

    assert sql  # rendered without tripping the reflect-call trap above


# ---- T-M-1g: online SQLite upgrade/downgrade ----


def test_online_sqlite_upgrade_adds_column_without_check_and_downgrade_removes_it() -> (
    None
):
    migration = load_migration_module(MIGRATION_PATH)
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        _legacy_tasks_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

            columns = {c["name"]: c for c in sa.inspect(connection).get_columns(TABLE)}
            assert COLUMN in columns
            assert columns[COLUMN]["nullable"] is True
            assert CONSTRAINT_NAME not in _check_constraint_names(connection)

            migration.downgrade()
            assert COLUMN not in _column_names(connection)


# ---- T-M-1h: online PostgreSQL upgrade/downgrade (disposable database) ----


@pytest.mark.postgresql
def test_online_postgresql_upgrade_adds_column_and_check_and_downgrade_removes_both() -> (
    None
):
    migration = load_migration_module(MIGRATION_PATH)

    with disposable_database_factory("xagent_w1_migration") as make_database:
        engine = make_database("upgrade_downgrade")

        with engine.begin() as connection:
            _legacy_tasks_table(connection)
            with patch.object(migration, "op", _operations(connection)):
                migration.upgrade()

                columns = {
                    c["name"]: c for c in sa.inspect(connection).get_columns(TABLE)
                }
                assert COLUMN in columns
                assert columns[COLUMN]["nullable"] is True
                assert CONSTRAINT_NAME in _check_constraint_names(connection)

                migration.downgrade()
                assert COLUMN not in _column_names(connection)
                assert CONSTRAINT_NAME not in _check_constraint_names(connection)


# ---- T-M-3a: online upgrade is idempotent on both backends ----


def _assert_idempotent_upgrade(migration, engine) -> None:
    with engine.begin() as connection:
        _legacy_tasks_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()

            columns = [c["name"] for c in sa.inspect(connection).get_columns(TABLE)]
            assert columns.count(COLUMN) == 1

            constraint_hits = [
                name
                for name in _check_constraint_names(connection)
                if name == CONSTRAINT_NAME
            ]
            if connection.dialect.name == "postgresql":
                assert len(constraint_hits) == 1
            else:
                assert len(constraint_hits) == 0


def test_online_upgrade_is_idempotent_sqlite() -> None:
    migration = load_migration_module(MIGRATION_PATH)
    _assert_idempotent_upgrade(migration, sa.create_engine("sqlite:///:memory:"))


@pytest.mark.postgresql
def test_online_upgrade_is_idempotent_postgresql() -> None:
    migration = load_migration_module(MIGRATION_PATH)
    with disposable_database_factory("xagent_w1_migration") as make_database:
        _assert_idempotent_upgrade(migration, make_database("idempotent_upgrade"))


# ---- T-M-3b: online downgrade is idempotent when the column is absent ----


def _assert_idempotent_downgrade_when_absent(migration, engine) -> None:
    with engine.begin() as connection:
        _legacy_tasks_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()
            migration.downgrade()

            assert COLUMN not in _column_names(connection)


def test_online_downgrade_is_idempotent_when_column_is_absent_sqlite() -> None:
    migration = load_migration_module(MIGRATION_PATH)
    _assert_idempotent_downgrade_when_absent(
        migration, sa.create_engine("sqlite:///:memory:")
    )


@pytest.mark.postgresql
def test_online_downgrade_is_idempotent_when_column_is_absent_postgresql() -> None:
    migration = load_migration_module(MIGRATION_PATH)
    with disposable_database_factory("xagent_w1_migration") as make_database:
        _assert_idempotent_downgrade_when_absent(
            migration, make_database("idempotent_downgrade")
        )


# ---- T-M-3c: no-op when the tasks table itself does not exist ----


def test_online_upgrade_and_downgrade_noop_without_the_tasks_table() -> None:
    migration = load_migration_module(MIGRATION_PATH)
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()

        assert TABLE not in sa.inspect(connection).get_table_names()


# ---- T-M-3d: upgrading a create_all-built database is a no-op ----


def _assert_create_all_then_upgrade_is_noop(migration, engine) -> None:
    reset_checkpoint_anchor_fk_create_rule()
    Base.metadata.create_all(engine)

    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()  # must not raise on a duplicate constraint name

        columns = {c["name"]: c for c in sa.inspect(connection).get_columns(TABLE)}
        assert COLUMN in columns
        if connection.dialect.name == "postgresql":
            assert CONSTRAINT_NAME in _check_constraint_names(connection)


def test_create_all_then_upgrade_is_noop_sqlite() -> None:
    migration = load_migration_module(MIGRATION_PATH)
    _assert_create_all_then_upgrade_is_noop(
        migration, sa.create_engine("sqlite:///:memory:")
    )


@pytest.mark.postgresql
def test_create_all_then_upgrade_is_noop_postgresql() -> None:
    migration = load_migration_module(MIGRATION_PATH)
    with disposable_database_factory("xagent_w1_migration") as make_database:
        _assert_create_all_then_upgrade_is_noop(
            migration, make_database("create_all_then_upgrade")
        )


# ---- T-M-4: CHECK semantics -- NULL/1 accepted, 2/0 rejected ----


def _assert_check_constraint_semantics(engine, extra_columns: dict) -> None:
    tasks = sa.Table(TABLE, sa.MetaData(), autoload_with=engine)

    for value in (None, 1):
        with engine.begin() as connection:
            connection.execute(
                sa.insert(tasks).values(
                    interaction_protocol_version=value, **extra_columns
                )
            )

    for value in (2, 0):
        with pytest.raises(sa.exc.IntegrityError):
            with engine.begin() as connection:
                connection.execute(
                    sa.insert(tasks).values(
                        interaction_protocol_version=value, **extra_columns
                    )
                )


@pytest.mark.postgresql
def test_check_constraint_semantics_on_an_online_migrated_postgresql_database() -> None:
    migration = load_migration_module(MIGRATION_PATH)
    with disposable_database_factory("xagent_w1_migration") as make_database:
        engine = make_database("check_semantics")
        with engine.begin() as connection:
            _legacy_tasks_table(connection)
            with patch.object(migration, "op", _operations(connection)):
                migration.upgrade()

        _assert_check_constraint_semantics(engine, extra_columns={})


def test_check_constraint_semantics_on_a_create_all_built_sqlite_database() -> None:
    reset_checkpoint_anchor_fk_create_rule()
    engine = sa.create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    _assert_check_constraint_semantics(
        engine, extra_columns={"user_id": 1, "title": "probe"}
    )


# ---- _target_schema(): catalog hit / empty-catalog fallback (mocked) ----


def test_target_schema_resolves_the_visible_tasks_relation() -> None:
    """version_table_schema names only the Alembic version table and
    current_schema() is merely the first search_path entry, so neither
    identifies the relation the unqualified DDL resolves to."""

    migration = load_migration_module(MIGRATION_PATH)
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

    migration = load_migration_module(MIGRATION_PATH)

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
