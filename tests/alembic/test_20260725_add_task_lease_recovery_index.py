"""Tests for the task lease recovery lookup index migration."""

import importlib.util
from contextlib import nullcontext
from io import StringIO
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

from xagent.web.models.task import Task

MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20260725_add_task_lease_recovery_index.py"
)
REVISION = "20260725_add_task_lease_recovery_index"
DOWN_REVISION = "20260724_seed_google_ads_mcp_app"
TABLE = "tasks"
INDEX = "ix_tasks_status_lease_expires_at"
INDEX_COLUMNS = ["status", "lease_expires_at", "id"]


def _load_migration_module():
    spec = importlib.util.spec_from_file_location(
        "add_task_lease_recovery_index_migration", MIGRATION_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection) -> Operations:
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


def _transactional_offline_sql(
    migration,
    dialect_name: str,
    operation: str,
) -> str:
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name=dialect_name,
        opts={"as_sql": True, "output_buffer": output},
    )

    with Operations.context(context), context.begin_transaction():
        getattr(migration, operation)()

    return output.getvalue()


def _create_tasks_table(connection, columns: str = "") -> None:
    connection.execute(
        sa.text(
            "CREATE TABLE tasks ("
            "id INTEGER PRIMARY KEY, "
            "status VARCHAR(32), "
            "lease_expires_at DATETIME"
            f"{columns}"
            ")"
        )
    )


def test_revision_metadata() -> None:
    migration = _load_migration_module()

    assert migration.revision == REVISION
    assert migration.down_revision == DOWN_REVISION


def test_task_model_and_migration_share_the_same_index_contract() -> None:
    model_index = next(index for index in Task.__table__.indexes if index.name == INDEX)

    assert [column.name for column in model_index.columns] == INDEX_COLUMNS


def test_online_upgrade_creates_lease_recovery_index_idempotently() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        _create_tasks_table(connection)
        operations = _operations(connection)
        with Operations.context(operations.get_context()):
            migration.upgrade()
            migration.upgrade()

        indexes = {
            index["name"]: index for index in sa.inspect(connection).get_indexes(TABLE)
        }
        assert indexes[INDEX]["column_names"] == INDEX_COLUMNS


@pytest.mark.parametrize(
    "schema",
    [
        None,
        "CREATE TABLE tasks (id INTEGER PRIMARY KEY, status VARCHAR(32))",
        "CREATE TABLE tasks (id INTEGER PRIMARY KEY, lease_expires_at DATETIME)",
    ],
)
def test_online_upgrade_noops_without_tasks_or_required_columns(
    schema: str | None,
) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        if schema is not None:
            connection.execute(sa.text(schema))

        operations = _operations(connection)
        with Operations.context(operations.get_context()):
            migration.upgrade()

        if schema is not None:
            assert INDEX not in {
                index["name"] for index in sa.inspect(connection).get_indexes(TABLE)
            }


def test_online_downgrade_drops_lease_recovery_index() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        _create_tasks_table(connection)
        operations = _operations(connection)
        with Operations.context(operations.get_context()):
            migration.upgrade()
            migration.downgrade()

        assert INDEX not in {
            index["name"] for index in sa.inspect(connection).get_indexes(TABLE)
        }


def test_postgresql_online_upgrade_retries_an_invalid_concurrent_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration = _load_migration_module()
    context = MigrationContext.configure(dialect_name="postgresql")
    operations = Operations(context)
    calls: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(
        migration,
        "_online_columns",
        lambda: set(INDEX_COLUMNS),
    )
    monkeypatch.setattr(
        migration,
        "_postgres_index_validity",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(
        migration,
        "_online_index_columns",
        lambda _index_name: tuple(INDEX_COLUMNS),
        raising=False,
    )
    monkeypatch.setattr(context, "autocommit_block", nullcontext)
    monkeypatch.setattr(
        operations,
        "drop_index",
        lambda *args, **kwargs: calls.append(("drop", kwargs)),
    )
    monkeypatch.setattr(
        operations,
        "create_index",
        lambda *args, **kwargs: calls.append(("create", kwargs)),
    )

    with Operations.context(context):
        monkeypatch.setattr(migration, "op", operations)
        migration.upgrade()

    assert calls == [
        (
            "drop",
            {
                "table_name": TABLE,
                "if_exists": True,
                "postgresql_concurrently": True,
            },
        ),
        (
            "create",
            {
                "if_not_exists": True,
                "postgresql_concurrently": True,
            },
        ),
    ]


def test_postgresql_online_upgrade_replaces_valid_wrong_index_definition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration = _load_migration_module()
    context = MigrationContext.configure(dialect_name="postgresql")
    operations = Operations(context)
    calls: list[str] = []

    monkeypatch.setattr(
        migration,
        "_online_columns",
        lambda: set(INDEX_COLUMNS),
    )
    monkeypatch.setattr(
        migration,
        "_postgres_index_validity",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(
        migration,
        "_online_index_columns",
        lambda _index_name: ("status", "lease_expires_at"),
        raising=False,
    )
    monkeypatch.setattr(context, "autocommit_block", nullcontext)
    monkeypatch.setattr(
        operations,
        "drop_index",
        lambda *_args, **_kwargs: calls.append("drop"),
    )
    monkeypatch.setattr(
        operations,
        "create_index",
        lambda *_args, **_kwargs: calls.append("create"),
    )

    with Operations.context(context):
        monkeypatch.setattr(migration, "op", operations)
        migration.upgrade()

    assert calls == ["drop", "create"]


def test_postgresql_offline_upgrade_and_downgrade_emit_concurrent_index_sql() -> None:
    migration = _load_migration_module()

    upgrade_sql = _offline_sql(migration, "postgresql", "upgrade")
    downgrade_sql = _offline_sql(migration, "postgresql", "downgrade")

    assert (
        f"CREATE INDEX CONCURRENTLY {INDEX} ON {TABLE} (status, lease_expires_at, id)"
    ) in upgrade_sql
    assert f"DROP INDEX CONCURRENTLY {INDEX}" in downgrade_sql


@pytest.mark.parametrize("operation", ["upgrade", "downgrade"])
def test_postgresql_offline_concurrent_ddl_escapes_outer_transaction(
    operation: str,
) -> None:
    migration = _load_migration_module()

    sql = _transactional_offline_sql(migration, "postgresql", operation)
    statement = (
        f"CREATE INDEX CONCURRENTLY {INDEX}"
        if operation == "upgrade"
        else f"DROP INDEX CONCURRENTLY {INDEX}"
    )

    before_statement, after_statement = sql.split(statement, 1)
    assert before_statement.rstrip().endswith("COMMIT;")
    assert after_statement.lstrip().startswith(
        "ON tasks (status, lease_expires_at, id);" if operation == "upgrade" else ";"
    )
    assert "BEGIN;" in after_statement


@pytest.mark.parametrize("dialect_name", ["sqlite", "mysql"])
def test_offline_upgrade_and_downgrade_emit_portable_index_sql(
    dialect_name: str,
) -> None:
    migration = _load_migration_module()

    upgrade_sql = _offline_sql(migration, dialect_name, "upgrade")
    downgrade_sql = _offline_sql(migration, dialect_name, "downgrade")

    assert (
        f"CREATE INDEX {INDEX} ON {TABLE} (status, lease_expires_at, id)" in upgrade_sql
    )
    assert f"DROP INDEX {INDEX}" in downgrade_sql
    if dialect_name == "mysql":
        assert f"DROP INDEX {INDEX} ON {TABLE}" in downgrade_sql
    for sql in (upgrade_sql, downgrade_sql):
        assert "%('" not in sql
        assert ":param" not in sql
        assert "?" not in sql
