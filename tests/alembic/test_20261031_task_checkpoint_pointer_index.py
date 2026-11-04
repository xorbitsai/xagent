"""Tests for the tasks.last_checkpoint_trace_event_id index migration."""

import importlib.util
from contextlib import nullcontext
from io import StringIO
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

from tests.shared.postgres_disposable import disposable_database_factory
from xagent.web.models.task import Task

MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20261031_task_checkpoint_pointer_index.py"
)
REVISION = "20261031_task_checkpoint_pointer_index"
DOWN_REVISION = "20261030_merge_history_freshdesk"
TABLE = "tasks"
INDEX = "ix_tasks_last_checkpoint_trace_event_id"
INDEX_COLUMNS = ["last_checkpoint_trace_event_id"]


def _load_migration_module():
    spec = importlib.util.spec_from_file_location(
        "task_checkpoint_pointer_index_migration", MIGRATION_PATH
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


def _create_tasks_table(connection) -> None:
    connection.execute(
        sa.text(
            "CREATE TABLE tasks ("
            "id INTEGER PRIMARY KEY, "
            "last_checkpoint_trace_event_id INTEGER"
            ")"
        )
    )


def _index_names(connection) -> set[str]:
    return {index["name"] for index in sa.inspect(connection).get_indexes(TABLE)}


def test_revision_metadata() -> None:
    migration = _load_migration_module()

    assert migration.revision == REVISION
    assert migration.down_revision == DOWN_REVISION


def test_task_model_and_migration_share_the_same_index_contract() -> None:
    migration = _load_migration_module()
    model_indexes = {
        index.name: [column.name for column in index.columns]
        for index in Task.__table__.indexes
    }

    assert migration.TABLE == Task.__tablename__
    assert model_indexes.get(migration.INDEX) == list(migration.INDEX_COLUMNS)


def test_online_upgrade_creates_pointer_index_idempotently() -> None:
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
        "CREATE TABLE tasks (id INTEGER PRIMARY KEY)",
    ],
)
def test_online_upgrade_noops_without_tasks_or_pointer_column(
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
            assert INDEX not in _index_names(connection)


def test_online_downgrade_drops_pointer_index() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        _create_tasks_table(connection)
        operations = _operations(connection)
        with Operations.context(operations.get_context()):
            migration.upgrade()
            migration.downgrade()

        assert INDEX not in _index_names(connection)


def test_postgresql_online_upgrade_retries_an_invalid_concurrent_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration = _load_migration_module()
    context = MigrationContext.configure(dialect_name="postgresql")
    operations = Operations(context)
    calls: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(migration, "_online_columns", lambda: set(INDEX_COLUMNS))
    monkeypatch.setattr(migration, "_postgres_index_validity", lambda: False)
    monkeypatch.setattr(
        migration,
        "_online_index_columns",
        lambda _index_name: tuple(INDEX_COLUMNS),
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

    monkeypatch.setattr(migration, "_online_columns", lambda: set(INDEX_COLUMNS))
    monkeypatch.setattr(migration, "_postgres_index_validity", lambda: True)
    monkeypatch.setattr(
        migration,
        "_online_index_columns",
        lambda _index_name: ("id",),
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


def test_postgresql_online_upgrade_keeps_a_valid_matching_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration = _load_migration_module()
    context = MigrationContext.configure(dialect_name="postgresql")
    operations = Operations(context)
    calls: list[str] = []

    monkeypatch.setattr(migration, "_online_columns", lambda: set(INDEX_COLUMNS))
    monkeypatch.setattr(migration, "_postgres_index_validity", lambda: True)
    monkeypatch.setattr(
        migration,
        "_online_index_columns",
        lambda _index_name: tuple(INDEX_COLUMNS),
    )
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

    assert calls == []


def test_postgresql_offline_upgrade_and_downgrade_emit_concurrent_index_sql() -> None:
    migration = _load_migration_module()

    upgrade_sql = _offline_sql(migration, "postgresql", "upgrade")
    downgrade_sql = _offline_sql(migration, "postgresql", "downgrade")

    assert (
        f"CREATE INDEX CONCURRENTLY {INDEX} ON {TABLE} (last_checkpoint_trace_event_id)"
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
        "ON tasks (last_checkpoint_trace_event_id);" if operation == "upgrade" else ";"
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
        f"CREATE INDEX {INDEX} ON {TABLE} (last_checkpoint_trace_event_id)"
        in upgrade_sql
    )
    assert f"DROP INDEX {INDEX}" in downgrade_sql
    if dialect_name == "mysql":
        assert f"DROP INDEX {INDEX} ON {TABLE}" in downgrade_sql


@pytest.fixture
def postgresql_engine_factory():
    with disposable_database_factory("xagent_checkpoint_pointer_index") as make:
        yield make


@pytest.mark.postgresql
def test_postgresql_fk_check_lookup_uses_the_index(postgresql_engine_factory) -> None:
    """The lookup PostgreSQL runs to check the FK on a trace_events delete
    must stop being a sequential scan of tasks once the index exists."""

    migration = _load_migration_module()
    engine = postgresql_engine_factory("fk_lookup")
    lookup = sa.text(
        "EXPLAIN (FORMAT JSON) "
        "SELECT 1 FROM ONLY tasks x WHERE last_checkpoint_trace_event_id = 1"
    )

    def scan_nodes(plan: dict) -> set[str]:
        nodes = {plan["Node Type"]}
        for child in plan.get("Plans", []):
            nodes |= scan_nodes(child)
        return nodes

    with engine.begin() as connection:
        _create_tasks_table(connection)
        connection.execute(
            sa.text(
                "INSERT INTO tasks (id, last_checkpoint_trace_event_id) "
                "SELECT g, g FROM generate_series(1, 20000) AS g"
            )
        )
        connection.execute(sa.text("ANALYZE tasks"))
        before = connection.execute(lookup).scalar_one()[0]["Plan"]

    with engine.connect() as connection:
        context = MigrationContext.configure(connection)
        with Operations.context(context), context.begin_transaction():
            migration.upgrade()

    with engine.begin() as connection:
        connection.execute(sa.text("ANALYZE tasks"))
        after = connection.execute(lookup).scalar_one()[0]["Plan"]

    assert "Seq Scan" in scan_nodes(before)
    assert "Seq Scan" not in scan_nodes(after)
