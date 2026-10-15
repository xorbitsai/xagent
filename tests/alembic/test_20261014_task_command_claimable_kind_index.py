"""Partial index on task_execution_commands for the reply-first claim lookup.

The dispatcher's candidate lookup filters claimable commands by ``kind`` and
takes the oldest ``id``; this migration adds a partial index scoped to the
claimable statuses so that lookup stays cheap as settled commands accumulate.
Both dialects the app runs on are exercised, since partial-index syntax and
planner behaviour differ between them.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from tests.shared.postgres_disposable import (
    disposable_database_factory,
    load_migration_module,
)
from xagent.db.config import create_alembic_config
from xagent.web.models import database as database_module
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.services import task_command_transport as transport

REVISION = "20261014_task_cmd_claimable_kind"
DOWN_REVISION = "20261013_atlassian_mcp_scope"
TABLE = "task_execution_commands"
INDEX = "ix_task_commands_claimable_kind"
INDEX_COLUMNS = ("kind", "id")
MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20261014_task_command_claimable_kind_index.py"
)


def _create_legacy_table(connection, *, with_status: bool = True) -> None:
    status_column = "status VARCHAR(16) NOT NULL, " if with_status else ""
    connection.execute(
        text(
            f"CREATE TABLE {TABLE} ("
            "id INTEGER PRIMARY KEY, task_id INTEGER NOT NULL, "
            f"kind VARCHAR(32) NOT NULL, {status_column}"
            "created_at DATETIME)"
        )
    )


def _stamp(connection) -> None:
    connection.execute(
        text("CREATE TABLE alembic_version (version_num VARCHAR(255) NOT NULL)")
    )
    connection.execute(
        text("INSERT INTO alembic_version (version_num) VALUES (:revision)"),
        {"revision": DOWN_REVISION},
    )


def _index_names(connection) -> set[str]:
    return {index["name"] for index in inspect(connection).get_indexes(TABLE)}


def test_sqlite_upgrade_adds_partial_index_and_downgrade_removes_it() -> None:
    engine = create_engine("sqlite:///:memory:")
    config = create_alembic_config(engine)

    with engine.begin() as connection:
        _stamp(connection)
        _create_legacy_table(connection)
        config.attributes["connection"] = connection

        command.upgrade(config, REVISION)

        indexes = {
            index["name"]: index for index in inspect(connection).get_indexes(TABLE)
        }
        assert INDEX in indexes
        assert tuple(indexes[INDEX]["column_names"]) == INDEX_COLUMNS
        ddl = connection.execute(
            text("SELECT sql FROM sqlite_master WHERE name = :name"),
            {"name": INDEX},
        ).scalar_one()
        assert "status IN ('pending', 'processing')" in ddl

        command.downgrade(config, DOWN_REVISION)
        assert INDEX not in _index_names(connection)


def test_sqlite_upgrade_skips_without_the_table() -> None:
    engine = create_engine("sqlite:///:memory:")
    config = create_alembic_config(engine)

    with engine.begin() as connection:
        _stamp(connection)
        config.attributes["connection"] = connection

        command.upgrade(config, REVISION)

        assert TABLE not in inspect(connection).get_table_names()
        assert (
            connection.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalar_one()
            == REVISION
        )


def test_sqlite_upgrade_skips_without_the_status_column() -> None:
    engine = create_engine("sqlite:///:memory:")
    config = create_alembic_config(engine)

    with engine.begin() as connection:
        _stamp(connection)
        _create_legacy_table(connection, with_status=False)
        config.attributes["connection"] = connection

        command.upgrade(config, REVISION)

        assert INDEX not in _index_names(connection)


def test_sqlite_upgrade_is_a_noop_when_the_index_already_exists() -> None:
    engine = create_engine("sqlite:///:memory:")
    config = create_alembic_config(engine)

    with engine.begin() as connection:
        _stamp(connection)
        _create_legacy_table(connection)
        connection.execute(
            text(
                f"CREATE INDEX {INDEX} ON {TABLE} (kind, id) "
                "WHERE status IN ('pending', 'processing')"
            )
        )
        config.attributes["connection"] = connection

        command.upgrade(config, REVISION)

        assert INDEX in _index_names(connection)


def test_migration_column_list_and_predicate_match_the_model() -> None:
    """A database built by ``create_all`` and one built by running every
    migration must end up with the same index."""

    migration = load_migration_module(MIGRATION_PATH, "claimable_kind_index_parity")

    model_index = next(
        index for index in TaskExecutionCommand.__table__.indexes if index.name == INDEX
    )
    model_columns = tuple(column.name for column in model_index.columns)
    assert model_columns == migration.INDEX_COLUMNS == INDEX_COLUMNS
    assert migration.INDEX == INDEX
    assert migration.TABLE == TABLE

    for dialect in ("postgresql", "sqlite"):
        predicate = model_index.dialect_options[dialect]["where"]
        assert str(predicate) == str(migration.PREDICATE)


def test_postgres_deparsed_in_list_matches_the_expected_predicate() -> None:
    migration = load_migration_module(MIGRATION_PATH, "claimable_kind_index_deparse")
    deparsed = (
        "((status)::text = ANY ((ARRAY['pending'::character varying, "
        "'processing'::character varying])::text[]))"
    )

    assert migration._postgres_predicate_matches_expected(deparsed)
    assert not migration._postgres_predicate_matches_expected(
        "((status)::text = 'pending'::text)"
    )
    assert not migration._postgres_predicate_matches_expected(
        "((status)::text = ANY ((ARRAY['pending'::character varying, "
        "'completed'::character varying])::text[]))"
    )
    assert not migration._postgres_predicate_matches_expected(
        "((status)::text <> ALL ((ARRAY['pending'::character varying, "
        "'processing'::character varying])::text[]))"
    )
    assert not migration._postgres_predicate_matches_expected(
        "(((status)::text > 'pending'::text) AND ((status)::text < 'processing'::text))"
    )
    assert not migration._postgres_predicate_matches_expected(None)


@pytest.fixture
def postgresql_engine_factory():
    with disposable_database_factory("xagent_claimable_kind_index") as make:
        yield make


def _run(migration, connection, operation: str) -> None:
    # ``autocommit_block()`` needs the connection outside any open transaction
    # other than the per-migration one begun below.
    connection.commit()
    context = MigrationContext.configure(
        connection, opts={"transaction_per_migration": True}
    )
    with Operations.context(context):
        with context.begin_transaction(_per_migration=True):
            getattr(migration, operation)()


def _create_pg_table(connection) -> None:
    connection.execute(
        text(
            f"CREATE TABLE {TABLE} ("
            "id SERIAL PRIMARY KEY, task_id INTEGER NOT NULL, "
            "kind VARCHAR(32) NOT NULL, status VARCHAR(16) NOT NULL)"
        )
    )


def _pg_index_columns(connection) -> tuple[str, ...] | None:
    for index in inspect(connection).get_indexes(TABLE):
        if index["name"] == INDEX:
            return tuple(index["column_names"])
    return None


def _pg_index_state(connection) -> tuple[bool | None, str | None]:
    row = connection.execute(
        text(
            "SELECT i.indisvalid, pg_catalog.pg_get_expr(i.indpred, i.indrelid) "
            "FROM pg_catalog.pg_index AS i "
            "WHERE i.indexrelid = pg_catalog.to_regclass(:name)"
        ),
        {"name": INDEX},
    ).one_or_none()
    return (None, None) if row is None else (row[0], row[1])


@pytest.mark.postgresql
def test_postgresql_upgrade_builds_a_valid_index_with_the_expected_predicate(
    postgresql_engine_factory,
) -> None:
    migration = load_migration_module(MIGRATION_PATH, "claimable_kind_index_pg")
    engine = postgresql_engine_factory("upgrade")

    with engine.connect() as connection:
        _create_pg_table(connection)
        connection.commit()

        _run(migration, connection, "upgrade")
        valid, predicate = _pg_index_state(connection)
        assert valid is True
        assert migration._postgres_predicate_matches_expected(predicate), predicate

        _run(migration, connection, "downgrade")
        assert INDEX not in _index_names(connection)


@pytest.mark.postgresql
def test_postgresql_reply_first_lookup_uses_the_partial_index(
    postgresql_engine_factory, monkeypatch
) -> None:
    """EXPLAIN the statement the real ``_find_command_candidate`` issues."""

    engine = postgresql_engine_factory("plan")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setattr(database_module, "_SessionLocal", factory)
    monkeypatch.setattr(transport, "get_runner_id", lambda: "me")

    with Session(engine) as session:
        user = User(username="plan-user", password_hash="hash", is_admin=False)
        session.add(user)
        session.flush()
        for task_id in range(1, 21):
            session.add(
                Task(
                    id=task_id,
                    user_id=user.id,
                    title=f"task {task_id}",
                    status=TaskStatus.RUNNING,
                    runner_id=None,
                    lease_expires_at=None,
                )
            )
        session.commit()

    with engine.connect() as connection:
        connection.execute(
            text(
                f"INSERT INTO {TABLE} (task_id, command_id, kind, payload, status, "
                "created_at, updated_at) "
                "SELECT (g % 20) + 1, 'c' || g, "
                "CASE WHEN g % 3 = 0 THEN 'resume_input' ELSE 'start' END, "
                "'{}', "
                "CASE WHEN g > 3990 THEN 'pending' ELSE 'completed' END, "
                "NOW(), NOW() FROM generate_series(1, 4000) AS g"
            )
        )
        connection.execute(text(f"ANALYZE {TABLE}"))
        connection.commit()

    captured: list[tuple[str, object]] = []

    def record(_conn, _cursor, statement, parameters, *_args) -> None:
        if statement.lstrip().upper().startswith("SELECT"):
            captured.append((statement, parameters))

    event.listen(engine, "before_cursor_execute", record)
    try:
        found = transport._find_command_candidate(None)
    finally:
        event.remove(engine, "before_cursor_execute", record)

    assert found is not None
    assert found[2] is transport.TaskCommandKind.RESUME_INPUT
    statement, parameters = captured[0]
    with engine.connect() as connection:
        plan = "\n".join(
            row[0]
            for row in connection.exec_driver_sql(
                "EXPLAIN " + statement, parameters
            ).all()
        )
    assert INDEX in plan, plan


# Each defect is a same-named index the upgrade must drop and rebuild:
# (column list, WHERE clause or None for a non-partial index).
_DEFECTIVE_INDEXES = {
    "wrong_predicate": ("kind, id", "status = 'pending'"),
    "wrong_columns": ("id, kind", "status IN ('pending', 'processing')"),
    "non_partial": ("kind, id", None),
    "invalid": ("kind, id", "status IN ('pending', 'processing')"),
}


@pytest.mark.postgresql
@pytest.mark.parametrize("defect", list(_DEFECTIVE_INDEXES))
def test_postgresql_upgrade_rebuilds_an_invalid_or_mismatched_index(
    postgresql_engine_factory, defect
) -> None:
    migration = load_migration_module(
        MIGRATION_PATH, f"claimable_kind_index_rebuild_{defect}"
    )
    engine = postgresql_engine_factory(defect)
    columns, where = _DEFECTIVE_INDEXES[defect]

    with engine.connect() as connection:
        _create_pg_table(connection)
        connection.execute(
            text(
                f"CREATE INDEX {INDEX} ON {TABLE} ({columns})"
                + ("" if where is None else f" WHERE {where}")
            )
        )
        if defect == "invalid":
            connection.execute(
                text(
                    "UPDATE pg_catalog.pg_index SET indisvalid = false "
                    "WHERE indexrelid = pg_catalog.to_regclass(:name)"
                ),
                {"name": INDEX},
            )
        connection.commit()
        before = _pg_index_state(connection)
        assert before[0] is (defect != "invalid")

        _run(migration, connection, "upgrade")

        valid, predicate = _pg_index_state(connection)
        assert valid is True
        assert migration._postgres_predicate_matches_expected(predicate), predicate
        assert _pg_index_columns(connection) == INDEX_COLUMNS


def test_offline_sql_rendering_emits_the_partial_index() -> None:
    from io import StringIO

    migration = load_migration_module(MIGRATION_PATH, "claimable_kind_index_offline")

    def _offline_sql(dialect: str, operation: str) -> str:
        output = StringIO()
        context = MigrationContext.configure(
            dialect_name=dialect,
            opts={"as_sql": True, "output_buffer": output},
        )
        with Operations.context(context):
            getattr(migration, operation)()
        return output.getvalue()

    upgrade_sql = _offline_sql("postgresql", "upgrade")
    assert f"CREATE INDEX CONCURRENTLY {INDEX} ON {TABLE}" in upgrade_sql
    assert "WHERE status IN ('pending', 'processing')" in upgrade_sql
    assert f"DROP INDEX CONCURRENTLY {INDEX}" in _offline_sql("postgresql", "downgrade")
    assert "WHERE status IN ('pending', 'processing')" in _offline_sql(
        "sqlite", "upgrade"
    )
