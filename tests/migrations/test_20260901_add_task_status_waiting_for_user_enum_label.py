"""The taskstatus enum repair (20260901_taskstatus_waiting_for_user).

``TaskStatus.WAITING_FOR_USER`` was added to the application enum after this
project had already been deployed, and PostgreSQL fixes a native enum's
labels at creation time. This file covers the revision that repairs such a
database, including the end-to-end shape the check exists for: a database
whose enum predates the label starts, migrates, and passes
``check_task_status_enum_drift`` -- rather than aborting startup with no
shipped repair.

The PostgreSQL cells mint their own disposable databases (the plumbing this
repository already centralizes in tests/shared/postgres_disposable.py), so
nothing here touches the base database XAGENT_TEST_POSTGRES_URL names.
"""

from __future__ import annotations

from contextlib import nullcontext
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import text

from tests.shared.postgres_disposable import (
    disposable_database_factory,
    load_migration_module,
)
from xagent.web.models.database import Base, _initialize_database_schema
from xagent.web.models.task import (
    TASKSTATUS_ENUM_REPAIR_REVISION,
    TaskStatus,
    TaskStatusEnumDriftError,
    check_task_status_enum_drift,
)

MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions"
    / "20260901_add_task_status_waiting_for_user_enum_label.py"
)
LABEL = "WAITING_FOR_USER"
ALL_LABELS = [member.name for member in TaskStatus]
LEGACY_LABELS = [name for name in ALL_LABELS if name != LABEL]

_COLUMN_LABELS_SQL = text(
    "SELECT e.enumlabel "
    "FROM pg_catalog.pg_attribute a "
    "JOIN pg_catalog.pg_enum e ON e.enumtypid = a.atttypid "
    "WHERE a.attrelid = pg_catalog.to_regclass('tasks') "
    "AND a.attname = 'status'"
)


def _column_labels(connectable) -> list[str]:
    with connectable.connect() as connection:
        return sorted(connection.scalars(_COLUMN_LABELS_SQL))


def _create_legacy_enum(connectable, labels: list[str] = LEGACY_LABELS) -> None:
    """A ``taskstatus`` type carrying exactly ``labels`` -- the shape a
    database initialized before the label existed still has."""
    body = ", ".join(f"'{label}'" for label in labels)
    with connectable.begin() as connection:
        connection.execute(text(f"CREATE TYPE taskstatus AS ENUM ({body})"))


def _create_legacy_tasks_table(connectable) -> None:
    """Migration-only schema: the one column this revision resolves."""
    with connectable.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE tasks (id SERIAL PRIMARY KEY, "
                "status taskstatus NOT NULL DEFAULT 'PENDING')"
            )
        )


def _stamp(connectable, revision: str) -> None:
    with connectable.begin() as connection:
        connection.execute(
            text("CREATE TABLE alembic_version (version_num VARCHAR(255) NOT NULL)")
        )
        connection.execute(
            text("INSERT INTO alembic_version (version_num) VALUES (:revision)"),
            {"revision": revision},
        )


def _run_module_upgrade(connectable) -> None:
    """Drive the revision module directly.

    ``autocommit_block`` asserts that the migration context owns a
    transaction, which only holds when Alembic itself is driving; the same
    substitution tests/alembic/test_20260725_add_task_lease_recovery_index.py
    makes. The end-to-end cell below runs the real runner, so the autocommit
    path is not left untested by this substitution.
    """
    migration = load_migration_module(MIGRATION_PATH)
    with connectable.connect() as connection:
        context = MigrationContext.configure(connection)
        with patch.object(context, "autocommit_block", nullcontext):
            with patch.object(migration, "op", Operations(context)):
                with Operations.context(context):
                    migration.upgrade()
        connection.commit()


def _offline_sql(dialect_name: str, operation: str = "upgrade") -> str:
    migration = load_migration_module(MIGRATION_PATH)
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name=dialect_name,
        opts={"as_sql": True, "output_buffer": output},
    )
    with Operations.context(context):
        with patch.object(migration, "op", Operations(context)):
            getattr(migration, operation)()
    return output.getvalue()


@pytest.fixture()
def postgresql_engine_factory():
    with disposable_database_factory("xagent_taskstatus_enum_repair") as make:
        yield make


# ---- the revision's identity ----


def test_the_startup_remedy_names_this_revision() -> None:
    """``check_task_status_enum_drift`` tells the operator which revision
    adds the label. Renaming this file or its revision id without updating
    that constant would leave the startup failure pointing at nothing --
    which is the exact defect this revision exists to remove."""
    migration = load_migration_module(MIGRATION_PATH)

    assert migration.revision == TASKSTATUS_ENUM_REPAIR_REVISION


# ---- end-to-end: an old database starts ----


@pytest.mark.postgresql
def test_a_pre_waiting_for_user_database_starts_after_upgrading_to_head(
    postgresql_engine_factory,
) -> None:
    """The shape this revision exists for, driven through the production
    startup path rather than through the revision module: a database whose
    native enum predates ``WAITING_FOR_USER`` runs
    ``_initialize_database_schema`` -- startup lock, migrations, create_all,
    drift check -- and comes out able to serve.
    """
    engine = postgresql_engine_factory("upgrade_from_old_schema")
    migration = load_migration_module(MIGRATION_PATH)

    _create_legacy_enum(engine)
    Base.metadata.create_all(bind=engine)
    _stamp(engine, migration.down_revision)

    assert _column_labels(engine) == sorted(LEGACY_LABELS)
    with engine.connect() as connection:
        with pytest.raises(TaskStatusEnumDriftError):
            check_task_status_enum_drift(connection)

    _initialize_database_schema(engine)

    assert _column_labels(engine) == sorted(ALL_LABELS)
    with engine.connect() as connection:
        check_task_status_enum_drift(connection)  # must not raise
    with engine.begin() as connection:
        version = connection.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalar_one()
        user_id = connection.execute(
            text(
                "INSERT INTO users (username, password_hash, is_admin) "
                "VALUES ('taskstatus_enum_repair', 'x', false) RETURNING id"
            )
        ).scalar_one()
        connection.execute(
            text(
                "INSERT INTO tasks (user_id, title, status) "
                "VALUES (:user_id, 'taskstatus enum repair smoke row', :label)"
            ),
            {"user_id": user_id, "label": LABEL},
        )
    assert version == migration.revision


# ---- the revision's own branches ----


@pytest.mark.postgresql
def test_online_upgrade_adds_the_label_idempotently(postgresql_engine_factory) -> None:
    engine = postgresql_engine_factory("legacy_enum")
    _create_legacy_enum(engine)
    _create_legacy_tasks_table(engine)

    _run_module_upgrade(engine)
    _run_module_upgrade(engine)

    assert _column_labels(engine) == sorted(ALL_LABELS)


@pytest.mark.postgresql
def test_online_upgrade_leaves_a_complete_enum_untouched(
    postgresql_engine_factory,
) -> None:
    engine = postgresql_engine_factory("complete_enum")
    _create_legacy_enum(engine, ALL_LABELS)
    _create_legacy_tasks_table(engine)

    _run_module_upgrade(engine)

    assert _column_labels(engine) == sorted(ALL_LABELS)


@pytest.mark.postgresql
@pytest.mark.parametrize(
    ("tag", "schema_sql"),
    [
        ("no_tasks_table", None),
        (
            "status_is_not_an_enum",
            "CREATE TABLE tasks (id SERIAL PRIMARY KEY, status VARCHAR(32))",
        ),
    ],
)
def test_online_upgrade_is_a_noop_without_a_native_enum_column(
    postgresql_engine_factory, tag: str, schema_sql: str | None
) -> None:
    """Neither shape is something ALTER TYPE can repair. A fresh database is
    the first one: create_all builds the type with every current label right
    after the migrations run."""
    engine = postgresql_engine_factory(tag)
    if schema_sql is not None:
        with engine.begin() as connection:
            connection.execute(text(schema_sql))

    _run_module_upgrade(engine)  # must not raise

    assert _column_labels(engine) == []


@pytest.mark.postgresql
def test_online_upgrade_repairs_the_columns_own_type_under_a_shadowing_search_path(
    postgresql_engine_factory,
) -> None:
    """``search_path`` resolves relation names and type names separately, so
    a complete ``taskstatus`` sitting ahead of the real one can hide a
    genuinely missing label from a lookup that matches on the name. This
    revision resolves the type through ``tasks.status`` -- the same
    resolution ``check_task_status_enum_drift`` makes -- so the repair lands
    on the type the column actually uses and the shadow copy is untouched.
    """
    engine = postgresql_engine_factory("shadow_search_path")
    complete = ", ".join(f"'{label}'" for label in ALL_LABELS)
    legacy = ", ".join(f"'{label}'" for label in LEGACY_LABELS)
    with engine.begin() as connection:
        connection.execute(text("CREATE SCHEMA shadow"))
        connection.execute(text("CREATE SCHEMA app"))
        connection.execute(text(f"CREATE TYPE shadow.taskstatus AS ENUM ({complete})"))
        connection.execute(text(f"CREATE TYPE app.taskstatus AS ENUM ({legacy})"))
        connection.execute(
            text(
                "CREATE TABLE app.tasks (id SERIAL PRIMARY KEY, "
                "status app.taskstatus NOT NULL DEFAULT 'PENDING')"
            )
        )

    migration = load_migration_module(MIGRATION_PATH)
    with engine.connect() as connection:
        connection.execute(text("SET search_path TO shadow, app"))
        connection.commit()
        context = MigrationContext.configure(connection)
        with patch.object(context, "autocommit_block", nullcontext):
            with patch.object(migration, "op", Operations(context)):
                with Operations.context(context):
                    migration.upgrade()
        connection.commit()

    with engine.connect() as connection:
        connection.execute(text("SET search_path TO shadow, app"))
        assert sorted(connection.scalars(_COLUMN_LABELS_SQL)) == sorted(ALL_LABELS)


# ---- non-PostgreSQL and offline ----


def test_sqlite_upgrade_and_downgrade_are_noops() -> None:
    """SQLite stores the Enum column as a plain string: there is no native
    type to repair, and the whole migration integration suite runs this
    revision on SQLite."""
    migration = load_migration_module(MIGRATION_PATH)
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        connection.execute(
            text("CREATE TABLE tasks (id INTEGER PRIMARY KEY, status VARCHAR(32))")
        )
        context = MigrationContext.configure(connection)
        with patch.object(migration, "op", Operations(context)):
            with Operations.context(context):
                migration.upgrade()
                migration.downgrade()

        columns = {c["name"] for c in sa.inspect(connection).get_columns("tasks")}
    assert columns == {"id", "status"}


def test_offline_postgresql_upgrade_emits_an_idempotent_add_value() -> None:
    sql = _offline_sql("postgresql")

    assert f"ALTER TYPE taskstatus ADD VALUE IF NOT EXISTS '{LABEL}'" in sql
    assert "%(" not in sql and ":table_name" not in sql


def test_offline_sqlite_upgrade_emits_nothing() -> None:
    assert _offline_sql("sqlite").strip() == ""


def test_offline_downgrade_emits_nothing_on_either_dialect() -> None:
    """PostgreSQL has no ALTER TYPE ... DROP VALUE; the downgrade is a no-op
    on purpose, and the migration integration suite downgrades the whole
    chain."""
    assert _offline_sql("postgresql", "downgrade").strip() == ""
    assert _offline_sql("sqlite", "downgrade").strip() == ""
