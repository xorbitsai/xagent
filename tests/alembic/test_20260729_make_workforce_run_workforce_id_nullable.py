"""Tests for the workforce_runs.workforce_id nullable migration (preview runs)."""

import importlib.util
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20260729_make_workforce_run_workforce_id_nullable.py"
)
TABLE = "workforce_runs"
COLUMN = "workforce_id"


def _load_migration_module():
    spec = importlib.util.spec_from_file_location(
        "make_workforce_run_workforce_id_nullable_migration", MIGRATION_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection) -> Operations:
    return Operations(MigrationContext.configure(connection))


def _create_legacy_schema(connection) -> None:
    """Migration-only schema (no create_all()) -- workforce_runs has FKs to
    all three of workforces/tasks/users, the same shape as the migration
    that needed reflect_kwargs=BATCH_REFLECT_KWARGS."""
    connection.exec_driver_sql("CREATE TABLE workforces (id INTEGER PRIMARY KEY)")
    connection.exec_driver_sql("CREATE TABLE tasks (id INTEGER PRIMARY KEY)")
    connection.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY)")
    connection.exec_driver_sql(
        f"""
        CREATE TABLE {TABLE} (
            id INTEGER PRIMARY KEY,
            workforce_id INTEGER NOT NULL REFERENCES workforces(id) ON DELETE CASCADE,
            task_id INTEGER UNIQUE REFERENCES tasks(id) ON DELETE SET NULL,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            status VARCHAR(20) NOT NULL DEFAULT 'pending',
            is_preview BOOLEAN NOT NULL DEFAULT 0,
            idempotency_key VARCHAR(128),
            snapshot JSON NOT NULL
        )
        """
    )
    connection.exec_driver_sql(
        f"CREATE UNIQUE INDEX uq_workforce_run_idempotency "
        f"ON {TABLE} (workforce_id, idempotency_key)"
    )
    connection.execute(sa.text("INSERT INTO workforces (id) VALUES (7)"))
    connection.execute(sa.text("INSERT INTO users (id) VALUES (1)"))
    connection.execute(sa.text("INSERT INTO tasks (id) VALUES (100), (101)"))
    connection.execute(
        sa.text(
            f"INSERT INTO {TABLE} "
            "(id, workforce_id, task_id, user_id, status, is_preview, snapshot) "
            "VALUES (1, 7, 100, 1, 'completed', 0, '{}')"
        )
    )


def _create_schema_without_tasks_table(connection) -> None:
    """Same shape as ``_create_legacy_schema`` but omits ``tasks`` entirely --
    reproducing the CI migration-integration harness, which upgrades a truly
    empty database through the whole revision history with no ORM
    ``create_all()`` step. No migration in this repo actually creates
    ``tasks`` (it predates Alembic), so on that harness it never exists.

    ``task_id`` still declares ``REFERENCES tasks(id)`` even though ``tasks``
    doesn't exist -- SQLite allows creating the constraint regardless (unlike
    Postgres) -- so this fixture actually reproduces the dangling-FK-target
    shape ``BATCH_REFLECT_KWARGS`` exists for. Without it, this test would
    pass identically whether or not ``resolve_fks=False`` were removed.
    """
    connection.exec_driver_sql("CREATE TABLE workforces (id INTEGER PRIMARY KEY)")
    connection.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY)")
    connection.exec_driver_sql(
        f"""
        CREATE TABLE {TABLE} (
            id INTEGER PRIMARY KEY,
            workforce_id INTEGER NOT NULL REFERENCES workforces(id) ON DELETE CASCADE,
            task_id INTEGER UNIQUE REFERENCES tasks(id) ON DELETE SET NULL,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            status VARCHAR(20) NOT NULL DEFAULT 'pending',
            is_preview BOOLEAN NOT NULL DEFAULT 0,
            idempotency_key VARCHAR(128),
            snapshot JSON NOT NULL
        )
        """
    )
    connection.exec_driver_sql(
        f"CREATE UNIQUE INDEX uq_workforce_run_idempotency "
        f"ON {TABLE} (workforce_id, idempotency_key)"
    )
    connection.execute(sa.text("INSERT INTO workforces (id) VALUES (7)"))
    connection.execute(sa.text("INSERT INTO users (id) VALUES (1)"))
    connection.execute(
        sa.text(
            f"INSERT INTO {TABLE} "
            "(id, workforce_id, task_id, user_id, status, is_preview, snapshot) "
            "VALUES (1, 7, 100, 1, 'completed', 0, '{}')"
        )
    )


def _columns(connection) -> dict[str, dict]:
    return {
        column["name"]: column for column in sa.inspect(connection).get_columns(TABLE)
    }


def _indexes(connection) -> dict[str, dict]:
    return {index["name"]: index for index in sa.inspect(connection).get_indexes(TABLE)}


def test_upgrade_makes_column_nullable_and_preserves_constraints() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        _create_legacy_schema(connection)
        with Operations.context(_operations(connection).get_context()):
            migration.upgrade()

        columns = _columns(connection)
        assert columns[COLUMN]["nullable"] is True

        # A preview run (workforce_id NULL) must now be insertable.
        connection.execute(
            sa.text(
                f"INSERT INTO {TABLE} "
                "(id, workforce_id, task_id, user_id, status, is_preview, snapshot) "
                "VALUES (2, NULL, 101, 1, 'pending', 1, '{}')"
            )
        )

        # uq_workforce_run_idempotency (unique index) must survive the
        # batch_alter_table rebuild.
        indexes = _indexes(connection)
        assert "uq_workforce_run_idempotency" in indexes
        assert bool(indexes["uq_workforce_run_idempotency"]["unique"])

        # task_id's uniqueness must also survive -- reusing task 100 (already
        # bound to run 1) must fail.
        with pytest.raises(Exception):
            connection.execute(
                sa.text(
                    f"INSERT INTO {TABLE} "
                    "(id, workforce_id, task_id, user_id, status, is_preview, snapshot) "
                    "VALUES (3, 7, 100, 1, 'pending', 0, '{}')"
                )
            )


def test_upgrade_is_idempotent() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        _create_legacy_schema(connection)
        with Operations.context(_operations(connection).get_context()):
            migration.upgrade()
            migration.upgrade()

        columns = _columns(connection)
        assert columns[COLUMN]["nullable"] is True


def test_downgrade_drops_preview_runs_and_leaves_their_tasks_as_orphans() -> None:
    """PR #1060 review, F3: an earlier version of this migration deleted the
    preview run's paired Task row directly, which raises a Postgres FK
    violation for any preview that executed even one turn (DAGExecution,
    TraceEvent, TraceMessageBlob, TraceCheckpointBlob all reference tasks.id
    with no ondelete clause). The fix leaves the hidden orphaned Task row in
    place -- harmless data debris, not a functional issue -- rather than
    trying to cascade-delete its children from a migration."""
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        _create_legacy_schema(connection)
        operations = _operations(connection)
        with Operations.context(operations.get_context()):
            migration.upgrade()

        connection.execute(
            sa.text(
                f"INSERT INTO {TABLE} "
                "(id, workforce_id, task_id, user_id, status, is_preview, snapshot) "
                "VALUES (2, NULL, 101, 1, 'pending', 1, '{}')"
            )
        )

        with Operations.context(operations.get_context()):
            migration.downgrade()

        columns = _columns(connection)
        assert columns[COLUMN]["nullable"] is False

        remaining_runs = connection.execute(
            sa.text(f"SELECT id FROM {TABLE} ORDER BY id")
        ).fetchall()
        assert [row[0] for row in remaining_runs] == [1]

        # Both tasks survive: 100 belongs to the surviving real workforce
        # run, and 101 -- the deleted preview run's Task -- is now a
        # deliberately-left orphan rather than a deleted row.
        remaining_tasks = connection.execute(
            sa.text("SELECT id FROM tasks ORDER BY id")
        ).fetchall()
        assert [row[0] for row in remaining_tasks] == [100, 101]


def test_downgrade_restores_not_null_when_tasks_table_is_missing() -> None:
    """Regression: a real CI run downgrading the full migration history from
    an empty database (never ORM-``create_all``'d) has no ``tasks`` table at
    all, since no migration creates it -- the raw DELETE against ``tasks``
    must not blow up with "relation tasks does not exist".
    """
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        _create_schema_without_tasks_table(connection)
        operations = _operations(connection)
        with Operations.context(operations.get_context()):
            migration.upgrade()

        connection.execute(
            sa.text(
                f"INSERT INTO {TABLE} "
                "(id, workforce_id, task_id, user_id, status, is_preview, snapshot) "
                "VALUES (2, NULL, 101, 1, 'pending', 1, '{}')"
            )
        )

        with Operations.context(operations.get_context()):
            migration.downgrade()

        columns = _columns(connection)
        assert columns[COLUMN]["nullable"] is False

        remaining_runs = connection.execute(
            sa.text(f"SELECT id FROM {TABLE} ORDER BY id")
        ).fetchall()
        assert [row[0] for row in remaining_runs] == [1]


def test_downgrade_skips_the_alter_when_column_nullability_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PR #1060 round 7 review, finding #6: _column_nullable returns None
    when the column can't be found in the reflected table (e.g. pre-existing
    schema drift). upgrade()'s guard (`is False`) already treats None as
    "don't touch it"; downgrade()'s guard used to read `is not False`, which
    treats the same None as "go ahead and alter" -- surfacing a confusing
    SQLAlchemy/Alembic error instead of a clean skip.

    Patched directly rather than dropping the workforce_id column for real:
    the earlier `DELETE FROM workforce_runs WHERE workforce_id IS NULL` in
    downgrade() would itself crash with "no such column" first if the column
    were genuinely absent, for a reason unrelated to this specific guard.
    """
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        _create_legacy_schema(connection)
        operations = _operations(connection)
        with Operations.context(operations.get_context()):
            migration.upgrade()

        monkeypatch.setattr(migration, "_column_nullable", lambda *a, **k: None)

        with Operations.context(operations.get_context()):
            migration.downgrade()

        # Still nullable (from the earlier upgrade): downgrade's alter-to-
        # NOT-NULL step was correctly skipped, not attempted against an
        # unknown nullability state.
        columns = _columns(connection)
        assert columns[COLUMN]["nullable"] is True


def test_upgrade_and_downgrade_skip_when_table_missing() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        with Operations.context(_operations(connection).get_context()):
            migration.upgrade()
            migration.downgrade()

        assert TABLE not in sa.inspect(connection).get_table_names()
