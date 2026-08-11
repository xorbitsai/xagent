"""Schema-parity tests for tasks.interaction_protocol_version.

The model's ``CheckConstraint`` in ``Task.__table_args__`` and the
``20260810_add_task_interaction_protocol_version`` migration are two
independent implementations of the same constraint: one runs whenever a test
calls ``Base.metadata.create_all()``, the other runs when a real database is
carried forward through the Alembic revision chain. Nothing keeps them in
sync automatically -- this module is that check.

The comparison is same-backend only: create_all-built SQLite against
migration-shaped SQLite, and create_all-built PostgreSQL against
migration-shaped PostgreSQL. It never compares a SQLite CHECK sqltext
against a PostgreSQL one. PostgreSQL is known to rewrite some CHECK
expressions on reflection (e.g. an ``IN (...)`` list becomes
``= ANY (ARRAY[...])``); this constraint's simple comparison happens not to
trigger that rewrite, but the comparison does not rely on that being true
forever.

Both PostgreSQL fixtures mint their own throwaway database via
``disposable_database_factory`` and drop it on teardown. Neither ever
touches the database XAGENT_TEST_POSTGRES_URL names -- that database hosts
fixtures other suites depend on. ``tests/migrations/test_migration_integration.py``
and ``tests/web/services/test_task_interaction_schema_postgresql.py`` are the
reason that rule exists: both run ``DROP SCHEMA public CASCADE`` /
``Base.metadata.drop_all()`` against whatever database that environment
variable names, so exporting it in a shell that also runs either of those
suites is destructive.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

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
from xagent.web.models.task import Task

MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20260810_add_task_interaction_protocol_version.py"
)
TABLE = "tasks"
COLUMN = "interaction_protocol_version"
CONSTRAINT_NAME = "ck_tasks_interaction_protocol_version"

# Strip the column definition and its CHECK constraint out of a create_all
# -rendered CREATE TABLE statement, to reach the shape a database carried
# through the real revision chain would have (SQLite never receives the
# CHECK -- see the module docstring and T-M-5 below). Same rename/recreate
# technique as reset_checkpoint_anchor_fk_create_rule's sibling helper in
# tests/web/services/checkpoint_anchor_shared.py, extended to also drop a
# column rather than just one FK clause.
_COLUMN_CLAUSE = re.compile(r"\s*interaction_protocol_version INTEGER,\s*")
_CHECK_CLAUSE = re.compile(
    r",\s*CONSTRAINT ck_tasks_interaction_protocol_version CHECK \([^)]*\)"
)


def _operations(connection: sa.Connection) -> Operations:
    return Operations(MigrationContext.configure(connection))


# ---- backend-free: the model and the migration agree on the CHECK text ----


def test_model_and_migration_constraint_texts_are_identical() -> None:
    """The model's CheckConstraint and the migration's CONSTRAINT_CONDITION
    are two copies of one contract; today they agree purely by discipline.
    This needs no backend: a one-character divergence fails here before any
    database ever sees either copy."""
    model_conditions = {
        item.name: str(item.sqltext)
        for item in Task.__table_args__
        if type(item).__name__ == "CheckConstraint"
    }

    migration = load_migration_module(MIGRATION_PATH)

    assert model_conditions[CONSTRAINT_NAME] == migration.CONSTRAINT_CONDITION


def _column_names(connection) -> set[str]:
    return {column["name"] for column in sa.inspect(connection).get_columns(TABLE)}


def _reflect_narrow_inventory(engine) -> dict:
    """The only slice of the tasks schema this migration touches: whether
    the column exists and is nullable, and the {name: sqltext} of every
    ck_tasks_* CHECK constraint (prefix-filtered so unrelated CHECKs already
    on tasks never enter the comparison)."""
    inspector = sa.inspect(engine)
    columns = {c["name"]: c for c in inspector.get_columns(TABLE)}
    checks = {
        str(item["name"]): str(item["sqltext"])
        for item in inspector.get_check_constraints(TABLE)
        if item.get("name") and str(item["name"]).startswith("ck_tasks_")
    }
    return {
        "column_present": COLUMN in columns,
        "column_nullable": columns[COLUMN]["nullable"] if COLUMN in columns else None,
        "checks": checks,
    }


def _diff_narrow_inventory(left: dict, right: dict) -> list[str]:
    """Human-readable differences between two narrow inventories, or an
    empty list when they match exactly. A comparison that only checks
    constraint *names* would pass even if a CHECK's condition were silently
    weakened -- this compares the {name: sqltext} maps, not just their key
    sets."""
    differences = []
    if left["column_present"] != right["column_present"]:
        differences.append(
            f"column present: {left['column_present']!r} != {right['column_present']!r}"
        )
    if left["column_nullable"] != right["column_nullable"]:
        differences.append(
            f"column nullable: {left['column_nullable']!r} != "
            f"{right['column_nullable']!r}"
        )
    if left["checks"] != right["checks"]:
        differences.append(f"checks: {left['checks']!r} != {right['checks']!r}")
    return differences


def _build_migration_shaped_sqlite_engine():
    """create_all(), then strip the column and its CHECK via SQLite's
    rename/recreate/copy/drop procedure, then run the migration's upgrade()
    -- so the resulting schema is what a database walked through the real
    revision chain would have, not create_all wearing a different name."""
    reset_checkpoint_anchor_fk_create_rule()
    engine = sa.create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with engine.begin() as connection:
        original_sql = connection.execute(
            sa.text("SELECT sql FROM sqlite_master WHERE type='table' AND name=:name"),
            {"name": TABLE},
        ).scalar()
        assert original_sql is not None

        stripped_sql, check_hits = _CHECK_CLAUSE.subn("", original_sql)
        assert check_hits == 1, (
            f"expected exactly one {CONSTRAINT_NAME} clause in the tasks "
            f"DDL, found {check_hits}"
        )
        stripped_sql, column_hits = _COLUMN_CLAUSE.subn("", stripped_sql)
        assert column_hits == 1, (
            f"expected exactly one {COLUMN} column clause in the tasks "
            f"DDL, found {column_hits}"
        )
        assert COLUMN not in stripped_sql

        legacy_columns = [name for name in _column_names(connection) if name != COLUMN]
        column_list = ", ".join(legacy_columns)

        connection.execute(sa.text(f"ALTER TABLE {TABLE} RENAME TO {TABLE}_old"))
        connection.execute(sa.text(stripped_sql))
        connection.execute(
            sa.text(
                f"INSERT INTO {TABLE} ({column_list}) "
                f"SELECT {column_list} FROM {TABLE}_old"
            )
        )
        connection.execute(sa.text(f"DROP TABLE {TABLE}_old"))

        migration = load_migration_module(MIGRATION_PATH)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

    return engine


def _build_migration_shaped_postgresql_engine(engine):
    """create_all(), then use the migration's own downgrade() to strip the
    column and CHECK back off (PostgreSQL can do this; SQLite cannot -- see
    T-M-5), then upgrade() to rebuild them through the migration path rather
    than through create_all."""
    reset_checkpoint_anchor_fk_create_rule()
    Base.metadata.create_all(engine)

    migration = load_migration_module(MIGRATION_PATH)
    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()
            migration.upgrade()

    return engine


def _build_create_all_sqlite_engine():
    reset_checkpoint_anchor_fk_create_rule()
    engine = sa.create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


# ---- T-M-2a / T-M-2b: SQLite -- column parity, CHECK asymmetry ----


def test_sqlite_column_matches_between_migration_and_create_all() -> None:
    migration_engine = _build_migration_shaped_sqlite_engine()
    create_all_engine = _build_create_all_sqlite_engine()

    migration_inv = _reflect_narrow_inventory(migration_engine)
    create_all_inv = _reflect_narrow_inventory(create_all_engine)

    assert migration_inv["column_present"] is True
    assert create_all_inv["column_present"] is True
    assert migration_inv["column_nullable"] == create_all_inv["column_nullable"]


def test_sqlite_check_asymmetry_is_expected() -> None:
    """SQLite cannot receive this CHECK through the migration path: adding a
    constraint to an existing table raises NotImplementedError on SQLite
    both online and offline, and the batch_alter_table workaround cannot run
    in --sql mode on either dialect, which the offline-SQL requirement rules
    out (see the migration and the model's __table_args__ comment). So a
    migration-shaped SQLite database has no ck_tasks_interaction_protocol_version
    CHECK, while a create_all-built one does (it comes straight from the
    model). This asymmetry is expected, not a defect: if this assertion goes
    red, check whether someone changed the SQLite branch of the migration to
    emit the CHECK -- that would break `alembic upgrade --sql` on SQLite.
    """
    migration_engine = _build_migration_shaped_sqlite_engine()
    create_all_engine = _build_create_all_sqlite_engine()

    migration_checks = _reflect_narrow_inventory(migration_engine)["checks"]
    create_all_checks = _reflect_narrow_inventory(create_all_engine)["checks"]

    assert CONSTRAINT_NAME not in migration_checks
    assert CONSTRAINT_NAME in create_all_checks


# ---- T-M-2c: PostgreSQL -- column and CHECK match exactly ----


@pytest.mark.postgresql
def test_postgresql_column_and_check_match_between_migration_and_create_all() -> None:
    with disposable_database_factory("xagent_w1_parity") as make_database:
        migration_engine = _build_migration_shaped_postgresql_engine(
            make_database("migration_side")
        )
        create_all_engine = _build_create_all_engine_postgresql(
            make_database("create_all_side")
        )

        migration_inv = _reflect_narrow_inventory(migration_engine)
        create_all_inv = _reflect_narrow_inventory(create_all_engine)

        differences = _diff_narrow_inventory(migration_inv, create_all_inv)
        assert differences == []


def _build_create_all_engine_postgresql(engine):
    reset_checkpoint_anchor_fk_create_rule()
    Base.metadata.create_all(engine)
    return engine


# ---- T-M-2d: the comparator itself catches a changed CHECK expression ----


def test_diff_narrow_inventory_flags_a_changed_check_expression() -> None:
    baseline = {
        "column_present": True,
        "column_nullable": True,
        "checks": {
            CONSTRAINT_NAME: (
                "interaction_protocol_version IS NULL "
                "OR interaction_protocol_version = 1"
            )
        },
    }
    changed = {
        "column_present": True,
        "column_nullable": True,
        "checks": {
            CONSTRAINT_NAME: (
                "interaction_protocol_version IS NULL "
                "OR interaction_protocol_version >= 1"
            )
        },
    }

    differences = _diff_narrow_inventory(baseline, changed)

    assert differences
    assert any("checks" in difference for difference in differences)


def test_diff_narrow_inventory_reports_no_differences_for_identical_inventories() -> (
    None
):
    inventory = {
        "column_present": True,
        "column_nullable": True,
        "checks": {
            CONSTRAINT_NAME: "interaction_protocol_version IS NULL OR interaction_protocol_version = 1"
        },
    }

    assert _diff_narrow_inventory(inventory, dict(inventory)) == []


# ---- T-M-5: create_all SQLite cannot be downgraded; migration-chain can ----


def test_create_all_sqlite_downgrade_fails_but_migration_shaped_sqlite_succeeds() -> (
    None
):
    """SQLite refuses to DROP a column that a CHECK constraint references,
    so calling downgrade() on a create_all-built SQLite database raises
    OperationalError("no such column: interaction_protocol_version"). This
    is a direct consequence of the CHECK asymmetry above, not a bug: it only
    happens on a shape that never goes through a real downgrade. A database
    maintained through the revision chain never received the CHECK on
    SQLite in the first place (T-M-1b/T-M-1d), so its downgrade() has no
    CHECK-referenced column to trip over. Both halves are asserted here so a
    reader who only sees the first half does not read it as a defect to fix
    -- "fixing" it by making downgrade() smarter would not change the fact
    that no real database ever hits this path.
    """
    migration = load_migration_module(MIGRATION_PATH)

    create_all_engine = _build_create_all_sqlite_engine()
    with create_all_engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            with pytest.raises(sa.exc.OperationalError, match="no such column"):
                migration.downgrade()

    migration_engine = _build_migration_shaped_sqlite_engine()
    with migration_engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()
            assert COLUMN not in _column_names(connection)
