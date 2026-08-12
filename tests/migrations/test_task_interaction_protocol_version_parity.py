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
        if isinstance(item, sa.CheckConstraint)
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

        # Without legacy_alter_table, SQLite's RENAME rewrites every sibling
        # table's stored FK DDL to point at the renamed name (tasks_old),
        # and the DROP TABLE below then leaves the whole schema with
        # dangling references -- any later operation that reflects the
        # dependency graph (op.batch_alter_table does) fails with
        # NoSuchTableError. legacy_alter_table=ON makes RENAME touch only
        # the tasks table itself, matching a real ALTER TABLE RENAME.
        connection.execute(sa.text("PRAGMA legacy_alter_table=ON"))
        connection.execute(sa.text(f"ALTER TABLE {TABLE} RENAME TO {TABLE}_old"))
        connection.execute(sa.text("PRAGMA legacy_alter_table=OFF"))
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


def _build_create_all_engine(engine):
    reset_checkpoint_anchor_fk_create_rule()
    Base.metadata.create_all(engine)
    return engine


# ---- T-M-2a / T-M-2b: SQLite -- column parity, CHECK asymmetry ----


def test_sqlite_column_matches_between_migration_and_create_all() -> None:
    migration_engine = _build_migration_shaped_sqlite_engine()
    create_all_engine = _build_create_all_engine(sa.create_engine("sqlite:///:memory:"))

    migration_inv = _reflect_narrow_inventory(migration_engine)
    create_all_inv = _reflect_narrow_inventory(create_all_engine)

    assert migration_inv["column_present"] is True
    assert create_all_inv["column_present"] is True
    assert migration_inv["column_nullable"] == create_all_inv["column_nullable"]


def test_sqlite_check_asymmetry_is_expected() -> None:
    """Two SQLite deployments of the same xagent version can enforce
    different invariants on this column depending on install history: a
    fresh install (stamp head, then create_all) HAS the CHECK, while a
    database that walked the revision chain does NOT. Offline, upgrade()
    cannot add a CHECK to an existing table on SQLite: a plain CHECK-add
    raises NotImplementedError there, and the batch_alter_table rebuild
    that could add it cannot render under --sql mode on either dialect,
    which offline SQL support (a hard requirement) rules out. Online,
    batch_alter_table could add the CHECK to an existing SQLite table the
    same way downgrade() already removes it -- upgrade() does not do that
    today; that convergence is deliberately deferred to the first
    production writer of this column, tracked in #1290 (see the migration
    and the model's __table_args__ comment). Concretely: inserting
    interaction_protocol_version=2 succeeds on the chain-walked shape and
    raises IntegrityError on the fresh-install shape.
    This asymmetry is expected today because the column has zero writers --
    nothing has ever inserted a non-NULL value, so nothing observes the
    divergence. The change that adds the first writer to this column must
    either converge the two shapes (e.g. teach upgrade() to add the CHECK on
    SQLite via batch_alter_table the way downgrade() now removes it) or
    explicitly justify writing to a column whose constraint isn't uniformly
    enforced. If this assertion goes red because someone changed the SQLite
    branch of upgrade() to emit the CHECK, that is this convergence
    happening -- update this test to match, don't just re-pin the old
    asymmetry.
    """
    migration_engine = _build_migration_shaped_sqlite_engine()
    create_all_engine = _build_create_all_engine(sa.create_engine("sqlite:///:memory:"))

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
        create_all_engine = _build_create_all_engine(make_database("create_all_side"))

        migration_inv = _reflect_narrow_inventory(migration_engine)
        create_all_inv = _reflect_narrow_inventory(create_all_engine)

        differences = _diff_narrow_inventory(migration_inv, create_all_inv)
        assert differences == []


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


# ---- T-M-5: create_all SQLite cannot be downgraded; migration-chain can ----


def test_sqlite_downgrade_succeeds_on_both_install_shapes() -> None:
    """A fresh install stamps head and then builds its schema via create_all
    (see src/xagent/db/migration.py's empty-DB stamp path and
    src/xagent/web/models/database.py's create_all call, at its
    _initialize_database_schema site), so the create_all-built SQLite shape
    is a real downgrade-reachable production shape, not a synthetic one --
    and SQLite 3.35+ refuses to DROP a column a CHECK constraint references,
    so the old unconditional drop_column() broke downgrade() on that shape.
    downgrade() now rebuilds the table via batch_alter_table on SQLite (see the
    migration's downgrade() online branch), which drops the CHECK and the
    column together regardless of which shape the database has. Both shapes
    are asserted here: the migration-shaped database never had the CHECK to
    begin with (T-M-1b/T-M-1d), while the create_all-built one does and must
    have both the column and the CHECK clause removed from the table DDL.
    """
    migration = load_migration_module(MIGRATION_PATH)

    create_all_engine = _build_create_all_engine(sa.create_engine("sqlite:///:memory:"))
    with create_all_engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()
            assert COLUMN not in _column_names(connection)

        ddl = connection.execute(
            sa.text("SELECT sql FROM sqlite_master WHERE type='table' AND name=:name"),
            {"name": TABLE},
        ).scalar()
        assert CONSTRAINT_NAME not in ddl

    migration_engine = _build_migration_shaped_sqlite_engine()
    with migration_engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()
            assert COLUMN not in _column_names(connection)
