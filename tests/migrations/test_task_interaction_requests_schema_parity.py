"""create_all/migration schema parity for task_interaction_requests.

The model (src/xagent/web/models/task_interaction.py) and the
20260809_add_task_interaction_requests migration are two independent
implementations of the same constraint inventory: create_all builds the
table straight from the model, the migration builds it from a hand-copied
CHECKS tuple. Nothing enforces they stay identical except this suite.

Migration-side construction: create_all every table *except*
task_interaction_requests, then run only this migration's upgrade()
against the connection -- not the full alembic revision chain, and not
checkpoint_anchor_shared.build_upgraded_sqlite_engine(). That helper
builds an "upgraded, FK-less tasks table" shape for a completely
different migration (the checkpoint-anchor column) and never runs any
alembic revision at all; using it here would make the "migration" side of
every comparison below a plain create_all wearing a different name, which
would stay green regardless of what this migration's CHECKS or guards
said. reset_checkpoint_anchor_fk_create_rule() is still required before
every create_all() call in this file, independent of which construction
this module uses -- see its docstring for the cross-dialect FK-caching bug
it works around.

Comparisons are same-backend only: PostgreSQL rewrites CHECK expressions
on the way in (e.g. `status IN (...)` becomes `status::text = ANY
(ARRAY[...]::text[])`) while SQLite stores them verbatim, so two
backends' sqltext values are never compared to each other or to the
model's literal. Both PostgreSQL fixtures build their own disposable,
throwaway database from XAGENT_TEST_POSTGRES_URL via CREATE DATABASE --
never against the database that URL itself names, which other suites in
this repo drop and recreate wholesale (see test_migration_integration.py
and test_task_interaction_schema_postgresql.py).
"""

from __future__ import annotations

from pathlib import Path

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
from tests.web.services.task_interaction_schema_shared import (
    EXPECTED_CHECK_CONSTRAINT_NAMES,
    diff_full_inventory,
    reflect_full_inventory,
)
from xagent.web.models.database import Base
from xagent.web.models.task_interaction import TaskInteractionRequest

MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20260809_add_task_interaction_requests.py"
)
TABLE = "task_interaction_requests"


def _load_migration_module():
    return load_migration_module(
        MIGRATION_PATH, "add_task_interaction_requests_migration_parity"
    )


def _build_create_all_schema(engine: sa.engine.Engine) -> None:
    reset_checkpoint_anchor_fk_create_rule()
    Base.metadata.create_all(engine)


def _build_migration_schema(engine: sa.engine.Engine) -> None:
    """Every table except task_interaction_requests is create_all'd, then
    only this revision's upgrade() runs against the connection -- see the
    module docstring for why the alembic chain and
    build_upgraded_sqlite_engine() are both wrong here."""
    reset_checkpoint_anchor_fk_create_rule()
    other_tables = [
        table for name, table in Base.metadata.tables.items() if name != TABLE
    ]
    Base.metadata.create_all(engine, tables=other_tables)
    migration = _load_migration_module()
    with engine.begin() as connection:
        context = MigrationContext.configure(connection)
        with Operations.context(context):
            migration.upgrade()


def _downgrade(engine: sa.engine.Engine) -> None:
    migration = _load_migration_module()
    with engine.begin() as connection:
        context = MigrationContext.configure(connection)
        with Operations.context(context):
            migration.downgrade()


def _model_declared_names() -> dict[str, set[str]]:
    checks: set[str] = set()
    unique: set[str] = set()
    indexes: set[str] = set()
    for item in TaskInteractionRequest.__table_args__:
        kind = type(item).__name__
        if kind == "CheckConstraint":
            checks.add(item.name)
        elif kind == "UniqueConstraint":
            unique.add(item.name)
        elif kind == "Index":
            indexes.add(item.name)
    foreign_keys = {fk.name for fk in TaskInteractionRequest.__table__.foreign_keys}
    return {
        "checks": checks,
        "unique": unique,
        "indexes": indexes,
        "foreign_keys": foreign_keys,
    }


@pytest.fixture
def postgresql_engine_factory():
    with disposable_database_factory("xagent_b2_parity") as make:
        yield make


def test_sqlite_migration_schema_matches_create_all(tmp_path) -> None:
    create_all_engine = sa.create_engine(f"sqlite:///{tmp_path / 'createall.db'}")
    migration_engine = sa.create_engine(f"sqlite:///{tmp_path / 'migration.db'}")

    _build_create_all_schema(create_all_engine)
    _build_migration_schema(migration_engine)

    left = reflect_full_inventory(create_all_engine)
    right = reflect_full_inventory(migration_engine)
    diff = diff_full_inventory(left, right)
    assert not diff, diff
    assert len(left["checks"]) == len(EXPECTED_CHECK_CONSTRAINT_NAMES)
    assert len(right["checks"]) == len(EXPECTED_CHECK_CONSTRAINT_NAMES)


def test_postgresql_migration_schema_matches_create_all(
    postgresql_engine_factory,
) -> None:
    create_all_engine = postgresql_engine_factory("createall")
    migration_engine = postgresql_engine_factory("migration")

    _build_create_all_schema(create_all_engine)
    _build_migration_schema(migration_engine)

    left = reflect_full_inventory(create_all_engine)
    right = reflect_full_inventory(migration_engine)
    diff = diff_full_inventory(left, right)
    assert not diff, diff
    assert len(left["checks"]) == len(EXPECTED_CHECK_CONSTRAINT_NAMES)
    assert len(right["checks"]) == len(EXPECTED_CHECK_CONSTRAINT_NAMES)


def test_reflected_names_match_the_model_declaration(tmp_path) -> None:
    expected = _model_declared_names()

    create_all_engine = sa.create_engine(f"sqlite:///{tmp_path / 'names_createall.db'}")
    migration_engine = sa.create_engine(f"sqlite:///{tmp_path / 'names_migration.db'}")
    _build_create_all_schema(create_all_engine)
    _build_migration_schema(migration_engine)

    for label, engine in (
        ("create_all", create_all_engine),
        ("migration", migration_engine),
    ):
        inv = reflect_full_inventory(engine)
        assert set(inv["checks"]) == expected["checks"], label
        assert set(inv["unique"]) == expected["unique"], label
        assert set(inv["foreign_keys"]) == expected["foreign_keys"], label
        assert set(inv["indexes"]["nonunique"]) == expected["indexes"], label


def test_downgrade_leaves_no_residue(tmp_path) -> None:
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'residue.db'}")

    reset_checkpoint_anchor_fk_create_rule()
    other_tables = [
        table for name, table in Base.metadata.tables.items() if name != TABLE
    ]
    Base.metadata.create_all(engine, tables=other_tables)
    tables_before = set(sa.inspect(engine).get_table_names())
    assert TABLE not in tables_before

    _build_migration_schema(engine)
    tables_with = set(sa.inspect(engine).get_table_names())
    assert tables_with == tables_before | {TABLE}

    _downgrade(engine)
    tables_after = set(sa.inspect(engine).get_table_names())
    assert tables_after == tables_before


def test_differ_flags_a_changed_check_expression() -> None:
    """Unit test of diff_full_inventory alone -- no reflection, no database.
    Constructs two synthetic inventories differing only in one CHECK's
    sqltext and asserts the comparison flags it: a name-only comparison
    would let ``active_slot = 1`` silently weaken to ``active_slot >= 1``
    straight through, with the name set identical between the two
    inventories either way. The end-to-end equivalent, comparing real
    reflected schemas, lives in test_sqlite_migration_schema_matches_create_all.
    """
    base = {
        "checks": {"ck_x": "active_slot IS NULL OR active_slot = 1"},
        "unique": {},
        "foreign_keys": {},
        "primary_key": {},
        "indexes": {"unique": {}, "nonunique": {}},
        "columns": {},
    }
    weakened = {
        **base,
        "checks": {"ck_x": "active_slot IS NULL OR active_slot >= 1"},
    }

    diff = diff_full_inventory(base, weakened)
    assert "ck_x" in diff["checks"]["changed"]
    assert diff["checks"]["changed"]["ck_x"] == (
        "active_slot IS NULL OR active_slot = 1",
        "active_slot IS NULL OR active_slot >= 1",
    )
    # Confirms a name-only comparison would have missed this: the name set
    # is identical between the two synthetic inventories.
    assert set(base["checks"]) == set(weakened["checks"])
