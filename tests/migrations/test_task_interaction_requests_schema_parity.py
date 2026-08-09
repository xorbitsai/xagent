"""create_all/migration schema parity for task_interaction_requests.

The model (src/xagent/web/models/task_interaction.py) and the
20260809_add_task_interaction_requests migration are two independent
implementations of the same constraint inventory: create_all builds the
table straight from the model, the migration builds it from a hand-copied
CHECKS tuple. Nothing enforces they stay identical except this suite.

Migration-side construction (candidate "乙" from the PR-B2 task book's
STOP-rule adjudication, §3): create_all every table *except*
task_interaction_requests, then run only this migration's upgrade()
against the connection -- not the full alembic revision chain, and not
checkpoint_anchor_shared.build_upgraded_sqlite_engine(). That helper
builds an "upgraded, FK-less tasks table" shape for a completely
different migration (the checkpoint-anchor column) and never runs any
alembic revision at all; using it here would make the "migration" side of
every comparison below a plain create_all wearing a different name, which
would stay green regardless of what this migration's CHECKS or guards
said. reset_checkpoint_anchor_fk_create_rule() is still required before
every create_all() call in this file, independent of that adjudication --
see its docstring for the cross-dialect FK-caching bug it works around.

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

import importlib.util
import os
import uuid
from pathlib import Path

import psycopg2
import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy.engine import make_url

from tests.web.services.checkpoint_anchor_shared import (
    reset_checkpoint_anchor_fk_create_rule,
)
from tests.web.services.task_interaction_schema_shared import (
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
    spec = importlib.util.spec_from_file_location(
        "add_task_interaction_requests_migration_parity", MIGRATION_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _build_create_all_schema(engine: sa.engine.Engine) -> None:
    reset_checkpoint_anchor_fk_create_rule()
    Base.metadata.create_all(engine)


def _build_migration_schema(engine: sa.engine.Engine) -> None:
    """See the module docstring: candidate 乙. Every table except
    task_interaction_requests is create_all'd, then only this revision's
    upgrade() runs against the connection."""
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


def _psycopg2_kwargs(base_url: str, dbname: str | None = None) -> dict[str, object]:
    parsed = make_url(base_url)
    return {
        "host": parsed.host,
        "port": parsed.port,
        "user": parsed.username,
        "password": parsed.password,
        "dbname": dbname if dbname is not None else parsed.database,
    }


@pytest.fixture
def postgresql_engine_factory():
    """Yields a callable that mints a brand-new, disposable PostgreSQL
    database per call (via CREATE DATABASE against the server
    XAGENT_TEST_POSTGRES_URL names) and drops every database it created on
    teardown. Skips the whole test if the env var is unset.

    Every engine here is built with an explicit ``creator=`` -- a plain
    function calling ``psycopg2.connect()`` directly -- instead of letting
    SQLAlchemy's psycopg2 dialect open the DBAPI connection itself.
    Measured against this fixture's target server: SQLAlchemy's own connect
    path intermittently (and, in one run, for 80 straight seconds of
    retries) fails to authenticate against a database created moments
    earlier, with the server returning a password-authentication FATAL;
    a bare ``psycopg2.connect()`` with identical host/port/user/password/
    dbname against the same freshly created database always succeeds. The
    ``creator=`` callable sidesteps whatever SQLAlchemy-side difference
    triggers that, while every other Engine/Connection/Inspector behavior
    (used throughout this fixture's callers: create_all, alembic
    Operations, inspection) stays identical.
    """
    base_url = os.getenv("XAGENT_TEST_POSTGRES_URL")
    if not base_url:
        pytest.skip("XAGENT_TEST_POSTGRES_URL is not set")

    def _admin_connection():
        conn = psycopg2.connect(**_psycopg2_kwargs(base_url))
        conn.autocommit = True
        return conn

    minted: list[tuple[str, sa.engine.Engine]] = []

    def _make(tag: str) -> sa.engine.Engine:
        dbname = f"xagent_b2_parity_{tag}_{uuid.uuid4().hex[:10]}"
        admin_conn = _admin_connection()
        try:
            admin_conn.cursor().execute(f'CREATE DATABASE "{dbname}"')
        finally:
            admin_conn.close()

        connect_kwargs = _psycopg2_kwargs(base_url, dbname)

        def _connect(_kwargs: dict[str, object] = connect_kwargs):
            return psycopg2.connect(**_kwargs)

        engine = sa.create_engine("postgresql://", creator=_connect)
        minted.append((dbname, engine))
        return engine

    try:
        yield _make
    finally:
        for dbname, engine in minted:
            try:
                engine.dispose()
            except Exception as exc:  # noqa: BLE001
                # Never stop here: an engine that will not dispose must not
                # block the drops below, or every database after it leaks.
                print(f"could not dispose the engine for {dbname}: {exc!r}")

        undropped: list[str] = []
        if minted:
            try:
                admin_conn = _admin_connection()
            except Exception as exc:
                raise RuntimeError(
                    "cannot reach the PostgreSQL server to drop the "
                    "disposable databases this fixture created; drop them "
                    "by hand: " + ", ".join(name for name, _ in minted)
                ) from exc
            try:
                # One autocommit connection serves every drop: a failed
                # DROP DATABASE leaves it usable, so a database still in
                # use costs its own name, not the rest of the list.
                for dbname, _engine in minted:
                    try:
                        admin_conn.cursor().execute(
                            f'DROP DATABASE IF EXISTS "{dbname}"'
                        )
                    except Exception as exc:  # noqa: BLE001
                        undropped.append(f"{dbname} ({exc!r})")
            finally:
                admin_conn.close()

        if undropped:
            raise RuntimeError(
                "disposable databases left behind on the server; drop them "
                "by hand: " + "; ".join(undropped)
            )


def test_sqlite_migration_schema_matches_create_all(tmp_path) -> None:
    create_all_engine = sa.create_engine(f"sqlite:///{tmp_path / 'createall.db'}")
    migration_engine = sa.create_engine(f"sqlite:///{tmp_path / 'migration.db'}")

    _build_create_all_schema(create_all_engine)
    _build_migration_schema(migration_engine)

    left = reflect_full_inventory(create_all_engine)
    right = reflect_full_inventory(migration_engine)
    diff = diff_full_inventory(left, right)
    assert not diff, diff
    assert len(left["checks"]) == 23
    assert len(right["checks"]) == 23


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
    assert len(left["checks"]) == 23
    assert len(right["checks"]) == 23


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


def test_expression_dimension_is_actually_compared() -> None:
    """Meta-test: diff_full_inventory must catch a CHECK whose predicate
    changed while its name did not -- a name-only comparison would let
    ``active_slot = 1`` silently weaken to ``active_slot >= 1`` straight
    through. Constructs two synthetic inventories differing only in one
    CHECK's sqltext and asserts the comparison flags it."""
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
