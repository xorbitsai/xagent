"""Tests for the task_interaction_requests table migration."""

from io import StringIO
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import text

from tests.shared.postgres_disposable import (
    disposable_database_factory,
    load_migration_module,
)
from tests.web.services.checkpoint_anchor_shared import (
    reset_checkpoint_anchor_fk_create_rule,
)
from tests.web.services.task_interaction_schema_shared import (
    EXPECTED_CHECK_CONSTRAINT_NAMES,
)
from xagent.web.models.database import Base

# Imported for its side effect: registering task_interaction_requests on
# Base.metadata, so Base.metadata.create_all() below builds it too.
from xagent.web.models.task_interaction import TaskInteractionRequest  # noqa: F401

MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20260809_add_task_interaction_requests.py"
)
REVISION = "20260809_add_task_interaction_requests"
DOWN_REVISION = "20260808_add_task_lease_attempt_id"
TABLE = "task_interaction_requests"
PARENT_TABLES = ("tasks", "trace_events", "users")

# Local alias: used throughout this file, kept short rather than spelling out
# EXPECTED_CHECK_CONSTRAINT_NAMES (task_interaction_schema_shared.py's own
# name, shared with the model/create_all parity suite) at every call site.
CHECK_NAMES = EXPECTED_CHECK_CONSTRAINT_NAMES
FK_NAMES = {
    "fk_task_interaction_requests_task_id",
    "fk_task_interaction_requests_resume_trace_event_id",
    "fk_task_interaction_requests_responder_user_id",
}
INDEX_NAMES = {
    "ix_task_interaction_requests_task_status",
    "ix_task_interaction_requests_resume_trace_event_id",
}


def _load_migration_module():
    return load_migration_module(
        MIGRATION_PATH, "add_task_interaction_requests_migration"
    )


def _offline_sql(migration, dialect_name: str, operation: str) -> str:
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name=dialect_name,
        opts={"as_sql": True, "output_buffer": output},
    )

    with Operations.context(context):
        getattr(migration, operation)()

    return output.getvalue()


def _create_parent_tables(connection, tables: tuple[str, ...] = PARENT_TABLES) -> None:
    """Create the minimal parent tables the migration's foreign keys need.

    Each parent only needs the single `id INTEGER PRIMARY KEY` column the
    FKs reference; the real tasks/trace_events/users tables carry many more
    columns this migration never touches.
    """
    for table in tables:
        connection.exec_driver_sql(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY)")


@pytest.fixture
def postgresql_engine_factory():
    with disposable_database_factory("xagent_b2_alembic") as make:
        yield make


def _pg_index_names(connection) -> set[str]:
    """Every index name in the public schema, independent of which table
    (if any) still owns it. Used by the residue test below instead of
    Inspector.get_indexes(TABLE), which cannot be called once TABLE no
    longer exists after downgrade -- querying by name catches an index
    left behind under a different (or no) table just as well as one still
    attached to task_interaction_requests.
    """
    rows = connection.execute(
        text("SELECT indexname FROM pg_indexes WHERE schemaname = 'public'")
    )
    return {row[0] for row in rows}


def test_revision_metadata() -> None:
    migration = _load_migration_module()

    assert migration.revision == REVISION
    assert migration.down_revision == DOWN_REVISION


@pytest.mark.parametrize("dialect_name", ["postgresql", "sqlite"])
def test_offline_upgrade_renders_create_table(dialect_name: str) -> None:
    migration = _load_migration_module()

    sql = _offline_sql(migration, dialect_name, "upgrade")

    assert f"CREATE TABLE {TABLE}" in sql
    for name in CHECK_NAMES:
        assert f"CONSTRAINT {name} CHECK" in sql, f"missing CHECK {name!r}"
    for name in FK_NAMES:
        assert f"CONSTRAINT {name} FOREIGN KEY" in sql, f"missing FK {name!r}"
    assert "ON DELETE CASCADE" in sql
    assert sql.count("ON DELETE SET NULL") == 2
    for name in INDEX_NAMES:
        assert f"CREATE INDEX {name} ON {TABLE}" in sql, f"missing index {name!r}"
    assert "ALTER TABLE" not in sql


@pytest.mark.parametrize("dialect_name", ["postgresql", "sqlite"])
def test_offline_downgrade_renders_on_both_dialects(dialect_name: str) -> None:
    migration = _load_migration_module()

    sql = _offline_sql(migration, dialect_name, "downgrade")

    assert f"DROP TABLE {TABLE}" in sql


def test_online_upgrade_builds_the_table_and_downgrade_removes_it() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        _create_parent_tables(connection)
        context = MigrationContext.configure(connection)

        with Operations.context(context):
            migration.upgrade()

        inspector = sa.inspect(connection)
        assert TABLE in inspector.get_table_names()
        checks = {c["name"] for c in inspector.get_check_constraints(TABLE)}
        assert checks == CHECK_NAMES
        fks = {fk["name"] for fk in inspector.get_foreign_keys(TABLE)}
        assert fks == FK_NAMES
        indexes = {ix["name"] for ix in inspector.get_indexes(TABLE)}
        assert indexes == INDEX_NAMES

        with Operations.context(context):
            migration.downgrade()

        assert TABLE not in sa.inspect(connection).get_table_names()


def test_online_upgrade_is_idempotent() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        _create_parent_tables(connection)
        context = MigrationContext.configure(connection)

        with Operations.context(context):
            migration.upgrade()
            migration.upgrade()

        inspector = sa.inspect(connection)
        assert TABLE in inspector.get_table_names()
        checks = {c["name"] for c in inspector.get_check_constraints(TABLE)}
        assert checks == CHECK_NAMES


def test_online_downgrade_is_idempotent() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        _create_parent_tables(connection)
        context = MigrationContext.configure(connection)

        with Operations.context(context):
            migration.upgrade()
            migration.downgrade()
            migration.downgrade()

        assert TABLE not in sa.inspect(connection).get_table_names()


@pytest.mark.parametrize("missing_table", PARENT_TABLES)
def test_upgrade_skips_when_a_parent_table_is_missing(missing_table: str) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    present = tuple(t for t in PARENT_TABLES if t != missing_table)

    with engine.begin() as connection:
        _create_parent_tables(connection, present)
        context = MigrationContext.configure(connection)

        with Operations.context(context):
            migration.upgrade()

        assert TABLE not in sa.inspect(connection).get_table_names()


def test_upgrade_builds_the_table_when_every_parent_table_is_present() -> None:
    """Positive control for test_upgrade_skips_when_a_parent_table_is_missing:
    without this, a migration that always skipped would also pass the
    skip tests above."""
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        _create_parent_tables(connection)
        context = MigrationContext.configure(connection)

        with Operations.context(context):
            migration.upgrade()

        assert TABLE in sa.inspect(connection).get_table_names()


def test_upgrade_is_a_no_op_against_a_create_all_built_table() -> None:
    """Positive control for guard 1, built from the real create_all-first
    shape rather than from a second upgrade() call (test_online_upgrade_is_
    idempotent already covers repeated upgrade() calls): build the table via
    Base.metadata.create_all(), the actual path production startups use
    before any migration ever runs, then confirm upgrade() leaves it
    untouched instead of erroring on a duplicate table or drifting its
    constraint inventory."""
    reset_checkpoint_anchor_fk_create_rule()
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with engine.begin() as connection:
        inspector_before = sa.inspect(connection)
        checks_before = {
            c["name"] for c in inspector_before.get_check_constraints(TABLE)
        }
        columns_before = {c["name"] for c in inspector_before.get_columns(TABLE)}

        context = MigrationContext.configure(connection)
        with Operations.context(context):
            migration.upgrade()

        inspector_after = sa.inspect(connection)
        checks_after = {c["name"] for c in inspector_after.get_check_constraints(TABLE)}
        columns_after = {c["name"] for c in inspector_after.get_columns(TABLE)}

    assert checks_before == checks_after == CHECK_NAMES
    assert columns_before == columns_after


# ---------------------------------------------------------------------------
# PostgreSQL counterparts of the online upgrade/downgrade behavior above.
# SQLite-only coverage would leave the guard clauses in upgrade()/downgrade()
# -- both written to behave identically on either backend -- verified on only
# one of the two backends production actually runs. Each test below mints its
# own disposable database (never the database XAGENT_TEST_POSTGRES_URL itself
# names) via postgresql_engine_factory and drops it on teardown.
# ---------------------------------------------------------------------------


@pytest.mark.postgresql
def test_postgresql_online_upgrade_builds_the_table_and_downgrade_removes_it(
    postgresql_engine_factory,
) -> None:
    migration = _load_migration_module()
    engine = postgresql_engine_factory("build")

    with engine.begin() as connection:
        _create_parent_tables(connection)
        context = MigrationContext.configure(connection)

        with Operations.context(context):
            migration.upgrade()

        inspector = sa.inspect(connection)
        assert TABLE in inspector.get_table_names()
        checks = {c["name"] for c in inspector.get_check_constraints(TABLE)}
        assert checks == CHECK_NAMES
        fks = {fk["name"] for fk in inspector.get_foreign_keys(TABLE)}
        assert fks == FK_NAMES
        # Unlike SQLite, PostgreSQL's get_indexes() also reports the backing
        # index for each UNIQUE constraint (unique=True) -- see
        # task_interaction_schema_shared.py's reflect_full_inventory, which
        # documents the same asymmetry. Filtering to unique=False first is
        # what makes this comparable to INDEX_NAMES, the plain-index set.
        indexes = {
            ix["name"] for ix in inspector.get_indexes(TABLE) if not ix["unique"]
        }
        assert indexes == INDEX_NAMES

        with Operations.context(context):
            migration.downgrade()

        assert TABLE not in sa.inspect(connection).get_table_names()


@pytest.mark.postgresql
def test_postgresql_online_upgrade_is_idempotent(postgresql_engine_factory) -> None:
    migration = _load_migration_module()
    engine = postgresql_engine_factory("idempotent-upgrade")

    with engine.begin() as connection:
        _create_parent_tables(connection)
        context = MigrationContext.configure(connection)

        with Operations.context(context):
            migration.upgrade()
            migration.upgrade()

        inspector = sa.inspect(connection)
        assert TABLE in inspector.get_table_names()
        checks = {c["name"] for c in inspector.get_check_constraints(TABLE)}
        assert checks == CHECK_NAMES


@pytest.mark.postgresql
def test_postgresql_online_downgrade_is_idempotent(postgresql_engine_factory) -> None:
    migration = _load_migration_module()
    engine = postgresql_engine_factory("idempotent-downgrade")

    with engine.begin() as connection:
        _create_parent_tables(connection)
        context = MigrationContext.configure(connection)

        with Operations.context(context):
            migration.upgrade()
            migration.downgrade()
            migration.downgrade()

        assert TABLE not in sa.inspect(connection).get_table_names()


@pytest.mark.postgresql
def test_postgresql_downgrade_leaves_no_residue(postgresql_engine_factory) -> None:
    """PostgreSQL counterpart of test_downgrade_leaves_no_residue in
    test_task_interaction_requests_schema_parity.py (SQLite only, tables
    only). Table residue is not the whole story on this backend: the ``id``
    column is a single-column integer primary key with no ``autoincrement``
    override, so SQLAlchemy gives it PostgreSQL's implicit SERIAL/IDENTITY
    treatment -- measured directly against this migration on a disposable
    PostgreSQL database: the reflected default for ``id`` reads
    ``"nextval('task_interaction_requests_id_seq'::regclass)"``, i.e. a
    sequence get_table_names() never lists. A DROP TABLE owns and drops
    that sequence along with the table, and the two plain indexes plus the
    two UNIQUE/PK-backing indexes disappear the same way -- this test
    checks all three (tables, sequences, indexes) by name instead of
    assuming the cascade, since indexes are checked by a database-wide
    pg_indexes query rather than Inspector.get_indexes(TABLE), which
    cannot be called once TABLE no longer exists after downgrade.
    """
    migration = _load_migration_module()
    engine = postgresql_engine_factory("residue")

    with engine.begin() as connection:
        _create_parent_tables(connection)

    inspector = sa.inspect(engine)
    tables_before = set(inspector.get_table_names())
    sequences_before = set(inspector.get_sequence_names())
    with engine.connect() as connection:
        index_names_before = _pg_index_names(connection)
    assert TABLE not in tables_before

    with engine.begin() as connection:
        context = MigrationContext.configure(connection)
        with Operations.context(context):
            migration.upgrade()

    inspector = sa.inspect(engine)
    tables_with = set(inspector.get_table_names())
    assert tables_with == tables_before | {TABLE}
    sequences_with = set(inspector.get_sequence_names())
    assert sequences_with == sequences_before | {f"{TABLE}_id_seq"}

    with engine.begin() as connection:
        context = MigrationContext.configure(connection)
        with Operations.context(context):
            migration.downgrade()

    inspector = sa.inspect(engine)
    tables_after = set(inspector.get_table_names())
    assert tables_after == tables_before
    sequences_after = set(inspector.get_sequence_names())
    assert sequences_after == sequences_before
    with engine.connect() as connection:
        index_names_after = _pg_index_names(connection)
    assert index_names_after == index_names_before


@pytest.mark.postgresql
def test_postgresql_upgrade_targets_the_search_path_schema_past_a_stale_shadow(
    postgresql_engine_factory,
) -> None:
    """Regression for the guard-1 shadowing failure this migration's guards
    used to have: a stale same-named table sitting earlier on search_path
    (public) than the live application schema (app) used to make
    ``TABLE in inspector.get_table_names()`` true against the *wrong*
    relation, so guard 1 treated the table as already built and returned
    without ever creating it in the schema the application actually reads
    from -- a silent no-op that still let the revision stamp as applied.

    Reflection and DDL must resolve the same schema this migration's guards
    inspect for the fix to hold: this test builds live parents only inside
    ``app``, leaves a decoy ``task_interaction_requests`` sitting in
    ``public`` (unrelated columns, so a schema mix-up is easy to tell apart
    from success), sets ``search_path`` to ``app, public`` -- naming the
    live schema first but not exclusively, matching how a real application
    role's search_path is configured -- and asserts upgrade() builds the
    real table inside ``app`` with its foreign keys bound to ``app``'s
    parents, while the ``public`` decoy is left untouched.
    """
    migration = _load_migration_module()
    engine = postgresql_engine_factory("shadow")

    with engine.begin() as connection:
        connection.exec_driver_sql("CREATE SCHEMA app")
        for table in PARENT_TABLES:
            connection.exec_driver_sql(
                f"CREATE TABLE app.{table} (id INTEGER PRIMARY KEY)"
            )
        # The decoy: same name as the migration's target, wrong schema, and
        # a shape the real table never has -- so an assertion that reads
        # this table by mistake fails loudly instead of coincidentally
        # matching.
        connection.exec_driver_sql(
            f"CREATE TABLE public.{TABLE} (decoy_marker INTEGER PRIMARY KEY)"
        )
        connection.exec_driver_sql("SET search_path TO app, public")

        context = MigrationContext.configure(connection)
        with Operations.context(context):
            migration.upgrade()

        inspector = sa.inspect(connection)
        assert TABLE in inspector.get_table_names(schema="app")

        fks = {
            fk["name"]: fk["referred_schema"]
            for fk in inspector.get_foreign_keys(TABLE, schema="app")
        }
        assert fks == {name: "app" for name in FK_NAMES}

        checks = {
            c["name"] for c in inspector.get_check_constraints(TABLE, schema="app")
        }
        assert checks == CHECK_NAMES

        # The public decoy must be untouched: still its original one-column
        # shape, never overwritten or extended by the migration.
        decoy_columns = {
            c["name"] for c in inspector.get_columns(TABLE, schema="public")
        }
        assert decoy_columns == {"decoy_marker"}
