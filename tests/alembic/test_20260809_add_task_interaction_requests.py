"""Tests for the task_interaction_requests table migration."""

import importlib.util
import os
import uuid
from io import StringIO
from pathlib import Path

import psycopg2
import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import text
from sqlalchemy.engine import make_url

MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20260809_add_task_interaction_requests.py"
)
REVISION = "20260809_add_task_interaction_requests"
DOWN_REVISION = "20260808_add_task_lease_attempt_id"
TABLE = "task_interaction_requests"
PARENT_TABLES = ("tasks", "trace_events", "users")

CHECK_NAMES = {
    "ck_task_interaction_requests_status",
    "ck_task_interaction_requests_kind",
    "ck_task_interaction_requests_origin",
    "ck_task_interaction_requests_resume_checkpoint_type",
    "ck_task_interaction_requests_resume_locator_format",
    "ck_task_interaction_requests_terminal_reason",
    "ck_task_interaction_requests_protocol_version_floor",
    "ck_task_interaction_requests_active_slot_value",
    "ck_task_interaction_requests_active_slot_pairs_status",
    "ck_task_interaction_requests_active_anchor",
    "ck_task_interaction_requests_active_protocol",
    "ck_task_interaction_requests_terminal_pairs_status",
    "ck_task_interaction_requests_terminated_at_pairs_status",
    "ck_task_interaction_requests_response_pairs_status",
    "ck_task_interaction_requests_responded_at_pairs_status",
    "ck_task_interaction_requests_responder_pairs_responded_at",
    "ck_task_interaction_requests_expiry_after_creation",
    "ck_task_interaction_requests_run_id_nonempty",
    "ck_task_interaction_requests_resume_event_id_nonempty",
    "ck_task_interaction_requests_resume_execution_id_nonempty",
    "ck_task_interaction_requests_resume_run_partition_nonempty",
    "ck_task_interaction_requests_request_idempotency_key_nonempty",
    "ck_task_interaction_requests_responder_identity_nonempty",
}
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
    spec = importlib.util.spec_from_file_location(
        "add_task_interaction_requests_migration", MIGRATION_PATH
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


def _create_parent_tables(connection, tables: tuple[str, ...] = PARENT_TABLES) -> None:
    """Create the minimal parent tables the migration's foreign keys need.

    Each parent only needs the single `id INTEGER PRIMARY KEY` column the
    FKs reference; the real tasks/trace_events/users tables carry many more
    columns this migration never touches.
    """
    for table in tables:
        connection.exec_driver_sql(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY)")


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

    Copied from test_task_interaction_requests_schema_parity.py's fixture
    of the same name (own module-local copy, not a shared import, following
    this repo's precedent of duplicating the PostgreSQL disposable-database
    pattern per test file rather than centralizing it -- see
    test_task_interaction_schema_postgresql.py's docstring, which cites the
    same pattern copied from test_task_status_storage_postgresql.py /
    test_runtime_key_transition_postgres.py). See that fixture's docstring
    for why the DBAPI connection is opened with an explicit ``creator=``
    callable instead of letting SQLAlchemy's psycopg2 dialect connect
    itself: measured against this fixture's target server, SQLAlchemy's own
    connect path intermittently fails to authenticate against a database
    created moments earlier, while a bare ``psycopg2.connect()`` with
    identical parameters always succeeds.
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
        dbname = f"xagent_b2_alembic_{tag}_{uuid.uuid4().hex[:10]}"
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
        for _dbname, engine in minted:
            engine.dispose()
        for dbname, _engine in minted:
            admin_conn = _admin_connection()
            try:
                admin_conn.cursor().execute(f'DROP DATABASE IF EXISTS "{dbname}"')
            finally:
                admin_conn.close()


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
        operations = _operations(connection)

        with Operations.context(operations.get_context()):
            migration.upgrade()

        inspector = sa.inspect(connection)
        assert TABLE in inspector.get_table_names()
        checks = {c["name"] for c in inspector.get_check_constraints(TABLE)}
        assert checks == CHECK_NAMES
        fks = {fk["name"] for fk in inspector.get_foreign_keys(TABLE)}
        assert fks == FK_NAMES
        indexes = {ix["name"] for ix in inspector.get_indexes(TABLE)}
        assert indexes == INDEX_NAMES

        with Operations.context(operations.get_context()):
            migration.downgrade()

        assert TABLE not in sa.inspect(connection).get_table_names()


def test_online_upgrade_is_idempotent() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        _create_parent_tables(connection)
        operations = _operations(connection)

        with Operations.context(operations.get_context()):
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
        operations = _operations(connection)

        with Operations.context(operations.get_context()):
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
        operations = _operations(connection)

        with Operations.context(operations.get_context()):
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
        operations = _operations(connection)

        with Operations.context(operations.get_context()):
            migration.upgrade()

        assert TABLE in sa.inspect(connection).get_table_names()


def test_upgrade_skips_when_the_table_already_exists() -> None:
    """Positive control for guard 1: create the table by another route
    first (mirroring a create_all-first startup), then confirm upgrade()
    is a no-op rather than erroring on a duplicate table."""
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        _create_parent_tables(connection)
        operations = _operations(connection)

        with Operations.context(operations.get_context()):
            migration.upgrade()
            checks_before = {
                c["name"] for c in sa.inspect(connection).get_check_constraints(TABLE)
            }
            migration.upgrade()

        checks_after = {
            c["name"] for c in sa.inspect(connection).get_check_constraints(TABLE)
        }
        assert checks_after == checks_before == CHECK_NAMES


# ---------------------------------------------------------------------------
# PostgreSQL counterparts of the online upgrade/downgrade behavior above.
# SQLite-only coverage would leave the guard clauses in upgrade()/downgrade()
# -- both written to behave identically on either backend -- verified on only
# one of the two backends production actually runs. Each test below mints its
# own disposable database (never the database XAGENT_TEST_POSTGRES_URL itself
# names) via postgresql_engine_factory and drops it on teardown.
# ---------------------------------------------------------------------------


def test_postgresql_online_upgrade_builds_the_table_and_downgrade_removes_it(
    postgresql_engine_factory,
) -> None:
    migration = _load_migration_module()
    engine = postgresql_engine_factory("build")

    with engine.begin() as connection:
        _create_parent_tables(connection)
        operations = _operations(connection)

        with Operations.context(operations.get_context()):
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

        with Operations.context(operations.get_context()):
            migration.downgrade()

        assert TABLE not in sa.inspect(connection).get_table_names()


def test_postgresql_online_upgrade_is_idempotent(postgresql_engine_factory) -> None:
    migration = _load_migration_module()
    engine = postgresql_engine_factory("idempotent-upgrade")

    with engine.begin() as connection:
        _create_parent_tables(connection)
        operations = _operations(connection)

        with Operations.context(operations.get_context()):
            migration.upgrade()
            migration.upgrade()

        inspector = sa.inspect(connection)
        assert TABLE in inspector.get_table_names()
        checks = {c["name"] for c in inspector.get_check_constraints(TABLE)}
        assert checks == CHECK_NAMES


def test_postgresql_online_downgrade_is_idempotent(postgresql_engine_factory) -> None:
    migration = _load_migration_module()
    engine = postgresql_engine_factory("idempotent-downgrade")

    with engine.begin() as connection:
        _create_parent_tables(connection)
        operations = _operations(connection)

        with Operations.context(operations.get_context()):
            migration.upgrade()
            migration.downgrade()
            migration.downgrade()

        assert TABLE not in sa.inspect(connection).get_table_names()


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
        operations = _operations(connection)
        with Operations.context(operations.get_context()):
            migration.upgrade()

    inspector = sa.inspect(engine)
    tables_with = set(inspector.get_table_names())
    assert tables_with == tables_before | {TABLE}
    sequences_with = set(inspector.get_sequence_names())
    assert sequences_with == sequences_before | {f"{TABLE}_id_seq"}

    with engine.begin() as connection:
        operations = _operations(connection)
        with Operations.context(operations.get_context()):
            migration.downgrade()

    inspector = sa.inspect(engine)
    tables_after = set(inspector.get_table_names())
    assert tables_after == tables_before
    sequences_after = set(inspector.get_sequence_names())
    assert sequences_after == sequences_before
    with engine.connect() as connection:
        index_names_after = _pg_index_names(connection)
    assert index_names_after == index_names_before
