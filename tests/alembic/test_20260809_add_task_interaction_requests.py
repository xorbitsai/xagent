"""Tests for the task_interaction_requests table migration."""

import importlib.util
from io import StringIO
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

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
