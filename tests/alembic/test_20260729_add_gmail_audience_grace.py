"""Tests for the Gmail callback-audience grace migration."""

import importlib.util
import logging
from io import StringIO
from pathlib import Path

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

from xagent.web.models.gmail_watch import GmailWatchState

MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20260729_add_gmail_audience_grace.py"
)
REVISION = "20260729_add_gmail_audience_grace"
DOWN_REVISION = "20260726_add_task_telegram_user_id"
TABLE = "gmail_watch_states"
COLUMNS = {
    "previous_push_audience",
    "previous_push_audience_expires_at",
}


def _load_migration_module():
    spec = importlib.util.spec_from_file_location(
        "add_gmail_audience_grace_migration",
        MIGRATION_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection) -> Operations:
    return Operations(MigrationContext.configure(connection))


def _offline_sql(migration, operation: str) -> str:
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="postgresql",
        opts={"as_sql": True, "output_buffer": output},
    )
    with Operations.context(context):
        getattr(migration, operation)()
    return output.getvalue()


def _create_watch_table(connection, extra_columns: str = "") -> None:
    connection.execute(
        sa.text(
            f"CREATE TABLE gmail_watch_states (id INTEGER PRIMARY KEY{extra_columns})"
        )
    )


def test_revision_metadata_and_model_contract() -> None:
    migration = _load_migration_module()

    assert migration.revision == REVISION
    assert migration.down_revision == DOWN_REVISION
    assert COLUMNS <= set(GmailWatchState.__table__.columns.keys())


def test_online_upgrade_and_downgrade_are_idempotent() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        _create_watch_table(connection)
        operations = _operations(connection)
        with Operations.context(operations.get_context()):
            migration.upgrade()
            migration.upgrade()

        assert COLUMNS <= {
            column["name"] for column in sa.inspect(connection).get_columns(TABLE)
        }

        with Operations.context(operations.get_context()):
            migration.downgrade()
            migration.downgrade()
        assert COLUMNS.isdisjoint(
            column["name"] for column in sa.inspect(connection).get_columns(TABLE)
        )


def test_online_upgrade_adds_only_missing_columns() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        _create_watch_table(
            connection,
            ", previous_push_audience TEXT",
        )
        operations = _operations(connection)
        with Operations.context(operations.get_context()):
            migration.upgrade()

        assert COLUMNS <= {
            column["name"] for column in sa.inspect(connection).get_columns(TABLE)
        }


def test_online_upgrade_warns_and_skips_a_missing_watch_table(caplog) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        operations = _operations(connection)
        with Operations.context(operations.get_context()):
            with caplog.at_level(logging.WARNING):
                migration.upgrade()

        assert "Required table 'gmail_watch_states' is missing" in caplog.text
        assert TABLE not in sa.inspect(connection).get_table_names()


def test_offline_upgrade_and_downgrade_emit_both_columns() -> None:
    migration = _load_migration_module()

    upgrade_sql = _offline_sql(migration, "upgrade")
    downgrade_sql = _offline_sql(migration, "downgrade")

    for column in COLUMNS:
        assert column in upgrade_sql
        assert column in downgrade_sql
