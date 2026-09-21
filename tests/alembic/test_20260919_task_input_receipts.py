"""Receipt migrations preserve accepted identities after task deletion."""

import importlib

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

from tests.web.services.task_database_shared import engine as engine_fixture

engine = engine_fixture


def test_receipt_migration_preserves_tombstones_and_is_idempotent(engine):
    migration = importlib.import_module(
        "xagent.migrations.versions.20260919_task_input_receipts"
    )
    metadata = sa.MetaData()
    tasks = sa.Table("tasks", metadata, sa.Column("id", sa.Integer, primary_key=True))
    commands = sa.Table(
        "task_execution_commands",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("task_id", sa.Integer, sa.ForeignKey("tasks.id", ondelete="CASCADE")),
    )
    metadata.create_all(engine)
    with (
        engine.begin() as connection,
        Operations.context(MigrationContext.configure(connection)),
    ):
        connection.execute(tasks.insert(), {"id": 1})
        connection.execute(commands.insert(), {"id": 2, "task_id": 1})
        migration.upgrade()
        receipts = sa.Table(
            "task_input_receipts", sa.MetaData(), autoload_with=connection
        )
        connection.execute(
            receipts.insert(),
            dict(
                identity_hash="a" * 64,
                payload_hash="b" * 64,
                task_id=1,
                command_db_id=2,
            ),
        )
        migration.upgrade()
        assert (
            connection.execute(
                sa.select(sa.func.count()).select_from(receipts)
            ).scalar_one()
            == 1
        )
        connection.execute(tasks.delete())
        row = connection.execute(receipts.select()).mappings().one()
        assert row["identity_hash"] == "a" * 64
        assert row["task_id"] is None
        assert row["command_db_id"] is None
        assert row["created_at"] is not None
        migration.downgrade()
        migration.downgrade()
        assert not sa.inspect(connection).has_table("task_input_receipts")
        migration.upgrade()
        assert sa.inspect(connection).has_table("task_input_receipts")


def test_alembic_only_missing_parent_tables_are_left_to_metadata(engine):
    migration = importlib.import_module(
        "xagent.migrations.versions.20260919_task_input_receipts"
    )
    with (
        engine.begin() as connection,
        Operations.context(MigrationContext.configure(connection)),
    ):
        migration.upgrade()
        assert not sa.inspect(connection).has_table("task_input_receipts")
        migration.downgrade()
