"""Offline transition keeps business state and separates only retry timing."""

import importlib
from datetime import datetime, timezone

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

from tests.web.services.task_database_shared import engine as engine_fixture

engine = engine_fixture


def test_upgrade_preserves_processing_evidence_and_pending_delay(engine):
    migration = importlib.import_module(
        "xagent.migrations.versions.20260918_command_retry_available_at"
    )
    metadata = sa.MetaData()
    table = sa.Table(
        "task_execution_commands",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("status", sa.String),
        sa.Column("claim_expires_at", sa.DateTime(timezone=True)),
        sa.Column("result", sa.String),
        sa.Column("attempt_count", sa.Integer),
    )
    metadata.create_all(engine)
    when = datetime(2026, 9, 18, tzinfo=timezone.utc)
    with (
        engine.begin() as connection,
        Operations.context(MigrationContext.configure(connection)),
    ):
        connection.execute(
            table.insert(),
            [
                dict(
                    id=i,
                    status=status,
                    claim_expires_at=when,
                    result="evidence",
                    attempt_count=2,
                )
                for i, status in enumerate(
                    ["pending", "processing", "completed", "failed"], 1
                )
            ],
        )
        before = connection.execute(table.select().order_by(table.c.id)).all()
        migration.upgrade()
        migration.upgrade()
        assert connection.execute(table.select().order_by(table.c.id)).all() == before
        rows = connection.execute(
            sa.text(
                "SELECT retry_available_at FROM task_execution_commands ORDER BY id"
            )
        ).all()
        assert rows[0][0] is not None
        assert [row[0] for row in rows[1:]] == [None, None, None]
        migration.downgrade()
        assert connection.execute(table.select().order_by(table.c.id)).all() == before


@pytest.mark.parametrize("direction", ["upgrade", "downgrade"])
def test_migration_skips_missing_metadata_owned_table(engine, direction):
    migration = importlib.import_module(
        "xagent.migrations.versions.20260918_command_retry_available_at"
    )
    with (
        engine.begin() as connection,
        Operations.context(MigrationContext.configure(connection)),
    ):
        before = sa.inspect(connection).get_table_names()
        assert "task_execution_commands" not in before
        getattr(migration, direction)()
        assert sa.inspect(connection).get_table_names() == before
