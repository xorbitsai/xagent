"""Existing deployments gain the durable channel outbox on upgrade."""

import importlib

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations


def test_delivery_upgrade_and_downgrade():
    migration = importlib.import_module(
        "xagent.migrations.versions.20260915_channel_delivery"
    )
    engine = sa.create_engine("sqlite:///:memory:")
    metadata = sa.MetaData()
    for name in ("task_execution_commands", "user_channels"):
        sa.Table(name, metadata, sa.Column("id", sa.Integer, primary_key=True))
    metadata.create_all(engine)
    with engine.begin() as connection:
        with Operations.context(MigrationContext.configure(connection)):
            migration.upgrade()
            migration.upgrade()
            inspector = sa.inspect(connection)
            assert {
                column["name"]
                for column in inspector.get_columns("task_channel_deliveries")
            } == {
                "command_id",
                "channel_id",
                "destination",
                "status",
                "claim_token",
                "available_at",
                "delivered_at",
            }
            assert len(inspector.get_foreign_keys("task_channel_deliveries")) == 2
            assert (
                inspector.get_indexes("task_channel_deliveries")[0]["name"]
                == "ix_task_channel_delivery_pending"
            )
            migration.downgrade()
            assert not sa.inspect(connection).has_table("task_channel_deliveries")
