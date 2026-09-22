"""The retry budget preserves existing destinations across upgrade/downgrade."""

import importlib

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

from tests.web.services.channel_delivery_shared import database_url  # noqa: F401


def test_retry_budget_upgrade_preserves_pending_rows(database_url):  # noqa: F811
    original = importlib.import_module(
        "xagent.migrations.versions.20260915_channel_delivery"
    )
    migration = importlib.import_module(
        "xagent.migrations.versions.20260915_channel_delivery_retry_budget"
    )
    engine = sa.create_engine(database_url)
    metadata = sa.MetaData()
    for name in ("task_execution_commands", "user_channels"):
        sa.Table(name, metadata, sa.Column("id", sa.Integer, primary_key=True))
    metadata.create_all(engine)
    try:
        with engine.begin() as connection:
            with Operations.context(MigrationContext.configure(connection)):
                original.upgrade()
                connection.execute(
                    sa.text("INSERT INTO task_execution_commands (id) VALUES (1)")
                )
                connection.execute(sa.text("INSERT INTO user_channels (id) VALUES (1)"))
                connection.execute(
                    sa.text(
                        "INSERT INTO task_channel_deliveries (command_id, channel_id, destination) VALUES (1, 1, '{}')"
                    )
                )
                migration.upgrade()
                migration.upgrade()
                assert connection.execute(
                    sa.text("SELECT status, failure_count FROM task_channel_deliveries")
                ).one() == ("pending", 0)
                connection.execute(
                    sa.text("UPDATE task_channel_deliveries SET failure_count = 3")
                )
                migration.downgrade()
                assert (
                    connection.execute(
                        sa.text("SELECT status FROM task_channel_deliveries")
                    ).scalar_one()
                    == "pending"
                )
                migration.upgrade()
                assert (
                    connection.execute(
                        sa.text("SELECT failure_count FROM task_channel_deliveries")
                    ).scalar_one()
                    == 0
                )
    finally:
        engine.dispose()
