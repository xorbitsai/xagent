"""The chat message metadata column migrates both ways and is idempotent."""

import importlib

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

from xagent.web.models.chat_message import TaskChatMessage

MIGRATION = "xagent.migrations.versions.20261117_task_chat_message_metadata"


def _columns(connection) -> set:
    return {
        column["name"]
        for column in sa.inspect(connection).get_columns(TaskChatMessage.__tablename__)
    }


def test_message_metadata_migrates_both_ways_and_is_idempotent(tmp_path):
    migration = importlib.import_module(MIGRATION)
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'm.db'}")
    metadata = sa.MetaData()
    sa.Table(
        TaskChatMessage.__tablename__,
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("content", sa.Text),
    )
    metadata.create_all(engine)
    with (
        engine.begin() as connection,
        Operations.context(MigrationContext.configure(connection)),
    ):
        migration.upgrade()
        migration.upgrade()
        assert "message_metadata" in _columns(connection)
        migration.downgrade()
        migration.downgrade()
        assert "message_metadata" not in _columns(connection)
