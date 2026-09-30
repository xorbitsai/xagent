"""The cleanup-obligation table is created without any task table present."""

import importlib

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

from tests.web.services.task_database_shared import engine as engine_fixture
from xagent.web.models.database import Base
from xagent.web.models.task_cleanup_obligation import TaskCleanupObligation

engine = engine_fixture

MIGRATION = "xagent.migrations.versions.20260925_task_cleanup_obligations"


def _sqlite_master_sql(connection, table: str) -> str:
    return connection.execute(
        sa.text("SELECT sql FROM sqlite_master WHERE type='table' AND name=:table"),
        {"table": table},
    ).scalar_one()


def test_migration_matches_the_model_and_is_idempotent(engine):
    migration = importlib.import_module(MIGRATION)
    with (
        engine.begin() as connection,
        Operations.context(MigrationContext.configure(connection)),
    ):
        # No ``tasks`` table: the obligations reference their task by value
        # only, so an Alembic-only database gets the table regardless.
        migration.upgrade()
        migration.upgrade()
        inspector = sa.inspect(connection)
        reflected = {
            column["name"] for column in inspector.get_columns(migration.TABLE)
        }
        assert reflected == set(TaskCleanupObligation.__table__.columns.keys())
        assert inspector.get_foreign_keys(migration.TABLE) == []
        # No uniqueness: a reused task id must be able to owe twice.
        assert inspector.get_unique_constraints(migration.TABLE) == []
        indexed = {
            tuple(index["column_names"])
            for index in inspector.get_indexes(migration.TABLE)
        }
        assert {("task_id",), ("status", "next_attempt_at")} <= indexed

        if connection.dialect.name == "sqlite":
            # Outcomes are fenced on (id, attempts): ids must never be reused.
            assert "AUTOINCREMENT" in _sqlite_master_sql(connection, migration.TABLE)

        migration.downgrade()
        migration.downgrade()
        assert not sa.inspect(connection).has_table(migration.TABLE)


def test_model_metadata_also_requests_sqlite_autoincrement(engine):
    """``Base.metadata.create_all`` is the path tests (and ``create_all``-based
    setups) use instead of Alembic; it must carry the same guard."""
    Base.metadata.create_all(engine)
    if engine.dialect.name != "sqlite":
        return
    with engine.connect() as connection:
        assert "AUTOINCREMENT" in _sqlite_master_sql(
            connection, TaskCleanupObligation.__tablename__
        )
