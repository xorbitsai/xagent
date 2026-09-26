"""The expired-task revision matches the models and is idempotent (#2565)."""

import importlib

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

from tests.web.services.task_database_shared import engine as engine_fixture
from xagent.web.models.database import Base
from xagent.web.models.expired_task import ExpiredTaskTombstone

engine = engine_fixture

MIGRATION = "xagent.migrations.versions.20260926_expired_task_tombstones"


def _drop_revision_objects(connection, migration) -> None:
    """Undo what ``create_all`` built for this revision, so upgrade has work."""
    migration.downgrade()
    inspector = sa.inspect(connection)
    assert not inspector.has_table(migration.TABLE)
    for table, column in migration.ADDED_COLUMNS:
        assert column not in {c["name"] for c in inspector.get_columns(table)}


def test_upgrade_matches_the_models_and_is_idempotent(engine):
    migration = importlib.import_module(MIGRATION)
    Base.metadata.create_all(engine)
    with (
        engine.begin() as connection,
        Operations.context(MigrationContext.configure(connection)),
    ):
        _drop_revision_objects(connection, migration)

        migration.upgrade()
        migration.upgrade()

        inspector = sa.inspect(connection)
        for table, column in migration.ADDED_COLUMNS:
            reflected = {c["name"]: c for c in inspector.get_columns(table)}
            assert reflected[column]["nullable"] is True
            # No server default: a default would stamp every existing row.
            assert reflected[column]["default"] is None

        assert {c["name"] for c in inspector.get_columns(migration.TABLE)} == set(
            ExpiredTaskTombstone.__table__.columns.keys()
        )
        foreign_keys = {
            fk["constrained_columns"][0]: (
                fk["referred_table"],
                (fk.get("options") or {}).get("ondelete", "").upper(),
            )
            for fk in inspector.get_foreign_keys(migration.TABLE)
        }
        assert foreign_keys == {
            "user_id": ("users", "CASCADE"),
            "agent_id": ("agents", "SET NULL"),
            "workforce_id": ("workforces", "SET NULL"),
        }
        indexed = {
            tuple(index["column_names"])
            for index in inspector.get_indexes(migration.TABLE)
        }
        assert {("user_id",), ("agent_id",), ("workforce_id",)} <= indexed
        # The same index names ``create_all`` produces from the model.
        model_indexes = {index.name for index in ExpiredTaskTombstone.__table__.indexes}
        assert model_indexes == {
            index["name"] for index in inspector.get_indexes(migration.TABLE)
        }

        migration.downgrade()
        migration.downgrade()
        inspector = sa.inspect(connection)
        assert not inspector.has_table(migration.TABLE)
        for table, column in migration.ADDED_COLUMNS:
            assert column not in {c["name"] for c in inspector.get_columns(table)}


def test_the_table_waits_for_its_referenced_tables(engine):
    """A database without ``workforces`` yet gets the columns but no table."""
    migration = importlib.import_module(MIGRATION)
    with (
        engine.begin() as connection,
        Operations.context(MigrationContext.configure(connection)),
    ):
        connection.execute(sa.text("CREATE TABLE users (id INTEGER PRIMARY KEY)"))
        connection.execute(sa.text("CREATE TABLE agents (id INTEGER PRIMARY KEY)"))
        connection.execute(sa.text("CREATE TABLE tasks (id INTEGER PRIMARY KEY)"))

        migration.upgrade()

        inspector = sa.inspect(connection)
        assert not inspector.has_table(migration.TABLE)
        assert "traces_expired_at" in {
            c["name"] for c in inspector.get_columns("tasks")
        }
