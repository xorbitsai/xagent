"""The auto-recovery tables are created, matched to the models and dropped."""

import importlib

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

from tests.web.services.task_database_shared import engine as engine_fixture
from xagent.web.models.task_auto_recovery import TaskAutoRecovery, TaskRecoveryEvent

engine = engine_fixture

MIGRATION = "xagent.migrations.versions.20261008_task_auto_recovery"
TABLES = (TaskAutoRecovery, TaskRecoveryEvent)


def _tasks_table(engine):
    metadata = sa.MetaData()
    sa.Table("tasks", metadata, sa.Column("id", sa.Integer, primary_key=True))
    metadata.create_all(engine)


def test_migration_revision_chain():
    migration = importlib.import_module(MIGRATION)
    assert migration.revision == "20261008_task_auto_recovery"
    assert migration.down_revision == "20261005_uploaded_file_cleanup_manifest"


def test_migration_matches_the_models_and_is_idempotent(engine):
    migration = importlib.import_module(MIGRATION)
    _tasks_table(engine)
    with (
        engine.begin() as connection,
        Operations.context(MigrationContext.configure(connection)),
    ):
        migration.upgrade()
        migration.upgrade()
        inspector = sa.inspect(connection)
        for model in TABLES:
            table = model.__table__
            reflected = {
                column["name"]: column for column in inspector.get_columns(table.name)
            }
            assert set(reflected) == set(table.columns.keys())
            for column in table.columns:
                assert reflected[column.name]["nullable"] == column.nullable, (
                    table.name,
                    column.name,
                )
                # Compiled form compares class and length: String(24) vs
                # String(32) must not pass.
                assert reflected[column.name]["type"].compile(
                    dialect=connection.dialect
                ) == column.type.compile(dialect=connection.dialect), (
                    table.name,
                    column.name,
                )
            assert [
                (fk["referred_table"], fk["constrained_columns"], fk["options"])
                for fk in inspector.get_foreign_keys(table.name)
            ] == [("tasks", ["task_id"], {"ondelete": "CASCADE"})]

        assert inspector.get_pk_constraint("task_auto_recovery")[
            "constrained_columns"
        ] == ["task_id"]
        due = [
            index
            for index in inspector.get_indexes("task_auto_recovery")
            if index["name"] == "ix_task_auto_recovery_due"
        ]
        assert [index["column_names"] for index in due] == [
            ["state", "next_attempt_at"]
        ]
        assert {
            index["name"] for index in inspector.get_indexes("task_auto_recovery")
        } == {"ix_task_auto_recovery_due"}
        assert {
            index["name"] for index in inspector.get_indexes("task_recovery_events")
        } == {"ix_task_recovery_events_task_id"}

        migration.downgrade()
        migration.downgrade()
        inspector = sa.inspect(connection)
        assert not inspector.has_table("task_auto_recovery")
        assert not inspector.has_table("task_recovery_events")
        # ``tasks`` is never touched.
        assert [c["name"] for c in inspector.get_columns("tasks")] == ["id"]


def test_server_defaults_apply_and_deleting_the_task_cascades(engine):
    migration = importlib.import_module(MIGRATION)
    _tasks_table(engine)
    with (
        engine.begin() as connection,
        Operations.context(MigrationContext.configure(connection)),
    ):
        migration.upgrade()
        connection.execute(sa.text("INSERT INTO tasks (id) VALUES (1)"))
        connection.execute(
            sa.text(
                "INSERT INTO task_auto_recovery (task_id, run_id, reason, state, "
                "paused_state_version, interrupted_at, episode_started_at) "
                "VALUES (1, 'r', 'lease_expired', 'scheduled', 3, "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO task_recovery_events (task_id, run_id, event) "
                "VALUES (1, 'r', 'interrupted')"
            )
        )
        row = (
            connection.execute(sa.text("SELECT * FROM task_auto_recovery"))
            .mappings()
            .one()
        )
        assert row["no_progress_resumes"] == 0
        assert row["total_resumes"] == 0
        assert row["updated_at"] is not None
        connection.execute(sa.text("DELETE FROM tasks WHERE id = 1"))
        for table in ("task_auto_recovery", "task_recovery_events"):
            count = connection.execute(sa.text(f"SELECT COUNT(*) FROM {table}"))
            assert count.scalar_one() == 0


def test_alembic_only_missing_tasks_table_is_left_to_metadata(engine):
    migration = importlib.import_module(MIGRATION)
    with (
        engine.begin() as connection,
        Operations.context(MigrationContext.configure(connection)),
    ):
        migration.upgrade()
        inspector = sa.inspect(connection)
        assert not inspector.has_table("task_auto_recovery")
        assert not inspector.has_table("task_recovery_events")
        migration.downgrade()
