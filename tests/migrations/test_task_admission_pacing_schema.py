"""Pacing migration preserves existing admitted tasks and budgets."""

from pathlib import Path

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

from tests.shared.postgres_disposable import load_migration_module
from tests.web.services.task_database_shared import engine as engine_fixture
from tests.web.services.task_database_shared import task_id as task_id_fixture
from xagent.web.models.task_admission import TaskAdmissionBucket
from xagent.web.models.task_admission_pacing import TaskAdmissionPacing

engine = engine_fixture
task_id = task_id_fixture


def test_pacing_upgrade_and_downgrade_preserve_existing_bucket(engine, task_id):
    migration = load_migration_module(
        Path(__file__).parents[2]
        / "src/xagent/migrations/versions/20260923_admission_pacing.py",
        "task_admission_pacing_migration",
    )
    TaskAdmissionPacing.__table__.drop(engine)
    with engine.begin() as connection:
        connection.execute(
            sa.insert(TaskAdmissionBucket).values(
                key="preserved", capacity=2, max_pending=10
            )
        )
        with Operations.context(MigrationContext.configure(connection)):
            migration.upgrade()
            migration.upgrade()
        actual = sa.inspect(connection).get_columns(TaskAdmissionPacing.__tablename__)
        assert {c["name"]: c["nullable"] for c in actual} == {
            c.name: c.nullable for c in TaskAdmissionPacing.__table__.columns
        }
        connection.execute(
            sa.insert(TaskAdmissionPacing).values(
                bucket_key="preserved", interval_seconds=1, burst=2, lane="batch"
            )
        )
        with Operations.context(MigrationContext.configure(connection)):
            migration.downgrade()
        assert (
            connection.scalar(
                sa.select(TaskAdmissionBucket.capacity).where(
                    TaskAdmissionBucket.key == "preserved"
                )
            )
            == 2
        )
        assert "task_admission_pacing" not in sa.inspect(connection).get_table_names()
