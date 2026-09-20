"""Shared SQLite/PostgreSQL engine and persisted task fixtures."""

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from tests.shared.postgres_disposable import disposable_database_factory
from xagent.web.models.database import Base
from xagent.web.models.task import Task
from xagent.web.models.user import User


@pytest.fixture(
    params=["sqlite", pytest.param("postgresql", marks=pytest.mark.postgresql)]
)
def engine(request, tmp_path):
    if request.param == "postgresql":
        with disposable_database_factory("tasks") as make:
            yield make("tasks")
    else:
        result = sa.create_engine(f"sqlite:///{tmp_path / 'tasks.db'}")

        @sa.event.listens_for(result, "connect")
        def enable_foreign_keys(connection, _record):
            connection.execute("PRAGMA foreign_keys=ON")

        try:
            yield result
        finally:
            result.dispose()


@pytest.fixture
def task_id(engine):
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        user = User(username="task-owner", password_hash="unused")
        db.add(user)
        db.flush()
        task = Task(user_id=user.id, title="Existing task", description="unchanged")
        db.add(task)
        db.commit()
        return task.id
