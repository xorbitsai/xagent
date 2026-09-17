"""The shared fixture teardown must survive a populated tasks/trace_events
constraint cycle -- the state any test that runs a task to a checkpoint
leaves behind.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from tests.shared.db_teardown import drop_all_tables
from tests.web.services.checkpoint_anchor_shared import (
    reset_checkpoint_anchor_fk_create_rule,
)
from xagent.db.sqlite import apply_sqlite_concurrency_pragmas
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TraceEvent
from xagent.web.models.user import User


def _fk_enforcing_sqlite_engine(db_path):
    engine = create_engine(f"sqlite:///{db_path}")
    apply_sqlite_concurrency_pragmas(engine)
    reset_checkpoint_anchor_fk_create_rule()
    Base.metadata.create_all(bind=engine)
    return engine


def _seed_task_with_anchored_checkpoint(engine) -> None:
    session = sessionmaker(bind=engine)()
    try:
        user = User(username="teardown-user", password_hash="hash", is_admin=False)
        session.add(user)
        session.flush()
        task = Task(user_id=int(user.id), title="task")
        session.add(task)
        session.flush()
        event = TraceEvent(
            task_id=int(task.id),
            build_id=None,
            event_id="evt-1",
            event_type="system_update_general",
            timestamp=datetime.now(timezone.utc),
            data={},
        )
        session.add(event)
        session.flush()
        task.last_checkpoint_event_id = "evt-1"
        task.last_checkpoint_trace_event_id = int(event.id)
        session.commit()
    finally:
        session.close()


def test_drop_all_tables_survives_a_populated_checkpoint_anchor(tmp_path) -> None:
    engine = _fk_enforcing_sqlite_engine(tmp_path / "teardown.db")
    _seed_task_with_anchored_checkpoint(engine)

    try:
        drop_all_tables(engine)

        remaining = set(inspect(engine).get_table_names())
        assert not remaining & {"tasks", "trace_events"}
    finally:
        engine.dispose()


def test_plain_drop_all_fails_on_a_populated_checkpoint_anchor(tmp_path) -> None:
    """Why the helper exists: the unguarded call this repo's fixtures used
    before raises once the cycle is populated, no matter the drop order."""
    from sqlalchemy.exc import IntegrityError

    engine = _fk_enforcing_sqlite_engine(tmp_path / "teardown-unguarded.db")
    _seed_task_with_anchored_checkpoint(engine)

    try:
        with pytest.raises(IntegrityError, match="FOREIGN KEY constraint failed"):
            Base.metadata.drop_all(bind=engine)
    finally:
        drop_all_tables(engine)
        engine.dispose()
